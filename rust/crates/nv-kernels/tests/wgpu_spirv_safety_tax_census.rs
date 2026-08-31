#![cfg(feature = "wgpu")]

use std::collections::HashMap;

use naga::back::spv;
use naga::proc::{BoundsCheckPolicies, BoundsCheckPolicy};
use naga::valid::{Capabilities, ValidationFlags, Validator};
use nv_kernels::wgpu_backend::compose;

const OP_NAME: u16 = 5;

const OP_EXT_INST: u16 = 12;

const OP_FUNCTION: u16 = 54;

const OP_FUNCTION_END: u16 = 56;

const OP_VARIABLE: u16 = 59;

const OP_LOOP_MERGE: u16 = 246;

const SPIRV_HEADER_WORDS: usize = 5;

const E4B_AND_G4_W4_DECODE_GEMV: &str = include_str!("../wgsl/gemv_w4a16.wgsl");

const E4B_DECODE_GEMV_PACKED_TAIL: &str = include_str!("../wgsl/e4b_gemv_w4_pk.wgsl");

const GOW_DECODE_MX_GEMV: &str = include_str!("../wgsl/gow_mx.wgsl");

const G4MOE_DECODE_EXPERT_GEMV: &str = include_str!("../wgsl/g4m_gemv_w4e_sglat_xv4.wgsl");

const E4B_DECODE_LMHEAD_I8: &str = include_str!("../wgsl/e4b_lmhead_i8.wgsl");

const WGPU30_VULKAN_HAL_ALWAYS_RESTRICTS_INDEX_ALWAYS_FORCES_LOOP_BOUNDING_ALWAYS_GUARDS_INT_DIV:
    &str = "wgpu-hal 30 vulkan adapter.rs builds spv::Options with index: Restrict \
            unconditionally, buffer: Unchecked only when VK_EXT_robustness2 robustBufferAccess2 \
            is present, force_loop_bounding: true unconditionally and emit_int_div_checks left at \
            its true default; the ONLY relaxation surface is \
            Device::create_shader_module_trusted(ShaderRuntimeChecks), which nv-kernels reaches \
            through NV_WGPU_UNCHECKED_SHADERS";

#[derive(Clone, Copy)]
struct Arm {
    index: BoundsCheckPolicy,
    buffer: BoundsCheckPolicy,
    force_loop_bounding: bool,
    emit_int_div_checks: bool,
}

const HAL_DEFAULT_ON_A_ROBUSTNESS2_ADAPTER: Arm = Arm {
    index: BoundsCheckPolicy::Restrict,
    buffer: BoundsCheckPolicy::Unchecked,
    force_loop_bounding: true,
    emit_int_div_checks: true,
};

const HAL_DEFAULT_WITHOUT_ROBUSTNESS2: Arm = Arm {
    buffer: BoundsCheckPolicy::Restrict,
    ..HAL_DEFAULT_ON_A_ROBUSTNESS2_ADAPTER
};

const HAL_DEFAULT_MINUS_LOOP_BOUNDING: Arm = Arm {
    force_loop_bounding: false,
    ..HAL_DEFAULT_ON_A_ROBUSTNESS2_ADAPTER
};

const HAL_DEFAULT_MINUS_INT_DIV_GUARDS: Arm = Arm {
    emit_int_div_checks: false,
    ..HAL_DEFAULT_ON_A_ROBUSTNESS2_ADAPTER
};

const TRUSTED_SHADER_RUNTIME_CHECKS_UNCHECKED: Arm = Arm {
    index: BoundsCheckPolicy::Unchecked,
    buffer: BoundsCheckPolicy::Unchecked,
    force_loop_bounding: false,
    emit_int_div_checks: false,
};

fn spirv_words(source: &str, arm: Arm, debug_names: bool) -> Vec<u32> {
    let module = naga::front::wgsl::parse_str(source)
        .unwrap_or_else(|e| panic!("wgsl parse: {}", e.emit_to_string(source)));
    let info = Validator::new(ValidationFlags::all(), Capabilities::all())
        .validate(&module)
        .unwrap_or_else(|e| panic!("wgsl validate: {e:?}"));
    let mut options = spv::Options {
        lang_version: (1, 6),
        bounds_check_policies: BoundsCheckPolicies {
            index: arm.index,
            buffer: arm.buffer,
            image_load: BoundsCheckPolicy::Unchecked,
            binding_array: BoundsCheckPolicy::Unchecked,
        },
        force_loop_bounding: arm.force_loop_bounding,
        emit_int_div_checks: arm.emit_int_div_checks,
        ..Default::default()
    };
    options.capabilities = None;
    if debug_names {
        options.flags |= spv::WriterFlags::DEBUG;
    } else {
        options.flags.remove(spv::WriterFlags::DEBUG);
    }
    spv::write_vec(&module, &info, &options, None).expect("spv write")
}

fn walk(words: &[u32], mut f: impl FnMut(u16, &[u32])) {
    let mut i = SPIRV_HEADER_WORDS;
    while i < words.len() {
        let len = (words[i] >> 16) as usize;
        assert!(len > 0, "zero-length SPIR-V instruction at word {i}");
        f((words[i] & 0xffff) as u16, &words[i..i + len]);
        i += len;
    }
}

fn instruction_count(words: &[u32]) -> usize {
    let mut n = 0;
    walk(words, |_, _| n += 1);
    n
}

fn op_count(words: &[u32], op: u16) -> usize {
    let mut n = 0;
    walk(words, |o, _| {
        if o == op {
            n += 1
        }
    });
    n
}

fn decode_literal_string(words: &[u32]) -> String {
    let mut out = Vec::new();
    'outer: for w in words {
        for b in w.to_le_bytes() {
            if b == 0 {
                break 'outer;
            }
            out.push(b);
        }
    }
    String::from_utf8_lossy(&out).into_owned()
}

struct FnBody {
    instructions: usize,
    loops: usize,
}

fn per_function_bodies(words: &[u32]) -> HashMap<String, FnBody> {
    let mut names: HashMap<u32, String> = HashMap::new();
    walk(words, |op, ins| {
        if op == OP_NAME && ins.len() >= 3 {
            names.insert(ins[1], decode_literal_string(&ins[2..]));
        }
    });
    let mut out: HashMap<String, FnBody> = HashMap::new();
    let mut cur: Option<(String, FnBody)> = None;
    walk(words, |op, ins| {
        if op == OP_FUNCTION && ins.len() >= 3 {
            let id = ins[2];
            let name = names
                .get(&id)
                .cloned()
                .unwrap_or_else(|| format!("fn%{id}"));
            cur = Some((
                name,
                FnBody {
                    instructions: 0,
                    loops: 0,
                },
            ));
            return;
        }
        if op == OP_FUNCTION_END {
            if let Some((name, body)) = cur.take() {
                out.insert(name, body);
            }
            return;
        }
        if let Some((_, body)) = cur.as_mut() {
            body.instructions += 1;
            if op == OP_LOOP_MERGE {
                body.loops += 1;
            }
        }
    });
    out
}

struct Source {
    name: &'static str,
    wgsl: String,
    hot_helpers: &'static [&'static str],
}

fn sources() -> Vec<Source> {
    vec![
        Source {
            name: "e4b w4a16 decode gemv (gemv_w4a16.wgsl + e4b_gemv_w4_pk.wgsl)",
            wgsl: compose(&format!(
                "{E4B_AND_G4_W4_DECODE_GEMV}\n{E4B_DECODE_GEMV_PACKED_TAIL}"
            )),
            hot_helpers: &[
                "w4a16_row_acc_v4",
                "w4a16_dot32_v4",
                "w4a16_dot8_v4",
                "w4a16_lane_reduce",
            ],
        },
        Source {
            name: "gow mxfp4 decode gemv (gow_mx.wgsl)",
            wgsl: compose(GOW_DECODE_MX_GEMV),
            hot_helpers: &[
                "gow_gemv_mx_sg",
                "gow_mx_dot_block",
                "gow_mx_dot8",
                "gow_mx_sg_butterfly32",
                "gow_mx_dot_word",
                "nvfp4_decode",
            ],
        },
        Source {
            name: "g4moe w4 expert decode gemv (g4m_gemv_w4e_sglat_xv4.wgsl)",
            wgsl: compose(G4MOE_DECODE_EXPERT_GEMV),
            hot_helpers: &["g4w_group_dot_xv4_one_vec4_per_mat_group_same_fma_order"],
        },
        Source {
            name: "e4b int8 lm_head (e4b_lmhead_i8.wgsl)",
            wgsl: compose(E4B_DECODE_LMHEAD_I8),
            hot_helpers: &[],
        },
    ]
}

#[test]
fn wgpu30_vulkan_default_shader_options_inject_a_measurable_safety_tax_into_every_decode_gemv() {
    eprintln!(
        "{WGPU30_VULKAN_HAL_ALWAYS_RESTRICTS_INDEX_ALWAYS_FORCES_LOOP_BOUNDING_ALWAYS_GUARDS_INT_DIV}"
    );
    eprintln!(
        "SPIRV-CENSUS arms: hal = index Restrict + buffer Unchecked + force_loop_bounding + \
         int_div guards; no_r2 = hal with buffer Restrict (what a non-robustness2 adapter pays); \
         no_flb = hal minus loop bounding; no_idiv = hal minus int-div guards; \
         trusted = ShaderRuntimeChecks::unchecked()"
    );

    let mut any_loop_bound_injection = false;
    for src in sources() {
        let hal = spirv_words(&src.wgsl, HAL_DEFAULT_ON_A_ROBUSTNESS2_ADAPTER, false);
        let no_r2 = spirv_words(&src.wgsl, HAL_DEFAULT_WITHOUT_ROBUSTNESS2, false);
        let no_flb = spirv_words(&src.wgsl, HAL_DEFAULT_MINUS_LOOP_BOUNDING, false);
        let no_idiv = spirv_words(&src.wgsl, HAL_DEFAULT_MINUS_INT_DIV_GUARDS, false);
        let trusted = spirv_words(&src.wgsl, TRUSTED_SHADER_RUNTIME_CHECKS_UNCHECKED, false);

        let (hi, ni, fi, di, ti) = (
            instruction_count(&hal),
            instruction_count(&no_r2),
            instruction_count(&no_flb),
            instruction_count(&no_idiv),
            instruction_count(&trusted),
        );
        let loops = op_count(&hal, OP_LOOP_MERGE);

        eprintln!(
            "SPIRV-CENSUS src=\"{}\" loops={loops} instr_hal={hi} instr_no_r2={ni} \
             instr_no_flb={fi} instr_no_idiv={di} instr_trusted={ti} \
             buffer_check_cost_we_do_NOT_pay={} loop_bound_cost={} per_loop={:.1} \
             int_div_guard_cost={} index_clamp_cost={} fn_vars_hal={} fn_vars_trusted={} \
             extinst_hal={} extinst_trusted={} hal_over_trusted={:.3}x",
            src.name,
            ni.saturating_sub(hi),
            hi.saturating_sub(fi),
            if loops == 0 {
                0.0
            } else {
                hi.saturating_sub(fi) as f64 / loops as f64
            },
            hi.saturating_sub(di),
            hi.saturating_sub(ti)
                .saturating_sub(hi.saturating_sub(fi))
                .saturating_sub(hi.saturating_sub(di)),
            op_count(&hal, OP_VARIABLE),
            op_count(&trusted, OP_VARIABLE),
            op_count(&hal, OP_EXT_INST),
            op_count(&trusted, OP_EXT_INST),
            hi as f64 / ti as f64
        );

        let hal_named = per_function_bodies(&spirv_words(
            &src.wgsl,
            HAL_DEFAULT_ON_A_ROBUSTNESS2_ADAPTER,
            true,
        ));
        let trusted_named = per_function_bodies(&spirv_words(
            &src.wgsl,
            TRUSTED_SHADER_RUNTIME_CHECKS_UNCHECKED,
            true,
        ));
        for helper in src.hot_helpers {
            match (hal_named.get(*helper), trusted_named.get(*helper)) {
                (Some(h), Some(t)) => eprintln!(
                    "  SPIRV-FN {helper}: body_instr_hal={} body_instr_trusted={} \
                     injected={} loops={} tax_share_of_body={:.1}%",
                    h.instructions,
                    t.instructions,
                    h.instructions.saturating_sub(t.instructions),
                    h.loops,
                    100.0 * h.instructions.saturating_sub(t.instructions) as f64
                        / h.instructions as f64
                ),
                _ => eprintln!(
                    "  SPIRV-FN {helper}: ABSENT from the emitted module -- naga inlined it or \
                     the name changed; a missing helper is a finding, not a pass"
                ),
            }
        }

        assert!(
            loops > 0,
            "{}: a decode gemv with no loop at all would make the loop-bounding column vacuous",
            src.name
        );
        assert!(
            hi >= fi && hi >= di && fi >= ti && di >= ti,
            "{}: relaxing a policy must never grow the module ({hi} {fi} {di} {ti})",
            src.name
        );
        if hi > fi {
            any_loop_bound_injection = true;
        }
    }

    assert!(
        any_loop_bound_injection,
        "naga 30 force_loop_bounding must inject a per-loop 64-bit counter; if this fires the \
         census instrument is wrong, not the compiler"
    );
}

#[test]
fn naga_force_loop_bounding_injects_a_64_bit_counter_at_the_head_of_every_loop_body() {
    let src = compose(GOW_DECODE_MX_GEMV);
    let module = naga::front::wgsl::parse_str(&src)
        .unwrap_or_else(|e| panic!("wgsl parse: {}", e.emit_to_string(&src)));
    let info = Validator::new(ValidationFlags::all(), Capabilities::all())
        .validate(&module)
        .expect("validate");
    let mut with = naga::back::msl::Options {
        lang_version: (2, 4),
        force_loop_bounding: true,
        ..Default::default()
    };
    with.bounds_check_policies = BoundsCheckPolicies {
        index: BoundsCheckPolicy::Restrict,
        buffer: BoundsCheckPolicy::Unchecked,
        image_load: BoundsCheckPolicy::Unchecked,
        binding_array: BoundsCheckPolicy::Unchecked,
    };
    let without = naga::back::msl::Options {
        force_loop_bounding: false,
        ..with.clone()
    };
    let pipe = naga::back::msl::PipelineOptions::default();

    let render = |o: &naga::back::msl::Options| match naga::back::msl::write_string(
        &module, &info, o, &pipe,
    ) {
        Ok((s, _)) => s,
        Err(e) => panic!("msl write: {e:?}"),
    };
    let on = render(&with);
    let off = render(&without);

    let excerpt = |s: &str, needle: &str, lines: usize| -> String {
        match s.find(needle) {
            Some(at) => s[at..].lines().take(lines).collect::<Vec<_>>().join("\n"),
            None => format!("<needle {needle} not found>"),
        }
    };
    eprintln!(
        "MSL-EVIDENCE the same naga transform rendered as text (the Vulkan path emits the SPIR-V \
         equivalent counted by the census); gow_gemv_mx_sg k-block loop:\n\
         --- force_loop_bounding = true (what wgpu ships) ---\n{}\n\
         --- force_loop_bounding = false (ShaderRuntimeChecks::unchecked) ---\n{}",
        excerpt(&on, "while(true)", 14),
        excerpt(&off, "while(true)", 10)
    );

    let loop_bound_decls = on.matches("loop_bound").count();
    assert!(
        loop_bound_decls > 0 && !off.contains("loop_bound"),
        "the loop_bound counter must be present under force_loop_bounding and absent without it \
         (present={loop_bound_decls}, absent_side_contains={})",
        off.contains("loop_bound")
    );
    eprintln!("MSL-EVIDENCE loop_bound mentions: with={loop_bound_decls} without=0");
}
