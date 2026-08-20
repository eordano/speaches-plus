#![cfg(feature = "wgpu")]

use nv_kernels::wgpu_backend::compose;
use nv_kernels::wgpu_backend::dispatch;
use nv_kernels::wgpu_backend::kernels as kn;
use nv_kernels::wgpu_backend::kernels::gemv_nvfp4::lin;
use nv_kernels::wgpu_backend::kernels::{flash_decode as fd, quant_gemv};

fn to_msl(tag: &str, source: &str) -> String {
    let module = naga::front::wgsl::parse_str(source)
        .unwrap_or_else(|e| panic!("{tag}: wgsl parse: {}", e.message()));
    let info = naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::all(),
    )
    .validate(&module)
    .unwrap_or_else(|e| panic!("{tag}: validate: {e}"));
    let opts = naga::back::msl::Options {
        lang_version: (3, 0),
        ..Default::default()
    };
    naga::back::msl::write_string(
        &module,
        &info,
        &opts,
        &naga::back::msl::PipelineOptions::default(),
    )
    .unwrap_or_else(|e| panic!("{tag}: msl-out: {e}"))
    .0
}

struct Entry {
    src: String,
    name: String,
    thread_arrays: Vec<String>,
    workgroup_vars: Vec<String>,
}

fn wgsl_name(msl_name: &str) -> String {
    msl_name.strip_suffix('_').unwrap_or(msl_name).to_string()
}

fn entries(tag: &str, msl: &str) -> Vec<Entry> {
    let mut out = Vec::new();
    let mut i = 0usize;
    while let Some(at) = msl[i..].find("kernel void ") {
        let start = i + at;
        let rest = &msl[start..];
        let name_end = rest.find('(').expect("kernel signature");
        let name = rest["kernel void ".len()..name_end].trim().to_string();
        let end = rest.find("\n}\n").unwrap_or(rest.len());
        let mut thread_arrays = Vec::new();
        let mut workgroup_vars = Vec::new();
        for l in rest[..end].lines() {
            let t = l.trim().trim_start_matches(", ");
            if let Some((head, tail)) = t.split_once(' ') {
                if head.starts_with("type_") && tail.ends_with(" = {};") {
                    thread_arrays.push(tail.trim_end_matches(" = {};").to_string());
                }
            }
            if let Some(decl) = t.strip_prefix("threadgroup ") {
                if let Some(v) = decl.rsplit(' ').next() {
                    workgroup_vars.push(v.trim_end_matches(',').to_string());
                }
            }
        }
        out.push(Entry {
            src: tag.to_string(),
            name: wgsl_name(&name),
            thread_arrays,
            workgroup_vars,
        });
        i = start + name_end;
    }
    out
}

fn all_entries() -> Vec<Entry> {
    let mut list: Vec<(&str, String)> = vec![
        ("assistant_drafter", compose(kn::assistant_drafter::WGSL)),
        (
            "attention_fp8_decode",
            compose(kn::attention_fp8_decode::WGSL),
        ),
        ("attn_decode", compose(kn::attn_decode::WGSL)),
        (
            "attn_decode_small_m",
            compose(kn::attn_decode_small_m::WGSL),
        ),
        (
            "attn_decode_small_m_fp8",
            compose(kn::attn_decode_small_m_fp8::WGSL),
        ),
        (
            "depthwise_conv1d_silu_bf16",
            compose(kn::depthwise_conv1d_silu_bf16::WGSL),
        ),
        ("flash_decode", compose(kn::flash_decode::WGSL)),
        ("fused_attn_chain", compose(kn::fused_attn_chain::WGSL)),
        ("fused_norm_chain", compose(kn::fused_norm_chain::WGSL)),
        ("gather_rows_bf16", compose(kn::gather_rows_bf16::WGSL)),
        ("gdn_gating", compose(kn::gdn_gating::WGSL)),
        ("gdn_recurrent", compose(kn::gdn_recurrent::WGSL)),
        ("gelu_tanh_mul", compose(kn::gelu_tanh_mul::WGSL)),
        ("gemv_bf16", compose(kn::gemv_bf16::WGSL)),
        ("gemv_nvfp4", compose(kn::gemv_nvfp4::WGSL)),
        ("gemv_nvfp4_v2", compose(kn::gemv_nvfp4_v2::WGSL)),
        ("gemv_w4a16", compose(kn::gemv_w4a16::WGSL)),
        (
            "gemv_w4a16_m1_proto",
            compose(kn::gemv_w4a16_m1_proto::WGSL),
        ),
        ("graph_decode", compose(kn::graph_decode::WGSL)),
        ("kv_fp8", compose(kn::kv_fp8::WGSL)),
        ("kv_fp8_paged", compose(kn::kv_fp8_paged::WGSL)),
        ("moe_permute", compose(kn::moe_permute::WGSL)),
        (
            "moe_unpermute_scatter",
            compose(kn::moe_unpermute_scatter::WGSL),
        ),
        ("quant_gemv", compose(kn::quant_gemv::WGSL)),
        (
            "quantize_nvfp4_bf16",
            compose(kn::quantize_nvfp4_bf16::WGSL),
        ),
        ("residual_scale", compose(kn::residual_scale::WGSL)),
        ("rmsnorm", compose(kn::rmsnorm::WGSL)),
        ("rmsnorm_residual", compose(kn::rmsnorm_residual::WGSL)),
        ("rope", compose(kn::rope::WGSL)),
        ("rope_bf16", compose(kn::rope_bf16::WGSL)),
        ("sampler", compose(kn::sampler::WGSL)),
        ("silu", compose(kn::silu::WGSL)),
        ("tree_verify_attn", compose(kn::tree_verify_attn::WGSL)),
        (
            "tree_verify_fp8",
            compose(&format!(
                "{}\n{}",
                kn::kv_fp8::WGSL,
                kn::tree_verify_fp8::WGSL
            )),
        ),
        ("verify_fused_norms", compose(kn::verify_fused_norms::WGSL)),
        ("gemm_nvfp4_scalar", kn::gemm_nvfp4::scalar_source()),
        ("moe_grouped_scalar", kn::moe_grouped_gemm::scalar_source()),
        (
            "gemm_bf16_small_m",
            kn::gemm_bf16_small_m::small_m_source(8, true),
        ),
        (
            "gemm_w4a16_small_m",
            kn::gemm_w4a16_small_m::small_m_source(),
        ),
        ("gemv_bf16_sg", kn::gemv_bf16::sg_source()),
        ("gemv_bf16_sg_pk", kn::gemv_bf16::sg_pk_source()),
        (
            "gemv_bf16_adaptive",
            kn::gemv_bf16::adaptive_source(16, 256),
        ),
        ("gemv_nvfp4_gemv", kn::gemv_nvfp4::gemv_source()),
        ("gemv_nvfp4_sg", kn::gemv_nvfp4::sg_gemv_source()),
        ("gemv_nvfp4_lin", lin::source()),
        (
            "gemv_nvfp4_sgw_deep",
            kn::gemv_nvfp4::sgw_source(kn::gemv_nvfp4::SGW_DEEP),
        ),
        (
            "quantize_nvfp4_act_grid",
            kn::quantize_nvfp4_bf16::act_grid_source(),
        ),
        ("gemv_w4a16_sg_pk", kn::gemv_w4a16::sg_pk_source()),
        ("gemv_w4a16_sg", kn::gemv_w4a16::sg_source(16, 256)),
        ("gemv_w4a16_sg_mk_arr", kn::gemv_w4a16::sg_mk_source(8)),
        (
            "gemv_w4a16_sg_mk_unrolled",
            kn::gemv_w4a16::sg_mk_unrolled_source(8),
        ),
        ("gemv_w4a16_sg_pk_mr", kn::gemv_w4a16::sg_pk_mr_source(4)),
    ];
    list.sort_by(|a, b| a.0.cmp(b.0));
    list.iter()
        .flat_map(|(tag, src)| entries(tag, &to_msl(tag, src)))
        .collect()
}

const SPILLS: &[(&str, &str, usize)] = &[
    ("attn_decode_small_m", "attn_decode_small_m_bf16kv", 3),
    ("attn_decode_small_m", "attn_decode_small_m_f32", 3),
    ("flash_decode", "flash_decode_f32", 1),
    ("flash_decode", "flash_smv2_stage1_bf16kv", 5),
    ("flash_decode", "flash_smv2_stage1_f32", 5),
    ("flash_decode", "flash_smv2_stage1_fp8kv", 5),
    ("flash_decode", "flash_splitk_stage1_bf16kv_mk", 3),
    ("gemv_nvfp4_v2", "gemv_nvfp4_mrow_pk", 5),
    ("flash_decode", "flash_splitk_stage1_fp8kv_mk", 3),
    ("gdn_recurrent", "gdn_recurrent_f32", 1),
    ("gemm_bf16_small_m", "gemm_bf16_small_m_vec8_8", 1),
    ("gemm_w4a16_small_m", "gemm_w4a16_small_m_m1", 1),
    ("gemm_w4a16_small_m", "gemm_w4a16_small_m_m2", 1),
    ("gemm_w4a16_small_m", "gemm_w4a16_small_m_m3", 1),
    ("gemm_w4a16_small_m", "gemm_w4a16_small_m_m4", 1),
    ("gemm_w4a16_small_m", "gemm_w4a16_small_m_m5", 1),
    ("gemm_w4a16_small_m", "gemm_w4a16_small_m_m6", 1),
    ("gemm_w4a16_small_m", "gemm_w4a16_small_m_m7", 1),
    ("gemm_w4a16_small_m", "gemm_w4a16_small_m_m8", 1),
    ("gemm_w4a16_small_m", "gemm_w4a16_small_m_m9", 1),
    ("gemv_bf16_adaptive", "gemv_bf16_sg_v4_adaptive", 2),
    ("gemv_nvfp4", "gemv_nvfp4_bf16_sg", 1),
    ("gemv_nvfp4", "gemv_nvfp4_bf16_sgq", 1),
    ("gemv_nvfp4", "gemv_nvfp4_bf16_sgw", 1),
    ("gemv_nvfp4", "quantize_row_nvfp4_bf16", 1),
    ("gemv_nvfp4_sg", "gemv_nvfp4_bf16_sg", 1),
    ("gemv_nvfp4_sgw_deep", "gemv_nvfp4_bf16_sg", 1),
    ("gemv_nvfp4_sgw_deep", "gemv_nvfp4_bf16_sgq", 1),
    ("gemv_nvfp4_sgw_deep", "gemv_nvfp4_bf16_sgw", 1),
    ("gemv_nvfp4_v2", "gemv_nvfp4_fmlut", 4),
    ("gemv_nvfp4_v2", "gemv_nvfp4_fmlut_pk", 5),
    ("gemv_nvfp4_v2", "gemv_nvfp4_fmrow", 4),
    ("gemv_nvfp4_v2", "gemv_nvfp4_mrow", 4),
    ("gemv_w4a16_sg_mk_arr", "gemv_w4a16_sg_mk_pk", 1),
    ("gemv_w4a16_sg_mk_arr", "gemv_w4a16_sg_mk_pk3", 1),
    ("gemv_w4a16_sg_pk_mr", "gemv_w4a16_sg_pkm", 4),
    ("gemv_w4a16_sg_pk_mr", "gemv_w4a16_sg_pkm3", 4),
    ("quantize_nvfp4_act_grid", "quantize_row_nvfp4_bf16", 1),
    ("tree_verify_attn", "tree_verify_attn_bf16", 1),
    ("tree_verify_fp8", "tree_verify_attn_fp8", 1),
];

#[test]
fn function_scope_thread_arrays_are_exactly_the_pinned_spill_list() {
    let all = all_entries();
    assert!(all.len() > 150, "census found only {} entries", all.len());
    let mut got: Vec<(String, String, usize)> = all
        .iter()
        .filter(|e| !e.thread_arrays.is_empty())
        .map(|e| (e.src.clone(), e.name.clone(), e.thread_arrays.len()))
        .collect();
    got.sort();
    let want: Vec<(String, String, usize)> = SPILLS
        .iter()
        .map(|(s, e, n)| (s.to_string(), e.to_string(), *n))
        .collect();
    for (s, e, n) in &got {
        eprintln!("msl-spill {s:<26} {e:<34} {n} thread array(s)");
    }
    eprintln!(
        "msl-spill total {} entries spill of {} entries in {} sources",
        got.len(),
        all.len(),
        all.iter()
            .map(|e| e.src.as_str())
            .collect::<std::collections::BTreeSet<_>>()
            .len()
    );
    let new: Vec<_> = got.iter().filter(|g| !want.contains(g)).collect();
    let gone: Vec<_> = want.iter().filter(|w| !got.contains(w)).collect();
    assert!(
        new.is_empty() && gone.is_empty(),
        "spill census moved.\n  newly spilling: {new:?}\n  no longer spilling (delete from SPILLS): {gone:?}"
    );
}

fn thread_arrays_of(tag: &str, msl: &str, entry: &str) -> Vec<String> {
    entries(tag, msl)
        .into_iter()
        .find(|e| e.name == entry)
        .unwrap_or_else(|| {
            panic!("{tag}: no kernel `{entry}` (or `{entry}_`) in the generated MSL")
        })
        .thread_arrays
}

#[test]
fn routed_entries_exist_by_name_and_do_not_spill() {
    let flash = to_msl("flash_decode", &compose(fd::WGSL));
    let qg = to_msl("quant_gemv", &quant_gemv::source());
    let cases: [(&str, &str, &[&str]); 2] = [
        (
            "flash_decode",
            flash.as_str(),
            &[
                fd::ENTRY_STAGE1_BF16,
                fd::ENTRY_STAGE1_FP8,
                fd::ENTRY_STAGE1_BF16_MK_U,
                fd::ENTRY_STAGE1_FP8_MK_U,
                fd::ENTRY_STAGE2,
                fd::ENTRY_STAGE2_MK,
                fd::ENTRY_STAGE2_U,
                fd::ENTRY_STAGE2_MK_U,
            ],
        ),
        (
            "quant_gemv",
            qg.as_str(),
            &[
                quant_gemv::INT8_GROUP_GELU_ENTRY,
                quant_gemv::INT8_GROUP_GELU_SG_ENTRY,
                quant_gemv::FP8_GROUP_GELU_ENTRY,
                quant_gemv::FP8_GROUP_GELU_SG_ENTRY,
                quant_gemv::INT8_GROUP_ENTRY,
                quant_gemv::INT8_GROUP_SG_ENTRY,
                quant_gemv::FP8_GROUP_ENTRY,
                quant_gemv::FP8_GROUP_SG_ENTRY,
            ],
        ),
    ];
    for (tag, msl, list) in cases {
        for entry in list {
            let found = thread_arrays_of(tag, msl, entry);
            eprintln!("routed {tag}::{entry}: thread arrays {found:?}");
            assert!(
                found.is_empty(),
                "{tag}::{entry} is routed in production and spilled to a thread-address-space \
                 array: {found:?}"
            );
        }
    }
}

#[test]
fn dot4i8packed_and_the_subgroup_butterfly_lower_the_way_the_alu_verdict_assumed() {
    let msl = to_msl("gemv_nvfp4_lin", &lin::source());
    if let Ok(p) = std::env::var("NV_MSL_DUMP_DIR") {
        let p = format!("{p}/gemv_nvfp4_lin.metal");
        std::fs::write(&p, &msl).expect("write msl");
        eprintln!("wrote {} bytes of MSL to {p}", msl.len());
    }
    assert!(
        msl.contains("simd_shuffle_xor"),
        "the subgroup butterfly no longer lowers to simd_shuffle_xor; the lin/v3 ladder timings \
         assumed a native shuffle"
    );

    let at = msl
        .find("float v3_dot8_(")
        .expect("no v3_dot8_ in the generated MSL");
    let rest = &msl[at..];
    let body = &rest[..rest.find("\n}").unwrap_or(rest.len())];
    let scalar_mults = body.matches("] * ").count();
    eprintln!(
        "v3_dot8_ MSL: {} chars, packed_char4 casts={}, scalar char mults={scalar_mults}",
        body.len(),
        body.matches("as_type<packed_char4>").count()
    );
    assert!(
        scalar_mults >= 8,
        "dot4I8Packed lowered to something other than scalar char mults ({scalar_mults}); the \
         ALU ladder verdict is stale, re-measure it"
    );
    assert!(
        !body.contains("simd") && !body.contains("dot("),
        "dot4I8Packed now has a vector/native lowering inside v3_dot8_; the 2026-08-09 ALU \
         verdict is stale"
    );
}

#[test]
fn no_nozi_audited_entry_in_this_crate_is_trivially_safe() {
    let all = all_entries();
    let audited = dispatch::nozi_audited_entries();
    let mut visible = 0usize;
    let mut trivially_safe = Vec::new();
    let mut missing: Vec<&str> = Vec::new();
    for e in audited {
        if let Some(got) = all.iter().find(|g| g.name == *e) {
            visible += 1;
            if got.workgroup_vars.is_empty() {
                trivially_safe.push(format!("{}::{}", got.src, got.name));
            }
        } else {
            missing.push(e);
        }
    }
    eprintln!(
        "nozi-census {} audited entries; {visible} reachable from nv-kernels, {} of those declare NO workgroup memory",
        audited.len(),
        trivially_safe.len()
    );
    eprintln!("nozi-census not reachable from nv-kernels (the graph crates must run this same check on their generated sources): {missing:?}");
    assert!(
        trivially_safe.is_empty(),
        "these audited entries declare no workgroup memory and can be discharged mechanically: {trivially_safe:?}"
    );
}
