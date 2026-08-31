#![cfg(feature = "wgpu")]

mod common;
use common::bf16_lin_gow_bias as bf16_lin;
use common::mx_stack;
use common::tiny_config_gpt_oss;
use common::LcgSplitMix64TwoSided as Lcg;

use nv_models::gpt_oss_wgpu as gow;
use nv_models::gpt_oss_wgpu::{GptOssConfig, GptOssLayerType};

const MAX_SEQ_384_IS_ENOUGH_ROUNDS_FOR_THE_RUNNING_MAX_TO_STOP_MOVING: usize = 384;

const GOW_SHIPPED_HEAD_DIM_64_FOLD_8_IS_THE_GEOMETRY_EVERY_MEASURED_SPAN_WAS_TAKEN_AT: (u32, u32) =
    (64, 8);

static ARM_ENVS_ARE_PROCESS_GLOBAL_SO_TESTS_MUTATING_THEM_MUST_NOT_INTERLEAVE: std::sync::Mutex<()> =
    std::sync::Mutex::new(());

fn hd32_kv8_config() -> GptOssConfig {
    let cfg = GptOssConfig {
        head_dim: 32,
        max_position_embeddings: 4096,
        ..tiny_config_gpt_oss()
    };
    assert!(
        cfg.layer_types.contains(&GptOssLayerType::Full),
        "the accumulator-file rewrite rides the kv-fp8 full-attention stage1; a config without a \
         full layer exercises nothing"
    );
    cfg
}

fn norm_vec(r: &mut Lcg, n: usize) -> Vec<u16> {
    (0..n)
        .map(|_| half::bf16::from_f32(1.0 + r.next_f32() * 0.1).to_bits())
        .collect()
}

fn tiny_weights(cfg: &GptOssConfig, seed: u64) -> gow::HostWeights {
    let mut r = Lcg::new(seed);
    let h = cfg.hidden_size;
    let hd = cfg.head_dim;
    let layers = (0..cfg.num_hidden_layers)
        .map(|_| gow::HostLayer {
            input_ln: norm_vec(&mut r, h),
            post_attn_ln: norm_vec(&mut r, h),
            attn: gow::HostAttn {
                q: bf16_lin(&mut r, cfg.num_attention_heads * hd, h, 0.2, true),
                k: bf16_lin(&mut r, cfg.num_key_value_heads * hd, h, 0.2, true),
                v: bf16_lin(&mut r, cfg.num_key_value_heads * hd, h, 0.2, true),
                o: bf16_lin(&mut r, h, cfg.num_attention_heads * hd, 0.2, true),
                sinks: (0..cfg.num_attention_heads)
                    .map(|_| 1.0 + r.next_f32() * 0.5)
                    .collect(),
            },
            moe: gow::HostMoe {
                router: bf16_lin(&mut r, cfg.num_local_experts, h, 0.2, true),
                gate_up: mx_stack(
                    &mut r,
                    cfg.num_local_experts,
                    2 * cfg.intermediate_size,
                    h,
                    0.2,
                ),
                down: mx_stack(&mut r, cfg.num_local_experts, h, cfg.intermediate_size, 0.2),
            },
        })
        .collect();
    gow::HostWeights {
        embed: r.bf16_vec(cfg.vocab_size * h, 0.6),
        final_norm: norm_vec(&mut r, h),
        lm_head: r.bf16_vec(cfg.vocab_size * h, 0.2),
        layers,
    }
}

struct EnvGuard {
    _serialized: std::sync::MutexGuard<'static, ()>,
    keys: Vec<&'static str>,
}

fn env_on(pairs: &[(&'static str, &str)]) -> EnvGuard {
    let g = ARM_ENVS_ARE_PROCESS_GLOBAL_SO_TESTS_MUTATING_THEM_MUST_NOT_INTERLEAVE
        .lock()
        .unwrap_or_else(|p| p.into_inner());
    let mut keys = Vec::new();
    std::env::set_var(gow::KV_FP8_ENV, "1");
    keys.push(gow::KV_FP8_ENV);
    for (k, v) in pairs {
        std::env::set_var(k, v);
        keys.push(k);
    }
    EnvGuard {
        _serialized: g,
        keys,
    }
}

impl Drop for EnvGuard {
    fn drop(&mut self) {
        for k in &self.keys {
            std::env::remove_var(k);
        }
    }
}

fn logit_bits_over_steps(pairs: &[(&'static str, &str)], steps: usize) -> Option<Vec<Vec<u32>>> {
    let _g = env_on(pairs);
    let ctx = match nv_kernels::wgpu_backend::WgpuContext::shared() {
        Ok(ctx) => ctx,
        Err(e) => {
            eprintln!("[skip] no wgpu adapter: {e}");
            return None;
        }
    };
    eprintln!("[wgpu] adapter: {}", ctx.info.name);
    let cfg = hd32_kv8_config();
    let hw = tiny_weights(&cfg, 0x51ed_2701);
    let mut gpu = gow::GptOssWgpu::new(
        cfg,
        &hw,
        MAX_SEQ_384_IS_ENOUGH_ROUNDS_FOR_THE_RUNNING_MAX_TO_STOP_MOVING,
    )
    .expect("build model");
    let mut tok = 3u32;
    let mut out = Vec::new();
    for _ in 0..steps {
        let (t, logits) = gpu.decode_step_logits(tok).expect("decode step");
        out.push(logits.iter().map(|v| v.to_bits()).collect::<Vec<u32>>());
        tok = t;
    }
    Some(out)
}

#[test]
fn every_accumulator_file_stage1_rewrite_validates_under_naga_at_the_real_gptoss_geometry_without_a_gpu(
) {
    use nv_kernels::wgpu_backend::kernels::flash_decode as fd;
    let (hd, fold) = GOW_SHIPPED_HEAD_DIM_64_FOLD_8_IS_THE_GEOMETRY_EVERY_MEASURED_SPAN_WAS_TAKEN_AT;
    let base = format!("{}\n{}", fd::WGSL, fd::fold_stage1_source_sd(hd, true, fold));
    for qreg in [false, true] {
        for smfast in [false, true] {
          for wgacc in [false, true] {
            let entry = gow::flash_acc_stage1_entry_sd(hd, fold, qreg, smfast, wgacc);
            let body = gow::flash_acc_stage1_source_sd(hd, fold, qreg, smfast, wgacc);
            let src = nv_kernels::wgpu_backend::compose(&format!("{base}\n{body}"));
            let module = naga::front::wgsl::parse_str(&src)
                .unwrap_or_else(|e| panic!("{entry}: wgsl parse failed: {e}"));
            naga::valid::Validator::new(
                naga::valid::ValidationFlags::all(),
                naga::valid::Capabilities::all(),
            )
            .validate(&module)
            .unwrap_or_else(|e| panic!("{entry}: validation failed: {e:?}"));
            assert!(
                src.contains(&format!("fn {entry}(")),
                "{entry}: the generated source must define the entry the model dispatches, or \
                 the arm silently falls back to the stock kernel and measures nothing"
            );
          }
        }
    }
}

#[test]
fn the_kvline_stage1_validates_under_naga_and_its_workgroup_stages_every_query_head_inside_the_device_limit(
) {
    use nv_kernels::wgpu_backend::kernels::flash_decode as fd;
    let (hd, group) = GOW_SHIPPED_HEAD_DIM_64_FOLD_8_IS_THE_GEOMETRY_EVERY_MEASURED_SPAN_WAS_TAKEN_AT;
    let nkv = 8u32;
    let base = format!("{}\n{}", fd::WGSL, fd::fold_stage1_source_sd(hd, true, group));
    for slots in [1u32, 2, 4] {
        let entry = gow::flash_kvline_stage1_entry_sd(hd, nkv, group, slots);
        let body = gow::flash_kvline_stage1_source_sd(hd, nkv, group, slots);
        let src = nv_kernels::wgpu_backend::compose(&format!("{base}\n{body}"));
        let module = naga::front::wgsl::parse_str(&src)
            .unwrap_or_else(|e| panic!("{entry}: wgsl parse failed: {e}"));
        naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::all(),
        )
        .validate(&module)
        .unwrap_or_else(|e| panic!("{entry}: validation failed: {e:?}"));
        assert!(
            src.contains(&format!("fn {entry}(")),
            "{entry}: the generated source must define the entry the model dispatches"
        );
        let invocations = gow::flash_kvline_workgroup_invocations(nkv, slots);
        let storage = gow::flash_kvline_workgroup_storage_bytes(hd, nkv, group, slots);
        assert_eq!(
            invocations,
            32 * nkv * slots,
            "the kvline workgroup is exactly one warp per kv head per position slot; any other \
             size breaks the warp-to-kv-head map the arm exists to create"
        );
        assert!(
            storage >= 4 * nkv * group * hd,
            "the kvline workgroup stages every query head, not the folded subset a per-kv-head \
             workgroup stages; a smaller footprint means the staging loop is wrong"
        );
    }
}

#[test]
fn moving_the_accumulator_file_into_workgroup_memory_cannot_reach_the_shipped_resident_workgroup_count(
) {
    let (hd, fold) = GOW_SHIPPED_HEAD_DIM_64_FOLD_8_IS_THE_GEOMETRY_EVERY_MEASURED_SPAN_WAS_TAKEN_AT;
    let bytes = gow::flash_accumulators_in_workgroup_memory_bytes_per_workgroup(hd, fold);
    let limit = match nv_kernels::wgpu_backend::WgpuContext::shared() {
        Ok(ctx) => {
            eprintln!("[wgpu] adapter: {}", ctx.info.name);
            ctx.caps.max_compute_workgroup_storage_size
        }
        Err(e) => {
            eprintln!("[skip] no wgpu adapter: {e}");
            return;
        }
    };
    let resident_hundredths = 100 * (limit / bytes);
    eprintln!(
        "[wg-mem-acc] {bytes} B/workgroup, device limit {limit} B, ceiling {} wg/SM",
        limit / bytes
    );
    assert!(
        resident_hundredths
            < gow::FLASH_SHIPPED_RESIDENT_WORKGROUPS_PER_SM_IN_HUNDREDTHS_IS_512_ON_188_SMS,
        "the workgroup-memory accumulator route exists to trade registers for resident warps, so \
         it is only worth building if its shared-memory ceiling exceeds the residency the \
         shipped arm already runs at; at gow geometry it does not, and every V update would \
         additionally become a workgroup read-modify-write on the pipe that already carries the \
         staged query reads"
    );
}

#[test]
fn the_query_register_arm_removes_every_per_round_workgroup_read_and_the_max_fast_path_removes_every_correction_exp(
) {
    let (hd, fold) = GOW_SHIPPED_HEAD_DIM_64_FOLD_8_IS_THE_GEOMETRY_EVERY_MEASURED_SPAN_WAS_TAKEN_AT;
    let stock = nv_kernels::wgpu_backend::kernels::flash_decode::fold_stage1_source_sd(hd, true, fold);
    let stock_qsh = stock.matches("_qsh[qo").count();
    assert_eq!(
        stock_qsh, fold as usize * 4,
        "the shipped fold stage1 is expected to read four staged query components per folded \
         head per round; if that changed, every span this arm is measured against changed with it"
    );
    let off = gow::flash_acc_stage1_source_sd(hd, fold, false, false, false);
    let on = gow::flash_acc_stage1_source_sd(hd, fold, true, false, false);
    assert_eq!(
        off.matches("_qsh[qo").count(),
        fold as usize * 4,
        "the unarmed accumulator-file rewrite must keep the shipped per-round workgroup reads, \
         or it is not the control the register arm is measured against"
    );
    assert_eq!(
        on.matches("_qsh[qo").count(),
        0,
        "NV_GOW_FLASH_QREG exists to take the loop-invariant query rows out of workgroup memory \
         and into the register file; a rewrite that still reads them per round is inert"
    );
    let fast = gow::flash_acc_stage1_source_sd(hd, fold, false, true, false);
    let wg = gow::flash_acc_stage1_source_sd(hd, fold, false, false, true);
    assert_eq!(
        wg.matches("_wacc: array<f32,").count(),
        1,
        "NV_GOW_FLASH_WGACC declares exactly one workgroup accumulator array"
    );
    assert_eq!(
        wg.matches("_wacc[").count(),
        4 * (fold as usize) * (hd as usize / 32),
        "NV_GOW_FLASH_WGACC must move every per-thread accumulator into the workgroup array: one \
         zero-init, a two-sided read-modify-write per V update and one epilogue read per folded \
         head per 32-lane strip, and it must declare the array"
    );
    assert_eq!(
        off.matches("fd_exp(").count(),
        fold as usize * 2,
        "the unarmed rewrite spends one correction exp and one weight exp per folded head per \
         round, which is what the fast path is measured against"
    );
    assert_eq!(
        fast.matches("fd_exp(").count(),
        fold as usize * 3,
        "NV_GOW_FLASH_SMFAST keeps the general branch intact and adds a weight-only branch, so \
         the generated source carries three exps per head: two general, one fast"
    );
}

#[test]
fn arming_the_accumulator_file_rewrite_is_bit_for_bit_the_unarmed_decode_because_every_variation_is_a_multiply_by_one(
) {
    let steps = 8usize;
    let Some(off) = logit_bits_over_steps(&[], steps) else {
        return;
    };
    assert_eq!(
        off.len(),
        steps,
        "the reference run produced no logits to compare"
    );
    for (tag, pairs) in [
        (
            "restructured base",
            vec![(gow::FLASH_ACC_ENV, "1")] as Vec<(&'static str, &str)>,
        ),
        ("query register file", vec![(gow::FLASH_QREG_ENV, "1")]),
        ("workgroup-memory accumulators", vec![(gow::FLASH_WGACC_ENV, "1")]),
        ("unmoved-max fast path", vec![(gow::FLASH_SMFAST_ENV, "1")]),
        (
            "both",
            vec![(gow::FLASH_QREG_ENV, "1"), (gow::FLASH_SMFAST_ENV, "1")],
        ),
    ] {
        let on = logit_bits_over_steps(&pairs, steps).expect("adapter was present a moment ago");
        assert_eq!(
            on, off,
            "the {tag} arm changed logit bits: hoisting a loop-invariant operand into a register \
             and skipping a rescale whose factor is exactly fd_exp(0) = 1.0 are both defined to \
             preserve every rounding, so a difference here is a defect and not an fp-reorder"
        );
    }
}

#[test]
fn the_parallel_stage2_is_bit_identical_here_and_that_is_exactly_why_this_test_cannot_certify_it() {
    let steps = 6usize;
    let Some(serial) = logit_bits_over_steps(&[(gow::FLASH_STAGE2_SCC_ENV, "1")], steps) else {
        return;
    };
    let par =
        logit_bits_over_steps(&[(gow::FLASH_STAGE2_PAR_ENV, "1")], steps).expect("adapter present");
    let mut worst = 0.0f32;
    for (a, b) in serial.iter().zip(par.iter()) {
        assert_eq!(a.len(), b.len(), "logit widths must match");
        for (x, y) in a.iter().zip(b.iter()) {
            let (x, y) = (f32::from_bits(*x), f32::from_bits(*y));
            let d = (x - y).abs() / x.abs().max(y.abs()).max(1e-3);
            worst = worst.max(d);
        }
    }
    eprintln!("[par-vs-serial] worst logit rel = {worst:e}");
    assert_eq!(
        serial, par,
        "the parallel stage2 is bit-identical to the serial scan at this geometry, and it has to \
         stay that way for this test to keep its one honest use: recording that a tiny-model \
         real-adapter comparison reading exactly zero does NOT certify the arm. On \
         openai/gpt-oss-20b the same arm moves chat-wrapped ppl 20.247 -> 14.831 with the record \
         stack and 13.672 -> 16.653 with the flash sliding arm off -- different in BOTH \
         directions, which is a defect and not a reassociation. NV_GOW_FLASH_STAGE2_PAR is \
         REJECTED on quality and stays default-off until the divergence is found"
    );
    assert!(worst == 0.0, "worst logit rel {worst:e} must be exactly zero here");
}

#[test]
fn the_accumulator_file_arms_and_the_parallel_stage2_are_default_off_and_the_split_cache_bounds_every_admissible_split_count(
) {
    for env in [
        gow::FLASH_ACC_ENV,
        gow::FLASH_QREG_ENV,
        gow::FLASH_SMFAST_ENV,
        gow::FLASH_STAGE2_PAR_ENV,
        gow::FLASH_KVLINE_ENV,
        gow::FLASH_KVLINE_SLOTS_ENV,
        gow::FLASH_WGACC_ENV,
    ] {
        assert!(
            std::env::var(env).is_err(),
            "{env} leaked into the default-contract probe; this test must see the shipped \
             defaults, not a caller's pins"
        );
    }
    assert!(
        !gow::flash_acc_stage1_hoists_every_loop_invariant_uniform_out_of_the_round_loop(),
        "the accumulator-file stage1 rewrite is an opt-in arm; no performance default may flip \
         without its own round"
    );
    assert!(
        !gow::flash_qreg_the_folded_query_rows_are_loop_invariant_and_cost_one_workgroup_load_per_component_per_head_per_round(),
        "NV_GOW_FLASH_QREG is opt-in"
    );
    assert!(
        !gow::flash_smfast_a_running_max_that_does_not_move_makes_every_rescale_an_exact_multiply_by_one(),
        "NV_GOW_FLASH_SMFAST is opt-in"
    );
    assert!(
        !gow::flash_stage2_par_a_serial_split_scan_over_64_workgroups_is_the_price_of_every_wider_stage1_grid(),
        "NV_GOW_FLASH_STAGE2_PAR stays OFF and is REJECTED ON QUALITY, not merely opt-in. It is \
         worth -9.6% at depth 256, -7.4% at 8192 and -1.7% at 122880 on the wall, and it cuts \
         stage2 from 0.629 to 0.197 ms/step at 64 splits and 1.981 to 0.360 at 256 -- but the \
         chat-wrapped ppl scorer reads 14.831 against a 20.247 serial control with the record \
         stack and 16.653 against 13.672 with NV_GPTOSS_WGPU_FLASH_SLIDING=0. Different in both \
         directions is a defect, and the real-adapter tiny-model logit comparison in this suite \
         reads bit-identical, so no kernel test in the tree can see it"
    );
    assert!(
        !gow::flash_kvline_one_warp_per_kv_head_makes_a_workgroup_read_the_whole_slot_stride_of_a_position(),
        "NV_GOW_FLASH_KVLINE is opt-in: it collapses the stage1 grid's head dimension into the \
         workgroup, which moves every split-k partial count and is the fp-reorder class"
    );
    assert_eq!(
        gow::flash_kvline_slots(),
        gow::FLASH_KVLINE_SLOTS_2_KEEPS_THE_RESIDENT_WARP_COUNT_AT_THE_SHIPPED_4096_WHILE_SPLITS_STAYS_INSIDE_THE_256_CAP,
        "the kvline slot count defaults to the value that keeps the arm's resident warp count \
         equal to the shipped arm's, so a span difference is the memory partition and not the \
         occupancy"
    );
    let wgsl = gow::flash_stage2_module_source_carries_the_serial_scc_and_par_entries_side_by_side();
    assert!(
        wgsl.contains(&format!("fn {}(", gow::FLASH_STAGE2_PAR_ENTRY)),
        "the parallel stage2 entry the model can dispatch must exist in the module, or arming it \
         fails pipeline creation at model load"
    );
    assert!(
        wgsl.contains(&format!(
            "GFD_PAR_SPLIT_CACHE_{}_COVERS_EIGHT_TIMES_THE_DEEPEST_SHIPPED_BAND: u32 = {}u;",
            gow::FLASH_STAGE2_PAR_SPLIT_CACHE_512_MUST_MATCH_THE_WGSL_WORKGROUP_ARRAY_BOUND,
            gow::FLASH_STAGE2_PAR_SPLIT_CACHE_512_MUST_MATCH_THE_WGSL_WORKGROUP_ARRAY_BOUND
        )),
        "the host bound and the shader's workgroup array size must be the same number, or a \
         split count the host admits writes past the cached scale array"
    );
    assert!(
        nv_kernels::wgpu_backend::kernels::flash_decode::MAX_SPLITS as u32
            <= gow::FLASH_STAGE2_PAR_SPLIT_CACHE_512_MUST_MATCH_THE_WGSL_WORKGROUP_ARRAY_BOUND,
        "every split count the shipped env validators admit must fit the parallel stage2 split \
         cache"
    );
}
