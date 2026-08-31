#![cfg(feature = "wgpu")]

mod common;
use common::ctx_or_skip_quiet;
use common::have_gpu;
use common::nvfp4_dense_lin;
use common::rel_err;
use common::tiny_config_q3d_mixed_layers as tiny_config;
use common::tiny_weights_q3d as tiny_weights;
use nv_kernels::wgpu_backend::compose;
use nv_kernels::wgpu_backend::device::WgpuContext;
use nv_kernels::wgpu_backend::dispatch;
use nv_kernels::wgpu_backend::kernels as wk;
use nv_models::qwen3_5_dense_wgpu as q3d;
use nv_models::qwen3_5_moe_wgpu::{
    nvfp4_v2_route, quant_lane_entries, quant_lane_mode, GemvNvfp4Params, QuantRowsParams,
    SiluPairParams,
};

const W4A8_ENV: &str =
    q3d::MLP_W4A8_ENV_DEFAULT_ON_COSTS_0_0070_NATS_OF_TEACHER_FORCED_FLOOR_FOR_2PCT_DECODE_MEASURED_POST_FLIP_THE_FLIP_CLAIM_HAD_NO_LANDED_ROW_SET_0_FOR_THE_BF16_ACTIVATION_MLP_ARM;

const TINY_INTER_192_IS_THE_SMALLEST_DIM_BOTH_THE_SHIPPING_FUSED_SILU_QUANT_AND_MROW2_BLOCK_PAIRS_ADMIT:
    usize = 192;

const ARM_VS_ARM_LOGIT_REL_TOL_BUDGETS_PER_TENSOR_Q8_VS_BLOCK16_FP4_ACTIVATION_DRIFT: f32 = 0.05;

const TINY_MLP_GAIN_0P25_KEEPS_THE_MLP_CHAIN_FINITE_OVER_FOUR_LAYERS: f32 = 0.25;

const MLP_LIVENESS_MIN_REL_0P15_IS_3X_THE_DRIFT_TOL_SO_A_SCRAMBLED_MLP_CANNOT_HIDE_IN_IT:
    f32 = 0.15;

fn zero_bf16_lin(n: usize, k: usize) -> q3d::HostDenseLin {
    q3d::HostDenseLin::Bf16(nv_models::qwen3_5_moe_wgpu::HostBf16Lin {
        w: vec![0u16; n * k],
        n,
        k,
    })
}

fn tiny_mlp_only_setup(mlp_zeroed: bool) -> (q3d::Qwen3_5DenseConfig, q3d::HostDenseWeights) {
    let mut cfg = tiny_config();
    cfg.intermediate_size =
        TINY_INTER_192_IS_THE_SMALLEST_DIM_BOTH_THE_SHIPPING_FUSED_SILU_QUANT_AND_MROW2_BLOCK_PAIRS_ADMIT;
    cfg.layer_types =
        vec![nv_models::qwen3_5_moe::LayerType::FullAttention; cfg.num_hidden_layers];
    let mut hw = tiny_weights(&cfg, 0xd15e_9b00_0048);
    let mut r = common::LcgOddSeedShift33SignedUnit::new(0xd15e_9b00_0049);
    let g = TINY_MLP_GAIN_0P25_KEEPS_THE_MLP_CHAIN_FINITE_OVER_FOUR_LAYERS;
    let (hidden, inter) = (cfg.hidden_size, cfg.intermediate_size);
    let o_k = cfg.num_attention_heads * cfg.head_dim;
    for layer in &mut hw.layers {
        if mlp_zeroed {
            layer.mlp.gate = zero_bf16_lin(inter, hidden);
            layer.mlp.up = zero_bf16_lin(inter, hidden);
            layer.mlp.down = zero_bf16_lin(hidden, inter);
        } else {
            layer.mlp.gate = nvfp4_dense_lin(common::bf16_lin(&mut r, inter, hidden, g));
            layer.mlp.up = nvfp4_dense_lin(common::bf16_lin(&mut r, inter, hidden, g));
            layer.mlp.down = nvfp4_dense_lin(common::bf16_lin(&mut r, hidden, inter, g));
        }
        let q3d::HostDenseMixer::Attn(a) = &mut layer.mixer else {
            panic!("tiny_mlp_only_setup forces FullAttention on every layer");
        };
        a.o = zero_bf16_lin(hidden, o_k);
    }
    (cfg, hw)
}

fn decode_all(
    gpu: &mut q3d::Qwen3_5DenseWgpu,
    tokens: &[u32],
) -> (Vec<u32>, Vec<Vec<f32>>) {
    let mut args = Vec::new();
    let mut logits = Vec::new();
    for t in tokens {
        let (a, l) = gpu.decode_step_logits(*t).expect("decode step");
        args.push(a);
        logits.push(l);
    }
    (args, logits)
}

#[test]
fn tiny_w4a8_arm_tracks_the_default_arm_and_drops_one_dispatch_per_layer() {
    if !have_gpu() {
        return;
    }
    assert!(
        std::env::var(W4A8_ENV).is_err(),
        "{W4A8_ENV} preset in the environment; this test owns that env var"
    );
    let (cfg, hw) = tiny_mlp_only_setup(false);
    let tokens: [u32; 6] = [3, 11, 5, 40, 2, 19];

    {
        let (zcfg, hz) = tiny_mlp_only_setup(true);
        let mut sza = q3d::RefState::new(&zcfg);
        let mut szb = q3d::RefState::new(&cfg);
        let mut worst = 0f32;
        for t in &tokens {
            let za = q3d::reference_step(&zcfg, &hz, &mut sza, *t).expect("zero-mlp ref");
            let zb = q3d::reference_step(&cfg, &hw, &mut szb, *t).expect("live-mlp ref");
            let (_, rel) = rel_err(&za, &zb);
            worst = worst.max(rel);
        }
        assert!(
            worst >= MLP_LIVENESS_MIN_REL_0P15_IS_3X_THE_DRIFT_TOL_SO_A_SCRAMBLED_MLP_CANNOT_HIDE_IN_IT,
            "zeroing the MLP moved the reference logits only rel {worst}; the attention \
             out-projections are zeroed so the MLP is the only residual transform, and a \
             harness where it is not decisive cannot catch an MLP routing bug"
        );
        eprintln!("[tiny-w4a8] MLP liveness: zeroed-MLP reference rel {worst:.3}");
    }

    std::env::set_var(W4A8_ENV, "0");
    let base = q3d::Qwen3_5DenseWgpu::new(cfg.clone(), &hw, 32);
    std::env::remove_var(W4A8_ENV);
    let mut base = base.expect("build bf16-activation arm pinned off because the default is now w4a8");
    let base_passes = base.pass_count();
    let (base_args, base_logits) = decode_all(&mut base, &tokens);
    drop(base);

    std::env::set_var(W4A8_ENV, "1");
    let arm = q3d::Qwen3_5DenseWgpu::new(cfg.clone(), &hw, 32);
    std::env::remove_var(W4A8_ENV);
    let mut arm = arm.expect("build w4a8 arm");
    let arm_passes = arm.pass_count();
    assert_eq!(
        base_passes - cfg.num_hidden_layers,
        arm_passes,
        "the w4a8 arm folds the standalone q3w_quant_rows dispatch into the post-attn \
         rmsnorm, so it must record exactly one fewer pass per layer \
         (default {base_passes}, w4a8 {arm_passes}, layers {})",
        cfg.num_hidden_layers
    );
    let (arm_args, arm_logits) = decode_all(&mut arm, &tokens);

    let mut st = q3d::RefState::new(&cfg);
    let mut any_diff = false;
    for (i, t) in tokens.iter().enumerate() {
        let want = q3d::reference_step(&cfg, &hw, &mut st, *t).expect("reference step");
        for (which, l) in [
            ("ref", &want),
            ("base", &base_logits[i]),
            ("w4a8", &arm_logits[i]),
        ] {
            assert!(
                l.iter().all(|v| v.is_finite()),
                "step {i}: {which} logits contain non-finite values; rel_err folds NaN \
                 to zero, so every comparison below would be vacuous"
            );
        }
        any_diff |= arm_logits[i]
            .iter()
            .zip(&base_logits[i])
            .any(|(a, b)| a.to_bits() != b.to_bits());
        let (_, rel_base) = rel_err(&base_logits[i], &want);
        let (_, rel_arm) = rel_err(&arm_logits[i], &want);
        let (abs_ab, rel_ab) = rel_err(&arm_logits[i], &base_logits[i]);
        eprintln!(
            "step {i}: tok={t} base_arg={} w4a8_arg={} rel_vs_ref base={rel_base:.6} \
             w4a8={rel_arm:.6} arm_vs_arm rel={rel_ab:.6} abs={abs_ab:.6}",
            base_args[i], arm_args[i]
        );
        assert_eq!(
            base_args[i], arm_args[i],
            "step {i}: argmax diverged between the default and w4a8 arms"
        );
        assert!(
            rel_ab < ARM_VS_ARM_LOGIT_REL_TOL_BUDGETS_PER_TENSOR_Q8_VS_BLOCK16_FP4_ACTIVATION_DRIFT,
            "step {i}: w4a8 logits drifted {rel_ab} from the default arm"
        );
        assert!(
            rel_arm < ARM_VS_ARM_LOGIT_REL_TOL_BUDGETS_PER_TENSOR_Q8_VS_BLOCK16_FP4_ACTIVATION_DRIFT,
            "step {i}: w4a8 logits drifted {rel_arm} from the CPU reference"
        );
    }
    assert!(
        any_diff,
        "w4a8 and default logits are bit-identical over every step; per-tensor q8 and \
         block16 fp4 activation quantization cannot coincide, so the w4a8 MLP route \
         did not actually run"
    );
}

const GEMV_W4A8_WGSL: &str = include_str!("../../nv-kernels/wgsl/gemv_nvfp4_w4a8.wgsl");
const W4A8_PRODUCER_WGSL: &str = include_str!("../../nv-kernels/wgsl/w4a8_producer_rowquant.wgsl");

const DRAM_HONEST_WEIGHT_REPLICAS_OVERFLOW_THE_128MB_BLACKWELL_L2: usize = 8;
const STEP_LADDER_ROUNDS: usize = 9;
const STEP_ITERS_PER_ROUND: usize = 100;

#[repr(C)]
#[derive(Clone, Copy, Default, bytemuck::Pod, bytemuck::Zeroable)]
struct RmsResParams {
    hidden: u32,
    batch: u32,
    eps: f32,
    words_per_row: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Default, bytemuck::Pod, bytemuck::Zeroable)]
struct ProdParams {
    k_blocks: u32,
    words_per_row: u32,
    hidden: u32,
    eps: f32,
}

struct Lcg(u64);

impl Lcg {
    fn new(seed: u64) -> Self {
        Lcg(seed | 1)
    }
    fn next_u32(&mut self) -> u32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (self.0 >> 32) as u32
    }
    fn words(&mut self, n: usize) -> Vec<u32> {
        (0..n).map(|_| self.next_u32()).collect()
    }
    fn bf16_words(&mut self, n_elems: usize) -> Vec<u32> {
        (0..n_elems / 2)
            .map(|_| {
                let unit = |r: &mut Self| {
                    half::bf16::from_f32((r.next_u32() as f32 / u32::MAX as f32) * 2.0 - 1.0)
                        .to_bits() as u32
                };
                let lo = unit(self);
                let hi = unit(self);
                lo | (hi << 16)
            })
            .collect()
    }
    fn scale_words(&mut self, n: usize) -> Vec<u32> {
        (0..n)
            .map(|_| {
                let mut w = 0u32;
                for b in 0..4 {
                    let exp = 1 + (self.next_u32() % 8);
                    let mant = self.next_u32() % 8;
                    w |= ((exp << 3) | mant) << (8 * b);
                }
                w
            })
            .collect()
    }
}

struct Pass {
    pipeline: std::sync::Arc<wgpu::ComputePipeline>,
    binds: Vec<wgpu::BindGroup>,
    groups: (u32, u32, u32),
}

struct Arm {
    label: String,
    passes: Vec<Pass>,
}

fn time_arm(ctx: &WgpuContext, arm: &Arm, iters: usize) -> f64 {
    let mut enc = ctx.device.create_command_encoder(&Default::default());
    {
        let mut pass = enc.begin_compute_pass(&Default::default());
        for i in 0..iters {
            for p in &arm.passes {
                pass.set_pipeline(&p.pipeline);
                pass.set_bind_group(0, &p.binds[i % p.binds.len()], &[]);
                pass.dispatch_workgroups(p.groups.0, p.groups.1, p.groups.2);
            }
        }
    }
    let cb = enc.finish();
    let t0 = std::time::Instant::now();
    ctx.queue.submit([cb]);
    ctx.poll_blocking().expect("timed poll");
    t0.elapsed().as_secs_f64()
}

fn med(v: &[f64]) -> f64 {
    let mut s = v.to_vec();
    s.sort_by(|a, b| a.partial_cmp(b).unwrap());
    s[s.len() / 2]
}

struct GemvW {
    w: Vec<wgpu::Buffer>,
    ws: Vec<wgpu::Buffer>,
    n: usize,
    k: usize,
}

fn gemv_weights(ctx: &WgpuContext, rng: &mut Lcg, n: usize, k: usize) -> GemvW {
    let kb = k / 16;
    let packed = rng.words(n * k / 8);
    let scales = rng.scale_words(wk::gemv_nvfp4::swizzled_scale_len(n, kb).div_ceil(4));
    GemvW {
        w: (0..DRAM_HONEST_WEIGHT_REPLICAS_OVERFLOW_THE_128MB_BLACKWELL_L2)
            .map(|r| dispatch::storage_from_slice(ctx, &format!("sw-{n}x{k}-{r}"), &packed))
            .collect(),
        ws: (0..DRAM_HONEST_WEIGHT_REPLICAS_OVERFLOW_THE_128MB_BLACKWELL_L2)
            .map(|r| dispatch::storage_from_slice(ctx, &format!("ss-{n}x{k}-{r}"), &scales))
            .collect(),
        n,
        k,
    }
}

fn mrow2_pass(
    ctx: &WgpuContext,
    w: &GemvW,
    xq: &wgpu::Buffer,
    xs: &wgpu::Buffer,
    sel: &wgpu::Buffer,
    alpha_dummy: &wgpu::Buffer,
    y: &wgpu::Buffer,
) -> Pass {
    let kb = w.k / 16;
    let r = nvfp4_v2_route(ctx, w.n, w.k, kb, 1).expect("shipping decode route");
    assert_eq!(
        r.entry, "q3w_gemv_nvfp4_mrow2",
        "the step ladder models the shipping decode route; another entry voids the A/B"
    );
    let groups = dispatch::workgroup_count_1d(ctx, w.n as u64, r.rows_per_group);
    let p = dispatch::uniform_from(
        ctx,
        "sp",
        &GemvNvfp4Params {
            alpha: 1.0,
            n_rows: w.n as u32,
            k_blocks: kb as u32,
            k_tiles: kb.div_ceil(4) as u32,
            groups_x: groups.0,
            w_e_stride_vec2: 0,
            sf_e_stride_bytes: 0,
            x_slot_stride_vec2: 0,
            xsf_slot_stride_bytes: 0,
            y_slot_stride_words: (w.n / 2) as u32,
            per_expert_alpha: 0,
            m_slots_sharing_expert_zero: 0,
        },
    );
    let (w_slot, x_slot) = if r.vec4 { (18, 19) } else { (10, 12) };
    let pipeline =
        dispatch::cached_compute_pipeline(ctx, "s-mrow2", &r.source, r.entry).expect("mrow2");
    let binds = w
        .w
        .iter()
        .zip(&w.ws)
        .map(|(wb, wsb)| {
            dispatch::bind_group(
                ctx,
                &pipeline,
                &[
                    (w_slot, wb),
                    (11, wsb),
                    (x_slot, xq),
                    (13, xs),
                    (14, &p),
                    (15, y),
                    (16, sel),
                    (17, alpha_dummy),
                ],
            )
        })
        .collect();
    std::mem::forget(p);
    Pass {
        pipeline,
        binds,
        groups,
    }
}

fn w4a8_pass(
    ctx: &WgpuContext,
    src: &str,
    w: &GemvW,
    xq: &wgpu::Buffer,
    xs: &wgpu::Buffer,
    y: &wgpu::Buffer,
) -> Pass {
    let groups = dispatch::workgroup_count_1d(ctx, w.n as u64, 8);
    let p = dispatch::uniform_from(
        ctx,
        "wp",
        &wk::gemv_nvfp4::gemv_params(1.0, w.n, w.k, groups.0),
    );
    let pipeline = dispatch::cached_compute_pipeline(ctx, "s-w4a8", src, "gemv_nvfp4_w4a8_mrow2")
        .expect("w4a8 gemv");
    let binds = w
        .w
        .iter()
        .zip(&w.ws)
        .map(|(wb, wsb)| {
            dispatch::bind_group(
                ctx,
                &pipeline,
                &[(0, wb), (1, wsb), (2, xq), (3, xs), (4, &p), (5, y)],
            )
        })
        .collect();
    std::mem::forget(p);
    Pass {
        pipeline,
        binds,
        groups,
    }
}

fn one_pass(
    ctx: &WgpuContext,
    label: &str,
    src: &str,
    entry: &str,
    binds: &[(u32, &wgpu::Buffer)],
    groups: (u32, u32, u32),
) -> Pass {
    let pipeline = dispatch::cached_compute_pipeline(ctx, label, src, entry).expect(label);
    let bind = dispatch::bind_group(ctx, &pipeline, binds);
    Pass {
        pipeline,
        binds: vec![bind],
        groups,
    }
}

#[test]
#[ignore = "GPU rate measurement of the decode MLP step, default vs NV_Q3D_MLP_W4A8 dispatch \
            sequences; run via nvk.sh probe with --nocapture"]
fn mlp_step_default_vs_w4a8_arm_ladder() {
    let Some(ctx) = ctx_or_skip_quiet("q3d_mlp_w4a8_step_ladder") else {
        return;
    };
    let (hidden, inter) = (5120usize, 17408usize);
    let (kb_h, kb_i) = (hidden / 16, inter / 16);
    let mut rng = Lcg::new(0xa1);

    let gate = gemv_weights(ctx, &mut rng, inter, hidden);
    let up = gemv_weights(ctx, &mut rng, inter, hidden);
    let down = gemv_weights(ctx, &mut rng, hidden, inter);

    let mixed = dispatch::storage_from_slice(ctx, "s-mixed", &rng.bf16_words(hidden));
    let res = dispatch::storage_from_slice(ctx, "s-res", &rng.bf16_words(hidden));
    let post_w = dispatch::storage_from_slice(ctx, "s-postw", &rng.bf16_words(hidden));
    let normed_post = dispatch::storage_zeroed(ctx, "s-np", (hidden * 2) as u64);
    let y_gate = dispatch::storage_zeroed(ctx, "s-yg", (inter * 2) as u64);
    let y_up = dispatch::storage_zeroed(ctx, "s-yu", (inter * 2) as u64);
    let mlp_out = dispatch::storage_zeroed(ctx, "s-mo", (hidden * 2) as u64);

    let xq4_h = dispatch::storage_zeroed(ctx, "s-xq4h", (hidden / 2) as u64);
    let xs4_h = dispatch::storage_zeroed(ctx, "s-xs4h", (kb_h.div_ceil(4) * 4) as u64);
    let xq4_i = dispatch::storage_zeroed(ctx, "s-xq4i", (inter / 2) as u64);
    let xs4_i = dispatch::storage_zeroed(ctx, "s-xs4i", (kb_i.div_ceil(4) * 4) as u64);
    let sel = dispatch::storage_from_slice(ctx, "s-sel", &[0u32]);
    let alpha_dummy = dispatch::storage_from_slice(ctx, "s-al", &[1.0f32]);
    let globals = dispatch::storage_from_slice(ctx, "s-gl", &[1.0f32]);

    let xq8_h = dispatch::storage_zeroed(ctx, "s-xq8h", hidden as u64);
    let xs8_h = dispatch::storage_from_slice(ctx, "s-xs8h", &[0f32]);
    let xq8_i = dispatch::storage_zeroed(ctx, "s-xq8i", inter as u64);
    let xs8_i = dispatch::storage_from_slice(ctx, "s-xs8i", &[0f32]);

    let rmsres_src = compose(wk::rmsnorm_residual::WGSL);
    let quant_src = nv_models::qwen3_5_moe_wgpu::nvfp4_quant_source();
    let w4a8_src = compose(GEMV_W4A8_WGSL);
    let prod_src = compose(W4A8_PRODUCER_WGSL);

    let rms_p = dispatch::uniform_from(
        ctx,
        "s-rmsp",
        &RmsResParams {
            hidden: hidden as u32,
            batch: 1,
            eps: 1e-6,
            words_per_row: (hidden / 2) as u32,
        },
    );
    let rmsres_pass = || {
        one_pass(
            ctx,
            "s-rms",
            &rmsres_src,
            "rmsnorm_residual_bf16",
            &[
                (0, &mixed),
                (1, &res),
                (2, &post_w),
                (3, &normed_post),
                (4, &rms_p),
            ],
            (1, 1, 1),
        )
    };

    let qr_p = dispatch::uniform_from(
        ctx,
        "s-qrp",
        &QuantRowsParams {
            k_blocks: kb_h as u32,
            n_slots: 1,
            use_sel: 0,
            x_slot_stride_elems: 0,
        },
    );
    let quant_rows_pass = one_pass(
        ctx,
        "s-qr",
        &quant_src,
        "q3w_quant_rows",
        &[
            (10, &normed_post),
            (11, &qr_p),
            (12, &xq4_h),
            (13, &xs4_h),
            (14, &sel),
            (15, &globals),
        ],
        ((kb_h as u32).div_ceil(256).max(1), 1, 1),
    );

    let lane = quant_lane_mode().and_then(|wg| {
        quant_lane_entries()
            .into_iter()
            .find(|(_, r, w)| *r == "q3w_silu_mul_quant" && *w == wg)
            .map(|(e, _, w)| (e, w))
    });
    let (silu_entry, silu_wgb) = match lane {
        Some((e, wg)) => (e, wg / 8),
        None => ("q3w_silu_mul_quant", 256),
    };
    let sq_p = dispatch::uniform_from(
        ctx,
        "s-sqp",
        &QuantRowsParams {
            k_blocks: kb_i as u32,
            n_slots: 1,
            use_sel: 0,
            x_slot_stride_elems: 0,
        },
    );
    let sq_pp = dispatch::uniform_from(
        ctx,
        "s-sqpp",
        &SiluPairParams {
            u_off_elems: 0,
            ..Default::default()
        },
    );
    let silu_quant_pass = one_pass(
        ctx,
        "s-sq",
        &quant_src,
        silu_entry,
        &[
            (11, &sq_p),
            (12, &xq4_i),
            (13, &xs4_i),
            (14, &sel),
            (15, &globals),
            (16, &y_gate),
            (17, &y_up),
            (18, &sq_pp),
        ],
        ((kb_i as u32).div_ceil(silu_wgb as u32).max(1), 1, 1),
    );

    let prod_h = dispatch::uniform_from(
        ctx,
        "s-ph",
        &ProdParams {
            k_blocks: kb_h as u32,
            words_per_row: (hidden / 2) as u32,
            hidden: hidden as u32,
            eps: 1e-6,
        },
    );
    let rmsres_fused_pass = one_pass(
        ctx,
        "s-rmsf",
        &prod_src,
        "rmsnorm_residual_bf16_rowquant_evenodd",
        &[
            (0, &mixed),
            (1, &res),
            (2, &post_w),
            (3, &normed_post),
            (4, &prod_h),
            (5, &xq8_h),
            (6, &xs8_h),
        ],
        (1, 1, 1),
    );
    let prod_i = dispatch::uniform_from(
        ctx,
        "s-pi",
        &ProdParams {
            k_blocks: kb_i as u32,
            words_per_row: (inter / 2) as u32,
            hidden: inter as u32,
            eps: 0.0,
        },
    );
    let silu_fused_pass = one_pass(
        ctx,
        "s-siluf",
        &prod_src,
        "silu_mul_rowquant_evenodd",
        &[
            (4, &prod_i),
            (5, &xq8_i),
            (6, &xs8_i),
            (7, &y_gate),
            (8, &y_up),
        ],
        (1, 1, 1),
    );

    let arms = vec![
        Arm {
            label: "default nvfp4 decode MLP".into(),
            passes: vec![
                rmsres_pass(),
                quant_rows_pass,
                mrow2_pass(ctx, &gate, &xq4_h, &xs4_h, &sel, &alpha_dummy, &y_gate),
                mrow2_pass(ctx, &up, &xq4_h, &xs4_h, &sel, &alpha_dummy, &y_up),
                silu_quant_pass,
                mrow2_pass(ctx, &down, &xq4_i, &xs4_i, &sel, &alpha_dummy, &mlp_out),
            ],
        },
        Arm {
            label: "w4a8 fused-producer MLP".into(),
            passes: vec![
                rmsres_fused_pass,
                w4a8_pass(ctx, &w4a8_src, &gate, &xq8_h, &xs8_h, &y_gate),
                w4a8_pass(ctx, &w4a8_src, &up, &xq8_h, &xs8_h, &y_up),
                silu_fused_pass,
                w4a8_pass(ctx, &w4a8_src, &down, &xq8_i, &xs8_i, &mlp_out),
            ],
        },
        Arm {
            label: "default MLP [NULL]".into(),
            passes: vec![
                rmsres_pass(),
                one_pass(
                    ctx,
                    "s-qr",
                    &quant_src,
                    "q3w_quant_rows",
                    &[
                        (10, &normed_post),
                        (11, &qr_p),
                        (12, &xq4_h),
                        (13, &xs4_h),
                        (14, &sel),
                        (15, &globals),
                    ],
                    ((kb_h as u32).div_ceil(256).max(1), 1, 1),
                ),
                mrow2_pass(ctx, &gate, &xq4_h, &xs4_h, &sel, &alpha_dummy, &y_gate),
                mrow2_pass(ctx, &up, &xq4_h, &xs4_h, &sel, &alpha_dummy, &y_up),
                one_pass(
                    ctx,
                    "s-sq",
                    &quant_src,
                    silu_entry,
                    &[
                        (11, &sq_p),
                        (12, &xq4_i),
                        (13, &xs4_i),
                        (14, &sel),
                        (15, &globals),
                        (16, &y_gate),
                        (17, &y_up),
                        (18, &sq_pp),
                    ],
                    ((kb_i as u32).div_ceil(silu_wgb as u32).max(1), 1, 1),
                ),
                mrow2_pass(ctx, &down, &xq4_i, &xs4_i, &sel, &alpha_dummy, &mlp_out),
            ],
        },
    ];

    let step_bytes =
        (2 * (inter * hidden / 2 + wk::gemv_nvfp4::swizzled_scale_len(inter, kb_h))
            + (hidden * inter / 2 + wk::gemv_nvfp4::swizzled_scale_len(hidden, kb_i)))
            as f64;
    eprintln!(
        "--- decode MLP step hidden={hidden} inter={inter} {:.1} MiB weights+scales/step, \
         x{} replicas, {} steps/round, {} interleaved rounds ---",
        step_bytes / (1024.0 * 1024.0),
        DRAM_HONEST_WEIGHT_REPLICAS_OVERFLOW_THE_128MB_BLACKWELL_L2,
        STEP_ITERS_PER_ROUND,
        STEP_LADDER_ROUNDS
    );
    for arm in &arms {
        for _ in 0..2 {
            time_arm(ctx, arm, STEP_ITERS_PER_ROUND);
        }
    }
    let mut rounds: Vec<Vec<f64>> = vec![Vec::new(); arms.len()];
    for _ in 0..STEP_LADDER_ROUNDS {
        for (i, arm) in arms.iter().enumerate() {
            rounds[i].push(time_arm(ctx, arm, STEP_ITERS_PER_ROUND));
        }
    }
    for (i, arm) in arms.iter().enumerate() {
        let ms: Vec<f64> = rounds[i]
            .iter()
            .map(|t| t * 1e3 / STEP_ITERS_PER_ROUND as f64)
            .collect();
        let ratio: Vec<f64> = rounds[0]
            .iter()
            .zip(&rounds[i])
            .map(|(b, t)| b / t)
            .collect();
        let rates: Vec<f64> = rounds[i]
            .iter()
            .map(|t| step_bytes * STEP_ITERS_PER_ROUND as f64 / t / 1e9)
            .collect();
        eprintln!(
            "{:<28} {:>8.4} ms/step med  {:>7.1} GB/s med | vs arm0 {:.3}x med",
            arm.label,
            med(&ms),
            med(&rates),
            med(&ratio)
        );
    }
}
