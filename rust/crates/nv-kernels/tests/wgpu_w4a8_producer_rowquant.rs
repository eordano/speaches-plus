#![cfg(feature = "wgpu")]

mod common;
use common::ctx_or_skip_quiet_unqualified as ctx_or_skip;
use half::bf16;
use nv_kernels::wgpu_backend::compose;
use nv_kernels::wgpu_backend::device::WgpuContext;
use nv_kernels::wgpu_backend::dispatch;
use nv_kernels::wgpu_backend::kernels::gemv_nvfp4 as g4;

const PRODUCER_WGSL: &str = include_str!("../wgsl/w4a8_producer_rowquant.wgsl");
const W48_WGSL: &str = include_str!("../wgsl/gemv_nvfp4_w4a8.wgsl");
const RMSRES_WGSL: &str = include_str!("../wgsl/rmsnorm_residual.wgsl");
const SILU_WGSL: &str = include_str!("../wgsl/q3d_misc.wgsl");

const RMSRES_FUSED_ENTRY: &str = "rmsnorm_residual_bf16_rowquant_evenodd";
const SILU_FUSED_ENTRY: &str = "silu_mul_rowquant_evenodd";
const EMIT_EVEN_ANCHOR: &str = "w48p_xq[2u * g] = q0.x | (q1.x << 8u) | (q2.x << 16u) | (q3.x << 24u);";

const RMS_EPS: f32 = 1e-6;
const QUANT_WG: u32 = 256;

fn producer_src() -> String {
    compose(PRODUCER_WGSL)
}

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

#[repr(C)]
#[derive(Clone, Copy, Default, bytemuck::Pod, bytemuck::Zeroable)]
struct SiluParams {
    n_words: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
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
    fn next_unit_f32(&mut self) -> f32 {
        (self.next_u32() as f32 / u32::MAX as f32) * 2.0 - 1.0
    }
    fn bf16_words(&mut self, n_elems: usize, gain: f32) -> Vec<u32> {
        assert!(n_elems % 2 == 0);
        (0..n_elems / 2)
            .map(|_| {
                let lo = bf16::from_f32(self.next_unit_f32() * gain).to_bits() as u32;
                let hi = bf16::from_f32(self.next_unit_f32() * gain).to_bits() as u32;
                lo | (hi << 16)
            })
            .collect()
    }
}

fn prod_params(k: usize) -> ProdParams {
    ProdParams {
        k_blocks: (k / 16) as u32,
        words_per_row: (k / 2) as u32,
        hidden: k as u32,
        eps: RMS_EPS,
    }
}

fn quant_groups(k: usize) -> (u32, u32, u32) {
    ((k / 8).div_ceil(QUANT_WG as usize) as u32, 1, 1)
}

struct QuantOut {
    res: Vec<u32>,
    out: Vec<u32>,
    xq: Vec<u32>,
    xs_bits: u32,
}

fn run_rmsres_unfused(ctx: &WgpuContext, k: usize, x: &[u32], res: &[u32], w: &[u32]) -> QuantOut {
    let words = k / 2;
    let xb = dispatch::storage_from_slice(ctx, "w48p-x", x);
    let rb = dispatch::storage_from_slice(ctx, "w48p-res", res);
    let wb = dispatch::storage_from_slice(ctx, "w48p-w", w);
    let ob = dispatch::storage_zeroed(ctx, "w48p-out", (words * 4) as u64);
    let rp = dispatch::uniform_from(
        ctx,
        "w48p-rp",
        &RmsResParams {
            hidden: k as u32,
            batch: 1,
            eps: RMS_EPS,
            words_per_row: words as u32,
        },
    );
    dispatch::run(
        ctx,
        "w48p-rmsres",
        &compose(RMSRES_WGSL),
        "rmsnorm_residual_bf16",
        &[(0, &xb), (1, &rb), (2, &wb), (3, &ob), (4, &rp)],
        (1, 1, 1),
    )
    .expect("rmsres dispatch");
    let qb = dispatch::storage_zeroed(ctx, "w48p-xq", (k / 4 * 4) as u64);
    let sb = dispatch::storage_from_slice(ctx, "w48p-xs", &[f32::NAN]);
    let up = dispatch::uniform_from(ctx, "w48p-qp", &g4::gemv_params(1.0, 1, k, 1));
    dispatch::run(
        ctx,
        "w48p-quant",
        &compose(W48_WGSL),
        "w48_quantize_x_evenodd",
        &[(6, &ob), (7, &qb), (8, &sb), (4, &up)],
        quant_groups(k),
    )
    .expect("quant dispatch");
    QuantOut {
        res: dispatch::read_back(ctx, &rb, words).expect("res read back"),
        out: dispatch::read_back(ctx, &ob, words).expect("out read back"),
        xq: dispatch::read_back(ctx, &qb, k / 4).expect("xq read back"),
        xs_bits: dispatch::read_back::<f32>(ctx, &sb, 1).expect("xs read back")[0].to_bits(),
    }
}

fn run_rmsres_fused(
    ctx: &WgpuContext,
    src: &str,
    k: usize,
    x: &[u32],
    res: &[u32],
    w: &[u32],
) -> QuantOut {
    let words = k / 2;
    let xb = dispatch::storage_from_slice(ctx, "w48p-x", x);
    let rb = dispatch::storage_from_slice(ctx, "w48p-res", res);
    let wb = dispatch::storage_from_slice(ctx, "w48p-w", w);
    let ob = dispatch::storage_zeroed(ctx, "w48p-out", (words * 4) as u64);
    let qb = dispatch::storage_zeroed(ctx, "w48p-xq", (k / 4 * 4) as u64);
    let sb = dispatch::storage_from_slice(ctx, "w48p-xs", &[f32::NAN]);
    let pp = dispatch::uniform_from(ctx, "w48p-pp", &prod_params(k));
    dispatch::run(
        ctx,
        "w48p-fused",
        src,
        RMSRES_FUSED_ENTRY,
        &[
            (0, &xb),
            (1, &rb),
            (2, &wb),
            (3, &ob),
            (4, &pp),
            (5, &qb),
            (6, &sb),
        ],
        (1, 1, 1),
    )
    .expect("fused rmsres dispatch");
    QuantOut {
        res: dispatch::read_back(ctx, &rb, words).expect("res read back"),
        out: dispatch::read_back(ctx, &ob, words).expect("out read back"),
        xq: dispatch::read_back(ctx, &qb, k / 4).expect("xq read back"),
        xs_bits: dispatch::read_back::<f32>(ctx, &sb, 1).expect("xs read back")[0].to_bits(),
    }
}

fn run_silu_unfused(ctx: &WgpuContext, k: usize, gate: &[u32], up: &[u32]) -> (Vec<u32>, u32) {
    let words = k / 2;
    let gb = dispatch::storage_from_slice(ctx, "w48p-g", gate);
    let ub = dispatch::storage_from_slice(ctx, "w48p-u", up);
    let ab = dispatch::storage_zeroed(ctx, "w48p-act", (words * 4) as u64);
    let sp = dispatch::uniform_from(
        ctx,
        "w48p-sp",
        &SiluParams {
            n_words: words as u32,
            ..Default::default()
        },
    );
    dispatch::run(
        ctx,
        "w48p-silu",
        &compose(SILU_WGSL),
        "q3w_silu_mul",
        &[(10, &gb), (11, &ub), (12, &ab), (13, &sp)],
        dispatch::workgroup_count_1d(ctx, words as u64, 64),
    )
    .expect("silu dispatch");
    let qb = dispatch::storage_zeroed(ctx, "w48p-xq", (k / 4 * 4) as u64);
    let sb = dispatch::storage_from_slice(ctx, "w48p-xs", &[f32::NAN]);
    let up_ = dispatch::uniform_from(ctx, "w48p-qp", &g4::gemv_params(1.0, 1, k, 1));
    dispatch::run(
        ctx,
        "w48p-quant",
        &compose(W48_WGSL),
        "w48_quantize_x_evenodd",
        &[(6, &ab), (7, &qb), (8, &sb), (4, &up_)],
        quant_groups(k),
    )
    .expect("quant dispatch");
    (
        dispatch::read_back(ctx, &qb, k / 4).expect("xq read back"),
        dispatch::read_back::<f32>(ctx, &sb, 1).expect("xs read back")[0].to_bits(),
    )
}

fn run_silu_fused(
    ctx: &WgpuContext,
    src: &str,
    k: usize,
    gate: &[u32],
    up: &[u32],
) -> (Vec<u32>, u32) {
    let gb = dispatch::storage_from_slice(ctx, "w48p-g", gate);
    let ub = dispatch::storage_from_slice(ctx, "w48p-u", up);
    let qb = dispatch::storage_zeroed(ctx, "w48p-xq", (k / 4 * 4) as u64);
    let sb = dispatch::storage_from_slice(ctx, "w48p-xs", &[f32::NAN]);
    let pp = dispatch::uniform_from(ctx, "w48p-pp", &prod_params(k));
    dispatch::run(
        ctx,
        "w48p-silu-fused",
        src,
        SILU_FUSED_ENTRY,
        &[(4, &pp), (5, &qb), (6, &sb), (7, &gb), (8, &ub)],
        (1, 1, 1),
    )
    .expect("fused silu dispatch");
    (
        dispatch::read_back(ctx, &qb, k / 4).expect("xq read back"),
        dispatch::read_back::<f32>(ctx, &sb, 1).expect("xs read back")[0].to_bits(),
    )
}

fn assert_words_eq(tag: &str, got: &[u32], want: &[u32]) {
    let diff: Vec<usize> = got
        .iter()
        .zip(want)
        .enumerate()
        .filter(|(_, (g, w))| g != w)
        .map(|(i, _)| i)
        .collect();
    assert!(
        diff.is_empty(),
        "{tag}: {} of {} words differ (first at {}); the fused epilogue replays the exact \
         producer and w48_quantize_x_evenodd expression sequences, so any drift is an \
         indexing or packing bug, not rounding",
        diff.len(),
        got.len(),
        diff.first().copied().unwrap_or(0)
    );
}

const RMSRES_SHAPES: [(usize, u64); 3] = [(5120, 0x61), (4096, 0x62), (736, 0x63)];
const SILU_SHAPES: [(usize, u64); 3] = [(17408, 0x71), (5120, 0x72), (736, 0x73)];

#[test]
fn fused_rmsnorm_rowquant_is_bitexact_vs_the_unfused_kernel_pair_and_deterministic() {
    let Some(ctx) = ctx_or_skip("w4a8_rmsres_rowquant_parity") else {
        return;
    };
    let src = producer_src();
    for &(k, seed) in &RMSRES_SHAPES {
        let mut rng = Lcg::new(seed);
        let x = rng.bf16_words(k, 1.0);
        let res = rng.bf16_words(k, 1.0);
        let w = rng.bf16_words(k, 0.5);
        let a = run_rmsres_unfused(ctx, k, &x, &res, &w);
        let b = run_rmsres_fused(ctx, &src, k, &x, &res, &w);
        assert_words_eq(&format!("k={k} res"), &b.res, &a.res);
        assert_words_eq(&format!("k={k} normed_post"), &b.out, &a.out);
        assert_words_eq(&format!("k={k} xq"), &b.xq, &a.xq);
        assert_eq!(
            b.xs_bits, a.xs_bits,
            "k={k}: per-tensor scale bits diverged (fused {:08x} unfused {:08x})",
            b.xs_bits, a.xs_bits
        );
        let c = run_rmsres_fused(ctx, &src, k, &x, &res, &w);
        assert_words_eq(&format!("k={k} rerun xq"), &c.xq, &b.xq);
        assert_eq!(c.xs_bits, b.xs_bits, "k={k}: rerun scale bits moved");
        eprintln!("rmsres rowquant k={k}: bit-exact vs unfused pair, rerun stable");
    }
}

#[test]
fn fused_silu_mul_rowquant_is_bitexact_vs_the_unfused_kernel_pair_and_deterministic() {
    let Some(ctx) = ctx_or_skip("w4a8_silu_rowquant_parity") else {
        return;
    };
    let src = producer_src();
    for &(k, seed) in &SILU_SHAPES {
        let mut rng = Lcg::new(seed);
        let gate = rng.bf16_words(k, 2.0);
        let up = rng.bf16_words(k, 1.0);
        let (xq_a, xs_a) = run_silu_unfused(ctx, k, &gate, &up);
        let (xq_b, xs_b) = run_silu_fused(ctx, &src, k, &gate, &up);
        assert_words_eq(&format!("k={k} xq"), &xq_b, &xq_a);
        assert_eq!(
            xs_b, xs_a,
            "k={k}: per-tensor scale bits diverged (fused {xs_b:08x} unfused {xs_a:08x})"
        );
        let (xq_c, xs_c) = run_silu_fused(ctx, &src, k, &gate, &up);
        assert_words_eq(&format!("k={k} rerun xq"), &xq_c, &xq_b);
        assert_eq!(xs_c, xs_b, "k={k}: rerun scale bits moved");
        eprintln!("silu rowquant k={k}: bit-exact vs unfused pair, rerun stable");
    }
}

#[test]
fn planted_even_word_swap_is_caught_by_the_parity_harness() {
    let Some(ctx) = ctx_or_skip("w4a8_producer_planted_bug") else {
        return;
    };
    let good = producer_src();
    let bad = good.replacen(
        EMIT_EVEN_ANCHOR,
        "w48p_xq[2u * g] = q0.y | (q1.y << 8u) | (q2.y << 16u) | (q3.y << 24u);",
        1,
    );
    assert_ne!(
        bad, good,
        "even-word emit anchor missing from w4a8_producer_rowquant.wgsl; the planted-bug \
         gate is vacuous"
    );
    let k = 736usize;
    let mut rng = Lcg::new(0x81);
    let gate = rng.bf16_words(k, 2.0);
    let up = rng.bf16_words(k, 1.0);
    let (xq_a, _) = run_silu_unfused(ctx, k, &gate, &up);
    let (xq_b, xs_b) = run_silu_fused(ctx, &bad, k, &gate, &up);
    let diff = xq_a.iter().zip(&xq_b).filter(|(a, b)| a != b).count();
    assert!(
        diff > 0,
        "swapped even-word emission survived parity; random activations make even and odd \
         q8 lanes disagree by construction, so the harness would miss a real packing \
         regression"
    );
    let (_, xs_a) = run_silu_unfused(ctx, k, &gate, &up);
    assert_eq!(
        xs_b, xs_a,
        "the planted emit swap must not move the per-tensor scale; if it does, the scale \
         and packing checks are entangled and the gate localizes nothing"
    );
}

const PRODUCER_BENCH_ROUNDS: usize = 9;
const PRODUCER_BENCH_ITERS: usize = 2000;

struct Arm {
    label: String,
    passes: Vec<(std::sync::Arc<wgpu::ComputePipeline>, wgpu::BindGroup, (u32, u32, u32))>,
}

fn time_arm(ctx: &WgpuContext, arm: &Arm, iters: usize) -> f64 {
    let mut enc = ctx.device.create_command_encoder(&Default::default());
    {
        let mut pass = enc.begin_compute_pass(&Default::default());
        for _ in 0..iters {
            for (p, b, g) in &arm.passes {
                pass.set_pipeline(p);
                pass.set_bind_group(0, b, &[]);
                pass.dispatch_workgroups(g.0, g.1, g.2);
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

#[test]
#[ignore = "GPU rate measurement; run via nvk.sh probe with --nocapture"]
fn producer_epilogue_cost_delta_bench() {
    let Some(ctx) = ctx_or_skip("w4a8_producer_bench") else {
        return;
    };
    let src = producer_src();
    let (hidden, inter) = (5120usize, 17408usize);
    let mut rng = Lcg::new(0x91);
    let x = rng.bf16_words(hidden, 1.0);
    let res = rng.bf16_words(hidden, 1.0);
    let w = rng.bf16_words(hidden, 0.5);
    let gate = rng.bf16_words(inter, 2.0);
    let upv = rng.bf16_words(inter, 1.0);

    let mk = |label: &str,
              source: &str,
              entry: &str,
              binds: &[(u32, &wgpu::Buffer)],
              grid: (u32, u32, u32)| {
        let p = dispatch::cached_compute_pipeline(ctx, label, source, entry).expect(label);
        let b = dispatch::bind_group(ctx, &p, binds);
        (p, b, grid)
    };

    let xb = dispatch::storage_from_slice(ctx, "b-x", &x);
    let rb = dispatch::storage_from_slice(ctx, "b-res", &res);
    let wb = dispatch::storage_from_slice(ctx, "b-w", &w);
    let ob = dispatch::storage_zeroed(ctx, "b-out", (hidden * 2) as u64);
    let qhb = dispatch::storage_zeroed(ctx, "b-xqh", hidden as u64);
    let shb = dispatch::storage_from_slice(ctx, "b-xsh", &[0f32]);
    let rp = dispatch::uniform_from(
        ctx,
        "b-rp",
        &RmsResParams {
            hidden: hidden as u32,
            batch: 1,
            eps: RMS_EPS,
            words_per_row: (hidden / 2) as u32,
        },
    );
    let pph = dispatch::uniform_from(ctx, "b-pph", &prod_params(hidden));
    let uph = dispatch::uniform_from(ctx, "b-uph", &g4::gemv_params(1.0, 1, hidden, 1));

    let gb = dispatch::storage_from_slice(ctx, "b-g", &gate);
    let ub = dispatch::storage_from_slice(ctx, "b-u", &upv);
    let ab = dispatch::storage_zeroed(ctx, "b-act", (inter * 2) as u64);
    let qib = dispatch::storage_zeroed(ctx, "b-xqi", inter as u64);
    let sib = dispatch::storage_from_slice(ctx, "b-xsi", &[0f32]);
    let sp = dispatch::uniform_from(
        ctx,
        "b-sp",
        &SiluParams {
            n_words: (inter / 2) as u32,
            ..Default::default()
        },
    );
    let ppi = dispatch::uniform_from(ctx, "b-ppi", &prod_params(inter));
    let upi = dispatch::uniform_from(ctx, "b-upi", &g4::gemv_params(1.0, 1, inter, 1));

    let rmsres_src = compose(RMSRES_WGSL);
    let w48_src = compose(W48_WGSL);
    let silu_src = compose(SILU_WGSL);

    let arms = vec![
        Arm {
            label: "rmsres plain".into(),
            passes: vec![mk(
                "b-rms",
                &rmsres_src,
                "rmsnorm_residual_bf16",
                &[(0, &xb), (1, &rb), (2, &wb), (3, &ob), (4, &rp)],
                (1, 1, 1),
            )],
        },
        Arm {
            label: "rmsres plain + w48 quant".into(),
            passes: vec![
                mk(
                    "b-rms",
                    &rmsres_src,
                    "rmsnorm_residual_bf16",
                    &[(0, &xb), (1, &rb), (2, &wb), (3, &ob), (4, &rp)],
                    (1, 1, 1),
                ),
                mk(
                    "b-qh",
                    &w48_src,
                    "w48_quantize_x_evenodd",
                    &[(6, &ob), (7, &qhb), (8, &shb), (4, &uph)],
                    quant_groups(hidden),
                ),
            ],
        },
        Arm {
            label: "rmsres+rowquant FUSED".into(),
            passes: vec![mk(
                "b-rmsf",
                &src,
                RMSRES_FUSED_ENTRY,
                &[
                    (0, &xb),
                    (1, &rb),
                    (2, &wb),
                    (3, &ob),
                    (4, &pph),
                    (5, &qhb),
                    (6, &shb),
                ],
                (1, 1, 1),
            )],
        },
        Arm {
            label: "silu_mul + w48 quant".into(),
            passes: vec![
                mk(
                    "b-silu",
                    &silu_src,
                    "q3w_silu_mul",
                    &[(10, &gb), (11, &ub), (12, &ab), (13, &sp)],
                    dispatch::workgroup_count_1d(ctx, (inter / 2) as u64, 64),
                ),
                mk(
                    "b-qi",
                    &w48_src,
                    "w48_quantize_x_evenodd",
                    &[(6, &ab), (7, &qib), (8, &sib), (4, &upi)],
                    quant_groups(inter),
                ),
            ],
        },
        Arm {
            label: "silu rowquant FUSED".into(),
            passes: vec![mk(
                "b-siluf",
                &src,
                SILU_FUSED_ENTRY,
                &[(4, &ppi), (5, &qib), (6, &sib), (7, &gb), (8, &ub)],
                (1, 1, 1),
            )],
        },
        Arm {
            label: "rmsres plain [NULL]".into(),
            passes: vec![mk(
                "b-rms",
                &rmsres_src,
                "rmsnorm_residual_bf16",
                &[(0, &xb), (1, &rb), (2, &wb), (3, &ob), (4, &rp)],
                (1, 1, 1),
            )],
        },
    ];

    for arm in &arms {
        for _ in 0..2 {
            time_arm(ctx, arm, PRODUCER_BENCH_ITERS);
        }
    }
    let mut rounds: Vec<Vec<f64>> = vec![Vec::new(); arms.len()];
    for _ in 0..PRODUCER_BENCH_ROUNDS {
        for (i, arm) in arms.iter().enumerate() {
            rounds[i].push(time_arm(ctx, arm, PRODUCER_BENCH_ITERS));
        }
    }
    eprintln!(
        "--- producer epilogue delta hidden={hidden} inter={inter} {} iters x {} rounds ---",
        PRODUCER_BENCH_ITERS, PRODUCER_BENCH_ROUNDS
    );
    for (i, arm) in arms.iter().enumerate() {
        let us: Vec<f64> = rounds[i]
            .iter()
            .map(|t| t * 1e6 / PRODUCER_BENCH_ITERS as f64)
            .collect();
        eprintln!("{:<28} {:>8.3} us/dispatch-set med", arm.label, med(&us));
    }
}
