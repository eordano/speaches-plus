#![cfg(feature = "wgpu")]

mod common;

use common::ctx_or_skip;
use common::LcgShift32TwoSided as Lcg;
use nv_kernels::wgpu_backend::device::WgpuContext;
use nv_kernels::wgpu_backend::kernels::gdn_recurrent as gr;
use nv_kernels::wgpu_backend::{compose, dispatch};

const SERVING_SHAPE_Q38_WGPU_B1_H48_D128: (usize, usize) = (1, 48);

const BITWISE_SHAPES_B_T_H_EDGE_TS_CROSS_PAIR_FOLDS_AND_HEAD_COUNTS: &[(usize, usize, usize)] = &[
    (1, 1, 48),
    (1, 5, 48),
    (2, 33, 32),
    (1, 64, 16),
    (1, 511, 48),
    (1, 2048, 48),
];

struct ScanInputs {
    q: Vec<f32>,
    k: Vec<f32>,
    v: Vec<f32>,
    g: Vec<f32>,
    beta: Vec<f32>,
}

fn unit(l: &mut Lcg) -> f32 {
    (l.next_f32() + 1.0) * 0.5
}

fn scan_inputs(b: usize, t: usize, h: usize, seed: u64) -> ScanInputs {
    let mut l = Lcg(seed);
    let rows = b * t * h;
    let vecs = rows * gr::HEAD_DIM;
    ScanInputs {
        q: l.f32_vec(vecs, 1.0),
        k: l.f32_vec(vecs, 1.0),
        v: l.f32_vec(vecs, 1.0),
        g: (0..rows).map(|_| 0.62 + unit(&mut l) * 0.379).collect(),
        beta: (0..rows).map(|_| 0.05 + unit(&mut l) * 0.9).collect(),
    }
}

fn run_arm(
    ctx: &WgpuContext,
    x: &ScanInputs,
    b: usize,
    t: usize,
    h: usize,
    arm: gr::ScanArm,
) -> (Vec<f32>, Vec<f32>) {
    let mut out = vec![0f32; b * t * h * gr::HEAD_DIM];
    let mut state = vec![0f32; b * h * gr::HEAD_DIM * gr::HEAD_DIM];
    gr::gdn_recurrent_f32_with_arm(
        ctx, &x.q, &x.k, &x.v, &x.g, &x.beta, &mut out, &mut state, b, h, t, arm,
    )
    .unwrap_or_else(|e| panic!("arm {arm:?} b={b} t={t} h={h}: {e}"));
    (out, state)
}

fn f32_bits(v: &[f32]) -> Vec<u32> {
    v.iter().map(|x| x.to_bits()).collect()
}

#[test]
fn w32_scan_is_bitwise_equal_to_the_barrier_kernel_on_out_and_state() {
    let ctx = match ctx_or_skip("gdn_scan_w32_bitwise") {
        Some(c) => c,
        None => return,
    };
    for &(b, t, h) in BITWISE_SHAPES_B_T_H_EDGE_TS_CROSS_PAIR_FOLDS_AND_HEAD_COUNTS {
        let x = scan_inputs(b, t, h, 0x5ca9_0000 + (t * h + b) as u64);
        let (out_v1, st_v1) = run_arm(ctx, &x, b, t, h, gr::ScanArm::Barrier128);
        let (out_w, st_w) = run_arm(ctx, &x, b, t, h, gr::ScanArm::Warp32Smem);
        assert_eq!(
            f32_bits(&out_v1),
            f32_bits(&out_w),
            "w32 out bits b={b} t={t} h={h}"
        );
        assert_eq!(
            f32_bits(&st_v1),
            f32_bits(&st_w),
            "w32 state bits b={b} t={t} h={h}"
        );
        eprintln!("w32 scan bitwise ok b={b} t={t} h={h}");
    }
}

#[test]
fn the_default_route_stays_the_barrier_kernel_unless_the_env_opts_in() {
    assert_eq!(
        gr::scan_arm(),
        if gr::scan_w32_env_opted_in() {
            gr::ScanArm::Warp32Smem
        } else {
            gr::ScanArm::Barrier128
        }
    );
    if std::env::var_os(gr::SCAN_W32_ENV_OPT_IN_THE_BARRIER_KERNEL_STAYS_THE_DEFAULT).is_none() {
        assert_eq!(gr::scan_arm(), gr::ScanArm::Barrier128);
    }
}

fn bench_gated() -> bool {
    std::env::var("NV_GDN_SCANW32_BENCH").as_deref() == Ok("1")
}

struct ResidentBufs {
    qb: wgpu::Buffer,
    kb: wgpu::Buffer,
    vb: wgpu::Buffer,
    gb: wgpu::Buffer,
    bb: wgpu::Buffer,
    ob: wgpu::Buffer,
    sb: wgpu::Buffer,
    pb: wgpu::Buffer,
}

fn resident_bufs(ctx: &WgpuContext, x: &ScanInputs, b: usize, t: usize, h: usize) -> ResidentBufs {
    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    struct P {
        batch: u32,
        seq: u32,
        heads: u32,
        pairs: u32,
    }
    let vecs = b * t * h * gr::HEAD_DIM;
    let state_len = b * h * gr::HEAD_DIM * gr::HEAD_DIM;
    ResidentBufs {
        qb: dispatch::storage_from_slice(ctx, "w32scan-q", &x.q),
        kb: dispatch::storage_from_slice(ctx, "w32scan-k", &x.k),
        vb: dispatch::storage_from_slice(ctx, "w32scan-v", &x.v),
        gb: dispatch::storage_from_slice(ctx, "w32scan-g", &x.g),
        bb: dispatch::storage_from_slice(ctx, "w32scan-beta", &x.beta),
        ob: dispatch::storage_zeroed(ctx, "w32scan-out", (vecs * 4) as u64),
        sb: dispatch::storage_zeroed(ctx, "w32scan-state", (state_len * 4) as u64),
        pb: dispatch::uniform_from(
            ctx,
            "w32scan-params",
            &P {
                batch: b as u32,
                seq: t as u32,
                heads: h as u32,
                pairs: (b * h) as u32,
            },
        ),
    }
}

fn dispatch_once(ctx: &WgpuContext, bufs: &ResidentBufs, arm: gr::ScanArm, pairs: u64) {
    let pair_groups = dispatch::workgroup_count_1d(ctx, pairs, 1);
    let (label, entry, groups) = match arm {
        gr::ScanArm::Barrier128 => (
            "nv_kernels_gdn_recurrent_f32",
            "gdn_recurrent_f32",
            pair_groups,
        ),
        gr::ScanArm::Warp32Smem => (
            "nv_kernels_gdn_recurrent_f32_w32",
            gr::W32_ENTRY,
            (
                pair_groups.0,
                pair_groups.1,
                gr::W32_COL_GROUPS_WG_Z_SPANS_THE_128_COLUMNS,
            ),
        ),
    };
    dispatch::run(
        ctx,
        label,
        &compose(gr::WGSL),
        entry,
        &[
            (0, &bufs.qb),
            (1, &bufs.kb),
            (2, &bufs.vb),
            (3, &bufs.gb),
            (4, &bufs.bb),
            (5, &bufs.ob),
            (6, &bufs.sb),
            (7, &bufs.pb),
        ],
        groups,
    )
    .unwrap();
}

fn wall_ms_per_dispatch(
    ctx: &WgpuContext,
    bufs: &ResidentBufs,
    arm: gr::ScanArm,
    pairs: u64,
    warmup: usize,
    reps: usize,
) -> f64 {
    for _ in 0..warmup {
        dispatch_once(ctx, bufs, arm, pairs);
    }
    ctx.poll_blocking().unwrap();
    let t0 = std::time::Instant::now();
    for _ in 0..reps {
        dispatch_once(ctx, bufs, arm, pairs);
    }
    ctx.poll_blocking().unwrap();
    t0.elapsed().as_secs_f64() * 1000.0 / reps as f64
}

#[test]
#[ignore = "measurement ladder, resident buffers, wall over reps; set NV_GDN_SCANW32_BENCH=1 -- prices the wgpu gdn library scan at the q38 serving pair count"]
fn gdn_wgpu_scan_w32_microbench_at_q38_serving_pairs() {
    if !bench_gated() {
        panic!(
            "NV_GDN_SCANW32_BENCH is not 1, so this microbench executed nothing. It is \
             #[ignore]d, which makes --ignored the opt-in; a suite whose precondition is \
             absent must fail or be ignored, never return green."
        );
    }
    let ctx = match ctx_or_skip("gdn_scan_w32_bench") {
        Some(c) => c,
        None => return,
    };
    let (b, h) = SERVING_SHAPE_Q38_WGPU_B1_H48_D128;
    for (t, warmup, reps) in [(512usize, 5usize, 30usize), (4096, 3, 10)] {
        let x = scan_inputs(b, t, h, 0xbe9c_0001);
        let bufs = resident_bufs(ctx, &x, b, t, h);
        let pairs = (b * h) as u64;
        let v1 = wall_ms_per_dispatch(ctx, &bufs, gr::ScanArm::Barrier128, pairs, warmup, reps);
        let w32 = wall_ms_per_dispatch(ctx, &bufs, gr::ScanArm::Warp32Smem, pairs, warmup, reps);
        let v1b = wall_ms_per_dispatch(ctx, &bufs, gr::ScanArm::Barrier128, pairs, warmup, reps);
        let null = v1 / v1b;
        eprintln!(
            "w32scan-bench t={t} pairs={pairs} v1={v1:.3}ms ({:.0} ns/tok) w32={w32:.3}ms \
             ({:.0} ns/tok, {:.2}x) null_v1_rerun={null:.2}x",
            v1 * 1e6 / t as f64,
            w32 * 1e6 / t as f64,
            v1 / w32,
        );
    }
}
