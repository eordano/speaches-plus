#![cfg(feature = "wgpu")]

mod common;
use common::ctx;
use common::FdParams;
use nv_kernels::wgpu_backend::compose;
use nv_kernels::wgpu_backend::device::WgpuContext;
use nv_kernels::wgpu_backend::dispatch;
use nv_kernels::wgpu_backend::kernels::flash_decode as fd;

struct Lcg(u64);

impl Lcg {
    fn next_f32(&mut self) -> f32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((self.0 >> 40) as f32 / 8_388_608.0) - 1.0
    }
    fn bf16_words(&mut self, len: usize) -> Vec<u32> {
        (0..len / 2)
            .map(|_| {
                let a = half::bf16::from_f32(self.next_f32()).to_bits() as u32;
                let b = half::bf16::from_f32(self.next_f32()).to_bits() as u32;
                a | (b << 16)
            })
            .collect()
    }
    fn fp8_words(&mut self, len: usize) -> Vec<u32> {
        (0..len / 4)
            .map(|_| {
                let mut w = 0u32;
                for i in 0..4 {
                    let b = nv_kernels::wgpu_backend::kernels::kv_fp8::encode_e4m3(self.next_f32());
                    w |= (b as u32) << (8 * i);
                }
                w
            })
            .collect()
    }
}

struct Shape {
    label: &'static str,
    n_heads: usize,
    n_kv: usize,
    hd: usize,
    total: usize,
    splits: usize,
}

const SHAPES: &[Shape] = &[
    Shape {
        label: "hd128 gqa32/8",
        n_heads: 32,
        n_kv: 8,
        hd: 128,
        total: 2048,
        splits: 16,
    },
    Shape {
        label: "hd256 gqa32/8",
        n_heads: 32,
        n_kv: 8,
        hd: 256,
        total: 2048,
        splits: 16,
    },
    Shape {
        label: "hd128 total256",
        n_heads: 32,
        n_kv: 8,
        hd: 128,
        total: 256,
        splits: 16,
    },
];

struct Rig {
    pipeline: wgpu::ComputePipeline,
    group: wgpu::BindGroup,
    scratch: wgpu::Buffer,
}

fn bench(ctx: &WgpuContext, rig: &Rig, grid: (u32, u32, u32), iters: usize) -> f64 {
    let submit = |count: usize| {
        let mut enc = ctx.device.create_command_encoder(&Default::default());
        {
            let mut pass = enc.begin_compute_pass(&Default::default());
            pass.set_pipeline(&rig.pipeline);
            pass.set_bind_group(0, &rig.group, &[]);
            for _ in 0..count {
                pass.dispatch_workgroups(grid.0, grid.1, grid.2);
            }
        }
        ctx.queue.submit([enc.finish()]);
    };
    submit(4);
    ctx.poll_blocking().expect("warmup poll");
    let start = std::time::Instant::now();
    submit(iters);
    ctx.poll_blocking().expect("timed poll");
    start.elapsed().as_secs_f64() * 1e6 / iters as f64
}

fn run_shape(
    ctx: &WgpuContext,
    s: &Shape,
    m: usize,
    fp8: bool,
    reps: usize,
) -> (f64, f64, f64, usize) {
    let mut rng = Lcg(0x5eed ^ (s.hd as u64) << 16 ^ (m as u64) << 8 ^ fp8 as u64);
    let q: Vec<f32> = (0..m * s.n_heads * s.hd).map(|_| rng.next_f32()).collect();
    let kv_elems = s.total * s.n_kv * s.hd;
    let (kw, vw) = if fp8 {
        (rng.fp8_words(kv_elems), rng.fp8_words(kv_elems))
    } else {
        (rng.bf16_words(kv_elems), rng.bf16_words(kv_elems))
    };
    let scales: Vec<f32> = (0..s.total * s.n_kv)
        .map(|_| 0.5 + 0.5 * (rng.next_f32() + 1.0))
        .collect();
    let params = FdParams {
        n_heads: s.n_heads as u32,
        n_kv: s.n_kv as u32,
        head_dim: s.hd as u32,
        total: s.total as u32,
        start: 0,
        splits: s.splits as u32,
        ring: 0,
        out_bf16: 1,
        scaling: 1.0 / (s.hd as f32).sqrt(),
        pad0: 0,
        fused: 1,
        pad2: 0,
        m_rows: m as u32,
        window: 0,
        pad3: 0,
        pad4: 0,
    };
    let scratch_elems = s.n_heads * m * s.splits * (s.hd + 2);

    let src = compose(fd::WGSL);
    let qb = dispatch::storage_from_slice(ctx, "s1-q", &q);
    let kb = dispatch::storage_from_slice(ctx, "s1-k", &kw);
    let vb = dispatch::storage_from_slice(ctx, "s1-v", &vw);
    let ksb = dispatch::storage_from_slice(ctx, "s1-ks", &scales);
    let vsb = dispatch::storage_from_slice(ctx, "s1-vs", &scales);
    let pb = dispatch::uniform_from(ctx, "s1-p", &params);

    let mk = |label: &str, entry: &str| {
        let pipeline =
            dispatch::compute_pipeline_opts(ctx, label, &src, entry, true).expect("pipeline");
        let scratch = dispatch::storage_zeroed(ctx, "s1-scratch", (scratch_elems * 4) as u64);
        let mut binds: Vec<(u32, &wgpu::Buffer)> =
            vec![(0, &qb), (4, &pb), (5, &kb), (6, &vb), (7, &scratch)];
        if fp8 {
            binds.push((8, &ksb));
            binds.push((9, &vsb));
        }
        let group = dispatch::bind_group(ctx, &pipeline, &binds);
        Rig {
            pipeline,
            group,
            scratch,
        }
    };
    let (rolled_entry, unrolled_entry) = if fp8 {
        (fd::ENTRY_STAGE1_FP8_MK, fd::ENTRY_STAGE1_FP8_MK_U)
    } else {
        (fd::ENTRY_STAGE1_BF16_MK, fd::ENTRY_STAGE1_BF16_MK_U)
    };
    let rolled = mk("s1-rolled", rolled_entry);
    let unrolled = mk("s1-unrolled", unrolled_entry);

    let null = mk("s1-null", rolled_entry);

    let grid = (s.n_heads as u32, s.splits as u32, 1);
    let iters = 200;
    let mut best = [f64::MAX; 3];
    for _ in 0..reps.max(1) {
        for (i, rig) in [&rolled, &unrolled, &null].iter().enumerate() {
            let us = bench(ctx, rig, grid, if reps == 0 { 1 } else { iters });
            if us < best[i] {
                best[i] = us;
            }
        }
    }
    let a: Vec<f32> = dispatch::read_back(ctx, &rolled.scratch, scratch_elems).expect("read a");
    let b: Vec<f32> = dispatch::read_back(ctx, &unrolled.scratch, scratch_elems).expect("read b");
    let nonzero = a.iter().filter(|v| **v != 0.0 && v.is_finite()).count();
    assert!(
        nonzero * 4 > scratch_elems,
        "{}: stage1 left {nonzero}/{scratch_elems} live scratch words -- the fixture measured nothing",
        s.label
    );
    let diff = a
        .iter()
        .zip(b.iter())
        .filter(|(x, y)| x.to_bits() != y.to_bits())
        .count();
    (best[0], best[1], best[2], diff)
}

fn unroll_ab(ctx: &WgpuContext, fp8: bool) {
    let mut cells = 0;
    for s in SHAPES {
        for m in [1usize, 2, 4, 8] {
            let (rolled, unrolled, null, diff) = run_shape(ctx, s, m, fp8, 7);
            eprintln!(
                "stage1 mk {:<4} {:<14} m={m} splits={:<3} | row-inner acc[mr][hd/32] {:>9.3} us | \
                 row-outer scalars {:>9.3} us | {:.4}x | null control {:>9.3} us ({:+.3} us, {:.2}%) | \
                 bit-exact ({diff} diffs)",
                if fp8 { "fp8" } else { "bf16" },
                s.label,
                s.splits,
                rolled,
                unrolled,
                rolled / unrolled,
                null,
                null - rolled,
                (null - rolled).abs() / rolled * 100.0,
            );
            assert_eq!(
                diff, 0,
                "{} m={m} fp8={fp8}: the row-outer loop changed {diff} scratch words",
                s.label
            );
            cells += 1;
        }
    }
    assert_eq!(cells, SHAPES.len() * 4, "ran {cells} cells");
}

#[test]
#[ignore = "kernel-rate suite: run alone, one per process"]
fn stage1_mk_bf16_row_outer_is_bit_exact_and_timed_against_the_row_inner_accumulator() {
    let Some(ctx) = ctx("stage1_mk_unroll_bf16") else {
        return;
    };
    unroll_ab(ctx, false);
}

#[test]
#[ignore = "kernel-rate suite: run alone, one per process"]
fn stage1_mk_fp8_row_outer_is_bit_exact_and_timed_against_the_row_inner_accumulator() {
    let Some(ctx) = ctx("stage1_mk_unroll_fp8") else {
        return;
    };
    unroll_ab(ctx, true);
}

#[test]
fn stage1_mk_row_outer_is_bit_exact_at_every_row_count() {
    let Some(ctx) = ctx("stage1_mk_unroll_exact") else {
        return;
    };
    let s = &SHAPES[0];
    for fp8 in [false, true] {
        for m in 1..=8usize {
            let (_, _, _, diff) = run_shape(ctx, s, m, fp8, 0);
            assert_eq!(
                diff, 0,
                "fp8={fp8} m={m}: the row-outer loop changed {diff} scratch words"
            );
        }
    }
}
