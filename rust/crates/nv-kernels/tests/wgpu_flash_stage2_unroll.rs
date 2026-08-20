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
}

fn build_scratch(
    ctx: &WgpuContext,
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    total: usize,
    splits: usize,
    seed: u64,
) -> Vec<f32> {
    let mut rng = Lcg(seed);
    let q: Vec<f32> = (0..n_heads * head_dim).map(|_| rng.next_f32()).collect();
    let kv_len = total * n_kv_heads * head_dim;
    let k: Vec<u16> = (0..kv_len)
        .map(|_| half::bf16::from_f32(rng.next_f32()).to_bits())
        .collect();
    let v: Vec<u16> = (0..kv_len)
        .map(|_| half::bf16::from_f32(rng.next_f32()).to_bits())
        .collect();
    let mut out = vec![0u16; n_heads * head_dim];
    let mut scratch = vec![0f32; n_heads * splits * (head_dim + 2)];
    fd::flash_decode_splitk_bf16kv(
        ctx,
        &q,
        &k,
        &v,
        &mut out,
        &mut scratch,
        &[total as i32],
        n_heads,
        n_kv_heads,
        head_dim,
        0,
        1.0 / (head_dim as f32).sqrt(),
        splits,
        0,
    )
    .expect("stage1 fill");
    scratch
}

fn build_scratch_mk(
    ctx: &WgpuContext,
    m: usize,
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    total: usize,
    splits: usize,
    seed: u64,
) -> Vec<f32> {
    let mut rng = Lcg(seed);
    let q: Vec<f32> = (0..m * n_heads * head_dim)
        .map(|_| rng.next_f32())
        .collect();
    let kv_len = total * n_kv_heads * head_dim;
    let k: Vec<u16> = (0..kv_len)
        .map(|_| half::bf16::from_f32(rng.next_f32()).to_bits())
        .collect();
    let v: Vec<u16> = (0..kv_len)
        .map(|_| half::bf16::from_f32(rng.next_f32()).to_bits())
        .collect();
    let mut out = vec![0u16; m * n_heads * head_dim];
    let mut scratch = vec![0f32; n_heads * m * splits * (head_dim + 2)];
    fd::flash_decode_fused_bf16kv_mk(
        ctx,
        &q,
        &k,
        &v,
        &mut out,
        &mut scratch,
        &[total as i32],
        0,
        m,
        n_heads,
        n_kv_heads,
        head_dim,
        0,
        splits,
    )
    .expect("mk stage1 fill");
    scratch
}

fn run_stage2(
    ctx: &WgpuContext,
    entry: &str,
    scratch: &[f32],
    params: &FdParams,
    n_heads: usize,
    head_dim: usize,
    warmup: usize,
    iters: usize,
) -> (Vec<u32>, f64) {
    let src = compose(fd::WGSL);
    let pipeline = dispatch::cached_compute_pipeline(ctx, entry, &src, entry).expect("pipeline");
    let sb = dispatch::storage_from_slice(ctx, "fd2-scratch", scratch);
    let out_elems = n_heads * head_dim * params.m_rows.max(1) as usize;
    let ob = dispatch::storage_zeroed(ctx, "fd2-out", (out_elems * 4) as u64);
    let pb = dispatch::uniform_from(ctx, "fd2-params", params);
    let group = dispatch::bind_group(ctx, &pipeline, &[(3, &ob), (4, &pb), (7, &sb)]);
    let grid_y = params.m_rows.max(1);

    let submit = |count: usize| {
        let mut enc = ctx.device.create_command_encoder(&Default::default());
        {
            let mut pass = enc.begin_compute_pass(&Default::default());
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &group, &[]);
            for _ in 0..count {
                pass.dispatch_workgroups(n_heads as u32, grid_y, 1);
            }
        }
        ctx.queue.submit([enc.finish()]);
    };
    submit(warmup.max(1));
    ctx.poll_blocking().expect("warmup poll");
    let start = std::time::Instant::now();
    submit(iters);
    ctx.poll_blocking().expect("timed poll");
    let secs = start.elapsed().as_secs_f64();
    let words: Vec<u32> = dispatch::read_back(ctx, &ob, out_elems).expect("stage2 read back");
    (words, secs)
}

#[test]
fn stage2_unrolled_is_bit_exact_and_timed_against_the_ssc_array() {
    let Some(ctx) = ctx("flash_stage2_unroll") else {
        return;
    };
    let head_dim = 128usize;
    let n_heads = 32usize;
    let n_kv_heads = 8usize;
    let mut cells = 0;
    for total in [512usize, 4096] {
        for splits in [8usize, 16, 32] {
            let scratch = build_scratch(
                ctx,
                n_heads,
                n_kv_heads,
                head_dim,
                total,
                splits,
                0xf1a5 ^ total as u64,
            );
            let nonzero = scratch.iter().filter(|v| **v != 0.0).count();
            assert!(
                nonzero > scratch.len() / 4,
                "stage1 produced a near-empty scratch ({nonzero}/{}) -- the fixture measured nothing",
                scratch.len()
            );
            let params = FdParams {
                n_heads: n_heads as u32,
                n_kv: n_kv_heads as u32,
                head_dim: head_dim as u32,
                total: total as u32,
                start: 0,
                splits: splits as u32,
                ring: 0,
                out_bf16: 1,
                scaling: 1.0 / (head_dim as f32).sqrt(),
                pad0: 0,
                fused: 0,
                pad2: 0,
                m_rows: 1,
                window: 0,
                pad3: 0,
                pad4: 0,
            };
            let iters = 2000;
            let mut best = [f64::MAX; 2];
            let mut outs: [Vec<u32>; 2] = [Vec::new(), Vec::new()];
            for rep in 0..3 {
                for (i, entry) in [fd::ENTRY_STAGE2, fd::ENTRY_STAGE2_U].iter().enumerate() {
                    let (o, secs) =
                        run_stage2(ctx, entry, &scratch, &params, n_heads, head_dim, 50, iters);
                    if secs < best[i] {
                        best[i] = secs;
                    }
                    if rep == 0 {
                        outs[i] = o;
                    }
                }
            }
            let diff = outs[0]
                .iter()
                .zip(outs[1].iter())
                .filter(|(a, b)| a != b)
                .count();
            assert_eq!(
                diff,
                0,
                "splits={splits} total={total}: stage2_u differs from stage2 in {diff}/{} elements",
                outs[0].len()
            );
            let us0 = best[0] * 1e6 / iters as f64;
            let us1 = best[1] * 1e6 / iters as f64;
            eprintln!(
                "flash stage2 total={total:<5} splits={splits:<3} | ssc[32] array {us0:>8.3} us | scalar recompute {us1:>8.3} us | {:.4}x | bit-exact ({} elems)",
                us0 / us1,
                outs[0].len()
            );
            cells += 1;
        }
    }
    assert_eq!(cells, 6);
}

#[test]
fn stage2_mk_unrolled_is_bit_exact_and_timed_against_the_ssc_array() {
    let Some(ctx) = ctx("flash_stage2_mk_unroll") else {
        return;
    };
    let head_dim = 128usize;
    let n_heads = 32usize;
    let n_kv_heads = 8usize;
    let total = 2048usize;
    let mut cells = 0;
    for m in [4usize, 8] {
        for splits in [8usize, 16, 32] {
            let scratch = build_scratch_mk(
                ctx,
                m,
                n_heads,
                n_kv_heads,
                head_dim,
                total,
                splits,
                0x2b1d ^ (m as u64) << 8 ^ splits as u64,
            );
            let nonzero = scratch.iter().filter(|v| **v != 0.0).count();
            assert!(
                nonzero > scratch.len() / 4,
                "mk stage1 produced a near-empty scratch ({nonzero}/{}) -- the fixture measured nothing",
                scratch.len()
            );
            let params = FdParams {
                n_heads: n_heads as u32,
                n_kv: n_kv_heads as u32,
                head_dim: head_dim as u32,
                total: total as u32,
                start: 0,
                splits: splits as u32,
                ring: 0,
                out_bf16: 1,
                scaling: 1.0 / (head_dim as f32).sqrt(),
                pad0: 0,
                fused: 1,
                pad2: 0,
                m_rows: m as u32,
                window: 0,
                pad3: 0,
                pad4: 0,
            };
            let iters = 1000;
            let mut best = [f64::MAX; 2];
            let mut outs: [Vec<u32>; 2] = [Vec::new(), Vec::new()];
            for rep in 0..3 {
                for (i, entry) in [fd::ENTRY_STAGE2_MK, fd::ENTRY_STAGE2_MK_U]
                    .iter()
                    .enumerate()
                {
                    let (o, secs) =
                        run_stage2(ctx, entry, &scratch, &params, n_heads, head_dim, 50, iters);
                    if secs < best[i] {
                        best[i] = secs;
                    }
                    if rep == 0 {
                        outs[i] = o;
                    }
                }
            }
            let diff = outs[0]
                .iter()
                .zip(outs[1].iter())
                .filter(|(a, b)| a != b)
                .count();
            assert_eq!(
                diff,
                0,
                "m={m} splits={splits}: stage2_mk_u differs from stage2_mk in {diff}/{} elements",
                outs[0].len()
            );
            let us0 = best[0] * 1e6 / iters as f64;
            let us1 = best[1] * 1e6 / iters as f64;
            eprintln!(
                "flash stage2_mk m={m:<2} splits={splits:<3} | ssc[32] array {us0:>8.3} us | scalar recompute {us1:>8.3} us | {:.4}x | bit-exact ({} elems)",
                us0 / us1,
                outs[0].len()
            );
            cells += 1;
        }
    }
    assert_eq!(cells, 6);
}
