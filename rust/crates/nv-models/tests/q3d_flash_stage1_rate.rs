#![cfg(feature = "wgpu")]

mod common;
use common::FdP;
use common::pack_u8;
use nv_kernels::wgpu_backend::device::WgpuContext;
use nv_kernels::wgpu_backend::kernels::flash_decode as fd;
use nv_kernels::wgpu_backend::kernels::kv_fp8;
use nv_kernels::wgpu_backend::{compose, dispatch};
use std::time::Instant;

const Q3D_NQ: u32 = 24;
const Q3D_NKV: u32 = 4;
const Q3D_HD: u32 = 256;
const Q3D_FOLD: u32 = 6;

fn ctx() -> &'static WgpuContext {
    let c = WgpuContext::shared().expect("wgpu adapter required for --features wgpu");
    assert!(
        c.qualify().qualified,
        "adapter not qualified: {:?}",
        c.qualify().reason
    );
    assert_eq!(
        c.subgroup_width(),
        Some(32),
        "q3d fold arms assume 32-lane subgroups; a different width would build the wrong butterfly"
    );
    c
}

#[derive(Clone, Copy, Debug)]
struct Shape {
    slots: u32,
    total: u32,
    splits: u32,
}

impl Shape {
    fn params(&self) -> FdP {
        FdP {
            n_heads: Q3D_NQ,
            n_kv: Q3D_NKV,
            head_dim: Q3D_HD,
            total: self.total,
            start: 0,
            splits: self.splits,
            ring: 0,
            out_bf16: 1,
            scaling: 1.0 / (Q3D_HD as f32).sqrt(),
            m_rows: 1,
            ..Default::default()
        }
    }

    fn scratch_elems(&self) -> usize {
        (Q3D_NQ * self.splits * (Q3D_HD + 2)) as usize
    }
}

fn transpose_k_words(k: &[u8], slots: usize) -> Vec<u32> {
    let nkv = Q3D_NKV as usize;
    let hd = Q3D_HD as usize;
    assert_eq!(k.len(), slots * nkv * hd);
    let mut out = vec![0u32; slots * nkv * hd / 4];
    for pos in 0..slots {
        for kvh in 0..nkv {
            let base = (pos * nkv + kvh) * hd;
            for d4 in 0..hd / 4 {
                let b = &k[base + d4 * 4..base + d4 * 4 + 4];
                let w = u32::from_le_bytes([b[0], b[1], b[2], b[3]]);
                out[fd::k_transposed_word_index(slots, nkv, hd, kvh, pos, d4)] = w;
            }
        }
    }
    out
}

struct Bufs {
    k: Vec<wgpu::Buffer>,
    kt: Vec<wgpu::Buffer>,
    v: Vec<wgpu::Buffer>,
    ks: Vec<wgpu::Buffer>,
    vs: Vec<wgpu::Buffer>,
    scratch: Vec<wgpu::Buffer>,
    q: wgpu::Buffer,
    p: wgpu::Buffer,
}

fn make_bufs_seed(ctx: &WgpuContext, s: &Shape, n: usize, seed: u64) -> Bufs {
    let kv_elems = (s.slots * Q3D_NKV * Q3D_HD) as usize;
    let sc_elems = (s.slots * Q3D_NKV) as usize;
    let mut lcg: u64 = seed | 1;
    let mut next = move || {
        lcg = lcg.wrapping_mul(6364136223846793005).wrapping_add(1);
        lcg
    };
    let byte = |n: &mut dyn FnMut() -> u64| 0x30u8 + ((n() >> 33) as u8 % 0x18);
    let mut k = Vec::new();
    let mut kt = Vec::new();
    let mut v = Vec::new();
    let mut ks = Vec::new();
    let mut vs = Vec::new();
    let mut scratch = Vec::new();
    for _ in 0..n {
        let kb: Vec<u8> = (0..kv_elems).map(|_| byte(&mut next)).collect();
        let vb: Vec<u8> = (0..kv_elems).map(|_| byte(&mut next)).collect();
        let sc: Vec<f32> = (0..sc_elems)
            .map(|j| 0.004 + ((next() >> 45) as f32 / 524_288.0) * 0.02 + (j % 7) as f32 * 0.003)
            .collect();
        k.push(dispatch::storage_from_slice(ctx, "q3k", &pack_u8(&kb)));
        kt.push(dispatch::storage_from_slice(
            ctx,
            "q3kt",
            &transpose_k_words(&kb, s.slots as usize),
        ));
        v.push(dispatch::storage_from_slice(ctx, "q3v", &pack_u8(&vb)));
        ks.push(dispatch::storage_from_slice(ctx, "q3ks", &sc));
        vs.push(dispatch::storage_from_slice(ctx, "q3vs", &sc));
        scratch.push(dispatch::storage_zeroed(
            ctx,
            "q3sc",
            (s.scratch_elems() * 4) as u64,
        ));
    }
    let q: Vec<f32> = (0..(Q3D_NQ * Q3D_HD) as usize)
        .map(|j| ((next() >> 40) as f32 / 8_388_608.0) - 1.0 + (j % 3) as f32 * 0.25)
        .collect();
    Bufs {
        k,
        kt,
        v,
        ks,
        vs,
        scratch,
        q: dispatch::storage_from_slice(ctx, "q3q", &q),
        p: dispatch::uniform_from(ctx, "q3p", &s.params()),
    }
}

#[derive(Clone, Copy, PartialEq)]
enum KSide {
    PositionMajor,
    Transposed,
}

struct Arm {
    name: &'static str,
    pl: wgpu::ComputePipeline,
    k_side: KSide,
    scratch_comparable_to_sg: bool,
}

fn arms(ctx: &'static WgpuContext) -> Vec<Arm> {
    let stock = compose(fd::WGSL);
    let mut out = Vec::new();
    for (name, body, entry, k_side, cmp) in [
        (
            "fold6 sd sg (shipping)",
            fd::fold_stage1_source_sd(Q3D_HD, true, Q3D_FOLD),
            fd::fold_stage1_entry_sd(Q3D_HD, true, Q3D_FOLD),
            KSide::PositionMajor,
            true,
        ),
        (
            "fold6 sd ra subgroupAdd",
            fd::fold_stage1_source_sd_ra(Q3D_HD, Q3D_FOLD),
            fd::fold_stage1_entry_sd_ra(Q3D_HD, Q3D_FOLD),
            KSide::PositionMajor,
            true,
        ),
        (
            "fold6 sd tp thread/pos",
            fd::fold_stage1_source_sd_tp(Q3D_HD, Q3D_FOLD),
            fd::fold_stage1_entry_sd_tp(Q3D_HD, Q3D_FOLD),
            KSide::Transposed,
            false,
        ),
    ] {
        let src = format!("{stock}\n{body}");
        let pl = dispatch::compute_pipeline_opts(ctx, &entry, &src, &entry, true)
            .unwrap_or_else(|e| panic!("{entry}: {e}"));
        out.push(Arm {
            name,
            pl,
            k_side,
            scratch_comparable_to_sg: cmp,
        });
    }
    out
}

fn stage2_pipeline(ctx: &WgpuContext) -> wgpu::ComputePipeline {
    let src = compose(fd::WGSL);
    dispatch::compute_pipeline_opts(ctx, "q3-stage2", &src, fd::ENTRY_STAGE2, true)
        .expect("stage2 pipeline")
}

fn stage1_bindings<'a>(arm: &Arm, b: &'a Bufs, i: usize) -> Vec<(u32, &'a wgpu::Buffer)> {
    let kb = match arm.k_side {
        KSide::PositionMajor => &b.k[i],
        KSide::Transposed => &b.kt[i],
    };
    vec![
        (0, &b.q),
        (4, &b.p),
        (5, kb),
        (6, &b.v[i]),
        (7, &b.scratch[i]),
        (8, &b.ks[i]),
        (9, &b.vs[i]),
    ]
}

fn scratch_of(ctx: &WgpuContext, arm: &Arm, b: &Bufs, s: &Shape) -> Vec<u32> {
    dispatch::dispatch(
        ctx,
        &arm.pl,
        &stage1_bindings(arm, b, 0),
        (Q3D_NQ / Q3D_FOLD, s.splits, 1),
    )
    .expect("stage1 dispatch");
    dispatch::read_back::<u32>(ctx, &b.scratch[0], s.scratch_elems()).expect("scratch readback")
}

fn out_of(ctx: &WgpuContext, arm: &Arm, s2: &wgpu::ComputePipeline, b: &Bufs, s: &Shape) -> Vec<u32> {
    dispatch::dispatch(
        ctx,
        &arm.pl,
        &stage1_bindings(arm, b, 0),
        (Q3D_NQ / Q3D_FOLD, s.splits, 1),
    )
    .expect("stage1 dispatch");
    let ob = dispatch::storage_zeroed(ctx, "q3out", (Q3D_NQ * Q3D_HD * 4) as u64);
    dispatch::dispatch(
        ctx,
        s2,
        &[(3, &ob), (4, &b.p), (7, &b.scratch[0])],
        (Q3D_NQ, 1, 1),
    )
    .expect("stage2 dispatch");
    dispatch::read_back::<u32>(ctx, &ob, (Q3D_NQ * Q3D_HD) as usize).expect("out readback")
}

const REDUCE_REASSOC_UNDER_CANCELLATION_CEILING: f32 = 1e-2;

fn assert_close_f32(name: &str, tag: &str, got: &[u32], want: &[u32], rel: f32) -> usize {
    assert_eq!(got.len(), want.len());
    let mut exact = 0usize;
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        if g == w {
            exact += 1;
            continue;
        }
        let a = f32::from_bits(*g);
        let b = f32::from_bits(*w);
        let tol = rel * a.abs().max(b.abs()) + 1e-30;
        assert!(
            (a - b).abs() <= tol,
            "{name} {tag}: word {i} got {a:e} want {b:e} beyond rel {rel:e} -- subgroupAdd order \
             is implementation-defined and q.k cancellation amplifies last-ulp reassociation, so \
             this bound is only a structural tripwire (wrong scale, wrong anchor, wrong position \
             set); fine-grained parity is graded on the stage2 bf16 output the model consumes"
        );
    }
    exact
}

const BF16_TWO_ULP_REL: f32 = 1.0 / 64.0;

fn assert_close_bf16(name: &str, tag: &str, got: &[u32], want: &[u32]) {
    assert_eq!(got.len(), want.len());
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        if g == w {
            continue;
        }
        let a = f32::from_bits((*g & 0xffff) << 16);
        let b = f32::from_bits((*w & 0xffff) << 16);
        let tol = BF16_TWO_ULP_REL * a.abs().max(b.abs()) + 1e-6;
        assert!(
            (a - b).abs() <= tol,
            "{name} {tag}: out {i} got {a:e} want {b:e} beyond 2 bf16 ulp -- the tp arm sums \
             the same positions in a different order, so only last-bit drift is admissible"
        );
    }
}

#[test]
fn q3d_fold_variants_match_the_shipping_fold_sd() {
    let ctx = ctx();
    let arms = arms(ctx);
    assert_eq!(arms.len(), 3, "an arm failed to build");
    let s2 = stage2_pipeline(ctx);
    let mut fixtures = 0usize;
    let mut scratch_exact = 0usize;
    let mut scratch_words = 0usize;
    for total in [1u32, 7, 64, 129, 513, 2048, 2600] {
        for seed in 0..4u64 {
            for splits in [16u32, 64] {
                let s = Shape {
                    slots: 4096,
                    total,
                    splits,
                };
                let b = make_bufs_seed(ctx, &s, 1, 0x9e37_79b9_7f4a_7c15u64.wrapping_mul(seed + 1));
                let want_scratch = scratch_of(ctx, &arms[0], &b, &s);
                let nz = want_scratch.iter().filter(|w| **w != 0).count();
                let live_splits = total.div_ceil(fd::WARPS as u32).min(splits);
                let want_nz = (Q3D_NQ * live_splits * Q3D_HD / 2) as usize;
                assert!(
                    nz >= want_nz,
                    "total={total} splits={splits} seed={seed}: reference scratch {nz} non-zero \
                     under the {want_nz} owed -- zeros compared to zeros stay green forever"
                );
                let want_out = out_of(ctx, &arms[0], &s2, &b, &s);
                for arm in arms.iter().skip(1) {
                    let tag = format!("total={total} splits={splits} seed={seed}");
                    if arm.scratch_comparable_to_sg {
                        let got = scratch_of(ctx, arm, &b, &s);
                        scratch_exact += assert_close_f32(
                            arm.name,
                            &tag,
                            &got,
                            &want_scratch,
                            REDUCE_REASSOC_UNDER_CANCELLATION_CEILING,
                        );
                        scratch_words += got.len();
                    }
                    let got_out = out_of(ctx, arm, &s2, &b, &s);
                    assert_close_bf16(arm.name, &tag, &got_out, &want_out);
                    fixtures += 1;
                }
            }
        }
    }
    eprintln!(
        "q3d fold parity: {fixtures} fixtures; ra scratch bit-exact {scratch_exact}/{scratch_words} \
         words, remainder under the {REDUCE_REASSOC_UNDER_CANCELLATION_CEILING:e} structural \
         ceiling; all arm outs within 2 bf16 ulp of shipping"
    );
}

#[test]
fn kv_fp8_kt_writer_is_a_bit_exact_permutation_of_the_position_major_writer() {
    let ctx = ctx();
    let n_tokens = 13usize;
    let nkv = Q3D_NKV as usize;
    let hd = Q3D_HD as usize;
    let slots = 32usize;
    let start = [5i32];
    let x: Vec<u16> = (0..n_tokens * nkv * hd)
        .map(|i| {
            let f = ((i as f32) * 0.37).sin() * 3.0;
            (f.to_bits() >> 16) as u16
        })
        .collect();
    let mut pm = vec![0u8; slots * nkv * hd];
    let mut pm_scales = vec![0f32; slots * nkv];
    kv_fp8::quantize_kv_fp8(ctx, &x, &mut pm, &mut pm_scales, &start, n_tokens, nkv, hd, 0)
        .expect("position-major quantize");
    let mut kt = vec![0u8; slots * nkv * hd];
    let mut kt_scales = vec![0f32; slots * nkv];
    kv_fp8::quantize_kv_fp8_kt(ctx, &x, &mut kt, &mut kt_scales, &start, n_tokens, nkv, hd, 0)
        .expect("transposed quantize");
    assert_eq!(pm_scales, kt_scales, "scales must not depend on the byte layout");
    let mut checked = 0usize;
    for t in 0..n_tokens {
        let slot = start[0] as usize + t;
        for kvh in 0..nkv {
            for d4 in 0..hd / 4 {
                let pm_off = ((slot * nkv + kvh) * hd / 4 + d4) * 4;
                let kt_off = fd::k_transposed_word_index(slots, nkv, hd, kvh, slot, d4) * 4;
                assert_eq!(
                    &pm[pm_off..pm_off + 4],
                    &kt[kt_off..kt_off + 4],
                    "slot {slot} kvh {kvh} d4 {d4}: the kt writer must emit the same fp8 bytes \
                     at the transposed word, nothing requantized"
                );
                checked += 4;
            }
        }
    }
    assert_eq!(checked, n_tokens * nkv * hd);
}

fn submit_copies(
    ctx: &WgpuContext,
    pl: &wgpu::ComputePipeline,
    groups: &[wgpu::BindGroup],
    grid: (u32, u32, u32),
    copies: usize,
    reps: usize,
) -> (f64, f64) {
    let mut best = f64::INFINITY;
    let mut worst: f64 = 0.0;
    for _ in 0..reps {
        let t0 = Instant::now();
        let mut enc = ctx.device.create_command_encoder(&Default::default());
        {
            let mut pass = enc.begin_compute_pass(&Default::default());
            pass.set_pipeline(pl);
            for c in 0..copies {
                pass.set_bind_group(0, &groups[c % groups.len()], &[]);
                pass.dispatch_workgroups(grid.0, grid.1, grid.2);
            }
        }
        ctx.queue.submit([enc.finish()]);
        ctx.poll_blocking().expect("drain");
        let ms = t0.elapsed().as_secs_f64() * 1e3;
        best = best.min(ms);
        worst = worst.max(ms);
    }
    (best, worst)
}

struct Priced {
    us: f64,
    drift_pct: f64,
}

fn price(
    ctx: &WgpuContext,
    pl: &wgpu::ComputePipeline,
    groups: &[wgpu::BindGroup],
    grid: (u32, u32, u32),
    lo: usize,
    hi: usize,
    reps: usize,
) -> Priced {
    let (a, _) = submit_copies(ctx, pl, groups, grid, lo, reps);
    let (h, _) = submit_copies(ctx, pl, groups, grid, hi, reps);
    let (a2, _) = submit_copies(ctx, pl, groups, grid, lo, reps);
    Priced {
        us: (h - 0.5 * (a + a2)) / (hi - lo) as f64 * 1e3,
        drift_pct: 100.0 * (a2 - a) / a,
    }
}

fn kv_stream_mb(total: u32) -> f64 {
    2.0 * total as f64 * Q3D_NKV as f64 * Q3D_HD as f64 / 1e6
}

#[test]
#[ignore = "timing instrument; set NV_Q3D_FLASH_RATE=1"]
fn q3d_flash_stage1_rate_ladder() {
    assert_eq!(
        std::env::var("NV_Q3D_FLASH_RATE").ok().as_deref(),
        Some("1"),
        "set NV_Q3D_FLASH_RATE=1 -- a silent skip here would report a pass"
    );
    let ctx = ctx();
    let arms = arms(ctx);
    eprintln!(
        "q3d stage1 rate: geometry {}q/{}kv hd {} fold {}, grid [{}xSx1], KV fp8-sd",
        Q3D_NQ,
        Q3D_NKV,
        Q3D_HD,
        Q3D_FOLD,
        Q3D_NQ / Q3D_FOLD
    );
    eprintln!(
        "{:>8} {:>7} {:>24} {:>10} {:>11} {:>9} {:>8}",
        "total", "splits", "arm", "us/disp", "KV MB/disp", "GB/s", "drift%"
    );
    for total in [2048u32, 16384, 65536, 172032] {
        let sets = if total >= 65536 { 2 } else { 4 };
        let (lo, hi, reps) = if total >= 65536 { (4, 16, 8) } else { (8, 64, 12) };
        for splits in [16u32, 64] {
            let s = Shape {
                slots: total,
                total,
                splits,
            };
            let b = make_bufs_seed(ctx, &s, sets, 0x1234_5678_9abc_def0);
            for arm in &arms {
                let groups: Vec<wgpu::BindGroup> = (0..sets)
                    .map(|i| dispatch::bind_group(ctx, &arm.pl, &stage1_bindings(arm, &b, i)))
                    .collect();
                let p = price(
                    ctx,
                    &arm.pl,
                    &groups,
                    (Q3D_NQ / Q3D_FOLD, splits, 1),
                    lo,
                    hi,
                    reps,
                );
                let mb = kv_stream_mb(total);
                eprintln!(
                    "{:>8} {:>7} {:>24} {:>10.2} {:>11.3} {:>9.1} {:>+8.2}",
                    total,
                    splits,
                    arm.name,
                    p.us,
                    mb,
                    mb / p.us * 1e3,
                    p.drift_pct
                );
            }
        }
    }
}
