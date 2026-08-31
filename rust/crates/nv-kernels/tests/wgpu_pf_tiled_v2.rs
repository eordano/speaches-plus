#![cfg(feature = "wgpu")]

mod common;
use common::FdParams;
use nv_kernels::wgpu_backend::device::WgpuContext;
use nv_kernels::wgpu_backend::kernels::flash_decode as fd;
use nv_kernels::wgpu_backend::{compose, dispatch};
use std::time::Instant;

const Q38_NQ: u32 = 24;
const Q38_NKV: u32 = 4;
const Q38_HD: u32 = 256;
const Q38_PF_M: u32 = 256;
const FDT_ROWS_MATCHES_THE_TILED_KERNELS_BAKED_TILE_HEIGHT: u32 = 32;

const PF_TILED_WGSL: &str = include_str!("../wgsl/q3m_pf_flash_tiled.wgsl");

fn ctx() -> &'static WgpuContext {
    let c = WgpuContext::shared().expect("wgpu adapter required for --features wgpu");
    assert_eq!(
        c.subgroup_width(),
        Some(32),
        "the sg tiled arms assume 32-lane subgroups"
    );
    c
}

struct Shape {
    slots: u32,
    total: u32,
    splits: u32,
}

impl Shape {
    fn params(&self) -> FdParams {
        FdParams {
            n_heads: Q38_NQ,
            n_kv: Q38_NKV,
            head_dim: Q38_HD,
            total: self.total,
            start: 0,
            splits: self.splits,
            ring: 0,
            out_bf16: 1,
            scaling: 1.0 / (Q38_HD as f32).sqrt(),
            m_rows: Q38_PF_M,
            ..Default::default()
        }
    }

    fn tiles(&self) -> u32 {
        Q38_PF_M.div_ceil(FDT_ROWS_MATCHES_THE_TILED_KERNELS_BAKED_TILE_HEIGHT)
    }

    fn scratch_elems(&self) -> usize {
        (Q38_NQ * Q38_PF_M * self.splits * (Q38_HD + 2)) as usize
    }
}

struct Lcg(u64);

impl Lcg {
    fn next(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.0
    }
    fn kv_byte(&mut self) -> u8 {
        0x30u8 + ((self.next() >> 33) as u8 % 0x18)
    }
    fn f32_pm1(&mut self) -> f32 {
        ((self.next() >> 40) as f32 / 8_388_608.0) - 1.0
    }
}

struct Bufs {
    q: wgpu::Buffer,
    q16: wgpu::Buffer,
    k: wgpu::Buffer,
    v: wgpu::Buffer,
    ks: wgpu::Buffer,
    vs: wgpu::Buffer,
    scratch: wgpu::Buffer,
    p: wgpu::Buffer,
}

fn make_bufs(ctx: &WgpuContext, s: &Shape, seed: u64) -> Bufs {
    let mut r = Lcg(seed | 1);
    let kv_elems = (s.slots * Q38_NKV * Q38_HD) as usize;
    let kw: Vec<u32> = (0..kv_elems / 4)
        .map(|_| u32::from_le_bytes([r.kv_byte(), r.kv_byte(), r.kv_byte(), r.kv_byte()]))
        .collect();
    let vw: Vec<u32> = (0..kv_elems / 4)
        .map(|_| u32::from_le_bytes([r.kv_byte(), r.kv_byte(), r.kv_byte(), r.kv_byte()]))
        .collect();
    let sc_elems = (s.slots * Q38_NKV) as usize;
    let ks: Vec<f32> = (0..sc_elems)
        .map(|j| 0.004 + ((r.next() >> 45) as f32 / 524_288.0) * 0.02 + (j % 7) as f32 * 0.003)
        .collect();
    let vs: Vec<f32> = (0..sc_elems)
        .map(|j| 0.004 + ((r.next() >> 45) as f32 / 524_288.0) * 0.02 + (j % 5) as f32 * 0.003)
        .collect();
    let q: Vec<f32> = (0..(Q38_PF_M * Q38_NQ * Q38_HD) as usize)
        .map(|_| half::f16::from_f32(r.f32_pm1()).to_f32())
        .collect();
    let q16: Vec<u16> = q.iter().map(|x| half::f16::from_f32(*x).to_bits()).collect();
    Bufs {
        q: dispatch::storage_from_slice(ctx, "pfv2-q", &q),
        q16: dispatch::storage_from_slice(ctx, "pfv2-q16", &q16),
        k: dispatch::storage_from_slice(ctx, "pfv2-k", &kw),
        v: dispatch::storage_from_slice(ctx, "pfv2-v", &vw),
        ks: dispatch::storage_from_slice(ctx, "pfv2-ks", &ks),
        vs: dispatch::storage_from_slice(ctx, "pfv2-vs", &vs),
        scratch: dispatch::storage_zeroed(ctx, "pfv2-scratch", (s.scratch_elems() * 4) as u64),
        p: dispatch::uniform_from(ctx, "pfv2-p", &s.params()),
    }
}

fn stage1_bindings<'a>(b: &'a Bufs, coop: bool) -> Vec<(u32, &'a wgpu::Buffer)> {
    vec![
        (0, if coop { &b.q16 } else { &b.q }),
        (4, &b.p),
        (5, &b.k),
        (6, &b.v),
        (7, &b.scratch),
        (8, &b.ks),
        (9, &b.vs),
    ]
}

fn stock_pipeline(ctx: &WgpuContext) -> wgpu::ComputePipeline {
    let src = compose(&format!("{}\n{}", fd::WGSL, PF_TILED_WGSL));
    dispatch::compute_pipeline_opts(
        ctx,
        "pfv2-stock",
        &src,
        "q3w_pf_flash1_fp8kv_tiled_slotml_sg",
        true,
    )
    .expect("stock tiled slotml_sg pipeline")
}

fn coop_pipeline(ctx: &'static WgpuContext) -> wgpu::ComputePipeline {
    use nv_kernels::wgpu_backend::kernels::pf_flash_tiled_coop as pfc;
    pfc::supported(ctx, Q38_HD as usize, Q38_PF_M as usize, 0, 0)
        .unwrap_or_else(|why| panic!("coop pf arm unsupported on this adapter: {why}"));
    dispatch::compute_pipeline_opts(ctx, "pfv2-coop", &pfc::source(), pfc::ENTRY_STAGE1, true)
        .expect("coop tiled stage1 pipeline")
}

fn stage2_out(ctx: &WgpuContext, b: &Bufs) -> Vec<u32> {
    let src = compose(fd::WGSL);
    let s2 = dispatch::compute_pipeline_opts(ctx, "pfv2-s2", &src, fd::ENTRY_STAGE2_MK, true)
        .expect("stage2_mk pipeline");
    let out_elems = (Q38_PF_M * Q38_NQ * Q38_HD) as usize;
    let ob = dispatch::storage_zeroed(ctx, "pfv2-out", (out_elems * 4) as u64);
    dispatch::dispatch(ctx, &s2, &[(3, &ob), (4, &b.p), (7, &b.scratch)], (Q38_NQ, Q38_PF_M, 1))
        .expect("stage2 dispatch");
    dispatch::read_back::<u32>(ctx, &ob, out_elems).expect("out readback")
}

fn run_stage1(ctx: &WgpuContext, pl: &wgpu::ComputePipeline, b: &Bufs, s: &Shape, coop: bool) {
    dispatch::dispatch(ctx, pl, &stage1_bindings(b, coop), (Q38_NQ, s.splits, s.tiles()))
        .expect("stage1 dispatch");
}

fn time_stage1_ms(
    ctx: &WgpuContext,
    pl: &wgpu::ComputePipeline,
    b: &Bufs,
    s: &Shape,
    coop: bool,
    copies: usize,
    reps: usize,
) -> f64 {
    let group = dispatch::bind_group(ctx, pl, &stage1_bindings(b, coop));
    let mut best = f64::INFINITY;
    for _ in 0..reps {
        let t0 = Instant::now();
        let mut enc = ctx.device.create_command_encoder(&Default::default());
        {
            let mut pass = enc.begin_compute_pass(&Default::default());
            pass.set_pipeline(pl);
            for _ in 0..copies {
                pass.set_bind_group(0, &group, &[]);
                pass.dispatch_workgroups(Q38_NQ, s.splits, s.tiles());
            }
        }
        ctx.queue.submit([enc.finish()]);
        ctx.poll_blocking().expect("drain");
        best = best.min(t0.elapsed().as_secs_f64() * 1e3);
    }
    best
}

fn price_stage1_ms(
    ctx: &WgpuContext,
    pl: &wgpu::ComputePipeline,
    b: &Bufs,
    s: &Shape,
    coop: bool,
) -> f64 {
    let (lo, hi, reps) = if s.total >= 65536 { (2, 6, 6) } else { (4, 20, 8) };
    let a = time_stage1_ms(ctx, pl, b, s, coop, lo, reps);
    let h = time_stage1_ms(ctx, pl, b, s, coop, hi, reps);
    (h - a) / (hi - lo) as f64
}

fn attn_gflop(s: &Shape) -> f64 {
    let m = Q38_PF_M as f64;
    let base = (s.total - Q38_PF_M) as f64;
    let mean_ctx = base + (m + 1.0) / 2.0;
    m * mean_ctx * Q38_NQ as f64 * Q38_HD as f64 * 4.0 / 1e9
}

#[test]
#[ignore = "timing instrument; set NV_PF_TILED_RATE=1 and run under nvk.sh probe"]
fn pf_tiled_stage1_rate_ladder_at_qwen38_geometry() {
    assert_eq!(
        std::env::var("NV_PF_TILED_RATE").ok().as_deref(),
        Some("1"),
        "set NV_PF_TILED_RATE=1 -- a silent skip here would report a pass"
    );
    let ctx = ctx();
    let stock = stock_pipeline(ctx);
    let coop = coop_pipeline(ctx);
    eprintln!(
        "pf tiled stage1 rate: geometry {}q/{}kv hd {} m {} tiles {}, fp8 KV exact decode",
        Q38_NQ,
        Q38_NKV,
        Q38_HD,
        Q38_PF_M,
        Q38_PF_M / 32
    );
    eprintln!(
        "{:>8} {:>7} {:>28} {:>10} {:>9} {:>9}",
        "total", "splits", "arm", "ms/disp", "GFLOP", "TFLOP/s"
    );
    for total in [4096u32, 16384, 65536, 172032] {
        let splits = fd::splits_for(total as usize) as u32;
        let s = Shape {
            slots: total,
            total,
            splits,
        };
        let b = make_bufs(ctx, &s, 0x9e37_79b9_7f4a_7c15);
        for (name, pl, is_coop) in [
            ("slotml_sg (shipping)", &stock, false),
            ("coop 16x16 f16", &coop, true),
        ] {
            run_stage1(ctx, pl, &b, &s, is_coop);
            let ms = price_stage1_ms(ctx, pl, &b, &s, is_coop);
            let gf = attn_gflop(&s);
            eprintln!(
                "{:>8} {:>7} {:>28} {:>10.3} {:>9.1} {:>9.2}",
                total,
                splits,
                name,
                ms,
                gf,
                gf / ms
            );
        }
    }
}

fn bf16_out_f32(words: &[u32]) -> Vec<f32> {
    words.iter().map(|w| f32::from_bits(w << 16)).collect()
}

#[test]
fn coop_pf_stage1_matches_the_slotml_arm_through_stage2_within_f16_reassociation() {
    let ctx = ctx();
    let stock = stock_pipeline(ctx);
    let coop = coop_pipeline(ctx);
    for (total, splits) in [
        (300u32, 16u32),
        (2113, 16),
        (4096, 64),
        (8192, 64),
    ] {
        let s = Shape {
            slots: total,
            total,
            splits,
        };
        let b = make_bufs(ctx, &s, 0x5eed_0000_0000_0001 ^ (total as u64) << 32);
        run_stage1(ctx, &stock, &b, &s, false);
        let want = bf16_out_f32(&stage2_out(ctx, &b));
        run_stage1(ctx, &coop, &b, &s, true);
        let got = bf16_out_f32(&stage2_out(ctx, &b));
        assert_eq!(want.len(), got.len());
        let peak = want.iter().fold(0f32, |a, v| a.max(v.abs())).max(1e-6);
        let mut max_rel = 0f32;
        let mut sum_rel = 0f64;
        for (i, (w, g)) in want.iter().zip(&got).enumerate() {
            assert!(
                w.is_finite() && g.is_finite(),
                "non-finite output at {i}: stock {w} coop {g} (total {total})"
            );
            let rel = (w - g).abs() / peak;
            sum_rel += rel as f64;
            max_rel = max_rel.max(rel);
        }
        let mean_rel = sum_rel / want.len() as f64;
        eprintln!(
            "[pf-coop-parity] total={total} splits={splits} peak={peak:.4} max_rel={max_rel:.3e} \
             mean_rel={mean_rel:.3e}"
        );
        assert!(
            max_rel < 2e-2 && mean_rel < 1e-3,
            "coop stage1 drifted past the f16-reassociation band at total {total}: \
             max_rel {max_rel:.3e} mean_rel {mean_rel:.3e}"
        );
    }
}
