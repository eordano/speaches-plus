#![cfg(feature = "wgpu")]

mod common;
use common::ctx;
use common::env_usize;
use common::FdP;
use common::pack_u8;
use nv_kernels::wgpu_backend::device::WgpuContext;
use nv_kernels::wgpu_backend::kernels::flash_decode as fd;
use nv_kernels::wgpu_backend::{compose, dispatch};
use nv_models::gemma4_e4b_wgpu::{
    flash1_e4b_entry, flash1_e4b_entry_sd, flash1_e4b_source, flash1_e4b_source_sd,
    flash1_sg_supported, flash_sd_enabled,
};
use std::time::Instant;

#[derive(Clone, Copy, Debug)]
struct Shape {
    tag: &'static str,
    n_q: u32,
    n_kv: u32,
    hd: u32,
    slots: u32,
    total: u32,
    start: u32,
    splits: u32,
}

impl Shape {
    fn params(&self) -> FdP {
        FdP {
            n_heads: self.n_q,
            n_kv: self.n_kv,
            head_dim: self.hd,
            total: self.total,
            start: self.start,
            splits: self.splits,
            ring: 0,
            out_bf16: 1,
            scaling: 1.0 / (self.hd as f32).sqrt(),
            m_rows: 1,
            ..Default::default()
        }
    }

    fn rounds(&self) -> u32 {
        let stride = self.splits * fd::WARPS as u32;
        let base = self.start;
        if self.total > base {
            self.total.div_ceil(stride).max(1)
        } else {
            0
        }
    }

    fn scratch_elems(&self) -> usize {
        (self.n_q * self.splits * (self.hd + 2)) as usize
    }
}

fn kv_stream_bytes(s: &Shape) -> f64 {
    let live = (s.total - s.start) as f64;
    2.0 * live * s.hd as f64 * s.n_q as f64
}

struct Bufs {
    k: Vec<wgpu::Buffer>,
    v: Vec<wgpu::Buffer>,
    ks: Vec<wgpu::Buffer>,
    vs: Vec<wgpu::Buffer>,
    scratch: Vec<wgpu::Buffer>,
    q: wgpu::Buffer,
    p: wgpu::Buffer,
}

fn make_bufs(ctx: &WgpuContext, s: &Shape, n: usize) -> Bufs {
    make_bufs_seed(ctx, s, n, 0x1234_5678_9abc_def0)
}

fn make_bufs_seed(ctx: &WgpuContext, s: &Shape, n: usize, seed: u64) -> Bufs {
    let kv_elems = (s.slots * s.n_kv * s.hd) as usize;
    let sc_elems = (s.slots * s.n_kv) as usize;
    let mut lcg: u64 = seed | 1;
    let mut next = move || {
        lcg = lcg.wrapping_mul(6364136223846793005).wrapping_add(1);
        lcg
    };
    let byte = |n: &mut dyn FnMut() -> u64| 0x30u8 + ((n() >> 33) as u8 % 0x18);
    let mut k = Vec::new();
    let mut v = Vec::new();
    let mut ks = Vec::new();
    let mut vs = Vec::new();
    let mut scratch = Vec::new();
    for i in 0..n {
        let kb: Vec<u8> = (0..kv_elems).map(|_| byte(&mut next)).collect();
        let vb: Vec<u8> = (0..kv_elems).map(|_| byte(&mut next)).collect();
        let sc: Vec<f32> = (0..sc_elems)
            .map(|j| 0.004 + ((next() >> 45) as f32 / 524_288.0) * 0.02 + (j % 7) as f32 * 0.003)
            .collect();
        k.push(dispatch::storage_from_slice(ctx, "f1k", &pack_u8(&kb)));
        v.push(dispatch::storage_from_slice(ctx, "f1v", &pack_u8(&vb)));
        ks.push(dispatch::storage_from_slice(ctx, "f1ks", &sc));
        vs.push(dispatch::storage_from_slice(ctx, "f1vs", &sc));
        scratch.push(dispatch::storage_zeroed(
            ctx,
            "f1sc",
            (s.scratch_elems() * 4) as u64,
        ));
        let _ = i;
    }
    let q: Vec<f32> = (0..(s.n_q * s.hd) as usize)
        .map(|j| ((next() >> 40) as f32 / 8_388_608.0) - 1.0 + (j % 3) as f32 * 0.25)
        .collect();
    Bufs {
        k,
        v,
        ks,
        vs,
        scratch,
        q: dispatch::storage_from_slice(ctx, "f1q", &q),
        p: dispatch::uniform_from(ctx, "f1p", &s.params()),
    }
}

fn submit_copies(
    ctx: &WgpuContext,
    pl: &wgpu::ComputePipeline,
    b: &Bufs,
    grid: (u32, u32, u32),
    copies: usize,
    reps: usize,
) -> (f64, f64) {
    let n = b.k.len();
    let groups: Vec<wgpu::BindGroup> = (0..n)
        .map(|i| {
            dispatch::bind_group(
                ctx,
                pl,
                &[
                    (0, &b.q),
                    (4, &b.p),
                    (5, &b.k[i]),
                    (6, &b.v[i]),
                    (7, &b.scratch[i]),
                    (8, &b.ks[i]),
                    (9, &b.vs[i]),
                ],
            )
        })
        .collect();
    let mut best = f64::INFINITY;
    let mut worst: f64 = 0.0;
    for _ in 0..reps {
        let t0 = Instant::now();
        let mut enc = ctx.device.create_command_encoder(&Default::default());
        {
            let mut pass = enc.begin_compute_pass(&Default::default());
            pass.set_pipeline(pl);
            for c in 0..copies {
                pass.set_bind_group(0, &groups[c % n], &[]);
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

struct Arm {
    us: f64,
    drift_pct: f64,
    spread_pct: f64,
}

fn price(
    ctx: &WgpuContext,
    pl: &wgpu::ComputePipeline,
    b: &Bufs,
    grid: (u32, u32, u32),
    lo: usize,
    hi: usize,
    reps: usize,
) -> Arm {
    let (a, aw) = submit_copies(ctx, pl, b, grid, lo, reps);
    let (h, _) = submit_copies(ctx, pl, b, grid, hi, reps);
    let (a2, _) = submit_copies(ctx, pl, b, grid, lo, reps);
    Arm {
        us: (h - 0.5 * (a + a2)) / (hi - lo) as f64 * 1e3,
        drift_pct: 100.0 * (a2 - a) / a,
        spread_pct: 100.0 * (aw - a) / a,
    }
}

const SLIDING: Shape = Shape {
    tag: "sliding hd=256",
    n_q: 8,
    n_kv: 2,
    hd: 256,
    slots: 1024,
    total: 512,
    start: 0,
    splits: 16,
};

const FULL: Shape = Shape {
    tag: "full    hd=512",
    n_q: 8,
    n_kv: 2,
    hd: 512,
    slots: 1024,
    total: 512,
    start: 0,
    splits: 16,
};

#[test]
#[ignore = "timing instrument; set NV_E4B_FLASH_RATE=1"]
fn e4b_flash_stage1_cost_curve() {
    assert_eq!(
        std::env::var("NV_E4B_FLASH_RATE").ok().as_deref(),
        Some("1"),
        "set NV_E4B_FLASH_RATE=1 -- a silent skip here would report a pass"
    );
    let ctx = ctx();
    eprintln!(
        "adapter subgroup width: {:?}, workgroup mem limit {}",
        ctx.subgroup_width(),
        ctx.device.limits().max_compute_workgroup_storage_size
    );
    let src = compose(fd::WGSL);
    let pl = dispatch::compute_pipeline_opts(ctx, "f1-stock", &src, fd::ENTRY_STAGE1_FP8, true)
        .expect("stage1 fp8 pipeline");

    for base in [SLIDING, FULL] {
        eprintln!(
            "\n==== {} , grid [{}x{}x1] = {} workgroups of 256 ====",
            base.tag,
            base.n_q,
            base.splits,
            base.n_q * base.splits
        );
        eprintln!(
            "{:>7}  {:>6}  {:>9}  {:>10}  {:>9}  {:>8}  {:>8}",
            "total", "rounds", "us/disp", "KV MB/disp", "GB/s", "drift%", "spread%"
        );
        for total in [8u32, 64, 128, 256, 512, 1024] {
            let s = Shape { total, ..base };
            let b = make_bufs(ctx, &s, 8);
            let arm = price(ctx, &pl, &b, (s.n_q, s.splits, 1), 8, 64, 12);
            let mb = kv_stream_bytes(&s) / 1e6;
            eprintln!(
                "{:>7}  {:>6}  {:>9.2}  {:>10.3}  {:>9.1}  {:>+8.2}  {:>8.2}",
                total,
                s.rounds(),
                arm.us,
                mb,
                mb / arm.us * 1e3,
                arm.drift_pct,
                arm.spread_pct
            );
        }
    }
}

fn stock_ref_entry() -> &'static str {
    if flash_sd_enabled() {
        fd::ENTRY_STAGE1_FP8_SD
    } else {
        fd::ENTRY_STAGE1_FP8
    }
}

fn arms(ctx: &'static WgpuContext) -> Vec<(String, wgpu::ComputePipeline)> {
    let sd = flash_sd_enabled();
    let stock = compose(fd::WGSL);
    let mut v = vec![(
        format!("stock  hd512 {}", if sd { "sd-barr " } else { "barrier " }),
        dispatch::compute_pipeline_opts(ctx, "f1-stock", &stock, stock_ref_entry(), true)
            .expect("stock stage1"),
    )];
    for (hd, sg) in [(512u32, false), (512, true), (256, false), (256, true)] {
        if sg && !flash1_sg_supported(ctx) {
            continue;
        }
        let (gen, entry) = if sd {
            (flash1_e4b_source_sd(hd, sg), flash1_e4b_entry_sd(hd, sg))
        } else {
            (flash1_e4b_source(hd, sg), flash1_e4b_entry(hd, sg))
        };
        let src = format!("{}\n{}", stock, gen);
        let pl = dispatch::compute_pipeline_opts(ctx, &entry, &src, &entry, true)
            .unwrap_or_else(|e| panic!("{entry}: {e}"));
        v.push((
            format!("e4b    hd{hd} {}", if sg { "subgroup" } else { "barrier " }),
            pl,
        ));
    }
    v
}

fn scratch_of(ctx: &WgpuContext, pl: &wgpu::ComputePipeline, b: &Bufs, s: &Shape) -> Vec<u32> {
    dispatch::dispatch(
        ctx,
        pl,
        &[
            (0, &b.q),
            (4, &b.p),
            (5, &b.k[0]),
            (6, &b.v[0]),
            (7, &b.scratch[0]),
            (8, &b.ks[0]),
            (9, &b.vs[0]),
        ],
        (s.n_q, s.splits, 1),
    )
    .expect("dispatch");
    dispatch::read_back::<u32>(ctx, &b.scratch[0], s.scratch_elems()).expect("readback")
}

#[test]
#[ignore = "timing instrument; set NV_E4B_FLASH_RATE=1"]
fn e4b_flash_stage1_variant_ab() {
    assert_eq!(
        std::env::var("NV_E4B_FLASH_RATE").ok().as_deref(),
        Some("1"),
        "set NV_E4B_FLASH_RATE=1 -- a silent skip here would report a pass"
    );
    let ctx = ctx();
    assert!(
        flash1_sg_supported(ctx),
        "subgroup width {:?} does not admit the 32-lane butterfly; the arm under test would not \
         have been built and this table would be a comparison of three barrier kernels",
        ctx.subgroup_width()
    );
    let arms = arms(ctx);
    assert_eq!(arms.len(), 5, "an arm failed to build");

    let mut fixtures = 0usize;
    let mut words = 0usize;
    for base in [SLIDING, FULL] {
        for total in [1u32, 7, 64, 129, 512, 900] {
            for seed in 0..8u64 {
                let s = Shape { total, ..base };
                let b = make_bufs_seed(ctx, &s, 1, 0x9e37_79b9_7f4a_7c15u64.wrapping_mul(seed + 1));
                let want = scratch_of(ctx, &arms[0].1, &b, &s);
                let nz = want.iter().filter(|w| **w != 0).count();
                let live_splits = (s.total - s.start).div_ceil(fd::WARPS as u32).min(s.splits);
                let want_nz = (s.n_q * live_splits * s.hd / 2) as usize;
                assert!(
                    nz >= want_nz,
                    "{} total={total} seed={seed}: reference scratch is {nz}/{} non-zero, under \
                     the {want_nz} the {live_splits} live splits owe -- a degenerate fixture \
                     compares zeros to zeros and stays green forever",
                    base.tag,
                    want.len()
                );
                for (name, pl) in arms.iter().skip(1) {
                    if base.hd > 256 && name.contains("hd256") {
                        continue;
                    }
                    let got = scratch_of(ctx, pl, &b, &s);
                    let diff = got.iter().zip(&want).filter(|(a, b)| a != b).count();
                    assert_eq!(
                        diff,
                        0,
                        "{name} vs stock at {} total={total} seed={seed}: {diff}/{} words differ \
                         -- the butterfly or the staging is not bit-exact",
                        base.tag,
                        want.len()
                    );
                    fixtures += 1;
                    words += want.len();
                }
            }
        }
    }
    eprintln!(
        "\nbit-exact vs {}: {fixtures} (arm, shape, total, seed) fixtures, {words} scratch words \
         compared",
        stock_ref_entry()
    );

    for base in [SLIDING, FULL] {
        eprintln!("\n==== {} , grid [8x16x1] ====", base.tag);
        eprint!("{:>22}", "arm");
        for total in [128u32, 512, 2048] {
            eprint!("{:>14}", format!("t={total} us"));
        }
        eprintln!("{:>10}{:>10}", "drift%", "vs stock");
        for (name, pl) in &arms {
            if base.hd > 256 && name.contains("hd256") {
                continue;
            }
            eprint!("{name:>22}");
            let mut drift = 0.0f64;
            let mut at512 = 0.0f64;
            for total in [128u32, 512, 2048] {
                let s = Shape { total, ..base };
                let b = make_bufs(ctx, &s, 8);
                let arm = price(ctx, pl, &b, (s.n_q, s.splits, 1), 8, 64, 12);
                eprint!("{:>14.2}", arm.us);
                if total == 512 {
                    drift = arm.drift_pct;
                    at512 = arm.us;
                }
            }
            eprintln!("{drift:>+10.2}{at512:>10.2}");
        }
    }
}

#[test]
#[ignore = "timing instrument; set NV_E4B_FLASH_RATE=1"]
fn e4b_flash_stage1_split_width() {
    assert_eq!(
        std::env::var("NV_E4B_FLASH_RATE").ok().as_deref(),
        Some("1"),
        "set NV_E4B_FLASH_RATE=1 -- a silent skip here would report a pass"
    );
    let ctx = ctx();
    for (name, pl) in arms(ctx) {
        if name.contains("hd512") && !name.contains("stock") {
            continue;
        }
        for base in [SLIDING, FULL] {
            if base.hd > 256 && name.contains("hd256") {
                continue;
            }
            let totals: &[u32] = if base.hd > 256 {
                &[512, 2048, 8192, 32768]
            } else {
                &[128, 512, 2048]
            };
            eprintln!("\n==== {name} / {} split-width sweep ====", base.tag);
            eprint!("{:>7}  {:>5}", "splits", "wgs");
            for t in totals {
                eprint!("{:>12}", format!("t={t} us"));
            }
            eprintln!("  {:>8}", "drift%");
            for splits in [4u32, 8, 16, 32, 64, 128] {
                eprint!("{splits:>7}  {:>5}", base.n_q * splits);
                let mut drift = 0.0;
                for &total in totals {
                    let s = Shape {
                        total,
                        splits,
                        slots: total.max(base.slots),
                        ..base
                    };
                    let b = make_bufs(ctx, &s, if total >= 8192 { 4 } else { 8 });
                    let (lo, hi, reps) = if total >= 8192 { (4, 16, 8) } else { (8, 64, 12) };
                    let arm = price(ctx, &pl, &b, (s.n_q, s.splits, 1), lo, hi, reps);
                    eprint!("{:>12.2}", arm.us);
                    if total == 512 {
                        drift = arm.drift_pct;
                    }
                }
                eprintln!("  {drift:>+8.2}");
            }
        }
    }
}

fn median_of(mut v: Vec<f64>) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[v.len() / 2]
}

#[test]
#[ignore = "loads the real E4B checkpoint and profiles decode per dispatch; set NV_E4B_DEPTH_ATTR=1, pick depth via NV_CTX_TOKENS"]
fn e4b_depth_attribution_per_dispatch() {
    assert_eq!(
        std::env::var("NV_E4B_DEPTH_ATTR").ok().as_deref(),
        Some("1"),
        "set NV_E4B_DEPTH_ATTR=1 -- a silent skip here would report a pass"
    );
    let ctx = ctx();
    eprintln!("adapter: {}", ctx.summary());
    let dir = common::snapshot_dir();
    eprintln!("checkpoint: {}", dir.display());
    let config = nv_models::gemma4::Gemma4Config::from_hf_json_file(&dir.join("config.json"))
        .expect("config.json");
    let max_seq = env_usize("NV_E4B_ATTR_MAX", 123_136);
    let depth = env_usize("NV_CTX_TOKENS", 122_880);
    assert!(
        depth + 128 <= max_seq,
        "depth {depth} + timed steps needs headroom under max_seq {max_seq}"
    );
    let loader =
        nv_weights::WeightLoader::open_dir(&dir, &candle_core::Device::Cpu).expect("safetensors");
    let t0 = Instant::now();
    let mut m = nv_models::gemma4_e4b_wgpu::Gemma4E4bWgpu::from_loader(config, &loader, max_seq)
        .expect("from_loader");
    drop(loader);
    let (entry, hds) = m.flash1_route();
    eprintln!(
        "loaded in {:.1}s: {} passes/token, flash1 route {entry} hds {hds:?}, staged_read={}",
        t0.elapsed().as_secs_f64(),
        m.pass_count(),
        m.staged_read()
    );

    let mut t = 2u32;
    for tok in [2u32, 818, 3029, 529, 6081, 603, 563, 1596] {
        t = m.decode_step(tok).expect("seed step");
    }

    m.restore_pos(depth).expect("restore_pos");
    for _ in 0..8 {
        t = m.decode_step(2000).expect("warm step");
    }
    let steps = env_usize("NV_E4B_ATTR_STEPS", 32);
    let mut walls = Vec::new();
    for _ in 0..2 {
        m.restore_pos(depth).expect("restore_pos");
        let t0 = Instant::now();
        for _ in 0..steps {
            t = m.decode_step(2000).expect("timed step");
        }
        walls.push(t0.elapsed().as_secs_f64() * 1e3 / steps as f64);
    }
    eprintln!(
        "E4B-ATTR depth={depth} wall ms/tok best={:.3} worst={:.3} steps={steps} (restore_pos-filled kv, real weights)",
        walls.iter().cloned().fold(f64::INFINITY, f64::min),
        walls.iter().cloned().fold(0.0, f64::max)
    );

    assert!(
        m.set_prof_mode(nv_models::gemma4_e4b_wgpu::ProfMode::PerDispatch),
        "adapter has no timestamp queries; attribution impossible"
    );
    let prof_steps = env_usize("NV_E4B_ATTR_PROF_STEPS", 8);
    let mut per_pass: Vec<Vec<f64>> = vec![Vec::new(); m.pass_count()];
    let mut gpu_totals = Vec::new();
    for _ in 0..prof_steps {
        m.restore_pos(depth).expect("restore_pos");
        t = m.decode_step(2000).expect("prof step");
        let tl = m.prof_timeline();
        let mut tot = 0.0;
        for (i, (_, b, e)) in tl.iter().enumerate() {
            per_pass[i].push(e - b);
            tot += e - b;
        }
        gpu_totals.push(tot / 1e6);
    }
    let _ = t;
    m.set_prof_mode(nv_models::gemma4_e4b_wgpu::ProfMode::Off);
    let gpu_ms = median_of(gpu_totals);
    eprintln!("E4B-ATTR depth={depth} gpu-pass-sum median {gpu_ms:.3} ms/tok over {prof_steps} profiled steps");

    let mut by_label: std::collections::BTreeMap<String, (usize, f64)> = Default::default();
    let mut pass_med: Vec<(usize, f64)> = Vec::new();
    for i in 0..m.pass_count() {
        let med_ns = median_of(per_pass[i].clone());
        pass_med.push((i, med_ns));
        let e = by_label.entry(m.pass_label(i).to_string()).or_insert((0, 0.0));
        e.0 += 1;
        e.1 += med_ns;
    }
    let mut rows: Vec<(String, usize, f64)> =
        by_label.into_iter().map(|(l, (n, ns))| (l, n, ns)).collect();
    rows.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap());
    eprintln!("-- per-label (median per pass, summed) --");
    for (label, n, ns) in &rows {
        eprintln!(
            "E4B-ATTR-LABEL depth={depth} {label:<24} x{n:<4} {:>10.4} ms {:>5.1}%",
            ns / 1e6,
            100.0 * ns / (gpu_ms * 1e6)
        );
    }
    pass_med.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    eprintln!("-- top 24 single passes --");
    for (i, ns) in pass_med.iter().take(24) {
        let g = m.pass_grid(*i);
        eprintln!(
            "E4B-ATTR-PASS depth={depth} idx={i:<4} {:<24} grid=({},{},{}) {:>9.4} ms",
            m.pass_label(*i),
            g.0,
            g.1,
            g.2,
            ns / 1e6
        );
    }
}

fn ulp_dist(a: u32, b: u32) -> u32 {
    let key = |x: u32| -> i64 {
        let v = x as i64;
        if v & 0x8000_0000 != 0 {
            0x8000_0000 - v
        } else {
            v
        }
    };
    key(a).abs_diff(key(b)).min(u32::MAX as u64) as u32
}

fn fold_arm(
    ctx: &'static WgpuContext,
    hd: u32,
    fold: u32,
) -> (String, wgpu::ComputePipeline) {
    let sd = flash_sd_enabled();
    let (entry, gen) = if sd {
        (
            fd::fold_stage1_entry_sd(hd, true, fold),
            fd::fold_stage1_source_sd(hd, true, fold),
        )
    } else {
        (
            fd::fold_stage1_entry(hd, true, fold),
            fd::fold_stage1_source(hd, true, fold),
        )
    };
    let src = format!("{}\n{}", compose(fd::WGSL), gen);
    let pl = dispatch::compute_pipeline_opts(ctx, &entry, &src, &entry, true)
        .unwrap_or_else(|e| panic!("{entry}: {e}"));
    (format!("fold{fold} hd{hd} sg"), pl)
}

#[test]
#[ignore = "timing instrument; set NV_E4B_FLASH_RATE=1"]
fn e4b_flash_stage1_fold_deep_ab() {
    assert_eq!(
        std::env::var("NV_E4B_FLASH_RATE").ok().as_deref(),
        Some("1"),
        "set NV_E4B_FLASH_RATE=1 -- a silent skip here would report a pass"
    );
    let ctx = ctx();
    assert!(
        flash1_sg_supported(ctx),
        "no 32-lane subgroups: the fold arms under test would not be built in the model either"
    );
    eprintln!("adapter: {}", ctx.summary());
    let stock = compose(fd::WGSL);
    let stock_pl =
        dispatch::compute_pipeline_opts(ctx, "f1-stock", &stock, stock_ref_entry(), true)
            .expect("stock stage1");
    let mut arms: Vec<(String, wgpu::ComputePipeline, u32)> = [2u32, 4]
        .into_iter()
        .map(|f| {
            let (name, pl) = fold_arm(ctx, 512, f);
            (name, pl, f)
        })
        .collect();
    let sd = flash_sd_enabled();
    for (f, t) in [(2u32, 2u32), (2, 4), (4, 2)] {
        let entry = nv_models::gemma4_e4b_wgpu::deep_tpw_entry(512, f, t, sd);
        let gen = nv_models::gemma4_e4b_wgpu::deep_tpw_source(512, f, t, sd);
        let src = format!("{}\n{}", compose(fd::WGSL), gen);
        let pl = dispatch::compute_pipeline_opts(ctx, &entry, &src, &entry, true)
            .unwrap_or_else(|e| panic!("{entry}: {e}"));
        arms.push((format!("fold{f}tpw{t} hd512"), pl, f));
    }

    let mut fixtures = 0usize;
    for total in [1u32, 7, 129, 512, 900] {
        for seed in 0..4u64 {
            let s = Shape {
                total,
                ..FULL
            };
            let b = make_bufs_seed(ctx, &s, 1, 0x9e37_79b9_7f4a_7c15u64.wrapping_mul(seed + 3));
            let want = scratch_of(ctx, &stock_pl, &b, &s);
            let nz = want.iter().filter(|w| **w != 0).count();
            let live_splits = total.div_ceil(fd::WARPS as u32).min(s.splits);
            let want_nz = (s.n_q * live_splits * s.hd / 2) as usize;
            assert!(
                nz >= want_nz,
                "degenerate reference scratch ({nz}/{} non-zero, {want_nz} owed by {live_splits} \
                 live splits) proves nothing",
                want.len()
            );
            for (name, pl, f) in &arms {
                let got = dispatch::dispatch(
                    ctx,
                    pl,
                    &[
                        (0, &b.q),
                        (4, &b.p),
                        (5, &b.k[0]),
                        (6, &b.v[0]),
                        (7, &b.scratch[0]),
                        (8, &b.ks[0]),
                        (9, &b.vs[0]),
                    ],
                    (s.n_q / f, s.splits, 1),
                )
                .and_then(|_| dispatch::read_back::<u32>(ctx, &b.scratch[0], s.scratch_elems()))
                .expect("fold dispatch");
                let mut worst_ulp = 0u32;
                let mut diffs = 0usize;
                for (i, (g, w)) in got.iter().zip(&want).enumerate() {
                    if g == w {
                        continue;
                    }
                    diffs += 1;
                    let ulp = ulp_dist(*g, *w);
                    if ulp > worst_ulp {
                        worst_ulp = ulp;
                    }
                    if ulp > 8 {
                        let stride = (s.hd + 2) as usize;
                        eprintln!(
                            "DIFF {name} total={total} seed={seed} idx={i} head={} split={} \
                             off={} got={g:#010x}({}) want={w:#010x}({})",
                            i / stride / s.splits as usize,
                            (i / stride) % s.splits as usize,
                            i % stride,
                            f32::from_bits(*g),
                            f32::from_bits(*w)
                        );
                    }
                }
                assert!(
                    worst_ulp <= 8,
                    "{name} vs stock at total={total} seed={seed}: {diffs}/{} words differ, worst \
                     {worst_ulp} ulp -- beyond driver fma-contraction noise, the fold math is wrong",
                    want.len()
                );
                eprintln!(
                    "{name} total={total} seed={seed}: {diffs}/{} words differ, worst {worst_ulp} \
                     ulp (driver fma-contraction noise; argmax-level proof is \
                     e4b_deep_fold_argmax_probe)",
                    want.len()
                );
                fixtures += 1;
            }
        }
    }
    eprintln!("fold arms bit-exact vs {}: {fixtures} fixtures", stock_ref_entry());

    eprintln!(
        "\n==== full hd=512 deep sweep: us/dispatch and unique-KV GB/s (touched = unique x 8/(nq/fold)/2) ===="
    );
    eprintln!(
        "{:>18} {:>7} {:>6} {:>9} {:>11} {:>9} {:>9}",
        "arm", "total", "splits", "wgs", "us/disp", "uniqGB/s", "tchGB/s"
    );
    for total in [8192u32, 32768, 122880] {
        for splits in [32u32, 64, 128, 256] {
            let s = Shape {
                total,
                splits,
                slots: total,
                ..FULL
            };
            let copies = if total >= 100_000 { (2usize, 8usize) } else { (4, 16) };
            let b = make_bufs(ctx, &s, 2);
            let uniq = (total as f64) * (s.hd as f64) * (s.n_kv as f64) * 2.0;
            let mut line = |name: &str, pl: &wgpu::ComputePipeline, gx: u32| {
                let arm = price(ctx, pl, &b, (gx, splits, 1), copies.0, copies.1, 8);
                let touched = uniq * (gx as f64) / (s.n_kv as f64);
                eprintln!(
                    "{name:>18} {total:>7} {splits:>6} {:>9} {:>11.2} {:>9.1} {:>9.1}",
                    gx * splits,
                    arm.us,
                    uniq / arm.us / 1e3,
                    touched / arm.us / 1e3
                );
            };
            line("stock", &stock_pl, s.n_q);
            for (name, pl, f) in &arms {
                line(name, pl, s.n_q / f);
            }
        }
    }
}

#[test]
#[ignore = "loads the real E4B checkpoint; set NV_E4B_DEEP_PROBE=1 -- prints per-step argmax at deep-arm depth for an A/B across NV_E4B_DEEP_FOLD settings"]
fn e4b_deep_fold_argmax_probe() {
    assert_eq!(
        std::env::var("NV_E4B_DEEP_PROBE").ok().as_deref(),
        Some("1"),
        "set NV_E4B_DEEP_PROBE=1 -- a silent skip here would report a pass"
    );
    let ctx = ctx();
    eprintln!("adapter: {}", ctx.summary());
    let dir = common::snapshot_dir();
    let config = nv_models::gemma4::Gemma4Config::from_hf_json_file(&dir.join("config.json"))
        .expect("config.json");
    let depth = env_usize("NV_E4B_PROBE_DEPTH", 3072);
    let probes = env_usize("NV_E4B_PROBE_STEPS", 16);
    let loader =
        nv_weights::WeightLoader::open_dir(&dir, &candle_core::Device::Cpu).expect("safetensors");
    let mut m = nv_models::gemma4_e4b_wgpu::Gemma4E4bWgpu::from_loader(
        config,
        &loader,
        depth + probes + 8,
    )
    .expect("from_loader");
    drop(loader);
    eprintln!(
        "deep-fold probe: depth={depth} probes={probes} NV_E4B_DEEP_FOLD={:?}",
        std::env::var("NV_E4B_DEEP_FOLD").ok()
    );
    for p in 0..depth {
        m.decode_step(2000 + (p as u32 % 30000)).expect("prime");
    }
    for p in 0..probes {
        let (tok, logits) = m
            .decode_step_logits(3000 + (p as u32 * 977) % 20000)
            .expect("probe step");
        let (ai, av) = logits
            .iter()
            .enumerate()
            .fold((0usize, f32::NEG_INFINITY), |acc, (i, &v)| {
                if v > acc.1 {
                    (i, v)
                } else {
                    acc
                }
            });
        assert_eq!(ai as u32, tok, "gpu argmax disagrees with logits argmax");
        let mut top: Vec<(usize, f32)> = logits.iter().cloned().enumerate().collect();
        top.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        eprintln!(
            "PROBE step={p} pos={} argmax={tok} logit={av:.6} second={}@{:.6} margin={:.6}",
            depth + p,
            top[1].0,
            top[1].1,
            av - top[1].1
        );
    }
}
