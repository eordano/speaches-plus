#![cfg(feature = "wgpu")]

mod common;
use common::env_usize;
use common::med;
use common::snapshot_dir;
use nv_kernels::wgpu_backend::{dispatch, WgpuContext};
use nv_models::gemma4::Gemma4Config;
use nv_models::gemma4_e4b_wgpu::Gemma4E4bWgpu;
use std::time::Instant;

fn lo_of(xs: &[f64]) -> f64 {
    xs.iter().cloned().fold(f64::INFINITY, f64::min)
}

const ARENA_WGSL: &str = r#"
struct SP { vec4s: u32, base: u32, stride: u32, pad: u32 };
@group(0) @binding(0) var<storage, read_write> arena: array<vec4<u32>>;
@group(0) @binding(1) var<storage, read_write> sink: array<u32>;
@group(0) @binding(2) var<uniform> sp: SP;

@compute @workgroup_size(256)
fn arena_fill(@builtin(global_invocation_id) gid: vec3<u32>) {
    var i = gid.x;
    loop {
        if (i >= sp.vec4s) { break; }
        let s = i * 2654435761u + 12345u;
        arena[i] = vec4<u32>(s, s ^ 0x9e3779b9u, s * 3u + 1u, ~s);
        i = i + sp.stride;
    }
    // Keeps binding 1 in this entry's derived layout, so both entries share
    // one bind group shape. Never true.
    if (sp.base == 0xffffffffu) { sink[0] = i; }
}

@compute @workgroup_size(256)
fn arena_read(@builtin(global_invocation_id) gid: vec3<u32>) {
    var acc = vec4<u32>(0u);
    var i = gid.x;
    loop {
        if (i >= sp.vec4s) { break; }
        acc = acc ^ arena[sp.base + i];
        i = i + sp.stride;
    }
    let t = acc.x ^ acc.y ^ acc.z ^ acc.w;
    // Never true for this fill; the compiler cannot prove it, so the loads
    // survive. A fixture the shader compiler folds to a constant is the
    // classic way this measurement measures nothing.
    if (t == 0xfeedfaceu && sp.base == 0xffffffffu) {
        sink[0] = t;
    }
}
"#;

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Sp {
    vec4s: u32,
    base: u32,
    stride: u32,
    pad: u32,
}

struct Arena {
    ctx: &'static WgpuContext,
    read: wgpu::ComputePipeline,
    buf: wgpu::Buffer,
    sink: wgpu::Buffer,
    bytes: u64,
}

impl Arena {
    fn new(ctx: &'static WgpuContext, want: u64) -> Self {
        let limit = ctx
            .caps
            .max_storage_buffer_binding_size
            .min(ctx.caps.max_buffer_size);
        let bytes = want.min(limit) & !4095;
        assert!(
            bytes >= 256 * 1024 * 1024,
            "arena capped at {bytes} B by the {limit} B binding limit; a roofline probe that fits \
             in cache measures cache, not the memory system"
        );
        let fill = dispatch::compute_pipeline(ctx, "e4b-budget-arena", ARENA_WGSL, "arena_fill")
            .expect("arena_fill");
        let read = dispatch::compute_pipeline(ctx, "e4b-budget-arena", ARENA_WGSL, "arena_read")
            .expect("arena_read");
        let buf = dispatch::storage_zeroed(ctx, "e4b-budget-arena", bytes);
        let sink = dispatch::storage_zeroed(ctx, "e4b-budget-sink", 256);
        let vec4s = (bytes / 16) as u32;
        let grid = (vec4s.div_ceil(256)).min(ctx.caps.max_compute_workgroups_per_dimension);
        let sp = dispatch::uniform_from(
            ctx,
            "e4b-budget-sp",
            &Sp {
                vec4s,
                base: 0,
                stride: grid * 256,
                pad: 0,
            },
        );
        let bind = dispatch::bind_group(ctx, &fill, &[(0, &buf), (1, &sink), (2, &sp)]);
        let mut enc = ctx.device.create_command_encoder(&Default::default());
        {
            let mut cp = enc.begin_compute_pass(&Default::default());
            cp.set_pipeline(&fill);
            cp.set_bind_group(0, &bind, &[]);
            cp.dispatch_workgroups(grid, 1, 1);
        }
        ctx.queue.submit([enc.finish()]);
        ctx.poll_blocking().expect("arena fill");
        Self {
            ctx,
            read,
            buf,
            sink,
            bytes,
        }
    }

    fn sweep(&self, footprint: u64, depth: u32, reps: usize) -> (f64, usize) {
        let ctx = self.ctx;
        let fp = footprint.max(4096) & !4095;
        let windows = (self.bytes / fp).max(1) as usize;
        let vec4s = (fp / 16) as u32;
        let grid = (vec4s / depth.max(1))
            .div_ceil(256)
            .min(ctx.caps.max_compute_workgroups_per_dimension)
            .max(1);
        let mut binds = Vec::with_capacity(windows);
        let mut keep = Vec::with_capacity(windows);
        for w in 0..windows {
            let sp = dispatch::uniform_from(
                ctx,
                "e4b-budget-sp",
                &Sp {
                    vec4s,
                    base: (w as u64 * fp / 16) as u32,
                    stride: grid * 256,
                    pad: 0,
                },
            );
            binds.push(dispatch::bind_group(
                ctx,
                &self.read,
                &[(0, &self.buf), (1, &self.sink), (2, &sp)],
            ));
            keep.push(sp);
        }
        let mut best = f64::INFINITY;
        for _ in 0..reps {
            let mut enc = ctx.device.create_command_encoder(&Default::default());
            {
                let mut cp = enc.begin_compute_pass(&Default::default());
                cp.set_pipeline(&self.read);
                for b in &binds {
                    cp.set_bind_group(0, b, &[]);
                    cp.dispatch_workgroups(grid, 1, 1);
                }
            }
            let t0 = Instant::now();
            ctx.queue.submit([enc.finish()]);
            ctx.poll_blocking().expect("arena read");
            best = best.min(t0.elapsed().as_secs_f64() * 1e3);
        }
        (best, windows)
    }
}

fn fit_law(points: &[(f64, f64, f64)]) -> (f64, f64) {
    let n = points.len() as f64;
    let (sx, sy): (f64, f64) = points
        .iter()
        .fold((0.0, 0.0), |(a, b), p| (a + p.0, b + p.2));
    let sxx: f64 = points.iter().map(|p| p.0 * p.0).sum();
    let sxy: f64 = points.iter().map(|p| p.0 * p.2).sum();
    let den = n * sxx - sx * sx;
    let a = (n * sxy - sx * sy) / den;
    let c = (sy - a * sx) / n;
    let bytes = points[0].1;
    (a, bytes / c * 1e-3)
}

const PROMPT: [u32; 8] = [2, 818, 3029, 529, 6081, 603, 563, 1596];

fn wall(m: &mut Gemma4E4bWgpu, warm: usize, steps: usize) -> (f64, f64) {
    m.reset();
    let mut t = 0u32;
    for &p in &PROMPT {
        t = m.decode_step(p).expect("prompt");
    }
    for _ in 0..warm {
        t = m.decode_step(t).expect("warm");
    }
    let mut xs = Vec::with_capacity(steps);
    for _ in 0..steps {
        let s = Instant::now();
        t = m.decode_step(t).expect("step");
        xs.push(s.elapsed().as_secs_f64() * 1e3);
    }
    (lo_of(&xs), med(&mut xs))
}

struct Cost {
    label: String,
    n: usize,
    hi: usize,
    ratio: f64,
    zero: f64,
    us: f64,
    null_us: f64,

    own_us: f64,
    base: f64,

    mb: f64,
    grid: (u32, u32, u32),
    wg: u64,
}

#[test]
#[ignore = "loads the E4B QAT checkpoint and replicates every pass class; set NV_E4B_BUDGET=1"]
fn e4b_decode_pass_budget() {
    assert_eq!(
        std::env::var("NV_E4B_BUDGET").ok().as_deref(),
        Some("1"),
        "set NV_E4B_BUDGET=1 -- a silent skip here would report a pass"
    );
    let ctx = WgpuContext::shared().expect("wgpu adapter");
    eprintln!("adapter: {}", ctx.summary());
    if let Ok(o) = std::process::Command::new("uptime").output() {
        eprintln!("host: {}", String::from_utf8_lossy(&o.stdout).trim());
    }

    let arena = Arena::new(
        ctx,
        env_usize("NV_E4B_BUDGET_ARENA_MIB", 1024) as u64 * (1 << 20),
    );
    eprintln!(
        "\n==== flat contiguous read, {:.0} MiB arena read once per arm ====",
        arena.bytes as f64 / (1u64 << 20) as f64
    );
    eprintln!(
        "{:>12} {:>10} {:>6} {:>12} {:>12} {:>10}",
        "footprint", "dispatches", "depth", "ms", "GB/s", "us/disp"
    );
    let mut law_points: Vec<(f64, f64, f64)> = Vec::new();
    let mut fp = 1u64 << 20;
    while fp <= arena.bytes {
        let mut best = (f64::INFINITY, 0usize, 1u32);
        for depth in [1u32, 2, 4, 8, 16, 32, 64] {
            let (ms, w) = arena.sweep(fp, depth, 3);
            if ms < best.0 {
                best = (ms, w, depth);
            }
        }
        let (ms, w, depth) = best;
        let gbs = arena.bytes as f64 / 1e9 / (ms * 1e-3);
        eprintln!(
            "{:>9.2} MiB {:>10} {:>6} {:>12.3} {:>12.1} {:>10.2}",
            fp as f64 / (1u64 << 20) as f64,
            w,
            depth,
            ms,
            gbs,
            ms * 1e3 / w as f64
        );
        law_points.push((w as f64, arena.bytes as f64, ms * 1e3));
        fp *= 2;
    }
    let (fixed_us, rate_gbs) = fit_law(&law_points);
    eprintln!(
        "fit over {} arms: t = {fixed_us:.2} us/dispatch + bytes / {rate_gbs:.1} GB/s",
        law_points.len()
    );

    assert!(
        (1.5..=14.0).contains(&fixed_us),
        "per-dispatch fixed cost fits at {fixed_us:.2} us; the published law is 3.9-4.3 us and \
         anything outside this band means the instrument or the box is off, not the kernel"
    );
    assert!(
        (300.0..=900.0).contains(&rate_gbs),
        "flat-read slope fits at {rate_gbs:.1} GB/s against a published 748-755 stream read; the \
         instrument or the box is off, not the kernel"
    );

    let dir = snapshot_dir();
    eprintln!("\ncheckpoint: {}", dir.display());
    let config = Gemma4Config::from_hf_json_file(&dir.join("config.json")).expect("config");
    let loader =
        nv_weights::WeightLoader::open_dir(&dir, &candle_core::Device::Cpu).expect("safetensors");
    let max_seq = env_usize("NV_E4B_BUDGET_SEQ", 512);
    let t0 = Instant::now();
    let mut m = Gemma4E4bWgpu::from_loader(config, &loader, max_seq).expect("build graph");
    let n_pass = m.pass_count();
    let (blk, v4, sg) = m.w4_route_census();
    eprintln!(
        "built in {:.1}s -- {n_pass} passes/token, {:.3} GB weights/token, w4 route block/v4/sg16 \
         {blk}/{v4}/{sg}, grain {:?}",
        t0.elapsed().as_secs_f64(),
        m.weight_bytes_per_token() as f64 / 1e9,
        m.w4_scale_grain()
    );
    assert!(
        sg > 0,
        "no sg16 w4 projections in the graph; this is not the shipping route and the budget would \
         price a kernel serving never runs"
    );

    let warm = env_usize("NV_E4B_BUDGET_WARM", 4);
    let steps = env_usize("NV_E4B_BUDGET_STEPS", 12);
    assert!(
        PROMPT.len() + warm + 3 * steps + 8 < max_seq,
        "{warm} warm + 3x{steps} rounds overruns a {max_seq}-slot kv cache"
    );

    let (wall_pre, med_pre) = wall(&mut m, warm, 24);
    m.set_preenc(false);
    let (wall_ms, wall_med) = wall(&mut m, warm, 24);
    let (wall_null, med_null) = wall(&mut m, warm, 24);
    let pos = m.current_pos();
    m.probe_at(0, pos.min(max_seq - 1)).expect("probe_at");
    let probe = |f: &dyn Fn(), reps: usize| -> f64 {
        let mut best = f64::INFINITY;
        for _ in 0..reps {
            let s = Instant::now();
            f();
            best = best.min(s.elapsed().as_secs_f64() * 1e3);
        }
        best
    };
    let t_empty = probe(&|| m.probe_prefix(0).expect("empty"), 200);
    let t_graph = probe(&|| m.probe_prefix(n_pass).expect("graph"), 16);
    let e_empty = probe(&|| m.probe_encode(0).expect("enc"), 400);
    let e_graph = probe(&|| m.probe_encode(n_pass).expect("enc"), 400);
    let encode = e_graph - e_empty;

    eprintln!("\n==== envelope for one decode token ====");
    eprintln!(
        "  preenc on  (shipping default)     : {wall_pre:.3} ms floor ({:.1} tok/s), {med_pre:.3} median",
        1e3 / wall_pre
    );
    eprintln!(
        "  preenc off (this instrument)      : {wall_ms:.3} ms floor ({:.1} tok/s), {wall_med:.3} median",
        1e3 / wall_ms
    );
    eprintln!(
        "  A-prime, the same arm again       : {wall_null:.3} ms floor, {med_null:.3} median   \
         (drift {:+.2}%)",
        100.0 * (wall_null - wall_ms) / wall_ms
    );
    eprintln!(
        "  submit an empty graph and drain   : {t_empty:.3} ms  ({:.2}%)",
        100.0 * t_empty / wall_ms
    );
    eprintln!(
        "  host encode of {n_pass} dispatches  : {encode:.3} ms  ({:.2}%)   {:.2} us/dispatch",
        100.0 * encode / wall_ms,
        1e3 * encode / n_pass as f64
    );
    eprintln!(
        "  GPU, whole graph, no readback     : {:.3} ms  ({:.1}%)",
        t_graph - t_empty - encode,
        100.0 * (t_graph - t_empty - encode) / wall_ms
    );

    let readback = wall_ms - t_graph;
    eprintln!(
        "  readback + map + host tail        : {readback:.3} ms  ({:.2}%){}",
        100.0 * readback / wall_ms,
        if readback < 0.0 {
            "   <- residual of two floors, below its own noise; read as 0"
        } else {
            ""
        }
    );

    let streamed_mb = |label: &str, widest: u64| -> f64 {
        if label.starts_with("embed_gather") {
            0.0
        } else if label.starts_with("gemv") || label.starts_with("lmhead") {
            widest as f64 / 1e6
        } else {
            0.0
        }
    };

    let mut keys: Vec<(String, u64)> = (0..n_pass)
        .map(|i| (m.pass_label(i).to_string(), m.pass_bound_bytes(i).1))
        .collect();
    keys.sort();
    keys.dedup();
    let mut counts: Vec<usize> = Vec::with_capacity(keys.len());
    let mut meta: Vec<(f64, (u32, u32, u32))> = Vec::with_capacity(keys.len());
    let mut attributed = 0u64;
    for (l, w) in &keys {
        let idx: Vec<usize> = (0..n_pass)
            .filter(|&i| m.pass_label(i) == l && m.pass_bound_bytes(i).1 == *w)
            .collect();
        counts.push(idx.len());
        let mb = streamed_mb(l, *w) * idx.len() as f64;
        attributed += (mb * 1e6) as u64;
        meta.push((mb, m.pass_grid(idx[0])));
    }

    let engine_bytes = m.weight_bytes_per_token();
    eprintln!(
        "\nbyte accounting: {} classes attribute {:.3} GB/token against the engine's own \
         weight_bytes_per_token() {:.3} GB ({:+.1}%)",
        keys.len(),
        attributed as f64 / 1e9,
        engine_bytes as f64 / 1e9,
        100.0 * (attributed as f64 - engine_bytes as f64) / engine_bytes as f64
    );
    assert!(
        (attributed as f64 - engine_bytes as f64).abs() < 0.12 * engine_bytes as f64,
        "the per-dispatch census attributes {:.3} GB where the engine reports {:.3} GB; fix the \
         attribution before quoting any rate",
        attributed as f64 / 1e9,
        engine_bytes as f64 / 1e9
    );

    let lo = env_usize("NV_E4B_BUDGET_LO", 1);
    let target_ms = env_usize("NV_E4B_BUDGET_TARGET_PCT", 55) as f64 / 100.0 * wall_ms;
    let s0 = Instant::now();
    let mut costs: Vec<Cost> = Vec::new();
    let mut bases: Vec<f64> = Vec::new();
    for (li, (label, width)) in keys.iter().enumerate() {
        let n = counts[li];
        let copy_ms = (n as f64 * fixed_us + meta[li].0 * 1e6 / (rate_gbs * 1e3)) / 1e3;
        let hi = lo + ((target_ms / copy_ms).round() as usize).clamp(3, 512);
        let added = m.probe_append_class(label, Some(*width), lo);
        assert_eq!(added, n * lo, "probe_append did not reach the graph");
        assert_eq!(
            m.pass_count(),
            n_pass + n * lo,
            "appended graph is the wrong length"
        );
        m.reset();
        let mut t = 0u32;
        for &p in &PROMPT {
            t = m.decode_step(p).expect("prompt");
        }
        for _ in 0..warm {
            t = m.decode_step(t).expect("warm");
        }
        let step = |m: &mut Gemma4E4bWgpu, t: &mut u32, copies: usize| -> f64 {
            m.probe_append_class(label, Some(*width), copies);
            let s = Instant::now();
            *t = m.decode_step(*t).expect("step");
            s.elapsed().as_secs_f64() * 1e3
        };
        let (mut ratio, mut zero, mut base) = (Vec::new(), Vec::new(), Vec::new());
        for _ in 0..steps {
            let a = step(&mut m, &mut t, lo);
            let b = step(&mut m, &mut t, hi);
            let a2 = step(&mut m, &mut t, lo);
            let lo_arm = (a + a2) / 2.0;

            ratio.push((b - lo_arm) / lo_arm);
            zero.push((a - a2).abs() / lo_arm);
            base.push(lo_arm);
        }
        m.probe_append_clear();
        assert_eq!(m.pass_count(), n_pass, "graph not restored");
        let base_lo_label = lo_of(&base);
        bases.push(base_lo_label);
        costs.push(Cost {
            label: format!("{label} {:.2}MB", *width as f64 / 1e6),
            n,
            hi,
            ratio: med(&mut ratio),
            zero: med(&mut zero),
            us: 0.0,
            null_us: 0.0,
            own_us: 0.0,
            base: base_lo_label,
            mb: meta[li].0,
            grid: meta[li].1,
            wg: meta[li].1 .0 as u64 * meta[li].1 .1 as u64 * meta[li].1 .2 as u64,
        });
    }
    let base_lo = lo_of(&bases);
    let base_med = med(&mut bases.clone());

    let scale = wall_ms.min(base_lo);
    for c in costs.iter_mut() {
        let solve = |r: f64, s: f64| -> f64 {
            let den = c.n as f64 * ((c.hi - lo) as f64 - r * lo as f64);
            if den.abs() < 1e-9 {
                0.0
            } else {
                1e3 * r * s / den
            }
        };
        c.us = solve(c.ratio, scale);
        c.null_us = solve(c.zero, scale);
        c.own_us = solve(c.ratio, c.base);
    }
    let tok = scale;
    eprintln!(
        "\n{} classes, {steps} interleaved rounds of lo/hi/lo each (lo={lo}, hi sized per class to \
         add ~{target_ms:.1} ms of duplicated work), in {:.1}s",
        keys.len(),
        s0.elapsed().as_secs_f64()
    );
    let spread = 100.0 * (base_med - base_lo) / base_lo;
    eprintln!(
        "lo arm floors: {base_med:.3} ms median across classes, {base_lo:.3} ms overall, spread \
         {spread:.1}% -- that spread IS the contamination bound on every number below; costs \
         quoted against a {tok:.3} ms token"
    );

    assert!(
        spread < 20.0,
        "the lo arm moved {spread:.1}% between classes ({base_lo:.3} to {base_med:.3} ms) -- the \
         box moved under the instrument and every per-dispatch number in this run is a blend of \
         kernel and weather. Re-run; do not weaken this gate."
    );

    costs.sort_by(|x, y| {
        (y.us * y.n as f64)
            .partial_cmp(&(x.us * x.n as f64))
            .unwrap()
    });
    eprintln!("\n==== per-dispatch cost by replication slope ====");
    eprintln!(
        "{:<32} {:>4} {:>4} {:>8} {:>8} {:>8} {:>7} {:>9} {:>7}  grid",
        "label", "n", "hi", "us each", "null us", "ms/tok", "% tok", "class MB", "GB/s"
    );
    let mut accounted = 0.0;
    let mut accounted_own = 0.0;
    let mut streamed = 0.0;
    let mut small_n = 0usize;
    let mut unresolved_ms = 0.0;
    for c in &costs {
        let ms = c.us * c.n as f64 / 1e3;
        accounted += ms;
        accounted_own += c.own_us * c.n as f64 / 1e3;
        streamed += c.mb;
        if c.mb < 0.5 {
            small_n += c.n;
        }
        let resolved = c.us > 2.0 * c.null_us;
        if !resolved {
            unresolved_ms += ms.abs();
        }
        let rate = if c.mb > 0.0 && c.us > 0.0 {
            c.mb / (c.us * c.n as f64) * 1e3
        } else {
            0.0
        };
        eprintln!(
            "{:<32} {:>4} {:>4} {:>8.1} {:>8.1} {:>8.3} {:>6.1}% {:>9.2} {:>7.0}  {}x{}x{}{}",
            c.label,
            c.n,
            c.hi,
            c.us,
            c.null_us,
            ms,
            100.0 * ms / tok,
            c.mb,
            rate,
            c.grid.0,
            c.grid.1,
            c.grid.2,
            if resolved { "" } else { "  UNRESOLVED" }
        );
    }

    let sync = t_empty + readback.max(0.0);
    eprintln!(
        "\nclosure, each label against its own contemporaneous token: {accounted_own:.3} ms \
         dispatches + {sync:.3} ms host sync = {:.3} ms of a {base_med:.3} ms token ({:.0}%). \
         Rows that never cleared their own null contribute {unresolved_ms:.3} ms of that.",
        accounted_own + sync,
        100.0 * (accounted_own + sync) / base_med
    );

    let cheapest = costs
        .iter()
        .filter(|c| c.mb == 0.0 && c.us > 0.0 && c.us > 2.0 * c.null_us)
        .map(|c| c.us)
        .fold(f64::INFINITY, f64::min);
    eprintln!(
        "\n==== what a dispatch costs before it moves a byte ====\n\
         cheapest resolved zero-weight dispatch in the decode graph : {cheapest:.2} us\n\
         intercept of the independent flat-read fit                 : {fixed_us:.2} us\n\
         two instruments, one quantity; the budget uses the graph's own number."
    );

    assert!(
        cheapest.is_finite(),
        "no resolved zero-weight dispatch class; the replication instrument resolved nothing"
    );
    let ratio = (cheapest / fixed_us).max(fixed_us / cheapest);
    assert!(
        ratio < 2.5,
        "the flat-read fit says {fixed_us:.2} us/dispatch and the graph's cheapest real dispatch \
         says {cheapest:.2} us -- {ratio:.2}x apart. Two instruments that share no code path \
         disagree, which on this box is almost always contention inflating the arena arm (it runs \
         first, uncorrelated with the graph arms). Re-run; do not publish either number."
    );
    let floor_us = cheapest;

    let mut predicted = 0.0;
    for c in &costs {
        predicted += (floor_us * c.n as f64 + c.mb * 1e6 / (rate_gbs * 1e3)) / 1e3;
    }
    let fixed_total = floor_us * n_pass as f64 / 1e3;
    eprintln!(
        "\n==== is the token explained by bytes and dispatch count? ====\n\
         weight traffic attributed to dispatches      : {:.3} GB\n\
         floor at the flat 738.5 GB/s of the doc      : {:.3} ms  <- an upper bound on ambition, \
         not a target\n\
         floor at {floor_us:.1} us/dispatch + bytes/{rate_gbs:.0} GB/s over the REAL census of \
         {n_pass} dispatches: {predicted:.3} ms\n\
         measured token                               : {tok:.3} ms floor, {base_med:.3} ms \
         typical  ({:.0}% of that floor)",
        streamed / 1e3,
        streamed / 738.5,
        100.0 * predicted / tok
    );
    eprintln!(
        "the {small_n} of {n_pass} dispatches that stream no weight at all pay {:.3} ms of pure \
         per-dispatch cost, {:.0}% of the token and {:.0}% of the modelled floor; every dispatch \
         together pays {fixed_total:.3} ms ({:.0}% of the token)",
        small_n as f64 * floor_us / 1e3,
        100.0 * (small_n as f64 * floor_us / 1e3) / tok,
        100.0 * (small_n as f64 * floor_us / 1e3) / predicted,
        100.0 * fixed_total / tok
    );

    let bucket = |pred: &dyn Fn(&Cost) -> bool| -> f64 {
        costs
            .iter()
            .filter(|c| pred(c))
            .map(|c| c.us * c.n as f64 / 1e3)
            .sum()
    };
    let streaming = bucket(&|c: &Cost| c.mb > 0.5);
    let small = accounted - streaming;
    eprintln!(
        "\nshape: weight-streaming classes {streaming:.3} ms ({:.0}% of accounted), the other \
         {small_n} dispatches {small:.3} ms ({:.0}%), host {:.3} ms ({:.1}% of the token)",
        100.0 * streaming / accounted,
        100.0 * small / accounted,
        encode + sync,
        100.0 * (encode + sync) / tok
    );

    eprintln!("\n==== top terms by absolute microseconds ====");
    for c in costs.iter().take(6) {
        let ms = c.us * c.n as f64 / 1e3;
        let resolved = c.us > 2.0 * c.null_us;

        let floor = (floor_us * c.n as f64 + c.mb * 1e6 / (rate_gbs * 1e3)) / 1e3;

        eprintln!(
            "  {:<30} {:>3} x {:>7.1} us = {ms:>6.3} ms/tok, class floor {floor:.3} ms, headroom \
             {:.3} ms, {} wg/dispatch{}",
            c.label,
            c.n,
            c.us,
            (ms - floor).max(0.0),
            c.wg,
            if resolved {
                ""
            } else {
                "  UNRESOLVED (cost is not 2x its own null) -- do not plan against this row"
            }
        );
    }

    let group = |keys: &[&str]| -> (usize, f64, f64) {
        let sel: Vec<&Cost> = costs
            .iter()
            .filter(|c| keys.iter().any(|k| c.label.starts_with(k)))
            .collect();
        let n: usize = sel.iter().map(|c| c.n).sum();
        let ms: f64 = sel.iter().map(|c| c.us * c.n as f64 / 1e3).sum();
        let fl: f64 = sel
            .iter()
            .map(|c| (floor_us * c.n as f64 + c.mb * 1e6 / (rate_gbs * 1e3)) / 1e3)
            .sum();
        (n, ms, fl)
    };
    eprintln!("\n==== grouped by what one fix would have to touch ====");
    for (name, ks) in [
        (
            "AltUp/LAUREL norm chain (fused_norm_a/b/c)",
            &["fused_norm_a", "fused_norm_b", "fused_norm_c"][..],
        ),
        (
            "attention, everything but the projections",
            &["flash_stage1", "flash_stage2", "fused_attn"][..],
        ),
        (
            "w4 FFN projections",
            &["gemv_w4_sg16 [1280", "gemv_w4_sg16 [160x1x1] 13"][..],
        ),
        ("tied bf16 lm_head", &["gemv_bf16"][..]),
    ] {
        let (n, ms, fl) = group(ks);
        eprintln!(
            "  {name:<44} {n:>4} dispatches {ms:>6.3} ms/tok ({:>4.1}% of the token), floor \
             {fl:.3} ms, headroom {:.3} ms",
            100.0 * ms / tok,
            (ms - fl).max(0.0)
        );
    }

    let cfg = m.config();
    let layers = cfg.num_hidden_layers;
    let kv_layers = layers - cfg.num_kv_shared_layers;
    let count_of = |pfx: &str| -> usize {
        costs
            .iter()
            .filter(|c| c.label.starts_with(pfx))
            .map(|c| c.n)
            .sum()
    };
    assert_eq!(
        count_of("flash_stage1"),
        layers,
        "flash stage 1 does not run once per layer"
    );
    assert_eq!(
        count_of("fused_attn_k"),
        kv_layers,
        "the K projection does not run on exactly the {kv_layers} layers that compute KV; the \
         shared-KV structure this budget priced is gone"
    );
    assert!(
        (300..=900).contains(&n_pass),
        "pass count moved to {n_pass}; re-measure the budget before trusting these bounds"
    );
    assert!(
        encode + sync < 0.25 * tok,
        "host cost grew to {:.3} ms of a {tok:.3} ms token",
        encode + sync
    );
    assert!(
        accounted_own > 0.70 * base_med && accounted_own < 1.25 * base_med,
        "replication accounts for {accounted_own:.3} ms of a {base_med:.3} ms token -- the split \
         above is fiction if this does not close"
    );

    for c in costs.iter().filter(|c| c.mb > 1.0 && c.us > 0.0) {
        let rate = c.mb / (c.us * c.n as f64) * 1e3;
        assert!(
            rate < 900.0,
            "{} reads {rate:.0} GB/s over its {:.1} MB class; nothing on this box streams that \
             fast. Either the byte accounting is wrong, or the base token this cost was scaled \
             against was contended while the class itself was not -- check the {spread:.1}% lo-arm \
             spread above before touching the attribution",
            c.label,
            c.mb
        );
    }

    assert!(
        small_n * 2 > n_pass,
        "only {small_n} of {n_pass} dispatches stream no weights; the census moved, re-derive"
    );
}
