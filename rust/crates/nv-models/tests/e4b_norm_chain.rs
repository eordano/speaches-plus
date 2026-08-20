#![cfg(feature = "wgpu")]

mod common;
use common::med;
use common::snapshot_dir;
use nv_kernels::wgpu_backend::dispatch;
use nv_kernels::wgpu_backend::kernels as wk;
use nv_kernels::wgpu_backend::{compose, WgpuContext};
use nv_models::gemma4_e4b_wgpu::{fnc_unrolled_source, FNCU_ENTRY_A, FNCU_ENTRY_B, FNCU_ENTRY_C};
use std::time::Instant;

const HIDDEN: usize = 2560;
const EPS: f32 = 1.0e-6;
const SCALE: f32 = 0.70710678;

fn ctx() -> &'static WgpuContext {
    WgpuContext::shared().expect("wgpu adapter -- this suite measures a GPU and must not skip")
}

fn bf16_enc(x: f32) -> u16 {
    if x.is_nan() {
        return 0x7fc0;
    }
    let b = x.to_bits();
    let r = 0x7fffu32 + ((b >> 16) & 1);
    ((b + r) >> 16) as u16
}

fn bf16_dec(h: u16) -> f32 {
    f32::from_bits((h as u32) << 16)
}

struct Rng(u64);

impl Rng {
    fn next_u32(&mut self) -> u32 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        (x >> 32) as u32
    }

    fn normalish(&mut self) -> u16 {
        let a = (self.next_u32() >> 8) as f32 / (1u32 << 24) as f32;
        let b = (self.next_u32() >> 8) as f32 / (1u32 << 24) as f32;
        let r = (-2.0 * (a + 1e-7).ln()).sqrt();
        bf16_enc(r * (std::f32::consts::TAU * b).cos())
    }
    fn vec(&mut self, n: usize) -> Vec<u16> {
        (0..n).map(|_| self.normalish()).collect()
    }
}

fn pack(src: &[u16]) -> Vec<u32> {
    src.chunks(2)
        .map(|c| c[0] as u32 | ((c[1] as u32) << 16))
        .collect()
}

fn unpack(words: &[u32]) -> Vec<u16> {
    let mut out = Vec::with_capacity(words.len() * 2);
    for w in words {
        out.push((*w & 0xffff) as u16);
        out.push((*w >> 16) as u16);
    }
    out
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Kind {
    A,
    B,
    C,
}

struct Inputs {
    x: Vec<u16>,
    res: Vec<u16>,
    w1: Vec<u16>,
    w2: Vec<u16>,
}

impl Inputs {
    fn new(seed: u64, hidden: usize) -> Self {
        let mut r = Rng(seed);
        Self {
            x: r.vec(hidden),
            res: r.vec(hidden),
            w1: r.vec(hidden),
            w2: r.vec(hidden),
        }
    }
}

fn run(ctx: &'static WgpuContext, source: &str, entry: &str, kind: Kind, i: &Inputs) -> Vec<u16> {
    let hidden = i.x.len();
    let words = hidden / 2;

    let pl = dispatch::compute_pipeline_opts(ctx, "fnc-gate", source, entry, true)
        .unwrap_or_else(|e| panic!("compile {entry}: {e}"));
    let p = dispatch::uniform_from(
        ctx,
        "fnc-gate-p",
        &wk::fused_norm_chain::FncParams {
            hidden: hidden as u32,
            batch: 1,
            eps: EPS,
            words_per_row: words as u32,
            scale: SCALE,
            ..Default::default()
        },
    );
    let xb = dispatch::storage_from_slice(ctx, "fnc-gate-x", &pack(&i.x));
    let rb = dispatch::storage_from_slice(ctx, "fnc-gate-res", &pack(&i.res));
    let w1b = dispatch::storage_from_slice(ctx, "fnc-gate-w1", &pack(&i.w1));
    let w2b = dispatch::storage_from_slice(ctx, "fnc-gate-w2", &pack(&i.w2));
    let ob = dispatch::storage_zeroed(ctx, "fnc-gate-out", (words * 4) as u64);
    let o2b = dispatch::storage_zeroed(ctx, "fnc-gate-out2", (words * 4) as u64);
    let binds: Vec<(u32, &wgpu::Buffer)> = match kind {
        Kind::A => vec![(0, &xb), (1, &rb), (2, &w1b), (3, &w2b), (4, &ob), (5, &p)],
        Kind::B => vec![(0, &xb), (1, &rb), (2, &w1b), (4, &ob), (5, &p)],
        Kind::C => vec![
            (0, &xb),
            (1, &rb),
            (2, &w1b),
            (3, &w2b),
            (4, &ob),
            (5, &p),
            (6, &o2b),
        ],
    };
    let bind = dispatch::bind_group(ctx, &pl, &binds);
    let mut enc = ctx.device.create_command_encoder(&Default::default());
    {
        let mut cp = enc.begin_compute_pass(&Default::default());
        cp.set_pipeline(&pl);
        cp.set_bind_group(0, &bind, &[]);
        cp.dispatch_workgroups(1, 1, 1);
    }
    ctx.queue.submit([enc.finish()]);
    ctx.poll_blocking().expect("drain");
    let rd = |b: &wgpu::Buffer| -> Vec<u16> {
        unpack(&dispatch::read_back::<u32>(ctx, b, words).expect("read_back"))
    };
    match kind {
        Kind::A => [rd(&rb), rd(&ob)].concat(),
        Kind::B => rd(&ob),
        Kind::C => [rd(&ob), rd(&o2b)].concat(),
    }
}

fn reference(kind: Kind, i: &Inputs) -> Vec<f64> {
    let n = i.x.len();
    let dec = |v: &[u16]| -> Vec<f64> { v.iter().map(|h| bf16_dec(*h) as f64).collect() };
    let (x, res, w1, w2) = (dec(&i.x), dec(&i.res), dec(&i.w1), dec(&i.w2));
    let rnd = |v: f64| -> f64 { bf16_dec(bf16_enc(v as f32)) as f64 };
    let inv_rms = |v: &[f64]| -> f64 {
        let s: f64 = v.iter().map(|a| a * a).sum();
        1.0 / (EPS as f64 + s / n as f64).sqrt()
    };
    let s1 = inv_rms(&x);
    let t: Vec<f64> = (0..n).map(|k| rnd(x[k] * s1 * w1[k])).collect();
    match kind {
        Kind::A => {
            let nr: Vec<f64> = (0..n).map(|k| rnd(t[k] + res[k])).collect();
            let s2 = inv_rms(&nr);
            let out: Vec<f64> = (0..n).map(|k| rnd(nr[k] * s2 * w2[k])).collect();
            [nr, out].concat()
        }
        Kind::B => (0..n)
            .map(|k| rnd((res[k] + t[k]) * SCALE as f64))
            .collect(),
        Kind::C => {
            let o: Vec<f64> = (0..n)
                .map(|k| rnd((res[k] + t[k]) * SCALE as f64))
                .collect();
            let s2 = inv_rms(&o);
            let o2: Vec<f64> = (0..n).map(|k| rnd(o[k] * s2 * w2[k])).collect();
            [o, o2].concat()
        }
    }
}

fn shipped_source() -> String {
    compose(wk::fused_norm_chain::WGSL)
}

fn unrolled_source() -> String {
    compose(
        &fnc_unrolled_source(HIDDEN).expect("E4B hidden 2560 must qualify for the unrolled emit"),
    )
}

fn shipped_entry(kind: Kind) -> &'static str {
    match kind {
        Kind::A => wk::fused_norm_chain::ENTRY_RMS_RES_RMS,
        Kind::B => wk::fused_norm_chain::ENTRY_RES_OF_RMS,
        Kind::C => wk::fused_norm_chain::ENTRY_RMS_RES_RMS_NEXT,
    }
}

fn unrolled_entry(kind: Kind) -> &'static str {
    match kind {
        Kind::A => FNCU_ENTRY_A,
        Kind::B => FNCU_ENTRY_B,
        Kind::C => FNCU_ENTRY_C,
    }
}

fn assert_live(what: &str, v: &[u16]) {
    let nz = v.iter().filter(|h| bf16_dec(**h) != 0.0).count();
    let mut d: Vec<u16> = v.to_vec();
    d.sort_unstable();
    d.dedup();
    let peak = v.iter().map(|h| bf16_dec(*h).abs()).fold(0.0f32, f32::max);
    assert!(
        nz * 10 >= v.len() * 9 && d.len() >= 64 && peak > 1e-3,
        "{what} is degenerate: {nz}/{} nonzero, {} distinct patterns, peak {peak:e}. A numeric \
         gate over a quantity that underflows its output type compares zeros against zeros.",
        v.len(),
        d.len()
    );
}

#[test]
fn unrolled_norm_chain_is_bitwise_identical() {
    let ctx = ctx();
    let ship = shipped_source();
    let fast = unrolled_source();

    assert!(
        fnc_unrolled_source(2560).is_some() && fnc_unrolled_source(2048).is_some(),
        "hidden divisible by 512 must qualify"
    );
    for bad in [2560 - 2, 1280, 0, 768] {
        assert!(
            fnc_unrolled_source(bad).is_none(),
            "hidden {bad} does not divide into exact trip counts and must fall back"
        );
    }

    let sweep: usize = std::env::var("NV_E4B_FNC_SEEDS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(512);
    let mut cases: Vec<(Kind, u64)> = Vec::new();
    for k in [Kind::A, Kind::B, Kind::C] {
        for s in 0..sweep {
            cases.push((k, 0xa11ce + s as u64 * 0x9e37));
        }
    }
    for (kind, seed) in cases {
        let i = Inputs::new(seed, HIDDEN);
        assert_live(&format!("{kind:?} input x"), &i.x);
        let s = run(ctx, &ship, shipped_entry(kind), kind, &i);
        let f = run(ctx, &fast, unrolled_entry(kind), kind, &i);
        assert_live(&format!("{kind:?} shipped output"), &s);
        assert_eq!(
            s.len(),
            f.len(),
            "{kind:?}: the two entries wrote different amounts"
        );
        let diff = s.iter().zip(&f).filter(|(a, b)| a != b).count();
        if diff > 0 {
            for (k, (a, b)) in s.iter().zip(&f).enumerate() {
                if a != b {
                    eprintln!(
                        "  {kind:?} seed {seed:#x} word {k} (of {}): shipped {a:#06x} ({}) \
                         unrolled {b:#06x} ({})",
                        s.len(),
                        bf16_dec(*a),
                        bf16_dec(*b)
                    );
                }
            }
        }
        assert_eq!(
            diff,
            0,
            "{kind:?}: {diff} of {} bf16 words differ between the shipped and the unrolled norm \
             chain. The unrolled emit reorders no arithmetic, so any difference is a bug in the \
             emit, not a rounding question -- do not weaken this to a tolerance.",
            s.len()
        );

        let poisoned = fast.replace(
            "let mean = fncu_div_rn(sum, f32(fncu_params.hidden));",
            "let mean = fncu_div_rn(sum, f32(fncu_params.hidden)) * 1.05;",
        );
        assert_ne!(
            poisoned, fast,
            "the poison did not apply; the control below would be the arm"
        );
        let p = run(ctx, &poisoned, unrolled_entry(kind), kind, &i);
        let pdiff = p.iter().zip(&s).filter(|(a, b)| a != b).count();
        assert!(
            pdiff * 4 > s.len(),
            "{kind:?}: a 5% error in the reciprocal RMS moved only {pdiff} of {} words. The \
             comparator above is not reading what the kernel wrote.",
            s.len()
        );
    }
}

#[test]
fn norm_chain_matches_f64_reference() {
    let ctx = ctx();
    let ship = shipped_source();
    let fast = unrolled_source();

    let ulp = |w: f64| -> f64 {
        if w == 0.0 {
            f64::MIN_POSITIVE
        } else {
            w.abs().log2().floor().exp2() * 2.0f64.powi(-7)
        }
    };
    for (kind, seed) in [(Kind::A, 0x5eed1), (Kind::B, 0x5eed2), (Kind::C, 0x5eed3)] {
        let i = Inputs::new(seed, HIDDEN);
        let want = reference(kind, &i);
        let live = want.iter().filter(|v| **v != 0.0).count();
        let peak = want.iter().fold(0.0f64, |m, v| m.max(v.abs()));
        assert!(
            live * 10 >= want.len() * 9 && peak > 1e-3,
            "{kind:?}: the f64 reference is degenerate ({live}/{} nonzero, peak {peak:e})",
            want.len()
        );
        for (name, src, entry) in [
            ("shipped", &ship, shipped_entry(kind)),
            ("unrolled", &fast, unrolled_entry(kind)),
        ] {
            let got = run(ctx, src, entry, kind, &i);
            assert_eq!(got.len(), want.len());
            let mut exact = 0usize;
            let mut worst = 0.0f64;
            for (k, h) in got.iter().enumerate() {
                let g = bf16_dec(*h) as f64;
                let w = want[k];
                if g == w {
                    exact += 1;
                }
                let e = (g - w).abs() / ulp(w);
                if e > worst {
                    worst = e;
                }
            }
            assert!(
                worst <= 1.0,
                "{kind:?}/{name}: worst error {worst:.3} bf16 ulps against the f64 reference; \
                 {exact}/{} words were exact. More than one ulp is not f32-vs-f64 reduction \
                 noise, it is different arithmetic.",
                want.len()
            );
            assert!(
                exact * 100 >= want.len() * 97,
                "{kind:?}/{name}: only {exact}/{} words reproduce the f64 reference exactly. \
                 Rounding noise explains a few; this many means the two disagree on the \
                 arithmetic, not on the last bit.",
                want.len()
            );
        }
    }
}

const FLOOR_WGSL: &str = r#"
struct FlParams { a: u32, b: u32, c: u32, d: u32 };
@group(0) @binding(0) var<storage, read_write> fl_out: array<u32>;
@group(0) @binding(1) var<uniform> fl_p: FlParams;

@compute @workgroup_size(256)
fn fnc_floor(@builtin(local_invocation_id) tid: vec3<u32>) {
    // Never true. The compiler cannot prove it, so the entry survives; what it
    // costs is a dispatch of one workgroup that moves nothing.
    if (fl_p.a == 0xffffffffu) { fl_out[tid.x] = fl_p.b; }
}
"#;

struct Arm {
    name: String,
    pl: wgpu::ComputePipeline,
    binds: Vec<wgpu::BindGroup>,
}

fn pass_ms(ctx: &'static WgpuContext, arm: &Arm, n: usize, reps: usize) -> f64 {
    let mut xs = Vec::with_capacity(reps);
    for _ in 0..reps {
        let mut enc = ctx.device.create_command_encoder(&Default::default());
        {
            let mut cp = enc.begin_compute_pass(&Default::default());
            cp.set_pipeline(&arm.pl);
            for k in 0..n {
                cp.set_bind_group(0, &arm.binds[k % arm.binds.len()], &[]);
                cp.dispatch_workgroups(1, 1, 1);
            }
        }
        let t0 = Instant::now();
        ctx.queue.submit([enc.finish()]);
        ctx.poll_blocking().expect("drain");
        xs.push(t0.elapsed().as_secs_f64() * 1e3);
    }
    med(&mut xs)
}

#[test]
#[ignore = "times the GPU; set NV_E4B_NORM_GEOM=1"]
fn unrolled_norm_chain_dispatch_cost() {
    assert_eq!(
        std::env::var("NV_E4B_NORM_GEOM").ok().as_deref(),
        Some("1"),
        "set NV_E4B_NORM_GEOM=1 -- a silent skip here would report a pass"
    );
    let ctx = ctx();
    eprintln!("adapter: {}", ctx.summary());
    if let Ok(o) = std::process::Command::new("uptime").output() {
        eprintln!("host: {}", String::from_utf8_lossy(&o.stdout).trim());
    }
    let words = HIDDEN / 2;
    let layers = 42usize;
    let ship = shipped_source();
    let fast = unrolled_source();

    let mut r = Rng(0x9e3779b97f4a7c15);
    let mk_buf = |label: &str, v: &[u16]| dispatch::storage_from_slice(ctx, label, &pack(v));
    let xb = mk_buf("fnc-t-x", &r.vec(HIDDEN));
    let rb = mk_buf("fnc-t-res", &r.vec(HIDDEN));
    let ob = dispatch::storage_zeroed(ctx, "fnc-t-out", (words * 4) as u64);
    let o2b = dispatch::storage_zeroed(ctx, "fnc-t-out2", (words * 4) as u64);
    let w1: Vec<wgpu::Buffer> = (0..layers)
        .map(|_| mk_buf("fnc-t-w1", &r.vec(HIDDEN)))
        .collect();
    let w2: Vec<wgpu::Buffer> = (0..layers)
        .map(|_| mk_buf("fnc-t-w2", &r.vec(HIDDEN)))
        .collect();
    let p = dispatch::uniform_from(
        ctx,
        "fnc-t-p",
        &wk::fused_norm_chain::FncParams {
            hidden: HIDDEN as u32,
            batch: 1,
            eps: EPS,
            words_per_row: words as u32,
            scale: SCALE,
            ..Default::default()
        },
    );

    let arm = |name: &str, src: &str, entry: &str, kind: Kind| -> Arm {
        let pl = dispatch::compute_pipeline_opts(ctx, name, src, entry, true)
            .unwrap_or_else(|e| panic!("compile {entry}: {e}"));
        let binds = (0..layers)
            .map(|l| {
                let b: Vec<(u32, &wgpu::Buffer)> = match kind {
                    Kind::A => vec![
                        (0, &xb),
                        (1, &rb),
                        (2, &w1[l]),
                        (3, &w2[l]),
                        (4, &ob),
                        (5, &p),
                    ],
                    Kind::B => vec![(0, &xb), (1, &rb), (2, &w1[l]), (4, &ob), (5, &p)],
                    Kind::C => vec![
                        (0, &xb),
                        (1, &rb),
                        (2, &w1[l]),
                        (3, &w2[l]),
                        (4, &ob),
                        (5, &p),
                        (6, &o2b),
                    ],
                };
                dispatch::bind_group(ctx, &pl, &b)
            })
            .collect();
        Arm {
            name: name.to_string(),
            pl,
            binds,
        }
    };

    let floor_pl =
        dispatch::compute_pipeline_opts(ctx, "fnc-floor", FLOOR_WGSL, "fnc_floor", true).unwrap();
    let fp = dispatch::uniform_from(ctx, "fnc-floor-p", &[0u32, 0, 0, 0]);
    let floor = Arm {
        name: "dispatch floor (1 wg, no traffic)".into(),
        binds: vec![dispatch::bind_group(ctx, &floor_pl, &[(0, &ob), (1, &fp)])],
        pl: floor_pl,
    };

    let mut arms: Vec<Arm> = vec![floor];
    for (kind, tag) in [(Kind::A, "a"), (Kind::B, "b"), (Kind::C, "c")] {
        for rep in 1..=2 {
            arms.push(arm(
                &format!("ship_{tag}#{rep}"),
                &ship,
                shipped_entry(kind),
                kind,
            ));
            arms.push(arm(
                &format!("fast_{tag}#{rep}"),
                &fast,
                unrolled_entry(kind),
                kind,
            ));
        }
    }

    let lo = 32usize;
    let hi = 288usize;
    let rounds = 7usize;

    for a in &arms {
        pass_ms(ctx, a, lo, 2);
    }
    let mut slopes: Vec<Vec<f64>> = vec![Vec::new(); arms.len()];
    let mut nulls: Vec<Vec<f64>> = vec![Vec::new(); arms.len()];
    for _ in 0..rounds {
        for (i, a) in arms.iter().enumerate() {
            let t0 = pass_ms(ctx, a, lo, 3);
            let th = pass_ms(ctx, a, hi, 3);
            let t1 = pass_ms(ctx, a, lo, 3);
            let base = (t0 + t1) / 2.0;
            slopes[i].push((th - base) / (hi - lo) as f64 * 1e3);
            nulls[i].push((t0 - t1).abs() / (hi - lo) as f64 * 1e3);
        }
    }

    eprintln!(
        "\n==== per-dispatch cost, one workgroup, hidden {HIDDEN}, {lo}->{hi} dispatches per pass, \
         {rounds} interleaved rounds ===="
    );
    eprintln!("{:<34} {:>10} {:>10}", "arm", "us/disp", "null us");
    let mut us = Vec::with_capacity(arms.len());
    for (i, a) in arms.iter().enumerate() {
        let u = med(&mut slopes[i]);
        let n = med(&mut nulls[i]);
        eprintln!("{:<34} {u:>10.2} {n:>10.2}", a.name);
        us.push((u, n));
    }
    let floor_us = us[0].0;
    assert!(
        (1.0..=14.0).contains(&floor_us),
        "a one-workgroup dispatch that moves nothing fits at {floor_us:.2} us; the published law \
         is 3.9-4.3 us and outside 1-14 the instrument is wrong, not the kernel"
    );

    eprintln!(
        "\n{:<10} {:>10} {:>10} {:>9} {:>9} {:>12}",
        "entry", "shipped", "unrolled", "delta", "null", "over floor"
    );
    let mut total_ship = 0.0;
    let mut total_fast = 0.0;
    for (k, tag) in ["a", "b", "c"].iter().enumerate() {
        let idx = 1 + k * 4;
        let mut s: Vec<f64> = vec![us[idx].0, us[idx + 2].0];
        let mut f: Vec<f64> = vec![us[idx + 1].0, us[idx + 3].0];
        let sm = med(&mut s);
        let fm = med(&mut f);

        let within = ((s[1] - s[0]).abs()).max((f[1] - f[0]).abs());
        let null = within.max(us[idx].1).max(us[idx + 1].1) / sm * 100.0;
        eprintln!(
            "{:<10} {sm:>10.2} {fm:>10.2} {:>8.1}% {null:>8.1}% {:>11.2}",
            format!("fnc_{tag}"),
            100.0 * (fm - sm) / sm,
            sm - floor_us
        );
        total_ship += sm * 42.0;
        total_fast += fm * 42.0;
    }
    eprintln!(
        "\n126 dispatches per token: {:.3} ms shipped -> {:.3} ms unrolled ({:+.3} ms/token). The \
         budget's own replication measured this class at 1.626 ms against a 0.442 ms floor; this \
         harness is a different instrument and only its RATIO transfers.",
        total_ship / 1e3,
        total_fast / 1e3,
        (total_fast - total_ship) / 1e3
    );
}

const PROMPT: [u32; 8] = [2, 818, 3029, 529, 6081, 603, 563, 1596];

#[test]
#[ignore = "loads the E4B QAT checkpoint three times; set NV_E4B_NORM_E2E=1"]
fn unrolled_norm_chain_end_to_end() {
    assert_eq!(
        std::env::var("NV_E4B_NORM_E2E").ok().as_deref(),
        Some("1"),
        "set NV_E4B_NORM_E2E=1 -- a silent skip here would report a pass"
    );
    use nv_models::gemma4::Gemma4Config;
    use nv_models::gemma4_e4b_wgpu::Gemma4E4bWgpu;

    let ctx = ctx();
    eprintln!("adapter: {}", ctx.summary());
    if let Ok(o) = std::process::Command::new("uptime").output() {
        eprintln!("host: {}", String::from_utf8_lossy(&o.stdout).trim());
    }
    let dir = snapshot_dir();
    let config = Gemma4Config::from_hf_json_file(&dir.join("config.json")).expect("config");
    let loader =
        nv_weights::WeightLoader::open_dir(&dir, &candle_core::Device::Cpu).expect("safetensors");
    let max_seq = 512usize;

    let arm = std::env::var("NV_E4B_NORM_E2E_ARM").unwrap_or_else(|_| "abc".into());
    let build = |unroll: &str| -> Gemma4E4bWgpu {
        std::env::set_var("NV_E4B_WGPU_FNC_UNROLL", unroll);
        Gemma4E4bWgpu::from_loader(config.clone(), &loader, max_seq).expect("graph")
    };

    let mut ms = [build("0"), build(&arm), build("0")];
    let want = [
        Some([false; 3]),
        Some([arm.contains('a'), arm.contains('b'), arm.contains('c')]),
        Some([false; 3]),
    ];
    for (i, m) in ms.iter().enumerate() {
        assert_eq!(
            m.fnc_unrolled(),
            want[i],
            "arm {i} did not build the norm chain it was asked for; the env never reached the \
             builder and every number below would be one arm measured three times"
        );
    }
    eprintln!("B arm unrolls {:?}", want[1].unwrap());
    let n_pass = ms[0].pass_count();
    for m in ms.iter() {
        assert_eq!(
            m.pass_count(),
            n_pass,
            "the unrolled emit changed the dispatch count; this A/B would then be pricing graph \
             shape, not the kernel"
        );
    }
    eprintln!(
        "3 graphs, {n_pass} passes/token each, {:.3} GB weights/token",
        ms[0].weight_bytes_per_token() as f64 / 1e9
    );

    let steps_eq = 40usize;
    for m in ms.iter_mut() {
        m.reset();
    }
    let mut cur = 0u32;
    let mut first_ab: Option<usize> = None;
    let mut first_aa: Option<usize> = None;
    let mut worst_ab = 0.0f32;
    let mut worst_aa = 0.0f32;
    let mut ident_ab = 0usize;
    let mut ident_aa = 0usize;
    let mut distinct = std::collections::BTreeSet::new();
    for s in 0..PROMPT.len() + steps_eq {
        let feed = if s < PROMPT.len() { PROMPT[s] } else { cur };
        let mut lg: Vec<Vec<f32>> = Vec::with_capacity(3);
        let mut tk = [0u32; 3];
        for (i, m) in ms.iter_mut().enumerate() {
            let (t, l) = m.decode_step_logits(feed).expect("logits");
            tk[i] = t;
            lg.push(l);
        }
        cur = tk[0];
        distinct.insert(tk[0]);
        assert!(
            lg[0].iter().any(|v| v.is_finite() && *v != 0.0),
            "step {s}: the logit vector is all zero or non-finite; every comparison below would \
             be zeros against zeros"
        );
        let cmp = |x: &[f32], y: &[f32]| -> (bool, f32) {
            let mut w = 0.0f32;
            let mut same = true;
            for (a, b) in x.iter().zip(y) {
                if a.to_bits() != b.to_bits() {
                    same = false;
                }
                w = w.max((a - b).abs());
            }
            (same, w)
        };
        let (eq_ab, d_ab) = cmp(&lg[0], &lg[1]);
        let (eq_aa, d_aa) = cmp(&lg[0], &lg[2]);
        ident_ab += eq_ab as usize;
        ident_aa += eq_aa as usize;
        worst_ab = worst_ab.max(d_ab);
        worst_aa = worst_aa.max(d_aa);
        if !eq_ab && first_ab.is_none() {
            first_ab = Some(s);
        }
        if !eq_aa && first_aa.is_none() {
            first_aa = Some(s);
        }
    }
    let n_cmp = PROMPT.len() + steps_eq;
    eprintln!(
        "\nlogit agreement over {n_cmp} forced-token steps ({} distinct greedy tokens):\n  \
         unrolled B vs rolled A  : {ident_ab}/{n_cmp} bit-identical, first divergence {first_ab:?}, \
         worst |delta| {worst_ab:e}\n  \
         rolled  A' vs rolled A  : {ident_aa}/{n_cmp} bit-identical, first divergence {first_aa:?}, \
         worst |delta| {worst_aa:e}",
        distinct.len()
    );
    assert!(
        distinct.len() > 4,
        "the greedy continuation collapsed to {} distinct tokens; too degenerate to prove \
         anything about equality",
        distinct.len()
    );

    assert!(
        ident_ab >= ident_aa && worst_ab <= worst_aa.max(0.0),
        "the unrolled norm chain moved the logits further than the graph moves against a second \
         copy of itself: B/A {ident_ab} identical, worst {worst_ab:e}; A'/A {ident_aa} identical, \
         worst {worst_aa:e}. The emit reorders no arithmetic, so this is a bug in it."
    );

    let warm = 3usize;
    let steps = 10usize;
    let visit = |m: &mut Gemma4E4bWgpu, t: &mut u32| -> f64 {
        for _ in 0..warm {
            *t = m.decode_step(*t).expect("warm");
        }
        let mut xs = Vec::with_capacity(steps);
        for _ in 0..steps {
            let s = Instant::now();
            *t = m.decode_step(*t).expect("step");
            xs.push(s.elapsed().as_secs_f64() * 1e3);
        }
        med(&mut xs)
    };
    let rounds = 8usize;
    let mut samples: Vec<Vec<f64>> = vec![Vec::new(); 3];
    let mut toks = [0u32; 3];
    for m in ms.iter_mut() {
        m.reset();
    }
    for (i, m) in ms.iter_mut().enumerate() {
        let mut t = 0u32;
        for &p in &PROMPT {
            t = m.decode_step(p).expect("prompt");
        }
        toks[i] = t;
    }

    for r in 0..=rounds {
        for i in 0..3 {
            let mut t = toks[i];
            let v = visit(&mut ms[i], &mut t);
            toks[i] = t;
            if r > 0 {
                samples[i].push(v);
            }
            if ms[i].current_pos() + 2 * (warm + steps) >= max_seq {
                ms[i].reset();
                let mut tt = 0u32;
                for &p in &PROMPT {
                    tt = ms[i].decode_step(p).expect("prompt");
                }
                toks[i] = tt;
            }
        }
    }

    let mut ratios: Vec<f64> = Vec::with_capacity(rounds);
    let mut nulls: Vec<f64> = Vec::with_capacity(rounds);
    let mut bases: Vec<f64> = Vec::with_capacity(rounds);
    for r in 0..rounds {
        let (a, b, c) = (samples[0][r], samples[1][r], samples[2][r]);
        let base = (a + c) / 2.0;
        ratios.push(b / base - 1.0);
        nulls.push((a - c).abs() / base);
        bases.push(base);
    }
    let effect = 100.0 * med(&mut ratios.clone());
    let null = 100.0 * med(&mut nulls.clone());
    let base_med = med(&mut bases.clone());
    let base_lo = bases.iter().cloned().fold(f64::INFINITY, f64::min);
    eprintln!(
        "\n==== E4B decode token, {rounds} rounds of {steps} timed steps per arm, A/B/A' \
         interleaved inside each round ===="
    );
    for (i, name) in ["rolled   A ", "unrolled B ", "rolled   A'"]
        .iter()
        .enumerate()
    {
        let mut v = samples[i].clone();
        let m = med(&mut v);
        eprintln!(
            "  {name} : {m:.3} ms/tok ({:.1} tok/s)   per-round {:?}",
            1e3 / m,
            samples[i]
                .iter()
                .map(|x| format!("{x:.2}"))
                .collect::<Vec<_>>()
        );
    }
    eprintln!(
        "  rolled base: {base_med:.3} ms/tok median across rounds, {base_lo:.3} ms best round\n           effect {effect:+.2}% ({:+.3} ms/token at the median base, {:.1} -> {:.1} tok/s), null \
         A-vs-A' {null:.2}%",
        effect / 100.0 * base_med,
        1e3 / base_med,
        1e3 / (base_med * (1.0 + effect / 100.0))
    );

    assert!(
        null < 4.0,
        "the two identically-built arms disagree by {null:.2}% -- the box moved under the \
         instrument and the effect below is a blend of kernel and weather. Re-run; do not weaken \
         this gate."
    );
    assert!(
        effect < -null.max(1.0),
        "the unrolled norm chain measured {effect:+.2}% against a {null:.2}% null. It is not a \
         win here; do not default it on."
    );
}
