#![cfg(feature = "wgpu")]

use nv_kernels::wgpu_backend::{dispatch, WgpuContext};
use std::time::Instant;

#[repr(C)]
#[derive(Clone, Copy, Default, bytemuck::Pod, bytemuck::Zeroable)]
struct Params {
    n_rows: u32,
    k_words: u32,
    groups_x: u32,
    out_f32: u32,
    w_row_words: u32,
    x_off_words: u32,
    y_off_words: u32,
    pad0: u32,
    alpha: f32,
    pad1: u32,
    pad2: u32,
    pad3: u32,
}

#[derive(Clone, Copy)]
struct Arm {
    label: &'static str,
    entry: &'static str,
}

const ARMS: [Arm; 4] = [
    Arm {
        label: "scalar ",
        entry: "q3w_gemv_bf16",
    },
    Arm {
        label: "u4     ",
        entry: "q3w_gemv_bf16_u4",
    },
    Arm {
        label: "u8     ",
        entry: "q3w_gemv_bf16_u8",
    },
    Arm {
        label: "scalar'",
        entry: "q3w_gemv_bf16",
    },
];

const SHAPES: [(&str, usize, usize); 3] = [
    ("dn-inproj 12352x2048", 12352, 2048),
    ("dn-oproj   2048x4096", 2048, 4096),
    ("moe-router  257x2048", 257, 2048),
];

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    fn bf16s(&mut self, n: usize) -> Vec<u16> {
        (0..n)
            .map(|_| {
                let t = (self.next() >> 40) as f32 / 8388608.0 - 1.0;
                half::bf16::from_f32(t * 0.05).to_bits()
            })
            .collect()
    }
}

fn pack(src: &[u16]) -> Vec<u32> {
    src.chunks(2)
        .map(|c| c[0] as u32 | ((*c.get(1).unwrap_or(&0) as u32) << 16))
        .collect()
}

fn ctx() -> &'static WgpuContext {
    WgpuContext::shared().expect("no wgpu adapter -- this suite measures nothing without one")
}

fn pct(xs: &[f64], p: f64) -> f64 {
    let mut s = xs.to_vec();
    s.sort_by(|a, b| a.partial_cmp(b).unwrap());
    s[(((s.len() - 1) as f64) * p).round() as usize]
}

struct Case {
    w: Vec<u32>,
    x: Vec<u32>,
    n: usize,
    k: usize,
}

fn case(n: usize, k: usize, seed: u64) -> Case {
    let mut r = Rng(seed);
    Case {
        w: pack(&r.bf16s(n * k)),
        x: pack(&r.bf16s(k)),
        n,
        k,
    }
}

fn params(ctx: &WgpuContext, c: &Case, groups_x: u32) -> Params {
    let _ = ctx;
    Params {
        n_rows: c.n as u32,
        k_words: (c.k / 2) as u32,
        groups_x,
        out_f32: 1,
        w_row_words: (c.k / 2) as u32,
        alpha: 1.0,
        ..Default::default()
    }
}

#[test]
fn every_load_width_twin_is_bit_identical_to_the_scalar_baseline() {
    let ctx = ctx();
    let src = nv_models::qwen3_5_moe_wgpu::gemv_bf16_source();
    let mut failures = Vec::new();

    let plans: [(&str, usize, usize); 5] = [
        ("dn-inproj", 12352, 2048),
        ("dn-oproj", 2048, 4096),
        ("router", 257, 2048),
        ("ragged-k", 65, 1024 + 256),
        ("odd-n", 7, 2048),
    ];

    for (name, n, k) in plans {
        let c = case(n, k, 0x243f6a8885a308d3);
        let grid = ctx.device.limits().max_compute_workgroups_per_dimension;
        let pairs = n.div_ceil(2) as u32;
        let gx = pairs.min(grid);
        let gy = pairs.div_ceil(gx);
        let p = dispatch::uniform_from(ctx, "gb-p", &params(ctx, &c, gx));
        let x = dispatch::storage_from_slice(ctx, "gb-x", &c.x);
        let wn = dispatch::storage_from_slice(ctx, "gb-w", &c.w);

        let mut results: Vec<(&str, Vec<f32>)> = Vec::new();
        for a in ARMS {
            let y = dispatch::storage_zeroed(ctx, "gb-y", (n * 4) as u64);
            let pl = dispatch::cached_compute_pipeline(ctx, a.entry, &src, a.entry).unwrap();
            let bg = dispatch::bind_group(ctx, &pl, &[(0, &wn), (1, &x), (2, &p), (3, &y)]);
            let mut enc = ctx.device.create_command_encoder(&Default::default());
            {
                let mut pass = enc.begin_compute_pass(&Default::default());
                pass.set_pipeline(&pl);
                pass.set_bind_group(0, &bg, &[]);
                pass.dispatch_workgroups(gx, gy, 1);
            }
            ctx.queue.submit([enc.finish()]);
            results.push((a.label, dispatch::read_back(ctx, &y, n).unwrap()));
        }

        let (_, ref base) = results[0];
        let nonzero = base.iter().filter(|v| **v != 0.0).count();
        assert_eq!(
            nonzero, n,
            "{name}: the baseline produced zeros, so this case measures nothing"
        );
        for (label, out) in results.iter().skip(1) {
            let bad = base
                .iter()
                .zip(out.iter())
                .position(|(a, b)| a.to_bits() != b.to_bits());
            eprintln!(
                "{name:<10} {label} {}",
                if bad.is_none() {
                    "bit-identical"
                } else {
                    "DIFFER"
                }
            );
            if let Some(i) = bad {
                failures.push(format!("{name}/{label}: y[{i}] {} vs {}", base[i], out[i]));
            }
        }
    }
    assert!(
        failures.is_empty(),
        "load-width parity failures: {failures:#?}"
    );
}

#[test]
#[ignore = "kernel-rate suite; run alone, one per process"]
fn gemv_bf16_load_width_rate_against_its_own_null_control() {
    let ctx = ctx();
    let src = nv_models::qwen3_5_moe_wgpu::gemv_bf16_source();
    let iters: usize = std::env::var("NV_Q3_GEMV_ITERS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(60);

    let copies: usize = std::env::var("NV_Q3_GEMV_COPIES")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(6);

    eprintln!("adapter: {:?}", ctx.adapter.get_info().name);
    eprintln!("{copies} weight copies per arm, {iters} timed samples");

    for (name, n, k) in SHAPES {
        let c = case(n, k, 0x9e3779b97f4a7c15);
        let bytes = (n * k * 2) as f64;
        let grid = ctx.device.limits().max_compute_workgroups_per_dimension;
        let pairs = n.div_ceil(2) as u32;
        let gx = pairs.min(grid);
        let gy = pairs.div_ceil(gx);
        let p = dispatch::uniform_from(ctx, "gb-p", &params(ctx, &c, gx));
        let x = dispatch::storage_from_slice(ctx, "gb-x", &c.x);
        let y = dispatch::storage_zeroed(ctx, "gb-y", (n * 4) as u64);
        let wn: Vec<wgpu::Buffer> = (0..copies)
            .map(|_| dispatch::storage_from_slice(ctx, "gb-w", &c.w))
            .collect();

        println!();
        println!("{name}  ({:.1} MiB/dispatch)", bytes / (1024.0 * 1024.0));
        println!(
            "{:<9} {:>10} {:>10} {:>9} {:>10} {:>10} {:>9}",
            "arm", "1x med ms", "8x med ms", "1x p05", "us/disp", "GB/s", "vs base"
        );
        let pls: Vec<_> = ARMS
            .iter()
            .map(|a| dispatch::cached_compute_pipeline(ctx, a.entry, &src, a.entry).unwrap())
            .collect();
        let bgss: Vec<Vec<wgpu::BindGroup>> = ARMS
            .iter()
            .zip(&pls)
            .map(|(_a, pl)| {
                wn.iter()
                    .map(|w| dispatch::bind_group(ctx, pl, &[(0, w), (1, &x), (2, &p), (3, &y)]))
                    .collect()
            })
            .collect();

        let once = |i: usize, rounds: usize| -> f64 {
            let t0 = Instant::now();
            let mut enc = ctx.device.create_command_encoder(&Default::default());
            {
                let mut pass = enc.begin_compute_pass(&Default::default());
                pass.set_pipeline(&pls[i]);
                for _ in 0..rounds {
                    for bg in &bgss[i] {
                        pass.set_bind_group(0, bg, &[]);
                        pass.dispatch_workgroups(gx, gy, 1);
                    }
                }
            }
            ctx.queue.submit([enc.finish()]);
            ctx.device
                .poll(wgpu::PollType::wait_indefinitely())
                .unwrap();
            t0.elapsed().as_secs_f64() * 1e3
        };

        let mut w1 = vec![Vec::with_capacity(iters); ARMS.len()];
        let mut w8 = vec![Vec::with_capacity(iters); ARMS.len()];
        for it in 0..iters + 5 {
            for i in 0..ARMS.len() {
                let a = once(i, 1);
                let b = once(i, 8);
                if it >= 5 {
                    w1[i].push(a);
                    w8[i].push(b);
                }
            }
        }

        let mut base_us: Option<f64> = None;
        for (i, a) in ARMS.iter().enumerate() {
            let m1 = pct(&w1[i], 0.5);
            let m8 = pct(&w8[i], 0.5);
            let per = (m8 - m1) / (7.0 * copies as f64) * 1e3;
            let gbs = bytes / (per * 1e-6) / 1e9;
            let vs = base_us.map(|b| b / per).unwrap_or(1.0);
            if base_us.is_none() {
                base_us = Some(per);
            }
            println!(
                "{:<9} {m1:>10.3} {m8:>10.3} {:>9.3} {per:>10.3} {gbs:>10.1} {vs:>8.3}x",
                a.label,
                pct(&w1[i], 0.05)
            );
            assert!(per > 0.0, "{name}/{}: non-positive slope", a.label);
        }
    }
}

#[test]
#[ignore = "holds two ~20 GB graphs resident; set NV_QWEN36_GEMV_LOAD_TEST=1"]
fn real_checkpoint_decode_is_bit_identical_across_load_width_arms() {
    use nv_models::qwen3_5_moe::Qwen3MoeConfig;
    use nv_models::qwen3_5_moe_wgpu as q3w;

    if std::env::var("NV_QWEN36_GEMV_LOAD_TEST").is_err() {
        panic!("set NV_QWEN36_GEMV_LOAD_TEST=1; a silent skip here would report a pass");
    }
    let dir = std::env::var("NV_QWEN36_DIR").unwrap_or_else(|_| {
        format!(
            "{}/.cache/huggingface/hub/models--RedHatAI--Qwen3.6-35B-A3B-NVFP4/snapshots/965bfb0e24d08e295cd641a15c7f231554078d0d",
            std::env::var("HOME").unwrap_or_default()
        )
    });
    let dir = std::path::PathBuf::from(dir);
    let cfg = Qwen3MoeConfig::from_hf_json_file(&dir.join("config.json")).expect("config");
    let loader = nv_weights::WeightLoader::open_dir(&dir, &candle_core::Device::Cpu)
        .expect("open safetensors");
    let tokens: [u32; 6] = [785, 6722, 315, 9625, 374, 279];
    const VAR: &str = "NV_Q3_WGPU_GEMV_BF16_LOAD";

    let max_seq: usize = std::env::var("NV_QWEN36_MAXSEQ")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(128);
    let build = |v: &str| {
        std::env::set_var(VAR, v);
        let t0 = Instant::now();
        let m = q3w::Qwen3MoeWgpu::from_loader(cfg.clone(), &loader, max_seq).expect("build");
        let (un, all) = m.gemv_bf16_unrolled_matrices();
        eprintln!(
            "arm {v:<6} built in {:.1}s, {} passes/token, {un}/{all} bf16 matrices unrolled",
            t0.elapsed().as_secs_f64(),
            m.pass_count()
        );
        assert_eq!(
            un > 0,
            v != "scalar",
            "{VAR}={v} did not reach the builder: {un}/{all} unrolled"
        );
        m
    };
    let mut arms = vec![("scalar", build("scalar")), ("u8", build("u8"))];

    if std::env::var("NV_Q3_GEMV_NULL").is_ok() {
        arms.push(("scalar'", build("scalar")));
    }
    std::env::remove_var(VAR);
    for a in &arms[1..] {
        assert_eq!(
            arms[0].1.pass_count(),
            a.1.pass_count(),
            "a load-count twin must not change the dispatch count"
        );
    }

    let mut logits: Vec<Vec<Vec<f32>>> = Vec::new();
    for (_, m) in arms.iter_mut() {
        let mut got = Vec::new();
        for t in tokens {
            got.push(m.decode_step_logits(t).expect("decode step").1);
        }
        got.iter()
            .for_each(|l| assert!(l.iter().any(|v| *v != 0.0)));
        logits.push(got);
    }
    let bad = logits[0]
        .iter()
        .zip(logits[1].iter())
        .enumerate()
        .find_map(|(s, (x, y))| {
            x.iter()
                .zip(y.iter())
                .position(|(p, q)| p.to_bits() != q.to_bits())
                .map(|i| (s, i, x[i], y[i]))
        });
    assert!(
        bad.is_none(),
        "u8 is not bit-identical to scalar on the real checkpoint: {bad:?}"
    );
    if let Some(null) = logits.get(2) {
        assert!(
            logits[0].iter().zip(null.iter()).all(|(x, y)| x
                .iter()
                .zip(y.iter())
                .all(|(p, q)| p.to_bits() == q.to_bits())),
            "the null arm disagrees with itself, so nothing here is a comparison"
        );
    }
    eprintln!(
        "whole-graph decode bit-identical over {} steps x {} logits",
        tokens.len(),
        logits[0][0].len()
    );

    let rounds: usize = std::env::var("NV_Q3_GEMV_ROUNDS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(40);
    let mut ms = vec![Vec::with_capacity(rounds); arms.len()];
    for (_, m) in arms.iter_mut() {
        m.reset().expect("reset");
    }
    for r in 0..rounds + 4 {
        for (i, (_, m)) in arms.iter_mut().enumerate() {
            let t0 = Instant::now();
            m.decode_step(785).expect("decode step");
            if r >= 4 {
                ms[i].push(t0.elapsed().as_secs_f64() * 1e3);
            }
        }
    }
    println!("max_seq {max_seq}, {rounds} timed steps per arm, round-robin");
    let base = pct(&ms[0], 0.5);
    for (i, (label, _)) in arms.iter().enumerate() {
        let m = pct(&ms[i], 0.5);
        println!(
            "{label:<7} {m:>8.3} ms/tok  {:>7.1} tok/s  {:>6.3}x  (p05 {:.3} p95 {:.3})",
            1e3 / m,
            base / m,
            pct(&ms[i], 0.05),
            pct(&ms[i], 0.95)
        );
    }
}
