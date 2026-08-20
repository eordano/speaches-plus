#![cfg(feature = "wgpu")]

mod common;
use common::env_usize;
use common::med;
use nv_models::qwen3_5_moe::Qwen3MoeConfig;
use nv_models::qwen3_5_moe_wgpu as q3w;
use std::time::Instant;

#[test]
fn moe_bookkeeping_msl_shape() {
    let want = [
        "q3w_router_topk",
        "q3w_router_topk_par",
        "q3w_quant_rows",
        "q3w_silu_mul_quant",
        "q3w_silu_mul",
        "q3w_moe_combine",
    ];
    let mut seen = 0usize;
    for (tag, src) in q3w::nozi_audit_sources() {
        let module = naga::front::wgsl::parse_str(&src)
            .unwrap_or_else(|e| panic!("{tag}: wgsl parse: {}", e.message()));
        let info = naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::all(),
        )
        .validate(&module)
        .unwrap_or_else(|e| panic!("{tag}: validate: {e}"));
        let msl = naga::back::msl::write_string(
            &module,
            &info,
            &naga::back::msl::Options {
                lang_version: (3, 0),
                ..Default::default()
            },
            &naga::back::msl::PipelineOptions::default(),
        )
        .unwrap_or_else(|e| panic!("{tag}: msl-out: {e}"))
        .0;
        if let Ok(d) = std::env::var("NV_QWEN36_MSL_DIR") {
            let p = std::path::Path::new(&d).join(format!("{}.metal", tag.replace(':', "_")));
            std::fs::write(&p, &msl).expect("write msl");
        }
        let mut i = 0usize;
        while let Some(at) = msl[i..].find("kernel void ") {
            let start = i + at;
            let rest = &msl[start..];
            let name_end = rest.find('(').expect("kernel signature");
            let name = rest["kernel void ".len()..name_end]
                .trim()
                .trim_end_matches('_')
                .to_string();
            let end = rest.find("\n}\n").unwrap_or(rest.len());
            if want.contains(&name.as_str()) {
                seen += 1;
                let mut tg: Vec<String> = Vec::new();
                let mut spill: Vec<String> = Vec::new();
                for l in rest[..end].lines() {
                    let t = l.trim().trim_start_matches(", ");
                    if let Some(d) = t.strip_prefix("threadgroup ") {
                        tg.push(d.trim_end_matches(',').to_string());
                    }
                    if let Some((h, tail)) = t.split_once(' ') {
                        if h.starts_with("type_") && tail.ends_with(" = {};") {
                            spill.push(format!("{h} {}", tail.trim_end_matches(" = {};")));
                        }
                    }
                }
                eprintln!("{tag}::{name}\n  threadgroup {tg:?}\n  thread-space arrays {spill:?}");
            }
            i = start + name_end;
        }
    }
    assert!(
        seen >= want.len(),
        "only {seen} of the wanted entries found"
    );
}

mod bookkeeping_rate {
    use nv_kernels::wgpu_backend::{dispatch, WgpuContext};
    use std::time::Instant;

    #[repr(C)]
    #[derive(Clone, Copy, Default, bytemuck::Pod, bytemuck::Zeroable)]
    struct RtParams {
        n_experts: u32,
        k: u32,
        shared_slot: u32,
        pad1: u32,
    }

    fn med(xs: &mut [f64]) -> f64 {
        xs.sort_by(|a, b| a.partial_cmp(b).unwrap());
        xs[xs.len() / 2]
    }

    struct Arm {
        entry: &'static str,
        us: f64,
        null_us: f64,
    }

    fn burst(
        ctx: &WgpuContext,
        pl: &wgpu::ComputePipeline,
        bg: &wgpu::BindGroup,
        copies: usize,
    ) -> f64 {
        let mut enc = ctx.device.create_command_encoder(&Default::default());
        {
            let mut pass = enc.begin_compute_pass(&Default::default());
            pass.set_pipeline(pl);
            pass.set_bind_group(0, bg, &[]);
            for _ in 0..copies {
                pass.dispatch_workgroups(1, 1, 1);
            }
        }
        let s = Instant::now();
        ctx.queue.submit([enc.finish()]);
        let _ = ctx.device.poll(wgpu::PollType::wait_indefinitely());
        s.elapsed().as_secs_f64() * 1e3
    }

    #[test]
    #[ignore = "needs a GPU; set NV_QWEN36_TOPK_RATE=1"]
    fn router_topk_cost_per_dispatch() {
        assert_eq!(
            std::env::var("NV_QWEN36_TOPK_RATE").ok().as_deref(),
            Some("1"),
            "set NV_QWEN36_TOPK_RATE=1 -- a silent skip here would report a pass"
        );
        let ctx = WgpuContext::shared().expect("wgpu adapter");
        eprintln!("adapter: {}", ctx.summary());
        if let Ok(o) = std::process::Command::new("uptime").output() {
            eprintln!("host: {}", String::from_utf8_lossy(&o.stdout).trim());
        }

        let mut seed = 0x9e3779b97f4a7c15u64;
        let logits: Vec<u32> = (0..256)
            .map(|_| {
                seed ^= seed << 13;
                seed ^= seed >> 7;
                seed ^= seed << 17;
                ((seed >> 40) as f32 / 8192.0 - 1.0).to_bits()
            })
            .collect();

        let src = nv_models::qwen3_5_moe_wgpu::moe_source();
        let lg = dispatch::storage_from_slice(ctx, "rt-logits", &logits);
        let ids = dispatch::storage_zeroed(ctx, "rt-ids", 17 * 4);
        let w = dispatch::storage_zeroed(ctx, "rt-w", 16 * 4);

        let cells = [
            (256usize, 8usize),
            (256, 4),
            (256, 2),
            (256, 1),
            (128, 8),
            (64, 8),
            (32, 8),
            (1, 1),
        ];
        let entries = [
            "q3w_router_topk_tree",
            "q3w_router_topk_r4",
            "q3w_router_topk_par",
            "q3w_router_topk_r16",
        ];

        let (lo, hi) = (8usize, 64usize);
        let reps = super::env_usize("NV_QWEN36_TOPK_REPS", 24);
        eprint!(
            "\n==== router top-k, us per dispatch (slope of {lo} vs {hi} copies in one pass, \
             median of {reps} interleaved reps) ====\n{:<10}",
            "n / k"
        );
        for e in entries {
            eprint!(" {:>13}", e.trim_start_matches("q3w_router_topk_"));
        }
        eprintln!(" {:>8}", "null us");
        let mut served: Vec<f64> = Vec::new();
        let mut floor = f64::INFINITY;
        for (n, k) in cells {
            let p = dispatch::uniform_from(
                ctx,
                "rt-p",
                &RtParams {
                    n_experts: n as u32,
                    k: k as u32,
                    shared_slot: 1,
                    pad1: 0,
                },
            );
            let built: Vec<_> = entries
                .iter()
                .map(|e| {
                    let pl = dispatch::cached_compute_pipeline(ctx, e, &src, e)
                        .unwrap_or_else(|err| panic!("{e}: {err}"));
                    let bg =
                        dispatch::bind_group(ctx, &pl, &[(0, &lg), (1, &ids), (2, &w), (3, &p)]);
                    (*e, pl, bg)
                })
                .collect();

            let want = {
                let pl = dispatch::cached_compute_pipeline(
                    ctx,
                    "q3w_router_topk",
                    &src,
                    "q3w_router_topk",
                )
                .expect("serial pipeline");
                let bg = dispatch::bind_group(ctx, &pl, &[(0, &lg), (1, &ids), (2, &w), (3, &p)]);
                burst(ctx, &pl, &bg, 1);
                (
                    dispatch::read_back::<u32>(ctx, &ids, k + 1).unwrap(),
                    dispatch::read_back::<f32>(ctx, &w, k).unwrap(),
                )
            };
            for (e, pl, bg) in &built {
                ctx.queue
                    .write_buffer(&ids, 0, bytemuck::cast_slice(&[0u32; 17]));
                ctx.queue
                    .write_buffer(&w, 0, bytemuck::cast_slice(&[0u32; 16]));
                burst(ctx, pl, bg, 1);
                let got_ids = dispatch::read_back::<u32>(ctx, &ids, k + 1).unwrap();
                let got_w = dispatch::read_back::<f32>(ctx, &w, k).unwrap();
                assert_eq!(
                    got_ids, want.0,
                    "{e} at n={n} k={k}: ids differ from serial"
                );
                assert!(
                    got_w
                        .iter()
                        .zip(want.1.iter())
                        .all(|(a, b)| a.to_bits() == b.to_bits()),
                    "{e} at n={n} k={k}: weights differ from serial: {got_w:?} vs {:?}",
                    want.1
                );
            }

            for (_, pl, bg) in &built {
                for _ in 0..4 {
                    burst(ctx, pl, bg, hi);
                }
            }
            let mut d: Vec<Vec<f64>> = vec![Vec::new(); built.len()];
            let mut z: Vec<Vec<f64>> = vec![Vec::new(); built.len()];
            for _ in 0..reps {
                for (i, (_, pl, bg)) in built.iter().enumerate() {
                    let a = burst(ctx, pl, bg, lo);
                    let b = burst(ctx, pl, bg, hi);
                    let a2 = burst(ctx, pl, bg, lo);
                    d[i].push(1e3 * (b - (a + a2) / 2.0) / (hi - lo) as f64);
                    z[i].push(1e3 * (a - a2).abs() / (hi - lo) as f64);
                }
            }
            let arms: Vec<Arm> = built
                .iter()
                .enumerate()
                .map(|(i, (e, _, _))| Arm {
                    entry: e,
                    us: med(&mut d[i]),
                    null_us: med(&mut z[i]),
                })
                .collect();
            let null = arms.iter().map(|a| a.null_us).fold(0.0, f64::max);
            eprint!("{:<10}", format!("{n} / {k}"));
            for a in &arms {
                eprint!(" {:>13.2}", a.us);
            }
            eprintln!(
                " {:>8.2}{}",
                null,
                if arms.iter().all(|a| a.us > 2.0 * a.null_us) {
                    ""
                } else {
                    "   UNRESOLVED -- do not quote"
                }
            );
            for a in &arms {
                assert!(
                    a.us > 2.0 * a.null_us,
                    "{} at n={n} k={k} is not resolved against its own null ({:.2} vs {:.2} us) \
                     -- the box moved inside a rep; re-run rather than quoting this",
                    a.entry,
                    a.us,
                    a.null_us
                );
                floor = floor.min(a.us);
            }
            if (n, k) == (256, 8) {
                served = arms.iter().map(|a| a.us).collect();
            }
        }
        assert_eq!(served.len(), entries.len(), "served cell");
        let base = served[0];
        eprintln!("\nserved shape n=256 k=8, 40 dispatches per token:");
        for (e, us) in entries.iter().zip(served.iter()) {
            eprintln!(
                "  {:<24} {us:>7.2} us   {:>6.3}x tree   {:>+7.3} ms/token",
                e,
                us / base,
                40.0 * (us - base) / 1e3
            );
        }
        eprintln!(
            "cheapest cell anywhere in the sweep {floor:.2} us -- that is what one workgroup \
             costs before it computes anything, and no top-k goes under it."
        );

        let ship = served[entries.iter().position(|e| e.ends_with("_par")).unwrap()];
        assert!(
            ship < 0.75 * base,
            "the shipped rank top-k ({ship:.2} us) is no longer clear of the barrier-ladder arm \
             ({base:.2} us); re-derive the router budget before trusting the census"
        );
        let w4 = served[entries.iter().position(|e| e.ends_with("_r4")).unwrap()];
        let w16 = served[entries.iter().position(|e| e.ends_with("_r16")).unwrap()];
        eprintln!(
            "unroll width 4 / 8 / 16 -> {w4:.2} / {ship:.2} / {w16:.2} us. Width 8 ships because \
             the curve is flat past it ({:.3}x from 8 to 16); if that stops holding, refit \
             `router_rank_entries()` rather than assuming.",
            w16 / ship
        );
    }
}

fn snapshot_dir() -> std::path::PathBuf {
    match std::env::var("NV_QWEN36_DIR") {
        Ok(d) => std::path::PathBuf::from(d),
        Err(_) => std::path::PathBuf::from(format!(
            "{}/.cache/huggingface/hub/models--RedHatAI--Qwen3.6-35B-A3B-NVFP4/snapshots/965bfb0e24d08e295cd641a15c7f231554078d0d",
            std::env::var("HOME").unwrap_or_default()
        )),
    }
}

fn wall(m: &mut q3w::Qwen3MoeWgpu, seed: u32, warm: usize, steps: usize) -> (f64, f64) {
    m.reset().expect("reset");
    let mut t = seed;
    for _ in 0..warm {
        t = m.decode_step(t).expect("warm");
    }
    let mut xs = Vec::with_capacity(steps);
    for _ in 0..steps {
        let s = Instant::now();
        t = m.decode_step(t).expect("step");
        xs.push(s.elapsed().as_secs_f64() * 1e3);
    }
    let lo = xs.iter().cloned().fold(f64::INFINITY, f64::min);
    (lo, med(&mut xs))
}

struct Cost {
    label: String,
    n: usize,
    ratio: f64,
    zero: f64,
    us: f64,
    null_us: f64,
    mb: f64,
    grid: (u32, u32, u32),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum WidenVerdict {

    DepthBound,

    ThroughputBound,

    Inconclusive,
}

const WIDEN_MIN_WORK_SPAN: f64 = 4.0;

const WIDEN_FLAT: f64 = 1.5;

fn widen_verdict(pts: &[(f64, f64)]) -> WidenVerdict {
    if pts.len() < 2 {
        return WidenVerdict::Inconclusive;
    }
    assert!(
        pts.iter()
            .all(|(us, w)| us.is_finite() && *us > 0.0 && w.is_finite() && *w > 0.0),
        "widen_verdict needs positive finite (us, work) points, got {pts:?}"
    );
    let (lo, hi) = pts.iter().fold((pts[0], pts[0]), |(lo, hi), p| {
        (
            if p.1 < lo.1 { *p } else { lo },
            if p.1 > hi.1 { *p } else { hi },
        )
    });
    let work_span = hi.1 / lo.1;
    if work_span < WIDEN_MIN_WORK_SPAN {
        return WidenVerdict::Inconclusive;
    }
    let us_hi = pts.iter().map(|p| p.0).fold(0.0, f64::max);
    let us_lo = pts.iter().map(|p| p.0).fold(f64::INFINITY, f64::min);
    if us_hi < WIDEN_FLAT * us_lo {
        return WidenVerdict::DepthBound;
    }
    if hi.0 / lo.0 >= 0.5 * work_span {
        return WidenVerdict::ThroughputBound;
    }
    WidenVerdict::Inconclusive
}

#[test]
fn widen_verdict_reports_inconclusive_rather_than_guessing() {
    assert_eq!(
        widen_verdict(&[(27.50, 1.0), (26.31, 72.0)]),
        WidenVerdict::DepthBound,
        "flat cost across a 72x work span is the depth-bound signature"
    );
    assert_eq!(
        widen_verdict(&[(7.0, 1.0), (400.0, 72.0)]),
        WidenVerdict::ThroughputBound
    );
    assert_eq!(
        widen_verdict(&[(9.4, 32.0), (14.3, 72.0)]),
        WidenVerdict::Inconclusive,
        "a 2.25x work span cannot separate depth from throughput"
    );
    assert_eq!(
        widen_verdict(&[(10.0, 1.0), (20.0, 72.0)]),
        WidenVerdict::Inconclusive,
        "2x the cost for 72x the work is neither flat nor proportional"
    );
    assert_eq!(widen_verdict(&[(14.3, 72.0)]), WidenVerdict::Inconclusive);
    assert_eq!(widen_verdict(&[]), WidenVerdict::Inconclusive);
}

#[test]
#[should_panic(expected = "positive finite")]
fn widen_verdict_panics_on_a_zero_work_point() {
    let _ = widen_verdict(&[(9.4, 0.0), (14.3, 72.0)]);
}

#[test]
#[ignore = "loads the 24 GB checkpoint and replicates every pass class; set NV_QWEN36_BUDGET=1"]
fn qwen3_6_a3b_decode_pass_budget() {
    assert_eq!(
        std::env::var("NV_QWEN36_BUDGET").ok().as_deref(),
        Some("1"),
        "set NV_QWEN36_BUDGET=1 -- a silent skip here would report a pass"
    );
    let ctx = nv_kernels::wgpu_backend::WgpuContext::shared().expect("wgpu adapter");
    eprintln!("adapter: {}", ctx.summary());
    if let Ok(o) = std::process::Command::new("uptime").output() {
        eprintln!("host: {}", String::from_utf8_lossy(&o.stdout).trim());
    }

    let dir = snapshot_dir();
    assert!(
        dir.join("config.json").is_file(),
        "checkpoint missing at {}",
        dir.display()
    );
    let mut cfg = Qwen3MoeConfig::from_hf_json_file(&dir.join("config.json")).expect("config");
    let full_layers = cfg.num_hidden_layers;
    let (hidden, inter) = (cfg.hidden_size, cfg.moe_intermediate_size);
    let layers = env_usize("NV_QWEN36_BUDGET_LAYERS", full_layers);
    assert!(layers > 0 && layers <= full_layers);
    cfg.num_hidden_layers = layers;
    cfg.layer_types.truncate(layers);

    let slot_frac =
        (cfg.num_experts_per_tok + 1) as f64 / cfg.num_experts as f64 * (1.0 + 1.0 / 16.0);
    let streamed_mb = move |label: &str, widest: u64| -> f64 {
        let mb = widest as f64 / 1e6;
        if label.contains("moe-gateup") || label.contains("moe-down") {
            mb * slot_frac
        } else if label.contains("gather") {
            0.0
        } else if label.contains("gemv") || label.contains("lmhead") {
            mb
        } else {
            0.0
        }
    };

    let loader = nv_weights::WeightLoader::open_dir(&dir, &candle_core::Device::Cpu)
        .expect("open safetensors");
    let max_seq = env_usize("NV_QWEN36_BUDGET_SEQ", 512);
    let t0 = Instant::now();
    let mut m = q3w::Qwen3MoeWgpu::from_loader(cfg, &loader, max_seq).expect("build graph");
    let n_pass = m.pass_count();
    let head = m.head_pass_start();
    eprintln!(
        "built in {:.1}s -- {layers}/{full_layers} layers, {n_pass} passes/token, head tail starts at {head}",
        t0.elapsed().as_secs_f64()
    );

    let seed = 9419u32;
    let warm = env_usize("NV_QWEN36_BUDGET_WARM", 4);
    let steps = env_usize("NV_QWEN36_BUDGET_STEPS", 10);
    assert!(
        warm + 3 * steps < max_seq,
        "{warm} warm + 3x{steps} rounds overruns a {max_seq}-slot kv cache"
    );

    let (wall_ms, wall_med) = wall(&mut m, seed, 8, 24);
    let (wall_null, _) = wall(&mut m, seed, 8, 24);
    let pos = m.current_pos();
    m.probe_at(seed, pos.min(max_seq - 1));
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
    let t_graph = probe(&|| m.probe_prefix(n_pass).expect("graph"), 12);
    let e_empty = probe(&|| m.probe_encode(0).expect("enc"), 500);
    let e_graph = probe(&|| m.probe_encode(n_pass).expect("enc"), 500);
    let encode = e_graph - e_empty;

    eprintln!("\n==== envelope for one decode token ====");
    eprintln!(
        "  measured decode_step wall        : {wall_ms:.3} ms floor ({:.1} tok/s), {wall_med:.3} ms median   null arm floor {wall_null:.3} ms",
        1e3 / wall_ms
    );
    eprintln!(
        "  submit an empty graph and drain   : {t_empty:.3} ms  ({:.2}%)",
        100.0 * t_empty / wall_ms
    );
    eprintln!(
        "  host encode of {n_pass} dispatches   : {encode:.3} ms  ({:.2}%)   {:.2} us/dispatch",
        100.0 * encode / wall_ms,
        1e3 * encode / n_pass as f64
    );
    eprintln!(
        "  GPU, whole graph, no readback     : {:.3} ms  ({:.1}%)",
        t_graph - t_empty - encode,
        100.0 * (t_graph - t_empty - encode) / wall_ms
    );
    eprintln!(
        "  readback + map + host tail        : {:.3} ms  ({:.2}%)",
        wall_ms - t_graph,
        100.0 * (wall_ms - t_graph) / wall_ms
    );
    if t_graph > wall_ms {
        eprintln!(
            "  !! ENVELOPE DOES NOT CLOSE: probe_prefix({n_pass}) is {:.3}x the decode_step wall, \
             so the host-tail term above is NEGATIVE and every total that adds it under-reads by \
             {:.3} ms. Two instruments, one graph, and they disagree -- read the per-label slopes \
             (which are differences of decode steps) and not the envelope.",
            t_graph / wall_ms,
            t_graph - wall_ms
        );
    }

    let lo = env_usize("NV_QWEN36_BUDGET_LO", 1);
    let hi = env_usize("NV_QWEN36_BUDGET_HI", 4);
    assert!(hi > lo, "replication needs hi > lo");
    let mut uniq: Vec<String> = m.pass_labels().iter().map(|s| s.to_string()).collect();
    uniq.sort();
    uniq.dedup();
    let mut counts: Vec<usize> = Vec::with_capacity(uniq.len());
    let mut meta: Vec<(f64, (u32, u32, u32))> = Vec::with_capacity(uniq.len());
    for l in &uniq {
        let idx: Vec<usize> = (0..n_pass)
            .filter(|&i| m.pass_labels()[i] == l.as_str())
            .collect();
        counts.push(idx.len());
        let i0 = idx[0];
        meta.push((streamed_mb(l, m.pass_bound_bytes(i0).1), m.pass_grid(i0)));
    }

    let s0 = Instant::now();
    let mut costs: Vec<Cost> = Vec::new();

    let mut bases: Vec<f64> = Vec::new();
    for (li, label) in uniq.iter().enumerate() {
        let n = counts[li];
        let added_lo = m.probe_append(label, lo);
        assert_eq!(added_lo, n * lo, "probe_append did not reach the graph");
        assert_eq!(
            m.pass_count(),
            n_pass + n * lo,
            "appended graph is the wrong length"
        );
        m.reset().expect("reset");
        let mut t = seed;
        for _ in 0..warm {
            t = m.decode_step(t).expect("warm");
        }
        let step = |m: &mut q3w::Qwen3MoeWgpu, t: &mut u32, copies: usize| -> f64 {
            m.probe_append(label, copies);
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
        bases.push(base.iter().cloned().fold(f64::INFINITY, f64::min));
        costs.push(Cost {
            label: label.clone(),
            n,
            ratio: med(&mut ratio),
            zero: med(&mut zero),
            us: 0.0,
            null_us: 0.0,
            mb: meta[li].0,
            grid: meta[li].1,
        });
    }
    let base_ms = med(&mut bases.clone());
    let base_lo = bases.iter().cloned().fold(f64::INFINITY, f64::min);

    let scale = wall_ms.min(base_lo);
    for c in costs.iter_mut() {
        let solve = |r: f64| -> f64 {
            let den = c.n as f64 * ((hi - lo) as f64 - r * lo as f64);
            if den.abs() < 1e-9 {
                0.0
            } else {
                1e3 * r * scale / den
            }
        };
        c.us = solve(c.ratio);
        c.null_us = solve(c.zero);
    }
    let wall_ms = scale;
    eprintln!(
        "\n{} labels, {steps} interleaved rounds of lo/hi/lo (copies {lo}/{hi}/{lo}), in {:.1}s",
        uniq.len(),
        s0.elapsed().as_secs_f64()
    );
    eprintln!(
        "lo arm (token + {lo} extra copy set): {base_ms:.3} ms median of per-label floors, {base_lo:.3} ms overall floor; costs scaled to a {wall_ms:.3} ms token"
    );

    costs.sort_by(|x, y| {
        (y.us * y.n as f64)
            .partial_cmp(&(x.us * x.n as f64))
            .unwrap()
    });
    eprintln!("\n==== per-dispatch cost by replication slope ====");
    eprintln!(
        "{:<42} {:>4} {:>9} {:>9} {:>8} {:>8} {:>8}  GB/s   grid",
        "label", "n", "us each", "null us", "ms/tok", "% tok", "MB read"
    );
    let mut accounted = 0.0;
    let mut streamed = 0.0;
    for c in &costs {
        let ms = c.us * c.n as f64 / 1e3;
        accounted += ms;
        streamed += c.mb * c.n as f64;
        eprintln!(
            "{:<42} {:>4} {:>9.1} {:>9.1} {:>8.3} {:>7.1}% {:>8.2} {:>8.0}  {}x{}x{}",
            c.label,
            c.n,
            c.us,
            c.null_us,
            ms,
            100.0 * ms / wall_ms,
            c.mb,
            if c.mb > 0.0 && c.us > 0.0 {
                c.mb / c.us * 1e3
            } else {
                0.0
            },
            c.grid.0,
            c.grid.1,
            c.grid.2
        );
    }
    let sync = t_empty + (wall_ms - t_graph);
    eprintln!(
        "\naccounted by replication {accounted:.3} ms + host encode {encode:.3} + host sync {sync:.3} = {:.3} ms of a {wall_ms:.3} ms token ({:.0}%)",
        accounted + encode + sync,
        100.0 * (accounted + encode + sync) / wall_ms
    );
    eprintln!(
        "weight traffic {:.3} GB/token -> {:.3} ms at the 738.5 GB/s roofline; the token is {wall_ms:.1} ms, so the graph realizes {:.0} GB/s ({:.0}% of roofline)",
        streamed / 1e3,
        streamed / 738.5,
        streamed / wall_ms,
        100.0 * streamed / wall_ms / 738.5
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
        "\nshape: weight-streaming dispatches {streaming:.3} ms ({:.0}% of accounted), everything else {small:.3} ms ({:.0}%), host {:.3} ms ({:.1}%)",
        100.0 * streaming / accounted,
        100.0 * small / accounted,
        encode + sync,
        100.0 * (encode + sync) / wall_ms
    );

    if layers != full_layers {
        eprintln!("\ntruncated to {layers} layers -- shape assertions skipped");
        return;
    }

    assert!(
        (600..=720).contains(&n_pass),
        "pass count moved to {n_pass}; re-measure the budget before trusting these bounds"
    );

    assert!(
        encode + sync < 0.10 * wall_ms,
        "host cost grew to {:.3} ms of a {wall_ms:.3} ms token",
        encode + sync
    );
    let encode_us = 1e3 * encode / n_pass as f64;
    assert!(encode_us < 3.0, "host encode is {encode_us:.2} us/dispatch");

    assert!(
        accounted > 0.70 * wall_ms && accounted < 1.30 * wall_ms,
        "replication accounts for {accounted:.3} ms of a {wall_ms:.3} ms token"
    );

    let rate = |needle: &str| -> f64 {
        let c = costs
            .iter()
            .find(|c| c.label.contains(needle))
            .unwrap_or_else(|| panic!("no {needle} in the census"));
        c.mb / c.us * 1e3
    };
    for (needle, want) in [("lmhead", 729.0), ("dn-inproj", 721.0), ("dn-oproj", 672.0)] {
        let got = rate(needle);
        eprintln!("calibration {needle}: {got:.0} GB/s against {want:.0} measured independently");
        assert!(
            got > 0.5 * want && got < 1.6 * want,
            "{needle} reads {got:.0} GB/s where an independent lane measured {want:.0}; the instrument is off, not the kernel"
        );
    }

    let moe = bucket(&|c: &Cost| c.label.contains("moe-gateup") || c.label.contains("moe-down"));
    let moe_mb: f64 = costs
        .iter()
        .filter(|c| c.label.contains("moe-gateup") || c.label.contains("moe-down"))
        .map(|c| c.mb * c.n as f64)
        .sum();

    const MOE_SHAPE_ROOFLINE: f64 = 496.0;
    eprintln!(
        "MoE expert GEMVs: {moe:.3} ms/token for {moe_mb:.0} MB, {:.0} GB/s ({:.0}% of the SHAPE's {MOE_SHAPE_ROOFLINE:.0} GB/s roofline)",
        moe_mb / moe,
        100.0 * moe_mb / moe / MOE_SHAPE_ROOFLINE
    );

    assert!(
        moe > 0.15 * accounted,
        "MoE expert GEMVs fell to {:.0}% of accounted time -- re-measure the budget",
        100.0 * moe / accounted
    );
    assert!(
        moe_mb / moe < 400.0,
        "MoE expert GEMVs now run at {:.0} GB/s -- the kernel was fixed, re-derive the budget",
        moe_mb / moe
    );

    let chore = bucket(&|c: &Cost| {
        ["moe-siluq", "moe-xquant", "moe-topk", "moe-combine"]
            .iter()
            .any(|k| c.label.contains(k))
    });
    eprintln!(
        "MoE bookkeeping (xquant, siluq, topk, combine): {chore:.3} ms/token over 160 dispatches moving no weights, {:.0}% of the token",
        100.0 * chore / wall_ms
    );
    assert!(
        chore > 0.06 * wall_ms,
        "MoE bookkeeping fell to {:.1}% of the token -- the fusion landed, re-derive the budget",
        100.0 * chore / wall_ms
    );

    let floor_1wg = costs
        .iter()
        .filter(|c| c.grid == (1, 1, 1) && c.us > 0.0)
        .map(|c| c.us)
        .fold(f64::INFINITY, f64::min);
    let workgroups = |c: &Cost| (c.grid.0 as u64) * (c.grid.1 as u64) * (c.grid.2 as u64);

    let blocks_of = |c: &Cost| -> Option<f64> {
        let kb = if c.label.contains("moe-xquant") {
            hidden / 16
        } else if c.label.contains("moe-siluq") {
            inter / 16
        } else {
            return None;
        };
        Some(c.grid.1 as f64 * kb as f64)
    };
    eprintln!(
        "\n==== can the bookkeeping be widened, or only fused? ====\ncheapest one-workgroup \
         dispatch on this graph: {floor_1wg:.1} us"
    );
    for c in costs.iter().filter(|c| {
        ["moe-siluq", "moe-xquant", "moe-topk", "moe-combine"]
            .iter()
            .any(|k| c.label.contains(k))
    }) {
        let resolved = c.us > 2.0 * c.null_us;
        let blocks = blocks_of(c);
        eprintln!(
            "  {:<38} {:>3} x {:>6.1} us on {} workgroup(s){} -> {:.3} ms/tok{}",
            c.label,
            c.n,
            c.us,
            workgroups(c),
            match blocks {
                Some(b) => format!(", {b:.0} nvfp4 blocks = {:.1} ns/block", 1e3 * c.us / b),
                None => String::new(),
            },
            c.us * c.n as f64 / 1e3,
            if resolved {
                ""
            } else {
                " :: UNRESOLVED (cost is not 2x its own null) -- do not plan against this row"
            }
        );
    }

    let quant: Vec<(&Cost, Option<f64>)> = costs
        .iter()
        .filter(|c| c.label.contains("quant_rows") || c.label.contains("silu_mul_quant"))
        .map(|c| (c, blocks_of(c)))
        .collect();
    eprintln!("\n==== every dispatch of the nvfp4 block quantizer ====");
    for (c, b) in &quant {
        eprintln!(
            "  {:<38} {:>3} x {:>6.1} us  null {:>5.1}  {} wg{}",
            c.label,
            c.n,
            c.us,
            c.null_us,
            workgroups(c),
            match b {
                Some(b) => format!(", {b:.0} nvfp4 blocks"),
                None => String::new(),
            }
        );
    }
    let live: Vec<&(&Cost, Option<f64>)> = quant
        .iter()
        .filter(|(c, _)| c.us > 0.0 && c.us > 2.0 * c.null_us)
        .collect();
    let entry_of = |c: &Cost| -> String {
        c.label
            .rsplit(':')
            .next()
            .unwrap_or(c.label.as_str())
            .to_string()
    };
    let mut entries: Vec<String> = live.iter().map(|(c, _)| entry_of(c)).collect();
    entries.sort();
    entries.dedup();
    eprintln!();
    for e in &entries {
        let mine: Vec<&&(&Cost, Option<f64>)> =
            live.iter().filter(|(c, _)| &entry_of(c) == e).collect();
        let all_blocks = mine.iter().all(|(_, b)| b.is_some());
        let unit = if all_blocks {
            "nvfp4 blocks"
        } else {
            "workgroups"
        };
        let pts: Vec<(f64, f64)> = mine
            .iter()
            .map(|(c, b)| {
                (
                    c.us,
                    if all_blocks {
                        b.unwrap()
                    } else {
                        workgroups(c) as f64
                    },
                )
            })
            .collect();
        let span = |f: &dyn Fn(&(f64, f64)) -> f64| -> f64 {
            pts.iter().map(f).fold(0.0, f64::max) / pts.iter().map(f).fold(f64::INFINITY, f64::min)
        };
        let (ws, ts) = (span(&|p| p.1), span(&|p| p.0));
        let verdict = widen_verdict(&pts);
        let ms: f64 = mine.iter().map(|(c, _)| c.us * c.n as f64 / 1e3).sum();
        let n: usize = mine.iter().map(|(c, _)| c.n).sum();
        eprintln!(
            "  {e}: {} resolved instance(s) spanning {ws:.2}x in {unit} and {ts:.2}x in cost \
             -- {verdict:?} ({ms:.3} ms/token over {n} dispatches)",
            pts.len()
        );
        match verdict {
            WidenVerdict::DepthBound => eprintln!(
                "    => THE AMOUNT OF WORK IS NOT WHAT THIS COSTS. The wall is ONE thread's serial \
                 descent through its own 16 elements plus the {floor_1wg:.1} us a dispatch costs \
                 before it computes anything, and the generated MSL puts that descent's 16-element \
                 staging array in the THREAD address space -- naga hands q3w_qz_core a pointer, and \
                 a pointer needs an address, so no unrolling can put it in registers. Widening the \
                 grid cannot help; shortening the descent, or deleting the dispatch by folding it \
                 into its producer, is the lever."
            ),
            WidenVerdict::ThroughputBound => eprintln!(
                "    => the cost tracks the work: this entry is throughput-bound and a wider grid \
                 IS the lever. Re-derive before planning against it."
            ),
            WidenVerdict::Inconclusive => eprintln!(
                "    => INCONCLUSIVE. This graph's instances of {e} do not span {:.0}x in {unit}, \
                 so nothing here separates 'one thread's depth' from 'not enough grid'. Do NOT \
                 read this as either verdict: the settled negative (1 -> 72 workgroups moved \
                 q3w_quant_rows 27.50 -> 26.31 us, 0.96x) was measured by the standalone sweep in \
                 tests/qwen3_5_moe_quant_rate.rs, which can pick its own shapes, and that is where \
                 it has to be re-measured.",
                WIDEN_MIN_WORK_SPAN
            ),
        }
    }
    if entries.is_empty() {
        eprintln!("  no quantize instance resolved against its own null this run.");
    }
    let topk = costs
        .iter()
        .find(|c| c.label.contains("moe-topk"))
        .expect("no top-k pass in the census");
    assert_eq!(
        topk.grid,
        (1, 1, 1),
        "the router top-k moved off a single workgroup. It is a reduction to one k-of-n \
         selection, so there is nowhere for a second workgroup to go without a cross-workgroup \
         barrier -- if one appeared, the whole budget is stale"
    );
}

#[test]
#[ignore = "holds two 24 GB graphs at once; set NV_QWEN36_QUANT_ABA=1"]
fn lane_split_quantize_ab_a_on_the_whole_token() {
    assert_eq!(
        std::env::var("NV_QWEN36_QUANT_ABA").ok().as_deref(),
        Some("1"),
        "set NV_QWEN36_QUANT_ABA=1 -- a silent skip here would report a pass"
    );
    let ctx = nv_kernels::wgpu_backend::WgpuContext::shared().expect("wgpu adapter");
    eprintln!("adapter: {}", ctx.summary());
    if let Ok(o) = std::process::Command::new("uptime").output() {
        eprintln!("host: {}", String::from_utf8_lossy(&o.stdout).trim());
    }
    let dir = snapshot_dir();
    assert!(
        dir.join("config.json").is_file(),
        "checkpoint missing at {}",
        dir.display()
    );
    let mut cfg = Qwen3MoeConfig::from_hf_json_file(&dir.join("config.json")).expect("config");
    let full_layers = cfg.num_hidden_layers;
    let layers = env_usize("NV_QWEN36_ABA_LAYERS", full_layers);
    assert!(layers > 0 && layers <= full_layers);
    cfg.num_hidden_layers = layers;
    cfg.layer_types.truncate(layers);
    let loader = nv_weights::WeightLoader::open_dir(&dir, &candle_core::Device::Cpu)
        .expect("open safetensors");
    let max_seq = env_usize("NV_QWEN36_ABA_SEQ", 512);
    let seed = 9419u32;
    let warm = env_usize("NV_QWEN36_ABA_WARM", 6);
    let steps = env_usize("NV_QWEN36_ABA_STEPS", 24);
    let reps = env_usize("NV_QWEN36_ABA_REPS", 9);

    let mut built = Vec::new();
    for arm in ["w32", "0"] {
        std::env::set_var("NV_Q3_WGPU_QUANT_LANE", arm);
        let t0 = Instant::now();
        let m = q3w::Qwen3MoeWgpu::from_loader(cfg.clone(), &loader, max_seq)
            .expect("build the decode graph");
        let (on, total) = m.quant_lane_passes();
        eprintln!(
            "arm {arm}: built in {:.1}s, {} passes/token, lane-split {on}/{total}",
            t0.elapsed().as_secs_f64(),
            m.pass_count()
        );
        assert_eq!(
            on,
            if arm == "0" { 0 } else { total },
            "arm {arm} did not reach the builder; NV_Q3_WGPU_QUANT_LANE={arm}"
        );
        assert!(
            total > 0,
            "the graph emitted no nvfp4 quantize passes at all"
        );
        built.push(m);
    }
    std::env::remove_var("NV_Q3_WGPU_QUANT_LANE");
    assert_eq!(
        built[0].pass_count(),
        built[1].pass_count(),
        "the arms do not have the same dispatch count; this is a kernel swap, and the \
         comparison is only clean while nothing else moved with it"
    );

    let mut effects: Vec<f64> = Vec::new();
    let mut nulls: Vec<f64> = Vec::new();
    let mut a_all: Vec<f64> = Vec::new();
    let mut b_all: Vec<f64> = Vec::new();
    for r in 0..reps {
        let w = if r == 0 { warm } else { 1 };
        let (_, a) = wall(&mut built[0], seed, w, steps);
        let (_, b) = wall(&mut built[1], seed, w, steps);
        let (_, a2) = wall(&mut built[0], seed, 1, steps);
        effects.push(b - (a + a2) / 2.0);
        nulls.push((a - a2).abs());
        a_all.push((a + a2) / 2.0);
        b_all.push(b);
        eprintln!("rep {r}: A {a:.3}  B {b:.3}  A' {a2:.3} ms/token");
    }
    let effect = med(&mut effects);
    let null = med(&mut nulls);
    let (a, b) = (med(&mut a_all), med(&mut b_all));
    eprintln!(
        "\n==== lane-split quantize, interleaved A/B/A on {layers}/{full_layers} layers, \
         {reps} reps x {steps} steps ====\n\
         lane-split ON  {a:.3} ms/token (median of per-rep medians)\n\
         lane-split OFF {b:.3} ms/token\n\
         effect B - mean(A, A') = {effect:.3} ms/token, own null |A - A'| = {null:.3} ms"
    );
    if effect <= 2.0 * null {
        eprintln!(
            "UNRESOLVED against its own null. Report the per-dispatch slope from \
             qwen3_5_moe_quant_rate.rs and NOT this number."
        );
    } else {
        eprintln!(
            "RESOLVED: {effect:.3} ms/token, {:.1}x its own null, {:.2}% of the ON token.",
            effect / null.max(1e-9),
            100.0 * effect / a
        );
    }
}
