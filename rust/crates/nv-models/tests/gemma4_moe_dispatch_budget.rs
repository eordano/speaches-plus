#![cfg(feature = "wgpu")]

mod common;
use common::env_usize;
use common::median;
use nv_models::gemma4_moe::Gemma4MoeConfig;
use nv_models::gemma4_moe_wgpu::{moe_weight_classes, Gemma4MoeWgpu};
use nv_weights::GgufLoader;

const LAW_FIXED_US: (f64, f64) = (3.9, 4.3);
const LAW_RATE_GBS: (f64, f64) = (748.0, 755.0);

fn gguf_path() -> String {
    let p =
        std::env::var("NV_GGUF_PATH").expect("set NV_GGUF_PATH to a gemma-4-26B-A4B-it-Q8_0 GGUF");
    assert!(
        std::path::Path::new(&p).exists(),
        "no GGUF at {p}; set NV_GGUF_PATH"
    );
    p
}

struct Geom {
    share: f64,
    hidden: usize,

    kv_row: f64,

    kv_read: f64,
}

fn effective_bytes(label: &str, bind_bytes: u64, n: usize, g: &Geom) -> f64 {
    let per = bind_bytes as f64 / n as f64;
    if label.starts_with("g4m-moe-gate")
        || label.starts_with("g4m-moe-up")
        || label.starts_with("g4m-moe-down")
    {
        per * g.share
    } else if label == "g4m-gather" {
        (g.hidden * 2) as f64
    } else if label == "g4m-at-kvwrite" {
        2.0 * g.kv_row / n as f64
    } else if label == "g4m-at-decode" {
        g.kv_read / n as f64
    } else {
        per
    }
}

struct Arm {
    class: String,
    n: usize,
    k: usize,
    per_dispatch_us: f64,
    spread: f64,
    bytes: f64,
}

#[test]
#[ignore = "wires ~17 GiB from a 26 GB GGUF; run with --ignored --release"]
fn decode_dispatch_budget_reconstructs_the_token() {
    let ctx = nv_kernels::wgpu_backend::WgpuContext::shared()
        .expect("this budget needs a wgpu adapter; there is no skip path");
    eprintln!("[budget] adapter: {}", ctx.info.name);

    let path = gguf_path();
    let gguf = GgufLoader::open(&path, &candle_core::Device::Cpu).expect("open gguf");
    let mut cfg: Gemma4MoeConfig =
        nv_models::gemma4_gguf::gemma4_moe_config_from_gguf(&gguf).expect("gguf config");

    if let Ok(n) = std::env::var("NV_G4MOE_BUDGET_LAYERS") {
        let n: usize = n.parse().expect("NV_G4MOE_BUDGET_LAYERS");
        assert!(n > 0 && n <= cfg.base.num_hidden_layers);
        cfg.base.num_hidden_layers = n;
        cfg.base.layer_types.truncate(n);
        eprintln!("[budget] TRUNCATED to {n} layers -- smoke run, not the model");
    }
    let hidden = cfg.base.hidden_size;
    let layers = cfg.base.num_hidden_layers;
    let share = cfg.top_k_experts as f64 / cfg.num_experts as f64;
    let max_seq = env_usize("NV_G4MOE_BUDGET_MAXSEQ", 512);
    let warm = env_usize("NV_G4MOE_BUDGET_WARM", 32);
    let reps = env_usize("NV_G4MOE_BUDGET_REPS", 9);
    let mid_pos = warm as f64 + reps as f64 / 2.0;
    let kv_row: f64 = (0..layers)
        .map(|i| {
            let k = cfg.base.layer_kind(i);
            (cfg.base.num_kv_heads_for(k) * cfg.base.head_dim_for(k) * 2 * 2) as f64
        })
        .sum();
    let geom = Geom {
        share,
        hidden,
        kv_row,
        kv_read: kv_row * (mid_pos + 1.0),
    };

    let t0 = std::time::Instant::now();
    let mut m = Gemma4MoeWgpu::from_loader(cfg.clone(), &gguf, max_seq).expect("build from gguf");
    eprintln!(
        "[budget] built full depth ({layers} layers) in {:.1}s, {} dispatches/token, wired {:.2} GiB",
        t0.elapsed().as_secs_f64(),
        m.pass_count(),
        m.load_report().wired_bytes as f64 / (1u64 << 30) as f64
    );

    let rows = m.pass_rows();
    assert_eq!(rows.len(), m.pass_count(), "census lost a dispatch");
    let mut order: Vec<String> = Vec::new();
    let mut census: std::collections::HashMap<String, (usize, String, (u32, u32, u32), u64)> =
        Default::default();
    for (label, entry, grid, bytes) in &rows {
        let e = census.entry(label.clone()).or_insert_with(|| {
            order.push(label.clone());
            (0, entry.clone(), *grid, 0)
        });
        e.0 += 1;
        e.3 += *bytes;
    }
    eprintln!(
        "[census] {} dispatches over {} classes",
        rows.len(),
        order.len()
    );
    eprintln!(
        "[census] {:<20} {:>5}  {:<26} {:>12}  {:>12}",
        "class", "n", "entry", "grid.x", "MB/token"
    );
    for label in &order {
        let (n, entry, grid, bytes) = &census[label];
        let eff = effective_bytes(label, *bytes, *n, &geom) * *n as f64;
        eprintln!(
            "[census] {label:<20} {n:>5}  {entry:<26} {:>12}  {:>12.2}",
            grid.0,
            eff / 1e6
        );
    }

    let wbpt = m.weight_bytes_per_token() as f64;
    let ckpt = gguf
        .active_bytes_per_token(cfg.top_k_experts, cfg.num_experts)
        .expect("checkpoint active bytes") as f64;
    eprintln!(
        "[bytes] checkpoint {:.3} GB/tok, graph {:.3} GB/tok, layout tax {:.4}x",
        ckpt / 1e9,
        wbpt / 1e9,
        wbpt / ckpt
    );

    let target_extra = env_usize("NV_G4MOE_BUDGET_EXTRA", 400);
    let tok0: u32 = 2;
    assert!(
        warm + reps + 2 < max_seq,
        "budget walk overruns the kv cache"
    );

    let run_paired =
        |m: &mut Gemma4MoeWgpu, class: Option<&str>, k: usize| -> (f64, f64, f64, u32) {
            m.reset().expect("reset");
            let mut next = tok0;
            for _ in 0..warm {
                next = m.decode_step(next).expect("warm decode");
            }
            let mut base = Vec::with_capacity(reps);
            let mut diff = Vec::with_capacity(reps);
            for _ in 0..reps {
                let t0 = std::time::Instant::now();
                next = m
                    .decode_step_replicated(next, None, 0)
                    .expect("paired baseline");
                let b = t0.elapsed().as_secs_f64() * 1e3;
                let t1 = std::time::Instant::now();
                next = m
                    .decode_step_replicated(next, class, k)
                    .expect("replicated decode");
                let a = t1.elapsed().as_secs_f64() * 1e3;
                base.push(b);
                diff.push(a - b);
            }
            let lo = diff.iter().cloned().fold(f64::MAX, f64::min);
            let hi = diff.iter().cloned().fold(f64::MIN, f64::max);
            let d = median(&mut diff);
            let bm = median(&mut base);
            (bm, d, (hi - lo) / d.abs().max(1e-9), next)
        };

    let cwarm = 2usize;
    let csteps = 2usize;
    for label in &order {
        let (n, _, _, _) = census[label];
        let k = (target_extra / n).max(1);
        let mut walk = |class: Option<&str>| -> Vec<Vec<f32>> {
            m.reset().expect("reset");
            let mut next = tok0;
            for _ in 0..cwarm {
                next = m.decode_step(next).expect("warm");
            }
            (0..csteps)
                .map(|_| {
                    next = m
                        .decode_step_replicated(next, class, if class.is_some() { k } else { 0 })
                        .expect("step");
                    m.read_logits().expect("logits")
                })
                .collect()
        };
        let reference = walk(None);
        let replicated = walk(Some(label));
        assert!(
            reference[0] != reference[1],
            "positive control failed at {label}: two decode steps at different positions \
             produced bit-identical logits, so this comparison cannot see a difference \
             and every identity claim below is vacuous"
        );
        assert!(
            reference == replicated,
            "replicating {label} changed the logits: its kernel writes something the rest \
             of the token reads, so its arm is not measuring the same graph"
        );
    }
    eprintln!(
        "[check] replication is bit-identical in the logits for all {} classes, and the \
         comparison is sensitive (consecutive-step control differs)",
        order.len()
    );

    let (base_a, null_a, _, _) = run_paired(&mut m, None, 0);
    eprintln!(
        "[base] A  {base_a:.3} ms/token  null control {null_a:+.3} ms ({:+.2}%)  \
         (pos {warm}.., {} dispatches)",
        null_a / base_a * 100.0,
        rows.len()
    );
    assert!(
        null_a.abs() / base_a < 0.05,
        "the null control prices {null_a:+.3} ms on a {base_a:.3} ms token; the paired \
         difference is not clean and no arm below can be trusted"
    );

    let mut arms: Vec<Arm> = Vec::new();
    let mut base_track: Vec<f64> = vec![base_a];
    for label in order.clone() {
        let (n, _, _, bytes) = census[&label];
        let k = (target_extra / n).max(1);
        let (bm, d, spread, _) = run_paired(&mut m, Some(&label), k);
        base_track.push(bm);
        let per = d * 1e3 / (k * n) as f64;
        eprintln!(
            "[arm] {label:<20} n={n:<4} k={k:<4} base {bm:7.3} ms  +{d:8.3} ms  \
             pair-spread {:>6.1}%  per-dispatch {per:8.3} us",
            spread * 100.0
        );
        arms.push(Arm {
            bytes: effective_bytes(&label, bytes, n, &geom),
            class: label,
            n,
            k,
            per_dispatch_us: per,
            spread,
        });
    }

    let (base_b, null_b, _, _) = run_paired(&mut m, None, 0);
    base_track.push(base_b);
    let drift = (base_b - base_a).abs() / base_a;
    let bt_lo = base_track.iter().cloned().fold(f64::MAX, f64::min);
    let bt_hi = base_track.iter().cloned().fold(0f64, f64::max);
    eprintln!(
        "[base] A' {base_b:.3} ms/token  null control {null_b:+.3} ms  |A'-A|/A = {:.2}%  \
         (baseline over the whole sweep: {bt_lo:.3}..{bt_hi:.3} ms, {:.1}% span -- this is \
         the drift the pairing removes, NOT an error bar on the arms)",
        drift * 100.0,
        (bt_hi - bt_lo) / bt_lo * 100.0
    );
    assert!(
        null_b.abs() / base_b < 0.05,
        "the closing null control prices {null_b:+.3} ms on a {base_b:.3} ms token"
    );
    let base_a = median(&mut base_track.clone());

    let by = |c: &str| -> &Arm {
        arms.iter()
            .find(|a| a.class == c)
            .unwrap_or_else(|| panic!("no arm for {c}"))
    };

    const TINY_BYTES: f64 = 64.0 * 1024.0;
    let fixed = arms
        .iter()
        .filter(|a| a.bytes < TINY_BYTES)
        .min_by(|x, y| x.per_dispatch_us.partial_cmp(&y.per_dispatch_us).unwrap())
        .expect("no near-zero-byte class to calibrate the fixed dispatch cost on");
    let a_us = fixed.per_dispatch_us;
    let head = by("g4m-lmhead");
    let rate = head.bytes / ((head.per_dispatch_us - a_us) * 1e-6) / 1e9;
    let law_rate = (LAW_RATE_GBS.0 + LAW_RATE_GBS.1) / 2.0;
    eprintln!(
        "[calib] in-situ fixed dispatch cost {a_us:.3} us (cheapest tiny class: {} at \
         {:.0} B); box sweep says {:?} us",
        fixed.class, fixed.bytes, LAW_FIXED_US
    );
    eprintln!(
        "[calib] in-situ stream rate {rate:.1} GB/s from {} at {:.1} MB/dispatch; the box's \
         large-dispatch asymptote is {:?} GB/s, so the FLOOR column below uses \
         {law_rate:.0} GB/s and every 'excess' is an UPPER bound on removable time -- part \
         of it is this shape's own roofline, not waste",
        head.class,
        head.bytes / 1e6,
        LAW_RATE_GBS
    );
    assert!(
        a_us > LAW_FIXED_US.0 * 0.6 && a_us < LAW_FIXED_US.1 * 2.0,
        "in-situ fixed dispatch cost {a_us:.3} us is nowhere near the box's measured \
         {LAW_FIXED_US:?} us; the replication difference is measuring something else"
    );
    assert!(
        rate > 0.0 && rate < LAW_RATE_GBS.1,
        "in-situ stream rate {rate:.1} GB/s is not a rate below this box's asymptote \
         ({LAW_RATE_GBS:?}); above it means cache-resident, negative means contaminated"
    );
    for a in &arms {
        assert!(
            a.per_dispatch_us > 0.0,
            "class {} priced at {:.3} us/dispatch -- a negative cost is contamination, \
             not a measurement",
            a.class,
            a.per_dispatch_us
        );
    }

    let mut total_us = 0.0;
    let mut law_us = 0.0;
    let mut excess: Vec<(String, f64, f64)> = Vec::new();
    eprintln!(
        "[budget] {:<20} {:>4} {:>4} {:>7} {:>10} {:>9} {:>9} {:>9} {:>7}",
        "class", "n", "k", "spread", "MB/disp", "us/disp", "law us", "ms/tok", "% tok"
    );
    let mut sorted: Vec<&Arm> = arms.iter().collect();
    sorted.sort_by(|x, y| {
        (y.per_dispatch_us * y.n as f64)
            .partial_cmp(&(x.per_dispatch_us * x.n as f64))
            .unwrap()
    });
    for a in &sorted {
        let ms = a.per_dispatch_us * a.n as f64 / 1e3;
        let law = a_us + a.bytes / (law_rate * 1e9) * 1e6;
        total_us += a.per_dispatch_us * a.n as f64;
        law_us += law * a.n as f64;
        excess.push((
            a.class.clone(),
            (a.per_dispatch_us - law) * a.n as f64 / 1e3,
            ms,
        ));
        eprintln!(
            "[budget] {:<20} {:>4} {:>4} {:>6.1}% {:>10.3} {:>9.3} {:>9.3} {:>9.3} {:>6.1}%",
            a.class,
            a.n,
            a.k,
            a.spread * 100.0,
            a.bytes / 1e6,
            a.per_dispatch_us,
            law,
            ms,
            ms / base_a * 100.0
        );
    }
    eprintln!(
        "[budget] reconstruction {:.3} ms vs measured token {:.3} ms ({:+.1}%); \
         dispatch-law prediction {:.3} ms ({:.0}% of the token)",
        total_us / 1e3,
        base_a,
        (total_us / 1e3 - base_a) / base_a * 100.0,
        law_us / 1e3,
        law_us / 1e3 / base_a * 100.0
    );
    excess.sort_by(|x, y| y.1.partial_cmp(&x.1).unwrap());
    eprintln!("[excess] class-by-class ms/token ABOVE the dispatch-size law:");
    for (c, e, ms) in excess.iter().take(10) {
        eprintln!("[excess] {c:<20} {e:+8.3} ms of {ms:8.3} ms");
    }

    assert!(
        (total_us / 1e3 - base_a).abs() / base_a < 0.30,
        "per-class budget reconstructs {:.3} ms but the token measures {base_a:.3} ms; \
         the attribution is not closed",
        total_us / 1e3
    );

    let report = m.vram_report();
    let mut bf16_dense = 0u64;
    for (class, _n, bytes) in &report.by_class {
        if !moe_weight_classes().contains(&class.as_str()) {
            continue;
        }

        if class.starts_with("moe-e") || class == "embed" {
            continue;
        }
        bf16_dense += *bytes;
    }
    let int8_g16 = bf16_dense as f64 * (1.25 / 2.0);
    eprintln!(
        "[tax] bf16 dense-projection bytes/token {:.3} GB of {:.3} GB total ({:.0}%); \
         at int8 group(16) they would be {:.3} GB, saving {:.3} GB/token",
        bf16_dense as f64 / 1e9,
        wbpt / 1e9,
        bf16_dense as f64 / wbpt * 100.0,
        int8_g16 / 1e9,
        (bf16_dense as f64 - int8_g16) / 1e9
    );
    let dense_classes = [
        "g4m-at-qproj",
        "g4m-at-kproj",
        "g4m-at-oproj",
        "g4m-mlp-gate",
        "g4m-mlp-up",
        "g4m-mlp-down",
    ];

    let mut now_ms = 0.0;
    let mut then_ms = 0.0;
    for c in dense_classes {
        let a = by(c);
        now_ms += a.per_dispatch_us * a.n as f64 / 1e3;
        then_ms += (a_us + (a.per_dispatch_us - a_us) * 0.625) * a.n as f64 / 1e3;
        let r = a.bytes / ((a.per_dispatch_us - a_us) * 1e-6) / 1e9;
        eprintln!(
            "[tax] {:<16} {:>4} x {:>7.3} us at {:>6.2} MB = {:>5.0} GB/s realized",
            a.class,
            a.n,
            a.per_dispatch_us,
            a.bytes / 1e6,
            r
        );
    }
    eprintln!(
        "[tax] those 6 classes measure {now_ms:.3} ms/token today; at int8 group(16), the \
         same realized rate and the same {a_us:.2} us floor they project to {then_ms:.3} ms \
         -- {:.3} ms/token, {:.1}% of this token, {:.1} tok/s at the measured baseline",
        now_ms - then_ms,
        (now_ms - then_ms) / base_a * 100.0,
        1000.0 / (base_a - (now_ms - then_ms))
    );
}
