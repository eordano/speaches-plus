#![cfg(feature = "wgpu")]

mod common;
use common::config_json_wrapped_text_config as config_json;
use common::envn;
use common::LcgCentered0p1Shift32 as Lcg;
use std::time::Instant;

use nv_models::gemma4::{Gemma4Config, LayerType};
use nv_models::gemma4_wgpu::{
    quantize_nvfp4_host, Gemma4Wgpu, HostBf16Lin, HostLayer, HostProj, HostWeights,
};
use common::gemma4_host_weights_nvfp4_ffn as host_weights;

pub const WHAT_THIS_MEASURES: &str = "\
The host side of the dense Gemma-4 decode loop, one item at a time, on ONE model instance so no
arm pays a rebuild: (A) today's shape -- encode on the critical path, blocking 4-byte readback
plus a full-device drain per token; (B) encode-ahead only; (C..) encode-ahead plus GPU token
feedback at k=2/4/8, where the sampled id is copied token_out -> tok_idx on the GPU and only one
readback happens per k tokens.

Every arm must emit the SAME token ids as arm A. The counters (preenc_hits, chain_steps) are read
back after each arm because a k=1 fallback inside decode_chain would produce identical ids and
identical timings and read as an honest null result.";

fn ctx_or_panic() -> &'static nv_kernels::wgpu_backend::WgpuContext {
    match nv_kernels::wgpu_backend::WgpuContext::shared() {
        Ok(ctx) => {
            eprintln!("[g4d-hs] adapter: {}", ctx.summary());
            ctx
        }
        Err(e) => panic!("gemma4 dense host-shape test needs a wgpu adapter: {e}"),
    }
}

fn free_run(
    m: &mut Gemma4Wgpu,
    seed: u32,
    ctxlen: usize,
    n: usize,
    k: usize,
) -> (Vec<u32>, Vec<u32>) {
    m.reset();
    let mut tok = seed;
    for i in 0..ctxlen {
        tok = m
            .decode_step(((i as u32 * 977) % 1000) + 3)
            .expect("context step");
    }
    let mut out = Vec::with_capacity(n);
    while out.len() < n {
        let want = k.min(n - out.len());
        let got = m.decode_chain(tok, want).expect("decode_chain");
        assert_eq!(got.len(), want);
        tok = *got.last().unwrap();
        out.extend(got);
    }
    let (_, logits) = m.decode_step_logits(tok).expect("closing logits");
    (out, logits.into_iter().map(f32::to_bits).collect())
}

fn rollback_gate(m: &mut Gemma4Wgpu, tag: &str, ctxlen: usize) {
    m.set_preenc(true);
    m.reset();
    let mut anchor = 3u32;
    for i in 0..ctxlen {
        anchor = m
            .decode_step(((i as u32 * 977) % 1000) + 3)
            .expect("context step");
    }
    let mark = m.current_pos();
    let a = m.decode_chain(anchor, 4).expect("chain");
    m.truncate_to(mark).expect("truncate_to");
    assert_eq!(m.current_pos(), mark);
    let b = m.decode_chain(anchor, 4).expect("chain after rollback");
    assert_eq!(
        a, b,
        "truncate_to did not restore a decode state the chain can be replayed from"
    );
    m.truncate_to(mark).expect("truncate_to");
    let mut c = Vec::with_capacity(4);
    let mut t = anchor;
    for _ in 0..4 {
        t = m.decode_step(t).expect("step after rollback");
        c.push(t);
    }
    assert_eq!(
        a, c,
        "single-step decode after truncate_to disagrees with the chain it replaces"
    );
    eprintln!("[{tag}] truncate_to({mark}) replay is bit-exact: {a:?}");
}

fn distinct<T: std::hash::Hash + Eq>(xs: &[T]) -> usize {
    xs.iter().collect::<std::collections::HashSet<_>>().len()
}

const GELU_FOLD_ENV: &str = "NV_G4_WGPU_GELU_FOLD";

fn gelu_fold_on() -> bool {
    std::env::var(GELU_FOLD_ENV).ok().as_deref() != Some("0")
}

fn census(layers: usize, fold: bool) -> usize {
    let per_layer = if fold { 9 } else { 10 };
    2 + (per_layer + 1) + (layers - 1) * per_layer + 5
}

fn fold_census_gate(config: &Gemma4Config, host: &HostWeights, max_seq: usize) {
    let layers = config.num_hidden_layers;
    let prev = std::env::var(GELU_FOLD_ENV).ok();
    let build = |v: &str| {
        std::env::set_var(GELU_FOLD_ENV, v);
        Gemma4Wgpu::new(config.clone(), host, max_seq)
            .expect("build")
            .pass_count()
    };
    let on = build("1");
    let off = build("0");
    match prev.as_deref() {
        Some(v) => std::env::set_var(GELU_FOLD_ENV, v),
        None => std::env::remove_var(GELU_FOLD_ENV),
    }
    assert_eq!(
        off,
        census(layers, false),
        "with {GELU_FOLD_ENV}=0 the dense decode graph is gather+scale, 10 dispatches per layer \
         (11 on layer 0), 5 in the head; it issued {off} for {layers} layers"
    );
    assert_eq!(
        on,
        census(layers, true),
        "with {GELU_FOLD_ENV}=1 the gelu rides the int8 gate_up epilogue, so it is 9 dispatches \
         per layer (10 on layer 0), 5 in the head; it issued {on} for {layers} layers"
    );
    assert_eq!(
        off - on,
        layers,
        "the gelu fold must remove exactly one dispatch per layer ({off} unfolded vs {on} folded \
         over {layers} layers); the census in this file is derived from that delta and is now wrong"
    );
    eprintln!(
        "[g4d-hs] fold census: {GELU_FOLD_ENV}=0 -> {off}, =1 -> {on}, delta {} == {layers} layers",
        off - on
    );
}

fn ioreg_key(key: &str) -> Option<u64> {
    let out = std::process::Command::new("ioreg")
        .args(["-r", "-d", "1", "-w", "0", "-c", "IOAccelerator"])
        .output()
        .ok()?;
    let text = String::from_utf8_lossy(&out.stdout);
    let pat = format!("\"{key}\"=");
    let tail = &text[text.find(&pat)? + pat.len()..];
    let n: String = tail.chars().take_while(|c| c.is_ascii_digit()).collect();
    n.parse().ok()
}

fn gpu_util() -> Option<u64> {
    ioreg_key("Device Utilization %")
}

fn wait_idle(tag: &str) {
    let t0 = Instant::now();
    let mut streak = 0;
    loop {
        match gpu_util() {
            Some(u) if u <= 2 => streak += 1,
            Some(_) => streak = 0,
            None => {
                eprintln!("{tag}: no gpu counter source; NOT idle-gated");
                return;
            }
        }
        if streak >= 2 {
            return;
        }
        if t0.elapsed().as_secs_f64() > 300.0 {
            eprintln!("{tag}: WARNING gpu never went idle within 300s; measuring anyway");
            return;
        }
        std::thread::sleep(std::time::Duration::from_millis(500));
    }
}

struct ArmSpec {
    label: &'static str,
    preenc: bool,
    k: usize,
    probe: usize,
}

const ARMS: [ArmSpec; 7] = [
    ArmSpec {
        label: "A0 base(preenc=0,k=1)",
        preenc: false,
        k: 1,
        probe: 0,
    },
    ArmSpec {
        label: "P  base+5 extra uniforms",
        preenc: false,
        k: 1,
        probe: 1,
    },
    ArmSpec {
        label: "B  preenc(k=1)",
        preenc: true,
        k: 1,
        probe: 0,
    },
    ArmSpec {
        label: "C  preenc+chain k=2",
        preenc: true,
        k: 2,
        probe: 0,
    },
    ArmSpec {
        label: "D  preenc+chain k=4",
        preenc: true,
        k: 4,
        probe: 0,
    },
    ArmSpec {
        label: "E  preenc+chain k=8",
        preenc: true,
        k: 8,
        probe: 0,
    },
    ArmSpec {
        label: "A1 base(repeat)",
        preenc: false,
        k: 1,
        probe: 0,
    },
];

fn measure_once(m: &mut Gemma4Wgpu, a: &ArmSpec, ctxlen: usize, steps: usize) -> f64 {
    m.set_preenc(a.preenc);
    m.set_uniform_probe(a.probe);
    m.reset();
    let mut tok = 3u32;
    for i in 0..ctxlen {
        tok = m
            .decode_step(((i as u32 * 977) % 1000) + 3)
            .expect("context step");
    }
    let t0 = Instant::now();
    let mut done = 0usize;
    while done < steps {
        let want = a.k.min(steps - done);
        let got = m.decode_chain(tok, want).expect("decode");
        tok = *got.last().unwrap();
        done += want;
    }
    t0.elapsed().as_secs_f64() * 1e3 / steps as f64
}

fn median(v: &[f64]) -> f64 {
    let mut s = v.to_vec();
    s.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = s.len();
    if n == 0 {
        return f64::NAN;
    }
    if n % 2 == 1 {
        s[n / 2]
    } else {
        0.5 * (s[n / 2 - 1] + s[n / 2])
    }
}

fn best(v: &[f64]) -> f64 {
    v.iter().cloned().fold(f64::MAX, f64::min)
}

fn spread_pct(v: &[f64]) -> f64 {
    let b = best(v);
    100.0 * (v.iter().cloned().fold(0.0, f64::max) - b) / b
}

fn run_suite(
    m: &mut Gemma4Wgpu,
    tag: &str,
    ctxlen: usize,
    rounds: usize,
    steps: usize,
    gen: usize,
) {
    eprintln!("{WHAT_THIS_MEASURES}");
    eprintln!(
        "[{tag}] {} passes/token ({} before the head, {} in the head), {} weight bytes/token",
        m.pass_count(),
        m.head_start(),
        m.pass_count() - m.head_start(),
        m.weight_bytes_per_token()
    );
    let layers = m.config().num_hidden_layers;
    let fold = gelu_fold_on();
    let per_layer = if fold { 9 } else { 10 };
    let expected = census(layers, fold);
    assert_eq!(
        m.pass_count(),
        expected,
        "the dense decode graph is no longer gather+scale, {per_layer} dispatches per layer \
         ({} on layer 0), 5 in the head, with the gelu fold {} ({GELU_FOLD_ENV}); re-do the \
         fusion census before quoting a dispatch budget",
        per_layer + 1,
        if fold { "ON (shipping default)" } else { "OFF" }
    );
    eprintln!(
        "[{tag}] graph shape: 2 prologue + {} (layer 0) + {} x {per_layer} + 5 head = {} (gelu fold {})",
        per_layer + 1,
        layers - 1,
        expected,
        if fold { "on" } else { "off" }
    );

    m.set_preenc(false);
    let (pb, cb) = m.host_shape_counters();
    let (reference, ref_logits) = free_run(m, 3, ctxlen, gen, 1);
    assert_eq!(reference.len(), gen);
    assert!(
        distinct(&ref_logits) > 1,
        "the closing logit vector folded to a single value; this fixture proves nothing"
    );
    let (p_ref, c_ref) = m.host_shape_counters();
    assert_eq!(
        (p_ref - pb, c_ref - cb),
        (0, 0),
        "reference arm must run today's shape: no pre-encoded buffer, no chained step"
    );
    eprintln!(
        "[{tag}] reference ids (first 16 of {gen}, {} distinct): {:?}",
        distinct(&reference),
        &reference[..16.min(gen)]
    );
    if distinct(&reference) < 2 {
        eprintln!(
            "[{tag}] WARNING: the reference stream is one repeated id, so every arm-vs-arm ID \
             comparison below is VACUOUS on this fixture -- only the bitwise closing-logit \
             comparison and the preenc/chain counters carry weight. The `distinct(&ref_logits) > 1` \
             guard above does NOT cover this: logits can vary while every argmax lands on one id."
        );
    }

    for a in ARMS.iter() {
        m.set_preenc(a.preenc);
        m.set_uniform_probe(a.probe);
        let (p0, c0) = m.host_shape_counters();
        let (ids, logits) = free_run(m, 3, ctxlen, gen, a.k);
        let (p1, c1) = m.host_shape_counters();
        assert_eq!(
            ids, reference,
            "[{}] token ids diverged from the base arm; this change does not ship",
            a.label
        );
        assert!(
            logits == ref_logits,
            "[{}] closing logits differ bitwise from the base arm in {} of {} slots",
            a.label,
            logits
                .iter()
                .zip(&ref_logits)
                .filter(|(a, b)| a != b)
                .count(),
            ref_logits.len()
        );
        let chain_steps = c1 - c0;
        let preenc_hits = p1 - p0;
        if a.k > 1 {
            assert!(
                chain_steps >= (gen / a.k * a.k) as u64,
                "[{}] decode_chain took the k=1 fallback ({chain_steps} chained steps); \
                 identical ids prove nothing about a path that did not run",
                a.label
            );
        }
        if a.preenc {
            assert!(
                preenc_hits > 0,
                "[{}] preenc is on but no pre-encoded command buffer was ever consumed",
                a.label
            );
        } else {
            assert_eq!(
                preenc_hits, 0,
                "[{}] preenc is off but a pre-encoded command buffer was consumed",
                a.label
            );
        }
        eprintln!(
            "[{tag}] {} bit-exact over {gen} tokens (preenc_hits {preenc_hits}, chained {chain_steps})",
            a.label
        );
    }

    if rounds == 0 || steps == 0 {
        return;
    }
    let n = ARMS.len();
    let mut samples: Vec<Vec<f64>> = vec![Vec::with_capacity(rounds); n];
    for r in 0..rounds {
        wait_idle(tag);
        let order: Vec<usize> = if r % 2 == 0 {
            (0..n).collect()
        } else {
            (0..n).rev().collect()
        };
        let u0 = gpu_util();
        for i in order {
            let ms = measure_once(m, &ARMS[i], ctxlen, steps);
            samples[i].push(ms);
        }
        eprintln!(
            "[{tag}] round {r} ({}): {}",
            if r % 2 == 0 { "fwd" } else { "rev" },
            samples
                .iter()
                .map(|v| format!("{:.2}", v[r]))
                .collect::<Vec<_>>()
                .join(" ")
        );
        let _ = u0;
    }
    m.set_uniform_probe(0);
    m.set_preenc(true);

    let base_med = median(&samples[0]);
    eprintln!("\n================ HOST SHAPE :: {tag} ================");
    eprintln!(
        "arms measured ROUND-ROBIN inside each round, order alternating fwd/rev, {rounds} rounds \
         x {steps} steps from a fresh {ctxlen}-token context. The paired column is the median over \
         rounds of (arm - A0) measured in the SAME round, which is what survives machine drift; \
         wins/rounds is a sign test against A0."
    );
    eprintln!(
        "{:<26} {:>9} {:>9} {:>9} {:>9} {:>11} {:>11}",
        "arm", "med ms", "best ms", "spread%", "tok/s", "paired dms", "wins/rounds"
    );
    for (i, a) in ARMS.iter().enumerate() {
        let med = median(&samples[i]);
        let paired: Vec<f64> = (0..rounds).map(|r| samples[i][r] - samples[0][r]).collect();
        let pm = median(&paired);
        let wins = paired.iter().filter(|d| **d < 0.0).count();
        eprintln!(
            "{:<26} {med:>9.3} {:>9.3} {:>9.2} {:>9.2} {pm:>11.3} {:>11}",
            a.label,
            best(&samples[i]),
            spread_pct(&samples[i]),
            1000.0 / med,
            format!("{wins}/{rounds}")
        );
        println!(
            "G4D-HOSTSHAPE {tag} {} med_ms={med:.3} best_ms={:.3} spread_pct={:.2} paired_dms={pm:.3} wins={wins}/{rounds}",
            a.label,
            best(&samples[i]),
            spread_pct(&samples[i])
        );
    }
    let ctrl: Vec<f64> = (0..rounds)
        .map(|r| samples[n - 1][r] - samples[0][r])
        .collect();
    eprintln!(
        "NULL CONTROL: A1 is the same config as A0. Its paired median is {:.3} ms and it wins \
         {}/{rounds}. No arm whose paired median is inside +/-{:.3} ms has been shown to do \
         anything. base median {base_med:.3} ms/tok.",
        median(&ctrl),
        ctrl.iter().filter(|d| **d < 0.0).count(),
        median(&ctrl).abs().max(0.001)
    );
    println!(
        "G4D-HOSTSHAPE {tag} null-control paired_dms={:.3} base_med_ms={base_med:.3}",
        median(&ctrl)
    );
}

#[test]
fn gemma4_dense_host_shape_is_bit_exact_synthetic() {
    if std::env::var("NV_G4D_HOST_SHAPE").as_deref() != Ok("1") {
        panic!("set NV_G4D_HOST_SHAPE=1 to run this GPU test (it must never silently skip)");
    }
    ctx_or_panic();
    let layers = envn("NV_G4D_HS_LAYERS", 4);
    let hidden = envn("NV_G4D_HS_HIDDEN", 512);
    let inter = envn("NV_G4D_HS_INTER", 1024);
    let vocab = envn("NV_G4D_HS_VOCAB", 2048);
    let config = Gemma4Config::from_hf_json_str(&config_json(layers, hidden, inter, vocab, 64))
        .expect("config");
    let host = host_weights(&config, 0x51ded00d);
    fold_census_gate(&config, &host, 1024);
    let mut m = Gemma4Wgpu::new(config, &host, 1024).expect("build");
    rollback_gate(&mut m, "synthetic", 8);
    run_suite(
        &mut m,
        "synthetic",
        envn("NV_G4D_HS_CTX", 8),
        envn("NV_G4D_HS_ROUNDS", 0),
        envn("NV_G4D_HS_STEPS", 0),
        envn("NV_G4D_HS_GEN", 48),
    );
}

#[test]
#[ignore]
fn gemma4_dense_host_shape_real_weights() {
    if std::env::var("NV_G4D_HOST_SHAPE").as_deref() != Ok("1") {
        panic!("set NV_G4D_HOST_SHAPE=1 to run this GPU test (it must never silently skip)");
    }
    ctx_or_panic();
    let dir = std::path::PathBuf::from(
        std::env::var("NV_GEMMA4_DIR").expect("set NV_GEMMA4_DIR to a Gemma-4 dense snapshot"),
    );
    assert!(
        dir.join("config.json").exists(),
        "NV_GEMMA4_DIR={} has no config.json",
        dir.display()
    );
    eprintln!("[snapshot] {}", dir.display());
    let config = Gemma4Config::from_hf_json_file(&dir.join("config.json")).expect("config");
    let t0 = Instant::now();
    let loader = nv_weights::WeightLoader::open_dir(&dir, &candle_core::Device::Cpu)
        .expect("open safetensors");
    let host =
        nv_models::gemma4_wgpu::host_weights_from_loader(&config, &loader).expect("stage host");
    drop(loader);
    eprintln!("host weight staging {:.1}s", t0.elapsed().as_secs_f64());
    let max_seq = envn("NV_G4D_HS_MAXSEQ", 2048);
    let mut m = Gemma4Wgpu::new(config, &host, max_seq).expect("build");
    rollback_gate(&mut m, "gemma-4-31B-IT-NVFP4", 8);
    run_suite(
        &mut m,
        "gemma-4-31B-IT-NVFP4",
        envn("NV_G4D_HS_CTX", 16),
        envn("NV_G4D_HS_ROUNDS", 3),
        envn("NV_G4D_HS_STEPS", 32),
        envn("NV_G4D_HS_GEN", 32),
    );
}

fn build_real(dir: &std::path::Path, max_seq: usize) -> (Gemma4Wgpu, Gemma4Config) {
    let config = Gemma4Config::from_hf_json_file(&dir.join("config.json")).expect("config");
    let loader = nv_weights::WeightLoader::open_dir(dir, &candle_core::Device::Cpu)
        .expect("open safetensors");
    let host =
        nv_models::gemma4_wgpu::host_weights_from_loader(&config, &loader).expect("stage host");
    drop(loader);
    let m = Gemma4Wgpu::new(config.clone(), &host, max_seq).expect("build");
    (m, config)
}

#[test]
#[ignore]
fn gemma4_dense_marginal_dispatch_cost() {
    if std::env::var("NV_G4D_HOST_SHAPE").as_deref() != Ok("1") {
        panic!("set NV_G4D_HOST_SHAPE=1 to run this GPU test (it must never silently skip)");
    }
    ctx_or_panic();
    let dir = std::path::PathBuf::from(
        std::env::var("NV_GEMMA4_DIR").expect("set NV_GEMMA4_DIR to a Gemma-4 dense snapshot"),
    );
    let max_seq = envn("NV_G4D_HS_MAXSEQ", 1024);
    let ctxlen = envn("NV_G4D_HS_CTX", 16);
    let steps = envn("NV_G4D_HS_STEPS", 24);
    let rounds = envn("NV_G4D_HS_ROUNDS", 11);
    let gen = envn("NV_G4D_HS_GEN", 32);

    std::env::set_var("NV_WGPU_FUSE", "0");
    let (mut unfused, _) = build_real(&dir, max_seq);
    std::env::remove_var("NV_WGPU_FUSE");
    let (mut fused, cfg) = build_real(&dir, max_seq);
    let d_unf = unfused.pass_count();
    let d_fus = fused.pass_count();
    assert!(
        d_unf > d_fus,
        "NV_WGPU_FUSE=0 did not add dispatches ({d_unf} vs {d_fus}); the knob never reached the graph"
    );
    eprintln!(
        "[g4d-fuse] {} layers: FUSE_ALL {d_fus} dispatches, FUSE=0 {d_unf}, delta {}",
        cfg.num_hidden_layers,
        d_unf - d_fus
    );

    let arm = ArmSpec {
        label: "fuse",
        preenc: true,
        k: 1,
        probe: 0,
    };
    let (ids_f, lg_f) = free_run(&mut fused, 3, ctxlen, gen, 1);
    let (ids_u, lg_u) = free_run(&mut unfused, 3, ctxlen, gen, 1);
    let id_match = ids_f == ids_u;
    let lg_diff = lg_f.iter().zip(&lg_u).filter(|(a, b)| a != b).count();
    eprintln!(
        "[g4d-fuse] fused vs unfused over {gen} tokens: ids {}, closing logits differ in {lg_diff} of {} slots",
        if id_match { "IDENTICAL" } else { "DIVERGED" },
        lg_f.len()
    );

    let mut sf: Vec<f64> = Vec::new();
    let mut su: Vec<f64> = Vec::new();
    for r in 0..rounds {
        wait_idle("g4d-fuse");
        if r % 2 == 0 {
            sf.push(measure_once(&mut fused, &arm, ctxlen, steps));
            su.push(measure_once(&mut unfused, &arm, ctxlen, steps));
        } else {
            su.push(measure_once(&mut unfused, &arm, ctxlen, steps));
            sf.push(measure_once(&mut fused, &arm, ctxlen, steps));
        }
        eprintln!(
            "[g4d-fuse] round {r}: fused {:.3} unfused {:.3} delta {:.3}",
            sf[r],
            su[r],
            su[r] - sf[r]
        );
    }
    let paired: Vec<f64> = (0..rounds).map(|r| su[r] - sf[r]).collect();
    let pm = median(&paired);
    let wins = paired.iter().filter(|d| **d > 0.0).count();
    let us = pm * 1000.0 / (d_unf - d_fus) as f64;
    eprintln!("\n================ MARGINAL DISPATCH :: gemma-4-31B-IT-NVFP4 ================");
    eprintln!(
        "fused median {:.3} ms/tok ({d_fus} dispatches), unfused median {:.3} ms/tok ({d_unf}), \
         paired median delta {pm:.3} ms over {rounds} rounds, unfused slower in {wins}/{rounds}",
        median(&sf),
        median(&su)
    );
    eprintln!("MARGINAL DISPATCH COST: {us:.3} us of wall per dispatch on this graph");
    println!(
        "G4D-DISPATCH fused_ms={:.3} unfused_ms={:.3} d_fused={d_fus} d_unfused={d_unf} paired_dms={pm:.3} us_per_dispatch={us:.3} wins={wins}/{rounds} ids_identical={id_match} logit_slots_differing={lg_diff}",
        median(&sf),
        median(&su)
    );
}
