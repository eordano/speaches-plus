#![cfg(feature = "wgpu")]

mod common;
use common::config_json_wrapped_text_config as config_json;
use common::distinct;
use common::envn;
use common::LcgCentered0p1Shift32 as Lcg;
use common::prompt_for;
use nv_models::gemma4::{Gemma4Config, LayerType};
use nv_models::gemma4_wgpu::{
    quantize_nvfp4_host, Gemma4Wgpu, HostBf16Lin, HostLayer, HostProj, HostWeights,
};
use common::gemma4_host_weights_quant_ffn_opt as host_weights;

fn ctx_or_panic() {
    match nv_kernels::wgpu_backend::WgpuContext::shared() {
        Ok(ctx) => eprintln!("[g4bd] adapter: {}", ctx.summary()),
        Err(e) => panic!("gemma4 batch decode needs a wgpu adapter: {e}"),
    }
}

fn solo_logits(
    m: &mut Gemma4Wgpu,
    slot: usize,
    prompt: &[u32],
    steps: usize,
) -> (Vec<u32>, Vec<Vec<u32>>) {
    m.reset_slot(slot).expect("reset slot");
    let (last, rest) = prompt.split_last().expect("prompt");
    let done = m.prefill_tokens(rest).expect("prefill_tokens");
    for t in &rest[done..] {
        m.prefill_step(*t).expect("prefill step");
    }
    let mut toks = Vec::with_capacity(steps);
    let mut bits = Vec::with_capacity(steps);
    let mut t = *last;
    for _ in 0..steps {
        let (n, lg) = m.decode_step_logits(t).expect("decode step logits");
        bits.push(lg.into_iter().map(f32::to_bits).collect::<Vec<u32>>());
        toks.push(n);
        t = n;
    }
    (toks, bits)
}

fn logit_bits(m: &mut Gemma4Wgpu, prompt: &[u32], steps: usize) -> (Vec<u32>, Vec<Vec<u32>>) {
    m.reset_slot(0).expect("reset slot 0");
    let (last, rest) = prompt.split_last().expect("prompt");
    let done = m.prefill_tokens(rest).expect("prefill_tokens");
    for t in &rest[done..] {
        m.prefill_step(*t).expect("prefill step");
    }
    let mut toks = Vec::with_capacity(steps);
    let mut bits = Vec::with_capacity(steps);
    let mut t = *last;
    for _ in 0..steps {
        let (n, lg) = m.decode_step_logits(t).expect("decode step logits");
        bits.push(lg.into_iter().map(f32::to_bits).collect::<Vec<u32>>());
        toks.push(n);
        t = n;
    }
    (toks, bits)
}

fn run_case(quant_ffn: bool, window: usize) {
    let layers = envn("NV_G4BD_LAYERS", 4);
    let hidden = envn("NV_G4BD_HIDDEN", 512);
    let inter = envn("NV_G4BD_INTER", 1024);
    let vocab = envn("NV_G4BD_VOCAB", 2048);
    let slots = envn("NV_G4BD_SLOTS", 4);
    let steps = envn("NV_G4BD_STEPS", 8);
    let max_seq = envn("NV_G4BD_MAXSEQ", 256);
    assert!(
        slots >= 2,
        "a 1-slot batch would make the whole test vacuous"
    );

    let raw = config_json(layers, hidden, inter, vocab, window);
    let config = Gemma4Config::from_hf_json_str(&raw).expect("config");
    let w = host_weights(&config, 0x9e3779b9, quant_ffn);
    let mut single = Gemma4Wgpu::new(config.clone(), &w, max_seq).expect("build single");
    let mut batched = Gemma4Wgpu::new_batched(config, &w, max_seq, slots).expect("build batched");
    drop(w);
    assert_eq!(
        single.batch_slots(),
        0,
        "the shipping constructor must stay single-stream"
    );
    assert_eq!(
        batched.batch_slots(),
        slots,
        "batch graph did not engage (quant_ffn={quant_ffn}); read the [gemma4_wgpu] boot lines above \
         for the disabler that fired -- the rest of this test would be vacuous"
    );
    assert!(
        batched.batch_pass_count() > 0,
        "batch pass list is empty; the test would be vacuous"
    );
    eprintln!(
        "[g4bd] quant_ffn={quant_ffn} window={window}: {} decode passes, {} batch passes for {slots} slots",
        batched.pass_count(),
        batched.batch_pass_count()
    );

    let prompts: Vec<Vec<u32>> = (0..slots)
        .map(|j| prompt_for(j, 17 + 3 * j, vocab))
        .collect();

    let (ref_toks, ref_bits) = logit_bits(&mut single, &prompts[0], steps);
    let (bt_toks, bt_bits) = logit_bits(&mut batched, &prompts[0], steps);
    let d = distinct(&ref_bits[0]);
    assert!(
        d > (vocab / 4).min(1000),
        "logits are degenerate ({d} distinct of {vocab}); the bit-compare would be vacuous"
    );
    for (i, (a, b)) in ref_bits.iter().zip(bt_bits.iter()).enumerate() {
        let diff = a.iter().zip(b.iter()).filter(|(x, y)| x != y).count();
        let worst = a
            .iter()
            .zip(b.iter())
            .map(|(x, y)| (f32::from_bits(*x) - f32::from_bits(*y)).abs())
            .fold(0.0f32, f32::max);
        assert_eq!(
            diff, 0,
            "step {i}: {diff} of {vocab} logit lanes differ once the batch graph is built \
             (max |delta| {worst:.3e})"
        );
    }
    assert_eq!(ref_toks, bt_toks, "single-stream token stream moved");
    eprintln!(
        "[g4bd] single-stream: {steps} steps bit-identical with and without the batch graph, {d} distinct logits"
    );

    batched.reset_slot(0).expect("reset slot 0");
    let mut b1 = Vec::with_capacity(steps);
    let mut t = {
        let (last, rest) = prompts[0].split_last().expect("prompt");
        let done = batched.prefill_tokens(rest).expect("prefill_tokens");
        for t in &rest[done..] {
            batched.prefill_step(*t).expect("prefill step");
        }
        *last
    };
    for _ in 0..steps {
        let out = batched
            .decode_step_batch(&[t])
            .expect("decode_step_batch b1");
        assert_eq!(out.len(), 1);
        t = out[0];
        b1.push(t);
    }
    assert_eq!(
        b1, ref_toks,
        "decode_step_batch at B == 1 diverged from the single-row decode"
    );
    eprintln!("[g4bd] B=1: {steps} tokens identical to the single-row path");

    let solo: Vec<(Vec<u32>, Vec<Vec<u32>>)> = (0..slots)
        .map(|j| solo_logits(&mut single, 0, &prompts[j], steps))
        .collect();

    for j in 1..slots {
        let same = (0..steps).all(|i| solo[j].1[i] == solo[0].1[i]);
        assert!(
            !same,
            "slot {j}'s solo logits equal slot 0's at every step; the cross-slot compare would be vacuous"
        );
    }

    let mut cur: Vec<u32> = Vec::with_capacity(slots);
    for (j, p) in prompts.iter().enumerate() {
        cur.push(batched.prefill_slot(j, p).expect("prefill slot"));
    }
    for (j, t) in cur.iter().enumerate() {
        assert_eq!(
            *t, solo[j].0[0],
            "slot {j}: prefill through the batched model already disagrees with the solo run"
        );
    }
    let mut worst_step = 0usize;
    for i in 1..steps {
        let nx = batched.decode_step_batch(&cur).expect("decode_step_batch");
        assert_eq!(nx.len(), slots);
        let lg = batched.batch_logits().expect("batch logits");
        assert_eq!(lg.len(), slots * vocab);
        for j in 0..slots {
            let want = &solo[j].1[i];
            let got: Vec<u32> = lg[j * vocab..(j + 1) * vocab]
                .iter()
                .map(|x| x.to_bits())
                .collect();
            let diff = want.iter().zip(got.iter()).filter(|(x, y)| x != y).count();
            let worst = want
                .iter()
                .zip(got.iter())
                .map(|(x, y)| (f32::from_bits(*x) - f32::from_bits(*y)).abs())
                .fold(0.0f32, f32::max);
            assert_eq!(
                diff, 0,
                "step {i} slot {j} (prompt {} tokens, pos {}): {diff} of {vocab} logit lanes differ \
                 from the same sequence run alone (max |delta| {worst:.3e}); \
                 next token {} vs {}",
                prompts[j].len(),
                batched.slot_pos(j),
                nx[j],
                solo[j].0[i]
            );
            assert_eq!(
                nx[j], solo[j].0[i],
                "step {i} slot {j}: sampled token differs"
            );
        }
        cur = nx;
        worst_step = i;
    }
    eprintln!(
        "[g4bd] B={slots}: all {slots} slots bit-identical to their solo runs through step {worst_step} \
         (prompt lengths {:?}, {vocab} logit lanes each)",
        prompts.iter().map(|p| p.len()).collect::<Vec<_>>()
    );

    let narrow = slots - 1;
    let mut cur: Vec<u32> = Vec::with_capacity(narrow);
    for j in 0..narrow {
        cur.push(batched.prefill_slot(j, &prompts[j]).expect("prefill slot"));
    }
    for i in 1..steps {
        let nx = batched
            .decode_step_batch(&cur)
            .expect("decode_step_batch narrow");
        assert_eq!(nx.len(), narrow);
        let lg = batched.batch_logits().expect("batch logits");
        for j in 0..narrow {
            let want = &solo[j].1[i];
            let got: Vec<u32> = lg[j * vocab..(j + 1) * vocab]
                .iter()
                .map(|x| x.to_bits())
                .collect();
            let diff = want.iter().zip(got.iter()).filter(|(x, y)| x != y).count();
            assert_eq!(
                diff, 0,
                "narrow batch B={narrow} of {slots}: step {i} slot {j}: {diff} of {vocab} logit lanes differ from the solo run"
            );
        }
        cur = nx;
    }
    eprintln!("[g4bd] B={narrow} on a {slots}-slot graph: still bit-identical to the solo runs");
}

#[test]
fn gemma4_batch_decode_matches_single_stream() {
    if std::env::var("NV_G4BD").as_deref() != Ok("1") {
        panic!("set NV_G4BD=1 to run this GPU test (it must never silently skip)");
    }
    ctx_or_panic();

    for window in [4096usize, 8] {
        run_case(false, window);
    }
}

#[test]
fn gemma4_batch_decode_q8_ffn_matches_single_stream() {
    if std::env::var("NV_G4BD").as_deref() != Ok("1") {
        panic!("set NV_G4BD=1 to run this GPU test (it must never silently skip)");
    }
    ctx_or_panic();
    run_case(true, 4096);
}

#[test]
#[ignore = "loads the ~20 GB Gemma-4-31B NVFP4 checkpoint; set NV_G4BD_REAL=1"]
fn real_gemma4_31b_batch_decode() {
    if std::env::var("NV_G4BD_REAL").as_deref() != Ok("1") {
        panic!("set NV_G4BD_REAL=1 to run this GPU test (it must never silently skip)");
    }
    ctx_or_panic();
    let slots = envn("NV_G4BD_REAL_SLOTS", 4);
    let steps = envn("NV_G4BD_REAL_STEPS", 8);
    let max_seq = envn("NV_G4BD_REAL_MAXSEQ", 512);
    let home = std::env::var("HOME").unwrap();
    let base = std::path::PathBuf::from(home)
        .join(".cache/huggingface/hub/models--nvidia--Gemma-4-31B-IT-NVFP4/snapshots");
    let dir = std::fs::read_dir(&base)
        .expect("hub snapshot dir")
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .find(|p| p.join("config.json").exists())
        .expect("no snapshot with config.json");
    eprintln!("[g4bd-real] loading {}", dir.display());
    let config = Gemma4Config::from_hf_json_file(&dir.join("config.json")).unwrap();
    let vocab = config.vocab_size;
    let t = std::time::Instant::now();
    let loader = nv_weights::WeightLoader::open_dir(&dir, &candle_core::Device::Cpu).unwrap();
    let host = nv_models::gemma4_wgpu::host_weights_from_loader(&config, &loader).unwrap();
    eprintln!("[g4bd-real] host staging {:.1}s", t.elapsed().as_secs_f64());
    let t = std::time::Instant::now();
    let mut m = Gemma4Wgpu::new_batched(config, &host, max_seq, slots).expect("build");
    drop(host);
    eprintln!(
        "[g4bd-real] built in {:.1}s: {} decode passes, {} batch passes for {} slots",
        t.elapsed().as_secs_f64(),
        m.pass_count(),
        m.batch_pass_count(),
        m.batch_slots()
    );
    assert_eq!(
        m.batch_slots(),
        slots,
        "the batch graph did not engage on the real checkpoint at default env; \
         read the [gemma4_wgpu] boot lines above for the disabler that fired"
    );

    let tokenizer = tokenizers::Tokenizer::from_file(dir.join("tokenizer.json")).unwrap();
    let seeds = [
        "The measurement of record for this repository carries its basis with every number, and ",
        "A kernel is fast, a format is not; the only operative quantity is bytes over rate, and ",
        "Batching amortizes one weight read across more rows, which is the largest lever, and ",
        "Speculative decoding cannot pay on a compute-bound model at any acceptance rate, and ",
    ];
    let prompts: Vec<Vec<u32>> = (0..slots)
        .map(|j| {
            let mut text = String::new();
            let want = 48 + 7 * j;
            while tokenizer
                .encode(text.as_str(), false)
                .unwrap()
                .get_ids()
                .len()
                < want
            {
                text.push_str(seeds[j % seeds.len()]);
            }
            let mut ids: Vec<u32> = vec![2];
            ids.extend(tokenizer.encode(text.as_str(), false).unwrap().get_ids());
            ids.truncate(want);
            ids
        })
        .collect();
    eprintln!(
        "[g4bd-real] prompt lengths {:?}",
        prompts.iter().map(|p| p.len()).collect::<Vec<_>>()
    );

    let t_solo = std::time::Instant::now();
    let solo: Vec<(Vec<u32>, Vec<Vec<u32>>)> = (0..slots)
        .map(|j| solo_logits(&mut m, 0, &prompts[j], steps))
        .collect();
    let solo_ms = t_solo.elapsed().as_secs_f64() * 1000.0 / (slots * steps) as f64;
    for j in 1..slots {
        assert!(
            (0..steps).any(|i| solo[j].1[i] != solo[0].1[i]),
            "slot {j}'s solo logits equal slot 0's at every step; the cross-slot compare would be vacuous"
        );
    }

    let mut cur: Vec<u32> = Vec::with_capacity(slots);
    for (j, p) in prompts.iter().enumerate() {
        cur.push(m.prefill_slot(j, p).expect("prefill slot"));
        assert_eq!(
            cur[j], solo[j].0[0],
            "slot {j}: prefill disagrees with the solo run"
        );
    }
    let t_batch = std::time::Instant::now();
    for i in 1..steps {
        let nx = m.decode_step_batch(&cur).expect("decode_step_batch");
        let lg = m.batch_logits().expect("batch logits");
        for j in 0..slots {
            let want = &solo[j].1[i];
            let got: Vec<u32> = lg[j * vocab..(j + 1) * vocab]
                .iter()
                .map(|x| x.to_bits())
                .collect();
            let diff = want.iter().zip(got.iter()).filter(|(x, y)| x != y).count();
            let worst = want
                .iter()
                .zip(got.iter())
                .map(|(x, y)| (f32::from_bits(*x) - f32::from_bits(*y)).abs())
                .fold(0.0f32, f32::max);
            assert_eq!(
                diff, 0,
                "step {i} slot {j}: {diff} of {vocab} logit lanes differ from the solo run \
                 (max |delta| {worst:.3e}); next token {} vs {}",
                nx[j], solo[j].0[i]
            );
            assert_eq!(
                nx[j], solo[j].0[i],
                "step {i} slot {j}: sampled token differs"
            );
        }
        cur = nx;
    }
    let batch_ms = t_batch.elapsed().as_secs_f64() * 1000.0 / (steps - 1) as f64;
    eprintln!(
        "[g4bd-real] B={slots} bit-identical to solo for {steps} steps x {vocab} lanes | \
         solo {solo_ms:.2} ms/token, batch step {batch_ms:.2} ms for {slots} tokens \
         ({:.2} ms/token, {:.2}x aggregate, {:.2}x per-stream latency)",
        batch_ms / slots as f64,
        solo_ms * slots as f64 / batch_ms,
        batch_ms / solo_ms
    );
}

#[test]
#[ignore = "loads the ~20 GB Gemma-4-31B NVFP4 checkpoint; set NV_G4BD_RATE=1"]
fn real_gemma4_31b_batch_decode_rate() {
    if std::env::var("NV_G4BD_RATE").as_deref() != Ok("1") {
        panic!("set NV_G4BD_RATE=1 to run this GPU test (it must never silently skip)");
    }
    ctx_or_panic();
    let slots = envn("NV_G4BD_RATE_SLOTS", 4);
    let steps = envn("NV_G4BD_RATE_STEPS", 16);
    let reps = envn("NV_G4BD_RATE_REPS", 3);
    let max_seq = envn("NV_G4BD_RATE_MAXSEQ", 1024);
    let home = std::env::var("HOME").unwrap();
    let base = std::path::PathBuf::from(home)
        .join(".cache/huggingface/hub/models--nvidia--Gemma-4-31B-IT-NVFP4/snapshots");
    let dir = std::fs::read_dir(&base)
        .expect("hub snapshot dir")
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .find(|p| p.join("config.json").exists())
        .expect("no snapshot with config.json");
    let config = Gemma4Config::from_hf_json_file(&dir.join("config.json")).unwrap();
    let loader = nv_weights::WeightLoader::open_dir(&dir, &candle_core::Device::Cpu).unwrap();
    let host = nv_models::gemma4_wgpu::host_weights_from_loader(&config, &loader).unwrap();
    let mut m = Gemma4Wgpu::new_batched(config, &host, max_seq, slots).expect("build");
    drop(host);
    assert_eq!(m.batch_slots(), slots, "batch graph did not engage");

    let tokenizer = tokenizers::Tokenizer::from_file(dir.join("tokenizer.json")).unwrap();
    let seed =
        "The measurement of record for this repository carries its basis with every number, and ";
    let mut text = String::new();
    let want = envn("NV_G4BD_RATE_PP", 64);
    while tokenizer
        .encode(text.as_str(), false)
        .unwrap()
        .get_ids()
        .len()
        < want
    {
        text.push_str(seed);
    }
    let mut prompt: Vec<u32> = vec![2];
    prompt.extend(tokenizer.encode(text.as_str(), false).unwrap().get_ids());
    prompt.truncate(want);

    let mut cur = Vec::with_capacity(slots);
    for j in 0..slots {
        cur.push(m.prefill_slot(j, &prompt).expect("prefill slot"));
    }
    let mut t0 = cur[0];

    let mut b1_ms: Vec<f64> = Vec::new();
    let mut bt_ms: Vec<f64> = Vec::new();
    let mut null_ms: Vec<f64> = Vec::new();
    for r in 0..=reps {
        m.select_slot(0).expect("slot 0");
        let t = std::time::Instant::now();
        for _ in 0..steps {
            t0 = m.decode_step(t0).expect("decode step");
        }
        let a = t.elapsed().as_secs_f64() * 1000.0 / steps as f64;

        let t = std::time::Instant::now();
        for _ in 0..steps {
            cur = m.decode_step_batch(&cur).expect("decode_step_batch");
        }
        let b = t.elapsed().as_secs_f64() * 1000.0 / steps as f64;

        m.select_slot(0).expect("slot 0");
        let t = std::time::Instant::now();
        for _ in 0..steps {
            t0 = m.decode_step(t0).expect("decode step");
        }
        let c = t.elapsed().as_secs_f64() * 1000.0 / steps as f64;

        if r == 0 {
            eprintln!("[g4bd-rate] warmup rep discarded: b1 {a:.2} | batch {b:.2} | b1' {c:.2} ms");
            continue;
        }
        eprintln!(
            "[g4bd-rate] rep {r}: b1 {a:.2} ms/tok | batch {b:.2} ms/step ({:.2} ms/tok) | null-control b1' {c:.2} ms/tok",
            b / slots as f64
        );
        b1_ms.push(a);
        bt_ms.push(b);
        null_ms.push(c);
    }
    let mean = |v: &[f64]| v.iter().sum::<f64>() / v.len() as f64;
    let (a, b, c) = (mean(&b1_ms), mean(&bt_ms), mean(&null_ms));
    eprintln!(
        "[g4bd-rate] B=1 {a:.2} ms/tok ({:.1} tok/s) | B={slots} {b:.2} ms/step = {:.2} ms/tok ({:.1} tok/s aggregate) \
         | {:.2}x aggregate at {:.2}x per-stream latency | NULL CONTROL b1 vs b1' {:.2}x ({c:.2} ms/tok)",
        1000.0 / a,
        b / slots as f64,
        slots as f64 * 1000.0 / b,
        a * slots as f64 / b,
        b / a,
        c / a
    );
}
