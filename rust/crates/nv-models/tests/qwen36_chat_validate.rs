#![cfg(feature = "cuda")]

mod common;
use common::argmax;
use common::Arm;
use common::build;
use common::snapshot;
use candle_core::Device;
use nv_models::graph_engine::GraphedQwen3Moe;
use nv_models::qwen3_5_moe::{Qwen3Moe, Qwen3MoeConfig};
use nv_weights::{QuantizationConfig, WeightLoader};
use std::path::PathBuf;
use std::time::Instant;
use tokenizers::Tokenizer;

const MIN_DISTINCT_RATIO: f64 = 0.35;
const MAX_IMMEDIATE_RUN: usize = 4;
const MAX_CYCLE_REPEATS: usize = 3;
const MAX_NEW_TOKENS: usize = 96;
const PROBE_STEPS: usize = 6;

fn nan_count(row: &[f32]) -> usize {
    row.iter().filter(|v| !v.is_finite()).count()
}

fn row_health(row: &[f32]) -> String {
    let nan = row.iter().filter(|v| v.is_nan()).count();
    let inf = row.iter().filter(|v| v.is_infinite()).count();
    let finite: Vec<f32> = row.iter().copied().filter(|v| v.is_finite()).collect();
    let (mn, mx) = finite
        .iter()
        .fold((f32::MAX, f32::MIN), |(a, b), v| (a.min(*v), b.max(*v)));
    format!(
        "nan={nan}/{} inf={inf} min={:.3} max={:.3}",
        row.len(),
        if finite.is_empty() { f32::NAN } else { mn },
        if finite.is_empty() { f32::NAN } else { mx }
    )
}

fn longest_immediate_run(ids: &[u32]) -> usize {
    let (mut best, mut cur) = (1usize, 1usize);
    for w in ids.windows(2) {
        cur = if w[0] == w[1] { cur + 1 } else { 1 };
        best = best.max(cur);
    }
    best
}

fn tail_cycle(ids: &[u32]) -> Option<(usize, usize)> {
    for k in 1..=16usize {
        if ids.len() < k * 2 {
            break;
        }
        let tail = &ids[ids.len() - k..];
        let mut reps = 0usize;
        let mut j = ids.len();
        while j >= k && &ids[j - k..j] == tail {
            reps += 1;
            j -= k;
        }
        if reps >= MAX_CYCLE_REPEATS {
            return Some((k, reps));
        }
    }
    None
}

const ARMS: [Arm; 3] = [
    Arm {
        label: "grouped+routed",
        grouped: true,
        routing: true,
    },
    Arm {
        label: "grouped+unrouted",
        grouped: true,
        routing: false,
    },
    Arm {
        label: "plain",
        grouped: false,
        routing: false,
    },
];

#[test]
#[ignore = "loads the ~22 GB Qwen3.6 checkpoint; set NV_QWEN36_CHAT=1"]
fn qwen36_chat_produces_non_degenerate_answer() {
    if std::env::var("NV_QWEN36_CHAT").as_deref() != Ok("1") {
        panic!("set NV_QWEN36_CHAT=1 to run (it must never silently skip)");
    }
    let dir = snapshot();
    assert!(
        dir.join("model.safetensors").is_file()
            || dir.join("model.safetensors.index.json").is_file(),
        "checkpoint {dir:?} has no weights"
    );
    let tok = Tokenizer::from_file(dir.join("tokenizer.json")).expect("tokenizer.json");
    let eos: Vec<u32> = ["<|im_end|>", "<|endoftext|>"]
        .iter()
        .filter_map(|s| tok.token_to_id(s))
        .collect();
    assert_eq!(eos.len(), 2, "expected both stop tokens in this vocab");

    let q = std::env::var("NV_QWEN36_Q")
        .unwrap_or_else(|_| "What is the capital of France? Answer in one short sentence.".into());
    let prompt_text =
        format!("<|im_start|>user\n{q}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n");
    let prompt: Vec<u32> = tok
        .encode(prompt_text.as_str(), false)
        .expect("encode")
        .get_ids()
        .to_vec();

    let device = Device::new_cuda_with_stream(0).expect("cuda");
    eprintln!("CHAT basis: {dir:?} prompt_toks={}", prompt.len());

    let want = std::env::var("NV_QWEN36_ARM").unwrap_or_else(|_| "grouped+routed".into());
    let arm = ARMS
        .iter()
        .find(|a| a.label == want)
        .unwrap_or_else(|| panic!("unknown NV_QWEN36_ARM={want}"));

    let t_load = Instant::now();
    let mut eng = build(&dir, &device, arm);
    let load_s = t_load.elapsed().as_secs_f64();
    {
        let row = eng.prefill(&prompt).expect("prefill");
        let pre_nan = nan_count(&row);
        let mut cur = argmax(&row);
        let mut ids = vec![cur];
        let mut dec_nan = Vec::new();
        for step in 0..PROBE_STEPS {
            eng.forward_decode(cur)
                .unwrap_or_else(|e| panic!("arm {} decode {step}: {e:#}", arm.label));
            let r = eng.logits_host().expect("logits");
            dec_nan.push(nan_count(&r));
            cur = argmax(&r);
            ids.push(cur);
        }
        eprintln!(
            "CHAT probe arm={} load={load_s:.1}s capture={} prefill[{}] decode_nan={dec_nan:?} ids={ids:?}",
            arm.label,
            eng.capture_active(),
            row_health(&row)
        );
        assert!(
            pre_nan == 0 && dec_nan.iter().all(|n| *n == 0),
            "arm {} produces non-finite logits: Qwen3.6-35B-A3B-NVFP4 cannot generate on this path",
            arm.label
        );
    }
    eng.reset().expect("reset");

    let t0 = Instant::now();
    let row = eng.prefill(&prompt).expect("prefill");
    let prefill_ms = t0.elapsed().as_secs_f64() * 1e3;
    assert_eq!(row.len(), eng.vocab_size(), "logit row width");

    let mut cur = argmax(&row);
    let mut out: Vec<u32> = vec![cur];
    let mut times: Vec<f64> = Vec::new();
    let mut stopped = false;
    for step in 0..MAX_NEW_TOKENS {
        if eos.contains(&cur) {
            stopped = true;
            break;
        }
        let t = Instant::now();
        eng.forward_decode(cur)
            .unwrap_or_else(|e| panic!("decode step {step}: {e:#}"));
        let r = eng.logits_host().expect("logits");
        times.push(t.elapsed().as_secs_f64() * 1e3);
        assert!(
            r.iter().all(|v| v.is_finite()),
            "non-finite logits at decode step {step}: {}",
            row_health(&r)
        );
        cur = argmax(&r);
        out.push(cur);
    }

    let body: Vec<u32> = out.iter().copied().filter(|t| !eos.contains(t)).collect();
    let text = tok.decode(&body, false).expect("decode");
    let distinct = {
        let mut v = body.clone();
        v.sort_unstable();
        v.dedup();
        v.len()
    };
    let ratio = distinct as f64 / body.len().max(1) as f64;
    let run = longest_immediate_run(&body);
    let cyc = tail_cycle(&body);
    let med = {
        let mut s = times.clone();
        s.sort_by(|a, b| a.partial_cmp(b).unwrap());
        s[s.len() / 2]
    };
    eprintln!(
        "CHAT arm={} capture_active={} prompt_toks={} new_toks={} stopped_on_eos={stopped} prefill={prefill_ms:.1}ms median={med:.3}ms/tok tok_per_s={:.1} distinct={distinct}/{} ratio={ratio:.3} longest_run={run} tail_cycle={cyc:?}",
        arm.label,
        eng.capture_active(),
        prompt.len(),
        out.len(),
        1000.0 / med,
        body.len()
    );
    eprintln!("CHAT text: {text}");

    assert!(body.len() >= 4, "model emitted almost nothing");
    assert!(
        ratio >= MIN_DISTINCT_RATIO,
        "degenerate output, distinct ratio {ratio:.3} < {MIN_DISTINCT_RATIO}"
    );
    assert!(
        run <= MAX_IMMEDIATE_RUN,
        "degenerate output, {run} identical tokens in a row"
    );
    assert!(
        cyc.is_none(),
        "degenerate output, repeating tail cycle {cyc:?}"
    );
    assert!(
        text.chars().any(|c| c.is_ascii_alphabetic()),
        "decoded text has no letters: {text:?}"
    );
}
