#![cfg(feature = "cuda")]

mod common;
use common::argmax;
use candle_core::{DType, Device, Tensor};
use nv_models::gemma4::{Gemma4, Gemma4Cache, Gemma4Config, LayerType};
use nv_models::paged_fp8::{
    DeriveVPlan, PagedGemma4Cache, PagedKvFp8Pool, PagedPoolConfig, DERIVE_V_ENV,
};
use nv_weights::{QuantizationConfig, WeightLoader};
use std::path::Path;
use std::sync::{Arc, Mutex};
use tokenizers::Tokenizer;

const BLOCK_SIZE: usize = 16;
const PREFILL_CHUNK: usize = 1024;
const DECODE_STEPS: usize = 64;

const MIN_PROMPT_TOKENS: usize = 32768;

const PROBE_LEN: usize = 4096;

const PROBE_SCALE_TOL: f32 = 3e-3;

fn rows(logits: &Tensor) -> Vec<f32> {
    logits
        .to_dtype(DType::F32)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1()
        .unwrap()
}

fn build_prompt(tok: &Tokenizer, want: usize) -> Vec<u32> {
    const SEED: &str = "The paged key-value cache stores one block of keys and one block of \
        values for every sixteen positions of context. On the full-attention layers of this \
        checkpoint the two are not independent: the same normalised tensor feeds both, so the \
        values can be rebuilt from the keys by undoing the rotation and dividing by a scalar. \
        A reader that does this reads half as many bytes and holds half as much memory. ";
    let mut text = String::new();
    let mut ids: Vec<u32> = Vec::new();
    while ids.len() < want + 1 {
        for _ in 0..64 {
            text.push_str(SEED);
        }
        ids = tok
            .encode(text.as_str(), false)
            .expect("tokenize")
            .get_ids()
            .to_vec();
    }
    ids.truncate(want);
    ids.insert(0, 2);
    ids
}

struct Run {
    tokens: Vec<u32>,
    logits: Vec<f32>,
    probe_k: Vec<f32>,
    probe_v: Vec<f32>,
    derive_layers: usize,
    dispatches: u64,
}

#[allow(clippy::too_many_arguments)]
fn run(
    model: &Gemma4,
    device: &Device,
    pool_cfg: &PagedPoolConfig,
    plan: &DeriveVPlan,
    prompt: &[u32],
    probe_layer: usize,
    derive: bool,
) -> Run {
    std::env::set_var(DERIVE_V_ENV, if derive { "1" } else { "0" });
    let pool = Arc::new(Mutex::new(
        PagedKvFp8Pool::new_derive_v(pool_cfg.clone(), device, plan)
            .unwrap_or_else(|e| panic!("pool (derive={derive}): {e}")),
    ));
    let derive_layers = pool.lock().unwrap().derive_layers();
    let table: Vec<u32> = (0..pool_cfg.num_blocks as u32).collect();
    let mut cache = PagedGemma4Cache::new(pool.clone(), device).expect("cache");
    cache.set_block_table(&table).expect("block table");

    let mut last = Vec::new();
    let mut at = 0usize;
    while at < prompt.len() {
        let n = PREFILL_CHUNK.min(prompt.len() - at);
        let ids = &prompt[at..at + n];
        let tokens = Tensor::from_vec(ids.to_vec(), (1usize, n), device).unwrap();
        let pos: Vec<i32> = (at as i32..(at + n) as i32).collect();
        let pos = Tensor::from_vec(pos, n, device).unwrap();
        let logits = model
            .forward_with_cache_last(&tokens, &pos, &mut cache)
            .expect("prefill chunk");
        last = rows(&logits);
        at += n;
    }
    let first_logits = last.clone();
    let (probe_k, probe_v) = {
        let (k, v) = Gemma4Cache::view(&mut cache, probe_layer, PROBE_LEN).expect("probe view");
        (rows(&k), rows(&v))
    };

    let mut tok = argmax(&last);
    let mut out = vec![tok];
    let mut position = prompt.len();
    for _ in 1..DECODE_STEPS {
        let mut caches: Vec<&mut PagedGemma4Cache> = vec![&mut cache];
        let logits = model
            .forward_decode_batched(&[tok], &[position], &mut caches)
            .expect("decode step");
        let v = rows(&logits);
        tok = argmax(&v);
        position += 1;
        out.push(tok);
    }
    let dispatches = pool.lock().unwrap().derive_dispatches();
    drop(cache);
    Run {
        tokens: out,
        logits: first_logits,
        probe_k,
        probe_v,
        derive_layers,
        dispatches,
    }
}

fn rel_rms(got: &[f32], want: &[f32]) -> f32 {
    let num: f64 = got
        .iter()
        .zip(want)
        .map(|(a, b)| ((a - b) as f64).powi(2))
        .sum();
    let den: f64 = want.iter().map(|b| (*b as f64).powi(2)).sum();
    (num.sqrt() / den.sqrt()) as f32
}

#[test]
#[ignore]
fn derived_v_decodes_the_same_tokens_at_long_context() {
    if std::env::var("NV_KV_DERIVE_V_AB").as_deref() != Ok("1") {
        panic!("PRECONDITION NOT MET, THIS TEST EXECUTED NOTHING: set NV_KV_DERIVE_V_AB=1");
    }
    let dir = std::env::var("NV_CHAT_MODEL_DIR")
        .expect("PRECONDITION NOT MET, THIS TEST EXECUTED NOTHING: NV_CHAT_MODEL_DIR unset");
    let dir = Path::new(&dir);
    let device = Device::new_cuda(0).expect("cuda device 0");

    let raw_cfg = std::fs::read_to_string(dir.join("config.json")).expect("config");
    let cfg = Gemma4Config::from_hf_json_str(&raw_cfg).expect("parse cfg");
    let qcfg = QuantizationConfig::from_hf_json_str(&raw_cfg).expect("parse qcfg");
    let weights = WeightLoader::open_dir(dir, &device).expect("weights");
    let model = Gemma4::from_loader_quantized(cfg, &weights, &qcfg, &device).expect("model");
    let tok = Tokenizer::from_file(dir.join("tokenizer.json")).expect("tokenizer");

    let prompt = build_prompt(&tok, MIN_PROMPT_TOKENS);
    let ctx = prompt.len() + DECODE_STEPS + BLOCK_SIZE;
    let full_blocks = ctx.div_ceil(BLOCK_SIZE);
    let pool_cfg = PagedPoolConfig::from_gemma4_hybrid(model.config(), full_blocks, BLOCK_SIZE, 1);
    let plan = DeriveVPlan::from_model(&model, &pool_cfg).expect("derive plan");
    eprintln!(
        "[derive-ab] {} prompt tokens, {DECODE_STEPS} greedy steps, full_blocks {full_blocks}, \
         block_size {BLOCK_SIZE}, plan covers {} layer(s), rope_angles {}",
        prompt.len(),
        plan.layer_count(),
        plan.rope_angles()
    );
    assert!(
        prompt.len() > MIN_PROMPT_TOKENS,
        "prompt is only {} tokens; this gate is about long context",
        prompt.len()
    );

    let probe_layer = model
        .config()
        .layer_types
        .iter()
        .position(|k| *k == LayerType::FullAttention)
        .expect("a full-attention layer");

    let off = run(
        &model,
        &device,
        &pool_cfg,
        &plan,
        &prompt,
        probe_layer,
        false,
    );
    let on = run(
        &model,
        &device,
        &pool_cfg,
        &plan,
        &prompt,
        probe_layer,
        true,
    );
    std::env::remove_var(DERIVE_V_ENV);

    let first_div = off
        .tokens
        .iter()
        .zip(&on.tokens)
        .position(|(a, b)| a != b)
        .unwrap_or(DECODE_STEPS);
    let dot: f64 = on
        .probe_v
        .iter()
        .zip(&off.probe_v)
        .map(|(a, b)| (*a as f64) * (*b as f64))
        .sum();
    let norm: f64 = off.probe_v.iter().map(|b| (*b as f64).powi(2)).sum();
    let scale = (dot / norm) as f32;
    eprintln!(
        "[derive-ab] OFF derive layers {} dispatches {}\n\
         [derive-ab] ON  derive layers {} dispatches {}\n\
         [derive-ab] last-prefill logits rel-rms {:e}, first token divergence at step {first_div} \
         of {DECODE_STEPS}\n\
         [derive-ab] layer {probe_layer} over {PROBE_LEN} positions: K rel-rms {:e}, \
         V rel-rms {:e}, V projection on stored V {scale:.6}",
        off.derive_layers,
        off.dispatches,
        on.derive_layers,
        on.dispatches,
        rel_rms(&on.logits, &off.logits),
        rel_rms(&on.probe_k, &off.probe_k),
        rel_rms(&on.probe_v, &off.probe_v),
    );
    eprintln!(
        "[derive-ab] OFF ids {:?}",
        &off.tokens[..8.min(off.tokens.len())]
    );
    eprintln!(
        "[derive-ab] ON  ids {:?}",
        &on.tokens[..8.min(on.tokens.len())]
    );

    assert_eq!(
        off.derive_layers, 0,
        "the OFF run derived on {} layer(s)",
        off.derive_layers
    );
    assert_eq!(
        off.dispatches, 0,
        "the OFF run dispatched the derive kernel"
    );
    assert!(
        on.derive_layers > 0 && on.dispatches > 0,
        "the ON run derived on {} layer(s) and dispatched {} times; agreement between a \
         path and itself is not evidence",
        on.derive_layers,
        on.dispatches
    );
    assert_eq!(
        on.probe_k, off.probe_k,
        "the two runs did not cache the same K, so nothing downstream compares anything"
    );
    assert!(
        (scale - 1.0).abs() <= PROBE_SCALE_TOL,
        "the derived V sits at {scale} of the stored V, not 1.0: the 1/w_k the read path \
         multiplies by is wrong by {:.3}%",
        (scale - 1.0) * 100.0
    );
    assert_eq!(
        off.tokens, on.tokens,
        "greedy decode diverged at step {first_div}: reconstructing V changed the output"
    );
}
