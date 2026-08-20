#![cfg(feature = "cuda")]

mod common;
use common::argmax;
use candle_core::{DType, Device, Tensor};
use nv_models::gemma4::{Gemma4, Gemma4Config};
use nv_models::paged_fp8::{
    PagedGemma4Cache, PagedKvFp8Pool, PagedPoolConfig, PAGED_ATTN_FP8_RING_ENV,
};
use nv_weights::{QuantizationConfig, WeightLoader};
use std::path::Path;
use std::sync::{Arc, Mutex};
use tokenizers::Tokenizer;

const BLOCK_SIZE: usize = 16;
const PREFILL_CHUNK: usize = 1024;
const DECODE_STEPS: usize = 64;

fn ring_capacity_slots(window: usize) -> usize {
    window + PREFILL_CHUNK + 128
}

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
    const SEED: &str = "A sliding-window layer stores its keys and values in a ring: position p \
        lands in slot p modulo the ring size, so once the context is longer than the window every \
        new write overwrites the slot of a position that has already left the window. A reader \
        that indexes through the same table as the writer sees exactly the live window and \
        nothing else. ";
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

#[derive(Clone, Copy, PartialEq)]
enum Arm {

    Dense,

    Kernel,

    RingKernel,
}

fn run(
    model: &Gemma4,
    device: &Device,
    pool_cfg: &PagedPoolConfig,
    prompt: &[u32],
    arm: Arm,
    chunk: usize,
) -> (Vec<u32>, Vec<f32>, Vec<f32>, u64) {
    match arm {
        Arm::Dense => {
            std::env::set_var("NV_PAGED_ATTN_FP8", "0");
            std::env::set_var(PAGED_ATTN_FP8_RING_ENV, "0");
        }
        Arm::Kernel => {
            std::env::remove_var("NV_PAGED_ATTN_FP8");
            std::env::set_var(PAGED_ATTN_FP8_RING_ENV, "0");
        }
        Arm::RingKernel => {
            std::env::remove_var("NV_PAGED_ATTN_FP8");
            std::env::remove_var(PAGED_ATTN_FP8_RING_ENV);
        }
    }
    let pool = Arc::new(Mutex::new(
        PagedKvFp8Pool::new(pool_cfg.clone(), device)
            .unwrap_or_else(|e| panic!("pool: {e}")),
    ));
    let table: Vec<u32> = (0..pool_cfg.num_blocks as u32).collect();
    let mut cache = PagedGemma4Cache::new(pool.clone(), device).expect("cache");
    cache.set_block_table(&table).expect("block table");

    let mut last = Vec::new();
    let mut at = 0usize;
    while at < prompt.len() {
        let n = chunk.min(prompt.len() - at);
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

    let mut tok = argmax(&last);
    let mut out = vec![tok];
    let mut position = prompt.len();
    let mut step1_logits: Vec<f32> = Vec::new();
    for _ in 1..DECODE_STEPS {
        let mut caches: Vec<&mut PagedGemma4Cache> = vec![&mut cache];
        let logits = model
            .forward_decode_batched(&[tok], &[position], &mut caches)
            .expect("decode step");
        let v = rows(&logits);
        if step1_logits.is_empty() {
            step1_logits = v.clone();
        }
        tok = argmax(&v);
        position += 1;
        out.push(tok);
    }
    let ring_decodes = cache.ring_decodes();
    (out, first_logits, step1_logits, ring_decodes)
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
fn a_ring_read_decodes_the_same_tokens_the_gather_fallback_decodes() {
    if std::env::var("NV_PAGED_RING_AB").as_deref() != Ok("1") {
        panic!("PRECONDITION NOT MET, THIS TEST EXECUTED NOTHING: set NV_PAGED_RING_AB=1");
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

    let window = model.config().sliding_window;
    assert!(window > 0, "this checkpoint has no sliding layers to gate");

    let want = std::env::var("NV_PAGED_RING_AB_TOKENS")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or_else(|| ring_capacity_slots(window) + window + PREFILL_CHUNK);
    let prompt = build_prompt(&tok, want);
    let wraps = prompt.len() > ring_capacity_slots(window);
    if wraps {
        eprintln!("[ring-ab] WRAPPED: {} tokens over a {}-slot ring", prompt.len(), ring_capacity_slots(window));
    } else {
        eprintln!(
            "[ring-ab] UNWRAPPED CONTROL: {} tokens inside a {}-slot ring; the ring never \
             overwrites, so any divergence here is NOT a wrap bug",
            prompt.len(),
            ring_capacity_slots(window)
        );
    }
    assert!(
        prompt.len() > window,
        "prompt of {} tokens does not even exceed the {window}-wide window, so no sliding \
         layer ever drops a position and the gate is vacuous",
        prompt.len()
    );
    let ctx = prompt.len() + DECODE_STEPS + BLOCK_SIZE;
    let full_blocks = ctx.div_ceil(BLOCK_SIZE);
    let pool_cfg = PagedPoolConfig::from_gemma4_hybrid(model.config(), full_blocks, BLOCK_SIZE, 1);
    eprintln!(
        "[ring-ab] {} prompt tokens, window {window}, ring capacity {} slots, {DECODE_STEPS} \
         greedy steps, full_blocks {full_blocks}",
        prompt.len(),
        ring_capacity_slots(window)
    );

    let control = std::env::var("NV_PAGED_RING_AB_CONTROL").as_deref() == Ok("1");

    let (off_tokens, off_logits, off_step1, off_ring) =
        run(&model, &device, &pool_cfg, &prompt, Arm::Kernel, PREFILL_CHUNK);

    let full_only = std::env::var("NV_PAGED_RING_AB_FULLONLY").as_deref() == Ok("1");
    let on_arm = match (control, full_only) {
        (true, _) => Arm::Kernel,
        (_, true) => Arm::Dense,
        _ => Arm::RingKernel,
    };
    if full_only {
        eprintln!(
            "[ring-ab] FULL-LAYER ISOLATION: kernel-vs-dense on the 10 full layers only; \
             sliding layers take the fallback in BOTH arms"
        );
    }
    let (on_tokens, on_logits, on_step1, on_ring) =
        run(&model, &device, &pool_cfg, &prompt, on_arm, PREFILL_CHUNK);
    if control {
        eprintln!(
            "[ring-ab] OFF/OFF CONTROL: both arms took the gather fallback. Any divergence \
             printed below is run-to-run nondeterminism, not the ring path."
        );
    }
    std::env::remove_var(PAGED_ATTN_FP8_RING_ENV);

    let first_div = off_tokens
        .iter()
        .zip(&on_tokens)
        .position(|(a, b)| a != b)
        .unwrap_or(DECODE_STEPS);
    eprintln!(
        "[ring-ab] OFF ring_decodes {off_ring}, ON ring_decodes {on_ring}\n\
         [ring-ab] last-prefill logits rel-rms {:e}, DECODE STEP 1 logits rel-rms {:e}, \
         first token divergence at step {first_div} of {DECODE_STEPS}\n\
         [ring-ab] OFF ids {:?}\n\
         [ring-ab] ON  ids {:?}",
        rel_rms(&on_logits, &off_logits),
        rel_rms(&on_step1, &off_step1),
        &off_tokens[..8.min(off_tokens.len())],
        &on_tokens[..8.min(on_tokens.len())],
    );

    assert_eq!(
        off_ring, 0,
        "the OFF run took the ring read path {off_ring} times; the default flipped"
    );
    assert!(
        control || full_only || on_ring >= DECODE_STEPS as u64,
        "the ON run dispatched the ring read only {on_ring} times over {DECODE_STEPS} steps; \
         agreement between a path and itself is not evidence"
    );
    let r = rel_rms(&on_step1, &off_step1);
    assert!(
        r < SLIDING_KERNEL_RELRMS_MAX_IS_ABOVE_THE_BENIGN_FULL_LAYER_4_56E_1,
        "reading the ring through the paged kernel moved the first decode step's logits by \
         rel-rms {r:e}, past the {:e} bound (tokens first differed at step {first_div}, which \
         on its own proves nothing -- benign rounding diverges too)",
        SLIDING_KERNEL_RELRMS_MAX_IS_ABOVE_THE_BENIGN_FULL_LAYER_4_56E_1
    );
}

#[test]
#[ignore]
fn the_paged_kernel_matches_the_dense_oracle_on_a_noring_pool() {
    if std::env::var("NV_PAGED_RING_AB").as_deref() != Ok("1") {
        panic!("PRECONDITION NOT MET, THIS TEST EXECUTED NOTHING: set NV_PAGED_RING_AB=1");
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

    let window = model.config().sliding_window;
    assert!(window > 0, "this checkpoint has no sliding layers to gate");
    let want = std::env::var("NV_PAGED_RING_AB_TOKENS")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(window + window / 2);
    let prompt = build_prompt(&tok, want);
    assert!(
        prompt.len() > window,
        "prompt of {} tokens never exceeds the {window}-wide window, so no sliding layer \
         drops a position and this gate is vacuous",
        prompt.len()
    );
    let ctx = prompt.len() + DECODE_STEPS + BLOCK_SIZE;
    let full_blocks = ctx.div_ceil(BLOCK_SIZE);
    let pool_cfg = PagedPoolConfig::from_gemma4(model.config(), full_blocks, BLOCK_SIZE);
    assert!(
        pool_cfg.layer_sliding.iter().all(|s| !s),
        "from_gemma4 must leave layer_sliding all false, or this is not the no-ring path"
    );
    eprintln!(
        "[noring-ab] {} prompt tokens, window {window}, {DECODE_STEPS} greedy steps, \
         non-hybrid pool ({} blocks/layer)",
        prompt.len(),
        full_blocks
    );

    let (dense_tokens, _, dense_step1, _) =
        run(&model, &device, &pool_cfg, &prompt, Arm::Dense, PREFILL_CHUNK);
    let (kern_tokens, _, kern_step1, _) =
        run(&model, &device, &pool_cfg, &prompt, Arm::Kernel, PREFILL_CHUNK);
    std::env::remove_var("NV_PAGED_ATTN_FP8");

    let first_div = dense_tokens
        .iter()
        .zip(&kern_tokens)
        .position(|(a, b)| a != b)
        .unwrap_or(DECODE_STEPS);
    eprintln!(
        "[noring-ab] decode step 1 logits rel-rms {:e}, first token divergence at step \
         {first_div} of {DECODE_STEPS}\n[noring-ab] dense {:?}\n[noring-ab] kern  {:?}",
        rel_rms(&kern_step1, &dense_step1),
        &dense_tokens[..8.min(dense_tokens.len())],
        &kern_tokens[..8.min(kern_tokens.len())],
    );
    let r = rel_rms(&kern_step1, &dense_step1);
    assert!(
        r < SLIDING_KERNEL_RELRMS_MAX_IS_ABOVE_THE_BENIGN_FULL_LAYER_4_56E_1,
        "the SHIPPED NV_KV_RING=0 decode path moved the first decode step's logits by rel-rms \
         {r:e} against the dense oracle, past the {:e} bound (tokens first differed at step \
         {first_div})",
        SLIDING_KERNEL_RELRMS_MAX_IS_ABOVE_THE_BENIGN_FULL_LAYER_4_56E_1
    );
}

fn run_nonpaged(
    model: &Gemma4,
    device: &Device,
    prompt: &[u32],
    chunk: usize,
) -> (Vec<u32>, Vec<f32>) {
    let vocab = model.config().vocab_size;
    let max_seq = prompt.len() + DECODE_STEPS + BLOCK_SIZE;
    let mut cache = model.new_kv_cache(max_seq).expect("kv cache");
    let seq = prompt.len();

    let mut v = Vec::new();
    let mut at = 0usize;
    while at < seq {
        let n = chunk.min(seq - at);
        let tokens = Tensor::from_vec(prompt[at..at + n].to_vec(), (1usize, n), device).unwrap();
        let pos: Vec<i32> = (at as i32..(at + n) as i32).collect();
        let pos = Tensor::from_vec(pos, n, device).unwrap();
        let logits = model
            .forward_with_cache(&tokens, &pos, &mut cache)
            .expect("nonpaged prefill");
        v = rows(&logits);
        at += n;
    }
    let mut tok = argmax(&v[v.len() - vocab..]);
    let mut out = vec![tok];
    let mut step1: Vec<f32> = Vec::new();
    for i in 1..DECODE_STEPS {
        let p = seq + i - 1;
        let t = Tensor::from_vec(vec![tok], (1usize, 1usize), device).unwrap();
        let pn = Tensor::from_vec(vec![p as i32], 1usize, device).unwrap();
        let logits = model
            .forward_with_cache(&t, &pn, &mut cache)
            .expect("nonpaged decode");
        let v = rows(&logits);
        let row = v[v.len() - vocab..].to_vec();
        if step1.is_empty() {
            step1 = row.clone();
        }
        tok = argmax(&row);
        out.push(tok);
    }
    (out, step1)
}

#[test]
#[ignore]
fn the_nonpaged_reference_says_which_paged_sliding_path_is_right() {
    if std::env::var("NV_PAGED_RING_AB").as_deref() != Ok("1") {
        panic!("PRECONDITION NOT MET, THIS TEST EXECUTED NOTHING: set NV_PAGED_RING_AB=1");
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

    let window = model.config().sliding_window;
    let want = std::env::var("NV_PAGED_RING_AB_TOKENS")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(window + window / 2);
    let prompt = build_prompt(&tok, want);
    assert!(
        prompt.len() > window,
        "prompt of {} tokens never exceeds the {window}-wide window; nothing is dropped and \
         the referee cannot distinguish anything",
        prompt.len()
    );
    let ctx = prompt.len() + DECODE_STEPS + BLOCK_SIZE;
    let pool_cfg =
        PagedPoolConfig::from_gemma4(model.config(), ctx.div_ceil(BLOCK_SIZE), BLOCK_SIZE);

    let (reference, ref_step1) = run_nonpaged(&model, &device, &prompt, NONPAGED_MAX_CHUNK_PAST_WINDOW_IS_SLIDING_COMPACT_SLACK_256);
    let (dense, _, dense_step1, _) =
        run(&model, &device, &pool_cfg, &prompt, Arm::Dense, NONPAGED_MAX_CHUNK_PAST_WINDOW_IS_SLIDING_COMPACT_SLACK_256);
    let (kernel, _, kern_step1, _) =
        run(&model, &device, &pool_cfg, &prompt, Arm::Kernel, NONPAGED_MAX_CHUNK_PAST_WINDOW_IS_SLIDING_COMPACT_SLACK_256);
    std::env::remove_var("NV_PAGED_ATTN_FP8");

    let agree = |a: &[u32]| {
        a.iter()
            .zip(&reference)
            .position(|(x, y)| x != y)
            .unwrap_or(DECODE_STEPS)
    };
    let (d, k) = (agree(&dense), agree(&kernel));
    eprintln!(
        "[referee] {} prompt tokens, window {window}, {DECODE_STEPS} greedy steps\n\
         [referee] non-paged vs paged DENSE  : first divergence at step {d}\n\
         [referee] non-paged vs paged KERNEL : first divergence at step {k}\n\
         [referee] ref    {:?}\n[referee] dense  {:?}\n[referee] kernel {:?}",
        prompt.len(),
        &reference[..8.min(reference.len())],
        &dense[..8.min(dense.len())],
        &kernel[..8.min(kernel.len())],
    );

    let dr = rel_rms(&dense_step1, &ref_step1);
    let kr = rel_rms(&kern_step1, &ref_step1);
    eprintln!(
        "[referee] step-1 distance to the reference: dense {dr:e}, kernel {kr:e} \
         (closer wins; bound {:e})",
        SLIDING_KERNEL_RELRMS_MAX_IS_ABOVE_THE_BENIGN_FULL_LAYER_4_56E_1
    );
    assert!(
        dr < SLIDING_KERNEL_RELRMS_MAX_IS_ABOVE_THE_BENIGN_FULL_LAYER_4_56E_1 && kr < SLIDING_KERNEL_RELRMS_MAX_IS_ABOVE_THE_BENIGN_FULL_LAYER_4_56E_1,
        "a paged path is grossly far from the non-paged reference: dense {dr:e}, kernel \
         {kr:e}, bound {:e} (token divergence was dense {d}, kernel {k}, which decides \
         nothing on its own)",
        SLIDING_KERNEL_RELRMS_MAX_IS_ABOVE_THE_BENIGN_FULL_LAYER_4_56E_1
    );
}

const CHUNK_PREFILL_RELRMS_MAX_IS_2X_THE_WORST_LEGITIMATE_CHUNK_CASE_2_5E_1: f32 = 5e-1;

const SLIDING_KERNEL_RELRMS_MAX_IS_ABOVE_THE_BENIGN_FULL_LAYER_4_56E_1: f32 = 6e-1;

#[test]
#[ignore]
fn paged_prefill_is_invariant_to_chunk_size() {
    if std::env::var("NV_PAGED_RING_AB").as_deref() != Ok("1") {
        panic!("PRECONDITION NOT MET, THIS TEST EXECUTED NOTHING: set NV_PAGED_RING_AB=1");
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

    let window = model.config().sliding_window;
    let want = std::env::var("NV_PAGED_RING_AB_TOKENS")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(window + window / 2);
    let prompt = build_prompt(&tok, want);
    let ctx = prompt.len() + DECODE_STEPS + BLOCK_SIZE;
    let pool_cfg =
        PagedPoolConfig::from_gemma4(model.config(), ctx.div_ceil(BLOCK_SIZE), BLOCK_SIZE);

    let (big, big_pre, _, _) = run(&model, &device, &pool_cfg, &prompt, Arm::Dense, 1024);
    let (small, small_pre, _, _) = run(&model, &device, &pool_cfg, &prompt, Arm::Dense, 256);
    std::env::remove_var("NV_PAGED_ATTN_FP8");
    let div = big
        .iter()
        .zip(&small)
        .position(|(a, b)| a != b)
        .unwrap_or(DECODE_STEPS);

    eprintln!(
        "[chunk-inv] {} prompt tokens, window {window}, dense arm at chunk 1024 vs 256\n\
         [chunk-inv] LAST-PREFILL logits rel-rms {:e}\n\
         [chunk-inv] first divergence at step {div} of {DECODE_STEPS}\n\
         [chunk-inv] 1024 {:?}\n[chunk-inv]  256 {:?}",
        prompt.len(),
        rel_rms(&small_pre, &big_pre),
        &big[..8.min(big.len())],
        &small[..8.min(small.len())],
    );
    let r = rel_rms(&small_pre, &big_pre);
    assert!(
        r < CHUNK_PREFILL_RELRMS_MAX_IS_2X_THE_WORST_LEGITIMATE_CHUNK_CASE_2_5E_1,
        "chunk 1024 vs 256 moved the last-prefill logits by rel-rms {r:e}, past the {:e} bound. \
         Token divergence alone would not prove this (it appeared at step {div}, and greedy \
         decode diverges on rounding sooner or later); the magnitude does",
        CHUNK_PREFILL_RELRMS_MAX_IS_2X_THE_WORST_LEGITIMATE_CHUNK_CASE_2_5E_1
    );
}

const NONPAGED_MAX_CHUNK_PAST_WINDOW_IS_SLIDING_COMPACT_SLACK_256: usize = 256;

const PPL_EVAL_TOKENS: usize = 256;

const W4A4_PPL_PROMPT_FITS_THE_NONPAGED_SLIDING_SLACK_1024: usize = 1024;

const FUSED_QKV_BITWISE_SAFE_MAX_M_16: usize = 16;

const SPLITTING_SUB_FLOOR_COSTS_0_34_NLL_SO_ONE_CHUNK_IS_BOUNDED_AT_0_30: f64 = 0.30;

const RING_PPL_VS_NONPAGED_MAX_IS_2X_THE_FP8_KV_COST_OF_7_08E_2: f64 = 1.5e-1;

fn nll_of(logits_row: &[f32], target: u32) -> f64 {
    let m = logits_row.iter().cloned().fold(f32::NEG_INFINITY, f32::max) as f64;
    let mut sum = 0.0f64;
    for &x in logits_row {
        sum += ((x as f64) - m).exp();
    }
    (m + sum.ln()) - (logits_row[target as usize] as f64)
}

fn teacher_forced_ppl(
    model: &Gemma4,
    device: &Device,
    pool_cfg: &PagedPoolConfig,
    prompt: &[u32],
    arm: Arm,
    chunk: usize,
) -> (f64, u64) {
    match arm {
        Arm::Dense => {
            std::env::set_var("NV_PAGED_ATTN_FP8", "0");
            std::env::set_var(PAGED_ATTN_FP8_RING_ENV, "0");
        }
        Arm::Kernel => {
            std::env::remove_var("NV_PAGED_ATTN_FP8");
            std::env::set_var(PAGED_ATTN_FP8_RING_ENV, "0");
        }
        Arm::RingKernel => {
            std::env::remove_var("NV_PAGED_ATTN_FP8");
            std::env::remove_var(PAGED_ATTN_FP8_RING_ENV);
        }
    }
    let pool = Arc::new(Mutex::new(
        PagedKvFp8Pool::new(pool_cfg.clone(), device).expect("pool"),
    ));
    let table: Vec<u32> = (0..pool_cfg.num_blocks as u32).collect();
    let mut cache = PagedGemma4Cache::new(pool, device).expect("cache");
    cache.set_block_table(&table).expect("block table");

    let split = prompt.len() - PPL_EVAL_TOKENS;
    let mut last: Vec<f32> = Vec::new();
    let mut at = 0usize;
    while at < split {
        let n = chunk.min(split - at);
        let tokens = Tensor::from_vec(prompt[at..at + n].to_vec(), (1usize, n), device).unwrap();
        let pos: Vec<i32> = (at as i32..(at + n) as i32).collect();
        let pos = Tensor::from_vec(pos, n, device).unwrap();
        let logits = model
            .forward_with_cache_last(&tokens, &pos, &mut cache)
            .expect("ppl prefill");
        last = rows(&logits);
        at += n;
    }

    let mut total = nll_of(&last, prompt[split]);
    let mut counted = 1u64;
    for i in split..prompt.len() - 1 {
        let mut caches: Vec<&mut PagedGemma4Cache> = vec![&mut cache];
        let logits = model
            .forward_decode_batched(&[prompt[i]], &[i], &mut caches)
            .expect("ppl decode");
        let v = rows(&logits);
        total += nll_of(&v, prompt[i + 1]);
        counted += 1;
    }
    let ring = cache.ring_decodes();
    (total / counted as f64, ring)
}

#[test]
#[ignore]
fn ring_reads_track_the_nonpaged_reference_perplexity() {
    if std::env::var("NV_PAGED_RING_AB").as_deref() != Ok("1") {
        panic!("PRECONDITION NOT MET, THIS TEST EXECUTED NOTHING: set NV_PAGED_RING_AB=1");
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

    let window = model.config().sliding_window;
    let want = window + window / 2 + PPL_EVAL_TOKENS;
    let prompt = build_prompt(&tok, want);
    assert!(
        prompt.len() - PPL_EVAL_TOKENS > window,
        "the scored region must start past the {window}-wide window or no sliding layer has \
         dropped anything and the eval is vacuous"
    );
    let ctx = prompt.len() + BLOCK_SIZE;
    let pool_cfg =
        PagedPoolConfig::from_gemma4_hybrid(model.config(), ctx.div_ceil(BLOCK_SIZE), BLOCK_SIZE, 1);

    let chunk = NONPAGED_MAX_CHUNK_PAST_WINDOW_IS_SLIDING_COMPACT_SLACK_256;
    let off = teacher_forced_ppl_nonpaged(&model, &device, &prompt, chunk);
    let off_ring = 0u64;
    let (on, on_ring) =
        teacher_forced_ppl(&model, &device, &pool_cfg, &prompt, Arm::RingKernel, chunk);
    std::env::remove_var(PAGED_ATTN_FP8_RING_ENV);

    let rel = (on - off).abs() / off;
    eprintln!(
        "[ring-ppl] {} prompt tokens, scoring the last {PPL_EVAL_TOKENS}, window {window}\n\
         [ring-ppl] non-paged reference mean NLL {off:.6} ppl {:.4}  (ring dispatches {off_ring})\n\
         [ring-ppl] ring ON  mean NLL {on:.6} ppl {:.4}  (ring dispatches {on_ring})\n\
         [ring-ppl] relative delta {rel:.3e}, bound {:.0e}",
        prompt.len(),
        off.exp(),
        on.exp(),
        RING_PPL_VS_NONPAGED_MAX_IS_2X_THE_FP8_KV_COST_OF_7_08E_2
    );
    assert_eq!(off_ring, 0, "the reference arm is not paged and cannot dispatch a ring read");
    assert!(
        on_ring >= PPL_EVAL_TOKENS as u64,
        "the ON arm dispatched the ring read only {on_ring} times over {PPL_EVAL_TOKENS} \
         scored steps; agreement between a path and itself is not evidence"
    );
    assert!(
        rel < RING_PPL_VS_NONPAGED_MAX_IS_2X_THE_FP8_KV_COST_OF_7_08E_2,
        "the ring read path sits {rel:.3e} from the NON-PAGED reference in teacher-forced \
         perplexity, past the {:.0e} bound. The oracle here is deliberately NOT the paged gather \
         fallback: that path measures 71% from this same reference and on the wrong side, so \
         agreeing with it would be the wrong contract. The residual this bound allows is the \
         fp8 KV quantisation cost, measured at 7.08e-2",
        RING_PPL_VS_NONPAGED_MAX_IS_2X_THE_FP8_KV_COST_OF_7_08E_2
    );
}

fn teacher_forced_ppl_nonpaged(
    model: &Gemma4,
    device: &Device,
    prompt: &[u32],
    chunk: usize,
) -> f64 {
    let vocab = model.config().vocab_size;
    let mut cache = model.new_kv_cache(prompt.len() + BLOCK_SIZE).expect("kv cache");
    let split = prompt.len() - PPL_EVAL_TOKENS;
    let mut last: Vec<f32> = Vec::new();
    let mut at = 0usize;
    while at < split {
        let n = chunk.min(split - at);
        let tokens = Tensor::from_vec(prompt[at..at + n].to_vec(), (1usize, n), device).unwrap();
        let pos: Vec<i32> = (at as i32..(at + n) as i32).collect();
        let pos = Tensor::from_vec(pos, n, device).unwrap();
        let logits = model
            .forward_with_cache(&tokens, &pos, &mut cache)
            .expect("nonpaged ppl prefill");
        let v = rows(&logits);
        last = v[v.len() - vocab..].to_vec();
        at += n;
    }
    let mut total = nll_of(&last, prompt[split]);
    let mut counted = 1u64;
    for i in split..prompt.len() - 1 {
        let t = Tensor::from_vec(vec![prompt[i]], (1usize, 1usize), device).unwrap();
        let pn = Tensor::from_vec(vec![i as i32], 1usize, device).unwrap();
        let logits = model
            .forward_with_cache(&t, &pn, &mut cache)
            .expect("nonpaged ppl decode");
        let v = rows(&logits);
        total += nll_of(&v[v.len() - vocab..], prompt[i + 1]);
        counted += 1;
    }
    total / counted as f64
}

#[test]
#[ignore]
fn the_nonpaged_reference_ppl_says_which_sliding_read_is_right() {
    if std::env::var("NV_PAGED_RING_AB").as_deref() != Ok("1") {
        panic!("PRECONDITION NOT MET, THIS TEST EXECUTED NOTHING: set NV_PAGED_RING_AB=1");
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

    let window = model.config().sliding_window;
    let want = window + window / 2 + PPL_EVAL_TOKENS;
    let prompt = build_prompt(&tok, want);
    let ctx = prompt.len() + BLOCK_SIZE;

    let reference = teacher_forced_ppl_nonpaged(&model, &device, &prompt, NONPAGED_MAX_CHUNK_PAST_WINDOW_IS_SLIDING_COMPACT_SLACK_256);
    let hybrid =
        PagedPoolConfig::from_gemma4_hybrid(model.config(), ctx.div_ceil(BLOCK_SIZE), BLOCK_SIZE, 1);
    let (fallback, _) =
        teacher_forced_ppl(&model, &device, &hybrid, &prompt, Arm::Kernel, NONPAGED_MAX_CHUNK_PAST_WINDOW_IS_SLIDING_COMPACT_SLACK_256);
    let (ring, ring_n) =
        teacher_forced_ppl(&model, &device, &hybrid, &prompt, Arm::RingKernel, NONPAGED_MAX_CHUNK_PAST_WINDOW_IS_SLIDING_COMPACT_SLACK_256);
    let noring = PagedPoolConfig::from_gemma4(model.config(), ctx.div_ceil(BLOCK_SIZE), BLOCK_SIZE);
    let (shipped, _) =
        teacher_forced_ppl(&model, &device, &noring, &prompt, Arm::Kernel, NONPAGED_MAX_CHUNK_PAST_WINDOW_IS_SLIDING_COMPACT_SLACK_256);
    std::env::remove_var(PAGED_ATTN_FP8_RING_ENV);
    std::env::remove_var("NV_PAGED_ATTN_FP8");

    let d = |x: f64| (x - reference).abs() / reference;
    eprintln!(
        "[ppl-referee] {plen} prompt tokens, scoring the last {PPL_EVAL_TOKENS}, window {window}\n\
         [ppl-referee] non-paged reference      NLL {reference:.6} ppl {:.4}\n\
         [ppl-referee] paged gather fallback    NLL {fallback:.6} ppl {:.4}  rel {:.3e}\n\
         [ppl-referee] paged kernel + ring      NLL {ring:.6} ppl {:.4}  rel {:.3e}  (dispatches {ring_n})\n\
         [ppl-referee] paged kernel, no-ring pool NLL {shipped:.6} ppl {:.4}  rel {:.3e}",
        reference.exp(),
        fallback.exp(),
        d(fallback),
        ring.exp(),
        d(ring),
        shipped.exp(),
        d(shipped),
        plen = prompt.len(),
    );
    assert!(
        ring_n >= PPL_EVAL_TOKENS as u64,
        "the ring arm dispatched the ring read only {ring_n} times; it did not take the path"
    );
}

const REFERENCE_CHUNK_DRIFT_MAX_IS_WELL_UNDER_THE_7_11E_1_IT_ADJUDICATES: f64 = 1e-1;

#[test]
#[ignore]
fn the_nonpaged_reference_is_chunk_robust_enough_to_be_a_reference() {
    if std::env::var("NV_PAGED_RING_AB").as_deref() != Ok("1") {
        panic!("PRECONDITION NOT MET, THIS TEST EXECUTED NOTHING: set NV_PAGED_RING_AB=1");
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

    let window = model.config().sliding_window;
    let prompt = build_prompt(&tok, window + window / 2 + PPL_EVAL_TOKENS);
    let big = teacher_forced_ppl_nonpaged(
        &model,
        &device,
        &prompt,
        NONPAGED_MAX_CHUNK_PAST_WINDOW_IS_SLIDING_COMPACT_SLACK_256,
    );
    let small = teacher_forced_ppl_nonpaged(&model, &device, &prompt, 128);
    let rel = (small - big).abs() / big;
    let (big_ppl, small_ppl) = (big.exp(), small.exp());
    let bound = REFERENCE_CHUNK_DRIFT_MAX_IS_WELL_UNDER_THE_7_11E_1_IT_ADJUDICATES;
    let plen = prompt.len();
    eprintln!(
        "[ref-robust] {plen} prompt tokens, scoring the last {PPL_EVAL_TOKENS}, window {window}\n\
         [ref-robust] chunk 256 mean NLL {big:.6} ppl {big_ppl:.4}\n\
         [ref-robust] chunk 128 mean NLL {small:.6} ppl {small_ppl:.4}\n\
         [ref-robust] relative drift {rel:.3e}, bound {bound:.1e}"
    );
    assert!(
        rel < REFERENCE_CHUNK_DRIFT_MAX_IS_WELL_UNDER_THE_7_11E_1_IT_ADJUDICATES,
        "the non-paged reference moved {rel:.3e} between prefill chunk 256 and 128. It is used \
         to adjudicate a 7.11e-1 gap between the paged kernel and the paged gather fallback, and \
         it had to be run at 256 because its compacting buffer refuses more past the window -- \
         the same chunk size at which the fallback's quality collapses. A reference that drifts \
         with chunk size cannot separate them, and every verdict resting on it is void"
    );
}

const SHIPPED_KERNEL_CHUNK_SPREAD_MAX_IS_THE_REFERENCES_OWN_8_1E_2_DOUBLED: f64 = 1.6e-1;

#[test]
#[ignore]
fn the_shipped_kernel_is_no_more_chunk_sensitive_than_the_reference() {
    if std::env::var("NV_PAGED_RING_AB").as_deref() != Ok("1") {
        panic!("PRECONDITION NOT MET, THIS TEST EXECUTED NOTHING: set NV_PAGED_RING_AB=1");
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

    let window = model.config().sliding_window;
    let prompt = build_prompt(&tok, window + window / 2 + PPL_EVAL_TOKENS);
    let ctx = prompt.len() + BLOCK_SIZE;
    let hybrid =
        PagedPoolConfig::from_gemma4_hybrid(model.config(), ctx.div_ceil(BLOCK_SIZE), BLOCK_SIZE, 1);

    let chunks = [1024usize, 256, 128];
    let mut kernel = Vec::new();
    let mut fallback = Vec::new();
    for c in chunks {
        kernel.push(teacher_forced_ppl(&model, &device, &hybrid, &prompt, Arm::RingKernel, c).0);

        fallback.push(teacher_forced_ppl(&model, &device, &hybrid, &prompt, Arm::Kernel, c).0);
    }
    std::env::remove_var(PAGED_ATTN_FP8_RING_ENV);
    std::env::remove_var("NV_PAGED_ATTN_FP8");

    let spread = |v: &[f64]| {
        let lo = v.iter().cloned().fold(f64::INFINITY, f64::min);
        let hi = v.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        (hi - lo) / lo
    };
    let (ks, fs) = (spread(&kernel), spread(&fallback));
    let plen = prompt.len();
    let bound = SHIPPED_KERNEL_CHUNK_SPREAD_MAX_IS_THE_REFERENCES_OWN_8_1E_2_DOUBLED;
    eprintln!(
        "[chunk-map] {plen} prompt tokens, scoring the last {PPL_EVAL_TOKENS}, window {window}\n\
         [chunk-map] chunks {chunks:?}\n\
         [chunk-map] shipped kernel NLL {kernel:.6?}  spread {ks:.3e}\n\
         [chunk-map] gather fallback NLL {fallback:.6?}  spread {fs:.3e}\n\
         [chunk-map] reference drifts 8.098e-2 between 256 and 128, for scale"
    );
    assert!(
        ks < bound,
        "the SHIPPED sliding-layer decode moved {ks:.3e} across prefill chunk sizes {chunks:?}, \
         past the {bound:.1e} bound. The non-paged reference itself drifts 8.098e-2 between 256 \
         and 128, so some movement is expected and this bound is twice that; the gather \
         fallback's own spread is {fs:.3e} and is the defect this is guarding the default \
         against"
    );
}

#[test]
#[ignore]
fn the_mk_prefill_path_matches_the_gather_prefill_it_replaces() {
    if std::env::var("NV_PAGED_RING_AB").as_deref() != Ok("1") {
        panic!("PRECONDITION NOT MET, THIS TEST EXECUTED NOTHING: set NV_PAGED_RING_AB=1");
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

    let window = model.config().sliding_window;
    let prompt = build_prompt(&tok, window + window / 2 + PPL_EVAL_TOKENS);
    let ctx = prompt.len() + BLOCK_SIZE;
    let hybrid =
        PagedPoolConfig::from_gemma4_hybrid(model.config(), ctx.div_ceil(BLOCK_SIZE), BLOCK_SIZE, 1);
    let chunk = NONPAGED_MAX_CHUNK_PAST_WINDOW_IS_SLIDING_COMPACT_SLACK_256;

    let reference = teacher_forced_ppl_nonpaged(&model, &device, &prompt, chunk);

    std::env::remove_var("NV_PAGED_PREFILL_FP8");
    let t0 = std::time::Instant::now();
    let (gather, _) = teacher_forced_ppl(&model, &device, &hybrid, &prompt, Arm::RingKernel, chunk);
    let gather_s = t0.elapsed().as_secs_f64();

    std::env::set_var("NV_PAGED_PREFILL_FP8", "1");
    let t1 = std::time::Instant::now();
    let (mk, _) = teacher_forced_ppl(&model, &device, &hybrid, &prompt, Arm::RingKernel, chunk);
    let mk_s = t1.elapsed().as_secs_f64();
    std::env::remove_var("NV_PAGED_PREFILL_FP8");
    std::env::remove_var(PAGED_ATTN_FP8_RING_ENV);

    let d = |x: f64| (x - reference).abs() / reference;
    eprintln!(
        "[mk-prefill] {} prompt tokens, scoring the last {PPL_EVAL_TOKENS}, window {window}, \
         chunk {chunk}\n\
         [mk-prefill] non-paged reference   NLL {reference:.6}\n\
         [mk-prefill] gather prefill        NLL {gather:.6}  rel {:.3e}  {gather_s:.1}s\n\
         [mk-prefill] mk (fp8 in place)     NLL {mk:.6}  rel {:.3e}  {mk_s:.1}s",
        prompt.len(),
        d(gather),
        d(mk),
    );
    assert!(
        (mk - gather).abs() / gather < 1e-6 || d(mk) <= d(gather),
        "the mk prefill path is FURTHER from the non-paged reference than the gather prefill it \
         replaces: mk {mk:.6} (rel {:.3e}) vs gather {gather:.6} (rel {:.3e}), reference \
         {reference:.6}. Reading fp8 in place must not cost quality against the path that \
         dequantises it first",
        d(mk),
        d(gather),
    );
}

fn prefill_and_hash_kv(
    model: &Gemma4,
    device: &Device,
    pool_cfg: &PagedPoolConfig,
    prompt: &[u32],
    chunk: usize,
    layers: &[usize],
) -> Vec<Vec<f32>> {
    std::env::remove_var("NV_PAGED_ATTN_FP8");
    std::env::remove_var(PAGED_ATTN_FP8_RING_ENV);
    let pool = Arc::new(Mutex::new(
        PagedKvFp8Pool::new(pool_cfg.clone(), device).expect("pool"),
    ));
    let table: Vec<u32> = (0..pool_cfg.num_blocks as u32).collect();
    let mut cache = PagedGemma4Cache::new(pool.clone(), device).expect("cache");
    cache.set_block_table(&table).expect("block table");
    let mut at = 0usize;
    while at < prompt.len() {
        let n = chunk.min(prompt.len() - at);
        let tokens = Tensor::from_vec(prompt[at..at + n].to_vec(), (1usize, n), device).unwrap();
        let pos: Vec<i32> = (at as i32..(at + n) as i32).collect();
        let pos = Tensor::from_vec(pos, n, device).unwrap();
        model
            .forward_with_cache_last(&tokens, &pos, &mut cache)
            .expect("prefill chunk");
        at += n;
    }
    let len = prompt.len();
    let candle_dev = match device {
        Device::Cuda(d) => d.clone(),
        _ => unreachable!(),
    };
    let stream = nv_layers::cuda_stream::current_stream(&candle_dev);
    let t32: Vec<i32> = table.iter().map(|b| *b as i32).collect();
    let table_dev = stream.memcpy_stod(&t32).expect("table htod");
    let mut out = Vec::new();
    for &l in layers {
        let (k, _v) = pool
            .lock()
            .unwrap()
            .gather_layer(l, len, &table_dev)
            .expect("gather");
        out.push(rows(&k));
    }
    out
}

const PREFILL_CHUNK_ROWS_BELOW_WHICH_THE_MODEL_SWITCHES_PRECISION_256: usize = 256;

fn smallest_chunk(len: usize, chunk: usize) -> usize {
    let tail = len % chunk;
    if tail == 0 { chunk.min(len) } else { chunk.min(tail) }
}

#[test]
#[ignore]
fn paged_prefill_chunk_divergence_by_layer_depth() {
    if std::env::var("NV_PAGED_RING_AB").as_deref() != Ok("1") {
        panic!("PRECONDITION NOT MET, THIS TEST EXECUTED NOTHING: set NV_PAGED_RING_AB=1");
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

    let n_layers = model.config().num_hidden_layers;
    let probe: Vec<usize> = vec![0, 1, 2, n_layers / 4, n_layers / 2, n_layers - 1];
    let window = model.config().sliding_window;
    let envn = |k: &str, d: usize| {
        std::env::var(k)
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(d)
    };
    let want = envn("NV_KVDEPTH_TOKENS", window + window / 2);
    let (chunk_a, chunk_b) = (envn("NV_KVDEPTH_CHUNK_A", 1024), envn("NV_KVDEPTH_CHUNK_B", 256));
    let prompt = build_prompt(&tok, want);
    let ctx = prompt.len() + BLOCK_SIZE;
    let pool_cfg =
        PagedPoolConfig::from_gemma4(model.config(), ctx.div_ceil(BLOCK_SIZE), BLOCK_SIZE);

    let big = prefill_and_hash_kv(&model, &device, &pool_cfg, &prompt, chunk_a, &probe);
    let small = prefill_and_hash_kv(&model, &device, &pool_cfg, &prompt, chunk_b, &probe);

    eprintln!(
        "[kv-depth] {} prompt tokens, {} layers, chunk {chunk_a} vs {chunk_b} \
         (tails {} and {})",
        prompt.len(),
        n_layers,
        prompt.len() % chunk_a,
        prompt.len() % chunk_b
    );
    let mut layer0 = 0.0f32;
    for (i, &l) in probe.iter().enumerate() {
        let worst = big[i]
            .iter()
            .zip(&small[i])
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        if l == 0 {
            layer0 = worst;
        }
        eprintln!("[kv-depth] layer {l:>3}  worst |K_{chunk_a} - K_{chunk_b}| = {worst:e}");
    }
    let floor = PREFILL_CHUNK_ROWS_BELOW_WHICH_THE_MODEL_SWITCHES_PRECISION_256;
    let (min_a, min_b) = (
        smallest_chunk(prompt.len(), chunk_a),
        smallest_chunk(prompt.len(), chunk_b),
    );
    eprintln!("[kv-depth] smallest chunk: {min_a} (arm A) and {min_b} (arm B), floor {floor}");
    if min_a >= floor && min_b >= floor {
        assert_eq!(
            layer0, 0.0,
            "every chunk in both schemes has at least {floor} rows (smallest {min_a} and \
             {min_b}), so both prefills take the same path through the projection GEMM and \
             must agree BIT FOR BIT. Measured {layer0:e} at layer 0. The chunks need NOT be \
             multiples of {floor}: 2048 tokens at chunk 448 is 448,448,448,448,256 and comes \
             back 0e0 against chunk 1024"
        );
    } else {
        assert!(
            layer0 > 0.0,
            "the smaller of the two schemes hands out a chunk of {} rows, under the {floor} \
             floor, so the two are expected to DIFFER. Measuring 0e0 means the floor moved and \
             this rule needs re-deriving, not that things got better. The floor was pinned to \
             the token at 2048 prompt tokens: chunks 1792,256 agree to 0e0 while 1793,255 \
             differ by 1.19e-1 on layer 0 K",
            min_a.min(min_b)
        );
    }
}

fn nonpaged_layer0_k(
    model: &Gemma4,
    device: &Device,
    prompt: &[u32],
    chunk: usize,
) -> Vec<f32> {
    use nv_models::gemma4::Gemma4Cache;
    let mut cache = model
        .new_kv_cache(prompt.len() + BLOCK_SIZE)
        .expect("kv cache");
    let mut at = 0usize;
    while at < prompt.len() {
        let n = chunk.min(prompt.len() - at);
        let tokens = Tensor::from_vec(prompt[at..at + n].to_vec(), (1usize, n), device).unwrap();
        let pos: Vec<i32> = (at as i32..(at + n) as i32).collect();
        let pos = Tensor::from_vec(pos, n, device).unwrap();
        model
            .forward_with_cache(&tokens, &pos, &mut cache)
            .expect("nonpaged prefill chunk");
        at += n;
    }
    let (k, _v) = cache.view(0, prompt.len()).expect("view layer 0");
    rows(&k)
}

#[test]
#[ignore]
fn how_much_of_the_layer0_chunk_divergence_survives_without_fp8_kv() {
    if std::env::var("NV_PAGED_RING_AB").as_deref() != Ok("1") {
        panic!("PRECONDITION NOT MET, THIS TEST EXECUTED NOTHING: set NV_PAGED_RING_AB=1");
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

    unsafe { std::env::set_var("NV_KV_NO_SLIDING", "1") };
    let window = model.config().sliding_window;
    let prompt = build_prompt(&tok, window + window / 2);
    let big = nonpaged_layer0_k(&model, &device, &prompt, 1024);
    let small = nonpaged_layer0_k(&model, &device, &prompt, 256);
    unsafe { std::env::remove_var("NV_KV_NO_SLIDING") };

    let worst = big
        .iter()
        .zip(&small)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    let mag = big.iter().map(|a| a.abs()).fold(0.0f32, f32::max);
    eprintln!(
        "[bf16-kv] {} prompt tokens, layer 0 K in bf16, chunk 1024 vs 256\n\
         [bf16-kv] worst |K_1024 - K_256| = {worst:e}, largest |K| = {mag:e}\n\
         [bf16-kv] the same comparison with fp8 KV measured 7.03e-2",
        prompt.len()
    );
    assert!(
        worst.is_finite() && mag > 0.0,
        "layer 0's K read back empty or non-finite, so this comparison measured nothing"
    );
}

const E4M3_MAX: f32 = 448.0;
const E4M3_MANTISSA_BITS: i32 = 3;
const E4M3_MIN_EXP: i32 = -6;

fn e4m3_round(v: f32) -> f32 {
    if v == 0.0 || !v.is_finite() {
        return v;
    }
    let e = v.abs().log2().floor().max(E4M3_MIN_EXP as f32) as i32;
    let step = (2.0f32).powi(e - E4M3_MANTISSA_BITS);
    ((v / step).round() * step).clamp(-E4M3_MAX, E4M3_MAX)
}

fn fp8_roundtrip(vals: &[f32], block: usize) -> Vec<f32> {
    let mut out = Vec::with_capacity(vals.len());
    for chunk in vals.chunks(block) {
        let amax = chunk.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
        let scale = if amax > 0.0 { amax / E4M3_MAX } else { 1.0 };
        for v in chunk {
            out.push(e4m3_round(v / scale) * scale);
        }
    }
    out
}

#[test]
#[ignore]
fn fp8_scale_granularity_versus_the_chunk_divergence_it_amplifies() {
    if std::env::var("NV_PAGED_RING_AB").as_deref() != Ok("1") {
        panic!("PRECONDITION NOT MET, THIS TEST EXECUTED NOTHING: set NV_PAGED_RING_AB=1");
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

    let head_dim = model.config().head_dim;
    unsafe { std::env::set_var("NV_KV_NO_SLIDING", "1") };
    let window = model.config().sliding_window;
    let prompt = build_prompt(&tok, window + window / 2);
    let big = nonpaged_layer0_k(&model, &device, &prompt, 1024);
    let small = nonpaged_layer0_k(&model, &device, &prompt, 256);
    unsafe { std::env::remove_var("NV_KV_NO_SLIDING") };

    let worst = |a: &[f32], b: &[f32]| {
        a.iter()
            .zip(b)
            .map(|(x, y)| (x - y).abs())
            .fold(0.0f32, f32::max)
    };
    let base = worst(&big, &small);
    eprintln!(
        "[granularity] {} prompt tokens, layer 0 K, chunk 1024 vs 256, head_dim {head_dim}\n\
         [granularity] bf16, no quantisation      worst {base:e}   (1.00x)",
        prompt.len()
    );
    for block in [head_dim, 128, 64, 32, 16] {
        let qb = fp8_roundtrip(&big, block);
        let qs = fp8_roundtrip(&small, block);
        let w = worst(&qb, &qs);
        let amp = w / base;
        let tag = if block == head_dim {
            " <- what we ship"
        } else if block == 16 {
            " <- NVFP4's granularity"
        } else {
            ""
        };
        eprintln!("[granularity] fp8, one scale per {block:>4}   worst {w:e}   ({amp:.2}x){tag}");
    }
    assert!(
        base > 0.0,
        "the two chunkings produced identical bf16 K, so there is no divergence to amplify \
         and this measured nothing"
    );
}

const E2M1_LEVELS: [f32; 8] = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0];
const E2M1_MAX: f32 = 6.0;
const NVFP4_BLOCK: usize = 16;

fn e2m1_round(v: f32) -> f32 {
    let a = v.abs().min(E2M1_MAX);
    let mut best = E2M1_LEVELS[0];
    let mut bd = f32::INFINITY;
    for l in E2M1_LEVELS {
        let d = (a - l).abs();
        if d < bd {
            bd = d;
            best = l;
        }
    }
    if v < 0.0 {
        -best
    } else {
        best
    }
}

fn nvfp4_roundtrip(vals: &[f32]) -> Vec<f32> {
    let amax = vals.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
    if amax == 0.0 {
        return vals.to_vec();
    }
    let s_tensor = amax / (E2M1_MAX * E4M3_MAX);
    let mut out = Vec::with_capacity(vals.len());
    for chunk in vals.chunks(NVFP4_BLOCK) {
        let bmax = chunk.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
        let want = if bmax > 0.0 {
            bmax / (E2M1_MAX * s_tensor)
        } else {
            1.0
        };
        let s_block = e4m3_round(want).max(f32::MIN_POSITIVE);
        let step = s_tensor * s_block;
        for v in chunk {
            out.push(e2m1_round(v / step) * step);
        }
    }
    out
}

fn rms(a: &[f32], b: &[f32]) -> f32 {
    let n: f64 = a
        .iter()
        .zip(b)
        .map(|(x, y)| ((x - y) as f64).powi(2))
        .sum::<f64>()
        / a.len() as f64;
    n.sqrt() as f32
}

#[test]
#[ignore]
fn nvfp4_versus_shipped_fp8_on_accuracy_and_on_chunk_amplification() {
    if std::env::var("NV_PAGED_RING_AB").as_deref() != Ok("1") {
        panic!("PRECONDITION NOT MET, THIS TEST EXECUTED NOTHING: set NV_PAGED_RING_AB=1");
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

    let head_dim = model.config().head_dim;
    unsafe { std::env::set_var("NV_KV_NO_SLIDING", "1") };
    let window = model.config().sliding_window;
    let prompt = build_prompt(&tok, window + window / 2);
    let big = nonpaged_layer0_k(&model, &device, &prompt, 1024);
    let small = nonpaged_layer0_k(&model, &device, &prompt, 256);
    unsafe { std::env::remove_var("NV_KV_NO_SLIDING") };

    let worst = |a: &[f32], b: &[f32]| {
        a.iter()
            .zip(b)
            .map(|(x, y)| (x - y).abs())
            .fold(0.0f32, f32::max)
    };
    let base = worst(&big, &small);

    let fp8_big = fp8_roundtrip(&big, head_dim);
    let fp8_small = fp8_roundtrip(&small, head_dim);
    let nv_big = nvfp4_roundtrip(&big);
    let nv_small = nvfp4_roundtrip(&small);

    let mag = big.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
    eprintln!(
        "[nvfp4] {} prompt tokens, layer 0 K, head_dim {head_dim}, largest |K| {mag:e}\n\
         [nvfp4]                          abs-err(max)  abs-err(rms)   chunk-amp\n\
         [nvfp4]  fp8 e4m3, scale/{head_dim:<4}   {:e}  {:e}   {:.2}x\n\
         [nvfp4]  NVFP4 e2m1, scale/16     {:e}  {:e}   {:.2}x",
        prompt.len(),
        worst(&big, &fp8_big),
        rms(&big, &fp8_big),
        worst(&fp8_big, &fp8_small) / base,
        worst(&big, &nv_big),
        rms(&big, &nv_big),
        worst(&nv_big, &nv_small) / base,
    );
    assert!(
        (worst(&fp8_big, &fp8_small) - 7.031_25e-2).abs() < 1e-4,
        "the fp8 emulation no longer reproduces the shipped quantize_kv_fp8_paged kernel, \
         which measured 7.03125e-2 on this same tensor pair. Every NVFP4 figure above is \
         only as good as that agreement"
    );
}

fn hadamard(a: &mut [f32]) {
    let n = a.len();
    assert!(n.is_power_of_two(), "Hadamard needs a power-of-two length, got {n}");
    let mut h = 1usize;
    while h < n {
        let mut i = 0usize;
        while i < n {
            for j in i..i + h {
                let (x, y) = (a[j], a[j + h]);
                a[j] = x + y;
                a[j + h] = x - y;
            }
            i += h * 2;
        }
        h *= 2;
    }
    let inv = 1.0 / (n as f32).sqrt();
    for v in a.iter_mut() {
        *v *= inv;
    }
}

fn rotate_rows(vals: &[f32], row: usize) -> Vec<f32> {
    let mut out = vals.to_vec();
    for chunk in out.chunks_mut(row) {
        hadamard(chunk);
    }
    out
}

fn fp8_roundtrip_e4m3scale(vals: &[f32], block: usize) -> Vec<f32> {
    let amax_t = vals.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
    if amax_t == 0.0 {
        return vals.to_vec();
    }
    let s_tensor = amax_t / (E4M3_MAX * E4M3_MAX);
    let mut out = Vec::with_capacity(vals.len());
    for chunk in vals.chunks(block) {
        let amax = chunk.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
        let want = if amax > 0.0 {
            amax / (E4M3_MAX * s_tensor)
        } else {
            1.0
        };
        let s_block = e4m3_round(want).max(f32::MIN_POSITIVE);
        let step = s_tensor * s_block;
        for v in chunk {
            out.push(e4m3_round(v / step) * step);
        }
    }
    out
}

#[test]
#[ignore]
fn which_kv_format_is_best_per_byte() {
    if std::env::var("NV_PAGED_RING_AB").as_deref() != Ok("1") {
        panic!("PRECONDITION NOT MET, THIS TEST EXECUTED NOTHING: set NV_PAGED_RING_AB=1");
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

    let hd = model.config().head_dim;
    unsafe { std::env::set_var("NV_KV_NO_SLIDING", "1") };
    let window = model.config().sliding_window;
    let prompt = build_prompt(&tok, window + window / 2);
    let big = nonpaged_layer0_k(&model, &device, &prompt, 1024);
    let small = nonpaged_layer0_k(&model, &device, &prompt, 256);
    unsafe { std::env::remove_var("NV_KV_NO_SLIDING") };

    let worst = |a: &[f32], b: &[f32]| {
        a.iter()
            .zip(b)
            .map(|(x, y)| (x - y).abs())
            .fold(0.0f32, f32::max)
    };
    let base = worst(&big, &small);

    let f32_scale = |block: usize| 1.0 + 4.0 / block as f64;
    let e4m3_scale = |payload: f64, block: usize| payload + 1.0 / block as f64;

    type Rt = Box<dyn Fn(&[f32]) -> Vec<f32>>;
    let cases: Vec<(&str, f64, Rt)> = vec![
        ("bf16 (no quantisation)      ", 2.0, Box::new(|v: &[f32]| v.to_vec())),
        (
            "fp8 e4m3 scale/256  SHIPPED ",
            f32_scale(256),
            Box::new(move |v: &[f32]| fp8_roundtrip(v, hd)),
        ),
        (
            "fp8 e4m3 scale/16           ",
            e4m3_scale(1.0, 16),
            Box::new(|v: &[f32]| fp8_roundtrip_e4m3scale(v, 16)),
        ),
        (
            "NVFP4 e2m1 scale/16         ",
            e4m3_scale(0.5, 16),
            Box::new(|v: &[f32]| nvfp4_roundtrip(v)),
        ),
        (
            "fp8 e4m3 scale/256 + rotate ",
            f32_scale(256),
            Box::new(move |v: &[f32]| {
                rotate_rows(&fp8_roundtrip(&rotate_rows(v, hd), hd), hd)
            }),
        ),
        (
            "fp8 e4m3 scale/16  + rotate ",
            e4m3_scale(1.0, 16),
            Box::new(move |v: &[f32]| {
                rotate_rows(&fp8_roundtrip_e4m3scale(&rotate_rows(v, hd), 16), hd)
            }),
        ),
        (
            "NVFP4 e2m1 scale/16 + rotate",
            e4m3_scale(0.5, 16),
            Box::new(move |v: &[f32]| rotate_rows(&nvfp4_roundtrip(&rotate_rows(v, hd)), hd)),
        ),
    ];

    eprintln!(
        "[kv-format] {} prompt tokens, layer 0 K, head_dim {hd}, largest |K| {:e}\n\
         [kv-format]                                B/elem  abs-err(max)  abs-err(rms)  chunk-amp",
        prompt.len(),
        big.iter().map(|v| v.abs()).fold(0.0f32, f32::max)
    );
    let mut shipped_rms = 0.0f32;
    for (tag, bytes, f) in &cases {
        let (qb, qs) = (f(&big), f(&small));
        let r = rms(&big, &qb);
        if tag.contains("SHIPPED") {
            shipped_rms = r;
        }
        eprintln!(
            "[kv-format]  {tag}  {bytes:.4}  {:e}  {r:e}  {:.2}x",
            worst(&big, &qb),
            worst(&qb, &qs) / base
        );
    }
    eprintln!("[kv-format]  (shipped rms {shipped_rms:e} is the bar every row is judged against)");
    assert!(
        shipped_rms > 0.0,
        "the shipped row round-tripped exactly, so there is no bar to compare against"
    );
}

#[test]
#[ignore]
fn the_hadamard_rotation_does_not_cost_perplexity_end_to_end() {
    if std::env::var("NV_PAGED_RING_AB").as_deref() != Ok("1") {
        panic!("PRECONDITION NOT MET, THIS TEST EXECUTED NOTHING: set NV_PAGED_RING_AB=1");
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

    let window = model.config().sliding_window;
    let aligned_len = 2048usize;
    assert!(
        smallest_chunk(aligned_len, NONPAGED_MAX_CHUNK_PAST_WINDOW_IS_SLIDING_COMPACT_SLACK_256)
            >= PREFILL_CHUNK_ROWS_BELOW_WHICH_THE_MODEL_SWITCHES_PRECISION_256,
        "this gate compares chunk sizes, so every chunk must clear the \
         {PREFILL_CHUNK_ROWS_BELOW_WHICH_THE_MODEL_SWITCHES_PRECISION_256}-row floor \
         or the arms differ for the reason task 56 found rather than for the rotation"
    );
    let prompt = build_prompt(&tok, aligned_len - 1);
    let ctx = prompt.len() + BLOCK_SIZE;
    let hybrid =
        PagedPoolConfig::from_gemma4_hybrid(model.config(), ctx.div_ceil(BLOCK_SIZE), BLOCK_SIZE, 1);
    let chunk = NONPAGED_MAX_CHUNK_PAST_WINDOW_IS_SLIDING_COMPACT_SLACK_256;

    let reference = teacher_forced_ppl_nonpaged(&model, &device, &prompt, chunk);
    eprintln!(
        "[rot-ppl] {} prompt tokens, scoring the last {PPL_EVAL_TOKENS}\n\
         [rot-ppl] non-paged reference at chunk {chunk}   NLL {reference:.6}",
        prompt.len()
    );
    let mut deltas = Vec::new();
    let mut dense_delta = 0.0f64;
    for (arm, tag, c) in [
        (Arm::Dense, "bf16 KV  ", chunk),
        (Arm::RingKernel, "fp8 KV   ", chunk),
        (Arm::RingKernel, "fp8 KV   ", 1024usize),
    ] {
        std::env::remove_var(nv_models::hadamard_kv::HADAMARD_KV_ENV);
        let (off, _) = teacher_forced_ppl(&model, &device, &hybrid, &prompt, arm, c);
        std::env::set_var(nv_models::hadamard_kv::HADAMARD_KV_ENV, "1");
        let (on, _) = teacher_forced_ppl(&model, &device, &hybrid, &prompt, arm, c);
        std::env::remove_var(nv_models::hadamard_kv::HADAMARD_KV_ENV);
        eprintln!(
            "[rot-ppl] {tag} chunk {c:>4}  OFF {off:.6}  ON {on:.6}  ON-OFF {:+.6}",
            on - off
        );
        if arm == Arm::Dense {
            dense_delta = on - off;
        } else {
            deltas.push(on - off);
        }
    }
    assert!(
        dense_delta.abs() < 1e-3,
        "with bf16 KV and NO quantisation anywhere, rotating moved perplexity by {dense_delta:+.6}. \
         An orthogonal H leaves Q.K^T exactly alone, so with nothing quantised the two arms must \
         agree: this is structural -- a path rotating one of the pair without the other, or V \
         relating to K differently than the model expects (attention_k_eq_v is true here) -- \
         and NOT the fp8 amax interaction"
    );
    std::env::remove_var(PAGED_ATTN_FP8_RING_ENV);
    assert!(
        deltas.iter().all(|d| d.is_finite()),
        "a rotated arm produced a non-finite NLL, so the rotation is destroying the cache \
         rather than changing its basis"
    );
    assert!(
        deltas.iter().all(|d| *d <= 0.0),
        "rotating Q and K COSTS perplexity: ON-OFF is {:+.6} at chunk {chunk} and {:+.6} at \
         1024, identical because task 56's alignment rule now holds, so this is the rotation \
         and not the harness. It contradicts the simulation, which measured K round-trip peak \
         error 4.69e-2 -> 1.70e-2 at the same bytes, and the unit gates, which hold Q.K exact \
         to 1e-4 on device. Something the simulation does not model is being changed by \
         rotating in the live path -- suspects in order: a site rotating Q without K at \
         runtime that the textual census cannot see, the fp8 per-(token, kv_head) amax being \
         taken over a rotated row whose outlier structure the kernel relies on, and V staying \
         unrotated while attention_k_eq_v ties it to K",
        deltas[0],
        deltas[1]
    );
    assert!(
        deltas[0].signum() == deltas[1].signum(),
        "rotation moves perplexity in OPPOSITE directions at chunk {chunk} ({:+.6}) and 1024 \
         ({:+.6}). A basis change cannot depend on how the prefill was chunked, so this is the \
         task-56 confound -- the paged and non-paged families disagree per chunk size -- and \
         neither number ranks the rotation. Judge it on K round-trip error, not on this",
        deltas[0],
        deltas[1]
    );
}

#[test]
#[ignore]
fn every_chunking_above_the_floor_gives_the_same_kv() {
    if std::env::var("NV_PAGED_RING_AB").as_deref() != Ok("1") {
        panic!("PRECONDITION NOT MET, THIS TEST EXECUTED NOTHING: set NV_PAGED_RING_AB=1");
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

    let n_layers = model.config().num_hidden_layers;
    let probe: Vec<usize> = vec![0, n_layers / 2, n_layers - 1];
    let prompt = build_prompt(&tok, 2048 - 1);

    let ctx = prompt.len() + BLOCK_SIZE;
    let pool_cfg =
        PagedPoolConfig::from_gemma4(model.config(), ctx.div_ceil(BLOCK_SIZE), BLOCK_SIZE);

    let chunks: Vec<usize> = match std::env::var("NV_ALIGN_CHUNKS") {
        Ok(v) => v.split(',').map(|c| c.trim().parse().expect("chunk list")).collect(),
        Err(_) => vec![1024usize, 700, 448, 256],
    };
    assert!(chunks.len() >= 2, "need at least two chunkings to compare");
    let base = prefill_and_hash_kv(&model, &device, &pool_cfg, &prompt, chunks[0], &probe);
    eprintln!(
        "[align-general] {} prompt tokens, chunkings {chunks:?} against {}",
        prompt.len(),
        chunks[0]
    );
    for &c in &chunks[1..] {
        let got = prefill_and_hash_kv(&model, &device, &pool_cfg, &prompt, c, &probe);
        let worst = probe
            .iter()
            .enumerate()
            .map(|(i, _)| {
                base[i]
                    .iter()
                    .zip(&got[i])
                    .map(|(a, b)| (a - b).abs())
                    .fold(0.0f32, f32::max)
            })
            .fold(0.0f32, f32::max);
        let tail = prompt.len() % c;
        eprintln!("[align-general] chunk {c:>4} (tail {tail:>4})  worst |dK| = {worst:e}");
        let floor = PREFILL_CHUNK_ROWS_BELOW_WHICH_THE_MODEL_SWITCHES_PRECISION_256;
        assert!(
            smallest_chunk(prompt.len(), c) >= floor,
            "chunk {c} on a {}-token prompt has a smallest chunk of {}, under the {floor} \
             floor, so this arm cannot test the rule and the sweep is checking nothing",
            prompt.len(),
            smallest_chunk(prompt.len(), c)
        );
        assert_eq!(
            worst, 0.0,
            "chunk {c} disagrees with chunk {} by {worst:e}. Every chunk in both schemes is at \
             least {floor} rows, which is the rule NV_PREFILL_CHUNK_MIN relies on -- any two \
             chunkings whose smallest chunk clears the floor compute the same KV -- so if this \
             fires, flooring the scheduler does not make a prompt independent of its \
             neighbours after all",
            chunks[0]
        );
    }
}

#[test]
#[ignore]
fn w4a4_prefill_costs_this_much_perplexity() {
    if std::env::var("NV_PAGED_RING_AB").as_deref() != Ok("1") {
        panic!("PRECONDITION NOT MET, THIS TEST EXECUTED NOTHING: set NV_PAGED_RING_AB=1");
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

    let floor = PREFILL_CHUNK_ROWS_BELOW_WHICH_THE_MODEL_SWITCHES_PRECISION_256;
    let prompt = build_prompt(&tok, W4A4_PPL_PROMPT_FITS_THE_NONPAGED_SLIDING_SLACK_1024 - 1);
    let scored_from = prompt.len() - PPL_EVAL_TOKENS;
    assert!(
        scored_from % 256 == 0 && scored_from % 192 == 0 && scored_from % 64 == 0,
        "the prefilled span is {scored_from} tokens and every chunking below must divide it \
         evenly, or a ragged tail lands under the switch and contaminates the engaged group"
    );

    let engaged = [768usize, 384, 256];
    let skipped = [192usize, 128, 64];
    for c in engaged {
        assert!(
            smallest_chunk(prompt.len(), c) >= floor,
            "chunk {c} was put in the W4A4-engaged group but its smallest chunk is {}, under \
             the {floor}-row switch, so the two groups are not what this gate says they are",
            smallest_chunk(prompt.len(), c)
        );
    }
    for c in skipped {
        assert!(
            smallest_chunk(prompt.len(), c) < floor,
            "chunk {c} was put in the W4A4-skipped group but every chunk clears the {floor}-row \
             switch"
        );
    }

    let measure = |c: usize| teacher_forced_ppl_nonpaged(&model, &device, &prompt, c);
    assert!(
        *skipped.iter().min().unwrap() > FUSED_QKV_BITWISE_SAFE_MAX_M_16,
        "a chunk of {} rows or fewer takes the fused_qkv_bitwise_safe branch, which is a \
         second m-dependent path, and the skipped group would then be measuring two switches \
         at once",
        FUSED_QKV_BITWISE_SAFE_MAX_M_16
    );
    let on: Vec<f64> = engaged.iter().map(|c| measure(*c)).collect();
    let off: Vec<f64> = skipped.iter().map(|c| measure(*c)).collect();
    let decode_ref = measure(1);

    eprintln!(
        "[w4a4-ppl] {} prompt tokens, {PPL_EVAL_TOKENS} scored, switch at m >= {floor}",
        prompt.len()
    );
    for (c, v) in engaged.iter().zip(&on) {
        eprintln!("[w4a4-ppl]  chunk {c:>4}  W4A4 ENGAGED  ppl {v:.6}");
    }
    for (c, v) in skipped.iter().zip(&off) {
        eprintln!("[w4a4-ppl]  chunk {c:>4}  W4A4 skipped  ppl {v:.6}");
    }

    let spread = |v: &[f64]| {
        v.iter().cloned().fold(f64::MIN, f64::max) - v.iter().cloned().fold(f64::MAX, f64::min)
    };
    let (spread_on, spread_off) = (spread(&on), spread(&off));
    let gap = on[0] - off[0];
    eprintln!(
        "[w4a4-ppl]  chunk    1  decode path   ppl {decode_ref:.6}  <-- reference: one token at \
         a time is what generation itself does"
    );
    eprintln!(
        "[w4a4-ppl]  within-group spread {spread_on:.6} (engaged) {spread_off:.6} (skipped), \
         across-group gap {gap:+.6}, engaged vs decode path {:+.6}",
        on[0] - decode_ref
    );

    if nv_models::gemma4::prefill_w4a4_env(std::env::var("NV_PREFILL_W4A4").ok().as_deref()) {
        assert_eq!(
            spread_on, 0.0,
            "the W4A4-engaged chunkings disagree among themselves by {spread_on:e}. Every chunk \
             here clears the {floor}-row switch, so they take one path and must score \
             identically; if they do not, chunk size is moving quality on its own and no gap \
             below can be attributed to the switch"
        );
        assert!(
            spread_off < gap.abs(),
            "the skipped group's own spread ({spread_off:e}) is as large as the gap to the \
             engaged group ({gap:+e}), so this gate cannot separate the switch from chunk size"
        );
    } else {
        assert!(
            spread_on > 0.0,
            "with NV_PREFILL_W4A4 off the m >= {floor} chunkings scored identically, which \
             would mean the ordinary QKV projection is chunk-stable and W4A4 is not what makes \
             prefill reproducible. Measured 0.604982 / 0.836492 / 1.119688 at chunks 768 / 384 \
             / 256 when this was written -- a spread of 0.51 NLL from chunk size alone"
        );
    }
}

#[test]
#[ignore]
fn a_single_prefill_chunk_is_judged_against_the_decode_path_at_its_own_length() {
    if std::env::var("NV_PAGED_RING_AB").as_deref() != Ok("1") {
        panic!("PRECONDITION NOT MET, THIS TEST EXECUTED NOTHING: set NV_PAGED_RING_AB=1");
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

    let floor = PREFILL_CHUNK_ROWS_BELOW_WHICH_THE_MODEL_SWITCHES_PRECISION_256;
    let spans = [64usize, 128, 192, 256, 320, 384, 768];

    eprintln!(
        "[span-ppl] each row prefills its whole span as ONE chunk and is judged against the \
         decode path over the SAME span, so context length is held fixed inside each pair and \
         only the deltas are compared across rows. switch at m >= {floor}"
    );
    let mut rows: Vec<(usize, f64, f64, f64)> = Vec::new();
    for span in spans {
        let prompt = build_prompt(&tok, span + PPL_EVAL_TOKENS - 1);
        assert_eq!(
            prompt.len() - PPL_EVAL_TOKENS,
            span,
            "the harness must prefill exactly {span} tokens for this row to mean what it says"
        );
        let one_chunk = teacher_forced_ppl_nonpaged(&model, &device, &prompt, span);
        let decode = teacher_forced_ppl_nonpaged(&model, &device, &prompt, 1);
        let delta = one_chunk - decode;
        eprintln!(
            "[span-ppl]  span {span:>4}  {}  one-chunk {one_chunk:.6}  decode {decode:.6}  \
             delta {delta:+.6}",
            if span >= floor { "ENGAGED" } else { "skipped" }
        );
        rows.push((span, one_chunk, decode, delta));
    }

    let distinct: std::collections::BTreeSet<u64> =
        rows.iter().map(|r| r.2.to_bits()).collect();
    assert_eq!(
        distinct.len(),
        rows.len(),
        "two spans produced the same decode reference to the bit, so the spans are not really \
         different contexts and the deltas below are not comparable"
    );
    let worst_engaged = rows
        .iter()
        .filter(|r| r.0 >= floor)
        .map(|r| r.3.abs())
        .fold(0.0f64, f64::max);
    let worst_skipped = rows
        .iter()
        .filter(|r| r.0 < floor)
        .map(|r| r.3.abs())
        .fold(0.0f64, f64::max);
    eprintln!(
        "[span-ppl]  worst |delta| {worst_engaged:.6} engaged, {worst_skipped:.6} skipped"
    );
    assert!(
        worst_engaged.is_finite() && worst_skipped.is_finite(),
        "a delta came back non-finite, so at least one arm did not actually score anything"
    );
    assert!(
        worst_skipped < SPLITTING_SUB_FLOOR_COSTS_0_34_NLL_SO_ONE_CHUNK_IS_BOUNDED_AT_0_30,
        "a single prefill chunk under the {floor}-row switch sits {worst_skipped:.6} from the \
         decode path, at or past the {SPLITTING_SUB_FLOOR_COSTS_0_34_NLL_SO_ONE_CHUNK_IS_BOUNDED_AT_0_30} that SPLITTING a \
         prefill into sub-floor chunks costs (w4a4_prefill_costs_this_much_perplexity: 0.88 to \
         0.96 against a 0.581 decode reference). Those are different things and this gate exists \
         to keep them apart: one chunk of any size is fine, many small chunks are not. Measured \
         -0.075425 / -0.102608 / -0.030016 at spans 64 / 128 / 192, so a short prompt does not \
         suffer for being unable to reach the switch. Cross-row deltas are noisy -- each span \
         is a different prompt and the decode reference itself ranges 0.58 to 2.83 -- so read \
         the within-pair deltas, not the trend"
    );
}
