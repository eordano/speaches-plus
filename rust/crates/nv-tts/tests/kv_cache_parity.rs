use std::collections::HashMap;

use candle_core::{DType, Device, Tensor};
use nv_tts::talker::{Qwen3TtsTalker, Qwen3TtsTalkerConfig};
use nv_weights::WeightLoader;

const HIDDEN: usize = 32;
const LAYERS: usize = 2;
const HEADS: usize = 4;
const KV_HEADS: usize = 2;
const HEAD_DIM: usize = 8;
const INTER: usize = 64;
const SPEECH_VOCAB: usize = 16;
const TEXT_HIDDEN: usize = 16;
const TEXT_TOKENS: usize = 3;
const STEPS: usize = 5;

const HIDDEN_TOL: f32 = 1e-4;

const HISTORY_SENSITIVITY_FLOOR: f32 = 1e-3;

fn cfg() -> Qwen3TtsTalkerConfig {
    Qwen3TtsTalkerConfig {
        hidden_size: HIDDEN,
        num_hidden_layers: LAYERS,
        num_attention_heads: HEADS,
        num_key_value_heads: KV_HEADS,
        head_dim: HEAD_DIM,
        intermediate_size: INTER,
        speech_vocab_size: SPEECH_VOCAB,
        text_vocab_size: 64,
        text_hidden_size: TEXT_HIDDEN,
        rope_theta: 10_000.0,
        max_position_embeddings: 64,
        rms_norm_eps: 1e-6,
        mrope_section: vec![2, 1, 1],
        dtype: DType::F32,
        spk_id: Vec::new(),
        language_id: Vec::new(),
    }
}

struct Lcg(u64);

impl Lcg {
    fn unit(&mut self) -> f32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (self.0 >> 33) as u32 as f32 / u32::MAX as f32 - 0.5
    }

    fn tensor(&mut self, shape: &[usize], scale: f32, dev: &Device) -> Tensor {
        let n: usize = shape.iter().product();
        let v: Vec<f32> = (0..n).map(|_| self.unit() * scale).collect();
        Tensor::from_vec(v, shape, dev).unwrap()
    }

    fn norm_tensor(&mut self, n: usize, dev: &Device) -> Tensor {
        let v: Vec<f32> = (0..n).map(|_| 1.0 + self.unit() * 0.2).collect();
        Tensor::from_vec(v, n, dev).unwrap()
    }
}

const EMBED_SCALE: f32 = 2.0;

const ATTN_SCALE: f32 = 0.6;

const MLP_SCALE: f32 = 0.1;

fn dense_weight_map(dev: &Device) -> HashMap<String, Tensor> {
    let mut r = Lcg(0x9e37_79b9_7f4a_7c15);
    let mut m: HashMap<String, Tensor> = HashMap::new();
    let h_q = HEADS * HEAD_DIM;
    let h_kv = KV_HEADS * HEAD_DIM;

    m.insert(
        "talker.text_projection.linear_fc1.weight".into(),
        r.tensor(&[TEXT_HIDDEN, TEXT_HIDDEN], ATTN_SCALE, dev),
    );
    m.insert(
        "talker.text_projection.linear_fc1.bias".into(),
        r.tensor(&[TEXT_HIDDEN], ATTN_SCALE, dev),
    );
    m.insert(
        "talker.text_projection.linear_fc2.weight".into(),
        r.tensor(&[HIDDEN, TEXT_HIDDEN], ATTN_SCALE, dev),
    );
    m.insert(
        "talker.text_projection.linear_fc2.bias".into(),
        r.tensor(&[HIDDEN], ATTN_SCALE, dev),
    );
    m.insert(
        "talker.model.codec_embedding.weight".into(),
        r.tensor(&[SPEECH_VOCAB, HIDDEN], EMBED_SCALE, dev),
    );

    for i in 0..LAYERS {
        let p = format!("talker.model.layers.{i}");
        m.insert(
            format!("{p}.input_layernorm.weight"),
            r.norm_tensor(HIDDEN, dev),
        );
        m.insert(
            format!("{p}.post_attention_layernorm.weight"),
            r.norm_tensor(HIDDEN, dev),
        );
        m.insert(
            format!("{p}.self_attn.q_proj.weight"),
            r.tensor(&[h_q, HIDDEN], ATTN_SCALE, dev),
        );
        m.insert(
            format!("{p}.self_attn.k_proj.weight"),
            r.tensor(&[h_kv, HIDDEN], ATTN_SCALE, dev),
        );
        m.insert(
            format!("{p}.self_attn.v_proj.weight"),
            r.tensor(&[h_kv, HIDDEN], ATTN_SCALE, dev),
        );
        m.insert(
            format!("{p}.self_attn.o_proj.weight"),
            r.tensor(&[HIDDEN, h_q], ATTN_SCALE, dev),
        );
        m.insert(
            format!("{p}.self_attn.q_norm.weight"),
            r.norm_tensor(HEAD_DIM, dev),
        );
        m.insert(
            format!("{p}.self_attn.k_norm.weight"),
            r.norm_tensor(HEAD_DIM, dev),
        );
        m.insert(
            format!("{p}.mlp.gate_proj.weight"),
            r.tensor(&[INTER, HIDDEN], MLP_SCALE, dev),
        );
        m.insert(
            format!("{p}.mlp.up_proj.weight"),
            r.tensor(&[INTER, HIDDEN], MLP_SCALE, dev),
        );
        m.insert(
            format!("{p}.mlp.down_proj.weight"),
            r.tensor(&[HIDDEN, INTER], MLP_SCALE, dev),
        );
    }

    m.insert("talker.model.norm.weight".into(), r.norm_tensor(HIDDEN, dev));
    m.insert(
        "talker.codec_head.weight".into(),
        r.tensor(&[SPEECH_VOCAB, HIDDEN], EMBED_SCALE, dev),
    );
    m
}

fn dense_talker(tag: &str, dev: &Device) -> Qwen3TtsTalker {
    let dir = std::env::temp_dir().join(format!("nv_tts_kv_parity_{tag}_{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("model.safetensors");
    candle_core::safetensors::save(&dense_weight_map(dev), &path).unwrap();
    let loader = WeightLoader::open_file(&path, dev).expect("open synthetic checkpoint");
    let mut t = Qwen3TtsTalker::new(cfg(), dev).expect("build talker");
    t.load_weights(&loader).expect("load synthetic weights");
    std::fs::remove_dir_all(&dir).ok();
    t
}

fn text_hidden(dev: &Device) -> Tensor {
    let v: Vec<f32> = (0..TEXT_TOKENS * TEXT_HIDDEN)
        .map(|i| ((i % 11) as f32 - 5.0) * 0.07)
        .collect();
    Tensor::from_vec(v, (1usize, TEXT_TOKENS, TEXT_HIDDEN), dev).unwrap()
}

fn row(t: &Tensor) -> Vec<f32> {
    t.flatten_all().unwrap().to_vec1::<f32>().unwrap()
}

fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max)
}

fn spread(v: &[f32]) -> f32 {
    let lo = v.iter().cloned().fold(f32::INFINITY, f32::min);
    let hi = v.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    hi - lo
}

#[test]
fn attention_history_actually_moves_the_hidden_state() {
    let dev = Device::Cpu;
    let t = dense_talker("sensitivity", &dev);
    let th = text_hidden(&dev);

    let (_, h_none) = t.step_with_hidden(&th, &[]).expect("step []");
    let (_, h_one) = t.step_with_hidden(&th, &[3]).expect("step [3]");
    let (_, h_two) = t.step_with_hidden(&th, &[3, 7]).expect("step [3, 7]");
    let (_, h_two_alt) = t.step_with_hidden(&th, &[9, 7]).expect("step [9, 7]");

    let a = row(&h_none);
    let b = row(&h_one);
    let c = row(&h_two);
    let d = row(&h_two_alt);

    println!(
        "hidden spread={:.3e} d(none,one)={:.3e} d(one,two)={:.3e} d(two,two_alt)={:.3e} tol={:.1e}",
        spread(&a),
        max_abs_diff(&a, &b),
        max_abs_diff(&b, &c),
        max_abs_diff(&c, &d),
        HIDDEN_TOL
    );

    assert!(
        spread(&a) > HISTORY_SENSITIVITY_FLOOR,
        "hidden state is constant across channels ({:.3e}); this checkpoint is degenerate and \
         every parity assertion built on it is vacuous",
        spread(&a)
    );
    assert!(
        max_abs_diff(&a, &b) > HISTORY_SENSITIVITY_FLOOR,
        "appending a speech token did not move the hidden state; with a zero-initialised attention \
         stack the KV cache cannot affect the output and cached-vs-uncached parity gates nothing"
    );
    assert!(
        max_abs_diff(&b, &c) > HISTORY_SENSITIVITY_FLOOR,
        "extending the history from one token to two did not move the hidden state"
    );
    assert!(
        max_abs_diff(&c, &d) > HISTORY_SENSITIVITY_FLOOR,
        "changing an older token with the newest token held fixed did not move the hidden state; \
         attention is not reading history, so a broken KV cache would be invisible"
    );
}

#[test]
fn cached_and_uncached_hidden_states_agree_step_for_step() {
    let dev = Device::Cpu;
    let t = dense_talker("parity", &dev);
    let th = text_hidden(&dev);

    let mut cache = t.new_kv_cache(64).expect("kv cache");
    let mut history: Vec<u32> = Vec::new();
    let mut prev_cached: Option<Vec<f32>> = None;
    let mut worst_parity = 0.0f32;
    let mut widest_step = 0.0f32;

    for step in 0..STEPS {
        let new_token = history.last().copied();
        let (tok_cached, h_cached) = t
            .step_cached_with_hidden(&th, new_token, &mut cache)
            .expect("cached step");
        let (tok_ref, h_ref) = t.step_with_hidden(&th, &history).expect("uncached step");

        let c = row(&h_cached);
        let r = row(&h_ref);
        let delta = max_abs_diff(&c, &r);
        let motion = prev_cached
            .as_ref()
            .map(|p| max_abs_diff(p, &c))
            .unwrap_or(0.0);
        println!("step {step}: |cached - uncached| = {delta:.3e}, step motion = {motion:.3e}");
        assert!(
            delta <= HIDDEN_TOL,
            "step {step}: cached hidden state diverged from the uncached recompute by {delta:.3e} \
             (> {HIDDEN_TOL:.1e}); history = {history:?}"
        );
        assert_eq!(
            tok_cached, tok_ref,
            "step {step}: cached and uncached token ids disagree; history = {history:?}"
        );

        worst_parity = worst_parity.max(delta);
        widest_step = widest_step.max(motion);
        prev_cached = Some(c);
        history.push(tok_cached);
    }

    assert!(
        widest_step > HISTORY_SENSITIVITY_FLOOR,
        "the hidden state never moved across {STEPS} steps (widest step {widest_step:.3e}); a \
         comparison over a frozen trajectory cannot see a broken cache"
    );
    assert!(
        widest_step > 100.0 * worst_parity.max(f32::MIN_POSITIVE),
        "cached-vs-uncached error {worst_parity:.3e} is not small against the trajectory's own \
         motion {widest_step:.3e}; the agreement is not evidence of anything"
    );

    assert_eq!(
        cache.current_len(),
        TEXT_TOKENS + STEPS - 1,
        "cache length must be the projected text tokens plus one embedding per incremental step"
    );
}
