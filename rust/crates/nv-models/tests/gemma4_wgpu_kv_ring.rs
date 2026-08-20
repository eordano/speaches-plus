#![cfg(feature = "wgpu")]

mod common;
use common::ctx_or_skip_no_require as ctx_or_skip;
use nv_models::gemma4::{Gemma4Config, LayerType};
use nv_models::gemma4_wgpu::{
    prefill_m, sliding_kv_ring_rows_window_plus_prefill_chunk_plus_headroom, Gemma4Wgpu,
    HostBf16Lin, HostLayer, HostProj, HostWeights, SLIDING_KV_RING_DEFAULT_ON,
};
use std::path::PathBuf;
use common::EnvPins;
use common::LcgShift33Centered0p1 as Lcg;

const TINY_CONFIG_WINDOW_8_SO_THE_RING_WRAPS_WITHIN_A_FAST_TEST: &str = r#"{
  "text_config": {
    "hidden_size": 256,
    "intermediate_size": 512,
    "num_hidden_layers": 6,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "num_global_key_value_heads": 1,
    "head_dim": 64,
    "global_head_dim": 128,
    "vocab_size": 512,
    "max_position_embeddings": 1024,
    "rms_norm_eps": 1e-6,
    "sliding_window": 8,
    "final_logit_softcapping": 30.0,
    "layer_types": ["sliding_attention", "sliding_attention", "sliding_attention",
                    "sliding_attention", "sliding_attention", "full_attention"],
    "attention_k_eq_v": true,
    "hidden_activation": "gelu_pytorch_tanh",
    "num_kv_shared_layers": 0,
    "rope_parameters": {
      "full_attention": {"partial_rotary_factor": 0.25, "rope_theta": 1000000.0},
      "sliding_attention": {"rope_theta": 10000.0}
    }
  },
  "tie_word_embeddings": true
}"#;

const TINY_MAX_SEQ_PAST_THE_152_SLOT_RING: usize = 384;
const TINY_PREFILL_M_PINNED: usize = 16;

fn tiny_host_weights(config: &Gemma4Config, seed: u64) -> HostWeights {
    let mut rng = Lcg(seed);
    let hidden = config.hidden_size;
    let inter = config.intermediate_size;
    let n_q = config.num_attention_heads;
    let mut layers = Vec::new();
    for i in 0..config.num_hidden_layers {
        let kind = config.layer_kind(i);
        let hd = config.head_dim_for(kind);
        let nkv = config.num_kv_heads_for(kind);
        let q_dim = n_q * hd;
        let kv_dim = nkv * hd;
        let has_v = !matches!(
            (kind, config.attention_k_eq_v),
            (LayerType::FullAttention, true)
        );
        let qkv_rows = q_dim + kv_dim * if has_v { 2 } else { 1 };
        let mk_proj = |rng: &mut Lcg, n: usize, k: usize| {
            HostProj::Bf16(HostBf16Lin {
                w: rng.bf16_vec(n * k),
                n,
                k,
            })
        };
        layers.push(HostLayer {
            kind,
            input_ln: rng.bf16_vec_around_one(hidden),
            post_attn_ln: rng.bf16_vec_around_one(hidden),
            pre_ff_ln: rng.bf16_vec_around_one(hidden),
            post_ff_ln: rng.bf16_vec_around_one(hidden),
            q_norm: rng.bf16_vec_around_one(hd),
            k_norm: rng.bf16_vec_around_one(hd),
            layer_scalar: 0.9,
            has_v,
            qkv: mk_proj(&mut rng, qkv_rows, hidden),
            o: mk_proj(&mut rng, hidden, q_dim),
            gate_up: mk_proj(&mut rng, 2 * inter, hidden),
            down: mk_proj(&mut rng, hidden, inter),
        });
    }
    HostWeights {
        embed: rng.bf16_vec(config.vocab_size * hidden),
        final_norm: rng.bf16_vec_around_one(hidden),
        layers,
    }
}

static ENV_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

fn env_lock() -> std::sync::MutexGuard<'static, ()> {
    ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner())
}

fn tiny_model(ring_on: bool, seed: u64) -> Gemma4Wgpu {
    let config =
        Gemma4Config::from_hf_json_str(TINY_CONFIG_WINDOW_8_SO_THE_RING_WRAPS_WITHIN_A_FAST_TEST)
            .unwrap();
    let weights = tiny_host_weights(&config, seed);
    let m_s = TINY_PREFILL_M_PINNED.to_string();
    let pins = EnvPins::pin(&[
        ("NV_G4_WGPU_KV_RING", Some(if ring_on { "1" } else { "0" })),
        ("NV_G4_WGPU_PREFILL_M", Some(m_s.as_str())),
        ("NV_G4_WGPU_W8_FFN", Some("0")),
        ("NV_WGPU_BATCH_SLOTS", None),
    ]);
    let m = Gemma4Wgpu::new(config, &weights, TINY_MAX_SEQ_PAST_THE_152_SLOT_RING);
    drop(pins);
    m.unwrap()
}

fn tiny_ring_rows() -> usize {
    sliding_kv_ring_rows_window_plus_prefill_chunk_plus_headroom(8, TINY_PREFILL_M_PINNED)
}

#[test]
fn ring_gate_defaults_off_and_shrinks_only_the_sliding_layers_when_on() {
    let _g = env_lock();
    assert!(
        !SLIDING_KV_RING_DEFAULT_ON,
        "the sliding kv ring must stay opt-in: published gemma4-wgpu numbers were measured on full-depth caches"
    );
    if ctx_or_skip().is_none() {
        return;
    }
    let max_seq = TINY_MAX_SEQ_PAST_THE_152_SLOT_RING;
    let ring = tiny_ring_rows();
    assert!(
        max_seq > ring,
        "test geometry must engage the ring: max_seq {max_seq} <= ring {ring}"
    );

    let off = tiny_model(false, 0x2b);
    let on = tiny_model(true, 0x2b);
    let sliding = (2, 64);
    let full = (1, 128);
    for li in 0..off.kv_layer_count() {
        let (nkv, hd) = if li == 5 { full } else { sliding };
        let off_lens = off.kv_layer_lens(li).unwrap();
        let on_lens = on.kv_layer_lens(li).unwrap();
        assert_eq!(
            off_lens,
            [
                max_seq * nkv * hd / 4,
                max_seq * nkv * hd / 4,
                max_seq * nkv,
                max_seq * nkv
            ],
            "ring off must keep full-depth kv at layer {li}"
        );
        let expect_rows = if li == 5 { max_seq } else { ring };
        assert_eq!(
            on_lens,
            [
                expect_rows * nkv * hd / 4,
                expect_rows * nkv * hd / 4,
                expect_rows * nkv,
                expect_rows * nkv
            ],
            "ring on must shrink sliding layers to {ring} rows and leave full-attention at {max_seq} (layer {li})"
        );
    }
}

fn decode_bits(m: &mut Gemma4Wgpu, steps: usize, seed: u64) -> Vec<u32> {
    let mut rng = Lcg(seed);
    let vocab = m.config().vocab_size as u32;
    let mut bits = Vec::new();
    for _ in 0..steps {
        let t = rng.next_u32() % vocab;
        let (tok, logits) = m.decode_step_logits(t).unwrap();
        bits.push(tok);
        bits.extend(logits.iter().map(|v| v.to_bits()));
    }
    bits
}

#[test]
fn ring_decode_is_bit_identical_to_full_depth_far_past_the_wraparound() {
    let _g = env_lock();
    if ctx_or_skip().is_none() {
        return;
    }
    let steps = 340;
    assert!(
        steps > tiny_ring_rows() + 8,
        "decode must run past the ring wrap to prove anything"
    );
    let mut off = tiny_model(false, 0x2b);
    let mut on = tiny_model(true, 0x2b);
    let bits_off = decode_bits(&mut off, steps, 0x517);
    let bits_on = decode_bits(&mut on, steps, 0x517);
    let diff = bits_off
        .iter()
        .zip(bits_on.iter())
        .filter(|(a, b)| a != b)
        .count();
    assert_eq!(
        diff, 0,
        "ring decode must be bit-identical to full depth: the same values land in wrapped slots and every sliding read stays inside the window ({diff}/{} words differ)",
        bits_off.len()
    );
}

#[test]
fn ring_chunked_prefill_then_decode_is_bit_identical_to_full_depth() {
    let _g = env_lock();
    if ctx_or_skip().is_none() {
        return;
    }
    let mut rng = Lcg(0xfeed);
    let prompt: Vec<u32> = (0..320).map(|_| rng.next_u32() % 512).collect();
    let mut off = tiny_model(false, 0x2b);
    let mut on = tiny_model(true, 0x2b);
    assert!(
        off.prefill_chunk_len() == TINY_PREFILL_M_PINNED
            && on.prefill_chunk_len() == TINY_PREFILL_M_PINNED,
        "both arms need live chunked prefill for this gate to mean anything (got {} / {})",
        off.prefill_chunk_len(),
        on.prefill_chunk_len()
    );
    assert!(
        prompt.len() > tiny_ring_rows(),
        "the prompt must push prefill past the ring wrap"
    );
    let done_off = off.prefill_tokens(&prompt).unwrap();
    let done_on = on.prefill_tokens(&prompt).unwrap();
    assert_eq!(done_off, done_on);
    assert_eq!(done_off, prompt.len());
    let bits_off = decode_bits(&mut off, 8, 0x99);
    let bits_on = decode_bits(&mut on, 8, 0x99);
    let diff = bits_off
        .iter()
        .zip(bits_on.iter())
        .filter(|(a, b)| a != b)
        .count();
    assert_eq!(
        diff, 0,
        "post-prefill ring decode must be bit-identical to full depth ({diff}/{} words differ)",
        bits_off.len()
    );
}

fn gemma4_31b_snapshot_dir_env_override_then_home_hub() -> PathBuf {
    if let Ok(d) = std::env::var("NV_G4_SNAPSHOT") {
        return PathBuf::from(d);
    }
    let home = std::env::var("HOME").expect("HOME");
    let base = PathBuf::from(&home)
        .join(".cache/huggingface/hub/models--nvidia--Gemma-4-31B-IT-NVFP4/snapshots");
    let mut candidates: Vec<PathBuf> = std::fs::read_dir(&base)
        .unwrap_or_else(|e| panic!("gemma4 snapshots dir {base:?} missing: {e}"))
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.join("config.json").is_file())
        .collect();
    candidates.sort();
    candidates
        .into_iter()
        .next()
        .expect("no gemma4 NVFP4 snapshot under HOME hub; set NV_G4_SNAPSHOT")
}

const LADDER_168K_TOKENS: usize = 168 * 1024;
const LADDER_DECODE_HEADROOM: usize = 64 + 16;

#[test]
#[ignore = "loads the real 31B at 168k depth; set NV_G4_KV_RING_168K_TEST=1 -- capability gate: with NV_G4_WGPU_KV_RING=1 the 168k-context graph must ALLOCATE (full-depth sliding caches are ~66 GiB fp8 and OOM the same card)"]
fn real_31b_with_kv_ring_allocates_the_168k_context_graph_and_steps() {
    let _g = env_lock();
    if std::env::var("NV_G4_KV_RING_168K_TEST").ok().as_deref() != Some("1") {
        panic!("set NV_G4_KV_RING_168K_TEST=1 to run; a silent skip would read as a pass");
    }
    let ctx = ctx_or_skip().expect("this gate needs a real wgpu adapter");
    let _ = ctx;
    let pins = EnvPins::pin(&[("NV_G4_WGPU_KV_RING", Some("1"))]);
    let dir = gemma4_31b_snapshot_dir_env_override_then_home_hub();
    let config = Gemma4Config::from_hf_json_file(&dir.join("config.json")).unwrap();
    let loader = nv_weights::WeightLoader::open_dir(&dir, &candle_core::Device::Cpu).unwrap();
    let host = nv_models::gemma4_wgpu::host_weights_from_loader(&config, &loader).unwrap();
    drop(loader);
    let window = config.sliding_window;
    let max_seq = LADDER_168K_TOKENS + LADDER_DECODE_HEADROOM;
    let mut m = Gemma4Wgpu::new(config, &host, max_seq)
        .expect("168k graph must allocate with the sliding kv ring on");
    drop(pins);
    let ring = sliding_kv_ring_rows_window_plus_prefill_chunk_plus_headroom(window, prefill_m());
    let mut kv_words: u64 = 0;
    for li in 0..m.kv_layer_count() {
        let lens = m.kv_layer_lens(li).unwrap();
        kv_words += lens.iter().map(|&l| l as u64).sum::<u64>();
    }
    let sliding_lens = m.kv_layer_lens(0).unwrap();
    let nkv_sliding = m.config().num_kv_heads_for(LayerType::SlidingAttention);
    assert_eq!(
        sliding_lens[2] / nkv_sliding,
        ring,
        "sliding layer 0 must hold exactly the ring rows, not max_seq {max_seq}"
    );
    let (tok, logits) = m.decode_step_logits(2).unwrap();
    assert!(
        logits.iter().all(|v| v.is_finite()),
        "first decode step at 168k capacity must produce finite logits"
    );
    eprintln!(
        "KV-RING-168K allocate=ok max_seq={max_seq} ring_rows={ring} kv_total_MiB={:.0} first_tok={tok}",
        kv_words as f64 * 4.0 / (1024.0 * 1024.0)
    );
}
