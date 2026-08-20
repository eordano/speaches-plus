#![cfg(feature = "cuda")]

use candle_core::{DType, Device, Tensor};
use nv_models::qwen3_5_moe::{LayerType, Qwen3Moe, Qwen3_5DenseConfig};
use nv_specdecode::qwen38_mtp::{
    Qwen38DenseMtpHead, Qwen38MtpDecodeSession, Qwen38MtpSelfSpecEngine,
};
use std::collections::HashMap;
use std::path::PathBuf;

const TINY_LAYERS_8_KEEPS_TWO_FULL_ATTENTION_SLOTS_OF_THE_INTERVAL_4_PATTERN: usize = 8;
const TINY_HIDDEN_128_HEADS_12_KV_2_HD_64_KEEP_THE_RELEASE_GQA_RATIO_24_OVER_4: (usize, usize, usize, usize) =
    (128, 12, 2, 64);
const TINY_INTER_192_VOCAB_64_MAX_POS_64: (usize, usize, usize) = (192, 64, 64);
const TINY_GDN_V6_K2_HD16_KEEP_THE_RELEASE_V_OVER_K_RATIO_3: (usize, usize, usize) = (6, 2, 16);

fn tiny_q38_config() -> Qwen3_5DenseConfig {
    let n = TINY_LAYERS_8_KEEPS_TWO_FULL_ATTENTION_SLOTS_OF_THE_INTERVAL_4_PATTERN;
    let (hidden, heads, kv, hd) = TINY_HIDDEN_128_HEADS_12_KV_2_HD_64_KEEP_THE_RELEASE_GQA_RATIO_24_OVER_4;
    let (inter, vocab, max_pos) = TINY_INTER_192_VOCAB_64_MAX_POS_64;
    let (gdn_v, gdn_k, gdn_hd) = TINY_GDN_V6_K2_HD16_KEEP_THE_RELEASE_V_OVER_K_RATIO_3;
    let layer_types: Vec<LayerType> = (0..n)
        .map(|i| {
            if (i + 1) % 4 == 0 {
                LayerType::FullAttention
            } else {
                LayerType::LinearAttention
            }
        })
        .collect();
    Qwen3_5DenseConfig {
        hidden_size: hidden,
        num_hidden_layers: n,
        num_attention_heads: heads,
        num_key_value_heads: kv,
        head_dim: hd,
        intermediate_size: inter,
        vocab_size: vocab,
        max_position_embeddings: max_pos,
        rope_theta: 10_000_000.0,
        rms_norm_eps: 1e-6,
        partial_rotary_factor: 0.25,
        bos_token_id: None,
        eos_token_id: 1,
        layer_types,
        linear_num_key_heads: gdn_k,
        linear_num_value_heads: gdn_v,
        linear_key_head_dim: gdn_hd,
        linear_value_head_dim: gdn_hd,
        linear_conv_kernel_dim: 4,
        attn_output_gate: true,
        tie_word_embeddings: false,
    }
}

struct Lcg(u64);

impl Lcg {
    fn new(seed: u64) -> Self {
        Self(seed | 1)
    }
    fn next_f32(&mut self) -> f32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((self.0 >> 33) as u32) as f32 / (1u64 << 31) as f32 - 1.0
    }
    fn vec(&mut self, n: usize, scale: f32) -> Vec<f32> {
        (0..n)
            .map(|_| half::bf16::from_f32(self.next_f32() * scale).to_f32())
            .collect()
    }
    fn norm_effective_near_one(&mut self, n: usize) -> Vec<f32> {
        (0..n)
            .map(|_| half::bf16::from_f32(1.0 + 0.1 * self.next_f32()).to_f32())
            .collect()
    }
}

fn bf16_tensor(vals: &[f32], shape: &[usize], device: &Device) -> Tensor {
    Tensor::from_vec(vals.to_vec(), shape, &Device::Cpu)
        .expect("cpu tensor")
        .to_dtype(DType::BF16)
        .expect("bf16")
        .to_device(device)
        .expect("to device")
}

fn stored_minus_one_because_the_loader_adds_one_to_zero_centered_norms(v: &[f32]) -> Vec<f32> {
    v.iter().map(|x| x - 1.0).collect()
}

fn write_tiny_trunk_safetensors_dir(cfg: &Qwen3_5DenseConfig, r: &mut Lcg) -> PathBuf {
    let cpu = Device::Cpu;
    let hidden = cfg.hidden_size;
    let inter = cfg.intermediate_size;
    let hd = cfg.head_dim;
    let key_dim = cfg.linear_num_key_heads * cfg.linear_key_head_dim;
    let value_dim = cfg.linear_num_value_heads * cfg.linear_value_head_dim;
    let conv_dim = 2 * key_dim + value_dim;
    let ks = cfg.linear_conv_kernel_dim;
    let n_v = cfg.linear_num_value_heads;
    let mut t: HashMap<String, Tensor> = HashMap::new();
    let mut norm = |r: &mut Lcg, n: usize| {
        stored_minus_one_because_the_loader_adds_one_to_zero_centered_norms(
            &r.norm_effective_near_one(n),
        )
    };
    t.insert(
        "model.language_model.embed_tokens.weight".into(),
        bf16_tensor(&r.vec(cfg.vocab_size * hidden, 0.6), &[cfg.vocab_size, hidden], &cpu),
    );
    t.insert(
        "model.language_model.norm.weight".into(),
        bf16_tensor(&norm(r, hidden), &[hidden], &cpu),
    );
    t.insert(
        "lm_head.weight".into(),
        bf16_tensor(&r.vec(cfg.vocab_size * hidden, 0.2), &[cfg.vocab_size, hidden], &cpu),
    );
    for (i, lt) in cfg.layer_types.iter().enumerate() {
        let p = format!("model.language_model.layers.{i}");
        t.insert(
            format!("{p}.input_layernorm.weight"),
            bf16_tensor(&norm(r, hidden), &[hidden], &cpu),
        );
        t.insert(
            format!("{p}.post_attention_layernorm.weight"),
            bf16_tensor(&norm(r, hidden), &[hidden], &cpu),
        );
        match lt {
            LayerType::LinearAttention => {
                let q = format!("{p}.linear_attn");
                t.insert(
                    format!("{q}.in_proj_qkv.weight"),
                    bf16_tensor(&r.vec(conv_dim * hidden, 0.12), &[conv_dim, hidden], &cpu),
                );
                t.insert(
                    format!("{q}.in_proj_z.weight"),
                    bf16_tensor(&r.vec(value_dim * hidden, 0.12), &[value_dim, hidden], &cpu),
                );
                t.insert(
                    format!("{q}.in_proj_a.weight"),
                    bf16_tensor(&r.vec(n_v * hidden, 0.12), &[n_v, hidden], &cpu),
                );
                t.insert(
                    format!("{q}.in_proj_b.weight"),
                    bf16_tensor(&r.vec(n_v * hidden, 0.12), &[n_v, hidden], &cpu),
                );
                t.insert(
                    format!("{q}.conv1d.weight"),
                    bf16_tensor(&r.vec(conv_dim * ks, 0.4), &[conv_dim, 1, ks], &cpu),
                );
                t.insert(format!("{q}.A_log"), bf16_tensor(&r.vec(n_v, 0.5), &[n_v], &cpu));
                t.insert(format!("{q}.dt_bias"), bf16_tensor(&r.vec(n_v, 0.5), &[n_v], &cpu));
                t.insert(
                    format!("{q}.norm.weight"),
                    bf16_tensor(
                        &r.norm_effective_near_one(cfg.linear_value_head_dim),
                        &[cfg.linear_value_head_dim],
                        &cpu,
                    ),
                );
                t.insert(
                    format!("{q}.out_proj.weight"),
                    bf16_tensor(&r.vec(hidden * value_dim, 0.12), &[hidden, value_dim], &cpu),
                );
            }
            LayerType::FullAttention => {
                let a = format!("{p}.self_attn");
                let q_out = cfg.num_attention_heads * hd * 2;
                let kv_out = cfg.num_key_value_heads * hd;
                t.insert(
                    format!("{a}.q_proj.weight"),
                    bf16_tensor(&r.vec(q_out * hidden, 0.12), &[q_out, hidden], &cpu),
                );
                t.insert(
                    format!("{a}.k_proj.weight"),
                    bf16_tensor(&r.vec(kv_out * hidden, 0.12), &[kv_out, hidden], &cpu),
                );
                t.insert(
                    format!("{a}.v_proj.weight"),
                    bf16_tensor(&r.vec(kv_out * hidden, 0.12), &[kv_out, hidden], &cpu),
                );
                t.insert(
                    format!("{a}.o_proj.weight"),
                    bf16_tensor(
                        &r.vec(hidden * cfg.num_attention_heads * hd, 0.12),
                        &[hidden, cfg.num_attention_heads * hd],
                        &cpu,
                    ),
                );
                t.insert(format!("{a}.q_norm.weight"), bf16_tensor(&norm(r, hd), &[hd], &cpu));
                t.insert(format!("{a}.k_norm.weight"), bf16_tensor(&norm(r, hd), &[hd], &cpu));
            }
        }
        t.insert(
            format!("{p}.mlp.gate_proj.weight"),
            bf16_tensor(&r.vec(inter * hidden, 0.15), &[inter, hidden], &cpu),
        );
        t.insert(
            format!("{p}.mlp.up_proj.weight"),
            bf16_tensor(&r.vec(inter * hidden, 0.15), &[inter, hidden], &cpu),
        );
        t.insert(
            format!("{p}.mlp.down_proj.weight"),
            bf16_tensor(&r.vec(hidden * inter, 0.15), &[hidden, inter], &cpu),
        );
    }
    let dir = std::env::temp_dir().join(format!(
        "q38-mtp-synth-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    std::fs::create_dir_all(&dir).expect("mk temp safetensors dir");
    candle_core::safetensors::save(&t, dir.join("model.safetensors")).expect("save tiny trunk");
    dir
}

fn synthetic_mtp_map(base: &Qwen3Moe, r: &mut Lcg, device: &Device) -> HashMap<String, Tensor> {
    let g = Qwen38DenseMtpHead::geometry_from_dense_base(base).expect("dense base geometry");
    let mut m = HashMap::new();
    for (name, shape) in g.expected_tensor_shapes() {
        let n: usize = shape.iter().product();
        let vals = if name.ends_with("norm.weight")
            || name.contains("norm_embedding")
            || name.contains("norm_hidden")
        {
            stored_minus_one_because_the_loader_adds_one_to_zero_centered_norms(
                &r.norm_effective_near_one(n),
            )
        } else {
            r.vec(n, 0.12)
        };
        m.insert(name, bf16_tensor(&vals, &shape, device));
    }
    m
}

struct Rig {
    base: Qwen3Moe,
    mtp: Qwen38DenseMtpHead,
    device: Device,
}

fn build_rig(seed: u64) -> Option<Rig> {
    let Ok(device) = Device::new_cuda_with_stream(0) else {
        eprintln!("[skip] no cuda device; the qwen38 mtp synthetic forward suite needs the card");
        return None;
    };
    let cfg = tiny_q38_config();
    let mut r = Lcg::new(seed);
    let dir = write_tiny_trunk_safetensors_dir(&cfg, &mut r);
    let weights = nv_weights::WeightLoader::open_dir(&dir, &device).expect("open tiny dir");
    let base = Qwen3Moe::from_loader_dense(cfg, &weights, &device).expect("build tiny cuda trunk");
    drop(weights);
    let _ = std::fs::remove_dir_all(&dir);
    let map = synthetic_mtp_map(&base, &mut r, &device);
    let mtp = Qwen38DenseMtpHead::from_map(&map, &base).expect("build tiny mtp head");
    Some(Rig { base, mtp, device })
}

const SEED: u64 = 0x938_027b_0002;

fn trunk_hidden_rows(rig: &Rig, tokens: &[u32]) -> Tensor {
    let seq = tokens.len();
    let mut cache = rig.base.new_kv_cache(64).expect("cache");
    let toks = Tensor::from_vec(tokens.to_vec(), (1usize, seq), &rig.device).expect("toks");
    let pos =
        Tensor::from_vec((0..seq as i32).collect::<Vec<i32>>(), seq, &rig.device).expect("pos");
    let (_logits, hidden) = rig
        .base
        .forward_with_cache_dispatched_hidden(&toks, &pos, &mut cache, None)
        .expect("trunk forward");
    hidden
}

#[test]
fn draft_step_kv_rows_bitwise_match_catch_up_rows_for_identical_inputs() {
    let Some(rig) = build_rig(SEED) else { return };
    let prompt: Vec<u32> = vec![3, 11, 5, 40, 2, 19];
    let hidden = trunk_hidden_rows(&rig, &prompt);
    let seq = prompt.len();
    let last_hidden = hidden.narrow(1, seq - 1, 1).unwrap().contiguous().unwrap();

    let mut cache_a = rig.mtp.new_kv_cache(64, &rig.device).expect("cache a");
    let mut cache_b = rig.mtp.new_kv_cache(64, &rig.device).expect("cache b");
    rig.mtp
        .prefill_prompt_kv_shifted_by_one_with_position_zero_on_a_zero_hidden(
            &rig.base, &prompt, &hidden, &mut cache_a,
        )
        .expect("prefill a");
    rig.mtp
        .prefill_prompt_kv_shifted_by_one_with_position_zero_on_a_zero_hidden(
            &rig.base, &prompt, &hidden, &mut cache_b,
        )
        .expect("prefill b");

    let anchor = 7u32;
    let (_logits, _h) = rig
        .mtp
        .forward_draft(&rig.base, &last_hidden, anchor, &mut cache_a)
        .expect("draft step on cache a");
    rig.mtp
        .catch_up_kv_recomputing_and_discarding_q_which_v1_prices_over_a_kv_only_projection(
            &rig.base,
            &[anchor],
            &last_hidden,
            &mut cache_b,
        )
        .expect("catch-up on cache b");

    assert_eq!(cache_a.len(), seq + 1);
    assert_eq!(cache_b.len(), seq + 1);
    let (ka, va) = (
        cache_a.k_rows_host_f32().unwrap(),
        cache_a.v_rows_host_f32().unwrap(),
    );
    let (kb, vb) = (
        cache_b.k_rows_host_f32().unwrap(),
        cache_b.v_rows_host_f32().unwrap(),
    );
    assert_eq!(
        ka, kb,
        "a draft step and a catch-up step over identical (token, trunk hidden) inputs must \
         write bitwise-identical K rows; divergence means the two projection paths forked"
    );
    assert_eq!(va, vb, "same invariant for V rows");
}

#[test]
fn draft_logits_read_the_kv_prefix_so_two_different_histories_disagree() {
    let Some(rig) = build_rig(SEED) else { return };
    let prompt_a: Vec<u32> = vec![3, 11, 5, 40, 2, 19];
    let prompt_b: Vec<u32> = vec![50, 14, 33, 21, 8, 12];
    let hidden_a = trunk_hidden_rows(&rig, &prompt_a);
    let hidden_b = trunk_hidden_rows(&rig, &prompt_b);
    let seq = prompt_a.len();
    let probe_hidden = hidden_a.narrow(1, seq - 1, 1).unwrap().contiguous().unwrap();

    let mut cache_a = rig.mtp.new_kv_cache(64, &rig.device).expect("cache a");
    let mut cache_b = rig.mtp.new_kv_cache(64, &rig.device).expect("cache b");
    rig.mtp
        .prefill_prompt_kv_shifted_by_one_with_position_zero_on_a_zero_hidden(
            &rig.base, &prompt_a, &hidden_a, &mut cache_a,
        )
        .expect("prefill a");
    rig.mtp
        .prefill_prompt_kv_shifted_by_one_with_position_zero_on_a_zero_hidden(
            &rig.base, &prompt_b, &hidden_b, &mut cache_b,
        )
        .expect("prefill b");

    let anchor = 7u32;
    let la = rig
        .mtp
        .forward_draft(&rig.base, &probe_hidden, anchor, &mut cache_a)
        .expect("draft over history a")
        .0
        .to_dtype(DType::F32)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();
    let lb = rig
        .mtp
        .forward_draft(&rig.base, &probe_hidden, anchor, &mut cache_b)
        .expect("draft over history b")
        .0
        .to_dtype(DType::F32)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();
    let max_abs_diff = la
        .iter()
        .zip(&lb)
        .fold(0f32, |m, (a, b)| m.max((a - b).abs()));
    assert!(la.iter().all(|v| v.is_finite()), "draft logits must be finite");
    assert!(
        max_abs_diff > 0.0,
        "identical draft inputs over two different KV histories produced identical logits: \
         the drafter attention is not reading its cache"
    );
}

#[test]
fn catch_up_over_an_extension_bitwise_equals_prefill_over_the_extended_prompt() {
    let Some(rig) = build_rig(SEED) else { return };
    let prompt: Vec<u32> = vec![3, 11, 5, 40, 2, 19];
    let ext: Vec<u32> = vec![7, 33, 21];
    let full: Vec<u32> = prompt.iter().chain(&ext).copied().collect();
    let hidden = trunk_hidden_rows(&rig, &full);
    let p = prompt.len();
    let e = ext.len();

    let mut cache_a = rig.mtp.new_kv_cache(64, &rig.device).expect("cache a");
    rig.mtp
        .prefill_prompt_kv_shifted_by_one_with_position_zero_on_a_zero_hidden(
            &rig.base, &full, &hidden, &mut cache_a,
        )
        .expect("prefill over the extended prompt");

    let mut cache_b = rig.mtp.new_kv_cache(64, &rig.device).expect("cache b");
    rig.mtp
        .prefill_prompt_kv_shifted_by_one_with_position_zero_on_a_zero_hidden(
            &rig.base,
            &prompt,
            &hidden.narrow(1, 0, p).unwrap().contiguous().unwrap(),
            &mut cache_b,
        )
        .expect("prefill over the base prompt");
    rig.mtp
        .catch_up_kv_recomputing_and_discarding_q_which_v1_prices_over_a_kv_only_projection(
            &rig.base,
            &ext,
            &hidden.narrow(1, p - 1, e).unwrap().contiguous().unwrap(),
            &mut cache_b,
        )
        .expect("catch-up over the extension");

    assert_eq!(cache_a.len(), p + e);
    assert_eq!(cache_b.len(), p + e);
    assert_eq!(
        cache_a.k_rows_host_f32().unwrap(),
        cache_b.k_rows_host_f32().unwrap(),
        "the shifted-by-one catch-up over accepted tokens must reproduce, bitwise, the KV a \
         from-scratch prompt prefill would write; a mismatch is an off-by-one in the \
         (token, preceding-hidden) pairing"
    );
    assert_eq!(
        cache_a.v_rows_host_f32().unwrap(),
        cache_b.v_rows_host_f32().unwrap(),
        "same invariant for V rows"
    );
}

#[test]
fn clairvoyant_oracle_rounds_commit_full_and_partial_accepts_losslessly() {
    let Some(rig) = build_rig(SEED) else { return };
    let prompt: Vec<u32> = vec![3, 11, 5, 40, 2, 19, 7, 33, 21, 8];
    const MAX_NEW: usize = 16;
    const MAX_SEQ: usize = 64;
    const K: usize = 3;
    let vocab = TINY_INTER_192_VOCAB_64_MAX_POS_64.1 as u32;

    let eng = Qwen38MtpSelfSpecEngine::new(&rig.base, &rig.mtp, K).expect("engine");
    let (ref_ids, _) = eng
        .generate_reference(&prompt, MAX_NEW, MAX_SEQ)
        .expect("reference stream");
    assert!(ref_ids.len() >= 1 + K + 1 + K, "reference too short to script rounds");

    let mut session =
        Qwen38MtpDecodeSession::start(&rig.base, &rig.mtp, K, &prompt, MAX_SEQ).expect("session");
    assert_eq!(session.anchor(), ref_ids[0]);
    let mut emitted_stream: Vec<u32> = vec![session.anchor()];

    let full_accept_drafts: Vec<u32> = ref_ids[1..1 + K].to_vec();
    let emitted = session
        .round_with_drafts_from_a_clairvoyant_test_oracle(&full_accept_drafts)
        .expect("full-accept round");
    assert_eq!(
        emitted.len(),
        K + 1,
        "a fully accepted round must emit k drafts plus the bonus token"
    );
    emitted_stream.extend(&emitted);
    assert_eq!(session.stats.accepted, K);

    let next = emitted_stream.len();
    let mut partial_drafts: Vec<u32> = vec![ref_ids[next]];
    let poison = (ref_ids[next + 1] + 1) % vocab;
    partial_drafts.push(poison);
    partial_drafts.push(poison);
    let emitted = session
        .round_with_drafts_from_a_clairvoyant_test_oracle(&partial_drafts)
        .expect("partial-accept round");
    assert_eq!(
        emitted.len(),
        2,
        "accepting exactly one draft must emit that draft plus the bonus token"
    );
    emitted_stream.extend(&emitted);
    assert_eq!(session.stats.accepted, K + 1);

    while emitted_stream.len() < ref_ids.len() && session.round_fits() {
        let emitted = session.round().expect("self-drafted round");
        emitted_stream.extend(&emitted);
    }
    emitted_stream.truncate(ref_ids.len());
    assert_eq!(
        emitted_stream, ref_ids,
        "scripted full-accept and partial-accept commits followed by self-drafted rounds must \
         reproduce the greedy reference exactly; divergence is a catch-up, rewind, or GDN \
         rollback bug on the accepted>=1 path"
    );
}

#[test]
fn spec_stream_equals_the_greedy_reference_on_synthetic_weights() {
    let Some(rig) = build_rig(SEED) else { return };
    let prompt: Vec<u32> = vec![3, 11, 5, 40, 2, 19, 7, 33, 21, 8];
    const MAX_NEW: usize = 24;
    const MAX_SEQ: usize = 64;
    for k in [1usize, 3] {
        let eng = Qwen38MtpSelfSpecEngine::new(&rig.base, &rig.mtp, k).expect("engine");
        let (ref_ids, ref_stats) = eng
            .generate_reference(&prompt, MAX_NEW, MAX_SEQ)
            .expect("reference stream");
        let (spec_ids, stats) = eng
            .generate_greedy(&prompt, MAX_NEW, MAX_SEQ)
            .expect("spec stream");
        eprintln!(
            "[q38-mtp-synth] basis: synthetic tiny geometry seed={SEED:#x} backend=cuda k={k} \
             max_new={MAX_NEW} rounds={} drafted={} accepted={} accept_rate={:.3}",
            stats.rounds,
            stats.drafted,
            stats.accepted,
            stats.accept_rate(),
        );
        assert_eq!(ref_stats.emitted + 1, ref_ids.len().max(1));
        assert_eq!(
            spec_ids, ref_ids,
            "k={k}: the speculative stream must equal the single-token greedy reference exactly; \
             divergence is a rollback, reanchor, or mtp-kv catch-up bug"
        );
        assert!(stats.rounds > 0 && stats.drafted == stats.rounds * k);
    }
}
