#![cfg(feature = "cuda")]

use candle_core::backend::BackendDevice;
use candle_core::{DType, Device, Tensor};
use nv_models::qwen3_5_moe::{LayerType, Qwen3Moe, Qwen3_5DenseConfig};
use std::collections::HashMap;
use std::path::PathBuf;

const TINY_LAYERS_8_KEEPS_TWO_FULL_ATTENTION_SLOTS_OF_THE_INTERVAL_4_PATTERN: usize = 8;
const TINY_HIDDEN_128_HEAD_DIM_64_Q12_KV2_MATCH_THE_GEOMETRY_SMOKE_TINY_ARM: (usize, usize, usize, usize) =
    (128, 64, 12, 2);
const TINY_GDN_V6_K2_DIMS_16_CONV_KERNEL_4: (usize, usize, usize, usize) = (6, 2, 16, 4);
const TINY_INTER_192_VOCAB_64_MAX_POS_64: (usize, usize, usize) = (192, 64, 64);

const PREFILL_6_THEN_DECODE_4_TOUCHES_CHUNK_PREFILL_AND_THE_GDN_AND_W4A8_DECODE_SCRATCH_PATHS:
    (usize, usize) = (6, 4);

const POST_DROP_FREE_TOL_MIB_64_AN_EAGER_TINY_ENGINE_HAS_NO_GRAPH_POOL_SO_ONLY_ALLOCATOR_ROUNDING_MAY_REMAIN:
    f64 = 64.0;

const CYCLE_SEED: u64 = 0x9380_27b0_c5a1;

fn tiny_q38_dense_config() -> Qwen3_5DenseConfig {
    let layers = TINY_LAYERS_8_KEEPS_TWO_FULL_ATTENTION_SLOTS_OF_THE_INTERVAL_4_PATTERN;
    let (hidden, head_dim, n_q, n_kv) = TINY_HIDDEN_128_HEAD_DIM_64_Q12_KV2_MATCH_THE_GEOMETRY_SMOKE_TINY_ARM;
    let (gdn_v, gdn_k, gdn_d, conv_k) = TINY_GDN_V6_K2_DIMS_16_CONV_KERNEL_4;
    let (inter, vocab, max_pos) = TINY_INTER_192_VOCAB_64_MAX_POS_64;
    let layer_types = (0..layers)
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
        num_hidden_layers: layers,
        num_attention_heads: n_q,
        num_key_value_heads: n_kv,
        head_dim,
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
        linear_key_head_dim: gdn_d,
        linear_value_head_dim: gdn_d,
        linear_conv_kernel_dim: conv_k,
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
        (0..n).map(|_| self.next_f32() * scale).collect()
    }
    fn near_zero_norm_vec_because_loaders_add_one(&mut self, n: usize) -> Vec<f32> {
        (0..n).map(|_| 0.1 * self.next_f32()).collect()
    }
}

fn bf16_tensor(vals: &[f32], shape: &[usize]) -> Tensor {
    Tensor::from_vec(vals.to_vec(), shape, &Device::Cpu)
        .expect("cpu tensor")
        .to_dtype(DType::BF16)
        .expect("bf16 cast")
}

fn tiny_safetensors_map(cfg: &Qwen3_5DenseConfig, seed: u64) -> HashMap<String, Tensor> {
    let mut r = Lcg::new(seed);
    let hidden = cfg.hidden_size;
    let inter = cfg.intermediate_size;
    let hd = cfg.head_dim;
    let n_v = cfg.linear_num_value_heads;
    let key_dim = cfg.linear_num_key_heads * cfg.linear_key_head_dim;
    let value_dim = n_v * cfg.linear_value_head_dim;
    let conv_dim = 2 * key_dim + value_dim;
    let ks = cfg.linear_conv_kernel_dim;
    let mut t: HashMap<String, Tensor> = HashMap::new();
    t.insert(
        "model.language_model.embed_tokens.weight".into(),
        bf16_tensor(&r.vec(cfg.vocab_size * hidden, 0.6), &[cfg.vocab_size, hidden]),
    );
    t.insert(
        "model.language_model.norm.weight".into(),
        bf16_tensor(&r.near_zero_norm_vec_because_loaders_add_one(hidden), &[hidden]),
    );
    t.insert(
        "lm_head.weight".into(),
        bf16_tensor(&r.vec(cfg.vocab_size * hidden, 0.2), &[cfg.vocab_size, hidden]),
    );
    for (i, lt) in cfg.layer_types.iter().enumerate() {
        let p = format!("model.language_model.layers.{i}");
        t.insert(
            format!("{p}.input_layernorm.weight"),
            bf16_tensor(&r.near_zero_norm_vec_because_loaders_add_one(hidden), &[hidden]),
        );
        t.insert(
            format!("{p}.post_attention_layernorm.weight"),
            bf16_tensor(&r.near_zero_norm_vec_because_loaders_add_one(hidden), &[hidden]),
        );
        match lt {
            LayerType::LinearAttention => {
                let q = format!("{p}.linear_attn");
                t.insert(
                    format!("{q}.in_proj_qkv.weight"),
                    bf16_tensor(&r.vec(conv_dim * hidden, 0.12), &[conv_dim, hidden]),
                );
                t.insert(
                    format!("{q}.in_proj_z.weight"),
                    bf16_tensor(&r.vec(value_dim * hidden, 0.12), &[value_dim, hidden]),
                );
                t.insert(
                    format!("{q}.in_proj_a.weight"),
                    bf16_tensor(&r.vec(n_v * hidden, 0.12), &[n_v, hidden]),
                );
                t.insert(
                    format!("{q}.in_proj_b.weight"),
                    bf16_tensor(&r.vec(n_v * hidden, 0.12), &[n_v, hidden]),
                );
                t.insert(
                    format!("{q}.conv1d.weight"),
                    bf16_tensor(&r.vec(conv_dim * ks, 0.4), &[conv_dim, 1, ks]),
                );
                t.insert(format!("{q}.A_log"), bf16_tensor(&r.vec(n_v, 0.5), &[n_v]));
                t.insert(format!("{q}.dt_bias"), bf16_tensor(&r.vec(n_v, 0.5), &[n_v]));
                t.insert(
                    format!("{q}.norm.weight"),
                    bf16_tensor(
                        &(0..cfg.linear_value_head_dim)
                            .map(|_| 1.0 + 0.1 * r.next_f32())
                            .collect::<Vec<f32>>(),
                        &[cfg.linear_value_head_dim],
                    ),
                );
                t.insert(
                    format!("{q}.out_proj.weight"),
                    bf16_tensor(&r.vec(hidden * value_dim, 0.12), &[hidden, value_dim]),
                );
            }
            LayerType::FullAttention => {
                let a = format!("{p}.self_attn");
                let q_out = cfg.num_attention_heads * hd * 2;
                let kv_out = cfg.num_key_value_heads * hd;
                t.insert(
                    format!("{a}.q_proj.weight"),
                    bf16_tensor(&r.vec(q_out * hidden, 0.12), &[q_out, hidden]),
                );
                t.insert(
                    format!("{a}.k_proj.weight"),
                    bf16_tensor(&r.vec(kv_out * hidden, 0.12), &[kv_out, hidden]),
                );
                t.insert(
                    format!("{a}.v_proj.weight"),
                    bf16_tensor(&r.vec(kv_out * hidden, 0.12), &[kv_out, hidden]),
                );
                t.insert(
                    format!("{a}.o_proj.weight"),
                    bf16_tensor(
                        &r.vec(hidden * cfg.num_attention_heads * hd, 0.12),
                        &[hidden, cfg.num_attention_heads * hd],
                    ),
                );
                t.insert(
                    format!("{a}.q_norm.weight"),
                    bf16_tensor(&r.near_zero_norm_vec_because_loaders_add_one(hd), &[hd]),
                );
                t.insert(
                    format!("{a}.k_norm.weight"),
                    bf16_tensor(&r.near_zero_norm_vec_because_loaders_add_one(hd), &[hd]),
                );
            }
        }
        t.insert(
            format!("{p}.mlp.gate_proj.weight"),
            bf16_tensor(&r.vec(inter * hidden, 0.15), &[inter, hidden]),
        );
        t.insert(
            format!("{p}.mlp.up_proj.weight"),
            bf16_tensor(&r.vec(inter * hidden, 0.15), &[inter, hidden]),
        );
        t.insert(
            format!("{p}.mlp.down_proj.weight"),
            bf16_tensor(&r.vec(hidden * inter, 0.15), &[hidden, inter]),
        );
    }
    t
}

fn write_fixture_dir(tag: &str, t: &HashMap<String, Tensor>) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("q38-vram-cycle-{tag}-{}", std::process::id()));
    std::fs::create_dir_all(&dir).expect("mk fixture dir");
    candle_core::safetensors::save(t, dir.join("model.safetensors")).expect("save tiny fixture");
    dir
}

fn free_mib() -> f64 {
    let (free, _total) = cudarc::driver::result::mem_get_info().expect("mem_get_info");
    free as f64 / (1u64 << 20) as f64
}

fn build_prefill_decode_drop(cfg: &Qwen3_5DenseConfig, dir: &PathBuf, device: &Device) {
    let weights = nv_weights::WeightLoader::open_dir(dir, device).expect("open tiny fixture");
    let model = Qwen3Moe::from_loader_dense(cfg.clone(), &weights, device).expect(
        "engine build failed: a rebuild after a full drop must succeed in the same process, the \
         one-engine-per-process restriction is a bug this suite exists to catch at tiny scale",
    );
    drop(weights);
    let (prefill, decode) =
        PREFILL_6_THEN_DECODE_4_TOUCHES_CHUNK_PREFILL_AND_THE_GDN_AND_W4A8_DECODE_SCRATCH_PATHS;
    let mut cache = model.new_kv_cache(prefill + decode + 2).expect("kv cache");
    let prompt: Vec<u32> = (0..prefill as u32).map(|i| (i * 7 + 3) % 64).collect();
    let tokens = Tensor::from_vec(prompt.clone(), (1usize, prefill), device).expect("tokens");
    let positions =
        Tensor::from_vec((0..prefill as i32).collect::<Vec<_>>(), prefill, device).expect("pos");
    let logits = model
        .forward_with_cache(&tokens, &positions, &mut cache)
        .expect("tiny prefill");
    let host: Vec<f32> = logits
        .to_dtype(DType::F32)
        .expect("f32")
        .flatten_all()
        .expect("flatten")
        .to_vec1()
        .expect("host");
    assert!(host.iter().all(|v| v.is_finite()), "prefill logits finite");
    for s in 0..decode {
        let pos = prefill + s;
        let tk = (pos as u32 * 5 + 1) % 64;
        let tokens = Tensor::from_vec(vec![tk], (1usize, 1usize), device).expect("token");
        let positions = Tensor::from_vec(vec![pos as i32], 1usize, device).expect("pos");
        let logits = model
            .forward_with_cache(&tokens, &positions, &mut cache)
            .expect("tiny decode step");
        let row: Vec<f32> = logits
            .to_dtype(DType::F32)
            .expect("f32")
            .flatten_all()
            .expect("flatten")
            .to_vec1()
            .expect("host");
        assert!(row.iter().all(|v| v.is_finite()), "decode logits finite at {pos}");
    }
    drop(cache);
    drop(model);
    if let Device::Cuda(d) = device {
        d.synchronize().expect("post-drop synchronize");
    }
}

#[test]
fn engine_cycle_x2_second_build_succeeds_and_post_drop_free_returns_to_the_cycle1_baseline() {
    let Ok(device) = Device::new_cuda(0) else {
        panic!(
            "no CUDA device 0: this suite is the tiny engine-cycle leak gate and must not report \
             success having executed nothing"
        );
    };
    let cfg = tiny_q38_dense_config();
    let dir = write_fixture_dir("cycle", &tiny_safetensors_map(&cfg, CYCLE_SEED));

    build_prefill_decode_drop(&cfg, &dir, &device);
    let free_after_cycle1 = free_mib();

    build_prefill_decode_drop(&cfg, &dir, &device);
    let free_after_cycle2 = free_mib();

    let _ = std::fs::remove_dir_all(&dir);
    let lost = free_after_cycle1 - free_after_cycle2;
    eprintln!(
        "[vram-cycle] basis: tiny synthetic q38 geometry seed={CYCLE_SEED:#x} eager cuda arm; \
         post-drop free cycle1={free_after_cycle1:.1} MiB cycle2={free_after_cycle2:.1} MiB \
         lost={lost:.1} MiB"
    );
    assert!(
        lost < POST_DROP_FREE_TOL_MIB_64_AN_EAGER_TINY_ENGINE_HAS_NO_GRAPH_POOL_SO_ONLY_ALLOCATOR_ROUNDING_MAY_REMAIN,
        "an identical build+prefill+decode+drop cycle left {lost:.1} MiB fewer free than the \
         previous cycle: engine drop is stranding device memory"
    );
}

#[test]
fn failed_load_missing_lm_head_frees_every_already_uploaded_layer_weight() {
    let Ok(device) = Device::new_cuda(0) else {
        panic!(
            "no CUDA device 0: this suite is the partial-construction leak gate and must not \
             report success having executed nothing"
        );
    };
    let cfg = tiny_q38_dense_config();
    let good = write_fixture_dir("warm", &tiny_safetensors_map(&cfg, CYCLE_SEED));
    build_prefill_decode_drop(&cfg, &good, &device);
    let _ = std::fs::remove_dir_all(&good);

    let mut missing_lm_head = tiny_safetensors_map(&cfg, CYCLE_SEED);
    missing_lm_head
        .remove("lm_head.weight")
        .expect("fixture writes lm_head.weight, removing it makes the LAST load in build fail so \
                 every layer weight is already uploaded when the error returns");
    let dir = write_fixture_dir("noheads", &missing_lm_head);

    let free_before = free_mib();
    let weights = nv_weights::WeightLoader::open_dir(&dir, &device).expect("open fixture");
    let err = Qwen3Moe::from_loader_dense(cfg.clone(), &weights, &device);
    assert!(
        err.is_err(),
        "a checkpoint without lm_head.weight and tie_word_embeddings=false must refuse to build"
    );
    drop(err);
    drop(weights);
    if let Device::Cuda(d) = &device {
        d.synchronize().expect("post-error synchronize");
    }
    let free_after = free_mib();
    let _ = std::fs::remove_dir_all(&dir);
    let lost = free_before - free_after;
    eprintln!(
        "[vram-error-path] basis: tiny synthetic q38 geometry, load aborted at lm_head; free \
         before={free_before:.1} MiB after={free_after:.1} MiB lost={lost:.1} MiB"
    );
    assert!(
        lost < POST_DROP_FREE_TOL_MIB_64_AN_EAGER_TINY_ENGINE_HAS_NO_GRAPH_POOL_SO_ONLY_ALLOCATOR_ROUNDING_MAY_REMAIN,
        "a failed engine build stranded {lost:.1} MiB of already-uploaded weights: the error \
         return path is not dropping partial construction"
    );
}
