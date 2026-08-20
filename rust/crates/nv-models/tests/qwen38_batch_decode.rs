#![cfg(feature = "cuda")]

mod common;
use common::qwen38_snapshot_dir_env_override_then_home_hub;
use common::argmax_partial_cmp as argmax;
use common::envn;
use common::prompt_for;
use candle_core::{DType, Device, Tensor};
use nv_models::gemma4_batch_graph::BucketPlan;
use nv_models::qwen3_5_moe::qwen38_batch::Qwen38BatchLanes;
use nv_models::qwen3_5_moe::{LayerType, Qwen3Moe, Qwen3MoeKvCache, Qwen3_5DenseConfig};
use std::collections::HashMap;
use std::path::PathBuf;

mod ctx_timing_common;
use common::LcgOddSeedShift33SignedUnitRows as Lcg;

const QWEN38_27B_CONFIG_JSON: &str = include_str!("qwen3_8_27b_config.json");

const TINY_LAYERS_8_KEEPS_TWO_FULL_ATTENTION_SLOTS_OF_THE_INTERVAL_4_PATTERN: usize = 8;
const TINY_HIDDEN_128: usize = 128;
const TINY_HEAD_DIM_64_SMALLEST_CUDA_FP8_DECODE_TEMPLATE_ARM: usize = 64;
const TINY_Q_HEADS_12_KV_2_KEEP_THE_RELEASE_GQA_RATIO: (usize, usize) = (12, 2);
const TINY_GDN_V6_K2_KEEP_THE_RELEASE_V_OVER_K_RATIO_3: (usize, usize) = (6, 2);
const TINY_GDN_HEAD_DIM_32_THE_FUSED_GDN_DECODE_FLOOR_D_V_MULTIPLE_OF_32: usize = 32;
const TINY_INTER_192_VOCAB_512_MAX_POS_64: (usize, usize, usize) = (192, 512, 64);

const STEPS_8_MATCHES_THE_GEMMA4_BATCH_BIT_IDENTITY_SUITE: usize = 8;
const SLOTS_4_THE_CONCURRENCY_TARGET: usize = 4;

fn tiny_q38_config() -> Qwen3_5DenseConfig {
    let mut cfg = Qwen3_5DenseConfig::from_hf_json_str(QWEN38_27B_CONFIG_JSON)
        .expect("release config.json parses");
    cfg.layer_types
        .truncate(TINY_LAYERS_8_KEEPS_TWO_FULL_ATTENTION_SLOTS_OF_THE_INTERVAL_4_PATTERN);
    cfg.num_hidden_layers = TINY_LAYERS_8_KEEPS_TWO_FULL_ATTENTION_SLOTS_OF_THE_INTERVAL_4_PATTERN;
    for (i, t) in cfg.layer_types.iter().enumerate() {
        let expected = if (i + 1) % 4 == 0 {
            LayerType::FullAttention
        } else {
            LayerType::LinearAttention
        };
        assert_eq!(*t, expected, "layer {i}: interval-4 pattern must survive truncation");
    }
    cfg.hidden_size = TINY_HIDDEN_128;
    cfg.head_dim = TINY_HEAD_DIM_64_SMALLEST_CUDA_FP8_DECODE_TEMPLATE_ARM;
    let (n_q, n_kv) = TINY_Q_HEADS_12_KV_2_KEEP_THE_RELEASE_GQA_RATIO;
    cfg.num_attention_heads = n_q;
    cfg.num_key_value_heads = n_kv;
    let (gdn_v, gdn_k) = TINY_GDN_V6_K2_KEEP_THE_RELEASE_V_OVER_K_RATIO_3;
    cfg.linear_num_value_heads = gdn_v;
    cfg.linear_num_key_heads = gdn_k;
    cfg.linear_key_head_dim = TINY_GDN_HEAD_DIM_32_THE_FUSED_GDN_DECODE_FLOOR_D_V_MULTIPLE_OF_32;
    cfg.linear_value_head_dim = TINY_GDN_HEAD_DIM_32_THE_FUSED_GDN_DECODE_FLOOR_D_V_MULTIPLE_OF_32;
    let (inter, vocab, max_pos) = TINY_INTER_192_VOCAB_512_MAX_POS_64;
    cfg.intermediate_size = inter;
    cfg.vocab_size = vocab;
    cfg.max_position_embeddings = max_pos;
    cfg.bos_token_id = None;
    cfg.eos_token_id = 1;
    cfg
}

fn bf16_tensor(vals: &[f32], shape: &[usize]) -> Tensor {
    Tensor::from_vec(vals.to_vec(), shape, &Device::Cpu)
        .expect("cpu tensor")
        .to_dtype(DType::BF16)
        .expect("bf16 cast")
}

fn minus_one_because_the_loader_adds_one_to_zero_centered_norm_weights(v: &[f32]) -> Vec<f32> {
    v.iter().map(|x| x - 1.0).collect()
}

fn write_tiny_safetensors_dir(cfg: &Qwen3_5DenseConfig, seed: u64) -> PathBuf {
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
        bf16_tensor(
            &r.bf16_rounded_f32_vec(cfg.vocab_size * hidden, 0.6),
            &[cfg.vocab_size, hidden],
        ),
    );
    for i in 0..cfg.num_hidden_layers {
        let p = format!("model.language_model.layers.{i}");
        t.insert(
            format!("{p}.input_layernorm.weight"),
            bf16_tensor(
                &minus_one_because_the_loader_adds_one_to_zero_centered_norm_weights(
                    &r.norm_effective_vec_near_one(hidden),
                ),
                &[hidden],
            ),
        );
        t.insert(
            format!("{p}.post_attention_layernorm.weight"),
            bf16_tensor(
                &minus_one_because_the_loader_adds_one_to_zero_centered_norm_weights(
                    &r.norm_effective_vec_near_one(hidden),
                ),
                &[hidden],
            ),
        );
        match cfg.layer_types[i] {
            LayerType::LinearAttention => {
                let q = format!("{p}.linear_attn");
                t.insert(
                    format!("{q}.in_proj_qkv.weight"),
                    bf16_tensor(&r.bf16_rounded_f32_vec(conv_dim * hidden, 0.12), &[conv_dim, hidden]),
                );
                t.insert(
                    format!("{q}.in_proj_z.weight"),
                    bf16_tensor(&r.bf16_rounded_f32_vec(value_dim * hidden, 0.12), &[value_dim, hidden]),
                );
                t.insert(
                    format!("{q}.in_proj_a.weight"),
                    bf16_tensor(&r.bf16_rounded_f32_vec(n_v * hidden, 0.12), &[n_v, hidden]),
                );
                t.insert(
                    format!("{q}.in_proj_b.weight"),
                    bf16_tensor(&r.bf16_rounded_f32_vec(n_v * hidden, 0.12), &[n_v, hidden]),
                );
                t.insert(
                    format!("{q}.conv1d.weight"),
                    bf16_tensor(&r.bf16_rounded_f32_vec(conv_dim * ks, 0.4), &[conv_dim, 1, ks]),
                );
                t.insert(
                    format!("{q}.A_log"),
                    bf16_tensor(&r.bf16_rounded_f32_vec(n_v, 0.5), &[n_v]),
                );
                t.insert(
                    format!("{q}.dt_bias"),
                    bf16_tensor(&r.bf16_rounded_f32_vec(n_v, 0.5), &[n_v]),
                );
                t.insert(
                    format!("{q}.norm.weight"),
                    bf16_tensor(
                        &r.norm_effective_vec_near_one(cfg.linear_value_head_dim),
                        &[cfg.linear_value_head_dim],
                    ),
                );
                t.insert(
                    format!("{q}.out_proj.weight"),
                    bf16_tensor(&r.bf16_rounded_f32_vec(hidden * value_dim, 0.12), &[hidden, value_dim]),
                );
            }
            LayerType::FullAttention => {
                let a = format!("{p}.self_attn");
                let q_out = cfg.num_attention_heads * hd * 2;
                assert!(cfg.attn_output_gate, "release config has attn_output_gate");
                let kv_out = cfg.num_key_value_heads * hd;
                t.insert(
                    format!("{a}.q_proj.weight"),
                    bf16_tensor(&r.bf16_rounded_f32_vec(q_out * hidden, 0.12), &[q_out, hidden]),
                );
                t.insert(
                    format!("{a}.k_proj.weight"),
                    bf16_tensor(&r.bf16_rounded_f32_vec(kv_out * hidden, 0.12), &[kv_out, hidden]),
                );
                t.insert(
                    format!("{a}.v_proj.weight"),
                    bf16_tensor(&r.bf16_rounded_f32_vec(kv_out * hidden, 0.12), &[kv_out, hidden]),
                );
                t.insert(
                    format!("{a}.o_proj.weight"),
                    bf16_tensor(
                        &r.bf16_rounded_f32_vec(hidden * cfg.num_attention_heads * hd, 0.12),
                        &[hidden, cfg.num_attention_heads * hd],
                    ),
                );
                t.insert(
                    format!("{a}.q_norm.weight"),
                    bf16_tensor(
                        &minus_one_because_the_loader_adds_one_to_zero_centered_norm_weights(
                            &r.norm_effective_vec_near_one(hd),
                        ),
                        &[hd],
                    ),
                );
                t.insert(
                    format!("{a}.k_norm.weight"),
                    bf16_tensor(
                        &minus_one_because_the_loader_adds_one_to_zero_centered_norm_weights(
                            &r.norm_effective_vec_near_one(hd),
                        ),
                        &[hd],
                    ),
                );
            }
        }
        t.insert(
            format!("{p}.mlp.gate_proj.weight"),
            bf16_tensor(&r.bf16_rounded_f32_vec(inter * hidden, 0.15), &[inter, hidden]),
        );
        t.insert(
            format!("{p}.mlp.up_proj.weight"),
            bf16_tensor(&r.bf16_rounded_f32_vec(inter * hidden, 0.15), &[inter, hidden]),
        );
        t.insert(
            format!("{p}.mlp.down_proj.weight"),
            bf16_tensor(&r.bf16_rounded_f32_vec(hidden * inter, 0.15), &[hidden, inter]),
        );
    }
    t.insert(
        "model.language_model.norm.weight".into(),
        bf16_tensor(
            &minus_one_because_the_loader_adds_one_to_zero_centered_norm_weights(
                &r.norm_effective_vec_near_one(cfg.hidden_size),
            ),
            &[cfg.hidden_size],
        ),
    );
    t.insert(
        "lm_head.weight".into(),
        bf16_tensor(
            &r.bf16_rounded_f32_vec(cfg.vocab_size * cfg.hidden_size, 0.2),
            &[cfg.vocab_size, cfg.hidden_size],
        ),
    );
    let dir = std::env::temp_dir().join(format!("q38-batch-tiny-{}", std::process::id()));
    std::fs::create_dir_all(&dir).expect("mk temp safetensors dir");
    candle_core::safetensors::save(&t, dir.join("model.safetensors")).expect("save tiny model");
    dir
}

fn bits(v: &[f32]) -> Vec<u32> {
    v.iter().map(|x| x.to_bits()).collect()
}

fn distinct(b: &[u32]) -> usize {
    b.iter().collect::<std::collections::HashSet<_>>().len()
}

fn solo_stream(
    model: &Qwen3Moe,
    cache: &mut Qwen3MoeKvCache,
    device: &Device,
    prompt: &[u32],
    steps: usize,
) -> (Vec<u32>, Vec<Vec<u32>>) {
    cache.reset();
    let (last, rest) = prompt.split_last().expect("prompt");
    let seq = rest.len();
    let tokens = Tensor::from_vec(rest.to_vec(), (1usize, seq), device).expect("tokens");
    let positions =
        Tensor::from_vec((0..seq as i32).collect::<Vec<_>>(), seq, device).expect("pos");
    model
        .forward_with_cache_dispatched_rows(&tokens, &positions, cache, None, Some(1))
        .expect("solo prefill");
    let mut toks = Vec::with_capacity(steps);
    let mut lg_bits = Vec::with_capacity(steps);
    let mut t = *last;
    for _ in 0..steps {
        let pos = cache.current_len();
        let tokens = Tensor::from_vec(vec![t], (1usize, 1usize), device).expect("token");
        let positions = Tensor::from_vec(vec![pos as i32], 1usize, device).expect("pos");
        let logits = model
            .forward_with_cache_dispatched(&tokens, &positions, cache, None)
            .expect("solo decode step");
        let row: Vec<f32> = logits
            .to_dtype(DType::F32)
            .expect("f32")
            .flatten_all()
            .expect("flat")
            .to_vec1()
            .expect("host");
        let n = argmax(&row);
        lg_bits.push(bits(&row));
        toks.push(n);
        t = n;
    }
    (toks, lg_bits)
}

const SYNTHETIC_BF16_WORST_REL_TOL_2_5E2_THE_M_ROW_TC_GEMM_IS_NOT_THE_GEMV_ROW_TWIN_ONLY_THE_QUANTIZED_MK_ARMS_ARE:
    f32 = 2.5e-2;

enum RowBar {
    BitIdentical,
    WorstRelWithin(f32),
}

fn diag_env_nv_q38_batch_diag_1_prints_instead_of_panicking() -> bool {
    std::env::var("NV_Q38_BATCH_DIAG").ok().as_deref() == Some("1")
}

fn assert_rows_match(got: &[f32], want_bits: &[u32], bar: &RowBar, ctx: &str) {
    let got_bits = bits(got);
    let diff = got_bits
        .iter()
        .zip(want_bits.iter())
        .filter(|(a, b)| a != b)
        .count();
    let scale = want_bits
        .iter()
        .fold(0f32, |a, b| a.max(f32::from_bits(*b).abs()))
        .max(1e-6);
    let worst = got_bits
        .iter()
        .zip(want_bits.iter())
        .map(|(a, b)| (f32::from_bits(*a) - f32::from_bits(*b)).abs())
        .fold(0.0f32, f32::max);
    if diag_env_nv_q38_batch_diag_1_prints_instead_of_panicking() {
        eprintln!(
            "[q38-batch-diag] {ctx}: {diff} of {} lanes differ, max |delta| {worst:.3e}, worst rel {:.3e}",
            want_bits.len(),
            worst / scale
        );
        return;
    }
    match bar {
        RowBar::BitIdentical => assert_eq!(
            diff,
            0,
            "{ctx}: {diff} of {} logit lanes differ from the solo run (max |delta| {worst:.3e})",
            want_bits.len()
        ),
        RowBar::WorstRelWithin(tol) => assert!(
            worst / scale < *tol,
            "{ctx}: worst rel {:.3e} exceeds {tol:.1e} vs the solo run ({diff} lanes differ)",
            worst / scale
        ),
    }
}

fn run_batch_parity_case(
    model: Qwen3Moe,
    device: &Device,
    vocab: usize,
    max_seq: usize,
    multi_row_bar: RowBar,
) {
    let slots = SLOTS_4_THE_CONCURRENCY_TARGET;
    let steps = STEPS_8_MATCHES_THE_GEMMA4_BATCH_BIT_IDENTITY_SUITE;
    let plan = BucketPlan::new(vec![1, 2, 4]);
    let mut lanes =
        Qwen38BatchLanes::new(model, device, max_seq, plan).expect("build batch lanes");
    assert_eq!(lanes.lanes(), slots);

    let prompts: Vec<Vec<u32>> = (0..slots)
        .map(|j| prompt_for(j, 9 + 3 * j, vocab))
        .collect();

    let mut solo_caches: Vec<Qwen3MoeKvCache> = (0..slots)
        .map(|_| {
            let mut c = lanes.model().new_kv_cache(max_seq).expect("solo cache");
            c.set_fused_lin_attn(true);
            c
        })
        .collect();
    let solo: Vec<(Vec<u32>, Vec<Vec<u32>>)> = (0..slots)
        .map(|j| solo_stream(lanes.model(), &mut solo_caches[j], device, &prompts[j], steps))
        .collect();
    drop(solo_caches);
    let d = distinct(&solo[0].1[0]);
    let distinct_floor_16_bf16_logits_at_tiny_hidden_round_to_few_values = 16;
    assert!(
        d > distinct_floor_16_bf16_logits_at_tiny_hidden_round_to_few_values,
        "solo logits are degenerate ({d} distinct of {vocab}); the bit-compare would be vacuous"
    );
    for j in 1..slots {
        assert!(
            (0..steps).any(|i| solo[j].1[i] != solo[0].1[i]),
            "slot {j}'s solo logits equal slot 0's at every step; the cross-lane compare would be vacuous"
        );
    }

    let mut cur: Vec<u32> = Vec::with_capacity(slots);
    for (j, p) in prompts.iter().enumerate() {
        lanes
            .prefill_lane(j, &p[..p.len() - 1])
            .expect("prefill lane");
        cur.push(*p.last().unwrap());
    }
    for i in 0..steps {
        let step: Vec<Option<u32>> = cur.iter().map(|&t| Some(t)).collect();
        let out = lanes.step_batch(&step).expect("batch step");
        for j in 0..slots {
            let row = out[j].as_ref().expect("active lane row");
            assert_rows_match(
                row,
                &solo[j].1[i],
                &multi_row_bar,
                &format!("B={slots} step {i} lane {j}"),
            );
            let n = argmax(row);
            if !diag_env_nv_q38_batch_diag_1_prints_instead_of_panicking() {
                assert_eq!(n, solo[j].0[i], "step {i} lane {j}: sampled token differs");
            }
            cur[j] = solo[j].0[i];
            let _ = n;
        }
    }
    eprintln!(
        "[q38-batch] B={slots}: {steps} steps x {vocab} lanes match solo at the case bar \
         (captures={} replays={} nodes={})",
        lanes.captures(),
        lanes.replays(),
        lanes.captured_node_count()
    );

    let narrow = slots - 1;
    let mut cur: Vec<u32> = Vec::with_capacity(narrow);
    for j in 0..narrow {
        let p = &prompts[j];
        lanes
            .prefill_lane(j, &p[..p.len() - 1])
            .expect("re-prefill lane");
        cur.push(*p.last().unwrap());
    }
    for i in 0..steps {
        let mut step: Vec<Option<u32>> = cur.iter().map(|&t| Some(t)).collect();
        step.push(None);
        let out = lanes.step_batch(&step).expect("narrow batch step");
        assert!(out[narrow].is_none(), "padded lane must return no row");
        for j in 0..narrow {
            let row = out[j].as_ref().expect("active lane row");
            assert_rows_match(
                row,
                &solo[j].1[i],
                &multi_row_bar,
                &format!("B={narrow}-on-{slots} step {i} lane {j}"),
            );
            if !diag_env_nv_q38_batch_diag_1_prints_instead_of_panicking() {
                assert_eq!(argmax(row), solo[j].0[i], "narrow step {i} lane {j}: token differs");
            }
            cur[j] = solo[j].0[i];
        }
    }
    eprintln!("[q38-batch] B={narrow} padded on a {slots}-row graph: still matches solo");

    lanes
        .prefill_lane(0, &prompts[0][..prompts[0].len() - 1])
        .expect("re-prefill lane 0");
    let mut t = *prompts[0].last().unwrap();
    for i in 0..steps {
        let out = lanes.step_batch(&[Some(t)]).expect("b1 batch step");
        let row = out[0].as_ref().expect("lane 0 row");
        assert_rows_match(row, &solo[0].1[i], &RowBar::BitIdentical, &format!("B=1 step {i}"));
        t = solo[0].0[i];
        let _ = argmax(row);
    }
    eprintln!("[q38-batch] B=1 bucket: bit-identical to solo (leading-1 routes are shared)");
    assert!(
        lanes.captures() >= 2,
        "expected one capture per touched bucket (4 then 1); captures={}",
        lanes.captures()
    );
    assert!(
        lanes.replays() > 0,
        "batch steps never replayed a captured graph; the graph arm is vacuous"
    );
}

#[test]
fn qwen38_tiny_batch_decode_matches_solo() {
    if std::env::var("NV_Q38_BATCH_TEST").as_deref() != Ok("1") {
        panic!("set NV_Q38_BATCH_TEST=1 to run this GPU test (it must never silently skip)");
    }
    let device = Device::new_cuda_with_stream(0).expect("cuda with stream");
    let cfg = tiny_q38_config();
    let vocab = cfg.vocab_size;
    let dir = write_tiny_safetensors_dir(&cfg, 0x38b0_0001);
    let weights = nv_weights::WeightLoader::open_dir(&dir, &device).expect("open tiny dir");
    let model = Qwen3Moe::from_loader_dense(cfg, &weights, &device).expect("build tiny model");
    drop(weights);
    run_batch_parity_case(
        model,
        &device,
        vocab,
        48,
        RowBar::WorstRelWithin(
            SYNTHETIC_BF16_WORST_REL_TOL_2_5E2_THE_M_ROW_TC_GEMM_IS_NOT_THE_GEMV_ROW_TWIN_ONLY_THE_QUANTIZED_MK_ARMS_ARE,
        ),
    );
    let _ = std::fs::remove_dir_all(&dir);
}

fn load_real_qwen38(device: &Device) -> Qwen3Moe {
    let dir = qwen38_snapshot_dir_env_override_then_home_hub();
    let raw_cfg = std::fs::read_to_string(dir.join("config.json")).expect("config.json");
    let cfg = Qwen3_5DenseConfig::from_hf_json_str(&raw_cfg).expect("parse dense config");
    let qcfg = nv_weights::QuantizationConfig::from_hf_json_str(&raw_cfg).expect("parse qcfg");
    let weights = nv_weights::WeightLoader::open_dir(&dir, device).expect("weights");
    let model = Qwen3Moe::from_loader_dense_quantized(cfg, &weights, &qcfg, device)
        .expect("build Qwen3.8-27B dense arm");
    assert!(model.is_dense(), "quantized dense loader must yield the dense arm");
    model
}

#[test]
#[ignore = "loads the Qwen3.8-27B NVFP4 checkpoint; set NV_Q38_TEST=1"]
fn real_qwen38_27b_batch_decode_bit_identity() {
    if std::env::var("NV_Q38_TEST").as_deref() != Ok("1") {
        panic!("set NV_Q38_TEST=1 to run this GPU test (it must never silently skip)");
    }
    let _one_gpu_test_at_a_time = ctx_timing_common::serialize_tests_sharing_the_gpu_because_libtest_threads_double_book_the_card_reading_2x_slow_and_a_sibling_engines_teardown_surfaces_a_one_shot_cuda_error_invalid_value();
    if std::env::var("NV_Q38_BATCH_ROWWISE").is_err() {
        std::env::set_var("NV_Q38_BATCH_ROWWISE", "ffn");
    }
    eprintln!(
        "[q38-batch] bit-identity gate arm: NV_Q38_BATCH_ROWWISE={} (ffn = per-row m=1 MLP, the \
         nvfp4 m-row route is the sole non-twin; the serving default keeps the batched MLP and \
         holds argmax-equality at ~2e-2 worst-rel)",
        std::env::var("NV_Q38_BATCH_ROWWISE").unwrap()
    );
    let device = Device::new_cuda_with_stream(0).expect("cuda with stream");
    let model = load_real_qwen38(&device);
    let vocab = model.config().vocab_size;
    let max_seq = envn("NV_Q38_BATCH_MAXSEQ", 128);
    run_batch_parity_case(model, &device, vocab, max_seq, RowBar::BitIdentical);
    std::env::remove_var("NV_Q38_BATCH_ROWWISE");
}

#[test]
#[ignore = "loads the Qwen3.8-27B NVFP4 checkpoint; set NV_Q38_RATE=1 -- B-ladder: per-lane ms/tok and aggregate tok/s at B=1/2/4, depth via NV_CTX_TOKENS (default 256), synthetic KV prime"]
fn real_qwen38_27b_batch_decode_rate_ladder() {
    if std::env::var("NV_Q38_RATE").as_deref() != Ok("1") {
        panic!("set NV_Q38_RATE=1 to run this GPU test (it must never silently skip)");
    }
    let _one_gpu_test_at_a_time = ctx_timing_common::serialize_tests_sharing_the_gpu_because_libtest_threads_double_book_the_card_reading_2x_slow_and_a_sibling_engines_teardown_surfaces_a_one_shot_cuda_error_invalid_value();
    let device = Device::new_cuda_with_stream(0).expect("cuda with stream");
    let model = load_real_qwen38(&device);
    let vocab = model.config().vocab_size;
    let n_kv = model.config().num_key_value_heads;
    let hd = model.config().head_dim;
    let depth = {
        let s = std::env::var("NV_CTX_TOKENS").unwrap_or_else(|_| "256".into());
        let s = s.trim();
        match s.strip_suffix('k') {
            Some(n) => n.parse::<usize>().expect("NV_CTX_TOKENS") * 1024,
            None => s.parse::<usize>().expect("NV_CTX_TOKENS"),
        }
    };
    let steps = envn("NV_Q38_RATE_STEPS", 16);
    let reps = envn("NV_Q38_RATE_REPS", 3);
    let max_seq = depth + 3 * (reps + 1) * steps + 64;
    let plan = BucketPlan::new(vec![1, 2, 4]);
    let mut lanes =
        Qwen38BatchLanes::new(model, &device, max_seq, plan).expect("build batch lanes");

    let prefill_len = 23usize;
    let chunk = 512usize
        .min(depth.saturating_sub(prefill_len))
        .max(1);
    let mut state = 0x9e3779b97f4a7c15u64;
    let vals: Vec<f32> = (0..chunk * n_kv * hd)
        .map(|_| {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (((state >> 40) as f32) / (1u64 << 24) as f32 - 0.5) * 0.5
        })
        .collect();
    let k_t = Tensor::from_vec(vals.clone(), (1usize, chunk, n_kv, hd), &device)
        .expect("k template")
        .to_dtype(DType::BF16)
        .expect("bf16");
    let v_t = Tensor::from_vec(vals, (1usize, chunk, n_kv, hd), &device)
        .expect("v template")
        .to_dtype(DType::BF16)
        .expect("bf16");

    let prompt = prompt_for(0, prefill_len + 1, vocab);
    for lane in 0..lanes.lanes() {
        lanes
            .prefill_lane(lane, &prompt[..prompt.len() - 1])
            .expect("prefill lane");
        while lanes.lane_pos(lane) + chunk <= depth {
            lanes
                .prime_lane_kv_depth_synthetically_for_ctx_timing_decode_reads_cache_size_not_values(
                    lane, &k_t, &v_t,
                )
                .expect("prime lane");
        }
    }
    eprintln!(
        "[q38-batch-rate] primed {} lanes to pos {} (target depth {depth})",
        lanes.lanes(),
        lanes.lane_pos(0)
    );

    let seed_tok = *prompt.last().unwrap();
    let mut results: Vec<(usize, f64)> = Vec::new();
    for &bsz in &[1usize, 2, 4] {
        let mut cur: Vec<Option<u32>> = (0..bsz).map(|_| Some(seed_tok)).collect();
        let mut ms_acc: Vec<f64> = Vec::new();
        for r in 0..=reps {
            let t0 = std::time::Instant::now();
            for _ in 0..steps {
                let out = lanes.step_batch(&cur).expect("rate step");
                for (j, o) in out.iter().enumerate() {
                    cur[j] = Some(argmax(o.as_ref().unwrap()));
                }
            }
            lanes.synchronize().expect("sync");
            let ms = t0.elapsed().as_secs_f64() * 1000.0 / steps as f64;
            if r == 0 {
                eprintln!("[q38-batch-rate] B={bsz} warmup rep discarded: {ms:.2} ms/step");
                continue;
            }
            ms_acc.push(ms);
        }
        let mean = ms_acc.iter().sum::<f64>() / ms_acc.len() as f64;
        results.push((bsz, mean));
        eprintln!(
            "[q38-batch-rate] B={bsz} depth={} step {mean:.2} ms = {:.2} ms/lane-tok | aggregate {:.1} tok/s (captures={} replays={})",
            lanes.lane_pos(0),
            mean / bsz as f64,
            bsz as f64 * 1000.0 / mean,
            lanes.captures(),
            lanes.replays()
        );
    }
    let b1 = results
        .iter()
        .find(|(b, _)| *b == 1)
        .map(|(_, m)| *m)
        .unwrap();
    for (bsz, ms) in &results {
        eprintln!(
            "[q38-batch-rate] SUMMARY B={bsz}: {:.2} ms/step, {:.2}x aggregate vs B=1, {:.2}x per-lane latency, basis: unsloth/Qwen3.8-27B-NVFP4, synthetic prime depth {}, {} timed steps x {} reps",
            ms,
            b1 * *bsz as f64 / ms,
            ms / b1,
            lanes.lane_pos(0),
            steps,
            reps
        );
    }
}

#[test]
#[ignore = "row-purity oracle for the nvfp4 gemm: same row through leading=1 and leading=4; set NV_Q38_TEST=1"]
fn real_qwen38_nvfp4_linear_m_row_purity_oracle() {
    if std::env::var("NV_Q38_TEST").as_deref() != Ok("1") {
        panic!("set NV_Q38_TEST=1 to run this GPU test (it must never silently skip)");
    }
    use std::sync::{Arc, Mutex};
    let dir = qwen38_snapshot_dir_env_override_then_home_hub();
    let device = Device::new_cuda_with_stream(0).expect("cuda");
    let weights = nv_weights::WeightLoader::open_dir(&dir, &device).expect("weights");
    let module = "model.language_model.layers.0.mlp.gate_proj";
    let dev = match &device {
        Device::Cuda(d) => d.clone(),
        _ => unreachable!(),
    };
    let (n, k) = {
        let t = weights
            .get(&format!("{module}.weight_packed"), DType::U8)
            .expect("packed tensor");
        let d2 = t.dims().to_vec();
        (d2[0], d2[1] * 2)
    };
    let runner = Arc::new(Mutex::new(
        nv_quant::nvfp4::Nvfp4GemmRunner::new(dev.cuda_stream()).expect("nvfp4 runner"),
    ));
    let native = nv_layers::moe::nvfp4_linear_from_disk_pub(&weights, module, n, k, runner.clone(), &device)
        .expect("native nvfp4 linear");
    let up = nv_layers::moe::nvfp4_linear_from_disk_pub(
        &weights,
        "model.language_model.layers.0.mlp.up_proj",
        n,
        k,
        runner.clone(),
        &device,
    )
    .expect("up");
    let down = nv_layers::moe::nvfp4_linear_from_disk_pub(
        &weights,
        "model.language_model.layers.0.mlp.down_proj",
        k,
        n,
        runner,
        &device,
    )
    .expect("down");
    let mut r = Lcg::new(0x38b0_0ac1e | 1);
    let x_rows: Vec<Vec<f32>> = (0..4).map(|_| r.bf16_rounded_f32_vec(k, 0.4)).collect();
    let flat: Vec<f32> = x_rows.iter().flatten().copied().collect();
    let x4 = Tensor::from_vec(flat, (1usize, 4usize, k), &device)
        .expect("x4")
        .to_dtype(DType::BF16)
        .expect("bf16");
    let y4 = native.forward(&x4).expect("m=4 forward");
    let y4h: Vec<f32> = y4
        .to_dtype(DType::F32)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1()
        .unwrap();
    for i in 0..4 {
        let x1 = Tensor::from_vec(x_rows[i].clone(), (1usize, 1usize, k), &device)
            .expect("x1")
            .to_dtype(DType::BF16)
            .expect("bf16");
        let y1 = native.forward(&x1).expect("m=1 forward");
        let y1h: Vec<f32> = y1
            .to_dtype(DType::F32)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1()
            .unwrap();
        let row4 = &y4h[i * n..(i + 1) * n];
        let diff = row4
            .iter()
            .zip(y1h.iter())
            .filter(|(a, b)| a.to_bits() != b.to_bits())
            .count();
        let worst = row4
            .iter()
            .zip(y1h.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        eprintln!(
            "[nvfp4-purity] row {i}: {diff} of {n} outputs differ between leading=1 and leading=4 (max |delta| {worst:.3e})"
        );
    }

    let gate2 = nv_layers::moe::nvfp4_linear_from_disk_pub(
        &weights,
        module,
        n,
        k,
        Arc::new(Mutex::new(
            nv_quant::nvfp4::Nvfp4GemmRunner::new(dev.cuda_stream()).expect("nvfp4 runner"),
        )),
        &device,
    )
    .expect("gate2");
    let mlp = nv_layers::mlp::Mlp::new(gate2, up, down).expect("mlp");
    let flat: Vec<f32> = x_rows.iter().flatten().copied().collect();
    let x4 = Tensor::from_vec(flat, (1usize, 4usize, k), &device)
        .expect("x4")
        .to_dtype(DType::BF16)
        .expect("bf16");
    let y4 = mlp.forward_fused_cuda(&x4).expect("mlp m=4");
    let y4h: Vec<f32> = y4
        .to_dtype(DType::F32)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1()
        .unwrap();
    for i in 0..4 {
        let x1 = Tensor::from_vec(x_rows[i].clone(), (1usize, 1usize, k), &device)
            .expect("x1")
            .to_dtype(DType::BF16)
            .expect("bf16");
        let y1 = mlp.forward_fused_cuda(&x1).expect("mlp m=1");
        let y1h: Vec<f32> = y1
            .to_dtype(DType::F32)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1()
            .unwrap();
        let row4 = &y4h[i * k..(i + 1) * k];
        let diff = row4
            .iter()
            .zip(y1h.iter())
            .filter(|(a, b)| a.to_bits() != b.to_bits())
            .count();
        let worst = row4
            .iter()
            .zip(y1h.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        eprintln!(
            "[nvfp4-purity] MLP composite row {i}: {diff} of {k} outputs differ (max |delta| {worst:.3e})"
        );
    }
}
