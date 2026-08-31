#![cfg(feature = "cuda")]

mod common;
use common::argmax;
use common::qwen38_snapshot_dir_env_override_then_home_hub;
use candle_core::{DType, Device, Tensor};
use nv_models::qwen3_5_moe::{
    Qwen3Moe, Qwen3MoeKvCache, Qwen3_5DenseConfig,
    M1_DECODE_KVSHARE_ENV_1_ALWAYS_GATE_AT_THE_MEASURED_KNEE_0_OR_UNSET_FOR_THE_FUSED_ARM,
};
use nv_weights::{QuantizationConfig, WeightLoader};

mod ctx_timing_common;

const SYNTHETIC_FILL_CHUNK_512_MATCHES_THE_PROVEN_QWEN35_DENSE_PREFILL_BLOCK: usize = 512;
const DEEP_DEPTH_DEFAULT_32768_PAST_THE_16384_KNEE_SO_THE_GATE_MODE_WOULD_ALSO_ENGAGE_HERE: usize =
    32768;
const GREEDY_STEPS_64_MATCHING_THE_SERVING_LADDERS_TIMED_WINDOW: usize = 64;
const KV_SLOT_HEADROOM_16_BEYOND_THE_GREEDY_STEPS: usize = 16;
const ADAPTIVE_TIE_MARGIN_MAX_1_BF16_LOGIT_UNIT_MATCHING_THE_QWEN38_CROSS_SHAPE_TIE_CLASS_BAR:
    f32 = 1.0;

fn fixed_seed_values_because_an_all_zero_cache_is_an_unrealistically_easy_softmax(
    len: usize,
) -> Vec<f32> {
    let mut state = 0x9e3779b97f4a7c15u64;
    (0..len)
        .map(|_| {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (((state >> 40) as f32) / (1u64 << 24) as f32 - 0.5) * 0.5
        })
        .collect()
}

fn depth_from_env_single_entry_default_deep() -> usize {
    match std::env::var("NV_CTX_TOKENS") {
        Ok(v) => {
            let entries: Vec<usize> = v
                .split(',')
                .map(|s| {
                    let s = s.trim();
                    let (num, mult) = match s.strip_suffix('k') {
                        Some(n) => (n, 1024usize),
                        None => (s, 1usize),
                    };
                    num.parse::<usize>().expect("NV_CTX_TOKENS entry") * mult
                })
                .collect();
            assert_eq!(
                entries.len(),
                1,
                "this parity gate primes its synthetic fill twice per depth; run ONE depth per \
                 process via NV_CTX_TOKENS (the #96 residual race convention)"
            );
            entries[0]
        }
        Err(_) => DEEP_DEPTH_DEFAULT_32768_PAST_THE_16384_KNEE_SO_THE_GATE_MODE_WOULD_ALSO_ENGAGE_HERE,
    }
}

fn argmax_top1_minus_top2_margin(row: &[f32]) -> f32 {
    let mut top1 = f32::NEG_INFINITY;
    let mut top2 = f32::NEG_INFINITY;
    for &x in row {
        if !x.is_finite() {
            continue;
        }
        if x > top1 {
            top2 = top1;
            top1 = x;
        } else if x > top2 {
            top2 = x;
        }
    }
    top1 - top2
}

#[test]
#[ignore = "loads the ~22.6 GB Qwen3.8-27B NVFP4 dense arm; set NV_QWEN38_SERVING_TEST=1 -- the \
            m=1 decode kvshare parity gate: on one process and the same synthetic fp8 KV fill, \
            the NV_Q38_DECODE_KVSHARE kvshare arm must match the fused splitk arm within the \
            bf16-rounding band (both accumulate f32 online-softmax over the same fp8 rows), and \
            a 64-step greedy decode stream per arm must emit identical tokens or diverge only \
            tie-class (argmax margin inside the 1-bf16-logit-unit band at the first split); \
            NV_CTX_TOKENS overrides the default 32768 depth, one depth per process"]
fn qwen38_m1_decode_kvshare_matches_fused_within_bf16_rounding_and_greedy_streams_split_only_tie_class(
) {
    if std::env::var("NV_QWEN38_SERVING_TEST").ok().as_deref() != Some("1") {
        if std::env::var("NV_MODELS_ALLOW_SKIP").as_deref() != Ok("1") {
            panic!(
                "set NV_QWEN38_SERVING_TEST=1 to run this gate, or NV_MODELS_ALLOW_SKIP=1 to \
                 skip it on purpose; a kvshare parity gate that silently reports ok would let \
                 the arm ship on routing coverage alone"
            );
        }
        eprintln!("SKIP (NV_MODELS_ALLOW_SKIP=1): set NV_QWEN38_SERVING_TEST=1 to run");
        return;
    }
    let _one_gpu_test_at_a_time = ctx_timing_common::serialize_tests_sharing_the_gpu_because_libtest_threads_double_book_the_card_reading_2x_slow_and_a_sibling_engines_teardown_surfaces_a_one_shot_cuda_error_invalid_value();
    std::env::set_var("NV_Q36_GRAPHED_DECODE_FIX", "1");
    std::env::set_var(
        M1_DECODE_KVSHARE_ENV_1_ALWAYS_GATE_AT_THE_MEASURED_KNEE_0_OR_UNSET_FOR_THE_FUSED_ARM,
        "1",
    );

    let dir = qwen38_snapshot_dir_env_override_then_home_hub();
    let device = Device::new_cuda_with_stream(0).expect("cuda");
    let raw_cfg = std::fs::read_to_string(dir.join("config.json")).expect("config.json");
    let cfg = Qwen3_5DenseConfig::from_hf_json_str(&raw_cfg).expect("parse dense config");
    let qcfg = QuantizationConfig::from_hf_json_str(&raw_cfg).expect("parse qcfg");
    let weights = WeightLoader::open_dir(&dir, &device).expect("weights");
    let model = Qwen3Moe::from_loader_dense_quantized(cfg, &weights, &qcfg, &device)
        .expect("build Qwen3.8-27B on the eager cuda dense arm for the kvshare parity gate");
    drop(weights);

    let depth = depth_from_env_single_entry_default_deep();
    let slots = depth
        + GREEDY_STEPS_64_MATCHING_THE_SERVING_LADDERS_TIMED_WINDOW
        + KV_SLOT_HEADROOM_16_BEYOND_THE_GREEDY_STEPS;
    let n_q = model.config().num_attention_heads;
    let n_kv = model.config().num_key_value_heads;
    let hd = model.config().head_dim;
    let chunk = SYNTHETIC_FILL_CHUNK_512_MATCHES_THE_PROVEN_QWEN35_DENSE_PREFILL_BLOCK;
    let vals = fixed_seed_values_because_an_all_zero_cache_is_an_unrealistically_easy_softmax(
        chunk * n_kv * hd,
    );
    let k_template = Tensor::from_vec(vals.clone(), (1usize, chunk, n_kv, hd), &device)
        .expect("k template")
        .to_dtype(DType::BF16)
        .expect("k bf16");
    let v_template = Tensor::from_vec(vals, (1usize, chunk, n_kv, hd), &device)
        .expect("v template")
        .to_dtype(DType::BF16)
        .expect("v bf16");
    let prime = |cache: &mut Qwen3MoeKvCache| {
        let mut pos = 0usize;
        while pos < depth {
            let n = chunk.min(depth - pos);
            let (kn, vn) = if n == chunk {
                (k_template.clone(), v_template.clone())
            } else {
                (
                    k_template.narrow(1, 0, n).expect("k tail"),
                    v_template.narrow(1, 0, n).expect("v tail"),
                )
            };
            cache
                .write_synthetic_rows_at_every_full_attention_slot_for_depth_timing_decode_reads_cache_size_not_values(
                    pos, &kn, &vn,
                )
                .unwrap_or_else(|e| panic!("synthetic fill chunk at pos {pos}: {e:#}"));
            pos += n;
        }
        device.synchronize().expect("sync after synthetic fill");
        assert_eq!(
            cache.current_len(),
            depth,
            "synthetic fill must advance current_len exactly the way prefill would"
        );
    };

    let mut cache_a = model.new_kv_cache(slots).expect("kv cache A");
    prime(&mut cache_a);

    let q_vals = fixed_seed_values_because_an_all_zero_cache_is_an_unrealistically_easy_softmax(
        n_q * hd,
    );
    let q = Tensor::from_vec(q_vals, (1usize, 1usize, n_q, hd), &device)
        .expect("q")
        .to_dtype(DType::BF16)
        .expect("q bf16");
    let scaling = 1.0 / (hd as f32).sqrt();
    let attn_arm = |cache: &mut Qwen3MoeKvCache, arm: &str| -> Vec<f32> {
        std::env::set_var(
            M1_DECODE_KVSHARE_ENV_1_ALWAYS_GATE_AT_THE_MEASURED_KNEE_0_OR_UNSET_FOR_THE_FUSED_ARM,
            arm,
        );
        let out = cache
            .decode_attention_fp8(0, &q, n_q, scaling)
            .unwrap_or_else(|e| panic!("decode attention arm={arm}: {e:#}"));
        out.to_dtype(DType::F32)
            .expect("f32")
            .flatten_all()
            .expect("flatten")
            .to_vec1::<f32>()
            .expect("to host")
    };
    let fused = attn_arm(&mut cache_a, "0");
    let kvshare = attn_arm(&mut cache_a, "1");
    assert!(
        fused.iter().all(|x| x.is_finite()) && kvshare.iter().all(|x| x.is_finite()),
        "non-finite m=1 decode attention output at depth {depth} on a pure arm"
    );
    assert!(
        fused != kvshare,
        "fused and kvshare produced bitwise-identical outputs at depth {depth}; the kvshare arm \
         splits the scan differently by construction, so bitwise identity means the env never \
         engaged and this gate measured one arm twice"
    );
    let max_mag_fused = fused.iter().map(|a| a.abs()).fold(0f32, f32::max);
    let max_diff = fused
        .iter()
        .zip(&kvshare)
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max);
    assert!(
        max_diff <= 1e-3 + 0.02 * max_mag_fused,
        "kvshare diverged beyond bf16-rounding from fused at depth {depth}: \
         max|diff|={max_diff:.3e} vs max|fused|={max_mag_fused:.3e}; both accumulate f32 \
         online-softmax over the same fp8 rows so disagreement beyond reordering noise means \
         an indexing bug, not numerics"
    );
    eprintln!(
        "KVSHARE-PARITY qwen38 depth={depth} attn_max_diff={max_diff:.3e} \
         attn_max_mag_fused={max_mag_fused:.3e} band={:.3e}",
        1e-3 + 0.02 * max_mag_fused
    );

    let greedy_stream = |cache: &mut Qwen3MoeKvCache, arm: &str| -> (Vec<u32>, Vec<f32>) {
        std::env::set_var(
            M1_DECODE_KVSHARE_ENV_1_ALWAYS_GATE_AT_THE_MEASURED_KNEE_0_OR_UNSET_FOR_THE_FUSED_ARM,
            arm,
        );
        let mut cur = 2000u32;
        let mut pos = depth;
        let mut toks = Vec::new();
        let mut margins = Vec::new();
        for _ in 0..GREEDY_STEPS_64_MATCHING_THE_SERVING_LADDERS_TIMED_WINDOW {
            let tokens =
                Tensor::from_vec(vec![cur], (1usize, 1usize), &device).expect("token");
            let positions =
                Tensor::from_vec(vec![pos as i32], 1usize, &device).expect("position");
            let logits = model
                .forward_with_cache(&tokens, &positions, cache)
                .unwrap_or_else(|e| panic!("greedy decode arm={arm} at pos {pos}: {e:#}"));
            let row = logits
                .to_dtype(DType::F32)
                .expect("f32")
                .flatten_all()
                .expect("flatten")
                .to_vec1::<f32>()
                .expect("to host");
            let tok = argmax(&row);
            toks.push(tok);
            margins.push(argmax_top1_minus_top2_margin(&row));
            cur = tok;
            pos += 1;
        }
        (toks, margins)
    };
    let (toks_fused, margins_fused) = greedy_stream(&mut cache_a, "0");
    drop(cache_a);
    std::env::set_var(
        M1_DECODE_KVSHARE_ENV_1_ALWAYS_GATE_AT_THE_MEASURED_KNEE_0_OR_UNSET_FOR_THE_FUSED_ARM,
        "1",
    );
    let mut cache_b = model.new_kv_cache(slots).expect("kv cache B");
    prime(&mut cache_b);
    let (toks_kvshare, margins_kvshare) = greedy_stream(&mut cache_b, "1");
    drop(cache_b);
    std::env::remove_var(
        M1_DECODE_KVSHARE_ENV_1_ALWAYS_GATE_AT_THE_MEASURED_KNEE_0_OR_UNSET_FOR_THE_FUSED_ARM,
    );
    std::env::remove_var("NV_Q36_GRAPHED_DECODE_FIX");

    let mismatches: Vec<usize> = (0..toks_fused.len())
        .filter(|&i| toks_fused[i] != toks_kvshare[i])
        .collect();
    eprintln!(
        "KVSHARE-PARITY qwen38 depth={depth} greedy_steps={} token_mismatches={} first={:?}",
        toks_fused.len(),
        mismatches.len(),
        mismatches.first()
    );
    if let Some(&i) = mismatches.first() {
        let mf = margins_fused[i];
        let mk = margins_kvshare[i];
        eprintln!(
            "KVSHARE-PARITY first divergence idx={i} fused_tok={} kvshare_tok={} \
             fused_margin={mf:.4} kvshare_margin={mk:.4} (later steps ride divergent \
             contexts and are not causally comparable)",
            toks_fused[i], toks_kvshare[i]
        );
        assert!(
            mf <= ADAPTIVE_TIE_MARGIN_MAX_1_BF16_LOGIT_UNIT_MATCHING_THE_QWEN38_CROSS_SHAPE_TIE_CLASS_BAR
                || mk <= ADAPTIVE_TIE_MARGIN_MAX_1_BF16_LOGIT_UNIT_MATCHING_THE_QWEN38_CROSS_SHAPE_TIE_CLASS_BAR,
            "greedy streams split at step {i} with BOTH arms confident (argmax margins \
             fused={mf:.4} kvshare={mk:.4}): a split-order reduction may flip greedy only \
             where at least one trajectory sits inside the argmax noise band; two confident \
             computations of the same committed prefix disagreeing is a kernel indexing bug, \
             not tie-class reordering"
        );
    }
}
