#![cfg(feature = "cuda")]

use candle_core::{DType, Device, Tensor, D};
use nv_models::qwen3_5_moe::{Qwen3Moe, Qwen3MoeKvCache, Qwen3_5DenseConfig};
use nv_specdecode::chain::{accept_prefix_argmax, build_chain_batch};
use nv_specdecode::dflash::{DFlashSpeculatorConfig, LoadedDFlashDrafter};
use std::path::PathBuf;
use std::time::Instant;

const DSPARK_TARGET_LAYER_IDS_ARE_THE_CHECKPOINTS_OWN_DFLASH_CONFIG_NOT_A_GUESS: [usize; 5] =
    [4, 16, 28, 40, 52];

const DSPARK_BLOCK_SIZE_6_GIVES_QUERY_ROWS_7_WHICH_IS_THE_REFERENCES_ROW_COUNT_AND_A_VERIFY_WIDTH_OF_EXACTLY_8: usize = 6;

const MK_VERIFY_TEMPLATE_CAP_8_IS_THE_HARD_CEILING_DSPARK_SITS_ON_WITH_ZERO_HEADROOM: usize = 8;

const SYNTHETIC_FILL_CHUNK_512_MATCHES_THE_RECORD_LADDERS_BLOCK: usize = 512;

const TAP_GATE_FILL_DEPTH_256_IS_A_CONTROL_RUNG_THE_TAP_IDENTITY_IS_DEPTH_INDEPENDENT: usize = 256;

const PRICE_WARM_LAPS_6_DISCARDED_BECAUSE_A_COLD_CLOCK_READS_2X_SLOW: usize = 6;

const PRICE_TIMED_LAPS_24_IS_A_MEDIAN_NOT_A_MEAN_SO_ONE_PREEMPTION_CANNOT_MOVE_IT: usize = 24;

const ACCEPT_WARM_ROUNDS_8_BOUND_DEPTH_DRIFT_OF_THE_COUNTED_WINDOW: usize = 8;

const ACCEPT_COUNTED_ROUNDS_48_AT_UP_TO_8_TOKENS_A_ROUND_IS_A_384_TOKEN_WINDOW: usize = 48;

const NULL_PAIR_ENV_REPLACES_REAL_TARGET_AUX_WITH_RMS_MATCHED_NOISE_SO_A_DEAD_FEATURE_PATH_CANNOT_POSE_AS_A_WEAK_DRAFTER: &str = "NV_Q38AUX_RANDN_CTX";

fn feature_rms(t: &Tensor) -> f32 {
    let v: Vec<f32> = t
        .to_dtype(DType::F32)
        .expect("f32")
        .flatten_all()
        .expect("flat")
        .to_vec1()
        .expect("host");
    let n = v.len().max(1) as f32;
    (v.iter().map(|x| x * x).sum::<f32>() / n).sqrt()
}

fn gate() -> bool {
    if std::env::var("NV_Q38AUX_TEST").ok().as_deref() != Some("1") {
        eprintln!("PRECONDITION NOT MET, THIS TEST EXECUTED NOTHING: set NV_Q38AUX_TEST=1");
        return false;
    }
    true
}

fn env_dir(key: &str) -> PathBuf {
    let v = std::env::var(key)
        .unwrap_or_else(|_| panic!("PRECONDITION NOT MET, THIS TEST EXECUTED NOTHING: set {key}"));
    let p = PathBuf::from(v);
    assert!(p.is_dir(), "{key} is not a directory: {}", p.display());
    p
}

fn env_usize(key: &str, default: usize) -> usize {
    match std::env::var(key) {
        Ok(v) if !v.is_empty() => v
            .parse()
            .unwrap_or_else(|_| panic!("{key} must be a plain integer, got {v:?}")),
        _ => default,
    }
}

fn median_of(v: &[f64]) -> f64 {
    let mut s = v.to_vec();
    s.sort_by(|a, b| a.partial_cmp(b).unwrap());
    if s.is_empty() {
        return f64::NAN;
    }
    if s.len() % 2 == 1 {
        s[s.len() / 2]
    } else {
        0.5 * (s[s.len() / 2 - 1] + s[s.len() / 2])
    }
}

fn load_target(dir: &std::path::Path, device: &Device) -> Qwen3Moe {
    let raw = std::fs::read_to_string(dir.join("config.json")).expect("target config.json");
    let cfg = Qwen3_5DenseConfig::from_hf_json_str(&raw).expect("dense config");
    let qcfg = nv_weights::QuantizationConfig::from_hf_json_str(&raw).expect("quant config");
    let weights = nv_weights::WeightLoader::open_dir(dir, device).expect("target weights");
    let m = Qwen3Moe::from_loader_dense_quantized(cfg, &weights, &qcfg, device)
        .expect("dense NVFP4 Qwen3.8-27B target");
    drop(weights);
    m
}

fn load_dspark(
    draft_dir: &std::path::Path,
    target_dir: &std::path::Path,
    device: &Device,
) -> LoadedDFlashDrafter {
    let mut cfg =
        DFlashSpeculatorConfig::from_hf_json_file(&draft_dir.join("config.json")).expect("dspark cfg");
    cfg.block_size = DSPARK_BLOCK_SIZE_6_GIVES_QUERY_ROWS_7_WHICH_IS_THE_REFERENCES_ROW_COUNT_AND_A_VERIFY_WIDTH_OF_EXACTLY_8;
    assert_eq!(
        cfg.query_rows() + 1,
        MK_VERIFY_TEMPLATE_CAP_8_IS_THE_HARD_CEILING_DSPARK_SITS_ON_WITH_ZERO_HEADROOM,
        "the anchor plus {} drafts must be exactly the mk-verify template cap",
        cfg.query_rows()
    );
    assert_eq!(
        cfg.aux_hidden_state_layer_ids,
        DSPARK_TARGET_LAYER_IDS_ARE_THE_CHECKPOINTS_OWN_DFLASH_CONFIG_NOT_A_GUESS.to_vec()
    );
    let target = nv_weights::WeightLoader::open_dir(target_dir, device).expect("target weights");
    let embed = nv_layers::weight_load::load_tensor(
        &target,
        "model.language_model.embed_tokens.weight",
        &[cfg.target_vocab_size, cfg.hidden_size],
        DType::BF16,
    )
    .expect("target embed_tokens");
    let lm_head = nv_layers::linear::fp8_e4m3_rowscale_checkpoint_dequant_linear(
        &target,
        "lm_head",
        cfg.target_vocab_size,
        cfg.hidden_size,
        DType::BF16,
    )
    .expect("target lm_head dequantized from fp8 + row scales");
    drop(target);
    LoadedDFlashDrafter::load_from_safetensors_with_target_heads(
        &cfg,
        &draft_dir.join("model.safetensors"),
        device,
        Some(&embed),
        Some(lm_head),
    )
    .expect("headless DSpark drafter against the target embed and lm_head")
}

fn synthetic_fill(cache: &mut Qwen3MoeKvCache, model: &Qwen3Moe, depth: usize, device: &Device) {
    let n_kv = model.config().num_key_value_heads;
    let hd = model.config().head_dim;
    let chunk = SYNTHETIC_FILL_CHUNK_512_MATCHES_THE_RECORD_LADDERS_BLOCK;
    let vals: Vec<f32> = (0..chunk * n_kv * hd)
        .map(|i| {
            let x = ((i as u64).wrapping_mul(2654435761) % 2048) as f32 / 2048.0;
            x - 0.5
        })
        .collect();
    let k = Tensor::from_vec(vals.clone(), (1usize, chunk, n_kv, hd), device)
        .expect("k template")
        .to_dtype(DType::BF16)
        .expect("k bf16");
    let v = Tensor::from_vec(vals, (1usize, chunk, n_kv, hd), device)
        .expect("v template")
        .to_dtype(DType::BF16)
        .expect("v bf16");
    let mut pos = 0usize;
    while pos < depth {
        let n = chunk.min(depth - pos);
        let (kn, vn) = if n == chunk {
            (k.clone(), v.clone())
        } else {
            (
                k.narrow(1, 0, n).expect("k tail"),
                v.narrow(1, 0, n).expect("v tail"),
            )
        };
        cache
            .write_synthetic_rows_at_every_full_attention_slot_for_depth_timing_decode_reads_cache_size_not_values(
                pos, &kn, &vn,
            )
            .unwrap_or_else(|e| panic!("synthetic fill at {pos}: {e:#}"));
        pos += n;
    }
    assert_eq!(cache.current_len(), depth, "synthetic fill must land at depth");
}

fn rows_tensors(ids: &[u32], first_pos: usize, device: &Device) -> (Tensor, Tensor) {
    let m = ids.len();
    let t = Tensor::from_vec(ids.to_vec(), (1usize, m), device).expect("row ids");
    let p = Tensor::from_vec(
        (first_pos as i32..(first_pos + m) as i32).collect::<Vec<i32>>(),
        m,
        device,
    )
    .expect("row positions");
    (t, p)
}

fn bits_of(t: &Tensor) -> Vec<u32> {
    t.to_dtype(DType::F32)
        .expect("f32")
        .flatten_all()
        .expect("flat")
        .to_vec1::<f32>()
        .expect("host")
        .into_iter()
        .map(|v| v.to_bits())
        .collect()
}

#[test]
#[ignore = "loads the ~22.6 GB Qwen3.8-27B NVFP4 target; NV_Q38AUX_TEST=1 + NV_DSPARK_TARGET_DIR. \
            Proves the new aux tap on the qwen3.8 trunk is numerically free: at the m=8 verify \
            shape the logits and the trunk hidden are BITWISE identical with the tap armed and \
            disarmed, and the tap of the LAST layer is bitwise the returned hidden."]
fn qwen38_aux_tap_is_bitwise_free_at_the_m8_verify_shape_and_its_last_layer_row_is_the_trunk_hidden() {
    if !gate() {
        return;
    }
    let target_dir = env_dir("NV_DSPARK_TARGET_DIR");
    let device = Device::new_cuda_with_stream(0).expect("cuda");
    let model = load_target(&target_dir, &device);
    let n_layers = model.config().num_hidden_layers;
    let hidden = model.config().hidden_size;
    let depth = TAP_GATE_FILL_DEPTH_256_IS_A_CONTROL_RUNG_THE_TAP_IDENTITY_IS_DEPTH_INDEPENDENT;
    let m = MK_VERIFY_TEMPLATE_CAP_8_IS_THE_HARD_CEILING_DSPARK_SITS_ON_WITH_ZERO_HEADROOM;
    let ids: Vec<u32> = (0..m as u32).map(|i| 1000 + i * 37).collect();

    let run = |aux_layers: &[usize]| -> (Vec<u32>, Vec<u32>, Vec<Vec<u32>>) {
        let mut cache = model.new_kv_cache(depth + 64).expect("kv cache");
        synthetic_fill(&mut cache, &model, depth, &device);
        let (t, p) = rows_tensors(&ids, depth, &device);
        cache.set_capture_lin_ckpts(true);
        let (logits, hid, aux) = model
            .forward_with_cache_dispatched_hidden_rows_aux(&t, &p, &mut cache, None, None, aux_layers)
            .expect("m=8 verify forward with the aux tap");
        (
            bits_of(&logits),
            bits_of(&hid),
            aux.iter().map(bits_of).collect(),
        )
    };

    let (l_off, h_off, a_off) = run(&[]);
    assert!(a_off.is_empty(), "an unrequested tap must collect nothing");
    let (l_on, h_on, a_on) =
        run(&DSPARK_TARGET_LAYER_IDS_ARE_THE_CHECKPOINTS_OWN_DFLASH_CONFIG_NOT_A_GUESS);
    assert_eq!(
        a_on.len(),
        5,
        "the DSpark tap must return one row block per target_layer_id"
    );
    for (i, a) in a_on.iter().enumerate() {
        assert_eq!(
            a.len(),
            m * hidden,
            "aux block {i} must be [m={m}, hidden={hidden}]"
        );
    }
    assert_eq!(
        l_off.len(),
        m * model.config().vocab_size,
        "the verify forward must return all {m} logit rows"
    );
    assert_eq!(
        l_off, l_on,
        "arming the aux tap changed the logits; the tap must be a read, not a numerics change"
    );
    assert_eq!(
        h_off, h_on,
        "arming the aux tap changed the trunk hidden; the tap must be a read, not a numerics change"
    );

    let (_, h_last, a_last) = run(&[n_layers - 1]);
    assert_eq!(
        a_last.len(),
        1,
        "one requested layer must yield exactly one aux block"
    );
    assert_eq!(
        a_last[0], h_last,
        "the tap of the last layer must be BITWISE the residual this forward already returns as \
         `hidden`; any difference means the tap reads a different tensor than the trunk carries \
         forward and every drafter feature built on it is off by a layer"
    );
    let mid = &a_on[4];
    assert_ne!(
        *mid, h_on,
        "layer 52's residual must not equal layer 63's; an equal pair would mean the tap fires \
         once and repeats, which is exactly how a silently-broken tap looks"
    );
    eprintln!(
        "[tap-gate] m={m} depth={depth} logits_bits={} hidden_bits={} aux_blocks={} ALL BITWISE OK",
        l_off.len(),
        h_off.len(),
        a_on.len()
    );
}

#[test]
#[ignore = "loads the ~22.6 GB Qwen3.8-27B NVFP4 target once and prices the VERIFY ROUND width \
            directly at m in {1,2,4,8} over a synthetic KV fill at NV_CTX_TOKENS, interleaved and \
            rotated with the warm laps discarded. m=1 is the dense one-token-per-forward control \
            measured in the SAME process, so the ratio is not differenced across runs."]
fn qwen38_verify_round_price_by_row_count_including_m8_the_dspark_width() {
    if !gate() {
        return;
    }
    let target_dir = env_dir("NV_DSPARK_TARGET_DIR");
    let depth = env_usize("NV_CTX_TOKENS", 16384);
    let laps = env_usize(
        "NV_Q38AUX_LAPS",
        PRICE_TIMED_LAPS_24_IS_A_MEDIAN_NOT_A_MEAN_SO_ONE_PREEMPTION_CANNOT_MOVE_IT,
    );
    let device = Device::new_cuda_with_stream(0).expect("cuda");
    let model = load_target(&target_dir, &device);
    let max_seq = depth + 64;
    assert!(
        max_seq < model.config().max_position_embeddings,
        "depth {depth} exceeds the model's declared max_position_embeddings"
    );
    let mut cache = model.new_kv_cache(max_seq).expect("kv cache");
    let fill_t0 = Instant::now();
    synthetic_fill(&mut cache, &model, depth, &device);
    let _ = device.synchronize();
    let fill_s = fill_t0.elapsed().as_secs_f64();

    let arms: [(usize, bool); 5] = [(1, false), (2, false), (4, false), (8, false), (8, true)];
    let mut samples: std::collections::BTreeMap<(usize, bool), Vec<f64>> =
        std::collections::BTreeMap::new();
    let ids: Vec<u32> = (0..8u32).map(|i| 1000 + i * 37).collect();
    let total_laps = PRICE_WARM_LAPS_6_DISCARDED_BECAUSE_A_COLD_CLOCK_READS_2X_SLOW + laps;
    for lap in 0..total_laps {
        for r in 0..arms.len() {
            let (m, tap) = arms[(lap + r) % arms.len()];
            let (t, p) = rows_tensors(&ids[..m], depth, &device);
            let aux: &[usize] = if tap {
                &DSPARK_TARGET_LAYER_IDS_ARE_THE_CHECKPOINTS_OWN_DFLASH_CONFIG_NOT_A_GUESS
            } else {
                &[]
            };
            cache.set_capture_lin_ckpts(true);
            let t0 = Instant::now();
            let (logits, _hid, collected) = model
                .forward_with_cache_dispatched_hidden_rows_aux(&t, &p, &mut cache, None, None, aux)
                .unwrap_or_else(|e| panic!("m={m} verify forward at depth {depth}: {e:#}"));
            let _guard: Vec<u32> = logits
                .narrow(0, 0, 1)
                .and_then(|r| r.argmax(D::Minus1))
                .and_then(|r| r.flatten_all())
                .and_then(|r| r.to_dtype(DType::U32))
                .and_then(|r| r.to_vec1())
                .expect("a forward whose result is never read can be scheduled away");
            let _ = device.synchronize();
            let ms = t0.elapsed().as_secs_f64() * 1e3;
            assert_eq!(collected.len(), aux.len());
            cache.set_current_len(depth);
            cache
                .rollback_lin_to(1)
                .expect("rollback_lin_to refuses 0, so the price lap commits one gdn row and \
                         rewinds the rest, which is exactly what a one-token-accepted round does");
            if lap >= PRICE_WARM_LAPS_6_DISCARDED_BECAUSE_A_COLD_CLOCK_READS_2X_SLOW {
                samples.entry((m, tap)).or_default().push(ms);
            }
        }
    }
    let m1 = median_of(samples.get(&(1, false)).expect("m=1 control"));
    eprintln!("[price] depth={depth} fill_s={fill_s:.1} laps={laps} arms=m1,m2,m4,m8,m8+tap");
    for ((m, tap), v) in &samples {
        let med = median_of(v);
        eprintln!(
            "[VERIFY-PRICE] depth={depth} m={m} tap={tap} rows={m} median_ms={med:.3} \
             per_dense_step={:.3} n={} min={:.3} max={:.3}",
            med / m1,
            v.len(),
            v.iter().cloned().fold(f64::INFINITY, f64::min),
            v.iter().cloned().fold(f64::NEG_INFINITY, f64::max)
        );
    }
    let m8 = median_of(samples.get(&(8, false)).expect("m=8"));
    eprintln!(
        "[BREAKEVEN] depth={depth} m8_verify_ms={m8:.3} m1_dense_ms={m1:.3} \
         accepted_tokens_needed_per_round={:.3} \
         (verify-only; the drafter block and the commit are NOT in this number)",
        m8 / m1
    );
    assert!(
        m8 > 0.0 && m1 > 0.0,
        "a zero median means the clock never advanced"
    );

    if std::env::var("NV_Q38AUX_DRAFT_PRICE").ok().as_deref() != Some("1") {
        eprintln!(
            "[price] drafter half NOT priced; set NV_Q38AUX_DRAFT_PRICE=1 with NV_DSPARK_DRAFT_DIR"
        );
        return;
    }
    let draft_dir = env_dir("NV_DSPARK_DRAFT_DIR");
    let drafter = load_dspark(&draft_dir, &target_dir, &device);
    let mut ctx = drafter
        .new_context_kv(depth + 64)
        .expect("drafter context kv sized once");
    let fill_chunk = 4096usize.min(depth);
    let mut fed = 0usize;
    while fed < depth {
        let n = fill_chunk.min(depth - fed);
        let states = Tensor::randn(0f32, 0.02f32, (n, drafter.config().hidden_size), &device)
            .expect("synthetic drafter context states")
            .to_dtype(drafter.dtype())
            .expect("drafter dtype");
        let positions: Vec<u32> = (fed as u32..(fed + n) as u32).collect();
        drafter
            .append_context_kv(&mut ctx, &states, &positions)
            .unwrap_or_else(|e| panic!("synthetic drafter ctx append at {fed}: {e:#}"));
        fed += n;
    }
    let _ = device.synchronize();
    assert_eq!(ctx.len(), depth);
    let mut block_ms: Vec<f64> = Vec::new();
    for lap in 0..(PRICE_WARM_LAPS_6_DISCARDED_BECAUSE_A_COLD_CLOCK_READS_2X_SLOW + laps.min(8)) {
        let t0 = Instant::now();
        let blk = drafter
            .draft_block_dspark(&ctx, 1000 + lap as u32)
            .unwrap_or_else(|e| panic!("dspark block at depth {depth}: {e:#}"));
        let _ = device.synchronize();
        assert_eq!(blk.ids.len(), drafter.config().query_rows());
        if lap >= PRICE_WARM_LAPS_6_DISCARDED_BECAUSE_A_COLD_CLOCK_READS_2X_SLOW {
            block_ms.push(t0.elapsed().as_secs_f64() * 1e3);
        }
    }
    eprintln!(
        "[DRAFT-PRICE] depth={depth} rows={} block_ms_median={:.3} n={} \
         (eager f32 sdpa over the whole context, NOT a tuned kernel: this is an implementation \
         cost, not a property of DSpark)",
        drafter.config().query_rows(),
        median_of(&block_ms),
        block_ms.len()
    );
}

#[test]
#[ignore = "loads the ~22.6 GB target AND the 2.7 GB DSpark drafter, does a REAL chunked prefill \
            of NV_Q38AUX_CORPUS to NV_CTX_TOKENS through the new aux tap, then runs speculative \
            rounds and counts ACCEPTED tokens. Acceptance is exact u32 argmax equality; the \
            tok/s convention here is ACCEPTED tokens per round, stated on every line."]
fn qwen38_dspark_acceptance_at_depth_on_real_target_aux_features() {
    if !gate() {
        return;
    }
    let target_dir = env_dir("NV_DSPARK_TARGET_DIR");
    let draft_dir = env_dir("NV_DSPARK_DRAFT_DIR");
    let depth = env_usize("NV_CTX_TOKENS", 16384);
    let rounds_wanted = env_usize(
        "NV_Q38AUX_ROUNDS",
        ACCEPT_COUNTED_ROUNDS_48_AT_UP_TO_8_TOKENS_A_ROUND_IS_A_384_TOKEN_WINDOW,
    );
    let corpus = std::env::var("NV_Q38AUX_CORPUS").unwrap_or_else(|_| {
        panic!(
            "PRECONDITION NOT MET, THIS TEST EXECUTED NOTHING: set NV_Q38AUX_CORPUS to a raw text \
             file with at least the rung's token count; a tiled paragraph reads as an upper \
             bracket, not an acceptance number"
        )
    });
    let markov_off = std::env::var("NV_Q38AUX_NO_MARKOV").ok().as_deref() == Some("1");
    let randn_ctx = std::env::var(
        NULL_PAIR_ENV_REPLACES_REAL_TARGET_AUX_WITH_RMS_MATCHED_NOISE_SO_A_DEAD_FEATURE_PATH_CANNOT_POSE_AS_A_WEAK_DRAFTER,
    )
    .ok()
    .as_deref()
        == Some("1");
    let device = Device::new_cuda_with_stream(0).expect("cuda");
    let tok = tokenizers::Tokenizer::from_file(target_dir.join("tokenizer.json"))
        .expect("target tokenizer.json");
    let text = std::fs::read_to_string(&corpus).expect("corpus file");
    let all: Vec<u32> = tok
        .encode(text.as_str(), false)
        .expect("encode corpus")
        .get_ids()
        .to_vec();
    assert!(
        all.len() >= depth,
        "corpus {corpus} holds {} tokens but the rung needs {depth}; tiling it would put \
         acceptance in the doc-repetition upper bracket instead of free prose",
        all.len()
    );
    let prompt = all[..depth].to_vec();

    let model = load_target(&target_dir, &device);
    let drafter = load_dspark(&draft_dir, &target_dir, &device);
    let m_draft = drafter.config().query_rows();
    let verify_rows = m_draft + 1;
    assert_eq!(
        verify_rows, MK_VERIFY_TEMPLATE_CAP_8_IS_THE_HARD_CEILING_DSPARK_SITS_ON_WITH_ZERO_HEADROOM
    );
    let aux_layers = model
        .aux_layer_ids_are_post_layer_residuals_because_hf_hidden_states_index_l_plus_1_is_the_output_of_layer_l(
            &drafter.config().aux_hidden_state_layer_ids,
        )
        .expect("the DSpark target_layer_ids must be in range for this trunk");

    let headroom = rounds_wanted * verify_rows
        + ACCEPT_WARM_ROUNDS_8_BOUND_DEPTH_DRIFT_OF_THE_COUNTED_WINDOW * verify_rows
        + 64;
    let max_seq = depth + headroom;
    let mut cache = model.new_kv_cache(max_seq).expect("kv cache");
    let mut ctx = drafter
        .new_context_kv(max_seq + verify_rows)
        .expect("drafter context kv sized once so no append reallocates mid-window");

    let chunk = nv_models::qwen3_5_moe::q38_prefill_chunk_tokens_from_env()
        .filter(|&c| depth > c)
        .unwrap_or(depth);
    let pre_t0 = Instant::now();
    let mut fed = 0usize;
    let mut last_logits: Option<Tensor> = None;
    while fed < depth {
        let n = chunk.min(depth - fed);
        let (t, p) = rows_tensors(&prompt[fed..fed + n], fed, &device);
        let (logits, _hid, aux) = model
            .forward_with_cache_dispatched_hidden_rows_aux(
                &t,
                &p,
                &mut cache,
                None,
                Some(1),
                &aux_layers,
            )
            .unwrap_or_else(|e| panic!("prefill chunk at {fed}: {e:#}"));
        let feat = Tensor::cat(&aux, 1).expect("aux concat is layer-major within a row");
        let feat = if randn_ctx {
            Tensor::randn(0f32, feature_rms(&feat), feat.dims2().expect("aux 2d"), &device)
                .expect("null-pair context")
                .to_dtype(feat.dtype())
                .expect("null-pair dtype")
        } else {
            feat
        };
        let proj = drafter.project_aux(&feat).expect("project_aux");
        let positions: Vec<u32> = (fed as u32..(fed + n) as u32).collect();
        drafter
            .append_context_kv(&mut ctx, &proj, &positions)
            .expect("append drafter context kv");
        last_logits = Some(logits);
        fed += n;
    }
    let _ = device.synchronize();
    let prefill_s = pre_t0.elapsed().as_secs_f64();
    assert_eq!(cache.current_len(), depth);
    assert_eq!(
        ctx.len(),
        depth,
        "the drafter context must cover exactly the committed trunk"
    );
    let mut anchor: u32 = last_logits
        .expect("non-empty prompt")
        .argmax(D::Minus1)
        .and_then(|t| t.flatten_all())
        .and_then(|t| t.to_dtype(DType::U32))
        .and_then(|t| t.to_vec1::<u32>())
        .expect("anchor argmax")
        .pop()
        .expect("one row of logits");

    let mut hist: std::collections::BTreeMap<usize, usize> = std::collections::BTreeMap::new();
    let mut round_ms: Vec<f64> = Vec::new();
    let mut draft_ms: Vec<f64> = Vec::new();
    let mut verify_ms: Vec<f64> = Vec::new();
    let mut commit_ms: Vec<f64> = Vec::new();
    let mut conf_sum = 0f64;
    let mut conf_n = 0usize;
    let mut accepted_total = 0usize;
    let mut emitted_total = 0usize;
    let mut drafted_total = 0usize;
    let total_rounds = ACCEPT_WARM_ROUNDS_8_BOUND_DEPTH_DRIFT_OF_THE_COUNTED_WINDOW + rounds_wanted;
    for r in 0..total_rounds {
        let l = cache.current_len();
        assert_eq!(ctx.len(), l, "drafter/trunk desync at round {r}");
        assert!(l + verify_rows <= max_seq, "kv headroom exhausted at round {r}");
        let round_t0 = Instant::now();
        let blk = if markov_off {
            let (ids, top_prob, _margin) = drafter
                .draft_block_all_rows_with_conf(&ctx, anchor)
                .unwrap_or_else(|e| panic!("plain dflash draft at round {r}: {e:#}"));
            nv_specdecode::dflash::DSparkBlockDraft {
                submit_len: ids.len(),
                accept_probs: top_prob,
                ids,
            }
        } else {
            drafter
                .draft_block_dspark(&ctx, anchor)
                .unwrap_or_else(|e| panic!("dspark draft at round {r}: {e:#}"))
        };
        let _ = device.synchronize();
        let d_ms = round_t0.elapsed().as_secs_f64() * 1e3;
        assert_eq!(blk.ids.len(), m_draft);
        assert_eq!(
            blk.submit_len, m_draft,
            "NV_DSPARK_ADAPTIVE must stay unset here so every round is fixed-k"
        );

        let v_t0 = Instant::now();
        let batch = build_chain_batch(anchor, &blk.ids, verify_rows, true).expect("verify batch");
        assert_eq!(batch.len(), verify_rows);
        let (t, p) = rows_tensors(&batch, l, &device);
        cache.set_capture_lin_ckpts(true);
        let (vlogits, _vhid, vaux) = model
            .forward_with_cache_dispatched_hidden_rows_aux(
                &t,
                &p,
                &mut cache,
                None,
                None,
                &aux_layers,
            )
            .unwrap_or_else(|e| panic!("m={verify_rows} verify at round {r}: {e:#}"));
        let greedy: Vec<u32> = vlogits
            .argmax(D::Minus1)
            .and_then(|t| t.flatten_all())
            .and_then(|t| t.to_dtype(DType::U32))
            .and_then(|t| t.to_vec1())
            .expect("verify argmax");
        let _ = device.synchronize();
        let v_ms = v_t0.elapsed().as_secs_f64() * 1e3;

        let c_t0 = Instant::now();
        let acc = accept_prefix_argmax(&batch, &greedy).expect("exact u32 accept");
        let consumed = acc.commit_len;
        let feat = Tensor::cat(&vaux, 1).expect("verify aux concat");
        let feat = feat.narrow(0, 0, consumed).expect("committed aux rows");
        let feat = if randn_ctx {
            Tensor::randn(0f32, feature_rms(&feat), feat.dims2().expect("aux 2d"), &device)
                .expect("null-pair context")
                .to_dtype(feat.dtype())
                .expect("null-pair dtype")
        } else {
            feat
        };
        let proj = drafter.project_aux(&feat).expect("project_aux");
        let positions: Vec<u32> = (l as u32..(l + consumed) as u32).collect();
        drafter
            .append_context_kv(&mut ctx, &proj, &positions)
            .expect("append committed rows to the drafter context");
        cache.set_current_len(l + consumed);
        cache.rollback_lin_to(consumed).expect("lin rollback");
        anchor = acc.next_bonus;
        let _ = device.synchronize();
        let c_ms = c_t0.elapsed().as_secs_f64() * 1e3;

        if r >= ACCEPT_WARM_ROUNDS_8_BOUND_DEPTH_DRIFT_OF_THE_COUNTED_WINDOW {
            *hist.entry(acc.draft_accepted).or_insert(0) += 1;
            accepted_total += acc.draft_accepted;
            emitted_total += acc.draft_accepted + 1;
            drafted_total += m_draft;
            round_ms.push(round_t0.elapsed().as_secs_f64() * 1e3);
            draft_ms.push(d_ms);
            verify_ms.push(v_ms);
            commit_ms.push(c_ms);
            for p in &blk.accept_probs {
                conf_sum += *p as f64;
                conf_n += 1;
            }
        }
    }
    let counted = round_ms.len();
    let mean_accept = accepted_total as f64 / counted as f64;
    let mean_emitted = emitted_total as f64 / counted as f64;
    let rm = median_of(&round_ms);
    eprintln!(
        "[DSPARK-ACCEPT] depth={depth} markov_off={markov_off} randn_ctx={randn_ctx} prefill_s={prefill_s:.1} rounds={counted} \
         draft_rows_per_round={m_draft} verify_rows={verify_rows} \
         mean_accepted_drafts={mean_accept:.3} mean_emitted_per_round={mean_emitted:.3} \
         accept_rate={:.4} round_ms_median={rm:.3} \
         ms_per_ACCEPTED_token={:.3} \
         draft_ms_med={:.3} verify_ms_med={:.3} commit_ms_med={:.3} \
         mean_confidence_head_p={:.4} hist={hist:?}",
        accepted_total as f64 / drafted_total as f64,
        rm / mean_emitted,
        median_of(&draft_ms),
        median_of(&verify_ms),
        median_of(&commit_ms),
        conf_sum / conf_n.max(1) as f64
    );
    assert!(
        counted == rounds_wanted,
        "the counted window must be exactly the requested rounds"
    );
}
