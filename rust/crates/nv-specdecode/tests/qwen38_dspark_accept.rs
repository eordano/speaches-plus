#![cfg(feature = "cuda")]

use candle_core::{DType, Device, Tensor, D};
use nv_models::qwen3_5_moe::{Qwen3Moe, Qwen3MoeKvCache, Qwen3_5DenseConfig};
use nv_specdecode::chain::{accept_prefix_argmax, build_chain_batch};
use nv_specdecode::dflash::{DFlashSpeculatorConfig, DSparkBlockDraft, LoadedDFlashDrafter};
use std::collections::BTreeMap;
use std::path::PathBuf;
use std::time::Instant;

const DSPARK_TARGET_LAYER_IDS_ARE_THE_CHECKPOINTS_OWN_DFLASH_CONFIG_NOT_A_GUESS: [usize; 5] =
    [4, 16, 28, 40, 52];

const DSPARK_BLOCK_SIZE_6_GIVES_QUERY_ROWS_7_WHICH_IS_THE_REFERENCES_ROW_COUNT_AND_A_VERIFY_WIDTH_OF_EXACTLY_8: usize = 6;

const ACCEPT_DEPTHS_256_CONTROL_AND_8192_THE_BAR_RUNG: [usize; 2] = [256, 8192];

const ACCEPT_ROUNDS_200_DEFAULT_SO_THE_HISTOGRAM_HAS_MASS_IN_EVERY_BIN: usize = 200;

const PREFILL_CHUNK_2048_BOUNDS_TRANSIENTS_WITHOUT_AN_ENV_KNOB: usize = 2048;

const ADAPTIVE_THRESHOLDS_BRACKET_THE_CUMULATIVE_CONFIDENCE_KNEE: [f32; 3] = [0.3, 0.5, 0.7];

const QUARTILE_SPLIT_4_EXPOSES_GREEDY_DEGENERATION_ACROSS_THE_COUNTED_WINDOW: usize = 4;

const CALIBRATION_BUCKETS_10_OF_WIDTH_0_1_OVER_PREDICTED_P: usize = 10;

const FIXED_ARM_CORPUS_OFFSETS_0_AND_100K_GIVE_TWO_INDEPENDENT_GREEDY_TRAJECTORIES_SO_THE_VERDICT_IS_NOT_ONE_TEXT_SEGMENT:
    [usize; 2] = [0, 100_000];

const REPETITION_NGRAM_4_DISTINCT_FRACTION_NEAR_ZERO_MEANS_THE_GREEDY_CONTINUATION_IS_LOOPING_AND_ACCEPTANCE_IS_INFLATED: usize = 4;

const DECODED_TAIL_CHARS_300_MAKE_DEGENERATION_VISIBLE_IN_THE_LOG: usize = 300;

const EMITTED_PER_ROUND_IS_ACCEPTED_DRAFTS_PLUS_ONE_MATCHING_THE_REFERENCES_ACCEPTANCE_LENGTH_PLUS_ONE: usize = 1;

const BAR_MTP_K3_EMITTED_PER_ROUND_ON_RECORD_LOW: f64 = 2.70;
const BAR_MTP_K3_EMITTED_PER_ROUND_ON_RECORD_HIGH: f64 = 3.16;
const BAR_LLAMA_CPP_NEXTN_CEILING_EMITTED_PER_ROUND: f64 = 3.55;
const BAR_EXTERNAL_LING_3_0_FLASH_DSPARK_ACCEPT_LENGTH_ON_A_COMPARABLE_HYBRID: f64 = 9.95;
const BAR_M8_VERIFY_BREAKEVEN_EMITTED_PER_ROUND: f64 = 2.41;
const BAR_SERVING_INTEGRATION_WORTHWHILE_MEAN_EMITTED_AT_8192: f64 = 4.5;

fn gate_loud() {
    if std::env::var("NV_DSPARK_TEST").ok().as_deref() != Some("1") {
        panic!("set NV_DSPARK_TEST=1 to run (it must never silently skip)");
    }
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
    let mut cfg = DFlashSpeculatorConfig::from_hf_json_file(&draft_dir.join("config.json"))
        .expect("dspark cfg");
    cfg.block_size =
        DSPARK_BLOCK_SIZE_6_GIVES_QUERY_ROWS_7_WHICH_IS_THE_REFERENCES_ROW_COUNT_AND_A_VERIFY_WIDTH_OF_EXACTLY_8;
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

struct RoundRec {
    submitted: usize,
    accepted: usize,
    probs: Vec<f32>,
}

struct ArmResult {
    recs: Vec<RoundRec>,
    emitted_ids: Vec<u32>,
    prefill_s: f64,
    rounds_s: f64,
    end_len: usize,
}

fn distinct_ngram_fraction(ids: &[u32]) -> f64 {
    let n = REPETITION_NGRAM_4_DISTINCT_FRACTION_NEAR_ZERO_MEANS_THE_GREEDY_CONTINUATION_IS_LOOPING_AND_ACCEPTANCE_IS_INFLATED;
    if ids.len() < n {
        return 1.0;
    }
    let total = ids.len() - n + 1;
    let distinct: std::collections::BTreeSet<&[u32]> = ids.windows(n).collect();
    distinct.len() as f64 / total as f64
}

fn prefill_through_aux_tap(
    model: &Qwen3Moe,
    drafter: &LoadedDFlashDrafter,
    aux_layers: &[usize],
    prompt: &[u32],
    cache: &mut Qwen3MoeKvCache,
    ctx: &mut nv_specdecode::dflash::DFlashContextKv,
    device: &Device,
) -> u32 {
    let depth = prompt.len();
    let mut fed = 0usize;
    let mut last_logits: Option<Tensor> = None;
    while fed < depth {
        let n = PREFILL_CHUNK_2048_BOUNDS_TRANSIENTS_WITHOUT_AN_ENV_KNOB.min(depth - fed);
        let (t, p) = rows_tensors(&prompt[fed..fed + n], fed, device);
        let (logits, _hid, aux) = model
            .forward_with_cache_dispatched_hidden_rows_aux(&t, &p, cache, None, Some(1), aux_layers)
            .unwrap_or_else(|e| panic!("prefill chunk at {fed}: {e:#}"));
        let feat = Tensor::cat(&aux, 1).expect("aux concat is layer-major within a row");
        let proj = drafter.project_aux(&feat).expect("project_aux");
        let positions: Vec<u32> = (fed as u32..(fed + n) as u32).collect();
        drafter
            .append_context_kv(ctx, &proj, &positions)
            .expect("append drafter context kv");
        last_logits = Some(logits);
        fed += n;
    }
    let _ = device.synchronize();
    assert_eq!(cache.current_len(), depth);
    assert_eq!(
        ctx.len(),
        depth,
        "the drafter context must cover exactly the committed trunk"
    );
    last_logits
        .expect("non-empty prompt")
        .argmax(D::Minus1)
        .and_then(|t| t.flatten_all())
        .and_then(|t| t.to_dtype(DType::U32))
        .and_then(|t| t.to_vec1::<u32>())
        .expect("anchor argmax")
        .pop()
        .expect("one row of logits")
}

fn run_arm(
    model: &Qwen3Moe,
    drafter: &LoadedDFlashDrafter,
    aux_layers: &[usize],
    prompt: &[u32],
    rounds_wanted: usize,
    threshold: Option<f32>,
    device: &Device,
) -> ArmResult {
    match threshold {
        Some(t) => std::env::set_var("NV_DSPARK_ADAPTIVE", format!("{t}")),
        None => std::env::remove_var("NV_DSPARK_ADAPTIVE"),
    }
    let depth = prompt.len();
    let m_draft = drafter.config().query_rows();
    let verify_cap = m_draft + 1;
    let max_seq = depth + rounds_wanted * verify_cap + 64;
    let mut cache = model.new_kv_cache(max_seq).expect("kv cache");
    let mut ctx = drafter
        .new_context_kv(max_seq + verify_cap)
        .expect("drafter context kv sized once so no append reallocates mid-window");
    let pre_t0 = Instant::now();
    let mut anchor = prefill_through_aux_tap(
        model, drafter, aux_layers, prompt, &mut cache, &mut ctx, device,
    );
    let prefill_s = pre_t0.elapsed().as_secs_f64();

    let mut recs: Vec<RoundRec> = Vec::with_capacity(rounds_wanted);
    let mut emitted_ids: Vec<u32> = Vec::with_capacity(rounds_wanted * verify_cap);
    let rounds_t0 = Instant::now();
    for r in 0..rounds_wanted {
        let l = cache.current_len();
        assert_eq!(ctx.len(), l, "drafter/trunk desync at round {r}");
        assert!(l + verify_cap <= max_seq, "kv headroom exhausted at round {r}");
        let blk: DSparkBlockDraft = drafter
            .draft_block_dspark(&ctx, anchor)
            .unwrap_or_else(|e| panic!("dspark draft at round {r}: {e:#}"));
        assert_eq!(blk.ids.len(), m_draft);
        let submitted = blk.submit_len;
        match threshold {
            Some(_) => assert!(
                (1..=m_draft).contains(&submitted),
                "adaptive submit_len {submitted} outside 1..={m_draft} at round {r}"
            ),
            None => assert_eq!(
                submitted, m_draft,
                "NV_DSPARK_ADAPTIVE must be unset on the fixed arm at round {r}"
            ),
        }
        let verify_rows = submitted + 1;
        let batch =
            build_chain_batch(anchor, &blk.ids[..submitted], verify_rows, true).expect("verify batch");
        assert_eq!(batch.len(), verify_rows);
        let (t, p) = rows_tensors(&batch, l, device);
        cache.set_capture_lin_ckpts(true);
        let (vlogits, _vhid, vaux) = model
            .forward_with_cache_dispatched_hidden_rows_aux(&t, &p, &mut cache, None, None, aux_layers)
            .unwrap_or_else(|e| panic!("m={verify_rows} verify at round {r}: {e:#}"));
        let greedy: Vec<u32> = vlogits
            .argmax(D::Minus1)
            .and_then(|t| t.flatten_all())
            .and_then(|t| t.to_dtype(DType::U32))
            .and_then(|t| t.to_vec1())
            .expect("verify argmax");
        let acc = accept_prefix_argmax(&batch, &greedy).expect("exact u32 accept");
        let consumed = acc.commit_len;
        let feat = Tensor::cat(&vaux, 1).expect("verify aux concat");
        let feat = feat.narrow(0, 0, consumed).expect("committed aux rows");
        let proj = drafter.project_aux(&feat).expect("project_aux");
        let positions: Vec<u32> = (l as u32..(l + consumed) as u32).collect();
        drafter
            .append_context_kv(&mut ctx, &proj, &positions)
            .expect("append committed rows to the drafter context");
        cache.set_current_len(l + consumed);
        cache.rollback_lin_to(consumed).expect("lin rollback");
        anchor = acc.next_bonus;
        emitted_ids.extend_from_slice(&batch[..consumed]);
        recs.push(RoundRec {
            submitted,
            accepted: acc.draft_accepted,
            probs: blk.accept_probs.clone(),
        });
    }
    let _ = device.synchronize();
    let rounds_s = rounds_t0.elapsed().as_secs_f64();
    std::env::remove_var("NV_DSPARK_ADAPTIVE");
    ArmResult {
        recs,
        emitted_ids,
        prefill_s,
        rounds_s,
        end_len: cache.current_len(),
    }
}

fn report_arm(depth: usize, arm: &str, res: &ArmResult) -> f64 {
    let n = res.recs.len();
    let accepted_total: usize = res.recs.iter().map(|r| r.accepted).sum();
    let submitted_total: usize = res.recs.iter().map(|r| r.submitted).sum();
    let emitted_total = accepted_total
        + n * EMITTED_PER_ROUND_IS_ACCEPTED_DRAFTS_PLUS_ONE_MATCHING_THE_REFERENCES_ACCEPTANCE_LENGTH_PLUS_ONE;
    let verify_rows_total = submitted_total + n;
    let mut hist: BTreeMap<usize, usize> = BTreeMap::new();
    let mut submit_hist: BTreeMap<usize, usize> = BTreeMap::new();
    for r in &res.recs {
        *hist.entry(r.accepted).or_default() += 1;
        *submit_hist.entry(r.submitted).or_default() += 1;
    }
    let q = QUARTILE_SPLIT_4_EXPOSES_GREEDY_DEGENERATION_ACROSS_THE_COUNTED_WINDOW;
    let quartiles: Vec<String> = (0..q)
        .map(|i| {
            let lo = i * n / q;
            let hi = (i + 1) * n / q;
            let chunk = &res.recs[lo..hi];
            let acc: usize = chunk.iter().map(|r| r.accepted).sum();
            format!(
                "q{i}={:.2}",
                (acc + chunk.len()) as f64 / chunk.len().max(1) as f64
            )
        })
        .collect();
    let mean_emitted = emitted_total as f64 / n as f64;
    let e = &res.emitted_ids;
    let last_q = &e[e.len() - e.len() / QUARTILE_SPLIT_4_EXPOSES_GREEDY_DEGENERATION_ACROSS_THE_COUNTED_WINDOW..];
    eprintln!(
        "[DSPARK-ACC] depth={depth} arm={arm} rounds={n} end_len={} prefill_s={:.1} rounds_s={:.1} \
         mean_accepted_drafts={:.3} mean_emitted_per_round={mean_emitted:.3} \
         accept_rate_over_submitted={:.4} mean_submitted={:.3} \
         emitted_per_verify_row={:.3} distinct_4gram={:.3} distinct_4gram_last_quarter={:.3} \
         accept_hist={hist:?} submit_hist={submit_hist:?} \
         emitted_quartiles=[{}]",
        res.end_len,
        res.prefill_s,
        res.rounds_s,
        accepted_total as f64 / n as f64,
        accepted_total as f64 / submitted_total.max(1) as f64,
        submitted_total as f64 / n as f64,
        emitted_total as f64 / verify_rows_total.max(1) as f64,
        distinct_ngram_fraction(e),
        distinct_ngram_fraction(last_q),
        quartiles.join(",")
    );
    mean_emitted
}

fn report_tail_text(depth: usize, arm: &str, res: &ArmResult, tok: &tokenizers::Tokenizer) {
    let text = tok.decode(&res.emitted_ids, true).unwrap_or_default();
    let tail: String = text
        .chars()
        .rev()
        .take(DECODED_TAIL_CHARS_300_MAKE_DEGENERATION_VISIBLE_IN_THE_LOG)
        .collect::<Vec<char>>()
        .into_iter()
        .rev()
        .collect();
    eprintln!("[DSPARK-TAIL] depth={depth} arm={arm} tail={tail:?}");
}

fn report_calibration(depth: usize, res: &ArmResult, m_draft: usize) {
    let buckets = CALIBRATION_BUCKETS_10_OF_WIDTH_0_1_OVER_PREDICTED_P;
    let mut bucket_reached = vec![(0usize, 0usize, 0f64); buckets];
    let mut bucket_all = vec![(0usize, 0usize, 0f64); buckets];
    let mut pos_n = vec![0usize; m_draft];
    let mut pos_acc = vec![0usize; m_draft];
    let mut pos_psum = vec![0f64; m_draft];
    for r in &res.recs {
        for (j, &p) in r.probs.iter().enumerate() {
            let b = ((p * buckets as f32) as usize).min(buckets - 1);
            let accepted = j < r.accepted;
            let reached = j <= r.accepted;
            bucket_all[b].0 += 1;
            bucket_all[b].1 += usize::from(accepted);
            bucket_all[b].2 += p as f64;
            if reached {
                bucket_reached[b].0 += 1;
                bucket_reached[b].1 += usize::from(accepted);
                bucket_reached[b].2 += p as f64;
            }
            pos_n[j] += 1;
            pos_acc[j] += usize::from(accepted);
            pos_psum[j] += p as f64;
        }
    }
    for b in 0..buckets {
        let (nr, ar, pr) = bucket_reached[b];
        let (na, aa, pa) = bucket_all[b];
        eprintln!(
            "[DSPARK-CAL] depth={depth} bucket={:.1}-{:.1} \
             reached_n={nr} reached_mean_p={:.3} reached_empirical={:.3} \
             all_n={na} all_mean_p={:.3} all_empirical={:.3}",
            b as f64 / buckets as f64,
            (b + 1) as f64 / buckets as f64,
            pr / nr.max(1) as f64,
            ar as f64 / nr.max(1) as f64,
            pa / na.max(1) as f64,
            aa as f64 / na.max(1) as f64,
        );
    }
    for j in 0..m_draft {
        eprintln!(
            "[DSPARK-CAL-POS] depth={depth} pos={j} n={} mean_predicted_p={:.3} \
             empirical_accept={:.3}",
            pos_n[j],
            pos_psum[j] / pos_n[j].max(1) as f64,
            pos_acc[j] as f64 / pos_n[j].max(1) as f64,
        );
    }
}

#[test]
#[ignore = "loads the ~22.6 GB Qwen3.8-27B NVFP4 target plus the 2.7 GB headless DSpark drafter, \
            primes a raw wikitext context at depths 256 and 8192 through the trunk aux tap, then \
            counts exact-u32-argmax acceptance over NV_DSPARK_ROUNDS (default 200) speculative \
            rounds per depth: one fixed-block arm plus NV_DSPARK_ADAPTIVE at 0.3/0.5/0.7. \
            NV_DSPARK_TEST=1 NV_DSPARK_TARGET_DIR NV_DSPARK_DRAFT_DIR NV_DSPARK_CORPUS required; \
            acceptance instrument only, wall time is not the metric here"]
fn qwen38_dspark_accept_at_256_and_8192_fixed_and_adaptive_arms_vs_the_4_5_emitted_bar() {
    gate_loud();
    let target_dir = env_dir("NV_DSPARK_TARGET_DIR");
    let draft_dir = env_dir("NV_DSPARK_DRAFT_DIR");
    let corpus = std::env::var("NV_DSPARK_CORPUS").unwrap_or_else(|_| {
        panic!(
            "PRECONDITION NOT MET, THIS TEST EXECUTED NOTHING: set NV_DSPARK_CORPUS to a raw text \
             file holding at least the deepest rung's token count; a tiled paragraph reads as an \
             upper bracket, not an acceptance number"
        )
    });
    let rounds_wanted = env_usize(
        "NV_DSPARK_ROUNDS",
        ACCEPT_ROUNDS_200_DEFAULT_SO_THE_HISTOGRAM_HAS_MASS_IN_EVERY_BIN,
    );
    let device = Device::new_cuda_with_stream(0).expect("cuda");
    let tok = tokenizers::Tokenizer::from_file(target_dir.join("tokenizer.json"))
        .expect("target tokenizer.json");
    let text = std::fs::read_to_string(&corpus).expect("corpus file");
    let all: Vec<u32> = tok
        .encode(text.as_str(), false)
        .expect("encode corpus")
        .get_ids()
        .to_vec();
    let deepest = *ACCEPT_DEPTHS_256_CONTROL_AND_8192_THE_BAR_RUNG
        .iter()
        .max()
        .expect("non-empty depth list");
    let max_offset = *FIXED_ARM_CORPUS_OFFSETS_0_AND_100K_GIVE_TWO_INDEPENDENT_GREEDY_TRAJECTORIES_SO_THE_VERDICT_IS_NOT_ONE_TEXT_SEGMENT
        .iter()
        .max()
        .expect("non-empty offset list");
    assert!(
        all.len() >= max_offset + deepest,
        "corpus {corpus} holds {} tokens but the deepest rung at the last offset needs {}",
        all.len(),
        max_offset + deepest
    );

    let model = load_target(&target_dir, &device);
    let drafter = load_dspark(&draft_dir, &target_dir, &device);
    let m_draft = drafter.config().query_rows();
    let aux_layers = model
        .aux_layer_ids_are_post_layer_residuals_because_hf_hidden_states_index_l_plus_1_is_the_output_of_layer_l(
            &drafter.config().aux_hidden_state_layer_ids,
        )
        .expect("the DSpark target_layer_ids must be in range for this trunk");

    let mut fixed_emitted_by_depth: BTreeMap<usize, Vec<f64>> = BTreeMap::new();
    for &depth in &ACCEPT_DEPTHS_256_CONTROL_AND_8192_THE_BAR_RUNG {
        for &off in
            &FIXED_ARM_CORPUS_OFFSETS_0_AND_100K_GIVE_TWO_INDEPENDENT_GREEDY_TRAJECTORIES_SO_THE_VERDICT_IS_NOT_ONE_TEXT_SEGMENT
        {
            let prompt = &all[off..off + depth];
            let fixed = run_arm(
                &model,
                &drafter,
                &aux_layers,
                prompt,
                rounds_wanted,
                None,
                &device,
            );
            assert_eq!(
                fixed.recs.len(),
                rounds_wanted,
                "the counted window must be exactly the requested rounds"
            );
            let arm = format!("fixed@{off}");
            let mean_emitted = report_arm(depth, &arm, &fixed);
            report_tail_text(depth, &arm, &fixed, &tok);
            if off == 0 {
                report_calibration(depth, &fixed, m_draft);
            }
            fixed_emitted_by_depth.entry(depth).or_default().push(mean_emitted);
        }
        let prompt = &all[..depth];
        for &t in &ADAPTIVE_THRESHOLDS_BRACKET_THE_CUMULATIVE_CONFIDENCE_KNEE {
            let adaptive = run_arm(
                &model,
                &drafter,
                &aux_layers,
                prompt,
                rounds_wanted,
                Some(t),
                &device,
            );
            let arm = format!("adaptive-{t}");
            report_arm(depth, &arm, &adaptive);
            report_tail_text(depth, &arm, &adaptive, &tok);
        }
    }

    let deep_fixed = fixed_emitted_by_depth
        .get(&8192)
        .expect("the 8192 rung is the bar rung and must have run");
    let at_8192 = deep_fixed.iter().sum::<f64>() / deep_fixed.len() as f64;
    eprintln!(
        "[DSPARK-VERDICT] fixed_mean_emitted_per_round_at_8192={at_8192:.3} per_offset={deep_fixed:?} \
         bar_serving_integration_worthwhile={BAR_SERVING_INTEGRATION_WORTHWHILE_MEAN_EMITTED_AT_8192} \
         m8_verify_breakeven={BAR_M8_VERIFY_BREAKEVEN_EMITTED_PER_ROUND} \
         mtp_k3_on_record={BAR_MTP_K3_EMITTED_PER_ROUND_ON_RECORD_LOW}-{BAR_MTP_K3_EMITTED_PER_ROUND_ON_RECORD_HIGH} \
         llama_cpp_nextn_ceiling={BAR_LLAMA_CPP_NEXTN_CEILING_EMITTED_PER_ROUND} \
         external_ling_flash_dspark={BAR_EXTERNAL_LING_3_0_FLASH_DSPARK_ACCEPT_LENGTH_ON_A_COMPARABLE_HYBRID} \
         verdict={} basis=(drafter=RadixArk/Qwen3.8-27B-DSpark block_size=6 headless \
         target=unsloth/Qwen3.8-27B-NVFP4 greedy exact-u32 accept, eager m<=8 verify, raw wikitext \
         prime, emitted=accepted_drafts+1)",
        if at_8192 >= BAR_SERVING_INTEGRATION_WORTHWHILE_MEAN_EMITTED_AT_8192 {
            "SERVING-INTEGRATION-WORTHWHILE"
        } else {
            "BELOW-BAR"
        }
    );
}
