#![cfg(all(feature = "cuda", feature = "wgpu"))]

mod common;
use common::CHAT_WRAPPED_PROMPT_CONTEXT_TOKENS_256_THEN_SCORE_512_SO_EVERY_CHAT_PPL_FAMILY_SHARES_ONE_SLICE;
mod ppl_common;

use candle_core::{DType, Device, Tensor};
use nv_models::gemma4::Gemma4Config;
use nv_models::gemma4_e4b::Gemma4E4b;
use nv_models::gemma4_e4b_wgpu::Gemma4E4bWgpu;
use ppl_common::{
    corpus_text_from_nv_ppl_corpus_env_failing_loudly,
    require_nv_ppl_test_gate_because_this_suite_must_never_silently_skip,
};
use tokenizers::Tokenizer;

const CUDA_INSTRUMENT_REPOS: [&str; 1] = ["models--google--gemma-4-E4B-it-qat-w4a16-ct"];

const WGPU_INSTRUMENT_REPOS: [&str; 2] = [
    "models--google--gemma-4-E4B-it-qat-w4a16-ct",
    "models--google--gemma-4-E4B-it",
];

const CHAT_WRAPPED_SCORED_CONTINUATION_TOKENS: usize = 512;

const SERVING_PREFILL_CHUNK_512_MATCHES_RUN_SAMPLING_GEMMA4_E4B: usize = 512;

const CATASTROPHIC_NLL_NATS_10_IS_FAR_ABOVE_THIS_SLICES_HEALTHY_TAIL: f32 = 10.0;

const CROSS_BACKEND_PER_POSITION_NLL_BAND_10_NATS_QUANT_KERNEL_DIFFS_READ_UNDER_3_WHILE_THE_UNCAPPED_DEFECT_READ_40_TO_55:
    f32 = 10.0;

const CROSS_BACKEND_MEAN_NLL_BAND_0P5_NATS_THE_UNCAPPED_DEFECT_MOVED_THE_MEAN_BY_3P75: f64 = 0.5;

const TOP_K_5_LOGITS_PER_DUMPED_POSITION: usize = 5;

struct PosStat {
    nll: f32,
    argmax: u32,
    true_logit: f32,
    max_logit: f32,
    min_logit: f32,
    top5: Vec<(u32, f32)>,
}

fn nll_and_argmax(row: &[f32], t: usize) -> (f32, u32) {
    let mut m = f32::NEG_INFINITY;
    let mut argmax = 0usize;
    for (j, &v) in row.iter().enumerate() {
        assert!(v.is_finite(), "non-finite logit");
        if v > m {
            m = v;
            argmax = j;
        }
    }
    let mut denom = 0f32;
    for &v in row {
        denom += (v - m).exp();
    }
    (-((row[t] - m) - denom.ln()), argmax as u32)
}

fn top_k(row: &[f32], k: usize) -> Vec<(u32, f32)> {
    let mut best: Vec<(u32, f32)> = Vec::with_capacity(k + 1);
    for (j, &v) in row.iter().enumerate() {
        if best.len() < k || v > best.last().expect("nonempty").1 {
            let at = best.partition_point(|&(_, bv)| bv >= v);
            best.insert(at, (j as u32, v));
            best.truncate(k);
        }
    }
    best
}

fn stat_from_row(row: &[f32], true_id: u32) -> PosStat {
    let t = true_id as usize;
    let (nll, argmax) = nll_and_argmax(row, t);
    PosStat {
        nll,
        argmax,
        true_logit: row[t],
        max_logit: row.iter().cloned().fold(f32::NEG_INFINITY, f32::max),
        min_logit: row.iter().cloned().fold(f32::INFINITY, f32::min),
        top5: top_k(row, TOP_K_5_LOGITS_PER_DUMPED_POSITION),
    }
}

fn fnv1a64(ids: &[u32]) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325;
    for &id in ids {
        for b in id.to_le_bytes() {
            h ^= b as u64;
            h = h.wrapping_mul(0x100000001b3);
        }
    }
    h
}

fn ids_and_score_start_for(
    repos: &[&str],
    corpus: &str,
) -> (std::path::PathBuf, Vec<u32>, usize) {
    let dir = common::snapshot_dir_env_override_then_home_hub_first_repo_with_config_and_weights(
        "NV_E4B_SNAPSHOT",
        repos,
    );
    let tok = Tokenizer::from_file(dir.join("tokenizer.json")).expect("tokenizer.json");
    let bos = tok.token_to_id("<bos>").expect("bos");
    let (ids, score_start) =
        ppl_common::chat_wrapped_continuation_ids_and_copy_start_rendered_through_the_snapshot_jinja(
            &dir,
            &tok,
            corpus,
            bos,
            CHAT_WRAPPED_PROMPT_CONTEXT_TOKENS_256_THEN_SCORE_512_SO_EVERY_CHAT_PPL_FAMILY_SHARES_ONE_SLICE,
            CHAT_WRAPPED_SCORED_CONTINUATION_TOKENS,
        );
    (dir, ids, score_start)
}

fn cuda_stats_via_forward_step_spec_chunked(
    m: &Gemma4E4b,
    ids: &[u32],
    score_start: usize,
) -> Vec<(usize, PosStat)> {
    let vocab = m.config().vocab_size;
    let n_layers = m.config().num_hidden_layers;
    let mut cache: Vec<Option<(Tensor, Tensor)>> = vec![None; n_layers];
    let mut out = Vec::new();
    let chunk = SERVING_PREFILL_CHUNK_512_MATCHES_RUN_SAMPLING_GEMMA4_E4B;
    let mut off = 0usize;
    while off < ids.len() {
        let c = (ids.len() - off).min(chunk);
        let (logits, _hidden) = m
            .forward_step_spec(&ids[off..off + c], off, &mut cache)
            .unwrap_or_else(|e| panic!("prefill chunk at {off}: {e:#}"));
        if (off..off + c).any(|p| p + 1 >= score_start && p + 1 < ids.len()) {
            let rows: Vec<Vec<f32>> = logits
                .to_dtype(DType::F32)
                .expect("f32 logits")
                .to_vec2()
                .expect("host rows");
            assert_eq!(rows[0].len(), vocab, "full vocab rows");
            for (i, row) in rows.iter().enumerate() {
                let p = off + i;
                if p + 1 >= score_start && p + 1 < ids.len() {
                    out.push((p, stat_from_row(row, ids[p + 1])));
                }
            }
        }
        off += c;
    }
    out
}

fn wgpu_stats_via_decode_step_logits(
    dir: &std::path::Path,
    ids: &[u32],
    score_start: usize,
) -> Vec<(usize, PosStat)> {
    let ctx = nv_kernels::wgpu_backend::WgpuContext::shared()
        .expect("wgpu adapter required: this gated probe must never silently skip");
    eprintln!("adapter: {}", ctx.summary());
    let config = Gemma4Config::from_hf_json_file(&dir.join("config.json")).expect("config.json");
    let loader =
        nv_weights::WeightLoader::open_dir(dir, &candle_core::Device::Cpu).expect("weights");
    let mut m = Gemma4E4bWgpu::from_loader(config, &loader, ids.len() + 8).expect("wgpu model");
    drop(loader);
    let vocab = m.config().vocab_size;
    let mut out = Vec::new();
    for p in 0..ids.len() - 1 {
        let row = m
            .decode_step_logits(ids[p])
            .unwrap_or_else(|err| panic!("wgpu decode step {p}: {err:#}"))
            .1;
        if p + 1 < score_start {
            continue;
        }
        assert_eq!(row.len(), vocab, "full vocab row");
        out.push((p, stat_from_row(&row, ids[p + 1])));
    }
    out
}

fn mean_nll(stats: &[(usize, PosStat)]) -> f64 {
    stats.iter().map(|(_, s)| s.nll as f64).sum::<f64>() / stats.len() as f64
}

fn token_class(text: &str) -> &'static str {
    if text.chars().all(|c| c.is_ascii_digit()) {
        "digits"
    } else if text.trim().is_empty() {
        "whitespace"
    } else if text.chars().any(|c| !c.is_ascii()) {
        "non-ascii"
    } else {
        "word"
    }
}

#[test]
#[ignore = "loads the E4B checkpoint twice (cuda then wgpu) to score the shared chat-wrapped \
            slice position-by-position on both serving paths; set NV_PPL_TEST=1 and \
            NV_PPL_CORPUS -- the cuda model scores the slice twice through \
            forward_step_spec chunk-512 (once with the capped serving default, once with \
            NV_E4B_STEP_LOGITS_UNCAPPED=1), the wgpu model scores it through \
            decode_step_logits, and the gate asserts the capped cuda default sits inside the \
            cross-backend band at every position, so a regression to raw lm_head rows \
            (705-vs-16 family ppl, 40-55 nat positions) fails loudly instead of hiding \
            behind matching top-1 accuracy"]
fn e4b_same_slice_per_position_nll_capped_cuda_serving_default_stays_in_the_wgpu_band() {
    require_nv_ppl_test_gate_because_this_suite_must_never_silently_skip();
    let corpus = corpus_text_from_nv_ppl_corpus_env_failing_loudly();

    let (cuda_dir, cuda_ids, cuda_start) = ids_and_score_start_for(&CUDA_INSTRUMENT_REPOS, &corpus);
    let (wgpu_dir, wgpu_ids, wgpu_start) = ids_and_score_start_for(&WGPU_INSTRUMENT_REPOS, &corpus);
    eprintln!(
        "LABEL-STREAM cuda dir={} n={} score_start={} fnv={:016x}",
        cuda_dir.display(),
        cuda_ids.len(),
        cuda_start,
        fnv1a64(&cuda_ids)
    );
    eprintln!(
        "LABEL-STREAM wgpu dir={} n={} score_start={} fnv={:016x}",
        wgpu_dir.display(),
        wgpu_ids.len(),
        wgpu_start,
        fnv1a64(&wgpu_ids)
    );
    assert_eq!(cuda_dir, wgpu_dir, "the two instruments resolve different snapshots");
    assert_eq!(
        (&cuda_ids, cuda_start),
        (&wgpu_ids, wgpu_start),
        "the two instruments score different label streams"
    );

    let cfg = Gemma4Config::from_hf_json_file(&cuda_dir.join("config.json")).expect("config.json");
    let cap = cfg.final_logit_softcapping;
    assert!(
        cap > 0.0 && cap.is_finite(),
        "this probe exists because the checkpoint declares final_logit_softcapping={cap}"
    );
    let tok = Tokenizer::from_file(cuda_dir.join("tokenizer.json")).expect("tokenizer.json");

    let uncap_env = nv_models::gemma4_e4b::NV_E4B_STEP_LOGITS_UNCAPPED_ENV_OPT_IN_1_RESTORES_THE_RAW_LM_HEAD_STEP_ROWS_THAT_SKIPPED_THE_CONFIGS_FINAL_LOGIT_SOFTCAPPING_WHICH_TRACE_WGPU_AND_GEMMA4_31B_ALL_APPLY_TANH_IS_MONOTONE_SO_GREEDY_OUTPUT_NEVER_MOVED_BUT_SAMPLING_LOGPROBS_AND_PPL_READ_THE_UNCAPPED_DISTRIBUTION_705_VS_16_ON_THE_CHAT_SLICE;
    std::env::remove_var(uncap_env);
    assert!(
        !nv_models::gemma4_e4b::step_logits_uncapped_env_opted_in(),
        "the default arm must score with the capped serving default"
    );
    let device = Device::new_cuda_with_stream(0).expect("cuda");
    let ccfg = Gemma4Config::from_hf_json_file(&cuda_dir.join("config.json")).expect("config.json");
    assert!(ccfg.has_per_layer_embeddings(), "not an E4B checkpoint: {cuda_dir:?}");
    let weights = nv_weights::WeightLoader::open_dir(&cuda_dir, &device).expect("weights");
    let m = Gemma4E4b::from_loader(ccfg, &weights, &device).expect("load Gemma4E4b");
    let cuda_capped = cuda_stats_via_forward_step_spec_chunked(&m, &cuda_ids, cuda_start);
    std::env::set_var(uncap_env, "1");
    assert!(
        nv_models::gemma4_e4b::step_logits_uncapped_env_opted_in(),
        "the characterization arm must score the raw pre-fix rows"
    );
    let cuda_raw = cuda_stats_via_forward_step_spec_chunked(&m, &cuda_ids, cuda_start);
    std::env::remove_var(uncap_env);
    drop(m);

    let wgpu = wgpu_stats_via_decode_step_logits(&wgpu_dir, &wgpu_ids, wgpu_start);
    assert_eq!(cuda_capped.len(), wgpu.len(), "backends scored different position counts");
    assert_eq!(cuda_capped.len(), cuda_raw.len(), "arms scored different position counts");
    for ((pc, _), (pw, _)) in cuda_capped.iter().zip(&wgpu) {
        assert_eq!(pc, pw, "backends scored different positions");
    }

    eprintln!(
        "MEAN-NLL cuda_capped_default={:.4} cuda_uncapped_env={:.4} wgpu={:.4} positions={} cap={cap}",
        mean_nll(&cuda_capped),
        mean_nll(&cuda_raw),
        mean_nll(&wgpu),
        cuda_capped.len()
    );
    eprintln!(
        "PPL cuda_capped_default={:.3} cuda_uncapped_env={:.3} wgpu={:.3}",
        mean_nll(&cuda_capped).exp(),
        mean_nll(&cuda_raw).exp(),
        mean_nll(&wgpu).exp()
    );

    let mut n_cat = 0usize;
    for (i, (p, r)) in cuda_raw.iter().enumerate() {
        if r.nll <= CATASTROPHIC_NLL_NATS_10_IS_FAR_ABOVE_THIS_SLICES_HEALTHY_TAIL {
            continue;
        }
        n_cat += 1;
        let c = &cuda_capped[i].1;
        let w = &wgpu[i].1;
        let true_id = cuda_ids[p + 1];
        let text = tok.decode(&[true_id], false).unwrap_or_default();
        let argmax_text = tok.decode(&[r.argmax], false).unwrap_or_default();
        eprintln!(
            "CAT p={p} true_id={true_id} true_text={text:?} class={} cuda_nll_raw={:.2} \
             cuda_nll_capped={:.2} wgpu_nll={:.2} cuda_true_logit_raw={:.2} \
             cuda_max_logit_raw={:.2} cuda_min_logit_raw={:.2} cuda_argmax={} \
             argmax_text={argmax_text:?} wgpu_argmax={} cuda_top5_raw={:?} wgpu_top5={:?}",
            token_class(&text),
            r.nll,
            c.nll,
            w.nll,
            r.true_logit,
            r.max_logit,
            r.min_logit,
            r.argmax,
            w.argmax,
            r.top5,
            w.top5
        );
    }
    eprintln!(
        "CAT-SUMMARY uncapped_catastrophic_positions={n_cat} of {} (raw nll > {} nats)",
        cuda_raw.len(),
        CATASTROPHIC_NLL_NATS_10_IS_FAR_ABOVE_THIS_SLICES_HEALTHY_TAIL
    );

    let mean_gap = (mean_nll(&cuda_capped) - mean_nll(&wgpu)).abs();
    assert!(
        mean_gap < CROSS_BACKEND_MEAN_NLL_BAND_0P5_NATS_THE_UNCAPPED_DEFECT_MOVED_THE_MEAN_BY_3P75,
        "capped cuda serving default drifted {mean_gap:.3} mean nats from the wgpu family scorer"
    );
    for (i, (p, c)) in cuda_capped.iter().enumerate() {
        let w = &wgpu[i].1;
        assert!(
            (c.nll - w.nll).abs()
                < CROSS_BACKEND_PER_POSITION_NLL_BAND_10_NATS_QUANT_KERNEL_DIFFS_READ_UNDER_3_WHILE_THE_UNCAPPED_DEFECT_READ_40_TO_55,
            "p={p}: capped cuda nll {:.2} vs wgpu nll {:.2} is outside the cross-backend band; \
             a gap this size is the raw-lm-head signature, not kernel noise",
            c.nll,
            w.nll
        );
    }
    assert!(
        mean_nll(&cuda_raw) > mean_nll(&cuda_capped),
        "the NV_E4B_STEP_LOGITS_UNCAPPED arm no longer restores the raw pre-fix rows"
    );
}
