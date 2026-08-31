#![cfg(feature = "cuda")]
#![allow(dead_code)]

mod common;
mod ctx_timing_common;
mod official_template;
mod ppl_common;

use candle_core::{DType, Device, Tensor};
use official_template::OfficialTemplate;
use std::path::Path;

const SCORED_TAIL_TARGETS_128_THE_SAME_CORPUS_WINDOW_AT_EVERY_RUNG_SO_DEEPER_CONTEXT_MUST_NOT_SCORE_WORSE:
    usize = 128;
const FREERUN_STEPS_64_ENOUGH_TO_EXPOSE_DEGENERATION_AND_SHORT_CYCLES: usize = 64;
const FREERUN_MIN_DISTINCT_RATIO_0_20_WIKITEXT_CONTINUATION_IS_LOOPIER_THAN_CHAT_BUT_NOT_DEGENERATE:
    f64 = 0.20;
const FREERUN_MAX_IMMEDIATE_RUN_12_WIKITEXT_HEADINGS_REPEAT_EQUALS_SIGNS_LEGITIMATELY: usize = 12;
const FREERUN_CYCLE_MAX_PERIOD_16_MIN_REPEATS_4: (usize, usize) = (16, 4);
const TF_PPL_DEPTH_CLIFF_RATIO_1_5_SAME_TARGETS_WITH_MORE_CONTEXT_MUST_NOT_READ_50_PCT_WORSE: f64 =
    1.5;
const TF_PPL_DOC_NATIVE_SANE_CEILING_50_PASSING_FAMILIES_READ_6_TO_23_WHILE_A_BROKEN_ARM_OR_A_CHAT_FRAME_OOD_MODEL_READS_HUNDREDS_TO_THOUSANDS:
    f64 = 50.0;
const TF_PHASE_SKIP_GEMMA4_TASK_94_RULES_RAW_CORPUS_GEMMA4_TF_PPL_AN_INVALID_EVAL_IN_ANY_FRAME:
    &str =
    "task #94 (ppl_gemma4.rs) rules raw-corpus gemma4 teacher forcing an invalid eval in any \
     frame: 1980-26388 ppl across template/doc-native/BOS variants with softcap-consistent \
     worst_nll ~50; this family's depth evidence is the official-template freerun + serving-arm \
     phases below plus the chat-wrapped ppl_gemma4 instrument, not a TF band";
const TF_PHASE_SKIP_GPT_OSS_RAW_TEXT_DEPTH_DEGRADATION_IS_MODEL_GENUINE_VLLM_CROSS_CHECKED:
    &str =
    "gpt-oss-20b degrades on long unstructured text in ANY engine: vLLM 0.26.0 on the same \
     openai/gpt-oss-20b checkpoint reads ppl 108.7 at depth 256 and 2049 at 8192, worse than \
     this arm's own doc-native ladder, so a TF depth band here would gate the model, not the \
     engine; this family's depth evidence is the official-template freerun + serving-arm phases \
     below";
const PREFILL_CHUNK_512_THE_PROVEN_QWEN_BLOCK: usize = 512;
const PREFILL_CHUNK_256_RING_LEGAL_FOR_SLIDING_CACHES: usize = 256;
const KV_SLOT_SLACK_1024_COVERS_SUFFIX_TF_AND_FREERUN_STEPS: usize = 1024;
const CORPUS_HEAD_CHARS_3M_TOKENIZES_PAST_196K_QWEN_TOKENS_WITHOUT_EATING_THE_WHOLE_TRAIN_FILE:
    usize = 3_000_000;
const DEPTH_LADDER_262144_MAX: [usize; 4] = [256, 8192, 65536, 196608];
const DEPTH_LADDER_131072_MAX: [usize; 4] = [256, 8192, 65536, 122880];
const DOC_SPLICE_MARKER_NEVER_IN_WIKITEXT: &str = "NVDEPTHDOCMARKER7391";
const DEFAULT_CORPUS: &str = "/home/user/tmp/wf/q38/xppl/wikitext-2-raw/wiki.train.raw";

fn suite_gate_runs_or_skips_loudly(rung: &str) -> bool {
    if std::env::var("NV_DEPTH_CORRECT").ok().as_deref() == Some("1") {
        return true;
    }
    if std::env::var("NV_MODELS_ALLOW_SKIP").as_deref() != Ok("1") {
        panic!(
            "set NV_DEPTH_CORRECT=1 to run the {rung} depth-correctness sweep, or \
             NV_MODELS_ALLOW_SKIP=1 to skip it on purpose; a depth gate that silently reports \
             ok would hide exactly the failure class task #106 exists to catch"
        );
    }
    eprintln!("SKIP (NV_MODELS_ALLOW_SKIP=1): set NV_DEPTH_CORRECT=1 to run {rung}");
    false
}

fn gate_sets_the_arm_env_because_the_gate_documents_what_serving_should_run(key: &str, why: &str) {
    if let Ok(v) = std::env::var(key) {
        assert_eq!(
            v, "1",
            "{key} is explicitly set to {v:?} but this depth-correctness rung documents that \
             serving must run it: {why}"
        );
    }
    unsafe { std::env::set_var(key, "1") };
}

fn depth_ladder(default: &[usize]) -> Vec<usize> {
    ctx_timing_common::nv_ctx_tokens_env_with_k_suffix().unwrap_or_else(|| default.to_vec())
}

fn corpus_ids_head_of_train_file(tok: &tokenizers::Tokenizer, vocab: usize) -> Vec<u32> {
    let path =
        std::env::var("NV_DEPTH_CORPUS").unwrap_or_else(|_| DEFAULT_CORPUS.to_string());
    let text = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("read corpus {path}: {e}; set NV_DEPTH_CORPUS"));
    let head: String = text
        .chars()
        .take(CORPUS_HEAD_CHARS_3M_TOKENIZES_PAST_196K_QWEN_TOKENS_WITHOUT_EATING_THE_WHOLE_TRAIN_FILE)
        .collect();
    let ids: Vec<u32> = tok
        .encode(head.as_str(), false)
        .expect("encode corpus head")
        .get_ids()
        .to_vec();
    let max_id = *ids.iter().max().expect("nonempty corpus") as usize;
    assert!(
        max_id < vocab,
        "corpus token id {max_id} out of model vocab {vocab}; wrong tokenizer for this checkpoint"
    );
    ids
}

fn template_spliced_prefix_suffix_ids(
    dir: &Path,
    tok: &tokenizers::Tokenizer,
) -> (Vec<u32>, Vec<u32>) {
    let user = format!(
        "{DOC_SPLICE_MARKER_NEVER_IN_WIKITEXT}\n\nContinue the passage above from exactly \
         where it stops, matching its style."
    );
    let rendered = OfficialTemplate::load(dir).render_user(&user);
    let (prefix_str, suffix_str) = rendered
        .split_once(DOC_SPLICE_MARKER_NEVER_IN_WIKITEXT)
        .unwrap_or_else(|| {
            panic!(
                "official template render lost the splice marker; rendered head: {:?}",
                &rendered[..rendered.len().min(200)]
            )
        });
    let prefix: Vec<u32> = tok
        .encode(prefix_str, false)
        .expect("encode template prefix")
        .get_ids()
        .to_vec();
    let suffix: Vec<u32> = tok
        .encode(suffix_str, false)
        .expect("encode template suffix")
        .get_ids()
        .to_vec();
    assert!(
        !suffix.is_empty(),
        "template suffix is empty: the render carries no generation prompt after the document"
    );
    (prefix, suffix)
}

fn doc_native_tf_prefix_bos_iff_declared_because_chat_rl_models_read_chat_framed_wikitext_at_ppl_in_the_thousands_reproduced_on_vllm(
    dir: &Path,
    tok: &tokenizers::Tokenizer,
) -> Vec<u32> {
    std::fs::read_to_string(dir.join("tokenizer_config.json"))
        .ok()
        .and_then(|s| serde_json::from_str::<serde_json::Value>(&s).ok())
        .and_then(|c| official_template::token_str(c.get("bos_token")?))
        .map(|b| {
            tok.encode(b.as_str(), false)
                .expect("encode bos")
                .get_ids()
                .to_vec()
        })
        .unwrap_or_default()
}

fn eos_ids_from_generation_config(dir: &Path) -> Vec<u32> {
    let raw = std::fs::read_to_string(dir.join("generation_config.json"))
        .unwrap_or_else(|e| panic!("read generation_config.json in {dir:?}: {e}"));
    let v: serde_json::Value = serde_json::from_str(&raw).expect("parse generation_config.json");
    let ids = match v.get("eos_token_id") {
        Some(serde_json::Value::Number(n)) => vec![n.as_u64().expect("eos id") as u32],
        Some(serde_json::Value::Array(a)) => a
            .iter()
            .map(|x| x.as_u64().expect("eos id") as u32)
            .collect(),
        other => panic!("generation_config.json eos_token_id unusable: {other:?}"),
    };
    assert!(!ids.is_empty(), "empty eos_token_id in {dir:?}");
    ids
}

fn nonfinite_count(row: &[f32]) -> usize {
    row.iter().filter(|v| !v.is_finite()).count()
}

fn nll_of_target_f64_lse(row: &[f32], target: usize) -> f64 {
    let m = row.iter().fold(f32::NEG_INFINITY, |a, &v| a.max(v));
    assert!(m.is_finite(), "all-non-finite logit row");
    let lse = m as f64
        + row
            .iter()
            .map(|&v| ((v - m) as f64).exp())
            .sum::<f64>()
            .ln();
    lse - row[target] as f64
}

fn argmax_row(row: &[f32]) -> u32 {
    common::argmax(row)
}

fn freerun_next_token_with_eos_masked_like_serving_min_tokens_because_e4b_at_8192_legitimately_completes_the_cut_sentence_then_closes_the_turn_at_step_4_so_collapse_must_read_as_degenerate_forced_text_not_as_an_early_eos(
    row: &[f32],
    eos: &[u32],
) -> (u32, bool) {
    let unmasked = argmax_row(row);
    if !eos.contains(&unmasked) {
        return (unmasked, false);
    }
    let mut best_id = u32::MAX;
    let mut best_v = f32::NEG_INFINITY;
    for (i, &v) in row.iter().enumerate() {
        if v > best_v && !eos.contains(&(i as u32)) {
            best_v = v;
            best_id = i as u32;
        }
    }
    assert!(
        best_id != u32::MAX && best_v.is_finite(),
        "no non-EOS candidate in the logit row"
    );
    (best_id, true)
}

fn longest_immediate_run(ids: &[u32]) -> usize {
    let (mut best, mut cur) = (1usize, 1usize);
    for w in ids.windows(2) {
        cur = if w[0] == w[1] { cur + 1 } else { 1 };
        best = best.max(cur);
    }
    best
}

fn tail_cycle(ids: &[u32]) -> Option<(usize, usize)> {
    let (max_period, min_reps) = FREERUN_CYCLE_MAX_PERIOD_16_MIN_REPEATS_4;
    for k in 1..=max_period {
        if ids.len() < k * min_reps {
            break;
        }
        let tail = &ids[ids.len() - k..];
        let mut reps = 0usize;
        let mut j = ids.len();
        while j >= k && &ids[j - k..j] == tail {
            reps += 1;
            j -= k;
        }
        if reps >= min_reps {
            return Some((k, reps));
        }
    }
    None
}

trait DepthCtxEngine {
    fn family(&self) -> &'static str;
    fn prefill_chunk_len(&self) -> usize;
    fn reset_ctx(&mut self, depth: usize) -> anyhow::Result<()>;
    fn prefill_chunk_last_row(&mut self, ids: &[u32]) -> anyhow::Result<Vec<f32>>;
    fn decode_row(&mut self, token: u32) -> anyhow::Result<Vec<f32>>;
    fn arm_label(&self) -> String;
    fn assert_serving_arm_after_freerun(&self, _depth: usize) {}
}

struct TfRung {
    depth: usize,
    tf_ppl: f64,
    tf_worst_nll: f64,
}

struct FreerunRung {
    depth: usize,
    body_len: usize,
    distinct_ratio: f64,
    run: usize,
    cycle: Option<(usize, usize)>,
    first_eos_pressure_step: Option<usize>,
}

fn prefill_chunked_nonfinite_checked(
    eng: &mut dyn DepthCtxEngine,
    ids: &[u32],
    depth: usize,
    phase: &str,
) -> (Vec<f32>, f64) {
    let chunk = eng.prefill_chunk_len();
    let t = std::time::Instant::now();
    let mut last_row: Vec<f32> = Vec::new();
    for c in ids.chunks(chunk) {
        last_row = eng.prefill_chunk_last_row(c).unwrap_or_else(|e| {
            panic!("{} {phase} prefill chunk at depth {depth}: {e:#}", eng.family())
        });
    }
    let prefill_s = t.elapsed().as_secs_f64();
    assert_eq!(
        nonfinite_count(&last_row),
        0,
        "{} depth {depth}: non-finite logits at the end of the {phase} prefill",
        eng.family()
    );
    (last_row, prefill_s)
}

fn run_tf_rung_doc_native(
    eng: &mut dyn DepthCtxEngine,
    bos_prefix: &[u32],
    corpus: &[u32],
    base: usize,
    depth: usize,
) -> TfRung {
    let scored = SCORED_TAIL_TARGETS_128_THE_SAME_CORPUS_WINDOW_AT_EVERY_RUNG_SO_DEEPER_CONTEXT_MUST_NOT_SCORE_WORSE;
    let overhead = bos_prefix.len();
    assert!(
        depth > overhead + scored + 8,
        "depth {depth} cannot hold doc-native overhead {overhead} plus {scored} scored targets"
    );
    let n_doc = depth - overhead;
    assert!(
        base >= n_doc,
        "corpus anchor {base} shorter than the {n_doc} document tokens rung {depth} needs"
    );
    let doc = &corpus[base - n_doc..base];
    eng.reset_ctx(depth)
        .unwrap_or_else(|e| panic!("{} reset at depth {depth}: {e:#}", eng.family()));

    let prefill_ids: Vec<u32> = bos_prefix
        .iter()
        .chain(doc[..n_doc - scored].iter())
        .copied()
        .collect();
    let (mut row, prefill_s) =
        prefill_chunked_nonfinite_checked(eng, &prefill_ids, depth, "doc-native TF");

    let scored_ids = &doc[n_doc - scored..];
    let t_tf = std::time::Instant::now();
    let mut nll_sum = 0f64;
    let mut worst = 0f64;
    let mut nll_trace: Vec<f64> = Vec::new();
    let mut argmax_trace: Vec<u32> = Vec::new();
    for (i, &target) in scored_ids.iter().enumerate() {
        assert_eq!(
            nonfinite_count(&row),
            0,
            "{} depth {depth}: non-finite logits at teacher-forced position {i}",
            eng.family()
        );
        let nll = nll_of_target_f64_lse(&row, target as usize);
        nll_sum += nll;
        worst = worst.max(nll);
        nll_trace.push(nll);
        argmax_trace.push(argmax_row(&row));
        if i + 1 < scored_ids.len() {
            row = eng.decode_row(target).unwrap_or_else(|e| {
                panic!("{} teacher-forced decode at depth {depth}: {e:#}", eng.family())
            });
        }
    }
    let tf_s = t_tf.elapsed().as_secs_f64();
    let tf_ppl = (nll_sum / scored as f64).exp();
    if std::env::var("NV_DEPTH_DEBUG_TRACE").as_deref() == Ok("1") {
        eprintln!(
            "DEPTH-DEBUG-NLL model={} depth={depth}: {:?}",
            eng.family(),
            nll_trace.iter().map(|v| (*v * 100.0).round() / 100.0).collect::<Vec<f64>>()
        );
        eprintln!(
            "DEPTH-DEBUG-TF-TARGETS model={} depth={depth}: {:?}",
            eng.family(),
            scored_ids
        );
        eprintln!(
            "DEPTH-DEBUG-TF-ARGMAX model={} depth={depth}: {:?}",
            eng.family(),
            argmax_trace
        );
    }
    eprintln!(
        "DEPTH-TF model={} depth={depth} arm={} frame=doc_native tf_ppl={tf_ppl:.4} \
         tf_worst_nll={worst:.2} scored={scored} prefill_s={prefill_s:.1} prefill_tok_s={:.0} \
         tf_s={tf_s:.1}",
        eng.family(),
        eng.arm_label(),
        prefill_ids.len() as f64 / prefill_s.max(1e-9),
    );

    assert!(
        tf_ppl.is_finite()
            && tf_ppl > 1.0
            && tf_ppl
                < TF_PPL_DOC_NATIVE_SANE_CEILING_50_PASSING_FAMILIES_READ_6_TO_23_WHILE_A_BROKEN_ARM_OR_A_CHAT_FRAME_OOD_MODEL_READS_HUNDREDS_TO_THOUSANDS,
        "{} depth {depth}: doc-native teacher-forced ppl {tf_ppl:.2} outside the sane band; \
         the depth arm is broken, not merely worse",
        eng.family()
    );

    TfRung {
        depth,
        tf_ppl,
        tf_worst_nll: worst,
    }
}

fn run_freerun_rung_official_template(
    eng: &mut dyn DepthCtxEngine,
    tok: &tokenizers::Tokenizer,
    prefix: &[u32],
    suffix: &[u32],
    corpus: &[u32],
    base: usize,
    depth: usize,
    eos: &[u32],
) -> FreerunRung {
    let overhead = prefix.len() + suffix.len();
    assert!(
        depth > overhead + 8,
        "depth {depth} cannot hold template overhead {overhead}"
    );
    let n_doc = depth - overhead;
    assert!(
        base >= n_doc,
        "corpus anchor {base} shorter than the {n_doc} document tokens rung {depth} needs"
    );
    let doc = &corpus[base - n_doc..base];
    eng.reset_ctx(depth)
        .unwrap_or_else(|e| panic!("{} reset at depth {depth}: {e:#}", eng.family()));

    let prefill_ids: Vec<u32> = prefix
        .iter()
        .chain(doc.iter())
        .chain(suffix.iter())
        .copied()
        .collect();
    let (mut row, prefill_s) =
        prefill_chunked_nonfinite_checked(eng, &prefill_ids, depth, "official-template freerun");

    let mut body: Vec<u32> = Vec::new();
    let mut first_eos_pressure_step: Option<usize> = None;
    let t_free = std::time::Instant::now();
    for step in 0..FREERUN_STEPS_64_ENOUGH_TO_EXPOSE_DEGENERATION_AND_SHORT_CYCLES {
        assert_eq!(
            nonfinite_count(&row),
            0,
            "{} depth {depth}: non-finite logits at freerun step {step}",
            eng.family()
        );
        let (cur, eos_was_argmax) = freerun_next_token_with_eos_masked_like_serving_min_tokens_because_e4b_at_8192_legitimately_completes_the_cut_sentence_then_closes_the_turn_at_step_4_so_collapse_must_read_as_degenerate_forced_text_not_as_an_early_eos(&row, eos);
        if eos_was_argmax && first_eos_pressure_step.is_none() {
            first_eos_pressure_step = Some(step);
        }
        body.push(cur);
        row = eng
            .decode_row(cur)
            .unwrap_or_else(|e| panic!("{} freerun decode at depth {depth}: {e:#}", eng.family()));
    }
    let free_s = t_free.elapsed().as_secs_f64();
    assert_eq!(
        body.len(),
        FREERUN_STEPS_64_ENOUGH_TO_EXPOSE_DEGENERATION_AND_SHORT_CYCLES,
        "{} depth {depth}: eos-masked freerun must fill every step",
        eng.family()
    );

    let distinct = {
        let mut v = body.clone();
        v.sort_unstable();
        v.dedup();
        v.len()
    };
    let distinct_ratio = distinct as f64 / body.len().max(1) as f64;
    let run = longest_immediate_run(&body);
    let cycle = tail_cycle(&body);
    let text = tok.decode(&body, false).unwrap_or_else(|e| panic!("decode body: {e}"));
    eprintln!(
        "DEPTH-FREERUN model={} depth={depth} arm={} frame=official_template freerun_len={} \
         eos_pressure_step={first_eos_pressure_step:?} distinct_ratio={distinct_ratio:.3} \
         longest_run={run} tail_cycle={cycle:?} prefill_s={prefill_s:.1} prefill_tok_s={:.0} \
         free_s={free_s:.1}",
        eng.family(),
        eng.arm_label(),
        body.len(),
        prefill_ids.len() as f64 / prefill_s.max(1e-9),
    );
    eprintln!(
        "DEPTH-FREERUN-TEXT model={} depth={depth}: {:?}",
        eng.family(),
        text.chars().take(320).collect::<String>()
    );
    eng.assert_serving_arm_after_freerun(depth);

    assert!(
        distinct_ratio >= FREERUN_MIN_DISTINCT_RATIO_0_20_WIKITEXT_CONTINUATION_IS_LOOPIER_THAN_CHAT_BUT_NOT_DEGENERATE,
        "{} depth {depth}: degenerate freerun, distinct ratio {distinct_ratio:.3}",
        eng.family()
    );
    assert!(
        run <= FREERUN_MAX_IMMEDIATE_RUN_12_WIKITEXT_HEADINGS_REPEAT_EQUALS_SIGNS_LEGITIMATELY,
        "{} depth {depth}: degenerate freerun, {run} identical tokens in a row",
        eng.family()
    );
    assert!(
        cycle.is_none(),
        "{} depth {depth}: degenerate freerun, repeating tail cycle {cycle:?}",
        eng.family()
    );

    FreerunRung {
        depth,
        body_len: body.len(),
        distinct_ratio,
        run,
        cycle,
        first_eos_pressure_step,
    }
}

fn run_sweep(
    eng: &mut dyn DepthCtxEngine,
    dir: &Path,
    depths: &[usize],
    max_pos: usize,
    tf_phase_skip: Option<&'static str>,
) {
    let mut tok = tokenizers::Tokenizer::from_file(dir.join("tokenizer.json"))
        .expect("tokenizer.json ships with the checkpoint");
    tok.with_truncation(None)
        .expect("clear checkpoint-baked truncation: qwen3.6 ships a 4096 cap that silently eats the corpus");
    let (prefix, suffix) = template_spliced_prefix_suffix_ids(dir, &tok);
    let bos_prefix =
        doc_native_tf_prefix_bos_iff_declared_because_chat_rl_models_read_chat_framed_wikitext_at_ppl_in_the_thousands_reproduced_on_vllm(
            dir, &tok,
        );
    let eos = eos_ids_from_generation_config(dir);
    let vocab_probe = tok.get_vocab_size(true);
    let corpus = corpus_ids_head_of_train_file(&tok, vocab_probe.max(1 << 20));
    let max_depth = depths.iter().copied().max().expect("nonempty ladder");
    assert!(
        max_depth + KV_SLOT_SLACK_1024_COVERS_SUFFIX_TF_AND_FREERUN_STEPS < max_pos,
        "ladder max {max_depth} plus slack exceeds max_position_embeddings {max_pos}; a model \
         declaring {max_pos} must serve this depth, so a trip here is a config bug"
    );
    let base = max_depth;
    assert!(
        corpus.len() >= base,
        "corpus tokenizes to {} tokens, the {max_depth} rung needs {base}; point \
         NV_DEPTH_CORPUS at a longer text",
        corpus.len()
    );

    if let Some(reason) = tf_phase_skip {
        eprintln!(
            "DEPTH-TF-SKIP model={} arm={}: {reason}",
            eng.family(),
            eng.arm_label()
        );
    }
    let mut tf_results: Vec<TfRung> = Vec::new();
    let mut freerun_results: Vec<FreerunRung> = Vec::new();
    for &depth in depths {
        if tf_phase_skip.is_none() {
            tf_results.push(run_tf_rung_doc_native(eng, &bos_prefix, &corpus, base, depth));
        }
        freerun_results.push(run_freerun_rung_official_template(
            eng, &tok, &prefix, &suffix, &corpus, base, depth, &eos,
        ));
    }

    if let Some(first) = tf_results.first() {
        for r in &tf_results[1..] {
            assert!(
                r.tf_ppl
                    <= first.tf_ppl
                        * TF_PPL_DEPTH_CLIFF_RATIO_1_5_SAME_TARGETS_WITH_MORE_CONTEXT_MUST_NOT_READ_50_PCT_WORSE,
                "{}: depth-dependent quality cliff: the SAME scored corpus window reads ppl \
                 {:.3} at depth {} but {:.3} at depth {}; deeper context saw strictly more of the \
                 same document, so this is a depth bug, not text variance",
                eng.family(),
                r.tf_ppl,
                r.depth,
                first.tf_ppl,
                first.depth
            );
        }
    }
    let tf_table = if tf_phase_skip.is_some() {
        "SKIPPED_DOCUMENTED".to_string()
    } else {
        tf_results
            .iter()
            .map(|r| format!("depth={} tf_ppl={:.4} worst_nll={:.2}", r.depth, r.tf_ppl, r.tf_worst_nll))
            .collect::<Vec<String>>()
            .join(" | ")
    };
    let freerun_table: Vec<String> = freerun_results
        .iter()
        .map(|r| {
            format!(
                "depth={} freerun={} eos_pressure={:?} ratio={:.3} run={} cycle={:?}",
                r.depth, r.body_len, r.first_eos_pressure_step, r.distinct_ratio, r.run, r.cycle
            )
        })
        .collect();
    eprintln!(
        "DEPTH-CORRECT-SUMMARY model={} arm={} tf_doc_native=[{}] freerun_official_template=[{}] verdict=PASS",
        eng.family(),
        eng.arm_label(),
        tf_table,
        freerun_table.join(" | ")
    );
}

struct QwenGraphedDepthEngine {
    eng: nv_models::graph_engine::GraphedQwen3Moe,
    family: &'static str,
    arm: String,
}

impl DepthCtxEngine for QwenGraphedDepthEngine {
    fn family(&self) -> &'static str {
        self.family
    }
    fn prefill_chunk_len(&self) -> usize {
        PREFILL_CHUNK_512_THE_PROVEN_QWEN_BLOCK
    }
    fn reset_ctx(&mut self, _depth: usize) -> anyhow::Result<()> {
        self.eng.reset()?;
        anyhow::ensure!(self.eng.current_pos() == 0, "reset must rewind the engine");
        Ok(())
    }
    fn prefill_chunk_last_row(&mut self, ids: &[u32]) -> anyhow::Result<Vec<f32>> {
        self.eng.prefill(ids)
    }
    fn decode_row(&mut self, token: u32) -> anyhow::Result<Vec<f32>> {
        self.eng.forward_decode(token)?;
        self.eng.logits_host()
    }
    fn arm_label(&self) -> String {
        format!(
            "{}+decode_kernel={}",
            self.arm,
            common::decode_fix_label_reports_the_nv_q36_graphed_decode_fix_gate_state()
        )
    }
    fn assert_serving_arm_after_freerun(&self, depth: usize) {
        assert!(
            self.eng.capture_active(),
            "the serving basis of this gate is the CAPTURED graphed decode and the engine fell \
             back to uncaptured at depth {depth}; the engine printed its blocker to stderr just \
             above -- diagnose that blocker, do not relabel the run"
        );
    }
}

#[test]
#[ignore = "loads the ~22.6 GB Qwen3.8-27B NVFP4 dense arm GRAPHED; set NV_DEPTH_CORRECT=1 -- \
            per rung at 256/8k/64k/196k: teacher-forced ppl on the SAME corpus window in the \
            document-native frame (BOS iff declared; the chat frame is model-genuine OOD on \
            wikitext, vLLM-reproduced), then a greedy freerun through the official chat \
            template with NaN/degeneracy/cycle checks, EOS masked like serving min_tokens; the gate sets \
            NV_Q36_GRAPHED_DECODE_FIX=1 because that splitk arm is what serving runs; one \
            depth per process via NV_CTX_TOKENS when the #96 residual race bites"]
fn depth_correct_qwen38_27b_cuda_graphed_dense_real_prefill_all_rungs() {
    use nv_models::graph_engine::GraphedQwen3Moe;
    use nv_models::qwen3_5_moe::{Qwen3Moe, Qwen3_5DenseConfig};
    use nv_weights::{QuantizationConfig, WeightLoader};
    if !suite_gate_runs_or_skips_loudly("qwen38-27b") {
        return;
    }
    gate_sets_the_arm_env_because_the_gate_documents_what_serving_should_run(
        "NV_Q36_GRAPHED_DECODE_FIX",
        "routes decode_attention_fp8 to the splitk flash_decode_fused_fp8kv kernel, the arm \
         every published qwen38 depth number runs",
    );
    let _one_gpu_test_at_a_time = ctx_timing_common::serialize_tests_sharing_the_gpu_because_libtest_threads_double_book_the_card_reading_2x_slow_and_a_sibling_engines_teardown_surfaces_a_one_shot_cuda_error_invalid_value();
    let dir = common::qwen38_snapshot_dir_env_override_then_home_hub();
    let device = Device::new_cuda_with_stream(0).expect("cuda");
    let raw_cfg = std::fs::read_to_string(dir.join("config.json")).expect("config.json");
    let cfg = Qwen3_5DenseConfig::from_hf_json_str(&raw_cfg).expect("parse dense config");
    let qcfg = QuantizationConfig::from_hf_json_str(&raw_cfg).expect("parse qcfg");
    let weights = WeightLoader::open_dir(&dir, &device).expect("weights");
    let model = Qwen3Moe::from_loader_dense_quantized(cfg, &weights, &qcfg, &device)
        .expect("build Qwen3.8-27B on the dense arm");
    drop(weights);
    let max_pos = model.config().max_position_embeddings;
    let depths = depth_ladder(&DEPTH_LADDER_262144_MAX);
    let slots = depths.iter().copied().max().unwrap()
        + KV_SLOT_SLACK_1024_COVERS_SUFFIX_TF_AND_FREERUN_STEPS;
    let mut eng = GraphedQwen3Moe::new(model, &device, slots)
        .unwrap_or_else(|e| panic!("GraphedQwen3Moe with {slots} kv slots: {e:#}"));
    eng.set_moe_dispatch(None);
    eng.install_grouped_moe()
        .expect("dense branch of install_grouped_moe arms capture without a dispatch");
    let mut e = QwenGraphedDepthEngine {
        eng,
        family: "qwen38-27b-cuda",
        arm: "graphed_dense".into(),
    };
    run_sweep(&mut e, &dir, &depths, max_pos, None);
}

#[test]
#[ignore = "loads the ~22 GB Qwen3.6-35B NVFP4 GRAPHED grouped-MoE; set NV_DEPTH_CORRECT=1 -- \
            per rung at 256/8k/64k/196k: teacher-forced ppl on the SAME corpus window in the \
            document-native frame (BOS iff declared; the chat frame is model-genuine OOD on \
            wikitext, vLLM-reproduced), then a greedy freerun through the official chat \
            template with NaN/degeneracy/cycle checks, EOS masked like serving min_tokens; the gate sets \
            NV_Q36_GRAPHED_DECODE_FIX=1 because serving runs the splitk arm"]
fn depth_correct_qwen36_35b_cuda_graphed_grouped_moe_real_prefill_all_rungs() {
    use nv_models::graph_engine::GraphedQwen3Moe;
    use nv_models::qwen3_5_moe::{Qwen3Moe, Qwen3MoeConfig};
    use nv_weights::{QuantizationConfig, WeightLoader};
    if !suite_gate_runs_or_skips_loudly("qwen36-35b") {
        return;
    }
    gate_sets_the_arm_env_because_the_gate_documents_what_serving_should_run(
        "NV_Q36_GRAPHED_DECODE_FIX",
        "routes decode_attention_fp8 to the splitk flash_decode_fused_fp8kv kernel; without it \
         the same rung reads two orders of magnitude slower and is not the serving arm",
    );
    let _one_gpu_test_at_a_time = ctx_timing_common::serialize_tests_sharing_the_gpu_because_libtest_threads_double_book_the_card_reading_2x_slow_and_a_sibling_engines_teardown_surfaces_a_one_shot_cuda_error_invalid_value();
    let dir = common::qwen36_nvfp4_snapshot_dir_env_override_then_home_hub();
    let device = Device::new_cuda_with_stream(0).expect("cuda");
    let raw_cfg = std::fs::read_to_string(dir.join("config.json")).expect("config.json");
    let cfg = Qwen3MoeConfig::from_hf_json_str(&raw_cfg).expect("parse config");
    let qcfg = QuantizationConfig::from_hf_json_str(&raw_cfg).expect("parse qcfg");
    let weights = WeightLoader::open_dir(&dir, &device).expect("weights");
    let model = Qwen3Moe::from_loader_quantized(cfg, &weights, &qcfg, &device).expect("model");
    drop(weights);
    let max_pos = model.config().max_position_embeddings;
    let depths = depth_ladder(&DEPTH_LADDER_262144_MAX);
    let slots = depths.iter().copied().max().unwrap()
        + KV_SLOT_SLACK_1024_COVERS_SUFFIX_TF_AND_FREERUN_STEPS;
    let mut eng = GraphedQwen3Moe::new(model, &device, slots)
        .unwrap_or_else(|e| panic!("GraphedQwen3Moe with {slots} kv slots: {e:#}"));
    eng.install_grouped_moe().expect("grouped moe install");
    eng.set_device_routing(true);
    let mut e = QwenGraphedDepthEngine {
        eng,
        family: "qwen36-35b-cuda",
        arm: "graphed_grouped_moe_routed".into(),
    };
    run_sweep(&mut e, &dir, &depths, max_pos, None);
}

struct LagunaEagerDepthEngine {
    model: nv_models::laguna::Laguna,
    cache: Option<nv_models::laguna::LagunaKvCache>,
    pos: usize,
    device: Device,
}

impl LagunaEagerDepthEngine {
    fn forward_ids(&mut self, ids: &[u32]) -> anyhow::Result<Vec<f32>> {
        let s = ids.len();
        let tokens = Tensor::from_vec(ids.to_vec(), (1usize, s), &self.device)?;
        let positions = Tensor::from_vec(
            (self.pos as i32..(self.pos + s) as i32).collect::<Vec<i32>>(),
            s,
            &self.device,
        )?;
        let cache = self.cache.as_mut().expect("reset_ctx before forward");
        let logits = self.model.forward_with_cache(&tokens, &positions, cache)?;
        self.pos += s;
        let row: Vec<f32> = logits
            .narrow(1, s - 1, 1)?
            .to_dtype(DType::F32)?
            .flatten_all()?
            .to_vec1()?;
        Ok(row)
    }
}

impl DepthCtxEngine for LagunaEagerDepthEngine {
    fn family(&self) -> &'static str {
        "laguna-xs21-cuda"
    }
    fn prefill_chunk_len(&self) -> usize {
        PREFILL_CHUNK_256_RING_LEGAL_FOR_SLIDING_CACHES
    }
    fn reset_ctx(&mut self, depth: usize) -> anyhow::Result<()> {
        self.cache = Some(self.model.new_kv_cache(
            depth + KV_SLOT_SLACK_1024_COVERS_SUFFIX_TF_AND_FREERUN_STEPS,
        )?);
        self.pos = 0;
        Ok(())
    }
    fn prefill_chunk_last_row(&mut self, ids: &[u32]) -> anyhow::Result<Vec<f32>> {
        self.forward_ids(ids)
    }
    fn decode_row(&mut self, token: u32) -> anyhow::Result<Vec<f32>> {
        self.forward_ids(&[token])
    }
    fn arm_label(&self) -> String {
        "default_eager_cuda_forward_with_cache".into()
    }
}

#[test]
#[ignore = "loads the Laguna-XS-2.1 NVFP4; set NV_DEPTH_CORRECT=1 -- per rung at \
            256/8k/64k/196k on the default eager cuda arm: teacher-forced ppl on the SAME \
            corpus window in the document-native frame, then a greedy freerun through the \
            official chat template with NaN/degeneracy/cycle checks, EOS masked like serving min_tokens"]
fn depth_correct_laguna_xs21_cuda_eager_real_prefill_all_rungs() {
    use nv_models::laguna::{Laguna, LagunaConfig};
    use nv_weights::{QuantizationConfig, WeightLoader};
    if !suite_gate_runs_or_skips_loudly("laguna-xs21") {
        return;
    }
    let _one_gpu_test_at_a_time = ctx_timing_common::serialize_tests_sharing_the_gpu_because_libtest_threads_double_book_the_card_reading_2x_slow_and_a_sibling_engines_teardown_surfaces_a_one_shot_cuda_error_invalid_value();
    let dir = common::laguna_xs21_nvfp4_snapshot_dir_env_override_then_home_hub();
    let device = Device::new_cuda(0).expect("cuda");
    let raw_cfg = std::fs::read_to_string(dir.join("config.json")).expect("config.json");
    let cfg = LagunaConfig::from_hf_json_str(&raw_cfg).expect("parse config");
    let qcfg = QuantizationConfig::from_hf_json_str(&raw_cfg).expect("parse qcfg");
    let weights = WeightLoader::open_dir(&dir, &device).expect("weights");
    let model = Laguna::from_loader_quantized(cfg, &weights, &qcfg, &device).expect("model");
    drop(weights);
    let max_pos = model.config().max_position_embeddings;
    let depths = depth_ladder(&DEPTH_LADDER_262144_MAX);
    let mut e = LagunaEagerDepthEngine {
        model,
        cache: None,
        pos: 0,
        device,
    };
    run_sweep(&mut e, &dir, &depths, max_pos, None);
}

struct Gemma4PagedDepthEngine {
    model: std::sync::Arc<nv_models::gemma4::Gemma4>,
    pool: Option<std::sync::Arc<std::sync::Mutex<nv_models::paged_fp8::PagedKvFp8Pool>>>,
    cache: Option<nv_models::paged_fp8::PagedGemma4Cache>,
    pos: usize,
    device: Device,
}

impl Gemma4PagedDepthEngine {
    fn forward_ids(&mut self, ids: &[u32]) -> anyhow::Result<Vec<f32>> {
        let s = ids.len();
        let tokens = Tensor::from_vec(ids.to_vec(), (1usize, s), &self.device)?;
        let positions = Tensor::from_vec(
            (self.pos as i32..(self.pos + s) as i32).collect::<Vec<i32>>(),
            s,
            &self.device,
        )?;
        let cache = self.cache.as_mut().expect("reset_ctx before forward");
        let logits = self
            .model
            .forward_with_cache_last(&tokens, &positions, cache)?;
        self.pos += s;
        let vocab = self.model.config().vocab_size;
        let v: Vec<f32> = logits
            .to_dtype(DType::F32)?
            .flatten_all()?
            .to_vec1()?;
        anyhow::ensure!(
            v.len() == vocab,
            "forward_with_cache_last returned {} logits, expected one row of {vocab}",
            v.len()
        );
        Ok(v)
    }
}

impl DepthCtxEngine for Gemma4PagedDepthEngine {
    fn family(&self) -> &'static str {
        "gemma4-31b-cuda"
    }
    fn prefill_chunk_len(&self) -> usize {
        PREFILL_CHUNK_256_RING_LEGAL_FOR_SLIDING_CACHES
    }
    fn reset_ctx(&mut self, depth: usize) -> anyhow::Result<()> {
        use nv_models::paged_fp8::{PagedGemma4Cache, PagedKvFp8Pool, PagedPoolConfig};
        use std::sync::{Arc, Mutex};
        let bs = 16usize;
        let slots = depth + KV_SLOT_SLACK_1024_COVERS_SUFFIX_TF_AND_FREERUN_STEPS;
        let seq_blocks = slots.div_ceil(bs);
        self.cache = None;
        self.pool = None;
        let hybrid_cfg =
            PagedPoolConfig::from_gemma4_hybrid(self.model.config(), seq_blocks, bs, 1);
        let pool = Arc::new(Mutex::new(PagedKvFp8Pool::new(hybrid_cfg, &self.device)?));
        let mut cache = PagedGemma4Cache::new(pool.clone(), &self.device)?;
        let table: Vec<u32> = (0..seq_blocks as u32).collect();
        cache.set_block_table(&table)?;
        self.pool = Some(pool);
        self.cache = Some(cache);
        self.pos = 0;
        Ok(())
    }
    fn prefill_chunk_last_row(&mut self, ids: &[u32]) -> anyhow::Result<Vec<f32>> {
        self.forward_ids(ids)
    }
    fn decode_row(&mut self, token: u32) -> anyhow::Result<Vec<f32>> {
        self.forward_ids(&[token])
    }
    fn arm_label(&self) -> String {
        "eager_paged_fp8_hybrid_ring_forward_with_cache".into()
    }
}

#[test]
#[ignore = "loads the 31B NVFP4; set NV_DEPTH_CORRECT=1 -- per rung at 256/8k/64k/196k on the \
            eager paged fp8 hybrid-ring arm (the arm that reaches 196k; the graphed pool \
            geometry cannot hold it): a greedy freerun through the official chat template with \
            NaN/degeneracy/cycle checks, EOS masked like serving min_tokens; the TF phase is skipped with the #94 rationale \
            printed"]
fn depth_correct_gemma4_31b_cuda_eager_paged_fp8_hybrid_ring_real_prefill_all_rungs() {
    use nv_models::gemma4::{Gemma4, Gemma4Config};
    use nv_weights::{QuantizationConfig, WeightLoader};
    use std::sync::Arc;
    if !suite_gate_runs_or_skips_loudly("gemma4-31b") {
        return;
    }
    let _one_gpu_test_at_a_time = ctx_timing_common::serialize_tests_sharing_the_gpu_because_libtest_threads_double_book_the_card_reading_2x_slow_and_a_sibling_engines_teardown_surfaces_a_one_shot_cuda_error_invalid_value();
    let dir = common::gemma4_snapshot_dir_env_override_then_home_hub();
    let device = Device::new_cuda_with_stream(0).expect("cuda");
    let raw_cfg = std::fs::read_to_string(dir.join("config.json")).expect("config.json");
    let cfg = Gemma4Config::from_hf_json_str(&raw_cfg).expect("parse config");
    let qcfg = QuantizationConfig::from_hf_json_str(&raw_cfg).expect("parse qcfg");
    let weights = WeightLoader::open_dir(&dir, &device).expect("weights");
    let model = Arc::new(Gemma4::from_loader_quantized(cfg, &weights, &qcfg, &device).expect("model"));
    drop(weights);
    let max_pos = model.config().max_position_embeddings;
    let depths = depth_ladder(&DEPTH_LADDER_262144_MAX);
    let mut e = Gemma4PagedDepthEngine {
        model,
        pool: None,
        cache: None,
        pos: 0,
        device,
    };
    run_sweep(
        &mut e,
        &dir,
        &depths,
        max_pos,
        Some(TF_PHASE_SKIP_GEMMA4_TASK_94_RULES_RAW_CORPUS_GEMMA4_TF_PPL_AN_INVALID_EVAL_IN_ANY_FRAME),
    );
}

struct E4bFp8DepthEngine {
    model: nv_models::gemma4_e4b::Gemma4E4b,
    cache: Option<nv_models::gemma4_e4b::E4bFp8Cache>,
    cfg: nv_models::gemma4::Gemma4Config,
    pos: usize,
    device: Device,
}

impl E4bFp8DepthEngine {
    fn forward_ids(&mut self, ids: &[u32]) -> anyhow::Result<Vec<f32>> {
        let cache = self.cache.as_mut().expect("reset_ctx before forward");
        let logits = self.model.forward_step_fp8(ids, self.pos, cache)?;
        self.pos += ids.len();
        Ok(logits.to_dtype(DType::F32)?.flatten_all()?.to_vec1()?)
    }
}

impl DepthCtxEngine for E4bFp8DepthEngine {
    fn family(&self) -> &'static str {
        "gemma4-e4b-cuda"
    }
    fn prefill_chunk_len(&self) -> usize {
        PREFILL_CHUNK_256_RING_LEGAL_FOR_SLIDING_CACHES
    }
    fn reset_ctx(&mut self, depth: usize) -> anyhow::Result<()> {
        self.cache = Some(nv_models::gemma4_e4b::E4bFp8Cache::new(
            &self.cfg,
            depth + KV_SLOT_SLACK_1024_COVERS_SUFFIX_TF_AND_FREERUN_STEPS,
            &self.device,
        )?);
        self.pos = 0;
        Ok(())
    }
    fn prefill_chunk_last_row(&mut self, ids: &[u32]) -> anyhow::Result<Vec<f32>> {
        self.forward_ids(ids)
    }
    fn decode_row(&mut self, token: u32) -> anyhow::Result<Vec<f32>> {
        self.forward_ids(&[token])
    }
    fn arm_label(&self) -> String {
        "eager_forward_step_fp8_paged_cache".into()
    }
}

#[test]
#[ignore = "loads gemma-4-E4B-it on cuda; set NV_DEPTH_CORRECT=1 -- per rung at \
            256/8k/64k/122880 (131072 max): a greedy freerun through the official chat \
            template with NaN/degeneracy/cycle checks, EOS masked like serving min_tokens; the TF phase is skipped with the #94 \
            rationale printed; fp8 paged cache arm because the cat cache re-concatenates the \
            whole depth every step"]
fn depth_correct_e4b_cuda_eager_fp8_cache_real_prefill_all_rungs() {
    use nv_models::gemma4::Gemma4Config;
    use nv_models::gemma4_e4b::Gemma4E4b;
    use nv_weights::WeightLoader;
    if !suite_gate_runs_or_skips_loudly("gemma4-e4b") {
        return;
    }
    let _one_gpu_test_at_a_time = ctx_timing_common::serialize_tests_sharing_the_gpu_because_libtest_threads_double_book_the_card_reading_2x_slow_and_a_sibling_engines_teardown_surfaces_a_one_shot_cuda_error_invalid_value();
    let dir = common::e4b_snapshot_dir();
    let device = Device::new_cuda_with_stream(0).expect("cuda");
    let cfg = Gemma4Config::from_hf_json_file(&dir.join("config.json")).expect("config");
    assert!(
        cfg.has_per_layer_embeddings(),
        "not an E4B checkpoint: {dir:?}"
    );
    let weights = WeightLoader::open_dir(&dir, &device).expect("weights");
    let model = Gemma4E4b::from_loader(cfg.clone(), &weights, &device).expect("load Gemma4E4b");
    drop(weights);
    let max_pos = cfg.max_position_embeddings;
    let depths = depth_ladder(&DEPTH_LADDER_131072_MAX);
    let mut e = E4bFp8DepthEngine {
        model,
        cache: None,
        cfg,
        pos: 0,
        device,
    };
    run_sweep(
        &mut e,
        &dir,
        &depths,
        max_pos,
        Some(TF_PHASE_SKIP_GEMMA4_TASK_94_RULES_RAW_CORPUS_GEMMA4_TF_PPL_AN_INVALID_EVAL_IN_ANY_FRAME),
    );
}

struct GptOssDepthEngine {
    model: nv_models::gpt_oss_cuda::GptOssCuda,
    cache: Option<nv_models::gpt_oss_cuda::GptOssKvCache>,
    pos: usize,
}

impl GptOssDepthEngine {
    fn forward_ids(&mut self, ids: &[u32]) -> anyhow::Result<Vec<f32>> {
        let positions: Vec<u32> = (self.pos as u32..(self.pos + ids.len()) as u32).collect();
        let cache = self.cache.as_mut().expect("reset_ctx before forward");
        let row = self.model.forward_last_logits(ids, &positions, cache)?;
        self.pos += ids.len();
        Ok(row)
    }
}

impl DepthCtxEngine for GptOssDepthEngine {
    fn family(&self) -> &'static str {
        "gpt-oss-20b-cuda"
    }
    fn prefill_chunk_len(&self) -> usize {
        PREFILL_CHUNK_256_RING_LEGAL_FOR_SLIDING_CACHES
    }
    fn reset_ctx(&mut self, depth: usize) -> anyhow::Result<()> {
        self.cache = Some(self.model.new_kv_cache(
            depth + KV_SLOT_SLACK_1024_COVERS_SUFFIX_TF_AND_FREERUN_STEPS,
        )?);
        self.pos = 0;
        Ok(())
    }
    fn prefill_chunk_last_row(&mut self, ids: &[u32]) -> anyhow::Result<Vec<f32>> {
        self.forward_ids(ids)
    }
    fn decode_row(&mut self, token: u32) -> anyhow::Result<Vec<f32>> {
        self.forward_ids(&[token])
    }
    fn arm_label(&self) -> String {
        "eager_bf16_dequant_sdpa_with_sinks".into()
    }
}

#[test]
#[ignore = "loads openai/gpt-oss-20b dequantized to bf16 (~42 GB resident); set \
            NV_DEPTH_CORRECT=1 -- per rung at 256/8k/64k/122880 (131072 max): a greedy freerun \
            through the official chat template with NaN/degeneracy/cycle checks, EOS masked like serving min_tokens; the TF \
            phase is skipped with the vLLM-cross-checked model-genuine rationale printed; \
            eager sdpa_with_sinks makes the deep prefill slow, run it last"]
fn depth_correct_gpt_oss_20b_cuda_eager_real_prefill_all_rungs() {
    use nv_models::gpt_oss::GptOssConfig;
    use nv_models::gpt_oss_cuda::GptOssCuda;
    use nv_weights::WeightLoader;
    if !suite_gate_runs_or_skips_loudly("gpt-oss-20b") {
        return;
    }
    let _one_gpu_test_at_a_time = ctx_timing_common::serialize_tests_sharing_the_gpu_because_libtest_threads_double_book_the_card_reading_2x_slow_and_a_sibling_engines_teardown_surfaces_a_one_shot_cuda_error_invalid_value();
    let dir = common::snapshot_dir_env_override_then_home_hub_first_repo_with_config_and_weights(
        "NV_GPTOSS_DIR",
        &["models--openai--gpt-oss-20b"],
    );
    let device = Device::new_cuda_with_stream(0).expect("cuda");
    let cfg = GptOssConfig::from_hf_json_file(&dir.join("config.json")).expect("config");
    let max_pos = cfg.max_position_embeddings;
    let weights = WeightLoader::open_dir(&dir, &device).expect("weights");
    let model = GptOssCuda::from_loader(cfg, &weights, &device).expect("load GptOssCuda");
    drop(weights);
    let depths = depth_ladder(&DEPTH_LADDER_131072_MAX);
    let mut e = GptOssDepthEngine {
        model,
        cache: None,
        pos: 0,
    };
    run_sweep(
        &mut e,
        &dir,
        &depths,
        max_pos,
        Some(TF_PHASE_SKIP_GPT_OSS_RAW_TEXT_DEPTH_DEGRADATION_IS_MODEL_GENUINE_VLLM_CROSS_CHECKED),
    );
}

const DEPTH_LADDER_262144_MAX_WITH_16384_BECAUSE_BOTH_STANDING_TARGETS_16K_AND_196K_MUST_BE_RUNGS:
    [usize; 5] = [256, 8192, 16384, 65536, 196608];

const TF_PHASE_SKIP_G4MOE_TASK_94_FAMILY_RULING_THE_CHAT_FRAME_PAIRED_SUITE_IS_THE_QUALITY_AXIS:
    &str =
    "task #94 (ppl_gemma4.rs) rules raw-corpus gemma4 teacher forcing an invalid eval in any \
     frame and the 26B-A4B is the same family; this rung's depth evidence is the \
     official-template freerun below, and the family's quality axis at depth is the chat-frame \
     paired-window suite in this file plus the chat-wrapped ppl_gemma4_moe instrument";

const G4QUAL_SCORED_512_MATCHES_THE_FAMILYS_CANONICAL_CHAT_PPL_CELL_WIDTH: usize = 512;

const G4QUAL_WINDOWS_DEFAULT_3_EACH_DEEP_WINDOW_COSTS_A_FULL_REAL_PREFILL_OVERRIDE_NV_G4QUAL_WINDOWS:
    usize = 3;

const G4QUAL_NULL_FULL_SPLITS_64_MATCHES_THE_FOLD_ARMS_OWN_DEEP_SPLIT_COUNT_WITHOUT_THE_FOLD: &str =
    "64";

const G4QUAL_NULL_SLIDING_SPLITS_PINNED_32_BECAUSE_THE_FOLDGATE_NEVER_TOUCHES_THE_SLIDING_KIND:
    &str = "32";

const G4QUAL_A_REASSOCIATION_NULL_MAY_NOT_MOVE_TOP1_ACCURACY_BY_MORE_THAN_THIS_OR_THE_SPLIT_COUNT_IS_NOT_THE_SAME_ARITHMETIC:
    f64 = 0.05;

const G4MOE_HOME_HUB_REPO_MATCHING_THE_SWEEP_SPEED_ROW: &str =
    "models--google--gemma-4-26B-A4B-it";

#[cfg(feature = "wgpu")]
fn g4moe_snapshot_dir_env_override_then_home_hub() -> std::path::PathBuf {
    if let Ok(d) = std::env::var("NV_G4MOE_SNAPSHOT") {
        return std::path::PathBuf::from(d);
    }
    let home = std::env::var("HOME").expect("HOME");
    let base = std::path::PathBuf::from(&home)
        .join(".cache/huggingface/hub")
        .join(G4MOE_HOME_HUB_REPO_MATCHING_THE_SWEEP_SPEED_ROW)
        .join("snapshots");
    let mut candidates: Vec<std::path::PathBuf> = std::fs::read_dir(&base)
        .unwrap_or_else(|e| panic!("g4moe snapshots dir {base:?} missing: {e}"))
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.join("config.json").is_file() && p.join("tokenizer.json").is_file())
        .collect();
    candidates.sort();
    candidates
        .into_iter()
        .next()
        .expect("no complete gemma-4-26B-A4B-it snapshot under HOME hub; set NV_G4MOE_SNAPSHOT")
}

#[cfg(feature = "wgpu")]
fn g4moe_arm_label_from_resolved_gate_values_because_env_presence_lies_after_a_default_flip(
) -> String {
    use nv_models::gemma4_moe_wgpu as g4m;
    let full_splits = g4m::flash_splits_grid_y_scratch_rows_and_stage1_stride_read_once_at_build();
    format!(
        "flash_decode={} flash_sliding={} kv_ring={} kv_fp8={} grouped_gemv={} rpnorm={} \
         rpfuse={} topk_nx2={} splits_full={} splits_sliding={} depthgate={} foldgate={} \
         prefill_m={}",
        g4m::flash_decode_enabled(),
        g4m::flash_sliding_enabled(),
        g4m::sliding_kv_ring_enabled(),
        g4m::kv_fp8_enabled(),
        g4m::grouped_gemv_enabled(),
        g4m::rpnorm_enabled(),
        g4m::rpfuse_enabled(),
        g4m::topk_nx2_enabled(),
        full_splits,
        g4m::flash_sliding_splits_the_ring_capped_scan_gains_no_stage1_rows_from_extra_splits_and_pays_every_one_in_stage2(full_splits),
        g4m::flash_depthgate_enabled_a_shallow_tuned_split_pin_starves_the_grid_at_depth(),
        g4m::flash_foldgate_enabled_full_attention_swaps_to_the_fold2_arm_at_depth_while_sliding_keeps_its_own_band(),
        g4m::prefill_m(),
    )
}

#[cfg(feature = "wgpu")]
fn g4moe_arm_env_pins_zero_env_default_except_the_three_arm_slots(
    foldgate: Option<&str>,
    full_splits: Option<&str>,
    sliding_splits: Option<&str>,
) -> common::EnvPins {
    use nv_models::gemma4_moe_wgpu as g4m;
    common::EnvPins::pin(&[
        ("NV_G4MOE_FLASH_DECODE", None),
        ("NV_G4MOE_FLASH_SLIDING", None),
        ("NV_G4MOE_FLASH_SG", None),
        (g4m::G4MOE_FLASH_DEPTHGATE_ENV, None),
        (g4m::G4MOE_FLASH_DEPTHGATE_SPLITS_ENV, None),
        (g4m::G4MOE_FLASH_FOLDGATE_ENV, foldgate),
        ("NV_G4MOE_FLASH_SPLITS", full_splits),
        (g4m::G4MOE_FLASH_SLIDING_SPLITS_ENV, sliding_splits),
        (g4m::FLASH_GQA_FOLD_ENV, None),
        (g4m::G4MOE_FLASH_SLIDING_FOLD_ENV, None),
        (g4m::FLASH_SACC_CHUNK_ENV, None),
        ("NV_G4MOE_PREENC", None),
        ("NV_G4MOE_QKVFOLD", None),
        ("NV_G4MOE_KV_FP8", None),
        ("NV_G4MOE_KV_RING", None),
        ("NV_G4MOE_GROUPED_GEMV", None),
        ("NV_G4MOE_RPNORM", None),
        ("NV_G4MOE_RPFUSE", None),
        ("NV_G4MOE_TOPK_NX2", None),
        ("NV_G4MOE_GEMV_V2", None),
        ("NV_G4MOE_GEMV_SGLAT", None),
        ("NV_G4MOE_GEMV_WIDE", None),
        ("NV_G4MOE_W4_GEMV", None),
        ("NV_G4MOE_ROUTER_TOPK", None),
        ("NV_G4MOE_LMHEAD_I8_QUALITY_UNGATED", None),
        ("NV_G4MOE_KIND_LABELS", None),
    ])
}

#[cfg(feature = "wgpu")]
fn g4moe_build_real_model_under_current_env(
    dir: &Path,
    max_seq: usize,
) -> nv_models::gemma4_moe_wgpu::Gemma4MoeWgpu {
    use nv_models::gemma4_moe::Gemma4MoeConfig;
    use nv_models::gemma4_moe_wgpu::Gemma4MoeWgpu;
    let config = Gemma4MoeConfig::from_hf_json_file(&dir.join("config.json")).expect("config.json");
    let loader =
        nv_weights::WeightLoader::open_dir(dir, &candle_core::Device::Cpu).expect("weights");
    let m = Gemma4MoeWgpu::from_loader(config, &loader, max_seq).expect("wgpu 26B-A4B model");
    drop(loader);
    m
}

#[cfg(feature = "wgpu")]
fn g4qual_bulk_chunk_prefill_is_arm_independent_because_the_pf_passes_never_read_the_split_or_fold_envs(
    m: &mut nv_models::gemma4_moe_wgpu::Gemma4MoeWgpu,
    prefill_ids: &[u32],
) -> (Vec<Vec<u32>>, usize) {
    let inner = m.prefill_chunk_len();
    assert!(
        inner > 0,
        "chunked prefill must be live: a serial 196k prefill is hours, not minutes"
    );
    m.reset().expect("reset before the shared window prefill");
    let bulk = (prefill_ids.len() - 1) / inner * inner;
    let t0 = std::time::Instant::now();
    let done = m.prefill_tokens(&prefill_ids[..bulk]).expect("bulk chunked prefill");
    assert_eq!(done, bulk, "prefill_tokens must consume every whole chunk it was given");
    assert_eq!(m.current_pos(), bulk, "bulk prefill must land exactly on the chunk boundary");
    let prefill_s = t0.elapsed().as_secs_f64();
    eprintln!(
        "G4QUAL-TIMING depth={} bulk={bulk} prefill_s={prefill_s:.1} prefill_tok_s={:.0}",
        prefill_ids.len(),
        bulk as f64 / prefill_s.max(1e-9)
    );
    let blobs: Vec<Vec<u32>> = (0..m.state_blob_count())
        .map(|i| {
            m.state_blob_download(i)
                .expect("state blob download")
                .expect("state blob index in range")
        })
        .collect();
    assert!(!blobs.is_empty(), "a model with no KV state blobs cannot share a prefill");
    (blobs, bulk)
}

#[cfg(feature = "wgpu")]
fn g4qual_restore_shared_prefill_state(
    m: &mut nv_models::gemma4_moe_wgpu::Gemma4MoeWgpu,
    blobs: &[Vec<u32>],
    bulk: usize,
) {
    assert_eq!(
        m.state_blob_count(),
        blobs.len(),
        "arms disagree on state blob layout; the snapshot cannot cross this build"
    );
    for (i, b) in blobs.iter().enumerate() {
        assert!(
            m.state_blob_restore(i, b).expect("state blob restore"),
            "state blob {i} out of range on restore"
        );
    }
    m.restore_pos(bulk).expect("restore_pos to the snapshot boundary");
}

#[cfg(feature = "wgpu")]
fn g4qual_feed_tail_then_score_targets_on_this_arms_own_decode_path(
    m: &mut nv_models::gemma4_moe_wgpu::Gemma4MoeWgpu,
    tail_ids: &[u32],
    targets: &[u32],
) -> G4qualWindowCell {
    assert!(!tail_ids.is_empty(), "the tail must produce the row for the first scored target");
    let mut row: Option<Vec<f32>> = None;
    for &t in tail_ids {
        row = Some(m.decode_step_logits(t).expect("tail decode step").1);
    }
    let mut row = row.expect("nonempty tail");
    let vocab = m.config().base.vocab_size;
    let mut acc = ppl_common::TeacherForcedNllFp32SoftmaxF64Sum::new();
    for (i, &t) in targets.iter().enumerate() {
        assert_eq!(
            row.len(),
            vocab,
            "decode_step_logits must return the full vocab row, not a top-k slice"
        );
        acc.add_position_full_vocab_row(&row, t);
        if i + 1 < targets.len() {
            row = m.decode_step_logits(t).expect("teacher-forced decode step").1;
        }
    }
    G4qualWindowCell {
        sum_nll_f64: acc.sum_neg_ln_p_f64,
        hits: acc.argmax_hits,
        scored: acc.scored_positions,
    }
}

#[cfg(feature = "wgpu")]
struct G4moeWgpuDepthEngine {
    m: nv_models::gemma4_moe_wgpu::Gemma4MoeWgpu,
}

#[cfg(feature = "wgpu")]
impl DepthCtxEngine for G4moeWgpuDepthEngine {
    fn family(&self) -> &'static str {
        "gemma4-26b-a4b-wgpu"
    }
    fn prefill_chunk_len(&self) -> usize {
        PREFILL_CHUNK_512_THE_PROVEN_QWEN_BLOCK
    }
    fn reset_ctx(&mut self, depth: usize) -> anyhow::Result<()> {
        anyhow::ensure!(
            depth + KV_SLOT_SLACK_1024_COVERS_SUFFIX_TF_AND_FREERUN_STEPS <= self.m.max_seq(),
            "rung depth {depth} plus slack exceeds the built max_seq {}",
            self.m.max_seq()
        );
        self.m.reset()?;
        anyhow::ensure!(self.m.current_pos() == 0, "reset must rewind the engine");
        Ok(())
    }
    fn prefill_chunk_last_row(&mut self, ids: &[u32]) -> anyhow::Result<Vec<f32>> {
        let inner = self.m.prefill_chunk_len();
        anyhow::ensure!(inner > 0, "chunked prefill must be live");
        let bulk = (ids.len() - 1) / inner * inner;
        let done = self.m.prefill_tokens(&ids[..bulk])?;
        anyhow::ensure!(done == bulk, "prefill_tokens consumed {done} of {bulk}");
        let mut row: Option<Vec<f32>> = None;
        for &t in &ids[bulk..] {
            row = Some(self.m.decode_step_logits(t)?.1);
        }
        Ok(row.expect("nonempty chunk"))
    }
    fn decode_row(&mut self, token: u32) -> anyhow::Result<Vec<f32>> {
        Ok(self.m.decode_step_logits(token)?.1)
    }
    fn arm_label(&self) -> String {
        g4moe_arm_label_from_resolved_gate_values_because_env_presence_lies_after_a_default_flip()
    }
    fn assert_serving_arm_after_freerun(&self, depth: usize) {
        assert!(
            !self.m.full_attention_flash1_pass_labels().is_empty(),
            "the serving basis of this rung is the flash decode arm and no full-attention flash1 \
             pass was recorded at depth {depth}; the split flash decode never engaged"
        );
        assert!(
            !self.m.foldgate_deep_arm_swapped_in_as_of_last_step(),
            "the zero-env rung must serve the control arm; a swapped-in fold arm at depth \
             {depth} means the foldgate env leaked into this build"
        );
    }
}

#[test]
#[ignore = "loads the real ~26B-A4B MoE onto the wgpu backend; set NV_DEPTH_CORRECT=1 -- per \
            rung at 256/8k/16k/64k/196k (16384 is a standing target depth, so this rung's \
            ladder includes it) on the zero-env serving default: a greedy freerun through the \
            official chat template with NaN/degeneracy/cycle checks, EOS masked like serving \
            min_tokens; the TF phase is skipped with the #94 family rationale printed; one \
            depth per process via NV_CTX_TOKENS when a deep prefill must run alone"]
#[cfg(feature = "wgpu")]
fn depth_correct_g4moe_26b_a4b_wgpu_serving_default_real_prefill_all_rungs() {
    if !suite_gate_runs_or_skips_loudly("gemma4-26b-a4b-wgpu") {
        return;
    }
    let _one_gpu_test_at_a_time = ctx_timing_common::serialize_tests_sharing_the_gpu_because_libtest_threads_double_book_the_card_reading_2x_slow_and_a_sibling_engines_teardown_surfaces_a_one_shot_cuda_error_invalid_value();
    let ctx = common::ctx();
    eprintln!("adapter: {}", ctx.summary());
    let dir = g4moe_snapshot_dir_env_override_then_home_hub();
    let depths = depth_ladder(
        &DEPTH_LADDER_262144_MAX_WITH_16384_BECAUSE_BOTH_STANDING_TARGETS_16K_AND_196K_MUST_BE_RUNGS,
    );
    let slots = depths.iter().copied().max().expect("nonempty ladder")
        + KV_SLOT_SLACK_1024_COVERS_SUFFIX_TF_AND_FREERUN_STEPS;
    let pins = g4moe_arm_env_pins_zero_env_default_except_the_three_arm_slots(None, None, None);
    let m = g4moe_build_real_model_under_current_env(&dir, slots);
    let max_pos = m.config().base.max_position_embeddings;
    let mut e = G4moeWgpuDepthEngine { m };
    run_sweep(
        &mut e,
        &dir,
        &depths,
        max_pos,
        Some(TF_PHASE_SKIP_G4MOE_TASK_94_FAMILY_RULING_THE_CHAT_FRAME_PAIRED_SUITE_IS_THE_QUALITY_AXIS),
    );
    drop(pins);
}

#[cfg(feature = "wgpu")]
struct G4qualWindowCell {
    sum_nll_f64: f64,
    hits: usize,
    scored: usize,
}

#[cfg(feature = "wgpu")]
fn g4qual_window_prefill_and_targets_disjoint_stride_depth_plus_scored(
    corpus: &[u32],
    prefix: &[u32],
    suffix: &[u32],
    depth: usize,
    window: usize,
    scored: usize,
) -> (Vec<u32>, Vec<u32>) {
    let overhead = prefix.len() + suffix.len();
    assert!(
        depth > overhead + 8,
        "depth {depth} cannot hold template overhead {overhead}"
    );
    let n_doc = depth - overhead;
    let end = (window + 1) * depth + window * scored;
    assert!(
        end + scored <= corpus.len(),
        "window {window} at depth {depth} needs corpus tokens up to {} but only {} exist",
        end + scored,
        corpus.len()
    );
    let doc = &corpus[end - n_doc..end];
    let prefill: Vec<u32> = prefix.iter().chain(doc.iter()).chain(suffix.iter()).copied().collect();
    assert_eq!(
        prefill.len(),
        depth,
        "the prefill must land the KV depth exactly on the rung"
    );
    (prefill, corpus[end..end + scored].to_vec())
}

#[cfg(feature = "wgpu")]
fn g4qual_independent_windows_at_depth_capped_by_env(corpus_len: usize, depth: usize) -> usize {
    let scored = G4QUAL_SCORED_512_MATCHES_THE_FAMILYS_CANONICAL_CHAT_PPL_CELL_WIDTH;
    let available = corpus_len.saturating_sub(scored) / (depth + scored);
    let cap = common::envn(
        "NV_G4QUAL_WINDOWS",
        G4QUAL_WINDOWS_DEFAULT_3_EACH_DEEP_WINDOW_COSTS_A_FULL_REAL_PREFILL_OVERRIDE_NV_G4QUAL_WINDOWS,
    );
    let n = available.min(cap);
    assert!(
        n >= 1,
        "corpus of {corpus_len} tokens yields zero independent windows at depth {depth}"
    );
    n
}

#[cfg(feature = "wgpu")]
fn g4qual_assert_arm_engaged_by_grid_not_env_presence(
    m: &nv_models::gemma4_moe_wgpu::Gemma4MoeWgpu,
    arm_name: &str,
    depth: usize,
    foldgate_armed: bool,
    built_full_splits: u32,
) {
    use nv_models::gemma4_moe_wgpu as g4m;
    let scored = G4QUAL_SCORED_512_MATCHES_THE_FAMILYS_CANONICAL_CHAT_PPL_CELL_WIDTH;
    assert_eq!(
        g4m::foldgate_deep_at_these_whole_context_tokens(depth as u32),
        g4m::foldgate_deep_at_these_whole_context_tokens((depth + scored) as u32),
        "a scored window straddling the foldgate knee mixes two arms in one cell"
    );
    let deep = foldgate_armed && g4m::foldgate_deep_at_these_whole_context_tokens(depth as u32);
    assert_eq!(
        m.foldgate_deep_arm_swapped_in_as_of_last_step(),
        deep,
        "{arm_name} at depth {depth}: the foldgate swap state is not what this row claims to \
         measure; the row is void without its engagement proof"
    );
    let n_q = m.config().base.num_attention_heads as u32;
    let want = if deep {
        (
            n_q / g4m::FOLDGATE_DEEP_FOLD_2_QUERY_HEADS_PER_WORKGROUP_HALVES_FULL_ATTENTION_KV_TRAFFIC,
            g4m::FOLDGATE_DEEP_SPLITS_64_RESTORES_512_WORKGROUPS_UNDER_FOLD_2_AND_IS_A_MEASURED_RUNG_THE_LADDER_HAS_A_REPRODUCED_POTHOLE_AT_40_47_48_SO_NEVER_INTERPOLATE,
        )
    } else {
        (n_q, built_full_splits)
    };
    assert_eq!(
        m.foldgate_full_attention_flash1_grid_xy_as_last_encoded(),
        Some(want),
        "{arm_name} at depth {depth}: the full-attention flash1 grid is not the one this arm \
         claims to dispatch"
    );
    assert_eq!(
        m.depthgate_sliding_flash1_grid_y_as_last_encoded(),
        Some(g4m::G4MOE_FLASH_SPLITS_32_IS_THE_MEASURED_KNEE_AT_DEPTH),
        "{arm_name} at depth {depth}: the sliding kind must stay on the default 32 splits in \
         every arm of this contrast, because the foldgate never touches the sliding kind"
    );
}

#[cfg(feature = "wgpu")]
fn g4qual_corpus_ids_whole_train_file(tok: &tokenizers::Tokenizer, vocab: usize) -> Vec<u32> {
    let path = std::env::var("NV_DEPTH_CORPUS").unwrap_or_else(|_| DEFAULT_CORPUS.to_string());
    let text = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("read corpus {path}: {e}; set NV_DEPTH_CORPUS"));
    let ids: Vec<u32> = tok
        .encode(text.as_str(), false)
        .expect("encode whole corpus")
        .get_ids()
        .to_vec();
    let max_id = *ids.iter().max().expect("nonempty corpus") as usize;
    assert!(
        max_id < vocab,
        "corpus token id {max_id} out of model vocab {vocab}; wrong tokenizer for this checkpoint"
    );
    ids
}

#[cfg(feature = "wgpu")]
fn g4qual_paired_mean_sd_t_over_windows(deltas: &[f64]) -> (f64, f64, f64) {
    let n = deltas.len() as f64;
    let mean = deltas.iter().sum::<f64>() / n;
    if deltas.len() < 2 {
        return (mean, f64::NAN, f64::NAN);
    }
    let sd = (deltas.iter().map(|d| (d - mean) * (d - mean)).sum::<f64>() / (n - 1.0)).sqrt();
    let t = if sd > 0.0 { mean / (sd / n.sqrt()) } else { f64::NAN };
    (mean, sd, t)
}

#[test]
#[ignore = "loads the real ~26B-A4B MoE onto the wgpu backend three times in one process; set \
            NV_DEPTH_CORRECT=1 -- prices NV_G4MOE_FLASH_FOLDGATE=1 (fold-2/splits-64/sacc-128 \
            on full attention past the 10240 knee) over paired chat-frame windows at \
            256/16384/196608 against the zero-env control and against a splits-64 \
            reassociation null of the arm's own scope; 512 teacher-forced positions per \
            window, shuffled control gates each depth, engagement proved per row from the \
            dispatched grid; depths via NV_CTX_TOKENS, windows via NV_G4QUAL_WINDOWS"]
#[cfg(feature = "wgpu")]
fn g4moe_26b_a4b_wgpu_foldgate_arm_is_priced_over_paired_depth_windows_against_the_zero_env_control_and_a_splits64_reassociation_null(
) {
    use nv_models::gemma4_moe_wgpu as g4m;
    use std::collections::BTreeMap;
    if !suite_gate_runs_or_skips_loudly("g4moe-foldgate-paired") {
        return;
    }
    let _one_gpu_test_at_a_time = ctx_timing_common::serialize_tests_sharing_the_gpu_because_libtest_threads_double_book_the_card_reading_2x_slow_and_a_sibling_engines_teardown_surfaces_a_one_shot_cuda_error_invalid_value();
    let ctx = common::ctx();
    eprintln!("adapter: {}", ctx.summary());
    let dir = g4moe_snapshot_dir_env_override_then_home_hub();
    let label = ppl_common::checkpoint_label_from_snapshot_dir(&dir);
    let mut tok = tokenizers::Tokenizer::from_file(dir.join("tokenizer.json"))
        .expect("tokenizer.json ships with the checkpoint");
    tok.with_truncation(None)
        .expect("clear checkpoint-baked truncation so the deep windows survive tokenization");
    let (prefix, suffix) = template_spliced_prefix_suffix_ids(&dir, &tok);
    let corpus = g4qual_corpus_ids_whole_train_file(&tok, 1 << 20);
    let scored = G4QUAL_SCORED_512_MATCHES_THE_FAMILYS_CANONICAL_CHAT_PPL_CELL_WIDTH;
    let depths = depth_ladder(&[256, 16384, 196608]);
    let slots = depths.iter().copied().max().expect("nonempty depth list")
        + KV_SLOT_SLACK_1024_COVERS_SUFFIX_TF_AND_FREERUN_STEPS;

    let arms: [(&str, Option<&str>, Option<&str>, Option<&str>); 3] = [
        ("control-zero-env-default", None, None, None),
        (
            "reassociation-null-full-splits-64-sliding-pinned-32",
            None,
            Some(G4QUAL_NULL_FULL_SPLITS_64_MATCHES_THE_FOLD_ARMS_OWN_DEEP_SPLIT_COUNT_WITHOUT_THE_FOLD),
            Some(G4QUAL_NULL_SLIDING_SPLITS_PINNED_32_BECAUSE_THE_FOLDGATE_NEVER_TOUCHES_THE_SLIDING_KIND),
        ),
        ("foldgate-fold2-splits64-sacc128-full-attention-only", Some("1"), None, None),
    ];

    let mut cells: BTreeMap<(usize, usize), Vec<G4qualWindowCell>> = BTreeMap::new();
    let mut arm_basis: Vec<String> = Vec::new();
    for &depth in &depths {
        let windows = g4qual_independent_windows_at_depth_capped_by_env(corpus.len(), depth);
        for w in 0..windows {
            let (prefill_ids, targets) =
                g4qual_window_prefill_and_targets_disjoint_stride_depth_plus_scored(
                    &corpus, &prefix, &suffix, depth, w, scored,
                );
            let mut shared_state: Option<(Vec<Vec<u32>>, usize)> = None;
            for (arm_idx, (arm_name, foldgate, full_splits, sliding_splits)) in
                arms.iter().enumerate()
            {
                let pins = g4moe_arm_env_pins_zero_env_default_except_the_three_arm_slots(
                    *foldgate,
                    *full_splits,
                    *sliding_splits,
                );
                let built_full_splits =
                    g4m::flash_splits_grid_y_scratch_rows_and_stage1_stride_read_once_at_build();
                let mut m = g4moe_build_real_model_under_current_env(&dir, slots);
                let resolved =
                    g4moe_arm_label_from_resolved_gate_values_because_env_presence_lies_after_a_default_flip();
                if arm_basis.len() < arms.len() {
                    arm_basis.push(format!("{arm_name}: {resolved}"));
                }
                eprintln!(
                    "G4QUAL-ARM {label} arm={arm_name} depth={depth} window={w} resolved[{resolved}]"
                );
                let (blobs, bulk) = match &shared_state {
                    Some((blobs, bulk)) => {
                        g4qual_restore_shared_prefill_state(&mut m, blobs, *bulk);
                        (blobs, *bulk)
                    }
                    None => {
                        let fresh = g4qual_bulk_chunk_prefill_is_arm_independent_because_the_pf_passes_never_read_the_split_or_fold_envs(
                            &mut m,
                            &prefill_ids,
                        );
                        shared_state = Some(fresh);
                        let (b, p) = shared_state.as_ref().expect("just filled");
                        (b, *p)
                    }
                };
                let cell = g4qual_feed_tail_then_score_targets_on_this_arms_own_decode_path(
                    &mut m,
                    &prefill_ids[bulk..],
                    &targets,
                );
                assert_eq!(
                    m.current_pos(),
                    depth + scored - 1,
                    "the scored walk must end exactly at depth+scored-1"
                );
                g4qual_assert_arm_engaged_by_grid_not_env_presence(
                    &m,
                    arm_name,
                    depth,
                    foldgate.is_some(),
                    built_full_splits,
                );
                let ppl = (cell.sum_nll_f64 / cell.scored as f64).exp();
                let acc = cell.hits as f64 / cell.scored as f64;
                eprintln!(
                    "G4QUAL-CELL {label} arm={arm_name} depth={depth} window={w} \
                     scored={} ppl={ppl:.4} acc={acc:.4} sum_nll={:.6}",
                    cell.scored, cell.sum_nll_f64
                );
                let real_scored = cell.scored;
                cells.entry((arm_idx, depth)).or_default().push(cell);
                if arm_idx == 0 && w == 0 {
                    let mut shuffled = targets.clone();
                    ppl_common::deterministic_xorshift_fisher_yates_shuffle_fixed_seed_so_every_config_scores_the_same_control(&mut shuffled);
                    g4qual_restore_shared_prefill_state(&mut m, blobs, bulk);
                    let sh = g4qual_feed_tail_then_score_targets_on_this_arms_own_decode_path(
                        &mut m,
                        &prefill_ids[bulk..],
                        &shuffled,
                    );
                    ppl_common::print_machine_line_with_acc_and_assert_real_beats_shuffled(
                        &format!("g4moe-depth{depth}-chat-wgpu"),
                        &label,
                        real_scored,
                        ppl,
                        (sh.sum_nll_f64 / sh.scored as f64).exp(),
                        acc,
                    );
                }
                drop(m);
                drop(pins);
            }
        }
    }

    for &depth in &depths {
        let ctl = &cells[&(0usize, depth)];
        let null = &cells[&(1usize, depth)];
        let gate = &cells[&(2usize, depth)];
        let windows = ctl.len();
        assert!(null.len() == windows && gate.len() == windows);
        let pooled = |v: &[G4qualWindowCell]| {
            let s: f64 = v.iter().map(|c| c.sum_nll_f64).sum();
            let n: usize = v.iter().map(|c| c.scored).sum();
            let h: usize = v.iter().map(|c| c.hits).sum();
            ((s / n as f64).exp(), h as f64 / n as f64)
        };
        let (ppl_c, acc_c) = pooled(ctl);
        let (ppl_n, acc_n) = pooled(null);
        let (ppl_g, acc_g) = pooled(gate);
        let deltas = |a: &[G4qualWindowCell]| -> Vec<f64> {
            a.iter()
                .zip(ctl.iter())
                .map(|(x, c)| (x.sum_nll_f64 - c.sum_nll_f64) / c.scored as f64)
                .collect()
        };
        let (dn_mean, dn_sd, dn_t) = g4qual_paired_mean_sd_t_over_windows(&deltas(null));
        let (dg_mean, dg_sd, dg_t) = g4qual_paired_mean_sd_t_over_windows(&deltas(gate));
        let signs = |a: &[G4qualWindowCell]| -> (usize, usize) {
            let worse = a
                .iter()
                .zip(ctl.iter())
                .filter(|(x, c)| x.sum_nll_f64 > c.sum_nll_f64)
                .count();
            (worse, a.len() - worse)
        };
        let (gate_worse, gate_not_worse) = signs(gate);
        let (null_worse, null_not_worse) = signs(null);
        assert!(
            (acc_n - acc_c).abs()
                <= G4QUAL_A_REASSOCIATION_NULL_MAY_NOT_MOVE_TOP1_ACCURACY_BY_MORE_THAN_THIS_OR_THE_SPLIT_COUNT_IS_NOT_THE_SAME_ARITHMETIC,
            "depth {depth}: the splits-64 null moved top-1 accuracy from {acc_c:.4} to \
             {acc_n:.4}; a reduction-order change flips only near-ties, so this null is a \
             defect on the measurement knob, not a floor"
        );
        if !g4m::foldgate_deep_at_these_whole_context_tokens((depth + scored) as u32) {
            for (w, (g, c)) in gate.iter().zip(ctl.iter()).enumerate() {
                assert!(
                    g.sum_nll_f64.to_bits() == c.sum_nll_f64.to_bits() && g.hits == c.hits,
                    "depth {depth} window {w}: below its knee the armed foldgate is bit-for-bit \
                     the control, so a moved digit here (gate {} vs control {}) is harness \
                     error, not arm effect -- the instrument is wrong before any verdict",
                    g.sum_nll_f64,
                    c.sum_nll_f64
                );
            }
        }
        let dppl_g = (ppl_g / ppl_c - 1.0) * 100.0;
        let dppl_n = (ppl_n / ppl_c - 1.0) * 100.0;
        eprintln!(
            "G4QUAL-PAIR {label} depth={depth} windows={windows} scored_per_window={scored} \
             control[ppl={ppl_c:.4} acc={acc_c:.4}] null[ppl={ppl_n:.4} acc={acc_n:.4} \
             dppl_pct={dppl_n:+.3} dnll_mean={dn_mean:+.6} sd={dn_sd:.6} t={dn_t:+.2} \
             worse={null_worse}/{}] gate[ppl={ppl_g:.4} acc={acc_g:.4} dppl_pct={dppl_g:+.3} \
             dnll_mean={dg_mean:+.6} sd={dg_sd:.6} t={dg_t:+.2} worse={gate_worse}/{}] \
             dacc_gate={:+.4} dacc_null={:+.4}",
            null_worse + null_not_worse,
            gate_worse + gate_not_worse,
            acc_g - acc_c,
            acc_n - acc_c
        );
        let null_band = dn_mean.abs().max(if dn_sd.is_finite() {
            dn_mean.abs() + 2.0 * dn_sd / (windows as f64).sqrt()
        } else {
            dn_mean.abs()
        });
        let verdict = if !g4m::foldgate_deep_at_these_whole_context_tokens(depth as u32) {
            "EXACT-ZERO-CONTROL-CELL (bitwise below the knee, asserted)"
        } else if windows < 2 {
            "UNRESOLVED (one window: no spread exists, neither for the arm nor its null)"
        } else if !dg_t.is_finite() || dg_t.abs() < 2.0 || dg_mean.abs() <= null_band {
            "SHIP-CONDITION-MET (fp-reorder: indistinguishable from this scorer's own \
             reassociation null over the same windows)"
        } else if dg_mean > 0.0 {
            "RESOLVED-WORSE-BEYOND-NULL (a resolvable fp-reorder is not a pure reorder: \
             mechanism owed before any flip, section 5 instruments next)"
        } else {
            "RESOLVED-BETTER-BEYOND-NULL (a resolvable fp-reorder is not a pure reorder: \
             mechanism owed before any claim, section 5 instruments next)"
        };
        eprintln!("G4QUAL-VERDICT {label} depth={depth} class=fp-reorder verdict={verdict}");
    }
    eprintln!("G4QUAL-BASIS {label} corpus=NV_DEPTH_CORPUS-or-default arms:");
    for b in &arm_basis {
        eprintln!("G4QUAL-BASIS-ARM {b}");
    }
}

#[test]
fn tail_cycle_flags_period_2_repeated_4_times_and_ignores_progressing_text() {
    let looping: Vec<u32> = [7u32, 9].iter().cycle().take(16).copied().collect();
    assert_eq!(
        tail_cycle(&looping),
        Some((2, 8)),
        "period-2 x8 tail must be flagged"
    );
    let progressing: Vec<u32> = (0..64).collect();
    assert_eq!(tail_cycle(&progressing), None);
    assert_eq!(longest_immediate_run(&[1, 1, 1, 2, 2]), 3);
    assert_eq!(longest_immediate_run(&[1, 2, 3]), 1);
}

#[test]
fn nll_of_target_matches_hand_softmax_on_a_tiny_row() {
    let row = [1.0f32, 2.0, 3.0];
    let z: f64 = row.iter().map(|&v| (v as f64).exp()).sum();
    let want = z.ln() - 3.0;
    let got = nll_of_target_f64_lse(&row, 2);
    assert!(
        (got - want).abs() < 1e-9,
        "nll {got} != hand softmax {want}"
    );
}
