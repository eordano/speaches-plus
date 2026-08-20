#![cfg(feature = "cuda")]

mod common;
use common::argmax;
use common::Arm;
use common::build;
use common::snapshot;
use candle_core::Device;
use nv_models::graph_engine::GraphedQwen3Moe;
use nv_models::qwen3_5_moe::{Qwen3Moe, Qwen3MoeConfig};
use nv_weights::{QuantizationConfig, WeightLoader};
use std::path::PathBuf;
use std::time::Instant;
use tokenizers::Tokenizer;
use common::decode_fix_label_reports_the_nv_q36_graphed_decode_fix_gate_state as decode_kernel_label_reports_the_nv_q36_graphed_decode_fix_gate_state;

const DEGENERATE_BELOW_DISTINCT_RATIO_SAME_THRESHOLD_AS_QWEN36_CHAT_VALIDATE: f64 = 0.35;
const DEGENERATE_ABOVE_IMMEDIATE_RUN_SAME_THRESHOLD_AS_QWEN36_CHAT_VALIDATE: usize = 4;
const TAIL_CYCLE_REPEATS_SAME_THRESHOLD_AS_QWEN36_CHAT_VALIDATE: usize = 3;
const TAIL_CYCLE_MAX_PERIOD_16_SAME_AS_QWEN36_CHAT_VALIDATE: usize = 16;
const MIN_NEW_TOKENS_96_BECAUSE_THE_PRIOR_8_TOKEN_EXPOSURE_WAS_AN_UNDERPOWERED_DISCRIMINATOR:
    usize = 96;
const DEFAULT_FREE_RUNS_10_SO_THE_TABLE_IS_A_BASE_RATE_NOT_AN_ANECDOTE: usize = 10;
const MIN_RUNS_2_BELOW_WHICH_NO_RATE_IS_QUOTABLE: usize = 2;
const GRAPHED_PREFILL_CHUNK_512_MATCHES_THE_SERVING_CTX_PRIME_PATH_PLAIN_DISPATCH: usize = 512;
const RUN0_FULLTEXT_SNIPPET_2000_CHARS_COVERS_97_GREEDY_TOKENS_FOR_DRIFT_VS_EAGER_ARM: usize =
    2000;
const LONG_ANSWER_PROMPT_SIZED_SO_GREEDY_DECODE_DOES_NOT_HIT_EOS_BEFORE_96_TOKENS: &str =
    "Explain in detail, in at least three paragraphs, how photosynthesis converts sunlight \
     into chemical energy. Cover the light-dependent reactions, the Calvin cycle, and the \
     role of chlorophyll.";

fn longest_immediate_run(ids: &[u32]) -> usize {
    let (mut best, mut cur) = (1usize, 1usize);
    for w in ids.windows(2) {
        cur = if w[0] == w[1] { cur + 1 } else { 1 };
        best = best.max(cur);
    }
    best
}

fn tail_cycle(ids: &[u32]) -> Option<(usize, usize)> {
    for k in 1..=TAIL_CYCLE_MAX_PERIOD_16_SAME_AS_QWEN36_CHAT_VALIDATE {
        if ids.len() < k * 2 {
            break;
        }
        let tail = &ids[ids.len() - k..];
        let mut reps = 0usize;
        let mut j = ids.len();
        while j >= k && &ids[j - k..j] == tail {
            reps += 1;
            j -= k;
        }
        if reps >= TAIL_CYCLE_REPEATS_SAME_THRESHOLD_AS_QWEN36_CHAT_VALIDATE {
            return Some((k, reps));
        }
    }
    None
}

const ARMS: [Arm; 3] = [
    Arm {
        label: "grouped+routed",
        grouped: true,
        routing: true,
    },
    Arm {
        label: "grouped+unrouted",
        grouped: true,
        routing: false,
    },
    Arm {
        label: "plain",
        grouped: false,
        routing: false,
    },
];

struct RunRecord {
    ids: Vec<u32>,
    body_len: usize,
    stopped_on_eos: bool,
    ratio: f64,
    longest_run: usize,
    cycle: Option<(usize, usize)>,
    nonfinite_rows: usize,
    degenerate: bool,
    wall_s: f64,
}

fn first_divergence_index(a: &[u32], b: &[u32]) -> Option<usize> {
    let n = a.len().min(b.len());
    (0..n).find(|&i| a[i] != b[i]).or_else(|| {
        if a.len() == b.len() {
            None
        } else {
            Some(n)
        }
    })
}

fn one_line_snippet(text: &str, max_chars: usize) -> String {
    let flat: String = text
        .chars()
        .map(|c| if c == '\n' || c == '\r' { '\u{23CE}' } else { c })
        .collect();
    flat.chars().take(max_chars).collect()
}

fn graphed_decode_mode_from_env_nv_q95_arm_default_eager_unchanged() -> bool {
    match std::env::var("NV_Q95_ARM").as_deref() {
        Ok("graphed") => true,
        Ok("eager") | Err(_) => false,
        Ok(other) => panic!("unknown NV_Q95_ARM={other}; expected graphed|eager"),
    }
}

fn graphed_arm_prime_via_plain_chunked_prefill_then_install_grouped_moe_matching_serving_ctx(
    eng: &mut GraphedQwen3Moe,
    prompt: &[u32],
    run_idx: usize,
) -> Vec<f32> {
    eng.set_moe_dispatch(None);
    eng.reset().expect("reset");
    assert_eq!(
        eng.current_pos(),
        0,
        "run {run_idx}: reset must rewind the engine before the graphed arm primes"
    );
    let mut pos = 0usize;
    let mut last_row: Vec<f32> = Vec::new();
    while pos < prompt.len() {
        let n = GRAPHED_PREFILL_CHUNK_512_MATCHES_THE_SERVING_CTX_PRIME_PATH_PLAIN_DISPATCH
            .min(prompt.len() - pos);
        last_row = eng
            .prefill(&prompt[pos..pos + n])
            .unwrap_or_else(|e| panic!("run {run_idx} plain prefill chunk at pos {pos}: {e:#}"));
        pos += n;
    }
    assert_eq!(
        eng.current_pos(),
        prompt.len(),
        "run {run_idx}: chunked prefill must leave the engine at the prompt depth"
    );
    let nan_in_last_row = last_row.iter().filter(|v| v.is_nan()).count();
    assert_eq!(
        nan_in_last_row, 0,
        "run {run_idx}: plain-dispatch chunked prefill produced {nan_in_last_row} NaN logits; \
         the all-NaN failure belongs to grouped eager prefill, which the graphed arm avoids by \
         installing grouped MoE only after priming"
    );
    eng.install_grouped_moe()
        .unwrap_or_else(|e| panic!("run {run_idx} grouped moe install after plain prefill: {e:#}"));
    last_row
}

#[test]
#[ignore = "loads the ~22 GB Qwen3.6 checkpoint; set NV_QWEN36_FREERUN_BASE=1 -- N greedy free-runs of >=96 new tokens each, records per-run distinct ratio and a base-rate table, NEVER asserts output quality so a degenerate run is data not a test failure; deliberately takes NO cross-process GPU lock so a co-tenant arm can share the card; NV_Q95_ARM=graphed replays the serving-ctx pattern (plain chunked prefill, grouped MoE installed after priming, CUDA-graph capture at first decode) because every clean repro so far ran EAGER and the historical 0.062 failure was presumably GRAPHED (the #95 discriminator)"]
fn qwen36_free_run_base_rate_records_degeneration_per_run_and_never_asserts_quality() {
    if std::env::var("NV_QWEN36_FREERUN_BASE").as_deref() != Ok("1") {
        panic!("set NV_QWEN36_FREERUN_BASE=1 to run (it must never silently skip)");
    }
    let runs: usize = std::env::var("NV_Q95_RUNS")
        .ok()
        .map(|v| v.parse().expect("NV_Q95_RUNS"))
        .unwrap_or(DEFAULT_FREE_RUNS_10_SO_THE_TABLE_IS_A_BASE_RATE_NOT_AN_ANECDOTE);
    assert!(
        runs >= MIN_RUNS_2_BELOW_WHICH_NO_RATE_IS_QUOTABLE,
        "NV_Q95_RUNS={runs} cannot support a base rate"
    );
    let new_tokens: usize = std::env::var("NV_Q95_NEW_TOKENS")
        .ok()
        .map(|v| v.parse().expect("NV_Q95_NEW_TOKENS"))
        .unwrap_or(MIN_NEW_TOKENS_96_BECAUSE_THE_PRIOR_8_TOKEN_EXPOSURE_WAS_AN_UNDERPOWERED_DISCRIMINATOR);
    assert!(
        new_tokens >= MIN_NEW_TOKENS_96_BECAUSE_THE_PRIOR_8_TOKEN_EXPOSURE_WAS_AN_UNDERPOWERED_DISCRIMINATOR,
        "NV_Q95_NEW_TOKENS={new_tokens} is below the 96-token exposure floor"
    );
    let cotenant_label =
        std::env::var("NV_Q95_COTENANT").unwrap_or_else(|_| "unlabeled".into());

    let dir = snapshot();
    assert!(
        dir.join("model.safetensors").is_file()
            || dir.join("model.safetensors.index.json").is_file(),
        "checkpoint {dir:?} has no weights"
    );
    let tok = Tokenizer::from_file(dir.join("tokenizer.json")).expect("tokenizer.json");
    let eos: Vec<u32> = ["<|im_end|>", "<|endoftext|>"]
        .iter()
        .filter_map(|s| tok.token_to_id(s))
        .collect();
    assert_eq!(eos.len(), 2, "expected both stop tokens in this vocab");

    let q = std::env::var("NV_QWEN36_Q").unwrap_or_else(|_| {
        LONG_ANSWER_PROMPT_SIZED_SO_GREEDY_DECODE_DOES_NOT_HIT_EOS_BEFORE_96_TOKENS.into()
    });
    let prompt_text =
        format!("<|im_start|>user\n{q}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n");
    let prompt: Vec<u32> = tok
        .encode(prompt_text.as_str(), false)
        .expect("encode")
        .get_ids()
        .to_vec();
    assert!(!prompt.is_empty(), "empty prompt encoding");

    let device = Device::new_cuda_with_stream(0).expect("cuda");
    let graphed = graphed_decode_mode_from_env_nv_q95_arm_default_eager_unchanged();
    let decode_mode_label = if graphed {
        assert!(
            std::env::var("NV_QWEN36_ARM").is_err(),
            "NV_Q95_ARM=graphed fixes the dispatch schedule (plain prefill, grouped captured \
             decode); unset NV_QWEN36_ARM instead of combining them"
        );
        "graphed_plain_chunk_prefill_grouped_captured_decode"
    } else {
        "eager"
    };
    let want = std::env::var("NV_QWEN36_ARM").unwrap_or_else(|_| {
        if graphed {
            "plain".into()
        } else {
            "grouped+routed".into()
        }
    });
    let arm = ARMS
        .iter()
        .find(|a| a.label == want)
        .unwrap_or_else(|| panic!("unknown NV_QWEN36_ARM={want}"));

    let t_load = Instant::now();
    let mut eng = build(&dir, &device, arm);
    let load_s = t_load.elapsed().as_secs_f64();
    eprintln!(
        "Q95-BASIS checkpoint={dir:?} arm={} decode_mode={decode_mode_label} decode_kernel={} capture_active={} cotenant={cotenant_label} prompt_toks={} new_toks_target={new_tokens} runs={runs} load_s={load_s:.1} decode=greedy_argmax_feed chat_template=im_start_user_assistant_empty_think",
        arm.label,
        decode_kernel_label_reports_the_nv_q36_graphed_decode_fix_gate_state(),
        eng.capture_active(),
        prompt.len()
    );

    let mut records: Vec<RunRecord> = Vec::with_capacity(runs);
    for i in 0..runs {
        let t_run = Instant::now();
        let row = if graphed {
            graphed_arm_prime_via_plain_chunked_prefill_then_install_grouped_moe_matching_serving_ctx(
                &mut eng, &prompt, i,
            )
        } else {
            eng.reset().expect("reset");
            eng.prefill(&prompt).expect("prefill")
        };
        assert_eq!(row.len(), eng.vocab_size(), "logit row width");
        let mut nonfinite_rows = if row.iter().any(|v| !v.is_finite()) {
            1
        } else {
            0
        };
        let mut cur = argmax(&row);
        let mut out: Vec<u32> = vec![cur];
        let mut stopped = false;
        for step in 0..new_tokens {
            if eos.contains(&cur) {
                stopped = true;
                break;
            }
            eng.forward_decode(cur)
                .unwrap_or_else(|e| panic!("run {i} decode step {step}: {e:#}"));
            if graphed && step == 0 {
                assert!(
                    eng.capture_active(),
                    "run {i}: the graphed arm's basis is the CAPTURED decode and the engine \
                     fell back to uncaptured at the first step; the engine printed its blocker \
                     to stderr just above -- diagnose that blocker, do not read this run as the \
                     graphed discriminator"
                );
            }
            let r = eng.logits_host().expect("logits");
            if r.iter().any(|v| !v.is_finite()) {
                nonfinite_rows += 1;
            }
            cur = argmax(&r);
            out.push(cur);
        }
        let wall_s = t_run.elapsed().as_secs_f64();

        let body: Vec<u32> = out.iter().copied().filter(|t| !eos.contains(t)).collect();
        let distinct = {
            let mut v = body.clone();
            v.sort_unstable();
            v.dedup();
            v.len()
        };
        let ratio = distinct as f64 / body.len().max(1) as f64;
        let run_len = longest_immediate_run(&body);
        let cyc = tail_cycle(&body);
        let degenerate = ratio < DEGENERATE_BELOW_DISTINCT_RATIO_SAME_THRESHOLD_AS_QWEN36_CHAT_VALIDATE
            || run_len > DEGENERATE_ABOVE_IMMEDIATE_RUN_SAME_THRESHOLD_AS_QWEN36_CHAT_VALIDATE
            || cyc.is_some()
            || nonfinite_rows > 0;
        let divergence = records
            .first()
            .map(|r0| first_divergence_index(&r0.ids, &out));
        let text = tok.decode(&body, false).expect("decode");
        eprintln!(
            "Q95-RUN idx={i} new_toks={} stopped_on_eos={stopped} distinct={distinct}/{} ratio={ratio:.3} longest_run={run_len} tail_cycle={cyc:?} nonfinite_rows={nonfinite_rows} first_divergence_vs_run0={divergence:?} degenerate={degenerate} wall_s={wall_s:.1} capture_active={}",
            out.len(),
            body.len(),
            eng.capture_active()
        );
        eprintln!("Q95-TEXT idx={i} {}", one_line_snippet(&text, 200));
        if i == 0 {
            eprintln!(
                "Q95-RUN0-FULLTEXT decode_mode={decode_mode_label} {}",
                one_line_snippet(
                    &text,
                    RUN0_FULLTEXT_SNIPPET_2000_CHARS_COVERS_97_GREEDY_TOKENS_FOR_DRIFT_VS_EAGER_ARM
                )
            );
        }
        records.push(RunRecord {
            ids: out,
            body_len: body.len(),
            stopped_on_eos: stopped,
            ratio,
            longest_run: run_len,
            cycle: cyc,
            nonfinite_rows,
            degenerate,
            wall_s,
        });
    }

    let degenerate_count = records.iter().filter(|r| r.degenerate).count();
    let eos_before_full = records
        .iter()
        .filter(|r| r.stopped_on_eos && r.body_len < new_tokens)
        .count();
    let diverged_from_run0 = records[1..]
        .iter()
        .filter(|r| first_divergence_index(&records[0].ids, &r.ids).is_some())
        .count();
    let mut ratios: Vec<f64> = records.iter().map(|r| r.ratio).collect();
    ratios.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let total_wall: f64 = records.iter().map(|r| r.wall_s).sum();
    let worst_run = records
        .iter()
        .map(|r| r.longest_run)
        .max()
        .expect("at least one run");
    let any_cycle = records.iter().filter(|r| r.cycle.is_some()).count();
    let any_nonfinite: usize = records.iter().map(|r| r.nonfinite_rows).sum();
    eprintln!(
        "Q95-TABLE cotenant={cotenant_label} arm={} decode_mode={decode_mode_label} decode_kernel={} runs={runs} degenerate={degenerate_count} eos_before_full={eos_before_full} diverged_from_run0={diverged_from_run0} ratio_min={:.3} ratio_med={:.3} ratio_max={:.3} worst_immediate_run={worst_run} runs_with_tail_cycle={any_cycle} nonfinite_rows_total={any_nonfinite} decode_wall_s={total_wall:.1} basis=greedy_argmax_free_run_{new_tokens}tok_chat_template_thresholds_ratio_lt_0p35_or_run_gt_4_or_cycle_or_nonfinite",
        arm.label,
        decode_kernel_label_reports_the_nv_q36_graphed_decode_fix_gate_state(),
        ratios[0],
        ratios[ratios.len() / 2],
        ratios[ratios.len() - 1]
    );
    assert!(
        records.iter().all(|r| r.body_len >= 1),
        "a run emitted zero body tokens; the discriminator has no exposure"
    );
}
