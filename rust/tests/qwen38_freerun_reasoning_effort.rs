#![cfg(feature = "cuda")]

#[path = "common/qwen38_fixture.rs"]
#[allow(dead_code)]
mod qwen38_fixture;

use candle_core::{Device, Tensor};
use nv_models::qwen3_5_moe::Qwen3Moe;
use qwen38_fixture::{
    eos_ids_from_generation_config, host_row_f32, load_qwen38_dense_on_the_cuda_serving_arm,
    qwen38_nvfp4_snapshot_dir_env_override_then_home_hub,
};
use speaches_plus::oapi::chat_template::ChatTemplate;
use std::time::Instant;
use tokenizers::Tokenizer;

const DEGENERATE_BELOW_DISTINCT_RATIO_SAME_THRESHOLD_AS_QWEN36_CHAT_VALIDATE: f64 = 0.35;
const DEGENERATE_ABOVE_IMMEDIATE_RUN_SAME_THRESHOLD_AS_QWEN36_CHAT_VALIDATE: usize = 4;
const TAIL_CYCLE_REPEATS_SAME_THRESHOLD_AS_QWEN36_CHAT_VALIDATE: usize = 3;
const TAIL_CYCLE_MAX_PERIOD_16_SAME_AS_QWEN36_CHAT_VALIDATE: usize = 16;
const DEFAULT_NEW_TOKENS_128_ABOVE_THE_QWEN36_96_FLOOR_BECAUSE_THINKING_ON_RUNS_MUST_ALSO_EXPOSE_THE_THINK_CLOSE:
    usize = 128;
const MIN_NEW_TOKENS_96_THE_QWEN36_UNDERPOWERED_DISCRIMINATOR_FLOOR: usize = 96;
const DEFAULT_FREE_RUNS_3_PER_EFFORT_BECAUSE_GREEDY_DECODE_MAKES_EXTRA_RUNS_A_NONDETERMINISM_PROBE_NOT_NEW_EXPOSURE:
    usize = 3;
const MIN_RUNS_2_BELOW_WHICH_NO_RATE_IS_QUOTABLE: usize = 2;
const REASONING_EFFORTS_UNDER_TEST_LOW_AND_MEDIUM_XHIGH_STAYS_AN_EXPLICIT_REQUEST: [&str; 2] =
    ["low", "medium"];
const THINK_CLOSE_LITERAL_THE_QWEN38_TEMPLATE_TRAINS: &str = "</think>";
const LONG_ANSWER_PROMPT_SIZED_SO_GREEDY_DECODE_DOES_NOT_HIT_EOS_EARLY: &str =
    "Explain in detail, in at least three paragraphs, how photosynthesis converts sunlight \
     into chemical energy. Cover the light-dependent reactions, the Calvin cycle, and the \
     role of chlorophyll.";

fn argmax(row: &[f32]) -> u32 {
    let (mut bi, mut bv) = (0u32, f32::NEG_INFINITY);
    for (i, &v) in row.iter().enumerate() {
        if v > bv {
            bv = v;
            bi = i as u32;
        }
    }
    bi
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

fn one_line_snippet(text: &str, max_chars: usize) -> String {
    let flat: String = text
        .chars()
        .map(|c| if c == '\n' || c == '\r' { '\u{23CE}' } else { c })
        .collect();
    flat.chars().take(max_chars).collect()
}

fn render_thinking_prompt_via_the_real_template_at(
    template: &ChatTemplate,
    effort: &str,
) -> String {
    let msgs = serde_json::json!([{
        "role": "user",
        "content": LONG_ANSWER_PROMPT_SIZED_SO_GREEDY_DECODE_DOES_NOT_HIT_EOS_EARLY
    }]);
    let mut kwargs = template.effective_template_kwargs();
    kwargs.insert(
        speaches_plus::oapi::chat_template::REASONING_EFFORT_KWARG.to_string(),
        serde_json::json!(effort),
    );
    let prompt = template
        .render_with_kwargs(&msgs, None, true, &kwargs)
        .unwrap_or_else(|e| panic!("render real qwen3.8 template at effort {effort}: {e:#}"));
    assert!(
        prompt.trim_end().ends_with("<think>"),
        "with enable_thinking left undefined the real template must OPEN a thought block at \
         every effort; a prompt that does not is a hand-built or mis-kwarged render: {prompt:?}"
    );
    match effort {
        "low" => assert!(
            prompt.contains("Reasoning effort is set to low"),
            "{prompt:?}"
        ),
        "medium" => assert!(
            !prompt.contains("Reasoning effort is set to"),
            "medium is the template's silent arm: {prompt:?}"
        ),
        other => panic!("unexpected effort under test {other}"),
    }
    prompt
}

struct RunRecord {
    ratio: f64,
    longest_run: usize,
    cycle: Option<(usize, usize)>,
    nonfinite_rows: usize,
    degenerate: bool,
    think_closed: bool,
    leaked: bool,
    body_len: usize,
    wall_s: f64,
}

fn free_run_greedy_via_the_eager_serving_forwards(
    model: &Qwen3Moe,
    device: &Device,
    prompt_ids: &[u32],
    eos: &[u32],
    new_tokens: usize,
) -> (Vec<u32>, usize, bool) {
    let mut cache = model
        .new_kv_cache(prompt_ids.len() + new_tokens + 8)
        .expect("kv cache");
    let k = prompt_ids.len();
    let tokens = Tensor::from_vec(prompt_ids.to_vec(), (1usize, k), device).expect("tokens");
    let positions =
        Tensor::from_vec((0..k as i32).collect::<Vec<_>>(), k, device).expect("positions");
    let logits = model
        .forward_with_cache(&tokens, &positions, &mut cache)
        .expect("prefill");
    let flat = host_row_f32(&logits);
    let vocab = flat.len() / k;
    let row = &flat[(k - 1) * vocab..k * vocab];
    let mut nonfinite_rows = usize::from(row.iter().any(|v| !v.is_finite()));
    let mut cur = argmax(row);
    let mut out: Vec<u32> = vec![cur];
    let mut stopped = false;
    for step in 0..new_tokens {
        if eos.contains(&cur) {
            stopped = true;
            break;
        }
        let pos = (k + step) as i32;
        let t = Tensor::from_vec(vec![cur], (1usize, 1usize), device).expect("token");
        let p = Tensor::from_vec(vec![pos], 1usize, device).expect("position");
        let logits = model
            .forward_with_cache(&t, &p, &mut cache)
            .unwrap_or_else(|e| panic!("decode step {step}: {e:#}"));
        let row = host_row_f32(&logits);
        if row.iter().any(|v| !v.is_finite()) {
            nonfinite_rows += 1;
        }
        cur = argmax(&row);
        out.push(cur);
    }
    (out, nonfinite_rows, stopped)
}

#[test]
#[ignore = "loads the ~16 GB Qwen3.8-27B NVFP4 checkpoint; set NV_QWEN38_FREERUN=1 -- greedy free-runs through the REAL chat template at reasoning_effort low and medium (thinking ON, the served default family), records per-run distinct ratio, think-close arrival and visible-text leakage as a base-rate table; degeneration or template leakage here is the #95-class failure this suite hunts, and quality is recorded rather than asserted"]
fn qwen38_free_run_base_rate_via_the_real_template_at_low_and_medium_reasoning_effort() {
    if std::env::var("NV_QWEN38_FREERUN").as_deref() != Ok("1") {
        panic!("set NV_QWEN38_FREERUN=1 to run (it must never silently skip)");
    }
    let runs: usize = std::env::var("NV_Q38_RUNS")
        .ok()
        .map(|v| v.parse().expect("NV_Q38_RUNS"))
        .unwrap_or(DEFAULT_FREE_RUNS_3_PER_EFFORT_BECAUSE_GREEDY_DECODE_MAKES_EXTRA_RUNS_A_NONDETERMINISM_PROBE_NOT_NEW_EXPOSURE);
    assert!(
        runs >= MIN_RUNS_2_BELOW_WHICH_NO_RATE_IS_QUOTABLE,
        "NV_Q38_RUNS={runs} cannot support a base rate"
    );
    let new_tokens: usize = std::env::var("NV_Q38_NEW_TOKENS")
        .ok()
        .map(|v| v.parse().expect("NV_Q38_NEW_TOKENS"))
        .unwrap_or(DEFAULT_NEW_TOKENS_128_ABOVE_THE_QWEN36_96_FLOOR_BECAUSE_THINKING_ON_RUNS_MUST_ALSO_EXPOSE_THE_THINK_CLOSE);
    assert!(
        new_tokens >= MIN_NEW_TOKENS_96_THE_QWEN36_UNDERPOWERED_DISCRIMINATOR_FLOOR,
        "NV_Q38_NEW_TOKENS={new_tokens} is below the 96-token exposure floor"
    );

    let dir = qwen38_nvfp4_snapshot_dir_env_override_then_home_hub();
    let tok = Tokenizer::from_file(dir.join("tokenizer.json")).expect("tokenizer.json");
    let template = ChatTemplate::load(&dir).expect(
        "the qwen3.8 snapshot ships chat_template.jinja and it must compile under minijinja; a \
         hand-built prompt here would be the #95-class artifact this suite exists to prevent",
    );
    assert_eq!(
        template.thinking_close_marker().as_deref(),
        Some(THINK_CLOSE_LITERAL_THE_QWEN38_TEMPLATE_TRAINS),
        "the close-marker derivation must agree with the literal this suite splits on"
    );
    let eos = eos_ids_from_generation_config(&dir);

    let device = Device::new_cuda_with_stream(0).expect("cuda");
    let t_load = Instant::now();
    let model = load_qwen38_dense_on_the_cuda_serving_arm(&dir, &device);
    let load_s = t_load.elapsed().as_secs_f64();

    for effort in REASONING_EFFORTS_UNDER_TEST_LOW_AND_MEDIUM_XHIGH_STAYS_AN_EXPLICIT_REQUEST {
        let prompt_text = render_thinking_prompt_via_the_real_template_at(&template, effort);
        let prompt_ids: Vec<u32> = tok
            .encode(prompt_text.as_str(), false)
            .expect("encode")
            .get_ids()
            .to_vec();
        assert!(!prompt_ids.is_empty(), "empty prompt encoding");
        eprintln!(
            "Q38-BASIS checkpoint={dir:?} arm=eager_dense_cuda effort={effort} \
             prompt_toks={} new_toks_target={new_tokens} runs={runs} load_s={load_s:.1} \
             decode=greedy_argmax_feed chat_template=real_chat_template_jinja_thinking_on",
            prompt_ids.len()
        );

        let mut records: Vec<RunRecord> = Vec::with_capacity(runs);
        for i in 0..runs {
            let t_run = Instant::now();
            let (out, nonfinite_rows, stopped) = free_run_greedy_via_the_eager_serving_forwards(
                &model,
                &device,
                &prompt_ids,
                &eos,
                new_tokens,
            );
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
            let degenerate = ratio
                < DEGENERATE_BELOW_DISTINCT_RATIO_SAME_THRESHOLD_AS_QWEN36_CHAT_VALIDATE
                || run_len > DEGENERATE_ABOVE_IMMEDIATE_RUN_SAME_THRESHOLD_AS_QWEN36_CHAT_VALIDATE
                || cyc.is_some()
                || nonfinite_rows > 0;
            let text = tok.decode(&body, false).expect("decode");
            let (think_closed, visible) =
                match text.split_once(THINK_CLOSE_LITERAL_THE_QWEN38_TEMPLATE_TRAINS) {
                    Some((_reasoning, visible)) => (true, visible.to_string()),
                    None => (false, String::new()),
                };
            let leaked = visible.contains("<think>")
                || visible.contains(THINK_CLOSE_LITERAL_THE_QWEN38_TEMPLATE_TRAINS)
                || visible.contains("<|im_start|>");
            eprintln!(
                "Q38-RUN effort={effort} idx={i} new_toks={} stopped_on_eos={stopped} \
                 distinct={distinct}/{} ratio={ratio:.3} longest_run={run_len} \
                 tail_cycle={cyc:?} nonfinite_rows={nonfinite_rows} degenerate={degenerate} \
                 think_closed={think_closed} visible_chars={} leak={leaked} wall_s={wall_s:.1}",
                out.len(),
                body.len(),
                visible.len()
            );
            eprintln!("Q38-TEXT effort={effort} idx={i} {}", one_line_snippet(&text, 300));
            records.push(RunRecord {
                ratio,
                longest_run: run_len,
                cycle: cyc,
                nonfinite_rows,
                degenerate,
                think_closed,
                leaked,
                body_len: body.len(),
                wall_s,
            });
        }

        let degenerate_count = records.iter().filter(|r| r.degenerate).count();
        let leak_runs = records.iter().filter(|r| r.leaked).count();
        let think_closed_runs = records.iter().filter(|r| r.think_closed).count();
        let mut ratios: Vec<f64> = records.iter().map(|r| r.ratio).collect();
        ratios.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let worst_run = records.iter().map(|r| r.longest_run).max().expect("runs");
        let any_cycle = records.iter().filter(|r| r.cycle.is_some()).count();
        let any_nonfinite: usize = records.iter().map(|r| r.nonfinite_rows).sum();
        let total_wall: f64 = records.iter().map(|r| r.wall_s).sum();
        eprintln!(
            "Q38-TABLE effort={effort} runs={runs} degenerate={degenerate_count} \
             leak_runs={leak_runs} think_closed_runs={think_closed_runs} \
             ratio_min={:.3} ratio_med={:.3} ratio_max={:.3} worst_immediate_run={worst_run} \
             runs_with_tail_cycle={any_cycle} nonfinite_rows_total={any_nonfinite} \
             decode_wall_s={total_wall:.1} \
             basis=greedy_argmax_free_run_{new_tokens}tok_real_template_thinking_on_thresholds_ratio_lt_0p35_or_run_gt_4_or_cycle_or_nonfinite",
            ratios[0],
            ratios[ratios.len() / 2],
            ratios[ratios.len() - 1]
        );
        assert!(
            records.iter().all(|r| r.body_len >= 1),
            "a run emitted zero body tokens; the discriminator has no exposure"
        );
    }
}
