#![cfg(feature = "cuda")]

#[path = "common/qwen38_fixture.rs"]
#[allow(dead_code)]
mod qwen38_fixture;

use std::sync::{Arc, Mutex};
use std::time::Instant;

use axum::body::Body;
use axum::http::{Request, StatusCode};
use axum::routing::post;
use axum::Router;
use futures_util::StreamExt;
use tower::ServiceExt;

use qwen38_fixture::qwen38_nvfp4_snapshot_dir_env_override_then_home_hub;
use speaches_plus::oapi::chat::{handle_chat_completions, ChatAppState, ChatEngine};
use speaches_plus::oapi::chat_engine::{ChatRegistry, NvEngineChat};

const REAL_WEIGHTS_TEST_ENV: &str = "NV_Q38_CONC_LADDER_TEST";

const SIXTEEN_DISTINCT_PROMPTS_SO_NO_TWO_LANES_SHARE_A_PREFIX_AND_A_CROSSED_LANE_SHOWS_AS_A_BYTE_DIFF:
    [&str; 16] = [
    "Write one paragraph of about 150 words describing a lighthouse keeper's morning routine.",
    "Write one paragraph of about 150 words explaining why the sky appears blue during the day.",
    "Write one paragraph of about 150 words describing how a river shapes a valley over time.",
    "Write one paragraph of about 150 words explaining how honeybees communicate about food.",
    "Write one paragraph of about 150 words describing the smell of a bakery before sunrise.",
    "Write one paragraph of about 150 words explaining how a suspension bridge carries its load.",
    "Write one paragraph of about 150 words describing the first snowfall in a mountain village.",
    "Write one paragraph of about 150 words explaining why tides follow the moon around the earth.",
    "Write one paragraph of about 150 words describing an old clockmaker repairing a pocket watch.",
    "Write one paragraph of about 150 words explaining how a glacier carves out a fjord.",
    "Write one paragraph of about 150 words describing a night market in a coastal city.",
    "Write one paragraph of about 150 words explaining why autumn leaves turn red and gold.",
    "Write one paragraph of about 150 words describing a violin luthier tuning a finished instrument.",
    "Write one paragraph of about 150 words explaining how sourdough starter leavens bread.",
    "Write one paragraph of about 150 words describing the last ferry crossing of a winter evening.",
    "Write one paragraph of about 150 words explaining how a lock and dam raise a barge upstream.",
];

const MAX_TOKENS_64_KEEPS_THE_WHOLE_LADDER_INSIDE_ONE_GPUQ_JOB_AND_STILL_OUTLASTS_PREFILL: u32 = 64;

const CONCURRENCY_LADDER_SPANS_SOLO_PARTIAL_GROUP_FULL_GROUP_AND_TWO_WAVES: [usize; 4] =
    [1, 2, 4, 8];

const FOUR_LANE_BUCKET_PLAN_IS_THE_SHIPPED_DEFAULT_NV_Q38_BATCH_SIZES: &str = "1,2,4";

const EIGHT_LANE_BUCKET_PLAN_ASKS_THE_ENGINE_FOR_A_TRUE_EIGHT_WIDE_GROUP_AND_TWO_WAVES_AT_SIXTEEN: &str = "1,2,4,8";

const SIXTEEN_LANE_BUCKET_PLAN_MAKES_C16_ONE_GROUP_INSTEAD_OF_TWO_WAVES_AND_NEEDS_NV_Q38_BATCH_GEMM_1: &str =
    "1,2,4,8,16";

const A_SIXTEEN_WIDE_GROUP_IS_THE_ONLY_HONEST_C16_ROW_BECAUSE_TWO_EIGHT_WIDE_WAVES_PAY_FORMATION_TWICE_AND_DECODE_AT_HALF_THE_BATCH:
    &str = "vLLM serves c=16 as one running batch. Two 8-wide waves under NV_Q38_BATCH_SIZES=1,2,4,8 \
            pay group formation twice and never reach B=16 in the step, so that row measures the \
            scheduler's wave policy, not the engine at 16. The m<=16 mk template arms admit a \
            single 16-group under NV_Q38_BATCH_GEMM=1, and that is the row this ladder reports";

const NV_Q38_BATCH_GEMM_1_IS_REQUIRED_BY_THE_SIXTEEN_LANE_POOL_AND_HELD_FOR_EVERY_LEVEL_SO_ONE_ARM_SPANS_THE_LADDER:
    &str = "Qwen38BatchLanes::new refuses a bucket above 8 without NV_Q38_BATCH_GEMM=1, and a \
            ladder whose c=16 row runs a different GEMM arm than its c=8 row is two ladders \
            spliced together";

const THE_GROUP_ARM_IS_BIT_IDENTICAL_TO_EAGER_M1_STEPPING_ONLY_UNDER_NV_Q38_BATCH_ROWWISE_FFN: &str =
    "the nvfp4 MLP m-row route is the sole non-twin kernel in the batch step, so the lane group \
     is byte-class against eager m=1 stepping only when NV_Q38_BATCH_ROWWISE=ffn splits the ffn \
     back into per-row m=1 twins; without it the group is tolerance-class and no byte assertion \
     is honest";

const MTP_IS_OFF_FOR_THE_WHOLE_LADDER_SO_ONE_EAGER_REFERENCE_SERVES_THE_SOLO_AND_THE_GROUP_ARM:
    &str = "NV_DRAFTER is unset for every engine in this suite: the batch scheduler's solo route \
            and the flag-off engine then run the same run_qwen38_dense_solo_body with mtp=None, \
            so a single eager sequential reference set is the oracle for c=1 and for every lane \
            of every group, instead of the two reference classes an MTP-on run would need";

const SCHEDULER_GROUP_FORMED_EVENT_IS_THE_PER_WAVE_MARKER_THE_SERVING_PATH_ALREADY_EMITS: &str =
    "qwen3.8 batch group formed";

const SCHEDULER_GROUP_DRAINED_EVENT_PROVES_A_WAVE_FINISHED_BEFORE_THE_NEXT_ONE_FORMED: &str =
    "qwen3.8 batch group drained";

const A_TRACING_SINK_READS_THE_WAVE_COUNT_OFF_THE_SHIPPED_SCHEDULER_EVENTS_SO_NO_SRC_ACCESSOR_IS_ADDED:
    &str = "run_group already emits one info event at formation carrying group=<size> and one at \
            drain; capturing the subscriber output is a complete and ordered wave log, so the \
            wave assertions need nothing added to the scheduler";

const WALL_BOUND_S_FOR_A_WHOLE_LEVEL_240_IS_TWENTY_TIMES_THE_EXPECTED_TWO_WAVE_DECODE_SO_ONLY_A_STARVED_REQUEST_TRIPS_IT:
    f64 = 240.0;

const P95_NEAREST_RANK_BECAUSE_A_LEVEL_HAS_AT_MOST_SIXTEEN_SAMPLES_AND_INTERPOLATION_INVENTS_VALUES:
    f64 = 0.95;

const ROWWISE_PINS_TRIED_FOR_AN_EIGHT_WIDE_GROUP_NARROWEST_FIRST: [&str; 2] = ["ffn", "all"];

const ONE_ENGINE_PER_ROWWISE_PIN_BECAUSE_THE_CAPTURED_GRAPH_KEY_IS_THE_BUCKET_ALONE: &str =
    "step_batch_captured keys its graph cache on the bucket width alone, so flipping \
     NV_Q38_BATCH_ROWWISE inside one process replays the graph captured under the previous pin \
     and the flip changes nothing; each pin therefore gets its own engine";

const A_LOUD_EIGHT_WIDE_REFUSAL_MUST_NAME_ONE_OF_THESE_CONSTRAINTS_OR_IT_IS_A_SILENT_FALLBACK:
    [&str; 10] = [
    "bucket",
    "lane",
    "graph",
    "capture",
    "memory",
    "alloc",
    "out of memory",
    "row",
    "batch",
    "NV_Q38_BATCH",
];

fn app(engine: Arc<dyn ChatEngine>) -> Router {
    Router::new()
        .route("/v1/chat/completions", post(handle_chat_completions))
        .with_state(ChatAppState {
            registry: ChatRegistry::single(engine),
        })
}

#[derive(Clone)]
struct SchedulerEventSink(Arc<Mutex<String>>);

impl SchedulerEventSink {
    fn new() -> Self {
        Self(Arc::new(Mutex::new(String::new())))
    }

    fn mark(&self) -> usize {
        self.0.lock().unwrap_or_else(|e| e.into_inner()).len()
    }

    fn waves_since(&self, mark: usize) -> Vec<WaveEvent> {
        let buf = self.0.lock().unwrap_or_else(|e| e.into_inner());
        wave_events_in(&buf[mark.min(buf.len())..])
    }
}

impl std::io::Write for SchedulerEventSink {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        self.0
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .push_str(&String::from_utf8_lossy(buf));
        let mut err = std::io::stderr();
        err.write_all(buf)?;
        Ok(buf.len())
    }

    fn flush(&mut self) -> std::io::Result<()> {
        std::io::stderr().flush()
    }
}

impl<'a> tracing_subscriber::fmt::MakeWriter<'a> for SchedulerEventSink {
    type Writer = SchedulerEventSink;

    fn make_writer(&'a self) -> Self::Writer {
        self.clone()
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum WaveEvent {
    Formed(usize),
    Drained(usize),
}

fn group_field_of(line: &str) -> usize {
    let at = line.find("group=").unwrap_or_else(|| {
        panic!("a scheduler wave line carried no group= field, so the wave size is unreadable: {line:?}")
    });
    let digits: String = line[at + "group=".len()..]
        .chars()
        .take_while(|c| c.is_ascii_digit())
        .collect();
    digits.parse().unwrap_or_else(|e| {
        panic!("scheduler wave line group= is not a number ({e}): {line:?}")
    })
}

fn wave_events_in(log: &str) -> Vec<WaveEvent> {
    let mut out = Vec::new();
    for line in log.lines() {
        if line.contains(SCHEDULER_GROUP_FORMED_EVENT_IS_THE_PER_WAVE_MARKER_THE_SERVING_PATH_ALREADY_EMITS) {
            out.push(WaveEvent::Formed(group_field_of(line)));
        } else if line
            .contains(SCHEDULER_GROUP_DRAINED_EVENT_PROVES_A_WAVE_FINISHED_BEFORE_THE_NEXT_ONE_FORMED)
        {
            out.push(WaveEvent::Drained(group_field_of(line)));
        }
    }
    out
}

fn formed_sizes(events: &[WaveEvent]) -> Vec<usize> {
    events
        .iter()
        .filter_map(|e| match e {
            WaveEvent::Formed(n) => Some(*n),
            WaveEvent::Drained(_) => None,
        })
        .collect()
}

fn expected_wave_shape_for(concurrency: usize, lanes: usize) -> Vec<usize> {
    let mut left = concurrency;
    let mut out = Vec::new();
    while left > 1 {
        let take = left.min(lanes);
        if take == 1 {
            break;
        }
        out.push(take);
        left -= take;
    }
    out
}

#[derive(Clone)]
struct RunOut {
    label: String,
    content: String,
    reasoning: String,
    finish_reason: String,
    completion_tokens: u64,
    prompt_tokens: u64,
    wall_s: f64,
    first_token_s: f64,
    error: Option<String>,
    sse_deltas_arrive_one_per_decode_step_so_they_stand_in_for_tokens: Vec<String>,
}

impl RunOut {
    fn stream(&self) -> String {
        format!("{}{}", self.reasoning, self.content)
    }

    fn ms_per_token_after_the_first(&self) -> f64 {
        if self.completion_tokens > 1 {
            (self.wall_s - self.first_token_s) * 1000.0 / (self.completion_tokens - 1) as f64
        } else {
            self.wall_s * 1000.0
        }
    }
}

fn chat_request(model_id: &str, prompt: &str) -> Request<Body> {
    let body = format!(
        r#"{{"model":"{}","max_tokens":{},"temperature":0,
             "stream":true,"stream_options":{{"include_usage":true}},
             "enable_thinking":false,
             "messages":[{{"role":"user","content":{}}}]}}"#,
        model_id,
        MAX_TOKENS_64_KEEPS_THE_WHOLE_LADDER_INSIDE_ONE_GPUQ_JOB_AND_STILL_OUTLASTS_PREFILL,
        serde_json::to_string(prompt).unwrap()
    );
    Request::builder()
        .method("POST")
        .uri("/v1/chat/completions")
        .header("content-type", "application/json")
        .body(Body::from(body))
        .unwrap()
}

async fn run_one(router: Router, label: String, model_id: String, prompt: String) -> RunOut {
    let t0 = Instant::now();
    let resp = router
        .oneshot(chat_request(&model_id, &prompt))
        .await
        .unwrap();
    assert_eq!(
        resp.status(),
        StatusCode::OK,
        "[{label}] the streaming chat request was rejected before any token was produced"
    );
    let mut data = resp.into_body().into_data_stream();
    let mut pending = String::new();
    let mut out = RunOut {
        label: label.clone(),
        content: String::new(),
        reasoning: String::new(),
        finish_reason: String::new(),
        completion_tokens: 0,
        prompt_tokens: 0,
        wall_s: 0.0,
        first_token_s: 0.0,
        error: None,
        sse_deltas_arrive_one_per_decode_step_so_they_stand_in_for_tokens: Vec::new(),
    };
    let mut first_token_s: Option<f64> = None;
    while let Some(chunk) = data.next().await {
        let chunk = chunk.unwrap_or_else(|e| panic!("[{label}] sse transport broke mid-stream: {e}"));
        pending.push_str(&String::from_utf8_lossy(&chunk));
        while let Some(at) = pending.find("\n\n") {
            let frame = pending[..at].to_string();
            pending = pending[at + 2..].to_string();
            let Some(payload) = frame.strip_prefix("data: ") else {
                continue;
            };
            if payload.trim() == "[DONE]" {
                continue;
            }
            let v: serde_json::Value = serde_json::from_str(payload)
                .unwrap_or_else(|e| panic!("[{label}] unparseable sse frame {payload:?}: {e}"));
            if let Some(msg) = v["error"]["message"].as_str() {
                out.error = Some(msg.to_string());
                continue;
            }
            if let Some(u) = v.get("usage").filter(|u| !u.is_null()) {
                out.prompt_tokens = u["prompt_tokens"].as_u64().unwrap_or(0);
                out.completion_tokens = u["completion_tokens"].as_u64().unwrap_or(0);
            }
            let delta = &v["choices"][0]["delta"];
            let c = delta["content"].as_str().unwrap_or_default();
            let r = delta["reasoning_content"].as_str().unwrap_or_default();
            if first_token_s.is_none() && !(c.is_empty() && r.is_empty()) {
                first_token_s = Some(t0.elapsed().as_secs_f64());
            }
            out.content.push_str(c);
            out.reasoning.push_str(r);
            if !(c.is_empty() && r.is_empty()) {
                out.sse_deltas_arrive_one_per_decode_step_so_they_stand_in_for_tokens
                    .push(format!("{r}{c}"));
            }
            if let Some(f) = v["choices"][0]["finish_reason"].as_str() {
                out.finish_reason = f.to_string();
            }
        }
    }
    out.wall_s = t0.elapsed().as_secs_f64();
    out.first_token_s = first_token_s.unwrap_or(out.wall_s);
    eprintln!(
        "[q38-conc] [{label}] prompt_tokens={} completion_tokens={} finish={:?} err={:?} \
         first_token={:.0}ms wall={:.2}s {:.1} ms/tok",
        out.prompt_tokens,
        out.completion_tokens,
        out.finish_reason,
        out.error,
        out.first_token_s * 1000.0,
        out.wall_s,
        out.ms_per_token_after_the_first()
    );
    out
}

fn rt() -> tokio::runtime::Runtime {
    tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap()
}

fn run_sequential(engine: &Arc<dyn ChatEngine>, label: &str, prompts: &[&str]) -> Vec<RunOut> {
    rt().block_on(async {
        let mut outs = Vec::new();
        for (i, p) in prompts.iter().enumerate() {
            outs.push(
                run_one(
                    app(engine.clone()),
                    format!("{label}-seq{i}"),
                    engine.model_id().to_string(),
                    (*p).to_string(),
                )
                .await,
            );
        }
        outs
    })
}

fn run_warmup(engine: &Arc<dyn ChatEngine>, label: &str) {
    rt().block_on(async {
        let _ = run_one(
            app(engine.clone()),
            format!("{label}-warmup"),
            engine.model_id().to_string(),
            SIXTEEN_DISTINCT_PROMPTS_SO_NO_TWO_LANES_SHARE_A_PREFIX_AND_A_CROSSED_LANE_SHOWS_AS_A_BYTE_DIFF[0]
                .to_string(),
        )
        .await;
    });
}

fn run_concurrent(engine: &Arc<dyn ChatEngine>, label: &str, prompts: &[&str]) -> (Vec<RunOut>, f64) {
    rt().block_on(async {
        let t = Instant::now();
        let futs = prompts.iter().enumerate().map(|(i, p)| {
            run_one(
                app(engine.clone()),
                format!("{label}-c{i}"),
                engine.model_id().to_string(),
                (*p).to_string(),
            )
        });
        let outs = futures_util::future::join_all(futs).await;
        (outs, t.elapsed().as_secs_f64())
    })
}

fn first_divergence(outs: &[RunOut], refs: &[RunOut]) -> Option<(usize, usize)> {
    for (i, (got, want)) in outs.iter().zip(refs.iter()).enumerate() {
        let (g, w) = (got.stream(), want.stream());
        if g != w {
            let at = g
                .chars()
                .zip(w.chars())
                .position(|(a, b)| a != b)
                .unwrap_or_else(|| g.chars().count().min(w.chars().count()));
            return Some((i, at));
        }
    }
    None
}

fn assert_same_stream(context: &str, got: &RunOut, want: &RunOut) {
    let (g, w) = (got.stream(), want.stream());
    if g != w {
        let at = g
            .chars()
            .zip(w.chars())
            .position(|(a, b)| a != b)
            .unwrap_or_else(|| g.chars().count().min(w.chars().count()));
        panic!(
            "[{}] {context} diverged from its solo sequential reference at char {at}. \
             {THE_GROUP_ARM_IS_BIT_IDENTICAL_TO_EAGER_M1_STEPPING_ONLY_UNDER_NV_Q38_BATCH_ROWWISE_FFN}\n \
             got: {g:?}\nwant: {w:?}",
            got.label
        );
    }
    assert_eq!(
        got.completion_tokens, want.completion_tokens,
        "[{}] {context}: same bytes but a different token count",
        got.label
    );
    assert_eq!(
        got.prompt_tokens, want.prompt_tokens,
        "[{}] {context}: same bytes but a different prompt token count",
        got.label
    );
}

fn assert_every_request_completed(level: usize, outs: &[RunOut]) {
    for o in outs {
        assert!(
            o.error.is_none(),
            "[{}] c={level}: the engine returned an error instead of a completion: {:?}",
            o.label,
            o.error
        );
        assert!(
            !o.finish_reason.is_empty(),
            "[{}] c={level}: the stream ended with no finish_reason, so the request neither \
             finished nor failed loudly",
            o.label
        );
        assert!(
            !o.content.trim().is_empty(),
            "[{}] c={level}: finish_reason={} but the content is empty",
            o.label,
            o.finish_reason
        );
        assert!(
            o.completion_tokens > 0,
            "[{}] c={level}: no tokens were reported in usage",
            o.label
        );
    }
}

fn percentile(mut xs: Vec<f64>, q: f64) -> f64 {
    if xs.is_empty() {
        return 0.0;
    }
    xs.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let rank = (q * xs.len() as f64).ceil().max(1.0) as usize;
    xs[rank.min(xs.len()) - 1]
}

fn emit_machine_line(tag: &str, level: usize, outs: &[RunOut], wall_s: f64, sizes: &[usize]) {
    let tokens: u64 = outs.iter().map(|o| o.completion_tokens).sum();
    let solo_dispatches = level - sizes.iter().sum::<usize>();
    let waves = sizes.len() + solo_dispatches;
    let ms_tok: Vec<f64> = outs.iter().map(|o| o.ms_per_token_after_the_first()).collect();
    let first_ms: Vec<f64> = outs.iter().map(|o| o.first_token_s * 1000.0).collect();
    let shape = if sizes.is_empty() {
        "solo".to_string()
    } else {
        sizes
            .iter()
            .map(|n| n.to_string())
            .collect::<Vec<_>>()
            .join("+")
    };
    println!(
        "CONC-E2E c={level} total_tok_s={:.1} p50_ms_tok={:.2} p95_first_token_ms={:.0} \
         waves={waves} group_shape={shape} tokens={tokens} wall_s={wall_s:.2} \
         p50_first_token_ms={:.0} min_ms_tok={:.2} max_ms_tok={:.2} tag={tag}",
        tokens as f64 / wall_s.max(1e-9),
        percentile(ms_tok.clone(), 0.5),
        percentile(
            first_ms.clone(),
            P95_NEAREST_RANK_BECAUSE_A_LEVEL_HAS_AT_MOST_SIXTEEN_SAMPLES_AND_INTERPOLATION_INVENTS_VALUES
        ),
        percentile(first_ms, 0.5),
        ms_tok.iter().cloned().fold(f64::INFINITY, f64::min),
        ms_tok.iter().cloned().fold(f64::NEG_INFINITY, f64::max),
    );
}

fn load_engine(dir: &std::path::Path, label: &str) -> anyhow::Result<Arc<dyn ChatEngine>> {
    let t = Instant::now();
    let engine = NvEngineChat::try_load(dir)?;
    eprintln!(
        "[q38-conc] [{label}] engine ready in {:.1}s",
        t.elapsed().as_secs_f64()
    );
    Ok(Arc::new(engine))
}

const THE_BYTE_IDENTITY_TESTS_CLEAR_NV_Q38_BATCH_GEMM_BECAUSE_THE_M_ROW_LADDER_IN_THIS_SAME_BINARY_SETS_IT_AND_ENV_IS_PROCESS_GLOBAL:
    &str = "every test in this file mutates process env, so a test that merely omits a variable \
            inherits whatever the previously run test left. The GEMM arm is tolerance-class by \
            construction, so a byte-identity test that inherits it fails for a reason that has \
            nothing to do with what it gates";

fn ladder_env_common(sizes: &str) {
    let _ = THE_BYTE_IDENTITY_TESTS_CLEAR_NV_Q38_BATCH_GEMM_BECAUSE_THE_M_ROW_LADDER_IN_THIS_SAME_BINARY_SETS_IT_AND_ENV_IS_PROCESS_GLOBAL;
    std::env::set_var("NV_QWEN35_DENSE_CUDA_SERVE", "1");
    std::env::set_var("NV_GDN_FUSED_DECODE", "1");
    std::env::remove_var("NV_DRAFTER");
    std::env::remove_var("NV_Q38_BATCH_GEMM");
    std::env::set_var("NV_Q38_BATCH_SIZES", sizes);
    std::env::set_var("NV_Q38_BATCH_ROWWISE", "ffn");
}

static THE_ONE_INSTALLED_SINK: std::sync::OnceLock<SchedulerEventSink> = std::sync::OnceLock::new();

fn install_scheduler_event_sink() -> SchedulerEventSink {
    THE_ONE_INSTALLED_SINK
        .get_or_init(|| {
            let sink = SchedulerEventSink::new();
            tracing_subscriber::fmt()
                .with_env_filter("speaches_plus=info")
                .with_ansi(false)
                .with_writer(sink.clone())
                .try_init()
                .expect(
                    "a process takes exactly one global tracing subscriber, so the second test in \
                     this binary must be handed the sink the first one installed; a fresh sink \
                     here is wired to nothing and every wave assertion would read an empty log",
                );
            sink
        })
        .clone()
}

fn gate_or_panic() {
    if std::env::var(REAL_WEIGHTS_TEST_ENV).as_deref() != Ok("1") {
        panic!(
            "this test was asked for BY NAME but {REAL_WEIGHTS_TEST_ENV}=1 is not set. This is a \
             SKIP, not a pass."
        );
    }
}

#[test]
#[ignore = "loads the Qwen3.8-27B NVFP4 checkpoint twice (flag-off eager reference, then \
            NV_Q38_BATCH=1 with the default 1,2,4 buckets) and drives an HTTP concurrency \
            ladder at c=1,2,4,8; set NV_Q38_CONC_LADDER_TEST=1"]
fn http_user_request_concurrency_ladder_1_2_4_8_stays_byte_identical_to_the_solo_references() {
    gate_or_panic();
    let sink = install_scheduler_event_sink();
    eprintln!(
        "[q38-conc] {MTP_IS_OFF_FOR_THE_WHOLE_LADDER_SO_ONE_EAGER_REFERENCE_SERVES_THE_SOLO_AND_THE_GROUP_ARM}"
    );
    eprintln!("[q38-conc] {A_TRACING_SINK_READS_THE_WAVE_COUNT_OFF_THE_SHIPPED_SCHEDULER_EVENTS_SO_NO_SRC_ACCESSOR_IS_ADDED}");
    let dir = qwen38_nvfp4_snapshot_dir_env_override_then_home_hub();

    ladder_env_common(FOUR_LANE_BUCKET_PLAN_IS_THE_SHIPPED_DEFAULT_NV_Q38_BATCH_SIZES);
    std::env::remove_var("NV_Q38_BATCH");
    let eager = load_engine(&dir, "eager-reference").expect("flag-off eager engine must load");
    run_warmup(&eager, "eager-reference");
    let refs = run_sequential(
        &eager,
        "eager-reference",
        &SIXTEEN_DISTINCT_PROMPTS_SO_NO_TWO_LANES_SHARE_A_PREFIX_AND_A_CROSSED_LANE_SHOWS_AS_A_BYTE_DIFF,
    );
    assert_every_request_completed(1, &refs);
    drop(eager);

    std::env::set_var("NV_Q38_BATCH", "1");
    let batch = load_engine(&dir, "batch").expect("NV_Q38_BATCH=1 engine must load");
    run_warmup(&batch, "batch");
    let lanes = FOUR_LANE_BUCKET_PLAN_IS_THE_SHIPPED_DEFAULT_NV_Q38_BATCH_SIZES
        .split(',')
        .filter_map(|s| s.trim().parse::<usize>().ok())
        .max()
        .unwrap();

    let mut eight_wide_events: Vec<WaveEvent> = Vec::new();
    for level in CONCURRENCY_LADDER_SPANS_SOLO_PARTIAL_GROUP_FULL_GROUP_AND_TWO_WAVES {
        let mark = sink.mark();
        let (outs, wall_s) = run_concurrent(
            &batch,
            &format!("batch-c{level}"),
            &SIXTEEN_DISTINCT_PROMPTS_SO_NO_TWO_LANES_SHARE_A_PREFIX_AND_A_CROSSED_LANE_SHOWS_AS_A_BYTE_DIFF
                [..level],
        );
        let events = sink.waves_since(mark);
        assert_every_request_completed(level, &outs);
        assert!(
            wall_s < WALL_BOUND_S_FOR_A_WHOLE_LEVEL_240_IS_TWENTY_TIMES_THE_EXPECTED_TWO_WAVE_DECODE_SO_ONLY_A_STARVED_REQUEST_TRIPS_IT,
            "c={level} took {wall_s:.1}s: a request was starved or the dispatcher stalled"
        );
        for (i, (got, want)) in outs.iter().zip(refs.iter()).enumerate() {
            assert_same_stream(&format!("c={level} request {i}"), got, want);
        }
        let sizes = formed_sizes(&events);
        assert_eq!(
            sizes,
            expected_wave_shape_for(level, lanes),
            "c={level} on {lanes} lanes formed groups {sizes:?}; the formation window must \
             collect every request already queued and a leftover single request must run solo, \
             so the wave shape is fixed by the policy, not by luck. An empty list when the \
             requests were served means the sink is not the installed subscriber, not that the \
             scheduler skipped the group. events={events:?}"
        );
        emit_machine_line("sizes=1,2,4", level, &outs, wall_s, &sizes);
        if level == 8 {
            eight_wide_events = events;
        }
    }

    assert_eq!(
        eight_wide_events,
        vec![
            WaveEvent::Formed(4),
            WaveEvent::Drained(4),
            WaveEvent::Formed(4),
            WaveEvent::Drained(4)
        ],
        "8 requests over 4 lanes must drain as two closed waves: a group is closed at formation \
         and a late joiner waits for the whole group to drain, so the only legal event order is \
         form, drain, form, drain"
    );
}

#[test]
#[ignore = "loads the Qwen3.8-27B NVFP4 checkpoint once per rowwise pin with \
            NV_Q38_BATCH_SIZES=1,2,4,8 and asks for a true eight-wide group; either some pin \
            holds byte identity or the refusal must name its constraint; set \
            NV_Q38_CONC_LADDER_TEST=1"]
fn an_eight_wide_group_under_nv_q38_batch_sizes_1_2_4_8_holds_byte_identity_or_refuses_loudly() {
    gate_or_panic();
    let sink = install_scheduler_event_sink();
    eprintln!("[q38-conc] {ONE_ENGINE_PER_ROWWISE_PIN_BECAUSE_THE_CAPTURED_GRAPH_KEY_IS_THE_BUCKET_ALONE}");
    let dir = qwen38_nvfp4_snapshot_dir_env_override_then_home_hub();
    let mut refs: Vec<RunOut> = Vec::new();
    let mut byte_exact_pins: Vec<&str> = Vec::new();
    let mut diverged_pins: Vec<String> = Vec::new();

    for pin in ROWWISE_PINS_TRIED_FOR_AN_EIGHT_WIDE_GROUP_NARROWEST_FIRST {
        ladder_env_common(EIGHT_LANE_BUCKET_PLAN_ASKS_THE_ENGINE_FOR_A_TRUE_EIGHT_WIDE_GROUP_AND_TWO_WAVES_AT_SIXTEEN);
        std::env::set_var("NV_Q38_BATCH_ROWWISE", pin);
        std::env::set_var("NV_Q38_BATCH", "1");
        let label = format!("batch-8lane-{pin}");
        let batch = match load_engine(&dir, &label) {
            Ok(e) => e,
            Err(err) => {
                let msg = format!("{err:#}");
                assert!(
                    A_LOUD_EIGHT_WIDE_REFUSAL_MUST_NAME_ONE_OF_THESE_CONSTRAINTS_OR_IT_IS_A_SILENT_FALLBACK
                        .iter()
                        .any(|k| msg.to_lowercase().contains(&k.to_lowercase())),
                    "the 8-lane pool refused to boot but the error names no constraint, so a \
                     reader cannot tell what to change: {msg}"
                );
                println!(
                    "CONC-E2E-B8 outcome=refused_at_boot rowwise={pin} loud=yes constraint={:?}",
                    msg.replace('\n', " ")
                );
                return;
            }
        };
        run_warmup(&batch, &label);
        if refs.is_empty() {
            refs = run_sequential(
                &batch,
                "batch-8lane-solo",
                &SIXTEEN_DISTINCT_PROMPTS_SO_NO_TWO_LANES_SHARE_A_PREFIX_AND_A_CROSSED_LANE_SHOWS_AS_A_BYTE_DIFF,
            );
            assert_every_request_completed(1, &refs);
        }

        let mark = sink.mark();
        let (outs, wall_s) = run_concurrent(
            &batch,
            &label,
            &SIXTEEN_DISTINCT_PROMPTS_SO_NO_TWO_LANES_SHARE_A_PREFIX_AND_A_CROSSED_LANE_SHOWS_AS_A_BYTE_DIFF,
        );
        let events = sink.waves_since(mark);
        let sizes = formed_sizes(&events);
        assert!(
            wall_s < WALL_BOUND_S_FOR_A_WHOLE_LEVEL_240_IS_TWENTY_TIMES_THE_EXPECTED_TWO_WAVE_DECODE_SO_ONLY_A_STARVED_REQUEST_TRIPS_IT,
            "the 8-wide attempt under rowwise={pin} took {wall_s:.1}s: a refusal must be \
             immediate and a success must not be slower than two 4-wide waves, so this is a hang \
             either way"
        );

        let failures: Vec<&RunOut> = outs.iter().filter(|o| o.error.is_some()).collect();
        if !failures.is_empty() {
            assert_eq!(
                failures.len(),
                outs.len(),
                "the 8-wide group failed for {} of {} requests under rowwise={pin}: a partial \
                 failure is the silent fallback this gate exists to catch, because the survivors \
                 were served at some other batch size while the caller was told nothing",
                failures.len(),
                outs.len()
            );
            for o in &failures {
                let msg = o.error.clone().unwrap_or_default();
                assert!(
                    A_LOUD_EIGHT_WIDE_REFUSAL_MUST_NAME_ONE_OF_THESE_CONSTRAINTS_OR_IT_IS_A_SILENT_FALLBACK
                        .iter()
                        .any(|k| msg.to_lowercase().contains(&k.to_lowercase())),
                    "[{}] the 8-wide group failed with an error that names no constraint: {msg}",
                    o.label
                );
            }
            println!(
                "CONC-E2E-B8 outcome=refused_at_step rowwise={pin} loud=yes formed={sizes:?} \
                 wall_s={wall_s:.2} constraint={:?}",
                failures[0].error.clone().unwrap_or_default().replace('\n', " ")
            );
            return;
        }

        assert_every_request_completed(8, &outs);
        assert_eq!(
            sizes,
            vec![8],
            "every request returned a completion under rowwise={pin}, so the engine claims it \
             served an 8-wide group; the scheduler must then have formed exactly one group of 8. \
             Two groups of 4 here would be a silent fallback lying about the batch size. \
             events={events:?}"
        );
        emit_machine_line(&format!("sizes=1,2,4,8 rowwise={pin}"), 8, &outs, wall_s, &sizes);
        match first_divergence(&outs, &refs) {
            None => byte_exact_pins.push(pin),
            Some((req, at)) => diverged_pins.push(format!("{pin}(request {req} char {at})")),
        }
        drop(batch);
        if !byte_exact_pins.is_empty() {
            break;
        }
    }

    println!(
        "CONC-E2E-B8 outcome=eight_wide_group_served loud=n/a byte_identical_under={:?} \
         diverged_under={:?}",
        byte_exact_pins, diverged_pins
    );
    assert!(
        !byte_exact_pins.is_empty(),
        "an 8-wide group was formed and served under every pin tried ({:?}), and none of them \
         reproduced the solo bytes. Qwen38BatchLanes::new admits buckets up to the m<=8 row-twin \
         kernel ceiling, so at m=8 some kernel outside the rowwise groups has stopped being a \
         per-row m=1 twin; the engine neither refused nor warned, so an 8-lane deployment would \
         quietly serve different text than the same request served alone. diverged: {diverged_pins:?}",
        ROWWISE_PINS_TRIED_FOR_AN_EIGHT_WIDE_GROUP_NARROWEST_FIRST
    );
}

const THE_SERVING_BAR_IS_GREEDY_ARGMAX_CLASS_NOT_BYTE_IDENTITY: &str =
    "byte-identity-to-solo is a debug bar for the rowwise pins; the serving bar for the default \
     m-row route is the greedy argmax drift CLASS vs the solo eager reference, reported as a \
     count of differing decode steps, never gated on zero, because the nvfp4 padded-TC non-twin \
     is a deterministic 1-2e-2 logit delta that is argmax-equal at small m and may drift at \
     larger m";

fn ladder_env_default_m_row_route(sizes: &str) {
    ladder_env_common(sizes);
    std::env::remove_var("NV_Q38_BATCH_ROWWISE");
    std::env::set_var("NV_Q38_BATCH_GEMM", "1");
}

fn gdn_prefill_scan_arm_because_the_group_formation_wall_is_the_gdn_prefill_scan_not_the_gemms() -> &'static str {
    if std::env::var("NV_Q38_GDN_CHUNK_PREFILL").ok().as_deref() == Some("1") {
        "gdn_chunk"
    } else {
        "gdn_candle_token_sequential"
    }
}

struct DriftStats {
    differing_steps: usize,
    total_steps: usize,
    requests_differing: usize,
    per_request: Vec<String>,
}

fn greedy_argmax_drift_vs_solo(outs: &[RunOut], refs: &[RunOut]) -> DriftStats {
    let mut stats = DriftStats {
        differing_steps: 0,
        total_steps: 0,
        requests_differing: 0,
        per_request: Vec::new(),
    };
    for (i, (got, want)) in outs.iter().zip(refs.iter()).enumerate() {
        let g = &got.sse_deltas_arrive_one_per_decode_step_so_they_stand_in_for_tokens;
        let w = &want.sse_deltas_arrive_one_per_decode_step_so_they_stand_in_for_tokens;
        let n = g.len().max(w.len());
        let d = (0..n).filter(|&j| g.get(j) != w.get(j)).count();
        let first = (0..n).find(|&j| g.get(j) != w.get(j));
        stats.total_steps += n;
        stats.differing_steps += d;
        if d > 0 {
            stats.requests_differing += 1;
        }
        stats
            .per_request
            .push(format!("req{i}:{d}/{n}@{first:?}"));
    }
    stats
}

fn emit_ours_conc_line(
    level: usize,
    sizes_env: &str,
    outs: &[RunOut],
    wall_s: f64,
    formed: &[usize],
    drift: &DriftStats,
) {
    let tokens: u64 = outs.iter().map(|o| o.completion_tokens).sum();
    let ms_tok: Vec<f64> = outs.iter().map(|o| o.ms_per_token_after_the_first()).collect();
    let first_ms: Vec<f64> = outs.iter().map(|o| o.first_token_s * 1000.0).collect();
    let shape = if formed.is_empty() {
        "solo".to_string()
    } else {
        formed
            .iter()
            .map(|n| n.to_string())
            .collect::<Vec<_>>()
            .join("+")
    };
    let class = if drift.differing_steps == 0 {
        "argmax_equal_to_solo"
    } else {
        "argmax_drift"
    };
    let steady_agg_tok_s: f64 = outs
        .iter()
        .map(|o| {
            let dt = o.wall_s - o.first_token_s;
            if dt > 0.0 && o.completion_tokens > 1 {
                (o.completion_tokens - 1) as f64 / dt
            } else {
                0.0
            }
        })
        .sum();
    println!(
        "OURS-CONC c={level} agg_tok_s={:.1} \
         agg_def=output_tokens_over_batch_wall_incl_ttft_same_as_vllm_bench_serving \
         steady_agg_tok_s={steady_agg_tok_s:.1} p50_tpot_ms={:.2} p50_ttft_ms={:.0} \
         p95_ttft_ms={:.0} tokens={tokens} wall_s={wall_s:.2} group_shape={shape} \
         sizes={sizes_env} rowwise=unset prefill_scan={} drift_steps={}/{} drift_requests={}/{} \
         drift_class={class} drift_detail={}",
        tokens as f64 / wall_s.max(1e-9),
        percentile(ms_tok, 0.5),
        percentile(first_ms.clone(), 0.5),
        percentile(
            first_ms,
            P95_NEAREST_RANK_BECAUSE_A_LEVEL_HAS_AT_MOST_SIXTEEN_SAMPLES_AND_INTERPOLATION_INVENTS_VALUES
        ),
        gdn_prefill_scan_arm_because_the_group_formation_wall_is_the_gdn_prefill_scan_not_the_gemms(),
        drift.differing_steps,
        drift.total_steps,
        drift.requests_differing,
        level,
        drift.per_request.join(","),
    );
}

#[test]
#[ignore = "loads the Qwen3.8-27B NVFP4 checkpoint (flag-off eager solo reference, then the \
            DEFAULT m-row batch route with NV_Q38_BATCH_ROWWISE unset and NV_Q38_BATCH_GEMM=1) \
            and drives c=1,2,4,8 on sizes=1,2,4,8 plus c=16 on sizes=1,2,4,8,16 as ONE 16-wide \
            group, reporting aggregate tok/s, steady aggregate, TPOT, TTFT p50/p95 and the \
            greedy argmax drift class vs solo without gating on byte identity; set \
            NV_Q38_CONC_LADDER_TEST=1"]
fn default_m_row_route_ladder_reports_throughput_and_argmax_drift_class_vs_solo() {
    gate_or_panic();
    let sink = install_scheduler_event_sink();
    eprintln!("[q38-conc] {THE_SERVING_BAR_IS_GREEDY_ARGMAX_CLASS_NOT_BYTE_IDENTITY}");
    eprintln!("[q38-conc] {A_SIXTEEN_WIDE_GROUP_IS_THE_ONLY_HONEST_C16_ROW_BECAUSE_TWO_EIGHT_WIDE_WAVES_PAY_FORMATION_TWICE_AND_DECODE_AT_HALF_THE_BATCH}");
    eprintln!("[q38-conc] {NV_Q38_BATCH_GEMM_1_IS_REQUIRED_BY_THE_SIXTEEN_LANE_POOL_AND_HELD_FOR_EVERY_LEVEL_SO_ONE_ARM_SPANS_THE_LADDER}");
    eprintln!(
        "[q38-conc] prefill_scan={}",
        gdn_prefill_scan_arm_because_the_group_formation_wall_is_the_gdn_prefill_scan_not_the_gemms()
    );
    let dir = qwen38_nvfp4_snapshot_dir_env_override_then_home_hub();

    ladder_env_default_m_row_route(EIGHT_LANE_BUCKET_PLAN_ASKS_THE_ENGINE_FOR_A_TRUE_EIGHT_WIDE_GROUP_AND_TWO_WAVES_AT_SIXTEEN);
    std::env::remove_var("NV_Q38_BATCH");
    let eager = load_engine(&dir, "mrow-eager-reference").expect("flag-off eager engine must load");
    run_warmup(&eager, "mrow-eager-reference");
    let refs = run_sequential(
        &eager,
        "mrow-eager-reference",
        &SIXTEEN_DISTINCT_PROMPTS_SO_NO_TWO_LANES_SHARE_A_PREFIX_AND_A_CROSSED_LANE_SHOWS_AS_A_BYTE_DIFF,
    );
    assert_every_request_completed(1, &refs);
    drop(eager);

    let plans: [(&str, &[usize]); 2] = [
        (
            EIGHT_LANE_BUCKET_PLAN_ASKS_THE_ENGINE_FOR_A_TRUE_EIGHT_WIDE_GROUP_AND_TWO_WAVES_AT_SIXTEEN,
            &[1, 2, 4, 8],
        ),
        (
            SIXTEEN_LANE_BUCKET_PLAN_MAKES_C16_ONE_GROUP_INSTEAD_OF_TWO_WAVES_AND_NEEDS_NV_Q38_BATCH_GEMM_1,
            &[16],
        ),
    ];
    for (sizes_env, levels) in plans {
        ladder_env_default_m_row_route(sizes_env);
        std::env::set_var("NV_Q38_BATCH", "1");
        let label = format!("mrow-batch-{sizes_env}");
        let batch = load_engine(&dir, &label).expect("NV_Q38_BATCH=1 engine must load");
        run_warmup(&batch, &label);
        for &level in levels {
            let mark = sink.mark();
            let (outs, wall_s) = run_concurrent(
                &batch,
                &format!("{label}-c{level}"),
                &SIXTEEN_DISTINCT_PROMPTS_SO_NO_TWO_LANES_SHARE_A_PREFIX_AND_A_CROSSED_LANE_SHOWS_AS_A_BYTE_DIFF
                    [..level],
            );
            let events = sink.waves_since(mark);
            assert_every_request_completed(level, &outs);
            assert!(
                wall_s < WALL_BOUND_S_FOR_A_WHOLE_LEVEL_240_IS_TWENTY_TIMES_THE_EXPECTED_TWO_WAVE_DECODE_SO_ONLY_A_STARVED_REQUEST_TRIPS_IT,
                "c={level} took {wall_s:.1}s: a request was starved or the dispatcher stalled"
            );
            let formed = formed_sizes(&events);
            if level > 1 {
                assert_eq!(
                    formed,
                    vec![level],
                    "c={level} on sizes={sizes_env} formed {formed:?}. \
                     {A_SIXTEEN_WIDE_GROUP_IS_THE_ONLY_HONEST_C16_ROW_BECAUSE_TWO_EIGHT_WIDE_WAVES_PAY_FORMATION_TWICE_AND_DECODE_AT_HALF_THE_BATCH} \
                     events={events:?}"
                );
            }
            let drift = greedy_argmax_drift_vs_solo(&outs, &refs);
            eprintln!(
                "[q38-conc] [mrow c={level}] drift per request: {}",
                drift.per_request.join(" ")
            );
            emit_ours_conc_line(level, sizes_env, &outs, wall_s, &formed, &drift);
        }
        drop(batch);
    }
}
