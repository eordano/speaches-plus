#![cfg(feature = "cuda")]

use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::Instant;

use axum::body::Body;
use axum::http::{Request, StatusCode};
use axum::routing::post;
use axum::Router;
use futures_util::StreamExt;
use tower::ServiceExt;

use speaches_plus::oapi::chat::{handle_chat_completions, ChatAppState, ChatEngine};
use speaches_plus::oapi::chat_engine::{ChatRegistry, NvEngineChat};

const REAL_WEIGHTS_TEST_ENV: &str = "NV_Q36_CONC_LADDER_TEST";

const FOUR_DISTINCT_PROMPTS_SO_NO_TWO_LANES_SHARE_A_PREFIX_AND_A_CROSSED_LANE_SHOWS_AS_A_BYTE_DIFF:
    [&str; 4] = [
    "Write one paragraph of about 150 words describing a lighthouse keeper's morning routine.",
    "Write one paragraph of about 150 words explaining why the sky appears blue during the day.",
    "Write one paragraph of about 150 words describing how a river shapes a valley over time.",
    "Write one paragraph of about 150 words explaining how honeybees communicate about food.",
];

const MAX_TOKENS_64_KEEPS_THE_BYTE_GATE_INSIDE_ONE_GPUQ_JOB_AND_STILL_OUTLASTS_PREFILL: u32 = 64;

const MAX_TOKENS_256_MATCHES_THE_ENGINE_RATE_LADDERS_DEPTH_SO_THE_HTTP_ROWS_COMPARE_TO_ITS_225_AND_384_TOK_S:
    u32 = 256;

const CONCURRENCY_LADDER_SPANS_SOLO_PARTIAL_GROUP_AND_FULL_GROUP: [usize; 3] = [1, 2, 4];

const SHIPPED_DEFAULT_BUCKET_PLAN_NV_Q36_BATCH_SIZES: &str = "1,2,4";

const THE_BYTE_GATE_RUNS_ROWWISE_MOE_ON_THE_TWIN_BASIS_BECAUSE_THAT_IS_THE_ENGINE_SUITES_BIT_IDENTITY_CONTRACT:
    &str = "the engine gate (nv-models/tests/qwen36_batch_decode.rs rowwise_moe arm) proves every \
            lane bit-identical to its solo hook-decode twin exactly under NV_Q36_BATCH_ROWWISE=moe \
            plus the minimal twin basis NV_Q36_GDN_FP8=1 + NV_Q36_LM_HEAD_FP8=1; this ladder \
            holds the same pin and basis, and its solo reference is the flag-off eager engine, \
            whose prefill and m=1 decode both route through the same installed grouped hook the \
            lane pool prefills and rowwise-decodes with";

const THE_DEFAULT_GROUPED_ROUTE_IS_ARGMAX_CLASS_NOT_BYTE_GATED: &str =
    "rowwise unset serves rows=B grouped MoE, the sole non-twin route at B<=4; the engine gate \
     measures it argmax-class (REC exact 0.839, near-tie 0.161 on junk prompts; templated text \
     runs far higher), so the throughput ladder reports the greedy argmax drift class vs the \
     solo eager reference and never gates it on zero";

const SCHEDULER_GROUP_FORMED_EVENT_IS_THE_PER_WAVE_MARKER_THE_SERVING_PATH_ALREADY_EMITS: &str =
    "qwen3.6 batch group formed";

const SCHEDULER_GROUP_DRAINED_EVENT_PROVES_A_WAVE_FINISHED_BEFORE_THE_NEXT_ONE_FORMED: &str =
    "qwen3.6 batch group drained";

const BOOT_PLAN_LANE_ITEM_MARKER_PROVES_THE_PLAN_ACCOUNTS_THE_PREPAID_LANE_POOL: &str = "kv-pools@";

const BOOT_PLAN_LANE_ITEM_ENGINE_NAME_PROVES_THE_MOE_ARM_NOT_THE_DENSE_ARM_WAS_PLANNED: &str =
    "Qwen36BatchLanes";

const WALL_BOUND_S_FOR_A_WHOLE_LEVEL_240_SO_ONLY_A_STARVED_REQUEST_TRIPS_IT: f64 = 240.0;

const PLAN_BUDGET_92_GIB_BECAUSE_RECORDED_PRIOR_ENGINE_PLANS_OUTLIVE_THE_DROPPED_REFERENCE_SO_TWO_SEQUENTIAL_BOOTS_SUM_PAST_THE_DEFAULT_WHILE_THE_CARD_ONLY_EVER_HOLDS_ONE:
    &str = "92";

fn app(engine: Arc<dyn ChatEngine>) -> Router {
    Router::new()
        .route("/v1/chat/completions", post(handle_chat_completions))
        .with_state(ChatAppState {
            registry: ChatRegistry::single(engine),
        })
}

fn qwen36_nvfp4_snapshot_dir_env_override_then_home_hub() -> PathBuf {
    if let Ok(d) = std::env::var("NV_QWEN36_DIR") {
        return PathBuf::from(d);
    }
    let home = std::env::var_os("HOME").expect("HOME");
    let snaps = PathBuf::from(home)
        .join(".cache/huggingface/hub")
        .join("models--RedHatAI--Qwen3.6-35B-A3B-NVFP4")
        .join("snapshots");
    let mut cand: Vec<PathBuf> = std::fs::read_dir(&snaps)
        .unwrap_or_else(|e| panic!("no qwen3.6 hub snapshots at {}: {e}", snaps.display()))
        .flatten()
        .map(|e| e.path())
        .filter(|p| p.join("config.json").exists())
        .collect();
    cand.sort();
    cand.pop().expect("an extracted qwen3.6 snapshot dir")
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

    fn text_since(&self, mark: usize) -> String {
        let buf = self.0.lock().unwrap_or_else(|e| e.into_inner());
        buf[mark.min(buf.len())..].to_string()
    }

    fn waves_since(&self, mark: usize) -> Vec<WaveEvent> {
        wave_events_in(&self.text_since(mark))
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
    digits
        .parse()
        .unwrap_or_else(|e| panic!("scheduler wave line group= is not a number ({e}): {line:?}"))
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

fn chat_request(model_id: &str, prompt: &str, max_tokens: u32) -> Request<Body> {
    let body = format!(
        r#"{{"model":"{}","max_tokens":{},"temperature":0,
             "stream":true,"stream_options":{{"include_usage":true}},
             "enable_thinking":false,
             "messages":[{{"role":"user","content":{}}}]}}"#,
        model_id,
        max_tokens,
        serde_json::to_string(prompt).unwrap()
    );
    Request::builder()
        .method("POST")
        .uri("/v1/chat/completions")
        .header("content-type", "application/json")
        .body(Body::from(body))
        .unwrap()
}

async fn run_one(
    router: Router,
    label: String,
    model_id: String,
    prompt: String,
    max_tokens: u32,
) -> RunOut {
    let t0 = Instant::now();
    let resp = router
        .oneshot(chat_request(&model_id, &prompt, max_tokens))
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
        let chunk =
            chunk.unwrap_or_else(|e| panic!("[{label}] sse transport broke mid-stream: {e}"));
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
        "[q36-conc] [{label}] prompt_tokens={} completion_tokens={} finish={:?} err={:?} \
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

fn run_sequential(
    engine: &Arc<dyn ChatEngine>,
    label: &str,
    prompts: &[&str],
    max_tokens: u32,
) -> Vec<RunOut> {
    rt().block_on(async {
        let mut outs = Vec::new();
        for (i, p) in prompts.iter().enumerate() {
            outs.push(
                run_one(
                    app(engine.clone()),
                    format!("{label}-seq{i}"),
                    engine.model_id().to_string(),
                    (*p).to_string(),
                    max_tokens,
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
            FOUR_DISTINCT_PROMPTS_SO_NO_TWO_LANES_SHARE_A_PREFIX_AND_A_CROSSED_LANE_SHOWS_AS_A_BYTE_DIFF
                [0]
            .to_string(),
            MAX_TOKENS_64_KEEPS_THE_BYTE_GATE_INSIDE_ONE_GPUQ_JOB_AND_STILL_OUTLASTS_PREFILL,
        )
        .await;
    });
}

fn run_concurrent(
    engine: &Arc<dyn ChatEngine>,
    label: &str,
    prompts: &[&str],
    max_tokens: u32,
) -> (Vec<RunOut>, f64) {
    rt().block_on(async {
        let t = Instant::now();
        let futs = prompts.iter().enumerate().map(|(i, p)| {
            run_one(
                app(engine.clone()),
                format!("{label}-c{i}"),
                engine.model_id().to_string(),
                (*p).to_string(),
                max_tokens,
            )
        });
        let outs = futures_util::future::join_all(futs).await;
        (outs, t.elapsed().as_secs_f64())
    })
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
             {THE_BYTE_GATE_RUNS_ROWWISE_MOE_ON_THE_TWIN_BASIS_BECAUSE_THAT_IS_THE_ENGINE_SUITES_BIT_IDENTITY_CONTRACT}\n \
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

fn expected_wave_shape_for(concurrency: usize) -> Vec<usize> {
    if concurrency > 1 {
        vec![concurrency]
    } else {
        Vec::new()
    }
}

fn load_engine(dir: &std::path::Path, label: &str) -> anyhow::Result<Arc<dyn ChatEngine>> {
    let t = Instant::now();
    let engine = NvEngineChat::try_load(dir)?;
    eprintln!(
        "[q36-conc] [{label}] engine ready in {:.1}s",
        t.elapsed().as_secs_f64()
    );
    Ok(Arc::new(engine))
}

fn ladder_env_common(sizes: &str) {
    std::env::remove_var("NV_DRAFTER");
    std::env::remove_var("NV_QWEN_GRAPH");
    std::env::remove_var("NV_QWEN35_DENSE_CUDA_SERVE");
    std::env::set_var("NV_GDN_FUSED_DECODE", "1");
    std::env::set_var("NV_Q36_GDN_FP8", "1");
    std::env::set_var("NV_Q36_LM_HEAD_FP8", "1");
    std::env::set_var("NV_Q36_BATCH_SIZES", sizes);
    std::env::set_var(
        "NV_VRAM_BUDGET_GIB",
        PLAN_BUDGET_92_GIB_BECAUSE_RECORDED_PRIOR_ENGINE_PLANS_OUTLIVE_THE_DROPPED_REFERENCE_SO_TWO_SEQUENTIAL_BOOTS_SUM_PAST_THE_DEFAULT_WHILE_THE_CARD_ONLY_EVER_HOLDS_ONE,
    );
}

fn record_env_stack_matching_the_engine_rate_ladder_basis() {
    for arm in [
        "NV_Q36_GRAPHED_DECODE_FIX",
        "NV_Q36_GDN_FP8",
        "NV_Q36_LM_HEAD_FP8",
        "NV_Q38_GDN_SPLIT",
        "NV_Q36_MOE_FUSED_CHAIN",
        "NV_MOE_ROUTE_FUSE",
        "NV_Q36_MOE_SHARED_FOLD",
        "NV_MOE_FP4_DECODE_GEMV",
    ] {
        std::env::set_var(arm, "1");
    }
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
#[ignore = "loads the Qwen3.6-35B NVFP4 checkpoint twice (flag-off eager reference, then \
            NV_Q36_BATCH=1 with the default 1,2,4 buckets under NV_Q36_BATCH_ROWWISE=moe and the \
            twin basis) and drives an HTTP concurrency ladder at c=1,2,4; set \
            NV_Q36_CONC_LADDER_TEST=1"]
fn http_user_request_concurrency_ladder_1_2_4_stays_byte_identical_to_the_solo_references() {
    gate_or_panic();
    let sink = install_scheduler_event_sink();
    eprintln!(
        "[q36-conc] {THE_BYTE_GATE_RUNS_ROWWISE_MOE_ON_THE_TWIN_BASIS_BECAUSE_THAT_IS_THE_ENGINE_SUITES_BIT_IDENTITY_CONTRACT}"
    );
    let dir = qwen36_nvfp4_snapshot_dir_env_override_then_home_hub();

    ladder_env_common(SHIPPED_DEFAULT_BUCKET_PLAN_NV_Q36_BATCH_SIZES);
    std::env::set_var("NV_Q36_BATCH_ROWWISE", "moe");
    std::env::remove_var("NV_Q36_BATCH");
    let eager = load_engine(&dir, "eager-reference").expect("flag-off eager engine must load");
    run_warmup(&eager, "eager-reference");
    let refs = run_sequential(
        &eager,
        "eager-reference",
        &FOUR_DISTINCT_PROMPTS_SO_NO_TWO_LANES_SHARE_A_PREFIX_AND_A_CROSSED_LANE_SHOWS_AS_A_BYTE_DIFF,
        MAX_TOKENS_64_KEEPS_THE_BYTE_GATE_INSIDE_ONE_GPUQ_JOB_AND_STILL_OUTLASTS_PREFILL,
    );
    assert_every_request_completed(1, &refs);
    drop(eager);

    std::env::set_var("NV_Q36_BATCH", "1");
    let boot_mark = sink.mark();
    let batch = load_engine(&dir, "batch").expect("NV_Q36_BATCH=1 engine must load");
    let boot_log = sink.text_since(boot_mark);
    assert!(
        boot_log.contains(BOOT_PLAN_LANE_ITEM_MARKER_PROVES_THE_PLAN_ACCOUNTS_THE_PREPAID_LANE_POOL)
            && boot_log
                .contains(BOOT_PLAN_LANE_ITEM_ENGINE_NAME_PROVES_THE_MOE_ARM_NOT_THE_DENSE_ARM_WAS_PLANNED),
        "the NV_Q36_BATCH=1 boot plan carried no prepaid lane-pool item; the vram plan must \
         account the lanes before the engine allocates them"
    );
    run_warmup(&batch, "batch");

    for level in CONCURRENCY_LADDER_SPANS_SOLO_PARTIAL_GROUP_AND_FULL_GROUP {
        let mark = sink.mark();
        let (outs, wall_s) = run_concurrent(
            &batch,
            &format!("batch-c{level}"),
            &FOUR_DISTINCT_PROMPTS_SO_NO_TWO_LANES_SHARE_A_PREFIX_AND_A_CROSSED_LANE_SHOWS_AS_A_BYTE_DIFF
                [..level],
            MAX_TOKENS_64_KEEPS_THE_BYTE_GATE_INSIDE_ONE_GPUQ_JOB_AND_STILL_OUTLASTS_PREFILL,
        );
        let events = sink.waves_since(mark);
        assert_every_request_completed(level, &outs);
        assert!(
            wall_s < WALL_BOUND_S_FOR_A_WHOLE_LEVEL_240_SO_ONLY_A_STARVED_REQUEST_TRIPS_IT,
            "c={level} took {wall_s:.1}s: a request was starved or the dispatcher stalled"
        );
        for (i, (got, want)) in outs.iter().zip(refs.iter()).enumerate() {
            assert_same_stream(&format!("c={level} request {i}"), got, want);
        }
        let sizes = formed_sizes(&events);
        assert_eq!(
            sizes,
            expected_wave_shape_for(level),
            "c={level} must form exactly one group of every queued request (c=1 runs solo, so \
             no group at all); the formation window collects the whole level because every \
             request is already queued when the first prefill finishes. An empty list when \
             requests were served means the sink is not the installed subscriber. \
             events={events:?}"
        );
        let tokens: u64 = outs.iter().map(|o| o.completion_tokens).sum();
        println!(
            "Q36-CONC-BYTE c={level} tokens={tokens} wall_s={wall_s:.2} group_shape={sizes:?} \
             byte_identical=yes rowwise=moe twin_basis=gdn_fp8+lm_head_fp8"
        );
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
        stats.per_request.push(format!("req{i}:{d}/{n}@{first:?}"));
    }
    stats
}

#[test]
#[ignore = "loads the Qwen3.6-35B NVFP4 checkpoint (flag-off eager solo reference, then the \
            serving-default grouped MoE route with NV_Q36_BATCH_ROWWISE unset) under the REC env \
            stack at max_tokens=256 and drives c=1,2,4, reporting aggregate tok/s vs the \
            engine-level 225/384 rows and the greedy argmax drift class vs solo without gating \
            on byte identity; set NV_Q36_CONC_LADDER_TEST=1"]
fn default_grouped_route_ladder_reports_throughput_and_argmax_drift_class_vs_solo() {
    gate_or_panic();
    let sink = install_scheduler_event_sink();
    eprintln!("[q36-conc] {THE_DEFAULT_GROUPED_ROUTE_IS_ARGMAX_CLASS_NOT_BYTE_GATED}");
    let dir = qwen36_nvfp4_snapshot_dir_env_override_then_home_hub();

    ladder_env_common(SHIPPED_DEFAULT_BUCKET_PLAN_NV_Q36_BATCH_SIZES);
    record_env_stack_matching_the_engine_rate_ladder_basis();
    std::env::remove_var("NV_Q36_BATCH_ROWWISE");
    std::env::remove_var("NV_Q36_BATCH");
    let eager = load_engine(&dir, "rec-eager-reference").expect("flag-off eager engine must load");
    run_warmup(&eager, "rec-eager-reference");
    let refs = run_sequential(
        &eager,
        "rec-eager-reference",
        &FOUR_DISTINCT_PROMPTS_SO_NO_TWO_LANES_SHARE_A_PREFIX_AND_A_CROSSED_LANE_SHOWS_AS_A_BYTE_DIFF,
        MAX_TOKENS_256_MATCHES_THE_ENGINE_RATE_LADDERS_DEPTH_SO_THE_HTTP_ROWS_COMPARE_TO_ITS_225_AND_384_TOK_S,
    );
    assert_every_request_completed(1, &refs);
    drop(eager);

    std::env::set_var("NV_Q36_BATCH", "1");
    let batch = load_engine(&dir, "rec-batch").expect("NV_Q36_BATCH=1 engine must load");
    run_warmup(&batch, "rec-batch");

    for level in CONCURRENCY_LADDER_SPANS_SOLO_PARTIAL_GROUP_AND_FULL_GROUP {
        let mark = sink.mark();
        let (outs, wall_s) = run_concurrent(
            &batch,
            &format!("rec-batch-c{level}"),
            &FOUR_DISTINCT_PROMPTS_SO_NO_TWO_LANES_SHARE_A_PREFIX_AND_A_CROSSED_LANE_SHOWS_AS_A_BYTE_DIFF
                [..level],
            MAX_TOKENS_256_MATCHES_THE_ENGINE_RATE_LADDERS_DEPTH_SO_THE_HTTP_ROWS_COMPARE_TO_ITS_225_AND_384_TOK_S,
        );
        let events = sink.waves_since(mark);
        assert_every_request_completed(level, &outs);
        assert!(
            wall_s < WALL_BOUND_S_FOR_A_WHOLE_LEVEL_240_SO_ONLY_A_STARVED_REQUEST_TRIPS_IT,
            "c={level} took {wall_s:.1}s: a request was starved or the dispatcher stalled"
        );
        let formed = formed_sizes(&events);
        if level > 1 {
            assert_eq!(
                formed,
                vec![level],
                "c={level} must serve as ONE lane group; a narrower formation (or two waves) \
                 measures the scheduler's wave policy, not the engine at B={level}. \
                 events={events:?}"
            );
        }
        let drift = greedy_argmax_drift_vs_solo(&outs, &refs);
        let tokens: u64 = outs.iter().map(|o| o.completion_tokens).sum();
        let ms_tok: Vec<f64> = outs.iter().map(|o| o.ms_per_token_after_the_first()).collect();
        let first_ms: Vec<f64> = outs.iter().map(|o| o.first_token_s * 1000.0).collect();
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
        let class = if drift.differing_steps == 0 {
            "argmax_equal_to_solo"
        } else {
            "argmax_drift"
        };
        println!(
            "Q36-CONC c={level} agg_tok_s={:.1} \
             agg_def=output_tokens_over_batch_wall_incl_ttft_same_as_vllm_bench_serving \
             steady_agg_tok_s={steady_agg_tok_s:.1} p50_tpot_ms={:.2} p50_ttft_ms={:.0} \
             tokens={tokens} wall_s={wall_s:.2} group_shape={formed:?} sizes={} rowwise=unset \
             drift_steps={}/{} drift_requests={}/{} drift_class={class} drift_detail={}",
            tokens as f64 / wall_s.max(1e-9),
            percentile(ms_tok, 0.5),
            percentile(first_ms, 0.5),
            SHIPPED_DEFAULT_BUCKET_PLAN_NV_Q36_BATCH_SIZES,
            drift.differing_steps,
            drift.total_steps,
            drift.requests_differing,
            level,
            drift.per_request.join(","),
        );
    }
}
