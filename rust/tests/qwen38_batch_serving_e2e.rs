#![cfg(feature = "cuda")]

#[path = "common/qwen38_fixture.rs"]
#[allow(dead_code)]
mod qwen38_fixture;

use std::sync::Arc;

use axum::body::{to_bytes, Body};
use axum::http::{Request, StatusCode};
use axum::routing::post;
use axum::Router;
use tower::ServiceExt;

use qwen38_fixture::qwen38_nvfp4_snapshot_dir_env_override_then_home_hub;
use speaches_plus::oapi::chat::{handle_chat_completions, ChatAppState, ChatEngine};
use speaches_plus::oapi::chat_engine::{ChatRegistry, NvEngineChat};

const REAL_WEIGHTS_TEST_ENV: &str = "NV_Q38_BATCH_SERVE_TEST";

const FOUR_PROMPTS_LONG_GENERATIONS_SO_DECODE_RATE_DOMINATES_PREFILL: [&str; 4] = [
    "Write one paragraph of about 150 words describing a lighthouse keeper's morning routine.",
    "Write one paragraph of about 150 words explaining why the sky appears blue during the day.",
    "Write one paragraph of about 150 words describing how a river shapes a valley over time.",
    "Write one paragraph of about 150 words explaining how honeybees communicate about food.",
];

const MAX_TOKENS_160_SO_EVERY_LANE_DECODES_LONG_ENOUGH_TO_MEASURE: u32 = 160;

const MTP_VERIFY_ROWS_ARE_A_DIFFERENT_NUMERIC_CLASS_THAN_M1_STEPPING: &str =
    "the MTP verify pass scores k+1 rows per round, which is not the m=1 row-twin route, so \
     MTP-greedy may flip a knife-edge token vs eager stepping; the batch lanes are proven \
     bit-identical to eager m=1 stepping under NV_Q38_BATCH_ROWWISE=ffn, so the group output is \
     asserted against the eager solo reference and the MTP reference is asserted only against \
     the flag-on solo path, which is the same code";

fn app(engine: Arc<dyn ChatEngine>) -> Router {
    Router::new()
        .route("/v1/chat/completions", post(handle_chat_completions))
        .with_state(ChatAppState {
            registry: ChatRegistry::single(engine),
        })
}

#[derive(Clone)]
struct RunOut {
    content: String,
    reasoning: String,
    finish_reason: String,
    completion_tokens: u64,
    prompt_tokens: u64,
    wall_s: f64,
}

impl RunOut {
    fn stream(&self) -> String {
        format!("{}{}", self.reasoning, self.content)
    }
}

fn chat_request(model_id: &str, prompt: &str) -> Request<Body> {
    let body = format!(
        r#"{{"model":"{}","max_tokens":{},"temperature":0,
             "enable_thinking":false,
             "messages":[{{"role":"user","content":{}}}]}}"#,
        model_id,
        MAX_TOKENS_160_SO_EVERY_LANE_DECODES_LONG_ENOUGH_TO_MEASURE,
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
    let t = std::time::Instant::now();
    let resp = router
        .oneshot(chat_request(&model_id, &prompt))
        .await
        .unwrap();
    let status = resp.status();
    let bytes = to_bytes(resp.into_body(), 1 << 22).await.unwrap();
    let text = String::from_utf8_lossy(&bytes).into_owned();
    let wall_s = t.elapsed().as_secs_f64();
    assert_eq!(status, StatusCode::OK, "[{label}] {text}");
    let v: serde_json::Value = serde_json::from_str(&text).unwrap();
    let out = RunOut {
        content: v["choices"][0]["message"]["content"]
            .as_str()
            .unwrap_or_default()
            .to_string(),
        reasoning: v["choices"][0]["message"]["reasoning_content"]
            .as_str()
            .unwrap_or_default()
            .to_string(),
        finish_reason: v["choices"][0]["finish_reason"]
            .as_str()
            .unwrap_or_default()
            .to_string(),
        completion_tokens: v["usage"]["completion_tokens"].as_u64().unwrap_or(0),
        prompt_tokens: v["usage"]["prompt_tokens"].as_u64().unwrap_or(0),
        wall_s,
    };
    eprintln!(
        "[q38-batch-e2e] [{label}] prompt_tokens={} completion_tokens={} finish={} wall={:.2}s \
         ({:.1} ms/tok incl. prefill)",
        out.prompt_tokens,
        out.completion_tokens,
        out.finish_reason,
        out.wall_s,
        out.wall_s * 1000.0 / out.completion_tokens.max(1) as f64
    );
    assert!(out.completion_tokens > 0, "[{label}] no tokens generated");
    out
}

fn run_sequential(engine: &Arc<dyn ChatEngine>, label: &str) -> Vec<RunOut> {
    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    rt.block_on(async {
        let mut outs = Vec::new();
        for (i, p) in FOUR_PROMPTS_LONG_GENERATIONS_SO_DECODE_RATE_DOMINATES_PREFILL
            .iter()
            .enumerate()
        {
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
    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    rt.block_on(async {
        let _ = run_one(
            app(engine.clone()),
            format!("{label}-warmup"),
            engine.model_id().to_string(),
            FOUR_PROMPTS_LONG_GENERATIONS_SO_DECODE_RATE_DOMINATES_PREFILL[0].to_string(),
        )
        .await;
    });
}

fn run_concurrent(engine: &Arc<dyn ChatEngine>, label: &str) -> (Vec<RunOut>, f64) {
    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    rt.block_on(async {
        let t = std::time::Instant::now();
        let futs = FOUR_PROMPTS_LONG_GENERATIONS_SO_DECODE_RATE_DOMINATES_PREFILL
            .iter()
            .enumerate()
            .map(|(i, p)| {
                run_one(
                    app(engine.clone()),
                    format!("{label}-conc{i}"),
                    engine.model_id().to_string(),
                    (*p).to_string(),
                )
            });
        let outs = futures_util::future::join_all(futs).await;
        (outs, t.elapsed().as_secs_f64())
    })
}

fn assert_same_stream(label: &str, got: &RunOut, want: &RunOut) {
    let (g, w) = (got.stream(), want.stream());
    if g != w {
        let at = g
            .chars()
            .zip(w.chars())
            .position(|(a, b)| a != b)
            .unwrap_or_else(|| g.chars().count().min(w.chars().count()));
        panic!(
            "[{label}] greedy output diverged from its reference at char {at}:\n got: \
             {g:?}\nwant: {w:?}"
        );
    }
    assert_eq!(got.completion_tokens, want.completion_tokens, "[{label}]");
    assert_eq!(got.prompt_tokens, want.prompt_tokens, "[{label}]");
}

fn load_engine(dir: &std::path::Path, label: &str) -> Arc<dyn ChatEngine> {
    let t = std::time::Instant::now();
    let engine = NvEngineChat::try_load(dir)
        .unwrap_or_else(|e| panic!("[{label}] NvEngineChat::try_load: {e:#}"));
    eprintln!(
        "[q38-batch-e2e] [{label}] engine ready in {:.1}s",
        t.elapsed().as_secs_f64()
    );
    Arc::new(engine)
}

#[test]
#[ignore = "loads the ~16 GB Qwen3.8-27B NVFP4 checkpoint three times (mtp solo, eager solo, \
            NV_Q38_BATCH=1); set NV_Q38_BATCH_SERVE_TEST=1"]
fn four_concurrent_requests_batch_decode_together_and_match_the_solo_reference() {
    if std::env::var(REAL_WEIGHTS_TEST_ENV).as_deref() != Ok("1") {
        panic!(
            "this test was asked for BY NAME but {REAL_WEIGHTS_TEST_ENV}=1 is not set. This is \
             a SKIP, not a pass."
        );
    }
    let _ = tracing_subscriber::fmt()
        .with_env_filter("speaches_plus=info")
        .with_writer(std::io::stderr)
        .try_init();
    eprintln!("[q38-batch-e2e] {MTP_VERIFY_ROWS_ARE_A_DIFFERENT_NUMERIC_CLASS_THAN_M1_STEPPING}");
    let dir = qwen38_nvfp4_snapshot_dir_env_override_then_home_hub();
    std::env::set_var("NV_QWEN35_DENSE_CUDA_SERVE", "1");
    std::env::set_var("NV_GDN_FUSED_DECODE", "1");
    std::env::remove_var("NV_Q38_BATCH");

    std::env::set_var("NV_DRAFTER", "mtp");
    let mtp_engine = load_engine(&dir, "mtp-solo");
    run_warmup(&mtp_engine, "mtp-solo");
    let mtp_ref = run_sequential(&mtp_engine, "mtp-solo");
    let mtp_repeat = run_sequential(&mtp_engine, "mtp-solo-repeat");
    for (i, (a, b)) in mtp_ref.iter().zip(mtp_repeat.iter()).enumerate() {
        assert_same_stream(&format!("mtp-solo-repeat{i}"), b, a);
    }
    drop(mtp_repeat);
    drop(mtp_engine);

    std::env::remove_var("NV_DRAFTER");
    let eager_engine = load_engine(&dir, "eager-solo");
    run_warmup(&eager_engine, "eager-solo");
    let eager_ref = run_sequential(&eager_engine, "eager-solo");
    drop(eager_engine);

    std::env::set_var("NV_DRAFTER", "mtp");
    std::env::set_var("NV_Q38_BATCH", "1");
    std::env::set_var("NV_Q38_BATCH_ROWWISE", "ffn");
    let batch_engine = load_engine(&dir, "batch");
    run_warmup(&batch_engine, "batch");
    let seq = run_sequential(&batch_engine, "batch-solo-path");
    for (i, (got, want)) in seq.iter().zip(mtp_ref.iter()).enumerate() {
        assert_same_stream(
            &format!(
                "batch-solo-path{i} (NV_Q38_BATCH=1 with one active request must route to the \
                 existing MTP solo path and match the flag-off MTP reference byte for byte)"
            ),
            got,
            want,
        );
    }

    let (conc, conc_wall) = run_concurrent(&batch_engine, "batch");
    for (i, (got, want)) in conc.iter().zip(eager_ref.iter()).enumerate() {
        assert_same_stream(
            &format!(
                "batch-conc{i} (the lane group is bit-identical to eager m=1 stepping under \
                 NV_Q38_BATCH_ROWWISE=ffn, so the group output must match the eager solo \
                 reference byte for byte)"
            ),
            got,
            want,
        );
    }

    for (i, (m, e)) in mtp_ref.iter().zip(eager_ref.iter()).enumerate() {
        let same = m.stream() == e.stream();
        eprintln!(
            "[q38-batch-e2e] TABLE numeric_class_check prompt={i} mtp_equals_eager={same}"
        );
    }

    let seq_wall: f64 = seq.iter().map(|o| o.wall_s).sum();
    let seq_tokens: u64 = seq.iter().map(|o| o.completion_tokens).sum();
    let eager_wall: f64 = eager_ref.iter().map(|o| o.wall_s).sum();
    let eager_tokens: u64 = eager_ref.iter().map(|o| o.completion_tokens).sum();
    let conc_tokens: u64 = conc.iter().map(|o| o.completion_tokens).sum();
    let seq_agg = seq_tokens as f64 / seq_wall;
    let eager_agg = eager_tokens as f64 / eager_wall;
    let conc_agg = conc_tokens as f64 / conc_wall;
    eprintln!(
        "[q38-batch-e2e] TABLE mode=sequential_solo_mtp requests=4 tokens={seq_tokens} \
         wall={seq_wall:.2}s agg={seq_agg:.1} tok/s"
    );
    eprintln!(
        "[q38-batch-e2e] TABLE mode=sequential_solo_eager requests=4 tokens={eager_tokens} \
         wall={eager_wall:.2}s agg={eager_agg:.1} tok/s"
    );
    eprintln!(
        "[q38-batch-e2e] TABLE mode=concurrent_batch requests=4 tokens={conc_tokens} \
         wall={conc_wall:.2}s agg={conc_agg:.1} tok/s vs_mtp_seq={:.2}x vs_eager_seq={:.2}x",
        conc_agg / seq_agg,
        conc_agg / eager_agg
    );
    for (i, o) in conc.iter().enumerate() {
        eprintln!(
            "[q38-batch-e2e] TABLE lane_request={i} tokens={} wall={:.2}s per_req={:.1} tok/s",
            o.completion_tokens,
            o.wall_s,
            o.completion_tokens as f64 / o.wall_s
        );
    }
    assert!(
        conc_wall < seq_wall * 1.10,
        "4 concurrent requests ({conc_wall:.2}s) must not lose more than 10% to 4 sequential \
         MTP requests ({seq_wall:.2}s) on the same build; a bigger loss means the group \
         scheduler is serializing"
    );
    assert!(
        conc_wall < eager_wall,
        "4 concurrent requests ({conc_wall:.2}s) must beat 4 sequential non-spec requests \
         ({eager_wall:.2}s), else batching is vacuous"
    );
}

#[test]
#[ignore = "loads the ~16 GB Qwen3.8-27B NVFP4 checkpoint once with NV_Q38_BATCH=1 and the \
            default (non-rowwise) batch routes; rate is REPORTED, outputs are sanity-checked \
            but not byte-asserted because the m=4 MLP route is tolerance-class, not byte-class; \
            set NV_Q38_BATCH_SERVE_TEST=1"]
fn concurrent_batch_rate_on_the_default_m4_routes_reported_not_byte_asserted() {
    if std::env::var(REAL_WEIGHTS_TEST_ENV).as_deref() != Ok("1") {
        panic!(
            "this test was asked for BY NAME but {REAL_WEIGHTS_TEST_ENV}=1 is not set. This is \
             a SKIP, not a pass."
        );
    }
    let dir = qwen38_nvfp4_snapshot_dir_env_override_then_home_hub();
    std::env::set_var("NV_QWEN35_DENSE_CUDA_SERVE", "1");
    std::env::set_var("NV_GDN_FUSED_DECODE", "1");
    std::env::set_var("NV_DRAFTER", "mtp");
    std::env::set_var("NV_Q38_BATCH", "1");
    std::env::remove_var("NV_Q38_BATCH_ROWWISE");
    let engine = load_engine(&dir, "batch-m4");
    run_warmup(&engine, "batch-m4");
    let seq = run_sequential(&engine, "batch-m4-solo-path");
    let (conc, conc_wall) = run_concurrent(&engine, "batch-m4");
    for (i, o) in conc.iter().enumerate() {
        assert!(
            !o.stream().trim().is_empty(),
            "[batch-m4-conc{i}] empty output"
        );
    }
    let seq_wall: f64 = seq.iter().map(|o| o.wall_s).sum();
    let seq_tokens: u64 = seq.iter().map(|o| o.completion_tokens).sum();
    let conc_tokens: u64 = conc.iter().map(|o| o.completion_tokens).sum();
    eprintln!(
        "[q38-batch-e2e] TABLE mode=sequential_solo_mtp_m4run requests=4 tokens={seq_tokens} \
         wall={seq_wall:.2}s agg={:.1} tok/s",
        seq_tokens as f64 / seq_wall
    );
    eprintln!(
        "[q38-batch-e2e] TABLE mode=concurrent_batch_default_m4 requests=4 tokens={conc_tokens} \
         wall={conc_wall:.2}s agg={:.1} tok/s vs_mtp_seq={:.2}x",
        conc_tokens as f64 / conc_wall,
        (conc_tokens as f64 / conc_wall) / (seq_tokens as f64 / seq_wall)
    );
}
