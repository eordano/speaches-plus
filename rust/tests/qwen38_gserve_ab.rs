#![cfg(feature = "cuda")]

#[path = "common/qwen38_fixture.rs"]
#[allow(dead_code)]
mod qwen38_fixture;

use std::sync::Arc;

use axum::body::Body;
use axum::http::{Request, StatusCode};
use axum::routing::post;
use axum::Router;
use futures_util::StreamExt;
use tower::ServiceExt;

use qwen38_fixture::qwen38_nvfp4_snapshot_dir_env_override_then_home_hub;
use speaches_plus::oapi::chat::{handle_chat_completions, ChatAppState, ChatEngine};
use speaches_plus::oapi::chat_engine::{ChatRegistry, NvEngineChat};

const REAL_WEIGHTS_TEST_ENV: &str = "NV_Q38_GSERVE_TEST";

const TIMED_MAX_TOKENS_256_THE_SCOREBOARD_GENERATION_LENGTH: u32 = 256;

const ARM_LABEL_MIRRORS_QWEN38_SERVE_GRAPHED_ENABLED_BECAUSE_UNSET_NOW_BOOTS_THE_GRAPHED_ENGINE_AND_A_ROW_LABELED_BY_THE_OLD_OPT_IN_POLARITY_PUBLISHES_THE_GRAPHED_NUMBER_AS_EAGER:
    &str = "NV_Q38_SERVE_GRAPHED unset (or 1/on) boots the graphed solo session; only 0/off \
     boots the eager per-request engine, and NV_Q38_BATCH=1 would hand the model to the lane \
     pool and make both labels wrong, so this bench refuses to run beside it";

const WARMUP_MAX_TOKENS_16_WARMS_JIT_WITHOUT_MOVING_THE_TIMED_WINDOW: u32 = 16;

const TILE_PARAGRAPH: &str = "The lighthouse keeper rose before dawn and climbed the spiral \
     stair, counting each cold iron step out of habit rather than need. Below, the sea worked \
     at the rocks with a patience older than the tower, and the lamp room smelled of brass \
     polish and paraffin. He logged the wind, trimmed the wick, and watched the first fishing \
     boats lean into the swell, their lanterns pale against the greying water. ";

fn app(engine: Arc<dyn ChatEngine>) -> Router {
    Router::new()
        .route("/v1/chat/completions", post(handle_chat_completions))
        .with_state(ChatAppState {
            registry: ChatRegistry::single(engine),
        })
}

struct RunOut {
    content_chars: usize,
    finish_reason: String,
    completion_tokens: u64,
    prompt_tokens: u64,
    wall_s: f64,
    first_token_s: f64,
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

async fn run_one(router: Router, label: &str, model_id: &str, prompt: &str, max_tokens: u32) -> RunOut {
    let t0 = std::time::Instant::now();
    let resp = router
        .oneshot(chat_request(model_id, prompt, max_tokens))
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
        content_chars: 0,
        finish_reason: String::new(),
        completion_tokens: 0,
        prompt_tokens: 0,
        wall_s: 0.0,
        first_token_s: 0.0,
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
                panic!("[{label}] engine error mid-stream: {msg}");
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
            out.content_chars += c.len() + r.len();
            if let Some(f) = v["choices"][0]["finish_reason"].as_str() {
                out.finish_reason = f.to_string();
            }
        }
    }
    out.wall_s = t0.elapsed().as_secs_f64();
    out.first_token_s = first_token_s.unwrap_or(out.wall_s);
    eprintln!(
        "[q38-gserve] [{label}] prompt_tokens={} completion_tokens={} finish={:?} \
         first_token={:.0}ms wall={:.2}s",
        out.prompt_tokens, out.completion_tokens, out.finish_reason,
        out.first_token_s * 1000.0, out.wall_s
    );
    out
}

fn depth_from_env_default_256() -> usize {
    std::env::var("NV_CTX_TOKENS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(256)
}

fn tiled_user_prompt_of_about(dir: &std::path::Path, target_tokens: usize) -> String {
    let tok = tokenizers::Tokenizer::from_file(dir.join("tokenizer.json")).expect("tokenizer");
    let ask = "Rewrite the following text, expanding every sentence with more detail:\n\n";
    let per_tile = tok
        .encode(TILE_PARAGRAPH, false)
        .expect("encode tile")
        .get_ids()
        .len();
    let template_margin = 64usize;
    let tiles = target_tokens
        .saturating_sub(template_margin)
        .div_ceil(per_tile)
        .max(1);
    let mut s = String::with_capacity(ask.len() + tiles * TILE_PARAGRAPH.len());
    s.push_str(ask);
    for _ in 0..tiles {
        s.push_str(TILE_PARAGRAPH);
    }
    let ids = tok.encode(s.as_str(), false).expect("encode tiled").get_ids().to_vec();
    if ids.len() > target_tokens.saturating_sub(template_margin).max(per_tile) {
        let cut = target_tokens.saturating_sub(template_margin).max(per_tile);
        return tok.decode(&ids[..cut], false).expect("decode cut");
    }
    s
}

fn snapshot_rev_label(dir: &std::path::Path) -> String {
    let rev = dir
        .file_name()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_else(|| "local".into());
    let repo = dir
        .ancestors()
        .find(|p| {
            p.file_name()
                .is_some_and(|n| n.to_string_lossy().starts_with("models--"))
        })
        .map(|p| p.file_name().unwrap().to_string_lossy().into_owned())
        .unwrap_or_else(|| "local-dir".into());
    format!("{repo}@{rev}")
}

#[test]
#[ignore = "loads the ~22.6 GB Qwen3.8-27B NVFP4 checkpoint through NvEngineChat::try_load and \
            times ONE streaming /v1/chat/completions generation at the NV_CTX_TOKENS prompt \
            depth (default 256) after a 16-token warmup request; set NV_Q38_GSERVE_TEST=1 and \
            NV_QWEN35_DENSE_CUDA_SERVE=1, pick the arm with NV_Q38_SERVE_GRAPHED (unset boots \
            the graphed default, 0 the eager engine), pin the \
            window with NV_KV_MAX_SEQ_LEN_Q38; the printed CTX-SCALING line is the serving \
            (HTTP+SSE) A/B row for the graphed-engine adoption -- ttft here includes the \
            per-request verify-graph capture the graphed arm pays after prefill; run ONE depth \
            per process via NV_CTX_TOKENS"]
fn qwen38_gserve_http_streaming_ttft_and_decode_rate_one_depth_per_process() {
    if std::env::var(REAL_WEIGHTS_TEST_ENV).as_deref() != Ok("1") {
        panic!("set {REAL_WEIGHTS_TEST_ENV}=1 to run (it must never silently skip)");
    }
    assert_eq!(
        std::env::var("NV_QWEN35_DENSE_CUDA_SERVE").as_deref(),
        Ok("1"),
        "PRECONDITION NOT MET, THIS TEST EXECUTED NOTHING: the qwen3.8 cuda serving arm is \
         opt-in per boot; pass NV_QWEN35_DENSE_CUDA_SERVE=1"
    );
    assert_ne!(
        std::env::var("NV_Q38_BATCH").ok().as_deref(),
        Some("1"),
        "{ARM_LABEL_MIRRORS_QWEN38_SERVE_GRAPHED_ENABLED_BECAUSE_UNSET_NOW_BOOTS_THE_GRAPHED_ENGINE_AND_A_ROW_LABELED_BY_THE_OLD_OPT_IN_POLARITY_PUBLISHES_THE_GRAPHED_NUMBER_AS_EAGER}"
    );
    let graphed = !matches!(
        std::env::var("NV_Q38_SERVE_GRAPHED").ok().as_deref(),
        Some("0") | Some("off")
    );
    let arm = if graphed { "gserve-graphed" } else { "gserve-eager" };
    let dir = qwen38_nvfp4_snapshot_dir_env_override_then_home_hub();
    let rev = snapshot_rev_label(&dir);
    let depth = depth_from_env_default_256();

    let t_boot = std::time::Instant::now();
    let engine: Arc<dyn ChatEngine> = Arc::new(
        NvEngineChat::try_load(&dir).unwrap_or_else(|e| panic!("[{arm}] try_load: {e:#}")),
    );
    let boot_s = t_boot.elapsed().as_secs_f64();
    let model_id = engine.model_id().to_string();
    let prompt = tiled_user_prompt_of_about(&dir, depth);

    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    let (warm, timed) = rt.block_on(async {
        let warm = run_one(
            app(engine.clone()),
            &format!("{arm}-warmup"),
            &model_id,
            "Describe a quiet harbor at dusk in one paragraph.",
            WARMUP_MAX_TOKENS_16_WARMS_JIT_WITHOUT_MOVING_THE_TIMED_WINDOW,
        )
        .await;
        let timed = run_one(
            app(engine.clone()),
            &format!("{arm}-timed"),
            &model_id,
            &prompt,
            TIMED_MAX_TOKENS_256_THE_SCOREBOARD_GENERATION_LENGTH,
        )
        .await;
        (warm, timed)
    });
    assert!(warm.completion_tokens > 0, "warmup generated nothing");
    assert_eq!(
        timed.finish_reason, "length",
        "the timed request must run to max_tokens so the rate is measured over the full \
         {TIMED_MAX_TOKENS_256_THE_SCOREBOARD_GENERATION_LENGTH}-token window; an early \
         '{}' stop makes the row incomparable",
        timed.finish_reason
    );
    let decode_toks = timed.completion_tokens.saturating_sub(1).max(1) as f64;
    let ms_per_tok = (timed.wall_s - timed.first_token_s) * 1000.0 / decode_toks;
    let tok_s = 1000.0 / ms_per_tok;
    println!(
        "CTX-SCALING qwen38-gserve depth={} arm={arm} steps={} warmup_steps={} \
         median_ms_tok={ms_per_tok:.3} tok_s={tok_s:.1} basis=wall_mean_after_first_delta \
         prefill_s={:.3} ttft_ms={:.0} boot_s={boot_s:.1} target_depth={depth} checkpoint={rev}",
        timed.prompt_tokens,
        timed.completion_tokens,
        warm.completion_tokens,
        timed.first_token_s,
        timed.first_token_s * 1000.0,
    );
}
