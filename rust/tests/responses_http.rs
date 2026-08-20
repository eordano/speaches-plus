use std::sync::Arc;

use axum::routing::{get, post};
use axum::Router;
use serde_json::{json, Value};

use speaches_plus::oapi::chat::{ChatAppState, ChatEngine};
use speaches_plus::oapi::chat_engine::{ChatRegistry, EchoEngine};
use speaches_plus::oapi::responses::{
    handle_delete_response, handle_get_response, handle_responses, RESPONSE_STORE_DIR_ENV,
};

const MODEL: &str = "echo-model";
const REPLY: &str = "alpha beta gamma";

static ENV_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

fn lock_env() -> std::sync::MutexGuard<'static, ()> {
    ENV_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}

struct StoreDir {
    path: std::path::PathBuf,
}

impl StoreDir {
    fn set() -> Self {
        let path = std::env::temp_dir().join(format!(
            "nvresp-{}-{:x}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos() as u64
        ));
        std::fs::create_dir_all(&path).unwrap();
        std::env::set_var(RESPONSE_STORE_DIR_ENV, &path);
        Self { path }
    }
}

impl Drop for StoreDir {
    fn drop(&mut self) {
        std::env::remove_var(RESPONSE_STORE_DIR_ENV);
        let _ = std::fs::remove_dir_all(&self.path);
    }
}

async fn boot_engine(eng: Arc<dyn ChatEngine>) -> u16 {
    let app = Router::new()
        .route("/v1/responses", post(handle_responses))
        .route(
            "/v1/responses/{id}",
            get(handle_get_response).delete(handle_delete_response),
        )
        .with_state(ChatAppState {
            registry: ChatRegistry::single(eng),
        });
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let port = listener.local_addr().unwrap().port();
    tokio::spawn(async move {
        axum::serve(listener, app).await.unwrap();
    });
    port
}

async fn boot() -> u16 {
    boot_engine(Arc::new(EchoEngine::new(MODEL, REPLY))).await
}

async fn post_json(port: u16, body: Value) -> (reqwest::StatusCode, Value) {
    let resp = reqwest::Client::new()
        .post(format!("http://127.0.0.1:{port}/v1/responses"))
        .json(&body)
        .send()
        .await
        .unwrap();
    let status = resp.status();
    let text = resp.text().await.unwrap();
    (
        status,
        serde_json::from_str(&text).unwrap_or(Value::String(text)),
    )
}

fn sse_events(body: &str) -> Vec<(String, Value)> {
    body.split("\n\n")
        .filter(|f| !f.trim().is_empty())
        .map(|frame| {
            let mut ev = String::new();
            let mut data = String::new();
            for line in frame.lines() {
                if let Some(rest) = line.strip_prefix("event: ") {
                    ev = rest.to_string();
                }
                if let Some(rest) = line.strip_prefix("data: ") {
                    data = rest.to_string();
                }
            }
            (ev, serde_json::from_str(&data).unwrap_or(Value::Null))
        })
        .collect()
}

struct CaptureEngine {
    inner: EchoEngine,
    prompt: Arc<std::sync::Mutex<String>>,
}

#[async_trait::async_trait]
impl ChatEngine for CaptureEngine {
    fn model_id(&self) -> &str {
        &self.inner.model_id
    }

    async fn generate(
        &self,
        req: speaches_plus::oapi::chat::ChatGenerateRequest,
        tx: tokio::sync::mpsc::Sender<speaches_plus::oapi::chat::ChatEvent>,
    ) -> anyhow::Result<()> {
        *self
            .prompt
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner) = req.prompt.clone();
        self.inner.generate(req, tx).await
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn non_streaming_returns_the_openai_response_shape() {
    let _guard = lock_env();
    let port = boot().await;
    let (status, v) = post_json(port, json!({"model": MODEL, "input": "hi"})).await;
    assert_eq!(status, reqwest::StatusCode::OK, "body: {v}");
    assert_eq!(v["object"], "response", "body: {v}");
    assert!(
        v["id"].as_str().unwrap_or_default().starts_with("resp_"),
        "body: {v}"
    );
    assert_eq!(v["status"], "completed", "body: {v}");
    assert_eq!(v["output"][0]["type"], "message", "body: {v}");
    assert_eq!(v["output"][0]["role"], "assistant", "body: {v}");
    assert_eq!(
        v["output"][0]["content"][0]["type"], "output_text",
        "body: {v}"
    );
    assert_eq!(v["output"][0]["content"][0]["text"], REPLY, "body: {v}");
    assert_eq!(v["output_text"], REPLY, "body: {v}");
    assert!(
        v["usage"]["input_tokens"].as_u64().unwrap_or(0) > 0,
        "body: {v}"
    );
    assert!(
        v["usage"]["output_tokens"].as_u64().unwrap_or(0) > 0,
        "body: {v}"
    );
    assert!(
        v["usage"]["input_tokens_details"]["cached_tokens"].is_u64(),
        "body: {v}"
    );
    assert_eq!(
        v["store"], false,
        "with no store dir configured the response must say it was not stored: {v}"
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn store_then_resume_rerenders_the_whole_conversation() {
    let _guard = lock_env();
    let _dir = StoreDir::set();
    let prompt = Arc::new(std::sync::Mutex::new(String::new()));
    let eng = CaptureEngine {
        inner: EchoEngine::new(MODEL, REPLY),
        prompt: prompt.clone(),
    };
    let port = boot_engine(Arc::new(eng)).await;

    let (status, first) = post_json(
        port,
        json!({"model": MODEL, "input": "the sky question", "store": true}),
    )
    .await;
    assert_eq!(status, reqwest::StatusCode::OK, "body: {first}");
    assert_eq!(first["store"], true, "body: {first}");
    let id = first["id"].as_str().unwrap().to_string();

    let (status, second) = post_json(
        port,
        json!({"model": MODEL, "input": "the follow-up", "previous_response_id": id}),
    )
    .await;
    assert_eq!(status, reqwest::StatusCode::OK, "body: {second}");
    assert_eq!(second["previous_response_id"], id, "body: {second}");
    let rendered = prompt
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .clone();
    assert!(
        rendered.contains("the sky question"),
        "resume must re-render the stored user turn: {rendered:?}"
    );
    assert!(
        rendered.contains(REPLY),
        "resume must re-render the stored assistant turn: {rendered:?}"
    );
    assert!(
        rendered.contains("the follow-up"),
        "resume must append the new turn: {rendered:?}"
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn get_returns_the_stored_response_and_delete_removes_it() {
    let _guard = lock_env();
    let _dir = StoreDir::set();
    let port = boot().await;
    let (_, first) = post_json(port, json!({"model": MODEL, "input": "hi"})).await;
    let id = first["id"].as_str().unwrap().to_string();

    let client = reqwest::Client::new();
    let got: Value = client
        .get(format!("http://127.0.0.1:{port}/v1/responses/{id}"))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(got["id"], id.as_str(), "body: {got}");
    assert_eq!(got["output_text"], REPLY, "body: {got}");

    let deleted: Value = client
        .delete(format!("http://127.0.0.1:{port}/v1/responses/{id}"))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(deleted["deleted"], true, "body: {deleted}");

    let after = client
        .get(format!("http://127.0.0.1:{port}/v1/responses/{id}"))
        .send()
        .await
        .unwrap();
    assert_eq!(after.status(), reqwest::StatusCode::NOT_FOUND);
}

#[tokio::test(flavor = "multi_thread")]
async fn resuming_an_unknown_id_is_a_404_not_a_cold_serve() {
    let _guard = lock_env();
    let _dir = StoreDir::set();
    let port = boot().await;
    let (status, v) = post_json(
        port,
        json!({
            "model": MODEL,
            "input": "hi",
            "previous_response_id": format!("resp_{}", "0".repeat(32)),
        }),
    )
    .await;
    assert_eq!(status, reqwest::StatusCode::NOT_FOUND, "body: {v}");
    assert_eq!(v["error"]["type"], "invalid_request_error", "body: {v}");
}

#[tokio::test(flavor = "multi_thread")]
async fn streaming_emits_the_response_event_sequence() {
    let _guard = lock_env();
    let port = boot().await;
    let resp = reqwest::Client::new()
        .post(format!("http://127.0.0.1:{port}/v1/responses"))
        .json(&json!({"model": MODEL, "input": "hi", "stream": true}))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), reqwest::StatusCode::OK);
    let body = resp.text().await.unwrap();
    let events = sse_events(&body);
    let names: Vec<&str> = events.iter().map(|(e, _)| e.as_str()).collect();
    for expected in [
        "response.created",
        "response.in_progress",
        "response.output_item.added",
        "response.content_part.added",
        "response.output_text.delta",
        "response.output_text.done",
        "response.content_part.done",
        "response.output_item.done",
        "response.completed",
    ] {
        assert!(names.contains(&expected), "missing {expected} in {names:?}");
    }
    let assembled: String = events
        .iter()
        .filter(|(e, _)| e == "response.output_text.delta")
        .map(|(_, d)| d["delta"].as_str().unwrap_or_default().to_string())
        .collect();
    assert_eq!(assembled, REPLY, "deltas must assemble to the final text");
    let completed = &events.last().unwrap().1;
    assert_eq!(
        completed["response"]["status"], "completed",
        "body: {completed}"
    );
    assert_eq!(
        completed["response"]["output_text"], REPLY,
        "body: {completed}"
    );
    let seqs: Vec<u64> = events
        .iter()
        .filter_map(|(_, d)| d["sequence_number"].as_u64())
        .collect();
    assert!(
        seqs.windows(2).all(|w| w[0] < w[1]),
        "sequence_number must be strictly increasing: {seqs:?}"
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn codex_function_tools_are_rendered_and_hosted_tools_are_ignored() {
    let _guard = lock_env();
    let prompt = Arc::new(std::sync::Mutex::new(String::new()));
    let eng = CaptureEngine {
        inner: EchoEngine::new(MODEL, REPLY),
        prompt: prompt.clone(),
    };
    let port = boot_engine(Arc::new(eng)).await;
    let (status, v) = post_json(
        port,
        json!({
            "model": MODEL,
            "input": "hi",
            "tools": [
                {
                    "type": "function",
                    "name": "exec_command",
                    "description": "Run a command",
                    "strict": false,
                    "parameters": {
                        "type": "object",
                        "properties": {"cmd": {"type": "string"}},
                        "required": ["cmd"],
                        "additionalProperties": false
                    }
                },
                {"type": "web_search"}
            ],
            "tool_choice": "auto",
            "parallel_tool_calls": false
        }),
    )
    .await;
    assert_eq!(status, reqwest::StatusCode::OK, "body: {v}");
    assert_eq!(v["output"][0]["type"], "message", "body: {v}");
    assert_eq!(v["tools"].as_array().map(Vec::len), Some(2), "body: {v}");
    assert_eq!(v["parallel_tool_calls"], false, "body: {v}");
    let rendered = prompt
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .clone();
    assert!(rendered.contains("exec_command"), "prompt: {rendered}");
}

#[tokio::test(flavor = "multi_thread")]
async fn streaming_emits_function_call_items_and_argument_events() {
    let _guard = lock_env();
    let reply = r#"<tool_call>{"name":"exec_command","arguments":{"cmd":"pwd"}}</tool_call>"#;
    let port = boot_engine(Arc::new(EchoEngine::new(MODEL, reply))).await;
    let resp = reqwest::Client::new()
        .post(format!("http://127.0.0.1:{port}/v1/responses"))
        .json(&json!({
            "model": MODEL,
            "input": "show the directory",
            "stream": true,
            "tools": [{
                "type": "function",
                "name": "exec_command",
                "parameters": {
                    "type": "object",
                    "properties": {"cmd": {"type": "string"}},
                    "required": ["cmd"]
                }
            }]
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), reqwest::StatusCode::OK);
    let events = sse_events(&resp.text().await.unwrap());
    let names: Vec<&str> = events.iter().map(|(name, _)| name.as_str()).collect();
    for expected in [
        "response.output_item.added",
        "response.function_call_arguments.delta",
        "response.function_call_arguments.done",
        "response.output_item.done",
        "response.completed",
    ] {
        assert!(names.contains(&expected), "missing {expected} in {names:?}");
    }
    let added = events
        .iter()
        .find(|(name, data)| {
            name == "response.output_item.added" && data["item"]["type"] == "function_call"
        })
        .map(|(_, data)| data)
        .unwrap();
    assert_eq!(added["item"]["name"], "exec_command", "event: {added}");
    let completed = &events.last().unwrap().1["response"];
    assert_eq!(
        completed["output"][0]["type"], "function_call",
        "body: {completed}"
    );
    assert_eq!(
        completed["output"][0]["arguments"], r#"{"cmd":"pwd"}"#,
        "body: {completed}"
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn function_call_outputs_are_accepted_as_continuation_input() {
    let _guard = lock_env();
    let prompt = Arc::new(std::sync::Mutex::new(String::new()));
    let eng = CaptureEngine {
        inner: EchoEngine::new(MODEL, REPLY),
        prompt: prompt.clone(),
    };
    let port = boot_engine(Arc::new(eng)).await;
    let (status, v) = post_json(
        port,
        json!({
            "model": MODEL,
            "input": [
                {"type": "message", "role": "user", "content": "show the directory"},
                {
                    "type": "function_call",
                    "id": "fc_test",
                    "call_id": "call_test",
                    "name": "exec_command",
                    "arguments": "{\"cmd\":\"pwd\"}"
                },
                {
                    "type": "function_call_output",
                    "call_id": "call_test",
                    "output": "/workspace"
                }
            ],
            "tools": [{
                "type": "function",
                "name": "exec_command",
                "parameters": {
                    "type": "object",
                    "properties": {"cmd": {"type": "string"}},
                    "required": ["cmd"]
                }
            }]
        }),
    )
    .await;
    assert_eq!(status, reqwest::StatusCode::OK, "body: {v}");
    let rendered = prompt
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .clone();
    assert!(rendered.contains("exec_command"), "prompt: {rendered}");
    assert!(rendered.contains("/workspace"), "prompt: {rendered}");
}

#[cfg(feature = "wgpu")]
mod gated {
    use super::*;
    use speaches_plus::oapi::chat_engine_wgpu::{WgpuChatEngine, PREFIX_REUSE_ENV};

    #[tokio::test(flavor = "multi_thread")]
    #[ignore]
    async fn a_resumed_response_serves_warm_after_the_live_state_diverged() {
        if std::env::var("NV_WGPU_SERVE_TEST").as_deref() != Ok("1") {
            panic!("PRECONDITION NOT MET, THIS TEST EXECUTED NOTHING: set NV_WGPU_SERVE_TEST=1");
        }
        let dir = std::env::var("NV_WGPU_SERVE_DIR").expect(
            "no wgpu servable model dir: set NV_WGPU_SERVE_DIR. This is a SKIP, not a pass.",
        );
        let _guard = lock_env();
        let _store = StoreDir::set();
        std::env::set_var(PREFIX_REUSE_ENV, "1");
        let engine = Arc::new(
            WgpuChatEngine::load_with(std::path::Path::new(&dir), 4096, None)
                .expect("wgpu engine did not load"),
        );
        let model = engine.model_id().to_string();
        let port = boot_engine(engine as Arc<dyn ChatEngine>).await;

        let ask = |input: &str, prev: Option<String>| {
            let mut body = json!({
                "model": model,
                "input": input,
                "max_output_tokens": 16,
                "temperature": 0,
                "store": true,
            });
            if let Some(p) = prev {
                body["previous_response_id"] = json!(p);
            }
            body
        };
        let (status, first) = post_json(port, ask("Count to five, digits only.", None)).await;
        assert_eq!(status, reqwest::StatusCode::OK, "body: {first}");
        let id = first["id"].as_str().unwrap().to_string();

        let (status, control) = post_json(port, ask("Now count to three.", Some(id.clone()))).await;
        assert_eq!(status, reqwest::StatusCode::OK, "body: {control}");

        let (status, wander) =
            post_json(port, ask("Name four rivers on different continents.", None)).await;
        assert_eq!(status, reqwest::StatusCode::OK, "body: {wander}");

        let (status, resumed) = post_json(port, ask("Now count to three.", Some(id))).await;
        assert_eq!(status, reqwest::StatusCode::OK, "body: {resumed}");
        let cached = resumed["usage"]["input_tokens_details"]["cached_tokens"]
            .as_u64()
            .unwrap_or(0);
        assert!(
            cached > 0,
            "a resume after divergence served nothing warm; the snapshot restore did not happen: {resumed}"
        );
        assert_eq!(
            resumed["output_text"], control["output_text"],
            "a resumed continuation must serve the same completion as the sequential one"
        );
        std::env::remove_var(PREFIX_REUSE_ENV);
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn resume_with_no_store_dir_is_a_404() {
    let _guard = lock_env();
    std::env::remove_var(RESPONSE_STORE_DIR_ENV);
    let port = boot().await;
    let (status, v) = post_json(
        port,
        json!({
            "model": MODEL,
            "input": "hi",
            "previous_response_id": format!("resp_{}", "a".repeat(32)),
        }),
    )
    .await;
    assert_eq!(status, reqwest::StatusCode::NOT_FOUND, "body: {v}");
}
