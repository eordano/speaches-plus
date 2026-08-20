use std::sync::Arc;

use axum::routing::post;
use axum::Router;
use serde_json::{json, Value};

use speaches_plus::oapi::chat::{
    ChatAppState, ChatEngine, ChatEvent, ChatGenerateRequest, EngineBusy, ENGINE_BUSY_PREFIX,
    MAX_MAX_TOKENS,
};
use speaches_plus::oapi::chat_engine::{ChatRegistry, EchoEngine};
use speaches_plus::oapi::messages::{handle_count_tokens, handle_messages, OVERLOADED_STATUS};

const MODEL: &str = "echo-model";
const REPLY: &str = "alpha beta gamma";

async fn boot() -> u16 {
    let eng: Arc<dyn ChatEngine> = Arc::new(EchoEngine::new(MODEL, REPLY));
    let app = Router::new()
        .route("/v1/messages", post(handle_messages))
        .route("/v1/messages/count_tokens", post(handle_count_tokens))
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

async fn post_json(port: u16, path: &str, body: Value) -> (reqwest::StatusCode, Value, String) {
    let resp = reqwest::Client::new()
        .post(format!("http://127.0.0.1:{port}{path}"))
        .header("x-api-key", "test")
        .header("anthropic-version", "2023-06-01")
        .json(&body)
        .send()
        .await
        .unwrap();
    let status = resp.status();
    let request_id = resp
        .headers()
        .get("request-id")
        .and_then(|v| v.to_str().ok())
        .unwrap_or_default()
        .to_string();
    let text = resp.text().await.unwrap();
    let v = serde_json::from_str(&text).unwrap_or(Value::String(text));
    (status, v, request_id)
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

#[tokio::test(flavor = "multi_thread")]
async fn non_streaming_returns_anthropic_message_shape() {
    let port = boot().await;
    let (status, v, request_id) = post_json(
        port,
        "/v1/messages",
        json!({
            "model": MODEL,
            "max_tokens": 64,
            "messages": [{"role": "user", "content": "hi"}],
        }),
    )
    .await;
    assert_eq!(status, reqwest::StatusCode::OK, "body: {v}");
    assert!(request_id.starts_with("req_"), "request-id: {request_id:?}");
    assert_eq!(v["type"], "message");
    assert_eq!(v["role"], "assistant");
    assert_eq!(v["model"], MODEL);
    assert!(v["id"].as_str().unwrap().starts_with("msg_"));
    assert_eq!(v["stop_reason"], "end_turn");
    assert_eq!(v["stop_sequence"], Value::Null);
    assert_eq!(v["content"][0]["type"], "text");
    assert_eq!(v["content"][0]["text"], REPLY);
    assert!(v["usage"]["input_tokens"].as_u64().unwrap() > 0);
    assert_eq!(v["usage"]["output_tokens"], 3);
}

#[tokio::test(flavor = "multi_thread")]
async fn streaming_emits_the_full_event_lifecycle() {
    let port = boot().await;
    let resp = reqwest::Client::new()
        .post(format!("http://127.0.0.1:{port}/v1/messages"))
        .header("anthropic-version", "2023-06-01")
        .json(&json!({
            "model": MODEL,
            "max_tokens": 64,
            "stream": true,
            "messages": [{"role": "user", "content": "hi"}],
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), reqwest::StatusCode::OK);
    assert_eq!(
        resp.headers()
            .get(reqwest::header::CONTENT_TYPE)
            .and_then(|v| v.to_str().ok()),
        Some("text/event-stream")
    );
    let body = resp.text().await.unwrap();
    let events = sse_events(&body);
    let names: Vec<&str> = events.iter().map(|(e, _)| e.as_str()).collect();

    assert_eq!(names[0], "message_start");
    assert_eq!(names[1], "ping");
    assert_eq!(names[2], "content_block_start");
    assert_eq!(*names.last().unwrap(), "message_stop");
    assert!(names.contains(&"content_block_delta"));
    assert!(names.contains(&"content_block_stop"));
    assert!(names.contains(&"message_delta"));

    let (_, start) = &events[0];
    assert_eq!(start["message"]["role"], "assistant");
    assert!(start["message"]["usage"]["input_tokens"].as_u64().unwrap() > 0);

    let (_, block_start) = &events[2];
    assert_eq!(block_start["content_block"]["type"], "text");
    assert_eq!(block_start["index"], 0);

    let text: String = events
        .iter()
        .filter(|(e, d)| e == "content_block_delta" && d["delta"]["type"] == "text_delta")
        .map(|(_, d)| d["delta"]["text"].as_str().unwrap().to_string())
        .collect();
    assert_eq!(text, REPLY);

    let (_, md) = events.iter().find(|(e, _)| e == "message_delta").unwrap();
    assert_eq!(md["delta"]["stop_reason"], "end_turn");
    assert_eq!(md["usage"]["output_tokens"], 3);
}

#[tokio::test(flavor = "multi_thread")]
async fn missing_max_tokens_is_an_invalid_request_error() {
    let port = boot().await;
    let (status, v, _) = post_json(
        port,
        "/v1/messages",
        json!({
            "model": MODEL,
            "messages": [{"role": "user", "content": "hi"}],
        }),
    )
    .await;
    assert_eq!(status, reqwest::StatusCode::BAD_REQUEST);
    assert_eq!(v["type"], "error");
    assert_eq!(v["error"]["type"], "invalid_request_error");
    assert!(v["error"]["message"].as_str().unwrap().contains("max_tokens"));
}

#[tokio::test(flavor = "multi_thread")]
async fn image_blocks_are_rejected_with_the_anthropic_envelope() {
    let port = boot().await;
    let (status, v, _) = post_json(
        port,
        "/v1/messages",
        json!({
            "model": MODEL,
            "max_tokens": 64,
            "messages": [{"role": "user", "content": [
                {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": ""}},
            ]}],
        }),
    )
    .await;
    assert_eq!(status, reqwest::StatusCode::BAD_REQUEST);
    assert_eq!(v["type"], "error");
    assert_eq!(v["error"]["type"], "invalid_request_error");
    assert!(v["error"]["message"].as_str().unwrap().contains("image"));
}

#[tokio::test(flavor = "multi_thread")]
async fn count_tokens_without_a_tokenizer_fails_honestly() {
    let _lock = TOK_ENV_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let port = boot().await;
    let (status, v, _) = post_json(
        port,
        "/v1/messages/count_tokens",
        json!({
            "model": MODEL,
            "messages": [{"role": "user", "content": "hi"}],
        }),
    )
    .await;
    assert_eq!(status, reqwest::StatusCode::NOT_IMPLEMENTED, "body: {v}");
    assert_eq!(v["type"], "error");
    assert_eq!(v["error"]["type"], "api_error");
}

#[tokio::test(flavor = "multi_thread")]
async fn tool_definitions_render_and_calls_parse() {
    let reply = "<tool_call>{\"name\": \"get_weather\", \"arguments\": {\"city\": \"Paris\"}}</tool_call>";
    let eng: Arc<dyn ChatEngine> = Arc::new(EchoEngine::new(MODEL, reply));
    let app = Router::new()
        .route("/v1/messages", post(handle_messages))
        .with_state(ChatAppState {
            registry: ChatRegistry::single(eng),
        });
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let port = listener.local_addr().unwrap().port();
    tokio::spawn(async move {
        axum::serve(listener, app).await.unwrap();
    });

    let (status, v, _) = post_json(
        port,
        "/v1/messages",
        json!({
            "model": MODEL,
            "max_tokens": 64,
            "tools": [{
                "name": "get_weather",
                "description": "weather lookup",
                "input_schema": {"type": "object", "properties": {"city": {"type": "string"}}},
            }],
            "messages": [{"role": "user", "content": "weather in paris?"}],
        }),
    )
    .await;
    assert_eq!(status, reqwest::StatusCode::OK, "body: {v}");
    assert_eq!(v["stop_reason"], "tool_use", "body: {v}");
    let block = v["content"]
        .as_array()
        .unwrap()
        .iter()
        .find(|b| b["type"] == "tool_use")
        .unwrap();
    assert_eq!(block["name"], "get_weather");
    assert_eq!(block["input"], json!({"city": "Paris"}));
    assert!(block["id"].as_str().unwrap().len() > 0);
}

struct ThinkEchoEngine {
    inner: EchoEngine,
}

#[async_trait::async_trait]
impl ChatEngine for ThinkEchoEngine {
    fn model_id(&self) -> &str {
        &self.inner.model_id
    }

    fn thinking_split_supported(&self) -> bool {
        true
    }

    async fn generate(
        &self,
        req: speaches_plus::oapi::chat::ChatGenerateRequest,
        tx: tokio::sync::mpsc::Sender<speaches_plus::oapi::chat::ChatEvent>,
    ) -> anyhow::Result<()> {
        self.inner.generate(req, tx).await
    }
}

async fn boot_engine(eng: Arc<dyn ChatEngine>) -> u16 {
    let app = Router::new()
        .route("/v1/messages", post(handle_messages))
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

async fn stream_events(port: u16, body: Value) -> Vec<(String, Value)> {
    let resp = reqwest::Client::new()
        .post(format!("http://127.0.0.1:{port}/v1/messages"))
        .header("anthropic-version", "2023-06-01")
        .json(&body)
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), reqwest::StatusCode::OK);
    sse_events(&resp.text().await.unwrap())
}

#[tokio::test(flavor = "multi_thread")]
async fn streaming_tool_use_emits_the_tool_block_lifecycle() {
    let reply = "<tool_call>{\"name\": \"get_weather\", \"arguments\": {\"city\": \"Paris\"}}</tool_call>";
    let port = boot_engine(Arc::new(EchoEngine::new(MODEL, reply))).await;
    let events = stream_events(
        port,
        json!({
            "model": MODEL,
            "max_tokens": 64,
            "stream": true,
            "tools": [{
                "name": "get_weather",
                "description": "weather lookup",
                "input_schema": {"type": "object", "properties": {"city": {"type": "string"}}},
            }],
            "messages": [{"role": "user", "content": "weather in paris?"}],
        }),
    )
    .await;

    let names: Vec<&str> = events.iter().map(|(e, _)| e.as_str()).collect();
    assert_eq!(names[0], "message_start");
    assert_eq!(names[1], "ping");
    assert_eq!(*names.last().unwrap(), "message_stop");

    let (_, tool_start) = events
        .iter()
        .find(|(e, d)| e == "content_block_start" && d["content_block"]["type"] == "tool_use")
        .expect("tool_use content_block_start");
    assert_eq!(tool_start["content_block"]["name"], "get_weather");
    assert_eq!(tool_start["content_block"]["input"], json!({}));
    assert!(tool_start["content_block"]["id"].as_str().unwrap().len() > 0);
    let tool_index = tool_start["index"].clone();

    let (_, jd) = events
        .iter()
        .find(|(e, d)| e == "content_block_delta" && d["delta"]["type"] == "input_json_delta")
        .expect("input_json_delta");
    assert_eq!(jd["index"], tool_index);
    let parsed: Value =
        serde_json::from_str(jd["delta"]["partial_json"].as_str().unwrap()).unwrap();
    assert_eq!(parsed, json!({"city": "Paris"}));

    assert!(events
        .iter()
        .any(|(e, d)| e == "content_block_stop" && d["index"] == tool_index));
    let (_, md) = events.iter().find(|(e, _)| e == "message_delta").unwrap();
    assert_eq!(md["delta"]["stop_reason"], "tool_use");
}

#[tokio::test(flavor = "multi_thread")]
async fn streaming_thinking_splits_into_thinking_then_text_blocks() {
    let eng = ThinkEchoEngine {
        inner: EchoEngine::new(MODEL, "<think> deep thought </think> final answer"),
    };
    let port = boot_engine(Arc::new(eng)).await;
    let events = stream_events(
        port,
        json!({
            "model": MODEL,
            "max_tokens": 64,
            "stream": true,
            "messages": [{"role": "user", "content": "hi"}],
        }),
    )
    .await;

    let (_, b0) = events
        .iter()
        .find(|(e, _)| e == "content_block_start")
        .unwrap();
    assert_eq!(b0["content_block"]["type"], "thinking");
    assert_eq!(b0["index"], 0);

    let thinking: String = events
        .iter()
        .filter(|(e, d)| e == "content_block_delta" && d["delta"]["type"] == "thinking_delta")
        .map(|(_, d)| d["delta"]["thinking"].as_str().unwrap().to_string())
        .collect();
    assert_eq!(thinking.trim(), "deep thought");

    assert!(events
        .iter()
        .any(|(e, d)| e == "content_block_delta"
            && d["delta"]["type"] == "signature_delta"
            && d["index"] == 0));

    let (_, b1) = events
        .iter()
        .find(|(e, d)| e == "content_block_start" && d["index"] == 1)
        .expect("second content block");
    assert_eq!(b1["content_block"]["type"], "text");
    let text: String = events
        .iter()
        .filter(|(e, d)| e == "content_block_delta" && d["delta"]["type"] == "text_delta")
        .map(|(_, d)| d["delta"]["text"].as_str().unwrap().to_string())
        .collect();
    assert_eq!(text.trim(), "final answer");

    let stops: Vec<&Value> = events
        .iter()
        .filter(|(e, _)| e == "content_block_stop")
        .map(|(_, d)| d)
        .collect();
    assert_eq!(stops.len(), 2);
}

#[tokio::test(flavor = "multi_thread")]
async fn non_streaming_thinking_becomes_a_thinking_block() {
    let eng = ThinkEchoEngine {
        inner: EchoEngine::new(MODEL, "<think> deep thought </think> final answer"),
    };
    let port = boot_engine(Arc::new(eng)).await;
    let (status, v, _) = post_json(
        port,
        "/v1/messages",
        json!({
            "model": MODEL,
            "max_tokens": 64,
            "messages": [{"role": "user", "content": "hi"}],
        }),
    )
    .await;
    assert_eq!(status, reqwest::StatusCode::OK, "body: {v}");
    assert_eq!(v["content"][0]["type"], "thinking");
    assert_eq!(
        v["content"][0]["thinking"].as_str().unwrap().trim(),
        "deep thought"
    );
    assert_eq!(v["content"][0]["signature"], "");
    assert_eq!(v["content"][1]["type"], "text");
    assert_eq!(v["content"][1]["text"].as_str().unwrap().trim(), "final answer");
    assert_eq!(v["stop_reason"], "end_turn");
}

#[tokio::test(flavor = "multi_thread")]
async fn stop_sequences_report_the_matched_sequence() {
    let port = boot_engine(Arc::new(EchoEngine::new(MODEL, "alpha beta STOPHERE gamma"))).await;
    let (status, v, _) = post_json(
        port,
        "/v1/messages",
        json!({
            "model": MODEL,
            "max_tokens": 64,
            "stop_sequences": ["STOPHERE"],
            "messages": [{"role": "user", "content": "hi"}],
        }),
    )
    .await;
    assert_eq!(status, reqwest::StatusCode::OK, "body: {v}");
    assert_eq!(v["stop_reason"], "stop_sequence");
    assert_eq!(v["stop_sequence"], "STOPHERE");
    assert_eq!(v["content"][0]["text"], "alpha beta");
}

#[tokio::test(flavor = "multi_thread")]
async fn streaming_stop_sequences_land_in_message_delta() {
    let port = boot_engine(Arc::new(EchoEngine::new(MODEL, "alpha beta STOPHERE gamma"))).await;
    let events = stream_events(
        port,
        json!({
            "model": MODEL,
            "max_tokens": 64,
            "stream": true,
            "stop_sequences": ["STOPHERE"],
            "messages": [{"role": "user", "content": "hi"}],
        }),
    )
    .await;
    let (_, md) = events.iter().find(|(e, _)| e == "message_delta").unwrap();
    assert_eq!(md["delta"]["stop_reason"], "stop_sequence");
    assert_eq!(md["delta"]["stop_sequence"], "STOPHERE");
}

static TOK_ENV_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

#[tokio::test(flavor = "multi_thread")]
async fn count_tokens_counts_with_a_fixture_tokenizer() {
    let _lock = TOK_ENV_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let base = std::env::temp_dir().join(format!("nv-anthro-tok-{}", std::process::id()));
    let dir = base.join("tok-model");
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(
        dir.join("tokenizer.json"),
        r#"{"version":"1.0","truncation":null,"padding":null,"added_tokens":[],
            "normalizer":null,"pre_tokenizer":{"type":"WhitespaceSplit"},
            "post_processor":null,"decoder":null,
            "model":{"type":"WordLevel","vocab":{"<unk>":0},"unk_token":"<unk>"}}"#,
    )
    .unwrap();
    std::env::set_var("NV_CHAT_MODEL_DIRS", &dir);

    let eng: Arc<dyn ChatEngine> = Arc::new(EchoEngine::new("tok-model", REPLY));
    let app = Router::new()
        .route("/v1/messages/count_tokens", post(handle_count_tokens))
        .with_state(ChatAppState {
            registry: ChatRegistry::single(eng),
        });
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let port = listener.local_addr().unwrap().port();
    tokio::spawn(async move {
        axum::serve(listener, app).await.unwrap();
    });

    let body = json!({
        "model": "tok-model",
        "messages": [{"role": "user", "content": "count these tokens"}],
    });
    let (status, v, _) = post_json(port, "/v1/messages/count_tokens", body.clone()).await;
    assert_eq!(status, reqwest::StatusCode::OK, "body: {v}");
    let n = v["input_tokens"].as_u64().unwrap();
    assert!(n > 0);
    let (_, v2, _) = post_json(port, "/v1/messages/count_tokens", body).await;
    assert_eq!(v2["input_tokens"].as_u64().unwrap(), n);

    std::env::remove_var("NV_CHAT_MODEL_DIRS");
    let _ = std::fs::remove_dir_all(&base);
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
async fn tool_definitions_actually_reach_the_rendered_prompt() {
    let prompt = Arc::new(std::sync::Mutex::new(String::new()));
    let eng = CaptureEngine {
        inner: EchoEngine::new(MODEL, "ok"),
        prompt: prompt.clone(),
    };
    let port = boot_engine(Arc::new(eng)).await;
    let (status, v, _) = post_json(
        port,
        "/v1/messages",
        json!({
            "model": MODEL,
            "max_tokens": 16,
            "tools": [{
                "name": "get_weather",
                "description": "weather lookup",
                "input_schema": {"type": "object", "properties": {"city": {"type": "string"}}},
            }],
            "messages": [{"role": "user", "content": "weather?"}],
        }),
    )
    .await;
    assert_eq!(status, reqwest::StatusCode::OK, "body: {v}");
    let rendered = prompt
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .clone();
    assert!(rendered.contains("get_weather"), "prompt: {rendered:?}");
    assert!(rendered.contains("weather lookup"), "prompt: {rendered:?}");
}

#[derive(Default, Clone)]
struct SeenRequest {
    prompt: String,
    max_new_tokens: usize,
    stop: Vec<String>,
    seed: Option<u64>,
    temperature: Option<f32>,
    top_p: Option<f32>,
    top_k: Option<u32>,
    guided: bool,
    guided_think_close: Option<String>,
    logprobs: bool,
}

struct SpyEngine {
    inner: EchoEngine,
    thinking_split: bool,
    seen: Arc<std::sync::Mutex<SeenRequest>>,
}

#[async_trait::async_trait]
impl ChatEngine for SpyEngine {
    fn model_id(&self) -> &str {
        &self.inner.model_id
    }

    fn thinking_split_supported(&self) -> bool {
        self.thinking_split
    }

    async fn generate(
        &self,
        req: ChatGenerateRequest,
        tx: tokio::sync::mpsc::Sender<ChatEvent>,
    ) -> anyhow::Result<()> {
        *self
            .seen
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner) = SeenRequest {
            prompt: req.prompt.clone(),
            max_new_tokens: req.max_new_tokens,
            stop: req.stop.clone(),
            seed: req.seed,
            temperature: req.temperature,
            top_p: req.top_p,
            top_k: req.top_k,
            guided: req.guided.is_some(),
            guided_think_close: req.guided_think_close.clone(),
            logprobs: req.logprobs,
        };
        self.inner.generate(req, tx).await
    }
}

async fn boot_spy(reply: &str, thinking_split: bool) -> (u16, Arc<std::sync::Mutex<SeenRequest>>) {
    let seen = Arc::new(std::sync::Mutex::new(SeenRequest::default()));
    let eng = SpyEngine {
        inner: EchoEngine::new(MODEL, reply),
        thinking_split,
        seen: seen.clone(),
    };
    (boot_engine(Arc::new(eng)).await, seen)
}

fn seen_of(seen: &Arc<std::sync::Mutex<SeenRequest>>) -> SeenRequest {
    seen.lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .clone()
}

struct ScriptEngine {
    model_id: String,
    script: Vec<ChatEvent>,
}

#[async_trait::async_trait]
impl ChatEngine for ScriptEngine {
    fn model_id(&self) -> &str {
        &self.model_id
    }

    async fn generate(
        &self,
        _req: ChatGenerateRequest,
        tx: tokio::sync::mpsc::Sender<ChatEvent>,
    ) -> anyhow::Result<()> {
        let script = self.script.clone();
        tokio::spawn(async move {
            for ev in script {
                if tx.send(ev).await.is_err() {
                    return;
                }
            }
        });
        Ok(())
    }
}

async fn boot_script(script: Vec<ChatEvent>) -> u16 {
    boot_engine(Arc::new(ScriptEngine {
        model_id: MODEL.to_string(),
        script,
    }))
    .await
}

struct BusyEngine;

#[async_trait::async_trait]
impl ChatEngine for BusyEngine {
    fn model_id(&self) -> &str {
        MODEL
    }

    async fn generate(
        &self,
        _req: ChatGenerateRequest,
        _tx: tokio::sync::mpsc::Sender<ChatEvent>,
    ) -> anyhow::Result<()> {
        Err(anyhow::Error::new(EngineBusy::new(4, 1500)))
    }
}

struct StallEngine;

#[async_trait::async_trait]
impl ChatEngine for StallEngine {
    fn model_id(&self) -> &str {
        MODEL
    }

    async fn generate(
        &self,
        _req: ChatGenerateRequest,
        tx: tokio::sync::mpsc::Sender<ChatEvent>,
    ) -> anyhow::Result<()> {
        tokio::spawn(async move {
            tokio::time::sleep(std::time::Duration::from_secs(30)).await;
            drop(tx);
        });
        Ok(())
    }
}

async fn post_with_headers(
    port: u16,
    path: &str,
    body: Value,
    headers: &[(&'static str, &str)],
) -> (reqwest::StatusCode, Value) {
    let mut b = reqwest::Client::new()
        .post(format!("http://127.0.0.1:{port}{path}"))
        .header("anthropic-version", "2023-06-01");
    for (k, v) in headers {
        b = b.header(*k, *v);
    }
    let resp = b.json(&body).send().await.unwrap();
    let status = resp.status();
    let text = resp.text().await.unwrap();
    (status, serde_json::from_str(&text).unwrap_or(Value::String(text)))
}

async fn post_raw(port: u16, path: &str, body: &'static str) -> (reqwest::StatusCode, Value) {
    let resp = reqwest::Client::new()
        .post(format!("http://127.0.0.1:{port}{path}"))
        .header("content-type", "application/json")
        .body(body)
        .send()
        .await
        .unwrap();
    let status = resp.status();
    let text = resp.text().await.unwrap();
    (status, serde_json::from_str(&text).unwrap_or(Value::String(text)))
}

fn overloaded_status() -> reqwest::StatusCode {
    reqwest::StatusCode::from_u16(OVERLOADED_STATUS).unwrap()
}

#[tokio::test(flavor = "multi_thread")]
async fn an_unknown_model_is_a_not_found_error() {
    let port = boot().await;
    for path in ["/v1/messages", "/v1/messages/count_tokens"] {
        let (status, v, _) = post_json(
            port,
            path,
            json!({
                "model": "no-such-model",
                "max_tokens": 8,
                "messages": [{"role": "user", "content": "hi"}],
            }),
        )
        .await;
        assert_eq!(
            status,
            reqwest::StatusCode::NOT_FOUND,
            "{path} served an unregistered model: {v}"
        );
        assert_eq!(v["error"]["type"], "not_found_error", "{path}: {v}");
        assert!(
            v["error"]["message"]
                .as_str()
                .unwrap()
                .contains("no-such-model"),
            "{path} must name the model it could not resolve: {v}"
        );
        assert!(v["request_id"].as_str().unwrap().starts_with("req_"), "{v}");
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn max_tokens_zero_and_empty_messages_are_rejected() {
    let port = boot().await;
    let (status, v, _) = post_json(
        port,
        "/v1/messages",
        json!({"model": MODEL, "max_tokens": 0, "messages": [{"role": "user", "content": "hi"}]}),
    )
    .await;
    assert_eq!(status, reqwest::StatusCode::BAD_REQUEST, "body: {v}");
    assert_eq!(v["error"]["type"], "invalid_request_error");

    let (status, v, _) = post_json(
        port,
        "/v1/messages",
        json!({"model": MODEL, "max_tokens": 8, "messages": []}),
    )
    .await;
    assert_eq!(status, reqwest::StatusCode::BAD_REQUEST, "body: {v}");
    assert!(
        v["error"]["message"].as_str().unwrap().contains("messages"),
        "body: {v}"
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn a_body_that_is_not_a_messages_request_is_an_invalid_request_error() {
    let port = boot().await;
    let (status, v) = post_raw(port, "/v1/messages", "{\"model\": 7}").await;
    assert_eq!(status, reqwest::StatusCode::BAD_REQUEST, "body: {v}");
    assert_eq!(v["error"]["type"], "invalid_request_error", "body: {v}");
}

#[tokio::test(flavor = "multi_thread")]
async fn tool_choice_naming_an_undeclared_tool_is_rejected() {
    let port = boot().await;
    let (status, v, _) = post_json(
        port,
        "/v1/messages",
        json!({
            "model": MODEL,
            "max_tokens": 8,
            "tools": [{"name": "get_weather", "input_schema": {"type": "object"}}],
            "tool_choice": {"type": "tool", "name": "get_stock_price"},
            "messages": [{"role": "user", "content": "hi"}],
        }),
    )
    .await;
    assert_eq!(status, reqwest::StatusCode::BAD_REQUEST, "body: {v}");
    assert!(
        v["error"]["message"]
            .as_str()
            .unwrap()
            .contains("get_stock_price"),
        "body: {v}"
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn an_unknown_thinking_type_is_rejected() {
    let port = boot().await;
    let (status, v, _) = post_json(
        port,
        "/v1/messages",
        json!({
            "model": MODEL,
            "max_tokens": 8,
            "thinking": {"type": "extended"},
            "messages": [{"role": "user", "content": "hi"}],
        }),
    )
    .await;
    assert_eq!(status, reqwest::StatusCode::BAD_REQUEST, "body: {v}");
    assert!(
        v["error"]["message"].as_str().unwrap().contains("thinking"),
        "body: {v}"
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn tool_choice_none_keeps_the_tools_out_of_the_prompt_and_the_markup_in_the_text() {
    let reply = "<tool_call>{\"name\": \"get_weather\", \"arguments\": {\"city\": \"Paris\"}}</tool_call>";
    let (port, seen) = boot_spy(reply, false).await;
    let (status, v, _) = post_json(
        port,
        "/v1/messages",
        json!({
            "model": MODEL,
            "max_tokens": 32,
            "tools": [{"name": "get_weather", "description": "weather lookup",
                       "input_schema": {"type": "object"}}],
            "tool_choice": {"type": "none"},
            "messages": [{"role": "user", "content": "weather?"}],
        }),
    )
    .await;
    assert_eq!(status, reqwest::StatusCode::OK, "body: {v}");
    let s = seen_of(&seen);
    assert!(
        !s.prompt.contains("get_weather"),
        "tool_choice none must render no tool definitions: {:?}",
        s.prompt
    );
    assert!(!s.guided, "tool_choice none must not constrain decoding");
    assert_eq!(
        v["stop_reason"], "end_turn",
        "with tools inactive nothing may be parsed as a call: {v}"
    );
    assert!(
        v["content"]
            .as_array()
            .unwrap()
            .iter()
            .all(|b| b["type"] != "tool_use"),
        "body: {v}"
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn a_forced_tool_constrains_decoding_and_names_the_call() {
    let (port, seen) = boot_spy("{\"city\": \"Paris\"}", false).await;
    let (status, v, _) = post_json(
        port,
        "/v1/messages",
        json!({
            "model": MODEL,
            "max_tokens": 32,
            "tools": [
                {"name": "get_weather", "input_schema":
                    {"type": "object", "properties": {"city": {"type": "string"}}}},
                {"name": "get_time", "input_schema": {"type": "object"}},
            ],
            "tool_choice": {"type": "tool", "name": "get_weather"},
            "messages": [{"role": "user", "content": "weather?"}],
        }),
    )
    .await;
    assert_eq!(status, reqwest::StatusCode::OK, "body: {v}");
    let s = seen_of(&seen);
    assert!(
        s.guided,
        "tool_choice tool must reach the engine as a grammar built from that tool's input_schema"
    );
    assert_eq!(v["stop_reason"], "tool_use", "body: {v}");
    let block = v["content"]
        .as_array()
        .unwrap()
        .iter()
        .find(|b| b["type"] == "tool_use")
        .unwrap_or_else(|| panic!("no tool_use block: {v}"));
    assert_eq!(block["name"], "get_weather");
    assert_eq!(
        block["input"],
        json!({"city": "Paris"}),
        "a grammar-constrained model emits bare arguments, which must be adopted as the call \
         input: {v}"
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn tool_choice_any_forces_only_when_one_tool_is_declared() {
    let (port, seen) = boot_spy("{\"city\": \"Paris\"}", false).await;
    let (status, v, _) = post_json(
        port,
        "/v1/messages",
        json!({
            "model": MODEL,
            "max_tokens": 32,
            "tools": [{"name": "get_weather", "input_schema":
                {"type": "object", "properties": {"city": {"type": "string"}}}}],
            "tool_choice": {"type": "any"},
            "messages": [{"role": "user", "content": "weather?"}],
        }),
    )
    .await;
    assert_eq!(status, reqwest::StatusCode::OK, "body: {v}");
    assert!(seen_of(&seen).guided, "one tool + any is a forced tool");
    assert_eq!(v["stop_reason"], "tool_use", "body: {v}");

    let (port, seen) = boot_spy("plain prose, no call", false).await;
    let (status, v, _) = post_json(
        port,
        "/v1/messages",
        json!({
            "model": MODEL,
            "max_tokens": 32,
            "tools": [
                {"name": "get_weather", "input_schema": {"type": "object"}},
                {"name": "get_time", "input_schema": {"type": "object"}},
            ],
            "tool_choice": {"type": "any"},
            "messages": [{"role": "user", "content": "weather?"}],
        }),
    )
    .await;
    assert_eq!(status, reqwest::StatusCode::OK, "body: {v}");
    assert!(
        !seen_of(&seen).guided,
        "two tools cannot be expressed as one grammar, so `any` is a prompt instruction only"
    );
    assert_eq!(
        v["stop_reason"], "end_turn",
        "`any` over several tools is unenforced: a model that answers in prose ends the turn: {v}"
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn thinking_plus_a_forced_tool_carries_a_guided_think_close_marker() {
    let (port, seen) = boot_spy("{\"city\": \"Paris\"}", true).await;
    let (status, v, _) = post_json(
        port,
        "/v1/messages",
        json!({
            "model": MODEL,
            "max_tokens": 32,
            "thinking": {"type": "enabled", "budget_tokens": 1024},
            "tools": [{"name": "get_weather", "input_schema":
                {"type": "object", "properties": {"city": {"type": "string"}}}}],
            "tool_choice": {"type": "tool", "name": "get_weather"},
            "messages": [{"role": "user", "content": "weather?"}],
        }),
    )
    .await;
    assert_eq!(status, reqwest::StatusCode::OK, "body: {v}");
    let s = seen_of(&seen);
    assert!(s.guided, "body: {v}");
    assert_eq!(
        s.guided_think_close.as_deref(),
        Some("</think>"),
        "a thinking-split engine under a grammar must be told where the grammar starts applying"
    );

    let (port, seen) = boot_spy("{\"city\": \"Paris\"}", true).await;
    let (status, v, _) = post_json(
        port,
        "/v1/messages",
        json!({
            "model": MODEL,
            "max_tokens": 32,
            "thinking": {"type": "disabled"},
            "tools": [{"name": "get_weather", "input_schema":
                {"type": "object", "properties": {"city": {"type": "string"}}}}],
            "tool_choice": {"type": "tool", "name": "get_weather"},
            "messages": [{"role": "user", "content": "weather?"}],
        }),
    )
    .await;
    assert_eq!(status, reqwest::StatusCode::OK, "body: {v}");
    assert_eq!(
        seen_of(&seen).guided_think_close, None,
        "thinking disabled leaves no think block for the grammar to skip"
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn sampling_fields_and_the_max_tokens_ceiling_reach_the_engine() {
    let (port, seen) = boot_spy(REPLY, false).await;
    let (status, v, _) = post_json(
        port,
        "/v1/messages",
        json!({
            "model": MODEL,
            "max_tokens": MAX_MAX_TOKENS as u64 * 4,
            "temperature": 0.25,
            "top_p": 0.8,
            "top_k": 40,
            "stop_sequences": ["STOPHERE", "ALSO"],
            "messages": [{"role": "user", "content": "hi"}],
        }),
    )
    .await;
    assert_eq!(status, reqwest::StatusCode::OK, "body: {v}");
    let s = seen_of(&seen);
    assert_eq!(
        s.max_new_tokens, MAX_MAX_TOKENS,
        "max_tokens above the surface ceiling is clamped, not refused"
    );
    assert_eq!(s.temperature, Some(0.25));
    assert_eq!(s.top_p, Some(0.8));
    assert_eq!(s.top_k, Some(40));
    assert_eq!(s.stop, vec!["STOPHERE".to_string(), "ALSO".to_string()]);
    assert!(
        s.seed.is_some(),
        "the anthropic surface exposes no seed field, so every request must draw its own"
    );
    assert!(!s.logprobs, "the anthropic surface has no logprobs field");
}

#[tokio::test(flavor = "multi_thread")]
async fn a_tool_result_turn_reaches_the_rendered_prompt() {
    let (port, seen) = boot_spy("ok", false).await;
    let (status, v, _) = post_json(
        port,
        "/v1/messages",
        json!({
            "model": MODEL,
            "max_tokens": 16,
            "system": "be terse",
            "tools": [{"name": "get_weather", "description": "weather lookup",
                       "input_schema": {"type": "object",
                                        "properties": {"city": {"type": "string"}}}}],
            "messages": [
                {"role": "user", "content": "weather?"},
                {"role": "assistant", "content": [
                    {"type": "thinking", "thinking": "hidden", "signature": "sig"},
                    {"type": "text", "text": "checking"},
                    {"type": "tool_use", "id": "toolu_1", "name": "get_weather",
                     "input": {"city": "Paris"}},
                ]},
                {"role": "user", "content": [
                    {"type": "tool_result", "tool_use_id": "toolu_1", "content": "72F"},
                ]},
            ],
        }),
    )
    .await;
    assert_eq!(status, reqwest::StatusCode::OK, "body: {v}");
    let p = seen_of(&seen).prompt;
    for needle in ["be terse", "checking", "Paris", "72F"] {
        assert!(
            p.contains(needle),
            "the rendered prompt dropped {needle:?}: {p:?}"
        );
    }
    assert!(
        !p.contains("hidden"),
        "assistant thinking blocks are dropped on the way in, not replayed: {p:?}"
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn an_error_tool_result_is_marked_for_the_model() {
    let (port, seen) = boot_spy("ok", false).await;
    let (status, v, _) = post_json(
        port,
        "/v1/messages",
        json!({
            "model": MODEL,
            "max_tokens": 16,
            "messages": [
                {"role": "user", "content": "weather?"},
                {"role": "assistant", "content": [
                    {"type": "tool_use", "id": "toolu_1", "name": "get_weather", "input": {}},
                ]},
                {"role": "user", "content": [
                    {"type": "tool_result", "tool_use_id": "toolu_1",
                     "content": "network unreachable", "is_error": true},
                ]},
            ],
        }),
    )
    .await;
    assert_eq!(status, reqwest::StatusCode::OK, "body: {v}");
    let p = seen_of(&seen).prompt;
    assert!(
        p.contains("Error: network unreachable"),
        "is_error must survive into the prompt or the model reads a failure as a result: {p:?}"
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn a_restored_prefix_is_reported_as_cache_read_input_tokens() {
    let port = boot_script(vec![
        ChatEvent::PromptCached { cached_tokens: 7 },
        ChatEvent::Started { prompt_tokens: 11 },
        ChatEvent::TextDelta("warm".into()),
        ChatEvent::Done {
            finish_reason: "stop".into(),
            completion_tokens: 1,
        },
    ])
    .await;
    let (status, v, _) = post_json(
        port,
        "/v1/messages",
        json!({"model": MODEL, "max_tokens": 8, "messages": [{"role": "user", "content": "hi"}]}),
    )
    .await;
    assert_eq!(status, reqwest::StatusCode::OK, "body: {v}");
    assert_eq!(v["usage"]["input_tokens"], 11, "body: {v}");
    assert_eq!(
        v["usage"]["cache_read_input_tokens"], 7,
        "tokens served out of a rewound (in-memory or disk-restored) kv prefix are the only \
         cache figure this surface reports: {v}"
    );
    assert_eq!(
        v["usage"]["cache_creation_input_tokens"], 0,
        "this surface never bills cache creation: {v}"
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn streaming_message_start_carries_a_cache_figure_emitted_before_started() {
    let port = boot_script(vec![
        ChatEvent::PromptCached { cached_tokens: 7 },
        ChatEvent::Started { prompt_tokens: 11 },
        ChatEvent::TextDelta("warm".into()),
        ChatEvent::Done {
            finish_reason: "stop".into(),
            completion_tokens: 1,
        },
    ])
    .await;
    let events = stream_events(
        port,
        json!({"model": MODEL, "max_tokens": 8, "stream": true,
               "messages": [{"role": "user", "content": "hi"}]}),
    )
    .await;
    let (_, start) = &events[0];
    assert_eq!(start["message"]["usage"]["input_tokens"], 11, "{start}");
    assert_eq!(
        start["message"]["usage"]["cache_read_input_tokens"], 7,
        "PromptCached is emitted before Started so the figure is known by message_start: {start}"
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn a_cache_figure_emitted_after_started_still_reaches_message_delta() {
    let port = boot_script(vec![
        ChatEvent::Started { prompt_tokens: 11 },
        ChatEvent::PromptCached { cached_tokens: 7 },
        ChatEvent::TextDelta("warm".into()),
        ChatEvent::Done {
            finish_reason: "stop".into(),
            completion_tokens: 1,
        },
    ])
    .await;
    let events = stream_events(
        port,
        json!({"model": MODEL, "max_tokens": 8, "stream": true,
               "messages": [{"role": "user", "content": "hi"}]}),
    )
    .await;
    let (_, start) = &events[0];
    assert_eq!(
        start["message"]["usage"]["cache_read_input_tokens"], 0,
        "message_start is flushed as soon as Started arrives, so a later figure cannot be in \
         it: {start}"
    );
    let (_, md) = events.iter().find(|(e, _)| e == "message_delta").unwrap();
    assert_eq!(
        md["usage"]["cache_read_input_tokens"], 7,
        "message_delta is the backstop: an out-of-order cache figure is late, never lost: {md}"
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn an_engine_that_will_not_start_sheds_with_the_overloaded_status() {
    let port = boot_engine(Arc::new(BusyEngine)).await;
    for stream in [false, true] {
        let (status, v) = post_with_headers(
            port,
            "/v1/messages",
            json!({"model": MODEL, "max_tokens": 8, "stream": stream,
                   "messages": [{"role": "user", "content": "hi"}]}),
            &[],
        )
        .await;
        assert_eq!(
            status,
            overloaded_status(),
            "stream={stream} must shed with {OVERLOADED_STATUS}, not a half-open stream: {v}"
        );
        assert_eq!(v["error"]["type"], "overloaded_error", "{v}");
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn a_busy_shed_arriving_as_an_event_is_also_the_overloaded_status() {
    let port = boot_script(vec![ChatEvent::Error(format!(
        "{ENGINE_BUSY_PREFIX} no slot"
    ))])
    .await;
    let (status, v, _) = post_json(
        port,
        "/v1/messages",
        json!({"model": MODEL, "max_tokens": 8, "messages": [{"role": "user", "content": "hi"}]}),
    )
    .await;
    assert_eq!(status, overloaded_status(), "body: {v}");
    assert_eq!(v["error"]["type"], "overloaded_error", "{v}");
}

#[tokio::test(flavor = "multi_thread")]
async fn a_plain_engine_error_is_an_api_error() {
    let port = boot_script(vec![ChatEvent::Error("decoder exploded".into())]).await;
    let (status, v, _) = post_json(
        port,
        "/v1/messages",
        json!({"model": MODEL, "max_tokens": 8, "messages": [{"role": "user", "content": "hi"}]}),
    )
    .await;
    assert_eq!(
        status,
        reqwest::StatusCode::INTERNAL_SERVER_ERROR,
        "body: {v}"
    );
    assert_eq!(v["error"]["type"], "api_error", "{v}");
}

#[tokio::test(flavor = "multi_thread")]
async fn an_error_after_the_stream_opened_becomes_a_terminal_sse_error_event() {
    let port = boot_script(vec![
        ChatEvent::Started { prompt_tokens: 3 },
        ChatEvent::TextDelta("half an ".into()),
        ChatEvent::Error("decoder exploded".into()),
    ])
    .await;
    let events = stream_events(
        port,
        json!({"model": MODEL, "max_tokens": 8, "stream": true,
               "messages": [{"role": "user", "content": "hi"}]}),
    )
    .await;
    let names: Vec<&str> = events.iter().map(|(e, _)| e.as_str()).collect();
    assert_eq!(
        *names.last().unwrap(),
        "error",
        "a mid-stream failure ends the stream with an error frame: {names:?}"
    );
    assert!(
        !names.contains(&"message_stop"),
        "a failed generation must not also claim to have stopped normally: {names:?}"
    );
    let (_, err) = events.last().unwrap();
    assert_eq!(err["error"]["type"], "api_error", "{err}");
    assert!(
        err["error"]["message"]
            .as_str()
            .unwrap()
            .contains("decoder exploded"),
        "{err}"
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn a_client_deadline_that_expires_before_the_first_event_sheds() {
    let port = boot_engine(Arc::new(StallEngine)).await;
    let (status, v) = post_with_headers(
        port,
        "/v1/messages",
        json!({"model": MODEL, "max_tokens": 8, "messages": [{"role": "user", "content": "hi"}]}),
        &[("x-request-timeout-ms", "50")],
    )
    .await;
    assert_eq!(status, overloaded_status(), "body: {v}");
    assert_eq!(v["error"]["type"], "overloaded_error", "{v}");
    assert!(
        v["error"]["message"]
            .as_str()
            .unwrap()
            .contains("x-request-timeout-ms"),
        "the shed must name the deadline source the caller set: {v}"
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn a_conversation_whose_every_turn_is_empty_is_rejected() {
    let port = boot().await;
    for content in [json!(""), json!([]), json!([{"type": "text", "text": ""}])] {
        let (status, v, _) = post_json(
            port,
            "/v1/messages",
            json!({
                "model": MODEL,
                "max_tokens": 16,
                "system": "be terse",
                "messages": [{"role": "user", "content": content}],
            }),
        )
        .await;
        assert_eq!(
            status,
            reqwest::StatusCode::BAD_REQUEST,
            "content {content} produced a promptless generation instead of an error: {v}"
        );
        assert_eq!(v["error"]["type"], "invalid_request_error", "{v}");
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn count_tokens_rejects_an_empty_conversation_the_same_way() {
    let port = boot().await;
    let (status, v, _) = post_json(
        port,
        "/v1/messages/count_tokens",
        json!({"model": MODEL, "messages": [{"role": "user", "content": ""}]}),
    )
    .await;
    assert_eq!(status, reqwest::StatusCode::BAD_REQUEST, "body: {v}");
    assert_eq!(v["error"]["type"], "invalid_request_error", "{v}");
}

#[tokio::test(flavor = "multi_thread")]
async fn a_user_turn_carrying_only_a_tool_result_is_not_empty() {
    let (port, seen) = boot_spy("ok", false).await;
    let (status, v, _) = post_json(
        port,
        "/v1/messages",
        json!({
            "model": MODEL,
            "max_tokens": 16,
            "messages": [{"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "toolu_1", "content": "72F"},
            ]}],
        }),
    )
    .await;
    assert_eq!(
        status,
        reqwest::StatusCode::OK,
        "a tool_result is content: the empty-turn guard must not swallow it: {v}"
    );
    assert!(seen_of(&seen).prompt.contains("72F"), "body: {v}");
}
