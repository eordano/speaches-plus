#![cfg(feature = "cuda")]

use std::path::{Path, PathBuf};
use std::sync::Arc;

use axum::routing::post;
use axum::Router;
use serde_json::{json, Value};

use speaches_plus::oapi::chat::{handle_chat_completions, ChatAppState, ChatEngine};
use speaches_plus::oapi::chat_engine::{ChatRegistry, NvEngineChat};

fn resolve_model_dir() -> Option<PathBuf> {
    if let Ok(d) = std::env::var("NV_CHAT_MODEL_DIR") {
        let p = PathBuf::from(d);
        if p.is_dir() {
            return Some(p);
        }
    }
    let home = std::env::var("HOME").ok()?;
    let root = PathBuf::from(home)
        .join(".cache/huggingface/hub/models--nvidia--Gemma-4-31B-IT-NVFP4/snapshots");
    std::fs::read_dir(&root)
        .ok()?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .find(|p| p.join("config.json").is_file())
}

async fn boot() -> (u16, String) {
    let Some(dir) = resolve_model_dir() else {
        panic!(
            "PRECONDITION NOT MET, THIS TEST EXECUTED NOTHING: no Gemma-4 NVFP4 snapshot \
             (set NV_CHAT_MODEL_DIR)"
        )
    };
    let eng: Arc<dyn ChatEngine> = match NvEngineChat::try_load(Path::new(&dir)) {
        Ok(e) => Arc::new(e),
        Err(err) => panic!(
            "NvEngineChat::try_load({}) failed: {err:#}. The checkpoint is present, so this \
             is a FAILURE, not a skip.",
            dir.display()
        ),
    };
    let model_id = eng.model_id().to_string();
    let app = Router::new()
        .route("/v1/chat/completions", post(handle_chat_completions))
        .with_state(ChatAppState {
            registry: ChatRegistry::single(eng),
        });
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let port = listener.local_addr().unwrap().port();
    tokio::spawn(async move {
        let _ = axum::serve(listener, app).await;
    });
    (port, model_id)
}

async fn post_json(port: u16, body: Value) -> (reqwest::StatusCode, Value) {
    let resp = reqwest::Client::new()
        .post(format!("http://127.0.0.1:{port}/v1/chat/completions"))
        .json(&body)
        .send()
        .await
        .expect("request");
    let status = resp.status();
    let v: Value = resp.json().await.expect("json body");
    (status, v)
}

fn weather_tool() -> Value {
    json!({
        "type": "function",
        "function": {
            "name": "get_current_weather",
            "description": "Get the current weather in a given city.",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "City name"}
                },
                "required": ["city"]
            }
        }
    })
}

fn gate() {
    if std::env::var("NV_TOOL_E2E").as_deref() != Ok("1") {
        panic!("PRECONDITION NOT MET, THIS TEST EXECUTED NOTHING: set NV_TOOL_E2E=1");
    }
}

#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn the_server_returns_a_tool_call_for_a_prompt_that_needs_one() {
    gate();
    let (port, model) = boot().await;
    let (status, v) = post_json(
        port,
        json!({
            "model": model,
            "messages": [
                {"role": "user",
                 "content": "What is the weather in Oslo right now? Use the tool."}
            ],
            "tools": [weather_tool()],
            "temperature": 0.0,
            "max_tokens": 128
        }),
    )
    .await;
    assert_eq!(status, reqwest::StatusCode::OK, "body: {v}");

    let msg = &v["choices"][0]["message"];
    let calls = msg["tool_calls"].as_array();
    assert!(
        calls.is_some_and(|c| !c.is_empty()),
        "no tool_calls in the response. content was {:?}, finish_reason {:?}. If the content \
         contains `call:` then the delimiters were stripped again and the wire tokens are not \
         reaching the parser.",
        msg["content"],
        v["choices"][0]["finish_reason"]
    );
    let call = &calls.unwrap()[0];
    assert_eq!(call["type"], "function");
    assert_eq!(
        call["function"]["name"], "get_current_weather",
        "wrong tool: {call}"
    );

    let args: Value = serde_json::from_str(
        call["function"]["arguments"]
            .as_str()
            .expect("arguments is a JSON string"),
    )
    .expect("arguments parse as JSON");
    assert!(
        args["city"]
            .as_str()
            .is_some_and(|c| c.to_lowercase().contains("oslo")),
        "arguments did not carry the city: {args}"
    );

    let content = msg["content"].as_str().unwrap_or_default();
    for marker in ["<|tool_call>", "<tool_call|>", "<|\"|>", "call:"] {
        assert!(
            !content.contains(marker),
            "{marker:?} leaked into content: {content:?}"
        );
    }
    assert_eq!(v["choices"][0]["finish_reason"], "tool_calls");
}

#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn without_tools_the_same_prompt_answers_in_prose() {
    gate();
    let (port, model) = boot().await;
    let (status, v) = post_json(
        port,
        json!({
            "model": model,
            "messages": [
                {"role": "user",
                 "content": "What is the weather in Oslo right now? Use the tool."}
            ],
            "temperature": 0.0,
            "max_tokens": 64
        }),
    )
    .await;
    assert_eq!(status, reqwest::StatusCode::OK, "body: {v}");
    let msg = &v["choices"][0]["message"];
    assert!(
        msg["tool_calls"].as_array().is_none_or(|c| c.is_empty()),
        "a tool call with no tools offered: {msg}"
    );
    let content = msg["content"].as_str().unwrap_or_default();
    assert!(!content.trim().is_empty(), "empty prose reply: {msg}");
}
