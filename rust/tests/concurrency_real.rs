#![cfg(feature = "cuda")]

use std::path::{Path, PathBuf};
use std::sync::Arc;

use axum::routing::post;
use axum::Router;
use serde_json::{json, Value};

use speaches_plus::oapi::chat::{handle_chat_completions, ChatAppState, ChatEngine};
use speaches_plus::oapi::chat_engine::{ChatRegistry, NvEngineChat};
use speaches_plus::oapi::completions::handle_completions;

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
            "no Gemma-4 NVFP4 snapshot found (set NV_CHAT_MODEL_DIR). boot() used to return \
             None here and every caller returned, printing `passed` in 0.00s. This is a SKIP, \
             not a pass."
        )
    };
    let eng: Arc<dyn ChatEngine> = match NvEngineChat::try_load(Path::new(&dir)) {
        Ok(e) => Arc::new(e),
        Err(err) => panic!(
            "NvEngineChat::try_load({}) failed: {err:#}. The checkpoint is present, so this is a \
             FAILURE, not a skip.",
            dir.display()
        ),
    };
    let model_id = eng.model_id().to_string();
    let app = Router::new()
        .route("/v1/chat/completions", post(handle_chat_completions))
        .route("/v1/completions", post(handle_completions))
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
        .unwrap();
    let status = resp.status();
    let v: Value = resp.json().await.unwrap_or(Value::Null);
    (status, v)
}

fn content(v: &Value) -> String {
    v["choices"][0]["message"]["content"]
        .as_str()
        .unwrap_or("")
        .to_string()
}

fn require_gate(test: &str) {
    if std::env::var("NV_TOOLS_REAL_TEST").is_err() {
        panic!(
            "{test} is #[ignore]d, so it was asked for BY NAME, but NV_TOOLS_REAL_TEST=1 is not \
             set, so it would have exercised nothing. This is a SKIP, not a pass."
        );
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 6)]
#[ignore = "real weights on the GPU: set NV_TOOLS_REAL_TEST=1 and run by name"]
async fn concurrent_distinct_prompts_no_cross_bleed() {
    require_gate("concurrent_distinct_prompts_no_cross_bleed");
    let (port, model) = boot().await;

    let cases = [
        ("What is 2+2? Reply with only the number.", "4"),
        ("What is the capital of France? One word.", "paris"),
        ("What color is grass? One word.", "green"),
        ("What is 10 multiplied by 10? Only the number.", "100"),
        ("What is the capital of Japan? One word.", "tokyo"),
        ("What is 5 plus 3? Only the number.", "8"),
    ];

    let handles: Vec<_> = cases
        .iter()
        .map(|(prompt, _)| {
            let body = json!({
                "model": model,
                "messages": [{"role": "user", "content": *prompt}],
                "max_tokens": 24,
                "temperature": 0.0
            });
            tokio::spawn(async move { post_json(port, body).await })
        })
        .collect();

    let mut ok = 0;
    for (i, h) in handles.into_iter().enumerate() {
        let (status, v) = h.await.unwrap();
        let c = content(&v).to_lowercase();
        let (_p, needle) = cases[i];
        eprintln!(
            "[{i}] status={status} needle={needle:?} content={:?}",
            content(&v)
        );
        assert_eq!(status, reqwest::StatusCode::OK, "case {i} status");
        assert!(
            c.contains(needle),
            "case {i}: expected {needle:?} in {c:?} (cross-request bleed?)"
        );
        ok += 1;
    }
    eprintln!("concurrent distinct prompts: {ok}/{} correct", cases.len());
}

#[tokio::test(flavor = "multi_thread", worker_threads = 6)]
#[ignore = "real weights on the GPU: set NV_TOOLS_REAL_TEST=1 and run by name"]
async fn concurrent_identical_greedy_is_deterministic() {
    require_gate("concurrent_identical_greedy_is_deterministic");
    let (port, model) = boot().await;

    let n = 6;
    let handles: Vec<_> = (0..n)
        .map(|_| {
            let body = json!({
                "model": model,
                "messages": [{"role": "user", "content": "List three primary colors, comma-separated."}],
                "max_tokens": 32,
                "temperature": 0.0,
                "seed": 42
            });
            tokio::spawn(async move { post_json(port, body).await })
        })
        .collect();

    let mut outputs = Vec::new();
    for h in handles {
        let (status, v) = h.await.unwrap();
        assert_eq!(status, reqwest::StatusCode::OK);
        outputs.push(content(&v));
    }
    eprintln!("identical greedy outputs:");
    for (i, o) in outputs.iter().enumerate() {
        eprintln!("  [{i}] {o:?}");
    }
    let first = &outputs[0];
    assert!(!first.is_empty(), "empty output");

    if std::env::var("NV_BATCH_ENGINE").is_ok() {
        for (i, o) in outputs.iter().enumerate() {
            let lo = o.to_lowercase();
            assert!(
                lo.contains("red") || lo.contains("blue") || lo.contains("yellow"),
                "batched output {i} lost its topic (primary colors): {o:?}"
            );
        }
    } else {
        for (i, o) in outputs.iter().enumerate() {
            assert_eq!(o, first, "output {i} diverged from #0 under concurrency");
        }
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 6)]
#[ignore = "real weights on the GPU: set NV_TOOLS_REAL_TEST=1 and run by name"]
async fn concurrent_mixed_workload() {
    require_gate("concurrent_mixed_workload");
    let (port, model) = boot().await;

    let plain = json!({
        "model": model,
        "messages": [{"role":"user","content":"Say hello in one word."}],
        "max_tokens": 16, "temperature": 0.0
    });
    let guided = json!({
        "model": model,
        "messages": [{"role":"user","content":"Give me a person record for Bob, age 40."}],
        "response_format": {"type":"json_schema","json_schema":{"name":"p","schema":{
            "type":"object","properties":{"name":{"type":"string"},"age":{"type":"integer"}},
            "required":["name","age"]}}},
        "max_tokens": 64, "temperature": 0.0
    });
    let tools = json!({
        "model": model,
        "messages": [{"role":"user","content":"What's the weather in Berlin? Use the tool."}],
        "tools": [{"type":"function","function":{"name":"get_weather","description":"weather",
            "parameters":{"type":"object","properties":{"location":{"type":"string"}},"required":["location"]}}}],
        "tool_choice": "auto",
        "max_tokens": 64, "temperature": 0.0
    });

    let (hp, hg, ht) = (
        tokio::spawn(async move { post_json(port, plain).await }),
        tokio::spawn(async move { post_json(port, guided).await }),
        tokio::spawn(async move { post_json(port, tools).await }),
    );
    let (sp, vp) = hp.await.unwrap();
    let (sg, vg) = hg.await.unwrap();
    let (st, vt) = ht.await.unwrap();

    eprintln!("plain  -> {sp} {:?}", content(&vp));
    eprintln!("guided -> {sg} {:?}", content(&vg));
    eprintln!(
        "tools  -> {st} tool_calls={}",
        vt["choices"][0]["message"]["tool_calls"]
    );

    assert_eq!(sp, reqwest::StatusCode::OK);
    assert!(!content(&vp).is_empty());

    assert_eq!(sg, reqwest::StatusCode::OK);
    let parsed: Value = serde_json::from_str(&content(&vg)).expect("guided output is valid JSON");
    assert!(
        parsed.get("name").is_some() && parsed.get("age").is_some(),
        "schema held: {parsed}"
    );

    assert_eq!(st, reqwest::StatusCode::OK);
    assert_eq!(
        vt["choices"][0]["message"]["tool_calls"][0]["function"]["name"],
        "get_weather"
    );
}
