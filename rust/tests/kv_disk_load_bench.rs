#![cfg(feature = "wgpu")]

use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

use axum::body::{to_bytes, Body};
use axum::http::{Request, StatusCode};
use axum::routing::post;
use axum::Router;
use tower::ServiceExt;

use speaches_plus::oapi::chat::{handle_chat_completions, ChatAppState, ChatEngine};
use speaches_plus::oapi::chat_engine::ChatRegistry;
use speaches_plus::oapi::chat_engine_wgpu::persist::KV_CACHE_DIR_ENV;
use speaches_plus::oapi::chat_engine_wgpu::{WgpuChatEngine, PREFIX_REUSE_ENV};

const SIZES: [usize; 4] = [8192, 16384, 49152, 131072];

fn sizes() -> Vec<usize> {
    match std::env::var("NV_KV_BENCH_SIZES") {
        Ok(v) => v
            .split(',')
            .filter_map(|s| s.trim().parse().ok())
            .filter(|n| *n > 0)
            .collect(),
        Err(_) => SIZES.to_vec(),
    }
}

fn model_dir() -> Option<PathBuf> {
    let d = std::env::var("NV_WGPU_SERVE_DIR").ok()?;
    let p = PathBuf::from(d);
    p.join("config.json").exists().then_some(p)
}

async fn post_json(app: &Router, body: String) -> (StatusCode, String) {
    let req = Request::builder()
        .method("POST")
        .uri("/v1/chat/completions")
        .header("content-type", "application/json")
        .body(Body::from(body))
        .unwrap();
    let resp = app.clone().oneshot(req).await.unwrap();
    let status = resp.status();
    let bytes = to_bytes(resp.into_body(), 1 << 22).await.unwrap();
    (status, String::from_utf8_lossy(&bytes).into_owned())
}

const TURNS: [&str; 3] = [
    "Count to five, digits only.",
    "Now count to three.",
    "Now count to two.",
];

fn max_tokens() -> u32 {
    std::env::var("NV_KV_BENCH_MAX_TOKENS")
        .ok()
        .and_then(|v| v.parse().ok())
        .filter(|v| *v > 0)
        .unwrap_or(8)
}

async fn serve(
    app: &Router,
    model: &str,
    label: &str,
    conversation: &[serde_json::Value],
) -> (u64, String) {
    let body = serde_json::json!({
        "model": model, "temperature": 0, "max_tokens": max_tokens(), "seed": 7,
        "messages": conversation,
    });
    let (status, resp) = post_json(app, body.to_string()).await;
    assert_eq!(status, StatusCode::OK, "[{label}] {resp}");
    let v: serde_json::Value = serde_json::from_str(&resp).unwrap();
    assert!(
        v["usage"]["completion_tokens"].as_u64().unwrap_or(0) > 0,
        "[{label}] no tokens generated"
    );
    let msg = &v["choices"][0]["message"];
    let content = msg["content"].as_str().unwrap_or_default();
    let reply = if content.is_empty() {
        msg["reasoning_content"].as_str().unwrap_or_default()
    } else {
        content
    };
    assert!(!reply.is_empty(), "[{label}] empty completion");
    (
        v["usage"]["prompt_tokens_details"]["cached_tokens"]
            .as_u64()
            .unwrap_or(0),
        reply.to_string(),
    )
}

fn user(text: &str) -> serde_json::Value {
    serde_json::json!({"role": "user", "content": text})
}

fn assistant(text: &str) -> serde_json::Value {
    serde_json::json!({"role": "assistant", "content": text})
}

struct Row {
    max_seq: usize,
    file_mb: f64,
    save_s: f64,
    reload_s: f64,
    cached_live: u64,
    cached_reload: u64,
}

#[test]
#[ignore]
fn kv_disk_save_and_restore_across_context_capacities() {
    if std::env::var("NV_KV_BENCH").as_deref() != Ok("1") {
        panic!("PRECONDITION NOT MET, THIS TEST EXECUTED NOTHING: set NV_KV_BENCH=1");
    }
    let Some(dir) = model_dir() else {
        panic!(
            "no wgpu servable model dir: set NV_WGPU_SERVE_DIR. This is a SKIP, not a pass."
        )
    };
    let _ = tracing_subscriber::fmt()
        .with_env_filter("speaches_plus=info")
        .with_writer(std::io::stderr)
        .try_init();
    std::env::set_var(PREFIX_REUSE_ENV, "1");

    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    let mut rows: Vec<Row> = Vec::new();
    let mut failures: Vec<String> = Vec::new();

    for max_seq in sizes() {
        let cache_dir =
            std::env::temp_dir().join(format!("nvkv-bench-{}-{max_seq}", std::process::id()));
        let _ = std::fs::remove_dir_all(&cache_dir);
        std::fs::create_dir_all(&cache_dir).unwrap();
        std::env::set_var(KV_CACHE_DIR_ENV, &cache_dir);

        let engine = match WgpuChatEngine::load_with(&dir, max_seq, None) {
            Ok(e) => Arc::new(e),
            Err(err) => {
                failures.push(format!("max_seq {max_seq}: cold load failed: {err:#}"));
                continue;
            }
        };
        let model = engine.model_id().to_string();
        let app = Router::new()
            .route("/v1/chat/completions", post(handle_chat_completions))
            .with_state(ChatAppState {
                registry: ChatRegistry::single(engine.clone() as Arc<dyn ChatEngine>),
            });
        let mut conv = vec![user(TURNS[0])];
        let (_, reply1) = rt.block_on(serve(&app, &model, &format!("warmup-{max_seq}"), &conv));
        conv.push(assistant(&reply1));
        conv.push(user(TURNS[1]));
        let (cached_live, reply2) =
            rt.block_on(serve(&app, &model, &format!("live-cont-{max_seq}"), &conv));
        conv.push(assistant(&reply2));
        conv.push(user(TURNS[2]));
        drop(app);
        let t_save = Instant::now();
        drop(engine);
        let save_s = t_save.elapsed().as_secs_f64();

        let file_mb = std::fs::read_dir(&cache_dir)
            .unwrap()
            .flatten()
            .filter_map(|e| e.metadata().ok())
            .map(|m| m.len())
            .sum::<u64>() as f64
            / (1024.0 * 1024.0);

        let t_reload = Instant::now();
        let engine = match WgpuChatEngine::load_with(&dir, max_seq, None) {
            Ok(e) => Arc::new(e),
            Err(err) => {
                failures.push(format!("max_seq {max_seq}: warm load failed: {err:#}"));
                continue;
            }
        };
        let app = Router::new()
            .route("/v1/chat/completions", post(handle_chat_completions))
            .with_state(ChatAppState {
                registry: ChatRegistry::single(engine.clone() as Arc<dyn ChatEngine>),
            });
        let (cached_reload, _) = rt.block_on(serve(
            &app,
            &model,
            &format!("restored-{max_seq}"),
            &conv,
        ));
        let reload_s = t_reload.elapsed().as_secs_f64();
        drop(app);
        std::env::remove_var(KV_CACHE_DIR_ENV);
        drop(engine);
        let _ = std::fs::remove_dir_all(&cache_dir);

        rows.push(Row {
            max_seq,
            file_mb,
            save_s,
            reload_s,
            cached_live,
            cached_reload,
        });
    }
    std::env::remove_var(PREFIX_REUSE_ENV);

    eprintln!("[kv-bench] model dir: {}", dir.display());
    eprintln!(
        "[kv-bench] max_seq | snapshot MB | drop+save s | reload+serve s | cached live | cached reload"
    );
    for r in &rows {
        eprintln!(
            "[kv-bench] {:>7} | {:>11.1} | {:>11.2} | {:>14.2} | {:>11} | {:>13}",
            r.max_seq, r.file_mb, r.save_s, r.reload_s, r.cached_live, r.cached_reload
        );
    }
    eprintln!(
        "[kv-bench] restore_ms / download_ms / write_ms per size are in the tracing lines above; \
         cached live == 0 means the chat template does not re-render history to the folded \
         stream, which caps every reuse path, not just the disk one"
    );
    for r in &rows {
        assert!(
            r.cached_live == 0 || r.cached_reload > 0,
            "max_seq {}: the live path served {} warm tokens but the restored snapshot served \
             none; the disk restore lost reuse the template permits",
            r.max_seq,
            r.cached_live
        );
        assert!(r.file_mb > 0.05, "max_seq {}: no snapshot written", r.max_seq);
    }
    assert!(failures.is_empty(), "some sizes failed:\n{}", failures.join("\n"));
}
