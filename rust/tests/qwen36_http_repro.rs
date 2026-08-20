
#[cfg(not(feature = "cuda"))]
#[test]
fn qwen36_http_repro_is_cfg_out_without_cuda() {
    eprintln!("qwen36_http_repro compiled OUT (no `cuda` feature). SKIP, not a pass.");
}

#[cfg(feature = "cuda")]
mod gated {
    use std::path::PathBuf;
    use std::sync::Arc;

    use axum::body::{to_bytes, Body};
    use axum::http::{Request, StatusCode};
    use axum::routing::post;
    use axum::Router;
    use tower::ServiceExt;

    use speaches_plus::oapi::chat::{handle_chat_completions, ChatAppState, ChatEngine};
    use speaches_plus::oapi::chat_engine::{ChatRegistry, NvEngineChat};

    const RUNS: usize = 5;

    fn snapshot_under(root: PathBuf) -> Option<PathBuf> {
        let snaps = root
            .join("models--RedHatAI--Qwen3.6-35B-A3B-NVFP4")
            .join("snapshots");
        let mut cand: Vec<PathBuf> = std::fs::read_dir(&snaps)
            .ok()?
            .flatten()
            .map(|e| e.path())
            .filter(|p| p.join("config.json").exists())
            .collect();
        cand.sort();
        cand.pop()
    }

    fn snapshot_dir() -> Option<PathBuf> {
        if let Some(hit) = std::env::var_os("HF_HUB_CACHE")
            .map(PathBuf::from)
            .and_then(snapshot_under)
        {
            return Some(hit);
        }
        std::env::var_os("HOME")
            .map(|h| PathBuf::from(h).join(".cache/huggingface/hub"))
            .and_then(snapshot_under)
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

    #[test]
    #[ignore]
    fn the_default_serving_path_answers_paris_identically_five_times() {
        if std::env::var("NV_QWEN36_HTTP").as_deref() != Ok("1") {
            panic!(
                "PRECONDITION NOT MET, THIS TEST EXECUTED NOTHING: set NV_QWEN36_HTTP=1 \
                 (loads the 35B onto the GPU)"
            );
        }
        let Some(dir) = snapshot_dir() else {
            if std::env::var("NV_QWEN36_ALLOW_SKIP").as_deref() == Ok("1") {
                eprintln!("SKIP (NV_QWEN36_ALLOW_SKIP=1): 35B snapshot absent");
                return;
            }
            panic!(
                "Qwen3.6-35B snapshot not found under the HF hub cache. This is #63's \
                 shipping-decider; it refuses to report success without running. Set \
                 NV_QWEN36_ALLOW_SKIP=1 to skip on purpose."
            );
        };

        let t0 = std::time::Instant::now();
        let engine = Arc::new(
            NvEngineChat::try_load(&dir).expect("NvEngineChat must load the 35B on default config"),
        ) as Arc<dyn ChatEngine>;
        eprintln!("engine ready in {:.1}s", t0.elapsed().as_secs_f64());
        let model_id = engine.model_id().to_string();

        let app = Router::new()
            .route("/v1/chat/completions", post(handle_chat_completions))
            .with_state(ChatAppState {
                registry: ChatRegistry::single(engine),
            });

        let body = serde_json::json!({
            "model": model_id,
            "messages": [{"role": "user", "content": "The capital of France is"}],
            "temperature": 0.0,
            "max_tokens": 256,
        })
        .to_string();

        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap();
        let contents: Vec<String> = rt.block_on(async {
            let mut out = Vec::new();
            for run in 0..RUNS {
                let (status, resp) = post_json(&app, body.clone()).await;
                assert_eq!(status, StatusCode::OK, "run {run}: {resp}");
                let v: serde_json::Value = serde_json::from_str(&resp).unwrap();
                if run == 0 {
                    eprintln!("run 0 raw choice: {}", v["choices"][0]);
                }
                let msg = &v["choices"][0]["message"];
                let mut content = msg["content"].as_str().unwrap_or_default().to_string();
                if content.is_empty() {
                    content = msg["reasoning_content"]
                        .as_str()
                        .unwrap_or_default()
                        .to_string();
                }
                eprintln!("run {run}: {content:?}");
                out.push(content);
            }
            out
        });

        for (i, c) in contents.iter().enumerate() {
            assert!(
                c.contains("Paris"),
                "run {i} does not contain Paris -- the default serving path is WRONG, not \
                 merely nondeterministic: {contents:?}"
            );
        }
        for (i, c) in contents.iter().enumerate().skip(1) {
            assert_eq!(
                c, &contents[0],
                "run {i} diverged from run 0 at temperature 0 -- the default serving path is \
                 NONDETERMINISTIC: {contents:?}"
            );
        }
        eprintln!(
            "VERDICT for #63: {RUNS} identical greedy runs through the real serving path, all \
             answering Paris -- shipping is not broken by the grouped-MoE defect on this prompt"
        );
    }
}
