#[cfg(not(feature = "wgpu"))]
#[test]
fn wgpu_prefill_serving_ab_is_cfg_out_without_the_wgpu_feature() {
    eprintln!(
        "wgpu_prefill_serving_ab compiled OUT (no `wgpu` feature). This is a SKIP, not a pass. \
         Re-run with --features wgpu and NV_WGPU_SERVE_TEST=1."
    );
}

#[cfg(feature = "wgpu")]
mod gated {
    use std::path::PathBuf;
    use std::sync::Arc;

    use axum::body::{to_bytes, Body};
    use axum::http::{Request, StatusCode};
    use axum::routing::post;
    use axum::Router;
    use tower::ServiceExt;

    use speaches_plus::oapi::chat::{handle_chat_completions, ChatAppState, ChatEngine};
    use speaches_plus::oapi::chat_engine::ChatRegistry;
    use speaches_plus::oapi::chat_engine_wgpu::WgpuChatEngine;

    static ENV_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

    const PARAGRAPH: &str = "Polynesian wayfinders crossed thousands of miles of open ocean by \
                             reading star paths, swell patterns, and the flight of birds, long \
                             before the magnetic compass reached Europe. Later, the marine \
                             chronometer finally solved the longitude problem, and satellite \
                             positioning eventually made a fix a matter of milliseconds rather \
                             than months of careful dead reckoning. ";

    fn short_prompt() -> String {
        format!("Here is a paragraph about navigation at sea: {PARAGRAPH}Summarize this paragraph in one sentence.")
    }

    fn long_prompt() -> String {
        let mut s =
            String::from("Here is a passage about navigation at sea, repeated for emphasis: ");
        for _ in 0..6 {
            s.push_str(PARAGRAPH);
        }
        s.push_str("Summarize this passage in one sentence.");
        s
    }

    fn enabled() -> bool {
        std::env::var("NV_WGPU_SERVE_TEST").ok().as_deref() == Some("1")
    }

    fn model_dir() -> Option<PathBuf> {
        if let Ok(d) = std::env::var("NV_WGPU_SERVE_DIR") {
            let p = PathBuf::from(d);
            return p.join("config.json").exists().then_some(p);
        }
        let root = PathBuf::from(std::env::var("HOME").ok()?)
            .join(".cache/huggingface/hub/models--google--gemma-4-E4B-it/snapshots");
        let mut cand: Vec<PathBuf> = std::fs::read_dir(&root)
            .ok()?
            .flatten()
            .map(|e| e.path())
            .filter(|p| p.join("config.json").exists() && p.join("tokenizer.json").exists())
            .collect();
        cand.sort();
        cand.pop()
    }

    fn app(engine: Arc<dyn ChatEngine>) -> Router {
        Router::new()
            .route("/v1/chat/completions", post(handle_chat_completions))
            .with_state(ChatAppState {
                registry: ChatRegistry::single(engine),
            })
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

    fn run_config(
        dir: &std::path::Path,
        label: &str,
        prompts: &[String],
    ) -> Vec<(String, u64, u64)> {
        let t0 = std::time::Instant::now();
        let engine = Arc::new(
            WgpuChatEngine::load_with(dir, 1024, None)
                .unwrap_or_else(|err| panic!("[{label}] wgpu engine did not load: {err:#}")),
        ) as Arc<dyn ChatEngine>;
        eprintln!(
            "[{label}] engine ready in {:.1}s",
            t0.elapsed().as_secs_f64()
        );
        let app = app(engine.clone());
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap();
        rt.block_on(async move {
            let mut out = Vec::new();
            for (i, prompt) in prompts.iter().enumerate() {
                let body = format!(
                    r#"{{"model":"{}","temperature":0,"max_tokens":48,"seed":7,
                         "messages":[{{"role":"user","content":{}}}]}}"#,
                    engine.model_id(),
                    serde_json::to_string(prompt).unwrap()
                );
                let t = std::time::Instant::now();
                let (status, resp) = post_json(&app, body).await;
                let wall = t.elapsed().as_secs_f64();
                assert_eq!(status, StatusCode::OK, "[{label}#{i}] {resp}");
                let v: serde_json::Value = serde_json::from_str(&resp).unwrap();
                let content = v["choices"][0]["message"]["content"]
                    .as_str()
                    .unwrap_or_default()
                    .to_string();
                let completion = v["usage"]["completion_tokens"].as_u64().unwrap_or(0);
                let prompt_toks = v["usage"]["prompt_tokens"].as_u64().unwrap_or(0);
                eprintln!(
                    "[{label}#{i}] prompt_tokens={prompt_toks} completion_tokens={completion} \
                     wall={wall:.2}s content={content:?}"
                );
                assert!(completion > 0, "[{label}#{i}] no tokens generated");
                out.push((content, completion, prompt_toks));
            }
            out
        })
    }

    #[test]
    #[ignore]
    fn chunked_prefill_serving_matches_stepped_prefill_serving() {
        if !enabled() {
            panic!(
                "this test is #[ignore]d, so it was asked for BY NAME, but NV_WGPU_SERVE_TEST=1 \
                 is not set. Returning here prints `1 passed` in 0.00s having served nothing. \
                 This is a SKIP, not a pass."
            );
        }
        let Some(dir) = model_dir() else {
            panic!(
                "no wgpu servable model dir: set NV_WGPU_SERVE_DIR, or cache \
                 google/gemma-4-E4B-it. This is a SKIP, not a pass."
            )
        };
        let _lock = ENV_LOCK.lock().unwrap();
        let _ = tracing_subscriber::fmt()
            .with_env_filter("speaches_plus=info")
            .with_writer(std::io::stderr)
            .try_init();

        let prompts = vec![short_prompt(), long_prompt()];

        std::env::set_var("NV_E4B_WGPU_W4_SG", "0");

        std::env::set_var("NV_WGPU_CHAT_CHUNKED_PREFILL", "1");
        let chunked = run_config(&dir, "chunked", &prompts);
        let chunked2 = run_config(&dir, "chunked-2", &prompts);

        std::env::set_var("NV_WGPU_CHAT_CHUNKED_PREFILL", "0");
        let stepped = run_config(&dir, "stepped", &prompts);
        std::env::remove_var("NV_WGPU_CHAT_CHUNKED_PREFILL");

        std::env::remove_var("NV_E4B_WGPU_W4_SG");

        for (i, (c, s)) in chunked.iter().zip(stepped.iter()).enumerate() {
            assert_eq!(
                c.2, s.2,
                "prompt #{i}: tokenization must not depend on prefill mode"
            );
            assert_eq!(
                c.0, s.0,
                "prompt #{i}: chunked prefill must serve the exact completion the stepped path serves"
            );
            assert_eq!(c.1, s.1, "prompt #{i}: completion token counts diverged");
        }
        for (i, (a, b)) in chunked.iter().zip(chunked2.iter()).enumerate() {
            assert_eq!(a.0, b.0, "prompt #{i}: chunked path not reproducible");
            assert_eq!(a.1, b.1);
        }
    }

    #[test]
    #[ignore]
    fn chunked_prefill_serving_matches_stepped_under_default_sg16_decode() {
        if !enabled() {
            panic!(
                "this test is #[ignore]d, so it was asked for BY NAME, but NV_WGPU_SERVE_TEST=1 \
                 is not set. Returning here prints `1 passed` in 0.00s having served nothing. \
                 This is a SKIP, not a pass."
            );
        }
        let Some(dir) = model_dir() else {
            panic!(
                "no wgpu servable model dir: set NV_WGPU_SERVE_DIR, or cache \
                 google/gemma-4-E4B-it. This is a SKIP, not a pass."
            )
        };
        let _lock = ENV_LOCK.lock().unwrap();
        let _ = tracing_subscriber::fmt()
            .with_env_filter("speaches_plus=info")
            .with_writer(std::io::stderr)
            .try_init();

        let prompts = vec![short_prompt(), long_prompt()];

        std::env::set_var("NV_WGPU_CHAT_CHUNKED_PREFILL", "1");
        let chunked = run_config(&dir, "sg16-chunked", &prompts);
        let chunked2 = run_config(&dir, "sg16-chunked-2", &prompts);

        std::env::set_var("NV_WGPU_CHAT_CHUNKED_PREFILL", "0");
        let stepped = run_config(&dir, "sg16-stepped", &prompts);
        std::env::remove_var("NV_WGPU_CHAT_CHUNKED_PREFILL");

        for (i, (c, s)) in chunked.iter().zip(stepped.iter()).enumerate() {
            assert_eq!(
                c.2, s.2,
                "prompt #{i}: tokenization must not depend on prefill mode"
            );
            assert_eq!(
                c.0, s.0,
                "prompt #{i}: sg16 chunked prefill must serve the exact completion the stepped path serves"
            );
            assert_eq!(c.1, s.1, "prompt #{i}: completion token counts diverged");
        }
        for (i, (a, b)) in chunked.iter().zip(chunked2.iter()).enumerate() {
            assert_eq!(a.0, b.0, "prompt #{i}: sg16 chunked path not reproducible");
            assert_eq!(a.1, b.1);
        }
    }

    #[test]
    #[ignore]
    fn default_env_serving_matches_the_prefill_off_stepped_path() {
        if !enabled() {
            panic!(
                "this test is #[ignore]d, so it was asked for BY NAME, but NV_WGPU_SERVE_TEST=1 \
                 is not set. Returning here prints `1 passed` in 0.00s having served nothing. \
                 This is a SKIP, not a pass."
            );
        }
        let Some(dir) = model_dir() else {
            panic!(
                "no wgpu servable model dir: set NV_WGPU_SERVE_DIR, or cache \
                 google/gemma-4-E4B-it. This is a SKIP, not a pass."
            )
        };
        let _lock = ENV_LOCK.lock().unwrap();
        let _ = tracing_subscriber::fmt()
            .with_env_filter("speaches_plus=info")
            .with_writer(std::io::stderr)
            .try_init();

        let prompts = vec![short_prompt()];
        let default_run = run_config(&dir, "default-env", &prompts);

        std::env::set_var("NV_E4B_WGPU_PREFILL", "0");
        let prefill_off = run_config(&dir, "model-prefill-off", &prompts);
        std::env::remove_var("NV_E4B_WGPU_PREFILL");

        assert_eq!(
            default_run[0].0, prefill_off[0].0,
            "default env must serve the exact completion of the pre-chunked-prefill stepped path"
        );
        assert_eq!(default_run[0].1, prefill_off[0].1);
        assert_eq!(default_run[0].2, prefill_off[0].2);
    }
}
