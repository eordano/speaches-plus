#[cfg(not(feature = "wgpu"))]
#[test]
fn wgpu_qwen35_dense_serving_ab_is_cfg_out_without_the_wgpu_feature() {
    eprintln!(
        "wgpu_qwen35_dense_serving_ab compiled OUT (no `wgpu` feature). This is a SKIP, not a \
         pass. Re-run with --features wgpu and NV_WGPU_SERVE_TEST=1."
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
    use speaches_plus::oapi::chat_engine_wgpu::{
        classify_wgpu_model, WgpuChatEngine, WgpuModelKind,
    };

    const DENSE_9B_STYLE_CONFIG: &str = r#"{
        "architectures": ["Qwen3_5ForConditionalGeneration"],
        "model_type": "qwen3_5",
        "text_config": {"hidden_size": 4096}
    }"#;

    const MOE_STYLE_CONFIG: &str = r#"{
        "architectures": ["Qwen3_5MoeForCausalLM"],
        "model_type": "qwen3_5_moe",
        "text_config": {"hidden_size": 2048}
    }"#;

    #[test]
    fn classify_routes_dense_after_moe() {
        let dense = classify_wgpu_model(DENSE_9B_STYLE_CONFIG).expect("classify dense");
        assert_eq!(dense, WgpuModelKind::Qwen3_5Dense);
        let moe = classify_wgpu_model(MOE_STYLE_CONFIG).expect("classify moe");
        assert_eq!(
            moe,
            WgpuModelKind::Qwen3_5Moe,
            "the qwen3_5 dense prefix match must not shadow the moe tags"
        );
    }

    fn enabled() -> bool {
        std::env::var("NV_WGPU_SERVE_TEST").ok().as_deref() == Some("1")
    }

    fn model_dir() -> Option<PathBuf> {
        if let Ok(d) = std::env::var("NV_QWEN35_DENSE_SERVE_DIR") {
            let p = PathBuf::from(d);
            return p.join("config.json").exists().then_some(p);
        }
        let root = PathBuf::from(std::env::var("HOME").ok()?)
            .join(".cache/huggingface/hub/models--Qwen--Qwen3.5-9B/snapshots");
        std::fs::read_dir(&root)
            .ok()?
            .flatten()
            .map(|e| e.path())
            .find(|p| {
                p.join("config.json").exists() && p.join("model.safetensors.index.json").exists()
            })
    }

    fn app(engine: Arc<dyn ChatEngine>) -> Router {
        Router::new()
            .route("/v1/chat/completions", post(handle_chat_completions))
            .with_state(ChatAppState {
                registry: ChatRegistry::single(engine),
            })
    }

    struct RunOut {
        content: String,
        reasoning: String,
        completion_tokens: u64,
        prompt_tokens: u64,
        wall_s: f64,
    }

    fn run_prompt(
        engine: &Arc<dyn ChatEngine>,
        label: &str,
        prompt: &str,
        max_tokens: u32,
    ) -> RunOut {
        let router = app(engine.clone());
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap();
        rt.block_on(async move {
            let body = format!(
                r#"{{"model":"{}","max_tokens":{max_tokens},"temperature":0,
                     "enable_thinking":false,
                     "messages":[{{"role":"user","content":{}}}]}}"#,
                engine.model_id(),
                serde_json::to_string(prompt).unwrap()
            );
            let req = Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header("content-type", "application/json")
                .body(Body::from(body))
                .unwrap();
            let t = std::time::Instant::now();
            let resp = router.clone().oneshot(req).await.unwrap();
            let status = resp.status();
            let bytes = to_bytes(resp.into_body(), 1 << 22).await.unwrap();
            let text = String::from_utf8_lossy(&bytes).into_owned();
            let wall_s = t.elapsed().as_secs_f64();
            assert_eq!(status, StatusCode::OK, "[{label}] {text}");
            let v: serde_json::Value = serde_json::from_str(&text).unwrap();
            let content = v["choices"][0]["message"]["content"]
                .as_str()
                .unwrap_or_default()
                .to_string();
            let reasoning = v["choices"][0]["message"]["reasoning_content"]
                .as_str()
                .unwrap_or_default()
                .to_string();
            let completion_tokens = v["usage"]["completion_tokens"].as_u64().unwrap_or(0);
            let prompt_tokens = v["usage"]["prompt_tokens"].as_u64().unwrap_or(0);
            eprintln!(
                "[{label}] prompt_tokens={prompt_tokens} completion_tokens={completion_tokens} \
                 content_len={} reasoning_len={} wall={wall_s:.2}s ({:.1} ms/tok incl. prefill)",
                content.len(),
                reasoning.len(),
                wall_s * 1000.0 / completion_tokens.max(1) as f64
            );
            assert!(completion_tokens > 0, "[{label}] no tokens generated");
            RunOut {
                content,
                reasoning,
                completion_tokens,
                prompt_tokens,
                wall_s,
            }
        })
    }

    #[test]
    #[ignore = "loads ~17 GB of bf16 weights; set NV_WGPU_SERVE_TEST=1"]
    fn qwen35_dense_greedy_serving_is_deterministic_across_requests_and_reload() {
        if !enabled() {
            panic!(
                "this test is #[ignore]d, so it was asked for BY NAME, but NV_WGPU_SERVE_TEST=1 \
                 is not set. This is a SKIP, not a pass."
            );
        }
        let Some(dir) = model_dir() else {
            panic!(
                "NEEDS DOWNLOAD: models--Qwen--Qwen3.5-9B is cached METADATA-ONLY on this box \
                 (config.json, no model.safetensors.index.json), so this A/B has never run. \
                 Hydrate it with LFS onto /tank (zroot has no room) or set \
                 NV_QWEN35_DENSE_SERVE_DIR. This is a SKIP, not a pass."
            )
        };
        let _ = tracing_subscriber::fmt()
            .with_env_filter("speaches_plus=info")
            .with_writer(std::io::stderr)
            .try_init();

        let prompt = "What is the capital of France? Answer in one short sentence.";

        let t0 = std::time::Instant::now();
        let engine_a = Arc::new(
            WgpuChatEngine::load_with(&dir, 1024, None).expect("wgpu engine A did not load"),
        );
        eprintln!("[engine-a] ready in {:.1}s", t0.elapsed().as_secs_f64());
        assert_eq!(engine_a.kind(), WgpuModelKind::Qwen3_5Dense);
        let a: Arc<dyn ChatEngine> = engine_a;

        let r1 = run_prompt(&a, "a-run1", prompt, 64);
        let r2 = run_prompt(&a, "a-run2", prompt, 64);
        assert_eq!(
            r1.content, r2.content,
            "greedy completion changed between two back-to-back requests"
        );
        assert_eq!(
            r1.reasoning, r2.reasoning,
            "greedy reasoning bytes changed between two back-to-back requests"
        );
        assert_eq!(r1.completion_tokens, r2.completion_tokens);
        assert_eq!(r1.prompt_tokens, r2.prompt_tokens);
        assert!(
            !r1.content.trim().is_empty(),
            "enable_thinking=false must yield a direct non-empty answer, got reasoning: {:?}",
            r1.reasoning
        );
        assert!(
            r1.content.to_lowercase().contains("paris"),
            "expected a factual answer mentioning Paris, got: {:?}",
            r1.content
        );
        drop(a);

        let t1 = std::time::Instant::now();
        let engine_b = Arc::new(
            WgpuChatEngine::load_with(&dir, 1024, None).expect("wgpu engine B did not load"),
        );
        eprintln!("[engine-b] ready in {:.1}s", t1.elapsed().as_secs_f64());
        let b: Arc<dyn ChatEngine> = engine_b;
        let r3 = run_prompt(&b, "b-run1", prompt, 64);
        assert_eq!(
            r1.content, r3.content,
            "greedy completion changed across an engine reload"
        );
        assert_eq!(
            r1.reasoning, r3.reasoning,
            "greedy reasoning bytes changed across an engine reload"
        );
        assert_eq!(r1.completion_tokens, r3.completion_tokens);
        eprintln!(
            "[done] deterministic across 2 requests + reload; last wall {:.2}s",
            r3.wall_s
        );
    }
}
