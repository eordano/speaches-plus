#[cfg(not(feature = "wgpu"))]
#[test]
fn wgpu_gptoss_serving_ab_is_cfg_out_without_the_wgpu_feature() {
    eprintln!(
        "wgpu_gptoss_serving_ab compiled OUT (no `wgpu` feature). This is a SKIP, not a pass. \
         Re-run with --features wgpu and NV_GPTOSS_SERVE_TEST=1."
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
    use speaches_plus::oapi::chat_template::harmony_final_text;

    const HARMONY_MARKUP: [&str; 6] = [
        "<|channel|>",
        "<|start|>",
        "<|return|>",
        "<|message|>",
        "<|end|>",
        "<|call|>",
    ];

    fn enabled() -> bool {
        std::env::var("NV_GPTOSS_SERVE_TEST").ok().as_deref() == Some("1")
    }

    fn model_dir() -> Option<PathBuf> {
        if let Ok(d) = std::env::var("NV_GPTOSS_DIR") {
            let p = PathBuf::from(d);
            return p.join("config.json").exists().then_some(p);
        }
        let base = format!(
            "{}/.cache/huggingface/hub/models--openai--gpt-oss-20b/snapshots",
            std::env::var("HOME").unwrap_or_default()
        );
        std::fs::read_dir(base)
            .ok()?
            .filter_map(|e| e.ok())
            .map(|e| e.path())
            .find(|p| p.join("config.json").is_file() && p.join("tokenizer.json").is_file())
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

    fn ask_with_usage(
        engine: Arc<dyn ChatEngine>,
        question: &str,
        max_tokens: u32,
    ) -> (String, u64) {
        let router = app(engine.clone());
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap();
        rt.block_on(async move {
            let body = format!(
                r#"{{"model":"{}","max_tokens":{max_tokens},"temperature":0,
                     "messages":[{{"role":"user","content":{}}}]}}"#,
                engine.model_id(),
                serde_json::to_string(question).unwrap()
            );
            let (status, resp) = post_json(&router, body).await;
            assert_eq!(status, StatusCode::OK, "{resp}");
            let v: serde_json::Value = serde_json::from_str(&resp).unwrap();
            let content = v["choices"][0]["message"]["content"]
                .as_str()
                .unwrap_or_default()
                .to_string();
            let prompt_tokens = v["usage"]["prompt_tokens"].as_u64().unwrap_or(0);
            (content, prompt_tokens)
        })
    }

    fn ask(engine: Arc<dyn ChatEngine>, question: &str, max_tokens: u32) -> String {
        let router = app(engine.clone());
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap();
        rt.block_on(async move {
            let body = format!(
                r#"{{"model":"{}","max_tokens":{max_tokens},"temperature":0,
                     "messages":[{{"role":"user","content":{}}}]}}"#,
                engine.model_id(),
                serde_json::to_string(question).unwrap()
            );
            let (status, resp) = post_json(&router, body).await;
            assert_eq!(status, StatusCode::OK, "{resp}");
            let v: serde_json::Value = serde_json::from_str(&resp).unwrap();
            v["choices"][0]["message"]["content"]
                .as_str()
                .unwrap_or_default()
                .to_string()
        })
    }

    #[test]
    fn classify_routes_gpt_oss_to_the_gpt_oss_decoder() {
        let cfg = r#"{"architectures":["GptOssForCausalLM"],"model_type":"gpt_oss"}"#;
        assert_eq!(classify_wgpu_model(cfg).unwrap(), WgpuModelKind::GptOss);
        let by_type = r#"{"model_type":"gpt_oss"}"#;
        assert_eq!(classify_wgpu_model(by_type).unwrap(), WgpuModelKind::GptOss);
    }

    #[test]
    fn real_template_renders_harmony_with_todays_date() {
        let Some(dir) = model_dir() else {
            panic!(
                "openai/gpt-oss-20b is not cached (set NV_GPTOSS_DIR). This test renders the \
                 shipped harmony template on the host -- no GPU, no env gate -- so a skip here \
                 is a SKIP, not a pass: it printed `1 passed` in 0.00s having rendered nothing."
            )
        };
        let template =
            speaches_plus::oapi::chat_template::ChatTemplate::load_reason(&dir).expect("template");
        let msgs = serde_json::json!([{"role": "user", "content": "What is 2+2?"}]);
        let out = template.render(&msgs, None, true).unwrap();
        assert!(out.starts_with("<|start|>system<|message|>"), "{out}");
        assert!(
            out.contains("<|start|>user<|message|>What is 2+2?<|end|>"),
            "{out}"
        );
        assert!(out.trim_end().ends_with("<|start|>assistant"), "{out}");
        assert!(
            !out.contains("1970-01-01"),
            "epoch strftime stub leaked: {out}"
        );
        let secs = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs() as i64;
        let today = speaches_plus::oapi::chat_template::format_utc_date("%Y-%m-%d", secs);
        assert!(
            out.contains(&format!("Current date: {today}")),
            "rendered prompt must carry today's date {today}: {out}"
        );
    }

    #[test]
    fn harmony_filter_is_wired_for_the_served_shape() {
        let raw = "<|channel|>analysis<|message|>The user asked. Compute.<|end|><|start|>assistant\
                   <|channel|>final<|message|>Paris.";
        assert_eq!(harmony_final_text(raw), "Paris.");
    }

    #[test]
    #[ignore = "loads 13 GB of gpt-oss-20b weights; set NV_GPTOSS_SERVE_TEST=1"]
    fn a_completion_truncated_before_the_final_channel_returns_its_reasoning() {
        if !enabled() {
            panic!(
                "this test is #[ignore]d, so it was asked for BY NAME, but \
                 NV_GPTOSS_SERVE_TEST=1 is not set. This is a SKIP, not a pass."
            );
        }
        let Some(dir) = model_dir() else {
            panic!(
                "openai/gpt-oss-20b is not cached (set NV_GPTOSS_DIR). This is a SKIP, not a pass."
            )
        };
        let engine = Arc::new(
            WgpuChatEngine::load_with(&dir, 1024, None).expect("load gpt-oss wgpu engine"),
        ) as Arc<dyn ChatEngine>;

        let router = app(engine.clone());
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap();
        let v: serde_json::Value = rt.block_on(async move {
            let body = format!(
                r#"{{"model":"{}","max_tokens":64,"temperature":0,
                     "messages":[{{"role":"user","content":"Count to five, digits only."}}]}}"#,
                engine.model_id()
            );
            let (status, resp) = post_json(&router, body).await;
            assert_eq!(status, StatusCode::OK, "{resp}");
            serde_json::from_str(&resp).unwrap()
        });

        let msg = &v["choices"][0]["message"];
        assert_eq!(
            v["choices"][0]["finish_reason"], "length",
            "the premise is a truncated completion; if it finished, raise nothing and \
             lower max_tokens instead: {v}"
        );
        assert!(
            v["usage"]["completion_tokens"].as_u64().unwrap_or(0) > 0,
            "no tokens generated, so there is nothing to lose and this proves nothing: {v}"
        );
        let reasoning = msg["reasoning_content"].as_str().unwrap_or("");
        assert!(
            !reasoning.trim().is_empty(),
            "a completion truncated before the final channel must return its analysis as \
             reasoning_content; an empty content with no reasoning discards every token the \
             caller paid for: {v}"
        );
        for m in HARMONY_MARKUP {
            assert!(
                !reasoning.contains(m),
                "harmony markup {m} leaked into reasoning_content: {reasoning:?}"
            );
        }
    }

    #[test]
    #[ignore = "loads 13 GB of gpt-oss-20b weights twice; set NV_GPTOSS_SERVE_TEST=1"]
    fn gptoss_chunked_prefill_matches_per_token_prefill_exactly() {
        if !enabled() {
            panic!(
                "this test is #[ignore]d, so it was asked for BY NAME, but \
                 NV_GPTOSS_SERVE_TEST=1 is not set. This is a SKIP, not a pass."
            );
        }
        let Some(dir) = model_dir() else {
            panic!(
                "openai/gpt-oss-20b is not cached (set NV_GPTOSS_DIR). This is a SKIP, not a pass."
            )
        };
        let _ = tracing_subscriber::fmt()
            .with_env_filter("speaches_plus=info")
            .with_writer(std::io::stderr)
            .try_init();

        let filler = "The following is background material that does not change the answer. "
            .repeat(40);
        let question = format!(
            "{filler}\nIgnoring all of the above, what is the capital of France? \
             Answer in one short sentence."
        );
        let question = question.as_str();
        let arm = |m: Option<&str>| {
            match m {
                Some(v) => std::env::set_var("NV_GPTOSS_WGPU_PREFILL_M", v),
                None => std::env::remove_var("NV_GPTOSS_WGPU_PREFILL_M"),
            }
            assert_eq!(
                nv_models::gpt_oss_wgpu::prefill_m() == 0,
                m == Some("0"),
                "the knob did not reach prefill_m, so this A/B compares one arm with itself"
            );
            let engine = Arc::new(
                WgpuChatEngine::load_with(&dir, 1024, None).expect("load gpt-oss wgpu engine"),
            ) as Arc<dyn ChatEngine>;
            ask_with_usage(engine, question, 192)
        };

        let (chunked, chunked_tokens) = arm(None);
        let (per_token, per_token_tokens) = arm(Some("0"));
        eprintln!("[gptoss] cross-arm prompt_tokens={chunked_tokens}");
        assert!(
            chunked_tokens > 16 * 4,
            "prompt is only {chunked_tokens} tokens, so it does not exercise the \
             multi-chunk path this test exists to cover"
        );
        assert_eq!(chunked_tokens, per_token_tokens, "arms saw different prompts");
        std::env::remove_var("NV_GPTOSS_WGPU_PREFILL_M");

        assert!(!chunked.trim().is_empty(), "empty answer from chunked arm");
        assert_eq!(
            chunked, per_token,
            "chunked prefill changed the greedy answer; it is a batching of the same \
             math and must be output-identical to NV_GPTOSS_WGPU_PREFILL_M=0"
        );
    }

    #[test]
    #[ignore = "loads 13 GB of gpt-oss-20b weights; set NV_GPTOSS_SERVE_TEST=1"]
    fn gptoss_chat_completion_is_clean_deterministic_and_restart_stable() {
        if !enabled() {
            panic!(
                "this test is #[ignore]d, so it was asked for BY NAME, but \
                 NV_GPTOSS_SERVE_TEST=1 is not set. This is a SKIP, not a pass."
            );
        }
        let Some(dir) = model_dir() else {
            panic!(
                "openai/gpt-oss-20b is not cached (set NV_GPTOSS_DIR). This is a SKIP, not a pass."
            )
        };
        let _ = tracing_subscriber::fmt()
            .with_env_filter("speaches_plus=info")
            .with_writer(std::io::stderr)
            .try_init();

        let question = "What is the capital of France? Answer in one short sentence.";

        let engine = Arc::new(
            WgpuChatEngine::load_with(&dir, 1024, None).expect("load gpt-oss wgpu engine"),
        ) as Arc<dyn ChatEngine>;

        let a = ask(engine.clone(), question, 192);
        eprintln!("[gptoss] answer A: {a:?}");
        assert!(!a.trim().is_empty(), "empty final-channel answer");
        for m in HARMONY_MARKUP {
            assert!(
                !a.contains(m),
                "harmony markup {m} leaked into content: {a:?}"
            );
        }
        assert!(
            a.to_ascii_lowercase().contains("paris"),
            "expected a coherent final answer mentioning Paris: {a:?}"
        );

        let b = ask(engine.clone(), question, 192);
        assert_eq!(
            a, b,
            "greedy answers must be byte-identical across requests"
        );

        drop(engine);
        let engine2 = Arc::new(
            WgpuChatEngine::load_with(&dir, 1024, None).expect("reload gpt-oss wgpu engine"),
        ) as Arc<dyn ChatEngine>;
        let c = ask(engine2, question, 192);
        assert_eq!(
            a, c,
            "greedy answers must be byte-identical across a restart"
        );
    }
}
