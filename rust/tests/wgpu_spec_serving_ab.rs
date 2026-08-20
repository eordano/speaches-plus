#[cfg(not(feature = "wgpu"))]
#[test]
fn wgpu_spec_serving_ab_is_cfg_out_without_the_wgpu_feature() {
    eprintln!(
        "wgpu_spec_serving_ab compiled OUT (no `wgpu` feature). This is a SKIP, not a pass. \
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
    use speaches_plus::oapi::chat_engine_wgpu::{
        spec::SpecKnobs, spec_route_eligible, WgpuChatEngine, WgpuModelKind,
    };

    static ENV_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

    const PARAGRAPH: &str = "Polynesian wayfinders crossed thousands of miles of open ocean by \
                             reading star paths, swell patterns, and the flight of birds, long \
                             before the magnetic compass reached Europe. Later, the marine \
                             chronometer finally solved the longitude problem, and satellite \
                             positioning eventually made a fix a matter of milliseconds rather \
                             than months of careful dead reckoning. ";

    fn structured_prompt() -> String {
        format!(
            "Repeat the following paragraph word for word, exactly as written, three times in a \
             row, with no commentary before or after: {PARAGRAPH}"
        )
    }

    fn prose_prompt() -> String {
        "Write a long, detailed essay about the history of navigation at sea, from Polynesian \
         wayfinding to satellite positioning. Cover the instruments, the key voyages, and the \
         science behind each advance."
            .to_string()
    }

    fn window_crossing_prompt() -> String {
        let mut s =
            String::from("Here is a passage about navigation at sea, repeated for emphasis: ");
        for _ in 0..8 {
            s.push_str(PARAGRAPH);
        }
        s.push_str("Now summarize the passage in detail, covering every instrument mentioned.");
        s
    }

    fn adversarial_prompt() -> String {
        "The cat sat on the mat. The cat ran to the red door. The cat sat on the sofa. The cat \
         ran to the green gate. The cat sat on the ledge. Continue this story for many more \
         sentences, keeping the same rhythm but always choosing new places and colors."
            .to_string()
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

    struct RunOut {
        content: String,
        completion_tokens: u64,
        prompt_tokens: u64,
        wall_s: f64,
    }

    fn run_config(
        dir: &std::path::Path,
        label: &str,
        prompts: &[String],
        max_tokens: u32,
        extra_params: &str,
    ) -> Vec<RunOut> {
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
                    r#"{{"model":"{}","max_tokens":{max_tokens},{extra_params}
                         "messages":[{{"role":"user","content":{}}}]}}"#,
                    engine.model_id(),
                    serde_json::to_string(prompt).unwrap()
                );
                let t = std::time::Instant::now();
                let (status, resp) = post_json(&app, body).await;
                let wall_s = t.elapsed().as_secs_f64();
                assert_eq!(status, StatusCode::OK, "[{label}#{i}] {resp}");
                let v: serde_json::Value = serde_json::from_str(&resp).unwrap();
                let content = v["choices"][0]["message"]["content"]
                    .as_str()
                    .unwrap_or_default()
                    .to_string();
                let completion_tokens = v["usage"]["completion_tokens"].as_u64().unwrap_or(0);
                let prompt_tokens = v["usage"]["prompt_tokens"].as_u64().unwrap_or(0);
                eprintln!(
                    "[{label}#{i}] prompt_tokens={prompt_tokens} \
                     completion_tokens={completion_tokens} wall={wall_s:.2}s"
                );
                assert!(completion_tokens > 0, "[{label}#{i}] no tokens generated");
                out.push(RunOut {
                    content,
                    completion_tokens,
                    prompt_tokens,
                    wall_s,
                });
            }
            out
        })
    }

    fn assert_streams_match(a: &[RunOut], b: &[RunOut], what: &str) {
        assert_eq!(a.len(), b.len());
        for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
            assert_eq!(
                x.prompt_tokens, y.prompt_tokens,
                "{what} prompt #{i}: tokenization diverged"
            );
            assert_eq!(
                x.completion_tokens, y.completion_tokens,
                "{what} prompt #{i}: completion token counts diverged"
            );
            assert_eq!(
                x.content, y.content,
                "{what} prompt #{i}: completions diverged"
            );
        }
    }

    #[test]
    fn spec_route_eligibility_table() {
        let on = SpecKnobs::parse(Some("1"), None, None);
        let default = SpecKnobs::parse(None, None, None);
        assert!(spec_route_eligible(WgpuModelKind::Gemma4E4b, false, on));
        assert!(!spec_route_eligible(WgpuModelKind::Gemma4E4b, true, on));
        assert!(!spec_route_eligible(WgpuModelKind::Gemma4Dense, false, on));
        assert!(!spec_route_eligible(WgpuModelKind::Qwen3_5Moe, false, on));
        assert!(spec_route_eligible(
            WgpuModelKind::Gemma4E4b,
            false,
            default
        ));
        assert!(!spec_route_eligible(
            WgpuModelKind::Gemma4E4b,
            true,
            default
        ));
        assert!(!spec_route_eligible(
            WgpuModelKind::Gemma4Dense,
            false,
            default
        ));
        let zero = SpecKnobs::parse(Some("0"), None, None);
        assert!(!spec_route_eligible(WgpuModelKind::Gemma4E4b, false, zero));
        assert!(!spec_route_eligible(WgpuModelKind::Gemma4E4b, true, zero));
    }

    #[test]
    #[ignore]
    fn spec_serving_is_token_identical_to_plain_greedy() {
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

        let prompts = vec![
            structured_prompt(),
            prose_prompt(),
            window_crossing_prompt(),
            adversarial_prompt(),
        ];
        let greedy = r#""temperature":0,"seed":7,"#;

        std::env::set_var("NV_WGPU_SPEC", "0");
        let plain = run_config(&dir, "spec-off", &prompts, 400, greedy);
        let plain2 = run_config(&dir, "spec-off-2", &prompts, 400, greedy);
        assert_streams_match(&plain, &plain2, "spec-off reproducibility");

        std::env::remove_var("NV_WGPU_SPEC");
        let spec = run_config(&dir, "spec-default-on", &prompts, 400, greedy);

        assert_streams_match(&plain, &spec, "spec-default-on vs spec-off");
    }

    #[test]
    #[ignore]
    fn sampled_requests_take_the_plain_loop_under_spec() {
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

        let prompts = vec![structured_prompt()];
        let sampled = r#""temperature":0.8,"top_p":0.95,"seed":1234,"#;

        std::env::set_var("NV_WGPU_SPEC", "0");
        let plain = run_config(&dir, "sampled-spec-off", &prompts, 96, sampled);

        std::env::remove_var("NV_WGPU_SPEC");
        let spec = run_config(&dir, "sampled-spec-default-on", &prompts, 96, sampled);

        assert_streams_match(&plain, &spec, "sampled route");
    }

    #[test]
    #[ignore]
    fn spec_timed_ab() {
        if !enabled() || std::env::var("NV_WGPU_SPEC_BENCH").ok().as_deref() != Some("1") {
            panic!(
                "this bench is #[ignore]d, so it was asked for BY NAME, but NV_WGPU_SERVE_TEST=1 \
                 and NV_WGPU_SPEC_BENCH=1 are not both set. This is a SKIP, not a pass."
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

        let prompts = vec![structured_prompt(), prose_prompt()];
        let greedy = r#""temperature":0,"seed":7,"#;

        for rep in 0..2 {
            std::env::set_var("NV_WGPU_SPEC", "0");
            let off = run_config(&dir, &format!("bench-off-{rep}"), &prompts, 400, greedy);
            std::env::remove_var("NV_WGPU_SPEC");
            let on = run_config(&dir, &format!("bench-on-{rep}"), &prompts, 400, greedy);
            for (i, (o, s)) in off.iter().zip(on.iter()).enumerate() {
                eprintln!(
                    "[bench rep {rep} prompt {i}] off wall={:.2}s on wall={:.2}s tokens {} vs {}",
                    o.wall_s, s.wall_s, o.completion_tokens, s.completion_tokens
                );
                assert_eq!(o.content, s.content, "bench prompt {i}: streams diverged");
            }
        }
    }
}
