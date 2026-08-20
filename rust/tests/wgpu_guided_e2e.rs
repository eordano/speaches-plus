#[cfg(not(feature = "wgpu"))]
#[test]
fn wgpu_guided_e2e_is_cfg_out_without_the_wgpu_feature() {
    eprintln!(
        "wgpu_guided_e2e compiled OUT (no `wgpu` feature). This is a SKIP, not a pass: a \
         cfg-out prints 0 passed AND 0 ignored. Re-run with \
         NVK_PKG=speaches-plus NVK_FEATURES=wgpu and NV_WGPU_SERVE_TEST=1."
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

    const DEFAULT_SNAPSHOT: &str = "models--google--gemma-4-E4B-it/snapshots/\
                                    83df0a889143b1dbfc61b591bbc639540fd9ce4c";

    fn enabled() -> bool {
        std::env::var("NV_WGPU_SERVE_TEST").ok().as_deref() == Some("1")
    }

    fn model_dir() -> Option<PathBuf> {
        if let Ok(d) = std::env::var("NV_WGPU_SERVE_DIR") {
            let p = PathBuf::from(d);
            return p.join("config.json").exists().then_some(p);
        }
        let home = std::env::var("HOME").ok()?;
        let p = PathBuf::from(home)
            .join(".cache/huggingface/hub")
            .join(DEFAULT_SNAPSHOT);
        p.join("config.json").exists().then_some(p)
    }

    fn max_seq() -> usize {
        std::env::var("NV_WGPU_SERVE_MAXSEQ")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(512)
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

    fn load_engine(dir: &std::path::Path) -> Arc<dyn ChatEngine> {
        Arc::new(
            WgpuChatEngine::load_with(dir, max_seq(), None).unwrap_or_else(|err| {
                panic!(
                    "NV_WGPU_SERVE_TEST=1 was set but the wgpu engine did not load: {err:#}\n\
                     A missing adapter here is a FAILURE, not a skip."
                )
            }),
        ) as Arc<dyn ChatEngine>
    }

    fn content_of(body: &str) -> (String, String) {
        let v: serde_json::Value = serde_json::from_str(body).unwrap();
        (
            v["choices"][0]["message"]["content"]
                .as_str()
                .unwrap_or_default()
                .to_string(),
            v["choices"][0]["finish_reason"]
                .as_str()
                .unwrap_or_default()
                .to_string(),
        )
    }

    fn guided_body(model: &str) -> String {
        format!(
            r#"{{"model":"{model}","temperature":0,"max_tokens":96,
                 "response_format":{{"type":"json_schema","json_schema":{{
                     "name":"person",
                     "schema":{{"type":"object",
                                "properties":{{"name":{{"type":"string"}},
                                              "age":{{"type":"integer"}}}},
                                "required":["name","age"]}}}}}},
                 "messages":[{{"role":"user",
                     "content":"Invent a person named Ada who is 36. Reply as JSON."}}]}}"#
        )
    }

    fn assert_schema_valid(content: &str) -> serde_json::Value {
        let parsed: serde_json::Value = serde_json::from_str(content)
            .unwrap_or_else(|e| panic!("guided output is not valid JSON: {e}\n{content:?}"));
        let obj = parsed
            .as_object()
            .unwrap_or_else(|| panic!("guided output is not a JSON object: {content:?}"));
        assert!(obj["name"].is_string(), "name is not a string: {content:?}");
        assert!(
            obj["age"].is_i64() || obj["age"].is_u64(),
            "age is not an integer: {content:?}"
        );
        parsed
    }

    #[test]
    #[ignore]
    fn guided_json_is_schema_valid_and_deterministic_across_engine_reloads() {
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
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap();

        let run_once = |tag: &str| -> String {
            let engine = load_engine(&dir);
            let app = app(engine.clone());
            let body = guided_body(engine.model_id());
            rt.block_on(async {
                let (status, resp) = post_json(&app, body).await;
                assert_eq!(status, StatusCode::OK, "{resp}");
                let (content, finish) = content_of(&resp);
                eprintln!("[guided:{tag}] finish_reason={finish} content: {content:?}");
                assert_schema_valid(&content);
                content
            })
        };

        let first = run_once("boot-1");
        let engine = load_engine(&dir);
        let app2 = app(engine.clone());
        let body = guided_body(engine.model_id());
        let repeat = rt.block_on(async {
            let (status, resp) = post_json(&app2, body).await;
            assert_eq!(status, StatusCode::OK, "{resp}");
            let (content, _) = content_of(&resp);
            content
        });
        assert_eq!(
            first, repeat,
            "greedy guided decode must be identical across engine restarts"
        );
        let second = run_once("boot-3");
        assert_eq!(first, second, "third boot diverged");
    }

    #[test]
    #[ignore]
    fn penalties_measurably_change_the_served_output() {
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
        let engine = load_engine(&dir);
        let app = app(engine.clone());
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap();
        rt.block_on(async move {
            let ask = |extra: &str| {
                format!(
                    r#"{{"model":"{}","temperature":0,"max_tokens":64{extra},
                         "messages":[{{"role":"user",
                             "content":"Repeat the word buffalo eight times."}}]}}"#,
                    engine.model_id()
                )
            };
            let (s1, r1) = post_json(&app, ask("")).await;
            assert_eq!(s1, StatusCode::OK, "{r1}");
            let (plain, _) = content_of(&r1);
            eprintln!("[plain] {plain:?}");

            let (s2, r2) = post_json(&app, ask(",\"repetition_penalty\":1.8")).await;
            assert_eq!(s2, StatusCode::OK, "{r2}");
            let (penalized, _) = content_of(&r2);
            eprintln!("[repetition_penalty=1.8] {penalized:?}");
            assert_ne!(
                plain, penalized,
                "a 1.8 repetition penalty did not change a repetition-bait prompt"
            );

            let (s3, r3) = post_json(&app, ask(",\"frequency_penalty\":1.5")).await;
            assert_eq!(s3, StatusCode::OK, "{r3}");
            let (freq, _) = content_of(&r3);
            eprintln!("[frequency_penalty=1.5] {freq:?}");
            assert_ne!(
                plain, freq,
                "a 1.5 frequency penalty did not change a repetition-bait prompt"
            );

            let (s4, r4) = post_json(&app, ask("")).await;
            assert_eq!(s4, StatusCode::OK, "{r4}");
            let (plain2, _) = content_of(&r4);
            assert_eq!(plain, plain2, "plain greedy stopped being reproducible");
        });
    }
}
