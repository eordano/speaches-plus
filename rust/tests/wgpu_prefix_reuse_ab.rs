#[cfg(not(feature = "wgpu"))]
#[test]
fn wgpu_prefix_reuse_ab_is_cfg_out_without_the_wgpu_feature() {
    eprintln!(
        "wgpu_prefix_reuse_ab compiled OUT (no `wgpu` feature). This is a SKIP, not a pass: a \
         cfg-out prints 0 passed AND 0 ignored. Re-run with --features wgpu and \
         NV_WGPU_SERVE_TEST=1."
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
    use speaches_plus::oapi::chat_engine_wgpu::persist::KV_CACHE_DIR_ENV;
    use speaches_plus::oapi::chat_engine_wgpu::{WgpuChatEngine, PREFIX_REUSE_ENV};

    static ENV_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

    struct EnvGuard;

    impl Drop for EnvGuard {
        fn drop(&mut self) {
            std::env::remove_var(PREFIX_REUSE_ENV);
            std::env::remove_var(KV_CACHE_DIR_ENV);
        }
    }

    const MAX_SEQ: usize = 4096;

    const PARAGRAPH: &str = "Polynesian wayfinders crossed thousands of miles of open ocean by \
                             reading star paths, swell patterns, and the flight of birds, long \
                             before the magnetic compass reached Europe. Later, the marine \
                             chronometer finally solved the longitude problem, and satellite \
                             positioning eventually made a fix a matter of milliseconds rather \
                             than months of careful dead reckoning. ";

    fn shared_passage() -> String {
        let mut s = String::from("Here is a passage about navigation at sea, repeated so that the \
                                  shared prefix is longer than one prefill chunk: ");
        for _ in 0..12 {
            s.push_str(PARAGRAPH);
        }
        s
    }

    fn prompt_a() -> String {
        format!("{}Summarize this passage in one sentence.", shared_passage())
    }

    fn prompt_b() -> String {
        format!(
            "{}List the three navigation methods this passage names.",
            shared_passage()
        )
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

    async fn serve(
        app: &Router,
        model: &str,
        label: &str,
        prompt: &str,
    ) -> (String, u64, Option<u64>, u64) {
        let body = format!(
            r#"{{"model":"{model}","temperature":0,"max_tokens":48,"seed":7,
                 "messages":[{{"role":"user","content":{}}}]}}"#,
            serde_json::to_string(prompt).unwrap()
        );
        let t = std::time::Instant::now();
        let (status, resp) = post_json(app, body).await;
        let wall = t.elapsed().as_secs_f64();
        assert_eq!(status, StatusCode::OK, "[{label}] {resp}");
        let v: serde_json::Value = serde_json::from_str(&resp).unwrap();
        let content = v["choices"][0]["message"]["content"]
            .as_str()
            .unwrap_or_default()
            .to_string();
        let completion = v["usage"]["completion_tokens"].as_u64().unwrap_or(0);
        let cached = v["usage"]["prompt_tokens_details"]["cached_tokens"].as_u64();
        let prompt_tokens = v["usage"]["prompt_tokens"].as_u64().unwrap_or(0);
        eprintln!(
            "[{label}] completion_tokens={completion} cached_tokens={cached:?} wall={wall:.2}s \
             content={content:?}"
        );
        assert!(completion > 0, "[{label}] no tokens generated");
        (content, completion, cached, prompt_tokens)
    }

    fn text_of((content, completion, ..): &(String, u64, Option<u64>, u64)) -> (String, u64) {
        (content.clone(), *completion)
    }

    #[test]
    #[ignore]
    fn a_reused_prefix_serves_the_completion_a_cold_prefill_serves() {
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
        let _lock = ENV_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let _env = EnvGuard;
        let _ = tracing_subscriber::fmt()
            .with_env_filter("speaches_plus=info")
            .with_writer(std::io::stderr)
            .try_init();

        std::env::remove_var(PREFIX_REUSE_ENV);
        let engine = Arc::new(
            WgpuChatEngine::load_with(&dir, MAX_SEQ, None)
                .unwrap_or_else(|err| panic!("wgpu engine did not load: {err:#}")),
        );
        let model = engine.model_id();
        let app = app(engine.clone() as Arc<dyn ChatEngine>);
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap();

        rt.block_on(async {
            let cold = serve(&app, &model, "cold", &prompt_a()).await;
            let cold_again = serve(&app, &model, "cold-again", &prompt_a()).await;
            assert_eq!(
                engine.prefix_tokens_reused(),
                0,
                "reuse is off by default: {PREFIX_REUSE_ENV} unset must reset every request"
            );
            assert_eq!(
                cold.2, None,
                "a cold request must not report usage.prompt_tokens_details"
            );
            assert_eq!(
                text_of(&cold),
                text_of(&cold_again),
                "this decoder is not reproducible request to request, so a byte-identity claim \
                 about prefix reuse could not mean anything"
            );

            std::env::set_var(PREFIX_REUSE_ENV, "1");
            serve(&app, &model, "warm-b", &prompt_b()).await;
            let before = engine.prefix_tokens_reused();
            let reused = serve(&app, &model, "reused", &prompt_a()).await;
            let gained = engine.prefix_tokens_reused() - before;
            std::env::remove_var(PREFIX_REUSE_ENV);

            assert!(
                gained > 0,
                "the reused request re-prefilled the whole prompt, so this comparison proves \
                 nothing about reuse"
            );
            eprintln!("[reused] prompt tokens served out of the cache: {gained}");
            assert_eq!(
                text_of(&cold),
                text_of(&reused),
                "a prompt served from a rewound prefix must be byte-identical to the same prompt \
                 served cold"
            );
            assert_eq!(
                reused.2,
                Some(gained as u64),
                "usage.prompt_tokens_details.cached_tokens must equal the tokens the engine \
                 actually served out of the cache"
            );
            assert!(
                (gained as u64) < reused.3,
                "cached_tokens {gained} must be strictly less than the {}-token prompt: at \
                 least the final position is always recomputed",
                reused.3
            );
        });
    }

    #[test]
    #[ignore]
    fn a_restored_kv_snapshot_serves_the_completion_a_cold_prefill_serves() {
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
        let _lock = ENV_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let _env = EnvGuard;
        let _ = tracing_subscriber::fmt()
            .with_env_filter("speaches_plus=info")
            .with_writer(std::io::stderr)
            .try_init();

        let cache_dir = std::env::temp_dir().join(format!("nvkv-ab-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&cache_dir);
        std::fs::create_dir_all(&cache_dir).unwrap();
        std::env::set_var(KV_CACHE_DIR_ENV, &cache_dir);
        std::env::set_var(PREFIX_REUSE_ENV, "1");

        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap();

        let engine = Arc::new(
            WgpuChatEngine::load_with(&dir, MAX_SEQ, None)
                .unwrap_or_else(|err| panic!("wgpu engine did not load: {err:#}")),
        );
        let model = engine.model_id().to_string();
        let app_v = app(engine.clone() as Arc<dyn ChatEngine>);
        let cold = rt.block_on(serve(&app_v, &model, "persist-cold", &prompt_a()));
        assert_eq!(
            engine.prefix_tokens_reused(),
            0,
            "an empty cache dir must not warm anything"
        );
        drop(app_v);
        drop(engine);

        let snapshot: Vec<std::path::PathBuf> = std::fs::read_dir(&cache_dir)
            .unwrap()
            .flatten()
            .map(|e| e.path())
            .filter(|p| p.extension().is_some_and(|x| x == "nvkv"))
            .collect();
        assert_eq!(
            snapshot.len(),
            1,
            "dropping the engine must leave exactly one kv snapshot in {}",
            cache_dir.display()
        );

        let engine = Arc::new(
            WgpuChatEngine::load_with(&dir, MAX_SEQ, None)
                .unwrap_or_else(|err| panic!("wgpu engine did not load: {err:#}")),
        );
        let app_v = app(engine.clone() as Arc<dyn ChatEngine>);
        let restored = rt.block_on(serve(&app_v, &model, "persist-restored", &prompt_a()));
        let gained = engine.prefix_tokens_reused();
        assert!(
            gained > 0,
            "the restored snapshot warmed nothing: the request re-prefilled the whole prompt"
        );
        eprintln!("[persist-restored] prompt tokens served out of the restored cache: {gained}");
        assert_eq!(
            text_of(&cold),
            text_of(&restored),
            "a prompt served from a disk-restored kv cache must be byte-identical to the same \
             prompt served cold"
        );
        assert_eq!(
            restored.2,
            Some(gained as u64),
            "usage.prompt_tokens_details.cached_tokens must equal the restored tokens actually \
             reused"
        );
        drop(app_v);
        drop(engine);

        let mut bytes = std::fs::read(&snapshot[0]).unwrap();
        let last = bytes.len() - 1;
        bytes[last] ^= 0xff;
        std::fs::write(&snapshot[0], &bytes).unwrap();

        let engine = Arc::new(
            WgpuChatEngine::load_with(&dir, MAX_SEQ, None)
                .unwrap_or_else(|err| panic!("wgpu engine did not load: {err:#}")),
        );
        let app_v = app(engine.clone() as Arc<dyn ChatEngine>);
        let after_corrupt = rt.block_on(serve(&app_v, &model, "persist-corrupt", &prompt_a()));
        assert_eq!(
            engine.prefix_tokens_reused(),
            0,
            "a corrupt snapshot must be rejected, not restored"
        );
        assert_eq!(
            text_of(&cold),
            text_of(&after_corrupt),
            "after rejecting a corrupt snapshot the engine must serve the cold completion"
        );
        drop(app_v);
        drop(engine);

        std::env::remove_var(KV_CACHE_DIR_ENV);
        std::env::remove_var(PREFIX_REUSE_ENV);
        let _ = std::fs::remove_dir_all(&cache_dir);
    }
}
