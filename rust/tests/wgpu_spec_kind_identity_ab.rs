#[cfg(not(feature = "wgpu"))]
#[test]
fn wgpu_spec_kind_identity_ab_is_cfg_out_without_the_wgpu_feature() {
    eprintln!(
        "wgpu_spec_kind_identity_ab compiled OUT (no `wgpu` feature). This is a SKIP, not a pass: \
         a cfg-out prints 0 passed AND 0 ignored. Re-run with \
         NVK_PKG=speaches-plus NVK_FEATURES=cuda,wgpu."
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
    use speaches_plus::oapi::chat_engine_wgpu::{spec, WgpuChatEngine};

    static ENV_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

    pub const IDENTITY_IS_THE_ONLY_ACCEPTANCE_BAR_FOR_A_CHAIN_ROUTE: &str =
        "the chain route accepts a drafted token only where the verifier's own argmax agrees with \
         it, so a greedy completion served with the route on must be byte-identical to the same \
         completion served with it off; a single differing byte means the multi-row verify forward \
         does not reproduce the M=1 stepping path it replaces";

    const GATE_ENV: &str = "NV_WGPU_SPEC_KIND_AB_TEST";

    fn enabled() -> bool {
        std::env::var(GATE_ENV).ok().as_deref() == Some("1")
    }

    fn hub_roots() -> Vec<PathBuf> {
        let mut out: Vec<PathBuf> = Vec::new();
        if let Ok(v) = std::env::var("HF_HUB_CACHE") {
            out.push(PathBuf::from(v));
        }
        if let Ok(h) = std::env::var("HOME") {
            out.push(PathBuf::from(h).join(".cache/huggingface/hub"));
        }
        out
    }

    fn snapshot(repo: &str) -> Option<PathBuf> {
        for root in hub_roots() {
            let Ok(entries) = std::fs::read_dir(root.join(repo).join("snapshots")) else {
                continue;
            };
            let mut cand: Vec<PathBuf> = entries
                .flatten()
                .map(|e| e.path())
                .filter(|p| p.join("config.json").exists() && p.join("tokenizer.json").exists())
                .collect();
            cand.sort();
            if let Some(p) = cand.pop() {
                return Some(p);
            }
        }
        None
    }

    fn model_dir(dir_env: &str, repo: &str) -> Option<PathBuf> {
        if let Ok(d) = std::env::var(dir_env) {
            let p = PathBuf::from(d);
            return p.join("config.json").exists().then_some(p);
        }
        snapshot(repo)
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
        wall_s: f64,
        stats: spec::SpecStats,
    }

    const PARAGRAPH: &str = "Polynesian wayfinders crossed thousands of miles of open ocean by \
                             reading star paths, swell patterns, and the flight of birds, long \
                             before the magnetic compass reached Europe. ";

    fn prompts() -> Vec<String> {
        vec![
            format!(
                "Repeat the following paragraph word for word, exactly as written, three times \
                 in a row, with no commentary before or after: {PARAGRAPH}"
            ),
            "Write a detailed paragraph about the history of navigation at sea, covering the \
             instruments and the science behind each advance."
                .to_string(),
            "The cat sat on the mat. The cat ran to the red door. The cat sat on the sofa. The \
             cat ran to the green gate. Continue this story, keeping the same rhythm but always \
             choosing new places and colors."
                .to_string(),
        ]
    }

    fn run_config(
        engine: &Arc<WgpuChatEngine>,
        label: &str,
        prompts: &[String],
        max_tokens: u32,
    ) -> Vec<RunOut> {
        let handle = engine.clone() as Arc<dyn ChatEngine>;
        let app = app(handle.clone());
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap();
        rt.block_on(async move {
            let mut out = Vec::new();
            for (i, prompt) in prompts.iter().enumerate() {
                let body = format!(
                    r#"{{"model":"{}","max_tokens":{max_tokens},"temperature":0,"seed":7,
                         "messages":[{{"role":"user","content":{}}}]}}"#,
                    handle.model_id(),
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
                let stats = engine.last_spec_stats();
                eprintln!(
                    "[{label}#{i}] completion_tokens={completion_tokens} wall={wall_s:.2}s \
                     spec[{}]",
                    stats.summary()
                );
                assert!(completion_tokens > 0, "[{label}#{i}] no tokens generated");
                out.push(RunOut {
                    content,
                    completion_tokens,
                    wall_s,
                    stats,
                });
            }
            out
        })
    }

    fn identity_ab(kind_slug: &str, dir_env: &str, repo: &str, max_tokens: u32) {
        if !enabled() {
            panic!(
                "this test is #[ignore]d, so it was asked for BY NAME, but {GATE_ENV}=1 is not \
                 set. Returning here prints `1 passed` in 0.00s having served nothing. This is a \
                 SKIP, not a pass."
            );
        }
        let Some(dir) = model_dir(dir_env, repo) else {
            panic!(
                "no checkpoint for {kind_slug}: set {dir_env}, or cache {repo}. This is a SKIP, \
                 not a pass."
            )
        };
        let _lock = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let _ = tracing_subscriber::fmt()
            .with_env_filter("speaches_plus=info")
            .with_writer(std::io::stderr)
            .try_init();

        let t0 = std::time::Instant::now();
        std::env::set_var(spec::SPEC_ENV, "0");
        std::env::remove_var(spec::SPEC_KINDS_ENV);
        let engine = Arc::new(
            WgpuChatEngine::load_with(&dir, 2048, None)
                .unwrap_or_else(|err| panic!("[{kind_slug}] wgpu engine did not load: {err:#}")),
        );
        eprintln!(
            "[{kind_slug}] engine ready in {:.1}s from {}",
            t0.elapsed().as_secs_f64(),
            dir.display()
        );

        let prompts = prompts();
        let off = run_config(&engine, &format!("{kind_slug}-spec-off"), &prompts, max_tokens);
        for (i, r) in off.iter().enumerate() {
            assert_eq!(
                r.stats,
                spec::SpecStats::default(),
                "[{kind_slug}-spec-off#{i}] NV_WGPU_SPEC=0 still ran the chain route"
            );
        }

        std::env::remove_var(spec::SPEC_ENV);
        std::env::set_var(spec::SPEC_KINDS_ENV, kind_slug);
        let on = run_config(&engine, &format!("{kind_slug}-spec-on"), &prompts, max_tokens);
        std::env::remove_var(spec::SPEC_KINDS_ENV);

        let mut rounds_with_draft = 0usize;
        let mut accepted = 0usize;
        for (i, (a, b)) in off.iter().zip(on.iter()).enumerate() {
            assert_eq!(
                a.completion_tokens, b.completion_tokens,
                "[{kind_slug}#{i}] token counts diverged. \
                 {IDENTITY_IS_THE_ONLY_ACCEPTANCE_BAR_FOR_A_CHAIN_ROUTE}"
            );
            assert_eq!(
                a.content, b.content,
                "[{kind_slug}#{i}] completions diverged. \
                 {IDENTITY_IS_THE_ONLY_ACCEPTANCE_BAR_FOR_A_CHAIN_ROUTE}"
            );
            assert!(
                b.stats.rounds > 0,
                "[{kind_slug}#{i}] NV_WGPU_SPEC_KINDS={kind_slug} did not put the request on the \
                 chain route: zero spec rounds is a vacuous pass, not an identity"
            );
            rounds_with_draft += b.stats.rounds_with_draft;
            accepted += b.stats.accepted;
            eprintln!(
                "[{kind_slug}#{i}] identical over {} tokens; off {:.2}s on {:.2}s",
                a.completion_tokens, a.wall_s, b.wall_s
            );
        }
        assert!(
            rounds_with_draft > 0 && accepted > 0,
            "[{kind_slug}] the suffix drafter never proposed an accepted token across the whole \
             prompt set, so the multi-row verify forward was never exercised at width > 1"
        );
    }

    #[test]
    #[ignore]
    fn qwen38_dense_chain_route_is_byte_identical_to_plain_greedy() {
        identity_ab(
            "qwen3.8",
            "NV_QWEN38_DIR",
            "models--unsloth--Qwen3.8-27B-NVFP4",
            192,
        );
    }

    #[test]
    #[ignore]
    fn gpt_oss_chain_route_is_byte_identical_to_plain_greedy() {
        identity_ab(
            "gpt-oss",
            "NV_GPTOSS_DIR",
            "models--openai--gpt-oss-20b",
            192,
        );
    }
}
