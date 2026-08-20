#[cfg(not(feature = "cuda"))]
#[test]
fn niah_probe_is_cfg_out_without_the_cuda_feature() {
    eprintln!(
        "niah_probe compiled OUT (no `cuda` feature). This is a SKIP, not a pass. \
         Re-run with --features cuda and NV_NIAH_TEST=1."
    );
}

#[cfg(feature = "cuda")]
mod gated {
    use std::path::PathBuf;
    use std::sync::Arc;
    use std::time::Instant;

    use axum::body::{to_bytes, Body};
    use axum::http::{Request, StatusCode};
    use axum::routing::post;
    use axum::Router;
    use tower::ServiceExt;

    use speaches_plus::oapi::chat::{handle_chat_completions, ChatAppState, ChatEngine};
    use speaches_plus::oapi::chat_engine::{ChatRegistry, NvEngineChat};

    const GATE: &str = "NV_NIAH_TEST";
    const LAGUNA_REPO: &str = "poolside/Laguna-XS-2.1-NVFP4";
    const NEEDLE_PASSPHRASE_PREFIX: &str = "orenda-niah";
    const FILLER_UNIT: &str = "The archive keeps routine logs of unrelated errands: grocery \
         lists, weather notes, and traffic updates from the morning commute. ";
    const ENGLISH_PROSE_CHARS_PER_TOKEN_ESTIMATE: usize = 4;

    fn require_gate() {
        if std::env::var(GATE).as_deref() != Ok("1") {
            panic!("{GATE}=1 not set; this #[ignore]d suite must never silently skip");
        }
    }

    fn hub_roots() -> Vec<PathBuf> {
        let mut out: Vec<PathBuf> = Vec::new();
        let mut push = |p: PathBuf| {
            if p.is_dir() && !out.contains(&p) {
                out.push(p);
            }
        };
        if let Ok(v) = std::env::var("HF_HUB_CACHE") {
            push(PathBuf::from(v));
        }
        push(PathBuf::from(std::env::var("HOME").unwrap_or_default()).join(".cache/huggingface/hub"));
        out
    }

    fn model_dir() -> PathBuf {
        if let Ok(d) = std::env::var("NV_NIAH_MODEL_DIR") {
            let p = PathBuf::from(d);
            assert!(p.join("config.json").is_file(), "NV_NIAH_MODEL_DIR has no config.json");
            return p;
        }
        let leaf = format!("models--{}", LAGUNA_REPO.replace('/', "--"));
        for root in hub_roots() {
            let snaps = root.join(&leaf).join("snapshots");
            let Ok(rd) = std::fs::read_dir(&snaps) else {
                continue;
            };
            let mut dirs: Vec<PathBuf> = rd
                .flatten()
                .map(|e| e.path())
                .filter(|p| {
                    p.join("config.json").is_file()
                        && (p.join("model.safetensors").is_file()
                            || p.join("model.safetensors.index.json").is_file())
                })
                .collect();
            dirs.sort();
            if let Some(p) = dirs.into_iter().next() {
                return p;
            }
        }
        panic!(
            "no cached {LAGUNA_REPO} snapshot with weights under any of {:?}; set NV_NIAH_MODEL_DIR",
            hub_roots()
        );
    }

    fn env_usize(key: &str, default: usize) -> usize {
        match std::env::var(key) {
            Ok(v) => v
                .parse()
                .unwrap_or_else(|_| panic!("{key}={v:?} is not a usize")),
            Err(_) => default,
        }
    }

    fn env_bool(key: &str, default: bool) -> bool {
        match std::env::var(key) {
            Ok(v) => matches!(v.as_str(), "1" | "true" | "TRUE" | "yes"),
            Err(_) => default,
        }
    }

    fn depths_for(trials: usize) -> Vec<f64> {
        if let Ok(v) = std::env::var("NV_NIAH_DEPTHS") {
            let parsed: Vec<f64> = v
                .split(',')
                .map(|s| {
                    s.trim()
                        .parse()
                        .unwrap_or_else(|_| panic!("NV_NIAH_DEPTHS entries must be floats in [0,1], got {s:?}"))
                })
                .collect();
            assert_eq!(
                parsed.len(),
                trials,
                "NV_NIAH_DEPTHS must supply exactly NV_NIAH_TRIALS={trials} depths"
            );
            return parsed;
        }
        if trials == 1 {
            return vec![0.5];
        }
        (0..trials)
            .map(|i| 0.1 + 0.8 * (i as f64) / ((trials - 1) as f64))
            .collect()
    }

    fn passphrase_for(idx: usize, depth: f64) -> String {
        format!("{NEEDLE_PASSPHRASE_PREFIX}-{idx:02}-{:03}", (depth * 1000.0).round() as u32)
    }

    fn haystack_prompt(idx: usize, depth: f64, target_tokens: usize) -> (String, String) {
        let passphrase = passphrase_for(idx, depth);
        let target_chars = target_tokens * ENGLISH_PROSE_CHARS_PER_TOKEN_ESTIMATE;
        let mut filler = String::new();
        while filler.len() < target_chars {
            filler.push_str(FILLER_UNIT);
        }
        let mut insert_at = ((filler.len() as f64) * depth).round() as usize;
        insert_at = insert_at.min(filler.len());
        while insert_at < filler.len() && !filler.is_char_boundary(insert_at) {
            insert_at += 1;
        }
        let (head, tail) = filler.split_at(insert_at);
        let needle = format!(
            "Pause and note this exactly once: the retrieval passphrase for this session is \
             {passphrase}. Keep reading the unrelated notes below.\n"
        );
        let question = "\n\nIgnore everything above except the one retrieval passphrase stated \
             earlier. What is it? Reply with only the passphrase, nothing else.";
        (format!("{head}{needle}{tail}{question}"), passphrase)
    }

    fn app(engine: Arc<dyn ChatEngine>) -> Router {
        Router::new()
            .route("/v1/chat/completions", post(handle_chat_completions))
            .with_state(ChatAppState {
                registry: ChatRegistry::single(engine),
            })
    }

    struct Out {
        content: String,
        prompt_tokens: u64,
        completion_tokens: u64,
        wall_s: f64,
    }

    fn post_body(engine: &Arc<dyn ChatEngine>, body: String, label: &str) -> Out {
        let router = app(engine.clone());
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap();
        rt.block_on(async move {
            let req = Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header("content-type", "application/json")
                .body(Body::from(body))
                .unwrap();
            let t = Instant::now();
            let resp = router.oneshot(req).await.unwrap();
            let status = resp.status();
            let bytes = to_bytes(resp.into_body(), 1 << 24).await.unwrap();
            let text = String::from_utf8_lossy(&bytes).into_owned();
            let wall_s = t.elapsed().as_secs_f64();
            assert_eq!(status, StatusCode::OK, "[{label}] {text}");
            let v: serde_json::Value = serde_json::from_str(&text).unwrap();
            Out {
                content: v["choices"][0]["message"]["content"]
                    .as_str()
                    .unwrap_or_default()
                    .to_string(),
                prompt_tokens: v["usage"]["prompt_tokens"].as_u64().unwrap_or(0),
                completion_tokens: v["usage"]["completion_tokens"].as_u64().unwrap_or(0),
                wall_s,
            }
        })
    }

    #[test]
    #[ignore = "boots the Laguna-XS-2.1-NVFP4 cuda serving engine; set NV_NIAH_TEST=1"]
    fn niah_probe_recalls_planted_passphrases() {
        require_gate();
        let dir = model_dir();
        let t0 = Instant::now();
        let engine = Arc::new(NvEngineChat::try_load(&dir).expect("load cuda serving engine"));
        eprintln!(
            "[niah_probe] engine loaded in {:.1}s from {}",
            t0.elapsed().as_secs_f64(),
            dir.display()
        );
        let engine: Arc<dyn ChatEngine> = engine;
        let model = engine.model_id().to_string();

        let trials = env_usize("NV_NIAH_TRIALS", 3);
        assert!(trials >= 1, "NV_NIAH_TRIALS must be at least 1");
        let target_tokens = env_usize("NV_NIAH_TOKENS", 4000);
        let max_new = env_usize("NV_NIAH_MAX_NEW_TOKENS", 24);
        let enable_thinking = env_bool("NV_NIAH_ENABLE_THINKING", false);
        let depths = depths_for(trials);

        let mut passed = 0usize;
        let mut failing_labels: Vec<String> = Vec::new();
        for (idx, &depth) in depths.iter().enumerate() {
            let (content, passphrase) = haystack_prompt(idx, depth, target_tokens);
            let body = serde_json::json!({
                "model": model,
                "max_tokens": max_new,
                "temperature": 0,
                "enable_thinking": enable_thinking,
                "messages": [{"role": "user", "content": content}]
            })
            .to_string();
            let label = format!("niah[{idx}] depth={depth:.2}");
            let r = post_body(&engine, body, &label);
            let ok = r.content.contains(&passphrase);
            eprintln!(
                "[niah_probe] {label} passphrase={passphrase:?} prompt_tokens={} \
                 completion_tokens={} wall={:.2}s pass={ok} reply={:?}",
                r.prompt_tokens, r.completion_tokens, r.wall_s, r.content
            );
            assert!(
                r.prompt_tokens as usize >= target_tokens / 2,
                "{label}: haystack collapsed to {} prompt tokens against a {target_tokens}-token \
                 target; the filler-growth loop is broken, this is a harness defect not a \
                 retrieval result",
                r.prompt_tokens
            );
            if ok {
                passed += 1;
            } else {
                failing_labels.push(label);
            }
        }

        eprintln!(
            "[niah_probe] {passed}/{trials} needle trials recalled the planted passphrase on {model}"
        );
        assert_eq!(
            passed, trials,
            "{passed}/{trials} recalled; this suite is the bf16/full-precision retrieval floor \
             any future int4 or other lossy KV-cache tier must clear before shipping, per \
             docs/book/08.1-quality-harness.md -- a REJECT here on the reference precision is a \
             defect in the harness or the serving path, not evidence about a KV tier. Failing: \
             {failing_labels:?}"
        );
    }
}
