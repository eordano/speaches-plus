#[cfg(not(feature = "cuda"))]
#[test]
fn gemma4_mm_e2e_is_cfg_out_without_the_cuda_feature() {
    eprintln!(
        "gemma4_mm_e2e compiled OUT (no `cuda` feature). This is a SKIP, not a pass. \
         Re-run with --features cuda,wgpu and NV_GEMMA4_MM_E2E_TEST=1."
    );
}

#[cfg(feature = "cuda")]
mod gated {
    use std::io::Cursor;
    use std::path::PathBuf;
    use std::sync::Arc;

    use axum::body::{to_bytes, Body};
    use axum::http::{Request, StatusCode};
    use axum::routing::post;
    use axum::Router;
    use tower::ServiceExt;

    use speaches_plus::oapi::chat::{handle_chat_completions, ChatAppState, ChatEngine};
    use speaches_plus::oapi::chat_engine::{ChatRegistry, NvEngineChat};

    const GATE: &str = "NV_GEMMA4_MM_E2E_TEST";

    fn require_gate() {
        if std::env::var(GATE).as_deref() != Ok("1") {
            panic!("{GATE}=1 not set; this #[ignore]d suite must never silently skip");
        }
    }

    fn snapshot_dir(env_var: &str, hub_repo: &str) -> PathBuf {
        if let Ok(d) = std::env::var(env_var) {
            let p = PathBuf::from(d);
            assert!(
                p.join("config.json").is_file(),
                "{env_var}={} has no config.json",
                p.display()
            );
            return p;
        }
        let root = PathBuf::from(std::env::var("HOME").expect("HOME"))
            .join(".cache/huggingface/hub")
            .join(hub_repo)
            .join("snapshots");
        std::fs::read_dir(&root)
            .unwrap_or_else(|e| {
                panic!(
                    "no snapshot dir {} ({e}); set {env_var} to the checkpoint dir. The cached \
                     cuda devenv exports NV_CHAT_MODEL_DIR/HF_HUB_CACHE over caller env, so this \
                     suite reads its own variables instead",
                    root.display()
                )
            })
            .flatten()
            .map(|e| e.path())
            .find(|p| p.join("config.json").is_file())
            .unwrap_or_else(|| panic!("no complete snapshot under {}", root.display()))
    }

    fn assert_checkpoint_declares_vision(dir: &PathBuf) {
        let raw = std::fs::read_to_string(dir.join("config.json")).expect("read config.json");
        let v: serde_json::Value = serde_json::from_str(&raw).expect("parse config.json");
        let vision = v
            .get("vision_config")
            .or_else(|| v.get("text_config").and_then(|t| t.get("vision_config")));
        assert!(
            matches!(vision, Some(x) if !x.is_null()),
            "{} declares no vision_config: this suite would prove nothing about the mm path",
            dir.display()
        );
    }

    fn solid_png_data_url(w: u32, h: u32, rgb: [u8; 3]) -> String {
        let img = image::RgbImage::from_pixel(w, h, image::Rgb(rgb));
        let mut buf = Vec::new();
        image::DynamicImage::ImageRgb8(img)
            .write_to(&mut Cursor::new(&mut buf), image::ImageFormat::Png)
            .unwrap();
        use base64::Engine as _;
        format!(
            "data:image/png;base64,{}",
            base64::engine::general_purpose::STANDARD.encode(&buf)
        )
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
        let rt = tokio::runtime::Builder::new_multi_thread()
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
            let t = std::time::Instant::now();
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

    fn image_body(model: &str, data_url: &str) -> String {
        serde_json::json!({
            "model": model,
            "max_tokens": 16,
            "temperature": 0,
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": "What color is this image? Answer with one word."},
                    {"type": "image_url", "image_url": {"url": data_url}}
                ]
            }]
        })
        .to_string()
    }

    fn text_body(model: &str) -> String {
        serde_json::json!({
            "model": model,
            "max_tokens": 64,
            "temperature": 0,
            "messages": [{
                "role": "user",
                "content": "What is the capital of France? Answer in one short sentence."
            }]
        })
        .to_string()
    }

    fn image_grounding_and_text_path_untouched(arm: &str, dir: PathBuf) {
        require_gate();
        assert_checkpoint_declares_vision(&dir);
        let engine = Arc::new(
            NvEngineChat::try_load(&dir)
                .unwrap_or_else(|e| panic!("[{arm}] load {}: {e:#}", dir.display())),
        );
        assert!(
            engine.supports_mm_input(),
            "[{arm}] a vision-carrying checkpoint must report supports_mm_input(); false here \
             means the towers did not load and image parts would silently route to the \
             perception bridge instead of this engine's own mm path"
        );
        let engine: Arc<dyn ChatEngine> = engine;
        let model = engine.model_id().to_string();

        let t_before = post_body(&engine, text_body(&model), "text-before-image");

        let img = post_body(
            &engine,
            image_body(&model, &solid_png_data_url(448, 448, [220, 20, 20])),
            "image",
        );
        eprintln!(
            "[{arm} image] model={model} prompt_tokens={} completion_tokens={} wall={:.2}s \
             content={:?}",
            img.prompt_tokens, img.completion_tokens, img.wall_s, img.content
        );
        assert!(
            img.prompt_tokens > t_before.prompt_tokens + 64,
            "[{arm}] the image request billed {} prompt tokens against {} for text alone: the \
             soft-token run never expanded, so no embeds were spliced",
            img.prompt_tokens,
            t_before.prompt_tokens
        );
        assert!(
            img.content.to_lowercase().contains("red"),
            "[{arm}] grounded color answer expected, got {:?}",
            img.content
        );

        let t_after = post_body(&engine, text_body(&model), "text-after-image");
        assert_eq!(
            t_before.content, t_after.content,
            "[{arm}] the text-only reply changed after an image request went through the same \
             engine: the mm path perturbed shared state"
        );
        assert_eq!(
            t_before.prompt_tokens, t_after.prompt_tokens,
            "[{arm}] text-only prompt token count drifted across requests"
        );
        if let Ok(path) = std::env::var("NV_GEMMA4_MM_TEXT_BASELINE") {
            let baseline = std::fs::read_to_string(&path)
                .unwrap_or_else(|e| panic!("read baseline {path}: {e}"));
            assert_eq!(
                t_before.content, baseline,
                "[{arm}] text-only reply drifted from the pre-change recording at {path}"
            );
        } else {
            eprintln!(
                "[{arm} text] NV_GEMMA4_MM_TEXT_BASELINE unset: this run checks only \
                 within-process stability, NOT byte-identity against pre-change main. Record \
                 {:?} on main to turn it into a regression gate.",
                t_before.content
            );
        }
    }

    #[test]
    #[ignore = "boots the dense Gemma-4-31B checkpoint plus its vision tower on CUDA; set NV_GEMMA4_MM_E2E_TEST=1 and NV_GEMMA4_DENSE_DIR"]
    fn dense_gemma4_serves_image_url_and_leaves_the_text_path_alone() {
        let dir = snapshot_dir("NV_GEMMA4_DENSE_DIR", "models--nvidia--Gemma-4-31B-IT-NVFP4");
        image_grounding_and_text_path_untouched("dense-31b", dir);
    }

    #[test]
    #[ignore = "boots the Gemma-4-26B-A4B MoE checkpoint plus its vision tower on CUDA; set NV_GEMMA4_MM_E2E_TEST=1 and NV_GEMMA4_MOE_DIR"]
    fn moe_gemma4_serves_image_url_and_leaves_the_text_path_alone() {
        let dir = snapshot_dir("NV_GEMMA4_MOE_DIR", "models--google--gemma-4-26B-A4B-it");
        image_grounding_and_text_path_untouched("moe-26b-a4b", dir);
    }
}
