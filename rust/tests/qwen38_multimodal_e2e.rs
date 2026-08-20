#[cfg(not(feature = "wgpu"))]
#[test]
fn qwen38_multimodal_e2e_is_cfg_out_without_the_wgpu_feature() {
    eprintln!(
        "qwen38_multimodal_e2e compiled OUT (no `wgpu` feature). This is a SKIP, not a pass. \
         Re-run with --features cuda,wgpu and NV_QWEN38_MM_TEST=1."
    );
}

#[cfg(feature = "wgpu")]
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
    use speaches_plus::oapi::chat_engine::ChatRegistry;
    use speaches_plus::oapi::chat_engine_wgpu::WgpuChatEngine;

    const GATE: &str = "NV_QWEN38_MM_TEST";
    const REPO_SUB: &str =
        ".cache/huggingface/hub/models--unsloth--Qwen3.8-27B-NVFP4/snapshots";

    fn require_gate() {
        if std::env::var(GATE).as_deref() != Ok("1") {
            panic!("{GATE}=1 not set; this #[ignore]d suite must never silently skip");
        }
    }

    fn model_dir() -> PathBuf {
        if let Ok(d) = std::env::var("NV_QWEN38_DIR") {
            let p = PathBuf::from(d);
            assert!(p.join("config.json").exists(), "NV_QWEN38_DIR has no config.json");
            return p;
        }
        let root = PathBuf::from(std::env::var("HOME").expect("HOME")).join(REPO_SUB);
        std::fs::read_dir(&root)
            .unwrap_or_else(|_| panic!("no snapshot dir {}; set NV_QWEN38_DIR", root.display()))
            .flatten()
            .map(|e| e.path())
            .find(|p| p.join("config.json").exists() && p.join("model.safetensors").exists())
            .unwrap_or_else(|| panic!("no complete snapshot under {}", root.display()))
    }

    fn red_png_data_url(w: u32, h: u32) -> String {
        let img = image::RgbImage::from_pixel(w, h, image::Rgb([220, 20, 20]));
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

    fn capital_text_body(model: &str) -> String {
        serde_json::json!({
            "model": model,
            "max_tokens": 64,
            "temperature": 0,
            "enable_thinking": false,
            "messages": [{"role": "user", "content": "What is the capital of France? Answer in one short sentence."}]
        })
        .to_string()
    }

    #[test]
    #[ignore = "records the pre-change text-only reply for the mm gate; set NV_QWEN38_MM_TEST=1 and NV_QWEN38_MM_TEXT_BASELINE_RECORD, run with the pre-change engine knobs"]
    fn record_text_only_baseline_reply() {
        require_gate();
        let out = std::env::var("NV_QWEN38_MM_TEXT_BASELINE_RECORD")
            .expect("set NV_QWEN38_MM_TEXT_BASELINE_RECORD to the output path");
        let dir = model_dir();
        let engine = Arc::new(WgpuChatEngine::load(&dir).expect("wgpu engine load"));
        let engine: Arc<dyn ChatEngine> = engine;
        let model = engine.model_id().to_string();
        let t = post_body(&engine, capital_text_body(&model), "baseline");
        std::fs::write(&out, &t.content).unwrap_or_else(|e| panic!("write {out}: {e}"));
        eprintln!(
            "[qwen38-mm baseline] wrote {} bytes to {out}: {:?}",
            t.content.len(),
            t.content
        );
    }

    #[test]
    #[ignore = "boots the 22.5 GB Qwen3.8-27B-NVFP4 wgpu engine + candle vision tower; set NV_QWEN38_MM_TEST=1"]
    fn qwen38_image_url_names_the_color_and_text_is_bit_identical() {
        require_gate();
        let dir = model_dir();
        let engine = Arc::new(WgpuChatEngine::load(&dir).expect("wgpu engine load"));
        assert!(
            engine.supports_mm_input(),
            "a vision-capable checkpoint must report supports_mm_input()"
        );
        let engine: Arc<dyn ChatEngine> = engine;
        let model = engine.model_id().to_string();

        let data_url = red_png_data_url(448, 448);
        let img_body = serde_json::json!({
            "model": model,
            "max_tokens": 16,
            "temperature": 0,
            "enable_thinking": false,
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": "What color is this image? Answer with one word."},
                    {"type": "image_url", "image_url": {"url": data_url}}
                ]
            }]
        })
        .to_string();
        let r = post_body(&engine, img_body, "image");
        eprintln!(
            "[qwen38-mm image] model={model} prompt_tokens={} completion_tokens={} wall={:.2}s content={:?}",
            r.prompt_tokens, r.completion_tokens, r.wall_s, r.content
        );
        assert!(
            r.content.to_lowercase().contains("red"),
            "grounded color answer expected, got {:?}",
            r.content
        );

        let t1 = post_body(&engine, capital_text_body(&model), "text-1");
        let t2 = post_body(&engine, capital_text_body(&model), "text-2");
        assert_eq!(
            t1.content, t2.content,
            "text-only greedy output must be deterministic across two runs"
        );
        let baseline_path = std::env::var("NV_QWEN38_MM_TEXT_BASELINE").unwrap_or_else(|_| {
            panic!(
                "set NV_QWEN38_MM_TEXT_BASELINE to a file recorded BEFORE this change by running \
                 the same text-only request through the current main engine; the no-mm regression \
                 gate never silently passes. Observed reply: {:?}",
                t1.content
            )
        });
        let baseline = std::fs::read_to_string(&baseline_path)
            .unwrap_or_else(|e| panic!("read baseline {baseline_path}: {e}"));
        assert_eq!(
            t1.content, baseline,
            "text-only reply drifted from the pre-change recording at {baseline_path}"
        );
    }
}
