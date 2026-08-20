#[cfg(not(feature = "wgpu"))]
#[test]
fn gemma4_wgpu_mm_e2e_is_cfg_out_without_the_wgpu_feature() {
    eprintln!(
        "gemma4_wgpu_mm_e2e compiled OUT (no `wgpu` feature). This is a SKIP, not a pass. \
         Re-run with --features cuda,wgpu and NV_G4_WGPU_MM_TEST=1."
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

    const GATE: &str = "NV_G4_WGPU_MM_TEST";
    const DIR_ENV: &str = "NV_G4_MM_DIR";
    const E4B_GATE: &str = "NV_G4_E4B_MM_TEST";
    const E4B_DIR_ENV: &str = "NV_G4_E4B_DIR";

    fn require(gate: &str) {
        if std::env::var(gate).as_deref() != Ok("1") {
            panic!("{gate}=1 not set; this #[ignore]d suite must never silently skip");
        }
    }

    fn dir_from(primary: &str, fallback: Option<&str>) -> PathBuf {
        let raw = std::env::var(primary)
            .ok()
            .or_else(|| fallback.and_then(|f| std::env::var(f).ok()))
            .unwrap_or_else(|| {
                panic!("set {primary} (or {fallback:?}) to a gemma4 wgpu checkpoint directory")
            });
        let p = PathBuf::from(raw);
        assert!(
            p.join("config.json").exists(),
            "{} has no config.json",
            p.display()
        );
        p
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

    fn wav_data_url(seconds: f32) -> String {
        let rate = 16000u32;
        let n = (rate as f32 * seconds) as u32;
        let mut bytes: Vec<u8> = Vec::with_capacity(44 + n as usize * 2);
        let data_len = n * 2;
        bytes.extend_from_slice(b"RIFF");
        bytes.extend_from_slice(&(36 + data_len).to_le_bytes());
        bytes.extend_from_slice(b"WAVEfmt ");
        bytes.extend_from_slice(&16u32.to_le_bytes());
        bytes.extend_from_slice(&1u16.to_le_bytes());
        bytes.extend_from_slice(&1u16.to_le_bytes());
        bytes.extend_from_slice(&rate.to_le_bytes());
        bytes.extend_from_slice(&(rate * 2).to_le_bytes());
        bytes.extend_from_slice(&2u16.to_le_bytes());
        bytes.extend_from_slice(&16u16.to_le_bytes());
        bytes.extend_from_slice(b"data");
        bytes.extend_from_slice(&data_len.to_le_bytes());
        for i in 0..n {
            let t = i as f32 / rate as f32;
            let s = (t * 440.0 * std::f32::consts::TAU).sin() * 0.3;
            bytes.extend_from_slice(&((s * 32767.0) as i16).to_le_bytes());
        }
        use base64::Engine as _;
        base64::engine::general_purpose::STANDARD.encode(&bytes)
    }

    fn app(engine: Arc<dyn ChatEngine>) -> Router {
        Router::new()
            .route("/v1/chat/completions", post(handle_chat_completions))
            .with_state(ChatAppState {
                registry: ChatRegistry::single(engine),
            })
    }

    struct Out {
        status: StatusCode,
        body: String,
        content: String,
        prompt_tokens: u64,
        cached_tokens: u64,
        wall_s: f64,
    }

    fn post_body(engine: &Arc<dyn ChatEngine>, body: String) -> Out {
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
            let body = String::from_utf8_lossy(&bytes).into_owned();
            let wall_s = t.elapsed().as_secs_f64();
            let v: serde_json::Value = serde_json::from_str(&body).unwrap_or(serde_json::Value::Null);
            Out {
                status,
                content: v["choices"][0]["message"]["content"]
                    .as_str()
                    .unwrap_or_default()
                    .to_string(),
                prompt_tokens: v["usage"]["prompt_tokens"].as_u64().unwrap_or(0),
                cached_tokens: v["usage"]["prompt_tokens_details"]["cached_tokens"]
                    .as_u64()
                    .unwrap_or(0),
                wall_s,
                body,
            }
        })
    }

    fn text_body(model: &str) -> String {
        serde_json::json!({
            "model": model,
            "max_tokens": 32,
            "temperature": 0,
            "messages": [{"role": "user", "content": "What is the capital of France? Answer in one short sentence."}]
        })
        .to_string()
    }

    #[test]
    #[ignore = "boots a gemma4 wgpu decoder plus the candle vision tower; set NV_G4_WGPU_MM_TEST=1 and NV_G4_MM_DIR (or NV_CHAT_MODEL_DIR)"]
    fn a_gemma4_wgpu_image_reaches_the_embed_row_prefill_and_grounds_the_answer() {
        require(GATE);
        let dir = dir_from(DIR_ENV, Some("NV_CHAT_MODEL_DIR"));
        let engine = Arc::new(WgpuChatEngine::load(&dir).expect("wgpu engine load"));
        assert!(
            engine.supports_mm_input(),
            "a gemma4 checkpoint with a vision_config must report supports_mm_input(); without \
             it chat.rs would route the image into the ocr bridge instead of the embed-row splice"
        );
        let engine: Arc<dyn ChatEngine> = engine;
        let model = engine.model_id().to_string();

        let t1 = post_body(&engine, text_body(&model));
        assert_eq!(t1.status, StatusCode::OK, "{}", t1.body);
        let t2 = post_body(&engine, text_body(&model));
        assert_eq!(
            t1.content, t2.content,
            "text-only greedy output must be deterministic across two runs"
        );

        let parts_body = |rgbs: &[[u8; 3]]| {
            let mut parts = vec![serde_json::json!({
                "type": "text",
                "text": "What color is this image? Answer with one word."
            })];
            for rgb in rgbs {
                parts.push(serde_json::json!({
                    "type": "image_url",
                    "image_url": {"url": solid_png_data_url(448, 448, *rgb)}
                }));
            }
            serde_json::json!({
                "model": model,
                "max_tokens": 16,
                "temperature": 0,
                "messages": [{"role": "user", "content": parts}]
            })
            .to_string()
        };

        let same_text_no_image = post_body(&engine, parts_body(&[]));
        assert_eq!(
            same_text_no_image.status,
            StatusCode::OK,
            "{}",
            same_text_no_image.body
        );
        let red = post_body(&engine, parts_body(&[[220, 20, 20]]));
        assert_eq!(red.status, StatusCode::OK, "{}", red.body);
        let blue = post_body(&engine, parts_body(&[[20, 20, 220]]));
        assert_eq!(blue.status, StatusCode::OK, "{}", blue.body);
        let two = post_body(&engine, parts_body(&[[220, 20, 20], [20, 20, 220]]));
        assert_eq!(two.status, StatusCode::OK, "{}", two.body);
        let first_image_rows = red.prompt_tokens as i64 - same_text_no_image.prompt_tokens as i64;
        let second_image_rows = two.prompt_tokens as i64 - red.prompt_tokens as i64;
        eprintln!(
            "[gemma4-wgpu-mm image] model={model} prompt_tokens text_only={} same_text_no_image={} \
             one_image={} two_images={} rows_per_image={first_image_rows}/{second_image_rows} \
             wall={:.2}s red={:?} blue={:?}",
            t1.prompt_tokens,
            same_text_no_image.prompt_tokens,
            red.prompt_tokens,
            two.prompt_tokens,
            red.wall_s,
            red.content,
            blue.content
        );
        assert_eq!(
            first_image_rows, second_image_rows,
            "each image_url part must bill the SAME placeholder run: the gemma4 splice reserves \
             boi + num_soft_tokens + eoi per image and overwrites exactly that run of gathered \
             rows, so the second image adds the same token count as the first. Unequal \
             increments mean one image was expanded and the other was served as a bare marker"
        );
        assert!(
            first_image_rows >= 200,
            "an image must expand the prompt by a soft-token run of its own; a 448x448 image \
             through the gemma4 vision tower reserves a run of hundreds of placeholder rows \
             between the boi/eoi markers. The same text billed {} tokens with an image against \
             {} without one, so the marker was served as text and no rows were spliced",
            red.prompt_tokens,
            same_text_no_image.prompt_tokens
        );
        assert!(
            !red.content.to_lowercase().contains("transcribed by ocr"),
            "the image must reach the wgpu embed-row splice, not the ocr bridge: {:?}",
            red.content
        );
        let warm = ["red", "pink", "crimson", "scarlet", "maroon", "magenta"];
        let cool = ["blue", "purple", "violet", "indigo", "navy"];
        let hit = |words: &[&str], s: &str| {
            let s = s.to_lowercase();
            words.iter().any(|w| s.contains(w))
        };
        assert!(
            hit(&warm, &red.content),
            "a red image must draw a warm colour word. This gate is on the FAMILY, not the \
             word: the wgpu decoder serves this NVFP4 checkpoint through an int8/128 \
             requantization of attention and ffn, so it need not repeat the exact word the \
             cuda NVFP4 arm records in tests/gemma4_mm_e2e.rs. The row arithmetic itself is \
             pinned bit-exactly by \
             crates/nv-models/tests/gemma4_wgpu_verify.rs::splicing_the_models_own_embedding_rows_is_bit_identical_to_plain_prefill. \
             Got {:?}",
            red.content
        );
        assert!(
            hit(&cool, &blue.content),
            "a blue image must draw a cool colour word; two answers that track their images is \
             what separates a real splice from a lucky prior. Got {:?}",
            blue.content
        );
        assert_ne!(
            red.content.to_lowercase(),
            blue.content.to_lowercase(),
            "two different images produced the same answer: the spliced rows are not reaching \
             the forward pass"
        );
    }

    #[test]
    #[ignore = "boots a gemma4 wgpu decoder plus the candle vision tower; set NV_G4_WGPU_MM_TEST=1 and NV_G4_MM_DIR (or NV_CHAT_MODEL_DIR)"]
    fn a_media_request_leaves_no_prefix_the_next_text_request_can_reuse() {
        require(GATE);
        let dir = dir_from(DIR_ENV, Some("NV_CHAT_MODEL_DIR"));
        let engine: Arc<dyn ChatEngine> =
            Arc::new(WgpuChatEngine::load(&dir).expect("wgpu engine load"));
        let model = engine.model_id().to_string();
        std::env::set_var(
            speaches_plus::oapi::chat_engine_wgpu::PREFIX_REUSE_ENV,
            "1",
        );

        let cold = post_body(&engine, text_body(&model));
        assert_eq!(cold.status, StatusCode::OK, "{}", cold.body);
        let warm = post_body(&engine, text_body(&model));
        assert_eq!(warm.status, StatusCode::OK, "{}", warm.body);
        assert!(
            warm.cached_tokens > 0,
            "text-then-same-text must reuse KV with {}=1, otherwise this suite proves nothing \
             about what a media request leaves behind: {}",
            speaches_plus::oapi::chat_engine_wgpu::PREFIX_REUSE_ENV,
            warm.body
        );

        let image = post_body(
            &engine,
            serde_json::json!({
                "model": model,
                "max_tokens": 8,
                "temperature": 0,
                "messages": [{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "What color is this image? Answer with one word."},
                        {"type": "image_url", "image_url": {"url": solid_png_data_url(448, 448, [220, 20, 20])}}
                    ]
                }]
            })
            .to_string(),
        );
        assert_eq!(image.status, StatusCode::OK, "{}", image.body);

        let after = post_body(&engine, text_body(&model));
        assert_eq!(after.status, StatusCode::OK, "{}", after.body);
        eprintln!(
            "[gemma4-wgpu-mm prefix] cold_cached={} warm_cached={} after_media_cached={} \
             cold={:?} after={:?}",
            cold.cached_tokens, warm.cached_tokens, after.cached_tokens, cold.content, after.content
        );
        assert_eq!(
            after.cached_tokens, 0,
            "a media request must leave the prefix cache empty: its KV rows carry spliced \
             embeddings, and the placeholder token ids are IDENTICAL for every image, so a \
             prefix match after one would replay another image's rows as this prompt's own"
        );
        assert_eq!(
            after.content, cold.content,
            "the text answer after a media request must equal the cold answer; a different \
             answer means the image's KV rows were reused"
        );
        std::env::remove_var(speaches_plus::oapi::chat_engine_wgpu::PREFIX_REUSE_ENV);
    }

    #[test]
    #[ignore = "boots a gemma4 wgpu decoder; set NV_G4_WGPU_MM_TEST=1 and NV_G4_MM_DIR (or NV_CHAT_MODEL_DIR)"]
    fn a_gemma4_wgpu_checkpoint_without_an_audio_tower_refuses_audio_by_name() {
        require(GATE);
        let dir = dir_from(DIR_ENV, Some("NV_CHAT_MODEL_DIR"));
        let raw = std::fs::read_to_string(dir.join("config.json")).expect("config.json");
        let cfg: serde_json::Value = serde_json::from_str(&raw).expect("parse config.json");
        let has_audio = !matches!(
            cfg.get("audio_config"),
            None | Some(serde_json::Value::Null)
        );
        assert!(
            !has_audio,
            "{} ships an audio_config; point {DIR_ENV} at a checkpoint whose audio_config is \
             null to exercise the refusal",
            dir.display()
        );
        let engine: Arc<dyn ChatEngine> =
            Arc::new(WgpuChatEngine::load(&dir).expect("wgpu engine load"));
        let model = engine.model_id().to_string();
        let body = serde_json::json!({
            "model": model,
            "max_tokens": 16,
            "temperature": 0,
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": "What do you hear?"},
                    {"type": "input_audio", "input_audio": {"data": wav_data_url(1.0), "format": "wav"}}
                ]
            }]
        })
        .to_string();
        let r = post_body(&engine, body);
        eprintln!("[gemma4-wgpu-mm audio refusal] status={} body={}", r.status, r.body);
        assert_eq!(
            r.status,
            StatusCode::BAD_REQUEST,
            "audio with no audio tower must be a client error, not a silent text-only serve: {}",
            r.body
        );
        assert!(r.body.contains("audio"), "{}", r.body);
        assert!(
            r.body.contains("gemma4"),
            "the refusal must name the kind: {}",
            r.body
        );
    }

    #[test]
    #[ignore = "boots the gemma4-e4b wgpu decoder to prove its media refusal; set NV_G4_E4B_MM_TEST=1 and NV_G4_E4B_DIR"]
    fn the_e4b_kind_refuses_images_by_name_instead_of_serving_a_marker_only_prompt() {
        require(E4B_GATE);
        let dir = dir_from(E4B_DIR_ENV, None);
        let engine: Arc<dyn ChatEngine> =
            Arc::new(WgpuChatEngine::load(&dir).expect("wgpu engine load"));
        assert!(
            !engine.supports_mm_input(),
            "e4b must not advertise mm input while its decoder has no embed-row prefill entry; \
             advertising it would take the labelled ocr bridge away from the default wgpu model"
        );
        let mut req = speaches_plus::oapi::chat::ChatGenerateRequest {
            prompt: "hello".into(),
            max_new_tokens: 8,
            stop: Vec::new(),
            seed: Some(1),
            temperature: None,
            top_p: None,
            top_k: None,
            min_p: None,
            presence_penalty: None,
            frequency_penalty: None,
            repetition_penalty: None,
            guided: None,
            guided_think_close: None,
            logit_bias: Vec::new(),
            logprobs: false,
            top_logprobs: 0,
            kv_resume: None,
            kv_store: None,
            mm: None,
        };
        req.mm = Some(speaches_plus::oapi::chat_multimodal::MmMedia {
            images: vec![image::RgbImage::new(8, 8)],
            audios: Vec::new(),
        });
        let (tx, _rx) = tokio::sync::mpsc::channel(8);
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap();
        let err = rt
            .block_on(engine.generate(req, tx))
            .expect_err("media on a kind with no embed-row prefill entry must be refused");
        let msg = format!("{err}");
        eprintln!("[gemma4-e4b media refusal] {msg}");
        assert!(msg.contains("gemma4-e4b"), "must name the kind: {msg}");
        assert!(
            msg.contains("prefill_tokens_with_embed_rows"),
            "must name the missing decoder entry: {msg}"
        );
    }
}
