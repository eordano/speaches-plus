use std::path::PathBuf;
use std::time::Instant;

use candle_core::{DType, Device, Tensor};

const BENCH_GATE: &str = "NV_VISION_BENCH";
const TIMED_RUNS_AFTER_ONE_WARMUP: usize = 2;

fn require_bench_gate() {
    if std::env::var(BENCH_GATE).as_deref() != Ok("1") {
        panic!("set {BENCH_GATE}=1 to run the vision encode/prefill benches; this #[ignore]d suite must never silently skip");
    }
}

fn hub_snapshot(repo_dirname: &str, need: &[&str]) -> Option<PathBuf> {
    let root = PathBuf::from(std::env::var("HOME").ok()?)
        .join(".cache/huggingface/hub")
        .join(repo_dirname)
        .join("snapshots");
    std::fs::read_dir(root)
        .ok()?
        .flatten()
        .map(|e| e.path())
        .find(|p| need.iter().all(|f| p.join(f).exists()))
}

fn dir_from_env_or_hub(env_key: &str, repo_dirname: &str, need: &[&str]) -> PathBuf {
    if let Ok(d) = std::env::var(env_key) {
        if !d.is_empty() {
            let p = PathBuf::from(&d);
            assert!(p.join("config.json").exists(), "{env_key}={d} has no config.json");
            return p;
        }
    }
    hub_snapshot(repo_dirname, need)
        .unwrap_or_else(|| panic!("no complete {repo_dirname} snapshot in the hub cache; set {env_key}"))
}

fn qwen38_dir() -> PathBuf {
    dir_from_env_or_hub(
        "NV_QWEN38_DIR",
        "models--unsloth--Qwen3.8-27B-NVFP4",
        &["config.json", "model.safetensors"],
    )
}

fn gemma4_31b_dir() -> PathBuf {
    dir_from_env_or_hub(
        "NV_G4_MM_DIR",
        "models--nvidia--Gemma-4-31B-IT-NVFP4",
        &["config.json", "model.safetensors.index.json"],
    )
}

fn tower_device() -> Device {
    #[cfg(feature = "cuda")]
    {
        Device::new_cuda(0).expect("cuda device for the vision tower bench")
    }
    #[cfg(not(feature = "cuda"))]
    {
        Device::Cpu
    }
}

fn synthetic_rgb(w: u32, h: u32) -> image::RgbImage {
    image::RgbImage::from_fn(w, h, |x, y| {
        image::Rgb([
            ((x * 7 + y * 3) % 251) as u8,
            ((x * 13 + y * 5) % 241) as u8,
            ((x * 3 + y * 11) % 233) as u8,
        ])
    })
}

fn normalized_chw_tensor(h: usize, w: usize, device: &Device) -> Tensor {
    let mut pixels = vec![0f32; 3 * h * w];
    for y in 0..h {
        for x in 0..w {
            let ch = [
                ((x * 7 + y * 3) % 251) as f32 / 251.0,
                ((x * 13 + y * 5) % 241) as f32 / 241.0,
                ((x * 3 + y * 11) % 233) as f32 / 233.0,
            ];
            for (c, v) in ch.iter().enumerate() {
                pixels[c * h * w + y * w + x] = (v - 0.5) / 0.5;
            }
        }
    }
    Tensor::from_vec(pixels, (1, 3, h, w), device).expect("pixel tensor")
}

fn sync_read(t: &Tensor) -> f32 {
    t.to_dtype(DType::F32)
        .and_then(|x| x.sum_all())
        .and_then(|x| x.to_scalar::<f32>())
        .expect("device sync via scalar readback")
}

fn timed_ms(mut f: impl FnMut() -> f32) -> (Vec<f64>, f32) {
    let mut sink = f();
    let mut runs = Vec::with_capacity(TIMED_RUNS_AFTER_ONE_WARMUP);
    for _ in 0..TIMED_RUNS_AFTER_ONE_WARMUP {
        let t0 = Instant::now();
        sink = f();
        runs.push(t0.elapsed().as_secs_f64() * 1e3);
    }
    (runs, sink)
}

fn fmt_runs(runs: &[f64]) -> String {
    runs.iter()
        .map(|r| format!("{r:.1}"))
        .collect::<Vec<_>>()
        .join("/")
}

#[test]
#[ignore = "loads the 27-layer qwen3.8 vision tower from the real NVFP4 snapshot; set NV_VISION_BENCH=1"]
fn qwen38_tower_encode_ms_by_resolution() {
    require_bench_gate();
    let dir = qwen38_dir();
    let device = tower_device();
    let cfg = nv_omni::Qwen3VisionConfig::from_hf_config_json(dir.join("config.json"))
        .expect("parse qwen3.8 vision_config");
    let weights = nv_weights::WeightLoader::open_dir(&dir, &device).expect("open snapshot weights");
    let mut tower = nv_omni::Qwen3VisionTower::new_empty(cfg, &device).expect("new_empty");
    tower.load_weights(&weights).expect("load model.visual.* weights");
    for (h, w) in [(448usize, 448usize), (896, 896), (1792, 1792)] {
        let pixels = normalized_chw_tensor(h, w, &device);
        let out = tower.forward(&pixels).expect("tower forward");
        let rows = out.dims()[0];
        let (runs, sink) = timed_ms(|| sync_read(&tower.forward(&pixels).expect("tower forward")));
        assert!(sink.is_finite(), "tower output must stay finite");
        eprintln!(
            "[vision-bench qwen38-tower] input={h}x{w} merged_rows={rows} encode_ms={}",
            fmt_runs(&runs)
        );
    }
}

#[test]
#[ignore = "loads the gemma4-31B vision tower from the real NVFP4 snapshot; set NV_VISION_BENCH=1"]
fn gemma4_tower_encode_ms_by_resolution() {
    require_bench_gate();
    let dir = gemma4_31b_dir();
    let device = tower_device();
    let towers =
        speaches_plus::oapi::chat_multimodal::Gemma4MmTowers::from_model_dir(&dir, &device)
            .expect("load gemma4 mm towers");
    let tower = towers
        .vision
        .as_ref()
        .expect("gemma4-31B checkpoint ships a vision tower");
    for (w, h) in [(336u32, 336u32), (672, 672), (1344, 1344)] {
        let img = synthetic_rgb(w, h);
        let patches = speaches_plus::oapi::chat_multimodal::preprocess_image(
            &img,
            tower.config(),
            &device,
        )
        .expect("gemma4 preprocess");
        let (runs, sink) = timed_ms(|| {
            sync_read(
                &tower
                    .forward(&patches.pixel_values, &patches.position_ids)
                    .expect("tower forward"),
            )
        });
        assert!(sink.is_finite(), "tower output must stay finite");
        eprintln!(
            "[vision-bench gemma4-tower] input={w}x{h} target={}x{} soft_tokens={} encode_ms={}",
            patches.target_width,
            patches.target_height,
            patches.num_soft_tokens,
            fmt_runs(&runs)
        );
    }
}

#[cfg(feature = "wgpu")]
mod e2e {
    use super::*;
    use std::io::Cursor;
    use std::sync::Arc;

    use axum::body::{to_bytes, Body};
    use axum::http::{Request, StatusCode};
    use axum::routing::post;
    use axum::Router;
    use tower::ServiceExt;

    use speaches_plus::oapi::chat::{handle_chat_completions, ChatAppState, ChatEngine};
    use speaches_plus::oapi::chat_engine::ChatRegistry;
    use speaches_plus::oapi::chat_engine_wgpu::WgpuChatEngine;

    fn png_data_url(w: u32, h: u32) -> String {
        let img = synthetic_rgb(w, h);
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

    struct Reply {
        prompt_tokens: u64,
        wall_ms: f64,
    }

    fn post_chat(engine: &Arc<dyn ChatEngine>, body: String, label: &str) -> Reply {
        let router = Router::new()
            .route("/v1/chat/completions", post(handle_chat_completions))
            .layer(axum::extract::DefaultBodyLimit::max(200 * 1024 * 1024))
            .with_state(ChatAppState {
                registry: ChatRegistry::single(engine.clone()),
            });
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
            let t0 = Instant::now();
            let resp = router.oneshot(req).await.unwrap();
            let status = resp.status();
            let bytes = to_bytes(resp.into_body(), 1 << 24).await.unwrap();
            let wall_ms = t0.elapsed().as_secs_f64() * 1e3;
            let text = String::from_utf8_lossy(&bytes).into_owned();
            assert_eq!(status, StatusCode::OK, "[{label}] {text}");
            let v: serde_json::Value = serde_json::from_str(&text).unwrap();
            Reply {
                prompt_tokens: v["usage"]["prompt_tokens"].as_u64().unwrap_or(0),
                wall_ms,
            }
        })
    }

    fn chat_body(model: &str, image: Option<&str>) -> String {
        let mut content = vec![serde_json::json!({
            "type": "text",
            "text": "Describe this image in one short sentence."
        })];
        if let Some(url) = image {
            content.push(serde_json::json!({
                "type": "image_url",
                "image_url": {"url": url}
            }));
        }
        serde_json::json!({
            "model": model,
            "max_tokens": 1,
            "temperature": 0,
            "enable_thinking": false,
            "messages": [{"role": "user", "content": content}]
        })
        .to_string()
    }

    fn bench_engine_prefill(dir: &PathBuf, tag: &str, image_sizes: &[(u32, u32)]) {
        let engine = Arc::new(WgpuChatEngine::load(dir).expect("wgpu engine load"));
        assert!(
            engine.supports_mm_input(),
            "a vision-capable checkpoint must report supports_mm_input()"
        );
        let engine: Arc<dyn ChatEngine> = engine;
        let model = engine.model_id().to_string();

        let text = chat_body(&model, None);
        post_chat(&engine, text.clone(), "text-warmup");
        for run in 0..TIMED_RUNS_AFTER_ONE_WARMUP {
            let r = post_chat(&engine, text.clone(), "text");
            eprintln!(
                "[vision-bench {tag}-e2e] text_only run={run} prompt_tokens={} wall_ms={:.1}",
                r.prompt_tokens, r.wall_ms
            );
        }
        for &(w, h) in image_sizes {
            let body = chat_body(&model, Some(&png_data_url(w, h)));
            post_chat(&engine, body.clone(), "image-warmup");
            for run in 0..TIMED_RUNS_AFTER_ONE_WARMUP {
                let r = post_chat(&engine, body.clone(), "image");
                eprintln!(
                    "[vision-bench {tag}-e2e] image={w}x{h} run={run} prompt_tokens={} wall_ms={:.1}",
                    r.prompt_tokens, r.wall_ms
                );
            }
        }
    }

    #[test]
    #[ignore = "boots the qwen3.8-27B wgpu engine plus the candle vision tower; set NV_VISION_BENCH=1"]
    fn qwen38_e2e_image_prefill_ms() {
        require_bench_gate();
        bench_engine_prefill(&qwen38_dir(), "qwen38", &[(448, 448), (896, 896), (1792, 1792)]);
    }

    #[test]
    #[ignore = "boots the gemma4-31B wgpu engine plus the candle vision tower; set NV_VISION_BENCH=1"]
    fn gemma4_31b_e2e_image_prefill_ms() {
        require_bench_gate();
        bench_engine_prefill(&gemma4_31b_dir(), "gemma4-31b", &[(336, 336), (672, 672), (1344, 1344)]);
    }
}
