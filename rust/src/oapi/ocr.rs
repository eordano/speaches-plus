use std::path::{Path, PathBuf};
use std::sync::mpsc::RecvTimeoutError;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use axum::extract::{Multipart, State};
use axum::http::{header, HeaderMap, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::routing::post;
use axum::{Json, Router};
use tracing::{debug, info, warn};

use nv_models::deepseek_ocr::decoder::prof as dsocr_prof;
use nv_models::deepseek_ocr::{DecoderPrecision, DeepSeekOcr2Pipeline, ResolutionMode};
use nv_models::dots_ocr::{DotsMode, DotsOcrPipeline, GenerateOptions, PixelBudget, PromptStyle};
use nv_models::got_ocr::{GotMode, GotOcrPipeline};
use nv_ocr::binarize;
use nv_ocr::{
    BackendKind, DeepSeekMode, DeepSeekOcr2Model, DotsOcrMode, DotsOcrModel, GreyImage, LayoutBox,
    LayoutPageResult, OcrEngine, OcrResult, ResolutionHint,
};

use super::gate::SurfaceGate;
use super::ocr_batch::{decode_rgb, BatchOptions, DsocrScheduler, JobInput, OcrJob};
use super::{kind, openai_error};

#[derive(Clone, Default)]
pub struct OcrAppState {
    pub tesseract: Option<Arc<OcrEngine>>,
    pub deepseek: Option<Arc<OcrEngine>>,
    pub dots: Option<Arc<OcrEngine>>,
    pub got: Option<Arc<OcrEngine>>,
}

#[derive(Clone)]
struct OcrRuntime {
    app: OcrAppState,
    generative: Arc<SurfaceGate>,
    classical: Option<Arc<SurfaceGate>>,
}

impl OcrRuntime {
    fn gate_for(&self, backend: Backend) -> Option<&SurfaceGate> {
        match backend {
            Backend::Tesseract => self.classical.as_deref(),
            Backend::DeepSeek | Backend::Dots | Backend::Got => Some(self.generative.as_ref()),
        }
    }

    #[cfg(test)]
    fn with_gates(
        app: OcrAppState,
        generative: SurfaceGate,
        classical: Option<SurfaceGate>,
    ) -> Self {
        Self {
            app,
            generative: Arc::new(generative),
            classical: classical.map(Arc::new),
        }
    }
}

pub fn router_with_classical_gate(state: OcrAppState, classical: Arc<SurfaceGate>) -> Router {
    let mut rt = OcrRuntime::from(state);
    rt.classical = Some(classical);
    Router::new()
        .route("/v1/ocr", post(handle_ocr))
        .with_state(rt)
}

impl From<OcrAppState> for OcrRuntime {
    fn from(app: OcrAppState) -> Self {
        let generative = Arc::new(SurfaceGate::from_env(
            "/v1/ocr",
            "NV_OCR_CONCURRENCY",
            "NV_OCR_QUEUE_MS",
            1,
            3_000,
        ));
        let classical = std::env::var_os("NV_OCR_CLASSICAL_CONCURRENCY").map(|_| {
            Arc::new(SurfaceGate::from_env(
                "/v1/ocr (classical)",
                "NV_OCR_CLASSICAL_CONCURRENCY",
                "NV_OCR_QUEUE_MS",
                1,
                3_000,
            ))
        });
        if classical.is_none() {
            info!(
                "/v1/ocr classical backend is ungated: it takes no lock, recognizes lines on \
                 NV_OCR_THREADS scoped threads, and 8-way concurrency is an existing contract. \
                 Set NV_OCR_CLASSICAL_CONCURRENCY to bound it too."
            );
        }
        Self {
            app,
            generative,
            classical,
        }
    }
}

pub fn router_from_env() -> Router {
    let state = OcrAppState {
        tesseract: load_tesseract_from_env(),
        deepseek: load_deepseek_from_env(),
        dots: load_dots_from_env(),
        got: load_got_from_env(),
    };
    log_default_backend_at_boot(&state);
    router(state)
}

fn log_default_backend_at_boot(state: &OcrAppState) {
    let loaded = loaded_backends(state);
    let configured = configured_default_backend();
    match pick_default_backend(&loaded, configured) {
        Ok(b) => info!(
            default_backend = backend_name(b),
            loaded = %describe_backends(&loaded),
            source = if configured.is_some() {
                "NV_OCR_DEFAULT_BACKEND"
            } else {
                "only-loaded-backend"
            },
            "/v1/ocr default backend resolved (requests without backend= are answered by this one)"
        ),
        Err(err) if err.code == "ocr_backend_ambiguous" => warn!(
            loaded = %describe_backends(&loaded),
            "/v1/ocr has NO default backend: {}",
            err.message
        ),
        Err(err) => warn!(
            loaded = %describe_backends(&loaded),
            "/v1/ocr has no usable default backend: {}",
            err.message
        ),
    }
}

pub fn router(state: OcrAppState) -> Router {
    Router::new()
        .route("/v1/ocr", post(handle_ocr))
        .with_state(OcrRuntime::from(state))
}

fn tessdata_root() -> Option<PathBuf> {
    if let Ok(p) = std::env::var("NV_OCR_TESSDATA") {
        return Some(PathBuf::from(p));
    }
    let home = std::env::var("HOME").ok()?;
    let cache = PathBuf::from(home).join(".cache/ocr-testdata");
    cache.exists().then_some(cache)
}

pub fn resolve_traineddata(root: &Path) -> PathBuf {
    if root.is_file() {
        return root.to_path_buf();
    }
    for candidate in [
        "eng.traineddata",
        "tessdata_best/eng.traineddata",
        "tessdata_fast/eng.traineddata",
    ] {
        let p = root.join(candidate);
        if p.is_file() {
            return p;
        }
    }
    root.to_path_buf()
}

fn load_tesseract_from_env() -> Option<Arc<OcrEngine>> {
    let Some(root) = tessdata_root() else {
        info!("NV_OCR_TESSDATA not set and no ~/.cache/ocr-testdata -- /v1/ocr tesseract backend disabled");
        return None;
    };
    let path = resolve_traineddata(&root);
    match OcrEngine::from_traineddata(&path, BackendKind::Classical) {
        Ok(engine) => {
            info!(path = %path.display(), "ocr tesseract backend loaded");
            Some(Arc::new(engine))
        }
        Err(err) => {
            warn!(error = %err, path = %path.display(), "ocr tesseract backend load failed; disabled");
            None
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PageMetrics {
    pub width: usize,
    pub height: usize,
    pub bands: usize,
    pub band_px: f32,
    pub scale: f32,
    pub scaled_band_px: f32,
    pub ink_frac: f32,
    pub ds_stroke_px: f32,
    pub ds_separability: f32,
    pub ds_acutance: f32,
}

pub fn downscale_box(g: &GreyImage, scale: f32) -> GreyImage {
    if scale >= 1.0 || g.w == 0 || g.h == 0 {
        return g.clone();
    }
    let nw = ((g.w as f32 * scale).round() as usize).max(1);
    let nh = ((g.h as f32 * scale).round() as usize).max(1);
    let mut out = GreyImage::new(nw, nh);
    for oy in 0..nh {
        let y0 = oy * g.h / nh;
        let y1 = (((oy + 1) * g.h).div_ceil(nh)).min(g.h).max(y0 + 1);
        for ox in 0..nw {
            let x0 = ox * g.w / nw;
            let x1 = (((ox + 1) * g.w).div_ceil(nw)).min(g.w).max(x0 + 1);
            let mut sum = 0u32;
            let mut n = 0u32;
            for y in y0..y1 {
                for x in x0..x1 {
                    sum += g.data[y * g.w + x] as u32;
                    n += 1;
                }
            }
            out.data[oy * nw + ox] = (sum / n.max(1)) as u8;
        }
    }
    out
}

fn downscaled_stats(ds: &GreyImage) -> (f32, f32, f32) {
    let hist = binarize::histogram(ds);
    let thr = binarize::otsu_threshold(&hist);
    let total: u64 = hist.iter().map(|&c| c as u64).sum();
    if total == 0 {
        return (0.0, 0.0, 0.0);
    }
    let mut w0 = 0u64;
    let mut s0 = 0u64;
    let mut sum_all = 0u64;
    for (v, &c) in hist.iter().enumerate() {
        sum_all += v as u64 * c as u64;
    }
    for (v, &c) in hist.iter().enumerate() {
        if v as u8 <= thr {
            w0 += c as u64;
            s0 += v as u64 * c as u64;
        }
    }
    let w1 = total - w0;
    let mean = sum_all as f64 / total as f64;
    let mut var = 0.0f64;
    for (v, &c) in hist.iter().enumerate() {
        let d = v as f64 - mean;
        var += d * d * c as f64;
    }
    var /= total as f64;
    let separability = if w0 == 0 || w1 == 0 || var <= 0.0 {
        0.0
    } else {
        let m0 = s0 as f64 / w0 as f64;
        let m1 = (sum_all - s0) as f64 / w1 as f64;
        let p0 = w0 as f64 / total as f64;
        let between = p0 * (1.0 - p0) * (m0 - m1) * (m0 - m1);
        (between / var) as f32
    };
    let dark_ink = binarize::ink_is_dark(ds);
    let mut runs: Vec<usize> = Vec::new();
    let mut grad_sum = 0f64;
    let mut grad_n = 0u64;
    for y in 0..ds.h {
        let row = &ds.data[y * ds.w..(y + 1) * ds.w];
        let mut run = 0usize;
        for x in 0..ds.w {
            let is_ink = if dark_ink {
                row[x] <= thr
            } else {
                row[x] > thr
            };
            if is_ink {
                run += 1;
            } else {
                if run > 0 && run <= 24 {
                    runs.push(run);
                }
                run = 0;
            }
            if x + 1 < ds.w {
                grad_sum += (row[x + 1] as i32 - row[x] as i32).abs() as f64;
                grad_n += 1;
            }
        }
        if run > 0 && run <= 24 {
            runs.push(run);
        }
    }
    runs.sort_unstable();
    let stroke = if runs.is_empty() {
        0.0
    } else {
        runs[runs.len() / 2] as f32
    };
    let mut cum = 0u64;
    let (mut p5, mut p95) = (0u8, 255u8);
    for (v, &c) in hist.iter().enumerate() {
        cum += c as u64;
        if cum as f64 <= 0.05 * total as f64 {
            p5 = v as u8;
        }
        if cum as f64 <= 0.95 * total as f64 {
            p95 = v as u8;
        }
    }
    let range = (p95.saturating_sub(p5) as f32).max(1.0);
    let acutance = if grad_n == 0 {
        0.0
    } else {
        (grad_sum / grad_n as f64) as f32 / range
    };
    (stroke, separability, acutance)
}

pub const BASE_VIEW_PX: usize = 1024;
pub const TILE_TRIGGER_PX: usize = 768;

pub fn grey_from_rgb(img: &nv_models::deepseek_ocr::RgbImage) -> GreyImage {
    let mut g = GreyImage::new(img.w, img.h);
    for i in 0..img.w * img.h {
        let r = img.data[i * 3] as u32;
        let gg = img.data[i * 3 + 1] as u32;
        let b = img.data[i * 3 + 2] as u32;
        g.data[i] = ((r * 77 + gg * 150 + b * 29) >> 8) as u8;
    }
    g
}

fn percentile(sorted: &[u32], q: f64) -> u32 {
    if sorted.is_empty() {
        return 0;
    }
    let idx = ((sorted.len() - 1) as f64 * q).round() as usize;
    sorted[idx.min(sorted.len() - 1)]
}

pub fn page_metrics(grey: &GreyImage) -> PageMetrics {
    page_metrics_inner(grey, false)
}

pub fn page_metrics_full(grey: &GreyImage) -> PageMetrics {
    page_metrics_inner(grey, true)
}

fn page_metrics_inner(grey: &GreyImage, with_ds: bool) -> PageMetrics {
    let (w, h) = (grey.w, grey.h);
    let long = w.max(h).max(1);
    let scale = (BASE_VIEW_PX as f32 / long as f32).min(1.0);
    if w == 0 || h == 0 {
        return PageMetrics {
            width: w,
            height: h,
            bands: 0,
            band_px: 0.0,
            scale,
            scaled_band_px: 0.0,
            ink_frac: 0.0,
            ds_stroke_px: 0.0,
            ds_separability: 0.0,
            ds_acutance: 0.0,
        };
    }
    let hist = binarize::histogram(grey);
    let thr = binarize::otsu_threshold(&hist);
    let dark_ink = binarize::ink_is_dark(grey);
    let mut rows: Vec<u32> = Vec::with_capacity(h);
    let mut total_ink: u64 = 0;
    for y in 0..h {
        let row = &grey.data[y * w..(y + 1) * w];
        let mut c = 0u32;
        for &v in row {
            let is_ink = if dark_ink { v <= thr } else { v > thr };
            c += is_ink as u32;
        }
        total_ink += c as u64;
        rows.push(c);
    }
    let mut nonzero: Vec<u32> = rows.iter().copied().filter(|&c| c > 0).collect();
    nonzero.sort_unstable();
    let p95 = percentile(&nonzero, 0.95);
    let cut = ((p95 as f32) * 0.15).max(1.0) as u32;
    let max_band = (h / 6).max(4);
    let mut heights: Vec<usize> = Vec::new();
    let mut run = 0usize;
    for y in 0..=h {
        let on = y < h && rows[y] >= cut;
        if on {
            run += 1;
        } else if run > 0 {
            if run >= 2 && run <= max_band {
                heights.push(run);
            }
            run = 0;
        }
    }
    heights.sort_unstable();
    let band_px = if heights.is_empty() {
        0.0
    } else {
        heights[heights.len() / 2] as f32
    };
    let (ds_stroke_px, ds_separability, ds_acutance) = if with_ds {
        downscaled_stats(&downscale_box(grey, scale))
    } else {
        (0.0, 0.0, 0.0)
    };
    PageMetrics {
        width: w,
        height: h,
        bands: heights.len(),
        band_px,
        scale,
        scaled_band_px: band_px * scale,
        ink_frac: total_ink as f32 / (w as f32 * h as f32),
        ds_stroke_px,
        ds_separability,
        ds_acutance,
    }
}

pub fn auto_max_ink() -> f32 {
    std::env::var("NV_OCR_DSOCR_AUTO_MAX_INK")
        .ok()
        .and_then(|v| v.trim().parse().ok())
        .unwrap_or(0.04)
}

pub fn auto_min_bands() -> usize {
    std::env::var("NV_OCR_DSOCR_AUTO_MIN_BANDS")
        .ok()
        .and_then(|v| v.trim().parse().ok())
        .unwrap_or(4)
}

pub fn decide_auto(m: &PageMetrics) -> (ResolutionMode, &'static str) {
    if m.width <= TILE_TRIGGER_PX && m.height <= TILE_TRIGGER_PX {
        return (ResolutionMode::Base1024, "small-page-tiling-is-a-noop");
    }
    if m.bands < auto_min_bands() {
        return (ResolutionMode::Gundam, "too-few-text-bands");
    }
    if m.ink_frac <= auto_max_ink() {
        (ResolutionMode::Base1024, "sparse-page-survives-downscale")
    } else {
        (ResolutionMode::Gundam, "dense-page-needs-tiles")
    }
}

pub fn default_resolution_hint() -> ResolutionHint {
    match std::env::var("NV_OCR_DSOCR_RESOLUTION").as_deref() {
        Ok(v) => ResolutionHint::parse(v.trim()).unwrap_or(ResolutionHint::Tiled),
        Err(_) => ResolutionHint::Tiled,
    }
}

struct DeepSeekPipelineModel {
    scheduler: DsocrScheduler,
    max_new_tokens: usize,
    default_hint: ResolutionHint,
}

const DSOCR_INFLIGHT_TICK: Duration = Duration::from_secs(15);

fn elapsed_ms(t: Instant) -> f64 {
    t.elapsed().as_secs_f64() * 1e3
}

impl DeepSeekPipelineModel {
    fn run(
        &self,
        image_bytes: &[u8],
        mode: DeepSeekMode,
        hint: ResolutionHint,
        max_new_override: Option<usize>,
    ) -> Result<OcrResult, nv_ocr::Error> {
        let effective = max_new_override.unwrap_or(self.max_new_tokens);
        let requested = hint;
        let hint = match hint {
            ResolutionHint::Auto => self.default_hint,
            other => other,
        };
        let img = decode_rgb(image_bytes).map_err(|e| nv_ocr::Error::Decode(format!("{e:#}")))?;
        let img = super::ocr_orient::maybe_auto_orient(img);
        let (input, res) = match hint {
            ResolutionHint::Tiled => (JobInput::Image(img), ResolutionMode::Gundam),
            ResolutionHint::Base1024 => (JobInput::Image(img), ResolutionMode::Base1024),
            ResolutionHint::Base768 => (JobInput::Image(img), ResolutionMode::Base768),
            ResolutionHint::Auto => {
                let m = page_metrics(&grey_from_rgb(&img));
                let (res, why) = decide_auto(&m);
                info!(
                    width = m.width,
                    height = m.height,
                    bands = m.bands,
                    band_px = m.band_px,
                    scaled_band_px = m.scaled_band_px,
                    picked = ?res,
                    why,
                    "ocr auto resolution"
                );
                (JobInput::Image(img), res)
            }
        };
        let slots = self.scheduler.options().slots;
        info!(
            requested = requested.as_str(),
            effective = hint.as_str(),
            resolution = ?res,
            tiled = res == ResolutionMode::Gundam,
            slots,
            max_new_tokens = effective,
            max_new_tokens_from_request = max_new_override.is_some(),
            bytes = image_bytes.len(),
            "ocr deepseek decode start"
        );
        let t0 = Instant::now();
        let rx = self.scheduler.submit(OcrJob {
            input,
            prompt: mode.prompt().to_string(),
            mode: res,
            max_new_tokens: effective,
        });
        let out = loop {
            match rx.recv_timeout(DSOCR_INFLIGHT_TICK) {
                Ok(r) => break r,
                Err(RecvTimeoutError::Timeout) => {
                    debug!(elapsed_ms = elapsed_ms(t0), "ocr deepseek in flight")
                }
                Err(RecvTimeoutError::Disconnected) => {
                    return Err(nv_ocr::Error::Backend(
                        "deepseek-ocr2: scheduler dropped the job".into(),
                    ))
                }
            }
        };
        let out = out.map_err(|e| nv_ocr::Error::Backend(format!("deepseek-ocr2: {e}")))?;
        let t = out.timings;
        info!(
            elapsed_ms = elapsed_ms(t0),
            queue_ms = t.queue_ms,
            prep_ms = t.prep_ms,
            vision_ms = t.vision_ms,
            decode_ms = t.decode_ms,
            out_tokens = t.out_tokens,
            vision_tokens = t.vision_tokens,
            looped = out.looped,
            hit_eos = out.hit_eos,
            chars = out.text.len(),
            "ocr deepseek decode complete"
        );
        Ok(OcrResult {
            text: out.text,
            tokens: Vec::new(),
            truncated: !out.hit_eos,
            looped: out.looped,
        })
    }
}

impl DeepSeekOcr2Model for DeepSeekPipelineModel {
    fn recognize_page(
        &self,
        image_bytes: &[u8],
        mode: DeepSeekMode,
    ) -> Result<OcrResult, nv_ocr::Error> {
        self.run(image_bytes, mode, ResolutionHint::Auto, None)
    }

    fn recognize_page_hinted(
        &self,
        image_bytes: &[u8],
        mode: DeepSeekMode,
        hint: ResolutionHint,
    ) -> Result<OcrResult, nv_ocr::Error> {
        self.run(image_bytes, mode, hint, None)
    }

    fn recognize_page_budgeted(
        &self,
        image_bytes: &[u8],
        mode: DeepSeekMode,
        hint: ResolutionHint,
        max_new_override: Option<usize>,
    ) -> Result<OcrResult, nv_ocr::Error> {
        self.run(image_bytes, mode, hint, max_new_override)
    }
}

pub const DEEPSEEK_HUB_REPO: &str = "models--deepseek-ai--DeepSeek-OCR-2";

pub fn hub_snapshot_missing_message(env_name: &str, repos: &[&str]) -> String {
    format!(
        "{env_name}=1 but no hub snapshot was found under ~/.cache/huggingface/hub for {}; the \
         backend is DISABLED. Point the matching *_DIR variable at a checkpoint directory, or \
         fetch the snapshot -- otherwise a request naming this backend gets 503, and an \
         unqualified request is answered by whatever else is loaded.",
        repos.join(" or ")
    )
}

fn deepseek_dir() -> Option<PathBuf> {
    if let Ok(d) = std::env::var("NV_OCR_DEEPSEEK_DIR") {
        return Some(PathBuf::from(d));
    }
    if std::env::var("NV_OCR_DEEPSEEK")
        .map(|v| v == "1")
        .unwrap_or(false)
    {
        let found = hub_snapshot(DEEPSEEK_HUB_REPO);
        if found.is_none() {
            tracing::error!(
                "{}",
                hub_snapshot_missing_message("NV_OCR_DEEPSEEK", &[DEEPSEEK_HUB_REPO])
            );
        }
        return found;
    }
    None
}

pub const DOTS_HUB_REPOS: [&str; 2] = [
    "models--dots-studio--dots.ocr",
    "models--rednote-hilab--dots.ocr",
];

fn dots_hub_snapshot() -> Option<PathBuf> {
    DOTS_HUB_REPOS.iter().find_map(|r| hub_snapshot(r))
}

fn hub_snapshot(repo: &str) -> Option<PathBuf> {
    let home = std::env::var("HOME").ok()?;
    let snaps = PathBuf::from(home)
        .join(".cache/huggingface/hub")
        .join(repo)
        .join("snapshots");
    let mut entries: Vec<PathBuf> = std::fs::read_dir(&snaps)
        .ok()?
        .flatten()
        .map(|e| e.path())
        .filter(|p| p.is_dir())
        .collect();
    entries.sort();
    entries.into_iter().next()
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum DevicePref {
    Auto,
    Cpu,
    Metal,
}

fn device_pref(raw: &str) -> DevicePref {
    let raw = raw.trim();
    if raw.eq_ignore_ascii_case("metal") {
        DevicePref::Metal
    } else if raw.eq_ignore_ascii_case("cpu") {
        DevicePref::Cpu
    } else {
        DevicePref::Auto
    }
}

fn ocr_device() -> candle_core::Device {
    let pref = device_pref(&std::env::var("NV_OCR_DEVICE").unwrap_or_default());
    if pref == DevicePref::Cpu {
        return candle_core::Device::Cpu;
    }
    #[cfg(feature = "cuda")]
    if pref == DevicePref::Auto {
        if let Ok(d) = candle_core::Device::new_cuda(0) {
            return d;
        }
    }
    #[cfg(feature = "metal")]
    {
        match candle_core::Device::new_metal(0) {
            Ok(d) => return d,
            Err(err) => warn!(
                error = %err,
                requested = if pref == DevicePref::Metal { "metal" } else { "auto" },
                "metal device unavailable; using cpu"
            ),
        }
    }
    #[cfg(not(feature = "metal"))]
    if pref == DevicePref::Metal {
        warn!("NV_OCR_DEVICE=metal requires the metal feature; using cpu");
    }
    candle_core::Device::Cpu
}

fn ocr_cpu_fallback_enabled() -> bool {
    std::env::var("NV_OCR_CPU_FALLBACK")
        .map(|v| v == "1")
        .unwrap_or(false)
}

fn ocr_chain_appends_cpu(first_is_cpu: bool, cpu_fallback: bool) -> bool {
    !first_is_cpu && cpu_fallback
}

fn ocr_device_chain_from(first: candle_core::Device, cpu_fallback: bool) -> Vec<candle_core::Device> {
    let first_is_cpu = matches!(first, candle_core::Device::Cpu);
    if ocr_chain_appends_cpu(first_is_cpu, cpu_fallback) {
        vec![first, candle_core::Device::Cpu]
    } else {
        vec![first]
    }
}

fn ocr_device_chain() -> Vec<candle_core::Device> {
    ocr_device_chain_from(ocr_device(), ocr_cpu_fallback_enabled())
}

fn device_label(d: &candle_core::Device) -> &'static str {
    match d {
        candle_core::Device::Cpu => "cpu",
        candle_core::Device::Cuda(_) => "cuda",
        candle_core::Device::Metal(_) => "metal",
    }
}

fn load_deepseek_from_env() -> Option<Arc<OcrEngine>> {
    let dir = deepseek_dir()?;
    let precision = match std::env::var("NV_OCR_DEEPSEEK_QUANT").as_deref() {
        Ok("nvfp4") => DecoderPrecision::Nvfp4,
        _ => DecoderPrecision::Bf16,
    };
    let max_new_tokens = std::env::var("NV_OCR_DEEPSEEK_MAX_NEW")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(4096);
    let chain = ocr_device_chain();
    let last = chain.len() - 1;
    for (i, device) in chain.into_iter().enumerate() {
        match DeepSeekOcr2Pipeline::load(&dir, &device, precision) {
            Ok(pipeline) => {
                let batch = BatchOptions::from_env();
                let default_hint = default_resolution_hint();
                info!(
                    dir = %dir.display(),
                    ?precision,
                    device = device_label(&device),
                    fell_back = i > 0,
                    slots = batch.slots,
                    prep_threads = batch.prep_threads,
                    resolution = default_hint.as_str(),
                    "ocr deepseek backend loaded"
                );
                let scheduler = DsocrScheduler::new(Arc::new(pipeline), batch);
                return Some(Arc::new(OcrEngine::from_deepseek(Box::new(
                    DeepSeekPipelineModel {
                        scheduler,
                        max_new_tokens,
                        default_hint,
                    },
                ))));
            }
            Err(err) if i < last => {
                warn!(
                    error = %format!("{err:#}"),
                    dir = %dir.display(),
                    device = device_label(&device),
                    "ocr deepseek backend load failed on this device; retrying on cpu"
                );
            }
            Err(err) => {
                warn!(
                    error = %format!("{err:#}"),
                    dir = %dir.display(),
                    device = device_label(&device),
                    "ocr deepseek backend load failed; disabled"
                );
            }
        }
    }
    None
}

struct DotsPipelineModel {
    pipeline: Mutex<DotsOcrPipeline>,
    max_new_tokens: usize,
}

fn dots_mode(mode: DotsOcrMode) -> DotsMode {
    match mode {
        DotsOcrMode::LayoutAll => DotsMode::LayoutAll,
        DotsOcrMode::LayoutOnly => DotsMode::LayoutOnly,
        DotsOcrMode::PlainOcr => DotsMode::PlainOcr,
    }
}

impl DotsOcrModel for DotsPipelineModel {
    fn recognize_page(
        &self,
        image_bytes: &[u8],
        mode: DotsOcrMode,
    ) -> Result<OcrResult, nv_ocr::Error> {
        Ok(self.recognize_layout(image_bytes, mode)?.result)
    }

    fn recognize_layout(
        &self,
        image_bytes: &[u8],
        mode: DotsOcrMode,
    ) -> Result<nv_ocr::DotsPageResult, nv_ocr::Error> {
        self.recognize_layout_budgeted(image_bytes, mode, None)
    }

    fn recognize_layout_budgeted(
        &self,
        image_bytes: &[u8],
        mode: DotsOcrMode,
        max_new_override: Option<usize>,
    ) -> Result<nv_ocr::DotsPageResult, nv_ocr::Error> {
        let effective = max_new_override.unwrap_or(self.max_new_tokens);
        let img = decode_rgb(image_bytes).map_err(|e| nv_ocr::Error::Decode(format!("{e:#}")))?;
        let img = super::ocr_orient::maybe_auto_orient(img);
        let opts = GenerateOptions {
            max_new_tokens: effective,
            ..Default::default()
        };
        let guard = self.pipeline.lock().unwrap_or_else(|e| {
            tracing::warn!(
                "dots.ocr pipeline mutex was poisoned by an earlier panic; recovering. \
                 DotsOcrPipeline::recognize takes &self and the module has no interior \
                 mutability, so there is no partially-updated state behind this lock."
            );
            e.into_inner()
        });
        let res = guard
            .recognize(&img, dots_mode(mode), &opts)
            .map_err(|e| nv_ocr::Error::Backend(format!("dots.ocr: {e:#}")))?;
        drop(guard);
        let elements = res
            .page
            .elements
            .iter()
            .enumerate()
            .map(|(i, e)| LayoutBox {
                order: i,
                bbox: e.bbox,
                category: e.category.clone(),
                text: e.text.clone(),
            })
            .collect();
        Ok(nv_ocr::DotsPageResult {
            result: OcrResult {
                text: res.text,
                tokens: Vec::new(),
                truncated: !res.hit_eos,
                looped: res.looped,
            },
            layout: LayoutPageResult {
                elements,
                truncated: res.page.truncated,
            },
        })
    }
}

fn dots_dir() -> Option<PathBuf> {
    if let Ok(d) = std::env::var("NV_OCR_DOTS_DIR") {
        return Some(PathBuf::from(d));
    }
    if std::env::var("NV_OCR_DOTS")
        .map(|v| v == "1")
        .unwrap_or(false)
    {
        let found = dots_hub_snapshot();
        if found.is_none() {
            tracing::error!(
                "{}",
                hub_snapshot_missing_message("NV_OCR_DOTS", &DOTS_HUB_REPOS)
            );
        }
        return found;
    }
    None
}

fn load_dots_from_env() -> Option<Arc<OcrEngine>> {
    let dir = dots_dir()?;
    let max_new_tokens = std::env::var("NV_OCR_DOTS_MAX_NEW")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(16384);
    let chain = ocr_device_chain();
    let last = chain.len() - 1;
    for (i, device) in chain.into_iter().enumerate() {
        match DotsOcrPipeline::load(&dir, &device) {
            Ok(pipeline) => {
                info!(
                    dir = %dir.display(),
                    device = device_label(&device),
                    fell_back = i > 0,
                    style = ?PromptStyle::from_env(),
                    budget = ?PixelBudget::from_env(),
                    max_new_tokens,
                    "ocr dots backend loaded"
                );
                return Some(Arc::new(OcrEngine::from_dots(Box::new(
                    DotsPipelineModel {
                        pipeline: Mutex::new(pipeline),
                        max_new_tokens,
                    },
                ))));
            }
            Err(err) if i < last => {
                warn!(
                    error = %format!("{err:#}"),
                    dir = %dir.display(),
                    device = device_label(&device),
                    "ocr dots backend load failed on this device; retrying on cpu"
                );
            }
            Err(err) => {
                warn!(
                    error = %format!("{err:#}"),
                    dir = %dir.display(),
                    device = device_label(&device),
                    "ocr dots backend load failed; disabled"
                );
            }
        }
    }
    None
}

struct GotPipelineModel {
    pipeline: Mutex<GotOcrPipeline>,
    max_new_tokens: usize,
}

fn got_mode(mode: DeepSeekMode) -> GotMode {
    match mode {
        DeepSeekMode::FreeOcr => GotMode::Plain,
        DeepSeekMode::Markdown => GotMode::Format,
    }
}

impl DeepSeekOcr2Model for GotPipelineModel {
    fn recognize_page(
        &self,
        image_bytes: &[u8],
        mode: DeepSeekMode,
    ) -> Result<OcrResult, nv_ocr::Error> {
        self.recognize_page_budgeted(image_bytes, mode, ResolutionHint::Auto, None)
    }

    fn recognize_page_budgeted(
        &self,
        image_bytes: &[u8],
        mode: DeepSeekMode,
        hint: ResolutionHint,
        max_new_override: Option<usize>,
    ) -> Result<OcrResult, nv_ocr::Error> {
        let _ = hint;
        let effective = max_new_override.unwrap_or(self.max_new_tokens);
        let img = decode_rgb(image_bytes).map_err(|e| nv_ocr::Error::Decode(format!("{e:#}")))?;
        let img = super::ocr_orient::maybe_auto_orient(img);
        let guard = self.pipeline.lock().unwrap_or_else(|e| {
            tracing::warn!(
                "GOT-OCR2 pipeline mutex was poisoned by an earlier panic; recovering. \
                 GotOcrPipeline::recognize takes &self and the module has no interior \
                 mutability, so there is no partially-updated state behind this lock."
            );
            e.into_inner()
        });
        let res = guard
            .recognize(&img, got_mode(mode), effective)
            .map_err(|e| nv_ocr::Error::Backend(format!("got-ocr2: {e:#}")))?;
        info!(
            max_new_tokens = effective,
            looped = res.looped,
            hit_eos = res.hit_eos,
            chars = res.text.len(),
            "ocr got decode complete"
        );
        Ok(OcrResult {
            text: res.text,
            tokens: Vec::new(),
            truncated: !res.hit_eos,
            looped: res.looped,
        })
    }
}

pub const GOT_HUB_REPO: &str = "models--stepfun-ai--GOT-OCR-2.0-hf";

fn got_dir() -> Option<PathBuf> {
    if let Ok(d) = std::env::var("NV_OCR_GOT_DIR") {
        return Some(PathBuf::from(d));
    }
    if std::env::var("NV_OCR_GOT").map(|v| v == "1").unwrap_or(false) {
        let found = hub_snapshot(GOT_HUB_REPO);
        if found.is_none() {
            tracing::error!(
                "{}",
                hub_snapshot_missing_message("NV_OCR_GOT", &[GOT_HUB_REPO])
            );
        }
        return found;
    }
    None
}

pub fn load_got_from_env() -> Option<Arc<OcrEngine>> {
    let dir = got_dir()?;
    let max_new_tokens = std::env::var("NV_OCR_GOT_MAX_NEW")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(4096);
    let chain = ocr_device_chain();
    let last = chain.len() - 1;
    for (i, device) in chain.into_iter().enumerate() {
        match GotOcrPipeline::load(&dir, &device) {
            Ok(pipeline) => {
                info!(
                    dir = %dir.display(),
                    device = device_label(&device),
                    fell_back = i > 0,
                    max_new_tokens,
                    "ocr got backend loaded"
                );
                return Some(Arc::new(OcrEngine::from_deepseek(Box::new(
                    GotPipelineModel {
                        pipeline: Mutex::new(pipeline),
                        max_new_tokens,
                    },
                ))));
            }
            Err(err) if i < last => {
                warn!(
                    error = %format!("{err:#}"),
                    dir = %dir.display(),
                    device = device_label(&device),
                    "ocr got backend load failed on this device; retrying on cpu"
                );
            }
            Err(err) => {
                warn!(
                    error = %format!("{err:#}"),
                    dir = %dir.display(),
                    device = device_label(&device),
                    "ocr got backend load failed; disabled"
                );
            }
        }
    }
    None
}

macro_rules! declare_backends {
    ($($variant:ident),+ $(,)?) => {
        #[derive(Debug, Clone, Copy, PartialEq, Eq)]
        enum Backend {
            $($variant),+
        }

        const BACKEND_PRIORITY: [Backend; [$(Backend::$variant),+].len()] =
            [$(Backend::$variant),+];
    };
}

declare_backends!(Tesseract, DeepSeek, Dots, Got);

fn backend_name(b: Backend) -> &'static str {
    match b {
        Backend::Tesseract => "tesseract",
        Backend::DeepSeek => "deepseek",
        Backend::Dots => "dots",
        Backend::Got => "got",
    }
}

fn parse_backend(s: &str) -> Option<Backend> {
    match s {
        "tesseract" | "classical" => Some(Backend::Tesseract),
        "deepseek" => Some(Backend::DeepSeek),
        "dots" | "dots.ocr" => Some(Backend::Dots),
        "got" | "got-ocr" | "got-ocr2" => Some(Backend::Got),
        _ => None,
    }
}

fn loaded_backends(state: &OcrAppState) -> Vec<Backend> {
    let mut out = Vec::with_capacity(3);
    if state.tesseract.is_some() {
        out.push(Backend::Tesseract);
    }
    if state.deepseek.is_some() {
        out.push(Backend::DeepSeek);
    }
    if state.dots.is_some() {
        out.push(Backend::Dots);
    }
    if state.got.is_some() {
        out.push(Backend::Got);
    }
    out
}

fn configured_default_backend() -> Option<Backend> {
    let raw = std::env::var("NV_OCR_DEFAULT_BACKEND").ok()?;
    let raw = raw.trim();
    if raw.is_empty() {
        return None;
    }
    match parse_backend(raw) {
        Some(b) => Some(b),
        None => {
            warn!(
                value = raw,
                "NV_OCR_DEFAULT_BACKEND is not a known backend; ignoring"
            );
            None
        }
    }
}

fn describe_backends(loaded: &[Backend]) -> String {
    if loaded.is_empty() {
        return "none".to_string();
    }
    loaded
        .iter()
        .map(|&b| backend_name(b))
        .collect::<Vec<_>>()
        .join(", ")
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct BackendPickError {
    status: StatusCode,
    code: &'static str,
    message: String,
}

fn pick_default_backend(
    loaded: &[Backend],
    configured: Option<Backend>,
) -> Result<Backend, BackendPickError> {
    if let Some(b) = configured {
        if loaded.contains(&b) {
            return Ok(b);
        }
        return Err(BackendPickError {
            status: StatusCode::SERVICE_UNAVAILABLE,
            code: "ocr_backend_not_loaded",
            message: format!(
                "ocr backend not loaded: {} (NV_OCR_DEFAULT_BACKEND); loaded backends: {}",
                backend_name(b),
                describe_backends(loaded)
            ),
        });
    }
    match loaded {
        [] => Err(BackendPickError {
            status: StatusCode::SERVICE_UNAVAILABLE,
            code: "ocr_backend_not_loaded",
            message: format!(
                "no ocr backend is loaded; loaded backends: {} -- set NV_OCR_TESSDATA (tesseract), NV_OCR_DEEPSEEK_DIR or NV_OCR_DEEPSEEK=1 (deepseek), NV_OCR_DOTS_DIR or NV_OCR_DOTS=1 (dots), or NV_OCR_GOT_DIR or NV_OCR_GOT=1 (got)",
                describe_backends(loaded)
            ),
        }),
        [only] => Ok(*only),
        many => Err(BackendPickError {
            status: StatusCode::BAD_REQUEST,
            code: "ocr_backend_ambiguous",
            message: format!(
                "ambiguous ocr backend: {} backends are loaded ({}) and the request named none. \
                 The server refuses to pick positionally -- a caller that believes it is \
                 measuring {} would otherwise silently be answered by {}. Send backend=<{}> on \
                 the request, or set NV_OCR_DEFAULT_BACKEND to pin the server-side default.",
                many.len(),
                describe_backends(many),
                backend_name(*many.last().unwrap()),
                backend_name(many[0]),
                describe_backends(many),
            ),
        }),
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ModeReq {
    Plain,
    Markdown,
    Layout,
    LayoutOnly,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Format {
    Text,
    Markdown,
    JsonBoxes,
    LayoutJson,
}

fn format_name(f: Format) -> &'static str {
    match f {
        Format::Text => "text",
        Format::Markdown => "markdown",
        Format::JsonBoxes => "json-boxes",
        Format::LayoutJson => "layout-json",
    }
}

fn mode_name(m: Option<ModeReq>) -> &'static str {
    match m {
        None => "default",
        Some(ModeReq::Plain) => "plain",
        Some(ModeReq::Markdown) => "markdown",
        Some(ModeReq::Layout) => "layout",
        Some(ModeReq::LayoutOnly) => "layout-only",
    }
}

fn backend_emits_word_boxes(b: Backend) -> bool {
    match b {
        Backend::Tesseract => true,
        Backend::DeepSeek | Backend::Dots | Backend::Got => false,
    }
}

fn default_format_for(backend: Backend, mode: Option<ModeReq>) -> Format {
    match backend {
        Backend::Tesseract => Format::JsonBoxes,
        Backend::DeepSeek | Backend::Got => match mode {
            Some(ModeReq::Markdown) => Format::Markdown,
            _ => Format::Text,
        },
        Backend::Dots => match mode.unwrap_or(ModeReq::Layout) {
            ModeReq::Plain => Format::Text,
            ModeReq::Markdown => Format::Markdown,
            ModeReq::Layout | ModeReq::LayoutOnly => Format::LayoutJson,
        },
    }
}

fn json_boxes_unsupported(b: Backend) -> String {
    let alt = match b {
        Backend::Dots => "format=layout-json for element boxes, or format=text or format=markdown",
        _ => "format=text or format=markdown",
    };
    format!(
        "format=json-boxes requires backend=tesseract; the {} backend is generative and emits no word boxes -- use {alt}",
        backend_name(b)
    )
}

const OCR_MAX_NEW_TOKENS_CEILING_IS_THE_DOTS_DECODER_MAX_POSITION_EMBEDDINGS: usize = 131072;

fn parse_max_new_tokens(v: &str) -> Result<Option<usize>, String> {
    let v = v.trim();
    if v.is_empty() {
        return Ok(None);
    }
    match v.parse::<usize>() {
        Ok(0) | Err(_) => Err("max_new_tokens must be a positive integer".into()),
        Ok(n) if n > OCR_MAX_NEW_TOKENS_CEILING_IS_THE_DOTS_DECODER_MAX_POSITION_EMBEDDINGS => {
            Err(format!(
                "max_new_tokens {n} exceeds {}, the largest decoder max_position_embeddings across \
                 the generative backends; dots sizes its KV budget as prompt + max_new_tokens with \
                 no internal clamp, so the surface caps the request instead",
                OCR_MAX_NEW_TOKENS_CEILING_IS_THE_DOTS_DECODER_MAX_POSITION_EMBEDDINGS
            ))
        }
        Ok(n) => Ok(Some(n)),
    }
}

fn bad_request(message: String, param: &str) -> Response {
    openai_error(
        StatusCode::BAD_REQUEST,
        message,
        kind::INVALID_REQUEST,
        Some(param),
        None,
    )
}

async fn handle_ocr(
    State(rt): State<OcrRuntime>,
    headers: HeaderMap,
    mut multipart: Multipart,
) -> Response {
    let client = crate::oapi::deadline::from_headers(&headers);
    let state = &rt.app;
    let mut file: Option<Vec<u8>> = None;
    let mut backend: Option<Backend> = None;
    let mut mode: Option<ModeReq> = None;
    let mut format: Option<Format> = None;
    let mut hint = ResolutionHint::Auto;
    let mut rotate = super::ocr_orient::ContentRotation::Upright;
    let mut expected_script = super::ocr_suspect::ExpectedScript::Any;
    let mut max_new_override: Option<usize> = None;

    loop {
        let field = match multipart.next_field().await {
            Ok(Some(f)) => f,
            Ok(None) => break,
            Err(err) => {
                return bad_request(format!("multipart: {err}"), "body");
            }
        };
        let name = field.name().unwrap_or_default().to_string();
        match name.as_str() {
            "file" => match field.bytes().await {
                Ok(b) => file = Some(b.to_vec()),
                Err(err) => {
                    return bad_request(format!("reading file field: {err}"), "file");
                }
            },
            "backend" => match field.text().await.as_deref() {
                Ok("") => {}
                Ok(other) => match parse_backend(other) {
                    Some(b) => backend = Some(b),
                    None => {
                        return bad_request(
                            format!(
                                "unknown backend {other:?}; expected tesseract, deepseek, dots, or got"
                            ),
                            "backend",
                        );
                    }
                },
                Err(err) => return bad_request(format!("reading backend field: {err}"), "backend"),
            },
            "mode" => match field.text().await.as_deref() {
                Ok("") => {}
                Ok("plain") | Ok("free-ocr") => mode = Some(ModeReq::Plain),
                Ok("markdown") => mode = Some(ModeReq::Markdown),
                Ok("layout") | Ok("layout-all") => mode = Some(ModeReq::Layout),
                Ok("layout-only") => mode = Some(ModeReq::LayoutOnly),
                Ok(other) => {
                    return bad_request(
                        format!(
                            "unknown mode {other:?}; expected plain, markdown, layout, or layout-only"
                        ),
                        "mode",
                    );
                }
                Err(err) => return bad_request(format!("reading mode field: {err}"), "mode"),
            },
            "format" => match field.text().await.as_deref() {
                Ok("") => {}
                Ok("text") => format = Some(Format::Text),
                Ok("markdown") => format = Some(Format::Markdown),
                Ok("json-boxes") | Ok("json") => format = Some(Format::JsonBoxes),
                Ok("layout-json") => format = Some(Format::LayoutJson),
                Ok(other) => {
                    return bad_request(
                        format!(
                            "unknown format {other:?}; expected text, markdown, json-boxes, or layout-json"
                        ),
                        "format",
                    );
                }
                Err(err) => return bad_request(format!("reading format field: {err}"), "format"),
            },
            "resolution" => match field.text().await.as_deref() {
                Ok(v) => match ResolutionHint::parse(v.trim()) {
                    Some(h) => hint = h,
                    None => {
                        return bad_request(
                            format!(
                                "unknown resolution {v:?}; expected auto, tiled, base1024, or base768"
                            ),
                            "resolution",
                        );
                    }
                },
                Err(err) => {
                    return bad_request(format!("reading resolution field: {err}"), "resolution")
                }
            },
            "max_new_tokens" => match field.text().await.as_deref() {
                Ok(v) => match parse_max_new_tokens(v) {
                    Ok(None) => {}
                    Ok(some) => max_new_override = some,
                    Err(message) => return bad_request(message, "max_new_tokens"),
                },
                Err(err) => {
                    return bad_request(
                        format!("reading max_new_tokens field: {err}"),
                        "max_new_tokens",
                    )
                }
            },
            "rotate" => match field.text().await.as_deref() {
                Ok("") | Ok("none") => {}
                Ok("cw90") | Ok("90") => rotate = super::ocr_orient::ContentRotation::ApplyCw90,
                Ok("180") => rotate = super::ocr_orient::ContentRotation::Apply180,
                Ok("ccw90") | Ok("270") => rotate = super::ocr_orient::ContentRotation::ApplyCcw90,
                Ok(other) => {
                    return bad_request(
                        format!("unknown rotate {other:?}; expected cw90, 180, ccw90, or none"),
                        "rotate",
                    );
                }
                Err(err) => return bad_request(format!("reading rotate field: {err}"), "rotate"),
            },
            "script" => match field.text().await.as_deref() {
                Ok(v) => match super::ocr_suspect::ExpectedScript::parse(v) {
                    Some(s) => expected_script = s,
                    None => {
                        return bad_request(
                            format!(
                                "unknown script {v:?}; expected any, latin, arabic, cjk, or cyrillic"
                            ),
                            "script",
                        );
                    }
                },
                Err(err) => return bad_request(format!("reading script field: {err}"), "script"),
            },
            _ => {}
        }
    }

    let Some(image_bytes) = file else {
        return bad_request("missing multipart field: file".into(), "file");
    };
    let image_bytes = if rotate == super::ocr_orient::ContentRotation::Upright {
        image_bytes
    } else {
        match super::ocr_orient::reencode_rotated(&image_bytes, rotate) {
            Ok(b) => b,
            Err(err) => return bad_request(format!("rotate: undecodable image: {err:#}"), "file"),
        }
    };
    let requested_backend = backend;
    let backend = match backend {
        Some(b) => b,
        None => {
            let loaded = loaded_backends(state);
            match pick_default_backend(&loaded, configured_default_backend()) {
                Ok(b) => b,
                Err(err) => {
                    warn!(reason = %err.message, "ocr request has no usable default backend");
                    let kind = if err.status == StatusCode::BAD_REQUEST {
                        kind::INVALID_REQUEST
                    } else {
                        kind::SERVICE_UNAVAIL
                    };
                    return openai_error(
                        err.status,
                        err.message,
                        kind,
                        Some("backend"),
                        Some(err.code),
                    );
                }
            }
        }
    };
    let requested_format = format;
    let format = format.unwrap_or_else(|| default_format_for(backend, mode));
    info!(
        backend = backend_name(backend),
        defaulted = requested_backend.is_none(),
        default_source = if requested_backend.is_some() {
            "request"
        } else if configured_default_backend().is_some() {
            "NV_OCR_DEFAULT_BACKEND"
        } else {
            "only-loaded-backend"
        },
        format_defaulted = requested_format.is_none(),
        mode = mode_name(mode),
        format = format_name(format),
        resolution = hint.as_str(),
        max_new_tokens = max_new_override.unwrap_or(0),
        bytes = image_bytes.len(),
        "ocr request accepted"
    );
    if format == Format::LayoutJson && backend != Backend::Dots {
        return bad_request(
            "format=layout-json requires backend=dots; only dots.ocr emits layout boxes".into(),
            "format",
        );
    }
    if format == Format::JsonBoxes && !backend_emits_word_boxes(backend) {
        return bad_request(json_boxes_unsupported(backend), "format");
    }
    if hint != ResolutionHint::Auto && backend != Backend::DeepSeek {
        return bad_request(
            "resolution applies to backend=deepseek only".into(),
            "resolution",
        );
    }
    if max_new_override.is_some() && backend == Backend::Tesseract {
        return bad_request(
            "max_new_tokens applies to generative backends (deepseek, dots, got); tesseract emits \
             no tokens"
                .into(),
            "max_new_tokens",
        );
    }
    let dots_mode_req = match mode.unwrap_or(ModeReq::Layout) {
        ModeReq::Plain => DotsOcrMode::PlainOcr,
        ModeReq::Markdown | ModeReq::Layout => DotsOcrMode::LayoutAll,
        ModeReq::LayoutOnly => DotsOcrMode::LayoutOnly,
    };
    let ds_mode_req = match mode.unwrap_or(ModeReq::Plain) {
        ModeReq::Plain => DeepSeekMode::FreeOcr,
        ModeReq::Markdown => DeepSeekMode::Markdown,
        ModeReq::Layout | ModeReq::LayoutOnly => {
            if backend != Backend::Dots {
                return bad_request(
                    "mode=layout requires backend=dots; only dots.ocr emits layout boxes".into(),
                    "mode",
                );
            }
            DeepSeekMode::FreeOcr
        }
    };
    if backend == Backend::Tesseract && ds_mode_req != DeepSeekMode::FreeOcr {
        return bad_request(
            "mode=markdown requires backend=deepseek; the tesseract backend emits plain text only"
                .into(),
            "mode",
        );
    }

    let engine = match backend {
        Backend::Tesseract => state.tesseract.clone(),
        Backend::DeepSeek => state.deepseek.clone(),
        Backend::Dots => state.dots.clone(),
        Backend::Got => state.got.clone(),
    };
    let Some(engine) = engine else {
        let message = match backend {
            Backend::Tesseract => {
                "ocr backend not loaded: tesseract -- set NV_OCR_TESSDATA to a directory containing eng.traineddata"
            }
            Backend::DeepSeek => {
                "ocr backend not loaded: deepseek -- set NV_OCR_DEEPSEEK_DIR to a DeepSeek-OCR-2 checkpoint dir (or NV_OCR_DEEPSEEK=1 for the hub cache)"
            }
            Backend::Dots => {
                "ocr backend not loaded: dots -- set NV_OCR_DOTS_DIR to a rednote-hilab/dots.ocr checkpoint dir (or NV_OCR_DOTS=1 for the hub cache)"
            }
            Backend::Got => {
                "ocr backend not loaded: got -- set NV_OCR_GOT_DIR to a stepfun-ai/GOT-OCR-2.0-hf checkpoint dir (or NV_OCR_GOT=1 for the hub cache)"
            }
        };
        return openai_error(
            StatusCode::SERVICE_UNAVAILABLE,
            message,
            kind::SERVICE_UNAVAIL,
            Some("backend"),
            Some("ocr_backend_not_loaded"),
        );
    };

    let _permit = match rt.gate_for(backend) {
        None => None,
        Some(gate) => match gate.acquire_with_deadline(client).await {
            Ok(permit) => Some(permit),
            Err(busy) => {
                warn!(
                    backend = backend_name(backend),
                    permits = gate.permits(),
                    queue_ms = gate.queue_ms(),
                    budget_ms = gate.budget_ms(client),
                    caller_deadline = client.is_some(),
                    "ocr request shed: the surface was saturated for the whole queue window"
                );
                return busy.into_response();
            }
        },
    };

    let t0 = Instant::now();
    match backend {
        Backend::Dots => {
            let dmode = dots_mode_req;
            let joined = tokio::task::spawn_blocking(move || {
                engine.recognize_dots_layout_budgeted(&image_bytes, dmode, max_new_override)
            })
            .await;
            let result = match joined {
                Ok(r) => r,
                Err(err) => return join_error(err),
            };
            match result {
                Ok(res) => {
                    info!(
                        backend = "dots",
                        elapsed_ms = elapsed_ms(t0),
                        elements = res.layout.elements.len(),
                        chars = res.result.text.len(),
                        truncated = res.result.truncated,
                        looped = res.result.looped,
                        "ocr request complete"
                    );
                    let truncated = res.result.truncated || res.layout.truncated;
                    let looped = res.result.looped;
                    let suspect =
                        super::ocr_suspect::suspect_reason(&res.result.text, expected_script);
                    let resp = match format {
                        Format::LayoutJson => Json(res.layout).into_response(),
                        Format::JsonBoxes => Json(res).into_response(),
                        Format::Text => (
                            [(header::CONTENT_TYPE, "text/plain; charset=utf-8")],
                            res.result.text,
                        )
                            .into_response(),
                        Format::Markdown => (
                            [(header::CONTENT_TYPE, "text/markdown; charset=utf-8")],
                            res.result.text,
                        )
                            .into_response(),
                    };
                    with_suspect_header(suspect, with_truncation_headers(truncated, looped, resp))
                }
                Err(err) => {
                    warn!(
                        backend = "dots",
                        elapsed_ms = elapsed_ms(t0),
                        error = %err,
                        "ocr request failed"
                    );
                    ocr_error_response(err)
                }
            }
        }
        _ => {
            let dmode = ds_mode_req;
            let profiling = backend == Backend::DeepSeek && dsocr_prof::enabled();
            if profiling {
                dsocr_prof::reset();
            }
            let joined = tokio::task::spawn_blocking(move || {
                engine.recognize_hinted_budgeted(&image_bytes, dmode, hint, max_new_override)
            })
            .await;
            let result = match joined {
                Ok(r) => r,
                Err(err) => return join_error(err),
            };
            if profiling {
                let table = dsocr_prof::report(1.0);
                if !table.trim().is_empty() {
                    info!(
                        calls = dsocr_prof::calls(),
                        "dsocr prefill profile (one request; serialize requests to read it)\n{table}"
                    );
                }
            }
            match result {
                Ok(res) => {
                    info!(
                        backend = backend_name(backend),
                        elapsed_ms = elapsed_ms(t0),
                        tokens = res.tokens.len(),
                        chars = res.text.len(),
                        truncated = res.truncated,
                        looped = res.looped,
                        "ocr request complete"
                    );
                    let (truncated, looped) = (res.truncated, res.looped);
                    let suspect = super::ocr_suspect::suspect_reason(&res.text, expected_script);
                    let resp = match format {
                        Format::LayoutJson | Format::JsonBoxes => Json(res).into_response(),
                        Format::Text => (
                            [(header::CONTENT_TYPE, "text/plain; charset=utf-8")],
                            res.text,
                        )
                            .into_response(),
                        Format::Markdown => (
                            [(header::CONTENT_TYPE, "text/markdown; charset=utf-8")],
                            res.text,
                        )
                            .into_response(),
                    };
                    with_suspect_header(suspect, with_truncation_headers(truncated, looped, resp))
                }
                Err(err) => {
                    warn!(
                        backend = backend_name(backend),
                        elapsed_ms = elapsed_ms(t0),
                        error = %err,
                        "ocr request failed"
                    );
                    ocr_error_response(err)
                }
            }
        }
    }
}

fn with_suspect_header(reason: Option<&'static str>, mut resp: Response) -> Response {
    if let Some(r) = reason {
        resp.headers_mut()
            .insert("x-ocr-suspect", header::HeaderValue::from_static(r));
    }
    resp
}

fn with_truncation_headers(truncated: bool, looped: bool, mut resp: Response) -> Response {
    if truncated {
        resp.headers_mut()
            .insert("x-ocr-truncated", header::HeaderValue::from_static("true"));
    }
    if looped {
        resp.headers_mut()
            .insert("x-ocr-looped", header::HeaderValue::from_static("true"));
    }
    resp
}

fn join_error(err: tokio::task::JoinError) -> Response {
    openai_error(
        StatusCode::INTERNAL_SERVER_ERROR,
        format!("ocr task join: {err}"),
        kind::SERVER,
        None,
        None,
    )
}

fn ocr_error_response(err: nv_ocr::Error) -> Response {
    match err {
        nv_ocr::Error::Decode(msg) => bad_request(format!("undecodable image: {msg}"), "file"),
        nv_ocr::Error::NotWired(msg) => openai_error(
            StatusCode::NOT_IMPLEMENTED,
            msg,
            kind::SERVER,
            Some("backend"),
            Some("ocr_backend_unimplemented"),
        ),
        other => openai_error(
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("ocr: {other}"),
            kind::SERVER,
            None,
            None,
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ocr_device_auto_prefers_an_accelerator_over_cpu() {
        assert_eq!(device_pref(""), DevicePref::Auto);
        assert_eq!(device_pref("cpu"), DevicePref::Cpu);
        assert_eq!(device_pref(" Metal "), DevicePref::Metal);
        assert_eq!(device_pref("gpu"), DevicePref::Auto);
        #[cfg(all(not(feature = "cuda"), not(feature = "metal")))]
        if std::env::var_os("NV_OCR_DEVICE").is_none() {
            assert_eq!(
                device_label(&ocr_device()),
                "cpu",
                "with neither accelerator feature compiled in the auto path must be cpu"
            );
        }
        #[cfg(feature = "metal")]
        if std::env::var_os("NV_OCR_DEVICE").is_none() && candle_core::utils::metal_is_available() {
            assert_eq!(
                device_label(&ocr_device()),
                "metal",
                "a metal build on a metal box must not silently serve OCR from the CPU"
            );
        }
    }

    #[test]
    fn ocr_device_chain_does_not_fall_back_to_cpu_unless_opted_in() {
        assert!(!ocr_chain_appends_cpu(false, false));
        assert!(ocr_chain_appends_cpu(false, true));
        assert!(!ocr_chain_appends_cpu(true, false));
        assert!(!ocr_chain_appends_cpu(true, true));

        let chain = ocr_device_chain_from(candle_core::Device::Cpu, true);
        assert_eq!(chain.len(), 1);
        assert_eq!(device_label(&chain[0]), "cpu");
    }

    #[test]
    fn an_absent_or_empty_max_new_tokens_leaves_the_env_budget_untouched() {
        assert_eq!(parse_max_new_tokens(""), Ok(None));
        assert_eq!(parse_max_new_tokens("   "), Ok(None));
    }

    #[test]
    fn max_new_tokens_accepts_a_positive_integer_up_to_the_ceiling() {
        assert_eq!(parse_max_new_tokens("1"), Ok(Some(1)));
        assert_eq!(parse_max_new_tokens(" 512 "), Ok(Some(512)));
        assert_eq!(
            parse_max_new_tokens("131072"),
            Ok(Some(
                OCR_MAX_NEW_TOKENS_CEILING_IS_THE_DOTS_DECODER_MAX_POSITION_EMBEDDINGS
            )),
            "the ceiling itself is a legal request, not one over the line"
        );
    }

    #[test]
    fn zero_and_unparseable_max_new_tokens_are_rejected_not_silently_defaulted() {
        for v in ["0", "-1", "abc", "1.5", "4096tokens", "1e4"] {
            let err = parse_max_new_tokens(v)
                .expect_err("a budget the surface cannot honour must 400, never fall back");
            assert_eq!(err, "max_new_tokens must be a positive integer", "input {v:?}");
        }
    }

    #[test]
    fn max_new_tokens_over_the_dots_kv_ceiling_is_rejected_at_the_surface() {
        let over = OCR_MAX_NEW_TOKENS_CEILING_IS_THE_DOTS_DECODER_MAX_POSITION_EMBEDDINGS + 1;
        let err = parse_max_new_tokens(&over.to_string())
            .expect_err("dots sizes its KV budget from max_new_tokens with no internal clamp");
        assert!(err.contains(&over.to_string()), "{err}");
        assert!(err.contains("131072"), "{err}");
    }

    #[test]
    fn the_ceiling_is_the_largest_generative_decoder_position_budget() {
        assert_eq!(
            OCR_MAX_NEW_TOKENS_CEILING_IS_THE_DOTS_DECODER_MAX_POSITION_EMBEDDINGS, 131072,
            "dots.ocr DotsDecoderConfig.max_position_embeddings; deepseek (8192) and got (32768) \
             self-clamp downstream, so capping lower would truncate a legal dots request"
        );
    }

    #[test]
    fn the_single_loaded_backend_is_the_default() {
        assert_eq!(
            pick_default_backend(&[Backend::DeepSeek], None),
            Ok(Backend::DeepSeek)
        );
        assert_eq!(
            pick_default_backend(&[Backend::Dots], None),
            Ok(Backend::Dots)
        );
    }

    #[test]
    fn multiple_loaded_backends_refuse_to_default_positionally() {
        for loaded in [
            vec![Backend::Tesseract, Backend::DeepSeek, Backend::Dots],
            vec![Backend::DeepSeek, Backend::Dots],
            vec![Backend::Tesseract, Backend::Dots],
        ] {
            let err = pick_default_backend(&loaded, None)
                .expect_err("an unqualified request must not pick positionally");
            assert_eq!(err.status, StatusCode::BAD_REQUEST);
            assert_eq!(err.code, "ocr_backend_ambiguous");
            assert!(err.message.contains("NV_OCR_DEFAULT_BACKEND"), "{err:?}");
            for b in &loaded {
                assert!(err.message.contains(backend_name(*b)), "{err:?}");
            }
        }
    }

    #[test]
    fn a_single_loaded_backend_is_unambiguous_and_still_defaults() {
        for b in BACKEND_PRIORITY {
            assert_eq!(pick_default_backend(&[b], None), Ok(b));
        }
    }

    #[test]
    fn a_configured_default_resolves_the_ambiguity() {
        for b in BACKEND_PRIORITY {
            assert_eq!(
                pick_default_backend(&BACKEND_PRIORITY, Some(b)),
                Ok(b),
                "{} should win when configured",
                backend_name(b)
            );
        }
    }

    #[test]
    fn hub_snapshot_missing_message_names_the_flag_and_repos() {
        let m = hub_snapshot_missing_message("NV_OCR_DOTS", &DOTS_HUB_REPOS);
        assert!(m.contains("NV_OCR_DOTS=1"), "{m}");
        assert!(m.contains("dots.ocr"), "{m}");
        assert!(m.contains("DISABLED"), "{m}");
        let d = hub_snapshot_missing_message("NV_OCR_DEEPSEEK", &[DEEPSEEK_HUB_REPO]);
        assert!(d.contains(DEEPSEEK_HUB_REPO), "{d}");
    }

    #[test]
    fn a_configured_backend_wins_over_the_priority_order() {
        assert_eq!(
            pick_default_backend(
                &[Backend::Tesseract, Backend::DeepSeek],
                Some(Backend::DeepSeek)
            ),
            Ok(Backend::DeepSeek)
        );
    }

    #[test]
    fn a_configured_backend_that_is_not_loaded_names_the_loaded_ones() {
        let err = pick_default_backend(&[Backend::Dots], Some(Backend::DeepSeek)).unwrap_err();
        assert_eq!(err.status, StatusCode::SERVICE_UNAVAILABLE);
        assert_eq!(err.code, "ocr_backend_not_loaded");
        assert!(err.message.contains("deepseek"), "{err:?}");
        assert!(err.message.contains("NV_OCR_DEFAULT_BACKEND"), "{err:?}");
        assert!(err.message.contains("dots"), "{err:?}");
    }

    #[test]
    fn no_loaded_backend_names_none_and_the_env_knobs() {
        let err = pick_default_backend(&[], None).unwrap_err();
        assert_eq!(err.status, StatusCode::SERVICE_UNAVAILABLE);
        assert!(err.message.contains("none"), "{err:?}");
        assert!(err.message.contains("NV_OCR_TESSDATA"), "{err:?}");
        assert!(err.message.contains("NV_OCR_DEEPSEEK_DIR"), "{err:?}");
        assert!(err.message.contains("NV_OCR_DOTS_DIR"), "{err:?}");
    }

    #[test]
    fn every_declared_backend_is_reachable_through_its_own_name() {
        for (i, b) in BACKEND_PRIORITY.iter().enumerate() {
            assert_eq!(
                BACKEND_PRIORITY.iter().position(|x| x == b),
                Some(i),
                "{} is listed twice in BACKEND_PRIORITY, so the suites that use the list \
                 as 'every backend' weight it twice and leave another variant untested",
                backend_name(*b)
            );
            assert_eq!(
                parse_backend(backend_name(*b)),
                Some(*b),
                "declare_backends! put {} in BACKEND_PRIORITY but parse_backend rejects \
                 the name backend_name prints for it, so no request can select it and \
                 the ocr_backend_ambiguous message names a value the field will refuse",
                backend_name(*b)
            );
        }
    }

    #[test]
    fn backend_field_aliases_parse() {
        assert_eq!(parse_backend("tesseract"), Some(Backend::Tesseract));
        assert_eq!(parse_backend("classical"), Some(Backend::Tesseract));
        assert_eq!(parse_backend("deepseek"), Some(Backend::DeepSeek));
        assert_eq!(parse_backend("dots.ocr"), Some(Backend::Dots));
        assert_eq!(parse_backend("got"), Some(Backend::Got));
        assert_eq!(parse_backend("got-ocr"), Some(Backend::Got));
        assert_eq!(parse_backend("got-ocr2"), Some(Backend::Got));
    }

    #[test]
    fn only_tesseract_defaults_to_json_boxes() {
        for b in BACKEND_PRIORITY {
            if default_format_for(b, None) == Format::JsonBoxes {
                assert!(
                    backend_emits_word_boxes(b),
                    "{} defaults to json-boxes but cannot populate tokens",
                    backend_name(b)
                );
            }
        }
        assert_eq!(
            default_format_for(Backend::Tesseract, None),
            Format::JsonBoxes
        );
    }

    #[test]
    fn the_default_format_follows_the_backend_and_mode() {
        assert_eq!(default_format_for(Backend::DeepSeek, None), Format::Text);
        assert_eq!(
            default_format_for(Backend::DeepSeek, Some(ModeReq::Markdown)),
            Format::Markdown
        );
        assert_eq!(default_format_for(Backend::Dots, None), Format::LayoutJson);
        assert_eq!(
            default_format_for(Backend::Dots, Some(ModeReq::LayoutOnly)),
            Format::LayoutJson
        );
        assert_eq!(
            default_format_for(Backend::Dots, Some(ModeReq::Plain)),
            Format::Text
        );
    }

    #[tokio::test]
    async fn json_boxes_on_deepseek_is_400_naming_a_format_it_supports() {
        let resp = post_ocr_fields(&[
            ("file", "x"),
            ("backend", "deepseek"),
            ("format", "json-boxes"),
        ])
        .await;
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
        let v = envelope(resp).await;
        assert_eq!(v["error"]["param"], "format");
        let msg = v["error"]["message"].as_str().unwrap();
        assert!(msg.contains("deepseek"), "{msg}");
        assert!(msg.contains("format=text"), "{msg}");
    }

    #[tokio::test]
    async fn json_boxes_on_dots_points_at_layout_json() {
        let resp =
            post_ocr_fields(&[("file", "x"), ("backend", "dots"), ("format", "json-boxes")]).await;
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
        let v = envelope(resp).await;
        let msg = v["error"]["message"].as_str().unwrap();
        assert!(msg.contains("dots"), "{msg}");
        assert!(msg.contains("format=layout-json"), "{msg}");
    }

    #[tokio::test]
    async fn json_boxes_on_tesseract_is_not_rejected() {
        let resp = post_ocr_fields(&[
            ("file", "x"),
            ("backend", "tesseract"),
            ("format", "json-boxes"),
        ])
        .await;
        assert_eq!(resp.status(), StatusCode::SERVICE_UNAVAILABLE);
        let v = envelope(resp).await;
        assert_eq!(v["error"]["code"], "ocr_backend_not_loaded");
    }

    async fn post_ocr_fields(fields: &[(&str, &str)]) -> Response {
        use axum::body::Body;
        use axum::extract::{FromRequest, Request};
        const BOUNDARY: &str = "ocrunitboundary";
        let mut body = String::new();
        for (name, value) in fields {
            body.push_str(&format!(
                "--{BOUNDARY}\r\nContent-Disposition: form-data; name=\"{name}\"\r\n\r\n{value}\r\n"
            ));
        }
        body.push_str(&format!("--{BOUNDARY}--\r\n"));
        let req = Request::builder()
            .method("POST")
            .uri("/v1/ocr")
            .header(
                header::CONTENT_TYPE,
                format!("multipart/form-data; boundary={BOUNDARY}"),
            )
            .body(Body::from(body))
            .unwrap();
        let multipart = Multipart::from_request(req, &()).await.unwrap();
        handle_ocr(
            State(OcrRuntime::from(OcrAppState::default())),
            HeaderMap::new(),
            multipart,
        )
        .await
    }

    async fn envelope(resp: Response) -> serde_json::Value {
        let bytes = axum::body::to_bytes(resp.into_body(), 1 << 16)
            .await
            .unwrap();
        serde_json::from_slice(&bytes).unwrap()
    }

    fn tiny_png() -> Vec<u8> {
        let img = nv_imgdec::image::RgbImage::from_fn(6, 4, |x, y| {
            nv_imgdec::image::Rgb([(x * 40) as u8, (y * 60) as u8, 128])
        });
        let mut out = std::io::Cursor::new(Vec::new());
        nv_imgdec::image::DynamicImage::ImageRgb8(img)
            .write_to(&mut out, nv_imgdec::image::ImageFormat::Png)
            .unwrap();
        out.into_inner()
    }

    async fn post_ocr_with_binary_file(file: &[u8], fields: &[(&str, &str)]) -> Response {
        use axum::body::Body;
        use axum::extract::{FromRequest, Request};
        const BOUNDARY: &str = "ocrbinboundary";
        let mut body: Vec<u8> = Vec::new();
        body.extend_from_slice(
            format!(
                "--{BOUNDARY}\r\nContent-Disposition: form-data; name=\"file\"; \
                 filename=\"t.png\"\r\nContent-Type: image/png\r\n\r\n"
            )
            .as_bytes(),
        );
        body.extend_from_slice(file);
        body.extend_from_slice(b"\r\n");
        for (name, value) in fields {
            body.extend_from_slice(
                format!(
                    "--{BOUNDARY}\r\nContent-Disposition: form-data; name=\"{name}\"\r\n\r\n{value}\r\n"
                )
                .as_bytes(),
            );
        }
        body.extend_from_slice(format!("--{BOUNDARY}--\r\n").as_bytes());
        let req = Request::builder()
            .method("POST")
            .uri("/v1/ocr")
            .header(
                header::CONTENT_TYPE,
                format!("multipart/form-data; boundary={BOUNDARY}"),
            )
            .body(Body::from(body))
            .unwrap();
        let multipart = Multipart::from_request(req, &()).await.unwrap();
        handle_ocr(
            State(OcrRuntime::from(OcrAppState::default())),
            HeaderMap::new(),
            multipart,
        )
        .await
    }

    #[tokio::test]
    async fn unknown_rotate_value_is_400_naming_the_accepted_set() {
        let resp = post_ocr_fields(&[("file", "x"), ("rotate", "45")]).await;
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
        let v = envelope(resp).await;
        assert_eq!(v["error"]["param"], "rotate");
        let msg = v["error"]["message"].as_str().unwrap();
        assert!(msg.contains("cw90"), "{msg}");
    }

    #[tokio::test]
    async fn unknown_script_value_is_400_naming_the_accepted_set() {
        let resp = post_ocr_fields(&[("file", "x"), ("script", "klingon")]).await;
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
        let v = envelope(resp).await;
        assert_eq!(v["error"]["param"], "script");
        let msg = v["error"]["message"].as_str().unwrap();
        assert!(msg.contains("latin"), "{msg}");
    }

    #[tokio::test]
    async fn rotate_on_a_real_png_reaches_backend_selection_not_a_decode_error() {
        let resp = post_ocr_with_binary_file(
            &tiny_png(),
            &[("backend", "deepseek"), ("rotate", "cw90")],
        )
        .await;
        assert_eq!(
            resp.status(),
            StatusCode::SERVICE_UNAVAILABLE,
            "decode-rotate-reencode must succeed and fall through to the unloaded-backend 503"
        );
        let v = envelope(resp).await;
        assert_eq!(v["error"]["code"], "ocr_backend_not_loaded");
    }

    #[tokio::test]
    async fn rotate_on_junk_bytes_is_400_blaming_the_file() {
        let resp = post_ocr_with_binary_file(
            b"definitely not an image",
            &[("backend", "deepseek"), ("rotate", "180")],
        )
        .await;
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
        let v = envelope(resp).await;
        assert_eq!(v["error"]["param"], "file");
        let msg = v["error"]["message"].as_str().unwrap();
        assert!(msg.contains("rotate"), "{msg}");
    }

    #[tokio::test]
    async fn script_prior_on_a_real_png_reaches_backend_selection() {
        let resp = post_ocr_with_binary_file(
            &tiny_png(),
            &[("backend", "got"), ("script", "latin")],
        )
        .await;
        assert_eq!(resp.status(), StatusCode::SERVICE_UNAVAILABLE);
        let v = envelope(resp).await;
        assert_eq!(v["error"]["code"], "ocr_backend_not_loaded");
    }

    fn gate_test_traineddata() -> Option<PathBuf> {
        let root = tessdata_root()?;
        let file = resolve_traineddata(&root);
        file.is_file().then_some(file)
    }

    fn gate_test_page() -> Vec<u8> {
        std::fs::read(
            Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("../conformance/fixtures/071-ocr-layout-report/input.png"),
        )
        .expect("fixture png")
    }

    async fn post_ocr_page(rt: &OcrRuntime, png: &[u8]) -> Response {
        post_ocr_page_hdr(rt, png, HeaderMap::new()).await
    }

    async fn post_ocr_page_hdr(rt: &OcrRuntime, png: &[u8], headers: HeaderMap) -> Response {
        use axum::body::Body;
        use axum::extract::{FromRequest, Request};
        const BOUNDARY: &str = "ocrgateboundary";
        let mut body: Vec<u8> = Vec::new();
        body.extend_from_slice(
            format!(
                "--{BOUNDARY}\r\nContent-Disposition: form-data; name=\"file\"; \
                 filename=\"page.png\"\r\nContent-Type: image/png\r\n\r\n"
            )
            .as_bytes(),
        );
        body.extend_from_slice(png);
        body.extend_from_slice(b"\r\n");
        for (name, value) in [("backend", "tesseract"), ("format", "text")] {
            body.extend_from_slice(
                format!(
                    "--{BOUNDARY}\r\nContent-Disposition: form-data; \
                     name=\"{name}\"\r\n\r\n{value}\r\n"
                )
                .as_bytes(),
            );
        }
        body.extend_from_slice(format!("--{BOUNDARY}--\r\n").as_bytes());
        let req = Request::builder()
            .method("POST")
            .uri("/v1/ocr")
            .header(
                header::CONTENT_TYPE,
                format!("multipart/form-data; boundary={BOUNDARY}"),
            )
            .body(Body::from(body))
            .unwrap();
        let multipart = Multipart::from_request(req, &()).await.unwrap();
        handle_ocr(State(rt.clone()), headers, multipart).await
    }

    fn tesseract_runtime(classical: Option<SurfaceGate>) -> Option<(OcrRuntime, Vec<u8>)> {
        let file = gate_test_traineddata()?;
        let engine = OcrEngine::from_traineddata(&file, BackendKind::Classical)
            .expect("tesseract engine from traineddata");
        let app = OcrAppState {
            tesseract: Some(Arc::new(engine)),
            deepseek: None,
            dots: None,
            got: None,
        };
        let rt = OcrRuntime::with_gates(app, SurfaceGate::new("/v1/ocr", 1, 40), classical);
        Some((rt, gate_test_page()))
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn a_saturated_ocr_gate_sheds_and_a_freed_or_wider_one_admits() {
        let Some((rt, png)) = tesseract_runtime(Some(SurfaceGate::new("/v1/ocr", 1, 40))) else {
            eprintln!("SKIP: no eng.traineddata under NV_OCR_TESSDATA/~/.cache/ocr-testdata");
            return;
        };
        let gate = rt.classical.clone().expect("gated runtime");
        let held = gate.acquire().await.expect("hold the only permit");
        let resp = post_ocr_page(&rt, &png).await;
        assert_eq!(
            resp.status(),
            StatusCode::SERVICE_UNAVAILABLE,
            "a surface with every permit held must shed at the queue deadline"
        );
        let v = envelope(resp).await;
        assert_eq!(v["error"]["code"], "surface_busy", "{v}");
        drop(held);

        let resp = post_ocr_page(&rt, &png).await;
        assert_eq!(
            resp.status(),
            StatusCode::OK,
            "the freed slot must serve the next request -- the permit leaked otherwise"
        );

        let (wide, png) = tesseract_runtime(Some(SurfaceGate::new("/v1/ocr", 2, 40)))
            .expect("traineddata was present a moment ago");
        let _held = wide
            .classical
            .clone()
            .expect("gated runtime")
            .acquire()
            .await
            .expect("hold one of two permits");
        let resp = post_ocr_page(&wide, &png).await;
        assert_eq!(
            resp.status(),
            StatusCode::OK,
            "a second permit must admit a request the 1-permit gate shed"
        );
    }

    fn deadline_header(v: &str) -> HeaderMap {
        let mut h = HeaderMap::new();
        h.insert(
            crate::oapi::deadline::HEADER,
            axum::http::HeaderValue::from_str(v).unwrap(),
        );
        h
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn a_caller_deadline_sheds_ocr_earlier_than_the_server_queue_window() {
        let Some((rt, png)) = tesseract_runtime(Some(SurfaceGate::new("/v1/ocr", 1, 3_000))) else {
            eprintln!("SKIP: no eng.traineddata under NV_OCR_TESSDATA/~/.cache/ocr-testdata");
            return;
        };
        let gate = rt.classical.clone().expect("gated runtime");
        let _held = gate.acquire().await.expect("hold the only permit");

        let t0 = Instant::now();
        let resp = post_ocr_page_hdr(&rt, &png, deadline_header("120")).await;
        let with_header = t0.elapsed();
        assert_eq!(resp.status(), StatusCode::SERVICE_UNAVAILABLE);
        assert_eq!(envelope(resp).await["error"]["code"], "surface_busy");

        let t0 = Instant::now();
        let resp = post_ocr_page(&rt, &png).await;
        let without_header = t0.elapsed();
        assert_eq!(resp.status(), StatusCode::SERVICE_UNAVAILABLE);
        assert_eq!(envelope(resp).await["error"]["code"], "surface_busy");

        assert!(
            with_header < Duration::from_millis(1_500),
            "the 120 ms caller deadline was ignored: shed took {with_header:?}"
        );
        assert!(
            without_header >= Duration::from_millis(2_500),
            "the server default window collapsed: no-header shed took {without_header:?}"
        );
        assert!(
            without_header > with_header * 2,
            "no measurable delta: {with_header:?} with the header vs {without_header:?} without"
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn a_malformed_ocr_deadline_header_falls_back_to_the_server_window() {
        let Some((rt, png)) = tesseract_runtime(Some(SurfaceGate::new("/v1/ocr", 1, 400))) else {
            eprintln!("SKIP: no eng.traineddata under NV_OCR_TESSDATA/~/.cache/ocr-testdata");
            return;
        };
        let gate = rt.classical.clone().expect("gated runtime");
        let held = gate.acquire().await.expect("hold the only permit");
        let t0 = Instant::now();
        let resp = post_ocr_page_hdr(&rt, &png, deadline_header("soon")).await;
        let elapsed = t0.elapsed();
        assert_eq!(
            resp.status(),
            StatusCode::SERVICE_UNAVAILABLE,
            "garbage must not 400 and must not admit"
        );
        assert_eq!(envelope(resp).await["error"]["code"], "surface_busy");
        assert!(
            elapsed >= Duration::from_millis(300),
            "garbage collapsed the window to the floor instead of falling back: {elapsed:?}"
        );
        drop(held);
        let resp = post_ocr_page_hdr(&rt, &png, deadline_header("soon")).await;
        assert_eq!(resp.status(), StatusCode::OK, "a freed slot still admits");
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn the_classical_backend_is_ungated_by_default() {
        let Some((rt, png)) = tesseract_runtime(None) else {
            eprintln!("SKIP: no eng.traineddata under NV_OCR_TESSDATA/~/.cache/ocr-testdata");
            return;
        };
        assert!(rt.gate_for(Backend::Tesseract).is_none());
        assert!(rt.gate_for(Backend::Dots).is_some());
        let held = rt
            .generative
            .acquire()
            .await
            .expect("hold the only generative permit");
        let resp = post_ocr_page(&rt, &png).await;
        assert_eq!(
            resp.status(),
            StatusCode::OK,
            "a saturated generative gate must not shed a classical request"
        );
        drop(held);
    }

    #[test]
    fn ocr_gates_are_sized_by_their_own_env_vars() {
        std::env::set_var("NV_OCR_CONCURRENCY", "3");
        std::env::set_var("NV_OCR_QUEUE_MS", "1234");
        std::env::set_var("NV_OCR_CLASSICAL_CONCURRENCY", "6");
        let rt = OcrRuntime::from(OcrAppState::default());
        assert_eq!(rt.generative.permits(), 3);
        assert_eq!(rt.generative.queue_ms(), 1234);
        let classical = rt.classical.as_deref().expect("classical gate opted in");
        assert_eq!(classical.permits(), 6);
        assert_eq!(classical.queue_ms(), 1234);

        std::env::remove_var("NV_OCR_CONCURRENCY");
        std::env::remove_var("NV_OCR_QUEUE_MS");
        std::env::remove_var("NV_OCR_CLASSICAL_CONCURRENCY");
        let rt = OcrRuntime::from(OcrAppState::default());
        assert_eq!(rt.generative.permits(), 1, "default /v1/ocr concurrency");
        assert_eq!(rt.generative.queue_ms(), 3_000, "default queue window");
        assert!(
            rt.classical.is_none(),
            "the classical path must stay ungated unless asked for"
        );
    }

    #[test]
    fn loaded_backends_reflects_state() {
        assert!(loaded_backends(&OcrAppState::default()).is_empty());
        assert_eq!(describe_backends(&[]), "none");
        assert_eq!(
            describe_backends(&[Backend::DeepSeek, Backend::Dots]),
            "deepseek, dots"
        );
    }
}
