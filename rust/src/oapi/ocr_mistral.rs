use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use axum::body::Body;
use axum::extract::{Multipart, Path};
use axum::http::{header, HeaderMap, HeaderValue, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use base64::engine::general_purpose::STANDARD as B64;
use base64::Engine as _;
use futures_util::StreamExt;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use tracing::{info, warn};

use nv_ocr::{DeepSeekMode, DotsOcrMode, DotsTierHint, LayoutBox, OcrEngine, ResolutionHint};

use super::chat_multimodal::{decode_data_url_bytes, ImageUrlSpec};
use super::ocr::{
    backend_name, configured_backends, configured_default_backend, demand, parse_backend,
    pick_default_backend, Backend, OcrAppState, OcrRuntime,
};
use super::ocr_batch::decode_rgb;
use super::Model;

pub const MISTRAL_OCR_ALIASES: [&str; 3] =
    ["mistral-ocr-latest", "mistral-ocr-2505", "mistral-ocr-2503"];
pub const OCR_TASK: &str = "ocr";
const FETCH_MAX_BYTES_DEFAULT_IS_MISTRALS_50_MB_DOCUMENT_LIMIT: usize = 50 * 1024 * 1024;
const PDF_MAX_PAGES_DEFAULT_IS_MISTRALS_1000_PAGE_LIMIT: usize = 1000;
const PDF_RASTER_DPI_DEFAULT_IS_WHAT_MISTRAL_REPORTS_FOR_PDF_PAGES: u32 = 200;
const RASTER_IMAGE_INPUT_HAS_NO_PHYSICAL_DPI_SO_REPORT_THE_CSS_REFERENCE: u32 = 72;
const FILES_STORE_MAX_BYTES_DEFAULT: usize = 256 * 1024 * 1024;
const CROP_JPEG_QUALITY: u8 = 85;
const ANNOTATION_MAX_TOKENS: usize = 1024;
const ANNOTATION_IMAGE_CAP_IS_MISTRALS_FIRST_EIGHT_BOXES: usize = 8;
const PICTURE_CATEGORY: &str = "Picture";

pub const CODE_INVALID_REQUEST: &str = "1000";
pub const CODE_INVALID_MODEL: &str = "1500";

static LOADED_STATE: OnceLock<OcrAppState> = OnceLock::new();

pub(super) fn note_state(state: &OcrAppState) {
    let _ = LOADED_STATE.set(state.clone());
}

pub(super) fn engine_model_id(b: Backend) -> &'static str {
    match b {
        Backend::Tesseract => "tesseract/eng",
        Backend::DeepSeek => "deepseek-ai/DeepSeek-OCR-2",
        Backend::Dots => "rednote-hilab/dots.ocr",
        Backend::Got => "stepfun-ai/GOT-OCR-2.0-hf",
        Backend::LightOn => "lightonai/LightOnOCR-3-0.8B",
    }
}

fn load_hint(b: Backend) -> &'static str {
    match b {
        Backend::Tesseract => "NV_OCR_TESSDATA",
        Backend::DeepSeek => "NV_OCR_DEEPSEEK_DIR or NV_OCR_DEEPSEEK=1",
        Backend::Dots => "NV_OCR_DOTS_DIR or NV_OCR_DOTS=1",
        Backend::Got => "NV_OCR_GOT_DIR or NV_OCR_GOT=1",
        Backend::LightOn => "NV_OCR_LIGHTON_DIR or NV_OCR_LIGHTON=1",
    }
}

pub fn model_rows() -> Vec<Model> {
    let Some(state) = LOADED_STATE.get() else {
        return Vec::new();
    };
    let loaded = configured_backends(state);
    if loaded.is_empty() {
        return Vec::new();
    }
    let default_id = pick_default_backend(&loaded, configured_default_backend())
        .ok()
        .map(engine_model_id);
    let row = |id: &str, owner: &str, extras: serde_json::Map<String, Value>| Model {
        id: id.to_string(),
        created: 1,
        owned_by: owner.to_string(),
        languages: None,
        task: OCR_TASK.into(),
        max_model_len: None,
        extras,
    };
    let mut out: Vec<Model> = loaded
        .iter()
        .map(|&b| {
            let id = engine_model_id(b);
            let mut extras = serde_json::Map::new();
            extras.insert("backend".into(), json!(backend_name(b)));
            row(id, id.split('/').next().unwrap_or("speaches-plus"), extras)
        })
        .collect();
    for alias in MISTRAL_OCR_ALIASES {
        let mut extras = serde_json::Map::new();
        extras.insert("parent".into(), json!(default_id));
        out.push(row(alias, "mistralai", extras));
    }
    out
}

pub fn mistral_error(status: StatusCode, message: Value, kind: &str, code: &str) -> Response {
    (
        status,
        Json(json!({
            "object": "error",
            "message": message,
            "type": kind,
            "param": null,
            "code": code,
        })),
    )
        .into_response()
}

fn invalid(status: StatusCode, message: impl Into<String>, code: &str) -> Response {
    mistral_error(
        status,
        Value::String(message.into()),
        "invalid_request_error",
        code,
    )
}

pub fn validation_error(loc: &[&str], msg: impl Into<String>, ty: &str) -> Response {
    let mut full = vec!["body"];
    full.extend_from_slice(loc);
    mistral_error(
        StatusCode::UNPROCESSABLE_ENTITY,
        json!({ "detail": [{ "type": ty, "loc": full, "msg": msg.into() }] }),
        "invalid_request_error",
        CODE_INVALID_REQUEST,
    )
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Document {
    DocumentUrl(String),
    ImageUrl(String),
    File(String),
}

impl Document {
    pub fn from_value(v: &Value) -> Result<Document, Response> {
        let Some(obj) = v.as_object() else {
            return Err(validation_error(
                &["document"],
                "Input should be a valid dictionary",
                "model_type",
            ));
        };
        let ty = obj.get("type").and_then(Value::as_str).or_else(|| {
            ["document_url", "image_url", "file_id"]
                .into_iter()
                .find(|k| obj.contains_key(*k))
                .map(|k| if k == "file_id" { "file" } else { k })
        });
        let text = |key: &str| -> Result<String, Response> {
            obj.get(key)
                .and_then(Value::as_str)
                .map(str::to_string)
                .ok_or_else(|| validation_error(&["document", key], "Field required", "missing"))
        };
        match ty {
            Some("document_url") => Ok(Document::DocumentUrl(text("document_url")?)),
            Some("image_url") => {
                let spec: ImageUrlSpec = obj
                    .get("image_url")
                    .cloned()
                    .and_then(|v| serde_json::from_value(v).ok())
                    .ok_or_else(|| {
                        validation_error(
                            &["document", "image_url"],
                            "Input should be a URL string or {\"url\": ...}",
                            "missing",
                        )
                    })?;
                Ok(Document::ImageUrl(spec.url().to_string()))
            }
            Some("file") => Ok(Document::File(text("file_id")?)),
            _ => Err(validation_error(
                &["document", "type"],
                "Input tag should be 'document_url', 'image_url' or 'file'",
                "union_tag_invalid",
            )),
        }
    }
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
pub enum PagesSpec {
    List(Vec<usize>),
    Text(String),
}

pub fn parse_pages(spec: &PagesSpec) -> Result<Vec<usize>, String> {
    let mut out = match spec {
        PagesSpec::List(v) => v.clone(),
        PagesSpec::Text(s) => {
            let mut v = Vec::new();
            for part in s.split(',').map(str::trim).filter(|p| !p.is_empty()) {
                let bounds = match part.split_once('-') {
                    Some((a, b)) => (a.trim().parse::<usize>(), b.trim().parse::<usize>()),
                    None => (part.parse::<usize>(), part.parse::<usize>()),
                };
                match bounds {
                    (Ok(a), Ok(b)) if a <= b => v.extend(a..=b),
                    _ => {
                        return Err(format!(
                            "pages: {part:?} is not a page number or an ascending range like 0-5"
                        ))
                    }
                }
            }
            v
        }
    };
    out.sort_unstable();
    out.dedup();
    Ok(out)
}

#[derive(Debug, Deserialize)]
pub struct OcrRequest {
    #[serde(default)]
    pub model: Option<String>,
    pub document: Value,
    #[serde(default)]
    pub pages: Option<PagesSpec>,
    #[serde(default)]
    pub include_image_base64: Option<bool>,
    #[serde(default)]
    pub image_limit: Option<usize>,
    #[serde(default)]
    pub image_min_size: Option<u32>,
    #[serde(default)]
    pub bbox_annotation_format: Option<Value>,
    #[serde(default)]
    pub document_annotation_format: Option<Value>,
    #[serde(default)]
    pub document_annotation_prompt: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct OcrResponse {
    pub pages: Vec<OcrPage>,
    pub model: String,
    pub document_annotation: Option<String>,
    pub usage_info: UsageInfo,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct OcrPage {
    pub index: usize,
    pub markdown: String,
    pub images: Vec<OcrImage>,
    pub dimensions: Dimensions,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct OcrImage {
    pub id: String,
    pub top_left_x: u32,
    pub top_left_y: u32,
    pub bottom_right_x: u32,
    pub bottom_right_y: u32,
    pub image_base64: Option<String>,
    pub image_annotation: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Dimensions {
    pub dpi: u32,
    pub height: u32,
    pub width: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct UsageInfo {
    pub pages_processed: usize,
    pub doc_size_bytes: Option<usize>,
}

fn env_usize(name: &str) -> Option<usize> {
    std::env::var(name).ok()?.trim().parse().ok()
}

fn fetch_cap() -> usize {
    env_usize("NV_OCR_FETCH_MAX_BYTES")
        .unwrap_or(FETCH_MAX_BYTES_DEFAULT_IS_MISTRALS_50_MB_DOCUMENT_LIMIT)
}

pub(super) fn resolve_backend(model: Option<&str>, state: &OcrAppState) -> Result<Backend, Response> {
    let loaded = configured_backends(state);
    let named = match model.map(str::trim) {
        None | Some("") => None,
        Some(m) if MISTRAL_OCR_ALIASES.contains(&m) => None,
        Some(m) => Some(
            parse_backend(m)
                .or_else(|| {
                    [
                        Backend::Tesseract,
                        Backend::DeepSeek,
                        Backend::Dots,
                        Backend::Got,
                        Backend::LightOn,
                    ]
                    .into_iter()
                    .find(|&b| engine_model_id(b) == m)
                })
                .ok_or_else(|| {
                    mistral_error(
                        StatusCode::BAD_REQUEST,
                        json!(format!("Invalid model: {m}")),
                        "invalid_model",
                        CODE_INVALID_MODEL,
                    )
                })?,
        ),
    };
    match named {
        Some(b) if loaded.contains(&b) => Ok(b),
        Some(b) => Err(invalid(
            StatusCode::SERVICE_UNAVAILABLE,
            format!(
                "ocr backend not loaded: {} -- set {}",
                backend_name(b),
                load_hint(b)
            ),
            "ocr_backend_not_loaded",
        )),
        None => pick_default_backend(&loaded, configured_default_backend())
            .map_err(|err| invalid(err.status, err.message, err.code)),
    }
}

fn file_id_in_url(url: &str) -> Option<&str> {
    let (_, rest) = url.split_once("/v1/files/")?;
    rest.strip_suffix("/content")
}

fn stored_bytes(id: &str) -> Result<Vec<u8>, Response> {
    stored(id)
        .map(|(_, bytes)| bytes.to_vec())
        .ok_or_else(|| file_not_found(id))
}

async fn fetch_document(doc: &Document) -> Result<Vec<u8>, Response> {
    let url = match doc {
        Document::File(id) => return stored_bytes(id),
        Document::DocumentUrl(u) | Document::ImageUrl(u) => u.as_str(),
    };
    let cap = fetch_cap();
    if url.starts_with("data:") {
        return decode_data_url_bytes(url).map_err(|e| {
            validation_error(&["document"], format!("{e:#}"), "value_error")
        });
    }
    if let Some(id) = file_id_in_url(url) {
        return stored_bytes(id);
    }
    if !(url.starts_with("http://") || url.starts_with("https://")) {
        return Err(validation_error(
            &["document"],
            "document_url must be an http(s) URL, a data: URL, or a /v1/files/{id}/content URL",
            "url_scheme",
        ));
    }
    let too_large = || {
        invalid(
            StatusCode::PAYLOAD_TOO_LARGE,
            format!("document exceeds the {cap}-byte fetch cap (NV_OCR_FETCH_MAX_BYTES)"),
            "ocr_document_too_large",
        )
    };
    let fetch_failed =
        |e: String| invalid(StatusCode::BAD_REQUEST, format!("fetch document_url: {e}"), "ocr_document_fetch_failed");
    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(120))
        .build()
        .map_err(|e| fetch_failed(e.to_string()))?;
    let resp = client
        .get(url)
        .send()
        .await
        .map_err(|e| fetch_failed(e.to_string()))?;
    if !resp.status().is_success() {
        return Err(fetch_failed(format!("upstream answered {}", resp.status())));
    }
    if resp.content_length().is_some_and(|n| n as usize > cap) {
        return Err(too_large());
    }
    let mut out = Vec::new();
    let mut stream = resp.bytes_stream();
    while let Some(chunk) = stream.next().await {
        let chunk = chunk.map_err(|e| fetch_failed(e.to_string()))?;
        if out.len() + chunk.len() > cap {
            return Err(too_large());
        }
        out.extend_from_slice(&chunk);
    }
    Ok(out)
}

pub fn is_pdf(bytes: &[u8]) -> bool {
    bytes.starts_with(b"%PDF")
}

pub fn page_index_from_raster_name(name: &str) -> Option<usize> {
    let stem = name.strip_suffix(".png")?;
    let digits = stem.rsplit('-').next()?;
    digits.parse::<usize>().ok()?.checked_sub(1)
}

fn pdftoppm_bin() -> String {
    std::env::var("NV_OCR_PDFTOPPM").unwrap_or_else(|_| "pdftoppm".to_string())
}

fn rasterize_pdf(
    bytes: &[u8],
    pages: Option<&[usize]>,
    dpi: u32,
) -> Result<Vec<(usize, Vec<u8>)>, Response> {
    let max_pages =
        env_usize("NV_OCR_PDF_MAX_PAGES").unwrap_or(PDF_MAX_PAGES_DEFAULT_IS_MISTRALS_1000_PAGE_LIMIT);
    let (first, last) = match pages {
        Some([]) => return Ok(Vec::new()),
        Some(p) => (p[0] + 1, (p[p.len() - 1] + 1).min(p[0] + max_pages)),
        None => (1, max_pages),
    };
    let scratch = std::env::temp_dir().join(format!("nv-ocr-pdf-{}", uuid::Uuid::new_v4()));
    let internal = |e: String| invalid(StatusCode::INTERNAL_SERVER_ERROR, e, "ocr_pdf_scratch");
    std::fs::create_dir_all(&scratch).map_err(|e| internal(format!("create pdf scratch dir: {e}")))?;
    let result = (|| {
        let input = scratch.join("in.pdf");
        std::fs::write(&input, bytes).map_err(|e| internal(format!("write pdf scratch: {e}")))?;
        let bin = pdftoppm_bin();
        let output = std::process::Command::new(&bin)
            .args(["-r", &dpi.to_string(), "-png", "-f", &first.to_string(), "-l", &last.to_string()])
            .arg(&input)
            .arg(scratch.join("page"))
            .output()
            .map_err(|e| {
                if e.kind() == std::io::ErrorKind::NotFound {
                    invalid(
                        StatusCode::NOT_IMPLEMENTED,
                        format!(
                            "pdf input needs the pdftoppm rasterizer ({bin:?} is not on PATH; install poppler-utils or point NV_OCR_PDFTOPPM at it)"
                        ),
                        "ocr_pdf_rasterizer_unavailable",
                    )
                } else {
                    internal(format!("run {bin}: {e}"))
                }
            })?;
        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            return Err(invalid(
                StatusCode::BAD_REQUEST,
                format!(
                    "pdftoppm could not render the document: {}",
                    stderr.trim().chars().take(300).collect::<String>()
                ),
                "ocr_pdf_unreadable",
            ));
        }
        let mut rendered: Vec<(usize, Vec<u8>)> = std::fs::read_dir(&scratch)
            .map_err(|e| internal(format!("list pdf scratch: {e}")))?
            .flatten()
            .filter_map(|e| {
                let name = e.file_name().to_string_lossy().into_owned();
                let index = page_index_from_raster_name(&name)?;
                if pages.is_some_and(|p| !p.contains(&index)) {
                    return None;
                }
                std::fs::read(e.path()).ok().map(|b| (index, b))
            })
            .collect();
        rendered.sort_by_key(|(i, _)| *i);
        Ok(rendered)
    })();
    let _ = std::fs::remove_dir_all(&scratch);
    result
}

struct ImageOpts {
    include_base64: bool,
    limit: usize,
    min_size: u32,
}

struct PageOut {
    page: OcrPage,
    crops: Vec<String>,
}

fn crop_data_url(img: &nv_models::deepseek_ocr::RgbImage, b: [u32; 4]) -> Option<String> {
    let full = image::RgbImage::from_raw(img.w as u32, img.h as u32, img.data.clone())?;
    let crop = image::imageops::crop_imm(&full, b[0], b[1], b[2] - b[0], b[3] - b[1]).to_image();
    let mut buf = std::io::Cursor::new(Vec::new());
    image::codecs::jpeg::JpegEncoder::new_with_quality(&mut buf, CROP_JPEG_QUALITY)
        .encode_image(&crop)
        .ok()?;
    Some(format!("data:image/jpeg;base64,{}", B64.encode(buf.into_inner())))
}

fn picture_bounds(e: &LayoutBox, w: u32, h: u32, min_size: u32) -> Option<[u32; 4]> {
    if e.category.as_deref() != Some(PICTURE_CATEGORY) {
        return None;
    }
    let b = e.bbox?;
    let x0 = (b[0].min(b[2]).floor().max(0.0) as u32).min(w);
    let y0 = (b[1].min(b[3]).floor().max(0.0) as u32).min(h);
    let x1 = (b[0].max(b[2]).ceil().max(0.0) as u32).min(w);
    let y1 = (b[1].max(b[3]).ceil().max(0.0) as u32).min(h);
    (x1 - x0 >= min_size.max(1) && y1 - y0 >= min_size.max(1)).then_some([x0, y0, x1, y1])
}

pub fn markdown_from_layout(elements: &[LayoutBox], image_ids: &HashMap<usize, String>) -> String {
    let mut parts: Vec<String> = Vec::new();
    for e in elements {
        if let Some(id) = image_ids.get(&e.order) {
            parts.push(format!("![{id}]({id})"));
        } else if e.category.as_deref() != Some(PICTURE_CATEGORY) {
            if let Some(t) = e.text.as_deref().map(str::trim).filter(|t| !t.is_empty()) {
                parts.push(t.to_string());
            }
        }
    }
    parts.join("\n\n")
}

fn ocr_page(
    engine: &OcrEngine,
    backend: Backend,
    index: usize,
    bytes: &[u8],
    dpi: u32,
    opts: &ImageOpts,
) -> Result<PageOut, nv_ocr::Error> {
    let img = decode_rgb(bytes).map_err(|e| nv_ocr::Error::Decode(format!("{e:#}")))?;
    let (w, h) = (img.w as u32, img.h as u32);
    let mut images = Vec::new();
    let mut crops = Vec::new();
    let markdown = match backend {
        Backend::Dots | Backend::LightOn => {
            let res = engine.recognize_dots_layout_hinted(
                bytes,
                DotsOcrMode::LayoutAll,
                None,
                DotsTierHint::Auto,
            )?;
            let mut ids = HashMap::new();
            for e in &res.layout.elements {
                if images.len() >= opts.limit {
                    break;
                }
                let Some(b) = picture_bounds(e, w, h, opts.min_size) else {
                    continue;
                };
                let id = format!("img-{}.jpeg", images.len());
                let crop = crop_data_url(&img, b);
                if opts.include_base64 {
                    crops.push(crop.clone().unwrap_or_default());
                }
                images.push(OcrImage {
                    id: id.clone(),
                    top_left_x: b[0],
                    top_left_y: b[1],
                    bottom_right_x: b[2],
                    bottom_right_y: b[3],
                    image_base64: crop,
                    image_annotation: None,
                });
                ids.insert(e.order, id);
            }
            let has_text = res.layout.elements.iter().any(|e| e.text.is_some());
            if has_text || !ids.is_empty() {
                markdown_from_layout(&res.layout.elements, &ids)
            } else {
                res.result.text
            }
        }
        Backend::DeepSeek | Backend::Got => {
            engine
                .recognize_hinted_budgeted(bytes, DeepSeekMode::Markdown, ResolutionHint::Auto, None)?
                .text
        }
        Backend::Tesseract => engine.recognize(bytes)?.text,
    };
    Ok(PageOut {
        page: OcrPage {
            index,
            markdown,
            images,
            dimensions: Dimensions {
                dpi,
                height: h,
                width: w,
            },
        },
        crops,
    })
}

fn annotation_schema(format: &Value, field: &str) -> Result<Value, Response> {
    match format.get("type").and_then(Value::as_str) {
        Some("json_schema") if format.get("json_schema").is_some() => Ok(format.clone()),
        _ => Err(validation_error(
            &[field],
            "Only json_schema is valid for this field",
            "value_error",
        )),
    }
}

fn annotation_model() -> Option<String> {
    std::env::var("NV_OCR_ANNOTATION_MODEL")
        .ok()
        .filter(|m| !m.trim().is_empty())
}

async fn annotate(
    client: &reqwest::Client,
    base: &str,
    auth: Option<&HeaderValue>,
    model: &str,
    content: Vec<Value>,
    format: &Value,
) -> Result<String, String> {
    let mut req = client.post(format!("{base}/v1/chat/completions")).json(&json!({
        "model": model,
        "messages": [{ "role": "user", "content": content }],
        "response_format": format,
        "max_tokens": ANNOTATION_MAX_TOKENS,
    }));
    if let Some(a) = auth {
        req = req.header(header::AUTHORIZATION, a.clone());
    }
    let resp = req
        .timeout(Duration::from_secs(300))
        .send()
        .await
        .map_err(|e| format!("annotation request: {e}"))?;
    let status = resp.status();
    let body: Value = resp
        .json()
        .await
        .map_err(|e| format!("annotation response: {e}"))?;
    if !status.is_success() {
        return Err(format!("annotation model answered {status}: {body}"));
    }
    body.pointer("/choices/0/message/content")
        .and_then(Value::as_str)
        .map(str::to_string)
        .ok_or_else(|| "annotation response carried no message content".to_string())
}

pub(super) async fn handle(rt: OcrRuntime, headers: HeaderMap, body: Body) -> Response {
    let client_deadline = super::deadline::from_headers(&headers);
    let cap = fetch_cap();
    let bytes = match axum::body::to_bytes(body, cap / 3 * 4 + 4096).await {
        Ok(b) => b,
        Err(e) => return invalid(StatusCode::PAYLOAD_TOO_LARGE, format!("request body: {e}"), "ocr_document_too_large"),
    };
    let req: OcrRequest = match serde_json::from_slice(&bytes) {
        Ok(r) => r,
        Err(e) if e.is_syntax() || e.is_eof() => {
            return validation_error(&[], format!("JSON decode error: {e}"), "json_invalid")
        }
        Err(e) if e.to_string().starts_with("missing field `document`") => {
            return validation_error(&["document"], "Field required", "missing")
        }
        Err(e) => return validation_error(&[], e.to_string(), "value_error"),
    };
    let document = match Document::from_value(&req.document) {
        Ok(d) => d,
        Err(r) => return r,
    };
    let pages = match req.pages.as_ref().map(parse_pages) {
        None => None,
        Some(Ok(p)) => Some(p),
        Some(Err(msg)) => return validation_error(&["pages"], msg, "value_error"),
    };
    let bbox_format = match req.bbox_annotation_format.as_ref() {
        None => None,
        Some(f) => match annotation_schema(f, "bbox_annotation_format") {
            Ok(f) => Some(f),
            Err(r) => return r,
        },
    };
    let doc_format = match req.document_annotation_format.as_ref() {
        None => None,
        Some(f) => match annotation_schema(f, "document_annotation_format") {
            Ok(f) => Some(f),
            Err(r) => return r,
        },
    };
    let annotation_model = annotation_model();
    if (bbox_format.is_some() || doc_format.is_some()) && annotation_model.is_none() {
        return invalid(
            StatusCode::NOT_IMPLEMENTED,
            "bbox_annotation_format and document_annotation_format need a vision chat model: set NV_OCR_ANNOTATION_MODEL to a loaded chat model id that accepts image_url parts and response_format json_schema",
            "ocr_annotation_unavailable",
        );
    }
    let backend = match resolve_backend(req.model.as_deref(), &rt.app) {
        Ok(b) => b,
        Err(r) => return r,
    };
    let model_echo = req
        .model
        .clone()
        .filter(|m| !m.trim().is_empty())
        .unwrap_or_else(|| engine_model_id(backend).to_string());
    let doc_bytes = match fetch_document(&document).await {
        Ok(b) => b,
        Err(r) => return r,
    };
    let doc_size = doc_bytes.len();
    let dpi = env_usize("NV_OCR_PDF_DPI")
        .map(|d| d as u32)
        .unwrap_or(PDF_RASTER_DPI_DEFAULT_IS_WHAT_MISTRAL_REPORTS_FOR_PDF_PAGES);
    let slot = match backend {
        Backend::Tesseract => rt.app.tesseract.clone(),
        Backend::DeepSeek => demand(&rt.app.deepseek).await,
        Backend::Dots => demand(&rt.app.dots).await,
        Backend::Got => demand(&rt.app.got).await,
        Backend::LightOn => demand(&rt.app.lighton).await,
    };
    let Some(engine) = slot else {
        return invalid(
            StatusCode::SERVICE_UNAVAILABLE,
            format!(
                "ocr backend {} is configured but its load failed; see the server log, the load is retried shortly",
                backend_name(backend)
            ),
            "ocr_backend_load_failed",
        );
    };
    let permit = match rt.gate_for(backend) {
        None => None,
        Some(gate) => match gate.acquire_with_deadline(client_deadline).await {
            Ok(p) => Some(p),
            Err(busy) => return busy.into_response(),
        },
    };
    let opts = ImageOpts {
        include_base64: req.include_image_base64.unwrap_or(false) || bbox_format.is_some(),
        limit: req.image_limit.unwrap_or(usize::MAX),
        min_size: req.image_min_size.unwrap_or(0),
    };
    let t0 = Instant::now();
    let joined = tokio::task::spawn_blocking(move || {
        let _permit = permit;
        let (inputs, page_dpi): (Vec<(usize, Vec<u8>)>, u32) = if is_pdf(&doc_bytes) {
            (rasterize_pdf(&doc_bytes, pages.as_deref(), dpi)?, dpi)
        } else if pages.as_ref().is_some_and(|p| !p.contains(&0)) {
            (Vec::new(), dpi)
        } else {
            (
                vec![(0, doc_bytes)],
                RASTER_IMAGE_INPUT_HAS_NO_PHYSICAL_DPI_SO_REPORT_THE_CSS_REFERENCE,
            )
        };
        let mut out = Vec::with_capacity(inputs.len());
        for (index, bytes) in &inputs {
            out.push(
                ocr_page(&engine, backend, *index, bytes, page_dpi, &opts).map_err(ocr_error)?,
            );
        }
        Ok::<_, Response>(out)
    })
    .await;
    let pages_out = match joined {
        Ok(Ok(p)) => p,
        Ok(Err(r)) => return r,
        Err(e) => return invalid(StatusCode::INTERNAL_SERVER_ERROR, format!("ocr task join: {e}"), "ocr_join"),
    };
    info!(
        backend = backend_name(backend),
        model = %model_echo,
        pages = pages_out.len(),
        doc_bytes = doc_size,
        elapsed_ms = t0.elapsed().as_millis() as u64,
        "mistral-shaped ocr request complete"
    );
    let mut pages: Vec<OcrPage> = Vec::with_capacity(pages_out.len());
    let mut crops: Vec<Vec<String>> = Vec::with_capacity(pages_out.len());
    for p in pages_out {
        pages.push(p.page);
        crops.push(p.crops);
    }
    let mut document_annotation = None;
    if let Some(model) = annotation_model.filter(|_| bbox_format.is_some() || doc_format.is_some()) {
        let Some(addr) = super::SELF_ADDR.get() else {
            return invalid(StatusCode::SERVICE_UNAVAILABLE, "annotation unavailable: listen address not bound", "ocr_annotation_unavailable");
        };
        let base = format!("http://{addr}");
        let client = reqwest::Client::new();
        let auth = headers.get(header::AUTHORIZATION);
        let annotation_failed = |e: String| invalid(StatusCode::BAD_GATEWAY, e, "ocr_annotation_failed");
        if let Some(format) = &bbox_format {
            let mut budget = ANNOTATION_IMAGE_CAP_IS_MISTRALS_FIRST_EIGHT_BOXES;
            for (page, page_crops) in pages.iter_mut().zip(&crops) {
                for (image, crop) in page.images.iter_mut().zip(page_crops) {
                    if budget == 0 || crop.is_empty() {
                        continue;
                    }
                    budget -= 1;
                    let content = vec![
                        json!({ "type": "image_url", "image_url": { "url": crop } }),
                        json!({ "type": "text", "text": "Describe this image. Answer with a JSON object that follows the schema." }),
                    ];
                    match annotate(&client, &base, auth, &model, content, format).await {
                        Ok(a) => image.image_annotation = Some(a),
                        Err(e) => return annotation_failed(e),
                    }
                    if !req.include_image_base64.unwrap_or(false) {
                        image.image_base64 = None;
                    }
                }
            }
        }
        if let Some(format) = &doc_format {
            let markdown: Vec<&str> = pages.iter().map(|p| p.markdown.as_str()).collect();
            let prompt = req
                .document_annotation_prompt
                .clone()
                .unwrap_or_else(|| "Extract the requested information from this document. Answer with a JSON object that follows the schema.".to_string());
            let content = vec![json!({ "type": "text", "text": format!("{prompt}\n\n{}", markdown.join("\n\n---\n\n")) })];
            match annotate(&client, &base, auth, &model, content, format).await {
                Ok(a) => document_annotation = Some(a),
                Err(e) => return annotation_failed(e),
            }
        }
    }
    let pages_processed = pages.len();
    Json(OcrResponse {
        pages,
        model: model_echo,
        document_annotation,
        usage_info: UsageInfo {
            pages_processed,
            doc_size_bytes: Some(doc_size),
        },
    })
    .into_response()
}

fn ocr_error(err: nv_ocr::Error) -> Response {
    match err {
        nv_ocr::Error::Decode(msg) => invalid(
            StatusCode::BAD_REQUEST,
            format!("unsupported document: expected a PDF or an image ({}); {msg}", nv_imgdec::SUPPORTED_IMAGE_FORMATS),
            "ocr_document_undecodable",
        ),
        nv_ocr::Error::NotWired(msg) => invalid(StatusCode::NOT_IMPLEMENTED, msg, "ocr_backend_unimplemented"),
        other => {
            warn!(error = %other, "mistral-shaped ocr request failed");
            invalid(StatusCode::INTERNAL_SERVER_ERROR, format!("ocr: {other}"), "ocr_failed")
        }
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct StoredFile {
    pub id: String,
    pub object: &'static str,
    pub bytes: usize,
    pub created_at: u64,
    pub filename: String,
    pub purpose: String,
    pub sample_type: Option<String>,
    pub source: &'static str,
}

#[derive(Default)]
struct FileStore {
    order: VecDeque<String>,
    map: HashMap<String, (StoredFile, Arc<Vec<u8>>)>,
    total: usize,
}

impl FileStore {
    fn get(&self, id: &str) -> Option<&(StoredFile, Arc<Vec<u8>>)> {
        self.map.get(id)
    }

    fn insert(&mut self, meta: StoredFile, bytes: Vec<u8>) {
        let cap = env_usize("NV_OCR_FILES_MAX_BYTES").unwrap_or(FILES_STORE_MAX_BYTES_DEFAULT);
        self.total += bytes.len();
        while self.total > cap {
            let Some(oldest) = self.order.pop_front() else {
                break;
            };
            self.remove(&oldest);
        }
        self.order.push_back(meta.id.clone());
        self.map.insert(meta.id.clone(), (meta, Arc::new(bytes)));
    }

    fn remove(&mut self, id: &str) -> bool {
        match self.map.remove(id) {
            Some((_, bytes)) => {
                self.total -= bytes.len();
                self.order.retain(|x| x != id);
                true
            }
            None => false,
        }
    }
}

fn files() -> &'static Mutex<FileStore> {
    static FILES: OnceLock<Mutex<FileStore>> = OnceLock::new();
    FILES.get_or_init(|| Mutex::new(FileStore::default()))
}

fn file_not_found(id: &str) -> Response {
    mistral_error(
        StatusCode::NOT_FOUND,
        json!(format!("File not found: {id}")),
        "not_found_error",
        "ocr_file_not_found",
    )
}

pub fn files_router() -> Router {
    Router::new()
        .route("/v1/files", post(upload_file))
        .route("/v1/files/{id}", get(get_file).delete(delete_file))
        .route("/v1/files/{id}/url", get(signed_url))
        .route("/v1/files/{id}/content", get(file_content))
}

async fn upload_file(mut multipart: Multipart) -> Response {
    let cap = fetch_cap();
    let mut file: Option<(String, Vec<u8>)> = None;
    let mut purpose = "ocr".to_string();
    loop {
        let field = match multipart.next_field().await {
            Ok(Some(f)) => f,
            Ok(None) => break,
            Err(e) => return validation_error(&[], format!("multipart: {e}"), "value_error"),
        };
        match field.name().unwrap_or_default() {
            "file" => {
                let name = field.file_name().unwrap_or("upload").to_string();
                match field.bytes().await {
                    Ok(b) if b.len() > cap => {
                        return invalid(StatusCode::PAYLOAD_TOO_LARGE, format!("file exceeds the {cap}-byte cap (NV_OCR_FETCH_MAX_BYTES)"), "ocr_document_too_large")
                    }
                    Ok(b) => file = Some((name, b.to_vec())),
                    Err(e) => return validation_error(&["file"], format!("reading file: {e}"), "value_error"),
                }
            }
            "purpose" => purpose = field.text().await.unwrap_or_default(),
            _ => {}
        }
    }
    let Some((filename, bytes)) = file else {
        return validation_error(&["file"], "Field required", "missing");
    };
    let meta = StoredFile {
        id: uuid::Uuid::new_v4().to_string(),
        object: "file",
        bytes: bytes.len(),
        created_at: SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0),
        filename,
        purpose,
        sample_type: None,
        source: "upload",
    };
    files()
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .insert(meta.clone(), bytes);
    Json(meta).into_response()
}

fn stored(id: &str) -> Option<(StoredFile, Arc<Vec<u8>>)> {
    files()
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .get(id)
        .cloned()
}

async fn get_file(Path(id): Path<String>) -> Response {
    match stored(&id) {
        Some((meta, _)) => {
            let mut v = serde_json::to_value(meta).unwrap_or_default();
            v["deleted"] = json!(false);
            Json(v).into_response()
        }
        None => file_not_found(&id),
    }
}

async fn delete_file(Path(id): Path<String>) -> Response {
    let removed = files()
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .remove(&id);
    if removed {
        Json(json!({ "id": id, "object": "file", "deleted": true })).into_response()
    } else {
        file_not_found(&id)
    }
}

async fn signed_url(Path(id): Path<String>, headers: HeaderMap) -> Response {
    if stored(&id).is_none() {
        return file_not_found(&id);
    }
    let host = headers
        .get(header::HOST)
        .and_then(|h| h.to_str().ok())
        .map(str::to_string)
        .or_else(|| super::SELF_ADDR.get().map(|a| a.to_string()))
        .unwrap_or_else(|| "localhost".to_string());
    let scheme = headers
        .get("x-forwarded-proto")
        .and_then(|h| h.to_str().ok())
        .unwrap_or("http");
    Json(json!({ "url": format!("{scheme}://{host}/v1/files/{id}/content") })).into_response()
}

async fn file_content(Path(id): Path<String>) -> Response {
    match stored(&id) {
        Some((_, bytes)) => (
            [(header::CONTENT_TYPE, "application/octet-stream")],
            bytes.to_vec(),
        )
            .into_response(),
        None => file_not_found(&id),
    }
}
