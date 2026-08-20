use std::path::PathBuf;
use std::sync::{Arc, OnceLock};

use anyhow::{anyhow, Result};
use nv_ocr::{BackendKind, OcrEngine};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(
    feature = "ts-bindings",
    derive(ts_rs::TS),
    ts(export, rename = "PiiOcrToken")
)]
pub struct OcrToken {
    pub start: usize,
    #[serde(rename = "endExclusive")]
    pub end_exclusive: usize,
    pub rect: PixelRect,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(
    feature = "ts-bindings",
    derive(ts_rs::TS),
    ts(export, rename = "PiiPixelRect")
)]
pub struct PixelRect {
    pub left: i32,
    pub top: i32,
    pub right: i32,
    pub bottom: i32,
}

pub struct OcrResult {
    pub text: String,
    pub tokens: Vec<OcrToken>,
}

static ENGINE: OnceLock<Option<Arc<OcrEngine>>> = OnceLock::new();

fn tessdata_root() -> Option<PathBuf> {
    if let Ok(p) = std::env::var("NV_OCR_TESSDATA") {
        return Some(PathBuf::from(p));
    }
    let home = std::env::var("HOME").ok()?;
    let cache = PathBuf::from(home).join(".cache/ocr-testdata");
    cache.exists().then_some(cache)
}

fn engine() -> Option<Arc<OcrEngine>> {
    ENGINE
        .get_or_init(|| {
            let root = tessdata_root()?;
            let path = crate::oapi::ocr::resolve_traineddata(&root);
            match OcrEngine::from_traineddata(&path, BackendKind::Classical) {
                Ok(engine) => {
                    tracing::info!(path = %path.display(), "PII redact OCR backend loaded");
                    Some(Arc::new(engine))
                }
                Err(err) => {
                    tracing::warn!(
                        error = %err,
                        path = %path.display(),
                        "PII redact OCR backend load failed; /v1/pii/redact/analyze disabled"
                    );
                    None
                }
            }
        })
        .clone()
}

pub fn run_ocr(image_bytes: &[u8]) -> Result<OcrResult> {
    let Some(engine) = engine() else {
        return Err(anyhow!(
            "ocr not available: set NV_OCR_TESSDATA to a directory containing eng.traineddata \
             (or place one under ~/.cache/ocr-testdata)"
        ));
    };
    let result = engine
        .recognize(image_bytes)
        .map_err(|e| anyhow!("ocr recognize: {e}"))?;
    Ok(OcrResult {
        text: result.text,
        tokens: result
            .tokens
            .into_iter()
            .map(|t| OcrToken {
                start: t.start,
                end_exclusive: t.end_exclusive,
                rect: PixelRect {
                    left: t.rect.left,
                    top: t.rect.top,
                    right: t.rect.right,
                    bottom: t.rect.bottom,
                },
            })
            .collect(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_unavailable_error_is_the_503_shape_the_handler_matches_on() {
        let msg =
            "ocr not available: set NV_OCR_TESSDATA to a directory containing eng.traineddata";
        assert!(
            msg.contains("not available"),
            "image_handler::analyze_post maps only messages containing \"not available\" to 503"
        );
    }

    #[test]
    fn tessdata_root_prefers_the_env_var() {
        std::env::set_var("NV_OCR_TESSDATA", "/tmp/some-tessdata-dir");
        assert_eq!(
            tessdata_root(),
            Some(PathBuf::from("/tmp/some-tessdata-dir"))
        );
        std::env::remove_var("NV_OCR_TESSDATA");
    }
}
