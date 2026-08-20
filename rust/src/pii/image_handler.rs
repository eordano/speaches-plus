use std::sync::Arc;

use axum::{
    extract::{FromRequest, Multipart, Request, State},
    http::{header, StatusCode},
    response::{IntoResponse, Response},
};
use serde::{Deserialize, Serialize};

use super::classifier::PiiClassifier;
use super::ocr;
use super::renderer::{FillMode, RedactRect};
use super::span_mapper;
use super::spans::PiiSpan;
use crate::oapi;

#[derive(Debug, Serialize)]
#[cfg_attr(
    feature = "ts-bindings",
    derive(ts_rs::TS),
    ts(export, rename = "PiiRedactAnalyzeResponse")
)]
struct AnalyzeResponse {
    text: String,
    tokens: Vec<ocr::OcrToken>,
    spans: Vec<PiiSpan>,
    rects: Vec<span_mapper::LabeledRect>,
}

#[derive(Debug, Deserialize)]
#[cfg_attr(
    feature = "ts-bindings",
    derive(ts_rs::TS),
    ts(export, rename = "PiiRedactRenderRect")
)]
struct RenderRect {
    left: i32,
    top: i32,
    right: i32,
    bottom: i32,
}

pub struct OaiMultipart(pub Multipart);

impl<S> FromRequest<S> for OaiMultipart
where
    S: Send + Sync,
{
    type Rejection = Response;

    async fn from_request(req: Request, state: &S) -> Result<Self, Self::Rejection> {
        match Multipart::from_request(req, state).await {
            Ok(m) => Ok(OaiMultipart(m)),
            Err(rejection) => Err(oapi::openai_error(
                rejection.status(),
                rejection.body_text(),
                oapi::kind::INVALID_REQUEST,
                None,
                Some("invalid_multipart"),
            )),
        }
    }
}

pub async fn analyze_post(
    State(classifier): State<Arc<PiiClassifier>>,
    OaiMultipart(mut multipart): OaiMultipart,
) -> Response {
    let mut image_bytes: Option<Vec<u8>> = None;

    while let Ok(Some(field)) = multipart.next_field().await {
        let name = field.name().unwrap_or("").to_string();
        if name == "file" {
            match field.bytes().await {
                Ok(b) => image_bytes = Some(b.to_vec()),
                Err(err) => {
                    return error_response(StatusCode::BAD_REQUEST, format!("file read: {err}"));
                }
            }
        } else {
            let _ = field.bytes().await;
        }
    }

    let Some(bytes) = image_bytes else {
        return error_response(StatusCode::BAD_REQUEST, "missing 'file' field".into());
    };

    let result = match tokio::task::spawn_blocking(move || analyze_image(&classifier, &bytes)).await
    {
        Ok(Ok(r)) => r,
        Ok(Err(err)) => {
            let msg = format!("{err:#}");
            return error_response(classify_error(&msg), format!("analyze: {msg}"));
        }
        Err(err) => {
            return error_response(StatusCode::INTERNAL_SERVER_ERROR, format!("join: {err}"));
        }
    };

    (
        StatusCode::OK,
        [(header::CONTENT_TYPE, "application/json")],
        serde_json::to_string(&result).unwrap(),
    )
        .into_response()
}

pub async fn render_post(OaiMultipart(mut multipart): OaiMultipart) -> Response {
    let mut image_bytes: Option<Vec<u8>> = None;
    let mut rects_json: Option<String> = None;
    let mut fill_mode_str = "solid".to_string();
    let mut fill_color_str = "#000000".to_string();

    while let Ok(Some(field)) = multipart.next_field().await {
        let name = field.name().unwrap_or("").to_string();
        match name.as_str() {
            "file" => match field.bytes().await {
                Ok(b) => image_bytes = Some(b.to_vec()),
                Err(err) => {
                    return error_response(StatusCode::BAD_REQUEST, format!("file read: {err}"));
                }
            },
            "rects" => match field.text().await {
                Ok(t) => rects_json = Some(t),
                Err(err) => {
                    return error_response(StatusCode::BAD_REQUEST, format!("rects read: {err}"));
                }
            },
            "fill_mode" => {
                if let Ok(v) = field.text().await {
                    fill_mode_str = v;
                }
            }
            "fill_color" => {
                if let Ok(v) = field.text().await {
                    fill_color_str = v;
                }
            }
            _ => {
                let _ = field.bytes().await;
            }
        }
    }

    let Some(bytes) = image_bytes else {
        return error_response(StatusCode::BAD_REQUEST, "missing 'file' field".into());
    };

    let Some(rects_str) = rects_json else {
        return error_response(StatusCode::BAD_REQUEST, "missing 'rects' field".into());
    };

    let rects: Vec<RenderRect> = match serde_json::from_str(&rects_str) {
        Ok(r) => r,
        Err(err) => {
            return error_response(StatusCode::BAD_REQUEST, format!("rects JSON parse: {err}"));
        }
    };

    let fill_mode = match fill_mode_str.as_str() {
        "shuffle" => FillMode::Shuffle,
        _ => FillMode::Solid,
    };

    let fill_color = parse_hex_color(&fill_color_str).unwrap_or([0, 0, 0]);

    let redact_rects: Vec<RedactRect> = rects
        .iter()
        .map(|r| RedactRect {
            left: r.left,
            top: r.top,
            right: r.right,
            bottom: r.bottom,
        })
        .collect();

    let result = match tokio::task::spawn_blocking(move || {
        super::renderer::render_redactions(&bytes, &redact_rects, fill_mode, fill_color)
    })
    .await
    {
        Ok(Ok(png_bytes)) => png_bytes,
        Ok(Err(err)) => {
            let msg = format!("{err:#}");
            return error_response(classify_error(&msg), format!("render: {msg}"));
        }
        Err(err) => {
            return error_response(StatusCode::INTERNAL_SERVER_ERROR, format!("join: {err}"));
        }
    };

    (
        StatusCode::OK,
        [(header::CONTENT_TYPE, "image/png")],
        result,
    )
        .into_response()
}

fn analyze_image(
    classifier: &PiiClassifier,
    image_bytes: &[u8],
) -> anyhow::Result<AnalyzeResponse> {
    let img = nv_imgdec::decode_oriented(image_bytes)?;
    let (img_width, img_height) = (img.width(), img.height());

    let ocr_result = ocr::run_ocr(image_bytes)?;
    let spans = classifier.classify_one(&ocr_result.text)?;
    let rects = span_mapper::map_spans(&ocr_result.tokens, &spans, img_width, img_height);

    Ok(AnalyzeResponse {
        text: ocr_result.text,
        tokens: ocr_result.tokens,
        spans,
        rects,
    })
}

fn parse_hex_color(s: &str) -> Option<[u8; 3]> {
    let s = s.strip_prefix('#').unwrap_or(s);
    if s.len() != 6 {
        return None;
    }
    let r = u8::from_str_radix(&s[0..2], 16).ok()?;
    let g = u8::from_str_radix(&s[2..4], 16).ok()?;
    let b = u8::from_str_radix(&s[4..6], 16).ok()?;
    Some([r, g, b])
}

fn classify_error(message: &str) -> StatusCode {
    if message.contains("not available") {
        StatusCode::SERVICE_UNAVAILABLE
    } else if message.contains("decode image") {
        StatusCode::BAD_REQUEST
    } else {
        StatusCode::INTERNAL_SERVER_ERROR
    }
}

fn error_response(status: StatusCode, message: String) -> Response {
    let kind = match status {
        StatusCode::BAD_REQUEST => oapi::kind::INVALID_REQUEST,
        StatusCode::SERVICE_UNAVAILABLE => oapi::kind::SERVICE_UNAVAIL,
        _ => oapi::kind::SERVER,
    };
    oapi::openai_error(status, message, kind, None, None)
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::body::to_bytes;

    async fn envelope(resp: Response) -> serde_json::Value {
        let bytes = to_bytes(resp.into_body(), 1 << 16).await.unwrap();
        serde_json::from_slice(&bytes).unwrap()
    }

    #[tokio::test]
    async fn bad_request_uses_invalid_request_envelope() {
        let resp = error_response(StatusCode::BAD_REQUEST, "missing 'file' field".into());
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
        let v = envelope(resp).await;
        assert_eq!(v["error"]["type"], "invalid_request_error");
        assert_eq!(v["error"]["message"], "missing 'file' field");
        assert!(v["error"]["param"].is_null());
        assert!(v["error"]["code"].is_null());
    }

    #[tokio::test]
    async fn non_multipart_body_is_rejected_with_the_openai_envelope() {
        use axum::body::Body;
        let req = Request::builder()
            .method("POST")
            .uri("/v1/pii/redact/render")
            .header("content-type", "application/json")
            .body(Body::from("{}"))
            .unwrap();
        let resp = OaiMultipart::from_request(req, &())
            .await
            .err()
            .expect("a non-multipart body must be rejected");
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
        let v = envelope(resp).await;
        assert_eq!(v["error"]["type"], "invalid_request_error");
        assert_eq!(v["error"]["code"], "invalid_multipart");
        assert!(!v["error"]["message"].as_str().unwrap().is_empty());
    }

    #[tokio::test]
    async fn unavailable_and_server_map_to_distinct_types() {
        let v = envelope(error_response(
            StatusCode::SERVICE_UNAVAILABLE,
            "ocr not available".into(),
        ))
        .await;
        assert_eq!(v["error"]["type"], "service_unavailable_error");

        let v = envelope(error_response(
            StatusCode::INTERNAL_SERVER_ERROR,
            "render: boom".into(),
        ))
        .await;
        assert_eq!(v["error"]["type"], "internal_server_error");
    }

    #[test]
    fn an_undecodable_upload_is_the_clients_fault_not_the_servers() {
        assert_eq!(
            classify_error("decode image: The image format could not be determined"),
            StatusCode::BAD_REQUEST
        );
        assert_eq!(
            classify_error("ocr not available: set NV_OCR_TESSDATA"),
            StatusCode::SERVICE_UNAVAILABLE
        );
        assert_eq!(
            classify_error("encode PNG: out of memory"),
            StatusCode::INTERNAL_SERVER_ERROR
        );
    }
}
