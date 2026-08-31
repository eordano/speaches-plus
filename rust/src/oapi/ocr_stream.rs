use std::convert::Infallible;
use std::sync::Arc;
use std::time::{Duration, Instant};

use axum::extract::{Multipart, State};
use axum::http::{HeaderMap, StatusCode};
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::{IntoResponse, Response};
use futures_util::StreamExt;
use serde_json::json;
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;
use tracing::{info, warn};

use nv_ocr::{DeepSeekMode, DotsOcrMode, OcrEngine, ResolutionHint};

use super::gate::SurfaceGate;
use super::ocr::{
    backend_name, bad_request, configured_default_backend, elapsed_ms, loaded_backends,
    parse_backend, parse_max_new_tokens, pick_default_backend, Backend, OcrRuntime,
};
use super::{kind, openai_error};

const HEARTBEAT_EVERY_SO_THE_VIEW_SEES_LIFE_DURING_A_LONG_DECODE: Duration =
    Duration::from_secs(1);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum StreamMode {
    Plain,
    Markdown,
    Layout,
    LayoutOnly,
}

fn parse_stream_mode(s: &str) -> Option<StreamMode> {
    match s {
        "plain" | "free-ocr" => Some(StreamMode::Plain),
        "markdown" => Some(StreamMode::Markdown),
        "layout" | "layout-all" => Some(StreamMode::Layout),
        "layout-only" => Some(StreamMode::LayoutOnly),
        _ => None,
    }
}

fn dots_mode(m: StreamMode) -> DotsOcrMode {
    match m {
        StreamMode::Plain => DotsOcrMode::PlainOcr,
        StreamMode::Markdown | StreamMode::Layout => DotsOcrMode::LayoutAll,
        StreamMode::LayoutOnly => DotsOcrMode::LayoutOnly,
    }
}

fn deepseek_mode(m: StreamMode) -> DeepSeekMode {
    match m {
        StreamMode::Markdown => DeepSeekMode::Markdown,
        _ => DeepSeekMode::FreeOcr,
    }
}

enum StreamOutcome {
    Dots(Result<nv_ocr::DotsPageResult, nv_ocr::Error>),
    Plain(Result<nv_ocr::OcrResult, nv_ocr::Error>),
}

async fn emit(tx: &mpsc::Sender<Event>, payload: serde_json::Value) -> bool {
    let data = payload.to_string();
    tx.send(Event::default().data(data)).await.is_ok()
}

fn stream_error_payload(err: &nv_ocr::Error) -> serde_json::Value {
    let (code, message) = match err {
        nv_ocr::Error::Decode(msg) => ("undecodable_image", format!("undecodable image: {msg}")),
        nv_ocr::Error::NotWired(msg) => ("ocr_backend_unimplemented", msg.to_string()),
        other => ("ocr_failed", format!("ocr: {other}")),
    };
    json!({"type": "error", "code": code, "message": message})
}

pub(super) async fn handle_ocr_stream(
    State(rt): State<OcrRuntime>,
    headers: HeaderMap,
    mut multipart: Multipart,
) -> Response {
    let client = super::deadline::from_headers(&headers);
    let mut file: Option<Vec<u8>> = None;
    let mut backend: Option<Backend> = None;
    let mut mode: Option<StreamMode> = None;
    let mut max_new_override: Option<usize> = None;

    loop {
        let field = match multipart.next_field().await {
            Ok(Some(f)) => f,
            Ok(None) => break,
            Err(err) => return bad_request(format!("multipart: {err}"), "body"),
        };
        let name = field.name().unwrap_or_default().to_string();
        match name.as_str() {
            "file" => match field.bytes().await {
                Ok(b) => file = Some(b.to_vec()),
                Err(err) => return bad_request(format!("reading file field: {err}"), "file"),
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
                        )
                    }
                },
                Err(err) => return bad_request(format!("reading backend field: {err}"), "backend"),
            },
            "mode" => match field.text().await.as_deref() {
                Ok("") => {}
                Ok(other) => match parse_stream_mode(other) {
                    Some(m) => mode = Some(m),
                    None => {
                        return bad_request(
                            format!(
                                "unknown mode {other:?}; expected plain, markdown, layout, or layout-only"
                            ),
                            "mode",
                        )
                    }
                },
                Err(err) => return bad_request(format!("reading mode field: {err}"), "mode"),
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
            _ => {}
        }
    }

    let Some(image_bytes) = file else {
        return bad_request("missing multipart field: file".into(), "file");
    };

    let backend = match backend {
        Some(b) => b,
        None => {
            let loaded = loaded_backends(&rt.app);
            match pick_default_backend(&loaded, configured_default_backend()) {
                Ok(b) => b,
                Err(err) => {
                    warn!(
                        reason = %err.message,
                        "ocr stream request has no usable default backend"
                    );
                    let k = if err.status == StatusCode::BAD_REQUEST {
                        kind::INVALID_REQUEST
                    } else {
                        kind::SERVICE_UNAVAIL
                    };
                    return openai_error(err.status, err.message, k, Some("backend"), Some(err.code));
                }
            }
        }
    };

    let mode = mode.unwrap_or(match backend {
        Backend::Dots => StreamMode::Layout,
        _ => StreamMode::Plain,
    });
    if matches!(mode, StreamMode::Layout | StreamMode::LayoutOnly) && backend != Backend::Dots {
        return bad_request(
            "mode=layout requires backend=dots; only dots.ocr emits layout boxes".into(),
            "mode",
        );
    }

    let engine = match backend {
        Backend::Tesseract => rt.app.tesseract.clone(),
        Backend::DeepSeek => rt.app.deepseek.clone(),
        Backend::Dots => rt.app.dots.clone(),
        Backend::Got => rt.app.got.clone(),
    };
    let Some(engine) = engine else {
        return openai_error(
            StatusCode::SERVICE_UNAVAILABLE,
            format!(
                "ocr backend not loaded: {} -- the same env knobs as /v1/ocr apply",
                backend_name(backend)
            ),
            kind::SERVICE_UNAVAIL,
            Some("backend"),
            Some("ocr_backend_not_loaded"),
        );
    };

    let gate = rt.gate_arc_for(backend);
    info!(
        backend = backend_name(backend),
        bytes = image_bytes.len(),
        max_new_tokens = max_new_override.unwrap_or(0),
        "ocr stream request accepted"
    );

    let (tx, rx) = mpsc::channel::<Event>(64);
    tokio::spawn(run_stream(
        tx,
        engine,
        gate,
        client,
        backend,
        mode,
        max_new_override,
        image_bytes,
    ));

    Sse::new(ReceiverStream::new(rx).map(Ok::<_, Infallible>))
        .keep_alive(
            KeepAlive::new()
                .interval(Duration::from_secs(10))
                .text("keep-alive"),
        )
        .into_response()
}

#[allow(clippy::too_many_arguments)]
async fn run_stream(
    tx: mpsc::Sender<Event>,
    engine: Arc<OcrEngine>,
    gate: Option<Arc<SurfaceGate>>,
    client: Option<Duration>,
    backend: Backend,
    mode: StreamMode,
    max_new_override: Option<usize>,
    image_bytes: Vec<u8>,
) {
    if !emit(
        &tx,
        json!({"type": "phase", "phase": "received",
               "backend": backend_name(backend), "bytes": image_bytes.len()}),
    )
    .await
    {
        return;
    }

    let dims_bytes = image_bytes.clone();
    let dims = tokio::task::spawn_blocking(move || {
        super::ocr_batch::decode_rgb(&dims_bytes)
            .map(|img| (img.w, img.h))
            .map_err(|e| format!("{e:#}"))
    })
    .await;
    match dims {
        Ok(Ok((w, h))) => {
            if !emit(&tx, json!({"type": "meta", "width": w, "height": h})).await {
                return;
            }
        }
        Ok(Err(msg)) => {
            emit(
                &tx,
                json!({"type": "error", "code": "undecodable_image",
                       "message": format!("undecodable image: {msg}")}),
            )
            .await;
            return;
        }
        Err(join) => {
            emit(
                &tx,
                json!({"type": "error", "code": "ocr_failed",
                       "message": format!("decode task join: {join}")}),
            )
            .await;
            return;
        }
    }

    let _permit = match gate {
        None => None,
        Some(gate) => {
            if !emit(&tx, json!({"type": "phase", "phase": "queued"})).await {
                return;
            }
            match gate.acquire_with_deadline(client).await {
                Ok(permit) => Some(permit),
                Err(_busy) => {
                    emit(
                        &tx,
                        json!({"type": "error", "code": "ocr_saturated",
                               "message": "the ocr surface was saturated for the whole queue window; retry later"}),
                    )
                    .await;
                    return;
                }
            }
        }
    };

    let t0 = Instant::now();
    if !emit(
        &tx,
        json!({"type": "phase", "phase": "recognizing", "elapsed_ms": 0.0}),
    )
    .await
    {
        return;
    }

    let mut job = tokio::task::spawn_blocking(move || match backend {
        Backend::Dots => StreamOutcome::Dots(engine.recognize_dots_layout_budgeted(
            &image_bytes,
            dots_mode(mode),
            max_new_override,
        )),
        _ => StreamOutcome::Plain(engine.recognize_hinted_budgeted(
            &image_bytes,
            deepseek_mode(mode),
            ResolutionHint::Auto,
            max_new_override,
        )),
    });
    let mut tick = tokio::time::interval(HEARTBEAT_EVERY_SO_THE_VIEW_SEES_LIFE_DURING_A_LONG_DECODE);
    tick.tick().await;
    let joined = loop {
        tokio::select! {
            res = &mut job => break res,
            _ = tick.tick() => {
                if !emit(
                    &tx,
                    json!({"type": "phase", "phase": "recognizing", "elapsed_ms": elapsed_ms(t0)}),
                )
                .await
                {
                    return;
                }
            }
        }
    };

    let outcome = match joined {
        Ok(o) => o,
        Err(join) => {
            emit(
                &tx,
                json!({"type": "error", "code": "ocr_failed",
                       "message": format!("ocr task join: {join}")}),
            )
            .await;
            return;
        }
    };

    match outcome {
        StreamOutcome::Dots(Ok(res)) => {
            for e in &res.layout.elements {
                if !emit(
                    &tx,
                    json!({"type": "region", "order": e.order, "bbox": e.bbox,
                           "category": e.category, "text": e.text}),
                )
                .await
                {
                    return;
                }
            }
            if res.layout.elements.is_empty() && !res.result.text.is_empty() {
                if !emit(&tx, json!({"type": "delta", "text": res.result.text})).await {
                    return;
                }
            }
            info!(
                backend = "dots",
                elapsed_ms = elapsed_ms(t0),
                elements = res.layout.elements.len(),
                chars = res.result.text.len(),
                "ocr stream request complete"
            );
            emit(
                &tx,
                json!({"type": "done", "text": res.result.text,
                       "elements": res.layout.elements.len(),
                       "truncated": res.result.truncated || res.layout.truncated,
                       "looped": res.result.looped,
                       "elapsed_ms": elapsed_ms(t0)}),
            )
            .await;
        }
        StreamOutcome::Plain(Ok(res)) => {
            if !res.text.is_empty() && !emit(&tx, json!({"type": "delta", "text": res.text})).await
            {
                return;
            }
            info!(
                backend = backend_name(backend),
                elapsed_ms = elapsed_ms(t0),
                chars = res.text.len(),
                "ocr stream request complete"
            );
            emit(
                &tx,
                json!({"type": "done", "text": res.text, "elements": 0,
                       "truncated": res.truncated, "looped": res.looped,
                       "elapsed_ms": elapsed_ms(t0)}),
            )
            .await;
        }
        StreamOutcome::Dots(Err(err)) | StreamOutcome::Plain(Err(err)) => {
            warn!(
                backend = backend_name(backend),
                elapsed_ms = elapsed_ms(t0),
                error = %err,
                "ocr stream request failed"
            );
            emit(&tx, stream_error_payload(&err)).await;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::body::Body;
    use axum::http::Request;
    use http_body_util::BodyExt;
    use nv_ocr::{LayoutBox, LayoutPageResult, OcrResult};
    use tower::ServiceExt;

    struct FakeDots;

    impl nv_ocr::DotsOcrModel for FakeDots {
        fn recognize_page(
            &self,
            image_bytes: &[u8],
            mode: nv_ocr::DotsOcrMode,
        ) -> Result<OcrResult, nv_ocr::Error> {
            Ok(self.recognize_layout(image_bytes, mode)?.result)
        }

        fn recognize_layout(
            &self,
            _image_bytes: &[u8],
            _mode: nv_ocr::DotsOcrMode,
        ) -> Result<nv_ocr::DotsPageResult, nv_ocr::Error> {
            Ok(nv_ocr::DotsPageResult {
                result: OcrResult {
                    text: "Local AI Stack Verification Report".into(),
                    tokens: Vec::new(),
                    truncated: false,
                    looped: false,
                },
                layout: LayoutPageResult {
                    elements: vec![LayoutBox {
                        order: 0,
                        bbox: Some([10.0, 20.0, 300.0, 60.0]),
                        category: Some("Title".into()),
                        text: Some("Local AI Stack Verification Report".into()),
                    }],
                    truncated: false,
                },
            })
        }
    }

    fn one_by_one_png() -> Vec<u8> {
        let mut buf = std::io::Cursor::new(Vec::new());
        let img = image::RgbImage::from_pixel(1, 1, image::Rgb([255, 255, 255]));
        image::DynamicImage::ImageRgb8(img)
            .write_to(&mut buf, image::ImageFormat::Png)
            .unwrap();
        buf.into_inner()
    }

    fn multipart_body(fields: &[(&str, &[u8])]) -> (String, Vec<u8>) {
        let boundary = "ocrstreamtestboundary";
        let mut body = Vec::new();
        for (name, value) in fields {
            body.extend_from_slice(format!("--{boundary}\r\n").as_bytes());
            if *name == "file" {
                body.extend_from_slice(
                    b"Content-Disposition: form-data; name=\"file\"; filename=\"t.png\"\r\n",
                );
                body.extend_from_slice(b"Content-Type: image/png\r\n\r\n");
            } else {
                body.extend_from_slice(
                    format!("Content-Disposition: form-data; name=\"{name}\"\r\n\r\n").as_bytes(),
                );
            }
            body.extend_from_slice(value);
            body.extend_from_slice(b"\r\n");
        }
        body.extend_from_slice(format!("--{boundary}--\r\n").as_bytes());
        (
            format!("multipart/form-data; boundary={boundary}"),
            body,
        )
    }

    async fn sse_frames(resp: axum::response::Response) -> Vec<serde_json::Value> {
        let bytes = resp.into_body().collect().await.unwrap().to_bytes();
        let text = String::from_utf8(bytes.to_vec()).unwrap();
        text.lines()
            .filter_map(|l| l.strip_prefix("data: "))
            .filter_map(|d| serde_json::from_str(d).ok())
            .collect()
    }

    fn fake_dots_router() -> axum::Router {
        let state = crate::oapi::ocr::OcrAppState {
            dots: Some(Arc::new(OcrEngine::from_dots(Box::new(FakeDots)))),
            ..Default::default()
        };
        crate::oapi::ocr::router(state)
    }

    #[tokio::test]
    async fn the_stream_emits_phases_meta_regions_then_done_in_order() {
        let png = one_by_one_png();
        let (ct, body) = multipart_body(&[("file", &png), ("mode", b"layout")]);
        let resp = fake_dots_router()
            .oneshot(
                Request::post("/v1/ocr/stream")
                    .header("content-type", ct)
                    .body(Body::from(body))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        assert_eq!(
            resp.headers()
                .get("content-type")
                .and_then(|v| v.to_str().ok()),
            Some("text/event-stream")
        );
        let frames = sse_frames(resp).await;
        let types: Vec<&str> = frames
            .iter()
            .filter_map(|f| f["type"].as_str())
            .filter(|t| *t != "phase")
            .collect();
        assert_eq!(
            types,
            ["meta", "region", "done"],
            "frames were: {frames:?}"
        );
        assert_eq!(frames[0]["type"], "phase", "the first frame announces receipt");
        let meta = frames.iter().find(|f| f["type"] == "meta").unwrap();
        assert_eq!(meta["width"], 1);
        assert_eq!(meta["height"], 1);
        let region = frames.iter().find(|f| f["type"] == "region").unwrap();
        assert_eq!(region["bbox"][0], 10.0);
        assert_eq!(region["category"], "Title");
        let done = frames.iter().find(|f| f["type"] == "done").unwrap();
        assert_eq!(done["text"], "Local AI Stack Verification Report");
        assert_eq!(done["elements"], 1);
        assert_eq!(done["truncated"], false);
    }

    #[tokio::test]
    async fn an_undecodable_image_streams_an_error_event_not_a_500() {
        let (ct, body) = multipart_body(&[("file", b"not a png"), ("mode", b"plain")]);
        let resp = fake_dots_router()
            .oneshot(
                Request::post("/v1/ocr/stream")
                    .header("content-type", ct)
                    .body(Body::from(body))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let frames = sse_frames(resp).await;
        let err = frames
            .iter()
            .find(|f| f["type"] == "error")
            .expect("an error frame must arrive on the open stream");
        assert_eq!(err["code"], "undecodable_image");
    }

    #[tokio::test]
    async fn a_layout_mode_request_against_a_non_dots_backend_is_a_400_before_the_stream_opens() {
        let png = one_by_one_png();
        let (ct, body) = multipart_body(&[
            ("file", &png),
            ("backend", b"deepseek"),
            ("mode", b"layout"),
        ]);
        let state = crate::oapi::ocr::OcrAppState {
            dots: Some(Arc::new(OcrEngine::from_dots(Box::new(FakeDots)))),
            ..Default::default()
        };
        let resp = crate::oapi::ocr::router(state)
            .oneshot(
                Request::post("/v1/ocr/stream")
                    .header("content-type", ct)
                    .body(Body::from(body))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
    }

    #[tokio::test]
    async fn a_missing_file_field_is_a_400_before_the_stream_opens() {
        let (ct, body) = multipart_body(&[("mode", b"plain")]);
        let resp = fake_dots_router()
            .oneshot(
                Request::post("/v1/ocr/stream")
                    .header("content-type", ct)
                    .body(Body::from(body))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
    }
}
