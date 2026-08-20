use std::path::PathBuf;

use axum::body::Body;
use axum::http::{header, Request, StatusCode};
use axum::response::Response;
use tower::ServiceExt;

use speaches_plus::oapi::ocr::{load_got_from_env, router, OcrAppState};

const BOUNDARY: &str = "gotservingboundary";

fn multipart(parts: &[(&str, Option<&str>, &[u8])]) -> Vec<u8> {
    let mut body: Vec<u8> = Vec::new();
    for (name, filename, value) in parts {
        let disp = match filename {
            Some(f) => format!(
                "Content-Disposition: form-data; name=\"{name}\"; filename=\"{f}\"\r\nContent-Type: image/png\r\n"
            ),
            None => format!("Content-Disposition: form-data; name=\"{name}\"\r\n"),
        };
        body.extend_from_slice(format!("--{BOUNDARY}\r\n{disp}\r\n").as_bytes());
        body.extend_from_slice(value);
        body.extend_from_slice(b"\r\n");
    }
    body.extend_from_slice(format!("--{BOUNDARY}--\r\n").as_bytes());
    body
}

async fn post(app: axum::Router, parts: &[(&str, Option<&str>, &[u8])]) -> Response {
    let req = Request::builder()
        .method("POST")
        .uri("/v1/ocr")
        .header(
            header::CONTENT_TYPE,
            format!("multipart/form-data; boundary={BOUNDARY}"),
        )
        .body(Body::from(multipart(parts)))
        .unwrap();
    app.oneshot(req).await.unwrap()
}

async fn json_body(resp: Response) -> serde_json::Value {
    let bytes = axum::body::to_bytes(resp.into_body(), 1 << 20).await.unwrap();
    serde_json::from_slice(&bytes).unwrap()
}

async fn text_body(resp: Response) -> String {
    let bytes = axum::body::to_bytes(resp.into_body(), 1 << 24).await.unwrap();
    String::from_utf8_lossy(&bytes).to_string()
}

fn content_type(resp: &Response) -> String {
    resp.headers()
        .get(header::CONTENT_TYPE)
        .and_then(|v| v.to_str().ok())
        .unwrap_or_default()
        .to_string()
}

#[tokio::test]
async fn got_backend_unloaded_returns_the_documented_503() {
    let resp = post(
        router(OcrAppState::default()),
        &[("file", None, b"x"), ("backend", None, b"got")],
    )
    .await;
    assert_eq!(resp.status(), StatusCode::SERVICE_UNAVAILABLE);
    let v = json_body(resp).await;
    assert_eq!(v["error"]["code"], "ocr_backend_not_loaded");
    let msg = v["error"]["message"].as_str().unwrap();
    assert!(msg.contains("NV_OCR_GOT_DIR"), "{msg}");

    let resp = post(
        router(OcrAppState::default()),
        &[("file", None, b"x"), ("backend", None, b"got-ocr2")],
    )
    .await;
    assert_eq!(
        resp.status(),
        StatusCode::SERVICE_UNAVAILABLE,
        "backend=got-ocr2 must parse (503 unloaded, not 400 unknown backend)"
    );
}

fn snapshot_dir() -> Option<PathBuf> {
    if let Ok(d) = std::env::var("NV_OCR_GOT_DIR") {
        let p = PathBuf::from(d);
        return p.is_dir().then_some(p);
    }
    let home = std::env::var("HOME").ok()?;
    let snaps = PathBuf::from(home)
        .join(".cache/huggingface/hub/models--stepfun-ai--GOT-OCR-2.0-hf/snapshots");
    let mut entries: Vec<PathBuf> = std::fs::read_dir(&snaps)
        .ok()?
        .flatten()
        .map(|e| e.path())
        .filter(|p| p.is_dir())
        .collect();
    entries.sort();
    entries.into_iter().next()
}

fn fixture_png(name: &str) -> Vec<u8> {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../conformance/fixtures")
        .join(name)
        .join("input.png");
    std::fs::read(&path).expect("read fixture png")
}

#[tokio::test]
#[ignore]
async fn got_serving_reads_a_rendered_page() {
    if std::env::var("NV_GOT_OCR_TEST").as_deref() != Ok("1") {
        eprintln!(
            "got_serving_reads_a_rendered_page: PRECONDITION NOT MET, THIS TEST EXECUTED NOTHING. \
             set NV_GOT_OCR_TEST=1 and NV_OCR_GOT_DIR to the GOT-OCR-2.0-hf snapshot"
        );
        return;
    }
    let dir = snapshot_dir().expect("GOT-OCR-2.0-hf snapshot present");
    std::env::set_var("NV_OCR_GOT_DIR", &dir);
    let engine = load_got_from_env().expect("got engine loads");
    let png = fixture_png("070-ocr-paragraph");

    let resp = post(
        router(OcrAppState {
            got: Some(engine.clone()),
            ..Default::default()
        }),
        &[
            ("file", Some("page.png"), &png),
            ("backend", None, b"got"),
        ],
    )
    .await;
    assert_eq!(resp.status(), StatusCode::OK);
    assert!(content_type(&resp).starts_with("text/plain"), "{}", content_type(&resp));
    let body = text_body(resp).await;
    assert!(
        body.to_lowercase().contains("the quick brown fox"),
        "[real] got serving text missing rendered words: {body:?}"
    );

    let resp = post(
        router(OcrAppState {
            got: Some(engine),
            ..Default::default()
        }),
        &[
            ("file", Some("page.png"), &png),
            ("backend", None, b"got"),
            ("mode", None, b"markdown"),
        ],
    )
    .await;
    assert_eq!(resp.status(), StatusCode::OK);
    assert!(
        content_type(&resp).starts_with("text/markdown"),
        "markdown mode must return text/markdown, got {}",
        content_type(&resp)
    );
}
