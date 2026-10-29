use axum::extract::State;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::Json;
use serde::{Deserialize, Serialize};

use crate::oapi::chat::ChatAppState;

pub const LAYER_CAPTURE_DEFAULT_CONTEXT: usize = 2048;
pub const CAPTURE_NOT_AVAILABLE: &str = "layer capture is not available on this engine";

#[derive(Clone, Debug, Deserialize)]
pub struct LayerCaptureRequest {
    #[serde(default)]
    pub model: Option<String>,
    pub layer: usize,
    #[serde(default)]
    pub texts: Vec<String>,
    #[serde(default)]
    pub inputs: Vec<Vec<u32>>,
    #[serde(default)]
    pub tokens: bool,
    #[serde(default)]
    pub max_context: Option<usize>,
}

#[derive(Clone, Debug, Serialize)]
pub struct CaptureRecord {
    pub tokens: usize,
    pub chunks: usize,
    pub input_token_ids: Vec<u32>,
    pub mean: Vec<f32>,
    pub last: Vec<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub token_states: Option<Vec<Vec<f32>>>,
}

#[derive(Clone, Debug, Serialize)]
pub struct LayerCaptureResponse {
    pub model: String,
    pub layer: usize,
    pub num_layers: usize,
    pub hidden_size: usize,
    pub records: Vec<CaptureRecord>,
    pub seconds: f64,
}

#[derive(Clone, Debug, Serialize, PartialEq, Eq)]
pub struct CaptureInfo {
    pub model: String,
    pub family: String,
    pub capture: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub num_layers: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub hidden_size: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reason: Option<String>,
}

impl CaptureInfo {
    pub fn unavailable(model: &str, family: &str, reason: &str) -> Self {
        Self {
            model: model.to_string(),
            family: family.to_string(),
            capture: false,
            num_layers: None,
            hidden_size: None,
            reason: Some(reason.to_string()),
        }
    }

    pub fn available(model: &str, family: &str, num_layers: usize, hidden_size: usize) -> Self {
        Self {
            model: model.to_string(),
            family: family.to_string(),
            capture: true,
            num_layers: Some(num_layers),
            hidden_size: Some(hidden_size),
            reason: None,
        }
    }

    pub fn shape(&self) -> anyhow::Result<(usize, usize)> {
        match (self.capture, self.num_layers, self.hidden_size) {
            (true, Some(n), Some(h)) => Ok((n, h)),
            _ => anyhow::bail!(
                "{CAPTURE_NOT_AVAILABLE} ({}, model {}): {}",
                self.family,
                self.model,
                self.reason.as_deref().unwrap_or("no capture path")
            ),
        }
    }
}

pub fn checked_max_context(
    req: &LayerCaptureRequest,
    num_layers: usize,
    limits: &[usize],
) -> anyhow::Result<usize> {
    anyhow::ensure!(
        req.layer < num_layers,
        "layer {} out of range for {num_layers} blocks",
        req.layer
    );
    Ok(limits.iter().fold(
        req.max_context.unwrap_or(LAYER_CAPTURE_DEFAULT_CONTEXT),
        |acc, &l| acc.min(l),
    ))
}

pub fn check_input_len(ids: &[u32], max_context: usize) -> anyhow::Result<()> {
    anyhow::ensure!(
        !ids.is_empty() && ids.len() <= max_context,
        "each input must hold 1..={max_context} tokens"
    );
    Ok(())
}

pub fn capture_record(
    ids: Vec<u32>,
    rows: &candle_core::Tensor,
    chunks: usize,
    want_tokens: bool,
) -> anyhow::Result<CaptureRecord> {
    let rows = rows.to_dtype(candle_core::DType::F32)?;
    let (n, _) = rows.dims2()?;
    anyhow::ensure!(
        n == ids.len() && n > 0,
        "capture produced {n} rows for {} tokens",
        ids.len()
    );
    let mean = rows.mean(0)?.to_vec1::<f32>()?;
    let last = rows.narrow(0, n - 1, 1)?.flatten_all()?.to_vec1::<f32>()?;
    anyhow::ensure!(
        mean.iter().chain(&last).all(|v| v.is_finite()),
        "capture produced a non-finite activation"
    );
    let token_states = if want_tokens {
        Some(rows.to_vec2::<f32>()?)
    } else {
        None
    };
    Ok(CaptureRecord {
        tokens: n,
        chunks,
        input_token_ids: ids,
        mean,
        last,
        token_states,
    })
}

pub fn capture_runs(
    tokenizer: &tokenizers::Tokenizer,
    req: &LayerCaptureRequest,
    bos: Option<u32>,
    max_context: usize,
) -> anyhow::Result<Vec<(Vec<u32>, usize)>> {
    let mut runs: Vec<(Vec<u32>, usize)> = req.inputs.iter().map(|ids| (ids.clone(), 0)).collect();
    for text in &req.texts {
        let ids = tokenizer
            .encode(text.as_str(), false)
            .map_err(|e| anyhow::anyhow!("tokenize: {e}"))?
            .get_ids()
            .to_vec();
        runs.push(match bos {
            Some(b) if ids.first() != Some(&b) => (std::iter::once(b).chain(ids).collect(), 1),
            _ => (ids, 0),
        });
    }
    anyhow::ensure!(!runs.is_empty(), "texts or inputs required");
    for (ids, skip) in &runs {
        check_input_len(ids, max_context)?;
        check_input_len(&ids[*skip..], max_context)?;
    }
    Ok(runs)
}

fn error(status: StatusCode, kind: &str, message: String) -> Response {
    (
        status,
        Json(serde_json::json!({"error": {"type": kind, "message": message}})),
    )
        .into_response()
}

pub async fn handle(
    State(state): State<ChatAppState>,
    Json(req): Json<LayerCaptureRequest>,
) -> Response {
    let Some(engine) = state.registry.resolve_with(req.model.as_deref(), false) else {
        return error(
            StatusCode::NOT_FOUND,
            "model_not_found",
            format!(
                "capture model {} is not served here (serving: {})",
                req.model.as_deref().unwrap_or(""),
                state.registry.model_ids().join(", ")
            ),
        );
    };
    if req.texts.is_empty() && req.inputs.is_empty() {
        return error(
            StatusCode::BAD_REQUEST,
            "invalid_request_error",
            "texts or inputs required".into(),
        );
    }
    match engine.capture_layer(req).await {
        Ok(out) => Json(out).into_response(),
        Err(err) => {
            let msg = format!("{err:#}");
            let status = if msg.contains(CAPTURE_NOT_AVAILABLE) {
                StatusCode::NOT_IMPLEMENTED
            } else if msg.contains("retired") || msg.contains("busy") {
                StatusCode::SERVICE_UNAVAILABLE
            } else {
                StatusCode::BAD_REQUEST
            };
            error(status, "capture_error", msg)
        }
    }
}

pub async fn handle_info(State(state): State<ChatAppState>) -> Response {
    let models: Vec<CaptureInfo> = state
        .registry
        .entries_in_order()
        .iter()
        .map(|(_, engine)| engine.capture_info())
        .collect();
    Json(serde_json::json!({ "models": models })).into_response()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::oapi::chat::{ChatEngine, ChatEvent, ChatGenerateRequest};
    use crate::oapi::chat_engine::{ChatRegistry, EchoEngine};
    use axum::body::Body;
    use axum::http::Request;
    use std::sync::Arc;
    use tower::ServiceExt;

    struct FakeCapture {
        id: String,
        layers: usize,
        hidden: usize,
    }

    #[async_trait::async_trait]
    impl ChatEngine for FakeCapture {
        fn model_id(&self) -> &str {
            &self.id
        }

        async fn generate(
            &self,
            _req: ChatGenerateRequest,
            _tx: tokio::sync::mpsc::Sender<ChatEvent>,
        ) -> anyhow::Result<()> {
            Ok(())
        }

        fn capture_info(&self) -> CaptureInfo {
            CaptureInfo::available(&self.id, "fake", self.layers, self.hidden)
        }

        async fn capture_layer(
            &self,
            req: LayerCaptureRequest,
        ) -> anyhow::Result<LayerCaptureResponse> {
            let (num_layers, hidden_size) = self.capture_info().shape()?;
            let max_context = checked_max_context(&req, num_layers, &[16])?;
            let mut records = Vec::new();
            for text in &req.texts {
                let ids: Vec<u32> = text.bytes().map(u32::from).collect();
                check_input_len(&ids, max_context)?;
                let rows: Vec<f32> = ids
                    .iter()
                    .flat_map(|&t| {
                        (0..hidden_size).map(move |j| (t as f32) + (req.layer * 100 + j) as f32)
                    })
                    .collect();
                let rows = candle_core::Tensor::from_vec(
                    rows,
                    (ids.len(), hidden_size),
                    &candle_core::Device::Cpu,
                )?;
                records.push(capture_record(ids, &rows, 1, req.tokens)?);
            }
            Ok(LayerCaptureResponse {
                model: self.id.clone(),
                layer: req.layer,
                num_layers,
                hidden_size,
                records,
                seconds: 0.0,
            })
        }
    }

    fn router(engines: Vec<Arc<dyn ChatEngine>>) -> axum::Router {
        let registry = ChatRegistry::from_engines(engines).unwrap();
        axum::Router::new()
            .route("/v1/capture", axum::routing::post(handle).get(handle_info))
            .route("/v1/laguna/capture", axum::routing::post(handle))
            .with_state(ChatAppState { registry })
    }

    async fn call(
        app: &axum::Router,
        method: &str,
        uri: &str,
        body: serde_json::Value,
    ) -> (StatusCode, serde_json::Value) {
        let req = Request::builder()
            .method(method)
            .uri(uri)
            .header("content-type", "application/json")
            .body(if method == "GET" {
                Body::empty()
            } else {
                Body::from(body.to_string())
            })
            .unwrap();
        let resp = app.clone().oneshot(req).await.unwrap();
        let status = resp.status();
        let bytes = axum::body::to_bytes(resp.into_body(), 1 << 20)
            .await
            .unwrap();
        (status, serde_json::from_slice(&bytes).unwrap())
    }

    fn fake() -> Arc<dyn ChatEngine> {
        Arc::new(FakeCapture {
            id: "acme/dense-8".into(),
            layers: 8,
            hidden: 3,
        })
    }

    #[tokio::test]
    async fn capture_names_model_layer_depth_and_width_on_both_routes() {
        let app = router(vec![fake()]);
        for uri in ["/v1/capture", "/v1/laguna/capture"] {
            let (status, body) = call(
                &app,
                "POST",
                uri,
                serde_json::json!({"model": "acme/dense-8", "layer": 3, "texts": ["ab"], "tokens": true}),
            )
            .await;
            assert_eq!(status, StatusCode::OK, "{uri}: {body}");
            assert_eq!(body["model"], "acme/dense-8");
            assert_eq!(body["layer"], 3);
            assert_eq!(body["num_layers"], 8);
            assert_eq!(body["hidden_size"], 3);
            let rec = &body["records"][0];
            assert_eq!(rec["tokens"], 2);
            assert_eq!(rec["input_token_ids"], serde_json::json!([97, 98]));
            assert_eq!(rec["mean"], serde_json::json!([397.5, 398.5, 399.5]));
            assert_eq!(rec["last"], serde_json::json!([398.0, 399.0, 400.0]));
            assert_eq!(rec["token_states"].as_array().unwrap().len(), 2);
        }
    }

    #[tokio::test]
    async fn an_unknown_model_is_404_even_where_chat_falls_back_to_the_only_engine() {
        let registry = ChatRegistry::from_engines(vec![fake()]).unwrap();
        assert!(registry
            .resolve_with(Some("poolside/Laguna-XS-2.1-NVFP4"), true)
            .is_some());
        let app = router(vec![fake()]);
        let (status, body) = call(
            &app,
            "POST",
            "/v1/capture",
            serde_json::json!({"model": "poolside/Laguna-XS-2.1-NVFP4", "layer": 3, "texts": ["ab"]}),
        )
        .await;
        assert_eq!(status, StatusCode::NOT_FOUND);
        assert_eq!(body["error"]["type"], "model_not_found");
        let msg = body["error"]["message"].as_str().unwrap();
        assert!(
            msg.contains("poolside/Laguna-XS-2.1-NVFP4") && msg.contains("acme/dense-8"),
            "{msg}"
        );
    }

    #[tokio::test]
    async fn an_engine_without_a_capture_path_is_501_naming_its_family() {
        let echo: Arc<dyn ChatEngine> = Arc::new(EchoEngine::new("echo", "hi"));
        let app = router(vec![echo]);
        let (status, body) = call(
            &app,
            "POST",
            "/v1/capture",
            serde_json::json!({"layer": 1, "texts": ["ab"]}),
        )
        .await;
        assert_eq!(status, StatusCode::NOT_IMPLEMENTED);
        let msg = body["error"]["message"].as_str().unwrap();
        assert!(
            msg.starts_with(CAPTURE_NOT_AVAILABLE) && msg.contains("(engine, model echo)"),
            "{msg}"
        );
    }

    #[tokio::test]
    async fn out_of_range_layers_and_context_overflow_are_400_with_the_split_message() {
        let app = router(vec![fake()]);
        let (status, body) = call(
            &app,
            "POST",
            "/v1/capture",
            serde_json::json!({"layer": 8, "texts": ["ab"]}),
        )
        .await;
        assert_eq!(status, StatusCode::BAD_REQUEST);
        assert!(body["error"]["message"]
            .as_str()
            .unwrap()
            .contains("out of range for 8 blocks"));
        let (status, body) = call(
            &app,
            "POST",
            "/v1/capture",
            serde_json::json!({"layer": 1, "texts": ["x".repeat(17)]}),
        )
        .await;
        assert_eq!(status, StatusCode::BAD_REQUEST);
        assert_eq!(
            body["error"]["message"],
            "each input must hold 1..=16 tokens"
        );
    }

    #[tokio::test]
    async fn get_lists_every_engine_with_its_capture_shape_or_reason() {
        let echo: Arc<dyn ChatEngine> = Arc::new(EchoEngine::new("echo", "hi"));
        let app = router(vec![fake(), echo]);
        let (status, body) = call(&app, "GET", "/v1/capture", serde_json::Value::Null).await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(
            body["models"][0],
            serde_json::json!({"model": "acme/dense-8", "family": "fake", "capture": true, "num_layers": 8, "hidden_size": 3})
        );
        assert_eq!(body["models"][1]["capture"], false);
        assert_eq!(body["models"][1]["model"], "echo");
        assert_eq!(body["models"][1]["family"], "engine");
    }

    fn words() -> tokenizers::Tokenizer {
        let model = tokenizers::models::wordlevel::WordLevel::builder()
            .vocab(
                ["alpha", "beta", "<unk>"]
                    .iter()
                    .enumerate()
                    .map(|(i, t)| ((*t).to_string(), i as u32))
                    .collect(),
            )
            .unk_token("<unk>".into())
            .build()
            .unwrap();
        let mut tok = tokenizers::Tokenizer::new(model);
        tok.with_pre_tokenizer(Some(tokenizers::pre_tokenizers::whitespace::Whitespace {}));
        tok
    }

    fn texts(texts: &[&str], inputs: Vec<Vec<u32>>) -> LayerCaptureRequest {
        LayerCaptureRequest {
            model: None,
            layer: 0,
            texts: texts.iter().map(|t| t.to_string()).collect(),
            inputs,
            tokens: false,
            max_context: None,
        }
    }

    #[test]
    fn gemma_runs_carry_bos_for_context_but_report_only_the_text_rows() {
        let tok = words();
        let runs = capture_runs(&tok, &texts(&["alpha beta"], vec![vec![7]]), Some(9), 8).unwrap();
        assert_eq!(runs, vec![(vec![7], 0), (vec![9, 0, 1], 1)]);
        let runs = capture_runs(&tok, &texts(&["alpha beta"], vec![]), None, 8).unwrap();
        assert_eq!(runs, vec![(vec![0, 1], 0)]);
        let err = capture_runs(&tok, &texts(&["alpha beta"], vec![]), Some(9), 2).unwrap_err();
        assert_eq!(err.to_string(), "each input must hold 1..=2 tokens");
        let err = capture_runs(&tok, &texts(&[""], vec![]), Some(9), 8).unwrap_err();
        assert_eq!(err.to_string(), "each input must hold 1..=8 tokens");
    }

    #[test]
    fn capture_record_rejects_a_row_count_that_disagrees_with_the_ids() {
        let rows =
            candle_core::Tensor::zeros((2, 4), candle_core::DType::F32, &candle_core::Device::Cpu)
                .unwrap();
        assert!(capture_record(vec![1, 2, 3], &rows, 1, false).is_err());
        let nan =
            candle_core::Tensor::from_vec(vec![f32::NAN; 4], (1, 4), &candle_core::Device::Cpu)
                .unwrap();
        assert!(capture_record(vec![1], &nan, 1, false).is_err());
    }
}
