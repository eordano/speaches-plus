use std::path::PathBuf;
use std::sync::{Arc, OnceLock};

use axum::http::{header, HeaderMap, StatusCode};
use axum::response::{IntoResponse, Response};
use serde::Deserialize;
use serde_json::json;

use crate::oapi;
use crate::oapi::deadline;
use crate::oapi::gate::SurfaceGate;

#[derive(Debug, Deserialize)]
#[cfg_attr(
    feature = "ts-bindings",
    derive(ts_rs::TS),
    ts(export, optional_fields = nullable)
)]
pub struct TextEmbeddingRequest {
    #[serde(default)]
    pub model: Option<String>,
    #[cfg_attr(feature = "ts-bindings", ts(type = "string | Array<string>"))]
    pub input: serde_json::Value,
}

#[cfg(feature = "ts-bindings")]
mod ts_wire {
    #![allow(dead_code)]
    use ts_rs::TS;

    #[derive(TS)]
    #[ts(export)]
    struct TextEmbeddingItem {
        #[ts(type = "\"embedding\"")]
        object: (),
        index: u32,
        embedding: Vec<f32>,
    }

    #[derive(TS)]
    #[ts(export)]
    struct TextEmbeddingUsage {
        prompt_tokens: u32,
        total_tokens: u32,
    }

    #[derive(TS)]
    #[ts(export)]
    struct TextEmbeddingResponse {
        #[ts(type = "\"list\"")]
        object: (),
        data: Vec<TextEmbeddingItem>,
        model: String,
        usage: TextEmbeddingUsage,
    }
}

mod backend {
    use anyhow::{Context, Result};
    use candle_core::{DType, Device, IndexOp, Tensor};
    use nv_models::qwen3::{Qwen3, Qwen3Config};
    use nv_weights::WeightLoader;
    use std::sync::Mutex;
    use tokenizers::Tokenizer;

    pub struct QwenEmbedder {
        model: Mutex<Qwen3>,
        tokenizer: Tokenizer,
        device: Device,
        hidden_size: usize,
        max_seq_len: usize,
        model_id: String,
    }

    fn embedding_device() -> Result<Device> {
        #[cfg(feature = "cuda")]
        {
            Device::new_cuda(0).context("acquire cuda device")
        }
        #[cfg(all(feature = "metal", not(feature = "cuda")))]
        {
            match Device::new_metal(0) {
                Ok(device) => Ok(device),
                Err(err) => {
                    tracing::warn!(
                        error = %err,
                        "metal device unavailable for /v1/embeddings; falling back to CPU"
                    );
                    Ok(Device::Cpu)
                }
            }
        }
        #[cfg(not(any(feature = "cuda", feature = "metal")))]
        {
            Ok(Device::Cpu)
        }
    }

    impl QwenEmbedder {
        pub fn load(dir: &std::path::Path) -> Result<Self> {
            let device = embedding_device()?;
            if matches!(device, Device::Cpu) {
                return Self::load_on(dir, device);
            }
            match Self::load_on(dir, device.clone()).and_then(|e| {
                e.embed("probe")?;
                Ok(e)
            }) {
                Ok(embedder) => Ok(embedder),
                Err(err) => {
                    tracing::warn!(
                        error = %format!("{err:#}"),
                        device = ?device,
                        "text embedding model did not load or forward on the accelerator device; \
                         retrying on CPU"
                    );
                    Self::load_on(dir, Device::Cpu)
                }
            }
        }

        pub fn device_label(&self) -> &'static str {
            match self.device {
                Device::Cpu => "cpu",
                Device::Cuda(_) => "cuda",
                Device::Metal(_) => "metal",
            }
        }

        fn load_on(dir: &std::path::Path, device: Device) -> Result<Self> {
            let cfg_path = dir.join("config.json");
            let config = Qwen3Config::from_hf_json_file(&cfg_path)
                .with_context(|| format!("parse Qwen3 embedding {}", cfg_path.display()))?;
            let weights = WeightLoader::open_dir(dir, &device)
                .context("open weight loader for embedding model")?;
            let model = Qwen3::from_loader(config.clone(), &weights, &device)
                .context("instantiate Qwen3 embedding model")?;
            let tok_path = dir.join("tokenizer.json");
            let mut tokenizer = Tokenizer::from_file(&tok_path)
                .map_err(|e| anyhow::anyhow!("tokenizer load: {e}"))?;

            nv_tokenizer::sanitize_for_serving(&mut tokenizer);
            let tokenizer = tokenizer;
            let model_id = crate::oapi::model_ids::model_id_for_dir(dir);
            Ok(Self {
                model: Mutex::new(model),
                tokenizer,
                device,
                hidden_size: config.hidden_size,
                max_seq_len: 2048.min(config.max_position_embeddings),
                model_id,
            })
        }

        pub fn hidden_size(&self) -> usize {
            self.hidden_size
        }

        pub fn model_id(&self) -> &str {
            &self.model_id
        }

        pub fn embed(&self, text: &str) -> Result<(Vec<f32>, usize)> {
            let encoded = self
                .tokenizer
                .encode(text, true)
                .map_err(|e| anyhow::anyhow!("tokenize: {e}"))?;
            let mut ids: Vec<u32> = encoded.get_ids().to_vec();
            if ids.is_empty() {
                anyhow::bail!("empty tokenization");
            }
            if ids.len() > self.max_seq_len {
                tracing::warn!(
                    input_tokens = ids.len(),
                    max_seq_len = self.max_seq_len,
                    "/v1/embeddings input truncated; the tail of this input does not influence \
                     its vector"
                );
                ids.truncate(self.max_seq_len);
            }
            let seq = ids.len();
            let positions: Vec<i32> = (0..seq as i32).collect();
            let tokens =
                Tensor::from_vec(ids, (1usize, seq), &self.device).context("tokens tensor")?;
            let pos = Tensor::from_vec(positions, seq, &self.device).context("positions tensor")?;
            let model = self.model.lock().unwrap_or_else(|e| {
                tracing::warn!(
                    "embedding model mutex was poisoned by an earlier panic; recovering. \
                     Qwen3::forward_hidden takes &self and the KV cache is per-call, so there \
                     is no partially-updated state behind this lock."
                );
                e.into_inner()
            });
            let mut cache = model.new_kv_cache(seq).context("alloc kv cache")?;
            let hidden = model
                .forward_hidden(&tokens, &pos, &mut cache)
                .context("forward_hidden")?;
            let last = hidden
                .i((0usize, seq - 1, ..))
                .context("slice last token")?
                .to_dtype(DType::F32)
                .context("cast to f32")?;
            let v: Vec<f32> = last.to_vec1().context("to_vec1")?;
            Ok((l2_normalize(v), seq))
        }
    }

    fn l2_normalize(mut v: Vec<f32>) -> Vec<f32> {
        let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 0.0 {
            for x in &mut v {
                *x /= norm;
            }
        }
        v
    }
}

static EMBEDDER: OnceLock<Option<Arc<backend::QwenEmbedder>>> = OnceLock::new();
static EMBED_GATE: OnceLock<SurfaceGate> = OnceLock::new();

fn new_embed_gate() -> SurfaceGate {
    SurfaceGate::from_env(
        "/v1/embeddings",
        "NV_EMBED_CONCURRENCY",
        "NV_EMBED_QUEUE_MS",
        1,
        3000,
    )
}

fn embed_gate() -> &'static SurfaceGate {
    EMBED_GATE.get_or_init(new_embed_gate)
}

fn embedding_dir_from_env() -> Option<PathBuf> {
    std::env::var("NV_EMBEDDING_MODEL_DIR")
        .ok()
        .map(PathBuf::from)
}

fn get_embedder() -> Option<Arc<backend::QwenEmbedder>> {
    EMBEDDER
        .get_or_init(|| {
            let dir = embedding_dir_from_env()?;
            match backend::QwenEmbedder::load(&dir) {
                Ok(e) => Some(Arc::new(e)),
                Err(err) => {
                    tracing::warn!(
                        error = %format!("{err:#}"),
                        path = %dir.display(),
                        "text embedding model load failed"
                    );
                    None
                }
            }
        })
        .clone()
}

pub fn embedding_model_id() -> Option<String> {
    let dir = embedding_dir_from_env()?;
    Some(oapi::model_ids::model_id_for_dir(&dir))
}

pub fn loaded_embedding_model_id() -> Option<String> {
    match EMBEDDER.get() {
        Some(Some(_)) => embedding_model_id(),
        _ => None,
    }
}

pub fn warm_embedder_at_boot() -> Option<String> {
    embedding_dir_from_env()?;
    match get_embedder() {
        Some(embedder) => {
            let started = std::time::Instant::now();
            match embedder.embed("warmup") {
                Ok(_) => tracing::info!(
                    warmup_ms = started.elapsed().as_millis() as u64,
                    "text embedding warmup forward complete"
                ),
                Err(err) => tracing::warn!(
                    error = %err,
                    "text embedding warmup forward failed; the first real request will pay \
                     the lazy cuBLAS/CUDA init inside the concurrency permit"
                ),
            }
            let id = embedding_model_id();
            tracing::info!(
                model = %id.clone().unwrap_or_default(),
                device = embedder.device_label(),
                dim = embedder.hidden_size(),
                "text embedding model loaded"
            );
            id
        }
        None => {
            tracing::error!(
                "NV_EMBEDDING_MODEL_DIR is set but the embedding model did not load: \
                 /v1/embeddings will 503 and /v1/models will NOT advertise an embeddings model \
                 (earlier builds advertised it from the env var alone)"
            );
            None
        }
    }
}

pub async fn text_embeddings_post(body: axum::body::Bytes) -> Response {
    text_embeddings_post_with_headers(HeaderMap::new(), body).await
}

pub async fn text_embeddings_post_with_headers(
    headers: HeaderMap,
    body: axum::body::Bytes,
) -> Response {
    let client = deadline::from_headers(&headers);
    let req: TextEmbeddingRequest = match serde_json::from_slice(&body) {
        Ok(v) => v,
        Err(err) => {
            return oapi::openai_error(
                StatusCode::BAD_REQUEST,
                format!("invalid json: {err}"),
                oapi::kind::INVALID_REQUEST,
                Some("body"),
                Some("json_decode_error"),
            );
        }
    };
    let inputs: Vec<String> = match &req.input {
        serde_json::Value::String(s) => vec![s.clone()],
        serde_json::Value::Array(arr) => {
            let mut out = Vec::with_capacity(arr.len());
            for v in arr {
                if let Some(s) = v.as_str() {
                    out.push(s.to_string());
                } else {
                    return oapi::openai_error(
                        StatusCode::BAD_REQUEST,
                        "input array must contain only strings",
                        oapi::kind::INVALID_REQUEST,
                        Some("input"),
                        Some("invalid_type"),
                    );
                }
            }
            out
        }
        _ => {
            return oapi::openai_error(
                StatusCode::BAD_REQUEST,
                "input must be a string or array of strings",
                oapi::kind::INVALID_REQUEST,
                Some("input"),
                Some("invalid_type"),
            );
        }
    };

    if inputs.is_empty() {
        return oapi::openai_error(
            StatusCode::BAD_REQUEST,
            "input is empty",
            oapi::kind::INVALID_REQUEST,
            Some("input"),
            Some("empty"),
        );
    }

    let Some(embedder) = get_embedder() else {
        return oapi::openai_error(
            StatusCode::SERVICE_UNAVAILABLE,
            "text embedding model not loaded; set NV_EMBEDDING_MODEL_DIR",
            oapi::kind::SERVICE_UNAVAIL,
            None,
            Some("model_not_loaded"),
        );
    };

    let _permit = match embed_gate().acquire_with_deadline(client).await {
        Ok(permit) => permit,
        Err(busy) => {
            tracing::warn!(
                permits = embed_gate().permits(),
                queue_ms = embed_gate().queue_ms(),
                budget_ms = embed_gate().budget_ms(client),
                caller_deadline = client.is_some(),
                "embeddings request shed: the surface was saturated for the whole queue window"
            );
            return busy.into_response();
        }
    };

    let mut data_items: Vec<serde_json::Value> = Vec::with_capacity(inputs.len());
    let mut total_tokens: usize = 0;
    for (idx, text) in inputs.iter().enumerate() {
        let embedder_c = embedder.clone();
        let text_c = text.clone();
        let result = tokio::task::spawn_blocking(move || embedder_c.embed(&text_c)).await;
        let (vector, tokens) = match result {
            Ok(Ok(v)) => v,
            Ok(Err(err)) => {
                return oapi::openai_error(
                    StatusCode::INTERNAL_SERVER_ERROR,
                    format!("embed (input index {idx}): {err}"),
                    oapi::kind::SERVER,
                    None,
                    Some("embed_failed"),
                );
            }
            Err(err) => {
                return oapi::openai_error(
                    StatusCode::INTERNAL_SERVER_ERROR,
                    format!("join: {err}"),
                    oapi::kind::SERVER,
                    None,
                    Some("join_error"),
                );
            }
        };
        total_tokens += tokens;
        data_items.push(json!({
            "object": "embedding",
            "index": idx,
            "embedding": vector,
        }));
    }
    if let Some(requested) = req.model.as_deref() {
        if requested != embedder.model_id() {
            tracing::debug!(
                requested,
                served = embedder.model_id(),
                "/v1/embeddings ignores the requested model id: one embedding model is loaded, \
                 and the response reports the model that actually produced the vectors"
            );
        }
    }
    let body = json!({
        "object": "list",
        "data": data_items,
        "model": embedder.model_id(),
        "usage": {
            "prompt_tokens": total_tokens,
            "total_tokens": total_tokens,
        },
    });
    (
        StatusCode::OK,
        [(header::CONTENT_TYPE, "application/json")],
        body.to_string(),
    )
        .into_response()
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::body::Bytes;

    const REAL_EMBED_GATE: &str = "NV_EMBED_REAL_TEST";
    const QWEN3_EMBEDDING_0_6B_HIDDEN_SIZE: usize = 1024;
    const UNIT_NORM_TOLERANCE: f32 = 1e-3;
    const DEGENERATE_STDDEV_FLOOR: f32 = 1e-4;

    fn cosine(a: &[f32], b: &[f32]) -> f32 {
        a.iter().zip(b).map(|(x, y)| x * y).sum()
    }

    #[test]
    #[ignore]
    fn the_real_qwen3_embedding_checkpoint_loads_and_embeds_non_degenerately() {
        if std::env::var(REAL_EMBED_GATE).ok().as_deref() != Some("1") {
            panic!(
                "the_real_qwen3_embedding_checkpoint_loads_and_embeds_non_degenerately: \
                 PRECONDITION NOT MET, THIS TEST EXECUTED NOTHING. \
                 Set {REAL_EMBED_GATE}=1 to run it against real weights."
            );
        }
        let dir = embedding_dir_from_env()
            .expect("NV_EMBEDDING_MODEL_DIR must be set for the real-weight embedding gate");
        assert!(
            dir.join("model.safetensors").is_file(),
            "checkpoint does not realize: {} has no model.safetensors",
            dir.display()
        );
        eprintln!("checkpoint : {}", dir.display());

        let t = std::time::Instant::now();
        let embedder = backend::QwenEmbedder::load(&dir).expect("load Qwen3 embedding checkpoint");
        eprintln!("load       : {:.0} ms", t.elapsed().as_secs_f64() * 1e3);
        eprintln!("device     : {}", embedder.device_label());
        eprintln!("model_id   : {}", embedder.model_id());
        eprintln!("hidden     : {}", embedder.hidden_size());
        assert_eq!(
            embedder.hidden_size(),
            QWEN3_EMBEDDING_0_6B_HIDDEN_SIZE,
            "wrong checkpoint: hidden_size says this is not Qwen3-Embedding-0.6B"
        );

        let texts = [
            "The cat sat on the warm windowsill.",
            "A feline rested on the sunny window ledge.",
            "Quarterly revenue rose on strong enterprise demand.",
        ];
        let mut vectors = Vec::new();
        let t = std::time::Instant::now();
        for text in texts {
            let (v, seq) = embedder.embed(text).expect("embed real text");
            assert_eq!(v.len(), QWEN3_EMBEDDING_0_6B_HIDDEN_SIZE, "wrong vector dim");
            assert!(v.iter().all(|x| x.is_finite()), "vector has NaN or inf");
            let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
            assert!(
                (norm - 1.0).abs() < UNIT_NORM_TOLERANCE,
                "vector is not L2-normalised: norm={norm}"
            );
            let mean = v.iter().sum::<f32>() / v.len() as f32;
            let stddev =
                (v.iter().map(|x| (x - mean) * (x - mean)).sum::<f32>() / v.len() as f32).sqrt();
            assert!(
                stddev > DEGENERATE_STDDEV_FLOOR,
                "degenerate vector: stddev={stddev} (a constant or all-zero vector is a FAIL \
                 even though its shape is right)"
            );
            eprintln!(
                "embed seq={seq:>3} norm={norm:.6} mean={mean:+.6} stddev={stddev:.6} \
                 head={:+.4},{:+.4},{:+.4}",
                v[0], v[1], v[2]
            );
            vectors.push(v);
        }
        let embed_ms = t.elapsed().as_secs_f64() * 1e3;
        eprintln!("embed x{}   : {embed_ms:.0} ms total", texts.len());

        let paraphrase = cosine(&vectors[0], &vectors[1]);
        let unrelated = cosine(&vectors[0], &vectors[2]);
        eprintln!("cos(paraphrase) = {paraphrase:.4}");
        eprintln!("cos(unrelated)  = {unrelated:.4}");
        assert!(
            paraphrase > unrelated,
            "the embedding space is not semantic: paraphrase {paraphrase:.4} did not score \
             above unrelated {unrelated:.4}"
        );
        assert!(
            paraphrase < 0.9999,
            "distinct inputs collapsed to the same vector: cos={paraphrase:.6}"
        );
    }

    #[tokio::test]
    async fn text_embeddings_returns_503_when_model_unset() {
        std::env::remove_var("NV_EMBEDDING_MODEL_DIR");
        let body = Bytes::from(r#"{"model":"qwen3-embedding","input":"hello world"}"#);
        let resp = text_embeddings_post(body).await;
        assert_eq!(resp.status(), StatusCode::SERVICE_UNAVAILABLE);
    }

    #[tokio::test]
    async fn text_embeddings_rejects_invalid_input_type() {
        let body = Bytes::from(r#"{"model":"m","input":42}"#);
        let resp = text_embeddings_post(body).await;
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
    }

    #[tokio::test]
    async fn text_embeddings_rejects_empty_array_input() {
        let body = Bytes::from(r#"{"model":"m","input":[]}"#);
        let resp = text_embeddings_post(body).await;
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
    }

    #[tokio::test]
    async fn text_embeddings_rejects_bad_json() {
        let body = Bytes::from("not json");
        let resp = text_embeddings_post(body).await;
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
    }

    #[tokio::test]
    async fn embed_gate_admits_one_sheds_the_next_and_reuses_the_slot() {
        std::env::set_var("NV_EMBED_CONCURRENCY", "1");
        std::env::set_var("NV_EMBED_QUEUE_MS", "80");
        let gate = new_embed_gate();
        assert_eq!(gate.permits(), 1);
        assert_eq!(gate.queue_ms(), 80);

        let held = gate.acquire().await.expect("first request is admitted");
        let busy = gate
            .acquire()
            .await
            .expect_err("second request must shed while the only slot is held");
        let resp = busy.into_response();
        assert_eq!(resp.status(), StatusCode::SERVICE_UNAVAILABLE);

        drop(held);
        let _readmitted = gate
            .acquire()
            .await
            .expect("the slot is admitted again once the holder releases it");

        std::env::remove_var("NV_EMBED_CONCURRENCY");
        std::env::remove_var("NV_EMBED_QUEUE_MS");
        let defaults = new_embed_gate();
        assert_eq!(defaults.permits(), 1);
        assert_eq!(defaults.queue_ms(), 3000);
    }

    fn deadline_header(v: &str) -> HeaderMap {
        let mut h = HeaderMap::new();
        h.insert(
            deadline::HEADER,
            axum::http::HeaderValue::from_str(v).unwrap(),
        );
        h
    }

    #[tokio::test]
    async fn a_caller_deadline_sheds_the_embed_gate_earlier_than_the_server_window() {
        let gate = SurfaceGate::new("/v1/embeddings", 1, 3_000);
        let _held = gate.acquire().await.expect("hold the only permit");

        let client = deadline::from_headers(&deadline_header("120"));
        assert_eq!(client, Some(std::time::Duration::from_millis(120)));
        assert_eq!(gate.budget_ms(client), 120);

        let t0 = std::time::Instant::now();
        let busy = gate
            .acquire_with_deadline(client)
            .await
            .expect_err("a held gate must shed");
        let elapsed = t0.elapsed();
        assert_eq!(busy.waited_ms, 120);
        assert!(
            elapsed < std::time::Duration::from_millis(1_500),
            "the caller deadline was ignored: {elapsed:?}"
        );
        assert_eq!(
            busy.into_response().status(),
            StatusCode::SERVICE_UNAVAILABLE
        );
    }

    #[test]
    fn embed_header_precedence_is_header_then_server_default() {
        let gate = SurfaceGate::new("/v1/embeddings", 1, 3_000);
        assert_eq!(
            gate.budget_ms(deadline::from_headers(&HeaderMap::new())),
            3_000
        );
        assert_eq!(
            gate.budget_ms(deadline::from_headers(&deadline_header("soon"))),
            3_000
        );
        assert_eq!(
            gate.budget_ms(deadline::from_headers(&deadline_header("250"))),
            250
        );
        assert_eq!(
            gate.budget_ms(deadline::from_headers(&deadline_header("0"))),
            50
        );
        assert!(deadline::max_ms() < 99_999_999);
        assert_eq!(
            gate.budget_ms(deadline::from_headers(&deadline_header("99999999"))),
            deadline::max_ms(),
            "an absurd header must be clamped to the server maximum, not honoured"
        );
    }
}
