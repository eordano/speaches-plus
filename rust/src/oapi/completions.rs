use std::sync::Arc;

use axum::body::Body;
use axum::extract::State;
use axum::http::{header, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::Json;
use bytes::Bytes;
use serde::{Deserialize, Serialize};
use serde_json::json;
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;
use tracing::warn;

use crate::oapi::chat::json_ext::OaiJson;
use crate::oapi::chat::{
    now_unix_secs, resolve_guided_fields, send_sse_json, send_sse_raw, system_fingerprint,
    ChatAppState, ChatEngine, ChatEvent, ChatGenerateRequest, LogprobEntry, StopField,
    StreamOptions, Usage, DEFAULT_MAX_TOKENS, GUIDED_FORCE_THINK_OFF_ENV, MAX_MAX_TOKENS,
};
use crate::oapi::{kind, openai_error};

const MAX_LOGPROBS: u32 = 20;

#[derive(Clone, Debug, Deserialize)]
#[cfg_attr(
    feature = "ts-bindings",
    derive(ts_rs::TS),
    ts(export, optional_fields = nullable)
)]
pub struct CompletionRequest {
    pub model: Option<String>,
    pub prompt: PromptField,
    #[serde(default)]
    pub stream: Option<bool>,
    #[serde(default)]
    pub stream_options: Option<StreamOptions>,
    #[serde(default)]
    pub max_tokens: Option<usize>,
    #[serde(default)]
    pub temperature: Option<f32>,
    #[serde(default)]
    pub top_p: Option<f32>,
    #[serde(default)]
    pub top_k: Option<u32>,
    #[serde(default)]
    pub min_p: Option<f32>,
    #[serde(default)]
    pub presence_penalty: Option<f32>,
    #[serde(default)]
    pub frequency_penalty: Option<f32>,
    #[serde(default)]
    pub repetition_penalty: Option<f32>,
    #[serde(default)]
    pub seed: Option<u64>,
    #[serde(default)]
    pub stop: Option<StopField>,
    #[serde(default)]
    pub stop_token_ids: Option<Vec<u32>>,
    #[serde(default)]
    pub logit_bias: Option<std::collections::HashMap<String, f32>>,
    #[serde(default)]
    pub user: Option<String>,
    #[serde(default)]
    pub suffix: Option<String>,
    #[serde(default)]
    pub echo: Option<bool>,
    #[serde(default)]
    pub n: Option<u32>,
    #[serde(default)]
    pub best_of: Option<u32>,
    #[serde(default)]
    pub logprobs: Option<u32>,
    #[serde(default)]
    pub response_format: Option<serde_json::Value>,
    #[serde(default)]
    pub guided_json: Option<serde_json::Value>,
    #[serde(default)]
    pub guided_regex: Option<String>,
    #[serde(default)]
    pub guided_choice: Option<Vec<String>>,
}

#[derive(Clone, Debug, Deserialize)]
#[cfg_attr(feature = "ts-bindings", derive(ts_rs::TS), ts(export))]
#[serde(untagged)]
pub enum PromptField {
    One(String),
    Many(Vec<String>),
}

#[derive(Clone, Debug, Serialize)]
#[cfg_attr(feature = "ts-bindings", derive(ts_rs::TS), ts(export))]
pub struct CompletionResponse {
    pub id: String,
    #[cfg_attr(feature = "ts-bindings", ts(type = "\"text_completion\""))]
    pub object: &'static str,
    pub created: i64,
    pub model: String,
    pub choices: Vec<CompletionChoice>,
    pub usage: Usage,
    #[serde(skip_serializing_if = "Option::is_none")]
    #[cfg_attr(feature = "ts-bindings", ts(optional))]
    pub system_fingerprint: Option<String>,
}

#[derive(Clone, Debug, Serialize)]
#[cfg_attr(feature = "ts-bindings", derive(ts_rs::TS), ts(export))]
pub struct CompletionChoice {
    pub text: String,
    pub index: u32,
    pub finish_reason: Option<String>,
    #[cfg_attr(
        feature = "ts-bindings",
        ts(
            type = "{ tokens: Array<string>, token_logprobs: Array<number>, top_logprobs: \
                    Array<{ [token: string]: number }>, text_offset: Array<number> } | null"
        )
    )]
    pub logprobs: Option<serde_json::Value>,
}

#[derive(Clone, Debug, Serialize)]
#[cfg_attr(feature = "ts-bindings", derive(ts_rs::TS), ts(export))]
pub struct CompletionChunk {
    pub id: String,
    #[cfg_attr(feature = "ts-bindings", ts(type = "\"text_completion\""))]
    pub object: &'static str,
    pub created: i64,
    pub model: String,
    pub choices: Vec<CompletionChunkChoice>,
    #[serde(skip_serializing_if = "Option::is_none")]
    #[cfg_attr(feature = "ts-bindings", ts(optional))]
    pub usage: Option<Usage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    #[cfg_attr(feature = "ts-bindings", ts(optional))]
    pub system_fingerprint: Option<String>,
}

#[derive(Clone, Debug, Serialize)]
#[cfg_attr(feature = "ts-bindings", derive(ts_rs::TS), ts(export))]
pub struct CompletionChunkChoice {
    pub text: String,
    pub index: u32,
    pub finish_reason: Option<String>,
    #[cfg_attr(
        feature = "ts-bindings",
        ts(
            type = "{ tokens: Array<string>, token_logprobs: Array<number>, top_logprobs: \
                    Array<{ [token: string]: number }>, text_offset: Array<number> } | null"
        )
    )]
    pub logprobs: Option<serde_json::Value>,
}

fn reject_bad_logprobs(logprobs: Option<u32>) -> Option<Response> {
    match logprobs {
        Some(n) if n > MAX_LOGPROBS => Some(openai_error(
            StatusCode::BAD_REQUEST,
            format!("logprobs must be in [0, {MAX_LOGPROBS}]"),
            "invalid_request_error",
            Some("logprobs"),
            None,
        )),
        _ => None,
    }
}

fn logprob_fields(logprobs: Option<u32>) -> (bool, usize) {
    (logprobs.is_some(), logprobs.unwrap_or(0) as usize)
}

fn legacy_logprobs(entries: &[LogprobEntry], offset: &mut usize) -> serde_json::Value {
    let mut tokens = Vec::with_capacity(entries.len());
    let mut token_logprobs = Vec::with_capacity(entries.len());
    let mut top_logprobs = Vec::with_capacity(entries.len());
    let mut text_offset = Vec::with_capacity(entries.len());
    for e in entries {
        text_offset.push(*offset);
        *offset += e.token.chars().count();
        tokens.push(e.token.clone());
        token_logprobs.push(e.logprob);
        let mut top = serde_json::Map::new();
        for t in &e.top_logprobs {
            top.insert(t.token.clone(), json!(t.logprob));
        }
        top_logprobs.push(serde_json::Value::Object(top));
    }
    json!({
        "tokens": tokens,
        "token_logprobs": token_logprobs,
        "top_logprobs": top_logprobs,
        "text_offset": text_offset,
    })
}

fn guided_think_close_for(engine: &dyn ChatEngine, guided: bool) -> Option<String> {
    use crate::oapi::chat_engine::{guided_think_close_marker, template_thinking_default};
    let forced_off = std::env::var(GUIDED_FORCE_THINK_OFF_ENV).ok().as_deref() == Some("1");
    let thinking_on = !forced_off && template_thinking_default(engine).unwrap_or(false);
    guided_think_close_marker(engine, guided, thinking_on)
}

pub async fn handle_completions(
    State(state): State<ChatAppState>,
    OaiJson(req): OaiJson<CompletionRequest>,
) -> Response {
    let spec = crate::oapi::chat::spec_decode_header_for(&state.registry, req.model.as_deref());
    let mut resp = completions_impl(state, req).await;
    crate::oapi::chat::set_spec_decode_header(&mut resp, spec);
    resp
}

async fn completions_impl(state: ChatAppState, req: CompletionRequest) -> Response {
    let prompt = match &req.prompt {
        PromptField::One(s) => s.clone(),
        PromptField::Many(v) if v.len() == 1 => v[0].clone(),
        PromptField::Many(_) => {
            return openai_error(
                StatusCode::BAD_REQUEST,
                "multi-prompt (prompt as array of length > 1) is not supported yet",
                "invalid_request_error",
                Some("prompt"),
                None,
            );
        }
    };
    if prompt.is_empty() {
        return crate::oapi::fastapi_validation_error(vec![crate::oapi::missing_field(&[
            "body", "prompt",
        ])]);
    }
    let engine = match state.registry.resolve(req.model.as_deref()) {
        Some(e) => e,
        None => return crate::oapi::chat::model_not_found(req.model.as_deref().unwrap_or("")),
    };
    if let Some(resp) = reject_bad_logprobs(req.logprobs) {
        return resp;
    }
    if req.n.unwrap_or(1) > 1 || req.best_of.unwrap_or(1) > 1 {
        return openai_error(
            StatusCode::BAD_REQUEST,
            "n>1 / best_of>1 is not supported yet",
            "invalid_request_error",
            Some("n"),
            None,
        );
    }

    let guided = match resolve_guided_fields(
        req.response_format.as_ref(),
        req.guided_json.as_ref(),
        req.guided_regex.as_ref(),
        req.guided_choice.as_ref(),
    ) {
        Ok(g) => g,
        Err(resp) => return resp,
    };
    let guided_think_close = guided_think_close_for(engine.as_ref(), guided.is_some());
    let logit_bias: Vec<(u32, f32)> = req
        .logit_bias
        .as_ref()
        .map(|m| {
            m.iter()
                .filter_map(|(k, &v)| k.parse::<u32>().ok().map(|id| (id, v)))
                .collect()
        })
        .unwrap_or_default();

    let mut stop = match &req.stop {
        Some(StopField::One(s)) => vec![s.clone()],
        Some(StopField::Many(v)) => v.clone(),
        None => Vec::new(),
    };

    let _ = &req.stop_token_ids;
    stop.retain(|s| !s.is_empty());

    let max_new_tokens = req
        .max_tokens
        .unwrap_or(DEFAULT_MAX_TOKENS)
        .clamp(1, MAX_MAX_TOKENS);

    let echo_prefix = if req.echo.unwrap_or(false) {
        prompt.clone()
    } else {
        String::new()
    };

    let (want_logprobs, top_logprobs) = logprob_fields(req.logprobs);

    let gen = ChatGenerateRequest {
        prompt,
        max_new_tokens,
        stop,
        seed: req.seed,
        temperature: req.temperature,
        top_p: req.top_p,
        top_k: req.top_k,
        min_p: req.min_p,
        presence_penalty: req.presence_penalty,
        frequency_penalty: req.frequency_penalty,
        repetition_penalty: req.repetition_penalty,
        guided,
        guided_think_close,
        logit_bias,
        logprobs: want_logprobs,
        top_logprobs,
        kv_resume: None,
        kv_store: None,
        mm: None,
    };

    let stream = req.stream.unwrap_or(false);
    let include_usage = req
        .stream_options
        .as_ref()
        .map(|o| o.include_usage)
        .unwrap_or(false);
    let model_id = engine.model_id().to_string();
    let id = format!("cmpl-{}", uuid::Uuid::new_v4().simple());
    let created = now_unix_secs();

    if stream {
        run_streaming(
            engine,
            gen,
            id,
            model_id,
            created,
            echo_prefix,
            include_usage,
        )
        .await
    } else {
        run_non_streaming(engine, gen, id, model_id, created, echo_prefix).await
    }
}

async fn run_non_streaming(
    engine: Arc<dyn ChatEngine>,
    gen: ChatGenerateRequest,
    id: String,
    model: String,
    created: i64,
    echo_prefix: String,
) -> Response {
    let (tx, mut rx) = mpsc::channel::<ChatEvent>(64);
    if let Err(err) = engine.generate(gen, tx).await {
        warn!(error = %err, "completion engine.generate failed to start");
        return openai_error(
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("engine: {err}"),
            kind::SERVER,
            None,
            Some("engine_unavailable"),
        );
    }

    let mut lp_offset = echo_prefix.chars().count();
    let mut lp_entries: Vec<LogprobEntry> = Vec::new();
    let mut text = echo_prefix;
    let mut prompt_tokens: u32 = 0;
    let mut completion_tokens: u32 = 0;
    let mut finish_reason = String::from("stop");
    while let Some(ev) = rx.recv().await {
        match ev {
            ChatEvent::Started { prompt_tokens: p } => prompt_tokens = p,
            ChatEvent::PromptCached { .. }
            | ChatEvent::StoppedBy { .. }
            | ChatEvent::ReasoningDelta(_) => {}
            ChatEvent::TextDelta(s) => text.push_str(&s),
            ChatEvent::Logprob(e) => lp_entries.push(e),
            ChatEvent::Done {
                finish_reason: r,
                completion_tokens: c,
            } => {
                finish_reason = r;
                completion_tokens = c;
            }
            ChatEvent::Error(msg) => {
                return openai_error(
                    StatusCode::INTERNAL_SERVER_ERROR,
                    msg,
                    kind::SERVER,
                    None,
                    Some("engine_error"),
                );
            }
        }
    }

    let logprobs = if lp_entries.is_empty() {
        None
    } else {
        Some(legacy_logprobs(&lp_entries, &mut lp_offset))
    };

    let resp = CompletionResponse {
        id,
        object: "text_completion",
        created,
        system_fingerprint: Some(system_fingerprint(&model)),
        model,
        choices: vec![CompletionChoice {
            text,
            index: 0,
            finish_reason: Some(finish_reason),
            logprobs,
        }],
        usage: Usage {
            prompt_tokens,
            completion_tokens,
            total_tokens: prompt_tokens + completion_tokens,
            prompt_tokens_details: None,
        },
    };
    (StatusCode::OK, Json(resp)).into_response()
}

#[allow(clippy::too_many_arguments)]
async fn run_streaming(
    engine: Arc<dyn ChatEngine>,
    gen: ChatGenerateRequest,
    id: String,
    model: String,
    created: i64,
    echo_prefix: String,
    include_usage: bool,
) -> Response {
    let (tx_bytes, rx_bytes) = mpsc::channel::<Result<Bytes, std::io::Error>>(64);
    let (tx_ev, mut rx_ev) = mpsc::channel::<ChatEvent>(64);

    let want_logprobs = gen.logprobs;
    let mut lp_offset = echo_prefix.chars().count();

    if let Err(err) = engine.generate(gen, tx_ev).await {
        warn!(error = %err, "completion engine.generate failed to start (streaming)");
        return openai_error(
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("engine: {err}"),
            kind::SERVER,
            None,
            Some("engine_unavailable"),
        );
    }

    let id_s = id.clone();
    let model_s = model.clone();
    tokio::spawn(async move {
        if !echo_prefix.is_empty() {
            let chunk = CompletionChunk {
                id: id_s.clone(),
                object: "text_completion",
                created,
                model: model_s.clone(),
                choices: vec![CompletionChunkChoice {
                    text: echo_prefix,
                    index: 0,
                    finish_reason: None,
                    logprobs: None,
                }],
                usage: None,
                system_fingerprint: None,
            };
            if send_sse_json(&tx_bytes, &chunk).await.is_err() {
                return;
            }
        }

        let mut prompt_tokens: u32 = 0;
        let mut completion_tokens: u32 = 0;
        let mut finish_reason: Option<String> = None;
        let mut pending_text: Option<String> = None;
        while let Some(ev) = rx_ev.recv().await {
            match ev {
                ChatEvent::Started { prompt_tokens: p } => prompt_tokens = p,
                ChatEvent::PromptCached { .. }
                | ChatEvent::StoppedBy { .. }
                | ChatEvent::ReasoningDelta(_) => {}
                ChatEvent::TextDelta(s) if want_logprobs => {
                    pending_text = Some(match pending_text.take() {
                        Some(mut p) => {
                            p.push_str(&s);
                            p
                        }
                        None => s,
                    });
                }
                ChatEvent::TextDelta(s) => {
                    let chunk = CompletionChunk {
                        id: id_s.clone(),
                        object: "text_completion",
                        created,
                        model: model_s.clone(),
                        choices: vec![CompletionChunkChoice {
                            text: s,
                            index: 0,
                            finish_reason: None,
                            logprobs: None,
                        }],
                        usage: None,
                        system_fingerprint: None,
                    };
                    if send_sse_json(&tx_bytes, &chunk).await.is_err() {
                        return;
                    }
                }
                ChatEvent::Logprob(e) => {
                    let lp = legacy_logprobs(std::slice::from_ref(&e), &mut lp_offset);
                    let chunk = CompletionChunk {
                        id: id_s.clone(),
                        object: "text_completion",
                        created,
                        model: model_s.clone(),
                        choices: vec![CompletionChunkChoice {
                            text: pending_text.take().unwrap_or_default(),
                            index: 0,
                            finish_reason: None,
                            logprobs: Some(lp),
                        }],
                        usage: None,
                        system_fingerprint: None,
                    };
                    if send_sse_json(&tx_bytes, &chunk).await.is_err() {
                        return;
                    }
                }
                ChatEvent::Done {
                    finish_reason: r,
                    completion_tokens: c,
                } => {
                    finish_reason = Some(r);
                    completion_tokens = c;
                }
                ChatEvent::Error(msg) => {
                    let body = json!({"error": {"message": msg, "type": kind::SERVER}});
                    let _ = send_sse_raw(&tx_bytes, &body.to_string()).await;
                    let _ = send_sse_raw(&tx_bytes, "[DONE]").await;
                    return;
                }
            }
        }

        let last = CompletionChunk {
            id: id_s.clone(),
            object: "text_completion",
            created,
            model: model_s,
            choices: vec![CompletionChunkChoice {
                text: pending_text.take().unwrap_or_default(),
                index: 0,
                finish_reason: Some(finish_reason.unwrap_or_else(|| "stop".into())),
                logprobs: None,
            }],
            usage: if include_usage {
                Some(Usage {
                    prompt_tokens,
                    completion_tokens,
                    total_tokens: prompt_tokens + completion_tokens,
                    prompt_tokens_details: None,
                })
            } else {
                None
            },
            system_fingerprint: None,
        };
        let _ = send_sse_json(&tx_bytes, &last).await;
        let _ = send_sse_raw(&tx_bytes, "[DONE]").await;
    });

    let stream = ReceiverStream::new(rx_bytes);
    Response::builder()
        .status(StatusCode::OK)
        .header(header::CONTENT_TYPE, "text/event-stream")
        .header(header::CACHE_CONTROL, "no-cache")
        .body(Body::from_stream(stream))
        .unwrap()
}

#[cfg(test)]
mod tests {
    use super::*;

    const THINKING_TEMPLATE: &str = "{% for m in messages %}<|im_start|>{{ m['role'] }}\n{{ \
                                     m['content'] }}<|im_end|>\n{% endfor %}{% if \
                                     add_generation_prompt %}<|im_start|>assistant\n{% if not \
                                     enable_thinking %}<think>\n\n</think>\n\n{% endif %}{% endif \
                                     %}";

    struct TemplatedEngine(Arc<crate::oapi::chat_template::ChatTemplate>);

    #[async_trait::async_trait]
    impl ChatEngine for TemplatedEngine {
        fn model_id(&self) -> &str {
            "fixture"
        }

        async fn generate(
            &self,
            _req: ChatGenerateRequest,
            _tx: mpsc::Sender<ChatEvent>,
        ) -> anyhow::Result<()> {
            Ok(())
        }

        fn official_template(&self) -> Option<&crate::oapi::chat_template::ChatTemplate> {
            Some(&self.0)
        }
    }

    fn templated_engine(tag: &str, thinking_default: bool) -> TemplatedEngine {
        let dir = std::env::temp_dir().join(format!("completions-{tag}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("chat_template.jinja"), THINKING_TEMPLATE).unwrap();
        std::fs::write(
            dir.join("tokenizer_config.json"),
            format!(
                r#"{{"bos_token":"","eos_token":"","default_chat_template_kwargs":{{"enable_thinking":{thinking_default}}}}}"#
            ),
        )
        .unwrap();
        TemplatedEngine(
            crate::oapi::chat_template::ChatTemplate::load(&dir)
                .expect("the fixture template must load or this test proves nothing"),
        )
    }

    #[test]
    fn a_completion_asking_for_a_schema_lets_a_thinking_model_finish_its_thought() {
        assert!(
            std::env::var(GUIDED_FORCE_THINK_OFF_ENV).is_err(),
            "{GUIDED_FORCE_THINK_OFF_ENV} is set in this test process, so the derivation cannot \
             be checked here"
        );
        let on = templated_engine("think-on", true);
        assert_eq!(
            guided_think_close_for(&on, true).as_deref(),
            Some("</think>"),
            "/v1/completions hardcoded None here, so a thinking model asked for a schema had the \
             grammar applied from token 0 and the reasoning the template primed was written as \
             schema"
        );
        assert_eq!(
            guided_think_close_for(&on, false),
            None,
            "with no grammar there is nothing to defer and nothing to force closed"
        );

        let off = templated_engine("think-off", false);
        assert_eq!(
            guided_think_close_for(&off, true),
            None,
            "a model whose own default is thinking-off writes no thought to wait for; deferring \
             would spend the budget on prose and then force out a marker it never meant"
        );
    }

    #[test]
    fn the_shared_close_rule_reads_the_resolved_thinking_flag_not_the_template_default() {
        use crate::oapi::chat_engine::guided_think_close_marker;

        let off = templated_engine("shared-think-off", false);
        assert_eq!(
            guided_think_close_marker(&off, true, true).as_deref(),
            Some("</think>"),
            "/v1/chat/completions lets a request switch thinking on over a template default of \
             off; a second copy of this rule that only ever reads the template default would \
             bind the schema to the reasoning that request just asked for"
        );

        let on = templated_engine("shared-think-on", true);
        assert_eq!(
            guided_think_close_marker(&on, true, false),
            None,
            "a request that switched thinking off writes no thought to wait for, whatever the \
             template default says"
        );
        assert_eq!(
            guided_think_close_marker(&on, false, true),
            None,
            "with no grammar there is nothing to defer and nothing to force closed"
        );
        assert!(
            std::env::var(GUIDED_FORCE_THINK_OFF_ENV).is_err(),
            "{GUIDED_FORCE_THINK_OFF_ENV} is set in this test process, so the delegation cannot \
             be checked here"
        );
        assert_eq!(
            guided_think_close_for(&on, true),
            guided_think_close_marker(&on, true, true),
            "/v1/completions must not keep its own answer to the same question"
        );
    }

    #[test]
    fn deserialize_string_and_array_prompt() {
        let r: CompletionRequest =
            serde_json::from_str(r#"{"model":"x","prompt":"hello"}"#).unwrap();
        assert!(matches!(r.prompt, PromptField::One(ref s) if s == "hello"));
        let r: CompletionRequest = serde_json::from_str(r#"{"prompt":["a","b"]}"#).unwrap();
        assert!(matches!(r.prompt, PromptField::Many(ref v) if v.len() == 2));
    }

    #[test]
    fn deserialize_vllm_extensions() {
        let r: CompletionRequest = serde_json::from_str(
            r#"{"prompt":"x","stop_token_ids":[1,2],"guided_regex":"a+","echo":true,"min_p":0.1}"#,
        )
        .unwrap();
        assert_eq!(r.stop_token_ids.as_ref().unwrap(), &vec![1, 2]);
        assert_eq!(r.echo, Some(true));
    }

    #[test]
    fn response_shape() {
        let resp = CompletionResponse {
            id: "x".into(),
            object: "text_completion",
            created: 7,
            model: "m".into(),
            system_fingerprint: None,
            choices: vec![CompletionChoice {
                text: "hello".into(),
                index: 0,
                finish_reason: Some("stop".into()),
                logprobs: None,
            }],
            usage: Usage {
                prompt_tokens: 1,
                completion_tokens: 1,
                total_tokens: 2,
                prompt_tokens_details: None,
            },
        };
        let v = serde_json::to_value(&resp).unwrap();
        assert_eq!(v["object"], "text_completion");
        assert_eq!(v["choices"][0]["text"], "hello");
    }

    async fn body_json(resp: Response) -> serde_json::Value {
        let bytes = axum::body::to_bytes(resp.into_body(), 64 * 1024)
            .await
            .unwrap();
        serde_json::from_slice(&bytes).unwrap()
    }

    #[tokio::test]
    async fn malformed_json_uses_the_openai_error_envelope() {
        use axum::extract::{FromRequest, Request};
        let req = Request::builder()
            .method("POST")
            .uri("/v1/completions")
            .header("content-type", "application/json")
            .body(Body::from("{\"prompt\": ["))
            .unwrap();
        let resp = OaiJson::<CompletionRequest>::from_request(req, &())
            .await
            .err()
            .expect("malformed JSON must be rejected");
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
        let v = body_json(resp).await;
        assert_eq!(v["error"]["type"], "invalid_request_error");
        assert_eq!(v["error"]["code"], "invalid_json");
        assert!(!v["error"]["message"].as_str().unwrap().is_empty());
    }

    #[tokio::test]
    async fn missing_content_type_uses_the_openai_error_envelope() {
        use axum::extract::{FromRequest, Request};
        let req = Request::builder()
            .method("POST")
            .uri("/v1/completions")
            .body(Body::from("{\"prompt\":\"x\"}"))
            .unwrap();
        let resp = OaiJson::<CompletionRequest>::from_request(req, &())
            .await
            .err()
            .expect("a missing JSON content-type must be rejected");
        assert_eq!(resp.status(), StatusCode::UNSUPPORTED_MEDIA_TYPE);
        let v = body_json(resp).await;
        assert_eq!(v["error"]["type"], "invalid_request_error");
        assert_eq!(v["error"]["code"], "unsupported_content_type");
    }

    #[test]
    fn logprobs_reaches_the_generate_request() {
        assert_eq!(logprob_fields(None), (false, 0));
        assert_eq!(logprob_fields(Some(0)), (true, 0));
        assert_eq!(logprob_fields(Some(5)), (true, 5));
    }

    #[tokio::test]
    async fn logprobs_above_the_cap_is_rejected_not_dropped() {
        assert!(reject_bad_logprobs(None).is_none());
        assert!(reject_bad_logprobs(Some(MAX_LOGPROBS)).is_none());
        let resp = reject_bad_logprobs(Some(MAX_LOGPROBS + 1)).expect("must be rejected");
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
        let v = body_json(resp).await;
        assert_eq!(v["error"]["type"], "invalid_request_error");
        assert_eq!(v["error"]["param"], "logprobs");
    }

    #[test]
    fn legacy_logprobs_shape_and_offsets() {
        use crate::oapi::chat::TopLogprob;
        let entries = vec![
            LogprobEntry {
                token: "he".into(),
                logprob: -0.5,
                bytes: b"he".to_vec(),
                top_logprobs: vec![TopLogprob {
                    token: "he".into(),
                    logprob: -0.5,
                    bytes: b"he".to_vec(),
                }],
            },
            LogprobEntry {
                token: "llo".into(),
                logprob: -1.25,
                bytes: b"llo".to_vec(),
                top_logprobs: Vec::new(),
            },
        ];
        let mut offset = 4;
        let v = legacy_logprobs(&entries, &mut offset);
        assert_eq!(v["tokens"], json!(["he", "llo"]));
        assert_eq!(v["token_logprobs"], json!([-0.5, -1.25]));
        assert_eq!(v["text_offset"], json!([4, 6]));
        assert_eq!(v["top_logprobs"][0]["he"], json!(-0.5));
        assert!(v["top_logprobs"][1].as_object().unwrap().is_empty());
        assert_eq!(offset, 9);
    }
}
