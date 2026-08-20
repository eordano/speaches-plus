use std::sync::Arc;

use axum::body::Body;
use axum::extract::State;
use axum::http::{header, HeaderMap, HeaderName, HeaderValue, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::Json;
use bytes::Bytes;
use serde::Deserialize;
use serde_json::{json, Value};
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;
use tracing::warn;

use crate::oapi::chat::{
    engine_event_error_by_surface, engine_start_error_by_surface, is_busy_shed,
    parse_model_tool_calls, rand_seed, render_chat_checked_kwargs,
    resolve_guided_think_close, resolve_tool_policy, split_thinking, template_required_for,
    ChatAppState, ChatEngine,
    ChatEvent, ChatGenerateRequest, ChatMessageIn, ClientDeadline, EngineBusy, FunctionCall,
    FunctionDef, MessageContent, TemplateKwargs, ThinkPostProcess, ThinkingStream, Tool, ToolCall,
    ToolChoice, ToolPolicy, MAX_MAX_TOKENS, THINK_OPEN,
};

pub mod akind {
    pub const INVALID_REQUEST: &str = "invalid_request_error";
    pub const AUTH: &str = "authentication_error";
    pub const NOT_FOUND: &str = "not_found_error";
    pub const API: &str = "api_error";
    pub const OVERLOADED: &str = "overloaded_error";
}

pub const OVERLOADED_STATUS: u16 = 529;

pub fn anthropic_error_body(err_type: &str, message: &str) -> Value {
    json!({"type": "error", "error": {"type": err_type, "message": message}})
}

pub fn anthropic_error(status: StatusCode, err_type: &str, message: impl Into<String>) -> Response {
    let id = format!("req_{}", uuid::Uuid::new_v4().simple());
    let body = json!({
        "type": "error",
        "error": {"type": err_type, "message": message.into()},
        "request_id": id,
    });
    let mut resp = (status, Json(body)).into_response();
    if let Ok(v) = HeaderValue::from_str(&id) {
        resp.headers_mut()
            .insert(HeaderName::from_static("request-id"), v);
    }
    resp
}

fn overloaded(message: impl Into<String>) -> Response {
    anthropic_error(
        StatusCode::from_u16(OVERLOADED_STATUS).unwrap_or(StatusCode::SERVICE_UNAVAILABLE),
        akind::OVERLOADED,
        message,
    )
}

fn invalid(message: impl Into<String>) -> Response {
    anthropic_error(StatusCode::BAD_REQUEST, akind::INVALID_REQUEST, message)
}

fn with_request_id(mut resp: Response) -> Response {
    let id = format!("req_{}", uuid::Uuid::new_v4().simple());
    if let Ok(v) = HeaderValue::from_str(&id) {
        resp.headers_mut()
            .insert(HeaderName::from_static("request-id"), v);
    }
    resp
}

#[derive(Debug, Deserialize)]
#[cfg_attr(
    feature = "ts-bindings",
    derive(ts_rs::TS),
    ts(export, optional_fields = nullable)
)]
pub struct MessagesRequest {
    pub model: String,
    pub messages: Vec<MessageParam>,
    #[serde(default)]
    pub max_tokens: Option<u64>,
    #[serde(default)]
    #[cfg_attr(
        feature = "ts-bindings",
        ts(
            optional = nullable,
            type = "string | Array<{ type: \"text\", text: string }> | null"
        )
    )]
    pub system: Option<Value>,
    #[serde(default)]
    pub stop_sequences: Option<Vec<String>>,
    #[serde(default)]
    pub stream: Option<bool>,
    #[serde(default)]
    pub temperature: Option<f32>,
    #[serde(default)]
    pub top_p: Option<f32>,
    #[serde(default)]
    pub top_k: Option<u32>,
    #[serde(default)]
    #[cfg_attr(
        feature = "ts-bindings",
        ts(optional = nullable, type = "{ [key: string]: unknown } | null")
    )]
    pub metadata: Option<Value>,
    #[serde(default)]
    #[cfg_attr(
        feature = "ts-bindings",
        ts(
            optional = nullable,
            type = "Array<{ type?: string | null, name: string, description?: string | null, \
                    input_schema: { [key: string]: unknown } }> | null"
        )
    )]
    pub tools: Option<Vec<Value>>,
    #[serde(default)]
    pub tool_choice: Option<ToolChoiceParam>,
    #[serde(default)]
    pub thinking: Option<ThinkingParam>,
}

#[derive(Debug, Deserialize)]
#[cfg_attr(feature = "ts-bindings", derive(ts_rs::TS), ts(export))]
pub struct MessageParam {
    #[cfg_attr(
        feature = "ts-bindings",
        ts(type = "\"user\" | \"assistant\" | \"system\"")
    )]
    pub role: String,
    #[cfg_attr(
        feature = "ts-bindings",
        ts(
            type = "string | Array<{ type: \"text\", text: string } \
                    | { type: \"tool_use\", id: string, name: string, input: unknown } \
                    | { type: \"tool_result\", tool_use_id: string, content?: string \
                        | Array<{ type: \"text\", text: string }>, is_error?: boolean | null } \
                    | { type: \"thinking\", thinking: string, signature?: string | null } \
                    | { type: \"redacted_thinking\", data?: string | null }>"
        )
    )]
    pub content: Value,
}

#[derive(Debug, Deserialize)]
#[cfg_attr(
    feature = "ts-bindings",
    derive(ts_rs::TS),
    ts(export, optional_fields = nullable)
)]
pub struct ToolChoiceParam {
    #[serde(rename = "type")]
    #[cfg_attr(
        feature = "ts-bindings",
        ts(type = "\"auto\" | \"any\" | \"tool\" | \"none\"")
    )]
    pub kind: String,
    #[serde(default)]
    pub name: Option<String>,
    #[serde(default)]
    pub disable_parallel_tool_use: Option<bool>,
}

#[derive(Debug, Deserialize)]
#[cfg_attr(
    feature = "ts-bindings",
    derive(ts_rs::TS),
    ts(export, optional_fields = nullable)
)]
pub struct ThinkingParam {
    #[serde(rename = "type")]
    #[cfg_attr(
        feature = "ts-bindings",
        ts(type = "\"enabled\" | \"adaptive\" | \"disabled\"")
    )]
    pub kind: String,
    #[serde(default)]
    pub budget_tokens: Option<u64>,
    #[serde(default)]
    pub display: Option<String>,
}

struct Translated {
    messages: Vec<ChatMessageIn>,
    tools: Vec<Tool>,
    choice: ToolChoice,
    enable_thinking: Option<bool>,
}

fn text_of_blocks(v: &Value, loc: &str) -> Result<String, String> {
    match v {
        Value::String(s) => Ok(s.clone()),
        Value::Array(blocks) => {
            let mut out = String::new();
            for b in blocks {
                match b.get("type").and_then(Value::as_str) {
                    Some("text") => {
                        let t = b
                            .get("text")
                            .and_then(Value::as_str)
                            .ok_or_else(|| format!("{loc}: text block missing `text`"))?;
                        if !out.is_empty() {
                            out.push('\n');
                        }
                        out.push_str(t);
                    }
                    Some(other) => {
                        return Err(format!(
                            "{loc}: `{other}` content blocks are not supported by this server; \
                             only `text` is accepted here"
                        ))
                    }
                    None => return Err(format!("{loc}: content block missing `type`")),
                }
            }
            Ok(out)
        }
        _ => Err(format!("{loc}: expected a string or an array of content blocks")),
    }
}

fn push_text(msgs: &mut Vec<ChatMessageIn>, role: &str, buf: &mut String) {
    if buf.is_empty() {
        return;
    }
    msgs.push(ChatMessageIn {
        role: role.to_string(),
        content: Some(MessageContent::Text(std::mem::take(buf))),
        ..Default::default()
    });
}

fn translate(req: &MessagesRequest) -> Result<Translated, Response> {
    let mut messages: Vec<ChatMessageIn> = Vec::with_capacity(req.messages.len() + 1);

    if let Some(system) = &req.system {
        let text = text_of_blocks(system, "system").map_err(invalid)?;
        if !text.is_empty() {
            messages.push(ChatMessageIn {
                role: "system".into(),
                content: Some(MessageContent::Text(text)),
                ..Default::default()
            });
        }
    }

    for (i, m) in req.messages.iter().enumerate() {
        let loc = format!("messages[{i}]");
        match m.role.as_str() {
            "assistant" => {
                let mut text = String::new();
                let mut calls: Vec<ToolCall> = Vec::new();
                match &m.content {
                    Value::String(s) => text = s.clone(),
                    Value::Array(blocks) => {
                        for b in blocks {
                            match b.get("type").and_then(Value::as_str) {
                                Some("text") => {
                                    let t = b.get("text").and_then(Value::as_str).ok_or_else(
                                        || invalid(format!("{loc}: text block missing `text`")),
                                    )?;
                                    if !text.is_empty() {
                                        text.push('\n');
                                    }
                                    text.push_str(t);
                                }
                                Some("tool_use") => {
                                    let id = b
                                        .get("id")
                                        .and_then(Value::as_str)
                                        .unwrap_or_default()
                                        .to_string();
                                    let name = b
                                        .get("name")
                                        .and_then(Value::as_str)
                                        .ok_or_else(|| {
                                            invalid(format!(
                                                "{loc}: tool_use block missing `name`"
                                            ))
                                        })?
                                        .to_string();
                                    let input = b.get("input").cloned().unwrap_or(json!({}));
                                    calls.push(ToolCall {
                                        index: None,
                                        id,
                                        kind: "function".into(),
                                        function: FunctionCall {
                                            name,
                                            arguments: input.to_string(),
                                        },
                                    });
                                }
                                Some("thinking") | Some("redacted_thinking") => {}
                                Some(other) => {
                                    return Err(invalid(format!(
                                        "{loc}: `{other}` content blocks are not supported by \
                                         this server"
                                    )))
                                }
                                None => {
                                    return Err(invalid(format!(
                                        "{loc}: content block missing `type`"
                                    )))
                                }
                            }
                        }
                    }
                    _ => {
                        return Err(invalid(format!(
                            "{loc}.content: expected a string or an array of content blocks"
                        )))
                    }
                }
                messages.push(ChatMessageIn {
                    role: "assistant".into(),
                    content: Some(MessageContent::Text(text)),
                    tool_calls: (!calls.is_empty()).then_some(calls),
                    ..Default::default()
                });
            }
            role @ ("user" | "system") => {
                match &m.content {
                    Value::String(s) => {
                        let mut buf = s.clone();
                        push_text(&mut messages, role, &mut buf);
                    }
                    Value::Array(blocks) => {
                        let mut buf = String::new();
                        for b in blocks {
                            match b.get("type").and_then(Value::as_str) {
                                Some("text") => {
                                    let t = b.get("text").and_then(Value::as_str).ok_or_else(
                                        || invalid(format!("{loc}: text block missing `text`")),
                                    )?;
                                    if !buf.is_empty() {
                                        buf.push('\n');
                                    }
                                    buf.push_str(t);
                                }
                                Some("tool_result") => {
                                    push_text(&mut messages, role, &mut buf);
                                    let id = b
                                        .get("tool_use_id")
                                        .and_then(Value::as_str)
                                        .ok_or_else(|| {
                                            invalid(format!(
                                                "{loc}: tool_result block missing `tool_use_id`"
                                            ))
                                        })?
                                        .to_string();
                                    let body = match b.get("content") {
                                        None => String::new(),
                                        Some(c) => text_of_blocks(
                                            c,
                                            &format!("{loc}.content (tool_result)"),
                                        )
                                        .map_err(invalid)?,
                                    };
                                    let is_error = b
                                        .get("is_error")
                                        .and_then(Value::as_bool)
                                        .unwrap_or(false);
                                    let body = if is_error {
                                        format!("Error: {body}")
                                    } else {
                                        body
                                    };
                                    messages.push(ChatMessageIn {
                                        role: "tool".into(),
                                        content: Some(MessageContent::Text(body)),
                                        tool_call_id: Some(id),
                                        ..Default::default()
                                    });
                                }
                                Some(other) => {
                                    return Err(invalid(format!(
                                        "{loc}: `{other}` content blocks are not supported by \
                                         this server; only `text` and `tool_result` are accepted \
                                         in {role} messages"
                                    )))
                                }
                                None => {
                                    return Err(invalid(format!(
                                        "{loc}: content block missing `type`"
                                    )))
                                }
                            }
                        }
                        push_text(&mut messages, role, &mut buf);
                    }
                    _ => {
                        return Err(invalid(format!(
                            "{loc}.content: expected a string or an array of content blocks"
                        )))
                    }
                }
            }
            other => {
                return Err(invalid(format!(
                    "{loc}.role: expected `user` or `assistant`, got {other:?}"
                )))
            }
        }
    }

    if !messages.iter().any(|m| m.role != "system") {
        return Err(invalid(
            "messages: every turn rendered empty, so there is nothing to answer; a request whose \
             only content is an empty string or an empty block array is rejected rather than \
             answered from a promptless conversation"
                .to_string(),
        ));
    }

    let mut tools: Vec<Tool> = Vec::new();
    if let Some(raw_tools) = &req.tools {
        for (i, t) in raw_tools.iter().enumerate() {
            let loc = format!("tools[{i}]");
            match t.get("type").and_then(Value::as_str) {
                None | Some("custom") => {}
                Some(other) => {
                    return Err(invalid(format!(
                        "{loc}: server-side tool type {other:?} is not supported; only \
                         client tools with `name` and `input_schema` are accepted"
                    )))
                }
            }
            let name = t
                .get("name")
                .and_then(Value::as_str)
                .ok_or_else(|| invalid(format!("{loc}: missing `name`")))?
                .to_string();
            let description = t
                .get("description")
                .and_then(Value::as_str)
                .map(str::to_string);
            let parameters = t.get("input_schema").cloned();
            tools.push(Tool {
                kind: "function".into(),
                function: FunctionDef {
                    name,
                    description,
                    parameters,
                },
            });
        }
    }

    let choice = match &req.tool_choice {
        None => ToolChoice::Auto,
        Some(tc) => match tc.kind.as_str() {
            "auto" => ToolChoice::Auto,
            "any" => ToolChoice::Required,
            "none" => ToolChoice::None,
            "tool" => {
                let name = tc.name.clone().ok_or_else(|| {
                    invalid("tool_choice: `tool` requires a `name`".to_string())
                })?;
                ToolChoice::Function(name)
            }
            other => {
                return Err(invalid(format!(
                    "tool_choice.type: expected auto|any|tool|none, got {other:?}"
                )))
            }
        },
    };
    if let ToolChoice::Function(name) = &choice {
        if !tools.iter().any(|t| &t.function.name == name) {
            return Err(invalid(format!("tool_choice names unknown tool {name:?}")));
        }
    }
    let enable_thinking = match &req.thinking {
        None => None,
        Some(t) => match t.kind.as_str() {
            "enabled" | "adaptive" => Some(true),
            "disabled" => Some(false),
            other => {
                return Err(invalid(format!(
                    "thinking.type: expected enabled|adaptive|disabled, got {other:?}"
                )))
            }
        },
    };

    Ok(Translated {
        messages,
        tools,
        choice,
        enable_thinking,
    })
}

pub fn map_stop_reason(finish: &str) -> &'static str {
    match finish {
        "length" => "max_tokens",
        "tool_calls" => "tool_use",
        _ => "end_turn",
    }
}

fn tool_input_value(arguments: &str) -> Value {
    serde_json::from_str::<Value>(arguments)
        .ok()
        .filter(Value::is_object)
        .unwrap_or(json!({}))
}

fn content_blocks(
    reasoning: Option<&str>,
    text: Option<&str>,
    calls: &[ToolCall],
) -> Vec<Value> {
    let mut out = Vec::new();
    if let Some(r) = reasoning {
        if !r.is_empty() {
            out.push(json!({"type": "thinking", "thinking": r, "signature": ""}));
        }
    }
    if let Some(t) = text {
        if !t.is_empty() {
            out.push(json!({"type": "text", "text": t}));
        }
    }
    for c in calls {
        out.push(json!({
            "type": "tool_use",
            "id": c.id,
            "name": c.function.name,
            "input": tool_input_value(&c.function.arguments),
        }));
    }
    out
}

const MESSAGES_SHED_NOTE: &str =
    "shed an anthropic messages request: the surface was at capacity";

fn engine_start_error(err: &anyhow::Error) -> Response {
    engine_start_error_by_surface(err, MESSAGES_SHED_NOTE, overloaded, |m| {
        anthropic_error(StatusCode::INTERNAL_SERVER_ERROR, akind::API, m)
    })
}

fn engine_event_error(msg: String) -> Response {
    engine_event_error_by_surface(msg, MESSAGES_SHED_NOTE, overloaded, |m| {
        anthropic_error(StatusCode::INTERNAL_SERVER_ERROR, akind::API, m)
    })
}

async fn first_event(
    rx: &mut mpsc::Receiver<ChatEvent>,
    client: Option<ClientDeadline>,
) -> Result<Option<ChatEvent>, Response> {
    let Some(client) = client else {
        return Ok(None);
    };
    match tokio::time::timeout(client.budget, rx.recv()).await {
        Ok(ev) => match ev {
            Some(ChatEvent::Error(msg)) if is_busy_shed(&msg) => Err(engine_event_error(msg)),
            other => Ok(other),
        },
        Err(_) => Err(overloaded(format!(
            "generation did not start within the {} ms client budget ({})",
            client.budget_ms(),
            client.source
        ))),
    }
}

pub async fn handle_messages(
    State(state): State<ChatAppState>,
    headers: HeaderMap,
    body: Bytes,
) -> Response {
    let req: MessagesRequest = match serde_json::from_slice(&body) {
        Ok(r) => r,
        Err(err) => return invalid(format!("{err}")),
    };
    let Some(max_tokens) = req.max_tokens else {
        return invalid("max_tokens: field required".to_string());
    };
    if max_tokens == 0 {
        return invalid("max_tokens: must be at least 1".to_string());
    }
    if req.messages.is_empty() {
        return invalid("messages: at least one message is required".to_string());
    }

    let engine = match state.registry.resolve(Some(req.model.as_str())) {
        Some(e) => e,
        None => {
            return anthropic_error(
                StatusCode::NOT_FOUND,
                akind::NOT_FOUND,
                format!("model: {}", req.model),
            )
        }
    };

    let tr = match translate(&req) {
        Ok(t) => t,
        Err(resp) => return resp,
    };

    let mut kwargs = TemplateKwargs::new();
    if let Some(on) = tr.enable_thinking {
        kwargs.insert("enable_thinking".into(), Value::Bool(on));
    }
    let ToolPolicy {
        tools_active,
        force_name,
        guided,
    } = resolve_tool_policy(&tr.tools, &tr.choice, None);
    let guided_think_close =
        resolve_guided_think_close(engine.as_ref(), guided.is_some(), &mut kwargs);

    let render_tools: &[Tool] = if tools_active { &tr.tools } else { &[] };
    let template_required = template_required_for(engine.model_id());
    let prompt = match render_chat_checked_kwargs(
        engine.as_ref(),
        &tr.messages,
        render_tools,
        &tr.choice,
        template_required,
        &kwargs,
    ) {
        Ok(p) => p,
        Err(message) => {
            tracing::error!(model = %engine.model_id(), reason = %message, "messages request refused");
            return anthropic_error(StatusCode::INTERNAL_SERVER_ERROR, akind::API, message);
        }
    };
    let think = ThinkPostProcess {
        active: engine.thinking_split_supported(),
        opened: prompt.trim_end().ends_with(THINK_OPEN),
    };

    let gen = ChatGenerateRequest {
        prompt,
        max_new_tokens: max_tokens.min(MAX_MAX_TOKENS as u64).max(1) as usize,
        stop: req.stop_sequences.clone().unwrap_or_default(),
        seed: Some(rand_seed()),
        temperature: req.temperature,
        top_p: req.top_p,
        top_k: req.top_k,
        min_p: None,
        presence_penalty: None,
        frequency_penalty: None,
        repetition_penalty: None,
        guided,
        guided_think_close,
        logit_bias: Vec::new(),
        logprobs: false,
        top_logprobs: 0,
        kv_resume: None,
        kv_store: None,
        mm: None,
    };

    let client = ClientDeadline::from_request(None, &headers);
    let model_id = engine.model_id().to_string();
    let id = format!("msg_{}", uuid::Uuid::new_v4().simple());

    if req.stream.unwrap_or(false) {
        run_streaming(engine, gen, id, model_id, tools_active, force_name, think, client).await
    } else {
        run_non_streaming(engine, gen, id, model_id, tools_active, force_name, think, client).await
    }
}

#[allow(clippy::too_many_arguments)]
async fn run_non_streaming(
    engine: Arc<dyn ChatEngine>,
    gen: ChatGenerateRequest,
    id: String,
    model: String,
    tools_active: bool,
    force_name: Option<String>,
    think: ThinkPostProcess,
    client: Option<ClientDeadline>,
) -> Response {
    let (tx, mut rx) = mpsc::channel::<ChatEvent>(64);
    if let Err(err) = engine.generate(gen, tx).await {
        warn!(error = %err, "messages engine.generate failed to start");
        return engine_start_error(&err);
    }
    let mut pre = match first_event(&mut rx, client).await {
        Ok(ev) => ev,
        Err(resp) => return resp,
    };

    let mut input_tokens: u32 = 0;
    let mut output_tokens: u32 = 0;
    let mut cached: Option<u32> = None;
    let mut stop_seq: Option<String> = None;
    let mut finish = String::from("stop");
    let mut text = String::new();
    loop {
        let Some(ev) = (match pre.take() {
            Some(ev) => Some(ev),
            None => rx.recv().await,
        }) else {
            break;
        };
        match ev {
            ChatEvent::Started { prompt_tokens } => input_tokens = prompt_tokens,
            ChatEvent::PromptCached { cached_tokens } => cached = Some(cached_tokens),
            ChatEvent::StoppedBy { stop_sequence } => stop_seq = Some(stop_sequence),
            ChatEvent::TextDelta(s) => text.push_str(&s),
            ChatEvent::ReasoningDelta(_) => {}
            ChatEvent::Logprob(_) => {}
            ChatEvent::Done {
                finish_reason,
                completion_tokens,
            } => {
                finish = finish_reason;
                output_tokens = completion_tokens;
            }
            ChatEvent::Error(msg) => return engine_event_error(msg),
        }
    }

    let (reasoning, text) = match think.active {
        true => match split_thinking(&text, think.opened) {
            Some((r, c)) => ((!r.is_empty()).then_some(r), c),
            None => (None, text),
        },
        false => (None, text),
    };
    let (text, calls, finish) = if tools_active {
        let parsed = parse_model_tool_calls(&text, force_name.as_deref());
        if parsed.tool_calls.is_empty() {
            (parsed.content.unwrap_or_default(), Vec::new(), finish)
        } else {
            (
                parsed.content.unwrap_or_default(),
                parsed.tool_calls,
                "tool_calls".to_string(),
            )
        }
    } else {
        (text, Vec::new(), finish)
    };

    let stop_reason = if finish == "stop" && stop_seq.is_some() {
        "stop_sequence"
    } else {
        map_stop_reason(&finish)
    };
    let body = json!({
        "id": id,
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": content_blocks(reasoning.as_deref(), Some(&text), &calls),
        "stop_reason": stop_reason,
        "stop_sequence": stop_seq,
        "usage": {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cache_creation_input_tokens": 0,
            "cache_read_input_tokens": cached.unwrap_or(0),
        },
    });
    with_request_id((StatusCode::OK, Json(body)).into_response())
}

async fn send_event(
    tx: &mpsc::Sender<Result<Bytes, std::io::Error>>,
    event: &str,
    data: &Value,
) -> Result<(), ()> {
    let frame = format!("event: {event}\ndata: {data}\n\n");
    tx.send(Ok(Bytes::from(frame.into_bytes())))
        .await
        .map_err(|_| ())
}

struct BlockWriter<'a> {
    tx: &'a mpsc::Sender<Result<Bytes, std::io::Error>>,
    index: u32,
    open: Option<&'static str>,
}

impl<'a> BlockWriter<'a> {
    fn new(tx: &'a mpsc::Sender<Result<Bytes, std::io::Error>>) -> Self {
        Self {
            tx,
            index: 0,
            open: None,
        }
    }

    async fn start(&mut self, block: Value) -> Result<(), ()> {
        send_event(
            self.tx,
            "content_block_start",
            &json!({"type": "content_block_start", "index": self.index, "content_block": block}),
        )
        .await
    }

    async fn delta(&self, delta: Value) -> Result<(), ()> {
        send_event(
            self.tx,
            "content_block_delta",
            &json!({"type": "content_block_delta", "index": self.index, "delta": delta}),
        )
        .await
    }

    async fn close(&mut self) -> Result<(), ()> {
        if self.open.is_none() {
            return Ok(());
        }
        if self.open == Some("thinking") {
            self.delta(json!({"type": "signature_delta", "signature": ""}))
                .await?;
        }
        send_event(
            self.tx,
            "content_block_stop",
            &json!({"type": "content_block_stop", "index": self.index}),
        )
        .await?;
        self.open = None;
        self.index += 1;
        Ok(())
    }

    async fn ensure(&mut self, kind: &'static str) -> Result<(), ()> {
        if self.open == Some(kind) {
            return Ok(());
        }
        self.close().await?;
        let block = match kind {
            "thinking" => json!({"type": "thinking", "thinking": "", "signature": ""}),
            _ => json!({"type": "text", "text": ""}),
        };
        self.start(block).await?;
        self.open = Some(kind);
        Ok(())
    }

    async fn thinking(&mut self, s: &str) -> Result<(), ()> {
        if s.is_empty() {
            return Ok(());
        }
        self.ensure("thinking").await?;
        self.delta(json!({"type": "thinking_delta", "thinking": s}))
            .await
    }

    async fn text(&mut self, s: &str) -> Result<(), ()> {
        if s.is_empty() {
            return Ok(());
        }
        self.ensure("text").await?;
        self.delta(json!({"type": "text_delta", "text": s})).await
    }

    async fn tool_use(&mut self, call: &ToolCall) -> Result<(), ()> {
        self.close().await?;
        self.start(json!({
            "type": "tool_use",
            "id": call.id,
            "name": call.function.name,
            "input": {},
        }))
        .await?;
        self.open = Some("tool_use");
        let input = tool_input_value(&call.function.arguments);
        self.delta(json!({"type": "input_json_delta", "partial_json": input.to_string()}))
            .await?;
        self.close().await
    }
}

#[allow(clippy::too_many_arguments)]
async fn run_streaming(
    engine: Arc<dyn ChatEngine>,
    gen: ChatGenerateRequest,
    id: String,
    model: String,
    tools_active: bool,
    force_name: Option<String>,
    think: ThinkPostProcess,
    client: Option<ClientDeadline>,
) -> Response {
    let (tx_bytes, rx_bytes) = mpsc::channel::<Result<Bytes, std::io::Error>>(64);

    let (tx_ev, mut rx_ev) = mpsc::channel::<ChatEvent>(64);
    if let Err(err) = engine.generate(gen, tx_ev).await {
        warn!(error = %err, "messages engine.generate failed to start");
        return engine_start_error(&err);
    }
    let mut pre = match first_event(&mut rx_ev, client).await {
        Ok(ev) => ev,
        Err(resp) => return resp,
    };

    tokio::spawn(async move {
        let mut input_tokens: u32 = 0;
        let mut cached: Option<u32> = None;
        if pre.is_none() {
            pre = rx_ev.recv().await;
        }
        if let Some(ChatEvent::PromptCached { cached_tokens }) = pre {
            cached = Some(cached_tokens);
            pre = rx_ev.recv().await;
        }
        if let Some(ChatEvent::Started { prompt_tokens }) = pre {
            input_tokens = prompt_tokens;
            pre = None;
        }
        let start = json!({
            "type": "message_start",
            "message": {
                "id": id,
                "type": "message",
                "role": "assistant",
                "model": model,
                "content": [],
                "stop_reason": Value::Null,
                "stop_sequence": Value::Null,
                "usage": {
                    "input_tokens": input_tokens,
                    "output_tokens": 0,
                    "cache_creation_input_tokens": 0,
                    "cache_read_input_tokens": cached.unwrap_or(0),
                },
            },
        });
        if send_event(&tx_bytes, "message_start", &start).await.is_err() {
            return;
        }
        if send_event(&tx_bytes, "ping", &json!({"type": "ping"}))
            .await
            .is_err()
        {
            return;
        }

        let mut w = BlockWriter::new(&tx_bytes);
        let mut output_tokens: u32 = 0;
        let mut stop_seq: Option<String> = None;
        let mut finish = String::from("stop");
        let mut buf = String::new();
        let mut splitter = ThinkingStream::new(think.opened);
        loop {
            let Some(ev) = (match pre.take() {
                Some(ev) => Some(ev),
                None => rx_ev.recv().await,
            }) else {
                break;
            };
            match ev {
                ChatEvent::Started { prompt_tokens } => input_tokens = prompt_tokens,
                ChatEvent::PromptCached { cached_tokens } => cached = Some(cached_tokens),
                ChatEvent::StoppedBy { stop_sequence } => stop_seq = Some(stop_sequence),
                ChatEvent::ReasoningDelta(_) => {}
                ChatEvent::TextDelta(s) if tools_active => buf.push_str(&s),
                ChatEvent::TextDelta(s) if think.active => {
                    let (r, c) = splitter.push(&s);
                    if w.thinking(&r).await.is_err() || w.text(&c).await.is_err() {
                        return;
                    }
                }
                ChatEvent::TextDelta(s) => {
                    if w.text(&s).await.is_err() {
                        return;
                    }
                }
                ChatEvent::Logprob(_) => {}
                ChatEvent::Done {
                    finish_reason,
                    completion_tokens,
                } => {
                    finish = finish_reason;
                    output_tokens = completion_tokens;
                }
                ChatEvent::Error(msg) => {
                    let kind = if is_busy_shed(&msg) {
                        akind::OVERLOADED
                    } else {
                        akind::API
                    };
                    let _ = send_event(&tx_bytes, "error", &anthropic_error_body(kind, &msg)).await;
                    return;
                }
            }
        }

        if tools_active {
            let mut reasoning: Option<String> = None;
            if think.active {
                if let Some((r, c)) = split_thinking(&buf, think.opened) {
                    reasoning = (!r.is_empty()).then_some(r);
                    buf = c;
                }
            }
            let parsed = parse_model_tool_calls(&buf, force_name.as_deref());
            if let Some(r) = reasoning {
                if w.thinking(&r).await.is_err() {
                    return;
                }
            }
            if let Some(c) = parsed.content {
                if w.text(&c).await.is_err() {
                    return;
                }
            }
            if !parsed.tool_calls.is_empty() {
                finish = "tool_calls".into();
                for call in &parsed.tool_calls {
                    if w.tool_use(call).await.is_err() {
                        return;
                    }
                }
            }
        } else if think.active {
            let (r, c) = splitter.finish();
            if w.thinking(&r).await.is_err() || w.text(&c).await.is_err() {
                return;
            }
        }
        if w.close().await.is_err() {
            return;
        }

        let stop_reason = if finish == "stop" && stop_seq.is_some() {
            "stop_sequence"
        } else {
            map_stop_reason(&finish)
        };
        let delta = json!({
            "type": "message_delta",
            "delta": {"stop_reason": stop_reason, "stop_sequence": stop_seq},
            "usage": {
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "cache_read_input_tokens": cached.unwrap_or(0),
            },
        });
        if send_event(&tx_bytes, "message_delta", &delta).await.is_err() {
            return;
        }
        let _ = send_event(&tx_bytes, "message_stop", &json!({"type": "message_stop"})).await;
    });

    let stream = ReceiverStream::new(rx_bytes);
    with_request_id(
        Response::builder()
            .status(StatusCode::OK)
            .header(header::CONTENT_TYPE, "text/event-stream")
            .header(header::CACHE_CONTROL, "no-cache")
            .body(Body::from_stream(stream))
            .unwrap(),
    )
}

fn tokenizer_for(model_id: &str) -> Option<Arc<tokenizers::Tokenizer>> {
    use std::collections::HashMap;
    use std::sync::Mutex;
    static CACHE: std::sync::OnceLock<Mutex<HashMap<String, Option<Arc<tokenizers::Tokenizer>>>>> =
        std::sync::OnceLock::new();
    let cache = CACHE.get_or_init(Default::default);
    if let Some(hit) = cache
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .get(model_id)
    {
        return hit.clone();
    }
    let loaded = crate::oapi::model_ids::chat_model_dirs_from_env()
        .iter()
        .find(|d| crate::oapi::model_ids::model_id_for_dir(d) == model_id)
        .and_then(|d| tokenizers::Tokenizer::from_file(d.join("tokenizer.json")).ok())
        .map(Arc::new);
    cache
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .insert(model_id.to_string(), loaded.clone());
    loaded
}

pub async fn handle_count_tokens(
    State(state): State<ChatAppState>,
    body: Bytes,
) -> Response {
    let req: MessagesRequest = match serde_json::from_slice(&body) {
        Ok(r) => r,
        Err(err) => return invalid(format!("{err}")),
    };
    if req.messages.is_empty() {
        return invalid("messages: at least one message is required".to_string());
    }
    let engine = match state.registry.resolve(Some(req.model.as_str())) {
        Some(e) => e,
        None => {
            return anthropic_error(
                StatusCode::NOT_FOUND,
                akind::NOT_FOUND,
                format!("model: {}", req.model),
            )
        }
    };
    let tr = match translate(&req) {
        Ok(t) => t,
        Err(resp) => return resp,
    };
    let out = tokio::task::spawn_blocking(move || -> Result<usize, Response> {
        let mut kwargs = TemplateKwargs::new();
        if let Some(on) = tr.enable_thinking {
            kwargs.insert("enable_thinking".into(), Value::Bool(on));
        }
        let policy = resolve_tool_policy(&tr.tools, &tr.choice, None);
        let _ = resolve_guided_think_close(engine.as_ref(), policy.guided.is_some(), &mut kwargs);
        let render_tools: &[Tool] = if policy.tools_active { &tr.tools } else { &[] };
        let prompt = render_chat_checked_kwargs(
            engine.as_ref(),
            &tr.messages,
            render_tools,
            &tr.choice,
            template_required_for(engine.model_id()),
            &kwargs,
        )
        .map_err(|m| anthropic_error(StatusCode::INTERNAL_SERVER_ERROR, akind::API, m))?;
        let tok = tokenizer_for(engine.model_id()).ok_or_else(|| {
            anthropic_error(
                StatusCode::NOT_IMPLEMENTED,
                akind::API,
                format!(
                    "no tokenizer.json found for model {}; count_tokens is unavailable for this \
                     model",
                    engine.model_id()
                ),
            )
        })?;
        tok.encode(prompt, true)
            .map(|enc| enc.get_ids().len())
            .map_err(|err| {
                anthropic_error(
                    StatusCode::INTERNAL_SERVER_ERROR,
                    akind::API,
                    format!("tokenize: {err}"),
                )
            })
    })
    .await;
    match out {
        Ok(Ok(n)) => {
            with_request_id((StatusCode::OK, Json(json!({"input_tokens": n}))).into_response())
        }
        Ok(Err(resp)) => resp,
        Err(err) => anthropic_error(
            StatusCode::INTERNAL_SERVER_ERROR,
            akind::API,
            format!("count_tokens: {err}"),
        ),
    }
}

#[cfg(feature = "ts-bindings")]
mod ts_wire {
    #![allow(dead_code)]
    use ts_rs::TS;

    #[derive(TS)]
    #[ts(export)]
    struct AnthropicThinkingBlock {
        #[ts(rename = "type", type = "\"thinking\"")]
        kind: (),
        thinking: String,
        signature: String,
    }

    #[derive(TS)]
    #[ts(export)]
    struct AnthropicTextBlock {
        #[ts(rename = "type", type = "\"text\"")]
        kind: (),
        text: String,
    }

    #[derive(TS)]
    #[ts(export)]
    struct AnthropicToolUseBlock {
        #[ts(rename = "type", type = "\"tool_use\"")]
        kind: (),
        id: String,
        name: String,
        #[ts(type = "{ [key: string]: unknown }")]
        input: (),
    }

    #[derive(TS)]
    #[ts(export, untagged)]
    enum AnthropicContentBlock {
        Thinking(AnthropicThinkingBlock),
        Text(AnthropicTextBlock),
        ToolUse(AnthropicToolUseBlock),
    }

    #[derive(TS)]
    #[ts(export)]
    struct AnthropicUsage {
        input_tokens: u32,
        output_tokens: u32,
        cache_creation_input_tokens: u32,
        cache_read_input_tokens: u32,
    }

    #[derive(TS)]
    #[ts(export)]
    struct AnthropicMessagesResponse {
        id: String,
        #[ts(rename = "type", type = "\"message\"")]
        kind: (),
        #[ts(type = "\"assistant\"")]
        role: (),
        model: String,
        content: Vec<AnthropicContentBlock>,
        #[ts(
            type = "\"end_turn\" | \"max_tokens\" | \"tool_use\" | \"stop_sequence\" | null"
        )]
        stop_reason: (),
        stop_sequence: Option<String>,
        usage: AnthropicUsage,
    }

    #[derive(TS)]
    #[ts(export)]
    struct AnthropicCountTokensResponse {
        input_tokens: u32,
    }

    #[derive(TS)]
    #[ts(export)]
    struct AnthropicErrorPayload {
        #[ts(rename = "type", type = "\"error\"")]
        kind: (),
        error: AnthropicErrorDetail,
    }

    #[derive(TS)]
    #[ts(export)]
    struct AnthropicErrorDetail {
        #[ts(
            rename = "type",
            type = "\"invalid_request_error\" | \"authentication_error\" | \
                    \"not_found_error\" | \"api_error\" | \"overloaded_error\""
        )]
        kind: (),
        message: String,
    }

    #[derive(TS)]
    #[ts(export)]
    struct AnthropicMessageStartEvent {
        #[ts(rename = "type", type = "\"message_start\"")]
        kind: (),
        message: AnthropicMessagesResponse,
    }

    #[derive(TS)]
    #[ts(export)]
    struct AnthropicPingEvent {
        #[ts(rename = "type", type = "\"ping\"")]
        kind: (),
    }

    #[derive(TS)]
    #[ts(export)]
    struct AnthropicContentBlockStartEvent {
        #[ts(rename = "type", type = "\"content_block_start\"")]
        kind: (),
        index: u32,
        content_block: AnthropicContentBlock,
    }

    #[derive(TS)]
    #[ts(export)]
    struct AnthropicTextDelta {
        #[ts(rename = "type", type = "\"text_delta\"")]
        kind: (),
        text: String,
    }

    #[derive(TS)]
    #[ts(export)]
    struct AnthropicThinkingDelta {
        #[ts(rename = "type", type = "\"thinking_delta\"")]
        kind: (),
        thinking: String,
    }

    #[derive(TS)]
    #[ts(export)]
    struct AnthropicSignatureDelta {
        #[ts(rename = "type", type = "\"signature_delta\"")]
        kind: (),
        signature: String,
    }

    #[derive(TS)]
    #[ts(export)]
    struct AnthropicInputJsonDelta {
        #[ts(rename = "type", type = "\"input_json_delta\"")]
        kind: (),
        partial_json: String,
    }

    #[derive(TS)]
    #[ts(export, untagged)]
    enum AnthropicContentDelta {
        Text(AnthropicTextDelta),
        Thinking(AnthropicThinkingDelta),
        Signature(AnthropicSignatureDelta),
        InputJson(AnthropicInputJsonDelta),
    }

    #[derive(TS)]
    #[ts(export)]
    struct AnthropicContentBlockDeltaEvent {
        #[ts(rename = "type", type = "\"content_block_delta\"")]
        kind: (),
        index: u32,
        delta: AnthropicContentDelta,
    }

    #[derive(TS)]
    #[ts(export)]
    struct AnthropicContentBlockStopEvent {
        #[ts(rename = "type", type = "\"content_block_stop\"")]
        kind: (),
        index: u32,
    }

    #[derive(TS)]
    #[ts(export)]
    struct AnthropicMessageDeltaUsage {
        output_tokens: u32,
        cache_read_input_tokens: u32,
    }

    #[derive(TS)]
    #[ts(export)]
    struct AnthropicMessageDeltaEvent {
        #[ts(rename = "type", type = "\"message_delta\"")]
        kind: (),
        #[ts(
            type = "{ stop_reason: \"end_turn\" | \"max_tokens\" | \"tool_use\" | \
                    \"stop_sequence\", stop_sequence: string | null }"
        )]
        delta: (),
        usage: AnthropicMessageDeltaUsage,
    }

    #[derive(TS)]
    #[ts(export)]
    struct AnthropicMessageStopEvent {
        #[ts(rename = "type", type = "\"message_stop\"")]
        kind: (),
    }

    #[derive(TS)]
    #[ts(export, untagged)]
    enum AnthropicMessagesStreamEvent {
        MessageStart(AnthropicMessageStartEvent),
        Ping(AnthropicPingEvent),
        ContentBlockStart(AnthropicContentBlockStartEvent),
        ContentBlockDelta(AnthropicContentBlockDeltaEvent),
        ContentBlockStop(AnthropicContentBlockStopEvent),
        MessageDelta(AnthropicMessageDeltaEvent),
        MessageStop(AnthropicMessageStopEvent),
        Error(AnthropicErrorPayload),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn req(v: Value) -> MessagesRequest {
        serde_json::from_value(v).expect("request parses")
    }

    #[test]
    fn system_and_text_messages_translate() {
        let r = req(json!({
            "model": "m",
            "max_tokens": 16,
            "system": "be terse",
            "messages": [
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": [{"type": "text", "text": "hello"}]},
                {"role": "user", "content": [{"type": "text", "text": "again"}]},
            ],
        }));
        let t = translate(&r).unwrap();
        assert_eq!(t.messages.len(), 4);
        assert_eq!(t.messages[0].role, "system");
        assert_eq!(t.messages[1].role, "user");
        assert_eq!(t.messages[1].text(), "hi");
        assert_eq!(t.messages[2].role, "assistant");
        assert_eq!(t.messages[2].text(), "hello");
    }

    #[test]
    fn tool_use_and_result_round_trip() {
        let r = req(json!({
            "model": "m",
            "max_tokens": 16,
            "tools": [{"name": "get_weather", "description": "d",
                       "input_schema": {"type": "object", "properties": {"city": {"type": "string"}}}}],
            "messages": [
                {"role": "user", "content": "weather?"},
                {"role": "assistant", "content": [
                    {"type": "text", "text": "checking"},
                    {"type": "tool_use", "id": "toolu_1", "name": "get_weather", "input": {"city": "Paris"}},
                ]},
                {"role": "user", "content": [
                    {"type": "tool_result", "tool_use_id": "toolu_1", "content": "72F"},
                ]},
            ],
        }));
        let t = translate(&r).unwrap();
        assert_eq!(t.tools.len(), 1);
        assert_eq!(t.tools[0].function.name, "get_weather");
        let asst = &t.messages[1];
        let calls = asst.tool_calls.as_ref().unwrap();
        assert_eq!(calls[0].id, "toolu_1");
        assert_eq!(
            serde_json::from_str::<Value>(&calls[0].function.arguments).unwrap(),
            json!({"city": "Paris"})
        );
        let tool_msg = &t.messages[2];
        assert_eq!(tool_msg.role, "tool");
        assert_eq!(tool_msg.tool_call_id.as_deref(), Some("toolu_1"));
        assert_eq!(tool_msg.text(), "72F");
    }

    #[test]
    fn image_blocks_are_rejected() {
        let r = req(json!({
            "model": "m",
            "max_tokens": 16,
            "messages": [{"role": "user", "content": [
                {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": ""}},
            ]}],
        }));
        assert!(translate(&r).is_err());
    }

    #[test]
    fn server_tool_types_are_rejected() {
        let r = req(json!({
            "model": "m",
            "max_tokens": 16,
            "tools": [{"type": "web_search_20260209", "name": "web_search"}],
            "messages": [{"role": "user", "content": "x"}],
        }));
        assert!(translate(&r).is_err());
    }

    #[test]
    fn tool_choice_maps() {
        for (tc, want_force) in [
            (json!({"type": "auto"}), false),
            (json!({"type": "any"}), true),
            (json!({"type": "tool", "name": "f"}), true),
        ] {
            let r = req(json!({
                "model": "m",
                "max_tokens": 16,
                "tools": [{"name": "f", "input_schema": {"type": "object"}}],
                "tool_choice": tc,
                "messages": [{"role": "user", "content": "x"}],
            }));
            let t = translate(&r).unwrap();
            let eng = crate::oapi::chat_engine::EchoEngine::new("m", "x");
            let mut kwargs = TemplateKwargs::new();
            let p = resolve_tool_policy(&t.tools, &t.choice, None);
            let _ = resolve_guided_think_close(&eng, p.guided.is_some(), &mut kwargs);
            assert_eq!(p.force_name.is_some(), want_force);
        }
    }

    #[test]
    fn stop_reasons_map() {
        assert_eq!(map_stop_reason("stop"), "end_turn");
        assert_eq!(map_stop_reason("length"), "max_tokens");
        assert_eq!(map_stop_reason("tool_calls"), "tool_use");
    }

    #[test]
    fn content_blocks_assemble_in_order() {
        let calls = vec![ToolCall {
            index: None,
            id: "toolu_1".into(),
            kind: "function".into(),
            function: FunctionCall {
                name: "f".into(),
                arguments: "{\"a\":1}".into(),
            },
        }];
        let blocks = content_blocks(Some("thought"), Some("answer"), &calls);
        assert_eq!(blocks[0]["type"], "thinking");
        assert_eq!(blocks[1]["type"], "text");
        assert_eq!(blocks[2]["type"], "tool_use");
        assert_eq!(blocks[2]["input"], json!({"a": 1}));
    }

    #[test]
    fn thinking_param_maps() {
        for (v, want) in [
            (json!({"type": "adaptive"}), Some(true)),
            (json!({"type": "enabled", "budget_tokens": 1024}), Some(true)),
            (json!({"type": "disabled"}), Some(false)),
        ] {
            let r = req(json!({
                "model": "m",
                "max_tokens": 16,
                "thinking": v,
                "messages": [{"role": "user", "content": "x"}],
            }));
            assert_eq!(translate(&r).unwrap().enable_thinking, want);
        }
    }
}
