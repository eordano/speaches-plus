use std::path::PathBuf;
use std::sync::Arc;

use axum::body::{Body, Bytes};
use axum::extract::{Path, State};
use axum::http::{header, HeaderMap, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::Json;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;
use tracing::warn;

use crate::oapi::chat::{
    engine_event_error_by_surface, engine_start_error_by_surface, is_busy_shed, now_unix_secs, parse_model_tool_calls, rand_seed, render_chat_checked_kwargs,
    resolve_guided_think_close, resolve_tool_policy, split_thinking, template_required_for,
    ChatAppState, ChatEngine, ChatEvent, ChatGenerateRequest, ChatMessageIn, ClientDeadline,
    EngineBusy, FunctionCall, FunctionDef, MessageContent, TemplateKwargs, ThinkPostProcess, Tool,
    ToolCall, ToolChoice, THINK_OPEN,
};

pub const RESPONSE_STORE_DIR_ENV: &str = "NV_KV_CACHE_DIR";

const DEFAULT_MAX_OUTPUT_TOKENS: u64 = 1024;
const MAX_MAX_OUTPUT_TOKENS: u64 = 32768;

fn store_dir() -> Option<PathBuf> {
    std::env::var(RESPONSE_STORE_DIR_ENV)
        .ok()
        .filter(|d| !d.trim().is_empty())
        .map(PathBuf::from)
}

fn valid_response_id(id: &str) -> bool {
    id.strip_prefix("resp_")
        .is_some_and(|h| h.len() == 32 && h.bytes().all(|b| b.is_ascii_hexdigit()))
}

fn sidecar_path(dir: &std::path::Path, id: &str) -> PathBuf {
    dir.join(format!("{id}.json"))
}

#[derive(Serialize, Deserialize)]
struct Sidecar {
    model: String,
    messages: Vec<ChatMessageIn>,
    response: ResponseObject,
}

fn read_sidecar(id: &str) -> Option<Sidecar> {
    let dir = store_dir()?;
    let bytes = std::fs::read(sidecar_path(&dir, id)).ok()?;
    serde_json::from_slice(&bytes).ok()
}

fn write_sidecar(id: &str, sidecar: &Sidecar) {
    let Some(dir) = store_dir() else {
        return;
    };
    if std::fs::create_dir_all(&dir).is_err() {
        return;
    }
    let path = sidecar_path(&dir, id);
    match serde_json::to_vec(sidecar) {
        Ok(bytes) => {
            if let Err(e) = std::fs::write(&path, bytes) {
                warn!(path = %path.display(), error = %e, "response sidecar write failed");
            }
        }
        Err(e) => warn!(error = %e, "response sidecar serialize failed"),
    }
}

fn error_response(status: StatusCode, kind: &str, message: impl Into<String>) -> Response {
    let body = json!({
        "error": {
            "message": message.into(),
            "type": kind,
            "param": null,
            "code": null,
        }
    });
    (status, Json(body)).into_response()
}

fn invalid(message: impl Into<String>) -> Response {
    error_response(StatusCode::BAD_REQUEST, "invalid_request_error", message)
}

const RESPONSES_SHED_NOTE: &str = "shed a responses request: the surface was at capacity";

fn engine_start_error(err: &anyhow::Error) -> Response {
    engine_start_error_by_surface(
        err,
        RESPONSES_SHED_NOTE,
        |m| error_response(StatusCode::SERVICE_UNAVAILABLE, "overloaded_error", m),
        |m| error_response(StatusCode::INTERNAL_SERVER_ERROR, "api_error", m),
    )
}

fn engine_event_error(msg: String) -> Response {
    engine_event_error_by_surface(
        msg,
        RESPONSES_SHED_NOTE,
        |m| error_response(StatusCode::SERVICE_UNAVAILABLE, "overloaded_error", m),
        |m| error_response(StatusCode::INTERNAL_SERVER_ERROR, "api_error", m),
    )
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
        Err(_) => Err(error_response(
            StatusCode::SERVICE_UNAVAILABLE,
            "overloaded_error",
            format!(
                "generation did not start within the {} ms client budget ({})",
                client.budget_ms(),
                client.source
            ),
        )),
    }
}

#[derive(Clone, Debug, Deserialize)]
#[cfg_attr(
    feature = "ts-bindings",
    derive(ts_rs::TS),
    ts(export, optional_fields = nullable)
)]
pub struct ResponsesRequest {
    pub model: Option<String>,
    pub input: Option<ResponsesInput>,
    pub instructions: Option<String>,
    pub store: Option<bool>,
    pub previous_response_id: Option<String>,
    pub stream: Option<bool>,
    pub max_output_tokens: Option<u64>,
    pub temperature: Option<f32>,
    pub top_p: Option<f32>,
    #[serde(default)]
    pub metadata: Option<std::collections::HashMap<String, Value>>,
    #[serde(default)]
    pub tools: Vec<Value>,
    #[serde(default)]
    pub tool_choice: Option<Value>,
    #[serde(default)]
    pub parallel_tool_calls: Option<bool>,
}

#[derive(Clone, Debug, Deserialize)]
#[cfg_attr(feature = "ts-bindings", derive(ts_rs::TS), ts(export))]
#[serde(untagged)]
pub enum ResponsesInput {
    Text(String),
    Items(Vec<ResponseInputItem>),
}

#[derive(Clone, Debug, Deserialize)]
#[cfg_attr(
    feature = "ts-bindings",
    derive(ts_rs::TS),
    ts(export, optional_fields = nullable)
)]
pub struct ResponseInputItem {
    #[serde(default, rename = "type")]
    pub kind: Option<String>,
    #[serde(default)]
    pub role: Option<String>,
    #[serde(default)]
    pub content: Option<ResponseInputContent>,
    #[serde(default)]
    pub id: Option<String>,
    #[serde(default)]
    pub call_id: Option<String>,
    #[serde(default)]
    pub name: Option<String>,
    #[serde(default)]
    pub arguments: Option<String>,
    #[serde(default)]
    pub output: Option<Value>,
}

#[derive(Clone, Debug, Deserialize)]
#[cfg_attr(feature = "ts-bindings", derive(ts_rs::TS), ts(export))]
#[serde(untagged)]
pub enum ResponseInputContent {
    Text(String),
    Parts(Vec<ResponseInputPart>),
}

#[derive(Clone, Debug, Deserialize)]
#[cfg_attr(
    feature = "ts-bindings",
    derive(ts_rs::TS),
    ts(export, optional_fields = nullable)
)]
pub struct ResponseInputPart {
    #[serde(default, rename = "type")]
    pub kind: Option<String>,
    #[serde(default)]
    pub text: Option<String>,
}

fn part_text(part: &ResponseInputPart) -> Result<String, Response> {
    match part.kind.as_deref() {
        Some("input_text") | Some("output_text") | Some("text") | None => {
            Ok(part.text.clone().unwrap_or_default())
        }
        Some(other) => Err(invalid(format!(
            "input content part type {other:?} is not supported; only text parts are"
        ))),
    }
}

fn normalize_input(input: &ResponsesInput) -> Result<Vec<ChatMessageIn>, Response> {
    let text_message = |role: &str, text: String| ChatMessageIn {
        role: role.to_string(),
        content: Some(MessageContent::Text(text)),
        ..ChatMessageIn::default()
    };
    match input {
        ResponsesInput::Text(s) => Ok(vec![text_message("user", s.clone())]),
        ResponsesInput::Items(items) => {
            let mut out = Vec::new();
            for item in items {
                match item.kind.as_deref() {
                    Some("reasoning") => continue,
                    Some("function_call") => {
                        let call_id = item
                            .call_id
                            .clone()
                            .or_else(|| item.id.clone())
                            .ok_or_else(|| invalid("function_call: call_id is required"))?;
                        let name = item
                            .name
                            .clone()
                            .ok_or_else(|| invalid("function_call: name is required"))?;
                        out.push(ChatMessageIn {
                            role: "assistant".to_string(),
                            tool_calls: Some(vec![ToolCall {
                                index: None,
                                id: call_id,
                                kind: "function".to_string(),
                                function: FunctionCall {
                                    name,
                                    arguments: item
                                        .arguments
                                        .clone()
                                        .unwrap_or_else(|| "{}".to_string()),
                                },
                            }]),
                            ..ChatMessageIn::default()
                        });
                        continue;
                    }
                    Some("function_call_output") => {
                        let call_id = item
                            .call_id
                            .clone()
                            .ok_or_else(|| invalid("function_call_output: call_id is required"))?;
                        let output = item
                            .output
                            .as_ref()
                            .ok_or_else(|| invalid("function_call_output: output is required"))?;
                        let text = output
                            .as_str()
                            .map(str::to_string)
                            .unwrap_or_else(|| output.to_string());
                        out.push(ChatMessageIn {
                            role: "tool".to_string(),
                            content: Some(MessageContent::Text(text)),
                            tool_call_id: Some(call_id),
                            ..ChatMessageIn::default()
                        });
                        continue;
                    }
                    Some("message") | None => {}
                    Some(other) => {
                        return Err(invalid(format!(
                            "input item type {other:?} is not supported"
                        )))
                    }
                }
                let role = item
                    .role
                    .as_deref()
                    .ok_or_else(|| invalid("input message: role is required"))?;
                let text = match &item.content {
                    Some(ResponseInputContent::Text(s)) => s.clone(),
                    Some(ResponseInputContent::Parts(parts)) => {
                        let mut t = String::new();
                        for p in parts {
                            t.push_str(&part_text(p)?);
                        }
                        t
                    }
                    None => return Err(invalid("input message: content is required")),
                };
                out.push(text_message(role, text));
            }
            Ok(out)
        }
    }
}

fn response_tools(raw: &[Value]) -> Result<Vec<Tool>, Response> {
    let mut tools = Vec::new();
    for (index, value) in raw.iter().enumerate() {
        if value.get("type").and_then(Value::as_str) != Some("function") {
            continue;
        }
        let function = value.get("function").unwrap_or(value);
        let name = function
            .get("name")
            .and_then(Value::as_str)
            .filter(|name| !name.is_empty())
            .ok_or_else(|| invalid(format!("tools[{index}]: function name is required")))?;
        tools.push(Tool {
            kind: "function".to_string(),
            function: FunctionDef {
                name: name.to_string(),
                description: function
                    .get("description")
                    .and_then(Value::as_str)
                    .map(str::to_string),
                parameters: function.get("parameters").cloned(),
            },
        });
    }
    Ok(tools)
}

fn response_tool_choice(raw: Option<&Value>) -> Result<ToolChoice, Response> {
    match raw {
        None => Ok(ToolChoice::Auto),
        Some(Value::String(choice)) => match choice.as_str() {
            "none" => Ok(ToolChoice::None),
            "auto" => Ok(ToolChoice::Auto),
            "required" => Ok(ToolChoice::Required),
            _ => Err(invalid(format!("invalid tool_choice {choice:?}"))),
        },
        Some(Value::Object(choice))
            if choice.get("type").and_then(Value::as_str) == Some("function") =>
        {
            let name = choice
                .get("name")
                .and_then(Value::as_str)
                .filter(|name| !name.is_empty())
                .ok_or_else(|| invalid("tool_choice: function name is required"))?;
            Ok(ToolChoice::Function(name.to_string()))
        }
        Some(other) => Err(invalid(format!("invalid tool_choice {other}"))),
    }
}

struct Outcome {
    input_tokens: u32,
    output_tokens: u32,
    cached: u32,
    text: String,
    tool_calls: Vec<ToolCall>,
    finish: String,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[cfg_attr(feature = "ts-bindings", derive(ts_rs::TS), ts(export))]
#[serde(rename_all = "snake_case")]
pub enum ResponseObjectStatus {
    InProgress,
    Completed,
    Incomplete,
    Failed,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[cfg_attr(feature = "ts-bindings", derive(ts_rs::TS), ts(export))]
pub struct ResponseErrorInfo {
    pub code: String,
    pub message: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[cfg_attr(feature = "ts-bindings", derive(ts_rs::TS), ts(export))]
pub struct ResponseIncompleteDetails {
    #[cfg_attr(feature = "ts-bindings", ts(type = "\"max_output_tokens\""))]
    pub reason: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[cfg_attr(feature = "ts-bindings", derive(ts_rs::TS), ts(export))]
pub struct ResponseInputTokensDetails {
    pub cached_tokens: u32,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[cfg_attr(feature = "ts-bindings", derive(ts_rs::TS), ts(export))]
pub struct ResponseOutputTokensDetails {
    pub reasoning_tokens: u32,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[cfg_attr(feature = "ts-bindings", derive(ts_rs::TS), ts(export))]
pub struct ResponseUsage {
    pub input_tokens: u32,
    pub input_tokens_details: ResponseInputTokensDetails,
    pub output_tokens: u32,
    pub output_tokens_details: ResponseOutputTokensDetails,
    pub total_tokens: u32,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[cfg_attr(feature = "ts-bindings", derive(ts_rs::TS), ts(export))]
pub struct ResponseOutputContent {
    #[serde(rename = "type")]
    #[cfg_attr(feature = "ts-bindings", ts(type = "\"output_text\""))]
    pub kind: String,
    pub annotations: Vec<Value>,
    pub text: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[cfg_attr(feature = "ts-bindings", derive(ts_rs::TS), ts(export))]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ResponseOutputItem {
    Message {
        id: String,
        status: String,
        role: String,
        content: Vec<ResponseOutputContent>,
    },
    FunctionCall {
        id: String,
        status: String,
        call_id: String,
        name: String,
        arguments: String,
    },
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[cfg_attr(feature = "ts-bindings", derive(ts_rs::TS), ts(export))]
pub struct ResponseObject {
    pub id: String,
    #[cfg_attr(feature = "ts-bindings", ts(type = "\"response\""))]
    pub object: String,
    pub created_at: i64,
    pub status: ResponseObjectStatus,
    pub error: Option<ResponseErrorInfo>,
    pub incomplete_details: Option<ResponseIncompleteDetails>,
    pub instructions: Option<String>,
    pub max_output_tokens: Option<u64>,
    pub model: String,
    pub output: Vec<ResponseOutputItem>,
    pub parallel_tool_calls: bool,
    pub previous_response_id: Option<String>,
    pub store: bool,
    pub temperature: Option<f32>,
    pub tool_choice: Value,
    pub tools: Vec<Value>,
    pub top_p: Option<f32>,
    #[cfg_attr(feature = "ts-bindings", ts(type = "\"disabled\""))]
    pub truncation: String,
    pub usage: Option<ResponseUsage>,
    pub metadata: std::collections::HashMap<String, Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[cfg_attr(feature = "ts-bindings", ts(optional))]
    pub output_text: Option<String>,
}

fn output_message(id: &str, text: &str) -> ResponseOutputItem {
    ResponseOutputItem::Message {
        id: format!("msg_{}", &id[5..]),
        status: "completed".to_string(),
        role: "assistant".to_string(),
        content: vec![ResponseOutputContent {
            kind: "output_text".to_string(),
            annotations: Vec::new(),
            text: text.to_string(),
        }],
    }
}

fn output_items(id: &str, out: &Outcome) -> Vec<ResponseOutputItem> {
    let mut items = Vec::new();
    if !out.text.is_empty() {
        items.push(output_message(id, &out.text));
    }
    items.extend(out.tool_calls.iter().enumerate().map(|(index, call)| {
        ResponseOutputItem::FunctionCall {
            id: format!("fc_{}_{}", &id[5..], index),
            status: "completed".to_string(),
            call_id: call.id.clone(),
            name: call.function.name.clone(),
            arguments: call.function.arguments.clone(),
        }
    }));
    items
}

fn response_object(
    id: &str,
    model: &str,
    created_at: i64,
    req: &ResponsesRequest,
    previous_response_id: Option<&str>,
    stored: bool,
    out: Option<&Outcome>,
) -> ResponseObject {
    let (status, output, usage, output_text) = match out {
        None => (ResponseObjectStatus::InProgress, Vec::new(), None, None),
        Some(o) => {
            let status = if o.finish == "length" {
                ResponseObjectStatus::Incomplete
            } else {
                ResponseObjectStatus::Completed
            };
            let usage = ResponseUsage {
                input_tokens: o.input_tokens,
                input_tokens_details: ResponseInputTokensDetails {
                    cached_tokens: o.cached,
                },
                output_tokens: o.output_tokens,
                output_tokens_details: ResponseOutputTokensDetails {
                    reasoning_tokens: 0,
                },
                total_tokens: o.input_tokens + o.output_tokens,
            };
            (
                status,
                output_items(id, o),
                Some(usage),
                (!o.text.is_empty()).then(|| o.text.clone()),
            )
        }
    };
    ResponseObject {
        id: id.to_string(),
        object: "response".to_string(),
        created_at,
        status,
        error: None,
        incomplete_details: out.filter(|o| o.finish == "length").map(|_| {
            ResponseIncompleteDetails {
                reason: "max_output_tokens".to_string(),
            }
        }),
        instructions: req.instructions.clone(),
        max_output_tokens: req.max_output_tokens,
        model: model.to_string(),
        output,
        parallel_tool_calls: req.parallel_tool_calls.unwrap_or(true),
        previous_response_id: previous_response_id.map(str::to_string),
        store: stored,
        temperature: req.temperature,
        tool_choice: req
            .tool_choice
            .clone()
            .unwrap_or_else(|| Value::String("auto".to_string())),
        tools: req.tools.clone(),
        top_p: req.top_p,
        truncation: "disabled".to_string(),
        usage,
        metadata: req.metadata.clone().unwrap_or_default(),
        output_text,
    }
}

fn visible_text(raw: &str, think: &ThinkPostProcess) -> String {
    if !think.active {
        return raw.to_string();
    }
    match split_thinking(raw, think.opened) {
        Some((_, content)) => content,
        None => raw.to_string(),
    }
}

fn visible_stream(raw: &str, think: &ThinkPostProcess) -> String {
    if !think.active {
        return raw.to_string();
    }
    if !(think.opened || raw.trim_start().starts_with(THINK_OPEN)) {
        return raw.to_string();
    }
    match raw.find(THINK_CLOSE) {
        Some(i) => raw[i + THINK_CLOSE.len()..]
            .trim_start_matches('\n')
            .to_string(),
        None => String::new(),
    }
}

const THINK_CLOSE: &str = "</think>";

pub async fn handle_responses(
    State(state): State<ChatAppState>,
    headers: HeaderMap,
    body: Bytes,
) -> Response {
    let req: ResponsesRequest = match serde_json::from_slice(&body) {
        Ok(r) => r,
        Err(err) => return invalid(format!("{err}")),
    };
    let Some(input) = req.input.as_ref() else {
        return invalid("input: field required");
    };
    let tools = match response_tools(&req.tools) {
        Ok(tools) => tools,
        Err(resp) => return resp,
    };
    let choice = match response_tool_choice(req.tool_choice.as_ref()) {
        Ok(choice) => choice,
        Err(resp) => return resp,
    };
    if let ToolChoice::Function(name) = &choice {
        if !tools.iter().any(|tool| &tool.function.name == name) {
            return invalid(format!("tool_choice names unknown function {name:?}"));
        }
    }

    let mut prior: Vec<ChatMessageIn> = Vec::new();
    let mut prior_id: Option<String> = None;
    if let Some(prev) = req.previous_response_id.as_deref() {
        if !valid_response_id(prev) {
            return invalid(format!(
                "previous_response_id: {prev:?} is not a response id"
            ));
        }
        let Some(sidecar) = read_sidecar(prev) else {
            return error_response(
                StatusCode::NOT_FOUND,
                "invalid_request_error",
                format!("previous response {prev} not found"),
            );
        };
        if let Some(m) = req.model.as_deref() {
            if m != sidecar.model {
                return invalid(format!(
                    "previous response {prev} belongs to model {}, not {m}",
                    sidecar.model
                ));
            }
        }
        prior = sidecar.messages;
        prior_id = Some(prev.to_string());
    }

    let engine = match state.registry.resolve(req.model.as_deref()) {
        Some(e) => e,
        None => {
            return error_response(
                StatusCode::NOT_FOUND,
                "invalid_request_error",
                format!("model: {}", req.model.as_deref().unwrap_or("<default>")),
            )
        }
    };

    let new_turns = match normalize_input(input) {
        Ok(m) => m,
        Err(resp) => return resp,
    };
    if new_turns.is_empty() {
        return invalid("input: at least one message is required");
    }

    let mut messages: Vec<ChatMessageIn> = Vec::new();
    if let Some(instructions) = req.instructions.as_deref() {
        messages.push(ChatMessageIn {
            role: "system".to_string(),
            content: Some(MessageContent::Text(instructions.to_string())),
            ..ChatMessageIn::default()
        });
    }
    messages.extend(prior);
    messages.extend(new_turns);

    let policy = resolve_tool_policy(&tools, &choice, None);
    let mut template_kwargs = TemplateKwargs::new();
    let guided_think_close = resolve_guided_think_close(
        engine.as_ref(),
        policy.guided.is_some(),
        &mut template_kwargs,
    );
    let render_tools = if policy.tools_active { &tools[..] } else { &[] };
    let prompt = match render_chat_checked_kwargs(
        engine.as_ref(),
        &messages,
        render_tools,
        &choice,
        template_required_for(engine.model_id()),
        &template_kwargs,
    ) {
        Ok(p) => p,
        Err(message) => {
            tracing::error!(model = %engine.model_id(), reason = %message, "responses request refused");
            return error_response(StatusCode::INTERNAL_SERVER_ERROR, "api_error", message);
        }
    };
    let think = ThinkPostProcess {
        active: engine.thinking_split_supported(),
        opened: prompt.trim_end().ends_with(THINK_OPEN),
    };

    let id = format!("resp_{}", uuid::Uuid::new_v4().simple());
    let stored = req.store.unwrap_or(true) && store_dir().is_some();
    let gen = ChatGenerateRequest {
        prompt,
        max_new_tokens: req
            .max_output_tokens
            .unwrap_or(DEFAULT_MAX_OUTPUT_TOKENS)
            .clamp(1, MAX_MAX_OUTPUT_TOKENS) as usize,
        stop: Vec::new(),
        seed: Some(rand_seed()),
        temperature: req.temperature,
        top_p: req.top_p,
        top_k: None,
        min_p: None,
        presence_penalty: None,
        frequency_penalty: None,
        repetition_penalty: None,
        guided: policy.guided,
        guided_think_close,
        logit_bias: Vec::new(),
        logprobs: false,
        top_logprobs: 0,
        kv_resume: prior_id.clone(),
        kv_store: stored.then(|| id.clone()),
        mm: None,
    };

    let client = ClientDeadline::from_request(None, &headers);
    let model_id = engine.model_id().to_string();
    let ctx = RequestCtx {
        id,
        model: model_id,
        created_at: now_unix_secs(),
        req,
        prior_id,
        stored,
        messages,
        think,
        tools_active: policy.tools_active,
        force_name: policy.force_name,
    };
    if ctx.req.stream.unwrap_or(false) {
        run_streaming(engine, gen, ctx, client).await
    } else {
        run_non_streaming(engine, gen, ctx, client).await
    }
}

struct RequestCtx {
    id: String,
    model: String,
    created_at: i64,
    req: ResponsesRequest,
    prior_id: Option<String>,
    stored: bool,
    messages: Vec<ChatMessageIn>,
    think: ThinkPostProcess,
    tools_active: bool,
    force_name: Option<String>,
}

fn finish_and_store(ctx: &RequestCtx, out: &Outcome) -> ResponseObject {
    let response = response_object(
        &ctx.id,
        &ctx.model,
        ctx.created_at,
        &ctx.req,
        ctx.prior_id.as_deref(),
        ctx.stored,
        Some(out),
    );
    if ctx.stored {
        let mut messages = ctx.messages.clone();
        messages.push(ChatMessageIn {
            role: "assistant".to_string(),
            content: (!out.text.is_empty()).then(|| MessageContent::Text(out.text.clone())),
            tool_calls: (!out.tool_calls.is_empty()).then(|| out.tool_calls.clone()),
            ..ChatMessageIn::default()
        });
        let stored_messages: Vec<ChatMessageIn> = messages
            .into_iter()
            .filter(|m| m.role != "system")
            .collect();
        write_sidecar(
            &ctx.id,
            &Sidecar {
                model: ctx.model.clone(),
                messages: stored_messages,
                response: response.clone(),
            },
        );
    }
    response
}

async fn run_non_streaming(
    engine: Arc<dyn ChatEngine>,
    gen: ChatGenerateRequest,
    ctx: RequestCtx,
    client: Option<ClientDeadline>,
) -> Response {
    let (tx, mut rx) = mpsc::channel::<ChatEvent>(64);
    if let Err(err) = engine.generate(gen, tx).await {
        warn!(error = %err, "responses engine.generate failed to start");
        return engine_start_error(&err);
    }
    let mut pre = match first_event(&mut rx, client).await {
        Ok(ev) => ev,
        Err(resp) => return resp,
    };

    let mut out = Outcome {
        input_tokens: 0,
        output_tokens: 0,
        cached: 0,
        text: String::new(),
        tool_calls: Vec::new(),
        finish: String::from("stop"),
    };
    loop {
        let Some(ev) = (match pre.take() {
            Some(ev) => Some(ev),
            None => rx.recv().await,
        }) else {
            break;
        };
        match ev {
            ChatEvent::Started { prompt_tokens } => out.input_tokens = prompt_tokens,
            ChatEvent::PromptCached { cached_tokens } => out.cached = cached_tokens,
            ChatEvent::StoppedBy { .. } => {}
            ChatEvent::TextDelta(s) => out.text.push_str(&s),
            ChatEvent::ReasoningDelta(_) => {}
            ChatEvent::Logprob(_) => {}
            ChatEvent::Done {
                finish_reason,
                completion_tokens,
            } => {
                out.finish = finish_reason;
                out.output_tokens = completion_tokens;
            }
            ChatEvent::Error(msg) => return engine_event_error(msg),
        }
    }
    out.text = visible_text(&out.text, &ctx.think);
    if ctx.tools_active {
        let parsed = parse_model_tool_calls(&out.text, ctx.force_name.as_deref());
        out.text = parsed.content.unwrap_or_default();
        out.tool_calls = parsed.tool_calls;
        if !out.tool_calls.is_empty() {
            out.finish = "tool_calls".to_string();
        }
    }
    let response = finish_and_store(&ctx, &out);
    (StatusCode::OK, Json(response)).into_response()
}

async fn send_event(
    tx: &mpsc::Sender<Result<Bytes, std::io::Error>>,
    seq: &mut u64,
    event: &str,
    mut data: Value,
) -> Result<(), ()> {
    data["sequence_number"] = json!(*seq);
    *seq += 1;
    let frame = format!("event: {event}\ndata: {data}\n\n");
    tx.send(Ok(Bytes::from(frame.into_bytes())))
        .await
        .map_err(|_| ())
}

async fn emit_message_events(
    tx: &mpsc::Sender<Result<Bytes, std::io::Error>>,
    seq: &mut u64,
    response_id: &str,
    item_id: &str,
    output_index: usize,
    text: &str,
) -> Result<(), ()> {
    send_event(
        tx,
        seq,
        "response.output_item.added",
        json!({
            "type": "response.output_item.added", "response_id": response_id,
            "output_index": output_index,
            "item": {"id": item_id, "type": "message", "status": "in_progress",
                     "role": "assistant", "content": []},
        }),
    )
    .await?;
    send_event(
        tx,
        seq,
        "response.content_part.added",
        json!({
            "type": "response.content_part.added", "response_id": response_id,
            "item_id": item_id, "output_index": output_index, "content_index": 0,
            "part": {"type": "output_text", "annotations": [], "text": ""},
        }),
    )
    .await?;
    send_event(
        tx,
        seq,
        "response.output_text.delta",
        json!({
            "type": "response.output_text.delta", "response_id": response_id,
            "item_id": item_id, "output_index": output_index, "content_index": 0,
            "delta": text,
        }),
    )
    .await?;
    send_event(
        tx,
        seq,
        "response.output_text.done",
        json!({
            "type": "response.output_text.done", "response_id": response_id,
            "item_id": item_id, "output_index": output_index, "content_index": 0,
            "text": text,
        }),
    )
    .await?;
    send_event(
        tx,
        seq,
        "response.content_part.done",
        json!({
            "type": "response.content_part.done", "response_id": response_id,
            "item_id": item_id, "output_index": output_index, "content_index": 0,
            "part": {"type": "output_text", "annotations": [], "text": text},
        }),
    )
    .await?;
    send_event(
        tx,
        seq,
        "response.output_item.done",
        json!({
            "type": "response.output_item.done", "response_id": response_id,
            "output_index": output_index,
            "item": {"id": item_id, "type": "message", "status": "completed",
                     "role": "assistant",
                     "content": [{"type": "output_text", "annotations": [], "text": text}]},
        }),
    )
    .await
}

async fn emit_function_call_events(
    tx: &mpsc::Sender<Result<Bytes, std::io::Error>>,
    seq: &mut u64,
    response_id: &str,
    output_index: usize,
    call: &ToolCall,
) -> Result<(), ()> {
    let item_id = format!("fc_{}_{}", &response_id[5..], output_index);
    send_event(
        tx,
        seq,
        "response.output_item.added",
        json!({
            "type": "response.output_item.added", "response_id": response_id,
            "output_index": output_index,
            "item": {"id": item_id, "type": "function_call", "status": "in_progress",
                     "call_id": call.id, "name": call.function.name, "arguments": ""},
        }),
    )
    .await?;
    if !call.function.arguments.is_empty() {
        send_event(
            tx,
            seq,
            "response.function_call_arguments.delta",
            json!({
                "type": "response.function_call_arguments.delta", "response_id": response_id,
                "item_id": item_id, "output_index": output_index,
                "delta": call.function.arguments,
            }),
        )
        .await?;
    }
    send_event(
        tx,
        seq,
        "response.function_call_arguments.done",
        json!({
            "type": "response.function_call_arguments.done", "response_id": response_id,
            "item_id": item_id, "output_index": output_index,
            "arguments": call.function.arguments,
        }),
    )
    .await?;
    send_event(
        tx,
        seq,
        "response.output_item.done",
        json!({
            "type": "response.output_item.done", "response_id": response_id,
            "output_index": output_index,
            "item": {"id": item_id, "type": "function_call", "status": "completed",
                     "call_id": call.id, "name": call.function.name,
                     "arguments": call.function.arguments},
        }),
    )
    .await
}

async fn run_streaming(
    engine: Arc<dyn ChatEngine>,
    gen: ChatGenerateRequest,
    ctx: RequestCtx,
    client: Option<ClientDeadline>,
) -> Response {
    let (tx_bytes, rx_bytes) = mpsc::channel::<Result<Bytes, std::io::Error>>(64);
    let (tx_ev, mut rx_ev) = mpsc::channel::<ChatEvent>(64);
    if let Err(err) = engine.generate(gen, tx_ev).await {
        warn!(error = %err, "responses engine.generate failed to start");
        return engine_start_error(&err);
    }
    let mut pre = match first_event(&mut rx_ev, client).await {
        Ok(ev) => ev,
        Err(resp) => return resp,
    };

    tokio::spawn(async move {
        let mut seq: u64 = 0;
        let msg_id = format!("msg_{}", &ctx.id[5..]);
        let shell = response_object(
            &ctx.id,
            &ctx.model,
            ctx.created_at,
            &ctx.req,
            ctx.prior_id.as_deref(),
            ctx.stored,
            None,
        );
        if send_event(
            &tx_bytes,
            &mut seq,
            "response.created",
            json!({"type": "response.created", "response": shell.clone()}),
        )
        .await
        .is_err()
        {
            return;
        }
        let _ = send_event(
            &tx_bytes,
            &mut seq,
            "response.in_progress",
            json!({"type": "response.in_progress", "response": shell}),
        )
        .await;
        if !ctx.tools_active {
            let _ = send_event(
                &tx_bytes,
                &mut seq,
                "response.output_item.added",
                json!({
                    "type": "response.output_item.added", "response_id": ctx.id,
                    "output_index": 0,
                    "item": {"id": msg_id, "type": "message", "status": "in_progress",
                             "role": "assistant", "content": []},
                }),
            )
            .await;
            let _ = send_event(
                &tx_bytes,
                &mut seq,
                "response.content_part.added",
                json!({
                    "type": "response.content_part.added", "response_id": ctx.id,
                    "item_id": msg_id, "output_index": 0, "content_index": 0,
                    "part": {"type": "output_text", "annotations": [], "text": ""},
                }),
            )
            .await;
        }

        let mut out = Outcome {
            input_tokens: 0,
            output_tokens: 0,
            cached: 0,
            text: String::new(),
            tool_calls: Vec::new(),
            finish: String::from("stop"),
        };
        let mut raw = String::new();
        let mut emitted = 0usize;
        loop {
            let Some(ev) = (match pre.take() {
                Some(ev) => Some(ev),
                None => rx_ev.recv().await,
            }) else {
                break;
            };
            match ev {
                ChatEvent::Started { prompt_tokens } => out.input_tokens = prompt_tokens,
                ChatEvent::PromptCached { cached_tokens } => out.cached = cached_tokens,
                ChatEvent::StoppedBy { .. } => {}
                ChatEvent::ReasoningDelta(_) => {}
                ChatEvent::Logprob(_) => {}
                ChatEvent::TextDelta(s) => {
                    raw.push_str(&s);
                    if ctx.tools_active {
                        continue;
                    }
                    let visible = visible_stream(&raw, &ctx.think);
                    if visible.len() > emitted {
                        let delta = visible[emitted..].to_string();
                        emitted = visible.len();
                        if send_event(
                            &tx_bytes,
                            &mut seq,
                            "response.output_text.delta",
                            json!({
                                "type": "response.output_text.delta",
                                "item_id": msg_id, "output_index": 0, "content_index": 0,
                                "delta": delta,
                            }),
                        )
                        .await
                        .is_err()
                        {
                            return;
                        }
                    }
                }
                ChatEvent::Done {
                    finish_reason,
                    completion_tokens,
                } => {
                    out.finish = finish_reason;
                    out.output_tokens = completion_tokens;
                }
                ChatEvent::Error(msg) => {
                    let failed = json!({
                        "type": "response.failed",
                        "response": {
                            "id": ctx.id, "object": "response", "status": "failed",
                            "error": {"code": "server_error", "message": msg},
                        },
                    });
                    let _ = send_event(&tx_bytes, &mut seq, "response.failed", failed).await;
                    return;
                }
            }
        }
        out.text = visible_stream(&raw, &ctx.think);
        if ctx.tools_active {
            let parsed = parse_model_tool_calls(&out.text, ctx.force_name.as_deref());
            out.text = parsed.content.unwrap_or_default();
            out.tool_calls = parsed.tool_calls;
            if !out.tool_calls.is_empty() {
                out.finish = "tool_calls".to_string();
            }
            let mut output_index = 0;
            if !out.text.is_empty() {
                if emit_message_events(
                    &tx_bytes,
                    &mut seq,
                    &ctx.id,
                    &msg_id,
                    output_index,
                    &out.text,
                )
                .await
                .is_err()
                {
                    return;
                }
                output_index += 1;
            }
            for call in &out.tool_calls {
                if emit_function_call_events(&tx_bytes, &mut seq, &ctx.id, output_index, call)
                    .await
                    .is_err()
                {
                    return;
                }
                output_index += 1;
            }
        } else {
            let _ = send_event(
                &tx_bytes,
                &mut seq,
                "response.output_text.done",
                json!({
                    "type": "response.output_text.done",
                    "item_id": msg_id, "output_index": 0, "content_index": 0,
                    "text": out.text,
                }),
            )
            .await;
            let _ = send_event(
                &tx_bytes,
                &mut seq,
                "response.content_part.done",
                json!({
                    "type": "response.content_part.done",
                    "item_id": msg_id, "output_index": 0, "content_index": 0,
                    "part": {"type": "output_text", "annotations": [], "text": out.text},
                }),
            )
            .await;
            let _ = send_event(
            &tx_bytes,
            &mut seq,
            "response.output_item.done",
            json!({
                "type": "response.output_item.done",
                "output_index": 0,
                "item": {"id": msg_id, "type": "message", "status": "completed",
                         "role": "assistant",
                         "content": [{"type": "output_text", "annotations": [], "text": out.text}]},
            }),
        )
        .await;
        }
        let response = finish_and_store(&ctx, &out);
        let _ = send_event(
            &tx_bytes,
            &mut seq,
            "response.completed",
            json!({"type": "response.completed", "response": response}),
        )
        .await;
    });

    let stream = ReceiverStream::new(rx_bytes);
    Response::builder()
        .status(StatusCode::OK)
        .header(header::CONTENT_TYPE, "text/event-stream")
        .header(header::CACHE_CONTROL, "no-cache")
        .body(Body::from_stream(stream))
        .unwrap()
}

pub async fn handle_get_response(Path(id): Path<String>) -> Response {
    if !valid_response_id(&id) {
        return invalid(format!("{id:?} is not a response id"));
    }
    match read_sidecar(&id) {
        Some(sidecar) => (StatusCode::OK, Json(sidecar.response)).into_response(),
        None => error_response(
            StatusCode::NOT_FOUND,
            "invalid_request_error",
            format!("response {id} not found"),
        ),
    }
}

pub async fn handle_delete_response(Path(id): Path<String>) -> Response {
    if !valid_response_id(&id) {
        return invalid(format!("{id:?} is not a response id"));
    }
    let Some(dir) = store_dir() else {
        return error_response(
            StatusCode::NOT_FOUND,
            "invalid_request_error",
            format!("response {id} not found"),
        );
    };
    let sidecar = sidecar_path(&dir, &id);
    if !sidecar.exists() {
        return error_response(
            StatusCode::NOT_FOUND,
            "invalid_request_error",
            format!("response {id} not found"),
        );
    }
    let _ = std::fs::remove_file(&sidecar);
    let _ = std::fs::remove_file(dir.join(format!("{id}.nvkv")));
    (
        StatusCode::OK,
        Json(ResponseDeleteAck {
            id,
            object: "response".to_string(),
            deleted: true,
        }),
    )
        .into_response()
}

#[derive(Clone, Debug, Serialize)]
#[cfg_attr(feature = "ts-bindings", derive(ts_rs::TS), ts(export))]
pub struct ResponseDeleteAck {
    pub id: String,
    #[cfg_attr(feature = "ts-bindings", ts(type = "\"response\""))]
    pub object: String,
    pub deleted: bool,
}

#[cfg(feature = "ts-bindings")]
mod ts_wire {
    #![allow(dead_code)]
    use super::{ResponseErrorInfo, ResponseObject, ResponseOutputContent, ResponseOutputItem};
    use ts_rs::TS;

    #[derive(TS)]
    #[ts(export)]
    struct ResponseFailedPayload {
        id: String,
        #[ts(type = "\"response\"")]
        object: (),
        #[ts(type = "\"failed\"")]
        status: (),
        error: ResponseErrorInfo,
    }

    #[derive(TS)]
    #[ts(export)]
    struct ResponseLifecycleEvent {
        #[ts(
            rename = "type",
            type = "\"response.created\" | \"response.in_progress\" | \"response.completed\""
        )]
        kind: (),
        response: ResponseObject,
        sequence_number: u64,
    }

    #[derive(TS)]
    #[ts(export)]
    struct ResponseFailedEvent {
        #[ts(rename = "type", type = "\"response.failed\"")]
        kind: (),
        response: ResponseFailedPayload,
        sequence_number: u64,
    }

    #[derive(TS)]
    #[ts(export)]
    struct ResponseOutputItemEvent {
        #[ts(
            rename = "type",
            type = "\"response.output_item.added\" | \"response.output_item.done\""
        )]
        kind: (),
        output_index: u32,
        item: ResponseOutputItem,
        sequence_number: u64,
    }

    #[derive(TS)]
    #[ts(export)]
    struct ResponseContentPartEvent {
        #[ts(
            rename = "type",
            type = "\"response.content_part.added\" | \"response.content_part.done\""
        )]
        kind: (),
        item_id: String,
        output_index: u32,
        content_index: u32,
        part: ResponseOutputContent,
        sequence_number: u64,
    }

    #[derive(TS)]
    #[ts(export)]
    struct ResponseOutputTextDeltaEvent {
        #[ts(rename = "type", type = "\"response.output_text.delta\"")]
        kind: (),
        item_id: String,
        output_index: u32,
        content_index: u32,
        delta: String,
        sequence_number: u64,
    }

    #[derive(TS)]
    #[ts(export)]
    struct ResponseOutputTextDoneEvent {
        #[ts(rename = "type", type = "\"response.output_text.done\"")]
        kind: (),
        item_id: String,
        output_index: u32,
        content_index: u32,
        text: String,
        sequence_number: u64,
    }

    #[derive(TS)]
    #[ts(export, untagged)]
    enum ResponsesStreamEvent {
        Lifecycle(ResponseLifecycleEvent),
        Failed(ResponseFailedEvent),
        OutputItem(ResponseOutputItemEvent),
        ContentPart(ResponseContentPartEvent),
        OutputTextDelta(ResponseOutputTextDeltaEvent),
        OutputTextDone(ResponseOutputTextDoneEvent),
    }
}
