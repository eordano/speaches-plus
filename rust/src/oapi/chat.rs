use std::sync::Arc;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use axum::body::Body;
use axum::extract::State;
use axum::http::{header, HeaderMap, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::Json;
use bytes::Bytes;
use serde::{Deserialize, Serialize};
use serde_json::json;
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;
use tracing::warn;

use crate::oapi::deadline;
use crate::oapi::{kind, openai_error};

#[path = "json_ext.rs"]
pub mod json_ext;

use self::json_ext::OaiJson;

pub const DEFAULT_MAX_TOKENS: usize = 256;

pub const MAX_MAX_TOKENS: usize = 8192;

pub const SPEC_DECODE_HEADER: &str = "x-spec-decode";

pub fn spec_decode_header_value(status: Option<&'static str>) -> &'static str {
    match status {
        Some("on") => "on",
        Some("degraded") => "degraded",
        Some(_) => "unknown",
        None => "off",
    }
}

pub fn set_spec_decode_header(resp: &mut Response, value: &'static str) {
    if let Ok(name) = axum::http::HeaderName::from_bytes(SPEC_DECODE_HEADER.as_bytes()) {
        resp.headers_mut()
            .insert(name, axum::http::HeaderValue::from_static(value));
    }
}

pub fn spec_decode_header_for(
    registry: &crate::oapi::chat_engine::ChatRegistry,
    model: Option<&str>,
) -> &'static str {
    match registry.resolve(model) {
        Some(engine) => spec_decode_header_value(engine.spec_decode_status()),
        None => "unknown",
    }
}

#[derive(Clone, Debug, Deserialize)]
#[cfg_attr(
    feature = "ts-bindings",
    derive(ts_rs::TS),
    ts(export, optional_fields = nullable)
)]
pub struct ChatCompletionRequest {
    pub model: Option<String>,
    pub messages: Vec<ChatMessageIn>,
    #[serde(default)]
    pub stream: Option<bool>,
    #[serde(default)]
    pub max_tokens: Option<usize>,

    #[serde(default)]
    pub max_completion_tokens: Option<usize>,
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
    pub user: Option<String>,

    #[serde(default)]
    pub response_format: Option<serde_json::Value>,

    #[serde(default)]
    pub guided_json: Option<serde_json::Value>,
    #[serde(default)]
    pub guided_regex: Option<String>,
    #[serde(default)]
    pub guided_choice: Option<Vec<String>>,

    #[serde(default)]
    pub logit_bias: Option<std::collections::HashMap<String, f32>>,

    #[serde(default)]
    pub n: Option<u32>,
    #[serde(default)]
    pub best_of: Option<u32>,
    #[serde(default)]
    pub tools: Option<Vec<Tool>>,
    #[serde(default)]
    #[cfg_attr(
        feature = "ts-bindings",
        ts(
            optional = nullable,
            type = "\"none\" | \"auto\" | \"required\" | { function: { name: string } }"
        )
    )]
    pub tool_choice: Option<ToolChoice>,
    #[serde(default)]
    pub stream_options: Option<StreamOptions>,

    #[serde(default)]
    pub logprobs: Option<bool>,

    #[serde(default)]
    pub top_logprobs: Option<u32>,

    #[serde(default)]
    pub chat_template_kwargs: Option<serde_json::Value>,

    #[serde(default)]
    pub enable_thinking: Option<bool>,

    #[serde(default)]
    pub reasoning_effort: Option<String>,

    #[serde(default)]
    pub timeout: Option<f64>,
}

#[derive(Clone, Debug, Default, Deserialize)]
#[cfg_attr(feature = "ts-bindings", derive(ts_rs::TS), ts(export))]
pub struct StreamOptions {
    #[serde(default)]
    pub include_usage: bool,
}

fn default_function_type() -> String {
    "function".to_string()
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[cfg_attr(feature = "ts-bindings", derive(ts_rs::TS), ts(export))]
pub struct Tool {
    #[serde(default = "default_function_type", rename = "type")]
    pub kind: String,
    pub function: FunctionDef,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[cfg_attr(
    feature = "ts-bindings",
    derive(ts_rs::TS),
    ts(export, optional_fields = nullable)
)]
pub struct FunctionDef {
    pub name: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub parameters: Option<serde_json::Value>,
}

#[derive(Clone, Debug, Default)]
pub enum ToolChoice {
    None,
    #[default]
    Auto,
    Required,
    Function(String),
}

impl<'de> Deserialize<'de> for ToolChoice {
    fn deserialize<D>(d: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        #[derive(Deserialize)]
        #[serde(untagged)]
        enum Raw {
            Str(String),
            Obj { function: NamedFn },
        }
        #[derive(Deserialize)]
        struct NamedFn {
            name: String,
        }
        match Raw::deserialize(d)? {
            Raw::Str(s) => match s.as_str() {
                "none" => Ok(ToolChoice::None),
                "auto" => Ok(ToolChoice::Auto),
                "required" => Ok(ToolChoice::Required),
                other => Err(serde::de::Error::custom(format!(
                    "invalid tool_choice {other:?} (want none|auto|required|{{function}})"
                ))),
            },
            Raw::Obj { function } => Ok(ToolChoice::Function(function.name)),
        }
    }
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[cfg_attr(
    feature = "ts-bindings",
    derive(ts_rs::TS),
    ts(export, optional_fields = nullable)
)]
pub struct ToolCall {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub index: Option<u32>,
    pub id: String,
    #[serde(default = "default_function_type", rename = "type")]
    pub kind: String,
    pub function: FunctionCall,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[cfg_attr(feature = "ts-bindings", derive(ts_rs::TS), ts(export))]
pub struct FunctionCall {
    pub name: String,

    pub arguments: String,
}

pub fn extract_response_schema(rf: &serde_json::Value) -> Option<serde_json::Value> {
    let ty = rf.get("type").and_then(|v| v.as_str())?;
    if ty != "json_schema" {
        return None;
    }

    let js = rf.get("json_schema")?;
    js.get("schema").cloned().or_else(|| Some(js.clone()))
}

#[allow(clippy::result_large_err)]
fn resolve_guided(
    req: &ChatCompletionRequest,
) -> Result<Option<nv_grammar::GrammarSpec>, Response> {
    resolve_guided_fields(
        req.response_format.as_ref(),
        req.guided_json.as_ref(),
        req.guided_regex.as_ref(),
        req.guided_choice.as_ref(),
    )
}

#[allow(clippy::result_large_err)]
pub const GUIDED_FORCE_THINK_OFF_ENV: &str = "NV_GUIDED_FORCE_THINK_OFF";

fn guided_force_think_off() -> bool {
    std::env::var(GUIDED_FORCE_THINK_OFF_ENV).ok().as_deref() == Some("1")
}

pub(crate) fn resolve_guided_fields(
    response_format: Option<&serde_json::Value>,
    guided_json: Option<&serde_json::Value>,
    guided_regex: Option<&String>,
    guided_choice: Option<&Vec<String>>,
) -> Result<Option<nv_grammar::GrammarSpec>, Response> {
    use nv_grammar::GrammarSpec;
    if let Some(schema) = response_format.and_then(extract_response_schema) {
        return Ok(Some(GrammarSpec::JsonSchema(schema)));
    }

    if response_format
        .and_then(|rf| rf.get("type"))
        .and_then(|t| t.as_str())
        == Some("json_object")
    {
        return Ok(Some(GrammarSpec::Regex(nv_grammar::json_object_regex(3))));
    }
    if let Some(j) = guided_json {
        let schema = match j {
            serde_json::Value::String(s) => serde_json::from_str(s).map_err(|e| {
                openai_error(
                    StatusCode::BAD_REQUEST,
                    format!("guided_json is not valid JSON: {e}"),
                    "invalid_request_error",
                    Some("guided_json"),
                    None,
                )
            })?,
            other => other.clone(),
        };
        return Ok(Some(GrammarSpec::JsonSchema(schema)));
    }
    if let Some(re) = guided_regex {
        return Ok(Some(GrammarSpec::Regex((*re).clone())));
    }
    if let Some(choices) = guided_choice {
        if choices.is_empty() {
            return Err(openai_error(
                StatusCode::BAD_REQUEST,
                "guided_choice must be a non-empty list",
                "invalid_request_error",
                Some("guided_choice"),
                None,
            ));
        }
        return Ok(Some(GrammarSpec::Regex(nv_grammar::choice_to_regex(
            choices,
        ))));
    }
    Ok(None)
}

#[derive(Clone, Debug, Default)]
pub(crate) struct ToolPostProcess {
    pub active: bool,
    pub force_name: Option<String>,
}

pub(crate) struct ToolPolicy {
    pub tools_active: bool,
    pub force_name: Option<String>,
    pub guided: Option<nv_grammar::GrammarSpec>,
}

pub(crate) fn resolve_tool_policy(
    tools: &[Tool],
    choice: &ToolChoice,
    base_guided: Option<nv_grammar::GrammarSpec>,
) -> ToolPolicy {
    let tools_active = !tools.is_empty() && !matches!(choice, ToolChoice::None);
    let force_name = match choice {
        ToolChoice::Function(n) => Some(n.clone()),
        ToolChoice::Required if tools.len() == 1 => Some(tools[0].function.name.clone()),
        _ => None,
    };
    let mut guided = base_guided;
    if guided.is_none() {
        if let Some(name) = &force_name {
            if let Some(t) = tools.iter().find(|t| &t.function.name == name) {
                guided = Some(tool_args_grammar(t));
            }
        }
    }
    ToolPolicy {
        tools_active,
        force_name,
        guided,
    }
}

pub(crate) fn resolve_guided_think_close(
    engine: &dyn ChatEngine,
    guided: bool,
    extra_kwargs: &mut TemplateKwargs,
) -> Option<String> {
    if guided && guided_force_think_off() && !extra_kwargs.contains_key("enable_thinking") {
        extra_kwargs.insert("enable_thinking".into(), serde_json::Value::Bool(false));
    }
    let thinking_on = extra_kwargs
        .get("enable_thinking")
        .and_then(|v| v.as_bool())
        .or_else(|| crate::oapi::chat_engine::template_thinking_default(engine))
        .unwrap_or(false);
    crate::oapi::chat_engine::guided_think_close_marker(engine, guided, thinking_on)
}

fn tool_preamble(tools: &[Tool], choice: &ToolChoice) -> String {
    let mut s = String::from("You can call tools. ");
    match choice {
        ToolChoice::Required => s.push_str("You MUST call exactly one tool. "),
        ToolChoice::Function(name) => s.push_str(&format!("You MUST call the tool \"{name}\". ")),
        _ => s.push_str("Call a tool only if it helps answer. "),
    }
    s.push_str(
        "To call a tool, output ONLY:\n<tool_call>{\"name\": <tool>, \"arguments\": <json-object>}</tool_call>\nOtherwise answer normally.\n\nTools:\n",
    );
    for t in tools {
        let params = t
            .function
            .parameters
            .clone()
            .unwrap_or_else(|| serde_json::json!({"type":"object","properties":{}}));
        let line = serde_json::json!({
            "name": t.function.name,
            "description": t.function.description,
            "parameters": params,
        });
        s.push_str(&line.to_string());
        s.push('\n');
    }
    s
}

fn build_tool_messages(
    messages: &[ChatMessageIn],
    tools: &[Tool],
    choice: &ToolChoice,
) -> Vec<ChatMessageIn> {
    let preamble = tool_preamble(tools, choice);
    let mut out: Vec<ChatMessageIn> = Vec::with_capacity(messages.len() + 1);

    let has_leading_system = messages
        .first()
        .map(|m| m.role == "system")
        .unwrap_or(false);
    let (sys_text, rest) = if has_leading_system {
        (
            format!("{}\n\n{}", messages[0].text(), preamble),
            &messages[1..],
        )
    } else {
        (preamble, messages)
    };
    out.push(ChatMessageIn {
        role: "system".into(),
        content: Some(MessageContent::Text(sys_text)),
        ..Default::default()
    });
    for m in rest {
        let mut body = m.text();
        if let Some(calls) = &m.tool_calls {
            for c in calls {
                let args: serde_json::Value =
                    serde_json::from_str(&c.function.arguments).unwrap_or_default();
                let blk = serde_json::json!({"name": c.function.name, "arguments": args});
                if !body.is_empty() {
                    body.push('\n');
                }
                body.push_str(&format!("<tool_call>{}</tool_call>", blk));
            }
        }
        let (role, content) = if m.role == "tool" || m.role == "function" {
            let who = m
                .name
                .clone()
                .or_else(|| m.tool_call_id.clone())
                .unwrap_or_default();
            (
                "user".to_string(),
                format!("Tool result ({who}): {}", body.trim()),
            )
        } else {
            (m.role.clone(), body)
        };
        out.push(ChatMessageIn {
            role,
            content: Some(MessageContent::Text(content)),
            tool_calls: None,
            tool_call_id: None,
            name: None,
        });
    }
    out
}

pub const NATIVE_CALL_OPEN: &str = "<|tool_call>";
pub const NATIVE_CALL_CLOSE: &str = "<tool_call|>";
pub const NATIVE_STR: &str = "<|\"|>";

pub const NATIVE_WIRE_TOKENS: [&str; 3] = [NATIVE_CALL_OPEN, NATIVE_CALL_CLOSE, NATIVE_STR];

pub const TOOL_WIRE_TOKENS: [&str; 5] = [
    NATIVE_CALL_OPEN,
    NATIVE_CALL_CLOSE,
    NATIVE_STR,
    crate::oapi::tool_parse::HERMES_CALL_OPEN,
    crate::oapi::tool_parse::HERMES_CALL_CLOSE,
];

pub fn rewrite_native_tool_calls(text: &str) -> Option<String> {
    if !text.contains(NATIVE_CALL_OPEN) {
        return None;
    }
    let mut out = String::with_capacity(text.len());
    let mut rest = text;
    let mut rewrote = false;
    while let Some(start) = rest.find(NATIVE_CALL_OPEN) {
        out.push_str(&rest[..start]);
        let after = &rest[start + NATIVE_CALL_OPEN.len()..];
        let Some(end) = after.find(NATIVE_CALL_CLOSE) else {
            out.push_str(&rest[start..]);
            return rewrote.then_some(out);
        };
        match native_call_to_json(after[..end].trim()) {
            Some(json) => {
                out.push_str("<tool_call>");
                out.push_str(&json);
                out.push_str("</tool_call>");
                rewrote = true;
            }
            None => out.push_str(
                &rest[start..start + NATIVE_CALL_OPEN.len() + end + NATIVE_CALL_CLOSE.len()],
            ),
        }
        rest = &after[end + NATIVE_CALL_CLOSE.len()..];
    }
    out.push_str(rest);
    rewrote.then_some(out)
}

fn native_call_to_json(body: &str) -> Option<String> {
    let body = body.strip_prefix("call:")?;
    let brace = body.find('{')?;
    let name = body[..brace].trim();
    if name.is_empty() {
        return None;
    }
    let (args, used) = parse_native_value(&body[brace..])?;
    if !body[brace + used..].trim().is_empty() {
        return None;
    }
    serde_json::to_string(&serde_json::json!({"name": name, "arguments": args})).ok()
}

fn parse_native_value(s: &str) -> Option<(serde_json::Value, usize)> {
    let lead = s.len() - s.trim_start().len();
    let (v, used) = parse_native_value_at(&s[lead..])?;
    Some((v, lead + used))
}

fn parse_native_value_at(s: &str) -> Option<(serde_json::Value, usize)> {
    let bytes = s.as_bytes();
    match bytes.first()? {
        b'{' => {
            let mut map = serde_json::Map::new();
            let mut i = 1;
            loop {
                while i < s.len() && (bytes[i] == b',' || bytes[i].is_ascii_whitespace()) {
                    i += 1;
                }
                if i >= s.len() {
                    return None;
                }
                if bytes[i] == b'}' {
                    return Some((serde_json::Value::Object(map), i + 1));
                }
                let key_end = native_key_end(&s[i..])? + i;
                let key = native_key(s[i..key_end].trim());
                let (val, used) = parse_native_value(&s[key_end + 1..])?;
                map.insert(key, val);
                i = key_end + 1 + used;
            }
        }
        b'[' => {
            let mut items = Vec::new();
            let mut i = 1;
            loop {
                while i < s.len() && (bytes[i] == b',' || bytes[i].is_ascii_whitespace()) {
                    i += 1;
                }
                if i >= s.len() {
                    return None;
                }
                if bytes[i] == b']' {
                    return Some((serde_json::Value::Array(items), i + 1));
                }
                let (val, used) = parse_native_value(&s[i..])?;
                items.push(val);
                i += used;
            }
        }
        _ => {
            if let Some(rest) = s.strip_prefix(NATIVE_STR) {
                let end = rest.find(NATIVE_STR)?;
                let used = NATIVE_STR.len() * 2 + end;
                return Some((serde_json::Value::String(rest[..end].to_string()), used));
            }
            let end = s.find([',', '}', ']']).unwrap_or(s.len());
            let raw = s[..end].trim();
            let val = serde_json::from_str::<serde_json::Value>(raw)
                .unwrap_or_else(|_| serde_json::Value::String(raw.to_string()));
            Some((val, end))
        }
    }
}

fn native_key(raw: &str) -> String {
    raw.strip_prefix(NATIVE_STR)
        .and_then(|k| k.strip_suffix(NATIVE_STR))
        .unwrap_or(raw)
        .trim_matches('"')
        .to_string()
}

fn native_key_end(s: &str) -> Option<usize> {
    let colon = s.find(':')?;
    if s[..colon].contains(['{', '}', ',']) {
        return None;
    }
    Some(colon)
}

pub(crate) fn parse_model_tool_calls(
    text: &str,
    force_name: Option<&str>,
) -> crate::oapi::tool_parse::ParsedOutput {
    match rewrite_native_tool_calls(text) {
        Some(rewritten) => {
            let mut parsed = crate::oapi::tool_parse::parse_tool_calls(&rewritten, None);
            if let Some(name) = force_name {
                for c in parsed.tool_calls.iter_mut() {
                    c.function.name = name.to_string();
                }
            }
            parsed
        }
        None => crate::oapi::tool_parse::parse_tool_calls(text, force_name),
    }
}

pub const THINK_OPEN: &str = "<think>";
pub const THINK_CLOSE: &str = "</think>";

#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct ThinkPostProcess {
    pub active: bool,

    pub opened: bool,
}

pub fn split_thinking(text: &str, opened: bool) -> Option<(String, String)> {
    let body = if opened {
        text
    } else {
        text.trim_start().strip_prefix(THINK_OPEN)?
    };
    match body.find(THINK_CLOSE) {
        Some(i) => Some((
            body[..i].trim().to_string(),
            body[i + THINK_CLOSE.len()..].trim_start().to_string(),
        )),
        None => Some((body.trim().to_string(), String::new())),
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum ThinkState {
    Undecided,
    Reasoning,
    Content,
}

pub(crate) struct ThinkingStream {
    state: ThinkState,
    pending: String,
    content_lead: bool,
}

impl ThinkingStream {
    pub(crate) fn new(opened: bool) -> Self {
        Self {
            state: if opened {
                ThinkState::Reasoning
            } else {
                ThinkState::Undecided
            },
            pending: String::new(),
            content_lead: false,
        }
    }

    fn emit_content(&mut self, s: &str) -> String {
        if !self.content_lead {
            return s.to_string();
        }
        let t = s.trim_start();
        if t.is_empty() {
            return String::new();
        }
        self.content_lead = false;
        t.to_string()
    }

    pub(crate) fn push(&mut self, s: &str) -> (String, String) {
        if self.state == ThinkState::Content {
            let content = self.emit_content(s);
            return (String::new(), content);
        }
        self.pending.push_str(s);
        if self.state == ThinkState::Undecided {
            let lead = self.pending.trim_start();
            if let Some(rest) = lead.strip_prefix(THINK_OPEN) {
                self.pending = rest.to_string();
                self.state = ThinkState::Reasoning;
            } else if lead.is_empty() || THINK_OPEN.starts_with(lead) {
                return (String::new(), String::new());
            } else {
                self.state = ThinkState::Content;
                return (String::new(), std::mem::take(&mut self.pending));
            }
        }
        if let Some(i) = self.pending.find(THINK_CLOSE) {
            let reasoning = self.pending[..i].to_string();
            let rest = self.pending[i + THINK_CLOSE.len()..].to_string();
            self.pending.clear();
            self.state = ThinkState::Content;
            self.content_lead = true;
            let content = self.emit_content(&rest);
            return (reasoning, content);
        }
        let keep = emittable_len(&self.pending);
        (self.pending.drain(..keep).collect(), String::new())
    }

    pub(crate) fn finish(&mut self) -> (String, String) {
        let rest = std::mem::take(&mut self.pending);
        let was = self.state;
        self.state = ThinkState::Content;
        match was {
            ThinkState::Reasoning => (rest, String::new()),
            _ => (String::new(), rest),
        }
    }
}

fn emittable_len(s: &str) -> usize {
    let hold = THINK_CLOSE.len() - 1;
    if s.len() <= hold {
        return 0;
    }
    let mut i = s.len() - hold;
    while i > 0 && !s.is_char_boundary(i) {
        i -= 1;
    }
    i
}

pub(crate) fn tool_args_grammar(tool: &Tool) -> nv_grammar::GrammarSpec {
    let params = tool
        .function
        .parameters
        .clone()
        .unwrap_or_else(|| serde_json::json!({"type":"object","properties":{}}));
    nv_grammar::GrammarSpec::JsonSchema(sanitize_args_schema(&params))
}

fn sanitize_args_schema(schema: &serde_json::Value) -> serde_json::Value {
    use serde_json::{json, Value};
    let Some(props) = schema.get("properties").and_then(|p| p.as_object()) else {
        return json!({"type":"object","properties":{},"required":[]});
    };
    let mut out_props = serde_json::Map::new();
    let mut required = Vec::new();
    for (k, v) in props {
        out_props.insert(k.clone(), sanitize_value_schema(v));
        required.push(Value::String(k.clone()));
    }
    json!({"type":"object","properties":out_props,"required":required})
}

fn sanitize_value_schema(v: &serde_json::Value) -> serde_json::Value {
    use serde_json::json;
    if v.get("enum").is_some() || v.get("const").is_some() {
        return v.clone();
    }
    match v.get("type").and_then(|t| t.as_str()) {
        Some("object") => sanitize_args_schema(v),
        Some("array") => {
            let items = v
                .get("items")
                .map(sanitize_value_schema)
                .unwrap_or_else(|| json!({"type":"string"}));
            json!({"type":"array","items":items})
        }
        Some(t @ ("string" | "integer" | "number" | "boolean" | "null")) => json!({"type":t}),
        _ => json!({"type":"string"}),
    }
}

#[derive(Clone, Debug, Default, Deserialize, Serialize)]
#[cfg_attr(
    feature = "ts-bindings",
    derive(ts_rs::TS),
    ts(export, optional_fields = nullable)
)]
pub struct ChatMessageIn {
    pub role: String,

    #[serde(default)]
    pub content: Option<MessageContent>,

    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ToolCall>>,

    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[cfg_attr(feature = "ts-bindings", derive(ts_rs::TS), ts(export))]
#[serde(untagged)]
pub enum MessageContent {
    Text(String),
    Parts(Vec<ContentPart>),
}

#[derive(Clone, Debug, Serialize)]
#[cfg_attr(feature = "ts-bindings", derive(ts_rs::TS), ts(export))]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ContentPart {
    Text {
        text: String,
    },
    Unsupported {
        kind: String,
        #[serde(skip)]
        #[cfg_attr(feature = "ts-bindings", ts(skip))]
        raw: serde_json::Value,
    },
}

impl<'de> Deserialize<'de> for ContentPart {
    fn deserialize<D>(d: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let v = serde_json::Value::deserialize(d)?;
        match v.get("type").and_then(|t| t.as_str()) {
            Some("text") => match v.get("text").and_then(|t| t.as_str()) {
                Some(text) => Ok(ContentPart::Text {
                    text: text.to_string(),
                }),
                None => Err(serde::de::Error::missing_field("text")),
            },
            Some(other) => Ok(ContentPart::Unsupported {
                kind: other.to_string(),
                raw: v.clone(),
            }),
            None => Err(serde::de::Error::missing_field("type")),
        }
    }
}

pub(crate) fn first_unsupported_part(messages: &[ChatMessageIn]) -> Option<(usize, &str)> {
    messages.iter().enumerate().find_map(|(i, m)| {
        let Some(MessageContent::Parts(parts)) = m.content.as_ref() else {
            return None;
        };
        parts.iter().find_map(|p| match p {
            ContentPart::Unsupported { kind, .. } => Some((i, kind.as_str())),
            ContentPart::Text { .. } => None,
        })
    })
}

pub(crate) fn reject_unsupported_parts(messages: &[ChatMessageIn]) -> Option<Response> {
    let (idx, part_type) = first_unsupported_part(messages)?;
    let param = format!("messages[{idx}].content");
    Some(openai_error(
        StatusCode::BAD_REQUEST,
        format!(
            "messages[{idx}].content contains a `{part_type}` part; this endpoint only supports \
             `text` parts. The request was rejected rather than answered without the \
             `{part_type}` part."
        ),
        kind::INVALID_REQUEST,
        Some(param.as_str()),
        Some("unsupported_content_part"),
    ))
}

pub(crate) const GEMMA4_IMAGE_MARKER: &str = "<|image>";
pub(crate) const GEMMA4_AUDIO_MARKER: &str = "<|audio>";

fn mm_part_error(idx: usize, detail: &str) -> Response {
    let param = format!("messages[{idx}].content");
    openai_error(
        StatusCode::BAD_REQUEST,
        format!("messages[{idx}].content: {detail}"),
        kind::INVALID_REQUEST,
        Some(param.as_str()),
        Some("invalid_mm_part"),
    )
}

pub(crate) fn extract_mm_media(
    messages: &[ChatMessageIn],
    markers: (&'static str, &'static str),
) -> Result<(Vec<ChatMessageIn>, crate::oapi::chat_multimodal::MmMedia), Response> {
    let (image_marker, audio_marker) = markers;
    use crate::oapi::chat_multimodal::{
        decode_audio_input, decode_image_ref, ImageUrlSpec, InputAudioSpec, MmMedia,
    };
    let mut media = MmMedia::default();
    let mut out = Vec::with_capacity(messages.len());
    for (idx, m) in messages.iter().enumerate() {
        let Some(MessageContent::Parts(parts)) = m.content.as_ref() else {
            out.push(m.clone());
            continue;
        };
        let mut text = String::new();
        for p in parts {
            match p {
                ContentPart::Text { text: t } => text.push_str(t),
                ContentPart::Unsupported { kind, raw } if kind == "image_url" => {
                    let spec: ImageUrlSpec = raw
                        .get("image_url")
                        .cloned()
                        .and_then(|v| serde_json::from_value(v).ok())
                        .ok_or_else(|| {
                            mm_part_error(
                                idx,
                                "image_url must be a URL string or {\"url\": ...}",
                            )
                        })?;
                    let img = decode_image_ref(spec.url())
                        .map_err(|e| mm_part_error(idx, &format!("{e:#}")))?;
                    media.images.push(img);
                    text.push_str(image_marker);
                }
                ContentPart::Unsupported { kind, raw } if kind == "input_audio" => {
                    let spec: InputAudioSpec = raw
                        .get("input_audio")
                        .cloned()
                        .and_then(|v| serde_json::from_value(v).ok())
                        .ok_or_else(|| {
                            mm_part_error(
                                idx,
                                "input_audio must be {\"data\": <base64>, \"format\": ...}",
                            )
                        })?;
                    let samples = decode_audio_input(&spec)
                        .map_err(|e| mm_part_error(idx, &format!("{e:#}")))?;
                    media.audios.push(samples);
                    text.push_str(audio_marker);
                }
                ContentPart::Unsupported { kind, .. } => {
                    return Err(mm_part_error(
                        idx,
                        &format!(
                            "contains a `{kind}` part; this model supports `text`, `image_url` \
                             and `input_audio` parts"
                        ),
                    ));
                }
            }
        }
        out.push(ChatMessageIn {
            content: Some(MessageContent::Text(text)),
            ..m.clone()
        });
    }
    Ok((out, media))
}

async fn bridge_post_file(
    client: &reqwest::Client,
    url: &str,
    field_extra: Option<(&str, &str)>,
    file_name: &str,
    bytes: Vec<u8>,
) -> Result<String, String> {
    let mut form = reqwest::multipart::Form::new().part(
        "file",
        reqwest::multipart::Part::bytes(bytes).file_name(file_name.to_string()),
    );
    if let Some((k, v)) = field_extra {
        form = form.text(k.to_string(), v.to_string());
    }
    let resp = client
        .post(url)
        .multipart(form)
        .timeout(std::time::Duration::from_secs(180))
        .send()
        .await
        .map_err(|e| format!("bridge request to {url}: {e}"))?;
    let status = resp.status();
    let body = resp.text().await.unwrap_or_default();
    if !status.is_success() {
        let trimmed = &body[..body.len().min(300)];
        return Err(format!("bridge backend {status}: {trimmed}"));
    }
    match serde_json::from_str::<serde_json::Value>(&body) {
        Ok(v) => Ok(v
            .get("text")
            .and_then(|t| t.as_str())
            .unwrap_or(body.trim())
            .to_string()),
        Err(_) => Ok(body.trim().to_string()),
    }
}

pub(crate) async fn bridge_mm_parts(
    messages: &[ChatMessageIn],
) -> Result<Vec<ChatMessageIn>, Response> {
    use crate::oapi::chat_multimodal::{
        decode_b64, decode_data_url_bytes, ImageUrlSpec, InputAudioSpec,
    };
    let Some(addr) = crate::oapi::SELF_ADDR.get() else {
        return Err(mm_part_error(0, "perception bridge unavailable: listen address not bound"));
    };
    let base = format!("http://{addr}/v1");
    let client = reqwest::Client::new();
    let mut out = Vec::with_capacity(messages.len());
    for (idx, m) in messages.iter().enumerate() {
        let Some(MessageContent::Parts(parts)) = m.content.as_ref() else {
            out.push(m.clone());
            continue;
        };
        let mut text = String::new();
        for p in parts {
            match p {
                ContentPart::Text { text: t } => text.push_str(t),
                ContentPart::Unsupported { kind, raw } if kind == "image_url" => {
                    let spec: ImageUrlSpec = raw
                        .get("image_url")
                        .cloned()
                        .and_then(|v| serde_json::from_value(v).ok())
                        .ok_or_else(|| {
                            mm_part_error(idx, "image_url must be a URL string or {\"url\": ...}")
                        })?;
                    let bytes = decode_data_url_bytes(spec.url())
                        .map_err(|e| mm_part_error(idx, &format!("{e:#}")))?;
                    let ocr = bridge_post_file(
                        &client,
                        &format!("{base}/ocr"),
                        Some(("mode", "plain")),
                        "image",
                        bytes,
                    )
                    .await
                    .map_err(|e| mm_part_error(idx, &e))?;
                    text.push_str("[image, transcribed by ocr]\n");
                    text.push_str(&ocr);
                    text.push_str("\n[/image]");
                }
                ContentPart::Unsupported { kind, raw } if kind == "input_audio" => {
                    let spec: InputAudioSpec = raw
                        .get("input_audio")
                        .cloned()
                        .and_then(|v| serde_json::from_value(v).ok())
                        .ok_or_else(|| {
                            mm_part_error(
                                idx,
                                "input_audio must be {\"data\": <base64>, \"format\": ...}",
                            )
                        })?;
                    let bytes = decode_b64(&spec.data)
                        .map_err(|e| mm_part_error(idx, &format!("{e:#}")))?;
                    let name = format!("audio.{}", spec.format);
                    let transcript = bridge_post_file(
                        &client,
                        &format!("{base}/audio/transcriptions"),
                        None,
                        &name,
                        bytes,
                    )
                    .await
                    .map_err(|e| mm_part_error(idx, &e))?;
                    text.push_str("[audio, transcribed]\n");
                    text.push_str(&transcript);
                    text.push_str("\n[/audio]");
                }
                ContentPart::Unsupported { kind, .. } => {
                    return Err(mm_part_error(
                        idx,
                        &format!(
                            "contains a `{kind}` part; this model supports `text`, `image_url` \
                             and `input_audio` parts"
                        ),
                    ));
                }
            }
        }
        out.push(ChatMessageIn {
            content: Some(MessageContent::Text(text)),
            ..m.clone()
        });
    }
    Ok(out)
}

#[derive(Clone, Debug, Deserialize)]
#[cfg_attr(feature = "ts-bindings", derive(ts_rs::TS), ts(export))]
#[serde(untagged)]
pub enum StopField {
    One(String),
    Many(Vec<String>),
}

impl ChatMessageIn {
    pub fn text(&self) -> String {
        match self.content.as_ref() {
            None => String::new(),
            Some(MessageContent::Text(s)) => s.clone(),
            Some(MessageContent::Parts(parts)) => {
                let mut out = String::new();
                for p in parts {
                    if let ContentPart::Text { text } = p {
                        if !out.is_empty() {
                            out.push('\n');
                        }
                        out.push_str(text);
                    }
                }
                out
            }
        }
    }
}

#[derive(Clone, Debug, Serialize)]
#[cfg_attr(feature = "ts-bindings", derive(ts_rs::TS), ts(export))]
pub struct ChatCompletionResponse {
    pub id: String,
    pub object: &'static str,
    pub created: i64,
    pub model: String,
    pub choices: Vec<ChatChoice>,
    pub usage: Usage,
    #[serde(skip_serializing_if = "Option::is_none")]
    #[cfg_attr(feature = "ts-bindings", ts(optional))]
    pub system_fingerprint: Option<String>,
}

#[derive(Clone, Debug, Serialize)]
#[cfg_attr(feature = "ts-bindings", derive(ts_rs::TS), ts(export))]
pub struct ChatChoice {
    pub index: u32,
    pub message: ChatMessageOut,
    pub finish_reason: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    #[cfg_attr(feature = "ts-bindings", ts(optional))]
    pub logprobs: Option<LogprobsObject>,
}

#[derive(Clone, Debug, Serialize)]
#[cfg_attr(feature = "ts-bindings", derive(ts_rs::TS), ts(export))]
pub struct ChatMessageOut {
    pub role: String,

    pub content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    #[cfg_attr(feature = "ts-bindings", ts(optional))]
    pub reasoning_content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    #[cfg_attr(feature = "ts-bindings", ts(optional))]
    pub tool_calls: Option<Vec<ToolCall>>,
}

#[derive(Clone, Debug, Serialize)]
#[cfg_attr(feature = "ts-bindings", derive(ts_rs::TS), ts(export))]
pub struct Usage {
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    pub total_tokens: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    #[cfg_attr(feature = "ts-bindings", ts(optional))]
    pub prompt_tokens_details: Option<PromptTokensDetails>,
}

#[derive(Clone, Copy, Debug, Serialize)]
#[cfg_attr(feature = "ts-bindings", derive(ts_rs::TS), ts(export))]
pub struct PromptTokensDetails {
    pub cached_tokens: u32,
}

#[derive(Clone, Debug, Serialize)]
#[cfg_attr(feature = "ts-bindings", derive(ts_rs::TS), ts(export))]
pub struct TopLogprob {
    pub token: String,
    pub logprob: f32,
    pub bytes: Vec<u8>,
}

#[derive(Clone, Debug, Serialize)]
#[cfg_attr(feature = "ts-bindings", derive(ts_rs::TS), ts(export))]
pub struct LogprobEntry {
    pub token: String,
    pub logprob: f32,
    pub bytes: Vec<u8>,
    pub top_logprobs: Vec<TopLogprob>,
}

#[derive(Clone, Debug, Serialize)]
#[cfg_attr(feature = "ts-bindings", derive(ts_rs::TS), ts(export))]
pub struct LogprobsObject {
    pub content: Vec<LogprobEntry>,
}

#[derive(Clone, Debug, Serialize)]
#[cfg_attr(feature = "ts-bindings", derive(ts_rs::TS), ts(export))]
pub struct ChatCompletionChunk {
    pub id: String,
    pub object: &'static str,
    pub created: i64,
    pub model: String,
    pub choices: Vec<ChunkChoice>,
    #[serde(skip_serializing_if = "Option::is_none")]
    #[cfg_attr(feature = "ts-bindings", ts(optional))]
    pub system_fingerprint: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    #[cfg_attr(feature = "ts-bindings", ts(optional))]
    pub usage: Option<Usage>,
}

#[derive(Clone, Debug, Serialize)]
#[cfg_attr(feature = "ts-bindings", derive(ts_rs::TS), ts(export))]
pub struct ChunkChoice {
    pub index: u32,
    pub delta: Delta,
    pub finish_reason: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    #[cfg_attr(feature = "ts-bindings", ts(optional))]
    pub logprobs: Option<LogprobsObject>,
}

#[derive(Clone, Debug, Default, Serialize)]
#[cfg_attr(
    feature = "ts-bindings",
    derive(ts_rs::TS),
    ts(export, optional_fields)
)]
pub struct Delta {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub role: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning_content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ToolCall>>,
}

#[derive(Debug, Clone)]
pub struct ChatGenerateRequest {
    pub prompt: String,
    pub max_new_tokens: usize,
    pub stop: Vec<String>,
    pub seed: Option<u64>,
    pub temperature: Option<f32>,
    pub top_p: Option<f32>,
    pub top_k: Option<u32>,
    pub min_p: Option<f32>,
    pub presence_penalty: Option<f32>,
    pub frequency_penalty: Option<f32>,
    pub repetition_penalty: Option<f32>,
    pub guided: Option<nv_grammar::GrammarSpec>,

    pub guided_think_close: Option<String>,
    pub logit_bias: Vec<(u32, f32)>,
    pub logprobs: bool,
    pub top_logprobs: usize,

    pub kv_resume: Option<String>,
    pub kv_store: Option<String>,

    pub mm: Option<crate::oapi::chat_multimodal::MmMedia>,
}

#[derive(Debug, Clone, Copy)]
pub struct EngineBusy {
    pub permits: usize,
    pub waited_ms: u64,
}

impl EngineBusy {
    pub fn new(permits: usize, waited_ms: u64) -> Self {
        Self { permits, waited_ms }
    }
}

pub const ENGINE_BUSY_PREFIX: &str = "engine busy:";

impl std::fmt::Display for EngineBusy {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{ENGINE_BUSY_PREFIX} the chat surface is at capacity ({} concurrent) and the request \
             waited {} ms without a slot. Retry shortly, or raise NV_CHAT_CONCURRENCY / \
             NV_CHAT_QUEUE_MS.",
            self.permits, self.waited_ms
        )
    }
}

impl std::error::Error for EngineBusy {}

#[derive(Debug, Clone)]
pub struct UnsupportedMedia(pub String);

impl std::fmt::Display for UnsupportedMedia {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for UnsupportedMedia {}

pub fn is_busy_shed(msg: &str) -> bool {
    msg.contains(ENGINE_BUSY_PREFIX) || msg.contains(crate::oapi::admission::REJECT_PREFIX)
}

pub(crate) fn engine_event_error_by_surface(
    msg: String,
    shed_note: &str,
    unavailable: impl FnOnce(String) -> Response,
    server: impl FnOnce(String) -> Response,
) -> Response {
    if is_busy_shed(&msg) {
        warn!(reason = %msg, "{}", shed_note);
        return unavailable(msg);
    }
    server(msg)
}

pub(crate) fn engine_start_error_by_surface(
    err: &anyhow::Error,
    shed_note: &str,
    unavailable: impl FnOnce(String) -> Response,
    server: impl FnOnce(String) -> Response,
) -> Response {
    if let Some(busy) = err.downcast_ref::<EngineBusy>() {
        warn!(permits = busy.permits, waited_ms = busy.waited_ms, "{}", shed_note);
        return unavailable(format!("{busy}"));
    }
    server(format!("engine: {err}"))
}

pub(crate) fn engine_event_error_response(msg: String) -> Response {
    engine_event_error_by_surface(
        msg,
        "shed a chat request: capacity ran out before generation started",
        |m| {
            openai_error(
                StatusCode::SERVICE_UNAVAILABLE,
                m,
                kind::SERVICE_UNAVAIL,
                None,
                Some("engine_busy"),
            )
        },
        |m| {
            openai_error(
                StatusCode::INTERNAL_SERVER_ERROR,
                m,
                kind::SERVER,
                None,
                Some("engine_error"),
            )
        },
    )
}

pub(crate) fn deadline_shed_response(budget_ms: u64, source: &'static str) -> Response {
    warn!(
        budget_ms,
        source,
        "shed a chat request: the caller-supplied deadline expired before generation started"
    );
    openai_error(
        StatusCode::SERVICE_UNAVAILABLE,
        format!(
            "the chat surface did not start this request within the caller-supplied deadline of \
             {budget_ms} ms (from the {source}); it is at capacity. Retry shortly, or send a \
             larger deadline."
        ),
        kind::SERVICE_UNAVAIL,
        None,
        Some("surface_busy"),
    )
}

#[derive(Debug, Clone, Copy)]
pub struct ClientDeadline {
    pub budget: Duration,
    pub source: &'static str,
}

impl ClientDeadline {
    pub fn from_request(timeout_secs: Option<f64>, headers: &HeaderMap) -> Option<Self> {
        let source = if deadline::from_body_seconds(timeout_secs).is_some() {
            "timeout body field"
        } else {
            "x-request-timeout-ms header"
        };
        let client = deadline::client_budget(deadline::from_body_seconds(timeout_secs), headers)?;
        Some(Self {
            budget: deadline::resolve(Some(client), Duration::ZERO),
            source,
        })
    }

    pub fn budget_ms(&self) -> u64 {
        self.budget.as_millis() as u64
    }
}

pub(crate) fn engine_start_error_response(err: &anyhow::Error) -> Response {
    if let Some(unsupported) = err.downcast_ref::<UnsupportedMedia>() {
        return openai_error(
            StatusCode::BAD_REQUEST,
            unsupported.0.clone(),
            kind::INVALID_REQUEST,
            Some("messages"),
            Some("unsupported_media"),
        );
    }
    engine_start_error_by_surface(
        err,
        "shed a chat request: the surface was at capacity for the whole queue window",
        |m| {
            openai_error(
                StatusCode::SERVICE_UNAVAILABLE,
                m,
                kind::SERVICE_UNAVAIL,
                None,
                Some("engine_busy"),
            )
        },
        |m| {
            openai_error(
                StatusCode::INTERNAL_SERVER_ERROR,
                m,
                kind::SERVER,
                None,
                Some("engine_unavailable"),
            )
        },
    )
}

#[derive(Debug, Clone)]
pub enum ChatEvent {
    Started {
        prompt_tokens: u32,
    },

    PromptCached {
        cached_tokens: u32,
    },

    StoppedBy {
        stop_sequence: String,
    },

    TextDelta(String),

    ReasoningDelta(String),

    Logprob(LogprobEntry),

    Done {
        finish_reason: String,
        completion_tokens: u32,
    },
    Error(String),
}

#[async_trait::async_trait]
pub trait ChatEngine: Send + Sync + 'static {
    fn model_id(&self) -> &str;

    fn spec_decode_status(&self) -> Option<&'static str> {
        None
    }

    fn supports_mm_input(&self) -> bool {
        false
    }

    fn mm_markers(&self) -> (&'static str, &'static str) {
        (GEMMA4_IMAGE_MARKER, GEMMA4_AUDIO_MARKER)
    }

    async fn generate(
        &self,
        req: ChatGenerateRequest,
        tx: mpsc::Sender<ChatEvent>,
    ) -> anyhow::Result<()>;

    fn render_prompt(&self, messages: &[ChatMessageIn]) -> String {
        render_chat_prompt(messages)
    }

    fn official_template(&self) -> Option<&crate::oapi::chat_template::ChatTemplate> {
        None
    }

    fn render_chat(
        &self,
        messages: &[ChatMessageIn],
        tools: &[Tool],
        choice: &ToolChoice,
    ) -> String {
        match self.official_template() {
            Some(t) => match render_official(t, messages, tools, choice) {
                Ok(s) => return s,
                Err(err) => {
                    note_builtin_fallback(format!("official chat template render failed: {err}"));
                    warn!(
                        model = %self.model_id(),
                        error = %err,
                        "official chat template render failed; falling back to the built-in renderer"
                    );
                }
            },
            None => {
                note_builtin_fallback("no official chat template is loaded for this model".into());
                log_missing_official_template(self.model_id());
            }
        }
        let msgs = if tools.is_empty() {
            std::borrow::Cow::Borrowed(messages)
        } else {
            std::borrow::Cow::Owned(build_tool_messages(messages, tools, choice))
        };
        self.render_prompt(&msgs)
    }

    fn render_chat_kwargs(
        &self,
        messages: &[ChatMessageIn],
        tools: &[Tool],
        choice: &ToolChoice,
        extra: &TemplateKwargs,
    ) -> String {
        if extra.is_empty() {
            return self.render_chat(messages, tools, choice);
        }
        let Some(t) = self.official_template() else {
            return self.render_chat(messages, tools, choice);
        };
        match render_official_with_kwargs(t, messages, tools, choice, extra) {
            Ok(s) => s,
            Err(err) => {
                note_builtin_fallback(format!("official chat template render failed: {err}"));
                warn!(
                    model = %self.model_id(),
                    error = %err,
                    "official chat template render failed with request chat_template_kwargs; \
                     falling back to the built-in renderer"
                );
                self.render_chat(messages, tools, choice)
            }
        }
    }

    fn thinking_split_supported(&self) -> bool {
        self.official_template()
            .map(|t| t.declares_thinking_switch())
            .unwrap_or(false)
    }
}

pub const REQUIRE_CHAT_TEMPLATE_ENV: &str = "NV_REQUIRE_CHAT_TEMPLATE";

pub const ALLOW_FALLBACK_ENV: &str = "NV_ALLOW_CHATML_FALLBACK";

pub fn require_official_template_from(raw: Option<&str>) -> bool {
    matches!(
        raw.unwrap_or_default().trim().to_ascii_lowercase().as_str(),
        "1" | "true" | "yes" | "on"
    )
}

pub fn allow_builtin_fallback_from(raw: Option<&str>) -> bool {
    require_official_template_from(raw)
}

fn explicitly_off(raw: Option<&str>) -> bool {
    matches!(
        raw.unwrap_or_default().trim().to_ascii_lowercase().as_str(),
        "0" | "false" | "no" | "off"
    )
}

pub fn template_required_from(
    require_raw: Option<&str>,
    allow_raw: Option<&str>,
    model_backed: bool,
) -> bool {
    if require_official_template_from(require_raw) {
        return true;
    }
    if allow_builtin_fallback_from(allow_raw) || explicitly_off(require_raw) {
        return false;
    }
    model_backed
}

pub fn template_required_for(model_id: &str) -> bool {
    template_required_from(
        std::env::var(REQUIRE_CHAT_TEMPLATE_ENV).ok().as_deref(),
        std::env::var(ALLOW_FALLBACK_ENV).ok().as_deref(),
        crate::oapi::chat_template::load_was_attempted(model_id),
    )
}

pub fn require_official_template() -> bool {
    template_required_from(
        std::env::var(REQUIRE_CHAT_TEMPLATE_ENV).ok().as_deref(),
        std::env::var(ALLOW_FALLBACK_ENV).ok().as_deref(),
        true,
    )
}

thread_local! {
    static BUILTIN_FALLBACK: std::cell::RefCell<Option<String>> =
        const { std::cell::RefCell::new(None) };
}

pub(crate) fn note_builtin_fallback(reason: String) {
    BUILTIN_FALLBACK.with(|c| *c.borrow_mut() = Some(reason));
}

pub fn take_builtin_fallback() -> Option<String> {
    BUILTIN_FALLBACK.with(|c| c.borrow_mut().take())
}

fn first_report_for(key: String) -> bool {
    static SEEN: std::sync::OnceLock<std::sync::Mutex<std::collections::HashSet<String>>> =
        std::sync::OnceLock::new();
    SEEN.get_or_init(Default::default)
        .lock()
        .map(|mut s| s.insert(key))
        .unwrap_or(false)
}

pub fn searched_dir_for(model_id: &str) -> Option<String> {
    crate::oapi::chat_template::load_attempt_for(model_id).map(|a| a.dir.display().to_string())
}

pub fn log_missing_official_template(model_id: &str) {
    if !first_report_for(format!("missing:{model_id}")) {
        return;
    }
    tracing::error!(
        model = %model_id,
        searched = %searched_dir_for(model_id).unwrap_or_else(|| "<no model directory>".into()),
        fallback = "chatml",
        "no official chat template for this model: requests would be rendered with the \
         built-in ChatML fallback (<|im_start|>role ... <|im_end|>). That is the wrong prompt \
         format for Gemma-4 (<|turn>role ... <turn|>) and drops Qwen3.6's <think> opener; \
         measured on Gemma-4-31B-IT it ends the turn on 3 of 7 prompts and makes the model \
         role-play both sides, so output quality and stop behaviour are NOT representative. \
         Point the engine at a model directory containing chat_template.jinja (or \
         tokenizer_config.json:chat_template). Serving it anyway now requires {}=1; {}=1 \
         refuses even for engines that never loaded a model directory.",
        ALLOW_FALLBACK_ENV,
        REQUIRE_CHAT_TEMPLATE_ENV
    );
}

pub fn log_chat_template_status(engine: &dyn ChatEngine) {
    match engine.official_template() {
        Some(_) => tracing::info!(
            model = %engine.model_id(),
            chat_template = "official",
            "chat prompts render with the model's official chat template"
        ),
        None => log_missing_official_template(engine.model_id()),
    }
}

pub fn missing_template_message(model_id: &str, reason: &str) -> String {
    let searched = searched_dir_for(model_id)
        .unwrap_or_else(|| "<none: this engine never loaded a model directory>".to_string());
    format!(
        "refusing to serve model `{model_id}` with the built-in ChatML fallback prompt: {reason}. \
         Searched for chat_template.jinja and tokenizer_config.json:chat_template in {searched}. \
         The fallback is measurably broken on Gemma-4 (3 of 7 prompts never end their turn, the \
         model role-plays both sides), so it is not served by default. Set {ALLOW_FALLBACK_ENV}=1 \
         to opt back in to it, or point the engine at a model directory that ships a chat template."
    )
}

fn missing_template_refusal(message: String) -> Response {
    openai_error(
        StatusCode::INTERNAL_SERVER_ERROR,
        message,
        "server_error",
        None,
        Some("chat_template_missing"),
    )
}

pub fn render_chat_checked(
    engine: &dyn ChatEngine,
    messages: &[ChatMessageIn],
    tools: &[Tool],
    choice: &ToolChoice,
    required: bool,
) -> Result<String, String> {
    render_chat_checked_kwargs(
        engine,
        messages,
        tools,
        choice,
        required,
        &TemplateKwargs::new(),
    )
}

pub fn render_chat_checked_kwargs(
    engine: &dyn ChatEngine,
    messages: &[ChatMessageIn],
    tools: &[Tool],
    choice: &ToolChoice,
    required: bool,
    extra: &TemplateKwargs,
) -> Result<String, String> {
    if engine.official_template().is_none() {
        log_missing_official_template(engine.model_id());
        if required {
            return Err(missing_template_message(
                engine.model_id(),
                "no official chat template is loaded for this model",
            ));
        }
    }
    let _ = take_builtin_fallback();
    let prompt = engine.render_chat_kwargs(messages, tools, choice, extra);
    match take_builtin_fallback() {
        Some(reason) if required => Err(missing_template_message(engine.model_id(), &reason)),
        _ => Ok(prompt),
    }
}

pub fn render_chat_strict(
    engine: &dyn ChatEngine,
    messages: &[ChatMessageIn],
    tools: &[Tool],
    choice: &ToolChoice,
) -> Result<String, String> {
    let required = template_required_for(engine.model_id());
    render_chat_checked(engine, messages, tools, choice, required)
}

pub(crate) fn tool_choice_directive(choice: &ToolChoice) -> Option<String> {
    match choice {
        ToolChoice::Required => {
            Some("You must call one of the declared tools before answering.".to_string())
        }
        ToolChoice::Function(name) => Some(format!(
            "You must call the declared tool `{name}` before answering."
        )),
        _ => None,
    }
}

pub(crate) fn with_system_directive(
    messages: &[ChatMessageIn],
    directive: &str,
) -> Vec<ChatMessageIn> {
    let mut out: Vec<ChatMessageIn> = Vec::with_capacity(messages.len() + 1);
    let leading_system = messages
        .first()
        .map(|m| m.role == "system")
        .unwrap_or(false);
    let text = if leading_system {
        format!("{}\n\n{directive}", messages[0].text())
    } else {
        directive.to_string()
    };
    out.push(ChatMessageIn {
        role: "system".into(),
        content: Some(MessageContent::Text(text)),
        ..Default::default()
    });
    out.extend(messages[usize::from(leading_system)..].iter().cloned());
    out
}

pub(crate) fn template_messages_json(
    messages: &[ChatMessageIn],
) -> anyhow::Result<serde_json::Value> {
    let mut msgs = serde_json::to_value(messages)?;
    if let Some(arr) = msgs.as_array_mut() {
        for m in arr.iter_mut() {
            let Some(calls) = m.get_mut("tool_calls").and_then(|c| c.as_array_mut()) else {
                continue;
            };
            for c in calls.iter_mut() {
                let Some(args) = c.pointer("/function/arguments").and_then(|a| a.as_str()) else {
                    continue;
                };
                let Ok(parsed) = serde_json::from_str::<serde_json::Value>(args) else {
                    continue;
                };
                if parsed.is_object() {
                    if let Some(slot) = c.pointer_mut("/function/arguments") {
                        *slot = parsed;
                    }
                }
            }
        }
    }
    Ok(msgs)
}

pub type TemplateKwargs = std::collections::BTreeMap<String, serde_json::Value>;

pub(crate) fn merged_template_kwargs(
    template: &crate::oapi::chat_template::ChatTemplate,
    extra: &TemplateKwargs,
) -> TemplateKwargs {
    let mut merged = template.effective_template_kwargs();
    for (k, v) in extra {
        merged.insert(k.clone(), v.clone());
    }
    merged
}

#[allow(clippy::result_large_err)]
pub(crate) fn request_template_kwargs(
    req: &ChatCompletionRequest,
) -> Result<TemplateKwargs, Response> {
    let mut out = TemplateKwargs::new();
    if let Some(on) = req.enable_thinking {
        out.insert("enable_thinking".into(), serde_json::Value::Bool(on));
    }
    if let Some(effort) = &req.reasoning_effort {
        if let Err(msg) = crate::oapi::chat_template::validate_reasoning_effort(effort) {
            return Err(openai_error(
                StatusCode::BAD_REQUEST,
                msg,
                kind::INVALID_REQUEST,
                Some("reasoning_effort"),
                None,
            ));
        }
        out.insert(
            crate::oapi::chat_template::REASONING_EFFORT_KWARG.into(),
            serde_json::Value::String(effort.clone()),
        );
    }
    match req.chat_template_kwargs.as_ref() {
        None | Some(serde_json::Value::Null) => {}
        Some(serde_json::Value::Object(map)) => {
            for (k, v) in map {
                out.insert(k.clone(), v.clone());
            }
        }
        Some(other) => {
            return Err(openai_error(
                StatusCode::BAD_REQUEST,
                format!("chat_template_kwargs must be a JSON object, got {other}"),
                kind::INVALID_REQUEST,
                Some("chat_template_kwargs"),
                None,
            ))
        }
    }
    Ok(out)
}

pub(crate) fn render_official(
    template: &crate::oapi::chat_template::ChatTemplate,
    messages: &[ChatMessageIn],
    tools: &[Tool],
    choice: &ToolChoice,
) -> anyhow::Result<String> {
    render_official_with_kwargs(template, messages, tools, choice, &TemplateKwargs::new())
}

pub(crate) fn render_official_with_kwargs(
    template: &crate::oapi::chat_template::ChatTemplate,
    messages: &[ChatMessageIn],
    tools: &[Tool],
    choice: &ToolChoice,
    extra: &TemplateKwargs,
) -> anyhow::Result<String> {
    let kw = merged_template_kwargs(template, extra);
    if tools.is_empty() {
        return template.render_with_kwargs(&template_messages_json(messages)?, None, true, &kw);
    }
    if template.supports_tools() {
        let tools_json = serde_json::to_value(tools)?;
        let msgs = match tool_choice_directive(choice) {
            Some(d) => std::borrow::Cow::Owned(with_system_directive(messages, &d)),
            None => std::borrow::Cow::Borrowed(messages),
        };
        return template.render_with_kwargs(
            &template_messages_json(&msgs)?,
            Some(&tools_json),
            true,
            &kw,
        );
    }
    warn!(
        tools = tools.len(),
        "this model's official chat template ignores the `tools` variable: falling back to a \
         synthetic system message that asks for <tool_call>{{...}}</tool_call>, which the model \
         was probably not trained to emit"
    );
    let flattened = build_tool_messages(messages, tools, choice);
    template.render_with_kwargs(&template_messages_json(&flattened)?, None, true, &kw)
}

#[derive(Clone)]
pub struct ChatAppState {
    pub registry: crate::oapi::chat_engine::ChatRegistry,
}

pub fn model_not_found(model: &str) -> Response {
    openai_error(
        StatusCode::NOT_FOUND,
        format!("The model `{model}` does not exist."),
        "invalid_request_error",
        Some("model"),
        Some("model_not_found"),
    )
}

pub async fn handle_chat_completions(
    State(state): State<ChatAppState>,
    headers: HeaderMap,
    OaiJson(req): OaiJson<ChatCompletionRequest>,
) -> Response {
    let spec = spec_decode_header_for(&state.registry, req.model.as_deref());
    let client = ClientDeadline::from_request(req.timeout, &headers);
    let mut resp = chat_completions_impl(state, req, client).await;
    set_spec_decode_header(&mut resp, spec);
    resp
}

async fn chat_completions_impl(
    state: ChatAppState,
    mut req: ChatCompletionRequest,
    client: Option<ClientDeadline>,
) -> Response {
    if req.messages.is_empty() {
        return crate::oapi::fastapi_validation_error(vec![crate::oapi::missing_field(&[
            "body", "messages",
        ])]);
    }

    let engine = match state.registry.resolve(req.model.as_deref()) {
        Some(e) => e,
        None => return model_not_found(req.model.as_deref().unwrap_or("")),
    };

    let mm_media = if first_unsupported_part(&req.messages).is_some() {
        if engine.supports_mm_input() {
            match extract_mm_media(&req.messages, engine.mm_markers()) {
                Ok((msgs, media)) => {
                    req.messages = msgs;
                    Some(media)
                }
                Err(resp) => return resp,
            }
        } else {
            match bridge_mm_parts(&req.messages).await {
                Ok(msgs) => {
                    req.messages = msgs;
                    None
                }
                Err(resp) => return resp,
            }
        }
    } else {
        None
    };

    let template_required = template_required_for(engine.model_id());

    let n = req.n.unwrap_or(1).max(1);
    const MAX_N: u32 = 16;
    if n > MAX_N {
        return openai_error(
            StatusCode::BAD_REQUEST,
            format!("n must be <= {MAX_N}"),
            "invalid_request_error",
            Some("n"),
            None,
        );
    }
    if let Some(best_of) = req.best_of {
        if best_of != n {
            return openai_error(
                StatusCode::BAD_REQUEST,
                "best_of is only supported when best_of == n",
                "invalid_request_error",
                Some("best_of"),
                None,
            );
        }
    }
    let tools = req.tools.clone().unwrap_or_default();
    let choice = req.tool_choice.clone().unwrap_or_default();
    if let ToolChoice::Function(name) = &choice {
        if !tools.iter().any(|t| &t.function.name == name) {
            return openai_error(
                StatusCode::BAD_REQUEST,
                format!("tool_choice names unknown function {name:?}"),
                "invalid_request_error",
                Some("tool_choice"),
                None,
            );
        }
    }
    if let Some(t) = req.top_logprobs {
        if t > 20 {
            return openai_error(
                StatusCode::BAD_REQUEST,
                "top_logprobs must be in [0, 20]",
                "invalid_request_error",
                Some("top_logprobs"),
                None,
            );
        }
    }
    let want_logprobs = req.logprobs.unwrap_or(false);
    let top_logprobs = if want_logprobs {
        req.top_logprobs.unwrap_or(0) as usize
    } else {
        0
    };

    let base_guided = match resolve_guided(&req) {
        Ok(g) => g,
        Err(resp) => return resp,
    };
    let logit_bias: Vec<(u32, f32)> = req
        .logit_bias
        .as_ref()
        .map(|m| {
            m.iter()
                .filter_map(|(k, &v)| k.parse::<u32>().ok().map(|id| (id, v)))
                .collect()
        })
        .unwrap_or_default();

    let mut extra_kwargs = match request_template_kwargs(&req) {
        Ok(k) => k,
        Err(resp) => return resp,
    };

    if let Some(v) = extra_kwargs.get("enable_thinking") {
        if !v.is_boolean() {
            return openai_error(
                StatusCode::BAD_REQUEST,
                format!("enable_thinking must be a boolean, got {v}"),
                kind::INVALID_REQUEST,
                Some("enable_thinking"),
                None,
            );
        }
    }
    let ToolPolicy {
        tools_active,
        force_name,
        guided,
    } = resolve_tool_policy(&tools, &choice, base_guided);
    let guided_think_close =
        resolve_guided_think_close(engine.as_ref(), guided.is_some(), &mut extra_kwargs);
    let render_tools: &[Tool] = if tools_active { &tools } else { &[] };
    let prompt = match render_chat_checked_kwargs(
        engine.as_ref(),
        &req.messages,
        render_tools,
        &choice,
        template_required,
        &extra_kwargs,
    ) {
        Ok(p) => p,
        Err(message) => {
            tracing::error!(model = %engine.model_id(), reason = %message, "chat request refused");
            return missing_template_refusal(message);
        }
    };
    let think = ThinkPostProcess {
        active: engine.thinking_split_supported(),
        opened: prompt.trim_end().ends_with(THINK_OPEN),
    };
    let stop = match &req.stop {
        Some(StopField::One(s)) => vec![s.clone()],
        Some(StopField::Many(v)) => v.clone(),
        None => Vec::new(),
    };
    let max_new_tokens = req
        .max_completion_tokens
        .or(req.max_tokens)
        .unwrap_or(DEFAULT_MAX_TOKENS)
        .clamp(1, MAX_MAX_TOKENS);
    if let Some(asked) = req
        .max_completion_tokens
        .or(req.max_tokens)
        .filter(|&asked| asked > MAX_MAX_TOKENS)
    {
        tracing::info!(
            asked,
            effective = max_new_tokens,
            "max_tokens clamped to MAX_MAX_TOKENS; the response will carry finish_reason=length \
             if generation reaches it"
        );
    }

    let gen = ChatGenerateRequest {
        prompt,
        max_new_tokens,
        stop,
        seed: None,
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
        mm: mm_media,
    };
    let stream = req.stream.unwrap_or(false);
    let include_usage = req
        .stream_options
        .as_ref()
        .map(|o| o.include_usage)
        .unwrap_or(false);
    let model_id = engine.model_id().to_string();
    let id = format!("chatcmpl-{}", uuid::Uuid::new_v4().simple());
    let created = now_unix_secs();

    let base_seed = req.seed.unwrap_or_else(rand_seed);
    let tpp = ToolPostProcess {
        active: tools_active,
        force_name,
    };
    if stream {
        run_streaming(
            engine,
            gen,
            n,
            base_seed,
            id,
            model_id,
            created,
            include_usage,
            tpp,
            think,
            client,
        )
        .await
    } else {
        run_non_streaming(
            engine, gen, n, base_seed, id, model_id, created, tpp, think, client,
        )
        .await
    }
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
            Some(ChatEvent::Error(msg)) if is_busy_shed(&msg) => {
                Err(engine_event_error_response(msg))
            }
            other => Ok(other),
        },
        Err(_) => Err(deadline_shed_response(client.budget_ms(), client.source)),
    }
}

pub(crate) fn rand_seed() -> u64 {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos() as u64)
        .unwrap_or(0);
    let pid = std::process::id() as u64;
    let mut z = nanos
        .wrapping_mul(0x9E37_79B9_7F4A_7C15)
        .wrapping_add(pid.wrapping_mul(0xBF58_476D_1CE4_E5B9));
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z ^ (z >> 31)
}

#[allow(clippy::too_many_arguments)]
async fn run_non_streaming(
    engine: Arc<dyn ChatEngine>,
    gen: ChatGenerateRequest,
    n: u32,
    base_seed: u64,
    id: String,
    model: String,
    created: i64,
    tpp: ToolPostProcess,
    think: ThinkPostProcess,
    client: Option<ClientDeadline>,
) -> Response {
    let mut choices: Vec<ChatChoice> = Vec::with_capacity(n as usize);
    let mut prompt_tokens: u32 = 0;
    let mut cached_tokens: Option<u32> = None;
    let mut total_completion_tokens: u32 = 0;

    for i in 0..n {
        let mut g = gen.clone();
        g.seed = Some(base_seed.wrapping_add(i as u64));

        let (tx, mut rx) = mpsc::channel::<ChatEvent>(64);
        if let Err(err) = engine.generate(g, tx).await {
            warn!(error = %err, "chat engine.generate failed to start");
            return engine_start_error_response(&err);
        }

        let client = if i == 0 { client } else { None };
        let mut pre = match first_event(&mut rx, client).await {
            Ok(ev) => ev,
            Err(resp) => return resp,
        };

        let mut text = String::new();
        let mut engine_reasoning = String::new();
        let mut completion_tokens: u32 = 0;
        let mut finish_reason = String::from("stop");
        let mut lp_entries: Vec<LogprobEntry> = Vec::new();
        #[allow(clippy::while_let_loop)]
        loop {
            let Some(ev) = (match pre.take() {
                Some(ev) => Some(ev),
                None => rx.recv().await,
            }) else {
                break;
            };
            match ev {
                ChatEvent::Started { prompt_tokens: p } => prompt_tokens = p,
                ChatEvent::PromptCached { cached_tokens: c } => cached_tokens = Some(c),
                ChatEvent::StoppedBy { .. } => {}
                ChatEvent::TextDelta(s) => text.push_str(&s),
                ChatEvent::ReasoningDelta(s) => engine_reasoning.push_str(&s),
                ChatEvent::Logprob(e) => lp_entries.push(e),
                ChatEvent::Done {
                    finish_reason: r,
                    completion_tokens: c,
                } => {
                    finish_reason = r;
                    completion_tokens = c;
                }
                ChatEvent::Error(msg) => return engine_event_error_response(msg),
            }
        }

        total_completion_tokens = total_completion_tokens.saturating_add(completion_tokens);
        let logprobs = if lp_entries.is_empty() {
            None
        } else {
            Some(LogprobsObject {
                content: lp_entries,
            })
        };
        let (reasoning_content, text) = match think.active {
            true => match split_thinking(&text, think.opened) {
                Some((r, c)) => ((!r.is_empty()).then_some(r), c),
                None => (None, text),
            },
            false => ((!engine_reasoning.is_empty()).then_some(engine_reasoning), text),
        };
        let (content, tool_calls, finish_reason) = if tpp.active {
            let parsed = parse_model_tool_calls(&text, tpp.force_name.as_deref());
            if parsed.tool_calls.is_empty() {
                (parsed.content, None, finish_reason)
            } else {
                (
                    parsed.content,
                    Some(parsed.tool_calls),
                    "tool_calls".to_string(),
                )
            }
        } else {
            (Some(text), None, finish_reason)
        };
        choices.push(ChatChoice {
            index: i,
            message: ChatMessageOut {
                role: "assistant".into(),
                content,
                reasoning_content,
                tool_calls,
            },
            finish_reason,
            logprobs,
        });
    }

    let resp = ChatCompletionResponse {
        id,
        object: "chat.completion",
        created,
        system_fingerprint: Some(system_fingerprint(&model)),
        model,
        choices,
        usage: Usage {
            prompt_tokens,
            completion_tokens: total_completion_tokens,
            total_tokens: prompt_tokens + total_completion_tokens,
            prompt_tokens_details: cached_tokens
                .map(|cached_tokens| PromptTokensDetails { cached_tokens }),
        },
    };
    (StatusCode::OK, Json(resp)).into_response()
}

#[allow(clippy::too_many_arguments)]
async fn run_streaming(
    engine: Arc<dyn ChatEngine>,
    gen: ChatGenerateRequest,
    n: u32,
    base_seed: u64,
    id: String,
    model: String,
    created: i64,
    include_usage: bool,
    tpp: ToolPostProcess,
    think: ThinkPostProcess,
    client: Option<ClientDeadline>,
) -> Response {
    let (tx_bytes, rx_bytes) = mpsc::channel::<Result<Bytes, std::io::Error>>(64);

    let mut g0 = gen.clone();
    g0.seed = Some(base_seed);
    let (tx_ev0, mut rx_ev0) = mpsc::channel::<ChatEvent>(64);
    if let Err(err) = engine.generate(g0, tx_ev0).await {
        warn!(error = %err, "chat engine.generate failed to start");
        return engine_start_error_response(&err);
    }

    let mut first_ev = match first_event(&mut rx_ev0, client).await {
        Ok(ev) => ev,
        Err(resp) => return resp,
    };

    let id_s = id.clone();
    let model_s = model.clone();
    let fp = system_fingerprint(&model);
    tokio::spawn(async move {
        let mut prompt_tokens: u32 = 0;
        let mut cached_tokens: Option<u32> = None;
        let mut total_completion_tokens: u32 = 0;
        let mut rx_first = Some(rx_ev0);

        for i in 0..n {
            let mut pre = first_ev.take();
            let mut rx_ev = match rx_first.take() {
                Some(rx) => rx,
                None => {
                    let mut g = gen.clone();
                    g.seed = Some(base_seed.wrapping_add(i as u64));
                    let (tx_ev, rx_ev) = mpsc::channel::<ChatEvent>(64);
                    if let Err(err) = engine.generate(g, tx_ev).await {
                        let body = json!({"error": {"message": format!("engine: {err}"), "type": kind::SERVER}});
                        let _ = send_sse_raw(&tx_bytes, &body.to_string()).await;
                        let _ = send_sse_raw(&tx_bytes, "[DONE]").await;
                        return;
                    }
                    rx_ev
                }
            };

            let first = ChatCompletionChunk {
                id: id_s.clone(),
                object: "chat.completion.chunk",
                created,
                model: model_s.clone(),
                system_fingerprint: Some(fp.clone()),
                usage: None,
                choices: vec![ChunkChoice {
                    index: i,
                    delta: Delta {
                        role: Some("assistant".into()),
                        ..Default::default()
                    },
                    finish_reason: None,
                    logprobs: None,
                }],
            };
            if send_sse_json(&tx_bytes, &first).await.is_err() {
                return;
            }

            let mut finish_reason: Option<String> = None;

            let mut buf = String::new();

            let pair_lp = gen.logprobs && !tpp.active && !think.active;
            let mut pending_content: Option<String> = None;
            let mut splitter = ThinkingStream::new(think.opened);
            #[allow(clippy::while_let_loop)]
            loop {
                let Some(ev) = (match pre.take() {
                    Some(ev) => Some(ev),
                    None => rx_ev.recv().await,
                }) else {
                    break;
                };
                match ev {
                    ChatEvent::Started { prompt_tokens: p } => prompt_tokens = p,
                    ChatEvent::PromptCached { cached_tokens: c } => cached_tokens = Some(c),
                    ChatEvent::StoppedBy { .. } => {}
                    ChatEvent::TextDelta(s) if tpp.active => buf.push_str(&s),
                    ChatEvent::TextDelta(s) if think.active => {
                        let (r, c) = splitter.push(&s);
                        if r.is_empty() && c.is_empty() {
                            continue;
                        }
                        let chunk = ChatCompletionChunk {
                            id: id_s.clone(),
                            object: "chat.completion.chunk",
                            created,
                            model: model_s.clone(),
                            system_fingerprint: Some(fp.clone()),
                            usage: None,
                            choices: vec![ChunkChoice {
                                index: i,
                                delta: Delta {
                                    content: (!c.is_empty()).then_some(c),
                                    reasoning_content: (!r.is_empty()).then_some(r),
                                    ..Default::default()
                                },
                                finish_reason: None,
                                logprobs: None,
                            }],
                        };
                        if send_sse_json(&tx_bytes, &chunk).await.is_err() {
                            return;
                        }
                    }
                    ChatEvent::TextDelta(s) if pair_lp => {
                        pending_content = Some(match pending_content.take() {
                            Some(mut p) => {
                                p.push_str(&s);
                                p
                            }
                            None => s,
                        });
                    }
                    ChatEvent::ReasoningDelta(s) => {
                        let chunk = ChatCompletionChunk {
                            id: id_s.clone(),
                            object: "chat.completion.chunk",
                            created,
                            model: model_s.clone(),
                            system_fingerprint: Some(fp.clone()),
                            usage: None,
                            choices: vec![ChunkChoice {
                                index: i,
                                delta: Delta {
                                    reasoning_content: Some(s),
                                    ..Default::default()
                                },
                                finish_reason: None,
                                logprobs: None,
                            }],
                        };
                        if send_sse_json(&tx_bytes, &chunk).await.is_err() {
                            return;
                        }
                    }
                    ChatEvent::TextDelta(s) => {
                        let chunk = ChatCompletionChunk {
                            id: id_s.clone(),
                            object: "chat.completion.chunk",
                            created,
                            model: model_s.clone(),
                            system_fingerprint: Some(fp.clone()),
                            usage: None,
                            choices: vec![ChunkChoice {
                                index: i,
                                delta: Delta {
                                    content: Some(s),
                                    ..Default::default()
                                },
                                finish_reason: None,
                                logprobs: None,
                            }],
                        };
                        if send_sse_json(&tx_bytes, &chunk).await.is_err() {
                            return;
                        }
                    }
                    ChatEvent::Logprob(e) => {
                        let chunk = ChatCompletionChunk {
                            id: id_s.clone(),
                            object: "chat.completion.chunk",
                            created,
                            model: model_s.clone(),
                            system_fingerprint: Some(fp.clone()),
                            usage: None,
                            choices: vec![ChunkChoice {
                                index: i,
                                delta: Delta {
                                    content: pending_content.take(),
                                    ..Default::default()
                                },
                                finish_reason: None,
                                logprobs: Some(LogprobsObject { content: vec![e] }),
                            }],
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
                        total_completion_tokens = total_completion_tokens.saturating_add(c);
                    }
                    ChatEvent::Error(msg) => {
                        let body = json!({"error": {"message": msg, "type": kind::SERVER}});
                        let _ = send_sse_raw(&tx_bytes, &body.to_string()).await;
                        let _ = send_sse_raw(&tx_bytes, "[DONE]").await;
                        return;
                    }
                }
            }

            if let Some(c) = pending_content.take() {
                let chunk = ChatCompletionChunk {
                    id: id_s.clone(),
                    object: "chat.completion.chunk",
                    created,
                    model: model_s.clone(),
                    system_fingerprint: Some(fp.clone()),
                    usage: None,
                    choices: vec![ChunkChoice {
                        index: i,
                        delta: Delta {
                            content: Some(c),
                            ..Default::default()
                        },
                        finish_reason: None,
                        logprobs: None,
                    }],
                };
                let _ = send_sse_json(&tx_bytes, &chunk).await;
            }

            if think.active && !tpp.active {
                let (r, c) = splitter.finish();
                if !r.is_empty() || !c.is_empty() {
                    let chunk = ChatCompletionChunk {
                        id: id_s.clone(),
                        object: "chat.completion.chunk",
                        created,
                        model: model_s.clone(),
                        system_fingerprint: Some(fp.clone()),
                        usage: None,
                        choices: vec![ChunkChoice {
                            index: i,
                            delta: Delta {
                                content: (!c.is_empty()).then_some(c),
                                reasoning_content: (!r.is_empty()).then_some(r),
                                ..Default::default()
                            },
                            finish_reason: None,
                            logprobs: None,
                        }],
                    };
                    let _ = send_sse_json(&tx_bytes, &chunk).await;
                }
            }

            if tpp.active {
                let mut reasoning: Option<String> = None;
                if think.active {
                    if let Some((r, c)) = split_thinking(&buf, think.opened) {
                        reasoning = (!r.is_empty()).then_some(r);
                        buf = c;
                    }
                }
                let parsed = parse_model_tool_calls(&buf, tpp.force_name.as_deref());
                let delta = if !parsed.tool_calls.is_empty() {
                    finish_reason = Some("tool_calls".into());

                    let calls: Vec<ToolCall> = parsed
                        .tool_calls
                        .into_iter()
                        .enumerate()
                        .map(|(idx, mut tc)| {
                            tc.index = Some(idx as u32);
                            tc
                        })
                        .collect();
                    Delta {
                        tool_calls: Some(calls),
                        reasoning_content: reasoning.take(),
                        ..Default::default()
                    }
                } else {
                    Delta {
                        content: parsed.content,
                        reasoning_content: reasoning.take(),
                        ..Default::default()
                    }
                };
                let chunk = ChatCompletionChunk {
                    id: id_s.clone(),
                    object: "chat.completion.chunk",
                    created,
                    model: model_s.clone(),
                    system_fingerprint: Some(fp.clone()),
                    usage: None,
                    choices: vec![ChunkChoice {
                        index: i,
                        delta,
                        finish_reason: None,
                        logprobs: None,
                    }],
                };
                if send_sse_json(&tx_bytes, &chunk).await.is_err() {
                    return;
                }
            }

            let last = ChatCompletionChunk {
                id: id_s.clone(),
                object: "chat.completion.chunk",
                created,
                model: model_s.clone(),
                system_fingerprint: Some(fp.clone()),
                usage: None,
                choices: vec![ChunkChoice {
                    index: i,
                    delta: Delta::default(),
                    finish_reason: Some(finish_reason.unwrap_or_else(|| "stop".into())),
                    logprobs: None,
                }],
            };
            if send_sse_json(&tx_bytes, &last).await.is_err() {
                return;
            }
        }

        if include_usage {
            let usage_chunk = ChatCompletionChunk {
                id: id_s.clone(),
                object: "chat.completion.chunk",
                created,
                model: model_s,
                system_fingerprint: Some(fp),
                usage: Some(Usage {
                    prompt_tokens,
                    completion_tokens: total_completion_tokens,
                    total_tokens: prompt_tokens + total_completion_tokens,
                    prompt_tokens_details: cached_tokens
                        .map(|cached_tokens| PromptTokensDetails { cached_tokens }),
                }),
                choices: Vec::new(),
            };
            if send_sse_json(&tx_bytes, &usage_chunk).await.is_err() {
                return;
            }
        }

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

pub(crate) async fn send_sse_json<T: Serialize>(
    tx: &mpsc::Sender<Result<Bytes, std::io::Error>>,
    v: &T,
) -> Result<(), ()> {
    let s = serde_json::to_string(v).map_err(|_| ())?;
    send_sse_raw(tx, &s).await
}

pub(crate) async fn send_sse_raw(
    tx: &mpsc::Sender<Result<Bytes, std::io::Error>>,
    body: &str,
) -> Result<(), ()> {
    let line = format!("data: {body}\n\n");
    tx.send(Ok(Bytes::from(line.into_bytes())))
        .await
        .map_err(|_| ())
}

pub fn render_chat_prompt(messages: &[ChatMessageIn]) -> String {
    let mut out = String::new();
    for m in messages {
        out.push_str("<|im_start|>");
        out.push_str(&m.role);
        out.push('\n');
        out.push_str(&m.text());
        out.push_str("<|im_end|>\n");
    }
    out.push_str("<|im_start|>assistant\n");
    out
}

pub fn system_fingerprint(model_id: &str) -> String {
    use std::hash::{Hash, Hasher};
    let mut h = std::collections::hash_map::DefaultHasher::new();
    model_id.hash(&mut h);
    env!("CARGO_PKG_VERSION").hash(&mut h);
    format!("fp_{:08x}", (h.finish() as u32))
}

pub(crate) fn now_unix_secs() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    async fn error_body(resp: Response) -> serde_json::Value {
        let bytes = axum::body::to_bytes(resp.into_body(), 64 * 1024)
            .await
            .expect("read error body");
        serde_json::from_slice(&bytes).expect("error body is json")
    }

    #[tokio::test]
    async fn engine_busy_is_a_503_with_the_engine_busy_code() {
        let err = anyhow::Error::new(EngineBusy::new(3, 3000));
        let resp = engine_start_error_response(&err);
        assert_eq!(resp.status(), StatusCode::SERVICE_UNAVAILABLE);
        let body = error_body(resp).await;
        assert_eq!(body["error"]["code"], "engine_busy");
        assert_eq!(body["error"]["type"], kind::SERVICE_UNAVAIL);
        let msg = body["error"]["message"].as_str().unwrap_or_default();
        assert!(
            msg.contains("3 concurrent"),
            "message lost the permit count: {msg}"
        );
        assert!(
            msg.contains("3000 ms"),
            "message lost the queue window: {msg}"
        );
    }

    fn deadline_header(v: &str) -> HeaderMap {
        let mut h = HeaderMap::new();
        h.insert(
            deadline::HEADER,
            axum::http::HeaderValue::from_str(v).unwrap(),
        );
        h
    }

    #[test]
    fn the_timeout_body_field_beats_the_header_and_both_beat_nothing() {
        let hdrs = deadline_header("9000");
        let body = ClientDeadline::from_request(Some(0.25), &hdrs).expect("body field wins");
        assert_eq!(body.budget_ms(), 250);
        assert_eq!(body.source, "timeout body field");

        let hdr = ClientDeadline::from_request(None, &hdrs).expect("header is the fallback");
        assert_eq!(hdr.budget_ms(), 9000);
        assert!(hdr.source.starts_with(deadline::HEADER));

        assert!(ClientDeadline::from_request(None, &HeaderMap::new()).is_none());
        assert!(
            ClientDeadline::from_request(None, &deadline_header("soon")).is_none(),
            "a malformed header must fall back to the server default, not 400 or collapse"
        );
    }

    #[test]
    fn a_caller_deadline_is_clamped_at_both_ends() {
        assert!(deadline::max_ms() < 99_999_999);
        let absurd =
            ClientDeadline::from_request(None, &deadline_header("99999999")).expect("parsed");
        assert_eq!(absurd.budget_ms(), deadline::max_ms());
        let zero = ClientDeadline::from_request(Some(0.0), &HeaderMap::new()).expect("parsed");
        assert_eq!(zero.budget_ms(), deadline::FLOOR_MS);
        let negative = ClientDeadline::from_request(Some(-5.0), &HeaderMap::new());
        assert!(negative.is_none(), "a negative timeout is not a deadline");
    }

    #[tokio::test]
    async fn first_event_sheds_at_the_caller_deadline_but_never_without_one() {
        let (tx, mut rx) = mpsc::channel::<ChatEvent>(4);
        let client = ClientDeadline::from_request(None, &deadline_header("80"));
        let t0 = std::time::Instant::now();
        let resp = first_event(&mut rx, client)
            .await
            .expect_err("nothing arrives, so the deadline must shed");
        let elapsed = t0.elapsed();
        assert_eq!(resp.status(), StatusCode::SERVICE_UNAVAILABLE);
        let body = error_body(resp).await;
        assert_eq!(body["error"]["code"], "surface_busy");
        assert!(
            elapsed < Duration::from_millis(1_000),
            "shed took {elapsed:?}, so the 80 ms budget was not used"
        );

        let (tx2, mut rx2) = mpsc::channel::<ChatEvent>(4);
        tx2.send(ChatEvent::Started { prompt_tokens: 7 })
            .await
            .unwrap();
        let peeked = first_event(&mut rx2, client)
            .await
            .expect("an event inside the budget is passed through, not shed");
        assert!(matches!(
            peeked,
            Some(ChatEvent::Started { prompt_tokens: 7 })
        ));

        let none = first_event(&mut rx, None)
            .await
            .expect("no caller deadline means no peek and no shed");
        assert!(none.is_none());
        drop(tx);
    }

    #[tokio::test]
    async fn first_event_turns_an_engine_side_shed_into_a_503_before_streaming_starts() {
        let (tx, mut rx) = mpsc::channel::<ChatEvent>(4);
        tx.send(ChatEvent::Error(format!("{}", EngineBusy::new(16, 3000))))
            .await
            .unwrap();
        let client = ClientDeadline::from_request(Some(5.0), &HeaderMap::new());
        let resp = first_event(&mut rx, client)
            .await
            .expect_err("a busy first event must abort with a response");
        assert_eq!(resp.status(), StatusCode::SERVICE_UNAVAILABLE);
        assert_eq!(error_body(resp).await["error"]["code"], "engine_busy");
    }

    #[tokio::test]
    async fn an_engine_event_shed_is_a_503_while_a_real_engine_error_stays_a_500() {
        let vram = format!(
            "{} request needs 2.14 GiB of VRAM headroom but 18.00 of 18.98 GiB is already in \
             flight across 8 request(s); waited 3001 ms; retry shortly: {}",
            crate::oapi::admission::REJECT_PREFIX,
            EngineBusy::new(16, 3001)
        );
        assert!(is_busy_shed(&vram));
        let resp = engine_event_error_response(vram);
        assert_eq!(resp.status(), StatusCode::SERVICE_UNAVAILABLE);
        let body = error_body(resp).await;
        assert_eq!(body["error"]["code"], "engine_busy");
        assert!(body["error"]["message"]
            .as_str()
            .unwrap_or_default()
            .contains("GiB"));

        let sem = format!("{}", EngineBusy::new(16, 3000));
        assert!(is_busy_shed(&sem));
        assert_eq!(
            engine_event_error_response(sem).status(),
            StatusCode::SERVICE_UNAVAILABLE
        );

        for real in [
            "cuda error: an illegal memory access was encountered",
            "tokenizer: unknown token",
            "engine stopped",
        ] {
            assert!(!is_busy_shed(real), "{real} must not read as a shed");
            let resp = engine_event_error_response(real.to_string());
            assert_eq!(
                resp.status(),
                StatusCode::INTERNAL_SERVER_ERROR,
                "{real} must stay a 500"
            );
            assert_eq!(error_body(resp).await["error"]["code"], "engine_error");
        }
    }

    #[tokio::test]
    async fn a_non_busy_start_failure_is_still_a_500() {
        let err = anyhow::anyhow!("cuda oom");
        let resp = engine_start_error_response(&err);
        assert_eq!(resp.status(), StatusCode::INTERNAL_SERVER_ERROR);
        let body = error_body(resp).await;
        assert_eq!(body["error"]["code"], "engine_unavailable");
    }

    #[test]
    fn spec_decode_header_covers_every_engine_state() {
        assert_eq!(spec_decode_header_value(Some("on")), "on");
        assert_eq!(spec_decode_header_value(Some("degraded")), "degraded");
        assert_eq!(spec_decode_header_value(None), "off");
        assert_eq!(spec_decode_header_value(Some("something-new")), "unknown");
    }

    #[test]
    fn spec_decode_header_is_a_valid_header_name_and_lands_on_a_response() {
        let mut resp = openai_error(
            StatusCode::BAD_REQUEST,
            "x",
            kind::INVALID_REQUEST,
            None,
            None,
        );
        set_spec_decode_header(&mut resp, "degraded");
        assert_eq!(
            resp.headers()
                .get(SPEC_DECODE_HEADER)
                .and_then(|v| v.to_str().ok()),
            Some("degraded")
        );
    }

    #[test]
    fn chat_message_in_text_handles_string_form() {
        let m = ChatMessageIn {
            role: "user".into(),
            content: Some(MessageContent::Text("hello".into())),
            ..Default::default()
        };
        assert_eq!(m.text(), "hello");
    }

    #[test]
    fn chat_message_in_text_concatenates_parts() {
        let m = ChatMessageIn {
            role: "user".into(),
            content: Some(MessageContent::Parts(vec![
                ContentPart::Text { text: "a".into() },
                ContentPart::Text { text: "b".into() },
            ])),
            ..Default::default()
        };
        assert_eq!(m.text(), "a\nb");
    }

    #[test]
    fn render_chat_prompt_uses_chatml() {
        let prompt = render_chat_prompt(&[
            ChatMessageIn {
                role: "system".into(),
                content: Some(MessageContent::Text("S".into())),
                ..Default::default()
            },
            ChatMessageIn {
                role: "user".into(),
                content: Some(MessageContent::Text("U".into())),
                ..Default::default()
            },
        ]);
        assert!(prompt.contains("<|im_start|>system\nS<|im_end|>"));
        assert!(prompt.contains("<|im_start|>user\nU<|im_end|>"));
        assert!(prompt.ends_with("<|im_start|>assistant\n"));
    }

    #[test]
    fn deserialize_minimal_request() {
        let body = r#"{"model":"x","messages":[{"role":"user","content":"hi"}]}"#;
        let r: ChatCompletionRequest = serde_json::from_str(body).unwrap();
        assert_eq!(r.messages.len(), 1);
        assert_eq!(r.messages[0].text(), "hi");
    }

    #[test]
    fn deserialize_stop_alternatives() {
        let one: ChatCompletionRequest =
            serde_json::from_str(r#"{"messages":[{"role":"user","content":"x"}],"stop":"END"}"#)
                .unwrap();
        assert!(matches!(one.stop, Some(StopField::One(_))));
        let many: ChatCompletionRequest = serde_json::from_str(
            r#"{"messages":[{"role":"user","content":"x"}],"stop":["A","B"]}"#,
        )
        .unwrap();
        assert!(matches!(many.stop, Some(StopField::Many(_))));
    }

    #[test]
    fn deserialize_content_parts_array_form() {
        let body = r#"{"messages":[{"role":"user","content":[{"type":"text","text":"hi"}]}]}"#;
        let r: ChatCompletionRequest = serde_json::from_str(body).unwrap();
        assert_eq!(r.messages[0].text(), "hi");
    }

    async fn body_json(resp: Response) -> serde_json::Value {
        let bytes = axum::body::to_bytes(resp.into_body(), 64 * 1024)
            .await
            .unwrap();
        serde_json::from_slice(&bytes).unwrap()
    }

    #[test]
    fn text_only_parts_are_not_rejected() {
        let body = r#"{"messages":[{"role":"user","content":[{"type":"text","text":"hi"}]}]}"#;
        let r: ChatCompletionRequest = serde_json::from_str(body).unwrap();
        assert!(first_unsupported_part(&r.messages).is_none());
    }

    #[test]
    fn image_and_audio_parts_are_flagged() {
        let img = r#"{"messages":[{"role":"user","content":[{"type":"text","text":"hi"},{"type":"image_url","image_url":{"url":"data:image/png;base64,AA=="}}]}]}"#;
        let r: ChatCompletionRequest = serde_json::from_str(img).unwrap();
        assert_eq!(first_unsupported_part(&r.messages), Some((0, "image_url")));

        let aud = r#"{"messages":[{"role":"user","content":"hi"},{"role":"user","content":[{"type":"input_audio","input_audio":{"data":"AA==","format":"wav"}}]}]}"#;
        let r: ChatCompletionRequest = serde_json::from_str(aud).unwrap();
        assert_eq!(
            first_unsupported_part(&r.messages),
            Some((1, "input_audio"))
        );
    }

    #[tokio::test]
    async fn image_part_is_rejected_with_400_naming_the_type() {
        let body = r#"{"messages":[{"role":"user","content":[{"type":"image_url","image_url":{"url":"data:image/png;base64,AA=="}}]}]}"#;
        let r: ChatCompletionRequest = serde_json::from_str(body).unwrap();
        let resp = reject_unsupported_parts(&r.messages).expect("image_url part must be rejected");
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
        let v = body_json(resp).await;
        assert_eq!(v["error"]["type"], "invalid_request_error");
        assert_eq!(v["error"]["code"], "unsupported_content_part");
        assert_eq!(v["error"]["param"], "messages[0].content");
        assert!(v["error"]["message"]
            .as_str()
            .unwrap()
            .contains("image_url"));
    }

    #[tokio::test]
    async fn malformed_json_uses_the_openai_error_envelope() {
        use axum::extract::{FromRequest, Request};
        let req = Request::builder()
            .method("POST")
            .uri("/v1/chat/completions")
            .header("content-type", "application/json")
            .body(Body::from("{\"messages\": ["))
            .unwrap();
        let resp = OaiJson::<ChatCompletionRequest>::from_request(req, &())
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
            .uri("/v1/chat/completions")
            .body(Body::from("{}"))
            .unwrap();
        let resp = OaiJson::<ChatCompletionRequest>::from_request(req, &())
            .await
            .err()
            .expect("a missing JSON content-type must be rejected");
        assert_eq!(resp.status(), StatusCode::UNSUPPORTED_MEDIA_TYPE);
        let v = body_json(resp).await;
        assert_eq!(v["error"]["type"], "invalid_request_error");
        assert_eq!(v["error"]["code"], "unsupported_content_type");
    }

    #[test]
    fn response_serializes_with_expected_shape() {
        let resp = ChatCompletionResponse {
            id: "x".into(),
            object: "chat.completion",
            created: 7,
            model: "m".into(),
            choices: vec![ChatChoice {
                index: 0,
                message: ChatMessageOut {
                    role: "assistant".into(),
                    content: Some("hello".into()),
                    reasoning_content: None,
                    tool_calls: None,
                },
                finish_reason: "stop".into(),
                logprobs: None,
            }],
            usage: Usage {
                prompt_tokens: 1,
                completion_tokens: 1,
                total_tokens: 2,
                prompt_tokens_details: None,
            },
            system_fingerprint: Some("fp_00000000".into()),
        };
        let v = serde_json::to_value(&resp).unwrap();
        assert_eq!(v["object"], "chat.completion");
        assert_eq!(v["choices"][0]["message"]["content"], "hello");
        assert_eq!(v["usage"]["total_tokens"], 2);
        assert!(v["choices"][0]["message"]
            .get("reasoning_content")
            .is_none());
    }

    #[test]
    fn native_tool_call_delimiters_parse_into_a_tool_call() {
        let raw = "<|tool_call>call:get_weather{city:<|\"|>Oslo<|\"|>}<tool_call|>";
        let p = parse_model_tool_calls(raw, None);
        assert_eq!(p.tool_calls.len(), 1);
        assert_eq!(p.tool_calls[0].function.name, "get_weather");
        assert_eq!(p.tool_calls[0].function.arguments, r#"{"city":"Oslo"}"#);
    }

    #[test]
    fn native_tool_call_tolerates_whitespace_after_the_colon() {
        let raw =
            "<|tool_call>call:get_weather{<|\"|>city<|\"|>: <|\"|>Oslo<|\"|>, days: 3}<tool_call|>";
        let p = parse_model_tool_calls(raw, None);
        assert_eq!(p.tool_calls.len(), 1);
        assert_eq!(
            p.tool_calls[0].function.arguments,
            r#"{"city":"Oslo","days":3}"#
        );
    }

    #[test]
    fn native_tool_call_string_may_contain_structure() {
        let raw = "<|tool_call>call:f{q:<|\"|>a,b}c{d<|\"|>,n:2}<tool_call|>";
        let p = parse_model_tool_calls(raw, None);
        assert_eq!(p.tool_calls.len(), 1);
        let v: serde_json::Value =
            serde_json::from_str(&p.tool_calls[0].function.arguments).unwrap();
        assert_eq!(v["q"], "a,b}c{d");
        assert_eq!(v["n"], 2);
    }

    #[test]
    fn native_tool_call_nested_and_typed_arguments() {
        let raw = "<|tool_call>call:f{flag:true,ratio:1.5,tags:[<|\"|>a<|\"|>,<|\"|>b<|\"|>],\
                   where:{lat:1.5,lon:-2}}<tool_call|>";
        let p = parse_model_tool_calls(raw, None);
        assert_eq!(p.tool_calls.len(), 1);
        let v: serde_json::Value =
            serde_json::from_str(&p.tool_calls[0].function.arguments).unwrap();
        assert_eq!(v["flag"], true);
        assert_eq!(v["ratio"], 1.5);
        assert_eq!(v["tags"][1], "b");
        assert_eq!(v["where"]["lon"], -2);
    }

    #[test]
    fn native_tool_call_two_calls_in_one_message() {
        let raw = "<|tool_call>call:a{x:1}<tool_call|><|tool_call>call:b{y:2}<tool_call|>";
        let p = parse_model_tool_calls(raw, None);
        assert_eq!(p.tool_calls.len(), 2);
        assert_eq!(p.tool_calls[0].function.name, "a");
        assert_eq!(p.tool_calls[1].function.name, "b");
    }

    #[test]
    fn native_tool_call_empty_arguments() {
        let p = parse_model_tool_calls("<|tool_call>call:now{}<tool_call|>", None);
        assert_eq!(p.tool_calls.len(), 1);
        assert_eq!(p.tool_calls[0].function.arguments, "{}");
    }

    #[test]
    fn native_tool_call_truncated_block_is_left_as_text() {
        let raw = "<|tool_call>call:f{x:1";
        let p = parse_model_tool_calls(raw, None);
        assert!(p.tool_calls.is_empty());
        assert_eq!(p.content.as_deref(), Some(raw));
    }

    #[test]
    fn a_call_is_unrecoverable_once_the_special_delimiters_are_stripped() {
        let stripped = "call:get_weather{city:Oslo,days:3}";
        let p = parse_model_tool_calls(stripped, None);
        assert!(
            p.tool_calls.is_empty(),
            "if this ever parses, the delimiters stopped being load-bearing and \
             this test no longer pins the bug"
        );
        assert_eq!(p.content.as_deref(), Some(stripped));
    }

    #[test]
    fn thinking_splits_on_the_close_tag_when_the_prompt_opened_it() {
        let (r, c) = split_thinking("weighing options\n</think>Four.", true).unwrap();
        assert_eq!(r, "weighing options");
        assert_eq!(c, "Four.");
    }

    #[test]
    fn thinking_splits_when_the_model_emits_the_open_tag_itself() {
        let (r, c) = split_thinking("<think>hmm</think>ok", false).unwrap();
        assert_eq!(r, "hmm");
        assert_eq!(c, "ok");
    }

    #[test]
    fn no_open_tag_and_no_close_tag_leaves_content_untouched() {
        assert!(split_thinking("just an answer", false).is_none());
    }

    #[test]
    fn unclosed_block_opened_by_the_prompt_is_all_reasoning() {
        let (r, c) = split_thinking("still thinking when we ran out of tokens", true).unwrap();
        assert_eq!(r, "still thinking when we ran out of tokens");
        assert!(c.is_empty());
    }

    #[test]
    fn unclosed_block_opened_by_the_model_is_all_reasoning() {
        let (r, c) = split_thinking("<think>hmm, budget gone", false).unwrap();
        assert_eq!(r, "hmm, budget gone");
        assert!(c.is_empty());
    }

    #[test]
    fn think_only_output_yields_empty_content() {
        let (r, c) = split_thinking("hmm</think>", true).unwrap();
        assert_eq!(r, "hmm");
        assert!(c.is_empty());
        let (r, c) = split_thinking("<think>hmm</think>", false).unwrap();
        assert_eq!(r, "hmm");
        assert!(c.is_empty());
    }

    #[test]
    fn streamed_content_after_the_close_tag_drops_the_leading_whitespace() {
        let mut s = ThinkingStream::new(true);
        let (r, c) = drive(&mut s, &["weighing options\n</think>", "\n", "\nFour."]);
        assert_eq!(r, "weighing options\n");
        assert_eq!(c, "Four.");
        let (_, c_split) = split_thinking("weighing options\n</think>\n\nFour.", true).unwrap();
        assert_eq!(c, c_split);
    }

    #[test]
    fn unclosed_split_matches_the_streaming_splitter() {
        for (text, opened) in [
            ("still thinking when we ran out of tokens", true),
            ("<think>hmm, budget gone", false),
        ] {
            let (r, c) = split_thinking(text, opened).unwrap();
            let mut s = ThinkingStream::new(opened);
            let (sr, sc) = drive(&mut s, &[text]);
            assert_eq!(r, sr.trim());
            assert_eq!(c, sc);
        }
    }

    #[test]
    fn a_close_tag_in_ordinary_content_does_not_split() {
        let text = "here is markup: ```\n</think>\n```";
        assert!(split_thinking(text, false).is_none());
    }

    #[test]
    fn only_the_first_close_tag_splits() {
        let (r, c) = split_thinking("a</think>b</think>c", true).unwrap();
        assert_eq!(r, "a");
        assert_eq!(c, "b</think>c");
    }

    fn drive(s: &mut ThinkingStream, pieces: &[&str]) -> (String, String) {
        let mut r = String::new();
        let mut c = String::new();
        for p in pieces {
            let (dr, dc) = s.push(p);
            r.push_str(&dr);
            c.push_str(&dc);
        }
        let (fr, fc) = s.finish();
        r.push_str(&fr);
        c.push_str(&fc);
        (r, c)
    }

    #[test]
    fn streaming_splitter_handles_a_close_tag_split_across_deltas() {
        let mut s = ThinkingStream::new(true);
        let (r, c) = drive(&mut s, &["weigh", "ing</th", "ink>Fo", "ur."]);
        assert_eq!(r, "weighing");
        assert_eq!(c, "Four.");
    }

    #[test]
    fn streaming_splitter_labels_everything_content_when_no_block_opens() {
        let mut s = ThinkingStream::new(false);
        let (r, c) = drive(&mut s, &["Fo", "ur.", " </think> is markup"]);
        assert!(r.is_empty());
        assert_eq!(c, "Four. </think> is markup");
    }

    #[test]
    fn streaming_splitter_detects_a_model_emitted_open_tag() {
        let mut s = ThinkingStream::new(false);
        let (r, c) = drive(&mut s, &["<thi", "nk>hmm</think>ok"]);
        assert_eq!(r, "hmm");
        assert_eq!(c, "ok");
    }

    #[test]
    fn streaming_splitter_keeps_unclosed_reasoning_out_of_content() {
        let mut s = ThinkingStream::new(true);
        let (r, c) = drive(&mut s, &["still thinking when we ran out of tokens"]);
        assert_eq!(r, "still thinking when we ran out of tokens");
        assert!(c.is_empty());
    }

    fn req_with(kwargs: serde_json::Value) -> ChatCompletionRequest {
        let body = json!({
            "messages": [{"role": "user", "content": "hi"}],
            "chat_template_kwargs": kwargs,
        });
        serde_json::from_value(body).unwrap()
    }

    #[test]
    fn chat_template_kwargs_are_parsed_and_validated() {
        let r = req_with(json!({"enable_thinking": false}));
        let kw = request_template_kwargs(&r).unwrap();
        assert_eq!(kw.get("enable_thinking"), Some(&json!(false)));

        let bad = req_with(json!([1, 2]));
        let err = request_template_kwargs(&bad).expect_err("array must be rejected");
        assert_eq!(err.status(), StatusCode::BAD_REQUEST);
    }

    #[test]
    fn chat_template_kwargs_win_over_the_enable_thinking_shorthand() {
        let body = json!({
            "messages": [{"role": "user", "content": "hi"}],
            "enable_thinking": true,
            "chat_template_kwargs": {"enable_thinking": false},
        });
        let r: ChatCompletionRequest = serde_json::from_value(body).unwrap();
        let kw = request_template_kwargs(&r).unwrap();
        assert_eq!(kw.get("enable_thinking"), Some(&json!(false)));
    }

    #[test]
    fn enable_thinking_shorthand_alone_reaches_the_template_kwargs() {
        let body = json!({
            "messages": [{"role": "user", "content": "hi"}],
            "enable_thinking": false,
        });
        let r: ChatCompletionRequest = serde_json::from_value(body).unwrap();
        let kw = request_template_kwargs(&r).unwrap();
        assert_eq!(kw.get("enable_thinking"), Some(&json!(false)));
    }

    #[test]
    fn a_request_without_kwargs_renders_exactly_as_before() {
        let body = json!({"messages": [{"role": "user", "content": "hi"}]});
        let r: ChatCompletionRequest = serde_json::from_value(body).unwrap();
        assert!(request_template_kwargs(&r).unwrap().is_empty());
    }

    #[test]
    fn the_reasoning_effort_field_reaches_the_template_kwargs_and_invalid_values_are_400s() {
        let body = json!({
            "messages": [{"role": "user", "content": "hi"}],
            "reasoning_effort": "xhigh",
        });
        let r: ChatCompletionRequest = serde_json::from_value(body).unwrap();
        let kw = request_template_kwargs(&r).unwrap();
        assert_eq!(kw.get("reasoning_effort"), Some(&json!("xhigh")));

        let bad = json!({
            "messages": [{"role": "user", "content": "hi"}],
            "reasoning_effort": "max",
        });
        let r: ChatCompletionRequest = serde_json::from_value(bad).unwrap();
        let err = request_template_kwargs(&r).expect_err(
            "an effort the template would raise_exception on must die at the request boundary, \
             not fall back to the built-in renderer mid-request",
        );
        assert_eq!(err.status(), StatusCode::BAD_REQUEST);
    }

    #[test]
    fn chat_template_kwargs_win_over_the_reasoning_effort_shorthand() {
        let body = json!({
            "messages": [{"role": "user", "content": "hi"}],
            "reasoning_effort": "low",
            "chat_template_kwargs": {"reasoning_effort": "xhigh"},
        });
        let r: ChatCompletionRequest = serde_json::from_value(body).unwrap();
        let kw = request_template_kwargs(&r).unwrap();
        assert_eq!(kw.get("reasoning_effort"), Some(&json!("xhigh")));
    }

    fn thinking_template(tag: &str) -> std::sync::Arc<crate::oapi::chat_template::ChatTemplate> {
        let mut d = std::env::temp_dir();
        d.push(format!("chatlane-{tag}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d).unwrap();
        std::fs::write(
            d.join("chat_template.jinja"),
            "{%- set enable_thinking = enable_thinking | default(false) -%}\
             {{ messages[0]['content'] }}{% if enable_thinking %}<think>{% endif %}",
        )
        .unwrap();
        std::fs::write(
            d.join("tokenizer_config.json"),
            r#"{"bos_token":"","eos_token":""}"#,
        )
        .unwrap();
        std::fs::write(
            d.join("generation_config.json"),
            r#"{"default_chat_template_kwargs":{"enable_thinking":true}}"#,
        )
        .unwrap();
        crate::oapi::chat_template::ChatTemplate::load(&d).expect("load")
    }

    #[test]
    fn request_kwargs_override_the_generation_config_default() {
        let t = thinking_template("kwargs-precedence");
        assert_eq!(
            t.default_template_kwargs().get("enable_thinking"),
            Some(&json!(true))
        );
        let mut extra = TemplateKwargs::new();
        extra.insert("enable_thinking".into(), json!(false));
        assert_eq!(
            merged_template_kwargs(&t, &extra).get("enable_thinking"),
            Some(&json!(false))
        );
        assert!(merged_template_kwargs(&t, &TemplateKwargs::new())
            .get("enable_thinking")
            .unwrap()
            .as_bool()
            .unwrap());

        let msgs = &[ChatMessageIn {
            role: "user".into(),
            content: Some(MessageContent::Text("hi".into())),
            ..Default::default()
        }];
        let on = render_official(&t, msgs, &[], &ToolChoice::Auto).unwrap();
        let off = render_official_with_kwargs(&t, msgs, &[], &ToolChoice::Auto, &extra).unwrap();
        assert!(on.ends_with(THINK_OPEN), "{on:?}");
        assert!(!off.ends_with(THINK_OPEN), "{off:?}");
    }
}
