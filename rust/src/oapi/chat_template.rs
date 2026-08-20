use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, OnceLock};

use minijinja::{Environment, Value as JinjaValue};
use serde_json::Value;

pub const TEMPLATE_KWARGS_ENV: &str = "NV_CHAT_TEMPLATE_KWARGS";

pub const REASONING_EFFORT_KWARG: &str = "reasoning_effort";

pub const Q38_REASONING_EFFORT_ENV: &str = "NV_Q38_REASONING_EFFORT";

pub const REASONING_EFFORT_VALUES_THE_QWEN38_TEMPLATE_ACCEPTS_WITH_HIGH_ALIASED_TO_XHIGH:
    [&str; 4] = ["low", "medium", "high", "xhigh"];

pub const SERVED_DEFAULT_REASONING_EFFORT_MEDIUM_NOT_THE_TEMPLATES_XHIGH_BECAUSE_A_DEFAULT_XHIGH_IS_WILLISON_OVERTHINKING_LONG_THOUGHT_SPIRALS_ON_TRIVIAL_TURNS_SO_XHIGH_STAYS_ONE_EXPLICIT_REQUEST_AWAY:
    &str = "medium";

pub fn validate_reasoning_effort(v: &str) -> Result<(), String> {
    if REASONING_EFFORT_VALUES_THE_QWEN38_TEMPLATE_ACCEPTS_WITH_HIGH_ALIASED_TO_XHIGH.contains(&v)
    {
        Ok(())
    } else {
        Err(format!(
            "reasoning_effort must be one of low|medium|high|xhigh (high aliases to xhigh), \
             got {v:?}"
        ))
    }
}

fn q38_reasoning_effort_env_override_validated_or_logged_and_ignored() -> Option<String> {
    let raw = std::env::var(Q38_REASONING_EFFORT_ENV).ok()?;
    match validate_reasoning_effort(&raw) {
        Ok(()) => Some(raw),
        Err(e) => {
            tracing::error!(error = %e, "ignoring {Q38_REASONING_EFFORT_ENV}");
            None
        }
    }
}

const MARKER_PROBE_TEXT: &str = "probe";

const MARKER_PROBE_REASONING: &str = "nvthinkprobereasoning";

pub struct ChatTemplate {
    env: Environment<'static>,
    bos_token: String,
    eos_token: String,
    source: String,
    default_kwargs: BTreeMap<String, Value>,
    tools_supported: OnceLock<bool>,
    thinking_close: OnceLock<Option<String>>,
}

#[derive(Clone, Debug)]
pub struct LoadAttempt {
    pub dir: PathBuf,
    pub error: Option<String>,
}

fn attempts() -> &'static Mutex<Vec<LoadAttempt>> {
    static ATTEMPTS: OnceLock<Mutex<Vec<LoadAttempt>>> = OnceLock::new();
    ATTEMPTS.get_or_init(|| Mutex::new(Vec::new()))
}

fn record_attempt(dir: &Path, error: Option<String>) {
    if let Ok(mut v) = attempts().lock() {
        v.retain(|a| a.dir != dir);
        v.push(LoadAttempt {
            dir: dir.to_path_buf(),
            error,
        });
    }
}

pub fn load_attempt_for(model_key: &str) -> Option<LoadAttempt> {
    let v = attempts().lock().ok()?;
    v.iter()
        .rev()
        .find(|a| {
            a.dir.file_name().map(|f| f == model_key).unwrap_or(false)
                || a.dir.to_string_lossy() == model_key
        })
        .cloned()
}

pub fn load_was_attempted(model_key: &str) -> bool {
    load_attempt_for(model_key).is_some()
}

pub fn forget_load_attempts() {
    if let Ok(mut v) = attempts().lock() {
        v.clear();
    }
}

impl ChatTemplate {
    pub fn load(model_dir: &Path) -> Option<Arc<ChatTemplate>> {
        match Self::load_reason(model_dir) {
            Ok(t) => Some(t),
            Err(reason) => {
                tracing::error!(
                    model_dir = %model_dir.display(),
                    reason = %reason,
                    "chat template unavailable; prompts will use the built-in renderer"
                );
                None
            }
        }
    }

    pub fn load_reason(model_dir: &Path) -> Result<Arc<ChatTemplate>, String> {
        let loaded = Self::load_inner(model_dir);
        record_attempt(model_dir, loaded.as_ref().err().cloned());
        loaded
    }

    fn load_inner(model_dir: &Path) -> Result<Arc<ChatTemplate>, String> {
        let source = read_template_source(model_dir).ok_or_else(|| {
            format!(
                "neither chat_template.jinja nor tokenizer_config.json:chat_template found in {}",
                model_dir.display()
            )
        })?;
        let (bos_token, eos_token) = read_special_tokens(model_dir);
        let default_kwargs = read_default_chat_template_kwargs(model_dir);
        let mut env = Environment::new();
        env.set_unknown_method_callback(minijinja_contrib::pycompat::unknown_method_callback);
        env.add_function("raise_exception", raise_exception);
        env.add_function("strftime_now", strftime_now);
        let source = strip_generation_tags(&source);
        env.add_template_owned("chat", source.clone())
            .map_err(|e| format!("chat template does not compile under minijinja: {e:#}"))?;
        env.get_template("chat")
            .map_err(|e| format!("compiled chat template did not load: {e:#}"))?;
        Ok(Arc::new(ChatTemplate {
            env,
            bos_token,
            eos_token,
            source,
            default_kwargs,
            tools_supported: OnceLock::new(),
            thinking_close: OnceLock::new(),
        }))
    }

    pub fn default_template_kwargs(&self) -> &BTreeMap<String, Value> {
        &self.default_kwargs
    }

    pub fn effective_template_kwargs(&self) -> BTreeMap<String, Value> {
        let mut out = self.default_kwargs.clone();
        if self.declares_reasoning_effort() {
            out.insert(
                REASONING_EFFORT_KWARG.to_string(),
                Value::String(SERVED_DEFAULT_REASONING_EFFORT_MEDIUM_NOT_THE_TEMPLATES_XHIGH_BECAUSE_A_DEFAULT_XHIGH_IS_WILLISON_OVERTHINKING_LONG_THOUGHT_SPIRALS_ON_TRIVIAL_TURNS_SO_XHIGH_STAYS_ONE_EXPLICIT_REQUEST_AWAY.to_string()),
            );
        }
        for (k, v) in env_template_kwargs() {
            out.insert(k, v);
        }
        if self.declares_reasoning_effort() {
            if let Some(e) = q38_reasoning_effort_env_override_validated_or_logged_and_ignored() {
                out.insert(REASONING_EFFORT_KWARG.to_string(), Value::String(e));
            }
        }
        out
    }

    pub fn declares_thinking_switch(&self) -> bool {
        self.source.contains("enable_thinking")
    }

    pub fn declares_reasoning_effort(&self) -> bool {
        self.source.contains(REASONING_EFFORT_KWARG)
    }

    pub fn thinking_on_when_the_switch_is_undefined_scoped_to_reasoning_effort_templates_so_qwen36_guided_defaults_are_untouched(
        &self,
    ) -> Option<bool> {
        if !self.declares_reasoning_effort() || !self.declares_thinking_switch() {
            return None;
        }
        let kwargs = self.effective_template_kwargs();
        if let Some(v) = kwargs.get("enable_thinking").and_then(|v| v.as_bool()) {
            return Some(v);
        }
        let probe = serde_json::json!([{"role": "user", "content": MARKER_PROBE_TEXT}]);
        let undefined_arm = self.render_with_kwargs(&probe, None, true, &kwargs).ok()?;
        let false_arm = self
            .render_with_kwargs(&probe, None, true, &self.thinking_kwargs(false))
            .ok()?;
        Some(undefined_arm != false_arm)
    }

    pub fn thinking_close_marker(&self) -> Option<String> {
        self.thinking_close
            .get_or_init(|| {
                if !self.declares_thinking_switch() {
                    return None;
                }
                self.marker_from_generation_prompt()
                    .or_else(|| self.marker_from_closed_thought())
                    .filter(|m| m != &self.bos_token && m != &self.eos_token)
            })
            .clone()
    }

    fn thinking_kwargs(&self, enabled: bool) -> BTreeMap<String, Value> {
        let mut kwargs = self.effective_template_kwargs();
        kwargs.insert("enable_thinking".to_string(), Value::Bool(enabled));
        kwargs
    }

    fn generation_prompt(&self, messages: &Value, enabled: bool) -> Option<String> {
        let kwargs = self.thinking_kwargs(enabled);
        let with = self
            .render_with_kwargs(messages, None, true, &kwargs)
            .ok()?;
        let without = self
            .render_with_kwargs(messages, None, false, &kwargs)
            .ok()?;
        with.strip_prefix(&without).map(str::to_string)
    }

    fn marker_from_generation_prompt(&self) -> Option<String> {
        let probe = serde_json::json!([{"role": "user", "content": MARKER_PROBE_TEXT}]);
        let on = self.generation_prompt(&probe, true)?;
        let off = self.generation_prompt(&probe, false)?;
        close_marker_between_thinking_on_and_off_prompts(&on, &off)
    }

    fn marker_from_closed_thought(&self) -> Option<String> {
        let probe = serde_json::json!([
            {"role": "user", "content": MARKER_PROBE_TEXT},
            {
                "role": "assistant",
                "content": MARKER_PROBE_TEXT,
                "reasoning": MARKER_PROBE_REASONING,
                "reasoning_content": MARKER_PROBE_REASONING,
                "tool_calls": [{
                    "id": MARKER_PROBE_TEXT,
                    "type": "function",
                    "function": {"name": MARKER_PROBE_TEXT, "arguments": {}}
                }]
            }
        ]);
        let rendered = self
            .render_with_kwargs(&probe, None, false, &self.thinking_kwargs(true))
            .ok()?;
        leading_tag(rendered.split_once(MARKER_PROBE_REASONING)?.1)
    }

    pub fn uses_tool_responses(&self) -> bool {
        self.source.contains("tool_responses")
    }

    pub fn supports_tools(&self) -> bool {
        *self.tools_supported.get_or_init(|| {
            let probe = serde_json::json!([{"role": "user", "content": "probe"}]);
            let tools = serde_json::json!([{
                "type": "function",
                "function": {
                    "name": "nv_probe_tool_support",
                    "description": "probe",
                    "parameters": {"type": "object", "properties": {}, "required": []}
                }
            }]);
            match (
                self.render(&probe, Some(&tools), true),
                self.render(&probe, None, true),
            ) {
                (Ok(with), Ok(without)) => with != without,
                _ => false,
            }
        })
    }

    pub fn render(
        &self,
        messages: &Value,
        tools: Option<&Value>,
        add_generation_prompt: bool,
    ) -> anyhow::Result<String> {
        self.render_with_kwargs(
            messages,
            tools,
            add_generation_prompt,
            &self.effective_template_kwargs(),
        )
    }

    pub fn render_with_kwargs(
        &self,
        messages: &Value,
        tools: Option<&Value>,
        add_generation_prompt: bool,
        kwargs: &BTreeMap<String, Value>,
    ) -> anyhow::Result<String> {
        let tmpl = self
            .env
            .get_template("chat")
            .map_err(|e| anyhow::anyhow!("get chat template: {e}"))?;
        let tools_val = match tools {
            Some(t) => JinjaValue::from_serialize(t),
            None => JinjaValue::from(()),
        };
        let mut ctx: BTreeMap<String, JinjaValue> = BTreeMap::new();
        for (k, v) in kwargs {
            ctx.insert(k.clone(), JinjaValue::from_serialize(v));
        }
        let msgs_val = match tool_call_arguments_parsed_from_openai_wire_strings(messages) {
            Some(normalized) => JinjaValue::from_serialize(&normalized),
            None => JinjaValue::from_serialize(messages),
        };
        ctx.insert("messages".into(), msgs_val);
        ctx.insert("tools".into(), tools_val);
        ctx.insert(
            "add_generation_prompt".into(),
            JinjaValue::from(add_generation_prompt),
        );
        ctx.insert("bos_token".into(), JinjaValue::from(self.bos_token.clone()));
        ctx.insert("eos_token".into(), JinjaValue::from(self.eos_token.clone()));
        tmpl.render(ctx)
            .map_err(|e| anyhow::anyhow!("render chat template: {e:#}"))
    }
}

fn tool_call_arguments_parsed_from_openai_wire_strings(messages: &Value) -> Option<Value> {
    let needs_parse = messages.as_array()?.iter().any(|m| {
        m.get("tool_calls")
            .and_then(Value::as_array)
            .is_some_and(|calls| {
                calls
                    .iter()
                    .any(|c| c.pointer("/function/arguments").is_some_and(Value::is_string))
            })
    });
    if !needs_parse {
        return None;
    }
    let mut out = messages.clone();
    for m in out.as_array_mut()? {
        let Some(calls) = m.get_mut("tool_calls").and_then(Value::as_array_mut) else {
            continue;
        };
        for c in calls {
            let Some(args) = c.pointer_mut("/function/arguments") else {
                continue;
            };
            let Some(raw) = args.as_str() else {
                continue;
            };
            let parsed = if raw.trim().is_empty() {
                Value::Object(Default::default())
            } else {
                match serde_json::from_str::<Value>(raw) {
                    Ok(v @ Value::Object(_)) => v,
                    _ => continue,
                }
            };
            *args = parsed;
        }
    }
    Some(out)
}

fn close_marker_between_thinking_on_and_off_prompts(on: &str, off: &str) -> Option<String> {
    trailing_tag(off.get(tag_aligned_divergence(on, off)..)?).filter(|m| !on.contains(m))
}

fn tag_aligned_divergence(on: &str, off: &str) -> usize {
    let mut n = on
        .as_bytes()
        .iter()
        .zip(off.as_bytes())
        .take_while(|(a, b)| a == b)
        .count();
    while n > 0 && !off.is_char_boundary(n) {
        n -= 1;
    }
    match off[..n].rfind('<') {
        Some(open)
            if !off[open..n].contains('>') && !off[open..n].contains(char::is_whitespace) =>
        {
            open
        }
        _ => n,
    }
}

fn is_lone_tag(tag: &str) -> bool {
    tag.len() > 2
        && tag.starts_with('<')
        && tag.ends_with('>')
        && !tag.contains(char::is_whitespace)
        && !tag[1..tag.len() - 1].contains(['<', '>'])
}

fn leading_tag(text: &str) -> Option<String> {
    let text = text.trim_start();
    let end = text.find('>')? + 1;
    Some(text[..end].to_string()).filter(|t| is_lone_tag(t))
}

fn trailing_tag(text: &str) -> Option<String> {
    let text = text.trim_end();
    let open = text.rfind('<')?;
    Some(text[open..].to_string()).filter(|t| is_lone_tag(t))
}

pub fn parse_template_kwargs(raw: &str) -> Result<BTreeMap<String, Value>, String> {
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        return Ok(BTreeMap::new());
    }
    let v: Value = serde_json::from_str(trimmed)
        .map_err(|e| format!("{TEMPLATE_KWARGS_ENV} is not valid JSON: {e}"))?;
    match v {
        Value::Object(map) => Ok(map.into_iter().collect()),
        other => Err(format!(
            "{TEMPLATE_KWARGS_ENV} must be a JSON object, got {other}"
        )),
    }
}

fn env_template_kwargs() -> BTreeMap<String, Value> {
    let Ok(raw) = std::env::var(TEMPLATE_KWARGS_ENV) else {
        return BTreeMap::new();
    };
    match parse_template_kwargs(&raw) {
        Ok(m) => m,
        Err(e) => {
            tracing::error!(error = %e, "ignoring {TEMPLATE_KWARGS_ENV}");
            BTreeMap::new()
        }
    }
}

fn read_default_chat_template_kwargs(model_dir: &Path) -> BTreeMap<String, Value> {
    let mut out = BTreeMap::new();
    for name in ["generation_config.json", "tokenizer_config.json"] {
        let Ok(raw) = std::fs::read_to_string(model_dir.join(name)) else {
            continue;
        };
        let Ok(cfg) = serde_json::from_str::<Value>(&raw) else {
            continue;
        };
        if let Some(Value::Object(map)) = cfg.get("default_chat_template_kwargs") {
            for (k, v) in map {
                out.entry(k.clone()).or_insert_with(|| v.clone());
            }
        }
    }
    out
}

pub fn strip_generation_tags(source: &str) -> String {
    let mut out = String::with_capacity(source.len());
    let mut rest = source;
    while let Some(open) = rest.find("{%") {
        let (before, tail) = rest.split_at(open);
        out.push_str(before);
        let Some(close) = tail.find("%}") else {
            out.push_str(tail);
            return out;
        };
        let tag = &tail[..close + 2];
        let body = tag[2..tag.len() - 2].trim_matches('-').trim();
        if body == "generation" || body == "endgeneration" {
            let lead = if tag.starts_with("{%-") { "{#-" } else { "{#" };
            let trail = if tag.ends_with("-%}") { "-#}" } else { "#}" };
            out.push_str(lead);
            out.push(' ');
            out.push_str(trail);
        } else {
            out.push_str(tag);
        }
        rest = &tail[close + 2..];
    }
    out.push_str(rest);
    out
}

fn read_template_source(model_dir: &Path) -> Option<String> {
    let jinja = model_dir.join("chat_template.jinja");
    if let Ok(s) = std::fs::read_to_string(&jinja) {
        if !s.trim().is_empty() {
            return Some(s);
        }
    }

    for name in TEMPLATE_JSON_FILES {
        let raw = match std::fs::read_to_string(model_dir.join(name)) {
            Ok(r) => r,
            Err(_) => continue,
        };
        let Ok(cfg) = serde_json::from_str::<Value>(&raw) else {
            continue;
        };
        if let Some(t) = chat_template_value(cfg.get("chat_template")) {
            return Some(t);
        }
    }
    None
}

const TEMPLATE_JSON_FILES: [&str; 2] = ["tokenizer_config.json", "chat_template.json"];

fn chat_template_value(v: Option<&Value>) -> Option<String> {
    match v? {
        Value::String(s) if !s.trim().is_empty() => Some(s.clone()),
        Value::Array(arr) => {
            let mut chosen = None;
            for entry in arr {
                let name = entry.get("name").and_then(|n| n.as_str());
                let tmpl = entry.get("template").and_then(|t| t.as_str());
                if let Some(t) = tmpl {
                    if name == Some("default") {
                        return Some(t.to_string());
                    }
                    chosen.get_or_insert_with(|| t.to_string());
                }
            }
            chosen
        }
        _ => None,
    }
}

fn token_str(v: &Value) -> Option<String> {
    match v {
        Value::String(s) => Some(s.clone()),
        Value::Object(o) => o.get("content").and_then(|c| c.as_str()).map(String::from),
        _ => None,
    }
}

fn read_special_tokens(model_dir: &Path) -> (String, String) {
    let cfg_path = model_dir.join("tokenizer_config.json");
    let mut bos = String::new();
    let mut eos = String::new();
    if let Ok(raw) = std::fs::read_to_string(&cfg_path) {
        if let Ok(cfg) = serde_json::from_str::<Value>(&raw) {
            if let Some(b) = cfg.get("bos_token").and_then(token_str) {
                bos = b;
            }
            if let Some(e) = cfg.get("eos_token").and_then(token_str) {
                eos = e;
            }
        }
    }
    (bos, eos)
}

fn raise_exception(msg: String) -> Result<JinjaValue, minijinja::Error> {
    Err(minijinja::Error::new(
        minijinja::ErrorKind::InvalidOperation,
        msg,
    ))
}

fn civil_from_days(z: i64) -> (i64, u32, u32) {
    let z = z + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z.rem_euclid(146_097);
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    (y + i64::from(m <= 2), m as u32, d as u32)
}

const MONTH_NAMES: [&str; 12] = [
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December",
];

const DAY_NAMES: [&str; 7] = [
    "Thursday",
    "Friday",
    "Saturday",
    "Sunday",
    "Monday",
    "Tuesday",
    "Wednesday",
];

pub fn format_utc_date(fmt: &str, epoch_secs: i64) -> String {
    let days = epoch_secs.div_euclid(86_400);
    let (y, m, d) = civil_from_days(days);
    let mut out = String::with_capacity(fmt.len() + 8);
    let mut chars = fmt.chars();
    while let Some(c) = chars.next() {
        if c != '%' {
            out.push(c);
            continue;
        }
        match chars.next() {
            Some('Y') => out.push_str(&y.to_string()),
            Some('y') => out.push_str(&format!("{:02}", y.rem_euclid(100))),
            Some('m') => out.push_str(&format!("{m:02}")),
            Some('d') => out.push_str(&format!("{d:02}")),
            Some('e') => out.push_str(&format!("{d:2}")),
            Some('B') => out.push_str(MONTH_NAMES[(m - 1) as usize]),
            Some('b') => out.push_str(&MONTH_NAMES[(m - 1) as usize][..3]),
            Some('A') => out.push_str(DAY_NAMES[days.rem_euclid(7) as usize]),
            Some('j') => {
                let jan1 = {
                    let mut acc = 0i64;
                    for mm in 1..m {
                        acc += days_in_month(y, mm);
                    }
                    acc
                };
                out.push_str(&format!("{:03}", jan1 + i64::from(d)));
            }
            Some('%') => out.push('%'),
            Some(other) => {
                out.push('%');
                out.push(other);
            }
            None => out.push('%'),
        }
    }
    out
}

fn days_in_month(y: i64, m: u32) -> i64 {
    match m {
        1 | 3 | 5 | 7 | 8 | 10 | 12 => 31,
        4 | 6 | 9 | 11 => 30,
        _ => {
            if (y % 4 == 0 && y % 100 != 0) || y % 400 == 0 {
                29
            } else {
                28
            }
        }
    }
}

fn strftime_now(fmt: String) -> Result<JinjaValue, minijinja::Error> {
    let secs = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0);
    Ok(JinjaValue::from(format_utc_date(&fmt, secs)))
}

fn next_harmony_marker(s: &str) -> Option<usize> {
    let mut best: Option<usize> = None;
    for m in [
        "<|channel|>",
        "<|message|>",
        "<|start|>",
        "<|end|>",
        "<|return|>",
        "<|call|>",
        "<|constrain|>",
    ] {
        if let Some(at) = s.find(m) {
            best = Some(best.map_or(at, |b: usize| b.min(at)));
        }
    }
    best
}

pub fn harmony_reasoning_text(raw: &str) -> String {
    harmony_channel_text(raw, "analysis")
}

pub fn harmony_final_text(raw: &str) -> String {
    harmony_channel_text(raw, "final")
}

fn harmony_channel_text(raw: &str, want: &str) -> String {
    const CH: &str = "<|channel|>";
    const MSG: &str = "<|message|>";
    if !raw.contains(CH) {
        return if want == "final" {
            raw.to_string()
        } else {
            String::new()
        };
    }
    let mut out = String::new();
    let mut rest = raw;
    while let Some(ch) = rest.find(CH) {
        let after = &rest[ch + CH.len()..];
        let Some(msg_rel) = after.find(MSG) else {
            break;
        };
        let header = &after[..msg_rel];
        let body = &after[msg_rel + MSG.len()..];
        let end = next_harmony_marker(body).unwrap_or(body.len());
        if header.trim_start().starts_with(want) {
            out.push_str(&body[..end]);
        }
        rest = &body[end..];
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn write(dir: &Path, name: &str, body: &str) {
        std::fs::write(dir.join(name), body).unwrap();
    }

    fn tmp(tag: &str) -> std::path::PathBuf {
        let mut p = std::env::temp_dir();
        p.push(format!("chattmpl-{tag}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&p);
        std::fs::create_dir_all(&p).unwrap();
        p
    }

    #[test]
    fn loads_jinja_file_and_renders_chatml() {
        let d = tmp("chatml");
        write(
            &d,
            "chat_template.jinja",
            "{{ bos_token }}{% for m in messages %}<|im_start|>{{ m['role'] }}\n{{ m['content'] }}<|im_end|>\n{% endfor %}{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}",
        );
        write(
            &d,
            "tokenizer_config.json",
            r#"{"bos_token": "<s>", "eos_token": "</s>"}"#,
        );
        let t = ChatTemplate::load(&d).expect("load");
        let msgs = json!([
            {"role": "system", "content": "be terse"},
            {"role": "user", "content": "hi"}
        ]);
        let out = t.render(&msgs, None, true).unwrap();
        assert!(out.starts_with("<s>"), "bos: {out}");
        assert!(out.contains("<|im_start|>system\nbe terse<|im_end|>"));
        assert!(out.contains("<|im_start|>user\nhi<|im_end|>"));
        assert!(out.trim_end().ends_with("<|im_start|>assistant"));
    }

    #[test]
    fn pycompat_split_and_startswith() {
        let d = tmp("pycompat");
        write(
            &d,
            "chat_template.jinja",
            "{% set p = messages[0]['content'].split(':')[0] %}{{ p }}|{{ messages[0]['content'].startswith('a') }}",
        );
        write(
            &d,
            "tokenizer_config.json",
            r#"{"bos_token":"","eos_token":""}"#,
        );
        let t = ChatTemplate::load(&d).expect("load");
        let out = t
            .render(&json!([{"role":"user","content":"abc:def"}]), None, false)
            .unwrap();

        assert_eq!(out, "abc|True");
    }

    #[test]
    fn falls_back_to_tokenizer_config_template() {
        let d = tmp("tokcfg");
        write(
            &d,
            "tokenizer_config.json",
            r#"{"bos_token":"<s>","eos_token":"</s>","chat_template":"{{ bos_token }}{{ messages[0]['content'] }}"}"#,
        );
        let t = ChatTemplate::load(&d).expect("load");
        let out = t
            .render(&json!([{"role":"user","content":"yo"}]), None, false)
            .unwrap();
        assert_eq!(out, "<s>yo");
    }

    #[test]
    fn missing_template_returns_none() {
        let d = tmp("none");
        write(&d, "tokenizer_config.json", r#"{"bos_token":"<s>"}"#);
        assert!(ChatTemplate::load(&d).is_none());
    }

    #[test]
    fn supports_tools_probes_the_rendered_output() {
        let with = tmp("tools-yes");
        write(
            &with,
            "chat_template.jinja",
            "{% if tools %}TOOLS:{% for t in tools %}{{ t['function']['name'] }}{% endfor %}\n{% endif %}{{ messages[0]['content'] }}",
        );
        write(
            &with,
            "tokenizer_config.json",
            r#"{"bos_token":"","eos_token":""}"#,
        );
        assert!(ChatTemplate::load(&with).unwrap().supports_tools());

        let without = tmp("tools-no");
        write(
            &without,
            "chat_template.jinja",
            "{{ messages[0]['content'] }}",
        );
        write(
            &without,
            "tokenizer_config.json",
            r#"{"bos_token":"","eos_token":""}"#,
        );
        assert!(!ChatTemplate::load(&without).unwrap().supports_tools());
    }

    #[test]
    fn load_attempts_are_recorded_for_hits_and_misses() {
        let ok = tmp("attempt-ok");
        write(&ok, "chat_template.jinja", "{{ messages[0]['content'] }}");
        write(
            &ok,
            "tokenizer_config.json",
            r#"{"bos_token":"","eos_token":""}"#,
        );
        assert!(ChatTemplate::load(&ok).is_some());

        let miss = tmp("attempt-miss");
        write(&miss, "tokenizer_config.json", r#"{"bos_token":"<s>"}"#);
        assert!(ChatTemplate::load(&miss).is_none());

        let ok_key = ok.file_name().unwrap().to_str().unwrap();
        let miss_key = miss.file_name().unwrap().to_str().unwrap();
        assert!(load_was_attempted(ok_key));
        assert!(load_attempt_for(ok_key).unwrap().error.is_none());
        let missed = load_attempt_for(miss_key).expect("miss recorded");
        assert_eq!(missed.dir, miss);
        assert!(missed
            .error
            .unwrap()
            .contains("neither chat_template.jinja nor tokenizer_config.json:chat_template"));
        assert!(!load_was_attempted("a-model-that-was-never-loaded"));
    }

    #[test]
    fn default_chat_template_kwargs_reach_the_render() {
        let d = tmp("kwargs");
        write(
            &d,
            "chat_template.jinja",
            "{%- set enable_thinking = enable_thinking | default(false) -%}\
             {{ messages[0]['content'] }}{% if enable_thinking %}<think>{% else %}</think>{% endif %}",
        );
        write(
            &d,
            "tokenizer_config.json",
            r#"{"bos_token":"","eos_token":""}"#,
        );
        write(
            &d,
            "generation_config.json",
            r#"{"eos_token_id":[2,24],"default_chat_template_kwargs":{"enable_thinking":true}}"#,
        );
        let t = ChatTemplate::load(&d).expect("load");
        assert_eq!(
            t.default_template_kwargs().get("enable_thinking"),
            Some(&serde_json::json!(true))
        );
        assert!(t.declares_thinking_switch());
        let msgs = json!([{"role":"user","content":"hi"}]);
        assert_eq!(t.render(&msgs, None, false).unwrap(), "hi<think>");

        let mut off = std::collections::BTreeMap::new();
        off.insert("enable_thinking".to_string(), serde_json::json!(false));
        assert_eq!(
            t.render_with_kwargs(&msgs, None, false, &off).unwrap(),
            "hi</think>"
        );
    }

    #[test]
    fn a_template_without_declared_kwargs_renders_exactly_as_before() {
        let d = tmp("nokwargs");
        write(
            &d,
            "chat_template.jinja",
            "{{ bos_token }}{{ messages[0]['content'] }}{% if add_generation_prompt %}!{% endif %}",
        );
        write(
            &d,
            "tokenizer_config.json",
            r#"{"bos_token":"<s>","eos_token":"</s>"}"#,
        );
        let t = ChatTemplate::load(&d).expect("load");
        assert!(t.default_template_kwargs().is_empty());
        assert!(!t.declares_thinking_switch());
        let msgs = json!([{"role":"user","content":"hi"}]);
        assert_eq!(t.render(&msgs, None, true).unwrap(), "<s>hi!");
    }

    #[test]
    fn template_kwargs_env_is_parsed_as_a_json_object() {
        let m = parse_template_kwargs(r#"{"enable_thinking": false, "n": 3}"#).unwrap();
        assert_eq!(m.get("enable_thinking"), Some(&serde_json::json!(false)));
        assert_eq!(m.get("n"), Some(&serde_json::json!(3)));
        assert!(parse_template_kwargs("   ").unwrap().is_empty());
        assert!(parse_template_kwargs("[1,2]").is_err());
        assert!(parse_template_kwargs("{oops").is_err());
    }

    #[test]
    fn strftime_now_renders_the_real_utc_date_not_the_epoch() {
        let d = tmp("strftime");
        write(
            &d,
            "chat_template.jinja",
            "Current date: {{ strftime_now('%Y-%m-%d') }}",
        );
        write(
            &d,
            "tokenizer_config.json",
            r#"{"bos_token":"","eos_token":""}"#,
        );
        let t = ChatTemplate::load(&d).expect("load");
        let out = t
            .render(&json!([{"role":"user","content":"x"}]), None, false)
            .unwrap();
        assert!(!out.contains("1970"), "epoch stub leaked: {out}");
        let secs = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs() as i64;
        assert_eq!(
            out,
            format!("Current date: {}", format_utc_date("%Y-%m-%d", secs))
        );
    }

    #[test]
    fn format_utc_date_handles_known_dates_and_codes() {
        assert_eq!(format_utc_date("%Y-%m-%d", 0), "1970-01-01");
        assert_eq!(format_utc_date("%Y-%m-%d", 1_786_060_800), "2026-08-07");
        assert_eq!(format_utc_date("%d %B %Y", 1_786_060_800), "07 August 2026");
        assert_eq!(format_utc_date("%A", 0), "Thursday");
        assert_eq!(format_utc_date("%j", 86_400 * 31), "032");
        assert_eq!(format_utc_date("100%% %q", 0), "100% %q");
        assert_eq!(format_utc_date("%Y-%m-%d", 951_782_400), "2000-02-29");
    }

    #[test]
    fn harmony_filter_keeps_only_the_final_channel() {
        let raw = "<|channel|>analysis<|message|>User asks capital of France. Easy.<|end|>\
                   <|start|>assistant<|channel|>final<|message|>Paris is the capital of France.<|return|>";
        assert_eq!(harmony_final_text(raw), "Paris is the capital of France.");
    }

    #[test]
    fn harmony_filter_streams_incrementally_and_never_leaks_markup() {
        let full = "<|channel|>analysis<|message|>thinking...<|end|><|start|>assistant\
                    <|channel|>final<|message|>Hello world<|return|>";
        let mut prev = String::new();
        for cut in 0..=full.len() {
            if !full.is_char_boundary(cut) {
                continue;
            }
            let vis = harmony_final_text(&full[..cut]);
            for m in [
                "<|channel|>",
                "<|start|>",
                "<|return|>",
                "<|message|>",
                "<|end|>",
            ] {
                assert!(!vis.contains(m), "markup {m} leaked at cut {cut}: {vis:?}");
            }
            assert!(
                vis.starts_with(&prev) || prev.starts_with(&vis),
                "stream regressed at cut {cut}: {prev:?} -> {vis:?}"
            );
            prev = vis;
        }
        assert_eq!(prev, "Hello world");
    }

    #[test]
    fn harmony_filter_passes_plain_text_through() {
        assert_eq!(harmony_final_text("just text"), "just text");
        assert_eq!(harmony_final_text(""), "");
    }

    #[test]
    fn harmony_filter_drops_tool_call_commentary() {
        let raw = "<|channel|>commentary to=functions.get_weather <|constrain|>json<|message|>\
                   {\"location\":\"Paris\"}<|call|>";
        assert_eq!(harmony_final_text(raw), "");
        assert_eq!(
            harmony_reasoning_text(raw),
            "",
            "a tool-call commentary channel is neither content nor reasoning"
        );
    }

    #[test]
    fn a_completion_that_never_reached_its_final_channel_still_has_its_reasoning() {

        let truncated = "<|channel|>analysis<|message|>Count: 1, 2, 3";
        assert_eq!(harmony_final_text(truncated), "");
        assert_eq!(harmony_reasoning_text(truncated), "Count: 1, 2, 3");

        let answered = "<|channel|>analysis<|message|>thinking\
                        <|end|><|start|>assistant<|channel|>final<|message|>Four.";
        assert_eq!(harmony_final_text(answered), "Four.");
        assert!(
            harmony_reasoning_text(answered).contains("thinking"),
            "the analysis is still recoverable once a final channel exists"
        );

        assert_eq!(
            harmony_reasoning_text("just text"),
            "",
            "unchannelled output is content, never reasoning"
        );
        assert_eq!(harmony_reasoning_text(""), "");
    }

    fn laguna_snapshot() -> Option<std::path::PathBuf> {
        let base = format!(
            "{}/.cache/huggingface/hub/models--poolside--Laguna-XS-2.1-NVFP4/snapshots",
            std::env::var("HOME").unwrap_or_default()
        );
        std::fs::read_dir(base)
            .ok()?
            .filter_map(|e| e.ok())
            .map(|e| e.path())
            .find(|p| p.join("chat_template.jinja").is_file())
    }

    #[test]
    fn renders_real_laguna_template_with_its_shipped_thinking_default() {
        let Some(snap) = laguna_snapshot() else {
            eprintln!("skip: Laguna-XS-2.1-NVFP4 not cached");
            return;
        };
        let t = ChatTemplate::load(&snap).expect("load real laguna template");
        assert_eq!(
            t.default_template_kwargs().get("enable_thinking"),
            Some(&serde_json::json!(true)),
            "generation_config.json ships default_chat_template_kwargs.enable_thinking=true"
        );

        let msgs = json!([{"role": "user", "content": "What is the capital of France?"}]);
        let shipped = t.render(&msgs, None, true).unwrap();
        eprintln!("--- laguna shipped-default render ---\n{shipped:?}\n--- end ---");
        assert!(
            shipped.starts_with("〈|EOS|〉<system>You are a helpful"),
            "{shipped:?}"
        );
        assert!(
            shipped.contains("made by Poolside"),
            "the shipped template carries a default Poolside system persona: {shipped:?}"
        );
        assert!(
            shipped.ends_with("<assistant><think>"),
            "thinking-on generation prompt opens <think>: {shipped:?}"
        );

        let mut off = std::collections::BTreeMap::new();
        off.insert("enable_thinking".to_string(), serde_json::json!(false));
        let thinkoff = t.render_with_kwargs(&msgs, None, true, &off).unwrap();
        assert!(thinkoff.ends_with("<assistant></think>"), "{thinkoff:?}");
        assert_ne!(
            shipped, thinkoff,
            "the two thinking modes must not render identically"
        );

        let hand_built = format!(
            "〈|EOS|〉<user>{}</user>\n<assistant></think>",
            "What is the capital of France?"
        );
        assert_ne!(
            thinkoff, hand_built,
            "the 30-file hand-built wrapper must differ from the official render"
        );
        eprintln!("hand-built wrapper: {hand_built:?}");
        eprintln!("official thinking-off render: {thinkoff:?}\ndelta = the default system block");

        let empty_sys = json!([
            {"role": "system", "content": ""},
            {"role": "user", "content": "What is the capital of France?"}
        ]);
        let no_sys = t.render_with_kwargs(&empty_sys, None, true, &off).unwrap();
        assert_eq!(
            no_sys, hand_built,
            "an EMPTY system message is the template's documented opt-out of the default persona; \
             only then does the official render equal the hand-built wrapper"
        );
    }

    fn gemma4_snapshot() -> Option<std::path::PathBuf> {
        let base = format!(
            "{}/.cache/huggingface/hub/models--nvidia--Gemma-4-31B-IT-NVFP4/snapshots",
            std::env::var("HOME").unwrap_or_default()
        );
        std::fs::read_dir(base)
            .ok()?
            .filter_map(|e| e.ok())
            .map(|e| e.path())
            .find(|p| p.join("chat_template.jinja").is_file())
    }

    #[test]
    fn gemma4_thinking_default_is_unset_and_renders_a_closed_empty_thought_block() {
        let Some(snap) = gemma4_snapshot() else {
            eprintln!("skip: Gemma-4-31B-IT-NVFP4 not cached");
            return;
        };
        let t = ChatTemplate::load(&snap).expect("load real gemma template");
        assert!(t.declares_thinking_switch());
        assert_eq!(
            t.default_template_kwargs().get("enable_thinking"),
            None,
            "Gemma 4 ships no default_chat_template_kwargs, so the served default leaves \
             enable_thinking UNSET -- which is not the same as false in this template"
        );

        let msgs = json!([{"role": "user", "content": "What is the capital of France?"}]);
        let served = t.render(&msgs, None, true).unwrap();
        eprintln!("--- gemma4 served-default render ---\n{served:?}\n--- end ---");
        assert!(
            served.ends_with("<|turn>model\n<|channel>thought\n<channel|>"),
            "unset enable_thinking must emit an already-closed empty thought block: {served:?}"
        );
        assert!(!served.contains("<|think|>"), "{served:?}");

        let mut on = std::collections::BTreeMap::new();
        on.insert("enable_thinking".to_string(), serde_json::json!(true));
        let thinking = t.render_with_kwargs(&msgs, None, true, &on).unwrap();
        eprintln!("--- gemma4 enable_thinking=true render ---\n{thinking:?}\n--- end ---");
        assert!(
            thinking.contains("<|turn>system\n<|think|>"),
            "enable_thinking=true injects <|think|> at the top of the first system turn: \
             {thinking:?}"
        );
        assert!(
            !thinking.contains("<|channel>thought"),
            "thinking-on must NOT pre-close the thought block: {thinking:?}"
        );
        assert_ne!(served, thinking);
    }

    #[test]
    fn renders_real_gemma4_template_with_tools() {
        let Some(snap) = gemma4_snapshot() else {
            eprintln!("skip: Gemma-4-31B-IT-NVFP4 not cached");
            return;
        };
        let t = ChatTemplate::load(&snap).expect("load real gemma template");
        assert!(
            t.uses_tool_responses(),
            "gemma template uses tool_responses"
        );

        let msgs = json!([
            {"role": "user", "content": "What's the weather in Paris?"}
        ]);
        let tools = json!([{
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get current weather",
                "parameters": {
                    "type": "object",
                    "properties": {"location": {"type": "string", "description": "city"}},
                    "required": ["location"]
                }
            }
        }]);
        let out = t.render(&msgs, Some(&tools), true).unwrap();
        eprintln!("--- real gemma render ---\n{out}\n--- end ---");
        assert!(out.contains("<|turn>system"), "system block: {out}");
        assert!(out.contains("declaration:get_weather"), "tool decl: {out}");
        assert!(out.contains("<|turn>user"), "user turn");

        assert!(out.contains("<|turn>model"), "gen prompt: {out}");
    }

    fn gemma4_w4a16_snapshot() -> Option<std::path::PathBuf> {
        let base = format!(
            "{}/.cache/huggingface/hub/models--google--gemma-4-E4B-it-qat-w4a16-ct/snapshots",
            std::env::var("HOME").unwrap_or_default()
        );
        std::fs::read_dir(base)
            .ok()?
            .filter_map(|e| e.ok())
            .map(|e| e.path())
            .find(|p| p.join("chat_template.jinja").is_file())
    }

    #[test]
    fn parses_openai_wire_string_tool_call_arguments_before_render() {
        let d = tmp("wire-args");
        write(
            &d,
            "chat_template.jinja",
            "{{ messages[0]['tool_calls'][0]['function']['arguments']['city'] }}|{{ messages[0]['tool_calls'][1]['function']['arguments'] | length }}|{{ messages[0]['tool_calls'][2]['function']['arguments'] is string }}",
        );
        write(
            &d,
            "tokenizer_config.json",
            r#"{"bos_token":"","eos_token":""}"#,
        );
        let t = ChatTemplate::load(&d).expect("load");
        let msgs = json!([{
            "role": "assistant",
            "content": null,
            "tool_calls": [
                {"id": "a", "type": "function", "function": {"name": "go", "arguments": "{\"city\":\"Oslo\"}"}},
                {"id": "b", "type": "function", "function": {"name": "go", "arguments": ""}},
                {"id": "c", "type": "function", "function": {"name": "go", "arguments": "not json"}}
            ]
        }]);
        let out = t.render(&msgs, None, false).unwrap();
        assert_eq!(
            out, "Oslo|0|True",
            "string arguments must render as parsed mappings, empty as {{}}, junk untouched"
        );
    }

    #[test]
    fn renders_real_gemma4_tool_call_history_with_the_trailing_user_turn_last() {
        let Some(snap) = gemma4_w4a16_snapshot() else {
            eprintln!("skip: gemma-4-E4B-it-qat-w4a16-ct not cached");
            return;
        };
        let t = ChatTemplate::load(&snap).expect("load real gemma template");
        let msgs = json!([
            {"role": "user", "content": "Call the tool named now."},
            {"role": "assistant", "content": null, "tool_calls": [
                {"id": "call_1", "type": "function",
                 "function": {"name": "now", "arguments": "{}"}}
            ]},
            {"role": "tool", "tool_call_id": "call_1", "content": "2026-08-17 05:05 local"},
            {"role": "user", "content": "Now just answer: what fruit is yellow?"}
        ]);
        let tools = json!([{
            "type": "function",
            "function": {
                "name": "now",
                "description": "Get the current local time.",
                "parameters": {"type": "object", "properties": {}}
            }
        }]);
        let out = t.render(&msgs, Some(&tools), true).unwrap();
        eprintln!("--- real gemma tool-history render ---\n{out:?}\n--- end ---");
        assert_eq!(
            out,
            "<bos><|turn>system\n<|tool>declaration:now{description:<|\"|>Get the current local time.<|\"|>,parameters:{type:<|\"|>OBJECT<|\"|>}}<tool|><turn|>\n\
             <|turn>user\nCall the tool named now.<turn|>\n\
             <|turn>model\n<|tool_call>call:now{}<tool_call|><|tool_response>response:now{value:<|\"|>2026-08-17 05:05 local<|\"|>}<tool_response|><turn|>\n\
             <|turn>user\nNow just answer: what fruit is yellow?<turn|>\n\
             <|turn>model\n",
            "the tool call renders inside the model turn, the tool result renders as its \
             tool_response block, and the trailing user question is the last turn before the \
             generation prompt"
        );
    }

    #[test]
    fn plain_no_tools_render_is_byte_identical_with_normalization_inactive() {
        let Some(snap) = gemma4_w4a16_snapshot() else {
            eprintln!("skip: gemma-4-E4B-it-qat-w4a16-ct not cached");
            return;
        };
        let msgs = json!([
            {"role": "system", "content": "be terse"},
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello!"},
            {"role": "user", "content": "bye?"}
        ]);
        assert!(
            tool_call_arguments_parsed_from_openai_wire_strings(&msgs).is_none(),
            "a conversation without string tool-call arguments must not be rewritten"
        );
        let t = ChatTemplate::load(&snap).expect("load real gemma template");
        let out = t.render(&msgs, None, true).unwrap();
        eprintln!("--- real gemma plain render ---\n{out:?}\n--- end ---");
        assert_eq!(
            out,
            "<bos><|turn>system\nbe terse<turn|>\n\
             <|turn>user\nhi<turn|>\n\
             <|turn>model\nhello!<turn|>\n\
             <|turn>user\nbye?<turn|>\n\
             <|turn>model\n"
        );
    }

    struct Derivation {
        id: &'static str,
        on: &'static str,
        off: &'static str,
        want: Option<&'static str>,
        why: &'static str,
    }

    const DERIVATIONS: &[Derivation] = &[
        Derivation {
            id: "qwen3_classic_off_arm_appends_an_open_tag_and_its_close",
            on: "<|im_start|>assistant\n",
            off: "<|im_start|>assistant\n<think>\n\n</think>\n\n",
            want: Some("</think>"),
            why: "the thinking-on arm appends NOTHING, so the whole divergent tail is an open tag \
                  plus its close and the answer is the LAST tag in the tail, not the first",
        },
        Derivation {
            id: "qwen36_both_arms_open_the_thought_and_only_the_off_arm_closes_it",
            on: "<|im_start|>assistant\n<think>\n",
            off: "<|im_start|>assistant\n<think>\n\n</think>\n\n",
            want: Some("</think>"),
            why: "the arms share a COMPLETE '<think>' before diverging, so the divergence must \
                  stay where the bytes differ and must NOT rewind to that shared tag's '<'",
        },
        Derivation {
            id: "laguna_both_arms_share_the_opening_angle_bracket_of_different_tags",
            on: "<assistant><think>",
            off: "<assistant></think>",
            want: Some("</think>"),
            why: "'<think>' and '</think>' share their leading '<', so a byte-wise common prefix \
                  ends INSIDE the marker and yields the truncated '/think>'; the divergence must \
                  rewind to the '<' that opened the unterminated tag",
        },
        Derivation {
            id: "gemma4_close_marker_is_not_a_slash_tag_and_is_not_the_literal_think_close",
            on: "<|turn>model\n",
            off: "<|turn>model\n<|channel>thought\n<channel|>",
            want: Some("<channel|>"),
            why: "the close is '<channel|>' and '</think>' is absent from this dialect entirely, \
                  so a derivation that quietly falls back to the literal '</think>' arms the \
                  grammar at a position the model never emits",
        },
        Derivation {
            id: "a_divergent_tail_that_is_prose_rather_than_a_tag_yields_nothing",
            on: "<|im_start|>assistant\n",
            off: "<|im_start|>assistant\nThinking is disabled for this turn.\n",
            want: None,
            why: "the arms differ, but by prose; None is the honest answer and any string \
                  returned here is the derivation inventing a marker out of neighbouring text",
        },
        Derivation {
            id: "a_tag_the_thinking_on_prompt_also_emits_cannot_be_what_closes_a_thought",
            on: "<a></think>",
            off: "<b></think>",
            want: None,
            why: "the tail's last tag appears in the thinking-ON prompt too, so it is part of \
                  the turn scaffolding rather than the close of a thought",
        },
        Derivation {
            id: "arms_that_render_identically_leave_no_tail_and_no_marker",
            on: "<|im_start|>assistant\n",
            off: "<|im_start|>assistant\n",
            want: None,
            why: "thinking armed at the top of the render cancels out of both generation \
                  prompts; only marker_from_closed_thought can speak for such a template",
        },
        Derivation {
            id: "the_eos_guard_lives_in_thinking_close_marker_and_not_in_this_derivation",
            on: "<|im_start|>assistant\n",
            off: "<|im_start|>assistant\n<|im_end|>",
            want: Some("<|im_end|>"),
            why: "'<|im_end|>' passes every lone-tag test, so this layer returns it and only the \
                  eos filter in thinking_close_marker rejects it; this row records where that \
                  responsibility sits so a lost guard cannot be read as a change here",
        },
    ];

    const DERIVATION_SHAPES_ON_RECORD: usize = 8;

    #[test]
    fn the_close_marker_derivation_is_decided_by_two_strings_and_needs_no_snapshot() {
        assert_eq!(
            DERIVATIONS.len(),
            DERIVATION_SHAPES_ON_RECORD,
            "a shape was added or dropped without moving the count; every published dialect this \
             repo serves is represented here and none of them may leave silently"
        );
        for c in DERIVATIONS {
            assert_eq!(
                close_marker_between_thinking_on_and_off_prompts(c.on, c.off).as_deref(),
                c.want,
                "{}\n    this shape exists because: {}\n    thinking-on  prompt: {:?}\n    \
                 thinking-off prompt: {:?}",
                c.id,
                c.why,
                c.on,
                c.off
            );
        }
    }

    #[test]
    fn tag_aligned_divergence_rewinds_only_into_a_tag_the_common_prefix_left_unterminated() {
        assert_eq!(
            tag_aligned_divergence("<assistant><think>", "<assistant></think>"),
            11,
            "the shared bytes end after the '<' of two different tags, so the divergence must \
             move back onto that '<' or the marker comes out truncated to '/think>'"
        );
        assert_eq!(
            tag_aligned_divergence(
                "<|im_start|>assistant\n<think>\n",
                "<|im_start|>assistant\n<think>\n\n</think>\n\n"
            ),
            30,
            "the last '<' before the divergence opened a tag the shared bytes already CLOSED, so \
             rewinding would swallow the shared '<think>' and hand back the wrong tag"
        );
        assert_eq!(
            tag_aligned_divergence("<|turn>model\n<foo bar", "<|turn>model\n<foo baz"),
            20,
            "an unterminated '<' followed by whitespace is prose, not a tag, so the divergence \
             stays where the bytes differ"
        );
        assert_eq!(
            tag_aligned_divergence("abc", "abd"),
            2,
            "no '<' precedes the divergence, so there is nothing to rewind onto"
        );
        assert_eq!(tag_aligned_divergence("", ""), 0);
    }

    #[test]
    fn tag_aligned_divergence_never_returns_an_index_inside_a_multibyte_char() {
        let on = "<tag><\u{2014}>";
        let off = "<tag><\u{2013}>";
        let at = tag_aligned_divergence(on, off);
        assert!(
            off.is_char_boundary(at),
            "the byte-wise prefix ends between the second and third byte of an em/en dash; \
             slicing there panics, so the walk back to a char boundary is load-bearing"
        );
        assert_eq!(at, 5);
        assert_eq!(
            close_marker_between_thinking_on_and_off_prompts(on, off).as_deref(),
            Some("<\u{2013}>"),
            "Laguna renders '\u{3008}|EOS|\u{3009}', so non-ASCII in a template is not \
             hypothetical and must survive the derivation intact"
        );
    }

    #[test]
    fn leading_tag_takes_the_first_whole_tag_and_nothing_that_merely_resembles_one() {
        assert_eq!(
            leading_tag("\n<channel|>the answer"),
            Some("<channel|>".into())
        );
        assert_eq!(leading_tag("   <think>x"), Some("<think>".into()));
        assert_eq!(
            leading_tag("<|channel|>analysis"),
            Some("<|channel|>".into())
        );
        assert_eq!(leading_tag("no tag at all"), None);
        assert_eq!(
            leading_tag("prose before the > sign"),
            None,
            "a '>' with no '<' opening it is punctuation, not the close of a thought"
        );
        assert_eq!(leading_tag("<>"), None);
        assert_eq!(leading_tag("<a<b>"), None);
        assert_eq!(leading_tag(""), None);
    }

    #[test]
    fn trailing_tag_takes_the_last_whole_tag_and_requires_it_to_end_the_text() {
        assert_eq!(
            trailing_tag("<think>\n\n</think>\n\n"),
            Some("</think>".into()),
            "the tail opens AND closes a thought; the close is the last tag, and trailing \
             whitespace must not hide it"
        );
        assert_eq!(
            trailing_tag("<|channel>thought\n<channel|>"),
            Some("<channel|>".into())
        );
        assert_eq!(trailing_tag("Thinking is disabled for this turn.\n"), None);
        assert_eq!(
            trailing_tag("<tag> and then prose"),
            None,
            "a tag with text after it is not what the render ends on, and returning it would \
             arm the grammar on template prose"
        );
        assert_eq!(trailing_tag("<a b>"), None);
        assert_eq!(trailing_tag(""), None);
    }

    #[test]
    fn is_lone_tag_rejects_empty_nested_and_unterminated_tags() {
        assert!(is_lone_tag("<a>"));
        assert!(is_lone_tag("</think>"));
        assert!(is_lone_tag("<channel|>"));
        assert!(is_lone_tag("<|im_end|>"));
        assert!(!is_lone_tag("<>"));
        assert!(!is_lone_tag("<think"));
        assert!(!is_lone_tag("think>"));
        assert!(!is_lone_tag("<a<b>"));
        assert!(!is_lone_tag("<a>b>"));
        assert!(!is_lone_tag("<a b>"));
        assert!(!is_lone_tag(""));
    }

    #[test]
    fn chat_template_json_alone_carries_a_template_and_an_empty_jinja_does_not_mask_it() {
        let d = tmp("json-only");
        write(&d, "chat_template.jinja", "   \n");
        write(
            &d,
            "chat_template.json",
            r#"{"chat_template":"{{ messages[0]['content'] }}!"}"#,
        );
        write(
            &d,
            "tokenizer_config.json",
            r#"{"bos_token":"","eos_token":""}"#,
        );
        let t = ChatTemplate::load(&d).expect(
            "a template carried only by chat_template.json must load; load() does not raise in \
             serving, it logs and returns None and the request is then prompted with the built-in \
             renderer in a format the model was never trained on",
        );
        assert_eq!(
            t.render(&json!([{"role":"user","content":"hi"}]), None, false)
                .unwrap(),
            "hi!"
        );
    }

    #[test]
    fn a_chat_template_shipped_as_a_named_list_resolves_to_the_default_entry() {
        assert_eq!(
            chat_template_value(Some(&json!([
                {"name": "tool_use", "template": "TOOLS"},
                {"name": "default", "template": "DEFAULT"}
            ]))),
            Some("DEFAULT".to_string()),
            "the legacy chat_template.json list form names its arms; picking the first would \
             prompt every plain chat turn with the tool-use dialect"
        );
        assert_eq!(
            chat_template_value(Some(&json!([{"name": "rag", "template": "RAG"}]))),
            Some("RAG".to_string())
        );
        assert_eq!(chat_template_value(Some(&json!([]))), None);
        assert_eq!(chat_template_value(Some(&json!("   "))), None);
        assert_eq!(chat_template_value(None), None);
    }

    fn q38_effort_env_lock_serializes_tests_that_touch_the_process_global_env(
    ) -> std::sync::MutexGuard<'static, ()> {
        static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
        LOCK.get_or_init(|| Mutex::new(()))
            .lock()
            .unwrap_or_else(|p| p.into_inner())
    }

    #[test]
    fn a_reasoning_effort_template_defaults_to_medium_and_the_env_override_and_request_kwargs_win()
    {
        let _env = q38_effort_env_lock_serializes_tests_that_touch_the_process_global_env();
        let d = tmp("effort");
        write(
            &d,
            "chat_template.jinja",
            "{%- set e = reasoning_effort|default('xhigh') -%}effort={{ e }}|{{ messages[0]['content'] }}",
        );
        write(
            &d,
            "tokenizer_config.json",
            r#"{"bos_token":"","eos_token":""}"#,
        );
        let t = ChatTemplate::load(&d).expect("load");
        assert!(t.declares_reasoning_effort());
        assert_eq!(
            t.effective_template_kwargs().get(REASONING_EFFORT_KWARG),
            Some(&serde_json::json!("medium")),
            "the served default must be medium, not the template's shipped xhigh"
        );
        let msgs = json!([{"role":"user","content":"hi"}]);
        assert_eq!(t.render(&msgs, None, false).unwrap(), "effort=medium|hi");

        std::env::set_var(Q38_REASONING_EFFORT_ENV, "xhigh");
        assert_eq!(
            t.render(&msgs, None, false).unwrap(),
            "effort=xhigh|hi",
            "{Q38_REASONING_EFFORT_ENV} must override the medium serving default"
        );
        std::env::set_var(Q38_REASONING_EFFORT_ENV, "not-an-effort");
        assert_eq!(
            t.render(&msgs, None, false).unwrap(),
            "effort=medium|hi",
            "an invalid {Q38_REASONING_EFFORT_ENV} is logged and ignored, not served"
        );
        std::env::remove_var(Q38_REASONING_EFFORT_ENV);

        let mut per_request = std::collections::BTreeMap::new();
        per_request.insert(REASONING_EFFORT_KWARG.to_string(), serde_json::json!("low"));
        let merged = crate::oapi::chat::merged_template_kwargs(&t, &per_request);
        assert_eq!(
            merged.get(REASONING_EFFORT_KWARG),
            Some(&serde_json::json!("low")),
            "request kwargs must win over the serving default"
        );

        let no_effort = tmp("no-effort");
        write(&no_effort, "chat_template.jinja", "{{ messages[0]['content'] }}");
        write(
            &no_effort,
            "tokenizer_config.json",
            r#"{"bos_token":"","eos_token":""}"#,
        );
        let plain = ChatTemplate::load(&no_effort).expect("load");
        assert!(!plain.declares_reasoning_effort());
        assert!(
            !plain
                .effective_template_kwargs()
                .contains_key(REASONING_EFFORT_KWARG),
            "templates that never mention reasoning_effort must render exactly as before"
        );
    }

    #[test]
    fn reasoning_effort_values_accepted_match_the_qwen38_template_contract() {
        for ok in REASONING_EFFORT_VALUES_THE_QWEN38_TEMPLATE_ACCEPTS_WITH_HIGH_ALIASED_TO_XHIGH {
            validate_reasoning_effort(ok).unwrap();
        }
        for bad in ["minimal", "max", "none", "MEDIUM", "", "x-high"] {
            assert!(validate_reasoning_effort(bad).is_err(), "{bad:?}");
        }
    }

    fn qwen38_snapshot() -> Option<std::path::PathBuf> {
        let base = format!(
            "{}/.cache/huggingface/hub/models--unsloth--Qwen3.8-27B-NVFP4/snapshots",
            std::env::var("HOME").unwrap_or_default()
        );
        std::fs::read_dir(base)
            .ok()?
            .filter_map(|e| e.ok())
            .map(|e| e.path())
            .find(|p| p.join("chat_template.jinja").is_file())
    }

    #[test]
    fn renders_real_qwen38_template_with_medium_default_and_an_open_think_block() {
        let _env = q38_effort_env_lock_serializes_tests_that_touch_the_process_global_env();
        let Some(snap) = qwen38_snapshot() else {
            eprintln!("skip: Qwen3.8-27B-NVFP4 not cached");
            return;
        };
        let t = ChatTemplate::load(&snap).expect("load real qwen3.8 template");
        assert!(t.declares_thinking_switch());
        assert!(t.declares_reasoning_effort());

        let msgs = json!([{"role": "user", "content": "What is the capital of France?"}]);
        let served = t.render(&msgs, None, true).unwrap();
        eprintln!("--- qwen38 served-default render ---\n{served:?}\n--- end ---");
        assert!(
            !served.contains("Reasoning effort is set to"),
            "medium is the template's silent arm: it must inject NO effort system line, and the \
             shipped xhigh default injecting one here means the medium serving default was lost: \
             {served:?}"
        );
        assert!(
            served.ends_with("<|im_start|>assistant\n<think>\n"),
            "enable_thinking left undefined must OPEN a thought block: {served:?}"
        );

        assert_eq!(
            t.thinking_on_when_the_switch_is_undefined_scoped_to_reasoning_effort_templates_so_qwen36_guided_defaults_are_untouched(),
            Some(true),
            "the undefined-switch probe must report thinking ON so guided requests defer their \
             grammar past the thought this prompt just opened"
        );
        assert_eq!(
            t.thinking_close_marker().as_deref(),
            Some("</think>"),
            "the qwen3.8 close marker derivation must land on the literal the model emits"
        );

        let mut off = std::collections::BTreeMap::new();
        off.insert("enable_thinking".to_string(), serde_json::json!(false));
        let thinkoff = t.render_with_kwargs(&msgs, None, true, &off).unwrap();
        assert!(
            thinkoff.ends_with("<think>\n\n</think>\n\n"),
            "enable_thinking=false must pre-close the thought block: {thinkoff:?}"
        );

        for (effort, expect_line) in [
            ("xhigh", Some("Reasoning effort is set to xhigh")),
            ("high", Some("Reasoning effort is set to xhigh")),
            ("low", Some("Reasoning effort is set to low")),
            ("medium", None),
        ] {
            let mut kw = std::collections::BTreeMap::new();
            kw.insert(REASONING_EFFORT_KWARG.to_string(), serde_json::json!(effort));
            let out = t.render_with_kwargs(&msgs, None, true, &kw).unwrap();
            match expect_line {
                Some(line) => assert!(
                    out.contains(line),
                    "effort {effort} must inject {line:?}: {out:?}"
                ),
                None => assert!(
                    !out.contains("Reasoning effort is set to"),
                    "effort medium must stay silent: {out:?}"
                ),
            }
        }

        let mut invalid = std::collections::BTreeMap::new();
        invalid.insert(
            REASONING_EFFORT_KWARG.to_string(),
            serde_json::json!("not-an-effort"),
        );
        assert!(
            t.render_with_kwargs(&msgs, None, true, &invalid).is_err(),
            "the template itself raises on an unknown effort, which is why the request boundary \
             validates before rendering"
        );
    }
}
