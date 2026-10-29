use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::path::{Path, PathBuf};

pub const TEXT_RECORD_MIN_CHARS: usize = 128;
pub const TEXT_RECORD_MAX_CHARS: usize = 1600;
pub const TOOL_RECORD_MAX_CHARS: usize = 1600;
pub const TOOL_RESULT_HEAD_LINES: usize = 6;

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq, Hash, Copy)]
#[serde(rename_all = "lowercase")]
pub enum Kind {
    User,
    Assistant,
    Tool,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Record {
    pub id: String,
    pub kind: Kind,
    pub agent: String,
    pub host: String,
    pub session: String,
    pub time: String,
    pub source: String,
    pub line: usize,
    pub text: String,
}

fn sha(s: &str) -> String {
    let mut h = Sha256::new();
    h.update(s.as_bytes());
    hex::encode(h.finalize())
}

fn head_lines(s: &str, n: usize) -> String {
    let mut out: Vec<&str> = s.lines().take(n).collect();
    let more = s.lines().count() > n;
    if more {
        out.push("…");
    }
    out.join("\n")
}

fn clip(s: &str, max: usize) -> String {
    if s.chars().count() <= max {
        return s.to_string();
    }
    let cut: String = s.chars().take(max - 1).collect();
    format!("{cut}…")
}

fn value_text(v: &Value) -> String {
    match v {
        Value::String(s) => s.clone(),
        Value::Array(items) => items
            .iter()
            .filter_map(|x| {
                x.get("text")
                    .and_then(Value::as_str)
                    .map(str::to_string)
                    .or_else(|| x.as_str().map(str::to_string))
            })
            .collect::<Vec<_>>()
            .join("\n"),
        Value::Null => String::new(),
        other => other.to_string(),
    }
}

fn render_tool(name: &str, input: &Value, result: Option<&str>) -> String {
    let args = match input {
        Value::Object(map) => map
            .iter()
            .map(|(k, v)| format!("{k}: {}", clip(&value_text(v), 300).replace('\n', " ")))
            .collect::<Vec<_>>()
            .join("\n"),
        other => clip(&value_text(other), 600),
    };
    let mut text = format!("tool {name}\n{args}");
    if let Some(r) = result.filter(|r| !r.trim().is_empty()) {
        text.push_str("\nresult:\n");
        text.push_str(&head_lines(r, TOOL_RESULT_HEAD_LINES));
    }
    clip(&text, TOOL_RECORD_MAX_CHARS)
}

fn keep_text(t: &str) -> bool {
    let n = t.chars().count();
    (TEXT_RECORD_MIN_CHARS..=TEXT_RECORD_MAX_CHARS).contains(&n) && !t.trim_start().starts_with('<')
}

pub struct Cutter {
    pub host: String,
    pub include_tools: bool,
}

impl Cutter {
    pub fn cut_file(&self, path: &Path, agent: &str) -> Result<Vec<Record>> {
        let raw = std::fs::read_to_string(path).with_context(|| path.display().to_string())?;
        let source = path.display().to_string();
        let session = sha(&source)[..16].to_string();
        Ok(match agent {
            "codex" => self.cut_codex(&raw, &source, &session),
            _ => self.cut_claude(&raw, &source, &session),
        })
    }

    fn push(
        &self,
        out: &mut Vec<Record>,
        kind: Kind,
        agent: &str,
        session: &str,
        time: &str,
        source: &str,
        line: usize,
        text: String,
    ) {
        let text = text.trim().to_string();
        let ok = match kind {
            Kind::Tool => self.include_tools && text.chars().count() >= 24,
            _ => keep_text(&text),
        };
        if !ok {
            return;
        }
        let id = sha(&format!("{kind:?}\n{text}"));
        out.push(Record {
            id,
            kind,
            agent: agent.into(),
            host: self.host.clone(),
            session: session.into(),
            time: time.into(),
            source: source.into(),
            line,
            text,
        });
    }

    fn cut_claude(&self, raw: &str, source: &str, session: &str) -> Vec<Record> {
        let mut out = Vec::new();
        let mut results: HashMap<String, String> = HashMap::new();
        let lines: Vec<Option<Value>> = raw.lines().map(|l| serde_json::from_str(l).ok()).collect();
        for v in lines.iter().flatten() {
            let Some(content) = v.pointer("/message/content") else {
                continue;
            };
            if let Value::Array(items) = content {
                for it in items {
                    if it.get("type").and_then(Value::as_str) == Some("tool_result") {
                        if let Some(id) = it.get("tool_use_id").and_then(Value::as_str) {
                            results.insert(
                                id.to_string(),
                                value_text(it.get("content").unwrap_or(&Value::Null)),
                            );
                        }
                    }
                }
            }
        }
        for (i, v) in lines.iter().enumerate() {
            let Some(v) = v else { continue };
            let kind = match v.get("type").and_then(Value::as_str) {
                Some("user") => Kind::User,
                Some("assistant") => Kind::Assistant,
                _ => continue,
            };
            let time = v
                .get("timestamp")
                .and_then(Value::as_str)
                .unwrap_or("")
                .to_string();
            let sess = v
                .get("sessionId")
                .and_then(Value::as_str)
                .map(|s| s[..s.len().min(16)].to_string())
                .unwrap_or_else(|| session.to_string());
            let Some(content) = v.pointer("/message/content") else {
                continue;
            };
            match content {
                Value::String(s) => self.push(
                    &mut out,
                    kind,
                    "claude",
                    &sess,
                    &time,
                    source,
                    i + 1,
                    s.clone(),
                ),
                Value::Array(items) => {
                    for it in items {
                        match it.get("type").and_then(Value::as_str) {
                            Some("text") => {
                                let t = it.get("text").and_then(Value::as_str).unwrap_or("");
                                self.push(
                                    &mut out,
                                    kind,
                                    "claude",
                                    &sess,
                                    &time,
                                    source,
                                    i + 1,
                                    t.to_string(),
                                );
                            }
                            Some("tool_use") => {
                                let id = it
                                    .get("id")
                                    .and_then(Value::as_str)
                                    .unwrap_or("")
                                    .to_string();
                                let name = it
                                    .get("name")
                                    .and_then(Value::as_str)
                                    .unwrap_or("tool")
                                    .to_string();
                                let input = it.get("input").cloned().unwrap_or(Value::Null);
                                let text = render_tool(
                                    &name,
                                    &input,
                                    results.get(&id).map(String::as_str),
                                );
                                self.push(
                                    &mut out,
                                    Kind::Tool,
                                    "claude",
                                    &sess,
                                    &time,
                                    source,
                                    i + 1,
                                    text,
                                );
                            }
                            _ => {}
                        }
                    }
                }
                _ => {}
            }
        }
        out
    }

    fn cut_codex(&self, raw: &str, source: &str, session: &str) -> Vec<Record> {
        let mut out = Vec::new();
        let mut outputs: HashMap<String, String> = HashMap::new();
        let lines: Vec<Option<Value>> = raw.lines().map(|l| serde_json::from_str(l).ok()).collect();
        for v in lines.iter().flatten() {
            if v.pointer("/payload/type").and_then(Value::as_str) == Some("function_call_output") {
                if let (Some(id), Some(o)) = (
                    v.pointer("/payload/call_id").and_then(Value::as_str),
                    v.pointer("/payload/output"),
                ) {
                    outputs.insert(id.to_string(), value_text(o));
                }
            }
        }
        for (i, v) in lines.iter().enumerate() {
            let Some(v) = v else { continue };
            if v.get("type").and_then(Value::as_str) != Some("response_item") {
                continue;
            }
            let time = v
                .get("timestamp")
                .and_then(Value::as_str)
                .unwrap_or("")
                .to_string();
            let p = &v["payload"];
            match p.get("type").and_then(Value::as_str) {
                Some("message") => {
                    let kind = match p.get("role").and_then(Value::as_str) {
                        Some("user") => Kind::User,
                        Some("assistant") => Kind::Assistant,
                        _ => continue,
                    };
                    let text = value_text(p.get("content").unwrap_or(&Value::Null));
                    self.push(&mut out, kind, "codex", session, &time, source, i + 1, text);
                }
                Some("function_call") | Some("custom_tool_call") => {
                    let name = p.get("name").and_then(Value::as_str).unwrap_or("tool");
                    let input = p
                        .get("arguments")
                        .and_then(Value::as_str)
                        .and_then(|s| serde_json::from_str::<Value>(s).ok())
                        .or_else(|| p.get("input").cloned())
                        .unwrap_or(Value::Null);
                    let id = p.get("call_id").and_then(Value::as_str).unwrap_or("");
                    let text = render_tool(name, &input, outputs.get(id).map(String::as_str));
                    self.push(
                        &mut out,
                        Kind::Tool,
                        "codex",
                        session,
                        &time,
                        source,
                        i + 1,
                        text,
                    );
                }
                _ => {}
            }
        }
        out
    }
}

pub fn walk_jsonl(root: &Path, max_depth: usize) -> Vec<PathBuf> {
    fn go(dir: &Path, depth: usize, max: usize, out: &mut Vec<PathBuf>) {
        let Ok(rd) = std::fs::read_dir(dir) else {
            return;
        };
        for e in rd.flatten() {
            let p = e.path();
            if p.is_dir() {
                if depth < max {
                    go(&p, depth + 1, max, out);
                }
            } else if p.extension().is_some_and(|x| x == "jsonl") {
                out.push(p);
            }
        }
    }
    let mut out = Vec::new();
    go(root, 0, max_depth, &mut out);
    out.sort();
    out
}

pub fn dedupe(records: Vec<Record>) -> Vec<Record> {
    let mut seen = std::collections::HashSet::new();
    records
        .into_iter()
        .filter(|r| seen.insert(r.id.clone()))
        .collect()
}
