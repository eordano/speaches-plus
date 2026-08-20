#![allow(dead_code)]

use std::path::{Path, PathBuf};

use minijinja::{context, Environment, Value as JinjaValue};
use serde_json::{json, Value};

pub fn token_str(v: &Value) -> Option<String> {
    match v {
        Value::String(s) => Some(s.clone()),
        Value::Object(o) => o.get("content").and_then(|c| c.as_str()).map(String::from),
        _ => None,
    }
}

fn raise_exception(msg: String) -> Result<JinjaValue, minijinja::Error> {
    Err(minijinja::Error::new(
        minijinja::ErrorKind::InvalidOperation,
        msg,
    ))
}

fn strftime_now(_fmt: String) -> Result<JinjaValue, minijinja::Error> {
    Ok(JinjaValue::from("1970-01-01"))
}

pub struct OfficialTemplate {
    env: Environment<'static>,
    pub bos: String,
    pub eos: String,
    pub source: String,
    pub source_path: PathBuf,
}

impl OfficialTemplate {
    pub fn try_load(dir: &Path) -> Result<Self, String> {
        let path = dir.join("chat_template.jinja");
        let source =
            std::fs::read_to_string(&path).map_err(|e| format!("read {}: {e}", path.display()))?;
        let cfg_path = dir.join("tokenizer_config.json");
        let cfg_raw = std::fs::read_to_string(&cfg_path)
            .map_err(|e| format!("read {}: {e}", cfg_path.display()))?;
        let cfg: Value = serde_json::from_str(&cfg_raw)
            .map_err(|e| format!("parse {}: {e}", cfg_path.display()))?;
        Self::build(path, source, &cfg)
    }

    pub fn load(dir: &Path) -> Self {
        Self::try_load(dir).unwrap_or_else(|e| panic!("{e}"))
    }

    pub fn load_lenient(dir: &Path) -> Self {
        let path = dir.join("chat_template.jinja");
        let source = std::fs::read_to_string(&path)
            .unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
        let cfg: Value = std::fs::read_to_string(dir.join("tokenizer_config.json"))
            .ok()
            .and_then(|s| serde_json::from_str(&s).ok())
            .unwrap_or(Value::Null);
        Self::build(path, source, &cfg).unwrap_or_else(|e| panic!("{e}"))
    }

    fn build(path: PathBuf, source: String, cfg: &Value) -> Result<Self, String> {
        let bos = cfg.get("bos_token").and_then(token_str).unwrap_or_default();
        let eos = cfg.get("eos_token").and_then(token_str).unwrap_or_default();
        let mut env = Environment::new();
        env.set_unknown_method_callback(minijinja_contrib::pycompat::unknown_method_callback);
        env.add_function("raise_exception", raise_exception);
        env.add_function("strftime_now", strftime_now);
        env.add_template_owned("chat", source.clone())
            .map_err(|e| format!("{} does not compile: {e:#}", path.display()))?;
        Ok(Self {
            env,
            bos,
            eos,
            source,
            source_path: path,
        })
    }

    fn render_core(
        &self,
        messages: &Value,
        tools: Option<&Value>,
        add_generation_prompt: bool,
        thinking: Option<bool>,
    ) -> Result<String, minijinja::Error> {
        let tmpl = self.env.get_template("chat").expect("get chat template");
        let tools_val = match tools {
            Some(t) => JinjaValue::from_serialize(t),
            None => JinjaValue::from(()),
        };
        let think = match thinking {
            Some(b) => JinjaValue::from(b),
            None => JinjaValue::UNDEFINED,
        };
        tmpl.render(context! {
            messages => JinjaValue::from_serialize(messages),
            tools => tools_val,
            add_generation_prompt => add_generation_prompt,
            bos_token => self.bos.clone(),
            eos_token => self.eos.clone(),
            enable_thinking => think,
        })
    }

    pub fn try_render(
        &self,
        messages: &Value,
        add_generation_prompt: bool,
    ) -> Result<String, String> {
        self.render_core(messages, None, add_generation_prompt, None)
            .map_err(|e| format!("render {}: {e:#}", self.source_path.display()))
    }

    pub fn render(
        &self,
        messages: &Value,
        tools: Option<&Value>,
        thinking: Option<bool>,
    ) -> String {
        self.render_core(messages, tools, true, thinking)
            .unwrap_or_else(|e| format!("<<RENDER ERROR: {e:#}>>"))
    }

    pub fn render_user(&self, user: &str) -> String {
        let msgs = json!([{ "role": "user", "content": user }]);
        self.render_core(&msgs, None, true, None)
            .unwrap_or_else(|e| panic!("render {}: {e:#}", self.source_path.display()))
    }
}

const GOLDEN_TEMPLATE: &str = "{{ bos_token }}{% for m in messages %}<|turn>{{ 'model' if m.role == 'assistant' else m.role.strip() }}\n{{ m.content }}<turn|>\n{% endfor %}{% if add_generation_prompt %}<|turn>model\n{% endif %}";

const GOLDEN_RENDER: &str =
    "<bos><|turn>user\nWhat is 2+2?<turn|>\n<|turn>model\n4<turn|>\n<|turn>user\nAnd times 3?<turn|>\n<|turn>model\n";

fn golden_dir(test: &str, template: &str, tokenizer_config: Option<&str>) -> PathBuf {
    let dir = std::env::temp_dir().join(format!(
        "nv-official-template-{}-{test}",
        std::process::id()
    ));
    std::fs::create_dir_all(&dir).expect("create golden temp dir");
    std::fs::write(dir.join("chat_template.jinja"), template).expect("write template");
    if let Some(cfg) = tokenizer_config {
        std::fs::write(dir.join("tokenizer_config.json"), cfg).expect("write tokenizer_config");
    }
    dir
}

#[test]
fn golden_turn_markers_render_byte_identically() {
    let dir = golden_dir(
        "golden",
        GOLDEN_TEMPLATE,
        Some(r#"{ "bos_token": { "content": "<bos>" }, "eos_token": "<turn|>" }"#),
    );
    let msgs = json!([
        { "role": "user", "content": "What is 2+2?" },
        { "role": "assistant", "content": "4" },
        { "role": "user", "content": "And times 3?" },
    ]);
    let strict = OfficialTemplate::load(&dir);
    assert_eq!(strict.bos, "<bos>");
    assert_eq!(strict.eos, "<turn|>");
    assert_eq!(strict.try_render(&msgs, true).unwrap(), GOLDEN_RENDER);
    assert_eq!(strict.render(&msgs, None, None), GOLDEN_RENDER);
    assert_eq!(
        strict.render_user("What is 2+2?"),
        "<bos><|turn>user\nWhat is 2+2?<turn|>\n<|turn>model\n"
    );
    let lenient = OfficialTemplate::load_lenient(&dir);
    assert_eq!(lenient.render(&msgs, None, None), GOLDEN_RENDER);
    assert_eq!(
        strict.try_render(&msgs, false).unwrap(),
        GOLDEN_RENDER.trim_end_matches("<|turn>model\n")
    );
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn lenient_load_defaults_bos_empty_without_tokenizer_config() {
    let dir = golden_dir("lenient", GOLDEN_TEMPLATE, None);
    let t = OfficialTemplate::load_lenient(&dir);
    assert_eq!(t.bos, "");
    assert_eq!(t.eos, "");
    assert!(t.render_user("hi").starts_with("<|turn>user\nhi<turn|>\n"));
    assert!(OfficialTemplate::try_load(&dir).is_err());
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn lenient_render_reports_errors_inline() {
    let dir = golden_dir(
        "raise",
        "{{ raise_exception('boom') }}",
        Some(r#"{ "bos_token": "<bos>", "eos_token": "<eos>" }"#),
    );
    let t = OfficialTemplate::load_lenient(&dir);
    let out = t.render(&json!([{ "role": "user", "content": "x" }]), None, None);
    assert!(
        out.starts_with("<<RENDER ERROR:"),
        "lenient render must inline errors, got {out:?}"
    );
    std::fs::remove_dir_all(&dir).ok();
}
