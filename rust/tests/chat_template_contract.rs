use std::path::{Path, PathBuf};
use std::sync::Arc;

use serde_json::json;
use tokio::sync::mpsc;

use speaches_plus::oapi::chat::{
    render_chat_checked, rewrite_native_tool_calls, template_required_from, ChatEngine, ChatEvent,
    ChatGenerateRequest, ChatMessageIn, FunctionCall, FunctionDef, MessageContent, Tool, ToolCall,
    ToolChoice,
};
use speaches_plus::oapi::chat_template::{load_attempt_for, load_was_attempted, ChatTemplate};

const NVFP4_31B: &str = "nvidia/Gemma-4-31B-IT-NVFP4";
const SNAP_NVFP4_MAIN: &str = "e5ef03afa233c35cb000323ff098d4291e1dd07c";

fn hub_roots() -> Vec<PathBuf> {
    let mut out: Vec<PathBuf> = Vec::new();
    if let Ok(v) = std::env::var("HF_HUB_CACHE") {
        out.push(PathBuf::from(v));
    }
    let home = std::env::var("HOME").unwrap_or_else(|_| "/root".into());
    out.push(PathBuf::from(home).join(".cache/huggingface/hub"));
    out
}

fn hub_root() -> PathBuf {
    hub_roots().pop().expect("HOME root is always pushed")
}

fn snapshot_dir(repo: &str, snap: &str) -> Option<PathBuf> {
    hub_roots().into_iter().find_map(|root| {
        let d = root
            .join(format!("models--{}", repo.replace('/', "--")))
            .join("snapshots")
            .join(snap);
        d.join("chat_template.jinja").is_file().then_some(d)
    })
}

fn scratch(tag: &str) -> PathBuf {
    let mut p = std::env::var("TMPDIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| hub_root().parent().unwrap().join("nvk-tmp"));
    p.push(format!("tmplcontract-{tag}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&p);
    std::fs::create_dir_all(&p).unwrap();
    p
}

fn model_dir(tag: &str, template: &str) -> PathBuf {
    let d = scratch(tag);
    std::fs::write(d.join("chat_template.jinja"), template).unwrap();
    std::fs::write(
        d.join("tokenizer_config.json"),
        r#"{"bos_token":"<bos>","eos_token":"<eos>"}"#,
    )
    .unwrap();
    d
}

fn key_of(dir: &Path) -> String {
    dir.file_name().unwrap().to_str().unwrap().to_string()
}

fn user(text: &str) -> ChatMessageIn {
    ChatMessageIn {
        role: "user".into(),
        content: Some(MessageContent::Text(text.into())),
        ..Default::default()
    }
}

fn weather_tool() -> Tool {
    Tool {
        kind: "function".into(),
        function: FunctionDef {
            name: "get_weather".into(),
            description: Some("Look up the weather".into()),
            parameters: Some(json!({
                "type": "object",
                "properties": { "city": { "type": "string", "description": "City name" } },
                "required": ["city"]
            })),
        },
    }
}

struct Engine {
    id: String,
    template: Option<Arc<ChatTemplate>>,
}

impl Engine {
    fn with_template(dir: &Path) -> Self {
        Engine {
            id: key_of(dir),
            template: Some(ChatTemplate::load(dir).expect("template loads")),
        }
    }
    fn without_template(dir: &Path) -> Self {
        assert!(ChatTemplate::load(dir).is_none());
        Engine {
            id: key_of(dir),
            template: None,
        }
    }
}

#[async_trait::async_trait]
impl ChatEngine for Engine {
    fn model_id(&self) -> &str {
        &self.id
    }
    fn official_template(&self) -> Option<&ChatTemplate> {
        self.template.as_deref()
    }
    async fn generate(
        &self,
        _req: ChatGenerateRequest,
        _tx: mpsc::Sender<ChatEvent>,
    ) -> anyhow::Result<()> {
        Ok(())
    }
}

const TURN_TEMPLATE: &str = "{{ bos_token }}{% for m in messages %}<|turn>{{ m['role'] }}\n{{ m['content'] }}<turn|>\n{% endfor %}{% if add_generation_prompt %}<|turn>model\n{% endif %}";

const TURN_TEMPLATE_WITH_TOOLS: &str = "{{ bos_token }}{% if tools %}<|turn>system\n{% for t in tools %}<|tool>declaration:{{ t['function']['name'] }}<tool|>{% endfor %}<turn|>\n{% endif %}{% for m in messages %}<|turn>{{ m['role'] }}\n{{ m['content'] }}<turn|>\n{% endfor %}{% if add_generation_prompt %}<|turn>model\n{% endif %}";

#[test]
fn a_model_with_a_shipped_template_renders_through_it() {
    let dir = model_dir("plain", TURN_TEMPLATE);
    let e = Engine::with_template(&dir);
    let out = render_chat_checked(&e, &[user("What is 2+2?")], &[], &ToolChoice::Auto, true)
        .expect("render");
    println!("{out:?}");
    assert_eq!(out, "<bos><|turn>user\nWhat is 2+2?<turn|>\n<|turn>model\n");
    assert!(
        !out.contains("<|im_start|>"),
        "ChatML must not appear: {out}"
    );
}

#[test]
fn tools_reach_the_official_template() {
    let dir = model_dir("tools", TURN_TEMPLATE_WITH_TOOLS);
    let e = Engine::with_template(&dir);
    assert!(e.official_template().unwrap().supports_tools());
    let out = render_chat_checked(
        &e,
        &[user("Weather in Oslo?")],
        &[weather_tool()],
        &ToolChoice::Auto,
        true,
    )
    .expect("render");
    println!("{out:?}");
    assert!(
        out.contains("<|tool>declaration:get_weather<tool|>"),
        "the jinja `tools` variable must be populated: {out}"
    );
    assert!(
        !out.contains("You can call tools."),
        "no synthetic system message once the template handles tools: {out}"
    );
    assert!(
        !out.contains("<tool_call>{\"name\": <tool>"),
        "the untrained tool-call syntax must not be requested: {out}"
    );
}

#[test]
fn a_forced_tool_choice_still_reaches_the_model_without_the_untrained_syntax() {
    let dir = model_dir("forced", TURN_TEMPLATE_WITH_TOOLS);
    let e = Engine::with_template(&dir);
    let out = render_chat_checked(
        &e,
        &[user("Weather in Oslo?")],
        &[weather_tool()],
        &ToolChoice::Function("get_weather".into()),
        true,
    )
    .expect("render");
    println!("{out:?}");
    assert!(
        out.contains("You must call the declared tool `get_weather` before answering."),
        "tool_choice survives the move off the synthetic preamble: {out}"
    );
    assert!(
        out.contains("<|tool>declaration:get_weather<tool|>"),
        "{out}"
    );
    assert!(!out.contains("You can call tools."), "{out}");
}

#[test]
fn a_template_without_tools_support_keeps_the_flattened_fallback() {
    let dir = model_dir("notools", TURN_TEMPLATE);
    let e = Engine::with_template(&dir);
    assert!(!e.official_template().unwrap().supports_tools());
    let out = render_chat_checked(
        &e,
        &[user("Weather in Oslo?")],
        &[weather_tool()],
        &ToolChoice::Auto,
        true,
    )
    .expect("render");
    println!("{out:?}");
    assert!(
        out.contains("You can call tools."),
        "flattening is the documented fallback for templates with no tools block: {out}"
    );
    assert!(
        out.starts_with("<bos><|turn>system\n"),
        "the flattened preamble still goes through the official template: {out}"
    );
    assert!(!out.contains("<|im_start|>"), "still never ChatML: {out}");
}

#[test]
fn the_chatml_fallback_does_not_fire_silently() {
    let dir = scratch("missing");
    std::fs::write(dir.join("tokenizer_config.json"), r#"{"bos_token":"<s>"}"#).unwrap();
    let e = Engine::without_template(&dir);
    assert!(load_was_attempted(e.model_id()));

    let err = render_chat_checked(&e, &[user("hi")], &[], &ToolChoice::Auto, true)
        .expect_err("a model-backed engine with no template must refuse");
    println!("{err}");
    assert!(err.contains(e.model_id()), "names the model: {err}");
    assert!(
        err.contains(dir.to_str().unwrap()),
        "names the directory searched: {err}"
    );
    assert!(
        err.contains("NV_ALLOW_CHATML_FALLBACK=1"),
        "names the opt-out: {err}"
    );

    let out = render_chat_checked(&e, &[user("hi")], &[], &ToolChoice::Auto, false)
        .expect("the explicit opt-out still serves ChatML");
    assert_eq!(
        out,
        "<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n"
    );
}

#[test]
fn a_failing_template_render_is_not_downgraded_to_chatml() {
    let dir = model_dir(
        "raises",
        "{{ raise_exception('this conversation shape is unsupported') }}",
    );
    let e = Engine::with_template(&dir);
    let err = render_chat_checked(&e, &[user("hi")], &[], &ToolChoice::Auto, true)
        .expect_err("a render failure must not silently serve ChatML");
    println!("{err}");
    assert!(
        err.contains("official chat template render failed"),
        "the reason is carried: {err}"
    );
    let out = render_chat_checked(&e, &[user("hi")], &[], &ToolChoice::Auto, false)
        .expect("opt-out still falls back");
    assert!(out.starts_with("<|im_start|>"), "{out}");
}

#[test]
fn missing_template_is_fatal_by_default_for_model_backed_engines() {
    assert!(
        template_required_from(None, None, true),
        "an engine that loaded a model directory refuses by default"
    );
    assert!(
        !template_required_from(None, None, false),
        "an engine that never loaded a model directory keeps the fallback"
    );
    assert!(
        !template_required_from(None, Some("1"), true),
        "NV_ALLOW_CHATML_FALLBACK=1 is the opt-out"
    );
    assert!(
        !template_required_from(Some("0"), None, true),
        "NV_REQUIRE_CHAT_TEMPLATE=0 keeps its legacy opt-out meaning"
    );
    assert!(
        template_required_from(Some("1"), Some("1"), false),
        "NV_REQUIRE_CHAT_TEMPLATE=1 wins and applies even to stub engines"
    );
    for on in ["1", "true", "yes", "on", "TRUE"] {
        assert!(!template_required_from(None, Some(on), true), "{on}");
    }
    for off in ["", "0", "false", "no", "off", "garbage"] {
        assert!(template_required_from(None, Some(off), true), "{off}");
    }
}

#[test]
fn template_load_attempts_are_recorded_for_hits_and_misses() {
    let hit = model_dir("attempt-ok", TURN_TEMPLATE);
    assert!(ChatTemplate::load(&hit).is_some());
    let miss = scratch("attempt-miss");
    std::fs::write(miss.join("tokenizer_config.json"), r#"{"bos_token":"<s>"}"#).unwrap();
    assert!(ChatTemplate::load(&miss).is_none());

    let ok = load_attempt_for(&key_of(&hit)).expect("hit recorded");
    assert_eq!(ok.dir, hit);
    assert!(ok.error.is_none());
    let bad = load_attempt_for(&key_of(&miss)).expect("miss recorded");
    assert_eq!(bad.dir, miss);
    assert!(bad
        .error
        .unwrap()
        .contains("neither chat_template.jinja nor tokenizer_config.json:chat_template"));
    assert!(!load_was_attempted("a-model-that-was-never-loaded"));
}

#[test]
fn native_gemma_tool_calls_are_rewritten_for_the_openai_parser() {
    let raw = "<|tool_call>call:get_weather{city:<|\"|>Oslo<|\"|>,days:3,metric:true}<tool_call|>";
    let rewritten = rewrite_native_tool_calls(raw).expect("native form is recognised");
    println!("{rewritten}");
    assert!(rewritten.starts_with("<tool_call>") && rewritten.ends_with("</tool_call>"));
    let parsed = speaches_plus::oapi::tool_parse::parse_tool_calls(&rewritten, None);
    assert_eq!(parsed.tool_calls.len(), 1);
    assert_eq!(parsed.tool_calls[0].function.name, "get_weather");
    let args: serde_json::Value =
        serde_json::from_str(&parsed.tool_calls[0].function.arguments).unwrap();
    assert_eq!(args["city"], "Oslo");
    assert_eq!(args["days"], 3);
    assert_eq!(args["metric"], true);
    assert!(parsed.content.is_none());
}

#[test]
fn native_tool_call_rewriting_handles_prose_nesting_and_non_matches() {
    assert!(rewrite_native_tool_calls("just an answer").is_none());
    assert!(
        rewrite_native_tool_calls(r#"<tool_call>{"name":"f","arguments":{}}</tool_call>"#)
            .is_none(),
        "the OpenAI form is left for the existing parser"
    );

    let two = concat!(
        "let me check. ",
        "<|tool_call>call:a{x:<|\"|>1<|\"|>}<tool_call|>",
        "<|tool_call>call:b{nested:{k:<|\"|>v<|\"|>},list:[<|\"|>p<|\"|>,<|\"|>q<|\"|>]}<tool_call|>"
    );
    let rewritten = rewrite_native_tool_calls(two).unwrap();
    println!("{rewritten}");
    let parsed = speaches_plus::oapi::tool_parse::parse_tool_calls(&rewritten, None);
    assert_eq!(parsed.tool_calls.len(), 2);
    assert_eq!(parsed.content.as_deref(), Some("let me check."));
    assert_eq!(parsed.tool_calls[1].function.name, "b");
    let args: serde_json::Value =
        serde_json::from_str(&parsed.tool_calls[1].function.arguments).unwrap();
    assert_eq!(args["nested"]["k"], "v");
    assert_eq!(args["list"], json!(["p", "q"]));
}

#[test]
fn real_gemma4_declares_tools_natively_and_round_trips_a_call() {
    let Some(dir) = snapshot_dir(NVFP4_31B, SNAP_NVFP4_MAIN) else {
        eprintln!("SKIP: {NVFP4_31B}@{SNAP_NVFP4_MAIN} not cached");
        return;
    };
    let e = Engine::with_template(&dir);
    assert!(
        e.official_template().unwrap().supports_tools(),
        "the Gemma 4 template has a tools block"
    );

    let out = render_chat_checked(
        &e,
        &[user("Weather in Oslo?")],
        &[weather_tool()],
        &ToolChoice::Auto,
        true,
    )
    .expect("render");
    println!("--- gemma4 tools render ---\n{out}\n--- end ---");
    assert!(
        out.contains("<|tool>declaration:get_weather"),
        "native declaration block: {out}"
    );
    assert!(!out.contains("You can call tools."), "{out}");
    assert!(!out.contains("<|im_start|>"), "{out}");
    assert!(
        out.ends_with("<|turn>model\n<|channel>thought\n<channel|>"),
        "generation prompt ends with a pre-closed thought channel: {out:?}"
    );

    let history = vec![
        user("Weather in Oslo?"),
        ChatMessageIn {
            role: "assistant".into(),
            content: None,
            tool_calls: Some(vec![ToolCall {
                index: None,
                id: "call_1".into(),
                kind: "function".into(),
                function: FunctionCall {
                    name: "get_weather".into(),
                    arguments: r#"{"city":"Oslo"}"#.into(),
                },
            }]),
            tool_call_id: None,
            name: None,
        },
        ChatMessageIn {
            role: "tool".into(),
            content: Some(MessageContent::Text("12C and raining".into())),
            tool_calls: None,
            tool_call_id: Some("call_1".into()),
            name: Some("get_weather".into()),
        },
    ];
    let out = render_chat_checked(&e, &history, &[weather_tool()], &ToolChoice::Auto, true)
        .expect("render history");
    println!("--- gemma4 tool history ---\n{out}\n--- end ---");
    assert!(
        out.contains("<|tool_call>call:get_weather{city:<|\"|>Oslo<|\"|>}<tool_call|>"),
        "assistant tool_calls render in the native form: {out}"
    );
    assert!(
        out.contains("<|tool_response>response:get_weather{"),
        "role:tool messages reach the template's native tool_response block: {out}"
    );
    assert!(
        !out.contains("Tool result ("),
        "tool results are no longer flattened into user text: {out}"
    );

    let emitted = out
        .split("<|tool_call>")
        .nth(1)
        .map(|s| format!("<|tool_call>{}", s.split("<tool_call|>").next().unwrap()))
        .map(|s| format!("{s}<tool_call|>"))
        .unwrap();
    let rewritten = rewrite_native_tool_calls(&emitted).expect("round trip");
    let parsed = speaches_plus::oapi::tool_parse::parse_tool_calls(&rewritten, None);
    assert_eq!(parsed.tool_calls.len(), 1);
    assert_eq!(parsed.tool_calls[0].function.name, "get_weather");
    assert_eq!(
        parsed.tool_calls[0].function.arguments,
        r#"{"city":"Oslo"}"#
    );
}
