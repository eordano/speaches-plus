#![allow(dead_code)]

#[path = "common/chat_eval_core.rs"]
mod harness_self_test_no_server_code;

use harness_self_test_no_server_code::*;
use serde_json::{json, Value};
use speaches_plus::oapi::chat_template::ChatTemplate;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use tokenizers::Tokenizer;

pub const GEMMA_31B: &str = "google/gemma-4-31B-it";
pub const GEMMA_26B: &str = "google/gemma-4-26B-A4B-it";
pub const GEMMA_E4B: &str = "google/gemma-4-E4B-it";
pub const NVFP4_31B: &str = "nvidia/Gemma-4-31B-IT-NVFP4";
pub const QWEN36: &str = "RedHatAI/Qwen3.6-35B-A3B-NVFP4";
pub const GPT_OSS: &str = "openai/gpt-oss-20b";

pub const SNAP_NVFP4_TOOLS_OLD: &str = "1365cf7aa2de42546878b8d2e4a425019a0be514";
pub const SNAP_NVFP4_TOOLS_NEW: &str = "e5ef03afa233c35cb000323ff098d4291e1dd07c";

pub fn open_nvfp4_preferring_authoritative_like_fp8_contract_prompts_gemma4_nvfp4_dir(
) -> anyhow::Result<ChatEvalModel> {
    ChatEvalModel::open_pinned(NVFP4_31B, SNAP_NVFP4_TOOLS_NEW).or_else(|new_err| {
        ChatEvalModel::open_pinned(NVFP4_31B, SNAP_NVFP4_TOOLS_OLD).map_err(|old_err| {
            anyhow::anyhow!(
                "tests that need ONE pinned NVFP4 snapshot take whichever of the two known \
                 templates the flake-pinned corpus ships, authoritative first; neither is \
                 cached: {new_err}; {old_err}"
            )
        })
    })
}

pub const WHY_THE_RENDER_LOOKBACK_IS_BOUNDED_TO_THREE_LINES: &str =
    "A pack prompt's own render may be tokenized -- that IS the behaviour this gate enforces. \
     But a line-based scan cannot follow data flow, so it may only trust provenance it can SEE \
     on the line itself: a `.rendered` field read off a pack prompt, a visible `.render(` call, \
     or one within the three preceding lines -- a bounded lookback, because rustfmt splits `let \
     enc = tok` from its `.encode(`, and that is the furthest a line scan may honestly follow a \
     binding. Matching the bare word `rendered` instead exempted any local of that name, \
     including `tok.encode(&hand_rendered_prompt)` -- precisely what the gate exists to catch -- \
     and silenced three real sites.";

fn tokenizes_a_visible_render(line: &str, window: &[&str], needle: &str) -> bool {
    if needle != ".encode(" {
        return false;
    }
    line.contains(".rendered")
        || line.contains(".render(")
        || window.iter().any(|l| l.contains(".render("))
}

pub const CHATML_FALLBACK_MARKERS: [&str; 2] = ["<|im_start|>", "<|im_end|>"];

pub const GUESSED_GEMMA4_CHAT_WRAPPER: &str = "<|turn>user\n{}<turn|>\n<|turn>model\n";

pub fn hub_roots() -> Vec<PathBuf> {
    let mut out: Vec<PathBuf> = Vec::new();
    let mut push = |p: PathBuf| {
        if p.is_dir() && !out.contains(&p) {
            out.push(p);
        }
    };
    if let Ok(v) = std::env::var("NV_CHAT_EVAL_HUB") {
        push(PathBuf::from(v));
    }
    if let Ok(v) = std::env::var("HF_HUB_CACHE") {
        push(PathBuf::from(v));
    }
    push(PathBuf::from(std::env::var("HOME").unwrap_or_default()).join(".cache/huggingface/hub"));
    out
}

pub fn hub_root() -> PathBuf {
    hub_roots().first().cloned().unwrap_or_default()
}

pub fn snapshots_of(repo: &str) -> Vec<(String, PathBuf)> {
    let leaf = format!("models--{}", repo.replace('/', "--"));
    let mut out: Vec<(String, PathBuf)> = Vec::new();
    for root in hub_roots() {
        let d = root.join(&leaf).join("snapshots");
        if let Ok(rd) = std::fs::read_dir(&d) {
            for e in rd.flatten() {
                let p = e.path();
                let id = e.file_name().to_string_lossy().to_string();
                let usable = p.join("chat_template.jinja").exists()
                    || p.join("tokenizer_config.json").exists();
                if usable && !out.iter().any(|(i, _)| *i == id) {
                    out.push((id, p));
                }
            }
        }
    }
    out.sort_by(|a, b| a.0.cmp(&b.0));
    out
}

fn snapshot_path(repo: &str, snapshot: &str) -> Option<PathBuf> {
    snapshots_of(repo)
        .into_iter()
        .find(|(id, _)| id == snapshot)
        .map(|(_, p)| p)
}

pub struct ChatEvalModel {
    pub repo: String,
    pub snapshot: String,
    pub dir: PathBuf,
    pub template: Arc<ChatTemplate>,
    pub tokenizer: Tokenizer,
    pub stops: StopSet,
    pub template_digest: String,
    pub template_bytes: usize,
}

impl ChatEvalModel {
    pub fn open(repo: &str) -> anyhow::Result<Self> {
        let snaps = snapshots_of(repo);
        anyhow::ensure!(
            !snaps.is_empty(),
            "no cached snapshot for {repo} under {}",
            hub_root().display()
        );
        anyhow::ensure!(
            snaps.len() == 1,
            "{repo} has {} cached snapshots ({}). Auto-picking one by directory order is exactly \
             how two lanes measured different chat templates for the same model. Pin it with \
             ChatEvalModel::open_pinned; see {HARNESS_DOC}.",
            snaps.len(),
            snaps
                .iter()
                .map(|s| s.0.as_str())
                .collect::<Vec<_>>()
                .join(", ")
        );
        Self::open_pinned(repo, &snaps[0].0)
    }

    pub fn open_pinned(repo: &str, snapshot: &str) -> anyhow::Result<Self> {
        let dir = snapshot_path(repo, snapshot).ok_or_else(|| {
            anyhow::anyhow!(
                "no snapshot {snapshot} for {repo} in any hub root {:?}",
                hub_roots()
            )
        })?;
        let template = ChatTemplate::load(&dir)
            .ok_or_else(|| anyhow::anyhow!("{repo}@{snapshot} has no loadable chat template"))?;
        let tokenizer = Tokenizer::from_file(dir.join("tokenizer.json"))
            .map_err(|e| anyhow::anyhow!("tokenizer for {repo}@{snapshot}: {e}"))?;
        let stops = StopSet::from_generation_config(&dir)?;
        let (template_digest, template_bytes) = template_digest_of_dir(&dir)?;
        Ok(Self {
            repo: repo.to_string(),
            snapshot: snapshot.to_string(),
            dir,
            template,
            tokenizer,
            stops,
            template_digest,
            template_bytes,
        })
    }

    pub fn render(&self, messages: &Value, tools: Option<&Value>) -> anyhow::Result<String> {
        let mut kwargs = self.template.effective_template_kwargs();
        if self.template.declares_thinking_switch() {
            let on = std::env::var("NV_CHAT_EVAL_THINKING")
                .map(|v| v == "1")
                .unwrap_or(false);
            kwargs.insert("enable_thinking".into(), serde_json::json!(on));
        }
        self.template
            .render_with_kwargs(messages, tools, true, &kwargs)
    }

    pub fn prompt(
        &self,
        label: &str,
        kind: PromptKind,
        messages: &Value,
    ) -> anyhow::Result<TemplatedPrompt> {
        let rendered = self.render(messages, None)?;
        let enc = self
            .tokenizer
            .encode(rendered.as_str(), false)
            .map_err(|e| anyhow::anyhow!("encode {label}: {e}"))?;
        Ok(TemplatedPrompt::from_official_render(
            label,
            kind,
            &self.repo,
            &self.snapshot,
            &self.template_digest,
            self.template_bytes,
            rendered,
            enc.get_ids().to_vec(),
        ))
    }

    pub fn decode(&self, ids: &[u32]) -> String {
        self.tokenizer.decode(ids, false).unwrap_or_default()
    }

    pub fn label_of(&self, id: u32) -> String {
        self.tokenizer
            .id_to_token(id)
            .unwrap_or_else(|| format!("<{id}>"))
    }

    pub fn pack(&self, prompts: Vec<TemplatedPrompt>) -> PromptPack {
        PromptPack {
            model_repo: self.repo.clone(),
            snapshot: self.snapshot.clone(),
            template_digest: self.template_digest.clone(),
            template_bytes: self.template_bytes,
            stop_ids: self.stops.ids.clone(),
            stop_source: self.stops.source.clone(),
            prompts,
        }
    }
}

pub fn user(text: &str) -> Value {
    json!([{"role": "user", "content": text}])
}

pub fn system_user(sys: &str, text: &str) -> Value {
    json!([{"role": "system", "content": sys}, {"role": "user", "content": text}])
}

pub fn chat3(u1: &str, a1: &str, u2: &str) -> Value {
    json!([
        {"role": "user", "content": u1},
        {"role": "assistant", "content": a1},
        {"role": "user", "content": u2}
    ])
}

pub const LONGCTX_PASSAGE: &str = "In the spring of 1911 the survey vessel Meridian left Hobart \
     for the Balleny Islands, carrying a crew of twenty-eight, three whaleboats, and a portable \
     magnetometer. Her supply ship, the Aurora, was to follow six weeks later with coal and \
     replacement chronometers. The expedition's second officer, a Norwegian named Halvorsen, kept \
     the only complete meteorological log; it recorded forty-one days of gale out of the ninety \
     they spent south of the convergence. When the Meridian's rudder post cracked near Sturge \
     Island, the crew rigged a jury steering oar from a whaleboat mast and made for open water, \
     reaching the rendezvous nine days late. The Aurora was still there.";

pub const LONGCTX256_PASSAGE_TAIL: &str = "On the return leg the Meridian called at Macquarie \
     Island to land stores for the wireless relay station. Her manifest for that call listed \
     forty-one crates of glass instruments, twelve drums of lamp oil, a crated theodolite for the \
     plateau survey, and mail for the five men wintering over. The station's cook, who had been a \
     lighthouse keeper on Maatsuyker Island until his retirement in 1907, traded the shore party \
     fresh eggs for tobacco and two of the crated barometers, an exchange Halvorsen recorded in \
     the log with evident disapproval. From Macquarie the vessel worked north through heavy \
     westerlies, hove to twice off the Snares, and raised the Tasmanian coast on the morning of \
     the hundred and thirty-first day out of Hobart. The harbourmaster's launch met her inside \
     the heads with orders to proceed directly to the government quay, where the expedition's \
     patron, a wool broker who had underwritten the charter against the advice of his partners, \
     waited with a brass band that had learned exactly one march. The magnetometer readings, \
     sealed in soldered tins since Sturge Island, went by train to the observatory the same \
     afternoon; the jury steering oar went to the museum, where the label misattributed it to a \
     sealing brig for thirty years before a curator matched the mast fittings to the Meridian's \
     surviving whaleboat and corrected the record.";

pub fn standard_suite(m: &ChatEvalModel) -> anyhow::Result<Vec<TemplatedPrompt>> {
    Ok(vec![
        m.prompt(
            "control-arithmetic",
            PromptKind::Control,
            &user("What is 2 + 2? Reply with the number only."),
        )?,
        m.prompt(
            "control-capital",
            PromptKind::Control,
            &user("What is the capital of France? Reply with the city name only."),
        )?,
        m.prompt(
            "control-literal",
            PromptKind::Control,
            &user("Reply with exactly the word BANANA and nothing else."),
        )?,
        m.prompt(
            "control-prose-proverb",
            PromptKind::Control,
            &user("Complete the proverb: \"A stitch in time saves ...\". Reply with the missing word only."),
        )?,
        m.prompt(
            "control-prose-antonym",
            PromptKind::Control,
            &user("What is the antonym of the word \"ascend\"? Reply with one word only."),
        )?,
        m.prompt(
            "control-chat-name",
            PromptKind::Control,
            &chat3(
                "My name is Alice and I live in Lyon.",
                "Nice to meet you, Alice. Lyon is a lovely city.",
                "What is my name? Reply with the name only.",
            ),
        )?,
        m.prompt(
            "control-chat-count",
            PromptKind::Control,
            &chat3(
                "I have 3 apples in my bag.",
                "Understood: you have 3 apples.",
                "I just bought 2 more. How many apples do I have now? Reply with the number only.",
            ),
        )?,
        m.prompt(
            "control-code-return",
            PromptKind::Control,
            &user(
                "What does this Python function return when called as f(3)?\n\ndef f(n):\n    \
                 return n * n + 1\n\nReply with the number only.",
            ),
        )?,
        m.prompt(
            "control-code-output",
            PromptKind::Control,
            &user("What does this Python line print?\n\nprint(len(\"kernel\"))\n\nReply with the number only."),
        )?,
        m.prompt(
            "control-longctx-ship",
            PromptKind::Control,
            &user(&format!(
                "{LONGCTX_PASSAGE}\n\nAccording to the passage, what was the name of the supply \
                 ship? Reply with the name only."
            )),
        )?,
        m.prompt(
            "control-longctx-days",
            PromptKind::Control,
            &user(&format!(
                "{LONGCTX_PASSAGE}\n\nAccording to the passage, how many days of gale did the \
                 log record? Reply with the number only."
            )),
        )?,
        m.prompt(
            "control-prefill256-cargo",
            PromptKind::Control,
            &user(&format!(
                "{LONGCTX_PASSAGE}\n\n{LONGCTX256_PASSAGE_TAIL}\n\nAccording to the passage, how \
                 many crates of glass instruments were on the Macquarie manifest? Reply with the \
                 number only."
            )),
        )?,
        m.prompt(
            "control-prefill256-year",
            PromptKind::Control,
            &user(&format!(
                "{LONGCTX_PASSAGE}\n\n{LONGCTX256_PASSAGE_TAIL}\n\nAccording to the passage, in \
                 what year did the station's cook retire from lighthouse keeping? Reply with the \
                 year only."
            )),
        )?,
        m.prompt(
            "openended-explain",
            PromptKind::OpenEnded,
            &user("Explain in two sentences why the sky is blue."),
        )?,
        m.prompt(
            "openended-list",
            PromptKind::OpenEnded,
            &user("List three prime numbers greater than 100."),
        )?,
        m.prompt(
            "openended-system",
            PromptKind::OpenEnded,
            &system_user("You are a terse assistant.", "Summarise what a GPU kernel is."),
        )?,
    ])
}

fn have(repo: &str) -> bool {
    !snapshots_of(repo).is_empty()
}

fn skip(repo: &str) -> bool {
    if have(repo) {
        return false;
    }
    eprintln!(
        "skipping {repo}: not cached in any hub root {:?}",
        hub_roots()
    );
    true
}

#[test]
fn hub_roots_are_reported_so_a_lane_cannot_silently_read_a_different_hub() {
    eprintln!(
        "NV_CHAT_EVAL_HUB={:?}\nHF_HUB_CACHE={:?}",
        std::env::var("NV_CHAT_EVAL_HUB").ok(),
        std::env::var("HF_HUB_CACHE").ok()
    );
    let roots = hub_roots();
    for (i, r) in roots.iter().enumerate() {
        eprintln!("hub root [{i}] {}", r.display());
    }
    for repo in [GEMMA_31B, GEMMA_26B, GEMMA_E4B, NVFP4_31B, QWEN36, GPT_OSS] {
        for (id, p) in snapshots_of(repo) {
            let which = roots
                .iter()
                .position(|r| p.starts_with(r))
                .map(|i| i.to_string())
                .unwrap_or_else(|| "?".into());
            let bytes = std::fs::metadata(p.join("chat_template.jinja"))
                .map(|m| m.len())
                .unwrap_or(0);
            eprintln!(
                "  {repo} @ {} from root[{which}]  template {bytes} bytes",
                &id[..8]
            );
        }
    }
    eprintln!(
        "NOTE: nvk.sh sources its cached devenv AFTER caller env, so HF_HUB_CACHE points at a \
         nix-store hub. That hub does NOT carry every snapshot in ~/.cache/huggingface/hub. This \
         harness searches every root and pins by snapshot id, so a lane can never silently read a \
         different template than the one it names."
    );
    assert!(!roots.is_empty(), "no hub root exists");
}

#[test]
fn hazard1_the_chatml_fallback_is_wrong_for_every_model_we_serve() {
    use speaches_plus::oapi::chat::{render_chat_prompt, ChatMessageIn, MessageContent};
    let msgs = vec![ChatMessageIn {
        role: "user".into(),
        content: Some(MessageContent::Text("What is 2 + 2?".into())),
        tool_calls: None,
        tool_call_id: None,
        name: None,
    }];
    let chatml = render_chat_prompt(&msgs);
    eprintln!("production ChatML fallback (chat.rs::render_chat_prompt):\n{chatml:?}");
    for m in CHATML_FALLBACK_MARKERS {
        assert!(
            chatml.contains(m),
            "fallback lost its ChatML markers: {chatml:?}"
        );
    }

    let mut checked = 0;
    let mut wholly_wrong = Vec::new();
    let mut subtly_wrong = Vec::new();
    for repo in [GEMMA_31B, GEMMA_26B, GEMMA_E4B, NVFP4_31B, QWEN36, GPT_OSS] {
        if skip(repo) {
            continue;
        }
        let snaps = snapshots_of(repo);
        let m = ChatEvalModel::open_pinned(repo, &snaps[0].0).unwrap();
        let official = m.render(&user("What is 2 + 2?"), None).unwrap();
        eprintln!(
            "--- {repo}@{} official render ---\n{official:?}",
            &m.snapshot[..8]
        );
        assert_ne!(
            official, chatml,
            "{repo} official render is byte-identical to the ChatML fallback"
        );
        if CHATML_FALLBACK_MARKERS.iter().any(|k| official.contains(k)) {
            subtly_wrong.push(repo);
        } else {
            wholly_wrong.push(repo);
        }
        checked += 1;
    }
    assert!(checked > 0, "no cached model to check");
    eprintln!(
        "HAZARD 1: chat.rs official_template() defaults to None and render_chat() then falls \
         through to render_chat_prompt()'s ChatML with no diagnostic.\n\
         Models where ChatML shares NO markers with the real template (catastrophic): {wholly_wrong:?}\n\
         Models whose real template is ChatML-SHAPED but still differs (subtle, and worse for \
         being plausible): {subtly_wrong:?}\n\
         For every cached model the fallback differs from the official render."
    );
    assert!(
        !wholly_wrong.is_empty(),
        "expected the Gemma4 family to share no ChatML markers"
    );
}

#[test]
fn there_is_no_such_thing_as_the_gemma4_format() {
    let mut seen: Vec<(&str, String)> = Vec::new();
    for repo in [GEMMA_31B, GEMMA_26B, GEMMA_E4B, NVFP4_31B] {
        if skip(repo) {
            continue;
        }
        let snaps = snapshots_of(repo);
        let m = ChatEvalModel::open_pinned(repo, &snaps[0].0).unwrap();
        let r = m.render(&user("q"), None).unwrap();
        eprintln!("{repo}@{}: {r:?}", &m.snapshot[..8]);
        seen.push((repo, r));
    }
    if seen.len() < 2 {
        eprintln!("need two Gemma4 variants cached to make the point");
        return;
    }
    let distinct: std::collections::BTreeSet<&String> = seen.iter().map(|(_, r)| r).collect();
    eprintln!(
        "{} Gemma4 variants produce {} DISTINCT generation prompts. Hardcoding one 'Gemma4 \
         format' is wrong for at least one of them; load each model's own template.",
        seen.len(),
        distinct.len()
    );
    assert!(
        distinct.len() > 1,
        "expected at least one Gemma4 variant to differ; all rendered {:?}",
        distinct
    );
}

#[test]
fn hazard1b_the_silent_none_path_stays_diagnosed() {
    let src =
        std::fs::read_to_string(Path::new(env!("CARGO_MANIFEST_DIR")).join("src/oapi/chat.rs"))
            .unwrap();
    assert!(
        src.contains("official chat template render failed; falling back"),
        "chat.rs no longer warns when render_official returns Err"
    );
    let default_body = src
        .find("fn official_template(&self) -> Option<&crate::oapi::chat_template::ChatTemplate> {")
        .expect("official_template default body moved");
    assert!(
        src[default_body..default_body + 200].contains("None"),
        "official_template default no longer returns None"
    );

    let rc = src.find("fn render_chat(").expect("render_chat moved");
    assert!(
        src[rc..rc + 200].contains("&[ChatMessageIn]"),
        "render_chat no longer takes &[ChatMessageIn] within its signature -- \
         re-anchor this guard to the real prompt-render entry point"
    );
    let body = &src[rc..rc + 1600];
    let none_arm = body
        .find("None =>")
        .map(|i| body[i..i + 160].to_string())
        .unwrap_or_default();
    let diagnosed = !none_arm.is_empty()
        && (none_arm.contains("warn")
            || none_arm.contains("log_missing_official_template")
            || none_arm.contains("tracing::"));
    eprintln!("render_chat None arm: {none_arm:?}");
    eprintln!(
        "default official_template() -> None; the None arm of render_chat is diagnosed: {diagnosed}"
    );
    assert!(
        diagnosed,
        "REGRESSION: render_chat() falls through to the ChatML built-in renderer with NO \
         diagnostic when official_template() returns None. That silent fallback is what produced \
         two false quality readings on 2026-08-06; see {HARNESS_DOC}."
    );
    let gated = src.contains("NV_REQUIRE_CHAT_TEMPLATE");
    eprintln!(
        "hard-fail env gate NV_REQUIRE_CHAT_TEMPLATE present: {gated}. Quality and conformance \
         runs should set it so a missing template cannot be served as ChatML."
    );
}

#[test]
fn hazard2_count_the_official_template_overrides_in_the_tree() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("src");
    let mut hits = Vec::new();
    let mut stack = vec![root.clone()];
    while let Some(d) = stack.pop() {
        for e in std::fs::read_dir(&d).unwrap().flatten() {
            let p = e.path();
            if p.is_dir() {
                stack.push(p);
            } else if p.extension().map(|x| x == "rs").unwrap_or(false) {
                let s = std::fs::read_to_string(&p).unwrap_or_default();
                if s.contains("fn official_template(") && !s.contains("trait ChatEngine") {
                    hits.push(p.strip_prefix(&root).unwrap().to_path_buf());
                }
            }
        }
    }
    hits.sort();
    eprintln!("official_template() overrides outside the trait default: {hits:?}");
    let engine = root.join("oapi/chat_engine.rs");
    let src = std::fs::read_to_string(&engine).unwrap();
    let cuda_gated =
        src.contains("#![cfg(feature = \"cuda\")]") || src.contains("#[cfg(feature = \"cuda\")]");
    eprintln!(
        "chat_engine.rs cuda-gated: {cuda_gated}. Any engine that does not override \
         official_template() serves ChatML."
    );
    assert!(
        !hits.is_empty(),
        "expected at least the chat_engine override"
    );
}

#[test]
fn hazard3_the_two_nvfp4_snapshots_agree_on_plain_chat_and_disagree_on_tools() {
    if skip(NVFP4_31B) {
        return;
    }
    let snaps = snapshots_of(NVFP4_31B);
    if snaps.len() < 2 {
        eprintln!(
            "only {} snapshot(s) cached; nothing to compare",
            snaps.len()
        );
        return;
    }
    let a = ChatEvalModel::open_pinned(NVFP4_31B, SNAP_NVFP4_TOOLS_OLD).unwrap();
    let b = ChatEvalModel::open_pinned(NVFP4_31B, SNAP_NVFP4_TOOLS_NEW).unwrap();
    eprintln!(
        "{}: {} bytes digest {}\n{}: {} bytes digest {}",
        &a.snapshot[..8],
        a.template_bytes,
        a.template_digest,
        &b.snapshot[..8],
        b.template_bytes,
        b.template_digest
    );
    assert_ne!(
        a.template_digest, b.template_digest,
        "templates are byte-identical"
    );

    let plain = [
        user("What is 2 + 2?"),
        system_user("You are terse.", "Explain gravity."),
        json!([
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
            {"role": "user", "content": "and now?"}
        ]),
    ];
    for (i, msgs) in plain.iter().enumerate() {
        let ra = a.render(msgs, None).unwrap();
        let rb = b.render(msgs, None).unwrap();
        assert_eq!(
            ra, rb,
            "plain-chat case {i} differs between snapshots:\n{ra:?}\n{rb:?}"
        );
    }
    eprintln!(
        "plain no-tools chat renders BYTE-IDENTICALLY across both snapshots ({} cases)",
        plain.len()
    );

    let tools = json!([{
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "weather",
            "parameters": {
                "type": "object",
                "properties": {
                    "loc": {"type": "string", "description": "city"},
                    "opts": {"type": "object", "properties": {"unit": {"type": "string"}}}
                },
                "required": ["loc"]
            }
        }
    }]);
    let ta = a.render(&user("weather in Paris?"), Some(&tools)).unwrap();
    let tb = b.render(&user("weather in Paris?"), Some(&tools)).unwrap();
    eprintln!("--- tools, {} ---\n{ta:?}", &a.snapshot[..8]);
    eprintln!("--- tools, {} ---\n{tb:?}", &b.snapshot[..8]);
    assert_ne!(ta, tb, "expected the tool path to differ between snapshots");
    eprintln!(
        "HAZARD 3 RESOLVED: the snapshots differ ONLY in tool-calling and multimodal placeholder \
         handling. For plain chat quality benchmarking either snapshot gives the same prompt, so \
         the earlier speed conclusion happens to carry -- but any tool-calling or image evaluation \
         MUST pin the snapshot. This harness pins it unconditionally."
    );
}

#[test]
fn hazard3b_open_refuses_to_auto_pick_among_two_snapshots() {
    if skip(NVFP4_31B) {
        return;
    }
    if snapshots_of(NVFP4_31B).len() < 2 {
        return;
    }
    let msg = match ChatEvalModel::open(NVFP4_31B) {
        Ok(_) => panic!("open() auto-picked a snapshot despite two being cached"),
        Err(e) => format!("{e}"),
    };
    eprintln!("{msg}");
    assert!(msg.contains("cached snapshots"), "{msg}");
    assert!(msg.contains("open_pinned"), "{msg}");
}

#[test]
fn stop_set_comes_from_generation_config_and_config_json_is_incomplete() {
    if skip(NVFP4_31B) {
        return;
    }
    let m = open_nvfp4_preferring_authoritative_like_fp8_contract_prompts_gemma4_nvfp4_dir()
        .unwrap();
    eprintln!("{}", m.stops);
    for id in &m.stops.ids {
        eprintln!("  stop {id} = {:?}", m.label_of(*id));
    }
    assert_eq!(m.stops.ids, vec![1, 106, 50], "Gemma4 eos_token_id changed");
    assert_eq!(m.label_of(106), "<turn|>");
    assert_eq!(m.label_of(1), "<eos>");
    assert_eq!(m.label_of(50), "<|tool_response>");

    let cfg: Value =
        serde_json::from_str(&std::fs::read_to_string(m.dir.join("config.json")).unwrap()).unwrap();
    let from_config: Vec<u32> = cfg
        .get("eos_token_id")
        .and_then(|v| v.as_array())
        .map(|a| {
            a.iter()
                .filter_map(|x| x.as_u64())
                .map(|x| x as u32)
                .collect()
        })
        .unwrap_or_default();
    eprintln!("config.json eos_token_id = {from_config:?} (INCOMPLETE)");
    assert_eq!(from_config, vec![1, 106]);
    assert!(
        !from_config.contains(&50),
        "config.json unexpectedly gained token 50"
    );
    eprintln!(
        "A harness that reads config.json misses token 50 <|tool_response>. StopSet reads \
         generation_config.json only."
    );
}

#[test]
fn the_guessed_gemma4_wrapper_used_by_a_benchmarking_lane_is_not_the_real_template() {
    if skip(NVFP4_31B) {
        return;
    }
    let m = open_nvfp4_preferring_authoritative_like_fp8_contract_prompts_gemma4_nvfp4_dir()
        .unwrap();
    let q = "What is 2 + 2?";
    let guessed = GUESSED_GEMMA4_CHAT_WRAPPER.replacen("{}", q, 1);
    let official = m.render(&user(q), None).unwrap();
    eprintln!("guessed  (nv-models/tests/cuda_fp8_freerun.rs::chat): {guessed:?}");
    eprintln!("official (chat_template.jinja):                        {official:?}");
    assert_ne!(guessed, official);

    let mut g_ids = m
        .tokenizer
        .encode(guessed.as_str(), false)
        .unwrap()
        .get_ids()
        .to_vec();
    let o_ids = m
        .tokenizer
        .encode(official.as_str(), false)
        .unwrap()
        .get_ids()
        .to_vec();
    assert_eq!(o_ids[0], 2, "official render must begin with <bos> (id 2)");
    assert_ne!(
        g_ids[0], 2,
        "the guessed wrapper STRING has no BOS; its call site adds one separately"
    );
    g_ids.insert(0, 2);
    eprintln!("guessed  ids (as the CALL SITE builds them, prompt_ids() prepends id 2): {g_ids:?}");
    eprintln!("official ids: {o_ids:?}");

    assert!(
        o_ids.starts_with(&g_ids),
        "expected the guessed call-site ids to be a strict prefix of the official ids"
    );
    let missing = &o_ids[g_ids.len()..];
    let missing_labels: Vec<String> = missing.iter().map(|t| m.label_of(*t)).collect();
    eprintln!(
        "residual delta at the call site: {} token(s) {missing:?} = {missing_labels:?}",
        missing.len()
    );
    assert!(
        official.contains("<|channel>thought"),
        "official render lost the thought channel: {official:?}"
    );
    assert!(
        !guessed.contains("<|channel>thought"),
        "the guessed wrapper unexpectedly has the thought channel"
    );
    assert!(
        !missing.is_empty(),
        "the guessed call site matched the official render exactly"
    );
    eprintln!(
        "HONEST STATEMENT OF THE DELTA: the guessed wrapper string omits the literal <bos>, but \
         cuda_fp8_freerun.rs::prompt_ids() prepends id 2 in code, so BOS is NOT actually missing at \
         the call site. The real residual is the '<|channel>thought\\n<channel|>' generation-prompt \
         suffix the template emits when enable_thinking is false. Small, but it is the last thing \
         the model reads before its first sampled token: it decides whether the model opens a \
         thought channel or answers directly, which is exactly the knife edge between continuing \
         and ending the turn."
    );
}

#[test]
fn the_channel_opener_is_per_size_so_no_wrapper_is_portable_across_checkpoints() {
    let q = "What is 2 + 2?";
    let expect: [(&str, bool); 3] = [(GEMMA_31B, true), (GEMMA_26B, true), (GEMMA_E4B, false)];
    let mut seen = 0;
    for (repo, wants_opener) in expect {
        if skip(repo) {
            continue;
        }
        let m = ChatEvalModel::open(repo).unwrap();
        let official = m.render(&user(q), None).unwrap();
        eprintln!(
            "{repo} @ {} ({} B template): {official:?}",
            m.snapshot, m.template_bytes
        );
        assert_eq!(
            official.ends_with("<|channel>thought\n<channel|>"),
            wants_opener,
            "{repo}'s generation prompt disagrees with the recorded per-size shape. The \
             gemma-4 VOCAB is shared across sizes but the TEMPLATES ARE NOT, so a wrapper \
             (hand-built or borrowed from another size) is off-distribution on at least one \
             of them: {official:?}"
        );
        seen += 1;
    }
    assert!(
        seen >= 2,
        "only {seen} gemma-4 size(s) cached, so the per-size difference this asserts was \
         never actually compared"
    );
}

#[test]
fn every_cached_model_renders_and_tokenizes_the_standard_suite() {
    let mut n = 0;
    for repo in [GEMMA_31B, GEMMA_26B, GEMMA_E4B, NVFP4_31B, QWEN36] {
        if skip(repo) {
            continue;
        }
        let snaps = snapshots_of(repo);
        let m = match ChatEvalModel::open_pinned(repo, &snaps[0].0) {
            Ok(m) => m,
            Err(e) => {
                eprintln!("{repo}: cannot open ({e})");
                continue;
            }
        };
        let prompts = standard_suite(&m).unwrap();
        eprintln!(
            "{repo}@{} template {} bytes, {}",
            &m.snapshot[..8],
            m.template_bytes,
            m.stops
        );
        for p in &prompts {
            eprintln!(
                "  {:<22} [{}] {:>4} ids  {:?}",
                p.label,
                p.kind,
                p.ids.len(),
                p.rendered
            );
            assert!(p.is_serving_shaped());
            assert_eq!(p.template_digest, m.template_digest);
        }
        assert!(prompts.iter().any(|p| p.kind == PromptKind::Control));
        n += 1;
    }
    assert!(n > 0);
}

#[test]
fn a_prompt_pack_round_trips_and_refuses_a_mismatched_snapshot() {
    if skip(NVFP4_31B) {
        return;
    }
    if snapshots_of(NVFP4_31B).len() < 2 {
        return;
    }
    let a = ChatEvalModel::open_pinned(NVFP4_31B, SNAP_NVFP4_TOOLS_OLD).unwrap();
    let pack = a.pack(standard_suite(&a).unwrap());
    let out = scratch_dir().join("pack-nvfp4-old.json");
    pack.write_json(&out).unwrap();
    eprintln!("wrote {}", out.display());

    let ok = PromptPack::load_for_snapshot(&out, &a.dir).unwrap();
    assert_eq!(ok.prompts.len(), pack.prompts.len());
    assert_eq!(ok.stop_ids, vec![1, 106, 50]);

    let b_dir = a.dir.parent().unwrap().join(SNAP_NVFP4_TOOLS_NEW);
    let err = PromptPack::load_for_snapshot(&out, &b_dir).unwrap_err();
    let msg = format!("{err}");
    eprintln!("{msg}");
    assert!(msg.contains("Re-render the pack"), "{msg}");
}

pub fn scratch_dir() -> PathBuf {
    let d = std::env::var("NV_CHAT_EVAL_OUT")
        .map(PathBuf::from)
        .unwrap_or_else(|_| {
            PathBuf::from(std::env::var("HOME").unwrap_or_default())
                .join(".cache/nvk-tmp/chat-eval")
        });
    std::fs::create_dir_all(&d).ok();
    d
}

#[test]
fn emit_prompt_packs_for_the_measure_lane() {
    let mut wrote = Vec::new();
    let targets: [(&str, Option<&str>); 5] = [
        (NVFP4_31B, Some(SNAP_NVFP4_TOOLS_NEW)),
        (NVFP4_31B, Some(SNAP_NVFP4_TOOLS_OLD)),
        (GEMMA_E4B, None),
        (GEMMA_26B, None),
        (QWEN36, None),
    ];
    for (repo, pin) in targets {
        if skip(repo) {
            continue;
        }
        let snaps = snapshots_of(repo);
        let snap = pin
            .map(|s| s.to_string())
            .unwrap_or_else(|| snaps[0].0.clone());
        let m = match ChatEvalModel::open_pinned(repo, &snap) {
            Ok(m) => m,
            Err(e) => {
                eprintln!("{repo}: {e}");
                continue;
            }
        };
        let pack = m.pack(standard_suite(&m).unwrap());
        assert!(
            pack.controls() >= 2,
            "{repo}@{snap} pack has {} control prompt(s). The fp8 harnesses default to CONTROLS \
             ONLY for any A/B that must be reproducible, because three runs of a byte-identical \
             config on one low-margin open-ended prompt measured 66 ended / 78 ended / 96 \
             not-ended on this box. Fewer than two controls means the pack cannot carry a claim.",
            pack.controls()
        );
        let name = format!("pack-{}-{}.json", repo.replace('/', "--"), &snap[..8]);
        let p = scratch_dir().join(&name);
        pack.write_json(&p).unwrap();
        eprintln!(
            "wrote {} ({} prompts, {} controls, stops {:?})",
            p.display(),
            pack.prompts.len(),
            pack.controls(),
            pack.stop_ids
        );
        wrote.push(p);
    }
    assert!(!wrote.is_empty(), "no pack written");
    let (_, harnesses) = prompt_pack_harnesses();
    eprintln!(
        "The fp8 harnesses ({}) auto-discover these files by template digest against the \
         weights directory they are about to measure, so a pack rendered from the wrong \
         snapshot is REFUSED rather than silently used. Override with NV_CHAT_EVAL_PACK.",
        harnesses
            .iter()
            .map(|(rel, ..)| rel.as_str())
            .collect::<Vec<_>>()
            .join(", ")
    );
}

fn workspace_test_sources() -> Vec<(String, String)> {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let mut dirs: Vec<PathBuf> = vec![root.join("tests")];
    if let Ok(rd) = std::fs::read_dir(root.join("crates")) {
        let mut crates: Vec<PathBuf> = rd
            .flatten()
            .map(|e| e.path())
            .filter(|p| p.is_dir())
            .collect();
        crates.sort();
        dirs.extend(crates.into_iter().map(|p| p.join("tests")));
    }
    let mut out = Vec::new();
    for d in dirs {
        let Ok(rd) = std::fs::read_dir(&d) else {
            continue;
        };
        let mut files: Vec<PathBuf> = rd
            .flatten()
            .map(|e| e.path())
            .filter(|p| p.extension().and_then(|e| e.to_str()) == Some("rs"))
            .collect();
        files.sort();
        for p in files {
            let rel = p
                .strip_prefix(root)
                .unwrap_or(&p)
                .to_string_lossy()
                .replace('\\', "/");
            let src =
                std::fs::read_to_string(&p).unwrap_or_else(|e| panic!("read {}: {e}", p.display()));
            out.push((rel, src));
        }
    }
    assert!(
        out.len() > 150,
        "walked only {} test sources under {} -- the walk is broken, and a scan \
         that finds nothing passes everything",
        out.len(),
        root.display()
    );
    out
}

fn defines_resolve_pack(line: &str) -> bool {
    line.trim_start().starts_with("pub fn resolve_pack")
}

fn file_stem(rel: &str) -> &str {
    rel.rsplit('/')
        .next()
        .unwrap_or(rel)
        .strip_suffix(".rs")
        .unwrap_or(rel)
}

fn module_alias(src: &str, stem: &str) -> Option<String> {
    let plain = format!("mod {stem};");
    let path_attr = format!("\"{stem}.rs\"");
    let mut lines = src.lines().map(str::trim).peekable();
    while let Some(line) = lines.next() {
        if line == plain || line == format!("pub {plain}") {
            return Some(stem.to_string());
        }
        if line.starts_with("#[path") && line.contains(&path_attr) {
            for next in lines.by_ref() {
                let next = next.trim();
                if next.is_empty() {
                    continue;
                }
                let alias = next
                    .trim_start_matches("pub ")
                    .strip_prefix("mod ")
                    .and_then(|s| s.strip_suffix(';'));
                return alias.map(|a| a.trim().to_string());
            }
        }
    }
    None
}

fn prompt_pack_harnesses() -> (Vec<String>, Vec<(String, String, String)>) {
    let all = workspace_test_sources();
    let modules: Vec<String> = all
        .iter()
        .filter(|(_, src)| src.lines().any(defines_resolve_pack))
        .map(|(rel, _)| file_stem(rel).to_string())
        .collect();
    assert!(
        !modules.is_empty(),
        "no prompt module found (a tests/*.rs defining `pub fn resolve_pack`). \
         Either they were all culled -- in which case say so -- or this discovery \
         no longer matches how prompts are shared, and it is now judging nobody."
    );
    let mut consumers = Vec::new();
    for (rel, src) in &all {
        if modules.iter().any(|m| m == file_stem(rel)) {
            continue;
        }
        for m in &modules {
            if let Some(alias) = module_alias(src, m) {
                consumers.push((rel.clone(), alias, src.clone()));
                break;
            }
        }
    }
    (modules, consumers)
}

#[test]
fn no_fp8_harness_still_builds_its_own_prompt() {
    let banned: [(&str, &str); 5] = [
        (
            ".encode(",
            "a harness that tokenizes a string into prompt ids is building its own prompt. \
             Measured 2026-08-06: the raw fragment fed this way produced a 96-token confident \
             repetition loop (median top-2 margin 11.354) that never ended its turn, and that loop \
             was the bf16 reference the whole fp8 investigation was validated against",
        ),
        (
            "format!(\"<|",
            "constructing a role/turn wrapper in Rust. Arm E measured 7/7 terminate but 4 tokens \
             short of the official render, with the thought-channel markers prefixed to every reply",
        ),
        (
            "push_str(\"<|",
            "constructing a role/turn wrapper in Rust; load the model's chat_template.jinja instead",
        ),
        (
            "<start_of_turn>",
            "Gemma 2/3 turn markers; in the Gemma 4 tokenizer these are NOT special tokens and \
             tokenize as literal text (Gemma 4 uses ids 105 and 106)",
        ),
        (
            SNAP_NVFP4_TOOLS_OLD,
            "a hardcoded snapshot id. flake.nix pins refs/main; the weight blobs are shared \
             between the two NVFP4 snapshots but the chat templates differ, so pin via \
             fp8_contract_prompts::gemma4_nvfp4_dir() and let the pack's template digest decide",
        ),
    ];
    let (modules, harnesses) = prompt_pack_harnesses();
    eprintln!("prompt modules: {modules:?}");
    assert!(
        harnesses.len() >= 4,
        "discovered only {} prompt-pack harness(es) {:?} across modules {modules:?}. The gate \
         judges what it discovers, so an implausibly small set means the discovery stopped \
         matching the tree -- not that the tree got clean.",
        harnesses.len(),
        harnesses.iter().map(|(r, ..)| r).collect::<Vec<_>>()
    );
    let mut hits = Vec::new();
    let mut checked = 0;
    for (rel, alias, src) in &harnesses {
        let (rel, src) = (rel.as_str(), src.as_str());
        checked += 1;
        for (needle, why) in banned {
            for (i, line) in src.lines().enumerate() {
                if line.contains(needle) {
                    if line.contains("guard-waiver: pack-derived") {
                        assert!(
                            line.contains("pack")
                                || src
                                    .lines()
                                    .nth(i.saturating_sub(1))
                                    .is_some_and(|l| l.contains("pack")),
                            "{rel}:{}: a guard-waiver must sit on a line whose code visibly \
                             derives from a pack render",
                            i + 1
                        );
                        continue;
                    }
                    let lines: Vec<&str> = src.lines().collect();
                    let window = &lines[i.saturating_sub(3)..i];
                    if tokenizes_a_visible_render(line, window, needle) {
                        continue;
                    }
                    hits.push(format!("{rel}:{} contains {needle:?} -- {why}", i + 1));
                }
            }
        }
        let packed = src.contains("resolve_pack")
            || src.contains("NV_TMPL_ARMS_PACK")
            || src.contains(&format!("{alias}::"));
        eprintln!(
            "{rel}: {} lines, prompt module as `{alias}`, pack-driven: {packed}",
            src.lines().count()
        );
        assert!(
            packed,
            "{rel} declares the prompt module but never calls into it, so its prompts \
             come from somewhere this gate cannot see"
        );
    }
    assert!(
        hits.is_empty(),
        "{WHY_THE_RENDER_LOOKBACK_IS_BOUNDED_TO_THREE_LINES}\n\nhand-built prompts are back:\n  \
         {}",
        hits.join("\n  ")
    );
    eprintln!(
        "checked {checked} fp8/quality harness file(s); none tokenizes a string literal and none \
         builds a role wrapper. Every prompt reaches the model through the model's own \
         chat_template.jinja via a PromptPack."
    );
}

#[test]
fn the_harness_discovery_can_actually_fail() {
    assert_eq!(file_stem("crates/nv-models/tests/a_b.rs"), "a_b");
    assert_eq!(
        module_alias("mod fp8_contract_prompts;\n", "fp8_contract_prompts").as_deref(),
        Some("fp8_contract_prompts")
    );
    assert_eq!(
        module_alias(
            "#[path = \"laguna_prompts.rs\"]\nmod prompts;\n",
            "laguna_prompts"
        )
        .as_deref(),
        Some("prompts"),
        "the #[path] form is how the laguna harnesses pull their prompt module in"
    );
    assert!(
        module_alias(
            "// mentions fp8_contract_prompts in prose\n",
            "fp8_contract_prompts"
        )
        .is_none(),
        "prose naming a module must not enrol a file in the scanned set"
    );
    assert!(
        module_alias(
            "use fp8_contract_prompts::resolve_pack;\n",
            "fp8_contract_prompts"
        )
        .is_none(),
        "only a `mod` declaration enrols a file; a `use` without one does not compile anyway"
    );
    assert!(defines_resolve_pack(
        "pub fn resolve_pack(dir: &Path) -> Result<()> {"
    ));
    assert!(
        !defines_resolve_pack("        .filter(|(_, src)| src.contains(\"pub fn resolve_pack\"))"),
        "a mention is not a definition -- matching the substring anywhere enrolled this \
         very file as a prompt module"
    );
    let (modules, harnesses) = prompt_pack_harnesses();
    assert!(
        !modules.contains(&"chat_eval".to_string()),
        "the scanner classified itself as a prompt module: {modules:?}"
    );
    assert!(
        harnesses
            .iter()
            .all(|(rel, ..)| !modules.contains(&file_stem(rel).to_string())),
        "a prompt module must not be judged as one of its own consumers: it is the one \
         place tokenization is sanctioned"
    );
}

#[test]
fn the_standard_suite_leads_with_high_margin_controls() {
    if skip(NVFP4_31B) {
        return;
    }
    let m = open_nvfp4_preferring_authoritative_like_fp8_contract_prompts_gemma4_nvfp4_dir()
        .unwrap();
    let prompts = standard_suite(&m).unwrap();
    let controls: Vec<&TemplatedPrompt> = prompts
        .iter()
        .filter(|p| p.kind == PromptKind::Control)
        .collect();
    for p in &prompts {
        eprintln!("  {:<22} [{}] {} ids", p.label, p.kind, p.ids.len());
    }
    eprintln!(
        "{} control(s) of {} prompts. MEASURED 2026-08-06: arm-A control prompts sat at median \
         top-2 margin 15.8 to 17.8 and were stable across repeats; open-ended prompts were not \
         (66 ended / 78 ended / 96 not-ended for a byte-identical config). Every fp8 A/B therefore \
         defaults to controls only.",
        controls.len(),
        prompts.len()
    );
    assert!(controls.len() >= 2, "need at least two controls");
    assert!(
        prompts
            .iter()
            .take(controls.len())
            .all(|p| p.kind == PromptKind::Control),
        "controls must come first so a truncated run still has A/B evidence"
    );
}

struct ScriptedArm {
    vocab: usize,
    prompt_len: usize,
    flip_at_gen: Option<usize>,
    seen: usize,
}

impl ScriptedArm {
    fn new(vocab: usize, prompt_len: usize) -> Self {
        Self {
            vocab,
            prompt_len,
            flip_at_gen: None,
            seen: 0,
        }
    }
    fn logits(&mut self, prev: u32) -> anyhow::Result<Vec<f32>> {
        let step = self.seen;
        self.seen += 1;
        let gen = step + 1 - self.prompt_len.min(step + 1);
        let mut v = vec![0.0f32; self.vocab];
        let base = ((prev as usize).wrapping_mul(2654435761) >> 8) % (self.vocab - 8) + 4;
        let decisive = gen % 5 != 3;
        let margin = if decisive { 6.0 } else { 0.02 };
        v[base] = margin;
        if self.flip_at_gen == Some(gen) {
            let alt = (base + 1) % (self.vocab - 8) + 4;
            v[alt] = margin + 0.001;
        }
        Ok(v)
    }
}

#[test]
fn end_to_end_synthetic_suite_prints_a_report_a_reader_can_act_on() {
    if skip(NVFP4_31B) {
        return;
    }
    let m = open_nvfp4_preferring_authoritative_like_fp8_contract_prompts_gemma4_nvfp4_dir()
        .unwrap();
    let prompts = standard_suite(&m).unwrap();
    let vocab = 4096usize;
    let stops = StopSet {
        ids: vec![4095],
        source: "synthetic".into(),
    };

    let mut suite = SuiteReport::new(
        "synthetic reference vs candidate (no GPU)",
        "ref",
        "cand-flips-one-open-ended-prompt",
    );
    for (i, p) in prompts.iter().enumerate() {
        let mut ra = ScriptedArm::new(vocab, p.ids.len());
        let mut a = free_running("ref", p, &stops, 24, |t| ra.logits(t)).unwrap();
        a.text = m.decode(&a.tokens);

        let mut rb = ScriptedArm::new(vocab, p.ids.len());
        if p.kind == PromptKind::OpenEnded && i == 2 {
            rb.flip_at_gen = Some(8);
        }
        let mut b = free_running("cand", p, &stops, 24, |t| rb.logits(t)).unwrap();
        b.text = m.decode(&b.tokens);

        suite.push(compare(p, &a, &b));
    }
    suite.validate().unwrap();
    eprintln!("{suite}");
    suite.assert_controls_exact().unwrap();
}
