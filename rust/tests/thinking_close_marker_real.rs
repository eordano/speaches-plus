use std::path::{Path, PathBuf};

use serde_json::json;
use speaches_plus::oapi::chat_template::ChatTemplate;
use tokenizers::Tokenizer;

const GGUF_SIDECAR_ENV: &str = "NV_GGUF_26B_DIR";

enum Where {
    Hub(&'static str),
    UnderHome(&'static str),
}

struct Case {
    id: &'static str,
    dir: Where,
    declares_switch: bool,
    marker: Option<&'static str>,
    read_from_the_template: &'static str,
}

const CASES: &[Case] = &[
    Case {
        id: "RedHatAI/Qwen3.6-35B-A3B-NVFP4",
        dir: Where::Hub("RedHatAI/Qwen3.6-35B-A3B-NVFP4"),
        declares_switch: true,
        marker: Some("</think>"),
        read_from_the_template: "add_generation_prompt emits '<|im_start|>assistant\\n' then \
                                 '<think>\\n\\n</think>\\n\\n' when enable_thinking is false and \
                                 '<think>\\n' when it is true; the assistant arm closes a carried \
                                 reasoning_content with '\\n</think>\\n\\n'",
    },
    Case {
        id: "mlx-community/Qwen3.5-35B-A3B-4bit",
        dir: Where::Hub("mlx-community/Qwen3.5-35B-A3B-4bit"),
        declares_switch: true,
        marker: Some("</think>"),
        read_from_the_template: "same generation-prompt arms as Qwen3.6; the two templates differ \
                                 only in the preserve_thinking gate and tool-argument encoding",
    },
    Case {
        id: "Qwen/Qwen3-Embedding-0.6B",
        dir: Where::Hub("Qwen/Qwen3-Embedding-0.6B"),
        declares_switch: true,
        marker: Some("</think>"),
        read_from_the_template: "the classic Qwen3 template in tokenizer_config.json:chat_template \
                                 appends '<think>\\n\\n</think>\\n\\n' after '<|im_start|>assistant\\n' \
                                 when enable_thinking is false and appends NOTHING when it is true, \
                                 so the divergent tail is an open tag plus the close, not the close",
    },
    Case {
        id: "poolside/Laguna-XS-2.1-NVFP4",
        dir: Where::Hub("poolside/Laguna-XS-2.1-NVFP4"),
        declares_switch: true,
        marker: Some("</think>"),
        read_from_the_template: "the generation prompt is '<assistant>' then '<think>' when \
                                 enable_thinking else '</think>'; the assistant arm writes \
                                 '<think>' + reasoning + '</think>'. The two arms share the '<' of \
                                 the tag, so a byte-wise common prefix cuts the marker in half",
    },
    Case {
        id: "nvidia/Gemma-4-31B-IT-NVFP4",
        dir: Where::Hub("nvidia/Gemma-4-31B-IT-NVFP4"),
        declares_switch: true,
        marker: Some("<channel|>"),
        read_from_the_template: "thinking is armed by '<|think|>' at the TOP of the first system \
                                 turn, and the generation prompt appends the already-closed empty \
                                 thought '<|channel>thought\\n<channel|>' when it is off; a carried \
                                 reasoning renders as '<|channel>thought\\n' + text + '\\n<channel|>'. \
                                 '</think>' is not in this vocabulary at all",
    },
    Case {
        id: "google/gemma-4-E4B-it",
        dir: Where::Hub("google/gemma-4-E4B-it"),
        declares_switch: true,
        marker: Some("<channel|>"),
        read_from_the_template: "same Gemma-4 dialect, but the generation prompt carries NO empty \
                                 thought block, so the only place the close appears is the \
                                 assistant reasoning arm '<|channel>thought\\n' + text + '\\n<channel|>', \
                                 which this template gates behind message.tool_calls",
    },
    Case {
        id: "google/gemma-4-E4B-it-qat-w4a16-ct",
        dir: Where::Hub("google/gemma-4-E4B-it-qat-w4a16-ct"),
        declares_switch: true,
        marker: Some("<channel|>"),
        read_from_the_template: "generation prompt is '<|turn>model\\n' either way; the close only \
                                 appears in the assistant reasoning arm '\\n<channel|>', gated on \
                                 the message following the last user message",
    },
    Case {
        id: "google/gemma-4-E4B-it-qat-q4_0-unquantized-assistant",
        dir: Where::Hub("google/gemma-4-E4B-it-qat-q4_0-unquantized-assistant"),
        declares_switch: true,
        marker: Some("<channel|>"),
        read_from_the_template: "byte-identical to the w4a16-ct template",
    },
    Case {
        id: "Qwen/Qwen3-Omni-30B-A3B-Instruct",
        dir: Where::Hub("Qwen/Qwen3-Omni-30B-A3B-Instruct"),
        declares_switch: true,
        marker: Some("</think>"),
        read_from_the_template: "ships its template ONLY as chat_template.json. The generation \
                                 prompt appends '<think>\\n\\n</think>\\n\\n' after \
                                 '<|im_start|>assistant\\n' when enable_thinking is false and \
                                 appends nothing when it is true, so the divergent tail is an open \
                                 tag PLUS the close -- the shape that yielded a wrong marker before \
                                 the trailing-tag rule. '</think>' is 151668 in added_tokens_decoder",
    },
    Case {
        id: "Qwen/Qwen3-ForcedAligner-0.6B",
        dir: Where::Hub("Qwen/Qwen3-ForcedAligner-0.6B"),
        declares_switch: false,
        marker: None,
        read_from_the_template: "also chat_template.json only, and its source never mentions \
                                 enable_thinking, so there is no thought to close",
    },
    Case {
        id: "gemma-4-26B-A4B-it-Q8_0 (GGUF sidecar)",
        dir: Where::UnderHome(".cache/nv-gguf-serve/gemma-4-26B-A4B-it-Q8_0"),
        declares_switch: true,
        marker: Some("<channel|>"),
        read_from_the_template: "the generation prompt appends '<|channel>thought\\n<channel|>' \
                                 whenever enable_thinking is falsy; this template has no assistant \
                                 reasoning arm at all, so the generation prompt is the only witness",
    },
];

struct Shape {
    id: &'static str,
    template: &'static str,
    eos: &'static str,
    declares_switch: bool,
    marker: Option<&'static str>,
    what_this_shape_proves: &'static str,
}

const SHAPES: &[Shape] = &[
    Shape {
        id: "qwen3_classic_off_arm_appends_an_open_tag_and_its_close",
        template: r"{% for m in messages %}{{ '<|im_start|>' ~ m.role ~ '\n' ~ m.content ~ '<|im_end|>\n' }}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% if not enable_thinking %}{{ '<think>\n\n</think>\n\n' }}{% endif %}{% endif %}",
        eos: "<|im_end|>",
        declares_switch: true,
        marker: Some("</think>"),
        what_this_shape_proves:
            "the thinking-on arm appends NOTHING, so the whole divergent tail is an open tag plus \
             the close and the answer is the LAST tag in it, not the first",
    },
    Shape {
        id: "qwen36_both_arms_open_the_thought_and_only_the_off_arm_closes_it",
        template: r"{% for m in messages %}{{ '<|im_start|>' ~ m.role ~ '\n' ~ m.content ~ '<|im_end|>\n' }}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% if enable_thinking %}{{ '<think>\n' }}{% else %}{{ '<think>\n\n</think>\n\n' }}{% endif %}{% endif %}",
        eos: "<|im_end|>",
        declares_switch: true,
        marker: Some("</think>"),
        what_this_shape_proves:
            "the two arms share a COMPLETE tag '<think>' before diverging; the divergence must \
             stay where the bytes differ and must NOT be rewound to that shared tag's '<', which \
             is the opposite of what the laguna shape needs",
    },
    Shape {
        id: "laguna_both_arms_share_the_opening_angle_bracket_of_different_tags",
        template: r"{% for m in messages %}{{ '<user>' ~ m.content ~ '</user>' }}{% endfor %}{% if add_generation_prompt %}{{ '<assistant>' }}{% if enable_thinking %}{{ '<think>' }}{% else %}{{ '</think>' }}{% endif %}{% endif %}",
        eos: "</assistant>",
        declares_switch: true,
        marker: Some("</think>"),
        what_this_shape_proves:
            "'<think>' and '</think>' share their leading '<', so a byte-wise common prefix ends \
             INSIDE the marker and yields the truncated '/think>'. The divergence must rewind to \
             the '<' that opened the unterminated tag",
    },
    Shape {
        id: "gemma4_close_marker_is_not_a_slash_tag_and_is_not_the_literal_think_close",
        template: r"{% for m in messages %}{{ '<|turn>' ~ m.role ~ '\n' ~ m.content ~ '\n' }}{% endfor %}{% if add_generation_prompt %}{{ '<|turn>model\n' }}{% if not enable_thinking %}{{ '<|channel>thought\n<channel|>' }}{% endif %}{% endif %}",
        eos: "<|turn|>",
        declares_switch: true,
        marker: Some("<channel|>"),
        what_this_shape_proves:
            "the close is '<channel|>' and '</think>' is absent from this dialect entirely, so a \
             derivation that quietly falls back to the literal '</think>' arms the grammar at a \
             position the model never emits",
    },
    Shape {
        id: "both_generation_prompts_match_so_only_a_carried_thought_witnesses_the_close",
        template: r"{% if enable_thinking %}{{ '<|think|>' }}{% endif %}{% for m in messages %}{% if m.role == 'assistant' and m.reasoning_content %}{{ '<|channel>thought\n' ~ m.reasoning_content ~ '\n<channel|>' }}{% endif %}{{ '<|turn>' ~ m.role ~ '\n' ~ m.content ~ '\n' }}{% endfor %}{% if add_generation_prompt %}{{ '<|turn>model\n' }}{% endif %}",
        eos: "<|turn|>",
        declares_switch: true,
        marker: Some("<channel|>"),
        what_this_shape_proves:
            "thinking is armed at the TOP of the render, so it cancels out of both generation \
             prompts and the tail is empty; the only witness left is the assistant reasoning arm, \
             where the close LEADS the text that follows the thought instead of trailing it",
    },
    Shape {
        id: "a_template_that_never_mentions_the_switch_has_no_thought_to_close",
        template: r"{% for m in messages %}{{ '<|im_start|>' ~ m.role ~ '\n' ~ m.content ~ '<|im_end|>\n' }}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}",
        eos: "<|im_end|>",
        declares_switch: false,
        marker: None,
        what_this_shape_proves:
            "no switch means no marker at all; deriving one here would arm guided decoding on a \
             model that never opens a thought",
    },
    Shape {
        id: "a_divergent_tail_that_is_prose_rather_than_a_tag_yields_nothing",
        template: r"{% for m in messages %}{{ '<|im_start|>' ~ m.role ~ '\n' ~ m.content ~ '<|im_end|>\n' }}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% if not enable_thinking %}{{ 'Thinking is disabled for this turn.\n' }}{% endif %}{% endif %}",
        eos: "<|im_end|>",
        declares_switch: true,
        marker: None,
        what_this_shape_proves:
            "the arms differ, but by prose; None is the honest answer and any string returned \
             here is the derivation inventing a marker out of neighbouring template text",
    },
    Shape {
        id: "a_tail_whose_only_tag_is_the_eos_token_yields_nothing",
        template: r"{% for m in messages %}{{ '<|im_start|>' ~ m.role ~ '\n' ~ m.content ~ '<|im_end|>\n' }}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% if not enable_thinking %}{{ '<|im_end|>' }}{% endif %}{% endif %}",
        eos: "<|im_end|>",
        declares_switch: true,
        marker: None,
        what_this_shape_proves:
            "'<|im_end|>' passes every lone-tag test, so only the eos guard stops it; a marker \
             equal to eos makes the grammar wait for end-of-turn to close a thought that the \
             turn already ended",
    },
];

fn synthetic_root() -> PathBuf {
    PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("synthetic_chat_templates")
}

fn materialize(id: &str, files: &[(&str, String)]) -> PathBuf {
    let dir = synthetic_root().join(id);
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap_or_else(|e| panic!("create {}: {e}", dir.display()));
    for (name, body) in files {
        std::fs::write(dir.join(name), body)
            .unwrap_or_else(|e| panic!("write {}/{name}: {e}", dir.display()));
    }
    dir
}

fn materialize_shape(s: &Shape) -> PathBuf {
    materialize(
        s.id,
        &[
            ("chat_template.jinja", s.template.to_string()),
            (
                "tokenizer_config.json",
                json!({"bos_token": "<|nvsynthetic_bos|>", "eos_token": s.eos}).to_string(),
            ),
        ],
    )
}

#[test]
fn the_close_marker_derivation_holds_on_synthetic_templates_that_need_no_downloaded_snapshot() {
    let mut bad: Vec<String> = Vec::new();
    let mut checked = 0usize;
    for s in SHAPES {
        let dir = materialize_shape(s);
        let t = ChatTemplate::load_reason(&dir)
            .unwrap_or_else(|e| panic!("{}: synthetic template did not load: {e}", s.id));
        checked += 1;
        if t.declares_thinking_switch() != s.declares_switch {
            bad.push(format!(
                "{}: declares_thinking_switch want {} got {}",
                s.id,
                s.declares_switch,
                t.declares_thinking_switch()
            ));
        }
        let got = t.thinking_close_marker();
        if got.as_deref() != s.marker {
            bad.push(format!(
                "{}\n    want {:?}\n    got  {:?}\n    this shape exists because: {}\n    \
                 thinking-on  render: {:?}\n    thinking-off render: {:?}",
                s.id,
                s.marker,
                got,
                s.what_this_shape_proves,
                generation_prompt(&t, true),
                generation_prompt(&t, false),
            ));
        }
    }
    assert_eq!(
        checked,
        SHAPES.len(),
        "every synthetic shape is written to disk by this test, so none of them can be absent; \
         a lower count means the loop stopped early"
    );
    assert!(
        bad.is_empty(),
        "these templates are written by this test, so they are present on EVERY box -- unlike \
         the hub-snapshot suites in this file, which verify nothing where the snapshots are not \
         downloaded. Each expectation below was read off the template literal next to it.\n\n{}",
        bad.join("\n\n")
    );
}

#[test]
fn a_synthetic_template_shipped_only_as_chat_template_json_loads_without_any_snapshot() {
    let body = r"{% for m in messages %}{{ '<|im_start|>' ~ m.role ~ '\n' ~ m.content ~ '<|im_end|>\n' }}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% if not enable_thinking %}{{ '<think>\n\n</think>\n\n' }}{% endif %}{% endif %}";
    let dir = materialize(
        "json_only_no_jinja_no_tokenizer_config_template",
        &[(
            "chat_template.json",
            json!({ "chat_template": body }).to_string(),
        )],
    );
    let t = ChatTemplate::load_reason(&dir).unwrap_or_else(|e| {
        panic!(
            "a template carried only by chat_template.json must load: {e}. This does not raise in \
             serving -- load() logs and returns None, and the request is then prompted with the \
             built-in renderer in a format the model was never trained on"
        )
    });
    assert_eq!(
        t.thinking_close_marker().as_deref(),
        Some("</think>"),
        "the json-only path must yield the same derivation as the same source in a .jinja"
    );
}

fn home() -> PathBuf {
    PathBuf::from(std::env::var("HOME").expect("HOME must be set to find the model cache"))
}

fn hub_roots() -> Vec<PathBuf> {
    let mut out = Vec::new();
    if let Ok(v) = std::env::var("HF_HUB_CACHE") {
        out.push(PathBuf::from(v));
    }
    out.push(home().join(".cache/huggingface/hub"));
    out
}

fn carries_a_template(p: &Path) -> bool {
    if p.join("chat_template.jinja").is_file() {
        return true;
    }
    ["tokenizer_config.json", "chat_template.json"].iter().any(|f| {
        std::fs::read_to_string(p.join(f))
            .map(|s| s.contains("\"chat_template\""))
            .unwrap_or(false)
    })
}

fn locate(case: &Case) -> Option<PathBuf> {
    match case.dir {
        Where::Hub(repo) => hub_roots().into_iter().find_map(|root| {
            let snaps = root
                .join(format!("models--{}", repo.replace('/', "--")))
                .join("snapshots");
            std::fs::read_dir(snaps)
                .ok()?
                .filter_map(|e| e.ok())
                .map(|e| e.path())
                .find(|p| carries_a_template(p))
        }),
        Where::UnderHome(rel) => {
            let p = std::env::var(GGUF_SIDECAR_ENV)
                .map(PathBuf::from)
                .unwrap_or_else(|_| home().join(rel));
            carries_a_template(&p).then_some(p)
        }
    }
}

fn generation_prompt(t: &ChatTemplate, enable_thinking: bool) -> String {
    let mut kwargs = t.effective_template_kwargs();
    kwargs.insert("enable_thinking".into(), json!(enable_thinking));
    let msgs = json!([{"role": "user", "content": "probe"}]);
    match t.render_with_kwargs(&msgs, None, true, &kwargs) {
        Ok(s) => s,
        Err(e) => format!("<render failed: {e}>"),
    }
}

fn added_token_id(dir: &Path, marker: &str) -> Option<u32> {
    let raw = std::fs::read_to_string(dir.join("tokenizer_config.json")).ok()?;
    let cfg: serde_json::Value = serde_json::from_str(&raw).ok()?;
    let decoder = cfg.get("added_tokens_decoder")?.as_object()?;
    decoder.iter().find_map(|(id, v)| {
        (v.get("content").and_then(|c| c.as_str()) == Some(marker)).then(|| id.parse().ok())?
    })
}

fn announce(absent: &[&str], checked: usize) {
    for id in absent {
        eprintln!(
            "SKIP {id}: no directory carrying its chat template on this machine, so nothing about \
             this model was verified"
        );
    }
    eprintln!(
        "thinking_close_marker: {checked} real template(s) checked, {} absent",
        absent.len()
    );
}

#[test]
fn every_real_chat_template_derives_the_close_marker_its_own_source_writes() {
    let mut checked = 0usize;
    let mut absent: Vec<&str> = Vec::new();
    let mut bad: Vec<String> = Vec::new();
    for case in CASES {
        let Some(dir) = locate(case) else {
            absent.push(case.id);
            continue;
        };
        let t = ChatTemplate::load_reason(&dir).unwrap_or_else(|e| {
            panic!(
                "{} is present at {} but its template did not load: {e}. A template that is here \
                 and unreadable is a defect, not a skip",
                case.id,
                dir.display()
            )
        });
        checked += 1;
        assert_eq!(
            t.declares_thinking_switch(),
            case.declares_switch,
            "{}: declares_thinking_switch changed for a template read from {}",
            case.id,
            dir.display()
        );
        let got = t.thinking_close_marker();
        if got.as_deref() != case.marker {
            bad.push(format!(
                "{}\n    at {}\n    want {:?}\n    got  {:?}\n    the template says: {}\n    \
                 thinking-on  render: {:?}\n    thinking-off render: {:?}",
                case.id,
                dir.display(),
                case.marker,
                got,
                case.read_from_the_template,
                generation_prompt(&t, true),
                generation_prompt(&t, false),
            ));
        }
    }
    announce(&absent, checked);
    assert!(
        bad.is_empty(),
        "the expected markers below were read out of each template's own source, not recorded \
         from this function's output. None is not a safe answer: the caller falls back to the \
         literal \"</think>\", which Gemma-4 does not have in its vocabulary at all, and a marker \
         that is merely close arms the grammar at a position the model never reaches, so guided \
         decoding never starts.\n\n{}",
        bad.join("\n\n")
    );
    assert!(
        checked > 0,
        "no chat template was found anywhere under {:?} or {GGUF_SIDECAR_ENV}, so this suite \
         proved nothing about the derivation",
        hub_roots()
    );
}

#[test]
fn every_derived_marker_is_a_single_token_in_that_models_own_vocabulary() {
    let mut checked = 0usize;
    let mut absent: Vec<&str> = Vec::new();
    let mut bad: Vec<String> = Vec::new();
    for case in CASES {
        let Some(dir) = locate(case) else {
            absent.push(case.id);
            continue;
        };
        let t = ChatTemplate::load_reason(&dir).expect("template present but unreadable");
        let Some(marker) = t.thinking_close_marker() else {
            continue;
        };
        checked += 1;
        let vocab = dir.join("tokenizer.json");
        let single = if vocab.is_file() {
            let tok = Tokenizer::from_file(&vocab).expect("tokenizer.json parses");
            tok.token_to_id(&marker).or_else(|| {
                match tok.encode(marker.as_str(), false).ok()?.get_ids() {
                    [one] => Some(*one),
                    _ => None,
                }
            })
        } else {
            added_token_id(&dir, &marker)
        };
        if single.is_none() {
            bad.push(format!("{}: {marker:?} is not one token", case.id));
        }
    }
    announce(&absent, checked);
    assert!(
        bad.is_empty(),
        "the grammar can now defer on a token sequence, so a multi-token marker is no longer \
         fatal -- but every model in this corpus closes its thought with ONE token, and a \
         marker that suddenly needs several means the derivation drifted and picked up \
         neighbouring template text rather than the tag.\n{}",
        bad.join("\n")
    );
    assert!(
        checked > 0,
        "not one of the {} models in this corpus was present with a derived marker ({} absent \
         under {:?}), so no marker was ever looked up in a real vocabulary and this suite \
         proved nothing. The synthetic shapes in this file cannot stand in for it: they carry \
         no tokenizer.json, and single-token-ness is a fact about a published vocabulary, not \
         about the derivation. Download a snapshot or set {GGUF_SIDECAR_ENV}.",
        CASES.len(),
        absent.len(),
        hub_roots()
    );
}

#[test]
fn a_template_shipped_only_as_chat_template_json_still_loads() {
    const JSON_ONLY: [&str; 2] = [
        "Qwen/Qwen3-Omni-30B-A3B-Instruct",
        "Qwen/Qwen3-ForcedAligner-0.6B",
    ];
    let mut checked = 0usize;
    let mut absent: Vec<&str> = Vec::new();
    for repo in JSON_ONLY {
        let found = hub_roots().into_iter().find_map(|root| {
            let snaps = root
                .join(format!("models--{}", repo.replace('/', "--")))
                .join("snapshots");
            std::fs::read_dir(snaps)
                .ok()?
                .filter_map(|e| e.ok())
                .map(|e| e.path())
                .find(|p| p.join("chat_template.json").is_file())
        });
        let Some(dir) = found else {
            absent.push(repo);
            continue;
        };
        assert!(
            !dir.join("chat_template.jinja").is_file(),
            "{repo}: a .jinja appeared, so this no longer exercises the json-only path"
        );
        assert!(
            !std::fs::read_to_string(dir.join("tokenizer_config.json"))
                .map(|s| s.contains("\"chat_template\""))
                .unwrap_or(false),
            "{repo}: tokenizer_config now carries the template, so this no longer exercises \
             the json-only path"
        );
        assert!(
            ChatTemplate::load(&dir).is_some(),
            "{repo}: the template must load from chat_template.json alone. A template that \
             fails to load does not raise -- serving silently falls back to the built-in \
             renderer and prompts the model in a format it was never trained on"
        );
        checked += 1;
    }
    announce(&absent, checked);
    assert!(
        checked > 0,
        "neither {JSON_ONLY:?} was present under {:?}, so the chat_template.json-only load path \
         was never exercised against a PUBLISHED template. The synthetic json-only case in this \
         file covers the loader, but not the shape a real hub repo ships -- which is what \
         regressed here before. Both siblings in this file carry this floor; this one did not, \
         which is the shape that lets a model-gated body return early and still read as coverage",
        hub_roots()
    );
}
