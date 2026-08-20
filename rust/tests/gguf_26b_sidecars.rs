use std::path::{Path, PathBuf};

use candle_core::Device;
use nv_weights::gguf::{
    ensure_gguf_sidecars, gguf_chat_template, gguf_tokenizer_json, GGUF_SIDECAR_FILES,
};
use nv_weights::GgufLoader;
use serde_json::Value;

const GGUF_ENV: &str = "NV_GGUF_26B";

const REFERENCE_REPO: &str = "models--google--gemma-4-E4B-it";

const GGUF_TYPES_EOS_AS_NORMAL: u64 = 1;

fn gguf_path() -> PathBuf {
    let p = std::env::var(GGUF_ENV)
        .map(PathBuf::from)
        .unwrap_or_else(|_| {
            panic!(
                "set {GGUF_ENV} to a gemma-4-26B-A4B-it-Q8_0.gguf. This suite exists to prove a \
             bare GGUF dir boots, so a missing checkpoint is a failure, not a skip"
            )
        });
    assert!(
        p.is_file(),
        "no 26B GGUF at {}; set {GGUF_ENV}",
        p.display()
    );
    p
}

fn reference_dir() -> PathBuf {
    let home = std::env::var("HOME").expect("HOME");
    let snaps = Path::new(&home)
        .join(".cache/huggingface/hub")
        .join(REFERENCE_REPO)
        .join("snapshots");
    let dir = std::fs::read_dir(&snaps)
        .unwrap_or_else(|e| {
            panic!(
                "no {} snapshots at {}: {e}",
                REFERENCE_REPO,
                snaps.display()
            )
        })
        .flatten()
        .map(|e| e.path())
        .find(|p| p.join("tokenizer.json").is_file())
        .unwrap_or_else(|| {
            panic!(
                "no snapshot under {} carries tokenizer.json; the published gemma-4 tokenizer is \
                 the reference this suite compares against and there is no substitute for it",
                snaps.display()
            )
        });
    dir
}

fn loader() -> GgufLoader {
    GgufLoader::open(gguf_path(), &Device::Cpu).expect("open 26B gguf")
}

struct TempDir(PathBuf);
impl Drop for TempDir {
    fn drop(&mut self) {
        std::fs::remove_dir_all(&self.0).ok();
    }
}

fn bare_gguf_dir(tag: &str) -> TempDir {
    let dir = std::env::temp_dir().join(format!("gguf26b_{}_{tag}", std::process::id()));
    std::fs::remove_dir_all(&dir).ok();
    std::fs::create_dir_all(&dir).unwrap();
    std::os::unix::fs::symlink(gguf_path(), dir.join("model.gguf")).unwrap();
    TempDir(dir)
}

#[test]
#[ignore = "reads the 26B gguf header and the published gemma-4 tokenizer.json; --ignored"]
fn synthesized_26b_tokenizer_matches_the_published_gemma4_tokenizer() {
    let mine = gguf_tokenizer_json(&loader()).expect("synthesize tokenizer.json from gguf");
    let refdir = reference_dir();
    let theirs: Value = serde_json::from_str(
        &std::fs::read_to_string(refdir.join("tokenizer.json")).expect("read reference"),
    )
    .expect("parse reference tokenizer.json");

    let (mv, tv) = (
        mine["model"]["vocab"].as_object().unwrap(),
        theirs["model"]["vocab"].as_object().unwrap(),
    );
    assert_eq!(
        mv.len(),
        262_144,
        "gemma-4 has a 262144-piece vocab; a different size means the wrong checkpoint"
    );
    assert_eq!(mv.len(), tv.len(), "vocab size differs from the reference");
    let mismatched: Vec<&String> = mv.keys().filter(|k| mv[*k] != tv[*k]).collect();
    assert!(
        mismatched.is_empty(),
        "{} pieces map to a different id than the published tokenizer, e.g. {:?}",
        mismatched.len(),
        &mismatched[..mismatched.len().min(5)]
    );

    let (mm, tm) = (
        mine["model"]["merges"].as_array().unwrap(),
        theirs["model"]["merges"].as_array().unwrap(),
    );
    assert_eq!(mm.len(), 514_906, "merge count changed");
    assert_eq!(mm.len(), tm.len(), "merge count differs from the reference");
    let bad = (0..mm.len()).find(|&i| mm[i] != tm[i]);
    assert!(
        bad.is_none(),
        "merge {} differs: {:?} vs published {:?}",
        bad.unwrap(),
        mm[bad.unwrap()],
        tm[bad.unwrap()]
    );

    for section in ["normalizer", "pre_tokenizer", "post_processor", "decoder"] {
        assert_eq!(
            mine[section], theirs[section],
            "{section} differs from the published pipeline"
        );
    }
    for field in [
        "type",
        "unk_token",
        "fuse_unk",
        "byte_fallback",
        "ignore_merges",
        "continuing_subword_prefix",
        "end_of_word_suffix",
        "dropout",
    ] {
        assert_eq!(
            mine["model"][field], theirs["model"][field],
            "model.{field} differs from the published tokenizer"
        );
    }

    let ids = |v: &Value| -> Vec<u64> {
        v["added_tokens"]
            .as_array()
            .unwrap()
            .iter()
            .map(|a| a["id"].as_u64().unwrap())
            .collect()
    };
    let (my_added, their_added) = (ids(&mine), ids(&theirs));
    let dropped: Vec<u64> = their_added
        .iter()
        .copied()
        .filter(|i| !my_added.contains(i))
        .collect();
    let invented: Vec<u64> = my_added
        .iter()
        .copied()
        .filter(|i| !their_added.contains(i))
        .collect();
    assert!(
        invented.is_empty(),
        "synthesis marked ids special that the published tokenizer does not: {invented:?}"
    );
    let types = loader().md_u64_list("tokenizer.ggml.token_type").unwrap();
    assert!(
        dropped.iter().all(|&i| types[i as usize] == GGUF_TYPES_EOS_AS_NORMAL),
        "an added token went missing for a reason other than the gguf typing it NORMAL: {dropped:?}"
    );
    assert_eq!(
        dropped,
        vec![1],
        "the only id the gguf types NORMAL while the published tokenizer marks it special is \
         <eos>=1; anything else is a synthesis regression"
    );
}

#[test]
#[ignore = "reads the 26B gguf header and the published gemma-4 tokenizer.json; --ignored"]
fn synthesized_and_published_tokenizers_encode_prompts_identically() {
    let d = bare_gguf_dir("encode");
    ensure_gguf_sidecars(&d.0).expect("synthesize sidecars");
    let mine = tokenizers::Tokenizer::from_file(d.0.join("tokenizer.json")).expect("load mine");
    let theirs = tokenizers::Tokenizer::from_file(reference_dir().join("tokenizer.json"))
        .expect("load reference");

    let corpus = [
        "What is the capital of France? Answer in one sentence.",
        "<bos><|turn>user\nhello<turn|>\n<|turn>model\n<|channel>thought\n<channel|>",
        "  leading and  doubled   spaces \tand\ttabs\n\nand newlines",
        "unicode: \u{221a}2 \u{2248} 1.41421, \u{4f60}\u{597d}, emoji \u{1f600}",
        "def f(x): return x**2  # code with punctuation, {braces} and [brackets]",
    ];
    for text in corpus {
        let a: Vec<u32> = mine.encode(text, false).expect("mine").get_ids().to_vec();
        let b: Vec<u32> = theirs
            .encode(text, false)
            .expect("theirs")
            .get_ids()
            .to_vec();
        assert!(
            a.len() > 3,
            "{text:?} encoded to {a:?}: too short to discriminate anything"
        );
        assert_eq!(a, b, "encodings differ on {text:?}");
        assert_eq!(
            mine.decode(&a, false).expect("decode"),
            theirs.decode(&b, false).expect("decode"),
            "decodings differ on {text:?}"
        );
    }
    let markers: Vec<u32> = mine
        .encode("<|turn>", false)
        .expect("marker")
        .get_ids()
        .to_vec();
    assert_eq!(
        markers.len(),
        1,
        "a turn marker must cost one id; {markers:?} means added_tokens did not reach the model \
         and every rendered prompt would be shredded off-distribution"
    );
}

#[test]
#[ignore = "reads the 26B gguf header and the published gemma-4 chat template; --ignored"]
fn the_synthesized_template_is_the_26bs_own_not_another_sizes() {
    let d = bare_gguf_dir("template");
    ensure_gguf_sidecars(&d.0).expect("synthesize sidecars");
    let written = std::fs::read_to_string(d.0.join("chat_template.jinja")).expect("read template");
    assert_eq!(
        written,
        gguf_chat_template(&loader()).unwrap(),
        "the sidecar must carry the 26B's own template bytes"
    );

    let e4b = std::fs::read_to_string(reference_dir().join("chat_template.jinja"))
        .expect("read the published E4B template");
    assert!(
        !e4b.trim().is_empty() && written.len() > 1000,
        "both templates must be substantial or the inequality below proves nothing: \
         26B={} bytes, E4B={} bytes",
        written.len(),
        e4b.len()
    );
    assert_ne!(
        written, e4b,
        "the 26B sidecar equals the E4B template. Assembling a serving dir by copying another \
         gemma-4 size's tokenizer directory is exactly the trap this gate exists for: the vocab \
         is shared, the template is not"
    );

    let tmpl = speaches_plus::oapi::chat_template::ChatTemplate::load_reason(&d.0)
        .expect("the synthesized template must compile under minijinja");
    let messages =
        serde_json::json!([{"role": "user", "content": "What is the capital of France?"}]);
    let rendered = tmpl.render(&messages, None, true).expect("render");
    for marker in ["<|turn>user", "<turn|>", "<|turn>model"] {
        assert!(
            rendered.contains(marker),
            "render is missing {marker:?}: {rendered:?}"
        );
    }
    assert!(
        rendered.ends_with("<|channel>thought\n<channel|>"),
        "the 26B generation prompt must end on the thought-channel opener; got {rendered:?}"
    );
    assert!(
        rendered.starts_with("<bos>"),
        "the template must emit bos_token from the synthesized tokenizer_config: {rendered:?}"
    );

    let tok = tokenizers::Tokenizer::from_file(d.0.join("tokenizer.json")).unwrap();
    let ids: Vec<u32> = tok
        .encode(rendered.as_str(), false)
        .unwrap()
        .get_ids()
        .to_vec();
    assert!(ids.len() > 10, "prompt encoded to {ids:?}");
    assert_eq!(
        tok.decode(&ids, false).unwrap(),
        rendered,
        "the rendered prompt does not survive a tokenize/detokenize round trip"
    );
}

#[cfg(feature = "wgpu")]
#[test]
#[ignore = "reads the 26B gguf header; --ignored"]
fn a_bare_26b_gguf_dir_reaches_the_weight_load_with_no_hand_placed_files() {
    use speaches_plus::oapi::chat_engine_wgpu as engine;

    let d = bare_gguf_dir("boot");
    for name in GGUF_SIDECAR_FILES {
        assert!(
            !d.0.join(name).exists(),
            "{name} must be absent at the start or this proves nothing"
        );
    }
    assert!(
        !d.0.join("config.json").exists(),
        "config.json must be absent; the loader synthesizes it in memory"
    );

    engine::ensure_serving_sidecars(&d.0).expect("the serving path must make a bare dir bootable");

    let gguf = nv_weights::gguf::lone_gguf_file(&d.0).expect("lone gguf");
    let cfg = engine::gguf_config_json(&gguf).expect("synthesize config.json from gguf");
    assert_eq!(
        engine::classify_wgpu_model(&cfg).expect("classify"),
        engine::WgpuModelKind::Gemma4Moe,
        "the 26B must classify as the gemma4 MoE decoder"
    );
    assert!(
        engine::eos_ids_from_dir(&d.0).is_err(),
        "a bare gguf dir has no generation_config.json; if this ever succeeds the gguf eos \
         fallback below has stopped being the path under test"
    );
    let eos = engine::eos_ids_for_serving(&d.0).expect("serving must fall back to the gguf eos");
    let want = GgufLoader::open(&gguf, &Device::Cpu)
        .unwrap()
        .md_u64("tokenizer.ggml.eos_token_id")
        .expect("the gguf must name a stop token, since no sidecar does") as u32;
    assert_eq!(
        eos,
        vec![want],
        "the stop condition must come from the checkpoint's own metadata"
    );

    speaches_plus::oapi::chat_template::ChatTemplate::load_reason(&d.0)
        .expect("template must load from the bare dir");
    tokenizers::Tokenizer::from_file(d.0.join("tokenizer.json"))
        .expect("tokenizer must load from the bare dir");
    let probe = engine::probe_prompt_head(
        &d.0,
        &[speaches_plus::oapi::chat::ChatMessageIn {
            role: "user".into(),
            content: Some(speaches_plus::oapi::chat::MessageContent::Text("hi".into())),
            tool_calls: None,
            tool_call_id: None,
            name: None,
        }],
    )
    .expect("probe_prompt_head must work on a bare gguf dir");
    assert!(
        probe.prompt_ids.len() > 5,
        "prompt head is degenerate: {:?}",
        probe.prompt_ids
    );
}
