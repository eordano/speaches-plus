use candle_core::quantized::gguf_file::{self, Value};
use candle_core::quantized::{GgmlDType, QTensor};
use candle_core::{Device, Tensor};
use nv_weights::gguf::{
    ensure_gguf_sidecars, gguf_chat_template, gguf_tokenizer_config_json, gguf_tokenizer_json,
    missing_gguf_sidecars, GGUF_SIDECAR_FILES,
};
use nv_weights::GgufLoader;
mod common;
use common::TempDir;

const TEMPLATE: &str =
    "{{ bos_token }}{%- for m in messages -%}<|turn>{{ m['role'] }}\n{{ m['content'] }}<turn|>\n{%- endfor -%}";

const NORMAL: u32 = 1;
const CONTROL: u32 = 3;
const USER_DEFINED: u32 = 4;
const BYTE: u32 = 6;

fn temp_dir(tag: &str) -> TempDir {
    let dir = std::env::temp_dir().join(format!(
        "gguf_sidecar_{}_{tag}_{:?}",
        std::process::id(),
        std::thread::current().id()
    ));
    std::fs::remove_dir_all(&dir).ok();
    std::fs::create_dir_all(&dir).unwrap();
    TempDir(dir)
}

struct Vocab {
    tokens: Vec<String>,
    types: Vec<u32>,
}

impl Vocab {
    fn push(&mut self, piece: &str, ty: u32) {
        self.tokens.push(piece.into());
        self.types.push(ty);
    }

    fn id(&self, piece: &str) -> u32 {
        self.tokens
            .iter()
            .position(|t| t == piece)
            .unwrap_or_else(|| panic!("toy vocab has no piece {piece:?}")) as u32
    }
}

fn toy_vocab() -> Vocab {
    let mut v = Vocab {
        tokens: Vec::new(),
        types: Vec::new(),
    };
    for (piece, ty) in [
        ("<pad>", CONTROL),
        ("<eos>", CONTROL),
        ("<bos>", CONTROL),
        ("<unk>", CONTROL),
        ("<|turn>", CONTROL),
        ("<turn|>", USER_DEFINED),
    ] {
        v.push(piece, ty);
    }
    for b in 0..=255u16 {
        v.push(&format!("<0x{b:02X}>"), BYTE);
    }
    for piece in [
        "\u{2581}",
        "\u{2581}h",
        "\u{2581}he",
        "\u{2581}hel",
        "\u{2581}hell",
        "\u{2581}hello",
        "\u{2581}w",
        "\u{2581}wo",
        "\u{2581}wor",
        "\u{2581}worl",
        "\u{2581}world",
        "h",
        "e",
        "l",
        "o",
        "w",
        "r",
        "d",
    ] {
        v.push(piece, NORMAL);
    }
    v
}

const TOY_MERGES: [&str; 10] = [
    "\u{2581} h",
    "\u{2581}h e",
    "\u{2581}he l",
    "\u{2581}hel l",
    "\u{2581}hell o",
    "\u{2581} w",
    "\u{2581}w o",
    "\u{2581}wo r",
    "\u{2581}wor l",
    "\u{2581}worl d",
];

struct Overrides {
    tokenizer_model: &'static str,
    add_space_prefix: bool,
    duplicate_piece: bool,
    template: &'static str,
    with_merges: bool,
}

impl Default for Overrides {
    fn default() -> Self {
        Self {
            tokenizer_model: "gemma4",
            add_space_prefix: false,
            duplicate_piece: false,
            template: TEMPLATE,
            with_merges: true,
        }
    }
}

fn write_toy_gguf(dir: &std::path::Path, o: &Overrides) -> std::path::PathBuf {
    let mut v = toy_vocab();
    if o.duplicate_piece {
        v.push("h", NORMAL);
    }
    let (bos, eos, unk, pad) = (v.id("<bos>"), v.id("<turn|>"), v.id("<unk>"), v.id("<pad>"));
    let tokens = Value::Array(v.tokens.iter().map(|t| Value::String(t.clone())).collect());
    let types = Value::Array(v.types.iter().map(|t| Value::U32(*t)).collect());
    let merges = Value::Array(
        TOY_MERGES
            .iter()
            .map(|m| Value::String((*m).into()))
            .collect(),
    );

    let mut md: Vec<(&str, Value)> = vec![
        ("general.architecture", Value::String("gemma4".into())),
        (
            "tokenizer.ggml.model",
            Value::String(o.tokenizer_model.into()),
        ),
        ("tokenizer.ggml.tokens", tokens),
        ("tokenizer.ggml.token_type", types),
        ("tokenizer.ggml.bos_token_id", Value::U32(bos)),
        ("tokenizer.ggml.eos_token_id", Value::U32(eos)),
        ("tokenizer.ggml.unknown_token_id", Value::U32(unk)),
        ("tokenizer.ggml.padding_token_id", Value::U32(pad)),
        ("tokenizer.ggml.add_bos_token", Value::Bool(false)),
        (
            "tokenizer.ggml.add_space_prefix",
            Value::Bool(o.add_space_prefix),
        ),
        ("tokenizer.chat_template", Value::String(o.template.into())),
    ];
    if o.with_merges {
        md.push(("tokenizer.ggml.merges", merges));
    }

    let t = QTensor::quantize(
        &Tensor::zeros((4, 32), candle_core::DType::F32, &Device::Cpu).unwrap(),
        GgmlDType::F32,
    )
    .unwrap();
    let path = dir.join("model.gguf");
    let mut f = std::fs::File::create(&path).unwrap();
    let md_refs: Vec<(&str, &Value)> = md.iter().map(|(k, v)| (*k, v)).collect();
    gguf_file::write(&mut f, &md_refs, &[("token_embd.weight", &t)]).unwrap();
    path
}

fn toy_loader(dir: &std::path::Path, o: &Overrides) -> GgufLoader {
    let path = write_toy_gguf(dir, o);
    GgufLoader::open(&path, &Device::Cpu).expect("open toy gguf")
}

fn load_tokenizer(dir: &std::path::Path) -> tokenizers::Tokenizer {
    tokenizers::Tokenizer::from_file(dir.join("tokenizer.json"))
        .unwrap_or_else(|e| panic!("synthesized tokenizer.json did not load: {e}"))
}

#[test]
fn a_bare_gguf_dir_gains_exactly_the_three_serving_sidecars() {
    let d = temp_dir("bare");
    assert_eq!(
        missing_gguf_sidecars(&d.0).len(),
        GGUF_SIDECAR_FILES.len(),
        "a fresh dir must be missing every sidecar, or this test proves nothing"
    );
    write_toy_gguf(&d.0, &Overrides::default());

    let written = ensure_gguf_sidecars(&d.0).expect("synthesize sidecars");
    assert_eq!(
        written,
        GGUF_SIDECAR_FILES.to_vec(),
        "every sidecar the serving path reads must be written"
    );
    for name in GGUF_SIDECAR_FILES {
        let p = d.0.join(name);
        let meta = std::fs::metadata(&p)
            .unwrap_or_else(|e| panic!("{} was reported written but is absent: {e}", p.display()));
        assert!(meta.len() > 0, "{name} was written empty");
    }
    assert!(
        missing_gguf_sidecars(&d.0).is_empty(),
        "the dir must report itself complete afterwards"
    );
    assert!(
        ensure_gguf_sidecars(&d.0).expect("second pass").is_empty(),
        "a second boot must be a no-op, not a rewrite"
    );
}

#[test]
fn synthesis_never_overwrites_a_sidecar_that_is_already_there() {
    let d = temp_dir("nooverwrite");
    write_toy_gguf(&d.0, &Overrides::default());
    let mine = "{# hand-written, must survive #}";
    std::fs::write(d.0.join("chat_template.jinja"), mine).unwrap();

    let written = ensure_gguf_sidecars(&d.0).expect("synthesize sidecars");
    assert!(
        !written.contains(&"chat_template.jinja"),
        "an existing sidecar must not be reported as written: {written:?}"
    );
    assert_eq!(
        std::fs::read_to_string(d.0.join("chat_template.jinja")).unwrap(),
        mine,
        "an operator-supplied template was clobbered by the gguf one"
    );
}

#[test]
fn an_operator_tokenizer_config_keeps_ownership_of_the_template() {
    let d = temp_dir("operatorcfg");
    write_toy_gguf(&d.0, &Overrides::default());
    std::fs::write(
        d.0.join("tokenizer_config.json"),
        r#"{"bos_token":"<bos>","chat_template":"{{ messages[0]['content'] }}"}"#,
    )
    .unwrap();

    let written = ensure_gguf_sidecars(&d.0).expect("synthesize sidecars");
    assert_eq!(
        written,
        vec!["tokenizer.json"],
        "only the tokenizer was missing; a tokenizer_config.json on disk may already carry a \
         chat_template, and dropping a .jinja beside it would silently take precedence"
    );
    assert!(!d.0.join("chat_template.jinja").exists());
}

#[test]
fn a_dir_with_no_gguf_is_left_alone() {
    let d = temp_dir("nogguf");
    std::fs::write(d.0.join("config.json"), "{}").unwrap();
    assert!(
        ensure_gguf_sidecars(&d.0).expect("no-op").is_empty(),
        "there is nothing to synthesize from"
    );
    for name in GGUF_SIDECAR_FILES {
        assert!(
            !d.0.join(name).exists(),
            "{name} was invented without a checkpoint to read it from"
        );
    }
}

#[test]
fn the_written_template_is_the_gguf_bytes_verbatim() {
    let d = temp_dir("template");
    let loader = toy_loader(&d.0, &Overrides::default());
    ensure_gguf_sidecars(&d.0).expect("synthesize sidecars");
    let on_disk = std::fs::read_to_string(d.0.join("chat_template.jinja")).unwrap();
    assert_eq!(
        on_disk,
        gguf_chat_template(&loader).unwrap(),
        "the sidecar must be this checkpoint's own template, byte for byte"
    );
    assert_eq!(on_disk, TEMPLATE);
}

#[test]
fn the_synthesized_tokenizer_round_trips_text_and_keeps_markers_whole() {
    let d = temp_dir("roundtrip");
    write_toy_gguf(&d.0, &Overrides::default());
    ensure_gguf_sidecars(&d.0).expect("synthesize sidecars");
    let tok = load_tokenizer(&d.0);
    let v = toy_vocab();

    let ids: Vec<u32> = tok
        .encode("hello world", false)
        .expect("encode")
        .get_ids()
        .to_vec();
    assert!(!ids.is_empty(), "encoded nothing");
    assert_eq!(
        tok.decode(&ids, false).expect("decode"),
        "hello world",
        "round trip through the synthesized tokenizer changed the text"
    );
    assert!(
        ids.contains(&v.id("\u{2581}world")),
        "the merge table did not reach the tokenizer: ' world' must fuse to one piece, got {ids:?}"
    );
    assert!(
        !ids.contains(&v.id("w")),
        "'w' survived as its own piece, so BPE merged nothing: {ids:?}"
    );

    let marked: Vec<u32> = tok
        .encode("<bos>hello world<turn|>", false)
        .expect("encode markers")
        .get_ids()
        .to_vec();
    assert_eq!(marked.first().copied(), Some(v.id("<bos>")));
    assert_eq!(marked.last().copied(), Some(v.id("<turn|>")));
    assert_eq!(
        marked.len(),
        ids.len() + 2,
        "control markers must cost exactly one id each: {marked:?}"
    );

    let bytes: Vec<u32> = tok
        .encode("\u{221a}", false)
        .expect("encode out-of-vocab")
        .get_ids()
        .to_vec();
    assert_eq!(
        bytes,
        vec![v.id("<0xE2>"), v.id("<0x88>"), v.id("<0x9A>")],
        "byte fallback is not wired: an out-of-vocab char must decompose to its utf-8 bytes"
    );
}

#[test]
fn the_special_id_metadata_reaches_tokenizer_config() {
    let d = temp_dir("cfg");
    let loader = toy_loader(&d.0, &Overrides::default());
    let cfg = gguf_tokenizer_config_json(&loader).expect("synthesize tokenizer_config");
    assert_eq!(cfg["bos_token"], "<bos>");
    assert_eq!(
        cfg["eos_token"], "<turn|>",
        "eos must follow tokenizer.ggml.eos_token_id, not the piece literally named <eos>"
    );
    assert_eq!(cfg["unk_token"], "<unk>");
    assert_eq!(cfg["pad_token"], "<pad>");
    assert_eq!(cfg["add_bos_token"], false);
}

#[test]
fn added_tokens_are_the_control_and_user_defined_pieces_only() {
    let d = temp_dir("added");
    let loader = toy_loader(&d.0, &Overrides::default());
    let json = gguf_tokenizer_json(&loader).expect("synthesize tokenizer.json");
    let added = json["added_tokens"].as_array().unwrap();
    let contents: Vec<&str> = added
        .iter()
        .map(|a| a["content"].as_str().unwrap())
        .collect();
    assert_eq!(
        contents,
        vec!["<pad>", "<eos>", "<bos>", "<unk>", "<|turn>", "<turn|>"],
        "added_tokens must be exactly the non-NORMAL, non-BYTE pieces, in id order"
    );
    assert!(
        added.iter().all(|a| a["special"] == true),
        "every added token must be marked special or it will be echoed to the client"
    );
    let ids: Vec<u64> = added.iter().map(|a| a["id"].as_u64().unwrap()).collect();
    assert_eq!(ids, vec![0, 1, 2, 3, 4, 5]);
    assert_eq!(
        json["model"]["vocab"].as_object().unwrap().len(),
        toy_vocab().tokens.len(),
        "the whole vocab must reach the model, not just the added tokens"
    );
    assert_eq!(json["model"]["merges"].as_array().unwrap().len(), 10);
}

fn synthesis_error(tag: &str, o: Overrides) -> String {
    let d = temp_dir(tag);
    let loader = toy_loader(&d.0, &o);
    match gguf_tokenizer_json(&loader) {
        Ok(_) => panic!("{tag}: synthesis accepted a checkpoint it cannot honestly convert"),
        Err(e) => format!("{e:#}"),
    }
}

#[test]
fn an_unverified_tokenizer_family_is_refused_rather_than_guessed() {
    let e = synthesis_error(
        "family",
        Overrides {
            tokenizer_model: "llama",
            ..Default::default()
        },
    );
    assert!(
        e.contains("tokenizer.ggml.model") && e.contains("llama"),
        "the refusal must name the family it saw: {e}"
    );
}

#[test]
fn add_space_prefix_is_refused_because_the_normalizer_does_not_implement_it() {
    let e = synthesis_error(
        "prefix",
        Overrides {
            add_space_prefix: true,
            ..Default::default()
        },
    );
    assert!(
        e.contains("add_space_prefix"),
        "the refusal must name the knob: {e}"
    );
}

#[test]
fn a_duplicated_vocab_piece_is_refused_rather_than_silently_dropped() {
    let e = synthesis_error(
        "dup",
        Overrides {
            duplicate_piece: true,
            ..Default::default()
        },
    );
    assert!(
        e.contains("repeats the piece"),
        "the refusal must say which piece collided: {e}"
    );
}

#[test]
fn a_gguf_with_no_merges_is_refused_rather_than_shipped_as_a_byte_tokenizer() {
    let e = synthesis_error(
        "nomerges",
        Overrides {
            with_merges: false,
            ..Default::default()
        },
    );
    assert!(
        e.contains("merges"),
        "the refusal must name the missing table: {e}"
    );
}

#[test]
fn a_blank_chat_template_is_refused_rather_than_written_as_an_empty_file() {
    let d = temp_dir("blanktmpl");
    let loader = toy_loader(
        &d.0,
        &Overrides {
            template: "   \n",
            ..Default::default()
        },
    );
    let e = format!(
        "{:#}",
        gguf_chat_template(&loader).expect_err("a blank template must not be accepted")
    );
    assert!(e.contains("chat_template"), "{e}");
    let err = ensure_gguf_sidecars(&d.0).expect_err("the boot path must fail, not half-write");
    assert!(
        format!("{err:#}").contains("chat_template"),
        "the boot-path error must point at the template: {err:#}"
    );
}
