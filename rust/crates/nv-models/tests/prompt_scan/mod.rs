#![allow(dead_code)]

use std::path::{Path, PathBuf};

pub const WEIGHT_EVIDENCE: [&str; 4] = [
    "WeightLoader::open_dir",
    "from_loader",
    "GgufLoader::open",
    "model.safetensors",
];

pub const TOKENIZER_EVIDENCE: [&str; 3] = [
    "Tokenizer::from_file",
    "tokenizers::Tokenizer",
    "tokenizer.json",
];

pub const TEMPLATE_EVIDENCE: [&str; 4] = [
    "OfficialTemplate",
    "ChatTemplate",
    "chat_template.jinja",
    "render_with_kwargs",
];

pub const MARKERS: [&str; 5] = [
    "<|turn>user",
    "<|turn>model",
    "<|turn>system",
    "<start_of_turn>",
    "<end_of_turn>",
];

pub const CONSTRUCTION: [&str; 2] = ["format!", "push_str"];

pub struct CrateScan<'a> {
    pub crate_name: &'a str,

    pub manifest_dir: &'a str,

    pub self_file: &'a str,

    pub min_walked: usize,

    pub allowed: &'a [(&'a str, &'a str)],
}

pub fn is_real_weights_harness(src: &str) -> bool {
    WEIGHT_EVIDENCE.iter().any(|m| src.contains(m))
        && TOKENIZER_EVIDENCE.iter().any(|m| src.contains(m))
}

pub fn renders_through_template(src: &str) -> bool {
    TEMPLATE_EVIDENCE.iter().any(|m| src.contains(m))
}

fn is_scannable(line: &str) -> bool {
    !line.starts_with("//") && !line.contains("assert")
}

pub fn hand_built_hits(src: &str) -> Vec<String> {
    let mut hits = Vec::new();
    for (i, raw) in src.lines().enumerate() {
        let line = raw.trim();
        if !is_scannable(line) || !CONSTRUCTION.iter().any(|c| line.contains(c)) {
            continue;
        }
        for m in MARKERS {
            if line.contains(m) {
                hits.push(format!("line {}: builds {m} by hand -- {line}", i + 1));
            }
        }
    }
    hits
}

pub fn unrendered_marker_hits(src: &str) -> Vec<String> {
    if renders_through_template(src) {
        return Vec::new();
    }
    let mut hits = Vec::new();
    for (i, raw) in src.lines().enumerate() {
        let line = raw.trim();
        if !is_scannable(line) {
            continue;
        }
        for m in MARKERS {
            if line.contains(m) {
                hits.push(format!("line {}: hand-typed {m} -- {line}", i + 1));
            }
        }
    }
    hits
}

impl CrateScan<'_> {
    pub fn tests_dir(&self) -> PathBuf {
        Path::new(self.manifest_dir).join("tests")
    }

    fn self_name(&self) -> String {
        Path::new(self.self_file)
            .file_name()
            .unwrap_or_default()
            .to_string_lossy()
            .into_owned()
    }

    fn harness_sources(&self) -> Vec<(String, String)> {
        let dir = self.tests_dir();
        let me = self.self_name();
        let mut out = Vec::new();
        let Ok(rd) = std::fs::read_dir(&dir) else {
            return out;
        };
        let mut paths: Vec<PathBuf> = rd
            .flatten()
            .map(|e| e.path())
            .filter(|p| p.is_file() && p.extension().is_some_and(|x| x == "rs"))
            .collect();
        paths.sort();
        for p in paths {
            let name = p
                .file_name()
                .unwrap_or_default()
                .to_string_lossy()
                .into_owned();
            if name == me {
                continue;
            }
            if let Ok(s) = std::fs::read_to_string(&p) {
                out.push((name, s));
            }
        }
        out
    }

    pub fn floor_violation(&self, walked: usize) -> Option<String> {
        (walked < self.min_walked).then(|| {
            format!(
                "[{}] walked {walked} files under {}, below the floor of {}; the discovery step \
                 is broken, so a clean result would prove nothing",
                self.crate_name,
                self.tests_dir().display(),
                self.min_walked
            )
        })
    }

    pub fn run(&self) {
        let sources = self.harness_sources();
        if let Some(msg) = self.floor_violation(sources.len()) {
            panic!("{msg}");
        }

        let mut scanned = 0usize;
        let mut allowed_hit: Vec<&str> = Vec::new();
        let mut built: Vec<String> = Vec::new();
        let mut typed: Vec<String> = Vec::new();
        for (name, src) in &sources {
            if !is_real_weights_harness(src) {
                continue;
            }
            scanned += 1;
            let b = hand_built_hits(src);
            let t = unrendered_marker_hits(src);
            if b.is_empty() && t.is_empty() {
                continue;
            }
            if let Some((f, _)) = self.allowed.iter().find(|(f, _)| f == name) {
                allowed_hit.push(*f);
                continue;
            }
            if !b.is_empty() {
                built.push(format!("{name}\n    {}", b.join("\n    ")));
            }
            if !t.is_empty() {
                typed.push(format!("{name}\n    {}", t.join("\n    ")));
            }
        }

        eprintln!(
            "[{}] prompt fidelity: {} files walked under {}, {scanned} real-weights harness(es) \
             checked, {} allow-listed",
            self.crate_name,
            sources.len(),
            self.tests_dir().display(),
            allowed_hit.len()
        );

        for (f, why) in self.allowed {
            assert!(
                allowed_hit.contains(f),
                "[{}] {f} is allow-listed but no longer hand-builds a Gemma marker -- delete its \
                 entry rather than leaving a standing exemption. Reason recorded was: {why}",
                self.crate_name
            );
        }
        assert!(
            built.is_empty(),
            "[{}] a real-weights harness builds a Gemma chat prompt by hand:\n\n{}\n\nGemma-4's \
             markers are <|turn> (105) / <turn|> (106); the Gemma-2/3 spellings are absent from \
             its vocab and shred into seven literal-text tokens each. Even the right-looking \
             Gemma-4 string is checkpoint-specific: 26B and 31B append \
             \"<|channel>thought\\n<channel|>\" to the generation prompt and E4B does not. Render \
             through the snapshot's own template:\n    mod official_template;\n    \
             OfficialTemplate::load(dir).render_user(user)",
            self.crate_name,
            built.join("\n")
        );
        assert!(
            typed.is_empty(),
            "[{}] a real-weights harness carries a hand-typed Gemma turn marker and never renders \
             through a template:\n\n{}\n\nNone of {:?} appears anywhere in the file, so nothing \
             asserts that this string equals what the checkpoint's own chat_template.jinja \
             produces -- and one string cannot be right for both, since 26B and 31B append \
             \"<|channel>thought\\n<channel|>\" to the generation prompt where E4B closes it at \
             \"<|turn>model\\n\". Render through the snapshot's own template:\n    mod \
             official_template;\n    OfficialTemplate::load(dir).render_user(user)",
            self.crate_name,
            typed.join("\n"),
            TEMPLATE_EVIDENCE
        );
    }
}

#[test]
fn the_detector_fires_on_a_hand_built_prompt_and_not_on_a_templated_one() {
    let weights = "WeightLoader::open_dir(dir, &device); Tokenizer::from_file(p);";

    let hand_built = format!(
        "{weights}\n    let p = format!(\"<bos><|turn>user\\n{{user}}<turn|>\\n<|turn>model\\n\");"
    );
    assert!(is_real_weights_harness(&hand_built));
    assert!(!renders_through_template(&hand_built));
    assert_eq!(
        hand_built_hits(&hand_built).len(),
        2,
        "must catch both <|turn>user and <|turn>model on the constructing line"
    );

    let gemma2 = format!("{weights}\n    let p = format!(\"<start_of_turn>user\\n{{q}}\");");
    assert_eq!(
        hand_built_hits(&gemma2).len(),
        1,
        "the retired Gemma-2/3 spelling must be caught too -- it shreds in the Gemma-4 vocab"
    );

    let templated =
        format!("{weights}\n    let p = OfficialTemplate::load(dir).render_user(user);");
    assert!(renders_through_template(&templated));
    assert!(hand_built_hits(&templated).is_empty());

    let fixture = format!("{weights}\n    assert!(p.starts_with(\"<|turn>user\"));");
    assert!(
        hand_built_hits(&fixture).is_empty(),
        "a marker inside an assert! is a fixture being checked, not a prompt being built"
    );

    let no_weights = "let p = format!(\"<|turn>user\\n{q}\");";
    assert!(!is_real_weights_harness(no_weights));

    let literal = format!(
        "{weights}\n    let prompts = [\"<|turn>user\\nExplain it.<turn|>\\n<|turn>model\\n\"];\n \
         tok.encode(prompts[0], false);"
    );
    assert!(
        hand_built_hits(&literal).is_empty(),
        "no format!/push_str on that line -- this is precisely the case the construction rule \
         cannot see, and why unrendered_marker_hits exists"
    );
    assert_eq!(
        unrendered_marker_hits(&literal).len(),
        2,
        "a bare literal marker in a harness with no rendering evidence is a hand-typed prompt"
    );

    let rendered_then_checked = format!(
        "{weights}\n    let r = OfficialTemplate::load(dir).render_user(u);\n    let tail = \
         \"<|turn>model\\n\";"
    );
    assert!(
        unrendered_marker_hits(&rendered_then_checked).is_empty(),
        "a marker literal in a file that visibly renders is pack-derived, not hand-typed; \
         flagging it is the false positive that gets the whole rule bypassed"
    );

    let bare_turn = format!("{weights}\n    let id = tok.token_to_id(\"<|turn>\").unwrap();");
    assert!(
        unrendered_marker_hits(&bare_turn).is_empty(),
        "MARKERS are role-qualified so a vocabulary lookup on the bare marker is not a prompt"
    );
}

#[test]
fn the_floor_rejects_a_walk_that_found_nothing() {
    let scan = CrateScan {
        crate_name: "synthetic",
        manifest_dir: "/nonexistent-manifest-dir-for-prompt-scan",
        self_file: file!(),
        min_walked: 1,
        allowed: &[],
    };
    assert!(
        scan.harness_sources().is_empty(),
        "a manifest dir that does not exist must yield no sources"
    );
    assert!(
        scan.floor_violation(0).is_some(),
        "a walk that read zero files must fail, not report clean"
    );
    assert!(scan.floor_violation(1).is_none());
}
