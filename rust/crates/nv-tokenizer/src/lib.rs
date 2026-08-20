use anyhow::Result;
use serde::{Deserialize, Serialize};

pub use tokenizers::Tokenizer;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ChatRequest {
    pub messages: Vec<ChatMessage>,
    pub add_generation_prompt: bool,
}

pub struct ChatTemplate {
    env: minijinja::Environment<'static>,
    name: String,
}

impl ChatTemplate {
    pub fn new(template_source: String) -> Result<Self> {
        let mut env = minijinja::Environment::new();
        env.set_unknown_method_callback(minijinja_contrib::pycompat::unknown_method_callback);
        env.add_template_owned("chat", template_source)?;
        Ok(Self {
            env,
            name: "chat".into(),
        })
    }

    pub fn render(&self, req: &ChatRequest) -> Result<String> {
        let tmpl = self.env.get_template(&self.name)?;
        Ok(tmpl.render(minijinja::context! { messages => req.messages, add_generation_prompt => req.add_generation_prompt })?)
    }
}

pub fn sanitize_for_serving(tokenizer: &mut Tokenizer) {
    let _ = tokenizer.with_truncation(None);
    tokenizer.with_padding(None);
}

pub fn load_tokenizer(path: &std::path::Path) -> Result<Tokenizer> {
    let mut tokenizer = Tokenizer::from_file(path).map_err(|e| anyhow::anyhow!(e.to_string()))?;
    sanitize_for_serving(&mut tokenizer);
    Ok(tokenizer)
}

#[derive(Clone, Debug, Default)]
pub struct IncrementalDecoder {
    ids: Vec<u32>,
    prefix_offset: usize,
    read_offset: usize,
}

impl IncrementalDecoder {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn ids(&self) -> &[u32] {
        &self.ids
    }

    fn decode(tokenizer: &Tokenizer, ids: &[u32]) -> Result<String> {
        tokenizer
            .decode(ids, true)
            .map_err(|e| anyhow::anyhow!("detokenize: {e}"))
    }

    pub fn push(&mut self, tokenizer: &Tokenizer, id: u32) -> Result<Option<String>> {
        self.ids.push(id);
        let prefix = Self::decode(tokenizer, &self.ids[self.prefix_offset..self.read_offset])?;
        let full = Self::decode(tokenizer, &self.ids[self.prefix_offset..])?;
        if full.len() > prefix.len()
            && !full.ends_with('\u{FFFD}')
            && full.is_char_boundary(prefix.len())
        {
            let piece = full[prefix.len()..].to_string();
            self.prefix_offset = self.read_offset;
            self.read_offset = self.ids.len();
            Ok(Some(piece))
        } else {
            Ok(None)
        }
    }

    pub fn flush(&mut self, tokenizer: &Tokenizer) -> Result<Option<String>> {
        let prefix = Self::decode(tokenizer, &self.ids[self.prefix_offset..self.read_offset])?;
        let full = Self::decode(tokenizer, &self.ids[self.prefix_offset..])?;
        self.prefix_offset = self.ids.len();
        self.read_offset = self.ids.len();
        if full.len() > prefix.len() && full.is_char_boundary(prefix.len()) {
            Ok(Some(full[prefix.len()..].to_string()))
        } else {
            Ok(None)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tiny_tokenizer() -> Tokenizer {
        let json = r#"{
            "version": "1.0",
            "truncation": null,
            "padding": null,
            "added_tokens": [
                {"id": 6, "content": "<eos>", "single_word": false, "lstrip": false,
                 "rstrip": false, "normalized": false, "special": true}
            ],
            "normalizer": null,
            "pre_tokenizer": null,
            "post_processor": null,
            "decoder": {"type": "ByteLevel", "add_prefix_space": true,
                        "trim_offsets": true, "use_regex": true},
            "model": {
                "type": "BPE",
                "dropout": null,
                "unk_token": null,
                "continuing_subword_prefix": null,
                "end_of_word_suffix": null,
                "fuse_unk": false,
                "byte_fallback": false,
                "ignore_merges": false,
                "vocab": {"h": 0, "Ã": 1, "©": 2, "l": 3, "o": 4, "e": 5, "<eos>": 6},
                "merges": []
            }
        }"#;
        Tokenizer::from_bytes(json.as_bytes()).expect("tiny tokenizer")
    }

    fn run_incremental(tok: &Tokenizer, ids: &[u32]) -> (String, Vec<Option<String>>) {
        let mut dec = IncrementalDecoder::new();
        let mut out = String::new();
        let mut pieces = Vec::new();
        for &id in ids {
            let p = dec.push(tok, id).unwrap();
            if let Some(s) = &p {
                out.push_str(s);
            }
            pieces.push(p);
        }
        if let Some(s) = dec.flush(tok).unwrap() {
            out.push_str(&s);
        }
        (out, pieces)
    }

    #[test]
    fn holds_back_incomplete_utf8_across_tokens() {
        let tok = tiny_tokenizer();
        let ids = [0u32, 1, 2, 3];
        let (out, pieces) = run_incremental(&tok, &ids);
        assert_eq!(pieces[0].as_deref(), Some("h"));
        assert_eq!(pieces[1], None, "0xC3 alone must be held back");
        assert_eq!(pieces[2].as_deref(), Some("é"));
        assert_eq!(pieces[3].as_deref(), Some("l"));
        assert_eq!(out, tok.decode(&ids, true).unwrap());
        assert_eq!(out, "hél");
    }

    #[test]
    fn flush_emits_trailing_incomplete_sequence() {
        let tok = tiny_tokenizer();
        let ids = [0u32, 1];
        let (out, pieces) = run_incremental(&tok, &ids);
        assert_eq!(pieces[0].as_deref(), Some("h"));
        assert_eq!(pieces[1], None);
        assert_eq!(out, tok.decode(&ids, true).unwrap());
        assert!(out.ends_with('\u{FFFD}'));
    }

    #[test]
    fn skipped_special_tokens_emit_nothing() {
        let tok = tiny_tokenizer();
        let ids = [0u32, 6, 5, 3, 3, 4];
        let (out, pieces) = run_incremental(&tok, &ids);
        assert_eq!(pieces[1], None, "special token decodes to nothing");
        assert_eq!(out, tok.decode(&ids, true).unwrap());
        assert_eq!(out, "hello");
    }

    #[test]
    fn matches_full_decode_on_all_prefix_permutations() {
        let tok = tiny_tokenizer();
        let base = [0u32, 1, 2, 3, 3, 4, 6, 5, 1, 2];
        for n in 1..=base.len() {
            let ids = &base[..n];
            let (out, _) = run_incremental(&tok, ids);
            assert_eq!(out, tok.decode(ids, true).unwrap(), "prefix len {n}");
        }
    }

    fn truncating_tokenizer_json() -> &'static str {
        r#"{
            "version": "1.0",
            "truncation": {"direction": "Right", "max_length": 3,
                           "strategy": "LongestFirst", "stride": 0},
            "padding": null,
            "added_tokens": [],
            "normalizer": null,
            "pre_tokenizer": null,
            "post_processor": null,
            "decoder": null,
            "model": {
                "type": "BPE",
                "dropout": null,
                "unk_token": null,
                "continuing_subword_prefix": null,
                "end_of_word_suffix": null,
                "fuse_unk": false,
                "byte_fallback": false,
                "ignore_merges": false,
                "vocab": {"h": 0, "e": 1, "l": 2, "o": 3},
                "merges": []
            }
        }"#
    }

    #[test]
    fn shipped_truncation_actually_truncates_without_sanitize() {
        let tok = Tokenizer::from_bytes(truncating_tokenizer_json().as_bytes()).unwrap();
        let ids = tok.encode("hellohello", false).unwrap().get_ids().to_vec();
        assert_eq!(
            ids.len(),
            3,
            "precondition: raw load must truncate to max_length"
        );
    }

    #[test]
    fn sanitize_for_serving_disables_shipped_truncation() {
        let mut tok = Tokenizer::from_bytes(truncating_tokenizer_json().as_bytes()).unwrap();
        assert!(tok.get_truncation().is_some());
        sanitize_for_serving(&mut tok);
        assert!(tok.get_truncation().is_none());
        assert!(tok.get_padding().is_none());
        let ids = tok.encode("hellohello", false).unwrap().get_ids().to_vec();
        assert_eq!(
            ids.len(),
            10,
            "encode past the shipped max_length must not truncate"
        );
    }

    #[test]
    fn load_tokenizer_strips_shipped_truncation() {
        let dir =
            std::env::temp_dir().join(format!("nv-tokenizer-trunc-test-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("tokenizer.json");
        std::fs::write(&path, truncating_tokenizer_json()).unwrap();
        let tok = load_tokenizer(&path).unwrap();
        let _ = std::fs::remove_dir_all(&dir);
        assert!(tok.get_truncation().is_none());
        let ids = tok.encode("hellohello", false).unwrap().get_ids().to_vec();
        assert_eq!(ids.len(), 10);
    }

    const HUB_ALLOW_SKIP: &str = "NV_TOKENIZER_ALLOW_SKIP";

    fn hub_roots() -> Vec<std::path::PathBuf> {
        let mut roots: Vec<std::path::PathBuf> = Vec::new();
        for k in ["NV_HUB_CACHE", "HF_HUB_CACHE"] {
            if let Some(v) = std::env::var_os(k) {
                roots.push(std::path::PathBuf::from(v));
            }
        }
        if let Some(v) = std::env::var_os("HF_HOME") {
            roots.push(std::path::PathBuf::from(v).join("hub"));
        }
        if let Some(v) = std::env::var_os("HOME") {
            roots.push(std::path::PathBuf::from(v).join(".cache/huggingface/hub"));
        }
        roots.dedup();
        roots
    }

    fn hub_tokenizer(repo: &str) -> Option<std::path::PathBuf> {
        for root in hub_roots() {
            let repo_dir = root.join(repo);
            let snaps = repo_dir.join("snapshots");
            if let Ok(sha) = std::fs::read_to_string(repo_dir.join("refs/main")) {
                let p = snaps.join(sha.trim()).join("tokenizer.json");
                if p.is_file() {
                    return Some(p);
                }
            }
            let mut cands: Vec<std::path::PathBuf> = std::fs::read_dir(&snaps)
                .ok()
                .into_iter()
                .flatten()
                .flatten()
                .map(|e| e.path().join("tokenizer.json"))
                .filter(|p| p.is_file())
                .collect();
            cands.sort();
            if let Some(p) = cands.pop() {
                return Some(p);
            }
        }
        None
    }

    fn require_hub_tokenizer(test: &str, repo: &str) -> Option<std::path::PathBuf> {
        if let Some(p) = hub_tokenizer(repo) {
            return Some(p);
        }
        if std::env::var(HUB_ALLOW_SKIP).as_deref() == Ok("1") {
            eprintln!(
                "SKIP ({HUB_ALLOW_SKIP}=1): {test}: no {repo} tokenizer.json under any of \
                 {:?}. This is a SKIP, not a pass -- nothing in this test was exercised.",
                hub_roots()
            );
            return None;
        }
        panic!(
            "{test}: no {repo} tokenizer.json under any of {:?}. A hub snapshot directory is \
             the commit sha, never the literal `main`, so a hardcoded path reports success \
             while running nothing. Fetch the checkpoint or set {HUB_ALLOW_SKIP}=1.",
            hub_roots()
        );
    }

    #[test]
    fn qwen_nvfp4_tokenizer_encodes_past_4096_after_serving_load() {
        let Some(path) = require_hub_tokenizer(
            "qwen_nvfp4_tokenizer_encodes_past_4096_after_serving_load",
            "models--RedHatAI--Qwen3.6-35B-A3B-NVFP4",
        ) else {
            return;
        };
        let raw = Tokenizer::from_file(&path).expect("raw load");
        assert!(
            raw.get_truncation().is_some(),
            "expected the shipped tokenizer.json to configure truncation; \
             if upstream fixed it, this gate can be retired"
        );
        let tok = load_tokenizer(&path).unwrap();
        assert!(tok.get_truncation().is_none());
        let text = "The quick brown fox jumps over the lazy dog. ".repeat(1200);
        let n = tok.encode(text.as_str(), false).unwrap().get_ids().len();
        assert!(
            n > 4096,
            "long prompt must tokenize past the shipped 4096 cap, got {n}"
        );
    }

    #[test]
    fn laguna_tokenizer_multibyte_roundtrip() {
        let Some(path) = require_hub_tokenizer(
            "laguna_tokenizer_multibyte_roundtrip",
            "models--poolside--Laguna-XS-2.1-NVFP4",
        ) else {
            return;
        };
        let tok = load_tokenizer(&path).unwrap();
        let text = "Hello 世界! émoji 🚀🇺🇸 mixed ẞtraße -- done.";
        let ids = tok.encode(text, false).unwrap().get_ids().to_vec();
        let (out, pieces) = run_incremental(&tok, &ids);
        assert_eq!(out, tok.decode(&ids, true).unwrap());
        assert!(
            pieces.iter().any(|p| p.is_none()),
            "expected at least one held-back piece"
        );
    }
}
