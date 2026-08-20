#![allow(dead_code)]

use std::collections::HashMap;
use std::path::Path;

use anyhow::{anyhow, Context, Result};
use serde::Deserialize;

use super::byte_map::{bpe_chars_to_bytes, bytes_to_bpe_chars};
use super::special_trie::SpecialNode;
use crate::defaults;

#[derive(Default)]
pub struct Tokenizer {
    vocab: HashMap<String, i64>,
    id_to_token: HashMap<i64, String>,
    merges: HashMap<(String, String), usize>,
    added_tokens: HashMap<String, i64>,
    special_trie: SpecialNode,
    im_start_id: i64,
    im_end_id: i64,
    has_im_tokens: bool,
}

#[derive(Deserialize)]
struct AddedTokenJson {
    id: i64,
    content: String,
    #[serde(default)]
    #[allow(dead_code)]
    special: bool,
}

#[derive(Deserialize, Default)]
struct ModelJson {
    #[serde(default, rename = "type")]
    ty: String,
    #[serde(default)]
    vocab: HashMap<String, i64>,
    #[serde(default)]
    merges: Vec<serde_json::Value>,
}

#[derive(Deserialize)]
struct TokenizerJson {
    #[serde(default)]
    added_tokens: Vec<AddedTokenJson>,
    #[serde(default)]
    model: ModelJson,
}

impl Tokenizer {
    pub fn load_from_path(path: impl AsRef<Path>) -> Result<Self> {
        let raw = std::fs::read_to_string(path.as_ref())
            .with_context(|| format!("read tokenizer {}", path.as_ref().display()))?;
        Self::load_from_json(&raw)
    }

    pub fn load_from_json(raw: &str) -> Result<Self> {
        let tj: TokenizerJson = serde_json::from_str(raw).context("parse tokenizer json")?;
        if !tj.model.ty.is_empty() && tj.model.ty != "BPE" {
            return Err(anyhow!(
                "unsupported tokenizer model type {:?} (only BPE supported)",
                tj.model.ty
            ));
        }
        let mut t = Tokenizer {
            im_start_id: -1,
            im_end_id: -1,
            ..Default::default()
        };
        for (tok, id) in &tj.model.vocab {
            t.vocab.insert(tok.clone(), *id);
            t.id_to_token.insert(*id, tok.clone());
        }
        for (rank, m) in tj.model.merges.iter().enumerate() {
            if let Some((l, r)) = parse_merge(m) {
                t.merges.insert((l, r), rank);
            }
        }
        for at in &tj.added_tokens {
            t.added_tokens.insert(at.content.clone(), at.id);
            t.id_to_token.insert(at.id, at.content.clone());
            t.vocab.entry(at.content.clone()).or_insert(at.id);
            t.special_trie.insert(&at.content, at.id);
            if at.content == defaults::eou::IM_START {
                t.im_start_id = at.id;
            } else if at.content == defaults::eou::IM_END {
                t.im_end_id = at.id;
            }
        }
        if t.im_end_id < 0 {
            if let Some(id) = t.vocab.get(defaults::eou::IM_END) {
                t.im_end_id = *id;
            }
        }
        if t.im_start_id < 0 {
            if let Some(id) = t.vocab.get(defaults::eou::IM_START) {
                t.im_start_id = *id;
            }
        }
        t.has_im_tokens = t.im_end_id >= 0;
        Ok(t)
    }

    pub fn im_end_id(&self) -> i64 {
        self.im_end_id
    }

    pub fn im_start_id(&self) -> i64 {
        self.im_start_id
    }

    pub fn vocab_size(&self) -> usize {
        self.id_to_token.len()
    }

    pub fn has_im_tokens(&self) -> bool {
        self.has_im_tokens
    }

    pub fn encode(&self, text: &str) -> Vec<i64> {
        if text.is_empty() {
            return Vec::new();
        }
        let pieces = self.special_trie.split(text);
        let mut out: Vec<i64> = Vec::new();
        for p in pieces {
            if p.special {
                out.push(p.id);
                continue;
            }
            out.extend(self.encode_plain(&p.text));
        }
        out
    }

    fn encode_plain(&self, text: &str) -> Vec<i64> {
        if text.is_empty() {
            return Vec::new();
        }
        let splits = gpt2_pre_split(text);
        let mut out: Vec<i64> = Vec::new();
        for s in splits {
            let bpe_input = bytes_to_bpe_chars(&s);
            let merged = self.bpe_merges(&bpe_input);
            for tok in merged {
                if let Some(id) = self.vocab.get(&tok) {
                    out.push(*id);
                } else {
                    for r in tok.chars() {
                        let key: String = r.to_string();
                        if let Some(id) = self.vocab.get(&key) {
                            out.push(*id);
                        }
                    }
                }
            }
        }
        out
    }

    fn bpe_merges(&self, s: &str) -> Vec<String> {
        if s.is_empty() {
            return Vec::new();
        }
        let mut tokens: Vec<String> = s.chars().map(|c| c.to_string()).collect();
        loop {
            let mut best_rank = usize::MAX;
            let mut best_idx: i64 = -1;
            for i in 0..tokens.len().saturating_sub(1) {
                let key = (tokens[i].clone(), tokens[i + 1].clone());
                if let Some(rank) = self.merges.get(&key) {
                    if *rank < best_rank {
                        best_rank = *rank;
                        best_idx = i as i64;
                    }
                }
            }
            if best_idx < 0 {
                break;
            }
            let i = best_idx as usize;
            let merged = format!("{}{}", tokens[i], tokens[i + 1]);
            tokens.splice(i..i + 2, std::iter::once(merged));
        }
        tokens
    }

    pub fn decode(&self, ids: &[i64]) -> String {
        let mut joined = String::new();
        for id in ids {
            if let Some(tok) = self.id_to_token.get(id) {
                joined.push_str(tok);
            }
        }
        bpe_chars_to_bytes(&joined)
    }
}

fn parse_merge(raw: &serde_json::Value) -> Option<(String, String)> {
    match raw {
        serde_json::Value::String(s) => {
            let i = s.find(' ')?;
            if i == 0 || i >= s.len() - 1 {
                return None;
            }
            Some((s[..i].to_string(), s[i + 1..].to_string()))
        }
        serde_json::Value::Array(a) => {
            if a.len() != 2 {
                return None;
            }
            let l = a[0].as_str()?.to_string();
            let r = a[1].as_str()?.to_string();
            Some((l, r))
        }
        _ => None,
    }
}

pub fn gpt2_pre_split(text: &str) -> Vec<String> {
    let chars: Vec<char> = text.chars().collect();
    let n = chars.len();
    let mut out: Vec<String> = Vec::new();
    let mut i = 0usize;
    while i < n {
        if chars[i] == '\'' && i + 1 < n {
            let next = chars[i + 1];
            if matches!(next, 's' | 'd' | 'm' | 't') {
                out.push(chars[i..i + 2].iter().collect());
                i += 2;
                continue;
            }
            if i + 2 < n {
                let two: String = chars[i + 1..i + 3].iter().collect();
                if matches!(two.as_str(), "ll" | "ve" | "re") {
                    out.push(chars[i..i + 3].iter().collect());
                    i += 3;
                    continue;
                }
            }
        }
        let start = i;
        let leading_space = chars[i] == ' ';
        let probe = if leading_space { i + 1 } else { i };
        if probe < n {
            let c = chars[probe];
            if is_letter(c) {
                let mut j = probe;
                while j < n && is_letter(chars[j]) {
                    j += 1;
                }
                if j > probe {
                    out.push(chars[start..j].iter().collect());
                    i = j;
                    continue;
                }
            }
            if is_number(c) {
                let mut j = probe;
                while j < n && is_number(chars[j]) {
                    j += 1;
                }
                if j > probe {
                    out.push(chars[start..j].iter().collect());
                    i = j;
                    continue;
                }
            }
            if !c.is_whitespace() && !is_letter(c) && !is_number(c) {
                let mut j = probe;
                while j < n
                    && !chars[j].is_whitespace()
                    && !is_letter(chars[j])
                    && !is_number(chars[j])
                {
                    j += 1;
                }
                if j > probe {
                    out.push(chars[start..j].iter().collect());
                    i = j;
                    continue;
                }
            }
        }
        if chars[i].is_whitespace() {
            let mut j = i;
            while j < n && chars[j].is_whitespace() {
                j += 1;
            }
            if j < n {
                let take_to = j;
                if take_to - 1 > i {
                    out.push(chars[i..take_to - 1].iter().collect());
                }
                out.push(chars[take_to - 1..j + 1].iter().collect());
                i = j + 1;
                continue;
            } else {
                out.push(chars[i..j].iter().collect());
                i = j;
                continue;
            }
        }
        out.push(chars[i..i + 1].iter().collect());
        i += 1;
    }
    out
}

fn is_letter(c: char) -> bool {
    c.is_alphabetic()
}

fn is_number(c: char) -> bool {
    c.is_numeric()
}
