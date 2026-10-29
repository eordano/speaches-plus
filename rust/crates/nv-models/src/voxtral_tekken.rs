use std::collections::HashMap;

use anyhow::{anyhow, Result};

pub const TEKKEN_V7_PRETOKENIZER_WITHOUT_THE_TRAILING_SPACE_LOOKAHEAD: &str = r"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+|[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n/]*|\s*[\r\n]+|\s+";

pub const A_WHITESPACE_RUN_BEFORE_A_WORD_LEAVES_ITS_LAST_SPACE_TO_THE_WORD: &str =
    "tekken's pattern ends in `\\s+(?!\\S)|\\s+`: a whitespace run directly followed by a \
     non-space keeps all but its last character, so that last space is consumed by the next \
     chunk's optional leading non-letter; the regex crate has no lookahead, so the shortening \
     is applied after the match";

pub struct TekkenEncoder {
    ranks: HashMap<Vec<u8>, u32>,
    pattern: regex::Regex,
    num_special_tokens: u32,
}

impl TekkenEncoder {
    pub fn new(token_bytes: &[Vec<u8>], num_special_tokens: usize) -> Result<Self> {
        let mut ranks = HashMap::with_capacity(token_bytes.len());
        for (rank, bytes) in token_bytes.iter().enumerate() {
            if !bytes.is_empty() {
                ranks.insert(bytes.clone(), rank as u32);
            }
        }
        for b in 0..=255u8 {
            if ranks.get(&vec![b]) != Some(&(b as u32)) {
                return Err(anyhow!(
                    "tekken vocab is not byte-level: byte {b} is not the token of rank {b}"
                ));
            }
        }
        Ok(Self {
            ranks,
            pattern: regex::Regex::new(
                TEKKEN_V7_PRETOKENIZER_WITHOUT_THE_TRAILING_SPACE_LOOKAHEAD,
            )?,
            num_special_tokens: num_special_tokens as u32,
        })
    }

    pub fn pretokenize<'a>(&self, text: &'a str) -> Vec<&'a str> {
        let _ = A_WHITESPACE_RUN_BEFORE_A_WORD_LEAVES_ITS_LAST_SPACE_TO_THE_WORD;
        let mut out = Vec::new();
        let mut pos = 0;
        while pos < text.len() {
            let Some(m) = self.pattern.find_at(text, pos) else {
                break;
            };
            if m.start() != pos {
                out.push(&text[pos..m.start()]);
            }
            let mut end = m.end();
            let chunk = m.as_str();
            let followed_by_non_space = text[end..]
                .chars()
                .next()
                .is_some_and(|c| !c.is_whitespace());
            if chunk.chars().all(char::is_whitespace) && followed_by_non_space {
                let last = chunk.chars().last().map(char::len_utf8).unwrap_or(0);
                if chunk.len() > last {
                    end -= last;
                }
            }
            if end > pos {
                out.push(&text[pos..end]);
            }
            pos = end.max(pos + 1);
        }
        out
    }

    pub fn bpe(&self, piece: &[u8]) -> Vec<u32> {
        if let Some(&r) = self.ranks.get(piece) {
            return vec![r];
        }
        let mut parts: Vec<Vec<u8>> = piece.iter().map(|&b| vec![b]).collect();
        loop {
            let mut best: Option<(u32, usize)> = None;
            for i in 0..parts.len().saturating_sub(1) {
                let mut merged = parts[i].clone();
                merged.extend_from_slice(&parts[i + 1]);
                if let Some(&r) = self.ranks.get(&merged) {
                    if best.is_none_or(|(br, _)| r < br) {
                        best = Some((r, i));
                    }
                }
            }
            let Some((_, i)) = best else {
                break;
            };
            let right = parts.remove(i + 1);
            parts[i].extend_from_slice(&right);
        }
        parts.iter().map(|p| self.ranks[p]).collect()
    }

    pub fn encode(&self, text: &str) -> Vec<u32> {
        self.pretokenize(text)
            .into_iter()
            .flat_map(|chunk| self.bpe(chunk.as_bytes()))
            .map(|rank| rank + self.num_special_tokens)
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(serde::Deserialize)]
    struct Case {
        text: String,
        ids: Vec<u32>,
    }

    #[test]
    fn encodes_every_mistral_common_reference_case() {
        let dir = std::env::var_os("VOXTRAL_MODEL_DIR").map(std::path::PathBuf::from);
        let Some(dir) = dir else {
            eprintln!("skip: VOXTRAL_MODEL_DIR unset (tekken.json needed)");
            return;
        };
        let vocab = crate::voxtral::TekkenVocab::from_json_str(
            &std::fs::read_to_string(dir.join("tekken.json")).unwrap(),
        )
        .unwrap();
        let enc = vocab.encoder().unwrap();
        let cases: Vec<Case> = serde_json::from_slice(
            &std::fs::read(concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/../../tests/data/tekken_encode_reference.json"
            ))
            .unwrap(),
        )
        .unwrap();
        assert!(cases.len() > 10);
        for c in &cases {
            assert_eq!(enc.encode(&c.text), c.ids, "text {:?}", c.text);
        }
    }

    #[test]
    fn pretokenizer_splits_like_tiktoken_on_spaces_digits_and_punctuation() {
        let enc = TekkenEncoder {
            ranks: (0..=255u8).map(|b| (vec![b], b as u32)).collect(),
            pattern: regex::Regex::new(TEKKEN_V7_PRETOKENIZER_WITHOUT_THE_TRAILING_SPACE_LOOKAHEAD)
                .unwrap(),
            num_special_tokens: 0,
        };
        assert_eq!(enc.pretokenize("What are"), vec!["What", " are"]);
        assert_eq!(
            enc.pretokenize("  double  space"),
            vec![" ", " double", " ", " space"]
        );
        assert_eq!(
            enc.pretokenize("x=1; y+=2"),
            vec!["x", "=", "1", ";", " y", "+=", "2"]
        );
        assert_eq!(enc.pretokenize("end.\n"), vec!["end", ".\n"]);
        assert_eq!(enc.pretokenize(" 1"), vec![" ", "1"]);
        assert_eq!(enc.pretokenize("don't"), vec!["don", "'t"]);
        assert_eq!(
            enc.pretokenize("Año 2026"),
            vec!["Año", " ", "2", "0", "2", "6"]
        );
    }

    #[test]
    fn bpe_merges_by_lowest_rank_first() {
        let mut ranks: HashMap<Vec<u8>, u32> = (0..=255u8).map(|b| (vec![b], b as u32)).collect();
        ranks.insert(b"ab".to_vec(), 300);
        ranks.insert(b"bc".to_vec(), 290);
        ranks.insert(b"abc".to_vec(), 310);
        let enc = TekkenEncoder {
            ranks,
            pattern: regex::Regex::new(TEKKEN_V7_PRETOKENIZER_WITHOUT_THE_TRAILING_SPACE_LOOKAHEAD)
                .unwrap(),
            num_special_tokens: 1000,
        };
        assert_eq!(enc.bpe(b"abc"), vec![310]);
        assert_eq!(enc.bpe(b"abcd"), vec![310, b'd' as u32]);
        assert_eq!(enc.bpe(b"xbc"), vec![b'x' as u32, 290]);
        assert_eq!(enc.encode("abc"), vec![1310]);
    }
}
