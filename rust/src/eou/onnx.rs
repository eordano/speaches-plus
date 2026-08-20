#![allow(dead_code)]

use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, OnceLock};

use anyhow::{anyhow, Context, Result};
use ort::session::{builder::GraphOptimizationLevel, Session};
use ort::value::Tensor;

use super::byte_map;
#[cfg(test)]
use super::chat_template;
use super::EouModel;
use crate::defaults;
use crate::vad::ort_err;

pub use super::bpe::Tokenizer;
#[allow(unused_imports)]
pub use super::chat_template::rolling_history;
pub use super::chat_template::{format_qwen_chat, Turn};

pub struct TextEouModel {
    session: Arc<Mutex<Session>>,
    tokenizer: Arc<Tokenizer>,
    max_ctx_tokens: usize,
    im_end_id: i64,
    has_attention_mask: bool,
}

impl TextEouModel {
    pub fn load(model_path: impl AsRef<Path>, tokenizer_path: impl AsRef<Path>) -> Result<Self> {
        Self::load_with_capacity(
            model_path,
            tokenizer_path,
            defaults::eou::MAX_CONTEXT_TOKENS as usize,
        )
    }

    pub fn load_with_capacity(
        model_path: impl AsRef<Path>,
        tokenizer_path: impl AsRef<Path>,
        max_ctx_tokens: usize,
    ) -> Result<Self> {
        let tok = Tokenizer::load_from_path(tokenizer_path.as_ref())
            .with_context(|| format!("load tokenizer {}", tokenizer_path.as_ref().display()))?;
        if tok.im_end_id() < 0 {
            return Err(anyhow!("tokenizer has no <|im_end|> token"));
        }
        let session = Session::builder()
            .map_err(ort_err)?
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .map_err(ort_err)?
            .with_intra_threads(1)
            .map_err(ort_err)?
            .commit_from_file(model_path.as_ref())
            .map_err(ort_err)
            .with_context(|| format!("load eou text model {}", model_path.as_ref().display()))?;
        let im_end_id = tok.im_end_id();
        Ok(Self {
            session: Arc::new(Mutex::new(session)),
            tokenizer: Arc::new(tok),
            max_ctx_tokens: if max_ctx_tokens == 0 {
                defaults::eou::MAX_CONTEXT_TOKENS as usize
            } else {
                max_ctx_tokens
            },
            im_end_id,
            has_attention_mask: true,
        })
    }

    pub fn from_parts(
        session: Arc<Mutex<Session>>,
        tokenizer: Arc<Tokenizer>,
        max_ctx_tokens: usize,
    ) -> Self {
        let im_end_id = tokenizer.im_end_id();
        Self {
            session,
            tokenizer,
            max_ctx_tokens: if max_ctx_tokens == 0 {
                defaults::eou::MAX_CONTEXT_TOKENS as usize
            } else {
                max_ctx_tokens
            },
            im_end_id,
            has_attention_mask: true,
        }
    }

    pub fn tokenizer(&self) -> &Tokenizer {
        &self.tokenizer
    }

    pub fn score_with_turns(&self, turns: &[Turn], partial: &str) -> f32 {
        let prompt = format_qwen_chat(turns, partial);
        self.score_prompt(&prompt)
    }

    pub fn score_prompt(&self, prompt: &str) -> f32 {
        let ids = self.tokenizer.encode(prompt);
        if ids.is_empty() {
            return defaults::eou::FAILURE_P_DEFAULT;
        }
        let truncated: Vec<i64> = if ids.len() > self.max_ctx_tokens {
            ids[ids.len() - self.max_ctx_tokens..].to_vec()
        } else {
            ids
        };
        match self.run_inference(&truncated) {
            Ok(p) => {
                if p.is_finite() && (0.0..=1.0).contains(&p) {
                    p
                } else {
                    f32::NAN
                }
            }
            Err(_) => f32::NAN,
        }
    }

    fn run_inference(&self, ids: &[i64]) -> Result<f32> {
        let n = ids.len();
        let input_ids: Vec<i64> = ids.to_vec();
        let mask: Vec<i64> = vec![1; n];

        let in_tensor = Tensor::<i64>::from_array(([1usize, n], input_ids.into_boxed_slice()))
            .map_err(ort_err)?;
        let mask_tensor =
            Tensor::<i64>::from_array(([1usize, n], mask.into_boxed_slice())).map_err(ort_err)?;

        let (shape_owned, logits) = {
            let mut session = self
                .session
                .lock()
                .map_err(|_| anyhow!("text-eou session poisoned"))?;
            let outputs = if self.has_attention_mask {
                session
                    .run(ort::inputs![
                        defaults::eou::INPUT_IDS => in_tensor,
                        defaults::eou::ATTENTION_MASK => mask_tensor,
                    ])
                    .map_err(ort_err)?
            } else {
                session
                    .run(ort::inputs![
                        defaults::eou::INPUT_IDS => in_tensor,
                    ])
                    .map_err(ort_err)?
            };
            let (shape, data) = outputs[defaults::eou::OUTPUT_LOGITS]
                .try_extract_tensor::<f32>()
                .map_err(ort_err)?;
            (shape.to_vec(), data.to_vec())
        };

        extract_im_end_prob(&logits, &shape_owned, self.im_end_id)
    }
}

impl EouModel for TextEouModel {
    fn score(&self, context: &str) -> f32 {
        self.score_with_turns(&[], context)
    }
}

pub fn extract_im_end_prob(logits: &[f32], shape: &[i64], im_end_id: i64) -> Result<f32> {
    if shape.len() < 2 || logits.is_empty() {
        return Err(anyhow!("empty logits (shape={:?})", shape));
    }
    let vocab = shape[shape.len() - 1] as usize;
    if vocab == 0 || im_end_id < 0 || (im_end_id as usize) >= vocab {
        return Err(anyhow!("im_end_id {} out of vocab {}", im_end_id, vocab));
    }
    if logits.len() < vocab {
        return Err(anyhow!("logits length {} < vocab {}", logits.len(), vocab));
    }
    let last_start = logits.len() - vocab;
    let row = &logits[last_start..last_start + vocab];
    let mut max_logit = row[0];
    for v in row.iter().copied().skip(1) {
        if v > max_logit {
            max_logit = v;
        }
    }
    let mut sum: f64 = 0.0;
    for v in row.iter().copied() {
        sum += ((v - max_logit) as f64).exp();
    }
    if sum <= 0.0 {
        return Err(anyhow!("degenerate logits (sum<=0)"));
    }
    let p = ((row[im_end_id as usize] - max_logit) as f64).exp() / sum;
    Ok(p as f32)
}

#[derive(Clone)]
pub struct TextEouPaths {
    pub model_path: PathBuf,
    pub tokenizer_path: PathBuf,
}

pub fn resolve_text_eou_paths() -> Option<TextEouPaths> {
    let model_path = std::env::var(defaults::env::EOU_MODEL_PATH)
        .ok()
        .map(PathBuf::from)?;
    if !model_path.exists() {
        return None;
    }
    let tokenizer_path = std::env::var(defaults::env::EOU_TOKENIZER_PATH)
        .ok()
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            let parent = model_path.parent().unwrap_or_else(|| Path::new("."));
            parent.join("tokenizer.json")
        });
    Some(TextEouPaths {
        model_path,
        tokenizer_path,
    })
}

pub fn build_mock_tokenizer_json() -> String {
    let mut vocab: Vec<(String, i64)> = Vec::new();
    let table = byte_map::byte_to_char_table();
    let mut id: i64 = 0;
    let mut seen: std::collections::HashSet<String> = std::collections::HashSet::new();
    for ch in table.iter() {
        let s = ch.to_string();
        if seen.insert(s.clone()) {
            vocab.push((s, id));
            id += 1;
        }
    }
    let extras = ["Ġh", "Ġhe", "Ġhel", "Ġhell", "Ġhello", "Ġworld"];
    for tok in extras {
        if seen.insert(tok.to_string()) {
            vocab.push((tok.to_string(), id));
            id += 1;
        }
    }
    let im_start_id = id;
    vocab.push((defaults::eou::IM_START.to_string(), im_start_id));
    id += 1;
    let im_end_id = id;
    vocab.push((defaults::eou::IM_END.to_string(), im_end_id));
    let merges = vec![
        "Ġ h", "Ġh e", "Ġhe l", "Ġhel l", "Ġhell o", "Ġ w", "Ġw o", "Ġwo r", "Ġwor l", "Ġworl d",
    ];
    let mut vocab_obj = serde_json::Map::new();
    for (k, v) in vocab {
        vocab_obj.insert(k, serde_json::Value::Number(v.into()));
    }
    let added = serde_json::json!([
        {"id": im_start_id, "content": defaults::eou::IM_START, "special": true},
        {"id": im_end_id, "content": defaults::eou::IM_END, "special": true},
    ]);
    let doc = serde_json::json!({
        "added_tokens": added,
        "model": {"type": "BPE", "vocab": vocab_obj, "merges": merges}
    });
    serde_json::to_string(&doc).expect("serialize mock tokenizer")
}

static SHARED_TEXT_EOU: OnceLock<Option<Arc<TextEouModel>>> = OnceLock::new();

pub fn shared_text_eou_model() -> Option<Arc<TextEouModel>> {
    SHARED_TEXT_EOU
        .get_or_init(|| {
            let paths = resolve_text_eou_paths()?;
            let max_ctx = std::env::var(defaults::env::EOU_MAX_CONTEXT_TOKENS)
                .ok()
                .and_then(|v| v.trim().parse::<usize>().ok())
                .unwrap_or(defaults::eou::MAX_CONTEXT_TOKENS as usize);
            match TextEouModel::load_with_capacity(
                &paths.model_path,
                &paths.tokenizer_path,
                max_ctx,
            ) {
                Ok(m) => {
                    tracing::info!(
                        path = %paths.model_path.display(),
                        tokenizer = %paths.tokenizer_path.display(),
                        max_ctx = max_ctx,
                        "eou: text ONNX model loaded"
                    );
                    Some(Arc::new(m))
                }
                Err(e) => {
                    tracing::warn!(
                        path = %paths.model_path.display(),
                        error = %e,
                        "eou: text ONNX model load failed; falling back to heuristic"
                    );
                    None
                }
            }
        })
        .clone()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chat_template_single_user_partial() {
        let got = format_qwen_chat(&[], "hello world");
        let want = "<|im_start|>user\nhello world";
        assert_eq!(got, want);
    }

    #[test]
    fn chat_template_prior_turns_then_partial() {
        let turns = vec![
            Turn::user("what's the weather"),
            Turn::assistant("it's sunny"),
        ];
        let got = format_qwen_chat(&turns, "and humid");
        let expected = "<|im_start|>user\nwhat's the weather<|im_end|>\n\
                        <|im_start|>assistant\nit's sunny<|im_end|>\n\
                        <|im_start|>user\nand humid";
        assert_eq!(got, expected);
    }

    #[test]
    fn chat_template_empty_partial_ends_with_im_end_newline() {
        let turns = vec![Turn::user("hi")];
        let got = format_qwen_chat(&turns, "");
        assert!(got.ends_with("<|im_end|>\n"), "{got}");
    }

    #[test]
    fn chat_template_defaults_role_to_user() {
        let t = Turn {
            role: String::new(),
            content: "no role".into(),
        };
        let got = format_qwen_chat(std::slice::from_ref(&t), "");
        assert!(got.contains("<|im_start|>user\nno role"), "{got}");
    }

    #[test]
    fn rolling_history_truncates() {
        let turns: Vec<Turn> = (1..=7)
            .map(|i| Turn {
                role: "user".into(),
                content: i.to_string(),
            })
            .collect();
        let got = chat_template::rolling_history(&turns, 4);
        assert_eq!(got.len(), 4);
        assert_eq!(got[0].content, "4");
        assert_eq!(got[3].content, "7");
    }

    #[test]
    fn byte_map_round_trip_all_bytes() {
        let table = byte_map::byte_to_char_table();
        for b in 0u32..256 {
            let ch = table[b as usize];
            let back = byte_map::char_to_byte(ch);
            assert_eq!(back, Some(b as u8), "byte {} ch {:?}", b, ch);
        }
    }

    #[test]
    fn byte_map_bpe_chars_round_trip() {
        for s in ["hello world", "line\nbreak", "\t\rweird", "\x00\x01\x02"] {
            let chars = byte_map::bytes_to_bpe_chars(s);
            let back = byte_map::bpe_chars_to_bytes(&chars);
            assert_eq!(back, s, "round-trip {s:?}");
        }
    }

    #[test]
    fn tokenizer_loads_mock_and_finds_im_end() {
        let raw = build_mock_tokenizer_json();
        let tok = Tokenizer::load_from_json(&raw).expect("load tokenizer");
        assert!(tok.im_end_id() >= 0, "im_end_id detected");
        assert!(tok.has_im_tokens());
    }

    #[test]
    fn tokenizer_encodes_special_tokens_atomically() {
        let raw = build_mock_tokenizer_json();
        let tok = Tokenizer::load_from_json(&raw).expect("load");
        let s = format!(
            "{}user\nhello{}",
            defaults::eou::IM_START,
            defaults::eou::IM_END
        );
        let ids = tok.encode(&s);
        assert!(
            ids.iter().any(|id| *id == tok.im_start_id()),
            "im_start present: {ids:?}"
        );
        assert!(
            ids.iter().any(|id| *id == tok.im_end_id()),
            "im_end present: {ids:?}"
        );
    }

    #[test]
    fn tokenizer_encode_decode_round_trip() {
        let raw = build_mock_tokenizer_json();
        let tok = Tokenizer::load_from_json(&raw).expect("load");
        let text = " hello world";
        let ids = tok.encode(text);
        let got = tok.decode(&ids);
        assert_eq!(got, text);
    }

    #[test]
    fn tokenizer_empty_returns_empty() {
        let raw = build_mock_tokenizer_json();
        let tok = Tokenizer::load_from_json(&raw).expect("load");
        assert!(tok.encode("").is_empty());
    }

    #[test]
    fn tokenizer_unsupported_model_errors() {
        let bad = serde_json::json!({"model": {"type": "WordPiece"}}).to_string();
        let r = Tokenizer::load_from_json(&bad);
        assert!(r.is_err(), "unsupported model type must error");
    }

    #[test]
    fn extract_im_end_prob_normalizes_via_softmax() {
        let vocab: usize = 5;
        let mut logits = vec![0.0f32; vocab * 2];
        logits[vocab + 3] = 100.0;
        let p = extract_im_end_prob(&logits, &[1, 2, vocab as i64], 3).expect("ok");
        assert!(p > 0.99, "expected near-1, got {p}");
    }

    #[test]
    fn extract_im_end_prob_rejects_oob_id() {
        let logits = vec![0.0f32; 5];
        let r = extract_im_end_prob(&logits, &[1, 1, 5], 9);
        assert!(r.is_err());
    }

    #[test]
    fn text_eou_model_load_skips_loud_when_artifacts_missing() {
        let model_path = std::path::PathBuf::from("/nonexistent/eou-model.onnx");
        let tok_path = std::path::PathBuf::from("/nonexistent/tokenizer.json");
        if !model_path.exists() || !tok_path.exists() {
            eprintln!(
                "[skip] text_eou_model_load: model files not present ({} / {})",
                model_path.display(),
                tok_path.display()
            );
            return;
        }
        let _ = TextEouModel::load(&model_path, &tok_path).expect("load");
    }

    #[test]
    fn text_eou_model_real_artifacts_when_env_set() {
        let Some(paths) = resolve_text_eou_paths() else {
            eprintln!("[skip] EOU_MODEL_PATH not set or model missing");
            return;
        };
        if !paths.tokenizer_path.exists() {
            eprintln!("[skip] tokenizer not at {}", paths.tokenizer_path.display());
            return;
        }
        let m =
            TextEouModel::load(&paths.model_path, &paths.tokenizer_path).expect("load real model");
        let p = m.score_with_turns(&[], "hello world");
        assert!(
            p.is_finite() && (0.0..=1.0).contains(&p),
            "score in [0,1]: got {p}"
        );
    }
}
