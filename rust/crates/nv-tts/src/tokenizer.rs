use std::collections::HashMap;
use std::path::{Path, PathBuf};

use anyhow::{anyhow, Context, Result};
use serde::Deserialize;
use tokenizers::{
    decoders::byte_level::ByteLevel as ByteLevelDecoder, models::bpe::BPE,
    pre_tokenizers::byte_level::ByteLevel as ByteLevelPre, AddedToken, DecoderWrapper,
    ModelWrapper, NormalizerWrapper, PostProcessorWrapper, PreTokenizerWrapper, Tokenizer,
    TokenizerBuilder,
};

pub const SPECIAL_IM_START: &str = "<|im_start|>";
pub const SPECIAL_IM_END: &str = "<|im_end|>";
pub const SPECIAL_ENDOFTEXT: &str = "<|endoftext|>";
pub const SPECIAL_AUDIO_START: &str = "<|audio_start|>";
pub const SPECIAL_AUDIO_END: &str = "<|audio_end|>";
pub const SPECIAL_AUDIO_PAD: &str = "<|audio_pad|>";
pub const SPECIAL_TTS_TEXT_BOS: &str = "<tts_text_bos>";
pub const SPECIAL_TTS_TEXT_BOS_SINGLE: &str = "<tts_text_bos_single>";
pub const SPECIAL_TTS_TEXT_EOD: &str = "<tts_text_eod>";
pub const SPECIAL_TTS_PAD: &str = "<tts_pad>";

pub const SPECIAL_ID_ENDOFTEXT: u32 = 151643;
pub const SPECIAL_ID_IM_START: u32 = 151644;
pub const SPECIAL_ID_IM_END: u32 = 151645;
pub const SPECIAL_ID_AUDIO_START: u32 = 151669;
pub const SPECIAL_ID_AUDIO_END: u32 = 151670;
pub const SPECIAL_ID_TTS_PAD: u32 = 151671;
pub const SPECIAL_ID_TTS_TEXT_BOS: u32 = 151672;
pub const SPECIAL_ID_TTS_TEXT_EOD: u32 = 151673;
pub const SPECIAL_ID_TTS_TEXT_BOS_SINGLE: u32 = 151674;
pub const SPECIAL_ID_AUDIO_PAD: u32 = 151675;

pub struct Qwen3TtsTokenizer {
    inner: Tokenizer,

    special_token_ids: HashMap<String, u32>,
}

#[derive(Debug, Deserialize)]
struct TokenizerConfig {
    #[serde(default)]
    added_tokens_decoder: HashMap<String, AddedTokenEntry>,
}

#[derive(Debug, Deserialize)]
struct AddedTokenEntry {
    content: String,
    #[serde(default)]
    lstrip: bool,
    #[serde(default)]
    rstrip: bool,
    #[serde(default)]
    single_word: bool,
    #[serde(default = "default_true")]
    normalized: bool,
    #[serde(default)]
    special: bool,
}

fn default_true() -> bool {
    true
}

impl Qwen3TtsTokenizer {
    pub fn from_dir(dir: &Path) -> Result<Self> {
        let vocab = dir.join("vocab.json");
        let merges = dir.join("merges.txt");
        let cfg_path = dir.join("tokenizer_config.json");
        if !vocab.is_file() {
            return Err(anyhow!("missing vocab.json in {}", dir.display()));
        }
        if !merges.is_file() {
            return Err(anyhow!("missing merges.txt in {}", dir.display()));
        }
        Self::from_files(
            &vocab,
            &merges,
            cfg_path.is_file().then_some(cfg_path.as_path()),
        )
    }

    pub fn from_files(vocab: &Path, merges: &Path, cfg_path: Option<&Path>) -> Result<Self> {
        let vocab_s = vocab
            .to_str()
            .ok_or_else(|| anyhow!("vocab path is not utf-8: {}", vocab.display()))?;
        let merges_s = merges
            .to_str()
            .ok_or_else(|| anyhow!("merges path is not utf-8: {}", merges.display()))?;

        let bpe: BPE = BPE::from_file(vocab_s, merges_s)
            .build()
            .map_err(|e| anyhow!("build BPE from {vocab_s} + {merges_s}: {e}"))?;

        let pre = ByteLevelPre::new(false, true, true);
        let dec = ByteLevelDecoder::new(false, true, true);

        let model: ModelWrapper = bpe.into();
        let mut tk: Tokenizer = TokenizerBuilder::<
            ModelWrapper,
            NormalizerWrapper,
            PreTokenizerWrapper,
            PostProcessorWrapper,
            DecoderWrapper,
        >::default()
        .with_model(model)
        .with_pre_tokenizer(Some(PreTokenizerWrapper::ByteLevel(pre)))
        .with_decoder(Some(DecoderWrapper::ByteLevel(dec)))
        .build()
        .map_err(|e| anyhow!("build tokenizer: {e}"))?
        .into();

        let mut special_token_ids: HashMap<String, u32> = HashMap::new();
        if let Some(p) = cfg_path {
            let raw = std::fs::read(p).with_context(|| format!("read {}", p.display()))?;
            let parsed: TokenizerConfig =
                serde_json::from_slice(&raw).with_context(|| format!("parse {}", p.display()))?;
            let mut entries: Vec<(u32, AddedTokenEntry)> = parsed
                .added_tokens_decoder
                .into_iter()
                .filter_map(|(k, v)| k.parse::<u32>().ok().map(|id| (id, v)))
                .collect();
            entries.sort_by_key(|(id, _)| *id);
            let added: Vec<AddedToken> = entries
                .iter()
                .map(|(_id, e)| {
                    AddedToken::from(e.content.clone(), e.special)
                        .lstrip(e.lstrip)
                        .rstrip(e.rstrip)
                        .single_word(e.single_word)
                        .normalized(e.normalized)
                })
                .collect();

            let _added_count = tk.add_special_tokens(&added);
            for (id, e) in entries {
                let actual = tk.token_to_id(&e.content);
                match actual {
                    Some(actual_id) => {
                        special_token_ids.insert(e.content.clone(), actual_id);
                        if actual_id != id {}
                    }
                    None => {
                        special_token_ids.insert(e.content.clone(), id);
                    }
                }
            }
        } else {
            let builtin = [
                (SPECIAL_ENDOFTEXT, SPECIAL_ID_ENDOFTEXT),
                (SPECIAL_IM_START, SPECIAL_ID_IM_START),
                (SPECIAL_IM_END, SPECIAL_ID_IM_END),
                (SPECIAL_AUDIO_START, SPECIAL_ID_AUDIO_START),
                (SPECIAL_AUDIO_END, SPECIAL_ID_AUDIO_END),
                (SPECIAL_TTS_PAD, SPECIAL_ID_TTS_PAD),
                (SPECIAL_TTS_TEXT_BOS, SPECIAL_ID_TTS_TEXT_BOS),
                (SPECIAL_TTS_TEXT_EOD, SPECIAL_ID_TTS_TEXT_EOD),
                (SPECIAL_TTS_TEXT_BOS_SINGLE, SPECIAL_ID_TTS_TEXT_BOS_SINGLE),
                (SPECIAL_AUDIO_PAD, SPECIAL_ID_AUDIO_PAD),
            ];
            let added: Vec<AddedToken> = builtin
                .iter()
                .map(|(s, _)| AddedToken::from(s.to_string(), true))
                .collect();
            let _ = tk.add_special_tokens(&added);
            for (s, id) in builtin {
                let actual = tk.token_to_id(s).unwrap_or(id);
                special_token_ids.insert(s.to_string(), actual);
            }
        }

        Ok(Self {
            inner: tk,
            special_token_ids,
        })
    }

    pub fn encode_text(&self, text: &str) -> Result<Vec<u32>> {
        let enc = self
            .inner
            .encode(text, false)
            .map_err(|e| anyhow!("encode: {e}"))?;
        Ok(enc.get_ids().to_vec())
    }

    pub fn decode_text(&self, ids: &[u32]) -> Result<String> {
        self.inner
            .decode(ids, false)
            .map_err(|e| anyhow!("decode: {e}"))
    }

    pub fn special_id(&self, literal: &str) -> Option<u32> {
        self.special_token_ids
            .get(literal)
            .copied()
            .or_else(|| self.inner.token_to_id(literal))
    }

    pub fn special_tokens(&self) -> Vec<&str> {
        self.special_token_ids.keys().map(|s| s.as_str()).collect()
    }

    pub fn bos_id(&self) -> u32 {
        self.special_id(SPECIAL_TTS_TEXT_BOS)
            .unwrap_or(SPECIAL_ID_TTS_TEXT_BOS)
    }

    pub fn eos_id(&self) -> u32 {
        self.special_id(SPECIAL_TTS_TEXT_EOD)
            .unwrap_or(SPECIAL_ID_TTS_TEXT_EOD)
    }

    pub fn inner(&self) -> &Tokenizer {
        &self.inner
    }
}

pub const QWEN3_TTS_SNAPSHOTS_UNDER_HOME: &str =
    ".cache/huggingface/hub/models--Qwen--Qwen3-TTS-12Hz-0.6B-CustomVoice/snapshots";

pub const QWEN3_TTS_SNAPSHOT_KEY_FILE: &str = "vocab.json";

pub fn qwen3_tts_cache_dir() -> Option<PathBuf> {
    let home = std::env::var_os("HOME")?;
    let base = PathBuf::from(home).join(QWEN3_TTS_SNAPSHOTS_UNDER_HOME);
    let entries = std::fs::read_dir(&base).ok()?;
    for e in entries.flatten() {
        let p = e.path();
        if p.is_dir() && p.join(QWEN3_TTS_SNAPSHOT_KEY_FILE).is_file() {
            return Some(p);
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn special_token_ids_compile_time_constants() {
        assert_eq!(SPECIAL_ID_IM_START + 1, SPECIAL_ID_IM_END);
        assert_eq!(SPECIAL_ID_TTS_TEXT_BOS + 1, SPECIAL_ID_TTS_TEXT_EOD);
    }

    #[test]
    fn encodes_decodes_round_trip() {
        let Some(dir) = crate::model_gate::require("tokenizer::encodes_decodes_round_trip") else {
            return;
        };
        let tk = Qwen3TtsTokenizer::from_dir(&dir).expect("load tokenizer");
        let text = "Hello world. This is a TTS round-trip test.";
        let ids = tk.encode_text(text).expect("encode");
        assert!(!ids.is_empty(), "expected at least one token");
        let decoded = tk.decode_text(&ids).expect("decode");

        assert_eq!(decoded, text, "byte-level BPE should be lossless on ASCII");

        assert_eq!(
            tk.special_id(SPECIAL_TTS_TEXT_BOS),
            Some(SPECIAL_ID_TTS_TEXT_BOS)
        );
        assert_eq!(
            tk.special_id(SPECIAL_TTS_TEXT_EOD),
            Some(SPECIAL_ID_TTS_TEXT_EOD)
        );
        assert_eq!(tk.bos_id(), SPECIAL_ID_TTS_TEXT_BOS);
        assert_eq!(tk.eos_id(), SPECIAL_ID_TTS_TEXT_EOD);

        let prompted = format!("{}Hello{}", SPECIAL_TTS_TEXT_BOS, SPECIAL_TTS_TEXT_EOD);
        let pids = tk.encode_text(&prompted).expect("encode prompted");
        assert!(pids.contains(&SPECIAL_ID_TTS_TEXT_BOS));
        assert!(pids.contains(&SPECIAL_ID_TTS_TEXT_EOD));
    }
}
