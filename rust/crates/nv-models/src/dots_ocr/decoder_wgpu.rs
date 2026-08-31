use anyhow::{Context, Result};
use std::path::Path;
use std::path::PathBuf;

use super::decoder::{argmax_banned, GenerateOptions, GenerateOutcome, EOS_TOKEN_IDS};
use crate::deepseek_ocr::decoder::{banned_tokens_windowed_ngram, detect_loop};
use crate::embed_row_splice::bf16_bits_round_nearest_even;
use crate::qwen3_5_dense_wgpu::{ImageRowSplice, Qwen3_5DenseConfig, Qwen3_5DenseWgpu};
use crate::qwen3_5_moe::LayerType;

pub const DOTS_DECODER_IS_QWEN2_ARCH_ON_THE_Q3D_GRAPH: &str =
    "the dots.ocr text decoder is a Qwen2.5-1.5B trunk (qkv projection biases, no qk-norm, \
     full rotary, untied lm_head); it rides Qwen3_5DenseWgpu's bias arm with every layer_type \
     FullAttention, and the unused linear_* fields carry benign nonzero placeholders because \
     no LinearAttention layer ever reads them";

pub fn dots_decoder_dense_config_from_hf_json_str(s: &str) -> Result<Qwen3_5DenseConfig> {
    let v: serde_json::Value = serde_json::from_str(s).context("parse dots_ocr config json")?;
    anyhow::ensure!(
        v.get("model_type").and_then(|x| x.as_str()) == Some("dots_ocr"),
        "config model_type {:?} != dots_ocr",
        v.get("model_type")
    );
    let get_u = |k: &str| -> Result<usize> {
        v.get(k)
            .and_then(|x| x.as_u64())
            .map(|x| x as usize)
            .ok_or_else(|| anyhow::anyhow!("missing/invalid {k}"))
    };
    let hidden_size = get_u("hidden_size")?;
    let num_attention_heads = get_u("num_attention_heads")?;
    let head_dim = match v.get("head_dim").and_then(|x| x.as_u64()) {
        Some(hd) => hd as usize,
        None => hidden_size / num_attention_heads,
    };
    let num_hidden_layers = get_u("num_hidden_layers")?;
    let eos_token_id = match v.get("eos_token_id") {
        Some(serde_json::Value::Array(a)) => a
            .first()
            .and_then(|x| x.as_u64())
            .ok_or_else(|| anyhow::anyhow!("empty eos_token_id list"))?
            as u32,
        Some(x) => x
            .as_u64()
            .ok_or_else(|| anyhow::anyhow!("invalid eos_token_id"))? as u32,
        None => anyhow::bail!("missing eos_token_id"),
    };
    Ok(Qwen3_5DenseConfig {
        hidden_size,
        num_hidden_layers,
        num_attention_heads,
        num_key_value_heads: get_u("num_key_value_heads")?,
        head_dim,
        intermediate_size: get_u("intermediate_size")?,
        vocab_size: get_u("vocab_size")?,
        max_position_embeddings: get_u("max_position_embeddings")?,
        rope_theta: v
            .get("rope_theta")
            .and_then(|x| x.as_f64())
            .unwrap_or(1_000_000.0) as f32,
        rms_norm_eps: v
            .get("rms_norm_eps")
            .and_then(|x| x.as_f64())
            .unwrap_or(1e-6),
        partial_rotary_factor: 1.0,
        bos_token_id: None,
        eos_token_id,
        layer_types: vec![LayerType::FullAttention; num_hidden_layers],
        linear_num_key_heads: 2,
        linear_num_value_heads: 4,
        linear_key_head_dim: 16,
        linear_value_head_dim: 16,
        linear_conv_kernel_dim: 4,
        attn_output_gate: false,
        tie_word_embeddings: v
            .get("tie_word_embeddings")
            .and_then(|x| x.as_bool())
            .unwrap_or(false),
    })
}

pub fn dots_decoder_dense_config_from_dir(dir: &Path) -> Result<Qwen3_5DenseConfig> {
    let p = dir.join("config.json");
    let s = std::fs::read_to_string(&p).with_context(|| format!("read {}", p.display()))?;
    dots_decoder_dense_config_from_hf_json_str(&s)
}

pub fn dots_decoder_wgpu_from_dir(dir: &Path, max_seq: usize) -> Result<Qwen3_5DenseWgpu> {
    dots_decoder_wgpu_from_dir_with_lora(dir, max_seq, None)
}

pub fn dots_decoder_wgpu_from_dir_with_lora(
    dir: &Path,
    max_seq: usize,
    adapter_dir: Option<&Path>,
) -> Result<Qwen3_5DenseWgpu> {
    let cfg = dots_decoder_dense_config_from_dir(dir)?;
    let loader = nv_weights::WeightLoader::open_dir(dir, &candle_core::Device::Cpu)
        .with_context(|| format!("open dots decoder weights in {}", dir.display()))?;
    let lora = adapter_dir
        .map(|d| crate::qwen3_5_dense_wgpu::Q3dLora::from_peft_dir(d, &cfg))
        .transpose()
        .with_context(|| "load dots lora adapter")?;
    if let (Some(l), Some(d)) = (&lora, adapter_dir) {
        eprintln!(
            "[dots] q3d lora adapter: {} modules matched at rank {} from {}",
            l.matched_modules(),
            l.rank(),
            d.display()
        );
    }
    Qwen3_5DenseWgpu::from_loader_with_lora(cfg, &loader, max_seq, lora.as_ref())
}

pub const DOTS_Q3D_MAX_SEQ_ENV_DEFAULT_16384: &str = "NV_DOTS_Q3D_MAX_SEQ";

pub const DOTS_Q3D_MLX_DIR_ENV_FALLS_BACK_TO_THE_HUB_SNAPSHOT: &str = "NV_DOTS_MLX_DIR";

pub const DOTS_Q3D_ARMS_SET_IF_UNSET_BECAUSE_THE_BIAS_LANE_NUMBERS_WERE_MEASURED_ON_THEM: [(
    &str,
    &str,
); 6] = [
    ("NV_Q3D_KV_FP8", "0"),
    ("NV_Q3D_ATTN_SHALLOW", "1"),
    ("NV_Q3D_BIAS_ROPE_KVW", "1"),
    ("NV_Q3D_AFFINE_MROW4H", "1"),
    ("NV_Q3D_AFFINE_GU_FUSED", "1"),
    (
        "NV_Q3D_QKV_FUSED_E2E_NULL_METAL_ALREADY_OVERLAPS_HAZARD_FREE_NEIGHBORS",
        "1",
    ),
];

pub fn dots_mlx_snapshot_dir() -> Result<PathBuf> {
    if let Ok(d) = std::env::var(DOTS_Q3D_MLX_DIR_ENV_FALLS_BACK_TO_THE_HUB_SNAPSHOT) {
        if !d.is_empty() {
            let p = PathBuf::from(&d);
            anyhow::ensure!(p.is_dir(), "{} = {d} is not a directory", DOTS_Q3D_MLX_DIR_ENV_FALLS_BACK_TO_THE_HUB_SNAPSHOT);
            return Ok(p);
        }
    }
    let home = std::env::var("HOME").context("HOME unset and NV_DOTS_MLX_DIR unset")?;
    let snaps = PathBuf::from(home)
        .join(".cache/huggingface/hub/models--mlx-community--dots.ocr-4bit/snapshots");
    let mut entries: Vec<PathBuf> = std::fs::read_dir(&snaps)
        .with_context(|| format!("no mlx-community/dots.ocr-4bit snapshot under {}", snaps.display()))?
        .flatten()
        .map(|e| e.path())
        .filter(|p| p.is_dir() && p.join("model.safetensors").exists())
        .collect();
    entries.sort();
    entries
        .pop()
        .ok_or_else(|| anyhow::anyhow!("empty mlx-community/dots.ocr-4bit snapshot dir"))
}

pub struct DotsQ3dDecoder {
    m: Qwen3_5DenseWgpu,
    max_seq: usize,
}

pub const DOTS_Q3D_LORA_DIR_ENV_HOT_LOADS_A_PEFT_ADAPTER_ON_THE_TRUNK: &str =
    "NV_DOTS_LORA_DIR";

impl DotsQ3dDecoder {
    pub fn from_snapshot() -> Result<Self> {
        let lora_dir = std::env::var(DOTS_Q3D_LORA_DIR_ENV_HOT_LOADS_A_PEFT_ADAPTER_ON_THE_TRUNK)
            .ok()
            .filter(|d| !d.is_empty())
            .map(PathBuf::from);
        Self::from_snapshot_with_lora(lora_dir.as_deref())
    }

    pub fn from_snapshot_with_lora(adapter_dir: Option<&Path>) -> Result<Self> {
        for (k, v) in DOTS_Q3D_ARMS_SET_IF_UNSET_BECAUSE_THE_BIAS_LANE_NUMBERS_WERE_MEASURED_ON_THEM
        {
            if std::env::var_os(k).is_none() {
                std::env::set_var(k, v);
            }
        }
        let dir = dots_mlx_snapshot_dir()?;
        let max_seq = std::env::var(DOTS_Q3D_MAX_SEQ_ENV_DEFAULT_16384)
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(16384);
        let m = dots_decoder_wgpu_from_dir_with_lora(&dir, max_seq, adapter_dir)?;
        Ok(Self { m, max_seq })
    }

    pub fn lora_resident(&self) -> bool {
        self.m.lora_resident()
    }

    pub fn pass_count(&self) -> usize {
        self.m.pass_count()
    }

    pub fn generate(
        &mut self,
        prompt_tokens: &[u32],
        vision_start: usize,
        vision_rows_f32: Option<&[f32]>,
        opts: &GenerateOptions,
    ) -> Result<GenerateOutcome> {
        let hidden = self.m.config().hidden_size;
        let prompt_len = prompt_tokens.len();
        anyhow::ensure!(prompt_len >= 2, "dots prompt needs at least two tokens");
        anyhow::ensure!(
            prompt_len + opts.max_new_tokens.min(64) < self.max_seq,
            "prompt {} tokens does not fit max_seq {} ({} raises it)",
            prompt_len,
            self.max_seq,
            DOTS_Q3D_MAX_SEQ_ENV_DEFAULT_16384
        );
        self.m.reset()?;
        let splices: Vec<ImageRowSplice> = match vision_rows_f32 {
            Some(f) => {
                anyhow::ensure!(
                    f.len().is_multiple_of(hidden),
                    "vision rows hold {} values, not a multiple of hidden {hidden}",
                    f.len()
                );
                let n_vis = f.len() / hidden;
                anyhow::ensure!(
                    vision_start + n_vis < prompt_len,
                    "vision span {vision_start}+{n_vis} must end before the prompt tail"
                );
                vec![ImageRowSplice {
                    position: vision_start,
                    rows_bf16: f.iter().map(|v| bf16_bits_round_nearest_even(*v)).collect(),
                }]
            }
            None => Vec::new(),
        };
        let done = self
            .m
            .prefill_tokens_with_image_rows(&prompt_tokens[..prompt_len - 1], &splices)?;
        anyhow::ensure!(
            done == prompt_len - 1,
            "chunked prefill covered {done} of {} prompt rows",
            prompt_len - 1
        );

        let eos: &[u32] = if opts.eos_token_ids.is_empty() {
            &EOS_TOKEN_IDS
        } else {
            &opts.eos_token_ids
        };
        let mut generated: Vec<u32> = Vec::new();
        let mut hit_eos = false;
        let mut prev = prompt_tokens[prompt_len - 1];
        for _ in 0..opts.max_new_tokens {
            if self.m.current_pos() + 1 >= self.max_seq {
                break;
            }
            let banned = match opts.ngram_size {
                Some(n) if n > 0 => banned_tokens_windowed_ngram(
                    &generated,
                    n,
                    opts.ngram_window,
                    &opts.ngram_whitelist,
                ),
                _ => Vec::new(),
            };
            let next = if banned.is_empty() {
                self.m.decode_step(prev)?
            } else {
                let (_, logits) = self.m.decode_step_logits(prev)?;
                argmax_banned(&logits, &banned)
            };
            if eos.contains(&next) {
                hit_eos = true;
                break;
            }
            generated.push(next);
            if opts.stop_on_loop
                && generated.len().is_multiple_of(64)
                && detect_loop(&generated).is_some()
            {
                break;
            }
            prev = next;
        }
        Ok(GenerateOutcome {
            loop_detection: detect_loop(&generated),
            tokens: generated,
            hit_eos,
        })
    }
}

const DOTS_Q3D_DECODER_IS_SEND_BECAUSE_THE_OCR_STREAM_ROUTE_HOLDS_IT_ACROSS_AWAIT: fn() = || {
    fn assert_send<T: Send>() {}
    assert_send::<DotsQ3dDecoder>();
};
