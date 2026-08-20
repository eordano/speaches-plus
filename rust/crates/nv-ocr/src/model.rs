use serde::Deserialize;

use crate::Error;

pub const MODEL_ID: &str = "stepfun-ai/GOT-OCR-2.0-hf";
pub const WEIGHT_FILE: &str = "model.safetensors";
pub const WEIGHT_BYTES_BF16: u64 = 1_121_057_280;
pub const WEIGHT_FILE_BYTES_INCLUDING_SAFETENSORS_HEADER: u64 = 1_121_114_488;
pub const WEIGHT_TENSOR_COUNT: usize = 471;

#[derive(Debug, Clone, Deserialize)]
pub struct ModelOcrConfig {
    pub model_type: String,
    pub image_seq_length: usize,
    pub image_token_index: u32,
    #[serde(default)]
    pub torch_dtype: Option<String>,
    pub text_config: TextConfig,
    #[serde(default)]
    pub vision_config: VisionConfig,
}

#[derive(Debug, Clone, Deserialize)]
pub struct TextConfig {
    pub model_type: String,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub vocab_size: usize,
    pub rope_theta: f64,
    #[serde(default)]
    pub tie_word_embeddings: bool,
    #[serde(default = "default_rms_norm_eps")]
    pub rms_norm_eps: f64,
}

fn default_rms_norm_eps() -> f64 {
    1e-6
}

#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct VisionConfig {
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_channels: usize,
    pub image_size: usize,
    pub patch_size: usize,
    pub output_channels: usize,
    pub mlp_dim: usize,
    pub window_size: usize,
    pub global_attn_indexes: Vec<usize>,
    pub qkv_bias: bool,
    pub use_abs_pos: bool,
    pub use_rel_pos: bool,
    pub layer_norm_eps: f64,
}

impl Default for VisionConfig {
    fn default() -> Self {
        Self {
            hidden_size: 768,
            num_hidden_layers: 12,
            num_attention_heads: 12,
            num_channels: 3,
            image_size: 1024,
            patch_size: 16,
            output_channels: 256,
            mlp_dim: 3072,
            window_size: 14,
            global_attn_indexes: vec![2, 5, 8, 11],
            qkv_bias: true,
            use_abs_pos: true,
            use_rel_pos: true,
            layer_norm_eps: 1e-6,
        }
    }
}

impl ModelOcrConfig {
    pub fn from_json_str(raw: &str) -> Result<Self, Error> {
        let cfg: Self = serde_json::from_str(raw)?;
        if cfg.model_type != "got_ocr2" {
            return Err(Error::Model(format!(
                "expected model_type got_ocr2, found {}",
                cfg.model_type
            )));
        }
        Ok(cfg)
    }

    pub fn vision_tokens(&self) -> usize {
        self.image_seq_length
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct PreprocessSpec {
    pub width: usize,
    pub height: usize,
    pub image_mean: [f32; 3],
    pub image_std: [f32; 3],
    pub rescale_factor: f32,
    pub resample_bicubic: bool,
    pub convert_rgb: bool,
}

impl PreprocessSpec {
    pub fn got_ocr2() -> Self {
        Self {
            width: 1024,
            height: 1024,
            image_mean: [0.481_454_66, 0.457_827_5, 0.408_210_73],
            image_std: [0.268_629_54, 0.261_302_6, 0.275_777_1],
            rescale_factor: 1.0 / 255.0,
            resample_bicubic: true,
            convert_rgb: true,
        }
    }

    pub fn pixel_len(&self) -> usize {
        3 * self.width * self.height
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WeightGroup {
    pub prefix: &'static str,
    pub expected: usize,
}

pub fn expected_weight_groups(cfg: &ModelOcrConfig) -> Vec<WeightGroup> {
    let v = &cfg.vision_config;
    let t = &cfg.text_config;
    let vision_per_layer = 11 + usize::from(v.qkv_bias) + if v.use_rel_pos { 2 } else { 0 };
    let mut groups = vec![
        WeightGroup {
            prefix: "vision_tower.patch_embed.",
            expected: 2,
        },
        WeightGroup {
            prefix: "vision_tower.pos_embed",
            expected: usize::from(v.use_abs_pos),
        },
        WeightGroup {
            prefix: "vision_tower.layers.",
            expected: v.num_hidden_layers * vision_per_layer,
        },
        WeightGroup {
            prefix: "vision_tower.neck.",
            expected: 6,
        },
        WeightGroup {
            prefix: "multi_modal_projector.",
            expected: 4,
        },
        WeightGroup {
            prefix: "language_model.model.embed_tokens.",
            expected: 1,
        },
        WeightGroup {
            prefix: "language_model.model.layers.",
            expected: t.num_hidden_layers * 12,
        },
        WeightGroup {
            prefix: "language_model.model.norm.",
            expected: 1,
        },
    ];
    if !t.tie_word_embeddings {
        groups.push(WeightGroup {
            prefix: "language_model.lm_head.",
            expected: 1,
        });
    }
    groups
}

pub fn verify_weight_map(cfg: &ModelOcrConfig, names: &[String]) -> Result<(), Error> {
    let groups = expected_weight_groups(cfg);
    let mut counts = vec![0usize; groups.len()];
    for name in names {
        match groups.iter().position(|g| name.starts_with(g.prefix)) {
            Some(i) => counts[i] += 1,
            None => return Err(Error::Model(format!("unexpected tensor {name}"))),
        }
    }
    for (group, count) in groups.iter().zip(counts.iter()) {
        if *count != group.expected {
            return Err(Error::Model(format!(
                "group {} has {} tensors, expected {}",
                group.prefix, count, group.expected
            )));
        }
    }
    Ok(())
}
