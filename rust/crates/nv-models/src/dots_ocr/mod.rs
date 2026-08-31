pub mod decoder;
#[cfg(feature = "wgpu")]
pub mod decoder_wgpu;
#[cfg(feature = "wgpu")]
pub mod decoder_wgpu_graph;
pub mod parse;
pub mod pipeline;
pub mod preprocess;
pub mod regions;
pub mod tiered;
pub mod vision;
#[cfg(feature = "wgpu")]
pub mod decode_gemv_msl;
#[cfg(feature = "wgpu")]
pub mod prefill_attn_msl;
pub mod vision_attn_msl;
#[cfg(feature = "wgpu")]
pub mod vision_wgpu;

pub use decoder::{
    build_prompt_tokens, DotsDecoder, DotsDecoderConfig, DotsKvCache, GenerateOptions,
    GenerateOutcome, PromptStyle, ASSISTANT_TOKEN_ID, ENDOFASSISTANT_TOKEN_ID, ENDOFIMG_TOKEN_ID,
    ENDOFTEXT_TOKEN_ID, ENDOFUSER_TOKEN_ID, EOS_TOKEN_IDS, IMGPAD_TOKEN_ID, IMG_TOKEN_ID,
    USER_TOKEN_ID,
};
#[cfg(feature = "wgpu")]
pub use decoder_wgpu_graph::{DotsDecoderWgpuGraph, DotsGraphQuant, RegionRow};
pub use parse::{parse_layout_json, LayoutElement, LayoutPage, CATEGORIES};
pub use pipeline::{
    DotsDecoderEngine, DotsMode, DotsOcrPipeline, DotsPageResult, DotsPrecision,
    DOTS_DEC_ENGINE_ENV, DOTS_PRECISION_ENV, PROMPT_LAYOUT_ALL_EN, PROMPT_LAYOUT_ONLY_EN,
    PROMPT_OCR,
};
pub use preprocess::{prepare, smart_resize, PixelBudget, PreparedImage};
pub use regions::{plan_regions, Region, CROP_MAX_PIXELS_ENV, LAYOUT_MAX_PIXELS_ENV};
pub use tiered::{TierPolicy, TIERED_ENV, TIERED_MIN_DOWNSCALE_ENV};
pub use vision::{DotsVisionConfig, DotsVisionTower};
#[cfg(feature = "wgpu")]
pub use vision_wgpu::{DotsVisionTowerWgpu, VisionGemmRoute};

pub use crate::deepseek_ocr::preprocess::RgbImage;
