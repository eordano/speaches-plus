pub mod decoder;
pub mod parse;
pub mod pipeline;
pub mod preprocess;
pub mod vision;

pub use decoder::{
    build_prompt_tokens, DotsDecoder, DotsDecoderConfig, DotsKvCache, GenerateOptions,
    GenerateOutcome, PromptStyle, ASSISTANT_TOKEN_ID, ENDOFASSISTANT_TOKEN_ID, ENDOFIMG_TOKEN_ID,
    ENDOFTEXT_TOKEN_ID, ENDOFUSER_TOKEN_ID, EOS_TOKEN_IDS, IMGPAD_TOKEN_ID, IMG_TOKEN_ID,
    USER_TOKEN_ID,
};
pub use parse::{parse_layout_json, LayoutElement, LayoutPage, CATEGORIES};
pub use pipeline::{
    DotsMode, DotsOcrPipeline, DotsPageResult, PROMPT_LAYOUT_ALL_EN, PROMPT_LAYOUT_ONLY_EN,
    PROMPT_OCR,
};
pub use preprocess::{prepare, smart_resize, PixelBudget, PreparedImage};
pub use vision::{DotsVisionConfig, DotsVisionTower};

pub use crate::deepseek_ocr::preprocess::RgbImage;
