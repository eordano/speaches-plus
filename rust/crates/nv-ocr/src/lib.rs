mod api;
pub mod binarize;
pub mod ctc;
pub mod layout;
pub mod line;
pub mod lstm;
mod model;
pub mod raster;
pub mod simd;
pub mod traineddata;
pub mod unicharset;
pub mod vgsl;

pub use api::{
    BackendKind, ClassicalAnalyzer, ClassicalBackend, DeepSeekMode, DeepSeekOcr2Model, DotsOcrMode,
    DotsOcrModel, DotsPageResult, LayoutBox, LayoutPageResult, LineRecognizer, OcrBackend,
    OcrEngine, PageAnalyzer, ResolutionHint,
};
pub use ctc::DecodedChar;
pub use model::{
    expected_weight_groups, verify_weight_map, ModelOcrConfig, PreprocessSpec, TextConfig,
    VisionConfig, WeightGroup, MODEL_ID, WEIGHT_BYTES_BF16, WEIGHT_FILE,
    WEIGHT_FILE_BYTES_INCLUDING_SAFETENSORS_HEADER, WEIGHT_TENSOR_COUNT,
};
pub use traineddata::{LstmLineRecognizer, LstmModel, RecognizedLine, Traineddata};
pub use unicharset::{Recoder, Unicharset};

use serde::{Deserialize, Serialize};

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("decode: {0}")]
    Decode(String),
    #[error("binarize: {0}")]
    Binarize(String),
    #[error("layout: {0}")]
    Layout(String),
    #[error("traineddata: {0}")]
    Traineddata(String),
    #[error("network: {0}")]
    Network(String),
    #[error("ctc: {0}")]
    Ctc(String),
    #[error("backend: {0}")]
    Backend(String),
    #[error("model: {0}")]
    Model(String),
    #[error("io: {0}")]
    Io(#[from] std::io::Error),
    #[error("json: {0}")]
    Json(#[from] serde_json::Error),
    #[error("not wired: {0}")]
    NotWired(&'static str),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GreyImage {
    pub w: usize,
    pub h: usize,
    pub data: Vec<u8>,
}

impl GreyImage {
    pub fn new(w: usize, h: usize) -> Self {
        Self {
            w,
            h,
            data: vec![0; w * h],
        }
    }

    pub fn get(&self, x: usize, y: usize) -> u8 {
        self.data[y * self.w + x]
    }

    pub fn set(&mut self, x: usize, y: usize, v: u8) {
        self.data[y * self.w + x] = v;
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BinImage {
    pub w: usize,
    pub h: usize,
    pub stride: usize,
    pub bits: Vec<u64>,
}

impl BinImage {
    pub fn new(w: usize, h: usize) -> Self {
        let stride = w.div_ceil(64);
        Self {
            w,
            h,
            stride,
            bits: vec![0; stride * h],
        }
    }

    pub fn get(&self, x: usize, y: usize) -> bool {
        self.bits[y * self.stride + x / 64] >> (x % 64) & 1 == 1
    }

    pub fn set(&mut self, x: usize, y: usize, v: bool) {
        let word = &mut self.bits[y * self.stride + x / 64];
        if v {
            *word |= 1 << (x % 64);
        } else {
            *word &= !(1 << (x % 64));
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[cfg_attr(feature = "ts-bindings", derive(ts_rs::TS), ts(export))]
pub struct PixelRect {
    pub left: i32,
    pub top: i32,
    pub right: i32,
    pub bottom: i32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WordBox {
    pub rect: PixelRect,
}

#[derive(Debug, Clone)]
pub struct Line {
    pub bbox: PixelRect,
    pub words: Vec<WordBox>,
    pub grey_strip: GreyImage,
    pub baseline_y: f32,
    pub x_height: f32,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CharSpan {
    pub byte_start: usize,
    pub byte_end: usize,
    pub t_start: usize,
    pub t_end: usize,
    pub confidence: f32,
}

#[derive(Debug, Clone, Default)]
pub struct LineText {
    pub text: String,
    pub spans: Vec<CharSpan>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "ts-bindings", derive(ts_rs::TS), ts(export))]
pub struct OcrToken {
    pub start: usize,
    #[serde(rename = "endExclusive")]
    pub end_exclusive: usize,
    pub rect: PixelRect,
    pub confidence: f32,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[cfg_attr(feature = "ts-bindings", derive(ts_rs::TS), ts(export))]
pub struct OcrResult {
    pub text: String,
    pub tokens: Vec<OcrToken>,
    #[serde(default)]
    pub truncated: bool,
    #[serde(default)]
    pub looped: bool,
}

#[derive(Debug, Clone, Default)]
pub struct TessdataComponents {
    pub lstm: Option<Vec<u8>>,
    pub lstm_punc_dawg: Option<Vec<u8>>,
    pub lstm_word_dawg: Option<Vec<u8>>,
    pub lstm_number_dawg: Option<Vec<u8>>,
    pub lstm_unicharset: Option<Vec<u8>>,
    pub lstm_recoder: Option<Vec<u8>>,
    pub version: Option<String>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Logits {
    pub data: Vec<f32>,
    pub timesteps: usize,
    pub classes: usize,
}
