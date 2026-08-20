pub mod dp;
pub mod output;
pub mod pipeline;

pub use dp::{frame_to_time_ms, viterbi_align, AlignError};
pub use output::{to_diarized_json, to_srt, to_vtt};
pub use pipeline::align_with_logprobs;

use serde::Serialize;

#[derive(Clone, Debug, Serialize)]
pub struct WordTiming {
    pub word: String,
    pub start: f32,
    pub end: f32,
}

#[derive(Clone, Debug, Serialize)]
pub struct AlignedSegment {
    pub text: String,
    pub start: f32,
    pub end: f32,
    pub words: Vec<WordTiming>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub speaker: Option<String>,
}

pub trait Aligner {
    fn align(
        &mut self,
        audio_pcm: &[f32],
        sample_rate: u32,
        text: &str,
    ) -> anyhow::Result<Vec<AlignedSegment>>;
}
