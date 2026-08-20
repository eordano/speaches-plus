pub mod audio_encoder;
pub mod codec_vel;
pub mod qwen3_vision;
pub mod talker;
pub mod thinker;
pub mod vision;
pub mod vocoder;

pub use audio_encoder::{audio_tokens_for_mel_frames, whisper_log_mel_128, AuTConfig, AudioEncoder};
pub use codec_vel::{LearnedVelField, LearnedVelFieldConfig};
pub use qwen3_vision::{Qwen3VisionConfig, Qwen3VisionTower};
pub use talker::{Talker, TalkerConfig};
pub use thinker::{
    build_mrope_positions, ModalitySplice, OmniDeepstack, OmniKvCache, OmniPositions,
    OmniSpecialIds, OmniThinker, OmniThinkerConfig,
};
pub use vision::{smart_resize, OmniVisionEncoder};
pub use vocoder::{
    Vocoder, VocoderConfig, FRAME_RATE_HZ, NUM_CODEBOOKS, PRE_UPSAMPLE_STRIDES, SAMPLES_PER_FRAME,
    SAMPLE_RATE_HZ, UPSAMPLE_STRIDES,
};

pub struct OmniConfig {
    pub model_id: String,
    pub audio_sample_rate: u32,
    pub speech_token_rate: u32,
}
