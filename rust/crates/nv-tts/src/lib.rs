pub mod code_predictor;
pub mod codec_decoder;
pub mod dense;
#[cfg(test)]
pub(crate) mod model_gate;
pub mod sampling;
pub mod speaker_encoder;
pub mod spk_mel;
pub mod streaming;
pub mod talker;
pub mod tokenizer;
pub mod vocoder_loader;
pub(crate) mod weight_helpers;
pub mod voice_profile;

pub use code_predictor::{CodecDecoderConfig, Qwen3TtsCodecDecoder, NUM_EXTRA_CODEBOOKS};
pub use codec_decoder::{
    Qwen3TtsCodecVocoder, CODEC_CODEBOOK_SIZE, CODEC_NUM_QUANTIZERS, CODEC_SAMPLES_PER_FRAME,
    CODEC_SAMPLE_RATE,
};
pub use sampling::{Sampler, SamplerConfig};
pub use speaker_encoder::{SpeakerEncoder, SpeakerEncoderConfig, CHECKPOINT_SPEAKER_PREFIX};
pub use spk_mel::{log_mel_24k, SPK_MEL_MIN_SAMPLES, SPK_MEL_N_MELS, SPK_MEL_SAMPLE_RATE};
pub use streaming::{TalkerLike, TtsAudioStream, TtsStream, DEFAULT_CHUNK_FRAMES};
pub use talker::{Qwen3TtsKvCache, Qwen3TtsTalker, Qwen3TtsTalkerConfig};
pub use tokenizer::{qwen3_tts_cache_dir, Qwen3TtsTokenizer};
pub use vocoder_loader::{
    load_from_qwen3_tts as load_vocoder_from_qwen3_tts, scan_vocoder_dir, vocoder_shard_path,
    VocoderInventory, VocoderLoadReport, DECODER_PREFIX, SPEECH_TOKENIZER_SHARD,
};
pub use voice_profile::{VoiceProfile, VoiceProfileStore};
