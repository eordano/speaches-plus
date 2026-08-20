pub const TARGET_SAMPLE_RATE: u32 = 16_000;
pub const S16_SCALE: f32 = 32_768.0;
pub const S24_SCALE: f32 = 8_388_608.0;
pub const S32_SCALE: f32 = 2_147_483_648.0;
pub const BYTES_PER_S16: usize = 2;
pub const MIME_RAW_PCM: &str = "audio/pcm";
pub const MIME_RAW: &str = "audio/raw";

pub const MIN_DECODE_SAMPLE_RATE: usize = 1_000;
pub const MAX_DECODE_SAMPLE_RATE: usize = 384_000;

pub const ENV_MAX_DECODE_SECONDS: &str = "SPEACHES_MAX_AUDIO_SECONDS";
pub const DEFAULT_MAX_DECODE_SECONDS: f64 = 1_800.0;
