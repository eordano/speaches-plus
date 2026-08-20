#![allow(dead_code)]

pub const RFC_VERSION: &str = "v3";

pub mod session {
    pub const MAX_DURATION_S: u64 = 1800;
    pub const MAX_DURATION_HARD_CAP_S: u64 = 3600;
}

pub mod turn_detection {
    pub const THRESHOLD: f32 = 0.5;
    pub const PREFIX_PADDING_MS: u32 = 300;
    pub const SILENCE_DURATION_MS: u32 = 350;
    pub const BARGE_IN_DELAY_MS: u32 = 0;
    pub const CREATE_RESPONSE: bool = true;

    pub const PREFIX_PADDING_MS_MAX: u32 = 1000;
    pub const SILENCE_DURATION_MS_MIN: u32 = 50;
    pub const SILENCE_DURATION_MS_MAX: u32 = 5000;
    pub const BARGE_IN_DELAY_MS_MAX: u32 = 1000;
}

pub mod eou {
    pub const P_THRESHOLD: f32 = 0.5;
    pub const MIN_DELAY_MS: u32 = 500;
    pub const MAX_DELAY_MS: u32 = 3000;
    pub const SILENCE_HARD_CAP_MS: u32 = 5000;
    pub const INFERENCE_TIMEOUT_MS: u32 = 250;
    pub const CONTEXT_TURNS: u32 = 4;
    pub const MAX_CONTEXT_TOKENS: u32 = 128;
    pub const AUDIO_WINDOW_MS: u32 = 8000;
    pub const CURVE_K: f32 = 12.0;
    pub const FAILURE_P_DEFAULT: f32 = 1.0;

    pub const FUSION_WEIGHT_TEXT: f32 = 0.5;

    pub const FUSION_RULE: &str = "gated";

    pub const CURVE_K_MAX: f32 = 30.0;
    pub const SILENCE_HARD_CAP_MS_MAX: u32 = 60_000;
    pub const INFERENCE_TIMEOUT_MS_MAX: u32 = 10_000;
    pub const CONTEXT_TURNS_MAX: u32 = 64;
    pub const SESSION_MAX_DURATION_S_MAX: u64 = 86_400;

    pub const EAGER_P_THRESHOLD_DISABLED: f32 = 1.0;
    pub const EAGER_P_THRESHOLD: f32 = 0.5;
    pub const EAGER_MAX_INFLIGHT: u32 = 1;
    pub const EAGER_PERIODIC_ENABLED: bool = false;
    pub const EAGER_INTERVAL_MS: u32 = 250;
    pub const PREDICTED_TOKEN_BUFFER_CAP: u32 = 256;
    pub const EOT_THRESHOLD: f32 = 0.7;
    pub const EAGER_EOT_THRESHOLD: f32 = 0.5;
    pub const EAGER_TRANSCRIPT_MISMATCH_RATIO: f32 = 0.5;

    pub const INPUT_IDS: &str = "input_ids";
    pub const ATTENTION_MASK: &str = "attention_mask";
    pub const OUTPUT_LOGITS: &str = "logits";

    pub const IM_START: &str = "<|im_start|>";
    pub const IM_END: &str = "<|im_end|>";

    pub mod eagerness {
        pub const LOW: (f32, u32, u32) = (0.7, 800, 3000);
        pub const MEDIUM: (f32, u32, u32) = (0.5, 500, 2500);
        pub const HIGH: (f32, u32, u32) = (0.4, 300, 1500);
    }

    pub mod audio {
        pub const SAMPLE_RATE: u32 = 16_000;
        pub const N_MELS: usize = 80;
        pub const N_FFT: usize = 400;
        pub const HOP_LENGTH: usize = 160;
        pub const CHUNK_LENGTH_S: usize = 8;
        pub const TARGET_SAMPLES: usize = CHUNK_LENGTH_S * SAMPLE_RATE as usize;
        pub const N_FRAMES: usize = TARGET_SAMPLES / HOP_LENGTH;
    }
}

pub mod buffer {
    pub const MIN_SPEECH_MS: u64 = 100;
    pub const MIN_SPEECH_FOR_RESPONSE_MS: u64 = 600;
    pub const SEALED_BUFFER_RETENTION_COUNT: usize = 4;
    pub const PARTIAL_INTERVAL_MS: u64 = 500;

    pub const MIN_SPEECH_MS_MAX: u64 = 60_000;
    pub const MIN_SPEECH_FOR_RESPONSE_MS_MAX: u64 = 60_000;
    pub const SEALED_BUFFER_RETENTION_COUNT_MAX: u32 = 1024;
}

pub mod response {
    pub const DRAIN_CAP_FLOOR_MS: u64 = 5_000;
    pub const DRAIN_CAP_CEILING_MS: u64 = 60_000;
}

pub mod wire {
    pub const OUTBOUND_QUEUE_CAP_EVENTS: u32 = 256;
    pub const OUTBOUND_QUEUE_CAP_MS: u64 = 5_000;
    pub const DATA_CHANNEL_FRAGMENT_MAX: usize = 900;
}

pub mod inspector {
    pub const TRANSITIONS_ENABLED: bool = false;
    pub const TRANSITIONS_SAMPLE_RATE_DEV: f32 = 1.0;
    pub const TRANSITIONS_SAMPLE_RATE_RELEASE: f32 = 0.0;
    pub const RELAY_CAP: u32 = 1024;
}

pub mod inspect {
    pub const RETENTION_COUNT: usize = 200;
    pub const RETENTION_BYTES: u64 = 500_000_000;
    pub const RETENTION_DAYS: u64 = 30;
}

pub mod vad {
    pub const SAMPLE_RATE: usize = 16_000;
    pub const WINDOW_SAMPLES: usize = 512;
    pub const CONTEXT_SAMPLES: usize = 64;
    pub const INPUT_SAMPLES: usize = CONTEXT_SAMPLES + WINDOW_SAMPLES;
    pub const FAILURE_THRESHOLD: u32 = 3;
}

pub mod vad_window {
    use super::vad::SAMPLE_RATE;

    pub const MAX_VAD_WINDOW_MS: usize = 3_000;
    pub const MAX_VAD_WINDOW_SAMPLES: usize = MAX_VAD_WINDOW_MS * SAMPLE_RATE / 1_000;

    pub const MIN_SPEECH_DURATION_MS: u32 = 100;

    pub const MAX_SPEECH_DURATION_S: f32 = 30.0;

    pub const MIN_SILENCE_AT_MAX_SPEECH_MS: u32 = 98;

    pub const MAX_SPEECH_CARRY_OVER_MS: u32 = 300;

    pub const NEG_THRESHOLD_DELTA: f32 = 0.15;
    pub const NEG_THRESHOLD_FLOOR: f32 = 0.01;
}

pub mod audio_format {
    pub const PCM16: &str = "pcm16";
    pub const PCM16_16K: &str = "pcm16_16k";
    pub const G711_ULAW: &str = "g711_ulaw";
    pub const G711_ALAW: &str = "g711_alaw";

    pub const SUPPORTED: &[&str] = &[PCM16, PCM16_16K, G711_ULAW, G711_ALAW];

    pub const DEFAULT: &str = PCM16;
}

pub mod session_object {
    pub const REALTIME_SESSION: &str = "realtime.session";
}

pub mod modality {
    pub const TEXT: &str = "text";
    pub const AUDIO: &str = "audio";

    pub const DEFAULT_PAIR: &[&str] = &[TEXT, AUDIO];
    pub const TEXT_ONLY: &[&str] = &[TEXT];
}

pub mod turn_detection_type {
    pub const SERVER_VAD: &str = "server_vad";
    pub const NONE: &str = "none";
}

pub mod failure_delay {
    pub const MIN: &str = "min";
    pub const MAX: &str = "max";
}

pub mod audio {
    pub const TTS_SAMPLE_RATE: usize = 24_000;
    pub const OUT_SAMPLE_RATE: usize = 48_000;
    pub const FRAME_MS: usize = 20;
    pub const FRAME_SAMPLES: usize = OUT_SAMPLE_RATE * FRAME_MS / 1000;
    pub const IN_CHUNK_SAMPLES: usize = 480;

    pub const INPUT_HZ: f64 = 48_000.0;
    pub const OUTPUT_HZ: f64 = 16_000.0;
    pub const MAX_DECODE_FRAMES: usize = 5_760;

    pub const OPUS_SAMPLE_RATE_HZ: u32 = 48_000;
    pub const OPUS_ENCODE_BUFFER_BYTES: usize = 4_000;
    pub const TTS_RESAMPLER_RATIO: f64 = OUT_SAMPLE_RATE as f64 / TTS_SAMPLE_RATE as f64;
}

pub mod stt {
    pub const SILENCE_PEAK_THRESHOLD: f32 = 0.005;
    pub const CT2_THREADS_DEFAULT: usize = 2;

    pub const BEAM_SIZE: usize = 5;
    pub const BEAM_SIZE_ENV_RESTORES_THE_GREEDY_DECODE: &str = "SPEACHES_STT_BEAM_SIZE";
}

pub mod kokoro {
    pub const MAX_PHONEME_LENGTH: usize = 510;

    pub const INTRA_THREADS_MIN: usize = 1;
    pub const INTRA_THREADS_MAX: usize = 4;

    pub const MAX_INPUT_CHARS: usize = 100_000;
    pub const CHUNK_CHARS: usize = 1_500;
    pub const SYNTH_BUDGET_UNIT_CHARS: usize = 8_192;
    pub const JOIN_SILENCE_MS: usize = 50;
    pub const QUEUE_WAIT_S: u64 = 30;
    pub const SYNTH_BUDGET_S: u64 = 300;
}

pub mod ws {
    pub const MAX_MESSAGE_BYTES: usize = 4 * 1024 * 1024;
    pub const OUTBOUND_QUEUE_CAP: usize = 256;
    pub const IDLE_TIMEOUT_S: u64 = 60;
    pub const MAX_CONCURRENT_SESSIONS: usize = 64;
    pub const PING_INTERVAL_S: u64 = 20;
}

pub mod env {
    pub const SESSION_MAX_DURATION_S: &str = "SESSION_MAX_DURATION_S";
    pub const BARGE_IN_DELAY_MS: &str = "BARGE_IN_DELAY_MS";
    pub const PARTIAL_STT_ENABLED: &str = "PARTIAL_STT_ENABLED";
    pub const SPEACHES_PLUS_MODELS: &str = "SPEACHES_PLUS_MODELS";

    pub const WS_MAX_MESSAGE_BYTES: &str = "WS_MAX_MESSAGE_BYTES";
    pub const WS_OUTBOUND_QUEUE_CAP: &str = "WS_OUTBOUND_QUEUE_CAP";
    pub const WS_IDLE_TIMEOUT_S: &str = "WS_IDLE_TIMEOUT_S";
    pub const WS_MAX_CONCURRENT_SESSIONS: &str = "WS_MAX_CONCURRENT_SESSIONS";

    pub const STT_BACKEND: &str = "STT_BACKEND";

    pub const NV_SERVE_BACKEND: &str = "NV_SERVE_BACKEND";

    pub const KOKORO_INTRA_THREADS: &str = "KOKORO_INTRA_THREADS";
    pub const KOKORO_ONNX_PROVIDER: &str = "KOKORO_ONNX_PROVIDER";
    pub const VAD_COMMIT_TAIL: &str = "VAD_COMMIT_TAIL";
    pub const EOU_TEXT_HEAD_PATH: &str = "EOU_TEXT_HEAD_PATH";
    pub const KOKORO_MAX_INPUT_CHARS: &str = "KOKORO_MAX_INPUT_CHARS";
    pub const KOKORO_CHUNK_CHARS: &str = "KOKORO_CHUNK_CHARS";
    pub const KOKORO_QUEUE_WAIT_S: &str = "KOKORO_QUEUE_WAIT_S";
    pub const KOKORO_SYNTH_BUDGET_S: &str = "KOKORO_SYNTH_BUDGET_S";
    pub const KOKORO_WARMUP: &str = "KOKORO_WARMUP";

    pub const OUTBOUND_QUEUE_CAP_MS: &str = "OUTBOUND_QUEUE_CAP_MS";
    pub const OUTBOUND_QUEUE_CAP: &str = "OUTBOUND_QUEUE_CAP";

    pub const INSPECTOR_TRANSITIONS: &str = "INSPECTOR_TRANSITIONS";
    pub const INSPECTOR_TRANSITIONS_SAMPLE_RATE: &str = "INSPECTOR_TRANSITIONS_SAMPLE_RATE";

    pub const CHAT_COMPLETION_BASE_URL: &str = "CHAT_COMPLETION_BASE_URL";
    pub const CHAT_COMPLETION_API_KEY: &str = "CHAT_COMPLETION_API_KEY";
    pub const DEFAULT_REALTIME_CONVERSATION_MODEL: &str = "DEFAULT_REALTIME_CONVERSATION_MODEL";

    pub const EOU_ENABLED: &str = "EOU_ENABLED";
    pub const EOU_KIND: &str = "EOU_KIND";
    pub const EOU_EAGERNESS: &str = "EOU_EAGERNESS";
    pub const EOU_P_THRESHOLD: &str = "EOU_P_THRESHOLD";
    pub const EOU_MIN_DELAY_MS: &str = "EOU_MIN_DELAY_MS";
    pub const EOU_MAX_DELAY_MS: &str = "EOU_MAX_DELAY_MS";
    pub const EOU_SILENCE_HARD_CAP_MS: &str = "EOU_SILENCE_HARD_CAP_MS";
    pub const EOU_INFERENCE_TIMEOUT_MS: &str = "EOU_INFERENCE_TIMEOUT_MS";
    pub const EOU_CONTEXT_TURNS: &str = "EOU_CONTEXT_TURNS";
    pub const EOU_AUDIO_WINDOW_MS: &str = "EOU_AUDIO_WINDOW_MS";
    pub const EOU_AUDIO_PAD_ALIGNMENT: &str = "EOU_AUDIO_PAD_ALIGNMENT";
    pub const EOU_THRESHOLDS: &str = "EOU_THRESHOLDS";
    pub const EOU_EAGER_P_THRESHOLD: &str = "EOU_EAGER_P_THRESHOLD";
    pub const EOU_EAGER_MAX_INFLIGHT: &str = "EOU_EAGER_MAX_INFLIGHT";
    pub const EOU_EAGER_PERIODIC: &str = "EOU_EAGER_PERIODIC";
    pub const EOU_EAGER_INTERVAL_MS: &str = "EOU_EAGER_INTERVAL_MS";
    pub const EOU_PREDICTED_TOKEN_BUFFER_CAP: &str = "EOU_PREDICTED_TOKEN_BUFFER_CAP";
    pub const EOU_EOT_THRESHOLD: &str = "EOU_EOT_THRESHOLD";
    pub const EOU_EAGER_EOT_THRESHOLD: &str = "EOU_EAGER_EOT_THRESHOLD";
    pub const EOU_FUSION_RULE: &str = "EOU_FUSION_RULE";
    pub const EOU_FUSION_WEIGHT_TEXT: &str = "EOU_FUSION_WEIGHT_TEXT";
    pub const EOU_AUDIO_MODEL_PATH: &str = "EOU_AUDIO_MODEL_PATH";
    pub const EOU_AUDIO_REQUIRED: &str = "EOU_AUDIO_REQUIRED";

    pub const EOU_MODEL_PATH: &str = "EOU_MODEL_PATH";
    pub const EOU_TOKENIZER_PATH: &str = "EOU_TOKENIZER_PATH";
    pub const EOU_VOCAB_PATH: &str = "EOU_VOCAB_PATH";
    pub const EOU_MERGES_PATH: &str = "EOU_MERGES_PATH";
    pub const EOU_LANGUAGES_PATH: &str = "EOU_LANGUAGES_PATH";
    pub const EOU_MAX_CONTEXT_TOKENS: &str = "EOU_MAX_CONTEXT_TOKENS";

    pub const MIN_SPEECH_FOR_RESPONSE_MS: &str = "MIN_SPEECH_FOR_RESPONSE_MS";

    pub const DIAR_THRESHOLD: &str = "DIAR_THRESHOLD";
    pub const DIAR_MAX_SPEAKERS: &str = "DIAR_MAX_SPEAKERS";
    pub const DIAR_MIN_SPAN_FRAMES: &str = "DIAR_MIN_SPAN_FRAMES";
    pub const DIAR_MEDIAN_FILTER_FRAMES: &str = "DIAR_MEDIAN_FILTER_FRAMES";
    pub const MIN_SPEECH_FOR_COMMIT_MS: &str = "MIN_SPEECH_FOR_COMMIT_MS";

    pub const INSPECT_SESSION_DIR: &str = "INSPECT_SESSION_DIR";
    pub const INSPECT_RETENTION_COUNT: &str = "INSPECT_RETENTION_COUNT";
    pub const INSPECT_RETENTION_BYTES: &str = "INSPECT_RETENTION_BYTES";
    pub const INSPECT_RETENTION_DAYS: &str = "INSPECT_RETENTION_DAYS";

    pub const OTEL_EXPORTER_OTLP_ENDPOINT: &str = "OTEL_EXPORTER_OTLP_ENDPOINT";
    pub const OTEL_EXPORTER_OTLP_PROTOCOL: &str = "OTEL_EXPORTER_OTLP_PROTOCOL";
    pub const OTEL_SERVICE_NAME: &str = "OTEL_SERVICE_NAME";
}

pub mod tracing {
    pub const TRACER_NAME: &str = "speaches/realtime";
    pub const SERVICE_NAME_DEFAULT: &str = "speaches-plus";
}

pub mod diarization {
    pub const REALTIME_ENV: &str = "REALTIME_DIARIZATION";
    pub const REALTIME_ENABLED: bool = false;
}

pub mod serve_backend {
    pub const DEFAULT: &str = "cuda";
    pub const ALLOWED: &[&str] = &["cuda", "wgpu", "auto"];
}
