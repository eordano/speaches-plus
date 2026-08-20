#![allow(dead_code)]

pub const ERR_KINDS: &[&str] = &[
    "error",
    "raised",
    "dropped",
    "failed",
    "phrase_error",
    "bargein_missed",
];

pub const LANES: &[&str] = &[
    "audio_level",
    "vad",
    "stt",
    "turn",
    "bargein",
    "eou",
    "diarization",
    "llm",
    "response",
    "tool",
    "tts_req",
    "tts_chunk",
    "tts_pacer",
    "wire",
    "state",
    "error",
];

pub fn is_error_kind(kind: &str) -> bool {
    ERR_KINDS.contains(&kind)
}

pub fn is_known_lane(lane: &str) -> bool {
    LANES.contains(&lane)
}
