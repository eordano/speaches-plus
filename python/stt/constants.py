from __future__ import annotations

WHISPER_NFFT = 400
WHISPER_HOP_LENGTH = 160
WHISPER_CHUNK_SECS = 30
WHISPER_SAMPLING_HZ = 16_000
WHISPER_NB_FRAMES = 3_000
WHISPER_PAD_SAMPLES = 480_000

SILENCE_PEAK_THRESHOLD = 0.005

DEFAULT_N_MELS = 80
LARGE_V3_N_MELS = 128

WHISPER_TIMESTAMP_STEP_MS = 20
WHISPER_TIMESTAMP_TOKEN_COUNT = 1501

LANGUAGE_CODES = (
    "en", "zh", "de", "es", "ru", "ko", "fr", "ja", "pt", "tr",
    "pl", "ca", "nl", "ar", "sv", "it", "id", "hi", "fi", "vi",
    "he", "uk", "el", "ms", "cs", "ro", "da", "hu", "ta", "no",
    "th", "ur", "hr", "bg", "lt", "la", "mi", "ml", "cy", "sk",
    "te", "fa", "lv", "bn", "sr", "az", "sl", "kn", "et", "mk",
    "br", "eu", "is", "hy", "ne", "mn", "bs", "kk", "sq", "sw",
    "gl", "mr", "pa", "si", "km", "sn", "yo", "so", "af", "oc",
    "ka", "be", "tg", "sd", "gu", "am", "yi", "lo", "uz", "fo",
    "ht", "ps", "tk", "nn", "mt", "sa", "lb", "my", "bo", "tl",
    "mg", "as", "tt", "haw", "ln", "ha", "ba", "jw", "su", "yue",
)
