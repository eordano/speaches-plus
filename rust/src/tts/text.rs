use std::sync::OnceLock;

use regex::Regex;

pub const KOKORO_SAMPLE_RATE: u32 = 24_000;
pub const MIN_SAMPLE_RATE: u32 = 8_000;
pub const MAX_SAMPLE_RATE: u32 = 48_000;
pub const SPEED_MIN: f32 = 0.5;
pub const SPEED_MAX: f32 = 2.0;
pub const MAX_CHUNK_CHARS: usize = 400;
pub const DEFAULT_VOICE: &str = "af_heart";
pub const DEFAULT_LANGUAGE: &str = "en-us";

#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Deserialize, serde::Serialize)]
#[serde(rename_all = "lowercase")]
#[derive(Default)]
pub enum ResponseFormat {
    Pcm,
    #[default]
    Mp3,
    Wav,
    Flac,
    Opus,
    Aac,
}

impl ResponseFormat {
    pub fn mime_type(self) -> &'static str {
        match self {
            ResponseFormat::Pcm => "audio/pcm",
            ResponseFormat::Mp3 => "audio/mpeg",
            ResponseFormat::Wav => "audio/wav",
            ResponseFormat::Flac => "audio/flac",
            ResponseFormat::Opus => "audio/opus",
            ResponseFormat::Aac => "audio/aac",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Deserialize, serde::Serialize)]
#[serde(rename_all = "lowercase")]
#[derive(Default)]
pub enum StreamFormat {
    #[default]
    Audio,
    Sse,
}

pub fn is_openai_voice_alias(name: &str) -> bool {
    matches!(
        name.to_ascii_lowercase().as_str(),
        "alloy" | "ash" | "ballad" | "coral" | "echo" | "sage" | "shimmer" | "verse"
    )
}

fn emoji_re() -> &'static Regex {
    static R: OnceLock<Regex> = OnceLock::new();
    R.get_or_init(|| {
        Regex::new(concat!(
            "[",
            "\u{1F600}-\u{1F64F}",
            "\u{1F300}-\u{1F5FF}",
            "\u{1F680}-\u{1F6FF}",
            "\u{1F700}-\u{1F77F}",
            "\u{1F780}-\u{1F7FF}",
            "\u{1F800}-\u{1F8FF}",
            "\u{1F900}-\u{1F9FF}",
            "\u{1FA00}-\u{1FA6F}",
            "\u{1FA70}-\u{1FAFF}",
            "\u{2702}-\u{27B0}",
            "]+",
        ))
        .expect("emoji regex compiles")
    })
}

fn md_re_bold() -> &'static Regex {
    static R: OnceLock<Regex> = OnceLock::new();
    R.get_or_init(|| Regex::new(r"\*\*(.*?)\*\*").unwrap())
}
fn md_re_italic_star() -> &'static Regex {
    static R: OnceLock<Regex> = OnceLock::new();
    R.get_or_init(|| Regex::new(r"\*(.*?)\*").unwrap())
}
fn md_re_under() -> &'static Regex {
    static R: OnceLock<Regex> = OnceLock::new();
    R.get_or_init(|| Regex::new(r"__(.*?)__").unwrap())
}
fn md_re_italic_under() -> &'static Regex {
    static R: OnceLock<Regex> = OnceLock::new();
    R.get_or_init(|| Regex::new(r"_(.*?)_").unwrap())
}
fn whitespace_re() -> &'static Regex {
    static R: OnceLock<Regex> = OnceLock::new();
    R.get_or_init(|| Regex::new(r"\s+").unwrap())
}
fn newline_re() -> &'static Regex {
    static R: OnceLock<Regex> = OnceLock::new();
    R.get_or_init(|| Regex::new(r"[\r\n]+").unwrap())
}

pub fn strip_emojis(s: &str) -> String {
    emoji_re().replace_all(s, "").into_owned()
}

pub fn strip_markdown_emphasis(s: &str) -> String {
    let mut out = md_re_bold().replace_all(s, "$1").into_owned();
    out = md_re_italic_star().replace_all(&out, "$1").into_owned();
    out = md_re_under().replace_all(&out, "$1").into_owned();
    out = md_re_italic_under().replace_all(&out, "$1").into_owned();
    out
}

pub fn normalize_for_tts(s: &str) -> String {
    let collapsed_nl = newline_re().replace_all(s, " ");
    whitespace_re()
        .replace_all(&collapsed_nl, " ")
        .trim()
        .to_string()
}

pub fn split_into_chunks(text: &str, max_chars: usize) -> Vec<String> {
    let max_chars = if max_chars == 0 {
        MAX_CHUNK_CHARS
    } else {
        max_chars
    };
    if text.is_empty() {
        return vec![];
    }
    if text.chars().count() <= max_chars {
        return vec![text.to_string()];
    }

    let sentences = split_sentences(text);
    let mut chunks: Vec<String> = Vec::new();
    let mut current = String::new();
    let flush = |current: &mut String, chunks: &mut Vec<String>| {
        let trimmed = current.trim();
        if !trimmed.is_empty() {
            chunks.push(trimmed.to_string());
        }
        current.clear();
    };

    for sentence in sentences {
        if sentence.chars().count() > max_chars {
            flush(&mut current, &mut chunks);
            for word in sentence.split_whitespace() {
                let projected = current.chars().count()
                    + word.chars().count()
                    + if current.is_empty() { 0 } else { 1 };
                if projected <= max_chars {
                    if !current.is_empty() {
                        current.push(' ');
                    }
                    current.push_str(word);
                } else {
                    flush(&mut current, &mut chunks);
                    current.push_str(word);
                }
            }
        } else {
            let projected = current.chars().count()
                + sentence.chars().count()
                + if current.is_empty() { 0 } else { 1 };
            if projected <= max_chars {
                if !current.is_empty() {
                    current.push(' ');
                }
                current.push_str(&sentence);
            } else {
                flush(&mut current, &mut chunks);
                current.push_str(&sentence);
            }
        }
    }
    flush(&mut current, &mut chunks);
    chunks
}

fn split_sentences(text: &str) -> Vec<String> {
    let chars: Vec<char> = text.chars().collect();
    let mut out: Vec<String> = Vec::new();
    let mut start: usize = 0;
    let mut i: usize = 0;
    while i < chars.len() {
        let c = chars[i];
        if (c == '.' || c == '!' || c == '?') && i + 1 < chars.len() && chars[i + 1].is_whitespace()
        {
            let end = i + 1;
            let mut k = end;
            while k < chars.len() && chars[k].is_whitespace() {
                k += 1;
            }
            out.push(chars[start..end].iter().collect());
            start = k;
            i = k;
            continue;
        }
        i += 1;
    }
    if start < chars.len() {
        out.push(chars[start..].iter().collect());
    }
    out
}

pub fn f32_to_s16le(samples: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(samples.len() * 2);
    for &s in samples {
        let v = (s * 32767.0).round();
        let v = v.clamp(-32768.0, 32767.0) as i16;
        out.extend_from_slice(&v.to_le_bytes());
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn strip_emojis_removes_supported_ranges() {
        assert_eq!(strip_emojis("hello 🌍 world"), "hello  world");
        assert_eq!(strip_emojis("😀😃😄 plain"), " plain");
        assert_eq!(strip_emojis("plain"), "plain");
        assert_eq!(strip_emojis("✂ scissors"), " scissors");
    }

    #[test]
    fn strip_markdown_handles_all_emphasis_styles() {
        assert_eq!(strip_markdown_emphasis("**bold**"), "bold");
        assert_eq!(strip_markdown_emphasis("*italic*"), "italic");
        assert_eq!(strip_markdown_emphasis("__under__"), "under");
        assert_eq!(strip_markdown_emphasis("_under_"), "under");
        assert_eq!(
            strip_markdown_emphasis("a **bold** and *italic* mix"),
            "a bold and italic mix"
        );
    }

    #[test]
    fn normalize_collapses_whitespace_and_newlines() {
        let out = normalize_for_tts("  hello\n\nworld\t\ttest\r\n");
        assert_eq!(out, "hello world test");
    }

    #[test]
    fn split_returns_single_chunk_for_short_text() {
        assert_eq!(split_into_chunks("short", 100), vec!["short".to_string()]);
        assert!(split_into_chunks("", 100).is_empty());
    }

    #[test]
    fn split_breaks_on_sentence_boundaries() {
        let text = "First sentence. Second sentence. Third sentence.";
        let chunks = split_into_chunks(text, 25);
        for c in &chunks {
            assert!(c.chars().count() <= 25, "chunk too long: {c:?}");
        }
        assert!(!chunks.is_empty());
    }

    #[test]
    fn f32_to_s16_clamps_extreme_values() {
        let samples = [0.0f32, 1.0, -1.0, 2.0, -2.0];
        let bytes = f32_to_s16le(&samples);
        assert_eq!(bytes.len(), samples.len() * 2);
        assert_eq!(&bytes[2..4], &[0xFF, 0x7F]);
        assert_eq!(&bytes[4..6], &[0x01, 0x80]);
        assert_eq!(&bytes[6..8], &[0xFF, 0x7F]);
        assert_eq!(&bytes[8..10], &[0x00, 0x80]);
    }
}
