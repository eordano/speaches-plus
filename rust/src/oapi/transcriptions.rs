use axum::http::{header, StatusCode};
use axum::response::{IntoResponse, Response};
use nv_aligner::{to_srt, to_vtt, AlignedSegment, WordTiming};

use crate::stt::{TimedSegment, TranscriptionResult};

pub fn timed_segments_to_aligned(segs: &[TimedSegment]) -> Vec<AlignedSegment> {
    segs.iter()
        .map(|s| {
            let words: Vec<WordTiming> = s
                .words
                .iter()
                .map(|w| WordTiming {
                    word: w.word.clone(),
                    start: w.start_ms as f32 / 1000.0,
                    end: w.end_ms as f32 / 1000.0,
                })
                .collect();
            AlignedSegment {
                text: s.text.clone(),
                start: s.t_start_ms as f32 / 1000.0,
                end: s.t_end_ms as f32 / 1000.0,
                words,
                speaker: None,
            }
        })
        .collect()
}

pub fn srt_response(stt: &TranscriptionResult) -> Response {
    let aligned = timed_segments_to_aligned(&stt.segments);
    let body = to_srt(&aligned);
    (
        StatusCode::OK,
        [(header::CONTENT_TYPE, "text/plain; charset=utf-8")],
        body,
    )
        .into_response()
}

pub fn vtt_response(stt: &TranscriptionResult) -> Response {
    let aligned = timed_segments_to_aligned(&stt.segments);
    let body = to_vtt(&aligned);
    (
        StatusCode::OK,
        [(header::CONTENT_TYPE, "text/plain; charset=utf-8")],
        body,
    )
        .into_response()
}

pub fn verbose_json_response(stt: &TranscriptionResult) -> Response {
    let aligned = timed_segments_to_aligned(&stt.segments);
    let segments_json: Vec<serde_json::Value> = aligned
        .iter()
        .zip(stt.segments.iter())
        .enumerate()
        .map(|(i, (s, raw))| {
            let words: Vec<serde_json::Value> = s
                .words
                .iter()
                .map(|w| {
                    serde_json::json!({
                        "word": w.word,
                        "start": w.start,
                        "end": w.end,
                    })
                })
                .collect();
            serde_json::json!({
                "id": i,
                "start": s.start,
                "end": s.end,
                "text": s.text,
                "words": words,
                "avg_logprob": raw.avg_logprob,
                "no_speech_prob": raw.no_speech_prob,
            })
        })
        .collect();
    let all_words: Vec<serde_json::Value> = aligned
        .iter()
        .flat_map(|s| {
            s.words.iter().map(|w| {
                serde_json::json!({
                    "word": w.word,
                    "start": w.start,
                    "end": w.end,
                })
            })
        })
        .collect();
    let body = serde_json::json!({
        "task": stt.task.as_str(),
        "language": stt.language,
        "duration": stt.duration_s,
        "text": stt.text,
        "segments": segments_json,
        "words": all_words,
    });
    (
        StatusCode::OK,
        [(header::CONTENT_TYPE, "application/json")],
        body.to_string(),
    )
        .into_response()
}

#[cfg(feature = "ts-bindings")]
mod ts_wire {
    #![allow(dead_code)]
    use ts_rs::TS;

    #[derive(TS)]
    #[ts(export)]
    struct TranscriptionJsonResponse {
        text: String,
    }

    #[derive(TS)]
    #[ts(export)]
    struct TranscriptionWord {
        word: String,
        start: f32,
        end: f32,
    }

    #[derive(TS)]
    #[ts(export)]
    struct TranscriptionVerboseSegment {
        id: usize,
        start: f32,
        end: f32,
        text: String,
        words: Vec<TranscriptionWord>,
        avg_logprob: Option<f32>,
        no_speech_prob: Option<f32>,
    }

    #[derive(TS)]
    #[ts(export)]
    struct TranscriptionVerboseJsonResponse {
        #[ts(type = "\"transcribe\" | \"translate\"")]
        task: (),
        language: Option<String>,
        duration: Option<f32>,
        text: String,
        segments: Vec<TranscriptionVerboseSegment>,
        words: Vec<TranscriptionWord>,
    }

    #[derive(TS)]
    #[ts(export)]
    struct TranscriptionDiarizedSegment {
        #[ts(rename = "type", type = "\"transcript.text.segment\"")]
        kind: (),
        id: String,
        speaker: Option<String>,
        start: f64,
        end: f64,
        duration: f64,
        text: String,
        avg_logprob: Option<f32>,
        no_speech_prob: Option<f32>,
        confidence: Option<f32>,
    }

    #[derive(TS)]
    #[ts(export)]
    struct TranscriptionDiarizedJsonResponse {
        text: String,
        avg_logprob: Option<f32>,
        no_speech_prob: Option<f32>,
        segments: Vec<TranscriptionDiarizedSegment>,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::stt::TimedSegment;
    use axum::body::to_bytes;

    fn ts(start: u32, end: u32, text: &str) -> TimedSegment {
        TimedSegment {
            t_start_ms: start,
            t_end_ms: end,
            text: text.into(),
            ..Default::default()
        }
    }

    #[test]
    fn timed_to_aligned_preserves_text_and_converts_ms_to_seconds() {
        let segs = vec![ts(0, 1234, "hello"), ts(1500, 3500, "world")];
        let aligned = timed_segments_to_aligned(&segs);
        assert_eq!(aligned.len(), 2);
        assert_eq!(aligned[0].text, "hello");
        assert!((aligned[0].start - 0.0).abs() < 1e-6);
        assert!((aligned[0].end - 1.234).abs() < 1e-3);
        assert!(aligned[0].words.is_empty());
        assert!(aligned[0].speaker.is_none());
        assert!((aligned[1].start - 1.5).abs() < 1e-6);
        assert!((aligned[1].end - 3.5).abs() < 1e-3);
    }

    #[tokio::test]
    async fn srt_response_starts_with_index_and_timestamp() {
        let stt = TranscriptionResult {
            text: "hello world".into(),
            segments: vec![ts(0, 1234, "hello"), ts(1500, 3500, "world")],
            ..Default::default()
        };
        let resp = srt_response(&stt);
        assert_eq!(resp.status(), StatusCode::OK);
        let ct = resp
            .headers()
            .get(header::CONTENT_TYPE)
            .unwrap()
            .to_str()
            .unwrap()
            .to_string();
        assert!(ct.starts_with("text/plain"), "{ct}");
        let bytes = to_bytes(resp.into_body(), 1 << 20).await.unwrap();
        let body = std::str::from_utf8(&bytes).unwrap();

        assert!(body.starts_with("1\n00:00:00,000 --> "), "{body:?}");
        assert!(body.contains("hello"), "{body:?}");

        assert!(body.contains("\n2\n"), "{body:?}");
        assert!(body.contains("world"), "{body:?}");
    }

    #[tokio::test]
    async fn vtt_response_has_webvtt_header_and_dot_separator() {
        let stt = TranscriptionResult {
            text: "hi".into(),
            segments: vec![ts(0, 500, "hi")],
            ..Default::default()
        };
        let resp = vtt_response(&stt);
        assert_eq!(resp.status(), StatusCode::OK);
        let bytes = to_bytes(resp.into_body(), 1 << 20).await.unwrap();
        let body = std::str::from_utf8(&bytes).unwrap();
        assert!(body.starts_with("WEBVTT\n\n"), "{body:?}");
        assert!(body.contains("00:00:00.000 --> 00:00:00.500"), "{body:?}");
        assert!(body.contains("hi"), "{body:?}");
    }

    #[tokio::test]
    async fn verbose_json_response_has_text_and_segments() {
        let stt = TranscriptionResult {
            text: "hello world".into(),
            segments: vec![ts(0, 1000, "hello"), ts(1000, 2000, "world")],
            ..Default::default()
        };
        let resp = verbose_json_response(&stt);
        assert_eq!(resp.status(), StatusCode::OK);
        let bytes = to_bytes(resp.into_body(), 1 << 20).await.unwrap();
        let v: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(v["text"], "hello world");
        let segs = v["segments"].as_array().unwrap();
        assert_eq!(segs.len(), 2);
        assert_eq!(segs[0]["id"], 0);
        assert_eq!(segs[0]["text"], "hello");
        assert_eq!(segs[1]["id"], 1);
        assert_eq!(segs[1]["text"], "world");

        assert!(v["words"].is_array());
    }

    #[tokio::test]
    async fn verbose_json_response_emits_task_language_and_duration() {
        let stt = TranscriptionResult {
            text: "hola".into(),
            segments: vec![ts(0, 1000, "hola")],
            language: Some("es".into()),
            duration_s: Some(8.42),
            task: crate::stt::WhisperTask::Translate,
            ..Default::default()
        };
        let resp = verbose_json_response(&stt);
        let bytes = to_bytes(resp.into_body(), 1 << 20).await.unwrap();
        let v: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(v["task"], "translate");
        assert_eq!(v["language"], "es");
        assert!((v["duration"].as_f64().unwrap() - 8.42).abs() < 1e-3, "{v}");
    }

    #[tokio::test]
    async fn verbose_json_response_defaults_task_to_transcribe() {
        let stt = TranscriptionResult {
            text: "hi".into(),
            segments: vec![ts(0, 500, "hi")],
            ..Default::default()
        };
        let resp = verbose_json_response(&stt);
        let bytes = to_bytes(resp.into_body(), 1 << 20).await.unwrap();
        let v: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(v["task"], "transcribe");
        assert!(v["language"].is_null(), "{v}");
        assert!(v["duration"].is_null(), "{v}");
    }

    #[tokio::test]
    async fn verbose_json_segments_carry_confidence_fields() {
        let mut seg = ts(0, 1000, "hello");
        seg.avg_logprob = Some(-0.25);
        seg.no_speech_prob = Some(0.01);
        let stt = TranscriptionResult {
            text: "hello".into(),
            segments: vec![seg],
            ..Default::default()
        };
        let resp = verbose_json_response(&stt);
        let bytes = to_bytes(resp.into_body(), 1 << 20).await.unwrap();
        let v: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        let s = &v["segments"][0];
        assert!(
            (s["avg_logprob"].as_f64().unwrap() + 0.25).abs() < 1e-6,
            "{v}"
        );
        assert!(
            (s["no_speech_prob"].as_f64().unwrap() - 0.01).abs() < 1e-6,
            "{v}"
        );
    }

    #[tokio::test]
    async fn verbose_json_words_carry_no_pseudo_tokens() {
        let mut seg = ts(0, 1000, "hello world");
        seg.words = vec![
            crate::stt::TimedWord {
                word: "hello".into(),
                start_ms: 0,
                end_ms: 500,
            },
            crate::stt::TimedWord {
                word: "world".into(),
                start_ms: 500,
                end_ms: 1000,
            },
        ];
        let stt = TranscriptionResult {
            text: "hello world".into(),
            segments: vec![seg],
            ..Default::default()
        };
        let resp = verbose_json_response(&stt);
        let bytes = to_bytes(resp.into_body(), 1 << 20).await.unwrap();
        let v: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        let words = v["words"].as_array().unwrap();
        assert_eq!(words.len(), 2);
        assert!(
            !words
                .iter()
                .any(|w| w["word"].as_str().unwrap_or_default().starts_with('[')),
            "{v}"
        );
    }

    #[tokio::test]
    async fn srt_response_empty_segments_yields_empty_body() {
        let stt = TranscriptionResult::default();
        let resp = srt_response(&stt);
        let bytes = to_bytes(resp.into_body(), 1 << 20).await.unwrap();
        assert!(bytes.is_empty());
    }
}
