use crate::AlignedSegment;
use serde_json::{json, Value};

fn fmt_ts(seconds: f32, ms_sep: char) -> String {
    let clamped = if seconds.is_finite() && seconds > 0.0 {
        seconds
    } else {
        0.0
    };

    let total_ms = (clamped * 1000.0).round() as u64;
    let ms = total_ms % 1000;
    let total_s = total_ms / 1000;
    let s = total_s % 60;
    let total_m = total_s / 60;
    let m = total_m % 60;
    let h = total_m / 60;
    format!("{:02}:{:02}:{:02}{}{:03}", h, m, s, ms_sep, ms)
}

pub fn to_srt(segments: &[AlignedSegment]) -> String {
    let mut out = String::new();
    for (i, seg) in segments.iter().enumerate() {
        out.push_str(&(i + 1).to_string());
        out.push('\n');
        out.push_str(&fmt_ts(seg.start, ','));
        out.push_str(" --> ");
        out.push_str(&fmt_ts(seg.end, ','));
        out.push('\n');
        out.push_str(&seg.text);
        out.push('\n');
        out.push('\n');
    }
    out
}

pub fn to_vtt(segments: &[AlignedSegment]) -> String {
    let mut out = String::from("WEBVTT\n\n");
    for seg in segments {
        out.push_str(&fmt_ts(seg.start, '.'));
        out.push_str(" --> ");
        out.push_str(&fmt_ts(seg.end, '.'));
        out.push('\n');
        out.push_str(&seg.text);
        out.push('\n');
        out.push('\n');
    }
    out
}

pub fn to_diarized_json(segments: &[AlignedSegment]) -> Value {
    let mut arr: Vec<Value> = Vec::with_capacity(segments.len());
    for seg in segments {
        let words: Vec<Value> = seg
            .words
            .iter()
            .map(|w| json!({"word": w.word, "start": w.start, "end": w.end}))
            .collect();
        let mut obj = json!({
            "start": seg.start,
            "end": seg.end,
            "text": seg.text,
            "words": words,
        });
        if let Some(spk) = seg.speaker.as_ref() {
            obj.as_object_mut()
                .expect("just constructed as object")
                .insert("speaker".to_string(), Value::String(spk.clone()));
        }
        arr.push(obj);
    }
    Value::Array(arr)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{AlignedSegment, WordTiming};

    fn seg(
        start: f32,
        end: f32,
        text: &str,
        words: Vec<WordTiming>,
        speaker: Option<&str>,
    ) -> AlignedSegment {
        AlignedSegment {
            start,
            end,
            text: text.to_string(),
            words,
            speaker: speaker.map(|s| s.to_string()),
        }
    }

    fn w(word: &str, start: f32, end: f32) -> WordTiming {
        WordTiming {
            word: word.to_string(),
            start,
            end,
        }
    }

    #[test]
    fn srt_one_segment_one_word() {
        let segs = vec![seg(0.0, 1.234, "Hello", vec![w("Hello", 0.0, 1.234)], None)];
        let want = "1\n00:00:00,000 --> 00:00:01,234\nHello\n\n";
        assert_eq!(to_srt(&segs), want);
    }

    #[test]
    fn srt_multi_segment() {
        let segs = vec![
            seg(
                0.0,
                3.5,
                "Hello world",
                vec![w("Hello", 0.0, 1.0), w("world", 1.0, 3.5)],
                None,
            ),
            seg(
                3.5,
                5.2,
                "How are you",
                vec![w("How", 3.5, 4.0), w("are", 4.0, 4.5), w("you", 4.5, 5.2)],
                None,
            ),
        ];
        let want = "1\n00:00:00,000 --> 00:00:03,500\nHello world\n\n2\n00:00:03,500 --> 00:00:05,200\nHow are you\n\n";
        assert_eq!(to_srt(&segs), want);
    }

    #[test]
    fn vtt_header_and_segments() {
        let segs = vec![seg(
            0.0,
            3.5,
            "Hello world",
            vec![w("Hello", 0.0, 1.0), w("world", 1.0, 3.5)],
            None,
        )];
        let want = "WEBVTT\n\n00:00:00.000 --> 00:00:03.500\nHello world\n\n";
        assert_eq!(to_vtt(&segs), want);
    }

    #[test]
    fn vtt_multi_segment_uses_dots() {
        let segs = vec![
            seg(0.0, 1.5, "a", vec![w("a", 0.0, 1.5)], None),
            seg(1.5, 3.0, "b", vec![w("b", 1.5, 3.0)], None),
        ];
        let want =
            "WEBVTT\n\n00:00:00.000 --> 00:00:01.500\na\n\n00:00:01.500 --> 00:00:03.000\nb\n\n";
        assert_eq!(to_vtt(&segs), want);
    }

    #[test]
    fn diarized_json_omits_speaker_when_none() {
        let segs = vec![seg(0.0, 1.0, "hi", vec![w("hi", 0.0, 1.0)], None)];
        let v = to_diarized_json(&segs);
        let want = json!([{
            "start": 0.0,
            "end": 1.0,
            "text": "hi",
            "words": [{"word": "hi", "start": 0.0, "end": 1.0}],
        }]);
        assert_eq!(v, want);
    }

    #[test]
    fn diarized_json_includes_speaker_when_some() {
        let segs = vec![
            seg(0.0, 1.0, "hi", vec![w("hi", 0.0, 1.0)], Some("S1")),
            seg(1.0, 2.0, "yo", vec![w("yo", 1.0, 2.0)], Some("S2")),
        ];
        let v = to_diarized_json(&segs);
        let want = json!([
            {
                "start": 0.0,
                "end": 1.0,
                "text": "hi",
                "words": [{"word": "hi", "start": 0.0, "end": 1.0}],
                "speaker": "S1",
            },
            {
                "start": 1.0,
                "end": 2.0,
                "text": "yo",
                "words": [{"word": "yo", "start": 1.0, "end": 2.0}],
                "speaker": "S2",
            },
        ]);
        assert_eq!(v, want);
    }

    #[test]
    fn fmt_ts_handles_large_times() {
        assert_eq!(fmt_ts(3723.456, ','), "01:02:03,456");
        assert_eq!(fmt_ts(3723.456, '.'), "01:02:03.456");
    }

    #[test]
    fn fmt_ts_handles_negatives_and_nans() {
        assert_eq!(fmt_ts(-1.0, ','), "00:00:00,000");
        assert_eq!(fmt_ts(f32::NAN, ','), "00:00:00,000");
    }
}
