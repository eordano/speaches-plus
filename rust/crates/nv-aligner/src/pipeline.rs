use crate::dp::{frame_to_time_ms, viterbi_align, AlignError};
use crate::{AlignedSegment, WordTiming};

#[allow(clippy::too_many_arguments)]
pub fn align_with_logprobs(
    log_probs: &[f32],
    t: usize,
    v: usize,
    token_ids: &[u32],
    blank_id: u32,
    vocab: &[String],
    hop_samples: usize,
    sample_rate: usize,
) -> Result<Vec<AlignedSegment>, AlignError> {
    if vocab.len() != v {
        return Err(AlignError::ShapeMismatch {
            got: vocab.len(),
            expected: v,
        });
    }

    let spans = viterbi_align(log_probs, t, v, token_ids, blank_id)?;

    let mut words: Vec<WordTiming> = Vec::with_capacity(token_ids.len());
    for (i, &id) in token_ids.iter().enumerate() {
        let (s_frame, e_frame) = spans[i];
        let start = frame_to_time_ms(s_frame, hop_samples as u32, sample_rate as u32) / 1000.0;

        let end = frame_to_time_ms(e_frame, hop_samples as u32, sample_rate as u32) / 1000.0;
        let word = vocab.get(id as usize).cloned().unwrap_or_default();
        words.push(WordTiming { word, start, end });
    }

    let mut segments: Vec<AlignedSegment> = Vec::new();
    let mut current: Vec<WordTiming> = Vec::new();
    let is_sentence_end = |w: &str| w.contains('.') || w.contains('?') || w.contains('!');

    for wt in words.into_iter() {
        let breaks = is_sentence_end(&wt.word);
        current.push(wt);
        if breaks {
            segments.push(finalize_segment(std::mem::take(&mut current)));
        }
    }
    if !current.is_empty() {
        segments.push(finalize_segment(current));
    }

    Ok(segments)
}

fn finalize_segment(words: Vec<WordTiming>) -> AlignedSegment {
    debug_assert!(!words.is_empty());
    let start = words.first().map(|w| w.start).unwrap_or(0.0);
    let end = words.last().map(|w| w.end).unwrap_or(0.0);
    let text = words
        .iter()
        .map(|w| w.word.as_str())
        .collect::<Vec<_>>()
        .join(" ")
        .trim()
        .to_string();
    AlignedSegment {
        start,
        end,
        text,
        words,
        speaker: None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_logprobs(schedule: &[u32], v: usize) -> Vec<f32> {
        let eps: f32 = 1e-3;
        let other = (eps / (v as f32 - 1.0)).ln();
        let target = (1.0_f32 - eps).ln();
        let mut out = vec![other; schedule.len() * v];
        for (ti, &cls) in schedule.iter().enumerate() {
            out[ti * v + (cls as usize)] = target;
        }
        out
    }

    fn approx(a: f32, b: f32, tol: f32) -> bool {
        (a - b).abs() <= tol
    }

    const HOP: usize = 160;
    const SR: usize = 16_000;
    const FRAME_S: f32 = 0.01;

    #[test]
    fn pipeline_single_segment_no_punct() {
        let v = 4;
        let vocab: Vec<String> = vec!["<blk>".into(), "hello".into(), "world".into(), "_".into()];
        let lp = make_logprobs(&[1, 0, 2, 2, 0], v);
        let segs = align_with_logprobs(&lp, 5, v, &[1, 2], 0, &vocab, HOP, SR).unwrap();

        assert_eq!(segs.len(), 1);
        let s = &segs[0];
        assert_eq!(s.text, "hello world");
        assert!(s.speaker.is_none());

        let total = 5.0 * FRAME_S;
        for w in &s.words {
            assert!(w.start >= 0.0 && w.end <= total + 1e-6, "{w:?}");
            assert!(w.end > w.start, "{w:?}");
        }

        assert!(approx(s.start, s.words[0].start, 1e-6));
        assert!(approx(s.end, s.words[s.words.len() - 1].end, 1e-6));
    }

    #[test]
    fn pipeline_two_segments_period_break() {
        let v = 5;
        let vocab: Vec<String> = vec![
            "<blk>".into(),
            "hi".into(),
            ".".into(),
            "ok".into(),
            "!".into(),
        ];
        let lp = make_logprobs(&[1, 2, 0, 0, 0, 0, 3, 0, 4], v);
        let segs = align_with_logprobs(&lp, 9, v, &[1, 2, 3, 4], 0, &vocab, HOP, SR).unwrap();

        assert_eq!(segs.len(), 2);
        assert_eq!(segs[0].text, "hi .");
        assert_eq!(segs[1].text, "ok !");

        assert_eq!(segs[0].words.len(), 2);
        assert_eq!(segs[1].words.len(), 2);

        assert!(segs[0].end <= segs[1].start + 1e-6);

        assert!(segs[0].speaker.is_none());
        assert!(segs[1].speaker.is_none());
    }

    #[test]
    fn pipeline_held_token_has_nonzero_duration() {
        let v = 3;
        let vocab: Vec<String> = vec!["<blk>".into(), "aaa".into(), "_".into()];
        let lp = make_logprobs(&[1, 1, 1], v);
        let segs = align_with_logprobs(&lp, 3, v, &[1], 0, &vocab, HOP, SR).unwrap();
        assert_eq!(segs.len(), 1);
        assert_eq!(segs[0].words.len(), 1);
        let w = &segs[0].words[0];

        assert!(w.start >= 0.0);
        assert!(w.end > w.start);
        assert!(w.end <= 3.0 * FRAME_S + 1e-6);
    }

    #[test]
    fn pipeline_vocab_size_mismatch_errors() {
        let v = 3;
        let vocab: Vec<String> = vec!["<blk>".into(), "x".into()];
        let lp = make_logprobs(&[1], v);
        let err = align_with_logprobs(&lp, 1, v, &[1], 0, &vocab, HOP, SR).unwrap_err();
        assert!(matches!(err, AlignError::ShapeMismatch { .. }));
    }

    #[test]
    fn pipeline_propagates_dp_errors() {
        let v = 3;
        let vocab: Vec<String> = vec!["<blk>".into(), "a".into(), "_".into()];
        let lp = make_logprobs(&[1, 1], v);
        let err = align_with_logprobs(&lp, 2, v, &[1, 1], 0, &vocab, HOP, SR).unwrap_err();
        assert!(matches!(err, AlignError::TooFewFrames { .. }));
    }
}
