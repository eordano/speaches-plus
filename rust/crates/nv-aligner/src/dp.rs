use std::fmt;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AlignError {
    ShapeMismatch {
        expected: usize,
        got: usize,
    },

    TooFewFrames {
        frames: usize,
        required: usize,
    },

    InvalidTargetToken {
        index: usize,
        token: u32,
        vocab: usize,
    },

    InvalidBlankId {
        blank: u32,
        vocab: usize,
    },
}

impl fmt::Display for AlignError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            AlignError::ShapeMismatch { expected, got } => {
                write!(f, "log_probs length mismatch: expected T*V = {expected}, got {got}")
            }
            AlignError::TooFewFrames { frames, required } => write!(
                f,
                "too few frames for forced alignment: have {frames}, need at least {required} (2*L+1)"
            ),
            AlignError::InvalidTargetToken { index, token, vocab } => write!(
                f,
                "target[{index}] = {token} is out of range for vocab size {vocab}"
            ),
            AlignError::InvalidBlankId { blank, vocab } => {
                write!(f, "blank_id {blank} is out of range for vocab size {vocab}")
            }
        }
    }
}

impl std::error::Error for AlignError {}

#[inline]
pub fn frame_to_time_ms(frame: usize, hop_length: u32, sample_rate: u32) -> f32 {
    (frame as f32) * (hop_length as f32) * 1000.0 / (sample_rate as f32)
}

pub fn viterbi_align(
    log_probs: &[f32],
    t: usize,
    v: usize,
    targets: &[u32],
    blank_id: u32,
) -> Result<Vec<(usize, usize)>, AlignError> {
    if log_probs.len() != t.saturating_mul(v) {
        return Err(AlignError::ShapeMismatch {
            expected: t.saturating_mul(v),
            got: log_probs.len(),
        });
    }
    if v == 0 {
        return Err(AlignError::InvalidBlankId {
            blank: blank_id,
            vocab: v,
        });
    }
    if (blank_id as usize) >= v {
        return Err(AlignError::InvalidBlankId {
            blank: blank_id,
            vocab: v,
        });
    }
    for (i, &tok) in targets.iter().enumerate() {
        if (tok as usize) >= v {
            return Err(AlignError::InvalidTargetToken {
                index: i,
                token: tok,
                vocab: v,
            });
        }
    }

    let l = targets.len();
    if l == 0 {
        return Ok(Vec::new());
    }

    let s = 2 * l + 1;
    let required = s;
    if t < required {
        return Err(AlignError::TooFewFrames {
            frames: t,
            required,
        });
    }

    let mut ext: Vec<u32> = Vec::with_capacity(s);
    for &tok in targets {
        ext.push(blank_id);
        ext.push(tok);
    }
    ext.push(blank_id);

    let emit = |frame: usize, state: usize| -> f32 { log_probs[frame * v + ext[state] as usize] };

    let neg_inf = f32::NEG_INFINITY;
    let mut alpha = vec![neg_inf; t * s];
    let mut bp = vec![0u8; t * s];

    alpha[0] = emit(0, 0);
    if s > 1 {
        alpha[1] = emit(0, 1);
    }

    for frame in 1..t {
        let prev_off = (frame - 1) * s;
        let cur_off = frame * s;
        for state in 0..s {
            let stay = alpha[prev_off + state];
            let from1 = if state >= 1 {
                alpha[prev_off + state - 1]
            } else {
                neg_inf
            };

            let from2 = if state >= 2 && ext[state] != blank_id && ext[state] != ext[state - 2] {
                alpha[prev_off + state - 2]
            } else {
                neg_inf
            };

            let (mut best, mut argmax) = (stay, 0u8);
            if from1 > best {
                best = from1;
                argmax = 1;
            }
            if from2 > best {
                best = from2;
                argmax = 2;
            }

            if best == neg_inf {
                continue;
            }

            let e = emit(frame, state);
            alpha[cur_off + state] = best + e;
            bp[cur_off + state] = argmax;
        }
    }

    let last_off = (t - 1) * s;
    let end_blank = alpha[last_off + (s - 1)];
    let end_token = if s >= 2 {
        alpha[last_off + (s - 2)]
    } else {
        neg_inf
    };
    let mut state = if end_blank >= end_token { s - 1 } else { s - 2 };
    if alpha[last_off + state] == neg_inf {
        return Err(AlignError::TooFewFrames {
            frames: t,
            required,
        });
    }

    let mut path = vec![0usize; t];
    path[t - 1] = state;
    for frame in (1..t).rev() {
        let off = bp[frame * s + state] as usize;
        state -= off;
        path[frame - 1] = state;
    }

    let mut out = vec![(0usize, 0usize); l];
    let mut seen = vec![false; l];
    for (frame, &st) in path.iter().enumerate() {
        if st % 2 == 1 {
            let i = st / 2;
            if !seen[i] {
                out[i].0 = frame;
                seen[i] = true;
            }
            out[i].1 = frame + 1;
        }
    }

    for (i, &ok) in seen.iter().enumerate() {
        if !ok {
            return Err(AlignError::TooFewFrames {
                frames: t,
                required: required + (l - i),
            });
        }
    }

    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lp(rows: &[&[f32]]) -> (Vec<f32>, usize, usize) {
        let t = rows.len();
        let v = rows[0].len();
        let mut out = Vec::with_capacity(t * v);
        for row in rows {
            assert_eq!(row.len(), v);
            let sum: f32 = row.iter().sum();
            for &p in *row {
                out.push((p / sum).ln());
            }
        }
        (out, t, v)
    }

    #[test]
    fn toy_3frame_2token_vocab3() {
        let (lps, t, v) = lp(&[
            &[0.98, 0.01, 0.01],
            &[0.01, 0.98, 0.01],
            &[0.98, 0.01, 0.01],
            &[0.01, 0.01, 0.98],
            &[0.98, 0.01, 0.01],
        ]);
        let out = viterbi_align(&lps, t, v, &[1, 2], 0).unwrap();
        assert_eq!(out, vec![(1, 2), (3, 4)]);
    }

    #[test]
    fn blanks_absorb_padding_at_start_and_end() {
        let (lps, t, v) = lp(&[
            &[0.98, 0.01, 0.01],
            &[0.98, 0.01, 0.01],
            &[0.01, 0.98, 0.01],
            &[0.01, 0.98, 0.01],
            &[0.98, 0.01, 0.01],
            &[0.98, 0.01, 0.01],
        ]);
        let out = viterbi_align(&lps, t, v, &[1], 0).unwrap();
        assert_eq!(out, vec![(2, 4)]);
    }

    #[test]
    fn empty_targets_returns_empty() {
        let (lps, t, v) = lp(&[&[0.5, 0.5], &[0.5, 0.5]]);
        let out = viterbi_align(&lps, t, v, &[], 0).unwrap();
        assert!(out.is_empty());
    }

    #[test]
    fn too_few_frames_errors() {
        let (lps, t, v) = lp(&[
            &[0.5, 0.3, 0.2],
            &[0.3, 0.5, 0.2],
            &[0.2, 0.3, 0.5],
            &[0.5, 0.3, 0.2],
        ]);
        let err = viterbi_align(&lps, t, v, &[1, 2], 0).unwrap_err();
        assert!(matches!(
            err,
            AlignError::TooFewFrames {
                frames: 4,
                required: 5
            }
        ));

        let msg = format!("{err}");
        assert!(msg.contains('4') && msg.contains('5'), "got: {msg}");
    }

    #[test]
    fn shape_mismatch_errors() {
        let lps = vec![0.0f32; 7];
        let err = viterbi_align(&lps, 3, 3, &[1], 0).unwrap_err();
        assert!(matches!(
            err,
            AlignError::ShapeMismatch {
                expected: 9,
                got: 7
            }
        ));
    }

    #[test]
    fn invalid_blank_and_token_errors() {
        let lps = vec![0.0f32; 6];
        assert!(matches!(
            viterbi_align(&lps, 2, 3, &[1], 99),
            Err(AlignError::InvalidBlankId {
                blank: 99,
                vocab: 3
            })
        ));
        assert!(matches!(
            viterbi_align(&lps, 2, 3, &[42], 0),
            Err(AlignError::InvalidTargetToken {
                index: 0,
                token: 42,
                vocab: 3
            })
        ));
    }

    #[test]
    fn repeated_token_requires_blank_between() {
        let (lps, t, v) = lp(&[
            &[0.7, 0.3],
            &[0.1, 0.9],
            &[0.7, 0.3],
            &[0.1, 0.9],
            &[0.7, 0.3],
        ]);
        let out = viterbi_align(&lps, t, v, &[1, 1], 0).unwrap();
        assert_eq!(out, vec![(1, 2), (3, 4)]);
    }

    #[test]
    fn distinct_tokens_can_skip_blank_between() {
        let (lps, t, v) = lp(&[
            &[0.98, 0.01, 0.01],
            &[0.01, 0.98, 0.01],
            &[0.01, 0.01, 0.98],
            &[0.98, 0.01, 0.01],
        ]);

        let err = viterbi_align(&lps, t, v, &[1, 2], 0).unwrap_err();
        assert!(matches!(
            err,
            AlignError::TooFewFrames {
                frames: 4,
                required: 5
            }
        ));
    }

    #[test]
    fn handles_neg_infinity_emissions_gracefully() {
        let neg = f32::NEG_INFINITY;
        let mut lps = vec![
            (0.99_f32).ln(),
            (0.01_f32).ln(),
            (0.99_f32).ln(),
            (0.01_f32).ln(),
            (0.01_f32).ln(),
            (0.99_f32).ln(),
            (0.99_f32).ln(),
            (0.01_f32).ln(),
        ];

        lps[1] = neg;
        let out = viterbi_align(&lps, 4, 2, &[1], 0).unwrap();
        assert_eq!(out, vec![(2, 3)]);

        assert!(out.iter().all(|&(s, e)| s < e));
    }

    #[test]
    fn frame_to_time_ms_defaults() {
        assert!((frame_to_time_ms(0, 320, 16_000) - 0.0).abs() < 1e-6);
        assert!((frame_to_time_ms(1, 320, 16_000) - 20.0).abs() < 1e-6);
        assert!((frame_to_time_ms(50, 320, 16_000) - 1000.0).abs() < 1e-6);
    }
}
