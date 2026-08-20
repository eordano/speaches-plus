const FULL_MS: u32 = 1_500;

const OFF_MS: u32 = 5_000;

const LOOSE_FLOOR: f32 = -3.0;

pub fn effective_avg_logprob_threshold(base: Option<f32>, duration_ms: u32) -> Option<f32> {
    let base = base?;
    if duration_ms <= FULL_MS {
        return Some(base);
    }
    if duration_ms >= OFF_MS {
        return None;
    }
    let frac = (duration_ms - FULL_MS) as f32 / (OFF_MS - FULL_MS) as f32;
    Some(base + frac * (LOOSE_FLOOR - base))
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NoiseRejection {
    NoSpeechProb,

    AvgLogprob,
}

impl NoiseRejection {
    pub fn as_str(self) -> &'static str {
        match self {
            NoiseRejection::NoSpeechProb => "no_speech_prob",
            NoiseRejection::AvgLogprob => "avg_logprob",
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct GateThresholds {
    pub no_speech_prob_threshold: Option<f32>,

    pub avg_logprob_threshold: Option<f32>,
}

impl GateThresholds {
    pub const fn disabled() -> Self {
        Self {
            no_speech_prob_threshold: None,
            avg_logprob_threshold: None,
        }
    }
}

pub fn evaluate(
    avg_no_speech_prob: Option<f32>,
    avg_logprob: Option<f32>,
    duration_ms: u32,
    thresholds: GateThresholds,
) -> Option<NoiseRejection> {
    if let (Some(nsp), Some(thr)) = (avg_no_speech_prob, thresholds.no_speech_prob_threshold) {
        if nsp > thr {
            return Some(NoiseRejection::NoSpeechProb);
        }
    }
    if let (Some(lp), Some(eff)) = (
        avg_logprob,
        effective_avg_logprob_threshold(thresholds.avg_logprob_threshold, duration_ms),
    ) {
        if lp < eff {
            return Some(NoiseRejection::AvgLogprob);
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn effective_returns_base_at_or_below_full_ms() {
        assert_eq!(effective_avg_logprob_threshold(Some(-1.0), 0), Some(-1.0));
        assert_eq!(
            effective_avg_logprob_threshold(Some(-1.0), 1500),
            Some(-1.0)
        );
    }

    #[test]
    fn effective_returns_none_at_or_above_off_ms() {
        assert_eq!(effective_avg_logprob_threshold(Some(-1.0), 5000), None);
        assert_eq!(effective_avg_logprob_threshold(Some(-1.0), 60_000), None);
    }

    #[test]
    fn effective_relaxes_linearly_in_window() {
        let t = effective_avg_logprob_threshold(Some(-1.0), 3250).unwrap();
        assert!((t - (-2.0)).abs() < 1e-3, "got {t}");
    }

    #[test]
    fn effective_disabled_when_base_none() {
        assert_eq!(effective_avg_logprob_threshold(None, 1000), None);
        assert_eq!(effective_avg_logprob_threshold(None, 8000), None);
    }

    #[test]
    fn evaluate_passes_when_thresholds_disabled() {
        assert_eq!(
            evaluate(Some(0.99), Some(-10.0), 1000, GateThresholds::disabled()),
            None
        );
    }

    #[test]
    fn evaluate_rejects_on_nsp_first() {
        let thr = GateThresholds {
            no_speech_prob_threshold: Some(0.6),
            avg_logprob_threshold: Some(-1.0),
        };

        assert_eq!(
            evaluate(Some(0.9), Some(-5.0), 500, thr),
            Some(NoiseRejection::NoSpeechProb)
        );
    }

    #[test]
    fn evaluate_rejects_on_logprob() {
        let thr = GateThresholds {
            no_speech_prob_threshold: Some(0.6),
            avg_logprob_threshold: Some(-1.0),
        };
        assert_eq!(
            evaluate(Some(0.1), Some(-2.0), 500, thr),
            Some(NoiseRejection::AvgLogprob)
        );
    }

    #[test]
    fn evaluate_passes_long_audio_through_avg_logprob_gate() {
        let thr = GateThresholds {
            no_speech_prob_threshold: Some(0.6),
            avg_logprob_threshold: Some(-0.5),
        };
        assert_eq!(evaluate(Some(0.1), Some(-10.0), 6_000, thr), None);
    }

    #[test]
    fn evaluate_skips_when_stats_missing() {
        let thr = GateThresholds {
            no_speech_prob_threshold: Some(0.0),
            avg_logprob_threshold: Some(0.0),
        };
        assert_eq!(evaluate(None, None, 1_000, thr), None);
    }
}
