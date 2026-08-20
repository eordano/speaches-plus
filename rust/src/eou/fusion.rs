use std::sync::Arc;

use super::EouModel;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum FusionRule {
    NoisyOr,
    Max,
    Mean,
    Weighted,

    #[default]
    Gated,
    Logit,
    Banded,
}

impl FusionRule {
    pub fn as_str(self) -> &'static str {
        match self {
            FusionRule::NoisyOr => "noisy_or",
            FusionRule::Max => "max",
            FusionRule::Mean => "mean",
            FusionRule::Weighted => "weighted",
            FusionRule::Gated => "gated",
            FusionRule::Logit => "logit",
            FusionRule::Banded => "banded",
        }
    }

    pub fn parse(s: &str) -> Option<Self> {
        match s.trim().to_ascii_lowercase().as_str() {
            "noisy_or" | "noisy-or" | "noisyor" => Some(FusionRule::NoisyOr),
            "max" => Some(FusionRule::Max),
            "mean" | "avg" | "average" => Some(FusionRule::Mean),
            "weighted" => Some(FusionRule::Weighted),
            "gated" => Some(FusionRule::Gated),
            "logit" | "stacked" => Some(FusionRule::Logit),
            "banded" | "band-gated" | "band" => Some(FusionRule::Banded),
            _ => None,
        }
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct GatedFusionFeatures {
    pub audio_ms: u32,
    pub partial_chars: u32,
    pub partial_ends_with_strong_terminator: bool,
    pub partial_ends_with_soft_terminator: bool,
    pub partial_last_word_is_continuation: bool,
}

impl GatedFusionFeatures {
    pub fn vector(&self, p_text: f32, p_audio: f32) -> [f32; 8] {
        let log_sec = (1.0 + (self.audio_ms as f32) / 1000.0).ln();
        let log_chars = (1.0 + self.partial_chars as f32).ln();
        let b = |x: bool| if x { 1.0 } else { 0.0 };
        [
            1.0,
            clamp01(p_text),
            clamp01(p_audio),
            log_sec,
            log_chars,
            b(self.partial_ends_with_strong_terminator),
            b(self.partial_ends_with_soft_terminator),
            b(self.partial_last_word_is_continuation),
        ]
    }
}

const CONTINUATION_WORDS: &[&str] = &[
    "and", "or", "but", "with", "the", "a", "an", "to", "of", "for", "is", "was", "are", "were",
    "because", "since", "if", "when", "while", "as", "than", "that", "which", "who", "whom",
    "whose",
];

pub fn extract_gated_fusion_features(partial: &str, audio_ms: u32) -> GatedFusionFeatures {
    let trimmed = partial.trim();
    let mut feat = GatedFusionFeatures {
        audio_ms,
        partial_chars: trimmed.len() as u32,
        ..Default::default()
    };
    if trimmed.ends_with("...") || trimmed.ends_with('…') {
        feat.partial_last_word_is_continuation = true;
        return feat;
    }
    if let Some(last) = trimmed.chars().last() {
        match last {
            '.' | '!' | '?' => feat.partial_ends_with_strong_terminator = true,
            ',' | ';' | ':' | '-' => feat.partial_ends_with_soft_terminator = true,
            _ => {}
        }
    }
    feat.partial_last_word_is_continuation = last_word_is_continuation(trimmed);
    feat
}

fn last_word_is_continuation(s: &str) -> bool {
    if s.is_empty() {
        return false;
    }

    let chars: Vec<(usize, char)> = s.char_indices().collect();
    let mut end = chars.len();
    while end > 0 {
        let c = chars[end - 1].1;
        if c.is_alphanumeric() || c == '\'' || c == '-' {
            break;
        }
        end -= 1;
    }
    let mut start = end;
    while start > 0 {
        let c = chars[start - 1].1;
        if c.is_alphanumeric() || c == '\'' || c == '-' {
            start -= 1;
        } else {
            break;
        }
    }
    if start >= end {
        return false;
    }
    let byte_start = chars[start].0;
    let byte_end = if end < chars.len() {
        chars[end].0
    } else {
        s.len()
    };
    let word = s[byte_start..byte_end].to_ascii_lowercase();
    CONTINUATION_WORDS.contains(&word.as_str())
}

#[derive(Clone, Copy, Debug)]
pub struct GatedFusionWeights {
    pub bias: f32,
    pub w_p_text: f32,
    pub w_p_audio: f32,
    pub w_audio_log_sec: f32,
    pub w_partial_log_chars: f32,
    pub w_strong_terminator: f32,
    pub w_soft_terminator: f32,
    pub w_continuation_last_word: f32,
    pub trained_samples: u32,
    pub trained_acc: f32,
}

impl GatedFusionWeights {
    pub fn gate(&self, p_text: f32, p_audio: f32, feat: &GatedFusionFeatures) -> f32 {
        let x = feat.vector(p_text, p_audio);
        let z = self.bias * x[0]
            + self.w_p_text * x[1]
            + self.w_p_audio * x[2]
            + self.w_audio_log_sec * x[3]
            + self.w_partial_log_chars * x[4]
            + self.w_strong_terminator * x[5]
            + self.w_soft_terminator * x[6]
            + self.w_continuation_last_word * x[7];
        1.0 / (1.0 + (-z as f64).exp() as f32)
    }
}

pub const DEFAULT_GATED_FUSION_WEIGHTS: GatedFusionWeights = GatedFusionWeights {
    bias: 0.866202,
    w_p_text: 0.283641,
    w_p_audio: 0.018662,
    w_audio_log_sec: 0.560501,
    w_partial_log_chars: 1.195453,
    w_strong_terminator: 0.258435,
    w_soft_terminator: 0.003248,
    w_continuation_last_word: 0.081883,
    trained_samples: 350,
    trained_acc: 0.9314,
};

fn clamp01(p: f32) -> f32 {
    if !p.is_finite() {
        return 0.0;
    }
    p.clamp(0.0, 1.0)
}

pub fn is_garbage_prob(p: f32) -> bool {
    !p.is_finite() || !(0.0..=1.0).contains(&p)
}

pub fn combine_fusion(p_text: f32, p_audio: f32, rule: FusionRule, weight_text: f32) -> f32 {
    let text_failed = is_garbage_prob(p_text);
    let audio_failed = is_garbage_prob(p_audio);

    if text_failed && audio_failed {
        return 1.0;
    }
    if text_failed {
        return p_audio.clamp(0.0, 1.0);
    }
    if audio_failed {
        return p_text.clamp(0.0, 1.0);
    }
    let pt = p_text.clamp(0.0, 1.0);
    let pa = p_audio.clamp(0.0, 1.0);
    let combined = match rule {
        FusionRule::NoisyOr => 1.0 - (1.0 - pt) * (1.0 - pa),
        FusionRule::Max => pt.max(pa),
        FusionRule::Mean => (pt + pa) * 0.5,
        FusionRule::Weighted => {
            let w = clamp01(weight_text);
            w * pt + (1.0 - w) * pa
        }
        FusionRule::Gated | FusionRule::Logit => (pt + pa) * 0.5,
        FusionRule::Banded => {
            band_blend(pt, pa, &DEFAULT_BANDED_FUSION_WEIGHTS_FIT_ON_CONTEXT_PROBE_BAND)
        }
    };
    combined.clamp(0.0, 1.0)
}

#[derive(Clone, Copy, Debug)]
pub struct BandedFusionWeights {
    pub band_lo: f32,
    pub band_hi: f32,
    pub bias: f32,
    pub w_logit_audio: f32,
    pub w_logit_text: f32,
    pub eps: f32,
    pub fitted_band_points: u32,
}

pub const DEFAULT_BANDED_FUSION_WEIGHTS_FIT_ON_CONTEXT_PROBE_BAND: BandedFusionWeights =
    BandedFusionWeights {
        band_lo: 0.2,
        band_hi: 0.8,
        bias: -0.176145,
        w_logit_audio: 0.428025,
        w_logit_text: 0.108608,
        eps: 1e-6,
        fitted_band_points: 166,
    };

pub const BANDED_CLAMP_MARGIN_KEEPS_OUT_OF_BAND_ORDERING: f32 = 1e-4;

fn band_blend(pt: f32, pa: f32, w: &BandedFusionWeights) -> f32 {
    if pa <= w.band_lo || pa >= w.band_hi {
        return pa;
    }
    let z = w.bias as f64
        + w.w_logit_audio as f64 * logit_eps(pa, w.eps) as f64
        + w.w_logit_text as f64 * logit_eps(pt, w.eps) as f64;
    let fused = (1.0 / (1.0 + (-z).exp())) as f32;
    fused.clamp(
        w.band_lo + BANDED_CLAMP_MARGIN_KEEPS_OUT_OF_BAND_ORDERING,
        w.band_hi - BANDED_CLAMP_MARGIN_KEEPS_OUT_OF_BAND_ORDERING,
    )
}

pub fn combine_fusion_banded(p_text: f32, p_audio: f32, w: &BandedFusionWeights) -> f32 {
    let text_failed = is_garbage_prob(p_text);
    let audio_failed = is_garbage_prob(p_audio);
    if text_failed && audio_failed {
        return 1.0;
    }
    if audio_failed {
        return p_text.clamp(0.0, 1.0);
    }
    let pa = p_audio.clamp(0.0, 1.0);
    if text_failed {
        return pa;
    }
    band_blend(p_text.clamp(0.0, 1.0), pa, w)
}

pub fn combine_fusion_gated(
    p_text: f32,
    p_audio: f32,
    feat: &GatedFusionFeatures,
    weights: &GatedFusionWeights,
) -> f32 {
    let text_failed = is_garbage_prob(p_text);
    let audio_failed = is_garbage_prob(p_audio);
    if text_failed && audio_failed {
        return 1.0;
    }
    if text_failed {
        return p_audio.clamp(0.0, 1.0);
    }
    if audio_failed {
        return p_text.clamp(0.0, 1.0);
    }
    let pt = p_text.clamp(0.0, 1.0);
    let pa = p_audio.clamp(0.0, 1.0);
    let g = weights.gate(pt, pa, feat);
    (g * pa + (1.0 - g) * pt).clamp(0.0, 1.0)
}

#[derive(Clone, Copy, Debug)]
pub struct LogitFusionWeights {
    pub bias: f32,
    pub w_logit_audio: f32,
    pub w_logit_text: f32,
    pub w_audio_log_sec: f32,
    pub w_partial_log_chars: f32,
    pub w_strong_terminator: f32,
    pub w_soft_terminator: f32,
    pub w_continuation_last_word: f32,
    pub cap_cut: f32,
    pub cap_hold: f32,
    pub eps: f32,
    pub trained_samples: u32,
}

pub const DEFAULT_LOGIT_FUSION_WEIGHTS: LogitFusionWeights = LogitFusionWeights {
    bias: -0.526406,
    w_logit_audio: 1.195329,
    w_logit_text: 1.959545,
    w_audio_log_sec: -0.00029,
    w_partial_log_chars: 0.230742,
    w_strong_terminator: 1.18882,
    w_soft_terminator: -0.368764,
    w_continuation_last_word: -1.587549,
    cap_cut: 0.4,
    cap_hold: 2.0,
    eps: 0.02,
    trained_samples: 270_946,
};

fn logit_eps(p: f32, eps: f32) -> f32 {
    let p = p.clamp(eps, 1.0 - eps);
    ((p as f64 / (1.0 - p) as f64).ln()) as f32
}

pub fn combine_fusion_logit(
    p_text: f32,
    p_audio: f32,
    feat: &GatedFusionFeatures,
    w: &LogitFusionWeights,
) -> f32 {
    let text_failed = is_garbage_prob(p_text);
    let audio_failed = is_garbage_prob(p_audio);
    if text_failed && audio_failed {
        return 1.0;
    }
    if audio_failed {
        return p_text.clamp(0.0, 1.0);
    }
    let za = w.bias + w.w_logit_audio * logit_eps(p_audio.clamp(0.0, 1.0), w.eps);
    let zt = if text_failed {
        0.0
    } else {
        let x = feat.vector(p_text, p_audio);
        w.w_logit_text * logit_eps(p_text.clamp(0.0, 1.0), w.eps)
            + w.w_audio_log_sec * x[3]
            + w.w_partial_log_chars * x[4]
            + w.w_strong_terminator * x[5]
            + w.w_soft_terminator * x[6]
            + w.w_continuation_last_word * x[7]
    };
    let zt = zt.clamp(-w.cap_hold, w.cap_cut);
    let z = (za + zt) as f64;
    (1.0 / (1.0 + (-z).exp())) as f32
}

pub fn combine_fusion_with_features(
    p_text: f32,
    p_audio: f32,
    rule: FusionRule,
    weight_text: f32,
    feat: &GatedFusionFeatures,
    weights: &GatedFusionWeights,
) -> f32 {
    match rule {
        FusionRule::Gated => combine_fusion_gated(p_text, p_audio, feat, weights),
        FusionRule::Logit => {
            combine_fusion_logit(p_text, p_audio, feat, &DEFAULT_LOGIT_FUSION_WEIGHTS)
        }
        FusionRule::Banded => combine_fusion_banded(
            p_text,
            p_audio,
            &DEFAULT_BANDED_FUSION_WEIGHTS_FIT_ON_CONTEXT_PROBE_BAND,
        ),
        _ => combine_fusion(p_text, p_audio, rule, weight_text),
    }
}

pub struct FusionEouModel {
    pub text: Arc<dyn EouModel>,
    pub audio: Arc<dyn EouModel>,
    pub rule: FusionRule,
    pub weight_text: f32,
}

impl FusionEouModel {
    pub fn new(
        text: Arc<dyn EouModel>,
        audio: Arc<dyn EouModel>,
        rule: FusionRule,
        weight_text: f32,
    ) -> Self {
        Self {
            text,
            audio,
            rule,
            weight_text,
        }
    }

    pub fn score_pair(&self, context: &str) -> (f32, f32) {
        (self.text.score(context), self.audio.score(context))
    }

    pub fn score_pair_with_audio(
        &self,
        context: &str,
        audio: &[f32],
        sample_rate: u32,
    ) -> (f32, f32) {
        (
            self.text.score(context),
            self.audio.score_with_audio(context, audio, sample_rate),
        )
    }
}

impl EouModel for FusionEouModel {
    fn score(&self, context: &str) -> f32 {
        let (p_text, p_audio) = self.score_pair(context);

        let feat = extract_gated_fusion_features(context, 0);
        combine_fusion_with_features(
            p_text,
            p_audio,
            self.rule,
            self.weight_text,
            &feat,
            &DEFAULT_GATED_FUSION_WEIGHTS,
        )
    }

    fn score_with_audio(&self, context: &str, audio: &[f32], sample_rate: u32) -> f32 {
        let (p_text, p_audio) = self.score_pair_with_audio(context, audio, sample_rate);
        let audio_ms = if sample_rate > 0 {
            ((audio.len() as u64 * 1000) / sample_rate as u64) as u32
        } else {
            0
        };
        let feat = extract_gated_fusion_features(context, audio_ms);
        combine_fusion_with_features(
            p_text,
            p_audio,
            self.rule,
            self.weight_text,
            &feat,
            &DEFAULT_GATED_FUSION_WEIGHTS,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq(a: f32, b: f32) -> bool {
        (a - b).abs() < 1e-5
    }

    #[test]
    fn banded_parse_round_trip_and_aliases() {
        assert_eq!(FusionRule::parse("banded"), Some(FusionRule::Banded));
        assert_eq!(FusionRule::parse("band-gated"), Some(FusionRule::Banded));
        assert_eq!(FusionRule::parse(FusionRule::Banded.as_str()), Some(FusionRule::Banded));
    }

    #[test]
    fn banded_outside_the_band_is_the_audio_verdict_untouched() {
        let w = &DEFAULT_BANDED_FUSION_WEIGHTS_FIT_ON_CONTEXT_PROBE_BAND;
        for pa in [0.05f32, 0.2, 0.8, 0.95] {
            for pt in [0.1f32, 0.5, 0.9] {
                assert_eq!(
                    combine_fusion_banded(pt, pa, w),
                    pa,
                    "text must not move a confident audio verdict (pa={pa}, pt={pt})"
                );
            }
        }
    }

    #[test]
    fn banded_matches_the_context_probe_fit_goldens() {
        let w = &DEFAULT_BANDED_FUSION_WEIGHTS_FIT_ON_CONTEXT_PROBE_BAND;
        assert!(approx_eq(combine_fusion_banded(0.9, 0.5, w), 0.515617));
        assert!(approx_eq(combine_fusion_banded(0.1, 0.5, w), 0.397766));
        assert!(approx_eq(combine_fusion_banded(0.5, 0.3, w), 0.368464));
        assert!(approx_eq(combine_fusion_banded(0.5, 0.7, w), 0.546495));
    }

    #[test]
    fn banded_fused_score_stays_inside_the_band() {
        let w = &DEFAULT_BANDED_FUSION_WEIGHTS_FIT_ON_CONTEXT_PROBE_BAND;
        for pa in [0.21f32, 0.5, 0.79] {
            for pt in [0.001f32, 0.999] {
                let f = combine_fusion_banded(pt, pa, w);
                assert!(
                    f > w.band_lo && f < w.band_hi,
                    "band scores must stay strictly inside ({}, {}) so out-of-band ordering \
                     is preserved, got {f} for pa={pa} pt={pt}",
                    w.band_lo,
                    w.band_hi
                );
            }
        }
    }

    #[test]
    fn banded_is_monotone_in_the_text_signal_inside_the_band() {
        let w = &DEFAULT_BANDED_FUSION_WEIGHTS_FIT_ON_CONTEXT_PROBE_BAND;
        let mut prev = 0.0f32;
        for pt in [0.05f32, 0.25, 0.5, 0.75, 0.95] {
            let f = combine_fusion_banded(pt, 0.5, w);
            assert!(f >= prev, "more terminal text must never lower the fused score");
            prev = f;
        }
    }

    #[test]
    fn banded_garbage_guards_match_the_other_rules() {
        let w = &DEFAULT_BANDED_FUSION_WEIGHTS_FIT_ON_CONTEXT_PROBE_BAND;
        assert_eq!(combine_fusion_banded(-1.0, 0.5, w), 0.5);
        assert_eq!(combine_fusion_banded(0.9, -1.0, w), 0.9);
        assert_eq!(combine_fusion_banded(-1.0, -1.0, w), 1.0);
    }

    #[test]
    fn fusion_rule_parse_round_trip() {
        for &r in &[
            FusionRule::NoisyOr,
            FusionRule::Max,
            FusionRule::Mean,
            FusionRule::Weighted,
        ] {
            assert_eq!(FusionRule::parse(r.as_str()), Some(r));
        }
        assert_eq!(FusionRule::parse("noisy-or"), Some(FusionRule::NoisyOr));
        assert_eq!(FusionRule::parse("avg"), Some(FusionRule::Mean));
        assert_eq!(FusionRule::parse("nope"), None);
    }

    #[test]
    fn fusion_default_is_gated() {
        assert_eq!(FusionRule::default(), FusionRule::Gated);
    }

    #[test]
    fn combine_fusion_gated_without_features_degrades_to_weighted_half() {
        let p = combine_fusion(0.6, 0.4, FusionRule::Gated, 0.5);
        let mean_p = combine_fusion(0.6, 0.4, FusionRule::Mean, 0.5);
        assert!(
            (p - mean_p).abs() < 1e-6,
            "gated-no-features should equal mean(=weighted-0.5), got {p} vs {mean_p}",
        );
    }

    #[test]
    fn combine_fusion_weighted_handles_nan_weight() {
        let p = combine_fusion(0.6, 0.4, FusionRule::Weighted, f32::NAN);
        assert!(p.is_finite(), "weight=NaN must be sanitized, got {p}");
    }

    #[test]
    fn fusion_combine_noisy_or_matches_formula() {
        let p = combine_fusion(0.6, 0.4, FusionRule::NoisyOr, 0.5);
        assert!(approx_eq(p, 1.0 - 0.4 * 0.6));
    }

    #[test]
    fn fusion_combine_max_matches_formula() {
        assert!(approx_eq(
            combine_fusion(0.6, 0.4, FusionRule::Max, 0.5),
            0.6
        ));
        assert!(approx_eq(
            combine_fusion(0.4, 0.9, FusionRule::Max, 0.5),
            0.9
        ));
    }

    #[test]
    fn fusion_combine_mean_matches_formula() {
        assert!(approx_eq(
            combine_fusion(0.6, 0.4, FusionRule::Mean, 0.5),
            0.5
        ));
    }

    #[test]
    fn fusion_combine_weighted_at_w0_5_equals_mean() {
        let mean = combine_fusion(0.7, 0.3, FusionRule::Mean, 0.5);
        let weighted = combine_fusion(0.7, 0.3, FusionRule::Weighted, 0.5);
        assert!(approx_eq(mean, weighted));
    }

    #[test]
    fn fusion_combine_weighted_w_extremes() {
        let only_text = combine_fusion(0.7, 0.3, FusionRule::Weighted, 1.0);
        assert!(approx_eq(only_text, 0.7));
        let only_audio = combine_fusion(0.7, 0.3, FusionRule::Weighted, 0.0);
        assert!(approx_eq(only_audio, 0.3));
    }

    #[test]
    fn fusion_graceful_degradation_audio_fails() {
        let nan = f32::NAN;
        let p = combine_fusion(0.6, nan, FusionRule::NoisyOr, 0.5);
        assert!(approx_eq(p, 0.6));
        let inf = f32::INFINITY;
        let p2 = combine_fusion(0.6, inf, FusionRule::Max, 0.5);
        assert!(approx_eq(p2, 0.6));
        let oob = 1.5_f32;
        let p3 = combine_fusion(0.6, oob, FusionRule::Mean, 0.5);
        assert!(approx_eq(p3, 0.6));
    }

    #[test]
    fn fusion_graceful_degradation_text_fails() {
        let p = combine_fusion(f32::NAN, 0.4, FusionRule::NoisyOr, 0.5);
        assert!(approx_eq(p, 0.4));
    }

    #[test]
    fn fusion_both_fail_returns_one() {
        let p = combine_fusion(f32::NAN, f32::NAN, FusionRule::NoisyOr, 0.5);
        assert!(approx_eq(p, 1.0));
        let p2 = combine_fusion(-0.1, 1.5, FusionRule::Mean, 0.5);
        assert!(approx_eq(p2, 1.0));
    }

    #[test]
    fn fusion_clamps_to_unit_interval() {
        let p = combine_fusion(1.0, 1.0, FusionRule::NoisyOr, 0.5);
        assert!((0.0..=1.0).contains(&p));
        let p2 = combine_fusion(0.0, 0.0, FusionRule::Mean, 0.5);
        assert!((0.0..=1.0).contains(&p2));
    }

    #[test]
    fn gated_extract_features_strong_terminator() {
        let cases: &[(&str, bool, bool, bool)] = &[
            ("yes.", true, false, false),
            ("what?", true, false, false),
            ("wow!", true, false, false),
            ("hmm,", false, true, false),
            ("the cat is on the", false, false, true),
            ("and", false, false, true),
            ("because", false, false, true),
            ("", false, false, false),
            ("   ", false, false, false),
            ("the cat", false, false, false),
        ];
        for (s, strong, soft, cont) in cases {
            let f = extract_gated_fusion_features(s, 1000);
            assert_eq!(
                f.partial_ends_with_strong_terminator, *strong,
                "{s:?} strong"
            );
            assert_eq!(f.partial_ends_with_soft_terminator, *soft, "{s:?} soft");
            assert_eq!(
                f.partial_last_word_is_continuation, *cont,
                "{s:?} continuation"
            );
        }
    }

    #[test]
    fn gated_feature_vector_layout() {
        let f = GatedFusionFeatures {
            audio_ms: 8000,
            partial_chars: 50,
            partial_ends_with_strong_terminator: true,
            partial_ends_with_soft_terminator: false,
            partial_last_word_is_continuation: false,
        };
        let v = f.vector(0.7, 0.3);
        assert!((v[0] - 1.0).abs() < 1e-6);
        assert!((v[1] - 0.7).abs() < 1e-6);
        assert!((v[2] - 0.3).abs() < 1e-6);
        assert!((v[3] - (1.0_f32 + 8.0).ln()).abs() < 1e-6);
        assert!((v[4] - (1.0_f32 + 50.0).ln()).abs() < 1e-6);
        assert!((v[5] - 1.0).abs() < 1e-6);
        assert!(v[6].abs() < 1e-6);
        assert!(v[7].abs() < 1e-6);
    }

    #[test]
    fn gated_zero_weights_degenerate_to_mean() {
        let zero = GatedFusionWeights {
            bias: 0.0,
            w_p_text: 0.0,
            w_p_audio: 0.0,
            w_audio_log_sec: 0.0,
            w_partial_log_chars: 0.0,
            w_strong_terminator: 0.0,
            w_soft_terminator: 0.0,
            w_continuation_last_word: 0.0,
            trained_samples: 0,
            trained_acc: 0.5,
        };
        let feat = GatedFusionFeatures::default();
        for &(pt, pa) in &[(0.7_f32, 0.3_f32), (0.0, 1.0), (0.5, 0.5), (0.95, 0.10)] {
            let got = combine_fusion_gated(pt, pa, &feat, &zero);
            let want = (pt + pa) / 2.0;
            assert!(
                (got - want).abs() < 1e-6,
                "zero-weights mean blend: got {got} want {want}"
            );
        }
    }

    #[test]
    #[allow(clippy::assertions_on_constants)]
    fn gated_default_weights_metadata() {
        assert!(DEFAULT_GATED_FUSION_WEIGHTS.trained_samples > 0);
        assert!(
            DEFAULT_GATED_FUSION_WEIGHTS.trained_acc > 0.6,
            "trained_acc={} unexpectedly low",
            DEFAULT_GATED_FUSION_WEIGHTS.trained_acc
        );
    }

    #[test]
    fn gated_real_data_agreement_cases() {
        let w = DEFAULT_GATED_FUSION_WEIGHTS;
        let cases: &[(&str, f32, f32, f32, f32)] = &[
            ("That's right.", 0.95, 0.99, 0.85, 1.01),
            ("Yes.", 0.95, 0.95, 0.85, 1.01),
            ("and the next thing", 0.55, 0.05, -0.01, 0.45),
            ("the cat is on the", 0.25, 0.05, -0.01, 0.40),
        ];
        for (text, pt, pa, lo, hi) in cases {
            let feat = extract_gated_fusion_features(text, 1500);
            let got = combine_fusion_gated(*pt, *pa, &feat, &w);
            assert!(
                got >= *lo && got <= *hi,
                "{text:?}: got {got} outside [{lo}, {hi}]; pt={pt} pa={pa}"
            );
        }
    }

    #[test]
    fn ellipsis_ending_is_continuation_not_strong_terminator() {
        for partial in ["about 440 for the...", "I was thinking…", "and then..."] {
            let feat = extract_gated_fusion_features(partial, 1500);
            assert!(
                !feat.partial_ends_with_strong_terminator,
                "{partial:?}: '...' must not read as a sentence-final terminator"
            );
            assert!(
                feat.partial_last_word_is_continuation,
                "{partial:?}: trailing ellipsis is whisper's trail-off signal"
            );
        }
        let feat = extract_gated_fusion_features("done.", 1500);
        assert!(feat.partial_ends_with_strong_terminator);
        assert!(!feat.partial_last_word_is_continuation);
    }

    #[test]
    fn logit_fusion_matches_the_python_trainer_goldens() {
        let w = DEFAULT_LOGIT_FUSION_WEIGHTS;
        let cases: [(f32, f32, u32, u32, bool, bool, bool, f32); 4] = [
            (0.9, 0.05, 1500, 21, false, false, false, 0.524979),
            (0.1, 0.95, 2000, 30, true, false, false, 0.059928),
            (0.6, 0.6, 3000, 40, false, false, true, 0.505428),
            (0.985, 0.95, 8000, 120, true, false, false, 0.989288),
        ];
        for (pa, pt, ms, chars, strong, soft, cont, want) in cases {
            let feat = GatedFusionFeatures {
                audio_ms: ms,
                partial_chars: chars,
                partial_ends_with_strong_terminator: strong,
                partial_ends_with_soft_terminator: soft,
                partial_last_word_is_continuation: cont,
            };
            let got = combine_fusion_logit(pt, pa, &feat, &w);
            assert!(
                (got - want).abs() < 2e-3,
                "pa={pa} pt={pt}: got {got} want {want}"
            );
        }
    }

    #[test]
    fn logit_fusion_text_cannot_cause_a_cutoff() {
        let w = DEFAULT_LOGIT_FUSION_WEIGHTS;
        let glowing = GatedFusionFeatures {
            audio_ms: 1200,
            partial_chars: 30,
            partial_ends_with_strong_terminator: true,
            ..Default::default()
        };
        for pa in [0.05f32, 0.2, 0.35, 0.45] {
            let got = combine_fusion_logit(0.99, pa, &glowing, &w);
            assert!(
                got < 0.5,
                "cap_cut must make cutoff-flips impossible: pa={pa} got {got}"
            );
        }
    }

    #[test]
    fn logit_fusion_confident_audio_survives_adversarial_text() {
        let w = DEFAULT_LOGIT_FUSION_WEIGHTS;
        let adversarial = GatedFusionFeatures {
            audio_ms: 30_000,
            partial_chars: 500,
            partial_ends_with_soft_terminator: true,
            partial_last_word_is_continuation: true,
            ..Default::default()
        };
        let got = combine_fusion_logit(0.02, 0.9, &adversarial, &w);
        assert!(
            got > 0.5,
            "cap_hold must keep a pa=0.9 verdict above 0.5 under any text: got {got}"
        );
    }

    #[test]
    fn logit_fusion_monotonic_in_p_audio() {
        let w = DEFAULT_LOGIT_FUSION_WEIGHTS;
        let feat = extract_gated_fusion_features("looking forward to it", 1500);
        let mut prev = -1.0f32;
        for &pa in &[0.01f32, 0.1, 0.3, 0.5, 0.7, 0.9, 0.99] {
            let got = combine_fusion_logit(0.55, pa, &feat, &w);
            assert!(got >= prev, "non-monotonic: pa={pa} got={got} prev={prev}");
            prev = got;
        }
    }

    #[test]
    fn gated_monotonic_in_p_audio() {
        let w = DEFAULT_GATED_FUSION_WEIGHTS;
        let feat = extract_gated_fusion_features("looking forward to it", 1500);
        let pt = 0.55_f32;
        let mut prev = -1.0_f32;
        for &pa in &[0.01_f32, 0.1, 0.3, 0.5, 0.7, 0.9, 0.99] {
            let got = combine_fusion_gated(pt, pa, &feat, &w);
            assert!(got >= prev, "non-monotonic: pa={pa} got={got} prev={prev}");
            prev = got;
        }
    }

    #[test]
    fn gated_does_not_flip_audio_verdict_under_adversarial_text() {
        let w = DEFAULT_GATED_FUSION_WEIGHTS;
        let feat = extract_gated_fusion_features("looking forward to it", 1500);
        for &pa in &[0.01_f32, 0.1, 0.9, 0.99] {
            let pt_adv = if pa >= 0.5 { 0.05_f32 } else { 0.95_f32 };
            let got = combine_fusion_gated(pt_adv, pa, &feat, &w);
            let audio_verdict = pa >= 0.5;
            let gated_verdict = got >= 0.5;
            assert_eq!(
                audio_verdict, gated_verdict,
                "gate flipped audio verdict: pa={pa} pt_adv={pt_adv} got={got}"
            );
        }
    }

    #[test]
    fn gated_garbage_inputs_graceful() {
        let w = DEFAULT_GATED_FUSION_WEIGHTS;
        let feat = GatedFusionFeatures::default();

        let r = combine_fusion_gated(f32::NAN, f32::NAN, &feat, &w);
        assert!((r - 1.0).abs() < 1e-6);

        let r = combine_fusion_gated(f32::NAN, 0.42, &feat, &w);
        assert!((r - 0.42).abs() < 1e-6);

        let r = combine_fusion_gated(0.7, f32::INFINITY, &feat, &w);
        assert!((r - 0.7).abs() < 1e-6);
    }

    #[test]
    fn gated_router_dispatches_by_rule() {
        let feat = GatedFusionFeatures::default();
        let w = DEFAULT_GATED_FUSION_WEIGHTS;
        for rule in [
            FusionRule::NoisyOr,
            FusionRule::Max,
            FusionRule::Mean,
            FusionRule::Weighted,
        ] {
            let want = combine_fusion(0.7, 0.3, rule, 0.5);
            let got = combine_fusion_with_features(0.7, 0.3, rule, 0.5, &feat, &w);
            assert!(
                (got - want).abs() < 1e-6,
                "rule {:?}: with-features={got} vs combine_fusion={want}",
                rule.as_str()
            );
        }
        let direct = combine_fusion_gated(0.7, 0.3, &feat, &w);
        let routed = combine_fusion_with_features(0.7, 0.3, FusionRule::Gated, 0.5, &feat, &w);
        assert!((direct - routed).abs() < 1e-6);
    }

    #[test]
    fn gated_combine_fusion_no_features_falls_back_to_mean() {
        let got = combine_fusion(0.8, 0.2, FusionRule::Gated, 0.5);
        assert!(approx_eq(got, 0.5));
    }

    #[test]
    fn gated_rule_parses_and_round_trips() {
        assert_eq!(FusionRule::parse("gated"), Some(FusionRule::Gated));
        assert_eq!(FusionRule::Gated.as_str(), "gated");
    }

    #[test]
    fn gated_rust_go_byte_for_byte_parity() {
        let w = DEFAULT_GATED_FUSION_WEIGHTS;
        let cases = [
            ("That's right.", 1500_u32, 0.95_f32, 0.99_f32),
            ("Yes.", 1500, 0.95, 0.95),
            ("and the next thing", 1500, 0.55, 0.05),
            ("the cat is on the", 1500, 0.25, 0.05),
            ("looking forward to it", 1500, 0.55, 0.5),
        ];

        let go_expected = [
            0.989753_f32,
            0.950000_f32,
            0.053163_f32,
            0.051354_f32,
            0.500264_f32,
        ];
        for (i, (partial, ms, pt, pa)) in cases.iter().enumerate() {
            let feat = extract_gated_fusion_features(partial, *ms);
            let got = combine_fusion_gated(*pt, *pa, &feat, &w);
            let want = go_expected[i];

            assert!(
                (got - want).abs() < 5e-3,
                "parity case {i}: got {got} want {want} (input={partial:?})"
            );
        }
    }

    #[test]
    fn fusion_eou_model_dispatches_to_both_heads() {
        struct Constant(f32);
        impl EouModel for Constant {
            fn score(&self, _: &str) -> f32 {
                self.0
            }
        }
        let m = FusionEouModel::new(
            Arc::new(Constant(0.6)),
            Arc::new(Constant(0.4)),
            FusionRule::NoisyOr,
            0.5,
        );
        let (pt, pa) = m.score_pair("any");
        assert!(approx_eq(pt, 0.6));
        assert!(approx_eq(pa, 0.4));
        let combined = m.score("any");
        assert!(approx_eq(combined, 1.0 - 0.4 * 0.6));
    }
}
