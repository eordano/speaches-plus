#![allow(dead_code)]

use crate::defaults;

pub mod audio;
pub mod bpe;
pub mod byte_map;
pub mod chat_template;
pub mod fusion;
pub mod heuristic;
pub mod integrated;
pub mod loader;
pub mod onnx;
pub mod special_trie;
pub mod text_head;

pub use fusion::{
    combine_fusion, combine_fusion_gated, combine_fusion_with_features,
    extract_gated_fusion_features, is_garbage_prob, FusionEouModel, FusionRule,
    GatedFusionFeatures, GatedFusionWeights, DEFAULT_GATED_FUSION_WEIGHTS,
};
pub use heuristic::HeuristicEouModel;
pub use integrated::{FakeIntegratedBackend, IntegratedEouBackend, IntegratedVerdict};
pub use loader::EouConfig;

pub trait EouModel: Send + Sync {
    fn score(&self, context: &str) -> f32;

    fn score_with_audio(&self, context: &str, _audio: &[f32], _sample_rate: u32) -> f32 {
        self.score(context)
    }
}

pub struct StubEouModel;

impl EouModel for StubEouModel {
    fn score(&self, _context: &str) -> f32 {
        1.0
    }
}

pub struct MissingAudioEouModel;

impl EouModel for MissingAudioEouModel {
    fn score(&self, _context: &str) -> f32 {
        f32::NAN
    }
}

pub fn audio_eou_required(raw: Option<&str>) -> bool {
    match raw {
        None => false,
        Some(v) => {
            let v = v.trim();
            !(v.is_empty() || v == "0" || v.eq_ignore_ascii_case("false"))
        }
    }
}

pub fn audio_eou_missing_message(
    kind: EouKind,
    path: Option<&str>,
    suggest_required_flag: bool,
) -> String {
    let where_ = match path {
        Some(p) => format!(
            "{} = {p:?} did not load",
            defaults::env::EOU_AUDIO_MODEL_PATH
        ),
        None => format!("{} is unset", defaults::env::EOU_AUDIO_MODEL_PATH),
    };
    let suggestion = if suggest_required_flag {
        format!(
            " Set {}=1 to make this fatal at startup instead of degraded.",
            defaults::env::EOU_AUDIO_REQUIRED
        )
    } else {
        String::new()
    };
    format!(
        "EOU_KIND={} needs the smart-turn audio end-of-utterance model, but {where_}. The audio \
         head scores every utterance as unusable, so turn-taking runs on {}. Point {} at a \
         smart-turn ONNX file, or set EOU_KIND=text/heuristic to choose that path \
         explicitly.{suggestion}",
        kind.as_str(),
        match kind {
            EouKind::Fusion => "the text head alone",
            _ => "the silence hard cap alone (every turn commits immediately)",
        },
        defaults::env::EOU_AUDIO_MODEL_PATH,
    )
}

pub fn audio_eou_wanted(kind: EouKind) -> bool {
    matches!(kind, EouKind::Audio | EouKind::Fusion)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AudioEouGate {
    NotWanted,
    Present,
    DegradedWarn,
    RequiredFail,
}

pub fn audio_eou_gate(wanted: bool, present: bool, required: bool) -> AudioEouGate {
    match (wanted, present, required) {
        (false, _, _) => AudioEouGate::NotWanted,
        (true, true, _) => AudioEouGate::Present,
        (true, false, true) => AudioEouGate::RequiredFail,
        (true, false, false) => AudioEouGate::DegradedWarn,
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Eagerness {
    Low,
    Medium,
    High,
    Auto,
}

impl Eagerness {
    pub fn parse(s: &str) -> Option<Self> {
        match s.trim().to_ascii_lowercase().as_str() {
            "low" => Some(Eagerness::Low),
            "medium" | "med" => Some(Eagerness::Medium),
            "high" => Some(Eagerness::High),
            "auto" => Some(Eagerness::Auto),
            _ => None,
        }
    }

    pub fn triple(self) -> (f32, u32, u32) {
        match self {
            Eagerness::Low => defaults::eou::eagerness::LOW,
            Eagerness::Medium => defaults::eou::eagerness::MEDIUM,
            Eagerness::High => defaults::eou::eagerness::HIGH,
            Eagerness::Auto => defaults::eou::eagerness::MEDIUM,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EouKind {
    Vad,
    Text,
    Audio,
    Fusion,
    Heuristic,
    Integrated,
}

impl EouKind {
    pub const V3_SPEC: &'static [Self] = &[Self::Vad, Self::Text, Self::Audio, Self::Fusion];

    pub const EXTENSIONS: &'static [Self] = &[Self::Heuristic, Self::Integrated];

    pub fn as_str(self) -> &'static str {
        match self {
            EouKind::Vad => "vad",
            EouKind::Heuristic => "heuristic",
            EouKind::Text => "text",
            EouKind::Audio => "audio",
            EouKind::Fusion => "fusion",
            EouKind::Integrated => "integrated",
        }
    }

    pub fn parse(s: &str) -> Option<Self> {
        match s.trim().to_ascii_lowercase().as_str() {
            "vad" => Some(EouKind::Vad),
            "heuristic" => Some(EouKind::Heuristic),
            "text" => Some(EouKind::Text),
            "audio" => Some(EouKind::Audio),
            "fusion" => Some(EouKind::Fusion),
            "integrated" => Some(EouKind::Integrated),
            _ => None,
        }
    }

    pub fn calls_classifier(self) -> bool {
        !matches!(self, EouKind::Vad)
    }

    pub fn is_v3_spec(self) -> bool {
        matches!(self, Self::Vad | Self::Text | Self::Audio | Self::Fusion)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AudioPadAlignment {
    Leading,
    Trailing,
}

impl AudioPadAlignment {
    pub fn as_str(self) -> &'static str {
        match self {
            AudioPadAlignment::Leading => "leading",
            AudioPadAlignment::Trailing => "trailing",
        }
    }

    pub fn parse(s: &str) -> Option<Self> {
        match s.trim().to_ascii_lowercase().as_str() {
            "leading" => Some(AudioPadAlignment::Leading),
            "trailing" => Some(AudioPadAlignment::Trailing),
            _ => None,
        }
    }
}

pub fn sigmoid_lerp(
    p: f32,
    p_threshold: f32,
    p_max: f32,
    max_delay_ms: u64,
    min_delay_ms: u64,
) -> u64 {
    if p <= p_threshold {
        return max_delay_ms;
    }
    if p >= p_max {
        return min_delay_ms;
    }
    let span = (p_max - p_threshold).max(f32::EPSILON);
    let x = (p - p_threshold) / span;
    let k = defaults::eou::CURVE_K;
    let z = k * (x - 0.5);
    let s = 1.0 / (1.0 + (-z).exp());
    let s0 = 1.0 / (1.0 + (k * 0.5_f32).exp());
    let s1 = 1.0 / (1.0 + (-k * 0.5_f32).exp());
    let t = ((s - s0) / (s1 - s0)).clamp(0.0, 1.0);
    let max = max_delay_ms as f32;
    let min = min_delay_ms as f32;
    let delay = max + (min - max) * t;
    delay.round().max(0.0) as u64
}

pub enum HardCapRace<T> {
    HardCap,
    Completed(T),
}

pub async fn race_hard_cap<F, T>(deadline: tokio::time::Instant, fut: F) -> HardCapRace<T>
where
    F: std::future::Future<Output = T>,
{
    tokio::select! {
        biased;
        _ = tokio::time::sleep_until(deadline) => HardCapRace::HardCap,
        v = fut => HardCapRace::Completed(v),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sigmoid_lerp_at_threshold_returns_max() {
        let d = sigmoid_lerp(0.5, 0.5, 1.0, 1500, 100);
        assert!((d as i64 - 1500).abs() <= 1, "expected ~1500, got {d}");
    }

    #[test]
    fn sigmoid_lerp_at_p_max_returns_min() {
        let d = sigmoid_lerp(1.0, 0.5, 1.0, 1500, 100);
        assert!((d as i64 - 100).abs() <= 1, "expected ~100, got {d}");
    }

    #[test]
    fn sigmoid_lerp_below_threshold_returns_max() {
        let d = sigmoid_lerp(0.2, 0.5, 1.0, 1500, 100);
        assert_eq!(d, 1500);
    }

    #[test]
    fn sigmoid_lerp_monotonic_decreasing() {
        let d_low = sigmoid_lerp(0.6, 0.5, 1.0, 1500, 100);
        let d_mid = sigmoid_lerp(0.75, 0.5, 1.0, 1500, 100);
        let d_high = sigmoid_lerp(0.9, 0.5, 1.0, 1500, 100);
        assert!(d_low > d_mid, "{d_low} > {d_mid}");
        assert!(d_mid > d_high, "{d_mid} > {d_high}");
    }

    #[test]
    fn stub_eou_model_returns_one() {
        let m = StubEouModel;
        assert_eq!(m.score("anything"), 1.0);
        assert_eq!(m.score(""), 1.0);
    }

    #[test]
    fn missing_audio_model_scores_are_obviously_invalid() {
        let m = MissingAudioEouModel;
        assert!(m.score("anything").is_nan());
        assert!(m.score_with_audio("anything", &[0.0; 16], 16_000).is_nan());
        assert!(fusion::is_garbage_prob(m.score("anything")));
    }

    #[test]
    fn missing_audio_head_degrades_fusion_to_the_text_head() {
        use std::sync::Arc;
        struct Text(f32);
        impl EouModel for Text {
            fn score(&self, _: &str) -> f32 {
                self.0
            }
        }
        let fused = fusion::FusionEouModel::new(
            Arc::new(Text(0.2)),
            Arc::new(MissingAudioEouModel),
            fusion::FusionRule::Gated,
            0.5,
        );
        let p = fused.score_with_audio("the cat is on the", &[0.0; 16_000], 16_000);
        assert!(
            (p - 0.2).abs() < 1e-6,
            "absent audio head must degrade to the text score, got {p}"
        );

        let stubbed = fusion::FusionEouModel::new(
            Arc::new(Text(0.2)),
            Arc::new(StubEouModel),
            fusion::FusionRule::Gated,
            0.5,
        );
        let p_stub = stubbed.score_with_audio("the cat is on the", &[0.0; 16_000], 16_000);
        assert!(
            p_stub > 0.4,
            "regression guard: the old constant-1.0 stub inflated fusion to {p_stub}"
        );
    }

    #[test]
    fn audio_eou_required_parses_like_the_other_required_flags() {
        assert!(!audio_eou_required(None));
        assert!(!audio_eou_required(Some("")));
        assert!(!audio_eou_required(Some("0")));
        assert!(!audio_eou_required(Some("false")));
        assert!(audio_eou_required(Some("1")));
    }

    #[test]
    fn audio_eou_wanted_only_for_the_kinds_that_score_audio() {
        assert!(audio_eou_wanted(EouKind::Audio));
        assert!(audio_eou_wanted(EouKind::Fusion));
        for k in [
            EouKind::Vad,
            EouKind::Text,
            EouKind::Heuristic,
            EouKind::Integrated,
        ] {
            assert!(!audio_eou_wanted(k), "{}", k.as_str());
        }
    }

    #[test]
    fn audio_eou_gate_matrix() {
        assert_eq!(audio_eou_gate(false, false, true), AudioEouGate::NotWanted);
        assert_eq!(audio_eou_gate(false, true, false), AudioEouGate::NotWanted);
        assert_eq!(audio_eou_gate(true, true, false), AudioEouGate::Present);
        assert_eq!(audio_eou_gate(true, true, true), AudioEouGate::Present);
        assert_eq!(
            audio_eou_gate(true, false, false),
            AudioEouGate::DegradedWarn
        );
        assert_eq!(
            audio_eou_gate(true, false, true),
            AudioEouGate::RequiredFail
        );
    }

    #[test]
    fn audio_eou_missing_message_names_the_knobs() {
        let m = audio_eou_missing_message(EouKind::Fusion, None, true);
        assert!(m.contains("EOU_AUDIO_MODEL_PATH"), "{m}");
        assert!(m.contains("EOU_AUDIO_REQUIRED"), "{m}");
        assert!(m.contains("text head alone"), "{m}");
        let m2 = audio_eou_missing_message(EouKind::Audio, Some("/no/such.onnx"), false);
        assert!(m2.contains("/no/such.onnx"), "{m2}");
        assert!(
            !m2.contains("EOU_AUDIO_REQUIRED"),
            "the fatal-flag suggestion is pointless once the flag is set: {m2}"
        );
    }

    #[test]
    fn eagerness_parse_case_insensitive() {
        assert_eq!(Eagerness::parse("low"), Some(Eagerness::Low));
        assert_eq!(Eagerness::parse("LOW"), Some(Eagerness::Low));
        assert_eq!(Eagerness::parse(" High "), Some(Eagerness::High));
        assert_eq!(Eagerness::parse("medium"), Some(Eagerness::Medium));
        assert_eq!(Eagerness::parse("MED"), Some(Eagerness::Medium));
        assert_eq!(Eagerness::parse("auto"), Some(Eagerness::Auto));
        assert_eq!(Eagerness::parse("nope"), None);
    }

    #[test]
    fn eagerness_triples() {
        assert_eq!(Eagerness::Low.triple(), (0.7, 800, 3000));
        assert_eq!(Eagerness::Medium.triple(), (0.5, 500, 2500));
        assert_eq!(Eagerness::High.triple(), (0.4, 300, 1500));
        assert_eq!(Eagerness::Auto.triple(), Eagerness::Medium.triple());
    }

    #[tokio::test(start_paused = true)]
    async fn race_hard_cap_fires_when_future_hangs_forever() {
        let start = tokio::time::Instant::now();
        let deadline = start + std::time::Duration::from_millis(500);
        let hung = std::future::pending::<u32>();
        let r = race_hard_cap(deadline, hung).await;
        assert!(matches!(r, HardCapRace::HardCap));
        let elapsed = tokio::time::Instant::now() - start;
        assert!(elapsed >= std::time::Duration::from_millis(500));
    }

    #[tokio::test(start_paused = true)]
    async fn race_hard_cap_yields_value_when_future_resolves_first() {
        let deadline = tokio::time::Instant::now() + std::time::Duration::from_secs(60);
        let r = race_hard_cap(deadline, async { 42_u32 }).await;
        match r {
            HardCapRace::Completed(v) => assert_eq!(v, 42),
            HardCapRace::HardCap => panic!("hard cap fired before completed future"),
        }
    }

    #[tokio::test(start_paused = true)]
    async fn deadline_observed_twice_still_caps_the_wait() {
        let start = tokio::time::Instant::now();
        let deadline = start + std::time::Duration::from_millis(500);

        let r1 = race_hard_cap(deadline, async { 0.9_f32 }).await;
        assert!(matches!(r1, HardCapRace::Completed(_)));

        let r2 = race_hard_cap(
            deadline,
            tokio::time::sleep(std::time::Duration::from_secs(5)),
        )
        .await;
        assert!(matches!(r2, HardCapRace::HardCap));
        assert!(tokio::time::Instant::now() - start >= std::time::Duration::from_millis(500));
        assert!(tokio::time::Instant::now() - start < std::time::Duration::from_secs(5));
    }
}
