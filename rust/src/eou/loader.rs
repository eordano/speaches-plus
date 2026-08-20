use std::collections::HashMap;

use crate::defaults;

use super::fusion::FusionRule;
use super::{AudioPadAlignment, Eagerness, EouKind};

#[derive(Clone, Debug)]
pub struct EouConfig {
    pub kind: EouKind,
    pub p_threshold: f32,
    pub min_delay_ms: u32,
    pub max_delay_ms: u32,

    pub silence_hard_cap_ms: u32,
    pub inference_timeout_ms: u32,

    pub context_turns: u32,
    pub audio_window_ms: u32,
    pub audio_pad_alignment: AudioPadAlignment,

    pub thresholds: HashMap<String, f32>,

    pub eagerness: Option<Eagerness>,

    pub min_speech_for_response_ms: u64,

    pub eager_p_threshold: f32,

    pub eager_max_inflight: u32,

    pub eager_periodic_enabled: bool,
    pub eager_interval_ms: u32,
    pub predicted_token_buffer_cap: u32,

    pub eot_threshold: f32,
    pub eager_eot_threshold: f32,

    pub fusion_rule: FusionRule,
    pub fusion_weight_text: f32,

    pub curve_k: f32,

    pub failure_p_default: f32,

    pub failure_delay_max: bool,
}

impl Default for EouConfig {
    fn default() -> Self {
        Self {
            kind: EouKind::Vad,
            p_threshold: defaults::eou::P_THRESHOLD,
            min_delay_ms: defaults::eou::MIN_DELAY_MS,
            max_delay_ms: defaults::eou::MAX_DELAY_MS,
            silence_hard_cap_ms: defaults::eou::SILENCE_HARD_CAP_MS,
            inference_timeout_ms: defaults::eou::INFERENCE_TIMEOUT_MS,
            context_turns: defaults::eou::CONTEXT_TURNS,
            audio_window_ms: defaults::eou::AUDIO_WINDOW_MS,
            audio_pad_alignment: AudioPadAlignment::Leading,
            thresholds: HashMap::new(),
            eagerness: None,
            min_speech_for_response_ms: defaults::buffer::MIN_SPEECH_FOR_RESPONSE_MS,
            eager_p_threshold: defaults::eou::EAGER_P_THRESHOLD,
            eager_max_inflight: defaults::eou::EAGER_MAX_INFLIGHT,
            eager_periodic_enabled: defaults::eou::EAGER_PERIODIC_ENABLED,
            eager_interval_ms: defaults::eou::EAGER_INTERVAL_MS,
            predicted_token_buffer_cap: defaults::eou::PREDICTED_TOKEN_BUFFER_CAP,
            eot_threshold: defaults::eou::EOT_THRESHOLD,
            eager_eot_threshold: defaults::eou::EAGER_EOT_THRESHOLD,
            fusion_rule: FusionRule::parse(defaults::eou::FUSION_RULE)
                .unwrap_or(FusionRule::NoisyOr),
            fusion_weight_text: defaults::eou::FUSION_WEIGHT_TEXT,
            curve_k: defaults::eou::CURVE_K,
            failure_p_default: defaults::eou::FAILURE_P_DEFAULT,
            failure_delay_max: false,
        }
    }
}

impl EouConfig {
    pub fn from_env() -> Self {
        let mut cfg = Self::default();
        let legacy_enabled = env_bool(defaults::env::EOU_ENABLED);

        if let Some(e) = env_str(defaults::env::EOU_EAGERNESS)
            .as_deref()
            .and_then(Eagerness::parse)
        {
            let (p, min, max) = e.triple();
            cfg.p_threshold = p;
            cfg.min_delay_ms = min;
            cfg.max_delay_ms = max;
            cfg.eagerness = Some(e);
        } else {
            if let Some(p) = env_unit_f32(defaults::env::EOU_P_THRESHOLD) {
                cfg.p_threshold = p;
            }
            if let Some(n) = env_parse::<u32>(defaults::env::EOU_MIN_DELAY_MS) {
                cfg.min_delay_ms = n;
            }
            if let Some(n) = env_parse::<u32>(defaults::env::EOU_MAX_DELAY_MS) {
                cfg.max_delay_ms = n;
            }
        }
        if let Some(v) = env_str(defaults::env::EOU_THRESHOLDS) {
            for entry in v.split(',') {
                let Some((lang, score)) = entry.trim().split_once(':') else {
                    continue;
                };
                let lang = lang.trim();
                let Ok(s) = score.trim().parse::<f32>() else {
                    continue;
                };
                if !s.is_finite() || lang.is_empty() {
                    continue;
                }
                cfg.thresholds.insert(lang.to_string(), s.clamp(0.0, 1.0));
            }
        }
        if let Some(n) = env_parse::<u64>(defaults::env::MIN_SPEECH_FOR_RESPONSE_MS) {
            cfg.min_speech_for_response_ms = n;
        } else if let Some(n) = env_parse::<u64>(defaults::env::MIN_SPEECH_FOR_COMMIT_MS) {
            cfg.min_speech_for_response_ms = n;
            tracing::warn!(
                "{} is deprecated (v2 name); use {} (v3, §17.4)",
                defaults::env::MIN_SPEECH_FOR_COMMIT_MS,
                defaults::env::MIN_SPEECH_FOR_RESPONSE_MS,
            );
        }
        if let Some(p) = env_unit_f32(defaults::env::EOU_EAGER_P_THRESHOLD) {
            cfg.eager_p_threshold = p;
        }
        if let Some(n) = env_parse::<u32>(defaults::env::EOU_EAGER_MAX_INFLIGHT) {
            cfg.eager_max_inflight = n;
        }
        if let Some(b) = env_bool(defaults::env::EOU_EAGER_PERIODIC) {
            cfg.eager_periodic_enabled = b;
        }
        if let Some(k) = env_str(defaults::env::EOU_KIND)
            .as_deref()
            .and_then(EouKind::parse)
        {
            cfg.kind = k;
        } else if legacy_enabled == Some(true) {
            cfg.kind = EouKind::Text;
        }
        if legacy_enabled == Some(false) {
            cfg.kind = EouKind::Vad;
        }
        if let Some(n) = env_parse::<u32>(defaults::env::EOU_SILENCE_HARD_CAP_MS) {
            cfg.silence_hard_cap_ms = n;
        }
        if let Some(n) = env_parse::<u32>(defaults::env::EOU_INFERENCE_TIMEOUT_MS) {
            cfg.inference_timeout_ms = n;
        }
        if let Some(n) = env_parse::<u32>(defaults::env::EOU_CONTEXT_TURNS) {
            cfg.context_turns = n;
        }
        if let Some(n) = env_parse::<u32>(defaults::env::EOU_AUDIO_WINDOW_MS) {
            cfg.audio_window_ms = n;
        }
        if let Some(a) = env_str(defaults::env::EOU_AUDIO_PAD_ALIGNMENT)
            .as_deref()
            .and_then(AudioPadAlignment::parse)
        {
            cfg.audio_pad_alignment = a;
        }
        if let Some(n) = env_parse::<u32>(defaults::env::EOU_EAGER_INTERVAL_MS) {
            cfg.eager_interval_ms = n;
        }
        if let Some(n) = env_parse::<u32>(defaults::env::EOU_PREDICTED_TOKEN_BUFFER_CAP) {
            cfg.predicted_token_buffer_cap = n;
        }
        if let Some(p) = env_unit_f32(defaults::env::EOU_EOT_THRESHOLD) {
            cfg.eot_threshold = p;
        }
        if let Some(p) = env_unit_f32(defaults::env::EOU_EAGER_EOT_THRESHOLD) {
            cfg.eager_eot_threshold = p;
        }
        if let Some(r) = env_str(defaults::env::EOU_FUSION_RULE)
            .as_deref()
            .and_then(FusionRule::parse)
        {
            cfg.fusion_rule = r;
        }
        if let Some(w) = env_unit_f32(defaults::env::EOU_FUSION_WEIGHT_TEXT) {
            cfg.fusion_weight_text = w;
        }
        cfg
    }

    pub fn threshold_for_language(&self, lang: Option<&str>) -> f32 {
        lang.and_then(|l| self.thresholds.get(l))
            .copied()
            .unwrap_or(self.p_threshold)
    }

    pub fn eager_disabled(&self) -> bool {
        !self.eager_p_threshold.is_finite() || self.eager_p_threshold >= 1.0
    }
}

fn env_str(name: &str) -> Option<String> {
    std::env::var(name).ok().map(|v| v.trim().to_string())
}

fn env_parse<T: std::str::FromStr>(name: &str) -> Option<T> {
    std::env::var(name).ok()?.trim().parse::<T>().ok()
}

fn env_unit_f32(name: &str) -> Option<f32> {
    env_parse::<f32>(name)
        .filter(|p| p.is_finite())
        .map(|p| p.clamp(0.0, 1.0))
}

fn env_bool(name: &str) -> Option<bool> {
    env_str(name).map(|v| matches!(v.to_ascii_lowercase().as_str(), "1" | "true" | "yes" | "on"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn eou_config_default_kind_vad() {
        let c = EouConfig::default();
        assert_eq!(c.kind, EouKind::Vad);
        assert!(!c.kind.calls_classifier());
        assert_eq!(c.p_threshold, 0.5);
        assert_eq!(c.min_delay_ms, 500);
        assert_eq!(c.max_delay_ms, 3000);
        assert_eq!(c.min_speech_for_response_ms, 600);
        assert_eq!(c.inference_timeout_ms, 250);
    }

    #[test]
    fn threshold_for_language_falls_back_when_lang_none_or_missing() {
        let mut c = EouConfig::default();
        c.thresholds.insert("fr".into(), 0.7);
        assert_eq!(c.threshold_for_language(None), 0.5);
        assert_eq!(c.threshold_for_language(Some("en")), 0.5);
        assert_eq!(c.threshold_for_language(Some("fr")), 0.7);
    }

    #[test]
    fn backchannel_decision_400ms_suppresses_response() {
        let cfg = EouConfig::default();
        let audio_ms: u64 = 400;
        let suppress = audio_ms < cfg.min_speech_for_response_ms;
        assert!(suppress);
        let long_audio: u64 = 2700;
        assert!(long_audio >= cfg.min_speech_for_response_ms);
    }

    #[test]
    fn eager_threshold_eligible_at_or_above() {
        let mut c = EouConfig::default();
        c.eager_p_threshold = 0.8;
        assert!(0.8_f32 >= c.eager_p_threshold && !c.eager_disabled());
        assert!(0.95_f32 >= c.eager_p_threshold && !c.eager_disabled());
        assert!(!(0.7_f32 >= c.eager_p_threshold && !c.eager_disabled()));
    }

    #[test]
    fn eager_max_inflight_default_one() {
        assert_eq!(EouConfig::default().eager_max_inflight, 1);
    }

    #[test]
    fn eager_periodic_default_off() {
        assert!(!EouConfig::default().eager_periodic_enabled);
    }

    #[test]
    fn eou_config_default_eager_on_at_sane_threshold() {
        let c = EouConfig::default();

        assert_eq!(c.eager_p_threshold, defaults::eou::EAGER_P_THRESHOLD);
        assert_eq!(c.eager_p_threshold, 0.5);
        assert_eq!(c.eager_max_inflight, 1);
        assert!(!c.eager_periodic_enabled);
        assert!(!c.eager_disabled());
    }

    #[test]
    fn eou_config_eager_disabled_helper() {
        let mut c = EouConfig::default();
        assert!(!c.eager_disabled());
        c.eager_p_threshold = defaults::eou::EAGER_P_THRESHOLD_DISABLED;
        assert!(c.eager_disabled());
        c.eager_p_threshold = 0.9;
        assert!(!c.eager_disabled());
        c.eager_p_threshold = 1.0;
        assert!(c.eager_disabled());
        c.eager_p_threshold = 1.5;
        assert!(c.eager_disabled());
    }

    #[test]
    fn fusion_default_in_eou_config() {
        assert_eq!(EouConfig::default().fusion_rule, FusionRule::Gated);
    }
}
