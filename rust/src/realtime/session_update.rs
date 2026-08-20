use serde_json::Value;

use super::session::{validate_session_max_duration_s, TurnDetectionKind};
use crate::defaults;
use crate::eou::{EouKind, FusionRule};
use crate::errors::code as errcode;

#[derive(Debug)]
pub(super) struct FieldErr {
    pub code: &'static str,
    pub param: String,
    pub message: String,
}

impl From<(&'static str, String)> for FieldErr {
    fn from((param, message): (&'static str, String)) -> Self {
        Self {
            code: errcode::SESSION_UPDATE_INVALID,
            param: param.to_string(),
            message,
        }
    }
}

const SESSION_APPLIED_KEYS: &[&str] = &[
    "instructions",
    "turn_detection",
    "session_max_duration_s",
    "voice",
    "min_speech_ms",
    "min_speech_for_response_ms",
    "no_speech_prob_threshold",
    "avg_logprob_threshold",
    "sealed_buffer_retention_count",
    "input_audio_format",
    "output_audio_format",
];

const SESSION_ACCEPTED_UNAPPLIED_KEYS: &[&str] = &[
    "id",
    "object",
    "type",
    "model",
    "modalities",
    "output_modalities",
    "audio",
    "input_audio_transcription",
    "tools",
    "tool_choice",
    "temperature",
    "max_response_output_tokens",
    "expires_at",
    "tracing",
    "include",
    "client_secret",
    "speed",
    "prompt",
    "truncation",
    "conversation",
    "partial_interval_ms",
    "drain_cap_floor_ms",
    "drain_cap_ceiling_ms",
    "outbound_queue_cap",
    "data_channel_fragment_max",
    "max_inter_delta_ms",
    "max_terminal_stall_ms",
    "predicted_flush_grace_ms",
    "inspector",
];

const TURN_DETECTION_KEYS: &[&str] = &[
    "type",
    "threshold",
    "neg_threshold",
    "min_speech_duration_ms",
    "prefix_padding_ms",
    "silence_duration_ms",
    "barge_in_delay_ms",
    "create_response",
    "eou",
    "idle_timeout_ms",
    "interrupt_response",
];

const EOU_KEYS: &[&str] = &[
    "kind",
    "text_model_path",
    "audio_model_path",
    "fusion_rule",
    "fusion_weight_text",
    "context_turns",
    "max_context_tokens",
    "audio_window_ms",
    "audio_pad_alignment",
    "p_threshold",
    "thresholds",
    "curve_k",
    "min_delay_ms",
    "max_delay_ms",
    "silence_hard_cap_ms",
    "inference_timeout_ms",
    "failure_p_default",
    "failure_delay",
];

#[derive(Clone, Copy, Debug)]
pub(super) struct EouCapability {
    pub kind: EouKind,
    pub fusion_rule: FusionRule,
}

fn reject_unknown_keys(obj: &Value, known: &[&[&str]], path: &str) -> Result<(), FieldErr> {
    let Some(map) = obj.as_object() else {
        return Ok(());
    };
    for key in map.keys() {
        if known.iter().any(|set| set.contains(&key.as_str())) {
            continue;
        }
        return Err(FieldErr {
            code: errcode::SESSION_UPDATE_INVALID,
            param: format!("{path}.{key}"),
            message: format!("unknown field {key:?} on {path}"),
        });
    }
    Ok(())
}

#[derive(Default)]
pub(super) struct StagedSessionUpdate {
    pub instructions: Option<StagedInstructions>,
    pub turn_detection: Option<StagedTurnDetection>,
    pub session_max_duration_s: Option<u64>,
    pub voice: Option<Option<String>>,
    pub min_speech_ms: Option<u64>,
    pub min_speech_for_response_ms: Option<u64>,
    pub no_speech_prob_threshold: Option<Option<f32>>,
    pub avg_logprob_threshold: Option<Option<f32>>,
    pub sealed_buffer_retention_count: Option<u32>,
    pub input_audio_format: Option<String>,
    pub output_audio_format: Option<String>,
}

pub(super) enum StagedInstructions {
    Set(String),
    Clear,
}

#[derive(Default, Debug)]
pub(super) struct StagedTurnDetection {
    pub kind: Option<TurnDetectionKind>,
    pub threshold: Option<f32>,
    pub neg_threshold: Option<Option<f32>>,
    pub min_speech_duration_ms: Option<u32>,
    pub prefix_padding_ms: Option<u32>,
    pub silence_duration_ms: Option<u32>,
    pub barge_in_delay_ms: Option<u32>,
    pub create_response: Option<bool>,
    pub eou: Option<StagedEou>,
}

#[derive(Default, Debug)]
pub(super) struct StagedEou {
    pub kind: Option<EouKind>,
    pub p_threshold: Option<f32>,
    pub curve_k: Option<f32>,
    pub min_delay_ms: Option<u32>,
    pub max_delay_ms: Option<u32>,
    pub silence_hard_cap_ms: Option<u32>,
    pub inference_timeout_ms: Option<u32>,
    pub context_turns: Option<u32>,
    pub failure_p_default: Option<f32>,
    pub failure_delay_max: Option<bool>,
    pub fusion_rule: Option<FusionRule>,
    pub fusion_weight_text: Option<f32>,
}

pub(super) fn parse_session_update(
    session_obj: &Value,
    caps: EouCapability,
) -> Result<StagedSessionUpdate, FieldErr> {
    let normalized = super::v2_compat::normalize_session_object(session_obj);
    let session_obj = &normalized;
    let mut staged = StagedSessionUpdate::default();

    reject_unknown_keys(
        session_obj,
        &[SESSION_APPLIED_KEYS, SESSION_ACCEPTED_UNAPPLIED_KEYS],
        "session",
    )?;

    if let Some(v) = session_obj.get("instructions") {
        staged.instructions = Some(match v {
            Value::String(s) if s.is_empty() => StagedInstructions::Clear,
            Value::String(s) => StagedInstructions::Set(s.clone()),
            _ => {
                return Err((
                    "session.instructions",
                    "instructions: must be a non-null string".into(),
                )
                    .into())
            }
        });
    }

    if let Some(v) = session_obj.get("turn_detection") {
        staged.turn_detection = Some(parse_turn_detection_update(v, caps)?);
    }

    if let Some(v) = session_obj.get("session_max_duration_s") {
        let n = v.as_u64().ok_or_else(|| {
            (
                "session.session_max_duration_s",
                format!(
                    "session_max_duration_s: must be in [1,{}]",
                    defaults::eou::SESSION_MAX_DURATION_S_MAX
                ),
            )
        })?;
        staged.session_max_duration_s = Some(validate_session_max_duration_s(n).map_err(|_| {
            (
                "session.session_max_duration_s",
                format!(
                    "session_max_duration_s: must be in [1,{}]",
                    defaults::eou::SESSION_MAX_DURATION_S_MAX
                ),
            )
        })?);
    }

    if let Some(v) = session_obj.get("voice") {
        staged.voice = Some(match v {
            Value::String(s) => Some(s.clone()),
            Value::Null => None,
            _ => return Err(("session.voice", "voice: must be a string or null".into()).into()),
        });
    }

    if let Some(v) = session_obj.get("min_speech_ms") {
        staged.min_speech_ms = Some(parse_bounded_u64(
            v,
            0,
            defaults::buffer::MIN_SPEECH_MS_MAX,
            "session.min_speech_ms",
            "min_speech_ms",
        )?);
    }

    if let Some(v) = session_obj.get("min_speech_for_response_ms") {
        staged.min_speech_for_response_ms = Some(parse_bounded_u64(
            v,
            0,
            defaults::buffer::MIN_SPEECH_FOR_RESPONSE_MS_MAX,
            "session.min_speech_for_response_ms",
            "min_speech_for_response_ms",
        )?);
    }

    if let Some(v) = session_obj.get("no_speech_prob_threshold") {
        staged.no_speech_prob_threshold = Some(parse_optional_unit_interval(
            v,
            "session.no_speech_prob_threshold",
            "no_speech_prob_threshold",
        )?);
    }
    if let Some(v) = session_obj.get("avg_logprob_threshold") {
        staged.avg_logprob_threshold = Some(match v {
            Value::Null => None,
            other => {
                let f = other.as_f64().ok_or((
                    "session.avg_logprob_threshold",
                    "avg_logprob_threshold: must be a number or null".to_string(),
                ))? as f32;
                if !f.is_finite() {
                    return Err((
                        "session.avg_logprob_threshold",
                        "avg_logprob_threshold: must be finite".into(),
                    )
                        .into());
                }
                Some(f)
            }
        });
    }

    if let Some(v) = session_obj.get("sealed_buffer_retention_count") {
        let n = parse_bounded_u64(
            v,
            0,
            defaults::buffer::SEALED_BUFFER_RETENTION_COUNT_MAX as u64,
            "session.sealed_buffer_retention_count",
            "sealed_buffer_retention_count",
        )?;
        staged.sealed_buffer_retention_count = Some(n as u32);
    }

    if let Some(v) = session_obj.get("input_audio_format") {
        staged.input_audio_format = Some(validate_audio_format(v).map_err(|msg| {
            (
                "session.input_audio_format",
                format!("input_audio_format: {msg}"),
            )
        })?);
    }

    if let Some(v) = session_obj.get("output_audio_format") {
        staged.output_audio_format = Some(validate_audio_format(v).map_err(|msg| {
            (
                "session.output_audio_format",
                format!("output_audio_format: {msg}"),
            )
        })?);
    }

    Ok(staged)
}

fn parse_optional_unit_interval(
    v: &Value,
    path: &'static str,
    name: &'static str,
) -> Result<Option<f32>, FieldErr> {
    match v {
        Value::Null => Ok(None),
        other => {
            let f = other
                .as_f64()
                .ok_or_else(|| (path, format!("{name}: must be a number or null")))?
                as f32;
            if !f.is_finite() || !(0.0..=1.0).contains(&f) {
                return Err((path, format!("{name}: must be in [0,1]")).into());
            }
            Ok(Some(f))
        }
    }
}

fn parse_bounded_u64(
    v: &Value,
    lo: u64,
    hi: u64,
    path: &'static str,
    name: &'static str,
) -> Result<u64, FieldErr> {
    match v.as_u64() {
        Some(n) if (lo..=hi).contains(&n) => Ok(n),
        _ => Err((path, format!("{name}: must be in [{lo},{hi}]")).into()),
    }
}

fn parse_turn_detection_update(
    td: &Value,
    caps: EouCapability,
) -> Result<StagedTurnDetection, FieldErr> {
    let mut staged = StagedTurnDetection::default();
    let obj = match td {
        Value::Object(o) => o,
        Value::Null => return Ok(staged),
        _ => {
            return Err((
                "session.turn_detection",
                "turn_detection: must be an object".into(),
            )
                .into())
        }
    };
    reject_unknown_keys(td, &[TURN_DETECTION_KEYS], "session.turn_detection")?;
    if let Some(t) = obj.get("type") {
        let s = t.as_str().ok_or((
            "session.turn_detection.type",
            "turn_detection.type: must be a string".to_string(),
        ))?;
        staged.kind = Some(TurnDetectionKind::parse(s).ok_or((
            "session.turn_detection.type",
            "turn_detection.type: must be 'server_vad' or 'none'".to_string(),
        ))?);
    }
    if let Some(v) = obj.get("threshold") {
        let f = v.as_f64().ok_or((
            "session.turn_detection.threshold",
            "turn_detection.threshold: must be a number".to_string(),
        ))? as f32;
        if !f.is_finite() || !(0.0..=1.0).contains(&f) {
            return Err((
                "session.turn_detection.threshold",
                "turn_detection.threshold: must be in [0,1]".into(),
            )
                .into());
        }
        staged.threshold = Some(f);
    }
    if let Some(v) = obj.get("neg_threshold") {
        staged.neg_threshold = Some(match v {
            Value::Null => None,
            other => {
                let f = other.as_f64().ok_or((
                    "session.turn_detection.neg_threshold",
                    "turn_detection.neg_threshold: must be a number or null".to_string(),
                ))? as f32;
                if !f.is_finite() || !(0.0..=1.0).contains(&f) {
                    return Err((
                        "session.turn_detection.neg_threshold",
                        "turn_detection.neg_threshold: must be in [0,1]".into(),
                    )
                        .into());
                }
                Some(f)
            }
        });
    }
    if let Some(v) = obj.get("min_speech_duration_ms") {
        let n = v.as_u64().ok_or((
            "session.turn_detection.min_speech_duration_ms",
            "turn_detection.min_speech_duration_ms: must be unsigned int".to_string(),
        ))?;
        if n > defaults::buffer::MIN_SPEECH_MS_MAX {
            return Err((
                "session.turn_detection.min_speech_duration_ms",
                format!(
                    "turn_detection.min_speech_duration_ms: must be in [0,{}]",
                    defaults::buffer::MIN_SPEECH_MS_MAX
                ),
            )
                .into());
        }
        staged.min_speech_duration_ms = Some(n as u32);
    }
    if let Some(v) = obj.get("prefix_padding_ms") {
        let n = v.as_u64().ok_or((
            "session.turn_detection.prefix_padding_ms",
            "turn_detection.prefix_padding_ms: must be unsigned int".to_string(),
        ))?;
        if n > defaults::turn_detection::PREFIX_PADDING_MS_MAX as u64 {
            return Err((
                "session.turn_detection.prefix_padding_ms",
                "turn_detection.prefix_padding_ms: must be in [0,1000]".into(),
            )
                .into());
        }
        staged.prefix_padding_ms = Some(n as u32);
    }
    if let Some(v) = obj.get("silence_duration_ms") {
        let n = v.as_u64().ok_or((
            "session.turn_detection.silence_duration_ms",
            "turn_detection.silence_duration_ms: must be unsigned int".to_string(),
        ))?;
        let lo = defaults::turn_detection::SILENCE_DURATION_MS_MIN as u64;
        let hi = defaults::turn_detection::SILENCE_DURATION_MS_MAX as u64;
        if !(lo..=hi).contains(&n) {
            return Err((
                "session.turn_detection.silence_duration_ms",
                "turn_detection.silence_duration_ms: must be in [50,5000]".into(),
            )
                .into());
        }
        staged.silence_duration_ms = Some(n as u32);
    }
    if let Some(v) = obj.get("barge_in_delay_ms") {
        let n = v.as_u64().ok_or((
            "session.turn_detection.barge_in_delay_ms",
            "turn_detection.barge_in_delay_ms: must be unsigned int".to_string(),
        ))?;
        if n > defaults::turn_detection::BARGE_IN_DELAY_MS_MAX as u64 {
            return Err((
                "session.turn_detection.barge_in_delay_ms",
                "turn_detection.barge_in_delay_ms: must be in [0,1000]".into(),
            )
                .into());
        }
        staged.barge_in_delay_ms = Some(n as u32);
    }
    if let Some(v) = obj.get("create_response") {
        staged.create_response = Some(v.as_bool().ok_or((
            "session.turn_detection.create_response",
            "turn_detection.create_response: must be a boolean".to_string(),
        ))?);
    }
    if let Some(v) = obj.get("eou") {
        staged.eou = Some(parse_eou_update(v, caps)?);
    }
    Ok(staged)
}

fn parse_eou_update(v: &Value, caps: EouCapability) -> Result<StagedEou, FieldErr> {
    let mut out = StagedEou::default();
    let obj = match v {
        Value::Object(o) => o,
        Value::Null => return Ok(out),
        _ => {
            return Err((
                "session.turn_detection.eou",
                "eou: must be an object".into(),
            )
                .into())
        }
    };
    reject_unknown_keys(v, &[EOU_KEYS], "session.turn_detection.eou")?;
    if let Some(v) = obj.get("kind") {
        let s = v.as_str().ok_or((
            "session.turn_detection.eou.kind",
            "eou.kind: must be a string".to_string(),
        ))?;
        let kind = EouKind::parse(s).ok_or((
            "session.turn_detection.eou.kind",
            "eou.kind: must be one of vad|heuristic|text|audio|fusion|integrated".to_string(),
        ))?;
        if kind != caps.kind {
            return Err(FieldErr {
                code: errcode::EOU_KIND_UNSUPPORTED,
                param: "session.turn_detection.eou.kind".into(),
                message: format!(
                    "eou.kind {:?} unsupported: this session is bound to {:?} at creation",
                    kind.as_str(),
                    caps.kind.as_str()
                ),
            });
        }
        out.kind = Some(kind);
    }
    if let Some(v) = obj.get("p_threshold") {
        let f = v.as_f64().ok_or((
            "session.turn_detection.eou.p_threshold",
            "eou.p_threshold: must be a number".to_string(),
        ))? as f32;
        if !f.is_finite() || !(0.0..=1.0).contains(&f) {
            return Err((
                "session.turn_detection.eou.p_threshold",
                "eou.p_threshold: must be in [0,1]".into(),
            )
                .into());
        }
        out.p_threshold = Some(f);
    }
    if let Some(v) = obj.get("curve_k") {
        let f = v.as_f64().ok_or((
            "session.turn_detection.eou.curve_k",
            "eou.curve_k: must be a number".to_string(),
        ))? as f32;
        if !f.is_finite() || f <= 0.0 || f > defaults::eou::CURVE_K_MAX {
            return Err((
                "session.turn_detection.eou.curve_k",
                "eou.curve_k: must be in (0,30]".into(),
            )
                .into());
        }
        out.curve_k = Some(f);
    }
    out.min_delay_ms = parse_opt_nonnegative_u32(
        obj.get("min_delay_ms"),
        "session.turn_detection.eou.min_delay_ms",
        "eou.min_delay_ms",
        None,
    )?;
    out.max_delay_ms = parse_opt_nonnegative_u32(
        obj.get("max_delay_ms"),
        "session.turn_detection.eou.max_delay_ms",
        "eou.max_delay_ms",
        None,
    )?;
    out.silence_hard_cap_ms = parse_opt_nonnegative_u32(
        obj.get("silence_hard_cap_ms"),
        "session.turn_detection.eou.silence_hard_cap_ms",
        "eou.silence_hard_cap_ms",
        Some((defaults::eou::SILENCE_HARD_CAP_MS_MAX, "[0,60000]")),
    )?;
    out.inference_timeout_ms = parse_opt_nonnegative_u32(
        obj.get("inference_timeout_ms"),
        "session.turn_detection.eou.inference_timeout_ms",
        "eou.inference_timeout_ms",
        Some((defaults::eou::INFERENCE_TIMEOUT_MS_MAX, "[0,10000]")),
    )?;
    out.context_turns = parse_opt_nonnegative_u32(
        obj.get("context_turns"),
        "session.turn_detection.eou.context_turns",
        "eou.context_turns",
        Some((defaults::eou::CONTEXT_TURNS_MAX, "[0,64]")),
    )?;
    if let Some(v) = obj.get("failure_p_default") {
        let f = v.as_f64().ok_or((
            "session.turn_detection.eou.failure_p_default",
            "eou.failure_p_default: must be a number".to_string(),
        ))? as f32;
        if f != 0.0 && f != 1.0 {
            return Err((
                "session.turn_detection.eou.failure_p_default",
                "eou.failure_p_default: must be 0.0 or 1.0".into(),
            )
                .into());
        }
        out.failure_p_default = Some(f);
    }
    if let Some(v) = obj.get("failure_delay") {
        let s = v.as_str().ok_or((
            "session.turn_detection.eou.failure_delay",
            "eou.failure_delay: must be a string".to_string(),
        ))?;
        out.failure_delay_max = Some(if s == defaults::failure_delay::MIN {
            false
        } else if s == defaults::failure_delay::MAX {
            true
        } else {
            return Err((
                "session.turn_detection.eou.failure_delay",
                "eou.failure_delay: must be \"min\" or \"max\"".into(),
            )
                .into());
        });
    }
    if let Some(v) = obj.get("fusion_rule") {
        let s = v.as_str().ok_or((
            "session.turn_detection.eou.fusion_rule",
            "eou.fusion_rule: must be a string".to_string(),
        ))?;
        let rule = FusionRule::parse(s).ok_or((
            "session.turn_detection.eou.fusion_rule",
            "eou.fusion_rule: must be one of noisy_or|max|mean|weighted|gated".to_string(),
        ))?;
        if rule != caps.fusion_rule {
            return Err(FieldErr {
                code: errcode::EOU_FUSION_RULE_UNSUPPORTED,
                param: "session.turn_detection.eou.fusion_rule".into(),
                message: format!(
                    "eou.fusion_rule {:?} unsupported: this session is bound to {:?} at creation",
                    rule.as_str(),
                    caps.fusion_rule.as_str()
                ),
            });
        }
        out.fusion_rule = Some(rule);
    }
    if let Some(v) = obj.get("fusion_weight_text") {
        let f = v.as_f64().ok_or((
            "session.turn_detection.eou.fusion_weight_text",
            "eou.fusion_weight_text: must be a number".to_string(),
        ))? as f32;
        if !f.is_finite() || !(0.0..=1.0).contains(&f) {
            return Err((
                "session.turn_detection.eou.fusion_weight_text",
                "eou.fusion_weight_text: must be in [0,1]".into(),
            )
                .into());
        }
        out.fusion_weight_text = Some(f);
    }
    Ok(out)
}

fn parse_opt_nonnegative_u32(
    v: Option<&Value>,
    path: &'static str,
    name: &'static str,
    cap: Option<(u32, &'static str)>,
) -> Result<Option<u32>, FieldErr> {
    let Some(v) = v else { return Ok(None) };
    let n = v
        .as_i64()
        .ok_or_else(|| (path, format!("{name}: must be an integer")))?;
    if n < 0 {
        if let Some((_, range)) = cap {
            return Err((path, format!("{name}: must be in {range}")).into());
        }
        return Err((path, format!("{name}: must be >= 0")).into());
    }
    if let Some((max, range)) = cap {
        if n > max as i64 {
            return Err((path, format!("{name}: must be in {range}")).into());
        }
    }
    Ok(Some(n as u32))
}

pub(super) fn validate_audio_format(v: &Value) -> Result<String, String> {
    let s = v.as_str().ok_or_else(|| "must be a string".to_string())?;
    if defaults::audio_format::SUPPORTED.contains(&s) {
        Ok(s.to_string())
    } else {
        Err(format!(
            "unsupported value {:?} (supported: {:?})",
            s,
            defaults::audio_format::SUPPORTED
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    const CAPS: EouCapability = EouCapability {
        kind: EouKind::Vad,
        fusion_rule: FusionRule::Gated,
    };

    fn parse(v: serde_json::Value) -> Result<StagedSessionUpdate, FieldErr> {
        parse_session_update(&v, CAPS)
    }

    #[test]
    fn no_speech_prob_threshold_set_value() {
        let s = parse(json!({"no_speech_prob_threshold": 0.6})).unwrap();

        assert_eq!(s.no_speech_prob_threshold, Some(Some(0.6)));
    }

    #[test]
    fn no_speech_prob_threshold_explicit_null_disables() {
        let s = parse(json!({"no_speech_prob_threshold": null})).unwrap();
        assert_eq!(s.no_speech_prob_threshold, Some(None));
    }

    #[test]
    fn no_speech_prob_threshold_absent_leaves_unchanged() {
        let s = parse(json!({})).unwrap();
        assert_eq!(s.no_speech_prob_threshold, None);
    }

    #[test]
    fn no_speech_prob_threshold_rejects_out_of_range() {
        let e = parse(json!({"no_speech_prob_threshold": 1.5}))
            .err()
            .expect("expected err");
        assert!(e.message.contains("must be in [0,1]"), "{e:?}");
        let e = parse(json!({"no_speech_prob_threshold": -0.1}))
            .err()
            .expect("expected err");
        assert!(e.message.contains("must be in [0,1]"), "{e:?}");
    }

    #[test]
    fn avg_logprob_threshold_accepts_negative_finite() {
        let s = parse(json!({"avg_logprob_threshold": -1.5})).unwrap();
        assert_eq!(s.avg_logprob_threshold, Some(Some(-1.5)));
    }

    #[test]
    fn avg_logprob_threshold_rejects_non_finite() {
        let s = parse(json!({"avg_logprob_threshold": null})).unwrap();
        assert_eq!(s.avg_logprob_threshold, Some(None));
    }

    #[test]
    fn neg_threshold_accepts_explicit_value() {
        let s = parse(json!({"turn_detection": {"neg_threshold": 0.4}})).unwrap();
        let td = s.turn_detection.expect("turn_detection set");
        assert_eq!(td.neg_threshold, Some(Some(0.4)));
    }

    #[test]
    fn neg_threshold_explicit_null_means_auto() {
        let s = parse(json!({"turn_detection": {"neg_threshold": null}})).unwrap();
        let td = s.turn_detection.expect("turn_detection set");
        assert_eq!(td.neg_threshold, Some(None));
    }

    #[test]
    fn neg_threshold_rejects_out_of_range() {
        let e = parse(json!({"turn_detection": {"neg_threshold": 1.2}}))
            .err()
            .expect("expected err");
        assert!(e.message.contains("must be in [0,1]"), "{e:?}");
    }

    #[test]
    fn min_speech_duration_ms_accepts_zero() {
        let s = parse(json!({"turn_detection": {"min_speech_duration_ms": 0}})).unwrap();
        let td = s.turn_detection.expect("turn_detection set");
        assert_eq!(td.min_speech_duration_ms, Some(0));
    }

    #[test]
    fn min_speech_duration_ms_rejects_oversize() {
        let too_big = defaults::buffer::MIN_SPEECH_MS_MAX + 1;
        let e = parse(json!({"turn_detection": {"min_speech_duration_ms": too_big}}))
            .err()
            .expect("expected err");
        assert!(e.message.contains("must be in"), "{e:?}");
    }

    #[test]
    fn unknown_top_level_field_is_rejected() {
        let e = parse(json!({"instrucshuns": "hi"}))
            .err()
            .expect("expected err");
        assert_eq!(e.code, errcode::SESSION_UPDATE_INVALID);
        assert_eq!(e.param, "session.instrucshuns");
        assert!(e.message.contains("unknown field"), "{e:?}");
    }

    #[test]
    fn unknown_turn_detection_field_is_rejected() {
        let e = parse(json!({"turn_detection": {"treshold": 0.5}}))
            .err()
            .expect("expected err");
        assert_eq!(e.param, "session.turn_detection.treshold");
    }

    #[test]
    fn unknown_eou_field_is_rejected() {
        let e = parse(json!({"turn_detection": {"eou": {"p_treshold": 0.5}}}))
            .err()
            .expect("expected err");
        assert_eq!(e.param, "session.turn_detection.eou.p_treshold");
    }

    #[test]
    fn echoed_session_created_view_is_accepted() {
        parse(json!({
            "id": "sess_x",
            "object": "realtime.session",
            "type": "realtime",
            "model": "m",
            "modalities": ["audio", "text"],
            "output_modalities": ["audio"],
            "input_audio_format": "pcm16",
            "output_audio_format": "pcm16",
            "input_audio_transcription": {"model": "whisper"},
            "audio": {"input": {"format": "pcm16"}, "output": {"format": "pcm16"}},
            "instructions": "be nice",
            "turn_detection": {"type": "server_vad"},
            "session_max_duration_s": 60,
            "min_speech_ms": 100,
            "min_speech_for_response_ms": 600,
            "sealed_buffer_retention_count": 4,
        }))
        .expect("round-tripped session view must be accepted");
    }

    #[test]
    fn instructions_null_is_rejected() {
        let e = parse(json!({"instructions": null}))
            .err()
            .expect("expected err");
        assert_eq!(e.code, errcode::SESSION_UPDATE_INVALID);
        assert_eq!(e.param, "session.instructions");
        assert!(e.message.contains("non-null string"), "{e:?}");
    }

    #[test]
    fn instructions_empty_string_clears() {
        let s = parse(json!({"instructions": ""})).unwrap();
        assert!(matches!(s.instructions, Some(StagedInstructions::Clear)));
    }

    #[test]
    fn eou_kind_not_bound_to_session_is_unsupported() {
        let e = parse(json!({"turn_detection": {"eou": {"kind": "fusion"}}}))
            .err()
            .expect("expected err");
        assert_eq!(e.code, errcode::EOU_KIND_UNSUPPORTED);
        assert_eq!(e.param, "session.turn_detection.eou.kind");
    }

    #[test]
    fn eou_kind_matching_session_is_accepted() {
        let s = parse(json!({"turn_detection": {"eou": {"kind": "vad"}}})).unwrap();
        let eou = s.turn_detection.and_then(|td| td.eou).expect("eou set");
        assert_eq!(eou.kind, Some(EouKind::Vad));
    }

    #[test]
    fn eou_fusion_rule_not_bound_to_session_is_unsupported() {
        let e = parse(json!({"turn_detection": {"eou": {"fusion_rule": "max"}}}))
            .err()
            .expect("expected err");
        assert_eq!(e.code, errcode::EOU_FUSION_RULE_UNSUPPORTED);
        assert_eq!(e.param, "session.turn_detection.eou.fusion_rule");
    }

    #[test]
    fn eou_unsupported_value_still_reported_as_invalid() {
        let e = parse(json!({"turn_detection": {"eou": {"kind": "telepathy"}}}))
            .err()
            .expect("expected err");
        assert_eq!(e.code, errcode::SESSION_UPDATE_INVALID);
    }
}
