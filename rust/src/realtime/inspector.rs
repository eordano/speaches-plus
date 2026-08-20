#![allow(dead_code)]

use std::collections::BTreeMap;
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::Arc;

use serde_json::{json, Value};

use crate::defaults;
use crate::inspect::{Corr, InspectorRelay};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HardCapPhase {
    DuringEou,
    DuringWait,
}

impl HardCapPhase {
    pub fn as_str(self) -> &'static str {
        match self {
            HardCapPhase::DuringEou => "during_eou",
            HardCapPhase::DuringWait => "during_wait",
        }
    }
}

#[derive(Debug, Clone)]
pub enum InspectorEvent {
    VadConfirmedStart {
        session_id: String,
        item_id: String,
        ms: u64,
    },
    VadConfirmedStop {
        session_id: String,
        item_id: String,
        ms: u64,
    },
    BargeinPending {
        session_id: String,
        delay_ms: u64,
    },
    BargeinFired {
        session_id: String,
        played_ms: u64,
    },
    BargeinSuppressed {
        session_id: String,
        reason: &'static str,
    },
    EouScored {
        session_id: String,
        kind: &'static str,
        score: f32,
        eager_score: Option<f32>,
        threshold: f32,
        language: Option<String>,
        input_chars: Option<u32>,
        input_audio_ms: Option<u32>,
        delay_ms: u64,
        elapsed_ms: u64,
        cancelled_by: &'static str,
        hard_cap_fired: bool,
    },

    EouHardCapFired {
        session_id: String,
        item_id: String,
        phase: HardCapPhase,
        score: Option<f32>,
    },
    EouEagerDispatch {
        session_id: String,
        response_id: String,
        item_id: String,
        score: f32,
        threshold: f32,
        epoch: u64,
    },
    EouPredictedOverflow {
        session_id: String,
        response_id: String,
        dropped_tokens: u32,
    },
    EouPredictedRollback {
        session_id: String,
        response_id: String,
        reason: &'static str,
        llm_chars_thrown: u32,
    },
    PacerPlayedMs {
        session_id: String,
        played_ms: u64,
    },
    StateTransition {
        session_id: String,
        phase: &'static str,
        from: String,
        to: String,
    },
    DrainStart {
        session_id: String,
        response_id: String,
        planned_ms: u64,
    },
    DrainComplete {
        session_id: String,
        response_id: String,
        played_ms: u64,
        status: &'static str,
    },
    PartialTranscription {
        session_id: String,
        item_id: String,
        transcript: String,
        ms: u64,
    },

    PredictedRollback {
        session_id: String,
        response_id: String,
        score: f32,
    },

    PredictedSuppressed {
        session_id: String,
        score: f32,
        inflight: u32,
    },

    PredictedPromoted {
        session_id: String,
        response_id: String,
        score: f32,
    },

    OutboundQueueExceeded {
        session_id: String,
        response_id: String,
        queued_ms: u64,
        cap_ms: u64,
    },

    InvariantViolation {
        session_id: String,
        violation: String,
    },

    VadFailed {
        session_id: String,
        reason: String,
    },

    BackchannelSuppressed {
        session_id: String,
        item_id: String,
        audio_ms: u64,
        transcript: Option<String>,
    },

    DiarizationEmitted {
        session_id: String,
        item_id: String,
        audio_end_ms: u64,
        num_segments: u32,
        num_speakers: u32,
        elapsed_ms: u64,
        failed: bool,
        reason: Option<String>,
    },

    SttFinal {
        session_id: String,
        item_id: String,
        text: String,
        audio_start_ms: u64,
        audio_end_ms: u64,
    },

    LlmRequest {
        session_id: String,
        response_id: String,
        model: String,
    },

    LlmFirstToken {
        session_id: String,
        response_id: String,
        elapsed_ms: u64,
    },

    LlmDone {
        session_id: String,
        response_id: String,
        reply_chars: usize,
        elapsed_ms: u64,
    },

    TtsPhraseStart {
        session_id: String,
        response_id: String,
        text: String,
        voice: String,
    },

    TtsChunk {
        session_id: String,
        response_id: String,
        chunk_idx: u32,
        ms_audio: u64,
        first: bool,
    },

    TurnStart {
        session_id: String,
        turn_id: String,
        role: &'static str,
    },

    TurnUserCommitted {
        session_id: String,
        item_id: String,
    },

    TurnEnd {
        session_id: String,
        turn_id: String,
    },
}

pub struct WirePayload {
    pub lane: &'static str,
    pub kind: &'static str,
    pub corr: Corr,
    pub payload: BTreeMap<String, Value>,
}

impl InspectorEvent {
    pub fn to_wire(&self) -> WirePayload {
        match self {
            InspectorEvent::VadConfirmedStart { item_id, ms, .. } => WirePayload {
                lane: "vad",
                kind: "confirmed_start",
                corr: Corr {
                    item_id: Some(item_id.clone()),
                    ..Default::default()
                },
                payload: into_map([("audio_start_ms", json!(ms))]),
            },
            InspectorEvent::VadConfirmedStop { item_id, ms, .. } => WirePayload {
                lane: "vad",
                kind: "stopped",
                corr: Corr {
                    item_id: Some(item_id.clone()),
                    ..Default::default()
                },
                payload: into_map([("audio_end_ms", json!(ms))]),
            },
            InspectorEvent::BargeinPending { delay_ms, .. } => WirePayload {
                lane: "bargein",
                kind: "bargein_pending",
                corr: Corr::default(),
                payload: into_map([("delay_ms", json!(delay_ms))]),
            },
            InspectorEvent::BargeinFired { played_ms, .. } => WirePayload {
                lane: "bargein",
                kind: "bargein_fired",
                corr: Corr::default(),
                payload: into_map([("played_ms", json!(played_ms))]),
            },
            InspectorEvent::BargeinSuppressed { reason, .. } => WirePayload {
                lane: "bargein",
                kind: "bargein_cancelled",
                corr: Corr::default(),
                payload: into_map([("reason", json!(reason))]),
            },
            InspectorEvent::EouScored {
                kind,
                score,
                eager_score,
                threshold,
                language,
                input_chars,
                input_audio_ms,
                delay_ms,
                elapsed_ms,
                cancelled_by,
                hard_cap_fired,
                ..
            } => WirePayload {
                lane: "eou",
                kind: "scored",
                corr: Corr::default(),
                payload: into_map([
                    ("eou_kind", json!(kind)),
                    ("score", json!(score)),
                    ("eager_score", json!(eager_score)),
                    ("threshold", json!(threshold)),
                    ("language", json!(language)),
                    ("input_chars", json!(input_chars)),
                    ("input_audio_ms", json!(input_audio_ms)),
                    ("delay_ms", json!(delay_ms)),
                    ("elapsed_ms", json!(elapsed_ms)),
                    ("cancelled_by", json!(cancelled_by)),
                    ("hard_cap_fired", json!(hard_cap_fired)),
                ]),
            },
            InspectorEvent::EouHardCapFired {
                item_id,
                phase,
                score,
                ..
            } => WirePayload {
                lane: "eou",
                kind: "hard_cap_fired",
                corr: Corr {
                    item_id: Some(item_id.clone()),
                    ..Default::default()
                },
                payload: into_map([("phase", json!(phase.as_str())), ("score", json!(score))]),
            },
            InspectorEvent::EouEagerDispatch {
                response_id,
                item_id,
                score,
                threshold,
                epoch,
                ..
            } => WirePayload {
                lane: "eou",
                kind: "eager_dispatch",
                corr: Corr {
                    response_id: Some(response_id.clone()),
                    item_id: Some(item_id.clone()),
                    ..Default::default()
                },
                payload: into_map([
                    ("score", json!(score)),
                    ("threshold", json!(threshold)),
                    ("epoch", json!(epoch)),
                ]),
            },
            InspectorEvent::EouPredictedOverflow {
                response_id,
                dropped_tokens,
                ..
            } => WirePayload {
                lane: "response",
                kind: "predicted_overflow",
                corr: Corr {
                    response_id: Some(response_id.clone()),
                    ..Default::default()
                },
                payload: into_map([("dropped_tokens", json!(dropped_tokens))]),
            },
            InspectorEvent::EouPredictedRollback {
                response_id,
                reason,
                llm_chars_thrown,
                ..
            } => WirePayload {
                lane: "response",
                kind: "predicted_rollback",
                corr: Corr {
                    response_id: Some(response_id.clone()),
                    ..Default::default()
                },
                payload: into_map([
                    ("reason", json!(reason)),
                    ("llm_chars_thrown", json!(llm_chars_thrown)),
                ]),
            },
            InspectorEvent::PacerPlayedMs { played_ms, .. } => WirePayload {
                lane: "tts_pacer",
                kind: "played_ms",
                corr: Corr::default(),
                payload: into_map([("played_ms", json!(played_ms))]),
            },
            InspectorEvent::StateTransition {
                phase, from, to, ..
            } => WirePayload {
                lane: "state",
                kind: "transition",
                corr: Corr::default(),
                payload: into_map([
                    ("phase", json!(phase)),
                    ("from", json!(from)),
                    ("to", json!(to)),
                ]),
            },
            InspectorEvent::DrainStart {
                response_id,
                planned_ms,
                ..
            } => WirePayload {
                lane: "response",
                kind: "drain_start",
                corr: Corr {
                    response_id: Some(response_id.clone()),
                    ..Default::default()
                },
                payload: into_map([("planned_ms", json!(planned_ms))]),
            },
            InspectorEvent::DrainComplete {
                response_id,
                played_ms,
                status,
                ..
            } => WirePayload {
                lane: "response",
                kind: "drain_complete",
                corr: Corr {
                    response_id: Some(response_id.clone()),
                    ..Default::default()
                },
                payload: into_map([("played_ms", json!(played_ms)), ("status", json!(status))]),
            },
            InspectorEvent::PartialTranscription {
                item_id,
                transcript,
                ms,
                ..
            } => WirePayload {
                lane: "stt",
                kind: "partial",
                corr: Corr {
                    item_id: Some(item_id.clone()),
                    ..Default::default()
                },
                payload: into_map([("text", json!(transcript)), ("audio_end_ms", json!(ms))]),
            },
            InspectorEvent::PredictedRollback {
                response_id, score, ..
            } => WirePayload {
                lane: "response",
                kind: "predicted_rollback",
                corr: Corr {
                    response_id: Some(response_id.clone()),
                    ..Default::default()
                },
                payload: into_map([("score", json!(score))]),
            },
            InspectorEvent::PredictedSuppressed {
                score, inflight, ..
            } => WirePayload {
                lane: "response",
                kind: "predicted_suppressed",
                corr: Corr::default(),
                payload: into_map([("score", json!(score)), ("inflight", json!(inflight))]),
            },
            InspectorEvent::PredictedPromoted {
                response_id, score, ..
            } => WirePayload {
                lane: "response",
                kind: "predicted_promoted",
                corr: Corr {
                    response_id: Some(response_id.clone()),
                    ..Default::default()
                },
                payload: into_map([("score", json!(score))]),
            },
            InspectorEvent::OutboundQueueExceeded {
                response_id,
                queued_ms,
                cap_ms,
                ..
            } => WirePayload {
                lane: "wire",
                kind: "err",
                corr: Corr {
                    response_id: Some(response_id.clone()),
                    ..Default::default()
                },
                payload: into_map([
                    ("reason", json!("client_too_slow")),
                    ("queued_ms", json!(queued_ms)),
                    ("cap_ms", json!(cap_ms)),
                ]),
            },
            InspectorEvent::InvariantViolation { violation, .. } => WirePayload {
                lane: "error",
                kind: "invariant_violation",
                corr: Corr::default(),
                payload: into_map([("violation", json!(violation))]),
            },
            InspectorEvent::VadFailed { reason, .. } => WirePayload {
                lane: "error",
                kind: "vad_failed",
                corr: Corr::default(),
                payload: into_map([("reason", json!(reason))]),
            },
            InspectorEvent::BackchannelSuppressed {
                item_id,
                audio_ms,
                transcript,
                ..
            } => WirePayload {
                lane: "turn",
                kind: "backchannel_suppressed",
                corr: Corr {
                    item_id: Some(item_id.clone()),
                    ..Default::default()
                },
                payload: into_map([
                    ("audio_ms", json!(audio_ms)),
                    ("transcript", json!(transcript)),
                ]),
            },
            InspectorEvent::DiarizationEmitted {
                item_id,
                audio_end_ms,
                num_segments,
                num_speakers,
                elapsed_ms,
                failed,
                reason,
                ..
            } => WirePayload {
                lane: "diarization",
                kind: if *failed {
                    "failed"
                } else if *num_segments == 0 {
                    "empty"
                } else {
                    "emitted"
                },
                corr: Corr {
                    item_id: Some(item_id.clone()),
                    ..Default::default()
                },
                payload: into_map([
                    ("audio_end_ms", json!(audio_end_ms)),
                    ("num_segments", json!(num_segments)),
                    ("num_speakers", json!(num_speakers)),
                    ("elapsed_ms", json!(elapsed_ms)),
                    ("failed", json!(failed)),
                    ("reason", json!(reason)),
                ]),
            },
            InspectorEvent::SttFinal {
                item_id,
                text,
                audio_start_ms,
                audio_end_ms,
                ..
            } => WirePayload {
                lane: "stt",
                kind: "final",
                corr: Corr {
                    item_id: Some(item_id.clone()),
                    ..Default::default()
                },
                payload: into_map([
                    ("text", json!(text)),
                    ("audio_start_ms", json!(audio_start_ms)),
                    ("audio_end_ms", json!(audio_end_ms)),
                ]),
            },
            InspectorEvent::LlmRequest {
                response_id, model, ..
            } => WirePayload {
                lane: "llm",
                kind: "request",
                corr: Corr {
                    response_id: Some(response_id.clone()),
                    ..Default::default()
                },
                payload: into_map([("model", json!(model))]),
            },
            InspectorEvent::LlmFirstToken {
                response_id,
                elapsed_ms,
                ..
            } => WirePayload {
                lane: "llm",
                kind: "first_token",
                corr: Corr {
                    response_id: Some(response_id.clone()),
                    ..Default::default()
                },
                payload: into_map([("elapsed_ms", json!(elapsed_ms))]),
            },
            InspectorEvent::LlmDone {
                response_id,
                reply_chars,
                elapsed_ms,
                ..
            } => WirePayload {
                lane: "llm",
                kind: "done",
                corr: Corr {
                    response_id: Some(response_id.clone()),
                    ..Default::default()
                },
                payload: into_map([
                    ("reply_chars", json!(reply_chars)),
                    ("elapsed_ms", json!(elapsed_ms)),
                ]),
            },
            InspectorEvent::TtsPhraseStart {
                response_id,
                text,
                voice,
                ..
            } => WirePayload {
                lane: "tts_req",
                kind: "phrase_sent",
                corr: Corr {
                    response_id: Some(response_id.clone()),
                    ..Default::default()
                },
                payload: into_map([("text", json!(text)), ("voice", json!(voice))]),
            },
            InspectorEvent::TtsChunk {
                response_id,
                chunk_idx,
                ms_audio,
                first,
                ..
            } => WirePayload {
                lane: "tts_chunk",
                kind: if *first { "first_chunk" } else { "chunk" },
                corr: Corr {
                    response_id: Some(response_id.clone()),
                    ..Default::default()
                },
                payload: into_map([
                    ("chunk_idx", json!(chunk_idx)),
                    ("ms_audio", json!(ms_audio)),
                ]),
            },
            InspectorEvent::TurnStart { turn_id, role, .. } => WirePayload {
                lane: "turn",
                kind: "turn_start",
                corr: Corr {
                    turn_id: Some(turn_id.clone()),
                    ..Default::default()
                },
                payload: into_map([("role", json!(role))]),
            },
            InspectorEvent::TurnUserCommitted { item_id, .. } => WirePayload {
                lane: "turn",
                kind: "user_committed",
                corr: Corr {
                    item_id: Some(item_id.clone()),
                    ..Default::default()
                },
                payload: BTreeMap::new(),
            },
            InspectorEvent::TurnEnd { turn_id, .. } => WirePayload {
                lane: "turn",
                kind: "turn_end",
                corr: Corr {
                    turn_id: Some(turn_id.clone()),
                    ..Default::default()
                },
                payload: BTreeMap::new(),
            },
        }
    }
}

fn into_map<const N: usize>(entries: [(&'static str, Value); N]) -> BTreeMap<String, Value> {
    let mut m = BTreeMap::new();
    for (k, v) in entries {
        m.insert(k.to_string(), v);
    }
    m
}

pub trait InspectorSink: Send + Sync {
    fn emit(&self, event: InspectorEvent);
}

pub struct RelayInspectorSink {
    relay: Arc<InspectorRelay>,
}

impl RelayInspectorSink {
    pub fn new(relay: Arc<InspectorRelay>) -> Self {
        Self { relay }
    }
}

impl InspectorSink for RelayInspectorSink {
    fn emit(&self, event: InspectorEvent) {
        let wire = event.to_wire();
        self.relay
            .publish(wire.lane, wire.kind, Some(wire.corr), wire.payload);
    }
}

pub struct FanoutSink {
    sinks: Vec<Arc<dyn InspectorSink>>,
}

impl FanoutSink {
    pub fn new(sinks: Vec<Arc<dyn InspectorSink>>) -> Self {
        Self { sinks }
    }
}

impl InspectorSink for FanoutSink {
    fn emit(&self, event: InspectorEvent) {
        for sink in &self.sinks {
            sink.emit(event.clone());
        }
    }
}

#[derive(Default)]
pub struct NoopSink;

impl InspectorSink for NoopSink {
    fn emit(&self, _event: InspectorEvent) {}
}

pub struct TracingSink {
    keep_per_thousand: u32,
    counter: AtomicU32,
}

impl TracingSink {
    pub fn new(sample_rate: f32) -> Self {
        let r = sample_rate.clamp(0.0, 1.0);
        Self {
            keep_per_thousand: (r * 1000.0).round() as u32,
            counter: AtomicU32::new(0),
        }
    }

    fn keep(&self, event: &InspectorEvent) -> bool {
        if self.keep_per_thousand >= 1000 {
            return true;
        }
        if self.keep_per_thousand == 0 {
            return false;
        }
        match event {
            InspectorEvent::StateTransition { .. } | InspectorEvent::PacerPlayedMs { .. } => {
                let n = self.counter.fetch_add(1, Ordering::Relaxed) % 1000;
                n < self.keep_per_thousand
            }
            _ => true,
        }
    }
}

impl InspectorSink for TracingSink {
    fn emit(&self, event: InspectorEvent) {
        if self.keep(&event) {
            tracing::debug!(?event, "inspector");
        }
    }
}

fn read_transitions_enabled() -> bool {
    match std::env::var(defaults::env::INSPECTOR_TRANSITIONS).ok() {
        Some(v) => matches!(
            v.trim().to_ascii_lowercase().as_str(),
            "1" | "true" | "yes" | "on"
        ),
        None => defaults::inspector::TRANSITIONS_ENABLED,
    }
}

fn read_transitions_sample_rate() -> f32 {
    std::env::var(defaults::env::INSPECTOR_TRANSITIONS_SAMPLE_RATE)
        .ok()
        .and_then(|s| s.trim().parse::<f32>().ok())
        .filter(|f| f.is_finite())
        .map(|f| f.clamp(0.0, 1.0))
        .unwrap_or(if cfg!(debug_assertions) {
            defaults::inspector::TRANSITIONS_SAMPLE_RATE_DEV
        } else {
            defaults::inspector::TRANSITIONS_SAMPLE_RATE_RELEASE
        })
}

pub fn default_sink() -> Arc<dyn InspectorSink> {
    if read_transitions_enabled() {
        Arc::new(TracingSink::new(read_transitions_sample_rate()))
    } else {
        Arc::new(NoopSink)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;

    struct VecSink(Mutex<Vec<InspectorEvent>>);

    impl InspectorSink for VecSink {
        fn emit(&self, event: InspectorEvent) {
            self.0.lock().unwrap().push(event);
        }
    }

    #[test]
    fn noop_sink_swallows_events() {
        let sink = NoopSink;
        sink.emit(InspectorEvent::PacerPlayedMs {
            session_id: "s".into(),
            played_ms: 100,
        });
    }

    #[test]
    fn vec_sink_records_events() {
        let sink = VecSink(Mutex::new(Vec::new()));
        sink.emit(InspectorEvent::BargeinFired {
            session_id: "s".into(),
            played_ms: 250,
        });
        let v = sink.0.lock().unwrap();
        assert_eq!(v.len(), 1);
    }

    #[test]
    fn fanout_dispatches_to_all_legs() {
        let a = Arc::new(VecSink(Mutex::new(Vec::new())));
        let b = Arc::new(VecSink(Mutex::new(Vec::new())));
        let fan = FanoutSink::new(vec![a.clone(), b.clone()]);
        fan.emit(InspectorEvent::PacerPlayedMs {
            session_id: "s".into(),
            played_ms: 1,
        });
        assert_eq!(a.0.lock().unwrap().len(), 1);
        assert_eq!(b.0.lock().unwrap().len(), 1);
    }

    #[test]
    fn to_wire_vad_confirmed_start() {
        let ev = InspectorEvent::VadConfirmedStart {
            session_id: "s".into(),
            item_id: "item_a".into(),
            ms: 100,
        };
        let w = ev.to_wire();
        assert_eq!(w.lane, "vad");
        assert_eq!(w.kind, "confirmed_start");
        assert_eq!(w.corr.item_id.as_deref(), Some("item_a"));
        assert_eq!(
            w.payload.get("audio_start_ms"),
            Some(&serde_json::json!(100))
        );
    }

    #[test]
    fn to_wire_eou_scored_full_envelope() {
        let ev = InspectorEvent::EouScored {
            session_id: "s".into(),
            kind: "vad",
            score: 0.8,
            eager_score: Some(0.9),
            threshold: 0.5,
            language: Some("en".into()),
            input_chars: Some(42),
            input_audio_ms: Some(2000),
            delay_ms: 500,
            elapsed_ms: 12,
            cancelled_by: "none",
            hard_cap_fired: false,
        };
        let w = ev.to_wire();
        assert_eq!(w.lane, "eou");
        assert_eq!(w.kind, "scored");
        assert_eq!(w.payload.get("delay_ms"), Some(&serde_json::json!(500)));
        assert_eq!(w.payload.get("threshold"), Some(&serde_json::json!(0.5)));
    }

    #[test]
    fn to_wire_invariant_violation_uses_error_lane() {
        let ev = InspectorEvent::InvariantViolation {
            session_id: "s".into(),
            violation: "I1".into(),
        };
        let w = ev.to_wire();
        assert_eq!(w.lane, "error");
        assert_eq!(w.kind, "invariant_violation");
        assert_eq!(w.payload.get("violation"), Some(&serde_json::json!("I1")));
    }

    #[test]
    fn to_wire_outbound_queue_exceeded_uses_wire_err() {
        let ev = InspectorEvent::OutboundQueueExceeded {
            session_id: "s".into(),
            response_id: "r".into(),
            queued_ms: 7000,
            cap_ms: 5000,
        };
        let w = ev.to_wire();
        assert_eq!(w.lane, "wire");
        assert_eq!(w.kind, "err");
        assert_eq!(w.corr.response_id.as_deref(), Some("r"));
    }

    #[test]
    fn to_wire_eou_eager_dispatch_is_eou_lane_with_score_and_threshold() {
        let ev = InspectorEvent::EouEagerDispatch {
            session_id: "s".into(),
            response_id: "resp_eager".into(),
            item_id: "item_e".into(),
            score: 0.5,
            threshold: 0.5,
            epoch: 4,
        };
        let w = ev.to_wire();
        assert_eq!(w.lane, "eou");
        assert_eq!(w.kind, "eager_dispatch");
        assert_eq!(w.corr.response_id.as_deref(), Some("resp_eager"));
        assert_eq!(w.corr.item_id.as_deref(), Some("item_e"));
        let score = w
            .payload
            .get("score")
            .and_then(|v| v.as_f64())
            .expect("score f64");
        assert!((score - 0.5_f64).abs() < 1e-5);
        let thr = w
            .payload
            .get("threshold")
            .and_then(|v| v.as_f64())
            .expect("threshold f64");
        assert!((thr - 0.5_f64).abs() < 1e-5);
        assert_eq!(w.payload.get("epoch"), Some(&serde_json::json!(4)));
    }

    #[test]
    fn to_wire_eou_predicted_overflow_includes_dropped_count() {
        let ev = InspectorEvent::EouPredictedOverflow {
            session_id: "s".into(),
            response_id: "resp_o".into(),
            dropped_tokens: 7,
        };
        let w = ev.to_wire();
        assert_eq!(w.lane, "response");
        assert_eq!(w.kind, "predicted_overflow");
        assert_eq!(w.payload.get("dropped_tokens"), Some(&serde_json::json!(7)));
    }

    #[test]
    fn to_wire_eou_predicted_rollback_includes_reason() {
        let ev = InspectorEvent::EouPredictedRollback {
            session_id: "s".into(),
            response_id: "resp_r".into(),
            reason: "transcript_mismatch",
            llm_chars_thrown: 12,
        };
        let w = ev.to_wire();
        assert_eq!(w.lane, "response");
        assert_eq!(w.kind, "predicted_rollback");
        assert_eq!(
            w.payload.get("reason"),
            Some(&serde_json::json!("transcript_mismatch"))
        );
        assert_eq!(
            w.payload.get("llm_chars_thrown"),
            Some(&serde_json::json!(12))
        );
    }

    #[test]
    fn to_wire_partial_transcription_uses_stt_partial() {
        let ev = InspectorEvent::PartialTranscription {
            session_id: "s".into(),
            item_id: "item_x".into(),
            transcript: "hello".into(),
            ms: 500,
        };
        let w = ev.to_wire();
        assert_eq!(w.lane, "stt");
        assert_eq!(w.kind, "partial");
        assert_eq!(w.payload.get("text"), Some(&serde_json::json!("hello")));
    }

    #[test]
    fn relay_sink_publishes_through_relay() {
        let dir = std::env::temp_dir().join(format!(
            "speaches-plus-relay-sink-{}-{}",
            std::process::id(),
            uuid::Uuid::new_v4().simple()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let relay = Arc::new(InspectorRelay::new("sess_t".into(), Some(dir.clone())));
        let sink = RelayInspectorSink::new(relay.clone());
        sink.emit(InspectorEvent::VadConfirmedStart {
            session_id: "sess_t".into(),
            item_id: "item_a".into(),
            ms: 100,
        });
        let path = dir.join("sess_t.ndjson");
        let body = std::fs::read_to_string(&path).unwrap();
        assert!(body.contains("\"lane\":\"vad\""));
        assert!(body.contains("\"kind\":\"confirmed_start\""));
        let _ = std::fs::remove_dir_all(&dir);
    }
}
