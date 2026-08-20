use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Duration;

use serde_json::json;
use tokio::sync::{mpsc, Mutex};
use tokio::task::JoinHandle;
use tracing::{debug, error, info, warn};
use webrtc::data_channel::data_channel_message::DataChannelMessage;
use webrtc::data_channel::RTCDataChannel;
use webrtc::peer_connection::RTCPeerConnection;
use webrtc::track::track_remote::TrackRemote;

use crate::RealtimeQuery;

use super::audio_in::AudioIngest;
use super::audio_in_ws::WsAudioIngest;
use super::audio_out;
use super::eou_integrated::IntegratedVerdictAction;
use super::inspector::{self, FanoutSink, InspectorEvent, InspectorSink, RelayInspectorSink};
use super::session_update::{parse_session_update, StagedInstructions};
use super::state::{
    self, ConversationItem, ItemContent, ItemRole, ItemStatus, PendingBargein, RespPhase,
    ResponseRuntime, SessionPhase, SessionState, VadPhase,
};
use super::transport::{EventSink, OutboundAudioSpec};
use super::Intent;
use crate::conversation::llm::{ChatMessage, LlmConfig};
use crate::defaults;
use crate::eou::audio as eou_audio;
use crate::eou::onnx as eou_text;
use crate::eou::{
    EouConfig, EouKind, EouModel, FusionEouModel, HeuristicEouModel, IntegratedEouBackend,
    IntegratedVerdict, MissingAudioEouModel, StubEouModel,
};
use crate::errors::code as errcode;
use crate::ids::{self, IdSource};
use crate::inspect::{self as inspect_mod, AudioStore, InspectorRelay};
use crate::models::Models;
use crate::types::{ItemId, ResponseId, SessionId};
use crate::vad::{
    TurnDetectionRead, VadEvent, VadInfer, VadModel, VadProcessor, MIN_SPEECH_MS,
    SAMPLE_RATE as VAD_SAMPLE_RATE, VAD_FAILURE_THRESHOLD,
};

pub struct Session {
    pub id: SessionId,
    #[allow(dead_code)]
    pub query: RealtimeQuery,
    pub models: Arc<Models>,
    pub intent: Intent,
    pub outbound_audio: Option<OutboundAudioSpec>,
    pub llm_config: Option<LlmConfig>,
    pub id_source: Arc<dyn IdSource>,
    pub event_seq: Arc<super::wire::EventSeq>,
    pub inspector: Arc<dyn InspectorSink>,
    pub eou_config: EouConfig,
    pub eou_model: Option<Arc<dyn EouModel>>,
    #[allow(dead_code)]
    pub integrated_backend: Mutex<Option<Arc<dyn IntegratedEouBackend>>>,

    pub turn_detection: Arc<TurnDetectionConfig>,

    pub outbound_queue_cap_ms: u64,
    pub outbound_queue_cap_events: u32,
    pub outbound_inflight: Arc<AtomicU32>,

    pub(super) outbound_send: Mutex<()>,
    pub(super) tts_abort: super::pipeline::TtsAbort,
    pub(super) cancel: super::cancel::SessionCancel,
    pub session_max_duration_s: AtomicU64,
    pub min_speech_ms: AtomicU64,
    pub min_speech_for_response_ms: AtomicU64,

    pub no_speech_prob_threshold_bits: AtomicU32,
    pub avg_logprob_threshold_bits: AtomicU32,
    pub sealed_buffer_retention_count: AtomicU32,
    pub input_audio_format: std::sync::Mutex<String>,
    pub output_audio_format: std::sync::Mutex<String>,
    pub voice: std::sync::Mutex<Option<String>>,
    pub model: std::sync::Mutex<String>,
    pub transcription_model: std::sync::Mutex<Option<String>>,

    pub eou_runtime: std::sync::Mutex<EouRuntime>,
    pub relay: Arc<InspectorRelay>,
    pub audio_store: Arc<AudioStore>,
    pub last_eager_dispatch_at: Mutex<Option<std::time::Instant>>,

    pub diarizer: Mutex<Option<crate::diarization::Diarizer>>,
    pub(super) audio_in_tx: Mutex<Option<mpsc::Sender<AudioIn>>>,
    pub(super) ws_ingest: Mutex<Option<WsAudioIngest>>,
    pub(super) input_clear_epoch: AtomicU64,
    pub(super) state: Mutex<SessionState>,
}

#[derive(Debug)]
pub(crate) enum AudioIn {
    Samples(Vec<f32>),
    ForceCommit,
}

#[derive(Debug, PartialEq, Eq)]
pub(super) enum CommitAction {
    RejectEmpty,
    Force,
}

pub(super) fn commit_action(vad: &VadPhase) -> CommitAction {
    match vad {
        VadPhase::Silent => CommitAction::RejectEmpty,
        VadPhase::Speaking { .. } | VadPhase::Stopped { .. } => CommitAction::Force,
    }
}

pub(super) struct InflightGuard {
    counter: Arc<AtomicU32>,
}

impl InflightGuard {
    pub(super) fn try_acquire(counter: &Arc<AtomicU32>, cap: u32) -> Result<Self, u32> {
        let inflight = counter.fetch_add(1, Ordering::AcqRel) + 1;
        if inflight > cap {
            counter.fetch_sub(1, Ordering::AcqRel);
            return Err(inflight);
        }
        Ok(Self {
            counter: counter.clone(),
        })
    }
}

impl Drop for InflightGuard {
    fn drop(&mut self) {
        self.counter.fetch_sub(1, Ordering::AcqRel);
    }
}

#[derive(Clone, Debug)]
pub struct EouRuntime {
    pub kind: crate::eou::EouKind,
    pub p_threshold: f32,
    pub curve_k: f32,
    pub min_delay_ms: u32,
    pub max_delay_ms: u32,
    pub hard_cap_ms: u32,
    pub inference_timeout_ms: u32,
    pub context_turns: u32,
    pub failure_p_default: f32,
    pub failure_delay_max: bool,
    pub fusion_rule: crate::eou::FusionRule,
    pub fusion_weight_text: f32,
}

impl EouRuntime {
    fn from_config(cfg: &crate::eou::EouConfig) -> Self {
        Self {
            kind: cfg.kind,
            p_threshold: cfg.p_threshold,
            curve_k: cfg.curve_k,
            min_delay_ms: cfg.min_delay_ms,
            max_delay_ms: cfg.max_delay_ms,
            hard_cap_ms: cfg.silence_hard_cap_ms,
            inference_timeout_ms: cfg.inference_timeout_ms,
            context_turns: cfg.context_turns,
            failure_p_default: cfg.failure_p_default,
            failure_delay_max: cfg.failure_delay_max,
            fusion_rule: cfg.fusion_rule,
            fusion_weight_text: cfg.fusion_weight_text,
        }
    }
}

pub struct TurnDetectionConfig {
    pub kind: std::sync::Mutex<TurnDetectionKind>,
    pub threshold: AtomicU32,

    pub neg_threshold_bits: AtomicU32,
    pub min_speech_duration_ms: AtomicU32,
    pub prefix_padding_ms: AtomicU32,
    pub silence_duration_ms: AtomicU32,
    pub barge_in_delay_ms: AtomicU32,
    pub create_response: AtomicBool,
}

const NEG_THRESHOLD_AUTO_SENTINEL: u32 = 0;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TurnDetectionKind {
    ServerVad,
    None,
}

impl TurnDetectionKind {
    pub fn as_str(self) -> &'static str {
        match self {
            TurnDetectionKind::ServerVad => defaults::turn_detection_type::SERVER_VAD,
            TurnDetectionKind::None => defaults::turn_detection_type::NONE,
        }
    }

    pub fn parse(s: &str) -> Option<Self> {
        if s == defaults::turn_detection_type::SERVER_VAD {
            Some(TurnDetectionKind::ServerVad)
        } else if s == defaults::turn_detection_type::NONE {
            Some(TurnDetectionKind::None)
        } else {
            None
        }
    }
}

impl TurnDetectionRead for TurnDetectionConfig {
    fn threshold(&self) -> f32 {
        self.threshold()
    }
    fn prefix_padding_samples(&self) -> usize {
        (self.prefix_padding_ms.load(Ordering::Relaxed) as usize) * VAD_SAMPLE_RATE / 1000
    }
    fn silence_duration_samples(&self) -> usize {
        (self.silence_duration_ms.load(Ordering::Relaxed) as usize) * VAD_SAMPLE_RATE / 1000
    }
    fn neg_threshold(&self) -> f32 {
        self.neg_threshold()
    }
    fn min_speech_duration_ms(&self) -> u32 {
        self.min_speech_duration_ms.load(Ordering::Relaxed)
    }
}

impl TurnDetectionConfig {
    pub fn from_env() -> Self {
        let barge_in_delay_ms = std::env::var(defaults::env::BARGE_IN_DELAY_MS)
            .ok()
            .and_then(|s| s.parse::<u32>().ok())
            .map(|v| v.min(defaults::turn_detection::BARGE_IN_DELAY_MS_MAX))
            .unwrap_or(defaults::turn_detection::BARGE_IN_DELAY_MS);
        Self {
            kind: std::sync::Mutex::new(TurnDetectionKind::ServerVad),
            threshold: AtomicU32::new(defaults::turn_detection::THRESHOLD.to_bits()),
            neg_threshold_bits: AtomicU32::new(NEG_THRESHOLD_AUTO_SENTINEL),
            min_speech_duration_ms: AtomicU32::new(defaults::vad_window::MIN_SPEECH_DURATION_MS),
            prefix_padding_ms: AtomicU32::new(defaults::turn_detection::PREFIX_PADDING_MS),
            silence_duration_ms: AtomicU32::new(defaults::turn_detection::SILENCE_DURATION_MS),
            barge_in_delay_ms: AtomicU32::new(barge_in_delay_ms),
            create_response: AtomicBool::new(defaults::turn_detection::CREATE_RESPONSE),
        }
    }

    pub fn barge_in_delay_ms(&self) -> u32 {
        self.barge_in_delay_ms.load(Ordering::Relaxed)
    }

    pub fn threshold(&self) -> f32 {
        f32::from_bits(self.threshold.load(Ordering::Relaxed))
    }

    pub fn neg_threshold(&self) -> f32 {
        let bits = self.neg_threshold_bits.load(Ordering::Relaxed);
        if bits == NEG_THRESHOLD_AUTO_SENTINEL {
            (self.threshold() - defaults::vad_window::NEG_THRESHOLD_DELTA)
                .max(defaults::vad_window::NEG_THRESHOLD_FLOOR)
        } else {
            f32::from_bits(bits)
        }
    }

    pub fn snapshot(&self) -> serde_json::Value {
        let kind = *self.kind.lock().expect("turn_detection kind poisoned");
        json!({
            "type": kind.as_str(),
            "threshold": self.threshold(),
            "neg_threshold": self.neg_threshold(),
            "min_speech_duration_ms": self.min_speech_duration_ms.load(Ordering::Relaxed),
            "prefix_padding_ms": self.prefix_padding_ms.load(Ordering::Relaxed),
            "silence_duration_ms": self.silence_duration_ms.load(Ordering::Relaxed),
            "barge_in_delay_ms": self.barge_in_delay_ms(),
            "create_response": self.create_response.load(Ordering::Relaxed),
        })
    }
}

pub(crate) fn log_audio_eou_missing(kind: EouKind) {
    static ONCE: std::sync::Once = std::sync::Once::new();
    ONCE.call_once(|| {
        error!(
            eou_kind = kind.as_str(),
            "{}",
            crate::eou::audio_eou_missing_message(
                kind,
                crate::eou::audio::resolve_audio_eou_paths().as_deref(),
                true,
            )
        );
    });
}

impl Session {
    pub fn new(
        query: RealtimeQuery,
        models: Arc<Models>,
        intent: Intent,
        outbound_audio: Option<OutboundAudioSpec>,
    ) -> Self {
        Self::with_dependencies(
            query,
            models,
            intent,
            outbound_audio,
            ids::default_source(),
            inspector::default_sink(),
        )
    }

    pub fn with_dependencies(
        query: RealtimeQuery,
        models: Arc<Models>,
        intent: Intent,
        outbound_audio: Option<OutboundAudioSpec>,
        id_source: Arc<dyn IdSource>,
        base_inspector: Arc<dyn InspectorSink>,
    ) -> Self {
        let _session_span = tracing::info_span!(
            target: "speaches/realtime",
            "session",
            intent = match intent {
                Intent::Conversation => "conversation",
                Intent::Transcription => "transcription",
            },
        )
        .entered();
        let llm_config = if intent == Intent::Conversation {
            LlmConfig::from_env()
        } else {
            None
        };
        let eou_config = EouConfig::from_env();
        let eou_model: Option<Arc<dyn EouModel>> = if eou_config.kind.calls_classifier() {
            let text_model: Arc<dyn EouModel> = match eou_text::shared_text_eou_model() {
                Some(m) => m,
                None => Arc::new(HeuristicEouModel),
            };
            let audio_model: Arc<dyn EouModel> = match eou_audio::shared_audio_eou_model(
                eou_config.audio_window_ms,
                eou_config.audio_pad_alignment,
            ) {
                Some(m) => m,
                None => {
                    if crate::eou::audio_eou_wanted(eou_config.kind) {
                        log_audio_eou_missing(eou_config.kind);
                    }
                    Arc::new(MissingAudioEouModel)
                }
            };
            match eou_config.kind {
                EouKind::Heuristic => Some(Arc::new(HeuristicEouModel)),
                EouKind::Text => Some(text_model),
                EouKind::Audio => Some(audio_model),
                EouKind::Fusion => Some(Arc::new(FusionEouModel::new(
                    text_model,
                    audio_model,
                    eou_config.fusion_rule,
                    eou_config.fusion_weight_text,
                ))),
                _ => Some(Arc::new(StubEouModel)),
            }
        } else {
            None
        };
        let id = id_source.session();
        let turn_detection = Arc::new(TurnDetectionConfig::from_env());
        let outbound_queue_cap_ms = audio_out::read_queue_cap_ms_from_env();
        let outbound_queue_cap_events = audio_out::read_queue_cap_events_from_env();
        let outbound_inflight = Arc::new(AtomicU32::new(0));
        let session_max_duration_s = AtomicU64::new(read_session_max_duration_s_from_env());

        inspect_mod::run_startup_cleanup();
        let session_dir = inspect_mod::session_dir();
        let relay = Arc::new(InspectorRelay::new(
            id.as_str().to_string(),
            session_dir.clone(),
        ));
        let audio_store = Arc::new(AudioStore::new(id.as_str().to_string(), session_dir));
        let model_label = match intent {
            Intent::Conversation => "conversation".to_string(),
            Intent::Transcription => "transcription".to_string(),
        };
        let relay_for_registry = relay.clone();
        inspect_mod::register(id.as_str(), relay_for_registry, model_label, || {
            "active".into()
        });

        let relay_sink: Arc<dyn InspectorSink> = Arc::new(RelayInspectorSink::new(relay.clone()));
        let inspector: Arc<dyn InspectorSink> =
            Arc::new(FanoutSink::new(vec![base_inspector, relay_sink]));

        let diarizer = if !super::diarization::realtime_enabled() {
            None
        } else {
            match (
                models.diar_segmentation.clone(),
                models.diar_embedding.clone(),
            ) {
                (Some(seg), Some(emb)) => Some(crate::diarization::Diarizer::new(
                    seg,
                    emb,
                    crate::diarization::DiarConfig::default(),
                )),
                _ => None,
            }
        };

        let eou_runtime_init = EouRuntime::from_config(&eou_config);
        let model_label = query.model.clone().unwrap_or_default();
        let transcription_model_label = query.transcription_model.clone();
        let voice_label = query.voice.clone();
        Self {
            id,
            query,
            models,
            intent,
            outbound_audio,
            llm_config,
            id_source,
            event_seq: Arc::new(super::wire::EventSeq::new()),
            inspector,
            eou_config,
            eou_model,
            integrated_backend: Mutex::new(None),
            turn_detection,
            outbound_queue_cap_ms,
            outbound_queue_cap_events,
            outbound_inflight,
            outbound_send: Mutex::new(()),
            tts_abort: super::pipeline::TtsAbort::new(),
            cancel: super::cancel::SessionCancel::new(),
            session_max_duration_s,
            min_speech_ms: AtomicU64::new(defaults::buffer::MIN_SPEECH_MS),
            min_speech_for_response_ms: AtomicU64::new(
                defaults::buffer::MIN_SPEECH_FOR_RESPONSE_MS,
            ),

            no_speech_prob_threshold_bits: AtomicU32::new(f32::NAN.to_bits()),
            avg_logprob_threshold_bits: AtomicU32::new(f32::NAN.to_bits()),
            sealed_buffer_retention_count: AtomicU32::new(
                defaults::buffer::SEALED_BUFFER_RETENTION_COUNT as u32,
            ),
            input_audio_format: std::sync::Mutex::new(defaults::audio_format::DEFAULT.to_string()),
            output_audio_format: std::sync::Mutex::new(defaults::audio_format::DEFAULT.to_string()),
            voice: std::sync::Mutex::new(voice_label),
            model: std::sync::Mutex::new(model_label),
            transcription_model: std::sync::Mutex::new(transcription_model_label),
            eou_runtime: std::sync::Mutex::new(eou_runtime_init),
            relay,
            audio_store,
            last_eager_dispatch_at: Mutex::new(None),
            diarizer: Mutex::new(diarizer),
            audio_in_tx: Mutex::new(None),
            ws_ingest: Mutex::new(None),
            input_clear_epoch: AtomicU64::new(0),
            state: Mutex::new(SessionState::default()),
        }
    }

    pub async fn attach_peer_connection(self: &Arc<Self>, pc: Arc<RTCPeerConnection>) {
        let mut state = self.state.lock().await;
        state.pc = Some(pc);
        check_or_react(self, &state);
    }

    pub async fn transition_to_terminated(self: &Arc<Self>) {
        self.transition_to_terminated_with(state::TerminationReason::ClientClosed)
            .await;
    }

    pub async fn cancel_session_lanes(&self) {
        if self.cancel.is_cancelled() {
            return;
        }
        let inflight = self.cancel.lanes_inflight();
        self.cancel.cancel().await;
        debug!(
            session_id = %self.id,
            lanes = inflight,
            "session cancellation fired; long-lived lanes torn down",
        );
    }

    pub async fn transition_to_terminated_with(self: &Arc<Self>, reason: state::TerminationReason) {
        self.cancel_session_lanes().await;
        let mut state = self.state.lock().await;
        if !matches!(state.session, SessionPhase::Terminated { .. }) {
            state.session = SessionPhase::Terminated { reason };
        }
        check_or_react(self, &state);
        drop(state);

        let mut tx_guard = self.audio_in_tx.lock().await;
        *tx_guard = None;
        drop(tx_guard);
        let mut ws_guard = self.ws_ingest.lock().await;
        *ws_guard = None;
        drop(ws_guard);
    }

    pub async fn peer_connection(&self) -> Option<Arc<RTCPeerConnection>> {
        self.state.lock().await.pc.clone()
    }

    pub async fn set_timeout_task(&self, handle: JoinHandle<()>) {
        let mut state = self.state.lock().await;
        if let Some(prev) = state.timeout_task.replace(handle) {
            prev.abort();
        }
    }

    pub async fn abort_timeout_task(&self) {
        let prev = self.state.lock().await.timeout_task.take();
        if let Some(h) = prev {
            h.abort();
        }
    }

    pub fn emit_wire_inspector(&self, direction: &'static str, event_type: &str, bytes: usize) {
        let mut payload = std::collections::BTreeMap::new();
        payload.insert("event_type".into(), serde_json::json!(event_type));
        payload.insert("bytes".into(), serde_json::json!(bytes));
        self.relay.publish("wire", direction, None, payload);
    }

    pub(super) fn stamp_event_id(&self, value: &mut serde_json::Value) {
        self.event_seq.stamp(value);
    }

    pub(super) async fn deliver_to_sink(&self, sink: &EventSink, mut value: serde_json::Value) {
        let _ordered = self.outbound_send.lock().await;
        self.stamp_event_id(&mut value);
        let typ = value
            .get("type")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();
        let bytes = serde_json::to_vec(&value).map(|b| b.len()).unwrap_or(0);
        self.emit_wire_inspector("out", &typ, bytes);
        sink.send_value(&value).await;
    }

    pub async fn send_to_client(&self, value: &serde_json::Value) {
        let Some(sink) = self.event_sink().await else {
            return;
        };
        self.deliver_to_sink(&sink, value.clone()).await;
    }

    pub async fn emit_session_done(&self, reason: &str) {
        let event = json!({
            "type": "session.done",
            "reason": reason,
        });
        self.send_to_client(&event).await;
    }

    pub async fn spawn_max_duration_timeout(self: &Arc<Self>, duration: Duration) {
        let session = self.clone();
        let id_for_log = session.id.as_str().to_string();
        let handle = tokio::spawn(async move {
            tokio::time::sleep(duration).await;
            let Some(sess) = super::lookup_session_pub(&id_for_log) else {
                return;
            };
            info!(session_id = %id_for_log, secs = duration.as_secs(), "session hard timeout reached");
            sess.emit_session_done("max_duration").await;
            sess.transition_to_terminated_with(state::TerminationReason::MaxDuration)
                .await;
            if let Some(pc) = sess.peer_connection().await {
                if let Err(err) = pc.close().await {
                    warn!(error = %err, session_id = %id_for_log, "pc.close failed on timeout");
                }
            }
            super::drop_session(&id_for_log);
        });
        self.set_timeout_task(handle).await;
    }

    pub async fn reschedule_max_duration_timeout(self: &Arc<Self>) {
        let new_secs = self.session_max_duration_s.load(Ordering::Relaxed);
        let created_at_ms = match self.state.lock().await.session {
            SessionPhase::Active { created_at_ms } => created_at_ms.raw(),
            _ => 0,
        };
        let now_ms = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_millis() as u64)
            .unwrap_or(created_at_ms);
        let remaining_ms = compute_remaining_timeout_ms(new_secs, created_at_ms, now_ms);
        self.spawn_max_duration_timeout(Duration::from_millis(remaining_ms))
            .await;
    }

    pub async fn attach_data_channel(self: &Arc<Self>, dc: Arc<RTCDataChannel>) {
        let label = dc.label().to_string();
        let dc_open = dc.clone();
        let session_open = self.clone();
        let label_open = label.clone();
        dc.on_open(Box::new(move || {
            let label = label_open;
            let dc = dc_open;
            let session = session_open;
            Box::pin(async move {
                info!(label = %label, "data channel open");
                {
                    let mut state = session.state.lock().await;
                    if matches!(state.session, SessionPhase::Pending) {
                        let now_ms = std::time::SystemTime::now()
                            .duration_since(std::time::UNIX_EPOCH)
                            .map(|d| d.as_millis() as u64)
                            .unwrap_or(0);
                        state.session = SessionPhase::Active {
                            created_at_ms: crate::types::Millis(now_ms),
                        };
                    }
                    check_or_react(&session, &state);
                }
                let view = session.current_session_view().await;
                let event = json!({
                    "type": "session.created",
                    "session": view,
                });
                session
                    .deliver_to_sink(&EventSink::DataChannel(dc), event)
                    .await;
            })
        }));
        let label_msg = label.clone();
        let session_msg = self.clone();
        dc.on_message(Box::new(move |msg: DataChannelMessage| {
            let label = label_msg.clone();
            let session = session_msg.clone();
            Box::pin(async move {
                if let Err(err) = session.handle_client_event(&label, msg.data).await {
                    warn!(error = %err, label, "data channel message handling failed");
                }
            })
        }));
        let mut state = self.state.lock().await;
        state.event_sink = Some(EventSink::DataChannel(dc));
        check_or_react(self, &state);
    }

    pub async fn attach_websocket(self: &Arc<Self>, ws_send: tokio::sync::mpsc::Sender<String>) {
        {
            let mut state = self.state.lock().await;
            if matches!(state.session, SessionPhase::Pending) {
                let now_ms = std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .map(|d| d.as_millis() as u64)
                    .unwrap_or(0);
                state.session = SessionPhase::Active {
                    created_at_ms: crate::types::Millis(now_ms),
                };
            }
        }
        let view = self.current_session_view().await;
        let event = json!({
            "type": "session.created",
            "session": view,
        });
        self.deliver_to_sink(&EventSink::WebSocket(ws_send.clone()), event)
            .await;
        let mut state = self.state.lock().await;
        state.event_sink = Some(EventSink::WebSocket(ws_send));
        check_or_react(self, &state);
    }

    pub(crate) async fn handle_client_event(
        self: &Arc<Self>,
        label: &str,
        data: bytes::Bytes,
    ) -> anyhow::Result<()> {
        let inbound_bytes = data.len();
        let value: serde_json::Value = match serde_json::from_slice(&data) {
            Ok(v) => v,
            Err(err) => {
                debug!(label, bytes = inbound_bytes, error = %err, "client event non-JSON message");
                self.emit_wire_inspector("in", "", inbound_bytes);
                super::events::emit_error(
                    self,
                    errcode::INVALID_REQUEST_ERROR,
                    &format!("malformed JSON: {err}"),
                    None,
                    None,
                )
                .await;
                return Ok(());
            }
        };
        let event_id = value
            .get("event_id")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string());
        let event_type = value.get("type").and_then(|v| v.as_str()).unwrap_or("");
        self.emit_wire_inspector("in", event_type, inbound_bytes);
        let phase = self.state.lock().await.session.clone();
        match phase {
            SessionPhase::Active { .. } => {}
            SessionPhase::Pending => {
                super::events::emit_error(
                    self,
                    errcode::SESSION_NOT_ACTIVE,
                    &format!("{event_type}: received before session.created"),
                    event_id.as_deref(),
                    Some("type"),
                )
                .await;
                return Ok(());
            }
            SessionPhase::Terminated { reason } => {
                debug!(
                    label,
                    event_type,
                    reason = reason.as_str(),
                    "inbound event after termination; dropped",
                );
                return Ok(());
            }
        }
        match event_type {
            "session.update" => self.handle_session_update(value, event_id.as_deref()).await,

            "input_audio_buffer.append" => {
                self.ingest_pcm16_b64(value, event_id.as_deref()).await;
            }
            "input_audio_buffer.commit" => {
                self.handle_buffer_commit(label, event_id.as_deref()).await
            }
            "input_audio_buffer.clear" => self.handle_buffer_clear(event_id.as_deref()).await,
            "response.cancel" => self.handle_response_cancel(event_id.as_deref()).await,
            "conversation.item.create" => {
                self.handle_conversation_item_create(value, event_id.as_deref())
                    .await
            }
            "conversation.item.delete" => {
                self.handle_conversation_item_delete(value, event_id.as_deref())
                    .await
            }
            "conversation.item.truncate" => {
                self.handle_conversation_item_truncate(value, event_id.as_deref())
                    .await
            }
            "conversation.item.retrieve" => {
                super::events::emit_error(
                    self,
                    errcode::INVALID_REQUEST_ERROR,
                    "conversation.item.retrieve is not yet implemented",
                    event_id.as_deref(),
                    None,
                )
                .await;
            }
            "response.create" => {
                self.handle_response_create(value, event_id.as_deref())
                    .await
            }
            "" => {
                super::events::emit_error(
                    self,
                    errcode::INVALID_REQUEST_ERROR,
                    "missing 'type' field",
                    event_id.as_deref(),
                    Some("type"),
                )
                .await;
            }
            other if super::v2_compat::is_known_v2_noop_event(other) => {
                debug!(label, event_type = other, "data channel inbound: v2 noop");
            }
            other => {
                super::events::emit_error(
                    self,
                    errcode::UNKNOWN_EVENT_TYPE,
                    &format!("unknown event type: {other}"),
                    event_id.as_deref(),
                    Some("type"),
                )
                .await;
                debug!(label, event_type = other, "data channel inbound: unknown");
            }
        }
        Ok(())
    }

    async fn handle_session_update(
        self: &Arc<Self>,
        value: serde_json::Value,
        event_id: Option<&str>,
    ) {
        let session_obj = match value.get("session") {
            Some(serde_json::Value::Object(_)) => value.get("session").unwrap(),
            _ => {
                super::events::emit_error(
                    self,
                    errcode::SESSION_UPDATE_INVALID,
                    "missing 'session' object",
                    event_id,
                    Some("session"),
                )
                .await;
                return;
            }
        };

        let caps = super::session_update::EouCapability {
            kind: self.eou_config.kind,
            fusion_rule: self.eou_config.fusion_rule,
        };
        let staged = match parse_session_update(session_obj, caps) {
            Ok(s) => s,
            Err(err) => {
                super::events::emit_error(self, err.code, &err.message, event_id, Some(&err.param))
                    .await;
                return;
            }
        };

        if let Some(instr) = staged.instructions {
            let mut state = self.state.lock().await;
            match instr {
                StagedInstructions::Set(s) => state.instructions = Some(s),
                StagedInstructions::Clear => state.instructions = None,
            }
            check_or_react(self, &state);
        }
        if let Some(parsed) = staged.turn_detection {
            let prev_kind = *self
                .turn_detection
                .kind
                .lock()
                .expect("turn_detection kind poisoned");
            self.commit_turn_detection_update(&parsed);
            if matches!(parsed.kind, Some(TurnDetectionKind::None))
                && !matches!(prev_kind, TurnDetectionKind::None)
            {
                self.reconcile_turn_detection_none().await;
            }
        }
        if let Some(secs) = staged.session_max_duration_s {
            self.session_max_duration_s.store(secs, Ordering::Relaxed);
            self.reschedule_max_duration_timeout().await;
        }
        if let Some(v) = staged.voice {
            *self.voice.lock().expect("voice poisoned") = v;
        }
        if let Some(n) = staged.min_speech_ms {
            self.min_speech_ms.store(n, Ordering::Relaxed);
        }
        if let Some(n) = staged.min_speech_for_response_ms {
            self.min_speech_for_response_ms.store(n, Ordering::Relaxed);
        }
        if let Some(opt) = staged.no_speech_prob_threshold {
            let bits = match opt {
                Some(f) => f.to_bits(),
                None => f32::NAN.to_bits(),
            };
            self.no_speech_prob_threshold_bits
                .store(bits, Ordering::Relaxed);
        }
        if let Some(opt) = staged.avg_logprob_threshold {
            let bits = match opt {
                Some(f) => f.to_bits(),
                None => f32::NAN.to_bits(),
            };
            self.avg_logprob_threshold_bits
                .store(bits, Ordering::Relaxed);
        }
        if let Some(n) = staged.sealed_buffer_retention_count {
            self.sealed_buffer_retention_count
                .store(n, Ordering::Relaxed);
        }
        if let Some(s) = staged.input_audio_format {
            *self
                .input_audio_format
                .lock()
                .expect("input_audio_format poisoned") = s;
        }
        if let Some(s) = staged.output_audio_format {
            *self
                .output_audio_format
                .lock()
                .expect("output_audio_format poisoned") = s;
        }
        let view = self.current_session_view().await;
        self.send_to_client(&json!({
            "type": "session.updated",
            "session": view,
        }))
        .await;
        info!("session.update applied");
    }

    pub fn noise_gate_thresholds(&self) -> crate::stt::noise_gate::GateThresholds {
        let nsp = f32::from_bits(self.no_speech_prob_threshold_bits.load(Ordering::Relaxed));
        let lp = f32::from_bits(self.avg_logprob_threshold_bits.load(Ordering::Relaxed));
        crate::stt::noise_gate::GateThresholds {
            no_speech_prob_threshold: if nsp.is_nan() { None } else { Some(nsp) },
            avg_logprob_threshold: if lp.is_nan() { None } else { Some(lp) },
        }
    }

    pub async fn current_session_view(&self) -> serde_json::Value {
        let instructions = self
            .state
            .lock()
            .await
            .instructions
            .clone()
            .map(serde_json::Value::String)
            .unwrap_or(serde_json::Value::Null);
        let model = self.model.lock().expect("model poisoned").clone();
        let voice = self.voice.lock().expect("voice poisoned").clone();
        let in_fmt = self
            .input_audio_format
            .lock()
            .expect("input_audio_format poisoned")
            .clone();
        let out_fmt = self
            .output_audio_format
            .lock()
            .expect("output_audio_format poisoned")
            .clone();
        let transcription_model = self
            .transcription_model
            .lock()
            .expect("transcription_model poisoned")
            .clone();

        let modalities: &[&str] = match self.intent {
            super::Intent::Conversation => defaults::modality::DEFAULT_PAIR,
            super::Intent::Transcription => defaults::modality::TEXT_ONLY,
        };
        let mut sess = json!({
            "id": self.id,
            "object": defaults::session_object::REALTIME_SESSION,
            "model": model,
            "modalities": modalities,
            "input_audio_format": in_fmt,
            "output_audio_format": out_fmt,
            "instructions": instructions,
            "turn_detection": self.turn_detection.snapshot(),
            "session_max_duration_s": self.session_max_duration_s.load(Ordering::Relaxed),
            "min_speech_ms": self.min_speech_ms.load(Ordering::Relaxed),
            "min_speech_for_response_ms": self.min_speech_for_response_ms.load(Ordering::Relaxed),
            "sealed_buffer_retention_count": self.sealed_buffer_retention_count.load(Ordering::Relaxed),
        });
        if let Some(obj) = sess.as_object_mut() {
            if let Some(v) = voice {
                obj.insert("voice".into(), serde_json::Value::String(v));
            }
            if let Some(m) = transcription_model {
                obj.insert("input_audio_transcription".into(), json!({ "model": m }));
            }
        }

        super::v2_compat::enrich_session_view(&mut sess);
        sess
    }

    async fn reconcile_turn_detection_none(self: &Arc<Self>) {
        let cancel_commit = {
            let mut state = self.state.lock().await;
            let speaking_snap = match &state.vad {
                VadPhase::Speaking {
                    item_id,
                    audio_start_ms,
                } => Some((item_id.clone(), *audio_start_ms)),
                _ => None,
            };
            if let Some((item_id, audio_start_ms)) = speaking_snap {
                let now_ms = std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .map(|d| d.as_millis() as u64)
                    .unwrap_or(audio_start_ms.raw());
                let end = crate::types::Millis(now_ms.max(audio_start_ms.raw()));
                state.vad = VadPhase::Stopped {
                    item_id,
                    audio_start_ms,
                    audio_end_ms: end,
                };
                check_or_react(self, &state);
            }
            state.commit_timer.take()
        };
        if let Some(timer) = cancel_commit {
            timer.abort();
        }
    }

    fn commit_turn_detection_update(&self, parsed: &super::session_update::StagedTurnDetection) {
        if let Some(kind) = parsed.kind {
            *self
                .turn_detection
                .kind
                .lock()
                .expect("turn_detection kind poisoned") = kind;
        }
        if let Some(f) = parsed.threshold {
            self.turn_detection
                .threshold
                .store(f.to_bits(), Ordering::Relaxed);
        }
        if let Some(neg) = parsed.neg_threshold {
            let bits = match neg {
                Some(f) => {
                    let b = f.to_bits();
                    if b == NEG_THRESHOLD_AUTO_SENTINEL {
                        f32::EPSILON.to_bits()
                    } else {
                        b
                    }
                }
                None => NEG_THRESHOLD_AUTO_SENTINEL,
            };
            self.turn_detection
                .neg_threshold_bits
                .store(bits, Ordering::Relaxed);
        }
        if let Some(n) = parsed.min_speech_duration_ms {
            self.turn_detection
                .min_speech_duration_ms
                .store(n, Ordering::Relaxed);
        }
        if let Some(n) = parsed.prefix_padding_ms {
            self.turn_detection
                .prefix_padding_ms
                .store(n, Ordering::Relaxed);
        }
        if let Some(n) = parsed.silence_duration_ms {
            self.turn_detection
                .silence_duration_ms
                .store(n, Ordering::Relaxed);
        }
        if let Some(n) = parsed.barge_in_delay_ms {
            self.turn_detection
                .barge_in_delay_ms
                .store(n, Ordering::Relaxed);
        }
        if let Some(b) = parsed.create_response {
            self.turn_detection
                .create_response
                .store(b, Ordering::Relaxed);
        }
        if let Some(eou) = &parsed.eou {
            let mut rt = self.eou_runtime.lock().expect("eou_runtime poisoned");
            if let Some(k) = eou.kind {
                rt.kind = k;
            }
            if let Some(f) = eou.p_threshold {
                rt.p_threshold = f;
            }
            if let Some(f) = eou.curve_k {
                rt.curve_k = f;
            }
            if let Some(n) = eou.min_delay_ms {
                rt.min_delay_ms = n;
            }
            if let Some(n) = eou.max_delay_ms {
                rt.max_delay_ms = n;
            }
            if let Some(n) = eou.silence_hard_cap_ms {
                rt.hard_cap_ms = n;
            }
            if let Some(n) = eou.inference_timeout_ms {
                rt.inference_timeout_ms = n;
            }
            if let Some(n) = eou.context_turns {
                rt.context_turns = n;
            }
            if let Some(f) = eou.failure_p_default {
                rt.failure_p_default = f;
            }
            if let Some(b) = eou.failure_delay_max {
                rt.failure_delay_max = b;
            }
            if let Some(r) = eou.fusion_rule {
                rt.fusion_rule = r;
            }
            if let Some(f) = eou.fusion_weight_text {
                rt.fusion_weight_text = f;
            }
        }
    }

    async fn handle_buffer_commit(self: &Arc<Self>, label: &str, event_id: Option<&str>) {
        let action = commit_action(&self.state.lock().await.vad);
        if action == CommitAction::RejectEmpty {
            super::events::emit_error(
                self,
                errcode::INPUT_AUDIO_BUFFER_COMMIT_EMPTY,
                &format!(
                    "buffer below min_speech_ms ({}ms): nothing to commit",
                    self.min_speech_ms.load(Ordering::Relaxed)
                ),
                event_id,
                None,
            )
            .await;
            return;
        }
        let tx = self.audio_in_tx.lock().await.clone();
        let Some(tx) = tx else {
            super::events::emit_error(
                self,
                errcode::INTERNAL_STATE_ERROR,
                "audio pipeline is not running; cannot commit the input buffer",
                event_id,
                None,
            )
            .await;
            return;
        };
        if tx.send(AudioIn::ForceCommit).await.is_err() {
            super::events::emit_error(
                self,
                errcode::INTERNAL_STATE_ERROR,
                "audio pipeline stopped; cannot commit the input buffer",
                event_id,
                None,
            )
            .await;
            return;
        }
        debug!(label, "input_audio_buffer.commit: forcing commit");
    }

    async fn handle_buffer_clear(self: &Arc<Self>, _event_id: Option<&str>) {
        self.clear_input_audio_buffer().await;
        self.send_to_client(&json!({"type": "input_audio_buffer.cleared"}))
            .await;
    }

    pub(super) async fn clear_input_audio_buffer(self: &Arc<Self>) {
        self.input_clear_epoch.fetch_add(1, Ordering::Release);
        let cancel_commit = {
            let mut state = self.state.lock().await;
            if !matches!(state.vad, VadPhase::Silent) {
                state.vad = VadPhase::Silent;
                check_or_react(self, &state);
            }
            state.commit_timer.take()
        };
        if let Some(timer) = cancel_commit {
            timer.abort();
        }
        self.set_current_speech_item(None).await;
    }

    async fn handle_response_cancel(self: &Arc<Self>, event_id: Option<&str>) {
        let was_predicted = matches!(self.state.lock().await.resp, RespPhase::Predicted { .. });
        if was_predicted {
            self.rollback_predicted_if_any("cancel_event").await;
            info!("response.cancel honored on Predicted (no wire event per I7)");
            return;
        }
        match self.cancel_current_response().await {
            Some(snap) => {
                if snap.wire_opened {
                    super::events::emit_cancelled_brackets(
                        self,
                        &snap,
                        CancelReason::ClientCancelled,
                    )
                    .await;
                } else {
                    info!(
                        cancelled_id = %snap.response_id,
                        "response.cancel: suppressed close cascade for never-opened response (W1/W2)",
                    );
                }
                self.apply_truncate_to_assistant_item(&snap).await;
                super::events::emit_server_truncate(self, &snap).await;
                info!(
                    cancelled_id = %snap.response_id,
                    played_ms = snap.played_ms,
                    "response.cancel honored",
                );
            }
            None => {
                super::events::emit_error(
                    self,
                    errcode::RESPONSE_CANCEL_NOT_ACTIVE,
                    "no response is currently active",
                    event_id,
                    None,
                )
                .await;
            }
        }
    }

    async fn handle_conversation_item_create(
        self: &Arc<Self>,
        value: serde_json::Value,
        event_id: Option<&str>,
    ) {
        let item_obj = match value.get("item") {
            Some(serde_json::Value::Object(_)) => value.get("item").unwrap(),
            _ => {
                super::events::emit_error(
                    self,
                    errcode::INVALID_REQUEST_ERROR,
                    "missing 'item' object",
                    event_id,
                    Some("item"),
                )
                .await;
                return;
            }
        };
        let item_type = item_obj.get("type").and_then(|v| v.as_str()).unwrap_or("");
        if item_type != "message" {
            super::events::emit_error(
                self,
                errcode::INVALID_REQUEST_ERROR,
                "only 'message' items are supported",
                event_id,
                Some("item.type"),
            )
            .await;
            return;
        }
        let role_str = match item_obj.get("role").and_then(|v| v.as_str()) {
            Some(r) => r,
            None => {
                super::events::emit_error(
                    self,
                    errcode::INVALID_REQUEST_ERROR,
                    "missing 'role'",
                    event_id,
                    Some("item.role"),
                )
                .await;
                return;
            }
        };
        let role = match ItemRole::parse(role_str) {
            Some(r) => r,
            None => {
                super::events::emit_error(
                    self,
                    errcode::INVALID_REQUEST_ERROR,
                    "invalid 'role'",
                    event_id,
                    Some("item.role"),
                )
                .await;
                return;
            }
        };
        let content_text = super::events::extract_text_from_content(item_obj.get("content"));
        let id = item_obj
            .get("id")
            .and_then(|v| v.as_str())
            .map(|s| ItemId::new(s.to_string()))
            .unwrap_or_else(|| self.id_source.item());
        let text = content_text.unwrap_or_default();
        let mut item =
            ConversationItem::new_text(id.as_str().to_string(), role, ItemStatus::Completed, text);
        item.client_speakable = matches!(role, ItemRole::Assistant)
            && !item.transcript().unwrap_or("").trim().is_empty();
        let echo = json!({
            "type": "conversation.item.added",
            "item": super::events::item_to_json(&item),
        });
        {
            let mut state = self.state.lock().await;
            state.conversation.push(item);
        }
        self.send_to_client(&echo).await;
    }

    async fn handle_conversation_item_delete(
        self: &Arc<Self>,
        value: serde_json::Value,
        event_id: Option<&str>,
    ) {
        let item_id = match value.get("item_id").and_then(|v| v.as_str()) {
            Some(s) => s.to_string(),
            None => {
                super::events::emit_error(
                    self,
                    errcode::INVALID_REQUEST_ERROR,
                    "missing 'item_id'",
                    event_id,
                    Some("item_id"),
                )
                .await;
                return;
            }
        };
        let removed = {
            let mut state = self.state.lock().await;
            if let Some(idx) = state.conversation.iter().position(|i| i.id == item_id) {
                state.conversation.remove(idx);
                true
            } else {
                false
            }
        };
        if !removed {
            super::events::emit_error(
                self,
                errcode::INVALID_REQUEST_ERROR,
                "no conversation item with that id",
                event_id,
                Some("item_id"),
            )
            .await;
            return;
        }
        self.send_to_client(&json!({
            "type": "conversation.item.deleted",
            "item_id": item_id,
        }))
        .await;
    }

    async fn handle_conversation_item_truncate(
        self: &Arc<Self>,
        value: serde_json::Value,
        event_id: Option<&str>,
    ) {
        let item_id = match value.get("item_id").and_then(|v| v.as_str()) {
            Some(s) => s.to_string(),
            None => {
                super::events::emit_error(
                    self,
                    errcode::INVALID_REQUEST_ERROR,
                    "missing 'item_id'",
                    event_id,
                    Some("item_id"),
                )
                .await;
                return;
            }
        };
        let requested_audio_end_ms = match value.get("audio_end_ms").and_then(|v| v.as_u64()) {
            Some(v) => v,
            None => {
                super::events::emit_error(
                    self,
                    errcode::INVALID_REQUEST_ERROR,
                    "missing or non-numeric 'audio_end_ms'",
                    event_id,
                    Some("audio_end_ms"),
                )
                .await;
                return;
            }
        };
        let content_index = value
            .get("content_index")
            .and_then(|v| v.as_u64())
            .unwrap_or(0);
        let outcome = {
            let mut state = self.state.lock().await;
            match state.conversation.iter_mut().find(|i| i.id == item_id) {
                None => Err("not_found"),
                Some(item) => match (&mut item.content, item.role) {
                    (
                        ItemContent::AssistantAudio {
                            audio_ms,
                            transcript,
                        },
                        ItemRole::Assistant,
                    ) => {
                        let clamped = requested_audio_end_ms.min(*audio_ms);
                        *audio_ms = clamped;
                        Ok((clamped, transcript.clone()))
                    }
                    _ => Err("not_assistant_audio"),
                },
            }
        };
        match outcome {
            Err("not_found") => {
                super::events::emit_error(
                    self,
                    errcode::INVALID_REQUEST_ERROR,
                    "no conversation item with that id",
                    event_id,
                    Some("item_id"),
                )
                .await;
            }
            Err(_) => {
                super::events::emit_error(
                    self,
                    errcode::INVALID_REQUEST_ERROR,
                    "item is not an assistant audio item",
                    event_id,
                    Some("item_id"),
                )
                .await;
            }
            Ok((clamped, _transcript)) => {
                self.send_to_client(&json!({
                    "type": "conversation.item.truncated",
                    "item_id": item_id,
                    "content_index": content_index,
                    "audio_end_ms": clamped,
                }))
                .await;
            }
        }
    }

    async fn handle_response_create(
        self: &Arc<Self>,
        value: serde_json::Value,
        event_id: Option<&str>,
    ) {
        let inbound_response = value.get("response");
        let override_instructions: Option<String> = inbound_response
            .and_then(|r| r.get("instructions"))
            .and_then(|v| v.as_str())
            .map(|s| s.to_string());
        let override_modalities: Option<Vec<String>> = inbound_response
            .and_then(|r| r.get("modalities"))
            .and_then(|v| v.as_array())
            .map(|arr| {
                arr.iter()
                    .filter_map(|x| x.as_str().map(|s| s.to_string()))
                    .collect()
            });
        if self.intent != Intent::Conversation {
            super::events::emit_error(
                self,
                errcode::INVALID_REQUEST_ERROR,
                "session intent is not 'conversation'",
                event_id,
                None,
            )
            .await;
            return;
        }
        {
            let state = self.state.lock().await;
            let busy = state.current_response.is_some()
                || !matches!(state.resp, super::state::RespPhase::None);
            if busy {
                drop(state);
                super::events::emit_error(
                    self,
                    errcode::RESPONSE_ALREADY_ACTIVE,
                    "a response is already active",
                    event_id,
                    None,
                )
                .await;
                return;
            }
        }
        let response_has_input = inbound_response.and_then(|r| r.get("input")).is_some();
        let speak_item_id: Option<String> = inbound_response
            .and_then(|r| r.get("speak_item_id"))
            .and_then(|v| v.as_str())
            .map(|s| s.to_string());
        let mut spoken_item: Option<ItemId> = None;
        let selection = {
            let mut state = self.state.lock().await;
            let source = super::state::select_response_source(
                &state.conversation,
                response_has_input,
                speak_item_id.as_deref(),
            );
            match source {
                super::state::ResponseSource::Speak { index, text } => {
                    if matches!(state.vad, VadPhase::Speaking { .. }) {
                        drop(state);
                        super::events::emit_error(
                            self,
                            errcode::INVALID_REQUEST_ERROR,
                            "cannot create a response while input speech is active",
                            event_id,
                            None,
                        )
                        .await;
                        return;
                    }
                    state.conversation[index].client_speakable = false;
                    spoken_item = Some(ItemId::new(state.conversation[index].id.clone()));
                    super::state::ResponseSource::Speak { index, text }
                }
                other => other,
            }
        };
        let (prompt_text, prefilled) = match selection {
            super::state::ResponseSource::SpeakUnavailable { reason } => {
                super::events::emit_error(
                    self,
                    errcode::INVALID_REQUEST_ERROR,
                    reason,
                    event_id,
                    Some("response.speak_item_id"),
                )
                .await;
                return;
            }
            super::state::ResponseSource::Speak { text, .. } => (
                String::new(),
                Some(super::pipeline::PrefilledText::ClientSupplied(text)),
            ),
            super::state::ResponseSource::Generate { prompt } => {
                if prompt.is_empty() {
                    super::events::emit_error(
                        self,
                        errcode::INVALID_REQUEST_ERROR,
                        "no user message in conversation to respond to",
                        event_id,
                        None,
                    )
                    .await;
                    return;
                }
                (prompt, None)
            }
        };
        let response_id = self.id_source.response();
        let assistant_item_id = spoken_item.unwrap_or_else(|| self.id_source.item());
        let played_ms = Arc::new(AtomicU64::new(0));
        let transcript_so_far = Arc::new(Mutex::new(String::new()));
        let wire_opened = Arc::new(AtomicBool::new(false));
        let session_for_task = self.clone();
        let response_id_for_task = response_id.clone();
        let assistant_item_id_for_task = assistant_item_id.as_str().to_string();
        let played_ms_for_task = played_ms.clone();
        let transcript_so_far_for_task = transcript_so_far.clone();
        let wire_opened_for_task = wire_opened.clone();
        let prompt_for_task = prompt_text.clone();
        let overrides = super::pipeline::ResponseOverrides {
            instructions: override_instructions,
            modalities: override_modalities,
        };
        let handle = tokio::spawn(self.cancel.wrap_unit(async move {
            let _ = super::pipeline::run_response(
                &session_for_task,
                response_id_for_task.as_str(),
                assistant_item_id_for_task,
                prompt_for_task,
                played_ms_for_task,
                transcript_so_far_for_task,
                prefilled,
                overrides,
                wire_opened_for_task,
            )
            .await;
            session_for_task
                .clear_response_if_matches(&response_id_for_task)
                .await;
        }));
        self.register_response(
            response_id,
            handle,
            played_ms,
            assistant_item_id,
            transcript_so_far,
            wire_opened,
        )
        .await;
    }

    pub(super) async fn instructions(&self) -> Option<String> {
        self.state.lock().await.instructions.clone()
    }

    pub(super) async fn complete_user_item_transcript(&self, id: &str, transcript: String) {
        let mut state = self.state.lock().await;
        if let Some(item) = state.conversation.iter_mut().find(|i| i.id == id) {
            item.status = ItemStatus::Completed;
            if let ItemContent::UserAudio { transcript: t, .. } = &mut item.content {
                *t = Some(transcript);
            }
        }
    }

    pub(super) async fn mark_user_item_incomplete(&self, id: &str) {
        let mut state = self.state.lock().await;
        if let Some(item) = state.conversation.iter_mut().find(|i| i.id == id) {
            item.status = ItemStatus::Incomplete;
        }
    }

    pub(super) async fn append_assistant_item(
        &self,
        id: String,
        transcript: String,
        audio_ms: u64,
    ) {
        let mut state = self.state.lock().await;
        state
            .conversation
            .push(ConversationItem::new_assistant_audio(
                id, transcript, audio_ms,
            ));
    }

    pub async fn build_eou_context(&self, k: usize) -> String {
        let state = self.state.lock().await;
        let total = state.conversation.len();
        let start = total.saturating_sub(k);
        let mut out = String::new();
        for item in state.conversation.iter().skip(start) {
            let Some(text) = item.transcript() else {
                continue;
            };
            if text.is_empty() {
                continue;
            }
            if !out.is_empty() {
                out.push('\n');
            }
            out.push_str(item.role.as_str());
            out.push_str(": ");
            out.push_str(text);
        }
        out
    }

    pub(super) async fn build_chat_messages(&self, instructions: Option<&str>) -> Vec<ChatMessage> {
        let state = self.state.lock().await;
        let mut out = Vec::new();
        if let Some(sys) = instructions.filter(|s| !s.is_empty()) {
            out.push(ChatMessage {
                role: "system".into(),
                content: sys.to_string(),
            });
        }
        for item in state.conversation.iter() {
            let Some(text) = item.transcript() else {
                continue;
            };
            if text.is_empty() {
                continue;
            }
            out.push(ChatMessage {
                role: item.role.as_str().to_string(),
                content: text.to_string(),
            });
        }
        out
    }

    pub(crate) async fn event_sink(&self) -> Option<EventSink> {
        self.state.lock().await.event_sink.clone()
    }

    pub async fn emit_event(&self, ev: serde_json::Value) {
        let typ = ev.get("type").and_then(|t| t.as_str()).unwrap_or("");
        let topic = state::Topic::classify(typ);
        let sink = {
            let state = self.state.lock().await;
            if matches!(topic, state::Topic::Response)
                && matches!(&state.resp, RespPhase::Predicted { .. })
            {
                warn!(
                    event = typ,
                    "I7: refused to emit response.* while Predicted"
                );
                return;
            }
            state.event_sink.clone()
        };
        let Some(sink) = sink else {
            return;
        };
        let _inflight = match InflightGuard::try_acquire(
            &self.outbound_inflight,
            self.outbound_queue_cap_events,
        ) {
            Ok(guard) => guard,
            Err(inflight) => {
                warn!(
                    inflight,
                    cap = self.outbound_queue_cap_events,
                    event = typ,
                    "outbound event queue cap exceeded; dropping event"
                );
                return;
            }
        };
        self.deliver_to_sink(&sink, ev).await;
    }

    pub async fn emit(&self, ev: super::wire::OutboundEvent) {
        let topic = ev.topic();
        let typ = ev.type_name();
        let sink = {
            let state = self.state.lock().await;
            if matches!(topic, state::Topic::Response)
                && matches!(&state.resp, RespPhase::Predicted { .. })
            {
                warn!(
                    event = typ,
                    "I7: refused to emit response.* while Predicted"
                );
                return;
            }
            state.event_sink.clone()
        };
        let Some(sink) = sink else {
            return;
        };
        let value = match serde_json::to_value(&ev) {
            Ok(v) => v,
            Err(err) => {
                warn!(event = typ, error = %err, "failed to serialize OutboundEvent");
                return;
            }
        };
        self.deliver_to_sink(&sink, value).await;
    }

    async fn set_current_speech_item(&self, item_id: Option<String>) {
        self.state.lock().await.current_speech_item = item_id;
    }

    async fn current_speech_item(&self) -> Option<String> {
        self.state.lock().await.current_speech_item.clone()
    }

    pub(super) async fn register_response(
        self: &Arc<Self>,
        id: ResponseId,
        handle: JoinHandle<()>,
        _played_ms: Arc<AtomicU64>,
        assistant_item_id: ItemId,
        transcript_so_far: Arc<Mutex<String>>,
        wire_opened: Arc<AtomicBool>,
    ) {
        let mut state = self.state.lock().await;
        let runtime = ResponseRuntime {
            handle,
            transcript_so_far,
            wire_opened,
        };
        if let Err(v) = state.resp_create_from_none(id, assistant_item_id, runtime) {
            warn!(violation = ?v, "register_response invariant violation");
            return;
        }
        check_or_react(self, &state);
    }

    pub(super) async fn clear_response_if_matches(self: &Arc<Self>, id: &ResponseId) {
        let mut state = self.state.lock().await;
        let phase_id_matches = state.resp.id().map(|rid| rid == id).unwrap_or(false);
        if phase_id_matches {
            let _ = state.resp_retire_to_none();
        } else if matches!(&state.resp, RespPhase::None) && state.current_response.is_some() {
            state.current_response = None;
        }
        check_or_react(self, &state);
    }

    pub(super) async fn mark_streaming(self: &Arc<Self>, id: &str) {
        let mut state = self.state.lock().await;
        let is_created_match =
            matches!(&state.resp, RespPhase::Created { id: rid, .. } if rid.as_str() == id);
        if is_created_match {
            let _ = state.resp_advance_to_streaming(Arc::new(AtomicU64::new(0)));
        }
        check_or_react(self, &state);
    }

    async fn set_pending_bargein(&self, pending: PendingBargein, task: JoinHandle<()>) {
        let mut state = self.state.lock().await;
        if let Some(prev) = state.bargein_task.replace(task) {
            prev.abort();
        }
        state.pending_bargein = Some(pending);
    }

    async fn take_pending_bargein_if(&self, item_id: &str) -> Option<PendingBargein> {
        let mut state = self.state.lock().await;
        match &state.pending_bargein {
            Some(p) if p.item_id == item_id => {
                let taken = state.pending_bargein.take();
                state.bargein_task = None;
                taken
            }
            _ => None,
        }
    }

    async fn clear_pending_bargein_for_suppression(&self, item_id: &str) -> bool {
        let mut state = self.state.lock().await;
        match &state.pending_bargein {
            Some(p) if p.item_id == item_id => {
                state.pending_bargein = None;
                if let Some(h) = state.bargein_task.take() {
                    h.abort();
                }
                true
            }
            _ => false,
        }
    }

    pub(super) async fn apply_truncate_to_assistant_item(
        self: &Arc<Self>,
        snap: &CancelledSnapshot,
    ) {
        let mut state = self.state.lock().await;
        state::apply_truncate_to_conversation(
            &mut state.conversation,
            snap.assistant_item_id.as_str(),
            snap.played_ms,
            &snap.transcript,
        );
        check_or_react(self, &state);
    }

    pub(super) async fn cancel_current_response(self: &Arc<Self>) -> Option<CancelledSnapshot> {
        let (runtime, response_id, assistant_item_id, played_ms_snapshot) = {
            let mut state = self.state.lock().await;
            let response_id = state.resp.id().cloned();
            let assistant_item_id = state.resp.item_id().cloned();
            let played_ms_snapshot = state
                .resp
                .played_ms()
                .map(|p| p.load(Ordering::Acquire))
                .unwrap_or(0);
            let runtime = state.current_response.take();
            state.resp = RespPhase::None;
            check_or_react(self, &state);
            (runtime, response_id, assistant_item_id, played_ms_snapshot)
        };
        let runtime = runtime?;
        runtime.handle.abort();
        if let Some(rid) = response_id.as_ref() {
            if self.tts_abort.cancel(rid.as_str()).await {
                info!(
                    cancelled_id = %rid,
                    "cancellation aborted the in-flight TTS worker (synthesis stopped, not just delivery)",
                );
            }
        }
        let transcript = runtime.transcript_so_far.lock().await.clone();
        let wire_opened = runtime.wire_opened.load(Ordering::Acquire);
        Some(CancelledSnapshot {
            response_id: response_id?,
            assistant_item_id: assistant_item_id?,
            played_ms: played_ms_snapshot,
            transcript,
            wire_opened,
        })
    }

    async fn install_commit_timer(&self, handle: JoinHandle<()>) {
        let mut state = self.state.lock().await;
        if let Some(prev) = state.commit_timer.take() {
            prev.abort();
        }
        state.commit_timer = Some(handle);
    }

    async fn cancel_commit_timer(&self) -> bool {
        let prev = {
            let mut state = self.state.lock().await;
            state.commit_timer.take()
        };
        if let Some(h) = prev {
            h.abort();
            true
        } else {
            false
        }
    }

    pub(super) async fn clear_commit_timer(&self) {
        let mut state = self.state.lock().await;
        state.commit_timer = None;
    }

    async fn rollback_predicted_if_any(self: &Arc<Self>, reason: &'static str) {
        let (runner, llm_runner_handle, llm_chars) = {
            let mut state = self.state.lock().await;
            if !matches!(state.resp, RespPhase::Predicted { .. }) {
                return;
            }
            let response_id = state
                .resp
                .id()
                .map(|r| r.as_str().to_string())
                .unwrap_or_default();
            let predicted_id_for_inspector = response_id.clone();
            let (runner_opt, llm_handle_opt) = state
                .resp_retire_predicted_full()
                .ok()
                .unwrap_or((None, None));
            let llm_chars = llm_handle_opt
                .as_ref()
                .map(|h| {
                    h.shared
                        .chars_seen
                        .load(std::sync::atomic::Ordering::Relaxed)
                })
                .unwrap_or(0);
            self.inspector.emit(InspectorEvent::PredictedRollback {
                session_id: self.id.as_str().to_string(),
                response_id: predicted_id_for_inspector.clone(),
                score: 0.0,
            });
            self.inspector.emit(InspectorEvent::EouPredictedRollback {
                session_id: self.id.as_str().to_string(),
                response_id: predicted_id_for_inspector,
                reason,
                llm_chars_thrown: llm_chars,
            });
            (runner_opt, llm_handle_opt, llm_chars)
        };
        let _ = llm_chars;
        if let Some(r) = runner {
            r.task.abort();
        }
        if let Some(h) = llm_runner_handle {
            h.into_runner().abort();
        }
        let mut last = self.last_eager_dispatch_at.lock().await;
        *last = None;
    }

    #[allow(dead_code)]
    pub async fn set_integrated_backend(&self, backend: Arc<dyn IntegratedEouBackend>) {
        let mut slot = self.integrated_backend.lock().await;
        if let Some(old) = slot.replace(backend) {
            old.reset();
        }
    }

    #[allow(dead_code)]
    pub async fn handle_integrated_verdict(
        self: &Arc<Self>,
        verdict: &IntegratedVerdict,
    ) -> IntegratedVerdictAction {
        if !matches!(self.eou_config.kind, EouKind::Integrated) {
            return IntegratedVerdictAction::Ignored;
        }
        let cfg = &self.eou_config;
        let p_eot = clip01(verdict.p_eot);
        let p_eager = clip01(verdict.p_eager_eot);

        self.inspector.emit(InspectorEvent::EouScored {
            session_id: self.id.as_str().to_string(),
            kind: EouKind::Integrated.as_str(),
            score: p_eot,
            eager_score: Some(p_eager),
            threshold: cfg.eot_threshold,
            language: None,
            input_chars: Some(verdict.transcript_so_far.chars().count() as u32),
            input_audio_ms: None,
            delay_ms: 0,
            elapsed_ms: 0,
            cancelled_by: "none",
            hard_cap_fired: false,
        });

        if p_eot >= cfg.eot_threshold {
            return IntegratedVerdictAction::Commit;
        }

        if p_eager >= cfg.eager_eot_threshold {
            let mut state = self.state.lock().await;
            if matches!(state.resp, RespPhase::None) {
                let predicted_id = self.id_source.response();
                let predicted_item_id = self.id_source.item();
                if state
                    .resp_start_predicted(predicted_id.clone(), predicted_item_id, p_eager, None)
                    .is_ok()
                {
                    self.inspector.emit(InspectorEvent::PredictedPromoted {
                        session_id: self.id.as_str().to_string(),
                        response_id: predicted_id.as_str().to_string(),
                        score: p_eager,
                    });
                    return IntegratedVerdictAction::StartedPredicted;
                }
            }
        }

        IntegratedVerdictAction::None
    }

    #[allow(dead_code)]
    pub async fn handle_stt_turn_resumed(self: &Arc<Self>) {
        self.rollback_predicted_if_any("turn_resumed").await;
    }
}

fn read_session_max_duration_s_from_env() -> u64 {
    std::env::var(defaults::env::SESSION_MAX_DURATION_S)
        .ok()
        .and_then(|v| v.parse::<u64>().ok())
        .unwrap_or(defaults::session::MAX_DURATION_S)
}

pub(crate) fn validate_session_max_duration_s(n: u64) -> Result<u64, &'static str> {
    if n == 0 {
        return Err("session_max_duration_s: must be >= 1");
    }
    if n > defaults::session::MAX_DURATION_HARD_CAP_S {
        return Err("session_max_duration_s: exceeds session_max_duration_hard_cap_s");
    }
    Ok(n)
}

pub(crate) fn compute_remaining_timeout_ms(new_secs: u64, created_at_ms: u64, now_ms: u64) -> u64 {
    let elapsed_ms = now_ms.saturating_sub(created_at_ms);
    let total_ms = new_secs.saturating_mul(1000);
    total_ms.saturating_sub(elapsed_ms)
}

#[allow(dead_code)]
fn clip01(x: f32) -> f32 {
    if !x.is_finite() {
        return 1.0;
    }
    x.clamp(0.0, 1.0)
}

pub(super) struct CancelledSnapshot {
    pub(super) response_id: ResponseId,
    pub(super) assistant_item_id: ItemId,
    pub(super) played_ms: u64,
    pub(super) transcript: String,

    pub(super) wire_opened: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum CancelReason {
    ClientCancelled,

    BargeIn,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum FailReason {
    LlmError,

    TtsError,

    ClientTooSlow,
}

impl From<FailReason> for super::wire::ResponseStatusReason {
    fn from(r: FailReason) -> Self {
        match r {
            FailReason::LlmError => super::wire::ResponseStatusReason::LlmError,
            FailReason::TtsError => super::wire::ResponseStatusReason::TtsError,
            FailReason::ClientTooSlow => super::wire::ResponseStatusReason::ClientTooSlow,
        }
    }
}

impl Session {
    pub(crate) async fn ensure_audio_in_pipeline(self: &Arc<Self>) -> mpsc::Sender<AudioIn> {
        let mut guard = self.audio_in_tx.lock().await;
        if let Some(tx) = guard.as_ref() {
            return tx.clone();
        }
        let (tx, rx) = mpsc::channel::<AudioIn>(64);
        match self.models.vad() {
            Ok(vad) => {
                let vad_model = VadModel::from_session(vad);
                let session = self.clone();
                spawn_vad_task(rx, vad_model, session);
            }

            Err(e) => {
                tracing::error!(error = %format!("{e:#}"), "realtime audio session started without VAD; incoming audio will be discarded")
            }
        }
        *guard = Some(tx.clone());
        tx
    }

    pub(crate) async fn ingest_pcm16_b64(
        self: &Arc<Self>,
        ev: serde_json::Value,
        event_id: Option<&str>,
    ) {
        let audio = ev
            .get("audio")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string());
        let Some(audio) = audio else {
            super::events::emit_error(
                self,
                errcode::INVALID_REQUEST_ERROR,
                "missing 'audio' field on input_audio_buffer.append",
                event_id,
                Some("audio"),
            )
            .await;
            return;
        };
        let format = self
            .input_audio_format
            .lock()
            .expect("input_audio_format poisoned")
            .clone();
        let samples = {
            let mut guard = self.ws_ingest.lock().await;
            if guard.is_none() {
                match WsAudioIngest::new(&format) {
                    Ok(i) => *guard = Some(i),
                    Err(err) => {
                        super::events::emit_error(
                            self,
                            errcode::INVALID_REQUEST_ERROR,
                            &format!("input_audio_buffer.append: {err}"),
                            event_id,
                            Some("input_audio_format"),
                        )
                        .await;
                        return;
                    }
                }
            }
            let ingest = guard.as_mut().expect("ws_ingest just initialized");
            match ingest.ingest_b64(&audio) {
                Ok(samples) => samples,
                Err(err) => {
                    super::events::emit_error(
                        self,
                        errcode::INVALID_REQUEST_ERROR,
                        &format!("input_audio_buffer.append: {err}"),
                        event_id,
                        Some("audio"),
                    )
                    .await;
                    return;
                }
            }
        };
        if samples.is_empty() {
            return;
        }
        self.audio_store.append_mic_in_f32(&samples);
        let tx = self.ensure_audio_in_pipeline().await;
        if tx.send(AudioIn::Samples(samples)).await.is_err() {
            debug!("VAD task dropped; ws audio frame discarded");
        }
    }

    pub async fn attach_audio_track(self: &Arc<Self>, track: Arc<TrackRemote>) {
        let codec = track.codec();
        let channels = codec.capability.channels as usize;
        let mut ingest = match AudioIngest::new(channels) {
            Ok(ingest) => ingest,
            Err(err) => {
                warn!(error = %err, channels, "failed to build audio pipeline");
                return;
            }
        };

        let tx = self.ensure_audio_in_pipeline().await;

        let audio_store = self.audio_store.clone();
        tokio::spawn(self.cancel.wrap_unit(async move {
            let mut packets = 0u64;
            loop {
                match track.read_rtp().await {
                    Ok((pkt, _attrs)) => {
                        packets += 1;
                        if let Err(err) = ingest.process(&pkt.payload) {
                            warn!(error = %err, "opus decode failed; dropping packet");
                            continue;
                        }
                        let samples = ingest.take();
                        if samples.is_empty() {
                            continue;
                        }
                        audio_store.append_mic_in_f32(&samples);
                        if tx.send(AudioIn::Samples(samples)).await.is_err() {
                            debug!(packets, "VAD task dropped; ending RTP loop");
                            break;
                        }
                    }
                    Err(err) => {
                        info!(error = %err, packets, "track ended");
                        break;
                    }
                }
            }
        }));
    }
}

pub(crate) fn vad_supervisor_step<M: VadInfer>(
    processor: &mut VadProcessor<M>,
    consecutive_errors: &mut u32,
    chunk: &[f32],
    threshold: u32,
) -> Vec<VadEvent> {
    match processor.push(chunk) {
        Ok(()) => {
            *consecutive_errors = 0;
            processor.take_events()
        }
        Err(err) => {
            *consecutive_errors = consecutive_errors.saturating_add(1);
            warn!(
                error = %err,
                consecutive = *consecutive_errors,
                threshold,
                "VAD push failed",
            );
            let mut events = processor.take_events();
            if *consecutive_errors >= threshold {
                events.push(VadEvent::Failed {
                    reason: err.to_string(),
                });
                *consecutive_errors = threshold;
            }
            events
        }
    }
}

fn spawn_vad_task(mut rx: mpsc::Receiver<AudioIn>, model: VadModel, session: Arc<Session>) {
    let partial_enabled = std::env::var(defaults::env::PARTIAL_STT_ENABLED)
        .map(|v| !v.is_empty() && v != "0" && !v.eq_ignore_ascii_case("false"))
        .unwrap_or(false);
    let partial_interval_samples =
        VAD_SAMPLE_RATE * defaults::buffer::PARTIAL_INTERVAL_MS as usize / 1000;
    let td_for_vad: Arc<dyn TurnDetectionRead> = session.turn_detection.clone();
    let vad_cancel = session.cancel.clone();
    tokio::spawn(vad_cancel.wrap_unit(async move {
        let mut processor = VadProcessor::new(model).with_turn_detection(td_for_vad);
        let partial_in_progress = Arc::new(AtomicBool::new(false));
        let mut partial_task: Option<JoinHandle<()>> = None;
        let mut total_samples: usize = 0;
        let mut last_partial_at: usize = 0;
        let mut consecutive_errors: u32 = 0;
        let mut failed_emitted = false;
        let mut clear_epoch = session.input_clear_epoch.load(Ordering::Acquire);
        debug!("VAD task started");
        while let Some(msg) = rx.recv().await {
            let epoch = session.input_clear_epoch.load(Ordering::Acquire);
            if epoch != clear_epoch {
                clear_epoch = epoch;
                let Ok(vad) = session.models.vad() else {
                    tracing::error!("VAD disappeared mid-session; stopping the VAD task");
                    return;
                };
                processor = VadProcessor::new(VadModel::from_session(vad))
                    .with_turn_detection(session.turn_detection.clone());
                if let Some(prev) = partial_task.take() {
                    prev.abort();
                }
                partial_in_progress.store(false, Ordering::SeqCst);
                total_samples = 0;
                last_partial_at = 0;
                consecutive_errors = 0;
                debug!("input buffer cleared; VAD processor reset");
            }
            let events = match msg {
                AudioIn::Samples(chunk) => {
                    total_samples += chunk.len();
                    vad_supervisor_step(
                        &mut processor,
                        &mut consecutive_errors,
                        &chunk,
                        VAD_FAILURE_THRESHOLD,
                    )
                }
                AudioIn::ForceCommit => {
                    if processor.force_commit() {
                        debug!("client commit: sealing the open buffer");
                        processor.take_events()
                    } else {
                        debug!("client commit: nothing open to seal");
                        Vec::new()
                    }
                }
            };
            for ev in events {
                if matches!(
                    ev,
                    VadEvent::SpeechCommitted { .. } | VadEvent::SpeechStarted { .. }
                ) {
                    if let Some(prev) = partial_task.take() {
                        prev.abort();
                    }
                    partial_in_progress.store(false, Ordering::SeqCst);
                    last_partial_at = total_samples;
                }
                let is_failed = matches!(ev, VadEvent::Failed { .. });
                if is_failed && failed_emitted {
                    continue;
                }
                if is_failed {
                    failed_emitted = true;
                }
                handle_vad_event(&session, ev).await;
                if failed_emitted {
                    info!("VAD supervisor exiting after Failed");
                    return;
                }
            }
            if partial_enabled
                && total_samples.saturating_sub(last_partial_at) >= partial_interval_samples
                && !partial_in_progress.load(Ordering::SeqCst)
            {
                if let Some((item_id, audio)) = processor.current_speech_audio() {
                    last_partial_at = total_samples;
                    partial_in_progress.store(true, Ordering::SeqCst);
                    let session_for_task = session.clone();
                    let flag = partial_in_progress.clone();
                    let audio_end_ms = (total_samples * 1000 / VAD_SAMPLE_RATE) as u64;
                    let partial_cancel = session_for_task.cancel.clone();
                    let h = tokio::spawn(partial_cancel.wrap_unit(async move {
                        run_partial_transcription(session_for_task, item_id, audio, audio_end_ms)
                            .await;
                        flag.store(false, Ordering::SeqCst);
                    }));
                    if let Some(prev) = partial_task.replace(h) {
                        prev.abort();
                    }
                }
            }
        }
        if let Some(prev) = partial_task.take() {
            prev.abort();
        }
        info!("VAD task ended");
    }));
}

async fn run_partial_transcription(
    session: Arc<Session>,
    item_id: String,
    audio: Vec<f32>,
    audio_end_ms: u64,
) {
    let Ok(whisper) = session.models.whisper() else {
        tracing::error!("partial transcription requested but speech-to-text is unavailable");
        return;
    };
    let transcript =
        match super::pipeline::run_stt(whisper, crate::types::MonoF32At16k::new(audio)).await {
            Ok(t) => t,
            Err(err) => {
                debug!(error = %err, "partial STT failed; dropping");
                return;
            }
        };
    if transcript.is_empty() {
        return;
    }
    let still_current = match session.current_speech_item().await {
        Some(current) => current == item_id,
        None => false,
    };
    if !still_current {
        debug!(%item_id, "partial result stale; dropping");
        return;
    }
    session
        .inspector
        .emit(InspectorEvent::PartialTranscription {
            session_id: session.id.as_str().to_string(),
            item_id: item_id.clone(),
            transcript: transcript.clone(),
            ms: audio_end_ms,
        });
    session
        .send_to_client(&json!({
            "type": "input_audio_buffer.partial_transcription",
            "item_id": item_id,
            "transcript": transcript,
            "audio_end_ms": audio_end_ms,
        }))
        .await;
}

async fn handle_vad_event(session: &Arc<Session>, ev: VadEvent) {
    match ev {
        VadEvent::SpeechStarted {
            item_id,
            audio_start_ms,
        } => {
            session.set_current_speech_item(Some(item_id.clone())).await;
            session.inspector.emit(InspectorEvent::TurnStart {
                session_id: session.id.as_str().to_string(),
                turn_id: item_id.clone(),
                role: "user",
            });
            session.inspector.emit(InspectorEvent::VadConfirmedStart {
                session_id: session.id.as_str().to_string(),
                item_id: item_id.clone(),
                ms: audio_start_ms,
            });
            if session.cancel_commit_timer().await {
                debug!("EOU commit timer aborted by new speech_started");
            }
            session.rollback_predicted_if_any("speech_resumed").await;
            let response_active = session.state.lock().await.current_response.is_some();
            let barge_in_delay = session.turn_detection.barge_in_delay_ms();
            if response_active && barge_in_delay > 0 {
                let delay_ms = barge_in_delay as u64;
                session.inspector.emit(InspectorEvent::BargeinPending {
                    session_id: session.id.as_str().to_string(),
                    delay_ms,
                });
                let session_for_delay = session.clone();
                let pending_item_id = item_id.clone();
                let task = tokio::spawn(session.cancel.wrap_unit(async move {
                    tokio::time::sleep(Duration::from_millis(delay_ms)).await;
                    if let Some(_taken) = session_for_delay
                        .take_pending_bargein_if(&pending_item_id)
                        .await
                    {
                        super::pipeline::commit_bargein(
                            &session_for_delay,
                            &pending_item_id,
                            audio_start_ms,
                        )
                        .await;
                    }
                }));
                session
                    .set_pending_bargein(
                        PendingBargein {
                            item_id: item_id.clone(),
                            audio_start_ms,
                        },
                        task,
                    )
                    .await;
                return;
            }
            super::pipeline::commit_bargein(session, &item_id, audio_start_ms).await;
        }
        VadEvent::SpeechCommitted {
            item_id,
            audio_end_ms,
            audio,
            speech_samples,
        } => {
            if session
                .clear_pending_bargein_for_suppression(&item_id)
                .await
            {
                session.inspector.emit(InspectorEvent::BargeinSuppressed {
                    session_id: session.id.as_str().to_string(),
                    reason: "speech_stopped",
                });
                info!(%item_id, "barge-in suppressed: speech_committed during delay");
                session.set_current_speech_item(None).await;
                return;
            }
            let speech_samples = speech_samples.min(audio.len());
            let audio_ms = (speech_samples as u64) * 1000 / 16_000;
            session.set_current_speech_item(None).await;
            session.inspector.emit(InspectorEvent::VadConfirmedStop {
                session_id: session.id.as_str().to_string(),
                item_id: item_id.clone(),
                ms: audio_end_ms,
            });
            info!(samples = audio.len(), %item_id, audio_ms, "speech stopped");

            {
                let mut state = session.state.lock().await;
                let audio_start_ms = match &state.vad {
                    VadPhase::Speaking { audio_start_ms, .. }
                    | VadPhase::Stopped { audio_start_ms, .. } => *audio_start_ms,
                    _ => crate::types::Millis::zero(),
                };
                state.vad = VadPhase::Stopped {
                    item_id: crate::types::ItemId::new(item_id.clone()),
                    audio_start_ms,
                    audio_end_ms: crate::types::Millis(audio_end_ms),
                };
                check_or_react(session, &state);
            }
            session
                .send_to_client(&json!({
                    "type": "input_audio_buffer.speech_stopped",
                    "item_id": item_id,
                    "audio_end_ms": audio_end_ms,
                }))
                .await;

            if audio_ms < MIN_SPEECH_MS {
                info!(
                    %item_id,
                    audio_ms,
                    min_speech_ms = MIN_SPEECH_MS,
                    "rejecting commit: buffer below min_speech_ms",
                );
                super::events::emit_error(
                    session,
                    errcode::INPUT_AUDIO_BUFFER_COMMIT_EMPTY,
                    &format!("buffer below min_speech_ms (got {}ms)", audio_ms),
                    None,
                    None,
                )
                .await;
                let mut state = session.state.lock().await;
                if matches!(state.vad, VadPhase::Stopped { .. }) {
                    state.vad = VadPhase::Silent;
                }
                check_or_react(session, &state);
                return;
            }

            if super::diarization::realtime_enabled() {
                let session_for_diar = session.clone();
                let item_id_for_diar = item_id.clone();
                let audio_for_diar = audio[..speech_samples].to_vec();
                tokio::spawn(session.cancel.wrap_unit(async move {
                    super::diarization::run_diarization(
                        session_for_diar,
                        item_id_for_diar,
                        audio_for_diar,
                        audio_end_ms,
                    )
                    .await;
                }));
            }

            if session.eou_config.kind.calls_classifier()
                && !matches!(session.eou_config.kind, EouKind::Integrated)
            {
                if let Some(model) = session.eou_model.as_ref() {
                    let cfg = session.eou_config.clone();
                    let context = session.build_eou_context(cfg.context_turns as usize).await;
                    let input_chars = context.chars().count() as u32;
                    let model_for_score = model.clone();
                    let session_timer = session.clone();
                    let item_id_timer = item_id.clone();
                    let suppress_response = audio_ms < cfg.min_speech_for_response_ms;
                    let handle = tokio::spawn(session.cancel.wrap_unit(async move {
                        super::pipeline::run_eou_dispatch(
                            session_timer,
                            cfg,
                            model_for_score,
                            context,
                            input_chars,
                            item_id_timer,
                            audio,
                            speech_samples,
                            audio_ms,
                            suppress_response,
                        )
                        .await;
                    }));
                    session.install_commit_timer(handle).await;
                    return;
                }
            }
            let suppress_response = audio_ms < session.eou_config.min_speech_for_response_ms;
            let mut audio = audio;
            audio.truncate(speech_samples);
            super::pipeline::commit_after_eou(session, item_id, audio, audio_ms, suppress_response)
                .await;
        }
        VadEvent::Failed { reason } => {
            session.inspector.emit(InspectorEvent::VadFailed {
                session_id: session.id.as_str().to_string(),
                reason: reason.clone(),
            });
            warn!(%reason, "VAD failed; terminating session");
            let session_for_term = session.clone();
            let reason_for_term = reason.clone();
            tokio::spawn(async move {
                terminate_with_error(
                    session_for_term,
                    errcode::VAD_FAILED,
                    &format!("VAD failed: {}", reason_for_term),
                )
                .await;
            });
        }
    }
}

pub(super) fn check_or_react(session: &Arc<Session>, state: &SessionState) {
    if let Err(v) = state::check_state(state) {
        let violation_dbg = format!("{:?}", v);
        error!(?v, "invariant violated; terminating session");
        session.inspector.emit(InspectorEvent::InvariantViolation {
            session_id: session.id.as_str().to_string(),
            violation: violation_dbg.clone(),
        });
        let session_for_term = session.clone();
        tokio::spawn(async move {
            terminate_with_error(
                session_for_term,
                errcode::INTERNAL_STATE_ERROR,
                &format!("RFC invariant violated: {}", violation_dbg),
            )
            .await;
        });
    }
}

async fn terminate_with_error(session: Arc<Session>, code: &'static str, message: &str) {
    super::events::emit_error(&session, code, message, None, None).await;
    let reason = match code {
        c if c == errcode::VAD_FAILED => state::TerminationReason::VadFailed,
        c if c == errcode::STT_FAILED => state::TerminationReason::SttFailed,
        c if c == errcode::INTERNAL_STATE_ERROR => state::TerminationReason::InternalStateError,
        c if c == errcode::MODEL_LOAD_FAILED => state::TerminationReason::ModelLoadFailed,
        _ => state::TerminationReason::InternalStateError,
    };
    session.transition_to_terminated_with(reason).await;
    if let Some(pc) = session.peer_connection().await {
        if let Err(err) = pc.close().await {
            warn!(error = %err, session_id = %session.id, "pc.close failed during terminate_with_error");
        }
    }
    super::drop_session(session.id.as_str());
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn client_commit_is_rejected_only_on_a_silent_buffer() {
        assert_eq!(commit_action(&VadPhase::Silent), CommitAction::RejectEmpty);
        assert_eq!(
            commit_action(&VadPhase::Speaking {
                item_id: crate::types::ItemId::new("item_a"),
                audio_start_ms: crate::types::Millis(0),
            }),
            CommitAction::Force,
            "RFC v3 §7.1: commit fires in any non-Terminated state"
        );
        assert_eq!(
            commit_action(&VadPhase::Stopped {
                item_id: crate::types::ItemId::new("item_a"),
                audio_start_ms: crate::types::Millis(0),
                audio_end_ms: crate::types::Millis(500),
            }),
            CommitAction::Force,
            "turn_detection=none parks the buffer in Stopped; commit must still seal it"
        );
    }

    #[test]
    fn validate_session_max_duration_rejects_zero() {
        assert!(validate_session_max_duration_s(0).is_err());
    }

    #[test]
    fn validate_session_max_duration_rejects_above_hard_cap() {
        let cap = defaults::session::MAX_DURATION_HARD_CAP_S;
        assert!(validate_session_max_duration_s(cap + 1).is_err());
    }

    #[test]
    fn validate_session_max_duration_accepts_in_range() {
        assert_eq!(validate_session_max_duration_s(1), Ok(1));
        let cap = defaults::session::MAX_DURATION_HARD_CAP_S;
        assert_eq!(validate_session_max_duration_s(cap), Ok(cap));
        assert_eq!(validate_session_max_duration_s(900), Ok(900));
    }

    #[test]
    fn remaining_timeout_preserves_elapsed() {
        let new_secs = 600;
        let created = 1_000_000_u64;
        let now = created + 5 * 60 * 1000;
        let remaining = compute_remaining_timeout_ms(new_secs, created, now);
        assert_eq!(remaining, 5 * 60 * 1000);
    }

    #[test]
    fn remaining_timeout_zero_when_new_value_below_elapsed() {
        let new_secs = 60;
        let created = 1_000_000_u64;
        let now = created + 90 * 1000;
        let remaining = compute_remaining_timeout_ms(new_secs, created, now);
        assert_eq!(remaining, 0);
    }

    #[test]
    fn remaining_timeout_full_when_just_started() {
        let new_secs = 1800;
        let created = 1_000_000_u64;
        let now = created;
        let remaining = compute_remaining_timeout_ms(new_secs, created, now);
        assert_eq!(remaining, 1_800_000);
    }

    #[test]
    fn eager_dispatch_default_is_enabled_at_sane_threshold() {
        let cfg = EouConfig::default();
        assert!(!cfg.eager_disabled());
        assert_eq!(cfg.eager_p_threshold, defaults::eou::EAGER_P_THRESHOLD);
        assert!(cfg.eager_p_threshold > 0.0 && cfg.eager_p_threshold < 1.0);
    }

    #[test]
    fn eager_threshold_gate_passes_at_or_above_threshold() {
        let cfg = EouConfig::default();
        let p_above = 0.6_f32;
        let p_below = 0.3_f32;
        assert!(!cfg.eager_disabled() && p_above >= cfg.eager_p_threshold);
        assert!(!cfg.eager_disabled() && p_below < cfg.eager_p_threshold);
    }

    #[test]
    fn eager_dispatch_throttle_window_uses_eager_interval_ms() {
        let cfg = EouConfig::default();
        let throttle = std::time::Duration::from_millis(cfg.eager_interval_ms as u64);
        let prev = std::time::Instant::now();
        let recent = prev + std::time::Duration::from_millis(50);
        let later = prev + throttle + std::time::Duration::from_millis(1);
        assert!(recent.duration_since(prev) < throttle);
        assert!(later.duration_since(prev) >= throttle);
    }

    #[test]
    fn eager_disabled_sentinel_fully_blocks_dispatch() {
        let mut cfg = EouConfig::default();
        cfg.eager_p_threshold = defaults::eou::EAGER_P_THRESHOLD_DISABLED;
        assert!(cfg.eager_disabled());
        let p = 0.99_f32;
        let gated = !cfg.eager_disabled() && p >= cfg.eager_p_threshold;
        assert!(!gated);
    }

    #[tokio::test]
    async fn promotion_replays_buffered_text_via_oneshot_channel() {
        let predicted = "the cached answer is here.".to_string();
        let (tx, mut rx) = mpsc::channel::<anyhow::Result<String>>(8);
        let predicted_text = predicted.clone();
        tokio::spawn(async move {
            let _ = tx.send(Ok(predicted_text)).await;
        });
        let first = rx.recv().await.unwrap().unwrap();
        assert_eq!(first, predicted);
        assert!(rx.recv().await.is_none());
    }

    fn probe_query() -> RealtimeQuery {
        RealtimeQuery {
            intent: Some("transcription".into()),
            model: None,
            transcription_model: None,
            voice: None,
            speech_model: None,
        }
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    #[ignore = "needs SPEACHES_PLUS_MODELS and REALTIME_DIARIZATION=1"]
    async fn sub_min_speech_commit_spawns_no_diarization_for_unannounced_item() {
        if !super::super::diarization::realtime_enabled() {
            eprintln!("SKIP: REALTIME_DIARIZATION not enabled");
            return;
        }
        let models = match Models::get_or_init() {
            Ok(m) => m,
            Err(e) => {
                eprintln!("SKIP: models unavailable: {e}");
                return;
            }
        };
        let session = Arc::new(Session::with_dependencies(
            probe_query(),
            models,
            Intent::Transcription,
            None,
            Arc::new(crate::ids::CounterIdSource::new()),
            inspector::default_sink(),
        ));
        let (tx, mut rx) = mpsc::channel::<String>(1024);
        session.attach_websocket(tx).await;

        let item_id = "item_sub_min".to_string();
        let diar_guard = session.diarizer.lock().await;
        let before = Arc::strong_count(&session);
        handle_vad_event(
            &session,
            VadEvent::SpeechCommitted {
                item_id: item_id.clone(),
                audio_end_ms: 50,
                audio: vec![0.0f32; 800],
                speech_samples: 800,
            },
        )
        .await;
        let after = Arc::strong_count(&session);
        drop(diar_guard);

        let mut trace = Vec::new();
        while let Ok(t) = rx.try_recv() {
            trace.push(serde_json::from_str::<serde_json::Value>(&t).expect("json"));
        }
        let announced = trace.iter().any(|e| {
            matches!(
                e.get("type").and_then(|t| t.as_str()),
                Some("conversation.item.added") | Some("input_audio_buffer.committed")
            )
        });
        assert!(
            !announced,
            "sub-min-speech commit unexpectedly announced an item: {trace:#?}"
        );
        assert!(
            super::super::order_harness::i4_no_unannounced_item_refs(&trace).is_empty(),
            "I4 violated on the real trace: {trace:#?}"
        );
        assert_eq!(
            after, before,
            "a diarization task was spawned for {item_id}, an item that is never announced"
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    #[ignore = "needs SPEACHES_PLUS_MODELS and REALTIME_DIARIZATION=1"]
    async fn above_min_speech_commit_still_announces_the_item() {
        if !super::super::diarization::realtime_enabled() {
            eprintln!("SKIP: REALTIME_DIARIZATION not enabled");
            return;
        }
        let models = match Models::get_or_init() {
            Ok(m) => m,
            Err(e) => {
                eprintln!("SKIP: models unavailable: {e}");
                return;
            }
        };
        let session = Arc::new(Session::with_dependencies(
            probe_query(),
            models,
            Intent::Transcription,
            None,
            Arc::new(crate::ids::CounterIdSource::new()),
            inspector::default_sink(),
        ));
        let (tx, mut rx) = mpsc::channel::<String>(1024);
        session.attach_websocket(tx).await;

        let item_id = "item_long".to_string();
        let diar_guard = session.diarizer.lock().await;
        let before = Arc::strong_count(&session);
        handle_vad_event(
            &session,
            VadEvent::SpeechCommitted {
                item_id: item_id.clone(),
                audio_end_ms: 1000,
                audio: vec![0.0f32; 16_000],
                speech_samples: 16_000,
            },
        )
        .await;
        let after = Arc::strong_count(&session);
        drop(diar_guard);
        assert!(
            after > before,
            "an above-min-speech commit must keep dispatching work"
        );

        let mut trace = Vec::new();
        let deadline = std::time::Instant::now() + Duration::from_secs(30);
        let mut announced = false;
        while std::time::Instant::now() < deadline && !announced {
            match tokio::time::timeout(Duration::from_secs(5), rx.recv()).await {
                Ok(Some(t)) => {
                    let v = serde_json::from_str::<serde_json::Value>(&t).expect("json");
                    announced = v.get("type").and_then(|t| t.as_str())
                        == Some("input_audio_buffer.committed");
                    trace.push(v);
                }
                _ => break,
            }
        }
        assert!(
            announced,
            "above-min-speech commit never announced its item: {trace:#?}"
        );
        assert!(
            super::super::order_harness::i4_no_unannounced_item_refs(&trace).is_empty(),
            "I4 violated on the real trace: {trace:#?}"
        );
    }
}
