#![allow(dead_code)]

use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, AtomicU64};
use std::sync::Arc;

use tokio::sync::Mutex;
use tokio::task::JoinHandle;
use webrtc::peer_connection::RTCPeerConnection;

use super::transport::EventSink;
use crate::defaults;
use crate::types::{Epoch, ItemId, Millis, ResponseId};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TerminationReason {
    ClientClosed,
    IdleTimeout,
    TransportError,
    MaxDuration,
    InternalStateError,
    VadFailed,
    SttFailed,
    ModelLoadFailed,
}

impl TerminationReason {
    pub fn as_str(self) -> &'static str {
        match self {
            TerminationReason::ClientClosed => "client_closed",
            TerminationReason::IdleTimeout => "idle_timeout",
            TerminationReason::TransportError => "transport_error",
            TerminationReason::MaxDuration => "max_duration",
            TerminationReason::InternalStateError => "internal_state_error",
            TerminationReason::VadFailed => "vad_failed",
            TerminationReason::SttFailed => "stt_failed",
            TerminationReason::ModelLoadFailed => "model_load_failed",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub enum SessionPhase {
    #[default]
    Pending,
    Active {
        created_at_ms: Millis,
    },
    Terminated {
        reason: TerminationReason,
    },
}

impl SessionPhase {
    pub fn is_active(&self) -> bool {
        matches!(self, SessionPhase::Active { .. })
    }

    pub fn is_terminated(&self) -> bool {
        matches!(self, SessionPhase::Terminated { .. })
    }
}

#[derive(Debug, Default)]
pub enum VadPhase {
    #[default]
    Silent,
    Speaking {
        item_id: ItemId,
        audio_start_ms: Millis,
    },
    Stopped {
        item_id: ItemId,
        audio_start_ms: Millis,
        audio_end_ms: Millis,
    },
}

pub struct PredictedSharedState {
    pub user_transcript: Mutex<Option<Result<String, String>>>,
    pub done: tokio::sync::Notify,
}

impl Default for PredictedSharedState {
    fn default() -> Self {
        Self::new()
    }
}

impl PredictedSharedState {
    pub fn new() -> Self {
        Self {
            user_transcript: Mutex::new(None),
            done: tokio::sync::Notify::new(),
        }
    }
}

pub struct PredictedRunner {
    pub task: JoinHandle<()>,
    pub shared: Arc<PredictedSharedState>,
}

impl std::fmt::Debug for PredictedRunner {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PredictedRunner").finish_non_exhaustive()
    }
}

pub struct PredictedLlmRunnerHandle {
    pub task: JoinHandle<()>,
    pub shared: Arc<super::eou_predicted::PredictedLlmShared>,
    pub cap: u32,
}

impl PredictedLlmRunnerHandle {
    pub fn from_runner(r: super::eou_predicted::PredictedLlmRunner) -> Self {
        Self {
            task: r.task,
            shared: r.shared,
            cap: r.cap,
        }
    }

    pub fn into_runner(self) -> super::eou_predicted::PredictedLlmRunner {
        super::eou_predicted::PredictedLlmRunner {
            task: self.task,
            shared: self.shared,
            cap: self.cap,
        }
    }
}

impl std::fmt::Debug for PredictedLlmRunnerHandle {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PredictedLlmRunnerHandle")
            .field("cap", &self.cap)
            .finish_non_exhaustive()
    }
}

#[derive(Debug, Default)]
pub enum RespPhase {
    #[default]
    None,
    Predicted {
        id: ResponseId,
        item_id: ItemId,
        epoch: Epoch,
        eou_score: f32,
        runner: Option<PredictedRunner>,
        llm_runner: Option<PredictedLlmRunnerHandle>,
    },
    Created {
        id: ResponseId,
        item_id: ItemId,
        epoch: Epoch,
    },
    Streaming {
        id: ResponseId,
        item_id: ItemId,
        epoch: Epoch,
        played_ms: Arc<AtomicU64>,
        planned_ms: Option<u64>,
    },
    Drain {
        id: ResponseId,
        item_id: ItemId,
        epoch: Epoch,
        played_ms: Arc<AtomicU64>,
        planned_ms: u64,
    },
}

impl RespPhase {
    pub fn id(&self) -> Option<&ResponseId> {
        match self {
            RespPhase::None => None,
            RespPhase::Predicted { id, .. }
            | RespPhase::Created { id, .. }
            | RespPhase::Streaming { id, .. }
            | RespPhase::Drain { id, .. } => Some(id),
        }
    }

    pub fn item_id(&self) -> Option<&ItemId> {
        match self {
            RespPhase::None => None,
            RespPhase::Predicted { item_id, .. }
            | RespPhase::Created { item_id, .. }
            | RespPhase::Streaming { item_id, .. }
            | RespPhase::Drain { item_id, .. } => Some(item_id),
        }
    }

    pub fn is_active(&self) -> bool {
        !matches!(self, RespPhase::None)
    }

    pub fn is_predicted(&self) -> bool {
        matches!(self, RespPhase::Predicted { .. })
    }

    pub fn epoch(&self) -> Option<Epoch> {
        match self {
            RespPhase::None => None,
            RespPhase::Predicted { epoch, .. }
            | RespPhase::Created { epoch, .. }
            | RespPhase::Streaming { epoch, .. }
            | RespPhase::Drain { epoch, .. } => Some(*epoch),
        }
    }

    pub fn played_ms(&self) -> Option<&Arc<AtomicU64>> {
        match self {
            RespPhase::Streaming { played_ms, .. } | RespPhase::Drain { played_ms, .. } => {
                Some(played_ms)
            }
            _ => None,
        }
    }
}

#[derive(Debug)]
pub struct OpenBuffer {
    pub id: ItemId,
    pub audio: Vec<f32>,
    pub audio_start_ms: Option<Millis>,
}

impl OpenBuffer {
    pub fn new(id: ItemId) -> Self {
        Self {
            id,
            audio: Vec::with_capacity(16_000 * 30),
            audio_start_ms: None,
        }
    }

    pub fn append(&mut self, samples: &[f32]) {
        self.audio.extend_from_slice(samples);
    }

    pub fn seal(self, audio_end_ms: Millis) -> SealedBuffer {
        let audio_start_ms = self.audio_start_ms.map(|m| m.raw()).unwrap_or(0);
        SealedBuffer {
            item_id: self.id.into_string(),
            audio: self.audio,
            audio_start_ms,
            audio_end_ms: audio_end_ms.raw(),
        }
    }
}

pub struct ResponseRuntime {
    pub handle: JoinHandle<()>,
    pub transcript_so_far: Arc<Mutex<String>>,

    pub wire_opened: Arc<AtomicBool>,
}

impl std::fmt::Debug for ResponseRuntime {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ResponseRuntime").finish_non_exhaustive()
    }
}

pub const SEALED_BUFFER_RETENTION_COUNT: usize = defaults::buffer::SEALED_BUFFER_RETENTION_COUNT;

pub struct SealedBuffer {
    pub item_id: String,
    pub audio: Vec<f32>,
    pub audio_start_ms: u64,
    pub audio_end_ms: u64,
}

impl std::fmt::Debug for SealedBuffer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SealedBuffer")
            .field("item_id", &self.item_id)
            .field("audio_len", &self.audio.len())
            .field("audio_start_ms", &self.audio_start_ms)
            .field("audio_end_ms", &self.audio_end_ms)
            .finish()
    }
}

pub struct SessionState {
    pub session: SessionPhase,
    pub vad: VadPhase,
    pub resp: RespPhase,
    pub instructions: Option<String>,
    pub pc: Option<Arc<RTCPeerConnection>>,
    pub event_sink: Option<EventSink>,
    pub current_response: Option<ResponseRuntime>,
    pub last_epoch: Epoch,
    pub timeout_task: Option<JoinHandle<()>>,
    pub commit_timer: Option<JoinHandle<()>>,
    pub bargein_task: Option<JoinHandle<()>>,
    pub pending_bargein: Option<PendingBargein>,
    pub conversation: Vec<ConversationItem>,
    pub current_speech_item: Option<String>,
    pub sealed_buffers: VecDeque<SealedBuffer>,
}

impl SessionState {
    pub fn store_sealed_buffer(&mut self, buf: SealedBuffer) {
        self.sealed_buffers.retain(|b| b.item_id != buf.item_id);
        self.sealed_buffers.push_back(buf);
        while self.sealed_buffers.len() > SEALED_BUFFER_RETENTION_COUNT {
            self.sealed_buffers.pop_front();
        }
    }

    pub fn drop_sealed_buffer(&mut self, item_id: &str) -> bool {
        let before = self.sealed_buffers.len();
        self.sealed_buffers.retain(|b| b.item_id != item_id);
        before != self.sealed_buffers.len()
    }

    pub fn sealed_buffer(&self, item_id: &str) -> Option<&SealedBuffer> {
        self.sealed_buffers.iter().find(|b| b.item_id == item_id)
    }
}

#[derive(Clone, Debug)]
pub struct PendingBargein {
    pub item_id: String,
    pub audio_start_ms: u64,
}

impl Default for SessionState {
    fn default() -> Self {
        Self {
            session: SessionPhase::default(),
            vad: VadPhase::default(),
            resp: RespPhase::default(),
            instructions: None,
            pc: None,
            event_sink: None,
            current_response: None,
            last_epoch: Epoch::zero(),
            timeout_task: None,
            commit_timer: None,
            bargein_task: None,
            pending_bargein: None,
            conversation: Vec::new(),
            current_speech_item: None,
            sealed_buffers: VecDeque::new(),
        }
    }
}

impl std::fmt::Debug for SessionState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SessionState")
            .field("session", &self.session)
            .field("vad", &self.vad)
            .field("resp", &self.resp)
            .field("instructions", &self.instructions.is_some())
            .field("pc", &self.pc.is_some())
            .field("event_sink", &self.event_sink.is_some())
            .field("current_response", &self.current_response)
            .field("last_epoch", &self.last_epoch)
            .finish()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ItemRole {
    User,
    Assistant,
    System,
}

impl ItemRole {
    pub fn as_str(self) -> &'static str {
        match self {
            ItemRole::User => "user",
            ItemRole::Assistant => "assistant",
            ItemRole::System => "system",
        }
    }

    pub fn parse(s: &str) -> Option<Self> {
        match s {
            "user" => Some(ItemRole::User),
            "assistant" => Some(ItemRole::Assistant),
            "system" => Some(ItemRole::System),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ItemStatus {
    InProgress,
    Completed,
    Incomplete,
}

impl ItemStatus {
    pub fn as_str(self) -> &'static str {
        match self {
            ItemStatus::InProgress => "in_progress",
            ItemStatus::Completed => "completed",
            ItemStatus::Incomplete => "incomplete",
        }
    }
}

#[derive(Debug, Clone)]
pub enum ItemContent {
    UserAudio {
        transcript: Option<String>,
        audio_end_ms: Option<u64>,
    },
    AssistantAudio {
        transcript: String,
        audio_ms: u64,
    },
    Text(String),
}

#[derive(Debug, Clone)]
pub struct ConversationItem {
    pub id: String,
    pub role: ItemRole,
    pub status: ItemStatus,
    pub content: ItemContent,
    pub client_speakable: bool,
}

impl ConversationItem {
    pub fn new_user_audio(id: String) -> Self {
        Self {
            id,
            role: ItemRole::User,
            status: ItemStatus::InProgress,
            content: ItemContent::UserAudio {
                transcript: None,
                audio_end_ms: None,
            },
            client_speakable: false,
        }
    }

    pub fn new_assistant_audio(id: String, transcript: String, audio_ms: u64) -> Self {
        Self {
            id,
            role: ItemRole::Assistant,
            status: ItemStatus::Completed,
            content: ItemContent::AssistantAudio {
                transcript,
                audio_ms,
            },
            client_speakable: false,
        }
    }

    pub fn new_text(id: String, role: ItemRole, status: ItemStatus, text: String) -> Self {
        Self {
            id,
            role,
            status,
            content: ItemContent::Text(text),
            client_speakable: false,
        }
    }

    pub fn transcript(&self) -> Option<&str> {
        match &self.content {
            ItemContent::UserAudio { transcript, .. } => transcript.as_deref(),
            ItemContent::AssistantAudio { transcript, .. } => Some(transcript.as_str()),
            ItemContent::Text(t) => Some(t.as_str()),
        }
    }
}

#[derive(Debug, PartialEq, Eq)]
pub enum ResponseSource {
    Speak { index: usize, text: String },
    SpeakUnavailable { reason: &'static str },
    Generate { prompt: String },
}

pub fn select_response_source(
    conv: &[ConversationItem],
    response_has_input: bool,
    speak_item_id: Option<&str>,
) -> ResponseSource {
    if let Some(want) = speak_item_id {
        if response_has_input {
            return ResponseSource::SpeakUnavailable {
                reason: "response.speak_item_id cannot be combined with response.input",
            };
        }
        let Some(index) = conv.iter().position(|i| i.id == want) else {
            return ResponseSource::SpeakUnavailable {
                reason: "response.speak_item_id names no item in this conversation",
            };
        };
        let item = &conv[index];
        if item.role != ItemRole::Assistant {
            return ResponseSource::SpeakUnavailable {
                reason: "response.speak_item_id must name an assistant message",
            };
        }
        if !item.client_speakable {
            return ResponseSource::SpeakUnavailable {
                reason: "that item was not created by this client, or has already been spoken",
            };
        }
        let text = item.transcript().unwrap_or_default();
        if text.trim().is_empty() {
            return ResponseSource::SpeakUnavailable {
                reason: "response.speak_item_id names an item with no text to speak",
            };
        }
        return ResponseSource::Speak {
            index,
            text: text.to_string(),
        };
    }
    let prompt = conv
        .iter()
        .rev()
        .find(|i| matches!(i.role, ItemRole::User))
        .and_then(|i| i.transcript().map(|s| s.to_string()))
        .unwrap_or_default();
    ResponseSource::Generate { prompt }
}

pub fn apply_truncate_to_conversation(
    conversation: &mut Vec<ConversationItem>,
    assistant_item_id: &str,
    played_ms: u64,
    transcript: &str,
) {
    if let Some(item) = conversation.iter_mut().find(|i| i.id == assistant_item_id) {
        item.status = ItemStatus::Incomplete;
        match &mut item.content {
            ItemContent::AssistantAudio {
                audio_ms,
                transcript: t,
            } => {
                *audio_ms = (*audio_ms).min(played_ms);
                *t = transcript.to_string();
            }
            _ => {
                item.content = ItemContent::AssistantAudio {
                    transcript: transcript.to_string(),
                    audio_ms: played_ms,
                };
            }
        }
        return;
    }
    let mut item = ConversationItem::new_assistant_audio(
        assistant_item_id.to_string(),
        transcript.to_string(),
        played_ms,
    );
    item.status = ItemStatus::Incomplete;
    conversation.push(item);
}

#[derive(Debug, PartialEq, Eq)]
pub enum InvariantViolation {
    SpeakingWithActiveResponse,
    StoppedWithoutEnd,
    DrainWithoutPlannedMs,
    EpochRegression {
        from: u64,
        to: u64,
    },
    SealedBufferAppend,
    ResponseRuntimeMismatch,
    IllegalRespTransition {
        from: &'static str,
        to: &'static str,
    },
    EmptyResponseId,
    RotationBeforeCommit {
        item_id: String,
    },
    ConvHasVoiced {
        item_id: String,
    },
}

pub fn check_invariants(
    session: &SessionPhase,
    vad: &VadPhase,
    resp: &RespPhase,
) -> Result<(), InvariantViolation> {
    if matches!(session, SessionPhase::Terminated { .. }) {
        return Ok(());
    }

    if matches!(vad, VadPhase::Speaking { .. })
        && matches!(
            resp,
            RespPhase::Created { .. } | RespPhase::Streaming { .. } | RespPhase::Drain { .. }
        )
    {
        return Err(InvariantViolation::SpeakingWithActiveResponse);
    }

    if let VadPhase::Stopped {
        audio_start_ms,
        audio_end_ms,
        ..
    } = vad
    {
        if audio_end_ms.raw() < audio_start_ms.raw() {
            return Err(InvariantViolation::StoppedWithoutEnd);
        }
    }

    Ok(())
}

pub fn check_state(state: &SessionState) -> Result<(), InvariantViolation> {
    check_invariants(&state.session, &state.vad, &state.resp)?;
    match (&state.resp, &state.current_response) {
        (RespPhase::None, Some(_)) => {
            return Err(InvariantViolation::ResponseRuntimeMismatch);
        }
        (RespPhase::Created { .. }, None)
        | (RespPhase::Streaming { .. }, None)
        | (RespPhase::Drain { .. }, None) => {
            return Err(InvariantViolation::ResponseRuntimeMismatch);
        }
        _ => {}
    }

    if matches!(
        state.resp,
        RespPhase::Created { .. } | RespPhase::Streaming { .. } | RespPhase::Drain { .. }
    ) && state
        .resp
        .id()
        .map(|id| id.as_str().is_empty())
        .unwrap_or(true)
    {
        return Err(InvariantViolation::EmptyResponseId);
    }

    if let VadPhase::Stopped { item_id, .. } = &state.vad {
        if state.sealed_buffer(item_id.as_str()).is_some() {
            return Err(InvariantViolation::RotationBeforeCommit {
                item_id: item_id.as_str().to_string(),
            });
        }
    }

    if let VadPhase::Speaking { item_id, .. } = &state.vad {
        let voiced = state.conversation.iter().any(|it| {
            it.id.as_str() == item_id.as_str()
                && it.role == ItemRole::User
                && it.status == ItemStatus::InProgress
        });
        if voiced {
            return Err(InvariantViolation::ConvHasVoiced {
                item_id: item_id.as_str().to_string(),
            });
        }
    }

    Ok(())
}

#[derive(Debug)]
pub struct ResponseGate(());

impl ResponseGate {
    pub fn open(resp: &RespPhase) -> Option<Self> {
        match resp {
            RespPhase::Predicted { .. } => None,
            _ => Some(Self(())),
        }
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum Topic {
    Session,
    Item,
    Buffer,
    Response,
    Error,
    Other,
}

impl Topic {
    pub fn classify(event_type: &str) -> Self {
        if event_type.starts_with("response.") {
            Topic::Response
        } else if event_type.starts_with("session.") || event_type.starts_with("rate_limits.") {
            Topic::Session
        } else if event_type.starts_with("conversation.item.") {
            Topic::Item
        } else if event_type.starts_with("input_audio_buffer.")
            || event_type.starts_with("output_audio_buffer.")
        {
            Topic::Buffer
        } else if event_type == "error" {
            Topic::Error
        } else {
            Topic::Other
        }
    }
}

impl SessionState {
    pub fn resp_create_from_none(
        &mut self,
        id: ResponseId,
        item_id: ItemId,
        runtime: ResponseRuntime,
    ) -> Result<Epoch, InvariantViolation> {
        if !matches!(self.resp, RespPhase::None) {
            return Err(InvariantViolation::IllegalRespTransition {
                from: phase_kind(&self.resp),
                to: "Created",
            });
        }
        if matches!(self.vad, VadPhase::Speaking { .. }) {
            return Err(InvariantViolation::SpeakingWithActiveResponse);
        }
        let epoch = self.last_epoch.next();
        self.last_epoch = epoch;
        self.resp = RespPhase::Created { id, item_id, epoch };
        self.current_response = Some(runtime);
        check_state(self)?;
        Ok(epoch)
    }

    pub fn resp_start_predicted(
        &mut self,
        id: ResponseId,
        item_id: ItemId,
        eou_score: f32,
        runner: Option<PredictedRunner>,
    ) -> Result<Epoch, InvariantViolation> {
        self.resp_start_predicted_with_llm(id, item_id, eou_score, runner, None)
    }

    pub fn resp_start_predicted_with_llm(
        &mut self,
        id: ResponseId,
        item_id: ItemId,
        eou_score: f32,
        runner: Option<PredictedRunner>,
        llm_runner: Option<PredictedLlmRunnerHandle>,
    ) -> Result<Epoch, InvariantViolation> {
        if !matches!(self.resp, RespPhase::None) {
            return Err(InvariantViolation::IllegalRespTransition {
                from: phase_kind(&self.resp),
                to: "Predicted",
            });
        }
        if matches!(self.vad, VadPhase::Speaking { .. }) {
            return Err(InvariantViolation::SpeakingWithActiveResponse);
        }
        let epoch = self.last_epoch.next();
        self.last_epoch = epoch;
        self.resp = RespPhase::Predicted {
            id,
            item_id,
            epoch,
            eou_score,
            runner,
            llm_runner,
        };
        check_state(self)?;
        Ok(epoch)
    }

    pub fn resp_attach_predicted_llm(
        &mut self,
        llm_runner: PredictedLlmRunnerHandle,
    ) -> Result<(), InvariantViolation> {
        if let RespPhase::Predicted {
            llm_runner: slot, ..
        } = &mut self.resp
        {
            if slot.is_none() {
                *slot = Some(llm_runner);
                return Ok(());
            }
            return Err(InvariantViolation::IllegalRespTransition {
                from: "Predicted(with-llm)",
                to: "Predicted(with-llm)",
            });
        }
        Err(InvariantViolation::IllegalRespTransition {
            from: phase_kind(&self.resp),
            to: "Predicted(attach_llm)",
        })
    }

    #[allow(clippy::type_complexity)]
    pub fn resp_promote_predicted_to_created(
        &mut self,
        runtime: ResponseRuntime,
    ) -> Result<
        (
            ResponseId,
            ItemId,
            Epoch,
            Option<PredictedRunner>,
            Option<PredictedLlmRunnerHandle>,
        ),
        InvariantViolation,
    > {
        let taken = std::mem::replace(&mut self.resp, RespPhase::None);
        match taken {
            RespPhase::Predicted {
                id,
                item_id,
                epoch,
                eou_score: _,
                runner,
                llm_runner,
            } => {
                self.resp = RespPhase::Created {
                    id: id.clone(),
                    item_id: item_id.clone(),
                    epoch,
                };
                self.current_response = Some(runtime);
                check_state(self)?;
                Ok((id, item_id, epoch, runner, llm_runner))
            }
            other => {
                let kind = phase_kind(&other);
                self.resp = other;
                Err(InvariantViolation::IllegalRespTransition {
                    from: kind,
                    to: "Created (from Predicted)",
                })
            }
        }
    }

    pub fn resp_advance_to_streaming(
        &mut self,
        played_ms: Arc<AtomicU64>,
    ) -> Result<(), InvariantViolation> {
        let taken = std::mem::replace(&mut self.resp, RespPhase::None);
        match taken {
            RespPhase::Created { id, item_id, epoch } => {
                self.resp = RespPhase::Streaming {
                    id,
                    item_id,
                    epoch,
                    played_ms,
                    planned_ms: None,
                };
                check_state(self)?;
                Ok(())
            }
            other => {
                let kind = phase_kind(&other);
                self.resp = other;
                Err(InvariantViolation::IllegalRespTransition {
                    from: kind,
                    to: "Streaming",
                })
            }
        }
    }

    pub fn resp_drain(&mut self, planned_ms: u64) -> Result<(), InvariantViolation> {
        let taken = std::mem::replace(&mut self.resp, RespPhase::None);
        match taken {
            RespPhase::Streaming {
                id,
                item_id,
                epoch,
                played_ms,
                ..
            } => {
                self.resp = RespPhase::Drain {
                    id,
                    item_id,
                    epoch,
                    played_ms,
                    planned_ms,
                };
                check_state(self)?;
                Ok(())
            }
            other => {
                let kind = phase_kind(&other);
                self.resp = other;
                Err(InvariantViolation::IllegalRespTransition {
                    from: kind,
                    to: "Drain",
                })
            }
        }
    }

    pub fn resp_retire_to_none(&mut self) -> Result<Option<ResponseRuntime>, InvariantViolation> {
        self.resp = RespPhase::None;
        let runtime = self.current_response.take();
        check_state(self)?;
        Ok(runtime)
    }

    pub fn resp_retire_predicted(&mut self) -> Result<Option<PredictedRunner>, InvariantViolation> {
        self.resp_retire_predicted_full().map(|(s, _)| s)
    }

    pub fn resp_retire_predicted_full(
        &mut self,
    ) -> Result<(Option<PredictedRunner>, Option<PredictedLlmRunnerHandle>), InvariantViolation>
    {
        let taken = std::mem::replace(&mut self.resp, RespPhase::None);
        match taken {
            RespPhase::Predicted {
                runner, llm_runner, ..
            } => {
                check_state(self)?;
                Ok((runner, llm_runner))
            }
            other => {
                let kind = phase_kind(&other);
                self.resp = other;
                Err(InvariantViolation::IllegalRespTransition {
                    from: kind,
                    to: "None (from Predicted)",
                })
            }
        }
    }
}

fn phase_kind(p: &RespPhase) -> &'static str {
    match p {
        RespPhase::None => "None",
        RespPhase::Predicted { .. } => "Predicted",
        RespPhase::Created { .. } => "Created",
        RespPhase::Streaming { .. } => "Streaming",
        RespPhase::Drain { .. } => "Drain",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rid(s: &str) -> ResponseId {
        ResponseId::new(s)
    }

    fn iid(s: &str) -> ItemId {
        ItemId::new(s)
    }

    fn ep(n: u64) -> Epoch {
        Epoch(n)
    }

    #[test]
    fn invariant_speaking_with_active_response() {
        let v = VadPhase::Speaking {
            item_id: ItemId::new("item_x".to_string()),
            audio_start_ms: Millis(0),
        };
        let r = RespPhase::Created {
            id: rid("resp_x"),
            item_id: iid("item_x"),
            epoch: ep(1),
        };
        assert_eq!(
            check_invariants(
                &SessionPhase::Active {
                    created_at_ms: Millis(0)
                },
                &v,
                &r
            ),
            Err(InvariantViolation::SpeakingWithActiveResponse),
        );
    }

    #[test]
    fn invariant_silent_during_response_is_fine() {
        let r = RespPhase::Streaming {
            id: rid("resp_x"),
            item_id: iid("item_a"),
            epoch: ep(1),
            played_ms: Arc::new(AtomicU64::new(0)),
            planned_ms: None,
        };
        assert!(check_invariants(
            &SessionPhase::Active {
                created_at_ms: Millis(0)
            },
            &VadPhase::Silent,
            &r
        )
        .is_ok());
    }

    #[test]
    fn invariant_resp_id_observable() {
        let r = RespPhase::Created {
            id: rid("resp_z"),
            item_id: iid("item_a"),
            epoch: ep(7),
        };
        assert_eq!(r.id().map(|i| i.as_str()), Some("resp_z"));
        assert_eq!(r.item_id().map(|i| i.as_str()), Some("item_a"));
        assert!(r.is_active());
        assert!(!RespPhase::None.is_active());
    }

    #[test]
    fn conversation_item_helpers() {
        let u = ConversationItem::new_user_audio("item_u".into());
        assert_eq!(u.role, ItemRole::User);
        assert_eq!(u.status, ItemStatus::InProgress);
        assert_eq!(u.transcript(), None);

        let a = ConversationItem::new_assistant_audio("item_a".into(), "hi there".into(), 1200);
        assert_eq!(a.role, ItemRole::Assistant);
        assert_eq!(a.status, ItemStatus::Completed);
        assert_eq!(a.transcript(), Some("hi there"));

        assert_eq!(ItemRole::parse("user"), Some(ItemRole::User));
        assert_eq!(ItemRole::parse("nope"), None);
        assert_eq!(ItemRole::User.as_str(), "user");
        assert_eq!(ItemStatus::Incomplete.as_str(), "incomplete");
    }

    #[test]
    fn open_buffer_seal_consumes_self() {
        let mut b = OpenBuffer::new(iid("item_a"));
        b.append(&[0.0; 8]);
        let s = b.seal(Millis(120));
        assert_eq!(s.audio_end_ms, 120);
        assert_eq!(s.audio.len(), 8);
    }

    #[test]
    fn termination_reasons_map_to_distinct_wire_strings() {
        let all = [
            TerminationReason::ClientClosed,
            TerminationReason::IdleTimeout,
            TerminationReason::TransportError,
            TerminationReason::MaxDuration,
            TerminationReason::InternalStateError,
            TerminationReason::VadFailed,
            TerminationReason::SttFailed,
            TerminationReason::ModelLoadFailed,
        ];
        let mut seen: Vec<&str> = all.iter().map(|r| r.as_str()).collect();
        let len = seen.len();
        seen.sort();
        seen.dedup();
        assert_eq!(len, seen.len(), "duplicate session.done reason: {seen:?}");
        assert_eq!(TerminationReason::IdleTimeout.as_str(), "idle_timeout");
        assert_eq!(
            TerminationReason::TransportError.as_str(),
            "transport_error"
        );
    }

    #[test]
    fn invariant_terminated_session_short_circuits() {
        let v = VadPhase::Speaking {
            item_id: ItemId::new("item_x".to_string()),
            audio_start_ms: Millis(0),
        };
        let r = RespPhase::Created {
            id: rid("resp_x"),
            item_id: iid("item_x"),
            epoch: ep(1),
        };
        assert!(check_invariants(
            &SessionPhase::Terminated {
                reason: TerminationReason::ClientClosed
            },
            &v,
            &r
        )
        .is_ok());
    }

    #[test]
    fn vad_phase_speaking_carries_typed_item_id() {
        let v = VadPhase::Speaking {
            item_id: ItemId::new("item_a"),
            audio_start_ms: Millis(0),
        };
        assert!(check_invariants(
            &SessionPhase::Active {
                created_at_ms: Millis(0)
            },
            &v,
            &RespPhase::None
        )
        .is_ok());
    }

    #[test]
    fn invariant_stopped_with_end_before_start() {
        let v = VadPhase::Stopped {
            item_id: ItemId::new("item_x".to_string()),
            audio_start_ms: Millis(500),
            audio_end_ms: Millis(100),
        };
        assert_eq!(
            check_invariants(
                &SessionPhase::Active {
                    created_at_ms: Millis(0)
                },
                &v,
                &RespPhase::None
            ),
            Err(InvariantViolation::StoppedWithoutEnd),
        );
    }

    #[test]
    fn valid_full_turn_transitions() {
        assert!(
            check_invariants(&SessionPhase::Pending, &VadPhase::Silent, &RespPhase::None).is_ok()
        );
        assert!(check_invariants(
            &SessionPhase::Active {
                created_at_ms: Millis(0)
            },
            &VadPhase::Silent,
            &RespPhase::None
        )
        .is_ok());
        let speaking = VadPhase::Speaking {
            item_id: ItemId::new("item_a".to_string()),
            audio_start_ms: Millis(0),
        };
        assert!(check_invariants(
            &SessionPhase::Active {
                created_at_ms: Millis(0)
            },
            &speaking,
            &RespPhase::None
        )
        .is_ok());
        let stopped = VadPhase::Stopped {
            item_id: ItemId::new("item_a".to_string()),
            audio_start_ms: Millis(0),
            audio_end_ms: Millis(1500),
        };
        assert!(check_invariants(
            &SessionPhase::Active {
                created_at_ms: Millis(0)
            },
            &stopped,
            &RespPhase::None
        )
        .is_ok());
        let r_created = RespPhase::Created {
            id: rid("resp_1"),
            item_id: iid("item_a"),
            epoch: ep(1),
        };
        assert!(check_invariants(
            &SessionPhase::Active {
                created_at_ms: Millis(0)
            },
            &VadPhase::Silent,
            &r_created
        )
        .is_ok());
        let r_streaming = RespPhase::Streaming {
            id: rid("resp_1"),
            item_id: iid("item_a"),
            epoch: ep(1),
            played_ms: Arc::new(AtomicU64::new(120)),
            planned_ms: Some(2000),
        };
        assert!(check_invariants(
            &SessionPhase::Active {
                created_at_ms: Millis(0)
            },
            &VadPhase::Silent,
            &r_streaming
        )
        .is_ok());
    }

    #[test]
    fn barge_in_path_violates_then_recovers() {
        let speaking = VadPhase::Speaking {
            item_id: ItemId::new("item_b".to_string()),
            audio_start_ms: Millis(1000),
        };
        let streaming = RespPhase::Streaming {
            id: rid("resp_2"),
            item_id: iid("item_b"),
            epoch: ep(2),
            played_ms: Arc::new(AtomicU64::new(50)),
            planned_ms: None,
        };
        assert_eq!(
            check_invariants(
                &SessionPhase::Active {
                    created_at_ms: Millis(0)
                },
                &speaking,
                &streaming
            ),
            Err(InvariantViolation::SpeakingWithActiveResponse),
        );
        assert!(check_invariants(
            &SessionPhase::Active {
                created_at_ms: Millis(0)
            },
            &speaking,
            &RespPhase::None
        )
        .is_ok());
    }

    fn dummy_runtime() -> ResponseRuntime {
        let rt = tokio::runtime::Builder::new_current_thread()
            .build()
            .expect("rt");
        let handle: JoinHandle<()> = rt.spawn(async {});
        ResponseRuntime {
            handle,
            transcript_so_far: Arc::new(tokio::sync::Mutex::new(String::new())),
            wire_opened: Arc::new(AtomicBool::new(false)),
        }
    }

    #[test]
    fn check_state_detects_orphan_runtime_handle() {
        let mut state = SessionState::default();
        state.session = SessionPhase::Active {
            created_at_ms: Millis(0),
        };
        state.resp = RespPhase::None;
        state.current_response = Some(dummy_runtime());
        assert_eq!(
            check_state(&state),
            Err(InvariantViolation::ResponseRuntimeMismatch),
        );
    }

    #[test]
    fn check_state_detects_missing_runtime_for_active_phase() {
        let mut state = SessionState::default();
        state.session = SessionPhase::Active {
            created_at_ms: Millis(0),
        };
        state.resp = RespPhase::Created {
            id: rid("resp_X"),
            item_id: iid("item_a"),
            epoch: ep(1),
        };
        state.current_response = None;
        assert_eq!(
            check_state(&state),
            Err(InvariantViolation::ResponseRuntimeMismatch),
        );
    }

    #[test]
    fn check_state_detects_empty_response_id() {
        let mut state = SessionState::default();
        state.session = SessionPhase::Active {
            created_at_ms: Millis(0),
        };
        state.resp = RespPhase::Created {
            id: rid(""),
            item_id: iid("item_a"),
            epoch: ep(1),
        };
        state.current_response = Some(dummy_runtime());
        assert_eq!(
            check_state(&state),
            Err(InvariantViolation::EmptyResponseId)
        );
    }

    #[test]
    fn check_state_detects_rotation_before_commit() {
        let mut state = SessionState::default();
        state.session = SessionPhase::Active {
            created_at_ms: Millis(0),
        };
        state.vad = VadPhase::Stopped {
            item_id: iid("item_x"),
            audio_start_ms: Millis(0),
            audio_end_ms: Millis(1000),
        };
        state.store_sealed_buffer(SealedBuffer {
            item_id: "item_x".to_string(),
            audio: Vec::new(),
            audio_start_ms: 0,
            audio_end_ms: 1000,
        });
        assert_eq!(
            check_state(&state),
            Err(InvariantViolation::RotationBeforeCommit {
                item_id: "item_x".to_string()
            }),
        );
    }

    #[test]
    fn check_state_detects_conv_has_voiced() {
        let mut state = SessionState::default();
        state.session = SessionPhase::Active {
            created_at_ms: Millis(0),
        };
        state.vad = VadPhase::Speaking {
            item_id: iid("item_x"),
            audio_start_ms: Millis(0),
        };
        state
            .conversation
            .push(ConversationItem::new_user_audio("item_x".into()));
        assert_eq!(
            check_state(&state),
            Err(InvariantViolation::ConvHasVoiced {
                item_id: "item_x".to_string()
            }),
        );
    }

    #[test]
    fn check_state_allows_speaking_with_predicted() {
        let mut state = SessionState::default();
        state.session = SessionPhase::Active {
            created_at_ms: Millis(0),
        };
        state.vad = VadPhase::Speaking {
            item_id: iid("item_x"),
            audio_start_ms: Millis(0),
        };
        state.resp = RespPhase::Predicted {
            id: rid("resp_pred"),
            item_id: iid("item_p"),
            epoch: ep(1),
            eou_score: 0.9,
            runner: None,
            llm_runner: None,
        };
        assert!(check_state(&state).is_ok());
    }

    #[test]
    fn predicted_phase_observable() {
        let p = RespPhase::Predicted {
            id: rid("resp_pred"),
            item_id: iid("item_p"),
            epoch: ep(3),
            eou_score: 0.9,
            runner: None,
            llm_runner: None,
        };
        assert_eq!(p.id().map(|i| i.as_str()), Some("resp_pred"));
        assert_eq!(p.item_id().map(|i| i.as_str()), Some("item_p"));
        assert_eq!(p.epoch().map(|e| e.raw()), Some(3));
        assert!(p.is_active());
        assert!(p.is_predicted());
        assert!(!RespPhase::None.is_predicted());
        let c = RespPhase::Created {
            id: rid("resp_c"),
            item_id: iid("item_c"),
            epoch: ep(4),
        };
        assert!(!c.is_predicted());
    }

    #[test]
    fn transition_create_from_none_increments_epoch() {
        let mut state = SessionState::default();
        state.session = SessionPhase::Active {
            created_at_ms: Millis(0),
        };
        let epoch = state
            .resp_create_from_none(rid("r"), iid("a"), dummy_runtime())
            .expect("transition ok");
        assert_eq!(epoch.raw(), 1);
        assert!(matches!(state.resp, RespPhase::Created { .. }));
        assert!(state.current_response.is_some());
    }

    #[test]
    fn transition_predicted_promote_preserves_id_epoch() {
        let mut state = SessionState::default();
        state.session = SessionPhase::Active {
            created_at_ms: Millis(0),
        };
        let _ = state
            .resp_start_predicted(rid("r_p"), iid("a"), 0.95, None)
            .expect("predicted");
        let (id, item_id, epoch, _runner, _llm) = state
            .resp_promote_predicted_to_created(dummy_runtime())
            .expect("promote");
        assert_eq!(id.as_str(), "r_p");
        assert_eq!(item_id.as_str(), "a");
        assert_eq!(epoch.raw(), 1);
        assert!(matches!(state.resp, RespPhase::Created { .. }));
    }

    #[test]
    fn transition_advance_to_streaming_carries_played_ms() {
        let mut state = SessionState::default();
        state.session = SessionPhase::Active {
            created_at_ms: Millis(0),
        };
        let _ = state
            .resp_create_from_none(rid("r"), iid("a"), dummy_runtime())
            .unwrap();
        let played = Arc::new(AtomicU64::new(0));
        state
            .resp_advance_to_streaming(played.clone())
            .expect("streaming");
        match &state.resp {
            RespPhase::Streaming { played_ms, .. } => {
                assert!(Arc::ptr_eq(played_ms, &played));
            }
            other => panic!("expected Streaming, got {other:?}"),
        }
    }

    #[test]
    fn transition_illegal_from_none_to_streaming_rejected() {
        let mut state = SessionState::default();
        state.session = SessionPhase::Active {
            created_at_ms: Millis(0),
        };
        let r = state.resp_advance_to_streaming(Arc::new(AtomicU64::new(0)));
        assert!(matches!(
            r,
            Err(InvariantViolation::IllegalRespTransition { from: "None", .. })
        ));
    }

    #[test]
    fn transition_drain_requires_streaming() {
        let mut state = SessionState::default();
        state.session = SessionPhase::Active {
            created_at_ms: Millis(0),
        };
        let _ = state
            .resp_create_from_none(rid("r"), iid("a"), dummy_runtime())
            .unwrap();
        let r = state.resp_drain(1500);
        assert!(matches!(
            r,
            Err(InvariantViolation::IllegalRespTransition {
                from: "Created",
                ..
            })
        ));
        state
            .resp_advance_to_streaming(Arc::new(AtomicU64::new(0)))
            .unwrap();
        state.resp_drain(1500).expect("drain ok");
        assert!(matches!(state.resp, RespPhase::Drain { .. }));
    }

    #[test]
    fn transition_retire_predicted_returns_runner() {
        let mut state = SessionState::default();
        state.session = SessionPhase::Active {
            created_at_ms: Millis(0),
        };
        let _ = state
            .resp_start_predicted(rid("r"), iid("a"), 0.9, None)
            .unwrap();
        let runner = state.resp_retire_predicted().expect("retire");
        assert!(runner.is_none());
        assert!(matches!(state.resp, RespPhase::None));
    }

    #[test]
    fn transition_retire_predicted_rejected_outside_predicted() {
        let mut state = SessionState::default();
        state.session = SessionPhase::Active {
            created_at_ms: Millis(0),
        };
        let r = state.resp_retire_predicted();
        assert!(matches!(
            r,
            Err(InvariantViolation::IllegalRespTransition {
                to: "None (from Predicted)",
                ..
            })
        ));
    }

    #[test]
    fn predicted_phase_observable_with_llm_runner_field() {
        let p = RespPhase::Predicted {
            id: rid("resp_pred2"),
            item_id: iid("item_p2"),
            epoch: ep(7),
            eou_score: 0.91,
            runner: None,
            llm_runner: None,
        };
        assert_eq!(p.id().map(|i| i.as_str()), Some("resp_pred2"));
        assert!(p.is_predicted());
    }

    #[test]
    fn transition_promote_returns_optional_llm_handle() {
        let mut state = SessionState::default();
        state.session = SessionPhase::Active {
            created_at_ms: Millis(0),
        };
        let _ = state
            .resp_start_predicted_with_llm(rid("r_eager"), iid("a"), 0.85, None, None)
            .expect("start with llm slot");
        let (id, _, _, _runner, llm) = state
            .resp_promote_predicted_to_created(dummy_runtime())
            .expect("promote");
        assert_eq!(id.as_str(), "r_eager");
        assert!(llm.is_none());
        assert!(matches!(state.resp, RespPhase::Created { .. }));
    }

    #[test]
    fn epoch_monotonic_across_rollback_then_new_dispatch() {
        let mut state = SessionState::default();
        state.session = SessionPhase::Active {
            created_at_ms: Millis(0),
        };
        let e1 = state
            .resp_start_predicted(rid("r1"), iid("a"), 0.6, None)
            .expect("first predicted");
        let _ = state.resp_retire_predicted_full().expect("rollback");
        let e2 = state
            .resp_start_predicted(rid("r2"), iid("b"), 0.8, None)
            .expect("second predicted");
        assert!(e2.raw() > e1.raw(), "epoch must be strictly increasing");
    }

    #[test]
    fn retire_predicted_full_returns_pair_of_runners() {
        let mut state = SessionState::default();
        state.session = SessionPhase::Active {
            created_at_ms: Millis(0),
        };
        let _ = state
            .resp_start_predicted_with_llm(rid("r"), iid("a"), 0.9, None, None)
            .expect("predicted");
        let (stt, llm) = state.resp_retire_predicted_full().expect("retire full");
        assert!(stt.is_none());
        assert!(llm.is_none());
        assert!(matches!(state.resp, RespPhase::None));
    }

    #[test]
    fn i7_response_gate_closes_during_predicted_with_llm_handle() {
        let p = RespPhase::Predicted {
            id: rid("r_pred"),
            item_id: iid("a"),
            epoch: ep(2),
            eou_score: 0.7,
            runner: None,
            llm_runner: None,
        };
        assert!(
            ResponseGate::open(&p).is_none(),
            "I7 must reject wire-emit gate while Predicted"
        );
        let c = RespPhase::Created {
            id: rid("r_pred"),
            item_id: iid("a"),
            epoch: ep(2),
        };
        assert!(
            ResponseGate::open(&c).is_some(),
            "wire-emit allowed once promoted to Created"
        );
    }

    #[test]
    fn truncate_clamps_existing_assistant_item() {
        let mut conv = vec![ConversationItem::new_assistant_audio(
            "item_a".into(),
            "the full response".into(),
            5_000,
        )];

        assert_eq!(conv[0].status, ItemStatus::Completed);
        apply_truncate_to_conversation(&mut conv, "item_a", 1_200, "the full");
        assert_eq!(conv.len(), 1);
        assert_eq!(conv[0].status, ItemStatus::Incomplete);
        match &conv[0].content {
            ItemContent::AssistantAudio {
                transcript,
                audio_ms,
            } => {
                assert_eq!(audio_ms, &1_200);
                assert_eq!(transcript, "the full");
            }
            other => panic!("expected AssistantAudio, got {:?}", other),
        }
    }

    #[test]
    fn truncate_clamp_never_inflates() {
        let mut conv = vec![ConversationItem::new_assistant_audio(
            "item_a".into(),
            "hi".into(),
            500,
        )];
        apply_truncate_to_conversation(&mut conv, "item_a", 9_999, "hi");
        match &conv[0].content {
            ItemContent::AssistantAudio { audio_ms, .. } => {
                assert_eq!(audio_ms, &500, "must not inflate beyond existing audio_ms");
            }
            _ => unreachable!(),
        }
        assert_eq!(conv[0].status, ItemStatus::Incomplete);
    }

    #[test]
    fn truncate_appends_when_item_missing() {
        let mut conv: Vec<ConversationItem> = Vec::new();
        apply_truncate_to_conversation(&mut conv, "item_a", 800, "partial text");
        assert_eq!(conv.len(), 1);
        assert_eq!(conv[0].id, "item_a");
        assert_eq!(conv[0].role, ItemRole::Assistant);
        assert_eq!(conv[0].status, ItemStatus::Incomplete);
        match &conv[0].content {
            ItemContent::AssistantAudio {
                transcript,
                audio_ms,
            } => {
                assert_eq!(audio_ms, &800);
                assert_eq!(transcript, "partial text");
            }
            _ => unreachable!(),
        }
    }

    fn user_text(id: &str, text: &str) -> ConversationItem {
        ConversationItem::new_text(
            id.into(),
            ItemRole::User,
            ItemStatus::Completed,
            text.into(),
        )
    }

    fn speakable(id: &str, text: &str) -> ConversationItem {
        let mut item = ConversationItem::new_text(
            id.into(),
            ItemRole::Assistant,
            ItemStatus::Completed,
            text.into(),
        );
        item.client_speakable = true;
        item
    }

    #[test]
    fn constructors_default_client_speakable_false() {
        assert!(!ConversationItem::new_user_audio("item_u".into()).client_speakable);
        assert!(
            !ConversationItem::new_assistant_audio("item_a".into(), "hi".into(), 100)
                .client_speakable
        );
        assert!(
            !ConversationItem::new_text(
                "item_t".into(),
                ItemRole::Assistant,
                ItemStatus::Completed,
                "hi".into()
            )
            .client_speakable
        );
    }

    #[test]
    fn speak_requires_an_explicit_item_id() {
        let conv = vec![
            user_text("item_1", "when did the build finish"),
            speakable("item_3", "The build finished in forty seconds."),
        ];
        assert_eq!(
            select_response_source(&conv, false, Some("item_3")),
            ResponseSource::Speak {
                index: 1,
                text: "The build finished in forty seconds.".to_string(),
            }
        );
    }

    #[test]
    fn a_bare_response_create_always_generates() {
        let conv = vec![
            user_text("item_1", "when did the build finish"),
            speakable("item_3", "The build finished in forty seconds."),
        ];
        assert_eq!(
            select_response_source(&conv, false, None),
            ResponseSource::Generate {
                prompt: "when did the build finish".to_string(),
            },
            "a client restoring a session ends on an assistant item and sends a bare \
             response.create; it must still generate, exactly as before this feature"
        );
    }

    #[test]
    fn speak_item_id_may_name_an_older_item() {
        let conv = vec![
            speakable("item_1", "first sentence"),
            user_text("item_2", "and then"),
        ];
        assert_eq!(
            select_response_source(&conv, false, Some("item_1")),
            ResponseSource::Speak {
                index: 0,
                text: "first sentence".to_string(),
            }
        );
    }

    #[test]
    fn server_authored_assistant_item_never_speaks() {
        let conv = vec![
            user_text("item_1", "what about monday"),
            ConversationItem::new_assistant_audio("item_2".into(), "monday is free".into(), 900),
        ];
        assert_eq!(
            select_response_source(&conv, false, Some("item_2")),
            ResponseSource::SpeakUnavailable {
                reason: "that item was not created by this client, or has already been spoken",
            }
        );
    }

    #[test]
    fn consumed_item_does_not_respeak() {
        let mut conv = vec![user_text("item_1", "status"), speakable("item_2", "done")];
        conv[1].client_speakable = false;
        assert_eq!(
            select_response_source(&conv, false, Some("item_2")),
            ResponseSource::SpeakUnavailable {
                reason: "that item was not created by this client, or has already been spoken",
            }
        );
    }

    #[test]
    fn unknown_speak_item_id_is_refused_not_generated() {
        let conv = vec![user_text("item_1", "status"), speakable("item_2", "done")];
        assert_eq!(
            select_response_source(&conv, false, Some("item_nope")),
            ResponseSource::SpeakUnavailable {
                reason: "response.speak_item_id names no item in this conversation",
            },
            "silently generating instead would speak the wrong thing"
        );
    }

    #[test]
    fn speak_item_id_naming_a_user_item_is_refused() {
        let conv = vec![user_text("item_1", "status")];
        assert_eq!(
            select_response_source(&conv, false, Some("item_1")),
            ResponseSource::SpeakUnavailable {
                reason: "response.speak_item_id must name an assistant message",
            }
        );
    }

    #[test]
    fn response_input_forces_generate() {
        let conv = vec![user_text("item_1", "status"), speakable("item_2", "done")];
        assert_eq!(
            select_response_source(&conv, true, None),
            ResponseSource::Generate {
                prompt: "status".to_string(),
            }
        );
    }

    #[test]
    fn response_input_with_speak_item_id_is_refused() {
        let conv = vec![user_text("item_1", "status"), speakable("item_2", "done")];
        assert_eq!(
            select_response_source(&conv, true, Some("item_2")),
            ResponseSource::SpeakUnavailable {
                reason: "response.speak_item_id cannot be combined with response.input",
            }
        );
    }

    #[test]
    fn whitespace_only_assistant_item_is_refused() {
        let conv = vec![user_text("item_1", "status"), speakable("item_2", "  \n ")];
        assert_eq!(
            select_response_source(&conv, false, Some("item_2")),
            ResponseSource::SpeakUnavailable {
                reason: "response.speak_item_id names an item with no text to speak",
            }
        );
    }

    #[test]
    fn empty_conversation_generates_empty_prompt() {
        let conv: Vec<ConversationItem> = Vec::new();
        assert_eq!(
            select_response_source(&conv, false, None),
            ResponseSource::Generate {
                prompt: String::new(),
            }
        );
    }

    #[test]
    fn speak_preserves_untrimmed_text() {
        let conv = vec![speakable("item_1", "  hello there  ")];
        assert_eq!(
            select_response_source(&conv, false, Some("item_1")),
            ResponseSource::Speak {
                index: 0,
                text: "  hello there  ".to_string(),
            }
        );
    }
}
