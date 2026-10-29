use std::collections::HashMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, OnceLock, Weak};
use std::time::Duration;

use anyhow::Context;
use tracing::{debug, info, warn};
use webrtc::data_channel::DataChannel;
use webrtc::media_stream::track_local::TrackLocal;
use webrtc::media_stream::track_remote::TrackRemote;
use webrtc::peer_connection::{
    register_default_interceptors, MediaEngine, PeerConnection, PeerConnectionBuilder,
    PeerConnectionEventHandler, RTCConfigurationBuilder, RTCIceConnectionState,
    RTCIceGatheringState, RTCPeerConnectionState, RTCSessionDescription, Registry,
};

use crate::RealtimeQuery;

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Intent {
    Transcription,
    Conversation,
}

impl Intent {
    fn from_query(q: &RealtimeQuery) -> Self {
        match q.intent.as_deref() {
            Some("conversation") => Intent::Conversation,
            _ => Intent::Transcription,
        }
    }
}

mod audio_in;
mod audio_in_ws;
mod audio_out;
mod audio_out_ws;
mod cancel;
mod diarization;
mod eou_eager;
mod eou_predicted;
mod events;
mod framing;
#[cfg(test)]
mod fuzz;
mod inspector;
#[cfg(test)]
mod order_harness;
mod pipeline;
mod sdp_filter;
mod session;
mod session_update;
pub mod speaker_id;
pub mod state;
mod transport;
mod v2_compat;
pub mod websocket;
mod wire;

use crate::models::Models;
pub use session::Session;

pub const SESSION_MAX_DURATION_S: u64 = crate::defaults::session::MAX_DURATION_S;
pub const RFC_VERSION: &str = crate::defaults::RFC_VERSION;

#[allow(dead_code)]
pub fn capabilities_json() -> serde_json::Value {
    capabilities_json_inner(None)
}

pub fn capabilities_json_with_models(models: &Models) -> serde_json::Value {
    capabilities_json_inner(Some(models))
}

fn capabilities_json_inner(models: Option<&Models>) -> serde_json::Value {
    use crate::eou::EouKind;
    let spec_kinds: Vec<&'static str> = EouKind::V3_SPEC.iter().map(|k| k.as_str()).collect();
    let extension_kinds: Vec<&'static str> =
        EouKind::EXTENSIONS.iter().map(|k| k.as_str()).collect();

    let (diar_enabled, diar_max_per_chunk, diar_max_per_frame, diar_emb_dim, diar_frame_hz) =
        match models {
            Some(m) => {
                let enabled = m.diar_segmentation.is_some() && m.diar_embedding.is_some();
                let max_pc = m
                    .diar_segmentation
                    .as_ref()
                    .map(|s| s.max_speakers_per_chunk() as u32)
                    .unwrap_or(0);
                let max_pf = m
                    .diar_segmentation
                    .as_ref()
                    .map(|s| s.max_speakers_per_frame() as u32)
                    .unwrap_or(0);
                let emb_dim = m
                    .diar_embedding
                    .as_ref()
                    .map(|e| e.embedding_dim() as u32)
                    .unwrap_or(0);
                let fr_hz = m
                    .diar_segmentation
                    .as_ref()
                    .map(|s| s.frame_rate_hz())
                    .unwrap_or(0);
                (enabled, max_pc, max_pf, emb_dim, fr_hz)
            }
            None => (false, 0, 0, 0, 0),
        };

    serde_json::json!({
        "rfc_version": RFC_VERSION,
        "features": {
            "eou_kinds": spec_kinds,

            "fusion_rules": ["noisy_or", "max", "mean", "weighted"],
            "input_audio_formats": crate::defaults::audio_format::SUPPORTED,
            "output_audio_formats": crate::defaults::audio_format::SUPPORTED,
        },
        "extensions": {
            "eou_kinds": extension_kinds,

            "fusion_rules": ["gated"],
            "eager_eou": true,
            "integrated_eou": true,
            "predicted_resp_phase": true,

            "diarization": {
                "enabled": diar_enabled,
                "max_speakers_per_chunk": diar_max_per_chunk,
                "max_speakers_per_frame": diar_max_per_frame,
                "embedding_dim": diar_emb_dim,
                "frame_rate_hz": diar_frame_hz,
                "endpoints": {
                    "audio_diarization": "/v1/audio/diarization",
                    "audio_embeddings": "/v1/audio/embeddings",
                    "transcription_diarized_json": "/v1/audio/transcriptions?response_format=diarized_json",
                    "realtime_event": "conversation.item.diarization",
                },
            },
        },
    })
}

#[cfg(test)]
mod capabilities_tests {
    use super::*;
    use crate::eou::EouKind;

    #[test]
    fn capabilities_partition_spec_kinds_from_extensions_and_carry_the_diarization_and_eou_flags() {
        let caps = capabilities_json();
        let listed: Vec<String> = caps["features"]["eou_kinds"]
            .as_array()
            .expect("eou_kinds is an array")
            .iter()
            .map(|v| v.as_str().expect("entry is a string").to_string())
            .collect();
        assert_eq!(listed, vec!["vad", "text", "audio", "fusion"]);
        {
            let caps = capabilities_json();
            let ext: Vec<String> = caps["extensions"]["eou_kinds"]
                .as_array()
                .expect("extensions.eou_kinds is an array")
                .iter()
                .map(|v| v.as_str().unwrap().to_string())
                .collect();
            assert!(ext.contains(&"heuristic".to_string()));
            assert!(ext.contains(&"integrated".to_string()));
            for k in EouKind::V3_SPEC {
                assert!(
                    !ext.contains(&k.as_str().to_string()),
                    "spec kind {} must not appear in extensions",
                    k.as_str(),
                );
            }
        }
        {
            for k in [EouKind::Vad, EouKind::Text, EouKind::Audio, EouKind::Fusion] {
                assert!(k.is_v3_spec(), "{} should be spec", k.as_str());
            }
            for k in [EouKind::Heuristic, EouKind::Integrated] {
                assert!(
                    !k.is_v3_spec(),
                    "{} must not be claimed as spec",
                    k.as_str()
                );
            }
        }
        {
            let caps = capabilities_json();
            let diar = &caps["extensions"]["diarization"];
            for key in [
                "enabled",
                "max_speakers_per_chunk",
                "max_speakers_per_frame",
                "embedding_dim",
                "frame_rate_hz",
                "endpoints",
            ] {
                assert!(!diar[key].is_null(), "extensions.diarization.{key} missing");
            }
            for key in [
                "audio_diarization",
                "audio_embeddings",
                "transcription_diarized_json",
                "realtime_event",
            ] {
                assert!(
                    diar["endpoints"][key].is_string(),
                    "extensions.diarization.endpoints.{key} missing"
                );
            }

            assert_eq!(diar["enabled"], serde_json::json!(false));
        }
        {
            let caps = capabilities_json();
            for key in ["eager_eou", "integrated_eou", "predicted_resp_phase"] {
                assert_eq!(
                    caps["extensions"][key],
                    serde_json::json!(true),
                    "extensions.{key} must be true"
                );
            }
        }
    }

}

fn session_max_duration() -> Duration {
    static CACHED: OnceLock<Duration> = OnceLock::new();
    *CACHED.get_or_init(|| {
        let secs = std::env::var(crate::defaults::env::SESSION_MAX_DURATION_S)
            .ok()
            .and_then(|v| v.parse::<u64>().ok())
            .unwrap_or(SESSION_MAX_DURATION_S);
        Duration::from_secs(secs)
    })
}

pub const CAPACITY_ERROR: &str = "concurrent session cap exceeded";

static ACTIVE_SESSIONS: AtomicUsize = AtomicUsize::new(0);

pub(crate) struct ActiveSessionGuard;

impl ActiveSessionGuard {
    pub(crate) fn try_acquire(cap: usize) -> Option<Self> {
        let prev = ACTIVE_SESSIONS.fetch_add(1, Ordering::AcqRel);
        if prev >= cap {
            ACTIVE_SESSIONS.fetch_sub(1, Ordering::AcqRel);
            None
        } else {
            Some(Self)
        }
    }
}

impl Drop for ActiveSessionGuard {
    fn drop(&mut self) {
        ACTIVE_SESSIONS.fetch_sub(1, Ordering::AcqRel);
    }
}

pub fn active_session_count() -> usize {
    ACTIVE_SESSIONS.load(Ordering::Acquire)
}

pub(crate) fn max_concurrent_sessions() -> usize {
    std::env::var(crate::defaults::env::WS_MAX_CONCURRENT_SESSIONS)
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(crate::defaults::ws::MAX_CONCURRENT_SESSIONS)
}

static SESSION_SLOTS: OnceLock<Mutex<HashMap<String, ActiveSessionGuard>>> = OnceLock::new();

fn session_slots() -> &'static Mutex<HashMap<String, ActiveSessionGuard>> {
    SESSION_SLOTS.get_or_init(|| Mutex::new(HashMap::new()))
}

fn reserve_session_slot(id: &str, guard: ActiveSessionGuard) {
    let mut map = session_slots().lock().expect("session slots poisoned");
    map.insert(id.to_string(), guard);
}

fn release_session_slot(id: &str) {
    let mut map = session_slots().lock().expect("session slots poisoned");
    map.remove(id);
}

static SESSIONS: OnceLock<Mutex<HashMap<String, Arc<Session>>>> = OnceLock::new();

fn sessions() -> &'static Mutex<HashMap<String, Arc<Session>>> {
    SESSIONS.get_or_init(|| Mutex::new(HashMap::new()))
}

fn register_session(session: Arc<Session>) {
    let mut map = sessions().lock().expect("sessions registry poisoned");
    map.insert(session.id.as_str().to_string(), session);
}

pub(crate) fn drop_session(id: &str) -> Option<Arc<Session>> {
    let mut map = sessions().lock().expect("sessions registry poisoned");
    let removed = map.remove(id);
    if let Some(sess) = removed.as_ref() {
        sess.audio_store.close();
    }
    release_session_slot(id);
    crate::inspect::unregister(id);
    removed
}

fn lookup_session(id: &str) -> Option<Arc<Session>> {
    let map = sessions().lock().expect("sessions registry poisoned");
    map.get(id).cloned()
}

pub fn live_session_count() -> usize {
    sessions().lock().expect("sessions registry poisoned").len()
}

#[cfg(test)]
mod admission_tests {
    use super::*;
    use std::sync::Barrier;

    static TEST_LOCK: tokio::sync::Mutex<()> = tokio::sync::Mutex::const_new(());

    #[test]
    fn concurrent_acquire_never_exceeds_cap() {
        let _serial = TEST_LOCK.blocking_lock();
        let base = active_session_count();
        let cap = base + 8;
        let threads = 64usize;
        let admitted = AtomicUsize::new(0);
        let peak = AtomicUsize::new(0);
        let barrier = Barrier::new(threads);

        std::thread::scope(|s| {
            for _ in 0..threads {
                s.spawn(|| {
                    let held = ActiveSessionGuard::try_acquire(cap);
                    if held.is_some() {
                        admitted.fetch_add(1, Ordering::AcqRel);
                    }
                    barrier.wait();
                    peak.fetch_max(active_session_count(), Ordering::AcqRel);
                    drop(held);
                });
            }
        });

        assert_eq!(admitted.load(Ordering::Acquire), 8);
        assert!(peak.load(Ordering::Acquire) <= cap);
        assert_eq!(active_session_count(), base);
    }

    #[tokio::test]
    async fn the_slot_is_held_for_the_whole_session_not_just_the_upgrade() {
        let _serial = TEST_LOCK.lock().await;
        let base = active_session_count();
        let guard = ActiveSessionGuard::try_acquire(base + 1).expect("slot available");
        let (tx, rx) = tokio::sync::oneshot::channel::<()>();

        let fut = hold_slot(guard, async move {
            let _ = rx.await;
        });
        let handle = tokio::spawn(fut);

        for _ in 0..8 {
            tokio::task::yield_now().await;
        }

        assert_eq!(
            active_session_count(),
            base + 1,
            "the slot was released before the session ended, so the cap counts \
             nothing and any number of sessions can be accepted"
        );

        assert!(
            ActiveSessionGuard::try_acquire(base + 1).is_none(),
            "cap not enforced while a session holds its slot"
        );

        tx.send(()).expect("receiver alive");
        handle.await.expect("session future");
        assert_eq!(
            active_session_count(),
            base,
            "the slot was not released when the session ended"
        );
    }

    #[test]
    fn session_slot_is_released_by_drop_session() {
        let _serial = TEST_LOCK.blocking_lock();
        let base = active_session_count();
        let id = "admission-test-session";
        let slot = ActiveSessionGuard::try_acquire(base + 1).expect("slot available");
        reserve_session_slot(id, slot);
        assert_eq!(active_session_count(), base + 1);
        assert!(ActiveSessionGuard::try_acquire(base + 1).is_none());
        drop_session(id);
        assert_eq!(active_session_count(), base);
        assert!(ActiveSessionGuard::try_acquire(base + 1).is_some());
    }

    #[test]
    fn cap_defaults_to_the_ws_constant() {
        if std::env::var(crate::defaults::env::WS_MAX_CONCURRENT_SESSIONS).is_ok() {
            return;
        }
        assert_eq!(
            max_concurrent_sessions(),
            crate::defaults::ws::MAX_CONCURRENT_SESSIONS
        );
    }
}

pub(crate) async fn hold_slot<F>(guard: ActiveSessionGuard, fut: F)
where
    F: std::future::Future<Output = ()>,
{
    let _hold = guard;
    fut.await
}

pub async fn handle_offer(offer_sdp: &str, query: &RealtimeQuery) -> anyhow::Result<String> {
    let cap = max_concurrent_sessions();
    let Some(slot) = ActiveSessionGuard::try_acquire(cap) else {
        warn!(cap, "rejecting realtime offer: {}", CAPACITY_ERROR);
        anyhow::bail!("{}", CAPACITY_ERROR);
    };

    let models = crate::models::get_or_init().context("load models")?;
    let intent = Intent::from_query(query);

    let outbound_track = if matches!(intent, Intent::Conversation) {
        Some(audio_out::build_outbound_track()?)
    } else {
        None
    };
    let outbound_audio = outbound_track
        .clone()
        .map(transport::OutboundAudioSpec::Webrtc);

    let session = Arc::new(Session::new(query.clone(), models, intent, outbound_audio));

    let (gather_tx, mut gather_rx) = tokio::sync::mpsc::channel::<()>(1);
    let handler = Arc::new(PcEvents {
        session: Arc::downgrade(&session),
        session_id: session.id.as_str().to_string(),
        gather_complete: gather_tx,
    });
    let pc = build_peer_connection(handler).await.context("build PC")?;

    let outbound_sender = match &outbound_track {
        Some(track) => {
            let dyn_track: Arc<dyn TrackLocal> = track.track.clone();
            Some(
                pc.add_track(dyn_track)
                    .await
                    .context("add outbound track")?,
            )
        }
        None => None,
    };

    let normalized = sdp_filter::normalize_offer(offer_sdp);
    let offer = RTCSessionDescription::offer(normalized).context("parse offer SDP")?;
    pc.set_remote_description(offer)
        .await
        .context("set_remote_description")?;

    let answer = pc.create_answer(None).await.context("create_answer")?;
    pc.set_local_description(answer)
        .await
        .context("set_local_description")?;
    if tokio::time::timeout(ICE_GATHER_TIMEOUT, gather_rx.recv())
        .await
        .is_err()
    {
        warn!(session_id = %session.id, "ICE gathering incomplete after {ICE_GATHER_TIMEOUT:?}; answering with the candidates gathered so far");
    }

    if let (Some(track), Some(sender)) = (&outbound_track, &outbound_sender) {
        let pt = track
            .bind_negotiated_payload_type(sender)
            .await
            .context("outbound payload type")?;
        debug!(session_id = %session.id, payload_type = pt, "outbound opus negotiated");
    }

    let local = pc
        .local_description()
        .await
        .context("local_description missing after gather")?;

    session.attach_peer_connection(pc).await;
    reserve_session_slot(session.id.as_str(), slot);
    register_session(session.clone());

    let timeout = session_max_duration();
    session.spawn_max_duration_timeout(timeout).await;

    Ok(local.sdp)
}

const ICE_GATHER_TIMEOUT: Duration = Duration::from_secs(10);

struct PcEvents {
    session: Weak<Session>,
    session_id: String,
    gather_complete: tokio::sync::mpsc::Sender<()>,
}

#[async_trait::async_trait]
impl PeerConnectionEventHandler for PcEvents {
    async fn on_ice_gathering_state_change(&self, state: RTCIceGatheringState) {
        if state == RTCIceGatheringState::Complete {
            let _ = self.gather_complete.try_send(());
        }
    }

    async fn on_ice_connection_state_change(&self, state: RTCIceConnectionState) {
        debug!(state = ?state, session_id = %self.session_id, "ICE state");
    }

    async fn on_connection_state_change(&self, state: RTCPeerConnectionState) {
        debug!(state = ?state, session_id = %self.session_id, "PC state");
        let reason = match state {
            RTCPeerConnectionState::Closed => state::TerminationReason::ClientClosed,
            RTCPeerConnectionState::Failed | RTCPeerConnectionState::Disconnected => {
                state::TerminationReason::TransportError
            }
            _ => return,
        };
        let Some(session) = self.session.upgrade() else {
            return;
        };
        session.transition_to_terminated_with(reason).await;
        if let Some(sess) = drop_session(&self.session_id) {
            sess.abort_timeout_task().await;
            info!(session_id = %self.session_id, ?state, "session dropped");
        }
    }

    async fn on_data_channel(&self, dc: Arc<dyn DataChannel>) {
        let Some(session) = self.session.upgrade() else {
            return;
        };
        let label = dc.label().await.unwrap_or_default();
        info!(label = %label, "data channel received from client");
        session.attach_data_channel(dc).await;
    }

    async fn on_track(&self, track: Arc<dyn TrackRemote>) {
        let Some(session) = self.session.upgrade() else {
            return;
        };
        let ssrc = track.ssrcs().await.first().copied();
        let codec = match ssrc {
            Some(ssrc) => track.codec(ssrc).await,
            None => None,
        };
        let kind = track.kind().await;
        info!(
            kind = ?kind,
            ssrc,
            mime = codec.as_ref().map(|c| c.mime_type.as_str()).unwrap_or(""),
            clock_rate = codec.as_ref().map(|c| c.clock_rate).unwrap_or(0),
            channels = codec.as_ref().map(|c| c.channels).unwrap_or(0),
            "track received"
        );
        session.attach_audio_track(track).await;
    }
}

pub(crate) fn lookup_session_pub(id: &str) -> Option<Arc<Session>> {
    lookup_session(id)
}

async fn build_peer_connection(
    handler: Arc<dyn PeerConnectionEventHandler>,
) -> anyhow::Result<Arc<dyn PeerConnection>> {
    let mut media = MediaEngine::default();
    media.register_default_codecs().context("register codecs")?;
    let registry = register_default_interceptors(Registry::new(), &mut media)
        .context("register interceptors")?;
    let pc = PeerConnectionBuilder::new()
        .with_configuration(RTCConfigurationBuilder::new().build())
        .with_media_engine(media)
        .with_interceptor_registry(registry)
        .with_handler(handler)
        .with_udp_addrs(vec!["0.0.0.0:0", "[::]:0"])
        .build()
        .await
        .context("new_peer_connection")?;
    Ok(Arc::new(pc))
}
