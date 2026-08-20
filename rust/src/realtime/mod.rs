use std::collections::HashMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::Duration;

use anyhow::Context;
use tracing::{debug, info, warn};
use webrtc::api::interceptor_registry::register_default_interceptors;
use webrtc::api::media_engine::MediaEngine;
use webrtc::api::APIBuilder;
use webrtc::data_channel::RTCDataChannel;
use webrtc::interceptor::registry::Registry;
use webrtc::peer_connection::configuration::RTCConfiguration;
use webrtc::peer_connection::peer_connection_state::RTCPeerConnectionState;
use webrtc::peer_connection::sdp::session_description::RTCSessionDescription;
use webrtc::peer_connection::RTCPeerConnection;
use webrtc::track::track_local::TrackLocal;
use webrtc::track::track_remote::TrackRemote;

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
mod eou_integrated;
mod eou_predicted;
mod events;
mod framing;
mod fuzz;
mod inspector;
#[cfg(test)]
mod order_harness;
mod pipeline;
mod sdp_filter;
mod session;
mod session_update;
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
    fn capabilities_eou_kinds_match_rfc_v3_section_6_2() {
        let caps = capabilities_json();
        let listed: Vec<String> = caps["features"]["eou_kinds"]
            .as_array()
            .expect("eou_kinds is an array")
            .iter()
            .map(|v| v.as_str().expect("entry is a string").to_string())
            .collect();
        assert_eq!(listed, vec!["vad", "text", "audio", "fusion"]);
    }

    #[test]
    fn extensions_namespace_separates_speaches_kinds() {
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

    #[test]
    fn is_v3_spec_partition_is_total() {
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

    #[test]
    fn capabilities_extensions_diarization_block_present() {
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

    #[test]
    fn capabilities_extensions_eou_flags_true() {
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

    static TEST_LOCK: Mutex<()> = Mutex::new(());

    #[test]
    fn concurrent_acquire_never_exceeds_cap() {
        let _serial = TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
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
        let _serial = TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
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
        let _serial = TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
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

pub(crate) fn hold_slot<F>(guard: ActiveSessionGuard, fut: F) -> impl std::future::Future<Output = ()>
where
    F: std::future::Future<Output = ()>,
{
    async move {
        let _hold = guard;
        fut.await
    }
}

pub async fn handle_offer(offer_sdp: &str, query: &RealtimeQuery) -> anyhow::Result<String> {
    let cap = max_concurrent_sessions();
    let Some(slot) = ActiveSessionGuard::try_acquire(cap) else {
        warn!(cap, "rejecting realtime offer: {}", CAPACITY_ERROR);
        anyhow::bail!("{}", CAPACITY_ERROR);
    };

    let models = crate::models::get_or_init().context("load models")?;
    let intent = Intent::from_query(query);

    let pc = build_peer_connection().await.context("build PC")?;

    let outbound_audio = if matches!(intent, Intent::Conversation) {
        let track = audio_out::build_outbound_track();
        let dyn_track: Arc<dyn TrackLocal + Send + Sync> = track.clone();
        pc.add_track(dyn_track)
            .await
            .context("add outbound track")?;
        Some(transport::OutboundAudioSpec::Webrtc(track))
    } else {
        None
    };

    let session = Arc::new(Session::new(query.clone(), models, intent, outbound_audio));

    let session_state_id = session.id.as_str().to_string();
    let session_state_ref = session.clone();
    pc.on_peer_connection_state_change(Box::new(move |s| {
        let id = session_state_id.clone();
        let session_ref = session_state_ref.clone();
        Box::pin(async move {
            debug!(state = ?s, session_id = %id, "PC state");
            match s {
                RTCPeerConnectionState::Failed
                | RTCPeerConnectionState::Closed
                | RTCPeerConnectionState::Disconnected => {
                    let reason = match s {
                        RTCPeerConnectionState::Closed => state::TerminationReason::ClientClosed,
                        _ => state::TerminationReason::TransportError,
                    };
                    session_ref.transition_to_terminated_with(reason).await;
                    if let Some(sess) = drop_session(&id) {
                        sess.abort_timeout_task().await;
                        info!(session_id = %id, ?s, "session dropped");
                    }
                }
                _ => {}
            }
        })
    }));

    let session_dc = session.clone();
    pc.on_data_channel(Box::new(move |dc: Arc<RTCDataChannel>| {
        let session = session_dc.clone();
        Box::pin(async move {
            info!(label = %dc.label(), "data channel received from client");
            session.attach_data_channel(dc).await;
        })
    }));

    let session_track = session.clone();
    pc.on_track(Box::new(move |track: Arc<TrackRemote>, _, _| {
        let session = session_track.clone();
        Box::pin(async move {
            let codec = track.codec();
            info!(
                kind = ?track.kind(),
                ssrc = track.ssrc(),
                mime = %codec.capability.mime_type,
                clock_rate = codec.capability.clock_rate,
                channels = codec.capability.channels,
                "track received"
            );
            session.attach_audio_track(track).await;
        })
    }));

    let normalized = sdp_filter::normalize_offer(offer_sdp);
    let offer = RTCSessionDescription::offer(normalized).context("parse offer SDP")?;
    pc.set_remote_description(offer)
        .await
        .context("set_remote_description")?;

    let answer = pc.create_answer(None).await.context("create_answer")?;
    let mut gather_complete = pc.gathering_complete_promise().await;
    pc.set_local_description(answer)
        .await
        .context("set_local_description")?;
    let _ = gather_complete.recv().await;

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

pub(crate) fn lookup_session_pub(id: &str) -> Option<Arc<Session>> {
    lookup_session(id)
}

async fn build_peer_connection() -> anyhow::Result<Arc<RTCPeerConnection>> {
    let mut media = MediaEngine::default();
    media.register_default_codecs().context("register codecs")?;
    let registry = register_default_interceptors(Registry::new(), &mut media)
        .context("register interceptors")?;
    let api = APIBuilder::new()
        .with_media_engine(media)
        .with_interceptor_registry(registry)
        .build();

    let pc = api
        .new_peer_connection(RTCConfiguration::default())
        .await
        .context("new_peer_connection")?;

    pc.on_ice_connection_state_change(Box::new(move |s| {
        debug!(state = ?s, "ICE state");
        Box::pin(async {})
    }));

    Ok(Arc::new(pc))
}
