use std::sync::Arc;
use std::sync::OnceLock;

use tracing::warn;

use super::inspector::InspectorEvent;
use super::session::Session;

pub(super) fn realtime_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(
        || match std::env::var(crate::defaults::diarization::REALTIME_ENV) {
            Ok(v) => matches!(
                v.trim().to_ascii_lowercase().as_str(),
                "1" | "true" | "yes" | "on"
            ),
            Err(_) => crate::defaults::diarization::REALTIME_ENABLED,
        },
    )
}

pub(super) async fn run_diarization(
    session: Arc<Session>,
    item_id: String,
    audio: Vec<f32>,
    audio_end_ms: u64,
) {
    let mut guard = session.diarizer.lock().await;
    let diarizer = match guard.as_mut() {
        Some(d) => d,
        None => return,
    };

    let utt_start_ms = audio_end_ms.saturating_sub((audio.len() as u64) * 1000 / 16_000);
    let t0 = std::time::Instant::now();
    let result = diarizer.diarize_utterance(&audio, utt_start_ms);
    drop(guard);
    let elapsed_ms = t0.elapsed().as_millis() as u64;

    let segments = match result {
        Ok(s) => s,
        Err(e) => {
            let reason = format!("{e}");
            warn!(error = %reason, %item_id, "diarization failed");
            session.inspector.emit(InspectorEvent::DiarizationEmitted {
                session_id: session.id.as_str().to_string(),
                item_id: item_id.clone(),
                audio_end_ms,
                num_segments: 0,
                num_speakers: 0,
                elapsed_ms,
                failed: true,
                reason: Some(reason),
            });
            return;
        }
    };

    let mut speakers: std::collections::BTreeSet<u32> = std::collections::BTreeSet::new();
    for s in &segments {
        speakers.insert(s.speaker);
    }
    let num_segments = segments.len() as u32;
    let num_speakers = speakers.len() as u32;

    if !segments.is_empty() {
        session
            .emit(super::wire::OutboundEvent::Diarization {
                item_id: item_id.clone().into(),
                audio_end_ms,
                elapsed_ms: Some(elapsed_ms),
                segments: segments
                    .iter()
                    .map(|s| {
                        serde_json::json!({
                            "speaker": format!("SPEAKER_{:02}", s.speaker),
                            "start": s.t_start_ms as f64 / 1000.0,
                            "end": s.t_end_ms as f64 / 1000.0,
                            "confidence": s.confidence,
                        })
                    })
                    .collect(),
            })
            .await;
    }

    session.inspector.emit(InspectorEvent::DiarizationEmitted {
        session_id: session.id.as_str().to_string(),
        item_id,
        audio_end_ms,
        num_segments,
        num_speakers,
        elapsed_ms,
        failed: false,
        reason: None,
    });
}
