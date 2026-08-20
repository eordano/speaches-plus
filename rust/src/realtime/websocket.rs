use std::sync::Arc;
use std::time::Duration;

use anyhow::Context;
use axum::extract::ws::{Message, WebSocket, WebSocketUpgrade};
use axum::extract::{Query, State};
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use futures_util::{SinkExt, StreamExt};
use tokio::sync::mpsc;
use tracing::{debug, info, warn};

use crate::AppState;
use crate::RealtimeQuery;

use super::transport::OutboundAudioSpec;
use super::{
    drop_session, register_session, session_max_duration, ActiveSessionGuard, Intent, Session,
};
use crate::defaults;
use crate::models;

pub use super::active_session_count;

fn read_env_usize(key: &str, fallback: usize) -> usize {
    std::env::var(key)
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(fallback)
}

fn read_env_u64(key: &str, fallback: u64) -> u64 {
    std::env::var(key)
        .ok()
        .and_then(|s| s.parse::<u64>().ok())
        .unwrap_or(fallback)
}

pub async fn realtime_ws(
    State(state): State<AppState>,
    Query(q): Query<RealtimeQuery>,
    ws: WebSocketUpgrade,
) -> Response {
    let cap = super::max_concurrent_sessions();
    let Some(guard) = ActiveSessionGuard::try_acquire(cap) else {
        warn!(cap, "rejecting ws upgrade: {}", super::CAPACITY_ERROR);
        return (StatusCode::SERVICE_UNAVAILABLE, super::CAPACITY_ERROR).into_response();
    };

    let models = match models::get_or_init().context("load models") {
        Ok(m) => m,
        Err(err) => {
            tracing::error!(error = %err, "models not loaded; rejecting WS upgrade");
            return (StatusCode::SERVICE_UNAVAILABLE, format!("{err:#}")).into_response();
        }
    };
    let _ = state;

    let max_msg = read_env_usize(
        defaults::env::WS_MAX_MESSAGE_BYTES,
        defaults::ws::MAX_MESSAGE_BYTES,
    );

    ws.max_message_size(max_msg)
        .max_frame_size(max_msg)
        .on_upgrade(move |socket| {
            super::hold_slot(guard, run_session(socket, q, models))
        })
}

async fn run_session(socket: WebSocket, query: RealtimeQuery, models: Arc<models::Models>) {
    let intent = Intent::from_query(&query);
    let queue_cap = read_env_usize(
        defaults::env::WS_OUTBOUND_QUEUE_CAP,
        defaults::ws::OUTBOUND_QUEUE_CAP,
    );
    let idle_timeout = Duration::from_secs(read_env_u64(
        defaults::env::WS_IDLE_TIMEOUT_S,
        defaults::ws::IDLE_TIMEOUT_S,
    ));

    let (ws_tx, ws_rx) = mpsc::channel::<String>(queue_cap);

    let outbound_audio = if matches!(intent, Intent::Conversation) {
        Some(OutboundAudioSpec::WebSocket {
            ws_send: ws_tx.clone(),
            format: defaults::audio_format::DEFAULT.to_string(),
        })
    } else {
        None
    };

    let session = Arc::new(Session::new(query.clone(), models, intent, outbound_audio));
    let _ = session.ensure_audio_in_pipeline().await;

    register_session(session.clone());
    let session_id = session.id.as_str().to_string();
    info!(session_id = %session_id, ?intent, "ws session created");

    session.attach_websocket(ws_tx).await;
    session
        .spawn_max_duration_timeout(session_max_duration())
        .await;

    let (mut sink, mut stream) = socket.split();

    let writer_session_id = session_id.clone();
    let writer = tokio::spawn(async move {
        let mut rx = ws_rx;
        let ping_every = Duration::from_secs(defaults::ws::PING_INTERVAL_S);
        let mut ping_tick = tokio::time::interval(ping_every);
        ping_tick.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
        loop {
            tokio::select! {
                msg = rx.recv() => match msg {
                    Some(text) => {
                        if let Err(err) = sink.send(Message::Text(text.into())).await {
                            warn!(session_id = %writer_session_id, error = %err, "ws send failed");
                            break;
                        }
                    }
                    None => break,
                },
                _ = ping_tick.tick() => {
                    if let Err(err) = sink.send(Message::Ping(Vec::new().into())).await {
                        debug!(session_id = %writer_session_id, error = %err, "ws ping failed; closing");
                        break;
                    }
                }
            }
        }
        let _ = sink.close().await;
    });

    let reader_session = session.clone();
    let reader_session_id = session_id.clone();
    let termination = loop {
        let frame = tokio::time::timeout(idle_timeout, stream.next()).await;
        let frame = match frame {
            Ok(f) => f,
            Err(_) => {
                warn!(
                    session_id = %reader_session_id,
                    timeout_s = idle_timeout.as_secs(),
                    "ws idle timeout"
                );
                break super::state::TerminationReason::IdleTimeout;
            }
        };
        match frame {
            Some(Ok(Message::Text(text))) => {
                let bytes = bytes::Bytes::copy_from_slice(text.as_str().as_bytes());
                if let Err(err) = reader_session.handle_client_event("ws", bytes).await {
                    warn!(
                        session_id = %reader_session_id,
                        error = %err,
                        "ws client event handler failed",
                    );
                }
            }
            Some(Ok(Message::Binary(_))) => {
                debug!(session_id = %reader_session_id, "ws ignored binary frame");
            }
            Some(Ok(Message::Ping(_))) | Some(Ok(Message::Pong(_))) => {}
            Some(Ok(Message::Close(_))) | None => {
                break super::state::TerminationReason::ClientClosed
            }
            Some(Err(err)) => {
                warn!(session_id = %reader_session_id, error = %err, "ws read error");
                break super::state::TerminationReason::TransportError;
            }
        }
    };

    info!(session_id = %session_id, reason = termination.as_str(), "ws session ending");

    session.cancel_session_lanes().await;
    session.emit_session_done(termination.as_str()).await;
    session.transition_to_terminated_with(termination).await;
    if let Some(sess) = drop_session(&session_id) {
        sess.abort_timeout_task().await;
    }
    drop(session);
    let _ = writer.await;
}
