use std::sync::Arc;

use anyhow::{Context, Result};
use serde_json::Value;
use tokio::sync::mpsc;
use tracing::warn;
use webrtc::data_channel::RTCDataChannel;
use webrtc::track::track_local::track_local_static_sample::TrackLocalStaticSample;

use super::framing;

#[derive(Clone)]
pub enum EventSink {
    DataChannel(Arc<RTCDataChannel>),
    WebSocket(mpsc::Sender<String>),
}

impl std::fmt::Debug for EventSink {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            EventSink::DataChannel(_) => f.write_str("EventSink::DataChannel(..)"),
            EventSink::WebSocket(_) => f.write_str("EventSink::WebSocket(..)"),
        }
    }
}

impl EventSink {
    pub async fn send_text(&self, text: String) -> Result<()> {
        match self {
            EventSink::DataChannel(dc) => {
                dc.send_text(text).await.context("data channel send_text")?;
                Ok(())
            }
            EventSink::WebSocket(tx) => tx
                .send(text)
                .await
                .map_err(|e| anyhow::anyhow!("ws writer dropped: {e}")),
        }
    }

    pub async fn send_value(&self, ev: &Value) {
        match self {
            EventSink::DataChannel(_) => {
                let frames = match framing::frame_event(ev) {
                    Ok(f) => f,
                    Err(err) => {
                        warn!(error = %err, "framing failed");
                        return;
                    }
                };
                for frame in frames {
                    if let Err(err) = self.send_text(frame).await {
                        warn!(error = %err, "event sink send failed");
                        break;
                    }
                }
            }
            EventSink::WebSocket(_) => {
                let text = match serde_json::to_string(ev) {
                    Ok(t) => t,
                    Err(err) => {
                        warn!(error = %err, "ws json serialize failed");
                        return;
                    }
                };
                if let Err(err) = self.send_text(text).await {
                    warn!(error = %err, "ws sink send failed");
                }
            }
        }
    }
}

#[derive(Clone)]
pub enum OutboundAudioSpec {
    Webrtc(Arc<TrackLocalStaticSample>),
    WebSocket {
        ws_send: mpsc::Sender<String>,
        format: String,
    },
}

impl std::fmt::Debug for OutboundAudioSpec {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            OutboundAudioSpec::Webrtc(_) => f.write_str("OutboundAudioSpec::Webrtc(..)"),
            OutboundAudioSpec::WebSocket { format, .. } => {
                write!(f, "OutboundAudioSpec::WebSocket(format={format})")
            }
        }
    }
}
