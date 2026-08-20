use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::Result;
use base64::engine::general_purpose::STANDARD as B64;
use base64::Engine as _;
use serde_json::json;
use tokio::sync::mpsc;
use tracing::warn;

use super::audio_out::OutboundPushError;
use super::wire::EventSeq;
use crate::audio::g711;
use crate::types::MonoF32At24k;

const KOKORO_HZ: usize = 24_000;
const FRAME_MS: usize = 20;
const OUTPUT_INDEX: u64 = 0;
const CONTENT_INDEX: u64 = 0;

#[derive(Debug, Clone, Copy)]
enum WireCodec {
    Pcm16Le,
    Ulaw,
    Alaw,
}

fn wire_for(format: &str) -> (WireCodec, usize) {
    match format {
        "pcm16_8k" => (WireCodec::Pcm16Le, 8_000),
        "pcm16_16k" => (WireCodec::Pcm16Le, 16_000),
        "pcm16_24k" => (WireCodec::Pcm16Le, 24_000),
        "pcm16_44k1" => (WireCodec::Pcm16Le, 44_100),
        "pcm16_48k" => (WireCodec::Pcm16Le, 48_000),
        "g711_ulaw" => (WireCodec::Ulaw, 8_000),
        "g711_alaw" => (WireCodec::Alaw, 8_000),
        _ => (WireCodec::Pcm16Le, 24_000),
    }
}

pub struct WsAudioPacer {
    ws_send: mpsc::Sender<String>,
    event_seq: Arc<EventSeq>,
    response_id: String,
    item_id: String,
    codec: WireCodec,
    sample_rate: usize,
    frame_samples: usize,
    carry: Vec<f32>,
    payload_buf: Vec<u8>,
    start: Option<Instant>,
    frames_written: u64,
    played_ms: Arc<AtomicU64>,
    last_sample: f32,
    src_position: f64,
}

impl WsAudioPacer {
    pub fn start(
        ws_send: mpsc::Sender<String>,
        event_seq: Arc<EventSeq>,
        played_ms: Arc<AtomicU64>,
        format: &str,
        response_id: &str,
        item_id: &str,
    ) -> Self {
        let (codec, sample_rate) = wire_for(format);
        let frame_samples = sample_rate * FRAME_MS / 1000;
        let payload_bytes = match codec {
            WireCodec::Pcm16Le => frame_samples * 2,
            WireCodec::Ulaw | WireCodec::Alaw => frame_samples,
        };
        Self {
            ws_send,
            event_seq,
            response_id: response_id.to_string(),
            item_id: item_id.to_string(),
            codec,
            sample_rate,
            frame_samples,
            carry: Vec::with_capacity(frame_samples * 4),
            payload_buf: vec![0u8; payload_bytes],
            start: None,
            frames_written: 0,
            played_ms,
            last_sample: 0.0,
            src_position: 0.0,
        }
    }

    pub async fn play(&mut self, audio_24k: MonoF32At24k) -> Result<(), OutboundPushError> {
        if audio_24k.is_empty() {
            return Ok(());
        }
        let samples_24k = audio_24k.into_vec();
        let resampled: Vec<f32> = if self.sample_rate == KOKORO_HZ {
            samples_24k
        } else {
            self.linear_resample(&samples_24k)
        };
        self.carry.extend_from_slice(&resampled);
        while self.carry.len() >= self.frame_samples {
            self.emit_one_frame()
                .await
                .map_err(OutboundPushError::Other)?;
        }
        Ok(())
    }

    pub async fn flush(&mut self) -> Result<()> {
        if self.carry.is_empty() {
            return Ok(());
        }

        while self.carry.len() < self.frame_samples {
            self.carry.push(0.0);
        }
        self.emit_one_frame().await
    }

    async fn emit_one_frame(&mut self) -> Result<()> {
        let frame: Vec<f32> = self.carry.drain(..self.frame_samples).collect();
        match self.codec {
            WireCodec::Pcm16Le => {
                for (i, s) in frame.iter().enumerate() {
                    let v = (s.clamp(-1.0, 1.0) * 32_767.0) as i16;
                    self.payload_buf[i * 2] = v as u8;
                    self.payload_buf[i * 2 + 1] = (v >> 8) as u8;
                }
            }
            WireCodec::Ulaw => {
                for (i, s) in frame.iter().enumerate() {
                    let v = (s.clamp(-1.0, 1.0) * 32_767.0) as i16;
                    self.payload_buf[i] = g711::ulaw_encode_sample(v);
                }
            }
            WireCodec::Alaw => {
                for (i, s) in frame.iter().enumerate() {
                    let v = (s.clamp(-1.0, 1.0) * 32_767.0) as i16;
                    self.payload_buf[i] = g711::alaw_encode_sample(v);
                }
            }
        }
        let event = json!({
            "event_id": self.event_seq.next_id().as_str(),
            "type": "response.output_audio.delta",
            "response_id": self.response_id,
            "item_id": self.item_id,
            "output_index": OUTPUT_INDEX,
            "content_index": CONTENT_INDEX,
            "delta": B64.encode(&self.payload_buf),
        });
        let text = match serde_json::to_string(&event) {
            Ok(t) => t,
            Err(err) => {
                warn!(error = %err, "audio.delta json serialize failed");
                return Ok(());
            }
        };
        if let Err(err) = self.ws_send.send(text).await {
            warn!(error = %err, "ws writer dropped while sending audio.delta");
            return Ok(());
        }
        let start = *self.start.get_or_insert_with(Instant::now);
        self.frames_written += 1;
        self.played_ms
            .store(self.frames_written * FRAME_MS as u64, Ordering::Release);
        let target = start + Duration::from_millis(FRAME_MS as u64) * self.frames_written as u32;
        let now = Instant::now();
        if target > now {
            tokio::time::sleep(target - now).await;
        }
        Ok(())
    }

    fn linear_resample(&mut self, src: &[f32]) -> Vec<f32> {
        if src.is_empty() {
            return Vec::new();
        }
        let ratio = self.sample_rate as f64 / KOKORO_HZ as f64;
        let out_len = ((src.len() as f64) * ratio).ceil() as usize;
        let mut out = Vec::with_capacity(out_len);
        let mut pos = self.src_position;
        let step = 1.0 / ratio;
        let mut last = self.last_sample;
        while pos < src.len() as f64 {
            let lo = pos.floor() as usize;
            let hi = (lo + 1).min(src.len() - 1);
            let frac = (pos - lo as f64) as f32;
            let s = if lo >= src.len() {
                last
            } else {
                let a = src[lo];
                let b = src[hi];
                a + (b - a) * frac
            };
            out.push(s);
            last = s;
            pos += step;
        }
        self.src_position = pos - src.len() as f64;
        self.last_sample = last;
        out
    }
}

pub enum AudioPacer {
    Webrtc(super::audio_out::OutboundPacer),
    WebSocket(WsAudioPacer),
}

impl AudioPacer {
    pub async fn play(&mut self, audio_24k: MonoF32At24k) -> Result<(), OutboundPushError> {
        match self {
            AudioPacer::Webrtc(p) => p.play(audio_24k).await,
            AudioPacer::WebSocket(p) => p.play(audio_24k).await,
        }
    }

    pub async fn flush(&mut self) -> Result<()> {
        match self {
            AudioPacer::Webrtc(p) => p.flush().await,
            AudioPacer::WebSocket(p) => p.flush().await,
        }
    }
}
