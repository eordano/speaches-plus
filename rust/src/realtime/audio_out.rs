use anyhow::{Context, Result};
use opus::{Application, Channels, Encoder};
use rubato::{FastFixedIn, PolynomialDegree, Resampler};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tracing::debug;
use webrtc::api::media_engine::MIME_TYPE_OPUS;
use webrtc::media::Sample;
use webrtc::rtp_transceiver::rtp_codec::RTCRtpCodecCapability;
use webrtc::track::track_local::track_local_static_sample::TrackLocalStaticSample;

#[allow(dead_code)]
pub const OUT_SAMPLE_RATE: usize = crate::defaults::audio::OUT_SAMPLE_RATE;
pub const FRAME_MS: usize = crate::defaults::audio::FRAME_MS;
pub const FRAME_SAMPLES: usize = crate::defaults::audio::FRAME_SAMPLES;

const IN_CHUNK_SAMPLES: usize = crate::defaults::audio::IN_CHUNK_SAMPLES;
const OPUS_SAMPLE_RATE_HZ: u32 = crate::defaults::audio::OPUS_SAMPLE_RATE_HZ;
const OPUS_ENCODE_BUFFER_BYTES: usize = crate::defaults::audio::OPUS_ENCODE_BUFFER_BYTES;
const TTS_RESAMPLER_RATIO: f64 = crate::defaults::audio::TTS_RESAMPLER_RATIO;

pub const DEFAULT_OUTBOUND_QUEUE_CAP_MS: u64 = crate::defaults::wire::OUTBOUND_QUEUE_CAP_MS;
pub const DEFAULT_OUTBOUND_QUEUE_CAP_EVENTS: u32 = crate::defaults::wire::OUTBOUND_QUEUE_CAP_EVENTS;

const STALL_TIMEOUT_MS: u64 = 3_000;
const FRAME_GATE_POLL_MS: u64 = 5;

#[derive(Debug)]
pub enum OutboundPushError {
    QueueFull { queued_ms: u64, cap_ms: u64 },
    Other(anyhow::Error),
}

impl std::fmt::Display for OutboundPushError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            OutboundPushError::QueueFull { queued_ms, cap_ms } => write!(
                f,
                "outbound queue cap exceeded ({queued_ms} ms buffered, cap {cap_ms} ms)"
            ),
            OutboundPushError::Other(err) => write!(f, "{err}"),
        }
    }
}

impl std::error::Error for OutboundPushError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            OutboundPushError::QueueFull { .. } => None,
            OutboundPushError::Other(err) => Some(err.as_ref()),
        }
    }
}

impl From<anyhow::Error> for OutboundPushError {
    fn from(err: anyhow::Error) -> Self {
        OutboundPushError::Other(err)
    }
}

impl OutboundPushError {
    #[allow(dead_code)]
    pub fn is_queue_full(&self) -> bool {
        matches!(self, OutboundPushError::QueueFull { .. })
    }
}

pub fn build_outbound_track() -> Arc<TrackLocalStaticSample> {
    Arc::new(TrackLocalStaticSample::new(
        RTCRtpCodecCapability {
            mime_type: MIME_TYPE_OPUS.to_string(),
            clock_rate: 48_000,
            channels: 1,
            sdp_fmtp_line: "minptime=10;useinbandfec=1".to_string(),
            ..Default::default()
        },
        "audio".to_string(),
        "speaches-tts".to_string(),
    ))
}

pub fn read_queue_cap_ms_from_env() -> u64 {
    std::env::var(crate::defaults::env::OUTBOUND_QUEUE_CAP_MS)
        .ok()
        .and_then(|s| s.parse::<u64>().ok())
        .unwrap_or(DEFAULT_OUTBOUND_QUEUE_CAP_MS)
}

pub fn read_queue_cap_events_from_env() -> u32 {
    std::env::var(crate::defaults::env::OUTBOUND_QUEUE_CAP)
        .ok()
        .and_then(|s| s.parse::<u32>().ok())
        .unwrap_or(DEFAULT_OUTBOUND_QUEUE_CAP_EVENTS)
}

#[derive(Clone)]
pub struct QueueGate {
    queued_ms: Arc<AtomicU64>,
    cap_ms: u64,
}

impl QueueGate {
    pub fn new(cap_ms: u64) -> Self {
        Self {
            queued_ms: Arc::new(AtomicU64::new(0)),
            cap_ms,
        }
    }

    #[allow(dead_code)]
    pub fn cap_ms(&self) -> u64 {
        self.cap_ms
    }

    pub fn queued_ms(&self) -> u64 {
        self.queued_ms.load(Ordering::Relaxed)
    }

    pub fn try_push(&self, chunk_ms: u64) -> Result<(), OutboundPushError> {
        let mut prior = self.queued_ms.load(Ordering::Relaxed);
        loop {
            let projected = prior.saturating_add(chunk_ms);
            if projected > self.cap_ms {
                return Err(OutboundPushError::QueueFull {
                    queued_ms: projected,
                    cap_ms: self.cap_ms,
                });
            }
            match self.queued_ms.compare_exchange_weak(
                prior,
                projected,
                Ordering::Relaxed,
                Ordering::Relaxed,
            ) {
                Ok(_) => return Ok(()),
                Err(observed) => prior = observed,
            }
        }
    }

    pub fn on_frame_sent(&self) {
        let frame_ms = FRAME_MS as u64;
        loop {
            let cur = self.queued_ms.load(Ordering::Relaxed);
            let next = cur.saturating_sub(frame_ms);
            if self
                .queued_ms
                .compare_exchange(cur, next, Ordering::Relaxed, Ordering::Relaxed)
                .is_ok()
            {
                break;
            }
        }
    }
}

pub struct OutboundPacer {
    track: Arc<TrackLocalStaticSample>,
    resampler: FastFixedIn<f32>,
    encoder: Encoder,
    in_buf: Vec<Vec<f32>>,
    out_buf: Vec<Vec<f32>>,
    carry: Vec<f32>,
    frame_i16: Vec<i16>,
    opus_buf: Vec<u8>,
    start: Option<Instant>,
    frames_written: u64,
    played_ms: Arc<AtomicU64>,
    gate: QueueGate,
}

impl OutboundPacer {
    pub fn start(
        track: Arc<TrackLocalStaticSample>,
        played_ms: Arc<AtomicU64>,
        queue_cap_ms: u64,
    ) -> Self {
        let resampler = FastFixedIn::<f32>::new(
            TTS_RESAMPLER_RATIO,
            1.0,
            PolynomialDegree::Septic,
            IN_CHUNK_SAMPLES,
            1,
        )
        .expect("rubato 24->48k");
        let chunk_out = resampler.output_frames_max();
        let encoder = Encoder::new(OPUS_SAMPLE_RATE_HZ, Channels::Mono, Application::Voip)
            .expect("opus encoder");
        Self {
            track,
            resampler,
            encoder,
            in_buf: vec![vec![0f32; IN_CHUNK_SAMPLES]],
            out_buf: vec![vec![0f32; chunk_out]],
            carry: Vec::with_capacity(chunk_out + FRAME_SAMPLES),
            frame_i16: vec![0i16; FRAME_SAMPLES],
            opus_buf: vec![0u8; OPUS_ENCODE_BUFFER_BYTES],
            start: None,
            frames_written: 0,
            played_ms,
            gate: QueueGate::new(queue_cap_ms),
        }
    }

    #[allow(dead_code)]
    pub fn queued_ms(&self) -> u64 {
        self.gate.queued_ms()
    }

    #[allow(dead_code)]
    pub fn cap_ms(&self) -> u64 {
        self.gate.cap_ms()
    }

    pub async fn play(
        &mut self,
        audio_24k: crate::types::MonoF32At24k,
    ) -> Result<(), OutboundPushError> {
        if audio_24k.is_empty() {
            return Ok(());
        }
        let samples = audio_24k.into_vec();
        let mut cursor = 0;
        while cursor < samples.len() {
            let take = (samples.len() - cursor).min(IN_CHUNK_SAMPLES);
            self.in_buf[0].clear();
            self.in_buf[0].extend_from_slice(&samples[cursor..cursor + take]);
            if take < IN_CHUNK_SAMPLES {
                self.in_buf[0].resize(IN_CHUNK_SAMPLES, 0.0);
            }
            let (_in_used, out_produced) = self
                .resampler
                .process_into_buffer(&self.in_buf, &mut self.out_buf, None)
                .context("rubato process")?;
            self.carry
                .extend_from_slice(&self.out_buf[0][..out_produced]);
            cursor += take;
            while self.carry.len() >= FRAME_SAMPLES {
                wait_for_frame_room(&self.gate, &self.played_ms).await?;
                self.emit_frame_from_carry()
                    .await
                    .map_err(OutboundPushError::Other)?;
            }
        }
        Ok(())
    }

    pub async fn flush(&mut self) -> Result<()> {
        if !self.carry.is_empty() {
            for (i, slot) in self.frame_i16.iter_mut().enumerate() {
                *slot = if i < self.carry.len() {
                    (self.carry[i].clamp(-1.0, 1.0) * 32_767.0) as i16
                } else {
                    0
                };
            }
            self.write_encoded_frame().await?;
            self.carry.clear();
        }
        debug!(frames = self.frames_written, "outbound TTS audio drained");
        Ok(())
    }

    async fn emit_frame_from_carry(&mut self) -> Result<()> {
        for (slot, &s) in self.frame_i16.iter_mut().zip(&self.carry[..FRAME_SAMPLES]) {
            *slot = (s.clamp(-1.0, 1.0) * 32_767.0) as i16;
        }
        self.carry.drain(..FRAME_SAMPLES);
        self.write_encoded_frame().await
    }

    async fn write_encoded_frame(&mut self) -> Result<()> {
        let n = self
            .encoder
            .encode(&self.frame_i16, &mut self.opus_buf)
            .context("opus encode")?;
        let sample = Sample {
            data: bytes::Bytes::copy_from_slice(&self.opus_buf[..n]),
            duration: Duration::from_millis(FRAME_MS as u64),
            ..Default::default()
        };
        self.track
            .write_sample(&sample)
            .await
            .context("write opus sample")?;
        let start = *self.start.get_or_insert_with(Instant::now);
        self.frames_written += 1;
        self.played_ms
            .store(self.frames_written * FRAME_MS as u64, Ordering::Release);
        self.gate.on_frame_sent();
        let target = start + Duration::from_millis(FRAME_MS as u64) * self.frames_written as u32;
        let now = Instant::now();
        if target > now {
            tokio::time::sleep(target - now).await;
        }
        Ok(())
    }
}

async fn wait_for_frame_room(
    gate: &QueueGate,
    played_ms: &Arc<AtomicU64>,
) -> Result<(), OutboundPushError> {
    let frame_ms = FRAME_MS as u64;
    if gate.try_push(frame_ms).is_ok() {
        return Ok(());
    }
    let mut last_played = played_ms.load(Ordering::Acquire);
    let mut last_progress = Instant::now();
    loop {
        tokio::time::sleep(Duration::from_millis(FRAME_GATE_POLL_MS)).await;
        if gate.try_push(frame_ms).is_ok() {
            return Ok(());
        }
        let now_played = played_ms.load(Ordering::Acquire);
        if now_played != last_played {
            last_played = now_played;
            last_progress = Instant::now();
        } else if last_progress.elapsed() >= Duration::from_millis(STALL_TIMEOUT_MS) {
            return Err(OutboundPushError::QueueFull {
                queued_ms: gate.queued_ms().saturating_add(frame_ms),
                cap_ms: gate.cap_ms(),
            });
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct StalledFakePacer {
        gate: QueueGate,
    }

    impl StalledFakePacer {
        fn new(cap_ms: u64) -> Self {
            Self {
                gate: QueueGate::new(cap_ms),
            }
        }

        fn push_chunk_ms(&self, chunk_ms: u64) -> Result<(), OutboundPushError> {
            self.gate.try_push(chunk_ms)
        }
    }

    #[test]
    fn under_cap_pushes_succeed() {
        let pacer = StalledFakePacer::new(200);

        for _ in 0..4 {
            pacer
                .push_chunk_ms(40)
                .expect("under-cap push must succeed");
        }
        assert_eq!(pacer.gate.queued_ms(), 160);
    }

    #[test]
    fn over_cap_push_returns_queue_full() {
        let pacer = StalledFakePacer::new(200);

        for _ in 0..4 {
            pacer.push_chunk_ms(50).expect("up-to-cap pushes succeed");
        }
        assert_eq!(pacer.gate.queued_ms(), 200);

        let err = pacer
            .push_chunk_ms(1)
            .expect_err("over-cap push must error");
        match err {
            OutboundPushError::QueueFull { queued_ms, cap_ms } => {
                assert_eq!(queued_ms, 201);
                assert_eq!(cap_ms, 200);
            }
            other => panic!("expected QueueFull, got {other:?}"),
        }

        assert_eq!(pacer.gate.queued_ms(), 200);
    }

    #[test]
    fn frame_send_decrements_queued_ms() {
        let gate = QueueGate::new(200);
        gate.try_push(100).unwrap();
        assert_eq!(gate.queued_ms(), 100);

        for _ in 0..5 {
            gate.on_frame_sent();
        }
        assert_eq!(gate.queued_ms(), 0);
    }

    #[test]
    fn frame_send_saturates_at_zero() {
        let gate = QueueGate::new(200);

        for _ in 0..3 {
            gate.on_frame_sent();
        }
        assert_eq!(gate.queued_ms(), 0);
    }

    #[test]
    fn read_queue_cap_ms_default_is_5000() {
        assert_eq!(DEFAULT_OUTBOUND_QUEUE_CAP_MS, 5_000);
    }
}
