#![allow(dead_code)]

use std::fs::{File, OpenOptions};
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::time::{Instant, SystemTime};

use serde_json::json;
use tracing::warn;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Channel {
    MicIn,
    TtsOut,
}

impl Channel {
    pub fn as_str(self) -> &'static str {
        match self {
            Channel::MicIn => "mic_in",
            Channel::TtsOut => "tts_out",
        }
    }

    pub fn parse(s: &str) -> Option<Self> {
        match s {
            "mic_in" => Some(Channel::MicIn),
            "tts_out" => Some(Channel::TtsOut),
            _ => None,
        }
    }

    pub fn sample_rate(self) -> u32 {
        match self {
            Channel::MicIn => 16_000,
            Channel::TtsOut => 24_000,
        }
    }
}

struct Track {
    channel: Channel,
    path: PathBuf,
    state: Mutex<TrackState>,
}

struct TrackState {
    fh: Option<File>,
    first_ns: Option<u128>,
    total_samples: u64,
}

impl Track {
    fn new(session_id: &str, channel: Channel, dir: &Path) -> Self {
        let path = dir.join(format!("{}.audio_{}.raw", session_id, channel.as_str()));
        let fh = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&path)
            .map_err(|err| {
                warn!(error = %err, path = %path.display(), "open audio track");
                err
            })
            .ok();
        Self {
            channel,
            path,
            state: Mutex::new(TrackState {
                fh,
                first_ns: None,
                total_samples: 0,
            }),
        }
    }

    fn append_pcm16(&self, pcm: &[u8], session_start: Instant) {
        if pcm.is_empty() {
            return;
        }
        let mut g = self.state.lock().expect("audio track poisoned");
        if g.first_ns.is_none() {
            g.first_ns = Some(session_start.elapsed().as_nanos());
        }
        if let Some(fh) = g.fh.as_mut() {
            if let Err(err) = fh.write_all(pcm) {
                warn!(error = %err, "write audio track");
                return;
            }
            g.total_samples = g.total_samples.saturating_add((pcm.len() / 2) as u64);
        }
    }

    fn append_f32(&self, samples: &[f32], session_start: Instant) {
        if samples.is_empty() {
            return;
        }
        let mut buf = Vec::with_capacity(samples.len() * 2);
        for &s in samples {
            let clipped = s.clamp(-1.0, 1.0);
            let i = (clipped * 32767.0).round() as i16;
            buf.extend_from_slice(&i.to_le_bytes());
        }
        self.append_pcm16(&buf, session_start);
    }

    fn offset_ms(&self, session_start_ns: u128) -> u64 {
        let g = self.state.lock().expect("audio track poisoned");
        match g.first_ns {
            None => 0,
            Some(first) => first.saturating_sub(session_start_ns) as u64 / 1_000_000,
        }
    }

    fn slice(&self, from_ms: u64, to_ms: u64) -> Vec<u8> {
        let sr = self.channel.sample_rate() as u64;
        let byte_offset = from_ms.saturating_mul(sr).saturating_mul(2) / 1000;
        let end_offset = if to_ms == 0 {
            None
        } else {
            Some(to_ms.saturating_mul(sr).saturating_mul(2) / 1000)
        };
        let _g = self.state.lock().expect("audio track poisoned");
        match File::open(&self.path) {
            Err(err) => {
                warn!(error = %err, path = %self.path.display(), "open audio track for slice");
                Vec::new()
            }
            Ok(mut fh) => {
                if fh.seek(SeekFrom::Start(byte_offset)).is_err() {
                    return Vec::new();
                }
                match end_offset {
                    None => {
                        let mut buf = Vec::new();
                        if fh.read_to_end(&mut buf).is_err() {
                            return Vec::new();
                        }
                        buf
                    }
                    Some(end) => {
                        let n = end.saturating_sub(byte_offset) as usize;
                        let mut buf = vec![0u8; n];
                        let read = fh.read(&mut buf).unwrap_or(0);
                        buf.truncate(read);
                        buf
                    }
                }
            }
        }
    }

    fn close(&self) {
        let mut g = self.state.lock().expect("audio track poisoned");
        if let Some(fh) = g.fh.as_mut() {
            let _ = fh.flush();
        }
        g.fh = None;
    }

    fn total_samples(&self) -> u64 {
        self.state
            .lock()
            .expect("audio track poisoned")
            .total_samples
    }
}

pub struct AudioStore {
    pub session_id: String,
    pub session_dir: Option<PathBuf>,
    pub session_start_wall: f64,
    pub session_start_instant: Instant,
    pub session_start_ns: u128,
    mic_in: Track,
    tts_out: Track,
}

impl AudioStore {
    pub fn new(session_id: String, session_dir: Option<PathBuf>) -> Self {
        if let Some(dir) = session_dir.as_ref() {
            let _ = std::fs::create_dir_all(dir);
        }
        let start_instant = Instant::now();
        let start_wall = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .map(|d| d.as_secs_f64())
            .unwrap_or(0.0);
        let track_dir = session_dir.clone().unwrap_or_else(|| PathBuf::from(""));
        let mic_in = Track::new(&session_id, Channel::MicIn, &track_dir);
        let tts_out = Track::new(&session_id, Channel::TtsOut, &track_dir);
        Self {
            session_id,
            session_dir,
            session_start_wall: start_wall,
            session_start_instant: start_instant,
            session_start_ns: 0,
            mic_in,
            tts_out,
        }
    }

    pub fn append_mic_in_f32(&self, samples: &[f32]) {
        self.mic_in.append_f32(samples, self.session_start_instant);
    }

    pub fn append_tts_out_f32(&self, samples: &[f32]) {
        self.tts_out.append_f32(samples, self.session_start_instant);
    }

    pub fn append_tts_out_pcm16(&self, pcm: &[u8]) {
        self.tts_out.append_pcm16(pcm, self.session_start_instant);
    }

    fn track(&self, channel: Channel) -> &Track {
        match channel {
            Channel::MicIn => &self.mic_in,
            Channel::TtsOut => &self.tts_out,
        }
    }

    pub fn track_offset_ms(&self, channel: Channel) -> u64 {
        self.track(channel).offset_ms(self.session_start_ns)
    }

    pub fn slice(&self, channel: Channel, from_ms: u64, to_ms: u64) -> Vec<u8> {
        let track = self.track(channel);
        let offset = track.offset_ms(self.session_start_ns);
        let adjusted_from = from_ms.saturating_sub(offset);
        let adjusted_to = if to_ms > 0 {
            to_ms.saturating_sub(offset)
        } else {
            0
        };
        track.slice(adjusted_from, adjusted_to)
    }

    pub fn close(&self) {
        if let Some(dir) = self.session_dir.as_ref() {
            let sidecar = dir.join(format!("{}.audio.json", self.session_id));
            let body = json!({
                "session_id": self.session_id,
                "started_at": self.session_start_wall,
                "tracks": {
                    "mic_in": {
                        "sample_rate": Channel::MicIn.sample_rate(),
                        "samples": self.mic_in.total_samples(),
                        "offset_ms": self.mic_in.offset_ms(self.session_start_ns),
                    },
                    "tts_out": {
                        "sample_rate": Channel::TtsOut.sample_rate(),
                        "samples": self.tts_out.total_samples(),
                        "offset_ms": self.tts_out.offset_ms(self.session_start_ns),
                    },
                }
            });
            if let Err(err) =
                std::fs::write(&sidecar, serde_json::to_string(&body).unwrap_or_default())
            {
                warn!(error = %err, path = %sidecar.display(), "write audio sidecar");
            }
        }
        self.mic_in.close();
        self.tts_out.close();
    }
}

pub fn wav_header(num_samples: u64, sample_rate: u32) -> Vec<u8> {
    let byte_rate = sample_rate as u64 * 2;
    let block_align: u16 = 2;
    let data_bytes = num_samples * 2;
    let mut buf = Vec::with_capacity(44);
    buf.extend_from_slice(b"RIFF");
    buf.extend_from_slice(&((36u64 + data_bytes) as u32).to_le_bytes());
    buf.extend_from_slice(b"WAVE");
    buf.extend_from_slice(b"fmt ");
    buf.extend_from_slice(&16u32.to_le_bytes());
    buf.extend_from_slice(&1u16.to_le_bytes());
    buf.extend_from_slice(&1u16.to_le_bytes());
    buf.extend_from_slice(&sample_rate.to_le_bytes());
    buf.extend_from_slice(&(byte_rate as u32).to_le_bytes());
    buf.extend_from_slice(&block_align.to_le_bytes());
    buf.extend_from_slice(&16u16.to_le_bytes());
    buf.extend_from_slice(b"data");
    buf.extend_from_slice(&(data_bytes as u32).to_le_bytes());
    buf
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp_dir() -> PathBuf {
        let p = std::env::temp_dir().join(format!(
            "speaches-plus-audio-{}-{}",
            std::process::id(),
            uuid::Uuid::new_v4().simple()
        ));
        std::fs::create_dir_all(&p).unwrap();
        p
    }

    #[test]
    fn append_and_slice_mic_in_round_trip() {
        let dir = temp_dir();
        let store = AudioStore::new("sess_audio".into(), Some(dir.clone()));
        let samples = vec![0.0_f32, 0.5, -0.5, 1.0];
        store.append_mic_in_f32(&samples);
        let raw = dir.join("sess_audio.audio_mic_in.raw");
        assert!(raw.exists());
        let bytes = std::fs::read(&raw).unwrap();
        assert_eq!(bytes.len(), samples.len() * 2);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn slice_clamps_and_truncates() {
        let dir = temp_dir();
        let store = AudioStore::new("sess_slice".into(), Some(dir.clone()));
        let samples = vec![0.5_f32; 16_000];
        store.append_mic_in_f32(&samples);
        let offset = store.track_offset_ms(Channel::MicIn);
        let chunk = store.slice(Channel::MicIn, offset, offset + 100);
        assert_eq!(chunk.len(), 16_000 * 100 / 1000 * 2);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn channel_round_trip() {
        assert_eq!(Channel::parse("mic_in"), Some(Channel::MicIn));
        assert_eq!(Channel::parse("tts_out"), Some(Channel::TtsOut));
        assert_eq!(Channel::parse("nope"), None);
        assert_eq!(Channel::MicIn.sample_rate(), 16_000);
        assert_eq!(Channel::TtsOut.sample_rate(), 24_000);
    }

    #[test]
    fn wav_header_format() {
        let h = wav_header(8000, 16_000);
        assert_eq!(&h[0..4], b"RIFF");
        assert_eq!(&h[8..12], b"WAVE");
        assert_eq!(&h[12..16], b"fmt ");
        assert_eq!(&h[36..40], b"data");
        let data_bytes = u32::from_le_bytes(h[40..44].try_into().unwrap());
        assert_eq!(data_bytes, 8000 * 2);
    }

    #[test]
    fn close_writes_sidecar() {
        let dir = temp_dir();
        let store = AudioStore::new("sess_side".into(), Some(dir.clone()));
        let samples = vec![0.1_f32; 1600];
        store.append_mic_in_f32(&samples);
        store.close();
        let sidecar = dir.join("sess_side.audio.json");
        assert!(sidecar.exists());
        let body = std::fs::read_to_string(&sidecar).unwrap();
        assert!(body.contains("\"sample_rate\":16000"));
        let _ = std::fs::remove_dir_all(&dir);
    }
}
