use std::borrow::Cow;
use std::path::Path;
use std::sync::{Arc, Mutex};

use anyhow::{anyhow, Result};
use ort::session::{builder::GraphOptimizationLevel, Session, SessionInputValue};
use ort::value::{Tensor, ValueType};
use tracing::debug;

use super::defaults;

pub const SAMPLE_RATE: usize = defaults::vad::SAMPLE_RATE;
pub const WINDOW_SAMPLES: usize = defaults::vad::WINDOW_SAMPLES;
pub const PREFIX_PADDING_MS: usize = defaults::turn_detection::PREFIX_PADDING_MS as usize;
pub const SILENCE_DURATION_MS: usize = defaults::turn_detection::SILENCE_DURATION_MS as usize;
pub const SPEECH_THRESHOLD: f32 = defaults::turn_detection::THRESHOLD;
pub const CONTEXT_SAMPLES: usize = defaults::vad::CONTEXT_SAMPLES;
pub const INPUT_SAMPLES: usize = defaults::vad::INPUT_SAMPLES;
pub const MIN_SPEECH_MS: u64 = defaults::buffer::MIN_SPEECH_MS;
pub const VAD_FAILURE_THRESHOLD: u32 = defaults::vad::FAILURE_THRESHOLD;

pub const MIN_SPEECH_DURATION_MS: u32 = defaults::vad_window::MIN_SPEECH_DURATION_MS;
pub const MAX_SPEECH_DURATION_S: f32 = defaults::vad_window::MAX_SPEECH_DURATION_S;
pub const MAX_VAD_WINDOW_SAMPLES: usize = defaults::vad_window::MAX_VAD_WINDOW_SAMPLES;
const MIN_SILENCE_AT_MAX_SPEECH_MS: u32 = defaults::vad_window::MIN_SILENCE_AT_MAX_SPEECH_MS;
const MAX_SPEECH_CARRY_OVER_MS: u32 = defaults::vad_window::MAX_SPEECH_CARRY_OVER_MS;
const NEG_THRESHOLD_DELTA: f32 = defaults::vad_window::NEG_THRESHOLD_DELTA;
const NEG_THRESHOLD_FLOOR: f32 = defaults::vad_window::NEG_THRESHOLD_FLOOR;

pub(crate) fn ort_err<R>(err: ort::Error<R>) -> anyhow::Error {
    anyhow!("{err}")
}

fn commit_tail_disabled() -> bool {
    std::env::var(defaults::env::VAD_COMMIT_TAIL).is_ok_and(|v| v.trim() == "0")
}

const MODERN_STATE_UNITS: usize = 128;
const LEGACY_HC_STATE_UNITS: usize = 64;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum SileroIo {
    Modern { feed_sr: bool },
    LegacyHc,
}

fn detect_silero_io(session: &Session) -> (SileroIo, usize) {
    let mut has_state = false;
    let mut has_h = false;
    let mut has_c = false;
    let mut has_sr = false;
    let mut units = 0usize;
    for outlet in session.inputs() {
        match outlet.name() {
            "state" | "h" => {
                if outlet.name() == "state" {
                    has_state = true;
                } else {
                    has_h = true;
                }
                if let ValueType::Tensor { shape, .. } = outlet.dtype() {
                    if let Some(&last) = shape.last() {
                        if last > 0 {
                            units = last as usize;
                        }
                    }
                }
            }
            "c" => has_c = true,
            "sr" => has_sr = true,
            _ => {}
        }
    }
    if has_h && has_c {
        return (
            SileroIo::LegacyHc,
            if units > 0 { units } else { LEGACY_HC_STATE_UNITS },
        );
    }
    if !has_state {
        let names: Vec<&str> = session.inputs().iter().map(|o| o.name()).collect();
        debug!(
            ?names,
            "silero: unrecognized input signature, assuming modern state/sr layout"
        );
    }
    (
        SileroIo::Modern { feed_sr: has_sr },
        if units > 0 { units } else { MODERN_STATE_UNITS },
    )
}

pub struct VadModel {
    session: Arc<Mutex<Session>>,
    io: SileroIo,
    state: Vec<f32>,
    legacy_c: Vec<f32>,
    context: Vec<f32>,
    sr: i64,
}

impl VadModel {
    #[allow(dead_code)]
    pub fn load(model_path: impl AsRef<Path>) -> Result<Self> {
        let session = Session::builder()
            .map_err(ort_err)?
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .map_err(ort_err)?
            .with_intra_threads(1)
            .map_err(ort_err)?
            .commit_from_file(model_path.as_ref())
            .map_err(ort_err)?;
        Ok(Self::from_session(Arc::new(Mutex::new(session))))
    }

    pub fn from_session(session: Arc<Mutex<Session>>) -> Self {
        let (io, units) = {
            let guard = session
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            detect_silero_io(&guard)
        };
        debug!(?io, state_units = units, "silero io signature");
        Self {
            session,
            io,
            state: vec![0.0; 2 * units],
            legacy_c: if io == SileroIo::LegacyHc {
                vec![0.0; 2 * units]
            } else {
                Vec::new()
            },
            context: if io == SileroIo::LegacyHc {
                Vec::new()
            } else {
                vec![0.0; CONTEXT_SAMPLES]
            },
            sr: SAMPLE_RATE as i64,
        }
    }

    fn state_tensor(units2: usize, data: &[f32]) -> Result<Tensor<f32>> {
        Tensor::<f32>::from_array(([2usize, 1, units2 / 2], data.to_vec().into_boxed_slice()))
            .map_err(ort_err)
    }

    pub fn process_window(&mut self, window: &[f32]) -> Result<f32> {
        if window.len() != WINDOW_SAMPLES {
            return Err(anyhow!(
                "expected window of {} samples, got {}",
                WINDOW_SAMPLES,
                window.len()
            ));
        }
        let mut full_input = Vec::with_capacity(self.context.len() + WINDOW_SAMPLES);
        full_input.extend_from_slice(&self.context);
        full_input.extend_from_slice(window);
        if !self.context.is_empty() {
            let n = self.context.len();
            self.context.copy_from_slice(&window[WINDOW_SAMPLES - n..]);
        }

        let input_len = full_input.len();
        let audio = Tensor::<f32>::from_array(([1usize, input_len], full_input.into_boxed_slice()))
            .map_err(ort_err)?;
        let sr_tensor =
            Tensor::<i64>::from_array(((), vec![self.sr].into_boxed_slice())).map_err(ort_err)?;

        let mut feed: Vec<(Cow<str>, SessionInputValue)> = vec![("input".into(), audio.into())];
        match self.io {
            SileroIo::Modern { feed_sr } => {
                feed.push((
                    "state".into(),
                    Self::state_tensor(self.state.len(), &self.state)?.into(),
                ));
                if feed_sr {
                    feed.push(("sr".into(), sr_tensor.into()));
                }
            }
            SileroIo::LegacyHc => {
                feed.push((
                    "h".into(),
                    Self::state_tensor(self.state.len(), &self.state)?.into(),
                ));
                feed.push((
                    "c".into(),
                    Self::state_tensor(self.legacy_c.len(), &self.legacy_c)?.into(),
                ));
                feed.push(("sr".into(), sr_tensor.into()));
            }
        }

        let (prob, new_state, new_c) = {
            let mut session = self
                .session
                .lock()
                .map_err(|_| anyhow!("VAD session poisoned"))?;
            let outputs = session.run(feed).map_err(ort_err)?;
            let (_, prob_data) = outputs["output"]
                .try_extract_tensor::<f32>()
                .map_err(ort_err)?;
            let prob = *prob_data.first().ok_or_else(|| anyhow!("empty output"))?;
            match self.io {
                SileroIo::Modern { .. } => {
                    let (_, state_data) = outputs["stateN"]
                        .try_extract_tensor::<f32>()
                        .map_err(ort_err)?;
                    (prob, state_data.to_vec(), Vec::new())
                }
                SileroIo::LegacyHc => {
                    let (_, hn) = outputs["hn"].try_extract_tensor::<f32>().map_err(ort_err)?;
                    let (_, cn) = outputs["cn"].try_extract_tensor::<f32>().map_err(ort_err)?;
                    (prob, hn.to_vec(), cn.to_vec())
                }
            }
        };

        if new_state.len() != self.state.len() {
            return Err(anyhow!(
                "state length mismatch: got {}, expected {}",
                new_state.len(),
                self.state.len()
            ));
        }
        self.state.copy_from_slice(&new_state);
        if !self.legacy_c.is_empty() {
            if new_c.len() != self.legacy_c.len() {
                return Err(anyhow!(
                    "cn length mismatch: got {}, expected {}",
                    new_c.len(),
                    self.legacy_c.len()
                ));
            }
            self.legacy_c.copy_from_slice(&new_c);
        }
        Ok(prob)
    }

    pub fn reset(&mut self) {
        self.state.fill(0.0);
        self.legacy_c.fill(0.0);
        self.context.fill(0.0);
    }
}

#[derive(Debug)]
pub enum VadEvent {
    SpeechStarted {
        item_id: String,
        audio_start_ms: u64,
    },
    SpeechCommitted {
        item_id: String,
        audio_end_ms: u64,
        audio: Vec<f32>,
        speech_samples: usize,
    },

    Failed {
        reason: String,
    },
}

pub trait TurnDetectionRead: Send + Sync {
    fn threshold(&self) -> f32;
    fn prefix_padding_samples(&self) -> usize;
    fn silence_duration_samples(&self) -> usize;

    fn neg_threshold(&self) -> f32 {
        (self.threshold() - NEG_THRESHOLD_DELTA).max(NEG_THRESHOLD_FLOOR)
    }

    fn min_speech_duration_ms(&self) -> u32 {
        MIN_SPEECH_DURATION_MS
    }

    fn max_speech_duration_s(&self) -> f32 {
        MAX_SPEECH_DURATION_S
    }
}

pub trait VadInfer: Send {
    fn process_window(&mut self, window: &[f32]) -> Result<f32>;
    fn reset(&mut self);
}

impl VadInfer for VadModel {
    fn process_window(&mut self, window: &[f32]) -> Result<f32> {
        VadModel::process_window(self, window)
    }
    fn reset(&mut self) {
        VadModel::reset(self)
    }
}

#[derive(Clone, Debug)]
pub struct VadOptions {
    pub threshold: f32,
    pub neg_threshold: Option<f32>,
    pub min_speech_duration_ms: u32,
    pub max_speech_duration_s: f32,
    pub min_silence_duration_ms: u32,
    pub speech_pad_ms: u32,
}

impl Default for VadOptions {
    fn default() -> Self {
        Self {
            threshold: SPEECH_THRESHOLD,
            neg_threshold: None,
            min_speech_duration_ms: MIN_SPEECH_DURATION_MS,
            max_speech_duration_s: MAX_SPEECH_DURATION_S,
            min_silence_duration_ms: SILENCE_DURATION_MS as u32,
            speech_pad_ms: PREFIX_PADDING_MS as u32,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SpeechTimestamp {
    pub start: usize,

    pub end: usize,
}

pub fn get_speech_timestamps<M: VadInfer>(
    model: &mut M,
    audio: &[f32],
    opts: &VadOptions,
    sample_rate: usize,
) -> Result<Vec<SpeechTimestamp>> {
    let window_size = WINDOW_SAMPLES;
    let audio_length = audio.len();

    let pad = (window_size - audio_length % window_size) % window_size;

    model.reset();
    let total_windows = (audio_length + pad) / window_size;
    let mut probs: Vec<f32> = Vec::with_capacity(total_windows);
    let mut window_buf = vec![0.0_f32; window_size];
    for w in 0..total_windows {
        let start = w * window_size;
        let end = (start + window_size).min(audio_length);
        let copy_len = end - start;
        window_buf[..copy_len].copy_from_slice(&audio[start..end]);
        if copy_len < window_size {
            for s in &mut window_buf[copy_len..] {
                *s = 0.0;
            }
        }
        probs.push(model.process_window(&window_buf)?);
    }

    model.reset();

    Ok(speech_timestamps_from_probs(
        &probs,
        audio_length,
        opts,
        sample_rate,
    ))
}

pub fn speech_timestamps_from_probs(
    probs: &[f32],
    audio_length: usize,
    opts: &VadOptions,
    sample_rate: usize,
) -> Vec<SpeechTimestamp> {
    let neg_threshold = opts
        .neg_threshold
        .unwrap_or_else(|| (opts.threshold - NEG_THRESHOLD_DELTA).max(NEG_THRESHOLD_FLOOR));

    let window_size = WINDOW_SAMPLES;
    let min_speech_samples = sample_rate * opts.min_speech_duration_ms as usize / 1000;
    let speech_pad_samples = sample_rate * opts.speech_pad_ms as usize / 1000;
    let max_speech_samples = ((sample_rate as f32 * opts.max_speech_duration_s) as usize)
        .saturating_sub(window_size)
        .saturating_sub(2 * speech_pad_samples);
    let min_silence_samples = sample_rate * opts.min_silence_duration_ms as usize / 1000;
    let min_silence_samples_at_max_speech =
        sample_rate * MIN_SILENCE_AT_MAX_SPEECH_MS as usize / 1000;

    let mut speeches: Vec<SpeechTimestamp> = Vec::new();
    let mut triggered = false;
    let mut current_start: usize = 0;
    let mut have_current = false;
    let mut temp_end: usize = 0;
    let mut prev_end: usize = 0;
    let mut next_start: usize = 0;

    for (i, &prob) in probs.iter().enumerate() {
        let pos = window_size * i;

        if prob >= opts.threshold && temp_end != 0 {
            temp_end = 0;
            if next_start < prev_end {
                next_start = pos;
            }
        }

        if prob >= opts.threshold && !triggered {
            triggered = true;
            current_start = pos;
            have_current = true;
            continue;
        }

        if triggered && pos - current_start > max_speech_samples {
            if prev_end != 0 {
                speeches.push(SpeechTimestamp {
                    start: current_start,
                    end: prev_end,
                });
                have_current = false;
                if next_start < prev_end {
                    triggered = false;
                } else {
                    current_start = next_start;
                    have_current = true;
                }
                prev_end = 0;
                next_start = 0;
                temp_end = 0;
            } else {
                speeches.push(SpeechTimestamp {
                    start: current_start,
                    end: pos,
                });
                have_current = false;
                prev_end = 0;
                next_start = 0;
                temp_end = 0;
                triggered = false;
                continue;
            }
        }

        if prob < neg_threshold && triggered {
            if temp_end == 0 {
                temp_end = pos;
            }
            if pos.saturating_sub(temp_end) > min_silence_samples_at_max_speech {
                prev_end = temp_end;
            }
            if pos.saturating_sub(temp_end) < min_silence_samples {
                continue;
            }
            let seg_end = temp_end;
            if have_current
                && seg_end > current_start
                && seg_end - current_start > min_speech_samples
            {
                speeches.push(SpeechTimestamp {
                    start: current_start,
                    end: seg_end,
                });
            }
            have_current = false;
            prev_end = 0;
            next_start = 0;
            temp_end = 0;
            triggered = false;
            continue;
        }
    }

    if have_current
        && audio_length > current_start
        && audio_length - current_start > min_speech_samples
    {
        speeches.push(SpeechTimestamp {
            start: current_start,
            end: audio_length,
        });
    }

    let n = speeches.len();
    for i in 0..n {
        if i == 0 {
            speeches[i].start = speeches[i].start.saturating_sub(speech_pad_samples);
        }
        if i != n - 1 {
            let next_start_pos = speeches[i + 1].start;
            let cur_end = speeches[i].end;
            let silence = next_start_pos.saturating_sub(cur_end);
            if silence < 2 * speech_pad_samples {
                let half = silence / 2;
                speeches[i].end += half;
                speeches[i + 1].start = speeches[i + 1].start.saturating_sub(half);
            } else {
                speeches[i].end = (speeches[i].end + speech_pad_samples).min(audio_length);
                speeches[i + 1].start = speeches[i + 1].start.saturating_sub(speech_pad_samples);
            }
        } else {
            speeches[i].end = (speeches[i].end + speech_pad_samples).min(audio_length);
        }
    }

    speeches
}

pub fn to_ms_speech_timestamps(timestamps: &[SpeechTimestamp]) -> Vec<SpeechTimestamp> {
    let div = SAMPLE_RATE / 1000;
    timestamps
        .iter()
        .map(|t| SpeechTimestamp {
            start: t.start / div,
            end: t.end / div,
        })
        .collect()
}

pub struct VadProcessor<M: VadInfer = VadModel> {
    model: M,

    buffer: Vec<f32>,

    pending_audio: Vec<f32>,

    probs: std::collections::VecDeque<f32>,

    probs_start_window: usize,

    duration_samples: usize,

    audio_start_ms: Option<u64>,

    audio_end_ms: Option<u64>,
    current_item: Option<String>,
    pending: Vec<VadEvent>,
    carry_over_samples: usize,
    carry_grace_until_samples: usize,
    td: Option<Arc<dyn TurnDetectionRead>>,
}

const MAX_PROB_RING: usize = MAX_VAD_WINDOW_SAMPLES.div_ceil(WINDOW_SAMPLES);

impl<M: VadInfer> VadProcessor<M> {
    pub fn new(model: M) -> Self {
        Self {
            model,
            buffer: Vec::with_capacity(SAMPLE_RATE * 30),
            pending_audio: Vec::with_capacity(WINDOW_SAMPLES * 2),
            probs: std::collections::VecDeque::with_capacity(MAX_PROB_RING),
            probs_start_window: 0,
            duration_samples: 0,
            audio_start_ms: None,
            audio_end_ms: None,
            current_item: None,
            pending: Vec::new(),
            carry_over_samples: 0,
            carry_grace_until_samples: 0,
            td: None,
        }
    }

    pub fn with_turn_detection(mut self, td: Arc<dyn TurnDetectionRead>) -> Self {
        self.td = Some(td);
        self
    }

    fn options(&self) -> VadOptions {
        if let Some(td) = &self.td {
            VadOptions {
                threshold: td.threshold(),
                neg_threshold: Some(td.neg_threshold()),
                min_speech_duration_ms: td.min_speech_duration_ms(),
                max_speech_duration_s: td.max_speech_duration_s(),
                min_silence_duration_ms: (td.silence_duration_samples() * 1000 / SAMPLE_RATE)
                    as u32,
                speech_pad_ms: (td.prefix_padding_samples() * 1000 / SAMPLE_RATE) as u32,
            }
        } else {
            VadOptions::default()
        }
    }

    fn duration_ms(&self) -> u64 {
        (self.duration_samples * 1000 / SAMPLE_RATE) as u64
    }

    pub fn current_speech_audio(&self) -> Option<(String, Vec<f32>)> {
        let started_ms = self.audio_start_ms?;
        let item_id = self.current_item.clone()?;
        let start_sample = (started_ms as usize) * (SAMPLE_RATE / 1000);
        if start_sample >= self.buffer.len() {
            return None;
        }

        Some((item_id, self.buffer[start_sample..].to_vec()))
    }

    pub fn push(&mut self, samples: &[f32]) -> Result<()> {
        if samples.is_empty() {
            return Ok(());
        }
        self.buffer.extend_from_slice(samples);
        self.duration_samples = self.buffer.len();

        if self.audio_end_ms.is_some() {
            return Ok(());
        }

        self.pending_audio.extend_from_slice(samples);
        while self.pending_audio.len() >= WINDOW_SAMPLES {
            let prob = self
                .model
                .process_window(&self.pending_audio[..WINDOW_SAMPLES])?;
            self.pending_audio.drain(..WINDOW_SAMPLES);
            if self.probs.len() == MAX_PROB_RING {
                self.probs.pop_front();
                self.probs_start_window += 1;
            }
            self.probs.push_back(prob);
        }

        if self.probs.is_empty() {
            return Ok(());
        }

        let opts = self.options();

        let ring_samples = self.probs.len() * WINDOW_SAMPLES;
        let probs_vec: Vec<f32> = self.probs.iter().copied().collect();
        let timestamps_samples =
            speech_timestamps_from_probs(&probs_vec, ring_samples, &opts, SAMPLE_RATE);
        let timestamps = to_ms_speech_timestamps(&timestamps_samples);
        let ring_ms = (ring_samples * 1000 / SAMPLE_RATE) as u64;
        let duration_ms = self.duration_ms();
        let last = timestamps.last().copied();

        if self.audio_start_ms.is_none() {
            if let Some(ts) = last {
                let audio_start_ms = duration_ms.saturating_sub(ring_ms) + ts.start as u64;
                let item_id = format!("item_{}", uuid::Uuid::new_v4().simple());
                debug!(
                    audio_start_ms,
                    item_id = %item_id,
                    "speech started"
                );
                self.audio_start_ms = Some(audio_start_ms);
                self.current_item = Some(item_id.clone());
                self.pending.push(VadEvent::SpeechStarted {
                    item_id,
                    audio_start_ms,
                });
            }
            return Ok(());
        }

        let stop_at_ms: Option<u64> = match last {
            None => {
                if self.duration_samples < self.carry_grace_until_samples {
                    None
                } else {
                    Some(duration_ms)
                }
            }
            Some(ts) => {
                self.carry_grace_until_samples = 0;
                let trailing = ring_ms.saturating_sub(ts.end as u64);
                if trailing >= opts.min_silence_duration_ms as u64 {
                    Some(duration_ms.saturating_sub(trailing))
                } else {
                    None
                }
            }
        };

        if let Some(stop_ms) = stop_at_ms {
            self.seal_utterance(stop_ms);
            return Ok(());
        }

        if let Some(start_ms) = self.audio_start_ms {
            let pad_samples = SAMPLE_RATE * opts.speech_pad_ms as usize / 1000;
            let ceiling_samples = ((SAMPLE_RATE as f32 * opts.max_speech_duration_s) as usize)
                .saturating_sub(WINDOW_SAMPLES)
                .saturating_sub(2 * pad_samples);
            let ceiling_ms = (ceiling_samples * 1000 / SAMPLE_RATE) as u64;
            if duration_ms.saturating_sub(start_ms) >= ceiling_ms {
                debug!(
                    ceiling_ms,
                    open_ms = duration_ms.saturating_sub(start_ms),
                    "max speech duration reached; sealing and carrying the tail over"
                );
                self.seal_utterance(duration_ms);
                self.carry_over_samples =
                    SAMPLE_RATE * MAX_SPEECH_CARRY_OVER_MS as usize / 1000;
            }
        }

        Ok(())
    }

    fn seal_utterance(&mut self, stop_ms: u64) {
        self.audio_end_ms = Some(stop_ms);
        let start_sample = (self.audio_start_ms.unwrap_or(0) as usize) * (SAMPLE_RATE / 1000);
        let end_sample = (stop_ms as usize) * (SAMPLE_RATE / 1000);
        let end_sample = end_sample.min(self.buffer.len());
        let start_sample = start_sample.min(end_sample);
        let tail_end = if commit_tail_disabled() {
            end_sample
        } else {
            self.buffer.len()
        };
        let utterance = self.buffer[start_sample..tail_end].to_vec();
        let speech_samples = end_sample - start_sample;
        let item_id = self
            .current_item
            .clone()
            .unwrap_or_else(|| format!("item_{}", uuid::Uuid::new_v4().simple()));
        debug!(
            samples = utterance.len(),
            speech_samples,
            ms = utterance.len() * 1000 / SAMPLE_RATE,
            item_id = %item_id,
            "speech committed"
        );
        self.pending.push(VadEvent::SpeechCommitted {
            item_id,
            audio_end_ms: stop_ms,
            audio: utterance,
            speech_samples,
        });
    }

    pub fn force_commit(&mut self) -> bool {
        if self.audio_start_ms.is_none() || self.audio_end_ms.is_some() {
            return false;
        }
        self.seal_utterance(self.duration_ms());
        true
    }

    pub fn take_events(&mut self) -> Vec<VadEvent> {
        let mut evs = std::mem::take(&mut self.pending);

        if self.audio_end_ms.is_some() {
            let carry = std::mem::take(&mut self.carry_over_samples).min(self.buffer.len());
            self.pending_audio.clear();
            self.probs.clear();
            self.probs_start_window = 0;
            self.audio_end_ms = None;
            self.model.reset();

            if carry > 0 {
                let drop_n = self.buffer.len() - carry;
                self.buffer.drain(..drop_n);
                self.duration_samples = self.buffer.len();
                let grace = self
                    .td
                    .as_ref()
                    .map(|td| td.silence_duration_samples())
                    .unwrap_or(0);
                self.carry_grace_until_samples = self.duration_samples + grace;
                let item_id = format!("item_{}", uuid::Uuid::new_v4().simple());
                self.audio_start_ms = Some(0);
                self.current_item = Some(item_id.clone());
                evs.push(VadEvent::SpeechStarted {
                    item_id,
                    audio_start_ms: 0,
                });
            } else {
                self.buffer.clear();
                self.duration_samples = 0;
                self.audio_start_ms = None;
                self.current_item = None;
                self.carry_grace_until_samples = 0;
            }
        }
        evs
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ms_from_samples_16k(n: usize) -> u64 {
        (n as u64) * 1000 / 16_000
    }

    #[test]
    fn min_speech_ms_default_is_100() {
        assert_eq!(MIN_SPEECH_MS, 100);
        assert_eq!(ms_from_samples_16k(1600), 100);
        assert_eq!(ms_from_samples_16k(800), 50);
        assert!(ms_from_samples_16k(800) < MIN_SPEECH_MS);
        assert!(ms_from_samples_16k(1600) >= MIN_SPEECH_MS);
    }

    #[test]
    fn loads_model() -> Result<()> {
        let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("models/silero_vad.onnx");
        if !path.exists() {
            eprintln!("skip: {} missing", path.display());
            return Ok(());
        }
        let mut model = VadModel::load(&path)?;
        let silence = vec![0.0f32; WINDOW_SAMPLES];
        let prob = model.process_window(&silence)?;
        assert!((0.0..=1.0).contains(&prob), "prob out of range: {prob}");
        assert!(prob < 0.3, "silence prob too high: {prob}");
        Ok(())
    }

    #[test]
    fn silero_zoo_every_signature_detects_speech() -> Result<()> {
        let Some(dir) = std::env::var_os("SILERO_ZOO_DIR") else {
            eprintln!("skip: set SILERO_ZOO_DIR to a directory of silero .onnx exports");
            return Ok(());
        };
        let fixture = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../conformance/fixtures/050-diarization-multispeaker/audio.wav");
        let mut reader = hound::WavReader::open(&fixture)
            .unwrap_or_else(|e| panic!("open {}: {e}", fixture.display()));
        assert_eq!(reader.spec().sample_rate as usize, SAMPLE_RATE);
        assert_eq!(reader.spec().channels, 1);
        let samples: Vec<f32> = reader
            .samples::<i16>()
            .map(|s| s.expect("sample") as f32 / i16::MAX as f32)
            .collect();

        let mut checked = 0usize;
        let mut entries: Vec<_> = std::fs::read_dir(dir)?
            .filter_map(|e| e.ok())
            .map(|e| e.path())
            .filter(|p| p.extension().is_some_and(|x| x == "onnx"))
            .collect();
        entries.sort();
        for path in entries {
            let mut model = VadModel::load(&path)
                .unwrap_or_else(|e| panic!("{} failed to load: {e:#}", path.display()));

            let silence = vec![0.0f32; WINDOW_SAMPLES];
            let mut silence_prob = 1.0f32;
            for _ in 0..4 {
                silence_prob = model.process_window(&silence)?;
            }
            assert!(
                silence_prob < 0.3,
                "{}: silence prob {silence_prob} too high",
                path.display()
            );

            model.reset();
            let mut max_prob = 0.0f32;
            for chunk in samples.chunks_exact(WINDOW_SAMPLES) {
                max_prob = max_prob.max(model.process_window(chunk)?);
            }
            assert!(
                max_prob > SPEECH_THRESHOLD,
                "{}: no window crossed the speech threshold (max {max_prob})",
                path.display()
            );
            eprintln!(
                "silero zoo : {} silence={silence_prob:.3} speech_max={max_prob:.3}",
                path.file_name().unwrap_or_default().to_string_lossy()
            );
            checked += 1;
        }
        assert!(checked >= 2, "zoo dir held {checked} models; expected several");
        Ok(())
    }

    struct ProbReplay {
        probs: Vec<f32>,
        idx: usize,
    }

    impl VadInfer for ProbReplay {
        fn process_window(&mut self, _window: &[f32]) -> Result<f32> {
            let p = *self
                .probs
                .get(self.idx)
                .ok_or_else(|| anyhow!("no more probs (idx {})", self.idx))?;
            self.idx += 1;
            Ok(p)
        }
        fn reset(&mut self) {
            self.idx = 0;
        }
    }

    fn opts(threshold: f32, min_silence_ms: u32, min_speech_ms: u32, pad_ms: u32) -> VadOptions {
        VadOptions {
            threshold,
            neg_threshold: None,
            min_speech_duration_ms: min_speech_ms,
            max_speech_duration_s: 30.0,
            min_silence_duration_ms: min_silence_ms,
            speech_pad_ms: pad_ms,
        }
    }

    #[test]
    fn hysteresis_silent_stays_silent() {
        let probs = vec![0.0_f32; 32];
        let mut m = ProbReplay { probs, idx: 0 };
        let audio = vec![0.0_f32; 32 * WINDOW_SAMPLES];
        let ts = get_speech_timestamps(&mut m, &audio, &opts(0.5, 100, 0, 0), SAMPLE_RATE).unwrap();
        assert!(ts.is_empty(), "expected no speech, got {ts:?}");
    }

    #[test]
    fn hysteresis_enter_at_threshold_leave_at_neg_threshold() {
        let probs = vec![0.0, 0.0, 0.7, 0.7, 0.7, 0.7, 0.4, 0.4, 0.0, 0.0];
        let mut m = ProbReplay { probs, idx: 0 };
        let audio = vec![0.0_f32; 10 * WINDOW_SAMPLES];
        let mut o = opts(0.5, 32, 0, 0);

        o.min_silence_duration_ms = 0;
        let ts = get_speech_timestamps(&mut m, &audio, &o, SAMPLE_RATE).unwrap();
        assert_eq!(ts.len(), 1, "{ts:?}");
        assert_eq!(ts[0].start, 2 * WINDOW_SAMPLES);

        assert_eq!(ts[0].end, 8 * WINDOW_SAMPLES);
    }

    #[test]
    fn hysteresis_dip_above_neg_threshold_does_not_release() {
        let probs = vec![0.0, 0.7, 0.7, 0.4, 0.7, 0.0, 0.0];
        let mut m = ProbReplay { probs, idx: 0 };
        let audio = vec![0.0_f32; 7 * WINDOW_SAMPLES];
        let mut o = opts(0.5, 0, 0, 0);
        o.min_silence_duration_ms = 0;
        let ts = get_speech_timestamps(&mut m, &audio, &o, SAMPLE_RATE).unwrap();
        assert_eq!(ts.len(), 1);
        assert_eq!(ts[0].start, WINDOW_SAMPLES);
        assert_eq!(ts[0].end, 5 * WINDOW_SAMPLES);
    }

    #[test]
    fn min_speech_filter_drops_short_segments() {
        let probs = vec![0.0, 0.7, 0.0, 0.0, 0.0, 0.0, 0.0];
        let mut m = ProbReplay { probs, idx: 0 };
        let audio = vec![0.0_f32; 7 * WINDOW_SAMPLES];

        let o = opts(0.5, 0, 200, 0);
        let ts = get_speech_timestamps(&mut m, &audio, &o, SAMPLE_RATE).unwrap();
        assert!(ts.is_empty(), "expected filter, got {ts:?}");
    }

    #[test]
    fn padding_pass_extends_segment_edges() {
        let probs = vec![0.0, 0.7, 0.7, 0.7, 0.0, 0.0];
        let mut m = ProbReplay { probs, idx: 0 };
        let audio = vec![0.0_f32; 6 * WINDOW_SAMPLES];
        let mut o = opts(0.5, 0, 0, 60);
        o.min_silence_duration_ms = 0;
        let ts = get_speech_timestamps(&mut m, &audio, &o, SAMPLE_RATE).unwrap();
        assert_eq!(ts.len(), 1);

        assert!(ts[0].start < WINDOW_SAMPLES);

        assert!(ts[0].end > 4 * WINDOW_SAMPLES);
    }

    struct FakeTd {
        threshold: f32,
        prefix_padding_ms: u32,
        silence_duration_ms: u32,
        min_speech_duration_ms: u32,
    }
    impl TurnDetectionRead for FakeTd {
        fn threshold(&self) -> f32 {
            self.threshold
        }
        fn prefix_padding_samples(&self) -> usize {
            (self.prefix_padding_ms as usize) * SAMPLE_RATE / 1000
        }
        fn silence_duration_samples(&self) -> usize {
            (self.silence_duration_ms as usize) * SAMPLE_RATE / 1000
        }
        fn min_speech_duration_ms(&self) -> u32 {
            self.min_speech_duration_ms
        }
    }

    fn td(silence_ms: u32, min_speech_ms: u32, pad_ms: u32) -> Arc<dyn TurnDetectionRead> {
        Arc::new(FakeTd {
            threshold: 0.5,
            prefix_padding_ms: pad_ms,
            silence_duration_ms: silence_ms,
            min_speech_duration_ms: min_speech_ms,
        })
    }

    fn push_n_windows<M: VadInfer>(p: &mut VadProcessor<M>, n: usize) {
        let z = vec![0.0_f32; WINDOW_SAMPLES];
        for _ in 0..n {
            p.push(&z).unwrap();
        }
    }

    #[test]
    fn driver_emits_speech_started_then_committed() {
        let probs = vec![0.0, 0.0, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.0, 0.0, 0.0, 0.0];
        let m = ProbReplay { probs, idx: 0 };
        let mut p = VadProcessor::new(m).with_turn_detection(td(64, 0, 0));
        push_n_windows(&mut p, 12);
        let evs = p.take_events();
        assert_eq!(evs.len(), 2, "{evs:?}");
        assert!(matches!(evs[0], VadEvent::SpeechStarted { .. }));
        assert!(matches!(evs[1], VadEvent::SpeechCommitted { .. }));
        if let VadEvent::SpeechCommitted { audio, .. } = &evs[1] {
            assert!(!audio.is_empty(), "committed audio should be non-empty");
        }
    }

    #[test]
    fn committed_audio_carries_the_silence_tail_for_eou() {
        let probs = vec![0.0, 0.0, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.0, 0.0, 0.0, 0.0];
        let m = ProbReplay { probs, idx: 0 };
        let mut p = VadProcessor::new(m).with_turn_detection(td(64, 0, 0));
        push_n_windows(&mut p, 12);
        let evs = p.take_events();
        let (len, speech) = evs
            .iter()
            .find_map(|e| match e {
                VadEvent::SpeechCommitted {
                    audio,
                    speech_samples,
                    ..
                } => Some((audio.len(), *speech_samples)),
                _ => None,
            })
            .expect("commit expected");
        assert!(speech > 0);
        assert!(
            len > speech,
            "committed audio must include the trailing silence the eou classifier was trained \
             on; feeding audio hard-cut at speech end costs ~5 points of complete-detection \
             (len {len} <= speech {speech})"
        );
    }

    #[test]
    fn force_commit_seals_an_open_utterance() {
        let probs = vec![0.0, 0.0, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7];
        let m = ProbReplay { probs, idx: 0 };
        let mut p = VadProcessor::new(m).with_turn_detection(td(64, 0, 0));
        push_n_windows(&mut p, 8);
        let evs = p.take_events();
        assert_eq!(evs.len(), 1, "{evs:?}");
        assert!(matches!(evs[0], VadEvent::SpeechStarted { .. }));

        assert!(p.force_commit(), "open utterance must be sealable");
        let evs = p.take_events();
        assert_eq!(evs.len(), 1, "{evs:?}");
        match &evs[0] {
            VadEvent::SpeechCommitted { audio, .. } => {
                assert!(!audio.is_empty(), "forced commit dropped the buffer")
            }
            other => panic!("expected SpeechCommitted, got {other:?}"),
        }
        assert!(p.audio_start_ms.is_none(), "take_events must reset");
    }

    #[test]
    fn gap_free_speech_seals_at_the_ceiling_instead_of_growing() {
        let windows = 16_000 * 31 / WINDOW_SAMPLES;
        let probs = vec![0.7_f32; windows];
        let m = ProbReplay { probs, idx: 0 };
        let mut p = VadProcessor::new(m).with_turn_detection(td(64, 0, 0));
        push_n_windows(&mut p, windows);
        let evs = p.take_events();

        assert!(matches!(evs[0], VadEvent::SpeechStarted { .. }), "{evs:?}");
        let committed = evs
            .iter()
            .find_map(|e| match e {
                VadEvent::SpeechCommitted { audio, .. } => Some(audio.len()),
                _ => None,
            })
            .expect("ceiling must seal the open utterance");
        let committed_ms = committed * 1000 / SAMPLE_RATE;
        assert!(
            (29_000..=30_000).contains(&committed_ms),
            "committed chunk must land just under whisper's 30s window, got {committed_ms}ms"
        );

        assert!(
            matches!(evs.last(), Some(VadEvent::SpeechStarted { .. })),
            "the continuation must open a fresh item: {evs:?}"
        );
        assert_eq!(p.audio_start_ms, Some(0), "continuation stays open");
    }

    #[test]
    fn ceiling_seal_carries_the_boundary_audio_into_the_next_item() {
        let windows = 16_000 * 31 / WINDOW_SAMPLES;
        let probs = vec![0.7_f32; windows];
        let m = ProbReplay { probs, idx: 0 };
        let mut p = VadProcessor::new(m).with_turn_detection(td(64, 0, 0));

        let mut ramp = vec![0.0_f32; WINDOW_SAMPLES];
        for (i, s) in ramp.iter_mut().enumerate() {
            *s = i as f32 / WINDOW_SAMPLES as f32;
        }
        for _ in 0..windows {
            p.push(&ramp).unwrap();
        }
        let _ = p.take_events();

        let carried = p.buffer.len();
        let carried_ms = carried * 1000 / SAMPLE_RATE;
        assert_eq!(
            carried_ms, MAX_SPEECH_CARRY_OVER_MS as usize,
            "the tail must be retained so a split word repeats rather than vanishing"
        );
        assert_eq!(
            p.duration_samples, carried,
            "the timebase must follow the retained tail"
        );
    }

    #[test]
    fn carried_tail_does_not_seal_itself_before_speech_resumes() {
        let windows = 16_000 * 31 / WINDOW_SAMPLES;
        let mut probs = vec![0.7_f32; windows];
        probs.extend(vec![0.7_f32; 40]);
        let m = ProbReplay { probs, idx: 0 };
        let mut p = VadProcessor::new(m).with_turn_detection(td(200, 0, 0));
        push_n_windows(&mut p, windows);
        let evs = p.take_events();
        let sealed_before = evs
            .iter()
            .filter(|e| matches!(e, VadEvent::SpeechCommitted { .. }))
            .count();
        assert_eq!(sealed_before, 1, "ceiling seals exactly once: {evs:?}");

        push_n_windows(&mut p, 2);
        let evs = p.take_events();
        assert!(
            !evs.iter()
                .any(|e| matches!(e, VadEvent::SpeechCommitted { .. })),
            "the carried tail must not commit itself as a fragment: {evs:?}"
        );
        assert_eq!(p.audio_start_ms, Some(0), "continuation stays open");
    }

    #[test]
    fn force_commit_is_a_noop_without_speech() {
        let probs = vec![0.0_f32; 8];
        let m = ProbReplay { probs, idx: 0 };
        let mut p = VadProcessor::new(m).with_turn_detection(td(64, 0, 0));
        push_n_windows(&mut p, 8);
        assert!(
            !p.force_commit(),
            "silence must not synthesize an utterance"
        );
        assert!(p.take_events().is_empty());
    }

    #[test]
    fn force_commit_does_not_double_commit() {
        let probs = vec![0.0, 0.0, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.0, 0.0, 0.0, 0.0];
        let m = ProbReplay { probs, idx: 0 };
        let mut p = VadProcessor::new(m).with_turn_detection(td(64, 0, 0));
        push_n_windows(&mut p, 12);
        assert!(
            !p.force_commit(),
            "an already-sealed utterance must not be sealed twice"
        );
        let evs = p.take_events();
        assert_eq!(evs.len(), 2, "{evs:?}");
    }

    #[test]
    fn driver_silent_input_emits_nothing() {
        let probs = vec![0.0_f32; 50];
        let m = ProbReplay { probs, idx: 0 };
        let mut p = VadProcessor::new(m).with_turn_detection(td(64, 0, 0));
        push_n_windows(&mut p, 50);
        let evs = p.take_events();
        assert!(evs.is_empty(), "{evs:?}");

        assert!(p.audio_start_ms.is_none());
    }

    #[test]
    fn driver_resets_state_after_commit_via_take_events() {
        let probs = vec![
            0.0, 0.7, 0.7, 0.7, 0.7, 0.7, 0.0, 0.0, 0.0, 0.0, 0.0, 0.7, 0.7, 0.7, 0.7, 0.7, 0.0,
            0.0, 0.0, 0.0,
        ];
        let m = ProbReplay { probs, idx: 0 };
        let mut p = VadProcessor::new(m).with_turn_detection(td(64, 0, 0));
        push_n_windows(&mut p, 10);
        let evs = p.take_events();
        let started = evs
            .iter()
            .filter(|e| matches!(e, VadEvent::SpeechStarted { .. }))
            .count();
        let committed = evs
            .iter()
            .filter(|e| matches!(e, VadEvent::SpeechCommitted { .. }))
            .count();
        assert_eq!((started, committed), (1, 1), "first turn: {evs:?}");

        push_n_windows(&mut p, 10);
        let evs = p.take_events();
        let started = evs
            .iter()
            .filter(|e| matches!(e, VadEvent::SpeechStarted { .. }))
            .count();
        let committed = evs
            .iter()
            .filter(|e| matches!(e, VadEvent::SpeechCommitted { .. }))
            .count();
        assert_eq!((started, committed), (1, 1), "second turn: {evs:?}");
    }

    #[test]
    fn driver_ignores_pushes_after_commit_until_take_events() {
        let probs = vec![0.0, 0.7, 0.7, 0.7, 0.0, 0.0, 0.0, 0.0];
        let m = ProbReplay { probs, idx: 0 };
        let mut p = VadProcessor::new(m).with_turn_detection(td(64, 0, 0));
        push_n_windows(&mut p, 8);

        let extra_silence = vec![0.0_f32; WINDOW_SAMPLES];
        for _ in 0..5 {
            p.push(&extra_silence).unwrap();
        }
        let evs = p.take_events();
        let committed = evs
            .iter()
            .filter(|e| matches!(e, VadEvent::SpeechCommitted { .. }))
            .count();
        assert_eq!(committed, 1, "{evs:?}");
    }
}
