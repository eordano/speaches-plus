#![allow(dead_code)]

pub mod clustering;
pub mod embedding;
pub mod embeddings_http;
pub mod ep;
pub mod fbank;
pub mod http;
pub mod powerset;
pub mod segmentation;

#[cfg(test)]
mod bench;
#[cfg(test)]
mod hop_sweep;

use std::sync::Arc;

use anyhow::Result;

pub use clustering::{ClusterId, OnlineClusterer};
pub use embedding::EmbeddingModel;
pub use powerset::PowersetDecoder;
pub use segmentation::SegmentationModel;

#[derive(Clone, Debug)]
pub struct DiarSegment {
    pub speaker: ClusterId,
    pub t_start_ms: u64,
    pub t_end_ms: u64,
    pub confidence: f32,
}

#[derive(Clone, Debug)]
pub struct DiarConfig {
    pub chunk_seconds: f32,

    pub hop_ratio: f32,

    pub median_filter_window: usize,

    pub min_span_frames: usize,

    pub clustering_threshold: f32,

    pub max_speakers: usize,

    pub seg_batch: usize,

    pub emb_batch: usize,

    pub feature_threads: usize,
}

impl Default for DiarConfig {
    fn default() -> Self {
        Self::from_env()
    }
}

impl DiarConfig {
    pub fn from_env() -> Self {
        use super::defaults::env;
        let env_f32 = |key: &str| std::env::var(key).ok().and_then(|s| s.parse::<f32>().ok());
        let env_usize = |key: &str| {
            std::env::var(key)
                .ok()
                .and_then(|s| s.parse::<usize>().ok())
        };

        Self {
            chunk_seconds: env_f32(ENV_CHUNK_SECONDS)
                .filter(|v| *v > 0.0)
                .unwrap_or(16.0),
            hop_ratio: env_f32(ENV_HOP_RATIO)
                .map(|v| v.clamp(0.01, 1.0))
                .unwrap_or(0.1),
            median_filter_window: env_usize(env::DIAR_MEDIAN_FILTER_FRAMES).unwrap_or(11),
            min_span_frames: env_usize(env::DIAR_MIN_SPAN_FRAMES).unwrap_or(8),
            clustering_threshold: env_f32(env::DIAR_THRESHOLD)
                .map(|v| v.clamp(0.0, 1.0))
                .unwrap_or(0.55),
            max_speakers: env_usize(env::DIAR_MAX_SPEAKERS).unwrap_or(16).max(1),
            seg_batch: env_usize(ENV_SEG_BATCH).unwrap_or(8).max(1),
            emb_batch: env_usize(ENV_EMB_BATCH).unwrap_or(8).max(1),
            feature_threads: env_usize(ENV_FEATURE_THREADS)
                .unwrap_or_else(default_feature_threads)
                .max(1),
        }
    }
}

const ENV_CHUNK_SECONDS: &str = "DIAR_CHUNK_SECONDS";
const ENV_HOP_RATIO: &str = "DIAR_HOP_RATIO";
const ENV_SEG_BATCH: &str = "DIAR_SEG_BATCH";
const ENV_EMB_BATCH: &str = "DIAR_EMB_BATCH";
const ENV_FEATURE_THREADS: &str = "DIAR_FEATURE_THREADS";

fn default_feature_threads() -> usize {
    std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1)
        .clamp(1, 8)
}

#[derive(Default, Clone, Debug)]
pub struct DiarStats {
    pub chunks: usize,
    pub spans: usize,
    pub unique_spans: usize,
    pub t_chunk_us: u128,
    pub t_seg_us: u128,
    pub t_post_us: u128,
    pub t_fbank_us: u128,
    pub t_emb_us: u128,
    pub t_cluster_us: u128,
}

pub struct Diarizer {
    cfg: DiarConfig,
    seg: Arc<SegmentationModel>,
    emb: Arc<EmbeddingModel>,
    decoder: PowersetDecoder,
    clusterer: OnlineClusterer,

    session_start_ms: Option<u64>,
}

impl Diarizer {
    pub fn new(seg: Arc<SegmentationModel>, emb: Arc<EmbeddingModel>, cfg: DiarConfig) -> Self {
        let decoder =
            PowersetDecoder::new(seg.max_speakers_per_chunk(), seg.max_speakers_per_frame());
        let clusterer = OnlineClusterer::new(cfg.clustering_threshold, cfg.max_speakers);
        Self {
            cfg,
            seg,
            emb,
            decoder,
            clusterer,
            session_start_ms: None,
        }
    }

    pub fn diarize_utterance(
        &mut self,
        audio: &[f32],
        t_start_ms: u64,
    ) -> Result<Vec<DiarSegment>> {
        let mut stats = DiarStats::default();
        self.diarize_utterance_with_stats(audio, t_start_ms, &mut stats)
    }

    pub fn diarize_utterance_with_stats(
        &mut self,
        audio: &[f32],
        t_start_ms: u64,
        stats: &mut DiarStats,
    ) -> Result<Vec<DiarSegment>> {
        use std::time::Instant;

        if self.session_start_ms.is_none() {
            self.session_start_ms = Some(t_start_ms);
        }

        let t = Instant::now();
        let chunks = framing::slide_chunks(
            audio,
            self.seg.sample_rate(),
            self.cfg.chunk_seconds,
            self.cfg.hop_ratio,
        );
        stats.t_chunk_us = t.elapsed().as_micros();
        stats.chunks = chunks.len();

        let mut spans: Vec<framing::Span> = Vec::new();
        for batch in chunks.chunks(self.cfg.seg_batch) {
            let refs: Vec<&[f32]> = batch.iter().map(|c| c.samples.as_slice()).collect();
            let t = Instant::now();
            let logits_batch = self.seg.run_batch(&refs)?;
            stats.t_seg_us += t.elapsed().as_micros();
            let t = Instant::now();
            for (chunk, logits) in batch.iter().zip(logits_batch) {
                let multihot = self.decoder.to_multilabel_hard(&logits);
                let smoothed =
                    framing::median_filter_multihot(&multihot, self.cfg.median_filter_window);
                spans.extend(framing::extract_spans(
                    &smoothed,
                    self.seg.frame_rate_hz(),
                    chunk.t_offset_ms,
                    self.cfg.min_span_frames,
                ));
            }
            stats.t_post_us += t.elapsed().as_micros();
        }
        stats.spans = spans.len();

        let min_samples = self.emb.min_input_samples();
        let mut usable: Vec<&framing::Span> = Vec::with_capacity(spans.len());
        let mut slot_of: Vec<usize> = Vec::with_capacity(spans.len());
        let mut unique: Vec<(usize, usize)> = Vec::new();
        let mut seen: std::collections::HashMap<(usize, usize), usize> =
            std::collections::HashMap::new();
        for span in &spans {
            let start = span.sample_start;
            let end = span.sample_end.min(audio.len());
            if end <= start || end - start < min_samples {
                continue;
            }
            let slot = *seen.entry((start, end)).or_insert_with(|| {
                unique.push((start, end));
                unique.len() - 1
            });
            usable.push(span);
            slot_of.push(slot);
        }

        stats.unique_spans = unique.len();
        let embeddings = self.embed_unique(audio, &unique, stats)?;

        let t = Instant::now();
        let mut emitted = Vec::with_capacity(usable.len());
        for (span, slot) in usable.iter().zip(slot_of.iter()) {
            let (cluster_id, score) = self.clusterer.assign(&embeddings[*slot]);
            emitted.push(DiarSegment {
                speaker: cluster_id,
                t_start_ms: t_start_ms + span.t_start_ms,
                t_end_ms: t_start_ms + span.t_end_ms,
                confidence: score,
            });
        }

        let out = framing::coalesce_segments(emitted);
        stats.t_cluster_us = t.elapsed().as_micros();
        Ok(out)
    }

    fn embed_unique(
        &self,
        audio: &[f32],
        unique: &[(usize, usize)],
        stats: &mut DiarStats,
    ) -> Result<Vec<Vec<f32>>> {
        use std::time::Instant;
        if unique.is_empty() {
            return Ok(Vec::new());
        }

        let t = Instant::now();
        let feats = self.compute_feats_parallel(audio, unique)?;
        stats.t_fbank_us = t.elapsed().as_micros();

        let t = Instant::now();
        let out = self.emb.embed_feats_batch(&feats, self.cfg.emb_batch);
        stats.t_emb_us = t.elapsed().as_micros();
        out
    }

    fn compute_feats_parallel(
        &self,
        audio: &[f32],
        unique: &[(usize, usize)],
    ) -> Result<Vec<Vec<f32>>> {
        let emb = self.emb.as_ref();
        let threads = self.cfg.feature_threads.min(unique.len()).max(1);
        if threads <= 1 {
            return unique
                .iter()
                .map(|&(s, e)| emb.compute_feats(&audio[s..e]))
                .collect();
        }

        let next = std::sync::atomic::AtomicUsize::new(0);
        let slots: Vec<std::sync::Mutex<Option<Result<Vec<f32>>>>> = (0..unique.len())
            .map(|_| std::sync::Mutex::new(None))
            .collect();

        std::thread::scope(|scope| {
            for _ in 0..threads {
                scope.spawn(|| loop {
                    let i = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    if i >= unique.len() {
                        break;
                    }
                    let (s, e) = unique[i];
                    let r = emb.compute_feats(&audio[s..e]);
                    *slots[i].lock().expect("feature slot poisoned") = Some(r);
                });
            }
        });

        let mut out = Vec::with_capacity(unique.len());
        for slot in slots {
            match slot.into_inner().expect("feature slot poisoned") {
                Some(Ok(v)) => out.push(v),
                Some(Err(e)) => return Err(e),
                None => return Err(anyhow::anyhow!("feature extraction slot never filled")),
            }
        }
        Ok(out)
    }

    pub fn reset(&mut self) {
        self.clusterer.reset();
        self.session_start_ms = None;
    }
}

mod framing;
