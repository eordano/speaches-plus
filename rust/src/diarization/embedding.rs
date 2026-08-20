use std::path::Path;
use std::sync::{Arc, Mutex};

use anyhow::{anyhow, Result};
use ort::session::Session;
use ort::value::Tensor;

use super::super::vad::ort_err;

pub const SAMPLE_RATE: u32 = 16_000;
pub const EMBEDDING_DIM: usize = 256;
pub const FRAME_LENGTH_SAMPLES: usize = 400;
pub const FRAME_SHIFT_SAMPLES: usize = 160;
pub const NUM_MEL_BINS: usize = 80;
pub const MIN_INPUT_SAMPLES: usize = 16_000;

pub struct EmbeddingModel {
    session: Arc<Mutex<Session>>,
    fbank: super::fbank::FBank,
    input_name: String,
    output_name: String,
    provider: &'static str,
    runs: std::sync::atomic::AtomicU64,
    rows: std::sync::atomic::AtomicU64,
    canonical_batch: usize,
}

impl EmbeddingModel {
    pub fn load(model_path: impl AsRef<Path>) -> Result<Self> {
        Self::load_with_io_names(model_path, "feats", "embs")
    }

    pub fn load_with_io_names(
        model_path: impl AsRef<Path>,
        input_name: &str,
        output_name: &str,
    ) -> Result<Self> {
        let path = model_path.as_ref();
        let loaded = super::ep::load_session(path, "speaker embedding")?;

        let model = Self {
            session: Arc::new(Mutex::new(loaded.session)),
            fbank: super::fbank::FBank::new(
                NUM_MEL_BINS,
                FRAME_LENGTH_SAMPLES,
                FRAME_SHIFT_SAMPLES,
            ),
            input_name: input_name.to_string(),
            output_name: output_name.to_string(),
            provider: loaded.provider,
            runs: std::sync::atomic::AtomicU64::new(0),
            rows: std::sync::atomic::AtomicU64::new(0),
            canonical_batch: super::DiarConfig::from_env().emb_batch,
        };
        model.warmup();
        Ok(model)
    }

    fn warmup(&self) {
        if !super::ep::warmup_enabled() {
            return;
        }
        let window = window_frames_for(self.provider);
        if window == 0 {
            return;
        }
        let batch = self.canonical_batch;
        let feats = vec![vec![0.0f32; window * NUM_MEL_BINS]; batch];
        let t = std::time::Instant::now();
        match self.embed_feats_batch(&feats, batch) {
            Ok(_) => tracing::info!(
                batch,
                window,
                elapsed_ms = t.elapsed().as_millis() as u64,
                "speaker embedding warmed up"
            ),
            Err(e) => tracing::warn!(error = %e, "speaker embedding warmup failed"),
        }
    }

    pub fn run_counters(&self) -> (u64, u64) {
        use std::sync::atomic::Ordering::Relaxed;
        (self.runs.load(Relaxed), self.rows.load(Relaxed))
    }

    #[inline]
    pub fn provider(&self) -> &'static str {
        self.provider
    }

    #[inline]
    pub fn on_gpu(&self) -> bool {
        super::ep::is_gpu(self.provider)
    }

    #[inline]
    pub fn sample_rate(&self) -> u32 {
        SAMPLE_RATE
    }

    #[inline]
    pub fn embedding_dim(&self) -> usize {
        EMBEDDING_DIM
    }

    #[inline]
    pub fn min_input_samples(&self) -> usize {
        MIN_INPUT_SAMPLES
    }

    pub fn compute_feats(&self, samples: &[f32]) -> Result<Vec<f32>> {
        if samples.len() < FRAME_LENGTH_SAMPLES {
            return Err(anyhow!(
                "embedding: input {} samples shorter than frame length {}",
                samples.len(),
                FRAME_LENGTH_SAMPLES
            ));
        }
        self.fbank.compute(samples)
    }

    pub fn embed_feats(&self, feats: &[f32]) -> Result<Vec<f32>> {
        let mut out =
            self.embed_feats_batch(std::slice::from_ref(&feats.to_vec()), self.canonical_batch)?;
        out.pop()
            .ok_or_else(|| anyhow!("embedding produced no rows"))
    }

    pub fn embed(&self, samples: &[f32]) -> Result<Vec<f32>> {
        let feats = self.compute_feats(samples)?;
        self.embed_feats(&feats)
    }

    pub fn embed_batch(&self, spans: &[&[f32]], max_batch: usize) -> Result<Vec<Vec<f32>>> {
        if spans.is_empty() {
            return Ok(Vec::new());
        }

        let mut feats: Vec<Vec<f32>> = Vec::with_capacity(spans.len());
        for s in spans {
            feats.push(self.compute_feats(s)?);
        }
        self.embed_feats_batch(&feats, max_batch)
    }

    pub fn embed_feats_batch(&self, feats: &[Vec<f32>], max_batch: usize) -> Result<Vec<Vec<f32>>> {
        if feats.is_empty() {
            return Ok(Vec::new());
        }
        let window = window_frames_for(self.provider);
        if window > 0 {
            return self.embed_feats_windowed(feats, max_batch.max(1), window);
        }
        self.embed_feats_quantized(feats, max_batch)
    }

    fn embed_feats_windowed(
        &self,
        feats: &[Vec<f32>],
        batch: usize,
        window: usize,
    ) -> Result<Vec<Vec<f32>>> {
        let wlen = window * NUM_MEL_BINS;

        let mut owner: Vec<usize> = Vec::new();
        let mut starts: Vec<usize> = Vec::new();
        for (i, f) in feats.iter().enumerate() {
            let frames = f.len() / NUM_MEL_BINS;
            let n = frames.div_ceil(window).max(1);
            for k in 0..n {
                owner.push(i);
                starts.push(k * wlen);
            }
        }

        let mut acc: Vec<Vec<f32>> = vec![vec![0.0f32; EMBEDDING_DIM]; feats.len()];
        let mut cnt: Vec<usize> = vec![0; feats.len()];

        let total = owner.len();
        let mut flat: Vec<f32> = Vec::with_capacity(batch * wlen);
        let mut idx = 0usize;
        while idx < total {
            let end = (idx + batch).min(total);
            let real = end - idx;
            flat.clear();
            for j in idx..end {
                push_window(&mut flat, &feats[owner[j]], starts[j], wlen);
            }
            for _ in real..batch {
                push_window(&mut flat, &feats[owner[end - 1]], starts[end - 1], wlen);
            }

            let rows = self.run_feats(&flat, batch)?;
            for (j, row) in (idx..end).zip(rows.into_iter().take(real)) {
                let o = owner[j];
                for (a, v) in acc[o].iter_mut().zip(row.iter()) {
                    *a += v;
                }
                cnt[o] += 1;
            }
            idx = end;
        }

        let mut out = Vec::with_capacity(feats.len());
        for (mut v, c) in acc.into_iter().zip(cnt) {
            if c > 1 {
                let inv = 1.0 / c as f32;
                for x in v.iter_mut() {
                    *x *= inv;
                }
            }
            l2_normalize(&mut v);
            out.push(v);
        }
        Ok(out)
    }

    fn embed_feats_quantized(&self, feats: &[Vec<f32>], max_batch: usize) -> Result<Vec<Vec<f32>>> {
        let max_batch = max_batch.max(1);
        let quantum = frame_quantum();

        let lens: Vec<usize> = feats
            .iter()
            .map(|f| quantize_frames(f.len() / NUM_MEL_BINS, quantum) * NUM_MEL_BINS)
            .collect();

        let mut order: Vec<usize> = (0..feats.len()).collect();
        order.sort_by_key(|&i| lens[i]);

        let mut out: Vec<Vec<f32>> = vec![Vec::new(); feats.len()];
        let mut group_start = 0usize;
        while group_start < order.len() {
            let len = lens[order[group_start]];
            let mut group_end = group_start + 1;
            while group_end < order.len()
                && lens[order[group_end]] == len
                && group_end - group_start < max_batch
            {
                group_end += 1;
            }

            let group = &order[group_start..group_end];
            let mut flat = Vec::with_capacity(group.len() * len);
            for &i in group {
                extend_to_len(&mut flat, &feats[i], len);
            }
            let rows = self.run_feats(&flat, group.len())?;
            for (&i, row) in group.iter().zip(rows) {
                out[i] = row;
            }
            group_start = group_end;
        }

        Ok(out)
    }

    fn run_feats(&self, feats: &[f32], batch: usize) -> Result<Vec<Vec<f32>>> {
        if batch == 0 {
            return Ok(Vec::new());
        }
        if !feats.len().is_multiple_of(batch * NUM_MEL_BINS) {
            return Err(anyhow!(
                "embedding: {} features not divisible by batch {} * {} mels",
                feats.len(),
                batch,
                NUM_MEL_BINS
            ));
        }
        let frames = feats.len() / (batch * NUM_MEL_BINS);
        if frames == 0 {
            return Err(anyhow!("embedding: zero frames"));
        }
        {
            use std::sync::atomic::Ordering::Relaxed;
            self.runs.fetch_add(1, Relaxed);
            self.rows.fetch_add(batch as u64, Relaxed);
            tracing::debug!(batch, frames, "embedding run");
        }

        let input = Tensor::<f32>::from_array((
            [batch, frames, NUM_MEL_BINS],
            feats.to_vec().into_boxed_slice(),
        ))
        .map_err(ort_err)?;

        let (rows, data) = {
            let mut sess = self
                .session
                .lock()
                .map_err(|_| anyhow!("embedding session poisoned"))?;
            let outputs = sess
                .run(ort::inputs![self.input_name.as_str() => input])
                .map_err(ort_err)?;

            let has_configured = outputs
                .iter()
                .any(|(name, _)| name == self.output_name.as_str());
            let chosen_name = if has_configured {
                self.output_name.clone()
            } else {
                outputs
                    .iter()
                    .next()
                    .map(|(name, _)| name.to_string())
                    .ok_or_else(|| anyhow!("embedding produced no outputs"))?
            };
            let (shape, data) = outputs[chosen_name.as_str()]
                .try_extract_tensor::<f32>()
                .map_err(ort_err)?;
            if shape.is_empty() || (shape.last().copied().unwrap_or(0) as usize) != EMBEDDING_DIM {
                return Err(anyhow!(
                    "embedding: expected last dim {}, got shape {:?}",
                    EMBEDDING_DIM,
                    shape
                ));
            }
            let rows = data.len() / EMBEDDING_DIM;
            if rows != batch {
                return Err(anyhow!(
                    "embedding: expected {} rows, got shape {:?}",
                    batch,
                    shape
                ));
            }
            (rows, data.to_vec())
        };

        Ok((0..rows)
            .map(|i| {
                let mut v = data[i * EMBEDDING_DIM..(i + 1) * EMBEDDING_DIM].to_vec();
                l2_normalize(&mut v);
                v
            })
            .collect())
    }
}

const ENV_FRAME_QUANTUM: &str = "DIAR_EMB_FRAME_QUANTUM";
const DEFAULT_FRAME_QUANTUM: usize = 100;
const ENV_WINDOW: &str = "DIAR_EMB_WINDOW";
const DEFAULT_WINDOW_FRAMES: usize = 300;

fn frame_quantum() -> usize {
    std::env::var(ENV_FRAME_QUANTUM)
        .ok()
        .and_then(|s| s.trim().parse::<usize>().ok())
        .unwrap_or(DEFAULT_FRAME_QUANTUM)
}

fn window_frames_for(provider: &str) -> usize {
    if let Some(n) = std::env::var(ENV_WINDOW)
        .ok()
        .and_then(|s| s.trim().parse::<usize>().ok())
    {
        return n;
    }
    if super::ep::is_gpu(provider) {
        DEFAULT_WINDOW_FRAMES
    } else {
        0
    }
}

fn push_window(dst: &mut Vec<f32>, src: &[f32], start: usize, wlen: usize) {
    if src.is_empty() {
        dst.resize(dst.len() + wlen, 0.0);
        return;
    }
    let mut written = 0usize;
    let mut pos = start % src.len();
    while written < wlen {
        let take = (wlen - written).min(src.len() - pos);
        dst.extend_from_slice(&src[pos..pos + take]);
        written += take;
        pos = (pos + take) % src.len();
    }
}

#[inline]
fn quantize_frames(frames: usize, quantum: usize) -> usize {
    if quantum <= 1 || frames == 0 {
        return frames;
    }
    frames.div_ceil(quantum) * quantum
}

fn extend_to_len(dst: &mut Vec<f32>, src: &[f32], target: usize) {
    if src.is_empty() {
        dst.resize(dst.len() + target, 0.0);
        return;
    }
    if src.len() >= target {
        dst.extend_from_slice(&src[..target]);
        return;
    }
    dst.extend_from_slice(src);
    let mut written = src.len();
    while written < target {
        let take = (target - written).min(src.len());
        dst.extend_from_slice(&src[..take]);
        written += take;
    }
}

#[inline]
fn l2_normalize(v: &mut [f32]) {
    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-9);
    for x in v.iter_mut() {
        *x /= norm;
    }
}

#[inline]
pub fn cosine_sim(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}
