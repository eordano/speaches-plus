use std::path::Path;
use std::sync::{Arc, Mutex};

use anyhow::{anyhow, Result};
use ort::session::Session;
use ort::value::Tensor;

use super::super::vad::ort_err;

pub const SAMPLE_RATE: u32 = 16_000;
pub const FRAME_RATE_HZ: u32 = 50;
pub const SAMPLES_PER_FRAME: usize = (SAMPLE_RATE / FRAME_RATE_HZ) as usize;

#[derive(Clone, Debug)]
pub struct SegmentationLogits {
    pub frames: usize,
    pub classes: usize,

    pub data: Vec<f32>,
}

impl SegmentationLogits {
    #[inline]
    pub fn row(&self, frame: usize) -> &[f32] {
        let start = frame * self.classes;
        &self.data[start..start + self.classes]
    }
}

pub struct SegmentationModel {
    session: Arc<Mutex<Session>>,
    max_speakers_per_chunk: usize,
    max_speakers_per_frame: usize,
    provider: &'static str,
}

impl SegmentationModel {
    pub const DEFAULT_MAX_SPEAKERS_PER_CHUNK: usize = 4;

    pub const DEFAULT_MAX_SPEAKERS_PER_FRAME: usize = 4;

    pub fn load(model_path: impl AsRef<Path>) -> Result<Self> {
        Self::load_with_topology(
            model_path,
            Self::DEFAULT_MAX_SPEAKERS_PER_CHUNK,
            Self::DEFAULT_MAX_SPEAKERS_PER_FRAME,
        )
    }

    pub fn load_with_topology(
        model_path: impl AsRef<Path>,
        max_speakers_per_chunk: usize,
        max_speakers_per_frame: usize,
    ) -> Result<Self> {
        let path = model_path.as_ref();
        let loaded = super::ep::load_session(path, "diarization segmentation")?;

        let model = Self {
            session: Arc::new(Mutex::new(loaded.session)),
            max_speakers_per_chunk,
            max_speakers_per_frame,
            provider: loaded.provider,
        };
        model.warmup();
        Ok(model)
    }

    fn warmup(&self) {
        if !super::ep::warmup_enabled() {
            return;
        }
        let cfg = super::DiarConfig::from_env();
        let n = (cfg.chunk_seconds * SAMPLE_RATE as f32) as usize;
        if n == 0 {
            return;
        }
        let silence = vec![0.0f32; n];
        let refs: Vec<&[f32]> = (0..cfg.seg_batch).map(|_| silence.as_slice()).collect();
        let t = std::time::Instant::now();
        match self.run_batch(&refs) {
            Ok(_) => tracing::info!(
                batch = cfg.seg_batch,
                samples = n,
                elapsed_ms = t.elapsed().as_millis() as u64,
                "diarization segmentation warmed up"
            ),
            Err(e) => tracing::warn!(error = %e, "diarization segmentation warmup failed"),
        }
    }

    pub fn from_session(
        session: Arc<Mutex<Session>>,
        max_speakers_per_chunk: usize,
        max_speakers_per_frame: usize,
    ) -> Self {
        Self {
            session,
            max_speakers_per_chunk,
            max_speakers_per_frame,
            provider: super::ep::EP_CPU,
        }
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
    pub fn frame_rate_hz(&self) -> u32 {
        FRAME_RATE_HZ
    }

    #[inline]
    pub fn max_speakers_per_chunk(&self) -> usize {
        self.max_speakers_per_chunk
    }

    #[inline]
    pub fn max_speakers_per_frame(&self) -> usize {
        self.max_speakers_per_frame
    }

    pub fn run(&self, samples: &[f32]) -> Result<SegmentationLogits> {
        if samples.is_empty() {
            return Err(anyhow!("segmentation: empty input"));
        }

        let n = samples.len();
        let input =
            Tensor::<f32>::from_array(([1usize, 1, n], samples.to_vec().into_boxed_slice()))
                .map_err(ort_err)?;

        let (frames, classes, data) = {
            let mut sess = self
                .session
                .lock()
                .map_err(|_| anyhow!("segmentation session poisoned"))?;
            let outputs = sess
                .run(ort::inputs!["waveform" => input])
                .map_err(ort_err)?;

            let first_name = outputs
                .iter()
                .next()
                .map(|(name, _)| name.to_string())
                .ok_or_else(|| anyhow!("segmentation produced no outputs"))?;
            let (shape, data) = outputs[first_name.as_str()]
                .try_extract_tensor::<f32>()
                .map_err(ort_err)?;

            if shape.len() != 3 {
                return Err(anyhow!(
                    "segmentation: expected 3D output, got shape {:?}",
                    shape
                ));
            }

            let frames = shape[1] as usize;
            let classes = shape[2] as usize;
            if frames * classes != data.len() {
                return Err(anyhow!(
                    "segmentation: shape {:?} disagrees with {} elements",
                    shape,
                    data.len()
                ));
            }

            (frames, classes, data.to_vec())
        };

        Ok(SegmentationLogits {
            frames,
            classes,
            data,
        })
    }

    pub fn run_batch(&self, chunks: &[&[f32]]) -> Result<Vec<SegmentationLogits>> {
        if chunks.is_empty() {
            return Ok(Vec::new());
        }
        let n = chunks[0].len();
        if n == 0 {
            return Err(anyhow!("segmentation: empty input"));
        }
        if chunks.iter().any(|c| c.len() != n) {
            return Err(anyhow!("segmentation: batched chunks must be equal length"));
        }
        let b = chunks.len();

        let mut flat = Vec::with_capacity(b * n);
        for c in chunks {
            flat.extend_from_slice(c);
        }
        let input = Tensor::<f32>::from_array(([b, 1usize, n], flat.into_boxed_slice()))
            .map_err(ort_err)?;

        let (frames, classes, data) = {
            let mut sess = self
                .session
                .lock()
                .map_err(|_| anyhow!("segmentation session poisoned"))?;
            let outputs = sess
                .run(ort::inputs!["waveform" => input])
                .map_err(ort_err)?;

            let first_name = outputs
                .iter()
                .next()
                .map(|(name, _)| name.to_string())
                .ok_or_else(|| anyhow!("segmentation produced no outputs"))?;
            let (shape, data) = outputs[first_name.as_str()]
                .try_extract_tensor::<f32>()
                .map_err(ort_err)?;

            if shape.len() != 3 {
                return Err(anyhow!(
                    "segmentation: expected 3D output, got shape {:?}",
                    shape
                ));
            }
            if shape[0] as usize != b {
                return Err(anyhow!(
                    "segmentation: output batch {} != input batch {}",
                    shape[0],
                    b
                ));
            }
            let frames = shape[1] as usize;
            let classes = shape[2] as usize;
            if b * frames * classes != data.len() {
                return Err(anyhow!(
                    "segmentation: shape {:?} disagrees with {} elements",
                    shape,
                    data.len()
                ));
            }
            (frames, classes, data.to_vec())
        };

        let per = frames * classes;
        Ok((0..b)
            .map(|i| SegmentationLogits {
                frames,
                classes,
                data: data[i * per..(i + 1) * per].to_vec(),
            })
            .collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;

    #[test]
    fn loads_model() -> Result<()> {
        let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("models/diarizen-segmentation.onnx");
        if !path.exists() {
            eprintln!(
                "skip: {} missing -- run scripts/export-diarizen-onnx.py",
                path.display()
            );
            return Ok(());
        }
        let model = SegmentationModel::load(&path)?;
        let samples = vec![0.0f32; SAMPLE_RATE as usize * 16];
        let out = model.run(&samples)?;
        assert!(out.frames > 0);
        assert_eq!(
            out.classes, 16,
            "default DiariZen v2 powerset = 16 classes (4 spk, <=4/frame)"
        );

        let row0 = out.row(0);
        let argmax = row0
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap()
            .0;
        assert_eq!(argmax, 0, "silence frame should pick class 0 (silence)");
        Ok(())
    }
}
