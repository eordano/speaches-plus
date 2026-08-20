use anyhow::{Context, Result};
use candle_core::{DType, Device, Tensor};
use nv_weights::WeightLoader;

pub mod compressor;
pub mod decoder;
#[cfg(feature = "cuda")]
pub mod decoder_graph;
#[cfg(feature = "cuda")]
pub mod decoder_graph_batch;
#[cfg(feature = "wgpu")]
pub mod decoder_wgpu;
pub mod pipeline;
pub mod preprocess;
pub mod sam;
pub mod visual_flow;

pub use pipeline::{DecoderPrecision, DeepSeekOcr2Pipeline};
pub use preprocess::{ResolutionMode, RgbImage};

pub use decoder::{
    banned_tokens_windowed_ngram, build_prompt_tokens, DeepseekOcrDecoder,
    DeepseekOcrDecoderConfig, DeepseekOcrKvCache, GenerateOptions, BOS_TOKEN_ID, EOS_TOKEN_ID,
    IMAGE_TOKEN_ID, PROMPT_FREE_OCR, PROMPT_GROUNDING_MARKDOWN, TD_CLOSE_TOKEN_ID,
    TD_OPEN_TOKEN_ID,
};

#[cfg(feature = "cuda")]
pub use decoder_graph::{graph_supported, DsocrDecodeGraph};
#[cfg(feature = "cuda")]
pub use decoder_graph_batch::{buckets_from_env, DsocrBatchDecodeGraph};

#[cfg(feature = "wgpu")]
pub use decoder_wgpu::DeepseekOcrDecoderWgpu;

pub fn default_snapshot_dir() -> Option<std::path::PathBuf> {
    if let Ok(d) = std::env::var("NV_DSOCR_DIR") {
        let p = std::path::PathBuf::from(d);
        return p.is_dir().then_some(p);
    }
    let home = std::env::var("HOME").ok()?;
    let snaps = std::path::PathBuf::from(home)
        .join(".cache/huggingface/hub/models--deepseek-ai--DeepSeek-OCR-2/snapshots");
    let mut entries: Vec<std::path::PathBuf> = std::fs::read_dir(&snaps)
        .ok()?
        .flatten()
        .map(|e| e.path())
        .filter(|p| p.is_dir())
        .collect();
    entries.sort();
    entries.into_iter().next()
}

use compressor::{Compressor, CompressorConfig};
use preprocess::PreparedViews;
use sam::{SamConfig, SamEncoder};
use visual_flow::{FlowConfig, VisualFlow};

#[cfg(feature = "cuda")]
pub(crate) mod h2d {
    use anyhow::Result;
    use candle_core::{CudaDevice, Tensor};
    use cudarc::driver::PinnedHostSlice;
    use half::bf16;
    use std::cell::RefCell;

    thread_local! {
        static STAGE: RefCell<Vec<(usize, PinnedHostSlice<bf16>)>> =
            const { RefCell::new(Vec::new()) };
    }

    pub fn enabled() -> bool {
        std::env::var("NV_DSOCR_H2D_PINNED")
            .map(|v| v != "0")
            .unwrap_or(true)
    }

    pub(crate) fn cache_budget_bytes() -> usize {
        static B: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
        *B.get_or_init(|| {
            std::env::var("NV_DSOCR_H2D_CACHE_MB")
                .ok()
                .and_then(|v| v.parse::<usize>().ok())
                .unwrap_or(64)
                .saturating_mul(1 << 20)
        })
    }

    #[allow(dead_code)]
    pub(crate) fn resident_bytes() -> usize {
        STAGE.with(|cell| {
            cell.borrow()
                .iter()
                .map(|(n, _)| n * std::mem::size_of::<bf16>())
                .sum()
        })
    }

    fn fill_threads() -> usize {
        static N: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
        *N.get_or_init(|| {
            std::env::var("NV_DSOCR_H2D_THREADS")
                .ok()
                .and_then(|v| v.parse::<usize>().ok())
                .filter(|n| *n > 0)
                .unwrap_or_else(|| {
                    std::thread::available_parallelism()
                        .map(|n| n.get())
                        .unwrap_or(1)
                        .min(8)
                })
        })
    }

    fn narrow(dst: &mut [bf16], src: &[f32]) {
        let nt = fill_threads();
        if nt <= 1 || src.len() < 1 << 16 {
            for (d, s) in dst.iter_mut().zip(src) {
                *d = bf16::from_f32(*s);
            }
            return;
        }
        let per = src.len().div_ceil(nt);
        std::thread::scope(|sc| {
            for (d, s) in dst.chunks_mut(per).zip(src.chunks(per)) {
                sc.spawn(move || {
                    for (d, s) in d.iter_mut().zip(s) {
                        *d = bf16::from_f32(*s);
                    }
                });
            }
        });
    }

    pub fn upload_bf16<S: Into<candle_core::Shape>>(
        chunks: &[&[f32]],
        dev: &CudaDevice,
        shape: S,
    ) -> Result<Tensor> {
        let n: usize = chunks.iter().map(|c| c.len()).sum();
        let stream = dev.cuda_stream();
        STAGE.with(|cell| {
            let mut cache = cell.borrow_mut();
            match cache.iter().position(|(k, _)| *k == n) {
                Some(i) => {
                    let hit = cache.remove(i);
                    cache.insert(0, hit);
                }
                None => {
                    let ctx = stream.context().clone();
                    let buf = unsafe { ctx.alloc_pinned::<bf16>(n) }
                        .map_err(|e| anyhow::anyhow!("alloc pinned h2d staging: {e:?}"))?;
                    cache.insert(0, (n, buf));

                    let elem = std::mem::size_of::<bf16>();
                    let budget = cache_budget_bytes();
                    let mut total: usize = cache.iter().map(|(k, _)| k * elem).sum();
                    while cache.len() > 1 && total > budget {
                        let (k, _) = cache.pop().expect("len > 1");
                        total -= k * elem;
                    }
                }
            }
            let buf = &mut cache[0].1;
            {
                let mut host = buf
                    .as_mut_slice()
                    .map_err(|e| anyhow::anyhow!("pinned staging slice: {e:?}"))?;
                for c in chunks {
                    let (dst, rest) = host.split_at_mut(c.len());
                    narrow(dst, c);
                    host = rest;
                }
            }
            let slice = stream
                .clone_htod(&*buf)
                .map_err(|e| anyhow::anyhow!("pinned h2d: {e:?}"))?;
            let storage = candle_core::CudaStorage::wrap_cuda_slice(slice, dev.clone());
            Ok(Tensor::from_storage(
                candle_core::Storage::Cuda(storage),
                shape,
                candle_core::op::BackpropOp::none(),
                false,
            ))
        })
    }
}

pub(crate) fn linear(x: &Tensor, w: &Tensor, b: Option<&Tensor>) -> Result<Tensor> {
    let dims = x.dims().to_vec();
    let k = *dims.last().context("linear on 0-d tensor")?;
    let rows: usize = dims[..dims.len() - 1].iter().product();
    let x2 = x.reshape((rows, k))?;
    let mut y = x2.matmul(&w.t()?)?;
    if let Some(b) = b {
        y = y.broadcast_add(b)?;
    }
    let out = w.dim(0)?;
    let mut out_dims = dims;
    *out_dims.last_mut().unwrap() = out;
    Ok(y.reshape(out_dims)?)
}

#[derive(Clone, Debug)]
pub struct VisionConfig {
    pub sam: SamConfig,
    pub compressor: CompressorConfig,
    pub flow: FlowConfig,
    pub proj_dim: usize,
}

impl VisionConfig {
    pub fn deepseek_ocr2() -> Self {
        Self {
            sam: SamConfig::vit_b(),
            compressor: CompressorConfig::deepseek_ocr2(),
            flow: FlowConfig::deepseek_ocr2(),
            proj_dim: 1280,
        }
    }
}

pub struct DeepSeekOcr2Vision {
    cfg: VisionConfig,
    sam: SamEncoder,
    compressor: Compressor,
    flow: VisualFlow,
    proj_w: Tensor,
    proj_b: Tensor,
    view_separator: Tensor,
    device: Device,
    dtype: DType,
}

impl DeepSeekOcr2Vision {
    pub fn from_loader(
        weights: &WeightLoader,
        prefix: &str,
        cfg: VisionConfig,
        device: &Device,
        dtype: DType,
    ) -> Result<Self> {
        let sam = SamEncoder::from_loader(
            weights,
            &format!("{prefix}sam_model."),
            cfg.sam.clone(),
            dtype,
        )?;
        let compressor = Compressor::from_loader(
            weights,
            &format!("{prefix}sam_model."),
            cfg.compressor.clone(),
            dtype,
        )?;
        let flow = VisualFlow::from_loader(
            weights,
            &format!("{prefix}qwen2_model."),
            cfg.flow.clone(),
            dtype,
        )?;
        let proj_w = weights
            .get(&format!("{prefix}projector.layers.weight"), dtype)
            .context("load projector weight")?;
        let proj_b = weights
            .get(&format!("{prefix}projector.layers.bias"), dtype)
            .context("load projector bias")?;
        let view_separator = weights
            .get(&format!("{prefix}view_seperator"), dtype)
            .context("load view_seperator")?;
        anyhow::ensure!(
            proj_w.dims2()? == (cfg.proj_dim, cfg.compressor.out_dim),
            "projector shape {:?} != [{}, {}]",
            proj_w.dims(),
            cfg.proj_dim,
            cfg.compressor.out_dim
        );
        Ok(Self {
            cfg,
            sam,
            compressor,
            flow,
            proj_w,
            proj_b,
            view_separator,
            device: device.clone(),
            dtype,
        })
    }

    pub fn sam(&self) -> &SamEncoder {
        &self.sam
    }

    pub fn compressor_stage(&self) -> &Compressor {
        &self.compressor
    }

    pub fn flow(&self) -> &VisualFlow {
        &self.flow
    }

    pub fn to_pixels(&self, data: &[f32], b: usize, s: usize) -> Result<Tensor> {
        self.upload(&[data], b, s)
    }

    fn upload(&self, chunks: &[&[f32]], b: usize, s: usize) -> Result<Tensor> {
        #[cfg(feature = "cuda")]
        if self.dtype == DType::BF16 && h2d::enabled() {
            if let Device::Cuda(dev) = &self.device {
                return h2d::upload_bf16(chunks, dev, (b, 3, s, s));
            }
        }
        let mut flat: Vec<f32> = Vec::with_capacity(b * 3 * s * s);
        for c in chunks {
            flat.extend_from_slice(c);
        }
        Ok(Tensor::from_vec(flat, (b, 3, s, s), &Device::Cpu)?
            .to_device(&self.device)?
            .to_dtype(self.dtype)?)
    }

    pub fn sam_compress(&self, pixels: &Tensor) -> Result<Tensor> {
        let feat = self.sam.forward(pixels)?;
        self.compressor.forward(&feat)
    }

    pub fn flow_features(&self, feat: &Tensor) -> Result<Tensor> {
        self.flow.forward(feat)
    }

    pub fn project(&self, flow: &Tensor) -> Result<Tensor> {
        linear(flow, &self.proj_w, Some(&self.proj_b))
    }

    pub fn encode_batch(&self, pixels: &Tensor) -> Result<Tensor> {
        let feat = self.sam_compress(pixels)?;
        let flow = self.flow.forward(&feat)?;
        self.project(&flow)
    }

    pub fn encode_views(&self, global: &Tensor, tiles: Option<&Tensor>) -> Result<Tensor> {
        let global_feats = self.encode_batch(global)?;
        let (gb, gn, gd) = global_feats.dims3()?;
        let global_flat = global_feats.reshape((gb * gn, gd))?;
        let sep = self.view_separator.reshape((1, gd))?;
        let out = match tiles {
            Some(t) => {
                let tile_feats = self.encode_batch(t)?;
                let (tb, tn, td) = tile_feats.dims3()?;
                let tiles_flat = tile_feats.reshape((tb * tn, td))?;
                Tensor::cat(&[&tiles_flat, &global_flat, &sep], 0)?
            }
            None => Tensor::cat(&[&global_flat, &sep], 0)?,
        };
        Ok(out)
    }

    pub fn encode_prepared(&self, prep: &PreparedViews) -> Result<Tensor> {
        let s = prep.global_size;
        let global = self.upload(&[&prep.global], 1, s)?;
        let tiles = if prep.tiles.is_empty() {
            None
        } else {
            let refs: Vec<&[f32]> = prep.tiles.iter().map(|t| t.as_slice()).collect();
            Some(self.upload(&refs, prep.tiles.len(), prep.tile_size)?)
        };
        let feats = self.encode_views(&global, tiles.as_ref())?;
        anyhow::ensure!(
            feats.dim(0)? == prep.vision_tokens(),
            "feature count {} != expected vision tokens {}",
            feats.dim(0)?,
            prep.vision_tokens()
        );
        Ok(feats)
    }

    pub fn config(&self) -> &VisionConfig {
        &self.cfg
    }

    pub fn device(&self) -> &Device {
        &self.device
    }

    pub fn dtype(&self) -> DType {
        self.dtype
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn linear_broadcasts_over_leading_dims() {
        let dev = Device::Cpu;
        let x = Tensor::from_vec(vec![1f32, 2.0, 3.0, 4.0], (2, 1, 2), &dev).unwrap();
        let w = Tensor::from_vec(vec![1f32, 0.0, 0.0, 1.0, 1.0, 1.0], (3, 2), &dev).unwrap();
        let b = Tensor::from_vec(vec![10f32, 20.0, 30.0], 3, &dev).unwrap();
        let y = linear(&x, &w, Some(&b)).unwrap();
        assert_eq!(y.dims(), &[2, 1, 3]);
        let v: Vec<f32> = y.flatten_all().unwrap().to_vec1().unwrap();
        assert_eq!(v, vec![11.0, 22.0, 33.0, 13.0, 24.0, 37.0]);
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn h2d_staging_cache_stays_under_its_budget_and_still_uploads_correctly() {
        let Ok(dev) = Device::new_cuda(0) else {
            eprintln!("skip: no CUDA device 0");
            return;
        };
        let Device::Cuda(cd) = &dev else {
            eprintln!("skip: device is not cuda");
            return;
        };

        std::env::set_var("NV_DSOCR_H2D_CACHE_MB", "1");
        let budget = h2d::cache_budget_bytes();

        let mut peak = 0usize;
        for k in 1..=24usize {
            let n = k * 32 * 1024;
            let src: Vec<f32> = (0..n).map(|i| ((i % 251) as f32) - 125.0).collect();
            let t = h2d::upload_bf16(&[&src], cd, (n,)).expect("upload");
            let got: Vec<f32> = t.to_dtype(DType::F32).unwrap().to_vec1().expect("readback");
            assert_eq!(got.len(), n, "k={k}");
            for i in [0usize, n / 3, n - 1] {
                assert_eq!(
                    got[i], src[i],
                    "k={k} elem {i}: the staged upload came back wrong, so an \
                     evicted buffer was reused or freed while in flight"
                );
            }
            peak = peak.max(h2d::resident_bytes());
        }

        let resident = h2d::resident_bytes();

        assert!(
            resident <= budget + 24 * 32 * 1024 * 2,
            "cache held {resident} B after 24 distinct keys, budget {budget} B"
        );
        let uncapped: usize = (1..=24usize).map(|k| k * 32 * 1024 * 2).sum();
        assert!(
            resident < uncapped / 2,
            "cache held {resident} B, barely under the {uncapped} B the old \
             uncapped HashMap would have held -- eviction is not running"
        );
        eprintln!(
            "h2d staging cache: peak {peak} B, final {resident} B, \
             uncapped equivalent {uncapped} B"
        );
    }
}
