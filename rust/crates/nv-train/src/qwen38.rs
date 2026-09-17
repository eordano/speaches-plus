use anyhow::{bail, ensure, Context, Result};
use candle_core::{DType, Device, Tensor, Var};
use candle_nn::optim::Optimizer;
use half::bf16;
use nv_layers::linear::{Linear, LoraDeltaHook};
use nv_layers::linear_attn::LinearAttentionConfig;
use nv_layers::norm::RmsNorm;
use nv_layers::rope::{Rope, RopeConfig, RopeKind};
use nv_weights::WeightLoader;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use crate::qwen38_gdn_chunked as chunked;
use crate::{LoraConfig, LoraTrainable, RawLora, TrainingLoraHook};

pub const THE_FROZEN_PREFIX_IS_CACHED_AS_HIDDEN_STATES_WHICH_IS_ONLY_SOUND_BECAUSE_NO_TRAINABLE_PARAMETER_LIVES_BELOW_THE_WINDOW: &str =
    "Layers below window_start hold no LoRA and no Var, so their output for a fixed token \
     sequence is a constant across training steps. The trainer therefore streams them once, \
     one layer resident at a time, caches the hidden state entering window_start, and frees \
     them. Placing a LoRA below window_start would silently train against a stale cache, so \
     target selection is refused for any layer under the window.";

pub const THE_GDN_RECURRENCE_RETAINS_ONE_STATE_MATRIX_PER_TIMESTEP_SO_THE_WINDOW_NOT_THE_TRUNK_IS_WHAT_FITS: &str =
    "An exact autograd graph over the gated-delta-net scan keeps three [n_v, d_k, d_v] f32 \
     tensors per token (3.1 MB each at 48x128x128), so one GDN layer at sequence length T \
     costs about 9.4 MB * T of activation memory. That is what bounds the trainable window, \
     not the weights: 64 dequantized bf16 layers would be ~54 GB but the graph over all 48 \
     GDN layers at T=256 would be ~113 GB on a 96 GB card.";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LayerKind {
    LinearAttention,
    FullAttention,
}

impl LayerKind {
    pub fn label(&self) -> &'static str {
        match self {
            LayerKind::LinearAttention => "gdn",
            LayerKind::FullAttention => "full_attn",
        }
    }
}

#[derive(Clone, Debug)]
pub struct Q38Config {
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub intermediate_size: usize,
    pub vocab_size: usize,
    pub rope_theta: f32,
    pub rms_norm_eps: f64,
    pub partial_rotary_factor: f32,
    pub attn_output_gate: bool,
    pub layer_types: Vec<LayerKind>,
    pub linear_num_key_heads: usize,
    pub linear_num_value_heads: usize,
    pub linear_key_head_dim: usize,
    pub linear_value_head_dim: usize,
    pub linear_conv_kernel_dim: usize,
}

impl Q38Config {
    pub fn from_hf_json_str(s: &str) -> Result<Self> {
        let v: serde_json::Value = serde_json::from_str(s).context("parse qwen3.8 config json")?;
        let text = v
            .get("text_config")
            .ok_or_else(|| anyhow::anyhow!("config has no text_config"))?;
        let u = |k: &str| -> Result<usize> {
            text.get(k)
                .and_then(|x| x.as_u64())
                .map(|x| x as usize)
                .ok_or_else(|| anyhow::anyhow!("config.text_config missing/invalid {k}"))
        };
        let layer_types: Vec<LayerKind> = text
            .get("layer_types")
            .and_then(|x| x.as_array())
            .ok_or_else(|| anyhow::anyhow!("config.text_config missing layer_types"))?
            .iter()
            .map(|x| match x.as_str() {
                Some("linear_attention") => Ok(LayerKind::LinearAttention),
                Some("full_attention") => Ok(LayerKind::FullAttention),
                other => bail!("unknown layer type {other:?}"),
            })
            .collect::<Result<Vec<_>>>()?;
        let rope = text
            .get("rope_parameters")
            .cloned()
            .unwrap_or(serde_json::Value::Null);
        let rope_theta = rope
            .get("rope_theta")
            .and_then(|x| x.as_f64())
            .or_else(|| text.get("rope_theta").and_then(|x| x.as_f64()))
            .unwrap_or(10_000_000.0) as f32;
        let partial_rotary_factor = rope
            .get("partial_rotary_factor")
            .and_then(|x| x.as_f64())
            .or_else(|| text.get("partial_rotary_factor").and_then(|x| x.as_f64()))
            .unwrap_or(1.0) as f32;
        let num_hidden_layers = u("num_hidden_layers")?;
        ensure!(
            layer_types.len() == num_hidden_layers,
            "layer_types has {} entries but num_hidden_layers is {num_hidden_layers}",
            layer_types.len()
        );
        Ok(Self {
            hidden_size: u("hidden_size")?,
            num_hidden_layers,
            num_attention_heads: u("num_attention_heads")?,
            num_key_value_heads: u("num_key_value_heads")?,
            head_dim: u("head_dim")?,
            intermediate_size: u("intermediate_size")?,
            vocab_size: u("vocab_size")?,
            rope_theta,
            rms_norm_eps: text
                .get("rms_norm_eps")
                .and_then(|x| x.as_f64())
                .ok_or_else(|| anyhow::anyhow!("config.text_config missing rms_norm_eps"))?,
            partial_rotary_factor,
            attn_output_gate: text
                .get("attn_output_gate")
                .and_then(|x| x.as_bool())
                .unwrap_or(false),
            layer_types,
            linear_num_key_heads: u("linear_num_key_heads")?,
            linear_num_value_heads: u("linear_num_value_heads")?,
            linear_key_head_dim: u("linear_key_head_dim")?,
            linear_value_head_dim: u("linear_value_head_dim")?,
            linear_conv_kernel_dim: u("linear_conv_kernel_dim")?,
        })
    }

    pub fn from_hf_json_file(p: &Path) -> Result<Self> {
        let raw = std::fs::read_to_string(p).with_context(|| format!("read {}", p.display()))?;
        Self::from_hf_json_str(&raw)
    }

    pub fn rotary_dim(&self) -> usize {
        (self.head_dim as f32 * self.partial_rotary_factor).round() as usize
    }

    pub fn linear_attn_config(&self) -> LinearAttentionConfig {
        LinearAttentionConfig {
            hidden_size: self.hidden_size,
            linear_num_key_heads: self.linear_num_key_heads,
            linear_num_value_heads: self.linear_num_value_heads,
            linear_key_head_dim: self.linear_key_head_dim,
            linear_value_head_dim: self.linear_value_head_dim,
            linear_conv_kernel_dim: self.linear_conv_kernel_dim,
            mamba_ssm_dtype: DType::F32,
            rms_eps: self.rms_norm_eps,
        }
    }
}

pub const THE_TRUNK_RMSNORM_GAMMAS_ARE_STORED_AS_GAMMA_MINUS_ONE_BUT_THE_GDN_INNER_NORM_IS_NOT:
    &str = "input_layernorm, post_attention_layernorm, self_attn.q_norm, self_attn.k_norm and the \
     final model.language_model.norm all ship centred on zero and the serving loader adds one \
     (qwen3_5_moe::load_rmsnorm_plus_one). linear_attn.norm.weight ships centred on its true \
     value and is loaded raw. Reading the first group raw multiplies the residual stream by \
     roughly zero at every layer, which does not crash and does not produce garbage: it produces \
     near-uniform logits, a cross entropy just above ln(vocab), and a loss curve that still falls \
     monotonically while the model underneath is destroyed.";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ModuleStorage {
    Bf16,
    Fp8RowScale,
    Nvfp4Packed,
}

impl ModuleStorage {
    pub fn label(&self) -> &'static str {
        match self {
            ModuleStorage::Bf16 => "bf16",
            ModuleStorage::Fp8RowScale => "fp8_rowscale",
            ModuleStorage::Nvfp4Packed => "nvfp4_packed",
        }
    }
}

fn par_row_blocks(
    rows: usize,
    cols: usize,
    per_block: &(dyn Fn(usize, usize) -> Result<Vec<f32>> + Sync),
) -> Result<Vec<f32>> {
    ensure!(
        rows > 0 && cols > 0,
        "host dequant degenerate weight shape [{rows}, {cols}]: chunks_mut(0) would panic, \
         a zero-dim rank-2 tensor is a malformed checkpoint"
    );
    let threads = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1)
        .min(rows.max(1));
    let block = rows.div_ceil(threads);
    let mut out = vec![0f32; rows * cols];
    let failure: Mutex<Option<String>> = Mutex::new(None);
    {
        let blocks: Vec<Mutex<&mut [f32]>> = out.chunks_mut(block * cols).map(Mutex::new).collect();
        let nblocks = blocks.len();
        let next = AtomicUsize::new(0);
        std::thread::scope(|s| {
            for _ in 0..threads {
                s.spawn(|| loop {
                    let bi = next.fetch_add(1, Ordering::Relaxed);
                    if bi >= nblocks {
                        break;
                    }
                    let start = bi * block;
                    let count = block.min(rows - start);
                    match per_block(start, count) {
                        Ok(vals) => {
                            let mut g = blocks[bi].lock().unwrap();
                            g.copy_from_slice(&vals);
                        }
                        Err(e) => {
                            let mut f = failure.lock().unwrap();
                            if f.is_none() {
                                *f = Some(format!("{e:#}"));
                            }
                        }
                    }
                });
            }
        });
    }
    if let Some(msg) = failure.into_inner().unwrap() {
        bail!("host dequant row block failed: {msg}");
    }
    Ok(out)
}

pub struct HostDequantWeights {
    loader: WeightLoader,
    device: Device,
    #[cfg(feature = "cuda")]
    serving_w4a4_runner: Option<Arc<Mutex<nv_quant::nvfp4::Nvfp4GemmRunner>>>,
    #[cfg(feature = "cuda")]
    resident_fp8_runner: Option<Arc<Mutex<nv_quant::fp8::Fp8GemmRunner>>>,
}

impl HostDequantWeights {
    pub fn open_dir(dir: &Path, device: &Device) -> Result<Self> {
        let file = dir.join("model.safetensors");
        let loader = if file.is_file() {
            WeightLoader::open_file(&file, device)?
        } else {
            WeightLoader::open_dir(dir, device)?
        };
        Ok(Self {
            loader,
            device: device.clone(),
            #[cfg(feature = "cuda")]
            serving_w4a4_runner: None,
            #[cfg(feature = "cuda")]
            resident_fp8_runner: None,
        })
    }

    pub fn open_single_file(path: &Path, device: &Device) -> Result<Self> {
        Ok(Self {
            loader: WeightLoader::open_file(path, device)?,
            device: device.clone(),
            #[cfg(feature = "cuda")]
            serving_w4a4_runner: None,
            #[cfg(feature = "cuda")]
            resident_fp8_runner: None,
        })
    }

    #[cfg(feature = "cuda")]
    pub fn with_serving_w4a4_arm(
        mut self,
        runner: Arc<Mutex<nv_quant::nvfp4::Nvfp4GemmRunner>>,
    ) -> Self {
        self.serving_w4a4_runner = Some(runner);
        self
    }

    #[cfg(feature = "cuda")]
    pub fn with_quant_resident_arms(
        mut self,
        nvfp4: Arc<Mutex<nv_quant::nvfp4::Nvfp4GemmRunner>>,
        fp8: Arc<Mutex<nv_quant::fp8::Fp8GemmRunner>>,
    ) -> Self {
        self.serving_w4a4_runner = Some(nvfp4);
        self.resident_fp8_runner = Some(fp8);
        self
    }

    pub fn has_any_quantized_linear(&self) -> bool {
        self.loader.names().iter().any(|n| {
            n.ends_with(".weight_packed")
                || (n.ends_with(".weight")
                    && self.loader.st_dtype_of(n) == Some(nv_weights::StDtype::F8_E4M3)
                    && self.loader.has(&format!("{n}_scale")))
        })
    }

    pub fn loader(&self) -> &WeightLoader {
        &self.loader
    }

    pub fn has(&self, name: &str) -> bool {
        self.loader.has(name)
    }

    pub fn classify(&self, module: &str) -> Result<(ModuleStorage, usize, usize)> {
        let packed = format!("{module}.weight_packed");
        if let Some(shape) = self.loader.shape_of(&packed) {
            ensure!(
                shape.len() == 2,
                "{packed}: expected rank-2 packed weight, got {shape:?}"
            );
            return Ok((ModuleStorage::Nvfp4Packed, shape[0], shape[1] * 2));
        }
        let wname = format!("{module}.weight");
        let shape = self
            .loader
            .shape_of(&wname)
            .ok_or_else(|| anyhow::anyhow!("{module}: neither {packed} nor {wname} present"))?;
        ensure!(
            shape.len() == 2,
            "{wname}: expected rank-2 weight, got {shape:?}"
        );
        let dt = self
            .loader
            .st_dtype_of(&wname)
            .ok_or_else(|| anyhow::anyhow!("{wname}: no dtype"))?;
        if dt == nv_weights::StDtype::F8_E4M3 && self.loader.has(&format!("{module}.weight_scale"))
        {
            return Ok((ModuleStorage::Fp8RowScale, shape[0], shape[1]));
        }
        if dt == nv_weights::StDtype::BF16 || dt == nv_weights::StDtype::F32 {
            return Ok((ModuleStorage::Bf16, shape[0], shape[1]));
        }
        bail!("{wname}: storage dtype {dt:?} has no host dequant for training")
    }

    fn scalar_f32(&self, name: &str) -> Result<f32> {
        let v: Vec<f32> = self
            .loader
            .get(name, DType::F32)?
            .flatten_all()?
            .to_vec1()?;
        Ok(*v.first().unwrap_or(&1.0))
    }

    pub fn dense_matrix(&self, module: &str, dtype: DType) -> Result<Tensor> {
        let (kind, rows, cols) = self.classify(module)?;
        let host = match kind {
            ModuleStorage::Bf16 => {
                let t = self.loader.get(&format!("{module}.weight"), dtype)?;
                return Ok(t.to_device(&self.device)?);
            }
            ModuleStorage::Fp8RowScale => {
                let bytes = self.loader.raw_bytes(&format!("{module}.weight"))?;
                let scale_vals: Vec<f32> = self
                    .loader
                    .get(&format!("{module}.weight_scale"), DType::F32)?
                    .flatten_all()?
                    .to_vec1()?;
                let scale_dims = self
                    .loader
                    .shape_of(&format!("{module}.weight_scale"))
                    .unwrap_or_else(|| vec![scale_vals.len()]);
                let scales = nv_weights::fp8_row_scales_from(&scale_dims, &scale_vals, rows)?;
                par_row_blocks(rows, cols, &|start, count| {
                    nv_quant::fp8::dequantize_e4m3_per_row(
                        &bytes[start * cols..(start + count) * cols],
                        count,
                        cols,
                        &scales[start..start + count],
                    )
                })?
            }
            ModuleStorage::Nvfp4Packed => {
                let packed = self.loader.raw_bytes(&format!("{module}.weight_packed"))?;
                let scale_bytes = self.loader.raw_bytes(&format!("{module}.weight_scale"))?;
                let global = self.scalar_f32(&format!("{module}.weight_global_scale"))?;
                let weight_mult = if global == 0.0 || !global.is_finite() {
                    1.0
                } else {
                    1.0 / global
                };
                let bpr = cols / nv_quant::nvfp4::BLOCK_SIZE;
                ensure!(
                    packed.len() == rows * cols / 2 && scale_bytes.len() == rows * bpr,
                    "{module}: packed/scale byte lengths do not match [{rows}, {cols}]"
                );
                par_row_blocks(rows, cols, &|start, count| {
                    Ok(nv_quant::nvfp4::dequantize_packed_linear(
                        &packed[start * cols / 2..(start + count) * cols / 2],
                        &scale_bytes[start * bpr..(start + count) * bpr],
                        count,
                        cols,
                        weight_mult,
                    ))
                })?
            }
        };
        let cpu = Tensor::from_vec(host, (rows, cols), &Device::Cpu)?;
        Ok(cpu.to_dtype(dtype)?.to_device(&self.device)?)
    }

    pub fn dense_linear(&self, module: &str, dtype: DType) -> Result<Linear> {
        #[cfg(feature = "cuda")]
        if let Some(runner) = &self.serving_w4a4_runner {
            if let Ok((ModuleStorage::Nvfp4Packed, rows, cols)) = self.classify(module) {
                return nv_layers::moe::nvfp4_linear_from_disk_pub(
                    &self.loader,
                    module,
                    rows,
                    cols,
                    runner.clone(),
                    &self.device,
                )
                .with_context(|| format!("w4a4 resident ingest {module}"));
            }
        }
        #[cfg(feature = "cuda")]
        if let Some(runner) = &self.resident_fp8_runner {
            if let Ok((ModuleStorage::Fp8RowScale, rows, cols)) = self.classify(module) {
                return nv_layers::linear::fp8_e4m3_rowscale_checkpoint_resident_linear(
                    &self.loader,
                    module,
                    rows,
                    cols,
                    &self.device,
                    runner.clone(),
                )
                .with_context(|| format!("fp8 resident ingest {module}"));
            }
        }
        let w = self
            .dense_matrix(module, dtype)
            .with_context(|| format!("host dequant {module}"))?;
        Linear::new_no_pretranspose(w, None)
    }

    pub fn trunk_rmsnorm_gamma(&self, name: &str, dtype: DType) -> Result<Tensor> {
        let _ =
            THE_TRUNK_RMSNORM_GAMMAS_ARE_STORED_AS_GAMMA_MINUS_ONE_BUT_THE_GDN_INNER_NORM_IS_NOT;
        let raw = self.plain(name, DType::F32)?;
        Ok(raw.affine(1.0, 1.0)?.to_dtype(dtype)?)
    }

    pub fn plain(&self, name: &str, dtype: DType) -> Result<Tensor> {
        Ok(self
            .loader
            .get(name, dtype)
            .with_context(|| format!("load {name}"))?
            .to_device(&self.device)?)
    }

    pub fn embed_rows(&self, name: &str, ids: &[u32], hidden: usize) -> Result<Tensor> {
        let bytes = self.loader.raw_bytes(name)?;
        let dt = self
            .loader
            .st_dtype_of(name)
            .ok_or_else(|| anyhow::anyhow!("{name}: no dtype"))?;
        ensure!(
            dt == nv_weights::StDtype::BF16,
            "{name}: embedding row gather expects BF16 storage, got {dt:?}"
        );
        let mut out: Vec<bf16> = Vec::with_capacity(ids.len() * hidden);
        for id in ids {
            let off = (*id as usize) * hidden * 2;
            ensure!(
                off + hidden * 2 <= bytes.len(),
                "{name}: token id {id} is past the embedding table"
            );
            for c in 0..hidden {
                out.push(bf16::from_le_bytes([
                    bytes[off + c * 2],
                    bytes[off + c * 2 + 1],
                ]));
            }
        }
        let cpu = Tensor::from_vec(out, (ids.len(), hidden), &Device::Cpu)?;
        Ok(cpu.to_device(&self.device)?)
    }
}

fn softplus(x: &Tensor) -> Result<Tensor> {
    let zero = Tensor::zeros(x.shape(), x.dtype(), x.device())?;
    let max_part = x.maximum(&zero)?;
    let neg_abs = x.abs()?.affine(-1.0, 0.0)?;
    let log_part = neg_abs.exp()?.affine(1.0, 1.0)?.log()?;
    Ok(max_part.add(&log_part)?)
}

fn l2_normalize_last(x: &Tensor) -> Result<Tensor> {
    let in_dt = x.dtype();
    let xf = x.to_dtype(DType::F32)?;
    let sum_sq = xf.sqr()?.sum_keepdim(candle_core::D::Minus1)?;
    let eps_t = Tensor::new(1e-6f32, x.device())?;
    let denom = sum_sq.broadcast_add(&eps_t)?.sqrt()?;
    Ok(xf.broadcast_div(&denom)?.to_dtype(in_dt)?)
}

fn rmsnorm_last(x: &Tensor, weight: &Tensor, eps: f64) -> Result<Tensor> {
    let in_dt = x.dtype();
    let xf = x.to_dtype(DType::F32)?;
    let var = xf.sqr()?.mean_keepdim(candle_core::D::Minus1)?;
    let eps_t = Tensor::new(eps as f32, x.device())?;
    let denom = var.broadcast_add(&eps_t)?.sqrt()?;
    let normed = xf.broadcast_div(&denom)?;
    let w = weight.to_dtype(DType::F32)?;
    Ok(normed.broadcast_mul(&w)?.to_dtype(in_dt)?)
}

pub struct TrainGdn {
    cfg: LinearAttentionConfig,
    pub in_proj_qkv: Linear,
    pub in_proj_z: Linear,
    pub in_proj_a: Linear,
    pub in_proj_b: Linear,
    conv1d_weight: Tensor,
    a_log: Tensor,
    dt_bias: Tensor,
    norm_weight: Tensor,
    pub out_proj: Linear,
}

impl TrainGdn {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        cfg: LinearAttentionConfig,
        in_proj_qkv: Linear,
        in_proj_z: Linear,
        in_proj_a: Linear,
        in_proj_b: Linear,
        conv1d_weight: Tensor,
        a_log: Tensor,
        dt_bias: Tensor,
        norm_weight: Tensor,
        out_proj: Linear,
    ) -> Self {
        Self {
            cfg,
            in_proj_qkv,
            in_proj_z,
            in_proj_a,
            in_proj_b,
            conv1d_weight,
            a_log,
            dt_bias,
            norm_weight,
            out_proj,
        }
    }

    pub fn from_weights(
        src: &HostDequantWeights,
        prefix: &str,
        cfg: LinearAttentionConfig,
        dtype: DType,
    ) -> Result<Self> {
        let n_v = cfg.linear_num_value_heads;
        Ok(Self::new(
            cfg,
            src.dense_linear(&format!("{prefix}.in_proj_qkv"), dtype)?,
            src.dense_linear(&format!("{prefix}.in_proj_z"), dtype)?,
            src.dense_linear(&format!("{prefix}.in_proj_a"), dtype)?,
            src.dense_linear(&format!("{prefix}.in_proj_b"), dtype)?,
            src.plain(&format!("{prefix}.conv1d.weight"), dtype)?
                .reshape((cfg.conv_dim(), 1, cfg.linear_conv_kernel_dim))?,
            src.plain(&format!("{prefix}.A_log"), dtype)?.reshape(n_v)?,
            src.plain(&format!("{prefix}.dt_bias"), dtype)?
                .reshape(n_v)?,
            src.plain(&format!("{prefix}.norm.weight"), dtype)?
                .reshape(cfg.linear_value_head_dim)?,
            src.dense_linear(&format!("{prefix}.out_proj"), dtype)?,
        ))
    }

    fn causal_depthwise_conv_silu(&self, x_btc: &Tensor, b: usize, t: usize) -> Result<Tensor> {
        let c = self.cfg.conv_dim();
        let k = self.cfg.linear_conv_kernel_dim;
        let pad = Tensor::zeros((b, k - 1, c), x_btc.dtype(), x_btc.device())?;
        let padded = Tensor::cat(&[&pad, x_btc], 1)?.contiguous()?;
        let mut acc: Option<Tensor> = None;
        for j in 0..k {
            let seg = padded.narrow(1, j, t)?;
            let wj = self.conv1d_weight.narrow(2, j, 1)?.reshape(c)?;
            let term = seg.broadcast_mul(&wj.to_dtype(seg.dtype())?)?;
            acc = Some(match acc {
                Some(a) => a.add(&term)?,
                None => term,
            });
        }
        let acc = acc.ok_or_else(|| anyhow::anyhow!("conv kernel width 0"))?;
        Ok(candle_nn::ops::silu(&acc)?)
    }

    pub fn scan_dims(&self) -> (usize, usize, usize) {
        (
            self.cfg.linear_num_value_heads,
            self.cfg.linear_key_head_dim,
            self.cfg.linear_value_head_dim,
        )
    }

    pub fn head(&self, x: &Tensor) -> Result<GdnParts> {
        let dims = x.dims().to_vec();
        ensure!(
            dims.len() == 3,
            "TrainGdn::head expects [B, T, H], got {dims:?}"
        );
        let (b, t, h) = (dims[0], dims[1], dims[2]);
        ensure!(
            h == self.cfg.hidden_size,
            "TrainGdn: hidden {h} != cfg {}",
            self.cfg.hidden_size
        );
        let acc = self.cfg.mamba_ssm_dtype;
        let n_k = self.cfg.linear_num_key_heads;
        let n_v = self.cfg.linear_num_value_heads;
        let d_k = self.cfg.linear_key_head_dim;
        let d_v = self.cfg.linear_value_head_dim;
        let key_dim = self.cfg.key_dim();
        let value_dim = self.cfg.value_dim();

        let qkv = self.in_proj_qkv.forward(x)?;
        let z = self.in_proj_z.forward(x)?;
        let a = self.in_proj_a.forward(x)?;
        let b_proj = self.in_proj_b.forward(x)?;

        let head_dims = crate::qwen38_gdn_head::GdnHeadDims {
            b,
            t,
            conv_dim: self.cfg.conv_dim(),
            key_dim,
            n_v,
            v_per_k: self.cfg.v_per_k(),
            head_dim: d_k,
            conv_k: self.cfg.linear_conv_kernel_dim,
        };
        #[cfg(feature = "cuda")]
        if d_v == d_k
            && acc == DType::F32
            && crate::qwen38_gdn_head::fused_head_kernel_serves(&qkv, &head_dims)
        {
            let op = crate::qwen38_gdn_head::GdnHeadPackedOp::new(&self.conv1d_weight, head_dims)?;
            let packed = qkv.contiguous()?.apply_op1(op)?;
            let q_scaled = packed.narrow(3, 0, d_k)?;
            let k_grp = packed.narrow(3, d_k, d_k)?;
            let v_f = packed.narrow(3, 2 * d_k, d_v)?;
            let beta = candle_nn::ops::sigmoid(&b_proj.to_dtype(acc)?)?;
            let a_plus = a
                .to_dtype(acc)?
                .broadcast_add(&self.dt_bias.to_dtype(acc)?)?;
            let neg_exp_alog = self.a_log.to_dtype(acc)?.exp()?.affine(-1.0, 0.0)?;
            let g_exp = softplus(&a_plus)?.broadcast_mul(&neg_exp_alog)?.exp()?;
            return Ok(GdnParts {
                q_scaled,
                k_grp,
                v_f,
                g_exp,
                beta,
                z,
                b,
                t,
            });
        }
        let _ = &head_dims;

        let mixed = self.causal_depthwise_conv_silu(&qkv, b, t)?;

        let q = mixed.narrow(2, 0, key_dim)?.reshape((b, t, n_k, d_k))?;
        let k = mixed
            .narrow(2, key_dim, key_dim)?
            .reshape((b, t, n_k, d_k))?;
        let v = mixed
            .narrow(2, 2 * key_dim, value_dim)?
            .reshape((b, t, n_v, d_v))?;

        let v_per_k = self.cfg.v_per_k();
        let expand = |g: Tensor| -> Result<Tensor> {
            if v_per_k == 1 {
                Ok(g)
            } else {
                Ok(g.unsqueeze(3)?
                    .expand((b, t, n_k, v_per_k, d_k))?
                    .contiguous()?
                    .reshape((b, t, n_v, d_k))?)
            }
        };
        let q_grp = l2_normalize_last(&expand(q)?.to_dtype(acc)?)?;
        let k_grp = l2_normalize_last(&expand(k)?.to_dtype(acc)?)?;
        let q_scaled = q_grp.affine(1.0 / (d_k as f64).sqrt(), 0.0)?;
        let v_f = v.to_dtype(acc)?;

        let beta = candle_nn::ops::sigmoid(&b_proj.to_dtype(acc)?)?;
        let a_plus = a
            .to_dtype(acc)?
            .broadcast_add(&self.dt_bias.to_dtype(acc)?)?;
        let neg_exp_alog = self.a_log.to_dtype(acc)?.exp()?.affine(-1.0, 0.0)?;
        let g_exp = softplus(&a_plus)?.broadcast_mul(&neg_exp_alog)?.exp()?;

        Ok(GdnParts {
            q_scaled,
            k_grp,
            v_f,
            g_exp,
            beta,
            z,
            b,
            t,
        })
    }

    pub fn tail(
        &self,
        core: &Tensor,
        z: &Tensor,
        in_dtype: DType,
        b: usize,
        t: usize,
    ) -> Result<Tensor> {
        let n_v = self.cfg.linear_num_value_heads;
        let d_v = self.cfg.linear_value_head_dim;
        let value_dim = self.cfg.value_dim();
        let core_bf = core.to_dtype(in_dtype)?;
        let z_re = z.reshape((b, t, n_v, d_v))?;
        let normed = rmsnorm_last(&core_bf, &self.norm_weight, self.cfg.rms_eps)?;
        let gate = candle_nn::ops::silu(&z_re.to_dtype(DType::F32)?)?.to_dtype(in_dtype)?;
        let gated = normed.mul(&gate)?;
        let flat = gated.reshape((b, t, value_dim))?;
        self.out_proj.forward(&flat)
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let in_dtype = x.dtype();
        let p = self.head(x)?;
        let (n_v, d_k, d_v) = self.scan_dims();
        let core = gdn_scan(
            &p.q_scaled,
            &p.k_grp,
            &p.v_f,
            &p.g_exp,
            &p.beta,
            p.b,
            p.t,
            n_v,
            d_k,
            d_v,
        )?;
        self.tail(&core, &p.z, in_dtype, p.b, p.t)
    }
}

pub struct GdnParts {
    pub q_scaled: Tensor,
    pub k_grp: Tensor,
    pub v_f: Tensor,
    pub g_exp: Tensor,
    pub beta: Tensor,
    pub z: Tensor,
    pub b: usize,
    pub t: usize,
}

#[allow(clippy::too_many_arguments)]
pub fn gdn_scan(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    g_exp: &Tensor,
    beta: &Tensor,
    b: usize,
    t: usize,
    n_v: usize,
    d_k: usize,
    d_v: usize,
) -> Result<Tensor> {
    let acc = q.dtype();
    let bn = b * n_v;
    let q3 = chunked::fold3(q, bn, t, d_k)?;
    let k3 = chunked::fold3(k, bn, t, d_k)?;
    let v3 = chunked::fold3(v, bn, t, d_v)?;
    let g2 = chunked::fold2(g_exp, bn, t)?;
    let be2 = chunked::fold2(beta, bn, t)?;

    #[cfg(feature = "cuda")]
    if chunked::fused_scan_kernel_serves(q, d_k, d_v) {
        let fwd = chunked::scan_chunked_fwd(q, k, v, g_exp, beta, b, t, n_v, d_k, d_v, t)?;
        if std::env::var("NV_Q38_GDN_SCAN_CHECK_PRINTS_THE_KERNEL_VS_CANDLE_DIVERGENCE_PER_CALL").as_deref() == Ok("1") {
            std::env::set_var(chunked::NV_Q38_GDN_SCAN_KERNEL_DEFAULT_ON_0_RESTORES_THE_PER_TOKEN_CANDLE_SCAN_WHOSE_TEN_LAUNCHES_PER_TOKEN_MADE_THE_WINDOW_BACKWARD_2_1_S_OF_A_3_0_S_STEP, "0");
            let reference = gdn_scan(q, k, v, g_exp, beta, b, t, n_v, d_k, d_v);
            std::env::remove_var(chunked::NV_Q38_GDN_SCAN_KERNEL_DEFAULT_ON_0_RESTORES_THE_PER_TOKEN_CANDLE_SCAN_WHOSE_TEN_LAUNCHES_PER_TOKEN_MADE_THE_WINDOW_BACKWARD_2_1_S_OF_A_3_0_S_STEP);
            let reference = reference?;
            let diff = fwd.out.sub(&reference)?.abs()?.flatten_all()?.max(0)?.to_scalar::<f32>()?;
            let scale = reference.abs()?.flatten_all()?.max(0)?.to_scalar::<f32>()?;
            let gmin = g_exp.flatten_all()?.min(0)?.to_scalar::<f32>()?;
            let gmax = g_exp.flatten_all()?.max(0)?.to_scalar::<f32>()?;
            let bmax = beta.flatten_all()?.max(0)?.to_scalar::<f32>()?;
            eprintln!(
                "[gdn-scan-check] b={b} t={t} n_v={n_v} dtype={:?} max|diff|={diff:e} max|ref|={scale:e} rel={:e} g_exp=[{gmin:e},{gmax:e}] beta_max={bmax:e}",
                q.dtype(),
                diff / scale.max(1e-30)
            );
        }
        return Ok(fwd.out);
    }

    let mut state = Tensor::zeros((bn, d_k, d_v), acc, q.device())?;
    let mut outs: Vec<Tensor> = Vec::with_capacity(t);
    for ti in 0..t {
        let q_t = q3.narrow(1, ti, 1)?.contiguous()?;
        let k_t = k3.narrow(1, ti, 1)?.contiguous()?;
        let v_t = v3.narrow(1, ti, 1)?.contiguous()?;
        let g_t = g2.narrow(1, ti, 1)?.reshape((bn, 1, 1))?;
        let be_t = be2.narrow(1, ti, 1)?.reshape((bn, 1, 1))?;
        let (s, o) = chunked::scan_step(&state, &q_t, &k_t, &v_t, &g_t, &be_t)?;
        state = s;
        outs.push(o);
    }
    let out = Tensor::cat(&outs, 1)?;
    chunked::unfold3(&out, b, n_v, t, d_v)
}

pub struct TrainAttn {
    pub q_proj: Linear,
    pub k_proj: Linear,
    pub v_proj: Linear,
    pub o_proj: Linear,
    q_norm: RmsNorm,
    k_norm: RmsNorm,
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    attn_output_gate: bool,
    rotary_dim: usize,
}

impl TrainAttn {
    pub fn from_weights(
        src: &HostDequantWeights,
        prefix: &str,
        cfg: &Q38Config,
        dtype: DType,
    ) -> Result<Self> {
        Ok(Self {
            q_proj: src.dense_linear(&format!("{prefix}.q_proj"), dtype)?,
            k_proj: src.dense_linear(&format!("{prefix}.k_proj"), dtype)?,
            v_proj: src.dense_linear(&format!("{prefix}.v_proj"), dtype)?,
            o_proj: src.dense_linear(&format!("{prefix}.o_proj"), dtype)?,
            q_norm: RmsNorm::new(
                src.trunk_rmsnorm_gamma(&format!("{prefix}.q_norm.weight"), dtype)?,
                cfg.rms_norm_eps,
            ),
            k_norm: RmsNorm::new(
                src.trunk_rmsnorm_gamma(&format!("{prefix}.k_norm.weight"), dtype)?,
                cfg.rms_norm_eps,
            ),
            n_heads: cfg.num_attention_heads,
            n_kv_heads: cfg.num_key_value_heads,
            head_dim: cfg.head_dim,
            attn_output_gate: cfg.attn_output_gate,
            rotary_dim: cfg.rotary_dim(),
        })
    }

    pub fn forward(&self, x: &Tensor, rope: &Rope, positions: &Tensor) -> Result<Tensor> {
        let dims = x.dims().to_vec();
        ensure!(
            dims.len() == 3,
            "TrainAttn::forward expects [B, T, H], got {dims:?}"
        );
        let (b, t) = (dims[0], dims[1]);
        let in_dtype = x.dtype();
        let hd = self.head_dim;

        let q_raw = self.q_proj.forward(x)?;
        let k_raw = self.k_proj.forward(x)?;
        let v_raw = self.v_proj.forward(x)?;

        let (q_value, q_gate) = if self.attn_output_gate {
            let q_view = q_raw.reshape((b, t, self.n_heads, hd * 2))?;
            (
                q_view.narrow(3, 0, hd)?.contiguous()?,
                Some(q_view.narrow(3, hd, hd)?.contiguous()?),
            )
        } else {
            (q_raw.reshape((b, t, self.n_heads, hd))?, None)
        };
        let k = k_raw.reshape((b, t, self.n_kv_heads, hd))?;
        let v = v_raw.reshape((b, t, self.n_kv_heads, hd))?;

        let q_normed = self.q_norm.forward_candle(&q_value)?;
        let k_normed = self.k_norm.forward_candle(&k)?;

        #[cfg(feature = "cuda")]
        let fused_rope = if nv_layers::rope::train_rope_op_serves(&q_normed) && q_normed.dtype() == k_normed.dtype() {
            let rd = self.rotary_dim.min(hd);
            Some((
                rope.apply_train_partial_scaled(&q_normed, positions, rd, hd, 1.0)?.to_dtype(in_dtype)?,
                rope.apply_train_partial_scaled(&k_normed, positions, rd, hd, 1.0)?.to_dtype(in_dtype)?,
            ))
        } else {
            None
        };
        #[cfg(not(feature = "cuda"))]
        let fused_rope: Option<(Tensor, Tensor)> = None;
        let (q_final, k_final) = if let Some(pair) = fused_rope {
            pair
        } else if self.rotary_dim >= hd {
            let (qr, kr) = rope.apply_candle(
                &q_normed.to_dtype(DType::F32)?,
                &k_normed.to_dtype(DType::F32)?,
                positions,
            )?;
            (qr.to_dtype(in_dtype)?, kr.to_dtype(in_dtype)?)
        } else {
            let q_rot = q_normed.narrow(3, 0, self.rotary_dim)?.contiguous()?;
            let q_pass = q_normed
                .narrow(3, self.rotary_dim, hd - self.rotary_dim)?
                .contiguous()?;
            let k_rot = k_normed.narrow(3, 0, self.rotary_dim)?.contiguous()?;
            let k_pass = k_normed
                .narrow(3, self.rotary_dim, hd - self.rotary_dim)?
                .contiguous()?;
            let (qr, kr) = rope.apply_candle(
                &q_rot.to_dtype(DType::F32)?,
                &k_rot.to_dtype(DType::F32)?,
                positions,
            )?;
            (
                Tensor::cat(&[&qr.to_dtype(in_dtype)?, &q_pass], 3)?.contiguous()?,
                Tensor::cat(&[&kr.to_dtype(in_dtype)?, &k_pass], 3)?.contiguous()?,
            )
        };

        let attn_out = nv_layers::attn::sdpa(
            &q_final,
            &k_final,
            &v,
            &nv_layers::attn::AttnConfig {
                num_heads: self.n_heads,
                num_kv_heads: self.n_kv_heads,
                head_dim: hd,
                softmax_scale: 1.0 / (hd as f32).sqrt(),
                causal: true,
            },
        )?;

        let attn_out = attn_out.reshape((b, t, self.n_heads * hd))?;
        let gated = match q_gate {
            Some(qg) => {
                let gate_flat = qg.reshape((b, t, self.n_heads * hd))?;
                let sig = candle_nn::ops::sigmoid(&gate_flat.to_dtype(DType::F32)?)?
                    .to_dtype(in_dtype)?;
                attn_out.mul(&sig)?
            }
            None => attn_out,
        };
        self.o_proj.forward(&gated)
    }
}

pub struct TrainMlp {
    pub gate_proj: Linear,
    pub up_proj: Linear,
    pub down_proj: Linear,
}

impl TrainMlp {
    pub fn from_weights(src: &HostDequantWeights, prefix: &str, dtype: DType) -> Result<Self> {
        Ok(Self {
            gate_proj: src.dense_linear(&format!("{prefix}.gate_proj"), dtype)?,
            up_proj: src.dense_linear(&format!("{prefix}.up_proj"), dtype)?,
            down_proj: src.dense_linear(&format!("{prefix}.down_proj"), dtype)?,
        })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let gate = self.gate_proj.forward(x)?;
        let up = self.up_proj.forward(x)?;
        let activated = candle_nn::ops::silu(&gate)?.mul(&up)?;
        self.down_proj.forward(&activated)
    }
}

pub enum TrainMixer {
    Full(TrainAttn),
    Linear(TrainGdn),
}

pub struct TrainLayer {
    pre_norm: RmsNorm,
    post_norm: RmsNorm,
    pub mixer: TrainMixer,
    pub mlp: TrainMlp,
    pub kind: LayerKind,
    pub index: usize,
}

impl TrainLayer {
    pub fn from_weights(
        src: &HostDequantWeights,
        cfg: &Q38Config,
        index: usize,
        dtype: DType,
    ) -> Result<Self> {
        let prefix = layer_prefix(index);
        let kind = cfg.layer_types[index];
        let mixer = match kind {
            LayerKind::FullAttention => TrainMixer::Full(TrainAttn::from_weights(
                src,
                &format!("{prefix}.self_attn"),
                cfg,
                dtype,
            )?),
            LayerKind::LinearAttention => TrainMixer::Linear(TrainGdn::from_weights(
                src,
                &format!("{prefix}.linear_attn"),
                cfg.linear_attn_config(),
                dtype,
            )?),
        };
        Ok(Self {
            pre_norm: RmsNorm::new(
                src.trunk_rmsnorm_gamma(&format!("{prefix}.input_layernorm.weight"), dtype)?,
                cfg.rms_norm_eps,
            ),
            post_norm: RmsNorm::new(
                src.trunk_rmsnorm_gamma(
                    &format!("{prefix}.post_attention_layernorm.weight"),
                    dtype,
                )?,
                cfg.rms_norm_eps,
            ),
            mixer,
            mlp: TrainMlp::from_weights(src, &format!("{prefix}.mlp"), dtype)?,
            kind,
            index,
        })
    }

    pub fn forward(&self, x: &Tensor, rope: &Rope, positions: &Tensor) -> Result<Tensor> {
        let normed = self.pre_norm.forward_candle(x)?;
        let mixed = match &self.mixer {
            TrainMixer::Full(a) => a.forward(&normed, rope, positions)?,
            TrainMixer::Linear(g) => g.forward(&normed)?,
        };
        let after_mix = x.add(&mixed)?;
        let normed_post = self.post_norm.forward_candle(&after_mix)?;
        let ffn = self.mlp.forward(&normed_post)?;
        Ok(after_mix.add(&ffn)?)
    }

    pub fn projections(&self) -> Vec<(&'static str, &Linear)> {
        let mut out: Vec<(&'static str, &Linear)> = match &self.mixer {
            TrainMixer::Full(a) => vec![
                ("q_proj", &a.q_proj),
                ("k_proj", &a.k_proj),
                ("v_proj", &a.v_proj),
                ("o_proj", &a.o_proj),
            ],
            TrainMixer::Linear(g) => vec![
                ("in_proj_qkv", &g.in_proj_qkv),
                ("in_proj_z", &g.in_proj_z),
                ("out_proj", &g.out_proj),
            ],
        };
        out.push(("gate_proj", &self.mlp.gate_proj));
        out.push(("up_proj", &self.mlp.up_proj));
        out.push(("down_proj", &self.mlp.down_proj));
        out
    }

    pub fn module_name(&self, leaf: &str) -> String {
        let prefix = layer_prefix(self.index);
        match leaf {
            "q_proj" | "k_proj" | "v_proj" | "o_proj" => format!("{prefix}.self_attn.{leaf}"),
            "in_proj_qkv" | "in_proj_z" | "out_proj" => format!("{prefix}.linear_attn.{leaf}"),
            _ => format!("{prefix}.mlp.{leaf}"),
        }
    }

    pub const GDN_CHUNKED_BACKWARD_SEVERS_THE_SCAN_AT_ITS_FIVE_INPUTS_AND_RE_SEEDS_THEM_FROM_A_SECOND_SURROGATE: &'static str =
        "the layer graph is built once with the scan replaced by a detached chunked forward \
         behind a Var seam at its output; the first surrogate backward covers the tail \
         (norm/gate/out_proj), the mlp, the z projection and the residuals, and yields d(core); \
         the reverse chunk sweep turns d(core) into grads at the five scan inputs; the second \
         surrogate re-seeds those grads into the head subgraph (conv+silu, l2norm, \
         softplus/sigmoid gating, in_proj_qkv/a/b), so no state matrix outlives its chunk";

    pub fn gdn_grads_chunked(
        &self,
        h_in: &Tensor,
        grad_next: &Tensor,
        vars: &[Var],
        chunk: usize,
    ) -> Result<(Vec<Option<Tensor>>, Tensor)> {
        let _ = Self::GDN_CHUNKED_BACKWARD_SEVERS_THE_SCAN_AT_ITS_FIVE_INPUTS_AND_RE_SEEDS_THEM_FROM_A_SECOND_SURROGATE;
        let TrainMixer::Linear(gdn) = &self.mixer else {
            bail!("gdn_grads_chunked called on a full-attention layer")
        };
        let in_dtype = h_in.dtype();
        let dev = h_in.device().clone();
        let x_var = Var::from_tensor(&h_in.detach())?;
        let (p, normed) = crate::train_prof::phase(&dev, "gdn_bwd_head_forward_graph", || {
            let normed = self.pre_norm.forward_candle(x_var.as_tensor())?;
            let p = gdn.head(&normed)?;
            Ok((p, normed))
        })?;
        let _ = normed;
        let (n_v, d_k, d_v) = gdn.scan_dims();
        let fwd = crate::train_prof::phase(&dev, "gdn_bwd_scan_forward_checkpointed", || {
            chunked::scan_chunked_fwd(
                &p.q_scaled,
                &p.k_grp,
                &p.v_f,
                &p.g_exp,
                &p.beta,
                p.b,
                p.t,
                n_v,
                d_k,
                d_v,
                chunk,
            )
        })?;
        let core_var = Var::from_tensor(&fwd.out)?;
        let (gs_tail, d_core) = crate::train_prof::phase(&dev, "gdn_bwd_tail_mlp_forward_and_surrogate_backward", || {
            let mixed = gdn.tail(core_var.as_tensor(), &p.z, in_dtype, p.b, p.t)?;
            let after_mix = x_var.as_tensor().add(&mixed)?;
            let normed_post = self.post_norm.forward_candle(&after_mix)?;
            let ffn = self.mlp.forward(&normed_post)?;
            let out = after_mix.add(&ffn)?;
            let surrogate_tail = out
                .to_dtype(DType::F32)?
                .mul(&grad_next.detach().to_dtype(DType::F32)?)?
                .sum_all()?;
            let gs_tail = surrogate_tail.backward()?;
            let d_core = gs_tail
                .get(core_var.as_tensor())
                .cloned()
                .ok_or_else(|| anyhow::anyhow!("no grad reached the scan output seam"))?;
            Ok((gs_tail, d_core))
        })?;
        let mut acc: Vec<Option<Tensor>> = vars.iter().map(|v| gs_tail.get(v).cloned()).collect();

        let (dq, dk, dv, dg, dbe) = crate::train_prof::phase(&dev, "gdn_bwd_scan_input_grads", || {
            chunked::scan_chunked_input_grads(
                &p.q_scaled,
                &p.k_grp,
                &p.v_f,
                &p.g_exp,
                &p.beta,
                &d_core,
                &fwd.checkpoints,
                p.b,
                p.t,
                n_v,
                d_k,
                d_v,
                chunk,
            )
        })?;

        let gs_head = crate::train_prof::phase(&dev, "gdn_bwd_head_surrogate_backward", || {
            let surrogate_head = p
                .q_scaled
                .mul(&dq)?
                .sum_all()?
                .add(&p.k_grp.mul(&dk)?.sum_all()?)?
                .add(&p.v_f.mul(&dv)?.sum_all()?)?
                .add(&p.g_exp.mul(&dg)?.sum_all()?)?
                .add(&p.beta.mul(&dbe)?.sum_all()?)?;
            surrogate_head.backward().map_err(Into::into)
        })?;
        for (vi, v) in vars.iter().enumerate() {
            if let Some(g) = gs_head.get(v) {
                let g = g.clone();
                acc[vi] = Some(match acc[vi].take() {
                    Some(a) => a.add(&g.to_dtype(a.dtype())?)?,
                    None => g,
                });
            }
        }
        let gx = match (
            gs_tail.get(x_var.as_tensor()),
            gs_head.get(x_var.as_tensor()),
        ) {
            (Some(a), Some(b)) => a.add(b)?,
            (Some(a), None) => a.clone(),
            (None, Some(b)) => b.clone(),
            (None, None) => bail!("no grad reached the layer input from either surrogate"),
        };
        Ok((acc, gx))
    }
}

pub fn layer_prefix(index: usize) -> String {
    format!("model.language_model.layers.{index}")
}

pub const EMBED_TOKENS: &str = "model.language_model.embed_tokens.weight";
pub const FINAL_NORM: &str = "model.language_model.norm.weight";
pub const LM_HEAD: &str = "lm_head";

pub fn build_rope(cfg: &Q38Config, seq_len: usize, device: &Device) -> Result<Rope> {
    Rope::new(
        RopeConfig {
            head_dim: cfg.rotary_dim().max(2),
            max_seq_len: seq_len.max(2),
            base: cfg.rope_theta,
            kind: RopeKind::Standard,
        },
        device,
    )
}

pub fn flat_positions(b: usize, t: usize, device: &Device) -> Result<Tensor> {
    let mut ids: Vec<u32> = Vec::with_capacity(b * t);
    for _ in 0..b {
        for p in 0..t {
            ids.push(p as u32);
        }
    }
    Ok(Tensor::from_vec(ids, b * t, device)?)
}

#[derive(Clone, Debug)]
pub struct Q38TrainArgs {
    pub base: PathBuf,
    pub out: PathBuf,
    pub corpus: PathBuf,
    pub corpus_bytes: usize,
    pub seq_len: usize,
    pub examples: usize,
    pub grad_layers: usize,
    pub rank: usize,
    pub alpha: f64,
    pub lr: f64,
    pub steps: usize,
    pub targets: Vec<String>,
    pub seed: u64,
    pub dtype: DType,
}

impl Q38TrainArgs {
    pub fn default_targets() -> Vec<String> {
        [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "in_proj_qkv",
            "in_proj_z",
            "out_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]
        .iter()
        .map(|s| s.to_string())
        .collect()
    }
}

#[derive(Debug)]
pub struct Q38TrainSummary {
    pub losses: Vec<f32>,
    pub step_seconds: Vec<f64>,
    pub prefix_seconds: f64,
    pub window_build_seconds: f64,
    pub window_start: usize,
    pub window_kinds: Vec<&'static str>,
    pub modules: Vec<String>,
    pub trainable_vars: usize,
    pub examples: usize,
    pub seq_len: usize,
    pub tokens_per_step: usize,
    pub min_module_delta_amax: f32,
    pub max_module_delta_amax: f32,
    pub all_grads_finite: bool,
    pub resident_base: bool,
    pub device: String,
    pub adapter_dir: PathBuf,
}

impl Q38TrainSummary {
    pub fn losses_are_monotonically_decreasing(&self) -> bool {
        self.losses.windows(2).all(|w| w[1] < w[0])
    }

    pub fn total_drop(&self) -> f32 {
        match (self.losses.first(), self.losses.last()) {
            (Some(a), Some(b)) => a - b,
            _ => 0.0,
        }
    }

    pub fn tokens_per_second(&self) -> f32 {
        let secs: f64 = self.step_seconds.iter().sum();
        if secs <= 0.0 {
            return 0.0;
        }
        (self.tokens_per_step * self.step_seconds.len()) as f32 / secs as f32
    }
}

pub fn byte_corpus(path: &Path, take_bytes: usize, vocab: usize) -> Result<Vec<u32>> {
    let raw = std::fs::read(path).with_context(|| format!("read corpus {}", path.display()))?;
    ensure!(!raw.is_empty(), "corpus {} is empty", path.display());
    let n = take_bytes.min(raw.len());
    Ok(raw[..n]
        .iter()
        .map(|b| (*b as u32) % vocab as u32)
        .collect())
}

pub fn tokenized_corpus(
    tokenizer_json: &Path,
    corpus: &Path,
    take_bytes: usize,
    vocab: usize,
) -> Result<Vec<u32>> {
    if !tokenizer_json.is_file() {
        return byte_corpus(corpus, take_bytes, vocab);
    }
    let tok = nv_tokenizer::load_tokenizer(tokenizer_json)?;
    let raw = std::fs::read_to_string(corpus)
        .with_context(|| format!("read corpus {}", corpus.display()))?;
    let mut end = take_bytes.min(raw.len());
    while end > 0 && !raw.is_char_boundary(end) {
        end -= 1;
    }
    let slice = &raw[..end];
    let enc = tok
        .encode(slice, false)
        .map_err(|e| anyhow::anyhow!("tokenize corpus: {e}"))?;
    let ids: Vec<u32> = enc
        .get_ids()
        .iter()
        .copied()
        .filter(|i| (*i as usize) < vocab)
        .collect();
    ensure!(!ids.is_empty(), "corpus tokenized to nothing");
    Ok(ids)
}

fn grad_store_from(
    vars: &[Var],
    grads: &[Option<Tensor>],
) -> Result<candle_core::backprop::GradStore> {
    let mut gs = vars[0].as_tensor().sum_all()?.backward()?;
    for (v, g) in vars.iter().zip(grads.iter()) {
        if let Some(g) = g {
            gs.insert(v.as_tensor(), g.clone());
        }
    }
    Ok(gs)
}

pub fn run(args: &Q38TrainArgs, device: &Device) -> Result<Q38TrainSummary> {
    Ok(run_inner(args, None, device)?.0)
}

pub fn run_with_mtp_cotrain(
    args: &Q38TrainArgs,
    co: &Q38MtpCoTrainArgs,
    device: &Device,
) -> Result<(Q38TrainSummary, Q38MtpCoTrainSummary)> {
    let (summary, co_summary) = run_inner(args, Some(co), device)?;
    Ok((
        summary,
        co_summary.expect("co-training was requested, so the speculator summary exists"),
    ))
}

fn run_inner(
    args: &Q38TrainArgs,
    co: Option<&Q38MtpCoTrainArgs>,
    device: &Device,
) -> Result<(Q38TrainSummary, Option<Q38MtpCoTrainSummary>)> {
    let _ = THE_FROZEN_PREFIX_IS_CACHED_AS_HIDDEN_STATES_WHICH_IS_ONLY_SOUND_BECAUSE_NO_TRAINABLE_PARAMETER_LIVES_BELOW_THE_WINDOW;
    let _ = THE_GDN_RECURRENCE_RETAINS_ONE_STATE_MATRIX_PER_TIMESTEP_SO_THE_WINDOW_NOT_THE_TRUNK_IS_WHAT_FITS;

    let cfg = Q38Config::from_hf_json_file(&args.base.join("config.json"))?;
    ensure!(
        args.grad_layers >= 1 && args.grad_layers <= cfg.num_hidden_layers,
        "grad_layers {} must be in 1..={}",
        args.grad_layers,
        cfg.num_hidden_layers
    );
    let window_start = cfg.num_hidden_layers - args.grad_layers;
    let (src, resident_base) = open_train_weights(&args.base, device)?;
    let dtype = args.dtype;

    let ids = tokenized_corpus(
        &args.base.join("tokenizer.json"),
        &args.corpus,
        args.corpus_bytes,
        cfg.vocab_size,
    )?;
    let need = args.examples * (args.seq_len + 1);
    ensure!(
        ids.len() >= need,
        "corpus yielded {} tokens but {} examples of length {}+1 need {need}",
        ids.len(),
        args.examples,
        args.seq_len
    );
    let mut inputs: Vec<Vec<u32>> = Vec::with_capacity(args.examples);
    let mut targets_ids: Vec<Vec<u32>> = Vec::with_capacity(args.examples);
    for e in 0..args.examples {
        let base = e * (args.seq_len + 1);
        inputs.push(ids[base..base + args.seq_len].to_vec());
        targets_ids.push(ids[base + 1..base + args.seq_len + 1].to_vec());
    }

    let rope = build_rope(&cfg, args.seq_len, device)?;
    let positions = flat_positions(1, args.seq_len, device)?;

    let t_prefix = std::time::Instant::now();
    let mut hidden: Vec<Tensor> = Vec::with_capacity(args.examples);
    for ex in &inputs {
        let rows = src.embed_rows(EMBED_TOKENS, ex, cfg.hidden_size)?;
        hidden.push(
            rows.reshape((1, args.seq_len, cfg.hidden_size))?
                .to_dtype(dtype)?,
        );
    }
    for li in 0..window_start {
        let layer = TrainLayer::from_weights(&src, &cfg, li, dtype)
            .with_context(|| format!("build frozen prefix layer {li}"))?;
        for h in hidden.iter_mut() {
            *h = layer.forward(h, &rope, &positions)?;
        }
        drop(layer);
    }
    let prefix_seconds = t_prefix.elapsed().as_secs_f64();

    let t_window = std::time::Instant::now();
    let mut window: Vec<TrainLayer> = Vec::with_capacity(args.grad_layers);
    for li in window_start..cfg.num_hidden_layers {
        window.push(
            TrainLayer::from_weights(&src, &cfg, li, dtype)
                .with_context(|| format!("build trainable window layer {li}"))?,
        );
    }
    let final_norm = RmsNorm::new(
        src.trunk_rmsnorm_gamma(FINAL_NORM, dtype)?,
        cfg.rms_norm_eps,
    );
    let lm_head = src.dense_linear(LM_HEAD, dtype)?;
    let window_build_seconds = t_window.elapsed().as_secs_f64();

    let lcfg = LoraConfig {
        r: args.rank,
        alpha: args.alpha,
        dropout: 0.0,
    };
    let (loras, vars) = attach_window_loras(&window, &args.targets, lcfg, args.seed, device)?;

    let cotrain = match co {
        None => None,
        Some(c) => {
            ensure!(
                window_start + window.len() == cfg.num_hidden_layers,
                "MTP co-training requires the trainable window to end at the final trunk layer: \
                 {}",
                MTP_COTRAIN_ACCEPTS_ANY_GRAD_LAYERS_BECAUSE_THE_WINDOW_ENDS_AT_THE_FINAL_TRUNK_LAYER_WHOSE_LIVE_OUTPUT_THE_HEAD_CONSUMES_DETACHED
            );
            let ids_refs: Vec<&[u32]> = inputs.iter().map(|v| v.as_slice()).collect();
            Some(MtpCoTrainEngine::build(
                c, &cfg, &args.base, &src, &ids_refs, dtype, device, args.seed,
            )?)
        }
    };

    let forward_window = |h: &Tensor| -> Result<Tensor> {
        let mut x = h.clone();
        for layer in &window {
            x = layer.forward(&x, &rope, &positions)?;
        }
        Ok(x)
    };
    let loss_from_window_out = |x: &Tensor, tgt: &[u32]| -> Result<Tensor> {
        let x = final_norm.forward_candle(x)?;
        let logits = lm_head.forward(&x)?;
        let logits2 = logits
            .reshape((args.seq_len, cfg.vocab_size))?
            .to_dtype(DType::F32)?;
        let t = Tensor::from_vec(tgt.to_vec(), args.seq_len, device)?;
        Ok(candle_nn::loss::cross_entropy(&logits2, &t)?)
    };

    let (losses, step_seconds, all_grads_finite, head_losses, all_head_grads_finite) = match (
        &cotrain, co,
    ) {
        (Some(ct), Some(c)) => {
            let ones = vec![1.0f32; args.seq_len];
            let o = co_train_steps(
                    &vars,
                    &loras,
                    &ct.vars,
                    &ct.loras,
                    args.steps,
                    args.lr,
                    B_LR_RATIO_DEFAULT_OF_ONE_IS_EXACTLY_THE_SINGLE_OPTIMIZER_PATH_AND_ABOVE_ONE_IS_LORA_PLUS,
                    args.lr * c.speculator_lr_ratio,
                    args.examples,
                    |i| {
                        let x = forward_window(&hidden[i])?;
                        let loss = loss_from_window_out(&x, &targets_ids[i])?;
                        let xd = x.detach();
                        let trunk = loss_and_grads_from_one_backward(loss, &vars)?;
                        let tgt_t =
                            Tensor::from_vec(targets_ids[i].clone(), args.seq_len, device)?;
                        let head = ct.head_loss_and_grads(
                            i,
                            &xd,
                            &rope,
                            &positions,
                            &lm_head,
                            &tgt_t,
                            &ones,
                            cfg.vocab_size,
                        )?;
                        Ok((trunk, head))
                    },
                )?;
            (
                o.losses,
                o.step_seconds,
                o.all_grads_finite,
                o.head_losses,
                o.all_head_grads_finite,
            )
        }
        _ => {
            let (l, s, f) = train_steps(
                    &vars,
                    &loras,
                    args.steps,
                    args.lr,
                    B_LR_RATIO_DEFAULT_OF_ONE_IS_EXACTLY_THE_SINGLE_OPTIMIZER_PATH_AND_ABOVE_ONE_IS_LORA_PLUS,
                    args.examples,
                    |i| {
                        let x = forward_window(&hidden[i])?;
                        loss_and_grads_from_one_backward(
                            loss_from_window_out(&x, &targets_ids[i])?,
                            &vars,
                        )
                    },
                )?;
            (l, s, f, Vec::new(), true)
        }
    };

    let (entries, min_amax, max_amax) = collect_raw_entries(&loras)?;
    let (speculator, head_modules) = match &cotrain {
        Some(ct) => {
            let (artifact, modules) = ct.artifact()?;
            (Some(artifact), modules)
        }
        None => (None, Vec::new()),
    };
    crate::save_peft_raw_with_speculator(
        &args.out,
        &entries,
        args.rank,
        args.alpha,
        0.0,
        &args.base.display().to_string(),
        speculator.as_ref(),
    )?;
    let co_summary = cotrain.as_ref().map(|ct| Q38MtpCoTrainSummary {
        head_losses,
        head_modules,
        head_trainable_vars: ct.vars.len(),
        all_head_grads_finite,
        speculator_kind: QWEN38_MTP_SPECULATOR_KIND.to_string(),
        speculator_dir: args.out.join(crate::SPECULATOR_SUBDIR),
    });

    let summary = Q38TrainSummary {
        losses,
        step_seconds,
        prefix_seconds,
        window_build_seconds,
        window_start,
        window_kinds: window.iter().map(|l| l.kind.label()).collect(),
        modules: entries.iter().map(|e| e.module.clone()).collect(),
        trainable_vars: vars.len(),
        examples: args.examples,
        seq_len: args.seq_len,
        tokens_per_step: args.examples * args.seq_len,
        min_module_delta_amax: if min_amax.is_finite() { min_amax } else { 0.0 },
        max_module_delta_amax: max_amax,
        all_grads_finite,
        resident_base,
        device: if device.is_cuda() {
            "cuda".to_string()
        } else {
            "cpu".to_string()
        },
        adapter_dir: args.out.clone(),
    };
    Ok((summary, co_summary))
}

pub fn attach_window_loras(
    window: &[TrainLayer],
    targets: &[String],
    lcfg: LoraConfig,
    seed0: u64,
    device: &Device,
) -> Result<(Vec<(String, Arc<LoraTrainable>)>, Vec<Var>)> {
    let mut loras: Vec<(String, Arc<LoraTrainable>)> = Vec::new();
    let mut vars: Vec<Var> = Vec::new();
    let mut seed = seed0;
    for layer in window {
        for (leaf, lin) in layer.projections() {
            if !targets.iter().any(|t| t == leaf) {
                continue;
            }
            let lora = Arc::new(LoraTrainable::from_features(
                lin.out_features(),
                lin.in_features(),
                lcfg,
                seed,
                device,
            )?);
            seed = seed.wrapping_add(1);
            vars.extend(lora.trainable_vars());
            let hook: Arc<dyn LoraDeltaHook> = Arc::new(TrainingLoraHook::new(lora.clone(), true));
            lin.attach_lora(hook)?;
            loras.push((layer.module_name(leaf), lora));
        }
    }
    ensure!(
        !loras.is_empty(),
        "no window module matched targets {:?}",
        targets
    );
    Ok((loras, vars))
}

pub(crate) fn loss_and_grads_from_one_backward(
    loss: Tensor,
    vars: &[Var],
) -> Result<(f32, Vec<Option<Tensor>>)> {
    let lv = loss.to_scalar::<f32>()?;
    let grads = loss.backward()?;
    Ok((lv, vars.iter().map(|v| grads.get(v).cloned()).collect()))
}

#[allow(clippy::too_many_arguments)]
fn window_loss_and_grads_checkpointed(
    window: &[TrainLayer],
    final_norm: &RmsNorm,
    lm_head: &Linear,
    rope: &Rope,
    positions: &Tensor,
    h0: &Tensor,
    tgt: &Tensor,
    mask: &[f32],
    vocab: usize,
    vars: &[Var],
) -> Result<(f32, Vec<Option<Tensor>>, Tensor)> {
    let t = h0.dims()[1];
    ensure!(
        mask.len() == t,
        "loss mask has {} weights for {t} target positions",
        mask.len()
    );
    let unmasked = mask.iter().sum::<f32>().max(1.0);
    let weights: Vec<f32> = mask.iter().map(|w| w / unmasked).collect();
    window_loss_and_grads_checkpointed_rows(
        window, final_norm, lm_head, rope, positions, h0, tgt, &weights, vocab, vars,
    )
}

#[allow(clippy::too_many_arguments)]
fn window_loss_and_grads_checkpointed_rows(
    window: &[TrainLayer],
    final_norm: &RmsNorm,
    lm_head: &Linear,
    rope: &Rope,
    positions: &Tensor,
    h0: &Tensor,
    tgt: &Tensor,
    weights: &[f32],
    vocab: usize,
    vars: &[Var],
) -> Result<(f32, Vec<Option<Tensor>>, Tensor)> {
    let dev = h0.device().clone();
    let rows = h0.dims()[0];
    let mut boundaries: Vec<Tensor> = Vec::with_capacity(window.len() + 1);
    boundaries.push(h0.detach());
    crate::train_prof::phase(&dev, "forward_window_chain", || {
        for layer in window {
            let out = layer
                .forward(boundaries.last().unwrap(), rope, positions)?
                .detach();
            boundaries.push(out);
        }
        Ok(())
    })?;

    let t = h0.dims()[1];
    let hv = Var::from_tensor(boundaries.last().unwrap())?;
    let (lv, gs) = crate::train_prof::phase(&dev, "head_loss_and_backward", || {
        let x = final_norm.forward_candle(hv.as_tensor())?;
        let logits = lm_head.forward(&x)?.reshape((rows * t, vocab))?;
        #[cfg(feature = "cuda")]
        let loss = if crate::qwen38_ce::fused_ce_serves(&logits) {
            crate::qwen38_ce::ce_rows_fused(&logits, tgt, weights)?
        } else {
            crate::ce_weighted_over_positions(&logits.to_dtype(DType::F32)?, tgt, weights)?
        };
        #[cfg(not(feature = "cuda"))]
        let loss = crate::ce_weighted_over_positions(&logits.to_dtype(DType::F32)?, tgt, weights)?;
        let lv = loss.to_scalar::<f32>()?;
        let gs = loss.backward()?;
        Ok((lv, gs))
    })?;
    let mut acc: Vec<Option<Tensor>> = vars.iter().map(|v| gs.get(v).cloned()).collect();
    let mut grad_next = gs
        .get(hv.as_tensor())
        .cloned()
        .ok_or_else(|| anyhow::anyhow!("no grad for the checkpointed head input"))?;

    let chunk = chunked::chunk_tokens_from_env();
    for li in (0..window.len()).rev() {
        if chunk > 0 && matches!(window[li].mixer, TrainMixer::Linear(_)) {
            let (layer_acc, gx) =
                crate::train_prof::phase(&dev, "gdn_layer_chunked_backward", || {
                    window[li].gdn_grads_chunked(&boundaries[li], &grad_next, vars, chunk)
                })?;
            for (vi, g) in layer_acc.into_iter().enumerate() {
                if let Some(g) = g {
                    acc[vi] = Some(match acc[vi].take() {
                        Some(a) => a.add(&g.to_dtype(a.dtype())?)?,
                        None => g,
                    });
                }
            }
            grad_next = gx;
            continue;
        }
        let inp = Var::from_tensor(&boundaries[li])?;
        let phase_name = match window[li].mixer {
            TrainMixer::Linear(_) => "gdn_layer_recompute_and_backward",
            _ => "attention_layer_recompute_and_backward",
        };
        let layer_gs = crate::train_prof::phase(&dev, phase_name, || {
            let out = window[li].forward(inp.as_tensor(), rope, positions)?;
            let surrogate = out
                .to_dtype(DType::F32)?
                .mul(&grad_next.detach().to_dtype(DType::F32)?)?
                .sum_all()?;
            surrogate.backward().map_err(Into::into)
        })?;
        for (vi, v) in vars.iter().enumerate() {
            if let Some(g) = layer_gs.get(v) {
                let g = g.clone();
                acc[vi] = Some(match acc[vi].take() {
                    Some(a) => a.add(&g.to_dtype(a.dtype())?)?,
                    None => g,
                });
            }
        }
        grad_next = layer_gs
            .get(inp.as_tensor())
            .cloned()
            .with_context(|| format!("no grad for checkpointed window layer {li} input"))?;
    }
    let window_out_already_detached = boundaries
        .last()
        .expect("boundaries holds h0 plus one entry per window layer")
        .clone();
    Ok((lv, acc, window_out_already_detached))
}

pub(crate) enum WindowOptimizer {
    Single(candle_nn::optim::AdamW),
    SplitBecauseLoraPlusRunsBFactorsHotter {
        a_vars: Vec<Var>,
        a_opt: candle_nn::optim::AdamW,
        b_vars: Vec<Var>,
        b_opt: candle_nn::optim::AdamW,
    },
}

fn adamw(vars: Vec<Var>, lr: f64) -> Result<candle_nn::optim::AdamW> {
    Ok(candle_nn::optim::AdamW::new(
        vars,
        candle_nn::optim::ParamsAdamW {
            lr,
            weight_decay: 0.0,
            ..Default::default()
        },
    )?)
}

impl WindowOptimizer {
    pub(crate) fn new(vars: &[Var], lr: f64, b_lr_ratio: f64) -> Result<Self> {
        if b_lr_ratio == B_LR_RATIO_DEFAULT_OF_ONE_IS_EXACTLY_THE_SINGLE_OPTIMIZER_PATH_AND_ABOVE_ONE_IS_LORA_PLUS {
            return Ok(Self::Single(adamw(vars.to_vec(), lr)?));
        }
        let a_vars: Vec<Var> = vars.iter().step_by(2).cloned().collect();
        let b_vars: Vec<Var> = vars.iter().skip(1).step_by(2).cloned().collect();
        Ok(Self::SplitBecauseLoraPlusRunsBFactorsHotter {
            a_opt: adamw(a_vars.clone(), lr)?,
            a_vars,
            b_opt: adamw(b_vars.clone(), lr * b_lr_ratio)?,
            b_vars,
        })
    }

    pub(crate) fn step(&mut self, vars: &[Var], acc: &[Option<Tensor>]) -> Result<()> {
        match self {
            Self::Single(opt) => Ok(opt.step(&grad_store_from(vars, acc)?)?),
            Self::SplitBecauseLoraPlusRunsBFactorsHotter {
                a_vars,
                a_opt,
                b_vars,
                b_opt,
            } => {
                let every_second = |off: usize| -> Vec<Option<Tensor>> {
                    acc.iter().skip(off).step_by(2).cloned().collect()
                };
                a_opt.step(&grad_store_from(a_vars, &every_second(0))?)?;
                b_opt.step(&grad_store_from(b_vars, &every_second(1))?)?;
                Ok(())
            }
        }
    }
}

pub(crate) fn fold_example_grads(
    acc: &mut [Option<Tensor>],
    grads: Vec<Option<Tensor>>,
) -> Result<()> {
    for (vi, g) in grads.into_iter().enumerate() {
        if let Some(g) = g {
            let g = g.to_dtype(DType::F32)?;
            acc[vi] = Some(match acc[vi].take() {
                Some(a) => a.add(&g)?,
                None => g,
            });
        }
    }
    Ok(())
}

pub(crate) fn average_grads_bailing_on_an_untouched_factor(
    acc: &mut [Option<Tensor>],
    loras: &[(String, Arc<LoraTrainable>)],
    inv: f64,
    step: usize,
) -> Result<bool> {
    if let Some(i) = acc.iter().position(|g| g.is_none()) {
        bail!(
            "step {step}: LoRA factor {} of {} received no gradient at all, so the \
             optimiser is a no-op on it and the adapter would save an untrained tensor",
            if i % 2 == 0 { "A" } else { "B" },
            loras[i / 2].0
        );
    }
    let mut finite = true;
    for slot in acc.iter_mut() {
        if let Some(g) = slot.take() {
            let g = g.affine(inv, 0.0)?;
            if !crate::grad_is_finite_by_one_scalar_readback_because_nan_and_inf_survive_a_sum(&g)?
            {
                finite = false;
            }
            *slot = Some(g);
        }
    }
    Ok(finite)
}

fn train_steps(
    vars: &[Var],
    loras: &[(String, Arc<LoraTrainable>)],
    steps: usize,
    lr: f64,
    b_lr_ratio: f64,
    n_examples: usize,
    mut loss_and_grads_for_example: impl FnMut(usize) -> Result<(f32, Vec<Option<Tensor>>)>,
) -> Result<(Vec<f32>, Vec<f64>, bool)> {
    let mut losses: Vec<f32> = Vec::with_capacity(steps);
    let mut step_seconds: Vec<f64> = Vec::with_capacity(steps);
    let mut all_grads_finite = true;
    if steps == 0 {
        return Ok((losses, step_seconds, all_grads_finite));
    }
    let mut opt = WindowOptimizer::new(vars, lr, b_lr_ratio)?;
    for step in 0..steps {
        let t0 = std::time::Instant::now();
        let mut acc: Vec<Option<Tensor>> = vec![None; vars.len()];
        let mut loss_sum = 0f32;
        for i in 0..n_examples {
            let (lv, grads) = loss_and_grads_for_example(i)?;
            loss_sum += lv;
            fold_example_grads(&mut acc, grads)?;
        }
        let inv = 1.0 / n_examples as f64;
        let dev = vars.first().map(|v| v.device().clone());
        let finite = match &dev {
            Some(d) => crate::train_prof::phase(d, "grad_average_and_finite_check", || {
                average_grads_bailing_on_an_untouched_factor(&mut acc, loras, inv, step)
            })?,
            None => average_grads_bailing_on_an_untouched_factor(&mut acc, loras, inv, step)?,
        };
        if !finite {
            all_grads_finite = false;
        }
        match &dev {
            Some(d) => crate::train_prof::phase(d, "optimizer_step", || opt.step(vars, &acc))?,
            None => opt.step(vars, &acc)?,
        }
        let lv = loss_sum * inv as f32;
        losses.push(lv);
        step_seconds.push(t0.elapsed().as_secs_f64());
        eprintln!(
            "[q38-lora-train] step {step:3}  loss {lv:.6}  {:.2}s",
            step_seconds[step]
        );
    }
    Ok((losses, step_seconds, all_grads_finite))
}

struct CoTrainStepOutcome {
    losses: Vec<f32>,
    step_seconds: Vec<f64>,
    all_grads_finite: bool,
    head_losses: Vec<f32>,
    all_head_grads_finite: bool,
}

#[allow(clippy::too_many_arguments)]
fn co_train_steps(
    vars: &[Var],
    loras: &[(String, Arc<LoraTrainable>)],
    head_vars: &[Var],
    head_loras: &[(String, Arc<LoraTrainable>)],
    steps: usize,
    lr: f64,
    b_lr_ratio: f64,
    head_lr: f64,
    n_examples: usize,
    mut both_losses_and_grads_for_example: impl FnMut(
        usize,
    ) -> Result<(
        (f32, Vec<Option<Tensor>>),
        (f32, Vec<Option<Tensor>>),
    )>,
) -> Result<CoTrainStepOutcome> {
    let mut out = CoTrainStepOutcome {
        losses: Vec::with_capacity(steps),
        step_seconds: Vec::with_capacity(steps),
        all_grads_finite: true,
        head_losses: Vec::with_capacity(steps),
        all_head_grads_finite: true,
    };
    if steps == 0 {
        return Ok(out);
    }
    let mut opt = WindowOptimizer::new(vars, lr, b_lr_ratio)?;
    let mut head_opt = WindowOptimizer::new(head_vars, head_lr, b_lr_ratio)?;
    for step in 0..steps {
        let t0 = std::time::Instant::now();
        let mut acc: Vec<Option<Tensor>> = vec![None; vars.len()];
        let mut head_acc: Vec<Option<Tensor>> = vec![None; head_vars.len()];
        let mut loss_sum = 0f32;
        let mut head_loss_sum = 0f32;
        for i in 0..n_examples {
            let ((lv, grads), (hl, head_grads)) = both_losses_and_grads_for_example(i)?;
            loss_sum += lv;
            head_loss_sum += hl;
            fold_example_grads(&mut acc, grads)?;
            fold_example_grads(&mut head_acc, head_grads)?;
        }
        let inv = 1.0 / n_examples as f64;
        if !average_grads_bailing_on_an_untouched_factor(&mut acc, loras, inv, step)? {
            out.all_grads_finite = false;
        }
        if !average_grads_bailing_on_an_untouched_factor(&mut head_acc, head_loras, inv, step)? {
            out.all_head_grads_finite = false;
        }
        opt.step(vars, &acc)?;
        head_opt.step(head_vars, &head_acc)?;
        let lv = loss_sum * inv as f32;
        let hl = head_loss_sum * inv as f32;
        out.losses.push(lv);
        out.head_losses.push(hl);
        out.step_seconds.push(t0.elapsed().as_secs_f64());
        eprintln!(
            "[q38-cotrain] step {step:3}  loss {lv:.6}  head_loss {hl:.6}  {:.2}s",
            out.step_seconds[step]
        );
    }
    Ok(out)
}

pub(crate) fn collect_raw_entries(
    loras: &[(String, Arc<LoraTrainable>)],
) -> Result<(Vec<RawLora>, f32, f32)> {
    let mut entries: Vec<RawLora> = Vec::new();
    let mut min_amax = f32::INFINITY;
    let mut max_amax = 0f32;
    for (module, lora) in loras {
        let a = lora.a_tensor().clone();
        let b = lora.b_tensor().clone();
        let delta = b
            .to_dtype(DType::F32)?
            .matmul(&a.to_dtype(DType::F32)?)?
            .affine(lora.scaling(), 0.0)?;
        let amax = delta.abs()?.flatten_all()?.max(0)?.to_scalar::<f32>()?;
        min_amax = min_amax.min(amax);
        max_amax = max_amax.max(amax);
        entries.push(RawLora {
            module: module.clone(),
            a,
            b,
        });
    }
    Ok((entries, min_amax, max_amax))
}

pub const QWEN38_MTP_SPECULATOR_KIND: &str = "qwen38_mtp_head";

pub const MTP_WEIGHTS_FILE_NAME: &str = "model_mtp.safetensors";

pub const MTP_COTRAIN_ACCEPTS_ANY_GRAD_LAYERS_BECAUSE_THE_WINDOW_ENDS_AT_THE_FINAL_TRUNK_LAYER_WHOSE_LIVE_OUTPUT_THE_HEAD_CONSUMES_DETACHED: &str =
    "The MTP head drafts from the trunk's pre-final-norm output, which is the trainable window's \
     last boundary: window_start = num_hidden_layers - grad_layers, so the window terminates at \
     the final trunk layer for every grad_layers in 1..=num_hidden_layers and the head always \
     sees hiddens recomputed under the current step's trunk LoRA state, never a stale cache. The \
     head consumes that boundary DETACHED, so its cross entropy reaches only head-LoRA vars: no \
     grad_layers choice can leak head loss into the target adapter, and the checkpointed window \
     backward needs no head-CE term folded into grad_next. A window that stopped short of the \
     final layer would hand the head an intermediate hidden the serving drafter never sees, so \
     co-training asserts the window covers through the last layer.";

pub const MTP_HEAD_TRAINING_REPLICATES_SERVING_ZERO_CENTERED_NORMS_EMB_FIRST_FC_SHIFT_BY_ONE_ZERO_HIDDEN_AT_POS_0_TRUNK_ROPE_SHARED_EMBED_AND_LM_HEAD: &str =
    "The differentiable head must be the served function \
     (Qwen38DenseMtpHead::forward_draft_from_token_ids_already_on_device in \
     nv-specdecode/src/qwen38_mtp.rs, unimportable from here because nv-specdecode depends on \
     nv-models which depends on nv-train): every mtp.* RMSNorm ships zero-centred and loads as \
     stored + 1; the fc input is cat(pre_fc_norm_embedding(embed(token[t])), \
     pre_fc_norm_hidden(trunk_hidden[t-1])) with the embedding half FIRST and a zero hidden \
     standing in at position 0; attention runs the trunk's rope and geometry over a causal \
     T-token graph, which chained m=1 serving steps factor through exactly; the head has no \
     embed or lm_head of its own, it reads the trunk's. Deviating on any of these trains a \
     different function than the one NV_DRAFTER=mtp serves, and because serving is \
     verify-anchored the damage would surface only as a silent acceptance-rate collapse.";

pub fn mtp_expected_tensor_names() -> Vec<String> {
    let mut names: Vec<String> = vec![
        "mtp.pre_fc_norm_embedding.weight".into(),
        "mtp.pre_fc_norm_hidden.weight".into(),
        "mtp.fc.weight".into(),
        "mtp.norm.weight".into(),
        "mtp.layers.0.input_layernorm.weight".into(),
        "mtp.layers.0.post_attention_layernorm.weight".into(),
        "mtp.layers.0.self_attn.q_norm.weight".into(),
        "mtp.layers.0.self_attn.k_norm.weight".into(),
    ];
    for leaf in ["q_proj", "k_proj", "v_proj", "o_proj"] {
        names.push(format!("mtp.layers.0.self_attn.{leaf}.weight"));
    }
    for leaf in ["gate_proj", "up_proj", "down_proj"] {
        names.push(format!("mtp.layers.0.mlp.{leaf}.weight"));
    }
    names
}

pub struct Q38MtpTrainHead {
    pre_fc_norm_embedding: RmsNorm,
    pre_fc_norm_hidden: RmsNorm,
    pub fc: Linear,
    input_layernorm: RmsNorm,
    pub attn: TrainAttn,
    post_norm: RmsNorm,
    pub mlp: TrainMlp,
    norm: RmsNorm,
    hidden: usize,
}

impl Q38MtpTrainHead {
    pub fn open_weights(base: &Path, device: &Device) -> Result<HostDequantWeights> {
        let dedicated = base.join(MTP_WEIGHTS_FILE_NAME);
        if dedicated.is_file() {
            return HostDequantWeights::open_single_file(&dedicated, device);
        }
        HostDequantWeights::open_dir(base, device)
    }

    pub fn from_checkpoint(
        base: &Path,
        cfg: &Q38Config,
        dtype: DType,
        device: &Device,
    ) -> Result<Self> {
        let src = Self::open_weights(base, device)?;
        let missing: Vec<String> = mtp_expected_tensor_names()
            .into_iter()
            .filter(|n| !src.has(n))
            .collect();
        ensure!(
            missing.is_empty(),
            "MTP co-training requested but the checkpoint at {} ships no complete MTP head: \
             missing tensors {missing:?} (searched {} then the main weight files). A base \
             without mtp.* weights has no self-drafting speculator to co-train",
            base.display(),
            base.join(MTP_WEIGHTS_FILE_NAME).display()
        );
        Self::from_weights(&src, cfg, dtype)
    }

    pub fn from_weights(src: &HostDequantWeights, cfg: &Q38Config, dtype: DType) -> Result<Self> {
        let _ = MTP_HEAD_TRAINING_REPLICATES_SERVING_ZERO_CENTERED_NORMS_EMB_FIRST_FC_SHIFT_BY_ONE_ZERO_HIDDEN_AT_POS_0_TRUNK_ROPE_SHARED_EMBED_AND_LM_HEAD;
        let eps = cfg.rms_norm_eps;
        let norm = |name: &str| -> Result<RmsNorm> {
            Ok(RmsNorm::new(src.trunk_rmsnorm_gamma(name, dtype)?, eps))
        };
        Ok(Self {
            pre_fc_norm_embedding: norm("mtp.pre_fc_norm_embedding.weight")?,
            pre_fc_norm_hidden: norm("mtp.pre_fc_norm_hidden.weight")?,
            fc: src.dense_linear("mtp.fc", dtype)?,
            input_layernorm: norm("mtp.layers.0.input_layernorm.weight")?,
            attn: TrainAttn::from_weights(src, "mtp.layers.0.self_attn", cfg, dtype)?,
            post_norm: norm("mtp.layers.0.post_attention_layernorm.weight")?,
            mlp: TrainMlp::from_weights(src, "mtp.layers.0.mlp", dtype)?,
            norm: norm("mtp.norm.weight")?,
            hidden: cfg.hidden_size,
        })
    }

    pub fn projections(&self) -> Vec<(&'static str, &Linear)> {
        vec![
            ("fc", &self.fc),
            ("q_proj", &self.attn.q_proj),
            ("k_proj", &self.attn.k_proj),
            ("v_proj", &self.attn.v_proj),
            ("o_proj", &self.attn.o_proj),
            ("gate_proj", &self.mlp.gate_proj),
            ("up_proj", &self.mlp.up_proj),
            ("down_proj", &self.mlp.down_proj),
        ]
    }

    pub fn module_name(leaf: &str) -> String {
        match leaf {
            "fc" => "mtp.fc".to_string(),
            "q_proj" | "k_proj" | "v_proj" | "o_proj" => {
                format!("mtp.layers.0.self_attn.{leaf}")
            }
            _ => format!("mtp.layers.0.mlp.{leaf}"),
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn shifted_draft_ce(
        &self,
        emb: &Tensor,
        trunk_out_detached: &Tensor,
        rope: &Rope,
        positions: &Tensor,
        lm_head: &Linear,
        tgt: &Tensor,
        mask: &[f32],
        vocab: usize,
    ) -> Result<Tensor> {
        let dims = trunk_out_detached.dims().to_vec();
        let h = self.hidden;
        ensure!(
            dims.len() == 3 && dims[0] == 1 && dims[2] == h,
            "MTP head CE expects trunk hiddens [1, T, {h}], got {dims:?}"
        );
        let t = dims[1];
        ensure!(
            emb.dims() == [1, t, h],
            "MTP head CE: {t} trunk rows need token embeddings [1, {t}, {h}], got {:?}",
            emb.dims()
        );
        let zero_hidden_stands_in_for_h_minus_one_and_rmsnorms_to_zero = Tensor::zeros(
            (1usize, 1usize, h),
            trunk_out_detached.dtype(),
            trunk_out_detached.device(),
        )?;
        let shifted = if t == 1 {
            zero_hidden_stands_in_for_h_minus_one_and_rmsnorms_to_zero
        } else {
            Tensor::cat(
                &[
                    &zero_hidden_stands_in_for_h_minus_one_and_rmsnorms_to_zero,
                    &trunk_out_detached.narrow(1, 0, t - 1)?,
                ],
                1,
            )?
            .contiguous()?
        };
        let norm_emb = self.pre_fc_norm_embedding.forward_candle(emb)?;
        let norm_hid = self.pre_fc_norm_hidden.forward_candle(&shifted)?;
        let fused = Tensor::cat(&[&norm_emb, &norm_hid], candle_core::D::Minus1)?.contiguous()?;
        let h0 = self.fc.forward(&fused)?;
        let normed = self.input_layernorm.forward_candle(&h0)?;
        let x = h0.add(&self.attn.forward(&normed, rope, positions)?)?;
        let normed_post = self.post_norm.forward_candle(&x)?;
        let x = x.add(&self.mlp.forward(&normed_post)?)?;
        let normed_final = self.norm.forward_candle(&x)?;
        let logits = lm_head
            .forward(&normed_final)?
            .reshape((t, vocab))?
            .to_dtype(DType::F32)?;
        crate::ce_mean_over_unmasked_positions(&logits, tgt, mask)
    }
}

pub const MTP_HEAD_LORA_SEEDS_LIVE_PAST_1_SHL_32_SO_NO_HEAD_FACTOR_SHARES_AN_INIT_WITH_A_WINDOW_MODULE_WHOSE_SEEDS_ADVANCE_BY_ONE:
    u64 = 1u64 << 32;

pub fn attach_mtp_head_loras(
    head: &Q38MtpTrainHead,
    lcfg: LoraConfig,
    seed0: u64,
    device: &Device,
) -> Result<(Vec<(String, Arc<LoraTrainable>)>, Vec<Var>)> {
    let mut loras: Vec<(String, Arc<LoraTrainable>)> = Vec::new();
    let mut vars: Vec<Var> = Vec::new();
    let mut seed =
        seed0.wrapping_add(MTP_HEAD_LORA_SEEDS_LIVE_PAST_1_SHL_32_SO_NO_HEAD_FACTOR_SHARES_AN_INIT_WITH_A_WINDOW_MODULE_WHOSE_SEEDS_ADVANCE_BY_ONE);
    for (leaf, lin) in head.projections() {
        let lora = Arc::new(LoraTrainable::from_features(
            lin.out_features(),
            lin.in_features(),
            lcfg,
            seed,
            device,
        )?);
        seed = seed.wrapping_add(1);
        vars.extend(lora.trainable_vars());
        let hook: Arc<dyn LoraDeltaHook> = Arc::new(TrainingLoraHook::new(lora.clone(), true));
        lin.attach_lora(hook)?;
        loras.push((Q38MtpTrainHead::module_name(leaf), lora));
    }
    Ok((loras, vars))
}

pub const MTP_COTRAIN_HEAD_RANK_DEFAULT_IS_32_AND_SPECULATOR_LR_RATIO_DEFAULT_IS_4_PER_THE_MEDUSA2_COTRAIN_RECIPE:
    (usize, f64) = (32, 4.0);

#[derive(Clone, Debug)]
pub struct Q38MtpCoTrainArgs {
    pub rank: usize,
    pub alpha: f64,
    pub speculator_lr_ratio: f64,
}

impl Default for Q38MtpCoTrainArgs {
    fn default() -> Self {
        let (rank, ratio) =
            MTP_COTRAIN_HEAD_RANK_DEFAULT_IS_32_AND_SPECULATOR_LR_RATIO_DEFAULT_IS_4_PER_THE_MEDUSA2_COTRAIN_RECIPE;
        Self {
            rank,
            alpha: rank as f64,
            speculator_lr_ratio: ratio,
        }
    }
}

#[derive(Debug)]
pub struct Q38MtpCoTrainSummary {
    pub head_losses: Vec<f32>,
    pub head_modules: Vec<String>,
    pub head_trainable_vars: usize,
    pub all_head_grads_finite: bool,
    pub speculator_kind: String,
    pub speculator_dir: PathBuf,
}

struct MtpCoTrainEngine {
    head: Q38MtpTrainHead,
    loras: Vec<(String, Arc<LoraTrainable>)>,
    vars: Vec<Var>,
    emb: Vec<Tensor>,
}

impl MtpCoTrainEngine {
    #[allow(clippy::too_many_arguments)]
    fn build(
        co: &Q38MtpCoTrainArgs,
        cfg: &Q38Config,
        base: &Path,
        main_src: &HostDequantWeights,
        input_ids_per_row: &[&[u32]],
        dtype: DType,
        device: &Device,
        seed: u64,
    ) -> Result<Self> {
        let head = Q38MtpTrainHead::from_checkpoint(base, cfg, dtype, device)?;
        let lcfg = LoraConfig {
            r: co.rank,
            alpha: co.alpha,
            dropout: 0.0,
        };
        let (loras, vars) = attach_mtp_head_loras(&head, lcfg, seed, device)?;
        let mut emb: Vec<Tensor> = Vec::with_capacity(input_ids_per_row.len());
        for ids in input_ids_per_row {
            emb.push(
                main_src
                    .embed_rows(EMBED_TOKENS, ids, cfg.hidden_size)?
                    .reshape((1, ids.len(), cfg.hidden_size))?
                    .to_dtype(dtype)?,
            );
        }
        Ok(Self {
            head,
            loras,
            vars,
            emb,
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn head_loss_and_grads(
        &self,
        i: usize,
        window_out_detached: &Tensor,
        rope: &Rope,
        positions: &Tensor,
        lm_head: &Linear,
        tgt: &Tensor,
        mask: &[f32],
        vocab: usize,
    ) -> Result<(f32, Vec<Option<Tensor>>)> {
        let loss = self.head.shifted_draft_ce(
            &self.emb[i],
            window_out_detached,
            rope,
            positions,
            lm_head,
            tgt,
            mask,
            vocab,
        )?;
        loss_and_grads_from_one_backward(loss, &self.vars)
    }

    fn artifact(&self) -> Result<(crate::SpeculatorArtifact, Vec<String>)> {
        let (entries, _min_amax, _max_amax) = collect_raw_entries(&self.loras)?;
        let modules: Vec<String> = entries.iter().map(|e| e.module.clone()).collect();
        Ok((
            crate::SpeculatorArtifact {
                kind: QWEN38_MTP_SPECULATOR_KIND.to_string(),
                entries,
                r: self.loras[0].1.rank(),
                alpha: self.loras[0].1.alpha(),
            },
            modules,
        ))
    }
}

pub const GRAD_LAYERS_ABOVE_WHICH_RUN_JSONL_CHECKPOINTS_THE_WINDOW_BECAUSE_A_DEEPER_FULL_GDN_GRAPH_OUTGROWS_THE_CARD:
    usize = 4;

fn window_checkpointing_with_nv_train_ckpt_as_the_named_tristate_escape_hatch(
    grad_layers: usize,
) -> bool {
    match std::env::var("NV_TRAIN_CKPT").ok().as_deref() {
        Some("1") => true,
        Some("0") => false,
        _ => {
            grad_layers
                > GRAD_LAYERS_ABOVE_WHICH_RUN_JSONL_CHECKPOINTS_THE_WINDOW_BECAUSE_A_DEEPER_FULL_GDN_GRAPH_OUTGROWS_THE_CARD
        }
    }
}

#[derive(Clone, Debug)]
pub struct Q38JsonlArgs {
    pub base: PathBuf,
    pub data: PathBuf,
    pub eval_data: Option<PathBuf>,
    pub out: PathBuf,
    pub grad_layers: usize,
    pub rank: usize,
    pub alpha: f64,
    pub targets: Vec<String>,
    pub steps: usize,
    pub lr: f64,
    pub b_lr_ratio: f64,
    pub seed: u64,
}

pub const B_LR_RATIO_DEFAULT_OF_ONE_IS_EXACTLY_THE_SINGLE_OPTIMIZER_PATH_AND_ABOVE_ONE_IS_LORA_PLUS:
    f64 = 1.0;

#[derive(Debug)]
pub struct Q38JsonlSummary {
    pub losses: Vec<f32>,
    pub eval_loss_adapter: Option<f32>,
    pub eval_loss_base: Option<f32>,
    pub eval_ce_adapter: Vec<f32>,
    pub eval_ce_base: Vec<f32>,
    pub modules: Vec<String>,
    pub trainable_vars: usize,
    pub num_examples: usize,
    pub num_eval_examples: usize,
    pub window_start: usize,
    pub window_kinds: Vec<&'static str>,
    pub base_dtype: String,
    pub resident_base: bool,
    pub all_grads_finite: bool,
    pub checkpointed: bool,
    pub device: String,
    pub adapter_path: PathBuf,
    pub config_path: PathBuf,
}

pub const NV_Q38_EVAL_W4A4_SCORES_HELDOUT_ROWS_UNDER_THE_SERVING_QUANTIZED_FORWARD: &str =
    "NV_Q38_EVAL_W4A4";

pub const W4A4_EVAL_ARM_STAYS_ALONGSIDE_QUANT_RESIDENT_TRAINING_BECAUSE_IT_SCORES_A_HOST_DEQUANT_TRAINED_ADAPTER_UNDER_THE_SERVING_FORWARD: &str =
    "Default training runs host-dequantized dense weights (HostDequantWeights::dense_linear -> \
     bf16 Linear storage); serving keeps packed NVFP4 resident and quantizes activations per \
     call (matmul_nvfp4_impl -> quantize_nvfp4_bf16), a W4A4 GEMM no upstream QLoRA trainer \
     runs. NV_Q38_EVAL_W4A4=1 trains once under the dense-bf16 forward, then scores the same \
     held-out rows under both forwards; NV_TRAIN_QUANT_RESIDENT=1 instead trains WITH the \
     resident forward via QuantLinearOp (dx-only backward), so the two arms answer different \
     questions and neither supersedes the other. W4A4 results go to stderr and \
     <out>/w4a4_eval.json because Q38JsonlSummary's field set is mirrored by an exhaustive \
     literal in src/oapi/fine_tuning.rs.";

fn w4a4_eval_arm_opted_in() -> bool {
    nv_config::env_arm_opt_in_eq_some_1_reread_every_call(
        NV_Q38_EVAL_W4A4_SCORES_HELDOUT_ROWS_UNDER_THE_SERVING_QUANTIZED_FORWARD,
    )
}

pub const NV_TRAIN_QUANT_RESIDENT_DEFAULT_ON_USER_SIGNOFF_2026_09_13_QLORA_QUANTIZED_ALWAYS_0_IS_THE_HOST_DEQUANT_ORACLE_KEEPS_NVFP4_AND_FP8_MODULES_RESIDENT_SO_TRAINING_RUNS_THE_SERVING_QUANT_GEMM: &str =
    "NV_TRAIN_QUANT_RESIDENT";

pub const QUANT_RESIDENT_TRAINS_THROUGH_QUANT_LINEAR_OP_AND_LEAVES_BF16_MODULES_HOST_DEQUANT: &str =
    "With NV_TRAIN_QUANT_RESIDENT=1 every checkpoint module classified Nvfp4Packed or \
     Fp8RowScale is ingested as a quantized-resident Linear (the serving loaders' storage), so \
     the frozen prefix runs the serving quantized GEMM as-is and window Linears exercise \
     QuantLinearOp: forward is the serving quantized GEMM, backward is dx-only through a device \
     dequant of the frozen base, and the LoRA delta hook carries the only trainable parameters. \
     Bf16-stored modules (and the MTP head, which loads through its own small \
     HostDequantWeights) stay host-dequant. This is the default since the 2026-09-13 ruling \
     that training is quantized always; NV_TRAIN_QUANT_RESIDENT=0 is the host-dequant oracle \
     the byte-identity test keeps, not a measured arm.";

pub const QUANT_RESIDENT_REFUSES_AN_ALL_BF16_CHECKPOINT_BECAUSE_A_SILENT_FALLTHROUGH_WOULD_FAKE_THE_ARM: &str =
    "the refuses-loudly design: a checkpoint with no Nvfp4Packed/Fp8RowScale linear would make \
     NV_TRAIN_QUANT_RESIDENT=1 an exact no-op, and a ppl row labelled resident that ran the \
     host-dequant path is worse than no row";

pub fn quant_resident_arm_opted_in() -> bool {
    nv_config::env_arm_opt_out_default_on_unless_eq_some_0_reread_every_call(
        NV_TRAIN_QUANT_RESIDENT_DEFAULT_ON_USER_SIGNOFF_2026_09_13_QLORA_QUANTIZED_ALWAYS_0_IS_THE_HOST_DEQUANT_ORACLE_KEEPS_NVFP4_AND_FP8_MODULES_RESIDENT_SO_TRAINING_RUNS_THE_SERVING_QUANT_GEMM,
    )
}

pub const A_BF16_ONLY_CHECKPOINT_UNDER_THE_QUANTIZED_ALWAYS_DEFAULT_IS_THE_TEST_ORACLE_AND_TRAINS_HOST_DEQUANT_ONLY_AN_EXPLICIT_1_REFUSES_IT: &str =
    "the 2026-09-13 ruling keeps bf16 only as an internal test oracle: a checkpoint with no \
     nvfp4-packed or fp8-rowscale linear (the tiny fixtures) has nothing to keep resident, so \
     under the default it trains on the host-dequant path and says so; NV_TRAIN_QUANT_RESIDENT=1 \
     set by hand still refuses it, because that arm must never be a silent no-op";

fn quant_resident_set_by_hand() -> bool {
    std::env::var(
        NV_TRAIN_QUANT_RESIDENT_DEFAULT_ON_USER_SIGNOFF_2026_09_13_QLORA_QUANTIZED_ALWAYS_0_IS_THE_HOST_DEQUANT_ORACLE_KEEPS_NVFP4_AND_FP8_MODULES_RESIDENT_SO_TRAINING_RUNS_THE_SERVING_QUANT_GEMM,
    )
    .as_deref()
        == Ok("1")
}

pub fn open_train_weights(base: &Path, device: &Device) -> Result<(HostDequantWeights, bool)> {
    let src = HostDequantWeights::open_dir(base, device)?;
    if !quant_resident_arm_opted_in() {
        return Ok((src, false));
    }
    let _ = QUANT_RESIDENT_TRAINS_THROUGH_QUANT_LINEAR_OP_AND_LEAVES_BF16_MODULES_HOST_DEQUANT;
    if !quant_resident_set_by_hand() && (!src.has_any_quantized_linear() || !device.is_cuda()) {
        let _ = A_BF16_ONLY_CHECKPOINT_UNDER_THE_QUANTIZED_ALWAYS_DEFAULT_IS_THE_TEST_ORACLE_AND_TRAINS_HOST_DEQUANT_ONLY_AN_EXPLICIT_1_REFUSES_IT;
        eprintln!(
            "[q38-lora-train] {} holds no quantized linear or runs on {:?}: host-dequant oracle path",
            base.display(),
            device.location()
        );
        return Ok((src, false));
    }
    ensure!(
        src.has_any_quantized_linear(),
        "NV_TRAIN_QUANT_RESIDENT=1 but {} holds no nvfp4-packed or fp8-rowscale linear -- {}",
        base.display(),
        QUANT_RESIDENT_REFUSES_AN_ALL_BF16_CHECKPOINT_BECAUSE_A_SILENT_FALLTHROUGH_WOULD_FAKE_THE_ARM
    );
    #[cfg(feature = "cuda")]
    {
        let cuda = match device {
            Device::Cuda(d) => d.clone(),
            _ => bail!(
                "NV_TRAIN_QUANT_RESIDENT=1 needs a CUDA device: quantized-resident storages do \
                 not exist on CPU"
            ),
        };
        let nvfp4 = Arc::new(Mutex::new(nv_quant::nvfp4::Nvfp4GemmRunner::new(
            cuda.cuda_stream(),
        )?));
        let fp8 = Arc::new(Mutex::new(nv_quant::fp8::Fp8GemmRunner::new(
            cuda.cuda_stream(),
        )?));
        Ok((src.with_quant_resident_arms(nvfp4, fp8), true))
    }
    #[cfg(not(feature = "cuda"))]
    {
        bail!(
            "NV_TRAIN_QUANT_RESIDENT=1 but this build has no cuda feature; quantized-resident \
             storages only exist under cuda"
        )
    }
}

pub type ShiftedRow = (Tensor, Tensor, Tensor, Vec<f32>);

fn embed_shifted_rows(
    src: &HostDequantWeights,
    rows: &[crate::TokenRow],
    hidden: usize,
    dtype: DType,
    device: &Device,
) -> Result<Vec<ShiftedRow>> {
    rows.iter()
        .map(|row| {
            let ids = &row.ids;
            let t = ids.len() - 1;
            let h = src
                .embed_rows(EMBED_TOKENS, &ids[..t], hidden)?
                .reshape((1, t, hidden))?
                .to_dtype(dtype)?;
            let positions = flat_positions(1, t, device)?;
            let tgt = Tensor::from_vec(ids[1..].to_vec(), t, device)?;
            let mask = row.target_mask_where_predicting_a_prompt_token_weighs_zero();
            Ok((h, positions, tgt, mask))
        })
        .collect()
}

pub(crate) fn logits_f32_after_head(
    final_norm: &RmsNorm,
    lm_head: &Linear,
    x: &Tensor,
    vocab: usize,
) -> Result<Tensor> {
    let t = x.dims()[1];
    let x = final_norm.forward_candle(x)?;
    Ok(lm_head
        .forward(&x)?
        .reshape((t, vocab))?
        .to_dtype(DType::F32)?)
}

fn detached_window_output(
    window: &[TrainLayer],
    rope: &Rope,
    h: &Tensor,
    positions: &Tensor,
) -> Result<Tensor> {
    let mut x = h.detach();
    for layer in window {
        x = layer.forward(&x, rope, positions)?.detach();
    }
    Ok(x)
}

fn per_row_holdout_ce_in_loaded_dataset_row_order(
    window: &[TrainLayer],
    final_norm: &RmsNorm,
    lm_head: &Linear,
    rope: &Rope,
    vocab: usize,
    exs: &[ShiftedRow],
) -> Result<Vec<f32>> {
    exs.iter()
        .map(|(h, positions, tgt, mask)| {
            let x = detached_window_output(window, rope, h, positions)?;
            let logits = logits_f32_after_head(final_norm, lm_head, &x, vocab)?;
            Ok(crate::ce_mean_over_unmasked_positions(&logits, tgt, mask)?.to_scalar::<f32>()?)
        })
        .collect()
}

#[cfg(feature = "cuda")]
fn row0_logits(
    window: &[TrainLayer],
    final_norm: &RmsNorm,
    lm_head: &Linear,
    rope: &Rope,
    vocab: usize,
    ex: &ShiftedRow,
) -> Result<Tensor> {
    let x = detached_window_output(window, rope, &ex.0, &ex.1)?;
    logits_f32_after_head(final_norm, lm_head, &x, vocab)
}

#[cfg(feature = "cuda")]
#[allow(clippy::too_many_arguments)]
fn run_w4a4_eval_arm(
    args: &Q38JsonlArgs,
    cfg: &Q38Config,
    eval_rows: &[crate::TokenRow],
    bf16_eval_ex: &[ShiftedRow],
    bf16_window: &[TrainLayer],
    loras: &[(String, Arc<crate::LoraTrainable>)],
    final_norm: &RmsNorm,
    lm_head: &Linear,
    rope: &Rope,
    window_start: usize,
    dtype: DType,
    device: &Device,
    train_losses: &[f32],
    eval_ce_base_bf16: &[f32],
    eval_ce_adapter_bf16: &[f32],
) -> Result<()> {
    let cuda = match device {
        Device::Cuda(d) => d.clone(),
        _ => bail!("NV_Q38_EVAL_W4A4=1 needs a CUDA device: the serving W4A4 GEMM has no CPU path"),
    };
    let runner = Arc::new(Mutex::new(nv_quant::nvfp4::Nvfp4GemmRunner::new(
        cuda.cuda_stream(),
    )?));
    let src = HostDequantWeights::open_dir(&args.base, device)?.with_serving_w4a4_arm(runner);
    let mut exs = embed_shifted_rows(&src, eval_rows, cfg.hidden_size, dtype, device)?;
    for li in 0..window_start {
        let layer = TrainLayer::from_weights(&src, cfg, li, dtype)
            .with_context(|| format!("build w4a4 prefix layer {li}"))?;
        for (h, positions, _, _) in exs.iter_mut() {
            *h = layer.forward(h, rope, positions)?.detach();
        }
        drop(layer);
    }
    let mut window: Vec<TrainLayer> = Vec::with_capacity(cfg.num_hidden_layers - window_start);
    for li in window_start..cfg.num_hidden_layers {
        window.push(
            TrainLayer::from_weights(&src, cfg, li, dtype)
                .with_context(|| format!("build w4a4 window layer {li}"))?,
        );
    }
    let vocab = cfg.vocab_size;
    let per_row_base_q = per_row_holdout_ce_in_loaded_dataset_row_order(
        &window, final_norm, lm_head, rope, vocab, &exs,
    )?;
    let base_q = crate::eval_loss_scalar_is_the_sum_of_the_per_row_ce_means_in_dataset_row_order(
        &per_row_base_q,
    )
    .unwrap_or(f32::NAN);
    let mut attached = 0usize;
    for layer in &window {
        for (leaf, lin) in layer.projections() {
            let name = layer.module_name(leaf);
            if let Some((_, lora)) = loras.iter().find(|(m, _)| *m == name) {
                lin.attach_lora(Arc::new(crate::StaticLoraHook::new(
                    lora.a_tensor().clone(),
                    lora.b_tensor().clone(),
                    lora.scaling(),
                )?))?;
                attached += 1;
            }
        }
    }
    ensure!(
        attached == loras.len(),
        "w4a4 eval arm attached {attached} of {} trained modules; a module missing from the \
         serving-forward score would fake parity",
        loras.len()
    );
    let per_row_adapter_q = per_row_holdout_ce_in_loaded_dataset_row_order(
        &window, final_norm, lm_head, rope, vocab, &exs,
    )?;
    let adapter_q =
        crate::eval_loss_scalar_is_the_sum_of_the_per_row_ce_means_in_dataset_row_order(
            &per_row_adapter_q,
        )
        .unwrap_or(f32::NAN);
    let bf16_l0 = row0_logits(
        bf16_window,
        final_norm,
        lm_head,
        rope,
        vocab,
        &bf16_eval_ex[0],
    )?;
    let w4a4_l0 = row0_logits(&window, final_norm, lm_head, rope, vocab, &exs[0])?;
    let row0_diff = crate::max_abs_diff(&bf16_l0, &w4a4_l0)?;
    eprintln!(
        "[q38-w4a4-eval] eval_loss_base_w4a4 = {base_q:.6e}  eval_loss_adapter_w4a4 = \
         {adapter_q:.6e}  row0_logit_max_abs_diff = {row0_diff:.6e}  examples {}",
        exs.len()
    );
    let sum_of = |v: &[f32]| {
        crate::eval_loss_scalar_is_the_sum_of_the_per_row_ce_means_in_dataset_row_order(v)
            .unwrap_or(f32::NAN)
    };
    let sidecar = serde_json::json!({
        "num_eval_examples": exs.len(),
        "train_losses": train_losses,
        "eval_loss_base_bf16": sum_of(eval_ce_base_bf16),
        "eval_loss_adapter_bf16": sum_of(eval_ce_adapter_bf16),
        "eval_loss_base_w4a4": base_q,
        "eval_loss_adapter_w4a4": adapter_q,
        "eval_ce_base_bf16": eval_ce_base_bf16,
        "eval_ce_adapter_bf16": eval_ce_adapter_bf16,
        "eval_ce_base_w4a4": per_row_base_q,
        "eval_ce_adapter_w4a4": per_row_adapter_q,
        "row0_logit_max_abs_diff_bf16_vs_w4a4": row0_diff,
    });
    std::fs::create_dir_all(&args.out)?;
    std::fs::write(
        args.out.join("w4a4_eval.json"),
        serde_json::to_string_pretty(&sidecar)?,
    )?;
    Ok(())
}

#[cfg(not(feature = "cuda"))]
#[allow(clippy::too_many_arguments)]
fn run_w4a4_eval_arm(
    _args: &Q38JsonlArgs,
    _cfg: &Q38Config,
    _eval_rows: &[crate::TokenRow],
    _bf16_eval_ex: &[ShiftedRow],
    _bf16_window: &[TrainLayer],
    _loras: &[(String, Arc<crate::LoraTrainable>)],
    _final_norm: &RmsNorm,
    _lm_head: &Linear,
    _rope: &Rope,
    _window_start: usize,
    _dtype: DType,
    _device: &Device,
    _train_losses: &[f32],
    _eval_ce_base_bf16: &[f32],
    _eval_ce_adapter_bf16: &[f32],
) -> Result<()> {
    bail!(
        "NV_Q38_EVAL_W4A4=1 but this build has no cuda feature; the serving W4A4 GEMM only \
         exists under cuda"
    )
}

pub const EVAL_ROWS_SHARE_THE_FROZEN_PREFIX_PASS_BECAUSE_NO_TRAINABLE_PARAMETER_LIVES_BELOW_THE_WINDOW: &str =
    "Held-out rows are embedded and streamed through the frozen prefix in the same one-layer-\
     resident pass as the training rows, before any optimiser step. That cache stays valid for \
     both the pre-attach base eval and the post-training adapter eval because every LoRA lives \
     inside the window; attaching a hook cannot change any hidden state entering it, and the \
     base weights never move, so scoring the window before any LoRA exists IS the detached eval.";

pub const WINDOW_ROWS_PER_GRAPH_ENV_DEFAULT_4_BECAUSE_FOUR_512_ROWS_THROUGH_THE_CHECKPOINTED_CHUNKED_WALK_RUN_2_81_S_AT_29_GIB_AND_EIGHT_RUN_2_59_S_AT_51_GIB_1_RESTORES_ONE_ROW_PER_GRAPH: &str =
    "NV_TRAIN_WINDOW_ROWS";

pub const WINDOW_ROWS_PEAK_GIB_LADDER_MEASURED_ON_8X512_AT_250W_ROWS_8_51_0_ROWS_4_29_1_ROWS_2_18_0_FASTEST_FIRST: [(usize, f64); 3] =
    [(8, 51.0), (4, 29.1), (2, 18.0)];

pub const WINDOW_ROWS_HEADROOM_GIB_8_BECAUSE_THE_PEAK_LADDER_WAS_MEASURED_ON_512_TOKEN_ROWS_AND_LONGER_ROWS_OR_A_WARM_ALLOCATOR_NEED_SLACK: f64 =
    8.0;

pub fn window_rows_per_graph_adaptive(device_total_gib: f64, used_by_others_gib: f64) -> usize {
    let budget = device_total_gib
        - used_by_others_gib
        - WINDOW_ROWS_HEADROOM_GIB_8_BECAUSE_THE_PEAK_LADDER_WAS_MEASURED_ON_512_TOKEN_ROWS_AND_LONGER_ROWS_OR_A_WARM_ALLOCATOR_NEED_SLACK;
    WINDOW_ROWS_PEAK_GIB_LADDER_MEASURED_ON_8X512_AT_250W_ROWS_8_51_0_ROWS_4_29_1_ROWS_2_18_0_FASTEST_FIRST
        .iter()
        .find(|(_, peak)| *peak <= budget)
        .map(|(rows, _)| *rows)
        .unwrap_or(1)
}

fn window_rows_per_graph(device: &Device) -> usize {
    if let Some(rows) = std::env::var(
        WINDOW_ROWS_PER_GRAPH_ENV_DEFAULT_4_BECAUSE_FOUR_512_ROWS_THROUGH_THE_CHECKPOINTED_CHUNKED_WALK_RUN_2_81_S_AT_29_GIB_AND_EIGHT_RUN_2_59_S_AT_51_GIB_1_RESTORES_ONE_ROW_PER_GRAPH,
    )
    .ok()
    .and_then(|v| v.parse::<usize>().ok())
    {
        return rows.max(1);
    }
    #[cfg(feature = "cuda")]
    if device.is_cuda() {
        if let Ok((free, total)) = cudarc::driver::result::mem_get_info() {
            let gib = |b: usize| b as f64 / (1u64 << 30) as f64;
            let used_by_others = (gib(total)
                - gib(free)
                - DEVICE_USED_BY_THIS_PROCESS_GIB_AT_LOAD
                    .lock()
                    .map(|g| *g)
                    .unwrap_or(0.0))
            .max(0.0);
            let rows = window_rows_per_graph_adaptive(gib(total), used_by_others);
            eprintln!(
                "[q38-lora-train] window rows per graph {rows} (device {:.1} GiB, others {:.1} GiB, this process {:.1} GiB at load, headroom {} GiB)",
                gib(total),
                used_by_others,
                DEVICE_USED_BY_THIS_PROCESS_GIB_AT_LOAD.lock().map(|g| *g).unwrap_or(0.0),
                WINDOW_ROWS_HEADROOM_GIB_8_BECAUSE_THE_PEAK_LADDER_WAS_MEASURED_ON_512_TOKEN_ROWS_AND_LONGER_ROWS_OR_A_WARM_ALLOCATOR_NEED_SLACK
            );
            return rows;
        }
    }
    let _ = device;
    4
}

static DEVICE_USED_BY_THIS_PROCESS_GIB_AT_LOAD: Mutex<f64> = Mutex::new(0.0);

#[cfg(feature = "cuda")]
fn note_device_use_of_this_process_between(before_free: usize) {
    if let Ok((free_now, _)) = cudarc::driver::result::mem_get_info() {
        let used = before_free.saturating_sub(free_now) as f64 / (1u64 << 30) as f64;
        if let Ok(mut g) = DEVICE_USED_BY_THIS_PROCESS_GIB_AT_LOAD.lock() {
            *g = used;
        }
    }
}

fn equal_length_row_groups(lengths: &[usize], rows_per_graph: usize) -> Vec<Vec<usize>> {
    let mut groups: Vec<Vec<usize>> = Vec::new();
    for (i, &t) in lengths.iter().enumerate() {
        match groups
            .last_mut()
            .filter(|g| g.len() < rows_per_graph && lengths[g[0]] == t)
        {
            Some(g) => g.push(i),
            None => groups.push(vec![i]),
        }
    }
    groups
}

pub fn run_jsonl(args: &Q38JsonlArgs, device: &Device) -> Result<Q38JsonlSummary> {
    Ok(run_jsonl_inner(args, None, device)?.0)
}

pub fn run_jsonl_with_mtp_cotrain(
    args: &Q38JsonlArgs,
    co: &Q38MtpCoTrainArgs,
    device: &Device,
) -> Result<(Q38JsonlSummary, Q38MtpCoTrainSummary)> {
    let (summary, co_summary) = run_jsonl_inner(args, Some(co), device)?;
    Ok((
        summary,
        co_summary.expect("co-training was requested, so the speculator summary exists"),
    ))
}

fn run_jsonl_inner(
    args: &Q38JsonlArgs,
    co: Option<&Q38MtpCoTrainArgs>,
    device: &Device,
) -> Result<(Q38JsonlSummary, Option<Q38MtpCoTrainSummary>)> {
    let _ = THE_FROZEN_PREFIX_IS_CACHED_AS_HIDDEN_STATES_WHICH_IS_ONLY_SOUND_BECAUSE_NO_TRAINABLE_PARAMETER_LIVES_BELOW_THE_WINDOW;
    let _ = EVAL_ROWS_SHARE_THE_FROZEN_PREFIX_PASS_BECAUSE_NO_TRAINABLE_PARAMETER_LIVES_BELOW_THE_WINDOW;

    let cfg = Q38Config::from_hf_json_file(&args.base.join("config.json"))?;
    let grad_layers = args.grad_layers.clamp(1, cfg.num_hidden_layers);
    let window_start = cfg.num_hidden_layers - grad_layers;
    #[cfg(feature = "cuda")]
    let device_free_before_this_process_loaded = cudarc::driver::result::mem_get_info()
        .map(|(free, _)| free)
        .unwrap_or(0);
    let (src, resident_base) = open_train_weights(&args.base, device)?;
    let dtype = if device.is_cuda() {
        DType::BF16
    } else {
        DType::F32
    };

    let max_seq =
        crate::JSONL_MAX_SEQ_IS_512_SHARED_BY_BOTH_ENDPOINT_ARMS_BECAUSE_THE_DEFAULT_QWEN_WINDOWS_GDN_GRAPH_AT_512_IS_ABOUT_14_GB;
    let train_rows =
        crate::load_jsonl_token_dataset(&args.data, &args.base, cfg.vocab_size, max_seq)?;
    let eval_rows = match &args.eval_data {
        Some(p) => Some(crate::load_jsonl_token_dataset(
            p,
            &args.base,
            cfg.vocab_size,
            max_seq,
        )?),
        None => None,
    };

    let max_t = train_rows
        .iter()
        .chain(eval_rows.iter().flatten())
        .map(|r| r.ids.len() - 1)
        .max()
        .unwrap_or(1);
    let rope = build_rope(&cfg, max_t, device)?;

    let mut train_ex = embed_shifted_rows(&src, &train_rows, cfg.hidden_size, dtype, device)?;
    let mut eval_ex = match &eval_rows {
        Some(rows) => embed_shifted_rows(&src, rows, cfg.hidden_size, dtype, device)?,
        None => Vec::new(),
    };

    for li in 0..window_start {
        let layer = TrainLayer::from_weights(&src, &cfg, li, dtype)
            .with_context(|| format!("build frozen prefix layer {li}"))?;
        for (h, positions, _, _) in train_ex.iter_mut().chain(eval_ex.iter_mut()) {
            *h = layer.forward(h, &rope, positions)?;
        }
        drop(layer);
    }

    let mut window: Vec<TrainLayer> = Vec::with_capacity(grad_layers);
    for li in window_start..cfg.num_hidden_layers {
        window.push(
            TrainLayer::from_weights(&src, &cfg, li, dtype)
                .with_context(|| format!("build trainable window layer {li}"))?,
        );
    }
    let final_norm = RmsNorm::new(
        src.trunk_rmsnorm_gamma(FINAL_NORM, dtype)?,
        cfg.rms_norm_eps,
    );
    let lm_head = src.dense_linear(LM_HEAD, dtype)?;

    let head_loss = |x: &Tensor, tgt: &Tensor, mask: &[f32]| -> Result<Tensor> {
        let t = x.dims()[1];
        let x = final_norm.forward_candle(x)?;
        let logits = lm_head.forward(&x)?;
        let logits2 = logits.reshape((t, cfg.vocab_size))?.to_dtype(DType::F32)?;
        crate::ce_mean_over_unmasked_positions(&logits2, tgt, mask)
    };
    let forward_loss =
        |h: &Tensor, positions: &Tensor, tgt: &Tensor, mask: &[f32]| -> Result<Tensor> {
            let mut x = h.clone();
            for layer in &window {
                x = layer.forward(&x, &rope, positions)?;
            }
            head_loss(&x, tgt, mask)
        };
    let eval_ce_base = per_row_holdout_ce_in_loaded_dataset_row_order(
        &window,
        &final_norm,
        &lm_head,
        &rope,
        cfg.vocab_size,
        &eval_ex,
    )?;
    let eval_loss_base =
        crate::eval_loss_scalar_is_the_sum_of_the_per_row_ce_means_in_dataset_row_order(
            &eval_ce_base,
        );

    let lcfg = LoraConfig {
        r: args.rank,
        alpha: args.alpha,
        dropout: 0.0,
    };
    let (loras, vars) = attach_window_loras(&window, &args.targets, lcfg, args.seed, device)?;

    let cotrain = match co {
        None => None,
        Some(c) => {
            ensure!(
                window_start + window.len() == cfg.num_hidden_layers,
                "MTP co-training requires the trainable window to end at the final trunk layer: \
                 {}",
                MTP_COTRAIN_ACCEPTS_ANY_GRAD_LAYERS_BECAUSE_THE_WINDOW_ENDS_AT_THE_FINAL_TRUNK_LAYER_WHOSE_LIVE_OUTPUT_THE_HEAD_CONSUMES_DETACHED
            );
            let ids_refs: Vec<&[u32]> = train_rows
                .iter()
                .map(|r| &r.ids[..r.ids.len() - 1])
                .collect();
            Some(MtpCoTrainEngine::build(
                c, &cfg, &args.base, &src, &ids_refs, dtype, device, args.seed,
            )?)
        }
    };

    let checkpointed =
        window_checkpointing_with_nv_train_ckpt_as_the_named_tristate_escape_hatch(grad_layers);
    let train_tokens_per_step: usize = train_rows.iter().map(|r| r.ids.len() - 1).sum();
    eprintln!(
        "[q38-lora-train] examples {}  train_tokens_per_step {train_tokens_per_step}  \
         resident_base {resident_base}  checkpointed {checkpointed}",
        train_ex.len()
    );
    let (losses, all_grads_finite, head_losses, all_head_grads_finite) = match (&cotrain, co) {
        (Some(ct), Some(c)) => {
            let o = co_train_steps(
                &vars,
                &loras,
                &ct.vars,
                &ct.loras,
                args.steps,
                args.lr,
                args.b_lr_ratio,
                args.lr * c.speculator_lr_ratio,
                train_ex.len(),
                |i| {
                    let (h, positions, tgt, mask) = &train_ex[i];
                    let (lv, trunk_grads, window_out_detached) = if checkpointed {
                        window_loss_and_grads_checkpointed(
                            &window,
                            &final_norm,
                            &lm_head,
                            &rope,
                            positions,
                            h,
                            tgt,
                            mask,
                            cfg.vocab_size,
                            &vars,
                        )?
                    } else {
                        let dev = h.device().clone();
                        let x = crate::train_prof::phase(&dev, "forward_window_chain", || {
                            let mut x = h.clone();
                            for layer in &window {
                                x = layer.forward(&x, &rope, positions)?;
                            }
                            Ok(x)
                        })?;
                        let loss = crate::train_prof::phase(&dev, "head_loss", || {
                            head_loss(&x, tgt, mask)
                        })?;
                        let xd = x.detach();
                        let (lv, grads) = crate::train_prof::phase(&dev, "backward_all", || {
                            loss_and_grads_from_one_backward(loss, &vars)
                        })?;
                        (lv, grads, xd)
                    };
                    let head = ct.head_loss_and_grads(
                        i,
                        &window_out_detached,
                        &rope,
                        positions,
                        &lm_head,
                        tgt,
                        mask,
                        cfg.vocab_size,
                    )?;
                    Ok(((lv, trunk_grads), head))
                },
            )?;
            (
                o.losses,
                o.all_grads_finite,
                o.head_losses,
                o.all_head_grads_finite,
            )
        }
        _ => {
            #[cfg(feature = "cuda")]
            note_device_use_of_this_process_between(device_free_before_this_process_loaded);
            let groups = {
                let lengths: Vec<usize> = train_ex.iter().map(|(h, ..)| h.dims()[1]).collect();
                equal_length_row_groups(&lengths, window_rows_per_graph(device))
            };
            eprintln!(
                "[q38-lora-train] window graphs per step {}  rows per graph {:?}",
                groups.len(),
                groups.iter().map(|g| g.len()).collect::<Vec<_>>()
            );
            let (l, _step_seconds, f) = train_steps(
                &vars,
                &loras,
                args.steps,
                args.lr,
                args.b_lr_ratio,
                groups.len(),
                |gi| {
                    let group = &groups[gi];
                    if group.len() > 1 {
                        let dev = train_ex[group[0]].0.device().clone();
                        let rows = group.len();
                        let t = train_ex[group[0]].0.dims()[1];
                        let hs: Vec<&Tensor> = group.iter().map(|&i| &train_ex[i].0).collect();
                        let tg: Vec<&Tensor> = group.iter().map(|&i| &train_ex[i].2).collect();
                        let mut weights: Vec<f32> = Vec::with_capacity(rows * t);
                        for &i in group {
                            let mask = &train_ex[i].3;
                            let unmasked: f32 = mask.iter().sum::<f32>().max(1.0);
                            weights.extend(mask.iter().map(|w| w / (unmasked * rows as f32)));
                        }
                        let h = Tensor::cat(&hs, 0)?;
                        let positions = flat_positions(rows, t, &dev)?;
                        let tgt = Tensor::cat(&tg, 0)?;
                        return window_loss_and_grads_checkpointed_rows(
                            &window,
                            &final_norm,
                            &lm_head,
                            &rope,
                            &positions,
                            &h,
                            &tgt,
                            &weights,
                            cfg.vocab_size,
                            &vars,
                        )
                        .map(|(lv, grads, _window_out)| (lv, grads));
                    }
                    let i = group[0];
                    let (h, positions, tgt, mask) = &train_ex[i];
                    if checkpointed {
                        window_loss_and_grads_checkpointed(
                            &window,
                            &final_norm,
                            &lm_head,
                            &rope,
                            positions,
                            h,
                            tgt,
                            mask,
                            cfg.vocab_size,
                            &vars,
                        )
                        .map(|(lv, grads, _window_out)| (lv, grads))
                    } else {
                        let dev = h.device().clone();
                        let loss =
                            crate::train_prof::phase(&dev, "forward_window_and_head_loss", || {
                                forward_loss(h, positions, tgt, mask)
                            })?;
                        crate::train_prof::phase(&dev, "backward_all", || {
                            loss_and_grads_from_one_backward(loss, &vars)
                        })
                    }
                },
            )?;
            (l, f, Vec::new(), true)
        }
    };

    let eval_ce_adapter = per_row_holdout_ce_in_loaded_dataset_row_order(
        &window,
        &final_norm,
        &lm_head,
        &rope,
        cfg.vocab_size,
        &eval_ex,
    )?;
    let eval_loss_adapter =
        crate::eval_loss_scalar_is_the_sum_of_the_per_row_ce_means_in_dataset_row_order(
            &eval_ce_adapter,
        );
    if let Some(with_adapter) = eval_loss_adapter {
        eprintln!(
            "[q38-lora-train] eval  loss_adapter = {with_adapter:.6e}  loss_base = {:.6e}  \
             examples {}",
            eval_loss_base.unwrap_or(f32::NAN),
            eval_ex.len()
        );
    }

    if w4a4_eval_arm_opted_in() {
        let _ = W4A4_EVAL_ARM_STAYS_ALONGSIDE_QUANT_RESIDENT_TRAINING_BECAUSE_IT_SCORES_A_HOST_DEQUANT_TRAINED_ADAPTER_UNDER_THE_SERVING_FORWARD;
        ensure!(
            !eval_ex.is_empty(),
            "NV_Q38_EVAL_W4A4=1 but no --eval-data rows: nothing to score under the serving \
             W4A4 forward"
        );
        run_w4a4_eval_arm(
            args,
            &cfg,
            eval_rows.as_deref().unwrap_or(&[]),
            &eval_ex,
            &window,
            &loras,
            &final_norm,
            &lm_head,
            &rope,
            window_start,
            dtype,
            device,
            &losses,
            &eval_ce_base,
            &eval_ce_adapter,
        )?;
    }

    let (entries, _min_amax, _max_amax) = collect_raw_entries(&loras)?;
    let (speculator, head_modules) = match &cotrain {
        Some(ct) => {
            let (artifact, modules) = ct.artifact()?;
            (Some(artifact), modules)
        }
        None => (None, Vec::new()),
    };
    crate::save_peft_raw_with_speculator(
        &args.out,
        &entries,
        args.rank,
        args.alpha,
        0.0,
        &args.base.display().to_string(),
        speculator.as_ref(),
    )?;
    let co_summary = cotrain.as_ref().map(|ct| Q38MtpCoTrainSummary {
        head_losses,
        head_modules,
        head_trainable_vars: ct.vars.len(),
        all_head_grads_finite,
        speculator_kind: QWEN38_MTP_SPECULATOR_KIND.to_string(),
        speculator_dir: args.out.join(crate::SPECULATOR_SUBDIR),
    });

    let summary = Q38JsonlSummary {
        losses,
        eval_loss_adapter,
        eval_loss_base,
        eval_ce_adapter,
        eval_ce_base,
        modules: entries.iter().map(|e| e.module.clone()).collect(),
        trainable_vars: vars.len(),
        num_examples: train_ex.len(),
        num_eval_examples: eval_ex.len(),
        window_start,
        window_kinds: window.iter().map(|l| l.kind.label()).collect(),
        base_dtype: if resident_base {
            format!("{dtype:?}+quant_resident")
        } else {
            format!("{dtype:?}")
        },
        resident_base,
        all_grads_finite,
        checkpointed,
        device: if device.is_cuda() {
            "cuda".to_string()
        } else {
            "cpu".to_string()
        },
        adapter_path: args.out.join("adapter_model.safetensors"),
        config_path: args.out.join("adapter_config.json"),
    };
    Ok((summary, co_summary))
}

#[cfg(test)]
mod window_row_groups {
    use super::equal_length_row_groups;

    #[test]
    fn equal_length_rows_share_a_graph_up_to_the_cap_and_a_length_change_starts_a_new_group() {
        assert_eq!(
            equal_length_row_groups(&[512, 512, 512, 512, 512, 512, 512, 512], 8),
            vec![vec![0, 1, 2, 3, 4, 5, 6, 7]]
        );
        assert_eq!(
            equal_length_row_groups(&[512, 512, 512], 2),
            vec![vec![0, 1], vec![2]]
        );
        assert_eq!(
            equal_length_row_groups(&[512, 256, 256, 512], 8),
            vec![vec![0], vec![1, 2], vec![3]]
        );
        assert_eq!(equal_length_row_groups(&[7, 7], 1), vec![vec![0], vec![1]]);
    }
}
