use anyhow::{bail, ensure, Context, Result};
use candle_core::{DType, Device, Tensor};
use nv_layers::linear::Linear;
use nv_layers::mlp::Mlp;
use nv_layers::norm::RmsNorm;
use nv_layers::rope::{Rope, RopeConfig, RopeKind};
#[cfg(feature = "cuda")]
use nv_weights::QuantScheme;
use nv_weights::{QuantizationConfig, WeightLoader};
use serde::Deserialize;

#[cfg(feature = "cuda")]
use std::sync::{Arc, Mutex};

use crate::gemma4::{Gemma4Cache, LayerType};
#[cfg(feature = "cuda")]
use crate::laguna_fp8::LagunaKvCacheFp8;
use crate::CausalLm;
#[cfg(feature = "cuda")]
use candle_core::op::BackpropOp;
#[cfg(feature = "cuda")]
use candle_core::CudaStorage;
#[cfg(feature = "cuda")]
use candle_core::Storage;
#[cfg(feature = "cuda")]
use cudarc::driver::CudaStream;
#[cfg(feature = "cuda")]
use nv_layers::moe_grouped::MoeGroupedWeights;
use serde_json::Value;
#[cfg(feature = "cuda")]
use std::collections::HashMap;
use std::env;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Once;
#[cfg(feature = "cuda")]
use std::sync::OnceLock;
use std::time::Instant;

#[cfg(feature = "cuda")]
pub mod prefix;

#[cfg(feature = "cuda")]
type Nvfp4RunnerHandle = Option<Arc<Mutex<nv_quant::nvfp4::Nvfp4GemmRunner>>>;
#[cfg(not(feature = "cuda"))]
type Nvfp4RunnerHandle = Option<()>;

#[cfg(feature = "cuda")]
pub(crate) type Fp8RunnerHandle = Option<Arc<Mutex<nv_quant::fp8::Fp8GemmRunner>>>;
#[cfg(not(feature = "cuda"))]
pub(crate) type Fp8RunnerHandle = Option<()>;

#[cfg(feature = "cuda")]
struct LagunaFp8Embedding {
    bytes: cudarc::driver::CudaSlice<u8>,
    row_scales: cudarc::driver::CudaSlice<f32>,
    vocab: usize,
    hidden: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LagunaGating {
    None,
    PerHead,
    PerElement,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MlpLayerType {
    Dense,
    Sparse,
}

#[derive(Clone, Debug, Deserialize)]
pub struct LagunaRopeParams {
    pub rope_theta: f32,
    #[serde(default = "default_partial_rotary_factor")]
    pub partial_rotary_factor: f32,
    #[serde(default)]
    pub rope_type: Option<String>,
    #[serde(default)]
    pub factor: Option<f32>,
    #[serde(default)]
    pub original_max_position_embeddings: Option<usize>,
    #[serde(default)]
    pub beta_fast: Option<f32>,
    #[serde(default)]
    pub beta_slow: Option<f32>,
    #[serde(default)]
    pub attention_factor: Option<f32>,
}

fn default_partial_rotary_factor() -> f32 {
    1.0
}

#[derive(Clone, Debug, Deserialize)]
struct LagunaRopeParameters {
    full_attention: LagunaRopeParams,
    sliding_attention: LagunaRopeParams,
}

#[derive(Clone, Debug, Deserialize)]
pub struct LagunaConfig {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub max_position_embeddings: usize,
    pub rms_norm_eps: f64,
    pub sliding_window: usize,
    pub layer_types: Vec<LayerType>,
    #[serde(default)]
    pub mlp_layer_types: Option<Vec<MlpLayerType>>,
    #[serde(default)]
    pub num_attention_heads_per_layer: Option<Vec<usize>>,
    #[serde(default)]
    pub mlp_only_layers: Vec<usize>,
    #[serde(default = "default_sparse_step")]
    pub decoder_sparse_step: usize,
    pub num_experts: usize,
    pub num_experts_per_tok: usize,
    pub moe_intermediate_size: usize,
    pub shared_expert_intermediate_size: usize,
    #[serde(default)]
    pub norm_topk_prob: bool,
    #[serde(default = "default_one_f32")]
    pub moe_routed_scaling_factor: f32,
    #[serde(default)]
    pub moe_router_logit_softcapping: f32,
    #[serde(default)]
    pub tie_word_embeddings: bool,
    #[serde(default = "default_gating", deserialize_with = "de_gating")]
    pub gating: LagunaGating,
    rope_parameters: LagunaRopeParameters,
    #[serde(default, deserialize_with = "de_token_ids")]
    pub eos_token_id: Vec<u32>,
}

fn default_sparse_step() -> usize {
    1
}

fn default_one_f32() -> f32 {
    1.0
}

fn default_gating() -> LagunaGating {
    LagunaGating::PerHead
}

fn de_gating<'de, D>(deserializer: D) -> std::result::Result<LagunaGating, D::Error>
where
    D: serde::Deserializer<'de>,
{
    let v = Value::deserialize(deserializer)?;
    match v {
        Value::Bool(false) => Ok(LagunaGating::None),
        Value::Bool(true) => Ok(LagunaGating::PerElement),
        Value::String(s) => match s.as_str() {
            "per-head" | "per_head" => Ok(LagunaGating::PerHead),
            "per-element" | "per_element" => Ok(LagunaGating::PerElement),
            other => Err(serde::de::Error::custom(format!(
                "unknown gating mode {other:?}"
            ))),
        },
        Value::Null => Ok(LagunaGating::None),
        other => Err(serde::de::Error::custom(format!(
            "unsupported gating value {other}"
        ))),
    }
}

fn de_token_ids<'de, D>(deserializer: D) -> std::result::Result<Vec<u32>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    let v = Option::<Value>::deserialize(deserializer)?;
    match v {
        None | Some(Value::Null) => Ok(Vec::new()),
        Some(Value::Number(n)) => Ok(vec![n.as_u64().unwrap_or(0) as u32]),
        Some(Value::Array(a)) => Ok(a
            .into_iter()
            .filter_map(|x| x.as_u64())
            .map(|x| x as u32)
            .collect()),
        Some(other) => Err(serde::de::Error::custom(format!(
            "unsupported eos_token_id {other}"
        ))),
    }
}

impl LagunaConfig {
    nv_weights::hf_json_from_file!(from_hf_json_file, from_hf_json_str);

    pub fn from_hf_json_str(s: &str) -> Result<Self> {
        let cfg: LagunaConfig = serde_json::from_str(s).context("deserialize laguna config")?;
        if cfg.layer_types.len() != cfg.num_hidden_layers {
            bail!(
                "laguna: layer_types len {} != num_hidden_layers {}",
                cfg.layer_types.len(),
                cfg.num_hidden_layers
            );
        }
        if let Some(heads) = &cfg.num_attention_heads_per_layer {
            if heads.len() != cfg.num_hidden_layers {
                bail!(
                    "laguna: num_attention_heads_per_layer len {} != num_hidden_layers {}",
                    heads.len(),
                    cfg.num_hidden_layers
                );
            }
            for (i, &h) in heads.iter().enumerate() {
                if h == 0 || h % cfg.num_key_value_heads != 0 {
                    bail!(
                        "laguna: layer {i} head count {h} not divisible by kv heads {}",
                        cfg.num_key_value_heads
                    );
                }
            }
        }
        if let Some(mlp_types) = &cfg.mlp_layer_types {
            if mlp_types.len() != cfg.num_hidden_layers {
                bail!(
                    "laguna: mlp_layer_types len {} != num_hidden_layers {}",
                    mlp_types.len(),
                    cfg.num_hidden_layers
                );
            }
        }
        Ok(cfg)
    }

    pub fn layer_kind(&self, idx: usize) -> LayerType {
        self.layer_types[idx]
    }

    pub fn num_heads_for_layer(&self, idx: usize) -> usize {
        self.num_attention_heads_per_layer
            .as_ref()
            .map(|v| v[idx])
            .unwrap_or(self.num_attention_heads)
    }

    pub fn is_moe_layer(&self, idx: usize) -> bool {
        if let Some(types) = &self.mlp_layer_types {
            return types[idx] == MlpLayerType::Sparse;
        }
        if self.mlp_only_layers.contains(&idx) {
            return false;
        }
        self.num_experts > 0 && (idx + 1).is_multiple_of(self.decoder_sparse_step.max(1))
    }

    pub fn rotary_dim_full(&self) -> usize {
        let partial = self.rope_parameters.full_attention.partial_rotary_factor;
        (((self.head_dim as f32) * partial) as usize).clamp(2, self.head_dim)
    }

    pub fn rotary_dim_sliding(&self) -> usize {
        let partial = self.rope_parameters.sliding_attention.partial_rotary_factor;
        (((self.head_dim as f32) * partial) as usize).clamp(2, self.head_dim)
    }

    pub fn full_rope_params(&self) -> &LagunaRopeParams {
        &self.rope_parameters.full_attention
    }

    pub fn sliding_rope_params(&self) -> &LagunaRopeParams {
        &self.rope_parameters.sliding_attention
    }

    pub fn full_attention_factor(&self) -> f32 {
        self.rope_parameters
            .full_attention
            .attention_factor
            .unwrap_or(1.0)
    }
}

pub fn yarn_inv_freq(dim: usize, params: &LagunaRopeParams) -> Vec<f32> {
    let base = params.rope_theta as f64;
    let half = dim / 2;
    let default: Vec<f32> = (0..half)
        .map(|i| (1.0 / base.powf((i as f64 * 2.0) / dim as f64)) as f32)
        .collect();
    if params.rope_type.as_deref() != Some("yarn") {
        return default;
    }
    let factor = params.factor.unwrap_or(1.0) as f64;
    let orig = params.original_max_position_embeddings.unwrap_or(0) as f64;
    if factor <= 1.0 || orig <= 0.0 {
        return default;
    }
    let beta_fast = params.beta_fast.unwrap_or(32.0) as f64;
    let beta_slow = params.beta_slow.unwrap_or(1.0) as f64;
    let find_correction_dim = |num_rot: f64| -> f64 {
        (dim as f64) * (orig / (num_rot * 2.0 * std::f64::consts::PI)).ln() / (2.0 * base.ln())
    };
    let mut low = find_correction_dim(beta_fast).floor();
    let mut high = find_correction_dim(beta_slow).ceil();
    low = low.max(0.0);
    high = high.min(dim as f64 - 1.0);
    if (high - low).abs() < f64::EPSILON {
        high += 0.001;
    }
    (0..half)
        .map(|i| {
            let pos_freq = base.powf((i as f64 * 2.0) / dim as f64);
            let extrap = 1.0 / pos_freq;
            let interp = 1.0 / (factor * pos_freq);
            let ramp = (((i as f64) - low) / (high - low)).clamp(0.0, 1.0);
            let extrap_factor = 1.0 - ramp;
            (interp * (1.0 - extrap_factor) + extrap * extrap_factor) as f32
        })
        .collect()
}

pub struct LagunaAttention {
    pub kind: LayerType,
    pub num_heads: usize,
    pub q_proj: Linear,
    pub k_proj: Linear,
    pub v_proj: Linear,
    pub o_proj: Linear,
    pub g_proj: Option<Linear>,
    pub q_norm: RmsNorm,
    pub k_norm: RmsNorm,

    #[cfg(feature = "cuda")]
    pub(crate) w8: Option<LagunaAttnW8>,
}

#[cfg(feature = "cuda")]
pub(crate) enum LagunaProjBytes {
    I8(cudarc::driver::CudaSlice<i8>),
    E4m3(cudarc::driver::CudaSlice<u8>),
}

#[cfg(feature = "cuda")]
pub(crate) struct LagunaProjW8 {
    wq: LagunaProjBytes,
    row_scale: cudarc::driver::CudaSlice<f32>,
    n: usize,
    k: usize,
    pub(crate) max_m: usize,
    marlin_fp8_twin_built_at_load_when_the_rows_route_is_on:
        Option<std::sync::Arc<nv_layers::linear::MarlinFp8Prepared>>,
}

#[cfg(feature = "cuda")]
const MARLIN_FP8_ROWS_ROUTE_MIN_ROWS_2_LEAVES_THE_TUNED_M1_PROJECTION_KERNELS_ALONE: usize = 2;
#[cfg(feature = "cuda")]
const MARLIN_FP8_ROWS_ROUTE_MAX_ROWS_16_IS_THE_INSTANTIATED_THREAD_M_BLOCKS_1: usize = 16;

#[cfg(feature = "cuda")]
impl LagunaProjW8 {
    pub(crate) fn bytes(&self) -> &LagunaProjBytes {
        &self.wq
    }

    pub(crate) fn row_scale(&self) -> &cudarc::driver::CudaSlice<f32> {
        &self.row_scale
    }
}

#[cfg(feature = "cuda")]
pub(crate) struct LagunaAttnW8 {
    pub(crate) q: LagunaProjW8,
    pub(crate) k: LagunaProjW8,
    pub(crate) v: LagunaProjW8,
    pub(crate) o: LagunaProjW8,
    ones_wn: Tensor,
    ones_rstd: Tensor,
    fp8: bool,
    cfg_scope: u8,
    scope: std::sync::atomic::AtomicU8,
    qkv_marlin_twin_one_launch_for_the_three_projections:
        Option<std::sync::Arc<nv_layers::linear::MarlinFp8Prepared>>,
    gate_rows_folded_into_the_qkv_twin_zero_when_the_gate_goes_through_its_linear: usize,
}

#[cfg(feature = "cuda")]
impl LagunaAttnW8 {
    fn quantize_proj(w: &Tensor, device: &Device, fp8: bool) -> Result<Option<LagunaProjW8>> {
        let _ = device;
        let (n, k) = w.dims2()?;
        if k % 16 != 0 || w.dtype() != DType::BF16 {
            return Ok(None);
        }
        let max_m = nv_kernels::cuda::gemv_i8_normed_mk_max_m(k as i32).max(0) as usize;
        if max_m == 0 {
            return Ok(None);
        }
        let (wq, row_scale) = if fp8 {
            let (wq, rs) = nv_layers::lm_head_resident_quant::rowquant_e4m3_dev(w)?;
            (LagunaProjBytes::E4m3(wq), rs)
        } else {
            let (wq, rs) = nv_layers::lm_head_resident_quant::rowquant_i8_dev(w)?;
            (LagunaProjBytes::I8(wq), rs)
        };
        let marlin_fp8_twin_built_at_load_when_the_rows_route_is_on = match (&wq, device) {
            (LagunaProjBytes::E4m3(bytes), Device::Cuda(dev))
                if nv_layers::linear::marlin_fp8_rows_env_on() && n % 64 == 0 && k % 16 == 0 =>
            {
                match nv_layers::linear::marlin_fp8_twin_from_e4m3_rows_dev(
                    dev, bytes, &row_scale, n, k,
                ) {
                    Ok(t) => Some(std::sync::Arc::new(t)),
                    Err(e) => {
                        eprintln!("[marlin_fp8_twin] attention projection twin unavailable n={n} k={k}: {e:#}");
                        None
                    }
                }
            }
            _ => None,
        };
        Ok(Some(LagunaProjW8 {
            wq,
            row_scale,
            n,
            k,
            max_m: max_m.min(MAX_VERIFY_MOE_TOKENS),
            marlin_fp8_twin_built_at_load_when_the_rows_route_is_on,
        }))
    }

    pub(crate) fn build(
        q_proj: &Linear,
        k_proj: &Linear,
        v_proj: &Linear,
        o_proj: &Linear,
        device: &Device,
        fp8: bool,
    ) -> Result<Option<Self>> {
        let (Some(q), Some(k), Some(v), Some(o)) = (
            q_proj.weight(),
            k_proj.weight(),
            v_proj.weight(),
            o_proj.weight(),
        ) else {
            return Ok(None);
        };
        Self::build_from_weights(q, k, v, o, None, device, fp8)
    }

    fn build_from_weights(
        q_weight: &Tensor,
        k_weight: &Tensor,
        v_weight: &Tensor,
        o_weight: &Tensor,
        per_head_gate_weight_for_the_fused_twin: Option<&Tensor>,
        device: &Device,
        fp8: bool,
    ) -> Result<Option<Self>> {
        let (q, k, v, o) = match (
            Self::quantize_proj(q_weight, device, fp8)?,
            Self::quantize_proj(k_weight, device, fp8)?,
            Self::quantize_proj(v_weight, device, fp8)?,
            Self::quantize_proj(o_weight, device, fp8)?,
        ) {
            (Some(q), Some(k), Some(v), Some(o)) => (q, k, v, o),
            _ => return Ok(None),
        };
        let cfg_scope = attn_w8_spec_env().scope;
        let max_k = q.k.max(k.k).max(v.k).max(o.k);
        let ones_wn = Tensor::ones(max_k, DType::BF16, device)?.contiguous()?;
        let ones_rstd = Tensor::ones(MAX_VERIFY_MOE_TOKENS, DType::F32, device)?.contiguous()?;
        let gate_w8 = match per_head_gate_weight_for_the_fused_twin {
            Some(g) if fp8 && g.dims2().map(|(_, gk)| gk == q.k).unwrap_or(false) => {
                Self::quantize_proj(g, device, true)?
            }
            _ => None,
        };
        let gate_rows = gate_w8.as_ref().map(|g| g.n).unwrap_or(0);
        let n_rows_unpadded = q.n + k.n + v.n + gate_rows;
        let n_rows_padded_to_marlin_tile = n_rows_unpadded.div_ceil(64) * 64;
        let qkv_marlin_twin_one_launch_for_the_three_projections = match device {
            Device::Cuda(dev)
                if nv_layers::linear::marlin_fp8_rows_env_on()
                    && q.k == k.k
                    && q.k == v.k
                    && q.k % 16 == 0 =>
            {
                let stream = dev.cuda_stream();
                let mut bytes: Vec<u8> = Vec::with_capacity(n_rows_padded_to_marlin_tile * q.k);
                let mut scales: Vec<f32> = Vec::with_capacity(n_rows_padded_to_marlin_tile);
                let mut ok = true;
                for p in [Some(&q), Some(&k), Some(&v), gate_w8.as_ref()]
                    .into_iter()
                    .flatten()
                {
                    match &p.wq {
                        LagunaProjBytes::E4m3(b) => {
                            bytes.extend_from_slice(&stream.clone_dtoh(b)?);
                            scales.extend_from_slice(&stream.clone_dtoh(&p.row_scale)?);
                        }
                        LagunaProjBytes::I8(_) => ok = false,
                    }
                }
                bytes.resize(n_rows_padded_to_marlin_tile * q.k, 0);
                scales.resize(n_rows_padded_to_marlin_tile, 0.0);
                if ok {
                    match nv_layers::linear::marlin_fp8_twin_from_e4m3_rows_host(
                        dev,
                        &bytes,
                        &scales,
                        n_rows_padded_to_marlin_tile,
                        q.k,
                    ) {
                        Ok(t) => Some(std::sync::Arc::new(t)),
                        Err(e) => {
                            eprintln!("[marlin_fp8_twin] fused qkv twin unavailable: {e:#}");
                            None
                        }
                    }
                } else {
                    None
                }
            }
            _ => None,
        };
        let gate_rows_folded_into_the_qkv_twin_zero_when_the_gate_goes_through_its_linear =
            if qkv_marlin_twin_one_launch_for_the_three_projections.is_some() {
                gate_rows
            } else {
                0
            };
        Ok(Some(Self {
            q,
            k,
            v,
            o,
            ones_wn,
            ones_rstd,
            fp8,
            cfg_scope,
            scope: std::sync::atomic::AtomicU8::new(cfg_scope),
            qkv_marlin_twin_one_launch_for_the_three_projections,
            gate_rows_folded_into_the_qkv_twin_zero_when_the_gate_goes_through_its_linear,
        }))
    }

    pub(crate) fn qkv_marlin_rows(
        &self,
        x: &Tensor,
        m: usize,
    ) -> Result<Option<(Tensor, Tensor, Tensor, Option<Tensor>)>> {
        let Some(tw) = self
            .qkv_marlin_twin_one_launch_for_the_three_projections
            .as_ref()
        else {
            return Ok(None);
        };
        let gate_rows =
            self.gate_rows_folded_into_the_qkv_twin_zero_when_the_gate_goes_through_its_linear;
        if !(MARLIN_FP8_ROWS_ROUTE_MIN_ROWS_2_LEAVES_THE_TUNED_M1_PROJECTION_KERNELS_ALONE
            ..=MARLIN_FP8_ROWS_ROUTE_MAX_ROWS_16_IS_THE_INSTANTIATED_THREAD_M_BLOCKS_1)
            .contains(&m)
            || !nv_layers::linear::marlin_fp8_rows_env_on()
            || x.dtype() != DType::BF16
            || x.elem_count() != m * self.q.k
        {
            return Ok(None);
        }
        let Device::Cuda(dev) = x.device().clone() else {
            return Ok(None);
        };
        use cudarc::driver::DevicePtr;
        let stream = nv_layers::cuda_stream::current_stream(&dev);
        let x_c = x.contiguous()?;
        let n_all = tw.n;
        let mut out: cudarc::driver::CudaSlice<half::bf16> =
            unsafe { stream.alloc::<half::bf16>(m * n_all)? };
        {
            let (xs, xl) = x_c.storage_and_layout();
            let x_cuda = match &*xs {
                Storage::Cuda(st) => st,
                _ => bail!("w8 marlin qkv rows: x must be CUDA"),
            };
            let x_view = x_cuda
                .as_cuda_slice::<half::bf16>()?
                .slice(xl.start_offset()..);
            let (xp, _gx) = x_view.device_ptr(&stream);
            nv_layers::linear::marlin_fp8_rows_gemm_into(&stream, tw, xp, m, &mut out)?;
        }
        static W8_MARLIN_QKV_FIRED_ONCE: Once = Once::new();
        W8_MARLIN_QKV_FIRED_ONCE.call_once(|| eprintln!("[laguna_w8_marlin_rows] fused qkv fired m={m} n={n_all} k={} gate_rows={gate_rows} (one Marlin launch for q|k|v|g)", self.q.k));
        let mut qo: cudarc::driver::CudaSlice<half::bf16> =
            unsafe { stream.alloc::<half::bf16>(m * self.q.n)? };
        let mut ko: cudarc::driver::CudaSlice<half::bf16> =
            unsafe { stream.alloc::<half::bf16>(m * self.k.n)? };
        let mut vo: cudarc::driver::CudaSlice<half::bf16> =
            unsafe { stream.alloc::<half::bf16>(m * self.v.n)? };
        let mut go: cudarc::driver::CudaSlice<half::bf16> =
            unsafe { stream.alloc::<half::bf16>(m * gate_rows.max(1))? };
        {
            use cudarc::driver::DevicePtrMut;
            let (sp, _g0) = out.device_ptr(&stream);
            let (qp, _g1) = qo.device_ptr_mut(&stream);
            let (kp, _g2) = ko.device_ptr_mut(&stream);
            let (vp, _g3) = vo.device_ptr_mut(&stream);
            let (gp, _g4) = go.device_ptr_mut(&stream);
            let rc = unsafe {
                nv_kernels::cuda::split_rows_bf16_four(
                    stream.cu_stream() as *mut std::ffi::c_void,
                    sp as *const u16,
                    qp as *mut u16,
                    kp as *mut u16,
                    vp as *mut u16,
                    if gate_rows > 0 {
                        gp as *mut u16
                    } else {
                        std::ptr::null_mut()
                    },
                    m as i32,
                    self.q.n as i32,
                    self.k.n as i32,
                    self.v.n as i32,
                    gate_rows as i32,
                    n_all as i32,
                )
            };
            ensure!(rc == 0, "w8 marlin qkv split rc={rc}");
        }
        let wrap = |s: cudarc::driver::CudaSlice<half::bf16>, n: usize| -> Tensor {
            let storage = CudaStorage::wrap_cuda_slice(s, dev.clone());
            Tensor::from_storage(Storage::Cuda(storage), (m, n), BackpropOp::none(), false)
        };
        let (q, k, v) = (wrap(qo, self.q.n), wrap(ko, self.k.n), wrap(vo, self.v.n));
        let g = (gate_rows > 0).then(|| wrap(go, gate_rows));
        drop(out);
        Ok(Some((q, k, v, g)))
    }

    pub(crate) fn scope_now(&self) -> u8 {
        self.scope.load(Ordering::Relaxed)
    }

    pub(crate) fn slot_on(&self, bit: u8) -> bool {
        self.scope_now() & bit != 0
    }

    pub(crate) fn cfg_slot_on(&self, bit: u8) -> bool {
        self.cfg_scope & bit != 0
    }

    pub(crate) fn set_scope(&self, on: bool) {
        let v = if on { self.cfg_scope } else { 0 };
        self.scope.store(v, Ordering::Relaxed);
    }

    pub(crate) fn is_fp8(&self) -> bool {
        self.fp8
    }

    pub(crate) fn forward(&self, p: &LagunaProjW8, x: &Tensor, m: usize) -> Result<Option<Tensor>> {
        let Device::Cuda(dev) = x.device().clone() else {
            return Ok(None);
        };
        if x.dtype() != DType::BF16 || x.elem_count() != m * p.k {
            return Ok(None);
        }
        if let Some(tw) = p
            .marlin_fp8_twin_built_at_load_when_the_rows_route_is_on
            .as_ref()
        {
            if (MARLIN_FP8_ROWS_ROUTE_MIN_ROWS_2_LEAVES_THE_TUNED_M1_PROJECTION_KERNELS_ALONE
                ..=MARLIN_FP8_ROWS_ROUTE_MAX_ROWS_16_IS_THE_INSTANTIATED_THREAD_M_BLOCKS_1)
                .contains(&m)
                && nv_layers::linear::marlin_fp8_rows_env_on()
            {
                use cudarc::driver::DevicePtr;
                let stream = nv_layers::cuda_stream::current_stream(&dev);
                let x_c = x.contiguous()?;
                let mut out: cudarc::driver::CudaSlice<half::bf16> =
                    unsafe { stream.alloc::<half::bf16>(m * p.n)? };
                {
                    let (xs, xl) = x_c.storage_and_layout();
                    let x_cuda = match &*xs {
                        Storage::Cuda(st) => st,
                        _ => bail!("w8 marlin rows: x must be CUDA"),
                    };
                    let x_view = x_cuda
                        .as_cuda_slice::<half::bf16>()?
                        .slice(xl.start_offset()..);
                    let (xp, _gx) = x_view.device_ptr(&stream);
                    nv_layers::linear::marlin_fp8_rows_gemm_into(&stream, tw, xp, m, &mut out)?;
                }
                static W8_MARLIN_ROWS_FIRED_ONCE: Once = Once::new();
                W8_MARLIN_ROWS_FIRED_ONCE.call_once(|| eprintln!("[laguna_w8_marlin_rows] fired m={m} n={} k={} (attention projection rows on the fp8 Marlin twin)", p.n, p.k));
                let storage = CudaStorage::wrap_cuda_slice(out, dev);
                return Ok(Some(Tensor::from_storage(
                    Storage::Cuda(storage),
                    (1usize, m, p.n),
                    BackpropOp::none(),
                    false,
                )));
            }
        }
        let wn = self.ones_wn.narrow(0, 0, p.k)?;
        let rstd = self.ones_rstd.narrow(0, 0, m)?;
        let y = match &p.wq {
            LagunaProjBytes::I8(wq) => nv_layers::lm_head_resident_quant::i8_mk_h_op(
                wq,
                &p.row_scale,
                x,
                &wn,
                &rstd,
                m,
                &dev,
            )?,
            LagunaProjBytes::E4m3(wq) => nv_layers::lm_head_resident_quant::e4m3_mk_h_op(
                wq,
                &p.row_scale,
                x,
                &wn,
                &rstd,
                m,
                &dev,
            )?,
        };
        Ok(Some(y.reshape((1usize, m, p.n))?))
    }
}

impl LagunaAttention {
    #[cfg(feature = "cuda")]
    pub(crate) fn fp8_gate_weight_and_scales(
        &self,
        hidden: usize,
        n_q: usize,
    ) -> Option<(
        &cudarc::driver::CudaSlice<u8>,
        &cudarc::driver::CudaSlice<f32>,
    )> {
        let gate = self.g_proj.as_ref()?;
        if gate.in_features() != hidden || gate.out_features() != n_q {
            return None;
        }
        let (weights, scales) =
            gate.fp8_e4m3_row_weight_and_scales_so_gdn_prenorm_folds_into_gemv_e4m3_mk_h()?;
        (weights.len() == hidden.checked_mul(n_q)? && scales.len() == n_q)
            .then_some((weights, scales))
    }

    #[cfg(feature = "cuda")]
    pub fn m1_w8_active_qkv(&self) -> bool {
        self.w8
            .as_ref()
            .map(|w| w.scope_now() & (ATTN_W8_Q | ATTN_W8_K | ATTN_W8_V))
            .unwrap_or(0)
            != 0
    }

    #[cfg(feature = "cuda")]
    pub fn m1_w8_active_o(&self) -> bool {
        self.w8.as_ref().is_some_and(|w| w.slot_on(ATTN_W8_O))
    }

    #[cfg(feature = "cuda")]
    pub(crate) fn m1_qkv_w8_fused(
        &self,
        hidden: usize,
        n_q: usize,
        n_kv: usize,
        hd: usize,
    ) -> Option<(&LagunaProjW8, &LagunaProjW8, &LagunaProjW8, bool)> {
        let w8 = self.w8.as_ref()?;
        let scope = w8.scope_now();
        if scope & (ATTN_W8_Q | ATTN_W8_K | ATTN_W8_V) != (ATTN_W8_Q | ATTN_W8_K | ATTN_W8_V) {
            return None;
        }
        if !hidden.is_multiple_of(16) {
            return None;
        }
        let shaped = |p: &LagunaProjW8, n: usize| p.k == hidden && p.n == n && p.max_m >= 1;
        if !shaped(&w8.q, n_q * hd) || !shaped(&w8.k, n_kv * hd) || !shaped(&w8.v, n_kv * hd) {
            return None;
        }
        Some((&w8.q, &w8.k, &w8.v, w8.is_fp8()))
    }

    #[cfg(feature = "cuda")]
    fn w8_try(
        &self,
        pick: fn(&LagunaAttnW8) -> &LagunaProjW8,
        bit: u8,
        x: &Tensor,
        m: usize,
    ) -> Result<Option<Tensor>> {
        if let Some(w8) = &self.w8 {
            if w8.slot_on(bit) {
                let p = pick(w8);
                if m >= 1 && m <= p.max_m {
                    return w8.forward(p, x, m);
                }
            }
        }
        Ok(None)
    }

    #[cfg(feature = "cuda")]
    pub fn proj_qkv_fused_rows(
        &self,
        x: &Tensor,
        m: usize,
    ) -> Result<Option<(Tensor, Tensor, Tensor, Option<Tensor>)>> {
        match &self.w8 {
            Some(w8) if w8.slot_on(ATTN_W8_Q) && w8.slot_on(ATTN_W8_K) && w8.slot_on(ATTN_W8_V) => {
                w8.qkv_marlin_rows(x, m)
            }
            _ => Ok(None),
        }
    }

    pub fn proj_q(&self, x: &Tensor, m: usize) -> Result<Tensor> {
        #[cfg(feature = "cuda")]
        if let Some(y) = self.w8_try(|w| &w.q, ATTN_W8_Q, x, m)? {
            return Ok(y);
        }
        let _ = m;
        self.q_proj.forward(x)
    }

    pub fn proj_k(&self, x: &Tensor, m: usize) -> Result<Tensor> {
        #[cfg(feature = "cuda")]
        if let Some(y) = self.w8_try(|w| &w.k, ATTN_W8_K, x, m)? {
            return Ok(y);
        }
        let _ = m;
        self.k_proj.forward(x)
    }

    pub fn proj_v(&self, x: &Tensor, m: usize) -> Result<Tensor> {
        #[cfg(feature = "cuda")]
        if let Some(y) = self.w8_try(|w| &w.v, ATTN_W8_V, x, m)? {
            return Ok(y);
        }
        let _ = m;
        self.v_proj.forward(x)
    }

    pub fn proj_o(&self, x: &Tensor, m: usize) -> Result<Tensor> {
        #[cfg(feature = "cuda")]
        if let Some(y) = self.w8_try(|w| &w.o, ATTN_W8_O, x, m)? {
            return Ok(y);
        }
        let _ = m;
        self.o_proj.forward(x)
    }
}

pub struct LagunaMoe {
    pub num_experts: usize,
    pub top_k: usize,
    pub norm_topk: bool,
    pub routed_scaling: f32,
    pub softcap: f32,
    pub gate: Linear,
    pub selection_bias: Tensor,
    pub experts: Vec<Mlp>,
    pub shared_expert: Mlp,
    #[cfg(feature = "cuda")]
    pub grouped: std::sync::Mutex<Option<Option<Arc<MoeGroupedWeights>>>>,
    #[cfg(feature = "cuda")]
    pub train_stacks: std::sync::Mutex<Option<Arc<nv_layers::moe_train::MarlinMoeTrainStacks>>>,
}

impl LagunaMoe {
    pub fn route_host(&self, x_flat: &Tensor, n_tokens: usize) -> Result<(Vec<u32>, Vec<f32>)> {
        let mut logits = self
            .gate
            .forward(x_flat)?
            .to_dtype(DType::F32)?
            .contiguous()?;
        if self.softcap > 0.0 {
            let scaled = logits.affine(1.0 / self.softcap as f64, 0.0)?;
            logits = scaled.tanh()?.affine(self.softcap as f64, 0.0)?;
        }
        let scores = candle_nn::ops::sigmoid(&logits)?;
        self.topk_host_matching_device_kernel_ties_break_to_lower_expert_index(&scores, n_tokens)
    }

    fn topk_host_matching_device_kernel_ties_break_to_lower_expert_index(
        &self,
        scores: &Tensor,
        n_tokens: usize,
    ) -> Result<(Vec<u32>, Vec<f32>)> {
        let e = self.num_experts;
        let k = self.top_k;
        let scores_host: Vec<f32> = scores.flatten_all()?.to_vec1()?;
        ensure!(
            scores_host.len() == n_tokens * e,
            "route_host scores len {} != {n_tokens}x{e}",
            scores_host.len()
        );
        let bias_host: Vec<f32> = self
            .selection_bias
            .to_dtype(DType::F32)?
            .flatten_all()?
            .to_vec1()?;
        let mut ids = Vec::with_capacity(n_tokens * k);
        let mut weights = Vec::with_capacity(n_tokens * k);
        let mut sel_row = vec![0f32; e];
        for t in 0..n_tokens {
            let s_row = &scores_host[t * e..(t + 1) * e];
            for i in 0..e {
                sel_row[i] = s_row[i] + bias_host[i];
            }
            let base = weights.len();
            for _ in 0..k {
                let mut best = f32::NEG_INFINITY;
                let mut best_i = usize::MAX;
                for (i, &v) in sel_row.iter().enumerate() {
                    if v > best {
                        best = v;
                        best_i = i;
                    }
                }
                ids.push(best_i as u32);
                weights.push(s_row[best_i]);
                sel_row[best_i] = f32::NEG_INFINITY;
            }
            if self.norm_topk {
                let mut sum = 0f32;
                for w in &weights[base..] {
                    sum += *w;
                }
                if sum > 0.0 {
                    for w in &mut weights[base..] {
                        *w /= sum;
                    }
                }
            }
        }
        Ok((ids, weights))
    }

    pub fn forward(&self, x_flat: &Tensor) -> Result<Tensor> {
        ensure!(
            self.num_experts > 0 && self.experts.len() == self.num_experts,
            "LagunaMoe.forward: individual routed experts unavailable; grouped-only loading forbids ungrouped fallback"
        );
        let dims = x_flat.dims();
        if dims.len() != 2 {
            bail!("LagunaMoe.forward: expected [tokens, hidden], got {dims:?}");
        }
        let n_tokens = dims[0];
        let hidden = dims[1];
        let device = x_flat.device().clone();

        let (top_idx_host, top_weights_host) = self.route_host(x_flat, n_tokens)?;

        let mut expert_rows: Vec<Vec<u32>> = vec![Vec::new(); self.num_experts];
        let mut expert_w: Vec<Vec<f32>> = vec![Vec::new(); self.num_experts];
        for n in 0..n_tokens {
            for j in 0..self.top_k {
                let e = top_idx_host[n * self.top_k + j] as usize;
                expert_rows[e].push(n as u32);
                expert_w[e].push(top_weights_host[n * self.top_k + j]);
            }
        }

        let mut acc = Tensor::zeros((n_tokens, hidden), DType::F32, &device)?;
        for e in 0..self.num_experts {
            let rows = &expert_rows[e];
            if rows.is_empty() {
                continue;
            }
            let m = rows.len();
            let idx_t = Tensor::from_vec(rows.clone(), m, &device)?;
            let gathered = x_flat.index_select(&idx_t, 0)?.contiguous()?;
            let y_e = self.experts[e].forward(&gathered)?.to_dtype(DType::F32)?;
            let w_t = Tensor::from_vec(expert_w[e].clone(), (m, 1), &device)?;
            let weighted = y_e.broadcast_mul(&w_t)?;
            acc = acc.index_add(&idx_t, &weighted, 0)?;
        }
        if self.routed_scaling != 1.0 {
            acc = acc.affine(self.routed_scaling as f64, 0.0)?;
        }

        let shared = self.shared_expert.forward(x_flat)?.to_dtype(DType::F32)?;
        acc.add(&shared).map_err(Into::into)
    }
}

pub enum LagunaFfn {
    Dense(Mlp),
    Moe(LagunaMoe),
}

pub struct LagunaLayer {
    pub kind: LayerType,
    pub input_layernorm: RmsNorm,
    pub post_attention_layernorm: RmsNorm,
    pub self_attn: LagunaAttention,
    pub ffn: LagunaFfn,
}

pub struct Laguna {
    config: LagunaConfig,
    embed_weight: Option<Tensor>,
    #[cfg(feature = "cuda")]
    embed_fp8: Option<LagunaFp8Embedding>,
    layers: Vec<LagunaLayer>,
    final_norm: RmsNorm,
    lm_head: Linear,
    sliding_rope: Rope,
    full_rope: Rope,
    full_attn_factor: f32,
    dtype: DType,
    device: Device,
    #[cfg(feature = "cuda")]
    moe_decode_ctx: std::sync::Mutex<Option<nv_layers::moe_grouped::GroupedDecodeContext>>,
    #[cfg(feature = "cuda")]
    moe_verify_ctx: std::sync::Mutex<HashMap<usize, nv_layers::moe_grouped::GroupedDecodeContext>>,
    #[cfg(feature = "cuda")]
    moe_graphs: std::sync::Mutex<Option<Option<crate::laguna_graph::LagunaMoeGraphs>>>,

    #[cfg(feature = "cuda")]
    moe_union_samples: std::sync::Mutex<Vec<(usize, usize)>>,
    #[cfg(feature = "cuda")]
    union_probe: AtomicBool,
    #[cfg(feature = "cuda")]
    batch_moe_fused_dispatches: std::sync::atomic::AtomicUsize,

    #[cfg(feature = "cuda")]
    lm_head_i8: Option<LagunaLmHeadI8>,
    #[cfg(feature = "cuda")]
    lm_head_fp8: Option<LagunaLmHeadFp8>,
    #[cfg(feature = "cuda")]
    lm_head_fp8_spec_off: bool,
    device_verify_routing: AtomicBool,
    attn_w8_shape: bool,
    prefill_chunk: usize,
    #[cfg(test)]
    test_moe_host_fallbacks: std::sync::atomic::AtomicUsize,
}

fn parse_prefill_chunk(value: &str) -> Result<usize> {
    let chunk: usize = value
        .parse()
        .context("NV_LAGUNA_PREFILL_CHUNK must be 256, 512, 1024, 2048, 4096 or 8192")?;
    ensure!(
        matches!(chunk, 256 | 512 | 1024 | 2048 | 4096 | 8192),
        "NV_LAGUNA_PREFILL_CHUNK must be 256, 512, 1024, 2048, 4096 or 8192, got {chunk}"
    );
    Ok(chunk)
}

const PREFILL_CHUNK_DEFAULT_1024_USER_SIGNOFF_20260912_CORPUS_INSTRUMENT_RULED_256_RESTORES_THE_REFERENCE_WIDTH: usize = 1024;

fn prefill_chunk_default_1024_user_signoff_20260912_unset_is_1024_and_256_restores_the_reference_width(
) -> Result<usize> {
    match env::var("NV_LAGUNA_PREFILL_CHUNK") {
        Ok(value) => parse_prefill_chunk(&value),
        Err(std::env::VarError::NotPresent) => Ok(PREFILL_CHUNK_DEFAULT_1024_USER_SIGNOFF_20260912_CORPUS_INSTRUMENT_RULED_256_RESTORES_THE_REFERENCE_WIDTH),
        Err(e) => Err(e).context("NV_LAGUNA_PREFILL_CHUNK must be valid UTF-8"),
    }
}

fn canonical_tail_prefill_len(remaining: usize, chunk: usize) -> usize {
    if remaining == 0 {
        return 0;
    }
    let tail = (remaining - 1) % 256 + 1;
    if remaining == tail {
        tail
    } else {
        chunk.min(remaining - tail)
    }
}

fn grouped_only_from_env() -> Result<bool> {
    match env::var("NV_LAGUNA_GROUPED_ONLY") {
        Err(std::env::VarError::NotPresent) => Ok(false),
        Ok(v) if v == "0" => Ok(false),
        Ok(v) if v == "1" => Ok(true),
        _ => bail!("NV_LAGUNA_GROUPED_ONLY must be unset, 0, or 1"),
    }
}

#[cfg(feature = "cuda")]
fn prebuild_grouped_and_release_routed_experts(
    moe: &mut LagunaMoe,
    hidden: usize,
    inter: usize,
    stream: &Arc<CudaStream>,
) -> Result<usize> {
    ensure!(
        (hidden, inter, moe.num_experts, moe.top_k) == (2048, 512, 256, 8)
            && moe.experts.len() == moe.num_experts,
        "grouped-only residency requires the Laguna XS packed 2048/512/256/8 expert layout"
    );
    let mut released_bytes = 0usize;
    for (e, mlp) in moe
        .experts
        .iter()
        .chain(std::iter::once(&moe.shared_expert))
        .enumerate()
    {
        for (linear, n, k) in [
            (mlp.gate_proj(), inter, hidden),
            (mlp.up_proj(), inter, hidden),
            (mlp.down_proj(), hidden, inter),
        ] {
            ensure!(
                (linear.out_features(), linear.in_features()) == (n, k)
                    && linear.bias().is_none()
                    && !linear.has_lora(),
                "grouped-only expert {e}: unexpected dimensions, bias, or LoRA"
            );
            let (w, scales, alpha, input_global) = linear.nvfp4_parts().with_context(|| {
                format!("grouped-only expert {e}: packed NVFP4 storage required")
            })?;
            ensure!(
                w.len() == n * k / 2
                    && scales.len() == n * (k / 16)
                    && alpha.is_finite()
                    && input_global.is_finite()
                    && input_global > 0.0,
                "grouped-only expert {e}: invalid packed payload, padded scale shape, or scalar"
            );
            if e < moe.num_experts {
                released_bytes = released_bytes
                    .checked_add(w.len() + scales.len())
                    .context("grouped-only released payload size overflow")?;
            }
        }
    }
    let slot = moe
        .grouped
        .get_mut()
        .map_err(|e| anyhow::anyhow!("grouped mutex poisoned: {e}"))?;
    ensure!(
        slot.is_none(),
        "grouped-only release must precede grouped consumers and capture"
    );
    let bank = MoeGroupedWeights::build_from_experts_folding_shared_as_a_fixed_extra_tile(
        &moe.experts,
        &moe.shared_expert,
        hidden,
        inter,
        stream,
    )?;
    ensure!(
        bank.num_experts == moe.num_experts
            && bank.hidden_size == hidden
            && bank.intermediate_size == inter
            && bank.folded_shared,
        "grouped-only prebuild returned inconsistent metadata"
    );
    stream
        .synchronize()
        .context("grouped-only copy completion before releasing originals")?;
    *slot = Some(Some(Arc::new(bank)));
    drop(std::mem::take(&mut moe.experts));
    stream
        .synchronize()
        .context("grouped-only original allocation release")?;
    Ok(released_bytes)
}

#[cfg(feature = "cuda")]
fn ck_profile_enabled() -> bool {
    nv_config::env_arm_present_at_all_reread_every_call("NV_SPEC_CK_PROFILE")
}

pub const ATTN_W8_Q: u8 = 1;
pub const ATTN_W8_K: u8 = 2;
pub const ATTN_W8_V: u8 = 4;
pub const ATTN_W8_O: u8 = 8;
pub const ATTN_W8_ALL: u8 = ATTN_W8_Q | ATTN_W8_K | ATTN_W8_V | ATTN_W8_O;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct AttnW8Spec {
    pub shape: bool,
    pub scope: u8,
}

pub fn parse_attn_w8_spec(value: &str) -> AttnW8Spec {
    let mut shape = false;
    let mut scope = 0u8;
    for part in value.split([':', ',', '+']) {
        let part = part.trim();
        if part.is_empty() {
            continue;
        }
        if part.eq_ignore_ascii_case("shape") {
            shape = true;
            continue;
        }
        let mut bits = 0u8;
        let mut ok = true;
        for c in part.chars() {
            match c.to_ascii_lowercase() {
                'q' => bits |= ATTN_W8_Q,
                'k' => bits |= ATTN_W8_K,
                'v' => bits |= ATTN_W8_V,
                'o' => bits |= ATTN_W8_O,
                _ => {
                    ok = false;
                    break;
                }
            }
        }
        if ok {
            scope |= bits;
        }
    }
    AttnW8Spec {
        shape,
        scope: if scope == 0 { ATTN_W8_ALL } else { scope },
    }
}

fn attn_w8_spec_env() -> AttnW8Spec {
    let mut shape = false;
    let mut scope = 0u8;
    for key in ["NV_LAGUNA_ATTN_W8", "NV_LAGUNA_ATTN_FP8"] {
        if let Ok(v) = env::var(key) {
            let s = parse_attn_w8_spec(&v);
            shape |= s.shape;
            scope |= s.scope;
        }
    }
    AttnW8Spec {
        shape,
        scope: if scope == 0 { ATTN_W8_ALL } else { scope },
    }
}

fn attn_w8_shape_env() -> bool {
    attn_w8_spec_env().shape
}

pub fn laguna_arm_key_explicitly_off(key: &str) -> bool {
    matches!(
        env::var(key).ok().as_deref(),
        Some("0") | Some("off") | Some("false") | Some("no")
    )
}

pub fn attn_w8_int8_arm_was_chosen_explicitly_and_therefore_outranks_the_fp8_default() -> bool {
    nv_config::env_arm_present_at_all_reread_every_call("NV_LAGUNA_ATTN_W8")
        && !laguna_arm_key_explicitly_off("NV_LAGUNA_ATTN_W8")
}

pub fn attn_quant_gate_check() -> Result<()> {
    if nv_config::env_arm_present_at_all_reread_every_call("NV_LAGUNA_ATTN_W8")
        && nv_config::env_arm_present_at_all_reread_every_call("NV_LAGUNA_ATTN_FP8")
    {
        bail!("NV_LAGUNA_ATTN_W8 and NV_LAGUNA_ATTN_FP8 are mutually exclusive; unset one");
    }
    Ok(())
}

pub fn lmhead_fp8_forced_on_for_the_spec_tier_too() -> bool {
    nv_config::env_arm_opt_in_eq_some_1_reread_every_call("NV_LAGUNA_LMHEAD_FP8")
}

pub fn lmhead_quant_gate_check() -> Result<()> {
    if nv_config::env_arm_present_at_all_reread_every_call("NV_LAGUNA_LMHEAD_INT8")
        && lmhead_fp8_forced_on_for_the_spec_tier_too()
    {
        bail!("NV_LAGUNA_LMHEAD_INT8 and NV_LAGUNA_LMHEAD_FP8 are mutually exclusive; unset one");
    }
    Ok(())
}

#[cfg(feature = "cuda")]
pub(crate) struct LagunaLmHeadI8 {
    wq: cudarc::driver::CudaSlice<i8>,
    row_scale: cudarc::driver::CudaSlice<f32>,
    max_m: usize,
}

#[cfg(feature = "cuda")]
pub(crate) struct LagunaLmHeadFp8 {
    wq: cudarc::driver::CudaSlice<u8>,
    row_scale: cudarc::driver::CudaSlice<f32>,
    max_m: usize,
}

pub const MAX_VERIFY_MOE_TOKENS: usize = 16;

impl Laguna {
    pub fn config(&self) -> &LagunaConfig {
        &self.config
    }
    pub fn dtype(&self) -> DType {
        self.dtype
    }
    pub fn device(&self) -> &Device {
        &self.device
    }
    pub fn layers(&self) -> &[LagunaLayer] {
        &self.layers
    }

    #[allow(dead_code)]
    pub(crate) fn rope_for_kind(&self, kind: LayerType) -> (&Rope, usize, f32) {
        match kind {
            LayerType::SlidingAttention => {
                (&self.sliding_rope, self.config.rotary_dim_sliding(), 1.0f32)
            }
            LayerType::FullAttention => (
                &self.full_rope,
                self.config.rotary_dim_full(),
                self.full_attn_factor,
            ),
        }
    }

    pub fn embed_weight(&self) -> &Tensor {
        self.embed_weight
            .as_ref()
            .expect("FP8 Laguna embedding has no BF16 compatibility weight")
    }

    pub(crate) fn embed_tokens(&self, tokens: &Tensor) -> Result<Tensor> {
        #[cfg(feature = "cuda")]
        if let Some(embed) = &self.embed_fp8 {
            use cudarc::driver::{DevicePtr, DevicePtrMut};
            use half::bf16;

            let Device::Cuda(dev) = self.device.clone() else {
                bail!("FP8 Laguna embedding requires CUDA")
            };
            let tokens = tokens.flatten_all()?.to_dtype(DType::U32)?.contiguous()?;
            let count = tokens.elem_count();
            ensure!(
                count <= i32::MAX as usize
                    && embed.hidden <= i32::MAX as usize
                    && embed.vocab <= i32::MAX as usize,
                "FP8 embedding shape exceeds its CUDA ABI"
            );
            let output_elems = count
                .checked_mul(embed.hidden)
                .context("FP8 embedding output size overflow")?;
            let stream = nv_layers::cuda_stream::current_stream(&dev);
            let mut out = unsafe { stream.alloc::<bf16>(output_elems)? };
            let rc = {
                let (ts, tl) = tokens.storage_and_layout();
                let tc = match &*ts {
                    Storage::Cuda(s) => s,
                    _ => bail!("FP8 Laguna embedding tokens must be CUDA"),
                };
                let token_slice = tc.as_cuda_slice::<u32>()?;
                let token_view = token_slice.slice(tl.start_offset()..);
                let (pt, _gt) = token_view.device_ptr(&stream);
                let (pw, _gw) = embed.bytes.device_ptr(&stream);
                let (ps, _gs) = embed.row_scales.device_ptr(&stream);
                let (po, _go) = out.device_ptr_mut(&stream);
                unsafe {
                    nv_kernels::cuda::gather_rows_e4m3_bf16(
                        stream.cu_stream() as *mut _,
                        pw as *const u8,
                        ps as *const f32,
                        pt as *const i32,
                        po as *mut u16,
                        count as i32,
                        embed.hidden as i32,
                        embed.vocab as i32,
                    )
                }
            };
            ensure!(rc == 0, "gather_rows_e4m3_bf16 returned {rc}");
            let storage = CudaStorage::wrap_cuda_slice(out, dev);
            return Ok(Tensor::from_storage(
                Storage::Cuda(storage),
                (count, embed.hidden),
                BackpropOp::none(),
                false,
            ));
        }
        self.embed_weight()
            .index_select(tokens, 0)
            .map_err(Into::into)
    }

    pub fn lm_head(&self) -> &Linear {
        &self.lm_head
    }

    #[cfg(feature = "cuda")]
    pub fn lm_head_i8_active(&self) -> bool {
        self.lm_head_i8.is_some()
    }

    #[cfg(feature = "cuda")]
    pub fn lm_head_fp8_active(&self) -> bool {
        self.lm_head_fp8.is_some()
    }

    pub fn attn_w8_active(&self) -> bool {
        #[cfg(feature = "cuda")]
        {
            self.layers.iter().any(|l| l.self_attn.w8.is_some())
        }
        #[cfg(not(feature = "cuda"))]
        false
    }

    pub fn set_attn_w8(&self, on: bool) {
        #[cfg(feature = "cuda")]
        for l in &self.layers {
            if let Some(w8) = &l.self_attn.w8 {
                w8.set_scope(on);
            }
        }
        #[cfg(not(feature = "cuda"))]
        let _ = on;
    }

    pub fn attn_fp8_active(&self) -> bool {
        #[cfg(feature = "cuda")]
        {
            self.layers
                .iter()
                .any(|l| l.self_attn.w8.as_ref().is_some_and(|w| w.fp8))
        }
        #[cfg(not(feature = "cuda"))]
        false
    }

    pub fn attn_w8_shape_policy(&self) -> bool {
        self.attn_w8_shape
    }

    pub fn attn_w8_scope(&self) -> u8 {
        #[cfg(feature = "cuda")]
        {
            self.layers
                .iter()
                .find_map(|l| l.self_attn.w8.as_ref().map(|w| w.scope_now()))
                .unwrap_or(0)
        }
        #[cfg(not(feature = "cuda"))]
        0
    }

    pub fn apply_attn_w8_for(&self, m: usize) {
        if self.attn_w8_shape {
            self.set_attn_w8(m == 1);
        }
    }

    pub fn lm_head_from_prenorm_scoped(&self, hidden: &Tensor, spec_tier: bool) -> Result<Tensor> {
        #[cfg(not(feature = "cuda"))]
        let _ = spec_tier;
        #[cfg(feature = "cuda")]
        if let Some(h8) = &self.lm_head_i8 {
            let t = hidden.elem_count() / self.config.hidden_size;
            if t >= 1 && t <= h8.max_m {
                return self.lm_head_i8_from_prenorm(h8, hidden, t);
            }
        }
        #[cfg(feature = "cuda")]
        if let Some(h8) = &self.lm_head_fp8 {
            if !(spec_tier && self.lm_head_fp8_spec_off) {
                let t = hidden.elem_count() / self.config.hidden_size;
                if t >= 1 && t <= h8.max_m {
                    return self.lm_head_fp8_from_prenorm(h8, hidden, t);
                }
            }
        }
        self.lm_head_bf16_from_prenorm(hidden)
    }

    pub fn lm_head_bf16_from_prenorm(&self, hidden: &Tensor) -> Result<Tensor> {
        let normed = self.final_norm.forward(hidden)?;
        self.lm_head.forward(&normed)
    }

    #[cfg(feature = "cuda")]
    fn lm_head_i8_from_prenorm(
        &self,
        h8: &LagunaLmHeadI8,
        hidden: &Tensor,
        t: usize,
    ) -> Result<Tensor> {
        let Device::Cuda(dev) = self.device.clone() else {
            bail!("int8 lm_head requires cuda")
        };
        let hflat = hidden.reshape((t, self.config.hidden_size))?;
        let rstd = crate::gemma4_e4b::rstd_op(&hflat, self.config.rms_norm_eps as f32)?;
        let raw = nv_layers::lm_head_resident_quant::lm_head_i8_normed_mk_op(
            &h8.wq,
            &h8.row_scale,
            &hflat,
            self.final_norm.weight_bf16(),
            &rstd,
            t,
            &dev,
        )?;
        raw.reshape((1usize, t, self.config.vocab_size))
            .map_err(Into::into)
    }

    #[cfg(feature = "cuda")]
    fn lm_head_fp8_from_prenorm(
        &self,
        h8: &LagunaLmHeadFp8,
        hidden: &Tensor,
        t: usize,
    ) -> Result<Tensor> {
        let Device::Cuda(dev) = self.device.clone() else {
            bail!("fp8 lm_head requires cuda")
        };
        let hflat = hidden.reshape((t, self.config.hidden_size))?;
        let rstd = crate::gemma4_e4b::rstd_op(&hflat, self.config.rms_norm_eps as f32)?;
        let raw = nv_layers::lm_head_resident_quant::e4m3_mk_h_op(
            &h8.wq,
            &h8.row_scale,
            &hflat,
            self.final_norm.weight_bf16(),
            &rstd,
            t,
            &dev,
        )?;
        raw.reshape((1usize, t, self.config.vocab_size))
            .map_err(Into::into)
    }

    pub fn from_loader(
        config: LagunaConfig,
        weights: &WeightLoader,
        device: &Device,
    ) -> Result<Self> {
        let qconfig = QuantizationConfig::none();
        Self::from_loader_quantized(config, weights, &qconfig, device)
    }

    pub fn from_loader_quantized(
        config: LagunaConfig,
        weights: &WeightLoader,
        qconfig: &QuantizationConfig,
        device: &Device,
    ) -> Result<Self> {
        attn_quant_gate_check()?;
        lmhead_quant_gate_check()?;
        let prefill_chunk = prefill_chunk_default_1024_user_signoff_20260912_unset_is_1024_and_256_restores_the_reference_width()?;
        let grouped_only = grouped_only_from_env()?;
        if grouped_only {
            ensure!(
                matches!(qconfig.scheme, nv_weights::QuantScheme::Nvfp4)
                    && (config.hidden_size, config.moe_intermediate_size, config.num_experts,
                        config.num_experts_per_tok, config.shared_expert_intermediate_size)
                        == (2048, 512, 256, 8, 512),
                "grouped-only loading requires Laguna XS NVFP4 2048/512/256/8 with shared intermediate 512"
            );
            #[cfg(feature = "cuda")]
            {
                let Device::Cuda(d) = device else {
                    bail!("grouped-only loading requires CUDA");
                };
                use cudarc::driver::sys::CUdevice_attribute as Attr;
                let stream = d.cuda_stream();
                let major = stream
                    .context()
                    .attribute(Attr::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR)?;
                let minor = stream
                    .context()
                    .attribute(Attr::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR)?;
                ensure!(
                    (major, minor) == (12, 0),
                    "grouped-only loading requires SM120, got SM{major}{minor}"
                );
            }
            #[cfg(not(feature = "cuda"))]
            bail!("grouped-only loading requires --features cuda");
        }
        let dtype = DType::BF16;

        #[cfg(feature = "cuda")]
        let nvfp4_runner: Nvfp4RunnerHandle = match qconfig.scheme {
            QuantScheme::Nvfp4 => match device {
                Device::Cuda(d) => {
                    let stream = d.cuda_stream();
                    Some(Arc::new(Mutex::new(nv_quant::nvfp4::Nvfp4GemmRunner::new(
                        stream.clone(),
                    )?)))
                }
                _ => bail!("laguna NVFP4 requires a CUDA device"),
            },
            _ => None,
        };
        #[cfg(not(feature = "cuda"))]
        let nvfp4_runner: Nvfp4RunnerHandle = {
            if !matches!(qconfig.scheme, nv_weights::QuantScheme::None) {
                bail!("laguna quantized load requires --features cuda");
            }
            None
        };

        let weight_linears_fp8 = weight_linears_fp8_enabled();
        ensure!(
            !weight_linears_fp8 || matches!(qconfig.scheme, nv_weights::QuantScheme::Nvfp4),
            "NV_LAGUNA_WEIGHT_LINEARS_FP8 requires an NVFP4 checkpoint"
        );
        #[cfg(feature = "cuda")]
        let fp8_runner: Fp8RunnerHandle = if weight_linears_fp8 {
            let Device::Cuda(d) = device else {
                bail!("NV_LAGUNA_WEIGHT_LINEARS_FP8 requires CUDA");
            };
            Some(Arc::new(Mutex::new(nv_quant::fp8::Fp8GemmRunner::new(
                d.cuda_stream().clone(),
            )?)))
        } else {
            None
        };
        #[cfg(not(feature = "cuda"))]
        let fp8_runner: Fp8RunnerHandle = {
            ensure!(
                !weight_linears_fp8,
                "NV_LAGUNA_WEIGHT_LINEARS_FP8 requires --features cuda"
            );
            None
        };

        let embed_weight = weights
            .get("model.embed_tokens.weight", dtype)
            .context("load model.embed_tokens.weight")?;
        let ed = embed_weight.dims();
        if ed.len() != 2 || ed[0] != config.vocab_size || ed[1] != config.hidden_size {
            bail!(
                "laguna embed: expected [{}, {}], got {:?}",
                config.vocab_size,
                config.hidden_size,
                ed
            );
        }

        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        #[cfg(feature = "cuda")]
        let mut residency = (0usize, 0usize, 0usize);
        for i in 0..config.num_hidden_layers {
            let layer = LagunaLayer::from_loader(
                &config,
                i,
                weights,
                qconfig,
                &nvfp4_runner,
                &fp8_runner,
                device,
                dtype,
            )?;
            #[cfg(feature = "cuda")]
            let layer = if grouped_only {
                let mut layer = layer;
                if let LagunaFfn::Moe(moe) = &mut layer.ffn {
                    let Device::Cuda(d) = device else {
                        bail!("grouped-only CUDA device unavailable");
                    };
                    let bytes = prebuild_grouped_and_release_routed_experts(
                        moe,
                        config.hidden_size,
                        config.moe_intermediate_size,
                        &d.cuda_stream(),
                    )
                    .with_context(|| {
                        format!("grouped-only prebuild/release at layer {i}; fallback disabled")
                    })?;
                    residency.0 += 1;
                    residency.1 = residency
                        .1
                        .checked_add(moe.num_experts)
                        .context("released expert count overflow")?;
                    residency.2 = residency
                        .2
                        .checked_add(bytes)
                        .context("released expert payload overflow")?;
                }
                layer
            } else {
                layer
            };
            layers.push(layer);
        }
        #[cfg(feature = "cuda")]
        if grouped_only {
            ensure!(
                residency.0 > 0,
                "grouped-only loading requires routed expert layers"
            );
            eprintln!("[laguna] grouped_only=1 grouped_layers={} released_original_experts={} released_weight_scale_payload_bytes={} retained_original_experts=0 shared_experts=retained copies_synchronized=1 release_per_layer=1", residency.0, residency.1, residency.2);
        }

        #[cfg(feature = "cuda")]
        if matches!(qconfig.scheme, QuantScheme::Nvfp4) && !grouped_only {
            if let Device::Cuda(d) = device {
                let stream = d.cuda_stream();
                for (i, layer) in layers.iter().enumerate() {
                    if let LagunaFfn::Moe(moe) = &layer.ffn {
                        let built =
                            MoeGroupedWeights::build_from_experts_folding_shared_as_a_fixed_extra_tile(
                                &moe.experts,
                                &moe.shared_expert,
                                config.hidden_size,
                                config.moe_intermediate_size,
                                &stream,
                            );
                        match built {
                            Ok(b) => {
                                *moe.grouped.lock().map_err(|e| {
                                    anyhow::anyhow!("grouped mutex poisoned: {e}")
                                })? = Some(Some(Arc::new(b)));
                            }
                            Err(e) => {
                                eprintln!(
                                    "[laguna] grouped MoE prebuild failed at layer {i}, host fallback: {e}"
                                );
                                break;
                            }
                        }
                    }
                }
            }
        }

        let final_norm = load_laguna_rmsnorm(
            weights,
            "model.norm.weight",
            config.hidden_size,
            config.rms_norm_eps,
            dtype,
            weight_linears_fp8,
        )?;

        let lm_head_weight = if config.tie_word_embeddings {
            embed_weight.clone()
        } else {
            weights
                .get("lm_head.weight", dtype)
                .context("load lm_head.weight")?
        };
        #[cfg(feature = "cuda")]
        let lm_head_i8 = if !weight_linears_fp8
            && nv_config::env_arm_present_at_all_reread_every_call("NV_LAGUNA_LMHEAD_INT8")
            && device.is_cuda()
        {
            let (wq, row_scale) =
                nv_layers::lm_head_resident_quant::rowquant_i8_dev(&lm_head_weight)?;
            let max_m = nv_kernels::cuda::gemv_i8_normed_mk_max_m(config.hidden_size as i32).max(0)
                as usize;
            ensure!(max_m >= 1, "int8 lm_head: unsupported hidden size");
            Some(LagunaLmHeadI8 {
                wq,
                row_scale,
                max_m,
            })
        } else {
            None
        };
        #[cfg(feature = "cuda")]
        let lm_head_fp8 =
            if !weight_linears_fp8 && lm_head_i8.is_none() && device.is_cuda() {
                let (wq, row_scale) =
                    nv_layers::lm_head_resident_quant::rowquant_e4m3_dev(&lm_head_weight)?;
                let max_m = nv_kernels::cuda::gemv_i8_normed_mk_max_m(config.hidden_size as i32)
                    .max(0) as usize;
                ensure!(max_m >= 1, "fp8 lm_head: unsupported hidden size");
                Some(LagunaLmHeadFp8 {
                    wq,
                    row_scale,
                    max_m,
                })
            } else {
                None
            };
        let lm_head = load_linear_from_tensor_maybe_fp8(lm_head_weight, &fp8_runner, device)?;

        #[cfg(feature = "cuda")]
        let embed_fp8 = if weight_linears_fp8 {
            let (bytes, row_scales) =
                nv_layers::lm_head_resident_quant::rowquant_e4m3_dev(&embed_weight)?;
            Some(LagunaFp8Embedding {
                bytes,
                row_scales,
                vocab: config.vocab_size,
                hidden: config.hidden_size,
            })
        } else {
            None
        };
        let embed_weight = if weight_linears_fp8 {
            None
        } else {
            Some(embed_weight)
        };

        let rope_len = config.max_position_embeddings;
        let full_dim = config.rotary_dim_full();
        let full_inv_freq = yarn_inv_freq(full_dim, config.full_rope_params());
        let full_rope = Rope::from_inv_freq(
            RopeConfig {
                head_dim: full_dim,
                max_seq_len: rope_len,
                base: config.full_rope_params().rope_theta,
                kind: RopeKind::Yarn,
            },
            &full_inv_freq,
            device,
        )?;
        let sliding_dim = config.rotary_dim_sliding();
        let sliding_inv_freq = yarn_inv_freq(sliding_dim, config.sliding_rope_params());
        let sliding_rope = Rope::from_inv_freq(
            RopeConfig {
                head_dim: sliding_dim,
                max_seq_len: rope_len,
                base: config.sliding_rope_params().rope_theta,
                kind: RopeKind::Standard,
            },
            &sliding_inv_freq,
            device,
        )?;
        let full_attn_factor = config.full_attention_factor();

        Ok(Self {
            config,
            embed_weight,
            #[cfg(feature = "cuda")]
            embed_fp8,
            layers,
            final_norm,
            lm_head,
            sliding_rope,
            full_rope,
            full_attn_factor,
            dtype,
            device: device.clone(),
            #[cfg(feature = "cuda")]
            moe_decode_ctx: std::sync::Mutex::new(None),
            #[cfg(feature = "cuda")]
            moe_verify_ctx: std::sync::Mutex::new(HashMap::new()),
            #[cfg(feature = "cuda")]
            moe_graphs: std::sync::Mutex::new(None),
            #[cfg(feature = "cuda")]
            moe_union_samples: std::sync::Mutex::new(Vec::new()),
            #[cfg(feature = "cuda")]
            union_probe: AtomicBool::new(ck_profile_enabled()),
            #[cfg(feature = "cuda")]
            batch_moe_fused_dispatches: std::sync::atomic::AtomicUsize::new(0),
            #[cfg(feature = "cuda")]
            lm_head_i8,
            #[cfg(feature = "cuda")]
            lm_head_fp8,
            #[cfg(feature = "cuda")]
            lm_head_fp8_spec_off: !lmhead_fp8_forced_on_for_the_spec_tier_too(),
            device_verify_routing: AtomicBool::new(false),
            attn_w8_shape: attn_w8_shape_env(),
            prefill_chunk,
            #[cfg(test)]
            test_moe_host_fallbacks: std::sync::atomic::AtomicUsize::new(0),
        })
    }

    pub fn prefill_chunk_tokens(&self) -> usize {
        self.prefill_chunk
    }

    pub fn next_prefill_chunk_len(&self, remaining: usize) -> usize {
        canonical_tail_prefill_len(remaining, self.prefill_chunk)
    }

    pub fn new_kv_cache(&self, max_seq_len: usize) -> Result<LagunaKvCache> {
        LagunaKvCache::new_with_prefill_chunk(
            &self.config,
            max_seq_len,
            &self.device,
            self.dtype,
            self.prefill_chunk,
        )
    }

    #[cfg(feature = "cuda")]
    pub fn new_kv_cache_fp8(&self, max_seq_len: usize) -> Result<LagunaKvCacheFp8> {
        LagunaKvCacheFp8::new_with_prefill_chunk(
            &self.config,
            max_seq_len,
            &self.device,
            self.dtype,
            self.prefill_chunk,
        )
    }

    pub fn forward(&self, tokens: &Tensor, positions: &Tensor) -> Result<Tensor> {
        let dims = tokens.dims();
        if dims.len() != 2 || dims[0] != 1 {
            bail!("Laguna.forward: tokens must be [1, seq], got {:?}", dims);
        }
        let seq = dims[1];
        let mut cache = self.new_kv_cache(seq.max(1))?;
        self.forward_with_cache(tokens, positions, &mut cache)
    }

    pub fn forward_with_cache<C: Gemma4Cache>(
        &self,
        tokens: &Tensor,
        positions: &Tensor,
        cache: &mut C,
    ) -> Result<Tensor> {
        let (logits, _) = self.forward_with_cache_aux(tokens, positions, cache, &[])?;
        Ok(logits)
    }

    pub fn forward_with_cache_aux<C: Gemma4Cache>(
        &self,
        tokens: &Tensor,
        positions: &Tensor,
        cache: &mut C,
        aux_layers: &[usize],
    ) -> Result<(Tensor, Vec<Tensor>)> {
        self.forward_with_cache_aux_scoped(tokens, positions, cache, aux_layers, false)
    }

    pub fn forward_with_cache_aux_scoped<C: Gemma4Cache>(
        &self,
        tokens: &Tensor,
        positions: &Tensor,
        cache: &mut C,
        aux_layers: &[usize],
        spec_tier: bool,
    ) -> Result<(Tensor, Vec<Tensor>)> {
        let (hidden, aux) =
            self.forward_hidden_with_cache_aux(tokens, positions, cache, aux_layers)?;
        let logits = self.lm_head_from_prenorm_scoped(&hidden, spec_tier)?;
        Ok((logits.to_dtype(DType::F32)?, aux))
    }

    pub fn prefill_skip_intermediate_head_enabled() -> bool {
        nv_config::env_arm_opt_in_eq_some_1_reread_every_call(
            "NV_LAGUNA_PREFILL_SKIP_INTERMEDIATE_HEAD",
        )
    }

    #[cfg(feature = "cuda")]
    fn prefill_fused_qk_normrope(
        &self,
        attn: &LagunaAttention,
        q: &Tensor,
        k: &Tensor,
        positions: &Tensor,
        rope: &Rope,
        seq: usize,
        n_q: usize,
        n_kv: usize,
        head_dim: usize,
        rotary_dim: usize,
        rot_scale: f32,
    ) -> Result<Option<(Tensor, Tensor)>> {
        use cudarc::driver::{DevicePtr, DevicePtrMut};
        use half::bf16;
        let Device::Cuda(dev) = &self.device else {
            return Ok(None);
        };
        if attn.q_norm.fp8_e4m3_active() != attn.k_norm.fp8_e4m3_active() {
            return Ok(None);
        }
        if positions.dtype() != DType::I32
            || positions.dims() != [seq]
            || head_dim > 256
            || rotary_dim < 2
            || rotary_dim > head_dim
        {
            return Ok(None);
        }
        let q = q.contiguous()?;
        let k = k.contiguous()?;
        let positions = positions.contiguous()?;
        let cos = rope.cos().contiguous()?;
        let sin = rope.sin().contiguous()?;
        ensure!(
            cos.dims().last() == Some(&(rotary_dim / 2)),
            "fused normrope: rope table half {:?} does not match rotary_dim {rotary_dim}",
            cos.dims()
        );
        let stream = nv_layers::cuda_stream::current_stream(dev);
        let mut q_out: cudarc::driver::CudaSlice<bf16> = unsafe {
            stream
                .alloc::<bf16>(seq * n_q * head_dim)
                .map_err(|e| anyhow::anyhow!("fused normrope q alloc: {e:?}"))?
        };
        let mut k_out: cudarc::driver::CudaSlice<bf16> = unsafe {
            stream
                .alloc::<bf16>(seq * n_kv * head_dim)
                .map_err(|e| anyhow::anyhow!("fused normrope k alloc: {e:?}"))?
        };
        fn ptr_of<'a, T: candle_core::cuda_backend::CudaDType + cudarc::driver::DeviceRepr + 'a>(
            storage: &'a Storage,
            offset: usize,
            stream: &'a std::sync::Arc<CudaStream>,
        ) -> Result<(u64, cudarc::driver::SyncOnDrop<'a>)> {
            let cuda = match storage {
                Storage::Cuda(c) => c,
                _ => bail!("fused normrope: tensor is not on CUDA"),
            };
            Ok(cuda.as_cuda_slice::<T>()?.slice(offset..).view_ptr(stream))
        }
        let rc = {
            let (qs, ql) = q.storage_and_layout();
            let (ks, kl) = k.storage_and_layout();
            let (cs, cl) = cos.storage_and_layout();
            let (ss, sl) = sin.storage_and_layout();
            let (ps, pl) = positions.storage_and_layout();
            let (qi, _qi) = ptr_of::<bf16>(&qs, ql.start_offset(), &stream)?;
            let (ki, _ki) = ptr_of::<bf16>(&ks, kl.start_offset(), &stream)?;
            let (cp, _cos) = ptr_of::<f32>(&cs, cl.start_offset(), &stream)?;
            let (sp, _sin) = ptr_of::<f32>(&ss, sl.start_offset(), &stream)?;
            let (pp, _pos) = ptr_of::<i32>(&ps, pl.start_offset(), &stream)?;
            let (qo, _g1) = q_out.device_ptr_mut(&stream);
            let (ko, _g2) = k_out.device_ptr_mut(&stream);
            if let (Some((qw, qs, qh)), Some((kw, ks, kh))) = (
                attn.q_norm.fp8_e4m3_weight_and_scale(),
                attn.k_norm.fp8_e4m3_weight_and_scale(),
            ) {
                ensure!(
                    qh == head_dim
                        && kh == head_dim
                        && qw.len() == head_dim
                        && kw.len() == head_dim
                        && qs.len() == 1
                        && ks.len() == 1,
                    "FP8 prefill norm storage does not match head_dim={head_dim}"
                );
                let (qwp, _qw) = qw.device_ptr(&stream);
                let (kwp, _kw) = kw.device_ptr(&stream);
                let (qsp, _qs) = qs.device_ptr(&stream);
                let (ksp, _ks) = ks.device_ptr(&stream);
                unsafe {
                    nv_kernels::cuda::laguna_prefill_qk_normrope_fp8_weight(
                        stream.cu_stream() as *mut std::ffi::c_void,
                        qi as *const u16,
                        ki as *const u16,
                        qo as *mut u16,
                        ko as *mut u16,
                        qwp as *const u8,
                        kwp as *const u8,
                        qsp as *const f32,
                        ksp as *const f32,
                        cp as *const f32,
                        sp as *const f32,
                        pp as *const i32,
                        n_q as i32,
                        n_kv as i32,
                        head_dim as i32,
                        rotary_dim as i32,
                        rot_scale,
                        attn.q_norm.eps() as f32,
                        attn.k_norm.eps() as f32,
                        seq as i32,
                    )
                }
            } else {
                let qw = attn.q_norm.weight_bf16().contiguous()?;
                let kw = attn.k_norm.weight_bf16().contiguous()?;
                let (qws, qwl) = qw.storage_and_layout();
                let (kws, kwl) = kw.storage_and_layout();
                let (qwp, _qw) = ptr_of::<bf16>(&qws, qwl.start_offset(), &stream)?;
                let (kwp, _kw) = ptr_of::<bf16>(&kws, kwl.start_offset(), &stream)?;
                unsafe {
                    nv_kernels::cuda::laguna_prefill_qk_normrope_bf16(
                        stream.cu_stream() as *mut std::ffi::c_void,
                        qi as *const u16,
                        ki as *const u16,
                        qo as *mut u16,
                        ko as *mut u16,
                        qwp as *const u16,
                        kwp as *const u16,
                        cp as *const f32,
                        sp as *const f32,
                        pp as *const i32,
                        n_q as i32,
                        n_kv as i32,
                        head_dim as i32,
                        rotary_dim as i32,
                        rot_scale,
                        attn.q_norm.eps() as f32,
                        attn.k_norm.eps() as f32,
                        seq as i32,
                    )
                }
            }
        };
        ensure!(rc == 0, "fused prefill normrope launch failed rc={rc}");
        let wrap = |slice: cudarc::driver::CudaSlice<bf16>, heads: usize| -> Result<Tensor> {
            Ok(Tensor::from_storage(
                Storage::Cuda(CudaStorage::wrap_cuda_slice(slice, dev.clone())),
                (1usize, seq, heads, head_dim),
                BackpropOp::none(),
                false,
            ))
        };
        Ok(Some((wrap(q_out, n_q)?, wrap(k_out, n_kv)?)))
    }

    #[cfg(feature = "cuda")]
    fn prefill_fused_gate(
        &self,
        attn_out: &Tensor,
        gate: &Tensor,
        seq: usize,
        n_q: usize,
        head_dim: usize,
    ) -> Result<Option<Tensor>> {
        use cudarc::driver::DevicePtrMut;
        use half::bf16;
        let Device::Cuda(dev) = &self.device else {
            return Ok(None);
        };
        if attn_out.dtype() != DType::BF16 || gate.dtype() != DType::BF16 || head_dim > 1024 {
            return Ok(None);
        }
        let attn_c = attn_out.contiguous()?;
        let gate_c = gate.contiguous()?;
        ensure!(
            attn_c.elem_count() == seq * n_q * head_dim && gate_c.elem_count() == seq * n_q,
            "fused gate: shape mismatch"
        );
        let stream = nv_layers::cuda_stream::current_stream(dev);
        let mut out: cudarc::driver::CudaSlice<bf16> = unsafe {
            stream
                .alloc::<bf16>(seq * n_q * head_dim)
                .map_err(|e| anyhow::anyhow!("fused gate alloc: {e:?}"))?
        };
        let rc = {
            fn ptr_of<'a>(
                storage: &'a Storage,
                offset: usize,
                stream: &'a std::sync::Arc<CudaStream>,
            ) -> Result<(u64, cudarc::driver::SyncOnDrop<'a>)> {
                let cuda = match storage {
                    Storage::Cuda(c) => c,
                    _ => bail!("fused gate: tensor is not on CUDA"),
                };
                Ok(cuda
                    .as_cuda_slice::<bf16>()?
                    .slice(offset..)
                    .view_ptr(stream))
            }
            let (attn_storage, attn_layout) = attn_c.storage_and_layout();
            let (gate_storage, gate_layout) = gate_c.storage_and_layout();
            let (ap, _attn) = ptr_of(&attn_storage, attn_layout.start_offset(), &stream)?;
            let (gp, _gate) = ptr_of(&gate_storage, gate_layout.start_offset(), &stream)?;
            let (op, _g) = out.device_ptr_mut(&stream);
            unsafe {
                nv_kernels::cuda::laguna_prefill_gate_bf16(
                    stream.cu_stream() as *mut std::ffi::c_void,
                    ap as *const u16,
                    gp as *const u16,
                    op as *mut u16,
                    seq as i32,
                    n_q as i32,
                    head_dim as i32,
                )
            }
        };
        ensure!(rc == 0, "fused prefill gate launch failed rc={rc}");
        Ok(Some(Tensor::from_storage(
            Storage::Cuda(CudaStorage::wrap_cuda_slice(out, dev.clone())),
            (1usize, seq, n_q, head_dim),
            BackpropOp::none(),
            false,
        )))
    }

    pub fn prefill_with_cache_aux_scoped<C: Gemma4Cache>(
        &self,
        tokens: &Tensor,
        positions: &Tensor,
        cache: &mut C,
        aux_layers: &[usize],
        spec_tier: bool,
        final_chunk: bool,
    ) -> Result<(Option<Tensor>, Vec<Tensor>)> {
        if !final_chunk && Self::prefill_skip_intermediate_head_enabled() {
            let (_, aux) =
                self.forward_hidden_with_cache_aux(tokens, positions, cache, aux_layers)?;
            Ok((None, aux))
        } else {
            let (logits, aux) = self
                .forward_with_cache_aux_scoped(tokens, positions, cache, aux_layers, spec_tier)?;
            Ok((Some(logits), aux))
        }
    }

    fn forward_hidden_with_cache_aux<C: Gemma4Cache>(
        &self,
        tokens: &Tensor,
        positions: &Tensor,
        cache: &mut C,
        aux_layers: &[usize],
    ) -> Result<(Tensor, Vec<Tensor>)> {
        let dims = tokens.dims();
        if dims.len() != 2 || dims[0] != 1 {
            bail!("Laguna.forward: tokens must be [1, seq], got {:?}", dims);
        }
        let seq = dims[1];
        if positions.dims() != [seq] {
            bail!(
                "Laguna.forward: positions must be [{}], got {:?}",
                seq,
                positions.dims()
            );
        }
        for &li in aux_layers {
            if li >= self.layers.len() {
                bail!(
                    "Laguna.forward_with_cache_aux: aux layer {} out of range ({} layers)",
                    li,
                    self.layers.len()
                );
            }
        }

        self.apply_attn_w8_for(seq);
        let tokens_flat = tokens.flatten_all()?.to_dtype(DType::U32)?;
        let x = self
            .embed_tokens(&tokens_flat)?
            .reshape((1usize, seq, self.config.hidden_size))?
            .to_dtype(self.dtype)?;
        let mut hidden = x;

        let write_start = cache.current_len();
        let new_total = write_start + seq;
        cache.prepare_for_decode(write_start, new_total)?;

        let mut aux: Vec<Tensor> = Vec::with_capacity(aux_layers.len());
        for li in 0..self.layers.len() {
            hidden = self.layer_forward(li, &hidden, positions, cache, seq, new_total)?;
            if aux_layers.contains(&li) {
                aux.push(hidden.clone());
            }
        }
        cache.advance(seq);

        Ok((hidden, aux))
    }

    fn layer_forward<C: Gemma4Cache>(
        &self,
        idx: usize,
        x: &Tensor,
        positions: &Tensor,
        cache: &mut C,
        seq: usize,
        new_total: usize,
    ) -> Result<Tensor> {
        let layer = &self.layers[idx];

        let prof =
            seq == 1 && nv_config::env_arm_present_at_all_reread_every_call("NV_MOE_DECODE_PROF");
        let sync = |label: &str, t: &mut Instant| {
            if prof {
                #[cfg(feature = "cuda")]
                if let Device::Cuda(d) = &self.device {
                    let _ = d.cuda_stream().synchronize();
                }
                eprintln!(
                    "[laguna_layer_prof] l{idx} {label}={:.3}ms",
                    t.elapsed().as_secs_f64() * 1e3
                );
                *t = Instant::now();
            }
        };
        let mut t = Instant::now();

        let family_attn = match layer.kind {
            LayerType::FullAttention => "prefill_attention_full_layer",
            LayerType::SlidingAttention => "prefill_attention_sliding_layer",
        };
        let after_attn = nv_train::train_prof::phase(&self.device, family_attn, || {
            let normed = layer.input_layernorm.forward(x)?;
            let attn_out =
                self.attention_forward(idx, &normed, positions, cache, seq, new_total)?;
            Ok(x.add(&attn_out.to_dtype(x.dtype())?)?)
        })?;
        sync("attn", &mut t);
        let family_ffn = match &layer.ffn {
            LagunaFfn::Dense(_) => "prefill_dense_mlp_layer",
            LagunaFfn::Moe(_) => "prefill_moe_layer",
        };
        let out = nv_train::train_prof::phase(&self.device, family_ffn, || {
            self.ffn_forward(idx, &after_attn, seq)
        });
        sync("ffn", &mut t);
        out
    }

    fn ffn_forward(&self, idx: usize, after_attn: &Tensor, seq: usize) -> Result<Tensor> {
        let layer = &self.layers[idx];
        #[cfg(feature = "cuda")]
        if seq == 1
            && self.device.is_cuda()
            && crate::laguna_graph::graph_enabled()
        {
            if let LagunaFfn::Moe(moe) = &layer.ffn {
                if let Some(out) = self.moe_layer_fused_graph(
                    idx,
                    moe,
                    &layer.post_attention_layernorm,
                    after_attn,
                )? {
                    return Ok(out);
                }
            }
        }

        let normed_mlp = layer.post_attention_layernorm.forward(after_attn)?;
        let ffn_out = match &layer.ffn {
            LagunaFfn::Dense(mlp) => mlp.forward(&normed_mlp)?,
            LagunaFfn::Moe(moe) => {
                let n_tokens = seq;
                let flat = normed_mlp
                    .reshape((n_tokens, self.config.hidden_size))?
                    .contiguous()?;
                let out = self.moe_forward(idx, moe, &flat, n_tokens)?;
                out.reshape((1usize, n_tokens, self.config.hidden_size))?
            }
        };
        after_attn
            .add(&ffn_out.to_dtype(after_attn.dtype())?)
            .map_err(Into::into)
    }

    pub fn set_device_verify_routing(&self, on: bool) {
        self.device_verify_routing.store(on, Ordering::Relaxed);
    }

    pub fn device_verify_routing(&self) -> bool {
        self.device_verify_routing.load(Ordering::Relaxed)
    }

    fn moe_forward(
        &self,
        layer_idx: usize,
        moe: &LagunaMoe,
        x_flat: &Tensor,
        n_tokens: usize,
    ) -> Result<Tensor> {
        #[cfg(feature = "cuda")]
        if self.device.is_cuda() {
            if n_tokens == 1 {
                if let Some(out) = self.moe_forward_decode_grouped(layer_idx, moe, x_flat)? {
                    return Ok(out);
                }
            } else {
                if n_tokens <= MAX_VERIFY_MOE_TOKENS && self.device_verify_routing() {
                    match self.moe_forward_verify_grouped(layer_idx, moe, x_flat, n_tokens, false) {
                        Ok(Some(out)) => return Ok(out),
                        Ok(None) => {}
                        Err(e) => {
                            if moe.experts.is_empty() {
                                return Err(e)
                                    .context("grouped-only verify failed; fallback disabled");
                            }
                            eprintln!(
                                "[laguna] device-routed verify MoE failed, host-routed fallback: {e:#}"
                            );
                            self.set_device_verify_routing(false);
                        }
                    }
                }
                if let Some(out) = self.moe_forward_prefill_grouped(moe, x_flat, n_tokens)? {
                    return Ok(out);
                }
            }
        }
        let _ = n_tokens;
        #[cfg(test)]
        self.test_moe_host_fallbacks.fetch_add(1, Ordering::Relaxed);
        let out = moe.forward(x_flat)?;
        let finite_probe = out.to_dtype(DType::F32)?.sum_all()?.to_scalar::<f32>()?;
        if !finite_probe.is_finite() {
            bail!(
                "laguna host-MoE fallback at layer {layer_idx} produced non-finite output \
                 (sum {finite_probe}); refusing to emit NaN logits. This failure has been \
                 observed under GPU memory pressure (co-tenant holding ~43 GiB); free VRAM \
                 and retry instead of trusting this forward pass"
            );
        }
        Ok(out)
    }

    #[cfg(feature = "cuda")]
    fn prepared_moe_context<'a>(
        &self,
        map: &'a mut HashMap<usize, nv_layers::moe_grouped::GroupedDecodeContext>,
        n_tokens: usize,
        folds_shared: bool,
        dev: &candle_core::CudaDevice,
    ) -> Result<&'a mut nv_layers::moe_grouped::GroupedDecodeContext> {
        use nv_layers::moe_grouped::GroupedDecodeContext;
        let ctx = match map.entry(n_tokens) {
            std::collections::hash_map::Entry::Occupied(e) => e.into_mut(),
            std::collections::hash_map::Entry::Vacant(v) => {
                let stream = nv_layers::cuda_stream::current_stream(dev);
                let new = if folds_shared {
                    GroupedDecodeContext::new_folded_shared
                } else {
                    GroupedDecodeContext::new_multi
                };
                v.insert(new(
                    self.config.hidden_size,
                    self.config.moe_intermediate_size,
                    self.config.num_experts_per_tok,
                    self.config.num_experts,
                    n_tokens,
                    &stream,
                )?)
            }
        };
        ensure!(
            ctx.folds_shared() == folds_shared,
            "prepared grouped context must match the fixed opt-in shared-fold policy"
        );
        Ok(ctx)
    }

    #[cfg(feature = "cuda")]
    fn moe_forward_verify_grouped(
        &self,
        layer_idx: usize,
        moe: &LagunaMoe,
        x_flat: &Tensor,
        n_tokens: usize,
        fused_batched_decode: bool,
    ) -> Result<Option<Tensor>> {
        let Device::Cuda(dev) = self.device.clone() else {
            return Ok(None);
        };
        let grouped = match self.grouped_weights(moe)? {
            Some(w) => w,
            None => return Ok(None),
        };
        let folds_shared = grouped.folded_shared && Self::prepared_shared_fold_opt_in();
        let mut map = self
            .moe_verify_ctx
            .lock()
            .map_err(|e| anyhow::anyhow!("moe_verify_ctx mutex poisoned: {e}"))?;
        let ctx = self.prepared_moe_context(&mut map, n_tokens, folds_shared, &dev)?;

        let logits = moe.gate.forward(x_flat)?;
        let routed = if fused_batched_decode {
            use cudarc::driver::DevicePtr;

            ensure!(
                matches!(n_tokens, 2 | 3 | 6)
                    && x_flat.dtype() == DType::BF16
                    && x_flat.is_contiguous()
                    && x_flat.elem_count() == n_tokens * self.config.hidden_size,
                "Laguna fused routed MoE requires contiguous BF16 (M,H) input with M2/M3/M6, got {:?} {:?}",
                x_flat.dims(),
                x_flat.dtype()
            );
            ensure!(
                logits.dtype() == DType::BF16
                    && logits.is_contiguous()
                    && logits.elem_count() == n_tokens * self.config.num_experts,
                "Laguna fused routed MoE requires contiguous BF16 (M,E) logits, got {:?} {:?}",
                logits.dims(),
                logits.dtype()
            );
            let logits = logits.contiguous()?;
            let x_flat = x_flat.contiguous()?;
            let bias = moe.selection_bias.to_dtype(DType::F32)?.contiguous()?;
            let stream = nv_layers::cuda_stream::current_stream(&dev);

            let (logits_storage, logits_layout) = logits.storage_and_layout();
            let logits_cuda = match &*logits_storage {
                Storage::Cuda(storage) => storage,
                _ => bail!("Laguna B2 fused router logits must be CUDA"),
            };
            let logits_slice = logits_cuda.as_cuda_slice::<half::bf16>()?;
            let (logits_ptr, _logits_guard) = logits_slice.device_ptr(&stream);
            let logits_ptr = logits_ptr
                + (logits_layout.start_offset() * std::mem::size_of::<half::bf16>()) as u64;
            ctx.cast_router_logits_bf16_into_logits_f32_softcap_applied_at_route_not_here(
                &stream, logits_ptr,
            )?;

            let (x_storage, x_layout) = x_flat.storage_and_layout();
            let x_cuda = match &*x_storage {
                Storage::Cuda(storage) => storage,
                _ => bail!("Laguna B2 fused input must be CUDA"),
            };
            let x_slice = x_cuda.as_cuda_slice::<half::bf16>()?;
            let (x_ptr, _x_guard) = x_slice.device_ptr(&stream);
            let x_ptr =
                x_ptr + (x_layout.start_offset() * std::mem::size_of::<half::bf16>()) as u64;
            ctx.stage_x_bf16_rows_dtod(&stream, x_ptr)?;

            let (bias_storage, bias_layout) = bias.storage_and_layout();
            let bias_cuda = match &*bias_storage {
                Storage::Cuda(storage) => storage,
                _ => bail!("Laguna B2 fused selection bias must be CUDA"),
            };
            let bias_slice = bias_cuda.as_cuda_slice::<f32>()?;
            let (bias_ptr, _bias_guard) = bias_slice.device_ptr(&stream);
            let bias_ptr =
                bias_ptr + (bias_layout.start_offset() * std::mem::size_of::<f32>()) as u64;
            nv_layers::moe_grouped::forward_grouped_decode_into(
                &grouped,
                ctx,
                bias_ptr,
                1,
                moe.softcap,
                moe.norm_topk,
                moe.routed_scaling,
                true,
                &stream,
                None,
                None,
            )?;
            ensure!(
                ctx.last_call_used_fused_small_m_route_and_chain(),
                "Laguna B2 fused routed MoE inner fused route/chain was not selected"
            );
            let routed = ctx.copy_scattered_f32_into_owned_tensor(&self.device, &stream)?;
            self.batch_moe_fused_dispatches
                .fetch_add(1, Ordering::Relaxed);
            routed
        } else {
            nv_layers::moe_grouped::forward_grouped_decode(
                &grouped,
                ctx,
                x_flat,
                &logits,
                Some(&moe.selection_bias),
                1,
                moe.softcap,
                moe.norm_topk,
                moe.routed_scaling,
                &self.device,
            )?
        };
        if nv_config::env_arm_present_at_all_reread_every_call("NV_LAGUNA_ROUTE_AB_PROBE") {
            self.route_ab_probe_report_device_vs_host_choices_on_identical_input(
                layer_idx, moe, x_flat, ctx, n_tokens, &dev,
            )?;
        }
        if self.union_probe.load(Ordering::Relaxed) {
            let stream = nv_layers::cuda_stream::current_stream(&dev);
            if let Ok(ids) = stream.clone_dtoh(&ctx.topk_ids) {
                let mut seen = std::collections::HashSet::new();
                for &e in &ids {
                    seen.insert(e);
                }
                if let Ok(mut samples) = self.moe_union_samples.lock() {
                    samples.push((n_tokens, seen.len()));
                }
            }
        }
        if folds_shared {
            return Ok(Some(routed));
        }
        let shared = moe.shared_expert.forward(x_flat)?.to_dtype(DType::F32)?;
        Ok(Some(routed.add(&shared)?))
    }

    #[cfg(feature = "cuda")]
    fn route_ab_probe_report_device_vs_host_choices_on_identical_input(
        &self,
        layer_idx: usize,
        moe: &LagunaMoe,
        x_flat: &Tensor,
        ctx: &nv_layers::moe_grouped::GroupedDecodeContext,
        n_tokens: usize,
        dev: &candle_core::CudaDevice,
    ) -> Result<()> {
        let stream = nv_layers::cuda_stream::current_stream(dev);
        stream.synchronize().ok();
        let dev_ids: Vec<i32> = stream
            .clone_dtoh(&ctx.topk_ids)
            .map_err(|e| anyhow::anyhow!("route_ab_probe topk_ids copy: {e:?}"))?;
        let (host_ids, _host_w) = moe.route_host(x_flat, n_tokens)?;
        let k = moe.top_k;
        let e_num = moe.num_experts;
        let mut lf = moe
            .gate
            .forward(x_flat)?
            .to_dtype(DType::F32)?
            .contiguous()?;
        if moe.softcap > 0.0 {
            lf = lf
                .affine(1.0 / moe.softcap as f64, 0.0)?
                .tanh()?
                .affine(moe.softcap as f64, 0.0)?;
        }
        let sel = candle_nn::ops::sigmoid(&lf)?.broadcast_add(&moe.selection_bias)?;
        let sel_host: Vec<f32> = sel.flatten_all()?.to_vec1()?;
        for t in 0..n_tokens {
            let mut sh: Vec<u32> = host_ids[t * k..(t + 1) * k].to_vec();
            let mut sd: Vec<u32> = dev_ids[t * k..(t + 1) * k]
                .iter()
                .map(|&v| v as u32)
                .collect();
            sh.sort_unstable();
            sd.sort_unstable();
            if sh == sd {
                continue;
            }
            let only_h: Vec<u32> = sh.iter().filter(|e| !sd.contains(e)).copied().collect();
            let only_d: Vec<u32> = sd.iter().filter(|e| !sh.contains(e)).copied().collect();
            for &e in only_h.iter().chain(only_d.iter()) {
                let s = sel_host[t * e_num + e as usize];
                eprintln!(
                    "[route-ab] layer {layer_idx} token {t}: expert {e} {} sel {s:.9} bits {:#010x}",
                    if only_h.contains(&e) {
                        "host-only"
                    } else {
                        "device-only"
                    },
                    s.to_bits()
                );
            }
        }
        Ok(())
    }

    #[cfg(feature = "cuda")]
    pub fn take_batch_moe_fused_dispatches(&self) -> usize {
        self.batch_moe_fused_dispatches.swap(0, Ordering::Relaxed)
    }

    #[cfg(feature = "cuda")]
    pub(crate) fn grouped_weights(
        &self,
        moe: &LagunaMoe,
    ) -> Result<Option<Arc<MoeGroupedWeights>>> {
        let Device::Cuda(dev) = self.device.clone() else {
            return Ok(None);
        };
        let mut slot = moe
            .grouped
            .lock()
            .map_err(|e| anyhow::anyhow!("LagunaMoe.grouped mutex poisoned: {e}"))?;
        if moe.experts.is_empty() {
            let bank = slot.as_ref().and_then(Option::as_ref).context(
                "grouped-only bank unavailable; released routed experts cannot be rebuilt or used as fallback",
            )?;
            ensure!(
                bank.num_experts == moe.num_experts
                    && bank.hidden_size == self.config.hidden_size
                    && bank.intermediate_size == self.config.moe_intermediate_size,
                "grouped-only bank dimensions changed"
            );
            return Ok(Some(Arc::clone(bank)));
        }
        if slot.is_none() {
            let stream = nv_layers::cuda_stream::current_stream(&dev);
            let built = MoeGroupedWeights::build_from_experts_folding_shared_as_a_fixed_extra_tile(
                &moe.experts,
                &moe.shared_expert,
                self.config.hidden_size,
                self.config.moe_intermediate_size,
                &stream,
            );
            *slot = Some(match built {
                Ok(b) => Some(Arc::new(b)),
                Err(e) => {
                    eprintln!("[laguna] grouped MoE init failed, host fallback: {e}");
                    None
                }
            });
        }
        Ok(slot.as_ref().unwrap().clone())
    }

    #[cfg(feature = "cuda")]
    fn moe_forward_prefill_grouped(
        &self,
        moe: &LagunaMoe,
        x_flat: &Tensor,
        n_tokens: usize,
    ) -> Result<Option<Tensor>> {
        let grouped = match self.grouped_weights(moe)? {
            Some(w) => w,
            None => return Ok(None),
        };
        let k = moe.top_k;
        let logits = moe.gate.forward(x_flat)?;
        let routing =
            nv_layers::moe_grouped::route_topk_and_plan_padded_on_device_for_prefill_dispatch(
                &logits,
                Some(&moe.selection_bias),
                n_tokens,
                moe.num_experts,
                k,
                1,
                moe.softcap,
                moe.norm_topk,
                1.0,
                &self.device,
            )?;
        let mut acc = match nv_layers::moe_grouped::forward_grouped_device_routing(
            &grouped,
            &grouped,
            x_flat,
            &routing,
            n_tokens,
            k,
            &self.device,
        ) {
            Ok(t) => t,
            Err(e) => {
                if moe.experts.is_empty() {
                    return Err(e).context("grouped-only device prefill failed; fallback disabled");
                }
                eprintln!("[laguna] device-planned grouped MoE prefill failed, host fallback: {e}");
                return Ok(None);
            }
        };
        if moe.routed_scaling != 1.0 {
            acc = acc.affine(moe.routed_scaling as f64, 0.0)?;
        }
        let shared = moe.shared_expert.forward(x_flat)?.to_dtype(DType::F32)?;
        Ok(Some(acc.add(&shared)?))
    }

    #[cfg(feature = "cuda")]
    fn moe_layer_fused_graph(
        &self,
        layer_idx: usize,
        moe: &LagunaMoe,
        norm: &RmsNorm,
        after_attn: &Tensor,
    ) -> Result<Option<Tensor>> {
        let Device::Cuda(dev) = self.device.clone() else {
            return Ok(None);
        };
        let grouped = match self.grouped_weights(moe)? {
            Some(w) => w,
            None => return Ok(None),
        };
        let mut ctx_slot = self
            .moe_decode_ctx
            .lock()
            .map_err(|e| anyhow::anyhow!("moe_decode_ctx mutex poisoned: {e}"))?;
        if ctx_slot.is_none() {
            let stream = nv_layers::cuda_stream::current_stream(&dev);
            *ctx_slot = Some(if grouped.folded_shared {
                nv_layers::moe_grouped::GroupedDecodeContext::new_folded_shared(
                    self.config.hidden_size,
                    self.config.moe_intermediate_size,
                    self.config.num_experts_per_tok,
                    self.config.num_experts,
                    1,
                    &stream,
                )?
            } else {
                nv_layers::moe_grouped::GroupedDecodeContext::new(
                    self.config.hidden_size,
                    self.config.moe_intermediate_size,
                    self.config.num_experts_per_tok,
                    self.config.num_experts,
                    &stream,
                )?
            });
        }
        let ctx = ctx_slot.as_mut().unwrap();

        let mut g_slot = self
            .moe_graphs
            .lock()
            .map_err(|e| anyhow::anyhow!("moe_graphs mutex poisoned: {e}"))?;
        if g_slot.is_none() {
            *g_slot = Some(
                match crate::laguna_graph::LagunaMoeGraphs::new(&dev, self.layers.len()) {
                    Ok(g) => Some(g),
                    Err(e) => {
                        if moe.experts.is_empty() {
                            return Err(e)
                                .context("grouped-only MoE graph init failed; fallback disabled");
                        }
                        eprintln!("[laguna] moe graph init failed, uncaptured path: {e}");
                        None
                    }
                },
            );
        }
        if let Some(Some(graphs)) = g_slot.as_mut() {
            ensure!(
                !moe.experts.is_empty() || !graphs.failed(),
                "grouped-only MoE graph is retired; uncaptured fallback disabled"
            );
            if !graphs.failed() {
                let hidden = self.config.hidden_size;
                let resid = after_attn.reshape((1usize, hidden))?;
                match graphs.forward_layer(layer_idx, moe, norm, &grouped, ctx, &resid, &dev) {
                    Ok(out) => {
                        return Ok(Some(out.reshape((1usize, 1usize, hidden))?));
                    }
                    Err(e) => {
                        if moe.experts.is_empty() {
                            graphs.mark_failed();
                            return Err(e)
                                .context("grouped-only MoE graph failed; fallback disabled");
                        }
                        eprintln!(
                            "[laguna] moe graph layer {layer_idx} failed, permanent uncaptured fallback: {e:#}"
                        );
                        graphs.mark_failed();
                        let _ = dev.cuda_stream().synchronize();
                    }
                }
            }
        }
        Ok(None)
    }

    #[cfg(feature = "cuda")]
    fn moe_forward_decode_grouped(
        &self,
        layer_idx: usize,
        moe: &LagunaMoe,
        x_flat: &Tensor,
    ) -> Result<Option<Tensor>> {
        let Device::Cuda(dev) = self.device.clone() else {
            return Ok(None);
        };
        let grouped = match self.grouped_weights(moe)? {
            Some(w) => w,
            None => return Ok(None),
        };

        let mut ctx_slot = self
            .moe_decode_ctx
            .lock()
            .map_err(|e| anyhow::anyhow!("moe_decode_ctx mutex poisoned: {e}"))?;
        if ctx_slot.is_none() {
            let stream = nv_layers::cuda_stream::current_stream(&dev);
            *ctx_slot = Some(if grouped.folded_shared {
                nv_layers::moe_grouped::GroupedDecodeContext::new_folded_shared(
                    self.config.hidden_size,
                    self.config.moe_intermediate_size,
                    self.config.num_experts_per_tok,
                    self.config.num_experts,
                    1,
                    &stream,
                )?
            } else {
                nv_layers::moe_grouped::GroupedDecodeContext::new(
                    self.config.hidden_size,
                    self.config.moe_intermediate_size,
                    self.config.num_experts_per_tok,
                    self.config.num_experts,
                    &stream,
                )?
            });
        }
        let ctx = ctx_slot.as_mut().unwrap();
        let _ = layer_idx;

        let prof = nv_config::env_arm_present_at_all_reread_every_call("NV_MOE_DECODE_PROF");
        let sync = |label: &str, t: &mut Instant| {
            if prof {
                let _ = dev.cuda_stream().synchronize();
                eprintln!(
                    "[laguna_moe_prof] {label}={:.3}ms",
                    t.elapsed().as_secs_f64() * 1e3
                );
                *t = Instant::now();
            }
        };
        let mut t = Instant::now();
        let logits = moe.gate.forward(x_flat)?;
        sync("gate_fwd", &mut t);
        let routed = nv_layers::moe_grouped::forward_grouped_decode(
            &grouped,
            ctx,
            x_flat,
            &logits,
            Some(&moe.selection_bias),
            1,
            moe.softcap,
            moe.norm_topk,
            moe.routed_scaling,
            &self.device,
        )?;
        sync("pipeline", &mut t);
        if grouped.folded_shared {
            return Ok(Some(routed));
        }
        let shared = moe.shared_expert.forward(x_flat)?.to_dtype(DType::F32)?;
        sync("shared_fwd", &mut t);
        let out = routed.add(&shared)?;
        sync("add", &mut t);
        Ok(Some(out))
    }

    fn attention_forward<C: Gemma4Cache>(
        &self,
        layer_idx: usize,
        x: &Tensor,
        positions: &Tensor,
        cache: &mut C,
        seq: usize,
        new_total: usize,
    ) -> Result<Tensor> {
        let layer = &self.layers[layer_idx];
        let attn = &layer.self_attn;
        let kind = attn.kind;
        let head_dim = self.config.head_dim;
        let n_q = attn.num_heads;
        let n_kv = self.config.num_key_value_heads;

        #[cfg(feature = "cuda")]
        let want_fused_normrope =
            seq > 1 && self.dtype == DType::BF16 && self.device.is_cuda();
        #[cfg(not(feature = "cuda"))]
        let want_fused_normrope = false;
        let q_raw = attn.proj_q(x, seq)?;
        let q = q_raw.reshape((1usize, seq, n_q, head_dim))?;
        let q_normed = if want_fused_normrope {
            None
        } else {
            Some(attn.q_norm.forward(&q)?)
        };
        let k_raw = attn.proj_k(x, seq)?;
        let k = k_raw.reshape((1usize, seq, n_kv, head_dim))?;
        let k_normed = if want_fused_normrope {
            None
        } else {
            Some(attn.k_norm.forward(&k)?)
        };
        let v = attn
            .proj_v(x, seq)?
            .reshape((1usize, seq, n_kv, head_dim))?;

        let (rope, rotary_dim, rot_scale) = match kind {
            LayerType::SlidingAttention => {
                (&self.sliding_rope, self.config.rotary_dim_sliding(), 1.0f32)
            }
            LayerType::FullAttention => (
                &self.full_rope,
                self.config.rotary_dim_full(),
                self.full_attn_factor,
            ),
        };
        #[cfg(feature = "cuda")]
        let fused_normrope = if want_fused_normrope {
            self.prefill_fused_qk_normrope(
                attn, &q, &k, positions, rope, seq, n_q, n_kv, head_dim, rotary_dim, rot_scale,
            )?
        } else {
            None
        };
        #[cfg(not(feature = "cuda"))]
        let fused_normrope: Option<(Tensor, Tensor)> = None;
        let (q_rot, k_rot) = match fused_normrope {
            Some(pair) => {
                static PREFILL_FUSED_NORMROPE_FIRED_ONCE: Once = Once::new();
                PREFILL_FUSED_NORMROPE_FIRED_ONCE.call_once(|| eprintln!("[laguna_prefill_fused] normrope=on rows={seq} n_q={n_q} n_kv={n_kv} rotary_dim={rotary_dim} rot_scale={rot_scale} (q/k rmsnorm + partial rope + scale + bf16 in one launch per row-head)"));
                pair
            }
            None => {
                let q_normed = match q_normed {
                    Some(q) => q,
                    None => attn.q_norm.forward(&q)?,
                };
                let k_normed = match k_normed {
                    Some(k) => k,
                    None => attn.k_norm.forward(&k)?,
                };
                let (q_rot, k_rot) = apply_partial_rope_scaled(
                    &q_normed, &k_normed, rope, positions, rotary_dim, head_dim, rot_scale,
                )?;
                (
                    q_rot.to_dtype(self.dtype)?.contiguous()?,
                    k_rot.to_dtype(self.dtype)?.contiguous()?,
                )
            }
        };
        let v_for_cache = v.contiguous()?;

        cache.write_at(layer_idx, &k_rot, &v_for_cache)?;

        let window = match kind {
            LayerType::SlidingAttention => Some(self.config.sliding_window),
            LayerType::FullAttention => None,
        };
        let scale = (head_dim as f32).powf(-0.5);

        let attn_out = {
            let mut fast_path_out = if seq == 1 {
                cache.try_decode_attention_fp8(layer_idx, &q_rot, n_q, window, scale)?
            } else if window.is_none() {
                cache.try_prefill_attention_fp8(layer_idx, &q_rot, n_q, seq, window, scale)?
            } else {
                None
            };
            #[cfg(feature = "cuda")]
            if seq == 1 && fast_path_out.is_none() {
                fast_path_out =
                    cache.try_decode_attention_gqa(layer_idx, &q_rot, n_q, window, scale)?;
            }
            if seq == 1 && fast_path_out.is_none() {
                fast_path_out =
                    cache.try_decode_attention_ring(layer_idx, &q_rot, n_q, window, scale)?;
            }
            if let Some(out) = fast_path_out {
                out
            } else {
                let (k_full, v_full) = cache.view(layer_idx, new_total)?;
                attention(
                    &q_rot, &k_full, &v_full, n_q, n_kv, head_dim, seq, scale, window,
                )?
            }
        };

        #[cfg(feature = "cuda")]
        let fused_gate = match (&attn.g_proj, self.config.gating) {
            (Some(g_proj), LagunaGating::PerHead) if seq > 1 => {
                let g = g_proj.forward(x)?;
                let fused = self.prefill_fused_gate(&attn_out, &g, seq, n_q, head_dim)?;
                if fused.is_some() {
                    static PREFILL_FUSED_GATE_FIRED_ONCE: Once = Once::new();
                    PREFILL_FUSED_GATE_FIRED_ONCE.call_once(|| eprintln!("[laguna_prefill_fused] gate=on rows={seq} n_q={n_q} (softplus gate x attention in one launch per row-head)"));
                }
                fused
            }
            _ => None,
        };
        #[cfg(not(feature = "cuda"))]
        let fused_gate: Option<Tensor> = None;
        let gated = match (fused_gate, &attn.g_proj, self.config.gating) {
            (Some(gated), _, _) => gated,
            (None, Some(g_proj), LagunaGating::PerHead) => {
                let g = softplus_f32(&g_proj.forward(x)?.to_dtype(DType::F32)?)?;
                let g = g
                    .reshape((1usize, seq, n_q, 1usize))?
                    .to_dtype(attn_out.dtype())?;
                attn_out.broadcast_mul(&g)?
            }
            (None, Some(g_proj), LagunaGating::PerElement) => {
                let g = softplus_f32(&g_proj.forward(x)?.to_dtype(DType::F32)?)?;
                let g = g
                    .reshape((1usize, seq, n_q, head_dim))?
                    .to_dtype(attn_out.dtype())?;
                attn_out.broadcast_mul(&g)?
            }
            _ => attn_out,
        };

        let attn_flat = gated.reshape((1usize, seq, n_q * head_dim))?;
        attn.proj_o(&attn_flat, seq)
    }

    #[cfg(feature = "cuda")]
    pub(crate) fn forward_decode_batched_prepared<C: Gemma4Cache>(
        &self,
        tokens: &Tensor,
        positions: &Tensor,
        caches: &mut [&mut C],
    ) -> Result<Tensor> {
        let b = caches.len();
        ensure!(
            matches!(b, 1 | 2)
                && tokens.dtype() == DType::U32
                && positions.dtype() == DType::I32
                && tokens.elem_count() == b
                && positions.elem_count() == b
                && !self.union_probe.load(Ordering::Relaxed),
            "prepared B2 decode requires one or two CUDA rows and device MoE routing"
        );
        let mut hidden = self
            .embed_tokens(tokens)?
            .reshape((1, b, self.config.hidden_size))?
            .to_dtype(self.dtype)?;
        for (idx, layer) in self.layers.iter().enumerate() {
            let fused_rows_attn = self.attention_forward_prepared_fused_rows(
                idx,
                &hidden,
                positions,
                b,
                false,
                |q, k, v| self.decode_attention_rows(idx, q, k, v, caches, b),
            )?;
            let attn = match fused_rows_attn {
                Some(attn) => attn,
                None => {
                    let normed = layer.input_layernorm.forward(&hidden)?;
                    self.attention_forward_decode_batched(idx, &normed, positions, caches, b)?
                }
            };
            let after = hidden.add(&attn.to_dtype(hidden.dtype())?)?;
            let normed = layer.post_attention_layernorm.forward(&after)?;
            let ffn = match &layer.ffn {
                LagunaFfn::Dense(mlp) => mlp.forward(&normed)?,
                LagunaFfn::Moe(moe) => {
                    let flat = normed.reshape((b, self.config.hidden_size))?.contiguous()?;
                    if let Some(next) =
                        self.moe_forward_prepared_fused_tail(moe, &flat, &after, b, b == 2)?
                    {
                        hidden = next;
                        continue;
                    }
                    self.moe_forward_verify_grouped(idx, moe, &flat, b, b == 2)?
                        .ok_or_else(|| {
                            anyhow::anyhow!(
                                "required prepared B2 grouped MoE unavailable at layer {idx}"
                            )
                        })?
                        .reshape((1, b, self.config.hidden_size))?
                }
            };
            hidden = after.add(&ffn.to_dtype(after.dtype())?)?;
        }
        self.lm_head_from_prenorm_scoped(&hidden, false)?
            .to_dtype(DType::F32)?
            .reshape((b, self.config.vocab_size))
            .map_err(Into::into)
    }

    #[cfg(feature = "cuda")]
    pub(crate) fn forward_verify_batched_prepared(
        &self,
        tokens: &Tensor,
        positions: &Tensor,
        caches: &mut [&mut LagunaKvCacheFp8],
        rows: usize,
        aux_layers: &[usize],
    ) -> Result<(Tensor, Vec<Tensor>)> {
        let b = caches.len();
        let m = b
            .checked_mul(rows)
            .ok_or_else(|| anyhow::anyhow!("batch verify row overflow"))?;
        ensure!(
            matches!(b, 1 | 2)
                && matches!(rows, 1 | 3)
                && m <= MAX_VERIFY_MOE_TOKENS
                && tokens.dtype() == DType::U32
                && positions.dtype() == DType::I32
                && tokens.elem_count() == m
                && positions.elem_count() == m
                && !self.union_probe.load(Ordering::Relaxed),
            "prepared speculative B2 requires CUDA M1/M2/M3/M6 and device MoE routing"
        );
        ensure!(
            !aux_layers.is_empty()
                && aux_layers.iter().all(|&i| i < self.layers.len())
                && aux_layers
                    .iter()
                    .enumerate()
                    .all(|(i, v)| !aux_layers[..i].contains(v)),
            "batch verify auxiliary layers must be unique valid target layers"
        );
        static FUSED_M3_M6: OnceLock<bool> = OnceLock::new();
        let fused_moe = m == 2
            || (matches!(m, 3 | 6)
                && *FUSED_M3_M6.get_or_init(|| {
                    nv_config::env_arm_opt_in_eq_some_1_reread_every_call(
                        "NV_LAGUNA_SPEC_MOE_FUSED",
                    )
                }));
        let mut hidden = self
            .embed_tokens(tokens)?
            .reshape((1, m, self.config.hidden_size))?
            .to_dtype(self.dtype)?;
        let mut aux: Vec<Option<Tensor>> = vec![None; aux_layers.len()];
        for (idx, layer) in self.layers.iter().enumerate() {
            let attn = match self.attention_forward_prepared_fused_rows(
                idx,
                &hidden,
                positions,
                m,
                true,
                |q, k, v| self.verify_attention_rows(idx, q, k, v, caches, b, rows),
            )? {
                Some(attn) => attn,
                None => {
                    let normed = layer.input_layernorm.forward(&hidden)?;
                    self.attention_forward_verify_batched_prepared(
                        idx, &normed, positions, caches, b, rows,
                    )?
                }
            };
            let after = hidden.add(&attn.to_dtype(hidden.dtype())?)?;
            let normed = layer.post_attention_layernorm.forward(&after)?;
            hidden = match &layer.ffn {
                LagunaFfn::Dense(mlp) => {
                    after.add(&mlp.forward(&normed)?.to_dtype(after.dtype())?)?
                }
                LagunaFfn::Moe(moe) => {
                    let flat = normed.reshape((m, self.config.hidden_size))?.contiguous()?;
                    match self.moe_forward_prepared_fused_tail(moe, &flat, &after, m, fused_moe)? {
                        Some(next) => next,
                        None => {
                            let ffn = self.moe_forward_verify_grouped(idx, moe, &flat, m, fused_moe)?
                                .ok_or_else(|| anyhow::anyhow!("required grouped speculative B2 MoE unavailable at layer {idx}, rows={m}"))?
                                .reshape((1, m, self.config.hidden_size))?;
                            after.add(&ffn.to_dtype(after.dtype())?)?
                        }
                    }
                }
            };
            if let Some(slot) = aux_layers.iter().position(|&i| i == idx) {
                aux[slot] = Some(hidden.clone());
            }
        }
        let aux = aux
            .into_iter()
            .map(|v| v.ok_or_else(|| anyhow::anyhow!("missing batch verify auxiliary output")))
            .collect::<Result<Vec<_>>>()?;
        let logits = self
            .lm_head_from_prenorm_scoped(&hidden, true)?
            .to_dtype(DType::F32)?;
        Ok((logits.reshape((1, m, self.config.vocab_size))?, aux))
    }

    pub fn forward_decode_batched<C: Gemma4Cache>(
        &self,
        tokens: &[u32],
        positions: &[usize],
        caches: &mut [&mut C],
    ) -> Result<Tensor> {
        let b = tokens.len();
        if b == 0 {
            bail!("Laguna.forward_decode_batched: empty batch");
        }
        if positions.len() != b || caches.len() != b {
            bail!(
                "Laguna.forward_decode_batched: ragged batch tokens={} positions={} caches={}",
                b,
                positions.len(),
                caches.len()
            );
        }
        self.apply_attn_w8_for(b);

        let tokens_t = Tensor::from_vec(tokens.to_vec(), b, &self.device)?.to_dtype(DType::U32)?;
        let mut hidden = self
            .embed_tokens(&tokens_t)?
            .reshape((1usize, b, self.config.hidden_size))?
            .to_dtype(self.dtype)?;

        for (i, cache) in caches.iter_mut().enumerate() {
            let len = cache.current_len();
            if len != positions[i] {
                bail!(
                    "Laguna.forward_decode_batched: lane {i} cache len {len} != position {}",
                    positions[i]
                );
            }
            cache.prepare_for_decode(len, len + 1)?;
        }

        let positions_t = {
            let p: Vec<i32> = positions.iter().map(|&p| p as i32).collect();
            Tensor::from_vec(p, b, &self.device)?
        };

        for li in 0..self.layers.len() {
            hidden = self.layer_forward_decode_batched(li, &hidden, &positions_t, caches, b)?;
        }

        for cache in caches.iter_mut() {
            cache.advance(1);
        }

        let logits = self.lm_head_from_prenorm_scoped(&hidden, false)?;
        let dims = logits.dims();
        let vocab = dims[dims.len() - 1];
        logits
            .to_dtype(DType::F32)?
            .reshape((b, vocab))
            .map_err(Into::into)
    }

    fn layer_forward_decode_batched<C: Gemma4Cache>(
        &self,
        idx: usize,
        x: &Tensor,
        positions: &Tensor,
        caches: &mut [&mut C],
        b: usize,
    ) -> Result<Tensor> {
        let layer = &self.layers[idx];
        let normed = layer.input_layernorm.forward(x)?;
        let attn_out = self.attention_forward_decode_batched(idx, &normed, positions, caches, b)?;
        let after_attn = x.add(&attn_out.to_dtype(x.dtype())?)?;
        self.ffn_forward_decode_batched(idx, &after_attn, b)
    }

    fn ffn_forward_decode_batched(
        &self,
        idx: usize,
        after_attn: &Tensor,
        b: usize,
    ) -> Result<Tensor> {
        #[cfg(feature = "cuda")]
        if b > 1 && b <= MAX_VERIFY_MOE_TOKENS && self.device.is_cuda() {
            if let LagunaFfn::Moe(moe) = &self.layers[idx].ffn {
                let normed_mlp = self.layers[idx]
                    .post_attention_layernorm
                    .forward(after_attn)?;
                let flat = normed_mlp
                    .reshape((b, self.config.hidden_size))?
                    .contiguous()?;
                let fused_batched_decode = b == 2
                    && nv_config::env_arm_opt_in_eq_some_1_reread_every_call(
                        "NV_LAGUNA_BATCH_MOE_FUSED",
                    );
                match self.moe_forward_verify_grouped(idx, moe, &flat, b, fused_batched_decode) {
                    Ok(Some(out)) => {
                        let out = out.reshape((1usize, b, self.config.hidden_size))?;
                        return after_attn
                            .add(&out.to_dtype(after_attn.dtype())?)
                            .map_err(Into::into);
                    }
                    Ok(None) => {}
                    Err(e) => {
                        if moe.experts.is_empty() {
                            return Err(e)
                                .context("grouped-only batched MoE failed; fallback disabled");
                        }
                        eprintln!(
                            "[laguna] batched decode verify-grouped MoE failed at layer {idx}, standard route: {e:#}"
                        );
                    }
                }
            }
        }
        self.ffn_forward(idx, after_attn, b)
    }

    #[cfg(feature = "cuda")]
    fn b2_fused_tail_opt_in() -> bool {
        static ENABLED: OnceLock<bool> = OnceLock::new();
        *ENABLED.get_or_init(|| {
            env::var("NV_LAGUNA_B2_FUSED_TAIL")
                .map(|v| v == "1")
                .unwrap_or(false)
        })
    }

    #[cfg(feature = "cuda")]
    fn prepared_shared_fold_opt_in() -> bool {
        static LATCHED: OnceLock<bool> = OnceLock::new();
        *LATCHED.get_or_init(|| {
            nv_config::env_arm_opt_in_eq_some_1_reread_every_call("NV_LAGUNA_PREPARED_SHARED_FOLD")
        })
    }

    #[cfg(feature = "cuda")]
    fn moe_forward_prepared_fused_tail(
        &self,
        moe: &LagunaMoe,
        x_flat: &Tensor,
        residual: &Tensor,
        m: usize,
        count_as_fused_dispatch: bool,
    ) -> Result<Option<Tensor>> {
        use crate::laguna_step_graph::{bf16_ptr, wrap_bf16};
        use cudarc::driver::{CudaSlice, DevicePtr, DevicePtrMut};
        use half::bf16;
        if !Self::b2_fused_tail_opt_in() || !matches!(m, 1 | 2 | 3 | 6) {
            return Ok(None);
        }
        let Device::Cuda(dev) = self.device.clone() else {
            return Ok(None);
        };
        let grouped = match self.grouped_weights(moe)? {
            Some(w) => w,
            None => return Ok(None),
        };
        let folds_shared = grouped.folded_shared && Self::prepared_shared_fold_opt_in();
        if !folds_shared {
            return Ok(None);
        }
        let hidden = self.config.hidden_size;
        ensure!(
            x_flat.dtype() == DType::BF16
                && x_flat.is_contiguous()
                && x_flat.elem_count() == m * hidden
                && residual.dtype() == DType::BF16
                && residual.is_contiguous()
                && residual.elem_count() == m * hidden,
            "prepared fused MoE tail requires contiguous BF16 ({m},{hidden}) input and residual"
        );
        static FUSED_TAIL_FIRED_ONCE: Once = Once::new();
        FUSED_TAIL_FIRED_ONCE.call_once(|| eprintln!("[laguna_b2_fused_tail] prepared routed MoE rows={m} through the fused small-M route/chain on the folded bank and moe_unpermute_scatter_tail (residual added in the scatter, no separate f32 copy/cast/add)"));
        let mut map = self
            .moe_verify_ctx
            .lock()
            .map_err(|e| anyhow::anyhow!("moe_verify_ctx mutex poisoned: {e}"))?;
        let stream = nv_layers::cuda_stream::current_stream(&dev);
        let ctx = self.prepared_moe_context(&mut map, m, folds_shared, &dev)?;
        let logits = moe.gate.forward(x_flat)?;
        ensure!(
            logits.dtype() == DType::BF16 && logits.elem_count() == m * self.config.num_experts,
            "prepared fused MoE tail requires BF16 ({m},E) router logits"
        );
        let logits = logits.contiguous()?;
        let bias = moe.selection_bias.to_dtype(DType::F32)?.contiguous()?;
        let logits_ptr = bf16_ptr(&logits, &stream, m * self.config.num_experts)?;
        ctx.cast_router_logits_bf16_into_logits_f32_softcap_applied_at_route_not_here(
            &stream, logits_ptr,
        )?;
        let x_ptr = bf16_ptr(x_flat, &stream, m * hidden)?;
        ctx.stage_x_bf16_rows_dtod(&stream, x_ptr)?;
        let bias_ptr = {
            let (st, l) = bias.storage_and_layout();
            let cuda = match &*st {
                Storage::Cuda(c) => c,
                _ => bail!("prepared fused MoE tail: selection bias not CUDA"),
            };
            let slice = cuda.as_cuda_slice::<f32>()?;
            let view = slice.slice(l.start_offset()..);
            let (p, _g) = view.device_ptr(&stream);
            p
        };
        nv_layers::moe_grouped::forward_grouped_decode_into(
            &grouped,
            ctx,
            bias_ptr,
            1,
            moe.softcap,
            moe.norm_topk,
            moe.routed_scaling,
            false,
            &stream,
            None,
            None,
        )?;
        ensure!(
            ctx.last_call_used_fused_small_m_route_and_chain(),
            "prepared fused MoE tail: the fused small-M route/chain was not selected for rows={m}"
        );
        let zero_shared: CudaSlice<f32> = stream.alloc_zeros::<f32>(m * hidden)?;
        let mut out: CudaSlice<bf16> = unsafe { stream.alloc::<bf16>(m * hidden)? };
        {
            let (zp, _g0) = zero_shared.device_ptr(&stream);
            let resid_ptr = bf16_ptr(residual, &stream, m * hidden)?;
            let (op, _g1) = out.device_ptr_mut(&stream);
            ctx.unpermute_scatter_tail_adding_shared_f32_then_resid_bf16(
                &stream, zp, resid_ptr, op,
            )?;
        }
        if count_as_fused_dispatch {
            self.batch_moe_fused_dispatches
                .fetch_add(1, Ordering::Relaxed);
        }
        Ok(Some(wrap_bf16(out, (1usize, m, hidden), &dev)))
    }

    #[cfg(feature = "cuda")]
    fn b2_fused_rows_opt_in() -> bool {
        static ENABLED: OnceLock<bool> = OnceLock::new();
        *ENABLED.get_or_init(|| {
            env::var("NV_LAGUNA_B2_FUSED_ROWS")
                .map(|v| v == "1")
                .unwrap_or(false)
        })
    }

    #[cfg(feature = "cuda")]
    fn spec_fused_rows_opt_in() -> bool {
        static ENABLED: OnceLock<bool> = OnceLock::new();
        *ENABLED.get_or_init(|| {
            nv_config::env_arm_opt_in_eq_some_1_reread_every_call("NV_LAGUNA_SPEC_FUSED_ROWS")
        })
    }

    #[cfg(feature = "cuda")]
    fn attention_forward_prepared_fused_rows(
        &self,
        layer_idx: usize,
        x_raw: &Tensor,
        positions: &Tensor,
        b: usize,
        speculative: bool,
        attend: impl FnOnce(&Tensor, &Tensor, &Tensor) -> Result<Vec<Tensor>>,
    ) -> Result<Option<Tensor>> {
        use crate::laguna_step_graph::{bf16_ptr, w8_bytes_ptr, w8_scale_ptr, wrap_bf16};
        use cudarc::driver::{CudaSlice, DevicePtr, DevicePtrMut};
        use half::bf16;
        use std::ffi::c_void;
        if !(if speculative {
            Self::spec_fused_rows_opt_in()
        } else {
            Self::b2_fused_rows_opt_in()
        }) {
            return Ok(None);
        }
        let layer = &self.layers[layer_idx];
        let attn = &layer.self_attn;
        let hidden = self.config.hidden_size;
        let hd = self.config.head_dim;
        let n_q = attn.num_heads;
        let n_kv = self.config.num_key_value_heads;
        let Device::Cuda(dev) = self.device.clone() else {
            return Ok(None);
        };
        let Some((pq, pk, pv, is_fp8)) = attn.m1_qkv_w8_fused(hidden, n_q, n_kv, hd) else {
            return Ok(None);
        };
        let g_lin = match (&attn.g_proj, self.config.gating) {
            (Some(g), LagunaGating::PerHead) => g,
            _ => return Ok(None),
        };
        let fp8_gate = attn.fp8_gate_weight_and_scales(hidden, n_q);
        let o8 = match attn.w8.as_ref() {
            Some(w) if attn.m1_w8_active_o() && w.is_fp8() => &w.o,
            _ => return Ok(None),
        };
        let the_rows_kernels_hold_up_to_8_rows_and_98304_dynamic_smem_bytes =
            (1..=8).contains(&b) && b * n_q * hd * 2 <= 98304;
        let shape_ok = is_fp8
            && the_rows_kernels_hold_up_to_8_rows_and_98304_dynamic_smem_bytes
            && hd.is_multiple_of(32)
            && hd <= 256
            && hidden.is_multiple_of(16)
            && hidden <= 4096
            && (fp8_gate.is_some() || g_lin.weight().is_some())
            && (!layer.input_layernorm.fp8_e4m3_active() || fp8_gate.is_some())
            && g_lin.bias().is_none()
            && attn.q_proj.bias().is_none()
            && attn.k_proj.bias().is_none()
            && attn.v_proj.bias().is_none()
            && attn.o_proj.bias().is_none()
            && matches!(o8.bytes(), LagunaProjBytes::E4m3(bytes) if bytes.len() == hidden * n_q * hd)
            && x_raw.dtype() == DType::BF16
            && x_raw.is_contiguous()
            && x_raw.elem_count() == b * hidden
            && positions.dtype() == DType::I32
            && positions.is_contiguous()
            && positions.elem_count() == b;
        if !shape_ok {
            return Ok(None);
        }
        if speculative {
            static FIRED: Once = Once::new();
            FIRED.call_once(|| eprintln!("[laguna_spec_fused_rows] shared prepared attention rows={b} qkvg_normed_rows_rstd_fused=true qk_normrope_rows=true gated_o=true fp8_norm_weights={}", layer.input_layernorm.fp8_e4m3_active()));
        } else {
            static FIRED: Once = Once::new();
            FIRED.call_once(|| eprintln!("[laguna_b2_fused_rows] shared prepared attention rows={b} qkvg_normed_rows_rstd_fused=true qk_normrope_rows=true gated_o=true fp8_norm_weights={}", layer.input_layernorm.fp8_e4m3_active()));
        }
        let s = nv_layers::cuda_stream::current_stream(&dev);
        let x_ptr = bf16_ptr(x_raw, &s, b * hidden)?;
        let mut q_raw: CudaSlice<bf16> = unsafe { s.alloc::<bf16>(b * n_q * hd)? };
        let mut k_raw: CudaSlice<bf16> = unsafe { s.alloc::<bf16>(b * n_kv * hd)? };
        let mut v_dev: CudaSlice<bf16> = unsafe { s.alloc::<bf16>(b * n_kv * hd)? };
        let mut g_dev: CudaSlice<bf16> = unsafe { s.alloc::<bf16>(b * n_q)? };
        {
            let (bq, sq) = (w8_bytes_ptr(pq, &s), w8_scale_ptr(pq, &s));
            let (bk, sk) = (w8_bytes_ptr(pk, &s), w8_scale_ptr(pk, &s));
            let (bv, sv) = (w8_bytes_ptr(pv, &s), w8_scale_ptr(pv, &s));
            let (yq, _g1) = q_raw.device_ptr_mut(&s);
            let (yk, _g2) = k_raw.device_ptr_mut(&s);
            let (yv, _g3) = v_dev.device_ptr_mut(&s);
            let (yg, _g4) = g_dev.device_ptr_mut(&s);
            let rc = unsafe {
                if let Some((gate_weight, gate_scale)) = fp8_gate {
                    let (wg, _gate_read) = gate_weight.device_ptr(&s);
                    let (sg, _scale_read) = gate_scale.device_ptr(&s);
                    if let Some((norm_weight, norm_scale, width)) =
                        layer.input_layernorm.fp8_e4m3_weight_and_scale()
                    {
                        ensure!(width == hidden && norm_weight.len() == hidden && norm_scale.len() == 1,
                            "prepared attention FP8 input norm storage does not match hidden={hidden}");
                        let (wn, _norm_read) = norm_weight.device_ptr(&s);
                        let (sn, _norm_scale_read) = norm_scale.device_ptr(&s);
                        nv_kernels::cuda::laguna_gemv_fp8_qkvg_fp8_norm_rows(
                            s.cu_stream() as *mut c_void,
                            bq as *const u8,
                            sq as *const f32,
                            bk as *const u8,
                            sk as *const f32,
                            bv as *const u8,
                            sv as *const f32,
                            wg as *const u8,
                            sg as *const f32,
                            x_ptr as *const u16,
                            wn as *const u8,
                            sn as *const f32,
                            layer.input_layernorm.eps() as f32,
                            yq as *mut u16,
                            yk as *mut u16,
                            yv as *mut u16,
                            yg as *mut u16,
                            (n_q * hd) as i32,
                            (n_kv * hd) as i32,
                            (n_kv * hd) as i32,
                            n_q as i32,
                            hidden as i32,
                            b as i32,
                        )
                    } else {
                        let wn = bf16_ptr(layer.input_layernorm.weight_bf16(), &s, hidden)?;
                        nv_kernels::cuda::laguna_gemv_fp8_qkvg_normed_rows_rstd_fused(
                            s.cu_stream() as *mut c_void,
                            bq as *const u8,
                            sq as *const f32,
                            bk as *const u8,
                            sk as *const f32,
                            bv as *const u8,
                            sv as *const f32,
                            wg as *const u8,
                            sg as *const f32,
                            x_ptr as *const u16,
                            wn as *const u16,
                            layer.input_layernorm.eps() as f32,
                            yq as *mut u16,
                            yk as *mut u16,
                            yv as *mut u16,
                            yg as *mut u16,
                            (n_q * hd) as i32,
                            (n_kv * hd) as i32,
                            (n_kv * hd) as i32,
                            n_q as i32,
                            hidden as i32,
                            b as i32,
                        )
                    }
                } else {
                    let wn = bf16_ptr(layer.input_layernorm.weight_bf16(), &s, hidden)?;
                    let wg = bf16_ptr(g_lin.weight().unwrap(), &s, n_q * hidden)?;
                    nv_kernels::cuda::laguna_gemv_q8_qkvg_normed_rows_rstd_fused(
                        s.cu_stream() as *mut c_void,
                        1,
                        bq as *const c_void,
                        sq as *const f32,
                        bk as *const c_void,
                        sk as *const f32,
                        bv as *const c_void,
                        sv as *const f32,
                        wg as *const u16,
                        x_ptr as *const u16,
                        wn as *const u16,
                        layer.input_layernorm.eps() as f32,
                        yq as *mut u16,
                        yk as *mut u16,
                        yv as *mut u16,
                        yg as *mut u16,
                        (n_q * hd) as i32,
                        (n_kv * hd) as i32,
                        (n_kv * hd) as i32,
                        n_q as i32,
                        hidden as i32,
                        b as i32,
                    )
                }
            };
            ensure!(
                rc == 0,
                "prepared B2 fused rows qkvg gemv rc={rc} layer {layer_idx} rows {b}"
            );
        }
        let (rope, rotary_dim, rot_scale) = self.rope_for_kind(attn.kind);
        let cos_c = rope.cos().contiguous()?;
        let sin_c = rope.sin().contiguous()?;
        let f32_ptr = |t: &Tensor| -> Result<u64> {
            let (st, l) = t.storage_and_layout();
            let cuda = match &*st {
                Storage::Cuda(c) => c,
                _ => bail!("prepared B2 fused rows: rope table not CUDA"),
            };
            let slice = cuda.as_cuda_slice::<f32>()?;
            let view = slice.slice(l.start_offset()..);
            let (p, _g) = view.device_ptr(&s);
            Ok(p)
        };
        let cos_ptr = f32_ptr(&cos_c)?;
        let sin_ptr = f32_ptr(&sin_c)?;
        let pos_ptr = {
            let (st, l) = positions.storage_and_layout();
            let cuda = match &*st {
                Storage::Cuda(c) => c,
                _ => bail!("prepared B2 fused rows: positions not CUDA"),
            };
            let slice = cuda.as_cuda_slice::<i32>()?;
            let view = slice.slice(l.start_offset()..l.start_offset() + b);
            let (p, _g) = view.device_ptr(&s);
            p
        };
        let mut q_rot: CudaSlice<bf16> = unsafe { s.alloc::<bf16>(b * n_q * hd)? };
        let mut k_rot: CudaSlice<bf16> = unsafe { s.alloc::<bf16>(b * n_kv * hd)? };
        {
            let (qi, _g1) = q_raw.device_ptr(&s);
            let (ki, _g2) = k_raw.device_ptr(&s);
            let (qo, _g3) = q_rot.device_ptr_mut(&s);
            let (ko, _g4) = k_rot.device_ptr_mut(&s);
            let rc = unsafe {
                match (
                    attn.q_norm.fp8_e4m3_weight_and_scale(),
                    attn.k_norm.fp8_e4m3_weight_and_scale(),
                ) {
                    (Some((qw, qs, qh)), Some((kw, ks, kh))) => {
                        ensure!(
                            qh == hd
                                && kh == hd
                                && qw.len() == hd
                                && kw.len() == hd
                                && qs.len() == 1
                                && ks.len() == 1,
                            "prepared attention FP8 Q/K norm storage does not match head_dim={hd}"
                        );
                        let (qw, _qr) = qw.device_ptr(&s);
                        let (qs, _qsr) = qs.device_ptr(&s);
                        let (kw, _kr) = kw.device_ptr(&s);
                        let (ks, _ksr) = ks.device_ptr(&s);
                        nv_kernels::cuda::laguna_qk_normrope_fp8_weight_rows(
                            s.cu_stream() as *mut c_void,
                            qi as *const u16,
                            ki as *const u16,
                            qo as *mut u16,
                            ko as *mut u16,
                            qw as *const u8,
                            qs as *const f32,
                            kw as *const u8,
                            ks as *const f32,
                            cos_ptr as *const f32,
                            sin_ptr as *const f32,
                            pos_ptr as *const i32,
                            n_q as i32,
                            n_kv as i32,
                            hd as i32,
                            rotary_dim as i32,
                            rot_scale,
                            attn.q_norm.eps() as f32,
                            attn.k_norm.eps() as f32,
                            b as i32,
                        )
                    }
                    (None, None) => {
                        let qw = bf16_ptr(attn.q_norm.weight_bf16(), &s, hd)?;
                        let kw = bf16_ptr(attn.k_norm.weight_bf16(), &s, hd)?;
                        nv_kernels::cuda::laguna_qk_normrope_bf16_rows(
                            s.cu_stream() as *mut c_void,
                            qi as *const u16,
                            ki as *const u16,
                            qo as *mut u16,
                            ko as *mut u16,
                            qw as *const u16,
                            kw as *const u16,
                            cos_ptr as *const f32,
                            sin_ptr as *const f32,
                            pos_ptr as *const i32,
                            n_q as i32,
                            n_kv as i32,
                            hd as i32,
                            rotary_dim as i32,
                            rot_scale,
                            attn.q_norm.eps() as f32,
                            attn.k_norm.eps() as f32,
                            b as i32,
                        )
                    }
                    _ => bail!("prepared attention Q/K norms must use the same weight format"),
                }
            };
            ensure!(
                rc == 0,
                "prepared fused rows normrope rc={rc} layer {layer_idx} rows={b}"
            );
        }
        let q_t = wrap_bf16(q_rot, (1usize, b, n_q * hd), &dev).reshape((1usize, b, n_q, hd))?;
        let k_t = wrap_bf16(k_rot, (1usize, b, n_kv * hd), &dev).reshape((1usize, b, n_kv, hd))?;
        let v_t = wrap_bf16(v_dev, (1usize, b, n_kv * hd), &dev).reshape((1usize, b, n_kv, hd))?;
        let mut rows = attend(&q_t, &k_t, &v_t)?;
        let attn_out = if rows.len() == 1 && (speculative || Self::b2_fused_tail_opt_in()) {
            rows.pop().unwrap()
        } else {
            let row_refs: Vec<&Tensor> = rows.iter().collect();
            Tensor::cat(&row_refs, 1)?
        };
        let attn_out = if attn_out.dtype() == DType::BF16 {
            attn_out.contiguous()?
        } else {
            attn_out.to_dtype(DType::BF16)?.contiguous()?
        };
        let a_ptr = bf16_ptr(&attn_out, &s, b * n_q * hd)?;
        let mut o_out: CudaSlice<bf16> = unsafe { s.alloc::<bf16>(b * hidden)? };
        {
            let (wo, so) = (w8_bytes_ptr(o8, &s), w8_scale_ptr(o8, &s));
            let (gp, _g0) = g_dev.device_ptr(&s);
            let (yo, _g1) = o_out.device_ptr_mut(&s);
            let rc = unsafe {
                nv_kernels::cuda::laguna_gemv_q8_rows_gated(
                    s.cu_stream() as *mut c_void,
                    1,
                    wo as *const c_void,
                    so as *const f32,
                    a_ptr as *const u16,
                    gp as *const u16,
                    n_q as i32,
                    hd as i32,
                    yo as *mut u16,
                    hidden as i32,
                    (n_q * hd) as i32,
                    b as i32,
                )
            };
            ensure!(
                rc == 0,
                "prepared B2 fused rows gated o_proj gemv rc={rc} layer {layer_idx} rows {b}"
            );
        }
        Ok(Some(wrap_bf16(o_out, (1usize, b, hidden), &dev)))
    }

    fn decode_attention_rows<C: Gemma4Cache>(
        &self,
        layer_idx: usize,
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        caches: &mut [&mut C],
        b: usize,
    ) -> Result<Vec<Tensor>> {
        let attn = &self.layers[layer_idx].self_attn;
        let head_dim = self.config.head_dim;
        let n_q = attn.num_heads;
        let n_kv = self.config.num_key_value_heads;
        let window = match attn.kind {
            LayerType::SlidingAttention => Some(self.config.sliding_window),
            LayerType::FullAttention => None,
        };
        let scale = (head_dim as f32).powf(-0.5);

        let mut rows: Vec<Tensor> = Vec::with_capacity(b);
        for i in 0..b {
            let q_i = q.narrow(1, i, 1)?.contiguous()?;
            let k_i = k.narrow(1, i, 1)?.contiguous()?;
            let v_i = v.narrow(1, i, 1)?.contiguous()?;

            let cache = &mut *caches[i];
            cache.write_at(layer_idx, &k_i, &v_i)?;
            let total = cache.current_len() + 1;

            let mut fast_path_out =
                cache.try_decode_attention_fp8(layer_idx, &q_i, n_q, window, scale)?;
            #[cfg(feature = "cuda")]
            if fast_path_out.is_none() {
                fast_path_out =
                    cache.try_decode_attention_gqa(layer_idx, &q_i, n_q, window, scale)?;
            }
            if fast_path_out.is_none() {
                fast_path_out =
                    cache.try_decode_attention_ring(layer_idx, &q_i, n_q, window, scale)?;
            }
            let out_i = if let Some(out) = fast_path_out {
                out
            } else {
                let (k_full, v_full) = cache.view(layer_idx, total)?;
                attention(
                    &q_i, &k_full, &v_full, n_q, n_kv, head_dim, 1, scale, window,
                )?
            };
            rows.push(out_i.reshape((1usize, 1usize, n_q, head_dim))?);
        }
        Ok(rows)
    }

    fn attention_forward_decode_batched<C: Gemma4Cache>(
        &self,
        layer_idx: usize,
        x: &Tensor,
        positions: &Tensor,
        caches: &mut [&mut C],
        b: usize,
    ) -> Result<Tensor> {
        let layer = &self.layers[layer_idx];
        let attn = &layer.self_attn;
        let kind = attn.kind;
        let head_dim = self.config.head_dim;
        let n_q = attn.num_heads;
        let n_kv = self.config.num_key_value_heads;

        let q_raw = attn.proj_q(x, b)?;
        let q = q_raw.reshape((1usize, b, n_q, head_dim))?;
        let q_normed = attn.q_norm.forward(&q)?;
        let k_raw = attn.proj_k(x, b)?;
        let k = k_raw.reshape((1usize, b, n_kv, head_dim))?;
        let k_normed = attn.k_norm.forward(&k)?;
        let v = attn.proj_v(x, b)?.reshape((1usize, b, n_kv, head_dim))?;

        let (rope, rotary_dim, rot_scale) = self.rope_for_kind(kind);
        let (q_rot, k_rot) = apply_partial_rope_scaled(
            &q_normed, &k_normed, rope, positions, rotary_dim, head_dim, rot_scale,
        )?;
        let q_rot = q_rot.to_dtype(self.dtype)?.contiguous()?;
        let k_rot = k_rot.to_dtype(self.dtype)?.contiguous()?;
        let v_for_cache = v.contiguous()?;

        let rows =
            self.decode_attention_rows(layer_idx, &q_rot, &k_rot, &v_for_cache, caches, b)?;
        let row_refs: Vec<&Tensor> = rows.iter().collect();
        let attn_out = Tensor::cat(&row_refs, 1)?;

        let gated = match (&attn.g_proj, self.config.gating) {
            (Some(g_proj), gating @ (LagunaGating::PerHead | LagunaGating::PerElement)) => {
                let width = if matches!(gating, LagunaGating::PerHead) {
                    1
                } else {
                    head_dim
                };
                let g = softplus_f32(&g_proj.forward(x)?.to_dtype(DType::F32)?)?;
                let g = g
                    .reshape((1usize, b, n_q, width))?
                    .to_dtype(attn_out.dtype())?;
                attn_out.broadcast_mul(&g)?
            }
            _ => attn_out,
        };

        let attn_flat = gated.reshape((1usize, b, n_q * head_dim))?;
        attn.proj_o(&attn_flat, b)
    }

    #[cfg(feature = "cuda")]
    fn verify_attention_rows(
        &self,
        layer_idx: usize,
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        caches: &mut [&mut LagunaKvCacheFp8],
        b: usize,
        rows_per_lane: usize,
    ) -> Result<Vec<Tensor>> {
        let attn = &self.layers[layer_idx].self_attn;
        let head_dim = self.config.head_dim;
        let n_q = attn.num_heads;
        let window = match attn.kind {
            LayerType::SlidingAttention => Some(self.config.sliding_window),
            LayerType::FullAttention => None,
        };
        let scale = (head_dim as f32).powf(-0.5);
        let mut rows = Vec::with_capacity(b);
        for i in 0..b {
            let q_i = q
                .narrow(1, i * rows_per_lane, rows_per_lane)?
                .contiguous()?;
            let k_i = k
                .narrow(1, i * rows_per_lane, rows_per_lane)?
                .contiguous()?;
            let v_i = v
                .narrow(1, i * rows_per_lane, rows_per_lane)?
                .contiguous()?;
            let cache = &mut *caches[i];
            cache.write_at(layer_idx, &k_i, &v_i)?;
            let out_i = cache.prepared_batch_verify_attention(
                layer_idx,
                &q_i,
                n_q,
                window,
                scale,
                rows_per_lane,
            )?;
            rows.push(out_i.reshape((1usize, rows_per_lane, n_q, head_dim))?);
        }
        Ok(rows)
    }

    #[cfg(feature = "cuda")]
    fn attention_forward_verify_batched_prepared(
        &self,
        layer_idx: usize,
        x: &Tensor,
        positions: &Tensor,
        caches: &mut [&mut LagunaKvCacheFp8],
        b: usize,
        rows_per_lane: usize,
    ) -> Result<Tensor> {
        let m = b
            .checked_mul(rows_per_lane)
            .ok_or_else(|| anyhow::anyhow!("verify row overflow"))?;
        let layer = &self.layers[layer_idx];
        let attn = &layer.self_attn;
        let kind = attn.kind;
        let head_dim = self.config.head_dim;
        let n_q = attn.num_heads;
        let n_kv = self.config.num_key_value_heads;

        let q_raw = attn.proj_q(x, m)?;
        let q = q_raw.reshape((1usize, m, n_q, head_dim))?;
        let q_normed = attn.q_norm.forward(&q)?;
        let k_raw = attn.proj_k(x, m)?;
        let k = k_raw.reshape((1usize, m, n_kv, head_dim))?;
        let k_normed = attn.k_norm.forward(&k)?;
        let v = attn.proj_v(x, m)?.reshape((1usize, m, n_kv, head_dim))?;

        let (rope, rotary_dim, rot_scale) = self.rope_for_kind(kind);
        let (q_rot, k_rot) = apply_partial_rope_scaled(
            &q_normed, &k_normed, rope, positions, rotary_dim, head_dim, rot_scale,
        )?;
        let q_rot = q_rot.to_dtype(self.dtype)?.contiguous()?;
        let k_rot = k_rot.to_dtype(self.dtype)?.contiguous()?;
        let v_for_cache = v.contiguous()?;

        let rows = self.verify_attention_rows(
            layer_idx,
            &q_rot,
            &k_rot,
            &v_for_cache,
            caches,
            b,
            rows_per_lane,
        )?;
        let row_refs: Vec<&Tensor> = rows.iter().collect();
        let attn_out = Tensor::cat(&row_refs, 1)?;

        let gated = match (&attn.g_proj, self.config.gating) {
            (Some(g_proj), gating @ (LagunaGating::PerHead | LagunaGating::PerElement)) => {
                let width = if matches!(gating, LagunaGating::PerHead) {
                    1
                } else {
                    head_dim
                };
                let g = softplus_f32(&g_proj.forward(x)?.to_dtype(DType::F32)?)?;
                let g = g
                    .reshape((1usize, m, n_q, width))?
                    .to_dtype(attn_out.dtype())?;
                attn_out.broadcast_mul(&g)?
            }
            _ => attn_out,
        };

        let attn_flat = gated.reshape((1usize, m, n_q * head_dim))?;
        attn.proj_o(&attn_flat, m)
    }
}

pub trait LagunaEmbedding {
    fn lookup(&self, tokens: &Tensor) -> Result<Tensor>;
    fn lookup_captured(&self, tokens: &Tensor, device: &Device) -> Result<Tensor>;
}

impl LagunaEmbedding for Tensor {
    fn lookup(&self, tokens: &Tensor) -> Result<Tensor> {
        self.index_select(tokens, 0).map_err(Into::into)
    }
    fn lookup_captured(&self, tokens: &Tensor, device: &Device) -> Result<Tensor> {
        crate::gemma4::embed_lookup_bf16_op(self, tokens, device)
    }
}

impl LagunaEmbedding for Laguna {
    fn lookup(&self, tokens: &Tensor) -> Result<Tensor> {
        self.embed_tokens(tokens)
    }
    fn lookup_captured(&self, tokens: &Tensor, device: &Device) -> Result<Tensor> {
        #[cfg(feature = "cuda")]
        if self.embed_fp8.is_some() {
            return self.embed_tokens(tokens);
        }
        self.embed_weight().lookup_captured(tokens, device)
    }
}

impl LagunaLayer {
    fn from_loader(
        config: &LagunaConfig,
        idx: usize,
        weights: &WeightLoader,
        qconfig: &QuantizationConfig,
        nvfp4_runner: &Nvfp4RunnerHandle,
        fp8_runner: &Fp8RunnerHandle,
        device: &Device,
        dtype: DType,
    ) -> Result<Self> {
        let prefix = format!("model.layers.{idx}");
        let kind = config.layer_kind(idx);
        let n_q = config.num_heads_for_layer(idx);
        let n_kv = config.num_key_value_heads;
        let head_dim = config.head_dim;
        let hidden = config.hidden_size;

        let input_layernorm = load_laguna_rmsnorm(
            weights,
            &format!("{prefix}.input_layernorm.weight"),
            hidden,
            config.rms_norm_eps,
            dtype,
            fp8_runner.is_some(),
        )?;
        let post_attention_layernorm = load_laguna_rmsnorm(
            weights,
            &format!("{prefix}.post_attention_layernorm.weight"),
            hidden,
            config.rms_norm_eps,
            dtype,
            fp8_runner.is_some(),
        )?;

        let q_weight = load_linear_weight(
            weights,
            &format!("{prefix}.self_attn.q_proj.weight"),
            n_q * head_dim,
            hidden,
            dtype,
        )?;
        let k_weight = load_linear_weight(
            weights,
            &format!("{prefix}.self_attn.k_proj.weight"),
            n_kv * head_dim,
            hidden,
            dtype,
        )?;
        let v_weight = load_linear_weight(
            weights,
            &format!("{prefix}.self_attn.v_proj.weight"),
            n_kv * head_dim,
            hidden,
            dtype,
        )?;
        let o_weight = load_linear_weight(
            weights,
            &format!("{prefix}.self_attn.o_proj.weight"),
            hidden,
            n_q * head_dim,
            dtype,
        )?;
        let g_proj = match config.gating {
            LagunaGating::None => None,
            LagunaGating::PerHead => Some(load_linear_plain(
                weights,
                &format!("{prefix}.self_attn.g_proj.weight"),
                n_q,
                hidden,
                dtype,
                fp8_runner,
                device,
            )?),
            LagunaGating::PerElement => Some(load_linear_plain(
                weights,
                &format!("{prefix}.self_attn.g_proj.weight"),
                n_q * head_dim,
                hidden,
                dtype,
                fp8_runner,
                device,
            )?),
        };
        let q_norm = load_laguna_rmsnorm(
            weights,
            &format!("{prefix}.self_attn.q_norm.weight"),
            head_dim,
            config.rms_norm_eps,
            dtype,
            fp8_runner.is_some(),
        )?;
        let k_norm = load_laguna_rmsnorm(
            weights,
            &format!("{prefix}.self_attn.k_norm.weight"),
            head_dim,
            config.rms_norm_eps,
            dtype,
            fp8_runner.is_some(),
        )?;

        #[cfg(feature = "cuda")]
        let w8 = {
            let fp8 =
                !attn_w8_int8_arm_was_chosen_explicitly_and_therefore_outranks_the_fp8_default();
            if device.is_cuda() && dtype == DType::BF16 {
                let gate_weight_for_the_fused_marlin_twin = match config.gating {
                    LagunaGating::PerHead if fp8 && nv_layers::linear::marlin_fp8_rows_env_on() => {
                        Some(load_linear_weight(
                            weights,
                            &format!("{prefix}.self_attn.g_proj.weight"),
                            n_q,
                            hidden,
                            dtype,
                        )?)
                    }
                    _ => None,
                };
                LagunaAttnW8::build_from_weights(
                    &q_weight,
                    &k_weight,
                    &v_weight,
                    &o_weight,
                    gate_weight_for_the_fused_marlin_twin.as_ref(),
                    device,
                    fp8,
                )?
            } else {
                None
            }
        };

        let q_proj = load_linear_from_tensor_maybe_fp8(q_weight, fp8_runner, device)?;
        let k_proj = load_linear_from_tensor_maybe_fp8(k_weight, fp8_runner, device)?;
        let v_proj = load_linear_from_tensor_maybe_fp8(v_weight, fp8_runner, device)?;
        let o_proj = load_linear_from_tensor_maybe_fp8(o_weight, fp8_runner, device)?;

        let self_attn = LagunaAttention {
            kind,
            num_heads: n_q,
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            g_proj,
            q_norm,
            k_norm,
            #[cfg(feature = "cuda")]
            w8,
        };

        let ffn = if config.is_moe_layer(idx) {
            let gate = load_linear_plain(
                weights,
                &format!("{prefix}.mlp.gate.weight"),
                config.num_experts,
                hidden,
                dtype,
                fp8_runner,
                device,
            )?;
            let bias_name = format!("{prefix}.mlp.experts.e_score_correction_bias");
            let selection_bias = if weights.has(&bias_name) {
                weights
                    .get(&bias_name, DType::F32)?
                    .reshape((1usize, config.num_experts))?
                    .to_device(device)?
            } else {
                Tensor::zeros((1usize, config.num_experts), DType::F32, device)?
            };
            let mut experts = Vec::with_capacity(config.num_experts);
            for e in 0..config.num_experts {
                let eprefix = format!("{prefix}.mlp.experts.{e}");
                let gate_proj = load_linear_expert(
                    weights,
                    &format!("{eprefix}.gate_proj"),
                    config.moe_intermediate_size,
                    hidden,
                    dtype,
                    qconfig,
                    nvfp4_runner,
                    device,
                )?;
                let up_proj = load_linear_expert(
                    weights,
                    &format!("{eprefix}.up_proj"),
                    config.moe_intermediate_size,
                    hidden,
                    dtype,
                    qconfig,
                    nvfp4_runner,
                    device,
                )?;
                let down_proj = load_linear_expert(
                    weights,
                    &format!("{eprefix}.down_proj"),
                    hidden,
                    config.moe_intermediate_size,
                    dtype,
                    qconfig,
                    nvfp4_runner,
                    device,
                )?;
                experts.push(Mlp::new(gate_proj, up_proj, down_proj)?);
            }
            let sprefix = format!("{prefix}.mlp.shared_expert");
            #[cfg_attr(not(feature = "cuda"), allow(unused_mut))]
            let mut se_gate = load_linear_expert(
                weights,
                &format!("{sprefix}.gate_proj"),
                config.shared_expert_intermediate_size,
                hidden,
                dtype,
                qconfig,
                nvfp4_runner,
                device,
            )?;
            #[cfg_attr(not(feature = "cuda"), allow(unused_mut))]
            let mut se_up = load_linear_expert(
                weights,
                &format!("{sprefix}.up_proj"),
                config.shared_expert_intermediate_size,
                hidden,
                dtype,
                qconfig,
                nvfp4_runner,
                device,
            )?;
            #[cfg_attr(not(feature = "cuda"), allow(unused_mut))]
            let mut se_down = load_linear_expert(
                weights,
                &format!("{sprefix}.down_proj"),
                hidden,
                config.shared_expert_intermediate_size,
                dtype,
                qconfig,
                nvfp4_runner,
                device,
            )?;
            #[cfg(feature = "cuda")]
            if let Some(rows) = shared_expert_rows_gemv_opt_in_row_limit() {
                for (name, lin) in [
                    ("gate", &mut se_gate),
                    ("up", &mut se_up),
                    ("down", &mut se_down),
                ] {
                    lin.set_nvfp4_bf16act_rows_before_sharing(rows)
                        .with_context(|| format!("{sprefix}.{name}_proj: NV_LAGUNA_SHARED_GEMV needs NVFP4 shared-expert weights"))?;
                }
            }
            LagunaFfn::Moe(LagunaMoe {
                num_experts: config.num_experts,
                top_k: config.num_experts_per_tok,
                norm_topk: config.norm_topk_prob,
                routed_scaling: config.moe_routed_scaling_factor,
                softcap: config.moe_router_logit_softcapping,
                gate,
                selection_bias,
                experts,
                shared_expert: Mlp::new(se_gate, se_up, se_down)?,
                #[cfg(feature = "cuda")]
                grouped: std::sync::Mutex::new(None),
                #[cfg(feature = "cuda")]
                train_stacks: std::sync::Mutex::new(None),
            })
        } else {
            let gate_proj = load_linear_plain(
                weights,
                &format!("{prefix}.mlp.gate_proj.weight"),
                config.intermediate_size,
                hidden,
                dtype,
                fp8_runner,
                device,
            )?;
            let up_proj = load_linear_plain(
                weights,
                &format!("{prefix}.mlp.up_proj.weight"),
                config.intermediate_size,
                hidden,
                dtype,
                fp8_runner,
                device,
            )?;
            let down_proj = load_linear_plain(
                weights,
                &format!("{prefix}.mlp.down_proj.weight"),
                hidden,
                config.intermediate_size,
                dtype,
                fp8_runner,
                device,
            )?;
            LagunaFfn::Dense(Mlp::new(gate_proj, up_proj, down_proj)?)
        };

        Ok(Self {
            kind,
            input_layernorm,
            post_attention_layernorm,
            self_attn,
            ffn,
        })
    }
}

pub(crate) const SLIDING_COMPACT_SLACK: usize = 256;

pub(crate) fn sliding_write_at(
    k_buf: &Tensor,
    v_buf: &Tensor,
    stored: usize,
    window: usize,
    cap: usize,
    n_kv: usize,
    head_dim: usize,
    k_new: &Tensor,
    v_new: &Tensor,
    layer: usize,
) -> Result<(Tensor, Tensor, usize)> {
    let t = k_new.dims()[1];
    let mut stored = stored;
    let mut k_buf = k_buf.clone();
    let mut v_buf = v_buf.clone();
    if stored + t > cap {
        let keep = stored.min(window);
        let src_start = stored - keep;
        if src_start > 0 && keep > 0 {
            let k_keep = k_buf.narrow(1, src_start, keep)?.contiguous()?;
            let v_keep = v_buf.narrow(1, src_start, keep)?.contiguous()?;
            k_buf = k_buf.slice_assign(&[0..1, 0..keep, 0..n_kv, 0..head_dim], &k_keep)?;
            v_buf = v_buf.slice_assign(&[0..1, 0..keep, 0..n_kv, 0..head_dim], &v_keep)?;
        }
        stored = keep;
    }
    if stored + t > cap {
        bail!("laguna kv: sliding layer {layer} write of {t} tokens exceeds capacity {cap}");
    }
    let end = stored + t;
    let k_updated = k_buf.slice_assign(&[0..1, stored..end, 0..n_kv, 0..head_dim], k_new)?;
    let v_updated = v_buf.slice_assign(&[0..1, stored..end, 0..n_kv, 0..head_dim], v_new)?;
    Ok((k_updated, v_updated, end))
}

#[cfg(feature = "cuda")]
struct LagunaKvRing {
    dev: candle_core::CudaDevice,
    meta_dev: cudarc::driver::CudaSlice<i32>,
    host_meta: Box<[i32; 4]>,
    s_stored: usize,
    full_stored: usize,
    s_cap: usize,
    s_window: usize,

    ring_attn: bool,
}

#[cfg(feature = "cuda")]
pub(crate) struct LagunaBf16Handoff<'a> {
    pub layers: &'a [(Tensor, Tensor)],
    pub windows: &'a [Option<usize>],
    pub frontier: usize,
    pub sliding_stored: usize,
    pub sliding_cap: usize,
    pub sliding_window: usize,
    pub max_seq_len: usize,
    pub n_kv: usize,
    pub head_dim: usize,
}

include!("laguna_study_capture.rs");

pub struct LagunaKvCache {
    layers: Vec<(Tensor, Tensor)>,
    layer_caps: Vec<usize>,
    layer_windows: Vec<Option<usize>>,
    layer_stored: Vec<usize>,
    n_kv: usize,
    head_dim: usize,
    current_len: usize,
    pending_write_pos: usize,
    max_seq_len: usize,
    #[cfg(feature = "cuda")]
    ring: Option<LagunaKvRing>,
    #[cfg(feature = "cuda")]
    gqa_bufs: Option<(
        cudarc::driver::CudaSlice<f32>,
        cudarc::driver::CudaSlice<u32>,
    )>,
    #[cfg(feature = "cuda")]
    prefill_attn: Option<crate::laguna_prefill_attn::LagunaPrefillAttention>,
}

impl LagunaKvCache {
    pub fn new(
        config: &LagunaConfig,
        max_seq_len: usize,
        device: &Device,
        dtype: DType,
    ) -> Result<Self> {
        Self::new_with_prefill_chunk(config, max_seq_len, device, dtype, SLIDING_COMPACT_SLACK)
    }

    pub(crate) fn new_with_prefill_chunk(
        config: &LagunaConfig,
        max_seq_len: usize,
        device: &Device,
        dtype: DType,
        prefill_chunk: usize,
    ) -> Result<Self> {
        #[cfg(feature = "cuda")]
        crate::laguna_fp8::LagunaFp8ScalePolicy::require_bf16_cache_compatible()?;
        Self::allocate_with_prefill_chunk(config, max_seq_len, device, dtype, prefill_chunk)
    }

    fn allocate_with_prefill_chunk(
        config: &LagunaConfig,
        max_seq_len: usize,
        device: &Device,
        dtype: DType,
        prefill_chunk: usize,
    ) -> Result<Self> {
        let ring =
            cfg!(feature = "cuda") && device.is_cuda() && dtype == DType::BF16;
        Self::new_with_mode_and_prefill_chunk(
            config,
            max_seq_len,
            device,
            dtype,
            ring,
            prefill_chunk,
        )
    }

    pub fn new_with_mode(
        config: &LagunaConfig,
        max_seq_len: usize,
        device: &Device,
        dtype: DType,
        ring: bool,
    ) -> Result<Self> {
        #[cfg(feature = "cuda")]
        crate::laguna_fp8::LagunaFp8ScalePolicy::require_bf16_cache_compatible()?;
        Self::new_with_mode_and_prefill_chunk(
            config,
            max_seq_len,
            device,
            dtype,
            ring,
            SLIDING_COMPACT_SLACK,
        )
    }

    fn new_with_mode_and_prefill_chunk(
        config: &LagunaConfig,
        max_seq_len: usize,
        device: &Device,
        dtype: DType,
        ring: bool,
        prefill_chunk: usize,
    ) -> Result<Self> {
        ensure!(
            matches!(prefill_chunk, 256 | 512 | 1024 | 2048 | 4096 | 8192),
            "invalid Laguna prefill capacity {prefill_chunk}"
        );
        let n_kv = config.num_key_value_heads;
        let head_dim = config.head_dim;
        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        let mut layer_caps = Vec::with_capacity(config.num_hidden_layers);
        let mut layer_windows = Vec::with_capacity(config.num_hidden_layers);
        for kind in &config.layer_types {
            let (cap, window) = match kind {
                LayerType::FullAttention => (max_seq_len, None),
                LayerType::SlidingAttention => {
                    let w = config.sliding_window.max(1);
                    (
                        max_seq_len.min(
                            w.checked_add(prefill_chunk)
                                .context("Laguna sliding capacity overflow")?,
                        ),
                        Some(w),
                    )
                }
            };
            let shape = (1usize, cap, n_kv, head_dim);
            let k = Tensor::zeros(shape, dtype, device)?;
            let v = Tensor::zeros(shape, dtype, device)?;
            layers.push((k, v));
            layer_caps.push(cap);
            layer_windows.push(window);
        }
        let layer_stored = vec![0usize; layers.len()];

        #[cfg(feature = "cuda")]
        let ring_state = if ring {
            let Device::Cuda(dev) = device.clone() else {
                bail!("LagunaKvCache ring mode requires a CUDA device")
            };
            if dtype != DType::BF16 {
                bail!("LagunaKvCache ring mode requires BF16, got {dtype:?}");
            }
            let mut s_cap = 0usize;
            let mut s_window = 0usize;
            for (cap, window) in layer_caps.iter().zip(layer_windows.iter()) {
                if let Some(w) = window {
                    if s_cap != 0 && (s_cap != *cap || s_window != *w) {
                        bail!(
                            "LagunaKvCache ring mode requires uniform sliding layers \
                             (cap {}/{} window {}/{})",
                            s_cap,
                            cap,
                            s_window,
                            w
                        );
                    }
                    s_cap = *cap;
                    s_window = *w;
                }
            }
            let stream = nv_layers::cuda_stream::current_stream(&dev);
            let meta_dev = stream.alloc_zeros::<i32>(4)?;
            Some(LagunaKvRing {
                dev,
                meta_dev,
                host_meta: Box::new([0i32; 4]),
                s_stored: 0,
                full_stored: 0,
                s_cap,
                s_window,
                ring_attn: nv_config::env_arm_present_at_all_reread_every_call(
                    "NV_LAGUNA_RING_ATTN",
                ),
            })
        } else {
            None
        };
        #[cfg(not(feature = "cuda"))]
        if ring {
            bail!("LagunaKvCache ring mode requires the cuda feature");
        }

        Ok(Self {
            layers,
            layer_caps,
            layer_windows,
            layer_stored,
            n_kv,
            head_dim,
            current_len: 0,
            pending_write_pos: 0,
            max_seq_len,
            #[cfg(feature = "cuda")]
            ring: ring_state,
            #[cfg(feature = "cuda")]
            gqa_bufs: None,
            #[cfg(feature = "cuda")]
            prefill_attn: None,
        })
    }

    pub fn reset(&mut self) {
        self.current_len = 0;
        for s in self.layer_stored.iter_mut() {
            *s = 0;
        }
        #[cfg(feature = "cuda")]
        if let Some(ring) = self.ring.as_mut() {
            ring.s_stored = 0;
            ring.full_stored = 0;
            *ring.host_meta = [0i32; 4];
        }
    }

    #[cfg(feature = "cuda")]
    pub(crate) fn checked_bf16_handoff(&self) -> Result<LagunaBf16Handoff<'_>> {
        let ring = self
            .ring
            .as_ref()
            .context("BF16 handoff requires a CUDA ring cache")?;
        ensure!(
            !ring.ring_attn,
            "BF16 handoff requires compacting sliding storage; NV_LAGUNA_RING_ATTN is unsupported"
        );
        ensure!(
            self.current_len > 0
                && self.current_len <= self.max_seq_len
                && ring.full_stored == self.current_len
                && ring.s_stored <= ring.s_cap
                && ring.s_stored <= self.current_len,
            "BF16 handoff requires a completed, nonempty cache frontier"
        );
        ensure!(
            self.layers.len() == self.layer_windows.len()
                && self.layers.len() == self.layer_stored.len()
                && self.layers.len() == self.layer_caps.len(),
            "BF16 handoff layer metadata mismatch"
        );
        for (layer, window) in self.layer_windows.iter().enumerate() {
            let stored = if window.is_some() {
                ring.s_stored
            } else {
                ring.full_stored
            };
            let cap = if window.is_some() {
                ring.s_cap
            } else {
                self.max_seq_len
            };
            ensure!(
                self.layer_stored[layer] == stored && self.layer_caps[layer] == cap,
                "BF16 handoff layer {layer} is not initialized through the committed frontier"
            );
        }
        Ok(LagunaBf16Handoff {
            layers: &self.layers,
            windows: &self.layer_windows,
            frontier: self.current_len,
            sliding_stored: ring.s_stored,
            sliding_cap: ring.s_cap,
            sliding_window: ring.s_window,
            max_seq_len: self.max_seq_len,
            n_kv: self.n_kv,
            head_dim: self.head_dim,
        })
    }

    pub fn max_seq_len(&self) -> usize {
        self.max_seq_len
    }

    #[cfg(feature = "cuda")]
    pub(crate) fn has_ring(&self) -> bool {
        self.ring.is_some()
    }

    #[cfg(feature = "cuda")]
    pub(crate) fn ring_meta_ptr(&self) -> Option<u64> {
        use cudarc::driver::DevicePtr;
        self.ring.as_ref().map(|r| {
            let stream = nv_layers::cuda_stream::current_stream(&r.dev);
            let (p, _g) = r.meta_dev.device_ptr(&stream);
            p
        })
    }

    #[cfg(feature = "cuda")]
    pub(crate) fn sliding_cap(&self) -> Option<usize> {
        self.ring.as_ref().map(|r| r.s_cap)
    }

    #[allow(dead_code)]
    pub(crate) fn layer_kv_bufs(&self, layer: usize) -> (Tensor, Tensor) {
        let (k, v) = &self.layers[layer];
        (k.clone(), v.clone())
    }

    pub fn layer_kv_rows_positions_by_kv_head_by_head_dim(&self, layer: usize) -> (Tensor, Tensor) {
        self.layer_kv_bufs(layer)
    }

    #[cfg(feature = "cuda")]
    pub(crate) fn note_graph_write(&mut self) {
        if let Some(r) = self.ring.as_ref() {
            let (fs, ss) = (r.full_stored, r.s_stored);
            for (li, w) in self.layer_windows.iter().enumerate() {
                self.layer_stored[li] = if w.is_some() { ss } else { fs };
            }
        }
    }

    pub fn rollback(&mut self, n: usize) -> Result<()> {
        if n == 0 {
            return Ok(());
        }
        if n > self.current_len {
            bail!(
                "LagunaKvCache.rollback: n {} > current_len {}",
                n,
                self.current_len
            );
        }
        for (layer, stored) in self.layer_stored.iter_mut().enumerate() {
            if *stored < n {
                bail!(
                    "LagunaKvCache.rollback: layer {layer} stored {} < n {}",
                    *stored,
                    n
                );
            }
            *stored -= n;
        }
        #[cfg(feature = "cuda")]
        if let Some(ring) = self.ring.as_mut() {
            if ring.full_stored < n || (ring.s_cap > 0 && ring.s_stored < n) {
                bail!(
                    "LagunaKvCache.rollback: ring stored underflow (full {}, sliding {}, n {})",
                    ring.full_stored,
                    ring.s_stored,
                    n
                );
            }
            ring.full_stored -= n;
            if ring.s_cap > 0 {
                ring.s_stored -= n;
            }
        }
        self.current_len -= n;
        Ok(())
    }

    #[cfg(feature = "cuda")]
    fn write_at_ring(
        &mut self,
        layer: usize,
        k_new: &Tensor,
        v_new: &Tensor,
        t: usize,
    ) -> Result<()> {
        let ring = self
            .ring
            .as_ref()
            .expect("write_at_ring without ring state");
        let (cap, meta_idx, committed) = match self.layer_windows[layer] {
            None => (self.max_seq_len, 0usize, ring.full_stored),
            Some(_) => (ring.s_cap, 1usize, ring.s_stored),
        };
        let stream = nv_layers::cuda_stream::current_stream(&ring.dev);
        let k_own;
        let k_src = if k_new.is_contiguous() {
            k_new
        } else {
            k_own = k_new.contiguous()?;
            &k_own
        };
        let v_own;
        let v_src = if v_new.is_contiguous() {
            v_new
        } else {
            v_own = v_new.contiguous()?;
            &v_own
        };
        let (k_buf, v_buf) = &self.layers[layer];
        for (src, dst) in [(k_src, k_buf), (v_src, v_buf)] {
            ring_append_launch(
                &stream,
                src,
                dst,
                &ring.meta_dev,
                meta_idx,
                t,
                cap,
                self.n_kv,
                self.head_dim,
            )?;
        }
        self.layer_stored[layer] = committed;
        Ok(())
    }
}

#[cfg(feature = "cuda")]
pub(crate) fn ring_prepare_decode_meta(
    cache_name: &str,
    ring_name: &str,
    write_pos: usize,
    n_total: usize,
    max_seq_len: usize,
    s_cap: usize,
    s_window: usize,
    s_stored: &mut usize,
    full_stored: &mut usize,
    host_meta: &mut [i32; 4],
    mut shift_windowed_layers: impl FnMut(usize, usize) -> Result<()>,
) -> Result<()> {
    if n_total < write_pos {
        bail!("{cache_name}.prepare_for_decode: n_total {n_total} < write_pos {write_pos}");
    }
    if n_total > max_seq_len {
        bail!(
            "{cache_name}.prepare_for_decode: n_total {n_total} exceeds max_seq_len {max_seq_len}"
        );
    }
    let n_new = n_total - write_pos;
    *full_stored = n_total;
    let mut slot = 0usize;
    if s_cap > 0 && n_new > 0 {
        if *s_stored + n_new > s_cap {
            let keep = (*s_stored).min(s_window);
            let shift = *s_stored - keep;
            if shift > 0 && keep > 0 {
                shift_windowed_layers(shift, keep)?;
            }
            *s_stored = keep;
        }
        if *s_stored + n_new > s_cap {
            bail!("{ring_name}: write of {n_new} tokens exceeds sliding capacity {s_cap}");
        }
        slot = *s_stored;
        *s_stored += n_new;
    }
    host_meta[0] = write_pos as i32;
    host_meta[1] = slot as i32;
    host_meta[2] = 0;
    host_meta[3] = *s_stored as i32;
    Ok(())
}

#[cfg(feature = "cuda")]
pub(crate) fn shift_rows_launch(
    stream: &std::sync::Arc<CudaStream>,
    buf: &Tensor,
    shift: usize,
    keep: usize,
    n_kv: usize,
    head_dim: usize,
) -> Result<()> {
    use cudarc::driver::DevicePtr;
    let (st, l) = buf.storage_and_layout();
    let cuda = match &*st {
        Storage::Cuda(s) => s,
        _ => bail!("kv shift: buffer must be on CUDA"),
    };
    let slice = cuda
        .as_cuda_slice::<half::bf16>()?
        .slice(l.start_offset()..);
    let (bp, _g) = slice.device_ptr(stream);
    let mut off = 0usize;
    while off < keep {
        let n = shift.min(keep - off);
        let rc = unsafe {
            nv_kernels::cuda::kv_shift_bf16(
                stream.cu_stream() as *mut std::ffi::c_void,
                bp as *mut u16,
                (shift + off) as i32,
                off as i32,
                n as i32,
                n_kv as i32,
                head_dim as i32,
            )
        };
        if rc != 0 {
            bail!("kv_shift_bf16 rc={rc}");
        }
        off += n;
    }
    Ok(())
}

#[cfg(feature = "cuda")]
pub(crate) fn ring_append_launch(
    stream: &std::sync::Arc<CudaStream>,
    src: &Tensor,
    dst: &Tensor,
    meta_dev: &cudarc::driver::CudaSlice<i32>,
    meta_idx: usize,
    t: usize,
    cap: usize,
    n_kv: usize,
    head_dim: usize,
) -> Result<()> {
    use cudarc::driver::DevicePtr;
    let (src_st, src_l) = src.storage_and_layout();
    let (dst_st, dst_l) = dst.storage_and_layout();
    let src_cuda = match &*src_st {
        Storage::Cuda(s) => s,
        _ => bail!("ring append: src must be on CUDA"),
    };
    let dst_cuda = match &*dst_st {
        Storage::Cuda(s) => s,
        _ => bail!("ring append: dst must be on CUDA"),
    };
    let src_slice = src_cuda
        .as_cuda_slice::<half::bf16>()?
        .slice(src_l.start_offset()..);
    let dst_slice = dst_cuda
        .as_cuda_slice::<half::bf16>()?
        .slice(dst_l.start_offset()..);
    let pos_view = meta_dev.slice(meta_idx..);
    let (sp, _g1) = src_slice.device_ptr(stream);
    let (dp, _g2) = dst_slice.device_ptr(stream);
    let (pp, _g3) = pos_view.device_ptr(stream);
    let rc = unsafe {
        nv_kernels::cuda::kv_ring_append_bf16(
            stream.cu_stream() as *mut std::ffi::c_void,
            sp as *const u16,
            dp as *mut u16,
            pp as *const i32,
            t as i32,
            cap as i32,
            n_kv as i32,
            head_dim as i32,
        )
    };
    if rc != 0 {
        bail!("kv_ring_append_bf16 rc={rc}");
    }
    Ok(())
}

impl Gemma4Cache for LagunaKvCache {
    fn current_len(&self) -> usize {
        self.current_len
    }
    fn advance(&mut self, n: usize) {
        self.current_len += n;
    }
    fn prepare_for_decode(&mut self, write_pos: usize, n_total: usize) -> Result<()> {
        self.pending_write_pos = write_pos;
        #[cfg(feature = "cuda")]
        if let Some(ring) = self.ring.as_mut() {
            let stream = nv_layers::cuda_stream::current_stream(&ring.dev);
            let layers = &self.layers;
            let layer_windows = &self.layer_windows;
            let n_kv = self.n_kv;
            let head_dim = self.head_dim;
            ring_prepare_decode_meta(
                "LagunaKvCache",
                "laguna kv ring",
                write_pos,
                n_total,
                self.max_seq_len,
                ring.s_cap,
                ring.s_window,
                &mut ring.s_stored,
                &mut ring.full_stored,
                &mut ring.host_meta,
                |shift, keep| {
                    for (li, w) in layer_windows.iter().enumerate() {
                        if w.is_none() {
                            continue;
                        }
                        let (k_buf, v_buf) = &layers[li];
                        for buf in [k_buf, v_buf] {
                            shift_rows_launch(&stream, buf, shift, keep, n_kv, head_dim)?;
                        }
                    }
                    Ok(())
                },
            )?;
            stream
                .memcpy_htod(&ring.host_meta[..], &mut ring.meta_dev)
                .map_err(|e| anyhow::anyhow!("htod kv ring meta: {e:?}"))?;
        }
        #[cfg(not(feature = "cuda"))]
        let _ = n_total;
        Ok(())
    }
    fn write_at(&mut self, layer: usize, k_new: &Tensor, v_new: &Tensor) -> Result<()> {
        let n_kv = self.n_kv;
        let head_dim = self.head_dim;
        let dims = k_new.dims();
        if dims.len() != 4 || dims[0] != 1 || dims[2] != n_kv || dims[3] != head_dim {
            bail!(
                "LagunaKvCache.write_at layer {layer}: expected [1, t, {n_kv}, {head_dim}], got {:?}",
                dims
            );
        }
        if v_new.dims() != dims {
            bail!(
                "LagunaKvCache.write_at: k/v shape mismatch k={:?} v={:?}",
                dims,
                v_new.dims()
            );
        }
        let t = dims[1];
        #[cfg(feature = "cuda")]
        if self.ring.is_some() {
            return self.write_at_ring(layer, k_new, v_new, t);
        }
        let cap = self.layer_caps[layer];

        if let Some(window) = self.layer_windows[layer] {
            let (k_buf, v_buf) = &self.layers[layer];
            let (k_updated, v_updated, new_stored) = sliding_write_at(
                k_buf,
                v_buf,
                self.layer_stored[layer],
                window,
                cap,
                n_kv,
                head_dim,
                k_new,
                v_new,
                layer,
            )?;
            self.layers[layer] = (k_updated, v_updated);
            self.layer_stored[layer] = new_stored;
            return Ok(());
        }
        let start = self.pending_write_pos;
        let end = start + t;
        if end > self.max_seq_len {
            bail!(
                "LagunaKvCache.write_at: end {} exceeds max_seq_len {}",
                end,
                self.max_seq_len
            );
        }
        let (k_buf, v_buf) = &self.layers[layer];
        let k_updated = k_buf.slice_assign(&[0..1, start..end, 0..n_kv, 0..head_dim], k_new)?;
        let v_updated = v_buf.slice_assign(&[0..1, start..end, 0..n_kv, 0..head_dim], v_new)?;
        self.layers[layer] = (k_updated, v_updated);
        self.layer_stored[layer] = end;
        Ok(())
    }
    fn view(&mut self, layer: usize, _len: usize) -> Result<(Tensor, Tensor)> {
        if layer >= self.layers.len() {
            bail!("LagunaKvCache.view: layer {layer} out of range");
        }
        #[cfg(feature = "cuda")]
        if let Some(ring) = self.ring.as_ref() {
            let (k, v) = &self.layers[layer];
            let stored = match self.layer_windows[layer] {
                None => ring.full_stored,
                Some(_) => ring.s_stored,
            };
            return Ok((k.narrow(1, 0, stored)?, v.narrow(1, 0, stored)?));
        }
        let stored = self.layer_stored[layer];
        let (k, v) = &self.layers[layer];
        let k = k.narrow(1, 0, stored)?;
        let v = v.narrow(1, 0, stored)?;
        Ok((k, v))
    }

    #[cfg(feature = "cuda")]
    fn try_prefill_attention_fp8(
        &mut self,
        layer: usize,
        q_rot: &Tensor,
        n_q: usize,
        seq: usize,
        sliding_window: Option<usize>,
        scaling: f32,
    ) -> Result<Option<Tensor>> {
        if !nv_config::env_arm_opt_in_eq_some_1_reread_every_call("NV_LAGUNA_PREFILL_FLASHINFER") {
            return Ok(None);
        }
        ensure!(
            layer < self.layers.len(),
            "Laguna full-prefill layer {layer} out of range"
        );
        ensure!(
            sliding_window.is_none(),
            "Laguna full-prefill FlashInfer route only supports full attention"
        );
        ensure!(
            self.n_kv == 8 && self.head_dim == 128 && n_q == 48,
            "Laguna full-prefill FlashInfer route requires 48 Q heads, 8 KV heads and head dimension 128, got {n_q}/{}/{}",
            self.n_kv,
            self.head_dim
        );
        let expected_scale = (self.head_dim as f32).powf(-0.5);
        ensure!(
            (scaling - expected_scale).abs() <= f32::EPSILON,
            "Laguna full-prefill FlashInfer route requires scale {expected_scale}, got {scaling}"
        );
        let capacity = self.layer_caps[layer];
        let kv_len = match self.ring.as_ref() {
            Some(ring) => ring.full_stored,
            None => self.layer_stored[layer],
        };
        ensure!(
            kv_len >= seq,
            "Laguna full-prefill KV length {kv_len} is shorter than query length {seq}"
        );
        let (k, v) = {
            let (k, v) = &self.layers[layer];
            (k.clone(), v.clone())
        };
        let runner = self.prefill_attn.get_or_insert_default();
        runner.run(q_rot, &k, &v, seq, kv_len, capacity).map(Some)
    }

    #[cfg(feature = "cuda")]
    fn try_decode_attention_gqa(
        &mut self,
        layer: usize,
        q_rot: &Tensor,
        n_q: usize,
        sliding_window: Option<usize>,
        scaling: f32,
    ) -> Result<Option<Tensor>> {
        use cudarc::driver::{DevicePtr, DevicePtrMut};
        let head_dim = self.head_dim;
        let n_kv = self.n_kv;
        if head_dim != 128 || n_kv == 0 || !n_q.is_multiple_of(n_kv) || !matches!(n_q / n_kv, 6 | 8)
        {
            return Ok(None);
        }
        let Some(ring) = self.ring.as_ref() else {
            return Ok(None);
        };
        if sliding_window.is_some() && (ring.s_cap == 0 || ring.s_stored == 0) {
            return Ok(None);
        }
        let expected = n_q * head_dim;
        let total: usize = q_rot.dims().iter().product();
        ensure!(
            total == expected,
            "try_decode_attention_gqa layer {layer}: expected {expected} elements, got {:?}",
            q_rot.dims()
        );
        let dev = ring.dev.clone();
        let stream = nv_layers::cuda_stream::current_stream(&dev);
        if self.gqa_bufs.is_none() {
            let elems = nv_kernels::cuda::laguna_flash_decode_gqa_scratch_elems(n_kv as i32);
            let scratch = stream
                .alloc_zeros::<f32>(elems)
                .map_err(|e| anyhow::anyhow!("gqa scratch: {e:?}"))?;
            let fan_in = stream
                .alloc_zeros::<u32>(n_kv)
                .map_err(|e| anyhow::anyhow!("gqa fan_in: {e:?}"))?;
            self.gqa_bufs = Some((scratch, fan_in));
        }
        let ring = self.ring.as_ref().unwrap();
        let (scratch, fan_in) = self.gqa_bufs.as_mut().unwrap();

        let mut out = unsafe { stream.alloc::<half::bf16>(expected)? };
        let q_c = q_rot.contiguous()?;
        let (q_st, ql) = q_c.storage_and_layout();
        let q_cuda = match &*q_st {
            Storage::Cuda(s) => s,
            _ => bail!("q_rot must be on CUDA"),
        };
        let q_view = q_cuda
            .as_cuda_slice::<half::bf16>()?
            .slice(ql.start_offset()..);
        let (k_buf, v_buf) = &self.layers[layer];
        let (k_st, kl) = k_buf.storage_and_layout();
        let (v_st, vl) = v_buf.storage_and_layout();
        let k_cuda = match &*k_st {
            Storage::Cuda(s) => s,
            _ => bail!("kv ring buffers must be on CUDA"),
        };
        let v_cuda = match &*v_st {
            Storage::Cuda(s) => s,
            _ => bail!("kv ring buffers must be on CUDA"),
        };
        let k_view = k_cuda
            .as_cuda_slice::<half::bf16>()?
            .slice(kl.start_offset()..);
        let v_view = v_cuda
            .as_cuda_slice::<half::bf16>()?
            .slice(vl.start_offset()..);
        let (meta_off, delta, win) = match sliding_window {
            Some(w) => (3usize, 0i32, w as i32),
            None => (0usize, 1i32, 0i32),
        };
        let meta_view = ring.meta_dev.slice(meta_off..);
        let rc = {
            let (qp, _gq) = q_view.device_ptr(&stream);
            let (kp, _gk) = k_view.device_ptr(&stream);
            let (vp, _gv) = v_view.device_ptr(&stream);
            let (mp, _gm) = meta_view.device_ptr(&stream);
            let (scp, _gs) = scratch.device_ptr_mut(&stream);
            let (fip, _gf) = fan_in.device_ptr_mut(&stream);
            let (op, _go) = out.device_ptr_mut(&stream);
            unsafe {
                nv_kernels::cuda::laguna_flash_decode_gqa(
                    stream.cu_stream() as *mut std::ffi::c_void,
                    qp as *const u16,
                    kp as *const u16,
                    vp as *const u16,
                    op as *mut u16,
                    mp as *const i32,
                    delta,
                    scp as *mut f32,
                    fip as *mut u32,
                    n_q as i32,
                    n_kv as i32,
                    head_dim as i32,
                    win,
                    scaling,
                )
            }
        };
        ensure!(rc == 0, "laguna_flash_decode_gqa rc={rc}");
        drop(q_st);
        let storage = CudaStorage::wrap_cuda_slice(out, dev);
        let tensor = candle_core::Tensor::from_storage(
            Storage::Cuda(storage),
            (1usize, 1usize, n_q, head_dim),
            BackpropOp::none(),
            false,
        );
        Ok(Some(tensor))
    }

    fn try_decode_attention_ring(
        &mut self,
        layer: usize,
        q_rot: &Tensor,
        n_q: usize,
        sliding_window: Option<usize>,
        scaling: f32,
    ) -> Result<Option<Tensor>> {
        #[cfg(feature = "cuda")]
        {
            use cudarc::driver::{DevicePtr, DevicePtrMut};
            let _ = sliding_window;
            let Some(ring) = self.ring.as_ref() else {
                return Ok(None);
            };
            if !ring.ring_attn {
                return Ok(None);
            }
            let Some(window) = self.layer_windows[layer] else {
                return Ok(None);
            };
            if ring.s_cap == 0 || ring.s_stored == 0 {
                return Ok(None);
            }
            let head_dim = self.head_dim;
            let n_kv = self.n_kv;
            let expected = n_q * head_dim;
            let total: usize = q_rot.dims().iter().product();
            if total != expected {
                bail!(
                    "LagunaKvCache.try_decode_attention_ring layer {layer}: expected {expected} \
                     elements, got dims {:?}",
                    q_rot.dims()
                );
            }
            let stream = nv_layers::cuda_stream::current_stream(&ring.dev);
            let mut out = unsafe { stream.alloc::<half::bf16>(expected)? };
            let q_c = q_rot.contiguous()?;
            let (q_st, ql) = q_c.storage_and_layout();
            let q_cuda = match &*q_st {
                Storage::Cuda(s) => s,
                _ => bail!("q_rot must be on CUDA"),
            };
            let q_view = q_cuda
                .as_cuda_slice::<half::bf16>()?
                .slice(ql.start_offset()..);
            let (k_buf, v_buf) = &self.layers[layer];
            let (k_st, kl) = k_buf.storage_and_layout();
            let (v_st, vl) = v_buf.storage_and_layout();
            let k_cuda = match &*k_st {
                Storage::Cuda(s) => s,
                _ => bail!("kv ring buffers must be on CUDA"),
            };
            let v_cuda = match &*v_st {
                Storage::Cuda(s) => s,
                _ => bail!("kv ring buffers must be on CUDA"),
            };
            let k_view = k_cuda
                .as_cuda_slice::<half::bf16>()?
                .slice(kl.start_offset()..);
            let v_view = v_cuda
                .as_cuda_slice::<half::bf16>()?
                .slice(vl.start_offset()..);

            let meta_view = ring.meta_dev.slice(2..);
            let rc = {
                let (qp, _gq) = q_view.device_ptr(&stream);
                let (kp, _gk) = k_view.device_ptr(&stream);
                let (vp, _gv) = v_view.device_ptr(&stream);
                let (mp, _gm) = meta_view.device_ptr(&stream);
                let (op, _go) = out.device_ptr_mut(&stream);
                unsafe {
                    nv_kernels::cuda::attention_bf16_decode_ring(
                        stream.cu_stream() as *mut std::ffi::c_void,
                        qp as *const u16,
                        kp as *const u16,
                        vp as *const u16,
                        op as *mut u16,
                        mp as *const i32,
                        ring.s_cap as i32,
                        window as i32,
                        n_q as i32,
                        n_kv as i32,
                        head_dim as i32,
                        scaling,
                    )
                }
            };
            if rc != 0 {
                bail!("attention_bf16_decode_ring rc={rc}");
            }
            drop(q_st);
            let dev = ring.dev.clone();
            let storage = CudaStorage::wrap_cuda_slice(out, dev);
            let tensor = candle_core::Tensor::from_storage(
                Storage::Cuda(storage),
                (1usize, 1usize, n_q, head_dim),
                BackpropOp::none(),
                false,
            );
            Ok(Some(tensor))
        }
        #[cfg(not(feature = "cuda"))]
        {
            let _ = (layer, q_rot, n_q, sliding_window, scaling);
            Ok(None)
        }
    }
}

pub struct LoadedLaguna {
    pub model: Laguna,
    pub cache: LagunaKvCache,
}

impl CausalLm for LoadedLaguna {
    fn forward(&mut self, tokens: &[u32], positions: &[u32]) -> Result<Vec<f32>> {
        let seq = tokens.len();
        let device = self.model.device().clone();
        let tokens_t = Tensor::from_vec(tokens.to_vec(), (1usize, seq), &device)?;
        let positions_i32: Vec<i32> = positions.iter().map(|&p| p as i32).collect();
        let positions_t = Tensor::from_vec(positions_i32, seq, &device)?;
        let logits = self
            .model
            .forward_with_cache(&tokens_t, &positions_t, &mut self.cache)?;
        let dims = logits.dims().to_vec();
        let vocab = *dims.last().unwrap();
        let last = logits
            .reshape((seq, vocab))?
            .narrow(0, seq - 1, 1)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        Ok(last)
    }

    fn vocab_size(&self) -> usize {
        self.model.config().vocab_size
    }
}

fn apply_partial_rope_scaled(
    q: &Tensor,
    k: &Tensor,
    rope: &Rope,
    positions: &Tensor,
    rotary_dim: usize,
    head_dim: usize,
    rot_scale: f32,
) -> Result<(Tensor, Tensor)> {
    let scale_rot = |t: Tensor| -> Result<Tensor> {
        if (rot_scale - 1.0).abs() < f32::EPSILON {
            Ok(t)
        } else {
            let f = t.to_dtype(DType::F32)?;
            Ok(f.affine(rot_scale as f64, 0.0)?)
        }
    };
    if rotary_dim >= head_dim {
        let (q_r, k_r) = rope.apply(
            &q.to_dtype(DType::F32)?,
            &k.to_dtype(DType::F32)?,
            positions,
        )?;
        return Ok((scale_rot(q_r)?, scale_rot(k_r)?));
    }
    let dims_q = q.dims().to_vec();
    let dims_k = k.dims().to_vec();
    let last = dims_q.len() - 1;
    let q_rot = q.narrow(last, 0, rotary_dim)?.contiguous()?;
    let q_pass = q
        .narrow(last, rotary_dim, head_dim - rotary_dim)?
        .to_dtype(DType::F32)?
        .contiguous()?;
    let k_rot = k.narrow(dims_k.len() - 1, 0, rotary_dim)?.contiguous()?;
    let k_pass = k
        .narrow(dims_k.len() - 1, rotary_dim, head_dim - rotary_dim)?
        .to_dtype(DType::F32)?
        .contiguous()?;
    let (q_r, k_r) = rope.apply(
        &q_rot.to_dtype(DType::F32)?,
        &k_rot.to_dtype(DType::F32)?,
        positions,
    )?;
    let q_r = scale_rot(q_r)?;
    let k_r = scale_rot(k_r)?;
    let q_out = Tensor::cat(&[&q_r, &q_pass], last)?;
    let k_out = Tensor::cat(&[&k_r, &k_pass], dims_k.len() - 1)?;
    Ok((q_out, k_out))
}

pub(crate) fn softplus_f32(x: &Tensor) -> Result<Tensor> {
    let relu = x.relu()?;
    let neg_abs = x.abs()?.neg()?;
    let log1p = neg_abs.exp()?.affine(1.0, 1.0)?.log()?;
    relu.add(&log1p).map_err(Into::into)
}

pub(crate) fn attention(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    n_q: usize,
    n_kv: usize,
    head_dim: usize,
    seq_q: usize,
    scale: f32,
    window: Option<usize>,
) -> Result<Tensor> {
    #[cfg(feature = "cuda")]
    if q.device().is_cuda() {
        use nv_layers::attn::{flash_attn_windowed, AttnConfig};
        let cfg = AttnConfig {
            num_heads: n_q,
            num_kv_heads: n_kv,
            head_dim,
            softmax_scale: scale,
            causal: true,
        };
        let out = flash_attn_windowed(q, k, v, &cfg, window.map(|w| w.saturating_sub(1)), Some(0))?;
        assert_eq!(out.dims(), &[1, seq_q, n_q, head_dim]);
        return Ok(out);
    }
    sdpa_windowed(q, k, v, n_q, n_kv, head_dim, seq_q, scale, window)
}

fn sdpa_windowed(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    n_q: usize,
    n_kv: usize,
    head_dim: usize,
    seq_q: usize,
    scale: f32,
    window: Option<usize>,
) -> Result<Tensor> {
    let kd = k.dims();
    let seq_k = kd[1];
    let q_f = q.to_dtype(DType::F32)?;
    let k_f = k.to_dtype(DType::F32)?;
    let v_f = v.to_dtype(DType::F32)?;

    let (k_exp, v_exp) = if n_kv == n_q {
        (k_f, v_f)
    } else {
        let factor = n_q / n_kv;
        let k_exp = k_f
            .unsqueeze(3)?
            .expand((1, seq_k, n_kv, factor, head_dim))?
            .reshape((1, seq_k, n_q, head_dim))?;
        let v_exp = v_f
            .unsqueeze(3)?
            .expand((1, seq_k, n_kv, factor, head_dim))?
            .reshape((1, seq_k, n_q, head_dim))?;
        (k_exp, v_exp)
    };

    let q_t = q_f.permute((0, 2, 1, 3))?.contiguous()?;
    let k_t = k_exp.permute((0, 2, 1, 3))?.contiguous()?;
    let v_t = v_exp.permute((0, 2, 1, 3))?.contiguous()?;
    let q_flat = q_t.reshape((n_q, seq_q, head_dim))?;
    let k_flat = k_t.reshape((n_q, seq_k, head_dim))?;
    let v_flat = v_t.reshape((n_q, seq_k, head_dim))?;
    let k_perm = k_flat.permute((0, 2, 1))?.contiguous()?;

    let mut scores = q_flat.matmul(&k_perm)?.affine(scale as f64, 0.0)?;

    let mut mask = vec![0f32; seq_q * seq_k];
    let offset = seq_k.saturating_sub(seq_q);
    for i in 0..seq_q {
        let qi = i + offset;
        for j in 0..seq_k {
            let causal_violation = j > qi;
            let window_violation = window.map(|w| qi >= j && qi - j >= w).unwrap_or(false);
            if causal_violation || window_violation {
                mask[i * seq_k + j] = f32::NEG_INFINITY;
            }
        }
    }
    let mask_t = Tensor::from_vec(mask, (1usize, seq_q, seq_k), q_flat.device())?;
    scores = scores.broadcast_add(&mask_t)?;

    let probs = candle_nn::ops::softmax_last_dim(&scores)?;
    let out = probs.matmul(&v_flat)?;
    let out = out
        .reshape((1, n_q, seq_q, head_dim))?
        .permute((0, 2, 1, 3))?
        .contiguous()?
        .to_dtype(q.dtype())?;
    Ok(out)
}

pub(crate) fn weight_linears_fp8_enabled() -> bool {
    nv_config::env_arm_opt_in_eq_some_1_reread_every_call("NV_LAGUNA_WEIGHT_LINEARS_FP8")
}

fn load_linear_plain(
    weights: &WeightLoader,
    name: &str,
    out_features: usize,
    in_features: usize,
    dtype: DType,
    fp8_runner: &Fp8RunnerHandle,
    device: &Device,
) -> Result<Linear> {
    let w = load_linear_weight(weights, name, out_features, in_features, dtype)?;
    load_linear_from_tensor_maybe_fp8(w, fp8_runner, device)
}

fn load_laguna_rmsnorm(
    weights: &WeightLoader,
    name: &str,
    dim: usize,
    eps: f64,
    dtype: DType,
    fp8: bool,
) -> Result<RmsNorm> {
    let weight = weights
        .get(name, dtype)
        .with_context(|| format!("load {name}"))?;
    ensure!(
        weight.dims() == [dim],
        "{name}: expected [{dim}], got {:?}",
        weight.dims()
    );
    #[cfg(feature = "cuda")]
    if fp8 {
        return RmsNorm::new_fp8_e4m3(weight, eps);
    }
    #[cfg(not(feature = "cuda"))]
    let _ = fp8;
    Ok(RmsNorm::new(weight, eps))
}

fn load_linear_weight(
    weights: &WeightLoader,
    name: &str,
    out_features: usize,
    in_features: usize,
    dtype: DType,
) -> Result<Tensor> {
    let w = weights
        .get(name, dtype)
        .with_context(|| format!("load {name}"))?;
    let d = w.dims();
    if d.len() != 2 || d[0] != out_features || d[1] != in_features {
        bail!(
            "linear {name}: expected [{}, {}], got {:?}",
            out_features,
            in_features,
            d
        );
    }
    Ok(w)
}

pub(crate) fn load_linear_from_tensor_maybe_fp8(
    weight: Tensor,
    fp8_runner: &Fp8RunnerHandle,
    device: &Device,
) -> Result<Linear> {
    #[cfg(feature = "cuda")]
    if let Some(runner) = fp8_runner {
        let (weight_u8, weight_scale_dev) =
            nv_layers::lm_head_resident_quant::rowquant_e4m3_dev(&weight)?;
        let (out_features, in_features) = weight.dims2()?;
        ensure!(
            weight_u8.len() == weight.elem_count() && weight_scale_dev.len() == out_features,
            "Laguna FP8 linear requires one byte per weight and one scale per output row"
        );
        let Device::Cuda(dev) = device else {
            bail!("Laguna FP8 linear requires CUDA");
        };
        let stream = nv_layers::cuda_stream::current_stream(dev);
        let weight_scale_rows = stream.clone_dtoh(&weight_scale_dev)?;
        return Linear::new_fp8_e4m3_row_scales_without_the_cublaslt_probe(
            weight_u8,
            weight_scale_rows,
            in_features,
            out_features,
            None,
            device,
            runner.clone(),
            nv_quant::fp8::Fp8ScaleMode::PerOuterRow,
        );
    }
    let _ = (fp8_runner, device);
    Linear::new(weight, None)
}

#[cfg(feature = "cuda")]
const SHARED_EXPERT_GEMV_ROWS_8_MATCHES_THE_VERIFY_BLOCK_AND_THE_MROWS_LADDER: usize = 8;

#[cfg(feature = "cuda")]
fn shared_expert_rows_gemv_opt_in_row_limit() -> Option<usize> {
    nv_config::env_arm_opt_in_eq_some_1_reread_every_call("NV_LAGUNA_SHARED_GEMV")
        .then_some(SHARED_EXPERT_GEMV_ROWS_8_MATCHES_THE_VERIFY_BLOCK_AND_THE_MROWS_LADDER)
}

fn load_linear_expert(
    weights: &WeightLoader,
    module: &str,
    out_features: usize,
    in_features: usize,
    dtype: DType,
    qconfig: &QuantizationConfig,
    nvfp4_runner: &Nvfp4RunnerHandle,
    device: &Device,
) -> Result<Linear> {
    #[cfg(feature = "cuda")]
    if let Some(runner) = nvfp4_runner {
        return nv_layers::moe::load_linear_maybe_quant(
            weights,
            &format!("{module}.weight"),
            out_features,
            in_features,
            dtype,
            qconfig,
            runner.clone(),
            device,
        );
    }
    let _ = (qconfig, nvfp4_runner);
    load_linear_plain(
        weights,
        &format!("{module}.weight"),
        out_features,
        in_features,
        dtype,
        &None,
        device,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    const TEST_CONFIG: &str = r#"{
        "architectures": ["LagunaForCausalLM"],
        "model_type": "laguna",
        "vocab_size": 128,
        "hidden_size": 32,
        "intermediate_size": 64,
        "num_hidden_layers": 4,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 8,
        "max_position_embeddings": 512,
        "rms_norm_eps": 1e-6,
        "num_experts": 4,
        "num_experts_per_tok": 2,
        "moe_intermediate_size": 16,
        "shared_expert_intermediate_size": 16,
        "norm_topk_prob": true,
        "mlp_only_layers": [0],
        "decoder_sparse_step": 1,
        "tie_word_embeddings": false,
        "gating": "per-head",
        "sliding_window": 8,
        "moe_routed_scaling_factor": 2.5,
        "eos_token_id": [2, 24],
        "rope_parameters": {
            "full_attention": {
                "rope_theta": 500000.0,
                "rope_type": "yarn",
                "factor": 32.0,
                "original_max_position_embeddings": 64,
                "beta_slow": 1.0,
                "beta_fast": 64.0,
                "attention_factor": 1.3465735902799727,
                "partial_rotary_factor": 0.5
            },
            "sliding_attention": {
                "rope_type": "default",
                "rope_theta": 10000.0,
                "partial_rotary_factor": 1.0
            }
        },
        "layer_types": ["full_attention", "sliding_attention", "sliding_attention", "sliding_attention"],
        "mlp_layer_types": ["dense", "sparse", "sparse", "sparse"],
        "num_attention_heads_per_layer": [4, 4, 4, 4]
    }"#;

    #[test]
    fn yarn_inv_freq_interpolates_low_frequencies() {
        let params = LagunaRopeParams {
            rope_theta: 500000.0,
            partial_rotary_factor: 0.5,
            rope_type: Some("yarn".into()),
            factor: Some(32.0),
            original_max_position_embeddings: Some(8192),
            beta_fast: Some(64.0),
            beta_slow: Some(1.0),
            attention_factor: Some(1.346_573_6),
        };
        let dim = 64usize;
        let yarn = yarn_inv_freq(dim, &params);
        let base: Vec<f32> = (0..dim / 2)
            .map(|i| (1.0f64 / 500000f64.powf((i as f64 * 2.0) / dim as f64)) as f32)
            .collect();
        assert_eq!(yarn.len(), dim / 2);
        assert!(
            (yarn[0] - base[0]).abs() < 1e-9,
            "high freq must be extrapolated as-is"
        );
        let last = dim / 2 - 1;
        let expected_interp = base[last] / 32.0;
        assert!(
            (yarn[last] - expected_interp).abs() / expected_interp < 1e-3,
            "low freq must be interpolated by factor: got {} want {}",
            yarn[last],
            expected_interp
        );
        for w in yarn.windows(2) {
            assert!(w[1] < w[0], "inv_freq must be strictly decreasing");
        }
    }

    fn make_moe(device: &Device) -> LagunaMoe {
        let hidden = 8usize;
        let inter = 4usize;
        let num_experts = 4usize;
        let det = |n: usize, seed: f32| -> Vec<f32> {
            (0..n)
                .map(|i| ((i as f32 + seed) * 0.19).sin() * 0.2)
                .collect()
        };
        let lin = |out: usize, inp: usize, seed: f32| -> Linear {
            let w = Tensor::from_vec(det(out * inp, seed), (out, inp), device).unwrap();
            Linear::new(w, None).unwrap()
        };
        let experts = (0..num_experts)
            .map(|e| {
                Mlp::new(
                    lin(inter, hidden, e as f32 + 1.0),
                    lin(inter, hidden, e as f32 + 11.0),
                    lin(hidden, inter, e as f32 + 21.0),
                )
                .unwrap()
            })
            .collect();
        LagunaMoe {
            num_experts,
            top_k: 2,
            norm_topk: true,
            routed_scaling: 2.5,
            softcap: 0.0,
            gate: lin(num_experts, hidden, 31.0),
            selection_bias: Tensor::zeros((1usize, num_experts), DType::F32, device).unwrap(),
            experts,
            shared_expert: Mlp::new(
                lin(inter, hidden, 41.0),
                lin(inter, hidden, 51.0),
                lin(hidden, inter, 61.0),
            )
            .unwrap(),
            #[cfg(feature = "cuda")]
            grouped: std::sync::Mutex::new(None),
            #[cfg(feature = "cuda")]
            train_stacks: std::sync::Mutex::new(None),
        }
    }

    #[test]
    fn released_routed_experts_refuse_individual_execution() {
        let device = Device::Cpu;
        let mut moe = make_moe(&device);
        let x = Tensor::zeros((1usize, 8usize), DType::F32, &device).unwrap();
        assert!(moe.forward(&x).is_ok());
        drop(std::mem::take(&mut moe.experts));
        let error = moe.forward(&x).unwrap_err().to_string();
        assert!(
            error.contains("individual routed experts unavailable"),
            "{error}"
        );
        assert!(moe.shared_expert.forward(&x).is_ok());
    }

    #[test]
    fn sdpa_windowed_masks_out_of_window_keys() {
        let device = Device::Cpu;
        let seq = 6usize;
        let n_q = 2usize;
        let n_kv = 1usize;
        let d = 4usize;
        let det = |n: usize, s: f32| -> Vec<f32> {
            (0..n).map(|i| ((i as f32 + s) * 0.37).sin()).collect()
        };
        let q = Tensor::from_vec(det(seq * n_q * d, 0.0), (1, seq, n_q, d), &device).unwrap();
        let k = Tensor::from_vec(det(seq * n_kv * d, 5.0), (1, seq, n_kv, d), &device).unwrap();
        let v = Tensor::from_vec(det(seq * n_kv * d, 9.0), (1, seq, n_kv, d), &device).unwrap();

        let full = sdpa_windowed(&q, &k, &v, n_q, n_kv, d, seq, 0.5, None).unwrap();
        let windowed = sdpa_windowed(&q, &k, &v, n_q, n_kv, d, seq, 0.5, Some(2)).unwrap();
        let f: Vec<f32> = full.flatten_all().unwrap().to_vec1().unwrap();
        let w: Vec<f32> = windowed.flatten_all().unwrap().to_vec1().unwrap();
        let head0 = &f[..n_q * d];
        let whead0 = &w[..n_q * d];
        for (a, b) in head0.iter().zip(whead0.iter()) {
            assert!(
                (a - b).abs() < 1e-6,
                "first token attends only itself in both"
            );
        }
        let mut max_diff = 0f32;
        for (a, b) in f.iter().zip(w.iter()) {
            max_diff = max_diff.max((a - b).abs());
        }
        assert!(max_diff > 1e-4, "window must change late-token outputs");
    }

    #[test]
    fn prefill_chunk_limits_canonical_tail_sliding_capacity_and_sliding_layer_storage_are_bounded() {
        for value in [
            "",
            "0",
            "1",
            "255",
            "257",
            "4097",
            "8193",
            "16384",
            "-1",
            "large",
            "18446744073709551616",
        ] {
            assert!(
                parse_prefill_chunk(value).is_err(),
                "invalid chunk {value} accepted"
            );
        }
        for chunk in [256, 512, 1024, 2048, 4096, 8192] {
            assert_eq!(parse_prefill_chunk(&chunk.to_string()).unwrap(), chunk);
            assert_eq!(canonical_tail_prefill_len(0, chunk), 0);
            for depth in (1..=32769).chain([168000, 261888, 262144]) {
                let mut remaining = depth;
                while remaining > 0 {
                    let m = canonical_tail_prefill_len(remaining, chunk);
                    assert!(m > 0 && m <= chunk && m <= remaining);
                    remaining -= m;
                    if remaining == 0 {
                        assert_eq!(
                            m,
                            (depth - 1) % 256 + 1,
                            "final head shape changed at depth {depth} chunk {chunk}"
                        );
                    } else {
                        assert_eq!(m % 256, 0);
                    }
                }
            }
        }

        let cfg = LagunaConfig::from_hf_json_str(TEST_CONFIG).unwrap();
        let n_kv = cfg.num_key_value_heads;
        let hd = cfg.head_dim;
        let window = cfg.sliding_window;
        for chunk in [256, 512, 1024, 2048, 4096, 8192] {
            let mut cache =
                LagunaKvCache::new_with_prefill_chunk(&cfg, 32768, &Device::Cpu, DType::F32, chunk)
                    .unwrap();
            assert_eq!(cache.layer_caps[0], 32768);
            assert_eq!(cache.layer_caps[1], window + chunk);
            let oversized =
                Tensor::zeros((1, window + chunk + 1, n_kv, hd), DType::F32, &Device::Cpu).unwrap();
            cache.prepare_for_decode(0, window + chunk + 1).unwrap();
            assert!(cache.write_at(1, &oversized, &oversized).is_err());
            assert_eq!(cache.current_len(), 0);
            assert_eq!(cache.layer_stored[1], 0);
            for step in 0..3 {
                let start = step * chunk;
                let data: Vec<f32> = (start..start + chunk)
                    .flat_map(|p| std::iter::repeat_n(p as f32, n_kv * hd))
                    .collect();
                let values = Tensor::from_vec(data, (1, chunk, n_kv, hd), &Device::Cpu).unwrap();
                cache.prepare_for_decode(start, start + chunk).unwrap();
                cache.write_at(1, &values, &values).unwrap();
                cache.advance(chunk);
                let (k, v) = cache.view(1, start + chunk).unwrap();
                assert_eq!(k.dims()[1], chunk + start.min(window));
                assert!(k.dims()[1] <= window + chunk);
                let actual = k.flatten_all().unwrap().to_vec1::<f32>().unwrap();
                for (i, row) in actual.chunks_exact(n_kv * hd).enumerate() {
                    assert!(row
                        .iter()
                        .all(|&v| v == (start.saturating_sub(window) + i) as f32));
                }
                assert_prefill_tensor_bits("sliding K/V retained rows", &k, &v);
            }
        }
        assert!(LagunaKvCache::new_with_prefill_chunk(
            &cfg,
            32768,
            &Device::Cpu,
            DType::F32,
            16384
        )
        .is_err());

        let cfg = LagunaConfig::from_hf_json_str(TEST_CONFIG).unwrap();
        let device = Device::Cpu;
        let mut cache = LagunaKvCache::new(&cfg, 4096, &device, DType::F32).unwrap();
        assert_eq!(cache.layer_caps[0], 4096);
        assert_eq!(cache.layer_caps[1], 8 + SLIDING_COMPACT_SLACK);
        let n_kv = cfg.num_key_value_heads;
        let d = cfg.head_dim;
        for step in 0..600usize {
            let k = Tensor::zeros((1usize, 1usize, n_kv, d), DType::F32, &device).unwrap();
            let v = Tensor::zeros((1usize, 1usize, n_kv, d), DType::F32, &device).unwrap();
            cache.prepare_for_decode(step, step + 1).unwrap();
            for layer in 0..cfg.num_hidden_layers {
                cache.write_at(layer, &k, &v).unwrap();
            }
            cache.advance(1);
        }
        assert_eq!(cache.current_len(), 600);
        assert_eq!(cache.layer_stored[0], 600);
        assert!(cache.layer_stored[1] <= 8 + SLIDING_COMPACT_SLACK);
        let (k1, _) = cache.view(1, 600).unwrap();
        assert!(k1.dims()[1] <= 8 + SLIDING_COMPACT_SLACK);

    }

    #[test]
    fn study_capture_isolated_from_serving_fp8_selection() {
        const CHILD: &str = "NV_STUDY_CAPTURE_CACHE_TEST_CHILD";
        if std::env::var_os(CHILD).is_none() {
            let status = std::process::Command::new(std::env::current_exe().unwrap())
                .args([
                    "--exact",
                    "laguna::tests::study_capture_isolated_from_serving_fp8_selection",
                    "--nocapture",
                ])
                .env(CHILD, "1")
                .env("NV_LAGUNA_FP8_KV", "1")
                .env("NV_LAGUNA_FP8_UNIT_SCALE", "1")
                .env_remove("NV_LAGUNA_KV8_INT8")
                .status()
                .unwrap();
            assert!(
                status.success(),
                "capture must work with production FP8 serving selection"
            );
            return;
        }
        let cfg = LagunaConfig::from_hf_json_str(TEST_CONFIG).unwrap();
        let model = build_tiny_model(&cfg, &Device::Cpu);
        #[cfg(feature = "cuda")]
        {
            assert!(
                model.new_kv_cache(64).is_err(),
                "serving guard must remain enforced"
            );
            assert!(
                LagunaKvCache::new_with_mode(&cfg, 64, &Device::Cpu, DType::F32, false).is_err()
            );
        }
        let mut state = model.new_study_capture_state(0, 64).unwrap();
        let (tokens, calls, chunks) = model
            .study_capture_record_tokens(&[1, 5, 9], &mut state)
            .unwrap();
        assert_eq!(tokens.dims(), &[3, cfg.hidden_size]);
        assert_eq!((calls, chunks), (1, 1));
        let (last, _, _) = model.study_capture_prefix(&[1, 5, 9], &mut state).unwrap();
        assert_eq!(
            last.to_vec1::<f32>().unwrap(),
            tokens.get(2).unwrap().to_vec1::<f32>().unwrap()
        );
        assert!(last
            .to_vec1::<f32>()
            .unwrap()
            .iter()
            .all(|value| value.is_finite()));
        assert_eq!(std::env::var("NV_LAGUNA_FP8_KV").unwrap(), "1");
        assert_eq!(std::env::var("NV_LAGUNA_FP8_UNIT_SCALE").unwrap(), "1");
    }

    #[test]
    fn cached_decode_matches_full_prefill() {
        let cfg = LagunaConfig::from_hf_json_str(TEST_CONFIG).unwrap();
        let device = Device::Cpu;
        let model = build_tiny_model(&cfg, &device);
        let toks = vec![1u32, 5, 9, 2];

        let tokens_full = Tensor::from_vec(toks.clone(), (1usize, toks.len()), &device).unwrap();
        let positions_full = Tensor::from_vec(
            (0..toks.len() as u32).collect::<Vec<u32>>(),
            toks.len(),
            &device,
        )
        .unwrap();
        let mut cache_full = model.new_kv_cache(64).unwrap();
        let logits_full = model
            .forward_with_cache(&tokens_full, &positions_full, &mut cache_full)
            .unwrap();
        let full_last: Vec<f32> = logits_full
            .narrow(1, toks.len() - 1, 1)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1()
            .unwrap();

        let mut cache_inc = model.new_kv_cache(64).unwrap();
        let mut inc_last: Vec<f32> = Vec::new();
        for (i, t) in toks.iter().enumerate() {
            let tt = Tensor::from_vec(vec![*t], (1usize, 1usize), &device).unwrap();
            let pp = Tensor::from_vec(vec![i as u32], 1usize, &device).unwrap();
            let logits = model.forward_with_cache(&tt, &pp, &mut cache_inc).unwrap();
            inc_last = logits.flatten_all().unwrap().to_vec1().unwrap();
        }

        assert_eq!(full_last.len(), inc_last.len());
        let mut max_diff = 0f32;
        for (a, b) in full_last.iter().zip(inc_last.iter()) {
            max_diff = max_diff.max((a - b).abs());
        }
        assert!(
            max_diff < 5e-2,
            "cached decode must match full prefill, max diff {max_diff}"
        );
    }

    fn assert_prefill_tensor_bits(tag: &str, reference: &Tensor, candidate: &Tensor) {
        assert_eq!(reference.dims(), candidate.dims(), "{tag}: shape changed");
        let before = reference
            .to_dtype(DType::F32)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        let after = candidate
            .to_dtype(DType::F32)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        assert!(
            before.iter().chain(&after).all(|v| v.is_finite()),
            "{tag}: non-finite output"
        );
        let first = before
            .iter()
            .zip(&after)
            .position(|(a, b)| a.to_bits() != b.to_bits());
        assert!(
            first.is_none(),
            "{tag}: first bit mismatch at {:?}",
            first.map(|i| (i, before[i], after[i]))
        );
    }

    fn assert_prefill_cache_bits(
        reference: &mut LagunaKvCache,
        candidate: &mut LagunaKvCache,
        layers: usize,
    ) {
        assert_eq!(reference.current_len(), candidate.current_len());
        let length = reference.current_len();
        for layer in 0..layers {
            let (rk, rv) = reference.view(layer, length).unwrap();
            let (ck, cv) = candidate.view(layer, length).unwrap();
            assert_prefill_tensor_bits(&format!("layer {layer} K at {length}"), &rk, &ck);
            assert_prefill_tensor_bits(&format!("layer {layer} V at {length}"), &rv, &cv);
        }
        assert_eq!(reference.layer_stored, candidate.layer_stored);
        assert_eq!(reference.layer_caps, candidate.layer_caps);
        assert_eq!(reference.layer_windows, candidate.layer_windows);
    }

    fn assert_prefill_head_omission(
        model: &Laguna,
        reference: &mut LagunaKvCache,
        candidate: &mut LagunaKvCache,
        depth: usize,
        chunk: usize,
        spec_tier: bool,
        public_prefill_api: bool,
        phase: &str,
        continuation_steps: usize,
    ) {
        reference.reset();
        candidate.reset();
        model.set_device_verify_routing(spec_tier);
        let aux_layers: Vec<usize> = (0..model.layers.len()).collect();
        let mut skipped = 0;
        let mut final_m = 0;
        for offset in (0..depth).step_by(chunk) {
            let m = chunk.min(depth - offset);
            let final_chunk = offset + m == depth;
            let ids: Vec<u32> = (offset..offset + m)
                .map(|i| ((31 * i + 17) % model.config.vocab_size) as u32)
                .collect();
            let tokens = Tensor::from_vec(ids, (1, m), model.device()).unwrap();
            let positions = Tensor::from_vec(
                (offset..offset + m).map(|p| p as i32).collect::<Vec<_>>(),
                m,
                model.device(),
            )
            .unwrap();
            let (old_logits, old_aux) = model
                .forward_with_cache_aux_scoped(
                    &tokens,
                    &positions,
                    reference,
                    &aux_layers,
                    spec_tier,
                )
                .unwrap();
            let old_logits = old_logits.to_device(&Device::Cpu).unwrap();
            let old_aux: Vec<_> = old_aux
                .iter()
                .map(|t| t.to_device(&Device::Cpu).unwrap())
                .collect();
            let (new_logits, new_aux) = if public_prefill_api {
                model
                    .prefill_with_cache_aux_scoped(
                        &tokens,
                        &positions,
                        candidate,
                        &aux_layers,
                        spec_tier,
                        final_chunk,
                    )
                    .unwrap()
            } else if final_chunk {
                let (logits, aux) = model
                    .forward_with_cache_aux_scoped(
                        &tokens,
                        &positions,
                        candidate,
                        &aux_layers,
                        spec_tier,
                    )
                    .unwrap();
                (Some(logits), aux)
            } else {
                let (_, aux) = model
                    .forward_hidden_with_cache_aux(&tokens, &positions, candidate, &aux_layers)
                    .unwrap();
                (None, aux)
            };
            assert_eq!(old_aux.len(), aux_layers.len());
            assert_eq!(new_aux.len(), old_aux.len());
            for (li, (a, b)) in old_aux.iter().zip(&new_aux).enumerate() {
                assert_prefill_tensor_bits(
                    &format!("{phase} aux layer {li} offset {offset}"),
                    a,
                    b,
                );
            }
            assert_prefill_cache_bits(reference, candidate, model.layers.len());
            if final_chunk {
                let new_logits = new_logits.expect("the final chunk must keep its full LM head");
                assert_eq!(new_logits.dims(), &[1, m, model.config.vocab_size]);
                assert_prefill_tensor_bits(
                    &format!("{phase} full final logits M={m}"),
                    &old_logits,
                    &new_logits,
                );
                final_m = m;
            } else {
                assert!(
                    new_logits.is_none(),
                    "unused intermediate LM head did not disengage"
                );
                skipped += 1;
            }
        }
        for step in 0..continuation_steps {
            let token = Tensor::from_vec(
                vec![((depth + step + 7) % model.config.vocab_size) as u32],
                (1, 1),
                model.device(),
            )
            .unwrap();
            let position =
                Tensor::from_vec(vec![(depth + step) as i32], 1, model.device()).unwrap();
            let (a, aa) = model
                .forward_with_cache_aux_scoped(&token, &position, reference, &aux_layers, spec_tier)
                .unwrap();
            let a = a.to_device(&Device::Cpu).unwrap();
            let aa: Vec<_> = aa
                .iter()
                .map(|t| t.to_device(&Device::Cpu).unwrap())
                .collect();
            let (b, ba) = model
                .forward_with_cache_aux_scoped(&token, &position, candidate, &aux_layers, spec_tier)
                .unwrap();
            assert_prefill_tensor_bits(&format!("{phase} continuation {step}"), &a, &b);
            assert_eq!(aa.len(), aux_layers.len());
            assert_eq!(ba.len(), aa.len());
            for (li, (a, b)) in aa.iter().zip(&ba).enumerate() {
                assert_prefill_tensor_bits(&format!("{phase} continuation {step} aux {li}"), a, b);
            }
            assert_prefill_cache_bits(reference, candidate, model.layers.len());
        }
        assert_eq!(skipped, depth.div_ceil(chunk) - 1);
        eprintln!("LAGUNA-PREFILL-HEAD-GATE phase={phase} depth={depth} final_m={final_m} spec_tier={spec_tier} skipped_heads={skipped} aux=bitwise visible_kv=bitwise final_logits=bitwise eager_continuation={continuation_steps}");
    }

    #[test]
    fn intermediate_prefill_heads_leave_aux_cache_and_continuation_unchanged() {
        let cfg = LagunaConfig::from_hf_json_str(TEST_CONFIG).unwrap();
        let model = build_tiny_model(&cfg, &Device::Cpu);
        let mut reference = model.new_kv_cache(512).unwrap();
        let mut candidate = model.new_kv_cache(512).unwrap();
        for spec_tier in [false, true] {
            for depth in [3, 17, 273] {
                assert_prefill_head_omission(
                    &model,
                    &mut reference,
                    &mut candidate,
                    depth,
                    16,
                    spec_tier,
                    false,
                    "cpu",
                    3,
                );
            }
        }
    }

    fn build_tiny_model(cfg: &LagunaConfig, device: &Device) -> Laguna {
        let det = |n: usize, seed: f32| -> Vec<f32> {
            (0..n)
                .map(|i| ((i as f32 * 0.618 + seed) * 0.371).sin() * 0.15)
                .collect()
        };
        let lin = |out: usize, inp: usize, seed: f32| -> Linear {
            let w = Tensor::from_vec(det(out * inp, seed), (out, inp), device).unwrap();
            Linear::new(w, None).unwrap()
        };
        let norm = |dim: usize| -> RmsNorm {
            let w = Tensor::from_vec(vec![1.0f32; dim], dim, device).unwrap();
            RmsNorm::new(w, cfg.rms_norm_eps)
        };
        let h = cfg.hidden_size;
        let hd = cfg.head_dim;
        let n_kv = cfg.num_key_value_heads;
        let mut layers = Vec::new();
        for i in 0..cfg.num_hidden_layers {
            let n_q = cfg.num_heads_for_layer(i);
            let seed = i as f32 * 100.0;
            let ffn = if cfg.is_moe_layer(i) {
                let experts = (0..cfg.num_experts)
                    .map(|e| {
                        Mlp::new(
                            lin(cfg.moe_intermediate_size, h, seed + e as f32 + 20.0),
                            lin(cfg.moe_intermediate_size, h, seed + e as f32 + 30.0),
                            lin(h, cfg.moe_intermediate_size, seed + e as f32 + 40.0),
                        )
                        .unwrap()
                    })
                    .collect();
                LagunaFfn::Moe(LagunaMoe {
                    num_experts: cfg.num_experts,
                    top_k: cfg.num_experts_per_tok,
                    norm_topk: cfg.norm_topk_prob,
                    routed_scaling: cfg.moe_routed_scaling_factor,
                    softcap: cfg.moe_router_logit_softcapping,
                    gate: lin(cfg.num_experts, h, seed + 50.0),
                    selection_bias: Tensor::zeros((1usize, cfg.num_experts), DType::F32, device)
                        .unwrap(),
                    experts,
                    shared_expert: Mlp::new(
                        lin(cfg.shared_expert_intermediate_size, h, seed + 60.0),
                        lin(cfg.shared_expert_intermediate_size, h, seed + 70.0),
                        lin(h, cfg.shared_expert_intermediate_size, seed + 80.0),
                    )
                    .unwrap(),
                    #[cfg(feature = "cuda")]
                    grouped: std::sync::Mutex::new(None),
                    #[cfg(feature = "cuda")]
                    train_stacks: std::sync::Mutex::new(None),
                })
            } else {
                LagunaFfn::Dense(
                    Mlp::new(
                        lin(cfg.intermediate_size, h, seed + 20.0),
                        lin(cfg.intermediate_size, h, seed + 30.0),
                        lin(h, cfg.intermediate_size, seed + 40.0),
                    )
                    .unwrap(),
                )
            };
            layers.push(LagunaLayer {
                kind: cfg.layer_kind(i),
                input_layernorm: norm(h),
                post_attention_layernorm: norm(h),
                self_attn: LagunaAttention {
                    kind: cfg.layer_kind(i),
                    num_heads: n_q,
                    q_proj: lin(n_q * hd, h, seed + 1.0),
                    k_proj: lin(n_kv * hd, h, seed + 2.0),
                    v_proj: lin(n_kv * hd, h, seed + 3.0),
                    o_proj: lin(h, n_q * hd, seed + 4.0),
                    g_proj: Some(lin(n_q, h, seed + 5.0)),
                    q_norm: norm(hd),
                    k_norm: norm(hd),
                    #[cfg(feature = "cuda")]
                    w8: None,
                },
                ffn,
            });
        }
        let embed_weight =
            Tensor::from_vec(det(cfg.vocab_size * h, 7.0), (cfg.vocab_size, h), device).unwrap();
        let rope_len = cfg.max_position_embeddings;
        let full_dim = cfg.rotary_dim_full();
        let full_rope = Rope::from_inv_freq(
            RopeConfig {
                head_dim: full_dim,
                max_seq_len: rope_len,
                base: cfg.full_rope_params().rope_theta,
                kind: RopeKind::Yarn,
            },
            &yarn_inv_freq(full_dim, cfg.full_rope_params()),
            device,
        )
        .unwrap();
        let sliding_dim = cfg.rotary_dim_sliding();
        let sliding_rope = Rope::from_inv_freq(
            RopeConfig {
                head_dim: sliding_dim,
                max_seq_len: rope_len,
                base: cfg.sliding_rope_params().rope_theta,
                kind: RopeKind::Standard,
            },
            &yarn_inv_freq(sliding_dim, cfg.sliding_rope_params()),
            device,
        )
        .unwrap();
        Laguna {
            config: cfg.clone(),
            embed_weight: Some(embed_weight.clone()),
            #[cfg(feature = "cuda")]
            embed_fp8: None,
            layers,
            final_norm: RmsNorm::new(
                Tensor::from_vec(vec![1.0f32; h], h, device).unwrap(),
                cfg.rms_norm_eps,
            ),
            lm_head: Linear::new(embed_weight, None).unwrap(),
            sliding_rope,
            full_rope,
            full_attn_factor: cfg.full_attention_factor(),
            dtype: DType::F32,
            device: device.clone(),
            #[cfg(feature = "cuda")]
            moe_decode_ctx: std::sync::Mutex::new(None),
            #[cfg(feature = "cuda")]
            moe_verify_ctx: std::sync::Mutex::new(std::collections::HashMap::new()),
            #[cfg(feature = "cuda")]
            moe_graphs: std::sync::Mutex::new(None),
            #[cfg(feature = "cuda")]
            moe_union_samples: std::sync::Mutex::new(Vec::new()),
            #[cfg(feature = "cuda")]
            union_probe: std::sync::atomic::AtomicBool::new(ck_profile_enabled()),
            #[cfg(feature = "cuda")]
            batch_moe_fused_dispatches: std::sync::atomic::AtomicUsize::new(0),
            #[cfg(feature = "cuda")]
            lm_head_i8: None,
            #[cfg(feature = "cuda")]
            lm_head_fp8: None,
            #[cfg(feature = "cuda")]
            lm_head_fp8_spec_off: false,
            device_verify_routing: std::sync::atomic::AtomicBool::new(false),
            attn_w8_shape: false,
            prefill_chunk: 256,
            test_moe_host_fallbacks: std::sync::atomic::AtomicUsize::new(0),
        }
    }
}
