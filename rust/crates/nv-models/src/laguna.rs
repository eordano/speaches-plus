use anyhow::{Context, Result};
use candle_core::{DType, Device, Tensor};
use nv_layers::linear::Linear;
use nv_layers::mlp::Mlp;
use nv_layers::norm::RmsNorm;
use nv_layers::rope::{Rope, RopeConfig, RopeKind};
#[cfg(feature = "cuda")]
use nv_weights::QuantScheme;
use nv_weights::QuantizationConfig;
use nv_weights::WeightLoader;
use crate::gemma4::load_rmsnorm;
use serde::Deserialize;

#[cfg(feature = "cuda")]
use std::sync::{Arc, Mutex};

use crate::gemma4::{Gemma4Cache, LayerType};
use crate::CausalLm;

#[cfg(feature = "cuda")]
type Nvfp4RunnerHandle = Option<Arc<Mutex<nv_quant::nvfp4::Nvfp4GemmRunner>>>;
#[cfg(not(feature = "cuda"))]
type Nvfp4RunnerHandle = Option<()>;

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
    let v = serde_json::Value::deserialize(deserializer)?;
    match v {
        serde_json::Value::Bool(false) => Ok(LagunaGating::None),
        serde_json::Value::Bool(true) => Ok(LagunaGating::PerElement),
        serde_json::Value::String(s) => match s.as_str() {
            "per-head" | "per_head" => Ok(LagunaGating::PerHead),
            "per-element" | "per_element" => Ok(LagunaGating::PerElement),
            other => Err(serde::de::Error::custom(format!(
                "unknown gating mode {other:?}"
            ))),
        },
        serde_json::Value::Null => Ok(LagunaGating::None),
        other => Err(serde::de::Error::custom(format!(
            "unsupported gating value {other}"
        ))),
    }
}

fn de_token_ids<'de, D>(deserializer: D) -> std::result::Result<Vec<u32>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    let v = Option::<serde_json::Value>::deserialize(deserializer)?;
    match v {
        None | Some(serde_json::Value::Null) => Ok(Vec::new()),
        Some(serde_json::Value::Number(n)) => Ok(vec![n.as_u64().unwrap_or(0) as u32]),
        Some(serde_json::Value::Array(a)) => Ok(a
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
            anyhow::bail!(
                "laguna: layer_types len {} != num_hidden_layers {}",
                cfg.layer_types.len(),
                cfg.num_hidden_layers
            );
        }
        if let Some(heads) = &cfg.num_attention_heads_per_layer {
            if heads.len() != cfg.num_hidden_layers {
                anyhow::bail!(
                    "laguna: num_attention_heads_per_layer len {} != num_hidden_layers {}",
                    heads.len(),
                    cfg.num_hidden_layers
                );
            }
            for (i, &h) in heads.iter().enumerate() {
                if h == 0 || h % cfg.num_key_value_heads != 0 {
                    anyhow::bail!(
                        "laguna: layer {i} head count {h} not divisible by kv heads {}",
                        cfg.num_key_value_heads
                    );
                }
            }
        }
        if let Some(mlp_types) = &cfg.mlp_layer_types {
            if mlp_types.len() != cfg.num_hidden_layers {
                anyhow::bail!(
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
}

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
}

#[cfg(feature = "cuda")]
fn rowquant_e4m3_dev(
    w: &Tensor,
) -> Result<(
    cudarc::driver::CudaSlice<u8>,
    cudarc::driver::CudaSlice<f32>,
)> {
    use cudarc::driver::{DevicePtr, DevicePtrMut};
    use half::bf16;
    let dev = match w.device() {
        Device::Cuda(d) => d.clone(),
        _ => anyhow::bail!("rowquant_e4m3 requires cuda"),
    };
    let (n, k) = w.dims2()?;
    let stream = nv_layers::cuda_stream::current_stream(&dev);
    let mut wq: cudarc::driver::CudaSlice<u8> =
        unsafe { stream.alloc::<u8>(n * k).map_err(|e| anyhow::anyhow!(e))? };
    let mut rs: cudarc::driver::CudaSlice<f32> =
        unsafe { stream.alloc::<f32>(n).map_err(|e| anyhow::anyhow!(e))? };
    let rc = {
        let (ws, wl) = w.storage_and_layout();
        let wc = match &*ws {
            candle_core::Storage::Cuda(s) => s,
            _ => anyhow::bail!("rowquant_e4m3: weight not cuda"),
        };
        let wsl = wc.as_cuda_slice::<bf16>()?;
        let wview = wsl.slice(wl.start_offset()..);
        let (wp, _gw) = wview.device_ptr(&stream);
        let (qp, _gq) = wq.device_ptr_mut(&stream);
        let (sp, _gs) = rs.device_ptr_mut(&stream);
        unsafe {
            nv_kernels::cuda::rowquant_e4m3(
                stream.cu_stream() as *mut _,
                wp as *const u16,
                qp as *mut u8,
                sp as *mut f32,
                n as i32,
                k as i32,
            )
        }
    };
    anyhow::ensure!(rc == 0, "rowquant_e4m3 returned {rc}");
    stream.synchronize().map_err(|e| anyhow::anyhow!(e))?;
    Ok((wq, rs))
}

#[cfg(feature = "cuda")]
#[allow(clippy::too_many_arguments)]
fn e4m3_mk_h_op(
    wq: &cudarc::driver::CudaSlice<u8>,
    row_scale: &cudarc::driver::CudaSlice<f32>,
    x: &Tensor,
    w_norm: &Tensor,
    rstd: &Tensor,
    m: usize,
    dev: &candle_core::CudaDevice,
) -> Result<Tensor> {
    use cudarc::driver::{DevicePtr, DevicePtrMut};
    use half::bf16;
    let hidden = x.elem_count() / m;
    let n_rows = wq.len() / hidden;
    anyhow::ensure!(rstd.elem_count() == m, "e4m3_mk_h: rstd rows mismatch");
    let x_c = x.reshape((m, hidden))?.contiguous()?;
    let stream = nv_layers::cuda_stream::current_stream(dev);
    let mut y_dev: cudarc::driver::CudaSlice<bf16> = unsafe {
        stream
            .alloc::<bf16>(m * n_rows)
            .map_err(|e| anyhow::anyhow!(e))?
    };
    let rc = {
        let (xs, xl) = x_c.storage_and_layout();
        let (x0, x1) = xl
            .contiguous_offsets()
            .ok_or_else(|| anyhow::anyhow!("e4m3_mk_h: x not dense"))?;
        let (ws, _wl) = w_norm.storage_and_layout();
        let (rs2, _rl) = rstd.storage_and_layout();
        let xc = match &*xs {
            candle_core::Storage::Cuda(s) => s,
            _ => anyhow::bail!("x not cuda"),
        };
        let wc = match &*ws {
            candle_core::Storage::Cuda(s) => s,
            _ => anyhow::bail!("w_norm not cuda"),
        };
        let rcu = match &*rs2 {
            candle_core::Storage::Cuda(s) => s,
            _ => anyhow::bail!("rstd not cuda"),
        };
        let xsl = xc.as_cuda_slice::<bf16>()?;
        let xview = xsl.slice(x0..x1);
        let wsl = wc.as_cuda_slice::<bf16>()?;
        let rsl = rcu.as_cuda_slice::<f32>()?;
        let (qp, _gq) = wq.device_ptr(&stream);
        let (scp, _gsc) = row_scale.device_ptr(&stream);
        let (xp, _gx) = xview.device_ptr(&stream);
        let (wp, _gw) = wsl.device_ptr(&stream);
        let (rp, _gr) = rsl.device_ptr(&stream);
        let (yp, _gy) = y_dev.device_ptr_mut(&stream);
        unsafe {
            nv_kernels::cuda::gemv_e4m3_mk_h(
                stream.cu_stream() as *mut _,
                qp as *const u8,
                scp as *const f32,
                xp as *const u16,
                wp as *const u16,
                rp as *const f32,
                yp as *mut u16,
                n_rows as i32,
                hidden as i32,
                m as i32,
            )
        }
    };
    anyhow::ensure!(rc == 0, "gemv_e4m3_mk_h returned {rc}");
    let storage = candle_core::CudaStorage::wrap_cuda_slice(y_dev, dev.clone());
    let storage = candle_core::Storage::Cuda(storage);
    Ok(Tensor::from_storage(
        storage,
        (m, n_rows),
        candle_core::op::BackpropOp::none(),
        false,
    ))
}

#[cfg(feature = "cuda")]
impl LagunaAttnW8 {
    fn quantize_proj(lin: &Linear, device: &Device, fp8: bool) -> Result<Option<LagunaProjW8>> {
        let _ = device;
        let w = match lin.weight() {
            Some(w) => w,
            None => return Ok(None),
        };
        let (n, k) = w.dims2()?;
        if k % 16 != 0 || w.dtype() != DType::BF16 {
            return Ok(None);
        }
        let max_m = nv_kernels::cuda::gemv_i8_normed_mk_max_m(k as i32).max(0) as usize;
        if max_m == 0 {
            return Ok(None);
        }
        let (wq, row_scale) = if fp8 {
            let (wq, rs) = rowquant_e4m3_dev(w)?;
            (LagunaProjBytes::E4m3(wq), rs)
        } else {
            let (wq, rs) = crate::gemma4_e4b::quantize_lm_head_i8(w)?;
            (LagunaProjBytes::I8(wq), rs)
        };
        Ok(Some(LagunaProjW8 {
            wq,
            row_scale,
            n,
            k,
            max_m: max_m.min(MAX_VERIFY_MOE_TOKENS),
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
        let (q, k, v, o) = match (
            Self::quantize_proj(q_proj, device, fp8)?,
            Self::quantize_proj(k_proj, device, fp8)?,
            Self::quantize_proj(v_proj, device, fp8)?,
            Self::quantize_proj(o_proj, device, fp8)?,
        ) {
            (Some(q), Some(k), Some(v), Some(o)) => (q, k, v, o),
            _ => return Ok(None),
        };
        let cfg_scope = attn_w8_spec_env().scope;
        let max_k = q.k.max(k.k).max(v.k).max(o.k);
        let ones_wn = Tensor::ones(max_k, DType::BF16, device)?.contiguous()?;
        let ones_rstd = Tensor::ones(MAX_VERIFY_MOE_TOKENS, DType::F32, device)?.contiguous()?;
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
        }))
    }

    pub(crate) fn scope_now(&self) -> u8 {
        self.scope.load(std::sync::atomic::Ordering::Relaxed)
    }

    pub(crate) fn slot_on(&self, bit: u8) -> bool {
        self.scope_now() & bit != 0
    }

    pub(crate) fn cfg_slot_on(&self, bit: u8) -> bool {
        self.cfg_scope & bit != 0
    }

    pub(crate) fn set_scope(&self, on: bool) {
        let v = if on { self.cfg_scope } else { 0 };
        self.scope.store(v, std::sync::atomic::Ordering::Relaxed);
    }

    pub(crate) fn is_fp8(&self) -> bool {
        self.fp8
    }

    pub(crate) fn forward(&self, p: &LagunaProjW8, x: &Tensor, m: usize) -> Result<Option<Tensor>> {
        let dev = match x.device() {
            Device::Cuda(d) => d.clone(),
            _ => return Ok(None),
        };
        if x.dtype() != DType::BF16 || x.elem_count() != m * p.k {
            return Ok(None);
        }
        let wn = self.ones_wn.narrow(0, 0, p.k)?;
        let rstd = self.ones_rstd.narrow(0, 0, m)?;
        let y = match &p.wq {
            LagunaProjBytes::I8(wq) => {
                crate::gemma4_e4b::i8_mk_h_op(wq, &p.row_scale, x, &wn, &rstd, m, &dev)?
            }
            LagunaProjBytes::E4m3(wq) => e4m3_mk_h_op(wq, &p.row_scale, x, &wn, &rstd, m, &dev)?,
        };
        Ok(Some(y.reshape((1usize, m, p.n))?))
    }
}

impl LagunaAttention {
    #[cfg(feature = "cuda")]
    pub fn m1_w8_active(&self) -> bool {
        self.w8.as_ref().map(|w| w.scope_now()).unwrap_or(0) != 0
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
        if hidden % 16 != 0 {
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
    pub grouped: std::sync::Mutex<Option<Option<Arc<nv_layers::moe_grouped::MoeGroupedWeights>>>>,
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
        if std::env::var_os("NV_LAGUNA_ROUTE_HOST_GPU_SORT").is_some() {
            let selection = scores.broadcast_add(&self.selection_bias)?;
            let (_, sorted_idx) = selection.sort_last_dim(false)?;
            let top_idx = sorted_idx.narrow(1, 0, self.top_k)?.contiguous()?;
            let mut top_weights = scores.gather(&top_idx, 1)?.contiguous()?;
            if self.norm_topk {
                let sums = top_weights.sum_keepdim(1)?;
                top_weights = top_weights.broadcast_div(&sums)?;
            }

            let routing = Tensor::cat(
                &[
                    &top_idx.to_dtype(DType::F32)?.flatten_all()?,
                    &top_weights.flatten_all()?,
                ],
                0,
            )?;
            let routing_host: Vec<f32> = routing.to_vec1()?;
            let (idx_half, w_half) = routing_host.split_at(n_tokens * self.top_k);
            return Ok((
                idx_half.iter().map(|x| *x as u32).collect(),
                w_half.to_vec(),
            ));
        }
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
        anyhow::ensure!(
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
        let dims = x_flat.dims();
        if dims.len() != 2 {
            anyhow::bail!("LagunaMoe.forward: expected [tokens, hidden], got {dims:?}");
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
    embed_weight: Tensor,
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
    moe_verify_ctx: std::sync::Mutex<
        std::collections::HashMap<usize, nv_layers::moe_grouped::GroupedDecodeContext>,
    >,
    #[cfg(feature = "cuda")]
    moe_graphs: std::sync::Mutex<Option<Option<crate::laguna_graph::LagunaMoeGraphs>>>,

    #[cfg(feature = "cuda")]
    moe_union_samples: std::sync::Mutex<Vec<(usize, usize)>>,
    #[cfg(feature = "cuda")]
    union_probe: std::sync::atomic::AtomicBool,

    #[cfg(feature = "cuda")]
    lm_head_i8: Option<LagunaLmHeadI8>,
    #[cfg(feature = "cuda")]
    lm_head_fp8: Option<LagunaLmHeadFp8>,
    #[cfg(feature = "cuda")]
    lm_head_fp8_spec_off: bool,
    device_verify_routing: std::sync::atomic::AtomicBool,
    host_moe: std::sync::atomic::AtomicBool,
    attn_w8_shape: bool,
}

fn host_moe_env() -> bool {
    std::env::var_os("NV_LAGUNA_HOST_MOE").is_some()
}

#[cfg(feature = "cuda")]
fn ck_profile_enabled() -> bool {
    std::env::var_os("NV_SPEC_CK_PROFILE").is_some()
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
        if let Ok(v) = std::env::var(key) {
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

pub fn attn_quant_gate_check() -> Result<()> {
    if std::env::var_os("NV_LAGUNA_ATTN_W8").is_some()
        && std::env::var_os("NV_LAGUNA_ATTN_FP8").is_some()
    {
        anyhow::bail!("NV_LAGUNA_ATTN_W8 and NV_LAGUNA_ATTN_FP8 are mutually exclusive; unset one");
    }
    Ok(())
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum LmHeadFp8Mode {
    ForceOff,
    ForceOn,
    DefaultScoped,
}

pub fn lmhead_fp8_mode() -> LmHeadFp8Mode {
    match std::env::var("NV_LAGUNA_LMHEAD_FP8") {
        Err(_) => LmHeadFp8Mode::DefaultScoped,
        Ok(v) if v == "0" => LmHeadFp8Mode::ForceOff,
        Ok(_) => LmHeadFp8Mode::ForceOn,
    }
}

pub fn lmhead_quant_gate_check() -> Result<()> {
    if std::env::var_os("NV_LAGUNA_LMHEAD_INT8").is_some()
        && lmhead_fp8_mode() == LmHeadFp8Mode::ForceOn
    {
        anyhow::bail!(
            "NV_LAGUNA_LMHEAD_INT8 and NV_LAGUNA_LMHEAD_FP8 are mutually exclusive; unset one"
        );
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
        &self.embed_weight
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

    pub fn lm_head_from_prenorm(&self, hidden: &Tensor) -> Result<Tensor> {
        self.lm_head_from_prenorm_scoped(hidden, false)
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
        let dev = match &self.device {
            Device::Cuda(d) => d.clone(),
            _ => anyhow::bail!("int8 lm_head requires cuda"),
        };
        let hflat = hidden.reshape((t, self.config.hidden_size))?;
        let rstd = crate::gemma4_e4b::rstd_op(&hflat, self.config.rms_norm_eps as f32)?;
        let raw = crate::gemma4_e4b::lm_head_i8_normed_mk_op(
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
        let dev = match &self.device {
            Device::Cuda(d) => d.clone(),
            _ => anyhow::bail!("fp8 lm_head requires cuda"),
        };
        let hflat = hidden.reshape((t, self.config.hidden_size))?;
        let rstd = crate::gemma4_e4b::rstd_op(&hflat, self.config.rms_norm_eps as f32)?;
        let raw = e4m3_mk_h_op(
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
                _ => anyhow::bail!("laguna NVFP4 requires a CUDA device"),
            },
            _ => None,
        };
        #[cfg(not(feature = "cuda"))]
        let nvfp4_runner: Nvfp4RunnerHandle = {
            if !matches!(qconfig.scheme, nv_weights::QuantScheme::None) {
                anyhow::bail!("laguna quantized load requires --features cuda");
            }
            None
        };

        let embed_weight = weights
            .get("model.embed_tokens.weight", dtype)
            .context("load model.embed_tokens.weight")?;
        let ed = embed_weight.dims();
        if ed.len() != 2 || ed[0] != config.vocab_size || ed[1] != config.hidden_size {
            anyhow::bail!(
                "laguna embed: expected [{}, {}], got {:?}",
                config.vocab_size,
                config.hidden_size,
                ed
            );
        }

        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for i in 0..config.num_hidden_layers {
            let layer = LagunaLayer::from_loader(
                &config,
                i,
                weights,
                qconfig,
                &nvfp4_runner,
                device,
                dtype,
            )?;
            layers.push(layer);
        }

        let host_moe = host_moe_env();
        #[cfg(feature = "cuda")]
        if matches!(qconfig.scheme, QuantScheme::Nvfp4) && !host_moe {
            if let Device::Cuda(d) = device {
                let stream = d.cuda_stream();
                for (i, layer) in layers.iter().enumerate() {
                    if let LagunaFfn::Moe(moe) = &layer.ffn {
                        let built = if nv_layers::moe_grouped::folded_shared_enabled() {
                            nv_layers::moe_grouped::MoeGroupedWeights::build_from_experts_folding_shared_as_a_fixed_extra_tile(
                                &moe.experts,
                                &moe.shared_expert,
                                config.hidden_size,
                                config.moe_intermediate_size,
                                &stream,
                            )
                        } else {
                            nv_layers::moe_grouped::MoeGroupedWeights::build_from_experts(
                                &moe.experts,
                                config.hidden_size,
                                config.moe_intermediate_size,
                                &stream,
                            )
                        };
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

        let final_norm = load_rmsnorm(
            weights,
            "model.norm.weight",
            config.hidden_size,
            config.rms_norm_eps,
            dtype,
        )?;

        let lm_head_weight = if config.tie_word_embeddings {
            embed_weight.clone()
        } else {
            weights
                .get("lm_head.weight", dtype)
                .context("load lm_head.weight")?
        };
        #[cfg(feature = "cuda")]
        let lm_head_i8 = if std::env::var_os("NV_LAGUNA_LMHEAD_INT8").is_some()
            && matches!(device, Device::Cuda(_))
        {
            let (wq, row_scale) = crate::gemma4_e4b::quantize_lm_head_i8(&lm_head_weight)?;
            let max_m = nv_kernels::cuda::gemv_i8_normed_mk_max_m(config.hidden_size as i32).max(0)
                as usize;
            anyhow::ensure!(max_m >= 1, "int8 lm_head: unsupported hidden size");
            Some(LagunaLmHeadI8 {
                wq,
                row_scale,
                max_m,
            })
        } else {
            None
        };
        #[cfg(feature = "cuda")]
        let lm_head_fp8 = if lm_head_i8.is_none()
            && lmhead_fp8_mode() != LmHeadFp8Mode::ForceOff
            && matches!(device, Device::Cuda(_))
        {
            let (wq, row_scale) = rowquant_e4m3_dev(&lm_head_weight)?;
            let max_m = nv_kernels::cuda::gemv_i8_normed_mk_max_m(config.hidden_size as i32).max(0)
                as usize;
            anyhow::ensure!(max_m >= 1, "fp8 lm_head: unsupported hidden size");
            Some(LagunaLmHeadFp8 {
                wq,
                row_scale,
                max_m,
            })
        } else {
            None
        };
        let lm_head = Linear::new(lm_head_weight, None)?;

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
            moe_verify_ctx: std::sync::Mutex::new(std::collections::HashMap::new()),
            #[cfg(feature = "cuda")]
            moe_graphs: std::sync::Mutex::new(None),
            #[cfg(feature = "cuda")]
            moe_union_samples: std::sync::Mutex::new(Vec::new()),
            #[cfg(feature = "cuda")]
            union_probe: std::sync::atomic::AtomicBool::new(ck_profile_enabled()),
            #[cfg(feature = "cuda")]
            lm_head_i8,
            #[cfg(feature = "cuda")]
            lm_head_fp8,
            #[cfg(feature = "cuda")]
            lm_head_fp8_spec_off: lmhead_fp8_mode() == LmHeadFp8Mode::DefaultScoped,
            device_verify_routing: std::sync::atomic::AtomicBool::new(false),
            host_moe: std::sync::atomic::AtomicBool::new(host_moe),
            attn_w8_shape: attn_w8_shape_env(),
        })
    }

    pub fn new_kv_cache(&self, max_seq_len: usize) -> Result<LagunaKvCache> {
        LagunaKvCache::new(&self.config, max_seq_len, &self.device, self.dtype)
    }

    #[cfg(feature = "cuda")]
    pub fn new_kv_cache_fp8(
        &self,
        max_seq_len: usize,
    ) -> Result<crate::laguna_fp8::LagunaKvCacheFp8> {
        crate::laguna_fp8::LagunaKvCacheFp8::new(
            &self.config,
            max_seq_len,
            &self.device,
            self.dtype,
        )
    }

    pub fn forward(&self, tokens: &Tensor, positions: &Tensor) -> Result<Tensor> {
        let dims = tokens.dims();
        if dims.len() != 2 || dims[0] != 1 {
            anyhow::bail!("Laguna.forward: tokens must be [1, seq], got {:?}", dims);
        }
        let seq = dims[1];
        let mut cache = self.new_kv_cache(seq.max(1))?;
        self.forward_with_cache(tokens, positions, &mut cache)
    }

    #[cfg(feature = "cuda")]
    pub fn moe_graph_stats(&self) -> Option<(usize, u64, u64)> {
        let slot = self.moe_graphs.lock().ok()?;
        match slot.as_ref() {
            Some(Some(g)) => Some((g.layers_cached(), g.captures(), g.replays())),
            _ => None,
        }
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
        let dims = tokens.dims();
        if dims.len() != 2 || dims[0] != 1 {
            anyhow::bail!("Laguna.forward: tokens must be [1, seq], got {:?}", dims);
        }
        let seq = dims[1];
        if positions.dims() != [seq] {
            anyhow::bail!(
                "Laguna.forward: positions must be [{}], got {:?}",
                seq,
                positions.dims()
            );
        }
        for &li in aux_layers {
            if li >= self.layers.len() {
                anyhow::bail!(
                    "Laguna.forward_with_cache_aux: aux layer {} out of range ({} layers)",
                    li,
                    self.layers.len()
                );
            }
        }

        self.apply_attn_w8_for(seq);
        let tokens_flat = tokens.flatten_all()?.to_dtype(DType::U32)?;
        let x = self
            .embed_weight
            .index_select(&tokens_flat, 0)?
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

        let logits = self.lm_head_from_prenorm_scoped(&hidden, spec_tier)?;
        Ok((logits.to_dtype(DType::F32)?, aux))
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

        let prof = seq == 1 && std::env::var_os("NV_MOE_DECODE_PROF").is_some();
        let sync = |label: &str, t: &mut std::time::Instant| {
            if prof {
                #[cfg(feature = "cuda")]
                if let Device::Cuda(d) = &self.device {
                    let _ = d.cuda_stream().synchronize();
                }
                eprintln!(
                    "[laguna_layer_prof] l{idx} {label}={:.3}ms",
                    t.elapsed().as_secs_f64() * 1e3
                );
                *t = std::time::Instant::now();
            }
        };
        let mut t = std::time::Instant::now();

        let normed = layer.input_layernorm.forward(x)?;
        let attn_out = self.attention_forward(idx, &normed, positions, cache, seq, new_total)?;
        let after_attn = x.add(&attn_out.to_dtype(x.dtype())?)?;
        sync("attn", &mut t);
        let out = self.ffn_forward(idx, &after_attn, seq);
        sync("ffn", &mut t);
        out
    }

    fn ffn_forward(&self, idx: usize, after_attn: &Tensor, seq: usize) -> Result<Tensor> {
        let layer = &self.layers[idx];
        #[cfg(feature = "cuda")]
        if seq == 1
            && matches!(self.device, Device::Cuda(_))
            && !self.host_moe()
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
        self.device_verify_routing
            .store(on, std::sync::atomic::Ordering::Relaxed);
    }

    pub fn set_host_moe(&self, on: bool) {
        self.host_moe
            .store(on, std::sync::atomic::Ordering::Relaxed);
    }

    pub fn host_moe(&self) -> bool {
        self.host_moe.load(std::sync::atomic::Ordering::Relaxed)
    }

    pub fn device_verify_routing(&self) -> bool {
        self.device_verify_routing
            .load(std::sync::atomic::Ordering::Relaxed)
    }

    fn moe_forward(
        &self,
        layer_idx: usize,
        moe: &LagunaMoe,
        x_flat: &Tensor,
        n_tokens: usize,
    ) -> Result<Tensor> {
        #[cfg(feature = "cuda")]
        if matches!(self.device, Device::Cuda(_)) && !self.host_moe() {
            if n_tokens == 1 {
                if let Some(out) = self.moe_forward_decode_grouped(layer_idx, moe, x_flat)? {
                    return Ok(out);
                }
            } else {
                if n_tokens <= MAX_VERIFY_MOE_TOKENS
                    && self.device_verify_routing()
                    && std::env::var_os("NV_LAGUNA_HOST_VERIFY").is_none()
                {
                    match self.moe_forward_verify_grouped(layer_idx, moe, x_flat, n_tokens) {
                        Ok(Some(out)) => return Ok(out),
                        Ok(None) => {}
                        Err(e) => {
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
        let out = moe.forward(x_flat)?;
        let finite_probe = out.to_dtype(DType::F32)?.sum_all()?.to_scalar::<f32>()?;
        if !finite_probe.is_finite() {
            anyhow::bail!(
                "laguna host-MoE fallback at layer {layer_idx} produced non-finite output \
                 (sum {finite_probe}); refusing to emit NaN logits. This failure has been \
                 observed under GPU memory pressure (co-tenant holding ~43 GiB); free VRAM \
                 and retry instead of trusting this forward pass"
            );
        }
        Ok(out)
    }

    #[cfg(feature = "cuda")]
    fn moe_forward_verify_grouped(
        &self,
        layer_idx: usize,
        moe: &LagunaMoe,
        x_flat: &Tensor,
        n_tokens: usize,
    ) -> Result<Option<Tensor>> {
        let dev = match &self.device {
            Device::Cuda(d) => d.clone(),
            _ => return Ok(None),
        };
        let grouped = match self.grouped_weights(moe)? {
            Some(w) => w,
            None => return Ok(None),
        };
        let mut map = self
            .moe_verify_ctx
            .lock()
            .map_err(|e| anyhow::anyhow!("moe_verify_ctx mutex poisoned: {e}"))?;
        let ctx = match map.entry(n_tokens) {
            std::collections::hash_map::Entry::Occupied(e) => e.into_mut(),
            std::collections::hash_map::Entry::Vacant(v) => {
                let stream = nv_layers::cuda_stream::current_stream(&dev);
                v.insert(nv_layers::moe_grouped::GroupedDecodeContext::new_multi(
                    self.config.hidden_size,
                    self.config.moe_intermediate_size,
                    self.config.num_experts_per_tok,
                    self.config.num_experts,
                    n_tokens,
                    &stream,
                )?)
            }
        };

        let logits = moe.gate.forward(x_flat)?;
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
        if std::env::var_os("NV_LAGUNA_ROUTE_AB_PROBE").is_some() {
            self.route_ab_probe_report_device_vs_host_choices_on_identical_input(
                layer_idx, moe, x_flat, ctx, n_tokens, &dev,
            )?;
        }
        if self.union_probe.load(std::sync::atomic::Ordering::Relaxed) {
            let stream = nv_layers::cuda_stream::current_stream(&dev);
            #[allow(deprecated)]
            if let Ok(ids) = stream.memcpy_dtov(&ctx.topk_ids) {
                let mut seen = std::collections::HashSet::new();
                for &e in &ids {
                    seen.insert(e);
                }
                if let Ok(mut samples) = self.moe_union_samples.lock() {
                    samples.push((n_tokens, seen.len()));
                }
            }
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
        #[allow(deprecated)]
        let dev_ids: Vec<i32> = stream
            .memcpy_dtov(&ctx.topk_ids)
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
    pub fn drain_union_samples(&self) -> Vec<(usize, usize)> {
        match self.moe_union_samples.lock() {
            Ok(mut s) => std::mem::take(&mut *s),
            Err(_) => Vec::new(),
        }
    }

    #[cfg(feature = "cuda")]
    pub fn set_union_probe(&self, on: bool) {
        self.union_probe
            .store(on, std::sync::atomic::Ordering::Relaxed);
    }

    #[cfg(feature = "cuda")]
    pub(crate) fn grouped_weights(
        &self,
        moe: &LagunaMoe,
    ) -> Result<Option<Arc<nv_layers::moe_grouped::MoeGroupedWeights>>> {
        let dev = match &self.device {
            Device::Cuda(d) => d.clone(),
            _ => return Ok(None),
        };
        let mut slot = moe
            .grouped
            .lock()
            .map_err(|e| anyhow::anyhow!("LagunaMoe.grouped mutex poisoned: {e}"))?;
        if slot.is_none() {
            let stream = nv_layers::cuda_stream::current_stream(&dev);
            let built = if nv_layers::moe_grouped::folded_shared_enabled() {
                nv_layers::moe_grouped::MoeGroupedWeights::build_from_experts_folding_shared_as_a_fixed_extra_tile(
                    &moe.experts,
                    &moe.shared_expert,
                    self.config.hidden_size,
                    self.config.moe_intermediate_size,
                    &stream,
                )
            } else {
                nv_layers::moe_grouped::MoeGroupedWeights::build_from_experts(
                    &moe.experts,
                    self.config.hidden_size,
                    self.config.moe_intermediate_size,
                    &stream,
                )
            };
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
        const MIN_TILE: usize = nv_quant::nvfp4::MIN_TILE;
        const PREFILL_SUBRANGE: usize = 256;

        let grouped = match self.grouped_weights(moe)? {
            Some(w) => w,
            None => return Ok(None),
        };
        let k = moe.top_k;
        let (ids, weights) = moe.route_host(x_flat, n_tokens)?;

        let mut ranges: Vec<(usize, usize)> = Vec::new();
        let mut start = 0usize;
        while start < n_tokens {
            let len = PREFILL_SUBRANGE.min(n_tokens - start);
            ranges.push((start, len));
            start += len;
        }
        ranges.reverse();

        let mut outs: Vec<Tensor> = Vec::new();
        let mut counts = vec![0u32; moe.num_experts];
        while let Some((start, len)) = ranges.pop() {
            if len > MIN_TILE {
                counts.iter_mut().for_each(|c| *c = 0);
                for &e in &ids[start * k..(start + len) * k] {
                    counts[e as usize] += 1;
                }
                if counts.iter().any(|&c| c as usize > MIN_TILE) {
                    let half = len / 2;
                    ranges.push((start + half, len - half));
                    ranges.push((start, half));
                    continue;
                }
            }
            let x_sub = x_flat.narrow(0, start, len)?;
            match nv_layers::moe_grouped::forward_grouped(
                &grouped,
                &grouped,
                &x_sub,
                &ids[start * k..(start + len) * k],
                &weights[start * k..(start + len) * k],
                len,
                k,
                &self.device,
            ) {
                Ok(t) => outs.push(t),
                Err(e) => {
                    eprintln!("[laguna] grouped MoE prefill failed, host fallback: {e}");
                    return Ok(None);
                }
            }
        }

        let mut acc = if outs.len() == 1 {
            outs.pop().unwrap()
        } else {
            let refs: Vec<&Tensor> = outs.iter().collect();
            Tensor::cat(&refs, 0)?
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
        let dev = match &self.device {
            Device::Cuda(d) => d.clone(),
            _ => return Ok(None),
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
                        eprintln!("[laguna] moe graph init failed, uncaptured path: {e}");
                        None
                    }
                },
            );
        }
        if let Some(Some(graphs)) = g_slot.as_mut() {
            if !graphs.failed() {
                let hidden = self.config.hidden_size;
                let resid = after_attn.reshape((1usize, hidden))?;
                match graphs.forward_layer(layer_idx, moe, norm, &grouped, ctx, &resid, &dev) {
                    Ok(out) => {
                        return Ok(Some(out.reshape((1usize, 1usize, hidden))?));
                    }
                    Err(e) => {
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
        let dev = match &self.device {
            Device::Cuda(d) => d.clone(),
            _ => return Ok(None),
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

        let prof = std::env::var_os("NV_MOE_DECODE_PROF").is_some();
        let sync = |label: &str, t: &mut std::time::Instant| {
            if prof {
                let _ = dev.cuda_stream().synchronize();
                eprintln!(
                    "[laguna_moe_prof] {label}={:.3}ms",
                    t.elapsed().as_secs_f64() * 1e3
                );
                *t = std::time::Instant::now();
            }
        };
        let mut t = std::time::Instant::now();
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

        let q_raw = attn.proj_q(x, seq)?;
        let q = q_raw.reshape((1usize, seq, n_q, head_dim))?;
        let q_normed = attn.q_norm.forward(&q)?;
        let k_raw = attn.proj_k(x, seq)?;
        let k = k_raw.reshape((1usize, seq, n_kv, head_dim))?;
        let k_normed = attn.k_norm.forward(&k)?;
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
        let (q_rot, k_rot) = apply_partial_rope_scaled(
            &q_normed, &k_normed, rope, positions, rotary_dim, head_dim, rot_scale,
        )?;
        let q_rot = q_rot.to_dtype(self.dtype)?.contiguous()?;
        let k_rot = k_rot.to_dtype(self.dtype)?.contiguous()?;
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

        let gated = match (&attn.g_proj, self.config.gating) {
            (Some(g_proj), LagunaGating::PerHead) => {
                let g = softplus_f32(&g_proj.forward(x)?.to_dtype(DType::F32)?)?;
                let g = g
                    .reshape((1usize, seq, n_q, 1usize))?
                    .to_dtype(attn_out.dtype())?;
                attn_out.broadcast_mul(&g)?
            }
            (Some(g_proj), LagunaGating::PerElement) => {
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

    pub fn forward_decode_batched<C: Gemma4Cache>(
        &self,
        tokens: &[u32],
        positions: &[usize],
        caches: &mut [&mut C],
    ) -> Result<Tensor> {
        let b = tokens.len();
        if b == 0 {
            anyhow::bail!("Laguna.forward_decode_batched: empty batch");
        }
        if positions.len() != b || caches.len() != b {
            anyhow::bail!(
                "Laguna.forward_decode_batched: ragged batch tokens={} positions={} caches={}",
                b,
                positions.len(),
                caches.len()
            );
        }
        self.apply_attn_w8_for(b);

        let tokens_t = Tensor::from_vec(tokens.to_vec(), b, &self.device)?.to_dtype(DType::U32)?;
        let mut hidden = self
            .embed_weight
            .index_select(&tokens_t, 0)?
            .reshape((1usize, b, self.config.hidden_size))?
            .to_dtype(self.dtype)?;

        for (i, cache) in caches.iter_mut().enumerate() {
            let len = cache.current_len();
            if len != positions[i] {
                anyhow::bail!(
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
        let attn_out =
            self.attention_forward_decode_batched(idx, &normed, positions, caches, b)?;
        let after_attn = x.add(&attn_out.to_dtype(x.dtype())?)?;
        self.ffn_forward_decode_batched(idx, &after_attn, b)
    }

    fn ffn_forward_decode_batched(&self, idx: usize, after_attn: &Tensor, b: usize) -> Result<Tensor> {
        #[cfg(feature = "cuda")]
        if b > 1
            && b <= MAX_VERIFY_MOE_TOKENS
            && matches!(self.device, Device::Cuda(_))
            && !self.host_moe()
        {
            if let LagunaFfn::Moe(moe) = &self.layers[idx].ffn {
                let normed_mlp = self.layers[idx]
                    .post_attention_layernorm
                    .forward(after_attn)?;
                let flat = normed_mlp
                    .reshape((b, self.config.hidden_size))?
                    .contiguous()?;
                match self.moe_forward_verify_grouped(idx, moe, &flat, b) {
                    Ok(Some(out)) => {
                        let out = out.reshape((1usize, b, self.config.hidden_size))?;
                        return after_attn
                            .add(&out.to_dtype(after_attn.dtype())?)
                            .map_err(Into::into);
                    }
                    Ok(None) => {}
                    Err(e) => eprintln!(
                        "[laguna] batched decode verify-grouped MoE failed at layer {idx}, \
                         standard route: {e:#}"
                    ),
                }
            }
        }
        self.ffn_forward(idx, after_attn, b)
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
        let (q_rot, k_rot) = apply_partial_rope_scaled(
            &q_normed, &k_normed, rope, positions, rotary_dim, head_dim, rot_scale,
        )?;
        let q_rot = q_rot.to_dtype(self.dtype)?.contiguous()?;
        let k_rot = k_rot.to_dtype(self.dtype)?.contiguous()?;
        let v_for_cache = v.contiguous()?;

        let window = match kind {
            LayerType::SlidingAttention => Some(self.config.sliding_window),
            LayerType::FullAttention => None,
        };
        let scale = (head_dim as f32).powf(-0.5);

        let mut rows: Vec<Tensor> = Vec::with_capacity(b);
        for i in 0..b {
            let q_i = q_rot.narrow(1, i, 1)?.contiguous()?;
            let k_i = k_rot.narrow(1, i, 1)?.contiguous()?;
            let v_i = v_for_cache.narrow(1, i, 1)?.contiguous()?;

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
        let row_refs: Vec<&Tensor> = rows.iter().collect();
        let attn_out = Tensor::cat(&row_refs, 1)?;

        let gated = match (&attn.g_proj, self.config.gating) {
            (Some(g_proj), LagunaGating::PerHead) => {
                let g = softplus_f32(&g_proj.forward(x)?.to_dtype(DType::F32)?)?;
                let g = g
                    .reshape((1usize, b, n_q, 1usize))?
                    .to_dtype(attn_out.dtype())?;
                attn_out.broadcast_mul(&g)?
            }
            (Some(g_proj), LagunaGating::PerElement) => {
                let g = softplus_f32(&g_proj.forward(x)?.to_dtype(DType::F32)?)?;
                let g = g
                    .reshape((1usize, b, n_q, head_dim))?
                    .to_dtype(attn_out.dtype())?;
                attn_out.broadcast_mul(&g)?
            }
            _ => attn_out,
        };

        let attn_flat = gated.reshape((1usize, b, n_q * head_dim))?;
        attn.proj_o(&attn_flat, b)
    }
}

impl LagunaLayer {
    #[allow(clippy::too_many_arguments)]
    fn from_loader(
        config: &LagunaConfig,
        idx: usize,
        weights: &WeightLoader,
        qconfig: &QuantizationConfig,
        nvfp4_runner: &Nvfp4RunnerHandle,
        device: &Device,
        dtype: DType,
    ) -> Result<Self> {
        let prefix = format!("model.layers.{idx}");
        let kind = config.layer_kind(idx);
        let n_q = config.num_heads_for_layer(idx);
        let n_kv = config.num_key_value_heads;
        let head_dim = config.head_dim;
        let hidden = config.hidden_size;

        let input_layernorm = load_rmsnorm(
            weights,
            &format!("{prefix}.input_layernorm.weight"),
            hidden,
            config.rms_norm_eps,
            dtype,
        )?;
        let post_attention_layernorm = load_rmsnorm(
            weights,
            &format!("{prefix}.post_attention_layernorm.weight"),
            hidden,
            config.rms_norm_eps,
            dtype,
        )?;

        let q_proj = load_linear_plain(
            weights,
            &format!("{prefix}.self_attn.q_proj.weight"),
            n_q * head_dim,
            hidden,
            dtype,
        )?;
        let k_proj = load_linear_plain(
            weights,
            &format!("{prefix}.self_attn.k_proj.weight"),
            n_kv * head_dim,
            hidden,
            dtype,
        )?;
        let v_proj = load_linear_plain(
            weights,
            &format!("{prefix}.self_attn.v_proj.weight"),
            n_kv * head_dim,
            hidden,
            dtype,
        )?;
        let o_proj = load_linear_plain(
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
            )?),
            LagunaGating::PerElement => Some(load_linear_plain(
                weights,
                &format!("{prefix}.self_attn.g_proj.weight"),
                n_q * head_dim,
                hidden,
                dtype,
            )?),
        };
        let q_norm = load_rmsnorm(
            weights,
            &format!("{prefix}.self_attn.q_norm.weight"),
            head_dim,
            config.rms_norm_eps,
            dtype,
        )?;
        let k_norm = load_rmsnorm(
            weights,
            &format!("{prefix}.self_attn.k_norm.weight"),
            head_dim,
            config.rms_norm_eps,
            dtype,
        )?;

        #[cfg(feature = "cuda")]
        let w8 = {
            let fp8 = std::env::var_os("NV_LAGUNA_ATTN_FP8").is_some();
            if (fp8 || std::env::var_os("NV_LAGUNA_ATTN_W8").is_some())
                && matches!(device, Device::Cuda(_))
                && dtype == DType::BF16
            {
                LagunaAttnW8::build(&q_proj, &k_proj, &v_proj, &o_proj, device, fp8)?
            } else {
                None
            }
        };

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
            let se_gate = load_linear_expert(
                weights,
                &format!("{sprefix}.gate_proj"),
                config.shared_expert_intermediate_size,
                hidden,
                dtype,
                qconfig,
                nvfp4_runner,
                device,
            )?;
            let se_up = load_linear_expert(
                weights,
                &format!("{sprefix}.up_proj"),
                config.shared_expert_intermediate_size,
                hidden,
                dtype,
                qconfig,
                nvfp4_runner,
                device,
            )?;
            let se_down = load_linear_expert(
                weights,
                &format!("{sprefix}.down_proj"),
                hidden,
                config.shared_expert_intermediate_size,
                dtype,
                qconfig,
                nvfp4_runner,
                device,
            )?;
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
            })
        } else {
            let gate_proj = load_linear_plain(
                weights,
                &format!("{prefix}.mlp.gate_proj.weight"),
                config.intermediate_size,
                hidden,
                dtype,
            )?;
            let up_proj = load_linear_plain(
                weights,
                &format!("{prefix}.mlp.up_proj.weight"),
                config.intermediate_size,
                hidden,
                dtype,
            )?;
            let down_proj = load_linear_plain(
                weights,
                &format!("{prefix}.mlp.down_proj.weight"),
                hidden,
                config.intermediate_size,
                dtype,
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

#[allow(clippy::too_many_arguments)]
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
        anyhow::bail!(
            "laguna kv: sliding layer {layer} write of {t} tokens exceeds capacity {cap}"
        );
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
}

impl LagunaKvCache {
    pub fn new(
        config: &LagunaConfig,
        max_seq_len: usize,
        device: &Device,
        dtype: DType,
    ) -> Result<Self> {
        let ring = cfg!(feature = "cuda")
            && matches!(device, Device::Cuda(_))
            && dtype == DType::BF16
            && std::env::var_os("NV_LAGUNA_HOST_KV").is_none();
        Self::new_with_mode(config, max_seq_len, device, dtype, ring)
    }

    pub fn new_with_mode(
        config: &LagunaConfig,
        max_seq_len: usize,
        device: &Device,
        dtype: DType,
        ring: bool,
    ) -> Result<Self> {
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
                    (max_seq_len.min(w + SLIDING_COMPACT_SLACK), Some(w))
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
            let dev = match device {
                Device::Cuda(d) => d.clone(),
                _ => anyhow::bail!("LagunaKvCache ring mode requires a CUDA device"),
            };
            if dtype != DType::BF16 {
                anyhow::bail!("LagunaKvCache ring mode requires BF16, got {dtype:?}");
            }
            let mut s_cap = 0usize;
            let mut s_window = 0usize;
            for (cap, window) in layer_caps.iter().zip(layer_windows.iter()) {
                if let Some(w) = window {
                    if s_cap != 0 && (s_cap != *cap || s_window != *w) {
                        anyhow::bail!(
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
            let meta_dev = stream
                .alloc_zeros::<i32>(4)
                .map_err(|e| anyhow::anyhow!(e))?;
            Some(LagunaKvRing {
                dev,
                meta_dev,
                host_meta: Box::new([0i32; 4]),
                s_stored: 0,
                full_stored: 0,
                s_cap,
                s_window,
                ring_attn: std::env::var_os("NV_LAGUNA_RING_ATTN").is_some(),
            })
        } else {
            None
        };
        #[cfg(not(feature = "cuda"))]
        if ring {
            anyhow::bail!("LagunaKvCache ring mode requires the cuda feature");
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
            anyhow::bail!(
                "LagunaKvCache.rollback: n {} > current_len {}",
                n,
                self.current_len
            );
        }
        for (layer, stored) in self.layer_stored.iter_mut().enumerate() {
            if *stored < n {
                anyhow::bail!(
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
                anyhow::bail!(
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
#[allow(clippy::too_many_arguments)]
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
        anyhow::bail!("{cache_name}.prepare_for_decode: n_total {n_total} < write_pos {write_pos}");
    }
    if n_total > max_seq_len {
        anyhow::bail!(
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
            anyhow::bail!(
                "{ring_name}: write of {n_new} tokens exceeds sliding capacity {s_cap}"
            );
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
    stream: &std::sync::Arc<cudarc::driver::CudaStream>,
    buf: &Tensor,
    shift: usize,
    keep: usize,
    n_kv: usize,
    head_dim: usize,
) -> Result<()> {
    use cudarc::driver::DevicePtr;
    let (st, l) = buf.storage_and_layout();
    let cuda = match &*st {
        candle_core::Storage::Cuda(s) => s,
        _ => anyhow::bail!("kv shift: buffer must be on CUDA"),
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
            anyhow::bail!("kv_shift_bf16 rc={rc}");
        }
        off += n;
    }
    Ok(())
}

#[cfg(feature = "cuda")]
#[allow(clippy::too_many_arguments)]
pub(crate) fn ring_append_launch(
    stream: &std::sync::Arc<cudarc::driver::CudaStream>,
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
        candle_core::Storage::Cuda(s) => s,
        _ => anyhow::bail!("ring append: src must be on CUDA"),
    };
    let dst_cuda = match &*dst_st {
        candle_core::Storage::Cuda(s) => s,
        _ => anyhow::bail!("ring append: dst must be on CUDA"),
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
        anyhow::bail!("kv_ring_append_bf16 rc={rc}");
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
            anyhow::bail!(
                "LagunaKvCache.write_at layer {layer}: expected [1, t, {n_kv}, {head_dim}], got {:?}",
                dims
            );
        }
        if v_new.dims() != dims {
            anyhow::bail!(
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
            anyhow::bail!(
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
            anyhow::bail!("LagunaKvCache.view: layer {layer} out of range");
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
    fn try_decode_attention_gqa(
        &mut self,
        layer: usize,
        q_rot: &Tensor,
        n_q: usize,
        sliding_window: Option<usize>,
        scaling: f32,
    ) -> Result<Option<Tensor>> {
        use cudarc::driver::{DevicePtr, DevicePtrMut};
        if !crate::laguna_step_graph::m1_flash_enabled() {
            return Ok(None);
        }
        let head_dim = self.head_dim;
        let n_kv = self.n_kv;
        if head_dim != 128 || n_kv == 0 || n_q % n_kv != 0 || !matches!(n_q / n_kv, 6 | 8) {
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
        anyhow::ensure!(
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

        let mut out = unsafe {
            stream
                .alloc::<half::bf16>(expected)
                .map_err(|e| anyhow::anyhow!(e))?
        };
        let q_c = q_rot.contiguous()?;
        let (q_st, ql) = q_c.storage_and_layout();
        let q_cuda = match &*q_st {
            candle_core::Storage::Cuda(s) => s,
            _ => anyhow::bail!("q_rot must be on CUDA"),
        };
        let q_view = q_cuda
            .as_cuda_slice::<half::bf16>()?
            .slice(ql.start_offset()..);
        let (k_buf, v_buf) = &self.layers[layer];
        let (k_st, kl) = k_buf.storage_and_layout();
        let (v_st, vl) = v_buf.storage_and_layout();
        let k_cuda = match &*k_st {
            candle_core::Storage::Cuda(s) => s,
            _ => anyhow::bail!("kv ring buffers must be on CUDA"),
        };
        let v_cuda = match &*v_st {
            candle_core::Storage::Cuda(s) => s,
            _ => anyhow::bail!("kv ring buffers must be on CUDA"),
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
        anyhow::ensure!(rc == 0, "laguna_flash_decode_gqa rc={rc}");
        drop(q_st);
        let storage = candle_core::CudaStorage::wrap_cuda_slice(out, dev);
        let tensor = candle_core::Tensor::from_storage(
            candle_core::Storage::Cuda(storage),
            (1usize, 1usize, n_q, head_dim),
            candle_core::op::BackpropOp::none(),
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
                anyhow::bail!(
                    "LagunaKvCache.try_decode_attention_ring layer {layer}: expected {expected} \
                     elements, got dims {:?}",
                    q_rot.dims()
                );
            }
            let stream = nv_layers::cuda_stream::current_stream(&ring.dev);
            let mut out = unsafe {
                stream
                    .alloc::<half::bf16>(expected)
                    .map_err(|e| anyhow::anyhow!(e))?
            };
            let q_c = q_rot.contiguous()?;
            let (q_st, ql) = q_c.storage_and_layout();
            let q_cuda = match &*q_st {
                candle_core::Storage::Cuda(s) => s,
                _ => anyhow::bail!("q_rot must be on CUDA"),
            };
            let q_view = q_cuda
                .as_cuda_slice::<half::bf16>()?
                .slice(ql.start_offset()..);
            let (k_buf, v_buf) = &self.layers[layer];
            let (k_st, kl) = k_buf.storage_and_layout();
            let (v_st, vl) = v_buf.storage_and_layout();
            let k_cuda = match &*k_st {
                candle_core::Storage::Cuda(s) => s,
                _ => anyhow::bail!("kv ring buffers must be on CUDA"),
            };
            let v_cuda = match &*v_st {
                candle_core::Storage::Cuda(s) => s,
                _ => anyhow::bail!("kv ring buffers must be on CUDA"),
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
                anyhow::bail!("attention_bf16_decode_ring rc={rc}");
            }
            drop(q_st);
            let dev = ring.dev.clone();
            let storage = candle_core::CudaStorage::wrap_cuda_slice(out, dev);
            let tensor = candle_core::Tensor::from_storage(
                candle_core::Storage::Cuda(storage),
                (1usize, 1usize, n_q, head_dim),
                candle_core::op::BackpropOp::none(),
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

#[allow(clippy::too_many_arguments)]
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

#[allow(clippy::too_many_arguments)]
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
    if matches!(q.device(), Device::Cuda(_)) {
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

#[allow(clippy::too_many_arguments)]
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

fn load_linear_plain(
    weights: &WeightLoader,
    name: &str,
    out_features: usize,
    in_features: usize,
    dtype: DType,
) -> Result<Linear> {
    let w = weights
        .get(name, dtype)
        .with_context(|| format!("load {name}"))?;
    let d = w.dims();
    if d.len() != 2 || d[0] != out_features || d[1] != in_features {
        anyhow::bail!(
            "linear {name}: expected [{}, {}], got {:?}",
            out_features,
            in_features,
            d
        );
    }
    Linear::new(w, None)
}

#[allow(clippy::too_many_arguments)]
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
    let _ = (qconfig, device, nvfp4_runner);
    load_linear_plain(
        weights,
        &format!("{module}.weight"),
        out_features,
        in_features,
        dtype,
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
    fn config_parses() {
        let cfg = LagunaConfig::from_hf_json_str(TEST_CONFIG).unwrap();
        assert_eq!(cfg.num_hidden_layers, 4);
        assert_eq!(cfg.gating, LagunaGating::PerHead);
        assert_eq!(cfg.layer_kind(0), LayerType::FullAttention);
        assert_eq!(cfg.layer_kind(1), LayerType::SlidingAttention);
        assert!(!cfg.is_moe_layer(0));
        assert!(cfg.is_moe_layer(1));
        assert_eq!(cfg.rotary_dim_full(), 4);
        assert_eq!(cfg.rotary_dim_sliding(), 8);
        assert_eq!(cfg.eos_token_id, vec![2, 24]);
        assert!((cfg.full_attention_factor() - 1.3465736).abs() < 1e-6);
        assert!((cfg.moe_routed_scaling_factor - 2.5).abs() < 1e-6);
    }

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

    #[test]
    fn softplus_matches_reference() {
        let device = Device::Cpu;
        let xs = vec![-30.0f32, -2.0, -0.5, 0.0, 0.5, 2.0, 30.0, 90.0];
        let t = Tensor::from_vec(xs.clone(), xs.len(), &device).unwrap();
        let out = softplus_f32(&t).unwrap().to_vec1::<f32>().unwrap();
        for (x, y) in xs.iter().zip(out.iter()) {
            let expected = if *x > 20.0 {
                *x
            } else {
                (1.0 + (*x as f64).exp()).ln() as f32
            };
            assert!(
                (y - expected).abs() < 1e-4,
                "softplus({x}) = {y}, expected {expected}"
            );
            assert!(y.is_finite());
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
        }
    }

    #[test]
    fn moe_router_weights_are_unbiased_and_normalized() {
        let device = Device::Cpu;
        let moe = make_moe(&device);
        let x = Tensor::from_vec(
            (0..16)
                .map(|i| (i as f32 * 0.3).cos())
                .collect::<Vec<f32>>(),
            (2usize, 8usize),
            &device,
        )
        .unwrap();
        let out = moe.forward(&x).unwrap();
        assert_eq!(out.dims(), &[2, 8]);

        let logits = moe.gate.forward(&x).unwrap().to_dtype(DType::F32).unwrap();
        let scores = candle_nn::ops::sigmoid(&logits).unwrap();
        let scores_host: Vec<Vec<f32>> = scores.to_vec2().unwrap();
        for row in &scores_host {
            let mut sorted: Vec<f32> = row.clone();
            sorted.sort_by(|a, b| b.partial_cmp(a).unwrap());
            let top_sum: f32 = sorted[..2].iter().sum();
            assert!(top_sum > 0.0);
        }
    }

    #[test]
    fn moe_selection_bias_changes_expert_choice_but_not_weight_source() {
        let device = Device::Cpu;
        let mut moe = make_moe(&device);
        let x = Tensor::from_vec(
            (0..8).map(|i| (i as f32 * 0.3).cos()).collect::<Vec<f32>>(),
            (1usize, 8usize),
            &device,
        )
        .unwrap();
        let out_unbiased = moe.forward(&x).unwrap().to_vec2::<f32>().unwrap();

        let bias =
            Tensor::from_vec(vec![100.0f32, 100.0, 0.0, 0.0], (1usize, 4usize), &device).unwrap();
        moe.selection_bias = bias;
        let out_biased = moe.forward(&x).unwrap().to_vec2::<f32>().unwrap();

        let diff: f32 = out_unbiased[0]
            .iter()
            .zip(out_biased[0].iter())
            .map(|(a, b)| (a - b).abs())
            .sum();
        assert!(
            diff > 1e-6,
            "large selection bias must redirect routing to experts 0/1"
        );
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
    fn kv_cache_sliding_layer_is_bounded() {
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
    fn tiny_model_forward_shapes() {
        let cfg = LagunaConfig::from_hf_json_str(TEST_CONFIG).unwrap();
        let device = Device::Cpu;
        let model = build_tiny_model(&cfg, &device);
        let tokens = Tensor::from_vec(vec![1u32, 5, 9], (1usize, 3usize), &device).unwrap();
        let positions = Tensor::from_vec(vec![0u32, 1, 2], 3usize, &device).unwrap();
        let mut cache = model.new_kv_cache(64).unwrap();
        let logits = model
            .forward_with_cache(&tokens, &positions, &mut cache)
            .unwrap();
        assert_eq!(logits.dims(), &[1, 3, cfg.vocab_size]);
        assert_eq!(cache.current_len(), 3);

        let next = Tensor::from_vec(vec![2u32], (1usize, 1usize), &device).unwrap();
        let next_pos = Tensor::from_vec(vec![3u32], 1usize, &device).unwrap();
        let logits2 = model
            .forward_with_cache(&next, &next_pos, &mut cache)
            .unwrap();
        assert_eq!(logits2.dims(), &[1, 1, cfg.vocab_size]);
        assert_eq!(cache.current_len(), 4);
        let host = logits2.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        assert!(host.iter().all(|x| x.is_finite()));
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
            embed_weight: embed_weight.clone(),
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
            lm_head_i8: None,
            #[cfg(feature = "cuda")]
            lm_head_fp8: None,
            #[cfg(feature = "cuda")]
            lm_head_fp8_spec_off: false,
            device_verify_routing: std::sync::atomic::AtomicBool::new(false),
            host_moe: std::sync::atomic::AtomicBool::new(host_moe_env()),
            attn_w8_shape: false,
        }
    }
}
