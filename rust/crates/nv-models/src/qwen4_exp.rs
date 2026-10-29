use anyhow::{bail, ensure, Context, Result};
use candle_core::{CudaDevice, DType, Device, Storage, Tensor, D};
use cudarc::driver::{CudaSlice, CudaStream, DevicePtr, DevicePtrMut};
use half::bf16;
use nv_kernels::cuda as nvk;
use nv_layers::attn::{flash_attn, AttnConfig};
use nv_layers::linear::Linear;
use nv_layers::linear_attn::{LinAttnState, LinearAttention, LinearAttentionConfig};
use nv_layers::moe_grouped::{
    forward_grouped_device_routing,
    route_topk_and_plan_padded_on_device_for_prefill_dispatch, swizzled_scale_bytes,
    GroupedDecodeContext, MoeGroupedWeights,
};
use nv_layers::rope::{Rope, RopeConfig, RopeKind};
use nv_quant::nvfp4::Nvfp4GemmRunner;
use nv_weights::WeightLoader;
use serde_json::Value;
use std::ffi::c_void;
use std::path::Path;
use std::sync::{Arc, Mutex};

pub const FUSED_GLUE_ROWS_MAX_8_THE_HC_AND_ATTENTION_ROW_KERNELS_ARE_SIZED_FOR_DECODE_AND_VERIFY_ROWS: usize = 8;
pub const HC_FUSED_ROWS_MAX_64_BELOW_WHICH_THE_DOT_GRID_BEATS_THREE_CUBLAS_CALLS_PER_MIX: usize = 64;
pub const STAGED_MOE_PREFILL_ROWS_MAX_16_BELOW_WHICH_THE_GEMV_CHAIN_BEATS_THE_GROUPED_GEMM_DISPATCH: usize = 16;
pub const SPARSE_ATTENTION_ROW_CHUNK_1024_BOUNDS_THE_SCORE_SCRATCH_AT_256_MIB_FOR_A_262K_CONTEXT: usize = 1024;
pub const NV_Q4EXP_EAGER_GLUE_1_ROUTES_HC_AND_ROW_ATTENTION_THROUGH_CANDLE_OPS_FOR_KERNEL_PARITY_CHECKS: &str =
    "NV_Q4EXP_EAGER_GLUE";
pub const NV_Q4EXP_PLE_HOST_1_GATHERS_THE_NGRAM_ROWS_ON_THE_CPU_FROM_THE_MMAP_INSTEAD_OF_PINNING_51_GB: &str =
    "NV_Q4EXP_PLE_HOST";
pub const MOE_FUSED_TAIL_ROWS_MAX_6_MATCHES_THE_E4M3_MK_ROWS2_FAMILY_THE_SHARED_EXPERT_REPRODUCES: usize = 6;

fn uninit<T: candle_core::cuda_backend::CudaDType + cudarc::driver::DeviceRepr>(device: &Device, shape: &[usize]) -> Result<Tensor> {
    let dev = cuda_dev(device)?;
    let stream = dev.cuda_stream();
    let n: usize = shape.iter().product();
    let slice: CudaSlice<T> = unsafe { stream.alloc::<T>(n.max(1))? };
    let storage = candle_core::CudaStorage::wrap_cuda_slice(slice, dev);
    let t = Tensor::from_storage(Storage::Cuda(storage), (n.max(1),), candle_core::op::BackpropOp::none(), false);
    Ok(if n == 0 { t.narrow(0, 0, 0)?.reshape(shape)? } else { t.reshape(shape)? })
}

fn eager_glue() -> bool {
    std::env::var(NV_Q4EXP_EAGER_GLUE_1_ROUTES_HC_AND_ROW_ATTENTION_THROUGH_CANDLE_OPS_FOR_KERNEL_PARITY_CHECKS)
        .as_deref()
        == Ok("1")
}

#[derive(Clone, Debug)]
pub struct Qwen4ExpConfig {
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub rotary_dim: usize,
    pub rope_theta: f32,
    pub rms_norm_eps: f64,
    pub vocab_size: usize,
    pub max_position_embeddings: usize,
    pub full_attention: Vec<bool>,
    pub linear_num_key_heads: usize,
    pub linear_num_value_heads: usize,
    pub linear_key_head_dim: usize,
    pub linear_value_head_dim: usize,
    pub linear_conv_kernel_dim: usize,
    pub num_experts: usize,
    pub num_experts_per_tok: usize,
    pub moe_intermediate_size: usize,
    pub shared_expert_intermediate_size: usize,
    pub hc_count: usize,
    pub hc_lowrank: usize,
    pub ple_layer_ids_zero_based: Vec<usize>,
    pub ple_embed_dim: usize,
    pub ple_conv_kernel_size: usize,
    pub ngram_size: usize,
    pub heads_per_ngram: usize,
    pub ngram_vocab_size_base: u64,
    pub make_ngram_vocab_size_divisible_by: usize,
    pub seed: u64,
    pub split_ngram_parts: usize,
    pub indexer_n_heads: usize,
    pub indexer_kv_heads: usize,
    pub indexer_head_dim: usize,
    pub indexer_budget: usize,
    pub indexer_compress_ratio: usize,
    pub eos_token_id: u32,
    pub tie_word_embeddings: bool,
    pub mtp_num_hidden_layers: usize,
}

fn json_u(v: &Value, k: &str) -> Result<usize> {
    v.get(k)
        .and_then(|x| x.as_u64())
        .map(|x| x as usize)
        .ok_or_else(|| anyhow::anyhow!("qwen4_exp config: missing/invalid {k}"))
}

fn json_f(v: &Value, k: &str) -> Result<f64> {
    v.get(k)
        .and_then(|x| x.as_f64())
        .ok_or_else(|| anyhow::anyhow!("qwen4_exp config: missing/invalid {k}"))
}

impl Qwen4ExpConfig {
    pub fn from_hf_json_str(s: &str) -> Result<Self> {
        let v: Value = serde_json::from_str(s).context("parse qwen4_exp config json")?;
        let t = v
            .get("text_config")
            .ok_or_else(|| anyhow::anyhow!("qwen4_exp config: missing text_config"))?;
        let full_attention: Vec<bool> = t
            .get("layer_types")
            .and_then(|x| x.as_array())
            .ok_or_else(|| anyhow::anyhow!("missing layer_types"))?
            .iter()
            .map(|x| match x.as_str() {
                Some("linear_attention") => Ok(false),
                Some("full_attention") | Some("indexed_attention") | Some("qwen_sparse_attention") => Ok(true),
                other => Err(anyhow::anyhow!("unknown layer type {other:?}")),
            })
            .collect::<Result<_>>()?;
        let rope = t.get("rope_parameters");
        let rope_theta = rope
            .and_then(|r| r.get("rope_theta"))
            .and_then(|x| x.as_f64())
            .or_else(|| t.get("rope_theta").and_then(|x| x.as_f64()))
            .unwrap_or(10_000_000.0) as f32;
        let partial = rope
            .and_then(|r| r.get("partial_rotary_factor"))
            .and_then(|x| x.as_f64())
            .or_else(|| t.get("partial_rotary_factor").and_then(|x| x.as_f64()))
            .unwrap_or(1.0);
        let head_dim = json_u(t, "head_dim")?;
        let hidden_size = json_u(t, "hidden_size")?;
        let ple_layer_ids_zero_based: Vec<usize> = t
            .get("ple_layer_ids")
            .and_then(|x| x.as_array())
            .map(|a| a.iter().filter_map(|x| x.as_u64()).map(|x| x as usize - 1).collect())
            .unwrap_or_default();
        let eos = match t.get("eos_token_id").or_else(|| v.get("eos_token_id")) {
            Some(Value::Number(n)) => n.as_u64().unwrap_or(0) as u32,
            Some(Value::Array(a)) => a.first().and_then(|x| x.as_u64()).unwrap_or(0) as u32,
            _ => bail!("qwen4_exp config: missing eos_token_id"),
        };
        let output_gate = t
            .get("output_gate_type")
            .and_then(|x| x.as_str())
            .unwrap_or("silu");
        ensure!(
            output_gate == "sigmoid",
            "qwen4_exp: output_gate_type {output_gate:?} is not the sigmoid this port wires"
        );
        let cfg = Self {
            hidden_size,
            num_hidden_layers: json_u(t, "num_hidden_layers")?,
            num_attention_heads: json_u(t, "num_attention_heads")?,
            num_key_value_heads: json_u(t, "num_key_value_heads")?,
            head_dim,
            rotary_dim: (head_dim as f64 * partial).round() as usize,
            rope_theta,
            rms_norm_eps: json_f(t, "rms_norm_eps")?,
            vocab_size: json_u(t, "vocab_size")?,
            max_position_embeddings: json_u(t, "max_position_embeddings")?,
            full_attention,
            linear_num_key_heads: json_u(t, "linear_num_key_heads")?,
            linear_num_value_heads: json_u(t, "linear_num_value_heads")?,
            linear_key_head_dim: json_u(t, "linear_key_head_dim")?,
            linear_value_head_dim: json_u(t, "linear_value_head_dim")?,
            linear_conv_kernel_dim: json_u(t, "linear_conv_kernel_dim")?,
            num_experts: json_u(t, "num_experts")?,
            num_experts_per_tok: json_u(t, "num_experts_per_tok")?,
            moe_intermediate_size: json_u(t, "moe_intermediate_size")?,
            shared_expert_intermediate_size: json_u(t, "shared_expert_intermediate_size")?,
            hc_count: json_u(t, "hc_count")?,
            hc_lowrank: json_u(t, "hc_lowrank")?,
            ple_layer_ids_zero_based,
            ple_embed_dim: json_u(t, "ple_embed_dim").unwrap_or(hidden_size),
            ple_conv_kernel_size: json_u(t, "ple_conv_kernel_size").unwrap_or(4),
            ngram_size: json_u(t, "ngram_size").unwrap_or(3),
            heads_per_ngram: json_u(t, "heads_per_ngram").unwrap_or(8),
            ngram_vocab_size_base: t
                .get("ngram_vocab_size_base")
                .and_then(|x| x.as_u64())
                .unwrap_or(20_000_000),
            make_ngram_vocab_size_divisible_by: json_u(t, "make_ngram_vocab_size_divisible_by")
                .unwrap_or(128),
            seed: t.get("seed").and_then(|x| x.as_u64()).unwrap_or(1234),
            split_ngram_parts: json_u(t, "split_ngram_parts").unwrap_or(512),
            indexer_n_heads: json_u(t, "indexer_n_heads")?,
            indexer_kv_heads: json_u(t, "indexer_kv_heads")?,
            indexer_head_dim: json_u(t, "indexer_head_dim")?,
            indexer_budget: json_u(t, "indexer_budget")?,
            indexer_compress_ratio: json_u(t, "indexer_compress_ratio")?,
            eos_token_id: eos,
            tie_word_embeddings: t
                .get("tie_word_embeddings")
                .and_then(|x| x.as_bool())
                .unwrap_or(false),
            mtp_num_hidden_layers: json_u(t, "mtp_num_hidden_layers").unwrap_or(0),
        };
        ensure!(
            cfg.full_attention.len() == cfg.num_hidden_layers,
            "layer_types has {} entries for {} layers",
            cfg.full_attention.len(),
            cfg.num_hidden_layers
        );
        ensure!(cfg.indexer_kv_heads == 1, "qwen4_exp QSA needs one indexer key head");
        ensure!(
            cfg.indexer_budget.is_multiple_of(cfg.indexer_compress_ratio),
            "indexer_budget must be a multiple of indexer_compress_ratio"
        );
        ensure!(cfg.ngram_size == 3, "qwen4_exp PLE port assumes ngram_size 3");
        Ok(cfg)
    }

    nv_weights::hf_json_from_file!(from_hf_json_file, from_hf_json_str);

    pub fn hc_hidden(&self) -> usize {
        self.hc_count * self.hidden_size
    }

    pub fn dense_attention_context_limit(&self) -> usize {
        self.indexer_budget + self.indexer_compress_ratio - 1
    }

    pub fn qsa_block_topk(&self) -> usize {
        self.indexer_budget / self.indexer_compress_ratio
    }

    pub fn ngram_heads(&self) -> usize {
        (self.ngram_size - 1) * self.heads_per_ngram
    }

    pub fn prefix_ckpt_bytes(&self) -> usize {
        let n_gdn = self.full_attention.iter().filter(|a| !**a).count();
        let recurrent = self.linear_num_value_heads * self.linear_key_head_dim * self.linear_value_head_dim * 4;
        let conv = self.linear_attn_config().conv_dim() * (self.linear_conv_kernel_dim - 1) * 2;
        n_gdn * (recurrent + conv) + (self.ple_conv_kernel_size - 1) * self.ngram_size * self.hc_hidden() * 4
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

pub fn vram_used_gib() -> f64 {
    match cudarc::driver::result::mem_get_info() {
        Ok((free, total)) => (total - free) as f64 / (1u64 << 30) as f64,
        Err(_) => f64::NAN,
    }
}

fn cuda_dev(device: &Device) -> Result<CudaDevice> {
    match device {
        Device::Cuda(d) => Ok(d.clone()),
        _ => bail!("qwen4_exp requires a CUDA device"),
    }
}

fn stream_of(device: &Device) -> Result<Arc<CudaStream>> {
    Ok(nv_layers::cuda_stream::current_stream(&cuda_dev(device)?))
}

fn cuda_ptr<T: candle_core::cuda_backend::CudaDType + cudarc::driver::DeviceRepr>(
    t: &Tensor,
    stream: &Arc<CudaStream>,
) -> Result<u64> {
    ensure!(t.is_contiguous(), "cuda_ptr needs a contiguous tensor");
    let (storage, layout) = t.storage_and_layout();
    let Storage::Cuda(s) = &*storage else {
        bail!("cuda_ptr: tensor is not on CUDA")
    };
    let slice = s.as_cuda_slice::<T>()?;
    let (p, _g) = slice.device_ptr(stream);
    Ok(p + (layout.start_offset() * std::mem::size_of::<T>()) as u64)
}

fn slice_ptr<T: cudarc::driver::DeviceRepr>(s: &CudaSlice<T>, stream: &Arc<CudaStream>) -> u64 {
    let (p, _g) = s.device_ptr(stream);
    p
}

fn slice_ptr_mut<T: cudarc::driver::DeviceRepr>(s: &mut CudaSlice<T>, stream: &Arc<CudaStream>) -> u64 {
    let (p, _g) = s.device_ptr_mut(stream);
    p
}

fn load_bf16(weights: &WeightLoader, name: &str) -> Result<Tensor> {
    weights
        .get(name, DType::BF16)
        .with_context(|| format!("load {name}"))
}

fn load_linear_bf16(weights: &WeightLoader, name: &str, out: usize, inp: usize) -> Result<Linear> {
    let w = load_bf16(weights, name)?;
    ensure!(
        w.dims() == [out, inp],
        "{name}: expected [{out}, {inp}], got {:?}",
        w.dims()
    );
    Linear::new_no_pretranspose(w.contiguous()?, None)
}

type Fp8Runner = Arc<Mutex<nv_quant::fp8::Fp8GemmRunner>>;

fn load_linear_side(
    weights: &WeightLoader,
    name: &str,
    out: usize,
    inp: usize,
    fp8: Option<&Fp8Runner>,
    device: &Device,
) -> Result<Linear> {
    let w = load_bf16(weights, name)?;
    ensure!(
        w.dims() == [out, inp],
        "{name}: expected [{out}, {inp}], got {:?}",
        w.dims()
    );
    match fp8 {
        Some(runner) if inp.is_multiple_of(16) && out > 1 => {
            let host: Vec<bf16> = w.flatten_all()?.to_vec1()?;
            let (bytes, rows) = nv_layers::linear::fp8_weight_payload(
                &host,
                out,
                inp,
                None,
                nv_quant::fp8::Fp8ScaleMode::PerOuterRow,
            )?;
            let stream = stream_of(device)?;
            let weight_u8 = stream.clone_htod(&bytes)?;
            Linear::new_fp8_e4m3_row_scales_without_the_cublaslt_probe(
                weight_u8,
                rows,
                inp,
                out,
                None,
                device,
                runner.clone(),
                nv_quant::fp8::Fp8ScaleMode::PerOuterRow,
            )
        }
        _ => Linear::new_no_pretranspose(w.contiguous()?, None),
    }
}

fn load_norm_plus_one_f32(weights: &WeightLoader, name: &str, dim: usize) -> Result<Tensor> {
    let w = weights
        .get(name, DType::F32)
        .with_context(|| format!("load {name}"))?;
    ensure!(w.dims() == [dim], "{name}: expected [{dim}], got {:?}", w.dims());
    Ok(w.affine(1.0, 1.0)?.contiguous()?)
}

fn group_rmsnorm(x: &Tensor, w_plus_one_f32: &Tensor, groups: usize, eps: f64) -> Result<Tensor> {
    let (t, width) = x.dims2()?;
    let g = width / groups;
    let xf = x.to_dtype(DType::F32)?.reshape((t, groups, g))?;
    let var = xf.sqr()?.mean_keepdim(D::Minus1)?;
    let inv = (var + eps)?.sqrt()?.recip()?;
    let normed = xf.broadcast_mul(&inv)?.reshape((t, width))?;
    Ok(normed.broadcast_mul(w_plus_one_f32)?.to_dtype(x.dtype())?)
}

fn rmsnorm_rows(x: &Tensor, w_plus_one_f32: &Tensor, eps: f64) -> Result<Tensor> {
    let (rows, n) = x.dims2()?;
    let x = x.contiguous()?;
    let out = Tensor::zeros((rows, n), DType::BF16, x.device())?;
    let stream = stream_of(x.device())?;
    let rc = unsafe {
        nvk::q4exp_rmsnorm_rows(
            stream.cu_stream() as *mut c_void,
            cuda_ptr::<bf16>(&x, &stream)? as *const u16,
            cuda_ptr::<f32>(w_plus_one_f32, &stream)? as *const f32,
            cuda_ptr::<bf16>(&out, &stream)? as *mut u16,
            rows as i32,
            n as i32,
            eps as f32,
        )
    };
    ensure!(rc == 0, "q4exp_rmsnorm_rows rc={rc}");
    Ok(out)
}

fn dtod_raw(dst: u64, src: u64, bytes: usize, stream: &Arc<CudaStream>) -> Result<()> {
    unsafe { cudarc::driver::result::memcpy_dtod_async(dst, src, bytes, stream.cu_stream()) }
        .map_err(|e| anyhow::anyhow!("dtod {bytes} bytes: {e:?}"))
}

fn head_rmsnorm_bf16(x: &Tensor, w_plus_one_f32: &Tensor, eps: f64) -> Result<Tensor> {
    let xf = x.to_dtype(DType::F32)?;
    let var = xf.sqr()?.mean_keepdim(D::Minus1)?;
    let inv = (var + eps)?.sqrt()?.recip()?;
    Ok(xf.broadcast_mul(&inv)?.broadcast_mul(w_plus_one_f32)?.to_dtype(DType::BF16)?)
}

pub struct HcGate {
    norm_w: Tensor,
    down: Linear,
    up: Linear,
    inject: Option<Linear>,
}

impl HcGate {
    pub fn from_parts(norm_w_plus_one_f32: Tensor, down: Linear, up: Linear, inject: Option<Linear>) -> Self {
        Self {
            norm_w: norm_w_plus_one_f32,
            down,
            up,
            inject,
        }
    }

    fn load(weights: &WeightLoader, prefix: &str, cfg: &Qwen4ExpConfig, with_inject: bool) -> Result<Self> {
        let hh = cfg.hc_hidden();
        Ok(Self {
            norm_w: load_norm_plus_one_f32(weights, &format!("{prefix}.hc_norm.weight"), hh)?,
            down: load_linear_bf16(weights, &format!("{prefix}.input_mix_weight_down.weight"), cfg.hc_lowrank, hh)?,
            up: load_linear_bf16(weights, &format!("{prefix}.input_mix_weight_up.weight"), hh, cfg.hc_lowrank)?,
            inject: if with_inject {
                Some(load_linear_bf16(weights, &format!("{prefix}.block_inject_weight.weight"), cfg.hc_count, hh)?)
            } else {
                None
            },
        })
    }

    pub fn mix(&self, hc: &Tensor, cfg: &Qwen4ExpConfig) -> Result<(Tensor, Option<Tensor>)> {
        let (t, _) = hc.dims2()?;
        if eager_glue() {
            return self.mix_eager(hc, cfg);
        }
        if t <= HC_FUSED_ROWS_MAX_64_BELOW_WHICH_THE_DOT_GRID_BEATS_THREE_CUBLAS_CALLS_PER_MIX {
            return self.mix_fused(hc, cfg);
        }
        self.mix_prefill(hc, cfg)
    }

    pub fn mix_prefill(&self, hc: &Tensor, cfg: &Qwen4ExpConfig) -> Result<(Tensor, Option<Tensor>)> {
        let (t, hh) = hc.dims2()?;
        let n = cfg.hc_count;
        let device = hc.device();
        let stream = stream_of(device)?;
        let hc = hc.contiguous()?;
        let xn = uninit::<bf16>(device, &[t, hh])?;
        let rc = unsafe {
            nvk::q4exp_hc_norm_rows(
                stream.cu_stream() as *mut c_void,
                cuda_ptr::<bf16>(&hc, &stream)? as *const u16,
                cuda_ptr::<f32>(&self.norm_w, &stream)? as *const f32,
                cuda_ptr::<bf16>(&xn, &stream)? as *mut u16,
                t as i32,
                hh as i32,
                n as i32,
                cfg.rms_norm_eps as f32,
            )
        };
        ensure!(rc == 0, "q4exp_hc_norm_rows rc={rc}");
        self.mix_prefill_after_norm(&xn, cfg)
    }

    fn mix_prefill_after_norm(&self, xn: &Tensor, cfg: &Qwen4ExpConfig) -> Result<(Tensor, Option<Tensor>)> {
        let (t, _) = xn.dims2()?;
        let n = cfg.hc_count;
        let h = cfg.hidden_size;
        let rank = cfg.hc_lowrank;
        let device = xn.device();
        let stream = stream_of(device)?;
        let xn = xn.clone();
        let d = self.down.forward(&xn)?.contiguous()?;
        let inj = match &self.inject {
            Some(lin) => Some(lin.forward(&xn)?.contiguous()?),
            None => None,
        };
        let lora = uninit::<bf16>(device, &[t, rank])?;
        let inject = uninit::<f32>(device, &[t, n])?;
        let rc = unsafe {
            nvk::q4exp_hc_lora_inject(
                stream.cu_stream() as *mut c_void,
                cuda_ptr::<bf16>(&d, &stream)? as *const u16,
                match &inj {
                    Some(x) => cuda_ptr::<bf16>(x, &stream)? as *const u16,
                    None => std::ptr::null(),
                },
                cuda_ptr::<bf16>(&lora, &stream)? as *mut u16,
                cuda_ptr::<f32>(&inject, &stream)? as *mut f32,
                t as i32,
                rank as i32,
                n as i32,
            )
        };
        ensure!(rc == 0, "q4exp_hc_lora_inject rc={rc}");
        let g = self.up.forward(&lora)?.contiguous()?;
        let mixed = uninit::<bf16>(device, &[t, h])?;
        let rc = unsafe {
            nvk::q4exp_hc_gate_mix(
                stream.cu_stream() as *mut c_void,
                cuda_ptr::<bf16>(&xn, &stream)? as *const u16,
                cuda_ptr::<bf16>(&g, &stream)? as *const u16,
                cuda_ptr::<bf16>(&mixed, &stream)? as *mut u16,
                t as i32,
                h as i32,
                n as i32,
            )
        };
        ensure!(rc == 0, "q4exp_hc_gate_mix rc={rc}");
        Ok((mixed, if inj.is_some() { Some(inject) } else { None }))
    }

    pub fn mix_eager(&self, hc: &Tensor, cfg: &Qwen4ExpConfig) -> Result<(Tensor, Option<Tensor>)> {
        let (t, _) = hc.dims2()?;
        let n = cfg.hc_count;
        let h = cfg.hidden_size;
        let xn = group_rmsnorm(hc, &self.norm_w, n, cfg.rms_norm_eps)?;
        let lora = self.down.forward(&xn)?.to_dtype(DType::F32)?.affine(1.0 / n as f64, 0.0)?;
        let lora = candle_nn::ops::silu(&lora)?.to_dtype(DType::BF16)?;
        let gate = candle_nn::ops::sigmoid(&self.up.forward(&lora)?.to_dtype(DType::F32)?)?;
        let xn_f = xn.to_dtype(DType::F32)?.reshape((t, n, h))?;
        let mixed = gate
            .reshape((t, n, h))?
            .mul(&xn_f)?
            .mean(1)?
            .to_dtype(DType::BF16)?;
        let inject = match &self.inject {
            Some(lin) => {
                let logits = lin.forward(&xn)?.to_dtype(DType::F32)?.affine(1.0 / n as f64, 0.0)?;
                Some(candle_nn::ops::sigmoid(&logits)?.affine(2.0, 0.0)?)
            }
            None => None,
        };
        Ok((mixed, inject))
    }

    pub fn mix_combined(
        &self,
        hc: &Tensor,
        pending: Option<(Tensor, Tensor)>,
        cfg: &Qwen4ExpConfig,
    ) -> Result<(Tensor, Tensor, Option<Tensor>)> {
        let (t, hh) = hc.dims2()?;
        let n = cfg.hc_count;
        let h = cfg.hidden_size;
        let rank = cfg.hc_lowrank;
        let device = hc.device();
        let stream = stream_of(device)?;
        let hc = hc.contiguous()?;
        let xn = uninit::<bf16>(device, &[t, hh])?;
        let (hc_new, y, inj) = match pending {
            Some((y, inj)) => (uninit::<bf16>(device, &[t, hh])?, Some(y.contiguous()?), Some(inj.contiguous()?)),
            None => (hc.clone(), None, None),
        };
        let rc = unsafe {
            nvk::q4exp_hc_combine_norm(
                stream.cu_stream() as *mut c_void,
                cuda_ptr::<bf16>(&hc, &stream)? as *const u16,
                match &y {
                    Some(y) => cuda_ptr::<bf16>(y, &stream)? as *const u16,
                    None => std::ptr::null(),
                },
                match &inj {
                    Some(i) => cuda_ptr::<f32>(i, &stream)? as *const f32,
                    None => std::ptr::null(),
                },
                cuda_ptr::<f32>(&self.norm_w, &stream)? as *const f32,
                if y.is_some() { cuda_ptr::<bf16>(&hc_new, &stream)? as *mut u16 } else { std::ptr::null_mut() },
                cuda_ptr::<bf16>(&xn, &stream)? as *mut u16,
                t as i32,
                hh as i32,
                h as i32,
                n as i32,
                cfg.rms_norm_eps as f32,
            )
        };
        ensure!(rc == 0, "q4exp_hc_combine_norm rc={rc}");
        if t > HC_FUSED_ROWS_MAX_64_BELOW_WHICH_THE_DOT_GRID_BEATS_THREE_CUBLAS_CALLS_PER_MIX {
            let (mixed, inject) = self.mix_prefill_after_norm(&xn, cfg)?;
            return Ok((hc_new, mixed, inject));
        }
        let lora = uninit::<f32>(device, &[t, rank])?;
        let inject = uninit::<f32>(device, &[t, n])?;
        let mixed = uninit::<bf16>(device, &[t, h])?;
        let w_down = self.down.weight().context("hc down weight not bf16-resident")?;
        let w_up = self.up.weight().context("hc up weight not bf16-resident")?;
        let w_inject = match &self.inject {
            Some(l) => cuda_ptr::<bf16>(l.weight().context("hc inject weight not bf16-resident")?, &stream)?,
            None => 0,
        };
        let rc = unsafe {
            nvk::q4exp_hc_down(
                stream.cu_stream() as *mut c_void,
                cuda_ptr::<bf16>(&xn, &stream)? as *const u16,
                cuda_ptr::<bf16>(w_down, &stream)? as *const u16,
                w_inject as *const u16,
                cuda_ptr::<f32>(&lora, &stream)? as *mut f32,
                cuda_ptr::<f32>(&inject, &stream)? as *mut f32,
                t as i32,
                hh as i32,
                n as i32,
                rank as i32,
            )
        };
        ensure!(rc == 0, "q4exp_hc_down rc={rc}");
        let rc = unsafe {
            nvk::q4exp_hc_up_mix(
                stream.cu_stream() as *mut c_void,
                cuda_ptr::<f32>(&lora, &stream)? as *const f32,
                cuda_ptr::<bf16>(w_up, &stream)? as *const u16,
                cuda_ptr::<bf16>(&xn, &stream)? as *const u16,
                cuda_ptr::<bf16>(&mixed, &stream)? as *mut u16,
                t as i32,
                h as i32,
                n as i32,
                rank as i32,
            )
        };
        ensure!(rc == 0, "q4exp_hc_up_mix rc={rc}");
        Ok((hc_new, mixed, self.inject.as_ref().map(|_| inject)))
    }

    pub fn mix_fused(&self, hc: &Tensor, cfg: &Qwen4ExpConfig) -> Result<(Tensor, Option<Tensor>)> {
        let (t, hh) = hc.dims2()?;
        let n = cfg.hc_count;
        let h = cfg.hidden_size;
        let rank = cfg.hc_lowrank;
        let dev = cuda_dev(hc.device())?;
        let stream = stream_of(hc.device())?;
        let hc = hc.contiguous()?;
        let xn = Tensor::zeros((t, hh), DType::BF16, hc.device())?;
        let lora = Tensor::zeros((t, rank), DType::F32, hc.device())?;
        let inject = Tensor::zeros((t, n), DType::F32, hc.device())?;
        let mixed = Tensor::zeros((t, h), DType::BF16, hc.device())?;
        let w_down = self.down.weight().context("hc down weight not bf16-resident")?;
        let w_up = self.up.weight().context("hc up weight not bf16-resident")?;
        let w_inject = match &self.inject {
            Some(l) => cuda_ptr::<bf16>(l.weight().context("hc inject weight not bf16-resident")?, &stream)?,
            None => 0,
        };
        let rc = unsafe {
            nvk::q4exp_hc_norm_down(
                stream.cu_stream() as *mut c_void,
                cuda_ptr::<bf16>(&hc, &stream)? as *const u16,
                cuda_ptr::<f32>(&self.norm_w, &stream)? as *const f32,
                cuda_ptr::<bf16>(w_down, &stream)? as *const u16,
                w_inject as *const u16,
                cuda_ptr::<bf16>(&xn, &stream)? as *mut u16,
                cuda_ptr::<f32>(&lora, &stream)? as *mut f32,
                cuda_ptr::<f32>(&inject, &stream)? as *mut f32,
                t as i32,
                hh as i32,
                n as i32,
                rank as i32,
                cfg.rms_norm_eps as f32,
            )
        };
        ensure!(rc == 0, "q4exp_hc_norm_down rc={rc}");
        let rc = unsafe {
            nvk::q4exp_hc_up_mix(
                stream.cu_stream() as *mut c_void,
                cuda_ptr::<f32>(&lora, &stream)? as *const f32,
                cuda_ptr::<bf16>(w_up, &stream)? as *const u16,
                cuda_ptr::<bf16>(&xn, &stream)? as *const u16,
                cuda_ptr::<bf16>(&mixed, &stream)? as *mut u16,
                t as i32,
                h as i32,
                n as i32,
                rank as i32,
            )
        };
        ensure!(rc == 0, "q4exp_hc_up_mix rc={rc}");
        let _ = dev;
        Ok((mixed, self.inject.as_ref().map(|_| inject)))
    }
}

pub fn hc_combine(hc: &Tensor, y: &Tensor, inject: &Tensor, cfg: &Qwen4ExpConfig) -> Result<Tensor> {
    if eager_glue() {
        hc_combine_eager(hc, y, inject, cfg)
    } else {
        hc_combine_kernel(hc, y, inject, cfg)
    }
}

pub fn hc_combine_kernel(hc: &Tensor, y: &Tensor, inject: &Tensor, cfg: &Qwen4ExpConfig) -> Result<Tensor> {
    let (t, _) = hc.dims2()?;
    let n = cfg.hc_count;
    let h = cfg.hidden_size;
    {
        let stream = stream_of(hc.device())?;
        let out = hc.contiguous()?.copy()?;
        let y = y.contiguous()?;
        let inject = inject.contiguous()?;
        let rc = unsafe {
            nvk::q4exp_hc_combine(
                stream.cu_stream() as *mut c_void,
                cuda_ptr::<bf16>(&out, &stream)? as *mut u16,
                cuda_ptr::<bf16>(&y, &stream)? as *const u16,
                cuda_ptr::<f32>(&inject, &stream)? as *const f32,
                t as i32,
                h as i32,
                n as i32,
            )
        };
        ensure!(rc == 0, "q4exp_hc_combine rc={rc}");
        Ok(out)
    }
}

pub fn hc_settle(hc: Tensor, pending: Option<(Tensor, Tensor)>, cfg: &Qwen4ExpConfig) -> Result<Tensor> {
    match pending {
        Some((y, inject)) => hc_combine(&hc, &y, &inject, cfg),
        None => Ok(hc),
    }
}

pub fn hc_combine_eager(hc: &Tensor, y: &Tensor, inject: &Tensor, cfg: &Qwen4ExpConfig) -> Result<Tensor> {
    let (t, hh) = hc.dims2()?;
    let n = cfg.hc_count;
    let h = cfg.hidden_size;
    let add = y
        .to_dtype(DType::F32)?
        .reshape((t, 1, h))?
        .broadcast_mul(&inject.reshape((t, n, 1))?)?;
    Ok((hc.to_dtype(DType::F32)?.reshape((t, n, h))? + add)?
        .reshape((t, hh))?
        .to_dtype(DType::BF16)?)
}

pub struct Attention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    q_norm_w: Tensor,
    k_norm_w: Tensor,
    idx_qk_proj: Linear,
    idx_q_norm_w: Tensor,
    idx_k_norm_w: Tensor,
}

impl Attention {
    fn load(weights: &WeightLoader, prefix: &str, cfg: &Qwen4ExpConfig, fp8: Option<&Fp8Runner>, device: &Device) -> Result<Self> {
        let h = cfg.hidden_size;
        let hd = cfg.head_dim;
        let nq = cfg.num_attention_heads;
        let nkv = cfg.num_key_value_heads;
        let idx_out = (cfg.indexer_n_heads + cfg.indexer_kv_heads) * cfg.indexer_head_dim;
        Ok(Self {
            q_proj: load_linear_side(weights, &format!("{prefix}.q_proj.weight"), nq * hd * 2, h, fp8, device)?,
            k_proj: load_linear_side(weights, &format!("{prefix}.k_proj.weight"), nkv * hd, h, fp8, device)?,
            v_proj: load_linear_side(weights, &format!("{prefix}.v_proj.weight"), nkv * hd, h, fp8, device)?,
            o_proj: load_linear_side(weights, &format!("{prefix}.o_proj.weight"), h, nq * hd, fp8, device)?,
            q_norm_w: load_norm_plus_one_f32(weights, &format!("{prefix}.q_norm.weight"), hd)?,
            k_norm_w: load_norm_plus_one_f32(weights, &format!("{prefix}.k_norm.weight"), hd)?,
            idx_qk_proj: load_linear_side(weights, &format!("{prefix}.indexer.index_qk_proj.weight"), idx_out, h, fp8, device)?,
            idx_q_norm_w: load_norm_plus_one_f32(weights, &format!("{prefix}.indexer.q_layernorm.weight"), cfg.indexer_head_dim)?,
            idx_k_norm_w: load_norm_plus_one_f32(weights, &format!("{prefix}.indexer.k_layernorm.weight"), cfg.indexer_head_dim)?,
        })
    }
}

pub struct MoeLayer {
    router: Linear,
    grouped: Arc<MoeGroupedWeights>,
    shared_gate_proj: Linear,
    shared_up_proj: Linear,
    shared_down_proj: Linear,
    shared_expert_gate: Linear,
    expert_lora: std::sync::RwLock<Option<Arc<lora::expert::ExpertLoraLayer>>>,
}

fn f32_scalar(weights: &WeightLoader, name: &str) -> Result<f32> {
    let t = weights
        .get(name, DType::F32)
        .with_context(|| format!("load {name}"))?;
    let v = t.flatten_all()?.to_vec1::<f32>()?;
    Ok(*v.first().ok_or_else(|| anyhow::anyhow!("{name}: empty scalar"))?)
}

fn load_grouped_experts_from_disk(
    weights: &WeightLoader,
    prefix: &str,
    cfg: &Qwen4ExpConfig,
    runner: Arc<Mutex<Nvfp4GemmRunner>>,
    stream: &Arc<CudaStream>,
) -> Result<MoeGroupedWeights> {
    let e = cfg.num_experts;
    let h = cfg.hidden_size;
    let inter = cfg.moe_intermediate_size;
    let gu_bytes = inter * h / 2;
    let gu_scale_bytes = swizzled_scale_bytes(inter, h);
    let dn_bytes = h * inter / 2;
    let dn_scale_bytes = swizzled_scale_bytes(h, inter);
    let mut gate_w = vec![0u8; e * gu_bytes];
    let mut up_w = vec![0u8; e * gu_bytes];
    let mut down_w = vec![0u8; e * dn_bytes];
    let mut gate_s = vec![0u8; e * gu_scale_bytes];
    let mut up_s = vec![0u8; e * gu_scale_bytes];
    let mut down_s = vec![0u8; e * dn_scale_bytes];
    let mut gate_alpha = vec![0f32; e];
    let mut up_alpha = vec![0f32; e];
    let mut down_alpha = vec![0f32; e];
    let mut gu_global = vec![0f32; e];
    let mut dn_global = vec![0f32; e];
    let recip = |x: f32| if x == 0.0 || !x.is_finite() { 1.0 } else { 1.0 / x };
    let read = |i: usize, proj: &str, rows: usize, k: usize, w_out: &mut [u8], s_out: &mut [u8]| -> Result<(f32, f32)> {
        let base = format!("{prefix}.experts.{i}.{proj}");
        let packed = weights.raw_bytes(&format!("{base}.weight"))?;
        ensure!(
            packed.len() == rows * k / 2,
            "{base}.weight: {} bytes, expected {}",
            packed.len(),
            rows * k / 2
        );
        w_out.copy_from_slice(packed);
        let scales = weights.raw_bytes(&format!("{base}.weight_scale"))?;
        ensure!(
            scales.len() == rows * k / 16,
            "{base}.weight_scale: {} bytes, expected {}",
            scales.len(),
            rows * k / 16
        );
        let sw = nv_quant::nvfp4::swizzle_scales(scales, rows, k / 16);
        ensure!(sw.len() == s_out.len(), "{base}: swizzled scale bytes {} != {}", sw.len(), s_out.len());
        s_out.copy_from_slice(&sw);
        let ws2 = f32_scalar(weights, &format!("{base}.weight_scale_2"))?;
        let is = f32_scalar(weights, &format!("{base}.input_scale"))?;
        Ok((ws2 * is, recip(is)))
    };
    let threads = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(8).min(32).max(1);
    let per = e.div_ceil(threads);
    std::thread::scope(|sc| -> Result<()> {
        let handles: Vec<_> = gate_w
            .chunks_mut(per * gu_bytes)
            .zip(up_w.chunks_mut(per * gu_bytes))
            .zip(down_w.chunks_mut(per * dn_bytes))
            .zip(gate_s.chunks_mut(per * gu_scale_bytes))
            .zip(up_s.chunks_mut(per * gu_scale_bytes))
            .zip(down_s.chunks_mut(per * dn_scale_bytes))
            .zip(gate_alpha.chunks_mut(per))
            .zip(up_alpha.chunks_mut(per))
            .zip(down_alpha.chunks_mut(per))
            .zip(gu_global.chunks_mut(per))
            .zip(dn_global.chunks_mut(per))
            .enumerate()
            .map(|(ti, ((((((((((gw, uw), dw), gs), us), ds), ga), ua), da), gg), dg))| {
                let read = &read;
                sc.spawn(move || -> Result<()> {
                    for j in 0..ga.len() {
                        let i = ti * per + j;
                        let (a_g, g_g) = read(i, "gate_proj", inter, h, &mut gw[j * gu_bytes..(j + 1) * gu_bytes], &mut gs[j * gu_scale_bytes..(j + 1) * gu_scale_bytes])?;
                        let (a_u, g_u) = read(i, "up_proj", inter, h, &mut uw[j * gu_bytes..(j + 1) * gu_bytes], &mut us[j * gu_scale_bytes..(j + 1) * gu_scale_bytes])?;
                        let (a_d, g_d) = read(i, "down_proj", h, inter, &mut dw[j * dn_bytes..(j + 1) * dn_bytes], &mut ds[j * dn_scale_bytes..(j + 1) * dn_scale_bytes])?;
                        ensure!(
                            (g_g - g_u).abs() <= 1e-6 * g_g.abs().max(1.0),
                            "{prefix}.experts.{i}: gate/up input_scale differ ({g_g} vs {g_u})"
                        );
                        ga[j] = a_g;
                        ua[j] = a_u;
                        da[j] = a_d;
                        gg[j] = g_g;
                        dg[j] = g_d;
                    }
                    Ok(())
                })
            })
            .collect();
        for hd in handles {
            hd.join().map_err(|_| anyhow::anyhow!("{prefix}: expert load thread panicked"))??;
        }
        Ok(())
    })?;
    assemble_grouped(e, h, inter, GroupedParts { gate_w, gate_s, gate_alpha, up_w, up_s, up_alpha, down_w, down_s, down_alpha, gu_global, dn_global }, runner, stream)
}

struct GroupedParts {
    gate_w: Vec<u8>,
    gate_s: Vec<u8>,
    gate_alpha: Vec<f32>,
    up_w: Vec<u8>,
    up_s: Vec<u8>,
    up_alpha: Vec<f32>,
    down_w: Vec<u8>,
    down_s: Vec<u8>,
    down_alpha: Vec<f32>,
    gu_global: Vec<f32>,
    dn_global: Vec<f32>,
}

fn assemble_grouped(
    e: usize,
    h: usize,
    inter: usize,
    p: GroupedParts,
    runner: Arc<Mutex<Nvfp4GemmRunner>>,
    stream: &Arc<CudaStream>,
) -> Result<MoeGroupedWeights> {
    let GroupedParts { gate_w, gate_s, gate_alpha, up_w, up_s, up_alpha, down_w, down_s, down_alpha, gu_global, dn_global } = p;
    let built = MoeGroupedWeights {
        num_experts: e,
        hidden_size: h,
        intermediate_size: inter,
        folded_shared: false,
        gate_w: stream.clone_htod(&gate_w)?,
        gate_w_scales: stream.clone_htod(&gate_s)?,
        gate_alphas: stream.clone_htod(&gate_alpha)?,
        gate_a_stride_elems: h as i64,
        gate_b_stride_elems: h as i64,
        gate_c_stride_elems: inter as i64,
        up_w: stream.clone_htod(&up_w)?,
        up_w_scales: stream.clone_htod(&up_s)?,
        up_alphas: stream.clone_htod(&up_alpha)?,
        down_w: stream.clone_htod(&down_w)?,
        down_w_scales: stream.clone_htod(&down_s)?,
        down_alphas: stream.clone_htod(&down_alpha)?,
        down_a_stride_elems: inter as i64,
        down_b_stride_elems: inter as i64,
        down_c_stride_elems: h as i64,
        runner,
        input_globals_gate_up: stream.clone_htod(&gu_global)?,
        input_globals_down: stream.clone_htod(&dn_global)?,
        input_globals_gate_up_host: gu_global,
        input_globals_down_host: dn_global,
        marlin_moe_twin_built_once_from_the_resident_grouped_buffers: std::sync::OnceLock::new(),
    };
    stream.synchronize()?;
    Ok(built)
}

fn bf16_at(bytes: &[u8], i: usize) -> f32 {
    bf16::from_bits(u16::from_le_bytes([bytes[2 * i], bytes[2 * i + 1]])).to_f32()
}

struct RequantProj {
    data: Vec<u8>,
    scales_swizzled: Vec<u8>,
    weight_scale_2: f32,
}

fn requantize_fp8_block_rows_to_nvfp4(weights: &WeightLoader, base: &str, rows: usize, cols: usize) -> Result<RequantProj> {
    let w = weights.raw_bytes(&format!("{base}.weight"))?;
    let sc = weights.raw_bytes(&format!("{base}.weight_scale_inv"))?;
    let cb = cols / 128;
    ensure!(w.len() == rows * cols, "{base}.weight: {} bytes, expected {}", w.len(), rows * cols);
    ensure!(sc.len() == 2 * (rows / 128) * cb, "{base}.weight_scale_inv: {} bytes, expected {}", sc.len(), 2 * (rows / 128) * cb);
    let mut amax = 0f32;
    let mut out: Vec<Vec<f32>> = Vec::with_capacity(rows);
    let mut row_scales = vec![0f32; cb];
    for r in 0..rows {
        let sr = (r / 128) * cb;
        for (j, slot) in row_scales.iter_mut().enumerate() {
            *slot = bf16_at(sc, sr + j);
        }
        let mut row = Vec::with_capacity(cols);
        for c in 0..cols {
            let v = fp8_e4m3_to_f32(w[r * cols + c]) * row_scales[c / 128];
            amax = amax.max(v.abs());
            row.push(v);
        }
        out.push(row);
    }
    let weight_scale_2 = if amax > 0.0 { amax / (6.0 * 448.0) } else { 1.0 };
    let t = nv_quant::nvfp4::Nvfp4Tensor::quantize_rows_with_global(&out, 1.0 / weight_scale_2);
    let scales_swizzled = t.scales_swizzled();
    Ok(RequantProj { data: t.data, scales_swizzled, weight_scale_2 })
}

fn load_grouped_experts_requantized_from_fp8_block_scales(
    weights: &WeightLoader,
    prefix: &str,
    cfg: &Qwen4ExpConfig,
    runner: Arc<Mutex<Nvfp4GemmRunner>>,
    stream: &Arc<CudaStream>,
    input_scale_gate_up: f32,
    input_scale_down: f32,
) -> Result<MoeGroupedWeights> {
    let e = cfg.num_experts;
    let h = cfg.hidden_size;
    let inter = cfg.moe_intermediate_size;
    let gu_bytes = inter * h / 2;
    let gu_scale_bytes = swizzled_scale_bytes(inter, h);
    let dn_bytes = h * inter / 2;
    let dn_scale_bytes = swizzled_scale_bytes(h, inter);
    let threads = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(8).min(32).max(1);
    let per = e.div_ceil(threads);
    let parts: Vec<Result<Vec<(usize, RequantProj, RequantProj, RequantProj)>>> = std::thread::scope(|sc| {
        let handles: Vec<_> = (0..threads)
            .map(|ti| {
                let lo = ti * per;
                let hi = ((ti + 1) * per).min(e);
                sc.spawn(move || {
                    let mut v = Vec::new();
                    for i in lo..hi {
                        let base = format!("{prefix}.experts.{i}");
                        let g = requantize_fp8_block_rows_to_nvfp4(weights, &format!("{base}.gate_proj"), inter, h)?;
                        let u = requantize_fp8_block_rows_to_nvfp4(weights, &format!("{base}.up_proj"), inter, h)?;
                        let d = requantize_fp8_block_rows_to_nvfp4(weights, &format!("{base}.down_proj"), h, inter)?;
                        v.push((i, g, u, d));
                    }
                    Ok(v)
                })
            })
            .collect();
        handles.into_iter().map(|hd| hd.join().expect("requant thread")).collect()
    });
    let mut p = GroupedParts {
        gate_w: vec![0u8; e * gu_bytes],
        gate_s: vec![0u8; e * gu_scale_bytes],
        gate_alpha: vec![0f32; e],
        up_w: vec![0u8; e * gu_bytes],
        up_s: vec![0u8; e * gu_scale_bytes],
        up_alpha: vec![0f32; e],
        down_w: vec![0u8; e * dn_bytes],
        down_s: vec![0u8; e * dn_scale_bytes],
        down_alpha: vec![0f32; e],
        gu_global: vec![1.0 / input_scale_gate_up; e],
        dn_global: vec![1.0 / input_scale_down; e],
    };
    for part in parts {
        for (i, g, u, d) in part? {
            ensure!(g.data.len() == gu_bytes && g.scales_swizzled.len() == gu_scale_bytes, "mtp expert {i} gate requant size");
            ensure!(d.data.len() == dn_bytes && d.scales_swizzled.len() == dn_scale_bytes, "mtp expert {i} down requant size");
            p.gate_w[i * gu_bytes..(i + 1) * gu_bytes].copy_from_slice(&g.data);
            p.gate_s[i * gu_scale_bytes..(i + 1) * gu_scale_bytes].copy_from_slice(&g.scales_swizzled);
            p.up_w[i * gu_bytes..(i + 1) * gu_bytes].copy_from_slice(&u.data);
            p.up_s[i * gu_scale_bytes..(i + 1) * gu_scale_bytes].copy_from_slice(&u.scales_swizzled);
            p.down_w[i * dn_bytes..(i + 1) * dn_bytes].copy_from_slice(&d.data);
            p.down_s[i * dn_scale_bytes..(i + 1) * dn_scale_bytes].copy_from_slice(&d.scales_swizzled);
            p.gate_alpha[i] = g.weight_scale_2 * input_scale_gate_up;
            p.up_alpha[i] = u.weight_scale_2 * input_scale_gate_up;
            p.down_alpha[i] = d.weight_scale_2 * input_scale_down;
        }
    }
    assemble_grouped(e, h, inter, p, runner, stream)
}

impl MoeLayer {
    fn load(
        weights: &WeightLoader,
        prefix: &str,
        cfg: &Qwen4ExpConfig,
        runner: Arc<Mutex<Nvfp4GemmRunner>>,
        stream: &Arc<CudaStream>,
        fp8: Option<&Fp8Runner>,
        device: &Device,
    ) -> Result<Self> {
        let grouped = load_grouped_experts_from_disk(weights, prefix, cfg, runner, stream)?;
        Self::load_with_grouped(weights, prefix, cfg, grouped, fp8, device)
    }

    fn load_with_grouped(
        weights: &WeightLoader,
        prefix: &str,
        cfg: &Qwen4ExpConfig,
        grouped: MoeGroupedWeights,
        fp8: Option<&Fp8Runner>,
        device: &Device,
    ) -> Result<Self> {
        let h = cfg.hidden_size;
        let si = cfg.shared_expert_intermediate_size;
        Ok(Self {
            router: load_linear_bf16(weights, &format!("{prefix}.gate.weight"), cfg.num_experts, h)?,
            grouped: Arc::new(grouped),
            shared_gate_proj: load_linear_side(weights, &format!("{prefix}.shared_expert.gate_proj.weight"), si, h, fp8, device)?,
            shared_up_proj: load_linear_side(weights, &format!("{prefix}.shared_expert.up_proj.weight"), si, h, fp8, device)?,
            shared_down_proj: load_linear_side(weights, &format!("{prefix}.shared_expert.down_proj.weight"), h, si, fp8, device)?,
            shared_expert_gate: load_linear_bf16(weights, &format!("{prefix}.shared_expert_gate.weight"), 1, h)?,
            expert_lora: std::sync::RwLock::new(None),
        })
    }

    fn shared(&self, x: &Tensor) -> Result<Tensor> {
        let g = self.shared_gate_proj.forward(x)?.to_dtype(DType::F32)?;
        let u = self.shared_up_proj.forward(x)?.to_dtype(DType::F32)?;
        let act = candle_nn::ops::silu(&g)?.mul(&u)?.to_dtype(DType::BF16)?;
        let y = self.shared_down_proj.forward(&act)?.to_dtype(DType::F32)?.contiguous()?;
        let gate_logits = self.shared_expert_gate.forward(x)?.to_dtype(DType::F32)?.contiguous()?;
        let (rows, hidden) = y.dims2()?;
        let out = Tensor::zeros((rows, hidden), DType::F32, x.device())?;
        let stream = stream_of(x.device())?;
        let rc = unsafe {
            nvk::mul_sigmoid_rowgate_f32(
                stream.cu_stream() as *mut c_void,
                cuda_ptr::<f32>(&y, &stream)? as *const f32,
                cuda_ptr::<f32>(&gate_logits, &stream)? as *const f32,
                cuda_ptr::<f32>(&out, &stream)? as *mut f32,
                rows as i32,
                hidden as i32,
            )
        };
        ensure!(rc == 0, "mul_sigmoid_rowgate_f32 rc={rc}");
        Ok(out)
    }

    fn fused_tail_ready(&self, t: usize) -> bool {
        t <= MOE_FUSED_TAIL_ROWS_MAX_6_MATCHES_THE_E4M3_MK_ROWS2_FAMILY_THE_SHARED_EXPERT_REPRODUCES
            && self.router.weight().is_some()
            && self.shared_expert_gate.weight().is_some()
            && [&self.router, &self.shared_expert_gate, &self.shared_gate_proj, &self.shared_up_proj, &self.shared_down_proj]
                .iter()
                .all(|l| !l.has_lora())
            && [&self.shared_gate_proj, &self.shared_up_proj, &self.shared_down_proj]
                .iter()
                .all(|l| l.fp8_e4m3_row_weight_and_scales_so_gdn_prenorm_folds_into_gemv_e4m3_mk_h().is_some())
    }

    fn forward_fused_tail(&self, x: &Tensor, cfg: &Qwen4ExpConfig, decode_ctx: &Mutex<GroupedDecodeContext>, aux: &Arc<CudaStream>) -> Result<Tensor> {
        let (t, h) = x.dims2()?;
        let device = x.device().clone();
        let stream = stream_of(&device)?;
        let x = x.contiguous()?;
        let si = cfg.shared_expert_intermediate_size;
        let k = cfg.num_experts_per_tok;
        let mut ctx = decode_ctx.lock().map_err(|e| anyhow::anyhow!("qwen4_exp decode ctx poisoned: {e}"))?;
        ensure!(ctx.n_tokens() == t, "qwen4_exp staged moe ctx holds {} rows, got {t}", ctx.n_tokens());
        ensure!(!ctx.folds_shared() && ctx.tiles_per_token() == k, "fused moe tail expects an unfolded {k}-slot context");
        let xp = cuda_ptr::<bf16>(&x, &stream)?;
        let (gw, gs) = self.shared_gate_proj.fp8_e4m3_row_weight_and_scales_so_gdn_prenorm_folds_into_gemv_e4m3_mk_h().context("shared gate fp8")?;
        let (uw, us) = self.shared_up_proj.fp8_e4m3_row_weight_and_scales_so_gdn_prenorm_folds_into_gemv_e4m3_mk_h().context("shared up fp8")?;
        let (dw, ds) = self.shared_down_proj.fp8_e4m3_row_weight_and_scales_so_gdn_prenorm_folds_into_gemv_e4m3_mk_h().context("shared down fp8")?;
        let act = uninit::<bf16>(&device, &[t, si])?;
        let fork = stream.record_event(None).map_err(|e| anyhow::anyhow!("moe shared-expert fork event: {e:?}"))?;
        aux.wait(&fork).map_err(|e| anyhow::anyhow!("moe shared-expert fork wait: {e:?}"))?;
        let rc = unsafe {
            nvk::q4exp_fp8_gateup_silu(
                aux.cu_stream() as *mut c_void,
                slice_ptr(gw, aux) as *const u8,
                slice_ptr(gs, aux) as *const f32,
                slice_ptr(uw, aux) as *const u8,
                slice_ptr(us, aux) as *const f32,
                xp as *const u16,
                cuda_ptr::<bf16>(&act, aux)? as *mut u16,
                t as i32,
                si as i32,
                h as i32,
            )
        };
        ensure!(rc == 0, "q4exp_fp8_gateup_silu rc={rc}");
        let join = aux.record_event(None).map_err(|e| anyhow::anyhow!("moe shared-expert join event: {e:?}"))?;
        let sg_rows_from_the_multi_row_gemm_kept_alive_until_the_tail_reads_them: Option<Tensor>;
        let sg_ptr = if t == 1 {
            let router_w = cuda_ptr::<bf16>(self.router.weight().context("router weight not bf16")?, &stream)?;
            let sg_w = cuda_ptr::<bf16>(self.shared_expert_gate.weight().context("shared gate weight not bf16")?, &stream)?;
            ctx.fused_router_gemv_sg_and_x_stage_replacing_gate_gemm_cast_and_dtod(&stream, xp, router_w, Some(sg_w))?;
            sg_rows_from_the_multi_row_gemm_kept_alive_until_the_tail_reads_them = None;
            ctx.sg_logits_bf16_written_by_the_router_gemv_arm_dev_ptr(&stream)
        } else {
            let logits = self.router.forward(&x)?.contiguous()?;
            ensure!(logits.dtype() == DType::BF16, "router logits must be bf16 for the staged chain");
            ctx.cast_router_logits_bf16_into_logits_f32_softcap_applied_at_route_not_here(&stream, cuda_ptr::<bf16>(&logits, &stream)?)?;
            ctx.stage_x_bf16_rows_dtod(&stream, xp)?;
            let sg = self.shared_expert_gate.forward(&x)?.contiguous()?;
            ensure!(sg.dtype() == DType::BF16 && sg.elem_count() == t, "shared expert gate logits must be bf16 [{t}, 1]");
            let p = cuda_ptr::<bf16>(&sg, &stream)?;
            sg_rows_from_the_multi_row_gemm_kept_alive_until_the_tail_reads_them = Some(sg);
            p
        };
        nv_layers::moe_grouped::forward_grouped_decode_into(&self.grouped, &mut ctx, 0, 0, 0.0, true, 1.0, false, &stream, None, None)?;
        stream.wait(&join).map_err(|e| anyhow::anyhow!("moe shared-expert join wait: {e:?}"))?;
        let out = uninit::<bf16>(&device, &[t, h])?;
        let rc = unsafe {
            nvk::q4exp_fp8_down_tail(
                stream.cu_stream() as *mut c_void,
                slice_ptr(dw, &stream) as *const u8,
                slice_ptr(ds, &stream) as *const f32,
                cuda_ptr::<bf16>(&act, &stream)? as *const u16,
                slice_ptr(&ctx.y_down, &stream) as *const u16,
                (nv_quant::nvfp4::MIN_TILE * h) as i64,
                slice_ptr(&ctx.topk_weights, &stream) as *const f32,
                k as i32,
                sg_ptr as *const u16,
                cuda_ptr::<bf16>(&out, &stream)? as *mut u16,
                t as i32,
                h as i32,
                si as i32,
            )
        };
        ensure!(rc == 0, "q4exp_fp8_down_tail rc={rc}");
        drop(sg_rows_from_the_multi_row_gemm_kept_alive_until_the_tail_reads_them);
        Ok(out)
    }

    fn forward_maybe_fused(
        &self,
        x: &Tensor,
        cfg: &Qwen4ExpConfig,
        decode_ctx: Option<&Mutex<GroupedDecodeContext>>,
        aux: Option<&Arc<CudaStream>>,
    ) -> Result<Tensor> {
        if let (Some(ctx), Some(aux)) = (decode_ctx, aux) {
            if self.fused_tail_ready(x.dims2()?.0) {
                return self.forward_fused_tail(x, cfg, ctx, aux);
            }
        }
        self.forward(x, cfg, decode_ctx)
    }

    fn forward(&self, x: &Tensor, cfg: &Qwen4ExpConfig, decode_ctx: Option<&Mutex<GroupedDecodeContext>>) -> Result<Tensor> {
        let (t, _) = x.dims2()?;
        let device = x.device().clone();
        let x = x.contiguous()?;
        let logits = self.router.forward(&x)?;
        let routed = if let Some(decode_ctx) = decode_ctx {
            let mut ctx = decode_ctx
                .lock()
                .map_err(|e| anyhow::anyhow!("qwen4_exp decode ctx poisoned: {e}"))?;
            ensure!(ctx.n_tokens() == t, "qwen4_exp staged moe ctx holds {} rows, got {t}", ctx.n_tokens());
            let stream = stream_of(&device)?;
            let logits = logits.contiguous()?;
            ensure!(logits.dtype() == DType::BF16, "router logits must be bf16 for the staged chain");
            ctx.cast_router_logits_bf16_into_logits_f32_softcap_applied_at_route_not_here(
                &stream,
                cuda_ptr::<bf16>(&logits, &stream)?,
            )?;
            ctx.stage_x_bf16_rows_dtod(&stream, cuda_ptr::<bf16>(&x, &stream)?)?;
            nv_layers::moe_grouped::forward_grouped_decode_into(
                &self.grouped,
                &mut ctx,
                0,
                0,
                0.0,
                true,
                1.0,
                true,
                &stream,
                None,
                None,
            )?;
            ctx.copy_scattered_f32_into_owned_tensor(&device, &stream)?
        } else {
            let routing = route_topk_and_plan_padded_on_device_for_prefill_dispatch(
                &logits,
                None,
                t,
                cfg.num_experts,
                cfg.num_experts_per_tok,
                0,
                0.0,
                true,
                1.0,
                &device,
            )?;
            forward_grouped_device_routing(&self.grouped, &self.grouped, &x, &routing, t, cfg.num_experts_per_tok, &device)?
        };
        let routed = routed.to_dtype(DType::F32)?;
        let routed = match self.expert_lora() {
            Some(l) => (routed + self.served_expert_lora_delta(&l, &x, &logits, cfg)?)?,
            None => routed,
        };
        let shared = self.shared(&x)?;
        let y = (routed + shared)?;
        Ok(y.to_dtype(DType::BF16)?)
    }
}

pub struct PinnedTable {
    ptr: *mut c_void,
    bytes: usize,
}

impl PinnedTable {
    fn from_bytes(src: &[u8], what: &str) -> Result<Self> {
        let bytes = src.len();
        let ptr = unsafe { cudarc::driver::result::malloc_host(bytes, 0) }
            .map_err(|e| anyhow::anyhow!("cuMemHostAlloc {bytes} bytes for {what}: {e:?}"))?;
        unsafe { std::slice::from_raw_parts_mut(ptr as *mut u8, bytes) }.copy_from_slice(src);
        Ok(Self { ptr, bytes })
    }

    fn device_ptr(&self) -> u64 {
        self.ptr as u64
    }
}

unsafe impl Send for PinnedTable {}
unsafe impl Sync for PinnedTable {}

impl Drop for PinnedTable {
    fn drop(&mut self) {
        unsafe {
            let _ = cudarc::driver::result::free_host(self.ptr);
        }
    }
}

pub struct PleLayer {
    shard_names: Vec<String>,
    rows_per_shard: usize,
    row_bytes: usize,
    scale: f32,
    multipliers: Vec<i64>,
    vocab_sizes: Vec<i64>,
    offsets: Vec<i64>,
    multipliers_dev: CudaSlice<i64>,
    vocab_sizes_dev: CudaSlice<i64>,
    offsets_dev: CudaSlice<i64>,
    table: Option<PinnedTable>,
    key_proj: Linear,
    value_proj: Linear,
    norm_key_w: Tensor,
    norm_query_w: Tensor,
    norm_conv_w: Tensor,
    conv_w: Tensor,
}

pub fn ngram_ids_host(
    cfg: &Qwen4ExpConfig,
    multipliers: &[i64],
    vocab_sizes: &[i64],
    offsets: &[i64],
    history: &[u32],
    tokens: &[u32],
) -> Vec<i64> {
    let eos = cfg.eos_token_id as i64;
    let ctx = cfg.ngram_size - 1;
    assert_eq!(history.len(), ctx, "ngram history must hold ngram_size-1 tokens");
    let mut full: Vec<i64> = history.iter().map(|&x| x as i64).collect();
    full.extend(tokens.iter().map(|&x| x as i64));
    let heads = cfg.heads_per_ngram;
    let mut out = Vec::with_capacity(tokens.len() * heads * ctx);
    for t in ctx..full.len() {
        let shifted = |shift: usize| -> i64 {
            if shift == 0 {
                return full[t];
            }
            if full[t - shift..t].contains(&eos) {
                return eos;
            }
            full[t - shift]
        };
        for ngram in 2..=cfg.ngram_size {
            let mut mixed = shifted(0).wrapping_mul(multipliers[0]);
            for p in 1..ngram {
                mixed ^= shifted(p).wrapping_mul(multipliers[p]);
            }
            let start = (ngram - 2) * heads;
            for hidx in start..start + heads {
                out.push(mixed.rem_euclid(vocab_sizes[hidx]) + offsets[hidx]);
            }
        }
    }
    out
}

pub fn ngram_ids_device(
    cfg: &Qwen4ExpConfig,
    multipliers: &[i64],
    vocab_sizes: &[i64],
    offsets: &[i64],
    history: &[u32],
    tokens: &[u32],
    device: &Device,
) -> Result<CudaSlice<i64>> {
    let stream = stream_of(device)?;
    let heads = cfg.ngram_heads();
    let t = tokens.len();
    let (tokens_dev, history_dev, m_dev, s_dev, o_dev): (
        CudaSlice<u32>,
        CudaSlice<u32>,
        CudaSlice<i64>,
        CudaSlice<i64>,
        CudaSlice<i64>,
    ) = (
        stream.clone_htod(tokens)?,
        stream.clone_htod(history)?,
        stream.clone_htod(multipliers)?,
        stream.clone_htod(vocab_sizes)?,
        stream.clone_htod(offsets)?,
    );
    let mut ids: CudaSlice<i64> = stream.alloc_zeros::<i64>(t * heads)?;
    let rc = unsafe {
        nvk::q4exp_ngram_ids(
            stream.cu_stream() as *mut c_void,
            slice_ptr(&tokens_dev, &stream) as *const u32,
            slice_ptr(&history_dev, &stream) as *const u32,
            slice_ptr(&m_dev, &stream) as *const i64,
            slice_ptr(&s_dev, &stream) as *const i64,
            slice_ptr(&o_dev, &stream) as *const i64,
            cfg.eos_token_id,
            t as i32,
            (cfg.ngram_size - 1) as i32,
            cfg.heads_per_ngram as i32,
            slice_ptr_mut(&mut ids, &stream) as *mut i64,
        )
    };
    ensure!(rc == 0, "q4exp_ngram_ids rc={rc}");
    Ok(ids)
}

pub fn ple_gather_bf16(table_host_ptr: u64, ids: &CudaSlice<i64>, scale: f32, t: usize, heads: usize, hd: usize, device: &Device) -> Result<Tensor> {
    let stream = stream_of(device)?;
    let emb = Tensor::zeros((t, heads * hd), DType::BF16, device)?;
    let rc = unsafe {
        nvk::q4exp_ple_gather_bf16(
            stream.cu_stream() as *mut c_void,
            table_host_ptr as *const u8,
            slice_ptr(ids, &stream) as *const i64,
            scale,
            t as i32,
            heads as i32,
            hd as i32,
            cuda_ptr::<bf16>(&emb, &stream)? as *mut u16,
        )
    };
    ensure!(rc == 0, "q4exp_ple_gather_bf16 rc={rc}");
    Ok(emb)
}

pub fn fp8_e4m3_to_f32(b: u8) -> f32 {
    let sign = if b & 0x80 != 0 { -1.0f32 } else { 1.0 };
    let exp = ((b >> 3) & 0x0F) as i32;
    let mant = (b & 0x07) as f32;
    if exp == 0 {
        sign * mant / 8.0 * 2f32.powi(-6)
    } else if b & 0x7F == 0x7F {
        f32::NAN
    } else {
        sign * (1.0 + mant / 8.0) * 2f32.powi(exp - 7)
    }
}

impl PleLayer {
    fn load(weights: &WeightLoader, prefix: &str, cfg: &Qwen4ExpConfig, stream: &Arc<CudaStream>) -> Result<Self> {
        let h = cfg.hidden_size;
        let hh = cfg.hc_hidden();
        let emb = cfg.ple_embed_dim;
        let heads = cfg.ngram_heads();
        let i64_vec = |name: &str| -> Result<Vec<i64>> {
            Ok(weights
                .get(&format!("{prefix}.ple_embedding.{name}"), DType::I64)?
                .flatten_all()?
                .to_vec1::<i64>()?)
        };
        let multipliers = i64_vec("layer_multipliers")?;
        ensure!(multipliers.len() == cfg.ngram_size, "ple multipliers {:?}", multipliers.len());
        let vocab_sizes = i64_vec("ngram_heads_vocab_sizes")?;
        let offsets = i64_vec("ngram_heads_offsets")?;
        ensure!(vocab_sizes.len() == heads && offsets.len() == heads, "ple head layout size");
        let mut shard_names = Vec::with_capacity(cfg.split_ngram_parts);
        let mut rows_per_shard = 0usize;
        for s in 0..cfg.split_ngram_parts {
            let name = format!("{prefix}.ple_embedding.ngram_embedding.shard_{s}.weight");
            let shape = weights
                .shape_of(&name)
                .ok_or_else(|| anyhow::anyhow!("missing {name}"))?;
            ensure!(shape.len() == 2 && shape[1] == emb / heads, "{name}: shape {shape:?}");
            if s == 0 {
                rows_per_shard = shape[0];
            } else {
                ensure!(shape[0] == rows_per_shard, "{name}: ragged shard rows {} vs {rows_per_shard}", shape[0]);
            }
            shard_names.push(name);
        }
        let row_bytes = emb / heads;
        let scale_t = weights.get(&format!("{prefix}.ple_embedding.ngram_embedding.weight_scale"), DType::F32)?;
        let scale = scale_t.flatten_all()?.to_vec1::<f32>()?[0];
        let conv_w = weights
            .get(&format!("{prefix}.conv1d.weight"), DType::F32)?
            .reshape((hh, cfg.ple_conv_kernel_size))?
            .contiguous()?;
        let host_gather = std::env::var(NV_Q4EXP_PLE_HOST_1_GATHERS_THE_NGRAM_ROWS_ON_THE_CPU_FROM_THE_MMAP_INSTEAD_OF_PINNING_51_GB)
            .as_deref()
            == Ok("1");
        let table = if host_gather {
            None
        } else {
            let t0 = std::time::Instant::now();
            let bytes = shard_names.len() * rows_per_shard * row_bytes;
            let ptr = unsafe { cudarc::driver::result::malloc_host(bytes, 0) }
                .map_err(|e| anyhow::anyhow!("cuMemHostAlloc {bytes} bytes for the PLE table: {e:?}"))?;
            let dst = unsafe { std::slice::from_raw_parts_mut(ptr as *mut u8, bytes) };
            let per = rows_per_shard * row_bytes;
            for (s, name) in shard_names.iter().enumerate() {
                let src = weights.raw_bytes(name)?;
                ensure!(src.len() == per, "{name}: {} bytes, expected {per}", src.len());
                dst[s * per..(s + 1) * per].copy_from_slice(src);
            }
            eprintln!(
                "[qwen4_exp] PLE table pinned in host memory: {:.1} GiB in {:.1}s",
                bytes as f64 / (1u64 << 30) as f64,
                t0.elapsed().as_secs_f64()
            );
            Some(PinnedTable { ptr, bytes })
        };
        let multipliers_dev = stream.clone_htod(&multipliers)?;
        let vocab_sizes_dev = stream.clone_htod(&vocab_sizes)?;
        let offsets_dev = stream.clone_htod(&offsets)?;
        Ok(Self {
            shard_names,
            rows_per_shard,
            row_bytes,
            scale,
            multipliers,
            vocab_sizes,
            offsets,
            multipliers_dev,
            vocab_sizes_dev,
            offsets_dev,
            table,
            key_proj: load_linear_bf16(weights, &format!("{prefix}.key_proj.weight"), hh, emb)?,
            value_proj: load_linear_bf16(weights, &format!("{prefix}.value_proj.weight"), h, emb)?,
            norm_key_w: load_norm_plus_one_f32(weights, &format!("{prefix}.norm_key.weight"), hh)?,
            norm_query_w: load_norm_plus_one_f32(weights, &format!("{prefix}.norm_query.weight"), hh)?,
            norm_conv_w: load_norm_plus_one_f32(weights, &format!("{prefix}.norm_conv.weight"), hh)?,
            conv_w,
        })
    }

    pub fn ngram_ids_host(&self, cfg: &Qwen4ExpConfig, history: &[u32], tokens: &[u32]) -> Vec<i64> {
        ngram_ids_host(cfg, &self.multipliers, &self.vocab_sizes, &self.offsets, history, tokens)
    }

    fn embeddings(
        &self,
        weights: &WeightLoader,
        cfg: &Qwen4ExpConfig,
        tokens: &[u32],
        history_host: &[u32],
        tokens_dev: &CudaSlice<u32>,
        history_dev: &CudaSlice<u32>,
        device: &Device,
    ) -> Result<Tensor> {
        let heads = cfg.ngram_heads();
        let hd = self.row_bytes;
        let t = tokens.len();
        let Some(table) = &self.table else {
            let ids = self.ngram_ids_host(cfg, history_host, tokens);
            let mut host = vec![bf16::ZERO; t * heads * hd];
            for (i, &id) in ids.iter().enumerate() {
                let id = id as usize;
                let shard = id / self.rows_per_shard;
                let row = id % self.rows_per_shard;
                let bytes = weights.raw_range_cow(&self.shard_names[shard], row * hd, hd)?;
                for (j, &b) in bytes.iter().enumerate() {
                    host[i * hd + j] = bf16::from_f32(bf16::from_f32(fp8_e4m3_to_f32(b)).to_f32() * self.scale);
                }
            }
            return Ok(Tensor::from_vec(host, (t, heads * hd), device)?);
        };
        let stream = stream_of(device)?;
        let mut ids: CudaSlice<i64> = stream.alloc_zeros::<i64>(t * heads)?;
        let rc = unsafe {
            nvk::q4exp_ngram_ids(
                stream.cu_stream() as *mut c_void,
                slice_ptr(tokens_dev, &stream) as *const u32,
                slice_ptr(history_dev, &stream) as *const u32,
                slice_ptr(&self.multipliers_dev, &stream) as *const i64,
                slice_ptr(&self.vocab_sizes_dev, &stream) as *const i64,
                slice_ptr(&self.offsets_dev, &stream) as *const i64,
                cfg.eos_token_id,
                t as i32,
                (cfg.ngram_size - 1) as i32,
                cfg.heads_per_ngram as i32,
                slice_ptr_mut(&mut ids, &stream) as *mut i64,
            )
        };
        ensure!(rc == 0, "q4exp_ngram_ids rc={rc}");
        let _ = table.bytes;
        ple_gather_bf16(table.ptr as u64, &ids, self.scale, t, heads, hd, device)
    }

    fn forward_device(
        &self,
        hc: &Tensor,
        emb: &Tensor,
        conv_state: &Tensor,
        cfg: &Qwen4ExpConfig,
        gvn_slab: Option<&Tensor>,
        update_window: bool,
    ) -> Result<Tensor> {
        let (t, hh) = hc.dims2()?;
        let n = cfg.hc_count;
        let h = cfg.hidden_size;
        let stream = stream_of(hc.device())?;
        let hc = hc.contiguous()?;
        let key = self.key_proj.forward(emb)?.contiguous()?;
        let value = self.value_proj.forward(emb)?.contiguous()?;
        let gv = Tensor::zeros((t, hh), DType::BF16, hc.device())?;
        let gvn = match gvn_slab {
            Some(slab) => {
                ensure!(slab.dims()[0] >= t && slab.dims()[1] == hh, "ple gvn slab {:?} holds fewer than {t} rows", slab.dims());
                slab.narrow(0, 0, t)?
            }
            None => Tensor::zeros((t, hh), DType::F32, hc.device())?,
        };
        let rc = unsafe {
            nvk::q4exp_ple_gate(
                stream.cu_stream() as *mut c_void,
                cuda_ptr::<bf16>(&key, &stream)? as *const u16,
                cuda_ptr::<bf16>(&value, &stream)? as *const u16,
                cuda_ptr::<bf16>(&hc, &stream)? as *const u16,
                cuda_ptr::<f32>(&self.norm_key_w, &stream)? as *const f32,
                cuda_ptr::<f32>(&self.norm_query_w, &stream)? as *const f32,
                cuda_ptr::<f32>(&self.norm_conv_w, &stream)? as *const f32,
                t as i32,
                h as i32,
                n as i32,
                cfg.rms_norm_eps as f32,
                cuda_ptr::<bf16>(&gv, &stream)? as *mut u16,
                cuda_ptr::<f32>(&gvn, &stream)? as *mut f32,
            )
        };
        ensure!(rc == 0, "q4exp_ple_gate rc={rc}");
        let dil = cfg.ngram_size;
        let k = cfg.ple_conv_kernel_size;
        let win = (k - 1) * dil;
        let out = Tensor::zeros((t, hh), DType::BF16, hc.device())?;
        let rc = unsafe {
            nvk::q4exp_ple_conv(
                stream.cu_stream() as *mut c_void,
                cuda_ptr::<f32>(&gvn, &stream)? as *const f32,
                cuda_ptr::<bf16>(&gv, &stream)? as *const u16,
                cuda_ptr::<f32>(conv_state, &stream)? as *mut f32,
                cuda_ptr::<f32>(&self.conv_w, &stream)? as *const f32,
                t as i32,
                hh as i32,
                k as i32,
                dil as i32,
                win as i32,
                update_window as i32,
                cuda_ptr::<bf16>(&out, &stream)? as *mut u16,
            )
        };
        ensure!(rc == 0, "q4exp_ple_conv rc={rc}");
        Ok(out)
    }

    fn advance_window(&self, conv_state: &Tensor, gvn_slab: &Tensor, rows: usize, cfg: &Qwen4ExpConfig) -> Result<()> {
        let stream = stream_of(conv_state.device())?;
        let hh = cfg.hc_hidden();
        let win = (cfg.ple_conv_kernel_size - 1) * cfg.ngram_size;
        let rc = unsafe {
            nvk::q4exp_ple_window_advance(
                stream.cu_stream() as *mut c_void,
                cuda_ptr::<f32>(gvn_slab, &stream)? as *const f32,
                cuda_ptr::<f32>(conv_state, &stream)? as *mut f32,
                rows as i32,
                hh as i32,
                win as i32,
            )
        };
        ensure!(rc == 0, "q4exp_ple_window_advance rc={rc}");
        Ok(())
    }

    fn forward(&self, hc: &Tensor, emb: &Tensor, conv_state: &Tensor, cfg: &Qwen4ExpConfig) -> Result<Tensor> {
        let (t, hh) = hc.dims2()?;
        let n = cfg.hc_count;
        let h = cfg.hidden_size;
        let key = self.key_proj.forward(emb)?;
        let value = self.value_proj.forward(emb)?.to_dtype(DType::F32)?;
        let kn = group_rmsnorm(&key, &self.norm_key_w, n, cfg.rms_norm_eps)?.to_dtype(DType::F32)?;
        let qn = group_rmsnorm(hc, &self.norm_query_w, n, cfg.rms_norm_eps)?.to_dtype(DType::F32)?;
        let gate = kn
            .reshape((t, n, h))?
            .mul(&qn.reshape((t, n, h))?)?
            .sum_keepdim(D::Minus1)?
            .affine(1.0 / (h as f64).sqrt(), 0.0)?;
        let abs = gate.abs()?;
        let mag = abs.affine(1.0, -1e-6)?.relu()?.affine(1.0, 1e-6)?.sqrt()?;
        let sign = gate.div(&abs.affine(1.0, -1e-30)?.relu()?.affine(1.0, 1e-30)?)?;
        let gate = mag.mul(&sign)?;
        let gv = candle_nn::ops::sigmoid(&gate)?
            .broadcast_mul(&value.reshape((t, 1, h))?)?
            .reshape((t, hh))?
            .to_dtype(DType::BF16)?;
        let gvn = group_rmsnorm(&gv, &self.norm_conv_w, n, cfg.rms_norm_eps)?.to_dtype(DType::F32)?;
        let dil = cfg.ngram_size;
        let k = cfg.ple_conv_kernel_size;
        let win = (k - 1) * dil;
        let full = Tensor::cat(&[conv_state, &gvn], 0)?;
        let mut acc = Tensor::zeros((t, hh), DType::F32, hc.device())?;
        for tap in 0..k {
            let x = full.narrow(0, tap * dil, t)?;
            let w = self.conv_w.narrow(1, tap, 1)?.reshape((1, hh))?;
            acc = (acc + x.broadcast_mul(&w)?)?;
        }
        let conv_out = candle_nn::ops::silu(&acc)?;
        let window = full.narrow(0, t, win)?.contiguous()?;
        conv_state.slice_set(&window, 0, 0)?;
        Ok((gv.to_dtype(DType::F32)? + conv_out)?.to_dtype(DType::BF16)?)
    }
}

pub enum Mixer {
    Gdn(LinearAttention),
    Attn(Attention),
}

pub struct Layer {
    attn_hc: HcGate,
    mlp_hc: HcGate,
    mixer: Mixer,
    moe: MoeLayer,
    ple: Option<PleLayer>,
    gdn_slot: Option<usize>,
    attn_slot: Option<usize>,
}

pub const NV_Q4EXP_MTP_DEFAULT_ON_0_SKIPS_THE_DRAFT_HEAD: &str = "NV_Q4EXP_MTP";

pub const MTP_PREFILL_ROWS_MAX_2048_MATCHES_THE_SERVING_PREFILL_CHUNK: usize = 2048;

fn mtp_on() -> bool {
    std::env::var(NV_Q4EXP_MTP_DEFAULT_ON_0_SKIPS_THE_DRAFT_HEAD).as_deref() != Ok("0")
}

pub struct MtpHead {
    pre_norm_emb: Tensor,
    pre_norm_hidden: Tensor,
    fc_embedding: Linear,
    fc_hidden: Linear,
    layer: Layer,
    mixer_hc: HcGate,
    inject_ones: Tensor,
}

impl MtpHead {
    fn load(
        weights: &WeightLoader,
        cfg: &Qwen4ExpConfig,
        runner: Arc<Mutex<Nvfp4GemmRunner>>,
        stream: &Arc<CudaStream>,
        fp8: Option<&Fp8Runner>,
        device: &Device,
    ) -> Result<Self> {
        let h = cfg.hidden_size;
        let hh = cfg.hc_hidden();
        let p = "mtp.layers.0";
        let last = format!("model.language_model.layers.{}.mlp.experts", cfg.num_hidden_layers - 1);
        let mut is_gu = 0f32;
        let mut is_dn = 0f32;
        for i in 0..cfg.num_experts {
            is_gu = is_gu.max(f32_scalar(weights, &format!("{last}.{i}.gate_proj.input_scale"))?);
            is_dn = is_dn.max(f32_scalar(weights, &format!("{last}.{i}.down_proj.input_scale"))?);
        }
        ensure!(is_gu > 0.0 && is_dn > 0.0, "mtp: no positive expert input_scale in the last target layer");
        let t0 = std::time::Instant::now();
        let grouped = load_grouped_experts_requantized_from_fp8_block_scales(weights, &format!("{p}.mlp"), cfg, runner, stream, is_gu, is_dn)?;
        eprintln!("[qwen4_exp] mtp experts requantized fp8-block -> nvfp4 in {:.1}s (input_scale gate/up {is_gu:.4e}, down {is_dn:.4e}), device {:.2} GiB used", t0.elapsed().as_secs_f64(), vram_used_gib());
        let attn = Attention::load(weights, &format!("{p}.self_attn"), cfg, fp8, device)?;
        Ok(Self {
            pre_norm_emb: load_norm_plus_one_f32(weights, "mtp.pre_fc_norm_embedding.weight", h)?,
            pre_norm_hidden: load_norm_plus_one_f32(weights, "mtp.pre_fc_norm_hidden.weight", hh)?,
            fc_embedding: load_linear_side(weights, "mtp.fc_embedding.weight", h, h, fp8, device)?,
            fc_hidden: load_linear_side(weights, "mtp.fc_hidden.weight", h, h, fp8, device)?,
            layer: Layer {
                attn_hc: HcGate::load(weights, &format!("{p}.attn_hyper_connection"), cfg, true)?,
                mlp_hc: HcGate::load(weights, &format!("{p}.mlp_hyper_connection"), cfg, true)?,
                mixer: Mixer::Attn(attn),
                moe: MoeLayer::load_with_grouped(weights, &format!("{p}.mlp"), cfg, grouped, fp8, device)?,
                ple: None,
                gdn_slot: None,
                attn_slot: None,
            },
            mixer_hc: HcGate::load(weights, "mtp.hyper_connection_mixer", cfg, false)?,
            inject_ones: Tensor::ones((MTP_PREFILL_ROWS_MAX_2048_MATCHES_THE_SERVING_PREFILL_CHUNK, cfg.hc_count), DType::F32, device)?,
        })
    }
}

pub const NV_Q4EXP_EMBED_DEVICE_1_KEEPS_THE_TOKEN_TABLE_IN_VRAM_INSTEAD_OF_PINNED_HOST_RAM: &str = "NV_Q4EXP_EMBED_DEVICE";

enum EmbedTable {
    Device(Tensor),
    Host(PinnedTable),
}

impl EmbedTable {
    fn ptr(&self, stream: &Arc<CudaStream>) -> Result<u64> {
        match self {
            EmbedTable::Device(t) => cuda_ptr::<bf16>(t, stream),
            EmbedTable::Host(p) => Ok(p.device_ptr()),
        }
    }
}

pub struct Qwen4Exp {
    cfg: Qwen4ExpConfig,
    weights: WeightLoader,
    embed: EmbedTable,
    layers: Vec<Layer>,
    final_hc: HcGate,
    lm_head: Linear,
    rope: Rope,
    device: Device,
    decode_ctx: Mutex<GroupedDecodeContext>,
    n_gdn: usize,
    n_attn: usize,
    mtp: Option<MtpHead>,
    sessions: [Mutex<Option<Qwen4ExpSession>>; 2],
    adapter_session: Mutex<Option<Qwen4ExpSession>>,
    base_identity: std::sync::OnceLock<String>,
    residency: Mutex<residency::Residency>,
    prefill_ctxs: Mutex<std::collections::HashMap<usize, Arc<Mutex<GroupedDecodeContext>>>>,
    moe_aux_stream: Arc<CudaStream>,
}

pub struct AttnSlot {
    k: Tensor,
    v: Tensor,
    raw_idx_k: Tensor,
    cmp_k: Tensor,
}

pub struct Qwen4ExpState {
    pub pos: usize,
    max_len: usize,
    gdn: Vec<LinAttnState>,
    attn: Vec<AttnSlot>,
    ple_history: Vec<u32>,
    ple_history_dev: CudaSlice<u32>,
    ple_conv: Tensor,
    rows_cap: usize,
    gdn_ckpts: Vec<Vec<LinAttnState>>,
    ple_gvn: Option<Tensor>,
    mtp_attn: Option<AttnSlot>,
    mm: mm::StateMm,
}

impl Qwen4ExpState {
    pub fn max_len(&self) -> usize {
        self.max_len
    }

    pub fn rows_cap(&self) -> usize {
        self.rows_cap
    }

    pub fn reset(&mut self, cfg: &Qwen4ExpConfig, device: &Device) -> Result<()> {
        let stream = stream_of(device)?;
        for s in &self.gdn {
            s.zero_data()?;
        }
        self.pos = 0;
        self.ple_history = vec![cfg.eos_token_id; cfg.ngram_size - 1];
        stream.memcpy_htod(&self.ple_history, &mut self.ple_history_dev)?;
        let zero = Tensor::zeros(self.ple_conv.dims(), DType::F32, device)?;
        self.ple_conv.slice_set(&zero, 0, 0)?;
        Ok(())
    }
}

pub struct DecodeBufs {
    rows_cap: usize,
    host_tok: Vec<u32>,
    host_pos: Box<[i32; 1]>,
    tok_buf: CudaSlice<u32>,
    pos_buf: CudaSlice<i32>,
    logits_buf: CudaSlice<f32>,
    hidden_buf: Option<Tensor>,
}

pub struct Qwen4ExpDecodeGraph {
    runner: nv_kernels::graph::CudaGraphRunner,
    bufs: DecodeBufs,
    captured: bool,
    verify_ctx: Vec<Mutex<GroupedDecodeContext>>,
}

pub const DECODE_GRAPH_SHAPE_TOKEN_ONE_GRAPH_PER_STATE_BECAUSE_EVERY_BUFFER_ADDRESS_IS_BAKED: u64 = 0x4e1;

struct FwdMode<'a> {
    pos_buf: Option<&'a CudaSlice<i32>>,
    moe_ctx: Option<&'a Mutex<GroupedDecodeContext>>,
    verify_rows: bool,
    last_layer: Option<usize>,
}

impl Qwen4ExpDecodeGraph {
    pub fn new(model: &Qwen4Exp) -> Result<Self> {
        Self::new_with_rows(model, 1, false)
    }

    pub fn new_with_rows(model: &Qwen4Exp, rows_cap: usize, keep_hidden: bool) -> Result<Self> {
        let dev = cuda_dev(&model.device)?;
        let stream = dev.cuda_stream();
        ensure!(
            !stream.cu_stream().is_null(),
            "qwen4_exp decode graph needs Device::new_cuda_with_stream (the legacy null stream cannot be captured)"
        );
        ensure!(rows_cap >= 1, "qwen4_exp decode graph needs rows_cap >= 1");
        nv_quant::nvfp4::ensure_workspace_for_stream(&stream)?;
        let cfg = &model.cfg;
        let tok_buf = stream.alloc_zeros::<u32>(rows_cap)?;
        let pos_buf = stream.alloc_zeros::<i32>(1)?;
        let logits_buf = stream.alloc_zeros::<f32>(rows_cap * cfg.vocab_size)?;
        let hidden_buf = if keep_hidden { Some(Tensor::zeros((rows_cap, cfg.hc_hidden()), DType::BF16, &model.device)?) } else { None };
        let mut verify_ctx = Vec::new();
        for rows in 2..=rows_cap {
            verify_ctx.push(Mutex::new(GroupedDecodeContext::new_multi(
                cfg.hidden_size,
                cfg.moe_intermediate_size,
                cfg.num_experts_per_tok,
                cfg.num_experts,
                rows,
                &stream,
            )?));
        }
        stream.synchronize()?;
        Ok(Self {
            runner: nv_kernels::graph::CudaGraphRunner::new(stream),
            bufs: DecodeBufs {
                rows_cap,
                host_tok: vec![0u32; rows_cap],
                host_pos: Box::new([0i32; 1]),
                tok_buf,
                pos_buf,
                logits_buf,
                hidden_buf,
            },
            captured: false,
            verify_ctx,
        })
    }

    pub fn rows_cap(&self) -> usize {
        self.bufs.rows_cap
    }

    pub fn captured(&self) -> bool {
        self.captured
    }
}

fn wrap_i32(buf: &CudaSlice<i32>, dev: &CudaDevice, n: usize) -> Result<Tensor> {
    let clone = buf.try_clone()?;
    let storage = candle_core::CudaStorage::wrap_cuda_slice(clone, dev.clone());
    Ok(Tensor::from_storage(Storage::Cuda(storage), (n,), candle_core::op::BackpropOp::none(), false))
}

pub fn qsa_compress_blocks(
    cfg: &Qwen4ExpConfig,
    raw_idx_k: &Tensor,
    cmp_k: &Tensor,
    idx_k_norm_w: &Tensor,
    rope: &Rope,
    first_block: usize,
    end_block: usize,
) -> Result<()> {
    let r = cfg.indexer_compress_ratio;
    let ih = cfg.indexer_head_dim;
    if end_block <= first_block {
        return Ok(());
    }
    let stream = stream_of(raw_idx_k.device())?;
    let rc = unsafe {
        nvk::q4exp_qsa_compress(
            stream.cu_stream() as *mut c_void,
            cuda_ptr::<bf16>(raw_idx_k, &stream)? as *const u16,
            first_block as i32,
            (end_block - first_block) as i32,
            cuda_ptr::<f32>(idx_k_norm_w, &stream)? as *const f32,
            cuda_ptr::<f32>(rope.cos(), &stream)? as *const f32,
            cuda_ptr::<f32>(rope.sin(), &stream)? as *const f32,
            cuda_ptr::<bf16>(cmp_k, &stream)? as *mut u16,
            r as i32,
            ih as i32,
            cfg.rotary_dim as i32,
            cfg.rms_norm_eps as f32,
        )
    };
    ensure!(rc == 0, "q4exp_qsa_compress rc={rc}");
    Ok(())
}

pub fn qsa_row_attention(
    cfg: &Qwen4ExpConfig,
    q: &Tensor,
    gate: &Tensor,
    iq: &Tensor,
    k_cache: &Tensor,
    v_cache: &Tensor,
    cmp_k: &Tensor,
    first_position: usize,
) -> Result<Tensor> {
    {
        let rows = q.dim(0)?;
        let row0 = 0usize;
        let pos = first_position;
        let device = q.device().clone();
        let nq = cfg.num_attention_heads;
        let nkv = cfg.num_key_value_heads;
        let hd = cfg.head_dim;
        let r = cfg.indexer_compress_ratio;
        let block_topk = cfg.qsa_block_topk();
        let width = cfg.indexer_budget + r - 1;
        let stream = stream_of(&device)?;
        let positions: Vec<i32> = (0..rows).map(|m| (pos + row0 + m) as i32).collect();
        let max_ctx = pos + row0 + rows;
        let sparse = max_ctx > cfg.dense_attention_context_limit();
        let positions_dev: CudaSlice<i32> = stream.clone_htod(&positions)?;
        let mut count: CudaSlice<i32> = unsafe { stream.alloc::<i32>(rows)? };
        let mut idx: Option<CudaSlice<i32>> = None;
        let max_count;
        if sparse {
            let nb: Vec<i32> = positions.iter().map(|&p| ((p + 1) as usize / r) as i32).collect();
            let nb_stride = *nb.iter().max().unwrap() as usize;
            let nb_dev: CudaSlice<i32> = stream.clone_htod(&nb)?;
            let mut scores: CudaSlice<f32> = unsafe { stream.alloc::<f32>(rows * nb_stride.max(1))? };
            let rc = unsafe {
                nvk::q4exp_qsa_scores(
                    stream.cu_stream() as *mut c_void,
                    cuda_ptr::<bf16>(iq, &stream)? as *const u16,
                    cuda_ptr::<bf16>(&cmp_k, &stream)? as *const u16,
                    slice_ptr(&nb_dev, &stream) as *const i32,
                    rows as i32,
                    cfg.indexer_n_heads as i32,
                    cfg.indexer_head_dim as i32,
                    nb_stride as i32,
                    1.0 / (cfg.indexer_head_dim as f32).sqrt(),
                    slice_ptr_mut(&mut scores, &stream) as *mut f32,
                )
            };
            ensure!(rc == 0, "q4exp_qsa_scores rc={rc}");
            let mut blocks: CudaSlice<i32> = unsafe { stream.alloc::<i32>(rows * block_topk)? };
            let mut block_count: CudaSlice<i32> = unsafe { stream.alloc::<i32>(rows)? };
            let rc = unsafe {
                nvk::q4exp_qsa_topk(
                    stream.cu_stream() as *mut c_void,
                    slice_ptr(&scores, &stream) as *const f32,
                    slice_ptr(&nb_dev, &stream) as *const i32,
                    rows as i32,
                    block_topk as i32,
                    nb_stride as i32,
                    slice_ptr_mut(&mut blocks, &stream) as *mut i32,
                    slice_ptr_mut(&mut block_count, &stream) as *mut i32,
                )
            };
            ensure!(rc == 0, "q4exp_qsa_topk rc={rc}");
            let mut idx_dev: CudaSlice<i32> = unsafe { stream.alloc::<i32>(rows * width)? };
            let rc = unsafe {
                nvk::q4exp_qsa_expand(
                    stream.cu_stream() as *mut c_void,
                    slice_ptr(&blocks, &stream) as *const i32,
                    slice_ptr(&block_count, &stream) as *const i32,
                    slice_ptr(&positions_dev, &stream) as *const i32,
                    rows as i32,
                    r as i32,
                    block_topk as i32,
                    width as i32,
                    slice_ptr_mut(&mut idx_dev, &stream) as *mut i32,
                    slice_ptr_mut(&mut count, &stream) as *mut i32,
                )
            };
            ensure!(rc == 0, "q4exp_qsa_expand rc={rc}");
            idx = Some(idx_dev);
            max_count = width.min(max_ctx);
        } else {
            let counts: Vec<i32> = positions.iter().map(|&p| p + 1).collect();
            {
                count = stream.clone_htod(&counts)?;
            }
            max_count = max_ctx;
        }
        let splits = (max_count.div_ceil(64)).clamp(1, 32);
        let mut partial: CudaSlice<f32> = unsafe { stream.alloc::<f32>(rows * nq * splits * (hd + 2))? };
        let out = uninit::<bf16>(&device, &[rows, nq * hd])?;
        let rc = unsafe {
            nvk::q4exp_attn_rows(
                stream.cu_stream() as *mut c_void,
                cuda_ptr::<bf16>(q, &stream)? as *const u16,
                cuda_ptr::<bf16>(&k_cache, &stream)? as *const u16,
                cuda_ptr::<bf16>(&v_cache, &stream)? as *const u16,
                idx.as_ref().map(|s| slice_ptr(s, &stream)).unwrap_or(0) as *const i32,
                slice_ptr(&count, &stream) as *const i32,
                rows as i32,
                nq as i32,
                nkv as i32,
                hd as i32,
                splits as i32,
                width as i32,
                1.0 / (hd as f32).sqrt(),
                slice_ptr_mut(&mut partial, &stream) as *mut f32,
            )
        };
        ensure!(rc == 0, "q4exp_attn_rows rc={rc}");
        let rc = unsafe {
            nvk::q4exp_attn_combine(
                stream.cu_stream() as *mut c_void,
                slice_ptr(&partial, &stream) as *const f32,
                cuda_ptr::<bf16>(gate, &stream)? as *const u16,
                rows as i32,
                nq as i32,
                hd as i32,
                splits as i32,
                cuda_ptr::<bf16>(&out, &stream)? as *mut u16,
            )
        };
        ensure!(rc == 0, "q4exp_attn_combine rc={rc}");
        Ok(out)
    }
}

impl Qwen4Exp {
    pub fn config(&self) -> &Qwen4ExpConfig {
        &self.cfg
    }

    pub fn device(&self) -> &Device {
        &self.device
    }

    pub fn load(dir: &Path, device: &Device) -> Result<Self> {
        let raw = std::fs::read_to_string(dir.join("config.json")).context("read config.json")?;
        let cfg = Qwen4ExpConfig::from_hf_json_str(&raw)?;
        let weights = WeightLoader::open_dir(dir, device)?;
        Self::from_loader(cfg, weights, device)
    }

    pub fn from_loader(cfg: Qwen4ExpConfig, weights: WeightLoader, device: &Device) -> Result<Self> {
        let dev = cuda_dev(device)?;
        let stream = dev.cuda_stream();
        {
            let ctx = stream.context().clone();
            if ctx.is_event_tracking() {
                unsafe { ctx.disable_event_tracking() };
                eprintln!("[qwen4_exp] cudarc event tracking disabled so decode graphs capture without event syncs");
            }
        }
        let runner = Arc::new(Mutex::new(Nvfp4GemmRunner::new(stream.clone())?));
        let fp8_runner = Arc::new(Mutex::new(nv_quant::fp8::Fp8GemmRunner::new(stream.clone())?));
        let fp8_side: Option<&Fp8Runner> = Some(&fp8_runner);
        let t0 = std::time::Instant::now();
        let embed = load_bf16(&weights, "model.language_model.embed_tokens.weight")?;
        ensure!(
            embed.dims() == [cfg.vocab_size, cfg.hidden_size],
            "embed_tokens shape {:?}",
            embed.dims()
        );
        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        eprintln!("[qwen4_exp] embeddings loaded, device {:.2} GiB used", vram_used_gib());
        let mut n_gdn = 0usize;
        let mut n_attn = 0usize;
        for i in 0..cfg.num_hidden_layers {
            let prefix = format!("model.language_model.layers.{i}");
            let (mixer, gdn_slot, attn_slot) = if cfg.full_attention[i] {
                let a = Attention::load(&weights, &format!("{prefix}.self_attn"), &cfg, fp8_side, device)?;
                n_attn += 1;
                (Mixer::Attn(a), None, Some(n_attn - 1))
            } else {
                let mut la = LinearAttention::from_loader_bf16_checkpoint_projections_quantized_to_fp8_resident_halving_gdn_decode_weight_traffic(
                    cfg.linear_attn_config(),
                    &format!("{prefix}.linear_attn"),
                    &weights,
                    DType::BF16,
                    device,
                    &fp8_runner,
                )?;
                la.set_output_gate_sigmoid_process_wide_because_the_gdn_kernels_read_one_constant_flag(true)?;
                n_gdn += 1;
                (Mixer::Gdn(la), Some(n_gdn - 1), None)
            };
            let ple = if cfg.ple_layer_ids_zero_based.contains(&i) {
                Some(PleLayer::load(&weights, &format!("{prefix}.ple"), &cfg, &stream)?)
            } else {
                None
            };
            layers.push(Layer {
                attn_hc: HcGate::load(&weights, &format!("{prefix}.attn_hyper_connection"), &cfg, true)?,
                mlp_hc: HcGate::load(&weights, &format!("{prefix}.mlp_hyper_connection"), &cfg, true)?,
                mixer,
                moe: MoeLayer::load(&weights, &format!("{prefix}.mlp"), &cfg, runner.clone(), &stream, fp8_side, device)?,
                ple,
                gdn_slot,
                attn_slot,
            });
            if i % 8 == 7 {
                eprintln!("[qwen4_exp] loaded {} / {} layers in {:.1}s, device {:.2} GiB used", i + 1, cfg.num_hidden_layers, t0.elapsed().as_secs_f64(), vram_used_gib());
            }
        }
        let final_hc = HcGate::load(&weights, "model.language_model.hyper_connection_mixer", &cfg, false)?;
        let keep_embed_on_device = cfg.tie_word_embeddings
            || std::env::var(NV_Q4EXP_EMBED_DEVICE_1_KEEPS_THE_TOKEN_TABLE_IN_VRAM_INSTEAD_OF_PINNED_HOST_RAM).as_deref() == Ok("1");
        let lm_head = if cfg.tie_word_embeddings {
            Linear::new_no_pretranspose(embed.clone(), None)?
        } else {
            load_linear_side(&weights, "lm_head.weight", cfg.vocab_size, cfg.hidden_size, fp8_side, device)?
        };
        let embed = if keep_embed_on_device {
            EmbedTable::Device(embed)
        } else {
            let host: Vec<bf16> = embed.flatten_all()?.to_vec1()?;
            drop(embed);
            let bytes: Vec<u8> = host.iter().flat_map(|v| v.to_bits().to_le_bytes()).collect();
            EmbedTable::Host(PinnedTable::from_bytes(&bytes, "the token embedding table")?)
        };
        let rope = Rope::new(
            RopeConfig {
                head_dim: cfg.rotary_dim,
                max_seq_len: cfg.max_position_embeddings,
                base: cfg.rope_theta,
                kind: RopeKind::Standard,
            },
            device,
        )?;
        let mtp = if mtp_on() && cfg.mtp_num_hidden_layers >= 1 {
            Some(MtpHead::load(&weights, &cfg, runner.clone(), &stream, fp8_side, device)?)
        } else {
            None
        };
        let decode_ctx = GroupedDecodeContext::new(
            cfg.hidden_size,
            cfg.moe_intermediate_size,
            cfg.num_experts_per_tok,
            cfg.num_experts,
            &stream,
        )?;
        stream.synchronize()?;
        eprintln!(
            "[qwen4_exp] model loaded in {:.1}s (gdn projections and attention/shared/lm_head side layers fp8 e4m3 rows), device {:.2} GiB used",
            t0.elapsed().as_secs_f64(),
            vram_used_gib()
        );
        Ok(Self {
            cfg,
            weights,
            embed,
            layers,
            final_hc,
            lm_head,
            rope,
            device: device.clone(),
            decode_ctx: Mutex::new(decode_ctx),
            n_gdn,
            n_attn,
            mtp,
            sessions: [Mutex::new(None), Mutex::new(None)],
            adapter_session: Mutex::new(None),
            base_identity: std::sync::OnceLock::new(),
            residency: Mutex::new(residency::Residency::new()),
            prefill_ctxs: Mutex::new(std::collections::HashMap::new()),
            moe_aux_stream: stream.context().new_stream()?,
        })
    }

    fn staged_moe_ctx_for_rows(&self, rows: usize) -> Result<Option<Arc<Mutex<GroupedDecodeContext>>>> {
        if rows < 2 || rows > STAGED_MOE_PREFILL_ROWS_MAX_16_BELOW_WHICH_THE_GEMV_CHAIN_BEATS_THE_GROUPED_GEMM_DISPATCH {
            return Ok(None);
        }
        let mut map = self.prefill_ctxs.lock().map_err(|e| anyhow::anyhow!("prefill ctx map poisoned: {e}"))?;
        if let Some(c) = map.get(&rows) {
            return Ok(Some(c.clone()));
        }
        let cfg = &self.cfg;
        let stream = stream_of(&self.device)?;
        let ctx = GroupedDecodeContext::new_multi(
            cfg.hidden_size,
            cfg.moe_intermediate_size,
            cfg.num_experts_per_tok,
            cfg.num_experts,
            rows,
            &stream,
        )?;
        stream.synchronize()?;
        let arc = Arc::new(Mutex::new(ctx));
        map.insert(rows, arc.clone());
        Ok(Some(arc))
    }

    pub fn table_placement(&self) -> serde_json::Value {
        let ple: Vec<&str> = self.layers.iter().filter_map(|l| l.ple.as_ref()).map(|p| if p.table.is_some() { "pinned_host" } else { "host_mmap" }).collect();
        serde_json::json!({
            "token_embedding": match self.embed { EmbedTable::Device(_) => "device", EmbedTable::Host(_) => "pinned_host" },
            "ple_ngram_tables": ple,
        })
    }

    pub fn set_base_identity(&self, id: String) {
        let _ = self.base_identity.set(id);
    }

    pub fn base_identity(&self) -> Option<&str> {
        self.base_identity.get().map(String::as_str)
    }

    pub fn has_mtp(&self) -> bool {
        self.mtp.is_some()
    }

    pub fn new_state(&self, max_len: usize) -> Result<Qwen4ExpState> {
        self.new_state_with_rows(max_len, 1, false)
    }

    fn new_attn_slot(&self, max_len: usize) -> Result<AttnSlot> {
        let cfg = &self.cfg;
        let nkv = cfg.num_key_value_heads;
        let hd = cfg.head_dim;
        let r = cfg.indexer_compress_ratio;
        Ok(AttnSlot {
            k: Tensor::zeros((max_len, nkv, hd), DType::BF16, &self.device)?,
            v: Tensor::zeros((max_len, nkv, hd), DType::BF16, &self.device)?,
            raw_idx_k: Tensor::zeros((max_len, cfg.indexer_head_dim), DType::BF16, &self.device)?,
            cmp_k: Tensor::zeros((max_len.div_ceil(r), cfg.indexer_head_dim), DType::BF16, &self.device)?,
        })
    }

    pub fn new_state_with_rows(&self, max_len: usize, rows_cap: usize, with_mtp: bool) -> Result<Qwen4ExpState> {
        let cfg = &self.cfg;
        ensure!(
            (1..=FUSED_GLUE_ROWS_MAX_8_THE_HC_AND_ATTENTION_ROW_KERNELS_ARE_SIZED_FOR_DECODE_AND_VERIFY_ROWS).contains(&rows_cap),
            "qwen4_exp state rows_cap {rows_cap} outside 1..=8"
        );
        let stream = stream_of(&self.device)?;
        let mut attn = Vec::with_capacity(self.n_attn);
        for _ in 0..self.n_attn {
            attn.push(self.new_attn_slot(max_len)?);
        }
        let mut gdn = Vec::with_capacity(self.n_gdn);
        let mut gdn_ckpts = Vec::with_capacity(self.n_gdn);
        for layer in &self.layers {
            if let Mixer::Gdn(la) = &layer.mixer {
                ensure!(la.fused_decode_supported(), "qwen4_exp needs the fused GDN decode state");
                gdn.push(la.new_fused_state(&self.device)?);
                if rows_cap > 1 {
                    gdn_ckpts.push(la.new_fused_verify_ckpt_rows_off_one_slab_so_chunk_kernels_write_ckpts_in_place(&self.device, rows_cap)?);
                }
            }
        }
        let ple_gvn = if rows_cap > 1 {
            Some(Tensor::zeros((rows_cap, cfg.hc_hidden()), DType::F32, &self.device)?)
        } else {
            None
        };
        let mtp_attn = if with_mtp { Some(self.new_attn_slot(max_len)?) } else { None };
        let win = (cfg.ple_conv_kernel_size - 1) * cfg.ngram_size;
        let ple_history = vec![cfg.eos_token_id; cfg.ngram_size - 1];
        let ple_history_dev = stream.clone_htod(&ple_history)?;
        stream.synchronize()?;
        Ok(Qwen4ExpState {
            pos: 0,
            max_len,
            gdn,
            attn,
            ple_history,
            ple_history_dev,
            ple_conv: Tensor::zeros((win, cfg.hc_hidden()), DType::F32, &self.device)?,
            rows_cap,
            gdn_ckpts,
            ple_gvn,
            mtp_attn,
            mm: mm::StateMm::new(&self.rope, max_len)?,
        })
    }

    fn attention_project(
        &self,
        attn: &Attention,
        x: &Tensor,
        positions: &Tensor,
        rope: &Rope,
    ) -> Result<(Tensor, Tensor, Tensor, Tensor, Tensor, Tensor)> {
        let cfg = &self.cfg;
        let (t, _) = x.dims2()?;
        let nq = cfg.num_attention_heads;
        let nkv = cfg.num_key_value_heads;
        let hd = cfg.head_dim;
        let q_raw = attn.q_proj.forward(x)?.reshape((t, nq, 2 * hd))?;
        let q = q_raw.narrow(2, 0, hd)?.contiguous()?;
        let gate = q_raw.narrow(2, hd, hd)?.contiguous()?.reshape((t, nq * hd))?;
        let k = attn.k_proj.forward(x)?.reshape((t, nkv, hd))?;
        let v = attn.v_proj.forward(x)?.reshape((t, nkv, hd))?.contiguous()?;
        let q = head_rmsnorm_bf16(&q, &attn.q_norm_w, cfg.rms_norm_eps)?;
        let k = head_rmsnorm_bf16(&k, &attn.k_norm_w, cfg.rms_norm_eps)?;
        let (q, k) = rope.apply_partial_oop_bf16_cuda_one_launch_replacing_slice_cast_rope_cast_concat(&q, &k, positions, hd)?;
        let ih = cfg.indexer_head_dim;
        let inh = cfg.indexer_n_heads;
        let idx = attn.idx_qk_proj.forward(x)?;
        let iq = idx.narrow(1, 0, inh * ih)?.reshape((t, inh, ih))?.contiguous()?;
        let ik = idx.narrow(1, inh * ih, ih)?.contiguous()?;
        let iq = head_rmsnorm_bf16(&iq, &attn.idx_q_norm_w, cfg.rms_norm_eps)?;
        let (iq, _) = rope.apply_partial_oop_bf16_cuda_one_launch_replacing_slice_cast_rope_cast_concat(&iq, &iq, positions, ih)?;
        Ok((q.contiguous()?, gate, k.contiguous()?, v, iq.contiguous()?, ik))
    }

    fn attention_forward(
        &self,
        attn: &Attention,
        x: &Tensor,
        slot: &mut AttnSlot,
        pos: usize,
        positions: &Tensor,
    ) -> Result<Tensor> {
        self.attention_forward_rope(attn, x, slot, pos, positions, &self.rope)
    }

    fn attention_forward_rope(
        &self,
        attn: &Attention,
        x: &Tensor,
        slot: &mut AttnSlot,
        pos: usize,
        positions: &Tensor,
        rope: &Rope,
    ) -> Result<Tensor> {
        let cfg = &self.cfg;
        let (t, _) = x.dims2()?;
        let nq = cfg.num_attention_heads;
        let nkv = cfg.num_key_value_heads;
        let hd = cfg.head_dim;
        let (q, gate, k, v, iq, ik) = self.attention_project(attn, x, positions, rope)?;
        slot.k.slice_set(&k, 0, pos)?;
        slot.v.slice_set(&v, 0, pos)?;
        slot.raw_idx_k.slice_set(&ik, 0, pos)?;
        let new_ctx = pos + t;
        let r = cfg.indexer_compress_ratio;
        qsa_compress_blocks(cfg, &slot.raw_idx_k, &slot.cmp_k, &attn.idx_k_norm_w, rope, pos / r, new_ctx / r)?;

        let dense_limit = cfg.dense_attention_context_limit();
        let fused_rows = t <= FUSED_GLUE_ROWS_MAX_8_THE_HC_AND_ATTENTION_ROW_KERNELS_ARE_SIZED_FOR_DECODE_AND_VERIFY_ROWS;
        let gated = if new_ctx <= dense_limit && (!fused_rows || eager_glue()) {
            let k_all = slot.k.narrow(0, 0, new_ctx)?.unsqueeze(0)?.contiguous()?;
            let v_all = slot.v.narrow(0, 0, new_ctx)?.unsqueeze(0)?.contiguous()?;
            let attn_cfg = AttnConfig {
                num_heads: nq,
                num_kv_heads: nkv,
                head_dim: hd,
                softmax_scale: 1.0 / (hd as f32).sqrt(),
                causal: true,
            };
            let out = flash_attn(&q.unsqueeze(0)?.contiguous()?, &k_all, &v_all, &attn_cfg)?;
            let out = out.reshape((t, nq * hd))?.to_dtype(DType::F32)?;
            out.mul(&candle_nn::ops::sigmoid(&gate.to_dtype(DType::F32)?)?)?
                .to_dtype(DType::BF16)?
        } else {
            let chunk = SPARSE_ATTENTION_ROW_CHUNK_1024_BOUNDS_THE_SCORE_SCRATCH_AT_256_MIB_FOR_A_262K_CONTEXT;
            let mut parts = Vec::new();
            let mut row0 = 0usize;
            while row0 < t {
                let rows = (t - row0).min(chunk);
                let qc = q.narrow(0, row0, rows)?.contiguous()?;
                let gc = gate.narrow(0, row0, rows)?.contiguous()?;
                let iqc = iq.narrow(0, row0, rows)?.contiguous()?;
                parts.push(if !fused_rows {
                    qsa_tc::qsa_row_attention_tc(cfg, &qc, &gc, &iqc, &slot.k, &slot.v, &slot.cmp_k, pos + row0)?
                } else {
                    qsa_row_attention(cfg, &qc, &gc, &iqc, &slot.k, &slot.v, &slot.cmp_k, pos + row0)?
                });
                row0 += rows;
            }
            if parts.len() == 1 {
                parts.pop().unwrap()
            } else {
                Tensor::cat(&parts, 0)?
            }
        };
        attn.o_proj.forward(&gated)
    }

    fn attention_forward_devpos(
        &self,
        attn: &Attention,
        x: &Tensor,
        slot: &AttnSlot,
        pos_buf: &CudaSlice<i32>,
        positions: &Tensor,
        rope: &Rope,
    ) -> Result<Tensor> {
        let cfg = &self.cfg;
        let (t, _) = x.dims2()?;
        let nq = cfg.num_attention_heads;
        let nkv = cfg.num_key_value_heads;
        let hd = cfg.head_dim;
        let ih = cfg.indexer_head_dim;
        let r = cfg.indexer_compress_ratio;
        let stream = stream_of(&self.device)?;
        let _ = positions;
        let inh = cfg.indexer_n_heads;
        let x = x.contiguous()?;
        let q_raw = attn.q_proj.forward(&x)?.contiguous()?;
        let k_raw = attn.k_proj.forward(&x)?.contiguous()?;
        let v_raw = attn.v_proj.forward(&x)?.contiguous()?;
        let idx_raw = attn.idx_qk_proj.forward(&x)?.contiguous()?;
        let q = uninit::<bf16>(&self.device, &[t, nq, hd])?;
        let gate = uninit::<bf16>(&self.device, &[t, nq * hd])?;
        let iq = uninit::<bf16>(&self.device, &[t, inh, ih])?;
        let rc = unsafe {
            nvk::q4exp_attn_prep(
                stream.cu_stream() as *mut c_void,
                cuda_ptr::<bf16>(&q_raw, &stream)? as *const u16,
                cuda_ptr::<bf16>(&k_raw, &stream)? as *const u16,
                cuda_ptr::<bf16>(&v_raw, &stream)? as *const u16,
                cuda_ptr::<bf16>(&idx_raw, &stream)? as *const u16,
                cuda_ptr::<f32>(&attn.q_norm_w, &stream)? as *const f32,
                cuda_ptr::<f32>(&attn.k_norm_w, &stream)? as *const f32,
                cuda_ptr::<f32>(&attn.idx_q_norm_w, &stream)? as *const f32,
                cuda_ptr::<f32>(rope.cos(), &stream)? as *const f32,
                cuda_ptr::<f32>(rope.sin(), &stream)? as *const f32,
                slice_ptr(pos_buf, &stream) as *const i32,
                t as i32,
                nq as i32,
                nkv as i32,
                hd as i32,
                inh as i32,
                ih as i32,
                cfg.rotary_dim as i32,
                cfg.rms_norm_eps as f32,
                cuda_ptr::<bf16>(&q, &stream)? as *mut u16,
                cuda_ptr::<bf16>(&gate, &stream)? as *mut u16,
                cuda_ptr::<bf16>(&slot.k, &stream)? as *mut u16,
                cuda_ptr::<bf16>(&slot.v, &stream)? as *mut u16,
                cuda_ptr::<bf16>(&slot.raw_idx_k, &stream)? as *mut u16,
                cuda_ptr::<bf16>(&iq, &stream)? as *mut u16,
            )
        };
        ensure!(rc == 0, "q4exp_attn_prep rc={rc}");
        let rc = unsafe {
            nvk::q4exp_qsa_compress_devpos(
                stream.cu_stream() as *mut c_void,
                cuda_ptr::<bf16>(&slot.raw_idx_k, &stream)? as *const u16,
                slice_ptr(pos_buf, &stream) as *const i32,
                t as i32,
                cuda_ptr::<f32>(&attn.idx_k_norm_w, &stream)? as *const f32,
                cuda_ptr::<f32>(rope.cos(), &stream)? as *const f32,
                cuda_ptr::<f32>(rope.sin(), &stream)? as *const f32,
                cuda_ptr::<bf16>(&slot.cmp_k, &stream)? as *mut u16,
                r as i32,
                ih as i32,
                cfg.rotary_dim as i32,
                cfg.rms_norm_eps as f32,
            )
        };
        ensure!(rc == 0, "q4exp_qsa_compress_devpos rc={rc}");
        let block_topk = cfg.qsa_block_topk();
        let width = cfg.indexer_budget + r - 1;
        let nb_stride = slot.cmp_k.dim(0)?;
        let mut positions_dev: CudaSlice<i32> = unsafe { stream.alloc::<i32>(t)? };
        let mut nb_dev: CudaSlice<i32> = unsafe { stream.alloc::<i32>(t)? };
        let rc = unsafe {
            nvk::q4exp_decode_meta(
                stream.cu_stream() as *mut c_void,
                slice_ptr(pos_buf, &stream) as *const i32,
                t as i32,
                r as i32,
                slice_ptr_mut(&mut positions_dev, &stream) as *mut i32,
                slice_ptr_mut(&mut nb_dev, &stream) as *mut i32,
            )
        };
        ensure!(rc == 0, "q4exp_decode_meta rc={rc}");
        let mut scores: CudaSlice<f32> = unsafe { stream.alloc::<f32>(t * nb_stride)? };
        let rc = unsafe {
            nvk::q4exp_qsa_scores(
                stream.cu_stream() as *mut c_void,
                cuda_ptr::<bf16>(&iq, &stream)? as *const u16,
                cuda_ptr::<bf16>(&slot.cmp_k, &stream)? as *const u16,
                slice_ptr(&nb_dev, &stream) as *const i32,
                t as i32,
                cfg.indexer_n_heads as i32,
                ih as i32,
                nb_stride as i32,
                1.0 / (ih as f32).sqrt(),
                slice_ptr_mut(&mut scores, &stream) as *mut f32,
            )
        };
        ensure!(rc == 0, "q4exp_qsa_scores rc={rc}");
        let mut blocks: CudaSlice<i32> = unsafe { stream.alloc::<i32>(t * block_topk)? };
        let mut block_count: CudaSlice<i32> = unsafe { stream.alloc::<i32>(t)? };
        let rc = unsafe {
            nvk::q4exp_qsa_topk(
                stream.cu_stream() as *mut c_void,
                slice_ptr(&scores, &stream) as *const f32,
                slice_ptr(&nb_dev, &stream) as *const i32,
                t as i32,
                block_topk as i32,
                nb_stride as i32,
                slice_ptr_mut(&mut blocks, &stream) as *mut i32,
                slice_ptr_mut(&mut block_count, &stream) as *mut i32,
            )
        };
        ensure!(rc == 0, "q4exp_qsa_topk rc={rc}");
        let mut idx: CudaSlice<i32> = unsafe { stream.alloc::<i32>(t * width)? };
        let mut count: CudaSlice<i32> = unsafe { stream.alloc::<i32>(t)? };
        let rc = unsafe {
            nvk::q4exp_qsa_expand(
                stream.cu_stream() as *mut c_void,
                slice_ptr(&blocks, &stream) as *const i32,
                slice_ptr(&block_count, &stream) as *const i32,
                slice_ptr(&positions_dev, &stream) as *const i32,
                t as i32,
                r as i32,
                block_topk as i32,
                width as i32,
                slice_ptr_mut(&mut idx, &stream) as *mut i32,
                slice_ptr_mut(&mut count, &stream) as *mut i32,
            )
        };
        ensure!(rc == 0, "q4exp_qsa_expand rc={rc}");
        let splits = 16usize;
        let mut partial: CudaSlice<f32> = unsafe { stream.alloc::<f32>(t * nq * splits * (hd + 2))? };
        let out = uninit::<bf16>(&self.device, &[t, nq * hd])?;
        let rc = unsafe {
            nvk::q4exp_attn_rows(
                stream.cu_stream() as *mut c_void,
                cuda_ptr::<bf16>(&q, &stream)? as *const u16,
                cuda_ptr::<bf16>(&slot.k, &stream)? as *const u16,
                cuda_ptr::<bf16>(&slot.v, &stream)? as *const u16,
                slice_ptr(&idx, &stream) as *const i32,
                slice_ptr(&count, &stream) as *const i32,
                t as i32,
                nq as i32,
                nkv as i32,
                hd as i32,
                splits as i32,
                width as i32,
                1.0 / (hd as f32).sqrt(),
                slice_ptr_mut(&mut partial, &stream) as *mut f32,
            )
        };
        ensure!(rc == 0, "q4exp_attn_rows rc={rc}");
        let rc = unsafe {
            nvk::q4exp_attn_combine(
                stream.cu_stream() as *mut c_void,
                slice_ptr(&partial, &stream) as *const f32,
                cuda_ptr::<bf16>(&gate, &stream)? as *const u16,
                t as i32,
                nq as i32,
                hd as i32,
                splits as i32,
                cuda_ptr::<bf16>(&out, &stream)? as *mut u16,
            )
        };
        ensure!(rc == 0, "q4exp_attn_combine rc={rc}");
        attn.o_proj.forward(&out)
    }

    fn gdn_forward(&self, la: &LinearAttention, x: &Tensor, state: &LinAttnState, ckpts: Option<&[LinAttnState]>) -> Result<Tensor> {
        let (t, _) = x.dims2()?;
        let x3 = x.unsqueeze(0)?.contiguous()?;
        if t == 1 && !eager_glue() {
            if let Some(out) = la.forward_decode_fused(&x3, state)? {
                return Ok(out.reshape((1, self.cfg.hidden_size))?);
            }
        }
        if let Some(ckpts) = ckpts {
            ensure!(ckpts.len() >= t, "gdn verify: {} ckpt rows for {t} rows", ckpts.len());
            let out = la
                .forward_verify_mrow_projections_once_because_per_row_fused_steps_reread_every_gdn_weight(&x3, state, ckpts, None)?
                .ok_or_else(|| anyhow::anyhow!("gdn verify rows={t}: the fused mrow path declined"))?;
            return Ok(out.reshape((t, self.cfg.hidden_size))?);
        }
        let mut tmp = Some(state.clone());
        let out = la.forward_with_state(&x3, &mut tmp)?;
        let fresh = tmp.ok_or_else(|| anyhow::anyhow!("gdn prefill returned no state"))?;
        state.copy_data_from(&fresh)?;
        Ok(out.squeeze(0)?)
    }

    fn layers_forward(
        &self,
        hc: Tensor,
        tokens: &[u32],
        tokens_dev: &CudaSlice<u32>,
        state: &mut Qwen4ExpState,
        pos: usize,
        positions: &Tensor,
        mode: FwdMode<'_>,
        prof: &mut PhaseProf,
        trace: bool,
    ) -> Result<(Tensor, Option<(Tensor, Tensor)>)> {
        let cfg = &self.cfg;
        let pos_buf = mode.pos_buf;
        let mut hc = hc;
        let fused = !eager_glue();
        let mut pending: Option<(Tensor, Tensor)> = None;
        let depth = mode.last_layer.map_or(self.layers.len(), |l| l + 1);
        for (li, layer) in self.layers.iter().enumerate().take(depth) {
            if let Some(ple) = &layer.ple {
                hc = hc_settle(hc, pending.take(), cfg)?;
                let emb = ple.embeddings(
                    &self.weights,
                    cfg,
                    tokens,
                    &state.ple_history,
                    tokens_dev,
                    &state.ple_history_dev,
                    &self.device,
                )?;
                let add = if pos_buf.is_some() || (!eager_glue() && tokens.len() <= FUSED_GLUE_ROWS_MAX_8_THE_HC_AND_ATTENTION_ROW_KERNELS_ARE_SIZED_FOR_DECODE_AND_VERIFY_ROWS) {
                    let (slab, update) = if mode.verify_rows {
                        (state.ple_gvn.as_ref(), false)
                    } else {
                        (None, true)
                    };
                    ple.forward_device(&hc, &emb, &state.ple_conv, cfg, slab, update)?
                } else {
                    ple.forward(&hc, &emb, &state.ple_conv, cfg)?
                };
                hc = (hc.to_dtype(DType::F32)? + add.to_dtype(DType::F32)?)?.to_dtype(DType::BF16)?;
                if trace {
                    trace_row("ple_add", &add)?;
                }
                prof.lap("ple");
            }
            let (mixed, inject) = if fused {
                let (h2, m, inj) = layer.attn_hc.mix_combined(&hc, pending.take(), cfg)?;
                hc = h2;
                (m, inj)
            } else {
                layer.attn_hc.mix(&hc, cfg)?
            };
            prof.lap("hc_mix");
            let y = match &layer.mixer {
                Mixer::Gdn(la) => {
                    let slot = layer.gdn_slot.unwrap();
                    let ckpts = if mode.verify_rows { state.gdn_ckpts.get(slot).map(|v| v.as_slice()) } else { None };
                    let y = self.gdn_forward(la, &mixed, &state.gdn[slot], ckpts)?;
                    prof.lap("gdn");
                    y
                }
                Mixer::Attn(a) => {
                    let slot = &mut state.attn[layer.attn_slot.unwrap()];
                    let rope = &state.mm.rope;
                    let y = match pos_buf {
                        Some(pb) => self.attention_forward_devpos(a, &mixed, slot, pb, positions, rope)?,
                        None => self.attention_forward_rope(a, &mixed, slot, pos, positions, rope)?,
                    };
                    prof.lap("attn");
                    y
                }
            };
            let inject = inject.context("attention hyper-connection carries no inject weights")?;
            let (mixed, inject) = if fused {
                let (h2, m, inj) = layer.mlp_hc.mix_combined(&hc, Some((y, inject)), cfg)?;
                hc = h2;
                (m, inj)
            } else {
                hc = hc_combine(&hc, &y, &inject, cfg)?;
                prof.lap("hc_combine");
                layer.mlp_hc.mix(&hc, cfg)?
            };
            prof.lap("hc_mix");
            let y = layer.moe.forward_maybe_fused(&mixed, cfg, mode.moe_ctx, fused.then_some(&self.moe_aux_stream))?;
            prof.lap("moe");
            let inject = inject.context("mlp hyper-connection carries no inject weights")?;
            if fused {
                pending = Some((y, inject));
            } else {
                hc = hc_combine(&hc, &y, &inject, cfg)?;
                prof.lap("hc_combine");
            }
            if trace {
                hc = hc_settle(hc, pending.take(), cfg)?;
                trace_row(&format!("layer{li}"), &hc)?;
            }
        }
        Ok((hc, pending))
    }

    fn embed_hc(&self, tokens_dev: &CudaSlice<u32>, t: usize) -> Result<Tensor> {
        self.embed_rows(tokens_dev, t, self.cfg.hc_count)
    }

    fn embed_rows(&self, tokens_dev: &CudaSlice<u32>, t: usize, nb: usize) -> Result<Tensor> {
        let cfg = &self.cfg;
        let stream = stream_of(&self.device)?;
        let hc = Tensor::zeros((t, nb * cfg.hidden_size), DType::BF16, &self.device)?;
        let rc = unsafe {
            nvk::q4exp_embed_hc(
                stream.cu_stream() as *mut c_void,
                self.embed.ptr(&stream)? as *const u16,
                slice_ptr(tokens_dev, &stream) as *const u32,
                t as i32,
                cfg.hidden_size as i32,
                nb as i32,
                cuda_ptr::<bf16>(&hc, &stream)? as *mut u16,
            )
        };
        ensure!(rc == 0, "q4exp_embed_hc rc={rc}");
        Ok(hc)
    }

    fn ple_layer(&self) -> Option<&PleLayer> {
        self.layers.iter().find_map(|l| l.ple.as_ref())
    }

    fn mtp_forward(
        &self,
        mtp: &MtpHead,
        hidden: &Tensor,
        tokens_dev: &CudaSlice<u32>,
        t: usize,
        pos: usize,
        positions: &Tensor,
        pos_buf: Option<&CudaSlice<i32>>,
        moe_ctx: Option<&Mutex<GroupedDecodeContext>>,
        state: &mut Qwen4ExpState,
    ) -> Result<(Tensor, Tensor)> {
        let cfg = &self.cfg;
        let h = cfg.hidden_size;
        let hh = cfg.hc_hidden();
        let n = cfg.hc_count;
        let emb = self.embed_rows(tokens_dev, t, 1)?;
        let emb = if pos_buf.is_none() { self.mm_splice_rows(state, emb, pos + 1, t, 1)? } else { emb };
        let en = rmsnorm_rows(&emb, &mtp.pre_norm_emb, cfg.rms_norm_eps)?;
        let e = mtp.fc_embedding.forward(&en)?;
        let hn = rmsnorm_rows(&hidden.contiguous()?, &mtp.pre_norm_hidden, cfg.rms_norm_eps)?;
        let fh = mtp.fc_hidden.forward(&hn.reshape((t * n, h))?)?.reshape((t, hh))?;
        ensure!(t <= MTP_PREFILL_ROWS_MAX_2048_MATCHES_THE_SERVING_PREFILL_CHUNK, "mtp forward: {t} rows exceed the {} row inject buffer", MTP_PREFILL_ROWS_MAX_2048_MATCHES_THE_SERVING_PREFILL_CHUNK);
        let ones = mtp.inject_ones.narrow(0, 0, t)?;
        let fused = !eager_glue() && t <= HC_FUSED_ROWS_MAX_64_BELOW_WHICH_THE_DOT_GRID_BEATS_THREE_CUBLAS_CALLS_PER_MIX;
        let layer = &mtp.layer;
        let (hc, mixed, inject) = if fused {
            layer.attn_hc.mix_combined(&fh, Some((e, ones)), cfg)?
        } else {
            let hc = hc_combine(&fh, &e, &ones, cfg)?;
            let (mixed, inject) = layer.attn_hc.mix(&hc, cfg)?;
            (hc, mixed, inject)
        };
        let Mixer::Attn(a) = &layer.mixer else {
            bail!("mtp layer is not an attention layer")
        };
        let slot = state.mtp_attn.as_mut().ok_or_else(|| anyhow::anyhow!("state has no mtp attention slot"))?;
        let rope = &state.mm.rope;
        let y = match pos_buf {
            Some(pb) => self.attention_forward_devpos(a, &mixed, slot, pb, positions, rope)?,
            None => self.attention_forward_rope(a, &mixed, slot, pos, positions, rope)?,
        };
        let inject = inject.context("mtp attention hyper-connection carries no inject weights")?;
        let (hc, mixed, inject) = if fused {
            layer.mlp_hc.mix_combined(&hc, Some((y, inject)), cfg)?
        } else {
            let hc = hc_combine(&hc, &y, &inject, cfg)?;
            let (mixed, inject) = layer.mlp_hc.mix(&hc, cfg)?;
            (hc, mixed, inject)
        };
        let y = layer.moe.forward_maybe_fused(&mixed, cfg, moe_ctx, fused.then_some(&self.moe_aux_stream))?;
        let inject = inject.context("mtp mlp hyper-connection carries no inject weights")?;
        let (hc, mixed) = if fused {
            let (hc, mixed, _) = mtp.mixer_hc.mix_combined(&hc, Some((y, inject)), cfg)?;
            (hc, mixed)
        } else {
            let hc = hc_combine(&hc, &y, &inject, cfg)?;
            let (mixed, _) = mtp.mixer_hc.mix(&hc, cfg)?;
            (hc, mixed)
        };
        let logits = self.lm_head.forward(&mixed)?.to_dtype(DType::F32)?.contiguous()?;
        Ok((hc, logits))
    }

    fn push_history_device(&self, state: &mut Qwen4ExpState, tokens_dev: &CudaSlice<u32>, t: usize) -> Result<()> {
        let cfg = &self.cfg;
        if cfg.ple_layer_ids_zero_based.is_empty() {
            return Ok(());
        }
        let stream = stream_of(&self.device)?;
        let rc = unsafe {
            nvk::q4exp_history_push(
                stream.cu_stream() as *mut c_void,
                slice_ptr_mut(&mut state.ple_history_dev, &stream) as *mut u32,
                slice_ptr(tokens_dev, &stream) as *const u32,
                (cfg.ngram_size - 1) as i32,
                t as i32,
            )
        };
        ensure!(rc == 0, "q4exp_history_push rc={rc}");
        Ok(())
    }

    fn push_history_host(&self, state: &mut Qwen4ExpState, tokens: &[u32]) {
        let ctx = self.cfg.ngram_size - 1;
        let mut hist = state.ple_history.clone();
        hist.extend_from_slice(tokens);
        state.ple_history = hist[hist.len() - ctx..].to_vec();
    }

    pub fn forward(&self, tokens: &[u32], state: &mut Qwen4ExpState) -> Result<Tensor> {
        Ok(self.forward_rows(tokens, state)?.0)
    }

    pub fn forward_rows(&self, tokens: &[u32], state: &mut Qwen4ExpState) -> Result<(Tensor, Tensor)> {
        let cfg = &self.cfg;
        let t = tokens.len();
        ensure!(t > 0, "qwen4_exp forward: empty token slice");
        ensure!(
            state.pos + t <= state.max_len,
            "qwen4_exp forward: {} + {t} tokens exceed the {}-token state",
            state.pos,
            state.max_len
        );
        let pos = state.pos;
        let stream = stream_of(&self.device)?;
        let ids = ();
        let tokens_dev: CudaSlice<u32> = stream.clone_htod(tokens)?;
        let positions: Vec<i32> = (pos..pos + t).map(|p| p as i32).collect();
        let positions = Tensor::from_vec(positions, (t,), &self.device)?;
        let mut prof = PhaseProf::begin(&self.device);
        let _ = ids;
        let hc = self.mm_splice_rows(state, self.embed_hc(&tokens_dev, t)?, pos, t, cfg.hc_count)?;
        let trace = std::env::var("NV_Q4EXP_TRACE").as_deref() == Ok("1");
        prof.lap("embed");
        let staged = self.staged_moe_ctx_for_rows(t)?;
        let mode = FwdMode {
            pos_buf: None,
            moe_ctx: if t == 1 { Some(&self.decode_ctx) } else { staged.as_deref() },
            verify_rows: false,
            last_layer: None,
        };
        let (hc, pending) = self.layers_forward(hc, tokens, &tokens_dev, state, pos, &positions, mode, &mut prof, trace)?;
        let hc = hc_settle(hc, pending, cfg)?;
        self.push_history_device(state, &tokens_dev, t)?;
        self.push_history_host(state, tokens);
        state.pos += t;
        let last = hc.narrow(0, t - 1, 1)?.contiguous()?;
        let (mixed, _) = self.final_hc.mix(&last, cfg)?;
        if trace {
            trace_row("final_mix", &mixed)?;
        }
        let logits = self.lm_head.forward(&mixed)?.to_dtype(DType::F32)?;
        prof.lap("final_lm_head");
        prof.report(t);
        Ok((logits, hc))
    }

    pub fn residual_rows_after_layer(&self, tokens: &[u32], layer: usize, state: &mut Qwen4ExpState) -> Result<Tensor> {
        let cfg = &self.cfg;
        ensure!(
            layer < self.layers.len(),
            "capture layer {layer} out of range for {} blocks",
            self.layers.len()
        );
        ensure!(
            !tokens.is_empty() && tokens.len() <= state.max_len,
            "qwen4_exp capture: {} tokens outside 1..={}",
            tokens.len(),
            state.max_len
        );
        state.reset(cfg, &self.device)?;
        let stream = stream_of(&self.device)?;
        let mut prof = PhaseProf::disabled();
        let mut rows = Vec::new();
        let mut off = 0usize;
        while off < tokens.len() {
            let c = (tokens.len() - off).min(PREFILL_CHUNK_2048_IS_THE_PREFIX_CHECKPOINT_GRID_SO_A_RESUMED_PREFILL_CHUNKS_EXACTLY_AS_A_COLD_ONE);
            let part = &tokens[off..off + c];
            let tokens_dev: CudaSlice<u32> = stream.clone_htod(part)?;
            let positions: Vec<i32> = (off..off + c).map(|p| p as i32).collect();
            let positions = Tensor::from_vec(positions, (c,), &self.device)?;
            let hc = self.embed_hc(&tokens_dev, c)?;
            let staged = self.staged_moe_ctx_for_rows(c)?;
            let mode = FwdMode {
                pos_buf: None,
                moe_ctx: if c == 1 { Some(&self.decode_ctx) } else { staged.as_deref() },
                verify_rows: false,
                last_layer: Some(layer),
            };
            let (hc, pending) = self.layers_forward(hc, part, &tokens_dev, state, off, &positions, mode, &mut prof, false)?;
            let hc = hc_settle(hc, pending, cfg)?;
            self.push_history_device(state, &tokens_dev, c)?;
            self.push_history_host(state, part);
            state.pos += c;
            rows.push(
                hc.to_dtype(DType::F32)?
                    .reshape((c, cfg.hc_count, cfg.hidden_size))?
                    .mean(1)?,
            );
            off += c;
        }
        Ok(Tensor::cat(&rows, 0)?)
    }

    fn step_captured(
        &self,
        state: &mut Qwen4ExpState,
        g: &mut DecodeBufs,
        rows: usize,
        moe_ctx: &Mutex<GroupedDecodeContext>,
    ) -> Result<()> {
        let cfg = &self.cfg;
        let dev = cuda_dev(&self.device)?;
        let stream = stream_of(&self.device)?;
        ensure!(rows >= 1 && rows <= g.rows_cap, "qwen4_exp step rows {rows} outside 1..={}", g.rows_cap);
        {
            stream.memcpy_htod(&g.host_tok[..rows], &mut g.tok_buf.slice_mut(0..rows))?;
            stream.memcpy_htod(&g.host_pos[..], &mut g.pos_buf)?;
        }
        let positions = wrap_i32(&g.pos_buf, &dev, 1)?;
        let mut prof = PhaseProf::disabled();
        let tokens_host: Vec<u32> = g.host_tok[..rows].to_vec();
        let tok_buf = g.tok_buf.try_clone()?;
        let hc = self.embed_hc(&tok_buf, rows)?;
        let mode = FwdMode {
            pos_buf: Some(&g.pos_buf),
            moe_ctx: Some(moe_ctx),
            verify_rows: rows > 1,
            last_layer: None,
        };
        let (hc, pending) = self.layers_forward(hc, &tokens_host, &tok_buf, state, state.pos, &positions, mode, &mut prof, false)?;
        if rows == 1 {
            self.push_history_device(state, &tok_buf, 1)?;
        }
        let (hc, mixed) = match pending {
            Some(p) => {
                let (hc, mixed, _) = self.final_hc.mix_combined(&hc, Some(p), cfg)?;
                (hc, mixed)
            }
            None => {
                let (mixed, _) = self.final_hc.mix(&hc, cfg)?;
                (hc, mixed)
            }
        };
        if let Some(hbuf) = g.hidden_buf.as_ref() {
            let hh = cfg.hc_hidden();
            let hc = hc.contiguous()?;
            dtod_raw(cuda_ptr::<bf16>(hbuf, &stream)?, cuda_ptr::<bf16>(&hc, &stream)?, rows * hh * 2, &stream)?;
        }
        let logits = self.lm_head.forward(&mixed)?.to_dtype(DType::F32)?.contiguous()?;
        let (storage, layout) = logits.storage_and_layout();
        let Storage::Cuda(s) = &*storage else {
            bail!("logits not on CUDA")
        };
        let n = rows * cfg.vocab_size;
        let src = s.as_cuda_slice::<f32>()?.slice(layout.start_offset()..layout.start_offset() + n);
        stream.memcpy_dtod(&src, &mut g.logits_buf.slice_mut(0..n))?;
        Ok(())
    }

    pub fn decode_graphed(&self, token: u32, state: &mut Qwen4ExpState, g: &mut Qwen4ExpDecodeGraph) -> Result<Vec<f32>> {
        ensure!(state.pos + 1 <= state.max_len, "qwen4_exp decode: state is full");
        if self.has_expert_lora() {
            let _ = lora::EXPERT_LORA_DECODE_RUNS_EAGER_BECAUSE_ITS_ROUTED_ROWS_ARE_PLANNED_ON_THE_HOST;
            return Ok(self.forward(&[token], state)?.flatten_all()?.to_vec1::<f32>()?);
        }
        let Qwen4ExpDecodeGraph { runner, bufs, captured, .. } = g;
        if !*captured && !runner.has_cached() {
            let warm = self.forward(&[token], state)?;
            let out: Vec<f32> = warm.flatten_all()?.to_vec1::<f32>()?;
            *captured = true;
            return Ok(out);
        }
        bufs.host_tok[0] = token;
        bufs.host_pos[0] = state.pos as i32;
        let stream = runner.stream().clone();
        let result = runner.run(DECODE_GRAPH_SHAPE_TOKEN_ONE_GRAPH_PER_STATE_BECAUSE_EVERY_BUFFER_ADDRESS_IS_BAKED, |s| {
            nv_layers::cuda_stream::with_stream(s.clone(), || self.step_captured(state, bufs, 1, &self.decode_ctx))
        });
        match result {
            Ok(()) => {}
            Err(e) => {
                runner.invalidate();
                *captured = false;
                return Err(e).context("qwen4_exp decode graph capture/replay failed");
            }
        }
        *captured = true;
        self.push_history_host(state, &[token]);
        state.pos += 1;
        stream.synchronize()?;
        let out: Vec<f32> = stream.clone_dtoh(&bufs.logits_buf.slice(0..self.cfg.vocab_size))?;
        Ok(out)
    }
}

pub struct MtpBufs {
    host_tok: Vec<u32>,
    host_pos: Box<[i32; 1]>,
    tok_buf: CudaSlice<u32>,
    pos_buf: CudaSlice<i32>,
    logits_buf: CudaSlice<f32>,
    mh_buf: Tensor,
}

#[derive(Default, Debug, Clone)]
pub struct SpecStats {
    pub rounds: usize,
    pub drafted: usize,
    pub accepted: usize,
    pub emitted: usize,
    pub draft_ms: f64,
    pub verify_ms: f64,
    pub commit_ms: f64,
}

impl SpecStats {
    pub fn accept_rate(&self) -> f64 {
        if self.drafted == 0 { 0.0 } else { self.accepted as f64 / self.drafted as f64 }
    }

    pub fn tokens_per_round(&self) -> f64 {
        if self.rounds == 0 { 0.0 } else { self.emitted as f64 / self.rounds as f64 }
    }
}

#[path = "qwen4_exp_residency.rs"]
mod residency;
#[path = "qwen4_exp_mm.rs"]
pub mod mm;
pub use mm::Qwen4ExpMm;
#[path = "qwen4_exp_lora.rs"]
pub mod lora;
#[path = "qwen4_exp_qsa_tc.rs"]
pub mod qsa_tc;
pub use residency::{weights_tag_for_dir, ResidencyStats, SessionResidency};

pub struct PrefixCkpt {
    pos: usize,
    gdn: Vec<LinAttnState>,
    ple_conv: Tensor,
}

pub struct Qwen4ExpSession {
    pub state: Qwen4ExpState,
    pub graph: Qwen4ExpDecodeGraph,
    mtp: Option<MtpBufs>,
    k: usize,
    anchor: u32,
    pending: Vec<u32>,
    pub stats: SpecStats,
    prefix: Vec<u32>,
    ckpts: Vec<PrefixCkpt>,
    ckpt_cap: usize,
    lora_tag: Option<String>,
}

impl Qwen4ExpSession {
    pub fn lora_tag(&self) -> Option<&str> {
        self.lora_tag.as_deref()
    }

    pub fn spec_k(&self) -> usize {
        if self.mtp.is_some() { self.k } else { 0 }
    }

    pub fn anchor(&self) -> u32 {
        self.anchor
    }

    pub fn round_fits(&self) -> bool {
        self.state.pos + self.k < self.state.max_len
    }

    pub fn reusable(&self, prompt: &[u32]) -> usize {
        let positions: Vec<usize> = self.ckpts.iter().map(|c| c.pos).collect();
        crate::prefix_reuse::saved_state_resume_point(&positions, &self.prefix, prompt)
    }
}

pub const ADAPTER_SESSIONS_NEVER_ENTER_THE_BASE_SESSION_CACHE_OR_ITS_RESIDENCY_TIERS: &str =
    "a session prefilled under a LoRA adapter carries that adapter's id@version tag; put_session \
     drops it, so base requests, their disk snapshots and MTP speculation never see adapter KV";

pub const NV_Q4EXP_LORA_SESSION_CACHE_1_KEEPS_ONE_ADAPTER_SESSION_REUSED_ONLY_UNDER_THE_SAME_ADAPTER_ID_AND_VERSION: &str =
    "NV_Q4EXP_LORA_SESSION_CACHE";

pub const MTP_GRAPH_SHAPE_TOKEN_BASE_0X5E0_PLUS_ROWS: u64 = 0x5e0;

pub const SESSION_MIN_LEN_8192_SHORT_REQUESTS_SHARE_ONE_STATE_SO_THEY_NEVER_OVERWRITE_THE_LONG_SESSIONS_PREFIX: usize = 8192;

pub const PREFILL_CHUNK_2048_IS_THE_PREFIX_CHECKPOINT_GRID_SO_A_RESUMED_PREFILL_CHUNKS_EXACTLY_AS_A_COLD_ONE: usize = 2048;

pub const NV_Q4EXP_PREFIX_CKPTS_DEFAULT_4_HIGHEST_GRID_STATES_PER_SESSION_0_IS_THE_COLD_PREFILL_ARM: &str = "NV_Q4EXP_PREFIX_CKPTS";

pub fn prefix_ckpt_cap() -> usize {
    std::env::var(NV_Q4EXP_PREFIX_CKPTS_DEFAULT_4_HIGHEST_GRID_STATES_PER_SESSION_0_IS_THE_COLD_PREFILL_ARM)
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(4)
}

pub use crate::spec_pick::{accept_drafts, GreedyPick, SpecPick};

impl Qwen4Exp {
    pub fn new_session(&self, max_len: usize, k: usize) -> Result<Qwen4ExpSession> {
        let spec = self.mtp.is_some() && k >= 1;
        let rows_cap = if spec { k + 1 } else { 1 };
        let mut state = self.new_state_with_rows(max_len, rows_cap, spec)?;
        let mut graph = Qwen4ExpDecodeGraph::new_with_rows(self, rows_cap, spec)?;
        let cfg = &self.cfg;
        let stream = stream_of(&self.device)?;
        let mut mtp = None;
        if spec {
            let mut mb = MtpBufs {
                host_tok: vec![cfg.eos_token_id; rows_cap],
                host_pos: Box::new([0i32; 1]),
                tok_buf: stream.alloc_zeros::<u32>(rows_cap)?,
                pos_buf: stream.alloc_zeros::<i32>(1)?,
                logits_buf: stream.alloc_zeros::<f32>(cfg.vocab_size)?,
                mh_buf: Tensor::zeros((1, cfg.hc_hidden()), DType::BF16, &self.device)?,
            };
            stream.synchronize()?;
            let t0 = std::time::Instant::now();
            for rows in 1..=rows_cap {
                self.mtp_step_run(&mut state, &mut graph, &mut mb, rows, false)?;
                self.mtp_step_run(&mut state, &mut graph, &mut mb, rows, true)?;
            }
            for rows in 2..=rows_cap {
                graph.bufs.host_tok[..rows].fill(cfg.eos_token_id);
                graph.bufs.host_pos[0] = 0;
                self.verify_run(&mut state, &mut graph, rows, false)?;
                self.verify_run(&mut state, &mut graph, rows, true)?;
            }
            stream.synchronize()?;
            eprintln!("[qwen4_exp] spec session k={k} len={max_len}: captured {} verify + {} draft graph shapes in {:.2}s", rows_cap - 1, rows_cap, t0.elapsed().as_secs_f64());
            state.reset(cfg, &self.device)?;
            mtp = Some(mb);
        }
        Ok(Qwen4ExpSession {
            state,
            graph,
            mtp,
            k: if spec { k } else { 0 },
            anchor: 0,
            pending: Vec::new(),
            stats: SpecStats::default(),
            prefix: Vec::new(),
            ckpts: Vec::new(),
            ckpt_cap: prefix_ckpt_cap(),
            lora_tag: None,
        })
    }

    fn home_slot(prompt: &[u32]) -> usize {
        usize::from(prompt.len() > SESSION_MIN_LEN_8192_SHORT_REQUESTS_SHARE_ONE_STATE_SO_THEY_NEVER_OVERWRITE_THE_LONG_SESSIONS_PREFIX)
    }

    pub fn take_session(&self, prompt: &[u32], max_len: usize, k: usize) -> Result<Qwen4ExpSession> {
        Ok(self.take_session_traced(prompt, max_len, k)?.0)
    }

    pub fn take_session_traced(&self, prompt: &[u32], max_len: usize, k: usize) -> Result<(Qwen4ExpSession, Option<crate::session_lru::Tier>)> {
        use crate::session_lru::{Held, Tier};
        let want_k = if self.mtp.is_some() { k } else { 0 };
        let mut slots = self.sessions.iter().map(|m| m.lock().map_err(|e| anyhow::anyhow!("session mutex poisoned: {e}"))).collect::<Result<Vec<_>>>()?;
        let reusable = [0, 1].map(|i| slots[i].as_ref().filter(|s| s.spec_k() == want_k).map_or(0, |s| s.reusable(prompt)));
        let home = Self::home_slot(prompt);
        let from = crate::prefix_reuse::session_source(home, reusable);
        if home == 1 {
            let mut res = self.residency.lock().map_err(|e| anyhow::anyhow!("residency poisoned: {e}"))?;
            let parked = self.best_parked(&res, prompt, reusable[from], want_k);
            if parked.is_some() || from != 1 || reusable[1] == 0 {
                if let Some(w) = slots[1].as_ref() {
                    self.park_working(&mut res, w);
                }
            }
            if let Some(i) = self.best_parked(&res, prompt, reusable[from], want_k).filter(|_| parked.is_some()) {
                let working = slots[1].take();
                drop(slots);
                let mut sess = match working.filter(|s| s.spec_k() == want_k && s.state.max_len >= max_len) {
                    Some(s) => s,
                    None => self.new_session(max_len, k)?,
                };
                let tier = res.lru.get(i).value.tier();
                let t0 = std::time::Instant::now();
                return match self.install_parked(&res, i, &mut sess) {
                    Ok(()) => {
                        res.lru.touch(i);
                        eprintln!("[qwen4_exp] restored a parked {}-token session prefix from {tier:?} in {:.3}s", sess.reusable(prompt), t0.elapsed().as_secs_f64());
                        Ok((sess, Some(tier)))
                    }
                    Err(e) => {
                        eprintln!("[qwen4_exp] parked session restore refused, prefilling cold: {e:#}");
                        crate::session_lru::METRICS.restore_failures.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                        let gone = res.lru.remove(i);
                        if let residency::Parked::Disk { path, .. } = &gone.value {
                            let _ = std::fs::remove_file(path);
                        }
                        sess.prefix.clear();
                        sess.ckpts.clear();
                        Ok((sess, None))
                    }
                };
            }
        }
        let source = (reusable[from] > 0).then_some(Tier::InPlace);
        if from != home {
            drop(slots[home].take());
        }
        let cached = slots[from].take();
        drop(slots);
        if let Some(s) = cached.filter(|s| s.spec_k() == want_k) {
            if s.state.max_len >= max_len {
                return Ok((s, source));
            }
            if s.reusable(prompt) > 0 {
                return Ok((self.grow_session(s, max_len, k)?, source));
            }
        }
        Ok((self.new_session(max_len, k)?, None))
    }

    pub fn put_session(&self, prompt: &[u32], s: Qwen4ExpSession) {
        if s.lora_tag.is_some() {
            let _ = ADAPTER_SESSIONS_NEVER_ENTER_THE_BASE_SESSION_CACHE_OR_ITS_RESIDENCY_TIERS;
            return;
        }
        let home = Self::home_slot(prompt);
        if home == 1 {
            if let Ok(mut res) = self.residency.lock() {
                self.register_working(&mut res, &s);
            }
        }
        if let Ok(mut slot) = self.sessions[home].lock() {
            *slot = Some(s);
        }
    }

    pub fn take_adapter_session(&self, tag: &str, max_len: usize) -> Result<Qwen4ExpSession> {
        let keep = std::env::var(NV_Q4EXP_LORA_SESSION_CACHE_1_KEEPS_ONE_ADAPTER_SESSION_REUSED_ONLY_UNDER_THE_SAME_ADAPTER_ID_AND_VERSION).as_deref() == Ok("1");
        let cached = self.adapter_session.lock().map_err(|e| anyhow::anyhow!("adapter session poisoned: {e}"))?.take();
        if let Some(s) = cached.filter(|s| keep && s.lora_tag.as_deref() == Some(tag) && s.state.max_len >= max_len) {
            return Ok(s);
        }
        let mut s = self.new_session(max_len, 0)?;
        s.lora_tag = Some(tag.to_string());
        Ok(s)
    }

    pub fn put_adapter_session(&self, s: Qwen4ExpSession) {
        let keep = std::env::var(NV_Q4EXP_LORA_SESSION_CACHE_1_KEEPS_ONE_ADAPTER_SESSION_REUSED_ONLY_UNDER_THE_SAME_ADAPTER_ID_AND_VERSION).as_deref() == Ok("1");
        if keep && s.lora_tag.is_some() {
            if let Ok(mut slot) = self.adapter_session.lock() {
                *slot = Some(s);
            }
        }
    }

    pub fn drop_adapter_session(&self) {
        if let Ok(mut slot) = self.adapter_session.lock() {
            *slot = None;
        }
    }

    fn grow_session(&self, old: Qwen4ExpSession, max_len: usize, k: usize) -> Result<Qwen4ExpSession> {
        let t0 = std::time::Instant::now();
        let keep = old.ckpts.last().map_or(0, |c| c.pos);
        let r = self.cfg.indexer_compress_ratio;
        let mut host = Vec::new();
        for slot in old.state.attn.iter().chain(old.state.mtp_attn.as_ref()) {
            for (t, n) in [(&slot.k, keep), (&slot.v, keep), (&slot.raw_idx_k, keep), (&slot.cmp_k, keep / r)] {
                host.push(t.narrow(0, 0, n)?.to_device(&Device::Cpu)?);
            }
        }
        let Qwen4ExpSession { state, graph, mtp, prefix, ckpts, .. } = old;
        drop((state, graph, mtp));
        let mut s = self.new_session(max_len, k)?;
        let slots: Vec<&AttnSlot> = s.state.attn.iter().chain(s.state.mtp_attn.as_ref()).collect();
        ensure!(host.len() == 4 * slots.len(), "grown session has {} attention slots, the old one {}", slots.len(), host.len() / 4);
        for (slot, rows) in slots.into_iter().zip(host.chunks(4)) {
            for (t, h) in [&slot.k, &slot.v, &slot.raw_idx_k, &slot.cmp_k].into_iter().zip(rows) {
                t.slice_set(&h.to_device(&self.device)?, 0, 0)?;
            }
        }
        s.prefix = prefix;
        s.ckpts = ckpts;
        stream_of(&self.device)?.synchronize()?;
        eprintln!("[qwen4_exp] session grown to {max_len} tokens keeping {keep} resident prefix tokens in {:.2}s", t0.elapsed().as_secs_f64());
        Ok(s)
    }

    fn save_prefix_ckpt(&self, sess: &mut Qwen4ExpSession, pos: usize) -> Result<()> {
        if sess.ckpt_cap == 0 {
            return Ok(());
        }
        let st = &sess.state;
        let ck = if sess.ckpts.len() < sess.ckpt_cap {
            PrefixCkpt { pos, gdn: st.gdn.iter().map(|g| g.deep_clone()).collect::<Result<_>>()?, ple_conv: st.ple_conv.copy()? }
        } else {
            let mut ck = sess.ckpts.remove(0);
            let pairs: Vec<(&LinAttnState, &LinAttnState)> = ck.gdn.iter().zip(&st.gdn).collect();
            nv_layers::linear_attn::batched_rollback_copy(&pairs)?;
            ck.ple_conv.slice_set(&st.ple_conv, 0, 0)?;
            ck.pos = pos;
            ck
        };
        sess.ckpts.push(ck);
        Ok(())
    }

    fn restore_prefix_ckpt(&self, sess: &mut Qwen4ExpSession, prompt: &[u32]) -> Result<()> {
        let ck = sess.ckpts.last().ok_or_else(|| anyhow::anyhow!("no prefix checkpoint to restore"))?;
        let ctx = self.cfg.ngram_size - 1;
        ensure!(ck.pos >= ctx && ck.pos < prompt.len(), "prefix checkpoint at {} outside the {}-token prompt", ck.pos, prompt.len());
        let st = &mut sess.state;
        let pairs: Vec<(&LinAttnState, &LinAttnState)> = st.gdn.iter().zip(&ck.gdn).collect();
        nv_layers::linear_attn::batched_rollback_copy(&pairs)?;
        st.ple_conv.slice_set(&ck.ple_conv, 0, 0)?;
        st.ple_history = prompt[ck.pos - ctx..ck.pos].to_vec();
        stream_of(&self.device)?.memcpy_htod(&st.ple_history, &mut st.ple_history_dev)?;
        st.pos = ck.pos;
        Ok(())
    }

    pub fn session_prefill(&self, sess: &mut Qwen4ExpSession, prompt: &[u32]) -> Result<(Vec<f32>, Tensor)> {
        ensure!(!prompt.is_empty(), "qwen4_exp session: empty prompt");
        let l = prompt.len();
        ensure!(l < sess.state.max_len, "prompt of {l} tokens does not fit the {}-token state", sess.state.max_len);
        let stream = stream_of(&self.device)?;
        let key = self.mm_prefix_key(&sess.state, prompt);
        let start = sess.reusable(&key);
        sess.prefix.clear();
        sess.ckpts.retain(|c| c.pos <= start);
        if start > 0 {
            self.restore_prefix_ckpt(sess, prompt)?;
        } else {
            sess.state.reset(&self.cfg, &self.device)?;
        }
        let chunk = PREFILL_CHUNK_2048_IS_THE_PREFIX_CHECKPOINT_GRID_SO_A_RESUMED_PREFILL_CHUNKS_EXACTLY_AS_A_COLD_ONE;
        let mut off = start;
        let mut last = None;
        while off < l {
            let c = (l - off).min(chunk);
            let (logits, hc) = self.forward_rows(&prompt[off..off + c], &mut sess.state)?;
            if let (Some(mtp), Some(_)) = (self.mtp.as_ref(), sess.mtp.as_ref()) {
                let e = (off + c).min(l - 1);
                if e > off {
                    let rows = e - off;
                    let toks: CudaSlice<u32> = stream.clone_htod(&prompt[off + 1..e + 1])?;
                    let positions: Vec<i32> = (off..e).map(|p| p as i32).collect();
                    let positions = Tensor::from_vec(positions, (rows,), &self.device)?;
                    let hidden = hc.narrow(0, 0, rows)?;
                    let staged = self.staged_moe_ctx_for_rows(rows)?;
                    let ctx = if rows == 1 { Some(&self.decode_ctx) } else { staged.as_deref() };
                    let _ = self.mtp_forward(mtp, &hidden, &toks, rows, off, &positions, None, ctx, &mut sess.state)?;
                }
            }
            last = Some((logits, hc.narrow(0, c - 1, 1)?.contiguous()?));
            off += c;
            if c == chunk && off < l {
                self.save_prefix_ckpt(sess, off)?;
            }
        }
        sess.prefix = key;
        sess.state.mm.prefill_done();
        let (logits, hidden) = last.ok_or_else(|| anyhow::anyhow!("prefill resumed at {start} past the {l}-token prompt"))?;
        Ok((logits.flatten_all()?.to_vec1::<f32>()?, hidden))
    }

    fn verify_run(&self, state: &mut Qwen4ExpState, g: &mut Qwen4ExpDecodeGraph, rows: usize, captured: bool) -> Result<()> {
        ensure!(rows >= 2 && rows <= g.bufs.rows_cap, "verify rows {rows} outside 2..={}", g.bufs.rows_cap);
        let Qwen4ExpDecodeGraph { runner, bufs, verify_ctx, .. } = g;
        let ctx = &verify_ctx[rows - 2];
        if !captured {
            let stream = runner.stream().clone();
            return nv_layers::cuda_stream::with_stream(stream, || self.step_captured(state, bufs, rows, ctx));
        }
        runner
            .run(DECODE_GRAPH_SHAPE_TOKEN_ONE_GRAPH_PER_STATE_BECAUSE_EVERY_BUFFER_ADDRESS_IS_BAKED + rows as u64, |s| {
                nv_layers::cuda_stream::with_stream(s.clone(), || self.step_captured(state, bufs, rows, ctx))
            })
            .context("qwen4_exp verify graph")
    }

    fn mtp_step_captured(&self, mtp: &MtpHead, state: &mut Qwen4ExpState, g: &DecodeBufs, mb: &mut MtpBufs, rows: usize, moe_ctx: &Mutex<GroupedDecodeContext>) -> Result<()> {
        let cfg = &self.cfg;
        let dev = cuda_dev(&self.device)?;
        let stream = stream_of(&self.device)?;
        {
            stream.memcpy_htod(&mb.host_tok[..rows], &mut mb.tok_buf.slice_mut(0..rows))?;
            stream.memcpy_htod(&mb.host_pos[..], &mut mb.pos_buf)?;
        }
        let positions = wrap_i32(&mb.pos_buf, &dev, 1)?;
        let tok_buf = mb.tok_buf.try_clone()?;
        let hidden = g.hidden_buf.as_ref().ok_or_else(|| anyhow::anyhow!("mtp step needs the hidden rows buffer"))?.narrow(0, 0, rows)?;
        let (hc, logits) = self.mtp_forward(mtp, &hidden, &tok_buf, rows, 0, &positions, Some(&mb.pos_buf), Some(moe_ctx), state)?;
        let hh = cfg.hc_hidden();
        let hc = hc.contiguous()?;
        dtod_raw(cuda_ptr::<bf16>(&mb.mh_buf, &stream)?, cuda_ptr::<bf16>(&hc, &stream)? + ((rows - 1) * hh * 2) as u64, hh * 2, &stream)?;
        let v = cfg.vocab_size;
        dtod_raw(slice_ptr_mut(&mut mb.logits_buf, &stream), cuda_ptr::<f32>(&logits, &stream)? + ((rows - 1) * v * 4) as u64, v * 4, &stream)?;
        Ok(())
    }

    fn mtp_step_run(&self, state: &mut Qwen4ExpState, g: &mut Qwen4ExpDecodeGraph, mb: &mut MtpBufs, rows: usize, captured: bool) -> Result<()> {
        let mtp = self.mtp.as_ref().ok_or_else(|| anyhow::anyhow!("no mtp head loaded"))?;
        let Qwen4ExpDecodeGraph { runner, bufs, verify_ctx, .. } = g;
        let ctx: &Mutex<GroupedDecodeContext> = if rows == 1 { &self.decode_ctx } else { &verify_ctx[rows - 2] };
        let stream = runner.stream().clone();
        if !captured {
            return nv_layers::cuda_stream::with_stream(stream, || self.mtp_step_captured(mtp, state, bufs, mb, rows, ctx));
        }
        runner
            .run(MTP_GRAPH_SHAPE_TOKEN_BASE_0X5E0_PLUS_ROWS + rows as u64, |s| {
                nv_layers::cuda_stream::with_stream(s.clone(), || self.mtp_step_captured(mtp, state, bufs, mb, rows, ctx))
            })
            .context("qwen4_exp mtp graph")
    }

    pub fn session_begin(&self, sess: &mut Qwen4ExpSession, prompt: &[u32]) -> Result<u32> {
        self.session_begin_with(sess, prompt, &mut GreedyPick)
    }

    pub fn session_begin_with(&self, sess: &mut Qwen4ExpSession, prompt: &[u32], pick: &mut dyn SpecPick) -> Result<u32> {
        let l = prompt.len();
        ensure!(l + sess.k < sess.state.max_len, "prompt of {l} tokens + {} draft rows exceeds the {}-token state", sess.k, sess.state.max_len);
        let stream = stream_of(&self.device)?;
        sess.stats = SpecStats::default();
        let (row, last_hidden) = self.session_prefill(sess, prompt)?;
        let anchor = pick.target(&[], &row)?;
        if let Some(hbuf) = sess.graph.bufs.hidden_buf.as_ref() {
            let hh = self.cfg.hc_hidden();
            dtod_raw(cuda_ptr::<bf16>(hbuf, &stream)?, cuda_ptr::<bf16>(&last_hidden, &stream)?, hh * 2, &stream)?;
        }
        sess.anchor = anchor;
        sess.pending = vec![anchor];
        stream.synchronize()?;
        Ok(anchor)
    }

    pub fn session_round(&self, sess: &mut Qwen4ExpSession) -> Result<Vec<u32>> {
        self.session_round_with(sess, &mut GreedyPick)
    }

    pub fn session_round_with(&self, sess: &mut Qwen4ExpSession, pick: &mut dyn SpecPick) -> Result<Vec<u32>> {
        let cfg = &self.cfg;
        let k = sess.k;
        ensure!(k >= 1 && sess.mtp.is_some(), "session_round needs a speculative session");
        let stream = stream_of(&self.device)?;
        let l = sess.state.pos;
        ensure!(l + k < sess.state.max_len, "verify overflow: {l} + {} rows > {}", k + 1, sess.state.max_len);
        let t0 = std::time::Instant::now();
        let c = sess.pending.len();
        ensure!(c >= 1 && c <= k + 1 && c <= l, "pending rows {c} out of range (l={l}, k={k})");
        let Qwen4ExpSession { state, graph, mtp, pending, anchor, stats, .. } = sess;
        let mb = mtp.as_mut().unwrap();
        mb.host_tok[..c].copy_from_slice(pending);
        mb.host_pos[0] = (l - c) as i32;
        self.mtp_step_run(state, graph, mb, c, true)?;
        stream.synchronize()?;
        let row: Vec<f32> = stream.clone_dtoh(&mb.logits_buf)?;
        let mut drafts = vec![pick.draft(&[], &row)];
        let hh = cfg.hc_hidden();
        for j in 1..k {
            let hbuf = graph.bufs.hidden_buf.as_ref().unwrap();
            dtod_raw(cuda_ptr::<bf16>(hbuf, &stream)?, cuda_ptr::<bf16>(&mb.mh_buf, &stream)?, hh * 2, &stream)?;
            mb.host_tok[0] = drafts[j - 1];
            mb.host_pos[0] = (l - 1 + j) as i32;
            self.mtp_step_run(state, graph, mb, 1, true)?;
            stream.synchronize()?;
            let row: Vec<f32> = stream.clone_dtoh(&mb.logits_buf)?;
            let d = pick.draft(&drafts, &row);
            drafts.push(d);
        }
        let draft_dt = t0.elapsed().as_secs_f64();
        let t1 = std::time::Instant::now();
        let rows = k + 1;
        graph.bufs.host_tok[0] = *anchor;
        graph.bufs.host_tok[1..rows].copy_from_slice(&drafts);
        graph.bufs.host_pos[0] = l as i32;
        self.verify_run(state, graph, rows, true)?;
        stream.synchronize()?;
        let v = cfg.vocab_size;
        let flat: Vec<f32> = stream.clone_dtoh(&graph.bufs.logits_buf.slice(0..rows * v))?;
        let verify_dt = t1.elapsed().as_secs_f64();
        let t2 = std::time::Instant::now();
        let (accepted, bonus) = accept_drafts(pick, &drafts, &flat, v)?;
        let consumed = accepted + 1;
        let mut emitted: Vec<u32> = drafts[..accepted].to_vec();
        emitted.push(bonus);
        let block: Vec<u32> = std::iter::once(*anchor).chain(drafts.iter().copied()).collect();
        self.commit_rows(state, &block[..consumed])?;
        *pending = block[1..consumed].to_vec();
        pending.push(bonus);
        *anchor = bonus;
        stream.synchronize()?;
        stats.rounds += 1;
        stats.drafted += k;
        stats.accepted += accepted;
        stats.emitted += emitted.len();
        stats.draft_ms += 1000.0 * draft_dt;
        stats.verify_ms += 1000.0 * verify_dt;
        stats.commit_ms += 1000.0 * t2.elapsed().as_secs_f64();
        Ok(emitted)
    }

    fn commit_rows(&self, state: &mut Qwen4ExpState, tokens: &[u32]) -> Result<()> {
        let cfg = &self.cfg;
        let consumed = tokens.len();
        ensure!(consumed >= 1, "commit_rows: nothing consumed");
        let stream = stream_of(&self.device)?;
        let pairs: Vec<(&LinAttnState, &LinAttnState)> = state
            .gdn
            .iter()
            .zip(state.gdn_ckpts.iter())
            .map(|(live, ck)| (live, &ck[consumed - 1]))
            .collect();
        nv_layers::linear_attn::batched_rollback_copy(&pairs)?;
        if let (Some(ple), Some(slab)) = (self.ple_layer(), state.ple_gvn.as_ref()) {
            ple.advance_window(&state.ple_conv, slab, consumed, cfg)?;
        }
        self.push_history_host(state, tokens);
        stream.memcpy_htod(&state.ple_history, &mut state.ple_history_dev)?;
        state.pos += consumed;
        Ok(())
    }
}

struct PhaseProf {
    on: bool,
    stream: Option<Arc<CudaStream>>,
    last: std::time::Instant,
    acc: std::collections::BTreeMap<&'static str, f64>,
}

impl PhaseProf {
    fn disabled() -> Self {
        Self { on: false, stream: None, last: std::time::Instant::now(), acc: std::collections::BTreeMap::new() }
    }

    fn begin(device: &Device) -> Self {
        let on = std::env::var("NV_Q4EXP_PROF").as_deref() == Ok("1");
        let stream = if on { stream_of(device).ok() } else { None };
        if let Some(s) = &stream {
            let _ = s.synchronize();
        }
        Self {
            on,
            stream,
            last: std::time::Instant::now(),
            acc: std::collections::BTreeMap::new(),
        }
    }

    fn lap(&mut self, label: &'static str) {
        if !self.on {
            return;
        }
        if let Some(s) = &self.stream {
            let _ = s.synchronize();
        }
        let now = std::time::Instant::now();
        *self.acc.entry(label).or_default() += (now - self.last).as_secs_f64() * 1e3;
        self.last = now;
    }

    fn report(&self, t: usize) {
        if !self.on {
            return;
        }
        let total: f64 = self.acc.values().sum();
        let mut rows: Vec<(&&str, &f64)> = self.acc.iter().collect();
        rows.sort_by(|a, b| b.1.partial_cmp(a.1).unwrap());
        let parts: Vec<String> = rows.iter().map(|(k, v)| format!("{k}={v:.3}")).collect();
        eprintln!("[q4exp-prof] t={t} total_ms={total:.3} {}", parts.join(" "));
    }
}

fn trace_row(label: &str, t: &Tensor) -> Result<()> {
    let rows = t.dim(0)?;
    let row = t.narrow(0, rows - 1, 1)?.flatten_all()?.to_dtype(DType::F32)?.to_vec1::<f32>()?;
    let n = row.len() as f32;
    let mean = row.iter().sum::<f32>() / n;
    let std = (row.iter().map(|x| (x - mean) * (x - mean)).sum::<f32>() / n).sqrt();
    let amax = row.iter().fold(0f32, |m, x| m.max(x.abs()));
    let head: Vec<String> = row.iter().take(8).map(|x| format!("{x:.5}")).collect();
    eprintln!("[trace] {label} mean={mean:.6} std={std:.6} absmax={amax:.4} head={}", head.join(","));
    Ok(())
}
