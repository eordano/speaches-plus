pub mod prefix;

use anyhow::{bail, ensure, Result};
use candle_core::{CudaStorage, DType, Device, Storage, Tensor};
use cudarc::driver::{CudaSlice, DevicePtr, DevicePtrMut};
use half::bf16;
use std::ffi::c_void;

use crate::gemma4::{Gemma4Cache, LayerType};
use crate::laguna::{ring_append_launch, shift_rows_launch, LagunaConfig, SLIDING_COMPACT_SLACK};
use crate::laguna_step_graph::XqaFp8Plan;
use candle_core::op::BackpropOp;
use std::env;
use std::sync::Arc;

const SMEM_LIMIT_BYTES: usize = 48 * 1024;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum LagunaFp8ScalePolicy {
    PerTokenHead,
    FixedOne,
    PerTokenHeadInt8,
}

impl LagunaFp8ScalePolicy {
    pub(crate) fn from_env() -> Result<Self> {
        let int8 = match env::var("NV_LAGUNA_KV8_INT8") {
            Err(std::env::VarError::NotPresent) => false,
            Ok(value) if value == "0" => false,
            Ok(value) if value == "1" => true,
            other => bail!("NV_LAGUNA_KV8_INT8 must be unset, 0 or 1, got {other:?}"),
        };
        match env::var("NV_LAGUNA_FP8_UNIT_SCALE") {
            Err(std::env::VarError::NotPresent) if int8 => Ok(Self::PerTokenHeadInt8),
            Ok(value) if value == "0" && int8 => Ok(Self::PerTokenHeadInt8),
            Err(std::env::VarError::NotPresent) => Ok(Self::PerTokenHead),
            Ok(value) if value == "0" => Ok(Self::PerTokenHead),
            Ok(value) if value == "1" && int8 => bail!("NV_LAGUNA_KV8_INT8=1 and NV_LAGUNA_FP8_UNIT_SCALE=1 are exclusive: int8 K/V carries per-token-per-head scales"),
            Ok(value) if value == "1" => Ok(Self::FixedOne),
            other => bail!("NV_LAGUNA_FP8_UNIT_SCALE must be unset, 0 or 1, got {other:?}"),
        }
    }

    pub(crate) fn require_bf16_cache_compatible() -> Result<()> {
        ensure!(
            Self::from_env()? == Self::PerTokenHead,
            "NV_LAGUNA_FP8_UNIT_SCALE=1 and NV_LAGUNA_KV8_INT8=1 require the 8-bit KV cache (NV_LAGUNA_FP8_KV=1); refusing a BF16 cache"
        );
        Ok(())
    }

    pub(crate) fn kv_elem_code_matching_the_kernels_0_bf16_1_e4m3_2_int8(&self) -> i32 {
        match self {
            Self::PerTokenHeadInt8 => 2,
            Self::PerTokenHead | Self::FixedOne => 1,
        }
    }
}

pub(crate) enum Fp8LayerSlot {
    Full {
        k_fp8: CudaSlice<u8>,
        v_fp8: CudaSlice<u8>,
        k_scales: CudaSlice<f32>,
        v_scales: CudaSlice<f32>,
    },
    Bf16 {
        k: Tensor,
        v: Tensor,
    },
}

pub struct LagunaKvCacheFp8 {
    scale_policy: LagunaFp8ScalePolicy,
    layers: Vec<Fp8LayerSlot>,
    layer_windows: Vec<Option<usize>>,
    layer_stored: Vec<usize>,
    n_kv: usize,
    head_dim: usize,
    current_len: usize,
    max_seq_len: usize,
    device: Device,
    dev: candle_core::CudaDevice,
    meta_dev: CudaSlice<i32>,
    host_meta: Box<[i32; 4]>,
    n_total_dev: CudaSlice<i32>,
    host_n_total: Box<[i32; 1]>,
    s_stored: usize,
    full_stored: usize,
    s_cap: usize,
    s_window: usize,
    scores_scratch: Option<CudaSlice<f32>>,
    scratch_heads: usize,
    prefill_attn: Option<crate::laguna_prefill_attn::LagunaPrefillFp8Attention>,
    batch_decode: Option<FixedOneBatchAttention>,
}

struct FixedOneBatchVerifyAttention {
    xqa: XqaFp8Plan,
    scratch: CudaSlice<f32>,
    fan_in: CudaSlice<u32>,
    rows_per_block: i32,
}

struct FixedOneBatchAttention {
    verify: Option<FixedOneBatchVerifyAttention>,
    xqa: XqaFp8Plan,
    scratch: CudaSlice<f32>,
    fan_in: CudaSlice<u32>,
}

impl LagunaKvCacheFp8 {
    pub(crate) fn enable_fixed_one_batch_attention(&mut self, config: &LagunaConfig) -> Result<()> {
        ensure!(
            self.batch_decode.is_none(),
            "batch attention already initialized"
        );
        ensure!(
            self.scale_policy == LagunaFp8ScalePolicy::FixedOne
                && self.n_kv == 8
                && self.head_dim == 128
                && self.s_window == 512,
            "batch attention requires immutable FixedOne KV8/head128/window512"
        );
        ensure!(
            config.layer_types.len() == self.layers.len()
                && config
                    .num_attention_heads_per_layer
                    .as_ref()
                    .is_none_or(|v| v.len() == self.layers.len()),
            "batch attention layer metadata shape mismatch"
        );
        for (i, kind) in config.layer_types.iter().enumerate() {
            let heads = config
                .num_attention_heads_per_layer
                .as_ref()
                .map(|v| v[i])
                .unwrap_or(config.num_attention_heads);
            ensure!(
                match kind {
                    LayerType::FullAttention =>
                        heads == 48 && matches!(self.layers[i], Fp8LayerSlot::Full { .. }),
                    LayerType::SlidingAttention =>
                        matches!(heads, 48 | 64)
                            && matches!(self.layers[i], Fp8LayerSlot::Bf16 { .. }),
                },
                "unsupported batch attention layer {i} heads={heads}"
            );
        }
        let xqa = XqaFp8Plan::new_for(
            1,
            self.max_seq_len,
            8,
            48,
            128,
            self.scale_policy,
            "b2-physical-lane-m1",
        )?
        .ok_or_else(|| anyhow::anyhow!("batch attention requires NV_LAGUNA_XQA_FP8=1"))?;
        let s = nv_layers::cuda_stream::current_stream(&self.dev);
        let elems = nv_kernels::cuda::laguna_flash_decode_gqa_scratch_elems(8);
        ensure!(elems > 0, "invalid batch sliding scratch extent");
        self.batch_decode = Some(FixedOneBatchAttention {
            verify: None,
            xqa,
            scratch: s.alloc_zeros::<f32>(elems as usize)?,
            fan_in: s.alloc_zeros::<u32>(8)?,
        });
        Ok(())
    }

    pub(crate) fn enable_fixed_one_batch_verify_attention(&mut self) -> Result<()> {
        let state = self
            .batch_decode
            .as_mut()
            .ok_or_else(|| anyhow::anyhow!("M3 requires initialized M1 attention"))?;
        ensure!(
            state.verify.is_none(),
            "M3 batch attention already initialized"
        );
        let rows_per_block = match env::var("NV_LAGUNA_VERIFY_SLIDING_ROWS").as_deref() {
            Ok("1") => 1,
            Ok("2") => 2,
            Ok("3") => 3,
            _ => bail!("m2-spec requires NV_LAGUNA_VERIFY_SLIDING_ROWS=1, 2 or 3"),
        };
        let xqa = XqaFp8Plan::new_for(
            3,
            self.max_seq_len,
            8,
            48,
            128,
            self.scale_policy,
            "b2-spec-physical-lane-m3",
        )?
        .ok_or_else(|| anyhow::anyhow!("m2-spec requires an M3 XQA artifact"))?;
        let stream = nv_layers::cuda_stream::current_stream(&self.dev);
        state.verify = Some(FixedOneBatchVerifyAttention {
            xqa,
            scratch: stream.alloc_zeros::<f32>(3 * 64 * 8 * (128 + 2))?,
            fan_in: stream.alloc_zeros::<u32>(3 * 8)?,
            rows_per_block,
        });
        Ok(())
    }

    pub(crate) fn batch_verify_attention_status_after_sync(&self, rows: usize) -> Result<()> {
        if rows == 1 {
            return self.batch_attention_status_after_sync();
        }
        ensure!(rows == 3, "unsupported batch verify rows");
        let state = self
            .batch_decode
            .as_ref()
            .and_then(|s| s.verify.as_ref())
            .ok_or_else(|| anyhow::anyhow!("M3 batch attention not initialized"))?;
        state
            .xqa
            .status_after_sync(&nv_layers::cuda_stream::current_stream(&self.dev))
    }

    pub(crate) fn prepared_batch_verify_attention(
        &mut self,
        layer: usize,
        q: &Tensor,
        heads: usize,
        window: Option<usize>,
        scale: f32,
        rows: usize,
    ) -> Result<Tensor> {
        if rows == 1 {
            return self.batch_attention(layer, q, heads, window, scale);
        }
        ensure!(
            rows == 3
                && q.dtype() == DType::BF16
                && q.is_contiguous()
                && q.elem_count() == rows * heads * 128
                && self.layer_windows.get(layer) == Some(&window)
                && scale.is_finite()
                && (scale - (128f32).powf(-0.5)).abs() <= 1e-7,
            "batch verify attention shape/window/scale mismatch"
        );
        self.batch_attention_rows(layer, q, heads, window, scale, rows)
    }

    pub(crate) fn require_initialized_batch_prefix(&self) -> Result<()> {
        ensure!(
            self.full_stored == self.current_len
                && self.s_stored <= self.current_len
                && self
                    .layer_stored
                    .iter()
                    .zip(&self.layer_windows)
                    .all(|(&n, w)| n
                        == if w.is_some() {
                            self.s_stored
                        } else {
                            self.full_stored
                        }),
            "prepared B2 cache prefix is not fully initialized"
        );
        Ok(())
    }

    pub(crate) fn batch_attention_status_after_sync(&self) -> Result<()> {
        let s = nv_layers::cuda_stream::current_stream(&self.dev);
        self.batch_decode
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("batch attention not initialized"))?
            .xqa
            .status_after_sync(&s)
    }

    fn batch_attention(
        &mut self,
        layer: usize,
        q: &Tensor,
        heads: usize,
        window: Option<usize>,
        scale: f32,
    ) -> Result<Tensor> {
        ensure!(
            q.dtype() == DType::BF16
                && q.is_contiguous()
                && q.elem_count() == heads * 128
                && self.layer_windows[layer] == window
                && scale.is_finite()
                && (scale - (128f32).powf(-0.5)).abs() <= 1e-7,
            "batch attention shape/window/scale mismatch"
        );
        self.batch_attention_rows(layer, q, heads, window, scale, 1)
    }

    fn batch_attention_rows(
        &mut self,
        layer: usize,
        q: &Tensor,
        heads: usize,
        window: Option<usize>,
        scale: f32,
        rows: usize,
    ) -> Result<Tensor> {
        let s = nv_layers::cuda_stream::current_stream(&self.dev);
        let mut out = s.alloc_zeros::<bf16>(rows * heads * 128)?;
        {
            let (storage, layout) = q.storage_and_layout();
            let cuda = match &*storage {
                Storage::Cuda(v) => v,
                _ => bail!(
                    "{}",
                    if rows == 1 {
                        "batch attention requires CUDA Q"
                    } else {
                        "batch verify requires CUDA Q"
                    }
                ),
            };
            let view = cuda.as_cuda_slice::<bf16>()?.slice(layout.start_offset()..);
            let (qp, _qg) = view.device_ptr(&s);
            let (op, _og) = out.device_ptr_mut(&s);
            let (xqa, scratch, fan_in, rows_per_block) = if rows == 1 {
                let state = self
                    .batch_decode
                    .as_mut()
                    .ok_or_else(|| anyhow::anyhow!("batch attention not initialized"))?;
                (&state.xqa, &mut state.scratch, &mut state.fan_in, 0)
            } else {
                let state = self
                    .batch_decode
                    .as_mut()
                    .and_then(|s| s.verify.as_mut())
                    .ok_or_else(|| anyhow::anyhow!("M3 batch attention not initialized"))?;
                (
                    &state.xqa,
                    &mut state.scratch,
                    &mut state.fan_in,
                    state.rows_per_block,
                )
            };
            match &self.layers[layer] {
                Fp8LayerSlot::Full { k_fp8, v_fp8, .. } => {
                    ensure!(
                        window.is_none() && heads == 48,
                        "full {} attention shape mismatch",
                        if rows == 1 { "batch" } else { "M3" }
                    );
                    let (kp, _kg) = k_fp8.device_ptr(&s);
                    let (vp, _vg) = v_fp8.device_ptr(&s);
                    let (mp, _mg) = self.meta_dev.device_ptr(&s);
                    xqa.launch(&s, qp, kp, vp, op, mp)?;
                }
                Fp8LayerSlot::Bf16 { k, v } => {
                    ensure!(
                        window == Some(512) && matches!(heads, 48 | 64),
                        "sliding {} attention shape mismatch",
                        if rows == 1 { "batch" } else { "M3" }
                    );
                    let (ks, kl) = k.storage_and_layout();
                    let (vs, vl) = v.storage_and_layout();
                    let (kc, vc) = match (&*ks, &*vs) {
                        (Storage::Cuda(kc), Storage::Cuda(vc)) => (kc, vc),
                        _ => bail!(
                            "{}",
                            if rows == 1 {
                                "batch sliding cache requires CUDA"
                            } else {
                                "sliding M3 cache requires CUDA"
                            }
                        ),
                    };
                    let kv = kc.as_cuda_slice::<bf16>()?.slice(kl.start_offset()..);
                    let vv = vc.as_cuda_slice::<bf16>()?.slice(vl.start_offset()..);
                    let mv = self.meta_dev.slice(3..4);
                    let (kp, _kg) = kv.device_ptr(&s);
                    let (vp, _vg) = vv.device_ptr(&s);
                    let (mp, _mg) = mv.device_ptr(&s);
                    let (sp, _sg) = scratch.device_ptr_mut(&s);
                    let (fp, _fg) = fan_in.device_ptr_mut(&s);
                    if rows == 1 {
                        let rc = unsafe {
                            nv_kernels::cuda::laguna_flash_decode_gqa(
                                s.cu_stream() as *mut c_void,
                                qp as *const u16,
                                kp as *const u16,
                                vp as *const u16,
                                op as *mut u16,
                                mp as *const i32,
                                0,
                                sp as *mut f32,
                                fp as *mut u32,
                                heads as i32,
                                8,
                                128,
                                512,
                                scale,
                            )
                        };
                        ensure!(rc == 0, "batch sliding GQA rc={rc}");
                    } else {
                        let rc = unsafe {
                            nv_kernels::cuda::laguna_flash_decode_verify_window(
                                s.cu_stream() as *mut c_void,
                                qp as *const u16,
                                kp as *const u16,
                                vp as *const u16,
                                op as *mut u16,
                                mp as *const i32,
                                sp as *mut f32,
                                fp as *mut u32,
                                heads as i32,
                                8,
                                128,
                                3,
                                i32::try_from(self.s_cap)?,
                                512,
                                rows_per_block,
                                scale,
                            )
                        };
                        ensure!(rc == 0, "sliding M3 attention rc={rc}");
                    }
                }
            }
        }
        Ok(Tensor::from_storage(
            Storage::Cuda(CudaStorage::wrap_cuda_slice(out, self.dev.clone())),
            (1, rows, heads, 128),
            BackpropOp::none(),
            false,
        ))
    }

    pub fn max_seq_len_for_fp8_decode(head_dim: usize) -> usize {
        let n_warps = head_dim / 32;
        SMEM_LIMIT_BYTES / std::mem::size_of::<f32>() - head_dim - n_warps
    }

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
        Self::new_with_scale_policy(
            config,
            max_seq_len,
            device,
            dtype,
            prefill_chunk,
            LagunaFp8ScalePolicy::from_env()?,
        )
    }

    pub fn new_fixed_one_with_prefill_chunk(
        config: &LagunaConfig,
        max_seq_len: usize,
        device: &Device,
        prefill_chunk: usize,
    ) -> Result<Self> {
        ensure!(
            !nv_config::env_arm_present_at_all_reread_every_call("NV_LAGUNA_FP8_BF16_FULL"),
            "explicit FixedOne construction refuses BF16 full-layer overrides"
        );
        ensure!(
            max_seq_len > 0
                && max_seq_len <= 262144
                && config.num_key_value_heads == 8
                && config.head_dim == 128,
            "explicit FixedOne handoff construction requires KV8/head128 and capacity 1..=262144"
        );
        Self::new_with_scale_policy(
            config,
            max_seq_len,
            device,
            DType::BF16,
            prefill_chunk,
            LagunaFp8ScalePolicy::FixedOne,
        )
    }

    fn new_with_scale_policy(
        config: &LagunaConfig,
        max_seq_len: usize,
        device: &Device,
        dtype: DType,
        prefill_chunk: usize,
        scale_policy: LagunaFp8ScalePolicy,
    ) -> Result<Self> {
        ensure!(
            matches!(prefill_chunk, 256 | 512 | 1024 | 2048 | 4096 | 8192),
            "invalid Laguna FP8 prefill capacity {prefill_chunk}"
        );
        let Device::Cuda(dev) = device.clone() else {
            bail!("LagunaKvCacheFp8 requires a CUDA device")
        };
        if dtype != DType::BF16 {
            bail!("LagunaKvCacheFp8 requires a BF16 model, got {dtype:?}");
        }
        let n_kv = config.num_key_value_heads;
        let head_dim = config.head_dim;
        let fp8_cap = Self::max_seq_len_for_fp8_decode(head_dim);
        let scratch_heads = if max_seq_len > fp8_cap {
            config
                .num_attention_heads_per_layer
                .as_ref()
                .and_then(|v| v.iter().copied().max())
                .unwrap_or(0)
                .max(config.num_attention_heads)
        } else {
            0
        };
        let bf16_full: std::collections::HashSet<usize> = env::var("NV_LAGUNA_FP8_BF16_FULL")
            .ok()
            .map(|s| s.split(',').filter_map(|x| x.trim().parse().ok()).collect())
            .unwrap_or_default();
        ensure!(
            scale_policy != LagunaFp8ScalePolicy::FixedOne
                || config.layer_types.iter().enumerate().any(|(idx, kind)| {
                    matches!(kind, LayerType::FullAttention) && !bf16_full.contains(&idx)
                }),
            "NV_LAGUNA_FP8_UNIT_SCALE=1 requires an FP8 full-attention layer; all layers use BF16"
        );
        let stream = nv_layers::cuda_stream::current_stream(&dev);
        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        let mut layer_windows = Vec::with_capacity(config.num_hidden_layers);
        let mut s_cap = 0usize;
        let mut s_window = 0usize;
        for (idx, kind) in config.layer_types.iter().enumerate() {
            let (slot, window) = match kind {
                LayerType::FullAttention if bf16_full.contains(&idx) => {
                    let shape = (1usize, max_seq_len, n_kv, head_dim);
                    (
                        Fp8LayerSlot::Bf16 {
                            k: Tensor::zeros(shape, dtype, device)?,
                            v: Tensor::zeros(shape, dtype, device)?,
                        },
                        None,
                    )
                }
                LayerType::FullAttention => {
                    let elem_count = max_seq_len * n_kv * head_dim;
                    let scale_count = max_seq_len * n_kv;
                    (
                        Fp8LayerSlot::Full {
                            k_fp8: stream.alloc_zeros::<u8>(elem_count)?,
                            v_fp8: stream.alloc_zeros::<u8>(elem_count)?,
                            k_scales: stream.alloc_zeros::<f32>(scale_count)?,
                            v_scales: stream.alloc_zeros::<f32>(scale_count)?,
                        },
                        None,
                    )
                }
                LayerType::SlidingAttention => {
                    let window = config.sliding_window.max(1);
                    let cap =
                        max_seq_len.min(window.checked_add(prefill_chunk).ok_or_else(|| {
                            anyhow::anyhow!("Laguna FP8 sliding capacity overflow")
                        })?);
                    if s_cap != 0 && (s_cap != cap || s_window != window) {
                        bail!(
                            "LagunaKvCacheFp8 requires uniform sliding layers \
                             (cap {}/{} window {}/{})",
                            s_cap,
                            cap,
                            s_window,
                            window
                        );
                    }
                    s_cap = cap;
                    s_window = window;
                    let shape = (1usize, cap, n_kv, head_dim);
                    (
                        Fp8LayerSlot::Bf16 {
                            k: Tensor::zeros(shape, dtype, device)?,
                            v: Tensor::zeros(shape, dtype, device)?,
                        },
                        Some(window),
                    )
                }
            };
            layers.push(slot);
            layer_windows.push(window);
        }
        let layer_stored = vec![0usize; layers.len()];
        let meta_dev = stream.alloc_zeros::<i32>(4)?;
        let n_total_dev = stream.alloc_zeros::<i32>(1)?;
        let scores_scratch = if scratch_heads > 0 {
            Some(
                stream
                    .alloc_zeros::<f32>(scratch_heads * max_seq_len)
                    .map_err(|e| anyhow::anyhow!("fp8 decode scores scratch: {e:?}"))?,
            )
        } else {
            None
        };
        if scale_policy == LagunaFp8ScalePolicy::PerTokenHeadInt8 {
            eprintln!("[laguna_fp8_cache] scale_policy=per-token-head-int8 elem=int8 writers=quantize_kv_int8/kv_append_int8 readers=kvq2");
        }
        if scale_policy == LagunaFp8ScalePolicy::FixedOne {
            eprintln!("[laguna_fp8_cache] scale_policy=fixed-one k_scale=1 v_scale=1 eager_writer=quantize_kv_fp8_unit_scale graph_writer=kv_append_fp8_unit_scale capacity={max_seq_len} fp8_layers={}", layers.iter().filter(|layer| matches!(layer, Fp8LayerSlot::Full { .. })).count());
        }
        Ok(Self {
            scale_policy,
            layers,
            layer_windows,
            layer_stored,
            n_kv,
            head_dim,
            current_len: 0,
            max_seq_len,
            device: device.clone(),
            dev,
            meta_dev,
            host_meta: Box::new([0i32; 4]),
            n_total_dev,
            host_n_total: Box::new([0i32; 1]),
            s_stored: 0,
            full_stored: 0,
            s_cap,
            s_window,
            scores_scratch,
            scratch_heads,
            prefill_attn: None,
            batch_decode: None,
        })
    }

    pub fn direct_fp8_prefill_enqueued_calls(&self) -> u64 {
        self.prefill_attn
            .as_ref()
            .map_or(0, |runner| runner.enqueued_calls())
    }

    pub fn copy_from_bf16(&mut self, source: &mut crate::laguna::LagunaKvCache) -> Result<()> {
        let source = source.checked_bf16_handoff()?;
        ensure!(
            self.scale_policy == LagunaFp8ScalePolicy::FixedOne,
            "BF16 handoff requires explicit FixedOne destination storage"
        );
        ensure!(
            self.current_len == 0
                && self.full_stored == 0
                && self.s_stored == 0
                && self.layer_stored.iter().all(|stored| *stored == 0),
            "BF16 handoff destination must be empty or reset"
        );
        ensure!(self.n_kv == 8 && self.head_dim == 128 && self.max_seq_len <= 262144
            && source.max_seq_len == self.max_seq_len && source.n_kv == self.n_kv
            && source.head_dim == self.head_dim && source.windows == self.layer_windows.as_slice()
            && source.layers.len() == self.layers.len() && source.sliding_cap == self.s_cap
            && source.sliding_window == self.s_window,
            "BF16 handoff requires matching Laguna cache geometry, layers, capacity and sliding layout");
        let stream = nv_layers::cuda_stream::current_stream(&self.dev);
        ensure!(
            stream.cu_stream() == self.dev.cuda_stream().cu_stream()
                && Arc::ptr_eq(stream.context(), self.dev.cuda_stream().context()),
            "BF16 handoff requires the cache's legacy stream"
        );
        ensure!(
            stream.capture_status()?
                == cudarc::driver::sys::CUstreamCaptureStatus::CU_STREAM_CAPTURE_STATUS_NONE,
            "BF16 handoff cannot run during CUDA graph capture"
        );
        let validate_tensor = |tensor: &Tensor, cap: usize| -> Result<()> {
            ensure!(
                tensor.dtype() == DType::BF16
                    && tensor.is_contiguous()
                    && tensor.dims() == [1, cap, self.n_kv, self.head_dim]
                    && tensor.device().same_device(&self.device),
                "BF16 handoff tensor dtype/layout/device mismatch"
            );
            let (storage, _) = tensor.storage_and_layout();
            let Storage::Cuda(storage) = &*storage else {
                bail!("BF16 handoff requires CUDA storage");
            };
            ensure!(
                Arc::ptr_eq(storage.as_cuda_slice::<bf16>()?.context(), stream.context()),
                "BF16 handoff requires one CUDA context"
            );
            Ok(())
        };
        for (layer, ((k, v), slot)) in source.layers.iter().zip(&self.layers).enumerate() {
            let cap = if self.layer_windows[layer].is_some() {
                self.s_cap
            } else {
                self.max_seq_len
            };
            validate_tensor(k, cap)?;
            validate_tensor(v, cap)?;
            match (self.layer_windows[layer], slot) {
                (
                    None,
                    Fp8LayerSlot::Full {
                        k_fp8,
                        v_fp8,
                        k_scales,
                        v_scales,
                    },
                ) => {
                    ensure!(
                        k_fp8.len() == cap * self.n_kv * self.head_dim
                            && v_fp8.len() == k_fp8.len()
                            && k_scales.len() == cap * self.n_kv
                            && v_scales.len() == k_scales.len()
                            && Arc::ptr_eq(k_fp8.context(), stream.context())
                            && Arc::ptr_eq(v_fp8.context(), stream.context())
                            && Arc::ptr_eq(k_scales.context(), stream.context())
                            && Arc::ptr_eq(v_scales.context(), stream.context()),
                        "BF16 handoff FP8 allocation mismatch at layer {layer}"
                    );
                }
                (Some(_), Fp8LayerSlot::Bf16 { k, v }) => {
                    validate_tensor(k, cap)?;
                    validate_tensor(v, cap)?;
                }
                _ => bail!("BF16 handoff refuses a full-layer precision fallback at layer {layer}"),
            }
        }
        stream
            .context()
            .synchronize()
            .map_err(|e| anyhow::anyhow!("BF16 handoff initial sync: {e}"))?;
        let result = (|| -> Result<()> {
            *self.host_meta = [0; 4];
            self.host_n_total[0] = 0;
            stream.memcpy_htod(&self.host_meta[..], &mut self.meta_dev)?;
            stream.memcpy_htod(&self.host_n_total[..], &mut self.n_total_dev)?;
            self.full_stored = source.frontier;
            self.s_stored = source.sliding_stored;
            for (layer, (k, v)) in source.layers.iter().enumerate() {
                let rows = if self.layer_windows[layer].is_some() {
                    source.sliding_stored
                } else {
                    source.frontier
                };
                if rows != 0 {
                    self.write_at_impl(layer, &k.narrow(1, 0, rows)?, &v.narrow(1, 0, rows)?)?;
                }
            }
            stream
                .synchronize()
                .map_err(|e| anyhow::anyhow!("BF16 handoff conversion sync: {e}"))?;
            *self.host_meta = [
                source.frontier as i32,
                source.sliding_stored as i32,
                0,
                source.sliding_stored as i32,
            ];
            self.host_n_total[0] = source.frontier as i32;
            stream.memcpy_htod(&self.host_meta[..], &mut self.meta_dev)?;
            stream.memcpy_htod(&self.host_n_total[..], &mut self.n_total_dev)?;
            stream
                .synchronize()
                .map_err(|e| anyhow::anyhow!("BF16 handoff metadata sync: {e}"))?;
            self.current_len = source.frontier;
            Ok(())
        })();
        if result.is_err() {
            self.reset();
            let _no_capture_in_flight = nv_kernels::capture_exclusion::no_capture_in_flight();
            if let Err(error) = stream.context().synchronize() {
                bail!("BF16 handoff failed: {result:?}; cleanup sync failed: {error}");
            }
        }
        result
    }

    pub fn scale_policy(&self) -> LagunaFp8ScalePolicy {
        self.scale_policy
    }

    pub fn current_len(&self) -> usize {
        self.current_len
    }
    pub fn max_seq_len(&self) -> usize {
        self.max_seq_len
    }
    pub fn advance(&mut self, n: usize) {
        self.current_len += n;
    }
    pub fn device(&self) -> &Device {
        &self.device
    }

    pub fn reset(&mut self) {
        self.current_len = 0;
        *self.host_meta = [0i32; 4];
        self.host_n_total[0] = 0;
        self.s_stored = 0;
        self.full_stored = 0;
        for s in self.layer_stored.iter_mut() {
            *s = 0;
        }
    }

    pub fn rollback(&mut self, n: usize) -> Result<()> {
        if n == 0 {
            return Ok(());
        }
        if n > self.current_len {
            bail!(
                "LagunaKvCacheFp8.rollback: n {} > current_len {}",
                n,
                self.current_len
            );
        }
        for (layer, stored) in self.layer_stored.iter_mut().enumerate() {
            if *stored < n {
                bail!(
                    "LagunaKvCacheFp8.rollback: layer {layer} stored {} < n {}",
                    *stored,
                    n
                );
            }
            *stored -= n;
        }
        if self.full_stored < n || (self.s_cap > 0 && self.s_stored < n) {
            bail!(
                "LagunaKvCacheFp8.rollback: stored underflow (full {}, sliding {}, n {})",
                self.full_stored,
                self.s_stored,
                n
            );
        }
        self.full_stored -= n;
        if self.s_cap > 0 {
            self.s_stored -= n;
        }
        self.current_len -= n;
        Ok(())
    }

    pub(crate) fn ring_meta_ptr(&self) -> u64 {
        let stream = nv_layers::cuda_stream::current_stream(&self.dev);
        let (p, _g) = self.meta_dev.device_ptr(&stream);
        p
    }

    pub(crate) fn sliding_cap(&self) -> usize {
        self.s_cap
    }

    pub(crate) fn layer_bf16_bufs(&self, layer: usize) -> Option<(Tensor, Tensor)> {
        match &self.layers[layer] {
            Fp8LayerSlot::Bf16 { k, v } => Some((k.clone(), v.clone())),
            Fp8LayerSlot::Full { .. } => None,
        }
    }

    pub(crate) fn layer_fp8_ptrs(&self, layer: usize) -> Option<(u64, u64, u64, u64)> {
        match &self.layers[layer] {
            Fp8LayerSlot::Bf16 { .. } => None,
            Fp8LayerSlot::Full {
                k_fp8,
                v_fp8,
                k_scales,
                v_scales,
            } => {
                let stream = nv_layers::cuda_stream::current_stream(&self.dev);
                let (kp, _g1) = k_fp8.device_ptr(&stream);
                let (vp, _g2) = v_fp8.device_ptr(&stream);
                let (ksp, _g3) = k_scales.device_ptr(&stream);
                let (vsp, _g4) = v_scales.device_ptr(&stream);
                Some((kp, vp, ksp, vsp))
            }
        }
    }

    pub(crate) fn note_graph_write(&mut self) {
        let (fs, ss) = (self.full_stored, self.s_stored);
        for (li, w) in self.layer_windows.iter().enumerate() {
            self.layer_stored[li] = if w.is_some() { ss } else { fs };
        }
    }

    pub fn prepare_for_decode_dev(&mut self, write_pos: usize, n_total: usize) -> Result<()> {
        let stream = nv_layers::cuda_stream::current_stream(&self.dev);
        let layers = &self.layers;
        let layer_windows = &self.layer_windows;
        let n_kv = self.n_kv;
        let head_dim = self.head_dim;
        crate::laguna::ring_prepare_decode_meta(
            "LagunaKvCacheFp8",
            "LagunaKvCacheFp8",
            write_pos,
            n_total,
            self.max_seq_len,
            self.s_cap,
            self.s_window,
            &mut self.s_stored,
            &mut self.full_stored,
            &mut self.host_meta,
            |shift, keep| {
                for (li, w) in layer_windows.iter().enumerate() {
                    if w.is_none() {
                        continue;
                    }
                    if let Fp8LayerSlot::Bf16 { k, v } = &layers[li] {
                        for buf in [k, v] {
                            shift_rows_launch(&stream, buf, shift, keep, n_kv, head_dim)?;
                        }
                    }
                }
                Ok(())
            },
        )?;
        stream
            .memcpy_htod(&self.host_meta[..], &mut self.meta_dev)
            .map_err(|e| anyhow::anyhow!("htod fp8 kv meta: {e:?}"))?;
        self.host_n_total[0] = n_total as i32;
        stream
            .memcpy_htod(&self.host_n_total[..], &mut self.n_total_dev)
            .map_err(|e| anyhow::anyhow!("htod n_total: {e:?}"))?;
        Ok(())
    }

    pub fn write_at_impl(&mut self, layer: usize, k_new: &Tensor, v_new: &Tensor) -> Result<()> {
        let n_kv = self.n_kv;
        let head_dim = self.head_dim;
        let dims = k_new.dims();
        if dims.len() != 4 || dims[0] != 1 || dims[2] != n_kv || dims[3] != head_dim {
            bail!(
                "LagunaKvCacheFp8.write_at layer {layer}: expected [1, t, {n_kv}, {head_dim}], got {:?}",
                dims
            );
        }
        if v_new.dims() != dims {
            bail!(
                "LagunaKvCacheFp8.write_at: k/v shape mismatch k={:?} v={:?}",
                dims,
                v_new.dims()
            );
        }
        let t = dims[1];
        let stream = nv_layers::cuda_stream::current_stream(&self.dev);
        let k_own;
        let k_new = if k_new.is_contiguous() {
            k_new
        } else {
            k_own = k_new.contiguous()?;
            &k_own
        };
        let v_own;
        let v_new = if v_new.is_contiguous() {
            v_new
        } else {
            v_own = v_new.contiguous()?;
            &v_own
        };

        match &mut self.layers[layer] {
            Fp8LayerSlot::Bf16 { k, v } => {
                let (cap, meta_idx, committed) = match self.layer_windows[layer] {
                    None => (self.max_seq_len, 0usize, self.full_stored),
                    Some(_) => (self.s_cap, 1usize, self.s_stored),
                };
                for (src, dst) in [(k_new, &*k), (v_new, &*v)] {
                    ring_append_launch(
                        &stream,
                        src,
                        dst,
                        &self.meta_dev,
                        meta_idx,
                        t,
                        cap,
                        n_kv,
                        head_dim,
                    )?;
                }
                self.layer_stored[layer] = committed;
                Ok(())
            }
            Fp8LayerSlot::Full {
                k_fp8,
                v_fp8,
                k_scales,
                v_scales,
            } => {
                let end = self.host_meta[0] as usize + t;
                if end > self.max_seq_len {
                    bail!(
                        "LagunaKvCacheFp8.write_at: end {} exceeds max_seq_len {}",
                        end,
                        self.max_seq_len
                    );
                }
                let start_view = self.meta_dev.slice(0..);
                let (start_dev_ptr, _gsp) = start_view.device_ptr(&stream);

                let (k_storage, kl) = k_new.storage_and_layout();
                let (v_storage, vl) = v_new.storage_and_layout();
                let k_cuda = match &*k_storage {
                    Storage::Cuda(s) => s,
                    _ => bail!("k_new must be on the CUDA device"),
                };
                let v_cuda = match &*v_storage {
                    Storage::Cuda(s) => s,
                    _ => bail!("v_new must be on the CUDA device"),
                };
                let k_slice = k_cuda.as_cuda_slice::<bf16>()?;
                let v_slice = v_cuda.as_cuda_slice::<bf16>()?;
                let k_view = k_slice.slice(kl.start_offset()..);
                let v_view = v_slice.slice(vl.start_offset()..);

                let (k_in_ptr, _gki) = k_view.device_ptr(&stream);
                let (v_in_ptr, _gvi) = v_view.device_ptr(&stream);
                let (k_fp8_base, _gkf) = k_fp8.device_ptr_mut(&stream);
                let (v_fp8_base, _gvf) = v_fp8.device_ptr_mut(&stream);
                let (k_sc_base, _gks) = k_scales.device_ptr_mut(&stream);
                let (v_sc_base, _gvs) = v_scales.device_ptr_mut(&stream);

                let quantize = match self.scale_policy {
                    LagunaFp8ScalePolicy::PerTokenHead => nv_kernels::cuda::quantize_kv_fp8,
                    LagunaFp8ScalePolicy::FixedOne => nv_kernels::cuda::quantize_kv_fp8_unit_scale,
                    LagunaFp8ScalePolicy::PerTokenHeadInt8 => nv_kernels::cuda::quantize_kv_int8,
                };
                let s_raw = stream.cu_stream() as *mut c_void;
                let rc_k = unsafe {
                    quantize(
                        s_raw,
                        k_in_ptr as *const u16,
                        k_fp8_base as *mut u8,
                        k_sc_base as *mut f32,
                        start_dev_ptr as *const i32,
                        t as i32,
                        n_kv as i32,
                        head_dim as i32,
                        0,
                    )
                };
                if rc_k != 0 {
                    bail!("quantize_kv_fp8(k) rc={rc_k}");
                }
                let rc_v = unsafe {
                    quantize(
                        s_raw,
                        v_in_ptr as *const u16,
                        v_fp8_base as *mut u8,
                        v_sc_base as *mut f32,
                        start_dev_ptr as *const i32,
                        t as i32,
                        n_kv as i32,
                        head_dim as i32,
                        0,
                    )
                };
                if rc_v != 0 {
                    bail!("quantize_kv_fp8(v) rc={rc_v}");
                }
                self.layer_stored[layer] = self.full_stored;
                Ok(())
            }
        }
    }

    pub fn view_bf16(&mut self, layer: usize, len: usize) -> Result<(Tensor, Tensor)> {
        let n_kv = self.n_kv;
        let head_dim = self.head_dim;
        match &self.layers[layer] {
            Fp8LayerSlot::Bf16 { k, v } => {
                let stored = match self.layer_windows[layer] {
                    None => self.full_stored,
                    Some(_) => self.s_stored,
                };
                let k = k.narrow(1, 0, stored)?;
                let v = v.narrow(1, 0, stored)?;
                Ok((k, v))
            }
            Fp8LayerSlot::Full {
                k_fp8,
                v_fp8,
                k_scales,
                v_scales,
            } => {
                if len > self.max_seq_len {
                    bail!(
                        "LagunaKvCacheFp8.view_bf16: len {len} > max_seq_len {}",
                        self.max_seq_len
                    );
                }
                let dev = self.dev.clone();
                let stream = nv_layers::cuda_stream::current_stream(&dev);
                let need = len * n_kv * head_dim;

                let mut k_out = unsafe { stream.alloc::<bf16>(need)? };
                let mut v_out = unsafe { stream.alloc::<bf16>(need)? };
                let dequantize = match self.scale_policy {
                    LagunaFp8ScalePolicy::PerTokenHeadInt8 => nv_kernels::cuda::dequantize_kv_int8,
                    _ => nv_kernels::cuda::dequantize_kv_fp8,
                };

                {
                    let (k_fp8_ptr, _gk) = k_fp8.device_ptr(&stream);
                    let (v_fp8_ptr, _gv) = v_fp8.device_ptr(&stream);
                    let (k_sc_ptr, _gks) = k_scales.device_ptr(&stream);
                    let (v_sc_ptr, _gvs) = v_scales.device_ptr(&stream);
                    let (k_out_ptr, _gko) = k_out.device_ptr_mut(&stream);
                    let (v_out_ptr, _gvo) = v_out.device_ptr_mut(&stream);
                    let s_raw = stream.cu_stream() as *mut c_void;
                    let rc_k = unsafe {
                        dequantize(
                            s_raw,
                            k_fp8_ptr as *const u8,
                            k_sc_ptr as *const f32,
                            k_out_ptr as *mut u16,
                            0,
                            len as i32,
                            n_kv as i32,
                            head_dim as i32,
                            0,
                        )
                    };
                    if rc_k != 0 {
                        bail!("dequantize_kv_fp8(k) rc={rc_k}");
                    }
                    let rc_v = unsafe {
                        dequantize(
                            s_raw,
                            v_fp8_ptr as *const u8,
                            v_sc_ptr as *const f32,
                            v_out_ptr as *mut u16,
                            0,
                            len as i32,
                            n_kv as i32,
                            head_dim as i32,
                            0,
                        )
                    };
                    if rc_v != 0 {
                        bail!("dequantize_kv_fp8(v) rc={rc_v}");
                    }
                }

                let k_storage = CudaStorage::wrap_cuda_slice(k_out, dev.clone());
                let v_storage = CudaStorage::wrap_cuda_slice(v_out, dev);
                let k = candle_core::Tensor::from_storage(
                    Storage::Cuda(k_storage),
                    (1usize, len, n_kv, head_dim),
                    BackpropOp::none(),
                    false,
                );
                let v = candle_core::Tensor::from_storage(
                    Storage::Cuda(v_storage),
                    (1usize, len, n_kv, head_dim),
                    BackpropOp::none(),
                    false,
                );
                Ok((k, v))
            }
        }
    }

    pub fn decode_attention_fp8(
        &mut self,
        layer: usize,
        q_rot: &Tensor,
        n_q: usize,
        sliding_window: Option<usize>,
        scaling: f32,
    ) -> Result<Option<Tensor>> {
        let n_kv = self.n_kv;
        let head_dim = self.head_dim;
        let (k_fp8, v_fp8, k_scales, v_scales) = match &self.layers[layer] {
            Fp8LayerSlot::Bf16 { .. } => return Ok(None),
            Fp8LayerSlot::Full {
                k_fp8,
                v_fp8,
                k_scales,
                v_scales,
            } => (k_fp8, v_fp8, k_scales, v_scales),
        };

        let dims = q_rot.dims();
        let expected = n_q * head_dim;
        let total: usize = dims.iter().product();
        if total != expected {
            bail!(
                "LagunaKvCacheFp8.decode_attention_fp8 layer {layer}: expected total {expected}, got dims {:?}",
                dims
            );
        }
        let dev = self.dev.clone();
        let stream = nv_layers::cuda_stream::current_stream(&dev);

        let mut out = unsafe { stream.alloc::<bf16>(expected)? };

        let q_c = q_rot.contiguous()?;
        let (q_storage, ql) = q_c.storage_and_layout();
        let q_cuda = match &*q_storage {
            Storage::Cuda(s) => s,
            _ => bail!("q_rot must be on CUDA"),
        };
        let q_slice = q_cuda.as_cuda_slice::<bf16>()?;
        let q_view = q_slice.slice(ql.start_offset()..);

        let (n_total_ptr, _gnt) = self.n_total_dev.device_ptr(&stream);
        let (q_ptr, _gq) = q_view.device_ptr(&stream);
        let (k_ptr, _gk) = k_fp8.device_ptr(&stream);
        let (v_ptr, _gv) = v_fp8.device_ptr(&stream);
        let (ks_ptr, _gks) = k_scales.device_ptr(&stream);
        let (vs_ptr, _gvs) = v_scales.device_ptr(&stream);
        let (out_ptr, _go) = out.device_ptr_mut(&stream);

        let sw_i32 = sliding_window.map(|w| w as i32).unwrap_or(0);
        let max_total = self.max_seq_len as i32;
        let s_raw = stream.cu_stream() as *mut c_void;
        let rc = match &self.scores_scratch {
            None => unsafe {
                nv_kernels::cuda::attention_fp8_decode(
                    s_raw,
                    q_ptr as *const u16,
                    k_ptr as *const u8,
                    v_ptr as *const u8,
                    ks_ptr as *const f32,
                    vs_ptr as *const f32,
                    out_ptr as *mut u16,
                    n_q as i32,
                    n_kv as i32,
                    head_dim as i32,
                    n_total_ptr as *const i32,
                    max_total,
                    sw_i32,
                    scaling,
                )
            },
            Some(scratch) => {
                if n_q > self.scratch_heads {
                    bail!(
                        "LagunaKvCacheFp8.decode_attention_fp8 layer {layer}: n_q {n_q} exceeds \
                         scores scratch heads {}",
                        self.scratch_heads
                    );
                }
                let (sc_ptr, _gsc) = scratch.device_ptr(&stream);
                unsafe {
                    nv_kernels::cuda::attention_fp8_decode_gscores(
                        s_raw,
                        q_ptr as *const u16,
                        k_ptr as *const u8,
                        v_ptr as *const u8,
                        ks_ptr as *const f32,
                        vs_ptr as *const f32,
                        out_ptr as *mut u16,
                        n_q as i32,
                        n_kv as i32,
                        head_dim as i32,
                        n_total_ptr as *const i32,
                        max_total,
                        sw_i32,
                        scaling,
                        sc_ptr as *mut f32,
                    )
                }
            }
        };
        if rc != 0 {
            bail!("attention_fp8_decode rc={rc}");
        }

        drop(_go);
        drop(_gq);
        drop(_gk);
        drop(_gv);
        drop(_gks);
        drop(_gvs);
        drop(_gnt);
        drop(q_storage);

        let storage = CudaStorage::wrap_cuda_slice(out, dev);
        let tensor = candle_core::Tensor::from_storage(
            Storage::Cuda(storage),
            (1usize, 1usize, n_q, head_dim),
            BackpropOp::none(),
            false,
        );
        Ok(Some(tensor))
    }
}

impl Gemma4Cache for LagunaKvCacheFp8 {
    fn current_len(&self) -> usize {
        LagunaKvCacheFp8::current_len(self)
    }
    fn advance(&mut self, n: usize) {
        LagunaKvCacheFp8::advance(self, n)
    }
    fn prepare_for_decode(&mut self, write_pos: usize, n_total: usize) -> Result<()> {
        LagunaKvCacheFp8::prepare_for_decode_dev(self, write_pos, n_total)
    }
    fn write_at(&mut self, layer: usize, k_new: &Tensor, v_new: &Tensor) -> Result<()> {
        LagunaKvCacheFp8::write_at_impl(self, layer, k_new, v_new)
    }
    fn view(&mut self, layer: usize, len: usize) -> Result<(Tensor, Tensor)> {
        LagunaKvCacheFp8::view_bf16(self, layer, len)
    }
    fn try_prefill_attention_fp8(
        &mut self,
        layer: usize,
        q_rot: &Tensor,
        n_q: usize,
        seq: usize,
        sliding_window: Option<usize>,
        scaling: f32,
    ) -> Result<Option<Tensor>> {
        if env::var("NV_LAGUNA_PREFILL_FLASHINFER").as_deref() != Ok("1")
            || self.scale_policy != LagunaFp8ScalePolicy::FixedOne
            || sliding_window.is_some()
        {
            return Ok(None);
        }
        let slot = self.layers.get(layer).ok_or_else(|| {
            anyhow::anyhow!("Laguna direct FP8 prefill layer {layer} is out of range")
        })?;
        let Fp8LayerSlot::Full { k_fp8, v_fp8, .. } = slot else {
            return Ok(None);
        };
        ensure!(
            self.layer_windows[layer].is_none()
                && n_q == 48
                && self.n_kv == 8
                && self.head_dim == 128,
            "Laguna direct FP8 prefill requires a full layer with Q48/KV8/HD128"
        );
        ensure!(
            scaling.is_finite() && (scaling - 1.0 / 128.0f32.sqrt()).abs() <= 1e-7,
            "Laguna direct FP8 prefill requires the native attention scale"
        );
        let kv_len = self.layer_stored[layer];
        ensure!(kv_len == self.full_stored && kv_len <= self.max_seq_len && seq > 1 && seq <= kv_len
            && usize::try_from(self.host_meta[0]).ok().and_then(|start| start.checked_add(seq)) == Some(kv_len),
            "Laguna direct FP8 prefill requires the initialized layer tail: layer={layer} stored={kv_len} prepared={} start={} M={seq}", self.full_stored, self.host_meta[0]);
        self.prefill_attn
            .get_or_insert_default()
            .run_fixed_one(
                q_rot,
                k_fp8,
                v_fp8,
                &self.device,
                seq,
                kv_len,
                self.max_seq_len,
            )
            .map(Some)
    }
    fn try_decode_attention_fp8(
        &mut self,
        layer: usize,
        q_rot: &Tensor,
        n_q: usize,
        sliding_window: Option<usize>,
        scaling: f32,
    ) -> Result<Option<Tensor>> {
        if self.batch_decode.is_some() {
            return self
                .batch_attention(layer, q_rot, n_q, sliding_window, scaling)
                .map(Some);
        }
        LagunaKvCacheFp8::decode_attention_fp8(self, layer, q_rot, n_q, sliding_window, scaling)
    }
}
