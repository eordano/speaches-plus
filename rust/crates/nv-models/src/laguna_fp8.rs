pub mod prefix;

use anyhow::Result;
use candle_core::{DType, Device, Tensor};
use cudarc::driver::{CudaSlice, DevicePtr, DevicePtrMut};
use half::bf16;
use std::ffi::c_void;

use crate::gemma4::{Gemma4Cache, LayerType};
use crate::laguna::{ring_append_launch, shift_rows_launch, LagunaConfig, SLIDING_COMPACT_SLACK};

const SMEM_LIMIT_BYTES: usize = 48 * 1024;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum LagunaFp8ScalePolicy {
    PerTokenHead,
    FixedOne,
    PerTokenHeadInt8,
}

impl LagunaFp8ScalePolicy {
    pub(crate) fn from_env() -> Result<Self> {
        let int8 = match std::env::var("NV_LAGUNA_KV8_INT8") {
            Err(std::env::VarError::NotPresent) => false,
            Ok(value) if value == "0" => false,
            Ok(value) if value == "1" => true,
            other => anyhow::bail!("NV_LAGUNA_KV8_INT8 must be unset, 0 or 1, got {other:?}"),
        };
        match std::env::var("NV_LAGUNA_FP8_UNIT_SCALE") {
            Err(std::env::VarError::NotPresent) if int8 => Ok(Self::PerTokenHeadInt8),
            Ok(value) if value == "0" && int8 => Ok(Self::PerTokenHeadInt8),
            Err(std::env::VarError::NotPresent) => Ok(Self::PerTokenHead),
            Ok(value) if value == "0" => Ok(Self::PerTokenHead),
            Ok(value) if value == "1" && int8 => anyhow::bail!("NV_LAGUNA_KV8_INT8=1 and NV_LAGUNA_FP8_UNIT_SCALE=1 are exclusive: int8 K/V carries per-token-per-head scales"),
            Ok(value) if value == "1" => Ok(Self::FixedOne),
            other => anyhow::bail!("NV_LAGUNA_FP8_UNIT_SCALE must be unset, 0 or 1, got {other:?}"),
        }
    }

    pub(crate) fn require_bf16_cache_compatible() -> Result<()> {
        anyhow::ensure!(
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
    xqa: crate::laguna_step_graph::XqaFp8Plan,
    scratch: CudaSlice<f32>,
    fan_in: CudaSlice<u32>,
    rows_per_block: i32,
}

struct FixedOneBatchAttention {
    verify: Option<FixedOneBatchVerifyAttention>,
    xqa: crate::laguna_step_graph::XqaFp8Plan,
    scratch: CudaSlice<f32>,
    fan_in: CudaSlice<u32>,
}

impl LagunaKvCacheFp8 {
    pub(crate) fn enable_fixed_one_batch_attention(&mut self, config: &LagunaConfig) -> Result<()> {
        anyhow::ensure!(
            self.batch_decode.is_none(),
            "batch attention already initialized"
        );
        anyhow::ensure!(
            self.scale_policy == LagunaFp8ScalePolicy::FixedOne
                && self.n_kv == 8
                && self.head_dim == 128
                && self.s_window == 512,
            "batch attention requires immutable FixedOne KV8/head128/window512"
        );
        anyhow::ensure!(
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
            anyhow::ensure!(
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
        let xqa = crate::laguna_step_graph::XqaFp8Plan::new_for(
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
        anyhow::ensure!(elems > 0, "invalid batch sliding scratch extent");
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
        anyhow::ensure!(
            state.verify.is_none(),
            "M3 batch attention already initialized"
        );
        let rows_per_block = match std::env::var("NV_LAGUNA_VERIFY_SLIDING_ROWS").as_deref() {
            Ok("1") => 1,
            Ok("2") => 2,
            Ok("3") => 3,
            _ => anyhow::bail!("m2-spec requires NV_LAGUNA_VERIFY_SLIDING_ROWS=1, 2 or 3"),
        };
        let xqa = crate::laguna_step_graph::XqaFp8Plan::new_for(
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
        anyhow::ensure!(rows == 3, "unsupported batch verify rows");
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
        anyhow::ensure!(
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
        anyhow::ensure!(
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
        anyhow::ensure!(
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
                candle_core::Storage::Cuda(v) => v,
                _ => anyhow::bail!(
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
                    anyhow::ensure!(
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
                    anyhow::ensure!(
                        window == Some(512) && matches!(heads, 48 | 64),
                        "sliding {} attention shape mismatch",
                        if rows == 1 { "batch" } else { "M3" }
                    );
                    let (ks, kl) = k.storage_and_layout();
                    let (vs, vl) = v.storage_and_layout();
                    let (kc, vc) = match (&*ks, &*vs) {
                        (candle_core::Storage::Cuda(kc), candle_core::Storage::Cuda(vc)) => {
                            (kc, vc)
                        }
                        _ => anyhow::bail!(
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
                        anyhow::ensure!(rc == 0, "batch sliding GQA rc={rc}");
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
                        anyhow::ensure!(rc == 0, "sliding M3 attention rc={rc}");
                    }
                }
            }
        }
        Ok(Tensor::from_storage(
            candle_core::Storage::Cuda(candle_core::CudaStorage::wrap_cuda_slice(
                out,
                self.dev.clone(),
            )),
            (1, rows, heads, 128),
            candle_core::op::BackpropOp::none(),
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
        anyhow::ensure!(
            std::env::var_os("NV_LAGUNA_FP8_BF16_FULL").is_none(),
            "explicit FixedOne construction refuses BF16 full-layer overrides"
        );
        anyhow::ensure!(
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
        anyhow::ensure!(
            matches!(prefill_chunk, 256 | 512 | 1024 | 2048 | 4096 | 8192),
            "invalid Laguna FP8 prefill capacity {prefill_chunk}"
        );
        let dev = match device {
            Device::Cuda(d) => d.clone(),
            _ => anyhow::bail!("LagunaKvCacheFp8 requires a CUDA device"),
        };
        if dtype != DType::BF16 {
            anyhow::bail!("LagunaKvCacheFp8 requires a BF16 model, got {dtype:?}");
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
        let bf16_full: std::collections::HashSet<usize> = std::env::var("NV_LAGUNA_FP8_BF16_FULL")
            .ok()
            .map(|s| s.split(',').filter_map(|x| x.trim().parse().ok()).collect())
            .unwrap_or_default();
        anyhow::ensure!(
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
                            k_fp8: stream
                                .alloc_zeros::<u8>(elem_count)
                                .map_err(|e| anyhow::anyhow!(e))?,
                            v_fp8: stream
                                .alloc_zeros::<u8>(elem_count)
                                .map_err(|e| anyhow::anyhow!(e))?,
                            k_scales: stream
                                .alloc_zeros::<f32>(scale_count)
                                .map_err(|e| anyhow::anyhow!(e))?,
                            v_scales: stream
                                .alloc_zeros::<f32>(scale_count)
                                .map_err(|e| anyhow::anyhow!(e))?,
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
                        anyhow::bail!(
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
        let meta_dev = stream
            .alloc_zeros::<i32>(4)
            .map_err(|e| anyhow::anyhow!(e))?;
        let n_total_dev = stream
            .alloc_zeros::<i32>(1)
            .map_err(|e| anyhow::anyhow!(e))?;
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
        anyhow::ensure!(
            self.scale_policy == LagunaFp8ScalePolicy::FixedOne,
            "BF16 handoff requires explicit FixedOne destination storage"
        );
        anyhow::ensure!(
            self.current_len == 0
                && self.full_stored == 0
                && self.s_stored == 0
                && self.layer_stored.iter().all(|stored| *stored == 0),
            "BF16 handoff destination must be empty or reset"
        );
        anyhow::ensure!(self.n_kv == 8 && self.head_dim == 128 && self.max_seq_len <= 262144
            && source.max_seq_len == self.max_seq_len && source.n_kv == self.n_kv
            && source.head_dim == self.head_dim && source.windows == self.layer_windows.as_slice()
            && source.layers.len() == self.layers.len() && source.sliding_cap == self.s_cap
            && source.sliding_window == self.s_window,
            "BF16 handoff requires matching Laguna cache geometry, layers, capacity and sliding layout");
        let stream = nv_layers::cuda_stream::current_stream(&self.dev);
        anyhow::ensure!(
            stream.cu_stream() == self.dev.cuda_stream().cu_stream()
                && std::sync::Arc::ptr_eq(stream.context(), self.dev.cuda_stream().context()),
            "BF16 handoff requires the cache's legacy stream"
        );
        anyhow::ensure!(
            stream.capture_status().map_err(|e| anyhow::anyhow!(e))?
                == cudarc::driver::sys::CUstreamCaptureStatus::CU_STREAM_CAPTURE_STATUS_NONE,
            "BF16 handoff cannot run during CUDA graph capture"
        );
        let validate_tensor = |tensor: &Tensor, cap: usize| -> Result<()> {
            anyhow::ensure!(
                tensor.dtype() == DType::BF16
                    && tensor.is_contiguous()
                    && tensor.dims() == [1, cap, self.n_kv, self.head_dim]
                    && tensor.device().same_device(&self.device),
                "BF16 handoff tensor dtype/layout/device mismatch"
            );
            let (storage, _) = tensor.storage_and_layout();
            let candle_core::Storage::Cuda(storage) = &*storage else {
                anyhow::bail!("BF16 handoff requires CUDA storage");
            };
            anyhow::ensure!(
                std::sync::Arc::ptr_eq(
                    storage.as_cuda_slice::<bf16>()?.context(),
                    stream.context()
                ),
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
                    anyhow::ensure!(
                        k_fp8.len() == cap * self.n_kv * self.head_dim
                            && v_fp8.len() == k_fp8.len()
                            && k_scales.len() == cap * self.n_kv
                            && v_scales.len() == k_scales.len()
                            && std::sync::Arc::ptr_eq(k_fp8.context(), stream.context())
                            && std::sync::Arc::ptr_eq(v_fp8.context(), stream.context())
                            && std::sync::Arc::ptr_eq(k_scales.context(), stream.context())
                            && std::sync::Arc::ptr_eq(v_scales.context(), stream.context()),
                        "BF16 handoff FP8 allocation mismatch at layer {layer}"
                    );
                }
                (Some(_), Fp8LayerSlot::Bf16 { k, v }) => {
                    validate_tensor(k, cap)?;
                    validate_tensor(v, cap)?;
                }
                _ => anyhow::bail!(
                    "BF16 handoff refuses a full-layer precision fallback at layer {layer}"
                ),
            }
        }
        stream
            .context()
            .synchronize()
            .map_err(|e| anyhow::anyhow!("BF16 handoff initial sync: {e}"))?;
        let result = (|| -> Result<()> {
            *self.host_meta = [0; 4];
            self.host_n_total[0] = 0;
            stream
                .memcpy_htod(&self.host_meta[..], &mut self.meta_dev)
                .map_err(|e| anyhow::anyhow!(e))?;
            stream
                .memcpy_htod(&self.host_n_total[..], &mut self.n_total_dev)
                .map_err(|e| anyhow::anyhow!(e))?;
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
            stream
                .memcpy_htod(&self.host_meta[..], &mut self.meta_dev)
                .map_err(|e| anyhow::anyhow!(e))?;
            stream
                .memcpy_htod(&self.host_n_total[..], &mut self.n_total_dev)
                .map_err(|e| anyhow::anyhow!(e))?;
            stream
                .synchronize()
                .map_err(|e| anyhow::anyhow!("BF16 handoff metadata sync: {e}"))?;
            self.current_len = source.frontier;
            Ok(())
        })();
        if result.is_err() {
            self.reset();
            if let Err(error) = stream.context().synchronize() {
                anyhow::bail!("BF16 handoff failed: {result:?}; cleanup sync failed: {error}");
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
    pub fn uses_score_scratch(&self) -> bool {
        self.scores_scratch.is_some()
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
            anyhow::bail!(
                "LagunaKvCacheFp8.rollback: n {} > current_len {}",
                n,
                self.current_len
            );
        }
        for (layer, stored) in self.layer_stored.iter_mut().enumerate() {
            if *stored < n {
                anyhow::bail!(
                    "LagunaKvCacheFp8.rollback: layer {layer} stored {} < n {}",
                    *stored,
                    n
                );
            }
            *stored -= n;
        }
        if self.full_stored < n || (self.s_cap > 0 && self.s_stored < n) {
            anyhow::bail!(
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
            anyhow::bail!(
                "LagunaKvCacheFp8.write_at layer {layer}: expected [1, t, {n_kv}, {head_dim}], got {:?}",
                dims
            );
        }
        if v_new.dims() != dims {
            anyhow::bail!(
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
                    anyhow::bail!(
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
                    candle_core::Storage::Cuda(s) => s,
                    _ => anyhow::bail!("k_new must be on the CUDA device"),
                };
                let v_cuda = match &*v_storage {
                    candle_core::Storage::Cuda(s) => s,
                    _ => anyhow::bail!("v_new must be on the CUDA device"),
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
                    anyhow::bail!("quantize_kv_fp8(k) rc={rc_k}");
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
                    anyhow::bail!("quantize_kv_fp8(v) rc={rc_v}");
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
                    anyhow::bail!(
                        "LagunaKvCacheFp8.view_bf16: len {len} > max_seq_len {}",
                        self.max_seq_len
                    );
                }
                let dev = self.dev.clone();
                let stream = nv_layers::cuda_stream::current_stream(&dev);
                let need = len * n_kv * head_dim;

                let mut k_out =
                    unsafe { stream.alloc::<bf16>(need).map_err(|e| anyhow::anyhow!(e))? };
                let mut v_out =
                    unsafe { stream.alloc::<bf16>(need).map_err(|e| anyhow::anyhow!(e))? };
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
                        anyhow::bail!("dequantize_kv_fp8(k) rc={rc_k}");
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
                        anyhow::bail!("dequantize_kv_fp8(v) rc={rc_v}");
                    }
                }

                let k_storage = candle_core::CudaStorage::wrap_cuda_slice(k_out, dev.clone());
                let v_storage = candle_core::CudaStorage::wrap_cuda_slice(v_out, dev);
                let k = candle_core::Tensor::from_storage(
                    candle_core::Storage::Cuda(k_storage),
                    (1usize, len, n_kv, head_dim),
                    candle_core::op::BackpropOp::none(),
                    false,
                );
                let v = candle_core::Tensor::from_storage(
                    candle_core::Storage::Cuda(v_storage),
                    (1usize, len, n_kv, head_dim),
                    candle_core::op::BackpropOp::none(),
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
            anyhow::bail!(
                "LagunaKvCacheFp8.decode_attention_fp8 layer {layer}: expected total {expected}, got dims {:?}",
                dims
            );
        }
        let dev = self.dev.clone();
        let stream = nv_layers::cuda_stream::current_stream(&dev);

        let mut out = unsafe {
            stream
                .alloc::<bf16>(expected)
                .map_err(|e| anyhow::anyhow!(e))?
        };

        let q_c = q_rot.contiguous()?;
        let (q_storage, ql) = q_c.storage_and_layout();
        let q_cuda = match &*q_storage {
            candle_core::Storage::Cuda(s) => s,
            _ => anyhow::bail!("q_rot must be on CUDA"),
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
                    anyhow::bail!(
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
            anyhow::bail!("attention_fp8_decode rc={rc}");
        }

        drop(_go);
        drop(_gq);
        drop(_gk);
        drop(_gv);
        drop(_gks);
        drop(_gvs);
        drop(_gnt);
        drop(q_storage);

        let storage = candle_core::CudaStorage::wrap_cuda_slice(out, dev);
        let tensor = candle_core::Tensor::from_storage(
            candle_core::Storage::Cuda(storage),
            (1usize, 1usize, n_q, head_dim),
            candle_core::op::BackpropOp::none(),
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
        if std::env::var("NV_LAGUNA_PREFILL_FLASHINFER").as_deref() != Ok("1")
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
        anyhow::ensure!(
            self.layer_windows[layer].is_none()
                && n_q == 48
                && self.n_kv == 8
                && self.head_dim == 128,
            "Laguna direct FP8 prefill requires a full layer with Q48/KV8/HD128"
        );
        anyhow::ensure!(
            scaling.is_finite() && (scaling - 1.0 / 128.0f32.sqrt()).abs() <= 1e-7,
            "Laguna direct FP8 prefill requires the native attention scale"
        );
        let kv_len = self.layer_stored[layer];
        anyhow::ensure!(kv_len == self.full_stored && kv_len <= self.max_seq_len && seq > 1 && seq <= kv_len
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

#[cfg(test)]
mod fixed_one_tests {
    use super::*;
    use nv_kernels::graph::CudaGraphRunner;

    fn decode_positive_e4m3(bits: u8) -> f32 {
        let exponent = (bits >> 3) as i32;
        let mantissa = (bits & 7) as f32;
        if exponent == 0 {
            mantissa / 512.0
        } else {
            (1.0 + mantissa / 8.0) * 2.0f32.powi(exponent - 7)
        }
    }

    fn encode_reference(value: bf16) -> u8 {
        let x = value.to_f32();
        assert!(x.is_finite());
        let magnitude = x.abs();
        let mut chosen = 0u8;
        let mut distance = f32::INFINITY;
        for bits in 0..=126u8 {
            let d = (magnitude - decode_positive_e4m3(bits)).abs();
            if d < distance || (d == distance && bits & 1 == 0) {
                chosen = bits;
                distance = d;
            }
        }
        chosen | if x.is_sign_negative() { 128 } else { 0 }
    }

    fn input(rows: usize, seed: usize) -> Vec<bf16> {
        let edge = [
            0.0,
            -0.0,
            0.0009765625,
            0.0029296875,
            1.0625,
            -1.1875,
            448.0,
            -512.0,
        ];
        (0..rows * 8 * 128)
            .map(|i| {
                bf16::from_f32(if i % 37 == 0 {
                    edge[(i / 37 + seed) % edge.len()]
                } else {
                    ((i * 17 + seed * 31) % 997) as f32 / 79.0 - 6.0
                })
            })
            .collect()
    }

    fn eager_append(cache: &mut LagunaKvCacheFp8, k: &[bf16], v: &[bf16]) -> Result<()> {
        let rows = k.len() / (8 * 128);
        let kt = Tensor::from_vec(k.to_vec(), (1, rows, 8, 128), cache.device())?;
        let vt = Tensor::from_vec(v.to_vec(), (1, rows, 8, 128), cache.device())?;
        cache
            .dev
            .cuda_stream()
            .synchronize()
            .map_err(|e| anyhow::anyhow!(e))?;
        let start = cache.current_len();
        cache.prepare_for_decode_dev(start, start + rows)?;
        cache.write_at_impl(0, &kt, &vt)?;
        cache.advance(rows);
        Ok(())
    }

    fn check_owned(
        cache: &mut LagunaKvCacheFp8,
        expected_k: &[bf16],
        expected_v: &[bf16],
    ) -> Result<()> {
        assert_eq!(cache.scale_policy(), LagunaFp8ScalePolicy::FixedOne);
        let rows = cache.current_len();
        assert_eq!(expected_k.len(), rows * 8 * 128);
        assert_eq!(expected_v.len(), expected_k.len());
        let stream = nv_layers::cuda_stream::current_stream(&cache.dev);
        stream.synchronize().map_err(|e| anyhow::anyhow!(e))?;
        let Fp8LayerSlot::Full {
            k_fp8,
            v_fp8,
            k_scales,
            v_scales,
        } = &cache.layers[0]
        else {
            unreachable!()
        };
        for (packed, scales, expected) in
            [(k_fp8, k_scales, expected_k), (v_fp8, v_scales, expected_v)]
        {
            let bytes = stream
                .clone_dtoh(&packed.slice(..rows * 8 * 128))
                .map_err(|e| anyhow::anyhow!(e))?;
            let scales = stream
                .clone_dtoh(&scales.slice(..rows * 8))
                .map_err(|e| anyhow::anyhow!(e))?;
            assert!(
                scales
                    .iter()
                    .all(|scale| scale.to_bits() == 1.0f32.to_bits()),
                "every initialized head scale must be exactly one"
            );
            let oracle: Vec<u8> = expected.iter().copied().map(encode_reference).collect();
            assert_eq!(
                bytes, oracle,
                "unit writer must match independent nearest-even E4M3 encoding"
            );
        }
        let (k, v) = cache.view_bf16(0, rows)?;
        stream.synchronize().map_err(|e| anyhow::anyhow!(e))?;
        for (tensor, expected) in [(k, expected_k), (v, expected_v)] {
            let output = tensor.flatten_all()?.to_vec1::<bf16>()?;
            let oracle: Vec<u16> = expected
                .iter()
                .map(|value| {
                    let code = encode_reference(*value);
                    let decoded =
                        decode_positive_e4m3(code & 127) * if code & 128 != 0 { -1.0 } else { 1.0 };
                    bf16::from_f32(decoded).to_bits()
                })
                .collect();
            assert_eq!(
                output
                    .iter()
                    .copied()
                    .map(bf16::to_bits)
                    .collect::<Vec<_>>(),
                oracle,
                "existing dequantization must read unit-scale metadata"
            );
        }
        Ok(())
    }

    #[test]
    #[ignore]
    fn fixed_one_eager_and_captured_writers_preserve_cache_policy_and_replay_positions(
    ) -> Result<()> {
        anyhow::ensure!(
            std::env::var("NV_LAGUNA_FP8_UNIT_SCALE").as_deref() == Ok("1"),
            "set NV_LAGUNA_FP8_UNIT_SCALE=1; this CUDA gate never skips as a pass"
        );
        anyhow::ensure!(
            std::env::var_os("NV_LAGUNA_FP8_BF16_FULL").is_none(),
            "BF16 full-layer overrides mask the unit writer under test"
        );
        let config = LagunaConfig::from_hf_json_str(
            r#"{
            "vocab_size":32,"hidden_size":6144,"intermediate_size":64,
            "num_hidden_layers":1,"num_attention_heads":48,"num_key_value_heads":8,
            "head_dim":128,"max_position_embeddings":64,"rms_norm_eps":0.00001,
            "sliding_window":4,"layer_types":["full_attention"],
            "num_experts":0,"num_experts_per_tok":0,"moe_intermediate_size":0,
            "shared_expert_intermediate_size":0,
            "rope_parameters":{"full_attention":{"rope_theta":500000},"sliding_attention":{"rope_theta":500000}}
        }"#,
        )?;
        let device = Device::new_cuda(0)?;
        assert!(
            crate::laguna::LagunaKvCache::new(&config, 64, &device, DType::BF16).is_err(),
            "the explicit arm must refuse a BF16 cache before allocation"
        );
        let Device::Cuda(dev) = &device else {
            unreachable!()
        };
        let stream = dev
            .cuda_stream()
            .context()
            .new_stream()
            .map_err(|e| anyhow::anyhow!(e))?;
        nv_layers::cuda_stream::with_stream(stream.clone(), || -> Result<()> {
            let mut eager = LagunaKvCacheFp8::new(&config, 64, &device, DType::BF16)?;
            let mut captured = LagunaKvCacheFp8::new(&config, 64, &device, DType::BF16)?;
            let mut expected_k = input(7, 1);
            let mut expected_v = input(7, 2);
            for cache in [&mut eager, &mut captured] {
                eager_append(cache, &expected_k, &expected_v)?;
            }
            for rows in [1usize, 5] {
                let mut k_in = stream
                    .alloc_zeros::<bf16>(rows * 8 * 128)
                    .map_err(|e| anyhow::anyhow!(e))?;
                let mut v_in = stream
                    .alloc_zeros::<bf16>(rows * 8 * 128)
                    .map_err(|e| anyhow::anyhow!(e))?;
                let mut runner = CudaGraphRunner::new(stream.clone());
                let (kp, vp, ks, vs) = captured.layer_fp8_ptrs(0).unwrap();
                let meta = captured.ring_meta_ptr();
                let ki = {
                    let (ptr, _guard) = k_in.device_ptr(&stream);
                    ptr
                };
                let vi = {
                    let (ptr, _guard) = v_in.device_ptr(&stream);
                    ptr
                };
                for seed in [11, 17, 29] {
                    let k = input(rows, seed);
                    let v = input(rows, seed + 3);
                    stream
                        .memcpy_htod(&k, &mut k_in)
                        .map_err(|e| anyhow::anyhow!(e))?;
                    stream
                        .memcpy_htod(&v, &mut v_in)
                        .map_err(|e| anyhow::anyhow!(e))?;
                    let start = captured.current_len();
                    captured.prepare_for_decode_dev(start, start + rows)?;
                    runner.run(rows as u64, |s| {
                        let rc = unsafe {
                            nv_kernels::cuda::kv_append_fp8_unit_scale(
                                s.cu_stream() as *mut c_void,
                                ki as *const u16,
                                vi as *const u16,
                                kp as *mut u8,
                                vp as *mut u8,
                                ks as *mut f32,
                                vs as *mut f32,
                                meta as *const i32,
                                rows as i32,
                                8,
                                128,
                                0,
                            )
                        };
                        anyhow::ensure!(rc == 0, "captured unit append rc={rc}");
                        Ok(())
                    })?;
                    stream.synchronize().map_err(|e| anyhow::anyhow!(e))?;
                    assert!(runner.has_cached_token(rows as u64) && runner.cached_node_count() > 0);
                    captured.advance(rows);
                    captured.note_graph_write();
                    eager_append(&mut eager, &k, &v)?;
                    expected_k.extend(k);
                    expected_v.extend(v);
                    for cache in [&mut eager, &mut captured] {
                        check_owned(cache, &expected_k, &expected_v)?;
                    }
                    if seed == 17 {
                        for cache in [&mut eager, &mut captured] {
                            cache.rollback(rows)?;
                        }
                        expected_k.truncate(expected_k.len() - rows * 8 * 128);
                        expected_v.truncate(expected_v.len() - rows * 8 * 128);
                    }
                }
                stream.synchronize().map_err(|e| anyhow::anyhow!(e))?;
                runner.invalidate();
                stream.synchronize().map_err(|e| anyhow::anyhow!(e))?;
            }
            for cache in [&mut eager, &mut captured] {
                let start = cache.current_len();
                assert!(cache.prepare_for_decode_dev(start, 65).is_err());
                check_owned(cache, &expected_k, &expected_v)?;
                cache.reset();
                eager_append(cache, &input(3, 41), &input(3, 43))?;
                check_owned(cache, &input(3, 41), &input(3, 43))?;
            }
            stream.synchronize().map_err(|e| anyhow::anyhow!(e))?;
            eprintln!("LAGUNA-FP8-UNIT-WRITERS real_cuda=true eager=true captured_m1_m5=true changed_positions=true rollback_overwrite=true reset_reuse=true exact_bytes=true exact_unit_metadata=true model_quality=false");
            Ok(())
        })
    }
    fn prefill_matches_owned_fp8_oracle(
        cache: &mut LagunaKvCacheFp8,
        rows: usize,
        seed: usize,
    ) -> Result<Vec<u16>> {
        let n = cache.current_len();
        let q_host: Vec<bf16> = (0..rows * 48 * 128)
            .map(|i| bf16::from_f32(((i * 13 + seed * 7) % 71) as f32 / 97.0 - 0.35))
            .collect();
        let q = Tensor::from_vec(q_host.clone(), (1, rows, 48, 128), cache.device())?;
        cache
            .dev
            .cuda_stream()
            .synchronize()
            .map_err(|e| anyhow::anyhow!(e))?;
        let before = cache.direct_fp8_prefill_enqueued_calls();
        let surface: &mut dyn Gemma4Cache = cache;
        let actual = surface
            .try_prefill_attention_fp8(0, &q, 48, rows, None, 1.0 / 128.0f32.sqrt())?
            .ok_or_else(|| {
                anyhow::anyhow!(
                    "the native serving cache hook silently missed the direct FP8 route"
                )
            })?;
        let stream = nv_layers::cuda_stream::current_stream(&cache.dev);
        stream.synchronize().map_err(|e| anyhow::anyhow!(e))?;
        assert_eq!(cache.direct_fp8_prefill_enqueued_calls(), before + 1);
        let Fp8LayerSlot::Full { k_fp8, v_fp8, .. } = &cache.layers[0] else {
            unreachable!()
        };
        let k = stream
            .clone_dtoh(&k_fp8.slice(..n * 8 * 128))
            .map_err(|e| anyhow::anyhow!(e))?;
        let v = stream
            .clone_dtoh(&v_fp8.slice(..n * 8 * 128))
            .map_err(|e| anyhow::anyhow!(e))?;
        let decode = |code: u8| -> f64 {
            assert_ne!(
                code & 127,
                127,
                "the finite-input oracle must not consume an E4M3 NaN"
            );
            (decode_positive_e4m3(code & 127) * if code & 128 != 0 { -1.0 } else { 1.0 }) as f64
        };
        let got = actual.flatten_all()?.to_vec1::<bf16>()?;
        for row in 0..rows {
            let visible = n - rows + row + 1;
            for head in 0..48 {
                let kv_head = head / 6;
                let scores: Vec<f64> = (0..visible)
                    .map(|token| {
                        (0..128)
                            .map(|d| {
                                q_host[(row * 48 + head) * 128 + d].to_f32() as f64
                                    * decode(k[(token * 8 + kv_head) * 128 + d])
                            })
                            .sum::<f64>()
                            / 128.0f64.sqrt()
                    })
                    .collect();
                let max_score = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                let weights: Vec<f64> = scores
                    .iter()
                    .map(|score| (score - max_score).exp())
                    .collect();
                let denominator: f64 = weights.iter().sum();
                for d in 0..128 {
                    let expected = weights
                        .iter()
                        .enumerate()
                        .map(|(token, weight)| weight * decode(v[(token * 8 + kv_head) * 128 + d]))
                        .sum::<f64>()
                        / denominator;
                    let index = (row * 48 + head) * 128 + d;
                    let observed = got[index].to_f32() as f64;
                    assert!(observed.is_finite() && expected.is_finite() && (observed - expected).abs() <= 0.002 + 0.01 * expected.abs(),
                        "direct FP8 causal attention mismatch at n={n} M={rows} row={row} head={head} d={d}: observed={observed} expected={expected}");
                }
            }
        }
        Ok(got.into_iter().map(bf16::to_bits).collect())
    }

    #[test]
    #[ignore]
    fn fixed_one_prefill_serving_hook_checks_written_span_and_graph_reuse() -> Result<()> {
        anyhow::ensure!(std::env::var("NV_LAGUNA_FP8_UNIT_SCALE").as_deref() == Ok("1") && std::env::var("NV_LAGUNA_PREFILL_FLASHINFER").as_deref() == Ok("1"),
            "set NV_LAGUNA_FP8_UNIT_SCALE=1 NV_LAGUNA_PREFILL_FLASHINFER=1; this real CUDA gate never skips as a pass");
        anyhow::ensure!(
            std::env::var_os("NV_LAGUNA_FP8_BF16_FULL").is_none(),
            "BF16 overrides mask the direct FP8 cache path under test"
        );
        let config = LagunaConfig::from_hf_json_str(
            r#"{
            "vocab_size":32,"hidden_size":6144,"intermediate_size":64,
            "num_hidden_layers":1,"num_attention_heads":48,"num_key_value_heads":8,
            "head_dim":128,"max_position_embeddings":262144,"rms_norm_eps":0.00001,
            "sliding_window":4,"layer_types":["full_attention"],
            "num_experts":0,"num_experts_per_tok":0,"moe_intermediate_size":0,
            "shared_expert_intermediate_size":0,
            "rope_parameters":{"full_attention":{"rope_theta":500000},"sliding_attention":{"rope_theta":500000}}
        }"#,
        )?;
        let device = Device::new_cuda(0)?;
        let Device::Cuda(dev) = &device else {
            unreachable!()
        };
        let stream = dev
            .cuda_stream()
            .context()
            .new_stream()
            .map_err(|e| anyhow::anyhow!(e))?;
        nv_layers::cuda_stream::with_stream(stream.clone(), || -> Result<()> {
            let mut cache = LagunaKvCacheFp8::new(&config, 262144, &device, DType::BF16)?;
            let q = Tensor::zeros((1, 7, 48, 128), DType::BF16, &device)?;
            dev.cuda_stream()
                .synchronize()
                .map_err(|e| anyhow::anyhow!(e))?;
            cache.prepare_for_decode_dev(0, 7)?;
            assert!(
                cache
                    .try_prefill_attention_fp8(0, &q, 48, 7, None, 1.0 / 128.0f32.sqrt())
                    .is_err(),
                "prepared but unwritten cache span must fail before attention"
            );
            assert_eq!(cache.direct_fp8_prefill_enqueued_calls(), 0);
            cache.reset();
            let values = |rows: usize, seed: usize| -> Vec<bf16> {
                (0..rows * 8 * 128)
                    .map(|i| bf16::from_f32(((i * 11 + seed * 17) % 137) as f32 / 103.0 - 0.65))
                    .collect()
            };
            eager_append(&mut cache, &values(7, 1), &values(7, 2))?;
            let first = prefill_matches_owned_fp8_oracle(&mut cache, 7, 3)?;
            assert_eq!(
                first,
                prefill_matches_owned_fp8_oracle(&mut cache, 7, 3)?,
                "same plan/cache replay must preserve owned output bytes"
            );
            let pointers = cache.layer_fp8_ptrs(0).unwrap();
            let (kp, vp, ks, vs) = pointers;
            let meta = cache.ring_meta_ptr();
            let mut k_in = stream
                .alloc_zeros::<bf16>(5 * 8 * 128)
                .map_err(|e| anyhow::anyhow!(e))?;
            let mut v_in = stream
                .alloc_zeros::<bf16>(5 * 8 * 128)
                .map_err(|e| anyhow::anyhow!(e))?;
            let ki = {
                let (ptr, _guard) = k_in.device_ptr(&stream);
                ptr
            };
            let vi = {
                let (ptr, _guard) = v_in.device_ptr(&stream);
                ptr
            };
            let mut graph = CudaGraphRunner::new(stream.clone());
            for seed in [7usize, 13, 19] {
                let k_host = values(5, seed);
                let v_host = values(5, seed + 1);
                stream
                    .memcpy_htod(&k_host, &mut k_in)
                    .map_err(|e| anyhow::anyhow!(e))?;
                stream
                    .memcpy_htod(&v_host, &mut v_in)
                    .map_err(|e| anyhow::anyhow!(e))?;
                let start = cache.current_len();
                cache.prepare_for_decode_dev(start, start + 5)?;
                graph.run(5, |s| {
                    let rc = unsafe {
                        nv_kernels::cuda::kv_append_fp8_unit_scale(
                            s.cu_stream() as *mut c_void,
                            ki as *const u16,
                            vi as *const u16,
                            kp as *mut u8,
                            vp as *mut u8,
                            ks as *mut f32,
                            vs as *mut f32,
                            meta as *const i32,
                            5,
                            8,
                            128,
                            0,
                        )
                    };
                    anyhow::ensure!(rc == 0, "captured unit append failed rc={rc}");
                    Ok(())
                })?;
                cache.note_graph_write();
                cache.advance(5);
                assert!(graph.has_cached_token(5) && graph.cached_node_count() > 0);
                prefill_matches_owned_fp8_oracle(&mut cache, 5, seed)?;
                if seed == 13 {
                    cache.rollback(5)?;
                }
            }
            stream.synchronize().map_err(|e| anyhow::anyhow!(e))?;
            graph.invalidate();
            stream.synchronize().map_err(|e| anyhow::anyhow!(e))?;
            cache.reset();
            eager_append(&mut cache, &values(3, 23), &values(3, 29))?;
            prefill_matches_owned_fp8_oracle(&mut cache, 3, 31)?;
            assert_eq!(
                cache.layer_fp8_ptrs(0).unwrap(),
                pointers,
                "reset/rollback must retain the original cache allocations"
            );
            stream.synchronize().map_err(|e| anyhow::anyhow!(e))?;
            eprintln!("LAGUNA-FP8-PREFILL-INTEGRATION real_cuda=true actual_capacity=262144 trait_hook=true eager_and_captured_writes=true changed_prefix=true rollback_overwrite=true reset_reuse=true independent_f64_oracle=true atol=0.002 rtol=0.01 enqueued_calls={} deep_168000=false model_quality=false", cache.direct_fp8_prefill_enqueued_calls());
            Ok(())
        })
    }
    fn handoff_values(start: usize, rows: usize, seed: usize) -> Vec<bf16> {
        let edge = [
            0.0,
            -0.0,
            0.0009765625,
            0.0029296875,
            1.0625,
            -1.1875,
            448.0,
            -512.0,
        ];
        (start * 1024..(start + rows) * 1024)
            .map(|i| {
                bf16::from_f32(if i % 37 == 0 {
                    edge[(i / 37 + seed) % edge.len()]
                } else {
                    ((i * 17 + seed * 31) % 997) as f32 / 79.0 - 6.0
                })
            })
            .collect()
    }

    fn write_bf16_handoff_fixture(
        source: &mut crate::laguna::LagunaKvCache,
        device: &Device,
        rows: usize,
        seed: usize,
        chunk: usize,
    ) -> Result<()> {
        source.reset();
        for start in (0..rows).step_by(chunk) {
            let n = (rows - start).min(chunk);
            source.prepare_for_decode(start, start + n)?;
            for layer in 0..2 {
                let k = Tensor::from_vec(
                    handoff_values(start, n, seed + layer * 2),
                    (1, n, 8, 128),
                    device,
                )?;
                let v = Tensor::from_vec(
                    handoff_values(start, n, seed + layer * 2 + 1),
                    (1, n, 8, 128),
                    device,
                )?;
                source.write_at(layer, &k, &v)?;
                device.synchronize()?;
            }
            source.advance(n);
        }
        Ok(())
    }

    fn check_handoff_fixture(cache: &LagunaKvCacheFp8, rows: usize, seed: usize) -> Result<()> {
        let stream = cache.dev.cuda_stream();
        assert_eq!(cache.current_len, rows);
        assert_eq!(cache.full_stored, rows);
        assert_eq!(cache.layer_stored, vec![rows, cache.s_stored]);
        assert_eq!(
            stream
                .clone_dtoh(&cache.meta_dev)
                .map_err(|e| anyhow::anyhow!(e))?,
            vec![rows as i32, cache.s_stored as i32, 0, cache.s_stored as i32]
        );
        assert_eq!(
            stream
                .clone_dtoh(&cache.n_total_dev)
                .map_err(|e| anyhow::anyhow!(e))?,
            vec![rows as i32]
        );
        let mut oracle = std::collections::HashMap::new();
        let Fp8LayerSlot::Full {
            k_fp8,
            v_fp8,
            k_scales,
            v_scales,
        } = &cache.layers[0]
        else {
            unreachable!()
        };
        for (packed, scales, offset) in [(k_fp8, k_scales, 0), (v_fp8, v_scales, 1)] {
            for start in (0..rows).step_by(1024) {
                let n = (rows - start).min(1024);
                let expected: Vec<u8> = handoff_values(start, n, seed + offset)
                    .iter()
                    .map(|value| {
                        *oracle
                            .entry(value.to_bits())
                            .or_insert_with(|| encode_reference(*value))
                    })
                    .collect();
                let actual = stream
                    .clone_dtoh(&packed.slice(start * 1024..(start + n) * 1024))
                    .map_err(|e| anyhow::anyhow!(e))?;
                assert_eq!(
                    actual, expected,
                    "entire initialized full layer must match independent E4M3 oracle"
                );
            }
            let actual = stream
                .clone_dtoh(&scales.slice(..rows * 8))
                .map_err(|e| anyhow::anyhow!(e))?;
            assert!(actual
                .iter()
                .all(|value| value.to_bits() == 1.0f32.to_bits()));
        }
        let Fp8LayerSlot::Bf16 { k, v } = &cache.layers[1] else {
            unreachable!()
        };
        for (tensor, offset) in [(k, 2), (v, 3)] {
            let actual = tensor
                .narrow(1, 0, cache.s_stored)?
                .flatten_all()?
                .to_vec1::<bf16>()?;
            let expected = handoff_values(rows - cache.s_stored, cache.s_stored, seed + offset);
            assert_eq!(
                actual
                    .iter()
                    .map(|value| value.to_bits())
                    .collect::<Vec<_>>(),
                expected
                    .iter()
                    .map(|value| value.to_bits())
                    .collect::<Vec<_>>(),
                "absolute sliding suffix must remain BF16-bitwise"
            );
        }
        Ok(())
    }

    fn handoff_allocation_pointers(cache: &LagunaKvCacheFp8) -> Vec<u64> {
        let stream = cache.dev.cuda_stream();
        let mut pointers = vec![
            cache.meta_dev.device_ptr(&stream).0,
            cache.n_total_dev.device_ptr(&stream).0,
        ];
        for layer in &cache.layers {
            match layer {
                Fp8LayerSlot::Full {
                    k_fp8,
                    v_fp8,
                    k_scales,
                    v_scales,
                } => pointers.extend([
                    k_fp8.device_ptr(&stream).0,
                    v_fp8.device_ptr(&stream).0,
                    k_scales.device_ptr(&stream).0,
                    v_scales.device_ptr(&stream).0,
                ]),
                Fp8LayerSlot::Bf16 { k, v } => {
                    for tensor in [k, v] {
                        let (storage, layout) = tensor.storage_and_layout();
                        let candle_core::Storage::Cuda(storage) = &*storage else {
                            unreachable!()
                        };
                        pointers.push(
                            storage
                                .as_cuda_slice::<bf16>()
                                .unwrap()
                                .slice(layout.start_offset()..)
                                .device_ptr(&stream)
                                .0,
                        );
                    }
                }
            }
        }
        if let Some(scratch) = &cache.scores_scratch {
            pointers.push(scratch.device_ptr(&stream).0);
        }
        pointers
    }

    fn handoff_refusal_sentinels(cache: &LagunaKvCacheFp8) -> Result<Vec<u32>> {
        let stream = cache.dev.cuda_stream();
        let mut values: Vec<u32> = stream
            .clone_dtoh(&cache.meta_dev)
            .map_err(|e| anyhow::anyhow!(e))?
            .into_iter()
            .map(|value| value as u32)
            .collect();
        values.extend(
            stream
                .clone_dtoh(&cache.n_total_dev)
                .map_err(|e| anyhow::anyhow!(e))?
                .into_iter()
                .map(|value| value as u32),
        );
        for layer in &cache.layers {
            match layer {
                Fp8LayerSlot::Full {
                    k_fp8,
                    v_fp8,
                    k_scales,
                    v_scales,
                } => {
                    for packed in [k_fp8, v_fp8] {
                        values.extend(
                            stream
                                .clone_dtoh(&packed.slice(..1024))
                                .map_err(|e| anyhow::anyhow!(e))?
                                .into_iter()
                                .map(u32::from),
                        );
                    }
                    for scales in [k_scales, v_scales] {
                        values.extend(
                            stream
                                .clone_dtoh(&scales.slice(..8))
                                .map_err(|e| anyhow::anyhow!(e))?
                                .into_iter()
                                .map(f32::to_bits),
                        );
                    }
                }
                Fp8LayerSlot::Bf16 { k, v } => {
                    for tensor in [k, v] {
                        values.extend(
                            tensor
                                .narrow(1, 0, 1)?
                                .flatten_all()?
                                .to_vec1::<bf16>()?
                                .into_iter()
                                .map(|value| value.to_bits() as u32),
                        );
                    }
                }
            }
        }
        Ok(values)
    }

    #[test]
    #[ignore]
    fn bf16_to_fixed_one_handoff_preserves_written_spans_and_graph_allocations() -> Result<()> {
        anyhow::ensure!(
            std::env::var("NV_LAGUNA_KV_HANDOFF_TEST").as_deref() == Ok("1"),
            "set NV_LAGUNA_KV_HANDOFF_TEST=1; this real CUDA gate never skips"
        );
        anyhow::ensure!(
            std::env::var_os("NV_LAGUNA_HOST_KV").is_none()
                && std::env::var_os("NV_LAGUNA_FP8_BF16_FULL").is_none()
                && LagunaFp8ScalePolicy::from_env()? == LagunaFp8ScalePolicy::PerTokenHead,
            "unset HOST_KV and FP8_BF16_FULL; UNIT_SCALE must be unset/0 for BF16 construction"
        );
        let chunk = match std::env::var("NV_LAGUNA_PREFILL_CHUNK") {
            Ok(value) => value.parse::<usize>()?,
            Err(std::env::VarError::NotPresent) => 1024,
            Err(error) => return Err(error.into()),
        };
        anyhow::ensure!(
            matches!(chunk, 256 | 512 | 1024 | 2048 | 4096 | 8192),
            "invalid handoff fixture chunk"
        );
        let depth = match std::env::var("NV_CTX_TOKENS") {
            Ok(value) => value.parse::<usize>()?,
            Err(std::env::VarError::NotPresent) => 2 * chunk + 9,
            Err(error) => return Err(error.into()),
        };
        anyhow::ensure!(
            (2 * chunk + 1..=168000).contains(&depth),
            "handoff gate depth must exercise sliding compaction and remain <=168000"
        );
        let config = LagunaConfig::from_hf_json_str(
            r#"{
            "vocab_size":32,"hidden_size":6144,"intermediate_size":64,
            "num_hidden_layers":2,"num_attention_heads":48,"num_key_value_heads":8,
            "head_dim":128,"max_position_embeddings":262144,"rms_norm_eps":0.00001,
            "sliding_window":512,"layer_types":["full_attention","sliding_attention"],
            "num_experts":0,"num_experts_per_tok":0,"moe_intermediate_size":0,
            "shared_expert_intermediate_size":0,
            "rope_parameters":{"full_attention":{"rope_theta":500000},"sliding_attention":{"rope_theta":500000}}
        }"#,
        )?;
        let device = Device::new_cuda(0)?;
        let Device::Cuda(dev) = &device else {
            unreachable!()
        };
        let stream = dev.cuda_stream();
        let mut source = crate::laguna::LagunaKvCache::new_with_prefill_chunk(
            &config,
            262144,
            &device,
            DType::BF16,
            chunk,
        )?;
        let mut destination =
            LagunaKvCacheFp8::new_fixed_one_with_prefill_chunk(&config, 262144, &device, chunk)?;
        let original_pointers = handoff_allocation_pointers(&destination);
        write_bf16_handoff_fixture(&mut source, &device, depth, 11, chunk)?;
        destination.copy_from_bf16(&mut source)?;
        check_handoff_fixture(&destination, depth, 11)?;
        assert_eq!(handoff_allocation_pointers(&destination), original_pointers);
        assert!(destination
            .copy_from_bf16(&mut source)
            .unwrap_err()
            .to_string()
            .contains("empty or reset"));
        check_handoff_fixture(&destination, depth, 11)?;
        destination.reset();
        let saved_sentinels = handoff_refusal_sentinels(&destination)?;
        source.prepare_for_decode(depth, depth + 1)?;
        assert!(destination
            .copy_from_bf16(&mut source)
            .unwrap_err()
            .to_string()
            .contains("completed"));
        assert_eq!(handoff_refusal_sentinels(&destination)?, saved_sentinels);
        assert_eq!(destination.current_len, 0);
        source.advance(1);
        assert!(destination
            .copy_from_bf16(&mut source)
            .unwrap_err()
            .to_string()
            .contains("not initialized"));
        assert_eq!(handoff_refusal_sentinels(&destination)?, saved_sentinels);
        write_bf16_handoff_fixture(&mut source, &device, 33, 29, chunk)?;
        destination.max_seq_len -= 1;
        assert!(destination
            .copy_from_bf16(&mut source)
            .unwrap_err()
            .to_string()
            .contains("geometry"));
        destination.max_seq_len += 1;
        destination.scale_policy = LagunaFp8ScalePolicy::PerTokenHead;
        assert!(destination
            .copy_from_bf16(&mut source)
            .unwrap_err()
            .to_string()
            .contains("FixedOne"));
        destination.scale_policy = LagunaFp8ScalePolicy::FixedOne;
        assert_eq!(handoff_refusal_sentinels(&destination)?, saved_sentinels);
        let forked = stream
            .context()
            .new_stream()
            .map_err(|e| anyhow::anyhow!(e))?;
        let wrong_stream =
            nv_layers::cuda_stream::with_stream(forked, || destination.copy_from_bf16(&mut source));
        assert!(wrong_stream
            .unwrap_err()
            .to_string()
            .contains("legacy stream"));
        assert_eq!(handoff_refusal_sentinels(&destination)?, saved_sentinels);
        for rows in [1usize, 5] {
            destination.reset();
            let input = Tensor::from_vec(handoff_values(33, rows, 71), (1, rows, 8, 128), &device)?;
            device.synchronize()?;
            let graph_stream = stream
                .context()
                .new_stream()
                .map_err(|e| anyhow::anyhow!(e))?;
            let mut graph = CudaGraphRunner::new(graph_stream.clone());
            let pointers = handoff_allocation_pointers(&destination);
            let sliding_cap = destination.s_cap;
            let input_ptr = {
                let (storage, layout) = input.storage_and_layout();
                let candle_core::Storage::Cuda(storage) = &*storage else {
                    unreachable!()
                };
                let slice = storage.as_cuda_slice::<bf16>()?;
                let view = slice.slice(layout.start_offset()..);
                let (ptr, _guard) = view.device_ptr(&stream);
                ptr
            };
            destination.prepare_for_decode_dev(0, rows)?;
            stream.synchronize().map_err(|e| anyhow::anyhow!(e))?;
            graph.run(rows as u64, |s| {
                let rc = unsafe {
                    nv_kernels::cuda::kv_append_fp8_unit_scale(
                        s.cu_stream() as *mut c_void,
                        input_ptr as *const u16,
                        input_ptr as *const u16,
                        pointers[2] as *mut u8,
                        pointers[3] as *mut u8,
                        pointers[4] as *mut f32,
                        pointers[5] as *mut f32,
                        pointers[0] as *const i32,
                        rows as i32,
                        8,
                        128,
                        0,
                    )
                };
                anyhow::ensure!(rc == 0, "handoff graph FP8 append rc={rc}");
                for ptr in &pointers[6..8] {
                    let rc = unsafe {
                        nv_kernels::cuda::kv_ring_append_bf16(
                            s.cu_stream() as *mut c_void,
                            input_ptr as *const u16,
                            *ptr as *mut u16,
                            (pointers[0] as *const i32).add(1),
                            rows as i32,
                            sliding_cap as i32,
                            8,
                            128,
                        )
                    };
                    anyhow::ensure!(rc == 0, "handoff graph sliding append rc={rc}");
                }
                Ok(())
            })?;
            graph_stream.synchronize().map_err(|e| anyhow::anyhow!(e))?;
            assert!(graph.has_cached_token(rows as u64) && graph.cached_node_count() > 0);
            destination.reset();
            destination.copy_from_bf16(&mut source)?;
            check_handoff_fixture(&destination, 33, 29)?;
            destination.prepare_for_decode_dev(33, 33 + rows)?;
            stream.synchronize().map_err(|e| anyhow::anyhow!(e))?;
            graph.run(rows as u64, |_| {
                anyhow::bail!("retained graph unexpectedly recaptured after handoff")
            })?;
            graph_stream.synchronize().map_err(|e| anyhow::anyhow!(e))?;
            destination.advance(rows);
            destination.note_graph_write();
            assert_eq!(destination.current_len, 33 + rows);
            assert_eq!(destination.layer_stored, vec![33 + rows, 33 + rows]);
            let Fp8LayerSlot::Full { k_fp8, .. } = &destination.layers[0] else {
                unreachable!()
            };
            let actual = stream
                .clone_dtoh(&k_fp8.slice(33 * 1024..(33 + rows) * 1024))
                .map_err(|e| anyhow::anyhow!(e))?;
            assert_eq!(
                actual,
                handoff_values(33, rows, 71)
                    .into_iter()
                    .map(encode_reference)
                    .collect::<Vec<_>>()
            );
            let Fp8LayerSlot::Bf16 { k, .. } = &destination.layers[1] else {
                unreachable!()
            };
            assert_eq!(
                k.narrow(1, 33, rows)?
                    .flatten_all()?
                    .to_vec1::<bf16>()?
                    .iter()
                    .map(|v| v.to_bits())
                    .collect::<Vec<_>>(),
                handoff_values(33, rows, 71)
                    .iter()
                    .map(|v| v.to_bits())
                    .collect::<Vec<_>>()
            );
            assert_eq!(handoff_allocation_pointers(&destination), original_pointers);
            graph.invalidate();
            stream
                .context()
                .synchronize()
                .map_err(|e| anyhow::anyhow!(e))?;
        }
        eprintln!("LAGUNA-KV-HANDOFF real_cuda=true physical_capacity=262144 depth={depth} chunk={chunk} full_layers=1 sliding_layers=1 source=bf16 destination=fixed_one_e4m3 sliding=bf16 all_initialized_fp8_bytes=exact independent_nearest_even_oracle=true reset_reuse=true captured_writer_m1_m5=true allocation_pointers=unchanged generated_tokens=0 model_quality=false timing_measured=false");
        Ok(())
    }
}
