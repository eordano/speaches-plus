use anyhow::Result;
use candle_core::{DType, Device, Tensor};
use cudarc::driver::{CudaSlice, DevicePtr, DevicePtrMut};
use half::bf16;
use std::ffi::c_void;

use crate::gemma4::{Gemma4Cache, LayerType};
use crate::laguna::{ring_append_launch, shift_rows_launch, LagunaConfig, SLIDING_COMPACT_SLACK};

const SMEM_LIMIT_BYTES: usize = 48 * 1024;

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
}

impl LagunaKvCacheFp8 {
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
                    let cap = max_seq_len.min(window + SLIDING_COMPACT_SLACK);
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
        Ok(Self {
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
        })
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

                let s_raw = stream.cu_stream() as *mut c_void;
                let rc_k = unsafe {
                    nv_kernels::cuda::quantize_kv_fp8(
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
                    nv_kernels::cuda::quantize_kv_fp8(
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

                {
                    let (k_fp8_ptr, _gk) = k_fp8.device_ptr(&stream);
                    let (v_fp8_ptr, _gv) = v_fp8.device_ptr(&stream);
                    let (k_sc_ptr, _gks) = k_scales.device_ptr(&stream);
                    let (v_sc_ptr, _gvs) = v_scales.device_ptr(&stream);
                    let (k_out_ptr, _gko) = k_out.device_ptr_mut(&stream);
                    let (v_out_ptr, _gvo) = v_out.device_ptr_mut(&stream);
                    let s_raw = stream.cu_stream() as *mut c_void;
                    let rc_k = unsafe {
                        nv_kernels::cuda::dequantize_kv_fp8(
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
                        nv_kernels::cuda::dequantize_kv_fp8(
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
    fn try_decode_attention_fp8(
        &mut self,
        layer: usize,
        q_rot: &Tensor,
        n_q: usize,
        sliding_window: Option<usize>,
        scaling: f32,
    ) -> Result<Option<Tensor>> {
        LagunaKvCacheFp8::decode_attention_fp8(self, layer, q_rot, n_q, sliding_window, scaling)
    }
}
