#![cfg(feature = "cuda")]

use anyhow::{Context, Result};
use candle_core::{DType, Device, Tensor};
use cudarc::driver::{CudaSlice, DevicePtr, DevicePtrMut};
use half::bf16;
use std::ffi::{c_char, c_void, CStr};

const ERROR_BYTES: usize = 256;
const NUM_Q_HEADS: usize = 48;
const NUM_KV_HEADS: usize = 8;
const HEAD_DIM: usize = 128;

#[repr(C)]
#[derive(Clone, Copy, Default)]
struct LagunaPrefillPlan {
    abi: u32,
    q_len: u32,
    kv_len: u32,
    capacity: u32,
    tile_q: u32,
    tile_kv: u32,
    splits: u32,
    ctas: u32,
    sm_count: u32,
    active_blocks_per_sm: u32,
    shared_bytes: u32,
    registers_per_thread: u32,
    scratch_bytes: u64,
    device: i32,
    stream: usize,
}

extern "C" {
    fn laguna_prefill_prepare(
        plan: *mut LagunaPrefillPlan,
        q_len: u32,
        kv_len: u32,
        capacity: u32,
        stream: *mut c_void,
        error: *mut c_char,
        error_bytes: usize,
    ) -> i32;
    fn laguna_prefill_run(
        plan: *const LagunaPrefillPlan,
        q: *const u16,
        k: *const u16,
        v: *const u16,
        output: *mut u16,
        scratch: *mut c_void,
        scratch_bytes: usize,
        stream: *mut c_void,
        error: *mut c_char,
        error_bytes: usize,
    ) -> i32;
}

fn adapter_error(error: &[c_char; ERROR_BYTES]) -> String {
    unsafe { CStr::from_ptr(error.as_ptr()) }
        .to_string_lossy()
        .into_owned()
}

#[derive(Default)]
pub struct LagunaPrefillAttention {
    plan: LagunaPrefillPlan,
    key: Option<(u32, u32, u32, usize)>,
    scratch: Option<CudaSlice<u8>>,
    scratch_bytes: usize,
    last_diagnostic: Option<(u32, u32, u32)>,
}

impl LagunaPrefillAttention {
    pub fn run(
        &mut self,
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        q_len: usize,
        kv_len: usize,
        capacity: usize,
    ) -> Result<Tensor> {
        anyhow::ensure!(
            q_len <= u32::MAX as usize
                && kv_len <= u32::MAX as usize
                && capacity <= u32::MAX as usize,
            "Laguna full-prefill attention extents exceed the native ABI"
        );
        anyhow::ensure!(
            q.dtype() == DType::BF16 && k.dtype() == DType::BF16 && v.dtype() == DType::BF16,
            "Laguna full-prefill attention requires BF16 Q/K/V, got {:?}/{:?}/{:?}",
            q.dtype(),
            k.dtype(),
            v.dtype()
        );
        anyhow::ensure!(
            q.is_contiguous() && k.is_contiguous() && v.is_contiguous(),
            "Laguna full-prefill attention requires contiguous Q/K/V"
        );
        anyhow::ensure!(
            q.dims() == [1, q_len, NUM_Q_HEADS, HEAD_DIM],
            "Laguna full-prefill attention Q shape mismatch: expected [1,{q_len},{NUM_Q_HEADS},{HEAD_DIM}], got {:?}",
            q.dims()
        );
        anyhow::ensure!(
            k.dims() == [1, capacity, NUM_KV_HEADS, HEAD_DIM]
                && v.dims() == [1, capacity, NUM_KV_HEADS, HEAD_DIM],
            "Laguna full-prefill attention K/V shape mismatch: expected [1,{capacity},{NUM_KV_HEADS},{HEAD_DIM}], got {:?}/{:?}",
            k.dims(),
            v.dims()
        );
        anyhow::ensure!(
            k.device().same_device(q.device()) && v.device().same_device(q.device()),
            "Laguna full-prefill attention requires Q/K/V on one CUDA device"
        );
        let dev = match q.device() {
            Device::Cuda(dev) => dev.clone(),
            _ => anyhow::bail!("Laguna full-prefill attention requires CUDA Q/K/V"),
        };
        let stream = nv_layers::cuda_stream::current_stream(&dev);
        let raw_stream = stream.cu_stream() as *mut c_void;
        if let Some((_, _, _, planned_stream)) = self.key {
            anyhow::ensure!(
                planned_stream == raw_stream as usize,
                "Laguna full-prefill attention cache cannot move between CUDA streams"
            );
        }
        let key = (
            q_len as u32,
            kv_len as u32,
            capacity as u32,
            raw_stream as usize,
        );
        if self.key != Some(key) {
            let mut plan = LagunaPrefillPlan::default();
            let mut error = [0 as c_char; ERROR_BYTES];
            let rc = unsafe {
                laguna_prefill_prepare(
                    &mut plan,
                    key.0,
                    key.1,
                    key.2,
                    raw_stream,
                    error.as_mut_ptr(),
                    error.len(),
                )
            };
            anyhow::ensure!(
                rc == 0,
                "Laguna full-prefill attention prepare failed: {}",
                adapter_error(&error)
            );
            let required = usize::try_from(plan.scratch_bytes)
                .context("Laguna full-prefill attention scratch does not fit usize")?
                .max(16);
            if required > self.scratch_bytes {
                self.scratch = Some(unsafe {
                    stream.alloc::<u8>(required).map_err(|e| {
                        anyhow::anyhow!("Laguna full-prefill scratch allocation: {e:?}")
                    })?
                });
                self.scratch_bytes = required;
            }
            let diagnostic = (plan.tile_q, plan.tile_kv, plan.splits);
            if self.last_diagnostic != Some(diagnostic) {
                eprintln!(
                    "[laguna_prefill_attn] M={} S={} capacity={} tile={}x{} splits={} ctas={} scratch_bytes={} occupancy={}",
                    plan.q_len,
                    plan.kv_len,
                    plan.capacity,
                    plan.tile_q,
                    plan.tile_kv,
                    plan.splits,
                    plan.ctas,
                    plan.scratch_bytes,
                    plan.active_blocks_per_sm
                );
                self.last_diagnostic = Some(diagnostic);
            }
            self.plan = plan;
            self.key = Some(key);
        }

        let mut output: CudaSlice<bf16> = unsafe {
            stream
                .alloc::<bf16>(q_len * NUM_Q_HEADS * HEAD_DIM)
                .map_err(|e| anyhow::anyhow!("Laguna full-prefill output allocation: {e:?}"))?
        };
        let scratch = self
            .scratch
            .as_mut()
            .context("Laguna full-prefill scratch was not allocated")?;
        let (q_storage, q_layout) = q.storage_and_layout();
        let (k_storage, k_layout) = k.storage_and_layout();
        let (v_storage, v_layout) = v.storage_and_layout();
        let q_cuda = match &*q_storage {
            candle_core::Storage::Cuda(storage) => storage,
            _ => anyhow::bail!("Laguna full-prefill Q storage is not CUDA"),
        };
        let k_cuda = match &*k_storage {
            candle_core::Storage::Cuda(storage) => storage,
            _ => anyhow::bail!("Laguna full-prefill K storage is not CUDA"),
        };
        let v_cuda = match &*v_storage {
            candle_core::Storage::Cuda(storage) => storage,
            _ => anyhow::bail!("Laguna full-prefill V storage is not CUDA"),
        };
        let q_view = q_cuda
            .as_cuda_slice::<bf16>()?
            .slice(q_layout.start_offset()..);
        let k_view = k_cuda
            .as_cuda_slice::<bf16>()?
            .slice(k_layout.start_offset()..);
        let v_view = v_cuda
            .as_cuda_slice::<bf16>()?
            .slice(v_layout.start_offset()..);
        let mut error = [0 as c_char; ERROR_BYTES];
        let rc = {
            let (q_ptr, _q_guard) = q_view.device_ptr(&stream);
            let (k_ptr, _k_guard) = k_view.device_ptr(&stream);
            let (v_ptr, _v_guard) = v_view.device_ptr(&stream);
            let (output_ptr, _output_guard) = output.device_ptr_mut(&stream);
            let (scratch_ptr, _scratch_guard) = scratch.device_ptr_mut(&stream);
            unsafe {
                laguna_prefill_run(
                    &self.plan,
                    q_ptr as *const u16,
                    k_ptr as *const u16,
                    v_ptr as *const u16,
                    output_ptr as *mut u16,
                    scratch_ptr as *mut c_void,
                    self.scratch_bytes,
                    raw_stream,
                    error.as_mut_ptr(),
                    error.len(),
                )
            }
        };
        anyhow::ensure!(
            rc == 0,
            "Laguna full-prefill attention launch failed: {}",
            adapter_error(&error)
        );
        let storage = candle_core::CudaStorage::wrap_cuda_slice(output, dev);
        Ok(Tensor::from_storage(
            candle_core::Storage::Cuda(storage),
            (1usize, q_len, NUM_Q_HEADS, HEAD_DIM),
            candle_core::op::BackpropOp::none(),
            false,
        ))
    }
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
struct LagunaPrefillFp8Plan {
    base: LagunaPrefillPlan,
    cache_elem_enum: u32,
    scale_mode: u32,
    k_scale: f32,
    v_scale: f32,
}

#[repr(C)]
struct LagunaPrefillFp8CacheView {
    cache_elem_enum: u32,
    scale_mode: u32,
    k_scale: f32,
    v_scale: f32,
    k: u64,
    v: u64,
    k_row_scales: u64,
    v_row_scales: u64,
}

extern "C" {
    fn laguna_prefill_fp8_prepare(
        plan: *mut LagunaPrefillFp8Plan,
        q_len: u32,
        kv_len: u32,
        capacity: u32,
        stream: *mut c_void,
        error: *mut c_char,
        error_bytes: usize,
    ) -> i32;
    fn laguna_prefill_fp8_run(
        plan: *const LagunaPrefillFp8Plan,
        q: *const u16,
        cache: *const LagunaPrefillFp8CacheView,
        output: *mut u16,
        scratch: *mut c_void,
        scratch_bytes: usize,
        stream: *mut c_void,
        error: *mut c_char,
        error_bytes: usize,
    ) -> i32;
}

#[derive(Default)]
pub(crate) struct LagunaPrefillFp8Attention {
    plan: LagunaPrefillFp8Plan,
    key: Option<(u32, u32, u32, usize)>,
    scratch: Option<CudaSlice<u8>>,
    scratch_bytes: usize,
    last_diagnostic: Option<(u32, u32, u32)>,
    enqueued_calls: u64,
}

impl LagunaPrefillFp8Attention {
    pub(crate) fn enqueued_calls(&self) -> u64 {
        self.enqueued_calls
    }

    pub(crate) fn run_fixed_one(
        &mut self,
        q: &Tensor,
        k: &CudaSlice<u8>,
        v: &CudaSlice<u8>,
        cache_device: &Device,
        q_len: usize,
        kv_len: usize,
        capacity: usize,
    ) -> Result<Tensor> {
        anyhow::ensure!(
            (1..=8192).contains(&q_len)
                && q_len <= kv_len
                && kv_len <= capacity
                && capacity <= 262144,
            "Laguna direct FP8 prefill requires 1<=M<=8192 and M<=S<=capacity<=262144"
        );
        anyhow::ensure!(
            q.dtype() == DType::BF16 && q.is_contiguous() && q.dims() == [1, q_len, NUM_Q_HEADS, HEAD_DIM],
            "Laguna direct FP8 prefill requires contiguous BF16 Q [1,{q_len},{NUM_Q_HEADS},{HEAD_DIM}]"
        );
        anyhow::ensure!(
            q.device().same_device(cache_device)
                && k.len() == capacity * NUM_KV_HEADS * HEAD_DIM
                && v.len() == k.len(),
            "Laguna direct FP8 prefill cache allocation/device mismatch"
        );
        let dev = match cache_device {
            Device::Cuda(dev) => dev.clone(),
            _ => anyhow::bail!("Laguna direct FP8 prefill requires CUDA"),
        };
        let stream = nv_layers::cuda_stream::current_stream(&dev);
        anyhow::ensure!(
            std::sync::Arc::ptr_eq(k.context(), stream.context())
                && std::sync::Arc::ptr_eq(v.context(), stream.context()),
            "Laguna direct FP8 prefill cache and launch stream must share one CUDA context"
        );
        let raw_stream = stream.cu_stream() as *mut c_void;
        if let Some((_, _, _, planned_stream)) = self.key {
            anyhow::ensure!(
                planned_stream == raw_stream as usize,
                "Laguna direct FP8 prefill cache cannot move between CUDA streams"
            );
        }
        let key = (
            q_len as u32,
            kv_len as u32,
            capacity as u32,
            raw_stream as usize,
        );
        if self.key != Some(key) {
            let mut plan = LagunaPrefillFp8Plan::default();
            let mut error = [0 as c_char; ERROR_BYTES];
            let rc = unsafe {
                laguna_prefill_fp8_prepare(
                    &mut plan,
                    key.0,
                    key.1,
                    key.2,
                    raw_stream,
                    error.as_mut_ptr(),
                    error.len(),
                )
            };
            anyhow::ensure!(
                rc == 0,
                "Laguna direct FP8 prefill prepare failed: {}",
                adapter_error(&error)
            );
            let required = usize::try_from(plan.base.scratch_bytes)
                .context("Laguna direct FP8 prefill scratch exceeds usize")?
                .max(16);
            if required > self.scratch_bytes {
                self.scratch = Some(unsafe {
                    stream.alloc::<u8>(required).map_err(|e| {
                        anyhow::anyhow!("Laguna direct FP8 prefill scratch allocation: {e:?}")
                    })?
                });
                self.scratch_bytes = required;
            }
            self.plan = plan;
            self.key = Some(key);
        }
        let mut output = unsafe {
            stream
                .alloc::<bf16>(q_len * NUM_Q_HEADS * HEAD_DIM)
                .map_err(|e| {
                    anyhow::anyhow!("Laguna direct FP8 prefill output allocation: {e:?}")
                })?
        };
        let scratch = self
            .scratch
            .as_mut()
            .context("Laguna direct FP8 prefill scratch missing")?;
        let (storage, layout) = q.storage_and_layout();
        let q_cuda = match &*storage {
            candle_core::Storage::Cuda(storage) => storage,
            _ => anyhow::bail!("Laguna direct FP8 prefill Q storage is not CUDA"),
        };
        let q_view = q_cuda
            .as_cuda_slice::<bf16>()?
            .slice(layout.start_offset()..);
        let mut error = [0 as c_char; ERROR_BYTES];
        let rc = {
            let (q_ptr, _q_guard) = q_view.device_ptr(&stream);
            let (k_ptr, _k_guard) = k.device_ptr(&stream);
            let (v_ptr, _v_guard) = v.device_ptr(&stream);
            let (output_ptr, _output_guard) = output.device_ptr_mut(&stream);
            let (scratch_ptr, _scratch_guard) = scratch.device_ptr_mut(&stream);
            let cache = LagunaPrefillFp8CacheView {
                cache_elem_enum: 2,
                scale_mode: 1,
                k_scale: 1.0,
                v_scale: 1.0,
                k: k_ptr,
                v: v_ptr,
                k_row_scales: 0,
                v_row_scales: 0,
            };
            unsafe {
                laguna_prefill_fp8_run(
                    &self.plan,
                    q_ptr as *const u16,
                    &cache,
                    output_ptr as *mut u16,
                    scratch_ptr as *mut c_void,
                    self.scratch_bytes,
                    raw_stream,
                    error.as_mut_ptr(),
                    error.len(),
                )
            }
        };
        anyhow::ensure!(
            rc == 0,
            "Laguna direct FP8 prefill launch failed: {}",
            adapter_error(&error)
        );
        self.enqueued_calls += 1;
        let p = &self.plan.base;
        let diagnostic = (p.tile_q, p.tile_kv, p.splits);
        if self.last_diagnostic != Some(diagnostic) {
            eprintln!("[laguna_prefill_attn] kv=fp8-e4m3 scale=fixed-one bf16_view=false launch_enqueued=true M={} S={} capacity={} tile={}x{} splits={} ctas={} scratch_bytes={} occupancy={}", p.q_len, p.kv_len, p.capacity, p.tile_q, p.tile_kv, p.splits, p.ctas, p.scratch_bytes, p.active_blocks_per_sm);
            self.last_diagnostic = Some(diagnostic);
        }
        Ok(Tensor::from_storage(
            candle_core::Storage::Cuda(candle_core::CudaStorage::wrap_cuda_slice(output, dev)),
            (1usize, q_len, NUM_Q_HEADS, HEAD_DIM),
            candle_core::op::BackpropOp::none(),
            false,
        ))
    }
}
