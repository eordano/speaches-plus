use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use candle_core::{DType, Device, Shape, Tensor, TensorId};
use half::{bf16, f16};
use nv_kernels::wgpu_backend::device::WgpuContext;
use nv_kernels::wgpu_backend::dispatch::GpuTensor;

use crate::backend::BackendError;

fn bridge(detail: impl std::fmt::Display) -> BackendError {
    BackendError::Bridge(detail.to_string())
}

pub fn dtype_size(dtype: DType) -> Result<usize, BackendError> {
    match dtype {
        DType::F32 | DType::U32 => Ok(4),
        DType::BF16 | DType::F16 => Ok(2),
        DType::U8 => Ok(1),
        other => Err(bridge(format!(
            "dtype {other:?} has no wgpu bridge representation (supported: F32, BF16, F16, U32, U8)"
        ))),
    }
}

fn words_from_bytes(bytes: &[u8]) -> Vec<u32> {
    let mut words = vec![0u32; bytes.len().div_ceil(4).max(1)];
    bytemuck::cast_slice_mut::<u32, u8>(&mut words)[..bytes.len()].copy_from_slice(bytes);
    words
}

fn host_bytes(t: &Tensor) -> Result<Vec<u8>, BackendError> {
    let flat = t.flatten_all().map_err(bridge)?;
    match t.dtype() {
        DType::F32 => Ok(bytemuck::cast_slice(&flat.to_vec1::<f32>().map_err(bridge)?).to_vec()),
        DType::U32 => Ok(bytemuck::cast_slice(&flat.to_vec1::<u32>().map_err(bridge)?).to_vec()),
        DType::BF16 => {
            let bits: Vec<u16> = flat
                .to_vec1::<bf16>()
                .map_err(bridge)?
                .into_iter()
                .map(bf16::to_bits)
                .collect();
            Ok(bytemuck::cast_slice(&bits).to_vec())
        }
        DType::F16 => {
            let bits: Vec<u16> = flat
                .to_vec1::<f16>()
                .map_err(bridge)?
                .into_iter()
                .map(f16::to_bits)
                .collect();
            Ok(bytemuck::cast_slice(&bits).to_vec())
        }
        DType::U8 => flat.to_vec1::<u8>().map_err(bridge),
        other => Err(bridge(format!(
            "dtype {other:?} has no wgpu bridge representation (supported: F32, BF16, F16, U32, U8)"
        ))),
    }
}

pub struct WgpuTensor {
    words: GpuTensor<u32>,
    dtype: DType,
    shape: Shape,
    len: usize,
}

impl WgpuTensor {
    pub fn from_candle(ctx: &WgpuContext, label: &str, t: &Tensor) -> Result<Self, BackendError> {
        let dtype = t.dtype();
        let shape = t.shape().clone();
        let len = shape.elem_count();
        dtype_size(dtype)?;
        let bytes = host_bytes(t)?;
        let words = words_from_bytes(&bytes);
        Ok(Self {
            words: GpuTensor::upload(ctx, label, &words),
            dtype,
            shape,
            len,
        })
    }

    pub fn zeroed(
        ctx: &WgpuContext,
        label: &str,
        dtype: DType,
        shape: impl Into<Shape>,
    ) -> Result<Self, BackendError> {
        let shape = shape.into();
        let len = shape.elem_count();
        let words = (len * dtype_size(dtype)?).div_ceil(4).max(1);
        Ok(Self {
            words: GpuTensor::zeroed(ctx, label, words),
            dtype,
            shape,
            len,
        })
    }

    pub fn write_candle(&self, ctx: &WgpuContext, t: &Tensor) -> Result<(), BackendError> {
        if t.dtype() != self.dtype {
            return Err(bridge(format!(
                "write_candle dtype {:?} into buffer of {:?}",
                t.dtype(),
                self.dtype
            )));
        }
        if t.shape().elem_count() != self.len {
            return Err(bridge(format!(
                "write_candle {} elements into buffer of {}",
                t.shape().elem_count(),
                self.len
            )));
        }
        let words = words_from_bytes(&host_bytes(t)?);
        self.words.write(ctx, &words).map_err(bridge)
    }

    pub fn to_candle(&self, ctx: &WgpuContext, device: &Device) -> Result<Tensor, BackendError> {
        let words = self.words.download(ctx).map_err(bridge)?;
        let bytes = &bytemuck::cast_slice::<u32, u8>(&words)[..self.len * dtype_size(self.dtype)?];
        let t = match self.dtype {
            DType::F32 => Tensor::from_vec(
                bytemuck::cast_slice::<u8, f32>(bytes).to_vec(),
                self.shape.clone(),
                device,
            ),
            DType::U32 => Tensor::from_vec(
                bytemuck::cast_slice::<u8, u32>(bytes).to_vec(),
                self.shape.clone(),
                device,
            ),
            DType::BF16 => Tensor::from_vec(
                bytemuck::cast_slice::<u8, u16>(bytes)
                    .iter()
                    .copied()
                    .map(bf16::from_bits)
                    .collect::<Vec<bf16>>(),
                self.shape.clone(),
                device,
            ),
            DType::F16 => Tensor::from_vec(
                bytemuck::cast_slice::<u8, u16>(bytes)
                    .iter()
                    .copied()
                    .map(f16::from_bits)
                    .collect::<Vec<f16>>(),
                self.shape.clone(),
                device,
            ),
            DType::U8 => Tensor::from_vec(bytes.to_vec(), self.shape.clone(), device),
            other => return Err(bridge(format!("dtype {other:?} cannot leave the bridge"))),
        };
        t.map_err(bridge)
    }

    pub fn dtype(&self) -> DType {
        self.dtype
    }

    pub fn shape(&self) -> &Shape {
        &self.shape
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub fn word_len(&self) -> usize {
        self.words.len()
    }

    pub fn byte_capacity(&self) -> u64 {
        self.words.len() as u64 * 4
    }

    pub fn gpu(&self) -> &GpuTensor<u32> {
        &self.words
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ResidencyStats {
    pub entries: usize,
    pub resident_bytes: u64,
    pub hits: u64,
    pub misses: u64,
}

#[derive(Default)]
pub struct ResidencyCache {
    map: Mutex<HashMap<TensorId, Arc<WgpuTensor>>>,
    resident_bytes: AtomicU64,
    hits: AtomicU64,
    misses: AtomicU64,
}

impl ResidencyCache {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn get_or_upload(
        &self,
        ctx: &WgpuContext,
        label: &str,
        t: &Tensor,
    ) -> Result<Arc<WgpuTensor>, BackendError> {
        let id = t.id();
        if let Some(hit) = self.map.lock().unwrap().get(&id) {
            self.hits.fetch_add(1, Ordering::Relaxed);
            return Ok(hit.clone());
        }
        let uploaded = Arc::new(WgpuTensor::from_candle(ctx, label, t)?);
        let mut map = self.map.lock().unwrap();
        if let Some(raced) = map.get(&id) {
            self.hits.fetch_add(1, Ordering::Relaxed);
            return Ok(raced.clone());
        }
        self.misses.fetch_add(1, Ordering::Relaxed);
        self.resident_bytes
            .fetch_add(uploaded.byte_capacity(), Ordering::Relaxed);
        map.insert(id, uploaded.clone());
        Ok(uploaded)
    }

    pub fn contains(&self, t: &Tensor) -> bool {
        self.map.lock().unwrap().contains_key(&t.id())
    }

    pub fn evict(&self, t: &Tensor) -> bool {
        match self.map.lock().unwrap().remove(&t.id()) {
            Some(gone) => {
                self.resident_bytes
                    .fetch_sub(gone.byte_capacity(), Ordering::Relaxed);
                true
            }
            None => false,
        }
    }

    pub fn clear(&self) {
        self.map.lock().unwrap().clear();
        self.resident_bytes.store(0, Ordering::Relaxed);
    }

    pub fn stats(&self) -> ResidencyStats {
        ResidencyStats {
            entries: self.map.lock().unwrap().len(),
            resident_bytes: self.resident_bytes.load(Ordering::Relaxed),
            hits: self.hits.load(Ordering::Relaxed),
            misses: self.misses.load(Ordering::Relaxed),
        }
    }
}
