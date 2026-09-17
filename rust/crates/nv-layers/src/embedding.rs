use anyhow::{ensure, Context, Result};
use candle_core::{DType, Device, Storage, Tensor};
use cudarc::driver::{sys, CudaSlice, CudaStream, DevicePtr, DevicePtrMut};
use half::bf16;
use std::sync::Arc;

pub struct Fp8Embedding {
    bytes: CudaSlice<u8>,
    row_scales: CudaSlice<f32>,
    vocab: usize,
    hidden: usize,
    device: Device,
}

impl Fp8Embedding {
    fn checked_weight_elements(vocab: usize, hidden: usize) -> Result<usize> {
        ensure!(
            vocab > 0 && vocab <= i32::MAX as usize,
            "FP8 embedding vocabulary must be positive and fit the CUDA ABI"
        );
        ensure!(
            hidden > 0 && hidden <= i32::MAX as usize - 255,
            "FP8 embedding hidden dimension must be in 1..={} so 256-thread CUDA loop increments fit i32",
            i32::MAX as usize - 255
        );
        let elements = vocab
            .checked_mul(hidden)
            .context("FP8 embedding weight size overflow")?;
        elements
            .checked_mul(2)
            .context("FP8 embedding BF16 source byte size overflow")?;
        vocab
            .checked_mul(4)
            .and_then(|n| elements.checked_add(n))
            .context("FP8 embedding resident byte size overflow")?;
        Ok(elements)
    }

    pub fn from_bf16(weight: Tensor) -> Result<Self> {
        let Device::Cuda(dev) = weight.device() else {
            anyhow::bail!("FP8 embedding weights must be CUDA");
        };
        ensure!(
            weight.dtype() == DType::BF16,
            "FP8 embedding weights must be BF16 at load"
        );
        let (vocab, hidden) = weight.dims2()?;
        let elements = Self::checked_weight_elements(vocab, hidden)?;
        {
            let (storage, layout) = weight.storage_and_layout();
            ensure!(
                layout.is_contiguous(),
                "FP8 embedding weights must be contiguous"
            );
            let Storage::Cuda(storage) = &*storage else {
                anyhow::bail!("FP8 embedding weight storage must be CUDA");
            };
            let end = layout
                .start_offset()
                .checked_add(elements)
                .context("FP8 embedding weight offset overflow")?;
            ensure!(
                end <= storage.as_cuda_slice::<bf16>()?.len(),
                "FP8 embedding weight view exceeds storage"
            );
        }
        let stream = Self::stream_for(dev)?;
        ensure!(
            stream.capture_status()? == sys::CUstreamCaptureStatus::CU_STREAM_CAPTURE_STATUS_NONE,
            "FP8 embedding conversion must finish before graph capture"
        );
        let (bytes, row_scales) = crate::lm_head_resident_quant::rowquant_e4m3_dev(&weight)?;
        Ok(Self {
            bytes,
            row_scales,
            vocab,
            hidden,
            device: weight.device().clone(),
        })
    }

    fn stream_for(dev: &candle_core::CudaDevice) -> Result<Arc<CudaStream>> {
        let stream = crate::cuda_stream::current_stream(dev);
        ensure!(
            stream.context().cu_ctx() == dev.cuda_stream().context().cu_ctx(),
            "FP8 embedding caller stream belongs to another CUDA context"
        );
        Ok(stream)
    }

    pub fn dimensions(&self) -> (usize, usize) {
        (self.vocab, self.hidden)
    }

    pub fn resident_bytes(&self) -> usize {
        self.bytes.len() + self.row_scales.len() * std::mem::size_of::<f32>()
    }

    pub fn fp8_parts(&self) -> (&CudaSlice<u8>, &CudaSlice<f32>) {
        (&self.bytes, &self.row_scales)
    }

    pub fn forward(&self, tokens: &Tensor) -> Result<Tensor> {
        ensure!(
            tokens.dtype() == DType::U32 && tokens.device().same_device(&self.device),
            "FP8 embedding token IDs must be U32 on the weight device"
        );
        let count = tokens.elem_count();
        ensure!(
            count <= i32::MAX as usize,
            "FP8 embedding token count exceeds the CUDA ABI"
        );
        let elements = count
            .checked_mul(self.hidden)
            .context("FP8 embedding output size overflow")?;
        elements
            .checked_mul(2)
            .context("FP8 embedding output byte size overflow")?;
        let (storage, layout) = tokens.storage_and_layout();
        ensure!(
            layout.is_contiguous(),
            "FP8 embedding token IDs must be contiguous"
        );
        let Storage::Cuda(storage) = &*storage else {
            anyhow::bail!("FP8 embedding token storage must be CUDA");
        };
        let data = storage.as_cuda_slice::<u32>()?;
        let end = layout
            .start_offset()
            .checked_add(count)
            .context("FP8 embedding token offset overflow")?;
        ensure!(
            end <= data.len(),
            "FP8 embedding token view exceeds storage"
        );
        let Device::Cuda(dev) = &self.device else {
            unreachable!()
        };
        let stream = Self::stream_for(dev)?;
        let mut output = if count == 0 {
            stream.null::<bf16>()?
        } else {
            unsafe { stream.alloc::<bf16>(elements)? }
        };
        if count != 0 {
            let view = data.slice(layout.start_offset()..end);
            let (ids, _ids_guard) = view.device_ptr(&stream);
            let (weights, _weights_guard) = self.bytes.device_ptr(&stream);
            let (scales, _scales_guard) = self.row_scales.device_ptr(&stream);
            let (out, _output_guard) = output.device_ptr_mut(&stream);
            let rc = unsafe {
                nv_kernels::cuda::gather_rows_e4m3_bf16(
                    stream.cu_stream() as *mut _,
                    weights as *const u8,
                    scales as *const f32,
                    ids as *const i32,
                    out as *mut u16,
                    count as i32,
                    self.hidden as i32,
                    self.vocab as i32,
                )
            };
            ensure!(rc == 0, "FP8 embedding gather returned {rc}");
        }
        Ok(Tensor::from_storage(
            Storage::Cuda(candle_core::CudaStorage::wrap_cuda_slice(
                output,
                dev.clone(),
            )),
            (count, self.hidden),
            candle_core::op::BackpropOp::none(),
            false,
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::Fp8Embedding;

    #[test]
    fn dimension_bound_keeps_cuda_stride_increments_in_i32() {
        let last_safe_width = (i32::MAX as usize + 1) - 256;
        for hidden in [
            5120,
            last_safe_width,
            last_safe_width + 1,
            i32::MAX as usize,
        ] {
            let every_increment_fits = (0..256usize).all(|lane| {
                let last = lane + (hidden - 1 - lane) / 256 * 256;
                i32::try_from(last).unwrap().checked_add(256).is_some()
            });
            assert_eq!(
                Fp8Embedding::checked_weight_elements(1, hidden).is_ok(),
                every_increment_fits,
                "hidden={hidden}: every final CUDA loop increment must fit i32"
            );
        }
        assert_eq!(
            Fp8Embedding::checked_weight_elements(248320, 5120).unwrap(),
            1_271_398_400
        );
        for (vocab, hidden) in [(0, 5120), (1, 0), (i32::MAX as usize + 1, 5120)] {
            assert!(Fp8Embedding::checked_weight_elements(vocab, hidden).is_err());
        }
    }
}
