#[cfg(feature = "cuda")]
use anyhow::{ensure, Result};
#[cfg(feature = "cuda")]
use candle_core::CudaStorage;
#[cfg(feature = "cuda")]
use candle_core::Error;
#[cfg(feature = "cuda")]
use candle_core::Shape;
use candle_core::{DType, Tensor};

pub fn fused_ce_serves(logits: &Tensor) -> bool {
    cfg!(feature = "cuda")
        && logits.dtype() == DType::BF16
        && logits.device().is_cuda()
}

#[cfg(feature = "cuda")]
pub struct CeRowsBf16Op {
    targets: Tensor,
    weights: Tensor,
    dlogits: std::sync::Mutex<Option<Tensor>>,
}

#[cfg(feature = "cuda")]
impl CeRowsBf16Op {
    pub fn new(targets: &Tensor, weights: &[f32]) -> Result<Self> {
        let n = targets.dims1()?;
        ensure!(
            weights.len() == n,
            "ce rows op: {} weights for {n} targets",
            weights.len()
        );
        Ok(Self {
            targets: targets.to_dtype(DType::U32)?.contiguous()?,
            weights: Tensor::from_vec(weights.to_vec(), n, targets.device())?,
            dlogits: std::sync::Mutex::new(None),
        })
    }
}

#[cfg(feature = "cuda")]
fn ptr_of<T: candle_core::cuda_backend::CudaDType + cudarc::driver::DeviceRepr>(
    t: &Tensor,
    stream: &std::sync::Arc<cudarc::driver::CudaStream>,
) -> candle_core::Result<u64> {
    use cudarc::driver::DevicePtr;
    let (st, l) = t.storage_and_layout();
    let candle_core::Storage::Cuda(c) = &*st else {
        return Err(Error::Msg("ce rows op: operand not cuda".into()));
    };
    let (p, _g) = c.as_cuda_slice::<T>()?.device_ptr(stream);
    Ok(p + (l.start_offset() * std::mem::size_of::<T>()) as u64)
}

#[cfg(feature = "cuda")]
impl candle_core::CustomOp1 for CeRowsBf16Op {
    fn name(&self) -> &'static str {
        "ce_rows_bf16"
    }

    fn cpu_fwd(
        &self,
        _s: &candle_core::CpuStorage,
        _l: &candle_core::Layout,
    ) -> candle_core::Result<(candle_core::CpuStorage, Shape)> {
        Err(Error::Msg("ce_rows_bf16 is CUDA-only".into()))
    }

    fn cuda_fwd(
        &self,
        s: &CudaStorage,
        l: &candle_core::Layout,
    ) -> candle_core::Result<(CudaStorage, Shape)> {
        use candle_core::backend::BackendStorage;
        use cudarc::driver::{DevicePtr, DevicePtrMut};
        let dev = s.device().clone();
        let stream = nv_layers::cuda_stream::current_stream(&dev);
        let dims = l.shape().dims().to_vec();
        if dims.len() != 2 {
            return Err(Error::Msg(format!(
                "ce rows op: logits must be 2-D, got {dims:?}"
            )));
        }
        let (rows, vocab) = (dims[0], dims[1]);
        let Some((start, _end)) = l.contiguous_offsets() else {
            return Err(Error::Msg("ce rows op: logits must be contiguous".into()));
        };
        if self.targets.dims1()? != rows {
            return Err(Error::Msg(format!(
                "ce rows op: {} targets for {rows} logits rows",
                self.targets.dims1()?
            )));
        }
        let x = s.as_cuda_slice::<half::bf16>()?;
        let x_view = x.slice(start..);
        let (xp, _g) = x_view.device_ptr(&stream);
        let tp = ptr_of::<u32>(&self.targets, &stream)?;
        let wp = ptr_of::<f32>(&self.weights, &stream)?;
        let mut loss: cudarc::driver::CudaSlice<f32> = unsafe { stream.alloc::<f32>(rows) }
            .map_err(|e| Error::Msg(format!("ce rows op alloc: {e:?}")))?;
        let mut dl: cudarc::driver::CudaSlice<half::bf16> =
            unsafe { stream.alloc::<half::bf16>(rows * vocab) }
                .map_err(|e| Error::Msg(format!("ce rows op alloc: {e:?}")))?;
        let rc = unsafe {
            let (lp, _g1) = loss.device_ptr_mut(&stream);
            let (dp, _g2) = dl.device_ptr_mut(&stream);
            nv_kernels::cuda::ce_rows_bf16_fwd_dlogits(
                stream.cu_stream() as *mut _,
                xp as *const u16,
                tp as *const u32,
                wp as *const f32,
                lp as *mut f32,
                dp as *mut u16,
                rows as i64,
                vocab as i32,
            )
        };
        if rc != 0 {
            return Err(Error::Msg(format!("ce_rows_bf16_fwd_dlogits rc={rc}")));
        }
        let dl_t = Tensor::from_storage(
            candle_core::Storage::Cuda(CudaStorage::wrap_cuda_slice(dl, dev.clone())),
            (rows, vocab),
            candle_core::op::BackpropOp::none(),
            false,
        );
        *self.dlogits.lock().unwrap() = Some(dl_t);
        Ok((CudaStorage::wrap_cuda_slice(loss, dev), Shape::from(rows)))
    }

    fn bwd(
        &self,
        arg: &Tensor,
        _res: &Tensor,
        grad_res: &Tensor,
    ) -> candle_core::Result<Option<Tensor>> {
        let candle_core::Device::Cuda(dev) = arg.device() else {
            return Err(Error::Msg("ce rows op bwd: not cuda".into()));
        };
        let stream = nv_layers::cuda_stream::current_stream(dev);
        let dl = self
            .dlogits
            .lock()
            .unwrap()
            .take()
            .ok_or_else(|| Error::Msg("ce rows op bwd: no saved dlogits".into()))?;
        let (rows, vocab) = dl.dims2()?;
        let scale = grad_res.detach().to_dtype(DType::F32)?.contiguous()?;
        let rc = unsafe {
            let dp = ptr_of::<half::bf16>(&dl, &stream)?;
            let sp = ptr_of::<f32>(&scale, &stream)?;
            nv_kernels::cuda::ce_rows_scale_dlogits(
                stream.cu_stream() as *mut _,
                dp as *mut u16,
                sp as *const f32,
                rows as i64,
                vocab as i32,
            )
        };
        if rc != 0 {
            return Err(Error::Msg(format!("ce_rows_scale_dlogits rc={rc}")));
        }
        Ok(Some(dl.to_dtype(arg.dtype())?))
    }
}

#[cfg(feature = "cuda")]
pub fn ce_rows_fused(logits_bf16: &Tensor, targets: &Tensor, weights: &[f32]) -> Result<Tensor> {
    let op = CeRowsBf16Op::new(targets, weights)?;
    Ok(logits_bf16.contiguous()?.apply_op1(op)?.sum_all()?)
}
