#[cfg(feature = "cuda")]
use anyhow::Result;
#[cfg(feature = "cuda")]
use candle_core::{DType, Device, Tensor};
use std::cell::RefCell;

pub struct StepArena {
    #[cfg(feature = "cuda")]
    slots: Vec<Tensor>,
    cursor: usize,
}

thread_local! {
    static INSTALLED_ARENA_FOR_THE_GRAPHED_STEP_ON_THIS_THREAD: RefCell<Option<StepArena>> = const { RefCell::new(None) };
}

impl StepArena {
    pub fn new() -> Self {
        Self {
            #[cfg(feature = "cuda")]
            slots: Vec::new(),
            cursor: 0,
        }
    }
}

impl Default for StepArena {
    fn default() -> Self {
        Self::new()
    }
}

pub fn install(arena: StepArena) -> Option<StepArena> {
    INSTALLED_ARENA_FOR_THE_GRAPHED_STEP_ON_THIS_THREAD.with(|a| a.borrow_mut().replace(arena))
}

pub fn uninstall() -> Option<StepArena> {
    INSTALLED_ARENA_FOR_THE_GRAPHED_STEP_ON_THIS_THREAD.with(|a| a.borrow_mut().take())
}

pub fn begin_step() {
    INSTALLED_ARENA_FOR_THE_GRAPHED_STEP_ON_THIS_THREAD.with(|a| {
        if let Some(ar) = a.borrow_mut().as_mut() {
            ar.cursor = 0;
        }
    });
}

#[cfg(feature = "cuda")]
fn fresh_uninitialised(dev: &Device, dtype: DType, elems: usize) -> Result<Tensor> {
    use cudarc::driver::CudaSlice;
    let cuda = match dev {
        Device::Cuda(d) => d.clone(),
        _ => anyhow::bail!("step arena tensors are cuda only"),
    };
    let stream = crate::cuda_stream::current_stream(&cuda);
    let storage = match dtype {
        DType::BF16 => {
            let s: CudaSlice<half::bf16> =
                unsafe { stream.alloc::<half::bf16>(elems) }.map_err(|e| anyhow::anyhow!(e))?;
            candle_core::CudaStorage::wrap_cuda_slice(s, cuda)
        }
        DType::F32 => {
            let s: CudaSlice<f32> =
                unsafe { stream.alloc::<f32>(elems) }.map_err(|e| anyhow::anyhow!(e))?;
            candle_core::CudaStorage::wrap_cuda_slice(s, cuda)
        }
        DType::U32 => {
            let s: CudaSlice<u32> =
                unsafe { stream.alloc::<u32>(elems) }.map_err(|e| anyhow::anyhow!(e))?;
            candle_core::CudaStorage::wrap_cuda_slice(s, cuda)
        }
        other => anyhow::bail!("step arena: unsupported dtype {other:?}"),
    };
    Ok(Tensor::from_storage(
        candle_core::Storage::Cuda(storage),
        elems,
        candle_core::op::BackpropOp::none(),
        false,
    ))
}

#[cfg(feature = "cuda")]
pub fn output_tensor_uninitialised_for_a_kernel_to_fill(
    dev: &Device,
    dtype: DType,
    shape: &[usize],
) -> Result<Tensor> {
    let elems: usize = shape.iter().product();
    let taken = INSTALLED_ARENA_FOR_THE_GRAPHED_STEP_ON_THIS_THREAD.with(|a| -> Result<Option<Tensor>> {
        let mut guard = a.borrow_mut();
        let Some(ar) = guard.as_mut() else {
            return Ok(None);
        };
        let idx = ar.cursor;
        ar.cursor += 1;
        if let Some(slot) = ar.slots.get(idx) {
            if slot.dtype() == dtype && slot.elem_count() == elems {
                return Ok(Some(slot.clone()));
            }
            anyhow::bail!(
                "step arena slot {idx} holds {:?}x{} but the step asked for {dtype:?}x{elems}: the captured sequence changed",
                slot.dtype(),
                slot.elem_count()
            );
        }
        let t = fresh_uninitialised(dev, dtype, elems)?;
        ar.slots.push(t.clone());
        Ok(Some(t))
    })?;
    match taken {
        Some(t) => t.reshape(shape).map_err(Into::into),
        None => fresh_uninitialised(dev, dtype, elems)?
            .reshape(shape)
            .map_err(Into::into),
    }
}

#[cfg(feature = "cuda")]
pub fn device_ptr_for_a_kernel_to_write(t: &Tensor) -> Result<u64> {
    use cudarc::driver::DevicePtr;
    let (storage, layout) = t.storage_and_layout();
    let cuda = match &*storage {
        candle_core::Storage::Cuda(c) => c,
        _ => anyhow::bail!("step arena: tensor not on cuda"),
    };
    let dev = match t.device() {
        Device::Cuda(d) => d.clone(),
        _ => anyhow::bail!("step arena: tensor not on cuda"),
    };
    let stream = crate::cuda_stream::current_stream(&dev);
    let off = layout.start_offset();
    Ok(match t.dtype() {
        DType::BF16 => {
            let (p, _g) = cuda.as_cuda_slice::<half::bf16>()?.device_ptr(&stream);
            p + (off * 2) as u64
        }
        DType::F32 => {
            let (p, _g) = cuda.as_cuda_slice::<f32>()?.device_ptr(&stream);
            p + (off * 4) as u64
        }
        DType::U32 => {
            let (p, _g) = cuda.as_cuda_slice::<u32>()?.device_ptr(&stream);
            p + (off * 4) as u64
        }
        other => anyhow::bail!("step arena: unsupported dtype {other:?}"),
    })
}
