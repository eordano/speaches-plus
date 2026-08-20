use anyhow::Result;
use candle_core::Tensor;
use candle_nn::VarBuilder;

use crate::linear::Linear;

pub struct Mlp {
    gate_proj: Linear,
    up_proj: Linear,
    down_proj: Linear,
}

impl Mlp {
    pub fn new(gate_proj: Linear, up_proj: Linear, down_proj: Linear) -> Result<Self> {
        if gate_proj.in_features() != up_proj.in_features() {
            anyhow::bail!(
                "Mlp: gate_proj in {} != up_proj in {}",
                gate_proj.in_features(),
                up_proj.in_features()
            );
        }
        if gate_proj.out_features() != up_proj.out_features() {
            anyhow::bail!(
                "Mlp: gate_proj out {} != up_proj out {}",
                gate_proj.out_features(),
                up_proj.out_features()
            );
        }
        if down_proj.in_features() != gate_proj.out_features() {
            anyhow::bail!(
                "Mlp: down_proj in {} != gate_proj out {}",
                down_proj.in_features(),
                gate_proj.out_features()
            );
        }
        Ok(Self {
            gate_proj,
            up_proj,
            down_proj,
        })
    }

    pub fn from_candle_vb(vb: VarBuilder, hidden: usize, intermediate: usize) -> Result<Self> {
        let gate_proj = Linear::from_candle_vb(vb.pp("gate_proj"), hidden, intermediate, false)?;
        let up_proj = Linear::from_candle_vb(vb.pp("up_proj"), hidden, intermediate, false)?;
        let down_proj = Linear::from_candle_vb(vb.pp("down_proj"), intermediate, hidden, false)?;
        Self::new(gate_proj, up_proj, down_proj)
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let gate = self.gate_proj.forward(x)?;
        let up = self.up_proj.forward(x)?;
        let activated = candle_nn::ops::silu(&gate)?.mul(&up)?;
        self.down_proj.forward(&activated)
    }

    #[cfg(feature = "cuda")]
    pub fn forward_fused_cuda(&self, x: &Tensor) -> Result<Tensor> {
        use candle_core::{DType, Device};
        use cudarc::driver::{DevicePtr, DevicePtrMut};
        use half::bf16;
        use std::ffi::c_void;

        let gate = self.gate_proj.forward(x)?;
        let up = self.up_proj.forward(x)?;
        if gate.dtype() != DType::BF16 || !matches!(gate.device(), Device::Cuda(_)) {
            let activated = candle_nn::ops::silu(&gate)?.mul(&up)?;
            return self.down_proj.forward(&activated);
        }
        let dev = match gate.device() {
            Device::Cuda(d) => d.clone(),
            _ => unreachable!(),
        };
        let stream = crate::cuda_stream::current_stream(&dev);
        let gate_c = gate.contiguous()?;
        let up_c = up.contiguous()?;
        let n = gate_c.elem_count();
        anyhow::ensure!(up_c.elem_count() == n, "fused silu: gate/up size mismatch");
        let mut y: cudarc::driver::CudaSlice<bf16> =
            unsafe { stream.alloc::<bf16>(n).map_err(|e| anyhow::anyhow!(e))? };
        {
            let (gs, gl) = gate_c.storage_and_layout();
            let (us, ul) = up_c.storage_and_layout();
            let g_cuda = match &*gs {
                candle_core::Storage::Cuda(s) => s,
                _ => anyhow::bail!("fused silu: gate must be CUDA"),
            };
            let u_cuda = match &*us {
                candle_core::Storage::Cuda(s) => s,
                _ => anyhow::bail!("fused silu: up must be CUDA"),
            };
            let g_slice = g_cuda.as_cuda_slice::<bf16>()?;
            let u_slice = u_cuda.as_cuda_slice::<bf16>()?;
            let g_view = g_slice.slice(gl.start_offset()..gl.start_offset() + n);
            let u_view = u_slice.slice(ul.start_offset()..ul.start_offset() + n);
            let (gp, _g1) = g_view.device_ptr(&stream);
            let (up_ptr, _g2) = u_view.device_ptr(&stream);
            let (yp, _g3) = y.device_ptr_mut(&stream);
            let rc = unsafe {
                nv_kernels::cuda::silu_mul_bf16_candle(
                    stream.cu_stream() as *mut c_void,
                    gp as *const u16,
                    up_ptr as *const u16,
                    yp as *mut u16,
                    n,
                )
            };
            anyhow::ensure!(rc == 0, "silu_mul_bf16_candle rc={rc}");
        }
        let storage = candle_core::CudaStorage::wrap_cuda_slice(y, dev);
        let activated = Tensor::from_storage(
            candle_core::Storage::Cuda(storage),
            gate_c.shape().clone(),
            candle_core::op::BackpropOp::none(),
            false,
        );
        self.down_proj.forward(&activated)
    }

    pub fn gate_proj(&self) -> &Linear {
        &self.gate_proj
    }

    pub fn up_proj(&self) -> &Linear {
        &self.up_proj
    }

    pub fn down_proj(&self) -> &Linear {
        &self.down_proj
    }
}
