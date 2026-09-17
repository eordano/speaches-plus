use anyhow::{ensure, Result};
use candle_core::{DType, Tensor};

pub const NV_Q38_GDN_HEAD_KERNEL_DEFAULT_ON_0_RESTORES_THE_CANDLE_CONV_SILU_EXPAND_L2NORM_CHAIN: &str =
    "NV_Q38_GDN_HEAD_KERNEL";

pub const PACKED_ROW_IS_Q_SCALED_THEN_K_GRP_THEN_V_PER_VALUE_HEAD: usize = 3;

#[derive(Clone, Copy)]
pub struct GdnHeadDims {
    pub b: usize,
    pub t: usize,
    pub conv_dim: usize,
    pub key_dim: usize,
    pub n_v: usize,
    pub v_per_k: usize,
    pub head_dim: usize,
    pub conv_k: usize,
}

pub fn fused_head_kernel_serves(x: &Tensor, d: &GdnHeadDims) -> bool {
    std::env::var(NV_Q38_GDN_HEAD_KERNEL_DEFAULT_ON_0_RESTORES_THE_CANDLE_CONV_SILU_EXPAND_L2NORM_CHAIN)
        .as_deref()
        != Ok("0")
        && cfg!(feature = "cuda")
        && x.dtype() == DType::BF16
        && matches!(x.device(), candle_core::Device::Cuda(_))
        && d.head_dim == 128
        && d.conv_k == 4
        && d.conv_dim % 1024 == 0
        && d.key_dim % 128 == 0
        && d.n_v * 128 == d.conv_dim - 2 * d.key_dim
        && (d.key_dim / 128) * d.v_per_k == d.n_v
}

#[cfg(feature = "cuda")]
pub struct GdnHeadPackedOp {
    w: Tensor,
    d: GdnHeadDims,
}

#[cfg(feature = "cuda")]
impl GdnHeadPackedOp {
    pub fn new(conv1d_weight: &Tensor, d: GdnHeadDims) -> Result<Self> {
        let w = conv1d_weight
            .to_dtype(DType::F32)?
            .reshape((d.conv_dim, d.conv_k))?
            .contiguous()?;
        Ok(Self { w, d })
    }

    fn q_scale(&self) -> f32 {
        1.0 / (self.d.head_dim as f32).sqrt()
    }
}

#[cfg(feature = "cuda")]
fn cuda_f32_ptr(t: &Tensor, stream: &std::sync::Arc<cudarc::driver::CudaStream>) -> candle_core::Result<u64> {
    use cudarc::driver::DevicePtr;
    let (st, l) = t.storage_and_layout();
    let candle_core::Storage::Cuda(c) = &*st else {
        return Err(candle_core::Error::Msg("gdn head op: operand not cuda".into()));
    };
    let (p, _g) = c.as_cuda_slice::<f32>()?.device_ptr(stream);
    Ok(p + (l.start_offset() * 4) as u64)
}

#[cfg(feature = "cuda")]
impl candle_core::CustomOp1 for GdnHeadPackedOp {
    fn name(&self) -> &'static str {
        "gdn_head_packed"
    }

    fn cpu_fwd(
        &self,
        _s: &candle_core::CpuStorage,
        _l: &candle_core::Layout,
    ) -> candle_core::Result<(candle_core::CpuStorage, candle_core::Shape)> {
        Err(candle_core::Error::Msg("gdn_head_packed is CUDA-only".into()))
    }

    fn cuda_fwd(
        &self,
        s: &candle_core::CudaStorage,
        l: &candle_core::Layout,
    ) -> candle_core::Result<(candle_core::CudaStorage, candle_core::Shape)> {
        use candle_core::backend::BackendStorage;
        use cudarc::driver::{DevicePtr, DevicePtrMut};
        let d = self.d;
        let dev = s.device().clone();
        let stream = nv_layers::cuda_stream::current_stream(&dev);
        let Some((start, _end)) = l.contiguous_offsets() else {
            return Err(candle_core::Error::Msg("gdn head op: input must be contiguous".into()));
        };
        if l.shape().dims() != [d.b, d.t, d.conv_dim] {
            return Err(candle_core::Error::Msg(format!(
                "gdn head op: input {:?} vs [{}, {}, {}]",
                l.shape(),
                d.b,
                d.t,
                d.conv_dim
            )));
        }
        let x = s.as_cuda_slice::<half::bf16>()?;
        let x_view = x.slice(start..);
        let (xp, _g) = x_view.device_ptr(&stream);
        let wp = cuda_f32_ptr(&self.w, &stream)?;
        let cols = PACKED_ROW_IS_Q_SCALED_THEN_K_GRP_THEN_V_PER_VALUE_HEAD * d.head_dim;
        let mut packed: cudarc::driver::CudaSlice<f32> =
            unsafe { stream.alloc::<f32>(d.b * d.t * d.n_v * cols) }
                .map_err(|e| candle_core::Error::Msg(format!("gdn head op alloc: {e:?}")))?;
        let rc = unsafe {
            let (pp, _g2) = packed.device_ptr_mut(&stream);
            nv_kernels::cuda::gdn_train_head_fwd(
                stream.cu_stream() as *mut _,
                xp as *const u16,
                wp as *const f32,
                pp as *mut f32,
                d.b as i32,
                d.t as i32,
                d.conv_dim as i32,
                d.key_dim as i32,
                d.n_v as i32,
                d.v_per_k as i32,
                d.conv_k as i32,
                self.q_scale(),
            )
        };
        if rc != 0 {
            return Err(candle_core::Error::Msg(format!("gdn_train_head_fwd rc={rc}")));
        }
        Ok((
            candle_core::CudaStorage::wrap_cuda_slice(packed, dev),
            candle_core::Shape::from((d.b, d.t, d.n_v, cols)),
        ))
    }

    fn bwd(
        &self,
        arg: &Tensor,
        _res: &Tensor,
        grad_res: &Tensor,
    ) -> candle_core::Result<Option<Tensor>> {
        use cudarc::driver::{DevicePtr, DevicePtrMut};
        let d = self.d;
        let candle_core::Device::Cuda(dev) = arg.device() else {
            return Err(candle_core::Error::Msg("gdn head op bwd: not cuda".into()));
        };
        let stream = nv_layers::cuda_stream::current_stream(dev);
        let x = arg.detach().to_dtype(DType::BF16)?.contiguous()?;
        let g = grad_res.detach().to_dtype(DType::F32)?.contiguous()?;
        let mut dy: cudarc::driver::CudaSlice<f32> =
            unsafe { stream.alloc::<f32>(d.b * d.t * d.conv_dim) }
                .map_err(|e| candle_core::Error::Msg(format!("gdn head op alloc: {e:?}")))?;
        let mut dx: cudarc::driver::CudaSlice<half::bf16> =
            unsafe { stream.alloc::<half::bf16>(d.b * d.t * d.conv_dim) }
                .map_err(|e| candle_core::Error::Msg(format!("gdn head op alloc: {e:?}")))?;
        let rc = {
            let (xs, xl) = x.storage_and_layout();
            let candle_core::Storage::Cuda(xc) = &*xs else {
                return Err(candle_core::Error::Msg("gdn head op bwd: x not cuda".into()));
            };
            let (xp, _g1) = xc.as_cuda_slice::<half::bf16>()?.device_ptr(&stream);
            let xp = xp + (xl.start_offset() * 2) as u64;
            let wp = cuda_f32_ptr(&self.w, &stream)?;
            let gp = cuda_f32_ptr(&g, &stream)?;
            let (dyp, _g2) = dy.device_ptr_mut(&stream);
            let (dxp, _g3) = dx.device_ptr_mut(&stream);
            unsafe {
                nv_kernels::cuda::gdn_train_head_bwd(
                    stream.cu_stream() as *mut _,
                    xp as *const u16,
                    wp as *const f32,
                    gp as *const f32,
                    dyp as *mut f32,
                    dxp as *mut u16,
                    d.b as i32,
                    d.t as i32,
                    d.conv_dim as i32,
                    d.key_dim as i32,
                    d.n_v as i32,
                    d.v_per_k as i32,
                    d.conv_k as i32,
                    self.q_scale(),
                )
            }
        };
        if rc != 0 {
            return Err(candle_core::Error::Msg(format!("gdn_train_head_bwd rc={rc}")));
        }
        let dx_t = Tensor::from_storage(
            candle_core::Storage::Cuda(candle_core::CudaStorage::wrap_cuda_slice(dx, dev.clone())),
            (d.b, d.t, d.conv_dim),
            candle_core::op::BackpropOp::none(),
            false,
        );
        Ok(Some(dx_t.to_dtype(arg.dtype())?))
    }
}

pub fn packed_dims_hold(d: &GdnHeadDims) -> Result<()> {
    ensure!(d.head_dim == 128 && d.conv_k == 4, "the fused gdn head serves head_dim 128 and conv kernel 4");
    Ok(())
}
