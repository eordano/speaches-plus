use anyhow::Result;
use candle_core::{DType, Tensor};

#[cfg(feature = "cuda")]
use candle_core::Device;

pub struct RmsNorm {
    weight: Option<Tensor>,
    weight_bf16: Option<Tensor>,
    weight_f32: Option<Tensor>,
    #[cfg(feature = "cuda")]
    weight_fp8: Option<RmsNormFp8Weight>,
    eps: f64,
    eps_t: Option<Tensor>,
}

#[cfg(feature = "cuda")]
struct RmsNormFp8Weight {
    bytes: cudarc::driver::CudaSlice<u8>,
    scale: cudarc::driver::CudaSlice<f32>,
    hidden: usize,
}

pub const EPS_UPLOADS_ONCE_AT_BUILD_NOT_ONCE_PER_FORWARD: &str =
    "forward_candle used to create a fresh scalar tensor for eps on every call -- a host \
     upload plus an encode per norm, priced at 120+ per qlora training step; the scalar is \
     device- and value-constant for the layer's lifetime, so it uploads at construction and \
     the per-call path reuses it bitwise";

impl RmsNorm {
    pub fn new(weight: Tensor, eps: f64) -> Self {
        let weight_bf16 = weight
            .to_dtype(DType::BF16)
            .and_then(|w| w.contiguous())
            .unwrap_or_else(|_| weight.clone());
        let weight_f32 = weight
            .to_dtype(DType::F32)
            .and_then(|w| w.contiguous())
            .unwrap_or_else(|_| weight.clone());
        let _ = EPS_UPLOADS_ONCE_AT_BUILD_NOT_ONCE_PER_FORWARD;
        let eps_t = Tensor::new(eps as f32, weight.device()).ok();
        Self {
            weight: Some(weight),
            weight_bf16: Some(weight_bf16),
            weight_f32: Some(weight_f32),
            #[cfg(feature = "cuda")]
            weight_fp8: None,
            eps,
            eps_t,
        }
    }

    #[cfg(feature = "cuda")]
    pub fn new_fp8_e4m3(weight: Tensor, eps: f64) -> Result<Self> {
        let weight = weight.to_dtype(DType::BF16)?.contiguous()?;
        let hidden = weight.elem_count();
        anyhow::ensure!(
            weight.dims() == [hidden],
            "FP8 RmsNorm weight must be rank one"
        );
        let weight_2d = weight.reshape((1usize, hidden))?;
        let (bytes, scale) = crate::lm_head_resident_quant::rowquant_e4m3_dev(&weight_2d)?;
        let eps_t = Tensor::new(eps as f32, weight.device()).ok();
        Ok(Self {
            weight: None,
            weight_bf16: None,
            weight_f32: None,
            weight_fp8: Some(RmsNormFp8Weight {
                bytes,
                scale,
                hidden,
            }),
            eps,
            eps_t,
        })
    }

    pub fn from_candle_vb(vb: candle_nn::VarBuilder, hidden: usize, eps: f64) -> Result<Self> {
        let weight = vb.get(hidden, "weight")?;
        Ok(Self::new(weight, eps))
    }

    pub fn weight(&self) -> &Tensor {
        self.weight
            .as_ref()
            .expect("FP8 RmsNorm has no full-precision weight Tensor")
    }

    pub fn weight_bf16(&self) -> &Tensor {
        self.weight_bf16
            .as_ref()
            .expect("FP8 RmsNorm has no BF16 compatibility weight")
    }

    #[cfg(feature = "cuda")]
    pub fn fp8_e4m3_weight_and_scale(
        &self,
    ) -> Option<(
        &cudarc::driver::CudaSlice<u8>,
        &cudarc::driver::CudaSlice<f32>,
        usize,
    )> {
        self.weight_fp8
            .as_ref()
            .map(|w| (&w.bytes, &w.scale, w.hidden))
    }

    #[cfg(feature = "cuda")]
    pub fn fp8_e4m3_active(&self) -> bool {
        self.weight_fp8.is_some()
    }

    pub fn eps(&self) -> f64 {
        self.eps
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        match (x.device(), x.dtype()) {
            #[cfg(feature = "cuda")]
            (Device::Cuda(_), DType::BF16) => self.forward_cuda_bf16(x),
            #[cfg(feature = "cuda")]
            (Device::Cuda(_), DType::F32) => self.forward_cuda_f32(x),
            _ => self.forward_candle(x),
        }
    }

    pub fn forward_residual(&self, x: &Tensor, residual: &Tensor) -> Result<(Tensor, Tensor)> {
        match (x.device(), x.dtype()) {
            #[cfg(feature = "cuda")]
            (Device::Cuda(_), DType::BF16) => self.forward_residual_cuda_bf16(x, residual),
            #[cfg(feature = "cuda")]
            (Device::Cuda(_), DType::F32) => self.forward_residual_cuda_f32(x, residual),
            _ => {
                let sum = x.add(residual)?;
                let normed = self.forward_candle(&sum)?;
                Ok((normed, sum))
            }
        }
    }

    pub fn forward_candle(&self, x: &Tensor) -> Result<Tensor> {
        #[cfg(feature = "cuda")]
        if let (Device::Cuda(_), DType::BF16 | DType::F32, Some(w)) =
            (x.device(), x.dtype(), self.weight_f32.as_ref())
        {
            let op = RmsNormTrainOp {
                w: w.contiguous()?,
                eps: self.eps as f32,
                rstd: std::sync::Mutex::new(None),
            };
            return Ok(x.contiguous()?.apply_op1(op)?);
        }
        self.forward_candle_chain(x)
    }

    pub fn forward_candle_chain(&self, x: &Tensor) -> Result<Tensor> {
        let xf = x.to_dtype(DType::F32)?;
        let mean_sq = xf.sqr()?.mean_keepdim(candle_core::D::Minus1)?;
        let eps_t = match &self.eps_t {
            Some(t) if t.device().same_device(x.device()) => t.clone(),
            _ => Tensor::new(self.eps as f32, x.device())?,
        };
        let denom = mean_sq.broadcast_add(&eps_t)?.sqrt()?;
        let normed = xf.broadcast_div(&denom)?;
        let w = self
            .weight_f32
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("FP8 RmsNorm supports CUDA BF16 activations only"))?
            .clone();
        let out = normed.broadcast_mul(&w)?;
        Ok(out.to_dtype(x.dtype())?)
    }

    #[cfg(feature = "cuda")]
    fn forward_cuda_bf16(&self, x: &Tensor) -> Result<Tensor> {
        use cudarc::driver::{DevicePtr, DevicePtrMut};
        use half::bf16;
        use nv_kernels::cuda as nvk;

        let x_c = x.contiguous()?;
        let dims = x_c.dims().to_vec();
        let hidden = *dims.last().unwrap();
        let batch: usize = dims[..dims.len() - 1].iter().product();

        let Device::Cuda(dev) = x_c.device().clone() else {
            unreachable!()
        };
        let stream = crate::cuda_stream::current_stream(&dev);

        let mut y_dev: cudarc::driver::CudaSlice<bf16> = unsafe {
            stream
                .alloc::<bf16>(batch * hidden)
                .map_err(|e| anyhow::anyhow!(e))?
        };

        let rc = {
            let (xs, xl) = x_c.storage_and_layout();
            let x_cuda = match &*xs {
                candle_core::Storage::Cuda(s) => s,
                _ => anyhow::bail!("expected cuda storage"),
            };
            let x_slice = x_cuda.as_cuda_slice::<bf16>()?;
            let x_view = x_slice.slice(xl.start_offset()..);
            let (px, _gx) = x_view.device_ptr(&stream);
            let (py, _gy) = y_dev.device_ptr_mut(&stream);
            if let Some(w) = &self.weight_fp8 {
                anyhow::ensure!(
                    w.hidden == hidden,
                    "FP8 RmsNorm hidden {} != input {hidden}",
                    w.hidden
                );
                let (pw, _gw) = w.bytes.device_ptr(&stream);
                let (ps, _gs) = w.scale.device_ptr(&stream);
                unsafe {
                    nvk::rmsnorm_e4m3_weight_bf16(
                        stream.cu_stream() as *mut _,
                        px as *const u16,
                        pw as *const u8,
                        ps as *const f32,
                        py as *mut u16,
                        batch,
                        hidden,
                        self.eps as f32,
                    )
                }
            } else {
                let w_c = self.weight_bf16.as_ref().unwrap().clone();
                let (ws, wl) = w_c.storage_and_layout();
                let w_cuda = match &*ws {
                    candle_core::Storage::Cuda(s) => s,
                    _ => anyhow::bail!("expected cuda storage"),
                };
                let w_slice = w_cuda.as_cuda_slice::<bf16>()?;
                let w_view = w_slice.slice(wl.start_offset()..);
                let (pw, _gw) = w_view.device_ptr(&stream);
                unsafe {
                    nvk::rmsnorm_bf16(
                        stream.cu_stream() as *mut _,
                        px as *const u16,
                        pw as *const u16,
                        py as *mut u16,
                        batch,
                        hidden,
                        self.eps as f32,
                    )
                }
            }
        };
        if rc != 0 {
            anyhow::bail!("rmsnorm_bf16 kernel returned {rc}");
        }

        let storage = candle_core::CudaStorage::wrap_cuda_slice(y_dev, dev);
        let storage = candle_core::Storage::Cuda(storage);
        let shape: candle_core::Shape = dims.into();
        Ok(candle_core::Tensor::from_storage(
            storage,
            shape,
            candle_core::op::BackpropOp::none(),
            false,
        ))
    }

    #[cfg(feature = "cuda")]
    fn forward_cuda_f32(&self, x: &Tensor) -> Result<Tensor> {
        use cudarc::driver::{DevicePtr, DevicePtrMut};
        use nv_kernels::cuda as nvk;

        let x_c = x.contiguous()?;
        let w_c = self
            .weight_f32
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("FP8 RmsNorm supports CUDA BF16 activations only"))?
            .clone();
        let dims = x_c.dims().to_vec();
        let hidden = *dims.last().unwrap();
        let batch: usize = dims[..dims.len() - 1].iter().product();

        let Device::Cuda(dev) = x_c.device().clone() else {
            unreachable!()
        };
        let stream = crate::cuda_stream::current_stream(&dev);

        let mut y_dev: cudarc::driver::CudaSlice<f32> = unsafe {
            stream
                .alloc::<f32>(batch * hidden)
                .map_err(|e| anyhow::anyhow!(e))?
        };

        let rc = {
            let (xs, xl) = x_c.storage_and_layout();
            let (ws, _wl) = w_c.storage_and_layout();
            let x_cuda = match &*xs {
                candle_core::Storage::Cuda(s) => s,
                _ => anyhow::bail!("expected cuda storage"),
            };
            let w_cuda = match &*ws {
                candle_core::Storage::Cuda(s) => s,
                _ => anyhow::bail!("expected cuda storage"),
            };
            let x_slice = x_cuda.as_cuda_slice::<f32>()?;
            let x_view = x_slice.slice(xl.start_offset()..);
            let w_slice = w_cuda.as_cuda_slice::<f32>()?;
            let w_view = w_slice.slice(_wl.start_offset()..);
            let (px, _gx) = x_view.device_ptr(&stream);
            let (pw, _gw) = w_view.device_ptr(&stream);
            let (py, _gy) = y_dev.device_ptr_mut(&stream);
            unsafe {
                nvk::rmsnorm_f32(
                    stream.cu_stream() as *mut _,
                    px as *const f32,
                    pw as *const f32,
                    py as *mut f32,
                    batch,
                    hidden,
                    self.eps as f32,
                )
            }
        };
        if rc != 0 {
            anyhow::bail!("rmsnorm_f32 kernel returned {rc}");
        }

        let storage = candle_core::CudaStorage::wrap_cuda_slice(y_dev, dev);
        let storage = candle_core::Storage::Cuda(storage);
        let shape: candle_core::Shape = dims.into();
        Ok(candle_core::Tensor::from_storage(
            storage,
            shape,
            candle_core::op::BackpropOp::none(),
            false,
        ))
    }

    #[cfg(feature = "cuda")]
    fn forward_residual_cuda_bf16(
        &self,
        x: &Tensor,
        residual: &Tensor,
    ) -> Result<(Tensor, Tensor)> {
        use cudarc::driver::{DevicePtr, DevicePtrMut};
        use half::bf16;
        use nv_kernels::cuda as nvk;

        if x.dims() != residual.dims() {
            anyhow::bail!(
                "rmsnorm_residual: x dims {:?} != residual dims {:?}",
                x.dims(),
                residual.dims()
            );
        }
        let x_c = x.contiguous()?;

        let res_c = residual.contiguous()?;
        let dims = x_c.dims().to_vec();
        let hidden = *dims.last().unwrap();
        let batch: usize = dims[..dims.len() - 1].iter().product();

        let Device::Cuda(dev) = x_c.device().clone() else {
            unreachable!()
        };
        let stream = crate::cuda_stream::current_stream(&dev);

        let mut out_dev: cudarc::driver::CudaSlice<bf16> = unsafe {
            stream
                .alloc::<bf16>(batch * hidden)
                .map_err(|e| anyhow::anyhow!(e))?
        };
        let mut new_res_dev: cudarc::driver::CudaSlice<bf16> = unsafe {
            stream
                .alloc::<bf16>(batch * hidden)
                .map_err(|e| anyhow::anyhow!(e))?
        };

        {
            let (rs, rl) = res_c.storage_and_layout();
            let r_cuda = match &*rs {
                candle_core::Storage::Cuda(s) => s,
                _ => anyhow::bail!("expected cuda storage"),
            };
            let r_slice = r_cuda.as_cuda_slice::<bf16>()?;
            let r_view = r_slice.slice(rl.start_offset()..rl.start_offset() + batch * hidden);
            stream
                .memcpy_dtod(&r_view, &mut new_res_dev)
                .map_err(|e| anyhow::anyhow!(e))?;
        }

        let rc = {
            let (xs, xl) = x_c.storage_and_layout();
            let x_cuda = match &*xs {
                candle_core::Storage::Cuda(s) => s,
                _ => anyhow::bail!("expected cuda storage"),
            };
            let x_slice = x_cuda.as_cuda_slice::<bf16>()?;
            let x_view = x_slice.slice(xl.start_offset()..);
            let (px, _gx) = x_view.device_ptr(&stream);
            let (pres, _gres) = new_res_dev.device_ptr_mut(&stream);
            let (pout, _gout) = out_dev.device_ptr_mut(&stream);
            if let Some(w) = &self.weight_fp8 {
                anyhow::ensure!(
                    w.hidden == hidden,
                    "FP8 RmsNorm hidden {} != input {hidden}",
                    w.hidden
                );
                let (pw, _gw) = w.bytes.device_ptr(&stream);
                let (ps, _gs) = w.scale.device_ptr(&stream);
                unsafe {
                    nvk::rmsnorm_residual_e4m3_weight_bf16(
                        stream.cu_stream() as *mut _,
                        px as *const u16,
                        pres as *mut u16,
                        pw as *const u8,
                        ps as *const f32,
                        pout as *mut u16,
                        batch,
                        hidden,
                        self.eps as f32,
                    )
                }
            } else {
                let w_c = self.weight_bf16.as_ref().unwrap().clone();
                let (ws, wl) = w_c.storage_and_layout();
                let w_cuda = match &*ws {
                    candle_core::Storage::Cuda(s) => s,
                    _ => anyhow::bail!("expected cuda storage"),
                };
                let w_slice = w_cuda.as_cuda_slice::<bf16>()?;
                let w_view = w_slice.slice(wl.start_offset()..);
                let (pw, _gw) = w_view.device_ptr(&stream);
                unsafe {
                    nvk::rmsnorm_residual_bf16(
                        stream.cu_stream() as *mut _,
                        px as *const u16,
                        pres as *mut u16,
                        pw as *const u16,
                        pout as *mut u16,
                        batch,
                        hidden,
                        self.eps as f32,
                    )
                }
            }
        };
        if rc != 0 {
            anyhow::bail!("rmsnorm_residual_bf16 kernel returned {rc}");
        }

        let shape: candle_core::Shape = dims.into();
        let out_storage = candle_core::CudaStorage::wrap_cuda_slice(out_dev, dev.clone());
        let normed = candle_core::Tensor::from_storage(
            candle_core::Storage::Cuda(out_storage),
            shape.clone(),
            candle_core::op::BackpropOp::none(),
            false,
        );
        let res_storage = candle_core::CudaStorage::wrap_cuda_slice(new_res_dev, dev);
        let new_residual = candle_core::Tensor::from_storage(
            candle_core::Storage::Cuda(res_storage),
            shape,
            candle_core::op::BackpropOp::none(),
            false,
        );
        Ok((normed, new_residual))
    }

    #[cfg(feature = "cuda")]
    fn forward_residual_cuda_f32(&self, x: &Tensor, residual: &Tensor) -> Result<(Tensor, Tensor)> {
        use cudarc::driver::{DevicePtr, DevicePtrMut};
        use nv_kernels::cuda as nvk;

        if x.dims() != residual.dims() {
            anyhow::bail!(
                "rmsnorm_residual: x dims {:?} != residual dims {:?}",
                x.dims(),
                residual.dims()
            );
        }
        let x_c = x.contiguous()?;
        let res_c = residual.contiguous()?;
        let w_c = self
            .weight_f32
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("FP8 RmsNorm supports CUDA BF16 activations only"))?
            .clone();
        let dims = x_c.dims().to_vec();
        let hidden = *dims.last().unwrap();
        let batch: usize = dims[..dims.len() - 1].iter().product();

        let Device::Cuda(dev) = x_c.device().clone() else {
            unreachable!()
        };
        let stream = crate::cuda_stream::current_stream(&dev);

        let mut out_dev: cudarc::driver::CudaSlice<f32> = unsafe {
            stream
                .alloc::<f32>(batch * hidden)
                .map_err(|e| anyhow::anyhow!(e))?
        };
        let mut new_res_dev: cudarc::driver::CudaSlice<f32> = unsafe {
            stream
                .alloc::<f32>(batch * hidden)
                .map_err(|e| anyhow::anyhow!(e))?
        };
        {
            let (rs, _rl) = res_c.storage_and_layout();
            let r_cuda = match &*rs {
                candle_core::Storage::Cuda(s) => s,
                _ => anyhow::bail!("expected cuda storage"),
            };
            let r_slice = r_cuda.as_cuda_slice::<f32>()?;
            let r0 = _rl.start_offset();
            let r_view = r_slice.slice(r0..r0 + batch * hidden);
            stream
                .memcpy_dtod(&r_view, &mut new_res_dev)
                .map_err(|e| anyhow::anyhow!(e))?;
        }

        let rc = {
            let (xs, _xl) = x_c.storage_and_layout();
            let (ws, _wl) = w_c.storage_and_layout();
            let x_cuda = match &*xs {
                candle_core::Storage::Cuda(s) => s,
                _ => anyhow::bail!("expected cuda storage"),
            };
            let w_cuda = match &*ws {
                candle_core::Storage::Cuda(s) => s,
                _ => anyhow::bail!("expected cuda storage"),
            };
            let x_slice = x_cuda.as_cuda_slice::<f32>()?;
            let x_view = x_slice.slice(_xl.start_offset()..);
            let w_slice = w_cuda.as_cuda_slice::<f32>()?;
            let w_view = w_slice.slice(_wl.start_offset()..);
            let (px, _gx) = x_view.device_ptr(&stream);
            let (pres, _gres) = new_res_dev.device_ptr_mut(&stream);
            let (pw, _gw) = w_view.device_ptr(&stream);
            let (pout, _gout) = out_dev.device_ptr_mut(&stream);
            unsafe {
                nvk::rmsnorm_residual_f32(
                    stream.cu_stream() as *mut _,
                    px as *const f32,
                    pres as *mut f32,
                    pw as *const f32,
                    pout as *mut f32,
                    batch,
                    hidden,
                    self.eps as f32,
                )
            }
        };
        if rc != 0 {
            anyhow::bail!("rmsnorm_residual_f32 kernel returned {rc}");
        }

        let shape: candle_core::Shape = dims.into();
        let out_storage = candle_core::CudaStorage::wrap_cuda_slice(out_dev, dev.clone());
        let normed = candle_core::Tensor::from_storage(
            candle_core::Storage::Cuda(out_storage),
            shape.clone(),
            candle_core::op::BackpropOp::none(),
            false,
        );
        let res_storage = candle_core::CudaStorage::wrap_cuda_slice(new_res_dev, dev);
        let new_residual = candle_core::Tensor::from_storage(
            candle_core::Storage::Cuda(res_storage),
            shape,
            candle_core::op::BackpropOp::none(),
            false,
        );
        Ok((normed, new_residual))
    }
}

#[cfg(test)]
mod paper_validation {

    use super::*;
    use candle_core::Device;

    fn tensor(vals: &[f32], shape: (usize, usize)) -> Tensor {
        Tensor::from_vec(vals.to_vec(), shape, &Device::Cpu).unwrap()
    }

    fn forward(vals: &[f32], hidden: usize, w: &[f32], eps: f64) -> Vec<f32> {
        let x = tensor(vals, (vals.len() / hidden, hidden));
        let w = Tensor::from_vec(w.to_vec(), hidden, &Device::Cpu).unwrap();
        let y = RmsNorm::new(w, eps).forward(&x).unwrap();
        y.flatten_all().unwrap().to_vec1::<f32>().unwrap()
    }

    #[test]
    fn rmsnorm_matches_reference_formula_with_eps_inside_the_sqrt_a_plain_weight_multiplier_and_an_exact_zero_row(
    ) {
        let hidden = 8usize;
        let x_host: Vec<f32> = (0..2 * hidden)
            .map(|i| ((i as f32) * 0.37).sin() * 3.0)
            .collect();
        let w_host: Vec<f32> = (0..hidden).map(|i| 0.5 + 0.1 * i as f32).collect();
        let eps = 1e-6f64;
        let y_host = forward(&x_host, hidden, &w_host, eps);
        for r in 0..2 {
            let row = &x_host[r * hidden..(r + 1) * hidden];
            let ms: f32 = row.iter().map(|v| v * v).sum::<f32>() / hidden as f32;
            let denom = (ms + eps as f32).sqrt();
            for i in 0..hidden {
                let want = row[i] / denom * w_host[i];
                let got = y_host[r * hidden + i];
                assert!(
                    (want - got).abs() <= 1e-5 * want.abs().max(1e-3),
                    "row {r} dim {i}: got {got}, reference {want}"
                );
            }
        }

        let hidden = 4usize;
        let y_host = forward(&[1e-4f32; 4], hidden, &[1.0f32; 4], 1e-6);
        let inside = 1e-4f32 / (1e-8f32 + 1e-6).sqrt();
        let outside = 1e-4f32 / (1e-8f32.sqrt() + 1e-6);
        for &got in &y_host {
            assert!(
                (got - inside).abs() < 1e-3 * inside,
                "output {got} does not match eps-inside-sqrt reference {inside}"
            );
            assert!(
                (got - outside).abs() > 0.5 * outside,
                "output {got} suspiciously matches the eps-outside variant {outside}"
            );
        }

        for v in forward(&[0.0f32; 4], hidden, &[2.0f32; 4], 1e-6) {
            assert_eq!(
                v, 0.0,
                "zero input must stay exactly zero (eps guards the div)"
            );
        }

        assert_eq!(
            forward(&[2.0, -2.0, 2.0, -2.0], hidden, &[1.0f32; 4], 0.0),
            vec![1.0, -1.0, 1.0, -1.0],
            "the weight is a plain multiplier, not one plus w"
        );
    }
}

#[cfg(feature = "cuda")]
pub struct RmsNormTrainOp {
    w: Tensor,
    eps: f32,
    rstd: std::sync::Mutex<Option<Tensor>>,
}

#[cfg(feature = "cuda")]
fn train_op_ptr(
    t: &Tensor,
    stream: &std::sync::Arc<cudarc::driver::CudaStream>,
) -> candle_core::Result<u64> {
    use cudarc::driver::DevicePtr;
    let (st, l) = t.storage_and_layout();
    let candle_core::Storage::Cuda(c) = &*st else {
        return Err(candle_core::Error::Msg(
            "rmsnorm train op: operand not cuda".into(),
        ));
    };
    Ok(match t.dtype() {
        DType::F32 => {
            let (p, _g) = c.as_cuda_slice::<f32>()?.device_ptr(stream);
            p + (l.start_offset() * 4) as u64
        }
        DType::BF16 => {
            let (p, _g) = c.as_cuda_slice::<half::bf16>()?.device_ptr(stream);
            p + (l.start_offset() * 2) as u64
        }
        other => {
            return Err(candle_core::Error::Msg(format!(
                "rmsnorm train op: dtype {other:?}"
            )))
        }
    })
}

#[cfg(feature = "cuda")]
impl candle_core::CustomOp1 for RmsNormTrainOp {
    fn name(&self) -> &'static str {
        "rmsnorm_train"
    }

    fn cpu_fwd(
        &self,
        _s: &candle_core::CpuStorage,
        _l: &candle_core::Layout,
    ) -> candle_core::Result<(candle_core::CpuStorage, candle_core::Shape)> {
        Err(candle_core::Error::Msg("rmsnorm_train is CUDA-only".into()))
    }

    fn cuda_fwd(
        &self,
        s: &candle_core::CudaStorage,
        l: &candle_core::Layout,
    ) -> candle_core::Result<(candle_core::CudaStorage, candle_core::Shape)> {
        use candle_core::backend::BackendStorage;
        use cudarc::driver::{DevicePtr, DevicePtrMut};
        let dev = s.device().clone();
        let stream = crate::cuda_stream::current_stream(&dev);
        let dims = l.shape().dims().to_vec();
        let dim = *dims.last().unwrap();
        let rows: usize = dims.iter().product::<usize>() / dim;
        let Some((start, _end)) = l.contiguous_offsets() else {
            return Err(candle_core::Error::Msg(
                "rmsnorm train op: input must be contiguous".into(),
            ));
        };
        if self.w.elem_count() != dim {
            return Err(candle_core::Error::Msg(format!(
                "rmsnorm train op: weight {} vs dim {dim}",
                self.w.elem_count()
            )));
        }
        let wp = train_op_ptr(&self.w, &stream)?;
        let mut rstd: cudarc::driver::CudaSlice<f32> = unsafe { stream.alloc::<f32>(rows) }
            .map_err(|e| candle_core::Error::Msg(format!("rmsnorm train op alloc: {e:?}")))?;
        let rp = {
            let (p, _g) = rstd.device_ptr_mut(&stream);
            p
        };
        let (out_storage, rc) = match s.dtype() {
            DType::BF16 => {
                let x = s.as_cuda_slice::<half::bf16>()?;
                let xv = x.slice(start..);
                let (xp, _g1) = xv.device_ptr(&stream);
                let mut y: cudarc::driver::CudaSlice<half::bf16> = unsafe {
                    stream.alloc::<half::bf16>(rows * dim)
                }
                .map_err(|e| candle_core::Error::Msg(format!("rmsnorm train op alloc: {e:?}")))?;
                let rc = unsafe {
                    let (yp, _g2) = y.device_ptr_mut(&stream);
                    nv_kernels::cuda::rmsnorm_train_fwd(
                        stream.cu_stream() as *mut _,
                        xp as *const std::ffi::c_void,
                        wp as *const f32,
                        yp as *mut std::ffi::c_void,
                        rp as *mut f32,
                        rows as i64,
                        dim as i32,
                        self.eps,
                        1,
                    )
                };
                (
                    candle_core::CudaStorage::wrap_cuda_slice(y, dev.clone()),
                    rc,
                )
            }
            DType::F32 => {
                let x = s.as_cuda_slice::<f32>()?;
                let xv = x.slice(start..);
                let (xp, _g1) = xv.device_ptr(&stream);
                let mut y: cudarc::driver::CudaSlice<f32> = unsafe {
                    stream.alloc::<f32>(rows * dim)
                }
                .map_err(|e| candle_core::Error::Msg(format!("rmsnorm train op alloc: {e:?}")))?;
                let rc = unsafe {
                    let (yp, _g2) = y.device_ptr_mut(&stream);
                    nv_kernels::cuda::rmsnorm_train_fwd(
                        stream.cu_stream() as *mut _,
                        xp as *const std::ffi::c_void,
                        wp as *const f32,
                        yp as *mut std::ffi::c_void,
                        rp as *mut f32,
                        rows as i64,
                        dim as i32,
                        self.eps,
                        0,
                    )
                };
                (
                    candle_core::CudaStorage::wrap_cuda_slice(y, dev.clone()),
                    rc,
                )
            }
            other => {
                return Err(candle_core::Error::Msg(format!(
                    "rmsnorm train op: dtype {other:?}"
                )))
            }
        };
        if rc != 0 {
            return Err(candle_core::Error::Msg(format!(
                "rmsnorm_train_fwd rc={rc}"
            )));
        }
        let rstd_t = Tensor::from_storage(
            candle_core::Storage::Cuda(candle_core::CudaStorage::wrap_cuda_slice(rstd, dev)),
            rows,
            candle_core::op::BackpropOp::none(),
            false,
        );
        *self.rstd.lock().unwrap() = Some(rstd_t);
        Ok((out_storage, candle_core::Shape::from(dims)))
    }

    fn bwd(
        &self,
        arg: &Tensor,
        _res: &Tensor,
        grad_res: &Tensor,
    ) -> candle_core::Result<Option<Tensor>> {
        let candle_core::Device::Cuda(dev) = arg.device() else {
            return Err(candle_core::Error::Msg(
                "rmsnorm train op bwd: not cuda".into(),
            ));
        };
        let stream = crate::cuda_stream::current_stream(dev);
        let x = arg.detach().contiguous()?;
        let dims = x.dims().to_vec();
        let dim = *dims.last().unwrap();
        let rows = x.elem_count() / dim;
        let g = grad_res.detach().to_dtype(x.dtype())?.contiguous()?;
        let rstd =
            self.rstd.lock().unwrap().clone().ok_or_else(|| {
                candle_core::Error::Msg("rmsnorm train op bwd: no saved rstd".into())
            })?;
        let dx = crate::step_arena::output_tensor_uninitialised_for_a_kernel_to_fill(
            arg.device(),
            x.dtype(),
            &dims,
        )
        .map_err(|e| candle_core::Error::Msg(format!("{e:#}")))?;
        let rc = unsafe {
            nv_kernels::cuda::rmsnorm_train_bwd(
                stream.cu_stream() as *mut _,
                train_op_ptr(&x, &stream)? as *const std::ffi::c_void,
                train_op_ptr(&self.w, &stream)? as *const f32,
                train_op_ptr(&rstd, &stream)? as *const f32,
                train_op_ptr(&g, &stream)? as *const std::ffi::c_void,
                crate::step_arena::device_ptr_for_a_kernel_to_write(&dx)
                    .map_err(|e| candle_core::Error::Msg(format!("{e:#}")))?
                    as *mut std::ffi::c_void,
                rows as i64,
                dim as i32,
                if x.dtype() == DType::BF16 { 1 } else { 0 },
            )
        };
        if rc != 0 {
            return Err(candle_core::Error::Msg(format!(
                "rmsnorm_train_bwd rc={rc}"
            )));
        }
        Ok(Some(dx))
    }
}
