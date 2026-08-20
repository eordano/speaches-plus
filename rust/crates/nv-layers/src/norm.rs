use anyhow::Result;
use candle_core::{DType, Tensor};

#[cfg(feature = "cuda")]
use candle_core::Device;

pub struct RmsNorm {
    weight: Tensor,

    weight_bf16: Tensor,
    weight_f32: Tensor,
    eps: f64,
}

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
        Self {
            weight,
            weight_bf16,
            weight_f32,
            eps,
        }
    }

    pub fn from_candle_vb(vb: candle_nn::VarBuilder, hidden: usize, eps: f64) -> Result<Self> {
        let weight = vb.get(hidden, "weight")?;
        Ok(Self::new(weight, eps))
    }

    pub fn weight(&self) -> &Tensor {
        &self.weight
    }

    pub fn weight_bf16(&self) -> &Tensor {
        &self.weight_bf16
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
        let xf = x.to_dtype(DType::F32)?;
        let mean_sq = xf.sqr()?.mean_keepdim(candle_core::D::Minus1)?;
        let eps_t = Tensor::new(self.eps as f32, x.device())?;
        let denom = mean_sq.broadcast_add(&eps_t)?.sqrt()?;
        let normed = xf.broadcast_div(&denom)?;
        let w = self.weight_f32.clone();
        let out = normed.broadcast_mul(&w)?;
        Ok(out.to_dtype(x.dtype())?)
    }

    #[cfg(feature = "cuda")]
    fn forward_cuda_bf16(&self, x: &Tensor) -> Result<Tensor> {
        use cudarc::driver::{DevicePtr, DevicePtrMut};
        use half::bf16;
        use nv_kernels::cuda as nvk;

        let x_c = x.contiguous()?;
        let w_c = self.weight_bf16.clone();
        let dims = x_c.dims().to_vec();
        let hidden = *dims.last().unwrap();
        let batch: usize = dims[..dims.len() - 1].iter().product();

        let dev = match x_c.device() {
            Device::Cuda(d) => d.clone(),
            _ => unreachable!(),
        };
        let stream = crate::cuda_stream::current_stream(&dev);

        let mut y_dev: cudarc::driver::CudaSlice<bf16> = unsafe {
            stream
                .alloc::<bf16>(batch * hidden)
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
            let x_slice = x_cuda.as_cuda_slice::<bf16>()?;
            let x_view = x_slice.slice(xl.start_offset()..);
            let w_slice = w_cuda.as_cuda_slice::<bf16>()?;
            let w_view = w_slice.slice(_wl.start_offset()..);
            let (px, _gx) = x_view.device_ptr(&stream);
            let (pw, _gw) = w_view.device_ptr(&stream);
            let (py, _gy) = y_dev.device_ptr_mut(&stream);
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
        let w_c = self.weight_f32.clone();
        let dims = x_c.dims().to_vec();
        let hidden = *dims.last().unwrap();
        let batch: usize = dims[..dims.len() - 1].iter().product();

        let dev = match x_c.device() {
            Device::Cuda(d) => d.clone(),
            _ => unreachable!(),
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
        let w_c = self.weight_bf16.clone();
        let dims = x_c.dims().to_vec();
        let hidden = *dims.last().unwrap();
        let batch: usize = dims[..dims.len() - 1].iter().product();

        let dev = match x_c.device() {
            Device::Cuda(d) => d.clone(),
            _ => unreachable!(),
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
            let (ws, _wl) = w_c.storage_and_layout();
            let x_cuda = match &*xs {
                candle_core::Storage::Cuda(s) => s,
                _ => anyhow::bail!("expected cuda storage"),
            };
            let w_cuda = match &*ws {
                candle_core::Storage::Cuda(s) => s,
                _ => anyhow::bail!("expected cuda storage"),
            };
            let x_slice = x_cuda.as_cuda_slice::<bf16>()?;
            let x_view = x_slice.slice(xl.start_offset()..);
            let w_slice = w_cuda.as_cuda_slice::<bf16>()?;
            let w_view = w_slice.slice(_wl.start_offset()..);
            let (px, _gx) = x_view.device_ptr(&stream);
            let (pres, _gres) = new_res_dev.device_ptr_mut(&stream);
            let (pw, _gw) = w_view.device_ptr(&stream);
            let (pout, _gout) = out_dev.device_ptr_mut(&stream);
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
        let w_c = self.weight_f32.clone();
        let dims = x_c.dims().to_vec();
        let hidden = *dims.last().unwrap();
        let batch: usize = dims[..dims.len() - 1].iter().product();

        let dev = match x_c.device() {
            Device::Cuda(d) => d.clone(),
            _ => unreachable!(),
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

    #[test]
    fn rmsnorm_matches_reference_formula() {
        let hidden = 8usize;
        let x_host: Vec<f32> = (0..2 * hidden)
            .map(|i| ((i as f32) * 0.37).sin() * 3.0)
            .collect();
        let w_host: Vec<f32> = (0..hidden).map(|i| 0.5 + 0.1 * i as f32).collect();
        let eps = 1e-6f64;
        let x = tensor(&x_host, (2, hidden));
        let w = Tensor::from_vec(w_host.clone(), hidden, &Device::Cpu).unwrap();
        let norm = RmsNorm::new(w, eps);
        let y = norm.forward(&x).unwrap();
        let y_host = y.flatten_all().unwrap().to_vec1::<f32>().unwrap();
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
    }

    #[test]
    fn rmsnorm_eps_is_inside_the_sqrt() {
        let hidden = 4usize;
        let x_host = vec![1e-4f32; hidden];
        let eps = 1e-6f64;
        let x = tensor(&x_host, (1, hidden));
        let w = Tensor::from_vec(vec![1.0f32; hidden], hidden, &Device::Cpu).unwrap();
        let norm = RmsNorm::new(w, eps);
        let y = norm.forward(&x).unwrap();
        let y_host = y.flatten_all().unwrap().to_vec1::<f32>().unwrap();
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
    }

    #[test]
    fn rmsnorm_zero_row_is_finite_and_zero() {
        let hidden = 4usize;
        let x = tensor(&vec![0.0f32; hidden], (1, hidden));
        let w = Tensor::from_vec(vec![2.0f32; hidden], hidden, &Device::Cpu).unwrap();
        let norm = RmsNorm::new(w, 1e-6);
        let y = norm.forward(&x).unwrap();
        for v in y.flatten_all().unwrap().to_vec1::<f32>().unwrap() {
            assert_eq!(
                v, 0.0,
                "zero input must stay exactly zero (eps guards the div)"
            );
        }
    }

    #[test]
    fn rmsnorm_weight_is_a_plain_multiplier_not_one_plus_w() {
        let hidden = 4usize;
        let x = tensor(&[2.0, -2.0, 2.0, -2.0], (1, hidden));
        let w = Tensor::from_vec(vec![1.0f32; hidden], hidden, &Device::Cpu).unwrap();
        let norm = RmsNorm::new(w, 0.0);
        let y = norm.forward(&x).unwrap();
        let y_host = y.flatten_all().unwrap().to_vec1::<f32>().unwrap();

        assert_eq!(y_host, vec![1.0, -1.0, 1.0, -1.0]);
    }
}
