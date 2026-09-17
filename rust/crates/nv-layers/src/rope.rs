use anyhow::Result;
use candle_core::{DType, Device, Tensor};

#[derive(Clone, Copy, Debug)]
pub enum RopeKind {
    Standard,
    NtkAware,
    Yarn,
}

#[derive(Clone, Copy, Debug)]
pub struct RopeConfig {
    pub head_dim: usize,
    pub max_seq_len: usize,
    pub base: f32,
    pub kind: RopeKind,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RopeTablePrecision {
    F32,
    F64,
}

pub fn rope_table_precision_from_env() -> RopeTablePrecision {
    match std::env::var("NV_ROPE_TABLE").ok().as_deref() {
        Some("f64") => RopeTablePrecision::F64,
        _ => RopeTablePrecision::F32,
    }
}

pub fn build_rope_tables_f32(inv_freq: &[f32], rows: usize) -> (Vec<f32>, Vec<f32>) {
    let half = inv_freq.len();
    let mut cos_host = vec![0f32; rows * half];
    let mut sin_host = vec![0f32; rows * half];
    for p in 0..rows {
        for i in 0..half {
            let theta = (p as f32) * inv_freq[i];
            cos_host[p * half + i] = theta.cos();
            sin_host[p * half + i] = theta.sin();
        }
    }
    (cos_host, sin_host)
}

pub fn build_rope_tables_f64(inv_freq: &[f64], rows: usize) -> (Vec<f32>, Vec<f32>) {
    let half = inv_freq.len();
    let mut cos_host = vec![0f32; rows * half];
    let mut sin_host = vec![0f32; rows * half];
    for p in 0..rows {
        for i in 0..half {
            let theta = (p as f64) * inv_freq[i];
            cos_host[p * half + i] = theta.cos() as f32;
            sin_host[p * half + i] = theta.sin() as f32;
        }
    }
    (cos_host, sin_host)
}

pub struct Rope {
    cfg: RopeConfig,
    cos: Tensor,
    sin: Tensor,
}

impl Rope {
    pub fn new(cfg: RopeConfig, device: &Device) -> Result<Self> {
        if !cfg.head_dim.is_multiple_of(2) {
            anyhow::bail!("head_dim must be even, got {}", cfg.head_dim);
        }
        let half = cfg.head_dim / 2;
        match rope_table_precision_from_env() {
            RopeTablePrecision::F32 => {
                let inv_freq: Vec<f32> = (0..half)
                    .map(|i| 1.0 / cfg.base.powf((i as f32 * 2.0) / (cfg.head_dim as f32)))
                    .collect();
                Self::from_inv_freq(cfg, &inv_freq, device)
            }
            RopeTablePrecision::F64 => {
                let base = cfg.base as f64;
                let inv_freq: Vec<f64> = (0..half)
                    .map(|i| 1.0 / base.powf((i as f64 * 2.0) / (cfg.head_dim as f64)))
                    .collect();
                Self::from_inv_freq_f64(cfg, &inv_freq, device)
            }
        }
    }

    pub fn from_inv_freq(cfg: RopeConfig, inv_freq: &[f32], device: &Device) -> Result<Self> {
        let half = Self::check_shape(&cfg, inv_freq.len())?;
        let (cos_host, sin_host) = match rope_table_precision_from_env() {
            RopeTablePrecision::F32 => build_rope_tables_f32(inv_freq, cfg.max_seq_len),
            RopeTablePrecision::F64 => {
                let wide: Vec<f64> = inv_freq.iter().map(|v| *v as f64).collect();
                build_rope_tables_f64(&wide, cfg.max_seq_len)
            }
        };
        let cos = Tensor::from_vec(cos_host, (cfg.max_seq_len, half), device)?;
        let sin = Tensor::from_vec(sin_host, (cfg.max_seq_len, half), device)?;
        Ok(Self { cfg, cos, sin })
    }

    pub fn from_precomputed_tables_one_row_per_token_so_mrope_rides_the_standard_apply(
        cfg: RopeConfig,
        cos: Tensor,
        sin: Tensor,
    ) -> Result<Self> {
        let half = cfg.head_dim / 2;
        let want = [cfg.max_seq_len, half];
        if cos.dims() != want || sin.dims() != want {
            anyhow::bail!(
                "rope from_precomputed_tables: cos {:?} sin {:?} must both be {:?}",
                cos.dims(),
                sin.dims(),
                want
            );
        }
        if cos.dtype() != DType::F32 || sin.dtype() != DType::F32 {
            anyhow::bail!(
                "rope from_precomputed_tables: tables must be F32, got {:?}/{:?}",
                cos.dtype(),
                sin.dtype()
            );
        }
        Ok(Self { cfg, cos, sin })
    }

    pub fn from_inv_freq_f64(cfg: RopeConfig, inv_freq: &[f64], device: &Device) -> Result<Self> {
        let half = Self::check_shape(&cfg, inv_freq.len())?;
        let (cos_host, sin_host) = build_rope_tables_f64(inv_freq, cfg.max_seq_len);
        let cos = Tensor::from_vec(cos_host, (cfg.max_seq_len, half), device)?;
        let sin = Tensor::from_vec(sin_host, (cfg.max_seq_len, half), device)?;
        Ok(Self { cfg, cos, sin })
    }

    fn check_shape(cfg: &RopeConfig, inv_len: usize) -> Result<usize> {
        if !cfg.head_dim.is_multiple_of(2) {
            anyhow::bail!("head_dim must be even, got {}", cfg.head_dim);
        }
        let half = cfg.head_dim / 2;
        if inv_len != half {
            anyhow::bail!("inv_freq length {} != head_dim/2 = {}", inv_len, half);
        }
        Ok(half)
    }

    pub fn cos(&self) -> &Tensor {
        &self.cos
    }

    pub fn sin(&self) -> &Tensor {
        &self.sin
    }

    pub fn config(&self) -> &RopeConfig {
        &self.cfg
    }

    pub fn apply(&self, q: &Tensor, k: &Tensor, positions: &Tensor) -> Result<(Tensor, Tensor)> {
        #[cfg(feature = "cuda")]
        if matches!(q.device(), Device::Cuda(_)) {
            if q.dtype() == DType::F32 {
                return self.apply_cuda_f32(q, k, positions);
            }
            if q.dtype() == DType::BF16 && k.dtype() == DType::BF16 {
                return self.apply_cuda_bf16(q, k, positions);
            }
        }
        self.apply_candle(q, k, positions)
    }

    pub fn apply_candle(
        &self,
        q: &Tensor,
        k: &Tensor,
        positions: &Tensor,
    ) -> Result<(Tensor, Tensor)> {
        let q_out = self.apply_one_candle(q, positions)?;
        let k_out = self.apply_one_candle(k, positions)?;
        Ok((q_out, k_out))
    }

    fn apply_one_candle(&self, x: &Tensor, positions: &Tensor) -> Result<Tensor> {
        let dtype = x.dtype();
        let dims = x.dims().to_vec();
        let head_dim = *dims.last().unwrap();
        let n_heads = dims[dims.len() - 2];
        if head_dim != self.cfg.head_dim {
            anyhow::bail!(
                "rope: head_dim {} != config {}",
                head_dim,
                self.cfg.head_dim
            );
        }
        let half = head_dim / 2;
        let tokens: usize = dims[..dims.len() - 2].iter().product();

        let xf = x
            .to_dtype(DType::F32)?
            .reshape((tokens, n_heads, head_dim))?;
        let positions_flat = normalize_positions_u32(positions, x.device())?;
        let cos = self.cos.index_select(&positions_flat, 0)?;
        let sin = self.sin.index_select(&positions_flat, 0)?;
        let cos = cos.unsqueeze(1)?;
        let sin = sin.unsqueeze(1)?;

        let lo = xf.narrow(2, 0, half)?;
        let hi = xf.narrow(2, half, half)?;
        let out_lo = lo.broadcast_mul(&cos)?.sub(&hi.broadcast_mul(&sin)?)?;
        let out_hi = lo.broadcast_mul(&sin)?.add(&hi.broadcast_mul(&cos)?)?;
        let out = Tensor::cat(&[&out_lo, &out_hi], 2)?;
        let out = out.reshape(dims)?.to_dtype(dtype)?;
        Ok(out)
    }

    #[cfg(feature = "cuda")]
    fn apply_cuda_f32(
        &self,
        q: &Tensor,
        k: &Tensor,
        positions: &Tensor,
    ) -> Result<(Tensor, Tensor)> {
        if crate::linear::this_call_is_building_an_autograd_graph(q) {
            return self.apply_candle(q, k, positions);
        }
        use cudarc::driver::{DevicePtr, DevicePtrMut};
        use nv_kernels::cuda as nvk;

        let q_dims = q.dims().to_vec();
        let k_dims = k.dims().to_vec();
        let head_dim = *q_dims.last().unwrap();
        if head_dim != self.cfg.head_dim {
            anyhow::bail!(
                "rope: head_dim {} != config {}",
                head_dim,
                self.cfg.head_dim
            );
        }
        let n_heads = q_dims[q_dims.len() - 2];
        let n_kv_heads = k_dims[k_dims.len() - 2];
        let tokens: usize = q_dims[..q_dims.len() - 2].iter().product();
        let k_tokens: usize = k_dims[..k_dims.len() - 2].iter().product();
        if k_tokens != tokens {
            anyhow::bail!("rope: q tokens {} != k tokens {}", tokens, k_tokens);
        }

        let q_in = q.contiguous()?.reshape((tokens, n_heads, head_dim))?;
        let k_in = k.contiguous()?.reshape((tokens, n_kv_heads, head_dim))?;
        let cos_c = self.cos.contiguous()?;
        let sin_c = self.sin.contiguous()?;

        let dev = match q_in.device() {
            Device::Cuda(d) => d.clone(),
            _ => unreachable!(),
        };
        let stream = crate::cuda_stream::current_stream(&dev);

        let mut q_out: cudarc::driver::CudaSlice<f32> = stream
            .alloc_zeros::<f32>(tokens * n_heads * head_dim)
            .map_err(|e| anyhow::anyhow!(e))?;
        let mut k_out: cudarc::driver::CudaSlice<f32> = stream
            .alloc_zeros::<f32>(tokens * n_kv_heads * head_dim)
            .map_err(|e| anyhow::anyhow!(e))?;

        let pos_i32 = positions_to_i32_cuda(positions, q_in.device())?;

        let rc = {
            let (qs, _ql) = q_in.storage_and_layout();
            let (ks, _kl) = k_in.storage_and_layout();
            let (cs, _cl) = cos_c.storage_and_layout();
            let (ss, _sl) = sin_c.storage_and_layout();
            let (ps, _pl) = pos_i32.storage_and_layout();
            let q_cuda = match &*qs {
                candle_core::Storage::Cuda(s) => s,
                _ => anyhow::bail!("expected cuda storage"),
            };
            let k_cuda = match &*ks {
                candle_core::Storage::Cuda(s) => s,
                _ => anyhow::bail!("expected cuda storage"),
            };
            let c_cuda = match &*cs {
                candle_core::Storage::Cuda(s) => s,
                _ => anyhow::bail!("expected cuda storage"),
            };
            let s_cuda = match &*ss {
                candle_core::Storage::Cuda(s) => s,
                _ => anyhow::bail!("expected cuda storage"),
            };
            let p_cuda = match &*ps {
                candle_core::Storage::Cuda(s) => s,
                _ => anyhow::bail!("expected cuda storage"),
            };
            let q_slice = q_cuda.as_cuda_slice::<f32>()?;
            let k_slice = k_cuda.as_cuda_slice::<f32>()?;
            let c_slice = c_cuda.as_cuda_slice::<f32>()?;
            let s_slice = s_cuda.as_cuda_slice::<f32>()?;
            let p_slice = p_cuda.as_cuda_slice::<i32>()?;
            stream
                .memcpy_dtod(q_slice, &mut q_out)
                .map_err(|e| anyhow::anyhow!(e))?;
            stream
                .memcpy_dtod(k_slice, &mut k_out)
                .map_err(|e| anyhow::anyhow!(e))?;
            let (pq, _gq) = q_out.device_ptr_mut(&stream);
            let (pk, _gk) = k_out.device_ptr_mut(&stream);
            let (pc, _gc) = c_slice.device_ptr(&stream);
            let (psn, _gs) = s_slice.device_ptr(&stream);
            let (pp, _gp) = p_slice.device_ptr(&stream);
            unsafe {
                nvk::rope_f32(
                    stream.cu_stream() as *mut _,
                    pq as *mut f32,
                    pk as *mut f32,
                    pc as *const f32,
                    psn as *const f32,
                    pp as *const i32,
                    tokens,
                    n_heads,
                    n_kv_heads,
                    head_dim,
                )
            }
        };
        if rc != 0 {
            anyhow::bail!("rope_f32 kernel returned {rc}");
        }

        let q_storage = candle_core::Storage::Cuda(candle_core::CudaStorage::wrap_cuda_slice(
            q_out,
            dev.clone(),
        ));
        let k_storage =
            candle_core::Storage::Cuda(candle_core::CudaStorage::wrap_cuda_slice(k_out, dev));
        let q_shape: candle_core::Shape = (tokens, n_heads, head_dim).into();
        let k_shape: candle_core::Shape = (tokens, n_kv_heads, head_dim).into();
        let q_t = candle_core::Tensor::from_storage(
            q_storage,
            q_shape,
            candle_core::op::BackpropOp::none(),
            false,
        )
        .reshape(q_dims)?;
        let k_t = candle_core::Tensor::from_storage(
            k_storage,
            k_shape,
            candle_core::op::BackpropOp::none(),
            false,
        )
        .reshape(k_dims)?;
        Ok((q_t, k_t))
    }
}

#[cfg(feature = "cuda")]
impl Rope {
    fn apply_cuda_bf16(
        &self,
        q: &Tensor,
        k: &Tensor,
        positions: &Tensor,
    ) -> Result<(Tensor, Tensor)> {
        if crate::linear::this_call_is_building_an_autograd_graph(q) {
            return self.apply_candle(q, k, positions);
        }
        use cudarc::driver::{DevicePtr, DevicePtrMut};
        use half::bf16;
        use nv_kernels::cuda as nvk;

        let q_dims = q.dims().to_vec();
        let k_dims = k.dims().to_vec();
        let head_dim = *q_dims.last().unwrap();
        if head_dim != self.cfg.head_dim {
            anyhow::bail!(
                "rope: head_dim {} != config {}",
                head_dim,
                self.cfg.head_dim
            );
        }
        let n_heads = q_dims[q_dims.len() - 2];
        let n_kv_heads = k_dims[k_dims.len() - 2];
        let tokens: usize = q_dims[..q_dims.len() - 2].iter().product();
        let k_tokens: usize = k_dims[..k_dims.len() - 2].iter().product();
        if k_tokens != tokens {
            anyhow::bail!("rope: q tokens {} != k tokens {}", tokens, k_tokens);
        }

        let q_in = q.contiguous()?.reshape((tokens, n_heads, head_dim))?;
        let k_in = k.contiguous()?.reshape((tokens, n_kv_heads, head_dim))?;
        let cos_c = self.cos.contiguous()?;
        let sin_c = self.sin.contiguous()?;

        let dev = match q_in.device() {
            Device::Cuda(d) => d.clone(),
            _ => unreachable!(),
        };
        let stream = crate::cuda_stream::current_stream(&dev);

        let mut q_out: cudarc::driver::CudaSlice<bf16> = unsafe {
            stream
                .alloc::<bf16>(tokens * n_heads * head_dim)
                .map_err(|e| anyhow::anyhow!(e))?
        };
        let mut k_out: cudarc::driver::CudaSlice<bf16> = unsafe {
            stream
                .alloc::<bf16>(tokens * n_kv_heads * head_dim)
                .map_err(|e| anyhow::anyhow!(e))?
        };

        let pos_i32 = positions_to_i32_cuda(positions, q_in.device())?;

        let rc = {
            let (qs, ql) = q_in.storage_and_layout();
            let (ks, kl) = k_in.storage_and_layout();
            let (cs, _cl) = cos_c.storage_and_layout();
            let (ss, _sl) = sin_c.storage_and_layout();
            let (ps, _pl) = pos_i32.storage_and_layout();
            let q_cuda = match &*qs {
                candle_core::Storage::Cuda(s) => s,
                _ => anyhow::bail!("expected cuda storage"),
            };
            let k_cuda = match &*ks {
                candle_core::Storage::Cuda(s) => s,
                _ => anyhow::bail!("expected cuda storage"),
            };
            let c_cuda = match &*cs {
                candle_core::Storage::Cuda(s) => s,
                _ => anyhow::bail!("expected cuda storage"),
            };
            let s_cuda = match &*ss {
                candle_core::Storage::Cuda(s) => s,
                _ => anyhow::bail!("expected cuda storage"),
            };
            let p_cuda = match &*ps {
                candle_core::Storage::Cuda(s) => s,
                _ => anyhow::bail!("expected cuda storage"),
            };

            let q_slice = q_cuda.as_cuda_slice::<bf16>()?;
            let q_view = q_slice.slice(ql.start_offset()..);
            let k_slice = k_cuda.as_cuda_slice::<bf16>()?;
            let k_view = k_slice.slice(kl.start_offset()..);
            let c_slice = c_cuda.as_cuda_slice::<f32>()?;
            let s_slice = s_cuda.as_cuda_slice::<f32>()?;
            let p_slice = p_cuda.as_cuda_slice::<i32>()?;
            let (pqi, _gqi) = q_view.device_ptr(&stream);
            let (pki, _gki) = k_view.device_ptr(&stream);
            let (pq, _gq) = q_out.device_ptr_mut(&stream);
            let (pk, _gk) = k_out.device_ptr_mut(&stream);
            let (pc, _gc) = c_slice.device_ptr(&stream);
            let (psn, _gs) = s_slice.device_ptr(&stream);
            let (pp, _gp) = p_slice.device_ptr(&stream);
            unsafe {
                nvk::rope_bf16_oop(
                    stream.cu_stream() as *mut _,
                    pqi as *const u16,
                    pki as *const u16,
                    pq as *mut u16,
                    pk as *mut u16,
                    pc as *const f32,
                    psn as *const f32,
                    pp as *const i32,
                    tokens,
                    n_heads,
                    n_kv_heads,
                    head_dim,
                )
            }
        };
        if rc != 0 {
            anyhow::bail!("rope_bf16_oop kernel returned {rc}");
        }

        let q_storage = candle_core::Storage::Cuda(candle_core::CudaStorage::wrap_cuda_slice(
            q_out,
            dev.clone(),
        ));
        let k_storage =
            candle_core::Storage::Cuda(candle_core::CudaStorage::wrap_cuda_slice(k_out, dev));
        let q_shape: candle_core::Shape = (tokens, n_heads, head_dim).into();
        let k_shape: candle_core::Shape = (tokens, n_kv_heads, head_dim).into();
        let q_t = candle_core::Tensor::from_storage(
            q_storage,
            q_shape,
            candle_core::op::BackpropOp::none(),
            false,
        )
        .reshape(q_dims)?;
        let k_t = candle_core::Tensor::from_storage(
            k_storage,
            k_shape,
            candle_core::op::BackpropOp::none(),
            false,
        )
        .reshape(k_dims)?;
        Ok((q_t, k_t))
    }

    pub fn apply_partial_oop_bf16_cuda_one_launch_replacing_slice_cast_rope_cast_concat(
        &self,
        q: &Tensor,
        k: &Tensor,
        positions: &Tensor,
        head_dim: usize,
    ) -> Result<(Tensor, Tensor)> {
        use cudarc::driver::{DevicePtr, DevicePtrMut};
        use half::bf16;
        use nv_kernels::cuda as nvk;

        let rotary_dim = self.cfg.head_dim;
        anyhow::ensure!(
            rotary_dim < head_dim && rotary_dim.is_multiple_of(2),
            "partial rope: rotary {rotary_dim} must be even and below head_dim {head_dim}"
        );
        anyhow::ensure!(
            q.dtype() == DType::BF16 && k.dtype() == DType::BF16,
            "partial rope kernel is bf16-only"
        );
        let q_dims = q.dims().to_vec();
        let k_dims = k.dims().to_vec();
        anyhow::ensure!(
            *q_dims.last().unwrap() == head_dim && *k_dims.last().unwrap() == head_dim,
            "partial rope: q/k last dim must be head_dim {head_dim}"
        );
        let n_heads = q_dims[q_dims.len() - 2];
        let n_kv_heads = k_dims[k_dims.len() - 2];
        let tokens: usize = q_dims[..q_dims.len() - 2].iter().product();
        anyhow::ensure!(
            k_dims[..k_dims.len() - 2].iter().product::<usize>() == tokens,
            "partial rope: q/k token counts differ"
        );

        let q_in = q.contiguous()?.reshape((tokens, n_heads, head_dim))?;
        let k_in = k.contiguous()?.reshape((tokens, n_kv_heads, head_dim))?;
        let cos_c = self.cos.contiguous()?;
        let sin_c = self.sin.contiguous()?;
        let dev = match q_in.device() {
            Device::Cuda(d) => d.clone(),
            _ => anyhow::bail!("partial rope kernel needs CUDA"),
        };
        let stream = crate::cuda_stream::current_stream(&dev);
        let mut q_out: cudarc::driver::CudaSlice<bf16> = unsafe {
            stream
                .alloc::<bf16>(tokens * n_heads * head_dim)
                .map_err(|e| anyhow::anyhow!(e))?
        };
        let mut k_out: cudarc::driver::CudaSlice<bf16> = unsafe {
            stream
                .alloc::<bf16>(tokens * n_kv_heads * head_dim)
                .map_err(|e| anyhow::anyhow!(e))?
        };
        let pos_i32 = positions_to_i32_cuda(positions, q_in.device())?;
        let rc = {
            let (qs, ql) = q_in.storage_and_layout();
            let (ks, kl) = k_in.storage_and_layout();
            let (cs, _cl) = cos_c.storage_and_layout();
            let (ss, _sl) = sin_c.storage_and_layout();
            let (ps, _pl) = pos_i32.storage_and_layout();
            let (q_cuda, k_cuda, c_cuda, s_cuda, p_cuda) = match (&*qs, &*ks, &*cs, &*ss, &*ps) {
                (
                    candle_core::Storage::Cuda(a),
                    candle_core::Storage::Cuda(b),
                    candle_core::Storage::Cuda(c),
                    candle_core::Storage::Cuda(d),
                    candle_core::Storage::Cuda(e),
                ) => (a, b, c, d, e),
                _ => anyhow::bail!("expected cuda storage"),
            };
            let q_view = q_cuda.as_cuda_slice::<bf16>()?.slice(ql.start_offset()..);
            let k_view = k_cuda.as_cuda_slice::<bf16>()?.slice(kl.start_offset()..);
            let c_slice = c_cuda.as_cuda_slice::<f32>()?;
            let s_slice = s_cuda.as_cuda_slice::<f32>()?;
            let p_slice = p_cuda.as_cuda_slice::<i32>()?;
            let (pqi, _gqi) = q_view.device_ptr(&stream);
            let (pki, _gki) = k_view.device_ptr(&stream);
            let (pq, _gq) = q_out.device_ptr_mut(&stream);
            let (pk, _gk) = k_out.device_ptr_mut(&stream);
            let (pc, _gc) = c_slice.device_ptr(&stream);
            let (psn, _gs) = s_slice.device_ptr(&stream);
            let (pp, _gp) = p_slice.device_ptr(&stream);
            unsafe {
                nvk::rope_bf16_partial_oop(
                    stream.cu_stream() as *mut _,
                    pqi as *const u16,
                    pki as *const u16,
                    pq as *mut u16,
                    pk as *mut u16,
                    pc as *const f32,
                    psn as *const f32,
                    pp as *const i32,
                    tokens,
                    n_heads,
                    n_kv_heads,
                    head_dim,
                    rotary_dim,
                )
            }
        };
        anyhow::ensure!(rc == 0, "rope_bf16_partial_oop kernel returned {rc}");
        let q_storage = candle_core::Storage::Cuda(candle_core::CudaStorage::wrap_cuda_slice(
            q_out,
            dev.clone(),
        ));
        let k_storage =
            candle_core::Storage::Cuda(candle_core::CudaStorage::wrap_cuda_slice(k_out, dev));
        let q_t = candle_core::Tensor::from_storage(
            q_storage,
            candle_core::Shape::from((tokens, n_heads, head_dim)),
            candle_core::op::BackpropOp::none(),
            false,
        )
        .reshape(q_dims)?;
        let k_t = candle_core::Tensor::from_storage(
            k_storage,
            candle_core::Shape::from((tokens, n_kv_heads, head_dim)),
            candle_core::op::BackpropOp::none(),
            false,
        )
        .reshape(k_dims)?;
        Ok((q_t, k_t))
    }
}

pub fn apply_rope(
    q: &Tensor,
    k: &Tensor,
    positions: &Tensor,
    rope: &Rope,
) -> Result<(Tensor, Tensor)> {
    rope.apply(q, k, positions)
}

impl Rope {
    pub fn apply_mrope(
        &self,
        q: &Tensor,
        k: &Tensor,
        positions: &[&Tensor],
        sections: &[usize],
    ) -> Result<(Tensor, Tensor)> {
        if positions.len() != sections.len() {
            anyhow::bail!(
                "apply_mrope: positions ({}) and sections ({}) length mismatch",
                positions.len(),
                sections.len()
            );
        }
        if positions.is_empty() {
            anyhow::bail!("apply_mrope: empty sections / positions");
        }
        let half = self.cfg.head_dim / 2;
        let sum: usize = sections.iter().sum();
        if sum != half {
            anyhow::bail!("apply_mrope: sections sum {} != head_dim/2 {}", sum, half);
        }

        if positions.len() >= 2 {
            let first_host = host_positions(positions[0])?;
            let mut all_equal = true;
            for p in &positions[1..] {
                let h = host_positions(p)?;
                if h != first_host {
                    all_equal = false;
                    break;
                }
            }
            if all_equal {
                return self.apply(q, k, positions[0]);
            }
        }

        let q_out = self.apply_one_mrope(q, positions, sections)?;
        let k_out = self.apply_one_mrope(k, positions, sections)?;
        Ok((q_out, k_out))
    }

    fn apply_one_mrope(
        &self,
        x: &Tensor,
        positions: &[&Tensor],
        sections: &[usize],
    ) -> Result<Tensor> {
        let dtype = x.dtype();
        let dims = x.dims().to_vec();
        if dims.len() < 2 {
            anyhow::bail!("apply_mrope: input must be rank >= 2, got {:?}", dims);
        }
        let head_dim = *dims.last().unwrap();
        let n_heads = dims[dims.len() - 2];
        if head_dim != self.cfg.head_dim {
            anyhow::bail!(
                "apply_mrope: head_dim {} != config {}",
                head_dim,
                self.cfg.head_dim
            );
        }
        let half = head_dim / 2;
        let tokens: usize = dims[..dims.len() - 2].iter().product();

        let mut cos_pieces: Vec<Tensor> = Vec::with_capacity(positions.len());
        let mut sin_pieces: Vec<Tensor> = Vec::with_capacity(positions.len());
        let mut offset = 0usize;
        for (axis, &width) in sections.iter().enumerate() {
            let pos_flat = normalize_positions_u32(positions[axis], x.device())?;
            if pos_flat.elem_count() != tokens {
                anyhow::bail!(
                    "apply_mrope: positions[{}] has {} elements, expected {}",
                    axis,
                    pos_flat.elem_count(),
                    tokens
                );
            }
            let cos_axis = self.cos.index_select(&pos_flat, 0)?;
            let sin_axis = self.sin.index_select(&pos_flat, 0)?;
            let cos_slice = cos_axis.narrow(1, offset, width)?;
            let sin_slice = sin_axis.narrow(1, offset, width)?;
            cos_pieces.push(cos_slice);
            sin_pieces.push(sin_slice);
            offset += width;
        }
        let cos = Tensor::cat(&cos_pieces, 1)?;
        let sin = Tensor::cat(&sin_pieces, 1)?;
        let cos = cos.unsqueeze(1)?;
        let sin = sin.unsqueeze(1)?;

        let xf = x
            .to_dtype(DType::F32)?
            .reshape((tokens, n_heads, head_dim))?;
        let lo = xf.narrow(2, 0, half)?;
        let hi = xf.narrow(2, half, half)?;
        let out_lo = lo.broadcast_mul(&cos)?.sub(&hi.broadcast_mul(&sin)?)?;
        let out_hi = lo.broadcast_mul(&sin)?.add(&hi.broadcast_mul(&cos)?)?;
        let out = Tensor::cat(&[&out_lo, &out_hi], 2)?;
        let out = out.reshape(dims)?.to_dtype(dtype)?;
        Ok(out)
    }
}

fn host_positions(positions: &Tensor) -> Result<Vec<u32>> {
    let n = positions.elem_count();
    let flat = positions.reshape(n)?;
    let cpu = flat.to_device(&Device::Cpu)?;
    Ok(match cpu.dtype() {
        DType::U32 => cpu.to_vec1::<u32>()?,
        DType::U8 => cpu.to_vec1::<u8>()?.into_iter().map(|v| v as u32).collect(),
        DType::I64 => cpu
            .to_vec1::<i64>()?
            .into_iter()
            .map(|v| v as u32)
            .collect(),
        DType::I32 => cpu
            .to_vec1::<i32>()?
            .into_iter()
            .map(|v| v as u32)
            .collect(),
        other => anyhow::bail!("host_positions: unsupported dtype {other:?}"),
    })
}

#[cfg(feature = "cuda")]
fn positions_to_i32_cuda(positions: &Tensor, target_device: &Device) -> Result<Tensor> {
    let n = positions.elem_count();
    let flat = positions.reshape(n)?;
    if flat.dtype() == DType::I32 && flat.device().same_device(target_device) {
        return Ok(flat.contiguous()?);
    }
    let cpu = flat.to_device(&Device::Cpu)?;
    let host_i32: Vec<i32> = match cpu.dtype() {
        DType::I32 => cpu.to_vec1::<i32>()?,
        DType::U32 => cpu
            .to_vec1::<u32>()?
            .into_iter()
            .map(|v| v as i32)
            .collect(),
        DType::U8 => cpu.to_vec1::<u8>()?.into_iter().map(|v| v as i32).collect(),
        DType::I64 => cpu
            .to_vec1::<i64>()?
            .into_iter()
            .map(|v| v as i32)
            .collect(),
        other => anyhow::bail!("positions_to_i32_cuda: unsupported dtype {other:?}"),
    };
    Ok(Tensor::from_vec(host_i32, n, target_device)?)
}

pub const ROPE_POS_RESIDENT_SKIPS_A_DEVICE_ROUNDTRIP_THAT_FENCES_THE_WHOLE_METAL_QUEUE: &str =
    "normalize_positions_u32 round-trips positions through the host even when they are already \
     u32 on the target device, and on Metal that to_vec1 is a wait_until_completed drain of \
     every queued command -- the qlora window forward pays it twice per layer per bucket (96 \
     pipeline drains per step). NV_ROPE_POS_RESIDENT=1 returns the resident u32 tensor as-is; \
     index_select consumes identical index values from identical bytes, so the arm is bitwise \
     by construction";

fn rope_pos_resident_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("NV_ROPE_POS_RESIDENT").as_deref() == Ok("1"))
}

fn normalize_positions_u32(positions: &Tensor, target_device: &Device) -> Result<Tensor> {
    let n = positions.elem_count();
    let flat = positions.reshape(n)?;
    if rope_pos_resident_enabled()
        && flat.dtype() == DType::U32
        && flat.device().same_device(target_device)
    {
        let _ = ROPE_POS_RESIDENT_SKIPS_A_DEVICE_ROUNDTRIP_THAT_FENCES_THE_WHOLE_METAL_QUEUE;
        return Ok(flat.contiguous()?);
    }
    let host_u32: Vec<u32> = match flat.dtype() {
        DType::U32 => flat.to_device(&Device::Cpu)?.to_vec1::<u32>()?,
        DType::U8 => flat
            .to_device(&Device::Cpu)?
            .to_vec1::<u8>()?
            .into_iter()
            .map(|v| v as u32)
            .collect(),
        DType::I64 => flat
            .to_device(&Device::Cpu)?
            .to_vec1::<i64>()?
            .into_iter()
            .map(|v| v as u32)
            .collect(),
        DType::I32 => flat
            .to_device(&Device::Cpu)?
            .to_vec1::<i32>()?
            .into_iter()
            .map(|v| v as u32)
            .collect(),
        other => anyhow::bail!("unsupported positions dtype {other:?}"),
    };
    Ok(Tensor::from_vec(host_u32, n, target_device)?)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn det_tensor(shape: (usize, usize, usize), device: &Device) -> Tensor {
        let (t, h, d) = shape;
        let n = t * h * d;
        let mut v = Vec::with_capacity(n);
        for i in 0..n {
            let x = ((i as f32) * 0.137).sin();
            v.push(x);
        }
        Tensor::from_vec(v, shape, device).unwrap()
    }

    #[test]
    fn mrope_collapses_to_1d_rope_when_positions_match() {
        let device = Device::Cpu;
        let head_dim = 128usize;
        let sections = vec![24usize, 20, 20];

        let cfg = RopeConfig {
            head_dim,
            max_seq_len: 16,
            base: 1_000_000.0,
            kind: RopeKind::Standard,
        };
        let rope = Rope::new(cfg, &device).unwrap();

        let q = det_tensor((4, 2, head_dim), &device);
        let k = det_tensor((4, 1, head_dim), &device);

        let pos_vec = vec![0u32, 1, 2, 3];
        let positions = Tensor::from_vec(pos_vec.clone(), 4usize, &device).unwrap();

        let (q1, k1) = rope.apply(&q, &k, &positions).unwrap();

        let pt = Tensor::from_vec(pos_vec.clone(), 4usize, &device).unwrap();
        let ph = Tensor::from_vec(pos_vec.clone(), 4usize, &device).unwrap();
        let pw = Tensor::from_vec(pos_vec, 4usize, &device).unwrap();
        let (q2, k2) = rope
            .apply_mrope(&q, &k, &[&pt, &ph, &pw], &sections)
            .unwrap();

        let q1_v = q1.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let q2_v = q2.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        assert_eq!(q1_v.len(), q2_v.len());
        for (i, (a, b)) in q1_v.iter().zip(q2_v.iter()).enumerate() {
            assert!(
                (a - b).abs() == 0.0,
                "q diff at {i}: {a} vs {b} (collapse should be bit-identical)"
            );
        }
        let k1_v = k1.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let k2_v = k2.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        for (i, (a, b)) in k1_v.iter().zip(k2_v.iter()).enumerate() {
            assert!(
                (a - b).abs() == 0.0,
                "k diff at {i}: {a} vs {b} (collapse should be bit-identical)"
            );
        }
    }

    #[test]
    fn rope_preserves_per_head_norm() {
        let device = Device::Cpu;
        let head_dim = 64usize;
        let cfg = RopeConfig {
            head_dim,
            max_seq_len: 128,
            base: 10_000.0,
            kind: RopeKind::Standard,
        };
        let rope = Rope::new(cfg, &device).unwrap();
        let t = 6usize;
        let q = det_tensor((t, 3, head_dim), &device);
        let k = det_tensor((t, 1, head_dim), &device);
        let positions = Tensor::from_vec(vec![0u32, 1, 5, 17, 63, 127], t, &device).unwrap();
        let (q_rot, k_rot) = rope.apply(&q, &k, &positions).unwrap();
        for (orig, rot, heads) in [(&q, &q_rot, 3usize), (&k, &k_rot, 1usize)] {
            let a = orig.flatten_all().unwrap().to_vec1::<f32>().unwrap();
            let b = rot.flatten_all().unwrap().to_vec1::<f32>().unwrap();
            for tok in 0..t {
                for h in 0..heads {
                    let off = (tok * heads + h) * head_dim;
                    let na: f32 = a[off..off + head_dim].iter().map(|v| v * v).sum();
                    let nb: f32 = b[off..off + head_dim].iter().map(|v| v * v).sum();
                    assert!(
                        (na.sqrt() - nb.sqrt()).abs() <= 1e-4 * na.sqrt().max(1e-3),
                        "norm not preserved at token {tok} head {h}: {na} vs {nb}"
                    );
                }
            }
        }
    }

    #[test]
    fn rope_inner_product_depends_only_on_relative_position() {
        let device = Device::Cpu;
        let head_dim = 32usize;
        let cfg = RopeConfig {
            head_dim,
            max_seq_len: 256,
            base: 10_000.0,
            kind: RopeKind::Standard,
        };
        let rope = Rope::new(cfg, &device).unwrap();
        let q = det_tensor((1, 1, head_dim), &device);
        let k = det_tensor((1, 1, head_dim), &device);

        let dot_at = |m: u32, n: u32| -> f32 {
            let pm = Tensor::from_vec(vec![m], 1usize, &device).unwrap();
            let pn = Tensor::from_vec(vec![n], 1usize, &device).unwrap();
            let (qm, _) = rope.apply(&q, &k, &pm).unwrap();
            let (_, kn) = rope.apply(&q, &k, &pn).unwrap();
            let qv = qm.flatten_all().unwrap().to_vec1::<f32>().unwrap();
            let kv = kn.flatten_all().unwrap().to_vec1::<f32>().unwrap();
            qv.iter().zip(kv.iter()).map(|(a, b)| a * b).sum()
        };

        for (m, n, shift) in [(5u32, 2u32, 40u32), (10, 10, 100), (7, 0, 31)] {
            let d0 = dot_at(m, n);
            let d1 = dot_at(m + shift, n + shift);
            assert!(
                (d0 - d1).abs() <= 1e-3 * d0.abs().max(1.0),
                "relative-position identity broken: <R_{m} q, R_{n} k> = {d0} but \
                 shifted by {shift} gives {d1}"
            );
        }

        assert!((dot_at(5, 2) - dot_at(9, 2)).abs() > 1e-3);
    }

    #[test]
    fn rope_matches_neox_rotate_half_reference() {
        let device = Device::Cpu;
        let head_dim = 16usize;
        let cfg = RopeConfig {
            head_dim,
            max_seq_len: 64,
            base: 10_000.0,
            kind: RopeKind::Standard,
        };
        let rope = Rope::new(cfg, &device).unwrap();
        let half = head_dim / 2;
        let q = det_tensor((3, 2, head_dim), &device);
        let k = det_tensor((3, 1, head_dim), &device);
        let pos_host = [0u32, 3, 41];
        let positions = Tensor::from_vec(pos_host.to_vec(), 3usize, &device).unwrap();
        let (q_rot, _) = rope.apply(&q, &k, &positions).unwrap();

        let base: f32 = 10_000.0;
        let qv = q.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let got = q_rot.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        for tok in 0..3 {
            for h in 0..2 {
                let off = (tok * 2 + h) * head_dim;
                for i in 0..half {
                    let theta =
                        pos_host[tok] as f32 / base.powf((i as f32 * 2.0) / head_dim as f32);
                    let (sin, cos) = theta.sin_cos();
                    let lo = qv[off + i];
                    let hi = qv[off + half + i];
                    let want_lo = lo * cos - hi * sin;
                    let want_hi = hi * cos + lo * sin;
                    assert!(
                        (got[off + i] - want_lo).abs() < 1e-4,
                        "tok {tok} head {h} dim {i}: {} vs rotate_half ref {want_lo}",
                        got[off + i]
                    );
                    assert!(
                        (got[off + half + i] - want_hi).abs() < 1e-4,
                        "tok {tok} head {h} dim {}: {} vs rotate_half ref {want_hi}",
                        half + i,
                        got[off + half + i]
                    );
                }
            }
        }
    }

    #[test]
    fn rope_zeroed_inv_freq_tail_is_identity_partial_rotary() {
        let device = Device::Cpu;
        let head_dim = 16usize;
        let half = head_dim / 2;
        let rotary = 2usize;
        let mut inv_freq = vec![0f32; half];
        for i in 0..rotary {
            inv_freq[i] = 1.0 / 10_000f32.powf((i as f32 * 2.0) / head_dim as f32);
        }
        let cfg = RopeConfig {
            head_dim,
            max_seq_len: 64,
            base: 10_000.0,
            kind: RopeKind::Standard,
        };
        let rope = Rope::from_inv_freq(cfg, &inv_freq, &device).unwrap();
        let q = det_tensor((2, 1, head_dim), &device);
        let k = det_tensor((2, 1, head_dim), &device);
        let positions = Tensor::from_vec(vec![13u32, 50], 2usize, &device).unwrap();
        let (q_rot, _) = rope.apply(&q, &k, &positions).unwrap();
        let a = q.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let b = q_rot.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        for tok in 0..2 {
            let off = tok * head_dim;
            for i in rotary..half {
                assert_eq!(a[off + i], b[off + i], "tail lo dim {i} must be identity");
                assert_eq!(
                    a[off + half + i],
                    b[off + half + i],
                    "tail hi dim {} must be identity",
                    half + i
                );
            }
            assert!(
                (a[off] - b[off]).abs() > 1e-6 || (a[off + half] - b[off + half]).abs() > 1e-6,
                "rotary head dims must actually rotate at nonzero positions"
            );
        }
    }

    #[test]
    fn rope_position_zero_is_identity() {
        let device = Device::Cpu;
        let head_dim = 32usize;
        let cfg = RopeConfig {
            head_dim,
            max_seq_len: 8,
            base: 10_000.0,
            kind: RopeKind::Standard,
        };
        let rope = Rope::new(cfg, &device).unwrap();
        let q = det_tensor((1, 2, head_dim), &device);
        let k = det_tensor((1, 1, head_dim), &device);
        let positions = Tensor::from_vec(vec![0u32], 1usize, &device).unwrap();
        let (q_rot, k_rot) = rope.apply(&q, &k, &positions).unwrap();
        let a = q.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let b = q_rot.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        for (x, y) in a.iter().zip(b.iter()) {
            assert!(
                (x - y).abs() < 1e-6,
                "position 0 must be the identity rotation"
            );
        }
        let a = k.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let b = k_rot.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        for (x, y) in a.iter().zip(b.iter()) {
            assert!((x - y).abs() < 1e-6);
        }
    }

    #[test]
    fn mrope_diverges_when_positions_differ() {
        let device = Device::Cpu;
        let head_dim = 128usize;
        let cfg = RopeConfig {
            head_dim,
            max_seq_len: 32,
            base: 1_000_000.0,
            kind: RopeKind::Standard,
        };
        let rope = Rope::new(cfg, &device).unwrap();
        let sections = [24usize, 20, 20];
        let q = det_tensor((2, 2, head_dim), &device);
        let k = det_tensor((2, 1, head_dim), &device);

        let same = Tensor::from_vec(vec![0u32, 1], 2usize, &device).unwrap();
        let (q_same, _k_same) = rope
            .apply_mrope(&q, &k, &[&same, &same, &same], &sections)
            .unwrap();

        let pt = Tensor::from_vec(vec![0u32, 1], 2usize, &device).unwrap();
        let ph = Tensor::from_vec(vec![3u32, 4], 2usize, &device).unwrap();
        let pw = Tensor::from_vec(vec![5u32, 6], 2usize, &device).unwrap();
        let (q_diff, _k_diff) = rope
            .apply_mrope(&q, &k, &[&pt, &ph, &pw], &sections)
            .unwrap();

        let a = q_same.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let b = q_diff.flatten_all().unwrap().to_vec1::<f32>().unwrap();

        let mut max_abs = 0f32;
        for (x, y) in a.iter().zip(b.iter()) {
            max_abs = max_abs.max((x - y).abs());
        }
        assert!(
            max_abs > 1e-4,
            "expected divergence when axes differ, but max|Δ| = {max_abs}"
        );
    }
}

pub const NV_TRAIN_ROPE_FUSED_ENV_DEFAULT_ON_0_RESTORES_THE_CANDLE_NARROW_MUL_CAT_CHAIN: &str =
    "NV_TRAIN_ROPE_FUSED";

pub fn train_rope_op_serves(x: &Tensor) -> bool {
    x.device().is_cuda()
        && matches!(x.dtype(), DType::BF16 | DType::F32)
        && nv_config::env_arm_opt_out_default_on_unless_eq_some_0_reread_every_call(
            NV_TRAIN_ROPE_FUSED_ENV_DEFAULT_ON_0_RESTORES_THE_CANDLE_NARROW_MUL_CAT_CHAIN,
        )
}

#[cfg(feature = "cuda")]
pub struct RopeTrainOp {
    cos: Tensor,
    sin: Tensor,
    positions: Tensor,
    head_dim: usize,
    rotary_dim: usize,
    scale: f32,
}

#[cfg(feature = "cuda")]
impl RopeTrainOp {
    fn table_ptrs(&self, stream: &std::sync::Arc<cudarc::driver::CudaStream>) -> candle_core::Result<(u64, u64, u64)> {
        use cudarc::driver::DevicePtr;
        let f32_ptr = |t: &Tensor| -> candle_core::Result<u64> {
            let (st, l) = t.storage_and_layout();
            let candle_core::Storage::Cuda(c) = &*st else {
                return Err(candle_core::Error::Msg("rope train op: table not cuda".into()));
            };
            let (p, _g) = c.as_cuda_slice::<f32>()?.device_ptr(stream);
            Ok(p + (l.start_offset() * 4) as u64)
        };
        let (pst, pl) = self.positions.storage_and_layout();
        let candle_core::Storage::Cuda(pc) = &*pst else {
            return Err(candle_core::Error::Msg("rope train op: positions not cuda".into()));
        };
        let (pp, _gp) = pc.as_cuda_slice::<u32>()?.device_ptr(stream);
        let pos = pp + (pl.start_offset() * 4) as u64;
        Ok((f32_ptr(&self.cos)?, f32_ptr(&self.sin)?, pos))
    }

    fn launch_ptrs(
        &self,
        stream: &std::sync::Arc<cudarc::driver::CudaStream>,
        x: u64,
        y: u64,
        dtype: DType,
        dims: &[usize],
        bwd: bool,
    ) -> candle_core::Result<()> {
        let heads = dims[dims.len() - 2];
        let tokens = dims.iter().product::<usize>() / (heads * self.head_dim);
        if self.positions.elem_count() != tokens {
            return Err(candle_core::Error::Msg(format!(
                "rope train op: {} positions for {tokens} tokens",
                self.positions.elem_count()
            )));
        }
        let (cp, sp, pp) = self.table_ptrs(stream)?;
        let is_bf16 = match dtype {
            DType::BF16 => 1,
            DType::F32 => 0,
            other => return Err(candle_core::Error::Msg(format!("rope train op: dtype {other:?}"))),
        };
        let rc = unsafe {
            if bwd {
                nv_kernels::cuda::rope_train_bwd(
                    stream.cu_stream() as *mut std::ffi::c_void,
                    x as *const std::ffi::c_void,
                    y as *mut std::ffi::c_void,
                    cp as *const f32,
                    sp as *const f32,
                    pp as *const u32,
                    tokens as i64,
                    heads as i32,
                    self.head_dim as i32,
                    self.rotary_dim as i32,
                    self.scale,
                    is_bf16,
                )
            } else {
                nv_kernels::cuda::rope_train_fwd(
                    stream.cu_stream() as *mut std::ffi::c_void,
                    x as *const std::ffi::c_void,
                    y as *mut std::ffi::c_void,
                    cp as *const f32,
                    sp as *const f32,
                    pp as *const u32,
                    tokens as i64,
                    heads as i32,
                    self.head_dim as i32,
                    self.rotary_dim as i32,
                    self.scale,
                    is_bf16,
                )
            }
        };
        if rc != 0 {
            return Err(candle_core::Error::Msg(format!("rope_train rc={rc} bwd={bwd}")));
        }
        Ok(())
    }
}

#[cfg(feature = "cuda")]
impl candle_core::CustomOp1 for RopeTrainOp {
    fn name(&self) -> &'static str {
        "rope_train"
    }

    fn cpu_fwd(
        &self,
        _s: &candle_core::CpuStorage,
        _l: &candle_core::Layout,
    ) -> candle_core::Result<(candle_core::CpuStorage, candle_core::Shape)> {
        Err(candle_core::Error::Msg("rope_train is CUDA-only".into()))
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
        let n = dims.iter().product::<usize>();
        let Some((start, _end)) = l.contiguous_offsets() else {
            return Err(candle_core::Error::Msg("rope train op: input must be contiguous".into()));
        };
        let out = match s.dtype() {
            DType::BF16 => {
                let x = s.as_cuda_slice::<half::bf16>()?;
                let xv = x.slice(start..);
                let (xp, _gx) = xv.device_ptr(&stream);
                let mut y: cudarc::driver::CudaSlice<half::bf16> = unsafe { stream.alloc::<half::bf16>(n) }
                    .map_err(|e| candle_core::Error::Msg(format!("rope train op alloc: {e:?}")))?;
                {
                    let (yp, _gy) = y.device_ptr_mut(&stream);
                    self.launch_ptrs(&stream, xp, yp, DType::BF16, &dims, false)?;
                }
                candle_core::CudaStorage::wrap_cuda_slice(y, dev)
            }
            DType::F32 => {
                let x = s.as_cuda_slice::<f32>()?;
                let xv = x.slice(start..);
                let (xp, _gx) = xv.device_ptr(&stream);
                let mut y: cudarc::driver::CudaSlice<f32> = unsafe { stream.alloc::<f32>(n) }
                    .map_err(|e| candle_core::Error::Msg(format!("rope train op alloc: {e:?}")))?;
                {
                    let (yp, _gy) = y.device_ptr_mut(&stream);
                    self.launch_ptrs(&stream, xp, yp, DType::F32, &dims, false)?;
                }
                candle_core::CudaStorage::wrap_cuda_slice(y, dev)
            }
            other => return Err(candle_core::Error::Msg(format!("rope train op: dtype {other:?}"))),
        };
        Ok((out, l.shape().clone()))
    }

    fn bwd(&self, arg: &Tensor, _res: &Tensor, grad_res: &Tensor) -> candle_core::Result<Option<Tensor>> {
        use cudarc::driver::DevicePtr;
        let candle_core::Device::Cuda(dev) = arg.device() else {
            return Err(candle_core::Error::Msg("rope train op bwd: not cuda".into()));
        };
        let stream = crate::cuda_stream::current_stream(dev);
        let g = grad_res.detach().to_dtype(arg.dtype())?.contiguous()?;
        let dims = g.dims().to_vec();
        let dx = crate::step_arena::output_tensor_uninitialised_for_a_kernel_to_fill(arg.device(), g.dtype(), &dims)
            .map_err(|e| candle_core::Error::Msg(format!("{e:#}")))?;
        let gp = {
            let (st, l) = g.storage_and_layout();
            let candle_core::Storage::Cuda(c) = &*st else {
                return Err(candle_core::Error::Msg("rope train op bwd: grad not cuda".into()));
            };
            match g.dtype() {
                DType::BF16 => {
                    let (p, _g) = c.as_cuda_slice::<half::bf16>()?.device_ptr(&stream);
                    p + (l.start_offset() * 2) as u64
                }
                DType::F32 => {
                    let (p, _g) = c.as_cuda_slice::<f32>()?.device_ptr(&stream);
                    p + (l.start_offset() * 4) as u64
                }
                other => return Err(candle_core::Error::Msg(format!("rope train op bwd: dtype {other:?}"))),
            }
        };
        let dp = crate::step_arena::device_ptr_for_a_kernel_to_write(&dx)
            .map_err(|e| candle_core::Error::Msg(format!("{e:#}")))?;
        self.launch_ptrs(&stream, gp, dp, g.dtype(), &dims, true)?;
        Ok(Some(dx))
    }
}

impl Rope {
    #[cfg(feature = "cuda")]
    pub fn apply_train_partial_scaled(
        &self,
        x: &Tensor,
        positions: &Tensor,
        rotary_dim: usize,
        head_dim: usize,
        scale: f32,
    ) -> Result<Tensor> {
        anyhow::ensure!(
            rotary_dim == self.cfg.head_dim && rotary_dim <= head_dim,
            "rope train op: rotary_dim {rotary_dim} must be the table's head_dim {} and at most {head_dim}",
            self.cfg.head_dim
        );
        let positions = if positions.dtype() == DType::U32 && positions.device().same_device(x.device()) {
            positions.reshape(positions.elem_count())?.contiguous()?
        } else {
            normalize_positions_u32(positions, x.device())?
        };
        let op = RopeTrainOp {
            cos: self.cos.clone(),
            sin: self.sin.clone(),
            positions,
            head_dim,
            rotary_dim,
            scale,
        };
        Ok(x.contiguous()?.apply_op1(op)?)
    }
}
