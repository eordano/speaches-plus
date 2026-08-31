use anyhow::Result;
use candle_core::{Device, Tensor};
use cudarc::driver::{CudaSlice, DevicePtr, DevicePtrMut};
use half::bf16;

pub fn rowquant_i8_dev(w: &Tensor) -> Result<(CudaSlice<i8>, CudaSlice<f32>)> {
    rowquant_dev::<i8>(w, "rowquant_i8", |stream, wp, qp, sp, n, k| unsafe {
        nv_kernels::cuda::rowquant_i8(stream, wp, qp as *mut i8, sp, n, k)
    })
}

pub fn rowquant_e4m3_dev(w: &Tensor) -> Result<(CudaSlice<u8>, CudaSlice<f32>)> {
    rowquant_dev::<u8>(w, "rowquant_e4m3", |stream, wp, qp, sp, n, k| unsafe {
        nv_kernels::cuda::rowquant_e4m3(stream, wp, qp as *mut u8, sp, n, k)
    })
}

fn rowquant_dev<Q: cudarc::driver::DeviceRepr>(
    w: &Tensor,
    kernel: &str,
    launch: impl FnOnce(*mut std::ffi::c_void, *const u16, u64, *mut f32, i32, i32) -> i32,
) -> Result<(CudaSlice<Q>, CudaSlice<f32>)> {
    let dev = match w.device() {
        Device::Cuda(d) => d.clone(),
        _ => anyhow::bail!("{kernel} requires cuda"),
    };
    let (n, k) = w.dims2()?;
    let stream = crate::cuda_stream::current_stream(&dev);
    let mut wq: CudaSlice<Q> = unsafe { stream.alloc::<Q>(n * k).map_err(|e| anyhow::anyhow!(e))? };
    let mut rs: CudaSlice<f32> = unsafe { stream.alloc::<f32>(n).map_err(|e| anyhow::anyhow!(e))? };
    let rc = {
        let (ws, wl) = w.storage_and_layout();
        let wc = match &*ws {
            candle_core::Storage::Cuda(s) => s,
            _ => anyhow::bail!("{kernel}: weight not cuda"),
        };
        let wsl = wc.as_cuda_slice::<bf16>()?;
        let wview = wsl.slice(wl.start_offset()..);
        let (wp, _gw) = wview.device_ptr(&stream);
        let (qp, _gq) = wq.device_ptr_mut(&stream);
        let (sp, _gs) = rs.device_ptr_mut(&stream);
        launch(
            stream.cu_stream() as *mut _,
            wp as *const u16,
            qp as u64,
            sp as *mut f32,
            n as i32,
            k as i32,
        )
    };
    anyhow::ensure!(rc == 0, "{kernel} returned {rc}");
    stream.synchronize().map_err(|e| anyhow::anyhow!(e))?;
    Ok((wq, rs))
}

pub enum ResidentQuantWq<'a> {
    I8 {
        wq: &'a CudaSlice<i8>,
        force_h: bool,
    },
    E4m3 {
        wq: &'a CudaSlice<u8>,
    },
}

impl ResidentQuantWq<'_> {
    fn len(&self) -> usize {
        match self {
            Self::I8 { wq, .. } => wq.len(),
            Self::E4m3 { wq } => wq.len(),
        }
    }
}

pub fn lm_head_i8_normed_mk_op(
    wq: &CudaSlice<i8>,
    row_scale: &CudaSlice<f32>,
    x: &Tensor,
    w_norm: &Tensor,
    rstd: &Tensor,
    m: usize,
    dev: &candle_core::CudaDevice,
) -> Result<Tensor> {
    rowquant_normed_mk_gemv(
        ResidentQuantWq::I8 { wq, force_h: false },
        row_scale,
        x,
        w_norm,
        rstd,
        m,
        dev,
    )
}

pub fn i8_mk_h_op(
    wq: &CudaSlice<i8>,
    row_scale: &CudaSlice<f32>,
    x: &Tensor,
    w_norm: &Tensor,
    rstd: &Tensor,
    m: usize,
    dev: &candle_core::CudaDevice,
) -> Result<Tensor> {
    rowquant_normed_mk_gemv(
        ResidentQuantWq::I8 { wq, force_h: true },
        row_scale,
        x,
        w_norm,
        rstd,
        m,
        dev,
    )
}

pub fn e4m3_mk_h_op(
    wq: &CudaSlice<u8>,
    row_scale: &CudaSlice<f32>,
    x: &Tensor,
    w_norm: &Tensor,
    rstd: &Tensor,
    m: usize,
    dev: &candle_core::CudaDevice,
) -> Result<Tensor> {
    rowquant_normed_mk_gemv(
        ResidentQuantWq::E4m3 { wq },
        row_scale,
        x,
        w_norm,
        rstd,
        m,
        dev,
    )
}

pub fn lm_head_i8_normed_op(
    wq: &CudaSlice<i8>,
    row_scale: &CudaSlice<f32>,
    x: &Tensor,
    w_norm: &Tensor,
    rstd: &Tensor,
    dev: &candle_core::CudaDevice,
) -> Result<Tensor> {
    let hidden = x.elem_count();
    let vocab = wq.len() / hidden;
    let x_c = x.contiguous()?;
    let stream = crate::cuda_stream::current_stream(dev);
    let mut y_dev: CudaSlice<bf16> = unsafe {
        stream
            .alloc::<bf16>(vocab)
            .map_err(|e| anyhow::anyhow!(e))?
    };
    let rc = {
        let (xs, xl) = x_c.storage_and_layout();
        let (x0, x1) = xl
            .contiguous_offsets()
            .ok_or_else(|| anyhow::anyhow!("lm_head_i8: x not dense"))?;
        let (ws, _wl) = w_norm.storage_and_layout();
        let (rs2, _rl) = rstd.storage_and_layout();
        let xc = match &*xs {
            candle_core::Storage::Cuda(s) => s,
            _ => anyhow::bail!("x not cuda"),
        };
        let wc = match &*ws {
            candle_core::Storage::Cuda(s) => s,
            _ => anyhow::bail!("w_norm not cuda"),
        };
        let rcu = match &*rs2 {
            candle_core::Storage::Cuda(s) => s,
            _ => anyhow::bail!("rstd not cuda"),
        };
        let xsl = xc.as_cuda_slice::<bf16>()?;
        let xview = xsl.slice(x0..x1);
        let wsl = wc.as_cuda_slice::<bf16>()?;
        let rsl = rcu.as_cuda_slice::<f32>()?;
        let (qp, _gq) = wq.device_ptr(&stream);
        let (scp, _gsc) = row_scale.device_ptr(&stream);
        let (xp, _gx) = xview.device_ptr(&stream);
        let (wp, _gw) = wsl.device_ptr(&stream);
        let (rp, _gr) = rsl.device_ptr(&stream);
        let (yp, _gy) = y_dev.device_ptr_mut(&stream);
        unsafe {
            nv_kernels::cuda::gemv_i8_normed(
                stream.cu_stream() as *mut _,
                qp as *const i8,
                scp as *const f32,
                xp as *const u16,
                wp as *const u16,
                rp as *const f32,
                yp as *mut u16,
                vocab as i32,
                hidden as i32,
            )
        }
    };
    anyhow::ensure!(rc == 0, "gemv_i8_normed returned {rc}");
    let storage = candle_core::CudaStorage::wrap_cuda_slice(y_dev, dev.clone());
    let storage = candle_core::Storage::Cuda(storage);
    Ok(Tensor::from_storage(
        storage,
        (1usize, 1usize, vocab),
        candle_core::op::BackpropOp::none(),
        false,
    ))
}

const PRENORM_SPLIT_WINS_ONLY_UP_TO_M: usize = 8;
const PRENORM_SPLIT_WINS_ONLY_FROM_HIDDEN: usize = 4096;

fn rowquant_normed_mk_gemv(
    wq: ResidentQuantWq,
    row_scale: &CudaSlice<f32>,
    x: &Tensor,
    w_norm: &Tensor,
    rstd: &Tensor,
    m: usize,
    dev: &candle_core::CudaDevice,
) -> Result<Tensor> {
    let hidden = x.elem_count() / m;
    let n_rows = wq.len() / hidden;
    anyhow::ensure!(rstd.elem_count() == m, "lm_head_mk: rstd rows mismatch");
    let x_c = x.reshape((m, hidden))?.contiguous()?;
    let stream = crate::cuda_stream::current_stream(dev);
    let mut y_dev: CudaSlice<bf16> = unsafe {
        stream
            .alloc::<bf16>(m * n_rows)
            .map_err(|e| anyhow::anyhow!(e))?
    };
    let (rc, launched) = {
        let (xs, xl) = x_c.storage_and_layout();
        let (x0, x1) = xl
            .contiguous_offsets()
            .ok_or_else(|| anyhow::anyhow!("lm_head_mk: x not dense"))?;
        let (ws, _wl) = w_norm.storage_and_layout();
        let (rs2, _rl) = rstd.storage_and_layout();
        let xc = match &*xs {
            candle_core::Storage::Cuda(s) => s,
            _ => anyhow::bail!("x not cuda"),
        };
        let wc = match &*ws {
            candle_core::Storage::Cuda(s) => s,
            _ => anyhow::bail!("w_norm not cuda"),
        };
        let rcu = match &*rs2 {
            candle_core::Storage::Cuda(s) => s,
            _ => anyhow::bail!("rstd not cuda"),
        };
        let xsl = xc.as_cuda_slice::<bf16>()?;
        let xview = xsl.slice(x0..x1);
        let wsl = wc.as_cuda_slice::<bf16>()?;
        let rsl = rcu.as_cuda_slice::<f32>()?;
        let (scp, _gsc) = row_scale.device_ptr(&stream);
        let (xp, _gx) = xview.device_ptr(&stream);
        let (wp, _gw) = wsl.device_ptr(&stream);
        let (rp, _gr) = rsl.device_ptr(&stream);
        let (yp, _gy) = y_dev.device_ptr_mut(&stream);
        match wq {
            ResidentQuantWq::I8 { wq, force_h } => {
                let (qp, _gq) = wq.device_ptr(&stream);
                let prenorm = !force_h
                    && m <= PRENORM_SPLIT_WINS_ONLY_UP_TO_M
                    && hidden >= PRENORM_SPLIT_WINS_ONLY_FROM_HIDDEN
                    && std::env::var("NV_LMI8_PRENORM").ok().as_deref() != Some("0");
                let rc = unsafe {
                    if prenorm {
                        let mut xn_dev: CudaSlice<bf16> = stream
                            .alloc::<bf16>(m * hidden)
                            .map_err(|e| anyhow::anyhow!(e))?;
                        let (xnp, _gxn) = xn_dev.device_ptr_mut(&stream);
                        let rc1 = nv_kernels::cuda::normx_mk(
                            stream.cu_stream() as *mut _,
                            xp as *const u16,
                            wp as *const u16,
                            rp as *const f32,
                            xnp as *mut u16,
                            hidden as i32,
                            m as i32,
                        );
                        if rc1 != 0 {
                            rc1
                        } else {
                            nv_kernels::cuda::gemv_i8_prenormed_mk(
                                stream.cu_stream() as *mut _,
                                qp as *const i8,
                                scp as *const f32,
                                xnp as *const u16,
                                yp as *mut u16,
                                n_rows as i32,
                                hidden as i32,
                                m as i32,
                            )
                        }
                    } else if force_h {
                        nv_kernels::cuda::gemv_i8_mk_h(
                            stream.cu_stream() as *mut _,
                            qp as *const i8,
                            scp as *const f32,
                            xp as *const u16,
                            wp as *const u16,
                            rp as *const f32,
                            yp as *mut u16,
                            n_rows as i32,
                            hidden as i32,
                            m as i32,
                        )
                    } else {
                        nv_kernels::cuda::gemv_i8_normed_mk(
                            stream.cu_stream() as *mut _,
                            qp as *const i8,
                            scp as *const f32,
                            xp as *const u16,
                            wp as *const u16,
                            rp as *const f32,
                            yp as *mut u16,
                            n_rows as i32,
                            hidden as i32,
                            m as i32,
                        )
                    }
                };
                (rc, format!("gemv_i8_normed_mk(force_h={force_h})"))
            }
            ResidentQuantWq::E4m3 { wq } => {
                let (qp, _gq) = wq.device_ptr(&stream);
                let rc = unsafe {
                    nv_kernels::cuda::gemv_e4m3_mk_h(
                        stream.cu_stream() as *mut _,
                        qp as *const u8,
                        scp as *const f32,
                        xp as *const u16,
                        wp as *const u16,
                        rp as *const f32,
                        yp as *mut u16,
                        n_rows as i32,
                        hidden as i32,
                        m as i32,
                    )
                };
                (rc, "gemv_e4m3_mk_h".to_string())
            }
        }
    };
    anyhow::ensure!(rc == 0, "{launched} returned {rc}");
    let storage = candle_core::CudaStorage::wrap_cuda_slice(y_dev, dev.clone());
    let storage = candle_core::Storage::Cuda(storage);
    Ok(Tensor::from_storage(
        storage,
        (m, n_rows),
        candle_core::op::BackpropOp::none(),
        false,
    ))
}
