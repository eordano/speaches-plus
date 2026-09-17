use anyhow::{ensure, Result};
use candle_core::{Tensor, Var};

pub const NV_Q38_GDN_CHUNK_BWD_IS_THE_SCAN_CHECKPOINT_INTERVAL_IN_TOKENS_DEFAULT_128_USER_SIGNOFF_2026_09_13_INBAND_INSTRUMENT_RULE_ZERO_KEEPS_THE_RETAINED_STATE_GRAPH_AND_IT_ENGAGES_ONLY_INSIDE_THE_CHECKPOINTED_WINDOW_WALK:
    &str = "NV_Q38_GDN_CHUNK_BWD";

pub const THE_BACKWARD_PAYS_ONE_EXTRA_FORWARD_SCAN_AND_RETAINS_CEIL_T_OVER_C_BOUNDARY_STATES_PLUS_ONE_LIVE_CHUNK_GRAPH: &str =
    "exact autograd over the gated-delta scan retains ~three [n_v,d_k,d_v] f32 state tensors per \
     token (9.4 MB/token/layer at 48x128x128), which is what bounds the trainable window; with \
     checkpoints every C tokens the reverse sweep re-forwards each chunk once under autograd, so \
     live scan memory is ceil(T/C) boundary states plus ~3C step tensors, and the compute cost is \
     exactly one extra forward scan regardless of C; C = sqrt(T/3) minimizes the sum, and at \
     T=512 any C in [8,128] stays within ~2.3x of the C*=13 minimum while full retention sits \
     ~19x above it";

pub fn chunk_tokens_from_env() -> usize {
    std::env::var(
        NV_Q38_GDN_CHUNK_BWD_IS_THE_SCAN_CHECKPOINT_INTERVAL_IN_TOKENS_DEFAULT_128_USER_SIGNOFF_2026_09_13_INBAND_INSTRUMENT_RULE_ZERO_KEEPS_THE_RETAINED_STATE_GRAPH_AND_IT_ENGAGES_ONLY_INSIDE_THE_CHECKPOINTED_WINDOW_WALK,
    )
    .ok()
    .and_then(|v| v.parse::<usize>().ok())
    .unwrap_or(GDN_CHUNK_128_TOKENS_TAKES_EIGHT_512_ROWS_FROM_3_47_TO_2_59_S_PER_STEP_AND_29_TO_51_GIB_STAYS_THE_FOUR_ROW_DEFAULTS_ENVELOPE)
}

pub const GDN_CHUNK_128_TOKENS_TAKES_EIGHT_512_ROWS_FROM_3_47_TO_2_59_S_PER_STEP_AND_29_TO_51_GIB_STAYS_THE_FOUR_ROW_DEFAULTS_ENVELOPE: usize = 128;

pub fn fold3(x: &Tensor, bn: usize, t: usize, d: usize) -> Result<Tensor> {
    Ok(x.transpose(1, 2)?.contiguous()?.reshape((bn, t, d))?)
}

pub fn fold2(x: &Tensor, bn: usize, t: usize) -> Result<Tensor> {
    Ok(x.transpose(1, 2)?.contiguous()?.reshape((bn, t))?)
}

pub fn unfold3(x: &Tensor, b: usize, n: usize, t: usize, d: usize) -> Result<Tensor> {
    Ok(x.reshape((b, n, t, d))?.transpose(1, 2)?.contiguous()?)
}

pub fn unfold2(x: &Tensor, b: usize, n: usize, t: usize) -> Result<Tensor> {
    Ok(x.reshape((b, n, t))?.transpose(1, 2)?.contiguous()?)
}

pub fn scan_step(
    state: &Tensor,
    q_t: &Tensor,
    k_t: &Tensor,
    v_t: &Tensor,
    g_t: &Tensor,
    be_t: &Tensor,
) -> Result<(Tensor, Tensor)> {
    let decayed = state.broadcast_mul(g_t)?;
    let kv_mem = k_t.matmul(&decayed)?;
    let delta = v_t.sub(&kv_mem)?.broadcast_mul(be_t)?;
    let update = k_t.transpose(1, 2)?.contiguous()?.matmul(&delta)?;
    let state = decayed.add(&update)?;
    let out = q_t.matmul(&state)?;
    Ok((state, out))
}

pub struct ChunkedScanFwd {
    pub out: Tensor,
    pub checkpoints: Vec<Tensor>,
    pub checkpoints_dense_from_the_fused_kernel: Option<Tensor>,
}

pub const NV_Q38_GDN_SCAN_KERNEL_DEFAULT_ON_0_RESTORES_THE_PER_TOKEN_CANDLE_SCAN_WHOSE_TEN_LAUNCHES_PER_TOKEN_MADE_THE_WINDOW_BACKWARD_2_1_S_OF_A_3_0_S_STEP: &str =
    "NV_Q38_GDN_SCAN_KERNEL";

#[cfg(feature = "cuda")]
pub fn fused_scan_kernel_serves(q: &Tensor, d_k: usize, d_v: usize) -> bool {
    std::env::var(NV_Q38_GDN_SCAN_KERNEL_DEFAULT_ON_0_RESTORES_THE_PER_TOKEN_CANDLE_SCAN_WHOSE_TEN_LAUNCHES_PER_TOKEN_MADE_THE_WINDOW_BACKWARD_2_1_S_OF_A_3_0_S_STEP)
        .as_deref()
        != Ok("0")
        && d_k == 128
        && d_v == 128
        && q.dtype() == candle_core::DType::F32
        && matches!(q.device(), candle_core::Device::Cuda(_))
}

#[cfg(not(feature = "cuda"))]
pub fn fused_scan_kernel_serves(_q: &Tensor, _d_k: usize, _d_v: usize) -> bool {
    false
}

#[cfg(feature = "cuda")]
mod fused {
    use anyhow::{ensure, Result};
    use candle_core::{DType, Device, Tensor};
    use cudarc::driver::DevicePtr;

    pub struct Dev {
        pub dev: candle_core::CudaDevice,
        pub stream: std::sync::Arc<cudarc::driver::CudaStream>,
    }

    pub fn dev_of(t: &Tensor) -> Result<Dev> {
        let Device::Cuda(dev) = t.device() else {
            anyhow::bail!("fused gdn scan needs a cuda tensor")
        };
        let stream = nv_layers::cuda_stream::current_stream(dev);
        Ok(Dev {
            dev: dev.clone(),
            stream,
        })
    }

    pub fn f32_ptr(t: &Tensor, d: &Dev) -> Result<u64> {
        ensure!(t.dtype() == DType::F32, "fused gdn scan operand must be f32");
        let (st, layout) = t.storage_and_layout();
        let candle_core::Storage::Cuda(c) = &*st else {
            anyhow::bail!("fused gdn scan operand not cuda")
        };
        let (p0, _p1) = layout
            .contiguous_offsets()
            .ok_or_else(|| anyhow::anyhow!("fused gdn scan operand not dense"))?;
        let (p, _g) = c.as_cuda_slice::<f32>()?.device_ptr(&d.stream);
        Ok(p + (p0 * 4) as u64)
    }

    pub fn out_f32(d: &Dev, shape: &[usize]) -> Result<Tensor> {
        nv_layers::step_arena::output_tensor_uninitialised_for_a_kernel_to_fill(
            &Device::Cuda(d.dev.clone()),
            DType::F32,
            shape,
        )
    }

    pub fn out_ptr(t: &Tensor) -> Result<u64> {
        nv_layers::step_arena::device_ptr_for_a_kernel_to_write(t)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn scan_fwd(
        q3: &Tensor,
        k3: &Tensor,
        v3: &Tensor,
        g2: &Tensor,
        be2: &Tensor,
        bn: usize,
        t: usize,
        chunk: usize,
    ) -> Result<(Tensor, Tensor)> {
        let d = dev_of(q3)?;
        let n_chunks = t.div_ceil(chunk);
        let out = out_f32(&d, &[bn, t, 128])?;
        let ckpt = out_f32(&d, &[n_chunks, bn, 128, 128])?;
        let (q3, k3, v3, g2, be2) = (
            q3.contiguous()?,
            k3.contiguous()?,
            v3.contiguous()?,
            g2.contiguous()?,
            be2.contiguous()?,
        );
        let rc = unsafe {
            nv_kernels::cuda::gdn_train_scan_fwd_f32(
                d.stream.cu_stream() as *mut _,
                f32_ptr(&q3, &d)? as *const f32,
                f32_ptr(&k3, &d)? as *const f32,
                f32_ptr(&v3, &d)? as *const f32,
                f32_ptr(&g2, &d)? as *const f32,
                f32_ptr(&be2, &d)? as *const f32,
                out_ptr(&out)? as *mut f32,
                out_ptr(&ckpt)? as *mut f32,
                bn as i32,
                t as i32,
                128,
                128,
                chunk as i32,
            )
        };
        ensure!(rc == 0, "gdn_train_scan_fwd_f32 returned {rc}");
        Ok((out, ckpt))
    }

    #[allow(clippy::too_many_arguments)]
    pub fn scan_bwd(
        q3: &Tensor,
        k3: &Tensor,
        v3: &Tensor,
        g2: &Tensor,
        be2: &Tensor,
        do3: &Tensor,
        ckpt: &Tensor,
        bn: usize,
        t: usize,
        chunk: usize,
    ) -> Result<(Tensor, Tensor, Tensor, Tensor, Tensor)> {
        let d = dev_of(q3)?;
        let scratch = out_f32(&d, &[bn, chunk, 128, 128])?;
        let dq = out_f32(&d, &[bn, t, 128])?;
        let dk = out_f32(&d, &[bn, t, 128])?;
        let dv = out_f32(&d, &[bn, t, 128])?;
        let dg = out_f32(&d, &[bn, t])?;
        let dbe = out_f32(&d, &[bn, t])?;
        let (q3, k3, v3, g2, be2, do3, ckpt) = (
            q3.contiguous()?,
            k3.contiguous()?,
            v3.contiguous()?,
            g2.contiguous()?,
            be2.contiguous()?,
            do3.contiguous()?,
            ckpt.contiguous()?,
        );
        let rc = unsafe {
            nv_kernels::cuda::gdn_train_scan_bwd_f32(
                d.stream.cu_stream() as *mut _,
                f32_ptr(&q3, &d)? as *const f32,
                f32_ptr(&k3, &d)? as *const f32,
                f32_ptr(&v3, &d)? as *const f32,
                f32_ptr(&g2, &d)? as *const f32,
                f32_ptr(&be2, &d)? as *const f32,
                f32_ptr(&do3, &d)? as *const f32,
                f32_ptr(&ckpt, &d)? as *const f32,
                out_ptr(&scratch)? as *mut f32,
                out_ptr(&dq)? as *mut f32,
                out_ptr(&dk)? as *mut f32,
                out_ptr(&dv)? as *mut f32,
                out_ptr(&dg)? as *mut f32,
                out_ptr(&dbe)? as *mut f32,
                bn as i32,
                t as i32,
                128,
                128,
                chunk as i32,
            )
        };
        ensure!(rc == 0, "gdn_train_scan_bwd_f32 returned {rc}");
        Ok((dq, dk, dv, dg, dbe))
    }
}

#[allow(clippy::too_many_arguments)]
pub fn scan_chunked_fwd(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    g_exp: &Tensor,
    beta: &Tensor,
    b: usize,
    t: usize,
    n_v: usize,
    d_k: usize,
    d_v: usize,
    chunk: usize,
) -> Result<ChunkedScanFwd> {
    ensure!(chunk >= 1, "chunk interval must be at least 1, got {chunk}");
    let bn = b * n_v;
    let q3 = fold3(&q.detach(), bn, t, d_k)?;
    let k3 = fold3(&k.detach(), bn, t, d_k)?;
    let v3 = fold3(&v.detach(), bn, t, d_v)?;
    let g2 = fold2(&g_exp.detach(), bn, t)?;
    let be2 = fold2(&beta.detach(), bn, t)?;

    #[cfg(feature = "cuda")]
    if fused_scan_kernel_serves(q, d_k, d_v) {
        let (out, ckpt) = fused::scan_fwd(&q3, &k3, &v3, &g2, &be2, bn, t, chunk)?;
        let n_chunks = t.div_ceil(chunk);
        let checkpoints = (0..n_chunks)
            .map(|cj| ckpt.narrow(0, cj, 1)?.squeeze(0))
            .collect::<candle_core::Result<Vec<_>>>()?;
        return Ok(ChunkedScanFwd {
            out: unfold3(&out, b, n_v, t, d_v)?,
            checkpoints,
            checkpoints_dense_from_the_fused_kernel: Some(ckpt),
        });
    }

    let mut state = Tensor::zeros((bn, d_k, d_v), q.dtype(), q.device())?;
    let mut checkpoints: Vec<Tensor> = Vec::with_capacity(t.div_ceil(chunk));
    let mut outs: Vec<Tensor> = Vec::with_capacity(t);
    for ti in 0..t {
        if ti % chunk == 0 {
            checkpoints.push(state.clone());
        }
        let q_t = q3.narrow(1, ti, 1)?.contiguous()?;
        let k_t = k3.narrow(1, ti, 1)?.contiguous()?;
        let v_t = v3.narrow(1, ti, 1)?.contiguous()?;
        let g_t = g2.narrow(1, ti, 1)?.reshape((bn, 1, 1))?;
        let be_t = be2.narrow(1, ti, 1)?.reshape((bn, 1, 1))?;
        let (s, o) = scan_step(&state, &q_t, &k_t, &v_t, &g_t, &be_t)?;
        state = s;
        outs.push(o);
    }
    let out = Tensor::cat(&outs, 1)?;
    Ok(ChunkedScanFwd {
        out: unfold3(&out, b, n_v, t, d_v)?,
        checkpoints,
        checkpoints_dense_from_the_fused_kernel: None,
    })
}

#[allow(clippy::too_many_arguments)]
pub fn scan_chunked_input_grads(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    g_exp: &Tensor,
    beta: &Tensor,
    d_out: &Tensor,
    checkpoints: &[Tensor],
    b: usize,
    t: usize,
    n_v: usize,
    d_k: usize,
    d_v: usize,
    chunk: usize,
) -> Result<(Tensor, Tensor, Tensor, Tensor, Tensor)> {
    ensure!(chunk >= 1, "chunk interval must be at least 1, got {chunk}");
    let bn = b * n_v;
    let n_chunks = t.div_ceil(chunk);
    ensure!(
        checkpoints.len() == n_chunks,
        "checkpoint count {} != ceil({t}/{chunk}) = {n_chunks}; the reverse sweep needs the \
         boundary state entering every chunk",
        checkpoints.len()
    );
    let q3 = fold3(&q.detach(), bn, t, d_k)?;
    let k3 = fold3(&k.detach(), bn, t, d_k)?;
    let v3 = fold3(&v.detach(), bn, t, d_v)?;
    let g2 = fold2(&g_exp.detach(), bn, t)?;
    let be2 = fold2(&beta.detach(), bn, t)?;
    let do3 = fold3(&d_out.detach(), bn, t, d_v)?;

    #[cfg(feature = "cuda")]
    if fused_scan_kernel_serves(q, d_k, d_v) {
        let ckpt = Tensor::stack(checkpoints, 0)?;
        let (dq, dk, dv, dg, dbe) =
            fused::scan_bwd(&q3, &k3, &v3, &g2, &be2, &do3, &ckpt, bn, t, chunk)?;
        if std::env::var("NV_Q38_GDN_SCAN_CHECK_PRINTS_THE_KERNEL_VS_CANDLE_DIVERGENCE_PER_CALL").as_deref() == Ok("1") {
            std::env::set_var(NV_Q38_GDN_SCAN_KERNEL_DEFAULT_ON_0_RESTORES_THE_PER_TOKEN_CANDLE_SCAN_WHOSE_TEN_LAUNCHES_PER_TOKEN_MADE_THE_WINDOW_BACKWARD_2_1_S_OF_A_3_0_S_STEP, "0");
            let r = scan_chunked_input_grads(q, k, v, g_exp, beta, d_out, checkpoints, b, t, n_v, d_k, d_v, chunk);
            std::env::remove_var(NV_Q38_GDN_SCAN_KERNEL_DEFAULT_ON_0_RESTORES_THE_PER_TOKEN_CANDLE_SCAN_WHOSE_TEN_LAUNCHES_PER_TOKEN_MADE_THE_WINDOW_BACKWARD_2_1_S_OF_A_3_0_S_STEP);
            let r = r?;
            let rel = |a: &Tensor, w: &Tensor| -> Result<(f32, f32)> {
                let d = a.sub(w)?.abs()?.flatten_all()?.max(0)?.to_scalar::<f32>()?;
                let m = w.abs()?.flatten_all()?.max(0)?.to_scalar::<f32>()?;
                Ok((d, m))
            };
            let (a, b0) = rel(&unfold3(&dq, b, n_v, t, d_k)?, &r.0)?;
            let (c, d0) = rel(&unfold3(&dk, b, n_v, t, d_k)?, &r.1)?;
            let (e, f0) = rel(&unfold3(&dv, b, n_v, t, d_v)?, &r.2)?;
            let (g, h0) = rel(&unfold2(&dg, b, n_v, t)?, &r.3)?;
            let (i, j0) = rel(&unfold2(&dbe, b, n_v, t)?, &r.4)?;
            eprintln!(
                "[gdn-scan-bwd-check] b={b} t={t} n_v={n_v} chunk={chunk} dq={a:e}/{b0:e} dk={c:e}/{d0:e} dv={e:e}/{f0:e} dg={g:e}/{h0:e} dbeta={i:e}/{j0:e}"
            );
        }
        return Ok((
            unfold3(&dq, b, n_v, t, d_k)?,
            unfold3(&dk, b, n_v, t, d_k)?,
            unfold3(&dv, b, n_v, t, d_v)?,
            unfold2(&dg, b, n_v, t)?,
            unfold2(&dbe, b, n_v, t)?,
        ));
    }

    let mut dq_c: Vec<Option<Tensor>> = vec![None; n_chunks];
    let mut dk_c: Vec<Option<Tensor>> = vec![None; n_chunks];
    let mut dv_c: Vec<Option<Tensor>> = vec![None; n_chunks];
    let mut dg_c: Vec<Option<Tensor>> = vec![None; n_chunks];
    let mut db_c: Vec<Option<Tensor>> = vec![None; n_chunks];
    let mut dstate: Option<Tensor> = None;

    for cj in (0..n_chunks).rev() {
        let t0 = cj * chunk;
        let clen = chunk.min(t - t0);
        let qv = Var::from_tensor(&q3.narrow(1, t0, clen)?.contiguous()?)?;
        let kv = Var::from_tensor(&k3.narrow(1, t0, clen)?.contiguous()?)?;
        let vv = Var::from_tensor(&v3.narrow(1, t0, clen)?.contiguous()?)?;
        let gv = Var::from_tensor(&g2.narrow(1, t0, clen)?.contiguous()?)?;
        let bev = Var::from_tensor(&be2.narrow(1, t0, clen)?.contiguous()?)?;
        let sv = Var::from_tensor(&checkpoints[cj])?;

        let mut state = sv.as_tensor().clone();
        let mut outs: Vec<Tensor> = Vec::with_capacity(clen);
        for ti in 0..clen {
            let q_t = qv.as_tensor().narrow(1, ti, 1)?.contiguous()?;
            let k_t = kv.as_tensor().narrow(1, ti, 1)?.contiguous()?;
            let v_t = vv.as_tensor().narrow(1, ti, 1)?.contiguous()?;
            let g_t = gv.as_tensor().narrow(1, ti, 1)?.reshape((bn, 1, 1))?;
            let be_t = bev.as_tensor().narrow(1, ti, 1)?.reshape((bn, 1, 1))?;
            let (s, o) = scan_step(&state, &q_t, &k_t, &v_t, &g_t, &be_t)?;
            state = s;
            outs.push(o);
        }
        let chunk_out = Tensor::cat(&outs, 1)?;
        let mut surrogate = chunk_out
            .mul(&do3.narrow(1, t0, clen)?.contiguous()?)?
            .sum_all()?;
        if let Some(ds) = &dstate {
            surrogate = surrogate.add(&state.mul(ds)?.sum_all()?)?;
        }
        let gs = surrogate.backward()?;
        let take = |var: &Var, name: &str| -> Result<Tensor> {
            Ok(gs
                .get(var.as_tensor())
                .ok_or_else(|| {
                    anyhow::anyhow!(
                        "chunk {cj}: no grad for {name}; every scan input is consumed by every \
                         step, so a missing grad means the chunk graph was severed"
                    )
                })?
                .detach())
        };
        dq_c[cj] = Some(take(&qv, "q")?);
        dk_c[cj] = Some(take(&kv, "k")?);
        dv_c[cj] = Some(take(&vv, "v")?);
        dg_c[cj] = Some(take(&gv, "g_exp")?);
        db_c[cj] = Some(take(&bev, "beta")?);
        dstate = Some(take(&sv, "chunk input state")?);
    }

    let cat = |cs: Vec<Option<Tensor>>| -> Result<Tensor> {
        let v: Vec<Tensor> = cs
            .into_iter()
            .map(|c| c.expect("the reverse sweep fills every chunk slot"))
            .collect();
        Ok(Tensor::cat(&v, 1)?)
    };
    Ok((
        unfold3(&cat(dq_c)?, b, n_v, t, d_k)?,
        unfold3(&cat(dk_c)?, b, n_v, t, d_k)?,
        unfold3(&cat(dv_c)?, b, n_v, t, d_v)?,
        unfold2(&cat(dg_c)?, b, n_v, t)?,
        unfold2(&cat(db_c)?, b, n_v, t)?,
    ))
}
