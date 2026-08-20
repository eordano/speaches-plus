#![cfg(feature = "cuda")]

use anyhow::{Context, Result};
use candle_core::{CudaDevice, DType, Tensor};
use cudarc::driver::{CudaSlice, CudaStream, DevicePtr, DevicePtrMut};
use half::bf16;
use std::ffi::c_void;
use std::sync::Arc;

use crate::laguna::LagunaMoe;
use crate::laguna_step_graph::{ProfPoint, StepProfStamps};
use nv_kernels::graph::CudaGraphRunner;
use nv_layers::moe_grouped::{
    forward_grouped_decode_into, GroupedDecodeContext, MoeGroupedWeights,
};
use nv_layers::norm::RmsNorm;

fn tail_fuse_enabled() -> bool {
    std::env::var("NV_MOE_TAIL_FUSE")
        .map(|v| v != "0" && !v.eq_ignore_ascii_case("false"))
        .unwrap_or(true)
}

pub fn graph_enabled() -> bool {
    std::env::var("NV_LAGUNA_GRAPH")
        .map(|v| {
            v == "1" || v == "2" || v.eq_ignore_ascii_case("true") || v.eq_ignore_ascii_case("full")
        })
        .unwrap_or(false)
}

pub fn whole_step_graph_enabled() -> bool {
    std::env::var("NV_LAGUNA_GRAPH")
        .map(|v| v == "2" || v.eq_ignore_ascii_case("full"))
        .unwrap_or(false)
}

pub struct LagunaMoeGraphs {
    forked: Arc<CudaStream>,
    aux_gemm: Arc<CudaStream>,
    aux_shared: Arc<CudaStream>,
    runners: Vec<Option<CudaGraphRunner>>,
    captures: u64,
    replays: u64,
    failed: bool,
}

unsafe impl Send for LagunaMoeGraphs {}

impl LagunaMoeGraphs {
    pub fn new(dev: &CudaDevice, num_layers: usize) -> Result<Self> {
        let raw_ctx = dev.cuda_stream().context().clone();
        crate::gemma4_batch_graph::graph_teardown::disable_event_tracking_before_capture(&raw_ctx);
        let mut ctor_guard = crate::gemma4_batch_graph::graph_teardown::CtorForkGuard::new();
        let forked = ctor_guard
            .fork(&raw_ctx)
            .map_err(|e| anyhow::anyhow!("forked stream: {e:?}"))?;
        let aux_gemm = ctor_guard
            .fork(&raw_ctx)
            .map_err(|e| anyhow::anyhow!("aux gemm stream: {e:?}"))?;
        let aux_shared = ctor_guard
            .fork(&raw_ctx)
            .map_err(|e| anyhow::anyhow!("aux shared stream: {e:?}"))?;
        forked
            .synchronize()
            .map_err(|e| anyhow::anyhow!("forked sync: {e:?}"))?;
        let runners = (0..num_layers).map(|_| None).collect();
        ctor_guard.the_built_engine_owns_teardown_now();
        Ok(Self {
            forked,
            aux_gemm,
            aux_shared,
            runners,
            captures: 0,
            replays: 0,
            failed: false,
        })
    }

    pub fn failed(&self) -> bool {
        self.failed
    }

    pub fn mark_failed(&mut self) {
        self.failed = true;
        for r in self.runners.iter_mut().flatten() {
            r.invalidate();
        }
        let _ = self.forked.synchronize();
        let ordinal = self.forked.context().ordinal() as i32;
        if let Ok(devh) = cudarc::driver::result::device::get(ordinal) {
            let _ = unsafe { cudarc::driver::sys::cuDeviceGraphMemTrim(devh) };
        }
    }

    pub fn layers_cached(&self) -> usize {
        self.runners
            .iter()
            .filter(|r| r.as_ref().map(|r| r.has_cached()).unwrap_or(false))
            .count()
    }

    pub fn captures(&self) -> u64 {
        self.captures
    }

    pub fn replays(&self) -> u64 {
        self.replays
    }

    pub fn synchronize(&self) -> Result<()> {
        self.forked
            .synchronize()
            .map_err(|e| anyhow::anyhow!("forked sync: {e:?}"))
    }

    pub fn forward_layer(
        &mut self,
        layer: usize,
        moe: &LagunaMoe,
        norm: &RmsNorm,
        w: &MoeGroupedWeights,
        ctx: &mut GroupedDecodeContext,
        resid: &Tensor,
        dev: &CudaDevice,
    ) -> Result<Tensor> {
        let hidden = w.hidden_size;
        let e = w.num_experts;
        anyhow::ensure!(layer < self.runners.len(), "layer {layer} out of range");
        anyhow::ensure!(
            resid.dims() == [1, hidden] && resid.dtype() == DType::BF16,
            "graph moe layer {layer}: expected [1, {hidden}] BF16 residual, got {:?} {:?}",
            resid.dims(),
            resid.dtype()
        );
        anyhow::ensure!(
            ctx.n_tokens() == 1,
            "laguna graph forward_layer stages exactly one token's residual; a ctx built for \
             {} tokens would replay the captured body over uninitialized rows",
            ctx.n_tokens()
        );
        anyhow::ensure!(
            ctx.hidden() == hidden && ctx.num_experts() == e,
            "graph moe layer {layer}: ctx shape mismatch"
        );

        let legacy = dev.cuda_stream();
        let raw_ctx = legacy.context().clone();
        if raw_ctx.is_event_tracking() {
            unsafe { raw_ctx.disable_event_tracking() };
            legacy
                .synchronize()
                .map_err(|e| anyhow::anyhow!("pre-capture legacy sync: {e:?}"))?;
        }

        {
            let x_c = resid.contiguous()?;
            let (xs, xl) = x_c.storage_and_layout();
            let x_cuda = match &*xs {
                candle_core::Storage::Cuda(s) => s,
                _ => anyhow::bail!("graph moe: residual must be CUDA"),
            };
            let x_slice = x_cuda.as_cuda_slice::<bf16>()?;
            let x_view = x_slice.slice(xl.start_offset()..xl.start_offset() + hidden);
            legacy
                .memcpy_dtod(&x_view, &mut ctx.resid_in)
                .map_err(|e| anyhow::anyhow!("resid stage dtod: {e:?}"))?;
        }

        if self.runners[layer].is_none() {
            self.runners[layer] = Some(CudaGraphRunner::new(self.forked.clone()));
        }

        let dev2 = dev.clone();
        let aux_gemm = self.aux_gemm.clone();
        let aux_shared = self.aux_shared.clone();
        let body = |s: &Arc<CudaStream>, ctx: &mut GroupedDecodeContext| -> Result<()> {
            moe_block_body(s, &aux_gemm, &aux_shared, moe, norm, w, ctx, &dev2, None)
        };

        let forked = self.forked.clone();
        let was_cached = self.runners[layer].as_ref().unwrap().has_cached();
        if !was_cached {
            legacy
                .synchronize()
                .map_err(|e| anyhow::anyhow!("pre-warm legacy sync: {e:?}"))?;
            nv_layers::cuda_stream::with_stream(forked.clone(), || body(&forked, &mut *ctx))
                .with_context(|| format!("warm pass, moe layer {layer}"))?;
            forked
                .synchronize()
                .map_err(|e| anyhow::anyhow!("warm sync: {e:?}"))?;
        }

        let runner = self.runners[layer].as_mut().unwrap();
        runner
            .run_on(1u64, Some(&legacy), |s| {
                nv_layers::cuda_stream::with_stream(s.clone(), || body(s, &mut *ctx))
            })
            .with_context(|| format!("graph capture/replay, moe layer {layer}"))?;
        if was_cached {
            self.replays += 1;
        } else {
            self.captures += 1;
            forked
                .synchronize()
                .map_err(|e| anyhow::anyhow!("post-capture sync: {e:?}"))?;
        }

        let mut out: CudaSlice<bf16> = unsafe {
            legacy
                .alloc::<bf16>(hidden)
                .map_err(|e| anyhow::anyhow!(e))?
        };
        legacy
            .memcpy_dtod(&ctx.out_bf16, &mut out)
            .map_err(|e| anyhow::anyhow!("out dtod: {e:?}"))?;
        let storage = candle_core::CudaStorage::wrap_cuda_slice(out, dev.clone());
        Ok(Tensor::from_storage(
            candle_core::Storage::Cuda(storage),
            (1usize, hidden),
            candle_core::op::BackpropOp::none(),
            false,
        ))
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn moe_block_body(
    s: &Arc<CudaStream>,
    aux_gemm: &Arc<CudaStream>,
    aux_shared: &Arc<CudaStream>,
    moe: &LagunaMoe,
    norm: &RmsNorm,
    w: &MoeGroupedWeights,
    ctx: &mut GroupedDecodeContext,
    dev2: &CudaDevice,
    prof: Option<(&StepProfStamps, usize)>,
) -> Result<()> {
    let hidden = w.hidden_size;
    let e = w.num_experts;
    let n = ctx.n_tokens();
    let mut resid_copy: CudaSlice<bf16> = unsafe {
        s.alloc::<bf16>(n * hidden)
            .map_err(|e| anyhow::anyhow!(e))?
    };
    s.memcpy_dtod(&ctx.resid_in, &mut resid_copy)
        .map_err(|e| anyhow::anyhow!("resid_copy dtod: {e:?}"))?;
    let resid_t = {
        let storage = candle_core::CudaStorage::wrap_cuda_slice(resid_copy, dev2.clone());
        Tensor::from_storage(
            candle_core::Storage::Cuda(storage),
            (n, hidden),
            candle_core::op::BackpropOp::none(),
            false,
        )
    };
    let x_t = norm.forward(&resid_t)?;
    anyhow::ensure!(
        x_t.dtype() == DType::BF16 && x_t.elem_count() == n * hidden,
        "graph moe: unexpected norm output {:?} {:?}",
        x_t.dims(),
        x_t.dtype()
    );
    {
        let (ns, nl) = x_t.storage_and_layout();
        anyhow::ensure!(nl.is_contiguous(), "graph moe: norm output not contiguous");
        let n_cuda = match &*ns {
            candle_core::Storage::Cuda(st) => st,
            _ => anyhow::bail!("graph moe: norm output must be CUDA"),
        };
        let n_slice = n_cuda.as_cuda_slice::<bf16>()?;
        let n_view = n_slice.slice(nl.start_offset()..nl.start_offset() + n * hidden);
        s.memcpy_dtod(&n_view, &mut ctx.x_in)
            .map_err(|e| anyhow::anyhow!("normed dtod: {e:?}"))?;
    }

    let logits_t = moe.gate.forward(&x_t)?;
    anyhow::ensure!(
        logits_t.dtype() == DType::BF16 && logits_t.elem_count() == n * e,
        "graph moe: unexpected router logits {:?} {:?}",
        logits_t.dims(),
        logits_t.dtype()
    );
    {
        let (ls, ll) = logits_t.storage_and_layout();
        anyhow::ensure!(
            ll.is_contiguous(),
            "graph moe: router logits not contiguous"
        );
        let l_cuda = match &*ls {
            candle_core::Storage::Cuda(s) => s,
            _ => anyhow::bail!("graph moe: logits must be CUDA"),
        };
        let l_slice = l_cuda.as_cuda_slice::<bf16>()?;
        let l_view = l_slice.slice(ll.start_offset()..ll.start_offset() + n * e);
        let (lp, _g1) = l_view.device_ptr(s);
        let (dp, _g2) = ctx.logits_f32.device_ptr_mut(s);
        let rc = unsafe {
            nv_kernels::cuda::tanh_softcap_bf16_to_f32(
                s.cu_stream() as *mut c_void,
                lp as *const u16,
                dp as *mut f32,
                0.0,
                n * e,
            )
        };
        anyhow::ensure!(rc == 0, "logits cast rc={rc}");
    }
    if let Some((p, li)) = prof {
        p.record(ProfPoint::MoeNormGate(li), s)?;
    }

    let bias_ptr: u64 = {
        let (bs, bl) = moe.selection_bias.storage_and_layout();
        anyhow::ensure!(
            bl.is_contiguous() && moe.selection_bias.dtype() == DType::F32,
            "graph moe: selection_bias must be contiguous F32"
        );
        let b_cuda = match &*bs {
            candle_core::Storage::Cuda(s) => s,
            _ => anyhow::bail!("graph moe: selection_bias must be CUDA"),
        };
        let b_slice = b_cuda.as_cuda_slice::<f32>()?;
        let b_view = b_slice.slice(bl.start_offset()..bl.start_offset() + e);
        let (bp, _g) = b_view.device_ptr(s);
        bp
    };

    let ev_shared = if ctx.folds_shared() {
        None
    } else {
        let ev_fork = s
            .record_event(None)
            .map_err(|e| anyhow::anyhow!("shared fork event: {e:?}"))?;
        aux_shared
            .wait(&ev_fork)
            .map_err(|e| anyhow::anyhow!("shared fork wait: {e:?}"))?;
        if let Some((p, li)) = prof {
            p.record(ProfPoint::SharedStart(li), aux_shared)?;
        }
        nv_layers::cuda_stream::with_stream(aux_shared.clone(), || -> Result<()> {
        let shared_t = moe.shared_expert.forward_fused_cuda(&x_t)?;
        anyhow::ensure!(
            shared_t.dtype() == DType::BF16 && shared_t.elem_count() == n * hidden,
            "graph moe: unexpected shared expert output {:?} {:?}",
            shared_t.dims(),
            shared_t.dtype()
        );
        let (ss, sl) = shared_t.storage_and_layout();
        anyhow::ensure!(
            sl.is_contiguous(),
            "graph moe: shared output not contiguous"
        );
        let s_cuda = match &*ss {
            candle_core::Storage::Cuda(st) => st,
            _ => anyhow::bail!("graph moe: shared output must be CUDA"),
        };
        let s_slice = s_cuda.as_cuda_slice::<bf16>()?;
        let s_view = s_slice.slice(sl.start_offset()..sl.start_offset() + n * hidden);
        let (sp, _g1) = s_view.device_ptr(aux_shared);
        let (dp, _g2) = ctx.shared_f32.device_ptr_mut(aux_shared);
        let rc = unsafe {
            nv_kernels::cuda::cast_bf16_f32(
                aux_shared.cu_stream() as *mut c_void,
                sp as *const u16,
                dp as *mut f32,
                (n * hidden) as i32,
            )
        };
        anyhow::ensure!(rc == 0, "shared cast rc={rc}");
        Ok(())
        })?;
        if let Some((p, li)) = prof {
            p.record(ProfPoint::SharedEnd(li), aux_shared)?;
        }
        Some(
            aux_shared
                .record_event(None)
                .map_err(|e| anyhow::anyhow!("shared join event: {e:?}"))?,
        )
    };

    let fuse_tail = tail_fuse_enabled();
    forward_grouped_decode_into(
        w,
        ctx,
        bias_ptr,
        1,
        moe.softcap,
        moe.norm_topk,
        moe.routed_scaling,
        !fuse_tail,
        s,
        Some(aux_gemm),
        prof.and_then(|(p, li)| p.moe_grouped_prof_base(li, s)),
    )?;
    if let Some(ev) = ev_shared.as_ref() {
        s.wait(ev)
            .map_err(|e| anyhow::anyhow!("shared join wait: {e:?}"))?;
    }
    if fuse_tail {
        let top_k = ctx.tiles_per_token();
        let (yp, _g1) = ctx.y_down.device_ptr(s);
        let (wp, _g2) = ctx.topk_weights.device_ptr(s);
        let (ip, _g3) = ctx.inv_perm.device_ptr(s);
        let (sp, _g4) = ctx.shared_f32.device_ptr(s);
        let (rp, _g5) = ctx.resid_in.device_ptr(s);
        let (op, _g6) = ctx.out_bf16.device_ptr_mut(s);
        let rc = unsafe {
            nv_kernels::cuda::moe_unpermute_scatter_tail(
                s.cu_stream() as *mut c_void,
                yp as *const u16,
                wp as *const f32,
                ip as *const i32,
                sp as *const f32,
                rp as *const u16,
                op as *mut u16,
                n as i32,
                top_k as i32,
                hidden as i32,
                hidden as i32,
            )
        };
        anyhow::ensure!(rc == 0, "moe scatter+tail rc={rc}");
    } else {
        {
            let (ap, _g1) = ctx.y_acc.device_ptr(s);
            let (bp, _g2) = ctx.shared_f32.device_ptr(s);
            let (yp, _g3) = ctx.out_f32.device_ptr_mut(s);
            let rc = unsafe {
                nv_kernels::cuda::add_scale_f32(
                    s.cu_stream() as *mut c_void,
                    ap as *const f32,
                    bp as *const f32,
                    yp as *mut f32,
                    1.0,
                    (n * hidden) as i32,
                )
            };
            anyhow::ensure!(rc == 0, "routed+shared add rc={rc}");
        }
        {
            let (fp, _g1) = ctx.out_f32.device_ptr(s);
            let (bp, _g2) = ctx.ffn_bf16.device_ptr_mut(s);
            let rc = unsafe {
                nv_kernels::cuda::cast_f32_bf16(
                    s.cu_stream() as *mut c_void,
                    fp as *const f32,
                    bp as *mut u16,
                    (n * hidden) as i32,
                )
            };
            anyhow::ensure!(rc == 0, "ffn cast rc={rc}");
        }
        {
            let (ap, _g1) = ctx.resid_in.device_ptr(s);
            let (bp, _g2) = ctx.ffn_bf16.device_ptr(s);
            let (yp, _g3) = ctx.out_bf16.device_ptr_mut(s);
            let rc = unsafe {
                nv_kernels::cuda::residual_add_scale_bf16(
                    s.cu_stream() as *mut c_void,
                    ap as *const u16,
                    bp as *const u16,
                    yp as *mut u16,
                    1.0,
                    n * hidden,
                )
            };
            anyhow::ensure!(rc == 0, "residual add rc={rc}");
        }
    }
    Ok(())
}

impl Drop for LagunaMoeGraphs {
    fn drop(&mut self) {
        let td = crate::gemma4_batch_graph::graph_teardown::GraphTeardown::new(&self.forked)
            .with_stream(&self.aux_gemm)
            .with_stream(&self.aux_shared);
        let runners = &mut self.runners;
        td.run(|| {
            for r in runners.iter_mut().flatten() {
                r.invalidate();
            }
        });
    }
}
