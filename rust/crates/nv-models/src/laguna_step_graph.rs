#![cfg(feature = "cuda")]

use anyhow::{Context, Result};
use candle_core::{CudaDevice, DType, Device, Tensor};
use cudarc::driver::{CudaContext, CudaSlice, CudaStream, DevicePtr, DevicePtrMut};
use half::bf16;
use std::ffi::c_void;
use std::sync::Arc;

use crate::gemma4::{Gemma4Cache, LayerType};
use crate::laguna::{Laguna, LagunaFfn, LagunaKvCache};
use crate::laguna_fa2::{varlen_fwd_bf16, VarlenArgs};
use crate::laguna_fp8::LagunaKvCacheFp8;
use crate::laguna_graph::moe_block_body;
use nv_kernels::graph::CudaGraphRunner;
use nv_layers::moe_grouped::{GroupedDecodeContext, MoeGroupedWeights};

pub(crate) fn verify_prof_enabled() -> bool {
    std::env::var("NV_LAGUNA_VERIFY_PROF")
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true") || v.eq_ignore_ascii_case("fine"))
        .unwrap_or(false)
}

pub(crate) fn fine_prof_enabled() -> bool {
    std::env::var("NV_LAGUNA_VERIFY_PROF")
        .map(|v| v.eq_ignore_ascii_case("fine"))
        .unwrap_or(false)
}

pub(crate) fn m1_flash_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        std::env::var("NV_LAGUNA_M1_FLASH")
            .map(|v| v != "0")
            .unwrap_or(true)
    })
}

struct M1FlashBufs {
    scratch: CudaSlice<f32>,
    fan_in: CudaSlice<u32>,
}

fn m1_qkv_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        std::env::var("NV_LAGUNA_M1_QKV")
            .map(|v| v != "0")
            .unwrap_or(true)
    })
}

fn m1_qkv_q8_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        std::env::var("NV_LAGUNA_M1_QKV_Q8")
            .map(|v| v != "0")
            .unwrap_or(false)
    })
}

fn w8_bytes_ptr(p: &crate::laguna::LagunaProjW8, stream: &Arc<CudaStream>) -> u64 {
    match p.bytes() {
        crate::laguna::LagunaProjBytes::I8(b) => {
            let (q, _g) = b.device_ptr(stream);
            q
        }
        crate::laguna::LagunaProjBytes::E4m3(b) => {
            let (q, _g) = b.device_ptr(stream);
            q
        }
    }
}

fn w8_scale_ptr(p: &crate::laguna::LagunaProjW8, stream: &Arc<CudaStream>) -> u64 {
    let (q, _g) = p.row_scale().device_ptr(stream);
    q
}

const FINE_PER_LAYER: usize = 16;

pub(crate) const FINE_STAGES: [&str; 21] = [
    "embed",
    "qkv_proj",
    "rope_norms",
    "kv_append",
    "flash_sliding",
    "flash_full",
    "attn_gate",
    "o_proj",
    "moe_norm_gate",
    "moe_route",
    "moe_gather_quant",
    "moe_gemm_gate_up",
    "moe_silu_quant",
    "moe_gemm_down",
    "moe_scatter",
    "moe_tail",
    "shared_expert",
    "dense_ffn",
    "lm_head",
    "logits_argmax",
    "total",
];

pub(crate) fn prof_timestamp_at(
    buf: &CudaSlice<u64>,
    slot: usize,
    s: &Arc<CudaStream>,
) -> Result<()> {
    let (p, _g) = buf.device_ptr(s);
    let rc = unsafe {
        nv_kernels::cuda::prof_timestamp(
            s.cu_stream() as *mut c_void,
            (p + (slot * 8) as u64) as *mut u64,
        )
    };
    anyhow::ensure!(rc == 0, "prof timestamp rc={rc}");
    Ok(())
}

pub struct StepProfStamps {
    buf: CudaSlice<u64>,
    stream: Arc<CudaStream>,
    n_layers: usize,
    fine: bool,
    layers: Vec<(bool, bool)>,
    acc: std::cell::RefCell<(u64, [f64; 21])>,
}

impl StepProfStamps {
    fn new(stream: &Arc<CudaStream>, layers: Vec<(bool, bool)>) -> Result<Self> {
        let n_layers = layers.len();
        let fine = fine_prof_enabled();
        let slots = if fine {
            4 + FINE_PER_LAYER * n_layers
        } else {
            4 + 2 * n_layers
        };
        let buf = stream
            .alloc_zeros::<u64>(slots)
            .map_err(|e| anyhow::anyhow!("prof buf: {e:?}"))?;
        Ok(Self {
            buf,
            stream: stream.clone(),
            n_layers,
            fine,
            layers,
            acc: std::cell::RefCell::new((0, [0.0; 21])),
        })
    }

    pub(crate) fn is_fine(&self) -> bool {
        self.fine
    }

    fn slot(&self, which: ProfPoint) -> usize {
        if self.fine {
            let base = |li: usize| 2 + FINE_PER_LAYER * li;
            return match which {
                ProfPoint::Start => 0,
                ProfPoint::Embed => 1,
                ProfPoint::Qkv(li) => base(li),
                ProfPoint::Rope(li) => base(li) + 1,
                ProfPoint::Append(li) => base(li) + 2,
                ProfPoint::Flash(li) => base(li) + 3,
                ProfPoint::Gate(li) => base(li) + 4,
                ProfPoint::Attn(li) => base(li) + 5,
                ProfPoint::MoeNormGate(li) => base(li) + 6,
                ProfPoint::MoeRoute(li) => base(li) + 7,
                ProfPoint::MoeScatter(li) => base(li) + 12,
                ProfPoint::SharedStart(li) => base(li) + 13,
                ProfPoint::SharedEnd(li) => base(li) + 14,
                ProfPoint::Ffn(li) => base(li) + 15,
                ProfPoint::LmHead => 2 + FINE_PER_LAYER * self.n_layers,
                ProfPoint::End => 3 + FINE_PER_LAYER * self.n_layers,
            };
        }
        match which {
            ProfPoint::Start => 0,
            ProfPoint::Embed => 1,
            ProfPoint::Attn(li) => 2 + 2 * li,
            ProfPoint::Ffn(li) => 3 + 2 * li,
            ProfPoint::LmHead => 2 + 2 * self.n_layers,
            ProfPoint::End => 3 + 2 * self.n_layers,
            _ => usize::MAX,
        }
    }

    pub(crate) fn record(&self, which: ProfPoint, s: &Arc<CudaStream>) -> Result<()> {
        let slot = self.slot(which);
        if slot == usize::MAX {
            return Ok(());
        }
        prof_timestamp_at(&self.buf, slot, s)
    }

    pub(crate) fn moe_grouped_prof_base(&self, li: usize, s: &Arc<CudaStream>) -> Option<u64> {
        if !self.fine {
            return None;
        }
        let (p, _g) = self.buf.device_ptr(s);
        Some(p + (self.slot(ProfPoint::MoeRoute(li)) * 8) as u64)
    }

    fn report_fine(&self, tag: &str, ts: &[u64]) -> Result<()> {
        let el = |a: ProfPoint, b: ProfPoint| -> f64 {
            (ts[self.slot(b)].saturating_sub(ts[self.slot(a)])) as f64 / 1e6
        };
        use ProfPoint as P;
        let mut st = [0f64; 21];
        st[0] = el(P::Start, P::Embed);
        let mut prev = P::Embed;
        for (li, &(sliding, dense)) in self.layers.iter().enumerate() {
            st[1] += el(prev, P::Qkv(li));
            st[2] += el(P::Qkv(li), P::Rope(li));
            st[3] += el(P::Rope(li), P::Append(li));
            let fl = el(P::Append(li), P::Flash(li));
            if sliding {
                st[4] += fl;
            } else {
                st[5] += fl;
            }
            st[6] += el(P::Flash(li), P::Gate(li));
            st[7] += el(P::Gate(li), P::Attn(li));
            if dense {
                st[17] += el(P::Attn(li), P::Ffn(li));
            } else {
                st[8] += el(P::Attn(li), P::MoeNormGate(li));
                let gb = self.slot(P::MoeRoute(li));
                let gel = |a: usize, b: usize| (ts[b].saturating_sub(ts[a])) as f64 / 1e6;
                st[9] += gel(self.slot(P::MoeNormGate(li)), gb);
                for i in 0..5 {
                    st[10 + i] += gel(gb + i, gb + i + 1);
                }
                st[15] += el(P::MoeScatter(li), P::Ffn(li));
                st[16] += el(P::SharedStart(li), P::SharedEnd(li));
            }
            prev = P::Ffn(li);
        }
        st[18] = el(prev, P::LmHead);
        st[19] = el(P::LmHead, P::End);
        st[20] = el(P::Start, P::End);
        let serial_sum: f64 = st[..16].iter().sum::<f64>() + st[17] + st[18] + st[19];
        {
            let mut acc = self.acc.borrow_mut();
            acc.0 += 1;
            if acc.0 > 1 {
                for i in 0..21 {
                    acc.1[i] += st[i];
                }
            }
            let n_acc = acc.0.saturating_sub(1).max(1) as f64;
            let mut line = format!("[laguna_fine_prof] {tag} n={}", acc.0);
            for (i, name) in FINE_STAGES.iter().enumerate() {
                line.push_str(&format!(" {name}={:.4}", st[i]));
            }
            line.push_str(&format!(" serial_sum={serial_sum:.4}"));
            eprintln!("{line}");
            if acc.0 > 1 {
                let mut mline = format!("[laguna_fine_prof_mean] {tag} n={}", acc.0 - 1);
                let mut msum = 0f64;
                for (i, name) in FINE_STAGES.iter().enumerate() {
                    let m = acc.1[i] / n_acc;
                    if i < 16 || i == 17 || i == 18 || i == 19 {
                        msum += m;
                    }
                    mline.push_str(&format!(" {name}={m:.4}"));
                }
                mline.push_str(&format!(" serial_sum={msum:.4}"));
                eprintln!("{mline}");
            }
        }
        Ok(())
    }

    fn report(&self, tag: &str) -> Result<()> {
        self.stream
            .synchronize()
            .map_err(|e| anyhow::anyhow!("prof sync: {e:?}"))?;
        #[allow(deprecated)]
        let ts: Vec<u64> = self
            .stream
            .memcpy_dtov(&self.buf)
            .map_err(|e| anyhow::anyhow!("prof dtoh: {e:?}"))?;
        if self.fine {
            return self.report_fine(tag, &ts);
        }
        let el = |a: usize, b: usize| -> f64 { (ts[b].saturating_sub(ts[a])) as f64 / 1e6 };
        let n = self.n_layers;
        let embed = el(self.slot(ProfPoint::Start), self.slot(ProfPoint::Embed));
        let mut attn_ms = vec![0f64; n];
        let mut ffn_ms = vec![0f64; n];
        let mut prev = self.slot(ProfPoint::Embed);
        for li in 0..n {
            attn_ms[li] = el(prev, self.slot(ProfPoint::Attn(li)));
            ffn_ms[li] = el(
                self.slot(ProfPoint::Attn(li)),
                self.slot(ProfPoint::Ffn(li)),
            );
            prev = self.slot(ProfPoint::Ffn(li));
        }
        let mut line = format!("[laguna_verify_prof] {tag} embed={embed:.3}");
        let mut attn_total = 0f64;
        let mut ffn_total = 0f64;
        let mut g0 = 0usize;
        while g0 < n {
            let g1 = (g0 + 10).min(n);
            let a: f64 = attn_ms[g0..g1].iter().sum();
            let f: f64 = ffn_ms[g0..g1].iter().sum();
            attn_total += a;
            ffn_total += f;
            line.push_str(&format!(
                " attn{g0:02}_{:02}={a:.3} ffn{g0:02}_{:02}={f:.3}",
                g1 - 1,
                g1 - 1
            ));
            g0 = g1;
        }
        let lm = el(prev, self.slot(ProfPoint::LmHead));
        let logits = el(self.slot(ProfPoint::LmHead), self.slot(ProfPoint::End));
        let total = el(self.slot(ProfPoint::Start), self.slot(ProfPoint::End));
        let sum = embed + attn_total + ffn_total + lm + logits;
        line.push_str(&format!(
            " attn_total={attn_total:.3} ffn_total={ffn_total:.3} lm_head={lm:.3} logits={logits:.3} sum={sum:.3} total={total:.3}"
        ));
        eprintln!("{line}");
        Ok(())
    }
}

#[derive(Clone, Copy)]
pub(crate) enum ProfPoint {
    Start,
    Embed,
    Qkv(usize),
    Rope(usize),
    Append(usize),
    Flash(usize),
    Gate(usize),
    Attn(usize),
    MoeNormGate(usize),
    MoeRoute(usize),
    MoeScatter(usize),
    SharedStart(usize),
    SharedEnd(usize),
    Ffn(usize),
    LmHead,
    End,
}

fn bf16_ptr(t: &Tensor, stream: &Arc<CudaStream>, len: usize) -> Result<u64> {
    let (st, l) = t.storage_and_layout();
    anyhow::ensure!(l.is_contiguous(), "step graph: tensor not contiguous");
    let cuda = match &*st {
        candle_core::Storage::Cuda(s) => s,
        _ => anyhow::bail!("step graph: tensor not CUDA"),
    };
    let slice = cuda.as_cuda_slice::<bf16>()?;
    let view = slice.slice(l.start_offset()..l.start_offset() + len);
    let (p, _g) = view.device_ptr(stream);
    Ok(p)
}

fn wrap_bf16(slice: CudaSlice<bf16>, dims: (usize, usize, usize), dev: &CudaDevice) -> Tensor {
    let storage = candle_core::CudaStorage::wrap_cuda_slice(slice, dev.clone());
    Tensor::from_storage(
        candle_core::Storage::Cuda(storage),
        dims,
        candle_core::op::BackpropOp::none(),
        false,
    )
}

pub struct LagunaStepGraph {
    model: Arc<Laguna>,
    cache: LagunaKvCache,
    dev: CudaDevice,
    device: Device,
    forked: Arc<CudaStream>,
    aux_gemm: Arc<CudaStream>,
    aux_shared: Arc<CudaStream>,
    runner: CudaGraphRunner,
    tok_buf: CudaSlice<u32>,
    host_tok: Box<[u32; 1]>,
    cu_q: CudaSlice<i32>,
    cu_full: CudaSlice<i32>,
    cu_slide: CudaSlice<i32>,
    logits_out: CudaSlice<f32>,
    am_part_val: CudaSlice<f32>,
    am_part_idx: CudaSlice<i32>,
    token_out: CudaSlice<u32>,
    meta_ptr: u64,
    layer_kv: Vec<Option<(Tensor, Tensor)>>,
    grouped: Vec<Option<Arc<MoeGroupedWeights>>>,
    moe_ctx: GroupedDecodeContext,
    s_cap: usize,
    captured: bool,
    prof: Option<StepProfStamps>,
    m1_flash: Option<M1FlashBufs>,

    _err_drain: CtxErrDrain,
}

struct CtxErrDrain(Arc<CudaContext>);

impl Drop for CtxErrDrain {
    fn drop(&mut self) {
        let _ = self.0.check_err();
    }
}

unsafe impl Send for LagunaStepGraph {}

impl LagunaStepGraph {
    pub fn new(model: Arc<Laguna>, cache: LagunaKvCache) -> Result<Self> {
        let device = model.device().clone();
        let dev = match &device {
            Device::Cuda(d) => d.clone(),
            _ => anyhow::bail!("LagunaStepGraph requires a CUDA device"),
        };
        anyhow::ensure!(
            model.dtype() == DType::BF16,
            "LagunaStepGraph requires BF16"
        );
        anyhow::ensure!(
            cache.has_ring(),
            "LagunaStepGraph requires the ring KV cache"
        );
        let meta_ptr = cache
            .ring_meta_ptr()
            .ok_or_else(|| anyhow::anyhow!("ring meta missing"))?;
        let s_cap = cache.sliding_cap().unwrap_or(0);
        anyhow::ensure!(s_cap > 0, "LagunaStepGraph: no sliding capacity");

        let raw_ctx: Arc<CudaContext> = dev.cuda_stream().context().clone();
        let mut ctor_guard = crate::gemma4_batch_graph::graph_teardown::CtorForkGuard::new();
        let forked = ctor_guard
            .fork(&raw_ctx)
            .map_err(|e| anyhow::anyhow!("step graph stream: {e:?}"))?;
        let aux_gemm = ctor_guard
            .fork(&raw_ctx)
            .map_err(|e| anyhow::anyhow!("step graph aux gemm stream: {e:?}"))?;
        let aux_shared = ctor_guard
            .fork(&raw_ctx)
            .map_err(|e| anyhow::anyhow!("step graph aux shared stream: {e:?}"))?;

        let cfg = model.config();
        let tok_buf = forked
            .alloc_zeros::<u32>(1)
            .map_err(|e| anyhow::anyhow!(e))?;
        #[allow(deprecated)]
        let cu_q = forked
            .memcpy_stod(&[0i32, 1i32])
            .map_err(|e| anyhow::anyhow!(e))?;
        let cu_full = forked
            .alloc_zeros::<i32>(2)
            .map_err(|e| anyhow::anyhow!(e))?;
        let cu_slide = forked
            .alloc_zeros::<i32>(2)
            .map_err(|e| anyhow::anyhow!(e))?;
        let logits_out = forked
            .alloc_zeros::<f32>(cfg.vocab_size)
            .map_err(|e| anyhow::anyhow!(e))?;
        let am_parts = nv_kernels::cuda::argmax_parts().max(1);
        let am_part_val = forked
            .alloc_zeros::<f32>(am_parts)
            .map_err(|e| anyhow::anyhow!(e))?;
        let am_part_idx = forked
            .alloc_zeros::<i32>(am_parts)
            .map_err(|e| anyhow::anyhow!(e))?;
        let token_out = forked
            .alloc_zeros::<u32>(1)
            .map_err(|e| anyhow::anyhow!(e))?;

        let mut layer_kv = Vec::with_capacity(model.layers().len());
        let mut grouped = Vec::with_capacity(model.layers().len());
        for (li, layer) in model.layers().iter().enumerate() {
            layer_kv.push(Some(cache.layer_kv_bufs(li)));
            grouped.push(match &layer.ffn {
                LagunaFfn::Moe(moe) => {
                    let w = model
                        .grouped_weights(moe)?
                        .ok_or_else(|| anyhow::anyhow!("grouped MoE weights unavailable"))?;
                    Some(w)
                }
                LagunaFfn::Dense(_) => None,
            });
        }
        let weights_fold_shared = grouped
            .iter()
            .flatten()
            .any(|w: &Arc<MoeGroupedWeights>| w.folded_shared);
        let moe_ctx = if weights_fold_shared {
            GroupedDecodeContext::new_folded_shared(
                cfg.hidden_size,
                cfg.moe_intermediate_size,
                cfg.num_experts_per_tok,
                cfg.num_experts,
                1,
                &forked,
            )?
        } else {
            GroupedDecodeContext::new(
                cfg.hidden_size,
                cfg.moe_intermediate_size,
                cfg.num_experts_per_tok,
                cfg.num_experts,
                &forked,
            )?
        };
        forked.synchronize().map_err(|e| anyhow::anyhow!(e))?;

        let prof = if verify_prof_enabled() {
            let layer_info: Vec<(bool, bool)> = model
                .layers()
                .iter()
                .map(|l| {
                    (
                        matches!(l.self_attn.kind, LayerType::SlidingAttention),
                        matches!(l.ffn, LagunaFfn::Dense(_)),
                    )
                })
                .collect();
            Some(StepProfStamps::new(&forked, layer_info)?)
        } else {
            None
        };
        let m1_flash = if m1_flash_enabled()
            && cfg.head_dim == 128
            && cfg.num_attention_heads % cfg.num_key_value_heads == 0
            && matches!(cfg.num_attention_heads / cfg.num_key_value_heads, 6 | 8)
        {
            let elems = nv_kernels::cuda::laguna_flash_decode_gqa_scratch_elems(
                cfg.num_key_value_heads as i32,
            );
            Some(M1FlashBufs {
                scratch: forked
                    .alloc_zeros::<f32>(elems)
                    .map_err(|e| anyhow::anyhow!("m1 flash scratch: {e:?}"))?,
                fan_in: forked
                    .alloc_zeros::<u32>(cfg.num_key_value_heads)
                    .map_err(|e| anyhow::anyhow!("m1 flash fan_in: {e:?}"))?,
            })
        } else {
            None
        };
        let runner = CudaGraphRunner::new(forked.clone());
        ctor_guard.the_built_engine_owns_teardown_now();
        Ok(Self {
            model,
            cache,
            dev,
            device,
            forked,
            aux_gemm,
            aux_shared,
            runner,
            tok_buf,
            host_tok: Box::new([0u32; 1]),
            cu_q,
            cu_full,
            cu_slide,
            logits_out,
            am_part_val,
            am_part_idx,
            token_out,
            meta_ptr,
            layer_kv,
            grouped,
            moe_ctx,
            s_cap,
            captured: false,
            prof,
            m1_flash,
            _err_drain: CtxErrDrain(raw_ctx),
        })
    }

    pub fn cache(&self) -> &LagunaKvCache {
        &self.cache
    }

    pub fn cache_mut(&mut self) -> &mut LagunaKvCache {
        &mut self.cache
    }

    pub fn current_len(&self) -> usize {
        self.cache.current_len()
    }

    pub fn synchronize(&self) -> Result<()> {
        nv_layers::cuda_stream::sync_legacy_then_forked(&self.dev, &self.forked)
    }

    pub fn step(&mut self, token: u32) -> Result<()> {
        self.model.apply_attn_w8_for(1);
        let write_pos = self.cache.current_len();
        self.cache
            .prepare_for_decode(write_pos, write_pos + 1)
            .context("step graph prepare_for_decode")?;
        self.host_tok[0] = token;
        let legacy = self.dev.cuda_stream();
        legacy
            .memcpy_htod(&self.host_tok[..], &mut self.tok_buf)
            .map_err(|e| anyhow::anyhow!("htod token: {e:?}"))?;

        let raw_ctx = legacy.context().clone();
        if raw_ctx.is_event_tracking() {
            unsafe { raw_ctx.disable_event_tracking() };
            legacy
                .synchronize()
                .map_err(|e| anyhow::anyhow!("pre-capture legacy sync: {e:?}"))?;
        }

        let was_captured = self.captured;
        let forked = self.forked.clone();
        let LagunaStepGraph {
            model,
            dev,
            device,
            aux_gemm,
            aux_shared,
            runner,
            tok_buf,
            cu_q,
            cu_full,
            cu_slide,
            logits_out,
            am_part_val,
            am_part_idx,
            token_out,
            meta_ptr,
            layer_kv,
            grouped,
            moe_ctx,
            s_cap,
            cache,
            prof,
            m1_flash,
            ..
        } = self;
        let meta_ptr = *meta_ptr;
        let s_cap = *s_cap;
        let max_seq_len = cache.max_seq_len();
        let prof = prof.as_ref();
        let m1_flash = m1_flash.as_ref();

        let logits_ptr = {
            let (lp, _g) = logits_out.device_ptr(&forked);
            lp
        };
        let mut body = |s: &Arc<CudaStream>, moe_ctx: &mut GroupedDecodeContext| -> Result<()> {
            step_body(
                model,
                dev,
                device,
                s,
                aux_gemm,
                aux_shared,
                tok_buf,
                cu_q,
                cu_full,
                cu_slide,
                logits_ptr,
                Some((&mut *am_part_val, &mut *am_part_idx, &mut *token_out)),
                meta_ptr,
                layer_kv,
                grouped,
                moe_ctx,
                s_cap,
                max_seq_len,
                1,
                &[],
                &[],
                prof,
                m1_flash,
                None,
                false,
            )
        };

        if !was_captured {
            legacy
                .synchronize()
                .map_err(|e| anyhow::anyhow!("pre-warm legacy sync: {e:?}"))?;
            nv_layers::cuda_stream::with_stream(forked.clone(), || body(&forked, moe_ctx))
                .context("step graph warm pass")?;
            forked
                .synchronize()
                .map_err(|e| anyhow::anyhow!("warm sync: {e:?}"))?;
        }

        runner
            .run_on(1u64, Some(&legacy), |s| {
                nv_layers::cuda_stream::with_stream(s.clone(), || body(s, moe_ctx))
            })
            .context("step graph capture/replay")?;
        if !was_captured {
            forked
                .synchronize()
                .map_err(|e| anyhow::anyhow!("post-capture sync: {e:?}"))?;
            self.captured = true;
        }
        if let Some(p) = self.prof.as_ref() {
            self.dev
                .cuda_stream()
                .synchronize()
                .map_err(|e| anyhow::anyhow!("prof legacy sync: {e:?}"))?;
            p.report("m1_step")?;
        }

        self.cache.advance(1);
        self.cache.note_graph_write();
        Ok(())
    }

    pub fn argmax_device(&self) -> Result<u32> {
        self.synchronize()?;
        let out = self
            .forked
            .clone_dtoh(&self.token_out)
            .map_err(|e| anyhow::anyhow!("dtoh token: {e:?}"))?;
        Ok(out[0])
    }

    pub fn logits_host(&self) -> Result<Vec<f32>> {
        self.synchronize()?;
        self.forked
            .clone_dtoh(&self.logits_out)
            .map_err(|e| anyhow::anyhow!("dtoh logits: {e:?}"))
    }

    pub fn argmax_host(&self) -> Result<u32> {
        let logits = self.logits_host()?;
        let mut best = 0usize;
        let mut best_v = f32::NEG_INFINITY;
        for (i, &v) in logits.iter().enumerate() {
            if v >= best_v {
                best_v = v;
                best = i;
            }
        }
        Ok(best as u32)
    }
}

impl Drop for LagunaStepGraph {
    fn drop(&mut self) {
        crate::gemma4_batch_graph::graph_teardown::GraphTeardown::new(&self.forked)
            .with_stream(&self.aux_gemm)
            .with_stream(&self.aux_shared)
            .run(|| self.runner.invalidate());
    }
}

pub(crate) struct Fp8VerifyLayer {
    k_fp8: u64,
    v_fp8: u64,
    k_scales: u64,
    v_scales: u64,
}

pub(crate) struct Fp8VerifyState {
    layers: Vec<Option<Fp8VerifyLayer>>,
    n_committed_ptr: u64,
    k_scratch: CudaSlice<bf16>,
    v_scratch: CudaSlice<bf16>,
}

#[allow(clippy::too_many_arguments)]
fn step_body(
    model: &Laguna,
    dev: &CudaDevice,
    device: &Device,
    s: &Arc<CudaStream>,
    aux_gemm: &Arc<CudaStream>,
    aux_shared: &Arc<CudaStream>,
    tok_buf: &CudaSlice<u32>,
    cu_q: &CudaSlice<i32>,
    cu_full: &mut CudaSlice<i32>,
    cu_slide: &mut CudaSlice<i32>,
    logits_ptr: u64,
    argmax_slots: Option<(
        &mut CudaSlice<f32>,
        &mut CudaSlice<i32>,
        &mut CudaSlice<u32>,
    )>,
    meta_ptr: u64,
    layer_kv: &[Option<(Tensor, Tensor)>],
    grouped: &[Option<Arc<MoeGroupedWeights>>],
    moe_ctx: &mut GroupedDecodeContext,
    s_cap: usize,
    max_seq_len: usize,
    t: usize,
    aux_layers: &[usize],
    aux_dst: &[u64],
    prof: Option<&StepProfStamps>,
    m1_flash: Option<&M1FlashBufs>,
    fp8: Option<&Fp8VerifyState>,
    spec_head: bool,
) -> Result<()> {
    let cfg = model.config();
    let hidden = cfg.hidden_size;
    let n_kv = cfg.num_key_value_heads;
    let hd = cfg.head_dim;
    anyhow::ensure!(moe_ctx.n_tokens() == t, "moe ctx n_tokens != t");
    anyhow::ensure!(aux_layers.len() == aux_dst.len(), "aux slot mismatch");
    if let Some(p) = prof {
        anyhow::ensure!(
            p.n_layers == model.layers().len(),
            "prof events sized wrong"
        );
        p.record(ProfPoint::Start, s)?;
    }

    {
        let (fp, _g1) = cu_full.device_ptr_mut(s);
        let (sp, _g2) = cu_slide.device_ptr_mut(s);
        let rc = unsafe {
            nv_kernels::cuda::laguna_seqlens_prep(
                s.cu_stream() as *mut c_void,
                meta_ptr as *const i32,
                fp as *mut i32,
                sp as *mut i32,
                t as i32,
            )
        };
        anyhow::ensure!(rc == 0, "seqlens prep rc={rc}");
    }

    let tok_clone = tok_buf
        .try_clone()
        .map_err(|e| anyhow::anyhow!("clone tok_buf: {e:?}"))?;
    let tokens_t = {
        let storage = candle_core::CudaStorage::wrap_cuda_slice(tok_clone, dev.clone());
        Tensor::from_storage(
            candle_core::Storage::Cuda(storage),
            (t,),
            candle_core::op::BackpropOp::none(),
            false,
        )
    };
    let mut x = crate::gemma4::embed_lookup_bf16_op(model.embed_weight(), &tokens_t, device)?
        .reshape((1usize, t, hidden))?;
    if let Some(p) = prof {
        p.record(ProfPoint::Embed, s)?;
    }

    let (cu_q_ptr, _gq) = cu_q.device_ptr(s);
    let (cu_full_ptr, _gf) = cu_full.device_ptr(s);
    let (cu_slide_ptr, _gs) = cu_slide.device_ptr(s);

    for (li, layer) in model.layers().iter().enumerate() {
        let attn = &layer.self_attn;
        let n_q = attn.num_heads;

        let g_lin = attn.g_proj.as_ref();
        let qkv_q8 = if m1_qkv_q8_enabled() {
            attn.m1_qkv_w8_fused(hidden, n_q, n_kv, hd)
        } else {
            None
        };
        let m1_fused = t == 1
            && m1_qkv_enabled()
            && hd % 32 == 0
            && hd <= 512
            && hidden % 8 == 0
            && hidden <= 4096
            && (!attn.m1_w8_active_qkv() || qkv_q8.is_some())
            && attn.q_proj.weight().is_some()
            && attn.k_proj.weight().is_some()
            && attn.v_proj.weight().is_some()
            && g_lin.map(|g| g.weight().is_some() && g.bias().is_none()) == Some(true)
            && attn.q_proj.bias().is_none()
            && attn.k_proj.bias().is_none()
            && attn.v_proj.bias().is_none();

        let (rope, rotary_dim, rot_scale) = model.rope_for_kind(attn.kind);
        let cos_c = rope.cos().contiguous()?;
        let sin_c = rope.sin().contiguous()?;
        let cos_ptr = {
            let (cs, cl) = cos_c.storage_and_layout();
            let c_cuda = match &*cs {
                candle_core::Storage::Cuda(st) => st,
                _ => anyhow::bail!("rope cos not CUDA"),
            };
            let c_slice = c_cuda.as_cuda_slice::<f32>()?;
            let view = c_slice.slice(cl.start_offset()..);
            let (p, _g) = view.device_ptr(s);
            p
        };
        let sin_ptr = {
            let (ss_, sl) = sin_c.storage_and_layout();
            let s_cuda = match &*ss_ {
                candle_core::Storage::Cuda(st) => st,
                _ => anyhow::bail!("rope sin not CUDA"),
            };
            let s_slice = s_cuda.as_cuda_slice::<f32>()?;
            let view = s_slice.slice(sl.start_offset()..);
            let (p, _g) = view.device_ptr(s);
            p
        };
        let mut q_rot_dev: CudaSlice<bf16> = unsafe {
            s.alloc::<bf16>(t * n_q * hd)
                .map_err(|e| anyhow::anyhow!(e))?
        };
        let mut k_rot_dev: CudaSlice<bf16> = unsafe {
            s.alloc::<bf16>(t * n_kv * hd)
                .map_err(|e| anyhow::anyhow!(e))?
        };
        let mut v_slice: Option<CudaSlice<bf16>> = None;
        let mut g_pre: Option<CudaSlice<bf16>> = None;
        let mut v_tensor: Option<Tensor> = None;
        let mut normed_tensor: Option<Tensor> = None;

        if m1_fused {
            let x_ptr = bf16_ptr(&x, s, t * hidden)?;
            let mut rstd_dev: CudaSlice<f32> =
                unsafe { s.alloc::<f32>(1).map_err(|e| anyhow::anyhow!(e))? };
            {
                let (rp, _g) = rstd_dev.device_ptr_mut(s);
                let rc = unsafe {
                    nv_kernels::cuda::laguna_rstd256_bf16(
                        s.cu_stream() as *mut c_void,
                        x_ptr as *const u16,
                        rp as *mut f32,
                        hidden as i32,
                        layer.input_layernorm.eps() as f32,
                    )
                };
                anyhow::ensure!(rc == 0, "m1 rstd rc={rc} layer {li}");
            }
            let mut q_raw: CudaSlice<bf16> =
                unsafe { s.alloc::<bf16>(n_q * hd).map_err(|e| anyhow::anyhow!(e))? };
            let mut k_raw: CudaSlice<bf16> =
                unsafe { s.alloc::<bf16>(n_kv * hd).map_err(|e| anyhow::anyhow!(e))? };
            let mut v_dev: CudaSlice<bf16> =
                unsafe { s.alloc::<bf16>(n_kv * hd).map_err(|e| anyhow::anyhow!(e))? };
            let mut g_dev: CudaSlice<bf16> =
                unsafe { s.alloc::<bf16>(n_q).map_err(|e| anyhow::anyhow!(e))? };
            {
                let wq = bf16_ptr(attn.q_proj.weight().unwrap(), s, n_q * hd * hidden)?;
                let wk = bf16_ptr(attn.k_proj.weight().unwrap(), s, n_kv * hd * hidden)?;
                let wv = bf16_ptr(attn.v_proj.weight().unwrap(), s, n_kv * hd * hidden)?;
                let wg = bf16_ptr(g_lin.unwrap().weight().unwrap(), s, n_q * hidden)?;
                let wn = bf16_ptr(layer.input_layernorm.weight_bf16(), s, hidden)?;
                let (rp, _g0) = rstd_dev.device_ptr(s);
                let (yq, _g1) = q_raw.device_ptr_mut(s);
                let (yk, _g2) = k_raw.device_ptr_mut(s);
                let (yv, _g3) = v_dev.device_ptr_mut(s);
                let (yg, _g4) = g_dev.device_ptr_mut(s);
                let rc = if let Some((pq, pk, pv, is_fp8)) = qkv_q8 {
                    let (bq, sq) = (w8_bytes_ptr(pq, s), w8_scale_ptr(pq, s));
                    let (bk, sk) = (w8_bytes_ptr(pk, s), w8_scale_ptr(pk, s));
                    let (bv, sv) = (w8_bytes_ptr(pv, s), w8_scale_ptr(pv, s));
                    unsafe {
                        nv_kernels::cuda::gemv_q8_qkvg_normed(
                            s.cu_stream() as *mut c_void,
                            is_fp8 as i32,
                            bq as *const c_void,
                            sq as *const f32,
                            bk as *const c_void,
                            sk as *const f32,
                            bv as *const c_void,
                            sv as *const f32,
                            wg as *const u16,
                            x_ptr as *const u16,
                            wn as *const u16,
                            rp as *const f32,
                            yq as *mut u16,
                            yk as *mut u16,
                            yv as *mut u16,
                            yg as *mut u16,
                            (n_q * hd) as i32,
                            (n_kv * hd) as i32,
                            (n_kv * hd) as i32,
                            n_q as i32,
                            hidden as i32,
                        )
                    }
                } else {
                    unsafe {
                        nv_kernels::cuda::gemv_bf16_qkvg_normed(
                            s.cu_stream() as *mut c_void,
                            wq as *const u16,
                            wk as *const u16,
                            wv as *const u16,
                            wg as *const u16,
                            x_ptr as *const u16,
                            wn as *const u16,
                            rp as *const f32,
                            yq as *mut u16,
                            yk as *mut u16,
                            yv as *mut u16,
                            yg as *mut u16,
                            (n_q * hd) as i32,
                            (n_kv * hd) as i32,
                            (n_kv * hd) as i32,
                            n_q as i32,
                            hidden as i32,
                        )
                    }
                };
                anyhow::ensure!(rc == 0, "m1 qkvg gemv rc={rc} layer {li}");
            }
            if let Some(p) = prof {
                p.record(ProfPoint::Qkv(li), s)?;
            }
            {
                let qw = bf16_ptr(attn.q_norm.weight_bf16(), s, hd)?;
                let kw = bf16_ptr(attn.k_norm.weight_bf16(), s, hd)?;
                let (qi, _g1) = q_raw.device_ptr(s);
                let (ki, _g2) = k_raw.device_ptr(s);
                let (qo, _g3) = q_rot_dev.device_ptr_mut(s);
                let (ko, _g4) = k_rot_dev.device_ptr_mut(s);
                let rc = unsafe {
                    nv_kernels::cuda::laguna_qk_normrope_bf16(
                        s.cu_stream() as *mut c_void,
                        qi as *const u16,
                        ki as *const u16,
                        qo as *mut u16,
                        ko as *mut u16,
                        qw as *const u16,
                        kw as *const u16,
                        cos_ptr as *const f32,
                        sin_ptr as *const f32,
                        meta_ptr as *const i32,
                        n_q as i32,
                        n_kv as i32,
                        hd as i32,
                        rotary_dim as i32,
                        rot_scale,
                        attn.q_norm.eps() as f32,
                        attn.k_norm.eps() as f32,
                    )
                };
                anyhow::ensure!(rc == 0, "m1 normrope rc={rc} layer {li}");
            }
            if let Some(p) = prof {
                p.record(ProfPoint::Rope(li), s)?;
            }
            v_slice = Some(v_dev);
            g_pre = Some(g_dev);
        } else {
            let normed = layer.input_layernorm.forward(&x)?;
            let q = attn.proj_q(&normed, t)?.reshape((1usize, t, n_q, hd))?;
            let q = attn.q_norm.forward(&q)?.contiguous()?;
            let k = attn.proj_k(&normed, t)?.reshape((1usize, t, n_kv, hd))?;
            let k = attn.k_norm.forward(&k)?.contiguous()?;
            let v = attn
                .proj_v(&normed, t)?
                .reshape((1usize, t, n_kv, hd))?
                .contiguous()?;
            if let Some(p) = prof {
                p.record(ProfPoint::Qkv(li), s)?;
            }

            {
                let q_ptr = bf16_ptr(&q, s, t * n_q * hd)?;
                let k_ptr = bf16_ptr(&k, s, t * n_kv * hd)?;
                let (qo, _g1) = q_rot_dev.device_ptr_mut(s);
                let (ko, _g2) = k_rot_dev.device_ptr_mut(s);
                let rc = unsafe {
                    nv_kernels::cuda::laguna_rope_scale_bf16(
                        s.cu_stream() as *mut c_void,
                        q_ptr as *const u16,
                        k_ptr as *const u16,
                        qo as *mut u16,
                        ko as *mut u16,
                        cos_ptr as *const f32,
                        sin_ptr as *const f32,
                        meta_ptr as *const i32,
                        t as i32,
                        n_q as i32,
                        n_kv as i32,
                        hd as i32,
                        rotary_dim as i32,
                        rot_scale,
                    )
                };
                anyhow::ensure!(rc == 0, "rope scale rc={rc} layer {li}");
            }
            if let Some(p) = prof {
                p.record(ProfPoint::Rope(li), s)?;
            }
            v_tensor = Some(v);
            normed_tensor = Some(normed);
        }

        let v_src_ptr = if let Some(vs) = &v_slice {
            let (p, _g) = vs.device_ptr(s);
            p
        } else {
            bf16_ptr(v_tensor.as_ref().unwrap(), s, t * n_kv * hd)?
        };
        let mut o_dev: CudaSlice<bf16> = unsafe {
            s.alloc::<bf16>(t * n_q * hd)
                .map_err(|e| anyhow::anyhow!(e))?
        };
        let fp8_layer = fp8.and_then(|st| st.layers[li].as_ref());
        if let Some(fpl) = fp8_layer {
            let st = fp8.unwrap();
            anyhow::ensure!(
                matches!(attn.kind, LayerType::FullAttention),
                "fp8 kv slot on non-full layer {li}"
            );
            {
                let (k_src, _g) = k_rot_dev.device_ptr(s);
                let rc = unsafe {
                    nv_kernels::cuda::kv_append_fp8(
                        s.cu_stream() as *mut c_void,
                        k_src as *const u16,
                        v_src_ptr as *const u16,
                        fpl.k_fp8 as *mut u8,
                        fpl.v_fp8 as *mut u8,
                        fpl.k_scales as *mut f32,
                        fpl.v_scales as *mut f32,
                        st.n_committed_ptr as *const i32,
                        t as i32,
                        n_kv as i32,
                        hd as i32,
                        0,
                    )
                };
                anyhow::ensure!(rc == 0, "fp8 kv append rc={rc} layer {li}");
            }
            if let Some(p) = prof {
                p.record(ProfPoint::Append(li), s)?;
            }
            let (k_scr, _gks) = st.k_scratch.device_ptr(s);
            let (v_scr, _gvs) = st.v_scratch.device_ptr(s);
            for (src, scales, dst) in [
                (fpl.k_fp8, fpl.k_scales, k_scr),
                (fpl.v_fp8, fpl.v_scales, v_scr),
            ] {
                let rc = unsafe {
                    nv_kernels::cuda::dequantize_kv_fp8(
                        s.cu_stream() as *mut c_void,
                        src as *const u8,
                        scales as *const f32,
                        dst as *mut u16,
                        0,
                        max_seq_len as i32,
                        n_kv as i32,
                        hd as i32,
                        0,
                    )
                };
                anyhow::ensure!(rc == 0, "fp8 verify dequant rc={rc} layer {li}");
            }
            {
                let lse_dev: CudaSlice<f32> = s
                    .alloc_zeros::<f32>(n_q * t)
                    .map_err(|e| anyhow::anyhow!(e))?;
                let (qp, _g1) = q_rot_dev.device_ptr(s);
                let (op, _g2) = o_dev.device_ptr_mut(s);
                let (lp, _g3) = lse_dev.device_ptr(s);
                let scale = (hd as f32).powf(-0.5);
                unsafe {
                    varlen_fwd_bf16(
                        s.cu_stream() as *mut c_void,
                        &VarlenArgs {
                            q_ptr: qp,
                            k_ptr: k_scr,
                            v_ptr: v_scr,
                            o_ptr: op,
                            lse_ptr: lp,
                            cu_seqlens_q: cu_q_ptr,
                            cu_seqlens_k: cu_full_ptr,
                            max_seqlen_q: t,
                            max_seqlen_k: max_seq_len,
                            h: n_q,
                            h_k: n_kv,
                            d: hd,
                            softmax_scale: scale,
                            window_size_left: None,
                            window_size_right: Some(0),
                        },
                    )?;
                }
            }
        } else {
            let (window, cap, meta_idx, cu_k_ptr, max_k) = match attn.kind {
                LayerType::SlidingAttention => {
                    (Some(cfg.sliding_window), s_cap, 1usize, cu_slide_ptr, s_cap)
                }
                LayerType::FullAttention => (None, max_seq_len, 0usize, cu_full_ptr, max_seq_len),
            };
            let (k_buf, v_buf) = layer_kv[li]
                .as_ref()
                .ok_or_else(|| anyhow::anyhow!("missing bf16 kv buffers, layer {li}"))?;
            let k_dst_ptr = bf16_ptr(k_buf, s, cap * n_kv * hd)?;
            let v_dst_ptr = bf16_ptr(v_buf, s, cap * n_kv * hd)?;
            {
                let (k_src, _g) = k_rot_dev.device_ptr(s);
                for (src, dst) in [(k_src, k_dst_ptr), (v_src_ptr, v_dst_ptr)] {
                    let rc = unsafe {
                        nv_kernels::cuda::kv_ring_append_bf16(
                            s.cu_stream() as *mut c_void,
                            src as *const u16,
                            dst as *mut u16,
                            (meta_ptr + (meta_idx * 4) as u64) as *const i32,
                            t as i32,
                            cap as i32,
                            n_kv as i32,
                            hd as i32,
                        )
                    };
                    anyhow::ensure!(rc == 0, "ring append rc={rc} layer {li}");
                }
            }
            if let Some(p) = prof {
                p.record(ProfPoint::Append(li), s)?;
            }

            let m1_gqa = m1_flash
                .filter(|_| t == 1 && hd == 128 && n_q % n_kv == 0 && matches!(n_q / n_kv, 6 | 8));
            if let Some(bufs) = m1_gqa {
                let (qp, _g1) = q_rot_dev.device_ptr(s);
                let (op, _g2) = o_dev.device_ptr_mut(s);
                let (scp, _g3) = bufs.scratch.device_ptr(s);
                let (fip, _g4) = bufs.fan_in.device_ptr(s);
                let scale = (hd as f32).powf(-0.5);
                let rc = unsafe {
                    nv_kernels::cuda::laguna_flash_decode_gqa(
                        s.cu_stream() as *mut c_void,
                        qp as *const u16,
                        k_dst_ptr as *const u16,
                        v_dst_ptr as *const u16,
                        op as *mut u16,
                        (cu_k_ptr + 4) as *const i32,
                        0,
                        scp as *mut f32,
                        fip as *mut u32,
                        n_q as i32,
                        n_kv as i32,
                        hd as i32,
                        window.map(|w| w as i32).unwrap_or(0),
                        scale,
                    )
                };
                anyhow::ensure!(rc == 0, "m1 flash decode rc={rc} layer {li}");
            } else {
                let lse_dev: CudaSlice<f32> = s
                    .alloc_zeros::<f32>(n_q * t)
                    .map_err(|e| anyhow::anyhow!(e))?;
                let (qp, _g1) = q_rot_dev.device_ptr(s);
                let (op, _g2) = o_dev.device_ptr_mut(s);
                let (lp, _g3) = lse_dev.device_ptr(s);
                let scale = (hd as f32).powf(-0.5);
                unsafe {
                    varlen_fwd_bf16(
                        s.cu_stream() as *mut c_void,
                        &VarlenArgs {
                            q_ptr: qp,
                            k_ptr: k_dst_ptr,
                            v_ptr: v_dst_ptr,
                            o_ptr: op,
                            lse_ptr: lp,
                            cu_seqlens_q: cu_q_ptr,
                            cu_seqlens_k: cu_k_ptr,
                            max_seqlen_q: t,
                            max_seqlen_k: max_k,
                            h: n_q,
                            h_k: n_kv,
                            d: hd,
                            softmax_scale: scale,
                            window_size_left: window.map(|w| w.saturating_sub(1)),
                            window_size_right: Some(0),
                        },
                    )?;
                }
            }
        }
        if let Some(p) = prof {
            p.record(ProfPoint::Flash(li), s)?;
        }
        let o_t = wrap_bf16(o_dev, (1usize, t, n_q * hd), dev);

        let g = if g_pre.is_some() {
            None
        } else {
            let g_proj = g_lin
                .ok_or_else(|| anyhow::anyhow!("step graph: expected g_proj on layer {li}"))?;
            Some(
                g_proj
                    .forward(normed_tensor.as_ref().unwrap())?
                    .contiguous()?,
            )
        };
        let gated = {
            let mut out_dev: CudaSlice<bf16> = unsafe {
                s.alloc::<bf16>(t * n_q * hd)
                    .map_err(|e| anyhow::anyhow!(e))?
            };
            let a_ptr = bf16_ptr(&o_t, s, t * n_q * hd)?;
            let g_ptr = if let Some(gd) = &g_pre {
                let (p, _g) = gd.device_ptr(s);
                p
            } else {
                bf16_ptr(g.as_ref().unwrap(), s, t * n_q)?
            };
            {
                let (op, _g) = out_dev.device_ptr_mut(s);
                let rc = unsafe {
                    nv_kernels::cuda::softplus_gate_exact_bf16(
                        s.cu_stream() as *mut c_void,
                        a_ptr as *const u16,
                        g_ptr as *const u16,
                        op as *mut u16,
                        (t * n_q) as i32,
                        hd as i32,
                    )
                };
                anyhow::ensure!(rc == 0, "softplus gate rc={rc} layer {li}");
            }
            wrap_bf16(out_dev, (1usize, t, n_q * hd), dev)
        };
        if let Some(p) = prof {
            p.record(ProfPoint::Gate(li), s)?;
        }
        let attn_out = attn.proj_o(&gated, t)?;
        let after_attn = crate::gemma4::residual_add_scale_bf16_op(&x, &attn_out, 1.0, device)?;
        if let Some(p) = prof {
            p.record(ProfPoint::Attn(li), s)?;
        }

        x = match &layer.ffn {
            LagunaFfn::Dense(mlp) => {
                let normed_mlp = layer.post_attention_layernorm.forward(&after_attn)?;
                let ffn = mlp.forward_fused_cuda(&normed_mlp)?;
                crate::gemma4::residual_add_scale_bf16_op(&after_attn, &ffn, 1.0, device)?
            }
            LagunaFfn::Moe(moe) => {
                let w = grouped[li]
                    .as_ref()
                    .ok_or_else(|| anyhow::anyhow!("missing grouped weights, layer {li}"))?;

                {
                    let (rs, rl) = after_attn.storage_and_layout();
                    anyhow::ensure!(rl.is_contiguous(), "resid not contiguous");
                    let r_cuda = match &*rs {
                        candle_core::Storage::Cuda(st) => st,
                        _ => anyhow::bail!("resid not CUDA"),
                    };
                    let r_slice = r_cuda.as_cuda_slice::<bf16>()?;
                    let r_view = r_slice.slice(rl.start_offset()..rl.start_offset() + t * hidden);
                    s.memcpy_dtod(&r_view, &mut moe_ctx.resid_in)
                        .map_err(|e| anyhow::anyhow!("resid stage dtod: {e:?}"))?;
                }
                moe_block_body(
                    s,
                    aux_gemm,
                    aux_shared,
                    moe,
                    &layer.post_attention_layernorm,
                    w,
                    moe_ctx,
                    dev,
                    prof.filter(|p| p.is_fine()).map(|p| (p, li)),
                )?;
                let mut out_dev: CudaSlice<bf16> = unsafe {
                    s.alloc::<bf16>(t * hidden)
                        .map_err(|e| anyhow::anyhow!(e))?
                };
                s.memcpy_dtod(&moe_ctx.out_bf16, &mut out_dev)
                    .map_err(|e| anyhow::anyhow!("moe out dtod: {e:?}"))?;
                wrap_bf16(out_dev, (1usize, t, hidden), dev)
            }
        };

        if let Some(slot) = aux_layers.iter().position(|&l| l == li) {
            let src_ptr = bf16_ptr(&x, s, t * hidden)?;
            unsafe {
                cudarc::driver::result::memcpy_dtod_async(
                    aux_dst[slot],
                    src_ptr,
                    t * hidden * 2,
                    s.cu_stream(),
                )
                .map_err(|e| anyhow::anyhow!("aux stage dtod: {e:?}"))?;
            }
        }
        if let Some(p) = prof {
            p.record(ProfPoint::Ffn(li), s)?;
        }
    }

    let raw = model.lm_head_from_prenorm_scoped(&x, spec_head)?;
    if let Some(p) = prof {
        p.record(ProfPoint::LmHead, s)?;
    }
    let vocab = cfg.vocab_size;
    {
        let raw_ptr = bf16_ptr(&raw, s, t * vocab)?;
        let rc = unsafe {
            nv_kernels::cuda::tanh_softcap_bf16_to_f32(
                s.cu_stream() as *mut c_void,
                raw_ptr as *const u16,
                logits_ptr as *mut f32,
                0.0,
                t * vocab,
            )
        };
        anyhow::ensure!(rc == 0, "logits cast rc={rc}");
    }

    if let Some((am_part_val, am_part_idx, token_out)) = argmax_slots {
        let raw_ptr = bf16_ptr(&raw, s, t * vocab)? + ((t - 1) * vocab * 2) as u64;
        let (vp, _g1) = am_part_val.device_ptr_mut(s);
        let (ip, _g2) = am_part_idx.device_ptr_mut(s);
        let (tp, _g3) = token_out.device_ptr_mut(s);
        let rc = unsafe {
            nv_kernels::cuda::argmax_bf16(
                s.cu_stream() as *mut c_void,
                raw_ptr as *const u16,
                vocab as i32,
                vp as *mut f32,
                ip as *mut i32,
                std::ptr::null(),
                tp as *mut u32,
                std::ptr::null_mut(),
                0,
            )
        };
        anyhow::ensure!(rc == 0, "device argmax rc={rc}");
    }
    if let Some(p) = prof {
        p.record(ProfPoint::End, s)?;
    }
    Ok(())
}

pub struct LagunaVerifyGraph<'m> {
    model: &'m Laguna,
    dev: CudaDevice,
    device: Device,
    forked: Arc<CudaStream>,
    aux_gemm: Arc<CudaStream>,
    aux_shared: Arc<CudaStream>,
    runner: CudaGraphRunner,
    t: usize,
    aux_layers: Vec<usize>,
    tok_buf: CudaSlice<u32>,
    host_toks: Vec<u32>,
    cu_q: CudaSlice<i32>,
    cu_full: CudaSlice<i32>,
    cu_slide: CudaSlice<i32>,
    logits_t: Tensor,
    aux_ts: Vec<Tensor>,
    meta_ptr: u64,
    layer_kv: Vec<Option<(Tensor, Tensor)>>,
    grouped: Vec<Option<Arc<MoeGroupedWeights>>>,
    moe_ctx: GroupedDecodeContext,
    s_cap: usize,
    max_seq_len: usize,
    captured: bool,
    prof: Option<StepProfStamps>,
    fp8: Option<Fp8VerifyState>,
    _err_drain: CtxErrDrain,
}

unsafe impl Send for LagunaVerifyGraph<'_> {}

enum VerifyCacheRef<'a> {
    Bf16(&'a LagunaKvCache),
    Fp8(&'a LagunaKvCacheFp8),
}

impl<'m> LagunaVerifyGraph<'m> {
    pub fn new(
        model: &'m Laguna,
        cache: &LagunaKvCache,
        t: usize,
        aux_layers: &[usize],
    ) -> Result<Self> {
        Self::new_inner(model, VerifyCacheRef::Bf16(cache), t, aux_layers)
    }

    pub fn new_fp8(
        model: &'m Laguna,
        cache: &LagunaKvCacheFp8,
        t: usize,
        aux_layers: &[usize],
    ) -> Result<Self> {
        Self::new_inner(model, VerifyCacheRef::Fp8(cache), t, aux_layers)
    }

    fn new_inner(
        model: &'m Laguna,
        cache: VerifyCacheRef<'_>,
        t: usize,
        aux_layers: &[usize],
    ) -> Result<Self> {
        let device = model.device().clone();
        let dev = match &device {
            Device::Cuda(d) => d.clone(),
            _ => anyhow::bail!("LagunaVerifyGraph requires a CUDA device"),
        };
        anyhow::ensure!(
            model.dtype() == DType::BF16,
            "LagunaVerifyGraph requires BF16"
        );
        anyhow::ensure!(t >= 1, "LagunaVerifyGraph: t must be >= 1");
        let (meta_ptr, s_cap, max_seq_len) = match &cache {
            VerifyCacheRef::Bf16(c) => {
                anyhow::ensure!(c.has_ring(), "LagunaVerifyGraph requires the ring KV cache");
                let meta_ptr = c
                    .ring_meta_ptr()
                    .ok_or_else(|| anyhow::anyhow!("ring meta missing"))?;
                (meta_ptr, c.sliding_cap().unwrap_or(0), c.max_seq_len())
            }
            VerifyCacheRef::Fp8(c) => (c.ring_meta_ptr(), c.sliding_cap(), c.max_seq_len()),
        };
        anyhow::ensure!(s_cap > 0, "LagunaVerifyGraph: no sliding capacity");

        let raw_ctx: Arc<CudaContext> = dev.cuda_stream().context().clone();
        let mut ctor_guard = crate::gemma4_batch_graph::graph_teardown::CtorForkGuard::new();
        let forked = ctor_guard
            .fork(&raw_ctx)
            .map_err(|e| anyhow::anyhow!("verify graph stream: {e:?}"))?;
        let aux_gemm = ctor_guard
            .fork(&raw_ctx)
            .map_err(|e| anyhow::anyhow!("verify aux gemm stream: {e:?}"))?;
        let aux_shared = ctor_guard
            .fork(&raw_ctx)
            .map_err(|e| anyhow::anyhow!("verify aux shared stream: {e:?}"))?;

        let cfg = model.config();
        let tok_buf = forked
            .alloc_zeros::<u32>(t)
            .map_err(|e| anyhow::anyhow!(e))?;
        #[allow(deprecated)]
        let cu_q = forked
            .memcpy_stod(&[0i32, t as i32])
            .map_err(|e| anyhow::anyhow!(e))?;
        let cu_full = forked
            .alloc_zeros::<i32>(2)
            .map_err(|e| anyhow::anyhow!(e))?;
        let cu_slide = forked
            .alloc_zeros::<i32>(2)
            .map_err(|e| anyhow::anyhow!(e))?;

        let logits_t = Tensor::zeros((1usize, t, cfg.vocab_size), DType::F32, &device)?;
        let mut aux_ts = Vec::with_capacity(aux_layers.len());
        for _ in aux_layers {
            aux_ts.push(Tensor::zeros(
                (1usize, t, cfg.hidden_size),
                DType::BF16,
                &device,
            )?);
        }

        let mut layer_kv = Vec::with_capacity(model.layers().len());
        let mut grouped = Vec::with_capacity(model.layers().len());
        let mut fp8_layers: Vec<Option<Fp8VerifyLayer>> = Vec::with_capacity(model.layers().len());
        for (li, layer) in model.layers().iter().enumerate() {
            match &cache {
                VerifyCacheRef::Bf16(c) => {
                    layer_kv.push(Some(c.layer_kv_bufs(li)));
                    fp8_layers.push(None);
                }
                VerifyCacheRef::Fp8(c) => {
                    layer_kv.push(c.layer_bf16_bufs(li));
                    fp8_layers.push(c.layer_fp8_ptrs(li).map(
                        |(k_fp8, v_fp8, k_scales, v_scales)| Fp8VerifyLayer {
                            k_fp8,
                            v_fp8,
                            k_scales,
                            v_scales,
                        },
                    ));
                    anyhow::ensure!(
                        layer_kv[li].is_some() != fp8_layers[li].is_some(),
                        "fp8 cache layer {li}: expected exactly one of bf16/fp8 storage"
                    );
                }
            }
            grouped.push(match &layer.ffn {
                LagunaFfn::Moe(moe) => Some(
                    model
                        .grouped_weights(moe)?
                        .ok_or_else(|| anyhow::anyhow!("grouped MoE weights unavailable"))?,
                ),
                LagunaFfn::Dense(_) => None,
            });
        }
        let fp8 = match &cache {
            VerifyCacheRef::Bf16(_) => None,
            VerifyCacheRef::Fp8(_) => {
                let scratch_elems = max_seq_len * cfg.num_key_value_heads * cfg.head_dim;
                let k_scratch = forked
                    .alloc_zeros::<bf16>(scratch_elems)
                    .map_err(|e| anyhow::anyhow!("fp8 verify k scratch: {e:?}"))?;
                let v_scratch = forked
                    .alloc_zeros::<bf16>(scratch_elems)
                    .map_err(|e| anyhow::anyhow!("fp8 verify v scratch: {e:?}"))?;
                Some(Fp8VerifyState {
                    layers: fp8_layers,
                    n_committed_ptr: meta_ptr,
                    k_scratch,
                    v_scratch,
                })
            }
        };
        let moe_ctx = GroupedDecodeContext::new_multi(
            cfg.hidden_size,
            cfg.moe_intermediate_size,
            cfg.num_experts_per_tok,
            cfg.num_experts,
            t,
            &forked,
        )?;
        forked.synchronize().map_err(|e| anyhow::anyhow!(e))?;

        let prof = if verify_prof_enabled() {
            let layer_info: Vec<(bool, bool)> = model
                .layers()
                .iter()
                .map(|l| {
                    (
                        matches!(l.self_attn.kind, LayerType::SlidingAttention),
                        matches!(l.ffn, LagunaFfn::Dense(_)),
                    )
                })
                .collect();
            Some(StepProfStamps::new(&forked, layer_info)?)
        } else {
            None
        };
        let runner = CudaGraphRunner::new(forked.clone());
        ctor_guard.the_built_engine_owns_teardown_now();
        Ok(Self {
            model,
            dev,
            device,
            forked,
            aux_gemm,
            aux_shared,
            runner,
            t,
            aux_layers: aux_layers.to_vec(),
            tok_buf,
            host_toks: vec![0u32; t],
            cu_q,
            cu_full,
            cu_slide,
            logits_t,
            aux_ts,
            meta_ptr,
            layer_kv,
            grouped,
            moe_ctx,
            s_cap,
            max_seq_len,
            captured: false,
            prof,
            fp8,
            _err_drain: CtxErrDrain(raw_ctx),
        })
    }

    pub(crate) fn drafts_device_ptr(&self, stream: &Arc<CudaStream>) -> (u64, usize) {
        let (p, _g) = self.tok_buf.device_ptr(stream);
        (
            p + std::mem::size_of::<u32>() as u64,
            self.t.saturating_sub(1),
        )
    }

    pub fn verify(
        &mut self,
        cache: &mut LagunaKvCache,
        tokens: &[u32],
    ) -> Result<(Tensor, Vec<Tensor>)> {
        anyhow::ensure!(
            self.fp8.is_none(),
            "verify: this graph was captured for the fp8 KV cache"
        );
        self.model.apply_attn_w8_for(self.t);
        anyhow::ensure!(tokens.len() == self.t, "verify: expected {} tokens", self.t);
        anyhow::ensure!(
            cache.ring_meta_ptr() == Some(self.meta_ptr),
            "verify: cache does not match the captured buffers"
        );
        let write_pos = cache.current_len();
        anyhow::ensure!(
            write_pos + self.t <= self.max_seq_len,
            "verify: KV capacity exceeded"
        );
        cache
            .prepare_for_decode(write_pos, write_pos + self.t)
            .context("verify graph prepare_for_decode")?;
        let out = self.run_captured(tokens)?;
        cache.advance(self.t);
        cache.note_graph_write();
        Ok(out)
    }

    pub fn verify_fp8(
        &mut self,
        cache: &mut LagunaKvCacheFp8,
        tokens: &[u32],
    ) -> Result<(Tensor, Vec<Tensor>)> {
        anyhow::ensure!(
            self.fp8.is_some(),
            "verify_fp8: this graph was captured for the bf16 ring cache"
        );
        self.model.apply_attn_w8_for(self.t);
        anyhow::ensure!(tokens.len() == self.t, "verify: expected {} tokens", self.t);
        anyhow::ensure!(
            cache.ring_meta_ptr() == self.meta_ptr,
            "verify_fp8: cache does not match the captured buffers"
        );
        let write_pos = LagunaKvCacheFp8::current_len(cache);
        anyhow::ensure!(
            write_pos + self.t <= self.max_seq_len,
            "verify: KV capacity exceeded"
        );
        cache
            .prepare_for_decode_dev(write_pos, write_pos + self.t)
            .context("fp8 verify graph prepare_for_decode")?;
        let out = self.run_captured(tokens)?;
        LagunaKvCacheFp8::advance(cache, self.t);
        cache.note_graph_write();
        Ok(out)
    }

    #[cfg(feature = "cuda")]
    pub(crate) fn verify_device(
        &mut self,
        cache: &mut LagunaKvCache,
        anchor: u32,
        drafts: &CudaSlice<u32>,
    ) -> Result<(Tensor, Vec<Tensor>)> {
        anyhow::ensure!(
            self.fp8.is_none(),
            "verify_device: this graph was captured for the fp8 KV cache"
        );
        self.model.apply_attn_w8_for(self.t);
        anyhow::ensure!(
            cache.ring_meta_ptr() == Some(self.meta_ptr),
            "verify_device: cache does not match the captured buffers"
        );
        let write_pos = cache.current_len();
        anyhow::ensure!(
            write_pos + self.t <= self.max_seq_len,
            "verify_device: KV capacity exceeded"
        );
        cache
            .prepare_for_decode(write_pos, write_pos + self.t)
            .context("verify_device prepare_for_decode")?;
        self.fill_tok_buf_device(anchor, drafts)?;
        let out = self.run_captured_inner()?;
        cache.advance(self.t);
        cache.note_graph_write();
        Ok(out)
    }

    #[cfg(feature = "cuda")]
    pub(crate) fn verify_fp8_device(
        &mut self,
        cache: &mut LagunaKvCacheFp8,
        anchor: u32,
        drafts: &CudaSlice<u32>,
    ) -> Result<(Tensor, Vec<Tensor>)> {
        anyhow::ensure!(
            self.fp8.is_some(),
            "verify_fp8_device: this graph was captured for the bf16 ring cache"
        );
        self.model.apply_attn_w8_for(self.t);
        anyhow::ensure!(
            cache.ring_meta_ptr() == self.meta_ptr,
            "verify_fp8_device: cache does not match the captured buffers"
        );
        let write_pos = LagunaKvCacheFp8::current_len(cache);
        anyhow::ensure!(
            write_pos + self.t <= self.max_seq_len,
            "verify_fp8_device: KV capacity exceeded"
        );
        cache
            .prepare_for_decode_dev(write_pos, write_pos + self.t)
            .context("verify_fp8_device prepare_for_decode")?;
        self.fill_tok_buf_device(anchor, drafts)?;
        let out = self.run_captured_inner()?;
        LagunaKvCacheFp8::advance(cache, self.t);
        cache.note_graph_write();
        Ok(out)
    }

    fn run_captured(&mut self, tokens: &[u32]) -> Result<(Tensor, Vec<Tensor>)> {
        self.host_toks.copy_from_slice(tokens);
        let legacy = self.dev.cuda_stream();
        legacy
            .memcpy_htod(&self.host_toks[..], &mut self.tok_buf)
            .map_err(|e| anyhow::anyhow!("htod verify tokens: {e:?}"))?;
        crate::laguna_dflash::decandle_stats::tick_htod();
        self.run_captured_inner()
    }

    #[cfg(feature = "cuda")]
    fn fill_tok_buf_device(&mut self, anchor: u32, drafts: &CudaSlice<u32>) -> Result<()> {
        anyhow::ensure!(
            drafts.len() >= self.t.saturating_sub(1),
            "fill_tok_buf_device: drafts len {} < t-1 {}",
            drafts.len(),
            self.t.saturating_sub(1)
        );
        let legacy = self.dev.cuda_stream();
        {
            let mut head = self.tok_buf.slice_mut(0..1);
            let anchor_host = [anchor];
            legacy
                .memcpy_htod(&anchor_host[..], &mut head)
                .map_err(|e| anyhow::anyhow!("htod verify anchor: {e:?}"))?;
        }
        crate::laguna_dflash::decandle_stats::tick_htod();
        let src = drafts.slice(0..self.t - 1);
        let mut dst = self.tok_buf.slice_mut(1..self.t);
        legacy
            .memcpy_dtod(&src, &mut dst)
            .map_err(|e| anyhow::anyhow!("dtod verify drafts: {e:?}"))?;
        Ok(())
    }

    fn run_captured_inner(&mut self) -> Result<(Tensor, Vec<Tensor>)> {
        let legacy = self.dev.cuda_stream();
        let raw_ctx = legacy.context().clone();
        if raw_ctx.is_event_tracking() {
            unsafe { raw_ctx.disable_event_tracking() };
            legacy
                .synchronize()
                .map_err(|e| anyhow::anyhow!("pre-capture legacy sync: {e:?}"))?;
        }

        let was_captured = self.captured;
        let forked = self.forked.clone();
        let logits_ptr = {
            let (ls, ll) = self.logits_t.storage_and_layout();
            let l_cuda = match &*ls {
                candle_core::Storage::Cuda(st) => st,
                _ => anyhow::bail!("verify logits not CUDA"),
            };
            let l_slice = l_cuda.as_cuda_slice::<f32>()?;
            let (p, _g) = l_slice.device_ptr(&forked);
            p + (ll.start_offset() * 4) as u64
        };
        let mut aux_dst = Vec::with_capacity(self.aux_ts.len());
        for a in &self.aux_ts {
            aux_dst.push(bf16_ptr(
                a,
                &forked,
                self.t * self.model.config().hidden_size,
            )?);
        }

        let LagunaVerifyGraph {
            model,
            dev,
            device,
            aux_gemm,
            aux_shared,
            runner,
            tok_buf,
            cu_q,
            cu_full,
            cu_slide,
            meta_ptr,
            layer_kv,
            grouped,
            moe_ctx,
            s_cap,
            max_seq_len,
            t,
            aux_layers,
            prof,
            fp8,
            ..
        } = self;
        let meta_ptr = *meta_ptr;
        let s_cap = *s_cap;
        let max_seq_len = *max_seq_len;
        let t = *t;
        let prof = prof.as_ref();
        let fp8 = fp8.as_ref();

        let mut body = |s: &Arc<CudaStream>, moe_ctx: &mut GroupedDecodeContext| -> Result<()> {
            step_body(
                model,
                dev,
                device,
                s,
                aux_gemm,
                aux_shared,
                tok_buf,
                cu_q,
                cu_full,
                cu_slide,
                logits_ptr,
                None,
                meta_ptr,
                layer_kv,
                grouped,
                moe_ctx,
                s_cap,
                max_seq_len,
                t,
                aux_layers,
                &aux_dst,
                prof,
                None,
                fp8,
                true,
            )
        };

        if !was_captured {
            legacy
                .synchronize()
                .map_err(|e| anyhow::anyhow!("pre-warm legacy sync: {e:?}"))?;
            nv_layers::cuda_stream::with_stream(forked.clone(), || body(&forked, moe_ctx))
                .context("verify graph warm pass")?;
            forked
                .synchronize()
                .map_err(|e| anyhow::anyhow!("warm sync: {e:?}"))?;
        }

        runner
            .run_on(t as u64, Some(&legacy), |s| {
                nv_layers::cuda_stream::with_stream(s.clone(), || body(s, moe_ctx))
            })
            .context("verify graph capture/replay")?;
        if !was_captured {
            forked
                .synchronize()
                .map_err(|e| anyhow::anyhow!("post-capture sync: {e:?}"))?;
            self.captured = true;
        }
        if let Some(p) = self.prof.as_ref() {
            self.dev
                .cuda_stream()
                .synchronize()
                .map_err(|e| anyhow::anyhow!("prof legacy sync: {e:?}"))?;
            p.report(&format!("verify_t{}", self.t))?;
        }

        Ok((self.logits_t.clone(), self.aux_ts.clone()))
    }
}

impl Drop for LagunaVerifyGraph<'_> {
    fn drop(&mut self) {
        crate::gemma4_batch_graph::graph_teardown::GraphTeardown::new(&self.forked)
            .with_stream(&self.aux_gemm)
            .with_stream(&self.aux_shared)
            .run(|| self.runner.invalidate());
    }
}
