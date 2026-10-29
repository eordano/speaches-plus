use anyhow::{ensure, Result};
use candle_core::backend::BackendStorage;
use candle_core::{DType, Device, Layout, MetalStorage, Shape, Tensor};
use candle_metal_kernels::metal::{Buffer, ComputePipeline};
use nv_layers::rope::Rope;
use objc2_metal::{MTLCompileOptions, MTLMathFloatingPointFunctions, MTLMathMode, MTLSize};
use std::sync::{Arc, Mutex, OnceLock};

pub const ROPE_FUSED_REPLACES_18_CANDLE_OPS_AND_A_HOST_ROUNDTRIP_PER_LAYER_WITH_TWO_DISPATCHES:
    &str =
    "NV_DOTS_ROPE_FUSED=1 serves the training rope for q and k as one MSL dispatch each with a \
     custom one-dispatch backward, replacing per tensor: a positions host round-trip that \
     fences the Metal queue, two index_selects, four broadcast_muls, a sub, an add and a cat \
     forward, plus their ~11-op autograd chain backward. The kernel module compiles under \
     MTLMathMode::Safe so no fma contraction reorders the arithmetic: forward computes \
     fl(fl(lo*c)-fl(hi*s)) / fl(fl(lo*s)+fl(hi*c)) and backward fl(fl(gh*s)+fl(gl*c)) / \
     fl(fl(gh*c)-fl(gl*s)), the exact rounding sequence and GradStore accumulation order of \
     the candle chain it replaces, so the arm gates on the canonical 20-step trajectory \
     reproducing bit-identically";

const MSL: &str = r#"
#include <metal_stdlib>
using namespace metal;

struct RP {
    uint tokens;
    uint heads;
    uint half_dim;
    uint max_seq;
};

kernel void dots_rope_fwd(
    device const float* x [[buffer(0)]],
    device const uint* pos [[buffer(1)]],
    device const float* cosb [[buffer(2)]],
    device const float* sinb [[buffer(3)]],
    device float* out [[buffer(4)]],
    constant RP& p [[buffer(5)]],
    uint gid [[thread_position_in_grid]])
{
    const uint total = p.tokens * p.heads * p.half_dim;
    if (gid >= total) return;
    const uint i = gid % p.half_dim;
    const uint th = gid / p.half_dim;
    const uint t = th / p.heads;
    const uint pr = min(pos[t], p.max_seq - 1);
    const ulong base = (ulong)th * (2 * p.half_dim);
    const float lo = x[base + i];
    const float hi = x[base + p.half_dim + i];
    const float c = cosb[pr * p.half_dim + i];
    const float s = sinb[pr * p.half_dim + i];
    const float m1 = lo * c;
    const float m2 = hi * s;
    const float m3 = lo * s;
    const float m4 = hi * c;
    out[base + i] = m1 - m2;
    out[base + p.half_dim + i] = m3 + m4;
}

kernel void dots_rope_bwd(
    device const float* g [[buffer(0)]],
    device const uint* pos [[buffer(1)]],
    device const float* cosb [[buffer(2)]],
    device const float* sinb [[buffer(3)]],
    device float* dx [[buffer(4)]],
    constant RP& p [[buffer(5)]],
    uint gid [[thread_position_in_grid]])
{
    const uint total = p.tokens * p.heads * p.half_dim;
    if (gid >= total) return;
    const uint i = gid % p.half_dim;
    const uint th = gid / p.half_dim;
    const uint t = th / p.heads;
    const uint pr = min(pos[t], p.max_seq - 1);
    const ulong base = (ulong)th * (2 * p.half_dim);
    const float gl = g[base + i];
    const float gh = g[base + p.half_dim + i];
    const float c = cosb[pr * p.half_dim + i];
    const float s = sinb[pr * p.half_dim + i];
    const float a1 = gh * s;
    const float a2 = gl * c;
    const float b1 = gh * c;
    const float b2 = gl * s;
    dx[base + i] = a1 + a2;
    dx[base + p.half_dim + i] = b1 - b2;
}
"#;

#[repr(C)]
#[derive(Clone, Copy)]
struct RParams {
    tokens: u32,
    heads: u32,
    half_dim: u32,
    max_seq: u32,
}

struct Pipes {
    fwd: ComputePipeline,
    bwd: ComputePipeline,
}

fn cerr(e: impl std::fmt::Display) -> candle_core::Error {
    candle_core::Error::Msg(format!("dots_rope: {e}"))
}

pub struct DotsRopeFused {
    cos: Tensor,
    sin: Tensor,
    half: usize,
    max_seq: usize,
    pipes: OnceLock<Pipes>,
}

impl DotsRopeFused {
    fn pipes(&self, dev: &candle_core::MetalDevice) -> candle_core::Result<&Pipes> {
        if self.pipes.get().is_none() {
            let opts = MTLCompileOptions::new();
            opts.setMathMode(MTLMathMode::Safe);
            opts.setMathFloatingPointFunctions(MTLMathFloatingPointFunctions::Precise);
            let lib = dev
                .metal_device()
                .new_library_with_source(MSL, Some(&opts))
                .map_err(cerr)?;
            let mk = |name: &str| -> candle_core::Result<ComputePipeline> {
                let f = lib.get_function(name, None).map_err(cerr)?;
                dev.metal_device()
                    .new_compute_pipeline_state_with_function(&f)
                    .map_err(cerr)
            };
            let pipes = Pipes {
                fwd: mk("dots_rope_fwd")?,
                bwd: mk("dots_rope_bwd")?,
            };
            let _ = self.pipes.set(pipes);
        }
        Ok(self.pipes.get().expect("pipes were just set"))
    }
}

pub fn rope_fused_requested_via_env() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var("NV_DOTS_ROPE_FUSED").as_deref() == Ok("1"))
}

pub fn tables_for(rope: &Rope) -> Result<Arc<DotsRopeFused>> {
    static REGISTRY: Mutex<Vec<(candle_core::TensorId, Arc<DotsRopeFused>)>> =
        Mutex::new(Vec::new());
    let cos = rope.cos();
    let mut reg = REGISTRY
        .lock()
        .map_err(|e| anyhow::anyhow!("rope table registry poisoned: {e}"))?;
    if let Some((_, t)) = reg.iter().find(|(id, _)| *id == cos.id()) {
        return Ok(t.clone());
    }
    ensure!(
        cos.device().is_metal() && cos.dtype() == DType::F32,
        "fused rope tables need f32 Metal-resident cos/sin"
    );
    let dims = cos.dims().to_vec();
    ensure!(dims.len() == 2, "rope cos table must be [max_seq, half]");
    let t = Arc::new(DotsRopeFused {
        cos: cos.contiguous()?,
        sin: rope.sin().contiguous()?,
        half: dims[1],
        max_seq: dims[0],
        pipes: OnceLock::new(),
    });
    reg.push((cos.id(), t.clone()));
    Ok(t)
}

fn metal_buf_and_off(t: &Tensor, what: &'static str) -> candle_core::Result<(Buffer, usize)> {
    let (storage, layout) = t.storage_and_layout();
    let s = match &*storage {
        candle_core::Storage::Metal(s) => s,
        _ => return Err(cerr(format!("{what} must be Metal-resident"))),
    };
    let (start, _end) = layout
        .contiguous_offsets()
        .ok_or_else(|| cerr(format!("{what} must be contiguous")))?;
    Ok((s.buffer().clone(), start * t.dtype().size_in_bytes()))
}

fn encode_rope(
    tables: &DotsRopeFused,
    pos: &Tensor,
    dev: &candle_core::MetalDevice,
    pipe_is_bwd: bool,
    x_buf: &Buffer,
    x_off: usize,
    tokens: usize,
    heads: usize,
) -> candle_core::Result<Arc<Buffer>> {
    let pipes = tables.pipes(dev)?;
    let n = tokens * heads * 2 * tables.half;
    let out = dev
        .new_buffer_builder()
        .with_size(n * DType::F32.size_in_bytes())
        .with_label(if pipe_is_bwd {
            "dots-rope-dx"
        } else {
            "dots-rope-out"
        })
        .build()?;
    let (pb, pb_off) = metal_buf_and_off(pos, "rope positions")?;
    let (cb, cb_off) = metal_buf_and_off(&tables.cos, "rope cos")?;
    let (sb, sb_off) = metal_buf_and_off(&tables.sin, "rope sin")?;
    let p = RParams {
        tokens: tokens as u32,
        heads: heads as u32,
        half_dim: tables.half as u32,
        max_seq: tables.max_seq as u32,
    };
    let enc = dev.command_encoder()?;
    enc.set_compute_pipeline_state(if pipe_is_bwd { &pipes.bwd } else { &pipes.fwd });
    let e = enc.as_ref();
    e.set_input_buffer(0, Some(x_buf), x_off);
    e.set_input_buffer(1, Some(&pb), pb_off);
    e.set_input_buffer(2, Some(&cb), cb_off);
    e.set_input_buffer(3, Some(&sb), sb_off);
    e.set_output_buffer(4, Some(&*out), 0);
    e.set_bytes(5, &p);
    let total = tokens * heads * tables.half;
    e.dispatch_thread_groups(
        MTLSize {
            width: total.div_ceil(256),
            height: 1,
            depth: 1,
        },
        MTLSize {
            width: 256,
            height: 1,
            depth: 1,
        },
    );
    Ok(out)
}

fn rope_dims(layout: &Layout, half: usize) -> candle_core::Result<(usize, usize)> {
    let dims = layout.shape().dims();
    if dims.len() < 2 || dims[dims.len() - 1] != 2 * half {
        return Err(cerr(format!(
            "fused rope input {dims:?} must end in head_dim {}",
            2 * half
        )));
    }
    let heads = dims[dims.len() - 2];
    let tokens: usize = dims[..dims.len() - 2].iter().product();
    Ok((tokens, heads))
}

struct RopeBwdOp {
    tables: Arc<DotsRopeFused>,
    pos: Tensor,
}

impl candle_core::CustomOp1 for RopeBwdOp {
    fn name(&self) -> &'static str {
        "nv-dots-rope-fused-backward"
    }

    fn cpu_fwd(
        &self,
        _storage: &candle_core::CpuStorage,
        _layout: &Layout,
    ) -> candle_core::Result<(candle_core::CpuStorage, Shape)> {
        Err(cerr("the fused rope is Metal-only"))
    }

    fn metal_fwd(
        &self,
        storage: &MetalStorage,
        layout: &Layout,
    ) -> candle_core::Result<(MetalStorage, Shape)> {
        let (tokens, heads) = rope_dims(layout, self.tables.half)?;
        let (start, _end) = layout
            .contiguous_offsets()
            .ok_or_else(|| cerr("fused rope backward needs a contiguous grad"))?;
        if storage.dtype() != DType::F32 {
            return Err(cerr("fused rope backward needs an f32 grad"));
        }
        let dev = storage.device().clone();
        let out = encode_rope(
            &self.tables,
            &self.pos,
            &dev,
            true,
            storage.buffer(),
            start * 4,
            tokens,
            heads,
        )?;
        Ok((
            MetalStorage::new(out, dev, layout.shape().elem_count(), DType::F32),
            layout.shape().clone(),
        ))
    }
}

pub struct FusedRopeOp {
    tables: Arc<DotsRopeFused>,
    pos: Tensor,
}

impl candle_core::CustomOp1 for FusedRopeOp {
    fn name(&self) -> &'static str {
        "nv-dots-rope-fused-forward-with-single-dispatch-backward"
    }

    fn cpu_fwd(
        &self,
        _storage: &candle_core::CpuStorage,
        _layout: &Layout,
    ) -> candle_core::Result<(candle_core::CpuStorage, Shape)> {
        Err(cerr("the fused rope is Metal-only"))
    }

    fn metal_fwd(
        &self,
        storage: &MetalStorage,
        layout: &Layout,
    ) -> candle_core::Result<(MetalStorage, Shape)> {
        let (tokens, heads) = rope_dims(layout, self.tables.half)?;
        let (start, _end) = layout
            .contiguous_offsets()
            .ok_or_else(|| cerr("fused rope needs a contiguous input"))?;
        if storage.dtype() != DType::F32 {
            return Err(cerr("fused rope needs an f32 input"));
        }
        if self.pos.dims() != [tokens] {
            return Err(cerr(format!(
                "positions {:?} != [{tokens}]",
                self.pos.dims()
            )));
        }
        let dev = storage.device().clone();
        let out = encode_rope(
            &self.tables,
            &self.pos,
            &dev,
            false,
            storage.buffer(),
            start * 4,
            tokens,
            heads,
        )?;
        Ok((
            MetalStorage::new(out, dev, layout.shape().elem_count(), DType::F32),
            layout.shape().clone(),
        ))
    }

    fn bwd(
        &self,
        _arg: &Tensor,
        _res: &Tensor,
        grad_res: &Tensor,
    ) -> candle_core::Result<Option<Tensor>> {
        let g = grad_res.contiguous()?;
        Ok(Some(g.apply_op1_no_bwd(&RopeBwdOp {
            tables: self.tables.clone(),
            pos: self.pos.clone(),
        })?))
    }
}

pub fn apply_fused(tables: &Arc<DotsRopeFused>, x: &Tensor, positions: &Tensor) -> Result<Tensor> {
    let _ = ROPE_FUSED_REPLACES_18_CANDLE_OPS_AND_A_HOST_ROUNDTRIP_PER_LAYER_WITH_TWO_DISPATCHES;
    ensure!(
        positions.dtype() == DType::U32 && positions.device().same_device(x.device()),
        "fused rope positions must be u32 on the compute device"
    );
    Ok(x.contiguous()?.apply_op1(FusedRopeOp {
        tables: tables.clone(),
        pos: positions.contiguous()?,
    })?)
}

pub fn fused_qk(
    rope: &Rope,
    q: &Tensor,
    k: &Tensor,
    positions: &Tensor,
) -> Result<Option<(Tensor, Tensor)>> {
    if !rope_fused_requested_via_env()
        || !q.device().is_metal()
        || q.dtype() != DType::F32
        || k.dtype() != DType::F32
        || positions.dtype() != DType::U32
        || !positions.device().same_device(q.device())
    {
        return Ok(None);
    }
    let tables = tables_for(rope)?;
    if q.dims()[q.dims().len() - 1] != 2 * tables.half {
        return Ok(None);
    }
    let qr = apply_fused(&tables, q, positions)?;
    let kr = apply_fused(&tables, k, positions)?;
    Ok(Some((qr, kr)))
}
