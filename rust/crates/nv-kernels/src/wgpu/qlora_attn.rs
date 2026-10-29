use crate::wgpu_backend::device::WgpuContext;
use crate::wgpu_backend::{dispatch, Result, WgpuError};

pub const QLORA_GROUPED_ATTN_VIEWS_Q_AS_GROUP_MAJOR_SO_DK_DV_FALL_OUT_OF_MATMUL_BACKWARD: &str =
    "the grouped-GQA formulation from nv_layers::attn's NV_ATTN_GQA_GROUPED arm: q is viewed \
     (B*H_kv, factor*sq, d) against the unexpanded (B*H_kv, sk, d) k and v, head h = g*factor+f \
     landing at group row f*sq+i, so no rank-5 expand is ever materialized and dK/dV are plain \
     matmul backward outputs (dV = P^T dO, dK = dS^T Q) instead of an expand-backward strided \
     reduce. The causal mask tiles to (factor*sq, sk) with group row r masking query position \
     r % sq at key offset sk-sq. Every reduction here is a sequential per-thread strided loop \
     plus a fixed binary tree over a 256-slot workgroup array, so reruns of the same binary on \
     the same device are bit-identical.";

pub const QLORA_GROUPED_ATTN_BWD_REASSOCIATES_AT_THE_1E9_CLASS_VS_EXPAND_SO_IT_GATES_ON_TOLERANCE:
    &str =
    "the grouped backward contracts dK and dV over factor*sq rows in one chain where the expand \
     path sums factor separate sq-row products, so the two arms differ at the reassociation \
     class (~1e-9 relative on f32 inputs of unit scale) BY CONSTRUCTION; parity gates compare \
     against a reference at relative tolerance and record the observed deltas, never bitwise, \
     and the arm must not be composed silently with bitwise-claimed arms.";

pub const WORKGROUP_SIZE: u32 = 256;

pub const MASKED_SCORE_IS_FINITE_NEG3E38_SO_EXP_UNDERFLOWS_TO_ZERO_WITHOUT_INF_ARITHMETIC: f32 =
    -3.0e38;

pub const WGSL: &str = include_str!("../../wgsl/qlora_attn.wgsl");

#[derive(Clone, Copy, Debug)]
pub struct GroupedAttnDims {
    pub groups: usize,
    pub factor: usize,
    pub sq: usize,
    pub sk: usize,
    pub head_dim: usize,
    pub scale: f32,
    pub causal: bool,
    pub window: usize,
}

impl GroupedAttnDims {
    pub fn m(&self) -> usize {
        self.factor * self.sq
    }

    pub fn q_len(&self) -> usize {
        self.groups * self.m() * self.head_dim
    }

    pub fn kv_len(&self) -> usize {
        self.groups * self.sk * self.head_dim
    }

    pub fn probs_len(&self) -> usize {
        self.groups * self.m() * self.sk
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct Params {
    g: u32,
    m: u32,
    sk: u32,
    d: u32,
    sq: u32,
    causal: u32,
    window: u32,
    scale: f32,
}

fn params_for(dims: &GroupedAttnDims) -> Result<Params> {
    for (what, v) in [
        ("groups", dims.groups),
        ("m", dims.m()),
        ("sk", dims.sk),
        ("head_dim", dims.head_dim),
        ("sq", dims.sq),
        ("window", dims.window),
    ] {
        if v > u32::MAX as usize {
            return Err(WgpuError::Shape(format!(
                "grouped attn {what} {v} exceeds the u32 index range"
            )));
        }
    }
    if dims.groups == 0 || dims.sq == 0 || dims.sk == 0 || dims.head_dim == 0 || dims.factor == 0 {
        return Err(WgpuError::Shape(format!(
            "grouped attn dims must be nonzero, got {dims:?}"
        )));
    }
    Ok(Params {
        g: dims.groups as u32,
        m: dims.m() as u32,
        sk: dims.sk as u32,
        d: dims.head_dim as u32,
        sq: dims.sq as u32,
        causal: dims.causal as u32,
        window: dims.window as u32,
        scale: dims.scale,
    })
}

fn elementwise_groups(ctx: &WgpuContext, what: &str, total: usize) -> Result<(u32, u32, u32)> {
    let wgs = (total as u64).div_ceil(WORKGROUP_SIZE as u64).max(1);
    if wgs > ctx.caps.max_compute_workgroups_per_dimension as u64 {
        return Err(WgpuError::Shape(format!(
            "{what}: {total} elements need {wgs} workgroups, device dimension limit is {}",
            ctx.caps.max_compute_workgroups_per_dimension
        )));
    }
    Ok((wgs as u32, 1, 1))
}

fn row_groups(ctx: &WgpuContext, what: &str, rows: usize) -> Result<(u32, u32, u32)> {
    if rows as u64 > ctx.caps.max_compute_workgroups_per_dimension as u64 {
        return Err(WgpuError::Shape(format!(
            "{what}: {rows} rows exceed the device workgroup dimension limit {}",
            ctx.caps.max_compute_workgroups_per_dimension
        )));
    }
    Ok((rows as u32, 1, 1))
}

pub struct GroupedAttnFwd {
    dims: GroupedAttnDims,
    params: wgpu::Buffer,
    q: wgpu::Buffer,
    k: wgpu::Buffer,
    v: wgpu::Buffer,
    probs: wgpu::Buffer,
    pub out: Vec<f32>,
}

pub fn grouped_attn_fwd(
    ctx: &WgpuContext,
    q_grouped: &[f32],
    k_grouped: &[f32],
    v_grouped: &[f32],
    dims: &GroupedAttnDims,
) -> Result<GroupedAttnFwd> {
    dispatch::drive_blocking(grouped_attn_fwd_arm(
        ctx,
        q_grouped,
        k_grouped,
        v_grouped,
        dims,
        dispatch::ReadbackArm::Blocking,
    ))
}

pub async fn grouped_attn_fwd_arm(
    ctx: &WgpuContext,
    q_grouped: &[f32],
    k_grouped: &[f32],
    v_grouped: &[f32],
    dims: &GroupedAttnDims,
    arm: dispatch::ReadbackArm,
) -> Result<GroupedAttnFwd> {
    let _ = QLORA_GROUPED_ATTN_VIEWS_Q_AS_GROUP_MAJOR_SO_DK_DV_FALL_OUT_OF_MATMUL_BACKWARD;
    dispatch::check_len("grouped attn q", q_grouped.len(), dims.q_len())?;
    dispatch::check_len("grouped attn k", k_grouped.len(), dims.kv_len())?;
    dispatch::check_len("grouped attn v", v_grouped.len(), dims.kv_len())?;
    dispatch::require_workgroup_and_scratch(ctx, "grouped attn", WORKGROUP_SIZE, 256 * 4)?;
    let params = params_for(dims)?;
    let pb = dispatch::uniform_from(ctx, "qla-params", &params);
    let qb = dispatch::storage_from_slice(ctx, "qla-q", q_grouped);
    let kb = dispatch::storage_from_slice(ctx, "qla-k", k_grouped);
    let vb = dispatch::storage_from_slice(ctx, "qla-v", v_grouped);
    let sp = dispatch::storage_zeroed(ctx, "qla-probs", (dims.probs_len() * 4) as u64);
    let ob = dispatch::storage_zeroed(ctx, "qla-out", (dims.q_len() * 4) as u64);

    dispatch::run(
        ctx,
        "qla-scores",
        WGSL,
        "qla_scores",
        &[(0, &pb), (1, &qb), (2, &kb), (4, &sp)],
        elementwise_groups(ctx, "qla_scores", dims.probs_len())?,
    )?;
    dispatch::run(
        ctx,
        "qla-softmax",
        WGSL,
        "qla_softmax",
        &[(0, &pb), (4, &sp)],
        row_groups(ctx, "qla_softmax", dims.groups * dims.m())?,
    )?;
    dispatch::run(
        ctx,
        "qla-out",
        WGSL,
        "qla_out",
        &[(0, &pb), (3, &vb), (4, &sp), (5, &ob)],
        elementwise_groups(ctx, "qla_out", dims.q_len())?,
    )?;
    let out: Vec<f32> = dispatch::read_back_arm(ctx, &ob, dims.q_len(), arm).await?;
    Ok(GroupedAttnFwd {
        dims: *dims,
        params: pb,
        q: qb,
        k: kb,
        v: vb,
        probs: sp,
        out,
    })
}

pub struct GroupedAttnGrads {
    pub dq: Vec<f32>,
    pub dk: Vec<f32>,
    pub dv: Vec<f32>,
}

pub fn grouped_attn_bwd(
    ctx: &WgpuContext,
    fwd: &GroupedAttnFwd,
    dout: &[f32],
) -> Result<GroupedAttnGrads> {
    dispatch::drive_blocking(grouped_attn_bwd_arm(
        ctx,
        fwd,
        dout,
        dispatch::ReadbackArm::Blocking,
    ))
}

pub async fn grouped_attn_bwd_arm(
    ctx: &WgpuContext,
    fwd: &GroupedAttnFwd,
    dout: &[f32],
    arm: dispatch::ReadbackArm,
) -> Result<GroupedAttnGrads> {
    let _ = QLORA_GROUPED_ATTN_BWD_REASSOCIATES_AT_THE_1E9_CLASS_VS_EXPAND_SO_IT_GATES_ON_TOLERANCE;
    let dims = &fwd.dims;
    dispatch::check_len("grouped attn dout", dout.len(), dims.q_len())?;
    let dob = dispatch::storage_from_slice(ctx, "qla-dout", dout);
    let dsb = dispatch::storage_zeroed(ctx, "qla-ds", (dims.probs_len() * 4) as u64);
    let drb = dispatch::storage_zeroed(ctx, "qla-drow", (dims.groups * dims.m() * 4) as u64);
    let dqb = dispatch::storage_zeroed(ctx, "qla-dq", (dims.q_len() * 4) as u64);
    let dkb = dispatch::storage_zeroed(ctx, "qla-dk", (dims.kv_len() * 4) as u64);
    let dvb = dispatch::storage_zeroed(ctx, "qla-dv", (dims.kv_len() * 4) as u64);
    let pb = &fwd.params;

    dispatch::run(
        ctx,
        "qla-dv",
        WGSL,
        "qla_dv",
        &[(0, pb), (4, &fwd.probs), (6, &dob), (11, &dvb)],
        elementwise_groups(ctx, "qla_dv", dims.kv_len())?,
    )?;
    dispatch::run(
        ctx,
        "qla-dp",
        WGSL,
        "qla_dp",
        &[(0, pb), (3, &fwd.v), (6, &dob), (7, &dsb)],
        elementwise_groups(ctx, "qla_dp", dims.probs_len())?,
    )?;
    dispatch::run(
        ctx,
        "qla-drow",
        WGSL,
        "qla_drow",
        &[(0, pb), (4, &fwd.probs), (7, &dsb), (8, &drb)],
        row_groups(ctx, "qla_drow", dims.groups * dims.m())?,
    )?;
    dispatch::run(
        ctx,
        "qla-ds",
        WGSL,
        "qla_ds",
        &[(0, pb), (4, &fwd.probs), (7, &dsb), (8, &drb)],
        elementwise_groups(ctx, "qla_ds", dims.probs_len())?,
    )?;
    dispatch::run(
        ctx,
        "qla-dq",
        WGSL,
        "qla_dq",
        &[(0, pb), (2, &fwd.k), (7, &dsb), (9, &dqb)],
        elementwise_groups(ctx, "qla_dq", dims.q_len())?,
    )?;
    dispatch::run(
        ctx,
        "qla-dk",
        WGSL,
        "qla_dk",
        &[(0, pb), (1, &fwd.q), (7, &dsb), (10, &dkb)],
        elementwise_groups(ctx, "qla_dk", dims.kv_len())?,
    )?;
    Ok(GroupedAttnGrads {
        dq: dispatch::read_back_arm(ctx, &dqb, dims.q_len(), arm).await?,
        dk: dispatch::read_back_arm(ctx, &dkb, dims.kv_len(), arm).await?,
        dv: dispatch::read_back_arm(ctx, &dvb, dims.kv_len(), arm).await?,
    })
}

pub fn q_bshd_to_grouped(
    q: &[f32],
    b: usize,
    sq: usize,
    h: usize,
    h_kv: usize,
    d: usize,
) -> Vec<f32> {
    assert_eq!(q.len(), b * sq * h * d, "q must be (B, sq, H, d)");
    assert_eq!(h % h_kv, 0, "H must be divisible by H_kv");
    let factor = h / h_kv;
    let mut out = vec![0f32; q.len()];
    for bi in 0..b {
        for i in 0..sq {
            for hh in 0..h {
                let g = hh / factor;
                let f = hh % factor;
                let src = ((bi * sq + i) * h + hh) * d;
                let dst = (((bi * h_kv + g) * factor + f) * sq + i) * d;
                out[dst..dst + d].copy_from_slice(&q[src..src + d]);
            }
        }
    }
    out
}

pub fn grouped_to_q_bshd(
    grouped: &[f32],
    b: usize,
    sq: usize,
    h: usize,
    h_kv: usize,
    d: usize,
) -> Vec<f32> {
    assert_eq!(
        grouped.len(),
        b * sq * h * d,
        "grouped must be (B*H_kv, factor*sq, d)"
    );
    let factor = h / h_kv;
    let mut out = vec![0f32; grouped.len()];
    for bi in 0..b {
        for i in 0..sq {
            for hh in 0..h {
                let g = hh / factor;
                let f = hh % factor;
                let src = (((bi * h_kv + g) * factor + f) * sq + i) * d;
                let dst = ((bi * sq + i) * h + hh) * d;
                out[dst..dst + d].copy_from_slice(&grouped[src..src + d]);
            }
        }
    }
    out
}

pub fn kv_bskd_to_grouped(kv: &[f32], b: usize, sk: usize, h_kv: usize, d: usize) -> Vec<f32> {
    assert_eq!(kv.len(), b * sk * h_kv * d, "kv must be (B, sk, H_kv, d)");
    let mut out = vec![0f32; kv.len()];
    for bi in 0..b {
        for j in 0..sk {
            for g in 0..h_kv {
                let src = ((bi * sk + j) * h_kv + g) * d;
                let dst = ((bi * h_kv + g) * sk + j) * d;
                out[dst..dst + d].copy_from_slice(&kv[src..src + d]);
            }
        }
    }
    out
}

pub fn grouped_to_kv_bskd(grouped: &[f32], b: usize, sk: usize, h_kv: usize, d: usize) -> Vec<f32> {
    assert_eq!(
        grouped.len(),
        b * sk * h_kv * d,
        "grouped must be (B*H_kv, sk, d)"
    );
    let mut out = vec![0f32; grouped.len()];
    for bi in 0..b {
        for j in 0..sk {
            for g in 0..h_kv {
                let src = ((bi * h_kv + g) * sk + j) * d;
                let dst = ((bi * sk + j) * h_kv + g) * d;
                out[dst..dst + d].copy_from_slice(&grouped[src..src + d]);
            }
        }
    }
    out
}
