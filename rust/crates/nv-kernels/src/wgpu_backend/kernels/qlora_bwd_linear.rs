use crate::wgpu_backend::device::WgpuContext;
use crate::wgpu_backend::{compose, dispatch, Result, WgpuError};

pub const WORKGROUP_SIZE: u32 = 256;
pub const W4_GROUP: u32 = 64;

pub const BWD_PRODUCES_ONLY_DX_AND_LORA_FACTOR_GRADS_THE_W4_BASE_STAYS_FROZEN: &str =
    "qlora backward through a hooked linear needs dX = dY @ W read straight from the \
     affine-u4-gs64 bytes (nv_quant::mlx_affine layout: 8 LSB-first nibbles per u32 along k, \
     one bf16-rounded f32 scale and bias per 64-element group, w = q*s + b), plus the four \
     small fp32 GEMMs of the LoRA chain. A and B stay in their stored [r,k] and [n,r] \
     layouts and are addressed transposed in-kernel, the NV_DOTS_LORA_TVIEW convention, so \
     no host transpose copies exist on the backward path either";

pub const SUM_ORDER_IS_FORCED_BY_EXPLICIT_FMA_SO_PARITY_GATES_ARE_BITWISE: &str =
    "every reduction here is one thread walking the inner dimension in index order through \
     an explicit fma chain, and fma is a single correctly-rounded operation Metal's default \
     fast math cannot contract further, so a CPU reference using f32::mul_add in the same \
     index order reproduces every output bit; the parity gates assert exact f32 equality, \
     and a mismatch means a compiler started reassociating the accumulator";

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, bytemuck::Pod, bytemuck::Zeroable)]
struct DxParams {
    m_rows: u32,
    n_rows: u32,
    k_cols: u32,
    w_row_words: u32,
    scale_stride: u32,
    total: u32,
    pad0: u32,
    pad1: u32,
}

const DX_WGSL: &str = include_str!("../../../wgsl/qlora_bwd_linear_dx.wgsl");

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, bytemuck::Pod, bytemuck::Zeroable)]
struct GemmParams {
    m_rows: u32,
    i_cols: u32,
    j_cols: u32,
    total: u32,
}

const GEMM_WGSL: &str = include_str!("../../../wgsl/qlora_bwd_linear_gemm.wgsl");

fn check_device(ctx: &WgpuContext) -> Result<()> {
    dispatch::require_workgroup(ctx, "qlora_bwd_linear", WORKGROUP_SIZE)
}

pub fn dx_w4_f32(
    ctx: &WgpuContext,
    dy: &[f32],
    w_words: &[u32],
    scales: &[f32],
    biases: &[f32],
    dx: &mut [f32],
    m: usize,
    n: usize,
    k: usize,
) -> Result<()> {
    dispatch::drive_blocking(dx_w4_f32_arm(
        ctx,
        dy,
        w_words,
        scales,
        biases,
        dx,
        m,
        n,
        k,
        dispatch::ReadbackArm::Blocking,
    ))
}

pub async fn dx_w4_f32_arm(
    ctx: &WgpuContext,
    dy: &[f32],
    w_words: &[u32],
    scales: &[f32],
    biases: &[f32],
    dx: &mut [f32],
    m: usize,
    n: usize,
    k: usize,
    arm: dispatch::ReadbackArm,
) -> Result<()> {
    let _ = BWD_PRODUCES_ONLY_DX_AND_LORA_FACTOR_GRADS_THE_W4_BASE_STAYS_FROZEN;
    if !k.is_multiple_of(W4_GROUP as usize) {
        return Err(WgpuError::Shape(format!(
            "dx_w4 dequant folds one scale+bias per {W4_GROUP}-element group; k={k} is not a multiple"
        )));
    }
    let groups = k / W4_GROUP as usize;
    dispatch::check_len("dx_w4 dy", dy.len(), m * n)?;
    dispatch::check_len("dx_w4 w_words", w_words.len(), n * k / 8)?;
    dispatch::check_len("dx_w4 scales", scales.len(), n * groups)?;
    dispatch::check_len("dx_w4 biases", biases.len(), n * groups)?;
    dispatch::check_len("dx_w4 dx", dx.len(), m * k)?;
    if m * k == 0 {
        return Ok(());
    }
    check_device(ctx)?;

    let params = DxParams {
        m_rows: m as u32,
        n_rows: n as u32,
        k_cols: k as u32,
        w_row_words: (k / 8) as u32,
        scale_stride: groups as u32,
        total: (m * k) as u32,
        pad0: 0,
        pad1: 0,
    };
    let dyb = dispatch::storage_from_slice(ctx, "qbl-dx-dy", dy);
    let wb = dispatch::storage_from_slice(ctx, "qbl-dx-w", w_words);
    let sb = dispatch::storage_from_slice(ctx, "qbl-dx-s", scales);
    let bb = dispatch::storage_from_slice(ctx, "qbl-dx-b", biases);
    let ob = dispatch::storage_zeroed(ctx, "qbl-dx-out", (m * k * 4) as u64);
    let pb = dispatch::uniform_from(ctx, "qbl-dx-params", &params);

    let groups_1d = dispatch::workgroup_count_1d(ctx, (m * k) as u64, WORKGROUP_SIZE);
    dispatch::run(
        ctx,
        "nv_kernels_qlora_bwd_dx_w4",
        &compose(DX_WGSL),
        "qlora_bwd_dx_w4",
        &[(0, &dyb), (1, &wb), (2, &sb), (3, &bb), (4, &ob), (5, &pb)],
        groups_1d,
    )?;

    let out: Vec<f32> = dispatch::read_back_arm(ctx, &ob, m * k, arm).await?;
    dx.copy_from_slice(&out);
    Ok(())
}

async fn gemm(
    ctx: &WgpuContext,
    entry: &'static str,
    label: &'static str,
    p: &[f32],
    q: &[f32],
    out: &mut [f32],
    m: usize,
    i: usize,
    j: usize,
    q_rows: usize,
    out_rows: usize,
    arm: dispatch::ReadbackArm,
) -> Result<()> {
    dispatch::check_len("gemm p", p.len(), m * i)?;
    dispatch::check_len("gemm q", q.len(), q_rows * j)?;
    dispatch::check_len("gemm out", out.len(), out_rows * j)?;
    if out.is_empty() {
        return Ok(());
    }
    check_device(ctx)?;
    let params = GemmParams {
        m_rows: m as u32,
        i_cols: i as u32,
        j_cols: j as u32,
        total: out.len() as u32,
    };
    let pb = dispatch::storage_from_slice(ctx, "qbl-gemm-p", p);
    let qb = dispatch::storage_from_slice(ctx, "qbl-gemm-q", q);
    let ob = dispatch::storage_zeroed(ctx, "qbl-gemm-out", (out.len() * 4) as u64);
    let ub = dispatch::uniform_from(ctx, "qbl-gemm-params", &params);
    let groups_1d = dispatch::workgroup_count_1d(ctx, out.len() as u64, WORKGROUP_SIZE);
    dispatch::run(
        ctx,
        label,
        &compose(GEMM_WGSL),
        entry,
        &[(0, &pb), (1, &qb), (2, &ob), (3, &ub)],
        groups_1d,
    )?;
    let got: Vec<f32> = dispatch::read_back_arm(ctx, &ob, out.len(), arm).await?;
    out.copy_from_slice(&got);
    Ok(())
}

pub async fn gemm_nn_f32_arm(
    ctx: &WgpuContext,
    p: &[f32],
    q: &[f32],
    out: &mut [f32],
    m: usize,
    i: usize,
    j: usize,
    arm: dispatch::ReadbackArm,
) -> Result<()> {
    gemm(
        ctx,
        "qlora_bwd_gemm_nn",
        "nv_kernels_qlora_bwd_gemm_nn",
        p,
        q,
        out,
        m,
        i,
        j,
        i,
        m,
        arm,
    )
    .await
}

pub async fn gemm_tn_f32_arm(
    ctx: &WgpuContext,
    p: &[f32],
    q: &[f32],
    out: &mut [f32],
    m: usize,
    i: usize,
    j: usize,
    arm: dispatch::ReadbackArm,
) -> Result<()> {
    gemm(
        ctx,
        "qlora_bwd_gemm_tn",
        "nv_kernels_qlora_bwd_gemm_tn",
        p,
        q,
        out,
        m,
        i,
        j,
        m,
        i,
        arm,
    )
    .await
}

pub struct LoraGrads {
    pub da: Vec<f32>,
    pub db: Vec<f32>,
    pub dx: Vec<f32>,
}

pub fn lora_grads_f32(
    ctx: &WgpuContext,
    dy: &[f32],
    x: &[f32],
    xr: &[f32],
    a: &[f32],
    b: &[f32],
    scaling: f32,
    m: usize,
    n: usize,
    k: usize,
    r: usize,
) -> Result<LoraGrads> {
    dispatch::drive_blocking(lora_grads_f32_arm(
        ctx,
        dy,
        x,
        xr,
        a,
        b,
        scaling,
        m,
        n,
        k,
        r,
        dispatch::ReadbackArm::Blocking,
    ))
}

pub async fn lora_grads_f32_arm(
    ctx: &WgpuContext,
    dy: &[f32],
    x: &[f32],
    xr: &[f32],
    a: &[f32],
    b: &[f32],
    scaling: f32,
    m: usize,
    n: usize,
    k: usize,
    r: usize,
    arm: dispatch::ReadbackArm,
) -> Result<LoraGrads> {
    dispatch::check_len("lora_grads dy", dy.len(), m * n)?;
    dispatch::check_len("lora_grads x", x.len(), m * k)?;
    dispatch::check_len("lora_grads xr", xr.len(), m * r)?;
    dispatch::check_len("lora_grads a", a.len(), r * k)?;
    dispatch::check_len("lora_grads b", b.len(), n * r)?;
    let g: Vec<f32> = dy.iter().map(|v| v * scaling).collect();
    let mut dxr = vec![0f32; m * r];
    gemm_nn_f32_arm(ctx, &g, b, &mut dxr, m, n, r, arm).await?;
    let mut db = vec![0f32; n * r];
    gemm_tn_f32_arm(ctx, &g, xr, &mut db, m, n, r, arm).await?;
    let mut da = vec![0f32; r * k];
    gemm_tn_f32_arm(ctx, &dxr, x, &mut da, m, r, k, arm).await?;
    let mut dx = vec![0f32; m * k];
    gemm_nn_f32_arm(ctx, &dxr, a, &mut dx, m, r, k, arm).await?;
    Ok(LoraGrads { da, db, dx })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn uniform_structs_and_both_sources_follow_the_shader_contract() {
        {
            assert_eq!(std::mem::size_of::<DxParams>(), 32);
            assert_eq!(std::mem::size_of::<GemmParams>(), 16);
        }
        {
            let src = compose(DX_WGSL);
            assert!(src.contains("fn u4_unpack("));
            assert!(src.contains("fn qlora_bwd_dx_w4("));
        }
        {
            let src = compose(GEMM_WGSL);
            assert!(src.contains("fn qlora_bwd_gemm_nn("));
            assert!(src.contains("fn qlora_bwd_gemm_tn("));
        }
    }
}
