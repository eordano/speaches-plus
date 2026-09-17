use crate::wgpu_backend::compose_enabled;
use crate::wgpu_backend::device::WgpuContext;
use crate::wgpu_backend::kernels::gemm_coop_f16;
use crate::wgpu_backend::qualify::{CoopDecision, CoopRequest, CoopScalar};

pub const WGSL: &str = include_str!("../../../wgsl/q3d_pf_flash_tiled_coop.wgsl");

pub const ENTRY_STAGE1: &str = "q3w_pf_flash1_fp8kv_tiled_coop_sg";

pub const ENTRY_QCAST: &str = "q3w_pf_flash_coop_qcast";

pub const COOP_HEAD_DIM_ONLY_256_BECAUSE_TILE_COUNTS_ARE_BAKED: usize = 256;

pub const COOP_TILE_ROWS_MATCH_THE_SCALAR_TILED_ARM: usize = 32;

pub const COOP_POS_PER_ROUND: usize = 16;

pub fn source() -> String {
    compose_enabled(&gemm_coop_f16::ENABLES, WGSL)
}

pub fn supported(
    ctx: &WgpuContext,
    head_dim: usize,
    m_rows: usize,
    ring: u32,
    window: u32,
) -> Result<(), String> {
    if head_dim != COOP_HEAD_DIM_ONLY_256_BECAUSE_TILE_COUNTS_ARE_BAKED {
        return Err(format!("head_dim {head_dim} != 256"));
    }
    if !m_rows.is_multiple_of(COOP_TILE_ROWS_MATCH_THE_SCALAR_TILED_ARM) {
        return Err(format!("m_rows {m_rows} not a multiple of 32"));
    }
    if ring != 0 || window != 0 {
        return Err(format!(
            "ring {ring} / window {window} unsupported: the coop arm streams linear slots \
             with a zero causal floor"
        ));
    }
    if ctx.subgroup_width() != Some(32) {
        return Err("subgroup width != 32".into());
    }
    match ctx
        .caps
        .coop_decision(&CoopRequest::square(16, CoopScalar::F16, CoopScalar::F32))
    {
        CoopDecision::Compile | CoopDecision::CompileUnadvertised(_) => Ok(()),
        CoopDecision::Skip(why) => Err(format!("no native 16x16x16 f16 coopmat: {why}")),
    }
}
