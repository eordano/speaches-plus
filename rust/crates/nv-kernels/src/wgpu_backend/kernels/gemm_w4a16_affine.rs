use crate::wgpu_backend::kernels::gemv_w4a16::ScaleGrain;
use crate::wgpu_backend::{compose, Result, WgpuError};

pub const WGSL: &str = include_str!("../../../wgsl/gemm_w4a16_affine.wgsl");

pub const WORKGROUP_SIZE: u32 = 256;
pub const LANES_PER_ROW: u32 = 32;
pub const ROWS_PER_GROUP: u32 = 8;
pub const MAX_M: u32 = 16;
pub const MROWS_ENTRY: &str = "gemm_w4a16_affine_mrows";
pub const MROWS_V2_PK_ENTRY_STAGES_X_IN_THREADGROUP_FOR_64_ROWS: &str =
    "gemm_w4a16_affine_mrows_v2_pk";
pub const MROWS_V2_ROWS_PER_GROUP: u32 = 64;
pub const BIAS_SLOT: u32 = 6;
pub const GRAIN: ScaleGrain = ScaleGrain::Ge32;

pub fn grain_rule(group_size: usize) -> Result<()> {
    if GRAIN.accepts(group_size) {
        return Ok(());
    }
    Err(WgpuError::Shape(format!(
        "gemm_w4a16_affine_mrows folds one scale+bias per 32-elem chunk and would misapply GS={group_size}; it requires a multiple of 32"
    )))
}

pub fn m_rule(m: usize) -> Result<()> {
    if (1..=MAX_M as usize).contains(&m) {
        return Ok(());
    }
    Err(WgpuError::Shape(format!(
        "gemm_w4a16_affine_mrows carries at most {MAX_M} live x rows per dispatch; got M={m}"
    )))
}

pub fn source() -> String {
    compose(WGSL)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn source_declares_the_mrows_entry_and_its_prelude() {
        let src = source();
        assert!(src.contains("fn gemm_w4a16_affine_mrows("));
        assert!(src.contains("fn bf16_decode("));
        assert!(src.contains("fn bf16_encode("));
        assert!(src.contains(&format!("W4AFFM_MAX_M: u32 = {MAX_M}u")));
    }

    #[test]
    fn the_workgroup_byte_const_tracks_every_workgroup_declaration() {
        assert!(WGSL.contains("W4AFFM_WORKGROUP_BYTES_TOTAL_VS_METAL_LIMIT_32768: u32 = 8192u"));
        assert_eq!(
            WGSL.matches("var<workgroup>").count(),
            1,
            "a new var<workgroup> must be summed into W4AFFM_WORKGROUP_BYTES_TOTAL_VS_METAL_LIMIT_32768 because Metal executes over-limit pipelines with aliased tiles"
        );
        assert!(WGSL.contains("var<workgroup> w4affm_xf_tile: array<vec4<f32>, 512>"));
    }

    #[test]
    fn the_rules_refuse_what_the_body_cannot_express() {
        assert!(grain_rule(16).is_err());
        assert!(grain_rule(48).is_err());
        assert!(grain_rule(32).is_ok());
        assert!(grain_rule(64).is_ok());
        assert!(m_rule(0).is_err());
        assert!(m_rule(17).is_err());
        assert!(m_rule(1).is_ok());
        assert!(m_rule(16).is_ok());
    }
}
