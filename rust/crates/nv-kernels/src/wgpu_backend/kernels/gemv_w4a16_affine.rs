use crate::wgpu_backend::compose;

pub const WGSL: &str = crate::wgsl!("../../../wgsl/gemv_w4a16_affine.wgsl");

pub const WORKGROUP_SIZE: u32 = 256;
pub const LANES_PER_ROW: u32 = 32;
pub const ROWS_PER_GROUP: u32 = 8;
pub const BLOCK_ENTRY: &str = "gemv_w4a16_affine_block";
pub const BLOCK_PK_ENTRY: &str = "gemv_w4a16_affine_block_pk";
pub const MROW4_ENTRY_REQUIRES_GS_MULTIPLE_OF_32: &str = "gemv_w4a16_affine_mrow4";
pub const MROW4_PK_ENTRY_REQUIRES_GS_MULTIPLE_OF_32: &str = "gemv_w4a16_affine_mrow4_pk";
pub const MROW4_ROWS_PER_GROUP: u32 = 32;
pub const MROW8_PROBE_ENTRY: &str = "gemv_w4a16_affine_mrow8";
pub const MROW8_ROWS_PER_GROUP: u32 = 64;
pub const MROW4H_PROBE_ENTRY: &str = "gemv_w4a16_affine_mrow4h";
pub const MROW4H_ROWS_PER_GROUP: u32 = 16;
pub const MROW4H_PK_ENTRY_REQUIRES_GS_MULTIPLE_OF_32: &str = "gemv_w4a16_affine_mrow4h_pk";
pub const GU_PK_ENTRY_SEGMENTS_MUST_BE_MULTIPLES_OF_4_ROWS: &str = "gemv_w4a16_affine_mrow4h_gu_pk";
pub const GU_W_B_SLOT: u32 = 20;
pub const GU_SCALE_B_SLOT: u32 = 21;
pub const GU_BIAS_B_SLOT: u32 = 22;
pub const BIAS_SLOT: u32 = 6;
pub const QKV_PK_ENTRY_SEGMENTS_MUST_BE_EVEN_ROWS: &str = "gemv_w4a16_affine_mrow4_qkv_pk";
pub const QKV_Y_K_SLOT: u32 = 5;
pub const QKV_Y_V_SLOT: u32 = 7;

pub fn source() -> String {
    compose(WGSL)
}
