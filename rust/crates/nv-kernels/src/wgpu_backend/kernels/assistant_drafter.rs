pub const WGSL: &str = include_str!("../../../wgsl/assistant_drafter.wgsl");

pub const ENTRY_EMBED_CONCAT: &str = "ad_embed_concat";
pub const ENTRY_GEMV: &str = "ad_gemv";
pub const ENTRY_RMSNORM: &str = "ad_rmsnorm";
pub const ENTRY_ROPE: &str = "ad_rope";
pub const ENTRY_ATTN_SCORES: &str = "ad_attn_scores";
pub const ENTRY_ATTN_SOFTMAX: &str = "ad_attn_softmax";
pub const ENTRY_ATTN_CTX: &str = "ad_attn_ctx";
pub const ENTRY_ADD_SCALE: &str = "ad_add_scale";
pub const ENTRY_TOPK: &str = "ad_topk";
pub const ENTRY_CAND_LOGITS: &str = "ad_cand_logits";
pub const ENTRY_PICK: &str = "ad_pick";

pub const EMBED_MAX_CHUNKS: usize = 4;

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, bytemuck::Pod, bytemuck::Zeroable)]
pub struct AdEmbedParams {
    pub bh: u32,
    pub rows_per_chunk: u32,
    pub norm: f32,
    pub pad0: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, bytemuck::Pod, bytemuck::Zeroable)]
pub struct AdGemvParams {
    pub n: u32,
    pub k: u32,
    pub act: u32,
    pub mode: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, bytemuck::Pod, bytemuck::Zeroable)]
pub struct AdRmsParams {
    pub rows: u32,
    pub dim: u32,
    pub eps: f32,
    pub pad0: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, bytemuck::Pod, bytemuck::Zeroable)]
pub struct AdRopeParams {
    pub nh: u32,
    pub hd: u32,
    pub pad0: u32,
    pub pad1: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, bytemuck::Pod, bytemuck::Zeroable)]
pub struct AdAttnParams {
    pub n_kv: u32,
    pub nh: u32,
    pub hd: u32,
    pub len: u32,
    pub start: u32,
    pub stride: u32,
    pub pad0: u32,
    pub pad1: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, bytemuck::Pod, bytemuck::Zeroable)]
pub struct AdAddParams {
    pub n: u32,
    pub pad0: u32,
    pub scale: f32,
    pub pad1: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, bytemuck::Pod, bytemuck::Zeroable)]
pub struct AdTopkParams {
    pub n: u32,
    pub k: u32,
    pub pad0: u32,
    pub pad1: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, bytemuck::Pod, bytemuck::Zeroable)]
pub struct AdCandParams {
    pub top: u32,
    pub vpc: u32,
    pub h: u32,
    pub pad0: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, bytemuck::Pod, bytemuck::Zeroable)]
pub struct AdPickParams {
    pub n: u32,
    pub pad0: u32,
    pub pad1: u32,
    pub pad2: u32,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wgsl_declares_every_entry_point() {
        for entry in [
            ENTRY_EMBED_CONCAT,
            ENTRY_GEMV,
            ENTRY_RMSNORM,
            ENTRY_ROPE,
            ENTRY_ATTN_SCORES,
            ENTRY_ATTN_SOFTMAX,
            ENTRY_ATTN_CTX,
            ENTRY_ADD_SCALE,
            ENTRY_TOPK,
            ENTRY_CAND_LOGITS,
            ENTRY_PICK,
        ] {
            assert!(WGSL.contains(&format!("fn {entry}(")));
        }
    }

    #[test]
    fn params_are_uniform_buffer_sized() {
        assert!(std::mem::size_of::<AdEmbedParams>().is_multiple_of(16));
        assert!(std::mem::size_of::<AdGemvParams>().is_multiple_of(16));
        assert!(std::mem::size_of::<AdRmsParams>().is_multiple_of(16));
        assert!(std::mem::size_of::<AdRopeParams>().is_multiple_of(16));
        assert!(std::mem::size_of::<AdAttnParams>().is_multiple_of(16));
        assert!(std::mem::size_of::<AdAddParams>().is_multiple_of(16));
        assert!(std::mem::size_of::<AdTopkParams>().is_multiple_of(16));
        assert!(std::mem::size_of::<AdCandParams>().is_multiple_of(16));
        assert!(std::mem::size_of::<AdPickParams>().is_multiple_of(16));
    }
}
