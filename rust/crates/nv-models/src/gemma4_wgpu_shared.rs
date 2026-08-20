use nv_kernels::wgpu_backend::WgpuError;

pub const GEMV_PK_ENTRY: &str = "g4w_gemv_bf16_vec8_pk";
pub const GEMV_PK3_ENTRY: &str = "g4w_gemv_bf16_vec8_pk3";
pub const ROPE_F32_ENTRY: &str = "g4w_rope_bf16_f32";
pub const FLASH2_PK_ENTRY: &str = "g4w_flash_splitk_stage2_pk";

pub const GEMV_PK_WGSL: &str = include_str!("../../nv-kernels/wgsl/g4shared_gemv_pk.wgsl");

pub const ROPE_F32_WGSL: &str = include_str!("../../nv-kernels/wgsl/g4shared_rope_f32.wgsl");

pub const FLASH2_PK_WGSL: &str = include_str!("../../nv-kernels/wgsl/g4shared_flash2_pk.wgsl");

pub fn bf16_bits(x: f32) -> u16 {
    (half::bf16::from_f32(x)).to_bits()
}

pub fn pack_pairs(src: &[u16]) -> Vec<u32> {
    let mut out = vec![0u32; src.len().div_ceil(2).max(1)];
    for (i, v) in src.iter().enumerate() {
        out[i / 2] |= (*v as u32) << (16 * (i % 2));
    }
    out
}

pub fn bytes_to_words(bytes: &[u8]) -> Vec<u32> {
    nv_kernels::wgpu_backend::pack::pack_u8_min_one_word(bytes)
}

pub fn rope_tables(
    head_dim: usize,
    rope_theta: f32,
    partial: f32,
    rows: usize,
) -> (Vec<f32>, Vec<f32>) {
    let half = head_dim / 2;
    let rope_angles = ((partial * head_dim as f32 / 2.0) as usize).min(half);
    let mut inv_freq = vec![0f32; half];
    for (i, f) in inv_freq.iter_mut().enumerate().take(rope_angles) {
        *f = 1.0 / rope_theta.powf((i as f32 * 2.0) / (head_dim as f32));
    }
    rope_tables_from_inv_freq(&inv_freq, rows)
}

pub fn rope_tables_from_inv_freq(inv_freq: &[f32], rows: usize) -> (Vec<f32>, Vec<f32>) {
    let half = inv_freq.len().max(1);
    let mut cos = vec![0f32; rows * half];
    let mut sin = vec![0f32; rows * half];
    for p in 0..rows {
        for (i, inv) in inv_freq.iter().enumerate() {
            let t = (p as f32) * *inv;
            cos[p * half + i] = t.cos();
            sin[p * half + i] = t.sin();
        }
    }
    (cos, sin)
}

pub fn err(e: WgpuError) -> anyhow::Error {
    anyhow::anyhow!("{e}")
}
