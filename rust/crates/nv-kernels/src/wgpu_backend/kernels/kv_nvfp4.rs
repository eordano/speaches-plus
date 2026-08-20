#![allow(clippy::too_many_arguments)]

use crate::wgpu_backend::device::WgpuContext;
use crate::wgpu_backend::dequant::bytes_to_words;
use crate::wgpu_backend::pack::pack_u16_even_min_one_word as pack_bf16;
use crate::wgpu_backend::pack::unpack_u8_by_element as words_to_bytes;
use crate::wgpu_backend::{compose, dispatch, Result, WgpuError};

pub const E2M1_VALUES: [f32; 8] = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0];

pub fn encode_e2m1(x: f32) -> u8 {
    let sign = if x.is_sign_negative() { 0b1000 } else { 0 };
    let abs = x.abs();
    let mut best = 0u8;
    let mut best_err = f32::INFINITY;
    for (i, v) in E2M1_VALUES.iter().enumerate() {
        let err = (abs - v).abs();
        if err < best_err {
            best_err = err;
            best = i as u8;
        }
    }
    sign | best
}

pub fn decode_e2m1(nibble: u8) -> f32 {
    let mag = E2M1_VALUES[(nibble & 0b0111) as usize];
    if nibble & 0b1000 != 0 {
        -mag
    } else {
        mag
    }
}

pub const WGSL: &str = include_str!("../../../wgsl/kv_nvfp4.wgsl");

pub const QUANTIZE_V_ROWS_ENTRY: &str = "quantize_kv_nvfp4_v_rows";
pub const QUANTIZE_K_BLOCKS_ENTRY: &str = "quantize_kv_nvfp4_k_channel_blocks";
pub const WORKGROUP_SIZE: u32 = 256;

pub const KV_NVFP4_SINK_SLOTS_STAY_FP8_TO_ANCHOR_SOFTMAX: u32 = 4;

pub const KV_NVFP4_K_BLOCK_TOKENS_A_BLOCK_FINALIZES_THE_STEP_ITS_LAST_TOKEN_LANDS: u32 = 32;

pub const KV_NVFP4_LAYOUT_K_PER_CHANNEL_V_PER_ROW_SCALES_F32: &str =
    "post-RoPE K carries channel-wise outliers, so K e2m1 nibbles scale per (32-token block, \
     kv head, head-dim channel) while V keeps the fp8 cache's per-(slot, kv head) row scale; \
     scales stay f32 because stage1 reads them as a plain array with no decode step, and the \
     per-block K scale vector is one hd-wide row per 32 slots, L2-resident across the block's \
     warps. Both quantizers read the bf16 KV cache after kv_write, so the same entry with \
     tokens=m IS the M-row chunk twin, and the K kernel re-quantizes the whole current block \
     prefix each dispatch so a block's scales finalize the step its last token lands";

pub const E2M1_MAX: f32 = 6.0;

const SCRATCH_BYTES: u32 = WORKGROUP_SIZE * 4 + 4 + 512 * 4;

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Kv4Params {
    pub n_kv: u32,
    pub head_dim: u32,
    pub tokens: u32,
    pub slots: u32,
}

pub fn kv4_div_rn(a: f32, b: f32) -> f32 {
    const KV4_DIV_BIG: f32 = 1.0e37;
    const KV4_DIV_DOWN: f32 = 0.00390625;
    let (bb, post) = if b.abs() > KV4_DIV_BIG {
        (b * KV4_DIV_DOWN, KV4_DIV_DOWN)
    } else {
        (b, 1.0)
    };
    let r = 1.0f32 / bb;
    let q = a * r;
    (-q).mul_add(bb, a).mul_add(r, q) * post
}

pub fn nibble_word_index(elem: usize) -> (usize, u32) {
    (elem >> 3, 4 * (elem as u32 & 7))
}

pub fn k_channel_scale_index(slot: usize, n_kv: usize, kvh: usize, head_dim: usize, d: usize) -> usize {
    let block = slot / KV_NVFP4_K_BLOCK_TOKENS_A_BLOCK_FINALIZES_THE_STEP_ITS_LAST_TOKEN_LANDS as usize;
    (block * n_kv + kvh) * head_dim + d
}

pub fn k_scale_blocks(slots: usize) -> usize {
    slots.div_ceil(KV_NVFP4_K_BLOCK_TOKENS_A_BLOCK_FINALIZES_THE_STEP_ITS_LAST_TOKEN_LANDS as usize)
}

fn bf16_at(cache: &[u16], idx: usize) -> f32 {
    f32::from_bits((cache[idx] as u32) << 16)
}

fn row_scale_pair(amax: f32) -> (f32, f32) {
    if amax > 0.0 {
        (kv4_div_rn(amax, E2M1_MAX), kv4_div_rn(E2M1_MAX, amax))
    } else {
        (1.0, 1.0)
    }
}

pub fn cpu_quantize_kv_nvfp4_v_rows(
    cache_bf16: &[u16],
    out: &mut [u8],
    scales: &mut [f32],
    start: usize,
    tokens: usize,
    n_kv: usize,
    head_dim: usize,
    slots: usize,
) {
    for token in 0..tokens {
        let slot = start + token;
        if slot >= slots {
            continue;
        }
        for kvh in 0..n_kv {
            let base = (slot * n_kv + kvh) * head_dim;
            let mut amax = 0.0f32;
            for d in 0..head_dim {
                amax = amax.max(bf16_at(cache_bf16, base + d).abs());
            }
            let (scale, inv) = row_scale_pair(amax);
            scales[slot * n_kv + kvh] = scale;
            for d in 0..head_dim {
                let nib = encode_e2m1(bf16_at(cache_bf16, base + d) * inv);
                let (w, sh) = nibble_word_index(base + d);
                let byte = &mut out[w * 4 + (sh / 8) as usize];
                if sh % 8 == 0 {
                    *byte = (*byte & 0xf0) | nib;
                } else {
                    *byte = (*byte & 0x0f) | (nib << 4);
                }
            }
        }
    }
}

pub fn cpu_quantize_kv_nvfp4_k_channel_blocks(
    cache_bf16: &[u16],
    out: &mut [u8],
    scales: &mut [f32],
    start: usize,
    tokens: usize,
    n_kv: usize,
    head_dim: usize,
    slots: usize,
) {
    let bt = KV_NVFP4_K_BLOCK_TOKENS_A_BLOCK_FINALIZES_THE_STEP_ITS_LAST_TOKEN_LANDS as usize;
    let end = (start + tokens).min(slots);
    if start >= end {
        return;
    }
    for block in start / bt..=(end - 1) / bt {
        let b0 = block * bt;
        let b1 = (b0 + bt).min(end);
        for kvh in 0..n_kv {
            for d in 0..head_dim {
                let mut amax = 0.0f32;
                for t in b0..b1 {
                    amax = amax.max(bf16_at(cache_bf16, (t * n_kv + kvh) * head_dim + d).abs());
                }
                let (scale, inv) = row_scale_pair(amax);
                scales[(block * n_kv + kvh) * head_dim + d] = scale;
                for t in b0..b1 {
                    let e = (t * n_kv + kvh) * head_dim + d;
                    let nib = encode_e2m1(bf16_at(cache_bf16, e) * inv);
                    let (w, sh) = nibble_word_index(e);
                    let byte = &mut out[w * 4 + (sh / 8) as usize];
                    if sh % 8 == 0 {
                        *byte = (*byte & 0xf0) | nib;
                    } else {
                        *byte = (*byte & 0x0f) | (nib << 4);
                    }
                }
            }
        }
    }
}

pub fn cpu_dequantize_kv_nvfp4_v(
    payload: &[u8],
    scales: &[f32],
    slot: usize,
    n_kv: usize,
    kvh: usize,
    head_dim: usize,
    d: usize,
) -> f32 {
    let e = (slot * n_kv + kvh) * head_dim + d;
    decode_e2m1(nibble_at(payload, e)) * scales[slot * n_kv + kvh]
}

pub fn cpu_dequantize_kv_nvfp4_k(
    payload: &[u8],
    scales: &[f32],
    slot: usize,
    n_kv: usize,
    kvh: usize,
    head_dim: usize,
    d: usize,
) -> f32 {
    let e = (slot * n_kv + kvh) * head_dim + d;
    decode_e2m1(nibble_at(payload, e))
        * scales[k_channel_scale_index(slot, n_kv, kvh, head_dim, d)]
}

pub fn nibble_at(payload: &[u8], elem: usize) -> u8 {
    let b = payload[elem / 2];
    if elem % 2 == 0 {
        b & 0xf
    } else {
        b >> 4
    }
}

fn check_geometry(head_dim: usize) -> Result<()> {
    if !head_dim.is_multiple_of(8) || head_dim > 512 {
        return Err(WgpuError::Unsupported(format!(
            "kv_nvfp4 packs 8 e2m1 nibbles per u32 word and stages up to 512 channel scales; \
             head_dim {head_dim} must be a multiple of 8 up to 512"
        )));
    }
    Ok(())
}

fn run_quantize(
    ctx: &WgpuContext,
    entry: &str,
    cache_bf16: &[u16],
    out: &mut [u8],
    scales: &mut [f32],
    start: usize,
    tokens: usize,
    n_kv: usize,
    head_dim: usize,
    slots: usize,
    grid: (u32, u32, u32),
) -> Result<()> {
    check_geometry(head_dim)?;
    dispatch::check_len("kv_nvfp4 cache", cache_bf16.len(), slots * n_kv * head_dim)?;
    dispatch::check_len("kv_nvfp4 out", out.len(), slots * n_kv * head_dim / 2)?;
    dispatch::require_workgroup_and_scratch(ctx, "kv_nvfp4", WORKGROUP_SIZE, SCRATCH_BYTES)?;

    let params = Kv4Params {
        n_kv: n_kv as u32,
        head_dim: head_dim as u32,
        tokens: tokens as u32,
        slots: slots as u32,
    };
    let src_buf = dispatch::storage_from_slice(ctx, "kv4.src", &pack_bf16(cache_bf16));
    let out_words = bytes_to_words(out);
    let out_buf = dispatch::storage_from_slice(ctx, "kv4.out", &out_words);
    let scales_buf = dispatch::storage_from_slice(ctx, "kv4.scales", scales);
    let start_buf = dispatch::storage_from_slice(ctx, "kv4.start", &[start as i32]);
    let params_buf = dispatch::uniform_from(ctx, "kv4.params", &params);

    dispatch::run(
        ctx,
        entry,
        &compose(WGSL),
        entry,
        &[
            (0, &src_buf),
            (1, &out_buf),
            (2, &scales_buf),
            (3, &start_buf),
            (4, &params_buf),
        ],
        grid,
    )?;

    let got_out: Vec<u32> = dispatch::read_back(ctx, &out_buf, out_words.len())?;
    words_to_bytes(&got_out, out);
    let got_scales: Vec<f32> = dispatch::read_back(ctx, &scales_buf, scales.len())?;
    scales.copy_from_slice(&got_scales);
    Ok(())
}

pub fn quantize_kv_nvfp4_v_rows(
    ctx: &WgpuContext,
    cache_bf16: &[u16],
    out: &mut [u8],
    scales: &mut [f32],
    start: usize,
    tokens: usize,
    n_kv: usize,
    head_dim: usize,
    slots: usize,
) -> Result<()> {
    dispatch::check_len("kv_nvfp4 v scales", scales.len(), slots * n_kv)?;
    run_quantize(
        ctx,
        QUANTIZE_V_ROWS_ENTRY,
        cache_bf16,
        out,
        scales,
        start,
        tokens,
        n_kv,
        head_dim,
        slots,
        (n_kv as u32, tokens as u32, 1),
    )
}

pub fn k_blocks_grid_y(tokens: usize) -> u32 {
    let bt = KV_NVFP4_K_BLOCK_TOKENS_A_BLOCK_FINALIZES_THE_STEP_ITS_LAST_TOKEN_LANDS as usize;
    (tokens.div_ceil(bt) + 1) as u32
}

pub fn quantize_kv_nvfp4_k_channel_blocks(
    ctx: &WgpuContext,
    cache_bf16: &[u16],
    out: &mut [u8],
    scales: &mut [f32],
    start: usize,
    tokens: usize,
    n_kv: usize,
    head_dim: usize,
    slots: usize,
) -> Result<()> {
    dispatch::check_len(
        "kv_nvfp4 k channel scales",
        scales.len(),
        k_scale_blocks(slots) * n_kv * head_dim,
    )?;
    run_quantize(
        ctx,
        QUANTIZE_K_BLOCKS_ENTRY,
        cache_bf16,
        out,
        scales,
        start,
        tokens,
        n_kv,
        head_dim,
        slots,
        (n_kv as u32, k_blocks_grid_y(tokens), 1),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn kv4_div_rn_matches_ieee_division_and_survives_huge_denominators() {
        for i in 1..5000u32 {
            let a = i as f32 * 0.73;
            assert_eq!(kv4_div_rn(a, E2M1_MAX).to_bits(), (a / E2M1_MAX).to_bits());
        }
        let huge = 2.0e38f32;
        assert!(kv4_div_rn(E2M1_MAX, huge).is_finite());
        assert!(kv4_div_rn(E2M1_MAX, huge) > 0.0);
    }

    #[test]
    fn nibble_packing_round_trips_through_the_cpu_reference() {
        let n_kv = 2;
        let hd = 16;
        let slots = 8;
        let cache: Vec<u16> = (0..slots * n_kv * hd)
            .map(|i| half::bf16::from_f32(((i * 37 % 113) as f32 - 56.0) * 0.11).to_bits())
            .collect();
        let mut out = vec![0u8; slots * n_kv * hd / 2];
        let mut scales = vec![0f32; slots * n_kv];
        cpu_quantize_kv_nvfp4_v_rows(&cache, &mut out, &mut scales, 0, slots, n_kv, hd, slots);
        for slot in 0..slots {
            for kvh in 0..n_kv {
                for d in 0..hd {
                    let e = (slot * n_kv + kvh) * hd + d;
                    let v = f32::from_bits((cache[e] as u32) << 16);
                    let q = cpu_dequantize_kv_nvfp4_v(&out, &scales, slot, n_kv, kvh, hd, d);
                    let sc = scales[slot * n_kv + kvh];
                    assert!(
                        (q - v).abs() <= 0.5 * sc + 1e-6,
                        "slot {slot} kvh {kvh} d {d}: {q} vs {v} (scale {sc}); e2m1 spacing \
                         never exceeds one scale step below E2M1_MAX"
                    );
                }
            }
        }
    }

    #[test]
    fn k_channel_scale_indexing_walks_blocks_of_32() {
        assert_eq!(k_channel_scale_index(0, 2, 0, 8, 3), 3);
        assert_eq!(k_channel_scale_index(31, 2, 1, 8, 3), 11);
        assert_eq!(k_channel_scale_index(32, 2, 0, 8, 0), 16);
        assert_eq!(k_scale_blocks(1), 1);
        assert_eq!(k_scale_blocks(32), 1);
        assert_eq!(k_scale_blocks(33), 2);
    }
}
