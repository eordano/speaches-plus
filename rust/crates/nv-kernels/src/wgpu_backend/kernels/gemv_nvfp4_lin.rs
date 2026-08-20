use crate::wgpu_backend::compose;
use crate::wgpu_backend::kernels::gemv_nvfp4 as g;

pub const WGSL: &str = include_str!("../../../wgsl/gemv_nvfp4_lin.wgsl");

pub const LIN_ENTRY: &str = "gemv_nvfp4_lin_sg";
pub const SWZ_ENTRY: &str = "gemv_nvfp4_swz_sg";
pub const NOSCALE_ENTRY: &str = "gemv_nvfp4_noscale_sg";
pub const NODEC_ENTRY: &str = "gemv_nvfp4_nodec_sg";
pub const XPRE_ENTRY: &str = "gemv_nvfp4_xpre_sg";
pub const V3_ENTRY: &str = "gemv_nvfp4_v3";
pub const V3_NODEC_ENTRY: &str = "gemv_nvfp4_v3_nodec";
pub const V3_STREAM_ENTRY: &str = "gemv_nvfp4_v3_stream";
pub const WORKGROUP_SIZE: u32 = 128;
pub const ROWS_PER_GROUP: u32 = 4;
pub const V3_ROWS_PER_GROUP: u32 = 4;

pub fn i8map(s: u32) -> u32 {
    let k = s & 0x0707_0707;
    let hm = ((k >> 2) & 0x0101_0101).wrapping_mul(255);
    let e7 = (k & (k >> 1) & (k >> 2)) & 0x0101_0101;
    let m = k.wrapping_add((k & 0x0303_0303) & hm).wrapping_add(e7 << 1);
    let sb = (s & (k.wrapping_add(0x0707_0707) & 0x0808_0808)) >> 3;
    (m ^ sb.wrapping_mul(255)).wrapping_add(sb)
}

pub fn x_i8_from_packed(x_packed: &[u32]) -> Vec<u32> {
    let mut out = Vec::with_capacity(x_packed.len() * 2);
    for w in x_packed {
        out.push(i8map(*w));
        out.push(i8map(*w >> 4));
    }
    out
}

pub const UE4M3_SUBNORMAL_STEP: f32 = 0.001953125;

pub fn ue4m3_decode(bits: u32) -> f32 {
    let b = bits & 127;
    if b < 8 {
        b as f32 * UE4M3_SUBNORMAL_STEP
    } else {
        f32::from_bits((b << 20).wrapping_add(0x3c00_0000))
    }
}

pub fn v3_shape_ok(k: usize) -> bool {
    k.is_multiple_of(32) && (k / crate::wgpu_backend::dequant::NVFP4_BLOCK_SIZE).is_multiple_of(2)
}

pub fn source() -> String {
    compose(&format!("{}\n{WGSL}", g::decode_block()))
}

pub fn ws_row_stride(k_blocks: usize) -> usize {
    k_blocks.div_ceil(4) * 4
}

pub fn linear_scale_len(n: usize, k_blocks: usize) -> usize {
    n * ws_row_stride(k_blocks)
}

pub fn linear_scales_from_swizzled(w_scales: &[u8], n: usize, k_blocks: usize) -> Vec<u32> {
    let stride = ws_row_stride(k_blocks);
    let k_tiles = g::k_tiles(k_blocks);
    let mut out = vec![0u8; n * stride];
    for row in 0..n {
        let m_tile = row / 128;
        let d2 = (row / 32) % 4;
        let d3 = row % 32;
        for block in 0..k_blocks {
            let k_tile = block / 4;
            let d5 = block % 4;
            let src = ((m_tile * k_tiles + k_tile) * 32 + d3) * 16 + d2 * 4 + d5;
            out[row * stride + block] = w_scales[src];
        }
    }
    out.chunks(4)
        .map(|c| {
            let mut w = 0u32;
            for (i, b) in c.iter().enumerate() {
                w |= (*b as u32) << (8 * i);
            }
            w
        })
        .collect()
}

pub fn params(alpha: f32, n: usize, k: usize, groups_x: u32) -> g::GemvParams {
    let mut p = g::gemv_params(alpha, n, k, groups_x);
    p.pad0 = ws_row_stride(k / crate::wgpu_backend::dequant::NVFP4_BLOCK_SIZE) as u32;
    p
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_lin_source_carries_the_shared_decode_block_and_every_entry() {
        let src = source();
        for e in [LIN_ENTRY, SWZ_ENTRY, NOSCALE_ENTRY] {
            assert!(src.contains(e), "missing {e}");
        }
        assert!(src.contains("fn gemv_dot8("));
        assert!(src.contains("fn gemv_i8map("));
        assert!(src.contains("fn nvfp4_scale_byte_index("));
        assert!(src.contains("subgroupShuffleXor"));
        assert!(!src.contains("enable subgroups"));
        assert!(!src.contains("var<workgroup>"));
    }

    #[test]
    fn the_repack_is_the_exact_inverse_of_the_swizzle_index() {
        let n = 300usize;
        let k_blocks = 21usize;
        let k_tiles = g::k_tiles(k_blocks);
        let swz_len = g::swizzled_scale_len(n, k_blocks);
        let swz: Vec<u8> = (0..swz_len).map(|i| (i % 251) as u8).collect();
        let lin = linear_scales_from_swizzled(&swz, n, k_blocks);
        let stride = ws_row_stride(k_blocks);
        assert_eq!(lin.len(), (n * stride).div_ceil(4));
        for row in [0usize, 1, 31, 32, 127, 128, 299] {
            for block in 0..k_blocks {
                let m_tile = row / 128;
                let d2 = (row / 32) % 4;
                let d3 = row % 32;
                let src = ((m_tile * k_tiles + block / 4) * 32 + d3) * 16 + d2 * 4 + block % 4;
                let idx = row * stride + block;
                let got = (lin[idx / 4] >> (8 * (idx % 4))) & 0xff;
                assert_eq!(got as u8, swz[src], "row={row} block={block}");
            }
        }
    }

    #[test]
    fn the_cpu_scale_decode_matches_nv_quant_on_all_256_codes_including_the_subnormal_band() {
        for b in 0u32..256 {
            assert_eq!(
                ue4m3_decode(b),
                nv_quant::nvfp4::decode_ue4m3((b & 127) as u8),
                "code {b:#04x}: this is the oracle oracle_v3 and oracle_tree in \
                 wgpu_gemv_nvfp4_scale_layout compare the GPU against. Its fixture draws scale \
                 bytes as 0x30 | rand4, so a biased exponent of 0 is never generated and a \
                 normal-formula-only decode of the smallest 8 codes cannot be caught there"
            );
        }
        assert_eq!(ue4m3_decode(0), 0.0);
        assert_eq!(ue4m3_decode(7), 7.0 * UE4M3_SUBNORMAL_STEP);
    }

    #[test]
    fn the_row_stride_is_word_aligned() {
        for k_blocks in [1usize, 2, 3, 4, 5, 336, 512, 1344] {
            let s = ws_row_stride(k_blocks);
            assert!(s >= k_blocks);
            assert_eq!(s % 4, 0);
            assert!(s - k_blocks < 4);
        }
    }
}
