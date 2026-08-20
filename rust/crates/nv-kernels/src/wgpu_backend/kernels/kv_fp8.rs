#![allow(clippy::too_many_arguments)]

use crate::wgpu_backend::device::WgpuContext;
use crate::wgpu_backend::dequant::bytes_to_words;
use crate::wgpu_backend::pack::unpack_u8_by_element as words_to_bytes;
use crate::wgpu_backend::{compose, dispatch, Result, WgpuError};
use crate::wgpu_backend::pack::{pack_u16_even_min_one_word as pack_bf16, unpack_u16_pairs_clamped as unpack_bf16};

pub const WGSL: &str = include_str!("../../../wgsl/kv_fp8.wgsl");

pub const QUANTIZE_ENTRY: &str = "quantize_kv_fp8";
pub const QUANTIZE_ENTRY_KT: &str = "quantize_kv_fp8_kt";
pub const QUANTIZE_PAIR_KV_WRITE_ENTRY_GRID_Y_IS_2_K_THEN_V_AND_ALSO_COPIES_THE_BF16_ROWS: &str =
    "quantize_kv_fp8_kv_write_bf16";
pub const DEQUANTIZE_ENTRY: &str = "dequantize_kv_fp8";
pub const WORKGROUP_SIZE: u32 = 256;
pub const FP8_E4M3_MAX: f32 = 448.0;

const SCRATCH_BYTES: u32 = WORKGROUP_SIZE * 4 + 4;

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, bytemuck::Pod, bytemuck::Zeroable)]
struct KvFp8Params {
    n_tokens: u32,
    n_kv: u32,
    head_dim: u32,
    ring: u32,
    pairs: u32,
    start: u32,
    slots: u32,
    reserved: u32,
}

pub fn encode_e4m3(x: f32) -> u8 {
    let b = x.to_bits();
    let sign = ((b >> 31) << 7) as u8;
    let mag = b & 0x7fff_ffff;
    if mag > 0x7f80_0000 {
        return sign | 0x7f;
    }
    let e = (mag >> 23) as i32 - 127;
    if e >= -6 {
        let lsb = (mag >> 20) & 1;
        let r = mag + 0x7_ffff + lsb;
        let e2 = (r >> 23) as i32 - 127;
        let m2 = ((r >> 20) & 7) as u8;
        if e2 > 8 || (e2 == 8 && m2 == 7) {
            return sign | 0x7e;
        }
        return sign | (((e2 + 7) as u8) << 3) | m2;
    }
    let s = (14 - e) as u32;
    if s >= 32 {
        return sign;
    }
    let full = 0x80_0000 | (mag & 0x7f_ffff);
    let q = full >> s;
    let round_bit = (full >> (s - 1)) & 1;
    let rest = full & ((1u32 << (s - 1)) - 1);
    let mut n = q;
    if round_bit == 1 && (rest != 0 || (q & 1) == 1) {
        n += 1;
    }
    sign | n as u8
}

pub fn decode_e4m3(b: u8) -> f32 {
    let mag = b & 0x7f;
    if mag == 0x7f {
        return f32::NAN;
    }
    let e = (mag >> 3) as i32;
    let m = (mag & 7) as f32;
    let v = if e == 0 {
        m * 0.001_953_125
    } else {
        (1.0 + m * 0.125) * (2f32).powi(e - 7)
    };
    if b & 0x80 != 0 {
        -v
    } else {
        v
    }
}

pub fn div_rn(a: f32, b: f32) -> f32 {
    let r = 1.0f32 / b;
    let q = a * r;
    (-q).mul_add(b, a).mul_add(r, q)
}

fn check_device(ctx: &WgpuContext) -> Result<()> {
    dispatch::require_workgroup_and_scratch(ctx, "kv_fp8", WORKGROUP_SIZE, SCRATCH_BYTES)
}

fn slot_count(bytes_or_scales: usize, per_slot: usize, what: &str) -> Result<usize> {
    if per_slot == 0 || !bytes_or_scales.is_multiple_of(per_slot) {
        return Err(WgpuError::Shape(format!(
            "{what}: length {bytes_or_scales} is not a multiple of {per_slot}"
        )));
    }
    Ok(bytes_or_scales / per_slot)
}

fn max_slot_touched(start: usize, n_tokens: usize, ring: usize) -> usize {
    if ring > 0 {
        ring - 1
    } else {
        start + n_tokens - 1
    }
}

pub fn quantize_kv_fp8(
    ctx: &WgpuContext,
    x: &[u16],
    out: &mut [u8],
    scales: &mut [f32],
    start: &[i32],
    n_tokens: usize,
    n_kv: usize,
    head_dim: usize,
    ring: usize,
) -> Result<()> {
    quantize_kv_fp8_entry(
        ctx,
        x,
        out,
        scales,
        start,
        n_tokens,
        n_kv,
        head_dim,
        ring,
        QUANTIZE_ENTRY,
    )
}

pub fn quantize_kv_fp8_kt(
    ctx: &WgpuContext,
    x: &[u16],
    out: &mut [u8],
    scales: &mut [f32],
    start: &[i32],
    n_tokens: usize,
    n_kv: usize,
    head_dim: usize,
    ring: usize,
) -> Result<()> {
    quantize_kv_fp8_entry(
        ctx,
        x,
        out,
        scales,
        start,
        n_tokens,
        n_kv,
        head_dim,
        ring,
        QUANTIZE_ENTRY_KT,
    )
}

fn quantize_kv_fp8_entry(
    ctx: &WgpuContext,
    x: &[u16],
    out: &mut [u8],
    scales: &mut [f32],
    start: &[i32],
    n_tokens: usize,
    n_kv: usize,
    head_dim: usize,
    ring: usize,
    entry: &str,
) -> Result<()> {
    if n_tokens == 0 || n_kv == 0 || head_dim == 0 {
        return Ok(());
    }
    if !head_dim.is_multiple_of(4) {
        return Err(WgpuError::Unsupported(format!(
            "kv_fp8 quantize needs head_dim divisible by 4 so fp8 bytes land on whole u32 words; got {head_dim}"
        )));
    }
    if start.is_empty() {
        return Err(WgpuError::Shape(
            "kv_fp8 quantize: start buffer is empty".to_string(),
        ));
    }
    if start[0] < 0 {
        return Err(WgpuError::Shape(format!(
            "kv_fp8 quantize: start must be non-negative; got {}",
            start[0]
        )));
    }
    dispatch::check_len("kv_fp8 x", x.len(), n_tokens * n_kv * head_dim)?;
    let slots = slot_count(out.len(), n_kv * head_dim, "kv_fp8 out")?;
    dispatch::check_len("kv_fp8 scales", scales.len(), slots * n_kv)?;
    let start0 = start[0] as usize;
    if max_slot_touched(start0, n_tokens, ring) >= slots {
        return Err(WgpuError::Shape(format!(
            "kv_fp8 quantize: start {start0} + n_tokens {n_tokens} (ring {ring}) exceeds {slots} slots"
        )));
    }
    check_device(ctx)?;

    let pairs = n_tokens * n_kv;
    let params = KvFp8Params {
        n_tokens: n_tokens as u32,
        n_kv: n_kv as u32,
        head_dim: head_dim as u32,
        ring: ring as u32,
        pairs: pairs as u32,
        start: start0 as u32,
        slots: slots as u32,
        reserved: 0,
    };

    let x_buf = dispatch::storage_from_slice(ctx, "kv_fp8.x", &pack_bf16(x));
    let out_words = bytes_to_words(out);
    let out_buf = dispatch::storage_from_slice(ctx, "kv_fp8.out", &out_words);
    let scales_buf = dispatch::storage_from_slice(ctx, "kv_fp8.scales", scales);
    let start_buf = dispatch::storage_from_slice(ctx, "kv_fp8.start", &start[..1]);
    let params_buf = dispatch::uniform_from(ctx, "kv_fp8.params", &params);

    let groups = dispatch::workgroup_count_1d(ctx, pairs as u64, 1);
    dispatch::run(
        ctx,
        entry,
        &compose(WGSL),
        entry,
        &[
            (0, &x_buf),
            (1, &out_buf),
            (2, &scales_buf),
            (3, &start_buf),
            (4, &params_buf),
        ],
        groups,
    )?;

    let got_out: Vec<u32> = dispatch::read_back(ctx, &out_buf, out_words.len())?;
    words_to_bytes(&got_out, out);
    let got_scales: Vec<f32> = dispatch::read_back(ctx, &scales_buf, scales.len())?;
    scales.copy_from_slice(&got_scales);
    Ok(())
}

pub fn dequantize_kv_fp8(
    ctx: &WgpuContext,
    src: &[u8],
    scales: &[f32],
    out: &mut [u16],
    n_tokens: usize,
    n_kv: usize,
    head_dim: usize,
) -> Result<()> {
    dequantize_kv_fp8_ring(ctx, src, scales, out, 0, n_tokens, n_kv, head_dim, 0)
}

pub fn dequantize_kv_fp8_ring(
    ctx: &WgpuContext,
    src: &[u8],
    scales: &[f32],
    out: &mut [u16],
    start: usize,
    n_tokens: usize,
    n_kv: usize,
    head_dim: usize,
    ring: usize,
) -> Result<()> {
    if n_tokens == 0 || n_kv == 0 || head_dim == 0 {
        return Ok(());
    }
    if !head_dim.is_multiple_of(2) {
        return Err(WgpuError::Unsupported(format!(
            "kv_fp8 dequantize needs an even head_dim so bf16 pairs land on whole u32 words; got {head_dim}"
        )));
    }
    dispatch::check_len("kv_fp8 out", out.len(), n_tokens * n_kv * head_dim)?;
    let slots = slot_count(src.len(), n_kv * head_dim, "kv_fp8 src")?;
    dispatch::check_len("kv_fp8 scales", scales.len(), slots * n_kv)?;
    if max_slot_touched(start, n_tokens, ring) >= slots {
        return Err(WgpuError::Shape(format!(
            "kv_fp8 dequantize: start {start} + n_tokens {n_tokens} (ring {ring}) exceeds {slots} slots"
        )));
    }
    check_device(ctx)?;

    let pairs = n_tokens * n_kv;
    let params = KvFp8Params {
        n_tokens: n_tokens as u32,
        n_kv: n_kv as u32,
        head_dim: head_dim as u32,
        ring: ring as u32,
        pairs: pairs as u32,
        start: start as u32,
        slots: slots as u32,
        reserved: 0,
    };

    let src_buf = dispatch::storage_from_slice(ctx, "kv_fp8.src", &bytes_to_words(src));
    let scales_buf = dispatch::storage_from_slice(ctx, "kv_fp8.dq_scales", scales);
    let out_words = out.len() / 2;
    let out_buf = dispatch::storage_zeroed(ctx, "kv_fp8.dq_out", (out_words * 4) as u64);
    let params_buf = dispatch::uniform_from(ctx, "kv_fp8.dq_params", &params);

    let groups = dispatch::workgroup_count_1d(ctx, pairs as u64, 1);
    dispatch::run(
        ctx,
        "nv_kernels_dequantize_kv_fp8",
        &compose(WGSL),
        DEQUANTIZE_ENTRY,
        &[
            (5, &src_buf),
            (6, &scales_buf),
            (7, &out_buf),
            (8, &params_buf),
        ],
        groups,
    )?;

    let got: Vec<u32> = dispatch::read_back(ctx, &out_buf, out_words)?;
    unpack_bf16(&got, out);
    Ok(())
}

pub fn cpu_quantize_kv_fp8(
    x: &[u16],
    out: &mut [u8],
    scales: &mut [f32],
    start: usize,
    n_tokens: usize,
    n_kv: usize,
    head_dim: usize,
    ring: usize,
) {
    for token in 0..n_tokens {
        for kv_head in 0..n_kv {
            let mut slot = start + token;
            if ring > 0 {
                slot %= ring;
            }
            let base_src = (token * n_kv + kv_head) * head_dim;
            let base_dst = (slot * n_kv + kv_head) * head_dim;
            let mut amax = 0.0f32;
            for d in 0..head_dim {
                let v = f32::from_bits((x[base_src + d] as u32) << 16);
                let a = v.abs();
                if a > amax {
                    amax = a;
                }
            }
            let (scale, inv) = if amax > 0.0 {
                (div_rn(amax, FP8_E4M3_MAX), div_rn(FP8_E4M3_MAX, amax))
            } else {
                (1.0, 1.0)
            };
            scales[slot * n_kv + kv_head] = scale;
            for d in 0..head_dim {
                let v = f32::from_bits((x[base_src + d] as u32) << 16);
                out[base_dst + d] = encode_e4m3(v * inv);
            }
        }
    }
}

pub fn cpu_dequantize_kv_fp8(
    src: &[u8],
    scales: &[f32],
    out: &mut [u16],
    start: usize,
    n_tokens: usize,
    n_kv: usize,
    head_dim: usize,
    ring: usize,
) {
    for token in 0..n_tokens {
        for kv_head in 0..n_kv {
            let mut slot = start + token;
            if ring > 0 {
                slot %= ring;
            }
            let base = (slot * n_kv + kv_head) * head_dim;
            let obase = (token * n_kv + kv_head) * head_dim;
            let scale = scales[slot * n_kv + kv_head];
            for d in 0..head_dim {
                let v = decode_e4m3(src[base + d]) * scale;
                out[obase + d] = half::bf16::from_f32(v).to_bits();
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn e4m3_round_trips_every_finite_code() {
        for b in 0u16..256 {
            let b = b as u8;
            if b & 0x7f == 0x7f {
                continue;
            }
            let v = decode_e4m3(b);
            let back = encode_e4m3(v);
            assert_eq!(
                back, b,
                "code {b:#04x} decoded to {v} re-encoded to {back:#04x}"
            );
        }
    }

    #[test]
    fn e4m3_encode_ties_to_even() {
        assert_eq!(encode_e4m3(1.0), 0x38);
        assert_eq!(encode_e4m3(1.0625), 0x38);
        assert_eq!(encode_e4m3(1.1875), 0x3a);
        assert_eq!(encode_e4m3(-1.0625), 0xb8);
        assert_eq!(encode_e4m3(448.0), 0x7e);
        assert_eq!(encode_e4m3(464.0), 0x7e);
        assert_eq!(encode_e4m3(f32::INFINITY), 0x7e);
        assert_eq!(encode_e4m3(f32::NEG_INFINITY), 0xfe);
        assert_eq!(encode_e4m3(0.0), 0x00);
        assert_eq!(encode_e4m3(-0.0), 0x80);
        assert_eq!(encode_e4m3(0.001_953_125), 0x01);
        assert_eq!(encode_e4m3(0.000_976_562_5), 0x00);
        assert_eq!(encode_e4m3(0.002_929_687_5), 0x02);
        assert_eq!(encode_e4m3(0.015_625), 0x08);
    }

    #[test]
    fn div_rn_matches_ieee_division() {
        let mut worst = 0u32;
        for i in 1..20000u32 {
            let a = i as f32 * 0.37;
            let b = 448.0f32;
            assert_eq!(div_rn(a, b).to_bits(), (a / b).to_bits());
            let d = div_rn(448.0, a);
            worst = worst
                .max((d.to_bits() as i64 - (448.0f32 / a).to_bits() as i64).unsigned_abs() as u32);
        }
        assert_eq!(worst, 0);
    }

    #[test]
    fn slot_bounds_are_checked() {
        assert!(slot_count(10, 4, "x").is_err());
        assert_eq!(slot_count(12, 4, "x").unwrap(), 3);
        assert_eq!(max_slot_touched(3, 2, 0), 4);
        assert_eq!(max_slot_touched(3, 2, 4), 3);
    }

    #[test]
    fn word_packing_round_trips() {
        let src: Vec<u16> = (0u16..32).map(|i| i.wrapping_mul(2731)).collect();
        let words = pack_bf16(&src);
        let mut back = vec![0u16; src.len()];
        unpack_bf16(&words, &mut back);
        assert_eq!(back, src);

        let bytes: Vec<u8> = (0u8..16).collect();
        let w = bytes_to_words(&bytes);
        let mut rb = vec![0u8; bytes.len()];
        words_to_bytes(&w, &mut rb);
        assert_eq!(rb, bytes);
    }
}
