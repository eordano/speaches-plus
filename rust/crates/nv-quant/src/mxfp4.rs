pub use crate::nvfp4::{decode_e2m1, encode_e2m1, pack_e2m1_pair, unpack_e2m1_pair};

pub const BLOCK_SIZE: usize = 32;
pub const BLOCK_BYTES: usize = 16;
pub const E8M0_BIAS: i32 = 127;
pub const E8M0_NAN: u8 = 0xFF;
pub const E2M1_MAX_EXP: i32 = 2;
pub const E2M1_MAX: f32 = 6.0;

pub fn exp2_i32(e: i32) -> f32 {
    if e >= -126 {
        f32::from_bits(((e + 127) as u32) << 23)
    } else if e >= -149 {
        f32::from_bits(1u32 << (e + 149))
    } else {
        0.0
    }
}

pub fn ilog2_f32(x: f32) -> i32 {
    let bits = x.to_bits() & 0x7FFF_FFFF;
    if bits == 0 {
        return i32::MIN;
    }
    let exp = (bits >> 23) as i32;
    if exp != 0 {
        exp - 127
    } else {
        31 - bits.leading_zeros() as i32 - 149
    }
}

pub fn decode_e8m0(byte: u8) -> f32 {
    if byte == E8M0_NAN {
        return f32::NAN;
    }
    exp2_i32(byte as i32 - E8M0_BIAS)
}

pub fn encode_e8m0(scale: f32) -> u8 {
    if !scale.is_finite() || scale <= 0.0 {
        return E8M0_NAN;
    }
    (ilog2_f32(scale) + E8M0_BIAS).clamp(0, 254) as u8
}

pub fn block_scale_byte(amax: f32) -> u8 {
    if amax == 0.0 || !amax.is_finite() {
        return E8M0_BIAS as u8;
    }
    (ilog2_f32(amax) - E2M1_MAX_EXP + E8M0_BIAS).clamp(0, 254) as u8
}

pub fn quantize_block(values: &[f32]) -> ([u8; BLOCK_BYTES], u8) {
    assert_eq!(values.len(), BLOCK_SIZE);
    let amax = values
        .iter()
        .filter(|v| v.is_finite())
        .fold(0f32, |a, b| a.max(b.abs()));
    let scale_byte = block_scale_byte(amax);
    let inv = exp2_i32(E8M0_BIAS - scale_byte as i32);
    let mut packed = [0u8; BLOCK_BYTES];
    for (i, pair) in values.chunks(2).enumerate() {
        let lo = encode_e2m1((pair[0] * inv).clamp(-E2M1_MAX, E2M1_MAX));
        let hi = encode_e2m1((pair[1] * inv).clamp(-E2M1_MAX, E2M1_MAX));
        packed[i] = pack_e2m1_pair(lo, hi);
    }
    (packed, scale_byte)
}

pub fn dequantize_block(packed: &[u8], scale_byte: u8) -> Vec<f32> {
    assert_eq!(packed.len(), BLOCK_BYTES);
    let scale = decode_e8m0(scale_byte);
    let mut out = Vec::with_capacity(BLOCK_SIZE);
    for byte in packed {
        let (lo, hi) = unpack_e2m1_pair(*byte);
        out.push(decode_e2m1(lo) * scale);
        out.push(decode_e2m1(hi) * scale);
    }
    out
}

pub struct Mxfp4Tensor {
    pub data: Vec<u8>,
    pub scales: Vec<u8>,
    pub rows: usize,
    pub cols: usize,
}

impl Mxfp4Tensor {
    pub fn quantize_rows(rows: &[Vec<f32>]) -> Self {
        let rows_n = rows.len();
        let cols = rows[0].len();
        assert!(
            cols.is_multiple_of(BLOCK_SIZE),
            "cols must be a multiple of {BLOCK_SIZE}"
        );
        let blocks_per_row = cols / BLOCK_SIZE;
        let mut data = Vec::with_capacity(rows_n * cols / 2);
        let mut scales = Vec::with_capacity(rows_n * blocks_per_row);
        for row in rows {
            for block in row.chunks(BLOCK_SIZE) {
                let (packed, scale) = quantize_block(block);
                data.extend_from_slice(&packed);
                scales.push(scale);
            }
        }
        Self {
            data,
            scales,
            rows: rows_n,
            cols,
        }
    }

    pub fn from_gpt_oss_row_major(blocks: &[u8], scales: &[u8], rows: usize, cols: usize) -> Self {
        assert!(cols.is_multiple_of(BLOCK_SIZE));
        let blocks_per_row = cols / BLOCK_SIZE;
        assert_eq!(blocks.len(), rows * blocks_per_row * BLOCK_BYTES);
        assert_eq!(scales.len(), rows * blocks_per_row);
        Self {
            data: blocks.to_vec(),
            scales: scales.to_vec(),
            rows,
            cols,
        }
    }

    pub fn dequantize(&self) -> Vec<Vec<f32>> {
        let blocks_per_row = self.cols / BLOCK_SIZE;
        let bytes_per_row = self.cols / 2;
        let mut out = Vec::with_capacity(self.rows);
        for r in 0..self.rows {
            let row_bytes = &self.data[r * bytes_per_row..(r + 1) * bytes_per_row];
            let row_scales = &self.scales[r * blocks_per_row..(r + 1) * blocks_per_row];
            let mut row = Vec::with_capacity(self.cols);
            for (b, scale_byte) in row_scales.iter().enumerate() {
                let packed = &row_bytes[b * BLOCK_BYTES..(b + 1) * BLOCK_BYTES];
                row.extend_from_slice(&dequantize_block(packed, *scale_byte));
            }
            out.push(row);
        }
        out
    }
}

pub fn cpu_mxfp4_matmul_weight_row(x: &[f32], w: &Mxfp4Tensor, m: usize) -> Vec<f32> {
    assert_eq!(x.len(), m * w.cols);
    let w_deq = w.dequantize();
    let mut d = vec![0f32; m * w.rows];
    for i in 0..m {
        let xi = &x[i * w.cols..(i + 1) * w.cols];
        for (j, wj) in w_deq.iter().enumerate() {
            let mut acc = 0f32;
            for p in 0..w.cols {
                acc += xi[p] * wj[p];
            }
            d[i * w.rows + j] = acc;
        }
    }
    d
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exp2_doubles_exactly_from_the_smallest_subnormal_to_max_exponent() {
        assert_eq!(exp2_i32(0), 1.0);
        assert_eq!(exp2_i32(-149), f32::from_bits(1));
        for e in -148..=127 {
            assert_eq!(exp2_i32(e), 2.0 * exp2_i32(e - 1), "e={e}");
        }
    }

    #[test]
    fn exp2_underflows_to_zero_below_the_smallest_subnormal() {
        assert_eq!(exp2_i32(-150), 0.0);
        assert_eq!(exp2_i32(-1000), 0.0);
        assert_eq!(exp2_i32(i32::MIN), 0.0);
    }

    #[test]
    fn ilog2_floors_normals_and_subnormals() {
        assert_eq!(ilog2_f32(1.0), 0);
        assert_eq!(ilog2_f32(1.5), 0);
        assert_eq!(ilog2_f32(2.0), 1);
        assert_eq!(ilog2_f32(6.0), 2);
        assert_eq!(ilog2_f32(0.5), -1);
        assert_eq!(ilog2_f32(-4.0), 2);
        assert_eq!(ilog2_f32(f32::from_bits(1)), -149);
        assert_eq!(ilog2_f32(f32::from_bits((1 << 9) | (1 << 8))), -140);
    }

    #[test]
    fn ilog2_of_zero_is_a_sentinel_not_a_shiftable_exponent() {
        assert_eq!(ilog2_f32(0.0), i32::MIN);
        assert_eq!(ilog2_f32(-0.0), i32::MIN);
        assert_eq!(exp2_i32(ilog2_f32(0.0)), 0.0);
    }

    #[test]
    fn e8m0_roundtrips_every_non_nan_byte() {
        for byte in 0u8..=254 {
            let v = decode_e8m0(byte);
            assert!(v > 0.0 && v.is_finite(), "byte {byte}");
            assert_eq!(encode_e8m0(v), byte, "byte {byte}");
        }
    }

    #[test]
    fn e8m0_nan_byte_decodes_nan_and_bad_scales_encode_nan_byte() {
        assert!(decode_e8m0(E8M0_NAN).is_nan());
        assert_eq!(encode_e8m0(0.0), E8M0_NAN);
        assert_eq!(encode_e8m0(-1.0), E8M0_NAN);
        assert_eq!(encode_e8m0(f32::NAN), E8M0_NAN);
        assert_eq!(encode_e8m0(f32::INFINITY), E8M0_NAN);
    }

    #[test]
    fn e8m0_encode_clamps_to_byte_bounds() {
        assert_eq!(encode_e8m0(f32::from_bits(1)), 0);
        assert_eq!(encode_e8m0(2f32.powi(127)), 254);
        assert_eq!(encode_e8m0(f32::MAX), 254);
    }

    #[test]
    fn block_scale_puts_amax_in_the_e2m1_range() {
        assert_eq!(block_scale_byte(0.0), E8M0_BIAS as u8);
        assert_eq!(block_scale_byte(f32::INFINITY), E8M0_BIAS as u8);
        assert_eq!(block_scale_byte(4.0), E8M0_BIAS as u8);
        assert_eq!(block_scale_byte(1.0), (E8M0_BIAS - 2) as u8);
        for amax in [0.03125f32, 0.9, 1.0, 3.7, 64.0, 1e30] {
            let scale = decode_e8m0(block_scale_byte(amax));
            let ratio = amax / scale;
            assert!(
                (4.0..8.0).contains(&ratio),
                "amax {amax}: ratio {ratio} outside [4, 8)"
            );
        }
    }

    #[test]
    fn exactly_representable_block_roundtrips_bit_exact() {
        let mut values = vec![0.0f32; BLOCK_SIZE];
        let exact = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0];
        for (i, v) in values.iter_mut().enumerate() {
            *v = exact[i % exact.len()] * if i % 2 == 0 { 1.0 } else { -1.0 };
        }
        let (packed, scale_byte) = quantize_block(&values);
        assert_eq!(decode_e8m0(scale_byte), 1.0);
        assert_eq!(dequantize_block(&packed, scale_byte), values);
    }

    #[test]
    fn power_of_two_amax_block_roundtrips_within_half_a_step() {
        let values: Vec<f32> = (0..BLOCK_SIZE)
            .map(|i| (i as f32 / (BLOCK_SIZE - 1) as f32) * 8.0 - 4.0)
            .collect();
        let (packed, scale_byte) = quantize_block(&values);
        let scale = decode_e8m0(scale_byte);
        for (v, d) in values.iter().zip(dequantize_block(&packed, scale_byte)) {
            assert!(
                (v - d).abs() <= 0.5 * scale + 1e-6,
                "v={v} deq={d} scale={scale}"
            );
        }
    }

    #[test]
    fn general_block_roundtrip_error_is_bounded_by_the_clamp_gap() {
        let values: Vec<f32> = (0..BLOCK_SIZE)
            .map(|i| ((i as f32 * 0.7311 + 0.17).sin()) * 7.9)
            .collect();
        let (packed, scale_byte) = quantize_block(&values);
        let scale = decode_e8m0(scale_byte);
        for (v, d) in values.iter().zip(dequantize_block(&packed, scale_byte)) {
            assert!(
                (v - d).abs() <= 2.0 * scale + 1e-6,
                "v={v} deq={d} scale={scale}"
            );
        }
    }

    #[test]
    fn all_zero_block_stays_zero_with_the_neutral_scale() {
        let (packed, scale_byte) = quantize_block(&[0.0f32; BLOCK_SIZE]);
        assert_eq!(scale_byte, E8M0_BIAS as u8);
        assert!(dequantize_block(&packed, scale_byte)
            .iter()
            .all(|&v| v == 0.0));
    }

    #[test]
    fn non_finite_values_are_ignored_for_scaling_and_clamped_or_zeroed() {
        let mut values = vec![1.0f32; BLOCK_SIZE];
        values[0] = f32::INFINITY;
        values[1] = f32::NEG_INFINITY;
        values[2] = f32::NAN;
        let (packed, scale_byte) = quantize_block(&values);
        let scale = decode_e8m0(scale_byte);
        assert!((4.0..8.0).contains(&(1.0 / scale)));
        let deq = dequantize_block(&packed, scale_byte);
        assert_eq!(deq[0], E2M1_MAX * scale);
        assert_eq!(deq[1], -E2M1_MAX * scale);
        assert_eq!(deq[2], 0.0);
        assert!(deq.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn all_non_finite_block_dequantizes_finite() {
        let values = vec![f32::INFINITY; BLOCK_SIZE];
        let (packed, scale_byte) = quantize_block(&values);
        assert_eq!(scale_byte, E8M0_BIAS as u8);
        assert!(dequantize_block(&packed, scale_byte)
            .iter()
            .all(|v| v.is_finite()));
    }

    fn ramp_rows(rows: usize, cols: usize) -> Vec<Vec<f32>> {
        (0..rows)
            .map(|r| {
                (0..cols)
                    .map(|c| ((r * cols + c) as f32 * 0.911).cos() * 5.0)
                    .collect()
            })
            .collect()
    }

    #[test]
    fn gpt_oss_row_major_layout_roundtrips_through_quantize_rows() {
        let rows = ramp_rows(3, 2 * BLOCK_SIZE);
        let t = Mxfp4Tensor::quantize_rows(&rows);
        let rebuilt = Mxfp4Tensor::from_gpt_oss_row_major(&t.data, &t.scales, t.rows, t.cols);
        assert_eq!(rebuilt.rows, 3);
        assert_eq!(rebuilt.cols, 2 * BLOCK_SIZE);
        assert_eq!(rebuilt.dequantize(), t.dequantize());
    }

    #[test]
    #[should_panic]
    fn gpt_oss_layout_rejects_wrong_block_byte_count() {
        let _ = Mxfp4Tensor::from_gpt_oss_row_major(&[0u8; BLOCK_BYTES], &[0u8; 2], 2, BLOCK_SIZE);
    }

    #[test]
    #[should_panic]
    fn gpt_oss_layout_rejects_wrong_scale_count() {
        let _ =
            Mxfp4Tensor::from_gpt_oss_row_major(&[0u8; 2 * BLOCK_BYTES], &[0u8; 1], 2, BLOCK_SIZE);
    }

    #[test]
    fn cpu_matmul_matches_exact_dot_products() {
        let exact = [0.5f32, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 0.0];
        let w_rows: Vec<Vec<f32>> = (0..2)
            .map(|r| {
                (0..BLOCK_SIZE)
                    .map(|c| exact[(r + c) % exact.len()])
                    .collect()
            })
            .collect();
        let w = Mxfp4Tensor::quantize_rows(&w_rows);
        let x: Vec<f32> = (0..BLOCK_SIZE).map(|c| (c % 3) as f32).collect();
        let d = cpu_mxfp4_matmul_weight_row(&x, &w, 1);
        for (j, row) in w_rows.iter().enumerate() {
            let want: f32 = row.iter().zip(&x).map(|(a, b)| a * b).sum();
            assert!((d[j] - want).abs() < 1e-4, "row {j}: {} vs {want}", d[j]);
        }
    }
}
