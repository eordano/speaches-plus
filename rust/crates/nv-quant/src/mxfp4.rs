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
