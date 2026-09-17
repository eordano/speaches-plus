pub const MLX_AFFINE_QUANTIZE_REFITS_THE_SCALE_SO_EXACT_ZERO_IS_ON_THE_GRID: &str =
    "Port of mlx affine_quantize (metal/kernels/quantized.h), verified bit-identical against \
     mx.quantize on the rednote dots.ocr master. Per group: w_min over the group but w_max \
     seeded with 0.0 (their init, not an accident to fix); scale = max((w_max-w_min)/15, 1e-7) \
     with its SIGN set negative when |w_max| >= |w_min|; then the refit that distinguishes this \
     from naive min-max: edge = the larger-magnitude end, q0 = round(edge/scale), and unless q0 \
     is zero the scale becomes edge/q0 with bias = edge -- which places exact 0.0 on the code \
     grid at -q0. Codes use the REFIT f32 scale, never the bf16-rounded stored one; scales and \
     biases are stored bf16-rounded because the checkpoint stores bf16. Rounding is \
     half-away-from-zero on both sides (metal round == rust f32::round). Nibbles pack LSB-first \
     into u32 words.";

pub fn quantize_mlx_affine_linear(
    w: &[f32],
    n: usize,
    k: usize,
    group_size: usize,
) -> (Vec<u32>, Vec<f32>, Vec<f32>) {
    let _ = MLX_AFFINE_QUANTIZE_REFITS_THE_SCALE_SO_EXACT_ZERO_IS_ON_THE_GRID;
    assert_eq!(w.len(), n * k, "weights must be n*k");
    assert_eq!(k % group_size, 0, "k must be divisible by group_size");
    assert_eq!(group_size % 8, 0, "group_size must pack whole u32 words");
    let groups = k / group_size;
    let n_bins = 15f32;
    let mut words = vec![0u32; n * k / 8];
    let mut scales = vec![0f32; n * groups];
    let mut biases = vec![0f32; n * groups];
    for row in 0..n {
        for g in 0..groups {
            let grp = &w[row * k + g * group_size..row * k + (g + 1) * group_size];
            let mut w_min = f32::MAX;
            let mut w_max = 0f32;
            for &v in grp {
                w_min = w_min.min(v);
                w_max = w_max.max(v);
            }
            let mut scale = ((w_max - w_min) / n_bins).max(1e-7);
            let side = w_min.abs() > w_max.abs();
            if !side {
                scale = -scale;
            }
            let edge = if side { w_min } else { w_max };
            let q0 = (edge / scale).round();
            let bias = if q0 == 0.0 {
                0.0
            } else {
                scale = edge / q0;
                edge
            };
            scales[row * groups + g] = half::bf16::from_f32(scale).to_f32();
            biases[row * groups + g] = half::bf16::from_f32(bias).to_f32();
            for (i, &v) in grp.iter().enumerate() {
                let code = (((v - bias) / scale).round().min(n_bins).max(0.0)) as u32;
                let k_idx = g * group_size + i;
                words[row * (k / 8) + k_idx / 8] |= (code & 0xF) << (4 * (k_idx % 8));
            }
        }
    }
    (words, scales, biases)
}

pub fn dequantize_mlx_affine_linear(
    words: &[u32],
    scales_f32: &[f32],
    biases_f32: &[f32],
    n: usize,
    k: usize,
    group_size: usize,
) -> Vec<f32> {
    assert_eq!(words.len(), n * k / 8, "packed words must be n*k/8");
    let groups = k / group_size;
    assert_eq!(scales_f32.len(), n * groups, "scales must be n*k/gs");
    assert_eq!(biases_f32.len(), n * groups, "biases must be n*k/gs");
    let mut out = vec![0f32; n * k];
    for row in 0..n {
        for kw in 0..k / 8 {
            let word = words[row * (k / 8) + kw];
            for e in 0..8 {
                let k_idx = kw * 8 + e;
                let g = k_idx / group_size;
                let q = ((word >> (4 * e)) & 0xF) as f32;
                out[row * k + k_idx] =
                    q * scales_f32[row * groups + g] + biases_f32[row * groups + g];
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lsb_first_affine_dequant_matches_the_mlx_reference_fixture() {
        let words: [u32; 16] = [
            1586079062u32,
            1459068087u32,
            2844571786u32,
            2580515947u32,
            68630393u32,
            3121869188u32,
            1821596007u32,
            3852874583u32,
            2106549864u32,
            2038215078u32,
            2058705742u32,
            1332722316u32,
            3094346412u32,
            1448692865u32,
            2882901996u32,
            124238845u32,
        ];
        let scales = [0x3b570000, 0xbb420000].map(f32::from_bits);
        let biases = [0xbcd70000, 0x3cda0000].map(f32::from_bits);
        let want = [
            0xbbd70000, 0xbc210000, 0xbc210000, 0x3bd70000, 0x3b570000, 0x00000000, 0x3ca10000,
            0xbc210000, 0xbb570000, 0x3c210000, 0x3c570000, 0x3b570000, 0xbb570000, 0x3cbc0000,
            0xbbd70000, 0xbc210000, 0x3bd70000, 0x00000000, 0x00000000, 0x3c210000, 0x3c570000,
            0x00000000, 0x3b570000, 0x3bd70000, 0x3c210000, 0xbbd70000, 0x3c570000, 0x00000000,
            0x3cbc0000, 0x3c570000, 0x3b570000, 0x3b570000, 0x3b570000, 0xbb570000, 0xbb570000,
            0xbc860000, 0xbb570000, 0xbcbc0000, 0xbc570000, 0xbcd70000, 0xbc570000, 0x00000000,
            0xbcbc0000, 0x3cbc0000, 0xbc860000, 0xbcbc0000, 0x3bd70000, 0x3c210000, 0xbb570000,
            0xbbd70000, 0x3b570000, 0xbc210000, 0xbc860000, 0x3b570000, 0x3c570000, 0xbbd70000,
            0xbb570000, 0xbc210000, 0xbc860000, 0xbc860000, 0xbbd70000, 0x3bd70000, 0xbc210000,
            0x3ca10000, 0x3b400000, 0x3c110000, 0x3c110000, 0x3c110000, 0xbc920000, 0x3b400000,
            0xbc420000, 0x3bc10000, 0x3c110000, 0xbb440000, 0x3cc20000, 0xbbc30000, 0xbc120000,
            0x3bc10000, 0xb8000000, 0x3bc10000, 0xbc730000, 0x3c720000, 0xbbc30000, 0x3c420000,
            0x3c420000, 0xbbc30000, 0xbb440000, 0x3bc10000, 0xbc120000, 0x3b400000, 0xbb440000,
            0xbbc30000, 0xbc920000, 0x3c110000, 0xbc920000, 0x3c720000, 0xbc120000, 0xbb440000,
            0xbb440000, 0xbc920000, 0xbc920000, 0x3c110000, 0x3b400000, 0xbbc30000, 0x3cc20000,
            0x3b400000, 0xbc120000, 0x3c720000, 0xb8000000, 0x3c420000, 0x3c110000, 0x3c420000,
            0xbc120000, 0xbc730000, 0x3bc10000, 0xb8000000, 0x3c420000, 0xbc420000, 0xbbc30000,
            0xbb440000, 0xbc420000, 0xbc920000, 0xbbc30000, 0xbbc30000, 0x3bc10000, 0x3c110000,
            0x3bc10000, 0x3cda0000,
        ]
        .map(f32::from_bits);
        let got = dequantize_mlx_affine_linear(&words, &scales, &biases, 1, 128, 64);
        for (i, (g, w)) in got.iter().zip(want.iter()).enumerate() {
            let ulp = (w.abs().max(f32::MIN_POSITIVE)) / 256.0;
            assert!(
                (g - w).abs() <= ulp,
                "element {i}: got {g}, mlx reference {w}, diff {} > 1 bf16 ulp {ulp} -- \
                 mx.dequantize rounds bf16 TWICE (q*scale, then +bias) while ours is one f32 \
                 fma, so agreement is within-1-bf16-ulp by construction, never bitwise \
                 (fixture: mlx-community/Qwen3.8-27B-4bit layers.0.mlp.gate_proj row 0)",
                (g - w).abs()
            );
        }
    }
}
