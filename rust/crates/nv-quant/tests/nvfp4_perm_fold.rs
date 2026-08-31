use nv_quant::nvfp4::{Nvfp4Tensor, BLOCK_SIZE};

const A_PERMUTATION_OF_THE_SWIGLU_INTERMEDIATE_AXIS_FOLDS_INTO_GATE_ROWS_UP_ROWS_AND_DOWN_COLUMNS_AND_NEEDS_NO_RUNTIME_TRANSFORM: usize =
    64;
const HIDDEN_IS_SMALL_BECAUSE_THE_FOLD_IS_ALGEBRAIC_AND_SIZE_ADDS_NOTHING: usize = 8;
const FOLDING_A_PERMUTATION_REORDERS_THE_DOWN_PROJ_CONTRACTION_SO_THE_OUTPUT_IS_EXACT_IN_EXACT_ARITHMETIC_AND_REASSOCIATION_SCALE_IN_FLOAT: f64 =
    1e-12;
const ONLY_MONOMIAL_MATRICES_COMMUTE_WITH_THE_ELEMENTWISE_SWIGLU_PRODUCT_SO_A_GENERAL_ROTATION_MUST_MOVE_THE_OUTPUT_BY_WHOLE_UNITS: f64 =
    1e-3;
const AN_ELEMENT_GRANULAR_PERMUTATION_MUST_MOVE_A_LARGE_MINORITY_OF_THE_BLOCK_SCALES_OR_IT_IS_NOT_ENGAGING_THE_QUANTIZER: f64 =
    0.25;

struct Xorshift(u64);

impl Xorshift {
    fn next_f32(&mut self) -> f32 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        ((self.0 >> 40) as f32 / 8_388_608.0) - 1.0
    }
    fn next_usize(&mut self, n: usize) -> usize {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        (self.0 >> 16) as usize % n
    }
}

fn matrix(rng: &mut Xorshift, rows: usize, cols: usize, spread: f32) -> Vec<Vec<f32>> {
    (0..rows)
        .map(|r| {
            let row_gain = if r % 37 == 0 { spread } else { 1.0 };
            (0..cols)
                .map(|c| {
                    let col_gain = if c % 53 == 0 { spread } else { 1.0 };
                    rng.next_f32() * row_gain * col_gain
                })
                .collect()
        })
        .collect()
}

fn permute_rows(w: &[Vec<f32>], perm: &[usize]) -> Vec<Vec<f32>> {
    perm.iter().map(|&j| w[j].clone()).collect()
}

fn permute_cols(w: &[Vec<f32>], perm: &[usize]) -> Vec<Vec<f32>> {
    w.iter()
        .map(|row| perm.iter().map(|&j| row[j]).collect())
        .collect()
}

fn matvec(w: &[Vec<f32>], x: &[f32]) -> Vec<f64> {
    w.iter()
        .map(|row| {
            row.iter()
                .zip(x)
                .map(|(&a, &b)| a as f64 * b as f64)
                .sum::<f64>()
        })
        .collect()
}

fn silu(z: f64) -> f64 {
    z / (1.0 + (-z).exp())
}

fn swiglu_mlp(gate: &[Vec<f32>], up: &[Vec<f32>], down: &[Vec<f32>], x: &[f32]) -> Vec<f64> {
    let g = matvec(gate, x);
    let u = matvec(up, x);
    let h: Vec<f32> = g
        .iter()
        .zip(&u)
        .map(|(&a, &b)| (silu(a) * b) as f32)
        .collect();
    matvec(down, &h)
}

fn identity_perm(n: usize) -> Vec<usize> {
    (0..n).collect()
}

fn shuffled(mut perm: Vec<usize>, rng: &mut Xorshift) -> Vec<usize> {
    for i in (1..perm.len()).rev() {
        let j = rng.next_usize(i + 1);
        perm.swap(i, j);
    }
    perm
}

fn block_granular_perm(inter: usize, rng: &mut Xorshift) -> Vec<usize> {
    let blocks = shuffled(identity_perm(inter / BLOCK_SIZE), rng);
    blocks
        .into_iter()
        .flat_map(|b| (0..BLOCK_SIZE).map(move |k| b * BLOCK_SIZE + k))
        .collect()
}

fn sorted_by_column_rms(down: &[Vec<f32>]) -> Vec<usize> {
    let cols = down[0].len();
    let mut keyed: Vec<(usize, f64)> = (0..cols)
        .map(|c| {
            let s: f64 = down.iter().map(|r| (r[c] as f64).powi(2)).sum();
            (c, (s / down.len() as f64).sqrt())
        })
        .collect();
    keyed.sort_by(|a, b| a.1.partial_cmp(&b.1).expect("finite key"));
    keyed.into_iter().map(|(c, _)| c).collect()
}

fn mean_within_block_amax_over_rms(w: &[Vec<f32>]) -> f64 {
    let mut acc = 0.0;
    let mut n = 0usize;
    for row in w {
        for block in row.chunks(BLOCK_SIZE) {
            let amax = block.iter().fold(0f64, |a, &v| a.max(v.abs() as f64));
            let rms =
                (block.iter().map(|&v| (v as f64).powi(2)).sum::<f64>() / block.len() as f64).sqrt();
            if rms > 0.0 {
                acc += amax / rms;
                n += 1;
            }
        }
    }
    acc / n as f64
}

fn stored_global(w: &[Vec<f32>]) -> f32 {
    let amax = w
        .iter()
        .flat_map(|r| r.iter())
        .fold(0f32, |a, &v| a.max(v.abs()));
    448.0 * 6.0 / amax
}

fn block_diagonal_hadamard(inter: usize) -> Vec<Vec<f32>> {
    let mut h = vec![vec![0f32; inter]; inter];
    let scale = 1.0 / (BLOCK_SIZE as f32).sqrt();
    for base in (0..inter).step_by(BLOCK_SIZE) {
        for i in 0..BLOCK_SIZE {
            for j in 0..BLOCK_SIZE {
                let sign = if ((i & j).count_ones() & 1) == 1 {
                    -1.0
                } else {
                    1.0
                };
                h[base + i][base + j] = sign * scale;
            }
        }
    }
    h
}

fn left_multiply_transpose(r: &[Vec<f32>], w: &[Vec<f32>]) -> Vec<Vec<f32>> {
    let inter = r.len();
    let cols = w[0].len();
    (0..inter)
        .map(|i| {
            (0..cols)
                .map(|c| (0..inter).map(|k| r[k][i] * w[k][c]).sum())
                .collect()
        })
        .collect()
}

fn right_multiply(w: &[Vec<f32>], r: &[Vec<f32>]) -> Vec<Vec<f32>> {
    let inter = r.len();
    w.iter()
        .map(|row| {
            (0..inter)
                .map(|j| (0..inter).map(|k| row[k] * r[k][j]).sum())
                .collect()
        })
        .collect()
}

fn max_relative_delta(a: &[f64], b: &[f64]) -> f64 {
    let norm = a.iter().fold(0f64, |m, &v| m.max(v.abs())).max(1e-30);
    a.iter()
        .zip(b)
        .fold(0f64, |m, (&x, &y)| m.max((x - y).abs()))
        / norm
}

fn fixture(rng: &mut Xorshift) -> (Vec<Vec<f32>>, Vec<Vec<f32>>, Vec<Vec<f32>>, Vec<f32>) {
    let inter =
        A_PERMUTATION_OF_THE_SWIGLU_INTERMEDIATE_AXIS_FOLDS_INTO_GATE_ROWS_UP_ROWS_AND_DOWN_COLUMNS_AND_NEEDS_NO_RUNTIME_TRANSFORM;
    let hidden = HIDDEN_IS_SMALL_BECAUSE_THE_FOLD_IS_ALGEBRAIC_AND_SIZE_ADDS_NOTHING;
    let gate = matrix(rng, inter, hidden, 8.0);
    let up = matrix(rng, inter, hidden, 8.0);
    let down = matrix(rng, hidden, inter, 12.0);
    let x: Vec<f32> = (0..hidden).map(|_| rng.next_f32()).collect();
    (gate, up, down, x)
}

#[test]
fn folding_an_intermediate_axis_permutation_into_gate_rows_up_rows_and_down_columns_leaves_the_swiglu_mlp_output_unchanged(
) {
    let mut rng = Xorshift(0x9E37_79B9_7F4A_7C15);
    let (gate, up, down, x) = fixture(&mut rng);
    let reference = swiglu_mlp(&gate, &up, &down, &x);
    for trial in 0..8 {
        let perm = shuffled(identity_perm(down[0].len()), &mut rng);
        let folded = swiglu_mlp(
            &permute_rows(&gate, &perm),
            &permute_rows(&up, &perm),
            &permute_cols(&down, &perm),
            &x,
        );
        let rel = max_relative_delta(&reference, &folded);
        assert!(
            rel < FOLDING_A_PERMUTATION_REORDERS_THE_DOWN_PROJ_CONTRACTION_SO_THE_OUTPUT_IS_EXACT_IN_EXACT_ARITHMETIC_AND_REASSOCIATION_SCALE_IN_FLOAT,
            "trial {trial}: folding a permutation of the intermediate axis moved the MLP output \
             by {rel:e} relative; the fold is exact and only the contraction order may change"
        );
    }
}

#[test]
fn a_general_rotation_of_the_intermediate_axis_does_not_fold_because_silu_of_gate_times_up_is_elementwise(
) {
    let mut rng = Xorshift(0xD1B5_4A32_D192_ED03);
    let (gate, up, down, x) = fixture(&mut rng);
    let reference = swiglu_mlp(&gate, &up, &down, &x);
    let h = block_diagonal_hadamard(down[0].len());
    let rotated = swiglu_mlp(
        &left_multiply_transpose(&h, &gate),
        &left_multiply_transpose(&h, &up),
        &right_multiply(&down, &h),
        &x,
    );
    let rel = max_relative_delta(&reference, &rotated);
    assert!(
        rel > ONLY_MONOMIAL_MATRICES_COMMUTE_WITH_THE_ELEMENTWISE_SWIGLU_PRODUCT_SO_A_GENERAL_ROTATION_MUST_MOVE_THE_OUTPUT_BY_WHOLE_UNITS,
        "a block-diagonal Hadamard folded the same way a permutation folds moved the MLP output \
         by only {rel:e} relative; if a general rotation folded exactly then rotation FP4 would \
         need no online transform, and it does"
    );
}

#[test]
fn a_block_granular_permutation_only_relabels_the_nvfp4_bytes_so_it_is_a_reassociation_null_not_a_quantization_change(
) {
    let mut rng = Xorshift(0x2545_F491_4F6C_DD1D);
    let (_, _, down, _) = fixture(&mut rng);
    let g = stored_global(&down);
    let base = Nvfp4Tensor::quantize_rows_with_global(&down, g);
    for trial in 0..8 {
        let perm = block_granular_perm(down[0].len(), &mut rng);
        let permuted = permute_cols(&down, &perm);
        assert_eq!(
            stored_global(&permuted),
            g,
            "trial {trial}: the tensor amax is permutation invariant so the stored global scale \
             must not move"
        );
        let arm = Nvfp4Tensor::quantize_rows_with_global(&permuted, g);
        let blocks_per_row = down[0].len() / BLOCK_SIZE;
        let block_order: Vec<usize> = (0..blocks_per_row)
            .map(|b| perm[b * BLOCK_SIZE] / BLOCK_SIZE)
            .collect();
        for row in 0..down.len() {
            for (dst, &src) in block_order.iter().enumerate() {
                assert_eq!(
                    arm.scales[row * blocks_per_row + dst],
                    base.scales[row * blocks_per_row + src],
                    "trial {trial} row {row} block {dst}: a block-granular permutation keeps every \
                     block's 16 members so its scale byte must be the relabelled original"
                );
                let (a, b) = (
                    (row * blocks_per_row + dst) * (BLOCK_SIZE / 2),
                    (row * blocks_per_row + src) * (BLOCK_SIZE / 2),
                );
                assert_eq!(
                    arm.data[a..a + BLOCK_SIZE / 2],
                    base.data[b..b + BLOCK_SIZE / 2],
                    "trial {trial} row {row} block {dst}: packed nibbles must be the relabelled \
                     original or the null is not a null"
                );
            }
        }
    }
}

#[test]
fn an_element_granular_permutation_regroups_the_nvfp4_blocks_and_moves_the_bytes_which_is_the_arms_engagement_proof(
) {
    let mut rng = Xorshift(0x6C07_8965_0F1B_2A3D);
    let (_, _, down, _) = fixture(&mut rng);
    let g = stored_global(&down);
    let base = Nvfp4Tensor::quantize_rows_with_global(&down, g);
    let perm = sorted_by_column_rms(&down);
    let permuted = permute_cols(&down, &perm);
    let arm = Nvfp4Tensor::quantize_rows_with_global(&permuted, g);
    let moved = arm
        .scales
        .iter()
        .zip(&base.scales)
        .filter(|(a, b)| a != b)
        .count() as f64
        / base.scales.len() as f64;
    assert!(
        moved > AN_ELEMENT_GRANULAR_PERMUTATION_MUST_MOVE_A_LARGE_MINORITY_OF_THE_BLOCK_SCALES_OR_IT_IS_NOT_ENGAGING_THE_QUANTIZER,
        "a magnitude-sorted regrouping moved only {moved:.4} of the block scales; a row whose \
         bytes barely move is a checkpoint that never took the arm"
    );
    let before = mean_within_block_amax_over_rms(&down);
    let after = mean_within_block_amax_over_rms(&permuted);
    assert!(
        after < before,
        "magnitude-sorted grouping raised mean within-block amax/rms {before:.4} -> {after:.4}; \
         that ratio is this arm's engagement metric and never its quality evidence, because \
         weight reconstruction error does not order perplexity on this checkpoint family"
    );
}
