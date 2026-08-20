#![cfg(feature = "wgpu")]

mod common;
use common::wgpu_allow_skip;
use half::bf16;
use nv_kernels::wgpu_backend::device::WgpuContext;
use nv_kernels::wgpu_backend::{compose, dispatch};
use common::LcgOddSeedShift32F64TwoSided as Lcg;
use common::ctx_or_skip_quiet_unqualified as ctx_or_skip;

const Q3D_GEMV_WGSL: &str = include_str!("../wgsl/q3d_gemv_bf16.wgsl");
const ENTRY: &str = "q3w_gemv_fp8_rowscale";

const KERNEL_VS_HOST_ORACLE_REL_TOL_PINNED_BY_GEMV_W4A16_SUITES: f32 = 1e-2;

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, bytemuck::Pod, bytemuck::Zeroable)]
struct Q3bParams {
    n_rows: u32,
    k_words: u32,
    groups_x: u32,
    out_f32: u32,
    w_row_words: u32,
    x_off_words: u32,
    y_off_words: u32,
    pad0: u32,
    alpha: f32,
    pad1: u32,
    pad2: u32,
    pad3: u32,
}

fn decode_e4m3(code: u8) -> f32 {
    let mag = code & 0x7f;
    assert_ne!(mag, 0x7f, "test inputs exclude the NaN code");
    let e = (mag >> 3) as i32;
    let m = (mag & 7) as f64;
    let v = if e == 0 {
        m * 2f64.powi(-9)
    } else {
        (1.0 + m / 8.0) * 2f64.powi(e - 7)
    } as f32;
    if code & 0x80 != 0 {
        -v
    } else {
        v
    }
}

fn gen_case(n: usize, k: usize, seed: u64) -> (Vec<u8>, Vec<u16>, Vec<f32>) {
    let mut rng = Lcg::new(seed);
    let w: Vec<u8> = (0..n * k)
        .map(|_| {
            let b = (rng.next_u32() & 0xff) as u8;
            if b & 0x7f == 0x7f {
                b & !0x08
            } else {
                b
            }
        })
        .collect();
    let x: Vec<u16> = (0..k)
        .map(|_| {
            let v = (rng.next_u32() as f32 / u32::MAX as f32) * 2.0 - 1.0;
            bf16::from_f32(v).to_bits()
        })
        .collect();
    let scales: Vec<f32> = (0..n)
        .map(|_| 0.001 + (rng.next_u32() as f32 / u32::MAX as f32) * 0.05)
        .collect();
    (w, x, scales)
}

fn host_ref(w: &[u8], x: &[u16], scales: &[f32], n: usize, k: usize) -> Vec<f32> {
    let mut y = vec![0f32; n];
    for row in 0..n {
        let mut acc = 0f32;
        for kk in 0..k {
            acc += decode_e4m3(w[row * k + kk]) * bf16::from_bits(x[kk]).to_f32();
        }
        y[row] = acc * scales[row];
    }
    y
}

fn pack_bytes(w: &[u8]) -> Vec<u32> {
    w.chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

fn pack_x(x: &[u16]) -> Vec<u32> {
    x.chunks(2)
        .map(|c| (c[0] as u32) | ((*c.get(1).unwrap_or(&0) as u32) << 16))
        .collect()
}

fn run_kernel(
    ctx: &WgpuContext,
    w: &[u8],
    x: &[u16],
    scales: &[f32],
    n: usize,
    k: usize,
) -> Vec<f32> {
    let row_words = k / 4;
    let pairs = n.div_ceil(2);
    let groups = dispatch::workgroup_count_1d(ctx, pairs as u64, 1);
    let params = Q3bParams {
        n_rows: n as u32,
        k_words: row_words as u32,
        groups_x: groups.0,
        out_f32: 1,
        w_row_words: row_words as u32,
        x_off_words: 0,
        y_off_words: 0,
        alpha: 1.0,
        ..Default::default()
    };
    let wb = dispatch::storage_from_slice(ctx, "q3dfp8-w", &pack_bytes(w));
    let xb = dispatch::storage_from_slice(ctx, "q3dfp8-x", &pack_x(x));
    let yb = dispatch::storage_from_slice(ctx, "q3dfp8-y", &vec![0x7fc00000u32; n]);
    let sb = dispatch::storage_from_slice(
        ctx,
        "q3dfp8-s",
        &nv_kernels::shift_decode_fold::fold_scales_for_e4m3_shift_decode(scales),
    );
    let ub = dispatch::uniform_from(ctx, "q3dfp8-p", &params);
    dispatch::run(
        ctx,
        "q3dfp8",
        &compose(Q3D_GEMV_WGSL),
        ENTRY,
        &[(0, &wb), (1, &xb), (2, &ub), (3, &yb), (4, &sb)],
        groups,
    )
    .expect("dispatch");
    let words: Vec<u32> = dispatch::read_back(ctx, &yb, n).expect("read back");
    words.iter().map(|v| f32::from_bits(*v)).collect()
}

#[test]
fn fp8_rowscale_lmhead_entry_matches_the_host_oracle_at_serving_shapes() {
    let Some(ctx) = ctx_or_skip("fp8_rowscale_lmhead_entry") else {
        return;
    };
    for &(n, k, seed) in &[
        (2usize, 64usize, 1u64),
        (7, 512, 2),
        (33, 5120, 3),
        (64, 5120, 4),
    ] {
        let (w, x, scales) = gen_case(n, k, seed);
        let got = run_kernel(ctx, &w, &x, &scales, n, k);
        let want = host_ref(&w, &x, &scales, n, k);
        for (row, (g, r)) in got.iter().zip(&want).enumerate() {
            let d = (g - r).abs() / r.abs().max(1e-3);
            assert!(
                d < KERNEL_VS_HOST_ORACLE_REL_TOL_PINNED_BY_GEMV_W4A16_SUITES,
                "n={n} k={k} seed={seed} row {row}: kernel {g} vs host {r} (rel {d:.3e}); \
                 the entry no longer implements the documented rowscale rule (shift-decode \
                 e4m3 landing 2^120 below true, accumulate, scale once per row with the \
                 uploaded row scale carrying the 2^120)"
            );
        }
    }
}

#[test]
fn distinct_row_scales_are_fuzzed_so_a_scale_index_bug_cannot_hide() {
    let Some(ctx) = ctx_or_skip("distinct_row_scales") else {
        return;
    };
    let (n, k) = (8usize, 256usize);
    let (w, x, _) = gen_case(n, k, 9);
    let ascending: Vec<f32> = (0..n).map(|r| 0.01 * (r + 1) as f32).collect();
    let got = run_kernel(ctx, &w, &x, &ascending, n, k);
    let want = host_ref(&w, &x, &ascending, n, k);
    for (row, (g, r)) in got.iter().zip(&want).enumerate() {
        let d = (g - r).abs() / r.abs().max(1e-3);
        assert!(
            d < KERNEL_VS_HOST_ORACLE_REL_TOL_PINNED_BY_GEMV_W4A16_SUITES,
            "row {row} under ascending per-row scales: kernel {g} vs host {r} (rel {d:.3e}); \
             uniform scales hide off-by-one scale indexing, so this case is mandatory \
             (the w4a16 group-16 incident class, row-scale edition)"
        );
    }
}
