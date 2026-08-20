#![cfg(feature = "wgpu")]

mod common;
use common::d;
use common::dot8;
use common::q;
use common::require;
use common::tree_sum;
use half::bf16;
use nv_kernels::wgpu_backend::device::WgpuContext;
use nv_kernels::wgpu_backend::kernels::gemv_w4a16;
use nv_kernels::wgpu_backend::WgpuError;
use common::LcgShift33W4a16Packs as Lcg;
use common::ctx_or_skip;

fn dot8_pairwise(word: u32, x: &[u16], kb: usize) -> f32 {
    let mut a = 0f32;
    for i in 0..4 {
        a += q(word, 2 * i) * d(x[kb + 2 * i]) + q(word, 2 * i + 1) * d(x[kb + 2 * i + 1]);
    }
    a
}

fn dot32(packed: &[u32], wbase: usize, x: &[u16], kbase: usize) -> f32 {
    let mut a = 0f32;
    for j in 0..4 {
        a = dot8(packed[wbase + j], x, kbase + j * 8, a);
    }
    a
}

fn lane_acc(
    packed: &[u32],
    scales: &[u16],
    x: &[u16],
    wbase: usize,
    sbase: usize,
    kv: usize,
    lane: usize,
    lanes: usize,
    gs: usize,
) -> f32 {
    let mut acc = 0f32;
    let mut v = lane;
    while v < kv {
        let kbase = v * 32;
        if gs >= 32 {
            let sc = d(scales[sbase + kbase / gs]);
            acc = sc.mul_add(dot32(packed, wbase + v * 4, x, kbase), acc);
        } else {
            for j in 0..4 {
                let kb = kbase + j * 8;
                let sc = d(scales[sbase + kb / gs]);
                let a = dot8(packed[wbase + v * 4 + j], x, kb, 0.0);
                acc = a.mul_add(sc, acc);
            }
        }
        v += lanes;
    }
    acc
}

fn cpu_block_row(
    packed: &[u32],
    scales: &[u16],
    x: &[u16],
    row: usize,
    k: usize,
    gs: usize,
) -> f32 {
    let kv = k / 32;
    let wbase = row * (k / 8);
    let sbase = row * (k / gs);
    let lanes: Vec<f32> = (0..32)
        .map(|lane| lane_acc(packed, scales, x, wbase, sbase, kv, lane, 32, gs))
        .collect();
    tree_sum(&lanes)
}

fn cpu_row_row(packed: &[u32], scales: &[u16], x: &[u16], row: usize, k: usize, gs: usize) -> f32 {
    let kv = k / 32;
    let wbase = row * (k / 8);
    let sbase = row * (k / gs);
    let mut threads = vec![0f32; 256];
    for (tid, slot) in threads.iter_mut().enumerate() {
        let mut acc = 0f32;
        let mut v = tid;
        while v < kv {
            let kbase = v * 32;
            let sc = if gs >= 32 {
                d(scales[sbase + kbase / gs])
            } else {
                0.0
            };
            let mut block_acc = 0f32;
            for j in 0..4 {
                let kb = kbase + j * 8;
                let a = dot8_pairwise(packed[wbase + v * 4 + j], x, kb);
                if gs >= 32 {
                    block_acc += a;
                } else {
                    block_acc = a.mul_add(d(scales[sbase + kb / gs]), block_acc);
                }
            }
            if gs >= 32 {
                acc = sc.mul_add(block_acc, acc);
            } else {
                acc += block_acc;
            }
            v += 256;
        }
        *slot = acc;
    }
    let warp_sums: Vec<f32> = (0..8)
        .map(|w| tree_sum(&threads[w * 32..w * 32 + 32]))
        .collect();
    tree_sum(&warp_sums)
}

fn cpu_oracle(
    packed: &[u32],
    scales: &[u16],
    x: &[u16],
    n: usize,
    k: usize,
    gs: usize,
) -> Vec<u16> {
    (0..n)
        .map(|row| {
            let acc = if k <= gemv_w4a16::MAX_SHARED_K {
                cpu_block_row(packed, scales, x, row, k, gs)
            } else {
                cpu_row_row(packed, scales, x, row, k, gs)
            };
            bf16::from_f32(acc).to_bits()
        })
        .collect()
}

fn gelu(v: f32) -> f32 {
    let c = 0.797_884_6f32;
    let t = (c * (v + 0.044715 * v * v * v)).tanh();
    0.5 * v * (1.0 + t)
}

fn ulp(a: u16, b: u16) -> i32 {
    (a as i32 - b as i32).abs()
}

fn report(name: &str, want: &[u16], got: &[u16]) -> (usize, i32) {
    let mut mismatch = 0usize;
    let mut max_ulp = 0i32;
    for (a, b) in want.iter().zip(got.iter()) {
        if a != b {
            mismatch += 1;
            max_ulp = max_ulp.max(ulp(*a, *b));
        }
    }
    eprintln!(
        "{name}: {mismatch}/{} bf16 words differ, max_ulp={max_ulp}",
        want.len()
    );
    (mismatch, max_ulp)
}

fn case(ctx: &WgpuContext, name: &str, n: usize, k: usize, gs: usize, seed: u64) -> (usize, i32) {
    let mut rng = Lcg::new(seed);
    let packed = rng.packed(n * (k / 8));
    let scales = rng.scales(n * (k / gs));
    let x = rng.bf16_words(k, 1.5);
    let want = cpu_oracle(&packed, &scales, &x, n, k, gs);
    let mut got = vec![0u16; n];
    gemv_w4a16::gemv_w4a16(ctx, &packed, &scales, &x, &mut got, n, k, gs).unwrap();
    report(&format!("{name} n={n} k={k} gs={gs}"), &want, &got)
}

#[test]
fn block_path_matches_the_cpu_oracle_bit_exactly() {
    let Some(ctx) = ctx_or_skip("block_path") else {
        return;
    };
    for (n, k, gs, seed) in [
        (8usize, 128usize, 128usize, 1u64),
        (37, 1024, 128, 2),
        (64, 3072, 3072, 3),
        (13, 256, 64, 4),
    ] {
        let (mismatch, max_ulp) = case(ctx, "block", n, k, gs, seed);
        assert_eq!(mismatch, 0, "block n={n} k={k} gs={gs} max_ulp={max_ulp}");
    }
}

#[test]
fn sub_warp_group_sizes_match_the_cpu_oracle() {
    let Some(ctx) = ctx_or_skip("sub_warp_groups") else {
        return;
    };
    for (n, k, gs, seed) in [
        (16usize, 512usize, 8usize, 11u64),
        (9, 1024, 16, 12),
        (33, 1920, 24, 13),
    ] {
        let (mismatch, max_ulp) = case(ctx, "block-subwarp", n, k, gs, seed);
        assert_eq!(mismatch, 0, "subwarp n={n} k={k} gs={gs} max_ulp={max_ulp}");
    }
}

#[test]
fn row_path_matches_the_cpu_oracle_within_one_bf16_ulp() {
    let Some(ctx) = ctx_or_skip("row_path") else {
        return;
    };
    for (n, k, gs, seed) in [
        (12usize, 4096usize, 128usize, 21u64),
        (5, 8192, 64, 22),
        (7, 4096, 16, 23),
    ] {
        let (mismatch, max_ulp) = case(ctx, "row", n, k, gs, seed);
        assert!(
            max_ulp <= 1,
            "row n={n} k={k} gs={gs} mismatch={mismatch} max_ulp={max_ulp}"
        );
    }
}

#[test]
fn gelu_pli_applies_the_cuda_epilogue() {
    let Some(ctx) = ctx_or_skip("gelu_pli") else {
        return;
    };
    let (n, k, gs) = (19usize, 1024usize, 128usize);
    let mut rng = Lcg::new(31);
    let packed = rng.packed(n * (k / 8));
    let scales = rng.scales(n * (k / gs));
    let x = rng.bf16_words(k, 1.0);
    let pli: Vec<f32> = (0..n).map(|i| 0.5 + (i as f32) * 0.03).collect();

    let want: Vec<u16> = (0..n)
        .map(|row| {
            let acc = cpu_block_row(&packed, &scales, &x, row, k, gs);
            bf16::from_f32(gelu(acc) * pli[row]).to_bits()
        })
        .collect();

    let mut got = vec![0u16; n];
    gemv_w4a16::gemv_w4a16_gelu_pli(ctx, &packed, &scales, &x, &pli, &mut got, n, k, gs).unwrap();
    let (_, max_ulp) = report("gelu_pli", &want, &got);
    assert!(max_ulp <= 1, "gelu_pli max_ulp={max_ulp}");
}

#[test]
fn degenerate_and_invalid_shapes_follow_the_cuda_host_guard() {
    let Some(ctx) = ctx_or_skip("shape_guard") else {
        return;
    };
    let mut y = vec![0u16; 4];
    assert!(gemv_w4a16::gemv_w4a16(ctx, &[], &[], &[], &mut y, 0, 128, 128).is_ok());
    assert!(gemv_w4a16::gemv_w4a16(ctx, &[], &[], &[], &mut y, 4, 0, 128).is_ok());
    assert!(gemv_w4a16::gemv_w4a16(ctx, &[], &[], &[], &mut y, 4, 128, 0).is_ok());
    let e = gemv_w4a16::gemv_w4a16(ctx, &[], &[], &[], &mut y, 4, 48, 16).unwrap_err();
    assert!(matches!(e, WgpuError::Shape(_)), "{e}");
    let e = gemv_w4a16::gemv_w4a16(ctx, &[], &[], &[], &mut y, 4, 128, 12).unwrap_err();
    assert!(matches!(e, WgpuError::Shape(_)), "{e}");
    let e =
        gemv_w4a16::gemv_w4a16_gelu_pli(ctx, &[], &[], &[], &[], &mut y, 4, 128, 16).unwrap_err();
    assert!(matches!(e, WgpuError::Shape(_)), "{e}");
    let e =
        gemv_w4a16::gemv_w4a16_gelu_pli(ctx, &[], &[], &[], &[], &mut y, 4, 4096, 128).unwrap_err();
    assert!(matches!(e, WgpuError::Shape(_)), "{e}");
}

#[test]
fn n_not_a_multiple_of_the_row_tile_leaves_no_row_unwritten() {
    let Some(ctx) = ctx_or_skip("ragged_n") else {
        return;
    };
    let (n, k, gs) = (11usize, 512usize, 128usize);
    let mut rng = Lcg::new(77);
    let packed = rng.packed(n * (k / 8));
    let scales = rng.scales(n * (k / gs));
    let x = rng.bf16_words(k, 1.0);
    let want = cpu_oracle(&packed, &scales, &x, n, k, gs);
    let mut got = vec![0xffffu16; n];
    gemv_w4a16::gemv_w4a16(ctx, &packed, &scales, &x, &mut got, n, k, gs).unwrap();
    assert_eq!(want, got);
}
