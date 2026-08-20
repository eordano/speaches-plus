#![cfg(feature = "wgpu")]

mod common;
use common::d;
use common::require;
use half::bf16;
use nv_kernels::wgpu_backend::device::WgpuContext;
use nv_kernels::wgpu_backend::kernels::gemv_bf16;
use common::ctx_or_skip;

struct Lcg(u64);

impl Lcg {
    fn next_f32(&mut self) -> f32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let bits = (self.0 >> 40) as u32;
        (bits as f32 / 8388608.0) - 1.0
    }
    fn bf16_words(&mut self, n: usize, gain: f32) -> Vec<u16> {
        (0..n)
            .map(|_| bf16::from_f32(self.next_f32() * gain).to_bits())
            .collect()
    }
}

const LANES: usize = 32;

fn lane_tree_sum(lanes: &[f32; LANES]) -> f32 {
    let mut s = *lanes;
    let mut stride = LANES / 2;
    while stride > 0 {
        for i in 0..stride {
            s[i] += s[i + stride];
        }
        stride >>= 1;
    }
    s[0]
}

fn cpu_gemv_oracle(w: &[u16], x: &[u16], n: usize, k: usize) -> Vec<u16> {
    let mut y = vec![0u16; n];
    for (row, slot) in y.iter_mut().enumerate() {
        let base = row * k;
        let mut lanes = [0f32; LANES];
        if k.is_multiple_of(8) {
            let kv = k / 8;
            for (lane, acc) in lanes.iter_mut().enumerate() {
                let mut v = lane;
                while v < kv {
                    for j in 0..4 {
                        let kb = v * 8 + 2 * j;
                        let w0 = d(w[base + kb]);
                        let w1 = d(w[base + kb + 1]);
                        let x0 = d(x[kb]);
                        let x1 = d(x[kb + 1]);
                        *acc += w0 * x0 + w1 * x1;
                    }
                    v += LANES;
                }
            }
        } else {
            for (lane, acc) in lanes.iter_mut().enumerate() {
                let mut kk = lane;
                while kk < k {
                    *acc += d(w[base + kk]) * d(x[kk]);
                    kk += LANES;
                }
            }
        }
        *slot = bf16::from_f32(lane_tree_sum(&lanes)).to_bits();
    }
    y
}

fn report(name: &str, got: &[u16], want: &[u16]) -> i32 {
    let mut mismatch = 0usize;
    let mut max_ulp = 0i32;
    let mut max_abs = 0f32;
    for (a, b) in got.iter().zip(want.iter()) {
        if a != b {
            mismatch += 1;
            max_ulp = max_ulp.max((*a as i32 - *b as i32).abs());
            max_abs = max_abs.max((d(*a) - d(*b)).abs());
        }
    }
    eprintln!(
        "{name}: {mismatch}/{} bf16 words differ, max_ulp={max_ulp} max_abs={max_abs:e}",
        want.len()
    );
    max_ulp
}

fn run_case(ctx: &WgpuContext, name: &str, n: usize, k: usize, seed: u64) {
    let mut rng = Lcg(seed);
    let w = rng.bf16_words(n * k, 1.0);
    let x = rng.bf16_words(k, 2.0);
    let mut y = vec![0u16; n];
    gemv_bf16::gemv_bf16(ctx, &w, &x, &mut y, n, k).expect("wgpu gemv_bf16");
    let want = cpu_gemv_oracle(&w, &x, n, k);
    let max_ulp = report(&format!("{name} n={n} k={k}"), &y, &want);
    assert_eq!(
        max_ulp, 0,
        "{name}: wgpu must match the lane-ordered cpu oracle bit-exactly"
    );
}

#[test]
fn vec8_path_matches_the_cpu_oracle() {
    let Some(ctx) = ctx_or_skip("vec8_path_matches_the_cpu_oracle") else {
        return;
    };
    run_case(ctx, "vec8", 37, 1024, 0x1234_5678);
    run_case(ctx, "vec8-tail-rows", 5, 4096, 0x0bad_c0de);
    run_case(ctx, "vec8-short-k", 64, 8, 0xfeed_face);
}

#[test]
fn vec8_path_matches_beyond_the_cuda_shared_limit() {
    let Some(ctx) = ctx_or_skip("vec8_path_matches_beyond_the_cuda_shared_limit") else {
        return;
    };
    run_case(ctx, "vec8-big-k", 24, 8192, 0x5eed_1234);
}

#[test]
fn scalar_path_matches_the_cpu_oracle() {
    let Some(ctx) = ctx_or_skip("scalar_path_matches_the_cpu_oracle") else {
        return;
    };
    run_case(ctx, "scalar", 33, 1026, 0xa5a5_5a5a);
    run_case(ctx, "scalar-tiny", 9, 2, 0x0000_1111);
}

#[test]
fn one_hot_input_selects_the_addressed_weight_exactly() {
    let Some(ctx) = ctx_or_skip("one_hot_input_selects_the_addressed_weight_exactly") else {
        return;
    };
    let (n, k) = (20usize, 1024usize);
    let mut rng = Lcg(0x3141_5926);
    let w = rng.bf16_words(n * k, 4.0);
    for p in [0usize, 1, 7, 8, 255, 256, 257, 1023] {
        let mut x = vec![0u16; k];
        x[p] = bf16::from_f32(1.0).to_bits();
        let mut y = vec![0u16; n];
        gemv_bf16::gemv_bf16(ctx, &w, &x, &mut y, n, k).expect("wgpu gemv_bf16");
        for row in 0..n {
            assert_eq!(
                y[row],
                w[row * k + p],
                "row {row} column {p}: one-hot input must return the weight unchanged"
            );
        }
    }
}

#[test]
fn many_rows_fold_across_the_workgroup_y_dimension() {
    let Some(ctx) = ctx_or_skip("many_rows_fold_across_the_workgroup_y_dimension") else {
        return;
    };
    let limit = ctx.caps.max_compute_workgroups_per_dimension as usize;
    let n = limit * 8 + 17;
    let k = 8usize;
    if (n * k) as u64 * 2 > ctx.caps.max_storage_buffer_binding_size {
        if !require() {
            eprintln!(
                "many_rows_fold_across_the_workgroup_y_dimension: SKIP \
                 (NV_KERNELS_WGPU_ALLOW_SKIP=1) {n}x{k} exceeds the storage binding limit"
            );
            return;
        }
        panic!(
            "many_rows_fold_across_the_workgroup_y_dimension: {n}x{k} exceeds the storage \
             binding limit {}, so the y-dimension fold this test exists to check cannot be \
             exercised at all. Set NV_KERNELS_WGPU_ALLOW_SKIP=1 to skip on purpose.",
            ctx.caps.max_storage_buffer_binding_size
        );
    }
    let mut rng = Lcg(0x7777_0001);
    let w = rng.bf16_words(n * k, 1.0);
    let x = rng.bf16_words(k, 1.0);
    let mut y = vec![0u16; n];
    gemv_bf16::gemv_bf16(ctx, &w, &x, &mut y, n, k).expect("wgpu gemv_bf16");
    let want = cpu_gemv_oracle(&w, &x, n, k);
    let max_ulp = report(&format!("fold n={n} k={k}"), &y, &want);
    assert_eq!(
        max_ulp, 0,
        "rows past the x-dimension limit must still be computed"
    );
}

#[test]
fn odd_k_is_rejected_like_the_cuda_host() {
    let Some(ctx) = ctx_or_skip("odd_k_is_rejected_like_the_cuda_host") else {
        return;
    };
    let mut y = vec![0u16; 2];
    let err = gemv_bf16::gemv_bf16(ctx, &[0u16; 6], &[0u16; 3], &mut y, 2, 3)
        .expect_err("odd K must be rejected");
    assert!(format!("{err}").contains("K must be even"), "{err}");
}

#[test]
fn empty_shapes_are_a_no_op() {
    let Some(ctx) = ctx_or_skip("empty_shapes_are_a_no_op") else {
        return;
    };
    let mut y: Vec<u16> = Vec::new();
    gemv_bf16::gemv_bf16(ctx, &[], &[], &mut y, 0, 128).expect("n=0 is a no-op");
    let mut y2 = vec![7u16; 4];
    gemv_bf16::gemv_bf16(ctx, &[], &[], &mut y2, 4, 0).expect("k=0 is a no-op");
    assert_eq!(y2, vec![7u16; 4]);
}

#[test]
fn weight_traffic_bandwidth() {
    let Some(ctx) = ctx_or_skip("weight_traffic_bandwidth") else {
        return;
    };
    let mut rng = Lcg(0x2024_0808);
    let mut ran = 0usize;
    for (n, k) in [(4096usize, 4096usize), (8192, 2048)] {
        if (n * k) as u64 * 2 > ctx.caps.max_storage_buffer_binding_size {
            eprintln!("SKIP {n}x{k}: exceeds the storage binding limit");
            continue;
        }
        ran += 1;
        let w = rng.bf16_words(n * k, 1.0);
        let x = rng.bf16_words(k, 1.0);
        let gbps =
            gemv_bf16::gemv_bf16_weight_gbps(ctx, &w, &x, n, k, 50).expect("bandwidth probe");
        eprintln!(
            "gemv_bf16 n={n} k={k}: {:.1} MiB of weights, {gbps:.1} GB/s of weight traffic",
            (n * k * 2) as f64 / (1024.0 * 1024.0)
        );
        assert!(gbps > 0.0);
    }
    assert!(
        ran > 0,
        "weight_traffic_bandwidth measured 0 of 2 shapes: every case fell past the storage \
         binding limit, so a pass here would mean nothing"
    );
}
