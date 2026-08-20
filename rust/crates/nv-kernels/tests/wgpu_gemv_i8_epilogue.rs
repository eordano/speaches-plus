#![cfg(feature = "wgpu")]

use nv_kernels::wgpu_backend::device::WgpuContext;
use nv_kernels::wgpu_backend::kernels::gemv_bf16 as g;

fn ctx() -> &'static WgpuContext {
    WgpuContext::shared().expect("wgpu adapter required for --features wgpu")
}

fn bf16(v: f32) -> u16 {
    let b = v.to_bits();
    let round = ((b >> 16) & 1) + 0x7fff;
    ((b + round) >> 16) as u16
}

fn unbf16(w: u16) -> f32 {
    f32::from_bits((w as u32) << 16)
}

struct Lcg(u64);
impl Lcg {
    fn next(&mut self) -> f32 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1);
        ((self.0 >> 33) as f32 / (1u32 << 31) as f32) * 2.0 - 1.0
    }
}

fn reference(
    wq: &[i8],
    scale: &[f32],
    x: &[u16],
    wn: &[u16],
    rstd: f32,
    n: usize,
    k: usize,
) -> Vec<f64> {
    (0..n)
        .map(|r| {
            let acc: f64 = (0..k)
                .map(|c| {
                    let xv = unbf16(x[c]) as f64 * rstd as f64 * unbf16(wn[c]) as f64;
                    wq[r * k + c] as f64 * xv
                })
                .sum();
            acc * scale[r] as f64
        })
        .collect()
}

fn worst_rel(got: &[u16], want: &[f64]) -> f64 {
    let mag = want.iter().fold(0.0f64, |m, v| m.max(v.abs())).max(1e-30);
    got.iter()
        .zip(want)
        .map(|(g, w)| (unbf16(*g) as f64 - w).abs() / mag)
        .fold(0.0, f64::max)
}

fn case(n: usize, k: usize, seed: u64, scale_of: impl Fn(usize) -> f32) -> f64 {
    let c = ctx();
    let mut r = Lcg(seed);
    let wq: Vec<i8> = (0..n * k).map(|_| (r.next() * 127.0) as i8).collect();
    let scale: Vec<f32> = (0..n).map(&scale_of).collect();
    let x: Vec<u16> = (0..k).map(|_| bf16(r.next())).collect();
    let wn: Vec<u16> = (0..k).map(|_| bf16(1.0 + 0.25 * r.next())).collect();
    let rstd = 0.7331f32;

    let want = reference(&wq, &scale, &x, &wn, rstd, n, k);
    let mut got = vec![0u16; n];
    g::gemv_i8_normed(c, &wq, &scale, &x, &wn, rstd, &mut got, n, k).expect("gemv_i8_normed");

    assert!(
        want.iter().any(|v| *v != 0.0),
        "reference is all zero at n={n} k={k}; the comparison would pass on any output"
    );
    assert!(
        got.iter().any(|w| unbf16(*w) != 0.0),
        "every output word is zero at n={n} k={k}; the product underflowed bf16 and this case \
         compares nothing against nothing"
    );
    worst_rel(&got, &want)
}

#[test]
fn i8_normed_matches_an_f64_host_reference() {
    for &(n, k) in &[(64usize, 256usize), (128, 512), (256, 1024), (37, 320)] {
        let e = case(n, k, 0x51ee_d000 ^ (n as u64) << 8 ^ k as u64, |r| {
            0.001 + 0.03 * ((r % 7) as f32)
        });
        assert!(
            e < 6e-3,
            "n={n} k={k}: worst relative error {e:.3e} against an f64 reference"
        );
    }
}

#[test]
fn the_per_row_scale_is_applied_and_is_per_row() {
    let c = ctx();
    let (n, k) = (96usize, 256usize);
    let mut r = Lcg(0xBEEF_0017);
    let wq: Vec<i8> = (0..n * k).map(|_| (r.next() * 100.0) as i8).collect();

    let scale: Vec<f32> = (0..n).map(|i| 0.002 * (1.0 + i as f32).powf(1.5)).collect();
    let x: Vec<u16> = (0..k).map(|_| bf16(r.next())).collect();
    let wn: Vec<u16> = (0..k).map(|_| bf16(1.0)).collect();

    let want = reference(&wq, &scale, &x, &wn, 1.0, n, k);
    let mut got = vec![0u16; n];
    g::gemv_i8_normed(c, &wq, &scale, &x, &wn, 1.0, &mut got, n, k).expect("gemv_i8_normed");
    assert!(
        worst_rel(&got, &want) < 6e-3,
        "scaled output does not match the f64 reference"
    );

    let ones = vec![1.0f32; n];
    let mut unscaled = vec![0u16; n];
    g::gemv_i8_normed(c, &wq, &ones, &x, &wn, 1.0, &mut unscaled, n, k).expect("gemv_i8_normed");
    let moved = got.iter().zip(&unscaled).filter(|(a, b)| a != b).count();
    assert!(
        moved > n / 2,
        "only {moved}/{n} rows changed when every row scale changed, so the scale buffer is \
         barely reaching the epilogue"
    );
}

#[test]
fn subnormal_row_scales_take_the_compensated_branch_and_still_match() {
    let e = case(48, 256, 0x5B_0001, |r| {
        f32::from_bits(0x007f_ffff - (r as u32 % 64) * 991)
    });
    assert!(
        e < 6e-3,
        "subnormal-scale branch: worst relative error {e:.3e} against an f64 reference"
    );
}

#[test]
fn the_mk_twin_agrees_with_the_single_row_entry() {
    let c = ctx();
    let (n, k, m) = (64usize, 256usize, 3usize);
    let mut r = Lcg(0x11AA_2200);
    let wq: Vec<i8> = (0..n * k).map(|_| (r.next() * 120.0) as i8).collect();
    let scale: Vec<f32> = (0..n).map(|i| 0.004 + 0.002 * (i % 11) as f32).collect();
    let wn: Vec<u16> = (0..k).map(|_| bf16(1.0 + 0.1 * r.next())).collect();
    let xs: Vec<u16> = (0..m * k).map(|_| bf16(r.next())).collect();
    let rstds: Vec<f32> = (0..m).map(|i| 0.5 + 0.1 * i as f32).collect();

    let mut mk = vec![0u16; m * n];
    g::gemv_i8_normed_mk(c, &wq, &scale, &xs, &wn, &rstds, &mut mk, m, n, k).expect("mk");

    for j in 0..m {
        let mut one = vec![0u16; n];
        g::gemv_i8_normed(
            c,
            &wq,
            &scale,
            &xs[j * k..(j + 1) * k],
            &wn,
            rstds[j],
            &mut one,
            n,
            k,
        )
        .expect("single");
        let want = reference(&wq, &scale, &xs[j * k..(j + 1) * k], &wn, rstds[j], n, k);
        assert!(
            worst_rel(&one, &want) < 6e-3,
            "row {j}: single-row entry disagrees with the f64 reference"
        );
        assert!(
            worst_rel(&mk[j * n..(j + 1) * n], &want) < 6e-3,
            "row {j}: mk twin disagrees with the f64 reference"
        );
    }
}

#[test]
fn rowquant_round_trips_within_int8_resolution() {
    let c = ctx();
    let (n, k) = (32usize, 512usize);
    let mut r = Lcg(0x9911_3311);
    let w: Vec<u16> = (0..n * k).map(|_| bf16(r.next() * 3.0)).collect();
    let mut q = vec![0i8; n * k];
    let mut s = vec![0f32; n];
    g::rowquant_i8(c, &w, &mut q, &mut s, n, k).expect("rowquant_i8");

    for row in 0..n {
        let peak = (0..k)
            .map(|c| unbf16(w[row * k + c]).abs())
            .fold(0.0f32, f32::max);
        assert!(s[row] > 0.0, "row {row} got a non-positive scale");
        let worst = (0..k)
            .map(|c| (q[row * k + c] as f32 * s[row] - unbf16(w[row * k + c])).abs())
            .fold(0.0f32, f32::max);

        assert!(
            worst <= peak / 127.0 * 1.5 + 1e-6,
            "row {row}: dequant error {worst:.3e} exceeds an int8 step for peak {peak:.3e}"
        );
    }
}
