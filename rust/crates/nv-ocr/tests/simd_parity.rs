use nv_ocr::lstm::matvec_i8_scalar;
use nv_ocr::simd::{f32_isa, i8_isa, F32Isa, I8Isa};
use rand_core::Rng;
use rand_pcg::Pcg64;

fn shapes() -> Vec<(usize, usize)> {
    vec![
        (1, 2),
        (1, 10),
        (3, 17),
        (4, 65),
        (7, 64),
        (8, 129),
        (16, 10),
        (48, 65),
        (64, 81),
        (96, 145),
        (111, 193),
        (111, 513),
        (192, 305),
        (512, 609),
    ]
}

fn rng_f32(rng: &mut Pcg64) -> f32 {
    (rng.next_u32() as f32 / u32::MAX as f32) * 2.0 - 1.0
}

fn ref_f32_reverse_order_in_f64(rows: usize, cols: usize, w: &[f32], u: &[f32]) -> Vec<f64> {
    let n = cols - 1;
    (0..rows)
        .map(|r| {
            let row = &w[r * cols..(r + 1) * cols];
            let mut acc = 0.0f64;
            for j in (0..n).rev() {
                acc += row[j] as f64 * u[j] as f64;
            }
            acc + row[n] as f64
        })
        .collect()
}

fn abs_sum_f32(cols: usize, row: &[f32], u: &[f32]) -> f64 {
    let n = cols - 1;
    let mut s = (row[n] as f64).abs();
    for j in 0..n {
        s += (row[j] as f64 * u[j] as f64).abs();
    }
    s
}

fn ref_i8_exact_in_i64(rows: usize, cols: usize, w: &[i8], scales: &[f32], u: &[i8]) -> Vec<f32> {
    let n = cols - 1;
    (0..rows)
        .map(|r| {
            let row = &w[r * cols..(r + 1) * cols];
            let mut acc = 0i64;
            for j in (0..n).rev() {
                acc += row[j] as i64 * u[j] as i64;
            }
            acc += row[n] as i64 * 127;
            assert!(
                acc >= i32::MIN as i64 && acc <= i32::MAX as i64,
                "row {r}: accumulator {acc} does not fit i32; the kernel would have wrapped and \
                 this reference would be comparing against undefined behaviour"
            );
            acc as i32 as f32 * scales[r]
        })
        .collect()
}

fn assert_has_resolving_power(label: &str, want: &[f64], tol: f64) {
    assert!(!want.is_empty(), "{label}: empty comparison");
    let mag = want.iter().fold(0.0f64, |a, b| a.max(b.abs()));
    assert!(
        mag > 1e-3,
        "{label}: reference is degenerate (max |ref| = {mag:.3e}); a gate over zeros stays green \
         forever"
    );
    if want.len() > 1 {
        let lo = want.iter().cloned().fold(f64::INFINITY, f64::min);
        let hi = want.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let spread = hi - lo;
        assert!(
            spread > 1e-3,
            "{label}: reference is constant (spread {spread:.3e}); it cannot see a permutation or \
             indexing bug"
        );
        assert!(
            spread > 1e3 * tol,
            "{label}: tolerance {tol:.3e} is not small against the signal (spread {spread:.3e}); \
             the gate has no resolving power"
        );
    }
}

#[test]
fn f32_dispatch_matches_f64_host_reference() {
    let mut rng = Pcg64::new(0x1d872b6f9c3a5e01, 0x9e3779b97f4a7c15);
    let mut worst_rel = 0.0f64;
    for (rows, cols) in shapes() {
        let w: Vec<f32> = (0..rows * cols).map(|_| rng_f32(&mut rng)).collect();
        let u: Vec<f32> = (0..cols - 1).map(|_| rng_f32(&mut rng)).collect();

        let want = ref_f32_reverse_order_in_f64(rows, cols, &w, &u);
        let mut got = vec![0f32; rows];
        nv_ocr::simd::matvec_f32(rows, cols, &w, &u, &mut got);

        let worst_tol = (0..rows)
            .map(|r| {
                8.0 * f32::EPSILON as f64
                    * abs_sum_f32(cols, &w[r * cols..(r + 1) * cols], &u)
                    * ((cols - 1) as f64).sqrt()
            })
            .fold(0.0f64, f64::max);
        assert_has_resolving_power(&format!("f32 {rows}x{cols}"), &want, worst_tol);

        for r in 0..rows {
            let tol = 8.0
                * f32::EPSILON as f64
                * abs_sum_f32(cols, &w[r * cols..(r + 1) * cols], &u)
                * ((cols - 1) as f64).sqrt();
            let err = (got[r] as f64 - want[r]).abs();
            assert!(
                err <= tol,
                "{:?} rows={rows} cols={cols} r={r}: kernel {} vs host f64 {} (err {err:.3e} > \
                 tol {tol:.3e})",
                f32_isa(),
                got[r],
                want[r]
            );
            if want[r].abs() > 1e-6 {
                worst_rel = worst_rel.max(err / want[r].abs());
            }
        }
    }
    println!(
        "f32 dispatch={:?}: worst relative error vs host f64 = {worst_rel:.3e}",
        f32_isa()
    );
}

#[test]
fn i8_dispatch_matches_exact_host_reference() {
    let mut rng = Pcg64::new(0x853c49e6748fea9b, 0xda3e39cb94b95bdb);
    for (rows, cols) in shapes() {
        let w: Vec<i8> = (0..rows * cols).map(|_| rng.next_u32() as i8).collect();
        let u: Vec<i8> = (0..cols - 1)
            .map(|_| (rng.next_u32() as i8).max(-127))
            .collect();
        let scales: Vec<f32> = (0..rows)
            .map(|_| (rng.next_u32() % 1000 + 1) as f32 * 1e-5)
            .collect();

        let want = ref_i8_exact_in_i64(rows, cols, &w, &scales, &u);
        let want_f64: Vec<f64> = want.iter().map(|v| *v as f64).collect();
        assert_has_resolving_power(&format!("i8 {rows}x{cols}"), &want_f64, 0.0);

        let mut got = vec![0f32; rows];
        nv_ocr::simd::matvec_i8(rows, cols, &w, &scales, &u, &mut got);
        assert_eq!(
            got,
            want,
            "{:?} rows={rows} cols={cols}: i8 matvec is not bit-exact against the integer \
             reference",
            i8_isa()
        );
    }
    println!(
        "i8 dispatch={:?}: bit-exact against the integer host reference",
        i8_isa()
    );
}

#[cfg(target_arch = "x86_64")]
#[test]
fn f32_kernels_match_scalar() {
    use nv_ocr::lstm::matvec_f32_scalar;
    use nv_ocr::simd::matvec_f32_at;
    let mut rng = Pcg64::new(0xcafef00dd15ea5e5, 0xa02bdbf7bb3c0a7);
    let mut isas = vec![];
    if is_x86_feature_detected!("avx512f") {
        isas.push(F32Isa::Avx512);
    }
    if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
        isas.push(F32Isa::Avx2);
    }
    assert!(
        !isas.is_empty(),
        "no vector f32 ISA on this x86_64 CPU: nothing to compare against the scalar kernel, and \
         the `auto` check below would be scalar-vs-scalar"
    );
    for (rows, cols) in shapes() {
        let w: Vec<f32> = (0..rows * cols).map(|_| rng_f32(&mut rng)).collect();
        let u: Vec<f32> = (0..cols - 1).map(|_| rng_f32(&mut rng)).collect();
        let mut want = vec![0f32; rows];
        matvec_f32_scalar(rows, cols, &w, &u, &mut want);
        for isa in &isas {
            let mut got = vec![0f32; rows];
            matvec_f32_at(*isa, rows, cols, &w, &u, &mut got);
            for r in 0..rows {
                let tol = 1e-4 * (1.0 + want[r].abs()) * (cols as f32).sqrt();
                assert!(
                    (got[r] - want[r]).abs() <= tol,
                    "{:?} rows={} cols={} r={} got={} want={}",
                    isa,
                    rows,
                    cols,
                    r,
                    got[r],
                    want[r]
                );
            }
        }
        let mut auto = vec![0f32; rows];
        nv_ocr::simd::matvec_f32(rows, cols, &w, &u, &mut auto);
        for r in 0..rows {
            let tol = 1e-4 * (1.0 + want[r].abs()) * (cols as f32).sqrt();
            assert!(
                (auto[r] - want[r]).abs() <= tol,
                "{:?} rows={rows} cols={cols} r={r} auto={} scalar={}",
                f32_isa(),
                auto[r],
                want[r]
            );
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[test]
fn i8_kernels_match_scalar_exactly() {
    use nv_ocr::simd::matvec_i8_at;
    let mut rng = Pcg64::new(0x853c49e6748fea9b, 0xda3e39cb94b95bdb);
    let mut isas = vec![];
    if is_x86_feature_detected!("avx512f")
        && is_x86_feature_detected!("avx512bw")
        && is_x86_feature_detected!("avx512vl")
    {
        isas.push(I8Isa::Avx512);
    }
    if is_x86_feature_detected!("avx2") {
        isas.push(I8Isa::Avx2);
    }
    assert!(
        !isas.is_empty(),
        "no vector i8 ISA on this x86_64 CPU: nothing to compare against the scalar kernel"
    );
    for (rows, cols) in shapes() {
        let w: Vec<i8> = (0..rows * cols).map(|_| rng.next_u32() as i8).collect();
        let u: Vec<i8> = (0..cols - 1)
            .map(|_| (rng.next_u32() as i8).max(-127))
            .collect();
        let scales: Vec<f32> = (0..rows)
            .map(|_| (rng.next_u32() % 1000 + 1) as f32 * 1e-5)
            .collect();
        let mut want = vec![0f32; rows];
        matvec_i8_scalar(rows, cols, &w, &scales, &u, &mut want);
        for isa in &isas {
            let mut got = vec![0f32; rows];
            matvec_i8_at(*isa, rows, cols, &w, &scales, &u, &mut got);
            assert_eq!(got, want, "{:?} rows={} cols={}", isa, rows, cols);
        }
        let mut auto = vec![0f32; rows];
        nv_ocr::simd::matvec_i8(rows, cols, &w, &scales, &u, &mut auto);
        assert_eq!(auto, want);
    }
}

#[test]
fn i8_extreme_values_exact() {
    for cols in [2usize, 64, 65, 129, 609] {
        let rows = 5usize;
        let w = [
            vec![127i8; cols],
            vec![-128i8; cols],
            vec![-127i8; cols],
            {
                let mut v = vec![0i8; cols];
                v[cols - 1] = -128;
                v
            },
            {
                let mut v: Vec<i8> = (0..cols)
                    .map(|j| if j % 2 == 0 { 127 } else { -128 })
                    .collect();
                v[cols - 1] = 127;
                v
            },
        ]
        .concat();
        let u: Vec<i8> = (0..cols - 1)
            .map(|j| if j % 3 == 0 { 127 } else { -127 })
            .collect();
        let scales = vec![3.7e-4f32; rows];
        let want = ref_i8_exact_in_i64(rows, cols, &w, &scales, &u);
        let mut scalar = vec![0f32; rows];
        matvec_i8_scalar(rows, cols, &w, &scales, &u, &mut scalar);
        assert_eq!(
            scalar, want,
            "cols={cols}: scalar kernel disagrees with the exact integer reference at the i8 \
             range limits"
        );
        let mut got = vec![0f32; rows];
        nv_ocr::simd::matvec_i8(rows, cols, &w, &scales, &u, &mut got);
        assert_eq!(
            got, want,
            "cols={cols}: dispatched kernel disagrees with the exact integer reference"
        );
    }
}

#[test]
fn dispatch_census_is_honest() {
    let f = f32_isa();
    let i = i8_isa();
    println!(
        "nv-ocr simd census: target_arch={} f32_isa={f:?} i8_isa={i:?}",
        std::env::consts::ARCH
    );

    if std::env::var_os("NV_OCR_NO_SIMD").is_some() {
        assert_eq!(
            f,
            F32Isa::Scalar,
            "NV_OCR_NO_SIMD set but f32 dispatch is {f:?}"
        );
        assert_eq!(
            i,
            I8Isa::Scalar,
            "NV_OCR_NO_SIMD set but i8 dispatch is {i:?}"
        );
        println!("NV_OCR_NO_SIMD is set: scalar forced, no vector path under test");
        return;
    }

    #[cfg(target_arch = "x86_64")]
    {
        let want_f32 = if is_x86_feature_detected!("avx512f") {
            F32Isa::Avx512
        } else if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            F32Isa::Avx2
        } else {
            F32Isa::Scalar
        };
        let want_i8 = if is_x86_feature_detected!("avx512f")
            && is_x86_feature_detected!("avx512bw")
            && is_x86_feature_detected!("avx512vl")
        {
            I8Isa::Avx512
        } else if is_x86_feature_detected!("avx2") {
            I8Isa::Avx2
        } else {
            I8Isa::Scalar
        };
        assert_eq!(
            f, want_f32,
            "f32 dispatch selected {f:?} but this CPU supports {want_f32:?}; the vector path is \
             no longer being reached and every comparison in this file degenerates to \
             scalar-vs-scalar"
        );
        assert_eq!(
            i, want_i8,
            "i8 dispatch selected {i:?} but this CPU supports {want_i8:?}"
        );
    }

    #[cfg(not(target_arch = "x86_64"))]
    {
        assert_eq!(
            f,
            F32Isa::Scalar,
            "F32Isa::{f:?} on {}: simd.rs has no non-x86_64 vector path, so this is unreachable \
             and something has changed",
            std::env::consts::ARCH
        );
        assert_eq!(
            i,
            I8Isa::Scalar,
            "I8Isa::{i:?} on {}: simd.rs has no non-x86_64 vector path",
            std::env::consts::ARCH
        );
        println!(
            "NOT A SIMD COMPARISON ON THIS TARGET: every vector path in nv-ocr/src/simd.rs is \
             #[cfg(target_arch = \"x86_64\")], so matvec_f32/matvec_i8 here ARE matvec_*_scalar. \
             f32_kernels_match_scalar and i8_kernels_match_scalar_exactly are not compiled on {}; \
             the coverage on this box is f32_dispatch_matches_f64_host_reference, \
             i8_dispatch_matches_exact_host_reference and i8_extreme_values_exact, all of which \
             compare the shipped kernel to a host reference derived independently of it.",
            std::env::consts::ARCH
        );
    }
}
