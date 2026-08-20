#![cfg(feature = "cuda")]

use cudarc::driver::CudaContext;
use half::bf16;
use nv_quant::matmul::TensorCoreGemm;

fn build_a(m: usize, k: usize) -> Vec<bf16> {
    (0..m * k)
        .map(|i| bf16::from_f32((i as f32 * 0.0007).sin()))
        .collect()
}

fn build_w(n: usize, k: usize) -> Vec<bf16> {
    (0..n * k)
        .map(|i| bf16::from_f32((i as f32 * 0.0011).cos()))
        .collect()
}

fn transpose(w: &[bf16], n: usize, k: usize) -> Vec<bf16> {
    let mut t = vec![bf16::from_f32(0.0); n * k];
    for r in 0..n {
        for c in 0..k {
            t[c * n + r] = w[r * k + c];
        }
    }
    t
}

fn model_shapes() -> Vec<(usize, usize, usize)> {
    vec![
        (9, 8192, 5376),
        (9, 4096, 5376),
        (9, 5376, 8192),
        (9, 16384, 5376),
        (9, 2048, 5376),
        (9, 5376, 16384),
        (128, 8192, 5376),
        (512, 5376, 8192),
        (512, 8192, 5376),
        (9, 262144, 5376),
    ]
}

fn ulp_distance(a: bf16, b: bf16) -> u32 {
    fn key(x: bf16) -> i32 {
        let b = x.to_bits() as i32;
        if b & 0x8000 != 0 {
            0x8000 - b
        } else {
            b
        }
    }
    (key(a) - key(b)).unsigned_abs()
}

#[test]
fn bf16_bt_rounding_equivalent_to_pretransposed() {
    let Ok(ctx) = CudaContext::new(0) else {
        eprintln!("skip: no CUDA device 0");
        return;
    };
    let stream = ctx.default_stream();
    let gemm = TensorCoreGemm::new(stream.clone()).unwrap();

    for (m, n, k) in model_shapes() {
        let a_host = build_a(m, k);
        let w_host = build_w(n, k);
        let wt_host = transpose(&w_host, n, k);

        #[allow(deprecated)]
        let a_dev = stream.clone_htod(&a_host).unwrap();
        #[allow(deprecated)]
        let w_dev = stream.clone_htod(&w_host).unwrap();
        #[allow(deprecated)]
        let wt_dev = stream.clone_htod(&wt_host).unwrap();
        let mut c_nn = stream.alloc_zeros::<bf16>(m * n).unwrap();
        let mut c_tn = stream.alloc_zeros::<bf16>(m * n).unwrap();

        gemm.bf16_matmul_row_major(
            &stream, &a_dev, &wt_dev, &mut c_nn, m as u64, n as u64, k as u64, 1.0, 0.0,
        )
        .unwrap();
        gemm.bf16_matmul_row_major_bt(
            &stream, &a_dev, &w_dev, &mut c_tn, m as u64, n as u64, k as u64, 1.0, 0.0,
        )
        .unwrap();
        stream.synchronize().unwrap();

        #[allow(deprecated)]
        let got_nn = stream.memcpy_dtov(&c_nn).unwrap();
        #[allow(deprecated)]
        let got_tn = stream.memcpy_dtov(&c_tn).unwrap();
        let mut mismatches: Vec<usize> = Vec::new();
        let mut max_ulp = 0u32;
        for (i, (x, y)) in got_nn.iter().zip(got_tn.iter()).enumerate() {
            if x.to_bits() != y.to_bits() {
                mismatches.push(i);
                max_ulp = max_ulp.max(ulp_distance(*x, *y));
            }
        }
        let frac = mismatches.len() as f64 / (m * n) as f64;
        eprintln!(
            "shape m={m} n={n} k={k}: mismatches={}/{} ({:.4}%) max_ulp={max_ulp}",
            mismatches.len(),
            m * n,
            frac * 100.0
        );

        const MAX_CHECKED: usize = 20_000;
        const C_REL: f64 = 5e-3;
        for &idx in mismatches.iter().take(MAX_CHECKED) {
            let i = idx / n;
            let j = idx % n;
            let mut exact = 0f64;
            let mut absum = 0f64;
            for p in 0..k {
                let t = a_host[i * k + p].to_f64() * w_host[j * k + p].to_f64();
                exact += t;
                absum += t.abs();
            }
            let tol = C_REL * absum + f64::MIN_POSITIVE;
            let e_nn = (got_nn[idx].to_f64() - exact).abs();
            let e_tn = (got_tn[idx].to_f64() - exact).abs();

            let bf16_round = exact.abs() * 0.0078125 + f64::MIN_POSITIVE;
            assert!(
                e_nn <= tol + bf16_round,
                "NN result outside reduction bound at ({i},{j}) m={m} n={n} k={k}: err={e_nn:e} tol={tol:e}"
            );
            assert!(
                e_tn <= tol + bf16_round,
                "TN result outside reduction bound at ({i},{j}) m={m} n={n} k={k}: err={e_tn:e} tol={tol:e}"
            );
        }
    }
}

#[test]
#[ignore = "perf micro-bench; run explicitly on an idle GPU"]
fn bf16_bt_perf_vs_pretransposed() {
    let Ok(ctx) = CudaContext::new(0) else {
        eprintln!("skip: no CUDA device 0");
        return;
    };
    let stream = ctx.default_stream();
    let gemm = TensorCoreGemm::new(stream.clone()).unwrap();

    for (m, n, k) in model_shapes() {
        let a_host = build_a(m, k);
        let w_host = build_w(n, k);
        let wt_host = transpose(&w_host, n, k);

        #[allow(deprecated)]
        let a_dev = stream.clone_htod(&a_host).unwrap();
        #[allow(deprecated)]
        let w_dev = stream.clone_htod(&w_host).unwrap();
        #[allow(deprecated)]
        let wt_dev = stream.clone_htod(&wt_host).unwrap();
        let mut c_dev = stream.alloc_zeros::<bf16>(m * n).unwrap();

        let iters = if n >= 262144 { 50 } else { 200 };

        for _ in 0..10 {
            gemm.bf16_matmul_row_major(
                &stream, &a_dev, &wt_dev, &mut c_dev, m as u64, n as u64, k as u64, 1.0, 0.0,
            )
            .unwrap();
            gemm.bf16_matmul_row_major_bt(
                &stream, &a_dev, &w_dev, &mut c_dev, m as u64, n as u64, k as u64, 1.0, 0.0,
            )
            .unwrap();
        }
        stream.synchronize().unwrap();

        let t0 = std::time::Instant::now();
        for _ in 0..iters {
            gemm.bf16_matmul_row_major(
                &stream, &a_dev, &wt_dev, &mut c_dev, m as u64, n as u64, k as u64, 1.0, 0.0,
            )
            .unwrap();
        }
        stream.synchronize().unwrap();
        let nn_us = t0.elapsed().as_micros() as f64 / iters as f64;

        let t1 = std::time::Instant::now();
        for _ in 0..iters {
            gemm.bf16_matmul_row_major_bt(
                &stream, &a_dev, &w_dev, &mut c_dev, m as u64, n as u64, k as u64, 1.0, 0.0,
            )
            .unwrap();
        }
        stream.synchronize().unwrap();
        let tn_us = t1.elapsed().as_micros() as f64 / iters as f64;

        eprintln!(
            "shape m={m} n={n} k={k}: NN(pretransposed)={nn_us:.1}us TN(native)={tn_us:.1}us ratio={:.3}",
            tn_us / nn_us
        );
    }
}

#[test]
fn bf16_bt_fused_qkv_matches_separate() {
    let Ok(ctx) = CudaContext::new(0) else {
        eprintln!("skip: no CUDA device 0");
        return;
    };
    let stream = ctx.default_stream();
    let gemm = TensorCoreGemm::new(stream.clone()).unwrap();

    let families: Vec<(usize, usize, bool)> = vec![(8192, 4096, true), (16384, 2048, false)];
    let k = 5376usize;

    let ms = [
        2usize, 3, 4, 5, 6, 7, 8, 9, 10, 12, 13, 16, 24, 33, 47, 64, 100, 127, 128, 129, 200, 256,
        300,
    ];

    for (q_dim, kv_dim, has_v) in families {
        let n_parts: Vec<usize> = if has_v {
            vec![q_dim, kv_dim, kv_dim]
        } else {
            vec![q_dim, kv_dim]
        };
        let n_total: usize = n_parts.iter().sum();

        let parts: Vec<Vec<bf16>> = n_parts
            .iter()
            .enumerate()
            .map(|(i, &n)| {
                (0..n * k)
                    .map(|j| bf16::from_f32(((j + i * 1_000_003) as f32 * 0.0011).cos()))
                    .collect()
            })
            .collect();
        let mut w_cat: Vec<bf16> = Vec::with_capacity(n_total * k);
        for p in &parts {
            w_cat.extend_from_slice(p);
        }
        #[allow(deprecated)]
        let w_cat_dev = stream.clone_htod(&w_cat).unwrap();
        let part_devs: Vec<_> = parts
            .iter()
            .map(|p| {
                #[allow(deprecated)]
                stream.clone_htod(p).unwrap()
            })
            .collect();

        for &m in &ms {
            let a_host = build_a(m, k);
            #[allow(deprecated)]
            let a_dev = stream.clone_htod(&a_host).unwrap();

            let mut c_sep: Vec<Vec<bf16>> = Vec::new();
            for (p_dev, &n) in part_devs.iter().zip(n_parts.iter()) {
                let mut c = stream.alloc_zeros::<bf16>(m * n).unwrap();
                gemm.bf16_matmul_row_major_bt(
                    &stream, &a_dev, p_dev, &mut c, m as u64, n as u64, k as u64, 1.0, 0.0,
                )
                .unwrap();
                stream.synchronize().unwrap();
                #[allow(deprecated)]
                c_sep.push(stream.memcpy_dtov(&c).unwrap());
            }

            let compare = |label: &str, fused_host: &[bf16], n_cols: &[(usize, usize)]| {
                let mut mismatches = 0usize;
                let mut first = String::new();
                for ((c_part, &(col_off, n)), _) in
                    c_sep.iter().zip(n_cols.iter()).zip(n_parts.iter())
                {
                    for row in 0..m {
                        for col in 0..n {
                            let f = fused_host[row * n_total + col_off + col];
                            let s = c_part[row * n + col];
                            if f.to_bits() != s.to_bits() {
                                if mismatches == 0 {
                                    first = format!(
                                        "{label} fam q={q_dim} kv={kv_dim} m={m} part_off={col_off} \
                                         row={row} col={col}: fused={} sep={}",
                                        f.to_f32(),
                                        s.to_f32()
                                    );
                                }
                                mismatches += 1;
                            }
                        }
                    }
                }
                assert_eq!(
                    mismatches, 0,
                    "{label}: bitwise mismatch fam q={q_dim} kv={kv_dim} m={m}: {first}"
                );
            };

            let col_offsets: Vec<(usize, usize)> = {
                let mut off = 0usize;
                n_parts
                    .iter()
                    .map(|&n| {
                        let e = (off, n);
                        off += n;
                        e
                    })
                    .collect()
            };

            if nv_quant::matmul::fused_qkv_bitwise_safe(m, has_v) {
                let mut c_fused = stream.alloc_zeros::<bf16>(m * n_total).unwrap();
                gemm.bf16_matmul_row_major_bt_det_nosplit(
                    &stream,
                    &a_dev,
                    &w_cat_dev,
                    &mut c_fused,
                    m as u64,
                    n_total as u64,
                    k as u64,
                    1.0,
                    0.0,
                )
                .unwrap();
                stream.synchronize().unwrap();
                #[allow(deprecated)]
                let fused_host = stream.memcpy_dtov(&c_fused).unwrap();
                compare("fused-det", &fused_host, &col_offsets);

                let mut c_sk = stream.alloc_zeros::<bf16>(m * n_total).unwrap();
                gemm.bf16_matmul_row_major_bt_det_splitk(
                    &stream,
                    &a_dev,
                    &w_cat_dev,
                    &mut c_sk,
                    m as u64,
                    n_total as u64,
                    k as u64,
                    1.0,
                    0.0,
                )
                .unwrap();
                stream.synchronize().unwrap();
                #[allow(deprecated)]
                let sk_host = stream.memcpy_dtov(&c_sk).unwrap();
                for rep in 0..3 {
                    let mut c_sk2 = stream.alloc_zeros::<bf16>(m * n_total).unwrap();
                    gemm.bf16_matmul_row_major_bt_det_splitk(
                        &stream,
                        &a_dev,
                        &w_cat_dev,
                        &mut c_sk2,
                        m as u64,
                        n_total as u64,
                        k as u64,
                        1.0,
                        0.0,
                    )
                    .unwrap();
                    stream.synchronize().unwrap();
                    #[allow(deprecated)]
                    let h2 = stream.memcpy_dtov(&c_sk2).unwrap();
                    let mm = sk_host
                        .iter()
                        .zip(h2.iter())
                        .filter(|(x, y)| x.to_bits() != y.to_bits())
                        .count();
                    assert_eq!(
                        mm, 0,
                        "fused-splitk nondeterministic fam q={q_dim} kv={kv_dim} m={m} rep={rep}: {mm} elems"
                    );
                }
            }

            {
                let mut slice_host = vec![bf16::from_f32(0.0); m * n_total];
                for (&(col_off, n), _) in col_offsets.iter().zip(n_parts.iter()) {
                    let mut c = stream.alloc_zeros::<bf16>(m * n).unwrap();
                    gemm.bf16_matmul_row_major_bt_off(
                        &stream,
                        &a_dev,
                        &w_cat_dev,
                        col_off * k,
                        &mut c,
                        m as u64,
                        n as u64,
                        k as u64,
                        1.0,
                        0.0,
                    )
                    .unwrap();
                    stream.synchronize().unwrap();
                    #[allow(deprecated)]
                    let h = stream.memcpy_dtov(&c).unwrap();
                    for row in 0..m {
                        slice_host[row * n_total + col_off..row * n_total + col_off + n]
                            .copy_from_slice(&h[row * n..(row + 1) * n]);
                    }
                }
                compare("slice-off", &slice_host, &col_offsets);
            }
        }
        eprintln!(
            "fused qkv fam q={q_dim} kv={kv_dim} n_total={n_total}: routing-parity bitwise OK across m sweep"
        );
    }
}
