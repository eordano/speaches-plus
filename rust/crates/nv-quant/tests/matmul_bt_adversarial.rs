#![cfg(feature = "cuda")]

use cudarc::driver::CudaContext;
use half::bf16;
use nv_kernels::graph::CudaGraphRunner;
use nv_quant::matmul::TensorCoreGemm;

fn transpose(w: &[bf16], n: usize, k: usize) -> Vec<bf16> {
    let mut t = vec![bf16::from_f32(0.0); n * k];
    for r in 0..n {
        for c in 0..k {
            t[c * n + r] = w[r * k + c];
        }
    }
    t
}

struct XorShift(u64);
impl XorShift {
    fn next(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }
    fn unit(&mut self) -> f32 {
        ((self.next() >> 11) as f64 / (1u64 << 53) as f64) as f32
    }
}

fn build_activation(pattern: &str, m: usize, k: usize, rng: &mut XorShift) -> Vec<bf16> {
    let mut a = vec![bf16::from_f32(0.0); m * k];
    match pattern {
        "denormal" => {
            for i in 0..m * k {
                a[i] = if i % 3 == 0 {
                    bf16::from_bits(((rng.next() % 127 + 1) as u16) | ((i as u16 % 2) << 15))
                } else {
                    bf16::from_f32(rng.unit() * 2.0 - 1.0)
                };
            }
        }

        "amax_spike" => {
            for r in 0..m {
                for c in 0..k {
                    a[r * k + c] = bf16::from_f32((rng.unit() * 2.0 - 1.0) * 1e-2);
                }
                let spike_col = (rng.next() as usize) % k;
                let sign = if rng.next() % 2 == 0 { 1.0 } else { -1.0 };
                a[r * k + spike_col] = bf16::from_f32(sign * 1e30);
            }
        }

        "cancel" => {
            for r in 0..m {
                for c in 0..k {
                    let mag = 1000.0 * (1.0 + rng.unit() * 1e-3);
                    let sign = if c % 2 == 0 { 1.0 } else { -1.0 };
                    a[r * k + c] = bf16::from_f32(sign * mag);
                }
            }
        }

        "ramp" => {
            for i in 0..m * k {
                let e = (i % 77) as i32 - 38;
                a[i] = bf16::from_f32((rng.unit() + 0.5) * (10f32).powi(e));
            }
        }
        _ => unreachable!(),
    }
    a
}

fn build_w(n: usize, k: usize) -> Vec<bf16> {
    (0..n * k)
        .map(|i| bf16::from_f32((i as f32 * 0.0011).cos()))
        .collect()
}

#[test]
fn bf16_bt_ulp_bound_under_adversarial_activations() {
    let Ok(ctx) = CudaContext::new(0) else {
        eprintln!("skip: no CUDA device 0");
        return;
    };
    let stream = ctx.default_stream();
    let gemm = TensorCoreGemm::new(stream.clone()).unwrap();

    let shapes = [
        (5usize, 8192usize, 5376usize),
        (5, 16384, 5376),
        (128, 8192, 5376),
    ];
    let patterns = ["denormal", "amax_spike", "cancel", "ramp"];
    let mut rng = XorShift(0x5eed5eed5eed5eed);

    for &(m, n, k) in &shapes {
        let w_host = build_w(n, k);
        let wt_host = transpose(&w_host, n, k);
        #[allow(deprecated)]
        let w_dev = stream.clone_htod(&w_host).unwrap();
        #[allow(deprecated)]
        let wt_dev = stream.clone_htod(&wt_host).unwrap();

        for pattern in patterns {
            let a_host = build_activation(pattern, m, k, &mut rng);
            #[allow(deprecated)]
            let a_dev = stream.clone_htod(&a_host).unwrap();
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
            for (i, (x, y)) in got_nn.iter().zip(got_tn.iter()).enumerate() {
                if x.to_bits() != y.to_bits() {
                    mismatches.push(i);
                }
            }
            let frac = mismatches.len() as f64 / (m * n) as f64;
            eprintln!(
                "pattern={pattern} m={m} n={n} k={k}: NN/TN bitwise mismatches={}/{} ({:.4}%)",
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
                let bf16_round = exact.abs() * 0.0078125 + f64::MIN_POSITIVE;
                let e_nn = (got_nn[idx].to_f64() - exact).abs();
                let e_tn = (got_tn[idx].to_f64() - exact).abs();
                assert!(
                    e_nn <= tol + bf16_round,
                    "pattern={pattern} NN outside reduction bound at ({i},{j}) m={m} n={n}: \
                     err={e_nn:e} tol={tol:e} exact={exact:e}"
                );
                assert!(
                    e_tn <= tol + bf16_round,
                    "pattern={pattern} TN outside reduction bound at ({i},{j}) m={m} n={n}: \
                     err={e_tn:e} tol={tol:e} exact={exact:e}"
                );
            }

            for (i, v) in got_tn.iter().enumerate() {
                assert!(
                    v.is_finite() || {
                        let r = i / n;
                        let c = i % n;
                        let absum: f64 = (0..k)
                            .map(|p| {
                                (a_host[r * k + p].to_f64() * w_host[c * k + p].to_f64()).abs()
                            })
                            .sum();
                        absum > 3.3e38
                    },
                    "pattern={pattern} m={m} n={n}: non-finite TN output at {i} with sub-f32-range absolute mass"
                );
            }
        }
    }
}

#[test]
fn det_and_tn_routing_bitwise_stable_over_1000_graph_replays() {
    let Ok(ctx) = CudaContext::new(0) else {
        eprintln!("skip: no CUDA device 0");
        return;
    };
    let main = ctx.new_stream().unwrap();
    let gemm = TensorCoreGemm::new(main.clone()).unwrap();

    let m = 5usize;
    let k1 = 5376usize;
    let n1 = 16384usize;
    let n2 = 5376usize;
    let k2 = 8192usize;

    let w1_host = build_w(n1, k1);
    let w2_host = build_w(n2, k2);
    #[allow(deprecated)]
    let w1_dev = main.clone_htod(&w1_host).unwrap();
    #[allow(deprecated)]
    let w2_dev = main.clone_htod(&w2_host).unwrap();

    let mut rng = XorShift(0xabcdef0123456789);

    let mut a1_dev = main.alloc_zeros::<bf16>(m * k1).unwrap();
    let mut a2_dev = main.alloc_zeros::<bf16>(m * k2).unwrap();
    let mut c1_out = main.alloc_zeros::<bf16>(m * n1).unwrap();
    let mut c2_out = main.alloc_zeros::<bf16>(m * n2).unwrap();

    let mut c1_ref = main.alloc_zeros::<bf16>(m * n1).unwrap();
    let mut c2_ref = main.alloc_zeros::<bf16>(m * n2).unwrap();

    let fill = |rng: &mut XorShift, len: usize, pattern: usize| -> Vec<bf16> {
        (0..len)
            .map(|i| match pattern % 3 {
                0 => bf16::from_f32(rng.unit() * 2.0 - 1.0),
                1 => bf16::from_f32(
                    (rng.unit() * 2.0 - 1.0) * if i % 997 == 0 { 1e30 } else { 1e-2 },
                ),
                _ => bf16::from_bits(((rng.next() % 255 + 1) as u16) | (((i % 2) as u16) << 15)),
            })
            .collect()
    };

    unsafe { ctx.disable_event_tracking() };
    let forked = ctx.new_stream().unwrap();

    let a1_h0 = fill(&mut rng, m * k1, 0);
    let a2_h0 = fill(&mut rng, m * k2, 0);
    forked.memcpy_htod(&a1_h0, &mut a1_dev).unwrap();
    forked.memcpy_htod(&a2_h0, &mut a2_dev).unwrap();

    gemm.bf16_matmul_row_major_bt_det(
        &forked,
        &a1_dev,
        &w1_dev,
        &mut c1_out,
        m as u64,
        n1 as u64,
        k1 as u64,
        1.0,
        0.0,
    )
    .unwrap();
    gemm.bf16_matmul_row_major_bt(
        &forked,
        &a2_dev,
        &w2_dev,
        &mut c2_out,
        m as u64,
        n2 as u64,
        k2 as u64,
        1.0,
        0.0,
    )
    .unwrap();
    forked.synchronize().unwrap();

    let mut runner = CudaGraphRunner::new(forked.clone());
    runner
        .run(7, |s| {
            gemm.bf16_matmul_row_major_bt_det(
                s,
                &a1_dev,
                &w1_dev,
                &mut c1_out,
                m as u64,
                n1 as u64,
                k1 as u64,
                1.0,
                0.0,
            )?;
            gemm.bf16_matmul_row_major_bt(
                s,
                &a2_dev,
                &w2_dev,
                &mut c2_out,
                m as u64,
                n2 as u64,
                k2 as u64,
                1.0,
                0.0,
            )?;
            Ok(())
        })
        .unwrap();
    forked.synchronize().unwrap();

    let mut total_replays = 0usize;
    for round in 0..1000 {
        if round % 100 == 0 {
            let a1_h = fill(&mut rng, m * k1, round / 100);
            let a2_h = fill(&mut rng, m * k2, round / 100);
            forked.memcpy_htod(&a1_h, &mut a1_dev).unwrap();
            forked.memcpy_htod(&a2_h, &mut a2_dev).unwrap();
        }
        runner.run(7, |_| unreachable!("must replay")).unwrap();
        total_replays += 1;

        if round % 50 == 0 || round % 100 <= 2 {
            forked.synchronize().unwrap();
            gemm.bf16_matmul_row_major_bt_det(
                &forked,
                &a1_dev,
                &w1_dev,
                &mut c1_ref,
                m as u64,
                n1 as u64,
                k1 as u64,
                1.0,
                0.0,
            )
            .unwrap();
            gemm.bf16_matmul_row_major_bt(
                &forked,
                &a2_dev,
                &w2_dev,
                &mut c2_ref,
                m as u64,
                n2 as u64,
                k2 as u64,
                1.0,
                0.0,
            )
            .unwrap();
            forked.synchronize().unwrap();
            #[allow(deprecated)]
            let g1 = forked.memcpy_dtov(&c1_out).unwrap();
            #[allow(deprecated)]
            let r1 = forked.memcpy_dtov(&c1_ref).unwrap();
            #[allow(deprecated)]
            let g2 = forked.memcpy_dtov(&c2_out).unwrap();
            #[allow(deprecated)]
            let r2 = forked.memcpy_dtov(&c2_ref).unwrap();
            let mm1 = g1
                .iter()
                .zip(r1.iter())
                .filter(|(x, y)| x.to_bits() != y.to_bits())
                .count();
            let mm2 = g2
                .iter()
                .zip(r2.iter())
                .filter(|(x, y)| x.to_bits() != y.to_bits())
                .count();
            assert_eq!(
                mm1, 0,
                "replay {round}: det GEMM diverged from eager ({mm1} elems)"
            );
            assert_eq!(
                mm2, 0,
                "replay {round}: plain TN GEMM diverged from eager ({mm2} elems)"
            );
        }
    }
    forked.synchronize().unwrap();
    eprintln!("det+TN graph routing: {total_replays} replays bitwise-stable vs eager");
    assert_eq!(total_replays, 1000);
}
