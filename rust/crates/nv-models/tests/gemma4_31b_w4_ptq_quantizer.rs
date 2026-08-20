#![cfg(feature = "wgpu")]

use nv_models::gemma4_wgpu::{
    dequantize_w4a16_host, gptq_factor_for_test, quantize_w4a16_host, w4_project, W4ChanStats,
    W4Method, W4PtqSpec, W4ScalePolicy,
};
mod common;
use common::bf16_val as bf16;

fn lcg(s: &mut u64) -> f32 {
    *s = s
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    ((*s >> 32) as u32 as f32 / 2147483648.0) - 1.0
}

fn f32_to_bf16_bits_rne(x: f32) -> u16 {
    let u = x.to_bits();
    let round = ((u >> 16) & 1) + 0x7fff;
    (u.wrapping_add(round) >> 16) as u16
}

fn rel_rms(a: &[f32], b: &[f32]) -> f64 {
    let (mut se, mut sr) = (0f64, 0f64);
    for (x, y) in a.iter().zip(b.iter()) {
        let d = (*y - *x) as f64;
        se += d * d;
        sr += (*x as f64) * (*x as f64);
    }
    (se / sr).sqrt()
}

fn fixture(n: usize, k: usize, seed: u64) -> (Vec<f32>, Vec<f32>) {
    let mut s = seed;
    let w: Vec<f32> = (0..n * k).map(|_| lcg(&mut s) * 0.05).collect();
    let act: Vec<f32> = (0..k)
        .map(|j| if j % 37 == 0 { 8.0 } else { 1.0 })
        .collect();
    (w, act)
}

fn fixture_salient_is_also_largest(n: usize, k: usize, seed: u64) -> (Vec<f32>, Vec<f32>) {
    let mut s = seed;
    let gains: Vec<f32> = (0..k)
        .map(|j| if j % 37 == 0 { 6.0 } else { 1.0 })
        .collect();
    let w: Vec<f32> = (0..n * k)
        .map(|i| lcg(&mut s) * 0.05 * gains[i % k])
        .collect();
    let act: Vec<f32> = (0..k)
        .map(|j| if j % 37 == 0 { 8.0 } else { 1.0 })
        .collect();
    (w, act)
}

fn stats_from_channel_scale(k: usize, block: usize, scale: &[f32], tokens: usize) -> W4ChanStats {
    let mut st = W4ChanStats::new(k, block);
    let mut s = 0xfeed_1234u64;
    for _ in 0..tokens {
        let x: Vec<f32> = (0..k).map(|j| lcg(&mut s) * scale[j]).collect();
        st.observe(&x);
    }
    st
}

fn spec(method: W4Method, group: usize) -> W4PtqSpec {
    W4PtqSpec {
        method,
        group,
        ..W4PtqSpec::default()
    }
}

#[test]
fn rtn_projection_agrees_with_the_packed_w4a16_encoder_bit_for_bit() {
    let (n, k) = (64usize, 256usize);
    let (w, _) = fixture(n, k, 0x1111);
    let bits: Vec<u16> = w.iter().map(|v| f32_to_bf16_bits_rne(*v)).collect();
    let staged: Vec<f32> = bits.iter().map(|b| bf16(*b)).collect();
    for (gs, method, policy) in [
        (32usize, W4Method::Rtn, W4ScalePolicy::Amax),
        (32, W4Method::RtnMse, W4ScalePolicy::MseSearch),
        (16, W4Method::Rtn, W4ScalePolicy::Amax),
        (16, W4Method::RtnMse, W4ScalePolicy::MseSearch),
    ] {
        let want = dequantize_w4a16_host(&quantize_w4a16_host(&bits, n, k, gs, policy));
        let mut got = staged.clone();
        w4_project(&mut got, n, k, &spec(method, gs), None);
        assert_eq!(
            got, want,
            "{method:?}/GS={gs}: the f32 projection and the packed encoder disagree"
        );
    }
    eprintln!("f32 projection == packed w4a16 encoder on 4 (policy, group) pairs");
}

#[test]
fn the_four_bit_code_is_amax_over_seven_on_a_hand_checked_group() {
    let k = 8usize;
    let mut w: Vec<f32> = vec![0.0; k];
    w[0] = 0.7;
    w[1] = -0.7;
    w[2] = 0.1;
    w[3] = 0.35;
    w[4] = -0.35;
    w[5] = 0.05;
    w[6] = 0.0;
    w[7] = 0.699;
    let s = bf16(f32_to_bf16_bits_rne(0.7 / 7.0));
    let want: Vec<f32> = w.iter().map(|v| (v / s).round().min(7.0) * s).collect();
    let mut got = w.clone();
    w4_project(&mut got, 1, k, &spec(W4Method::Rtn, 8), None);
    assert_eq!(got, want, "amax/7 code");
    assert!(
        (want[0] / s - 7.0).abs() < 1e-3,
        "the group max must land on code +7, got {}",
        want[0] / s
    );
    eprintln!("hand oracle: step {s:.6e}, amax on code +7");
}

#[test]
fn the_int8_serving_encode_is_negligible_next_to_the_four_bit_step() {
    use nv_kernels::wgpu_backend::kernels::quant_gemv as qg;
    let (n, k) = (256usize, 1024usize);
    let (w, _) = fixture(n, k, 0x2222);
    let bits: Vec<u16> = w.iter().map(|v| f32_to_bf16_bits_rne(*v)).collect();
    let staged: Vec<f32> = bits.iter().map(|b| bf16(*b)).collect();

    let i8_roundtrip = |src: &[f32]| -> Vec<f32> {
        let b: Vec<u16> = src.iter().map(|v| f32_to_bf16_bits_rne(*v)).collect();
        let (wq, scales) = qg::quantize_groups(&b, n, k, 128, qg::QFormat::Int8);
        let per_row = k / 128;
        let mut out = vec![0f32; n * k];
        for r in 0..n {
            for g in 0..per_row {
                let s = scales[r * per_row + g];
                for i in 0..128 {
                    let idx = r * k + g * 128 + i;
                    let byte = ((wq[idx / 4] >> (8 * (idx % 4))) & 0xff) as u8 as i8;
                    out[idx] = (byte as f32) * s;
                }
            }
        }
        out
    };

    let mut w4 = staged.clone();
    w4_project(&mut w4, n, k, &spec(W4Method::RtnMse, 32), None);
    let served = i8_roundtrip(&w4);
    let control = i8_roundtrip(&staged);

    let e_w4 = rel_rms(&staged, &w4);
    let e_served = rel_rms(&staged, &served);
    let e_encode = rel_rms(&w4, &served);
    let e_control = rel_rms(&staged, &control);
    println!(
        "4-bit step {e_w4:.4e} | int8 encode of the 4-bit grid {e_encode:.4e} | \
         served total {e_served:.4e} | int8-only control {e_control:.4e}"
    );
    println!(
        "encode/step {:.4}   served/step {:.6}",
        e_encode / e_w4,
        e_served / e_w4
    );
    assert!(
        e_encode < e_w4 / 10.0,
        "the serving encode is not negligible: {e_encode:.4e} vs 4-bit step {e_w4:.4e}. \
         The battery would then be measuring int8-of-int4, not int4."
    );
    assert!(
        (e_served / e_w4 - 1.0).abs() < 0.02,
        "serving the 4-bit grid through int8 moved the total by more than 2%: {:.4}",
        e_served / e_w4
    );
}

#[test]
fn awq_never_loses_to_its_own_alpha_zero() {
    let (n, k, block) = (128usize, 512usize, 128usize);
    let (w, act) = fixture(n, k, 0x3333);
    let st = stats_from_channel_scale(k, block, &act, 32);
    let h = st.hdiag_for_test();

    let obj = |q: &[f32]| -> f64 {
        let mut e = 0f64;
        for (rw, rq) in w.chunks(k).zip(q.chunks(k)) {
            for j in 0..k {
                let d = (rw[j] - rq[j]) as f64;
                e += d * d * h[j] as f64;
            }
        }
        e
    };

    let mut base = w.clone();
    w4_project(&mut base, n, k, &spec(W4Method::Wmse, 32), Some(&st));
    let mut awq = w.clone();
    w4_project(&mut awq, n, k, &spec(W4Method::Awq, 32), Some(&st));
    let (o_base, o_awq) = (obj(&base), obj(&awq));
    println!(
        "output-weighted error: wmse {o_base:.6e} -> awq {o_awq:.6e} ({:.4}x)",
        o_awq / o_base
    );
    assert!(
        o_awq <= o_base * 1.001,
        "AWQ lost to its own alpha=0 member: {o_awq:.6e} vs {o_base:.6e}"
    );
    assert!(
        o_awq < o_base * 0.99,
        "AWQ bought nothing on a fixture with an 8x activation outlier channel; the salience \
         is not reaching the scales"
    );

    let (w2, act2) = fixture_salient_is_also_largest(n, k, 0x3334);
    let st2 = stats_from_channel_scale(k, block, &act2, 32);
    let h2 = st2.hdiag_for_test();
    let obj2 = |q: &[f32]| -> f64 {
        let mut e = 0f64;
        for (rw, rq) in w2.chunks(k).zip(q.chunks(k)) {
            for j in 0..k {
                let d = (rw[j] - rq[j]) as f64;
                e += d * d * h2[j] as f64;
            }
        }
        e
    };
    let mut b2 = w2.clone();
    w4_project(&mut b2, n, k, &spec(W4Method::Wmse, 32), Some(&st2));
    let mut a2 = w2.clone();
    w4_project(&mut a2, n, k, &spec(W4Method::Awq, 32), Some(&st2));
    println!(
        "salience-on-the-largest-weight fixture: wmse {:.6e} -> awq {:.6e} ({:.4}x)",
        obj2(&b2),
        obj2(&a2),
        obj2(&a2) / obj2(&b2)
    );
    assert!(
        obj2(&a2) <= obj2(&b2) * 1.001,
        "the search must never come out worse than its own alpha=0"
    );
}

#[test]
fn the_weighted_ladder_trades_weight_rms_for_output_error() {
    let (n, k, block) = (128usize, 512usize, 128usize);
    let (w, act) = fixture(n, k, 0x4444);
    let st = stats_from_channel_scale(k, block, &act, 32);
    let h = st.hdiag_for_test();
    let out_err = |q: &[f32]| -> f64 {
        let mut e = 0f64;
        for (rw, rq) in w.chunks(k).zip(q.chunks(k)) {
            for j in 0..k {
                let d = (rw[j] - rq[j]) as f64;
                e += d * d * h[j] as f64;
            }
        }
        e
    };
    let mut plain = w.clone();
    w4_project(&mut plain, n, k, &spec(W4Method::RtnMse, 32), Some(&st));
    let mut weighted = w.clone();
    w4_project(&mut weighted, n, k, &spec(W4Method::Wmse, 32), Some(&st));
    println!(
        "weight rms {:.4e} -> {:.4e} ({:+.2}%)   output-weighted {:.4e} -> {:.4e} ({:+.2}%)",
        rel_rms(&w, &plain),
        rel_rms(&w, &weighted),
        100.0 * (rel_rms(&w, &weighted) / rel_rms(&w, &plain) - 1.0),
        out_err(&plain),
        out_err(&weighted),
        100.0 * (out_err(&weighted) / out_err(&plain) - 1.0),
    );
    assert!(
        out_err(&weighted) <= out_err(&plain),
        "the weighted ladder must not lose on the objective it optimises"
    );
}

#[test]
fn gptq_factor_is_the_upper_cholesky_of_h_inverse() {
    let n = 16usize;
    let mut s = 0x5555u64;
    let x: Vec<f32> = (0..n * 64).map(|_| lcg(&mut s)).collect();
    let mut gram = vec![0f32; n * n];
    for t in 0..64 {
        for i in 0..n {
            for j in 0..n {
                gram[i * n + j] += x[t * n + i] * x[t * n + j];
            }
        }
    }
    let damp = 0.01f32;
    let u = gptq_factor_for_test(&gram, n, damp).expect("factor");
    let mut mean = 0f64;
    for i in 0..n {
        mean += gram[i * n + i] as f64;
    }
    mean /= n as f64;
    let lam = damp as f64 * mean;

    let mut hinv = vec![0f64; n * n];
    for i in 0..n {
        for j in 0..n {
            let mut acc = 0f64;
            for t in 0..n {
                acc += u[t * n + i] * u[t * n + j];
            }
            hinv[i * n + j] = acc;
        }
    }
    let mut worst = 0f64;
    for i in 0..n {
        for j in 0..n {
            let mut acc = 0f64;
            for t in 0..n {
                let h = gram[t * n + j] as f64 + if t == j { lam } else { 0.0 };
                acc += hinv[i * n + t] * h;
            }
            let want = if i == j { 1.0 } else { 0.0 };
            worst = worst.max((acc - want).abs());
        }
    }
    println!("max |U^T U H - I| = {worst:.3e}");
    assert!(
        worst < 1e-6,
        "gptq_factor is not the inverse factor: {worst:.3e}"
    );
}

#[test]
fn gptq_recovers_output_error_that_rtn_discards() {
    let (n, k, block) = (128usize, 256usize, 128usize);
    let mut s = 0x6666u64;
    let w: Vec<f32> = (0..n * k).map(|_| lcg(&mut s) * 0.05).collect();

    let mut st = W4ChanStats::new(k, block);
    let mut xs: Vec<Vec<f32>> = Vec::new();
    for _ in 0..256 {
        let f: Vec<f32> = (0..8).map(|_| lcg(&mut s)).collect();
        let x: Vec<f32> = (0..k)
            .map(|j| f[j % 8] * 1.5 + f[(j / 8) % 8] * 0.8 + lcg(&mut s) * 0.15)
            .collect();
        st.observe(&x);
        xs.push(x);
    }
    let out_err = |q: &[f32]| -> f64 {
        let mut e = 0f64;
        for x in &xs {
            for (rw, rq) in w.chunks(k).zip(q.chunks(k)) {
                let mut d = 0f64;
                for j in 0..k {
                    d += ((rq[j] - rw[j]) * x[j]) as f64;
                }
                e += d * d;
            }
        }
        e
    };
    let mut rtn = w.clone();
    w4_project(&mut rtn, n, k, &spec(W4Method::Wmse, 32), Some(&st));
    let mut gptq = w.clone();
    let mut sp = spec(W4Method::Gptq, 32);
    sp.block = block;
    w4_project(&mut gptq, n, k, &sp, Some(&st));
    let (e_rtn, e_gptq) = (out_err(&rtn), out_err(&gptq));
    println!(
        "true output error over 256 held-in activations: wmse {e_rtn:.6e} -> gptq {e_gptq:.6e} \
         ({:.4}x); weight rms {:.4e} -> {:.4e}",
        e_gptq / e_rtn,
        rel_rms(&w, &rtn),
        rel_rms(&w, &gptq)
    );
    assert!(
        e_gptq < e_rtn,
        "GPTQ did not beat the weighted ladder on correlated inputs: {e_gptq:.6e} vs {e_rtn:.6e}"
    );
}

#[test]
fn an_unknown_ptq_key_is_refused_rather_than_ignored() {
    let prev = std::env::var("NV_G4_W4PTQ").ok();
    std::env::set_var("NV_G4_W4PTQ", "m=awq,r=ffn,gruop=32");
    let e = std::panic::catch_unwind(W4PtqSpec::from_env).unwrap_err();
    let msg = e
        .downcast_ref::<String>()
        .cloned()
        .unwrap_or_else(|| "?".into());
    eprintln!("{msg}");
    assert!(msg.contains("gruop"), "{msg}");
    std::env::set_var("NV_G4_W4PTQ", "m=awq,r=ffn,g=32,l=0:60");
    let s = W4PtqSpec::from_env().expect("spec");
    assert_eq!(s.method, W4Method::Awq);
    assert!(s.gate_up && s.down && !s.attn);
    assert_eq!((s.lo, s.hi, s.group), (0, 60, 32));
    println!("{} -> {:.4} B/weight", s.label(), s.bytes_per_weight());
    match prev {
        Some(v) => std::env::set_var("NV_G4_W4PTQ", v),
        None => std::env::remove_var("NV_G4_W4PTQ"),
    }
}
