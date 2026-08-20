#![cfg(feature = "wgpu")]

mod common;
use common::ctx_or_skip;
use half::bf16;
use nv_kernels::wgpu_backend::device::WgpuContext;
use nv_kernels::wgpu_backend::kernels::kv_fp8;
use nv_kernels::wgpu_backend::WgpuError;
use common::sample_bf16;

fn ref_decode_e4m3(b: u8) -> f32 {
    let mag = b & 0x7f;
    if mag == 0x7f {
        return f32::NAN;
    }
    let e = (mag >> 3) as i32;
    let m = (mag & 7) as f32;
    let v = if e == 0 {
        m * 0.001_953_125
    } else {
        (1.0 + m * 0.125) * (2f32).powi(e - 7)
    };
    if b & 0x80 != 0 {
        -v
    } else {
        v
    }
}

fn ref_encode_e4m3_bruteforce(x: f32) -> u8 {
    if x.is_nan() {
        return 0x7f;
    }
    let sign = if x.is_sign_negative() { 0x80u8 } else { 0 };
    if x.abs() > 448.0 {
        return sign | 0x7e;
    }
    let mut best: Option<(f64, u8)> = None;
    for code in 0u16..256 {
        let code = code as u8;
        if code & 0x7f == 0x7f {
            continue;
        }
        let v = ref_decode_e4m3(code) as f64;
        let d = (v - x as f64).abs();
        match best {
            None => best = Some((d, code)),
            Some((bd, bc)) => {
                if d < bd {
                    best = Some((d, code));
                } else if d == bd {
                    let cur_even = (bc & 1) == 0;
                    let new_even = (code & 1) == 0;
                    if new_even && !cur_even {
                        best = Some((d, code));
                    } else if new_even == cur_even && ref_decode_e4m3(code) == ref_decode_e4m3(bc) {
                        let want_sign = x.is_sign_negative();
                        if ((code & 0x80) != 0) == want_sign {
                            best = Some((d, code));
                        }
                    }
                }
            }
        }
    }
    best.unwrap().1
}

fn bf16_of(bits: u16) -> f32 {
    f32::from_bits((bits as u32) << 16)
}

struct Oracle {
    fp8: Vec<u8>,
    scales: Vec<f32>,
}

fn oracle_quantize(
    x: &[u16],
    base_fp8: &[u8],
    base_scales: &[f32],
    start: usize,
    n_tokens: usize,
    n_kv: usize,
    head_dim: usize,
    ring: usize,
) -> Oracle {
    let mut fp8 = base_fp8.to_vec();
    let mut scales = base_scales.to_vec();
    for token in 0..n_tokens {
        for kv_head in 0..n_kv {
            let mut slot = start + token;
            if ring > 0 {
                slot %= ring;
            }
            let base_src = (token * n_kv + kv_head) * head_dim;
            let base_dst = (slot * n_kv + kv_head) * head_dim;
            let mut amax = 0.0f32;
            for d in 0..head_dim {
                let a = bf16_of(x[base_src + d]).abs();
                if a > amax {
                    amax = a;
                }
            }
            let scale = if amax > 0.0 { amax / 448.0 } else { 1.0 };
            let inv = if amax > 0.0 { 448.0 / amax } else { 1.0 };
            scales[slot * n_kv + kv_head] = scale;
            for d in 0..head_dim {
                let v = bf16_of(x[base_src + d]) * inv;
                fp8[base_dst + d] = ref_encode_e4m3_bruteforce(v);
            }
        }
    }
    Oracle { fp8, scales }
}

fn oracle_dequantize(
    src: &[u8],
    scales: &[f32],
    start: usize,
    n_tokens: usize,
    n_kv: usize,
    head_dim: usize,
    ring: usize,
) -> Vec<u16> {
    let mut out = vec![0u16; n_tokens * n_kv * head_dim];
    for token in 0..n_tokens {
        for kv_head in 0..n_kv {
            let mut slot = start + token;
            if ring > 0 {
                slot %= ring;
            }
            let base = (slot * n_kv + kv_head) * head_dim;
            let obase = (token * n_kv + kv_head) * head_dim;
            let scale = scales[slot * n_kv + kv_head];
            for d in 0..head_dim {
                let v = ref_decode_e4m3(src[base + d]) * scale;
                out[obase + d] = bf16::from_f32(v).to_bits();
            }
        }
    }
    out
}

#[test]
fn encode_e4m3_matches_bruteforce_over_every_bf16_input() {
    let mut checked = 0usize;
    for bits in 0u32..=0xffff {
        let v = bf16_of(bits as u16);
        if !v.is_finite() {
            continue;
        }
        let got = kv_fp8::encode_e4m3(v);
        let want = ref_encode_e4m3_bruteforce(v);
        assert_eq!(
            got, want,
            "bf16 {bits:#06x} = {v}: got {got:#04x} want {want:#04x}"
        );
        checked += 1;
    }
    assert!(checked > 60000, "only checked {checked} bf16 patterns");
    eprintln!("encode_e4m3: byte-exact on {checked} bf16 inputs");
}

#[test]
fn encode_e4m3_matches_bruteforce_on_scaled_products() {
    for i in 1..40000u32 {
        let v = (i as f32) * 0.011_37 - 220.0;
        let got = kv_fp8::encode_e4m3(v);
        let want = ref_encode_e4m3_bruteforce(v);
        assert_eq!(got, want, "{v}: got {got:#04x} want {want:#04x}");
    }
}

#[test]
fn wgpu_quantize_matches_the_cpu_oracle_byte_for_byte() {
    let Some(ctx) = ctx_or_skip("wgpu_quantize_matches_the_cpu_oracle_byte_for_byte") else {
        return;
    };
    let (n_tokens, n_kv, head_dim, slots) = (5usize, 3usize, 128usize, 9usize);
    let x = sample_bf16(n_tokens * n_kv * head_dim, 7);
    let base_fp8: Vec<u8> = (0..slots * n_kv * head_dim)
        .map(|i| (i % 251) as u8)
        .collect();
    let base_scales: Vec<f32> = (0..slots * n_kv).map(|i| -1.0 - i as f32).collect();

    let start = 2i32;
    let mut fp8 = base_fp8.clone();
    let mut scales = base_scales.clone();
    kv_fp8::quantize_kv_fp8(
        ctx,
        &x,
        &mut fp8,
        &mut scales,
        &[start],
        n_tokens,
        n_kv,
        head_dim,
        0,
    )
    .expect("quantize");

    let want = oracle_quantize(
        &x,
        &base_fp8,
        &base_scales,
        start as usize,
        n_tokens,
        n_kv,
        head_dim,
        0,
    );
    let diff = fp8
        .iter()
        .zip(want.fp8.iter())
        .filter(|(a, b)| a != b)
        .count();
    assert_eq!(diff, 0, "{diff}/{} fp8 bytes differ", fp8.len());
    for (i, (g, w)) in scales.iter().zip(want.scales.iter()).enumerate() {
        assert_eq!(g.to_bits(), w.to_bits(), "scale {i}: got {g} want {w}");
    }
    eprintln!(
        "quantize: byte-exact on {} bytes, bit-exact on {} scales",
        fp8.len(),
        scales.len()
    );
}

#[test]
fn wgpu_quantize_is_exact_on_rounding_ties() {
    let Some(ctx) = ctx_or_skip("wgpu_quantize_is_exact_on_rounding_ties") else {
        return;
    };
    let (n_tokens, n_kv, head_dim) = (2usize, 2usize, 64usize);
    let ties: [f32; 16] = [
        1.0625,
        1.1875,
        1.3125,
        1.4375,
        1.5625,
        1.6875,
        1.8125,
        1.9375,
        0.000_976_562_5,
        0.002_929_687_5,
        0.004_882_812_5,
        0.006_835_937_5,
        0.008_789_062_5,
        0.010_742_187_5,
        0.012_695_312_5,
        0.014_648_437_5,
    ];
    let mut x = vec![0u16; n_tokens * n_kv * head_dim];
    for token in 0..n_tokens {
        for kv_head in 0..n_kv {
            let base = (token * n_kv + kv_head) * head_dim;
            x[base] = bf16::from_f32(448.0).to_bits();
            for d in 1..head_dim {
                let mut v = ties[(d + kv_head) % ties.len()];
                if (d + token) % 2 == 0 {
                    v = -v;
                }
                x[base + d] = bf16::from_f32(v).to_bits();
            }
        }
    }
    let slots = n_tokens;
    let base_fp8 = vec![0u8; slots * n_kv * head_dim];
    let base_scales = vec![0f32; slots * n_kv];
    let mut fp8 = base_fp8.clone();
    let mut scales = base_scales.clone();
    kv_fp8::quantize_kv_fp8(
        ctx,
        &x,
        &mut fp8,
        &mut scales,
        &[0],
        n_tokens,
        n_kv,
        head_dim,
        0,
    )
    .expect("quantize");
    let want = oracle_quantize(&x, &base_fp8, &base_scales, 0, n_tokens, n_kv, head_dim, 0);
    for (i, (g, w)) in fp8.iter().zip(want.fp8.iter()).enumerate() {
        assert_eq!(g, w, "tie byte {i}: got {g:#04x} want {w:#04x}");
    }
    for (i, s) in scales.iter().enumerate() {
        assert_eq!(
            s.to_bits(),
            1.0f32.to_bits(),
            "scale {i} should be exactly 1.0, got {s}"
        );
    }
}

#[test]
fn wgpu_quantize_honours_the_ring_wrap() {
    let Some(ctx) = ctx_or_skip("wgpu_quantize_honours_the_ring_wrap") else {
        return;
    };
    let (n_tokens, n_kv, head_dim, ring) = (3usize, 2usize, 64usize, 4usize);
    let x = sample_bf16(n_tokens * n_kv * head_dim, 3);
    let base_fp8 = vec![0xabu8; ring * n_kv * head_dim];
    let base_scales = vec![-7.0f32; ring * n_kv];
    let mut fp8 = base_fp8.clone();
    let mut scales = base_scales.clone();
    let start = 3i32;
    kv_fp8::quantize_kv_fp8(
        ctx,
        &x,
        &mut fp8,
        &mut scales,
        &[start],
        n_tokens,
        n_kv,
        head_dim,
        ring,
    )
    .expect("quantize");
    let want = oracle_quantize(
        &x,
        &base_fp8,
        &base_scales,
        start as usize,
        n_tokens,
        n_kv,
        head_dim,
        ring,
    );
    assert_eq!(fp8, want.fp8);
    for (g, w) in scales.iter().zip(want.scales.iter()) {
        assert_eq!(g.to_bits(), w.to_bits());
    }
    let untouched = 2usize;
    for d in 0..head_dim {
        assert_eq!(
            fp8[(untouched * n_kv) * head_dim + d],
            0xab,
            "slot 2 must be untouched"
        );
    }
}

#[test]
fn wgpu_dequantize_matches_the_cpu_oracle_bit_for_bit() {
    let Some(ctx) = ctx_or_skip("wgpu_dequantize_matches_the_cpu_oracle_bit_for_bit") else {
        return;
    };
    let (n_tokens, n_kv, head_dim, slots) = (6usize, 4usize, 128usize, 8usize);
    let src: Vec<u8> = (0..slots * n_kv * head_dim)
        .map(|i| {
            let b = (i % 256) as u8;
            if b & 0x7f == 0x7f {
                0x12
            } else {
                b
            }
        })
        .collect();
    let scales: Vec<f32> = (0..slots * n_kv)
        .map(|i| 0.002 + (i as f32) * 0.0037)
        .collect();

    let mut out = vec![0u16; n_tokens * n_kv * head_dim];
    kv_fp8::dequantize_kv_fp8(ctx, &src, &scales, &mut out, n_tokens, n_kv, head_dim)
        .expect("dequantize");
    let want = oracle_dequantize(&src, &scales, 0, n_tokens, n_kv, head_dim, 0);
    let diff = out.iter().zip(want.iter()).filter(|(a, b)| a != b).count();
    assert_eq!(diff, 0, "{diff}/{} bf16 words differ", out.len());

    let start = 2usize;
    let ring = 5usize;
    let mut out2 = vec![0u16; n_tokens * n_kv * head_dim];
    kv_fp8::dequantize_kv_fp8_ring(
        ctx, &src, &scales, &mut out2, start, n_tokens, n_kv, head_dim, ring,
    )
    .expect("dequantize ring");
    let want2 = oracle_dequantize(&src, &scales, start, n_tokens, n_kv, head_dim, ring);
    assert_eq!(out2, want2);
    eprintln!("dequantize: bit-exact on {} bf16 words", out.len());
}

#[test]
fn wgpu_round_trip_stays_inside_the_fp8_grid() {
    let Some(ctx) = ctx_or_skip("wgpu_round_trip_stays_inside_the_fp8_grid") else {
        return;
    };
    let (n_tokens, n_kv, head_dim) = (4usize, 2usize, 128usize);
    let x = sample_bf16(n_tokens * n_kv * head_dim, 11);
    let mut fp8 = vec![0u8; n_tokens * n_kv * head_dim];
    let mut scales = vec![0f32; n_tokens * n_kv];
    kv_fp8::quantize_kv_fp8(
        ctx,
        &x,
        &mut fp8,
        &mut scales,
        &[0],
        n_tokens,
        n_kv,
        head_dim,
        0,
    )
    .expect("quantize");
    let mut back = vec![0u16; x.len()];
    kv_fp8::dequantize_kv_fp8(ctx, &fp8, &scales, &mut back, n_tokens, n_kv, head_dim)
        .expect("dequantize");

    let mut max_rel = 0f32;
    for (a, b) in x.iter().zip(back.iter()) {
        let want = bf16_of(*a);
        let got = bf16_of(*b);
        let denom = want.abs().max(1e-3);
        max_rel = max_rel.max((want - got).abs() / denom);
    }
    eprintln!("round trip max relative error {max_rel:.5}");
    assert!(max_rel < 0.07, "round trip max relative error {max_rel}");
}

#[test]
fn odd_head_dim_is_rejected_rather_than_corrupting_memory() {
    let Some(ctx) = ctx_or_skip("odd_head_dim_is_rejected_rather_than_corrupting_memory") else {
        return;
    };
    let x = vec![0u16; 6];
    let mut fp8 = vec![0u8; 6];
    let mut scales = vec![0f32; 1];
    let e = kv_fp8::quantize_kv_fp8(ctx, &x, &mut fp8, &mut scales, &[0], 1, 1, 6, 0).unwrap_err();
    assert!(matches!(e, WgpuError::Unsupported(_)), "{e}");

    let mut out = vec![0u16; 5];
    let e = kv_fp8::dequantize_kv_fp8(ctx, &[0u8; 5], &[1.0f32], &mut out, 1, 1, 5).unwrap_err();
    assert!(matches!(e, WgpuError::Unsupported(_)), "{e}");
}

fn bf16_bits_of(x: f32) -> u16 {
    if x.is_nan() {
        return 0x7fc0;
    }
    let b = x.to_bits();
    let r = 0x7fff + ((b >> 16) & 1);
    ((b + r) >> 16) as u16
}

fn scale_guard_cpu_reference(x: &[u16]) -> (Vec<u8>, f32) {
    let amax = x.iter().fold(0f32, |a, &b| a.max(bf16_of(b).abs()));
    let scale = if amax > 0.0 {
        kv_fp8::div_rn(amax, kv_fp8::FP8_E4M3_MAX)
    } else {
        1.0
    };
    let inv = if amax > 0.0 {
        kv_fp8::FP8_E4M3_MAX / amax
    } else {
        1.0
    };
    (
        x.iter()
            .map(|&b| kv_fp8::encode_e4m3(bf16_of(b) * inv))
            .collect(),
        scale,
    )
}

#[test]
fn huge_amax_does_not_zero_the_head_vector() {
    let Some(ctx) = ctx_or_skip("huge_amax_does_not_zero_the_head_vector") else {
        return;
    };
    let head_dim = 256usize;

    for &exp in &[100i32, 120, 126, 127] {
        let amax = (2f32).powi(exp);
        assert!(amax.is_finite(), "test bug: 2^{exp} is not finite in f32");
        let mut x = vec![0u16; head_dim];
        for (d, slot) in x.iter_mut().enumerate() {
            *slot = bf16_bits_of(amax * (2f32).powi(-((d % 16) as i32)));
        }
        let mut got = vec![0u8; head_dim];
        let mut scales = vec![0f32; 1];
        kv_fp8::quantize_kv_fp8(ctx, &x, &mut got, &mut scales, &[0i32], 1, 1, head_dim, 0)
            .expect("quantize_kv_fp8");
        let (want, want_scale) = scale_guard_cpu_reference(&x);

        let nonzero = got.iter().filter(|&&b| b & 0x7f != 0).count();
        assert!(
            nonzero > 0,
            "amax=2^{exp}: kernel zeroed the entire head vector ({head_dim} elements). \
             This is the subnormal-reciprocal bug: 1.0/amax underflows to a subnormal, the \
             GPU flushes it to zero, and kv_div_rn then returns 0 for inv_scale."
        );
        assert!(
            want.iter().any(|b| b & 0x7f != 0),
            "amax=2^{exp}: the CPU reference is itself all-zero, so the byte comparison below \
             would pass against any output"
        );
        assert_eq!(
            got, want,
            "amax=2^{exp}: fp8 bytes disagree with the CPU reference encoder"
        );
        assert_eq!(
            scales[0].to_bits(),
            want_scale.to_bits(),
            "amax=2^{exp}: stored scale disagrees with the CPU reference"
        );
    }
}

#[test]
fn div_rn_matches_exact_division_across_the_subnormal_reciprocal_boundary() {
    for &exp in &[120i32, 125, 126, 127] {
        let b = (2f32).powi(exp);
        assert!(b.is_finite(), "test bug: 2^{exp} is not finite in f32");
        let got = kv_fp8::div_rn(kv_fp8::FP8_E4M3_MAX, b);
        let want = kv_fp8::FP8_E4M3_MAX / b;
        assert_eq!(
            got.to_bits(),
            want.to_bits(),
            "div_rn(448, 2^{exp}) = {got:e}, exact = {want:e}; 1/2^{exp} = {:e}",
            1.0f32 / b
        );
    }
    let b = f32::MAX;
    assert_eq!(
        kv_fp8::div_rn(kv_fp8::FP8_E4M3_MAX, b).to_bits(),
        (kv_fp8::FP8_E4M3_MAX / b).to_bits(),
        "div_rn(448, f32::MAX) must match exact division"
    );
}

#[test]
fn non_finite_amax_is_a_known_cross_backend_divergence() {
    let got = kv_fp8::div_rn(kv_fp8::FP8_E4M3_MAX, f32::INFINITY);
    assert!(
        got.is_nan(),
        "div_rn(448, inf) = {got:e}; this test records the CURRENT wgpu behaviour. \
         The CUDA kernel computes 448.0f/inf = 0 and then encodes every element to e4m3 0x00, \
         so an Inf in K or V makes the two backends disagree on the whole head vector: \
         wgpu yields e4m3 NaN (0x7f), CUDA yields signed zero. Neither is 'right'; the \
         divergence is only reachable once activations have already blown up. If this \
         assertion starts failing, the two backends may have been brought into agreement -- \
         update the note in kv_fp8.wgsl and kv_fp8.cu together."
    );
}

#[test]
fn kv_scaling_is_per_token_per_head_not_per_row() {
    let Some(ctx) = ctx_or_skip("kv_scaling_is_per_token_per_head_not_per_row") else {
        return;
    };
    let head_dim = 256usize;
    let n_kv = 4usize;
    let n_tokens = 3usize;

    let mut x = vec![0u16; n_tokens * n_kv * head_dim];
    for t in 0..n_tokens {
        for h in 0..n_kv {
            let mag = (2f32).powi((t * n_kv + h) as i32 * 3 - 8);
            for d in 0..head_dim {
                x[(t * n_kv + h) * head_dim + d] = bf16_bits_of(mag * (1.0 + (d as f32) / 256.0));
            }
        }
    }

    let mut out = vec![0u8; n_tokens * n_kv * head_dim];
    let mut scales = vec![0f32; n_tokens * n_kv];
    kv_fp8::quantize_kv_fp8(
        ctx,
        &x,
        &mut out,
        &mut scales,
        &[0i32],
        n_tokens,
        n_kv,
        head_dim,
        0,
    )
    .expect("quantize_kv_fp8");

    assert_eq!(scales.len(), n_tokens * n_kv);
    for i in 1..scales.len() {
        assert!(
            scales[i] > scales[i - 1],
            "scale[{i}]={} not greater than scale[{}]={}; each (token, kv_head) pair must carry \
             its own amax-derived scale, so a 8x magnitude ramp across pairs must show up as a \
             strictly increasing scale vector",
            scales[i],
            i - 1,
            scales[i - 1]
        );
    }

    let ratio = scales[scales.len() - 1] / scales[0];
    let expected = (2f32).powi(((n_tokens * n_kv - 1) * 3) as i32);
    assert!(
        (ratio / expected - 1.0).abs() < 1e-3,
        "scale ratio {ratio:e} != expected {expected:e}: per-pair scaling is not tracking amax"
    );

    let mut deq = vec![0u16; n_tokens * n_kv * head_dim];
    kv_fp8::dequantize_kv_fp8(ctx, &out, &scales, &mut deq, n_tokens, n_kv, head_dim)
        .expect("dequantize_kv_fp8");
    let mut se = 0f64;
    let mut sx = 0f64;
    for i in 0..deq.len() {
        let a = bf16_of(x[i]) as f64;
        let b = bf16_of(deq[i]) as f64;
        se += (a - b) * (a - b);
        sx += a * a;
    }
    assert!(
        sx > 0.0,
        "the fixture decoded to all zeros; the round-trip ratio below would compare nothing"
    );
    let rms_rel = (se / sx).sqrt();
    assert!(
        rms_rel < 0.05,
        "per-(token,head) fp8 KV round-trip rms_rel={rms_rel:.5} exceeds the 5% e4m3 budget; \
         a per-row (one scale for all {n_kv} heads) scheme would be the likely cause"
    );
    eprintln!(
        "kv fp8 round-trip rms_rel = {rms_rel:.5} over {} elements",
        deq.len()
    );
}
