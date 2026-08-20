#![cfg(feature = "wgpu")]

mod common;
use common::wgpu_allow_skip;
use half::bf16;
use nv_kernels::wgpu_backend::device::WgpuContext;
use nv_kernels::wgpu_backend::kernels::quant_gemv::{self, QFormat};

fn ctx_or_skip(test: &str) -> Option<&'static WgpuContext> {
    match WgpuContext::shared() {
        Ok(ctx) => {
            eprintln!("{test}: {}", ctx.summary());
            let st = ctx.qualify();
            if !st.qualified {
                if !wgpu_allow_skip() {
                    panic!(
                        "{test}: wgpu adapter not qualified: {:?}. This gate refuses to \
                         report success without running; set NV_KERNELS_WGPU_ALLOW_SKIP=1 to \
                         skip on purpose.",
                        st.reason
                    );
                }
                eprintln!(
                    "{test}: SKIP (NV_KERNELS_WGPU_ALLOW_SKIP=1) adapter not qualified: {:?}",
                    st.reason
                );
                return None;
            }
            Some(ctx)
        }
        Err(e) => {
            if !wgpu_allow_skip() {
                panic!(
                    "{test}: no wgpu adapter: {e}. This gate refuses to report success \
                     without running; set NV_KERNELS_WGPU_ALLOW_SKIP=1 to skip on purpose."
                );
            }
            eprintln!("{test}: SKIP (NV_KERNELS_WGPU_ALLOW_SKIP=1) no wgpu adapter: {e}");
            None
        }
    }
}

struct Lcg(u64);

impl Lcg {
    fn next(&mut self) -> f32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((self.0 >> 32) as f64 / 4294967296.0) as f32 - 0.5
    }
    fn gauss_bf16(&mut self, n: usize, scale: f32) -> Vec<u16> {
        (0..n)
            .map(|_| {
                let g: f32 = (0..12).map(|_| self.next()).sum();
                bf16::from_f32(g * scale).to_bits()
            })
            .collect()
    }
}

fn f(b: u16) -> f32 {
    f32::from_bits((b as u32) << 16)
}

fn rel(got: &[u16], want: &[f32]) -> f32 {
    let mut se = 0f64;
    let mut sr = 0f64;
    for (g, w) in got.iter().zip(want.iter()) {
        let d = (f(*g) - *w) as f64;
        se += d * d;
        sr += (*w as f64) * (*w as f64);
    }
    (se / sr.max(1e-30)).sqrt() as f32
}

#[test]
fn e4m3_encoder_contract_peak_saturation_rounding_and_subnormals() {
    use nv_kernels::wgpu_backend::kernels::kv_fp8::{decode_e4m3, encode_e4m3};

    assert_eq!(
        decode_e4m3(0x7e),
        448.0,
        "OCP E4M3FN max finite must be 448"
    );
    assert!(
        decode_e4m3(0x7f).is_nan(),
        "0x7f must be the NaN code, not 480"
    );
    assert_eq!(encode_e4m3(448.0), 0x7e);
    assert_eq!(encode_e4m3(-448.0), 0xfe);
    for v in [449.0f32, 1e4, 1e30, f32::INFINITY] {
        assert_eq!(
            encode_e4m3(v),
            0x7e,
            "{v} must saturate to the max finite code"
        );
        assert_eq!(
            encode_e4m3(-v),
            0xfe,
            "-{v} must saturate to the max finite code"
        );
    }
    assert_eq!(encode_e4m3(f32::NAN) & 0x7f, 0x7f);

    assert_eq!(decode_e4m3(0x08), 0.015625, "min normal must be 2^-6");
    assert_eq!(decode_e4m3(0x01), 0.001953125, "min subnormal must be 2^-9");
    assert_eq!(encode_e4m3(0.001953125), 0x01);
    assert_eq!(
        encode_e4m3(0.0009765625),
        0x00,
        "half the min subnormal ties to even = 0"
    );
    assert_eq!(
        encode_e4m3(0.0029296875),
        0x02,
        "1.5 subnormal ulp ties to even = 2"
    );

    for c in 0u16..256 {
        let b = c as u8;
        if (b & 0x7f) == 0x7f {
            continue;
        }
        let v = decode_e4m3(b);
        let back = encode_e4m3(v);
        assert_eq!(
            back,
            if v == 0.0 { b & 0x80 } else { b },
            "code {b:#04x} did not round-trip"
        );
    }

    let mut ties = 0usize;
    for e in -6i32..8 {
        for m in 0..7 {
            let lo = (1.0 + m as f32 / 8.0) * (2f32).powi(e);
            let hi = (1.0 + (m + 1) as f32 / 8.0) * (2f32).powi(e);
            let mid = 0.5 * (lo + hi);
            let got = decode_e4m3(encode_e4m3(mid));
            let want = if m % 2 == 0 { lo } else { hi };
            assert_eq!(
                got, want,
                "tie at {mid} must round to even ({want}), got {got}"
            );
            ties += 1;
        }
    }
    assert!(ties > 90, "expected a full tie sweep, only checked {ties}");
    eprintln!("  e4m3 contract: peak 448 (0x7e), NaN 0x7f, min normal 2^-6, min subnormal 2^-9, {ties} RTNE ties verified");
}

#[test]
fn scale_is_applied_to_the_f32_group_accumulator_not_per_element() {
    let n = 2usize;
    let k = 2048usize;
    let mut rng = Lcg(31337);
    let w = rng.gauss_bf16(n * k, 0.02);
    let x = rng.gauss_bf16(k, 0.3);
    for fmt in [QFormat::E4m3, QFormat::Int8] {
        for group in [0usize, 128] {
            let (wq, sc) = quant_gemv::quantize_groups(&w, n, k, group, fmt);
            let acc = quant_gemv::cpu_gemv_groups(&wq, &sc, &x, n, k, group, fmt);
            let g = if group == 0 { k } else { group };
            let mut per_elem = vec![0f32; n];
            for (r, dst) in per_elem.iter_mut().enumerate() {
                let mut a = 0f32;
                for i in 0..k {
                    let idx = r * k + i;
                    let byte = ((wq[idx / 4] >> (8 * (idx % 4))) & 0xff) as u8;
                    let v = match fmt {
                        QFormat::E4m3 => {
                            nv_kernels::wgpu_backend::kernels::kv_fp8::decode_e4m3(byte)
                        }
                        QFormat::Int8 => (byte as i8) as f32,
                    };
                    a += half::bf16::from_f32(v * sc[r * (k / g) + i / g]).to_f32() * f(x[i]);
                }
                *dst = a;
            }
            let d: f32 = acc
                .iter()
                .zip(per_elem.iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0.0, f32::max);
            let mag = acc.iter().fold(0f32, |a, b| a.max(b.abs()));
            eprintln!(
                "  {:<5} group={group:<4} accumulator-scaled vs bf16-per-element-scaled differ by {:.3e} rel",
                fmt.label(),
                d / mag
            );
            assert!(
                d / mag > 1e-4,
                "if these matched, the kernel would be dropping precision by scaling per element"
            );
        }
    }
}

#[test]
fn group_gemv_matches_cpu_dequant_for_every_format_and_group() {
    let Some(ctx) = ctx_or_skip("group_gemv_matches_cpu_dequant") else {
        return;
    };
    let n = 128usize;
    let mut rng = Lcg(0x51de);
    for k in [5376usize, 8192, 16384] {
        let w = rng.gauss_bf16(n * k, 0.02);
        let x = rng.gauss_bf16(k, 0.3);
        for fmt in [QFormat::E4m3, QFormat::Int8] {
            for group in [0usize, 128, 64, 32] {
                if quant_gemv::group_rule(k, group).is_err() {
                    continue;
                }
                let (wq, sc) = quant_gemv::quantize_groups(&w, n, k, group, fmt);
                assert_eq!(sc.len(), n * quant_gemv::scales_per_row(k, group));
                let want = quant_gemv::cpu_gemv_groups(&wq, &sc, &x, n, k, group, fmt);
                let mut got = vec![0u16; n];
                quant_gemv::gemv_group_bf16(ctx, &wq, &sc, &x, &mut got, n, k, group, fmt).unwrap();
                let r = rel(&got, &want);
                eprintln!(
                    "  k={k} {:<5} group={group:<4} gpu-vs-cpu-dequant rms_rel {r:.3e}",
                    fmt.label()
                );
                assert!(
                    r < 5e-3,
                    "k={k} {:?} group={group}: GPU group gemv disagrees with the CPU dequant \
                     reference by {r:.3e} - that is a kernel bug, not quantization error",
                    fmt
                );
            }
        }
    }
}

#[test]
fn group_zero_gpu_path_reproduces_the_row_scale_kernel() {
    let Some(ctx) = ctx_or_skip("group_zero_reproduces_row_kernel") else {
        return;
    };
    let n = 256usize;
    let k = 5376usize;
    let mut rng = Lcg(99);
    let w = rng.gauss_bf16(n * k, 0.02);
    let x = rng.gauss_bf16(k, 0.3);
    for fmt in [QFormat::E4m3, QFormat::Int8] {
        let (wq, sc) = quant_gemv::quantize_groups(&w, n, k, 0, fmt);
        let mut a = vec![0u16; n];
        let mut b = vec![0u16; n];
        match fmt {
            QFormat::E4m3 => quant_gemv::gemv_fp8_bf16(ctx, &wq, &sc, &x, &mut a, n, k).unwrap(),
            QFormat::Int8 => quant_gemv::gemv_int8_bf16(ctx, &wq, &sc, &x, &mut a, n, k).unwrap(),
        }
        quant_gemv::gemv_group_bf16(ctx, &wq, &sc, &x, &mut b, n, k, 0, fmt).unwrap();
        let af: Vec<f32> = a.iter().map(|v| f(*v)).collect();
        let r = rel(&b, &af);
        eprintln!(
            "  {:<5} row-kernel vs group-kernel(group=0) rms_rel {r:.3e}",
            fmt.label()
        );
        assert!(
            r < 2e-3,
            "{:?}: group=0 must reproduce the row kernel, got {r:.3e}",
            fmt
        );
    }
}

#[test]
fn flat_and_chunked_accumulators_agree_even_on_massive_activations() {
    let Some(ctx) = ctx_or_skip("flat_vs_chunked_accumulator") else {
        return;
    };
    let n = 256usize;
    let k = 8192usize;
    let mut rng = Lcg(0xace);
    let w = rng.gauss_bf16(n * k, 0.02);
    let mut x = rng.gauss_bf16(k, 0.5);
    for i in 0..k {
        if i % 512 == 7 {
            x[i] = bf16::from_f32(if i % 1024 == 7 { 2600.0 } else { -2400.0 }).to_bits();
        }
    }

    let mut exact = vec![0f64; n];
    for (r, dst) in exact.iter_mut().enumerate() {
        let mut acc = 0f64;
        for i in 0..k {
            acc += (f(w[r * k + i]) as f64) * (f(x[i]) as f64);
        }
        *dst = acc;
    }

    let (wq, sc) = quant_gemv::quantize_groups(&w, n, k, 0, QFormat::E4m3);
    let mut flat = vec![0u16; n];
    let mut chunked = vec![0u16; n];
    quant_gemv::gemv_fp8_bf16(ctx, &wq, &sc, &x, &mut flat, n, k).unwrap();
    quant_gemv::gemv_group_bf16(ctx, &wq, &sc, &x, &mut chunked, n, k, 0, QFormat::E4m3).unwrap();

    let mut deq = vec![0f64; n];
    for (r, dst) in deq.iter_mut().enumerate() {
        let mut acc = 0f64;
        for i in 0..k {
            let idx = r * k + i;
            let byte = ((wq[idx / 4] >> (8 * (idx % 4))) & 0xff) as u8;
            acc += (nv_kernels::wgpu_backend::kernels::kv_fp8::decode_e4m3(byte) as f64)
                * (sc[r] as f64)
                * (f(x[i]) as f64);
        }
        *dst = acc;
    }

    let err = |got: &[u16]| -> f64 {
        let mut se = 0f64;
        let mut sr = 0f64;
        for (g, d) in got.iter().zip(deq.iter()) {
            se += ((f(*g) as f64) - d).powi(2);
            sr += d * d;
        }
        (se / sr.max(1e-300)).sqrt()
    };
    let ef = err(&flat);
    let ec = err(&chunked);
    let quant_only = {
        let mut se = 0f64;
        let mut sr = 0f64;
        for (a, b) in deq.iter().zip(exact.iter()) {
            se += (a - b).powi(2);
            sr += b * b;
        }
        (se / sr).sqrt()
    };
    eprintln!(
        "  massive-activation x (|x|_inf {:.0}): f32 summation error vs the exact f64 dequant \
         reference -- flat row accumulator {ef:.4e}, chunked 16-wide accumulator {ec:.4e}; \
         e4m3 quantization error alone is {quant_only:.4e}",
        x.iter().fold(0f32, |a, b| a.max(f(*b).abs()))
    );
    assert!(
        (ef - ec).abs() < 1e-4,
        "flat and chunked f32 accumulation agree to the bf16 output floor here: {ef:.4e} vs {ec:.4e}"
    );
    assert!(
        ec < 3e-3,
        "both should sit at the bf16 output-rounding floor, got {ec:.4e}"
    );
    assert!(
        quant_only > 10.0 * ec,
        "e4m3 quantization error must dominate f32 summation error: {quant_only:.4e} vs {ec:.4e}"
    );
}

#[test]
fn quantization_error_ranking_on_gaussian_rows() {
    let Some(ctx) = ctx_or_skip("quantization_error_ranking") else {
        return;
    };
    let n = 256usize;
    let k = 8192usize;
    let mut rng = Lcg(4242);
    let w = rng.gauss_bf16(n * k, 0.02);
    let x = rng.gauss_bf16(k, 0.3);
    let mut exact = vec![0f32; n];
    for (r, dst) in exact.iter_mut().enumerate() {
        let mut acc = 0f64;
        for i in 0..k {
            acc += (f(w[r * k + i]) * f(x[i])) as f64;
        }
        *dst = acc as f32;
    }
    let mut table: Vec<(String, f32)> = Vec::new();
    for fmt in [QFormat::E4m3, QFormat::Int8] {
        for group in [0usize, 128, 32] {
            let (wq, sc) = quant_gemv::quantize_groups(&w, n, k, group, fmt);
            let mut got = vec![0u16; n];
            quant_gemv::gemv_group_bf16(ctx, &wq, &sc, &x, &mut got, n, k, group, fmt).unwrap();
            let r = rel(&got, &exact);
            eprintln!(
                "  {:<5} group={group:<4} rms_rel vs exact {r:.4e}",
                fmt.label()
            );
            table.push((format!("{}/{group}", fmt.label()), r));
        }
    }
    let get = |k: &str| table.iter().find(|(n, _)| n == k).unwrap().1;
    assert!(
        get("e4m3/0") / get("e4m3/32") < 1.6,
        "e4m3 must be nearly granularity-insensitive: row {:.3e} vs group32 {:.3e}",
        get("e4m3/0"),
        get("e4m3/32")
    );
    assert!(
        get("int8/128") * 2.0 < get("e4m3/0"),
        "int8 group=128 must beat e4m3 per-row by >2x: {:.3e} vs {:.3e}",
        get("int8/128"),
        get("e4m3/0")
    );
}
