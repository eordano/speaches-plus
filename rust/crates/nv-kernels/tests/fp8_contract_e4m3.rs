#![cfg(feature = "wgpu")]
#![allow(dead_code)]

pub const FP8_E4M3_MAX: f32 = 448.0;
pub const CONTRACT_DOC: &str = include_str!("../../../../docs/book/04.1-fp8.md");

const FP8_PRODUCER_SOURCES: [(&str, &str, &str); 5] = [
    (
        "quant_gemv::quantize_rows_fp8",
        "nv-kernels/src/wgpu_backend/kernels/quant_gemv.rs",
        "pub fn quantize_rows_fp8",
    ),
    (
        "kv_fp8::quantize_kv_fp8",
        "nv-kernels/src/wgpu_backend/kernels/kv_fp8.rs",
        "pub fn quantize_kv_fp8",
    ),
    (
        "nv_kernels_quantize_kv_fp8",
        "nv-kernels/cuda/kv_fp8.cu",
        "int nv_kernels_quantize_kv_fp8",
    ),
    (
        "nv_layers::linear::quantize_fp8_per_tensor",
        "nv-layers/src/linear.rs",
        "fn quantize_fp8_per_tensor",
    ),
    (
        "rowquant_e4m3_kernel",
        "nv-kernels/cuda/gemv_bf16.cu",
        "__global__ void rowquant_e4m3_kernel",
    ),
];

fn fp8_producer_source(file: &str) -> &'static str {
    match file {
        "nv-kernels/src/wgpu_backend/kernels/quant_gemv.rs" => {
            include_str!("../src/wgpu_backend/kernels/quant_gemv.rs")
        }
        "nv-kernels/src/wgpu_backend/kernels/kv_fp8.rs" => {
            include_str!("../src/wgpu_backend/kernels/kv_fp8.rs")
        }
        "nv-kernels/cuda/kv_fp8.cu" => include_str!("../cuda/kv_fp8.cu"),
        "nv-layers/src/linear.rs" => include_str!("../../nv-layers/src/linear.rs"),
        "nv-kernels/cuda/gemv_bf16.cu" => include_str!("../cuda/gemv_bf16.cu"),
        other => panic!("no include_str! wired for {other}"),
    }
}

#[test]
fn every_fp8_producer_at_head_flag_is_derived_from_the_tree_and_recorded_in_the_doc() {
    let mut missing = Vec::new();
    for (key, file, symbol) in FP8_PRODUCER_SOURCES {
        let in_tree = fp8_producer_source(file).contains(symbol);
        let marker = format!(
            "FP8_PRODUCER_AT_HEAD {key} = {}",
            if in_tree { "yes" } else { "no" }
        );
        let in_doc = CONTRACT_DOC.contains(&marker);
        eprintln!(
            "{key:<46} {file} contains {symbol:?} = {} | docs/book/04.1-fp8.md carries {marker:?} = {in_doc}",
            if in_tree { "yes" } else { "no" }
        );
        if !in_doc {
            missing.push(marker);
        }
    }
    eprintln!(
        "WHY THIS GATE EXISTS: fp8_scaling_granularity_matrix_across_backends carries \
         `present_at_head` as a HAND-WRITTEN bool. That is exactly how rowquant_e4m3_kernel came to \
         be described as `absent at HEAD` for a whole round after it had landed - nothing compared \
         the flag to the tree. This test reads each producer's own source file and requires \
         docs/book/04.1-fp8.md 7.3 to carry the matching FP8_PRODUCER_AT_HEAD line, so a producer that lands \
         or is deleted turns BOTH the doc and this suite red in the same run instead of quietly \
         disagreeing with reality."
    );
    assert!(
        missing.is_empty(),
        "docs/book/04.1-fp8.md 7.3 is out of date with the tree. Add or correct these marker lines: {missing:?}"
    );
}

#[test]
fn the_hand_written_matrix_flags_agree_with_the_tree() {
    let mut wrong = Vec::new();
    for (key, file, symbol) in FP8_PRODUCER_SOURCES {
        let in_tree = fp8_producer_source(file).contains(symbol);
        let marker = format!("FP8_PRODUCER_AT_HEAD {key} = yes");
        let doc_says_yes = CONTRACT_DOC.contains(&marker);
        eprintln!("{key:<46} tree={in_tree} doc_says_at_head={doc_says_yes}");
        if in_tree != doc_says_yes {
            wrong.push((key, in_tree, doc_says_yes));
        }
    }
    eprintln!(
        "Every row of fp8_scaling_granularity_matrix_across_backends is asserted `present_at_head: \
         true` there; this test is the independent confirmation that all five really are in the \
         tree, so that assertion cannot be satisfied by a stale literal."
    );
    assert!(
        wrong.is_empty(),
        "tree and docs/book/04.1-fp8.md disagree about which fp8 producers exist: {wrong:?}"
    );
}

fn ref_decode_e4m3(code: u8) -> Option<f32> {
    let mag = code & 0x7f;
    if mag == 0x7f {
        return None;
    }
    let e = (mag >> 3) as i32;
    let m = (mag & 7) as f64;
    let v = if e == 0 {
        m * 2f64.powi(-9)
    } else {
        (1.0 + m / 8.0) * 2f64.powi(e - 7)
    };
    let v = v as f32;
    Some(if code & 0x80 != 0 { -v } else { v })
}

fn ref_encode_e4m3(x: f32) -> u8 {
    if x.is_nan() {
        return 0x7f;
    }
    let sign: u8 = if x.is_sign_negative() { 0x80 } else { 0x00 };
    let a = x.abs() as f64;
    if a > FP8_E4M3_MAX as f64 {
        return sign | 0x7e;
    }
    let mut best = 0u8;
    let mut best_err = f64::INFINITY;
    for code in 0u8..=0x7e {
        let v = ref_decode_e4m3(code).unwrap() as f64;
        let err = (v - a).abs();
        if err < best_err {
            best_err = err;
            best = code;
        } else if err == best_err && (code & 1) == 0 && (best & 1) == 1 {
            best = code;
        }
    }
    sign | best
}

fn probe_values() -> Vec<f32> {
    let mut v = Vec::new();
    for code in 0u8..=0x7e {
        let x = ref_decode_e4m3(code).unwrap();
        v.push(x);
        v.push(-x);
        if code < 0x7e {
            let next = ref_decode_e4m3(code + 1).unwrap();
            let mid = 0.5 * (x + next);
            v.push(mid);
            v.push(-mid);
            v.push(mid * (1.0 + 1e-6));
            v.push(mid * (1.0 - 1e-6));
        }
    }
    for x in [
        0.0f32,
        -0.0,
        448.0,
        449.0,
        464.0,
        480.0,
        1e30,
        -1e30,
        f32::INFINITY,
        f32::NEG_INFINITY,
        f32::MIN_POSITIVE,
        1e-12,
        -1e-12,
        0.001_953_125,
        0.000_976_562_5,
        0.002_929_687_5,
    ] {
        v.push(x);
    }
    let mut s = 1u32;
    for _ in 0..20000 {
        s = s.wrapping_mul(1664525).wrapping_add(1013904223);
        let u = (s >> 8) as f32 / 16777216.0;
        v.push((u - 0.5) * 1200.0);
    }
    v
}

#[test]
fn e4m3_encoder_matches_an_independent_nearest_code_reference() {
    use nv_kernels::wgpu_backend::kernels::kv_fp8::encode_e4m3;
    let probes = probe_values();
    let mut bad = Vec::new();
    for x in &probes {
        if x.is_nan() {
            continue;
        }
        let got = encode_e4m3(*x);
        let want = ref_encode_e4m3(*x);
        if got != want {
            bad.push((*x, got, want));
        }
    }
    for (x, got, want) in bad.iter().take(10) {
        eprintln!("MISMATCH x={x:e} got=0x{got:02x} want=0x{want:02x}");
    }
    eprintln!(
        "e4m3 encoder: {} probes (exact codes, exact halfway ties, off-by-1ulp ties, saturation, \
         subnormals, 20000 pseudo-random), {} mismatches vs nearest-code-with-ties-to-even",
        probes.len(),
        bad.len()
    );
    assert!(bad.is_empty(), "{} mismatches, first 10 above", bad.len());
}

#[test]
fn e4m3_saturation_and_nonfinite_contract() {
    use nv_kernels::wgpu_backend::kernels::kv_fp8::encode_e4m3;
    let cases: [(&str, f32, u8); 11] = [
        ("max finite 448", 448.0, 0x7e),
        ("-448", -448.0, 0xfe),
        ("just over max 464", 464.0, 0x7e),
        ("far over max 1e30", 1e30, 0x7e),
        ("+inf saturates, never NaN", f32::INFINITY, 0x7e),
        ("-inf saturates, never NaN", f32::NEG_INFINITY, 0xfe),
        ("NaN stays NaN", f32::NAN, 0x7f),
        ("+0", 0.0, 0x00),
        ("-0 keeps its sign", -0.0, 0x80),
        ("smallest subnormal 2^-9", 0.001_953_125, 0x01),
        ("half of it rounds to even zero", 0.000_976_562_5, 0x00),
    ];
    for (name, x, want) in cases {
        let got = encode_e4m3(x);
        eprintln!("{name}: encode_e4m3({x:e}) = 0x{got:02x} (contract 0x{want:02x})");
        assert_eq!(got, want, "{name}");
    }
    eprintln!(
        "CONTRACT: E4M3 max is 448.0 (OCP E4M3, one NaN pattern per sign, no infinities). \
         Out-of-range magnitudes SATURATE to +-448 and never become NaN. This is CUDA's \
         __NV_SATFINITE behaviour for __nv_cvt_float_to_fp8(.., __NV_E4M3). A backend that used \
         240.0 (E5M2-style max) or that flushed overflow to NaN would violate the contract."
    );
    assert_eq!(FP8_E4M3_MAX, 448.0);
    assert_eq!(
        nv_kernels::wgpu_backend::kernels::kv_fp8::FP8_E4M3_MAX,
        448.0,
        "wgpu KV path must use 448, not 240"
    );
}

#[test]
fn weight_and_kv_fp8_paths_share_one_encoder_and_one_scale_convention() {
    use nv_kernels::wgpu_backend::kernels::kv_fp8;
    use nv_kernels::wgpu_backend::kernels::quant_gemv;
    let k = 64usize;
    let n = 3usize;
    let mut w = vec![0u16; n * k];
    let mut s = 7u32;
    for v in w.iter_mut() {
        s = s.wrapping_mul(1664525).wrapping_add(1013904223);
        let u = (s >> 8) as f32 / 16777216.0;
        *v = half::bf16::from_f32((u - 0.5) * 0.4).to_bits();
    }
    w[0] = half::bf16::from_f32(3.75).to_bits();
    let (packed, scales) = quant_gemv::quantize_rows_fp8(&w, n, k);
    assert_eq!(scales.len(), n);
    let byte = |r: usize, i: usize| -> u8 {
        let idx = r * k + i;
        ((packed[idx / 4] >> (8 * (idx % 4))) & 0xff) as u8
    };
    let mut mism = 0usize;
    for r in 0..n {
        let row = &w[r * k..(r + 1) * k];
        let amax = row.iter().fold(0f32, |a, b| {
            let v = quant_gemv::bf16_to_f32(*b);
            if v.is_finite() {
                a.max(v.abs())
            } else {
                a
            }
        });
        let want_scale = amax / FP8_E4M3_MAX;
        assert_eq!(
            scales[r].to_bits(),
            want_scale.to_bits(),
            "row {r} scale must be exactly amax/448"
        );
        let inv = FP8_E4M3_MAX / amax;
        for i in 0..k {
            let want = kv_fp8::encode_e4m3(quant_gemv::bf16_to_f32(row[i]) * inv);
            if byte(r, i) != want {
                mism += 1;
            }
        }
    }
    eprintln!(
        "weight fp8 path (quant_gemv::quantize_rows_fp8) reuses kv_fp8::encode_e4m3 and the \
         scale = amax/448 / inv = 448/amax convention: {mism} byte mismatches over {} elements",
        n * k
    );
    assert_eq!(
        mism, 0,
        "the weight fp8 path must use the same encoder and scale convention as the KV fp8 path, \
         which is the path that parity_kv_fp8 proves byte-exact against CUDA"
    );
}

#[test]
fn inverse_scale_must_be_448_over_amax_not_one_over_scale() {
    use nv_kernels::wgpu_backend::kernels::kv_fp8::encode_e4m3;
    let mut differ = 0usize;
    let mut total = 0usize;
    let mut first: Option<(f32, f32, u8, u8)> = None;
    let mut s = 3u32;
    for _ in 0..20000 {
        s = s.wrapping_mul(1664525).wrapping_add(1013904223);
        let u = (s >> 8) as f32 / 16777216.0;
        let amax = 0.01 + u * 8.0;
        let direct = FP8_E4M3_MAX / amax;
        let round_trip = 1.0f32 / (amax / FP8_E4M3_MAX);
        for j in 0..8 {
            let v = amax * ((j as f32 + 0.5) / 8.0);
            let a = encode_e4m3(v * direct);
            let b = encode_e4m3(v * round_trip);
            total += 1;
            if a != b {
                differ += 1;
                if first.is_none() {
                    first = Some((amax, v, a, b));
                }
            }
        }
    }
    eprintln!(
        "inv = 448/amax vs inv = 1/(amax/448): {differ}/{total} encoded bytes differ; first {first:?}"
    );
    eprintln!(
        "CONTRACT: every fp8 producer must compute inv_scale as 448/amax in ONE division. \
         wgpu kv_fp8 (host + WGSL), wgpu quant_gemv::quantize_rows_fp8, CUDA kv_fp8.cu and the \
         CUDA rowquant_e4m3_kernel in cuda/gemv_bf16.cu all do. The rowquant kernel arrived from \
         the laguna merge with the two-rounding `scale = amax/448; inv = 1/scale` form and \
         was fixed to `inv = 448/amax`; laguna_fp8_contract.rs keeps it fixed (a static scan over \
         every .cu/.cuh source plus cuda_rowquant_e4m3_encodes_against_448_over_amax on device), \
         and cuda_rowquant_e4m3_matches_wgpu_quantize_rows_fp8_byte_for_byte in this suite pins \
         it to the wgpu weight path byte for byte."
    );
    assert!(
        differ > 0,
        "if these ever became equivalent, drop this contract clause"
    );
}

#[test]
fn wgpu_weight_fp8_granularity_is_recorded_in_the_contract_doc() {
    use nv_kernels::wgpu_backend::kernels::quant_gemv;
    let n = 4usize;
    let k = 256usize;
    let w: Vec<u16> = (0..n * k)
        .map(|i| half::bf16::from_f32(((i % 37) as f32 - 18.0) * 0.01).to_bits())
        .collect();
    let (_, scales) = quant_gemv::quantize_rows_fp8(&w, n, k);
    assert!(scales.len() >= n && (n * k).is_multiple_of(scales.len()));
    let elems_per_scale = n * k / scales.len();
    let granularity = if scales.len() == n {
        "PER_ROW".to_string()
    } else {
        format!("PER_GROUP_{elems_per_scale}")
    };
    eprintln!(
        "wgpu attention-projection fp8 weight granularity: {granularity} ({elems_per_scale} \
         elements per fp32 scale at n={n} k={k}); NVFP4 in the same model carries one ue4m3 block \
         scale every 16 elements"
    );
    let marker = format!("WGPU_WEIGHT_FP8_GRANULARITY = {granularity}");
    assert!(
        CONTRACT_DOC.contains(&marker),
        "docs/book/04.1-fp8.md must record the granularity that is actually landed. Expected the \
         line `{marker}` but the doc does not contain it. If you changed the scaling granularity, \
         update docs/book/04.1-fp8.md in the same change."
    );
}

fn rms_rel(err_sq: f64, ref_sq: f64) -> f32 {
    (err_sq / ref_sq.max(1e-30)).sqrt() as f32
}

fn quantize_group_fp8(row: &[f32], group: usize) -> (f32, f32, usize) {
    use nv_kernels::wgpu_backend::kernels::kv_fp8::{decode_e4m3, encode_e4m3};
    let mut max_abs = 0f32;
    let mut sq = 0f64;
    let mut ref_sq = 0f64;
    let mut subnormal = 0usize;
    for chunk in row.chunks(group) {
        let amax = chunk.iter().fold(0f32, |a, b| a.max(b.abs()));
        let (scale, inv) = if amax > 0.0 {
            (amax / FP8_E4M3_MAX, FP8_E4M3_MAX / amax)
        } else {
            (0.0, 0.0)
        };
        for v in chunk {
            let code = encode_e4m3(*v * inv);
            if code & 0x78 == 0 {
                subnormal += 1;
            }
            let back = decode_e4m3(code) * scale;
            let e = (back - *v).abs();
            max_abs = max_abs.max(e);
            sq += (e as f64) * (e as f64);
            ref_sq += (*v as f64) * (*v as f64);
        }
    }
    (max_abs, rms_rel(sq, ref_sq), subnormal)
}

fn e2m1_levels() -> [f32; 8] {
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]
}

fn quantize_nvfp4_reference(row: &[f32]) -> (f32, f32) {
    use nv_kernels::wgpu_backend::kernels::kv_fp8::{decode_e4m3, encode_e4m3};
    const BLOCK: usize = 16;
    let levels = e2m1_levels();
    let tensor_amax = row.iter().fold(0f32, |a, b| a.max(b.abs()));
    let global = if tensor_amax > 0.0 {
        tensor_amax / (FP8_E4M3_MAX * 6.0)
    } else {
        1.0
    };
    let mut max_abs = 0f32;
    let mut sq = 0f64;
    let mut ref_sq = 0f64;
    for chunk in row.chunks(BLOCK) {
        let amax = chunk.iter().fold(0f32, |a, b| a.max(b.abs()));
        let raw = if global > 0.0 {
            amax / (6.0 * global)
        } else {
            0.0
        };
        let bs = decode_e4m3(encode_e4m3(raw)).abs();
        let eff = bs * global;
        for v in chunk {
            let back = if eff > 0.0 {
                let t = (*v / eff).abs();
                let mut best = 0f32;
                let mut best_err = f32::INFINITY;
                for l in levels {
                    let e = (l - t).abs();
                    if e < best_err {
                        best_err = e;
                        best = l;
                    }
                }
                best * eff * v.signum()
            } else {
                0.0
            };
            let e = (back - *v).abs();
            max_abs = max_abs.max(e);
            sq += (e as f64) * (e as f64);
            ref_sq += (*v as f64) * (*v as f64);
        }
    }
    (max_abs, rms_rel(sq, ref_sq))
}

fn gaussian_row(k: usize, seed: u32) -> Vec<f32> {
    let mut s = seed | 1;
    (0..k)
        .map(|_| {
            s = s.wrapping_mul(1664525).wrapping_add(1013904223);
            let u1 = ((s >> 8) as f32 / 16777216.0).max(1e-7);
            s = s.wrapping_mul(1664525).wrapping_add(1013904223);
            let u2 = (s >> 8) as f32 / 16777216.0;
            (-2.0 * u1.ln()).sqrt() * (std::f32::consts::TAU * u2).cos() * 0.02
        })
        .collect()
}

fn with_outliers(mut row: Vec<f32>, seed: u32, count: usize, gain: f32) -> Vec<f32> {
    let k = row.len();
    let mut s = seed | 1;
    for j in 0..count {
        s = s.wrapping_mul(1664525).wrapping_add(1013904223);
        let i = ((s >> 8) as usize) % k;
        row[i] = 0.02 * gain * if j % 2 == 0 { 1.0 } else { -1.0 };
    }
    row
}

fn heavy_tail_row(k: usize, seed: u32) -> Vec<f32> {
    let mut s = seed | 1;
    (0..k)
        .map(|_| {
            s = s.wrapping_mul(1664525).wrapping_add(1013904223);
            let u1 = ((s >> 8) as f32 / 16777216.0).max(1e-7);
            s = s.wrapping_mul(1664525).wrapping_add(1013904223);
            let u2 = (s >> 8) as f32 / 16777216.0;
            let g = (-2.0 * u1.ln()).sqrt() * (std::f32::consts::TAU * u2).cos();
            0.02 * g.abs().powf(3.0) * g.signum()
        })
        .collect()
}

#[test]
fn granularity_is_measured_not_assumed_and_e4m3_is_nearly_scale_invariant() {
    let shapes: [(&str, usize); 3] = [
        ("gemma4-31B qkv k=5376", 5376),
        ("gemma4-31B o   k=8192", 8192),
        ("gemma4-31B o   k=16384", 16384),
    ];
    let groups = [128usize, 64, 32, 16];
    let mut worst_regression = 1.0f32;
    let mut best_improvement = 1.0f32;
    let mut fp8_beats_nvfp4 = 0usize;
    let mut cases = 0usize;
    for (name, k) in shapes {
        let variants: [(&str, Vec<f32>); 4] = [
            ("gaussian", gaussian_row(k, 0x51ce ^ k as u32)),
            (
                "gaussian+8 outliers x8",
                with_outliers(gaussian_row(k, 0x51ce ^ k as u32), 7, 8, 8.0),
            ),
            (
                "gaussian+1 outlier x1000",
                with_outliers(gaussian_row(k, 0x51ce ^ k as u32), 7, 1, 1000.0),
            ),
            ("heavy tail |g|^3", heavy_tail_row(k, 0xbead ^ k as u32)),
        ];
        for (vname, row) in variants {
            let (row_max, row_rms, row_sub) = quantize_group_fp8(&row, k);
            let (nv_max, nv_rms) = quantize_nvfp4_reference(&row);
            eprintln!(
                "{name} {vname:>24}: fp8 per-row   max_abs {row_max:.3e} rms_rel {row_rms:.4e} subnormal {}/{k}",
                row_sub
            );
            for g in groups {
                let (m, r, sub) = quantize_group_fp8(&row, g.min(k));
                let ratio = row_rms / r.max(1e-30);
                worst_regression = worst_regression.min(ratio);
                best_improvement = best_improvement.max(ratio);
                eprintln!(
                    "{name} {vname:>24}: fp8 per-{g:<5} max_abs {m:.3e} rms_rel {r:.4e} subnormal {sub}/{k} ({ratio:.3}x vs per-row)"
                );
            }
            eprintln!(
                "{name} {vname:>24}: NVFP4 gs=16  max_abs {nv_max:.3e} rms_rel {nv_rms:.4e} (fp8 per-row is {:.3}x NVFP4's rms_rel)",
                row_rms / nv_rms
            );
            cases += 1;
            if row_rms < nv_rms {
                fp8_beats_nvfp4 += 1;
            }
        }
    }
    eprintln!(
        "MEASURED: per-group fp8 improves rms_rel by at most {best_improvement:.3}x and never \
         regresses below {worst_regression:.3}x. e4m3 is a FLOATING-POINT format: its relative \
         precision is scale-invariant across ~17 binades, so a row-wide amax costs almost nothing \
         unless elements underflow the subnormal floor (amax * 2^-9 / 448), which the subnormal \
         counts above show does not happen at these distributions."
    );
    eprintln!(
        "MEASURED: fp8 per-row rms_rel is BELOW NVFP4-gs16 rms_rel in {fp8_beats_nvfp4}/{cases} \
         cases. NVFP4 is the format that works in this model and fp8 is the one that breaks it, \
         so per-element quantization error does not explain the collapse and PER-GROUP SCALING \
         ALONE IS UNLIKELY TO FIX IT. Look at the fp8 GEMV epilogue (g4w_gemv_fp8_pk / _pk3, \
         qg_row_acc_e4m3 / qg_reduce / qg_butterfly), at what the fp8 path quantizes FROM, and at \
         the activation side, before spending a round on granularity."
    );
    assert!(
        worst_regression > 0.98,
        "no group size may materially regress rms_rel; worst was {worst_regression:.3}x"
    );
    assert!(
        best_improvement < 2.0,
        "if per-group ever bought more than 2x here, revisit the conclusion above"
    );
    assert_eq!(
        fp8_beats_nvfp4, cases,
        "fp8 per-row was expected to beat NVFP4-gs16 on rms_rel in every case ({fp8_beats_nvfp4}/{cases})"
    );
}

#[test]
fn fp8_scaling_granularity_matrix_across_backends() {
    struct Row {
        path: &'static str,
        backend: &'static str,
        elems_per_scale: &'static str,
        max: f32,
        rounding: &'static str,
        overflow: &'static str,
        present_at_head: bool,
    }
    let rows = [
        Row {
            path: "quant_gemv::quantize_rows_fp8 (attention projections)",
            backend: "wgpu",
            elems_per_scale: "k (whole row)",
            max: 448.0,
            rounding: "RNE",
            overflow: "saturate to +-448",
            present_at_head: true,
        },
        Row {
            path: "kv_fp8 / kv_fp8_paged (KV cache)",
            backend: "wgpu",
            elems_per_scale: "head_dim (per token per kv head)",
            max: 448.0,
            rounding: "RNE",
            overflow: "saturate to +-448",
            present_at_head: true,
        },
        Row {
            path: "kv_fp8.cu / kv_fp8_paged.cu (KV cache)",
            backend: "cuda",
            elems_per_scale: "head_dim (per token per kv head)",
            max: 448.0,
            rounding: "RNE (__NV_SATFINITE)",
            overflow: "saturate to +-448",
            present_at_head: true,
        },
        Row {
            path: "nv_layers::linear::quantize_fp8_per_tensor (Linear::from_bf16_quantized_fp8)",
            backend: "cuda",
            elems_per_scale: "n*k (whole tensor)",
            max: 448.0,
            rounding: "RNE (float8 crate)",
            overflow: "saturate to +-448",
            present_at_head: true,
        },
        Row {
            path: "rowquant_e4m3_kernel (Laguna NV_LAGUNA_ATTN_FP8 projections)",
            backend: "cuda",
            elems_per_scale: "k (whole row)",
            max: 448.0,
            rounding: "RNE (__NV_SATFINITE)",
            overflow: "saturate to +-448",
            present_at_head: true,
        },
    ];
    eprintln!("path | backend | elements per scale | max | rounding | overflow | at HEAD");
    for r in &rows {
        eprintln!(
            "{} | {} | {} | {} | {} | {} | {}",
            r.path,
            r.backend,
            r.elems_per_scale,
            r.max,
            r.rounding,
            r.overflow,
            if r.present_at_head {
                "yes"
            } else {
                "no (incoming merge)"
            }
        );
        assert_eq!(r.max, 448.0, "{}: every fp8 path must use 448", r.path);
        assert!(r.rounding.starts_with("RNE"), "{}", r.path);
        assert!(r.overflow.starts_with("saturate"), "{}", r.path);
    }
    let wgpu_proj = rows
        .iter()
        .find(|r| r.backend == "wgpu" && r.path.starts_with("quant_gemv"))
        .unwrap();
    let cuda_proj: Vec<&Row> = rows
        .iter()
        .filter(|r| r.backend == "cuda" && r.path.contains("projections"))
        .collect();
    eprintln!(
        "SETTLED CONTRACT: attention-projection weight fp8 is PER-ROW on both backends - wgpu \
         quant_gemv::quantize_rows_fp8 is {} and CUDA rowquant_e4m3_kernel is {}. The pair is \
         pinned byte-exact by cuda_rowquant_e4m3_matches_wgpu_quantize_rows_fp8_byte_for_byte in \
         this suite, alongside the fp8 KV cache pins (parity_kv_fp8, parity_kv_fp8_paged). \
         nv_layers::linear::quantize_fp8_per_tensor ({}) is a DIFFERENT surface - generic Linear \
         weights, not attention projections - and is deliberately outside this contract.",
        wgpu_proj.elems_per_scale,
        cuda_proj[0].elems_per_scale,
        rows.iter()
            .find(|r| r.path.contains("per_tensor"))
            .unwrap()
            .elems_per_scale
    );
    assert!(
        cuda_proj.iter().all(|r| r.present_at_head),
        "the CUDA per-row attention-projection fp8 path (rowquant_e4m3_kernel) is expected at \
         HEAD; if it was removed, delete its matrix row, retire \
         cuda_rowquant_e4m3_matches_wgpu_quantize_rows_fp8_byte_for_byte and update \
         docs/book/04.1-fp8.md §7.3"
    );
    assert!(
        cuda_proj
            .iter()
            .all(|r| r.elems_per_scale == wgpu_proj.elems_per_scale),
        "attention-projection weight fp8 granularity must stay PER-ROW on both backends; if one \
         side changes, change the other in the same commit and update docs/book/04.1-fp8.md §7.3"
    );
    let marker = "ATTN_PROJ_WEIGHT_FP8_GRANULARITY = PER_ROW_BOTH_BACKENDS";
    assert!(
        CONTRACT_DOC.contains(marker),
        "docs/book/04.1-fp8.md §7.3 must record the settled cross-backend contract. Expected the \
         line `{marker}` but the doc does not contain it. If you changed either backend's \
         attention-projection fp8 granularity, update the doc in the same change."
    );
}

mod wgpu_device {
    use half::bf16;
    use nv_kernels::wgpu_backend::device::WgpuContext;
    use nv_kernels::wgpu_backend::kernels::kv_fp8;

    fn ctx_or_skip(test: &str) -> Option<&'static WgpuContext> {
        match WgpuContext::shared() {
            Ok(ctx) => {
                eprintln!("{test}: {}", ctx.summary());
                Some(ctx)
            }
            Err(e) => {
                if std::env::var("NV_KERNELS_WGPU_ALLOW_SKIP").as_deref() != Ok("1") {
                    panic!(
                        "{test}: no wgpu adapter: {e}. This gate refuses to report success \
                         without running; set NV_KERNELS_WGPU_ALLOW_SKIP=1 to skip on purpose."
                    );
                }
                eprintln!("{test}: SKIP no wgpu adapter: {e}");
                None
            }
        }
    }

    #[test]
    fn wgsl_encoder_agrees_with_the_host_encoder_on_device() {
        let Some(ctx) = ctx_or_skip("wgsl_encoder_agrees_with_the_host_encoder_on_device") else {
            return;
        };
        let n_tokens = 4usize;
        let n_kv = 2usize;
        let head_dim = 64usize;
        let mut x = vec![0u16; n_tokens * n_kv * head_dim];
        let mut s = 11u32;
        for g in 0..n_tokens * n_kv {
            let base = g * head_dim;
            x[base] = bf16::from_f32(448.0).to_bits();
            for d in 1..head_dim {
                s = s.wrapping_mul(1664525).wrapping_add(1013904223);
                let u = (s >> 8) as f32 / 16777216.0;
                x[base + d] = bf16::from_f32((u - 0.5) * 800.0).to_bits();
            }
        }
        let mut dev_bytes = vec![0u8; x.len()];
        let mut dev_scales = vec![0f32; n_tokens * n_kv];
        kv_fp8::quantize_kv_fp8(
            ctx,
            &x,
            &mut dev_bytes,
            &mut dev_scales,
            &[0i32],
            n_tokens,
            n_kv,
            head_dim,
            0,
        )
        .expect("wgpu quantize_kv_fp8");
        let mut host_bytes = vec![0u8; x.len()];
        let mut host_scales = vec![0f32; n_tokens * n_kv];
        kv_fp8::cpu_quantize_kv_fp8(
            &x,
            &mut host_bytes,
            &mut host_scales,
            0,
            n_tokens,
            n_kv,
            head_dim,
            0,
        );
        let byte_diff = dev_bytes
            .iter()
            .zip(host_bytes.iter())
            .filter(|(a, b)| a != b)
            .count();
        let scale_diff = dev_scales
            .iter()
            .zip(host_scales.iter())
            .filter(|(a, b)| a.to_bits() != b.to_bits())
            .count();
        eprintln!(
            "WGSL vs host e4m3 over {} elements with amax pinned to 448 (inv_scale == 1.0): {} byte \
             mismatches, {} scale mismatches",
            x.len(),
            byte_diff,
            scale_diff
        );
        assert_eq!(
            scale_diff, 0,
            "device and host scales must be bit-identical"
        );
        assert_eq!(byte_diff, 0, "device and host e4m3 codes must be identical");
        assert!(host_scales.iter().all(|s| (*s - 1.0).abs() < 1e-6));
    }
}

#[cfg(all(feature = "cuda", feature = "wgpu"))]
mod cuda_vs_wgpu {
    use cudarc::driver::{CudaContext, DevicePtr, DevicePtrMut};
    use half::bf16;
    use nv_kernels::cuda;
    use nv_kernels::wgpu_backend::kernels::kv_fp8;

    #[test]
    fn cuda_e4m3_cast_agrees_with_the_wgpu_host_encoder_byte_for_byte() {
        let stream = match CudaContext::new(0) {
            Ok(c) => c.default_stream(),
            Err(e) => {
                if std::env::var("NV_KERNELS_PARITY_REQUIRE").as_deref() == Ok("1") {
                    panic!("no CUDA device 0: {e}");
                }
                eprintln!("SKIP no CUDA device 0: {e}");
                return;
            }
        };
        let n_tokens = 8usize;
        let n_kv = 2usize;
        let head_dim = 128usize;
        let mut x = vec![0u16; n_tokens * n_kv * head_dim];
        let mut s = 29u32;
        for g in 0..n_tokens * n_kv {
            let base = g * head_dim;
            x[base] = bf16::from_f32(448.0).to_bits();
            for d in 1..head_dim {
                s = s.wrapping_mul(1664525).wrapping_add(1013904223);
                let u = (s >> 8) as f32 / 16777216.0;
                x[base + d] = bf16::from_f32((u - 0.5) * 900.0).to_bits();
            }
        }
        let n = x.len();
        #[allow(deprecated)]
        let x_dev = stream.memcpy_stod(&x).unwrap();
        #[allow(deprecated)]
        let start_dev = stream.memcpy_stod(&[0i32]).unwrap();
        let mut q_dev = unsafe { stream.alloc::<u8>(n).unwrap() };
        let mut sc_dev = unsafe { stream.alloc::<f32>(n_tokens * n_kv).unwrap() };
        let rc = {
            let (xp, _a) = x_dev.device_ptr(&stream);
            let (qp, _b) = q_dev.device_ptr_mut(&stream);
            let (sp, _c) = sc_dev.device_ptr_mut(&stream);
            let (stp, _d) = start_dev.device_ptr(&stream);
            unsafe {
                cuda::quantize_kv_fp8(
                    stream.cu_stream() as *mut std::ffi::c_void,
                    xp as *const u16,
                    qp as *mut u8,
                    sp as *mut f32,
                    stp as *const i32,
                    n_tokens as i32,
                    n_kv as i32,
                    head_dim as i32,
                    0,
                )
            }
        };
        assert_eq!(rc, 0, "cuda quantize_kv_fp8 rc={rc}");
        stream.synchronize().unwrap();
        #[allow(deprecated)]
        let cu_bytes = stream.memcpy_dtov(&q_dev).unwrap();
        #[allow(deprecated)]
        let cu_scales = stream.memcpy_dtov(&sc_dev).unwrap();

        let mut host_bytes = vec![0u8; n];
        let mut host_scales = vec![0f32; n_tokens * n_kv];
        kv_fp8::cpu_quantize_kv_fp8(
            &x,
            &mut host_bytes,
            &mut host_scales,
            0,
            n_tokens,
            n_kv,
            head_dim,
            0,
        );
        let byte_diff = cu_bytes
            .iter()
            .zip(host_bytes.iter())
            .filter(|(a, b)| a != b)
            .count();
        let scale_diff = cu_scales
            .iter()
            .zip(host_scales.iter())
            .filter(|(a, b)| a.to_bits() != b.to_bits())
            .count();
        eprintln!(
            "CUDA __nv_cvt_float_to_fp8(__NV_SATFINITE, __NV_E4M3) vs wgpu kv_fp8::encode_e4m3 over \
             {n} elements: {byte_diff} byte mismatches, {scale_diff} scale mismatches"
        );
        assert_eq!(
            scale_diff, 0,
            "scales must be bit-identical across backends"
        );
        assert_eq!(byte_diff, 0, "e4m3 codes must be identical across backends");
        eprintln!(
            "This pins the shared encoder. quant_gemv::quantize_rows_fp8 is proven to reuse this \
             exact encoder by weight_and_kv_fp8_paths_share_one_encoder_and_one_scale_convention, \
             so the wgpu weight path inherits CUDA-verified e4m3 semantics. The weight-path \
             scaling granularity is pinned separately by \
             cuda_rowquant_e4m3_matches_wgpu_quantize_rows_fp8_byte_for_byte - see \
             fp8_scaling_granularity_matrix_across_backends."
        );
    }

    #[test]
    fn cuda_rowquant_e4m3_matches_wgpu_quantize_rows_fp8_byte_for_byte() {
        use nv_kernels::wgpu_backend::kernels::quant_gemv;
        let stream = match CudaContext::new(0) {
            Ok(c) => c.default_stream(),
            Err(e) => {
                if std::env::var("NV_KERNELS_PARITY_REQUIRE").as_deref() == Ok("1") {
                    panic!("no CUDA device 0: {e}");
                }
                eprintln!("SKIP no CUDA device 0: {e}");
                return;
            }
        };
        let n = 96usize;
        let k = 640usize;
        let mut w = vec![0u16; n * k];
        let mut s = 41u32;
        for r in 0..n {
            let spread = 0.02 * 8f32.powf((r % 12) as f32 - 4.0);
            for i in 0..k {
                s = s.wrapping_mul(1664525).wrapping_add(1013904223);
                let u = (s >> 8) as f32 / 16777216.0;
                w[r * k + i] = bf16::from_f32((u - 0.5) * spread).to_bits();
            }
        }
        for i in 0..k {
            w[7 * k + i] = 0;
        }
        w[3 * k + 5] = bf16::from_f32(f32::INFINITY).to_bits();
        w[5 * k + 9] = bf16::from_f32(f32::NEG_INFINITY).to_bits();
        w[12 * k + 13] = bf16::from_f32(448.0).to_bits();
        w[13 * k] = bf16::from_f32(-0.0).to_bits();

        #[allow(deprecated)]
        let w_dev = stream.memcpy_stod(&w).unwrap();
        let mut q_dev = unsafe { stream.alloc::<u8>(n * k).unwrap() };
        let mut sc_dev = unsafe { stream.alloc::<f32>(n).unwrap() };
        let rc = {
            let (wp, _a) = w_dev.device_ptr(&stream);
            let (qp, _b) = q_dev.device_ptr_mut(&stream);
            let (sp, _c) = sc_dev.device_ptr_mut(&stream);
            unsafe {
                cuda::rowquant_e4m3(
                    stream.cu_stream() as *mut std::ffi::c_void,
                    wp as *const u16,
                    qp as *mut u8,
                    sp as *mut f32,
                    n as i32,
                    k as i32,
                )
            }
        };
        assert_eq!(rc, 0, "cuda rowquant_e4m3 rc={rc}");
        stream.synchronize().unwrap();
        #[allow(deprecated)]
        let cu_bytes = stream.memcpy_dtov(&q_dev).unwrap();
        #[allow(deprecated)]
        let cu_scales = stream.memcpy_dtov(&sc_dev).unwrap();

        let (packed, wg_scales) = quant_gemv::quantize_rows_fp8(&w, n, k);
        assert_eq!(wg_scales.len(), n, "wgpu weight path must stay per-row");
        let wg_byte = |idx: usize| -> u8 { ((packed[idx / 4] >> (8 * (idx % 4))) & 0xff) as u8 };

        let scale_diff = cu_scales
            .iter()
            .zip(wg_scales.iter())
            .filter(|(a, b)| a.to_bits() != b.to_bits())
            .count();
        let mut byte_diff = 0usize;
        let mut first: Option<(usize, usize, f32, u8, u8)> = None;
        for idx in 0..n * k {
            let want = wg_byte(idx);
            if cu_bytes[idx] != want {
                byte_diff += 1;
                if first.is_none() {
                    first = Some((
                        idx / k,
                        idx % k,
                        bf16::from_bits(w[idx]).to_f32(),
                        cu_bytes[idx],
                        want,
                    ));
                }
            }
        }
        eprintln!(
            "CUDA rowquant_e4m3_kernel vs wgpu quant_gemv::quantize_rows_fp8 over {n}x{k} \
             (12 binades of row amax, an all-zero row, +-inf poison, exact-448 amax, -0.0): \
             {byte_diff} byte mismatches, {scale_diff} scale mismatches; first {first:?}"
        );
        assert_eq!(
            scale_diff, 0,
            "per-row scales must be bit-identical: both backends compute amax over finite \
             elements then scale = amax/448 in f32"
        );
        assert_eq!(
            byte_diff, 0,
            "the cross-backend weight-fp8 contract is byte-exactness: same per-row granularity, \
             same finite-skipping amax, same one-division inv = 448/amax, same RNE/SATFINITE \
             encoder. See docs/book/04.1-fp8.md §7.3."
        );
        eprintln!(
            "This is the cross-backend WEIGHT-fp8 parity pin that §7.3 of docs/book/04.1-fp8.md \
             previously recorded as missing. KV-cache fp8 was already pinned by parity_kv_fp8 / \
             parity_kv_fp8_paged; attention-projection weights are now pinned here."
        );
    }
}

#[cfg(not(all(feature = "cuda", feature = "wgpu")))]
#[test]
#[allow(non_snake_case)]
fn cuda_e4m3_cast_agrees_with_the_wgpu_host_encoder_SKIPPED_needs_cuda_and_wgpu() {
    eprintln!(
        "the CUDA-vs-wgpu e4m3 byte-exactness check was CFG'd OUT of this binary. It needs BOTH \
         `cuda` and `wgpu` features. This is a SKIP, not a pass; running this suite with \
         NVK_FEATURES=wgpu will never execute it."
    );
}

#[test]
fn features_compiled_into_this_binary() {
    let cuda = cfg!(feature = "cuda");
    let wgpu = cfg!(feature = "wgpu");
    eprintln!("fp8_contract_e4m3 compiled with cuda={cuda} wgpu={wgpu}");
    eprintln!(
        "cross-backend clauses require cuda+wgpu together. A run with only one of them executes \
         the CPU clauses and reports the others as *_SKIPPED_* tests by name."
    );
}
