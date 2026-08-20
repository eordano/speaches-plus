use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

const CUDA_DIR: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/cuda");

const GEMV_CU: &str = include_str!("../cuda/gemv_nvfp4.cu");
const QUANT_CU: &str = include_str!("../cuda/quantize_nvfp4_bf16.cu");
const RMSQ_CU: &str = include_str!("../cuda/rmsnorm_quantize_nvfp4_bf16.cu");
const MOE_CU: &str = include_str!("../cuda/moe_grouped_fp4_gemv.cu");

const CUDA_SOURCE_FLOOR: usize = 40;

const UE4M3_MIN_NORMAL: f32 = 0.015625;
const UE4M3_SUBNORMAL_STEP: f32 = 0.001953125;
const UE4M3_MAX: f32 = 448.0;
const UE4M3_NAN_CODE: u8 = 0x7f;
const UE4M3_MAX_CODE: u8 = 0x7e;

const PREFIX_CODES_OVER_THE_SUBNORMAL_BAND: [u8; 7] = [0x00, 0x00, 0x04, 0x00, 0x02, 0x04, 0x06];
const PREFIX_WORST_DECODE_RATIO: f64 = 4.5;
const PREFIX_LOST_BLOCKS_OVER_THE_SUBNORMAL_BAND: usize = 3;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum CodecForm {
    Encode,
    Decode,
    DecodeBranchlessBitcast,
}

const CUDA_UE4M3_CODEC_CENSUS: [(&str, &str, CodecForm); 11] = [
    (
        "gemm_nvfp4_w4a16_mk_mma.cu",
        "decode_ue4m3_scale",
        CodecForm::Decode,
    ),
    (
        "gemv_nvfp4.cu",
        "decode_ue4m3_branchless_dev",
        CodecForm::DecodeBranchlessBitcast,
    ),
    ("gemv_nvfp4.cu", "decode_ue4m3_dev", CodecForm::Decode),
    ("gemv_nvfp4.cu", "encode_ue4m3_dev", CodecForm::Encode),
    (
        "gemv_nvfp4_w4a16_decode.cu",
        "decode_ue4m3_scale",
        CodecForm::Decode,
    ),
    (
        "gemv_nvfp4_w4a8_decode.cu",
        "decode_ue4m3_scale",
        CodecForm::Decode,
    ),
    (
        "moe_grouped_fp4_gemv.cu",
        "decode_ue4m3_dev",
        CodecForm::Decode,
    ),
    (
        "quantize_nvfp4_bf16.cu",
        "decode_ue4m3_dev",
        CodecForm::Decode,
    ),
    (
        "quantize_nvfp4_bf16.cu",
        "encode_ue4m3_dev",
        CodecForm::Encode,
    ),
    (
        "rmsnorm_quantize_nvfp4_bf16.cu",
        "rq_decode_ue4m3",
        CodecForm::Decode,
    ),
    (
        "rmsnorm_quantize_nvfp4_bf16.cu",
        "rq_encode_ue4m3",
        CodecForm::Encode,
    ),
];

const CUDA_ENCODE_BODY_TRANSCRIBED_BY_CU_ENCODE_UE4M3: &str = "{ if (!isfinite(scale) || scale <= \
     0.f) return 0; float clamped = fminf(scale, 448.f); if (clamped < NV_UE4M3_MIN_NORMAL) { int \
     sub = (int)roundf(clamped / NV_UE4M3_SUBNORMAL_STEP); if (sub <= 0) return 0; if (sub <= 7) \
     return (uint8_t)sub; return 0x08; } int e2; frexpf(clamped, &e2); int exp_v = e2 - 1; float \
     mant_f = ldexpf(clamped, -exp_v) - 1.f; int mant = (int)roundf(mant_f * 8.f); if (mant < 0) \
     mant = 0; if (mant > 7) { mant = 0; exp_v += 1; } int biased = exp_v + 7; if (biased < 1) \
     biased = 1; if (biased > 15) biased = 15; uint8_t byte = ((uint8_t)biased << 3) | \
     (uint8_t)(mant & 0x07); return (byte == 0x7F) ? 0x7E : byte; }";

const CUDA_DECODE_BODY_TRANSCRIBED_BY_CU_DECODE_UE4M3: &str = "{ int biased = (int)(b >> 3) & \
     0x0F; float mant = (float)(b & 0x07); if (biased == 0) return mant * \
     NV_UE4M3_SUBNORMAL_STEP; return (1.f + mant / 8.f) * exp2f((float)(biased - 7)); }";

const CUDA_DECODE_BRANCHLESS_BODY_TRANSCRIBED_BY_CU_DECODE_UE4M3_BRANCHLESS: &str = "{ unsigned \
     biased = (unsigned)(b >> 3) & 0x0Fu; unsigned mant = (unsigned)(b & 0x07u); float norm = \
     __uint_as_float(((biased + 120u) << 23) | (mant << 20)); return biased ? norm : (float)mant \
     * NV_UE4M3_SUBNORMAL_STEP; }";

const PRE_FIX_DECODE_FORM: &str = "((int)(b >> 3) & 0x0F) - 7";
const PRE_FIX_MANTISSA_CLAMP: &str = "if (mant > 7) mant = 7;";

fn exp2i(k: i32) -> f32 {
    assert!(
        (-126..=127).contains(&k),
        "exp2i({k}) leaves the f32 normal range; no ue4m3 path may reach it"
    );
    f32::from_bits(((k + 127) as u32) << 23)
}

fn ldexp_f32(x: f32, k: i32) -> f32 {
    let mut y = x;
    let mut r = k;
    while r > 100 {
        y *= exp2i(100);
        r -= 100;
    }
    while r < -100 {
        y *= exp2i(-100);
        r += 100;
    }
    y * exp2i(r)
}

fn frexp_exp(x: f32) -> i32 {
    assert!(
        x.is_finite() && x > 0.0,
        "frexp_exp({x:e}): the CUDA guard `!isfinite(scale) || scale <= 0` should have \
         returned before this point"
    );
    let e = ((x.to_bits() >> 23) & 0xff) as i32;
    if e != 0 {
        return e - 126;
    }
    let y = x * exp2i(64);
    (((y.to_bits() >> 23) & 0xff) as i32 - 126) - 64
}

fn cu_encode_ue4m3(scale: f32) -> u8 {
    if !scale.is_finite() || scale <= 0.0 {
        return 0;
    }
    let clamped = scale.min(UE4M3_MAX);
    if clamped < UE4M3_MIN_NORMAL {
        let sub = (clamped / UE4M3_SUBNORMAL_STEP).round() as i32;
        if sub <= 0 {
            return 0;
        }
        if sub <= 7 {
            return sub as u8;
        }
        return 0x08;
    }
    let e2 = frexp_exp(clamped);
    let mut exp_v = e2 - 1;
    let mant_f = ldexp_f32(clamped, -exp_v) - 1.0;
    let mut mant = (mant_f * 8.0).round() as i32;
    if mant < 0 {
        mant = 0;
    }
    if mant > 7 {
        mant = 0;
        exp_v += 1;
    }
    let mut biased = exp_v + 7;
    if biased < 1 {
        biased = 1;
    }
    if biased > 15 {
        biased = 15;
    }
    let byte = ((biased as u8) << 3) | (mant as u8 & 0x07);
    if byte == UE4M3_NAN_CODE {
        UE4M3_MAX_CODE
    } else {
        byte
    }
}

fn cu_decode_ue4m3(b: u8) -> f32 {
    let biased = (b >> 3) as i32 & 0x0f;
    let mant = (b & 0x07) as f32;
    if biased == 0 {
        return mant * UE4M3_SUBNORMAL_STEP;
    }
    (1.0 + mant / 8.0) * exp2i(biased - 7)
}

fn cu_decode_ue4m3_branchless(b: u8) -> f32 {
    let biased = (b >> 3) as u32 & 0x0f;
    let mant = (b & 0x07) as u32;
    let norm = f32::from_bits(((biased + 120) << 23) | (mant << 20));
    if biased == 0 {
        mant as f32 * UE4M3_SUBNORMAL_STEP
    } else {
        norm
    }
}

fn prefix_encode_ue4m3(scale: f32) -> u8 {
    if !scale.is_finite() || scale <= 0.0 {
        return 0;
    }
    let clamped = scale.min(UE4M3_MAX);
    let e2 = frexp_exp(clamped);
    let mut exp_v = e2 - 1;
    let mant_f = ldexp_f32(clamped, -exp_v) - 1.0;
    let mut mant = (mant_f * 8.0).round() as i32;
    if mant < 0 {
        mant = 0;
    }
    if mant > 7 {
        mant = 0;
        exp_v += 1;
    }
    let mut biased = exp_v + 7;
    if biased < 0 {
        biased = 0;
    }
    if biased > 15 {
        biased = 15;
    }
    let byte = ((biased as u8) << 3) | (mant as u8 & 0x07);
    if byte == UE4M3_NAN_CODE {
        UE4M3_MAX_CODE
    } else {
        byte
    }
}

fn rq_prefix_encode_ue4m3(scale: f32) -> u8 {
    if !scale.is_finite() || scale <= 0.0 {
        return 0;
    }
    let clamped = scale.min(UE4M3_MAX);
    let e2 = frexp_exp(clamped);
    let exp_v = e2 - 1;
    let mant_f = ldexp_f32(clamped, -exp_v) - 1.0;
    let mut mant = (mant_f * 8.0).round() as i32;
    if mant < 0 {
        mant = 0;
    }
    if mant > 7 {
        mant = 7;
    }
    let mut biased = exp_v + 7;
    if biased < 0 {
        biased = 0;
    }
    if biased > 15 {
        biased = 15;
    }
    let byte = ((biased as u8) << 3) | (mant as u8 & 0x07);
    if byte == UE4M3_NAN_CODE {
        UE4M3_MAX_CODE
    } else {
        byte
    }
}

fn prefix_decode_ue4m3(b: u8) -> f32 {
    let exp_v = ((b >> 3) as i32 & 0x0f) - 7;
    let mant = (b & 0x07) as f32;
    (1.0 + mant / 8.0) * exp2i(exp_v)
}

fn e4m3fn_signed_value(byte: u8) -> Option<f64> {
    let s = byte >> 7;
    let e = ((byte >> 3) & 0x0f) as i32;
    let m = (byte & 0x07) as f64;
    if e == 0x0f && (byte & 0x07) == 0x07 {
        return None;
    }
    let mag = if e == 0 {
        m * (-9f64).exp2()
    } else {
        (1.0 + m / 8.0) * ((e - 7) as f64).exp2()
    };
    Some(if s == 1 { -mag } else { mag })
}

fn e4m3fn_value(byte: u8) -> Option<f64> {
    assert!(byte < 0x80, "e4m3fn_value is the non-negative half only");
    e4m3fn_signed_value(byte)
}

fn e4m3fn_nearest_code(target: f64) -> u8 {
    let mut best = 0u8;
    let mut best_d = f64::INFINITY;
    for b in 0u16..0x80 {
        let Some(v) = e4m3fn_value(b as u8) else {
            continue;
        };
        let d = (v - target).abs();
        if d <= best_d {
            best_d = d;
            best = b as u8;
        }
    }
    best
}

fn step_bits(x: f32, steps: i32) -> f32 {
    f32::from_bits((x.to_bits() as i64 + steps as i64) as u32)
}

fn subnormal_band_targets() -> Vec<f32> {
    (1..=7).map(|m| m as f32 * exp2i(-9)).collect()
}

fn encode_probes() -> Vec<f32> {
    let mut p: Vec<f32> = Vec::new();
    for b in 0u8..=UE4M3_MAX_CODE {
        let v = cu_decode_ue4m3(b);
        if v > 0.0 {
            for j in -3..=3i32 {
                p.push(step_bits(v, j));
            }
        }
    }
    for b in 0u8..UE4M3_MAX_CODE {
        let lo = cu_decode_ue4m3(b) as f64;
        let hi = cu_decode_ue4m3(b + 1) as f64;
        let mid = 0.5 * (lo + hi);
        for j in -2..=2i32 {
            let m = mid as f32;
            if m > 0.0 {
                p.push(step_bits(m, j));
            }
        }
    }
    for i in 0..=20000u32 {
        p.push(UE4M3_MIN_NORMAL * (i as f32) / 20000.0);
    }
    for i in 0..=4000u32 {
        p.push(UE4M3_MAX * (i as f32) / 4000.0);
    }
    for e in -14..=10i32 {
        let base = exp2i(e);
        for j in -3..=3i32 {
            p.push(step_bits(base, j));
        }
    }
    p.extend_from_slice(&[
        0.0,
        UE4M3_MAX,
        step_bits(UE4M3_MAX, -1),
        step_bits(UE4M3_MAX, 1),
        500.0,
        1.0e30,
        f32::MIN_POSITIVE,
        f32::from_bits(1),
    ]);
    p.retain(|v| v.is_finite() && *v >= 0.0);
    p
}

fn normalize(s: &str) -> String {
    s.split_whitespace().collect::<Vec<_>>().join(" ")
}

fn cuda_sources() -> Vec<(String, String)> {
    fn walk(dir: &Path, out: &mut Vec<PathBuf>) {
        let entries = std::fs::read_dir(dir).unwrap_or_else(|e| {
            panic!(
                "{}: read_dir failed ({e}). A swallowed walk error is an empty census, and an \
                 empty census reads as a clean tree",
                dir.display()
            )
        });
        for entry in entries {
            let path = entry.expect("dir entry").path();
            if path.is_dir() {
                walk(&path, out);
            } else {
                let ext = path.extension().and_then(|s| s.to_str()).unwrap_or("");
                if matches!(ext, "cu" | "cuh" | "h") {
                    out.push(path);
                }
            }
        }
    }
    let root = Path::new(CUDA_DIR);
    let mut paths = Vec::new();
    walk(root, &mut paths);
    paths.sort();
    assert!(
        paths.len() >= CUDA_SOURCE_FLOOR,
        "walked only {} CUDA sources under {CUDA_DIR}; the census cannot be complete over a \
         tree this small, so a pass here would mean nothing",
        paths.len()
    );
    paths
        .into_iter()
        .map(|p| {
            let rel = p
                .strip_prefix(root)
                .expect("under cuda/")
                .to_string_lossy()
                .replace('\\', "/");
            let src = std::fs::read_to_string(&p)
                .unwrap_or_else(|e| panic!("{}: read failed ({e})", p.display()));
            (rel, src)
        })
        .collect()
}

fn device_fn_names(src: &str) -> Vec<String> {
    let mut names = Vec::new();
    for line in src.lines() {
        let t = line.trim_start();
        if !t.starts_with("__device__") && !t.starts_with("__global__") {
            continue;
        }
        let Some(paren) = t.find('(') else { continue };
        let name = t[..paren]
            .rsplit(|c: char| !(c.is_alphanumeric() || c == '_'))
            .next()
            .unwrap_or("");
        if !name.is_empty() {
            names.push(name.to_string());
        }
    }
    names
}

fn extract_fn(src: &str, file: &str, name: &str) -> String {
    let at = src
        .match_indices(&format!(" {name}("))
        .find(|(i, _)| {
            let line_start = src[..*i].rfind('\n').map(|n| n + 1).unwrap_or(0);
            src[line_start..*i].trim_start().starts_with("__device__")
        })
        .map(|(i, _)| i)
        .unwrap_or_else(|| {
            panic!(
                "{file}: no `__device__` definition of `{name}`. This gate pins the CUDA ue4m3 \
                 codec by text; a rename is a finding, not a pass."
            )
        });
    let open = at
        + src[at..]
            .find('{')
            .unwrap_or_else(|| panic!("{file}: `{name}` has no body"));
    let bytes = src.as_bytes();
    let mut depth = 0usize;
    let mut i = open;
    while i < bytes.len() {
        match bytes[i] {
            b'{' => depth += 1,
            b'}' => {
                depth -= 1;
                if depth == 0 {
                    return normalize(&src[open..=i]);
                }
            }
            _ => {}
        }
        i += 1;
    }
    panic!("{file}: `{name}` body is unbalanced");
}

fn extract_define(src: &str, file: &str, name: &str) -> f32 {
    let needle = format!("#define {name}");
    let at = src.find(&needle).unwrap_or_else(|| {
        panic!("{file}: `#define {name}` is gone; the ue4m3 subnormal branch is defined by it")
    });
    let line = src[at + needle.len()..].lines().next().unwrap_or("").trim();
    let lit = line.trim_end_matches('f');
    lit.parse::<f32>()
        .unwrap_or_else(|e| panic!("{file}: `{name}` value {lit:?} does not parse as f32: {e}"))
}

#[test]
fn every_cuda_ue4m3_codec_in_the_tree_is_the_same_pinned_function() {
    let sources = cuda_sources();
    let mut found: BTreeSet<(String, String)> = BTreeSet::new();
    for (file, src) in &sources {
        for name in device_fn_names(src) {
            if name.to_ascii_lowercase().contains("ue4m3") {
                found.insert((file.clone(), name));
            }
        }
        assert!(
            !src.contains(PRE_FIX_DECODE_FORM),
            "cuda/{file}: carries the pre-fix ue4m3 decode form `{PRE_FIX_DECODE_FORM}`. Biased \
             exponent 0 is the e4m3fn SUBNORMAL encoding (value = mant * 2^-9), not \
             (1 + mant/8) * 2^-7, so codes 0x00-0x07 read up to \
             {PREFIX_WORST_DECODE_RATIO}x high and the zero byte reads nonzero."
        );
        assert!(
            !src.contains(PRE_FIX_MANTISSA_CLAMP),
            "cuda/{file}: carries `{PRE_FIX_MANTISSA_CLAMP}`, which saturates the ue4m3 mantissa \
             instead of carrying into the exponent, so every scale that rounds up through a \
             power of two encodes one code low."
        );
    }
    let want: BTreeSet<(String, String)> = CUDA_UE4M3_CODEC_CENSUS
        .iter()
        .map(|(f, n, _)| ((*f).to_string(), (*n).to_string()))
        .collect();
    assert_eq!(
        found,
        want,
        "the set of ue4m3 codecs under cuda/ moved. Every copy must be the SAME function: a \
         self-consistently wrong encode/decode pair is invisible to any cross-backend parity \
         suite, and fixing one copy alone converts a hidden bug into a visible mismatch. Add the \
         new site to CUDA_UE4M3_CODEC_CENSUS and make its body one of the two pinned literals, \
         in the same commit.\nfound: {found:?}\nwant:  {want:?}"
    );

    let mut per_form: Vec<(CodecForm, usize)> = Vec::new();
    for (file, name, form) in CUDA_UE4M3_CODEC_CENSUS {
        let src = &sources
            .iter()
            .find(|(f, _)| f == file)
            .unwrap_or_else(|| panic!("cuda/{file} is gone"))
            .1;
        let body = extract_fn(src, file, name);
        let (want_body, label) = match form {
            CodecForm::Encode => (
                CUDA_ENCODE_BODY_TRANSCRIBED_BY_CU_ENCODE_UE4M3,
                "cu_encode_ue4m3",
            ),
            CodecForm::Decode => (
                CUDA_DECODE_BODY_TRANSCRIBED_BY_CU_DECODE_UE4M3,
                "cu_decode_ue4m3",
            ),
            CodecForm::DecodeBranchlessBitcast => (
                CUDA_DECODE_BRANCHLESS_BODY_TRANSCRIBED_BY_CU_DECODE_UE4M3_BRANCHLESS,
                "cu_decode_ue4m3_branchless",
            ),
        };
        assert_eq!(
            body, want_body,
            "cuda/{file}: `{name}` is no longer the function `{label}` in this test file \
             transcribes. Nothing else here reads the .cu: the other tests in this suite run the \
             RUST copy against an f64 e4m3fn oracle, so an edit to the CUDA that this literal \
             does not notice leaves them green while the shipped device code changes -- \
             roundf -> truncf in two .cu copies was measured to keep all four green before this \
             pin existed. Re-derive `{label}` from the new body, then update this literal in the \
             SAME commit.\ngot: {body}"
        );
        assert_eq!(
            extract_subnormal_step(src, file, form),
            UE4M3_SUBNORMAL_STEP,
            "cuda/{file}: the ue4m3 subnormal step drifted from 2^-9"
        );
        if form == CodecForm::Encode {
            assert_eq!(
                extract_define(src, file, "NV_UE4M3_MIN_NORMAL"),
                UE4M3_MIN_NORMAL,
                "cuda/{file}: NV_UE4M3_MIN_NORMAL drifted from 2^-6, the smallest ue4m3 normal"
            );
        }
        match per_form.iter_mut().find(|(f, _)| *f == form) {
            Some((_, n)) => *n += 1,
            None => per_form.push((form, 1)),
        }
    }
    let counted = |form: CodecForm| {
        per_form
            .iter()
            .find(|(f, _)| *f == form)
            .map(|(_, n)| *n)
            .unwrap_or(0)
    };
    assert_eq!(counted(CodecForm::Encode), 3, "three ue4m3 encoders were pinned");
    assert_eq!(
        counted(CodecForm::Decode),
        7,
        "seven #define-step ue4m3 decoders were pinned: four decode_ue4m3 and the three \
         decode_ue4m3_scale copies unified onto the same body -- a drifting copy hides from \
         cross-backend parity"
    );
    assert_eq!(
        counted(CodecForm::DecodeBranchlessBitcast),
        1,
        "one branchless bitcast ue4m3 decoder was pinned"
    );
    eprintln!(
        "every_cuda_ue4m3_codec_in_the_tree_is_the_same_pinned_function: {} CUDA sources walked, \
         {} codecs found across {:?}, each byte-identical to its pinned form",
        sources.len(),
        found.len(),
        per_form
    );
}

fn extract_subnormal_step(src: &str, file: &str, _form: CodecForm) -> f32 {
    extract_define(src, file, "NV_UE4M3_SUBNORMAL_STEP")
}

#[test]
fn the_cuda_decode_is_the_e4m3fn_value_set_on_every_code() {
    let mut distinct = std::collections::BTreeSet::new();
    for b in 0u8..=UE4M3_MAX_CODE {
        let want = e4m3fn_value(b).expect("0x00-0x7e are not the NaN code");
        let got = cu_decode_ue4m3(b) as f64;
        assert_eq!(
            got, want,
            "decode_ue4m3_dev({b:#04x}) = {got:e}, the e4m3fn definition says {want:e}"
        );
        let got_branchless = cu_decode_ue4m3_branchless(b) as f64;
        assert_eq!(
            got_branchless, want,
            "decode_ue4m3_branchless_dev({b:#04x}) = {got_branchless:e}, the e4m3fn definition \
             says {want:e}; the bitcast form builds (1 + mant/8) * 2^(biased - 7) from raw \
             exponent and mantissa fields and is only admissible while that identity holds"
        );
        distinct.insert(got.to_bits());
    }
    assert_eq!(
        distinct.len(),
        (UE4M3_MAX_CODE as usize) + 1,
        "the oracle collapsed codes together; agreement would prove nothing"
    );
    assert_eq!(
        cu_decode_ue4m3(0x00),
        0.0,
        "the zero scale byte must decode to zero"
    );
    assert!(
        e4m3fn_value(UE4M3_NAN_CODE).is_none(),
        "0x7f is the e4m3fn NaN code and has no value"
    );

    let mut pre_bad = 0usize;
    let mut worst_ratio = 1.0f64;
    let mut worst_byte = 0u8;
    for b in 0u8..=UE4M3_MAX_CODE {
        let want = e4m3fn_value(b).unwrap();
        let got = prefix_decode_ue4m3(b) as f64;
        if got != want {
            pre_bad += 1;
            if want > 0.0 && (got / want) > worst_ratio {
                worst_ratio = got / want;
                worst_byte = b;
            }
        }
    }
    assert_eq!(
        pre_bad, 8,
        "the pre-fix decode is meant to be wrong on exactly the 8 biased-exponent-0 codes; \
         got {pre_bad}. This control is what proves the assertions above can fail."
    );
    assert_eq!(worst_byte, 0x01);
    assert_eq!(worst_ratio, PREFIX_WORST_DECODE_RATIO);
    eprintln!(
        "the_cuda_decode_is_the_e4m3fn_value_set_on_every_code: 127 codes exact; pre-fix control wrong on {pre_bad} codes, worst {worst_ratio}x at {worst_byte:#04x}"
    );
}

#[test]
fn the_cuda_codec_is_the_e4m3fn_definition_over_all_256_byte_patterns() {
    let mut nan_codes = Vec::new();
    let mut subnormals = 0usize;
    let mut magnitudes = BTreeSet::new();
    for b in 0u16..=0xff {
        let b = b as u8;
        let Some(v) = e4m3fn_signed_value(b) else {
            nan_codes.push(b);
            assert_eq!(
                cu_encode_ue4m3(cu_decode_ue4m3(b)),
                UE4M3_MAX_CODE,
                "{b:#04x} is an e4m3fn NaN pattern; encode must never produce it"
            );
            continue;
        };
        let want = v.abs();
        let got = cu_decode_ue4m3(b) as f64;
        assert_eq!(
            got, want,
            "decode_ue4m3_dev({b:#04x}) = {got:e}; e4m3fn says {v:e}, magnitude {want:e}. The \
             ue4m3 decode drops bit 7 -- NVFP4 block scales are non-negative -- so it must equal \
             the magnitude of the signed e4m3fn value on every one of the 256 patterns."
        );
        if ((b >> 3) & 0x0f) == 0 {
            subnormals += 1;
            assert_eq!(
                got,
                (b & 0x07) as f64 * (-9f64).exp2(),
                "{b:#04x} has biased exponent 0, so its value is mant * 2^-9 by definition"
            );
        }
        magnitudes.insert(got.to_bits());
    }
    assert_eq!(
        nan_codes,
        vec![0x7f, 0xff],
        "e4m3fn has exactly two NaN patterns and no infinities"
    );
    assert_eq!(subnormals, 16, "16 of 256 patterns have biased exponent 0");
    assert_eq!(
        magnitudes.len(),
        127,
        "the 254 finite patterns carry 127 distinct magnitudes; a collapsed oracle would let \
         agreement prove nothing"
    );
    assert_eq!(
        e4m3fn_signed_value(0x7e),
        Some(448.0),
        "0x7e is the largest finite e4m3fn value and the ue4m3 clamp target"
    );
    assert_eq!(e4m3fn_signed_value(0x80), Some(-0.0));
    assert_eq!(e4m3fn_signed_value(0x08), Some(0.015625));
    assert_eq!(e4m3fn_signed_value(0x07), Some(7.0 * (-9f64).exp2()));

    let mut pre_bad = 0usize;
    for b in 0u16..=0xff {
        let b = b as u8;
        if let Some(v) = e4m3fn_signed_value(b) {
            if prefix_decode_ue4m3(b) as f64 != v.abs() {
                pre_bad += 1;
            }
        }
    }
    assert_eq!(
        pre_bad, 16,
        "the pre-fix decode must be wrong on exactly the 16 biased-exponent-0 patterns; this \
         control is what proves the 256-pattern sweep can fail"
    );
    eprintln!(
        "the_cuda_codec_is_the_e4m3fn_definition_over_all_256_byte_patterns: 254 finite patterns \
         exact ({subnormals} subnormal), 2 NaN patterns unreachable from encode, pre-fix control \
         wrong on {pre_bad}"
    );
}

#[test]
fn the_cuda_encode_picks_the_nearest_e4m3fn_code_including_below_two_to_the_minus_six() {
    let probes = encode_probes();
    assert!(
        probes.len() > 25000,
        "probe set collapsed to {} points",
        probes.len()
    );
    let in_band = probes
        .iter()
        .filter(|v| **v > 0.0 && **v < UE4M3_MIN_NORMAL)
        .count();
    assert!(
        in_band > 10000,
        "only {in_band} probes land below 2^-6; this gate exists for that band"
    );

    let mut bad: Vec<(f32, u8, u8)> = Vec::new();
    for t in &probes {
        let want = e4m3fn_nearest_code(*t as f64);
        let got = cu_encode_ue4m3(*t);
        if got != want {
            bad.push((*t, want, got));
        }
    }
    assert!(
        bad.is_empty(),
        "{}/{} probes: encode_ue4m3_dev did not pick the nearest e4m3fn code. First: \
         target {:e} nearest {:#04x} ({:e}) got {:#04x} ({:e})",
        bad.len(),
        probes.len(),
        bad[0].0,
        bad[0].1,
        e4m3fn_value(bad[0].1).unwrap(),
        bad[0].2,
        e4m3fn_value(bad[0].2).unwrap()
    );

    let mut pre_bad = 0usize;
    for t in &probes {
        if prefix_encode_ue4m3(*t) != e4m3fn_nearest_code(*t as f64) {
            pre_bad += 1;
        }
    }
    assert!(
        pre_bad >= in_band / 2,
        "the pre-fix encode control missed only {pre_bad} of {} probes ({in_band} in band); \
         it is supposed to be wrong across the whole subnormal band, so this gate is not \
         measuring what it claims",
        probes.len()
    );
    eprintln!(
        "the_cuda_encode_picks_the_nearest_e4m3fn_code_including_below_two_to_the_minus_six: {} probes ({in_band} below 2^-6) all exact; pre-fix control missed {pre_bad}",
        probes.len()
    );
}

#[test]
fn the_rmsnorm_prefix_encode_missed_the_nearest_code_in_two_independent_ways() {
    let probes = encode_probes();
    let mut in_band = 0usize;
    let mut at_carry = 0usize;
    let mut first_carry: Option<(f32, u8, u8)> = None;
    for t in &probes {
        let want = e4m3fn_nearest_code(*t as f64);
        let got = rq_prefix_encode_ue4m3(*t);
        if got == want {
            continue;
        }
        if *t < UE4M3_MIN_NORMAL {
            in_band += 1;
        } else {
            at_carry += 1;
            if first_carry.is_none() {
                first_carry = Some((*t, want, got));
            }
        }
    }
    assert!(
        in_band > 10000,
        "the rmsnorm pre-fix encode missed only {in_band} probes below 2^-6; it had no \
         subnormal branch at all, so it must miss the whole band"
    );
    let (t, want, got) = first_carry.expect(
        "the rmsnorm pre-fix encode saturated the mantissa at 7 instead of carrying into the \
         exponent, so it must ALSO miss in the normal band -- a defect independent of the \
         subnormal one, and the reason this file needed more than the subnormal port",
    );
    assert!(
        at_carry > 100,
        "only {at_carry} normal-band misses; the mantissa-carry defect should show up at every \
         power of two"
    );
    assert_ne!(got, want);
    assert!(
        cu_encode_ue4m3(t) == want,
        "the shipped encode must pick the nearest code at {t:e} where the pre-fix one picked \
         {got:#04x} instead of {want:#04x}"
    );
    eprintln!(
        "the_rmsnorm_prefix_encode_missed_the_nearest_code_in_two_independent_ways: {in_band} \
         misses below 2^-6, {at_carry} in the normal band (first {t:e} -> {got:#04x}, nearest \
         {want:#04x})"
    );
}

#[test]
fn the_seven_subnormal_scale_targets_a_checkpoint_emits_round_trip_within_one_code() {
    let targets = subnormal_band_targets();
    assert_eq!(targets.len(), 7);
    let half_step = 0.5 * UE4M3_SUBNORMAL_STEP as f64;

    let mut got_codes = Vec::new();
    for (i, t) in targets.iter().enumerate() {
        assert!(
            *t > 0.0 && *t < UE4M3_MIN_NORMAL,
            "target {i} = {t:e} is not in the ue4m3 subnormal band"
        );
        let b = cu_encode_ue4m3(*t);
        assert_eq!(
            b,
            (i + 1) as u8,
            "scale {t:e} = {}*2^-9 must encode to {:#04x}",
            i + 1,
            i + 1
        );
        assert_ne!(
            b, 0,
            "scale {t:e} encoded to 0x00: the block's scale is zero and the whole block is lost"
        );
        let back = cu_decode_ue4m3(b) as f64;
        assert!(
            (back - *t as f64).abs() <= half_step,
            "round trip of {t:e} through {b:#04x} lands at {back:e}, more than half a \
             subnormal step ({half_step:e}) away -- encode_ue4m3_dev and decode_ue4m3_dev \
             disagree about the subnormal codes, which is what sets `inv` for the nibbles"
        );
        got_codes.push(b);
    }

    let pre_codes: Vec<u8> = targets.iter().map(|t| prefix_encode_ue4m3(*t)).collect();
    assert_eq!(
        pre_codes.as_slice(),
        PREFIX_CODES_OVER_THE_SUBNORMAL_BAND.as_slice(),
        "the pre-fix encode control no longer reproduces the measured defect; without it \
         this gate has no proof it can fail"
    );
    let lost = pre_codes.iter().filter(|b| **b == 0).count();
    assert_eq!(
        lost, PREFIX_LOST_BLOCKS_OVER_THE_SUBNORMAL_BAND,
        "the pre-fix control is meant to lose {PREFIX_LOST_BLOCKS_OVER_THE_SUBNORMAL_BAND} \
         of 7 blocks outright"
    );
    let mut worst_pre = 1.0f64;
    for (t, b) in targets.iter().zip(pre_codes.iter()) {
        let back = prefix_decode_ue4m3(*b) as f64;
        let r = back / *t as f64;
        if r > worst_pre {
            worst_pre = r;
        }
    }
    eprintln!(
        "the_seven_subnormal_scale_targets_a_checkpoint_emits_round_trip_within_one_code: post-fix codes {got_codes:?}; pre-fix {pre_codes:?} ({lost} lost, worst round-trip {worst_pre:.4}x)"
    );
}

#[test]
fn the_four_pinned_cuda_files_are_the_ones_this_suite_compiles_against() {
    let sources = cuda_sources();
    for (label, src) in [
        ("gemv_nvfp4.cu", GEMV_CU),
        ("quantize_nvfp4_bf16.cu", QUANT_CU),
        ("rmsnorm_quantize_nvfp4_bf16.cu", RMSQ_CU),
        ("moe_grouped_fp4_gemv.cu", MOE_CU),
    ] {
        let on_disk = sources
            .iter()
            .find(|(f, _)| f == label)
            .unwrap_or_else(|| panic!("cuda/{label} is gone"))
            .1
            .as_str();
        assert_eq!(
            src, on_disk,
            "cuda/{label} changed without rebuilding this test. The include_str! copies exist so \
             a cargo run after a .cu edit recompiles the gate instead of replaying a stale one."
        );
        assert!(
            CUDA_UE4M3_CODEC_CENSUS.iter().any(|(f, _, _)| *f == label),
            "cuda/{label} is include_str!'d here but carries no censused ue4m3 codec"
        );
    }
}

#[test]
fn the_branchless_decode_is_the_same_function_as_the_pinned_decode_on_every_byte() {
    for b in 0u16..=255 {
        let b = b as u8;
        let got = cu_decode_ue4m3_branchless(b);
        let want = cu_decode_ue4m3(b);
        assert_eq!(
            got.to_bits(),
            want.to_bits(),
            "decode_ue4m3_branchless_dev({b:#04x}) = {got:e} but decode_ue4m3_dev says {want:e}; \
             the branchless bit-construction is only admissible in the census while it stays \
             bit-identical to the pinned decode on every byte pattern"
        );
    }
}
