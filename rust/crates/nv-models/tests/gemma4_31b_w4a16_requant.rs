#![cfg(feature = "wgpu")]

use nv_models::gemma4_wgpu::{
    dequantize_nvfp4_host, dequantize_w4a16_host, nvfp4_to_w4a16, quantize_nvfp4_host,
    quantize_w4a16_host, HostNvfp4Lin, W4ScalePolicy,
};
mod common;
use common::bf16_val as bf16_f32;

const HIDDEN: usize = 5376;
const INTER: usize = 21504;

fn snapshot_dir() -> Option<std::path::PathBuf> {
    if let Ok(d) = std::env::var("NV_GEMMA4_DIR") {
        return Some(std::path::PathBuf::from(d));
    }
    let home = std::env::var("HOME").ok()?;
    let base = std::path::PathBuf::from(home)
        .join(".cache/huggingface/hub/models--nvidia--Gemma-4-31B-IT-NVFP4/snapshots");
    std::fs::read_dir(&base)
        .ok()?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .find(|p| p.join("config.json").exists())
}

fn gated(var: &str) -> bool {
    if std::env::var(var).ok().as_deref() == Some("1") {
        return true;
    }
    eprintln!("skipping: set {var}=1 to run (reads the real 31B checkpoint)");
    false
}

fn banner(dir: &std::path::Path) {
    eprintln!("checkpoint: {}", dir.display());
    let up = std::process::Command::new("uptime").output().unwrap();
    eprintln!("uptime: {}", String::from_utf8_lossy(&up.stdout).trim());
}

fn rows_env() -> usize {
    std::env::var("NV_G31_W4A16_ROWS")
        .ok()
        .and_then(|v| v.trim().parse().ok())
        .unwrap_or(2048)
}

fn layers_env() -> Vec<usize> {
    std::env::var("NV_G31_W4A16_LAYERS")
        .ok()
        .map(|v| v.split(',').filter_map(|s| s.trim().parse().ok()).collect())
        .unwrap_or_else(|| vec![0, 30, 59])
}

fn safe_recip(x: f32) -> f32 {
    if x == 0.0 || !x.is_finite() {
        1.0
    } else {
        1.0 / x
    }
}

fn scalar_f32(w: &nv_weights::WeightLoader, name: &str) -> f32 {
    let t = w.get(name, candle_core::DType::F32).expect("scalar tensor");
    let v: Vec<f32> = t.flatten_all().unwrap().to_vec1().unwrap();
    *v.first().unwrap_or(&1.0)
}

fn load_nvfp4_rows(
    w: &nv_weights::WeightLoader,
    module: &str,
    n_full: usize,
    k: usize,
    rows: usize,
) -> HostNvfp4Lin {
    let pname = format!("{module}.weight");
    let shape = w
        .shape_of(&pname)
        .unwrap_or_else(|| panic!("missing {pname}"));
    assert_eq!(shape, vec![n_full, k / 2], "{pname} shape");
    let n = rows.min(n_full);
    let packed = w.raw_bytes(&pname).unwrap()[..n * (k / 2)].to_vec();
    let scales_lin = &w.raw_bytes(&format!("{module}.weight_scale")).unwrap()[..n * (k / 16)];
    let stored_w = safe_recip(scalar_f32(w, &format!("{module}.weight_scale_2")));
    let stored_x = safe_recip(scalar_f32(w, &format!("{module}.input_scale")));
    HostNvfp4Lin {
        packed,
        scales_swizzled: nv_quant::nvfp4::swizzle_scales(scales_lin, n, k / 16),
        alpha: safe_recip(stored_w) * safe_recip(stored_x),
        input_global: stored_x,
        n,
        k,
    }
}

fn load_bf16_rows(
    w: &nv_weights::WeightLoader,
    name: &str,
    rows: usize,
) -> (Vec<u16>, usize, usize) {
    let shape = w.shape_of(name).unwrap_or_else(|| panic!("missing {name}"));
    assert_eq!(shape.len(), 2, "{name} is not a matrix: {shape:?}");
    let (n_full, k) = (shape[0], shape[1]);
    assert_eq!(
        w.st_dtype_of(name),
        Some(nv_weights::StDtype::BF16),
        "{name} must be stored bf16 for the PTQ arm to mean anything"
    );
    let n = rows.min(n_full);
    let bytes = &w.raw_bytes(name).unwrap()[..n * k * 2];
    let bits: Vec<u16> = bytes
        .chunks_exact(2)
        .map(|c| u16::from_le_bytes([c[0], c[1]]))
        .collect();
    (bits, n, k)
}

fn f32_to_bf16_bits_rne(x: f32) -> u16 {
    let u = x.to_bits();
    let round = ((u >> 16) & 1) + 0x7fff;
    (u.wrapping_add(round) >> 16) as u16
}

struct Err2 {
    rel_rms: f64,
    max_abs: f64,
    rms_ref: f64,
}

fn err_of(reference: &[f32], got: &[f32]) -> Err2 {
    assert_eq!(reference.len(), got.len());
    let (mut se, mut sr, mut mx) = (0f64, 0f64, 0f64);
    for (a, b) in reference.iter().zip(got.iter()) {
        let d = (*b as f64) - (*a as f64);
        se += d * d;
        sr += (*a as f64) * (*a as f64);
        if d.abs() > mx {
            mx = d.abs();
        }
    }
    let n = reference.len() as f64;
    Err2 {
        rel_rms: (se / n).sqrt() / (sr / n).sqrt(),
        max_abs: mx,
        rms_ref: (sr / n).sqrt(),
    }
}

fn probe_x(k: usize, seed: u64) -> Vec<f32> {
    let mut s = seed;
    (0..k)
        .map(|_| {
            let mut acc = 0f32;
            for _ in 0..4 {
                s = s
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                acc += ((s >> 32) as u32 as f32 / 2147483648.0) - 1.0;
            }
            acc * 0.5
        })
        .collect()
}

fn gemv_rel_err(reference: &[f32], got: &[f32], n: usize, k: usize, x: &[f32]) -> f64 {
    let (mut se, mut sr) = (0f64, 0f64);
    for r in 0..n {
        let (mut ya, mut yb) = (0f64, 0f64);
        for j in 0..k {
            ya += (reference[r * k + j] * x[j]) as f64;
            yb += (got[r * k + j] * x[j]) as f64;
        }
        se += (yb - ya) * (yb - ya);
        sr += ya * ya;
    }
    (se / sr).sqrt()
}

fn block_scale_spread(l: &HostNvfp4Lin, group: usize) -> (Vec<usize>, f64) {
    let bpg = group / 16;
    let k_blocks = l.k / 16;
    let k_tiles = k_blocks.div_ceil(4);
    let mut hist = vec![0usize; 5];
    let mut worst = 1f64;
    for r in 0..l.n {
        for g in 0..(k_blocks / bpg) {
            let (mut lo, mut hi) = (f32::INFINITY, 0f32);
            for bb in 0..bpg {
                let kb = g * bpg + bb;
                let si =
                    ((r / 128 * k_tiles + kb / 4) * 32 + r % 32) * 16 + ((r / 32) % 4) * 4 + kb % 4;
                let sb = l.scales_swizzled[si] as u32;
                let (e, m) = ((sb >> 3) & 15, sb & 7);
                let s = if e == 0 {
                    (m as f32) * 0.001953125f32
                } else {
                    f32::from_bits(((e + 120) << 23) | (m << 20))
                };
                lo = lo.min(s);
                hi = hi.max(s);
            }
            let ratio = if lo > 0.0 {
                (hi / lo) as f64
            } else {
                f64::INFINITY
            };
            if ratio.is_finite() {
                worst = worst.max(ratio);
            }
            let b = if !ratio.is_finite() || ratio >= 8.0 {
                4
            } else if ratio <= 1.0001 {
                0
            } else if ratio < 2.0 {
                1
            } else if ratio < 4.0 {
                2
            } else {
                3
            };
            hist[b] += 1;
        }
    }
    (hist, worst)
}

fn int8_roundtrip(bits: &[u16], n: usize, k: usize, group: usize) -> Vec<f32> {
    use nv_kernels::wgpu_backend::kernels::quant_gemv as qg;
    let (wq, scales) = qg::quantize_groups(bits, n, k, group, qg::QFormat::Int8);
    let per_row = k / group;
    let mut out = vec![0f32; n * k];
    for r in 0..n {
        for g in 0..per_row {
            let s = scales[r * per_row + g];
            for i in 0..group {
                let idx = r * k + g * group + i;
                let byte = ((wq[idx / 4] >> (8 * (idx % 4))) & 0xff) as u8 as i8;
                out[idx] = (byte as f32) * s;
            }
        }
    }
    out
}

fn nvfp4_true(l: &HostNvfp4Lin) -> Vec<f32> {
    let gi = if l.input_global == 0.0 || !l.input_global.is_finite() {
        1.0
    } else {
        l.input_global
    };
    let mut w = dequantize_nvfp4_host(l);
    for v in w.iter_mut() {
        *v *= gi;
    }
    w
}

fn row(tag: &str, gs: usize, e: &Err2, ye: f64, bpw: f64) {
    println!(
        "{tag:<34}{gs:>5}{:>12.4e}{:>12.4e}{:>12.4e}{bpw:>10.4}",
        e.rel_rms, e.max_abs, ye
    );
}

fn header() {
    println!(
        "{:<34}{:>5}{:>12}{:>12}{:>12}{:>10}",
        "tensor / arm", "GS", "rel_rms_w", "max_abs_w", "rel_err_y", "B/weight"
    );
}

#[test]
#[ignore]
fn gemma4_31b_w4a16_requant_roundtrip_error() {
    if !gated("NV_G31_W4A16_STUDY") {
        return;
    }
    let dir = snapshot_dir().expect("no Gemma-4-31B-IT-NVFP4 snapshot; set NV_GEMMA4_DIR");
    banner(&dir);
    let rows = rows_env();
    eprintln!("rows sampled per tensor: {rows}");
    let loader = nv_weights::WeightLoader::open_dir(&dir, &candle_core::Device::Cpu)
        .expect("open the 31B safetensors dir");

    header();
    let (mut worst_ctl, mut worst_w4_16, mut worst_w4_32) = (0f64, 0f64, 0f64);
    for li in layers_env() {
        let p = format!("model.language_model.layers.{li}");
        for (nm, module, n_full, k) in [
            (
                format!("L{li}.gate"),
                format!("{p}.mlp.gate_proj"),
                INTER,
                HIDDEN,
            ),
            (
                format!("L{li}.up"),
                format!("{p}.mlp.up_proj"),
                INTER,
                HIDDEN,
            ),
            (
                format!("L{li}.down"),
                format!("{p}.mlp.down_proj"),
                HIDDEN,
                INTER,
            ),
        ] {
            let l = load_nvfp4_rows(&loader, &module, n_full, k, rows);
            let (n, k) = (l.n, l.k);
            let reference = nvfp4_true(&l);
            assert!(
                err_of(&reference, &reference).rms_ref > 0.0,
                "{nm}: reference is all zero -- the loader read nothing"
            );
            let x = probe_x(k, 0x51ed);

            let (hist, worst) = block_scale_spread(&l, 32);
            let tot: f64 = hist.iter().sum::<usize>() as f64;
            eprintln!(
                "[{nm}] nvfp4 block-scale spread inside a 32-group: \
                 =1 {:.1}%  <2x {:.1}%  <4x {:.1}%  <8x {:.1}%  >=8x {:.1}%  worst {worst:.1}x",
                100.0 * hist[0] as f64 / tot,
                100.0 * hist[1] as f64 / tot,
                100.0 * hist[2] as f64 / tot,
                100.0 * hist[3] as f64 / tot,
                100.0 * hist[4] as f64 / tot,
            );

            let bits: Vec<u16> = reference.iter().map(|v| f32_to_bf16_bits_rne(*v)).collect();
            let staged: Vec<f32> = bits.iter().map(|b| bf16_f32(*b)).collect();
            row(
                &format!("{nm} bf16 staging only"),
                0,
                &err_of(&reference, &staged),
                gemv_rel_err(&reference, &staged, n, k, &x),
                2.0,
            );
            drop(staged);

            let ctl = int8_roundtrip(&bits, n, k, 128);
            let e = err_of(&reference, &ctl);
            row(
                &format!("{nm} int8 CONTROL"),
                128,
                &e,
                gemv_rel_err(&reference, &ctl, n, k, &x),
                1.0 + 4.0 / 128.0,
            );
            worst_ctl = worst_ctl.max(e.rel_rms);
            drop(ctl);

            for (gs, pol, tag) in [
                (16usize, W4ScalePolicy::Amax, "w4a16 amax"),
                (16, W4ScalePolicy::MseSearch, "w4a16 mse"),
                (32, W4ScalePolicy::Amax, "w4a16 amax"),
                (32, W4ScalePolicy::MseSearch, "w4a16 mse"),
            ] {
                let q = quantize_w4a16_host(&bits, n, k, gs, pol);
                assert_eq!(q.packed.len(), n * k / 8, "{nm}: packed word count");
                assert_eq!(q.scales.len(), n * k / gs, "{nm}: scale count");
                let deq = dequantize_w4a16_host(&q);
                let e = err_of(&reference, &deq);
                row(
                    &format!("{nm} {tag}"),
                    gs,
                    &e,
                    gemv_rel_err(&reference, &deq, n, k, &x),
                    0.5 + 2.0 / gs as f64,
                );
                if pol == W4ScalePolicy::MseSearch {
                    if gs == 16 {
                        worst_w4_16 = worst_w4_16.max(e.rel_rms);
                    } else {
                        worst_w4_32 = worst_w4_32.max(e.rel_rms);
                    }
                }
            }
        }
    }
    println!(
        "\nworst-tensor relative weight rms:  int8/128 CONTROL {worst_ctl:.4e}  |  \
         w4a16/16 mse {worst_w4_16:.4e} ({:.1}x control)  |  w4a16/32 mse {worst_w4_32:.4e} ({:.1}x control)",
        worst_w4_16 / worst_ctl,
        worst_w4_32 / worst_ctl
    );
    assert!(
        worst_ctl > 0.0 && worst_w4_16 > 0.0,
        "study produced no rows -- the layer list was empty"
    );
}

#[test]
#[ignore]
fn gemma4_31b_4bit_code_comparison() {
    if !gated("NV_G31_W4A16_STUDY") {
        return;
    }
    let dir = snapshot_dir().expect("no Gemma-4-31B-IT-NVFP4 snapshot; set NV_GEMMA4_DIR");
    banner(&dir);
    let rows = rows_env();
    eprintln!("rows sampled per tensor: {rows}");
    let loader = nv_weights::WeightLoader::open_dir(&dir, &candle_core::Device::Cpu).unwrap();

    header();
    for li in layers_env() {
        let p = format!("model.language_model.layers.{li}");
        for (nm, name) in [
            (format!("L{li}.q"), format!("{p}.self_attn.q_proj.weight")),
            (format!("L{li}.o"), format!("{p}.self_attn.o_proj.weight")),
        ] {
            let (bits, n, k) = load_bf16_rows(&loader, &name, rows);
            let reference: Vec<f32> = bits.iter().map(|b| bf16_f32(*b)).collect();
            let x = probe_x(k, 0x51ed);

            let ctl = int8_roundtrip(&bits, n, k, 128);
            row(
                &format!("{nm} bf16->int8 SHIPPING"),
                128,
                &err_of(&reference, &ctl),
                gemv_rel_err(&reference, &ctl, n, k, &x),
                1.0 + 4.0 / 128.0,
            );
            drop(ctl);

            let nv = quantize_nvfp4_host(&bits, n, k);
            let deq = nvfp4_true(&nv);
            let e_nv = err_of(&reference, &deq);
            row(
                &format!("{nm} bf16->nvfp4 e2m1"),
                16,
                &e_nv,
                gemv_rel_err(&reference, &deq, n, k, &x),
                0.5 + 1.0 / 16.0,
            );
            drop(deq);
            drop(nv);

            for gs in [16usize, 32] {
                let q = quantize_w4a16_host(&bits, n, k, gs, W4ScalePolicy::MseSearch);
                let deq = dequantize_w4a16_host(&q);
                let e = err_of(&reference, &deq);
                row(
                    &format!("{nm} bf16->w4a16 int4"),
                    gs,
                    &e,
                    gemv_rel_err(&reference, &deq, n, k, &x),
                    0.5 + 2.0 / gs as f64,
                );
                if gs == 16 {
                    println!(
                        "   -> at 4 bits and group-16, uniform int4 is {:.2}x the error of e2m1",
                        e.rel_rms / e_nv.rel_rms
                    );
                }
            }
        }
    }
}

#[test]
#[ignore]
fn gemma4_31b_w4a16_direct_from_bf16_vs_transcode_same_tensor() {
    if !gated("NV_G31_W4A16_STUDY") {
        return;
    }
    let dir = snapshot_dir().expect("no Gemma-4-31B-IT-NVFP4 snapshot; set NV_GEMMA4_DIR");
    banner(&dir);
    let rows = rows_env();
    eprintln!("rows sampled per tensor: {rows}");
    let loader = nv_weights::WeightLoader::open_dir(&dir, &candle_core::Device::Cpu).unwrap();

    header();
    let mut ratios: Vec<(usize, f64)> = Vec::new();
    for li in layers_env() {
        let p = format!("model.language_model.layers.{li}");
        for (nm, name) in [
            (format!("L{li}.q"), format!("{p}.self_attn.q_proj.weight")),
            (format!("L{li}.k"), format!("{p}.self_attn.k_proj.weight")),
            (format!("L{li}.o"), format!("{p}.self_attn.o_proj.weight")),
        ] {
            let (bits, n, k) = load_bf16_rows(&loader, &name, rows);
            let reference: Vec<f32> = bits.iter().map(|b| bf16_f32(*b)).collect();
            let x = probe_x(k, 0x51ed);

            let nv = quantize_nvfp4_host(&bits, n, k);

            for gs in [16usize, 32] {
                let a = quantize_w4a16_host(&bits, n, k, gs, W4ScalePolicy::MseSearch);
                let ea = err_of(&reference, &dequantize_w4a16_host(&a));
                row(
                    &format!("{nm} A DIRECT bf16->w4"),
                    gs,
                    &ea,
                    gemv_rel_err(&reference, &dequantize_w4a16_host(&a), n, k, &x),
                    0.5 + 2.0 / gs as f64,
                );

                let b = nvfp4_to_w4a16(&nv, gs, W4ScalePolicy::MseSearch)
                    .unwrap_or_else(|| panic!("{nm}: nvfp4_to_w4a16 refused GS={gs}"));
                let eb = err_of(&reference, &dequantize_w4a16_host(&b));
                row(
                    &format!("{nm} B TRANSCODE nv->w4"),
                    gs,
                    &eb,
                    gemv_rel_err(&reference, &dequantize_w4a16_host(&b), n, k, &x),
                    0.5 + 2.0 / gs as f64,
                );
                println!(
                    "   -> GS={gs}: transcoding costs {:.3}x  (direct {:.4e} -> transcoded {:.4e})",
                    eb.rel_rms / ea.rel_rms,
                    ea.rel_rms,
                    eb.rel_rms
                );
                ratios.push((gs, eb.rel_rms / ea.rel_rms));
                assert!(ea.rel_rms > 0.0 && eb.rel_rms > 0.0, "{nm}: empty arm");
            }
        }
    }
    assert!(!ratios.is_empty(), "study produced no rows");
    for gs in [16usize, 32] {
        let v: Vec<f64> = ratios
            .iter()
            .filter(|(g, _)| *g == gs)
            .map(|(_, r)| *r)
            .collect();
        println!(
            "\nGS={gs}: transcode penalty over {} tensors -- min {:.3}x  mean {:.3}x  max {:.3}x",
            v.len(),
            v.iter().cloned().fold(f64::INFINITY, f64::min),
            v.iter().sum::<f64>() / v.len() as f64,
            v.iter().cloned().fold(0.0, f64::max),
        );
    }
}

#[test]
#[ignore]
fn gemma4_31b_which_tensors_have_a_bf16_source() {
    if !gated("NV_G31_W4A16_STUDY") {
        return;
    }
    let dir = snapshot_dir().expect("no Gemma-4-31B-IT-NVFP4 snapshot; set NV_GEMMA4_DIR");
    banner(&dir);
    let loader = nv_weights::WeightLoader::open_dir(&dir, &candle_core::Device::Cpu).unwrap();

    let p = "model.language_model.layers.0";
    let mut bf16_roles: Vec<&str> = Vec::new();
    let mut quant_roles: Vec<&str> = Vec::new();
    for (role, sub) in [
        ("q", "self_attn.q_proj"),
        ("k", "self_attn.k_proj"),
        ("v", "self_attn.v_proj"),
        ("o", "self_attn.o_proj"),
        ("gate", "mlp.gate_proj"),
        ("up", "mlp.up_proj"),
        ("down", "mlp.down_proj"),
    ] {
        let name = format!("{p}.{sub}.weight");
        let dt = loader
            .st_dtype_of(&name)
            .unwrap_or_else(|| panic!("missing {name} -- checkpoint layout changed"));
        let shape = loader.shape_of(&name).unwrap();
        let has_nv_scale = loader
            .st_dtype_of(&format!("{p}.{sub}.weight_scale"))
            .is_some();
        println!("{role:<6} {name:<58} {dt:?} {shape:?} nvfp4_scales={has_nv_scale}");
        if dt == nv_weights::StDtype::BF16 {
            bf16_roles.push(role);
        } else {
            quant_roles.push(role);
        }
    }
    println!("\nbf16 source available for: {bf16_roles:?}");
    println!("no bf16 source (nvfp4 only): {quant_roles:?}");

    assert_eq!(
        quant_roles,
        vec!["gate", "up", "down"],
        "FFN storage changed -- re-price the w4a16 lane, the blocker may be gone"
    );
    assert_eq!(
        bf16_roles,
        vec!["q", "k", "v", "o"],
        "attention storage changed"
    );

    let bf16_repo = std::env::var("HOME").map(|h| {
        std::path::PathBuf::from(h)
            .join(".cache/huggingface/hub/models--google--gemma-4-31B-it/snapshots")
    });
    let present = bf16_repo
        .as_ref()
        .ok()
        .and_then(|b| std::fs::read_dir(b).ok())
        .map(|rd| {
            rd.filter_map(|e| e.ok()).any(|e| {
                std::fs::read_dir(e.path())
                    .map(|inner| {
                        inner.filter_map(|f| f.ok()).any(|f| {
                            let n = f.file_name();
                            let n = n.to_string_lossy();
                            n.starts_with("model") && n.ends_with(".safetensors")
                        })
                    })
                    .unwrap_or(false)
            })
        })
        .unwrap_or(false);
    println!(
        "google/gemma-4-31B-it (bf16) safetensors present: {present}  \
         -- if false, 'quantize the FFN from bf16' is not executable here"
    );
}

#[test]
fn w4a16_host_pack_is_exactly_invertible() {
    let (n, k, gs) = (8usize, 128usize, 32usize);
    let mut s = 0x1234_5678u64;
    let w: Vec<u16> = (0..n * k)
        .map(|_| {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);

            let u = ((s >> 32) as u32 as f32 / 2147483648.0) - 1.0;
            f32_to_bf16_bits_rne(u * 0.05)
        })
        .collect();
    let reference: Vec<f32> = w.iter().map(|b| bf16_f32(*b)).collect();
    for policy in [W4ScalePolicy::Amax, W4ScalePolicy::MseSearch] {
        let q = quantize_w4a16_host(&w, n, k, gs, policy);
        let deq = dequantize_w4a16_host(&q);
        let bits: Vec<u16> = deq.iter().map(|v| f32_to_bf16_bits_rne(*v)).collect();
        let q2 = quantize_w4a16_host(&bits, n, k, gs, policy);
        let deq2 = dequantize_w4a16_host(&q2);
        if policy == W4ScalePolicy::Amax {
            assert_eq!(
                q.scales, q2.scales,
                "{policy:?}: scales moved on re-quantize"
            );
            assert_eq!(
                q.packed, q2.packed,
                "{policy:?}: nibbles moved on re-quantize"
            );
        } else {
            let (e1, e2) = (
                err_of(&reference, &deq).rel_rms,
                err_of(&reference, &deq2).rel_rms,
            );
            assert!(
                e2 <= e1 * 1.05,
                "{policy:?}: re-quantize drifted, {e1:.4e} -> {e2:.4e}"
            );
            eprintln!("{policy:?}: re-quantize does not drift ({e1:.4e} -> {e2:.4e})");
        }
        let (mut lo, mut hi) = (15u32, 0u32);
        for word in &q.packed {
            for e in 0..8 {
                let v = (word >> (4 * e)) & 15;
                lo = lo.min(v);
                hi = hi.max(v);
            }
        }

        assert!(
            hi == 15 && lo <= 1,
            "{policy:?}: code range [{lo}, {hi}] does not span +-7"
        );
        eprintln!("{policy:?}: pack/unpack idempotent, nibble range [{lo}, {hi}]");
    }
}

#[test]
fn w4a16_dequant_matches_the_shader_expression() {
    use nv_models::gemma4_wgpu::HostW4a16Lin;
    let scale = half::bf16::from_f32(0.0125).to_bits();
    let l = HostW4a16Lin {
        packed: vec![0x7654_3210, 0xfedc_ba98],
        scales: vec![scale, scale],
        n: 1,
        k: 16,
        group: 8,
    };
    let got = dequantize_w4a16_host(&l);
    let s = bf16_f32(scale);
    let want: Vec<f32> = (0..16).map(|q| (q as f32 - 8.0) * s).collect();
    assert_eq!(got, want, "dequantize_w4a16_host != (nibble - 8) * scale");
    eprintln!("dequant matches (nibble - 8) * bf16(scale) on all 16 codes");
}
