#![cfg(feature = "wgpu")]

use candle_core::{DType, Device};
use nv_kernels::wgpu_backend::kernels::kv_fp8::{decode_e4m3, encode_e4m3};
use nv_models::gemma4::Gemma4Config;
mod common;
use common::bf16_val as f;

struct Lcg(u64);

impl Lcg {
    fn next(&mut self) -> f32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let bits = (self.0 >> 32) as u32;
        (bits as f64 / 4294967296.0) as f32 - 0.5
    }
    fn gauss(&mut self) -> f32 {
        let mut s = 0f32;
        for _ in 0..12 {
            s += self.next();
        }
        s
    }
}

fn amax(v: &[f32]) -> f32 {
    v.iter()
        .fold(0f32, |a, b| if b.is_finite() { a.max(b.abs()) } else { a })
}

fn q_e4m3_groups(row: &[f32], group: usize) -> (Vec<f32>, usize) {
    let g = if group == 0 { row.len() } else { group };
    let mut out = vec![0f32; row.len()];
    let mut subn = 0usize;
    for (gi, chunk) in row.chunks(g).enumerate() {
        let a = amax(chunk);
        let (s, inv) = if a > 0.0 {
            (a / 448.0, 448.0 / a)
        } else {
            (0.0, 0.0)
        };
        for (i, v) in chunk.iter().enumerate() {
            let code = encode_e4m3(*v * inv);
            if (code & 0x78) == 0 && (code & 7) != 0 {
                subn += 1;
            }
            out[gi * g + i] = decode_e4m3(code) * s;
        }
    }
    (out, subn)
}

fn q_int8_groups(row: &[f32], group: usize) -> Vec<f32> {
    let g = if group == 0 { row.len() } else { group };
    let mut out = vec![0f32; row.len()];
    for (gi, chunk) in row.chunks(g).enumerate() {
        let a = amax(chunk);
        let (s, inv) = if a > 0.0 {
            (a / 127.0, 127.0 / a)
        } else {
            (0.0, 0.0)
        };
        for (i, v) in chunk.iter().enumerate() {
            let q = (*v * inv).round().clamp(-127.0, 127.0);
            out[gi * g + i] = q * s;
        }
    }
    out
}

const E2M1: [f32; 8] = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0];

fn e2m1_rtn(v: f32) -> f32 {
    let a = v.abs();
    let mut best = 0f32;
    let mut bd = f32::INFINITY;
    for c in E2M1 {
        let d = (a - c).abs();
        if d < bd {
            bd = d;
            best = c;
        }
    }
    if v < 0.0 {
        -best
    } else {
        best
    }
}

fn ue4m3_round(x: f32) -> f32 {
    if x <= 0.0 || !x.is_finite() {
        return 0.0;
    }
    let c = encode_e4m3(x) & 0x7f;
    decode_e4m3(c)
}

fn q_nvfp4(row: &[f32], global: f32) -> Vec<f32> {
    let mut out = vec![0f32; row.len()];
    for (bi, chunk) in row.chunks(16).enumerate() {
        let a = amax(chunk);
        let sc = ue4m3_round((a * global) / 6.0);
        let eff = if sc > 0.0 { sc / global } else { 0.0 };
        for (i, v) in chunk.iter().enumerate() {
            out[bi * 16 + i] = if eff > 0.0 {
                e2m1_rtn(*v / eff) * eff
            } else {
                0.0
            };
        }
    }
    out
}

struct Stat {
    w_rel: f64,
    y_rel: f64,
    #[allow(dead_code)]
    bytes: f64,
}

fn eval(
    name: &str,
    w: &[f32],
    wq: &[f32],
    x: &[f32],
    n: usize,
    k: usize,
    bits_per_elem: f64,
    scales_per_row: f64,
) -> Stat {
    let mut se = 0f64;
    let mut sw = 0f64;
    for (a, b) in w.iter().zip(wq.iter()) {
        let d = (*a - *b) as f64;
        se += d * d;
        sw += (*a as f64) * (*a as f64);
    }
    let w_rel = (se / sw.max(1e-30)).sqrt();
    let mut ye = 0f64;
    let mut yr = 0f64;
    for r in 0..n {
        let mut a = 0f64;
        let mut b = 0f64;
        for i in 0..k {
            a += (w[r * k + i] * x[i]) as f64;
            b += (wq[r * k + i] * x[i]) as f64;
        }
        ye += (a - b) * (a - b);
        yr += a * a;
    }
    let y_rel = (ye / yr.max(1e-30)).sqrt();
    let bytes = (bits_per_elem / 8.0) + scales_per_row * 4.0 / k as f64;
    eprintln!(
        "    {name:<22} rms(dW)/rms(W) {w_rel:.4e}   rms(dY)/rms(Y) {y_rel:.4e}   bytes/elem {bytes:.4}"
    );
    Stat {
        w_rel,
        y_rel,
        bytes,
    }
}

fn snapshot_dir() -> std::path::PathBuf {
    match std::env::var("NV_GEMMA4_DIR") {
        Ok(d) => std::path::PathBuf::from(d),
        Err(_) => {
            let home = std::env::var("HOME").unwrap();
            let base = std::path::PathBuf::from(home)
                .join(".cache/huggingface/hub/models--nvidia--Gemma-4-31B-IT-NVFP4/snapshots");
            std::fs::read_dir(&base)
                .expect("hub snapshot dir")
                .filter_map(|e| e.ok())
                .map(|e| e.path())
                .find(|p| p.join("config.json").exists())
                .expect("no snapshot with config.json")
        }
    }
}

#[test]
#[ignore]
fn real_gemma4_attn_weight_quant_error_by_format_and_granularity() {
    if std::env::var("NV_GEMMA4_WGPU_TEST").ok().as_deref() != Some("1") {
        eprintln!("skipping: set NV_GEMMA4_WGPU_TEST=1 to run");
        return;
    }
    let dir = snapshot_dir();
    let config = Gemma4Config::from_hf_json_file(&dir.join("config.json")).unwrap();
    let loader = nv_weights::WeightLoader::open_dir(&dir, &Device::Cpu).unwrap();
    eprintln!(
        "config: hidden {} layers {} heads {}",
        config.hidden_size, config.num_hidden_layers, config.num_attention_heads
    );

    let layers: Vec<usize> = std::env::var("NV_FP8_LAYERS")
        .ok()
        .map(|s| s.split(',').filter_map(|t| t.parse().ok()).collect())
        .unwrap_or_else(|| {
            vec![
                0,
                config.num_hidden_layers / 2,
                config.num_hidden_layers - 1,
            ]
        });

    let mut rng = Lcg(0xf8f8);
    for li in layers {
        let p = format!("model.language_model.layers.{li}");
        for proj in ["q_proj", "k_proj", "v_proj", "o_proj"] {
            let name = format!("{p}.self_attn.{proj}.weight");
            if !loader.has(&name) {
                eprintln!("  layer {li} {proj}: absent");
                continue;
            }
            let t = loader.get(&name, DType::BF16).unwrap();
            let shape = t.dims().to_vec();
            let (n_full, k) = (shape[0], shape[1]);
            let n = n_full.min(256);
            let w: Vec<f32> = t
                .narrow(0, 0, n)
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_dtype(DType::F32)
                .unwrap()
                .to_vec1::<f32>()
                .unwrap()
                .iter()
                .map(|v| f(half::bf16::from_f32(*v).to_bits()))
                .collect();
            let x: Vec<f32> = (0..k).map(|_| rng.gauss()).collect();

            let mut rmax = 0f64;
            let mut rmin = f64::INFINITY;
            let mut kurt = 0f64;
            for r in 0..n {
                let row = &w[r * k..(r + 1) * k];
                let a = amax(row);
                let m2 = row.iter().map(|v| (*v as f64) * (*v as f64)).sum::<f64>() / k as f64;
                let m4 = row
                    .iter()
                    .map(|v| {
                        let s = (*v as f64) * (*v as f64);
                        s * s
                    })
                    .sum::<f64>()
                    / k as f64;
                let ratio = (a as f64) / m2.sqrt().max(1e-30);
                rmax = rmax.max(ratio);
                rmin = rmin.min(ratio);
                kurt += m4 / (m2 * m2).max(1e-30);
            }
            eprintln!(
                "  layer {li} {proj} [{n_full} x {k}] (sampling {n} rows): amax/rms per row min {rmin:.2} max {rmax:.2}, kurtosis {:.2}",
                kurt / n as f64
            );

            let mut wq = vec![0f32; n * k];
            let mut subn_total = 0usize;
            for r in 0..n {
                let (q, s) = q_e4m3_groups(&w[r * k..(r + 1) * k], 0);
                wq[r * k..(r + 1) * k].copy_from_slice(&q);
                subn_total += s;
            }
            eval("e4m3 per-row", &w, &wq, &x, n, k, 8.0, 1.0);
            eprintln!(
                "      e4m3 per-row subnormal codes: {subn_total} / {} ({:.4}%)",
                n * k,
                100.0 * subn_total as f64 / (n * k) as f64
            );

            for g in [128usize, 64, 32, 16] {
                for r in 0..n {
                    let (q, _) = q_e4m3_groups(&w[r * k..(r + 1) * k], g);
                    wq[r * k..(r + 1) * k].copy_from_slice(&q);
                }
                eval(
                    &format!("e4m3 group={g}"),
                    &w,
                    &wq,
                    &x,
                    n,
                    k,
                    8.0,
                    (k / g) as f64,
                );
            }

            for g in [0usize, 128, 64, 32] {
                for r in 0..n {
                    let q = q_int8_groups(&w[r * k..(r + 1) * k], g);
                    wq[r * k..(r + 1) * k].copy_from_slice(&q);
                }
                let sc = k.checked_div(g).map_or(1.0, |v| v as f64);
                eval(&format!("int8 group={g}"), &w, &wq, &x, n, k, 8.0, sc);
            }

            let gmax = amax(&w);
            let global = if gmax > 0.0 {
                (448.0 * 6.0) / gmax
            } else {
                1.0
            };
            for r in 0..n {
                let q = q_nvfp4(&w[r * k..(r + 1) * k], global);
                wq[r * k..(r + 1) * k].copy_from_slice(&q);
            }
            eval(
                "nvfp4 (e2m1/16)",
                &w,
                &wq,
                &x,
                n,
                k,
                4.0,
                (k / 16) as f64 / 4.0,
            );
        }
    }
}

#[test]
fn same_sign_random_weights_understate_fp8_error() {
    let n = 64usize;
    let k = 4096usize;
    let mut rng = Lcg(7);
    let mut same: Vec<f32> = Vec::with_capacity(n * k);
    let mut zero: Vec<f32> = Vec::with_capacity(n * k);
    for _ in 0..n * k {
        let u = rng.next() + 0.5;
        same.push(-0.1 * u);
        zero.push(0.2 * rng.next());
    }
    let xs: Vec<f32> = (0..k).map(|_| -0.1 * (rng.next() + 0.5)).collect();
    let xz: Vec<f32> = (0..k).map(|_| 0.2 * rng.next()).collect();

    let quant = |w: &[f32]| -> Vec<f32> {
        let mut out = vec![0f32; w.len()];
        for r in 0..n {
            let (q, _) = q_e4m3_groups(&w[r * k..(r + 1) * k], 0);
            out[r * k..(r + 1) * k].copy_from_slice(&q);
        }
        out
    };
    let a = eval(
        "same-sign  (old rng)",
        &same,
        &quant(&same),
        &xs,
        n,
        k,
        8.0,
        1.0,
    );
    let b = eval(
        "zero-mean  (honest)",
        &zero,
        &quant(&zero),
        &xz,
        n,
        k,
        8.0,
        1.0,
    );
    eprintln!(
        "weight-space error is comparable ({:.3e} vs {:.3e}) but OUTPUT-space error differs {:.1}x",
        a.w_rel,
        b.w_rel,
        b.y_rel / a.y_rel
    );
    assert!(
        b.y_rel > 5.0 * a.y_rel,
        "expected the same-sign generator to hide output error; got {:.3e} vs {:.3e}",
        a.y_rel,
        b.y_rel
    );
}
