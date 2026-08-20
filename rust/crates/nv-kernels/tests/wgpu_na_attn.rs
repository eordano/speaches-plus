#![cfg(feature = "wgpu")]

mod common;
use common::lcg;
use common::require;
use common::rnd_f;
use nv_kernels::wgpu_backend::device::WgpuContext;
use nv_kernels::wgpu_backend::{dispatch, na_attn};

fn ctx(test: &str) -> Option<&'static WgpuContext> {
    match WgpuContext::shared() {
        Ok(c) => {
            eprintln!("{test}: {}", c.summary());
            Some(c)
        }
        Err(e) => {
            if require() {
                panic!("{test}: no wgpu adapter: {e}");
            }
            eprintln!("{test}: SKIP no wgpu adapter: {e}");
            None
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct FdParams {
    n_heads: u32,
    n_kv: u32,
    head_dim: u32,
    total: u32,
    start: u32,
    splits: u32,
    ring: u32,
    out_bf16: u32,
    scaling: f32,
    pad0: u32,
    fused: u32,
    pad2: u32,
    m_rows: u32,
    window: u32,
    pad3: u32,
    pad4: u32,
}

fn e4m3_decode(b: u8) -> f32 {
    let b = b as u32;
    if (b & 127) == 127 {
        return f32::NAN;
    }
    let e = (b >> 3) & 15;
    let m = b & 7;
    let mag = if e == 0 {
        m as f32 * 0.001953125
    } else {
        f32::from_bits(((e + 120) << 23) | (m << 20))
    };
    if b & 128 != 0 {
        -mag
    } else {
        mag
    }
}

fn bf16_of(x: f32) -> f32 {
    half::bf16::from_f32(x).to_f32()
}

struct Case {
    mr: usize,
    total: usize,
    window: usize,
    hd: usize,
}

#[allow(clippy::too_many_arguments)]
fn oracle_f64(
    q: &[f32],
    kb: &[u8],
    vb: &[u8],
    ks: &[f32],
    vs: &[f32],
    n_heads: usize,
    nkv: usize,
    hd: usize,
    c: &Case,
    scaling: f32,
) -> Vec<f64> {
    let group = n_heads / nkv;
    let mut out = vec![0f64; c.mr * n_heads * hd];
    for qi in 0..c.mr {
        for h in 0..n_heads {
            let kvh = h / group;
            let tr = c.total - (c.mr - 1 - qi);
            let st = if c.window > 0 && tr > c.window {
                tr - c.window
            } else {
                0
            };
            let mut scores = Vec::new();
            for p in st..tr {
                let base = (p * nkv + kvh) * hd;
                let mut s = 0f64;
                for d in 0..hd {
                    s += q[(qi * n_heads + h) * hd + d] as f64 * e4m3_decode(kb[base + d]) as f64;
                }
                scores.push(s * ks[p * nkv + kvh] as f64 * scaling as f64);
            }
            let m = scores.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let mut l = 0f64;
            let mut acc = vec![0f64; hd];
            for (i, s) in scores.iter().enumerate() {
                let p = st + i;
                let w = (s - m).exp();
                l += w;
                let base = (p * nkv + kvh) * hd;
                let wv = w * vs[p * nkv + kvh] as f64;
                for d in 0..hd {
                    acc[d] += wv * e4m3_decode(vb[base + d]) as f64;
                }
            }
            for d in 0..hd {
                out[(qi * n_heads + h) * hd + d] = acc[d] / l;
            }
        }
    }
    out
}

#[allow(clippy::too_many_arguments)]
fn wgsl_sim_f32(
    q: &[f32],
    kb: &[u8],
    vb: &[u8],
    ks: &[f32],
    vs: &[f32],
    n_heads: usize,
    nkv: usize,
    hd: usize,
    c: &Case,
    scaling: f32,
) -> Vec<f32> {
    let group = n_heads / nkv;
    let mut out = vec![0f32; c.mr * n_heads * hd];
    for qi in 0..c.mr {
        for h in 0..n_heads {
            let kvh = h / group;
            let tr = c.total - (c.mr - 1 - qi);
            let st = if c.window > 0 && tr > c.window {
                tr - c.window
            } else {
                0
            };
            let mut m = f32::NEG_INFINITY;
            let mut l = 0f32;
            let mut acc = vec![0f32; hd];
            for p in st..tr {
                let base = (p * nkv + kvh) * hd;
                let mut s = 0f32;
                for d in 0..hd {
                    s += q[(qi * n_heads + h) * hd + d] * e4m3_decode(kb[base + d]);
                }
                let score = (s * ks[p * nkv + kvh]) * scaling;
                let m_new = m.max(score);
                let corr = ((m - m_new) * std::f32::consts::LOG2_E).exp2();
                let w = ((score - m_new) * std::f32::consts::LOG2_E).exp2();
                l = l * corr + w;
                let wv = w * vs[p * nkv + kvh];
                for d in 0..hd {
                    acc[d] = wv.mul_add(e4m3_decode(vb[base + d]), acc[d] * corr);
                }
                m = m_new;
            }
            let inv = 1.0 / l;
            for d in 0..hd {
                out[(qi * n_heads + h) * hd + d] = bf16_of(acc[d] * inv);
            }
        }
    }
    out
}

fn max_rel(a: &[f64], b: &[f64]) -> f64 {
    let mut worst = 0f64;
    for (x, y) in a.iter().zip(b) {
        let denom = y.abs().max(1e-2);
        let rel = (x - y).abs() / denom;
        if rel > worst {
            worst = rel;
        }
    }
    worst
}

#[test]
#[cfg_attr(
    not(target_os = "macos"),
    ignore = "na_attn is an MSL passthrough kernel (MetalPerformancePrimitives tensor_ops); \
              na::supported() requires Backend::Metal, which no non-macOS adapter can provide"
)]
fn na_attn_prefill_matches_flash_reference() {
    let Some(ctx) = ctx("na_attn_prefill_matches_flash_reference") else {
        return;
    };
    let (n_heads, nkv) = (8usize, 2usize);
    let scaling = 1.0f32;
    let cases = [
        Case {
            mr: 128,
            total: 128,
            window: 0,
            hd: 256,
        },
        Case {
            mr: 128,
            total: 512,
            window: 0,
            hd: 256,
        },
        Case {
            mr: 128,
            total: 512,
            window: 512,
            hd: 256,
        },
        Case {
            mr: 10,
            total: 138,
            window: 512,
            hd: 256,
        },
        Case {
            mr: 128,
            total: 128,
            window: 0,
            hd: 512,
        },
        Case {
            mr: 128,
            total: 512,
            window: 0,
            hd: 512,
        },
        Case {
            mr: 10,
            total: 138,
            window: 0,
            hd: 512,
        },
    ];
    let mut state = 0x5eed_u64;
    for c in &cases {
        let hd = c.hd;
        let pipeline = match if hd == 512 {
            na_attn::pipeline_g(ctx)
        } else {
            na_attn::pipeline(ctx)
        } {
            Ok(p) => p,
            Err(e) => {
                if require() {
                    panic!("na_attn pipeline unavailable: {e}");
                }
                eprintln!("SKIP: na_attn pipeline unavailable: {e}");
                return;
            }
        };
        let q: Vec<f32> = (0..c.mr * n_heads * hd)
            .map(|_| bf16_of(rnd_f(&mut state) * 2.0))
            .collect();
        let mut kb = vec![0u8; c.total * nkv * hd];
        let mut vb = vec![0u8; c.total * nkv * hd];
        for b in kb.iter_mut().chain(vb.iter_mut()) {
            let mut byte = (lcg(&mut state) >> 32) as u8;
            if (byte & 127) == 127 {
                byte &= 0xef;
            }
            *b = byte;
        }
        let ks: Vec<f32> = (0..c.total * nkv)
            .map(|_| 0.005 + 0.045 * (rnd_f(&mut state) * 0.5 + 0.5))
            .collect();
        let vs: Vec<f32> = (0..c.total * nkv)
            .map(|_| 0.005 + 0.045 * (rnd_f(&mut state) * 0.5 + 0.5))
            .collect();

        let params = FdParams {
            n_heads: n_heads as u32,
            n_kv: nkv as u32,
            head_dim: hd as u32,
            total: c.total as u32,
            start: 0,
            splits: 16,
            ring: 0,
            out_bf16: 1,
            scaling,
            pad0: 0,
            fused: 0,
            pad2: 0,
            m_rows: c.mr as u32,
            window: c.window as u32,
            pad3: 0,
            pad4: 0,
        };
        let q_buf = dispatch::storage_from_slice(ctx, "naat.q", &q);
        let k_buf =
            dispatch::storage_from_slice(ctx, "naat.k", bytemuck::cast_slice::<u8, u32>(&kb));
        let v_buf =
            dispatch::storage_from_slice(ctx, "naat.v", bytemuck::cast_slice::<u8, u32>(&vb));
        let ks_buf = dispatch::storage_from_slice(ctx, "naat.ks", &ks);
        let vs_buf = dispatch::storage_from_slice(ctx, "naat.vs", &vs);
        let out_words = c.mr * n_heads * hd / 2;
        let out_buf = dispatch::storage_zeroed(ctx, "naat.out", (out_words * 4) as u64);
        let p_buf = dispatch::uniform_from(ctx, "naat.fd", &params);
        dispatch::dispatch(
            ctx,
            &pipeline,
            &[
                (0, &q_buf),
                (1, &k_buf),
                (2, &v_buf),
                (3, &ks_buf),
                (4, &vs_buf),
                (5, &out_buf),
                (6, &p_buf),
            ],
            na_attn::grid(n_heads as u32, c.mr as u32),
        )
        .unwrap();
        let words: Vec<u32> = dispatch::read_back(ctx, &out_buf, out_words).unwrap();
        let mut got = vec![0f32; c.mr * n_heads * hd];
        for (i, w) in words.iter().enumerate() {
            got[2 * i] = f32::from_bits(w << 16);
            got[2 * i + 1] = f32::from_bits(w & 0xffff_0000);
        }

        let oracle = oracle_f64(&q, &kb, &vb, &ks, &vs, n_heads, nkv, hd, c, scaling);
        let sim = wgsl_sim_f32(&q, &kb, &vb, &ks, &vs, n_heads, nkv, hd, c, scaling);
        let got64: Vec<f64> = got.iter().map(|&x| x as f64).collect();
        let sim64: Vec<f64> = sim.iter().map(|&x| x as f64).collect();
        let rel_na = max_rel(&got64, &oracle);
        let rel_sim = max_rel(&sim64, &oracle);
        let rel_cross = max_rel(&got64, &sim64);
        eprintln!(
            "case hd={} mr={} total={} window={}: max rel vs f64 oracle: na_attn {:.3e}, wgsl-order sim {:.3e}; na_attn vs sim {:.3e}",
            hd, c.mr, c.total, c.window, rel_na, rel_sim, rel_cross
        );
        assert!(
            got.iter().all(|x| x.is_finite()),
            "non-finite outputs (hd={hd} mr={} total={} window={})",
            c.mr,
            c.total,
            c.window
        );
        assert!(
            rel_na < 0.03,
            "na_attn diverges from f64 oracle: {rel_na:.3e}"
        );
        assert!(
            rel_na < rel_sim.max(0.004) * 4.0,
            "na_attn error {rel_na:.3e} far exceeds wgsl-order f32 band {rel_sim:.3e}"
        );
    }
}
