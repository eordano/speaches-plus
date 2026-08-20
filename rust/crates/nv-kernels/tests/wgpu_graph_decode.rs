#![cfg(feature = "wgpu")]
#![allow(clippy::too_many_arguments)]

mod common;
use common::require;
use common::to_bf16;
use half::bf16;
use nv_kernels::wgpu_backend::device::WgpuContext;
use nv_kernels::wgpu_backend::kernels::graph_decode as gd;
use nv_kernels::wgpu_backend::WgpuError;

fn ctx_or_skip(test: &str) -> Option<&'static WgpuContext> {
    match WgpuContext::shared() {
        Ok(ctx) => {
            eprintln!("{test}: {}", ctx.summary());
            let st = ctx.qualify();
            if !st.qualified {
                if require() {
                    panic!("adapter not qualified: {:?}", st.reason);
                }
                eprintln!("{test}: SKIP adapter not qualified: {:?}", st.reason);
                return None;
            }
            Some(ctx)
        }
        Err(e) => {
            if require() {
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

fn lcg(seed: u64) -> impl FnMut() -> f32 {
    let mut s = seed;
    move || {
        s = s
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((s >> 33) as f32 / (1u64 << 31) as f32) * 2.0 - 1.0
    }
}

fn from_bf16(b: u16) -> f32 {
    f32::from_bits((b as u32) << 16)
}

fn bf16_ordered(b: u16) -> i64 {
    if b & 0x8000 != 0 {
        -((b & 0x7fff) as i64)
    } else {
        b as i64
    }
}

#[test]
fn incr_pos_matches_cpu_oracle() {
    let Some(ctx) = ctx_or_skip("incr_pos") else {
        return;
    };
    for start in [0i32, 5, 1023, -1] {
        let mut p = vec![start];
        gd::incr_pos(ctx, &mut p).unwrap();
        assert_eq!(p[0], start + 1, "incr_pos({start})");
    }
    eprintln!("incr_pos: 4/4 starts match oracle exactly");
}

#[test]
fn incr_pos_rope_matches_cpu_oracle() {
    let Some(ctx) = ctx_or_skip("incr_pos_rope") else {
        return;
    };
    for start in [0i32, 5, 1023] {
        let mut p = vec![start];
        let mut r = vec![-1i32];
        gd::incr_pos_rope(ctx, &mut p, &mut r).unwrap();
        assert_eq!(r[0], start, "incr_pos_rope rope_pos({start})");
        assert_eq!(p[0], start + 1, "incr_pos_rope pos({start})");
    }
    eprintln!("incr_pos_rope: 3/3 starts match oracle exactly");
}

#[test]
fn token_map_u32_matches_cpu_oracle() {
    let Some(ctx) = ctx_or_skip("token_map_u32") else {
        return;
    };
    let map: Vec<u32> = (0..1024u32).map(|i| i * 3 + 11).collect();
    for idx in [0u32, 7, 1023] {
        let mut out = vec![0u32];
        gd::token_map_u32(ctx, &map, &[idx], &mut out).unwrap();
        assert_eq!(out[0], map[idx as usize]);
    }
    let mut out = vec![0u32];
    let e = gd::token_map_u32(ctx, &map, &[4096], &mut out).unwrap_err();
    assert!(matches!(e, WgpuError::Shape(_)), "{e}");
    eprintln!("token_map_u32: 3/3 lookups exact, out-of-range rejected");
}

#[test]
fn cast_roundtrip_matches_cpu_oracle() {
    let Some(ctx) = ctx_or_skip("casts") else {
        return;
    };
    let mut rnd = lcg(0xbeef01);
    let xf: Vec<f32> = (0..4097).map(|_| rnd() * 30.0).collect();
    let x = to_bf16(&xf);
    let n = x.len();

    let mut y = vec![0f32; n];
    gd::cast_bf16_f32(ctx, &x, &mut y, n).unwrap();
    let bad = y
        .iter()
        .zip(x.iter())
        .filter(|(a, b)| a.to_bits() != from_bf16(**b).to_bits())
        .count();
    assert_eq!(bad, 0, "cast_bf16_f32 vs oracle");

    let mut back = vec![0u16; n];
    gd::cast_f32_bf16(ctx, &y, &mut back, n).unwrap();
    assert_eq!(back, x, "cast_f32_bf16 must invert cast_bf16_f32 exactly");

    let mut scaled = vec![0f32; n];
    gd::cast_scale_bf16_f32(ctx, &x, &mut scaled, 0.375, n).unwrap();
    let bad = scaled
        .iter()
        .zip(x.iter())
        .filter(|(a, b)| a.to_bits() != (from_bf16(**b) * 0.375).to_bits())
        .count();
    assert_eq!(bad, 0, "cast_scale_bf16_f32 vs oracle");
    eprintln!("casts: 0/{n} mismatches against the CPU oracle for all three casts");
}

#[test]
fn add_scale_f32_matches_cpu_oracle() {
    let Some(ctx) = ctx_or_skip("add_scale_f32") else {
        return;
    };
    let mut rnd = lcg(0xbeef02);
    let n = 5000usize;
    let a: Vec<f32> = (0..n).map(|_| rnd() * 4.0).collect();
    let b: Vec<f32> = (0..n).map(|_| rnd() * 4.0).collect();
    let mut y = vec![0f32; n];
    gd::add_scale_f32(ctx, &a, &b, &mut y, 0.625, n).unwrap();
    let bad = (0..n)
        .filter(|i| y[*i].to_bits() != ((a[*i] + b[*i]) * 0.625).to_bits())
        .count();
    assert_eq!(bad, 0, "add_scale_f32 vs oracle");
    eprintln!("add_scale_f32: 0/{n} mismatches against the CPU oracle");
}

#[test]
fn write_kv_f32_matches_cpu_oracle() {
    let Some(ctx) = ctx_or_skip("write_kv_f32") else {
        return;
    };
    let nkv = 3usize;
    let hd = 70usize;
    let slots = 8usize;
    let mut rnd = lcg(0xbeef03);
    let sk: Vec<f32> = (0..nkv * hd).map(|_| rnd()).collect();
    let sv: Vec<f32> = (0..nkv * hd).map(|_| rnd()).collect();
    let base: Vec<f32> = (0..slots * nkv * hd).map(|_| rnd()).collect();

    for pos in [0i32, 1, 5] {
        let mut ck = base.clone();
        let mut cv = base.clone();
        gd::write_kv_f32(ctx, &sk, &sv, &mut ck, &mut cv, &[pos], nkv, hd).unwrap();
        let mut ok = base.clone();
        let mut ov = base.clone();
        let slot = pos - 1;
        if slot >= 0 {
            for h in 0..nkv {
                for d in 0..hd {
                    ok[(slot as usize * nkv + h) * hd + d] = sk[h * hd + d];
                    ov[(slot as usize * nkv + h) * hd + d] = sv[h * hd + d];
                }
            }
        }
        let bad = ck
            .iter()
            .zip(ok.iter())
            .filter(|(a, b)| a.to_bits() != b.to_bits())
            .count()
            + cv.iter()
                .zip(ov.iter())
                .filter(|(a, b)| a.to_bits() != b.to_bits())
                .count();
        assert_eq!(bad, 0, "write_kv_f32 pos={pos} vs oracle");
    }
    eprintln!("write_kv_f32: 0 mismatches against the CPU oracle for pos in 0,1,5");
}

#[test]
fn multi_zero_bf16_keeps_the_sub_eight_tail() {
    let Some(ctx) = ctx_or_skip("multi_zero_bf16") else {
        return;
    };
    let lens = [64usize, 8, 13, 1];
    let mut owned: Vec<Vec<u16>> = lens
        .iter()
        .map(|l| (0..*l).map(|i| (i + 1) as u16).collect())
        .collect();
    let before = owned.clone();
    {
        let mut views: Vec<&mut [u16]> = owned.iter_mut().map(|v| v.as_mut_slice()).collect();
        gd::multi_zero_bf16(ctx, &mut views).unwrap();
    }
    for (i, (after, orig)) in owned.iter().zip(before.iter()).enumerate() {
        let zeroed = (lens[i] / 8) * 8;
        for (j, v) in after.iter().enumerate() {
            if j < zeroed {
                assert_eq!(*v, 0, "buffer {i} element {j} should be zeroed");
            } else {
                assert_eq!(*v, orig[j], "buffer {i} tail element {j} must be untouched");
            }
        }
        eprintln!(
            "multi_zero_bf16 buf{i} len={}: zeroed {zeroed}, tail {} preserved",
            lens[i],
            lens[i] - zeroed
        );
    }
}

#[test]
fn rms_family_matches_f64_oracle() {
    let Some(ctx) = ctx_or_skip("rms_family") else {
        return;
    };
    let eps = 1e-6f32;
    let rows = 4usize;
    let dim = 512usize;
    let mut rnd = lcg(0xbeef04);
    let xf: Vec<f32> = (0..rows * dim).map(|_| rnd() * 2.0).collect();
    let wf: Vec<f32> = (0..dim).map(|_| 1.0 + rnd() * 0.3).collect();
    let x = to_bf16(&xf);
    let weight = to_bf16(&wf);

    let mut inv = vec![0f64; rows];
    for r in 0..rows {
        let mut s = 0f64;
        for d in 0..dim {
            let v = from_bf16(x[r * dim + d]) as f64;
            s += v * v;
        }
        inv[r] = 1.0 / (s / dim as f64 + eps as f64).sqrt();
    }

    let mut y = vec![0f32; rows * dim];
    gd::rms_no_weight_bf16_f32(ctx, &x, &mut y, rows, dim, eps).unwrap();
    let mut max_rel = 0f64;
    for r in 0..rows {
        for d in 0..dim {
            let want = from_bf16(x[r * dim + d]) as f64 * inv[r];
            if want.abs() > 1e-6 {
                max_rel = max_rel.max(((y[r * dim + d] as f64 - want) / want).abs());
            }
        }
    }
    eprintln!("rms_no_weight_bf16_f32: max_rel={max_rel:e} vs f64 oracle");
    assert!(max_rel < 1e-5, "rms_no_weight_bf16_f32 max_rel {max_rel:e}");

    let mut yw = vec![0f32; rows * dim];
    gd::rmsnorm_bf16w_f32out(ctx, &x, &weight, &mut yw, rows, dim, eps).unwrap();
    let mut max_rel_w = 0f64;
    for r in 0..rows {
        for d in 0..dim {
            let want = from_bf16(x[r * dim + d]) as f64 * inv[r] * from_bf16(weight[d]) as f64;
            if want.abs() > 1e-6 {
                max_rel_w = max_rel_w.max(((yw[r * dim + d] as f64 - want) / want).abs());
            }
        }
    }
    eprintln!("rmsnorm_bf16w_f32out: max_rel={max_rel_w:e} vs f64 oracle");
    assert!(
        max_rel_w < 1e-5,
        "rmsnorm_bf16w_f32out max_rel {max_rel_w:e}"
    );

    let mut rstd = vec![0f32; rows];
    gd::rstd_bf16(ctx, &x, &mut rstd, rows, dim, eps).unwrap();
    let mut max_rel_r = 0f64;
    for r in 0..rows {
        max_rel_r = max_rel_r.max(((rstd[r] as f64 - inv[r]) / inv[r]).abs());
    }
    eprintln!("rstd_bf16: max_rel={max_rel_r:e} vs f64 oracle");
    assert!(max_rel_r < 1e-6, "rstd_bf16 max_rel {max_rel_r:e}");

    let mut applied = vec![0u16; rows * dim];
    gd::rms_apply_bf16(ctx, &x, &weight, &rstd, &mut applied, rows * dim, dim).unwrap();
    let mut bad = 0usize;
    for r in 0..rows {
        for d in 0..dim {
            let want = bf16::from_f32(from_bf16(x[r * dim + d]) * rstd[r] * from_bf16(weight[d]))
                .to_bits();
            if applied[r * dim + d] != want {
                bad += 1;
            }
        }
    }
    eprintln!(
        "rms_apply_bf16: {bad}/{} bf16 words differ from the f32 oracle",
        rows * dim
    );
    assert_eq!(bad, 0, "rms_apply_bf16 vs oracle");
}

#[test]
fn argmax_bf16_matches_cpu_oracle() {
    let Some(ctx) = ctx_or_skip("argmax_bf16") else {
        return;
    };
    let vocab = 65_536usize;
    let mut rnd = lcg(0xbeef05);
    let mut lf: Vec<f32> = (0..vocab).map(|_| rnd() * 5.0).collect();
    lf[40_000] = 7.5;
    lf[9] = 7.5;
    let logits = to_bf16(&lf);

    let mut best = f32::NEG_INFINITY;
    let mut best_i = 0usize;
    for (i, w) in logits.iter().enumerate() {
        let v = from_bf16(*w);
        if v > best {
            best = v;
            best_i = i;
        }
    }

    let ring_mask = 255i32;
    let mut ring = vec![0u32; 256];
    let mut token = vec![0u32];
    gd::argmax_bf16(
        ctx,
        &logits,
        &[10],
        &mut token,
        Some(&mut ring),
        ring_mask,
        vocab,
    )
    .unwrap();
    eprintln!("argmax_bf16: token={} oracle={best_i}", token[0]);
    assert_eq!(token[0] as usize, best_i, "argmax_bf16 token vs oracle");
    assert_eq!(ring[9], best_i as u32, "argmax_bf16 ring slot (pos-1)&mask");

    let parts = gd::argmax_bf16_part_count();
    let mut pv = vec![0f32; parts];
    let mut pi = vec![0i32; parts];
    gd::argmax_bf16_parts(ctx, &logits, &mut pv, &mut pi, vocab).unwrap();
    let overall = pi
        .iter()
        .zip(pv.iter())
        .fold((f32::NEG_INFINITY, i32::MAX), |acc, (i, v)| {
            if *v > acc.0 || (*v == acc.0 && *i < acc.1) {
                (*v, *i)
            } else {
                acc
            }
        });
    assert_eq!(
        overall.1 as usize, best_i,
        "argmax_bf16_parts merge vs oracle"
    );
    eprintln!(
        "argmax_bf16_parts: merged index {} matches oracle",
        overall.1
    );
}

#[test]
fn argmax_f32_rows_matches_cpu_oracle() {
    let Some(ctx) = ctx_or_skip("argmax_f32_rows") else {
        return;
    };
    let rows = 5usize;
    let n = 40_000usize;
    let mut rnd = lcg(0xbeef06);
    let mut logits: Vec<f32> = (0..rows * n).map(|_| rnd() * 6.0).collect();
    logits[n + 3] = f32::NAN;
    logits[n + 900] = 12.0;
    logits[2 * n + 1] = f32::INFINITY;
    logits[2 * n + 77] = 11.0;
    for i in 0..n {
        logits[3 * n + i] = f32::NEG_INFINITY;
    }

    let mut want = vec![0u32; rows];
    for r in 0..rows {
        let mut best = f32::NEG_INFINITY;
        let mut best_i: Option<usize> = None;
        for i in 0..n {
            let v = logits[r * n + i];
            if v.is_finite() && (best_i.is_none() || v > best) {
                best = v;
                best_i = Some(i);
            }
        }
        want[r] = best_i.unwrap_or(0) as u32;
    }

    let mut got = vec![0u32; rows];
    gd::argmax_f32_rows(ctx, &logits, &mut got, rows, n).unwrap();
    eprintln!("argmax_f32_rows: got={got:?} oracle={want:?}");
    assert_eq!(got, want, "argmax_f32_rows vs oracle");
}

#[test]
fn attn_decode_dev_f32_matches_f64_oracle() {
    let Some(ctx) = ctx_or_skip("attn_decode_dev_f32") else {
        return;
    };
    let (nh, nkv, hd) = (4usize, 2usize, 64usize);
    let slots = 32usize;
    let pos = 20i32;
    let window = 8usize;
    let mut rnd = lcg(0xbeef07);
    let q: Vec<f32> = (0..nh * hd).map(|_| rnd()).collect();
    let k: Vec<f32> = (0..slots * nkv * hd).map(|_| rnd()).collect();
    let v: Vec<f32> = (0..slots * nkv * hd).map(|_| rnd()).collect();

    let mut out = vec![0f32; nh * hd];
    gd::attn_decode_dev_f32(ctx, &q, &k, &v, &mut out, &[pos], nh, nkv, hd, window).unwrap();

    let total = pos as usize;
    let start = total - window;
    let group = nh / nkv;
    let mut max_abs = 0f64;
    for h in 0..nh {
        let kvh = h / group;
        let mut scores = Vec::new();
        for p in start..total {
            let mut s = 0f64;
            for d in 0..hd {
                s += q[h * hd + d] as f64 * k[(p * nkv + kvh) * hd + d] as f64;
            }
            scores.push(s);
        }
        let m = scores.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let ws: Vec<f64> = scores.iter().map(|s| (s - m).exp()).collect();
        let l: f64 = ws.iter().sum();
        for d in 0..hd {
            let mut acc = 0f64;
            for (i, p) in (start..total).enumerate() {
                acc += ws[i] * v[(p * nkv + kvh) * hd + d] as f64;
            }
            max_abs = max_abs.max((out[h * hd + d] as f64 - acc / l).abs());
        }
    }
    eprintln!("attn_decode_dev_f32: max_abs={max_abs:e} vs f64 oracle");
    assert!(max_abs < 2e-6, "attn_decode_dev_f32 max_abs {max_abs:e}");

    let mut zero = vec![1f32; nh * hd];
    gd::attn_decode_dev_f32(ctx, &q, &k, &v, &mut zero, &[0], nh, nkv, hd, 0).unwrap();
    assert!(
        zero.iter().all(|v| *v == 0.0),
        "attn_decode_dev_f32 with pos=0 must write zeros like the CUDA kernel"
    );
}

#[test]
fn qkv_prep_matches_f64_oracle() {
    let Some(ctx) = ctx_or_skip("qkv_prep") else {
        return;
    };
    let (nh, nkv, hd) = (4usize, 2usize, 64usize);
    let half = hd / 2;
    let eps = 1e-6f32;
    let rope_pos = 12i32;
    let delta = 0i32;
    let slots = 16usize;
    let heads = nh + 2 * nkv;
    let mut rnd = lcg(0xbeef08);
    let xf: Vec<f32> = (0..heads * hd).map(|_| rnd() * 2.0).collect();
    let qwf: Vec<f32> = (0..hd).map(|_| 1.0 + rnd() * 0.2).collect();
    let kwf: Vec<f32> = (0..hd).map(|_| 1.0 + rnd() * 0.2).collect();
    let qkv = to_bf16(&xf);
    let qw = to_bf16(&qwf);
    let kw = to_bf16(&kwf);
    let cos_tbl: Vec<f32> = (0..64 * half).map(|i| ((i as f32) * 0.01).cos()).collect();
    let sin_tbl: Vec<f32> = (0..64 * half).map(|i| ((i as f32) * 0.01).sin()).collect();

    let mut q_out = vec![0f32; nh * hd];
    let mut kc = vec![0u16; slots * nkv * hd];
    let mut vc = vec![0u16; slots * nkv * hd];
    let rp_host = [rope_pos];
    let cp_host = [rope_pos + 1];
    gd::qkv_prep(
        ctx,
        &qkv,
        &qw,
        Some(&kw[..]),
        &cos_tbl,
        &sin_tbl,
        &rp_host,
        Some(&cp_host[..]),
        delta,
        &mut q_out,
        Some(&mut kc),
        Some(&mut vc),
        nh,
        nkv,
        hd,
        eps,
    )
    .unwrap();

    let mut max_abs = 0f64;
    for h in 0..nh {
        let mut s = 0f64;
        for d in 0..hd {
            let v = from_bf16(qkv[h * hd + d]) as f64;
            s += v * v;
        }
        let inv = 1.0 / (s / hd as f64 + eps as f64).sqrt();
        let ns: Vec<f64> = (0..hd)
            .map(|d| from_bf16(qkv[h * hd + d]) as f64 * inv * from_bf16(qw[d]) as f64)
            .collect();
        let cb = rope_pos as usize * half;
        for d in 0..hd {
            let i = if d < half { d } else { d - half };
            let a = ns[i];
            let b = ns[i + half];
            let c = cos_tbl[cb + i] as f64;
            let sn = sin_tbl[cb + i] as f64;
            let want = if d < half {
                a * c - b * sn
            } else {
                a * sn + b * c
            };
            max_abs = max_abs.max((q_out[h * hd + d] as f64 - want).abs());
        }
    }
    eprintln!("qkv_prep q: max_abs={max_abs:e} vs f64 oracle");
    assert!(max_abs < 1e-5, "qkv_prep q max_abs {max_abs:e}");

    let slot = rope_pos as usize;
    let touched = kc[(slot * nkv) * hd..(slot * nkv + nkv) * hd]
        .iter()
        .filter(|v| **v != 0)
        .count();
    assert!(touched > 0, "qkv_prep must have written the k cache slot");
    let untouched = kc[..(slot * nkv) * hd].iter().filter(|v| **v != 0).count();
    assert_eq!(untouched, 0, "qkv_prep must not touch earlier cache slots");
    eprintln!("qkv_prep kcache: slot {slot} written ({touched} nonzero), earlier slots untouched");
}

#[test]
fn gelu_mul_bf16f32_matches_f64_oracle() {
    let Some(ctx) = ctx_or_skip("gelu_mul_bf16f32") else {
        return;
    };
    let n = 4096usize;
    let mut rnd = lcg(0xbeef09);
    let gf: Vec<f32> = (0..n).map(|i| rnd() * (1.0 + (i % 11) as f32)).collect();
    let gate = to_bf16(&gf);
    let pli: Vec<f32> = (0..n).map(|_| rnd() * 2.0).collect();
    let mut y = vec![0u16; n];
    gd::gelu_mul_bf16f32(ctx, &gate, &pli, &mut y, n).unwrap();

    let mut worst = 0i64;
    let mut worst_at = 0usize;
    for i in 0..n {
        let g = from_bf16(gate[i]) as f64;
        let inner = 0.797_884_560_802_865_4 * (g + 0.044715 * g * g * g);
        let t = inner.tanh() as f32;
        let one_plus_t = (1.0f32 + t) as f64;
        let want = bf16::from_f64(0.5 * g * one_plus_t * pli[i] as f64).to_bits();
        let d = (bf16_ordered(want) - bf16_ordered(y[i])).abs();
        if d > worst {
            worst = d;
            worst_at = i;
        }
    }
    eprintln!(
        "gelu_mul_bf16f32: max bf16 ulp vs f32-faithful tanh oracle = {worst} (at i={worst_at}, g={})",
        from_bf16(gate[worst_at])
    );
    assert_eq!(worst, 0, "gelu_mul_bf16f32 max bf16 ulp {worst}");
}

#[test]
fn shape_errors_are_reported() {
    let Some(ctx) = ctx_or_skip("shape_errors") else {
        return;
    };
    let mut y = vec![0f32; 4];
    let e = gd::cast_bf16_f32(ctx, &[0u16; 3], &mut y, 4).unwrap_err();
    assert!(matches!(e, WgpuError::Shape(_)), "{e}");

    let mut q = vec![0f32; 8];
    let e = gd::qkv_prep(
        ctx,
        &[0u16; 8],
        &[0u16; 8],
        None,
        &[0f32; 8],
        &[0f32; 8],
        &[0],
        None,
        0,
        &mut q,
        None,
        None,
        1,
        1,
        7,
        1e-6,
    )
    .unwrap_err();
    assert!(matches!(e, WgpuError::Unsupported(_)), "{e}");
    eprintln!("shape_errors: both malformed calls rejected");
}

#[test]
fn rms_family_narrow_dims_matches_f64_oracle() {
    let Some(ctx) = ctx_or_skip("rms_family_narrow_dims") else {
        return;
    };
    let eps = 1e-6f32;
    let mut worst_rms = 0f64;
    let mut worst_rstd = 0f64;
    for (rows, dim) in [
        (4usize, 1usize),
        (3, 3),
        (5, 31),
        (4, 33),
        (2, 64),
        (3, 127),
        (2, 128),
        (2, 129),
    ] {
        let mut rnd = lcg(0xbead_0000 + (dim * 29 + rows) as u64);
        let xf: Vec<f32> = (0..rows * dim).map(|_| rnd() * 2.0).collect();
        let wf: Vec<f32> = (0..dim).map(|_| 1.0 + rnd() * 0.4).collect();
        let x = to_bf16(&xf);
        let weight = to_bf16(&wf);

        let mut inv = vec![0f64; rows];
        for r in 0..rows {
            let mut s = 0f64;
            for d in 0..dim {
                let v = from_bf16(x[r * dim + d]) as f64;
                s += v * v;
            }
            inv[r] = 1.0 / (s / dim as f64 + eps as f64).sqrt();
        }

        let mut y = vec![0f32; rows * dim];
        gd::rms_no_weight_bf16_f32(ctx, &x, &mut y, rows, dim, eps).unwrap();
        let mut yw = vec![0f32; rows * dim];
        gd::rmsnorm_bf16w_f32out(ctx, &x, &weight, &mut yw, rows, dim, eps).unwrap();
        let mut rstd = vec![0f32; rows];
        gd::rstd_bf16(ctx, &x, &mut rstd, rows, dim, eps).unwrap();

        for r in 0..rows {
            let d = ((rstd[r] as f64 - inv[r]) / inv[r]).abs();
            worst_rstd = worst_rstd.max(d);
            for c in 0..dim {
                let xv = from_bf16(x[r * dim + c]) as f64;
                let want = xv * inv[r];
                let want_w = want * from_bf16(weight[c]) as f64;
                if want.abs() > 1e-6 {
                    worst_rms = worst_rms.max(((y[r * dim + c] as f64 - want) / want).abs());
                }
                if want_w.abs() > 1e-6 {
                    worst_rms = worst_rms.max(((yw[r * dim + c] as f64 - want_w) / want_w).abs());
                }
            }
        }
    }
    eprintln!(
        "rms narrow dims vs f64 oracle: worst rel {worst_rms:e} (values), {worst_rstd:e} (rstd)"
    );
    assert!(worst_rms < 1e-6, "rms narrow dims max_rel {worst_rms:e}");
    assert!(worst_rstd < 1e-6, "rstd narrow dims max_rel {worst_rstd:e}");
}

#[test]
fn elementwise_tail_shapes_match_cpu_oracle() {
    let Some(ctx) = ctx_or_skip("elementwise_tail_shapes") else {
        return;
    };
    for n in [1usize, 2, 127, 128, 129, 255, 4097] {
        let mut rnd = lcg(0x9111_0000 + n as u64);
        let a: Vec<f32> = (0..n).map(|_| rnd() * 4.0).collect();
        let b: Vec<f32> = (0..n).map(|_| rnd() * 4.0).collect();
        let scale = 0.375f32;

        let mut y = vec![0f32; n];
        gd::add_scale_f32(ctx, &a, &b, &mut y, scale, n).unwrap();
        for i in 0..n {
            assert_eq!(
                y[i].to_bits(),
                ((a[i] + b[i]) * scale).to_bits(),
                "add_scale_f32 n={n} idx={i}"
            );
        }

        let xb = to_bf16(&a);
        let mut z = vec![0f32; n];
        gd::cast_scale_bf16_f32(ctx, &xb, &mut z, scale, n).unwrap();
        for i in 0..n {
            assert_eq!(
                z[i].to_bits(),
                (from_bf16(xb[i]) * scale).to_bits(),
                "cast_scale_bf16_f32 n={n} idx={i}"
            );
        }

        let mut w = vec![0f32; n];
        gd::cast_bf16_f32(ctx, &xb, &mut w, n).unwrap();
        let mut back = vec![0u16; n];
        gd::cast_f32_bf16(ctx, &w, &mut back, n).unwrap();
        assert_eq!(back, xb, "cast roundtrip n={n}");
    }
    eprintln!("elementwise tails: 7 lengths bit-exact vs CPU oracle");
}

#[test]
fn argmax_bf16_breaks_ties_toward_lowest_index() {
    let Some(ctx) = ctx_or_skip("argmax_bf16_ties") else {
        return;
    };
    for (vocab, a, b) in [
        (1usize, 0usize, 0usize),
        (127, 3, 126),
        (128, 0, 127),
        (129, 0, 128),
        (32769, 5, 32768),
    ] {
        let mut rnd = lcg(0x4ace_0000 + vocab as u64);
        let mut lf: Vec<f32> = (0..vocab).map(|_| rnd() * 4.0).collect();
        lf[a] = 12.5;
        lf[b] = 12.5;
        let logits = to_bf16(&lf);
        let mut token = vec![0u32; 1];
        let mut ring = vec![0xa5a5_a5a5u32; 64];
        gd::argmax_bf16(ctx, &logits, &[130], &mut token, Some(&mut ring), 63, vocab).unwrap();
        assert_eq!(
            token[0] as usize, a,
            "argmax_bf16 vocab={vocab} must pick the lowest tied index"
        );
        assert_eq!(ring[(130 - 1) & 63], token[0], "argmax_bf16 ring slot");
        let best = logits
            .iter()
            .map(|w| from_bf16(*w))
            .fold(f32::NEG_INFINITY, f32::max);
        assert_eq!(
            from_bf16(logits[token[0] as usize]),
            best,
            "argmax_bf16 vocab={vocab} must pick a maximal value"
        );
    }
    eprintln!("argmax_bf16 ties: 5 vocab sizes pick the lowest tied index");
}

#[test]
fn qkv_prep_max_head_dim_matches_f64_oracle() {
    let Some(ctx) = ctx_or_skip("qkv_prep_max_head_dim") else {
        return;
    };
    let (nh, nkv, hd) = (2usize, 1usize, 512usize);
    let half = hd / 2;
    let eps = 1e-6f32;
    let rope_pos = 5i32;
    let heads = nh + 2 * nkv;
    let slots = 8usize;
    let mut rnd = lcg(0x2bad_f00d);
    let xf: Vec<f32> = (0..heads * hd).map(|_| rnd() * 2.0).collect();
    let qwf: Vec<f32> = (0..hd).map(|_| 1.0 + rnd() * 0.3).collect();
    let kwf: Vec<f32> = (0..hd).map(|_| 1.0 + rnd() * 0.3).collect();
    let qkv = to_bf16(&xf);
    let qw = to_bf16(&qwf);
    let kw = to_bf16(&kwf);
    let cos_tbl: Vec<f32> = (0..16 * half).map(|i| ((i as f32) * 0.011).cos()).collect();
    let sin_tbl: Vec<f32> = (0..16 * half).map(|i| ((i as f32) * 0.011).sin()).collect();
    let mut kc = vec![0u16; slots * nkv * hd];
    let mut vc = vec![0u16; slots * nkv * hd];
    let mut q_out = vec![0f32; nh * hd];
    gd::qkv_prep(
        ctx,
        &qkv,
        &qw,
        Some(&kw[..]),
        &cos_tbl,
        &sin_tbl,
        &[rope_pos],
        Some(&[rope_pos + 1][..]),
        0,
        &mut q_out,
        Some(&mut kc),
        Some(&mut vc),
        nh,
        nkv,
        hd,
        eps,
    )
    .unwrap();

    let mut worst = 0f64;
    for h in 0..nh {
        let mut s = 0f64;
        for d in 0..hd {
            let v = from_bf16(qkv[h * hd + d]) as f64;
            s += v * v;
        }
        let inv = 1.0 / (s / hd as f64 + eps as f64).sqrt();
        let ns: Vec<f64> = (0..hd)
            .map(|d| from_bf16(qkv[h * hd + d]) as f64 * inv * from_bf16(qw[d]) as f64)
            .collect();
        let base = rope_pos as usize * half;
        for d in 0..hd {
            let i = if d < half { d } else { d - half };
            let (a, b) = (ns[i], ns[i + half]);
            let (c, sn) = (cos_tbl[base + i] as f64, sin_tbl[base + i] as f64);
            let want = if d < half {
                a * c - b * sn
            } else {
                a * sn + b * c
            };
            if want.abs() > 1e-5 {
                worst = worst.max(((q_out[h * hd + d] as f64 - want) / want).abs());
            }
        }
    }
    let vslot = rope_pos as usize * nkv * hd;
    let vtouched = vc[vslot..vslot + hd].iter().filter(|w| **w != 0).count();
    eprintln!(
        "qkv_prep hd=512 vs f64 oracle: worst rel {worst:e}, v-cache slot wrote {vtouched}/{hd}"
    );
    assert!(worst < 1e-5, "qkv_prep hd=512 q max_rel {worst:e}");
    assert!(
        vtouched > hd / 2,
        "qkv_prep hd=512 must fill the v-cache slot"
    );
}

#[test]
fn every_nan_bf16_encode_collapses_to_the_cuda_canonical_pattern() {
    let Some(ctx) = ctx_or_skip("nan_encode") else {
        return;
    };

    let mut x: Vec<f32> = Vec::new();
    for hi in 0x7f81u32..0x8000u32 {
        x.push(f32::from_bits(hi << 16));
        x.push(f32::from_bits((hi << 16) | 0x8000_0000));
    }
    x.push(f32::from_bits(0x7f80_0001));
    x.push(f32::from_bits(0xff80_0001));
    x.push(f32::from_bits(0x7fff_ffff));
    let n = x.len();
    let mut y = vec![0u16; n];
    gd::cast_f32_bf16(ctx, &x, &mut y, n).unwrap();
    let bad = y.iter().filter(|v| **v != 0x7fff).count();
    eprintln!("cast_f32_bf16 nan canonicalisation: {bad}/{n} words are not 0x7fff");
    assert_eq!(
        bad, 0,
        "every nan must encode to the 0x7fff pattern cuda emits"
    );

    let gate: Vec<u16> = vec![0x3f80, 0x0000, 0x7f80, 0x3f80, 0xff80];
    let pli: Vec<f32> = vec![
        f32::NAN,
        f32::INFINITY,
        0.0,
        f32::NEG_INFINITY,
        f32::INFINITY,
    ];
    let mut g = vec![0u16; gate.len()];
    gd::gelu_mul_bf16f32(ctx, &gate, &pli, &mut g, gate.len()).unwrap();
    eprintln!("gelu nan products: {g:?}");
    assert_eq!(g[0], 0x7fff, "nan * finite gelu must encode as nan");
    assert_eq!(g[1], 0x7fff, "inf * zero gelu must encode as nan");
    assert_eq!(g[2], 0x7fff, "zero * inf gelu must encode as nan");
    assert_eq!(g[3], 0xff80, "-inf * positive gelu must stay -inf");

    let dim = 130usize;
    let mut xw = to_bf16(&vec![0.5f32; dim]);
    let w = to_bf16(&vec![1.0f32; dim]);
    let rstd = vec![f32::NAN];
    let mut out = vec![0u16; dim];
    xw[7] = 0x7f80;
    gd::rms_apply_bf16(ctx, &xw, &w, &rstd, &mut out, dim, dim).unwrap();
    assert!(
        out.iter().all(|v| *v == 0x7fff),
        "a nan rstd must drive every lane to the canonical nan pattern"
    );
}

#[test]
fn infinite_row_sums_survive_the_round_to_nearest_division() {
    let Some(ctx) = ctx_or_skip("div_rn_inf") else {
        return;
    };
    let eps = 1e-6f32;
    for dim in [31usize, 128, 130, 4096] {
        let rows = 3usize;
        let mut rnd = lcg(0x2020 + dim as u64);
        let xf: Vec<f32> = (0..rows * dim).map(|_| rnd() * 2.0).collect();
        let mut x = to_bf16(&xf);
        x[0] = 0x7f80;
        x[dim + 1] = 0xff80;
        x[2 * dim + 2] = 0x7fc0;

        let mut rstd = vec![0f32; rows];
        gd::rstd_bf16(ctx, &x, &mut rstd, rows, dim, eps).unwrap();
        eprintln!("rstd dim={dim} with inf/-inf/nan rows: {rstd:?}");
        assert_eq!(
            rstd[0].to_bits(),
            0u32,
            "an +inf row sums to inf, so total/dim must stay inf and rsqrt must give +0"
        );
        assert_eq!(
            rstd[1].to_bits(),
            0u32,
            "a -inf row squares to +inf, so rsqrt must give +0"
        );
        assert!(rstd[2].is_nan(), "a nan row must stay nan");

        let mut y = vec![0f32; rows * dim];
        gd::rms_no_weight_bf16_f32(ctx, &x, &mut y, rows, dim, eps).unwrap();
        let zeros = y[dim..2 * dim].iter().filter(|v| **v == 0.0).count();
        assert!(
            zeros >= dim - 1,
            "an inf row scales by rstd=0 so every finite lane must land on zero, got {zeros}/{dim}"
        );
        assert!(
            y[dim + 1].is_nan(),
            "the -inf lane itself is inf*0 and must be nan"
        );
    }
}
