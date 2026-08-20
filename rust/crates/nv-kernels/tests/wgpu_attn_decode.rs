#![cfg(feature = "wgpu")]

mod common;
use common::ctx_or_skip;
use nv_kernels::wgpu_backend::device::WgpuContext;
use nv_kernels::wgpu_backend::kernels::attn_decode;

const BLOCK: usize = 128;

fn cpu_attn_decode(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    start: usize,
    total: usize,
    scaling: f32,
) -> Vec<f32> {
    let mut out = vec![0f32; n_heads * head_dim];
    let group = n_heads / n_kv_heads;
    for h in 0..n_heads {
        let kvh = h / group;
        let qh = &q[h * head_dim..(h + 1) * head_dim];

        let mut acc = vec![0f32; head_dim];
        let mut m = f32::NEG_INFINITY;
        let mut l = 0f32;

        for p in start..total {
            let kbase = (p * n_kv_heads + kvh) * head_dim;
            let mut red = vec![0f32; BLOCK];
            for (tid, slot) in red.iter_mut().enumerate() {
                let mut partial = 0f32;
                let mut d = tid;
                while d < head_dim {
                    partial = qh[d].mul_add(k[kbase + d], partial);
                    d += BLOCK;
                }
                *slot = partial;
            }
            let mut s = BLOCK / 2;
            while s > 0 {
                for tid in 0..s {
                    red[tid] += red[tid + s];
                }
                s >>= 1;
            }
            let score = red[0] * scaling;

            let m_new = m.max(score);
            let corr = (m - m_new).exp();
            let w = (score - m_new).exp();
            l = l.mul_add(corr, w);
            let vbase = (p * n_kv_heads + kvh) * head_dim;
            for (d, a) in acc.iter_mut().enumerate() {
                *a = a.mul_add(corr, w * v[vbase + d]);
            }
            m = m_new;
        }

        let inv_l = if l > 0.0 { 1.0 / l } else { 0.0 };
        for d in 0..head_dim {
            out[h * head_dim + d] = acc[d] * inv_l;
        }
    }
    out
}

fn sample(n: usize, phase: f32, amp: f32) -> Vec<f32> {
    (0..n).map(|i| ((i as f32) * phase).sin() * amp).collect()
}

fn err_report(name: &str, got: &[f32], want: &[f32]) -> (f32, f32) {
    let mut max_abs = 0f32;
    let mut max_rel = 0f32;
    for (g, w) in got.iter().zip(want.iter()) {
        let d = (g - w).abs();
        max_abs = max_abs.max(d);
        if w.abs() > 1e-3 {
            max_rel = max_rel.max(d / w.abs());
        }
    }
    eprintln!("{name}: max_abs={max_abs:e} max_rel={max_rel:e}");
    (max_abs, max_rel)
}

fn run_case(
    ctx: &'static WgpuContext,
    label: &str,
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    start: usize,
    total: usize,
    scaling: f32,
) -> (f32, f32) {
    let q = sample(n_heads * head_dim, 0.017, 1.0);
    let k = sample(total * n_kv_heads * head_dim, 0.0031, 1.0);
    let v = sample(total * n_kv_heads * head_dim, 0.0043, 2.0);
    let mut got = vec![0f32; n_heads * head_dim];
    attn_decode::attn_decode_f32(
        ctx, &q, &k, &v, &mut got, n_heads, n_kv_heads, head_dim, start, total, scaling,
    )
    .expect("attn_decode_f32");
    let want = cpu_attn_decode(
        &q, &k, &v, n_heads, n_kv_heads, head_dim, start, total, scaling,
    );
    err_report(label, &got, &want)
}

#[test]
fn wgpu_attn_decode_matches_cpu_oracle() {
    let Some(ctx) = ctx_or_skip("wgpu_attn_decode_matches_cpu_oracle") else {
        return;
    };
    let cases: [(usize, usize, usize, usize, usize); 7] = [
        (1, 1, 64, 0, 1),
        (4, 4, 128, 0, 7),
        (8, 2, 128, 0, 129),
        (8, 8, 64, 0, 256),
        (2, 1, 512, 0, 33),
        (16, 4, 128, 0, 1024),
        (4, 2, 96, 0, 2048),
    ];
    for (nh, nkv, hd, start, total) in cases {
        let label = format!("nh={nh} nkv={nkv} hd={hd} start={start} total={total}");
        let (abs, rel) = run_case(ctx, &label, nh, nkv, hd, start, total, 1.0);
        assert!(abs < 1e-4, "{label}: max_abs {abs:e}");
        assert!(rel < 1e-3, "{label}: max_rel {rel:e}");
    }
}

#[test]
fn wgpu_attn_decode_honours_start_window() {
    let Some(ctx) = ctx_or_skip("wgpu_attn_decode_honours_start_window") else {
        return;
    };
    for (start, total) in [(0usize, 512usize), (1, 512), (300, 512), (511, 512)] {
        let label = format!("start={start} total={total}");
        let (abs, _rel) = run_case(ctx, &label, 4, 2, 128, start, total, 1.0);
        assert!(abs < 1e-4, "{label}: max_abs {abs:e}");
    }
}

#[test]
fn wgpu_attn_decode_applies_scaling() {
    let Some(ctx) = ctx_or_skip("wgpu_attn_decode_applies_scaling") else {
        return;
    };
    let scaling = 1.0f32 / (128f32).sqrt();
    let (abs, rel) = run_case(ctx, "scaled", 8, 2, 128, 0, 300, scaling);
    assert!(abs < 1e-4, "scaled max_abs {abs:e}");
    assert!(rel < 1e-3, "scaled max_rel {rel:e}");
}

#[test]
fn wgpu_attn_decode_empty_window_is_zero() {
    let Some(ctx) = ctx_or_skip("wgpu_attn_decode_empty_window_is_zero") else {
        return;
    };
    let (nh, nkv, hd, total) = (4usize, 2usize, 64usize, 16usize);
    let q = sample(nh * hd, 0.017, 1.0);
    let k = sample(total * nkv * hd, 0.0031, 1.0);
    let v = sample(total * nkv * hd, 0.0043, 2.0);
    let mut got = vec![7f32; nh * hd];
    attn_decode::attn_decode_f32(ctx, &q, &k, &v, &mut got, nh, nkv, hd, total, total, 1.0)
        .expect("attn_decode_f32");
    assert!(
        got.iter().all(|x| *x == 0.0),
        "empty window must produce zeros (l==0 sentinel)"
    );
}

#[test]
fn wgpu_attn_decode_uniform_scores_average_v() {
    let Some(ctx) = ctx_or_skip("wgpu_attn_decode_uniform_scores_average_v") else {
        return;
    };
    let (nh, nkv, hd, total) = (2usize, 1usize, 128usize, 64usize);
    let q = vec![0f32; nh * hd];
    let k = sample(total * nkv * hd, 0.0031, 1.0);
    let v = sample(total * nkv * hd, 0.0043, 2.0);
    let mut got = vec![0f32; nh * hd];
    attn_decode::attn_decode_f32(ctx, &q, &k, &v, &mut got, nh, nkv, hd, 0, total, 1.0)
        .expect("attn_decode_f32");

    let mut want = vec![0f32; hd];
    for p in 0..total {
        for d in 0..hd {
            want[d] += v[p * hd + d];
        }
    }
    for w in want.iter_mut() {
        *w /= total as f32;
    }
    let mut max_abs = 0f32;
    for h in 0..nh {
        for d in 0..hd {
            max_abs = max_abs.max((got[h * hd + d] - want[d]).abs());
        }
    }
    eprintln!("uniform scores max_abs={max_abs:e}");
    assert!(max_abs < 1e-5, "q==0 must average V: {max_abs:e}");
}

#[test]
fn wgpu_attn_decode_rejects_bad_inputs() {
    let Some(ctx) = ctx_or_skip("wgpu_attn_decode_rejects_bad_inputs") else {
        return;
    };
    let mut out = vec![0f32; 8];
    let e = attn_decode::attn_decode_f32(
        ctx,
        &[0f32; 8],
        &[0f32; 8],
        &[0f32; 16],
        &mut out,
        2,
        1,
        4,
        0,
        4,
        1.0,
    )
    .unwrap_err();
    eprintln!("short k rejection: {e}");

    let hd = attn_decode::MAX_HEAD_DIM + 1;
    let mut wide = vec![0f32; hd];
    let e = attn_decode::attn_decode_f32(
        ctx,
        &vec![0f32; hd],
        &vec![0f32; hd],
        &vec![0f32; hd],
        &mut wide,
        1,
        1,
        hd,
        0,
        1,
        1.0,
    )
    .unwrap_err();
    eprintln!("head_dim rejection: {e}");

    let mut o = vec![0f32; 3 * 4];
    let e = attn_decode::attn_decode_f32(
        ctx,
        &[0f32; 12],
        &[0f32; 8],
        &[0f32; 8],
        &mut o,
        3,
        2,
        4,
        0,
        1,
        1.0,
    )
    .unwrap_err();
    eprintln!("gqa group rejection: {e}");
}
