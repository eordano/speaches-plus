#![cfg(feature = "wgpu")]

mod common;
use common::require;
use common::rnd_f;
use nv_kernels::wgpu_backend::device::WgpuContext;
use nv_kernels::wgpu_backend::kernels::flash_decode as fd;

fn ctx(test: &str) -> Option<&'static WgpuContext> {
    match WgpuContext::shared() {
        Ok(c) if c.qualify().qualified => {
            eprintln!("{test}: {}", c.summary());
            Some(c)
        }
        Ok(c) => {
            if require() {
                panic!(
                    "{test}: wgpu adapter not qualified: {:?}",
                    c.qualify().reason
                );
            }
            eprintln!("{test}: SKIP adapter not qualified");
            None
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

fn rnd_bf16(state: &mut u64) -> u16 {
    half::bf16::from_f32(rnd_f(state)).to_bits()
}

fn cpu_attention(
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
    let group = n_heads / n_kv_heads;
    let mut out = vec![0f64; n_heads * head_dim];
    for h in 0..n_heads {
        let kvh = h / group;
        let mut scores = Vec::with_capacity(total - start);
        for p in start..total {
            let base = (p * n_kv_heads + kvh) * head_dim;
            let mut s = 0f64;
            for d in 0..head_dim {
                s += f64::from(q[h * head_dim + d]) * f64::from(k[base + d]);
            }
            scores.push(s * f64::from(scaling));
        }
        let m = scores.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let mut denom = 0f64;
        for (i, s) in scores.iter().enumerate() {
            let w = (s - m).exp();
            denom += w;
            let base = ((start + i) * n_kv_heads + kvh) * head_dim;
            for d in 0..head_dim {
                out[h * head_dim + d] += w * f64::from(v[base + d]);
            }
        }
        if denom > 0.0 {
            for d in 0..head_dim {
                out[h * head_dim + d] /= denom;
            }
        }
    }
    out.into_iter().map(|x| x as f32).collect()
}

#[test]
fn the_adapter_flushes_f32_subnormals_unlike_cuda() {
    let Some(c) = ctx("wgpu_flash_decode_ftz") else {
        return;
    };
    let (nh, nkv, hd, total) = (1usize, 1usize, 64usize, 1usize);
    let q: Vec<f32> = vec![0.5; nh * hd];
    let k: Vec<f32> = vec![0.25; total * nkv * hd];
    let mut v: Vec<f32> = vec![0f32; total * nkv * hd];
    for (i, slot) in v.iter_mut().enumerate() {
        *slot = f32::from_bits(1u32 << (i % 24));
    }
    let mut got = vec![0f32; nh * hd];
    fd::flash_decode_dev_f32(
        c,
        &q,
        &k,
        &v,
        &mut got,
        &[total as i32],
        nh,
        nkv,
        hd,
        0,
        1.0,
        16,
    )
    .expect("wgpu flash_decode_dev_f32");

    let mut subnormal_flushed = 0usize;
    let mut subnormal_total = 0usize;
    let mut normal_bad = 0usize;
    for (a, b) in v.iter().zip(got.iter()) {
        if a.abs() < f32::MIN_POSITIVE {
            subnormal_total += 1;
            if *b == 0.0 {
                subnormal_flushed += 1;
            }
        } else if a.to_bits() != b.to_bits() {
            normal_bad += 1;
        }
    }
    eprintln!(
        "ftz: {subnormal_flushed}/{subnormal_total} subnormal V values came back as zero, \
         {normal_bad} normal values wrong"
    );
    assert_eq!(
        normal_bad, 0,
        "a single-position decode must return V verbatim for normal f32 values"
    );
    assert_eq!(
        subnormal_flushed, subnormal_total,
        "this adapter preserved some f32 subnormals; the CUDA-parity suite carves out \
         flush-to-zero divergence on the assumption that it flushes all of them"
    );
}

#[test]
fn flash_decode_dev_f32_matches_a_cpu_softmax_oracle() {
    let Some(c) = ctx("wgpu_flash_decode_oracle") else {
        return;
    };
    let cases: [(usize, usize, usize, usize, usize); 4] = [
        (1, 1, 64, 1, 0),
        (4, 2, 128, 63, 0),
        (8, 8, 64, 257, 0),
        (4, 2, 96, 300, 64),
    ];
    let mut worst = 0f32;
    for (nh, nkv, hd, total, window) in cases {
        let mut st = 0x2468_ace0u64 ^ (total as u64);
        let q: Vec<f32> = (0..nh * hd).map(|_| rnd_f(&mut st)).collect();
        let k: Vec<f32> = (0..total * nkv * hd).map(|_| rnd_f(&mut st)).collect();
        let v: Vec<f32> = (0..total * nkv * hd).map(|_| rnd_f(&mut st)).collect();
        let start = if window > 0 && total > window {
            total - window
        } else {
            0
        };
        let want = cpu_attention(&q, &k, &v, nh, nkv, hd, start, total, 1.0);
        let mut got = vec![0f32; nh * hd];
        fd::flash_decode_dev_f32(
            c,
            &q,
            &k,
            &v,
            &mut got,
            &[total as i32],
            nh,
            nkv,
            hd,
            window,
            1.0,
            16,
        )
        .expect("wgpu flash_decode_dev_f32");
        let mut max_abs = 0f32;
        for (a, b) in want.iter().zip(got.iter()) {
            max_abs = max_abs.max((a - b).abs());
        }
        eprintln!(
            "oracle nh={nh} nkv={nkv} hd={hd} total={total} win={window}: max_abs={max_abs:e}"
        );
        worst = worst.max(max_abs);
        assert!(max_abs < 1e-5, "nh={nh} hd={hd} total={total}: {max_abs:e}");
        assert!(
            got.iter().any(|x| *x != 0.0),
            "nh={nh} hd={hd}: output is all zeros"
        );
    }
    eprintln!("wgpu flash_decode oracle worst max_abs={worst:e}");
}

#[test]
fn splitk_bf16kv_ring_wrap_matches_the_linearized_cache() {
    let Some(c) = ctx("wgpu_flash_decode_ring") else {
        return;
    };
    let (nh, nkv, hd) = (8usize, 2usize, 128usize);
    let (total, window, ring) = (700usize, 128usize, 256usize);
    let mut st = 0x7777_1111u64;
    let q: Vec<f32> = (0..nh * hd).map(|_| rnd_f(&mut st)).collect();
    let ring_k: Vec<u16> = (0..ring * nkv * hd).map(|_| rnd_bf16(&mut st)).collect();
    let ring_v: Vec<u16> = (0..ring * nkv * hd).map(|_| rnd_bf16(&mut st)).collect();

    let per_slot = nkv * hd;
    let mut lin_k = vec![0u16; total * per_slot];
    let mut lin_v = vec![0u16; total * per_slot];
    for p in 0..total {
        let src = (p % ring) * per_slot;
        lin_k[p * per_slot..(p + 1) * per_slot].copy_from_slice(&ring_k[src..src + per_slot]);
        lin_v[p * per_slot..(p + 1) * per_slot].copy_from_slice(&ring_v[src..src + per_slot]);
    }

    let elems = fd::flash_splitk_scratch_elems(nh, hd, 16).unwrap();
    let mut s_ring = vec![0f32; elems];
    let mut s_lin = vec![0f32; elems];
    let mut o_ring = vec![0u16; nh * hd];
    let mut o_lin = vec![0u16; nh * hd];

    fd::flash_decode_splitk_bf16kv(
        c,
        &q,
        &ring_k,
        &ring_v,
        &mut o_ring,
        &mut s_ring,
        &[total as i32],
        nh,
        nkv,
        hd,
        window,
        1.0,
        16,
        ring,
    )
    .expect("ring");
    fd::flash_decode_splitk_bf16kv(
        c,
        &q,
        &lin_k,
        &lin_v,
        &mut o_lin,
        &mut s_lin,
        &[total as i32],
        nh,
        nkv,
        hd,
        window,
        1.0,
        16,
        0,
    )
    .expect("linear");

    let out_diff = o_ring
        .iter()
        .zip(o_lin.iter())
        .filter(|(a, b)| a != b)
        .count();
    let scr_diff = s_ring
        .iter()
        .zip(s_lin.iter())
        .filter(|(a, b)| a.to_bits() != b.to_bits())
        .count();
    eprintln!(
        "ring wrap: out_diff={out_diff}/{} scratch_diff={scr_diff}/{elems}",
        o_ring.len()
    );
    assert_eq!(scr_diff, 0, "ring scratch differs in {scr_diff} floats");
    assert_eq!(out_diff, 0, "ring output differs in {out_diff} words");
    assert!(o_ring.iter().any(|x| *x != 0), "ring output is all zeros");
}

#[test]
fn split_count_repartitions_the_kv_range() {
    let Some(c) = ctx("wgpu_flash_decode_split_order") else {
        return;
    };
    let (nh, nkv, hd, total) = (4usize, 2usize, 128usize, 1030usize);
    let mut st = 0x5150_2024u64;
    let q: Vec<f32> = (0..nh * hd).map(|_| rnd_f(&mut st)).collect();
    let k: Vec<u16> = (0..total * nkv * hd).map(|_| rnd_bf16(&mut st)).collect();
    let v: Vec<u16> = (0..total * nkv * hd).map(|_| rnd_bf16(&mut st)).collect();

    let mut runs: Vec<(usize, Vec<u16>, Vec<f32>)> = Vec::new();
    for splits in [8usize, 16, 32] {
        let elems = fd::flash_splitk_scratch_elems(nh, hd, splits).unwrap();
        assert_eq!(elems, nh * splits * (hd + 2));
        let mut scratch = vec![0f32; elems];
        let mut out = vec![0u16; nh * hd];
        fd::flash_decode_splitk_bf16kv(
            c,
            &q,
            &k,
            &v,
            &mut out,
            &mut scratch,
            &[total as i32],
            nh,
            nkv,
            hd,
            0,
            1.0,
            splits,
            0,
        )
        .expect("splits sweep");
        runs.push((splits, out, scratch));
    }

    for (splits, out, scratch) in runs.iter() {
        assert!(
            out.iter().any(|x| *x != 0),
            "splits={splits} output all zero"
        );
        let stride = hd + 2;
        for h in 0..nh {
            for s in 0..*splits {
                let m = scratch[(h * splits + s) * stride];
                let l = scratch[(h * splits + s) * stride + 1];
                assert!(
                    m.is_finite() && l > 0.0,
                    "splits={splits} h={h} split={s}: stage-1 partial never ran (m={m} l={l}); \
                     the split count is not reaching the dispatch grid"
                );
            }
        }
    }

    let ms = |splits: usize, scratch: &[f32]| -> Vec<f32> {
        (0..nh * splits).map(|i| scratch[i * (hd + 2)]).collect()
    };
    let m8 = ms(runs[0].0, &runs[0].2);
    let m16 = ms(runs[1].0, &runs[1].2);
    let m32 = ms(runs[2].0, &runs[2].2);
    assert_ne!(
        m8.len(),
        m16.len(),
        "the split count must change how many stage-1 partials exist"
    );
    let shared = |a: &[f32], b: &[f32]| a.iter().filter(|x| b.contains(x)).count();
    eprintln!(
        "split repartition: |m8|={} |m16|={} |m32|={} m16_in_m8={} m32_in_m16={}",
        m8.len(),
        m16.len(),
        m32.len(),
        shared(&m16, &m8),
        shared(&m32, &m16)
    );
    assert!(
        shared(&m16, &m8) < m16.len(),
        "every splits=16 block maximum already existed under splits=8; the finer split did \
         not subdivide the position range"
    );
    assert!(
        shared(&m32, &m16) < m32.len(),
        "every splits=32 block maximum already existed under splits=16; the finer split did \
         not subdivide the position range"
    );

    let dec = |x: u16| f32::from_bits(u32::from(x) << 16);
    for (splits, out, _) in runs.iter().skip(1) {
        let mut worst = 0f32;
        let mut worst_ulp = 0u32;
        for (a, b) in runs[0].1.iter().zip(out.iter()) {
            worst = worst.max((dec(*a) - dec(*b)).abs());
            worst_ulp = worst_ulp.max((i32::from(*a) - i32::from(*b)).unsigned_abs());
        }
        eprintln!(
            "split order: splits=8 vs splits={splits} max_abs={worst:e} max_bf16_ulp={worst_ulp}"
        );
        assert!(
            worst_ulp <= 1,
            "splits=8 and splits={splits} disagree by {worst_ulp} bf16 ulp (max_abs {worst:e}); \
             the split-K combine is wrong"
        );
    }
}

fn cpu_attention_bf16kv(
    q: &[f32],
    k: &[u16],
    v: &[u16],
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    start: usize,
    total: usize,
    scaling: f32,
) -> Vec<f32> {
    let dec = |x: u16| f32::from_bits(u32::from(x) << 16);
    let kf: Vec<f32> = k.iter().map(|x| dec(*x)).collect();
    let vf: Vec<f32> = v.iter().map(|x| dec(*x)).collect();
    cpu_attention(
        q, &kf, &vf, n_heads, n_kv_heads, head_dim, start, total, scaling,
    )
}

#[test]
fn splitk_and_fused_bf16kv_match_a_cpu_softmax_oracle() {
    let Some(c) = ctx("wgpu_flash_decode_bf16_oracle") else {
        return;
    };
    let cases: [(usize, usize, usize, usize, usize); 5] = [
        (1, 1, 64, 1, 0),
        (4, 2, 128, 63, 0),
        (8, 8, 64, 257, 0),
        (4, 2, 96, 300, 64),
        (4, 2, 66, 300, 0),
    ];
    let dec = |x: u16| f32::from_bits(u32::from(x) << 16);
    let mut worst = 0f32;
    for (nh, nkv, hd, total, window) in cases {
        for scaling in [1.0f32, 0.125, 3.0] {
            let mut st = 0x3141_5926u64 ^ (total as u64) ^ ((hd as u64) << 20);
            let q: Vec<f32> = (0..nh * hd).map(|_| rnd_f(&mut st)).collect();
            let k: Vec<u16> = (0..total * nkv * hd).map(|_| rnd_bf16(&mut st)).collect();
            let v: Vec<u16> = (0..total * nkv * hd).map(|_| rnd_bf16(&mut st)).collect();
            let start = if window > 0 && total > window {
                total - window
            } else {
                0
            };
            let want = cpu_attention_bf16kv(&q, &k, &v, nh, nkv, hd, start, total, scaling);
            let elems = fd::flash_splitk_scratch_elems(nh, hd, 16).unwrap();
            for fused in [false, true] {
                let mut scratch = vec![0f32; elems];
                let mut out = vec![0u16; nh * hd];
                if fused {
                    fd::flash_decode_fused_bf16kv(
                        c,
                        &q,
                        &k,
                        &v,
                        &mut out,
                        &mut scratch,
                        &[total as i32],
                        0,
                        nh,
                        nkv,
                        hd,
                        window,
                        scaling,
                        16,
                        0,
                    )
                    .expect("wgpu flash_decode_fused_bf16kv");
                } else {
                    fd::flash_decode_splitk_bf16kv(
                        c,
                        &q,
                        &k,
                        &v,
                        &mut out,
                        &mut scratch,
                        &[total as i32],
                        nh,
                        nkv,
                        hd,
                        window,
                        scaling,
                        16,
                        0,
                    )
                    .expect("wgpu flash_decode_splitk_bf16kv");
                }
                let mut max_rel = 0f32;
                for (a, b) in want.iter().zip(out.iter()) {
                    let g = dec(*b);
                    let denom = a.abs().max(g.abs()).max(1e-3);
                    max_rel = max_rel.max((a - g).abs() / denom);
                }
                eprintln!(
                    "bf16 oracle fused={fused} nh={nh} nkv={nkv} hd={hd} total={total} \
                     win={window} s={scaling}: max_rel={max_rel:e}"
                );
                worst = worst.max(max_rel);
                assert!(
                    max_rel < 4.5e-3,
                    "fused={fused} nh={nh} hd={hd} total={total} s={scaling}: bf16 split-k \
                     output is {max_rel:e} away from the f64 oracle"
                );
                assert!(
                    out.iter().any(|x| *x != 0),
                    "fused={fused} nh={nh} hd={hd}: output is all zeros"
                );
            }
        }
    }
    eprintln!("wgpu bf16 split-k oracle worst max_rel={worst:e}");
}

#[test]
fn write_kv_bf16_scatters_into_the_ring_slot() {
    let Some(c) = ctx("wgpu_write_kv_bf16") else {
        return;
    };
    let mut st = 0x1357_9bdfu64;
    for (nkv, hd, ring) in [
        (4usize, 128usize, 8usize),
        (3, 33, 5),
        (1, 65, 4),
        (2, 97, 3),
    ] {
        let per_slot = nkv * hd;
        let seed: Vec<u16> = (0..ring * per_slot).map(|_| rnd_bf16(&mut st)).collect();
        for pos in [1i32, 3, 8, 9, 17] {
            let src_k: Vec<u16> = (0..per_slot).map(|_| rnd_bf16(&mut st)).collect();
            let src_v: Vec<u16> = (0..per_slot).map(|_| rnd_bf16(&mut st)).collect();
            let mut want_k = seed.clone();
            let mut want_v = seed.clone();
            let slot = ((pos - 1) as usize) % ring;
            want_k[slot * per_slot..(slot + 1) * per_slot].copy_from_slice(&src_k);
            want_v[slot * per_slot..(slot + 1) * per_slot].copy_from_slice(&src_v);

            let mut got_k = seed.clone();
            let mut got_v = seed.clone();
            fd::write_kv_bf16(
                c,
                &src_k,
                &src_v,
                &mut got_k,
                &mut got_v,
                &[pos],
                nkv,
                hd,
                ring,
            )
            .expect("wgpu write_kv_bf16");

            let dk = want_k
                .iter()
                .zip(got_k.iter())
                .filter(|(a, b)| a != b)
                .count();
            let dv = want_v
                .iter()
                .zip(got_v.iter())
                .filter(|(a, b)| a != b)
                .count();
            eprintln!(
                "write_kv nkv={nkv} hd={hd} ring={ring} pos={pos} slot={slot}: \
                 k_diff={dk} v_diff={dv}"
            );
            assert_eq!(
                dk, 0,
                "nkv={nkv} hd={hd} pos={pos}: K cache differs in {dk} words"
            );
            assert_eq!(
                dv, 0,
                "nkv={nkv} hd={hd} pos={pos}: V cache differs in {dv} words"
            );
        }
    }
}

#[test]
fn write_kv_bf16_is_a_noop_at_position_zero() {
    let Some(c) = ctx("wgpu_write_kv_bf16_zero") else {
        return;
    };
    let (nkv, hd) = (2usize, 64usize);
    let per_slot = nkv * hd;
    let mut st = 0xabcd_ef01u64;
    let seed: Vec<u16> = (0..4 * per_slot).map(|_| rnd_bf16(&mut st)).collect();
    let src_k: Vec<u16> = (0..per_slot).map(|_| rnd_bf16(&mut st)).collect();
    let src_v: Vec<u16> = (0..per_slot).map(|_| rnd_bf16(&mut st)).collect();
    let mut got_k = seed.clone();
    let mut got_v = seed.clone();
    fd::write_kv_bf16(c, &src_k, &src_v, &mut got_k, &mut got_v, &[0], nkv, hd, 0).unwrap();
    assert_eq!(got_k, seed);
    assert_eq!(got_v, seed);
}
