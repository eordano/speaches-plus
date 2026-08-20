#![cfg(feature = "wgpu")]

mod common;
use common::ctx_or_skip;
use nv_kernels::wgpu_backend::device::WgpuContext;
use nv_kernels::wgpu_backend::kernels::sampler;

const BLOCK: usize = 256;
const LOG2E: f32 = std::f32::consts::LOG2_E;

fn tree_sum(vals: &[f32; BLOCK]) -> f32 {
    let mut s = *vals;
    let mut stride = BLOCK / 2;
    while stride > 0 {
        for t in 0..stride {
            s[t] += s[t + stride];
        }
        stride >>= 1;
    }
    s[0]
}

fn tree_max(vals: &[f32; BLOCK]) -> f32 {
    let mut s = *vals;
    let mut stride = BLOCK / 2;
    while stride > 0 {
        for t in 0..stride {
            if s[t + stride] > s[t] {
                s[t] = s[t + stride];
            }
        }
        stride >>= 1;
    }
    s[0]
}

fn tree_sum_u32(vals: &[u32; BLOCK]) -> u32 {
    let mut s = *vals;
    let mut stride = BLOCK / 2;
    while stride > 0 {
        for t in 0..stride {
            s[t] = s[t].wrapping_add(s[t + stride]);
        }
        stride >>= 1;
    }
    s[0]
}

fn splitmix64(mut z: u64) -> u64 {
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    z ^ (z >> 31)
}

fn unit_float(r: u64) -> f32 {
    let mant = (r >> 40) as u32;
    mant as f32 * (1.0f32 / 16777216.0f32)
}

fn per_thread<F: FnMut(usize, usize) -> f32>(vocab: usize, mut f: F) -> [f32; BLOCK] {
    let mut acc = [0f32; BLOCK];
    for (t, slot) in acc.iter_mut().enumerate() {
        let mut i = t;
        while i < vocab {
            *slot += f(t, i);
            i += BLOCK;
        }
    }
    acc
}

pub fn cpu_sampler(
    logits: &[f32],
    seeds: &[u64],
    batch: usize,
    vocab: usize,
    temperature: f32,
    top_k: u32,
    top_p: f32,
) -> (Vec<f32>, Vec<u32>) {
    let mut probs = vec![0f32; batch * vocab];
    let mut tokens = vec![0u32; batch];

    let inv_t = if temperature <= 0.0 {
        1.0e6f32
    } else {
        1.0f32 / temperature
    };

    for row in 0..batch {
        let base = row * vocab;
        let rl = &logits[base..base + vocab];
        let rp = &mut probs[base..base + vocab];

        let mut lmax = [f32::MIN; BLOCK];
        for (t, slot) in lmax.iter_mut().enumerate() {
            let mut i = t;
            while i < vocab {
                let v = rl[i] * inv_t;
                if v > *slot {
                    *slot = v;
                }
                i += BLOCK;
            }
        }
        let row_max = tree_max(&lmax);

        let mut lsum = [0f32; BLOCK];
        for (t, slot) in lsum.iter_mut().enumerate() {
            let mut i = t;
            while i < vocab {
                let v = rl[i] * inv_t;
                let e = ((v - row_max) * LOG2E).exp2();
                rp[i] = e;
                *slot += e;
                i += BLOCK;
            }
        }
        let row_sum = tree_sum(&lsum);
        let inv_sum = if row_sum > 0.0 { 1.0f32 / row_sum } else { 0.0 };
        for v in rp.iter_mut() {
            *v *= inv_sum;
        }

        if top_k > 0 && (top_k as usize) < vocab {
            let mut lo = 0.0f32;
            let mut hi = 1.0f32 + 1e-6f32;
            for _ in 0..40 {
                let mid = 0.5f32 * (lo + hi);
                let mut counts = [0u32; BLOCK];
                for (t, slot) in counts.iter_mut().enumerate() {
                    let mut i = t;
                    while i < vocab {
                        if rp[i] >= mid {
                            *slot += 1;
                        }
                        i += BLOCK;
                    }
                }
                if tree_sum_u32(&counts) > top_k {
                    lo = mid;
                } else {
                    hi = mid;
                }
            }
            for v in rp.iter_mut() {
                if *v < hi {
                    *v = 0.0;
                }
            }
            let seg = per_thread(vocab, |_, i| rp[i]);
            let sum2 = tree_sum(&seg);
            let inv2 = if sum2 > 0.0 { 1.0f32 / sum2 } else { 0.0 };
            for v in rp.iter_mut() {
                *v *= inv2;
            }
        }

        if top_p < 1.0 && top_p > 0.0 {
            let mut lo = 0.0f32;
            let mut hi = 1.0f32;
            for _ in 0..40 {
                let mid = 0.5f32 * (lo + hi);
                let mut mass = [0f32; BLOCK];
                for (t, slot) in mass.iter_mut().enumerate() {
                    let mut i = t;
                    while i < vocab {
                        let p = rp[i];
                        if p >= mid {
                            *slot += p;
                        }
                        i += BLOCK;
                    }
                }
                if tree_sum(&mass) > top_p {
                    lo = mid;
                } else {
                    hi = mid;
                }
            }
            for v in rp.iter_mut() {
                if *v < lo {
                    *v = 0.0;
                }
            }
            let seg = per_thread(vocab, |_, i| rp[i]);
            let sum3 = tree_sum(&seg);
            let inv3 = if sum3 > 0.0 { 1.0f32 / sum3 } else { 0.0 };
            for v in rp.iter_mut() {
                *v *= inv3;
            }
        }

        let seg = per_thread(vocab, |_, i| rp[i]);
        let total = tree_sum(&seg);

        let mixed = splitmix64(seeds[row] ^ 0x9E37_79B9_7F4A_7C15u64.wrapping_add(row as u64));
        let mut u = unit_float(mixed);
        if u >= 1.0 {
            u = 0.99999994;
        }
        let target = u * total;

        let mut cum = 0.0f32;
        let mut found: i32 = -1;
        let mut prefix = 0.0f32;
        for (t, s) in seg.iter().enumerate() {
            if cum + *s >= target {
                found = t as i32;
                prefix = cum;
                break;
            }
            cum += *s;
        }
        if found < 0 {
            found = BLOCK as i32 - 1;
            prefix = total;
        }

        let ft = found as usize;
        let mut cum = prefix;
        let mut pick = u32::MAX;
        let mut i = ft;
        while i < vocab {
            let p = rp[i];
            cum += p;
            if cum >= target && p > 0.0 {
                pick = i as u32;
                break;
            }
            i += BLOCK;
        }
        if pick == u32::MAX {
            let mut j = vocab;
            while j > 0 {
                if rp[j - 1] > 0.0 {
                    pick = (j - 1) as u32;
                    break;
                }
                j -= 1;
            }
        }
        tokens[row] = pick;
    }

    (probs, tokens)
}

fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max)
}

fn run(
    ctx: &WgpuContext,
    logits: &[f32],
    seeds: &[u64],
    batch: usize,
    vocab: usize,
    temperature: f32,
    top_k: u32,
    top_p: f32,
) -> (Vec<f32>, Vec<u32>) {
    let mut probs = vec![0f32; batch * vocab];
    let mut toks = vec![0u32; batch];
    sampler::sampler_topk_topp_seeds(
        ctx,
        logits,
        seeds,
        &mut probs,
        &mut toks,
        batch,
        vocab,
        temperature,
        top_k,
        top_p,
    )
    .unwrap();
    (probs, toks)
}

fn synth_logits(batch: usize, vocab: usize) -> Vec<f32> {
    (0..batch * vocab)
        .map(|i| {
            let x = i as f32;
            (x * 0.00137).sin() * 4.0 + (x * 0.0071).cos() * 2.0
        })
        .collect()
}

#[test]
fn sampler_matches_cpu_oracle_plain_softmax() {
    let Some(ctx) = ctx_or_skip("sampler_plain") else {
        return;
    };
    let (batch, vocab) = (5usize, 1024usize);
    let logits = synth_logits(batch, vocab);
    let seeds: Vec<u64> = (0..batch as u64)
        .map(|r| 0xDEAD_BEEF ^ (r * 7919))
        .collect();

    let (gp, gt) = run(ctx, &logits, &seeds, batch, vocab, 1.0, 0, 1.0);
    let (cp, ct) = cpu_sampler(&logits, &seeds, batch, vocab, 1.0, 0, 1.0);

    let err = max_abs_diff(&gp, &cp);
    eprintln!("sampler_plain: max_abs_prob_err={err:e} tokens gpu={gt:?} cpu={ct:?}");
    assert!(err < 1e-8, "prob mismatch {err:e}");
    assert_eq!(gt, ct);
    for r in 0..batch {
        let s: f32 = gp[r * vocab..(r + 1) * vocab].iter().sum();
        assert!((s - 1.0).abs() < 1e-3, "row {r} sums to {s}");
    }
}

#[test]
fn sampler_top_k_one_is_argmax() {
    let Some(ctx) = ctx_or_skip("sampler_topk1") else {
        return;
    };
    let (batch, vocab) = (4usize, 777usize);
    let mut logits = synth_logits(batch, vocab);
    for r in 0..batch {
        logits[r * vocab + (r * 131 + 17) % vocab] = 40.0;
    }
    let seeds: Vec<u64> = (0..batch as u64).map(|r| 12345 + r).collect();

    let (gp, gt) = run(ctx, &logits, &seeds, batch, vocab, 1.0, 1, 1.0);
    let (cp, ct) = cpu_sampler(&logits, &seeds, batch, vocab, 1.0, 1, 1.0);
    eprintln!(
        "sampler_topk1: max_abs_prob_err={:e} tokens={gt:?}",
        max_abs_diff(&gp, &cp)
    );
    assert_eq!(gt, ct);
    for r in 0..batch {
        assert_eq!(gt[r] as usize, (r * 131 + 17) % vocab);
    }
}

#[test]
fn sampler_low_temperature_is_argmax() {
    let Some(ctx) = ctx_or_skip("sampler_lowtemp") else {
        return;
    };
    let (batch, vocab) = (3usize, 512usize);
    let mut logits = synth_logits(batch, vocab);
    for r in 0..batch {
        logits[r * vocab + (r * 61 + 5) % vocab] = 9.0;
    }
    let seeds = vec![0xABCD_EF01u64; batch];
    let (_, gt) = run(ctx, &logits, &seeds, batch, vocab, 1e-4, 0, 1.0);
    for r in 0..batch {
        assert_eq!(gt[r] as usize, (r * 61 + 5) % vocab);
    }
}

#[test]
fn sampler_all_equal_logits_is_bit_exact() {
    let Some(ctx) = ctx_or_skip("sampler_uniform") else {
        return;
    };
    let (batch, vocab) = (3usize, 4096usize);
    let logits = vec![1.25f32; batch * vocab];
    let seeds: Vec<u64> = (0..batch as u64).map(|r| 999 + r * 31).collect();

    let (gp, gt) = run(ctx, &logits, &seeds, batch, vocab, 1.0, 0, 1.0);
    let (cp, ct) = cpu_sampler(&logits, &seeds, batch, vocab, 1.0, 0, 1.0);
    let bitexact = gp
        .iter()
        .zip(cp.iter())
        .all(|(a, b)| a.to_bits() == b.to_bits());
    eprintln!("sampler_uniform: probs bit-exact={bitexact} tokens gpu={gt:?} cpu={ct:?}");
    assert!(bitexact, "uniform-logit probs must be bit-exact");
    assert_eq!(gt, ct);
    for t in &gt {
        assert!((*t as usize) < vocab);
    }
}

#[test]
fn sampler_one_dominant_logit() {
    let Some(ctx) = ctx_or_skip("sampler_dominant") else {
        return;
    };
    let (batch, vocab) = (2usize, 2048usize);
    let mut logits = vec![-30.0f32; batch * vocab];
    logits[300] = 30.0;
    logits[vocab + 1900] = 30.0;
    let seeds = vec![7u64, 8u64];
    let (gp, gt) = run(ctx, &logits, &seeds, batch, vocab, 1.0, 0, 1.0);
    let (cp, ct) = cpu_sampler(&logits, &seeds, batch, vocab, 1.0, 0, 1.0);
    eprintln!(
        "sampler_dominant: max_abs_prob_err={:e} tokens={gt:?}",
        max_abs_diff(&gp, &cp)
    );
    assert_eq!(gt, ct);
    assert_eq!(gt, vec![300u32, 1900u32]);
}

#[test]
fn sampler_top_p_one_is_a_noop() {
    let Some(ctx) = ctx_or_skip("sampler_p1") else {
        return;
    };
    let (batch, vocab) = (3usize, 600usize);
    let logits = synth_logits(batch, vocab);
    let seeds = vec![0x5151_5151u64; batch];
    let (_, a) = run(ctx, &logits, &seeds, batch, vocab, 1.0, 0, 1.0);
    let (_, b) = run(ctx, &logits, &seeds, batch, vocab, 1.0, 0, 1.5);
    assert_eq!(a, b);
}

#[test]
fn sampler_top_k_restricts_support() {
    let Some(ctx) = ctx_or_skip("sampler_topk") else {
        return;
    };
    let (batch, vocab) = (1usize, 1024usize);
    let logits = synth_logits(batch, vocab);
    let mut ranked: Vec<usize> = (0..vocab).collect();
    ranked.sort_by(|a, b| logits[*b].partial_cmp(&logits[*a]).unwrap());
    let top8: std::collections::HashSet<usize> = ranked[..8].iter().copied().collect();

    for s in 0..24u64 {
        let seeds = vec![s.wrapping_mul(0x9E37_79B9)];
        let (gp, gt) = run(ctx, &logits, &seeds, batch, vocab, 1.0, 8, 1.0);
        let (_, ct) = cpu_sampler(&logits, &seeds, batch, vocab, 1.0, 8, 1.0);
        assert_eq!(gt, ct, "seed {s}");
        assert!(
            top8.contains(&(gt[0] as usize)),
            "seed {s} picked {}",
            gt[0]
        );
        let nz = gp.iter().filter(|v| **v > 0.0).count();
        assert!(nz <= 8, "seed {s}: {nz} nonzero probs");
    }
}

#[test]
fn sampler_top_p_concentrates_mass() {
    let Some(ctx) = ctx_or_skip("sampler_topp") else {
        return;
    };
    let (batch, vocab) = (1usize, 1024usize);
    let mut logits = vec![0f32; vocab];
    for (i, v) in logits.iter_mut().enumerate() {
        *v = 10.0 - (i as f32) * 0.5;
    }
    for s in 0..16u64 {
        let seeds = vec![s + 4242];
        let (gp, gt) = run(ctx, &logits, &seeds, batch, vocab, 1.0, 0, 0.9);
        let (_, ct) = cpu_sampler(&logits, &seeds, batch, vocab, 1.0, 0, 0.9);
        assert_eq!(gt, ct, "seed {s}");
        assert!(gt[0] < 16, "seed {s} picked {}", gt[0]);
        let sum: f32 = gp.iter().sum();
        assert!((sum - 1.0).abs() < 1e-3, "renormalised sum {sum}");
    }
}

#[test]
fn sampler_same_seed_is_deterministic() {
    let Some(ctx) = ctx_or_skip("sampler_determinism") else {
        return;
    };
    let (batch, vocab) = (4usize, 1500usize);
    let logits = synth_logits(batch, vocab);
    let seeds = vec![0xFACE_B00Cu64; batch];
    let (p1, t1) = run(ctx, &logits, &seeds, batch, vocab, 0.8, 64, 0.95);
    let (p2, t2) = run(ctx, &logits, &seeds, batch, vocab, 0.8, 64, 0.95);
    assert_eq!(t1, t2);
    assert!(p1
        .iter()
        .zip(p2.iter())
        .all(|(a, b)| a.to_bits() == b.to_bits()));
}

#[test]
fn sampler_different_seeds_differ() {
    let Some(ctx) = ctx_or_skip("sampler_seed_spread") else {
        return;
    };
    let (batch, vocab) = (1usize, 4096usize);
    let logits = synth_logits(batch, vocab);
    let mut seen = std::collections::HashSet::new();
    for s in 0..32u64 {
        let seeds = vec![s.wrapping_mul(0x1234_5678_9ABCu64) ^ 0x77];
        let (_, t) = run(ctx, &logits, &seeds, batch, vocab, 1.0, 0, 1.0);
        seen.insert(t[0]);
    }
    assert!(seen.len() > 4, "only {} distinct tokens", seen.len());
}

#[test]
fn sampler_vocab_smaller_than_block() {
    let Some(ctx) = ctx_or_skip("sampler_small_vocab") else {
        return;
    };
    let (batch, vocab) = (3usize, 37usize);
    let logits = synth_logits(batch, vocab);
    let seeds: Vec<u64> = (0..batch as u64).map(|r| r * 101 + 3).collect();
    let (gp, gt) = run(ctx, &logits, &seeds, batch, vocab, 1.0, 5, 0.8);
    let (cp, ct) = cpu_sampler(&logits, &seeds, batch, vocab, 1.0, 5, 0.8);
    eprintln!(
        "sampler_small_vocab: max_abs_prob_err={:e} tokens={gt:?}",
        max_abs_diff(&gp, &cp)
    );
    assert_eq!(gt, ct);
    for t in &gt {
        assert!((*t as usize) < vocab);
    }
}

#[test]
fn sampler_broadcast_seed_matches_seed_array() {
    let Some(ctx) = ctx_or_skip("sampler_broadcast") else {
        return;
    };
    let (batch, vocab) = (6usize, 800usize);
    let logits = synth_logits(batch, vocab);
    let seed = 0x0BAD_C0DEu64;
    let mut probs = vec![0f32; batch * vocab];
    let mut toks = vec![0u32; batch];
    sampler::sampler_topk_topp(
        ctx, &logits, &mut probs, &mut toks, batch, vocab, 1.0, 0, 1.0, seed,
    )
    .unwrap();
    let (_, ref_toks) = run(ctx, &logits, &vec![seed; batch], batch, vocab, 1.0, 0, 1.0);
    assert_eq!(toks, ref_toks);
}

#[test]
fn sampler_shape_errors_are_reported() {
    let Some(ctx) = ctx_or_skip("sampler_shape") else {
        return;
    };
    let mut probs = vec![0f32; 10];
    let mut toks = vec![0u32; 2];
    let e =
        sampler::sampler_topk_topp(ctx, &[0f32; 8], &mut probs, &mut toks, 2, 5, 1.0, 0, 1.0, 1)
            .unwrap_err();
    assert!(format!("{e}").contains("shape mismatch"), "{e}");
}
