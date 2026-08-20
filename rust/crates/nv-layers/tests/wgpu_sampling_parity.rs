#![cfg(feature = "wgpu")]

use nv_kernels::wgpu_backend::kernels::sampler::{
    sampler_exact_token, unit_from_seed, ExactSampling, EXACT_SENTINEL,
};
use nv_layers::sampler::{argmax_checked, sample_token_checked, SamplingParams};

#[path = "wgpu_common.rs"]
mod wgpu_common;

use wgpu_common::wgpu_ctx_or_skip as ctx_or_skip;

fn lcg(state: &mut u64) -> f32 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    ((*state >> 11) as f64 / (1u64 << 53) as f64) as f32
}

fn synth(n: usize, seed: u64, spread: f32) -> Vec<f32> {
    let mut s = seed | 1;
    (0..n).map(|_| (lcg(&mut s) - 0.5) * spread).collect()
}

fn host_params(p: &ExactSampling) -> SamplingParams {
    SamplingParams {
        temperature: p.temperature,
        top_k: Some(p.top_k as usize),
        top_p: Some(p.top_p),
        min_p: Some(p.min_p),
        ..Default::default()
    }
}

#[test]
fn exact_sampler_matches_the_host_sampler_token_for_token() {
    let Some(ctx) = ctx_or_skip("exact_sampler_matches_the_host_sampler_token_for_token") else {
        return;
    };
    let mut draws = 0usize;
    let mut agree = 0usize;
    let mut disagreements: Vec<String> = Vec::new();
    let mut distinct: std::collections::HashSet<u32> = std::collections::HashSet::new();
    for (vi, &vocab) in [64usize, 517, 4096, 32768].iter().enumerate() {
        for &spread in &[0.4f32, 6.0, 30.0] {
            let logits = synth(vocab, 0xBEEF_0001 + vi as u64 * 104_729, spread);
            for &temp in &[0.35f32, 0.8, 1.0, 1.6] {
                for &k in &[1u32, 2, 17, 64, 256] {
                    if k as usize > vocab {
                        continue;
                    }
                    for &tp in &[1.0f32, 0.9, 0.5, 0.15] {
                        for &mp in &[0.0f32, 0.06] {
                            let gp = ExactSampling {
                                temperature: temp,
                                top_k: k,
                                top_p: tp,
                                min_p: mp,
                                u01: None,
                                seed: 0,
                            };
                            let hp = host_params(&gp);
                            let mut batch_u: Vec<f32> = Vec::new();
                            for step in 0..9u32 {
                                batch_u.push(step as f32 / 9.0 + 0.021);
                            }
                            for &u in &batch_u {
                                let gp = ExactSampling { u01: Some(u), ..gp };
                                let got =
                                    sampler_exact_token(ctx, &logits, 1, vocab, &gp).unwrap()[0];
                                assert_ne!(
                                    got, EXACT_SENTINEL,
                                    "supported config returned SENTINEL: \
                                     vocab={vocab} temp={temp} k={k} tp={tp} mp={mp} u={u}"
                                );
                                let want = sample_token_checked(&logits, &hp, u).unwrap();
                                draws += 1;
                                distinct.insert(want);
                                if got == want {
                                    agree += 1;
                                } else if disagreements.len() < 12 {
                                    disagreements.push(format!(
                                        "vocab={vocab} spread={spread} temp={temp} k={k} tp={tp} \
                                         mp={mp} u={u} gpu={got} host={want}"
                                    ));
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    println!(
        "EXACTPARITY draws={draws} agree={agree} mismatches={} distinct_tokens={}",
        draws - agree,
        distinct.len()
    );
    for d in &disagreements {
        println!("EXACTPARITY mismatch {d}");
    }
    assert!(
        draws > 1000,
        "sweep degenerated to {draws} draws; the parameter grid did not run"
    );
    assert!(
        distinct.len() > 32,
        "the whole sweep produced only {} distinct tokens over {draws} draws; the sampler is \
         behaving like argmax and agreement proves nothing about the stochastic path",
        distinct.len()
    );
    assert_eq!(
        agree, draws,
        "in-shader sampler must match the host sampler exactly"
    );
}

#[test]
fn top_k_one_is_greedy_and_temperature_zero_is_greedy() {
    let Some(ctx) = ctx_or_skip("top_k_one_is_greedy_and_temperature_zero_is_greedy") else {
        return;
    };
    for &vocab in &[97usize, 8192] {
        let logits = synth(vocab, 0x5EED, 11.0);
        let am = argmax_checked(&logits).unwrap();
        for step in 0..17u32 {
            let u = step as f32 / 17.0;
            let k1 = ExactSampling {
                temperature: 0.9,
                top_k: 1,
                top_p: 1.0,
                min_p: 0.0,
                u01: Some(u),
                seed: 0,
            };
            assert_eq!(
                sampler_exact_token(ctx, &logits, 1, vocab, &k1).unwrap()[0],
                am
            );
            let t0 = ExactSampling {
                temperature: 0.0,
                ..k1
            };
            assert_eq!(
                sampler_exact_token(ctx, &logits, 1, vocab, &t0).unwrap()[0],
                am
            );
        }
    }
}

#[test]
fn top_p_one_equals_pure_temperature_sampling_within_top_k() {
    let Some(ctx) = ctx_or_skip("top_p_one_equals_pure_temperature_sampling_within_top_k") else {
        return;
    };
    let vocab = 2048usize;
    let logits = synth(vocab, 0x1234_9999, 7.0);
    for step in 0..41u32 {
        let u = step as f32 / 41.0;
        let base = ExactSampling {
            temperature: 0.85,
            top_k: 64,
            top_p: 1.0,
            min_p: 0.0,
            u01: Some(u),
            seed: 0,
        };
        let got = sampler_exact_token(ctx, &logits, 1, vocab, &base).unwrap()[0];
        let hp = SamplingParams {
            temperature: 0.85,
            top_k: Some(64),
            top_p: None,
            ..Default::default()
        };
        assert_eq!(got, sample_token_checked(&logits, &hp, u).unwrap(), "u={u}");
    }
}

#[test]
fn degenerate_distributions_do_not_divide_by_zero() {
    let Some(ctx) = ctx_or_skip("degenerate_distributions_do_not_divide_by_zero") else {
        return;
    };
    let vocab = 512usize;
    let p = ExactSampling {
        temperature: 0.7,
        top_k: 16,
        top_p: 0.9,
        min_p: 0.0,
        u01: Some(0.5),
        seed: 0,
    };
    let flat = vec![0.0f32; vocab];
    let got = sampler_exact_token(ctx, &flat, 1, vocab, &p).unwrap()[0];
    assert_eq!(
        got,
        sample_token_checked(&flat, &host_params(&p), 0.5).unwrap()
    );

    let mut spike = vec![-60.0f32; vocab];
    spike[311] = 40.0;
    let got = sampler_exact_token(ctx, &spike, 1, vocab, &p).unwrap()[0];
    assert_eq!(got, 311, "a single dominant logit must win");

    let mut ties = vec![1.0f32; vocab];
    for (i, v) in ties.iter_mut().enumerate() {
        if i % 3 == 0 {
            *v = 4.0;
        }
    }
    for step in 0..23u32 {
        let u = step as f32 / 23.0;
        let q = ExactSampling { u01: Some(u), ..p };
        let got = sampler_exact_token(ctx, &ties, 1, vocab, &q).unwrap()[0];
        assert_eq!(
            got,
            sample_token_checked(&ties, &host_params(&q), u).unwrap(),
            "tie-break must follow the host's (value desc, index asc) order at u={u}"
        );
    }
}

#[test]
fn unsupported_configurations_report_a_sentinel_rather_than_a_wrong_token() {
    let Some(ctx) =
        ctx_or_skip("unsupported_configurations_report_a_sentinel_rather_than_a_wrong_token")
    else {
        return;
    };
    let vocab = 1024usize;
    let logits = synth(vocab, 7, 5.0);
    for k in [0u32, 257, 4096] {
        let p = ExactSampling {
            temperature: 0.9,
            top_k: k,
            top_p: 0.9,
            min_p: 0.0,
            u01: Some(0.3),
            seed: 0,
        };
        assert!(
            !p.supported(vocab),
            "k={k} must be advertised as unsupported"
        );
        let got = sampler_exact_token(ctx, &logits, 1, vocab, &p).unwrap()[0];
        assert_eq!(
            got, EXACT_SENTINEL,
            "k={k} must not silently produce a token"
        );
    }
}

#[test]
fn seeded_draws_are_deterministic_and_match_the_host_twin_of_the_shader_rng() {
    let Some(ctx) =
        ctx_or_skip("seeded_draws_are_deterministic_and_match_the_host_twin_of_the_shader_rng")
    else {
        return;
    };
    let vocab = 4096usize;
    let logits = synth(vocab, 0xABCD, 9.0);
    for seed in [1u64, 42, 0x9e37_79b9_7f4a_7c15] {
        let p = ExactSampling {
            temperature: 1.0,
            top_k: 40,
            top_p: 0.95,
            min_p: 0.0,
            u01: None,
            seed,
        };
        let a = sampler_exact_token(ctx, &logits, 1, vocab, &p).unwrap()[0];
        let b = sampler_exact_token(ctx, &logits, 1, vocab, &p).unwrap()[0];
        assert_eq!(a, b, "same seed must give the same token");
        let u = unit_from_seed(seed, 0);
        let want = sample_token_checked(&logits, &host_params(&p), u).unwrap();
        assert_eq!(
            a, want,
            "shader RNG must agree with unit_from_seed for seed={seed}"
        );
    }
}

#[test]
fn batched_rows_each_get_their_own_token() {
    let Some(ctx) = ctx_or_skip("batched_rows_each_get_their_own_token") else {
        return;
    };
    let vocab = 777usize;
    let batch = 5usize;
    let mut logits = Vec::with_capacity(batch * vocab);
    for r in 0..batch {
        logits.extend(synth(vocab, 100 + r as u64, 8.0));
    }
    let p = ExactSampling {
        temperature: 0.75,
        top_k: 32,
        top_p: 0.9,
        min_p: 0.0,
        u01: Some(0.61),
        seed: 0,
    };
    let got = sampler_exact_token(ctx, &logits, batch, vocab, &p).unwrap();
    assert_eq!(got.len(), batch);
    for r in 0..batch {
        let row = &logits[r * vocab..(r + 1) * vocab];
        let want = sample_token_checked(row, &host_params(&p), 0.61).unwrap();
        assert_eq!(got[r], want, "row {r}");
    }
}
