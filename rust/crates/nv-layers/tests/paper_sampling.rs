use nv_layers::sampler::{
    apply_penalties, argmax_checked, distribution, sample_from_checked, SamplingParams,
};

fn params(temperature: f32) -> SamplingParams {
    SamplingParams {
        temperature,
        ..Default::default()
    }
}

struct Lcg(u64);
impl Lcg {
    fn next_f32(&mut self, lo: f32, hi: f32) -> f32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let u = ((self.0 >> 11) as f64 / (1u64 << 53) as f64) as f32;
        lo + (hi - lo) * u
    }
}

fn softmax_f64(logits: &[f32], temperature: f64) -> Vec<f64> {
    let scaled: Vec<f64> = logits.iter().map(|&x| x as f64 / temperature).collect();
    let m = scaled.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let exps: Vec<f64> = scaled.iter().map(|&x| (x - m).exp()).collect();
    let s: f64 = exps.iter().sum();
    exps.iter().map(|&e| e / s).collect()
}

fn desc_order(p: &[f64]) -> Vec<usize> {
    let mut idx: Vec<usize> = (0..p.len()).collect();
    idx.sort_by(|&a, &b| p[b].partial_cmp(&p[a]).unwrap().then(a.cmp(&b)));
    idx
}

fn support(d: &[f32]) -> Vec<usize> {
    d.iter()
        .enumerate()
        .filter(|(_, &v)| v > 0.0)
        .map(|(i, _)| i)
        .collect()
}

#[test]
fn nucleus_support_is_smallest_prefix_with_mass_at_least_p() {
    let mut rng = Lcg(0xB0B0_0001);
    for case in 0..200 {
        let n = 5 + (case % 12);
        let logits: Vec<f32> = (0..n).map(|_| rng.next_f32(-4.0, 4.0)).collect();
        let tp = rng.next_f32(0.05, 0.98);
        let p = SamplingParams {
            temperature: 1.0,
            top_p: Some(tp),
            ..Default::default()
        };

        let probs = softmax_f64(&logits, 1.0);
        let order = desc_order(&probs);
        let mut cum = 0.0f64;
        let mut want: Vec<usize> = Vec::new();
        for &i in &order {
            want.push(i);
            cum += probs[i];
            if cum >= tp as f64 {
                break;
            }
        }
        want.sort_unstable();

        let boundary: f64 = want.iter().map(|&i| probs[i]).sum();
        if (boundary - tp as f64).abs() < 1e-4 {
            continue;
        }
        let d = distribution(&logits, &p);
        assert_eq!(support(&d), want, "case {case}: tp={tp} logits={logits:?}");

        for &i in &want {
            let expect = probs[i] / boundary;
            assert!(
                (d[i] as f64 - expect).abs() < 1e-4,
                "case {case}: token {i} renorm {} vs {expect}",
                d[i]
            );
        }
    }
}

#[test]
fn top_k_support_is_k_highest_logits_with_conditional_softmax() {
    let mut rng = Lcg(0xB0B0_0002);
    for case in 0..200 {
        let n = 4 + (case % 10);
        let logits: Vec<f32> = (0..n).map(|_| rng.next_f32(-3.0, 3.0)).collect();
        let k = 1 + (case % n);
        let p = SamplingParams {
            temperature: 0.9,
            top_k: Some(k),
            ..Default::default()
        };

        let probs = softmax_f64(&logits, 0.9);
        let mut want: Vec<usize> = desc_order(&probs).into_iter().take(k).collect();
        want.sort_unstable();

        let d = distribution(&logits, &p);
        assert_eq!(support(&d), want, "case {case}: k={k} logits={logits:?}");

        let mass: f64 = want.iter().map(|&i| probs[i]).sum();
        for &i in &want {
            let expect = probs[i] / mass;
            assert!(
                (d[i] as f64 - expect).abs() < 1e-4,
                "case {case}: token {i} got {} want {expect}",
                d[i]
            );
        }
    }
}

#[test]
fn min_p_keeps_token_iff_prob_at_least_scaled_max_inclusive() {
    let mut rng = Lcg(0xB0B0_0003);
    for case in 0..200 {
        let n = 4 + (case % 12);
        let logits: Vec<f32> = (0..n).map(|_| rng.next_f32(-5.0, 5.0)).collect();
        let mp = rng.next_f32(0.02, 0.9);
        let p = SamplingParams {
            temperature: 1.0,
            min_p: Some(mp),
            ..Default::default()
        };

        let probs = softmax_f64(&logits, 1.0);
        let pmax = probs.iter().cloned().fold(0.0f64, f64::max);
        let thresh = mp as f64 * pmax;
        let want: Vec<usize> = (0..n).filter(|&i| probs[i] >= thresh).collect();

        if probs.iter().any(|&q| (q - thresh).abs() < 1e-6) {
            continue;
        }
        let d = distribution(&logits, &p);
        assert_eq!(support(&d), want, "case {case}: mp={mp} logits={logits:?}");
    }

    let logits = [(4.0f32).ln(), 0.0];
    let p = SamplingParams {
        temperature: 1.0,
        min_p: Some(0.25),
        ..Default::default()
    };
    let d = distribution(&logits, &p);
    assert!(
        d[1] > 0.0,
        "min-p threshold must be inclusive (p == p_scaled kept), got {d:?}"
    );
}

#[test]
fn temperature_distribution_is_boltzmann_softmax() {
    let mut rng = Lcg(0xB0B0_0004);
    for case in 0..100 {
        let n = 3 + (case % 8);
        let logits: Vec<f32> = (0..n).map(|_| rng.next_f32(-4.0, 4.0)).collect();
        for &t in &[0.25f32, 0.7, 1.0, 1.5, 3.0] {
            let d = distribution(&logits, &params(t));
            let want = softmax_f64(&logits, t as f64);
            for i in 0..n {
                assert!(
                    (d[i] as f64 - want[i]).abs() < 1e-4,
                    "case {case} T={t} token {i}: {} vs {}",
                    d[i],
                    want[i]
                );
            }
        }
    }
}

#[test]
fn zero_temperature_is_argmax_point_mass_with_lowest_index_tie_break() {
    let logits = [1.0f32, 3.0, 3.0, 0.5];
    let d = distribution(&logits, &params(0.0));
    assert_eq!(d, vec![0.0, 1.0, 0.0, 0.0], "first maximum must win ties");
    assert_eq!(argmax_checked(&logits), Some(1));

    let d2 = distribution(&logits, &params(1e-7));
    assert_eq!(d2, vec![0.0, 1.0, 0.0, 0.0]);
}

#[test]
fn repetition_penalty_lowers_seen_probability_for_any_logit_sign() {
    for &l_seen in &[2.0f32, 0.5, 0.0, -0.5, -2.0] {
        let logits = vec![l_seen, 1.0, -1.0];
        let p = SamplingParams {
            repetition_penalty: 1.7,
            ..Default::default()
        };
        let before = softmax_f64(&logits, 1.0)[0];
        let mut penalized = logits.clone();
        apply_penalties(&mut penalized, &[(0, 1)], &p);
        let after = softmax_f64(&penalized, 1.0)[0];
        if l_seen == 0.0 {
            assert!((after - before).abs() < 1e-9, "zero logit is a fixed point");
        } else {
            assert!(
                after < before,
                "theta>1 must lower P(seen): logit={l_seen} before={before} after={after}"
            );
        }

        assert_eq!(penalized[1], 1.0);
        assert_eq!(penalized[2], -1.0);
    }
}

#[test]
fn presence_and_frequency_penalties_match_openai_formula() {
    let mut rng = Lcg(0xB0B0_0005);
    for case in 0..100 {
        let n = 6;
        let logits: Vec<f32> = (0..n).map(|_| rng.next_f32(-3.0, 3.0)).collect();
        let presence = rng.next_f32(0.0, 2.0);
        let frequency = rng.next_f32(0.0, 2.0);
        let seen: Vec<(u32, u32)> = vec![(0, 1), (2, 3), (4, 7), (5, 0)];
        let p = SamplingParams {
            presence_penalty: presence,
            frequency_penalty: frequency,
            ..Default::default()
        };
        let mut got = logits.clone();
        apply_penalties(&mut got, &seen, &p);
        for i in 0..n {
            let count = seen
                .iter()
                .find(|&&(t, _)| t as usize == i)
                .map(|&(_, c)| c)
                .unwrap_or(0);
            let want = if count > 0 {
                logits[i] - presence - frequency * count as f32
            } else {
                logits[i]
            };
            assert!(
                (got[i] - want).abs() < 1e-5,
                "case {case} token {i}: got {} want {want}",
                got[i]
            );
        }
    }
}

#[test]
fn filter_composition_order_differs_from_vllm_min_p_first() {
    let logits: Vec<f32> = [0.4f32, 0.3, 0.2, 0.1].iter().map(|p| p.ln()).collect();
    let ours = distribution(
        &logits,
        &SamplingParams {
            temperature: 1.0,
            top_p: Some(0.75),
            min_p: Some(0.3),
            ..Default::default()
        },
    );
    assert_eq!(
        support(&ours),
        vec![0, 1, 2],
        "our order keeps the 0.2 token"
    );

    let after_min_p = distribution(
        &logits,
        &SamplingParams {
            temperature: 1.0,
            min_p: Some(0.3),
            ..Default::default()
        },
    );
    let cond_logits: Vec<f32> = after_min_p
        .iter()
        .map(|&p| if p > 0.0 { p.ln() } else { f32::NEG_INFINITY })
        .collect();
    let vllm = distribution(
        &cond_logits,
        &SamplingParams {
            temperature: 1.0,
            top_p: Some(0.75),
            ..Default::default()
        },
    );
    assert_eq!(
        support(&vllm),
        vec![0, 1],
        "vLLM's order drops the 0.2 token"
    );
    assert_ne!(
        support(&ours),
        support(&vllm),
        "composition order is observable"
    );
}

#[test]
fn sample_from_is_generalized_inverse_cdf() {
    let mut rng = Lcg(0xB0B0_0006);
    for case in 0..50 {
        let n = 3 + (case % 8);
        let raw: Vec<f32> = (0..n).map(|_| rng.next_f32(0.01, 1.0)).collect();
        let s: f32 = raw.iter().sum();
        let probs: Vec<f32> = raw.iter().map(|&x| x / s).collect();
        let cdf: Vec<f64> = probs
            .iter()
            .scan(0.0f64, |acc, &p| {
                *acc += p as f64;
                Some(*acc)
            })
            .collect();
        for step in 0..500 {
            let u = (step as f64 + 0.5) / 500.0;

            if cdf.iter().any(|c| (c - u).abs() < 1e-4) {
                continue;
            }
            let want = cdf.iter().position(|&c| u < c).unwrap_or(n - 1) as u32;
            let got = sample_from_checked(&probs, u as f32).unwrap();
            assert_eq!(got, want, "case {case} u={u} probs={probs:?}");
        }
    }
}
