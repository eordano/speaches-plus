use anyhow::Result;
use candle_core::Tensor;

#[derive(Clone, Copy, Debug)]
pub struct SamplingParams {
    pub temperature: f32,

    pub top_k: Option<usize>,

    pub top_p: Option<f32>,

    pub min_p: Option<f32>,

    pub presence_penalty: f32,

    pub frequency_penalty: f32,

    pub repetition_penalty: f32,
}

impl Default for SamplingParams {
    fn default() -> Self {
        Self {
            temperature: 1.0,
            top_k: None,
            top_p: None,
            min_p: None,
            presence_penalty: 0.0,
            frequency_penalty: 0.0,
            repetition_penalty: 1.0,
        }
    }
}

impl SamplingParams {
    pub fn is_greedy(&self) -> bool {
        self.temperature <= 1e-6
    }

    pub fn has_penalties(&self) -> bool {
        self.presence_penalty != 0.0
            || self.frequency_penalty != 0.0
            || (self.repetition_penalty - 1.0).abs() > f32::EPSILON
    }
}

pub fn argmax_checked(logits: &[f32]) -> Option<u32> {
    let mut best: Option<(usize, f32)> = None;
    for (i, &v) in logits.iter().enumerate() {
        if !v.is_finite() {
            continue;
        }
        match best {
            Some((_, bv)) if v <= bv => {}
            _ => best = Some((i, v)),
        }
    }
    best.map(|(i, _)| i as u32)
}

pub fn argmax(logits: &[f32]) -> u32 {
    argmax_checked(logits).unwrap_or(0)
}

pub fn argmax_host_row(row: &[f32]) -> Result<u32> {
    let (top, _) = row
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
        .ok_or_else(|| anyhow::anyhow!("argmax_host_row over an empty logits row"))?;
    Ok(top as u32)
}

fn point_mass(n: usize, at: Option<u32>) -> Vec<f32> {
    let mut probs = vec![0.0f32; n];
    if let Some(i) = at {
        if (i as usize) < n {
            probs[i as usize] = 1.0;
        }
    }
    probs
}

pub fn apply_penalties(logits: &mut [f32], seen: &[(u32, u32)], p: &SamplingParams) {
    apply_penalties_with_prompt(logits, seen, &[], p)
}

pub fn apply_penalties_with_prompt(
    logits: &mut [f32],
    seen: &[(u32, u32)],
    prompt: &[u32],
    p: &SamplingParams,
) {
    if !p.has_penalties() {
        return;
    }
    let rep = p.repetition_penalty;
    let rep_active = (rep - 1.0).abs() > f32::EPSILON;
    for &(tok, cnt) in seen {
        let i = tok as usize;
        if i >= logits.len() || cnt == 0 {
            continue;
        }
        if rep_active {
            let l = logits[i];
            logits[i] = if l > 0.0 { l / rep } else { l * rep };
        }
        logits[i] -= p.presence_penalty;
        logits[i] -= p.frequency_penalty * cnt as f32;
    }
    if rep_active && !prompt.is_empty() {
        let mut applied: std::collections::HashSet<u32> = seen
            .iter()
            .filter(|&&(_, c)| c > 0)
            .map(|&(t, _)| t)
            .collect();
        for &tok in prompt {
            let i = tok as usize;
            if i >= logits.len() || !applied.insert(tok) {
                continue;
            }
            let l = logits[i];
            logits[i] = if l > 0.0 { l / rep } else { l * rep };
        }
    }
}

pub fn distribution(logits: &[f32], p: &SamplingParams) -> Vec<f32> {
    let n = logits.len();
    if n == 0 {
        return Vec::new();
    }
    if p.is_greedy() {
        return point_mass(n, argmax_checked(logits));
    }

    let inv_t = 1.0f32 / p.temperature.max(1e-6);
    let mut scaled: Vec<f32> = logits.iter().map(|&x| x * inv_t).collect();

    if let Some(k) = p.top_k {
        let k = k.min(n);
        if k == 0 {
            return point_mass(n, argmax_checked(logits));
        }
        if k < n {
            let mut idx: Vec<usize> = (0..n).collect();
            idx.sort_unstable_by(|&a, &b| {
                scaled[b]
                    .partial_cmp(&scaled[a])
                    .unwrap_or(std::cmp::Ordering::Equal)
                    .then(a.cmp(&b))
            });
            for &i in idx.iter().skip(k) {
                scaled[i] = f32::NEG_INFINITY;
            }
        }
    }

    let max = scaled.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    if !max.is_finite() {
        return point_mass(n, argmax_checked(logits));
    }
    let mut probs: Vec<f32> = scaled.iter().map(|&x| (x - max).exp()).collect();
    let sum: f32 = probs.iter().sum();
    if sum <= 0.0 || !sum.is_finite() {
        return point_mass(n, argmax_checked(logits));
    }
    for v in probs.iter_mut() {
        *v /= sum;
    }

    if let Some(tp) = p.top_p {
        if tp > 0.0 && tp < 1.0 {
            let mut order: Vec<usize> = (0..n).collect();
            order.sort_unstable_by(|&a, &b| {
                probs[b]
                    .partial_cmp(&probs[a])
                    .unwrap_or(std::cmp::Ordering::Equal)
                    .then(a.cmp(&b))
            });
            let mut cum = 0.0f32;
            let mut keep = vec![false; n];
            for &i in &order {
                keep[i] = true;
                cum += probs[i];
                if cum >= tp {
                    break;
                }
            }
            for (i, v) in probs.iter_mut().enumerate() {
                if !keep[i] {
                    *v = 0.0;
                }
            }
        }
    }

    if let Some(mp) = p.min_p {
        if mp > 0.0 {
            let pmax = probs.iter().cloned().fold(0.0f32, f32::max);
            let thresh = mp * pmax;
            for v in probs.iter_mut() {
                if *v < thresh {
                    *v = 0.0;
                }
            }
        }
    }

    let renorm: f32 = probs.iter().sum();
    if renorm <= 0.0 || !renorm.is_finite() {
        return point_mass(n, argmax_checked(logits));
    }
    for v in probs.iter_mut() {
        *v /= renorm;
    }
    probs
}

pub fn logprobs_full(logits: &[f32], temperature: f32) -> Vec<f32> {
    let n = logits.len();
    if n == 0 {
        return Vec::new();
    }
    let t = if temperature <= 1e-6 {
        1.0
    } else {
        temperature
    };
    let inv_t = 1.0f32 / t;
    let max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    if !max.is_finite() {
        return vec![f32::NEG_INFINITY; n];
    }
    let mut exps = Vec::with_capacity(n);
    let mut sum = 0.0f64;
    for &x in logits {
        let e = ((x - max) * inv_t).exp();
        sum += e as f64;
        exps.push(e);
    }
    let ln_sum = (sum.max(f64::MIN_POSITIVE)).ln() as f32;
    exps.iter()
        .map(|&e| {
            if e > 0.0 {
                e.ln() - ln_sum
            } else {
                f32::NEG_INFINITY
            }
        })
        .collect()
}

pub fn top_n_indices(values: &[f32], n: usize) -> Vec<usize> {
    let n = n.min(values.len());
    let mut idx: Vec<usize> = (0..values.len()).collect();
    idx.sort_unstable_by(|&a, &b| {
        values[b]
            .partial_cmp(&values[a])
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.cmp(&b))
    });
    idx.truncate(n);
    idx
}

pub fn sample_token(logits: &[f32], p: &SamplingParams, u01: f32) -> u32 {
    sample_token_checked(logits, p, u01).unwrap_or(0)
}

pub fn sample_token_checked(logits: &[f32], p: &SamplingParams, u01: f32) -> Option<u32> {
    if logits.is_empty() {
        return None;
    }
    if p.is_greedy() {
        return argmax_checked(logits);
    }
    let probs = distribution(logits, p);
    sample_from_checked(&probs, u01)
}

pub fn sample_from(probs: &[f32], u01: f32) -> u32 {
    sample_from_checked(probs, u01).unwrap_or(0)
}

pub fn residual_sample_checked(probs: &[f32], excluded: u32, u01: f64) -> Option<u32> {
    let ex = excluded as usize;
    let mut total = 0.0f64;
    for (i, &p) in probs.iter().enumerate() {
        if i != ex && p > 0.0 {
            total += p as f64;
        }
    }
    if !total.is_finite() || total <= 0.0 {
        return None;
    }
    let target = u01.clamp(0.0, 1.0 - f64::EPSILON) * total;
    let mut acc = 0.0f64;
    let mut last: Option<u32> = None;
    for (i, &p) in probs.iter().enumerate() {
        if i == ex || !matches!(p.partial_cmp(&0.0), Some(std::cmp::Ordering::Greater)) {
            continue;
        }
        acc += p as f64;
        last = Some(i as u32);
        if target < acc {
            return Some(i as u32);
        }
    }
    last
}

pub fn sample_from_checked(probs: &[f32], u01: f32) -> Option<u32> {
    let u = u01.clamp(0.0, 1.0 - f32::EPSILON);
    let mut acc = 0.0f32;
    for (i, &pr) in probs.iter().enumerate() {
        acc += pr;
        if u < acc {
            return Some(i as u32);
        }
    }
    for i in (0..probs.len()).rev() {
        if probs[i] > 0.0 {
            return Some(i as u32);
        }
    }
    None
}

#[derive(Clone, Copy, Debug)]
#[deprecated(note = "use SamplingParams + sample_token")]
pub struct SamplerConfig {
    pub top_k: usize,
    pub top_p: f32,
    pub temperature: f32,
    pub repetition_penalty: f32,
    pub seed: u64,
}

#[allow(deprecated)]
impl Default for SamplerConfig {
    fn default() -> Self {
        Self {
            top_k: 0,
            top_p: 1.0,
            temperature: 1.0,
            repetition_penalty: 1.0,
            seed: 0,
        }
    }
}

#[allow(deprecated)]
pub fn sample(_logits: &Tensor, _cfg: &SamplerConfig) -> Result<Tensor> {
    anyhow::bail!("Tensor sampler not wired; use sample_token on a logits slice")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn residual_reference(probs: &[f32], excluded: u32, u01: f32) -> Option<u32> {
        let mut p = probs.to_vec();
        if let Some(v) = p.get_mut(excluded as usize) {
            *v = 0.0;
        }
        let s: f32 = p.iter().sum();
        if !(s > 0.0) || !s.is_finite() {
            return None;
        }
        for v in p.iter_mut() {
            *v /= s;
        }
        sample_from_checked(&p, u01)
    }

    #[test]
    fn residual_sample_matches_explicit_renormalization() {
        let probs = [0.05f32, 0.30, 0.01, 0.24, 0.00, 0.40];
        for excluded in 0..probs.len() as u32 {
            let total: f64 = probs
                .iter()
                .enumerate()
                .filter(|(i, _)| *i != excluded as usize)
                .map(|(_, &p)| p as f64)
                .sum();
            let mut cuts: Vec<f64> = Vec::new();
            let mut acc = 0.0f64;
            for (i, &p) in probs.iter().enumerate() {
                if i == excluded as usize {
                    continue;
                }
                acc += p as f64;
                cuts.push(acc / total);
            }
            for step in 0..2000u32 {
                let u = step as f64 / 2000.0;

                if cuts.iter().any(|c| (c - u).abs() < 1e-5) {
                    continue;
                }
                let got = residual_sample_checked(&probs, excluded, u);
                let want = residual_reference(&probs, excluded, u as f32);
                assert_eq!(got, want, "excluded={excluded} u={u}");
            }
        }
    }

    #[test]
    fn residual_sample_never_returns_the_excluded_token() {
        let probs = [0.1f32, 0.7, 0.2];
        for step in 0..1000u32 {
            let u = step as f64 / 1000.0;
            for excluded in 0..3u32 {
                assert_ne!(residual_sample_checked(&probs, excluded, u), Some(excluded));
            }
        }
    }

    #[test]
    fn residual_sample_preserves_conditional_mass() {
        let probs = [0.05f32, 0.30, 0.01, 0.24, 0.40];
        let excluded = 3u32;
        let rest: f64 = 1.0 - probs[excluded as usize] as f64;
        let n = 200_000u32;
        let mut hist = vec![0u64; probs.len()];
        for step in 0..n {
            let u = (step as f64 + 0.5) / n as f64;
            hist[residual_sample_checked(&probs, excluded, u).unwrap() as usize] += 1;
        }
        for t in 0..probs.len() {
            let emp = hist[t] as f64 / n as f64;
            let want = if t == excluded as usize {
                0.0
            } else {
                probs[t] as f64 / rest
            };
            assert!((emp - want).abs() < 1e-3, "token {t}: {emp} vs {want}");
        }
    }

    #[test]
    fn residual_sample_rejects_degenerate_mass() {
        assert_eq!(residual_sample_checked(&[0.0, 1.0, 0.0], 1, 0.5), None);
        assert_eq!(residual_sample_checked(&[], 0, 0.5), None);
        assert_eq!(residual_sample_checked(&[f32::NAN, 0.0], 1, 0.5), None);
    }

    #[test]
    fn greedy_is_argmax() {
        let l = [0.1, 3.0, 0.2, 2.9];
        let p = SamplingParams {
            temperature: 0.0,
            ..Default::default()
        };
        assert_eq!(sample_token(&l, &p, 0.999), 1);
        assert!(p.is_greedy());
    }

    #[test]
    fn top_k_one_is_deterministic() {
        let l = [0.1, 3.0, 0.2, 2.9];
        let p = SamplingParams {
            temperature: 1.0,
            top_k: Some(1),
            ..Default::default()
        };
        for &u in &[0.0f32, 0.3, 0.7, 0.999] {
            assert_eq!(sample_token(&l, &p, u), 1, "top_k=1 must pin argmax");
        }
    }

    #[test]
    fn distribution_sums_to_one() {
        let l = [1.0, 2.0, 0.5, -1.0, 3.0];
        let p = SamplingParams {
            temperature: 0.8,
            top_p: Some(0.9),
            ..Default::default()
        };
        let d = distribution(&l, &p);
        let s: f32 = d.iter().sum();
        assert!((s - 1.0).abs() < 1e-5, "sum={s}");

        assert!(d.iter().all(|&x| x >= 0.0));
    }

    #[test]
    fn min_p_filters_low_mass() {
        let l = [6.0, 0.0, -1.0, -2.0];
        let p = SamplingParams {
            temperature: 1.0,
            min_p: Some(0.5),
            ..Default::default()
        };
        let d = distribution(&l, &p);
        assert!(d[0] > 0.0);

        assert_eq!(d[2], 0.0);
        assert_eq!(d[3], 0.0);
        assert!((d.iter().sum::<f32>() - 1.0).abs() < 1e-5);
    }

    #[test]
    fn repetition_penalty_lowers_seen_token() {
        let mut l = vec![2.0f32, 2.0, 2.0];
        let p = SamplingParams {
            repetition_penalty: 2.0,
            ..Default::default()
        };
        apply_penalties(&mut l, &[(1, 1)], &p);
        assert!(l[1] < l[0], "seen token 1 must be penalized: {l:?}");
        assert_eq!(l[0], 2.0);
        assert_eq!(l[2], 2.0);
    }

    #[test]
    fn frequency_and_presence_penalty() {
        let mut l = vec![5.0f32, 5.0];
        let p = SamplingParams {
            presence_penalty: 1.0,
            frequency_penalty: 0.5,
            ..Default::default()
        };
        apply_penalties(&mut l, &[(0, 4)], &p);

        assert!((l[0] - 2.0).abs() < 1e-5, "{l:?}");
        assert_eq!(l[1], 5.0);
    }

    #[test]
    fn repetition_penalty_covers_prompt_tokens() {
        let mut l = vec![2.0f32, 2.0, -2.0, 2.0];
        let p = SamplingParams {
            repetition_penalty: 2.0,
            ..Default::default()
        };
        apply_penalties_with_prompt(&mut l, &[], &[1, 2], &p);
        assert_eq!(l[0], 2.0);
        assert_eq!(l[1], 1.0, "positive prompt token: l/rep");
        assert_eq!(l[2], -4.0, "negative prompt token: l*rep");
        assert_eq!(l[3], 2.0);
    }

    #[test]
    fn repetition_penalty_prompt_and_generated_token_penalized_once() {
        let mut both = vec![2.0f32, 4.0];
        let p = SamplingParams {
            repetition_penalty: 2.0,
            ..Default::default()
        };
        apply_penalties_with_prompt(&mut both, &[(1, 3)], &[1, 1, 1], &p);
        assert_eq!(both[1], 2.0, "must be 4.0/2.0, not divided repeatedly");

        let mut dup = vec![4.0f32, 2.0];
        apply_penalties_with_prompt(&mut dup, &[], &[0, 0, 0], &p);
        assert_eq!(dup[0], 2.0);
    }

    #[test]
    fn presence_frequency_penalties_ignore_prompt_tokens() {
        let mut l = vec![5.0f32, 5.0];
        let p = SamplingParams {
            presence_penalty: 1.0,
            frequency_penalty: 0.5,
            ..Default::default()
        };
        apply_penalties_with_prompt(&mut l, &[], &[0], &p);
        assert_eq!(
            l[0], 5.0,
            "prompt-only token gets no presence/frequency penalty"
        );
        assert_eq!(l[1], 5.0);
    }

    #[test]
    fn apply_penalties_without_prompt_unchanged() {
        let mut a = vec![2.0f32, 2.0, 2.0];
        let mut b = a.clone();
        let p = SamplingParams {
            repetition_penalty: 1.3,
            presence_penalty: 0.2,
            frequency_penalty: 0.1,
            ..Default::default()
        };
        apply_penalties(&mut a, &[(1, 2)], &p);
        apply_penalties_with_prompt(&mut b, &[(1, 2)], &[], &p);
        assert_eq!(a, b);
    }

    #[test]
    fn sample_from_inverse_cdf() {
        let probs = [0.2f32, 0.5, 0.3];
        assert_eq!(sample_from(&probs, 0.0), 0);
        assert_eq!(sample_from(&probs, 0.19), 0);
        assert_eq!(sample_from(&probs, 0.21), 1);
        assert_eq!(sample_from(&probs, 0.69), 1);
        assert_eq!(sample_from(&probs, 0.71), 2);
        assert_eq!(sample_from(&probs, 0.999), 2);
    }

    #[test]
    fn empirical_frequencies_match_distribution() {
        let l = [1.0f32, 2.0, 3.0];
        let p = SamplingParams {
            temperature: 1.0,
            ..Default::default()
        };
        let d = distribution(&l, &p);
        let mut counts = [0u32; 3];
        let mut state: u64 = 0x9E3779B97F4A7C15;
        let n = 200_000;
        for _ in 0..n {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let u = ((state >> 11) as f64 / (1u64 << 53) as f64) as f32;
            counts[sample_token(&l, &p, u) as usize] += 1;
        }
        for i in 0..3 {
            let emp = counts[i] as f32 / n as f32;
            assert!((emp - d[i]).abs() < 0.01, "tok {i}: emp={emp} exp={}", d[i]);
        }
    }
}
