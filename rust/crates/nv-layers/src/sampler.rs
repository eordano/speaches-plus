use anyhow::Result;

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
            idx.select_nth_unstable_by(k, |&a, &b| descending_then_index(&scaled, a, b));
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
            let mut window = TOP_P_FIRST_PARTIAL_SORT_WINDOW_8192_HOLDS_THE_NUCLEUS_OF_A_PEAKED_DISTRIBUTION.min(n);
            let kept = loop {
                if window < n {
                    order.select_nth_unstable_by(window, |&a, &b| descending_then_index(&probs, a, b));
                }
                order[..window].sort_unstable_by(|&a, &b| descending_then_index(&probs, a, b));
                let mut cum = 0.0f32;
                let closed = order[..window].iter().position(|&i| {
                    cum += probs[i];
                    cum >= tp
                });
                match closed {
                    Some(j) => break j + 1,
                    None if window == n => break n,
                    None => window = (window * 8).min(n),
                }
            };
            let mut keep = vec![false; n];
            for &i in &order[..kept] {
                keep[i] = true;
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
    if n < idx.len() {
        idx.select_nth_unstable_by(n, |&a, &b| descending_then_index(values, a, b));
    }
    idx.truncate(n);
    idx.sort_unstable_by(|&a, &b| descending_then_index(values, a, b));
    idx
}

pub const TOP_P_FIRST_PARTIAL_SORT_WINDOW_8192_HOLDS_THE_NUCLEUS_OF_A_PEAKED_DISTRIBUTION: usize = 8192;

fn descending_then_index(values: &[f32], a: usize, b: usize) -> std::cmp::Ordering {
    values[b]
        .partial_cmp(&values[a])
        .unwrap_or(std::cmp::Ordering::Equal)
        .then(a.cmp(&b))
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
    if let Some(tok) = sample_top_p_nucleus_sparse_reproducing_the_dense_distribution(logits, p, u01) {
        return Some(tok);
    }
    let probs = distribution(logits, p);
    sample_from_checked(&probs, u01)
}

pub const TOP_P_SPARSE_CANDIDATE_FLOOR_RELATIVE_TO_THE_PEAK_PROBABILITY: f32 = 1.0 / 65536.0;

fn sample_top_p_nucleus_sparse_reproducing_the_dense_distribution(logits: &[f32], p: &SamplingParams, u01: f32) -> Option<u32> {
    let tp = p.top_p?;
    if p.top_k.is_some() || p.min_p.is_some_and(|mp| mp > 0.0) || !(tp > 0.0 && tp < 1.0) {
        return None;
    }
    let inv_t = 1.0f32 / p.temperature.max(1e-6);
    let max = logits.iter().map(|&x| x * inv_t).fold(f32::NEG_INFINITY, f32::max);
    if !max.is_finite() {
        return None;
    }
    let mut probs: Vec<f32> = logits.iter().map(|&x| (x * inv_t - max).exp()).collect();
    let sum: f32 = probs.iter().sum();
    if sum <= 0.0 || !sum.is_finite() {
        return None;
    }
    let mut pmax = 0.0f32;
    for v in probs.iter_mut() {
        *v /= sum;
        pmax = pmax.max(*v);
    }
    let floor = pmax * TOP_P_SPARSE_CANDIDATE_FLOOR_RELATIVE_TO_THE_PEAK_PROBABILITY;
    let mut cand: Vec<usize> = (0..probs.len()).filter(|&i| probs[i] >= floor).collect();
    cand.sort_unstable_by(|&a, &b| descending_then_index(&probs, a, b));
    let mut cum = 0.0f32;
    let kept = cand.iter().position(|&i| {
        cum += probs[i];
        cum >= tp
    })? + 1;
    let mut nucleus: Vec<usize> = cand[..kept].to_vec();
    nucleus.sort_unstable();
    let renorm: f32 = nucleus.iter().map(|&i| probs[i]).sum();
    if renorm <= 0.0 || !renorm.is_finite() {
        return None;
    }
    let u = u01.clamp(0.0, 1.0 - f32::EPSILON);
    let mut acc = 0.0f32;
    for &i in &nucleus {
        acc += probs[i] / renorm;
        if u < acc {
            return Some(i as u32);
        }
    }
    nucleus.iter().rev().find(|&&i| probs[i] / renorm > 0.0).map(|&i| i as u32)
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
    fn residual_sample_matches_explicit_renormalization_never_returns_the_excluded_token_and_rejects_degenerate_mass(
    ) {
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
                let got = residual_sample_checked(&probs, excluded, u);
                assert_ne!(got, Some(excluded), "excluded={excluded} u={u}");
                if cuts.iter().any(|c| (c - u).abs() < 1e-5) {
                    continue;
                }
                let want = residual_reference(&probs, excluded, u as f32);
                assert_eq!(got, want, "excluded={excluded} u={u}");
            }
        }
        assert_eq!(residual_sample_checked(&[0.0, 1.0, 0.0], 1, 0.5), None);
        assert_eq!(residual_sample_checked(&[], 0, 0.5), None);
        assert_eq!(residual_sample_checked(&[f32::NAN, 0.0], 1, 0.5), None);
    }

    #[test]
    fn greedy_and_top_k_one_pin_the_argmax_and_sample_from_is_the_inverse_cdf() {
        let l = [0.1, 3.0, 0.2, 2.9];
        let p = SamplingParams {
            temperature: 0.0,
            ..Default::default()
        };
        assert_eq!(sample_token(&l, &p, 0.999), 1);
        assert!(p.is_greedy());
        let p = SamplingParams {
            temperature: 1.0,
            top_k: Some(1),
            ..Default::default()
        };
        for &u in &[0.0f32, 0.3, 0.7, 0.999] {
            assert_eq!(sample_token(&l, &p, u), 1, "top_k=1 must pin argmax");
        }
        let probs = [0.2f32, 0.5, 0.3];
        for (u, want) in [(0.0f32, 0u32), (0.19, 0), (0.21, 1), (0.69, 1), (0.71, 2), (0.999, 2)] {
            assert_eq!(sample_from(&probs, u), want, "u={u}");
        }
    }

    #[test]
    fn distribution_sums_to_one_and_min_p_filters_low_mass() {
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
    fn apply_penalties_lowers_seen_tokens_and_applies_presence_and_frequency_once() {
        let mut l = vec![2.0f32, 2.0, 2.0];
        let p = SamplingParams {
            repetition_penalty: 2.0,
            ..Default::default()
        };
        apply_penalties(&mut l, &[(1, 1)], &p);
        assert!(l[1] < l[0], "seen token 1 must be penalized: {l:?}");
        assert_eq!(l[0], 2.0);
        assert_eq!(l[2], 2.0);

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
    fn apply_penalties_with_prompt_covers_prompt_tokens_once_and_only_for_repetition() {
        let rep = SamplingParams {
            repetition_penalty: 2.0,
            ..Default::default()
        };
        let mut l = vec![2.0f32, 2.0, -2.0, 2.0];
        apply_penalties_with_prompt(&mut l, &[], &[1, 2], &rep);
        assert_eq!(l[0], 2.0);
        assert_eq!(l[1], 1.0, "positive prompt token: l/rep");
        assert_eq!(l[2], -4.0, "negative prompt token: l*rep");
        assert_eq!(l[3], 2.0);

        let mut both = vec![2.0f32, 4.0];
        apply_penalties_with_prompt(&mut both, &[(1, 3)], &[1, 1, 1], &rep);
        assert_eq!(both[1], 2.0, "must be 4.0/2.0, not divided repeatedly");
        let mut dup = vec![4.0f32, 2.0];
        apply_penalties_with_prompt(&mut dup, &[], &[0, 0, 0], &rep);
        assert_eq!(dup[0], 2.0);

        let mut l = vec![5.0f32, 5.0];
        let pf = SamplingParams {
            presence_penalty: 1.0,
            frequency_penalty: 0.5,
            ..Default::default()
        };
        apply_penalties_with_prompt(&mut l, &[], &[0], &pf);
        assert_eq!(
            l[0], 5.0,
            "prompt-only token gets no presence/frequency penalty"
        );
        assert_eq!(l[1], 5.0);

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

    fn full_sort_distribution_reference(logits: &[f32], p: &SamplingParams) -> Vec<f32> {
        let n = logits.len();
        let inv_t = 1.0f32 / p.temperature.max(1e-6);
        let mut scaled: Vec<f32> = logits.iter().map(|&x| x * inv_t).collect();
        let by = |v: &Vec<f32>, a: usize, b: usize| v[b].partial_cmp(&v[a]).unwrap_or(std::cmp::Ordering::Equal).then(a.cmp(&b));
        if let Some(k) = p.top_k {
            if k < n {
                let mut idx: Vec<usize> = (0..n).collect();
                idx.sort_unstable_by(|&a, &b| by(&scaled, a, b));
                for &i in idx.iter().skip(k) {
                    scaled[i] = f32::NEG_INFINITY;
                }
            }
        }
        let max = scaled.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let mut probs: Vec<f32> = scaled.iter().map(|&x| (x - max).exp()).collect();
        let sum: f32 = probs.iter().sum();
        for v in probs.iter_mut() {
            *v /= sum;
        }
        if let Some(tp) = p.top_p {
            let mut order: Vec<usize> = (0..n).collect();
            order.sort_unstable_by(|&a, &b| by(&probs, a, b));
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
        let renorm: f32 = probs.iter().sum();
        for v in probs.iter_mut() {
            *v /= renorm;
        }
        probs
    }

    #[test]
    fn partial_selection_top_k_top_p_reproduce_the_full_sort_bitwise_on_a_248k_vocab() {
        let n = 248_320usize;
        let mut st = 0x9e3779b97f4a7c15u64;
        let mut next = || {
            st ^= st << 13;
            st ^= st >> 7;
            st ^= st << 17;
            (st >> 40) as f32 / (1u64 << 24) as f32
        };
        for (spread, ties) in [(30.0f32, false), (4.0, false), (0.05, false), (30.0, true)] {
            let mut logits: Vec<f32> = (0..n).map(|_| (next() - 0.5) * spread).collect();
            if ties {
                for i in (0..n).step_by(97) {
                    logits[i] = logits[(i * 7) % n];
                }
            }
            for (t, k, tp) in [(0.7f32, None, Some(0.95f32)), (1.0, Some(40usize), Some(0.9)), (0.6, Some(20), None), (1.3, None, Some(0.999))] {
                let p = SamplingParams { temperature: t, top_k: k, top_p: tp, ..SamplingParams::default() };
                let got = distribution(&logits, &p);
                let want = full_sort_distribution_reference(&logits, &p);
                assert!(
                    got.iter().zip(&want).all(|(a, b)| a.to_bits() == b.to_bits()),
                    "spread {spread} ties {ties} t {t} k {k:?} top_p {tp:?}: partial selection diverged from the full sort"
                );
            }
        }
        for spread in [30.0f32, 4.0, 0.3] {
            let logits: Vec<f32> = (0..n).map(|_| (next() - 0.5) * spread).collect();
            for (t, tp) in [(0.7f32, 0.95f32), (1.0, 0.8), (1.2, 0.99)] {
                let p = SamplingParams { temperature: t, top_p: Some(tp), ..SamplingParams::default() };
                let dense = distribution(&logits, &p);
                for q in 0..64 {
                    let u = (q as f32 + next()) / 64.0;
                    assert_eq!(
                        sample_token_checked(&logits, &p, u),
                        sample_from_checked(&dense, u),
                        "spread {spread} t {t} top_p {tp} u {u}: the sparse nucleus draw diverged from the dense distribution"
                    );
                }
            }
        }
        let vals: Vec<f32> = (0..n).map(|_| next()).collect();
        let mut full: Vec<usize> = (0..n).collect();
        full.sort_unstable_by(|&a, &b| vals[b].partial_cmp(&vals[a]).unwrap().then(a.cmp(&b)));
        assert_eq!(top_n_indices(&vals, 20), full[..20].to_vec());
    }
}
