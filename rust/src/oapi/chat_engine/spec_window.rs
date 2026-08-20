use super::*;

pub(crate) fn pad_suffix_draft(mut sd: Vec<u32>, k: usize, bonus: u32) -> Vec<u32> {
    if sd.len() + 1 < k {
        let pad = sd.last().copied().unwrap_or(bonus);
        sd.resize(k - 1, pad);
    }
    sd
}

pub(crate) const VERIFY_CACHE_GRAIN: usize = 256;

pub(crate) fn verify_cache_capacity(needed: usize) -> usize {
    needed.max(1).div_ceil(VERIFY_CACHE_GRAIN) * VERIFY_CACHE_GRAIN
}

pub(crate) fn verify_graph_reusable(
    cached_k: usize,
    cached_capacity: usize,
    k: usize,
    needed: usize,
) -> bool {
    cached_k == k && cached_capacity >= needed
}

pub(crate) fn take_reusable_or_build<T, E>(
    slot: &mut Option<T>,
    reusable: impl FnOnce(&T) -> bool,
    build: impl FnOnce() -> Result<T, E>,
) -> Result<T, E> {
    match slot.take() {
        Some(prev) if reusable(&prev) => Ok(prev),
        stale => {
            drop(stale);
            build()
        }
    }
}

pub(crate) fn kv_window(
    prompt_len: usize,
    max_new: usize,
    kv_max_seq_len: usize,
) -> Option<(usize, usize)> {
    if prompt_len >= kv_max_seq_len {
        return None;
    }
    let max_new = max_new.min(kv_max_seq_len - (prompt_len + 1));
    Some((prompt_len + max_new + 1, max_new))
}

pub(crate) struct AdaptiveK {
    pub(crate) k_min: usize,
    pub(crate) k_graph: usize,
    pub(crate) k_cur: usize,
    pub(crate) p_ema: f64,
    pub(crate) d_graph_ms: f64,
    pub(crate) d_eager_ms: f64,
    pub(crate) verify_ms: f64,
}

pub(crate) const ADAPTIVE_K_EMA_BETA: f64 = 0.12;
pub(crate) const ADAPTIVE_K_HYSTERESIS: f64 = 0.97;

impl AdaptiveK {
    pub(crate) fn new(k_graph: usize, k_init: usize) -> Self {
        Self {
            k_min: EAGLE3_K_MIN,
            k_graph,
            k_cur: k_init.clamp(EAGLE3_K_MIN, k_graph),
            p_ema: 0.55,
            d_graph_ms: 1.2,
            d_eager_ms: 2.4,
            verify_ms: 30.0,
        }
    }

    pub(crate) fn k_eff(&self) -> usize {
        self.k_cur
    }

    pub(crate) fn observe(
        &mut self,
        offered: usize,
        accepted: usize,
        draft_ms: f64,
        verify_ms: f64,
    ) {
        let b = ADAPTIVE_K_EMA_BETA;
        if offered > 0 {
            let a = accepted.min(offered);
            let trials = a + usize::from(a < offered);
            let r = a as f64 / trials as f64;
            self.p_ema = (1.0 - b) * self.p_ema + b * r;
        }
        if offered > 0 && draft_ms.is_finite() && draft_ms > 0.0 {
            let per = draft_ms / offered as f64;
            let slot = if self.k_cur == self.k_graph {
                &mut self.d_graph_ms
            } else {
                &mut self.d_eager_ms
            };
            *slot = (1.0 - b) * *slot + b * per;
        }
        if verify_ms.is_finite() && verify_ms > 0.0 {
            self.verify_ms = (1.0 - b) * self.verify_ms + b * verify_ms;
        }
        self.k_cur = self.choose();
    }

    pub(crate) fn cost_ms_per_tok(&self, k: usize) -> f64 {
        let p = self.p_ema.clamp(0.02, 0.98);
        let d = if k == self.k_graph {
            self.d_graph_ms
        } else {
            self.d_eager_ms
        };
        let tau = (1.0 - p.powi(k as i32)) / (1.0 - p);
        (d * (k - 1) as f64 + self.verify_ms) / tau
    }

    pub(crate) fn choose(&self) -> usize {
        let mut best = self.k_cur;
        let mut best_cost = self.cost_ms_per_tok(self.k_cur) * ADAPTIVE_K_HYSTERESIS;
        for k in self.k_min..=self.k_graph {
            if k == self.k_cur {
                continue;
            }
            let c = self.cost_ms_per_tok(k);
            if c < best_cost {
                best = k;
                best_cost = c;
            }
        }
        best
    }
}

#[cfg(test)]
pub(crate) fn spec_verify_cache_len(
    prompt_len: usize,
    max_new: usize,
    k: usize,
    kv_max_seq_len: usize,
) -> Option<usize> {
    let (committed_max, _) = kv_window(prompt_len, max_new, kv_max_seq_len)?;
    committed_max
        .checked_add(k)?
        .checked_add(SPEC_VERIFY_HEADROOM)
}

pub(crate) fn spec_verify_window(
    prompt_len: usize,
    max_new: usize,
    k: usize,
    kv_max_seq_len: usize,
) -> Option<(usize, usize)> {
    let (_, window_clamped) = kv_window(prompt_len, max_new, kv_max_seq_len)?;
    let reserve = k.checked_add(SPEC_VERIFY_HEADROOM)?;

    let room = kv_max_seq_len.saturating_sub(reserve);
    let clamped = window_clamped.min(room.saturating_sub(prompt_len.saturating_add(1)));
    let committed_max = prompt_len.checked_add(clamped)?.checked_add(1)?;

    let max_seq = committed_max.checked_add(reserve)?.min(kv_max_seq_len);
    Some((max_seq, clamped))
}

#[cfg(any(test, kani))]
pub(crate) fn assert_kv_window_invariants(
    prompt_len: usize,
    max_new: usize,
    kv_max_seq_len: usize,
) {
    match kv_window(prompt_len, max_new, kv_max_seq_len) {
        None => assert!(
            prompt_len >= kv_max_seq_len,
            "kv_window rejected a prompt that fits"
        ),
        Some((cache_len, clamped)) => {
            assert!(
                prompt_len < kv_max_seq_len,
                "kv_window accepted a prompt that does not fit"
            );
            assert!(clamped <= max_new, "kv_window grew max_new");
            assert!(
                cache_len <= kv_max_seq_len,
                "kv_window sized the cache past the fp8 KV window"
            );
            assert!(
                cache_len == prompt_len + clamped + 1,
                "kv_window cache_len does not match prompt + clamped + 1"
            );
        }
    }
}

#[cfg(any(test, kani))]
pub(crate) fn assert_kv_step_in_bounds(
    prompt_len: usize,
    step: usize,
    cache_len: usize,
    kv_max_seq_len: usize,
) {
    let write_idx = prompt_len + step;
    assert!(write_idx < cache_len, "decode step writes past cache_len");
    assert!(
        write_idx < kv_max_seq_len,
        "decode step writes past the fp8 KV window"
    );
}

#[cfg(kani)]
mod kani_proofs {
    use super::{assert_kv_step_in_bounds, assert_kv_window_invariants, kv_window};

    #[kani::proof]
    fn kv_window_postconditions() {
        assert_kv_window_invariants(kani::any(), kani::any(), kani::any());
    }

    #[kani::proof]
    fn kv_window_bounds_every_decode_step() {
        let prompt_len: usize = kani::any();
        let max_new: usize = kani::any();
        let kv_max_seq_len: usize = kani::any();
        if let Some((cache_len, clamped)) = kv_window(prompt_len, max_new, kv_max_seq_len) {
            let step: usize = kani::any();
            kani::assume(step < clamped);
            assert_kv_step_in_bounds(prompt_len, step, cache_len, kv_max_seq_len);
        }
    }
}
