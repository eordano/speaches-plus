pub const RING_REWIND_RESERVED_SLOTS: usize = 1;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RewindLimits {
    pub ring_slots: usize,
    pub window: usize,
    pub prefill_chunk: usize,
    pub positional_state_only: bool,
}

impl RewindLimits {
    pub const NONE: Self = Self {
        ring_slots: 0,
        window: 0,
        prefill_chunk: 0,
        positional_state_only: false,
    };

    pub fn positional(ring_slots: usize, window: usize, prefill_chunk: usize) -> Self {
        Self {
            ring_slots,
            window,
            prefill_chunk,
            positional_state_only: true,
        }
    }

    pub fn max_advance(&self) -> Option<usize> {
        if !self.positional_state_only {
            return None;
        }
        if self.ring_slots == 0 {
            return Some(usize::MAX);
        }
        Some((self.ring_slots + 1).saturating_sub(self.window + RING_REWIND_RESERVED_SLOTS))
    }

    pub fn admits(&self, frontier: usize, target: usize) -> bool {
        let Some(max) = self.max_advance() else {
            return false;
        };
        if target == 0 || target > frontier || frontier - target > max {
            return false;
        }
        self.prefill_chunk == 0 || target.is_multiple_of(self.prefill_chunk)
    }

    pub fn target(&self, frontier: usize, lcp: usize) -> Option<usize> {
        let target = match self.prefill_chunk {
            0 => lcp,
            m => lcp - lcp % m,
        };
        self.admits(frontier, target).then_some(target)
    }
}

pub fn common_prefix_len(a: &[u32], b: &[u32]) -> usize {
    a.iter().zip(b.iter()).take_while(|(x, y)| x == y).count()
}

pub fn exact_extend_target(frontier: usize, lcp: usize, live_pos: usize) -> Option<usize> {
    (frontier > 0 && lcp >= frontier && live_pos == frontier).then_some(frontier)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_target_never_reaches_past_the_tokens_the_two_prompts_actually_share() {
        let prev = [1u32, 2, 3, 4, 5, 6, 7, 8];
        for (new, shared) in [
            (vec![1u32, 2, 3, 9], 3usize),
            (vec![9, 2, 3], 0),
            (vec![1, 2, 3, 4, 5, 6, 7, 8], 8),
            (vec![], 0),
        ] {
            let lcp = common_prefix_len(&prev, &new);
            assert_eq!(lcp, shared, "lcp of {prev:?} and {new:?}");
            let lim = RewindLimits::positional(64, 4, 0);
            if let Some(t) = lim.target(prev.len(), lcp) {
                assert!(
                    t <= lcp,
                    "target {t} exceeds the {lcp} tokens the prompts share, so the KV past \
                     the divergence would be replayed as if it belonged to this prompt"
                );
            }
        }
    }

    #[test]
    fn the_chunked_target_rounds_down_and_never_up() {

        let lim = RewindLimits::positional(1024, 0, 64);
        for lcp in [0usize, 1, 63, 64, 65, 127, 128, 200] {
            if let Some(t) = lim.target(500, lcp) {
                assert!(t <= lcp, "lcp {lcp} rounded UP to {t}");
                assert_eq!(t % 64, 0, "lcp {lcp} gave non-boundary target {t}");
            }
        }
        assert_eq!(lim.target(500, 128), Some(128));
        assert_eq!(lim.target(500, 191), Some(128), "191 must land on 128, not 192");
    }

    #[test]
    fn a_decoder_that_cannot_rewind_positionally_admits_nothing() {

        let none = RewindLimits::NONE;
        assert_eq!(none.max_advance(), None);
        for (frontier, target) in [(10usize, 1usize), (10, 10), (0, 0), (100, 50)] {
            assert!(!none.admits(frontier, target), "NONE admitted {target}/{frontier}");
        }
        assert_eq!(none.target(10, 5), None);
    }

    #[test]
    fn the_ring_reserves_a_slot_so_a_rewind_cannot_eat_its_own_window() {

        let lim = RewindLimits::positional(8, 4, 0);
        assert_eq!(lim.max_advance(), Some(8 + 1 - (4 + RING_REWIND_RESERVED_SLOTS)));
        assert!(lim.admits(100, 96), "a 4-token rewind fits");
        assert!(!lim.admits(100, 95), "a 5-token rewind exceeds the ring");

        let tight = RewindLimits::positional(2, 8, 0);
        assert_eq!(tight.max_advance(), Some(0), "saturating, never wrapped");
        assert!(!tight.admits(100, 99));

        assert_eq!(RewindLimits::positional(0, 4, 0).max_advance(), Some(usize::MAX));
    }

    #[test]
    fn a_target_of_zero_or_one_past_the_frontier_is_refused() {
        let lim = RewindLimits::positional(1024, 0, 0);
        assert!(!lim.admits(10, 0), "zero is not a reuse, it is a cold start");
        assert!(!lim.admits(10, 11), "past the frontier there is no KV to keep");
        assert!(lim.admits(10, 10), "the frontier itself is the exact-extend case");
    }

    #[test]
    fn the_exact_extend_case_needs_the_decoder_to_be_standing_on_the_frontier() {

        assert_eq!(exact_extend_target(8, 8, 8), Some(8));
        assert_eq!(exact_extend_target(8, 12, 8), Some(8), "lcp past the frontier still extends");
        assert_eq!(exact_extend_target(8, 7, 8), None, "lcp short of the frontier diverged");
        assert_eq!(
            exact_extend_target(8, 8, 9),
            None,
            "the decoder moved past the frontier, so its KV is no longer the snapshot's"
        );
        assert_eq!(exact_extend_target(0, 0, 0), None, "nothing to extend from");
    }
}
