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

    pub fn admits_from_high_water(
        &self,
        frontier: usize,
        high_water: usize,
        target: usize,
    ) -> bool {
        target <= frontier && self.admits(frontier.max(high_water), target)
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

pub fn saved_state_resume_point(positions_ascending: &[usize], prev: &[u32], prompt: &[u32]) -> usize {
    let lcp = common_prefix_len(prev, prompt);
    positions_ascending.iter().rev().copied().find(|&p| p < lcp).unwrap_or(0)
}

pub fn session_source(home: usize, reusable: [usize; 2]) -> usize {
    if home == 1 && reusable[0] > reusable[1] {
        0
    } else {
        home
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_target_never_reaches_past_the_shared_tokens_and_the_ring_reserves_a_slot_so_a_rewind_cannot_eat_its_own_window() {
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

        let lim = RewindLimits::positional(8, 4, 0);
        assert_eq!(
            lim.max_advance(),
            Some(8 + 1 - (4 + RING_REWIND_RESERVED_SLOTS))
        );
        assert!(lim.admits(100, 96), "a 4-token rewind fits");
        assert!(!lim.admits(100, 95), "a 5-token rewind exceeds the ring");

        let tight = RewindLimits::positional(2, 8, 0);
        assert_eq!(tight.max_advance(), Some(0), "saturating, never wrapped");
        assert!(!tight.admits(100, 99));

        assert_eq!(
            RewindLimits::positional(0, 4, 0).max_advance(),
            Some(usize::MAX)
        );

    }

    #[test]
    fn a_saved_state_resumes_strictly_below_the_shared_prefix() {
        let prev: Vec<u32> = (0..10).collect();
        let ckpts = [2usize, 4, 6, 8];
        let diverge_at = |i: usize| -> Vec<u32> {
            let mut p = prev.clone();
            p[i] = 99;
            p
        };
        assert_eq!(saved_state_resume_point(&ckpts, &prev, &prev), 8);
        assert_eq!(
            saved_state_resume_point(&ckpts, &prev, &diverge_at(6)),
            4,
            "the checkpoint at 6 holds the MTP row for position 5, which read token 6"
        );
        assert_eq!(saved_state_resume_point(&ckpts, &prev, &diverge_at(7)), 6);
        assert_eq!(saved_state_resume_point(&ckpts, &prev, &prev[..8]), 6, "a prompt ending on a checkpoint still needs its last token stepped");
        assert_eq!(saved_state_resume_point(&ckpts, &prev, &diverge_at(0)), 0);
        assert_eq!(saved_state_resume_point(&[], &prev, &prev), 0);
        assert_eq!(saved_state_resume_point(&ckpts, &[], &prev), 0, "an invalidated stream reuses nothing");
    }

    #[test]
    fn a_short_prompt_never_takes_the_long_state_and_a_growing_conversation_carries_its_own_up() {
        assert_eq!(session_source(0, [0, 30720]), 0, "a side call with a short prompt must not overwrite the long conversation");
        assert_eq!(session_source(0, [0, 4096]), 0, "a short prompt sharing the long state's system prompt still stays short");
        assert_eq!(session_source(1, [8192, 0]), 0, "a conversation crossing into the long state brings its checkpoints");
        assert_eq!(session_source(1, [8192, 2048]), 0);
        assert_eq!(session_source(1, [0, 38912]), 1);
        assert_eq!(session_source(1, [2048, 2048]), 1, "a tie keeps the long state");
        assert_eq!(session_source(1, [0, 0]), 1);
    }
}
