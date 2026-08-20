use anyhow::{bail, Result};

pub const MIN_RECOMPUTED_PREFILL_TOKENS: usize = 1;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SequenceState {
    Waiting,
    Prefill,
    Decode,
    Finished,
    Aborted,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum FinishReason {
    MaxTokens,
    Eos,
    Aborted,
}

#[derive(Debug)]
pub struct Sequence {
    pub id: u64,
    pub prompt: Vec<u32>,
    pub output: Vec<u32>,
    pub state: SequenceState,
    pub block_table: Vec<u32>,
    pub max_new_tokens: usize,
    pub eos_token_id: Option<u32>,
    pub finish_reason: Option<FinishReason>,
    pub draft_tokens: Vec<u32>,
    pub accepted_token_count: usize,
    pub num_computed_tokens: usize,
}

impl Sequence {
    pub fn new(id: u64, prompt: Vec<u32>, max_new_tokens: usize) -> Self {
        Self {
            id,
            prompt,
            output: Vec::new(),
            state: SequenceState::Waiting,
            block_table: Vec::new(),
            max_new_tokens,
            eos_token_id: None,
            finish_reason: None,
            draft_tokens: Vec::new(),
            accepted_token_count: 0,
            num_computed_tokens: 0,
        }
    }

    pub fn with_eos(mut self, eos_token_id: Option<u32>) -> Self {
        self.eos_token_id = eos_token_id;
        self
    }

    pub fn total_len(&self) -> usize {
        self.prompt.len() + self.output.len()
    }

    pub fn prompt_len(&self) -> usize {
        self.prompt.len()
    }

    pub fn output_len(&self) -> usize {
        self.output.len()
    }

    pub fn tokens(&self) -> Vec<u32> {
        let mut out = Vec::with_capacity(self.total_len());
        out.extend_from_slice(&self.prompt);
        out.extend_from_slice(&self.output);
        out
    }

    pub fn last_token(&self) -> Option<u32> {
        self.output
            .last()
            .copied()
            .or_else(|| self.prompt.last().copied())
    }

    pub fn is_finished(&self) -> bool {
        matches!(self.state, SequenceState::Finished | SequenceState::Aborted)
    }

    pub fn has_drafts(&self) -> bool {
        !self.draft_tokens.is_empty()
    }

    pub fn draft_len(&self) -> usize {
        self.draft_tokens.len()
    }

    pub fn set_drafts(&mut self, drafts: Vec<u32>) {
        self.draft_tokens = drafts;
    }

    pub fn clear_drafts(&mut self) {
        self.draft_tokens.clear();
    }

    pub fn remaining_uncomputed(&self) -> usize {
        self.total_len().saturating_sub(self.num_computed_tokens)
    }

    pub fn record_computed(&mut self, n: usize) {
        self.num_computed_tokens += n;
    }

    pub fn prefill_complete(&self) -> bool {
        self.num_computed_tokens >= self.total_len()
    }

    pub fn reset_for_recompute(&mut self) {
        self.num_computed_tokens = 0;
        self.state = SequenceState::Waiting;
    }

    pub fn num_blocks_needed(&self, block_size: usize) -> usize {
        let n = self.total_len();
        if n == 0 {
            0
        } else {
            n.div_ceil(block_size)
        }
    }

    pub fn block_tokens(&self, idx: usize, block_size: usize) -> Vec<u32> {
        let start = idx * block_size;
        let end = (start + block_size).min(self.total_len());
        let mut out = Vec::with_capacity(end - start);
        let prompt_len = self.prompt.len();
        for i in start..end {
            if i < prompt_len {
                out.push(self.prompt[i]);
            } else {
                out.push(self.output[i - prompt_len]);
            }
        }
        out
    }

    pub fn transition_to_prefill(&mut self) -> Result<()> {
        self.transition_to_prefill_with_cache(0)
    }

    pub fn transition_to_prefill_with_cache(&mut self, cached_tokens: usize) -> Result<()> {
        match self.state {
            SequenceState::Waiting => {
                self.state = SequenceState::Prefill;
                self.num_computed_tokens = cached_tokens.min(
                    self.total_len()
                        .saturating_sub(MIN_RECOMPUTED_PREFILL_TOKENS),
                );
                Ok(())
            }
            _ => bail!("invalid transition to Prefill from {:?}", self.state),
        }
    }

    pub fn transition_to_decode(&mut self) -> Result<()> {
        match self.state {
            SequenceState::Prefill | SequenceState::Decode => {
                self.state = SequenceState::Decode;
                Ok(())
            }
            _ => bail!("invalid transition to Decode from {:?}", self.state),
        }
    }

    pub fn append_token(&mut self, tok: u32) -> Result<()> {
        match self.state {
            SequenceState::Decode => {
                self.output.push(tok);
                Ok(())
            }
            _ => bail!("append_token only valid in Decode, state={:?}", self.state),
        }
    }

    pub fn check_stop(&mut self) -> Option<FinishReason> {
        if self.output.len() >= self.max_new_tokens {
            return Some(FinishReason::MaxTokens);
        }
        if let (Some(eos), Some(last)) = (self.eos_token_id, self.output.last().copied()) {
            if last == eos {
                return Some(FinishReason::Eos);
            }
        }
        None
    }

    pub fn finish(&mut self, reason: FinishReason) {
        self.state = SequenceState::Finished;
        self.finish_reason = Some(reason);
    }

    pub fn abort(&mut self) {
        self.state = SequenceState::Aborted;
        self.finish_reason = Some(FinishReason::Aborted);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    impl Sequence {
        fn with_eos_for_test(mut self, eos: Option<u32>) -> Self {
            self.eos_token_id = eos;
            self
        }
    }

    fn seq(prompt: &[u32], max_new: usize) -> Sequence {
        Sequence::new(1, prompt.to_vec(), max_new)
    }

    fn decoding(prompt: &[u32], max_new: usize) -> Sequence {
        let mut s = seq(prompt, max_new);
        s.transition_to_prefill().unwrap();
        s.transition_to_decode().unwrap();
        s
    }

    #[test]
    fn a_full_cache_hit_still_recomputes_one_token() {

        let mut s = seq(&[1, 2, 3, 4], 8);
        s.transition_to_prefill_with_cache(usize::MAX).unwrap();
        assert_eq!(s.num_computed_tokens, 4 - MIN_RECOMPUTED_PREFILL_TOKENS);
        assert!(!s.prefill_complete(), "a clamped prefill still has work to do");
        assert_eq!(s.remaining_uncomputed(), MIN_RECOMPUTED_PREFILL_TOKENS);

        let mut s = seq(&[1, 2, 3, 4], 8);
        s.transition_to_prefill_with_cache(2).unwrap();
        assert_eq!(s.num_computed_tokens, 2);

        let mut s = seq(&[9], 8);
        s.transition_to_prefill_with_cache(5).unwrap();
        assert_eq!(s.num_computed_tokens, 0);
    }

    #[test]
    fn the_state_machine_refuses_every_transition_it_does_not_name() {
        let mut s = seq(&[1, 2], 4);
        assert!(s.transition_to_decode().is_err(), "Waiting -> Decode skips prefill");
        assert!(s.append_token(7).is_err(), "a token before Decode has no KV behind it");

        s.transition_to_prefill().unwrap();
        assert!(s.transition_to_prefill().is_err(), "Prefill -> Prefill would rewind progress");
        s.transition_to_decode().unwrap();
        s.transition_to_decode().unwrap();
        s.append_token(7).unwrap();

        s.finish(FinishReason::Eos);
        assert!(s.is_finished());
        assert!(s.append_token(8).is_err(), "a finished sequence must not grow");
        assert!(s.transition_to_prefill().is_err());
    }

    #[test]
    fn max_tokens_is_checked_before_eos_so_the_cap_is_never_overrun() {
        let mut s = decoding(&[1], 2).with_eos_for_test(Some(99));
        s.append_token(5).unwrap();
        assert_eq!(s.check_stop(), None, "one of two");
        s.append_token(99).unwrap();

        assert_eq!(
            s.check_stop(),
            Some(FinishReason::MaxTokens),
            "both conditions hold; Eos here would say the model chose to stop when it was cut off"
        );

        let mut s = decoding(&[1], 8).with_eos_for_test(Some(99));
        s.append_token(99).unwrap();
        assert_eq!(s.check_stop(), Some(FinishReason::Eos));

        let mut s = decoding(&[99], 8).with_eos_for_test(Some(99));
        assert_eq!(
            s.check_stop(),
            None,
            "eos is only read from generated output; a prompt ending in eos must not stop the \
             sequence before it has produced anything"
        );
    }

    #[test]
    fn a_block_spans_the_prompt_output_seam_without_dropping_a_token() {
        let mut s = decoding(&[1, 2, 3], 8);
        for t in [4, 5, 6, 7] {
            s.append_token(t).unwrap();
        }
        assert_eq!(s.tokens(), vec![1, 2, 3, 4, 5, 6, 7]);
        assert_eq!(s.block_tokens(0, 4), vec![1, 2, 3, 4], "the seam is inside this block");
        assert_eq!(s.block_tokens(1, 4), vec![5, 6, 7], "the tail block is short, not padded");
        assert_eq!(s.num_blocks_needed(4), 2);
        assert_eq!(s.num_blocks_needed(7), 1, "an exact fit is one block, not two");
        assert_eq!(seq(&[], 4).num_blocks_needed(4), 0, "nothing needs no block");
    }

    #[test]
    fn last_token_falls_back_to_the_prompt_before_anything_is_generated() {

        let s = seq(&[1, 2, 3], 4);
        assert_eq!(s.last_token(), Some(3));
        let mut s = decoding(&[1, 2, 3], 4);
        s.append_token(9).unwrap();
        assert_eq!(s.last_token(), Some(9));
        assert_eq!(seq(&[], 4).last_token(), None);
    }

    #[test]
    fn a_recompute_sends_the_sequence_back_to_waiting_with_no_progress_kept() {
        let mut s = seq(&[1, 2, 3], 8);
        s.transition_to_prefill_with_cache(2).unwrap();
        s.record_computed(1);
        assert_eq!(s.num_computed_tokens, 3);
        s.reset_for_recompute();
        assert_eq!(s.num_computed_tokens, 0, "kept progress would skip real work");
        assert_eq!(s.state, SequenceState::Waiting, "and it must be schedulable again");
        s.transition_to_prefill().unwrap();
    }

    #[test]
    fn drafts_are_a_transient_and_abort_records_its_own_reason() {
        let mut s = decoding(&[1], 8);
        assert!(!s.has_drafts());
        s.set_drafts(vec![4, 5]);
        assert!(s.has_drafts() && s.draft_len() == 2);
        s.clear_drafts();
        assert!(!s.has_drafts() && s.draft_len() == 0);

        s.abort();
        assert!(s.is_finished(), "aborted counts as finished for the scheduler");
        assert_eq!(s.finish_reason, Some(FinishReason::Aborted));
    }
}
