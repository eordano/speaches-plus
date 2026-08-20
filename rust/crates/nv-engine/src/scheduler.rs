use crate::block_manager::BlockManager;
use crate::sequence::{FinishReason, Sequence, SequenceState};
use anyhow::{bail, Result};
use std::collections::{HashMap, VecDeque};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum BatchKind {
    Prefill,
    Decode,
    Mixed,
    Verify,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ScheduledSeqItem {
    pub seq_id: u64,
    pub num_scheduled_tokens: usize,
    pub is_prefill: bool,
    pub is_final_prefill_chunk: bool,
}

impl ScheduledSeqItem {
    pub fn produces_token(&self) -> bool {
        !self.is_prefill || self.is_final_prefill_chunk
    }
}

#[derive(Debug)]
pub struct ScheduledBatch {
    pub kind: BatchKind,
    pub seq_ids: Vec<u64>,
    pub items: Vec<ScheduledSeqItem>,
}

#[derive(Clone, Debug)]
pub struct StepFailure {
    pub seq_id: u64,
    pub message: String,
}

#[derive(Clone, Debug)]
pub struct VerifyOutcome {
    pub accepted_count: usize,
    pub bonus_token: u32,
    pub accepted_drafts: Vec<u32>,
}

#[derive(Clone, Copy, Debug)]
pub struct SchedulerConfig {
    pub max_batch_size: usize,
    pub max_batched_tokens: usize,
    pub block_size: usize,
    pub num_blocks: usize,
}

pub struct Scheduler {
    pub config: SchedulerConfig,
    pub block_manager: BlockManager,
    pub waiting: VecDeque<Sequence>,
    pub running: VecDeque<Sequence>,
    pub finished: Vec<Sequence>,
    pub last_batch: Option<ScheduledBatch>,
    pub num_running_with_drafts: usize,
    pub prefix_cache_reuse: bool,
    every_eos_id_by_seq: HashMap<u64, Vec<u32>>,
}

pub const PREFILL_CHUNK_MIN_ENV: &str = "NV_PREFILL_CHUNK_MIN";

pub const PREFILL_CHUNK_ROWS_BELOW_WHICH_THE_MODEL_SWITCHES_PRECISION_256: usize = 256;

pub fn prefill_chunk_min() -> usize {
    static ON: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *ON.get_or_init(|| match std::env::var(PREFILL_CHUNK_MIN_ENV).as_deref() {
        Ok("1") => PREFILL_CHUNK_ROWS_BELOW_WHICH_THE_MODEL_SWITCHES_PRECISION_256,
        Ok(v) => v.parse().unwrap_or(0),
        Err(_) => 0,
    })
}

pub fn floored_take(remaining: usize, budget: usize, floor: usize) -> usize {
    let take = remaining.min(budget);
    if floor <= 1 || take == remaining || take < floor {
        return take;
    }
    if remaining - take >= floor {
        return take;
    }
    let pulled = remaining - floor;
    if pulled >= floor {
        pulled
    } else {
        take
    }
}

impl Scheduler {
    pub fn new(config: SchedulerConfig) -> Self {
        let block_manager = BlockManager::new(config.num_blocks, config.block_size);
        Self {
            config,
            block_manager,
            waiting: VecDeque::new(),
            running: VecDeque::new(),
            finished: Vec::new(),
            last_batch: None,
            num_running_with_drafts: 0,
            prefix_cache_reuse: true,
            every_eos_id_by_seq: HashMap::new(),
        }
    }

    pub fn enqueue(&mut self, seq: Sequence) {
        self.waiting.push_back(seq);
    }

    pub fn enqueue_with_eos_ids(&mut self, seq: Sequence, eos_token_ids: Vec<u32>) {
        self.every_eos_id_by_seq.insert(seq.id, eos_token_ids);
        self.enqueue(seq);
    }

    fn stop_reason(&self, seq: &mut Sequence) -> Option<FinishReason> {
        if let Some(reason) = seq.check_stop() {
            return Some(reason);
        }
        let last = seq.output.last().copied()?;
        self.every_eos_id_by_seq
            .get(&seq.id)
            .filter(|ids| ids.contains(&last))
            .map(|_| FinishReason::Eos)
    }

    pub fn note_drafts_set(&mut self, had_drafts: bool, has_drafts: bool) {
        if has_drafts && !had_drafts {
            self.num_running_with_drafts += 1;
        } else if had_drafts && !has_drafts {
            self.num_running_with_drafts -= 1;
        }
    }

    pub fn set_drafts(&mut self, seq_id: u64, drafts: Vec<u32>) -> Result<()> {
        let (had, has) = {
            let seq = self
                .running
                .iter_mut()
                .find(|s| s.id == seq_id)
                .ok_or_else(|| anyhow::anyhow!("set_drafts: sequence {} not running", seq_id))?;
            let had = seq.has_drafts();
            seq.set_drafts(drafts);
            (had, seq.has_drafts())
        };
        self.note_drafts_set(had, has);
        Ok(())
    }

    pub fn has_work(&self) -> bool {
        !self.waiting.is_empty() || !self.running.is_empty()
    }

    pub fn abort(&mut self, seq_id: u64) {
        if let Some(idx) = self.waiting.iter().position(|s| s.id == seq_id) {
            let mut seq = self.waiting.remove(idx).unwrap();
            seq.abort();
            self.finished.push(seq);
            return;
        }
        if let Some(idx) = self.running.iter().position(|s| s.id == seq_id) {
            let mut seq = self.running.remove(idx).unwrap();
            if seq.has_drafts() {
                self.num_running_with_drafts -= 1;
                seq.clear_drafts();
            }
            self.block_manager.deallocate(&seq);
            seq.abort();
            self.finished.push(seq);
        }
    }

    pub fn step(&mut self) -> Result<ScheduledBatch> {
        let batch = if self.has_verify_ready_drafts() {
            self.build_verify_batch()
        } else {
            self.build_mixed_batch()?
        };
        self.last_batch = Some(batch.clone());
        Ok(batch)
    }

    fn has_verify_ready_drafts(&self) -> bool {
        self.num_running_with_drafts > 0
            && self.running.iter().any(|s| {
                !s.is_finished() && s.has_drafts() && matches!(s.state, SequenceState::Decode)
            })
    }

    fn build_mixed_batch(&mut self) -> Result<ScheduledBatch> {
        let mut items: Vec<ScheduledSeqItem> = Vec::new();
        let mut budget = self.config.max_batched_tokens;

        for seq in self.running.iter() {
            if budget == 0 || items.len() >= self.config.max_batch_size {
                break;
            }
            if seq.is_finished() || seq.has_drafts() {
                continue;
            }
            if !matches!(seq.state, SequenceState::Decode) {
                continue;
            }
            items.push(ScheduledSeqItem {
                seq_id: seq.id,
                num_scheduled_tokens: 1,
                is_prefill: false,
                is_final_prefill_chunk: false,
            });
            budget -= 1;
        }

        for seq in self.running.iter() {
            if budget == 0 || items.len() >= self.config.max_batch_size {
                break;
            }
            if seq.is_finished() || !matches!(seq.state, SequenceState::Prefill) {
                continue;
            }
            let remaining = seq.remaining_uncomputed();
            if remaining == 0 {
                continue;
            }
            let take = floored_take(remaining, budget, prefill_chunk_min());
            if take == 0 {
                continue;
            }
            items.push(ScheduledSeqItem {
                seq_id: seq.id,
                num_scheduled_tokens: take,
                is_prefill: true,
                is_final_prefill_chunk: take == remaining,
            });
            budget -= take;
        }

        while budget > 0 && items.len() < self.config.max_batch_size {
            let Some(front) = self.waiting.front() else {
                break;
            };
            if front.total_len() == 0 {
                let mut s = self.waiting.pop_front().unwrap();
                s.abort();
                self.finished.push(s);
                continue;
            }
            let mut seq = self.waiting.pop_front().unwrap();
            let cached = match self.block_manager.allocate_with_prefix(&seq) {
                Ok(alloc) => {
                    seq.block_table = alloc.block_table;
                    if self.prefix_cache_reuse {
                        alloc.cached_tokens
                    } else {
                        0
                    }
                }
                Err(_) => {
                    self.waiting.push_front(seq);
                    break;
                }
            };
            seq.transition_to_prefill_with_cache(cached)?;
            let remaining = seq.remaining_uncomputed();
            let take = floored_take(remaining, budget, prefill_chunk_min());
            if take == 0 {
                self.running.push_back(seq);
                break;
            }
            items.push(ScheduledSeqItem {
                seq_id: seq.id,
                num_scheduled_tokens: take,
                is_prefill: true,
                is_final_prefill_chunk: take == remaining,
            });
            budget -= take;
            self.running.push_back(seq);
        }

        let seq_ids = items.iter().map(|it| it.seq_id).collect();
        Ok(ScheduledBatch {
            kind: Self::kind_for_items(&items),
            seq_ids,
            items,
        })
    }

    fn kind_for_items(items: &[ScheduledSeqItem]) -> BatchKind {
        let any_prefill = items.iter().any(|it| it.is_prefill);
        let any_decode = items.iter().any(|it| !it.is_prefill);
        match (any_prefill, any_decode) {
            (true, true) => BatchKind::Mixed,
            (true, false) => BatchKind::Prefill,
            _ => BatchKind::Decode,
        }
    }

    fn build_verify_batch(&mut self) -> ScheduledBatch {
        let candidates: Vec<u64> = self
            .running
            .iter()
            .filter(|s| !s.is_finished() && matches!(s.state, SequenceState::Decode))
            .map(|s| s.id)
            .collect();
        let mut seq_ids = Vec::new();
        for sid in candidates {
            if seq_ids.len() >= self.config.max_batch_size {
                break;
            }
            let Some(idx) = self.running.iter().position(|s| s.id == sid) else {
                continue;
            };
            let extra = 1 + self.running[idx].draft_len();
            let mut seq = self.running.remove(idx).unwrap();
            match self.block_manager.extend_for_slots(&mut seq, extra) {
                Ok(()) => {
                    self.running.insert(idx, seq);
                    seq_ids.push(sid);
                }
                Err(_) => {
                    if seq.has_drafts() {
                        self.num_running_with_drafts -= 1;
                    }
                    seq.clear_drafts();
                    self.block_manager.deallocate(&seq);
                    seq.block_table.clear();
                    seq.state = SequenceState::Waiting;
                    self.waiting.push_front(seq);
                }
            }
        }
        ScheduledBatch {
            kind: BatchKind::Verify,
            seq_ids,
            items: Vec::new(),
        }
    }

    pub fn complete_step(&mut self, sampled_tokens: &[u32]) -> Result<Vec<StepFailure>> {
        let Some(batch) = self.last_batch.take() else {
            bail!("complete_step called without a prior step()");
        };
        if batch.kind == BatchKind::Verify {
            bail!("verify batch must be completed via complete_verify_step");
        }
        let expected = batch.items.iter().filter(|it| it.produces_token()).count();
        if expected != sampled_tokens.len() {
            bail!(
                "complete_step token count {} != token-producing item count {}",
                sampled_tokens.len(),
                expected
            );
        }
        let mut failures = Vec::new();
        let mut tokens = sampled_tokens.iter();
        let mut needs_block_extension: Vec<u64> = Vec::new();
        for item in &batch.items {
            let tok = if item.produces_token() {
                Some(*tokens.next().unwrap())
            } else {
                None
            };
            match self.apply_scheduled_item(item, tok) {
                Ok(true) => needs_block_extension.push(item.seq_id),
                Ok(false) => {}
                Err(e) => failures.push(StepFailure {
                    seq_id: item.seq_id,
                    message: e.to_string(),
                }),
            }
        }
        for sid in needs_block_extension {
            if let Some(f) = self.extend_blocks_or_preempt(sid) {
                failures.push(f);
            }
        }
        Ok(failures)
    }

    fn apply_scheduled_item(&mut self, item: &ScheduledSeqItem, tok: Option<u32>) -> Result<bool> {
        let idx = self
            .running
            .iter()
            .position(|s| s.id == item.seq_id)
            .ok_or_else(|| anyhow::anyhow!("sequence {} not running", item.seq_id))?;
        let mut seq = self.running.remove(idx).unwrap();

        if item.is_prefill {
            seq.record_computed(item.num_scheduled_tokens);
        }
        let Some(tok) = tok else {
            self.running.insert(idx, seq);
            return Ok(false);
        };

        let mut outcome = if item.is_final_prefill_chunk {
            seq.transition_to_decode()
        } else {
            Ok(())
        };
        if outcome.is_ok() {
            outcome = seq.append_token(tok);
        }
        if let Err(e) = outcome {
            self.block_manager.deallocate(&seq);
            seq.abort();
            self.finished.push(seq);
            return Err(e);
        }
        seq.record_computed(1);
        let valid_tokens = seq.total_len().saturating_sub(1);
        self.block_manager.mark_kv_computed(&seq, valid_tokens);
        if self.prefix_cache_reuse {
            self.block_manager
                .publish_computed_blocks(&seq, valid_tokens);
        }

        if let Some(reason) = self.stop_reason(&mut seq) {
            self.block_manager.deallocate(&seq);
            seq.finish(reason);
            self.finished.push(seq);
            Ok(false)
        } else {
            self.running.insert(idx, seq);
            Ok(true)
        }
    }

    fn extend_blocks_or_preempt(&mut self, sid: u64) -> Option<StepFailure> {
        let idx = self.running.iter().position(|s| s.id == sid)?;
        let mut seq = self.running.remove(idx).unwrap();
        loop {
            match self.block_manager.extend_for(&mut seq) {
                Ok(()) => {
                    let at = idx.min(self.running.len());
                    self.running.insert(at, seq);
                    return None;
                }
                Err(e) => {
                    if seq.num_blocks_needed(self.config.block_size) > self.config.num_blocks {
                        self.block_manager.deallocate(&seq);
                        if seq.has_drafts() {
                            self.num_running_with_drafts -= 1;
                            seq.clear_drafts();
                        }
                        let seq_id = seq.id;
                        seq.abort();
                        self.finished.push(seq);
                        return Some(StepFailure {
                            seq_id,
                            message: e.to_string(),
                        });
                    }
                    let has_younger_running_seq = idx < self.running.len();
                    if has_younger_running_seq {
                        let victim = self.running.pop_back().unwrap();
                        self.preempt_to_waiting_for_recompute(victim);
                    } else {
                        self.preempt_to_waiting_for_recompute(seq);
                        return None;
                    }
                }
            }
        }
    }

    fn preempt_to_waiting_for_recompute(&mut self, mut seq: Sequence) {
        if seq.has_drafts() {
            self.num_running_with_drafts -= 1;
            seq.clear_drafts();
        }
        self.block_manager.deallocate(&seq);
        seq.block_table.clear();
        seq.reset_for_recompute();
        self.waiting.push_front(seq);
    }

    pub fn complete_verify_step(&mut self, outcomes: &[VerifyOutcome]) -> Result<Vec<StepFailure>> {
        let Some(batch) = self.last_batch.take() else {
            bail!("complete_verify_step called without a prior step()");
        };
        if batch.kind != BatchKind::Verify {
            bail!("complete_verify_step called for a {:?} batch", batch.kind);
        }
        if batch.seq_ids.len() != outcomes.len() {
            bail!(
                "complete_verify_step outcome count {} != batch size {}",
                outcomes.len(),
                batch.seq_ids.len()
            );
        }
        let mut failures = Vec::new();
        for (sid, outcome) in batch.seq_ids.iter().zip(outcomes.iter()) {
            if let Err(e) = self.apply_verify_outcome(*sid, outcome) {
                failures.push(StepFailure {
                    seq_id: *sid,
                    message: e.to_string(),
                });
            }
        }
        Ok(failures)
    }

    fn apply_verify_outcome(&mut self, sid: u64, outcome: &VerifyOutcome) -> Result<()> {
        let idx = self
            .running
            .iter()
            .position(|s| s.id == sid)
            .ok_or_else(|| anyhow::anyhow!("sequence {} not running", sid))?;

        let mut seq = self.running.remove(idx).unwrap();
        if seq.has_drafts() {
            self.num_running_with_drafts -= 1;
        }
        seq.accepted_token_count = outcome.accepted_count;
        seq.clear_drafts();

        let mut finished = false;
        let mut appended: Result<()> = Ok(());
        for &tok in outcome
            .accepted_drafts
            .iter()
            .chain(std::iter::once(&outcome.bonus_token))
        {
            appended = seq.append_token(tok);
            if appended.is_err() {
                break;
            }
            if let Some(reason) = self.stop_reason(&mut seq) {
                seq.finish(reason);
                finished = true;
                break;
            }
        }
        if appended.is_ok() && !finished {
            appended = self.block_manager.extend_for(&mut seq);
        }
        if let Err(e) = appended {
            self.block_manager.deallocate(&seq);
            seq.abort();
            self.finished.push(seq);
            return Err(e);
        }
        if finished {
            self.block_manager.deallocate(&seq);
            self.finished.push(seq);
        } else {
            self.running.insert(idx, seq);
        }
        Ok(())
    }

    pub fn running_seq(&self, seq_id: u64) -> Option<&Sequence> {
        self.running.iter().find(|s| s.id == seq_id)
    }

    pub fn drain_finished(&mut self) -> Vec<Sequence> {
        let drained = std::mem::take(&mut self.finished);
        for seq in &drained {
            self.every_eos_id_by_seq.remove(&seq.id);
        }
        drained
    }
}

impl Clone for ScheduledBatch {
    fn clone(&self) -> Self {
        Self {
            kind: self.kind,
            seq_ids: self.seq_ids.clone(),
            items: self.items.clone(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sequence::{FinishReason, Sequence};

    fn cfg() -> SchedulerConfig {
        SchedulerConfig {
            max_batch_size: 4,
            max_batched_tokens: 1024,
            block_size: 4,
            num_blocks: 16,
        }
    }

    #[test]
    fn first_step_is_prefill() {
        let mut sch = Scheduler::new(cfg());
        sch.enqueue(Sequence::new(1, vec![1, 2, 3, 4, 5], 4));
        let batch = sch.step().unwrap();
        assert_eq!(batch.kind, BatchKind::Prefill);
        assert_eq!(batch.seq_ids, vec![1]);
        assert_eq!(sch.running_seq(1).unwrap().state, SequenceState::Prefill);
    }

    #[test]
    fn second_step_is_decode() {
        let mut sch = Scheduler::new(cfg());
        sch.enqueue(Sequence::new(1, vec![1, 2, 3, 4, 5], 4));
        let _ = sch.step().unwrap();
        sch.complete_step(&[42]).unwrap();
        assert_eq!(sch.running_seq(1).unwrap().state, SequenceState::Decode);
        assert_eq!(sch.running_seq(1).unwrap().output, vec![42]);

        let batch = sch.step().unwrap();
        assert_eq!(batch.kind, BatchKind::Decode);
        assert_eq!(batch.seq_ids, vec![1]);
    }

    #[test]
    fn max_new_tokens_finishes_sequence() {
        let mut sch = Scheduler::new(cfg());
        sch.enqueue(Sequence::new(1, vec![1, 2, 3], 3));
        let _ = sch.step().unwrap();
        sch.complete_step(&[10]).unwrap();
        let _ = sch.step().unwrap();
        sch.complete_step(&[11]).unwrap();
        let _ = sch.step().unwrap();
        sch.complete_step(&[12]).unwrap();

        assert!(sch.running.is_empty());
        let done = sch.drain_finished();
        assert_eq!(done.len(), 1);
        assert_eq!(done[0].state, SequenceState::Finished);
        assert_eq!(done[0].finish_reason, Some(FinishReason::MaxTokens));
        assert_eq!(done[0].output, vec![10, 11, 12]);
    }

    #[test]
    fn eos_finishes_sequence_immediately() {
        let mut sch = Scheduler::new(cfg());
        let seq = Sequence::new(1, vec![1, 2, 3], 16).with_eos(Some(99));
        sch.enqueue(seq);
        let _ = sch.step().unwrap();
        sch.complete_step(&[99]).unwrap();
        assert!(sch.running.is_empty());
        let done = sch.drain_finished();
        assert_eq!(done.len(), 1);
        assert_eq!(done[0].finish_reason, Some(FinishReason::Eos));
        assert_eq!(done[0].output, vec![99]);
    }

    const EOS_IDS_A_31B_CHECKPOINT_CARRIES: [u32; 3] = [1, 50, 106];

    #[test]
    fn decode_finishes_on_a_non_first_eos_member() {
        let mut sch = Scheduler::new(cfg());
        sch.enqueue_with_eos_ids(
            Sequence::new(1, vec![1, 2, 3], 16),
            EOS_IDS_A_31B_CHECKPOINT_CARRIES.to_vec(),
        );
        let _ = sch.step().unwrap();
        sch.complete_step(&[50]).unwrap();
        assert!(
            sch.running.is_empty(),
            "50 is a member of {EOS_IDS_A_31B_CHECKPOINT_CARRIES:?} and must end the sequence"
        );
        let done = sch.drain_finished();
        assert_eq!(done.len(), 1);
        assert_eq!(done[0].finish_reason, Some(FinishReason::Eos));
        assert_eq!(done[0].output, vec![50]);
    }

    #[test]
    fn verify_finishes_on_a_non_first_eos_member() {
        let mut sch = Scheduler::new(cfg());
        sch.enqueue_with_eos_ids(
            Sequence::new(1, vec![1, 2, 3, 4, 5], 16),
            EOS_IDS_A_31B_CHECKPOINT_CARRIES.to_vec(),
        );
        let _ = sch.step().unwrap();
        sch.complete_step(&[100]).unwrap();
        sch.set_drafts(1, vec![106, 203]).unwrap();
        let _ = sch.step().unwrap();
        sch.complete_verify_step(&[VerifyOutcome {
            accepted_count: 2,
            bonus_token: 5,
            accepted_drafts: vec![106, 203],
        }])
        .unwrap();
        assert!(
            sch.running.is_empty(),
            "106 is a member of {EOS_IDS_A_31B_CHECKPOINT_CARRIES:?} and must end the verify batch"
        );
        let done = sch.drain_finished();
        assert_eq!(done.len(), 1);
        assert_eq!(done[0].finish_reason, Some(FinishReason::Eos));
        assert_eq!(done[0].output, vec![100, 106]);
    }

    fn decoding_seq(sch: &mut Scheduler, id: u64, prompt: Vec<u32>, max_new: usize) {
        sch.enqueue(Sequence::new(id, prompt, max_new));
        let batch = sch.step().unwrap();
        let producing = batch.items.iter().filter(|it| it.produces_token()).count();
        sch.complete_step(&vec![100; producing]).unwrap();
        assert_eq!(sch.running_seq(id).unwrap().state, SequenceState::Decode);
    }

    #[test]
    fn note_drafts_set_counter_bookkeeping() {
        let mut sch = Scheduler::new(cfg());
        assert_eq!(sch.num_running_with_drafts, 0);
        sch.note_drafts_set(false, true);
        assert_eq!(sch.num_running_with_drafts, 1);
        sch.note_drafts_set(true, true);
        assert_eq!(sch.num_running_with_drafts, 1);
        sch.note_drafts_set(true, false);
        assert_eq!(sch.num_running_with_drafts, 0);
    }

    #[test]
    fn set_drafts_triggers_verify_mode() {
        let mut sch = Scheduler::new(cfg());
        decoding_seq(&mut sch, 1, vec![1, 2, 3, 4, 5], 16);

        let batch = sch.step().unwrap();
        assert_eq!(batch.kind, BatchKind::Decode);
        sch.complete_step(&[101]).unwrap();

        sch.set_drafts(1, vec![201, 202]).unwrap();
        assert_eq!(sch.num_running_with_drafts, 1);

        let batch = sch.step().unwrap();
        assert_eq!(batch.kind, BatchKind::Verify);
        assert_eq!(batch.seq_ids, vec![1]);
    }

    #[test]
    fn verify_partial_acceptance_appends_and_advances() {
        let mut sch = Scheduler::new(cfg());
        decoding_seq(&mut sch, 1, vec![1, 2, 3, 4, 5], 16);
        sch.set_drafts(1, vec![201, 202, 203]).unwrap();
        let batch = sch.step().unwrap();
        assert_eq!(batch.kind, BatchKind::Verify);

        sch.complete_verify_step(&[VerifyOutcome {
            accepted_count: 1,
            bonus_token: 999,
            accepted_drafts: vec![201],
        }])
        .unwrap();

        let seq = sch.running_seq(1).unwrap();
        assert_eq!(seq.output, vec![100, 201, 999]);
        assert_eq!(seq.accepted_token_count, 1);
        assert!(!seq.has_drafts());
        assert_eq!(sch.num_running_with_drafts, 0);

        let batch = sch.step().unwrap();
        assert_eq!(batch.kind, BatchKind::Decode);
    }

    #[test]
    fn verify_full_rejection_keeps_only_bonus() {
        let mut sch = Scheduler::new(cfg());
        decoding_seq(&mut sch, 1, vec![1, 2, 3, 4, 5], 16);
        sch.set_drafts(1, vec![201, 202]).unwrap();
        let _ = sch.step().unwrap();
        sch.complete_verify_step(&[VerifyOutcome {
            accepted_count: 0,
            bonus_token: 777,
            accepted_drafts: vec![],
        }])
        .unwrap();
        let seq = sch.running_seq(1).unwrap();
        assert_eq!(seq.output, vec![100, 777]);
        assert_eq!(sch.num_running_with_drafts, 0);
    }

    #[test]
    fn verify_eos_in_drafts_finishes_sequence() {
        let mut sch = Scheduler::new(cfg());
        sch.enqueue(Sequence::new(1, vec![1, 2, 3, 4, 5], 16).with_eos(Some(42)));
        let _ = sch.step().unwrap();
        sch.complete_step(&[100]).unwrap();
        sch.set_drafts(1, vec![42, 203]).unwrap();
        let _ = sch.step().unwrap();
        sch.complete_verify_step(&[VerifyOutcome {
            accepted_count: 2,
            bonus_token: 5,
            accepted_drafts: vec![42, 203],
        }])
        .unwrap();
        assert!(sch.running.is_empty());
        let done = sch.drain_finished();
        assert_eq!(done.len(), 1);
        assert_eq!(done[0].finish_reason, Some(FinishReason::Eos));
        assert_eq!(done[0].output, vec![100, 42]);
        assert_eq!(sch.num_running_with_drafts, 0);
    }

    #[test]
    fn complete_step_rejects_verify_batch() {
        let mut sch = Scheduler::new(cfg());
        decoding_seq(&mut sch, 1, vec![1, 2, 3, 4, 5], 16);
        sch.set_drafts(1, vec![201]).unwrap();
        let _ = sch.step().unwrap();
        assert!(sch.complete_step(&[1]).is_err());
    }

    #[test]
    fn complete_step_isolates_aborted_sequence() {
        let mut sch = Scheduler::new(cfg());
        sch.enqueue(Sequence::new(1, vec![1, 2, 3], 8));
        sch.enqueue(Sequence::new(2, vec![4, 5, 6], 8));
        let batch = sch.step().unwrap();
        assert_eq!(batch.seq_ids, vec![1, 2]);

        sch.abort(1);
        let failures = sch.complete_step(&[10, 20]).unwrap();

        assert_eq!(failures.len(), 1);
        assert_eq!(failures[0].seq_id, 1);
        let seq2 = sch
            .running_seq(2)
            .expect("seq 2 must survive seq 1's failure");
        assert_eq!(seq2.output, vec![20]);
        assert_eq!(seq2.state, SequenceState::Decode);
    }

    #[test]
    fn preempt_youngest_instead_of_abort_on_block_exhaustion() {
        let mut sch = Scheduler::new(SchedulerConfig {
            max_batch_size: 4,
            max_batched_tokens: 1024,
            block_size: 1,
            num_blocks: 6,
        });
        sch.enqueue(Sequence::new(1, vec![1, 2], 8));
        sch.enqueue(Sequence::new(2, vec![3, 4], 8));
        let _ = sch.step().unwrap();
        assert!(sch.complete_step(&[10, 20]).unwrap().is_empty());
        assert_eq!(sch.block_manager.num_free(), 0);

        let batch = sch.step().unwrap();
        assert_eq!(batch.kind, BatchKind::Decode);
        assert_eq!(batch.seq_ids, vec![1, 2]);
        let failures = sch.complete_step(&[11, 21]).unwrap();

        assert!(failures.is_empty());
        let seq1 = sch.running_seq(1).expect("older seq keeps running");
        assert_eq!(seq1.output, vec![10, 11]);
        assert!(sch.running_seq(2).is_none());
        let victim = sch
            .waiting
            .front()
            .expect("youngest seq preempted to waiting");
        assert_eq!(victim.id, 2);
        assert_eq!(victim.state, SequenceState::Waiting);
        assert!(victim.block_table.is_empty());
        assert_eq!(victim.output, vec![20, 21]);
        assert!(sch.drain_finished().is_empty());
    }

    #[test]
    fn self_preemption_when_sole_running_seq() {
        let mut sch = Scheduler::new(SchedulerConfig {
            max_batch_size: 4,
            max_batched_tokens: 1024,
            block_size: 1,
            num_blocks: 5,
        });
        let mut hog = Sequence::new(99, vec![7, 8, 9], 1);
        hog.block_table = sch.block_manager.allocate_for(&hog).unwrap();

        sch.enqueue(Sequence::new(1, vec![1, 2], 8));
        let _ = sch.step().unwrap();
        let failures = sch.complete_step(&[10]).unwrap();

        assert!(failures.is_empty());
        assert!(sch.running.is_empty());
        assert!(sch.drain_finished().is_empty());
        let seq = sch
            .waiting
            .front()
            .expect("sole seq self-preempts to waiting");
        assert_eq!(seq.id, 1);
        assert_eq!(seq.state, SequenceState::Waiting);
        assert_eq!(seq.output, vec![10]);
        assert!(seq.block_table.is_empty());

        sch.block_manager.deallocate(&hog);
        let batch = sch.step().unwrap();
        assert_eq!(batch.kind, BatchKind::Prefill);
        assert_eq!(batch.seq_ids, vec![1]);
        assert!(batch.items[0].is_final_prefill_chunk);
        sch.complete_step(&[11]).unwrap();
        let seq = sch.running_seq(1).unwrap();
        assert_eq!(seq.state, SequenceState::Decode);
        assert_eq!(seq.output, vec![10, 11]);
    }

    #[test]
    fn abort_when_sequence_exceeds_whole_pool() {
        let mut sch = Scheduler::new(SchedulerConfig {
            max_batch_size: 4,
            max_batched_tokens: 1024,
            block_size: 1,
            num_blocks: 2,
        });
        sch.enqueue(Sequence::new(1, vec![1, 2], 8));
        let _ = sch.step().unwrap();
        let failures = sch.complete_step(&[10]).unwrap();

        assert_eq!(failures.len(), 1);
        assert_eq!(failures[0].seq_id, 1);
        assert!(sch.waiting.is_empty());
        let done = sch.drain_finished();
        assert_eq!(done.len(), 1);
        assert_eq!(done[0].finish_reason, Some(FinishReason::Aborted));
    }

    #[test]
    fn complete_verify_step_isolates_aborted_sequence() {
        let mut sch = Scheduler::new(cfg());
        decoding_seq(&mut sch, 1, vec![1, 2, 3], 16);
        decoding_seq(&mut sch, 2, vec![4, 5, 6], 16);
        sch.set_drafts(1, vec![201]).unwrap();
        sch.set_drafts(2, vec![202]).unwrap();
        let batch = sch.step().unwrap();
        assert_eq!(batch.kind, BatchKind::Verify);
        assert_eq!(batch.seq_ids, vec![1, 2]);

        sch.abort(1);
        let failures = sch
            .complete_verify_step(&[
                VerifyOutcome {
                    accepted_count: 1,
                    bonus_token: 900,
                    accepted_drafts: vec![201],
                },
                VerifyOutcome {
                    accepted_count: 1,
                    bonus_token: 901,
                    accepted_drafts: vec![202],
                },
            ])
            .unwrap();

        assert_eq!(failures.len(), 1);
        assert_eq!(failures[0].seq_id, 1);
        let seq2 = sch
            .running_seq(2)
            .expect("seq 2 must survive seq 1's failure");
        assert_eq!(seq2.output, vec![100, 202, 901]);
        assert_eq!(sch.num_running_with_drafts, 0);
    }

    #[test]
    fn budget_accounting_with_recycled_sequence() {
        let mut sch = Scheduler::new(SchedulerConfig {
            max_batch_size: 4,
            max_batched_tokens: 6,
            block_size: 4,
            num_blocks: 16,
        });
        decoding_seq(&mut sch, 1, vec![1, 2, 3], 16);
        let mut seq = sch.running.pop_front().unwrap();
        sch.block_manager.deallocate(&seq);
        seq.block_table.clear();
        seq.state = SequenceState::Waiting;
        assert_eq!(seq.total_len(), 4);
        sch.waiting.push_front(seq);
        sch.enqueue(Sequence::new(2, vec![4, 5, 6], 16));

        let batch = sch.step().unwrap();
        assert_eq!(batch.kind, BatchKind::Prefill);
        assert_eq!(batch.seq_ids, vec![1, 2]);
        assert_eq!(
            batch.items[0],
            ScheduledSeqItem {
                seq_id: 1,
                num_scheduled_tokens: 4,
                is_prefill: true,
                is_final_prefill_chunk: true,
            },
            "recycled seq costs prompt + generated tokens against the batch budget"
        );
        assert_eq!(
            batch.items[1],
            ScheduledSeqItem {
                seq_id: 2,
                num_scheduled_tokens: 2,
                is_prefill: true,
                is_final_prefill_chunk: false,
            },
            "leftover budget admits a partial chunk of the next waiting seq"
        );
        let seq1 = sch.running_seq(1).unwrap();
        assert_eq!(
            seq1.block_table.len(),
            seq1.num_blocks_needed(sch.config.block_size)
        );

        sch.complete_step(&[500]).unwrap();
        assert_eq!(sch.running_seq(1).unwrap().output, vec![100, 500]);
        let seq2 = sch.running_seq(2).unwrap();
        assert_eq!(seq2.state, SequenceState::Prefill);
        assert_eq!(seq2.num_computed_tokens, 2);
    }

    #[test]
    fn shared_prefix_skips_recomputation_on_the_second_sequence() {
        let mut sch = Scheduler::new(cfg());
        let prefix: Vec<u32> = (1..=8).collect();
        sch.enqueue(Sequence::new(1, prefix.clone(), 16));
        let batch = sch.step().unwrap();
        assert_eq!(batch.items[0].num_scheduled_tokens, 8);
        sch.complete_step(&[500]).unwrap();

        let mut second = prefix.clone();
        second.extend_from_slice(&[70, 71]);
        sch.enqueue(Sequence::new(2, second, 16));
        let batch = sch.step().unwrap();
        let item = batch
            .items
            .iter()
            .find(|it| it.seq_id == 2)
            .expect("second sequence admitted");

        assert_eq!(
            sch.running_seq(2).unwrap().num_computed_tokens,
            8,
            "the two whole blocks shared with seq 1 already hold valid KV: prefill must start \
             past them, not recompute them"
        );
        assert_eq!(
            item.num_scheduled_tokens, 2,
            "only the uncached tail belongs in the prefill chunk"
        );
        assert!(item.is_final_prefill_chunk);

        sch.complete_step(&[600, 601]).unwrap();
        assert_eq!(sch.running_seq(2).unwrap().state, SequenceState::Decode);
        assert_eq!(sch.running_seq(2).unwrap().output, vec![601]);
    }

    #[test]
    fn a_second_turn_reuses_the_blocks_the_first_turn_decoded() {
        let mut sch = Scheduler::new(cfg());
        sch.enqueue(Sequence::new(1, vec![1, 2, 3], 16));
        let _ = sch.step().unwrap();
        sch.complete_step(&[500]).unwrap();
        for tok in 501..=507u32 {
            let _ = sch.step().unwrap();
            sch.complete_step(&[tok]).unwrap();
        }

        let turn1 = sch.running_seq(1).expect("first turn still running");
        assert_eq!(turn1.total_len(), 11);
        let next_prompt = turn1.tokens();
        let decoded: Vec<u32> = turn1.block_table[..2].to_vec();

        sch.enqueue(Sequence::new(2, next_prompt, 16));
        let batch = sch.step().unwrap();
        let item = *batch
            .items
            .iter()
            .find(|it| it.seq_id == 2)
            .expect("second turn admitted");
        let turn2 = sch.running_seq(2).unwrap();

        assert_eq!(
            turn2.num_computed_tokens, 8,
            "turn 2's prompt is turn 1's prompt plus turn 1's completion, so the whole shared \
             region was produced by decode; a prefix cache that only publishes prompt-aligned \
             blocks cannot see it"
        );
        assert_eq!(
            turn2.block_table[..2],
            decoded[..],
            "the reused blocks must be the very blocks decode filled, not fresh copies"
        );
        assert_eq!(
            item.num_scheduled_tokens, 3,
            "only the tokens past the last cached block belong in the prefill chunk"
        );
        assert!(item.is_final_prefill_chunk);

        sch.complete_step(&[508, 600]).unwrap();
        assert_eq!(sch.running_seq(2).unwrap().state, SequenceState::Decode);
        assert_eq!(sch.running_seq(2).unwrap().output, vec![600]);
    }

    #[test]
    fn a_stepper_that_cannot_resume_mid_prompt_publishes_nothing_new() {
        let mut sch = Scheduler::new(cfg());
        sch.prefix_cache_reuse = false;
        sch.enqueue(Sequence::new(1, vec![1, 2, 3], 16));
        let _ = sch.step().unwrap();
        sch.complete_step(&[500]).unwrap();
        for tok in 501..=507u32 {
            let _ = sch.step().unwrap();
            sch.complete_step(&[tok]).unwrap();
        }

        let next_prompt = sch.running_seq(1).unwrap().tokens();
        let decoded: Vec<u32> = sch.running_seq(1).unwrap().block_table[..2].to_vec();
        sch.enqueue(Sequence::new(2, next_prompt, 16));
        let _ = sch.step().unwrap();
        let turn2 = sch.running_seq(2).unwrap();

        assert_eq!(turn2.num_computed_tokens, 0);
        assert_ne!(
            turn2.block_table[0], decoded[0],
            "a stepper that cannot start a chunk past token 0 would re-prefill over the shared \
             block while turn 1 is still attending to it: publish only what reuse will consume"
        );
    }

    fn chunk_cfg(max_batched_tokens: usize) -> SchedulerConfig {
        SchedulerConfig {
            max_batch_size: 8,
            max_batched_tokens,
            block_size: 4,
            num_blocks: 64,
        }
    }

    pub const PREFILL_CHUNK_FLOOR_THAT_MAKES_A_PROMPT_REPRODUCIBLE_256: usize = 256;

    #[test]
    fn no_prefill_chunk_falls_under_the_floor_so_a_prompt_does_not_depend_on_its_neighbours() {
        let floor = PREFILL_CHUNK_FLOOR_THAT_MAKES_A_PROMPT_REPRODUCIBLE_256;
        let cfg = SchedulerConfig {
            max_batch_size: 8,
            max_batched_tokens: 1024,
            block_size: 16,
            num_blocks: 512,
        };
        let mut sch = Scheduler::new(cfg);
        sch.enqueue(Sequence::new(1, (1..=300u32).collect(), 16));
        sch.enqueue(Sequence::new(2, (1..=4000u32).collect(), 16));

        let batch = sch.step().unwrap();
        let observed: Vec<(u64, usize, bool)> = batch
            .items
            .iter()
            .map(|i| (i.seq_id, i.num_scheduled_tokens, i.is_final_prefill_chunk))
            .collect();
        eprintln!("[sched-align] batch = {observed:?}");
        assert!(
            batch
                .items
                .iter()
                .any(|i| i.is_prefill && !i.is_final_prefill_chunk),
            "no non-final prefill chunk in the batch {observed:?}, so this gate examined \
             nothing and would pass however the splitter behaves"
        );
        let under: Vec<(u64, usize)> = batch
            .items
            .iter()
            .filter(|i| i.is_prefill && !i.is_final_prefill_chunk)
            .filter(|i| i.num_scheduled_tokens < floor)
            .map(|i| (i.seq_id, i.num_scheduled_tokens))
            .collect();
        for (rem, bud, want) in [
            (4000usize, 724usize, 724usize),
            (4000, 1024, 1024),
            (4000, 3900, 3744),
            (4000, 255, 255),
            (300, 724, 300),
            (400, 300, 300),
        ] {
            assert_eq!(
                floored_take(rem, bud, floor),
                want,
                "floored_take({rem}, {bud}, {floor}): a FINAL chunk keeps its exact length \
                 whatever it is; a non-final one is handed out as-is unless it would strand a \
                 remainder under the floor, in which case it pulls back to leave exactly the \
                 floor behind; and a budget that cannot reach the floor at all is left alone so \
                 a small-budget scheduler still makes progress instead of deadlocking. Note \
                 724 is NOT rounded to 512: the rule is a floor on chunk size, not a divisor"
            );
        }
        assert!(
            under.is_empty() || prefill_chunk_min() == 0,
            "these non-final prefill chunks are under {floor} rows: {under:?}. `take = \
             remaining.min(budget)` hands the second sequence whatever the first left over, so \
             a prompt's chunk sizes depend on what else was in flight. A prefill chunk of fewer \
             than {floor} rows takes a different path through the projection GEMM and computes \
             different K (nv-models paged_prefill_chunk_divergence_by_layer_depth, 2048 tokens: \
             chunks 1792,256 agree to 0e0 while 1793,255 differ by 1.19e-1 on layer 0 K -- one \
             token of chunk size flips it). Chunks need NOT be multiples of {floor}: 448 rows \
             five times over is exact. So the fix is a floor, not an alignment"
        );
    }

    #[test]
    fn long_prompt_prefills_in_chunks_across_steps() {
        let mut sch = Scheduler::new(chunk_cfg(4));
        sch.enqueue(Sequence::new(1, (1..=10).collect(), 4));

        let batch = sch.step().unwrap();
        assert_eq!(batch.kind, BatchKind::Prefill);
        assert_eq!(
            batch.items,
            vec![ScheduledSeqItem {
                seq_id: 1,
                num_scheduled_tokens: 4,
                is_prefill: true,
                is_final_prefill_chunk: false,
            }]
        );
        assert!(sch.complete_step(&[]).unwrap().is_empty());
        assert_eq!(sch.running_seq(1).unwrap().num_computed_tokens, 4);
        assert_eq!(sch.running_seq(1).unwrap().state, SequenceState::Prefill);

        let batch = sch.step().unwrap();
        assert_eq!(batch.items[0].num_scheduled_tokens, 4);
        assert!(!batch.items[0].is_final_prefill_chunk);
        sch.complete_step(&[]).unwrap();
        assert_eq!(sch.running_seq(1).unwrap().num_computed_tokens, 8);

        let batch = sch.step().unwrap();
        assert_eq!(batch.items[0].num_scheduled_tokens, 2);
        assert!(batch.items[0].is_final_prefill_chunk);
        sch.complete_step(&[500]).unwrap();
        let seq = sch.running_seq(1).unwrap();
        assert_eq!(seq.state, SequenceState::Decode);
        assert_eq!(seq.output, vec![500]);
        assert_eq!(seq.num_computed_tokens, seq.total_len());
    }

    #[test]
    fn final_chunk_transitions_to_decode() {
        let mut sch = Scheduler::new(chunk_cfg(4));
        sch.enqueue(Sequence::new(1, (1..=6).collect(), 4));
        let _ = sch.step().unwrap();
        sch.complete_step(&[]).unwrap();

        let batch = sch.step().unwrap();
        assert_eq!(batch.kind, BatchKind::Prefill);
        assert!(batch.items[0].is_prefill);
        assert!(batch.items[0].is_final_prefill_chunk);
        assert_eq!(batch.items[0].num_scheduled_tokens, 2);
        sch.complete_step(&[42]).unwrap();
        let seq = sch.running_seq(1).unwrap();
        assert_eq!(seq.state, SequenceState::Decode);
        assert_eq!(seq.output, vec![42]);
        assert_eq!(seq.num_computed_tokens, seq.total_len());
    }

    #[test]
    fn chunk_and_decode_mix_in_one_batch() {
        let mut sch = Scheduler::new(chunk_cfg(4));
        decoding_seq(&mut sch, 1, vec![1, 2, 3], 16);
        sch.enqueue(Sequence::new(2, (10..20).collect(), 4));

        let batch = sch.step().unwrap();
        assert_eq!(batch.kind, BatchKind::Mixed);
        assert_eq!(
            batch.items,
            vec![
                ScheduledSeqItem {
                    seq_id: 1,
                    num_scheduled_tokens: 1,
                    is_prefill: false,
                    is_final_prefill_chunk: false,
                },
                ScheduledSeqItem {
                    seq_id: 2,
                    num_scheduled_tokens: 3,
                    is_prefill: true,
                    is_final_prefill_chunk: false,
                },
            ]
        );
        sch.complete_step(&[7]).unwrap();
        assert_eq!(sch.running_seq(1).unwrap().output, vec![100, 7]);
        assert_eq!(sch.running_seq(2).unwrap().num_computed_tokens, 3);
    }

    #[test]
    fn decode_not_starved_by_long_prefill() {
        let mut sch = Scheduler::new(chunk_cfg(5));
        decoding_seq(&mut sch, 1, vec![1, 2, 3], 16);
        sch.enqueue(Sequence::new(2, (100..120).collect(), 4));

        for i in 0..4u32 {
            let batch = sch.step().unwrap();
            assert_eq!(batch.kind, BatchKind::Mixed);
            assert_eq!(batch.items[0].seq_id, 1);
            assert!(!batch.items[0].is_prefill);
            assert_eq!(batch.items[1].num_scheduled_tokens, 4);
            sch.complete_step(&[200 + i]).unwrap();
            assert_eq!(sch.running_seq(1).unwrap().output.len(), 2 + i as usize);
        }

        let batch = sch.step().unwrap();
        assert_eq!(batch.kind, BatchKind::Mixed);
        assert!(batch.items[1].is_final_prefill_chunk);
        sch.complete_step(&[204, 300]).unwrap();
        assert_eq!(
            sch.running_seq(1).unwrap().output,
            vec![100, 200, 201, 202, 203, 204]
        );
        let seq2 = sch.running_seq(2).unwrap();
        assert_eq!(seq2.state, SequenceState::Decode);
        assert_eq!(seq2.output, vec![300]);
    }

    #[test]
    fn verify_takes_priority_over_pending_chunks() {
        let mut sch = Scheduler::new(chunk_cfg(8));
        decoding_seq(&mut sch, 1, vec![1, 2, 3], 16);
        sch.set_drafts(1, vec![201, 202]).unwrap();
        sch.enqueue(Sequence::new(2, (10..30).collect(), 4));

        let batch = sch.step().unwrap();
        assert_eq!(batch.kind, BatchKind::Verify);
        assert_eq!(batch.seq_ids, vec![1]);
        assert!(batch.items.is_empty());

        sch.complete_verify_step(&[VerifyOutcome {
            accepted_count: 1,
            bonus_token: 900,
            accepted_drafts: vec![201],
        }])
        .unwrap();

        let batch = sch.step().unwrap();
        assert_eq!(batch.kind, BatchKind::Mixed);
        assert_eq!(batch.items[0].seq_id, 1);
        assert!(!batch.items[0].is_prefill);
        assert_eq!(batch.items[1].seq_id, 2);
        assert!(batch.items[1].is_prefill);
    }

    #[test]
    fn drafts_on_prefill_seq_do_not_wedge_step() {
        let mut sch = Scheduler::new(chunk_cfg(4));
        sch.enqueue(Sequence::new(1, (1..=10).collect(), 4));
        let _ = sch.step().unwrap();
        sch.complete_step(&[]).unwrap();
        sch.set_drafts(1, vec![201]).unwrap();
        assert_eq!(sch.num_running_with_drafts, 1);

        let batch = sch.step().unwrap();
        assert_eq!(batch.kind, BatchKind::Prefill);
        assert_eq!(batch.seq_ids, vec![1]);
        assert_eq!(batch.items[0].num_scheduled_tokens, 4);
    }

    #[test]
    fn budget_forces_second_seq_to_wait() {
        let mut sch = Scheduler::new(SchedulerConfig {
            max_batch_size: 4,
            max_batched_tokens: 5,
            block_size: 4,
            num_blocks: 16,
        });
        sch.enqueue(Sequence::new(1, vec![1, 2, 3, 4, 5], 4));
        sch.enqueue(Sequence::new(2, vec![10, 11, 12, 13, 14], 4));
        let batch = sch.step().unwrap();
        assert_eq!(batch.kind, BatchKind::Prefill);
        assert_eq!(batch.seq_ids, vec![1]);
        assert_eq!(sch.waiting.len(), 1);
        assert_eq!(sch.waiting.front().unwrap().id, 2);
    }
}
