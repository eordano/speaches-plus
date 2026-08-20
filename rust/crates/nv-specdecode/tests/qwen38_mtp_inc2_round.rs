use anyhow::Result;
use nv_specdecode::qwen38_mtp::{
    mtp_verify_replay_selected, run_mtp_verify_round, MtpBatchedVerifyTarget,
    MTP_VERIFY_INC2_COMMITS_BATCHED_ROWS_AND_TOLERATES_MROW_VS_M1_DRIFT,
};

#[derive(Debug, PartialEq, Clone)]
enum Call {
    Verify(Vec<u32>),
    VerifyCommitOnly(Vec<u32>),
    Advance(usize),
}

#[derive(Default)]
struct ScriptedTarget {
    amax_per_verify: Vec<Vec<u32>>,
    calls: Vec<Call>,
    verifies: usize,
}

impl ScriptedTarget {
    fn new(amax_per_verify: Vec<Vec<u32>>) -> Self {
        Self {
            amax_per_verify,
            calls: Vec::new(),
            verifies: 0,
        }
    }
}

impl MtpBatchedVerifyTarget for ScriptedTarget {
    fn verify_chain(&mut self, batch: &[u32]) -> Result<Vec<u32>> {
        self.calls.push(Call::Verify(batch.to_vec()));
        let i = self.verifies;
        self.verifies += 1;
        Ok(self
            .amax_per_verify
            .get(i)
            .cloned()
            .unwrap_or_else(|| batch.to_vec()))
    }

    fn verify_chain_commit_only(&mut self, batch: &[u32]) -> Result<()> {
        self.calls.push(Call::VerifyCommitOnly(batch.to_vec()));
        Ok(())
    }

    fn advance(&mut self, n: usize) -> Result<()> {
        self.calls.push(Call::Advance(n));
        Ok(())
    }
}

const ANCHOR: u32 = 11;

#[test]
fn full_accept_commits_the_batched_forward_in_place_with_no_second_forward() {
    let mut t = ScriptedTarget::new(vec![vec![5, 6, 7, 42]]);
    let r = run_mtp_verify_round(&mut t, ANCHOR, &[5, 6, 7], false).expect("round");
    assert_eq!(r.batch, vec![ANCHOR, 5, 6, 7]);
    assert_eq!(r.accept.commit_len, 4);
    assert_eq!(r.accept.draft_accepted, 3);
    assert_eq!(r.emitted, vec![5, 6, 7, 42]);
    assert!(!r.prefix_reforwarded_batched);
    assert_eq!(
        t.calls,
        vec![Call::Verify(vec![ANCHOR, 5, 6, 7]), Call::Advance(4)],
        "a full accept must commit the single batched forward in place; \
         {MTP_VERIFY_INC2_COMMITS_BATCHED_ROWS_AND_TOLERATES_MROW_VS_M1_DRIFT}"
    );
}

#[test]
fn partial_accept_rolls_back_then_reforwards_only_the_accepted_prefix_batched() {
    let mut t = ScriptedTarget::new(vec![vec![5, 99, 0, 0]]);
    let r = run_mtp_verify_round(&mut t, ANCHOR, &[5, 6, 7], false).expect("round");
    assert_eq!(r.accept.commit_len, 2);
    assert_eq!(r.accept.draft_accepted, 1);
    assert_eq!(r.accept.next_bonus, 99);
    assert_eq!(
        r.emitted,
        vec![5, 99],
        "the bonus must come from the FIRST verify forward; the prefix re-forward only rebuilds \
         committed state"
    );
    assert!(r.prefix_reforwarded_batched);
    assert_eq!(
        t.calls,
        vec![
            Call::Verify(vec![ANCHOR, 5, 6, 7]),
            Call::Advance(0),
            Call::VerifyCommitOnly(vec![ANCHOR, 5]),
            Call::Advance(2),
        ],
        "a partial accept must advance(0) to roll back, re-forward batch[..commit_len] batched \
         commit-only (no lm_head tail: the prefix tokens are already known), then \
         full-accept-advance that prefix; \
         {MTP_VERIFY_INC2_COMMITS_BATCHED_ROWS_AND_TOLERATES_MROW_VS_M1_DRIFT}"
    );
}

#[test]
fn replay_commit_keeps_the_increment1_single_verify_call_shape() {
    let mut t = ScriptedTarget::new(vec![vec![5, 99, 0, 0]]);
    let r = run_mtp_verify_round(&mut t, ANCHOR, &[5, 6, 7], true).expect("round");
    assert_eq!(r.emitted, vec![5, 99]);
    assert!(!r.prefix_reforwarded_batched);
    assert_eq!(
        t.calls,
        vec![Call::Verify(vec![ANCHOR, 5, 6, 7]), Call::Advance(2)],
        "replay_commit=true must delegate the partial-accept commit to the target's advance() \
         (the increment-1 rollback + M=1 replay escape)"
    );
}

#[test]
fn zero_draft_round_is_a_full_accept_of_the_anchor_alone() {
    let mut t = ScriptedTarget::new(vec![vec![77]]);
    let r = run_mtp_verify_round(&mut t, ANCHOR, &[], false).expect("round");
    assert_eq!(r.batch, vec![ANCHOR]);
    assert_eq!(r.accept.commit_len, 1);
    assert_eq!(r.emitted, vec![77]);
    assert!(!r.prefix_reforwarded_batched);
    assert_eq!(t.calls, vec![Call::Verify(vec![ANCHOR]), Call::Advance(1)]);
}

#[derive(Default)]
struct FullVerifyOnlyTarget {
    calls: Vec<Call>,
}

impl MtpBatchedVerifyTarget for FullVerifyOnlyTarget {
    fn verify_chain(&mut self, batch: &[u32]) -> Result<Vec<u32>> {
        self.calls.push(Call::Verify(batch.to_vec()));
        Ok(if self.calls.len() == 1 {
            vec![5, 99, 0, 0]
        } else {
            batch.to_vec()
        })
    }

    fn advance(&mut self, n: usize) -> Result<()> {
        self.calls.push(Call::Advance(n));
        Ok(())
    }
}

#[test]
fn a_target_without_a_commit_only_path_reforwards_through_full_verify_chain_by_default() {
    let mut t = FullVerifyOnlyTarget::default();
    let r = run_mtp_verify_round(&mut t, ANCHOR, &[5, 6, 7], false).expect("round");
    assert_eq!(r.emitted, vec![5, 99]);
    assert!(r.prefix_reforwarded_batched);
    assert_eq!(
        t.calls,
        vec![
            Call::Verify(vec![ANCHOR, 5, 6, 7]),
            Call::Advance(0),
            Call::Verify(vec![ANCHOR, 5]),
            Call::Advance(2),
        ],
        "the trait's default verify_chain_commit_only must fall back to the full verify_chain so \
         targets without a commit-only path keep the inc2 round shape"
    );
}

#[test]
fn replay_escape_env_parses_only_a_literal_1() {
    assert!(!mtp_verify_replay_selected(None));
    assert!(!mtp_verify_replay_selected(Some("0")));
    assert!(!mtp_verify_replay_selected(Some("true")));
    assert!(mtp_verify_replay_selected(Some("1")));
    assert!(mtp_verify_replay_selected(Some(" 1 ")));
}
