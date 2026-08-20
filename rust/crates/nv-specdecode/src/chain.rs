use anyhow::{ensure, Result};

pub fn lower_tri_mask(n: usize) -> Vec<u8> {
    let mut m = vec![0u8; n * n];
    for i in 0..n {
        for j in 0..=i {
            m[i * n + j] = 1;
        }
    }
    m
}

pub fn chain_positions(committed: usize, k: usize) -> Vec<i32> {
    (committed as i32..(committed + k) as i32).collect()
}

pub fn build_chain_batch(bonus: u32, draft: &[u32], k: usize, shift: bool) -> Result<Vec<u32>> {
    ensure!(k >= 1, "chain batch needs k >= 1, got {k}");
    let mut batch = Vec::with_capacity(k);
    batch.push(bonus);
    if shift {
        ensure!(
            draft.len() >= k - 1,
            "shift chain batch needs {} draft tokens, got {}",
            k - 1,
            draft.len()
        );
        batch.extend_from_slice(&draft[..k - 1]);
    } else {
        ensure!(
            draft.len() >= k,
            "chain batch needs {k} draft tokens, got {}",
            draft.len()
        );
        batch.extend_from_slice(&draft[1..k]);
    }
    Ok(batch)
}

pub fn aux_row_extract(
    gaux: &[f32],
    n_layers: usize,
    k: usize,
    hidden: usize,
    j: usize,
) -> Result<Vec<f32>> {
    ensure!(
        gaux.len() == n_layers * k * hidden,
        "aux buffer has {} floats, expected n_layers={n_layers} * k={k} * hidden={hidden} = {}",
        gaux.len(),
        n_layers * k * hidden
    );
    ensure!(j < k, "aux row index {j} out of range for k={k}");
    let mut r = Vec::with_capacity(n_layers * hidden);
    for l in 0..n_layers {
        let base = l * k * hidden + j * hidden;
        r.extend_from_slice(&gaux[base..base + hidden]);
    }
    Ok(r)
}

pub fn drafter_kv_cap_from_env(
    window: Option<usize>,
    sink: Option<usize>,
    default_sink: usize,
) -> Option<(usize, usize)> {
    match window {
        Some(w) if w > 0 => Some((sink.unwrap_or(default_sink), w)),
        _ => None,
    }
}

pub fn effective_drafter_kv_rows(
    kv_max_seq_len: usize,
    kv_cap: Option<(usize, usize)>,
    slack: usize,
) -> usize {
    match kv_cap {
        Some((sink, window)) if window > 0 => {
            kv_max_seq_len.min(sink.saturating_add(window).saturating_add(slack))
        }
        _ => kv_max_seq_len,
    }
}

pub fn chain_graph_cap(
    kv_max_seq_len: usize,
    k: usize,
    kv_cap: Option<(usize, usize)>,
    slack: usize,
) -> usize {
    effective_drafter_kv_rows(kv_max_seq_len, kv_cap, slack).saturating_add(k)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ChainAccept {
    pub commit_len: usize,
    pub draft_accepted: usize,
    pub next_bonus: u32,
}

pub fn judge_slot_argmax(batch: &[u32], amax: &[u32], i: usize) -> Option<u32> {
    if i == 0 {
        return None;
    }
    let proposed = *batch.get(i)?;
    let judged = *amax.get(i - 1)?;
    if proposed == judged {
        None
    } else {
        Some(judged)
    }
}

pub fn accept_prefix_argmax(batch: &[u32], amax: &[u32]) -> Result<ChainAccept> {
    let k = batch.len();
    ensure!(k >= 1, "chain accept needs a non-empty batch");
    ensure!(
        amax.len() == k,
        "chain accept: {} argmax rows for a k={k} batch",
        amax.len()
    );
    let mut commit_len = k;
    let mut next_bonus = amax[k - 1];
    for i in 1..k {
        if let Some(repl) = judge_slot_argmax(batch, amax, i) {
            commit_len = i;
            next_bonus = repl;
            break;
        }
    }
    Ok(ChainAccept {
        commit_len,
        draft_accepted: commit_len - 1,
        next_bonus,
    })
}

#[derive(Clone, Debug)]
pub struct ChainState {
    context: Vec<u32>,
    committed: usize,
    aux_rows: usize,
    fc_in: usize,
}

impl ChainState {
    pub fn new(prompt_ids: &[u32], fc_in: usize) -> Result<Self> {
        ensure!(
            !prompt_ids.is_empty(),
            "chain state needs a non-empty prompt"
        );
        ensure!(fc_in > 0, "chain state needs fc_in > 0");
        Ok(Self {
            context: prompt_ids.to_vec(),
            committed: prompt_ids.len(),
            aux_rows: prompt_ids.len(),
            fc_in,
        })
    }

    pub fn commit_token(&mut self, tok: u32, aux_row: &[f32]) -> Result<()> {
        ensure!(
            aux_row.len() == self.fc_in,
            "aux row has {} floats, expected fc_in={}",
            aux_row.len(),
            self.fc_in
        );
        self.context.push(tok);
        self.committed += 1;
        self.aux_rows += 1;
        debug_assert_eq!(self.committed, self.context.len());
        Ok(())
    }

    pub fn assert_round_start(&self, k: usize, cache_capacity: usize) -> Result<()> {
        ensure!(
            self.committed == self.context.len(),
            "chain state desync: committed={} but context has {} tokens",
            self.committed,
            self.context.len()
        );
        ensure!(
            self.aux_rows == self.committed,
            "aux lockstep broken: {} aux rows for {} committed tokens",
            self.aux_rows,
            self.committed
        );
        ensure!(
            self.committed + k <= cache_capacity,
            "verify cache overflow: committed={} + k={k} > capacity={cache_capacity}",
            self.committed
        );
        Ok(())
    }

    pub fn context(&self) -> &[u32] {
        &self.context
    }

    pub fn committed(&self) -> usize {
        self.committed
    }

    pub fn aux_rows(&self) -> usize {
        self.aux_rows
    }
}

#[derive(Clone, Debug)]
pub enum ChainJudgment {
    Argmax(Vec<u32>),
    Logits { vocab: usize, data: Vec<f32> },
}

#[derive(Clone, Debug)]
pub struct ChainVerifyOut {
    pub judgment: ChainJudgment,
    pub aux: Vec<f32>,
}

pub trait ChainVerifier {
    fn verify_chain(
        &mut self,
        batch: &[u32],
        positions: &[i32],
        mask: &[u8],
        committed: usize,
        want_logits: bool,
    ) -> Result<ChainVerifyOut>;
}

#[cfg(feature = "cuda")]
impl<M: std::ops::Deref<Target = nv_models::gemma4::Gemma4>> ChainVerifier
    for nv_models::gemma4_graph::GraphedGemma4Verify<M>
{
    fn verify_chain(
        &mut self,
        batch: &[u32],
        positions: &[i32],
        mask: &[u8],
        committed: usize,
        want_logits: bool,
    ) -> Result<ChainVerifyOut> {
        if want_logits {
            let vocab = self.model().config().vocab_size;
            let (data, aux) = self.run(batch, positions, mask, committed)?;
            Ok(ChainVerifyOut {
                judgment: ChainJudgment::Logits { vocab, data },
                aux,
            })
        } else {
            let (amax, aux) = self.run_argmax(batch, positions, mask, committed)?;
            Ok(ChainVerifyOut {
                judgment: ChainJudgment::Argmax(amax),
                aux,
            })
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::eagle3::{flatten_with_mask, DraftTree};

    const K: usize = 8;

    fn tok(pos: usize) -> u32 {
        ((pos * 7 + 3) % 50) as u32
    }

    #[test]
    fn kv_cap_from_env_matches_drafter_cache_semantics() {
        use crate::eagle3_loader::{DrafterKvCache, DRAFTER_KV_CAP_DEFAULT_SINK};
        assert_eq!(
            drafter_kv_cap_from_env(None, None, DRAFTER_KV_CAP_DEFAULT_SINK),
            None
        );
        assert_eq!(
            drafter_kv_cap_from_env(Some(0), Some(4), DRAFTER_KV_CAP_DEFAULT_SINK),
            None
        );
        assert_eq!(
            drafter_kv_cap_from_env(Some(2048), None, DRAFTER_KV_CAP_DEFAULT_SINK),
            Some((DRAFTER_KV_CAP_DEFAULT_SINK, 2048))
        );
        assert_eq!(
            drafter_kv_cap_from_env(Some(2048), Some(4), DRAFTER_KV_CAP_DEFAULT_SINK),
            Some((4, 2048))
        );
        for (w, s) in [
            (None, None),
            (Some(0), Some(4)),
            (Some(2048), None),
            (Some(1024), Some(8)),
        ] {
            let mut c = DrafterKvCache::new();
            c.set_kv_cap(s.unwrap_or(DRAFTER_KV_CAP_DEFAULT_SINK), w.unwrap_or(0));
            assert_eq!(
                drafter_kv_cap_from_env(w, s, DRAFTER_KV_CAP_DEFAULT_SINK),
                c.kv_cap(),
                "w={w:?} s={s:?}"
            );
        }
    }

    #[test]
    fn effective_rows_uncapped_is_max_seq() {
        assert_eq!(effective_drafter_kv_rows(32768, None, 256), 32768);
        assert_eq!(effective_drafter_kv_rows(32768, Some((16, 0)), 256), 32768);
        assert_eq!(effective_drafter_kv_rows(0, None, 256), 0);
    }

    #[test]
    fn effective_rows_capped_is_sink_window_slack() {
        assert_eq!(
            effective_drafter_kv_rows(32768, Some((16, 2048)), 256),
            16 + 2048 + 256
        );
        assert_eq!(
            effective_drafter_kv_rows(32768, Some((0, 1024)), 256),
            1024 + 256
        );
    }

    #[test]
    fn effective_rows_never_exceeds_max_seq() {
        assert_eq!(
            effective_drafter_kv_rows(4096, Some((16, 32768)), 256),
            4096
        );
        assert_eq!(
            effective_drafter_kv_rows(4096, Some((16, usize::MAX)), 256),
            4096
        );
        assert_eq!(
            effective_drafter_kv_rows(4096, Some((usize::MAX, usize::MAX)), usize::MAX),
            4096
        );
    }

    #[test]
    fn chain_graph_cap_matches_legacy_when_uncapped() {
        for &(m, k) in &[(32768usize, 4usize), (8192, 3), (1, 1)] {
            assert_eq!(chain_graph_cap(m, k, None, 256), m + k);
        }
    }

    #[test]
    fn chain_graph_cap_covers_projected_phys_guard() {
        let slack = crate::eagle3_loader::DRAFTER_KV_CAP_SLACK;
        for &(sink, window) in &[(16usize, 1024usize), (0, 4096), (16, 2048)] {
            for k in 1..=8usize {
                let kd = k - 1;
                let cap = chain_graph_cap(32768, k, Some((sink, window)), slack);
                let worst_projected_phys = sink + window + slack;
                assert!(
                    worst_projected_phys + kd <= cap,
                    "sink={sink} window={window} k={k}: {worst_projected_phys}+{kd} > {cap}"
                );
            }
        }
    }

    #[test]
    fn chain_graph_cap_capped_is_smaller_than_max_seq_sizing() {
        let cap = chain_graph_cap(32768, 4, Some((16, 2048)), 256);
        assert_eq!(cap, 16 + 2048 + 256 + 4);
        assert!(cap < 32768 + 4);
    }

    #[test]
    fn accept_all_drafts_commits_k_and_carries_last_amax() {
        let batch: Vec<u32> = vec![10, 11, 12, 13];
        let amax: Vec<u32> = vec![11, 12, 13, 99];
        let acc = accept_prefix_argmax(&batch, &amax).unwrap();
        assert_eq!(acc.commit_len, 4);
        assert_eq!(acc.draft_accepted, 3);
        assert_eq!(acc.next_bonus, 99);
    }

    #[test]
    fn first_draft_reject_commits_bonus_only() {
        let batch: Vec<u32> = vec![10, 11, 12, 13];
        let amax: Vec<u32> = vec![77, 12, 13, 99];
        let acc = accept_prefix_argmax(&batch, &amax).unwrap();
        assert_eq!(acc.commit_len, 1);
        assert_eq!(acc.draft_accepted, 0);
        assert_eq!(acc.next_bonus, 77);
    }

    #[test]
    fn mid_reject_commits_prefix_and_replacement_bonus() {
        for reject_at in 2..K {
            let batch: Vec<u32> = (0..K).map(|i| 100 + i as u32).collect();
            let mut amax: Vec<u32> = (0..K).map(|i| 101 + i as u32).collect();
            amax[reject_at - 1] = 999;
            let acc = accept_prefix_argmax(&batch, &amax).unwrap();
            assert_eq!(acc.commit_len, reject_at, "reject_at={reject_at}");
            assert_eq!(acc.draft_accepted, reject_at - 1);
            assert_eq!(acc.next_bonus, 999);
        }
    }

    #[test]
    fn bonus_slot_never_judged() {
        let batch: Vec<u32> = vec![5, 6, 7];
        let amax: Vec<u32> = vec![0, 0, 0];
        let acc = accept_prefix_argmax(&batch, &amax).unwrap();
        assert!(acc.commit_len >= 1);
        assert_eq!(acc.commit_len, 1);
        assert_eq!(judge_slot_argmax(&batch, &amax, 0), None);
    }

    #[test]
    fn judge_slot_argmax_out_of_range_is_none_not_panic() {
        let batch: Vec<u32> = vec![5, 6, 7];
        assert_eq!(judge_slot_argmax(&batch, &[9, 9], 3), None);
        assert_eq!(judge_slot_argmax(&batch, &[9], 2), None);
        assert_eq!(judge_slot_argmax(&batch, &[], 1), None);
        assert_eq!(judge_slot_argmax(&[], &[], 1), None);
        assert_eq!(judge_slot_argmax(&batch, &[9, 9, 9], 2), Some(9));
    }

    #[test]
    fn judge_slot_argmax_matches_whole_round_walk() {
        let mut rng: u64 = 0x9e3779b97f4a7c15;
        let mut next = move || {
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            rng
        };
        for _ in 0..200 {
            let k = 2 + (next() % 7) as usize;
            let batch: Vec<u32> = (0..k).map(|_| (next() % 4) as u32).collect();
            let amax: Vec<u32> = (0..k).map(|_| (next() % 4) as u32).collect();
            let acc = accept_prefix_argmax(&batch, &amax).unwrap();
            let mut walk_commit = k;
            let mut walk_bonus = amax[k - 1];
            for i in 1..k {
                if let Some(repl) = judge_slot_argmax(&batch, &amax, i) {
                    walk_commit = i;
                    walk_bonus = repl;
                    break;
                }
            }
            assert_eq!(acc.commit_len, walk_commit);
            assert_eq!(acc.next_bonus, walk_bonus);
            assert_eq!(acc.draft_accepted, walk_commit - 1);
        }
    }

    #[test]
    fn build_chain_batch_shift_and_bonus_modes() {
        let draft: Vec<u32> = vec![20, 21, 22, 23];
        let shifted = build_chain_batch(9, &draft, 4, true).unwrap();
        assert_eq!(shifted, vec![9, 20, 21, 22]);
        let unshifted = build_chain_batch(9, &draft, 4, false).unwrap();
        assert_eq!(unshifted, vec![9, 21, 22, 23]);
        assert!(build_chain_batch(9, &draft[..2], 4, true).is_err());
        assert!(build_chain_batch(9, &draft[..3], 4, false).is_err());
        assert_eq!(
            build_chain_batch(9, &draft[..3], 4, true).unwrap(),
            vec![9, 20, 21, 22]
        );
    }

    #[test]
    fn lower_tri_mask_matches_chain_tree_flatten() {
        for n in 1..=6 {
            let tree = DraftTree {
                tokens: (0..n as u32).collect(),
                parents: (0..n)
                    .map(|i| if i == 0 { None } else { Some(i - 1) })
                    .collect(),
                depths: (1..=n).collect(),
            };
            let (_, tree_mask) = flatten_with_mask(&tree);
            assert_eq!(lower_tri_mask(n), tree_mask, "n={n}");
        }
    }

    #[test]
    fn chain_positions_are_contiguous_from_committed() {
        for &(c, k) in &[(0usize, 1usize), (5, 4), (117, 8), (4096, 16)] {
            let pos = chain_positions(c, k);
            assert_eq!(pos.len(), k);
            for (i, p) in pos.iter().enumerate() {
                assert_eq!(*p, (c + i) as i32);
            }
        }
    }

    #[test]
    fn aux_row_extract_concatenates_layer_blocks_in_order() {
        let n_layers = 3;
        let k = 4;
        let hidden = 5;
        let mut gaux = vec![0.0f32; n_layers * k * hidden];
        for l in 0..n_layers {
            for j in 0..k {
                for h in 0..hidden {
                    gaux[l * k * hidden + j * hidden + h] = (l * 1000 + j * 10 + h) as f32;
                }
            }
        }
        for j in 0..k {
            let row = aux_row_extract(&gaux, n_layers, k, hidden, j).unwrap();
            assert_eq!(row.len(), n_layers * hidden);
            for l in 0..n_layers {
                for h in 0..hidden {
                    assert_eq!(row[l * hidden + h], (l * 1000 + j * 10 + h) as f32);
                }
            }
        }
        assert!(aux_row_extract(&gaux, n_layers, k, hidden, k).is_err());
        assert!(aux_row_extract(&gaux[..gaux.len() - 1], n_layers, k, hidden, 0).is_err());
    }

    #[test]
    fn chain_state_lockstep_commit_and_reject_bookkeeping() {
        let fc_in = 6;
        let prompt: Vec<u32> = vec![1, 2, 3];
        let mut st = ChainState::new(&prompt, fc_in).unwrap();
        assert_eq!(st.committed(), 3);
        assert_eq!(st.aux_rows(), 3);
        st.assert_round_start(4, 64).unwrap();

        let batch: Vec<u32> = vec![40, 41, 42, 43];
        let amax: Vec<u32> = vec![41, 999, 0, 0];
        let acc = accept_prefix_argmax(&batch, &amax).unwrap();
        assert_eq!(acc.commit_len, 2);
        let aux_row = vec![0.5f32; fc_in];
        for &t in &batch[..acc.commit_len] {
            st.commit_token(t, &aux_row).unwrap();
        }
        assert_eq!(st.committed(), 5);
        assert_eq!(st.context(), &[1, 2, 3, 40, 41]);
        assert_eq!(st.aux_rows(), st.committed());
        assert!(!st.context().contains(&42));
        assert!(!st.context().contains(&43));

        st.assert_round_start(4, 64).unwrap();
        assert!(st.commit_token(50, &aux_row[..fc_in - 1]).is_err());
    }

    #[test]
    fn chain_state_round_start_guard_rejects_overflow_and_desync() {
        let fc_in = 4;
        let st = ChainState::new(&[1, 2, 3, 4, 5], fc_in).unwrap();
        st.assert_round_start(3, 8).unwrap();
        assert!(st.assert_round_start(4, 8).is_err());
        assert!(st.assert_round_start(3, 7).is_err());

        let mut desync = st.clone();
        desync.aux_rows += 1;
        assert!(desync.assert_round_start(1, 100).is_err());
    }

    struct MockChainVerifier {
        n_layers: usize,
        k: usize,
        hidden: usize,
        initial: usize,
        calls: Vec<(usize, Vec<i32>, Vec<u8>)>,
    }

    impl ChainVerifier for MockChainVerifier {
        fn verify_chain(
            &mut self,
            batch: &[u32],
            positions: &[i32],
            mask: &[u8],
            committed: usize,
            want_logits: bool,
        ) -> Result<ChainVerifyOut> {
            assert_eq!(batch.len(), self.k);
            assert!(!want_logits);
            self.calls
                .push((committed, positions.to_vec(), mask.to_vec()));
            let amax: Vec<u32> = (0..self.k).map(|i| tok(committed + i + 1)).collect();
            let aux = vec![1.0f32; self.n_layers * self.k * self.hidden];
            Ok(ChainVerifyOut {
                judgment: ChainJudgment::Argmax(amax),
                aux,
            })
        }
    }

    #[test]
    fn mock_chain_verifier_drives_multi_round_loop() {
        let k = 4;
        let n_layers = 2;
        let hidden = 3;
        let fc_in = n_layers * hidden;
        let prompt: Vec<u32> = (0..5).map(tok).collect();
        let initial = prompt.len();
        let mut st = ChainState::new(&prompt, fc_in).unwrap();
        let mut mock = MockChainVerifier {
            n_layers,
            k,
            hidden,
            initial,
            calls: Vec::new(),
        };

        let correct_per_round = [k - 1, 0, 1, k - 1, 2];
        let mut bonus = tok(initial);
        let mut emitted: Vec<u32> = Vec::new();
        for &n_correct in &correct_per_round {
            st.assert_round_start(k, 4096).unwrap();
            let c = st.committed();
            let draft: Vec<u32> = (0..k - 1)
                .map(|j| {
                    if j < n_correct {
                        tok(c + 1 + j)
                    } else {
                        tok(c) ^ 1
                    }
                })
                .collect();
            let batch = build_chain_batch(bonus, &draft, k, true).unwrap();
            let positions = chain_positions(c, k);
            let mask = lower_tri_mask(k);
            let out = mock
                .verify_chain(&batch, &positions, &mask, c, false)
                .unwrap();
            let amax = match &out.judgment {
                ChainJudgment::Argmax(a) => a.clone(),
                _ => unreachable!(),
            };
            let acc = accept_prefix_argmax(&batch, &amax).unwrap();
            assert_eq!(acc.commit_len, (n_correct + 1).min(k));
            let aux_before = st.aux_rows();
            for (i, &t) in batch[..acc.commit_len].iter().enumerate() {
                let row = aux_row_extract(&out.aux, n_layers, k, hidden, i).unwrap();
                st.commit_token(t, &row).unwrap();
                emitted.push(t);
            }
            assert_eq!(st.aux_rows() - aux_before, acc.commit_len);
            bonus = acc.next_bonus;
        }

        let expected: Vec<u32> = (initial..st.committed()).map(tok).collect();
        assert_eq!(
            emitted, expected,
            "emitted stream must equal the greedy argmax stream"
        );
        assert_eq!(st.context()[initial..], emitted[..]);

        let mut expect_committed = initial;
        for (call_idx, (committed, positions, mask)) in mock.calls.iter().enumerate() {
            assert_eq!(*committed, expect_committed, "round {call_idx}");
            assert_eq!(positions, &chain_positions(*committed, k));
            assert_eq!(mask, &lower_tri_mask(k));
            expect_committed += (correct_per_round[call_idx] + 1).min(k);
        }
        assert_eq!(expect_committed, st.committed());
        assert_eq!(mock.initial, initial);
    }
}
