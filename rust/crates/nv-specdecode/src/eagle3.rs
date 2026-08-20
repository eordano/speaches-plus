use anyhow::Result;

pub trait DraftScorer {
    fn score(&mut self, context: &[u32]) -> Result<Vec<f32>>;
}

impl<F> DraftScorer for F
where
    F: FnMut(&[u32]) -> Result<Vec<f32>>,
{
    fn score(&mut self, context: &[u32]) -> Result<Vec<f32>> {
        (self)(context)
    }
}

#[derive(Clone, Debug)]
pub struct Eagle3Config {
    pub max_depth: usize,

    pub branch_factor: usize,

    pub total_budget: usize,

    pub vocab_size: usize,
}

impl Default for Eagle3Config {
    fn default() -> Self {
        Self {
            max_depth: 4,
            branch_factor: 4,
            total_budget: 16,
            vocab_size: 0,
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct DraftTree {
    pub tokens: Vec<u32>,
    pub parents: Vec<Option<usize>>,
    pub depths: Vec<usize>,
}

impl DraftTree {
    pub fn len(&self) -> usize {
        self.tokens.len()
    }

    pub fn is_empty(&self) -> bool {
        self.tokens.is_empty()
    }
}

pub struct Eagle3Proposer<S: DraftScorer> {
    scorer: S,
    cfg: Eagle3Config,

    last_tree: Option<DraftTree>,
}

impl<S: DraftScorer> Eagle3Proposer<S> {
    pub fn new(scorer: S, cfg: Eagle3Config) -> Self {
        Self {
            scorer,
            cfg,
            last_tree: None,
        }
    }

    pub fn config(&self) -> &Eagle3Config {
        &self.cfg
    }

    pub fn last_tree(&self) -> Option<&DraftTree> {
        self.last_tree.as_ref()
    }

    pub fn scorer_mut(&mut self) -> &mut S {
        &mut self.scorer
    }

    pub fn scorer(&self) -> &S {
        &self.scorer
    }

    pub fn expand_tree(&mut self, context: &[u32]) -> Result<DraftTree> {
        let mut tree = DraftTree::default();
        if self.cfg.max_depth == 0 || self.cfg.branch_factor == 0 || self.cfg.total_budget == 0 {
            return Ok(tree);
        }

        let mut frontier: Vec<(Option<usize>, f32, Vec<u32>)> =
            vec![(None, 0.0_f32, context.to_vec())];

        for depth in 1..=self.cfg.max_depth {
            if frontier.is_empty() {
                break;
            }
            let remaining = self.cfg.total_budget.saturating_sub(tree.len());
            if remaining == 0 {
                break;
            }

            let mut candidates: Vec<(Option<usize>, u32, f32, Vec<u32>)> = Vec::new();

            for (parent_idx, parent_logp, prefix) in &frontier {
                let logits = self.scorer.score(prefix)?;
                anyhow::ensure!(
                    logits.len() == self.cfg.vocab_size,
                    "DraftScorer returned {} logits, expected vocab_size={}",
                    logits.len(),
                    self.cfg.vocab_size,
                );
                let log_probs = log_softmax(&logits);

                let topk = crate::util::top_k_indices(&log_probs, self.cfg.branch_factor);
                for tok in topk {
                    let edge_lp = log_probs[tok];
                    let joint = parent_logp + edge_lp;
                    let mut child_prefix = prefix.clone();
                    child_prefix.push(tok as u32);
                    candidates.push((*parent_idx, tok as u32, joint, child_prefix));
                }
            }

            if candidates.is_empty() {
                break;
            }

            candidates.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal));
            candidates.truncate(remaining);

            let mut next_frontier: Vec<(Option<usize>, f32, Vec<u32>)> =
                Vec::with_capacity(candidates.len());
            for (parent_idx, tok, joint, prefix) in candidates {
                let new_idx = tree.tokens.len();
                tree.tokens.push(tok);
                tree.parents.push(parent_idx);
                tree.depths.push(depth);
                next_frontier.push((Some(new_idx), joint, prefix));
            }
            frontier = next_frontier;

            if tree.len() >= self.cfg.total_budget {
                break;
            }
        }

        Ok(tree)
    }
}

pub fn flatten_with_mask(tree: &DraftTree) -> (Vec<u32>, Vec<u8>) {
    let n = tree.len();
    let mut mask = vec![0u8; n * n];
    for i in 0..n {
        let mut cur = Some(i);
        while let Some(idx) = cur {
            mask[i * n + idx] = 1;
            cur = tree.parents[idx];
        }
    }
    (tree.tokens.clone(), mask)
}

fn log_softmax(logits: &[f32]) -> Vec<f32> {
    if logits.is_empty() {
        return Vec::new();
    }
    let max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut sum_exp = 0.0_f64;
    for &x in logits {
        sum_exp += ((x - max) as f64).exp();
    }
    let log_z = (sum_exp.ln() as f32) + max;
    logits.iter().map(|x| x - log_z).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn uniform_scorer(vocab_size: usize) -> impl FnMut(&[u32]) -> Result<Vec<f32>> {
        move |_ctx: &[u32]| Ok(vec![0.0_f32; vocab_size])
    }

    fn const_scorer(logits: Vec<f32>) -> impl FnMut(&[u32]) -> Result<Vec<f32>> {
        move |_ctx: &[u32]| Ok(logits.clone())
    }

    #[test]
    fn linear_chain_when_branch_factor_is_one() {
        let cfg = Eagle3Config {
            max_depth: 5,
            branch_factor: 1,
            total_budget: 16,
            vocab_size: 8,
        };

        let logits = vec![0.0, 0.0, 0.0, 10.0, 0.0, 0.0, 0.0, 0.0];
        let mut p = Eagle3Proposer::new(const_scorer(logits), cfg);
        let tree = p.expand_tree(&[42]).unwrap();
        assert_eq!(tree.len(), 5, "chain length must equal max_depth");

        for (i, d) in tree.depths.iter().enumerate() {
            assert_eq!(*d, i + 1);
        }

        assert!(tree.tokens.iter().all(|&t| t == 3));

        assert_eq!(tree.parents[0], None);
        for i in 1..tree.len() {
            assert_eq!(tree.parents[i], Some(i - 1));
        }
    }

    #[test]
    fn tree_shape_respects_total_budget() {
        let cfg = Eagle3Config {
            max_depth: 6,
            branch_factor: 4,
            total_budget: 7,
            vocab_size: 32,
        };
        let mut p = Eagle3Proposer::new(uniform_scorer(32), cfg.clone());
        let tree = p.expand_tree(&[1, 2, 3]).unwrap();
        assert!(
            tree.len() <= cfg.total_budget,
            "tree size {} exceeded total_budget {}",
            tree.len(),
            cfg.total_budget,
        );

        for w in tree.depths.windows(2) {
            assert!(w[0] <= w[1], "depths not in BFS order: {:?}", tree.depths);
        }
    }

    #[test]
    fn top_k_selection_picks_highest_logits() {
        let logits = vec![-10.0, -10.0, 5.0, -10.0, 9.0, 1.0];
        let cfg = Eagle3Config {
            max_depth: 1,
            branch_factor: 3,
            total_budget: 10,
            vocab_size: 6,
        };
        let mut p = Eagle3Proposer::new(const_scorer(logits.clone()), cfg);
        let tree = p.expand_tree(&[0]).unwrap();
        assert_eq!(tree.tokens, vec![4, 2, 5]);

        assert!(tree.depths.iter().all(|&d| d == 1));
        assert!(tree.parents.iter().all(|p| p.is_none()));

        assert_eq!(crate::util::top_k_indices(&logits, 3), vec![4, 2, 5]);
    }

    #[test]
    fn parents_form_a_valid_tree() {
        let cfg = Eagle3Config {
            max_depth: 4,
            branch_factor: 3,
            total_budget: 20,
            vocab_size: 16,
        };

        let mut p = Eagle3Proposer::new(
            |ctx: &[u32]| {
                let mut v = vec![0.0_f32; 16];

                let seed = *ctx.last().unwrap_or(&0) as f32;
                for (i, x) in v.iter_mut().enumerate() {
                    *x = ((i as f32) * 0.5 + seed * 0.13).sin();
                }
                Ok(v)
            },
            cfg,
        );
        let tree = p.expand_tree(&[7, 11]).unwrap();
        assert!(!tree.is_empty());
        for i in 0..tree.len() {
            match tree.parents[i] {
                None => {
                    assert_eq!(tree.depths[i], 1, "root must be at depth 1");
                }
                Some(parent) => {
                    assert!(parent < i, "parent {} must precede child {}", parent, i);
                    assert_eq!(
                        tree.depths[i],
                        tree.depths[parent] + 1,
                        "child depth must be parent depth + 1",
                    );
                }
            }
        }
    }

    #[test]
    fn expand_tree_returns_tokens_in_bfs_order() {
        let cfg = Eagle3Config {
            max_depth: 3,
            branch_factor: 2,
            total_budget: 12,
            vocab_size: 8,
        };
        let mut p = Eagle3Proposer::new(uniform_scorer(8), cfg);
        let tree = p.expand_tree(&[5]).expect("expand_tree");
        assert!(!tree.is_empty());

        for w in tree.depths.windows(2) {
            assert!(w[0] <= w[1]);
        }

        for i in 0..tree.len() {
            if let Some(par) = tree.parents[i] {
                assert!(par < i);
            }
        }
    }

    #[test]
    fn flatten_with_mask_is_tree_causal() {
        let tree = DraftTree {
            tokens: vec![100, 200, 300],
            parents: vec![None, None, Some(0)],
            depths: vec![1, 1, 2],
        };
        let (toks, mask) = flatten_with_mask(&tree);
        assert_eq!(toks, vec![100, 200, 300]);
        let n = 3;

        for i in 0..n {
            assert_eq!(mask[i * n + i], 1);
        }

        assert_eq!(mask[1], 0);
        assert_eq!(mask[n], 0);

        assert_eq!(mask[2 * n], 1);
        assert_eq!(mask[2 * n + 1], 0);
        assert_eq!(mask[2 * n + 2], 1);
    }
}
