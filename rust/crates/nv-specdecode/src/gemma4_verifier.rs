#![cfg(any(feature = "cuda", feature = "wgpu"))]

use anyhow::{anyhow, bail, Context, Result};
use candle_core::{DType, Device, Tensor};
use nv_models::gemma4::Gemma4;

use crate::eagle3::DraftTree;

#[derive(Clone, Debug)]
pub struct VerifyStep {
    pub emitted: Vec<u32>,

    pub num_accepted: usize,

    pub aux_hidden: Tensor,
}

pub struct Gemma4Verifier<'a> {
    pub model: &'a Gemma4,
    pub device: Device,
    pub aux_layers: Vec<usize>,

    pub last_aux: Option<Tensor>,
}

impl<'a> Gemma4Verifier<'a> {
    pub fn new(model: &'a Gemma4, device: Device, aux_layers: Vec<usize>) -> Self {
        Self {
            model,
            device,
            aux_layers,
            last_aux: None,
        }
    }

    pub fn verify_tree(
        &mut self,
        context: &[u32],
        tree: &DraftTree,
        tree_mask: Option<&[u8]>,
    ) -> Result<VerifyStep> {
        if context.is_empty() {
            bail!("Gemma4Verifier.verify_tree: empty context");
        }
        let n_tree = tree.len();
        if let Some(mask) = tree_mask {
            if mask.len() != n_tree * n_tree {
                bail!("tree_mask len {} != n*n where n = {n_tree}", mask.len());
            }
        }
        let tree_is_chain = tree_is_lower_tri(tree, tree_mask);

        let mut joint: Vec<u32> = Vec::with_capacity(context.len() + n_tree);
        joint.extend_from_slice(context);
        joint.extend_from_slice(&tree.tokens);
        let seq = joint.len();

        let tokens = Tensor::from_vec(joint.clone(), (1usize, seq), &self.device)
            .context("alloc joint tokens tensor")?;
        let positions = build_tree_positions(context.len(), tree, &self.device)?;

        let (logits, hidden_states) = if tree_is_chain {
            self.model
                .forward_with_aux_hidden(&tokens, &positions, &self.aux_layers)
                .context("Gemma4.forward_with_aux_hidden")?
        } else {
            let full_mask = build_full_attn_mask(context.len(), tree, tree_mask, &self.device)?;
            self.model
                .forward_with_aux_hidden_masked(
                    &tokens,
                    &positions,
                    &self.aux_layers,
                    Some(&full_mask),
                )
                .context("Gemma4.forward_with_aux_hidden_masked")?
        };

        let vocab = *logits
            .dims()
            .last()
            .ok_or_else(|| anyhow!("logits tensor has no dims"))?;
        let data: Vec<f32> = logits.to_dtype(DType::F32)?.flatten_all()?.to_vec1()?;
        if data.len() != seq * vocab {
            bail!("logits len {} != seq*vocab = {}*{}", data.len(), seq, vocab,);
        }

        let pref_last = context.len() - 1;
        let (emitted, num_accepted) = walk_accept(tree, &data, vocab, pref_last, context.len());

        let mut squeezed: Vec<Tensor> = Vec::with_capacity(hidden_states.len());
        for h in &hidden_states {
            let s = h.squeeze(0)?;
            squeezed.push(s);
        }
        let aux_hidden = Tensor::cat(&squeezed.iter().collect::<Vec<_>>()[..], 1)?;
        self.last_aux = Some(aux_hidden.clone());

        Ok(VerifyStep {
            emitted,
            num_accepted,
            aux_hidden,
        })
    }
}

fn tree_is_lower_tri(tree: &DraftTree, mask: Option<&[u8]>) -> bool {
    if tree.is_empty() {
        return true;
    }
    if let Some(m) = mask {
        return m == crate::chain::lower_tri_mask(tree.len()).as_slice();
    }
    for (i, par) in tree.parents.iter().enumerate() {
        if let Some(p) = par {
            if *p + 1 != i {
                return false;
            }
        } else if i != 0 {
            return false;
        }
    }
    true
}

fn build_tree_positions(context_len: usize, tree: &DraftTree, device: &Device) -> Result<Tensor> {
    let n = tree.len();
    let seq = context_len + n;
    let mut pos: Vec<i32> = Vec::with_capacity(seq);
    for i in 0..context_len {
        pos.push(i as i32);
    }
    for i in 0..n {
        let depth = tree.depths[i];
        let abs = context_len + depth - 1;
        pos.push(abs as i32);
    }
    Tensor::from_vec(pos, seq, device).context("alloc tree positions")
}

fn build_full_attn_mask(
    context_len: usize,
    tree: &DraftTree,
    tree_mask: Option<&[u8]>,
    device: &Device,
) -> Result<Tensor> {
    let n = tree.len();
    let seq = context_len + n;
    let mut m = vec![0f32; seq * seq];
    for i in 0..context_len {
        for j in 0..=i {
            m[i * seq + j] = 1.0;
        }
    }
    for ti in 0..n {
        let row = context_len + ti;
        for j in 0..context_len {
            m[row * seq + j] = 1.0;
        }
        if let Some(tm) = tree_mask {
            for tj in 0..n {
                if tm[ti * n + tj] != 0 {
                    m[row * seq + (context_len + tj)] = 1.0;
                }
            }
        } else {
            let mut cur = Some(ti);
            while let Some(idx) = cur {
                m[row * seq + (context_len + idx)] = 1.0;
                cur = tree.parents[idx];
            }
        }
    }
    Tensor::from_vec(m, (seq, seq), device).context("alloc full attn mask")
}

fn walk_accept(
    tree: &DraftTree,
    data: &[f32],
    vocab: usize,
    pref_last: usize,
    context_len: usize,
) -> (Vec<u32>, usize) {
    let n = tree.len();
    let mut emitted: Vec<u32> = Vec::new();
    let mut num_accepted = 0usize;
    if n == 0 {
        let row_start = pref_last * vocab;
        let row = &data[row_start..row_start + vocab];
        emitted.push(argmax_row(row));
        return (emitted, 0);
    }

    let mut children_by_parent: Vec<Vec<usize>> = vec![Vec::new(); n];
    let mut roots: Vec<usize> = Vec::new();
    for (i, par) in tree.parents.iter().enumerate() {
        match par {
            Some(p) => children_by_parent[*p].push(i),
            None => roots.push(i),
        }
    }

    let row_start = pref_last * vocab;
    let row = &data[row_start..row_start + vocab];
    let vtok = argmax_row(row);
    let mut node: Option<usize> = roots.iter().copied().find(|&c| tree.tokens[c] == vtok);
    emitted.push(vtok);
    if node.is_none() {
        return (emitted, 0);
    }
    num_accepted += 1;

    while let Some(cur) = node {
        let next_row = context_len + cur;
        let row_start = next_row * vocab;
        let row = &data[row_start..row_start + vocab];
        let vtok = argmax_row(row);
        let kids = &children_by_parent[cur];
        let picked = kids.iter().copied().find(|&c| tree.tokens[c] == vtok);
        emitted.push(vtok);
        match picked {
            Some(c) => {
                num_accepted += 1;
                node = Some(c);
            }
            None => {
                node = None;
            }
        }
    }
    (emitted, num_accepted)
}

fn argmax_row(row: &[f32]) -> u32 {
    crate::util::argmax_f32(row).0 as u32
}
