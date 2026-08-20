use crate::sequence::{Sequence, MIN_RECOMPUTED_PREFILL_TOKENS};
use anyhow::{bail, Result};
use std::collections::{HashMap, VecDeque};
use xxhash_rust::xxh3::xxh3_64_with_seed;

pub const ADAPTER_SEED_TAG: u64 = 0x9e37_79b9_7f4a_7c15;

#[derive(Debug)]
pub struct Block {
    pub id: u32,
    pub ref_count: u32,
    pub hash: Option<u64>,
    pub tokens: Vec<u32>,
    pub kv_computed: bool,
}

impl Block {
    fn new(id: u32) -> Self {
        Self {
            id,
            ref_count: 0,
            hash: None,
            tokens: Vec::new(),
            kv_computed: false,
        }
    }

    fn reset(&mut self) {
        self.ref_count = 1;
        self.hash = None;
        self.tokens.clear();
        self.kv_computed = false;
    }
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct Allocation {
    pub block_table: Vec<u32>,
    pub cached_tokens: usize,
}

pub struct BlockManager {
    pub block_size: usize,
    pub num_blocks: usize,
    pub blocks: Vec<Block>,
    pub free: VecDeque<u32>,
    pub hash_to_block: HashMap<u64, u32>,
}

impl BlockManager {
    pub fn new(num_blocks: usize, block_size: usize) -> Self {
        let blocks = (0..num_blocks as u32).map(Block::new).collect();
        let free = (0..num_blocks as u32).collect();
        Self {
            block_size,
            num_blocks,
            blocks,
            free,
            hash_to_block: HashMap::new(),
        }
    }

    pub fn num_free(&self) -> usize {
        self.free.len()
    }

    pub fn compute_block_hash(tokens: &[u32], prefix: u64) -> u64 {
        Self::compute_block_hash_tagged(tokens, prefix, None)
    }

    pub fn compute_block_hash_tagged(tokens: &[u32], prefix: u64, adapter: Option<&str>) -> u64 {
        let seed = match adapter {
            Some(name) => xxh3_64_with_seed(name.as_bytes(), prefix ^ ADAPTER_SEED_TAG),
            None => prefix,
        };
        let bytes = unsafe {
            std::slice::from_raw_parts(tokens.as_ptr() as *const u8, std::mem::size_of_val(tokens))
        };
        xxh3_64_with_seed(bytes, seed)
    }

    fn pop_free(&mut self) -> Result<u32> {
        let Some(id) = self.free.pop_front() else {
            bail!("block manager out of memory: no free blocks");
        };
        let block = &mut self.blocks[id as usize];
        debug_assert_eq!(block.ref_count, 0);
        if let Some(h) = block.hash.take() {
            if self.hash_to_block.get(&h).copied() == Some(id) {
                self.hash_to_block.remove(&h);
            }
        }
        block.reset();
        Ok(id)
    }

    fn share_or_pop(&mut self, hash: u64, tokens: &[u32]) -> Result<(u32, bool)> {
        if let Some(&id) = self.hash_to_block.get(&hash) {
            if self.blocks[id as usize].tokens == tokens {
                let block = &mut self.blocks[id as usize];
                if block.ref_count == 0 {
                    self.free.retain(|x| *x != id);
                    block.ref_count = 1;
                } else {
                    block.ref_count += 1;
                }
                return Ok((id, true));
            }
        }
        let id = self.pop_free()?;
        let block = &mut self.blocks[id as usize];
        block.tokens = tokens.to_vec();
        block.hash = Some(hash);
        self.hash_to_block.insert(hash, id);
        Ok((id, false))
    }

    pub fn allocate_for(&mut self, seq: &Sequence) -> Result<Vec<u32>> {
        self.allocate_for_tagged(seq, None)
    }

    pub fn allocate_for_tagged(
        &mut self,
        seq: &Sequence,
        adapter: Option<&str>,
    ) -> Result<Vec<u32>> {
        Ok(self.allocate_with_prefix_tagged(seq, adapter)?.block_table)
    }

    pub fn allocate_with_prefix(&mut self, seq: &Sequence) -> Result<Allocation> {
        self.allocate_with_prefix_tagged(seq, None)
    }

    pub fn allocate_with_prefix_tagged(
        &mut self,
        seq: &Sequence,
        adapter: Option<&str>,
    ) -> Result<Allocation> {
        if !seq.block_table.is_empty() {
            bail!("allocate_for: sequence {} already has blocks", seq.id);
        }
        let total = seq.total_len();
        if total == 0 {
            return Ok(Allocation::default());
        }
        let total_blocks = seq.num_blocks_needed(self.block_size);
        let full_blocks = total / self.block_size;
        let has_partial = !total.is_multiple_of(self.block_size);

        let mut required_fresh: usize = 0;
        let mut plan: Vec<(Option<u64>, Vec<u32>)> = Vec::with_capacity(total_blocks);
        let mut prefix: u64 = 0;
        for i in 0..full_blocks {
            let tokens = seq.block_tokens(i, self.block_size);
            let hash = Self::compute_block_hash_tagged(&tokens, prefix, adapter);
            prefix = hash;

            let shareable = match self.hash_to_block.get(&hash) {
                Some(&id) => self.blocks[id as usize].tokens == tokens,
                None => false,
            };
            match self.hash_to_block.get(&hash) {
                Some(&id) if shareable && self.blocks[id as usize].ref_count == 0 => {
                    required_fresh += 1;
                }
                _ if !shareable => {
                    required_fresh += 1;
                }
                _ => {}
            }
            plan.push((Some(hash), tokens));
        }
        if has_partial {
            let tokens = seq.block_tokens(full_blocks, self.block_size);
            required_fresh += 1;
            plan.push((None, tokens));
        }
        if self.free.len() < required_fresh {
            bail!(
                "block manager out of memory: need {} fresh blocks, have {} free",
                required_fresh,
                self.free.len()
            );
        }

        let mut table = Vec::with_capacity(total_blocks);
        let mut cached_tokens = 0usize;
        let mut leading_run = true;
        for (maybe_hash, tokens) in plan {
            let acquired = match maybe_hash {
                Some(h) => match self.share_or_pop(h, &tokens) {
                    Ok((id, was_shared)) => {
                        if leading_run && was_shared && self.blocks[id as usize].kv_computed {
                            cached_tokens += self.block_size;
                        } else {
                            leading_run = false;
                        }
                        Ok(id)
                    }
                    Err(e) => Err(e),
                },
                None => {
                    leading_run = false;
                    self.pop_free().inspect(|&id| {
                        let block = &mut self.blocks[id as usize];
                        block.tokens = tokens;
                        block.hash = None;
                    })
                }
            };
            match acquired {
                Ok(id) => table.push(id),
                Err(e) => {
                    self.release_ids(&table);
                    return Err(e);
                }
            }
        }
        if cached_tokens + MIN_RECOMPUTED_PREFILL_TOKENS > total {
            cached_tokens = cached_tokens.saturating_sub(self.block_size);
        }
        Ok(Allocation {
            block_table: table,
            cached_tokens,
        })
    }

    fn fully_computed_blocks(&self, seq: &Sequence, valid_tokens: usize) -> usize {
        (valid_tokens / self.block_size).min(seq.block_table.len())
    }

    pub fn mark_kv_computed(&mut self, seq: &Sequence, valid_tokens: usize) {
        let full = self.fully_computed_blocks(seq, valid_tokens);
        for &id in seq.block_table.iter().take(full) {
            self.blocks[id as usize].kv_computed = true;
        }
    }

    pub fn publish_computed_blocks(&mut self, seq: &Sequence, valid_tokens: usize) {
        let full = self.fully_computed_blocks(seq, valid_tokens);
        for i in 0..full {
            let block = &self.blocks[seq.block_table[i] as usize];
            debug_assert!(
                block.kv_computed,
                "publish_computed_blocks ran ahead of mark_kv_computed: a hash reachable by \
                 another sequence would advertise KV the model has not written"
            );
            if block.hash.is_none() {
                self.commit_filled_block(seq, i);
            }
        }
    }

    fn release_ids(&mut self, ids: &[u32]) {
        for &id in ids.iter().rev() {
            let block = &mut self.blocks[id as usize];
            if block.ref_count == 0 {
                continue;
            }
            block.ref_count -= 1;
            if block.ref_count == 0 {
                self.free.push_back(id);
            }
        }
    }

    pub fn deallocate(&mut self, seq: &Sequence) {
        for &id in seq.block_table.iter().rev() {
            let block = &mut self.blocks[id as usize];
            if block.ref_count == 0 {
                continue;
            }
            block.ref_count -= 1;
            if block.ref_count == 0 {
                self.free.push_back(id);
            }
        }
    }

    pub fn extend_for(&mut self, seq: &mut Sequence) -> Result<()> {
        let needed = seq.num_blocks_needed(self.block_size);
        if needed <= seq.block_table.len() {
            return Ok(());
        }
        let to_add = needed - seq.block_table.len();
        for _ in 0..to_add {
            let id = self.pop_free()?;
            seq.block_table.push(id);
        }
        Ok(())
    }

    pub fn extend_for_slots(&mut self, seq: &mut Sequence, extra: usize) -> Result<()> {
        let target = seq.total_len() + extra;
        let needed = if target == 0 {
            0
        } else {
            target.div_ceil(self.block_size)
        };
        if needed <= seq.block_table.len() {
            return Ok(());
        }
        let to_add = needed - seq.block_table.len();
        if self.free.len() < to_add {
            bail!(
                "block manager out of memory: need {} blocks for {} slots, have {} free",
                to_add,
                extra,
                self.free.len()
            );
        }
        for _ in 0..to_add {
            let id = self.pop_free()?;
            seq.block_table.push(id);
        }
        Ok(())
    }

    pub fn ref_count(&self, id: u32) -> u32 {
        self.blocks[id as usize].ref_count
    }

    pub fn commit_filled_block(&mut self, seq: &Sequence, logical_idx: usize) {
        self.commit_filled_block_tagged(seq, logical_idx, None)
    }

    pub fn commit_filled_block_tagged(
        &mut self,
        seq: &Sequence,
        logical_idx: usize,
        adapter: Option<&str>,
    ) {
        if logical_idx >= seq.block_table.len() {
            return;
        }
        let tokens = seq.block_tokens(logical_idx, self.block_size);
        if tokens.len() != self.block_size {
            return;
        }
        let mut prefix: u64 = 0;
        for i in 0..logical_idx {
            let prev = seq.block_tokens(i, self.block_size);
            if prev.len() != self.block_size {
                return;
            }
            prefix = Self::compute_block_hash_tagged(&prev, prefix, adapter);
        }
        let hash = Self::compute_block_hash_tagged(&tokens, prefix, adapter);
        let id = seq.block_table[logical_idx];
        let block = &mut self.blocks[id as usize];
        if block.hash.is_some() {
            return;
        }
        block.tokens = tokens;
        block.hash = Some(hash);
        self.hash_to_block.entry(hash).or_insert(id);
    }

    pub fn fork_tail_for_write(&mut self, seq: &mut Sequence) -> Result<Option<CowCopy>> {
        let Some(&tail) = seq.block_table.last() else {
            return Ok(None);
        };
        let logical_idx = seq.block_table.len() - 1;
        if self.blocks[tail as usize].ref_count <= 1 {
            return Ok(None);
        }
        let new_id = self.pop_free()?;
        {
            let src_tokens = self.blocks[tail as usize].tokens.clone();
            let dst = &mut self.blocks[new_id as usize];
            dst.ref_count = 1;
            dst.hash = None;
            dst.tokens = src_tokens;
        }
        self.blocks[tail as usize].ref_count -= 1;
        seq.block_table[logical_idx] = new_id;
        Ok(Some(CowCopy {
            logical_idx,
            src_block: tail,
            dst_block: new_id,
        }))
    }

    pub fn used_blocks(&self) -> usize {
        self.num_blocks - self.free.len()
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CowCopy {
    pub logical_idx: usize,
    pub src_block: u32,
    pub dst_block: u32,
}

#[derive(Clone, Copy, Debug)]
pub struct PoolGeometry {
    pub num_layers: usize,
    pub block_size: usize,
    kv_elems_per_slot_all_layers: usize,
    scale_elems_per_slot_all_layers: usize,
}

impl PoolGeometry {
    pub fn uniform(num_layers: usize, block_size: usize, n_kv: usize, head_dim: usize) -> Self {
        Self {
            num_layers,
            block_size,
            kv_elems_per_slot_all_layers: num_layers * n_kv * head_dim,
            scale_elems_per_slot_all_layers: num_layers * n_kv,
        }
    }

    pub fn from_layer_shapes(block_size: usize, layer_shapes: &[(usize, usize)]) -> Self {
        let kv: usize = layer_shapes.iter().map(|(n_kv, hd)| n_kv * hd).sum();
        let sc: usize = layer_shapes.iter().map(|(n_kv, _)| *n_kv).sum();
        Self {
            num_layers: layer_shapes.len(),
            block_size,
            kv_elems_per_slot_all_layers: kv,
            scale_elems_per_slot_all_layers: sc,
        }
    }

    pub fn bytes_per_block(&self) -> usize {
        let slots = self.block_size;
        let kv_bytes = 2 * slots * self.kv_elems_per_slot_all_layers;
        let scale_bytes =
            2 * slots * self.scale_elems_per_slot_all_layers * std::mem::size_of::<f32>();
        kv_bytes + scale_bytes
    }

    pub fn pool_bytes(&self, num_blocks: usize) -> usize {
        num_blocks * self.bytes_per_block()
    }

    pub fn max_blocks_for_budget(&self, budget_bytes: usize) -> usize {
        budget_bytes
            .checked_div(self.bytes_per_block())
            .unwrap_or(0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sequence::{Sequence, SequenceState};

    fn waiting_seq(id: u64, prompt: Vec<u32>) -> Sequence {
        Sequence::new(id, prompt, 16)
    }

    #[test]
    fn first_sequence_allocates_fresh_blocks() {
        let mut bm = BlockManager::new(8, 4);
        let seq = waiting_seq(1, vec![10, 11, 12, 13, 20, 21]);
        let table = bm.allocate_for(&seq).unwrap();
        assert_eq!(table.len(), 2);
        assert_eq!(bm.ref_count(table[0]), 1);
        assert_eq!(bm.ref_count(table[1]), 1);
        assert_eq!(bm.num_free(), 6);
    }

    #[test]
    fn shared_prefix_increments_refcount() {
        let mut bm = BlockManager::new(8, 4);
        let s1 = waiting_seq(1, vec![1, 2, 3, 4, 5, 6]);
        let t1 = bm.allocate_for(&s1).unwrap();
        assert_eq!(t1.len(), 2);

        let s2 = waiting_seq(2, vec![1, 2, 3, 4, 9, 9]);
        let t2 = bm.allocate_for(&s2).unwrap();
        assert_eq!(t2.len(), 2);

        assert_eq!(t1[0], t2[0]);
        assert_ne!(t1[1], t2[1]);

        assert_eq!(bm.ref_count(t1[0]), 2);
        assert_eq!(bm.ref_count(t1[1]), 1);
        assert_eq!(bm.ref_count(t2[1]), 1);
    }

    #[test]
    fn sharing_a_block_whose_kv_was_never_computed_is_not_a_cache_hit() {
        let mut bm = BlockManager::new(8, 4);
        let mut s1 = waiting_seq(1, vec![1, 2, 3, 4, 5, 6, 7, 8]);
        s1.block_table = bm.allocate_for(&s1).unwrap();

        let s2 = waiting_seq(2, vec![1, 2, 3, 4, 5, 6, 7, 8, 9]);
        let alloc = bm.allocate_with_prefix(&s2).unwrap();
        assert_eq!(alloc.block_table[0], s1.block_table[0]);
        assert_eq!(
            alloc.cached_tokens, 0,
            "allocate registers a block's hash before the model has written its KV; \
             reusing that block's bytes would attend to uninitialised memory"
        );
    }

    #[test]
    fn cache_hits_count_only_the_leading_contiguous_run() {
        let mut bm = BlockManager::new(8, 4);
        let mut s1 = waiting_seq(1, vec![1, 2, 3, 4, 5, 6, 7, 8]);
        s1.block_table = bm.allocate_for(&s1).unwrap();
        assert_eq!(s1.block_table.len(), 2);
        bm.blocks[s1.block_table[1] as usize].kv_computed = true;

        let s2 = waiting_seq(2, vec![1, 2, 3, 4, 5, 6, 7, 8, 9]);
        let alloc = bm.allocate_with_prefix(&s2).unwrap();
        assert_eq!(alloc.block_table[1], s1.block_table[1]);
        assert_eq!(
            alloc.cached_tokens, 0,
            "block 1 is valid but block 0 is not: attention over block 1 needs every \
             preceding key, so a run that does not start at token 0 buys nothing"
        );
    }

    #[test]
    fn a_fully_cached_prompt_still_recomputes_its_last_block() {
        let mut bm = BlockManager::new(8, 4);
        let mut s1 = waiting_seq(1, vec![1, 2, 3, 4, 5, 6, 7, 8]);
        s1.block_table = bm.allocate_for(&s1).unwrap();
        bm.mark_kv_computed(&s1, 8);

        let s2 = waiting_seq(2, vec![1, 2, 3, 4, 5, 6, 7, 8]);
        let alloc = bm.allocate_with_prefix(&s2).unwrap();
        assert_eq!(
            alloc.cached_tokens, 4,
            "prefill must still run the last block: the first sampled token comes from \
             the logits of the final position, which a fully skipped prompt never produces"
        );
        assert!(alloc.cached_tokens + MIN_RECOMPUTED_PREFILL_TOKENS <= s2.total_len());
    }

    #[test]
    fn a_freed_block_revived_by_its_hash_keeps_its_computed_kv() {
        let mut bm = BlockManager::new(8, 4);
        let mut s1 = waiting_seq(1, vec![1, 2, 3, 4, 5, 6, 7, 8]);
        s1.block_table = bm.allocate_for(&s1).unwrap();
        bm.mark_kv_computed(&s1, 8);
        bm.deallocate(&s1);
        assert_eq!(bm.num_free(), 8);

        let s2 = waiting_seq(2, vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
        let alloc = bm.allocate_with_prefix(&s2).unwrap();
        assert_eq!(alloc.block_table[0], s1.block_table[0]);
        assert_eq!(alloc.block_table[1], s1.block_table[1]);
        assert_eq!(
            alloc.cached_tokens, 8,
            "a ref_count==0 block is only ever re-issued through pop_free, which drops \
             its hash first, so nothing can overwrite it while it is still findable"
        );
    }

    #[test]
    fn popping_a_free_block_retires_the_hash_that_would_have_revived_it() {
        let mut bm = BlockManager::new(2, 4);
        let mut s1 = waiting_seq(1, vec![1, 2, 3, 4]);
        s1.block_table = bm.allocate_for(&s1).unwrap();
        bm.mark_kv_computed(&s1, 4);
        let reused = s1.block_table[0];
        bm.deallocate(&s1);

        let mut filler = waiting_seq(2, vec![9, 9, 9, 9, 9, 9, 9, 9]);
        filler.block_table = bm.allocate_for(&filler).unwrap();
        assert!(filler.block_table.contains(&reused));

        bm.deallocate(&filler);
        let s3 = waiting_seq(3, vec![1, 2, 3, 4, 5]);
        let alloc = bm.allocate_with_prefix(&s3).unwrap();
        assert_eq!(
            alloc.cached_tokens, 0,
            "the block that held this prefix was handed out and overwritten; its hash \
             must not survive to advertise stale KV"
        );
    }

    fn decode_and_publish(bm: &mut BlockManager, id: u64, steps: u32) -> Sequence {
        let mut s = waiting_seq(id, vec![1, 2, 3]);
        s.block_table = bm.allocate_for(&s).unwrap();
        s.state = SequenceState::Decode;
        assert!(bm.blocks[s.block_table[0] as usize].hash.is_none());
        for tok in 4..4 + steps {
            s.append_token(tok).unwrap();
            bm.extend_for(&mut s).unwrap();
            let valid_tokens = s.total_len() - 1;
            bm.mark_kv_computed(&s, valid_tokens);
            bm.publish_computed_blocks(&s, valid_tokens);
        }
        s
    }

    #[test]
    fn a_block_is_published_only_once_its_last_position_has_kv() {
        let mut bm = BlockManager::new(8, 4);
        let s1 = decode_and_publish(&mut bm, 1, 5);

        assert_eq!(s1.block_table.len(), 2);
        assert!(
            bm.blocks[s1.block_table[0] as usize].hash.is_some(),
            "decode filled this block and every one of its positions has been through the \
             model; allocate only ever registers prompt-aligned blocks, so nothing else will \
             ever publish it"
        );
        assert!(
            bm.blocks[s1.block_table[1] as usize].hash.is_none(),
            "this block holds four token ids but the model has not run on the last of them: \
             its KV is uninitialised and a later append can still change what it contains"
        );

        let s2 = waiting_seq(2, vec![1, 2, 3, 4, 5, 6, 7, 8, 9]);
        let alloc = bm.allocate_with_prefix(&s2).unwrap();
        assert_eq!(alloc.block_table[0], s1.block_table[0]);
        assert_eq!(
            alloc.cached_tokens, 4,
            "the reusable run ends at the first unpublished block"
        );
    }

    #[test]
    fn retiring_a_duplicate_publication_leaves_the_live_mapping_intact() {
        let mut bm = BlockManager::new(6, 4);
        let s1 = decode_and_publish(&mut bm, 1, 2);
        let s2 = decode_and_publish(&mut bm, 2, 2);
        assert_ne!(s2.block_table[0], s1.block_table[0]);
        assert_eq!(
            bm.blocks[s1.block_table[0] as usize].hash,
            bm.blocks[s2.block_table[0] as usize].hash,
            "two sequences that decoded the same tokens publish the same prefix hash, which \
             only became possible once decode published anything at all"
        );

        bm.deallocate(&s2);
        let mut filler = waiting_seq(3, (100..116).collect());
        filler.block_table = bm.allocate_for(&filler).unwrap();
        assert!(filler.block_table.contains(&s2.block_table[0]));
        bm.deallocate(&filler);

        let s4 = waiting_seq(4, vec![1, 2, 3, 4, 9]);
        let alloc = bm.allocate_with_prefix(&s4).unwrap();
        assert_eq!(alloc.block_table[0], s1.block_table[0]);
        assert_eq!(
            alloc.cached_tokens, 4,
            "pop_free retires only a hash the map still points at, so handing out the losing \
             duplicate must not unmap the block that still holds that prefix's KV"
        );
    }

    #[test]
    fn deallocate_drops_refcount_only() {
        let mut bm = BlockManager::new(8, 4);
        let mut s1 = waiting_seq(1, vec![1, 2, 3, 4, 5, 6]);
        s1.block_table = bm.allocate_for(&s1).unwrap();
        let mut s2 = waiting_seq(2, vec![1, 2, 3, 4, 9, 9]);
        s2.block_table = bm.allocate_for(&s2).unwrap();

        let shared = s1.block_table[0];
        assert_eq!(bm.ref_count(shared), 2);

        bm.deallocate(&s1);
        assert_eq!(bm.ref_count(shared), 1);
        assert!(!bm.free.contains(&shared));

        bm.deallocate(&s2);
        assert_eq!(bm.ref_count(shared), 0);
        assert!(bm.free.contains(&shared));
    }

    #[test]
    fn allocate_fails_when_out_of_memory() {
        let mut bm = BlockManager::new(2, 4);
        let s1 = waiting_seq(1, vec![1, 2, 3, 4, 5, 6, 7, 8, 9]);
        let err = bm.allocate_for(&s1);
        assert!(err.is_err());
    }

    #[test]
    fn extend_for_allocates_one_block_at_boundary() {
        let mut bm = BlockManager::new(8, 4);
        let mut seq = waiting_seq(1, vec![1, 2, 3, 4]);
        seq.state = SequenceState::Decode;
        seq.block_table = bm.allocate_for(&seq).unwrap();
        assert_eq!(seq.block_table.len(), 1);
        let free_before = bm.num_free();

        seq.output.push(99);
        bm.extend_for(&mut seq).unwrap();
        assert_eq!(seq.block_table.len(), 2);
        assert_eq!(bm.num_free(), free_before - 1);

        seq.output.push(100);
        bm.extend_for(&mut seq).unwrap();
        assert_eq!(seq.block_table.len(), 2);
    }

    #[test]
    fn commit_filled_block_registers_hash_for_later_sharing() {
        let mut bm = BlockManager::new(8, 4);
        let mut s1 = waiting_seq(1, vec![1, 2, 3]);
        s1.state = SequenceState::Decode;
        s1.block_table = bm.allocate_for(&s1).unwrap();
        assert_eq!(s1.block_table.len(), 1);
        s1.output.push(4);
        bm.extend_for(&mut s1).unwrap();
        bm.commit_filled_block(&s1, 0);
        let id0 = s1.block_table[0];
        assert!(bm.blocks[id0 as usize].hash.is_some());

        let s2 = waiting_seq(2, vec![1, 2, 3, 4, 9]);
        let t2 = bm.allocate_for(&s2).unwrap();
        assert_eq!(t2[0], id0);
        assert_eq!(bm.ref_count(id0), 2);
    }

    #[test]
    fn fork_tail_for_write_copies_shared_tail() {
        let mut bm = BlockManager::new(8, 4);
        let mut s1 = waiting_seq(1, vec![1, 2, 3, 4]);
        s1.block_table = bm.allocate_for(&s1).unwrap();
        let mut s2 = waiting_seq(2, vec![1, 2, 3, 4]);
        s2.block_table = bm.allocate_for(&s2).unwrap();
        let shared = s1.block_table[0];
        assert_eq!(bm.ref_count(shared), 2);

        let cow = bm.fork_tail_for_write(&mut s2).unwrap().unwrap();
        assert_eq!(cow.src_block, shared);
        assert_eq!(cow.logical_idx, 0);
        assert_ne!(cow.dst_block, shared);
        assert_eq!(s2.block_table[0], cow.dst_block);
        assert_eq!(bm.ref_count(shared), 1);
        assert_eq!(bm.ref_count(cow.dst_block), 1);
        assert_eq!(bm.blocks[cow.dst_block as usize].tokens, vec![1, 2, 3, 4]);
    }

    #[test]
    fn fork_tail_for_write_noop_when_unshared() {
        let mut bm = BlockManager::new(8, 4);
        let mut s1 = waiting_seq(1, vec![1, 2, 3, 4]);
        s1.block_table = bm.allocate_for(&s1).unwrap();
        assert_eq!(bm.ref_count(s1.block_table[0]), 1);
        assert!(bm.fork_tail_for_write(&mut s1).unwrap().is_none());
    }

    #[test]
    fn pool_geometry_uniform_accounting() {
        let g = PoolGeometry::uniform(48, 16, 8, 256);
        let kv_bytes = 2 * 16 * 8 * 256;
        let scale_bytes = 2 * 16 * 8 * 4;
        assert_eq!(g.bytes_per_block(), 48 * (kv_bytes + scale_bytes));
        assert_eq!(g.pool_bytes(10), 10 * g.bytes_per_block());
        let budget = 4usize * 1024 * 1024 * 1024;
        assert_eq!(
            g.max_blocks_for_budget(budget),
            budget / g.bytes_per_block()
        );
    }

    #[test]
    fn pool_geometry_mixed_layer_shapes() {
        let shapes = vec![(8usize, 256usize), (4, 256), (8, 512)];
        let g = PoolGeometry::from_layer_shapes(16, &shapes);
        let kv_elems: usize = shapes.iter().map(|(k, h)| k * h).sum();
        let sc_elems: usize = shapes.iter().map(|(k, _)| *k).sum();
        let expected = 2 * 16 * kv_elems + 2 * 16 * sc_elems * 4;
        assert_eq!(g.bytes_per_block(), expected);
    }

    #[test]
    fn paged_pool_conservation_refcounts_and_content_sharing() {
        let num_blocks = 16usize;
        let bs = 4usize;
        let mut bm = BlockManager::new(num_blocks, bs);
        let mut rng = 0xabcdef01u64;
        let mut next = move |m: u64| {
            rng = rng
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (rng >> 33) % m
        };

        let mut live: Vec<Sequence> = Vec::new();
        for round in 0..200u64 {
            let action = next(3);

            if action < 2 && live.len() < 5 && bm.num_free() >= 4 {
                let mut prompt: Vec<u32> = Vec::new();
                if let Some(other) = live.first() {
                    if next(2) == 0 {
                        let take = (next(other.prompt.len() as u64 + 1)) as usize;
                        prompt.extend_from_slice(&other.prompt[..take]);
                    }
                }
                while prompt.len() < 1 + (next(10) as usize) {
                    prompt.push(next(7) as u32);
                }
                let mut seq = Sequence::new(round, prompt, 16);
                if let Ok(table) = bm.allocate_for(&seq) {
                    seq.block_table = table;
                    live.push(seq);
                }
            } else if !live.is_empty() {
                let i = (next(live.len() as u64)) as usize;
                let seq = live.swap_remove(i);
                bm.deallocate(&seq);
            }

            for b in 0..num_blocks as u32 {
                let holders = live
                    .iter()
                    .map(|s| s.block_table.iter().filter(|&&x| x == b).count())
                    .sum::<usize>() as u32;
                assert_eq!(
                    bm.ref_count(b),
                    holders,
                    "round {round}: block {b} ref_count {} but {holders} table refs",
                    bm.ref_count(b)
                );
            }

            let referenced: std::collections::HashSet<u32> = live
                .iter()
                .flat_map(|s| s.block_table.iter().copied())
                .collect();
            assert_eq!(
                referenced.len() + bm.num_free(),
                num_blocks,
                "round {round}: pool leak -- {} referenced + {} free != {num_blocks}",
                referenced.len(),
                bm.num_free()
            );

            for (ai, a) in live.iter().enumerate() {
                for b in live.iter().skip(ai + 1) {
                    for (la, &blk) in a.block_table.iter().enumerate() {
                        if let Some(lb) = b.block_table.iter().position(|&x| x == blk) {
                            assert_eq!(
                                a.block_tokens(la, bs),
                                b.block_tokens(lb, bs),
                                "block {blk} shared with different content"
                            );
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn failed_alloc_releases_its_blocks_and_leaves_the_pool_intact() {
        let mut bm = BlockManager::new(3, 4);

        let mut s0 = Sequence::new(0, vec![9, 9, 9, 9], 16);
        s0.block_table = bm.allocate_for(&s0).unwrap();
        assert_eq!(bm.num_free(), 2);

        let mut s1 = Sequence::new(1, vec![1, 2, 3, 4], 16);
        s1.block_table = bm.allocate_for(&s1).unwrap();
        bm.deallocate(&s1);
        assert_eq!(bm.num_free(), 2);

        let s2 = Sequence::new(2, vec![1, 2, 3, 4, 5, 6, 7, 8, 9], 16);
        let res = bm.allocate_for(&s2);
        assert!(res.is_err(), "over-capacity allocation must be refused");

        assert_eq!(
            bm.num_free(),
            2,
            "a refused allocation must leave the free list untouched"
        );
        let leaked: Vec<u32> = (0..3u32)
            .filter(|&b| !s0.block_table.contains(&b) && bm.ref_count(b) > 0)
            .collect();
        assert!(
            leaked.is_empty(),
            "blocks leaked from a failed allocation: {leaked:?}"
        );

        let mut s3 = Sequence::new(3, vec![7, 7, 7, 7], 16);
        s3.block_table = bm
            .allocate_for(&s3)
            .expect("pool still usable after refusal");
        assert_eq!(s3.block_table.len(), 1);

        bm.deallocate(&s0);
        bm.deallocate(&s3);
        assert_eq!(
            bm.num_free(),
            3,
            "every block returns once both sequences release"
        );
    }

    #[test]
    fn release_ids_returns_blocks_exactly_once() {
        let mut bm = BlockManager::new(4, 4);
        let mut s = Sequence::new(0, vec![1, 2, 3, 4, 5, 6, 7, 8], 16);
        s.block_table = bm.allocate_for(&s).unwrap();
        assert_eq!(s.block_table.len(), 2);
        assert_eq!(bm.num_free(), 2);

        let taken = s.block_table.clone();
        bm.release_ids(&taken);
        assert_eq!(bm.num_free(), 4, "released blocks must return to the pool");
        for &id in &taken {
            assert_eq!(bm.ref_count(id), 0);
        }

        bm.release_ids(&taken);
        assert_eq!(bm.num_free(), 4, "double release corrupted the free list");
    }

    #[test]
    fn used_blocks_tracks_allocation() {
        let mut bm = BlockManager::new(8, 4);
        assert_eq!(bm.used_blocks(), 0);
        let mut s1 = waiting_seq(1, vec![1, 2, 3, 4, 5]);
        s1.block_table = bm.allocate_for(&s1).unwrap();
        assert_eq!(bm.used_blocks(), 2);
        bm.deallocate(&s1);
        assert_eq!(bm.used_blocks(), 0);
    }
}
