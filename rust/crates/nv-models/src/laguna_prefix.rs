use std::collections::VecDeque;

pub const PREFIX_REUSE_ENV: &str = "NV_LAGUNA_PREFIX_REUSE";
pub const PREFIX_LRU_ENTRIES_ENV: &str = "NV_LAGUNA_PREFIX_LRU";
pub const PREFIX_LRU_BYTES_ENV: &str = "NV_LAGUNA_PREFIX_LRU_BYTES";
pub const PREFIX_LRU_ENTRIES_DEFAULT: usize = 4;
pub const PREFIX_LRU_BYTES_DEFAULT: usize = 2 << 30;

pub fn prefix_reuse_enabled() -> bool {
    std::env::var(PREFIX_REUSE_ENV).ok().as_deref() != Some("0")
}

pub fn lru_entries_from_env() -> usize {
    std::env::var(PREFIX_LRU_ENTRIES_ENV)
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(PREFIX_LRU_ENTRIES_DEFAULT)
}

pub fn lru_bytes_from_env() -> usize {
    std::env::var(PREFIX_LRU_BYTES_ENV)
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(PREFIX_LRU_BYTES_DEFAULT)
}

pub fn checkpoint_position(n: usize, chunk: usize) -> usize {
    if n == 0 || chunk == 0 {
        return 0;
    }
    (n - 1) / chunk * chunk
}

pub fn checkpoint_preserves_sliding_layout(
    pos: usize,
    chunk: usize,
    window: usize,
    max_seq: usize,
    next_m: usize,
) -> bool {
    if !matches!(chunk, 256 | 512 | 1024 | 2048)
        || pos == 0
        || !pos.is_multiple_of(chunk)
        || window == 0
        || next_m == 0
        || next_m > chunk
    {
        return false;
    }
    let Some(cap) = window.checked_add(chunk).map(|cap| cap.min(max_seq)) else {
        return false;
    };
    if pos.checked_add(next_m).is_none_or(|end| end > max_seq) {
        return false;
    }
    let prepared_len = |stored: usize| -> Option<usize> {
        let total = stored.checked_add(next_m)?;
        if total <= cap {
            Some(total)
        } else {
            stored
                .min(window)
                .checked_add(next_m)
                .filter(|total| *total <= cap)
        }
    };
    match (prepared_len(pos.min(cap)), prepared_len(pos.min(window))) {
        (Some(cold), Some(restored)) => cold == restored,
        _ => false,
    }
}

pub fn common_prefix_len(a: &[u32], b: &[u32]) -> usize {
    crate::prefix_reuse::common_prefix_len(a, b)
}

#[derive(Debug)]
pub struct Checkpoint<S> {
    pub id: u64,
    pub tokens: Vec<u32>,
    pub pos: usize,
    pub bytes: usize,
    pub state: S,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Hit {
    pub id: u64,
    pub pos: usize,
}

pub struct CheckpointLru<S> {
    entries: VecDeque<Checkpoint<S>>,
    max_entries: usize,
    max_bytes: usize,
    next_id: u64,
    shared_allocation: fn(&S) -> Option<(usize, usize)>,
}

impl<S> CheckpointLru<S> {
    pub fn new(max_entries: usize, max_bytes: usize) -> Self {
        Self::with_shared_allocation(max_entries, max_bytes, |_| None)
    }

    pub fn with_shared_allocation(
        max_entries: usize,
        max_bytes: usize,
        shared_allocation: fn(&S) -> Option<(usize, usize)>,
    ) -> Self {
        Self {
            entries: VecDeque::new(),
            max_entries,
            max_bytes,
            next_id: 1,
            shared_allocation,
        }
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    pub fn bytes(&self) -> usize {
        self.retained_bytes(|_| true).unwrap_or(usize::MAX)
    }

    fn retained_bytes(&self, keep: impl Fn(&Checkpoint<S>) -> bool) -> Option<usize> {
        let mut total = 0usize;
        for (i, e) in self.entries.iter().enumerate() {
            if !keep(e) {
                continue;
            }
            total = total
                .checked_add(e.bytes)?
                .checked_add(token_bytes(e.tokens.capacity())?)?;
            if let Some((identity, bytes)) = (self.shared_allocation)(&e.state) {
                let previous = self
                    .entries
                    .iter()
                    .take(i)
                    .filter(|e| keep(e))
                    .find_map(|e| {
                        (self.shared_allocation)(&e.state).filter(|(other, _)| *other == identity)
                    });
                match previous {
                    Some((_, previous_bytes)) if previous_bytes != bytes => return None,
                    Some(_) => {}
                    None => total = total.checked_add(bytes)?,
                }
            }
        }
        Some(total)
    }

    fn additional_shared_bytes(&self, state: &S) -> Option<usize> {
        let Some((identity, bytes)) = (self.shared_allocation)(state) else {
            return Some(0);
        };
        let mut additional = bytes;
        for e in &self.entries {
            if let Some((other, other_bytes)) = (self.shared_allocation)(&e.state) {
                if other == identity {
                    if other_bytes != bytes {
                        return None;
                    }
                    additional = 0;
                }
            }
        }
        Some(additional)
    }

    pub fn can_admit(&self, extra_entries: usize, extra_bytes: usize, protected: &[u64]) -> bool {
        if self.max_entries == 0 || protected.iter().any(|id| self.get(*id).is_none()) {
            return false;
        }
        let count = self
            .entries
            .iter()
            .filter(|e| protected.contains(&e.id))
            .count();
        count
            .checked_add(extra_entries)
            .is_some_and(|n| n <= self.max_entries)
            && self
                .retained_bytes(|e| protected.contains(&e.id))
                .and_then(|n| n.checked_add(extra_bytes))
                .is_some_and(|n| n <= self.max_bytes)
    }

    fn fits(&self, extra_entries: usize, extra_bytes: usize) -> bool {
        self.len()
            .checked_add(extra_entries)
            .is_some_and(|n| n <= self.max_entries)
            && self
                .retained_bytes(|_| true)
                .and_then(|n| n.checked_add(extra_bytes))
                .is_some_and(|n| n <= self.max_bytes)
    }

    fn make_room(
        &mut self,
        extra_entries: usize,
        extra_bytes: usize,
        protected: &[u64],
        replace: Option<u64>,
    ) -> bool {
        if replace.is_some_and(|id| protected.contains(&id))
            || !self.can_admit(extra_entries, extra_bytes, protected)
        {
            return false;
        }
        if let Some(id) = replace {
            self.remove(id);
        }
        while !self.fits(extra_entries, extra_bytes) {
            let Some(i) = self.entries.iter().position(|e| !protected.contains(&e.id)) else {
                return false;
            };
            self.entries.remove(i);
        }
        true
    }

    pub fn select(&self, prompt: &[u32]) -> Option<Hit> {
        self.select_where(prompt, |_| true)
    }

    pub fn select_where(
        &self,
        prompt: &[u32],
        eligible: impl Fn(&Checkpoint<S>) -> bool,
    ) -> Option<Hit> {
        let mut best: Option<Hit> = None;
        for e in &self.entries {
            if e.pos == 0 || e.pos > e.tokens.len() || e.pos >= prompt.len() || !eligible(e) {
                continue;
            }
            if best.is_some_and(|h| e.pos < h.pos) || common_prefix_len(&e.tokens, prompt) < e.pos {
                continue;
            }
            best = Some(Hit {
                id: e.id,
                pos: e.pos,
            });
        }
        best
    }

    pub fn iter(&self) -> impl Iterator<Item = &Checkpoint<S>> {
        self.entries.iter()
    }

    pub fn get(&self, id: u64) -> Option<&Checkpoint<S>> {
        self.entries.iter().find(|e| e.id == id)
    }

    pub fn get_mut(&mut self, id: u64) -> Option<&mut Checkpoint<S>> {
        self.entries.iter_mut().find(|e| e.id == id)
    }

    pub fn touch(&mut self, id: u64) {
        if let Some(e) = self.remove(id) {
            self.entries.push_back(e);
        }
    }

    fn push(&mut self, tokens: Vec<u32>, pos: usize, bytes: usize, state: S) -> Option<u64> {
        let id = self.next_id;
        self.next_id = self.next_id.checked_add(1)?;
        self.entries.push_back(Checkpoint {
            id,
            tokens,
            pos,
            bytes,
            state,
        });
        Some(id)
    }

    pub fn insert(&mut self, tokens: Vec<u32>, pos: usize, bytes: usize, state: S) -> Option<u64> {
        if pos == 0 || pos > tokens.len() || self.max_entries == 0 || self.next_id == u64::MAX {
            return None;
        }
        let private = bytes.checked_add(token_bytes(tokens.capacity())?)?;
        let shared = (self.shared_allocation)(&state).map_or(0, |(_, bytes)| bytes);
        if private.checked_add(shared)? > self.max_bytes {
            return None;
        }
        self.additional_shared_bytes(&state)?;
        let replace = self
            .entries
            .iter()
            .find(|e| e.pos == pos && e.tokens == tokens)
            .map(|e| e.id);
        if let Some(id) = replace {
            self.remove(id);
        }
        while !self.fits(
            1,
            private.checked_add(self.additional_shared_bytes(&state)?)?,
        ) {
            self.entries.pop_front()?;
        }
        self.push(tokens, pos, bytes, state)
    }

    pub fn insert_with<E>(
        &mut self,
        tokens: &[u32],
        pos: usize,
        reserved_private_bytes: usize,
        protected: &[u64],
        make_state: impl FnOnce() -> Result<(S, usize), E>,
    ) -> Result<Option<u64>, E> {
        let reserved =
            token_bytes(tokens.len()).and_then(|n| n.checked_add(reserved_private_bytes));
        let Some(reserved) = reserved else {
            return Ok(None);
        };
        if pos == 0 || pos > tokens.len() || self.next_id == u64::MAX {
            return Ok(None);
        }
        let replace = self
            .entries
            .iter()
            .find(|e| e.pos == pos && e.tokens == tokens)
            .map(|e| e.id);
        if !self.make_room(1, reserved, protected, replace) {
            return Ok(None);
        }
        let (state, actual_private_bytes) = make_state()?;
        if actual_private_bytes > reserved_private_bytes {
            return Ok(None);
        }
        let tokens = tokens.to_vec();
        let actual = token_bytes(tokens.capacity())
            .and_then(|n| n.checked_add(actual_private_bytes))
            .and_then(|n| n.checked_add(self.additional_shared_bytes(&state)?));
        if !actual.is_some_and(|n| n <= reserved && self.fits(1, n)) {
            return Ok(None);
        }
        Ok(self.push(tokens, pos, actual_private_bytes, state))
    }

    pub fn attach_with<T, E>(
        &mut self,
        protected: &[u64],
        reserved_shared_bytes: usize,
        make_allocation: impl FnOnce() -> Result<(T, usize), E>,
        mut attach: impl FnMut(u64, &mut S, &T),
    ) -> Result<bool, E> {
        if protected.is_empty() || !self.make_room(0, reserved_shared_bytes, protected, None) {
            return Ok(false);
        }
        let before = self.bytes();
        let (allocation, actual_shared_bytes) = make_allocation()?;
        if actual_shared_bytes > reserved_shared_bytes {
            return Ok(false);
        }
        for e in &mut self.entries {
            if protected.contains(&e.id) {
                attach(e.id, &mut e.state, &allocation);
            }
        }
        let valid = self.retained_bytes(|_| true).is_some_and(|after| {
            after <= self.max_bytes
                && before
                    .checked_add(actual_shared_bytes)
                    .is_some_and(|n| after <= n)
        });
        if !valid {
            self.entries.retain(|e| !protected.contains(&e.id));
        }
        Ok(valid)
    }

    pub fn grow(&mut self, id: u64, extra: usize) -> bool {
        let Some(bytes) = self.get(id).and_then(|e| e.bytes.checked_add(extra)) else {
            self.remove(id);
            return false;
        };
        if !self.make_room(0, extra, &[id], None) {
            self.remove(id);
            return false;
        }
        self.get_mut(id)
            .expect("protected checkpoint survives admission")
            .bytes = bytes;
        true
    }

    pub fn remove(&mut self, id: u64) -> Option<Checkpoint<S>> {
        let i = self.entries.iter().position(|e| e.id == id)?;
        self.entries.remove(i)
    }
}

fn token_bytes(capacity: usize) -> Option<usize> {
    capacity.checked_mul(std::mem::size_of::<u32>())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::Cell;
    use std::sync::Arc;

    #[test]
    fn sliding_layout_small_chunks_preserve_every_checkpoint_stage() {
        for (chunk, positions) in [
            (256, [256, 512, 768, 1024, 167936]),
            (512, [512, 1024, 1536, 2048, 167936]),
        ] {
            for pos in positions {
                for next_m in [1, 64, 256, chunk] {
                    assert!(
                        checkpoint_preserves_sliding_layout(pos, chunk, 512, 262144, next_m),
                        "chunk={chunk} pos={pos} next_m={next_m}"
                    );
                }
            }
        }
    }

    #[test]
    fn sliding_layout_first_large_chunk_rejects_through_768_remaining_tokens() {
        for chunk in [1024, 2048] {
            for remaining in 1..=768 {
                let next_m = match remaining {
                    1..=256 => remaining,
                    257..=512 => 256,
                    _ => 512,
                };
                assert!(
                    !checkpoint_preserves_sliding_layout(chunk, chunk, 512, 262144, next_m),
                    "new prompt {} must miss before restoring checkpoint {chunk}",
                    chunk + remaining
                );
            }
            for next_m in [768, 1024] {
                assert!(checkpoint_preserves_sliding_layout(
                    chunk, chunk, 512, 262144, next_m
                ));
            }
        }
    }

    #[test]
    fn sliding_layout_later_and_168k_checkpoints_preserve_next_preparation() {
        for chunk in [1024, 2048] {
            for (pos, next_m) in [(2 * chunk, 1), (3 * chunk, 256), (167936, 64)] {
                assert!(checkpoint_preserves_sliding_layout(
                    pos, chunk, 512, 262144, next_m
                ));
            }
        }
        assert!(!checkpoint_preserves_sliding_layout(
            1024, 1024, 512, 1280, 256
        ));
        assert!(checkpoint_preserves_sliding_layout(512, 256, 512, 768, 256));
    }

    #[test]
    fn sliding_layout_invalid_shapes_and_overflows_refuse_without_panicking() {
        for args in [
            (0, 256, 512, 262144, 1),
            (1024, 0, 512, 262144, 1),
            (1024, 128, 512, 262144, 1),
            (1025, 1024, 512, 262144, 1),
            (1024, 1024, 0, 262144, 1),
            (1024, 1024, 512, 262144, 0),
            (1024, 1024, 512, 262144, 1025),
            (1024, 1024, 512, 0, 1),
            (1024, 1024, 512, 1024, 1),
            (1024, 1024, usize::MAX, usize::MAX, 1),
            (usize::MAX - 255, 256, 512, usize::MAX, 256),
        ] {
            assert!(
                !checkpoint_preserves_sliding_layout(args.0, args.1, args.2, args.3, args.4),
                "{args:?}"
            );
        }
    }

    #[test]
    fn sliding_layout_ineligible_hit_is_a_nonmutating_lru_miss() {
        let mut lru = CheckpointLru::new(2, 16384);
        let id = lru.insert(vec![7; 1024], 1024, 0, ()).unwrap();
        assert!(lru
            .select_where(&vec![7; 1280], |entry| checkpoint_preserves_sliding_layout(
                entry.pos, 1024, 512, 262144, 256
            ))
            .is_none());
        assert_eq!(lru.len(), 1);
        assert!(lru.get(id).is_some());
        let hit = lru
            .select_where(&vec![7; 1793], |entry| {
                checkpoint_preserves_sliding_layout(entry.pos, 1024, 512, 262144, 768)
            })
            .unwrap();
        assert_eq!((hit.id, hit.pos), (id, 1024));
    }

    type Shared = Option<Arc<Vec<u8>>>;

    fn shared_allocation(state: &Shared) -> Option<(usize, usize)> {
        state
            .as_ref()
            .map(|a| (Arc::as_ptr(a) as usize, a.capacity()))
    }

    fn shared_lru(n: usize, bytes: usize) -> CheckpointLru<Shared> {
        CheckpointLru::with_shared_allocation(n, bytes, shared_allocation)
    }

    #[test]
    fn the_checkpoint_sits_at_the_last_cold_boundary_inside_the_prompt() {
        assert_eq!(checkpoint_position(0, 256), 0);
        assert_eq!(checkpoint_position(1, 256), 0);
        assert_eq!(checkpoint_position(256, 256), 0);
        assert_eq!(checkpoint_position(257, 256), 256);
        assert_eq!(checkpoint_position(1000, 256), 768);
        assert_eq!(checkpoint_position(1024, 256), 768);
        assert_eq!(checkpoint_position(1500, 512), 1024);
        assert_eq!(checkpoint_position(usize::MAX, 0), 0);
    }

    #[test]
    fn selection_backtracks_after_thinking_removal_and_ignores_unavailable_snapshots() {
        let mut l = CheckpointLru::new(8, 1 << 30);
        let history: Vec<u32> = (0..2000).collect();
        let a = l.insert(history[..768].to_vec(), 768, 10, true).unwrap();
        let b = l.insert(history[..1536].to_vec(), 1536, 10, false).unwrap();
        assert_eq!(l.select(&history), Some(Hit { id: b, pos: 1536 }));
        assert_eq!(
            l.select_where(&history, |e| e.state),
            Some(Hit { id: a, pos: 768 })
        );
        let mut edited = history.clone();
        edited.drain(1000..1400);
        assert_eq!(l.select(&edited), Some(Hit { id: a, pos: 768 }));
        assert_eq!(l.select(&history[..768]), None);
        assert_eq!(l.select(&[9000; 2000]), None);
        let c = l.insert(history[..800].to_vec(), 768, 10, true).unwrap();
        assert_eq!(
            l.select_where(&history, |e| e.state),
            Some(Hit { id: c, pos: 768 })
        );
        l.touch(a);
        assert_eq!(
            l.select_where(&history, |e| e.state),
            Some(Hit { id: a, pos: 768 })
        );
    }

    #[test]
    fn shared_allocation_stays_charged_after_its_first_owner_is_evicted() {
        let mut l = shared_lru(2, 128);
        let a = l.insert(vec![1], 1, 8, None).unwrap();
        let b = l.insert(vec![2], 1, 8, None).unwrap();
        let allocation = Arc::new(vec![0; 64]);
        let weak = Arc::downgrade(&allocation);
        assert!(l
            .attach_with(
                &[a, b],
                64,
                || Ok::<_, ()>((allocation, 64)),
                |_, s, a| {
                    *s = Some(Arc::clone(a));
                }
            )
            .unwrap());
        assert_eq!(l.bytes(), 88);
        let c = l.insert(vec![3], 1, 8, None).unwrap();
        assert!(l.get(a).is_none());
        assert!(l.get(b).is_some() && l.get(c).is_some());
        assert_eq!(l.bytes(), 88);
        assert!(weak.upgrade().is_some());
        drop(l.remove(b));
        assert_eq!(l.bytes(), 12);
        assert!(weak.upgrade().is_none());
    }

    #[test]
    fn inserting_an_existing_shared_allocation_does_not_evict_its_other_owner() {
        let mut l = shared_lru(2, 88);
        let allocation = Arc::new(vec![0; 64]);
        let a = l
            .insert(vec![1], 1, 8, Some(Arc::clone(&allocation)))
            .unwrap();
        let b = l.insert(vec![2], 1, 8, Some(allocation)).unwrap();
        assert_eq!(l.bytes(), 88);
        assert!(l.get(a).is_some() && l.get(b).is_some());
    }

    #[test]
    fn admission_rejects_impossible_copies_without_materializing_or_evicting() {
        let mut l = shared_lru(2, 100);
        let id = l.insert(vec![1], 1, 8, None).unwrap();
        let calls = Cell::new(0);
        assert!(!l.can_admit(0, 89, &[id]));
        assert_eq!(l.len(), 1);
        assert!(!l
            .attach_with(
                &[id],
                89,
                || {
                    calls.set(calls.get() + 1);
                    Ok::<_, ()>((Arc::new(vec![0; 89]), 89))
                },
                |_, s, a| *s = Some(Arc::clone(a))
            )
            .unwrap());
        assert!(l
            .insert_with(&[2], 1, 100, &[], || {
                calls.set(calls.get() + 1);
                Ok::<_, ()>((None, 100))
            })
            .unwrap()
            .is_none());
        assert_eq!(calls.get(), 0);
        assert!(l.get(id).is_some());
    }

    #[test]
    fn smaller_checkpoint_can_fit_when_the_latest_full_snapshot_cannot() {
        let mut l = shared_lru(2, 100);
        let a = l.insert(vec![1], 1, 8, None).unwrap();
        let b = l.insert(vec![1, 2], 2, 8, None).unwrap();
        assert!(!l.can_admit(0, 85, &[a, b]));
        assert!(!l.can_admit(0, 85, &[b]));
        assert!(l.can_admit(0, 80, &[a]));
        assert!(l
            .attach_with(
                &[a],
                80,
                || { Ok::<_, ()>((Arc::new(vec![0; 80]), 80)) },
                |_, s, a| *s = Some(Arc::clone(a))
            )
            .unwrap());
        assert!(l.get(b).is_none());
        assert_eq!(l.bytes(), 92);
    }

    #[test]
    fn zero_entry_or_byte_cap_prevents_deferred_allocation() {
        for (entries, bytes) in [(0, 100), (2, 0)] {
            let mut l: CheckpointLru<()> = CheckpointLru::new(entries, bytes);
            assert!(l
                .insert_with(&[1], 1, 0, &[], || -> Result<(_, usize), ()> {
                    panic!("disabled checkpoint storage must not copy state")
                })
                .unwrap()
                .is_none());
            assert!(l.is_empty());
            assert_eq!(l.bytes(), 0);
            assert_eq!(l.entries.capacity(), 0);
        }
    }

    #[test]
    fn real_lru_order_survives_byte_pressure_and_protected_candidates() {
        let mut l = CheckpointLru::new(2, 100);
        let a = l.insert(vec![1], 1, 36, "a").unwrap();
        let b = l.insert(vec![2], 1, 36, "b").unwrap();
        l.touch(a);
        let c = l.insert(vec![3], 1, 36, "c").unwrap();
        assert!(l.get(b).is_none());
        assert!(l.grow(c, 30));
        assert!(l.get(a).is_none());
        assert_eq!(l.bytes(), 70);
        assert!(!l.grow(c, 31));
        assert!(l.is_empty());
        let a = l.insert(vec![4], 1, 36, "protected oldest").unwrap();
        let b = l.insert(vec![5], 1, 36, "newer").unwrap();
        assert!(l
            .insert_with(&[6], 1, 36, &[a], || Ok::<_, ()>(("new", 36)))
            .unwrap()
            .is_some());
        assert!(l.get(a).is_some());
        assert!(l.get(b).is_none());
    }

    #[test]
    fn token_capacity_is_charged_and_duplicate_positions_replace() {
        let mut l = CheckpointLru::new(4, 1000);
        let mut tokens = Vec::with_capacity(32);
        tokens.extend_from_slice(&[1, 2, 3]);
        let expected = 10 + tokens.capacity() * 4;
        l.insert(tokens, 2, 10, "first").unwrap();
        assert_eq!(l.bytes(), expected);
        let id = l
            .insert_with(&[1, 2, 3], 2, 20, &[], || Ok::<_, ()>(("second", 20)))
            .unwrap()
            .unwrap();
        assert_eq!(l.len(), 1);
        assert_eq!(l.bytes(), 32);
        assert_eq!(l.get(id).unwrap().state, "second");
    }

    #[test]
    fn invalid_positions_overflows_and_missing_protection_do_not_copy() {
        let mut l: CheckpointLru<()> = CheckpointLru::new(2, usize::MAX);
        for (tokens, pos, bytes, protected) in [
            (&[][..], 0, 0, &[][..]),
            (&[1][..], 2, 0, &[][..]),
            (&[1][..], 1, usize::MAX, &[][..]),
            (&[1][..], 1, 0, &[99][..]),
        ] {
            assert!(l
                .insert_with(
                    tokens,
                    pos,
                    bytes,
                    protected,
                    || -> Result<(_, usize), ()> {
                        panic!("invalid admission must not materialize")
                    }
                )
                .unwrap()
                .is_none());
        }
        assert_eq!(token_bytes(usize::MAX), None);
        assert!(!l.can_admit(usize::MAX, usize::MAX, &[]));
        let id = l.insert(vec![1], 1, 0, ()).unwrap();
        assert!(!l.grow(id, usize::MAX));
        assert!(l.is_empty());
        l.next_id = u64::MAX;
        assert!(l
            .insert_with(&[1], 1, 0, &[], || -> Result<(_, usize), ()> {
                panic!("exhausted IDs must not materialize")
            })
            .unwrap()
            .is_none());
        assert!(l.is_empty());
    }

    #[test]
    fn underestimated_or_inconsistent_metadata_is_not_retained() {
        let mut l = shared_lru(2, 100);
        assert!(l
            .insert_with(&[1], 1, 8, &[], || Ok::<_, ()>((None, 9)))
            .unwrap()
            .is_none());
        let id = l.insert(vec![1], 1, 8, None).unwrap();
        assert!(!l
            .attach_with(
                &[id],
                64,
                || { Ok::<_, ()>((Arc::new(vec![0; 65]), 65)) },
                |_, s, a| *s = Some(Arc::clone(a))
            )
            .unwrap());
        assert!(l.get(id).unwrap().state.is_none());
        assert!(!l
            .attach_with(
                &[id],
                64,
                || { Ok::<_, ()>((Arc::new(vec![0; 65]), 64)) },
                |_, s, a| *s = Some(Arc::clone(a))
            )
            .unwrap());
        assert!(l.is_empty());
        assert_eq!(l.bytes(), 0);
    }

    #[test]
    fn copy_failure_leaves_protected_candidates_usable() {
        let mut l = shared_lru(2, 100);
        let id = l.insert(vec![1], 1, 8, None).unwrap();
        let failed: Result<bool, &str> = l.attach_with::<Arc<Vec<u8>>, _>(
            &[id],
            64,
            || Err("copy failed"),
            |_, s, a| *s = Some(Arc::clone(a)),
        );
        assert_eq!(failed, Err("copy failed"));
        assert_eq!(l.bytes(), 12);
        assert!(l.get(id).unwrap().state.is_none());
        let unrelated = l.insert(vec![2], 1, 8, None).unwrap();
        assert!(l
            .attach_with(
                &[id, unrelated],
                64,
                || { Ok::<_, ()>((Arc::new(vec![0; 64]), 64)) },
                |candidate, s, a| {
                    if candidate == id {
                        *s = Some(Arc::clone(a));
                    }
                }
            )
            .unwrap());
        assert!(l.get(id).unwrap().state.is_some());
        assert!(l.get(unrelated).unwrap().state.is_none());
        assert_eq!(l.bytes(), 88);
    }
}
