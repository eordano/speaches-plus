
use crate::prefix_reuse::saved_state_resume_point;
use std::sync::atomic::{AtomicU64, Ordering::Relaxed};

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Tier {
    InPlace,
    Device,
    Disk,
}

pub trait Held {
    fn tier(&self) -> Tier;
    fn bytes(&self) -> u64;
}

pub struct Entry<T> {
    pub model: u64,
    pub tokens: Vec<u32>,
    pub pos: usize,
    pub used: u64,
    pub value: T,
}

impl<T> Entry<T> {
    pub fn resume_point(&self, prompt: &[u32]) -> usize {
        saved_state_resume_point(&[self.pos], &self.tokens, prompt)
    }

    fn covered_by(&self, newer: &Entry<T>) -> bool {
        self.model == newer.model
            && self.pos <= newer.pos
            && newer.tokens.len() > self.pos
            && self.tokens.len() > self.pos
            && self.tokens[..=self.pos] == newer.tokens[..=self.pos]
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Budget {
    pub device_bytes: u64,
    pub disk_bytes: u64,
}

pub struct SessionLru<T> {
    entries: Vec<Entry<T>>,
    budget: Budget,
    clock: u64,
}

pub struct Evicted<T> {
    pub from_device: Vec<Entry<T>>,
    pub from_disk: Vec<Entry<T>>,
}

impl<T: Held> SessionLru<T> {
    pub fn new(budget: Budget) -> Self {
        Self { entries: Vec::new(), budget, clock: 0 }
    }

    pub fn budget(&self) -> Budget {
        self.budget
    }

    pub fn set_budget(&mut self, budget: Budget) {
        self.budget = budget;
    }

    pub fn tick(&mut self) -> u64 {
        self.clock += 1;
        self.clock
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    pub fn entries(&self) -> &[Entry<T>] {
        &self.entries
    }

    pub fn bytes(&self, tier: Tier) -> u64 {
        self.entries.iter().filter(|e| e.value.tier() == tier).map(|e| e.value.bytes()).sum()
    }

    pub fn count(&self, tier: Tier) -> usize {
        self.entries.iter().filter(|e| e.value.tier() == tier).count()
    }

    pub fn fits_device(&self, bytes: u64) -> bool {
        bytes <= self.budget.device_bytes
    }

    pub fn fits_disk(&self, bytes: u64) -> bool {
        bytes <= self.budget.disk_bytes
    }

    pub fn best(&self, model: u64, prompt: &[u32], tiers: &[Tier]) -> Option<(usize, usize)> {
        self.entries
            .iter()
            .enumerate()
            .filter(|(_, e)| e.model == model && tiers.contains(&e.value.tier()))
            .map(|(i, e)| (i, e.resume_point(prompt)))
            .filter(|&(_, r)| r > 0)
            .max_by_key(|&(i, r)| (r, std::cmp::Reverse(self.entries[i].value.tier()), self.entries[i].used))
    }

    pub fn find_in_place(&self, model: u64) -> Option<usize> {
        self.entries.iter().position(|e| e.model == model && e.value.tier() == Tier::InPlace)
    }

    pub fn get(&self, i: usize) -> &Entry<T> {
        &self.entries[i]
    }

    pub fn touch(&mut self, i: usize) {
        let t = self.tick();
        self.entries[i].used = t;
    }

    pub fn remove(&mut self, i: usize) -> Entry<T> {
        self.entries.remove(i)
    }

    pub fn insert(&mut self, e: Entry<T>) -> Vec<Entry<T>> {
        let mut gone = Vec::new();
        let mut i = 0;
        while i < self.entries.len() {
            if self.entries[i].covered_by(&e) {
                gone.push(self.entries.remove(i));
            } else {
                i += 1;
            }
        }
        self.entries.push(e);
        gone
    }

    fn pinned(&self, i: usize) -> bool {
        let e = &self.entries[i];
        e.value.tier() == Tier::InPlace
            || !self.entries.iter().any(|o| o.model == e.model && o.used > e.used)
    }

    fn evict_lru(&mut self, tier: Tier, limit: u64) -> Vec<Entry<T>> {
        let mut out = Vec::new();
        while self.bytes(tier) > limit {
            let victim = (0..self.entries.len())
                .filter(|&i| self.entries[i].value.tier() == tier && !self.pinned(i))
                .min_by_key(|&i| self.entries[i].used);
            match victim {
                Some(i) => out.push(self.entries.remove(i)),
                None => break,
            }
        }
        out
    }

    pub fn enforce(&mut self) -> Evicted<T> {
        let from_device = self.evict_lru(Tier::Device, self.budget.device_bytes);
        let from_disk = self.evict_lru(Tier::Disk, self.budget.disk_bytes);
        Evicted { from_device, from_disk }
    }
}

#[derive(Default)]
pub struct SessionCacheMetrics {
    pub requests: AtomicU64,
    pub hits_in_place: AtomicU64,
    pub hits_device: AtomicU64,
    pub hits_disk: AtomicU64,
    pub hits_short: AtomicU64,
    pub misses: AtomicU64,
    pub reused_tokens: AtomicU64,
    pub prefilled_tokens: AtomicU64,
    pub parked_device: AtomicU64,
    pub spilled: AtomicU64,
    pub dropped: AtomicU64,
    pub restore_failures: AtomicU64,
    pub resident_device_sessions: AtomicU64,
    pub resident_device_bytes: AtomicU64,
    pub spilled_sessions: AtomicU64,
    pub spilled_bytes: AtomicU64,
}

pub static METRICS: SessionCacheMetrics = SessionCacheMetrics {
    requests: AtomicU64::new(0),
    hits_in_place: AtomicU64::new(0),
    hits_device: AtomicU64::new(0),
    hits_disk: AtomicU64::new(0),
    hits_short: AtomicU64::new(0),
    misses: AtomicU64::new(0),
    reused_tokens: AtomicU64::new(0),
    prefilled_tokens: AtomicU64::new(0),
    parked_device: AtomicU64::new(0),
    spilled: AtomicU64::new(0),
    dropped: AtomicU64::new(0),
    restore_failures: AtomicU64::new(0),
    resident_device_sessions: AtomicU64::new(0),
    resident_device_bytes: AtomicU64::new(0),
    spilled_sessions: AtomicU64::new(0),
    spilled_bytes: AtomicU64::new(0),
};

impl SessionCacheMetrics {
    pub fn record_request(&self, prompt_tokens: usize, reused: usize, source: Option<Tier>, short: bool) {
        self.requests.fetch_add(1, Relaxed);
        self.reused_tokens.fetch_add(reused as u64, Relaxed);
        self.prefilled_tokens.fetch_add(prompt_tokens.saturating_sub(reused) as u64, Relaxed);
        let counter = match (reused, source, short) {
            (0, _, _) => &self.misses,
            (_, _, true) => &self.hits_short,
            (_, Some(Tier::Device), _) => &self.hits_device,
            (_, Some(Tier::Disk), _) => &self.hits_disk,
            _ => &self.hits_in_place,
        };
        counter.fetch_add(1, Relaxed);
    }

    pub fn set_resident<T: Held>(&self, lru: &SessionLru<T>) {
        self.resident_device_sessions
            .store((lru.count(Tier::Device) + lru.count(Tier::InPlace)) as u64, Relaxed);
        self.resident_device_bytes.store(lru.bytes(Tier::Device), Relaxed);
        self.spilled_sessions.store(lru.count(Tier::Disk) as u64, Relaxed);
        self.spilled_bytes.store(lru.bytes(Tier::Disk), Relaxed);
    }

    pub fn render(&self, s: &mut String) {
        use std::fmt::Write as _;
        let g = |a: &AtomicU64| a.load(Relaxed);
        let _ = writeln!(s, "# TYPE speaches_prefix_session_requests_total counter");
        let _ = writeln!(s, "speaches_prefix_session_requests_total {}", g(&self.requests));
        let _ = writeln!(s, "# TYPE speaches_prefix_session_lookups_total counter");
        for (result, v) in [
            ("hit_in_place", &self.hits_in_place),
            ("hit_device", &self.hits_device),
            ("hit_disk", &self.hits_disk),
            ("hit_short", &self.hits_short),
            ("miss", &self.misses),
        ] {
            let _ = writeln!(s, "speaches_prefix_session_lookups_total{{result=\"{result}\"}} {}", g(v));
        }
        let _ = writeln!(s, "# TYPE speaches_prefix_session_tokens_total counter");
        let _ = writeln!(s, "speaches_prefix_session_tokens_total{{kind=\"reused\"}} {}", g(&self.reused_tokens));
        let _ = writeln!(s, "speaches_prefix_session_tokens_total{{kind=\"prefilled\"}} {}", g(&self.prefilled_tokens));
        let _ = writeln!(s, "# TYPE speaches_prefix_session_evictions_total counter");
        for (to, v) in [("device", &self.parked_device), ("disk", &self.spilled), ("dropped", &self.dropped)] {
            let _ = writeln!(s, "speaches_prefix_session_evictions_total{{to=\"{to}\"}} {}", g(v));
        }
        let _ = writeln!(s, "# TYPE speaches_prefix_session_restore_failures_total counter");
        let _ = writeln!(s, "speaches_prefix_session_restore_failures_total {}", g(&self.restore_failures));
        let _ = writeln!(s, "# TYPE speaches_prefix_session_resident gauge");
        let _ = writeln!(s, "speaches_prefix_session_resident{{tier=\"device\"}} {}", g(&self.resident_device_sessions));
        let _ = writeln!(s, "speaches_prefix_session_resident{{tier=\"disk\"}} {}", g(&self.spilled_sessions));
        let _ = writeln!(s, "# TYPE speaches_prefix_session_resident_bytes gauge");
        let _ = writeln!(s, "speaches_prefix_session_resident_bytes{{tier=\"device\"}} {}", g(&self.resident_device_bytes));
        let _ = writeln!(s, "speaches_prefix_session_resident_bytes{{tier=\"disk\"}} {}", g(&self.spilled_bytes));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Debug, Clone, Copy, PartialEq)]
    struct H(Tier, u64);

    impl Held for H {
        fn tier(&self) -> Tier {
            self.0
        }
        fn bytes(&self) -> u64 {
            self.1
        }
    }

    fn convo(seed: u32, n: usize) -> Vec<u32> {
        (0..n as u32).map(|i| seed * 100_000 + i).collect()
    }

    fn entry(lru: &mut SessionLru<H>, model: u64, tokens: Vec<u32>, pos: usize, h: H) -> Entry<H> {
        let used = lru.tick();
        Entry { model, tokens, pos, used, value: h }
    }

    fn budget(device: u64, disk: u64) -> Budget {
        Budget { device_bytes: device, disk_bytes: disk }
    }

    #[test]
    fn a_prompt_resumes_only_from_its_own_exact_token_prefix_and_ties_prefer_the_nearer_tier() {
        let mut lru = SessionLru::new(budget(1 << 40, 0));
        let a = convo(1, 9000);
        let b = convo(2, 9000);
        let e = entry(&mut lru, 7, a.clone(), 8192, H(Tier::Device, 10));
        lru.insert(e);
        let mut a2 = a.clone();
        a2.extend([5, 6, 7]);
        assert_eq!(lru.best(7, &a2, &[Tier::Device]), Some((0, 8192)));
        assert_eq!(lru.best(7, &b, &[Tier::Device]), None, "another conversation must never resume from it");
        assert_eq!(lru.best(8, &a2, &[Tier::Device]), None, "another model must never resume from it");
        let mut diverged = a2.clone();
        diverged[8192] ^= 1;
        assert_eq!(lru.best(7, &diverged, &[Tier::Device]), None, "a divergence at the checkpoint itself refuses");
        let mut early = a2.clone();
        early[100] ^= 1;
        assert_eq!(lru.best(7, &early, &[Tier::Device]), None);
        assert_eq!(lru.best(7, &a[..8192], &[Tier::Device]), None, "a prompt that ends at the checkpoint has no token to feed");

        let mut lru = SessionLru::new(budget(100, 100));
        let t = convo(1, 9000);
        let d = entry(&mut lru, 1, t.clone(), 8192, H(Tier::Disk, 10));
        lru.insert(d);
        let other = convo(1, 8500);
        let v = entry(&mut lru, 1, other, 8192, H(Tier::Device, 10));
        lru.entries.push(v);
        let mut p = t.clone();
        p.push(1);
        let (i, r) = lru.best(1, &p, &[Tier::Device, Tier::Disk]).unwrap();
        assert_eq!((lru.get(i).value.tier(), r), (Tier::Device, 8192), "ties prefer the working state, then device, then disk");
    }

    #[test]
    fn the_budget_keeps_two_sessions_resident_evicts_least_recently_used_but_never_a_models_latest_and_in_place_costs_nothing() {
        let mut lru = SessionLru::new(budget(25, 0));
        let mut a = convo(1, 9000);
        let mut b = convo(2, 9000);
        for turn in 0..6 {
            for (m, s) in [(1u32, &mut a), (2, &mut b)] {
                if turn > 0 {
                    assert!(lru.best(0, s, &[Tier::Device]).is_some(), "session {m} turn {turn} must hit");
                }
                s.extend((0..700).map(|i| m * 1000 + turn * 10 + i));
                let pos = (s.len() - 1) / 2048 * 2048;
                let e = entry(&mut lru, 0, s.clone(), pos, H(Tier::Device, 10));
                let gone = lru.insert(e);
                if turn > 0 {
                    assert_eq!(gone.len(), 1, "the new turn supersedes the same session's previous state");
                }
                let ev = lru.enforce();
                assert!(ev.from_device.is_empty());
            }
        }
        assert_eq!(lru.len(), 2);

        let mut lru = SessionLru::new(budget(30, 0));
        let ids: Vec<(u64, u32)> = vec![(1, 1), (2, 2), (1, 3), (1, 4), (2, 5)];
        for &(m, s) in &ids {
            let e = entry(&mut lru, m, convo(s, 5000), 4096, H(Tier::Device, 10));
            lru.insert(e);
        }
        let ev = lru.enforce();
        let seeds: Vec<u32> = ev.from_device.iter().map(|e| e.tokens[0] / 100_000).collect();
        assert_eq!(seeds, vec![1, 2], "the two least recently used go first");
        assert_eq!(lru.bytes(Tier::Device), 30);

        let mut lru = SessionLru::new(budget(0, 0));
        for &(m, s) in &ids {
            let e = entry(&mut lru, m, convo(s, 5000), 4096, H(Tier::Device, 10));
            lru.insert(e);
        }
        let ev = lru.enforce();
        let seeds: Vec<u32> = ev.from_device.iter().map(|e| e.tokens[0] / 100_000).collect();
        assert_eq!(seeds, vec![1, 2, 3], "over any budget, each model's latest (4 and 5) stays");
        let kept: Vec<u32> = lru.entries().iter().map(|e| e.tokens[0] / 100_000).collect();
        assert_eq!(kept, vec![4, 5]);

        let mut lru = SessionLru::new(budget(10, 0));
        let a = entry(&mut lru, 1, convo(1, 5000), 4096, H(Tier::Device, 10));
        let b = entry(&mut lru, 1, convo(2, 5000), 4096, H(Tier::Device, 10));
        lru.insert(a);
        lru.insert(b);
        lru.touch(0);
        let ev = lru.enforce();
        assert_eq!(ev.from_device[0].tokens[0] / 100_000, 2, "a touch makes the older entry the recent one");

        let mut lru = SessionLru::new(budget(25, 100));
        let live = entry(&mut lru, 1, convo(9, 9000), 8192, H(Tier::InPlace, 0));
        lru.insert(live);
        for s in 0..5 {
            let e = entry(&mut lru, 1, convo(s, 9000), 8192, H(Tier::Device, 10));
            lru.insert(e);
            let ev = lru.enforce();
            for v in ev.from_device {
                let spilled = Entry { value: H(Tier::Disk, v.value.1), ..v };
                lru.insert(spilled);
            }
            assert!(lru.bytes(Tier::Device) <= 25);
        }
        lru.touch(lru.find_in_place(1).unwrap());
        let ev = lru.enforce();
        assert!(ev.from_device.is_empty() && ev.from_disk.is_empty());
        assert_eq!((lru.count(Tier::InPlace), lru.count(Tier::Device), lru.count(Tier::Disk)), (1, 2, 3));
        assert!(!lru.fits_device(26));

        let mut tight = SessionLru::new(budget(0, 15));
        for s in 0..3 {
            let e = entry(&mut tight, 1, convo(s, 9000), 8192, H(Tier::Disk, 10));
            tight.insert(e);
        }
        let ev = tight.enforce();
        assert_eq!(ev.from_disk.len(), 2, "the disk budget evicts too, keeping the latest");
    }

    #[test]
    fn metrics_count_hits_misses_and_tokens() {
        let m = SessionCacheMetrics::default();
        m.record_request(10000, 8192, Some(Tier::Device), false);
        m.record_request(10000, 0, None, false);
        m.record_request(10000, 8192, Some(Tier::InPlace), false);
        let mut s = String::new();
        m.render(&mut s);
        assert!(s.contains("speaches_prefix_session_lookups_total{result=\"hit_device\"} 1"));
        assert!(s.contains("speaches_prefix_session_lookups_total{result=\"miss\"} 1"));
        assert!(s.contains("speaches_prefix_session_tokens_total{kind=\"reused\"} 16384"));
        assert!(s.contains("speaches_prefix_session_tokens_total{kind=\"prefilled\"} 13616"));
    }
}
