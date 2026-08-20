#[path = "../src/oapi/chat_engine_wgpu/spec.rs"]
#[allow(dead_code)]
mod spec;

use spec::{run_spec_round, ChainVerifyTarget, SpecKnobs, SpecLoop, SpecStats};

struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Self {
        Self(seed | 1)
    }

    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }

    fn below(&mut self, n: u64) -> u64 {
        self.next() % n.max(1)
    }
}

fn mix(mut h: u64, t: u32) -> u64 {
    h = h.wrapping_add(0x9e3779b97f4a7c15).wrapping_add(t as u64);
    h = (h ^ (h >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
    h = (h ^ (h >> 27)).wrapping_mul(0x94d049bb133111eb);
    h ^ (h >> 31)
}

fn hash_model(seed: u64, vocab: u32) -> impl Fn(&[u32]) -> u32 + Copy {
    move |ctx: &[u32]| {
        let mut h = seed;
        for &t in ctx {
            h = mix(h, t);
        }
        (h % vocab as u64) as u32
    }
}

struct FnTarget<F: Fn(&[u32]) -> u32> {
    ctx: Vec<u32>,
    scratch: Vec<u32>,
    f: F,
    cap: usize,
}

impl<F: Fn(&[u32]) -> u32> FnTarget<F> {
    fn new(f: F, cap: usize) -> Self {
        Self {
            ctx: Vec::new(),
            scratch: Vec::new(),
            f,
            cap,
        }
    }

    fn committed(&self) -> usize {
        self.ctx.len()
    }

    fn decode1(&mut self, t: u32) -> u32 {
        let amax = self.verify_chain(&[t]).unwrap();
        self.advance(1).unwrap();
        amax[0]
    }

    fn prefill(&mut self, prompt: &[u32]) -> u32 {
        let mut last = 0;
        for &t in prompt {
            last = self.decode1(t);
        }
        last
    }
}

impl<F: Fn(&[u32]) -> u32> ChainVerifyTarget for FnTarget<F> {
    fn verify_chain(&mut self, batch: &[u32]) -> anyhow::Result<Vec<u32>> {
        anyhow::ensure!(!batch.is_empty(), "empty verify batch");
        anyhow::ensure!(
            batch.len() <= self.cap,
            "verify batch {} exceeds capacity {}",
            batch.len(),
            self.cap
        );
        self.scratch = batch.to_vec();
        let mut probe = self.ctx.clone();
        let mut amax = Vec::with_capacity(batch.len());
        for &t in batch {
            probe.push(t);
            amax.push((self.f)(&probe));
        }
        Ok(amax)
    }

    fn advance(&mut self, n: usize) -> anyhow::Result<()> {
        anyhow::ensure!(
            n >= 1 && n <= self.scratch.len(),
            "advance {n} outside 1..={} scratch rows",
            self.scratch.len()
        );
        self.ctx.extend_from_slice(&self.scratch[..n]);
        self.scratch.clear();
        Ok(())
    }

    fn capacity(&self) -> usize {
        self.cap
    }
}

fn greedy_stream<F: Fn(&[u32]) -> u32>(f: &F, prompt: &[u32], n: usize) -> Vec<u32> {
    let mut seq = prompt.to_vec();
    let mut out = Vec::with_capacity(n);
    for _ in 0..n {
        let t = f(&seq);
        out.push(t);
        seq.push(t);
    }
    out
}

fn lcp(a: &[u32], b: &[u32]) -> usize {
    a.iter().zip(b).take_while(|(x, y)| x == y).count()
}

fn assert_lossless_rounds<F, D>(
    f: F,
    prompt: &[u32],
    cap: usize,
    n_rounds: usize,
    mut draft_for: D,
) -> Vec<u32>
where
    F: Fn(&[u32]) -> u32 + Copy,
    D: FnMut(&[u32], usize) -> Vec<u32>,
{
    let cont = greedy_stream(&f, prompt, n_rounds * cap + cap + 2);
    let mut tgt = FnTarget::new(f, cap);
    let bonus0 = tgt.prefill(prompt);
    assert_eq!(bonus0, cont[0], "prefill bonus diverges from target stream");
    let mut emitted = vec![bonus0];
    let mut bonus = bonus0;
    for round in 0..n_rounds {
        let limit = cap - 1;
        let tail = &cont[emitted.len()..];
        let draft = draft_for(tail, limit);
        assert!(draft.len() <= limit, "drafter exceeded limit");
        let (acc, out) = run_spec_round(&mut tgt, bonus, &draft).unwrap();
        assert_eq!(
            acc.commit_len - 1,
            lcp(&draft, tail),
            "round {round}: commit_len-1 != LCP(draft, target continuation)"
        );
        assert!(acc.commit_len >= 1 && acc.commit_len <= draft.len() + 1);
        assert_eq!(acc.draft_accepted, acc.commit_len - 1);
        assert_eq!(out.len(), acc.commit_len);
        assert_eq!(
            out,
            &cont[emitted.len()..emitted.len() + acc.commit_len],
            "round {round}: emitted tokens diverge from target stream"
        );
        emitted.extend_from_slice(&out);
        bonus = *out.last().unwrap();
        assert_eq!(bonus, acc.next_bonus);
        assert_eq!(
            tgt.committed(),
            prompt.len() + emitted.len() - 1,
            "round {round}: committed pos desynced from emitted count"
        );
        assert_eq!(emitted, &cont[..emitted.len()]);
    }
    emitted
}

#[test]
fn knobs_defaults_and_gate() {
    let k = SpecKnobs::parse(None, None, None);
    assert!(k.enabled);
    assert_eq!(k.k, 8);
    assert_eq!(k.min_match, 3);
    assert!(SpecKnobs::parse(Some("1"), None, None).enabled);
    assert!(!SpecKnobs::parse(Some("0"), None, None).enabled);
    assert!(!SpecKnobs::parse(Some(" 0 "), None, None).enabled);
    assert!(SpecKnobs::parse(Some("true"), None, None).enabled);
    assert!(SpecKnobs::parse(Some(" 1 "), None, None).enabled);
    assert_eq!(SpecKnobs::default(), SpecKnobs::parse(None, None, None));
}

#[test]
fn knobs_clamp_and_fallback() {
    assert_eq!(SpecKnobs::parse(None, Some("0"), None).k, 1);
    assert_eq!(SpecKnobs::parse(None, Some("1"), None).k, 1);
    assert_eq!(SpecKnobs::parse(None, Some("5"), None).k, 5);
    assert_eq!(SpecKnobs::parse(None, Some("99"), None).k, 8);
    assert_eq!(SpecKnobs::parse(None, Some("junk"), None).k, 8);
    assert_eq!(SpecKnobs::parse(None, None, Some("0")).min_match, 1);
    assert_eq!(SpecKnobs::parse(None, None, Some("7")).min_match, 7);
    assert_eq!(SpecKnobs::parse(None, None, Some("x")).min_match, 3);
    let e = SpecKnobs::from_env();
    assert!(e.k >= 1 && e.k <= 8 && e.min_match >= 1);
}

#[test]
fn stats_math() {
    let s = SpecStats::default();
    assert_eq!(s.tau(), 0.0);
    assert_eq!(s.accept_rate(), 0.0);
    let s = SpecStats {
        rounds: 4,
        rounds_with_draft: 3,
        drafted: 12,
        accepted: 6,
        emitted: 10,
    };
    assert!((s.tau() - 2.5).abs() < 1e-12);
    assert!((s.accept_rate() - 0.5).abs() < 1e-12);
    assert!(s.summary().contains("tau=2.500"));
}

#[test]
fn empty_draft_round_is_plain_decode() {
    let f = hash_model(11, 97);
    let mut a = FnTarget::new(f, 9);
    let mut b = FnTarget::new(f, 9);
    let prompt = [5, 9, 2, 4];
    let bonus_a = a.prefill(&prompt);
    let bonus_b = b.prefill(&prompt);
    assert_eq!(bonus_a, bonus_b);
    let (acc, out) = run_spec_round(&mut a, bonus_a, &[]).unwrap();
    assert_eq!(acc.commit_len, 1);
    assert_eq!(acc.draft_accepted, 0);
    let plain = b.decode1(bonus_b);
    assert_eq!(out, vec![plain]);
    assert_eq!(a.committed(), b.committed());
}

#[test]
fn oversized_draft_is_rejected() {
    let f = hash_model(3, 50);
    let mut tgt = FnTarget::new(f, 4);
    tgt.prefill(&[1, 2, 3]);
    assert!(run_spec_round(&mut tgt, 0, &[1, 2, 3, 4]).is_err());
    assert!(run_spec_round(&mut tgt, 0, &[1, 2, 3]).is_ok());
}

#[test]
fn zero_capacity_is_rejected() {
    let f = hash_model(3, 50);
    let mut tgt = FnTarget::new(f, 0);
    assert!(run_spec_round(&mut tgt, 0, &[]).is_err());
}

#[test]
fn lossless_perfect_drafter() {
    let f = hash_model(0xabcdef, 211);
    let prompt = [7, 1, 3, 3, 9];
    let emitted = assert_lossless_rounds(f, &prompt, 9, 40, |tail, limit| {
        tail[..limit.min(tail.len())].to_vec()
    });
    let cont = greedy_stream(&f, &prompt, emitted.len());
    assert_eq!(emitted, cont);
    assert_eq!(emitted.len(), 1 + 40 * 9);
}

#[test]
fn lossless_constant_wrong_drafter() {
    let vocab = 61;
    let f = hash_model(0x5150, vocab);
    let prompt = [2, 8, 2, 8];
    let emitted = assert_lossless_rounds(f, &prompt, 9, 60, |_, limit| vec![vocab; limit]);
    assert_eq!(emitted.len(), 1 + 60);
}

#[test]
fn lossless_random_drafter() {
    let f = hash_model(0xfeed, 5);
    let prompt = [0, 1, 2];
    let mut rng = Rng::new(42);
    assert_lossless_rounds(f, &prompt, 9, 120, |_, limit| {
        let len = rng.below(limit as u64 + 1) as usize;
        (0..len).map(|_| rng.below(5) as u32).collect()
    });
}

#[test]
fn lossless_adversarial_drafter() {
    let vocab = 7u32;
    let f = hash_model(0xdead1077, vocab);
    let prompt = [1, 1, 2, 3, 5];
    let mut rng = Rng::new(1234567);
    assert_lossless_rounds(f, &prompt, 9, 150, |tail, limit| {
        let good = rng.below(limit as u64 + 1) as usize;
        let mut d: Vec<u32> = tail[..good.min(tail.len())].to_vec();
        if d.len() < limit && rng.below(4) != 0 {
            d.push(vocab);
            while d.len() < limit && rng.below(2) == 0 {
                d.push(rng.below(vocab as u64 + 1) as u32);
            }
        }
        d
    });
}

#[test]
fn property_random_targets_and_drafters() {
    let mut rng = Rng::new(0x243f6a8885a308d3);
    for iter in 0..200 {
        let vocab = 2 + rng.below(8) as u32;
        let seed = rng.next();
        let f = hash_model(seed, vocab);
        let cap = 1 + rng.below(9) as usize;
        let plen = 1 + rng.below(12) as usize;
        let prompt: Vec<u32> = (0..plen).map(|_| rng.below(vocab as u64) as u32).collect();
        let n_rounds = 5 + rng.below(25) as usize;
        let mut kind_rng = Rng::new(rng.next());
        let emitted =
            assert_lossless_rounds(f, &prompt, cap, n_rounds, |tail, limit| {
                match kind_rng.below(4) {
                    0 => tail[..limit.min(tail.len())].to_vec(),
                    1 => vec![vocab; kind_rng.below(limit as u64 + 1) as usize],
                    2 => {
                        let len = kind_rng.below(limit as u64 + 1) as usize;
                        (0..len)
                            .map(|_| kind_rng.below(vocab as u64) as u32)
                            .collect()
                    }
                    _ => {
                        let good = kind_rng.below(limit as u64 + 1) as usize;
                        let mut d: Vec<u32> = tail[..good.min(tail.len())].to_vec();
                        if d.len() < limit {
                            d.push(vocab + kind_rng.below(3) as u32);
                        }
                        d
                    }
                }
            });
        let cont = greedy_stream(&f, &prompt, emitted.len());
        assert_eq!(emitted, cont, "iter {iter} diverged");
    }
}

fn run_spec_loop<F: Fn(&[u32]) -> u32 + Copy>(
    f: F,
    prompt: &[u32],
    cap: usize,
    knobs: SpecKnobs,
    n_tokens: usize,
) -> (Vec<u32>, SpecStats) {
    let mut tgt = FnTarget::new(f, cap);
    let bonus0 = tgt.prefill(prompt);
    let mut sl = SpecLoop::new(knobs);
    sl.prime(prompt);
    sl.prime(&[bonus0]);
    let mut emitted = vec![bonus0];
    let mut bonus = bonus0;
    while emitted.len() < n_tokens {
        let out = sl.round(&mut tgt, bonus).unwrap();
        assert!(!out.is_empty());
        emitted.extend_from_slice(&out);
        bonus = *out.last().unwrap();
        assert_eq!(tgt.committed(), prompt.len() + emitted.len() - 1);
        assert_eq!(sl.context_len(), tgt.committed() + 1);
    }
    (emitted, sl.stats())
}

#[test]
fn spec_loop_with_real_suffix_drafter_is_lossless_on_hash_model() {
    let f = hash_model(0x777, 4);
    let prompt = [0, 1, 2, 3, 0, 1, 2, 3];
    let (emitted, stats) = run_spec_loop(f, &prompt, 9, SpecKnobs::default(), 300);
    let cont = greedy_stream(&f, &prompt, emitted.len());
    assert_eq!(emitted, cont);
    assert_eq!(stats.emitted, emitted.len() - 1);
    assert!(stats.accepted <= stats.drafted);
    assert!(stats.rounds_with_draft <= stats.rounds);
}

#[test]
fn spec_loop_accelerates_periodic_target_and_stays_lossless() {
    let pattern = [10u32, 11, 12, 13, 14, 15, 16];
    let f = move |ctx: &[u32]| pattern[ctx.len() % pattern.len()];
    let prompt = [9u32, 9, 9];
    let (emitted, stats) = run_spec_loop(f, &prompt, 9, SpecKnobs::default(), 200);
    let cont = greedy_stream(&f, &prompt, emitted.len());
    assert_eq!(emitted, cont);
    assert!(stats.rounds_with_draft > 0, "suffix drafter never engaged");
    assert!(stats.accepted > 0, "no draft token was ever accepted");
    assert!(
        stats.tau() > 1.5,
        "expected tau > 1.5 on periodic stream, got {}",
        stats.tau()
    );
}

#[test]
fn spec_loop_degenerates_on_non_repeating_stream() {
    let f = |ctx: &[u32]| ctx.len() as u32 + 1000;
    let prompt = [1u32, 2, 3, 4];
    let (emitted, stats) = run_spec_loop(f, &prompt, 9, SpecKnobs::default(), 80);
    let cont = greedy_stream(&f, &prompt, emitted.len());
    assert_eq!(emitted, cont);
    assert_eq!(stats.rounds_with_draft, 0);
    assert_eq!(stats.drafted, 0);
    assert_eq!(stats.emitted, stats.rounds);
    assert!((stats.tau() - 1.0).abs() < 1e-12);
}

#[test]
fn spec_loop_respects_k_and_capacity_clamps() {
    let pattern = [1u32, 2, 3, 4, 5];
    let f = move |ctx: &[u32]| pattern[ctx.len() % pattern.len()];
    let prompt = [8u32, 8];
    let knobs = SpecKnobs::parse(Some("1"), Some("2"), Some("1"));
    assert_eq!(knobs.k, 2);
    let mut tgt = FnTarget::new(f, 9);
    let mut bonus = tgt.prefill(&prompt);
    let mut sl = SpecLoop::new(knobs);
    sl.prime(&prompt);
    sl.prime(&[bonus]);
    let mut emitted = vec![bonus];
    for _ in 0..30 {
        let d = sl.propose_draft(tgt.capacity());
        assert!(d.len() <= 2, "draft {} exceeds k=2 clamp", d.len());
        let out = sl.round(&mut tgt, bonus).unwrap();
        assert!(out.len() <= 3);
        emitted.extend_from_slice(&out);
        bonus = *out.last().unwrap();
    }
    let cont = greedy_stream(&f, &prompt, emitted.len());
    assert_eq!(emitted, cont);
    assert!(sl.stats().drafted > 0);
    assert!(sl.ema_value() >= 0.0);
    let d3 = sl.propose_draft(3);
    assert!(d3.len() <= 2);
    let d2 = sl.propose_draft(2);
    assert!(d2.len() <= 1);
    assert!(sl.propose_draft(1).is_empty());
    assert!(sl.propose_draft(0).is_empty());
}

#[test]
fn spec_loop_small_capacity_forces_plain_rounds() {
    let pattern = [1u32, 2, 3, 1, 2, 3];
    let f = move |ctx: &[u32]| pattern[ctx.len() % pattern.len()];
    let prompt = [7u32];
    let (emitted, stats) = run_spec_loop(f, &prompt, 1, SpecKnobs::default(), 40);
    let cont = greedy_stream(&f, &prompt, emitted.len());
    assert_eq!(emitted, cont);
    assert_eq!(stats.drafted, 0);
}
