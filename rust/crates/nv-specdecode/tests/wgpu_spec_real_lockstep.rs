#![cfg(feature = "wgpu")]

use anyhow::Result;
use nv_specdecode::wgpu_spec::{
    ChainDrafter, LockstepChainSpec, ModelDrafter, PromptLookupDrafter, StepDecoder,
};

const VOCAB: usize = 64;

#[derive(Clone)]
struct TableDecoder {
    name: String,
    table: Vec<u32>,
    pos: usize,
    steps: usize,
}

impl TableDecoder {
    fn new(name: &str, table: Vec<u32>) -> Self {
        assert_eq!(table.len(), VOCAB);
        Self {
            name: name.to_string(),
            table,
            pos: 0,
            steps: 0,
        }
    }

    fn cyclic(name: &str, period: u32) -> Self {
        let table = (0..VOCAB as u32)
            .map(|t| {
                if t < period {
                    (t + 1) % period
                } else {
                    t % period
                }
            })
            .collect();
        Self::new(name, table)
    }

    fn perturbed(base: &TableDecoder, name: &str, every: usize) -> Self {
        let mut table = base.table.clone();
        for (i, v) in table.iter_mut().enumerate() {
            if i % every == 0 {
                *v = (*v + 7) % VOCAB as u32;
            }
        }
        Self::new(name, table)
    }
}

impl StepDecoder for TableDecoder {
    fn label(&self) -> String {
        self.name.clone()
    }

    fn vocab(&self) -> usize {
        VOCAB
    }

    fn pos(&self) -> usize {
        self.pos
    }

    fn reset_state(&mut self) -> Result<()> {
        self.pos = 0;
        Ok(())
    }

    fn step(&mut self, token: u32) -> Result<u32> {
        assert!((token as usize) < VOCAB, "token {token} out of vocab");
        self.pos += 1;
        self.steps += 1;
        Ok(self.table[token as usize])
    }
}

struct SilentDrafter;

impl ChainDrafter for SilentDrafter {
    fn label(&self) -> String {
        "silent".into()
    }

    fn reset_state(&mut self) -> Result<()> {
        Ok(())
    }

    fn observe(&mut self, _token: u32) -> Result<Option<u32>> {
        Ok(None)
    }
}

fn prompt() -> Vec<u32> {
    vec![9, 3, 11, 40, 2]
}

#[test]
fn lockstep_identical_drafter_fills_every_batch() {
    let verifier = TableDecoder::cyclic("v", 23);
    let drafter = ModelDrafter::new(TableDecoder::cyclic("d", 23));
    let mut spec = LockstepChainSpec::new(verifier, drafter, 4, vec![]).unwrap();
    let stats = spec.generate(&prompt(), 64).unwrap();
    println!("identical drafter: {}", stats.summary());
    assert_eq!(stats.emitted.len(), 64);
    assert_eq!(stats.accepted_drafts, stats.draft_slots);
    assert_eq!(stats.acceptance_rate(), 1.0);
    assert!((stats.tokens_per_round() - 4.0).abs() < 1e-9);
    let greedy = spec.greedy(&prompt(), 64).unwrap();
    assert_eq!(stats.emitted, greedy);
}

#[test]
fn lockstep_silent_drafter_never_accepts_but_stream_is_greedy() {
    let verifier = TableDecoder::cyclic("v", 23);
    let mut spec = LockstepChainSpec::new(verifier, SilentDrafter, 4, vec![]).unwrap();
    let stats = spec.generate(&prompt(), 48).unwrap();
    println!("silent drafter: {}", stats.summary());
    assert_eq!(stats.accepted_drafts, 0);
    assert_eq!(stats.acceptance_rate(), 0.0);
    assert!((stats.tokens_per_round() - 1.0).abs() < 1e-9);
    let greedy = spec.greedy(&prompt(), 48).unwrap();
    assert_eq!(stats.emitted, greedy);
}

#[test]
fn lockstep_perturbed_drafter_is_between_zero_and_one_and_stream_is_greedy() {
    let base = TableDecoder::cyclic("v", 23);
    let drafter = ModelDrafter::new(TableDecoder::perturbed(&base, "d", 3));
    let mut spec = LockstepChainSpec::new(base, drafter, 4, vec![]).unwrap();
    let stats = spec.generate(&prompt(), 96).unwrap();
    println!("perturbed drafter: {}", stats.summary());
    let r = stats.acceptance_rate();
    assert!(
        r > 0.0 && r < 1.0,
        "acceptance rate {r} must be strictly interior"
    );
    assert!(stats.tokens_per_round() > 1.0 && stats.tokens_per_round() < 4.0);
    let greedy = spec.greedy(&prompt(), 96).unwrap();
    assert_eq!(stats.emitted, greedy);
}

#[test]
fn lockstep_prompt_lookup_learns_a_repeating_verifier() {
    let verifier = TableDecoder::cyclic("v", 5);
    let mut spec =
        LockstepChainSpec::new(verifier, PromptLookupDrafter::new(2), 4, vec![]).unwrap();
    let stats = spec.generate(&prompt(), 120).unwrap();
    println!("prompt lookup on a period-5 verifier: {}", stats.summary());
    assert!(
        stats.acceptance_rate() > 0.8,
        "prompt lookup must lock onto a periodic stream, got {}",
        stats.acceptance_rate()
    );
    let greedy = spec.greedy(&prompt(), 120).unwrap();
    assert_eq!(stats.emitted, greedy);
}

#[test]
fn lockstep_stops_at_eos() {
    let verifier = TableDecoder::cyclic("v", 23);
    let drafter = ModelDrafter::new(TableDecoder::cyclic("d", 23));
    let eos_probe = {
        let mut v = TableDecoder::cyclic("probe", 23);
        let mut out = Vec::new();
        let mut next = 0;
        for t in prompt() {
            next = v.step(t).unwrap();
        }
        for _ in 0..12 {
            out.push(next);
            next = v.step(next).unwrap();
        }
        out
    };
    let eos = vec![eos_probe[7]];
    let mut spec = LockstepChainSpec::new(verifier, drafter, 4, eos.clone()).unwrap();
    let stats = spec.generate(&prompt(), 64).unwrap();
    println!("eos stop: {}", stats.summary());
    assert!(stats.hit_eos, "must report EOS");
    assert_eq!(stats.emitted.len(), 8);
    assert_eq!(*stats.emitted.last().unwrap(), eos[0]);
    assert!(
        !stats.emitted[..stats.emitted.len() - 1].contains(&eos[0]),
        "EOS must appear only once, at the end"
    );
}

#[test]
fn lockstep_rejects_k_below_two() {
    let verifier = TableDecoder::cyclic("v", 23);
    assert!(LockstepChainSpec::new(verifier, SilentDrafter, 1, vec![]).is_err());
}
