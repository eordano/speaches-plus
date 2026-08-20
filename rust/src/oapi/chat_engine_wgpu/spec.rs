use std::sync::atomic::{AtomicU64, Ordering};

use anyhow::{ensure, Result};
use nv_specdecode::chain::{accept_prefix_argmax, build_chain_batch, ChainAccept};
use nv_specdecode::suffix_automaton::{AcceptEma, SuffixAutomaton};

pub const SPEC_ENV: &str = "NV_WGPU_SPEC";
pub const SPEC_KINDS_ENV: &str = "NV_WGPU_SPEC_KINDS";
pub const SPEC_K_ENV: &str = "NV_WGPU_SPEC_K";
pub const SPEC_MIN_MATCH_ENV: &str = "NV_WGPU_SPEC_MIN_MATCH";
pub const SPEC_DRAFTER_ENV: &str = "NV_WGPU_SPEC_DRAFTER";
pub const SPEC_ASSISTANT_DIR_ENV: &str = "NV_WGPU_SPEC_ASSISTANT_DIR";
pub const SPEC_ASSISTANT_EMA_FLOOR_ENV: &str = "NV_WGPU_SPEC_ASSISTANT_EMA_FLOOR";
pub const SPEC_ASSISTANT_EMA_FLOOR_DEFAULT: f64 = 2.05;
pub const SPEC_PROBE_MIN_MATCH_ENV: &str = "NV_WGPU_SPEC_PROBE_MIN_MATCH";
pub const SPEC_PROBE_MIN_MATCH_DEFAULT: usize = 16;
pub const SPEC_ASSISTANT_PROBE_ROUNDS_ENV: &str = "NV_WGPU_SPEC_ASSISTANT_PROBE_ROUNDS";
pub const SPEC_ASSISTANT_PROBE_ROUNDS_DEFAULT: u64 = 16;
pub const SPEC_ASSISTANT_PROBE_MIN_ROUNDS: u64 = 4;
pub const SPEC_ASSISTANT_GATE_MARGIN: f64 = 1.05;
pub const SPEC_ASSISTANT_GATE_HOPELESS: f64 = 0.6;
pub const SPEC_K_MAX: usize = 8;
pub const SPEC_K_DEFAULT: usize = 8;
pub const SPEC_MIN_MATCH_DEFAULT: usize = 3;

pub const SPEC_KINDS_DEFAULT_IS_GEMMA4_E4B_ALONE_BECAUSE_A_VERIFY_FORWARD_IS_NOT_A_MEASURED_WIN: &str =
    "admitting a kind to the chain route the moment its decoder grows verify_chain would flip a \
     serving default onto a path measured SLOWER: on unsloth/Qwen3.8-27B-NVFP4 at max_seq 2048, \
     192 greedy tokens over three prompts ran 7.93/7.22/7.71 s with the route off and \
     8.66/8.37/12.94 s with it on at tau 2.73/1.05/1.38, byte-identical output either way, \
     because qwen3.5/3.8 pay a state rollback plus a full M=1 replay of the accepted prefix on \
     every partial accept -- DeltaNet state is recurrent, not position-masked. gpt-oss has no \
     real-weights wgpu decode on record at all. NV_WGPU_SPEC_KINDS names the extra kinds by slug; \
     gemma4-e4b is the one kind whose win is measured. That loss was measured with the model-free \
     suffix drafter; NV_Q3D_MTP=1 separately admits qwen3.5/3.8 with the checkpoint's own MTP head \
     drafting, which changes the acceptance economics and is gated per-session by the \
     AssistantGate.";

pub const SPEC_KINDS_ABSENT_FROM_THE_SLUG_TABLE_HAVE_NO_CHAIN_TARGET_ARM_ON_THIS_SEAM: &str =
    "gemma4-dense, gemma4-moe, qwen3.5-moe and laguna carry decoder-level verify_chain entries but \
     no ChainVerifyTarget arm in chat_engine_wgpu; giving them a slug would admit a route that \
     bails at the first multi-row round instead of refusing at admission.";

pub const SPEC_CHAIN_ROUTE_SLUGS: [&[&str]; 3] = [
    &["gemma4-e4b"],
    &["qwen3.5-dense", "qwen3.8"],
    &["gpt-oss"],
];

pub const SPEC_CHAIN_ROUTE_ROW_OF_THE_ONLY_KIND_ADMITTED_WITHOUT_AN_OPT_IN: usize = 0;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SpecKinds(u8);

impl SpecKinds {
    pub fn parse(list: Option<&str>) -> Self {
        let mut bits = 1u8 << SPEC_CHAIN_ROUTE_ROW_OF_THE_ONLY_KIND_ADMITTED_WITHOUT_AN_OPT_IN;
        if let Some(list) = list {
            for tok in list.split(',').map(str::trim).filter(|t| !t.is_empty()) {
                if let Some(row) = slug_row(tok) {
                    bits |= 1u8 << row;
                }
            }
        }
        Self(bits)
    }

    pub fn from_env() -> Self {
        Self::parse(std::env::var(SPEC_KINDS_ENV).ok().as_deref())
    }

    pub fn admits(self, slug: &str) -> bool {
        slug_row(slug).is_some_and(|row| self.0 & (1u8 << row) != 0)
    }
}

fn slug_row(slug: &str) -> Option<usize> {
    SPEC_CHAIN_ROUTE_SLUGS
        .iter()
        .position(|slugs| slugs.iter().any(|s| slug.eq_ignore_ascii_case(s)))
}

impl Default for SpecKinds {
    fn default() -> Self {
        Self::parse(None)
    }
}

pub fn chain_capacity(verify_rows: usize, pos: usize, chunk_span: usize, max_seq: usize) -> usize {
    if verify_rows < 2 {
        return 1;
    }
    if pos + verify_rows.max(chunk_span) > max_seq {
        return 1;
    }
    verify_rows
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SpecKnobs {
    pub enabled: bool,
    pub k: usize,
    pub min_match: usize,
    pub kinds: SpecKinds,
}

impl SpecKnobs {
    pub fn parse(spec: Option<&str>, k: Option<&str>, min_match: Option<&str>) -> Self {
        Self::parse_with_kinds(spec, k, min_match, None)
    }

    pub fn parse_with_kinds(
        spec: Option<&str>,
        k: Option<&str>,
        min_match: Option<&str>,
        kinds: Option<&str>,
    ) -> Self {
        let enabled = spec.map(str::trim) != Some("0");
        let k = k
            .and_then(|v| v.trim().parse::<usize>().ok())
            .unwrap_or(SPEC_K_DEFAULT)
            .clamp(1, SPEC_K_MAX);
        let min_match = min_match
            .and_then(|v| v.trim().parse::<usize>().ok())
            .unwrap_or(SPEC_MIN_MATCH_DEFAULT)
            .max(1);
        Self {
            enabled,
            k,
            min_match,
            kinds: SpecKinds::parse(kinds),
        }
    }

    pub fn from_env() -> Self {
        Self::parse_with_kinds(
            std::env::var(SPEC_ENV).ok().as_deref(),
            std::env::var(SPEC_K_ENV).ok().as_deref(),
            std::env::var(SPEC_MIN_MATCH_ENV).ok().as_deref(),
            std::env::var(SPEC_KINDS_ENV).ok().as_deref(),
        )
    }
}

impl Default for SpecKnobs {
    fn default() -> Self {
        Self::parse(None, None, None)
    }
}

pub fn assistant_probe_rounds() -> u64 {
    std::env::var(SPEC_ASSISTANT_PROBE_ROUNDS_ENV)
        .ok()
        .and_then(|v| v.trim().parse::<u64>().ok())
        .unwrap_or(SPEC_ASSISTANT_PROBE_ROUNDS_DEFAULT)
}

#[derive(Debug, Default)]
pub struct AssistantGate {
    decode_ns: AtomicU64,
    decode_rounds: AtomicU64,
    spec_ns: AtomicU64,
    spec_rounds: AtomicU64,
    spec_emitted: AtomicU64,
}

impl AssistantGate {
    pub const fn new() -> Self {
        Self {
            decode_ns: AtomicU64::new(0),
            decode_rounds: AtomicU64::new(0),
            spec_ns: AtomicU64::new(0),
            spec_rounds: AtomicU64::new(0),
            spec_emitted: AtomicU64::new(0),
        }
    }

    pub fn observe_decode(&self, ns: u64) {
        self.decode_ns.fetch_add(ns, Ordering::Relaxed);
        self.decode_rounds.fetch_add(1, Ordering::Relaxed);
    }

    pub fn observe_spec(&self, ns: u64, emitted: usize) {
        self.spec_ns.fetch_add(ns, Ordering::Relaxed);
        self.spec_rounds.fetch_add(1, Ordering::Relaxed);
        self.spec_emitted
            .fetch_add(emitted as u64, Ordering::Relaxed);
    }

    fn mean(sum: &AtomicU64, n: &AtomicU64) -> Option<f64> {
        let c = n.load(Ordering::Relaxed);
        if c == 0 {
            return None;
        }
        Some(sum.load(Ordering::Relaxed) as f64 / c as f64)
    }

    pub fn decode_ms(&self) -> Option<f64> {
        Self::mean(&self.decode_ns, &self.decode_rounds).map(|v| v / 1e6)
    }

    pub fn spec_ms(&self) -> Option<f64> {
        Self::mean(&self.spec_ns, &self.spec_rounds).map(|v| v / 1e6)
    }

    pub fn breakeven_tau(&self) -> Option<f64> {
        let d = self.decode_ms()?;
        let s = self.spec_ms()?;
        if d <= 0.0 {
            return None;
        }
        Some(s / d)
    }

    pub fn measured_tau(&self) -> Option<f64> {
        let r = self.spec_rounds.load(Ordering::Relaxed);
        if r == 0 {
            return None;
        }
        Some(self.spec_emitted.load(Ordering::Relaxed) as f64 / r as f64)
    }

    pub fn should_draft(&self, probe_rounds: u64) -> bool {
        let rounds = self.spec_rounds.load(Ordering::Relaxed);
        if rounds < probe_rounds.min(SPEC_ASSISTANT_PROBE_MIN_ROUNDS) {
            return true;
        }
        match (self.measured_tau(), self.breakeven_tau()) {
            (Some(tau), Some(be)) => {
                if rounds < probe_rounds && tau >= be * SPEC_ASSISTANT_GATE_HOPELESS {
                    return true;
                }
                tau >= be * SPEC_ASSISTANT_GATE_MARGIN
            }
            _ => rounds < probe_rounds,
        }
    }

    pub fn summary(&self) -> String {
        let f = |v: Option<f64>| v.map(|x| format!("{x:.3}")).unwrap_or_else(|| "n/a".into());
        format!(
            "decode_ms={} spec_round_ms={} tau={} breakeven_tau={} probe_rounds_done={}",
            f(self.decode_ms()),
            f(self.spec_ms()),
            f(self.measured_tau()),
            f(self.breakeven_tau()),
            self.spec_rounds.load(Ordering::Relaxed),
        )
    }
}

pub trait ChainVerifyTarget {
    fn verify_chain(&mut self, batch: &[u32]) -> Result<Vec<u32>>;
    fn advance(&mut self, n: usize) -> Result<()>;
    fn capacity(&self) -> usize;
}

pub fn run_spec_round<T: ChainVerifyTarget + ?Sized>(
    target: &mut T,
    bonus: u32,
    draft: &[u32],
) -> Result<(ChainAccept, Vec<u32>)> {
    let cap = target.capacity();
    ensure!(cap >= 1, "verify target reports zero capacity");
    ensure!(
        draft.len() < cap,
        "draft len {} needs verify capacity {}, target has {cap}",
        draft.len(),
        draft.len() + 1
    );
    let k = draft.len() + 1;
    let batch = build_chain_batch(bonus, draft, k, true)?;
    let amax = target.verify_chain(&batch)?;
    let acc = accept_prefix_argmax(&batch, &amax)?;
    target.advance(acc.commit_len)?;
    let mut emitted = Vec::with_capacity(acc.commit_len);
    emitted.extend_from_slice(&draft[..acc.commit_len - 1]);
    emitted.push(acc.next_bonus);
    Ok((acc, emitted))
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct SpecStats {
    pub rounds: usize,
    pub rounds_with_draft: usize,
    pub drafted: usize,
    pub accepted: usize,
    pub emitted: usize,
}

impl SpecStats {
    pub fn tau(&self) -> f64 {
        if self.rounds == 0 {
            return 0.0;
        }
        self.emitted as f64 / self.rounds as f64
    }

    pub fn accept_rate(&self) -> f64 {
        if self.drafted == 0 {
            return 0.0;
        }
        self.accepted as f64 / self.drafted as f64
    }

    pub fn summary(&self) -> String {
        format!(
            "rounds={} rounds_with_draft={} drafted={} accepted={} emitted={} tau={:.3} accept_rate={:.3}",
            self.rounds,
            self.rounds_with_draft,
            self.drafted,
            self.accepted,
            self.emitted,
            self.tau(),
            self.accept_rate(),
        )
    }
}

pub struct SpecLoop {
    suffix: SuffixAutomaton,
    ema: AcceptEma,
    knobs: SpecKnobs,
    stats: SpecStats,
}

impl SpecLoop {
    pub fn new(knobs: SpecKnobs) -> Self {
        Self {
            suffix: SuffixAutomaton::new(),
            ema: AcceptEma::new(0.2, 2.0),
            knobs,
            stats: SpecStats::default(),
        }
    }

    pub fn prime(&mut self, tokens: &[u32]) {
        self.suffix.extend_slice(tokens);
    }

    pub fn context_len(&self) -> usize {
        self.suffix.len()
    }

    pub fn knobs(&self) -> SpecKnobs {
        self.knobs
    }

    pub fn stats(&self) -> SpecStats {
        self.stats
    }

    pub fn ema_value(&self) -> f64 {
        self.ema.value()
    }

    pub fn propose_draft(&self, cap: usize) -> Vec<u32> {
        self.propose_draft_min(cap, self.knobs.min_match)
    }

    pub fn propose_draft_min(&self, cap: usize, min_match: usize) -> Vec<u32> {
        let limit = self.knobs.k.min(cap.saturating_sub(1));
        if limit == 0 {
            return Vec::new();
        }
        match self.suffix.propose(limit, min_match) {
            Some(p) => p.tokens,
            None => Vec::new(),
        }
    }

    pub fn round<T: ChainVerifyTarget + ?Sized>(
        &mut self,
        target: &mut T,
        bonus: u32,
    ) -> Result<Vec<u32>> {
        let draft = self.propose_draft(target.capacity());
        self.round_with_draft(target, bonus, draft)
    }

    pub fn round_with_draft<T: ChainVerifyTarget + ?Sized>(
        &mut self,
        target: &mut T,
        bonus: u32,
        mut draft: Vec<u32>,
    ) -> Result<Vec<u32>> {
        draft.truncate(target.capacity().saturating_sub(1));
        let (acc, emitted) = run_spec_round(target, bonus, &draft)?;
        self.stats.rounds += 1;
        if !draft.is_empty() {
            self.stats.rounds_with_draft += 1;
            self.stats.drafted += draft.len();
            self.stats.accepted += acc.draft_accepted;
            self.ema.observe(acc.draft_accepted);
        }
        self.stats.emitted += emitted.len();
        self.suffix.extend_slice(&emitted);
        Ok(emitted)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ms(v: f64) -> u64 {
        (v * 1e6) as u64
    }

    #[test]
    fn gate_probes_then_disables_when_tau_below_breakeven() {
        let g = AssistantGate::new();
        assert!(g.should_draft(16), "must probe before it has any samples");
        for _ in 0..SPEC_ASSISTANT_PROBE_MIN_ROUNDS {
            g.observe_decode(ms(26.0));
            g.observe_spec(ms(163.0), 1);
        }
        let be = g.breakeven_tau().expect("breakeven");
        assert!((be - 163.0 / 26.0).abs() < 1e-6, "breakeven {be}");
        assert!(
            !g.should_draft(16),
            "tau 1.0 vs breakeven {be} is hopeless; must abort the probe early"
        );
    }

    #[test]
    fn gate_enables_when_tau_clears_breakeven_with_margin() {
        let g = AssistantGate::new();
        for _ in 0..32 {
            g.observe_decode(ms(26.0));
            g.observe_spec(ms(30.0), 3);
        }
        let be = g.breakeven_tau().expect("breakeven");
        assert!((be - 30.0 / 26.0).abs() < 1e-6, "breakeven {be}");
        assert_eq!(g.measured_tau(), Some(3.0));
        assert!(g.should_draft(16), "tau 3.0 clears breakeven {be} * margin");
    }

    #[test]
    fn gate_keeps_probing_when_marginal() {
        let g = AssistantGate::new();
        for _ in 0..SPEC_ASSISTANT_PROBE_MIN_ROUNDS {
            g.observe_decode(ms(26.0));
            g.observe_spec(ms(30.0), 1);
        }
        let be = g.breakeven_tau().expect("breakeven");
        let tau = g.measured_tau().expect("tau");
        assert!(tau < be * SPEC_ASSISTANT_GATE_MARGIN, "not yet profitable");
        assert!(tau >= be * SPEC_ASSISTANT_GATE_HOPELESS, "not hopeless");
        assert!(g.should_draft(16), "marginal case must keep probing");
        assert!(
            !g.should_draft(SPEC_ASSISTANT_PROBE_MIN_ROUNDS),
            "budget spent"
        );
    }

    #[test]
    fn gate_without_samples_never_claims_profitability() {
        let g = AssistantGate::new();
        assert_eq!(g.breakeven_tau(), None);
        assert_eq!(g.measured_tau(), None);
        assert!(!g.should_draft(0), "no probe budget and no evidence");
    }
}
