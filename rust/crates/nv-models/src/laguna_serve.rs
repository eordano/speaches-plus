#![cfg(feature = "cuda")]

use anyhow::{Context, Result};
use candle_core::{Device, Tensor};
use std::collections::HashMap;
use std::path::Path;
use std::sync::mpsc::{Receiver, Sender};
use std::sync::Arc;

use crate::laguna::{Laguna, LagunaKvCache};
use crate::laguna_dflash::{
    accept_block_on_device, accept_block_on_host, adapt_truncate_len, argmax_row,
    dflash_adapt_enabled, dflash_adapt_thresh, dflash_window_mode, resolve_tap_layers,
    tap_list_mode, AcceptSlots, DflashCtxCache, DflashGraphProposer, DflashWindowMode,
    LagunaDflash, LagunaDflashConfig, LookupState, ROPE_TABLE_CAP,
};
use crate::laguna_step_graph::{LagunaStepGraph, LagunaVerifyGraph};

const PREFILL_CHUNK: usize = 256;

const PROPOSER_INIT_ATTEMPTS_2_A_PERSISTENT_CAPTURE_FAILURE_MUST_NOT_FORK_A_STREAM_EVERY_ROUND:
    usize = 2;

pub struct ProposerInitBudget(HashMap<u32, usize>);

impl ProposerInitBudget {
    pub fn new() -> Self {
        Self(HashMap::new())
    }

    pub fn admit(&mut self, key: u32) -> Option<usize> {
        let n = self.0.entry(key).or_insert(0);
        if *n >= PROPOSER_INIT_ATTEMPTS_2_A_PERSISTENT_CAPTURE_FAILURE_MUST_NOT_FORK_A_STREAM_EVERY_ROUND
        {
            return None;
        }
        *n += 1;
        Some(*n)
    }
}

impl Default for ProposerInitBudget {
    fn default() -> Self {
        Self::new()
    }
}

pub enum SpecServeEvent {
    Tokens(Vec<u32>),
    Done,
    Error(String),
}

pub struct SpecServeJob {
    pub prompt_ids: Vec<u32>,
    pub prompt_text: String,
    pub max_new: usize,
    pub eos_ids: Vec<u32>,
    pub emit: Box<dyn FnMut(SpecServeEvent) -> bool + Send>,
}

pub fn load_dflash_draft(dir: &Path, device: &Device) -> Result<LagunaDflash> {
    let cfg = LagunaDflashConfig::from_hf_json_file(&dir.join("config.json"))
        .with_context(|| format!("parse dflash config in {}", dir.display()))?;
    let weights = nv_weights::WeightLoader::open_dir(dir, device)
        .with_context(|| format!("open dflash weights in {}", dir.display()))?;
    LagunaDflash::from_loader(cfg, &weights, device).context("instantiate DFlash draft")
}

fn validate_draft(target: &Laguna, draft: &LagunaDflash, num_spec: usize) -> Result<()> {
    let dcfg = draft.config();
    anyhow::ensure!(
        num_spec >= 1 && num_spec < dcfg.dflash_config.block_size,
        "num_speculative must be in 1..{}, got {num_spec}",
        dcfg.dflash_config.block_size
    );
    let n_target = target.config().num_hidden_layers;
    anyhow::ensure!(
        dcfg.dflash_config.num_target_layers == n_target,
        "draft expects target with {} layers, got {n_target}",
        dcfg.dflash_config.num_target_layers
    );
    for &li in &dcfg.dflash_config.target_layer_ids {
        anyhow::ensure!(
            li < n_target,
            "aux layer {li} out of range for target ({n_target})"
        );
    }
    anyhow::ensure!(
        target.config().hidden_size == dcfg.hidden_size,
        "target/draft hidden size mismatch"
    );
    anyhow::ensure!(
        target.config().vocab_size == dcfg.vocab_size,
        "target/draft vocab size mismatch"
    );
    Ok(())
}

enum Flow {
    Continue,
    Stop,
}

fn push_round(job: &mut SpecServeJob, toks: &[u32], total: &mut usize) -> Flow {
    let budget = job.max_new;
    let remaining = budget.saturating_sub(*total);
    if remaining == 0 {
        return Flow::Stop;
    }
    let eos_idx = toks.iter().position(|t| job.eos_ids.contains(t));
    let (kept, hard_stop) = match eos_idx {
        Some(i) if i < remaining => (&toks[..=i], true),
        _ if toks.len() > remaining => (&toks[..remaining], true),
        _ => (toks, false),
    };
    *total += kept.len();
    let alive = (job.emit)(SpecServeEvent::Tokens(kept.to_vec()));
    if !alive || hard_stop || *total >= budget {
        Flow::Stop
    } else {
        Flow::Continue
    }
}

struct SpecState<'m> {
    target: &'m Laguna,
    draft: Option<&'m LagunaDflash>,
    device: Device,
    step: LagunaStepGraph,
    ctx: Option<DflashCtxCache>,
    proposers: HashMap<u32, DflashGraphProposer>,
    proposer_init_budget: ProposerInitBudget,
    verify_graphs: HashMap<usize, LagunaVerifyGraph<'m>>,
    accept_slots: HashMap<usize, AcceptSlots>,
    aux_layers: Vec<usize>,
    lookup: Option<LookupState>,
    num_spec: usize,
    max_seq: usize,
}

impl<'m> SpecState<'m> {
    fn new(
        target: &'m Arc<Laguna>,
        draft: Option<&'m LagunaDflash>,
        num_spec: usize,
        max_seq: usize,
    ) -> Result<Self> {
        let device = target.device().clone();
        anyhow::ensure!(
            matches!(device, Device::Cuda(_)),
            "laguna spec serving requires a CUDA device"
        );
        let cache = target.new_kv_cache(max_seq)?;
        let step = LagunaStepGraph::new(Arc::clone(target), cache)
            .context("laguna spec serving: step graph init")?;
        let num_spec = match draft {
            Some(d) => {
                let cap = d.config().dflash_config.block_size.saturating_sub(1).max(1);
                if num_spec > cap {
                    eprintln!(
                        "[laguna_serve] num_speculative {num_spec} > block_size-1 ({cap}); clamping"
                    );
                    cap
                } else {
                    num_spec
                }
            }
            None => num_spec,
        };
        let draft = match draft {
            Some(d) => match validate_draft(target, d, num_spec) {
                Ok(()) => {
                    target.set_device_verify_routing(true);
                    Some(d)
                }
                Err(e) => {
                    eprintln!("[laguna_serve] draft incompatible, serving M=1 only: {e:#}");
                    None
                }
            },
            None => None,
        };
        let ctx = draft.map(|d| d.new_ctx_cache());
        let aux_layers = draft
            .map(|d| resolve_tap_layers(d.config(), tap_list_mode()))
            .unwrap_or_default();
        let lookup = if draft.is_some() {
            let l = LookupState::from_env();
            if let Some(l) = &l {
                eprintln!("[laguna_serve] lookup draft on: {}", l.describe());
            }
            l
        } else {
            None
        };
        Ok(Self {
            target: target.as_ref(),
            draft,
            device,
            step,
            ctx,
            proposers: HashMap::new(),
            proposer_init_budget: ProposerInitBudget::new(),
            verify_graphs: HashMap::new(),
            accept_slots: HashMap::new(),
            aux_layers,
            lookup,
            num_spec,
            max_seq,
        })
    }

    fn warmup(&mut self) -> Result<()> {
        let ids: Vec<u32> = (0..48u32).map(|i| 100 + i).collect();
        let max_new = 2 * (self.num_spec + 2);
        let mut texts: Vec<&str> = vec!["Hello there, how are you doing today?"];
        if self.draft.is_some() {
            texts.push("```python\ndef square(x):\n    return x * x\n```");
        }
        for text in texts {
            let mut job = SpecServeJob {
                prompt_ids: ids.clone(),
                prompt_text: text.to_string(),
                max_new,
                eos_ids: Vec::new(),
                emit: Box::new(|_| true),
            };
            self.serve_one(&mut job)
                .with_context(|| format!("laguna spec serving warmup ({text:?})"))?;
        }
        self.device.synchronize().ok();
        Ok(())
    }

    fn serve_one(&mut self, job: &mut SpecServeJob) -> Result<()> {
        anyhow::ensure!(!job.prompt_ids.is_empty(), "spec serve: empty prompt");
        let n = job.prompt_ids.len();
        anyhow::ensure!(
            n + 2 <= self.max_seq,
            "spec serve: prompt ({n}) exceeds KV capacity ({})",
            self.max_seq
        );
        self.step.cache_mut().reset();
        if let Some(c) = self.ctx.as_mut() {
            c.reset();
        }
        if let Some(l) = self.lookup.as_mut() {
            l.reset();
            l.extend_slice(&job.prompt_ids);
        }
        if let Some(d) = self.draft {
            d.select_for_prompt_ctx(&job.prompt_text, n);
        }

        let mut last_logits: Option<Tensor> = None;
        let mut offset = 0usize;
        while offset < n {
            let m = PREFILL_CHUNK.min(n - offset);
            let toks = Tensor::from_vec(
                job.prompt_ids[offset..offset + m].to_vec(),
                (1usize, m),
                &self.device,
            )?;
            let pos: Vec<i32> = (offset as i32..(offset + m) as i32).collect();
            let pos_t = Tensor::from_vec(pos.clone(), m, &self.device)?;
            let (logits, aux) = self.target.forward_with_cache_aux_scoped(
                &toks,
                &pos_t,
                self.step.cache_mut(),
                &self.aux_layers,
                self.draft.is_some(),
            )?;
            if let (Some(d), Some(ctx)) = (self.draft, self.ctx.as_mut()) {
                let combined = d.combine_aux(&aux)?;
                let cpos = Tensor::from_vec(pos, m, &self.device)?;
                d.append_context(ctx, &combined, &cpos)?;
            }
            last_logits = Some(logits);
            offset += m;
        }
        let logits = last_logits.expect("non-empty prompt");
        let m_last = logits.dims()[1];
        let last_row: Vec<f32> = logits.narrow(1, m_last - 1, 1)?.flatten_all()?.to_vec1()?;
        let anchor = argmax_row(&last_row);
        drop(logits);
        if let Some(l) = self.lookup.as_mut() {
            l.extend(anchor);
        }

        let mut total = 0usize;
        if let Flow::Stop = push_round(job, &[anchor], &mut total) {
            return Ok(());
        }
        if self.draft.is_some() {
            self.decode_spec(job, anchor, n, &mut total)
        } else {
            self.decode_m1(job, anchor, n, &mut total)
        }
    }

    fn decode_m1(
        &mut self,
        job: &mut SpecServeJob,
        anchor: u32,
        num_ctx: usize,
        total: &mut usize,
    ) -> Result<()> {
        let mut last = anchor;
        let mut pos = num_ctx;
        loop {
            if pos + 1 > self.max_seq {
                return Ok(());
            }
            self.step.step(last)?;
            let tok = self.step.argmax_device()?;
            pos += 1;
            last = tok;
            if let Flow::Stop = push_round(job, &[tok], total) {
                return Ok(());
            }
        }
    }

    fn decode_spec(
        &mut self,
        job: &mut SpecServeJob,
        mut anchor: u32,
        mut num_ctx: usize,
        total: &mut usize,
    ) -> Result<()> {
        let draft = self.draft.expect("decode_spec without draft");
        let k = self.num_spec;
        let adapt = dflash_adapt_enabled();
        let adapt_thresh = dflash_adapt_thresh();
        let stats_on = std::env::var_os("NV_LAGUNA_SERVE_STATS").is_some();
        let prompt_n = num_ctx;
        let mut rounds = 0usize;
        let mut acc_total = 0usize;
        let mut pos0 = 0usize;
        let mut lookup_rounds = 0usize;
        let mut lookup_accepted = 0usize;
        let t0 = std::time::Instant::now();
        let log_stats = |rounds: usize,
                         acc_total: usize,
                         pos0: usize,
                         total: usize,
                         dt: f64,
                         lookup_rounds: usize,
                         lookup_accepted: usize| {
            if stats_on && rounds > 0 {
                let tau = (total.saturating_sub(1)) as f64 / rounds as f64;
                eprintln!(
                    "[laguna_serve_stats] prompt_tokens={prompt_n} k={k} rounds={rounds} emitted={total} tau={tau:.3} pos0={:.3} accepted={acc_total} lookup_rounds={lookup_rounds} lookup_accepted={lookup_accepted} decode_ms_tok={:.3}",
                    pos0 as f64 / rounds as f64,
                    1000.0 * dt / (total.saturating_sub(1).max(1)) as f64
                );
            }
        };
        loop {
            if num_ctx + k + 1 > self.max_seq {
                log_stats(
                    rounds,
                    acc_total,
                    pos0,
                    *total,
                    t0.elapsed().as_secs_f64(),
                    lookup_rounds,
                    lookup_accepted,
                );
                return Ok(());
            }
            let lookup_prop = self.lookup.as_ref().and_then(|l| l.propose(k));
            let from_lookup = lookup_prop.is_some();
            let ctx = self.ctx.as_ref().expect("spec ctx");
            let (mut drafts, conf) = match lookup_prop {
                Some(d) => (d, None),
                None => propose_round(
                    draft,
                    ctx,
                    &mut self.proposers,
                    &mut self.proposer_init_budget,
                    self.target,
                    anchor,
                    num_ctx,
                    k,
                )?,
            };
            if adapt {
                if let Some(c) = conf.as_deref() {
                    drafts.truncate(adapt_truncate_len(c, adapt_thresh));
                }
            }
            let mut block: Vec<u32> = Vec::with_capacity(drafts.len() + 1);
            block.push(anchor);
            block.extend_from_slice(&drafts);

            let (vlogits, vaux) = verify_block(
                self.target,
                &mut self.step,
                &mut self.verify_graphs,
                &self.aux_layers,
                &block,
                num_ctx,
                &self.device,
            )?;
            let (accepted, emitted) = match accept_block_on_device(
                self.target,
                &mut self.accept_slots,
                &vlogits,
                &drafts,
                None,
            ) {
                Ok(r) => r,
                Err(e) => {
                    eprintln!("[laguna_serve] device accept failed, host fallback: {e:#}");
                    accept_block_on_host(&vlogits, &drafts)?
                }
            };
            let consumed = 1 + accepted;
            self.step.cache_mut().rollback(block.len() - consumed)?;

            let ctx = self.ctx.as_mut().expect("spec ctx");
            let mut vaux_kept = Vec::with_capacity(vaux.len());
            for a in &vaux {
                vaux_kept.push(a.narrow(1, 0, consumed)?);
            }
            let combined = draft.combine_aux(&vaux_kept)?;
            let new_pos: Vec<i32> = (0..consumed).map(|i| (num_ctx + i) as i32).collect();
            let np = Tensor::from_vec(new_pos, consumed, &self.device)?;
            draft.append_context(ctx, &combined, &np)?;

            num_ctx += consumed;
            anchor = *emitted.last().expect("non-empty accept output");
            rounds += 1;
            acc_total += accepted;
            if accepted > 0 {
                pos0 += 1;
            }
            if let Some(l) = self.lookup.as_mut() {
                if from_lookup {
                    lookup_rounds += 1;
                    lookup_accepted += accepted;
                } else {
                    l.observe_dflash_round(accepted);
                }
                l.extend_slice(&emitted);
            }
            if let Flow::Stop = push_round(job, &emitted, total) {
                log_stats(
                    rounds,
                    acc_total,
                    pos0,
                    *total,
                    t0.elapsed().as_secs_f64(),
                    lookup_rounds,
                    lookup_accepted,
                );
                return Ok(());
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn propose_round(
    draft: &LagunaDflash,
    ctx: &DflashCtxCache,
    proposers: &mut HashMap<u32, DflashGraphProposer>,
    init_budget: &mut ProposerInitBudget,
    target: &Laguna,
    anchor: u32,
    num_ctx: usize,
    k: usize,
) -> Result<(Vec<u32>, Option<Vec<f32>>)> {
    let graph_ok = ctx.has_ring()
        && dflash_window_mode() == DflashWindowMode::Relaxed
        && num_ctx + k + 1 <= ROPE_TABLE_CAP;
    if graph_ok {
        let key = draft.active_rope_theta().to_bits();
        if !proposers.contains_key(&key) {
            if let Some(attempt) = init_budget.admit(key) {
                match DflashGraphProposer::new(draft, k) {
                    Ok(p) => {
                        proposers.insert(key, p);
                    }
                    Err(e) => eprintln!(
                        "[laguna_serve] graph proposer init failed (attempt {attempt} of {}{}), eager: {e:#}",
                        PROPOSER_INIT_ATTEMPTS_2_A_PERSISTENT_CAPTURE_FAILURE_MUST_NOT_FORK_A_STREAM_EVERY_ROUND,
                        if attempt
                            == PROPOSER_INIT_ATTEMPTS_2_A_PERSISTENT_CAPTURE_FAILURE_MUST_NOT_FORK_A_STREAM_EVERY_ROUND
                        {
                            "; falling back to eager drafting permanently"
                        } else {
                            ""
                        }
                    ),
                }
            }
        }
        if let Some(p) = proposers.get_mut(&key) {
            match p.propose(
                draft,
                ctx,
                anchor,
                num_ctx,
                target.embed_weight(),
                target.lm_head(),
            ) {
                Ok(v) => {
                    let c = p.last_conf().map(|s| s.to_vec());
                    return Ok((v, c));
                }
                Err(e) => eprintln!("[laguna_serve] graph propose failed, eager: {e:#}"),
            }
        }
    }
    if dflash_adapt_enabled() {
        let (d, c) = draft.propose_k_conf(
            ctx,
            anchor,
            num_ctx,
            k,
            target.embed_weight(),
            target.lm_head(),
        )?;
        Ok((d, Some(c)))
    } else {
        Ok((
            draft.propose_k(
                ctx,
                anchor,
                num_ctx,
                k,
                target.embed_weight(),
                target.lm_head(),
            )?,
            None,
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn batch_lane_spec_parsing() {
        assert_eq!(batch_lanes_from_spec(None), None);
        assert_eq!(batch_lanes_from_spec(Some("")), None);
        assert_eq!(batch_lanes_from_spec(Some("0")), None);
        assert_eq!(
            batch_lanes_from_spec(Some("1")),
            Some(LAGUNA_BATCH_DEFAULT_LANES)
        );
        assert_eq!(batch_lanes_from_spec(Some("2")), Some(2));
        assert_eq!(batch_lanes_from_spec(Some(" 4 ")), Some(4));
        assert_eq!(
            batch_lanes_from_spec(Some("64")),
            Some(LAGUNA_BATCH_MAX_LANES)
        );
        assert_eq!(batch_lanes_from_spec(Some("x")), None);
    }

    #[test]
    fn proposer_init_budget_admits_the_cap_then_permanently_refuses() {
        let cap =
            PROPOSER_INIT_ATTEMPTS_2_A_PERSISTENT_CAPTURE_FAILURE_MUST_NOT_FORK_A_STREAM_EVERY_ROUND;
        let mut b = ProposerInitBudget::new();
        for attempt in 1..=cap {
            assert_eq!(
                b.admit(42),
                Some(attempt),
                "admit must count attempts up to the cap"
            );
        }
        for round in 0..1000 {
            assert_eq!(
                b.admit(42),
                None,
                "round {round} past the cap re-admitted the proposer ctor: every re-admission \
                 forks a stream whose warmup installs per-stream quant caches, which is the \
                 unbounded S1 retry this budget exists to stop"
            );
        }
    }

    #[test]
    fn proposer_init_budget_counts_each_rope_theta_key_separately() {
        let mut b = ProposerInitBudget::new();
        assert_eq!(b.admit(1), Some(1));
        assert_eq!(b.admit(2), Some(1));
        assert_eq!(b.admit(1), Some(2));
        assert_eq!(b.admit(1), None);
        assert_eq!(b.admit(2), Some(2));
        assert_eq!(b.admit(2), None);
    }
}

fn verify_block<'m>(
    target: &'m Laguna,
    step: &mut LagunaStepGraph,
    graphs: &mut HashMap<usize, LagunaVerifyGraph<'m>>,
    aux_layers: &[usize],
    block: &[u32],
    num_ctx: usize,
    device: &Device,
) -> Result<(Tensor, Vec<Tensor>)> {
    let bs = block.len();
    if let Some(g) = graphs.get_mut(&bs) {
        match g.verify(step.cache_mut(), block) {
            Ok(r) => return Ok(r),
            Err(e) => {
                eprintln!("[laguna_serve] verify graph failed, eager: {e:#}");
                graphs.remove(&bs);
            }
        }
    } else {
        match LagunaVerifyGraph::new(target, step.cache(), bs, aux_layers) {
            Ok(mut g) => match g.verify(step.cache_mut(), block) {
                Ok(r) => {
                    graphs.insert(bs, g);
                    return Ok(r);
                }
                Err(e) => {
                    eprintln!("[laguna_serve] verify graph capture failed (bs={bs}), eager: {e:#}")
                }
            },
            Err(e) => eprintln!("[laguna_serve] verify graph init failed (bs={bs}), eager: {e:#}"),
        }
    }
    let pos: Vec<i32> = (0..bs).map(|i| (num_ctx + i) as i32).collect();
    let bt = Tensor::from_vec(block.to_vec(), (1usize, bs), device)?;
    let bp = Tensor::from_vec(pos, bs, device)?;
    target.forward_with_cache_aux_scoped(&bt, &bp, step.cache_mut(), aux_layers, true)
}

pub const LAGUNA_BATCH_ENV: &str = "NV_LAGUNA_BATCH";
pub const LAGUNA_BATCH_DEFAULT_LANES: usize = 4;
pub const LAGUNA_BATCH_MAX_LANES: usize = 8;

pub fn batch_lanes_from_spec(spec: Option<&str>) -> Option<usize> {
    let t = spec?.trim();
    if t.is_empty() || t == "0" {
        return None;
    }
    if t == "1" {
        return Some(LAGUNA_BATCH_DEFAULT_LANES);
    }
    t.parse::<usize>()
        .ok()
        .filter(|&n| n >= 2)
        .map(|n| n.min(LAGUNA_BATCH_MAX_LANES))
}

pub fn batch_lanes_from_env() -> Option<usize> {
    batch_lanes_from_spec(std::env::var(LAGUNA_BATCH_ENV).ok().as_deref())
}

pub struct BatchM1State {
    target: Arc<Laguna>,
    device: Device,
    max_seq: usize,
    lanes: Vec<LagunaKvCache>,
}

impl BatchM1State {
    pub fn new(target: &Arc<Laguna>, lanes: usize, max_seq: usize) -> Result<Self> {
        anyhow::ensure!(
            matches!(target.device(), Device::Cuda(_)),
            "laguna batched serving requires a CUDA device"
        );
        anyhow::ensure!(lanes >= 2, "a 1-lane batch loop is just the M=1 loop");
        let mut v = Vec::with_capacity(lanes);
        for _ in 0..lanes {
            v.push(target.new_kv_cache(max_seq)?);
        }
        Ok(Self {
            target: Arc::clone(target),
            device: target.device().clone(),
            max_seq,
            lanes: v,
        })
    }

    pub fn lane_count(&self) -> usize {
        self.lanes.len()
    }

    fn prefill_lane(&mut self, lane: usize, ids: &[u32]) -> Result<u32> {
        anyhow::ensure!(!ids.is_empty(), "batch serve: empty prompt");
        let n = ids.len();
        anyhow::ensure!(
            n + 2 <= self.max_seq,
            "batch serve: prompt ({n}) exceeds KV capacity ({})",
            self.max_seq
        );
        self.lanes[lane].reset();
        let mut last_logits: Option<Tensor> = None;
        let mut offset = 0usize;
        while offset < n {
            let m = PREFILL_CHUNK.min(n - offset);
            let toks = Tensor::from_vec(ids[offset..offset + m].to_vec(), (1usize, m), &self.device)?;
            let pos: Vec<i32> = (offset as i32..(offset + m) as i32).collect();
            let pos_t = Tensor::from_vec(pos, m, &self.device)?;
            let logits = self
                .target
                .forward_with_cache(&toks, &pos_t, &mut self.lanes[lane])?;
            last_logits = Some(logits);
            offset += m;
        }
        let logits = last_logits.expect("non-empty prompt");
        let m_last = logits.dims()[1];
        let last_row: Vec<f32> = logits.narrow(1, m_last - 1, 1)?.flatten_all()?.to_vec1()?;
        Ok(argmax_row(&last_row))
    }

    fn admit(&mut self, lane: usize, mut job: SpecServeJob) -> Option<LaneRun> {
        match self.prefill_lane(lane, &job.prompt_ids) {
            Ok(anchor) => {
                let pos = job.prompt_ids.len();
                let mut total = 0usize;
                if let Flow::Stop = push_round(&mut job, &[anchor], &mut total) {
                    let _ = (job.emit)(SpecServeEvent::Done);
                    return None;
                }
                Some(LaneRun {
                    job,
                    last: anchor,
                    pos,
                    total,
                })
            }
            Err(e) => {
                let _ = (job.emit)(SpecServeEvent::Error(format!("{e:#}")));
                None
            }
        }
    }

    fn decode_step(&mut self, slots: &mut [Option<LaneRun>]) -> Result<()> {
        for s in slots.iter_mut() {
            if s.as_ref().is_some_and(|r| r.pos + 1 > self.max_seq) {
                let mut r = s.take().expect("checked occupied");
                let _ = (r.job.emit)(SpecServeEvent::Done);
            }
        }
        let active: Vec<usize> = slots
            .iter()
            .enumerate()
            .filter_map(|(i, s)| s.as_ref().map(|_| i))
            .collect();
        if active.is_empty() {
            return Ok(());
        }
        let tokens: Vec<u32> = active
            .iter()
            .map(|&i| slots[i].as_ref().expect("active slot").last)
            .collect();
        let positions: Vec<usize> = active
            .iter()
            .map(|&i| slots[i].as_ref().expect("active slot").pos)
            .collect();
        let mut cache_refs: Vec<&mut LagunaKvCache> = self
            .lanes
            .iter_mut()
            .enumerate()
            .filter(|(i, _)| active.contains(i))
            .map(|(_, c)| c)
            .collect();
        let logits = self
            .target
            .forward_decode_batched(&tokens, &positions, &mut cache_refs)?;
        let host: Vec<f32> = logits.flatten_all()?.to_vec1()?;
        let vocab = host.len() / active.len();
        for (row, &i) in active.iter().enumerate() {
            let tok = argmax_row(&host[row * vocab..(row + 1) * vocab]);
            let r = slots[i].as_mut().expect("active slot");
            r.pos += 1;
            r.last = tok;
            if let Flow::Stop = push_round(&mut r.job, &[tok], &mut r.total) {
                let mut r = slots[i].take().expect("active slot");
                let _ = (r.job.emit)(SpecServeEvent::Done);
            }
        }
        Ok(())
    }

    pub fn serve_batch(&mut self, jobs: Vec<SpecServeJob>) -> Result<()> {
        anyhow::ensure!(
            jobs.len() <= self.lanes.len(),
            "serve_batch: {} jobs > {} lanes",
            jobs.len(),
            self.lanes.len()
        );
        let mut slots: Vec<Option<LaneRun>> = (0..self.lanes.len()).map(|_| None).collect();
        for (i, job) in jobs.into_iter().enumerate() {
            slots[i] = self.admit(i, job);
        }
        while slots.iter().any(|s| s.is_some()) {
            self.decode_step(&mut slots)?;
        }
        Ok(())
    }

    fn warmup(&mut self) -> Result<()> {
        let ids: Vec<u32> = (0..48u32).map(|i| 100 + i).collect();
        let n_jobs = self.lanes.len().min(2);
        let batch: Vec<SpecServeJob> = (0..n_jobs)
            .map(|_| SpecServeJob {
                prompt_ids: ids.clone(),
                prompt_text: "Hello there, how are you doing today?".to_string(),
                max_new: 4,
                eos_ids: Vec::new(),
                emit: Box::new(|_| true),
            })
            .collect();
        self.serve_batch(batch)
            .context("laguna batch serving warmup")?;
        self.device.synchronize().ok();
        Ok(())
    }
}

struct LaneRun {
    job: SpecServeJob,
    last: u32,
    pos: usize,
    total: usize,
}

fn batch_m1_serve_loop(
    target: Arc<Laguna>,
    lanes: usize,
    max_seq: usize,
    jobs: Receiver<SpecServeJob>,
    ready: Sender<bool>,
) -> Result<()> {
    let mut st = BatchM1State::new(&target, lanes, max_seq)?;
    st.warmup()?;
    eprintln!(
        "[laguna_serve] batched M=1 lanes on: {} lanes, {} max_seq per lane, \
         free lanes admit queued jobs at every step boundary",
        st.lane_count(),
        max_seq
    );
    let _ = ready.send(false);
    let mut slots: Vec<Option<LaneRun>> = (0..lanes).map(|_| None).collect();
    loop {
        if slots.iter().all(|s| s.is_none()) {
            match jobs.recv() {
                Ok(job) => {
                    slots[0] = st.admit(0, job);
                }
                Err(_) => return Ok(()),
            }
        }
        loop {
            let Some(free) = slots.iter().position(|s| s.is_none()) else {
                break;
            };
            match jobs.try_recv() {
                Ok(job) => slots[free] = st.admit(free, job),
                Err(_) => break,
            }
        }
        if slots.iter().all(|s| s.is_none()) {
            continue;
        }
        if let Err(e) = st.decode_step(&mut slots) {
            let msg = format!("{e:#}");
            for s in slots.iter_mut() {
                if let Some(mut r) = s.take() {
                    let _ = (r.job.emit)(SpecServeEvent::Error(msg.clone()));
                }
            }
        }
    }
}

pub fn spec_serve_loop(
    target: Arc<Laguna>,
    draft: Option<LagunaDflash>,
    num_spec: usize,
    max_seq: usize,
    jobs: Receiver<SpecServeJob>,
    ready: Sender<bool>,
) -> Result<()> {
    let target = target;
    let draft = draft;
    if let Some(lanes) = batch_lanes_from_env() {
        if draft.is_none() {
            return batch_m1_serve_loop(target, lanes, max_seq, jobs, ready);
        }
        eprintln!(
            "[laguna_serve] {LAGUNA_BATCH_ENV} ignored: DFlash draft active, batched lanes are M=1 only"
        );
    }
    let mut st = SpecState::new(&target, draft.as_ref(), num_spec, max_seq)?;
    st.warmup()?;
    let has_draft = st.draft.is_some();
    let _ = ready.send(has_draft);
    while let Ok(mut job) = jobs.recv() {
        match st.serve_one(&mut job) {
            Ok(()) => {
                let _ = (job.emit)(SpecServeEvent::Done);
            }
            Err(e) => {
                let _ = (job.emit)(SpecServeEvent::Error(format!("{e:#}")));
            }
        }
    }
    Ok(())
}
