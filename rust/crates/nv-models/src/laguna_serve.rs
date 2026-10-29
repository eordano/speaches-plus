#![cfg(feature = "cuda")]

use anyhow::{bail, ensure, Context, Result};
use candle_core::{DType, Device, Tensor};
use std::collections::HashMap;
use std::path::Path;
use std::sync::mpsc::{Receiver, Sender};
use std::sync::Arc;

use crate::laguna::{Laguna, LagunaKvCache};
use crate::laguna_batch_graph::LagunaBatchStepGraph;
use crate::laguna_dflash::context_commit::DflashCtxCommitGraph;
use crate::laguna_dflash::{
    accept_block_on_device_with_status, accept_block_on_host, adapt_truncate_len,
    checked_target_argmax_row, dflash_adapt_enabled, dflash_adapt_thresh, resolve_tap_layers,
    tap_list_mode, vegas_enabled, AcceptSlots, DflashCtxCache, DflashGraphProposer, LagunaDflash,
    LagunaDflashConfig, LookupState,
};
use crate::laguna_step_graph::{LagunaStepGraph, LagunaVerifyGraph, StepCache};
use std::env;

mod prefix;

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

    PromptCached(usize),
    Done,
    Error(String),
}

pub struct SpecServeJob {
    pub prompt_ids: Vec<u32>,
    pub max_new: usize,
    pub eos_ids: Vec<u32>,
    pub emit: Box<dyn FnMut(SpecServeEvent) -> bool + Send>,
}

impl SpecServeJob {
    pub fn client_is_live(&mut self) -> bool {
        (self.emit)(SpecServeEvent::Tokens(Vec::new()))
    }

    fn report_execution_result(&mut self, result: Result<()>) -> Result<()> {
        let result = result.context(LagunaRequestExecutionFailure);
        if let Err(error) = &result {
            let _ = (self.emit)(SpecServeEvent::Error(format!("{error:#}")));
            eprintln!("[laguna_serve] request execution failed: {error:#}");
        }
        result
    }
}

pub const WORKER_RETIRED_ERROR_IS_THE_EXACT_STRING_THE_HTTP_SHARED_RETIREMENT_GUARD_MATCHES: &str =
    "engine busy: Laguna worker retired; try later.";

#[derive(Debug)]
pub struct LagunaRequestExecutionFailure;

impl std::fmt::Display for LagunaRequestExecutionFailure {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("engine busy: Laguna request execution failed; try later")
    }
}

impl std::error::Error for LagunaRequestExecutionFailure {}

pub struct RetirementProbe {
    preallocated_so_healthy_allocation_exhaustion_is_not_misread_as_a_dead_context: Tensor,
}

impl RetirementProbe {
    fn new(device: &Device) -> Result<Self> {
        Ok(Self {
            preallocated_so_healthy_allocation_exhaustion_is_not_misread_as_a_dead_context:
                Tensor::zeros(1, DType::F32, device)?,
        })
    }
    fn device_still_serves(&self) -> Result<()> {
        self.preallocated_so_healthy_allocation_exhaustion_is_not_misread_as_a_dead_context
            .to_vec1::<f32>()?;
        Ok(())
    }
}

fn retire_queued_jobs_before_the_worker_exits(jobs: &Receiver<SpecServeJob>) {
    let mut drained = 0usize;
    while let Ok(mut queued) = jobs.try_recv() {
        let _ = (queued.emit)(SpecServeEvent::Error(
            WORKER_RETIRED_ERROR_IS_THE_EXACT_STRING_THE_HTTP_SHARED_RETIREMENT_GUARD_MATCHES
                .to_string(),
        ));
        drained += 1;
    }
    eprintln!("[laguna_serve] retirement marker delivered to the failing job and to {drained} queued job(s); the worker now exits and drops its engine state");
}

fn request_error_or_worker_retirement(
    probe: &RetirementProbe,
    e: anyhow::Error,
) -> (String, Option<anyhow::Error>) {
    request_error_or_worker_retirement_from_probe(probe.device_still_serves(), e)
}

fn request_error_or_worker_retirement_from_probe(
    probe_result: Result<()>,
    e: anyhow::Error,
) -> (String, Option<anyhow::Error>) {
    match probe_result {
        Ok(()) => (format!("{e:#}"), None),
        Err(fault) => {
            eprintln!("[laguna_serve] retiring worker: request failed ({e:#}); the device probe after the failure failed too ({fault:#})");
            let retired = fault.context(format!(
                "Laguna worker retired after a nonrecoverable device fault; request error: {e:#}"
            ));
            (
                WORKER_RETIRED_ERROR_IS_THE_EXACT_STRING_THE_HTTP_SHARED_RETIREMENT_GUARD_MATCHES
                    .to_string(),
                Some(retired),
            )
        }
    }
}

fn retirement_teardown_marker(what: &str) {
    if nv_config::env_arm_opt_in_eq_some_1_reread_every_call("NV_GRAPH_TEARDOWN_DEBUG") {
        eprintln!("[laguna-teardown] {what}");
    }
}

pub fn load_dflash_draft(dir: &Path, device: &Device) -> Result<LagunaDflash> {
    let cfg = LagunaDflashConfig::from_hf_json_file(&dir.join("config.json"))
        .with_context(|| format!("parse dflash config in {}", dir.display()))?;
    let weights = nv_weights::WeightLoader::open_dir(dir, device)
        .with_context(|| format!("open dflash weights in {}", dir.display()))?;
    let draft =
        LagunaDflash::from_loader(cfg, &weights, device).context("instantiate DFlash draft")?;
    let proof = draft.quantization_summary();
    eprintln!("[laguna_dflash_weights] model_dir={} weight_format={} packed_linears={} excluded_linears={} excluded_norms={} bf16_fallback_linears={} w4a16_rows={} armed_linears={} packed_weight_bytes={} scale_bytes={}", dir.display(), proof.weight_format, proof.packed_linears, proof.excluded_linears, proof.excluded_norms, proof.bf16_fallback_linears, proof.bf16act_rows, proof.bf16act_armed_linears, proof.packed_weight_bytes, proof.scale_bytes);
    Ok(draft)
}

fn validate_draft(target: &Laguna, draft: &LagunaDflash, num_spec: usize) -> Result<()> {
    let dcfg = draft.config();
    ensure!(
        num_spec >= 1 && num_spec < dcfg.dflash_config.block_size,
        "num_speculative must be in 1..{}, got {num_spec}",
        dcfg.dflash_config.block_size
    );
    let n_target = target.config().num_hidden_layers;
    ensure!(
        dcfg.dflash_config.num_target_layers == n_target,
        "draft expects target with {} layers, got {n_target}",
        dcfg.dflash_config.num_target_layers
    );
    for &li in &dcfg.dflash_config.target_layer_ids {
        ensure!(
            li < n_target,
            "aux layer {li} out of range for target ({n_target})"
        );
    }
    ensure!(
        target.config().hidden_size == dcfg.hidden_size,
        "target/draft hidden size mismatch"
    );
    ensure!(
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

fn device_draft_transport_from_env() -> Result<bool> {
    match env::var("NV_LAGUNA_DFLASH_DEVICE_TRANSPORT") {
        Err(std::env::VarError::NotPresent) => Ok(false),
        Ok(v) if v == "0" => Ok(false),
        Ok(v) if v == "1" => Ok(true),
        _ => bail!("NV_LAGUNA_DFLASH_DEVICE_TRANSPORT must be 0 or 1"),
    }
}

struct SpecState<'m> {
    target: &'m Laguna,
    draft: Option<&'m LagunaDflash>,
    device: Device,
    probe: RetirementProbe,
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
    device_draft_transport: bool,
    handoff_source: Option<crate::laguna::LagunaKvCache>,
    prefix: prefix::PrefixState,
}

pub fn bf16_prefill_handoff_opt_in() -> bool {
    nv_config::env_arm_opt_in_eq_some_1_reread_every_call("NV_LAGUNA_PREFILL_BF16_HANDOFF")
}

fn validate_bf16_prefill_handoff_selection() -> Result<()> {
    if bf16_prefill_handoff_opt_in() {
        ensure!(!crate::laguna_prefix::prefix_reuse_enabled(),
            "NV_LAGUNA_PREFILL_BF16_HANDOFF=1 requires NV_LAGUNA_PREFIX_REUSE=0; source-cache checkpoint reuse is not qualified");
        ensure!(
            batch_m2_graph_enabled_from_env()
                || batch_m2_spec_enabled_from_env()
                || (batch_lanes_from_env().is_none() && !batch_graph_enabled_from_env()?),
            "NV_LAGUNA_PREFILL_BF16_HANDOFF=1 requires solo serving or prepared m2/m2-spec lanes"
        );
        ensure!(
            !nv_config::env_arm_present_at_all_reread_every_call("NV_LAGUNA_RING_ATTN"),
            "BF16 prefill handoff requires compacting CUDA ring KV; unset NV_LAGUNA_RING_ATTN"
        );
    }
    Ok(())
}

const HANDOFF_PREFLIGHT_RESERVE_GIB_2_KEEPS_GRAPH_CAPTURE_AND_PREFILL_SCRATCH_ROOM_AFTER_BOTH_CACHES: f64 = 2.0;

fn device_free_total_bytes(device: &Device) -> Option<(u64, u64)> {
    let Device::Cuda(d) = device else { return None };
    d.cuda_stream().context().bind_to_thread().ok()?;
    cudarc::driver::result::mem_get_info()
        .ok()
        .map(|(free, total)| (free as u64, total as u64))
}

pub fn vram_phase_marker(device: &Device, phase: &str, detail: &str) -> u64 {
    let gib = |b: u64| b as f64 / (1u64 << 30) as f64;
    match device_free_total_bytes(device) {
        Some((free, total)) => {
            eprintln!(
                "[laguna_vram_phase] phase={phase} device_used_gib={:.3} free_gib={:.3} {detail}",
                gib(total - free),
                gib(free)
            );
            total - free
        }
        None => {
            eprintln!("[laguna_vram_phase] phase={phase} device_used_gib=unavailable {detail}");
            0
        }
    }
}

fn handoff_cache_bytes(
    config: &crate::laguna::LagunaConfig,
    max_seq: usize,
    chunk: usize,
) -> Result<(u64, u64)> {
    use crate::gemma4::LayerType;
    ensure!(
        (1..=262144).contains(&max_seq)
            && matches!(chunk, 256 | 512 | 1024 | 2048 | 4096 | 8192)
            && config.num_key_value_heads == 8
            && config.head_dim == 128
            && config.layer_types.len() == config.num_hidden_layers,
        "BF16 handoff preflight requires valid KV8/head128 layers, capacity and chunk"
    );
    let product = |values: &[u64]| {
        values
            .iter()
            .try_fold(1u64, |p, v| p.checked_mul(*v))
            .context("BF16 handoff cache byte multiplication overflow")
    };
    let sum = |values: &[u64]| {
        values
            .iter()
            .try_fold(0u64, |p, v| p.checked_add(*v))
            .context("BF16 handoff cache byte addition overflow")
    };
    let (h, d, n) = (
        config.num_key_value_heads as u64,
        config.head_dim as u64,
        max_seq as u64,
    );
    let full = config
        .layer_types
        .iter()
        .filter(|t| matches!(t, LayerType::FullAttention))
        .count() as u64;
    let sliding = config.layer_types.len() as u64 - full;
    let slots = n.min(sum(&[config.sliding_window.max(1) as u64, chunk as u64])?);
    let max_q = config
        .num_attention_heads_per_layer
        .as_ref()
        .and_then(|v| v.iter().copied().max())
        .unwrap_or(0)
        .max(config.num_attention_heads) as u64;
    let bf16 = product(&[
        4,
        h,
        d,
        sum(&[product(&[full, n])?, product(&[sliding, slots])?])?,
    ])?;
    let fp8 = sum(&[
        product(&[full, n, sum(&[product(&[2, h, d])?, product(&[8, h])?])?])?,
        product(&[4, sliding, slots, h, d])?,
        product(&[4, max_q, n])?,
    ])?;
    Ok((bf16, fp8))
}

fn preflight_handoff_caches(
    device: &Device,
    config: &crate::laguna::LagunaConfig,
    max_seq: usize,
    chunk: usize,
) -> Result<()> {
    let (source, destination) = handoff_cache_bytes(config, max_seq, chunk)?;
    let gib = |b: u64| b as f64 / (1u64 << 30) as f64;
    let reserve = (HANDOFF_PREFLIGHT_RESERVE_GIB_2_KEEPS_GRAPH_CAPTURE_AND_PREFILL_SCRATCH_ROOM_AFTER_BOTH_CACHES * (1u64 << 30) as f64) as u64;
    let required = source
        .checked_add(destination)
        .and_then(|n| n.checked_add(reserve))
        .context("BF16 handoff cache and reserve total overflow")?;
    let (free, _) = device_free_total_bytes(device).context(
        "laguna spec serving: the handoff preflight needs cuMemGetInfo on the worker thread",
    )?;
    eprintln!(
        "[laguna_vram_preflight] route=bf16-handoff source_bf16_gib={:.3} destination_e4m3_gib={:.3} reserve_gib={:.1} free_gib={:.3} cap={max_seq} chunk={chunk}",
        gib(source), gib(destination), gib(reserve), gib(free)
    );
    ensure!(
        free >= required,
        "laguna spec serving: refusing the explicitly selected BF16 handoff route before allocating: source {:.2} GiB + destination {:.2} GiB + reserve {:.1} GiB exceed the {:.2} GiB free after the target loaded; no silent fallback",
        gib(source), gib(destination), gib(reserve), gib(free)
    );
    Ok(())
}

impl<'m> SpecState<'m> {
    fn new(
        target: &'m Arc<Laguna>,
        draft: Option<&'m LagunaDflash>,
        num_spec: usize,
        max_seq: usize,
    ) -> Result<Self> {
        let device_draft_transport = device_draft_transport_from_env()?;
        let device = target.device().clone();
        ensure!(
            device.is_cuda(),
            "laguna spec serving requires a CUDA device"
        );
        validate_bf16_prefill_handoff_selection()?;
        let handoff = bf16_prefill_handoff_opt_in();
        vram_phase_marker(
            &device,
            "target-and-draft-loaded",
            &format!("draft={}", draft.is_some()),
        );
        if handoff {
            preflight_handoff_caches(
                &device,
                target.config(),
                max_seq,
                target.prefill_chunk_tokens(),
            )?;
        }
        let step = if handoff {
            ensure!(!nv_config::env_arm_present_at_all_reread_every_call("NV_LAGUNA_FP8_KV"),
                "NV_LAGUNA_PREFILL_BF16_HANDOFF=1 builds its own typed FixedOne destination; unset NV_LAGUNA_FP8_KV");
            ensure!(!nv_config::env_arm_present_at_all_reread_every_call("NV_LAGUNA_FP8_UNIT_SCALE"),
                "NV_LAGUNA_PREFILL_BF16_HANDOFF=1 constructs the FixedOne destination explicitly and the BF16 source refuses NV_LAGUNA_FP8_UNIT_SCALE; unset it");
            let chunk = target.prefill_chunk_tokens();
            let destination = crate::laguna_fp8::LagunaKvCacheFp8::new_fixed_one_with_prefill_chunk(target.config(), max_seq, &device, chunk)
                .context("laguna spec serving: FixedOne destination for the BF16 prefill handoff")?;
            LagunaStepGraph::new_fp8(Arc::clone(target), destination)
        } else if nv_config::env_arm_present_at_all_reread_every_call("NV_LAGUNA_FP8_KV") {
            LagunaStepGraph::new_fp8(Arc::clone(target), target.new_kv_cache_fp8(max_seq)?)
        } else {
            LagunaStepGraph::new(Arc::clone(target), target.new_kv_cache(max_seq)?)
        }
        .context("laguna spec serving: step graph init")?;
        let handoff_source = if handoff {
            let chunk = target.prefill_chunk_tokens();
            let source = crate::laguna::LagunaKvCache::new_with_prefill_chunk(
                target.config(),
                max_seq,
                &device,
                target.dtype(),
                chunk,
            )
            .context("laguna spec serving: BF16 source cache for the prefill handoff")?;
            eprintln!("[laguna_serve] bf16-handoff route: source=bf16-ring-cache chunk={chunk} cap={max_seq} destination=fixed-one-e4m3 (typed constructor, scale_policy={:?}) prefill_flashinfer={} (the request prefills the BF16 source with the packed target, converts once on the GPU into the reset destination, then decodes on the captured route; conversion is inside TTFT)",
                match step.kv() { StepCache::Fp8(c) => c.scale_policy(), StepCache::Bf16(_) => bail!("handoff destination is not FP8") },
                nv_config::env_arm_opt_in_eq_some_1_reread_every_call("NV_LAGUNA_PREFILL_FLASHINFER"));
            Some(source)
        } else {
            None
        };
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
        vram_phase_marker(
            &device,
            "worker-caches-allocated",
            &format!("handoff={handoff} cap={max_seq}"),
        );
        let state = Self {
            target: target.as_ref(),
            draft,
            probe: RetirementProbe::new(&device)?,
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
            device_draft_transport,
            handoff_source,
            prefix: prefix::PrefixState::from_env(),
        };
        state.prefix_worker_started();
        Ok(state)
    }

    fn warmup(&mut self) -> Result<()> {
        let ids: Vec<u32> = (0..48u32).map(|i| 100 + i).collect();
        let max_new = 2 * (self.num_spec + 2);
        let runs = if self.draft.is_some() { 2 } else { 1 };
        for run in 0..runs {
            let mut job = SpecServeJob {
                prompt_ids: ids.clone(),
                max_new,
                eos_ids: Vec::new(),
                emit: Box::new(|_| true),
            };
            self.serve_one(&mut job)
                .with_context(|| format!("laguna spec serving warmup run {run}"))?;
        }
        self.device.synchronize().ok();
        Ok(())
    }

    fn serve_one_inner(&mut self, job: &mut SpecServeJob) -> Result<()> {
        if !job.client_is_live() {
            return Ok(());
        }
        ensure!(!job.prompt_ids.is_empty(), "spec serve: empty prompt");
        let n = job.prompt_ids.len();
        ensure!(
            n + 2 <= self.max_seq,
            "spec serve: prompt ({n}) exceeds KV capacity ({})",
            self.max_seq
        );
        if let Some(draft) = self.draft {
            draft.check_rope_span(0, n)?;
        }
        let result = self.serve_validated_job(job);
        job.report_execution_result(result)
    }

    fn serve_validated_job(&mut self, job: &mut SpecServeJob) -> Result<()> {
        let n = job.prompt_ids.len();
        if self.handoff_source.is_some() {
            ensure!(
                !crate::laguna_prefix::prefix_reuse_enabled(),
                "engine busy: BF16 prefill handoff requires prefix reuse disabled; try later."
            );
        }
        let resume = self.prefix_prepare(job)?;
        if let Some(source) = self.handoff_source.as_mut() {
            ensure!(
                resume == 0,
                "BF16 prefill handoff cannot resume an FP8 checkpoint"
            );
            source.reset();
        }
        if let Some(l) = self.lookup.as_mut() {
            l.reset();
            l.extend_slice(&job.prompt_ids);
        }

        let mut last_logits: Option<Tensor> = None;
        let mut offset = resume;
        while offset < n {
            if !job.client_is_live() {
                return Ok(());
            }
            self.prefix_checkpoint_hook(job, offset)
                .context("engine busy: Laguna prefix checkpoint storage failed; try later.")?;
            let m = self.target.next_prefill_chunk_len(n - offset);
            let toks = Tensor::from_vec(
                job.prompt_ids[offset..offset + m].to_vec(),
                (1usize, m),
                &self.device,
            )?;
            let pos: Vec<i32> = (offset as i32..(offset + m) as i32).collect();
            let pos_t = Tensor::from_vec(pos.clone(), m, &self.device)?;
            let (logits, aux) = if let Some(src) = self.handoff_source.as_mut() {
                self.target.prefill_with_cache_aux_scoped(
                    &toks,
                    &pos_t,
                    src,
                    &self.aux_layers,
                    self.draft.is_some(),
                    offset + m == n,
                )?
            } else {
                match self.step.kv_mut() {
                    StepCache::Bf16(c) => self.target.prefill_with_cache_aux_scoped(
                        &toks,
                        &pos_t,
                        c,
                        &self.aux_layers,
                        self.draft.is_some(),
                        offset + m == n,
                    )?,
                    StepCache::Fp8(c) => self.target.prefill_with_cache_aux_scoped(
                        &toks,
                        &pos_t,
                        c,
                        &self.aux_layers,
                        self.draft.is_some(),
                        offset + m == n,
                    )?,
                }
            };
            if let (Some(d), Some(ctx)) = (self.draft, self.ctx.as_mut()) {
                let combined = d.combine_aux(&aux)?;
                d.append_context(ctx, &combined, &pos)?;
            }
            last_logits = logits;
            offset += m;
        }
        if let Some(src) = self.handoff_source.as_mut() {
            let t_handoff = std::time::Instant::now();
            match self.step.kv_mut() {
                StepCache::Fp8(dst) => dst
                    .copy_from_bf16(src)
                    .context("bf16 prefill handoff into the FixedOne destination")?,
                StepCache::Bf16(_) => {
                    bail!("bf16 prefill handoff needs the FixedOne destination")
                }
            }
            eprintln!("[laguna_serve] bf16-handoff prompt_tokens={n} converted_rows={} destination_len={} convert_ms={:.1}", crate::gemma4::Gemma4Cache::current_len(src), self.step.kv().current_len(), t_handoff.elapsed().as_secs_f64() * 1e3);
        }
        let logits = last_logits.expect("non-empty prompt");
        let m_last = logits.dims()[1];
        let last_row: Vec<f32> = logits.narrow(1, m_last - 1, 1)?.flatten_all()?.to_vec1()?;
        let anchor = checked_target_argmax_row(&last_row)
            .context("engine busy: Laguna prefill target logits are invalid; try later.")?;
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
            if !job.client_is_live() {
                return Ok(());
            }
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
        let stats_on = nv_config::env_arm_present_at_all_reread_every_call("NV_LAGUNA_SERVE_STATS");
        let prompt_n = num_ctx;
        let mut rounds = 0usize;
        let mut acc_total = 0usize;
        let mut pos0 = 0usize;
        let mut lookup_rounds = 0usize;
        let mut lookup_accepted = 0usize;
        let mut graph_propose_rounds = 0usize;
        let mut eager_propose_rounds = 0usize;
        let mut device_transport_rounds = 0usize;
        let device_transport_allowed = self.device_draft_transport && {
            let packed = draft.quantization_summary();
            packed.weight_format == "nvfp4-derived"
                && packed.packed_linears > 0
                && packed.bf16_fallback_linears == 0
        };
        let transport_requested = self.device_draft_transport;
        let t0 = std::time::Instant::now();
        let log_stats = |rounds: usize,
                         acc_total: usize,
                         pos0: usize,
                         total: usize,
                         dt: f64,
                         lookup_rounds: usize,
                         lookup_accepted: usize,
                         graph_propose_rounds: usize,
                         eager_propose_rounds: usize,
                         device_transport_rounds: usize| {
            if transport_requested {
                eprintln!("[laguna_dflash_transport] prompt_tokens={prompt_n} requested=1 device_rounds={device_transport_rounds} host_rounds={} completed_rounds={rounds} packed_eligible={device_transport_allowed} basis=actual_completed_transport", rounds.saturating_sub(device_transport_rounds));
            }
            if stats_on && rounds > 0 {
                let tau = (total.saturating_sub(1)) as f64 / rounds as f64;
                eprintln!(
                    "[laguna_serve_stats] prompt_tokens={prompt_n} k={k} rounds={rounds} emitted={total} tau={tau:.3} pos0={:.3} accepted={acc_total} lookup_rounds={lookup_rounds} lookup_accepted={lookup_accepted} graph_propose_rounds={graph_propose_rounds} eager_propose_rounds={eager_propose_rounds} decode_ms_tok={:.3}",
                    pos0 as f64 / rounds as f64,
                    1000.0 * dt / (total.saturating_sub(1).max(1)) as f64
                );
            }
        };
        loop {
            if !job.client_is_live() {
                return Ok(());
            }
            if num_ctx + k + 1 > self.max_seq {
                log_stats(
                    rounds,
                    acc_total,
                    pos0,
                    *total,
                    t0.elapsed().as_secs_f64(),
                    lookup_rounds,
                    lookup_accepted,
                    graph_propose_rounds,
                    eager_propose_rounds,
                    device_transport_rounds,
                );
                return Ok(());
            }
            let lookup_prop = self.lookup.as_ref().and_then(|l| l.propose(k));
            let from_lookup = lookup_prop.is_some();
            let ctx = self.ctx.as_ref().expect("spec ctx");
            let device_eligible = device_transport_allowed
                && !from_lookup
                && !adapt
                && !vegas_enabled()
                && self.verify_graphs.contains_key(&(k + 1));
            let (mut drafts, conf, used_graph, device_transport) = match lookup_prop {
                Some(d) => (d, None, false, false),
                None => propose_round(
                    draft,
                    ctx,
                    &mut self.proposers,
                    &mut self.proposer_init_budget,
                    self.target,
                    anchor,
                    num_ctx,
                    k,
                    device_eligible,
                )?,
            };
            if !from_lookup {
                if used_graph {
                    graph_propose_rounds += 1;
                } else {
                    eager_propose_rounds += 1;
                }
                crate::laguna_prefix_diagnostics::draft_round(used_graph);
            }
            if adapt {
                if let Some(c) = conf.as_deref() {
                    drafts.truncate(adapt_truncate_len(c, adapt_thresh));
                }
            }
            let mut block: Vec<u32> = Vec::with_capacity(drafts.len() + 1);
            block.push(anchor);
            block.extend_from_slice(&drafts);

            let (vlogits, vaux) = if device_transport {
                let key = draft.active_rope_theta().to_bits();
                let proposer = self
                    .proposers
                    .get(&key)
                    .context("device draft proposer disappeared before verification")?;
                verify_device_block(
                    &mut self.step,
                    &mut self.verify_graphs,
                    anchor,
                    proposer.out_buf(),
                    block.len(),
                )?
            } else {
                verify_block(
                    self.target,
                    &mut self.step,
                    &mut self.verify_graphs,
                    &self.aux_layers,
                    &block,
                    num_ctx,
                    &self.device,
                )?
            };
            let xqa_status_device = self
                .verify_graphs
                .get(&block.len())
                .and_then(LagunaVerifyGraph::accept_status_device);
            let device_drafts = if device_transport {
                let graph = self
                    .verify_graphs
                    .get(&block.len())
                    .context("device verifier disappeared before acceptance")?;
                let Device::Cuda(dev) = &self.device else {
                    bail!("device draft transport requires CUDA")
                };
                let (ptr, count) = graph.drafts_device_ptr(&dev.cuda_stream());
                ensure!(
                    count == drafts.len(),
                    "device draft count changed before acceptance"
                );
                Some(ptr)
            } else {
                None
            };
            let (accepted, emitted) = match accept_block_on_device_with_status(
                self.target,
                &mut self.accept_slots,
                &vlogits,
                &drafts,
                device_drafts,
                xqa_status_device,
            ) {
                Ok(r) => r,
                Err(e) if device_transport => {
                    return Err(
                        e.context("engine busy: device-draft acceptance failed; try later.")
                    );
                }
                Err(e) if xqa_status_device.is_some() => {
                    return Err(e.context("engine busy: XQA acceptance failed; try later."));
                }
                Err(e) => {
                    eprintln!("[laguna_serve] device accept failed, host fallback: {e:#}");
                    accept_block_on_host(&vlogits, &drafts)?
                }
            };
            let consumed = 1 + accepted;
            self.step.kv_mut().rollback(block.len() - consumed)?;

            let ctx = self.ctx.as_mut().expect("spec ctx");
            let mut vaux_kept = Vec::with_capacity(vaux.len());
            for a in &vaux {
                vaux_kept.push(a.narrow(1, 0, consumed)?);
            }
            let combined = draft.combine_aux(&vaux_kept)?;
            let new_pos: Vec<i32> = (0..consumed).map(|i| (num_ctx + i) as i32).collect();
            draft.append_context(ctx, &combined, &new_pos)?;

            num_ctx += consumed;
            anchor = *emitted.last().expect("non-empty accept output");
            rounds += 1;
            if device_transport {
                device_transport_rounds += 1;
                if device_transport_rounds == 1 {
                    eprintln!("[laguna_dflash_transport] route=device-drafts prompt_tokens={prompt_n} completed_round={rounds} k={k} verifier_rows={} target_kv_fp8={} xqa_status_checked={} proposal_ids_dtoh=0 verify_drafts_htod=0 accept_drafts_htod=0", block.len(), self.step.kv().is_fp8(), xqa_status_device.is_some());
                }
            }
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
                    graph_propose_rounds,
                    eager_propose_rounds,
                    device_transport_rounds,
                );
                return Ok(());
            }
        }
    }
}

fn propose_round(
    draft: &LagunaDflash,
    ctx: &DflashCtxCache,
    proposers: &mut HashMap<u32, DflashGraphProposer>,
    init_budget: &mut ProposerInitBudget,
    target: &Laguna,
    anchor: u32,
    num_ctx: usize,
    k: usize,
    device_transport: bool,
) -> Result<(Vec<u32>, Option<Vec<f32>>, bool, bool)> {
    let graph_ok = ctx.has_ring() && draft.check_rope_span(num_ctx, k + 1).is_ok();
    if graph_ok {
        let key = draft.active_rope_theta().to_bits();
        if let std::collections::hash_map::Entry::Vacant(entry) = proposers.entry(key) {
            if let Some(attempt) = init_budget.admit(key) {
                match DflashGraphProposer::new(draft, k) {
                    Ok(p) => {
                        entry.insert(p);
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
            let proposed = if device_transport {
                p.propose_device(draft, ctx, anchor, num_ctx, target, target.lm_head())
                    .and_then(|n| {
                        ensure!(n == k, "device proposer returned {n} drafts, expected {k}");
                        Ok(vec![0u32; n])
                    })
            } else {
                p.propose(draft, ctx, anchor, num_ctx, target, target.lm_head())
            };
            match proposed {
                Ok(v) => {
                    let c = if device_transport {
                        None
                    } else {
                        p.last_conf().map(|s| s.to_vec())
                    };
                    return Ok((v, c, true, device_transport));
                }
                Err(e) => eprintln!("[laguna_serve] graph propose failed, eager: {e:#}"),
            }
        }
    }
    if dflash_adapt_enabled() {
        let (d, c) = draft.propose_k_conf(ctx, anchor, num_ctx, k, target, target.lm_head())?;
        Ok((d, Some(c), false, false))
    } else {
        Ok((
            draft.propose_k(ctx, anchor, num_ctx, k, target, target.lm_head())?,
            None,
            false,
            false,
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn proposer_init_budget_admits_the_cap_then_permanently_refuses_counting_each_rope_theta_key_separately() {
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
        let mut b = ProposerInitBudget::new();
        assert_eq!(b.admit(1), Some(1));
        assert_eq!(b.admit(2), Some(1));
        assert_eq!(b.admit(1), Some(2));
        assert_eq!(b.admit(1), None);
        assert_eq!(b.admit(2), Some(2));
        assert_eq!(b.admit(2), None);

    }

}

const VERIFY_FAILED_AFTER_CACHE_PREPARE_TERMINATES_THE_REQUEST_BECAUSE_AN_EAGER_RETRY_WOULD_RESERVE_THE_BLOCK_TWICE: &str =
    "verify failed after the KV cache was prepared for the block; the request is terminated and the next request starts from fresh state";

fn new_verify_graph<'m>(
    target: &'m Laguna,
    cache: &StepCache,
    bs: usize,
    aux_layers: &[usize],
) -> Result<LagunaVerifyGraph<'m>> {
    match cache {
        StepCache::Bf16(c) => LagunaVerifyGraph::new(target, c, bs, aux_layers),
        StepCache::Fp8(c) => LagunaVerifyGraph::new_fp8(target, c, bs, aux_layers),
    }
}

fn verify_with(
    g: &mut LagunaVerifyGraph<'_>,
    cache: &mut StepCache,
    block: &[u32],
) -> Result<(Tensor, Vec<Tensor>)> {
    match cache {
        StepCache::Bf16(c) => g.verify(c, block),
        StepCache::Fp8(c) => g.verify_fp8(c, block),
    }
}

fn verify_device_block<'m>(
    step: &mut LagunaStepGraph,
    graphs: &mut HashMap<usize, LagunaVerifyGraph<'m>>,
    anchor: u32,
    drafts: &nv_layers::cudarc::driver::CudaSlice<u32>,
    rows: usize,
) -> Result<(Tensor, Vec<Tensor>)> {
    let graph = graphs
        .get_mut(&rows)
        .context("device draft transport requires an existing verifier bucket")?;
    let result = match step.kv_mut() {
        StepCache::Bf16(cache) => graph.verify_device(cache, anchor, drafts),
        StepCache::Fp8(cache) => graph.verify_fp8_device(cache, anchor, drafts),
    };
    result.map_err(|e| {
        graphs.remove(&rows);
        e.context(VERIFY_FAILED_AFTER_CACHE_PREPARE_TERMINATES_THE_REQUEST_BECAUSE_AN_EAGER_RETRY_WOULD_RESERVE_THE_BLOCK_TWICE)
    })
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
        return verify_with(g, step.kv_mut(), block).map_err(|e| {
            graphs.remove(&bs);
            e.context(VERIFY_FAILED_AFTER_CACHE_PREPARE_TERMINATES_THE_REQUEST_BECAUSE_AN_EAGER_RETRY_WOULD_RESERVE_THE_BLOCK_TWICE)
        });
    }
    match new_verify_graph(target, step.kv(), bs, aux_layers) {
        Ok(mut g) => {
            let r = verify_with(&mut g, step.kv_mut(), block)
                .context(VERIFY_FAILED_AFTER_CACHE_PREPARE_TERMINATES_THE_REQUEST_BECAUSE_AN_EAGER_RETRY_WOULD_RESERVE_THE_BLOCK_TWICE)?;
            graphs.insert(bs, g);
            return Ok(r);
        }
        Err(e) => eprintln!("[laguna_serve] verify graph init failed before touching the cache (bs={bs}), eager: {e:#}"),
    }
    let pos: Vec<i32> = (0..bs).map(|i| (num_ctx + i) as i32).collect();
    let bt = Tensor::from_vec(block.to_vec(), (1usize, bs), device)?;
    let bp = Tensor::from_vec(pos, bs, device)?;
    match step.kv_mut() {
        StepCache::Bf16(c) => target.forward_with_cache_aux_scoped(&bt, &bp, c, aux_layers, true),
        StepCache::Fp8(c) => target.forward_with_cache_aux_scoped(&bt, &bp, c, aux_layers, true),
    }
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
    batch_lanes_from_spec(env::var(LAGUNA_BATCH_ENV).ok().as_deref())
}

pub fn batch_graph_enabled_from_env() -> Result<bool> {
    match env::var("NV_LAGUNA_BATCH_GRAPH") {
        Ok(value) => batch_graph_enabled_from_spec(Some(&value)),
        Err(std::env::VarError::NotPresent) => Ok(false),
        Err(e) => Err(e).context("NV_LAGUNA_BATCH_GRAPH must be 0, 1, m2 or m2-spec"),
    }
}

fn batch_graph_enabled_from_spec(spec: Option<&str>) -> Result<bool> {
    match spec {
        None | Some("0") => Ok(false),
        Some("1") | Some("m2") | Some("m2-spec") => Ok(true),
        _ => bail!("NV_LAGUNA_BATCH_GRAPH must be 0, 1, m2 or m2-spec"),
    }
}

pub fn batch_m2_graph_enabled_from_env() -> bool {
    env::var("NV_LAGUNA_BATCH_GRAPH").as_deref() == Ok("m2")
}

fn validate_b2_prefill_selection(selector: &str) -> Result<()> {
    for key in ["NV_LAGUNA_FP8_KV", "NV_LAGUNA_FP8_UNIT_SCALE"] {
        if bf16_prefill_handoff_opt_in() {
            ensure!(env::var_os(key).is_none(),
                "{selector} with BF16 prefill handoff constructs typed FixedOne destinations; unset {key}");
        } else {
            ensure!(
                env::var(key).as_deref() == Ok("1"),
                "{selector} with direct FP8 prefill requires {key}=1"
            );
        }
    }
    ensure!(
        !crate::laguna_prefix::prefix_reuse_enabled(),
        "{selector} requires prefix reuse disabled"
    );
    Ok(())
}

pub fn validate_m2_graph_selection() -> Result<()> {
    if !batch_m2_graph_enabled_from_env() {
        return Ok(());
    }
    for (key, expected) in [
        ("NV_LAGUNA_BATCH", "2"),
        ("NV_LAGUNA_BATCH_INTERLEAVE_PREFILL", "1"),
        ("NV_LAGUNA_XQA_FP8", "1"),
        ("NV_LAGUNA_PREFILL_FLASHINFER", "1"),
        ("NV_LAGUNA_BATCH_MOE_FUSED", "1"),
        ("NV_LAGUNA_SERVE_DRAFT", "0"),
    ] {
        ensure!(
            env::var(key).as_deref() == Ok(expected),
            "NV_LAGUNA_BATCH_GRAPH=m2 requires {key}={expected}"
        );
    }
    validate_b2_prefill_selection("NV_LAGUNA_BATCH_GRAPH=m2")?;
    for key in ["NV_LAGUNA_RING_ATTN", "NV_LAGUNA_FP8_BF16_FULL"] {
        ensure!(
            env::var_os(key).is_none(),
            "NV_LAGUNA_BATCH_GRAPH=m2 requires {key} unset"
        );
    }
    Ok(())
}

pub fn batch_m2_spec_enabled_from_env() -> bool {
    env::var("NV_LAGUNA_BATCH_GRAPH").as_deref() == Ok("m2-spec")
}

pub fn validate_m2_spec_selection() -> Result<()> {
    if !batch_m2_spec_enabled_from_env() {
        return Ok(());
    }
    for (key, expected) in [
        ("NV_LAGUNA_BATCH", "2"),
        ("NV_LAGUNA_BATCH_INTERLEAVE_PREFILL", "1"),
        ("NV_LAGUNA_XQA_FP8", "1"),
        ("NV_LAGUNA_PREFILL_FLASHINFER", "1"),
        ("NV_LAGUNA_BATCH_MOE_FUSED", "1"),
        ("NV_LAGUNA_SERVE_DRAFT", "1"),
        ("NV_LAGUNA_DFLASH_NVFP4", "1"),
        ("NV_DFLASH_K", "2"),
    ] {
        ensure!(
            env::var(key).as_deref() == Ok(expected),
            "NV_LAGUNA_BATCH_GRAPH=m2-spec requires {key}={expected}"
        );
    }
    validate_b2_prefill_selection("NV_LAGUNA_BATCH_GRAPH=m2-spec")?;
    ensure!(
        !dflash_adapt_enabled()
            && !vegas_enabled()
            && !crate::laguna_dflash::lookup_draft_enabled()
            && crate::laguna_dflash::dflash_graph_enabled(),
        "m2-spec requires graph drafting, no adaptive K, lookup or Vegas"
    );
    ensure!(
        !device_draft_transport_from_env()?,
        "m2-spec uses host draft IDs; device transport is not composed"
    );
    for key in ["NV_LAGUNA_RING_ATTN", "NV_LAGUNA_FP8_BF16_FULL"] {
        ensure!(env::var_os(key).is_none(), "m2-spec requires {key} unset");
    }
    Ok(())
}

fn batch_spec_rows(position: usize, capacity: usize) -> Result<usize> {
    let remaining = capacity
        .checked_sub(position)
        .context("B2 position exceeds capacity")?;
    ensure!(remaining > 0, "B2 cache has no remaining slot");
    Ok(if remaining >= 3 { 3 } else { 1 })
}

fn b2_draft_context_bytes(cfg: &LagunaDflashConfig) -> Result<u64> {
    let row = (cfg.num_key_value_heads as u64)
        .checked_mul(cfg.head_dim as u64)
        .and_then(|n| n.checked_mul(2))
        .context("draft ring row bytes overflow")?;
    let window = cfg.sliding_window.max(1) as u64;
    let cap = window
        .checked_add(cfg.dflash_config.block_size as u64)
        .context("draft ring capacity overflow")?;
    (cfg.num_hidden_layers as u64)
        .checked_mul(2)
        .and_then(|n| n.checked_mul(cap))
        .and_then(|n| n.checked_add(window))
        .and_then(|n| n.checked_mul(row))
        .context("draft ring bytes overflow")
}

struct BatchSpecLane {
    commit: DflashCtxCommitGraph,
    proposer: DflashGraphProposer,
    ctx: DflashCtxCache,
    accept: HashMap<usize, AcceptSlots>,
    rounds: usize,
    accepted: usize,
}

struct BatchSpecState {
    lanes: [BatchSpecLane; 2],
    draft: LagunaDflash,
    aux_layers: Vec<usize>,
    theta: u32,
}

impl BatchM1State {
    pub(crate) fn new_spec(
        target: &Arc<Laguna>,
        draft: LagunaDflash,
        k: usize,
        max_seq: usize,
    ) -> Result<Self> {
        validate_bf16_prefill_handoff_selection()?;
        validate_m2_spec_selection()?;
        validate_draft(target, &draft, k)?;
        ensure!(
            k == 2 && max_seq > 0 && max_seq <= draft.rope_max_seq_len(),
            "m2-spec requires K2 and draft rotary tables covering target capacity"
        );
        let packed = draft.quantization_summary();
        ensure!(
            packed.weight_format == "nvfp4-derived"
                && packed.packed_linears == 36
                && packed.excluded_linears == 5
                && packed.bf16_fallback_linears == 0,
            "m2-spec requires packed target-trained draft 36/5/0: {packed:?}"
        );
        let cfg = draft.config();
        ensure!(
            cfg.hidden_size == target.config().hidden_size
                && cfg.vocab_size == target.config().vocab_size
                && cfg.num_hidden_layers == 5
                && cfg.num_key_value_heads == 8
                && cfg.head_dim == 128
                && cfg.sliding_window == 512
                && cfg.dflash_config.block_size == 16,
            "m2-spec draft geometry differs from the reviewed five-layer window512/block16 model"
        );
        let aux_layers = resolve_tap_layers(cfg, tap_list_mode());
        let (bf16, single) =
            handoff_cache_bytes(target.config(), max_seq, target.prefill_chunk_tokens())?;
        let source = if bf16_prefill_handoff_opt_in() {
            bf16
        } else {
            0
        };
        let destinations = single
            .checked_mul(2)
            .context("B2 spec cache bytes overflow")?;
        let rings = b2_draft_context_bytes(cfg)?
            .checked_mul(2)
            .context("B2 spec draft bytes overflow")?;
        let outputs = (target.config().vocab_size as u64)
            .checked_mul(8 * 4)
            .and_then(|n| {
                n.checked_add((target.config().hidden_size as u64).checked_mul(6 * 5 * 2)?)
            })
            .context("B2 spec output bytes overflow")?;
        let commit_staging = DflashCtxCommitGraph::retained_bytes(cfg.hidden_size)?
            .checked_mul(2)
            .context("B2 spec commit staging bytes overflow")?;
        let reserve = 2u64 << 30;
        let needed = destinations
            .checked_add(source)
            .and_then(|n| n.checked_add(rings))
            .and_then(|n| n.checked_add(outputs))
            .and_then(|n| n.checked_add(commit_staging))
            .and_then(|n| n.checked_add(2 * 798_816))
            .and_then(|n| n.checked_add(reserve))
            .context("B2 spec preflight overflow")?;
        let (free, _) = device_free_total_bytes(target.device())
            .context("B2 spec preflight requires a bound CUDA context")?;
        eprintln!("[laguna_vram_preflight] route=prepared-b2-spec-k2 physical_lanes=2 destination_bytes={destinations} draft_ring_bytes={rings} output_bytes={outputs} reserve_bytes={reserve} reserve_is_estimate=true free_bytes={free} capacity_per_lane={max_seq} source_bf16_bytes={source} context_commit_staging_bytes={commit_staging} draft_weights_already_loaded=true");
        ensure!(free >= needed, "engine busy: B2 spec allocation estimate {needed} exceeds free CUDA memory {free}; try later.");
        let batched = LagunaBatchStepGraph::new_spec(target.clone(), max_seq, &aux_layers)?;
        let handoff_source = BatchPrefillSource::new(target, max_seq)?;
        let make = || -> Result<BatchSpecLane> {
            let ctx = draft.new_ctx_cache();
            ensure!(
                ctx.has_ring(),
                "required B2 draft ring allocation failed; refusing legacy context growth"
            );
            let commit = DflashCtxCommitGraph::new(&draft, &ctx)?;
            Ok(BatchSpecLane {
                commit,
                proposer: DflashGraphProposer::new(&draft, 2)?,
                ctx,
                accept: HashMap::new(),
                rounds: 0,
                accepted: 0,
            })
        };
        let lanes = [make()?, make()?];
        let theta = draft.active_rope_theta().to_bits();
        Ok(Self {
            pending_retirement: None,
            probe: RetirementProbe::new(target.device())?,
            spec: Some(BatchSpecState {
                lanes,
                draft,
                aux_layers,
                theta,
            }),
            target: target.clone(),
            device: target.device().clone(),
            max_seq,
            lanes: Vec::new(),
            batched: Some(batched),
            handoff_source,
            interleave_prefill: true,
        })
    }

    fn warm_spec_masks(&mut self) -> Result<()> {
        let ids: Vec<u32> = (100..148).collect();
        for lane in 0..2 {
            self.reset_lane(lane)?;
            let job = SpecServeJob {
                prompt_ids: ids.clone(),
                max_new: 32,
                eos_ids: Vec::new(),
                emit: Box::new(|_| true),
            };
            let mut offset = 0;
            while offset < ids.len() {
                self.prefill_lane_chunk(lane, &job, &mut offset)?;
            }
        }
        for rows in [3usize, 1] {
            for active in [&[0usize][..], &[1usize][..], &[0usize, 1usize][..]] {
                for _ in 0..2 {
                    let graph = self.batched.as_mut().context("B2 warmup graph missing")?;
                    let positions: Vec<usize> =
                        active.iter().map(|&i| graph.current_lengths()[i]).collect();
                    let (logits, aux) =
                        graph.verify(active, &vec![100; active.len() * rows], &positions, rows)?;
                    checked_batch_target_tokens(
                        &logits.flatten_all()?.to_vec1::<f32>()?,
                        active.len() * rows,
                    )?;
                    let spec = self.spec.as_mut().context("B2 warmup draft missing")?;
                    for (slot, &lane) in active.iter().enumerate() {
                        let lane_aux = aux
                            .iter()
                            .map(|a| a.narrow(1, slot * rows, rows))
                            .collect::<candle_core::Result<Vec<_>>>()?;
                        let combined = spec.draft.combine_aux(&lane_aux)?;
                        let pos = (positions[slot]..positions[slot] + rows)
                            .map(i32::try_from)
                            .collect::<std::result::Result<Vec<_>, _>>()?;
                        spec.draft
                            .append_context(&mut spec.lanes[lane].ctx, &combined, &pos)?;
                    }
                    self.device.synchronize()?;
                }
            }
        }
        self.batched
            .as_ref()
            .unwrap()
            .report_spec("all-masks-warmed");
        for lane in 0..2 {
            self.reset_lane(lane)?;
        }
        Ok(())
    }

    fn decode_spec_step(&mut self, slots: &mut [Option<LaneRun>]) -> Result<()> {
        for rows in [3usize, 1] {
            let mut active = Vec::with_capacity(2);
            for (lane, run) in slots.iter().enumerate() {
                if let Some(run) = run {
                    if batch_spec_rows(run.pos, self.max_seq)? == rows {
                        active.push(lane);
                    }
                }
            }
            if active.is_empty() {
                continue;
            }
            let spec = self.spec.as_mut().context("B2 draft state disappeared")?;
            ensure!(
                spec.theta == spec.draft.active_rope_theta().to_bits(),
                "B2 draft theta changed after capture"
            );
            let mut blocks = Vec::with_capacity(active.len() * rows);
            let mut proposals = Vec::with_capacity(active.len());
            let positions: Vec<usize> = active
                .iter()
                .map(|&i| slots[i].as_ref().unwrap().pos)
                .collect();
            for &lane in &active {
                let r = slots[lane].as_ref().context("B2 active lane disappeared")?;
                let state = &mut spec.lanes[lane];
                ensure!(
                    state.ctx.len() == r.pos,
                    "B2 lane {lane} draft/target frontier differs"
                );
                let draft_ids = if rows == 3 {
                    state.proposer.propose(
                        &spec.draft,
                        &state.ctx,
                        r.last,
                        r.pos,
                        self.target.as_ref(),
                        self.target.lm_head(),
                    )?
                } else {
                    Vec::new()
                };
                ensure!(
                    draft_ids.len() + 1 == rows,
                    "B2 proposer returned a different K"
                );
                blocks.push(r.last);
                blocks.extend_from_slice(&draft_ids);
                proposals.push(draft_ids);
            }
            let graph = self
                .batched
                .as_mut()
                .context("B2 target graph disappeared")?;
            let (logits, aux) = graph.verify(&active, &blocks, &positions, rows)?;
            let mut accepted = Vec::with_capacity(active.len());
            for (slot, &lane) in active.iter().enumerate() {
                let lane_logits = logits.narrow(1, slot * rows, rows)?;
                let answer = if rows == 3 {
                    accept_block_on_device_with_status(
                        &self.target,
                        &mut spec.lanes[lane].accept,
                        &lane_logits,
                        &proposals[slot],
                        None,
                        None,
                    )?
                } else {
                    let values = lane_logits.flatten_all()?.to_vec1::<f32>()?;
                    (
                        0,
                        vec![checked_target_argmax_row(&values).context("invalid B2 tail logits")?],
                    )
                };
                ensure!(
                    answer.0 < rows && answer.1.len() == answer.0 + 1,
                    "B2 acceptance output shape is invalid"
                );
                accepted.push(answer);
            }
            for (slot, &lane) in active.iter().enumerate() {
                let consumed = 1 + accepted[slot].0;
                graph.cache_mut(lane)?.rollback(rows - consumed)?;
                let kept = aux
                    .iter()
                    .map(|a| a.narrow(1, slot * rows, consumed))
                    .collect::<candle_core::Result<Vec<_>>>()?;
                let mut pos = [0i32; 3];
                for (i, p) in pos[..consumed].iter_mut().enumerate() {
                    *p = i32::try_from(
                        positions[slot]
                            .checked_add(i)
                            .context("B2 commit position overflow")?,
                    )?;
                }
                let state = &mut spec.lanes[lane];
                state
                    .commit
                    .enqueue(&spec.draft, &state.ctx, &kept, &pos[..consumed])?;
            }
            self.device.synchronize()?;
            for &lane in &active {
                let state = &mut spec.lanes[lane];
                state.commit.finish_after_sync(&mut state.ctx)?;
                ensure!(
                    state.ctx.len() == graph.current_lengths()[lane],
                    "B2 lane {lane} committed draft/target frontier differs"
                );
            }
            for (slot, &lane) in active.iter().enumerate() {
                let (a, emitted) = &accepted[slot];
                let r = slots[lane]
                    .as_mut()
                    .context("B2 active lane disappeared before emission")?;
                r.pos = r
                    .pos
                    .checked_add(1 + *a)
                    .context("B2 committed frontier overflow")?;
                r.last = *emitted.last().context("empty B2 acceptance")?;
                spec.lanes[lane].rounds += usize::from(rows == 3);
                spec.lanes[lane].accepted += *a;
            }
            for (slot, &lane) in active.iter().enumerate() {
                let r = slots[lane]
                    .as_mut()
                    .context("B2 active lane disappeared before output")?;
                if let Flow::Stop = push_round(&mut r.job, &accepted[slot].1, &mut r.total) {
                    let mut r = slots[lane].take().unwrap();
                    eprintln!("[laguna_b2_spec_stats] lane={lane} prompt_tokens={} emitted={} committed={} k=2 rounds={} accepted={} graph_propose_rounds={} eager_propose_rounds=0 target_batch_max=2 target_rows_max=6 capacity={} proposal_transport=host",
                        r.job.prompt_ids.len(), r.total, r.pos, spec.lanes[lane].rounds,
                        spec.lanes[lane].accepted, spec.lanes[lane].rounds, self.max_seq);
                    spec.lanes[lane].commit.report("request-complete");
                    let _ = (r.job.emit)(SpecServeEvent::Done);
                }
            }
        }
        Ok(())
    }
}

fn preflight_b2_caches(target: &Laguna, max_seq: usize) -> Result<()> {
    let (bf16, single) =
        handoff_cache_bytes(target.config(), max_seq, target.prefill_chunk_tokens())?;
    let source = if bf16_prefill_handoff_opt_in() {
        bf16
    } else {
        0
    };
    let destinations = single.checked_mul(2).context("B2 cache bytes overflow")?;
    let reserve = 2u64 << 30;
    let needed = destinations
        .checked_add(source)
        .and_then(|n| n.checked_add(reserve))
        .context("B2 preflight overflow")?;
    let (free, _) = device_free_total_bytes(target.device())
        .context("B2 preflight needs bound worker CUDA context")?;
    eprintln!("[laguna_vram_preflight] route=prepared-b2-fixed-one physical_lanes=2 destination_bytes={destinations} reserve_bytes={reserve} free_bytes={free} capacity_per_lane={max_seq} source_bf16_bytes={source} draft_bytes=0 reserve_is_estimate=true");
    ensure!(free >= needed,
        "engine busy: required B2 destinations ({destinations} bytes), shared BF16 source ({source} bytes) and reserve ({reserve}) exceed free CUDA memory ({free}); try later.");
    Ok(())
}

fn validate_graph_lanes(lanes: usize, interleave: bool, fp8_kv: bool) -> Result<()> {
    ensure!(
        lanes == 2,
        "NV_LAGUNA_BATCH_GRAPH=1 requires NV_LAGUNA_BATCH=2"
    );
    ensure!(
        interleave,
        "NV_LAGUNA_BATCH_GRAPH=1 requires NV_LAGUNA_BATCH_INTERLEAVE_PREFILL=1"
    );
    ensure!(
        !fp8_kv,
        "NV_LAGUNA_BATCH_GRAPH=1 requires NV_LAGUNA_FP8_KV unset"
    );
    Ok(())
}

enum BatchLane {
    Eager(LagunaKvCache),
    Graph {
        step: Box<LagunaStepGraph>,
        replays: u64,
    },
}

impl BatchLane {
    fn cache_mut(&mut self) -> &mut LagunaKvCache {
        match self {
            Self::Eager(cache) => cache,
            Self::Graph { step, .. } => step.cache_mut(),
        }
    }
}

struct BatchPrefillSource {
    cache: LagunaKvCache,
    owner: Option<usize>,
}

impl BatchPrefillSource {
    fn new(target: &Laguna, max_seq: usize) -> Result<Option<Self>> {
        if !bf16_prefill_handoff_opt_in() {
            return Ok(None);
        }
        let cache = LagunaKvCache::new_with_prefill_chunk(
            target.config(),
            max_seq,
            target.device(),
            target.dtype(),
            target.prefill_chunk_tokens(),
        )
        .context("B2 shared BF16 prefill source allocation")?;
        eprintln!("[laguna_b2_handoff] created source_caches=1 destination_caches=2 source_kv=bf16 destination_kv=fixed-one-e4m3 prefill_policy=one_owner_until_complete decode_between_chunks=true cap={max_seq}");
        Ok(Some(Self { cache, owner: None }))
    }

    fn reset(&mut self) {
        self.cache.reset();
        self.owner = None;
    }
}

fn select_prefill_lane(
    prefills: &[Option<LanePrefill>],
    cursor: usize,
    owner: Option<usize>,
) -> Result<Option<usize>> {
    if let Some(lane) = owner {
        ensure!(
            prefills.get(lane).is_some_and(Option::is_some),
            "engine busy: shared BF16 prefill owner has no pending request; try later."
        );
        return Ok(Some(lane));
    }
    Ok((0..prefills.len())
        .map(|i| (cursor + i) % prefills.len())
        .find(|&i| prefills[i].is_some()))
}

pub struct BatchM1State {
    spec: Option<BatchSpecState>,
    pending_retirement: Option<anyhow::Error>,
    probe: RetirementProbe,
    target: Arc<Laguna>,
    device: Device,
    max_seq: usize,
    lanes: Vec<BatchLane>,
    batched: Option<LagunaBatchStepGraph>,
    handoff_source: Option<BatchPrefillSource>,
    interleave_prefill: bool,
}

impl BatchM1State {
    pub fn new(target: &Arc<Laguna>, lanes: usize, max_seq: usize) -> Result<Self> {
        ensure!(
            target.device().is_cuda(),
            "laguna batched serving requires a CUDA device"
        );
        ensure!(lanes >= 2, "a 1-lane batch loop is just the M=1 loop");
        let interleave_prefill = match env::var("NV_LAGUNA_BATCH_INTERLEAVE_PREFILL") {
            Ok(v) if v == "1" => true,
            Ok(v) if v == "0" => false,
            Err(std::env::VarError::NotPresent) => false,
            _ => bail!("NV_LAGUNA_BATCH_INTERLEAVE_PREFILL must be 0 or 1"),
        };
        let graph_lanes = batch_graph_enabled_from_env()?;
        let m2_graph = batch_m2_graph_enabled_from_env();
        validate_bf16_prefill_handoff_selection()?;
        validate_m2_graph_selection()?;
        if m2_graph {
            ensure!(
                lanes == 2 && interleave_prefill,
                "prepared B2 graph requires two interleaved slots"
            );
            preflight_b2_caches(target, max_seq)?;
            let batched = LagunaBatchStepGraph::new(target.clone(), max_seq)?;
            let handoff_source = BatchPrefillSource::new(target, max_seq)?;
            return Ok(Self {
                spec: None,
                pending_retirement: None,
                probe: RetirementProbe::new(target.device())?,
                target: target.clone(),
                device: target.device().clone(),
                max_seq,
                lanes: Vec::new(),
                batched: Some(batched),
                handoff_source,
                interleave_prefill,
            });
        }
        if graph_lanes {
            validate_graph_lanes(
                lanes,
                interleave_prefill,
                nv_config::env_arm_present_at_all_reread_every_call("NV_LAGUNA_FP8_KV"),
            )?;
        }
        let mut v = Vec::with_capacity(lanes);
        for _ in 0..lanes {
            let cache = target.new_kv_cache(max_seq)?;
            v.push(if graph_lanes {
                BatchLane::Graph {
                    step: Box::new(LagunaStepGraph::new(Arc::clone(target), cache)?),
                    replays: 0,
                }
            } else {
                BatchLane::Eager(cache)
            });
        }
        Ok(Self {
            spec: None,
            pending_retirement: None,
            probe: RetirementProbe::new(target.device())?,
            target: Arc::clone(target),
            device: target.device().clone(),
            max_seq,
            lanes: v,
            batched: None,
            handoff_source: None,
            interleave_prefill,
        })
    }

    pub fn lane_count(&self) -> usize {
        if self.batched.is_some() {
            2
        } else {
            self.lanes.len()
        }
    }

    fn reset_lane(&mut self, lane: usize) -> Result<()> {
        if let Some(spec) = self.spec.as_mut() {
            spec.lanes[lane].commit.require_idle()?;
            self.device.synchronize()?;
            spec.lanes[lane].ctx.reset();
            spec.lanes[lane].rounds = 0;
            spec.lanes[lane].accepted = 0;
        }
        if let Some(source) = self
            .handoff_source
            .as_mut()
            .filter(|s| s.owner == Some(lane))
        {
            source.reset();
        }
        if let Some(graph) = self.batched.as_mut() {
            graph.cache_mut(lane)?.reset();
        } else {
            self.lanes[lane].cache_mut().reset();
        }
        Ok(())
    }

    fn request_error(&self, e: anyhow::Error) -> (String, Option<anyhow::Error>) {
        if self.batched.is_some() {
            eprintln!("[laguna_b2_graph] retiring worker after failure without reusing partially mutated state: {e:#}");
            (
                WORKER_RETIRED_ERROR_IS_THE_EXACT_STRING_THE_HTTP_SHARED_RETIREMENT_GUARD_MATCHES
                    .to_string(),
                Some(e),
            )
        } else {
            request_error_or_worker_retirement(&self.probe, e)
        }
    }

    fn prefill_lane(&mut self, lane: usize, job: &mut SpecServeJob) -> Result<u32> {
        ensure!(!job.prompt_ids.is_empty(), "batch serve: empty prompt");
        let n = job.prompt_ids.len();
        ensure!(
            n + 2 <= self.max_seq,
            "batch serve: prompt ({n}) exceeds KV capacity ({})",
            self.max_seq
        );
        self.reset_lane(lane)?;
        let mut offset = 0usize;
        while offset < n {
            ensure!(
                job.client_is_live(),
                "batch serve: client disconnected before prefill completed"
            );
            if let Some(anchor) = self.prefill_lane_chunk(lane, job, &mut offset)? {
                return Ok(anchor);
            }
        }
        bail!("batch serve: prefill completed without logits")
    }

    fn prefill_lane_chunk(
        &mut self,
        lane: usize,
        job: &SpecServeJob,
        offset: &mut usize,
    ) -> Result<Option<u32>> {
        let n = job.prompt_ids.len();
        let m = self.target.next_prefill_chunk_len(n - *offset);
        let toks = Tensor::from_vec(
            job.prompt_ids[*offset..*offset + m].to_vec(),
            (1usize, m),
            &self.device,
        )?;
        let pos: Vec<i32> = (*offset as i32..(*offset + m) as i32).collect();
        let pos_t = Tensor::from_vec(pos, m, &self.device)?;
        let aux_layers = self
            .spec
            .as_ref()
            .map(|s| s.aux_layers.as_slice())
            .unwrap_or(&[]);
        let spec_tier = self.spec.is_some();
        let (logits, aux) = if let Some(source) = self.handoff_source.as_mut() {
            if source.owner.is_none() {
                ensure!(
                    *offset == 0,
                    "B2 shared BF16 prefill cannot resume an unowned prompt"
                );
                source.cache.reset();
                source.owner = Some(lane);
                eprintln!("[laguna_b2_handoff] acquired lane={lane} prompt_tokens={n}");
            }
            ensure!(
                source.owner == Some(lane)
                    && crate::gemma4::Gemma4Cache::current_len(&source.cache) == *offset,
                "B2 shared BF16 prefill owner or frontier mismatch"
            );
            self.target.prefill_with_cache_aux_scoped(
                &toks,
                &pos_t,
                &mut source.cache,
                aux_layers,
                spec_tier,
                *offset + m == n,
            )?
        } else if let Some(graph) = self.batched.as_mut() {
            self.target.prefill_with_cache_aux_scoped(
                &toks,
                &pos_t,
                graph.cache_mut(lane)?,
                aux_layers,
                spec_tier,
                *offset + m == n,
            )?
        } else {
            self.target.prefill_with_cache_aux_scoped(
                &toks,
                &pos_t,
                self.lanes[lane].cache_mut(),
                &[],
                false,
                *offset + m == n,
            )?
        };
        if let Some(spec) = self.spec.as_mut() {
            ensure!(
                spec.theta == spec.draft.active_rope_theta().to_bits()
                    && spec.lanes[lane].ctx.len() == *offset,
                "B2 draft prefill frontier/theta mismatch"
            );
            let combined = spec.draft.combine_aux(&aux)?;
            let positions = (*offset..*offset + m)
                .map(i32::try_from)
                .collect::<std::result::Result<Vec<_>, _>>()?;
            spec.draft
                .append_context(&mut spec.lanes[lane].ctx, &combined, &positions)?;
            ensure!(
                spec.lanes[lane].ctx.len() == *offset + m,
                "B2 draft context did not commit the complete prefill chunk"
            );
        }
        *offset += m;
        if *offset != n {
            return Ok(None);
        }
        if let Some(BatchLane::Graph { step, replays }) = self.lanes.get(lane) {
            eprintln!("[laguna_serve] graph lane prefill complete lane={lane} prompt_tokens={n} kv=bf16 graph_nodes={} prior_total_replays={replays}", step.captured_node_count());
        }
        let logits = logits.context("batch serve: final prefill chunk did not return logits")?;
        let m_last = logits.dims()[1];
        let last_row: Vec<f32> = logits.narrow(1, m_last - 1, 1)?.flatten_all()?.to_vec1()?;
        let anchor = checked_target_argmax_row(&last_row)
            .context("engine busy: Laguna batch prefill target logits are invalid; try later.")?;
        if let Some(source) = self.handoff_source.as_mut() {
            let start = std::time::Instant::now();
            let graph = self
                .batched
                .as_mut()
                .context("B2 BF16 handoff requires the prepared graph")?;
            let destination = graph.cache_mut(lane)?;
            ensure!(
                crate::gemma4::Gemma4Cache::current_len(&source.cache) == n,
                "B2 BF16 handoff requires the entire prompt in its source"
            );
            destination
                .copy_from_bf16(&mut source.cache)
                .context("B2 shared BF16 prefill handoff")?;
            ensure!(
                destination.current_len() == n
                    && destination.direct_fp8_prefill_enqueued_calls() == 0,
                "B2 BF16 handoff destination frontier or prefill route mismatch"
            );
            eprintln!("[laguna_b2_handoff] complete lane={lane} prompt_tokens={n} converted_rows={n} destination_len={} convert_ms={:.3} direct_fp8_prefill_calls=0 draft_context_len={}",
                destination.current_len(), start.elapsed().as_secs_f64() * 1e3,
                self.spec.as_ref().map_or(0, |spec| spec.lanes[lane].ctx.len()));
            source.reset();
        }
        if let Some(graph) = self.batched.as_ref() {
            graph.report("prefill-complete");
        }
        Ok(Some(anchor))
    }

    fn reserve_prefill(&mut self, lane: usize, mut job: SpecServeJob) -> Option<LanePrefill> {
        if !job.client_is_live() {
            return None;
        }
        let n = job.prompt_ids.len();
        let required = if self.spec.is_some() {
            job.max_new
        } else {
            job.max_new.max(2)
        };
        if n == 0
            || !n
                .checked_add(required)
                .is_some_and(|end| end <= self.max_seq)
        {
            let _ = (job.emit)(SpecServeEvent::Error(format!(
                "batch serve: prompt ({n}) plus output ({}) exceeds KV capacity ({}) or prompt is empty",
                job.max_new, self.max_seq,
            )));
            return None;
        }
        if job.max_new == 0 {
            let _ = (job.emit)(SpecServeEvent::Done);
            return None;
        }
        if let Err(e) = self.reset_lane(lane) {
            let (message, retirement) = self.request_error(e);
            let _ = (job.emit)(SpecServeEvent::Error(message));
            self.pending_retirement = retirement;
            return None;
        }
        Some(LanePrefill { job, offset: 0 })
    }

    fn interleaved_turn(
        &mut self,
        slots: &mut [Option<LaneRun>],
        prefills: &mut [Option<LanePrefill>],
        cursor: &mut usize,
    ) -> Result<()> {
        let result = self.interleaved_turn_inner(slots, prefills, cursor);
        if result.is_err() {
            if let Some(source) = self.handoff_source.as_mut() {
                source.reset();
            }
        }
        result
    }

    fn cancel_shared_prefill(&mut self, lane: usize) -> Result<()> {
        if self.handoff_source.is_some() {
            let owner = self.handoff_source.as_ref().and_then(|source| source.owner);
            self.reset_lane(lane)?;
            eprintln!(
                "[laguna_b2_handoff] cancelled lane={lane} source_reset={} draft_context_reset={}",
                owner == Some(lane),
                self.spec.is_some()
            );
        }
        Ok(())
    }

    fn interleaved_turn_inner(
        &mut self,
        slots: &mut [Option<LaneRun>],
        prefills: &mut [Option<LanePrefill>],
        cursor: &mut usize,
    ) -> Result<()> {
        for (lane, pending) in prefills.iter_mut().enumerate() {
            if pending.as_mut().is_some_and(|p| !p.job.client_is_live()) {
                pending.take();
                self.cancel_shared_prefill(lane)?;
            }
        }
        let decoded = self.decode_step(slots);
        if decoded.is_err() && self.batched.is_none() {
            self.device.synchronize()?;
        }
        decoded?;
        let owner = self.handoff_source.as_ref().and_then(|source| source.owner);
        let Some(lane) = select_prefill_lane(prefills, *cursor, owner)? else {
            return Ok(());
        };
        *cursor = (lane + 1) % prefills.len();
        let pending = prefills[lane].as_mut().expect("selected prefill lane");
        if !pending.job.client_is_live() {
            prefills[lane].take();
            self.cancel_shared_prefill(lane)?;
            return Ok(());
        }
        let result = self.prefill_lane_chunk(lane, &pending.job, &mut pending.offset);
        let anchor = finish_batch_prefill(result, self.batched.is_some(), || {
            Ok(self.device.synchronize()?)
        })?;
        if let Some(anchor) = anchor {
            let mut job = prefills[lane].take().expect("completed prefill lane").job;
            let pos = job.prompt_ids.len();
            let mut total = 0;
            if let Flow::Stop = push_round(&mut job, &[anchor], &mut total) {
                let _ = (job.emit)(SpecServeEvent::Done);
            } else {
                slots[lane] = Some(LaneRun {
                    job,
                    last: anchor,
                    pos,
                    total,
                });
            }
        }
        Ok(())
    }

    fn admit(&mut self, lane: usize, mut job: SpecServeJob) -> Option<LaneRun> {
        if !job.client_is_live() {
            return None;
        }
        match self.prefill_lane(lane, &mut job) {
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
                let (message, retirement) = self.request_error(e);
                let _ = (job.emit)(SpecServeEvent::Error(message));
                self.pending_retirement = retirement;
                None
            }
        }
    }

    fn decode_step(&mut self, slots: &mut [Option<LaneRun>]) -> Result<()> {
        for s in slots.iter_mut() {
            if s.as_mut().is_some_and(|r| !r.job.client_is_live()) {
                s.take();
                continue;
            }
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
        if self.spec.is_some() {
            return self.decode_spec_step(slots);
        }
        let tokens: Vec<u32> = active
            .iter()
            .map(|&i| slots[i].as_ref().expect("active slot").last)
            .collect();
        let positions: Vec<usize> = active
            .iter()
            .map(|&i| slots[i].as_ref().expect("active slot").pos)
            .collect();
        let next_tokens = if let Some(graph) = self.batched.as_mut() {
            let logits = graph.step(&active, &tokens, &positions)?;
            checked_batch_target_tokens(&logits, active.len())?
        } else if matches!(&self.lanes[active[0]], BatchLane::Graph { .. }) {
            let mut next = Vec::with_capacity(active.len());
            for (row, &i) in active.iter().enumerate() {
                let BatchLane::Graph { step, replays } = &mut self.lanes[i] else {
                    bail!("graph batch lane unexpectedly contains an eager cache");
                };
                ensure!(
                    step.current_len() == positions[row],
                    "graph lane {i} cache position mismatch"
                );
                let was_captured = step.captured_node_count() > 0;
                step.step(tokens[row])
                    .with_context(|| format!("graph lane {i} step"))?;
                next.push(
                    step.argmax_device()
                        .with_context(|| format!("graph lane {i} argmax"))?,
                );
                ensure!(
                    step.captured_node_count() > 0,
                    "graph lane {i} step returned without a captured graph"
                );
                *replays += u64::from(was_captured);
            }
            next
        } else {
            let mut cache_refs: Vec<&mut LagunaKvCache> = self
                .lanes
                .iter_mut()
                .enumerate()
                .filter(|(i, _)| active.contains(i))
                .map(|(_, lane)| lane.cache_mut())
                .collect();
            let logits =
                self.target
                    .forward_decode_batched(&tokens, &positions, &mut cache_refs)?;
            let host: Vec<f32> = logits.flatten_all()?.to_vec1()?;
            checked_batch_target_tokens(&host, active.len())?
        };
        for (row, &i) in active.iter().enumerate() {
            let tok = next_tokens[row];
            let r = slots[i].as_mut().expect("active slot");
            r.pos += 1;
            r.last = tok;
            if let Flow::Stop = push_round(&mut r.job, &[tok], &mut r.total) {
                let mut r = slots[i].take().expect("active slot");
                if let Some(graph) = self.batched.as_ref() {
                    graph.report("request-complete");
                }
                if let Some(BatchLane::Graph { step, replays }) = self.lanes.get(i) {
                    eprintln!("[laguna_serve] graph lane request complete lane={i} prompt_tokens={} emitted={} graph_nodes={} total_successful_replays={replays} cache_len={}", r.job.prompt_ids.len(), r.total, step.captured_node_count(), step.current_len());
                }
                let _ = (r.job.emit)(SpecServeEvent::Done);
            }
        }
        Ok(())
    }

    pub fn serve_batch(&mut self, jobs: Vec<SpecServeJob>) -> Result<()> {
        if let Some(fault) = self.pending_retirement.as_ref() {
            bail!("{fault:#}");
        }
        ensure!(
            jobs.len() <= self.lane_count(),
            "serve_batch: {} jobs > {} lanes",
            jobs.len(),
            self.lane_count()
        );
        let mut slots: Vec<Option<LaneRun>> = (0..self.lane_count()).map(|_| None).collect();
        if self.interleave_prefill {
            let mut prefills: Vec<Option<LanePrefill>> =
                (0..self.lane_count()).map(|_| None).collect();
            let mut cursor = 0;
            for (i, job) in jobs.into_iter().enumerate() {
                prefills[i] = self.reserve_prefill(i, job);
            }
            while slots.iter().any(Option::is_some) || prefills.iter().any(Option::is_some) {
                self.interleaved_turn(&mut slots, &mut prefills, &mut cursor)?;
            }
            return Ok(());
        }
        for (i, job) in jobs.into_iter().enumerate() {
            slots[i] = self.admit(i, job);
            if let Some(fault) = self.pending_retirement.as_ref() {
                retire_active_jobs_before_the_worker_exits(&mut slots);
                bail!("{fault:#}");
            }
        }
        while slots.iter().any(|s| s.is_some()) {
            self.decode_step(&mut slots)?;
        }
        Ok(())
    }

    fn warmup(&mut self) -> Result<()> {
        let ids: Vec<u32> = (0..48u32).map(|i| 100 + i).collect();
        let n_jobs = self.lane_count().min(2);
        let batch: Vec<SpecServeJob> = (0..n_jobs)
            .map(|_| SpecServeJob {
                prompt_ids: ids.clone(),
                max_new: 4,
                eos_ids: Vec::new(),
                emit: Box::new(|_| true),
            })
            .collect();
        self.serve_batch(batch)
            .context("laguna batch serving warmup")?;
        for (lane, storage) in self.lanes.iter().enumerate() {
            if let BatchLane::Graph { step, replays } = storage {
                step.synchronize()?;
                ensure!(
                    step.captured_node_count() > 0 && *replays > 0,
                    "graph lane {lane} warmup did not capture and replay"
                );
                eprintln!("[laguna_serve] graph lane ready lane={lane} kv=bf16 max_seq={} graph_nodes={} total_successful_replays={replays}", self.max_seq, step.captured_node_count());
            }
        }
        if self.spec.is_some() {
            self.warm_spec_masks()?;
        } else if let Some(graph) = self.batched.as_mut() {
            graph.warm_all_masks()?;
        }
        self.device.synchronize().ok();
        Ok(())
    }
}

fn finish_batch_prefill<T>(
    result: Result<T>,
    prepared: bool,
    synchronize: impl FnOnce() -> Result<()>,
) -> Result<T> {
    if prepared && result.is_err() {
        return result;
    }
    let synchronized = synchronize();
    let value = result?;
    synchronized?;
    Ok(value)
}

fn checked_batch_target_tokens(logits: &[f32], rows: usize) -> Result<Vec<u32>> {
    ensure!(
        rows > 0 && !logits.is_empty() && logits.len().is_multiple_of(rows),
        "engine busy: Laguna batch target logits have an invalid row shape; try later."
    );
    logits
        .chunks_exact(logits.len() / rows)
        .enumerate()
        .map(|(row, values)| {
            checked_target_argmax_row(values).with_context(|| {
                format!(
                    "engine busy: Laguna batch target logits are invalid at row {row}; try later."
                )
            })
        })
        .collect()
}

struct LaneRun {
    job: SpecServeJob,
    last: u32,
    pos: usize,
    total: usize,
}

fn retire_active_jobs_before_the_worker_exits(slots: &mut [Option<LaneRun>]) {
    for slot in slots {
        if let Some(mut run) = slot.take() {
            let _ = (run.job.emit)(SpecServeEvent::Error(
                WORKER_RETIRED_ERROR_IS_THE_EXACT_STRING_THE_HTTP_SHARED_RETIREMENT_GUARD_MATCHES
                    .to_string(),
            ));
        }
    }
}

struct LanePrefill {
    job: SpecServeJob,
    offset: usize,
}

fn batch_m1_interleaved_loop(mut st: BatchM1State, jobs: Receiver<SpecServeJob>) -> Result<()> {
    let lanes = st.lane_count();
    let mut slots: Vec<Option<LaneRun>> = (0..lanes).map(|_| None).collect();
    let mut prefills: Vec<Option<LanePrefill>> = (0..lanes).map(|_| None).collect();
    let mut cursor = 0;
    loop {
        if slots.iter().all(Option::is_none) && prefills.iter().all(Option::is_none) {
            match jobs.recv() {
                Ok(job) => prefills[0] = st.reserve_prefill(0, job),
                Err(_) => return Ok(()),
            }
        }
        for _ in 0..lanes {
            let Some(free) = (0..lanes).find(|&i| slots[i].is_none() && prefills[i].is_none())
            else {
                break;
            };
            match jobs.try_recv() {
                Ok(job) => prefills[free] = st.reserve_prefill(free, job),
                Err(_) => break,
            }
        }
        let turn = if let Some(e) = st.pending_retirement.take() {
            Err(e)
        } else {
            st.interleaved_turn(&mut slots, &mut prefills, &mut cursor)
        };
        if let Err(e) = turn {
            let (message, retirement) = st.request_error(e);
            for slot in &mut slots {
                if let Some(mut run) = slot.take() {
                    let _ = (run.job.emit)(SpecServeEvent::Error(message.clone()));
                }
            }
            for pending in &mut prefills {
                if let Some(mut run) = pending.take() {
                    let _ = (run.job.emit)(SpecServeEvent::Error(message.clone()));
                }
            }
            if let Some(fault) = retirement {
                retire_queued_jobs_before_the_worker_exits(&jobs);
                retirement_teardown_marker("batch worker: dropping BatchM1State");
                drop(st);
                retirement_teardown_marker(
                    "batch worker: BatchM1State dropped; returning the fault",
                );
                return Err(fault);
            }
        }
    }
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
         free lanes admit queued jobs at every step boundary, interleave_prefill={} prefill_chunk={}",
        st.lane_count(),
        max_seq,
        st.interleave_prefill,
        st.target.prefill_chunk_tokens(),
    );
    if matches!(st.lanes.first(), Some(BatchLane::Graph { .. })) {
        eprintln!("[laguna_serve] graph lane pool ready slots={} execution=sequential_persistent_m1_graphs gpu_batch=1 draft=false kv=bf16 max_seq_per_slot={max_seq}", st.lane_count());
    }
    let _ = ready.send(false);
    if st.interleave_prefill {
        return batch_m1_interleaved_loop(st, jobs);
    }
    let mut slots: Vec<Option<LaneRun>> = (0..lanes).map(|_| None).collect();
    loop {
        if slots.iter().all(|s| s.is_none()) {
            match jobs.recv() {
                Ok(job) => {
                    slots[0] = st.admit(0, job);
                    if let Some(fault) = st.pending_retirement.take() {
                        retire_active_jobs_before_the_worker_exits(&mut slots);
                        retire_queued_jobs_before_the_worker_exits(&jobs);
                        return Err(fault);
                    }
                }
                Err(_) => return Ok(()),
            }
        }
        while let Some(free) = slots.iter().position(|s| s.is_none()) {
            match jobs.try_recv() {
                Ok(job) => {
                    slots[free] = st.admit(free, job);
                    if let Some(fault) = st.pending_retirement.take() {
                        retire_active_jobs_before_the_worker_exits(&mut slots);
                        retire_queued_jobs_before_the_worker_exits(&jobs);
                        return Err(fault);
                    }
                }
                Err(_) => break,
            }
        }
        if slots.iter().all(|s| s.is_none()) {
            continue;
        }
        if let Err(e) = st.decode_step(&mut slots) {
            let (msg, retirement) = st.request_error(e);
            for s in slots.iter_mut() {
                if let Some(mut r) = s.take() {
                    let _ = (r.job.emit)(SpecServeEvent::Error(msg.clone()));
                }
            }
            if let Some(fault) = retirement {
                retire_queued_jobs_before_the_worker_exits(&jobs);
                retirement_teardown_marker("batch worker: dropping BatchM1State");
                drop(st);
                retirement_teardown_marker(
                    "batch worker: BatchM1State dropped; returning the fault",
                );
                return Err(fault);
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
    validate_bf16_prefill_handoff_selection()?;
    validate_m2_graph_selection()?;
    validate_m2_spec_selection()?;
    if batch_m2_spec_enabled_from_env() {
        let mut state = BatchM1State::new_spec(
            &target,
            draft.context("m2-spec requires packed draft")?,
            num_spec,
            max_seq,
        )?;
        state.warmup()?;
        state.batched.as_ref().unwrap().report_spec("ready");
        let _ = ready.send(true);
        return batch_m1_interleaved_loop(state, jobs);
    }
    let target = target;
    let draft = draft;
    if batch_graph_enabled_from_env()? {
        ensure!(
            draft.is_none(),
            "NV_LAGUNA_BATCH_GRAPH=1 requires draft-free serving"
        );
        ensure!(
            batch_lanes_from_env() == Some(2),
            "NV_LAGUNA_BATCH_GRAPH=1 requires NV_LAGUNA_BATCH=2"
        );
    }
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
    vram_phase_marker(&st.device, "warmup-graphs-captured", "");
    let has_draft = st.draft.is_some();
    let _ = ready.send(has_draft);
    while let Ok(mut job) = jobs.recv() {
        match st.serve_one(&mut job) {
            Ok(()) => {}
            Err(e) => {
                let execution_error_already_delivered = e.is::<LagunaRequestExecutionFailure>();
                let (message, retirement) = request_error_or_worker_retirement(&st.probe, e);
                if !execution_error_already_delivered || retirement.is_some() {
                    let _ = (job.emit)(SpecServeEvent::Error(message));
                }
                if let Some(fault) = retirement {
                    retire_queued_jobs_before_the_worker_exits(&jobs);
                    drop(job);
                    retirement_teardown_marker("worker: failing job released; dropping SpecState (step graph, proposers, verify graphs, ctx cache)");
                    drop(st);
                    retirement_teardown_marker(
                        "worker: SpecState dropped; dropping the packed draft",
                    );
                    drop(draft);
                    retirement_teardown_marker(
                        "worker: draft dropped; returning the fault to the spawner",
                    );
                    return Err(fault);
                }
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod spec_capacity_tests {
    #[test]
    fn k2_uses_real_remaining_capacity_without_truncating_the_last_two_slots() {
        assert_eq!(super::batch_spec_rows(262141, 262144).unwrap(), 3);
        assert_eq!(super::batch_spec_rows(262142, 262144).unwrap(), 1);
        assert_eq!(super::batch_spec_rows(262143, 262144).unwrap(), 1);
        assert!(super::batch_spec_rows(262144, 262144).is_err());
        assert!(super::batch_spec_rows(usize::MAX, 262144).is_err());
        assert!(super::batch_spec_rows(0, 0).is_err());
    }
}

#[cfg(test)]
mod host_target_logits_tests {
    use super::{checked_batch_target_tokens, checked_target_argmax_row};

    #[test]
    fn target_rows_preserve_token_zero_masks_and_first_index_ties_and_a_batch_rejects_any_invalid_row_before_emission() {
        for (row, expected) in [
            (vec![9.0, 1.0, f32::NEG_INFINITY], 0),
            (vec![f32::NEG_INFINITY, -4.0, -4.0], 1),
            (vec![-0.0, 0.0], 0),
            (vec![0.0, -0.0], 0),
            (vec![1.0, 7.0, 7.0], 1),
            (vec![f32::NEG_INFINITY, -7.0, -2.0], 2),
        ] {
            assert_eq!(checked_target_argmax_row(&row), Some(expected), "{row:?}");
        }
        for row in [
            vec![9.0, 1.0, f32::NAN],
            vec![9.0, 1.0, f32::INFINITY],
            vec![9.0, f32::NAN, -100.0],
            vec![9.0, f32::INFINITY, -100.0],
            vec![f32::NAN, 9.0],
            vec![f32::INFINITY, 9.0],
            vec![f32::NEG_INFINITY, f32::NEG_INFINITY],
            vec![f32::NAN, f32::NAN],
            Vec::new(),
        ] {
            assert_eq!(
                checked_target_argmax_row(&row),
                None,
                "invalid values anywhere in a target row must reject it: {row:?}"
            );
        }
        let mut emissions = Vec::new();
        for bad_tail in [f32::NAN, f32::INFINITY] {
            let logits = [5.0, 1.0, -1.0, 9.0, 0.0, bad_tail, 0.0, 7.0, 2.0];
            let result =
                checked_batch_target_tokens(&logits, 3).map(|tokens| emissions.push(tokens));
            let error = result.expect_err("a valid first lane must not hide an invalid later lane");
            let message = format!("{error:#}");
            assert!(
                message.starts_with("engine busy:")
                    && message.contains("row 1")
                    && message.contains("try later"),
                "{message}"
            );
            assert!(
                emissions.is_empty(),
                "no lane's tokens may reach emission until every row validates"
            );
        }
        for (logits, rows) in [(vec![], 1), (vec![1.0], 0), (vec![1.0, 2.0, 3.0], 2)] {
            assert!(checked_batch_target_tokens(&logits, rows).is_err());
        }
        let good = [5.0, 5.0, f32::NEG_INFINITY, f32::NEG_INFINITY, -2.0, -2.0];
        let tokens = checked_batch_target_tokens(&good, 2).unwrap();
        assert_eq!(tokens, vec![0, 1]);
        emissions.push(tokens);
        assert_eq!(
            emissions,
            vec![vec![0, 1]],
            "a rejected batch cannot contaminate the next valid reduction"
        );
    }
}

#[cfg(test)]
mod retirement_producer_tests {
    use super::*;
    use std::sync::{mpsc, Mutex};

    #[test]
    fn prepared_prefill_error_reaches_retirement_without_a_cleanup_barrier_and_a_successful_prefill_waits_before_its_anchor() {
        let log = Arc::new(Mutex::new(Vec::new()));
        let mut active = vec![Some(LaneRun {
            job: closed_client_job(0, &log),
            last: 1,
            pos: 1,
            total: 0,
        })];
        let error =
            finish_batch_prefill::<()>(Err(anyhow::anyhow!("returned GPU failure")), true, || {
                panic!(
                    "known prepared-batch failure must reach retirement before another CUDA wait"
                )
            })
            .unwrap_err();
        assert_eq!(error.to_string(), "returned GPU failure");
        retire_active_jobs_before_the_worker_exits(&mut active);
        assert_eq!(
            log.lock().unwrap().as_slice(),
            [
                "0:event:engine busy: Laguna worker retired; try later.",
                "0:released",
            ]
        );
        for prepared in [false, true] {
            let synchronized = std::cell::Cell::new(false);
            let anchor = finish_batch_prefill(Ok(Some(42u32)), prepared, || {
                synchronized.set(true);
                Ok(())
            })
            .unwrap();
            assert!(synchronized.get());
            assert_eq!(anchor, Some(42));
            assert_eq!(
                finish_batch_prefill(Ok(Some(42u32)), prepared, || {
                    Err(anyhow::anyhow!("completion barrier failed"))
                })
                .unwrap_err()
                .to_string(),
                "completion barrier failed"
            );
        }
        let synchronized = std::cell::Cell::new(false);
        let error =
            finish_batch_prefill::<()>(Err(anyhow::anyhow!("original eager error")), false, || {
                synchronized.set(true);
                Err(anyhow::anyhow!("later cleanup error"))
            })
            .unwrap_err();
        assert!(synchronized.get());
        assert_eq!(error.to_string(), "original eager error");
    }

    #[test]
    fn typed_execution_failure_preserves_cause_probe_controls_retirement_and_a_successful_probe_never_classifies_text() {
        for failed_probe in [false, true] {
            let error = anyhow::anyhow!("Laguna FMHAv2 FP8 requires an explicit CUDA stream")
                .context(LagunaRequestExecutionFailure);
            assert!(error.is::<LagunaRequestExecutionFailure>());
            assert_eq!(
                error.root_cause().to_string(),
                "Laguna FMHAv2 FP8 requires an explicit CUDA stream"
            );
            let probe = if failed_probe {
                Err(anyhow::anyhow!("probe fault"))
            } else {
                Ok(())
            };
            let (message, retirement) = request_error_or_worker_retirement_from_probe(probe, error);
            if failed_probe {
                assert_eq!(message, WORKER_RETIRED_ERROR_IS_THE_EXACT_STRING_THE_HTTP_SHARED_RETIREMENT_GUARD_MATCHES);
                assert!(format!("{:#}", retirement.unwrap())
                    .contains("requires an explicit CUDA stream"));
            } else {
                assert_eq!(message, "engine busy: Laguna request execution failed; try later: Laguna FMHAv2 FP8 requires an explicit CUDA stream");
                assert!(retirement.is_none());
            }
        }
        for cause in [
            "invalid prompt",
            "CUDA_ERROR_OUT_OF_MEMORY",
            "cuda error: an illegal memory access was encountered",
            "engine busy: invalid verification logits; try later.",
        ] {
            let request = anyhow::anyhow!(cause.to_string()).context("request boundary");
            let expected = format!("{request:#}");
            let (message, retirement) =
                request_error_or_worker_retirement_from_probe(Ok(()), request);
            assert_eq!(message, expected);
            assert!(
                retirement.is_none(),
                "a successful probe cannot be replaced by a string classifier"
            );
        }

    }

    struct CallbackReservation {
        id: usize,
        log: Arc<Mutex<Vec<String>>>,
    }

    impl CallbackReservation {
        fn event(&self, event: SpecServeEvent) -> bool {
            let message = match event {
                SpecServeEvent::Error(message) => message,
                SpecServeEvent::Done => "unexpected Done".into(),
                SpecServeEvent::Tokens(_) => "unexpected Tokens".into(),
                SpecServeEvent::PromptCached(_) => "unexpected PromptCached".into(),
            };
            self.log
                .lock()
                .unwrap()
                .push(format!("{}:event:{message}", self.id));
            false
        }
    }

    impl Drop for CallbackReservation {
        fn drop(&mut self) {
            self.log
                .lock()
                .unwrap()
                .push(format!("{}:released", self.id));
        }
    }

    fn closed_client_job(id: usize, log: &Arc<Mutex<Vec<String>>>) -> SpecServeJob {
        let reservation = CallbackReservation {
            id,
            log: log.clone(),
        };
        SpecServeJob {
            prompt_ids: vec![1],
            max_new: 1,
            eos_ids: Vec::new(),
            emit: Box::new(move |event| reservation.event(event)),
        }
    }

    #[test]
    fn retirement_drains_active_and_queued_jobs_even_when_delivery_returns_false_and_an_execution_error_keeps_its_callback_reserved() {
        let log = Arc::new(Mutex::new(Vec::new()));
        let mut active = vec![
            Some(LaneRun {
                job: closed_client_job(0, &log),
                last: 1,
                pos: 1,
                total: 0,
            }),
            None,
            Some(LaneRun {
                job: closed_client_job(1, &log),
                last: 1,
                pos: 1,
                total: 0,
            }),
        ];
        let (tx, rx) = mpsc::sync_channel(2);
        assert!(tx.try_send(closed_client_job(2, &log)).is_ok());
        assert!(tx.try_send(closed_client_job(3, &log)).is_ok());
        retire_active_jobs_before_the_worker_exits(&mut active);
        retire_queued_jobs_before_the_worker_exits(&rx);
        assert!(active.iter().all(Option::is_none));
        assert!(matches!(rx.try_recv(), Err(mpsc::TryRecvError::Empty)));
        let expected: Vec<String> = (0..4)
            .flat_map(|id| {
                [
                    format!("{id}:event:engine busy: Laguna worker retired; try later."),
                    format!("{id}:released"),
                ]
            })
            .collect();
        assert_eq!(
            *log.lock().unwrap(),
            expected,
            "every callback must observe retirement before its reservation releases, without Done"
        );
        retire_active_jobs_before_the_worker_exits(&mut active);
        retire_queued_jobs_before_the_worker_exits(&rx);
        assert_eq!(
            *log.lock().unwrap(),
            expected,
            "empty drains must not manufacture events"
        );

        let log = Arc::new(Mutex::new(Vec::new()));
        let mut job = closed_client_job(0, &log);
        assert!(job.report_execution_result(Ok(())).is_ok());
        assert!(
            log.lock().unwrap().is_empty(),
            "successful or cancelled execution must not produce Error"
        );
        let error = job
            .report_execution_result(Err(anyhow::anyhow!("fixture execution failure")))
            .unwrap_err();
        assert!(error.is::<LagunaRequestExecutionFailure>());
        assert_eq!(log.lock().unwrap().as_slice(), [
            "0:event:engine busy: Laguna request execution failed; try later: fixture execution failure",
        ]);
        let (message, retirement) = request_error_or_worker_retirement_from_probe(
            Err(anyhow::anyhow!("fixture probe fault")),
            error,
        );
        assert!(retirement.is_some());
        assert!(!(job.emit)(SpecServeEvent::Error(message)));
        drop(job);
        assert_eq!(log.lock().unwrap().as_slice(), [
            "0:event:engine busy: Laguna request execution failed; try later: fixture execution failure",
            "0:event:engine busy: Laguna worker retired; try later.",
            "0:released",
        ]);

    }
}

#[cfg(test)]
mod graph_lane_policy_tests {
    use super::{
        batch_graph_enabled_from_spec, batch_lanes_from_spec, select_prefill_lane,
        validate_graph_lanes, LanePrefill, SpecServeJob, LAGUNA_BATCH_DEFAULT_LANES,
        LAGUNA_BATCH_MAX_LANES,
    };

    fn pending() -> Option<LanePrefill> {
        Some(LanePrefill {
            job: SpecServeJob {
                prompt_ids: vec![1, 2, 3],
                max_new: 4,
                eos_ids: Vec::new(),
                emit: Box::new(|_| true),
            },
            offset: 0,
        })
    }

    #[test]
    fn batch_lane_graph_lane_and_shared_prefill_owner_selection_are_explicit_and_refuse_bad_specs_or_lost_owners() {
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

        assert!(!batch_graph_enabled_from_spec(None).unwrap());
        assert!(!batch_graph_enabled_from_spec(Some("0")).unwrap());
        assert!(batch_graph_enabled_from_spec(Some("1")).unwrap());
        assert!(batch_graph_enabled_from_spec(Some("m2")).unwrap());
        assert!(batch_graph_enabled_from_spec(Some("m2-spec")).unwrap());
        for value in ["", "2", "true", " 1"] {
            assert!(batch_graph_enabled_from_spec(Some(value)).is_err());
        }
        validate_graph_lanes(2, true, false).unwrap();
        for (lanes, interleave, fp8) in [
            (1, true, false),
            (4, true, false),
            (2, false, false),
            (2, true, true),
        ] {
            assert!(validate_graph_lanes(lanes, interleave, fp8).is_err());
        }

        let mut prefills = [pending(), pending()];
        for cursor in [0, 1, 4] {
            assert_eq!(
                select_prefill_lane(&prefills, cursor, Some(0)).unwrap(),
                Some(0)
            );
            assert_eq!(
                select_prefill_lane(&prefills, cursor, Some(1)).unwrap(),
                Some(1)
            );
        }
        prefills[0] = None;
        assert!(select_prefill_lane(&prefills, 1, Some(0)).is_err());
        assert!(select_prefill_lane(&prefills, 0, Some(2)).is_err());
        assert_eq!(select_prefill_lane(&prefills, 0, Some(1)).unwrap(), Some(1));
        assert_eq!(select_prefill_lane(&prefills, 0, None).unwrap(), Some(1));
        prefills[0] = pending();
        assert_eq!(select_prefill_lane(&prefills, 0, None).unwrap(), Some(0));
        assert_eq!(select_prefill_lane(&prefills, 1, None).unwrap(), Some(1));
        assert_eq!(select_prefill_lane(&[None, None], 0, None).unwrap(), None);

    }
}

#[cfg(test)]
mod shared_prefill_handoff_cuda_tests {
    use super::*;
    use crate::gemma4::{Gemma4Cache, LayerType};
    use cudarc::driver::DevicePtr;
    use std::sync::{
        atomic::{AtomicBool, Ordering},
        Mutex,
    };

    #[derive(Default)]
    struct Events {
        tokens: Vec<u32>,
        done: usize,
        errors: Vec<String>,
    }

    struct Client {
        live: Arc<AtomicBool>,
        events: Arc<Mutex<Events>>,
    }

    impl Client {
        fn cancel(&self) {
            self.live.store(false, Ordering::SeqCst);
        }
        fn tokens(&self) -> Vec<u32> {
            self.events.lock().unwrap().tokens.clone()
        }
        fn assert_complete(&self, count: usize) {
            let e = self.events.lock().unwrap();
            assert_eq!(e.tokens.len(), count);
            assert_eq!(e.done, 1);
            assert!(
                e.errors.is_empty(),
                "unexpected serving errors: {:?}",
                e.errors
            );
        }
    }

    fn job(ids: &[u32], max_new: usize) -> (SpecServeJob, Client) {
        let live = Arc::new(AtomicBool::new(true));
        let events = Arc::new(Mutex::new(Events::default()));
        let client = Client {
            live: live.clone(),
            events: events.clone(),
        };
        (
            SpecServeJob {
                prompt_ids: ids.to_vec(),
                max_new,
                eos_ids: Vec::new(),
                emit: Box::new(move |event| {
                    if !live.load(Ordering::SeqCst) {
                        return false;
                    }
                    let mut e = events.lock().unwrap();
                    match event {
                        SpecServeEvent::Tokens(ids) => e.tokens.extend(ids),
                        SpecServeEvent::Done => e.done += 1,
                        SpecServeEvent::Error(error) => e.errors.push(error),
                        SpecServeEvent::PromptCached(_) => {
                            panic!("handoff must not reuse prefixes")
                        }
                    }
                    true
                }),
            },
            client,
        )
    }

    fn source_frontier(st: &BatchM1State) -> (Option<usize>, usize) {
        let source = st
            .handoff_source
            .as_ref()
            .expect("one BF16 source required");
        (source.owner, source.cache.current_len())
    }

    fn source_pointers(st: &BatchM1State) -> Vec<u64> {
        let cache = &st.handoff_source.as_ref().unwrap().cache;
        let state = cache.checked_bf16_handoff().unwrap();
        let Device::Cuda(dev) = &st.device else {
            unreachable!()
        };
        let stream = dev.cuda_stream();
        state
            .layers
            .iter()
            .flat_map(|(k, v)| [k, v])
            .map(|tensor| {
                let (storage, layout) = tensor.storage_and_layout();
                assert_eq!(layout.start_offset(), 0);
                let candle_core::Storage::Cuda(storage) = &*storage else {
                    unreachable!()
                };
                let (pointer, _guard) = storage
                    .as_cuda_slice::<half::bf16>()
                    .unwrap()
                    .device_ptr(&stream);
                pointer
            })
            .collect()
    }

    fn source_full_prefix(st: &BatchM1State, rows: usize) -> Vec<u16> {
        let state = st
            .handoff_source
            .as_ref()
            .unwrap()
            .cache
            .checked_bf16_handoff()
            .unwrap();
        assert!(rows <= state.frontier);
        let layer = state.windows.iter().position(Option::is_none).unwrap();
        let (k, v) = &state.layers[layer];
        [k, v]
            .into_iter()
            .flat_map(|t| {
                t.narrow(1, 0, rows)
                    .unwrap()
                    .flatten_all()
                    .unwrap()
                    .to_vec1::<half::bf16>()
                    .unwrap()
                    .into_iter()
                    .map(half::bf16::to_bits)
            })
            .collect()
    }

    fn destination_state(st: &mut BatchM1State, lane: usize) -> Vec<u16> {
        let layers = [LayerType::FullAttention, LayerType::SlidingAttention].map(|kind| {
            st.target
                .config()
                .layer_types
                .iter()
                .position(|v| std::mem::discriminant(v) == std::mem::discriminant(&kind))
                .unwrap()
        });
        let cache = st.batched.as_mut().unwrap().cache_mut(lane).unwrap();
        cache.require_initialized_batch_prefix().unwrap();
        assert_eq!(cache.direct_fp8_prefill_enqueued_calls(), 0);
        layers
            .into_iter()
            .flat_map(|layer| {
                let (k, v) = cache.view(layer, cache.current_len()).unwrap();
                [k, v]
                    .into_iter()
                    .flat_map(|t| {
                        t.flatten_all()
                            .unwrap()
                            .to_vec1::<half::bf16>()
                            .unwrap()
                            .into_iter()
                            .map(half::bf16::to_bits)
                    })
                    .collect::<Vec<_>>()
            })
            .collect()
    }

    fn draft_rows(st: &BatchM1State, lane: usize) -> Vec<u16> {
        st.spec.as_ref().map_or_else(Vec::new, |spec| {
            spec.lanes[lane]
                .ctx
                .initialized_ring_rows_for_test()
                .unwrap()
        })
    }

    fn assert_draft_frontier(st: &BatchM1State, lane: usize, expected: usize) {
        if let Some(spec) = &st.spec {
            assert_eq!(spec.lanes[lane].ctx.len(), expected);
        }
    }

    fn destination_pointers(st: &mut BatchM1State) -> Vec<u64> {
        let Device::Cuda(dev) = &st.device else {
            unreachable!()
        };
        let stream = dev.cuda_stream();
        let mut pointers = Vec::new();
        for lane in 0..2 {
            let cache = st.batched.as_mut().unwrap().cache_mut(lane).unwrap();
            pointers.push(cache.ring_meta_ptr());
            for layer in 0..st.target.config().num_hidden_layers {
                if let Some((k, v, ks, vs)) = cache.layer_fp8_ptrs(layer) {
                    pointers.extend([k, v, ks, vs]);
                } else {
                    let (k, v) = cache.layer_bf16_bufs(layer).unwrap();
                    for tensor in [k, v] {
                        let (storage, layout) = tensor.storage_and_layout();
                        assert_eq!(layout.start_offset(), 0);
                        let candle_core::Storage::Cuda(storage) = &*storage else {
                            unreachable!()
                        };
                        let (pointer, _guard) = storage
                            .as_cuda_slice::<half::bf16>()
                            .unwrap()
                            .device_ptr(&stream);
                        pointers.push(pointer);
                    }
                }
            }
        }
        pointers
    }

    fn turn(
        st: &mut BatchM1State,
        slots: &mut [Option<LaneRun>; 2],
        prefills: &mut [Option<LanePrefill>; 2],
        cursor: &mut usize,
    ) {
        let active: Vec<usize> = slots
            .iter()
            .enumerate()
            .filter_map(|(i, s)| s.as_ref().map(|_| i))
            .collect();
        let speculative = st.spec.is_some();
        let before: Vec<u64> = if speculative {
            st.batched
                .as_ref()
                .unwrap()
                .spec_graph_observation()
                .1
                .to_vec()
        } else {
            st.batched.as_ref().unwrap().replay_counts().to_vec()
        };
        st.interleaved_turn(slots, prefills, cursor).unwrap();
        let (captured, after) = if speculative {
            let (captured, after) = st.batched.as_ref().unwrap().spec_graph_observation();
            (captured.to_vec(), after.to_vec())
        } else {
            (
                st.batched.as_ref().unwrap().captured_masks().to_vec(),
                st.batched.as_ref().unwrap().replay_counts().to_vec(),
            )
        };
        assert!(captured.into_iter().all(|value| value));
        let mask = match active.as_slice() {
            [] => None,
            [0] => Some(0),
            [1] => Some(1),
            [0, 1] => Some(2),
            _ => unreachable!(),
        };
        let mask = mask.map(|index| index + if speculative { 3 } else { 0 });
        for i in 0..after.len() {
            assert_eq!(
                after[i],
                before[i] + u64::from(mask == Some(i)),
                "every ready physical mask must replay its retained graph, not recapture"
            );
        }
        for lane in 0..2 {
            assert_eq!(
                st.batched
                    .as_mut()
                    .unwrap()
                    .cache_mut(lane)
                    .unwrap()
                    .direct_fp8_prefill_enqueued_calls(),
                0
            );
            if let Some(run) = &slots[lane] {
                assert_draft_frontier(st, lane, run.pos);
            }
            if let Some(pending) = &prefills[lane] {
                assert_draft_frontier(st, lane, pending.offset);
            }
        }
    }

    fn drain(
        st: &mut BatchM1State,
        slots: &mut [Option<LaneRun>; 2],
        prefills: &mut [Option<LanePrefill>; 2],
        cursor: &mut usize,
    ) {
        for _ in 0..128 {
            if slots.iter().all(Option::is_none) && prefills.iter().all(Option::is_none) {
                return;
            }
            turn(st, slots, prefills, cursor);
        }
        panic!("bounded scheduler lifecycle did not finish");
    }

    fn file_sha(path: &Path) -> String {
        let result = std::process::Command::new("sha256sum")
            .arg(path)
            .output()
            .unwrap();
        assert!(
            result.status.success(),
            "sha256sum failed: {}",
            path.display()
        );
        String::from_utf8(result.stdout)
            .unwrap()
            .split_whitespace()
            .next()
            .unwrap()
            .to_owned()
    }

    fn run_lifecycle(speculative: bool) {
        assert_eq!(
            std::env::var("NV_LAGUNA_B2_HANDOFF_TEST").as_deref(),
            Ok("1"),
            "explicit real CUDA gate opt-in required"
        );
        assert!(bf16_prefill_handoff_opt_in());
        assert_eq!(
            std::env::var("NV_LAGUNA_BATCH_GRAPH").as_deref(),
            Ok(if speculative { "m2-spec" } else { "m2" })
        );
        validate_bf16_prefill_handoff_selection().unwrap();
        validate_m2_graph_selection().unwrap();
        validate_m2_spec_selection().unwrap();
        let dir = std::path::PathBuf::from(
            std::env::var("NV_LAGUNA_TEST_MODEL_DIR").expect("pinned target directory required"),
        );
        let manifest_path = std::path::PathBuf::from(
            std::env::var("NV_LAGUNA_TEST_MODEL_MANIFEST")
                .expect("pinned target manifest required"),
        );
        let manifest: serde_json::Value =
            serde_json::from_slice(&std::fs::read(&manifest_path).unwrap()).unwrap();
        assert_eq!(
            manifest["revision"].as_str(),
            Some("d32afde8b09af1539b49ff96ff5551c674485f8e")
        );
        assert_eq!(
            file_sha(&dir.join("config.json")),
            "8bba8588c6197aa04be89a405a0af640e1914c6323934f7e0860f7cedc528f4e"
        );
        let weights = manifest["weights"].as_object().unwrap();
        assert_eq!(weights.len(), 5);
        let actual: std::collections::BTreeSet<String> = std::fs::read_dir(&dir)
            .unwrap()
            .map(|p| p.unwrap().file_name().into_string().unwrap())
            .filter(|p| p.ends_with(".safetensors"))
            .collect();
        assert_eq!(actual, weights.keys().cloned().collect());
        for (name, expected) in weights {
            assert_eq!(
                Path::new(name).file_name().unwrap().to_str(),
                Some(name.as_str())
            );
            assert_eq!(
                file_sha(&dir.join(name)),
                expected.as_str().unwrap(),
                "packed target shard changed"
            );
        }
        let raw = std::fs::read_to_string(dir.join("config.json")).unwrap();
        let config = crate::laguna::LagunaConfig::from_hf_json_str(&raw).unwrap();
        let quant = nv_weights::QuantizationConfig::from_hf_json_str(&raw).unwrap();
        assert_eq!(quant.scheme, nv_weights::QuantScheme::Nvfp4);
        let device = Device::new_cuda_with_stream(0)
            .expect("real CUDA device and explicit model stream required");
        let weights = nv_weights::WeightLoader::open_dir(&dir, &device).unwrap();
        let model =
            Arc::new(Laguna::from_loader_quantized(config, &weights, &quant, &device).unwrap());
        drop(weights);
        let chunk = model.prefill_chunk_tokens();
        assert!(
            matches!(chunk, 2048 | 4096 | 8192),
            "the serving handoff gate requires an explicit full prefill chunk"
        );
        let tokenizer = tokenizers::Tokenizer::from_file(dir.join("tokenizer.json")).unwrap();
        let prompt = |seed: &str, depth: usize| {
            let mut ids = tokenizer
                .encode(seed.repeat(depth), true)
                .unwrap()
                .get_ids()
                .to_vec();
            assert!(ids.len() >= depth);
            ids.truncate(depth);
            ids
        };
        let a = prompt(
            "A copper sphere rolls across the wooden table. ",
            3 * chunk + 13,
        );
        let b = prompt("Glaciers shape valleys in high mountains. ", 2 * chunk + 11);
        let c = prompt("Explain how a river changes its path. ", chunk + 17);
        let d = prompt("Rain brings water into the old forest. ", chunk + 5);
        let e = prompt("Light travels through a clear glass prism. ", 2 * chunk + 7);
        let mut st = if speculative {
            let draft_dir = std::path::PathBuf::from(
                std::env::var("NV_LAGUNA_DFLASH_DIR").expect("pinned packed draft source required"),
            );
            assert_eq!(
                file_sha(&draft_dir.join("config.json")),
                "22bdc5fd9fd41dc6945523fd504f1b70809af0f5ab26c09716e33616033ecf9a"
            );
            assert_eq!(
                file_sha(&draft_dir.join("model.safetensors")),
                "67beb7f0dcb8a121ef912782396335cb5103e39f30328decdc83981509d4db84"
            );
            let draft = load_dflash_draft(&draft_dir, &device).unwrap();
            let summary = draft.quantization_summary();
            assert_eq!(
                (
                    summary.packed_linears,
                    summary.excluded_linears,
                    summary.bf16_fallback_linears
                ),
                (36, 5, 0)
            );
            assert_eq!(summary.weight_format, "nvfp4-derived");
            BatchM1State::new_spec(&model, draft, 2, 262144).unwrap()
        } else {
            BatchM1State::new(&model, 2, 262144).unwrap()
        };
        assert!(st.lanes.is_empty() && st.handoff_source.is_some());
        st.warmup().unwrap();
        assert_eq!(source_frontier(&st), (None, 0));
        let replay_start = st.batched.as_ref().unwrap().replay_counts();
        let spec_replay_start =
            speculative.then(|| st.batched.as_ref().unwrap().spec_graph_observation().1);
        let destination_addresses = destination_pointers(&mut st);
        let mut slots: [Option<LaneRun>; 2] = [None, None];
        let mut prefills: [Option<LanePrefill>; 2] = [None, None];
        let mut cursor = 0;
        let (a_job, a_client) = job(&a, 32);
        prefills[0] = st.reserve_prefill(0, a_job);
        turn(&mut st, &mut slots, &mut prefills, &mut cursor);
        assert_eq!(source_frontier(&st), (Some(0), chunk));
        let source_addresses = source_pointers(&st);
        let prior_rows = source_full_prefix(&st, chunk);
        let prior_draft_rows = draft_rows(&st, 0);
        assert_draft_frontier(&st, 0, chunk);
        assert_eq!(st.batched.as_ref().unwrap().current_lengths(), [0, 0]);
        let (nonowner_job, nonowner_client) = job(&b, 16);
        prefills[1] = st.reserve_prefill(1, nonowner_job);
        assert_eq!(source_frontier(&st), (Some(0), chunk));
        assert_eq!(
            source_full_prefix(&st, chunk),
            prior_rows,
            "admitting another lane changed the owner"
        );
        assert_eq!(
            draft_rows(&st, 0),
            prior_draft_rows,
            "admitting the nonowner changed the owner's draft context"
        );
        assert_draft_frontier(&st, 1, 0);
        nonowner_client.cancel();
        turn(&mut st, &mut slots, &mut prefills, &mut cursor);
        assert!(prefills[1].is_none());
        assert_eq!(source_frontier(&st), (Some(0), 2 * chunk));
        assert_eq!(
            source_full_prefix(&st, chunk),
            prior_rows,
            "cancelling the nonowner changed committed owner rows"
        );
        assert!(nonowner_client.tokens().is_empty());
        while prefills[0].is_some() {
            turn(&mut st, &mut slots, &mut prefills, &mut cursor);
        }
        assert_eq!(source_frontier(&st), (None, 0));
        assert_eq!(st.batched.as_ref().unwrap().current_lengths()[0], a.len());
        assert_eq!(a_client.tokens().len(), 1);
        let (owner_job, owner_client) = job(&b, 16);
        prefills[1] = st.reserve_prefill(1, owner_job);
        let ready_tokens = a_client.tokens().len();
        turn(&mut st, &mut slots, &mut prefills, &mut cursor);
        assert_eq!(source_frontier(&st), (Some(1), chunk));
        assert_eq!(
            source_pointers(&st),
            source_addresses,
            "a second BF16 source was substituted"
        );
        assert!(
            (ready_tokens + 1..=ready_tokens + if speculative { 3 } else { 1 })
                .contains(&a_client.tokens().len()),
            "ready decode stalled while the other lane prefills"
        );
        owner_client.cancel();
        turn(&mut st, &mut slots, &mut prefills, &mut cursor);
        assert!(prefills[1].is_none());
        assert_eq!(
            source_frontier(&st),
            (None, 0),
            "owner cancellation must release and reset the source"
        );
        assert_draft_frontier(&st, 1, 0);
        assert_eq!(st.batched.as_ref().unwrap().current_lengths()[1], 0);
        let (c_job, c_client) = job(&c, 32);
        prefills[1] = st.reserve_prefill(1, c_job);
        turn(&mut st, &mut slots, &mut prefills, &mut cursor);
        assert_eq!(
            source_frontier(&st),
            (Some(1), chunk),
            "next acquisition must start at zero"
        );
        assert_eq!(source_pointers(&st), source_addresses);
        while prefills[1].is_some() {
            turn(&mut st, &mut slots, &mut prefills, &mut cursor);
        }
        assert_eq!(source_frontier(&st), (None, 0));
        assert_eq!(st.batched.as_ref().unwrap().current_lengths()[1], c.len());
        let replacement_prefix = destination_state(&mut st, 1);
        let replacement_draft_rows = draft_rows(&st, 1);
        assert_draft_frontier(&st, 1, c.len());
        assert_eq!(
            destination_pointers(&mut st),
            destination_addresses,
            "handoff replaced a captured destination allocation"
        );
        let replacement_anchor = c_client.tokens()[0];
        drain(&mut st, &mut slots, &mut prefills, &mut cursor);
        a_client.assert_complete(32);
        c_client.assert_complete(32);
        let (mut cold_job, _) = job(&c, 1);
        assert_eq!(
            st.prefill_lane(1, &mut cold_job).unwrap(),
            replacement_anchor
        );
        assert_eq!(destination_state(&mut st, 1), replacement_prefix,
            "cancel/replacement and an uninterrupted same-arm cold prefill must commit the same full/sliding rows");
        assert_eq!(draft_rows(&st, 1), replacement_draft_rows,
            "cancel/replacement and uninterrupted cold prefill must commit the same auxiliary-derived draft ring");
        let mut previous = None;
        for _ in 0..2 {
            let (d_job, d_client) = job(&d, 8);
            let (e_job, e_client) = job(&e, 8);
            prefills[0] = st.reserve_prefill(0, d_job);
            prefills[1] = st.reserve_prefill(1, e_job);
            drain(&mut st, &mut slots, &mut prefills, &mut cursor);
            d_client.assert_complete(8);
            e_client.assert_complete(8);
            assert_eq!(source_frontier(&st), (None, 0));
            let output = [d_client.tokens(), e_client.tokens()];
            if let Some(previous) = previous.as_ref() {
                assert_eq!(&output, previous, "identical handoff arm and physical-mask schedule must repeat across cache refill");
            }
            previous = Some(output);
            for lane in 0..2 {
                destination_state(&mut st, lane);
            }
            assert_eq!(destination_pointers(&mut st), destination_addresses);
        }
        let replays = if let Some(before) = spec_replay_start {
            let after = st.batched.as_ref().unwrap().spec_graph_observation().1;
            assert!(after[3..]
                .iter()
                .zip(&before[3..])
                .all(|(after, before)| after > before));
            st.warm_spec_masks().unwrap();
            let (captured, after_refill) = st.batched.as_ref().unwrap().spec_graph_observation();
            assert_eq!(captured, [true; 6]);
            assert!(
                after_refill
                    .iter()
                    .zip(after)
                    .all(|(after, before)| *after > before),
                "refilled destinations must replay all retained M1/M2/M3/M6 masks"
            );
            st.batched
                .as_ref()
                .unwrap()
                .report_spec("shared-bf16-handoff-lifecycle-passed");
            after_refill.to_vec()
        } else {
            let after = st.batched.as_ref().unwrap().replay_counts();
            assert!(after
                .iter()
                .zip(replay_start)
                .all(|(after, before)| *after > before));
            st.batched
                .as_ref()
                .unwrap()
                .report("shared-bf16-handoff-lifecycle-passed");
            after.to_vec()
        };
        assert_eq!(destination_pointers(&mut st), destination_addresses);
        let (orphan_job, _) = job(&c, 8);
        prefills[0] = st.reserve_prefill(0, orphan_job);
        turn(&mut st, &mut slots, &mut prefills, &mut cursor);
        assert_eq!(source_frontier(&st), (Some(0), chunk));
        prefills[0].take();
        assert!(
            st.interleaved_turn(&mut slots, &mut prefills, &mut cursor)
                .is_err(),
            "orphaned ownership must be refused, not handed to another prompt"
        );
        assert_eq!(
            source_frontier(&st),
            (None, 0),
            "error must release the shared source before worker retirement"
        );
        drop(st);
        device.synchronize().unwrap();
        let marker = if speculative {
            "LAGUNA-M6-HANDOFF-LIFECYCLE-GATE"
        } else {
            "LAGUNA-B2-HANDOFF-LIFECYCLE-GATE"
        };
        eprintln!("{marker} passed cap=262144 chunk={chunk} source_caches=1 destination_caches=2 k={} nonowner_cancel=preserved owner_cancel=reset draft_context=validated ready_decode=progressed direct_fp8_prefill_calls=0 destination_pointers=retained error_release=true replays={replays:?}", if speculative { 2 } else { 0 });
    }

    #[test]
    #[ignore]
    fn real_shared_bf16_prefill_handoff_preserves_owners_and_retained_b2_masks() {
        run_lifecycle(false);
    }

    #[test]
    #[ignore]
    fn real_shared_bf16_prefill_handoff_preserves_packed_k2_context_and_retained_m6_masks() {
        run_lifecycle(true);
    }
}
