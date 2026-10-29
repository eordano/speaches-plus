use std::sync::{Arc, Mutex};

use anyhow::Result;

use super::{SpecServeEvent, SpecServeJob, SpecState};
use crate::laguna_dflash::prefix::CtxSnapshot;
use crate::laguna_prefix::{
    checkpoint_position, checkpoint_preserves_sliding_layout, common_prefix_len,
    lru_bytes_from_env, lru_entries_from_env, prefix_reuse_enabled, CheckpointLru,
};
use crate::laguna_prefix_diagnostics as diagnostics;
use crate::laguna_step_graph::prefix::{TargetFullSnapshot, TargetSlidingSnapshot};
use crate::laguna_step_graph::StepCache;

pub(super) struct CkptState {
    sliding: TargetSlidingSnapshot,
    ctx: Option<CtxSnapshot>,
    full: Option<Arc<TargetFullSnapshot>>,
    draft_rope_theta: Option<u32>,
}

struct Resident {
    ckpt: Option<u64>,

    tokens: Vec<u32>,
    frontier: usize,
    complete: bool,
    draft_rope_theta: Option<u32>,
}

pub(super) struct PrefixState {
    lru: CheckpointLru<CkptState>,
    configured: bool,
    max_entries: usize,
    max_bytes: usize,
    resident: Option<Resident>,
    resume_from: usize,
    stats: bool,
    pub(super) reused_total: usize,
}

impl PrefixState {
    pub(super) fn from_env() -> Self {
        let enabled = prefix_reuse_enabled();
        let entries = lru_entries_from_env();
        let bytes = lru_bytes_from_env();
        eprintln!(
            "[laguna_prefix] reuse={} lru_entries={entries} lru_bytes={bytes} \
             (NV_LAGUNA_PREFIX_REUSE=0 / NV_LAGUNA_PREFIX_LRU / NV_LAGUNA_PREFIX_LRU_BYTES)",
            if enabled { "on" } else { "off" }
        );
        Self {
            lru: CheckpointLru::with_shared_allocation(entries, bytes, |state: &CkptState| {
                state
                    .full
                    .as_ref()
                    .map(|full| (Arc::as_ptr(full) as usize, full.bytes()))
            }),
            configured: entries > 0 && bytes > 0,
            max_entries: entries,
            max_bytes: bytes,
            resident: None,
            resume_from: 0,
            stats: nv_config::env_arm_opt_in_eq_some_1_reread_every_call("NV_LAGUNA_SERVE_STATS"),
            reused_total: 0,
        }
    }
}

#[derive(Default)]
struct Recorded {
    emitted: Vec<u32>,
    disconnected: bool,
}

struct Recorder(Arc<Mutex<Recorded>>);

impl Recorder {
    fn attach(job: &mut SpecServeJob) -> Self {
        let sink = Arc::new(Mutex::new(Recorded::default()));
        let seen = Arc::clone(&sink);
        let mut inner = std::mem::replace(&mut job.emit, Box::new(|_| true));
        job.emit = Box::new(move |ev| {
            if let SpecServeEvent::Tokens(tokens) = &ev {
                if !tokens.is_empty() {
                    seen.lock()
                        .expect("recorder")
                        .emitted
                        .extend_from_slice(tokens);
                }
            }
            let alive = inner(ev);
            if !alive {
                seen.lock().expect("recorder").disconnected = true;
            }
            alive
        });
        Self(sink)
    }

    fn take(&self) -> Recorded {
        std::mem::take(&mut *self.0.lock().expect("recorder"))
    }
}

impl<'m> SpecState<'m> {
    pub(super) fn prefix_worker_started(&self) {
        let proof = self.draft.map(|draft| draft.quantization_summary());
        let kv_kind = match self.step.kv() {
            StepCache::Bf16(_) => "bf16",
            StepCache::Fp8(cache)
                if cache.scale_policy() == crate::laguna_fp8::LagunaFp8ScalePolicy::FixedOne =>
            {
                "fp8-fixed-one"
            }
            StepCache::Fp8(_) => "fp8-other",
        };
        diagnostics::worker_started(diagnostics::WorkerInfo {
            kv_kind,
            prefix_supported: self.handoff_source.is_none()
                && self.step.kv().prefix_supported()
                && self.ctx.as_ref().is_none_or(|cache| cache.has_ring()),
            max_seq: self.max_seq,
            chunk: self.target.prefill_chunk_tokens(),
            max_entries: self.prefix.max_entries,
            max_bytes: self.prefix.max_bytes,
            draft_enabled: self.draft.is_some(),
            draft_packed_linears: proof.as_ref().map_or(0, |proof| proof.packed_linears),
            draft_fallback_linears: proof
                .as_ref()
                .map_or(0, |proof| proof.bf16_fallback_linears),
        });
    }

    pub(super) fn serve_one(&mut self, job: &mut SpecServeJob) -> Result<()> {
        let reuse = self.reuse_possible();
        if !reuse && !self.prefix.stats {
            let out = self.serve_one_inner(job);
            if out.is_ok() {
                (job.emit)(SpecServeEvent::Done);
            }
            return out;
        }
        diagnostics::request_started(job.prompt_ids.len());
        let rec = Recorder::attach(job);
        let out = self.serve_one_inner(job);
        if out.is_ok() {
            (job.emit)(SpecServeEvent::Done);
        }
        let recorded = rec.take();
        let outcome = if out.is_err() {
            "failed"
        } else if recorded.disconnected {
            "cancelled"
        } else {
            "completed"
        };
        if reuse && out.is_ok() && !recorded.disconnected {
            self.prefix_finish(job, &recorded.emitted);
        } else {
            self.invalidate_resident();
        }
        diagnostics::state(
            self.prefix.lru.len(),
            self.prefix.lru.bytes(),
            self.prefix
                .resident
                .as_ref()
                .map_or(0, |resident| resident.tokens.len()),
            self.prefix
                .resident
                .as_ref()
                .map_or(0, |resident| resident.frontier),
        );
        diagnostics::request_finished(outcome, &recorded.emitted);
        out
    }

    fn reuse_possible(&self) -> bool {
        self.handoff_source.is_none()
            && prefix_reuse_enabled()
            && (!self.step.kv().is_fp8()
                || nv_config::env_arm_opt_in_eq_some_1_reread_every_call("NV_LAGUNA_PREFIX_REUSE"))
            && self.prefix.configured
            && self.step.kv().prefix_supported()
            && self.ctx.as_ref().is_none_or(|cache| cache.has_ring())
    }

    fn invalidate_resident(&mut self) {
        self.prefix.resident = None;
        self.prefix.resume_from = 0;
        let stale: Vec<u64> = self
            .prefix
            .lru
            .iter()
            .filter(|entry| entry.state.full.is_none())
            .map(|entry| entry.id)
            .collect();
        for id in stale {
            self.prefix.lru.remove(id);
        }
    }

    fn cold(&mut self) {
        self.step.kv_mut().reset();
        if let Some(c) = self.ctx.as_mut() {
            c.reset();
        }
        self.invalidate_resident();
    }

    pub(super) fn prefix_prepare(&mut self, job: &mut SpecServeJob) -> Result<usize> {
        if !self.reuse_possible() {
            self.cold();
            return Ok(0);
        }
        let prompt = &job.prompt_ids;
        let n = prompt.len();
        let chunk = self.target.prefill_chunk_tokens();
        let draft_rope_theta = self.draft.map(|draft| draft.active_rope_theta().to_bits());

        if let Some(r) = self.prefix.resident.as_ref() {
            let kv_len = self.step.kv().current_len();
            let ctx_ok = self.ctx.as_ref().is_none_or(|c| c.len() == r.frontier);
            if nv_config::env_arm_opt_in_eq_some_1_reread_every_call(
                "NV_LAGUNA_PREFIX_EXACT_EXTEND",
            ) && r.complete
                && r.draft_rope_theta == draft_rope_theta
                && r.frontier > 0
                && r.frontier < n
                && kv_len == r.frontier
                && ctx_ok
                && common_prefix_len(&r.tokens, prompt) >= r.frontier
            {
                let at = r.frontier;
                if let Some(id) = r.ckpt {
                    self.prefix.lru.touch(id);
                }
                return Ok(self.resumed(job, at, "extend"));
            }
        }

        let resident_tokens = self
            .prefix
            .resident
            .as_ref()
            .map(|resident| resident.tokens.as_slice());
        let kv = self.step.kv();
        let hit = self.prefix.lru.select_where(prompt, |entry| {
            chunk > 0
                && entry.pos % chunk == 0
                && checkpoint_preserves_sliding_layout(
                    entry.pos,
                    chunk,
                    self.target.config().sliding_window.max(1),
                    self.max_seq,
                    self.target.next_prefill_chunk_len(n - entry.pos),
                )
                && entry.state.draft_rope_theta == draft_rope_theta
                && (entry.state.full.is_some()
                    || resident_tokens.is_some_and(|tokens| {
                        common_prefix_len(tokens, &entry.tokens) >= entry.pos
                            && kv.prefix_holds_full_rows(entry.pos)
                    }))
        });
        if let Some(hit) = hit {
            let resident = self.prefix.resident.as_ref().is_some_and(|current| {
                common_prefix_len(&current.tokens, prompt) >= hit.pos
                    && self.step.kv().prefix_holds_full_rows(hit.pos)
            });
            if !resident {
                self.evict_resident(Some(hit.id)).map_err(|error| {
                    error
                        .context("engine busy: Laguna prefix checkpoint storage failed; try later.")
                })?;
            }
            let outcome = {
                let Some(entry) = self.prefix.lru.get(hit.id) else {
                    self.cold();
                    return Ok(0);
                };
                let full = if resident {
                    None
                } else {
                    entry.state.full.as_deref()
                };
                if entry.state.sliding.pos() != hit.pos
                    || entry
                        .state
                        .ctx
                        .as_ref()
                        .is_some_and(|snapshot| snapshot.pos != hit.pos)
                {
                    Err(anyhow::anyhow!(
                        "checkpoint target and draft positions disagree"
                    ))
                } else if !resident && full.is_none() {
                    Err(anyhow::anyhow!(
                        "checkpoint {} has no full rows off-device",
                        hit.id
                    ))
                } else {
                    let draft_valid = match (self.ctx.as_ref(), entry.state.ctx.as_ref()) {
                        (Some(ctx), Some(snapshot)) => ctx.validate_snapshot(snapshot),
                        (Some(_), None) => Err(anyhow::anyhow!("checkpoint has no draft context")),
                        (None, _) => Ok(()),
                    };
                    draft_valid
                        .and_then(|()| {
                            self.step
                                .kv_mut()
                                .prefix_rewind_to(&entry.state.sliding, full)
                        })
                        .and_then(|()| match (self.ctx.as_mut(), entry.state.ctx.as_ref()) {
                            (Some(ctx), Some(snap)) => ctx.restore(snap),
                            (Some(_), None) => {
                                Err(anyhow::anyhow!("checkpoint has no draft context"))
                            }
                            (None, _) => Ok(()),
                        })
                }
            };
            match outcome {
                Ok(()) => {
                    self.prefix.lru.touch(hit.id);
                    let tokens = prompt[..hit.pos].to_vec();
                    self.prefix.resident = Some(Resident {
                        ckpt: Some(hit.id),
                        tokens,
                        frontier: hit.pos,
                        complete: false,
                        draft_rope_theta,
                    });
                    return Ok(self.resumed(
                        job,
                        hit.pos,
                        if resident { "resident" } else { "restored" },
                    ));
                }
                Err(error) => {
                    self.prefix.lru.remove(hit.id);
                    self.invalidate_resident();
                    return Err(
                        error.context("engine busy: Laguna prefix restore failed; try later.")
                    );
                }
            }
        }

        self.evict_resident(None).map_err(|error| {
            error.context("engine busy: Laguna prefix checkpoint storage failed; try later.")
        })?;
        self.cold();
        Ok(0)
    }

    fn resumed(&mut self, job: &mut SpecServeJob, at: usize, how: &'static str) -> usize {
        self.prefix.resume_from = at;
        self.prefix.reused_total += at;
        diagnostics::selected(how, at);
        if self.prefix.stats {
            eprintln!(
                "[laguna_prefix] {how}: prefill resumes at {at} of {} prompt tokens",
                job.prompt_ids.len()
            );
        }
        (job.emit)(SpecServeEvent::PromptCached(at));
        at
    }

    fn evict_resident(&mut self, preserve: Option<u64>) -> Result<()> {
        let Some(resident) = self.prefix.resident.take() else {
            return Ok(());
        };
        let mut candidates: Vec<(u64, usize)> = self
            .prefix
            .lru
            .iter()
            .filter(|entry| {
                entry.state.full.is_none()
                    && entry.state.draft_rope_theta == resident.draft_rope_theta
                    && common_prefix_len(&resident.tokens, &entry.tokens) >= entry.pos
                    && self.step.kv().prefix_holds_full_rows(entry.pos)
            })
            .map(|entry| (entry.id, entry.pos))
            .collect();
        candidates.sort_unstable_by_key(|entry| std::cmp::Reverse(entry.1));
        for &(id, pos) in &candidates {
            let mut protected = vec![id];
            if let Some(keep) = preserve {
                if !protected.contains(&keep) {
                    protected.push(keep);
                }
            }
            let bytes = self.step.kv().prefix_full_bytes(pos)?;
            if !self.prefix.lru.can_admit(0, bytes, &protected) {
                diagnostics::budget_refused("full", bytes);
                continue;
            }
            let kv = self.step.kv();
            let saved = self.prefix.lru.attach_with(
                &protected,
                bytes,
                || -> Result<_> {
                    diagnostics::copy_started("full", bytes);
                    let full = Arc::new(kv.prefix_snapshot_full(pos)?);
                    let actual = full.bytes();
                    Ok((full, actual))
                },
                |entry_id, state, full| {
                    if entry_id == id && state.full.is_none() {
                        state.full = Some(Arc::clone(full));
                    }
                },
            )?;
            if saved {
                let image = self
                    .prefix
                    .lru
                    .get(id)
                    .and_then(|entry| entry.state.full.clone());
                if let Some(image) = image {
                    let ancestors: Vec<u64> = self
                        .prefix
                        .lru
                        .iter()
                        .filter(|entry| {
                            entry.state.full.is_none()
                                && entry.pos <= pos
                                && entry.state.draft_rope_theta == resident.draft_rope_theta
                                && common_prefix_len(&resident.tokens, &entry.tokens) >= entry.pos
                        })
                        .map(|entry| entry.id)
                        .collect();
                    for ancestor in ancestors {
                        if let Some(entry) = self.prefix.lru.get_mut(ancestor) {
                            entry.state.full = Some(Arc::clone(&image));
                        }
                    }
                }
            }
            if self.prefix.stats {
                eprintln!("[laguna_prefix] eviction checkpoint={id} pos={pos} admitted={saved} full_bytes={bytes} retained_bytes={}", self.prefix.lru.bytes());
            }
            if saved {
                break;
            }
        }
        let stale: Vec<u64> = self
            .prefix
            .lru
            .iter()
            .filter(|entry| entry.state.full.is_none())
            .map(|entry| entry.id)
            .collect();
        for id in stale {
            self.prefix.lru.remove(id);
        }
        Ok(())
    }

    pub(super) fn prefix_checkpoint_hook(
        &mut self,
        job: &SpecServeJob,
        offset: usize,
    ) -> Result<()> {
        if !self.reuse_possible() {
            return Ok(());
        }
        let n = job.prompt_ids.len();
        let chunk = self.target.prefill_chunk_tokens();
        let pos = checkpoint_position(n, chunk);
        if pos == 0 || offset != pos || self.prefix.resume_from == pos {
            return Ok(());
        }
        let kv = self.step.kv();
        anyhow::ensure!(
            kv.current_len() == pos,
            "laguna prefix: cache at {} at boundary {pos}",
            kv.current_len()
        );
        let ctx_cache = self.ctx.as_ref();
        anyhow::ensure!(
            ctx_cache.is_none_or(|cache| cache.len() == pos),
            "laguna prefix: draft context is not at target checkpoint {pos}"
        );
        let draft_rope_theta = self.draft.map(|draft| draft.active_rope_theta().to_bits());
        let bytes = kv
            .prefix_sliding_bytes()?
            .checked_add(ctx_cache.map_or(Ok(0), |cache| cache.snapshot_bytes())?)
            .ok_or_else(|| anyhow::anyhow!("Laguna prefix checkpoint size overflow"))?;
        let id = self.prefix.lru.insert_with(
            &job.prompt_ids[..pos],
            pos,
            bytes,
            &[],
            || -> Result<_> {
                diagnostics::copy_started("windows", bytes);
                let sliding = kv.prefix_snapshot_sliding()?;
                let ctx = ctx_cache.map(|cache| cache.snapshot()).transpose()?;
                let actual = sliding
                    .bytes()
                    .checked_add(ctx.as_ref().map_or(0, |cache| cache.bytes))
                    .ok_or_else(|| anyhow::anyhow!("Laguna prefix checkpoint size overflow"))?;
                Ok((
                    CkptState {
                        sliding,
                        ctx,
                        full: None,
                        draft_rope_theta,
                    },
                    actual,
                ))
            },
        )?;
        if id.is_none() {
            diagnostics::budget_refused("windows", bytes);
        }
        if self.prefix.stats {
            eprintln!(
                "[laguna_prefix] checkpoint {id:?} at {pos} of {n}: {bytes} bytes of window rows, lru {} entries {} bytes",
                self.prefix.lru.len(),
                self.prefix.lru.bytes()
            );
        }
        self.prefix.resident = Some(Resident {
            ckpt: id,
            tokens: job.prompt_ids[..pos].to_vec(),
            frontier: pos,
            complete: false,
            draft_rope_theta,
        });
        Ok(())
    }

    fn prefix_finish(&mut self, job: &mut SpecServeJob, emitted: &[u32]) {
        let kv_len = self.step.kv().current_len();
        let mut tokens = std::mem::take(&mut job.prompt_ids);
        tokens.extend_from_slice(emitted);
        let complete = !emitted.is_empty()
            && kv_len == tokens.len().saturating_sub(1)
            && self.ctx.as_ref().is_none_or(|cache| cache.len() == kv_len);
        let checkpoint = self
            .prefix
            .resident
            .as_ref()
            .and_then(|resident| resident.ckpt);
        self.prefix.resident = Some(Resident {
            ckpt: checkpoint,
            tokens,
            frontier: kv_len,
            complete,
            draft_rope_theta: self.draft.map(|draft| draft.active_rope_theta().to_bits()),
        });
    }
}
