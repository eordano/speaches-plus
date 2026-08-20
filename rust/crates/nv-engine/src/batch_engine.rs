use crate::scheduler::{BatchKind, Scheduler, SchedulerConfig, StepFailure};
use crate::sequence::{FinishReason, Sequence};
use anyhow::{bail, Result};
use std::collections::HashMap;
use tokio::sync::mpsc;

#[derive(Clone, Debug, Default)]
pub struct SamplingConfig {
    pub temperature: f32,
    pub top_p: Option<f32>,
    pub top_k: Option<u32>,
    pub min_p: Option<f32>,
    pub seed: Option<u64>,
    pub presence_penalty: Option<f32>,
    pub frequency_penalty: Option<f32>,
    pub repetition_penalty: Option<f32>,
}

#[derive(Debug)]
pub struct GenRequest {
    pub prompt_tokens: Vec<u32>,
    pub max_new_tokens: usize,
    pub eos_token_ids: Vec<u32>,
    pub sampling: SamplingConfig,
    pub reply: mpsc::Sender<EngineEvent>,
}

#[derive(Debug, Clone)]
pub enum EngineEvent {
    Started {
        seq_id: u64,
        prompt_tokens: u32,
    },
    Token {
        seq_id: u64,
        token: u32,
    },
    Done {
        seq_id: u64,
        reason: FinishReason,
        completion_tokens: u32,
    },
    Error {
        seq_id: u64,
        message: String,
    },
}

#[derive(Clone, Debug)]
pub struct SeqInput {
    pub seq_id: u64,
    pub position: usize,
    pub token: u32,
    pub block_table: Vec<u32>,
}

#[derive(Clone, Debug)]
pub struct PrefillInput {
    pub seq_id: u64,
    pub tokens: Vec<u32>,
    pub block_table: Vec<u32>,
}

#[derive(Clone, Debug)]
pub struct PrefillChunk {
    pub seq_id: u64,
    pub tokens: Vec<u32>,
    pub chunk_start: usize,
    pub is_final: bool,
    pub block_table: Vec<u32>,
}

pub trait BatchStepper: Send {
    fn on_admit(&mut self, _seq_id: u64, _sampling: &SamplingConfig) {}

    fn reuses_cached_prefix(&self) -> bool {
        false
    }

    fn prefill(&mut self, items: &[PrefillInput]) -> Result<Vec<StepResult>>;

    fn decode(&mut self, items: &[SeqInput]) -> Result<Vec<StepResult>>;

    fn step_mixed(
        &mut self,
        chunks: &[PrefillChunk],
        decodes: &[SeqInput],
    ) -> Result<Vec<StepResult>> {
        let final_prefills: Vec<PrefillInput> = chunks
            .iter()
            .filter(|c| c.is_final)
            .map(|c| PrefillInput {
                seq_id: c.seq_id,
                tokens: c.tokens.clone(),
                block_table: c.block_table.clone(),
            })
            .collect();
        let mut results = Vec::with_capacity(final_prefills.len() + decodes.len());
        if !final_prefills.is_empty() {
            results.extend(self.prefill(&final_prefills)?);
        }
        if !decodes.is_empty() {
            results.extend(self.decode(decodes)?);
        }
        Ok(results)
    }

    fn release(&mut self, _seq_id: u64) {}
}

#[derive(Clone, Debug)]
pub struct StepResult {
    pub seq_id: u64,
    pub token: u32,
}

struct ReqMeta {
    reply: mpsc::Sender<EngineEvent>,
    completion_tokens: u32,
    started: bool,
}

pub fn max_waiting_for(max_batch_size: usize, env_override: Option<usize>) -> usize {
    match env_override {
        Some(n) if n > 0 => n,
        _ => (max_batch_size * 4).max(16),
    }
}

fn step_prof_every() -> usize {
    static N: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *N.get_or_init(|| {
        std::env::var("NV_BATCH_STEP_PROF")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            .unwrap_or(0)
    })
}

pub struct BatchEngine<S: BatchStepper> {
    scheduler: Scheduler,
    stepper: S,
    meta: HashMap<u64, ReqMeta>,
    next_seq_id: u64,
    max_waiting: usize,
    prof_steps: u64,
}

impl<S: BatchStepper> BatchEngine<S> {
    pub fn new(config: SchedulerConfig, stepper: S) -> Self {
        let max_waiting = max_waiting_for(
            config.max_batch_size,
            std::env::var("NV_BATCH_MAX_WAITING")
                .ok()
                .and_then(|s| s.parse::<usize>().ok()),
        );
        let mut scheduler = Scheduler::new(config);
        scheduler.prefix_cache_reuse = stepper.reuses_cached_prefix();
        Self {
            scheduler,
            stepper,
            meta: HashMap::new(),
            next_seq_id: 1,
            max_waiting,
            prof_steps: 0,
        }
    }

    pub fn set_max_waiting(&mut self, n: usize) {
        self.max_waiting = n.max(1);
    }

    pub fn has_work(&self) -> bool {
        self.scheduler.has_work() || !self.meta.is_empty()
    }

    pub fn admit(&mut self, req: GenRequest) -> u64 {
        let seq_id = self.next_seq_id;
        self.next_seq_id += 1;

        let capacity_tokens = self
            .scheduler
            .config
            .num_blocks
            .saturating_mul(self.scheduler.config.block_size);
        if req.prompt_tokens.len() + 1 > capacity_tokens {
            let _ = req.reply.try_send(EngineEvent::Error {
                seq_id,
                message: format!(
                    "prompt of {} tokens exceeds the engine's KV capacity of {} tokens \
                     ({} blocks x {}); it can never be scheduled",
                    req.prompt_tokens.len(),
                    capacity_tokens,
                    self.scheduler.config.num_blocks,
                    self.scheduler.config.block_size
                ),
            });
            let _ = req.reply.try_send(EngineEvent::Done {
                seq_id,
                reason: FinishReason::Aborted,
                completion_tokens: 0,
            });
            return seq_id;
        }
        if self.scheduler.waiting.len() >= self.max_waiting {
            let _ = req.reply.try_send(EngineEvent::Error {
                seq_id,
                message: format!(
                    "engine overloaded: {} requests already waiting (cap {}); retry later \
                     or raise NV_BATCH_MAX_WAITING",
                    self.scheduler.waiting.len(),
                    self.max_waiting
                ),
            });
            let _ = req.reply.try_send(EngineEvent::Done {
                seq_id,
                reason: FinishReason::Aborted,
                completion_tokens: 0,
            });
            return seq_id;
        }
        let seq = Sequence::new(seq_id, req.prompt_tokens, req.max_new_tokens);
        self.stepper.on_admit(seq_id, &req.sampling);
        self.scheduler.enqueue_with_eos_ids(seq, req.eos_token_ids);
        self.meta.insert(
            seq_id,
            ReqMeta {
                reply: req.reply,
                completion_tokens: 0,
                started: false,
            },
        );
        seq_id
    }

    pub fn abort(&mut self, seq_id: u64) {
        self.scheduler.abort(seq_id);
        self.stepper.release(seq_id);
    }

    pub fn fail_all(&mut self, message: &str) -> usize {
        let ids: Vec<u64> = self.meta.keys().copied().collect();
        for sid in &ids {
            if let Some(m) = self.meta.get(sid) {
                let _ = m.reply.try_send(EngineEvent::Error {
                    seq_id: *sid,
                    message: message.to_string(),
                });
            }
            self.scheduler.abort(*sid);
            self.stepper.release(*sid);
        }

        self.meta.clear();
        self.drain_finished();
        ids.len()
    }

    pub fn step(&mut self) -> Result<()> {
        self.drain_finished();
        if !self.scheduler.has_work() {
            return Ok(());
        }
        let batch = self.scheduler.step()?;
        if batch.seq_ids.is_empty() {
            self.drain_finished();
            return Ok(());
        }

        if batch.kind == BatchKind::Verify {
            bail!("BatchEngine does not yet drive speculative verify batches");
        }

        let mut chunks: Vec<PrefillChunk> = Vec::new();
        let mut decodes: Vec<SeqInput> = Vec::new();
        let mut producing_ids: Vec<u64> = Vec::new();
        let mut started: Vec<(u64, u32)> = Vec::new();
        for item in &batch.items {
            let sid = item.seq_id;
            let seq = self
                .scheduler
                .running_seq(sid)
                .ok_or_else(|| anyhow::anyhow!("step: seq {sid} not running"))?;
            if item.is_prefill {
                let computed = seq.num_computed_tokens;
                let mut tokens = seq.tokens();
                tokens.truncate(computed + item.num_scheduled_tokens);
                chunks.push(PrefillChunk {
                    seq_id: sid,
                    tokens,
                    chunk_start: computed,
                    is_final: item.is_final_prefill_chunk,
                    block_table: seq.block_table.clone(),
                });
                started.push((sid, seq.prompt.len() as u32));
                if item.is_final_prefill_chunk {
                    producing_ids.push(sid);
                }
            } else {
                let token = seq
                    .last_token()
                    .ok_or_else(|| anyhow::anyhow!("decode: seq {sid} has no token"))?;
                decodes.push(SeqInput {
                    seq_id: sid,
                    position: seq.total_len() - 1,
                    token,
                    block_table: seq.block_table.clone(),
                });
                producing_ids.push(sid);
            }
        }
        for (sid, prompt_tokens) in started {
            self.emit_started(sid, prompt_tokens);
        }

        let prof_every = step_prof_every();
        let t0 = (prof_every > 0).then(std::time::Instant::now);

        let results = self.stepper.step_mixed(&chunks, &decodes)?;
        let tokens = Self::order_tokens(&producing_ids, results)?;
        let failures = self.scheduler.complete_step(&tokens)?;
        self.emit_tokens(&producing_ids, &tokens, &failures);

        if let Some(t0) = t0 {
            self.prof_steps += 1;
            if self.prof_steps.is_multiple_of(prof_every as u64) {
                let prefill_tokens: usize = chunks.iter().map(|c| c.tokens.len()).sum();
                eprintln!(
                    "[batch-step] n={} kind={:?} prefill={} prefill_tok={} decode={} running={} waiting={} blocks_free={}/{} max_seqs={} ms={:.2}",
                    self.prof_steps,
                    batch.kind,
                    chunks.len(),
                    prefill_tokens,
                    decodes.len(),
                    self.scheduler.running.len(),
                    self.scheduler.waiting.len(),
                    self.scheduler.block_manager.num_free(),
                    self.scheduler.config.num_blocks,
                    self.scheduler.config.max_batch_size,
                    t0.elapsed().as_secs_f64() * 1e3,
                );
            }
        }

        self.drain_finished();
        Ok(())
    }

    fn order_tokens(seq_ids: &[u64], results: Vec<StepResult>) -> Result<Vec<u32>> {
        if results.len() != seq_ids.len() {
            bail!(
                "stepper returned {} results for {} sequences",
                results.len(),
                seq_ids.len()
            );
        }
        let mut by_id: HashMap<u64, u32> = HashMap::with_capacity(results.len());
        for r in results {
            by_id.insert(r.seq_id, r.token);
        }
        let mut out = Vec::with_capacity(seq_ids.len());
        for sid in seq_ids {
            let tok = by_id
                .get(sid)
                .copied()
                .ok_or_else(|| anyhow::anyhow!("stepper missing result for seq {sid}"))?;
            out.push(tok);
        }
        Ok(out)
    }

    fn emit_started(&mut self, seq_id: u64, prompt_tokens: u32) {
        if let Some(m) = self.meta.get_mut(&seq_id) {
            if !m.started {
                m.started = true;
                let _ = m.reply.try_send(EngineEvent::Started {
                    seq_id,
                    prompt_tokens,
                });
            }
        }
    }

    fn emit_tokens(&mut self, seq_ids: &[u64], tokens: &[u32], failures: &[StepFailure]) {
        for (sid, &tok) in seq_ids.iter().zip(tokens.iter()) {
            if let Some(f) = failures.iter().find(|f| f.seq_id == *sid) {
                if let Some(m) = self.meta.get_mut(sid) {
                    let _ = m.reply.try_send(EngineEvent::Error {
                        seq_id: *sid,
                        message: f.message.clone(),
                    });
                }
                continue;
            }
            if let Some(m) = self.meta.get_mut(sid) {
                m.completion_tokens += 1;
                let _ = m.reply.try_send(EngineEvent::Token {
                    seq_id: *sid,
                    token: tok,
                });
            }
        }
    }

    fn drain_finished(&mut self) {
        let finished = self.scheduler.drain_finished();
        for seq in finished {
            self.stepper.release(seq.id);
            if let Some(m) = self.meta.remove(&seq.id) {
                let reason = seq.finish_reason.unwrap_or(FinishReason::Aborted);
                let _ = m.reply.try_send(EngineEvent::Done {
                    seq_id: seq.id,
                    reason,
                    completion_tokens: m.completion_tokens,
                });
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct CountingStepper {
        next: u32,
        prefilled: Vec<usize>,
        decoded: Vec<usize>,
    }

    impl CountingStepper {
        fn new() -> Self {
            Self {
                next: 1000,
                prefilled: Vec::new(),
                decoded: Vec::new(),
            }
        }
    }

    impl BatchStepper for CountingStepper {
        fn prefill(&mut self, items: &[PrefillInput]) -> Result<Vec<StepResult>> {
            self.prefilled.push(items.len());
            Ok(items
                .iter()
                .map(|it| {
                    let token = self.next;
                    self.next += 1;
                    StepResult {
                        seq_id: it.seq_id,
                        token,
                    }
                })
                .collect())
        }

        fn decode(&mut self, items: &[SeqInput]) -> Result<Vec<StepResult>> {
            self.decoded.push(items.len());
            Ok(items
                .iter()
                .map(|it| {
                    let token = self.next;
                    self.next += 1;
                    StepResult {
                        seq_id: it.seq_id,
                        token,
                    }
                })
                .collect())
        }
    }

    fn cfg() -> SchedulerConfig {
        SchedulerConfig {
            max_batch_size: 8,
            max_batched_tokens: 4096,
            block_size: 4,
            num_blocks: 256,
        }
    }

    fn collect(rx: &mut mpsc::Receiver<EngineEvent>) -> Vec<EngineEvent> {
        let mut out = Vec::new();
        while let Ok(ev) = rx.try_recv() {
            out.push(ev);
        }
        out
    }

    #[test]
    fn abort_settles_the_engine_and_notifies_the_client() {
        let mut engine = BatchEngine::new(cfg(), CountingStepper::new());
        let (tx, mut rx) = mpsc::channel(64);
        let id = engine.admit(GenRequest {
            prompt_tokens: vec![1, 2, 3, 4, 5],
            max_new_tokens: 64,
            eos_token_ids: vec![],
            sampling: SamplingConfig::default(),
            reply: tx,
        });
        engine.step().unwrap();

        engine.abort(id);

        let mut steps = 0;
        while engine.has_work() && steps < 8 {
            engine.step().unwrap();
            steps += 1;
        }
        assert!(
            !engine.has_work(),
            "engine still reports work {steps} steps after abort: spin livelock"
        );
        let events = collect(&mut rx);
        assert!(
            events.iter().any(|e| matches!(e, EngineEvent::Done { .. })),
            "aborted request never received Done: {events:?}"
        );
    }

    #[test]
    fn single_request_runs_to_completion() {
        let mut engine = BatchEngine::new(cfg(), CountingStepper::new());
        let (tx, mut rx) = mpsc::channel(64);
        engine.admit(GenRequest {
            prompt_tokens: vec![1, 2, 3, 4, 5],
            max_new_tokens: 3,
            eos_token_ids: vec![],
            sampling: SamplingConfig::default(),
            reply: tx,
        });

        let mut guard = 0;
        while engine.has_work() && guard < 100 {
            engine.step().unwrap();
            guard += 1;
        }

        let events = collect(&mut rx);
        let tokens: Vec<u32> = events
            .iter()
            .filter_map(|e| match e {
                EngineEvent::Token { token, .. } => Some(*token),
                _ => None,
            })
            .collect();
        assert_eq!(tokens.len(), 3);
        let done = events.iter().any(|e| {
            matches!(
                e,
                EngineEvent::Done {
                    reason: FinishReason::MaxTokens,
                    completion_tokens: 3,
                    ..
                }
            )
        });
        assert!(done, "expected MaxTokens Done event, got {events:?}");
    }

    #[test]
    fn two_requests_decode_in_the_same_batch() {
        let mut engine = BatchEngine::new(cfg(), CountingStepper::new());
        let (tx1, mut rx1) = mpsc::channel(64);
        let (tx2, mut rx2) = mpsc::channel(64);
        engine.admit(GenRequest {
            prompt_tokens: vec![1, 2, 3],
            max_new_tokens: 4,
            eos_token_ids: vec![],
            sampling: SamplingConfig::default(),
            reply: tx1,
        });
        engine.admit(GenRequest {
            prompt_tokens: vec![7, 8, 9],
            max_new_tokens: 4,
            eos_token_ids: vec![],
            sampling: SamplingConfig::default(),
            reply: tx2,
        });

        let mut guard = 0;
        while engine.has_work() && guard < 100 {
            engine.step().unwrap();
            guard += 1;
        }

        let max_decode_batch = engine.stepper.decoded.iter().copied().max().unwrap_or(0);
        assert_eq!(max_decode_batch, 2, "two seqs should share a decode step");

        let t1 = collect(&mut rx1)
            .iter()
            .filter(|e| matches!(e, EngineEvent::Token { .. }))
            .count();
        let t2 = collect(&mut rx2)
            .iter()
            .filter(|e| matches!(e, EngineEvent::Token { .. }))
            .count();
        assert_eq!(t1, 4);
        assert_eq!(t2, 4);
    }

    #[test]
    fn recycled_prefill_resends_generated_tokens() {
        struct RecordingStepper {
            next: u32,
            prefills: Vec<Vec<u32>>,
        }
        impl BatchStepper for RecordingStepper {
            fn prefill(&mut self, items: &[PrefillInput]) -> Result<Vec<StepResult>> {
                for it in items {
                    self.prefills.push(it.tokens.clone());
                }
                Ok(items
                    .iter()
                    .map(|it| {
                        let token = self.next;
                        self.next += 1;
                        StepResult {
                            seq_id: it.seq_id,
                            token,
                        }
                    })
                    .collect())
            }
            fn decode(&mut self, items: &[SeqInput]) -> Result<Vec<StepResult>> {
                Ok(items
                    .iter()
                    .map(|it| {
                        let token = self.next;
                        self.next += 1;
                        StepResult {
                            seq_id: it.seq_id,
                            token,
                        }
                    })
                    .collect())
            }
        }

        let mut engine = BatchEngine::new(
            cfg(),
            RecordingStepper {
                next: 1000,
                prefills: Vec::new(),
            },
        );
        let (tx, _rx) = mpsc::channel(64);
        let id = engine.admit(GenRequest {
            prompt_tokens: vec![1, 2, 3],
            max_new_tokens: 16,
            eos_token_ids: vec![],
            sampling: SamplingConfig::default(),
            reply: tx,
        });
        engine.step().unwrap();
        engine.step().unwrap();

        let idx = engine
            .scheduler
            .running
            .iter()
            .position(|s| s.id == id)
            .unwrap();
        let mut seq = engine.scheduler.running.remove(idx).unwrap();
        assert_eq!(seq.output, vec![1000, 1001]);
        engine.scheduler.block_manager.deallocate(&seq);
        seq.block_table.clear();
        seq.state = crate::sequence::SequenceState::Waiting;
        engine.scheduler.waiting.push_front(seq);

        engine.step().unwrap();

        assert_eq!(
            engine.stepper.prefills,
            vec![vec![1, 2, 3], vec![1, 2, 3, 1000, 1001]],
            "a recycled sequence must be re-prefilled with prompt + generated tokens"
        );
        let seq = engine.scheduler.running_seq(id).unwrap();
        assert_eq!(seq.output, vec![1000, 1001, 1002]);
    }

    #[test]
    fn preempted_request_resumes_transparently() {
        let mut engine = BatchEngine::new(
            SchedulerConfig {
                max_batch_size: 8,
                max_batched_tokens: 4096,
                block_size: 1,
                num_blocks: 5,
            },
            CountingStepper::new(),
        );
        let (tx1, mut rx1) = mpsc::channel(64);
        let (tx2, mut rx2) = mpsc::channel(64);
        engine.admit(GenRequest {
            prompt_tokens: vec![1, 2],
            max_new_tokens: 2,
            eos_token_ids: vec![],
            sampling: SamplingConfig::default(),
            reply: tx1,
        });
        engine.admit(GenRequest {
            prompt_tokens: vec![3, 4],
            max_new_tokens: 2,
            eos_token_ids: vec![],
            sampling: SamplingConfig::default(),
            reply: tx2,
        });

        let mut guard = 0;
        while engine.has_work() && guard < 100 {
            engine.step().unwrap();
            guard += 1;
        }

        let e1 = collect(&mut rx1);
        assert!(
            e1.iter()
                .any(|e| matches!(e, EngineEvent::Token { token: 1000, .. })),
            "seq 1 must still get its sampled token, got {e1:?}"
        );
        let e2 = collect(&mut rx2);
        assert!(
            !e2.iter().any(|e| matches!(e, EngineEvent::Error { .. })),
            "preemption must be invisible to the client, got {e2:?}"
        );
        let t2: Vec<u32> = e2
            .iter()
            .filter_map(|e| match e {
                EngineEvent::Token { token, .. } => Some(*token),
                _ => None,
            })
            .collect();
        assert_eq!(
            t2,
            vec![1001, 1003],
            "preempted seq keeps its pre-preemption token and resumes"
        );
        assert!(e2.iter().any(|e| matches!(
            e,
            EngineEvent::Done {
                reason: FinishReason::MaxTokens,
                completion_tokens: 2,
                ..
            }
        )));
    }

    fn chunk_cfg() -> SchedulerConfig {
        SchedulerConfig {
            max_batch_size: 8,
            max_batched_tokens: 4,
            block_size: 4,
            num_blocks: 256,
        }
    }

    #[test]
    fn default_step_mixed_defers_chunks_to_full_prefill() {
        struct RecordingStepper {
            next: u32,
            prefills: Vec<Vec<u32>>,
            decode_calls: usize,
        }
        impl BatchStepper for RecordingStepper {
            fn prefill(&mut self, items: &[PrefillInput]) -> Result<Vec<StepResult>> {
                for it in items {
                    self.prefills.push(it.tokens.clone());
                }
                Ok(items
                    .iter()
                    .map(|it| {
                        let token = self.next;
                        self.next += 1;
                        StepResult {
                            seq_id: it.seq_id,
                            token,
                        }
                    })
                    .collect())
            }
            fn decode(&mut self, items: &[SeqInput]) -> Result<Vec<StepResult>> {
                self.decode_calls += 1;
                Ok(items
                    .iter()
                    .map(|it| {
                        let token = self.next;
                        self.next += 1;
                        StepResult {
                            seq_id: it.seq_id,
                            token,
                        }
                    })
                    .collect())
            }
        }

        let mut engine = BatchEngine::new(
            chunk_cfg(),
            RecordingStepper {
                next: 1000,
                prefills: Vec::new(),
                decode_calls: 0,
            },
        );
        let (tx1, _rx1) = mpsc::channel(64);
        engine.admit(GenRequest {
            prompt_tokens: vec![1, 2, 3],
            max_new_tokens: 16,
            eos_token_ids: vec![],
            sampling: SamplingConfig::default(),
            reply: tx1,
        });
        engine.step().unwrap();
        assert_eq!(engine.stepper.prefills, vec![vec![1, 2, 3]]);

        let (tx2, _rx2) = mpsc::channel(64);
        let long_prompt: Vec<u32> = (10..20).collect();
        let long_id = engine.admit(GenRequest {
            prompt_tokens: long_prompt.clone(),
            max_new_tokens: 4,
            eos_token_ids: vec![],
            sampling: SamplingConfig::default(),
            reply: tx2,
        });

        for step in 0..4 {
            engine.step().unwrap();
            assert_eq!(
                engine.stepper.decode_calls,
                step + 1,
                "decode peer runs every step"
            );
        }

        assert_eq!(
            engine.stepper.prefills,
            vec![vec![1, 2, 3], long_prompt],
            "non-final chunks defer compute; final chunk issues one full-prompt prefill"
        );
        let seq = engine.scheduler.running_seq(long_id).unwrap();
        assert_eq!(seq.output.len(), 1);
    }

    #[test]
    fn chunk_native_stepper_receives_ordered_chunks() {
        struct ChunkNativeStepper {
            next: u32,
            seen: Vec<(usize, usize, bool)>,
        }
        impl BatchStepper for ChunkNativeStepper {
            fn prefill(&mut self, _items: &[PrefillInput]) -> Result<Vec<StepResult>> {
                bail!("chunk-native stepper must not receive prefill()");
            }
            fn decode(&mut self, items: &[SeqInput]) -> Result<Vec<StepResult>> {
                Ok(items
                    .iter()
                    .map(|it| {
                        let token = self.next;
                        self.next += 1;
                        StepResult {
                            seq_id: it.seq_id,
                            token,
                        }
                    })
                    .collect())
            }
            fn step_mixed(
                &mut self,
                chunks: &[PrefillChunk],
                decodes: &[SeqInput],
            ) -> Result<Vec<StepResult>> {
                let mut out = Vec::new();
                for c in chunks {
                    self.seen
                        .push((c.chunk_start, c.tokens.len() - c.chunk_start, c.is_final));
                    if c.is_final {
                        let token = self.next;
                        self.next += 1;
                        out.push(StepResult {
                            seq_id: c.seq_id,
                            token,
                        });
                    }
                }
                out.extend(self.decode(decodes)?);
                Ok(out)
            }
        }

        let mut engine = BatchEngine::new(
            chunk_cfg(),
            ChunkNativeStepper {
                next: 1000,
                seen: Vec::new(),
            },
        );
        let (tx, _rx) = mpsc::channel(64);
        engine.admit(GenRequest {
            prompt_tokens: (10..20).collect(),
            max_new_tokens: 1,
            eos_token_ids: vec![],
            sampling: SamplingConfig::default(),
            reply: tx,
        });

        let mut guard = 0;
        while engine.has_work() && guard < 100 {
            engine.step().unwrap();
            guard += 1;
        }

        assert_eq!(
            engine.stepper.seen,
            vec![(0, 4, false), (4, 4, false), (8, 2, true)],
            "chunks must cover the prompt contiguously with is_final only on the last"
        );
    }

    #[test]
    fn mixed_batch_emits_started_once_per_request() {
        let mut engine = BatchEngine::new(chunk_cfg(), CountingStepper::new());
        let (tx, mut rx) = mpsc::channel(64);
        engine.admit(GenRequest {
            prompt_tokens: (10..20).collect(),
            max_new_tokens: 1,
            eos_token_ids: vec![],
            sampling: SamplingConfig::default(),
            reply: tx,
        });

        let mut guard = 0;
        while engine.has_work() && guard < 100 {
            engine.step().unwrap();
            guard += 1;
        }

        let events = collect(&mut rx);
        let started = events
            .iter()
            .filter(|e| matches!(e, EngineEvent::Started { .. }))
            .count();
        assert_eq!(
            started, 1,
            "Started must fire on the first chunk only, got {events:?}"
        );
    }

    struct ChunkStartRecorder {
        next: u32,
        reuse: bool,
        starts: Vec<(u64, usize, usize)>,
    }

    impl BatchStepper for ChunkStartRecorder {
        fn reuses_cached_prefix(&self) -> bool {
            self.reuse
        }

        fn prefill(&mut self, _items: &[PrefillInput]) -> Result<Vec<StepResult>> {
            bail!("chunk-native stepper must not receive prefill()");
        }

        fn decode(&mut self, items: &[SeqInput]) -> Result<Vec<StepResult>> {
            Ok(items
                .iter()
                .map(|it| {
                    let token = self.next;
                    self.next += 1;
                    StepResult {
                        seq_id: it.seq_id,
                        token,
                    }
                })
                .collect())
        }

        fn step_mixed(
            &mut self,
            chunks: &[PrefillChunk],
            decodes: &[SeqInput],
        ) -> Result<Vec<StepResult>> {
            let mut out = Vec::new();
            for c in chunks {
                self.starts.push((c.seq_id, c.chunk_start, c.tokens.len()));
                if c.is_final {
                    let token = self.next;
                    self.next += 1;
                    out.push(StepResult {
                        seq_id: c.seq_id,
                        token,
                    });
                }
            }
            out.extend(self.decode(decodes)?);
            Ok(out)
        }
    }

    fn shared_prefix_chunk_starts(reuse: bool) -> Vec<(u64, usize, usize)> {
        let mut engine = BatchEngine::new(
            cfg(),
            ChunkStartRecorder {
                next: 1000,
                reuse,
                starts: Vec::new(),
            },
        );
        let (tx1, _rx1) = mpsc::channel(64);
        engine.admit(GenRequest {
            prompt_tokens: (1..=8).collect(),
            max_new_tokens: 4,
            eos_token_ids: vec![],
            sampling: SamplingConfig::default(),
            reply: tx1,
        });
        engine.step().unwrap();

        let (tx2, _rx2) = mpsc::channel(64);
        let mut second: Vec<u32> = (1..=8).collect();
        second.extend_from_slice(&[70, 71]);
        engine.admit(GenRequest {
            prompt_tokens: second,
            max_new_tokens: 1,
            eos_token_ids: vec![],
            sampling: SamplingConfig::default(),
            reply: tx2,
        });
        engine.step().unwrap();
        engine.stepper.starts.clone()
    }

    #[test]
    fn a_reuse_capable_stepper_resumes_a_shared_prompt_mid_way() {
        let starts = shared_prefix_chunk_starts(true);
        assert_eq!(
            starts,
            vec![(1, 0, 8), (2, 8, 10)],
            "the second prompt shares two whole blocks with the first, so its only chunk \
             must start at token 8 and forward just the tail"
        );
    }

    #[test]
    fn a_stepper_that_cannot_resume_mid_prompt_still_gets_whole_prompts() {
        let starts = shared_prefix_chunk_starts(false);
        assert_eq!(
            starts,
            vec![(1, 0, 8), (2, 0, 10)],
            "a stepper that tracks its own per-sequence cache length would fail a chunk \
             starting past 0, so reuse stays off until it claims it"
        );
    }

    #[test]
    fn max_waiting_for_defaults_and_overrides() {
        assert_eq!(max_waiting_for(8, None), 32);
        assert_eq!(max_waiting_for(1, None), 16);
        assert_eq!(max_waiting_for(8, Some(100)), 100);
        assert_eq!(max_waiting_for(8, Some(0)), 32);
    }

    #[test]
    fn admit_rejects_when_waiting_queue_is_full() {
        let mut engine = BatchEngine::new(cfg(), CountingStepper::new());
        engine.set_max_waiting(1);
        let (tx1, mut rx1) = mpsc::channel(64);
        engine.admit(GenRequest {
            prompt_tokens: vec![1, 2, 3],
            max_new_tokens: 4,
            eos_token_ids: vec![],
            sampling: SamplingConfig::default(),
            reply: tx1,
        });
        let (tx2, mut rx2) = mpsc::channel(64);
        engine.admit(GenRequest {
            prompt_tokens: vec![4, 5, 6],
            max_new_tokens: 4,
            eos_token_ids: vec![],
            sampling: SamplingConfig::default(),
            reply: tx2,
        });

        let e2 = collect(&mut rx2);
        assert!(
            e2.iter().any(|e| matches!(e, EngineEvent::Error { .. })),
            "second request must be rejected while queue is full, got {e2:?}"
        );
        assert!(e2.iter().any(|e| matches!(
            e,
            EngineEvent::Done {
                reason: FinishReason::Aborted,
                ..
            }
        )));

        let mut guard = 0;
        while engine.has_work() && guard < 100 {
            engine.step().unwrap();
            guard += 1;
        }
        let t1 = collect(&mut rx1)
            .iter()
            .filter(|e| matches!(e, EngineEvent::Token { .. }))
            .count();
        assert_eq!(t1, 4, "first request must still run to completion");
    }

    #[test]
    fn admit_rejects_prompt_exceeding_kv_capacity() {
        let mut engine = BatchEngine::new(cfg(), CountingStepper::new());
        let (tx, mut rx) = mpsc::channel(64);
        engine.admit(GenRequest {
            prompt_tokens: (0..1024u32).collect(),
            max_new_tokens: 4,
            eos_token_ids: vec![],
            sampling: SamplingConfig::default(),
            reply: tx,
        });
        let events = collect(&mut rx);
        assert!(
            events.iter().any(|e| matches!(
                e,
                EngineEvent::Error { message, .. } if message.contains("KV capacity")
            )),
            "oversized prompt must be rejected with a clear error, got {events:?}"
        );
        assert!(events.iter().any(|e| matches!(
            e,
            EngineEvent::Done {
                reason: FinishReason::Aborted,
                ..
            }
        )));
        assert!(!engine.has_work(), "rejected request must not be enqueued");

        let (tx2, mut rx2) = mpsc::channel(64);
        engine.admit(GenRequest {
            prompt_tokens: (0..1023u32).collect(),
            max_new_tokens: 1,
            eos_token_ids: vec![],
            sampling: SamplingConfig::default(),
            reply: tx2,
        });
        let mut guard = 0;
        while engine.has_work() && guard < 2100 {
            engine.step().unwrap();
            guard += 1;
        }
        let t2 = collect(&mut rx2)
            .iter()
            .filter(|e| matches!(e, EngineEvent::Token { .. }))
            .count();
        assert_eq!(t2, 1, "capacity-fitting request must still complete");
    }

    #[test]
    fn eos_stops_one_seq_while_other_continues() {
        struct EosStepper {
            n: u32,
        }
        impl BatchStepper for EosStepper {
            fn prefill(&mut self, items: &[PrefillInput]) -> Result<Vec<StepResult>> {
                Ok(items
                    .iter()
                    .map(|it| StepResult {
                        seq_id: it.seq_id,
                        token: 5,
                    })
                    .collect())
            }
            fn decode(&mut self, items: &[SeqInput]) -> Result<Vec<StepResult>> {
                self.n += 1;
                Ok(items
                    .iter()
                    .map(|it| {
                        let token = if it.seq_id == 1 && self.n == 1 { 99 } else { 5 };
                        StepResult {
                            seq_id: it.seq_id,
                            token,
                        }
                    })
                    .collect())
            }
        }

        let mut engine = BatchEngine::new(cfg(), EosStepper { n: 0 });
        let (tx1, mut rx1) = mpsc::channel(64);
        let (tx2, mut rx2) = mpsc::channel(64);
        engine.admit(GenRequest {
            prompt_tokens: vec![1, 2, 3],
            max_new_tokens: 10,
            eos_token_ids: vec![99],
            sampling: SamplingConfig::default(),
            reply: tx1,
        });
        engine.admit(GenRequest {
            prompt_tokens: vec![4, 5, 6],
            max_new_tokens: 3,
            eos_token_ids: vec![99],
            sampling: SamplingConfig::default(),
            reply: tx2,
        });

        let mut guard = 0;
        while engine.has_work() && guard < 100 {
            engine.step().unwrap();
            guard += 1;
        }

        let e1 = collect(&mut rx1);
        let done1 = e1.iter().find_map(|e| match e {
            EngineEvent::Done { reason, .. } => Some(*reason),
            _ => None,
        });
        assert_eq!(done1, Some(FinishReason::Eos));

        let e2 = collect(&mut rx2);
        let done2 = e2.iter().find_map(|e| match e {
            EngineEvent::Done { reason, .. } => Some(*reason),
            _ => None,
        });
        assert_eq!(done2, Some(FinishReason::MaxTokens));
    }

    #[test]
    fn every_eos_id_stops_a_sequence_not_only_the_first() {
        const EOS_IDS_A_31B_CHECKPOINT_CARRIES: [u32; 3] = [1, 50, 106];

        struct NonFirstEosStepper;
        impl BatchStepper for NonFirstEosStepper {
            fn prefill(&mut self, items: &[PrefillInput]) -> Result<Vec<StepResult>> {
                Ok(items
                    .iter()
                    .map(|it| StepResult {
                        seq_id: it.seq_id,
                        token: 7,
                    })
                    .collect())
            }
            fn decode(&mut self, items: &[SeqInput]) -> Result<Vec<StepResult>> {
                Ok(items
                    .iter()
                    .map(|it| StepResult {
                        seq_id: it.seq_id,
                        token: if it.seq_id == 1 { 50 } else { 106 },
                    })
                    .collect())
            }
        }

        let mut engine = BatchEngine::new(cfg(), NonFirstEosStepper);
        let (tx1, mut rx1) = mpsc::channel(256);
        let (tx2, mut rx2) = mpsc::channel(256);
        for tx in [tx1, tx2] {
            engine.admit(GenRequest {
                prompt_tokens: vec![1, 2, 3],
                max_new_tokens: 64,
                eos_token_ids: EOS_IDS_A_31B_CHECKPOINT_CARRIES.to_vec(),
                sampling: SamplingConfig::default(),
                reply: tx,
            });
        }

        let mut guard = 0;
        while engine.has_work() && guard < 200 {
            engine.step().unwrap();
            guard += 1;
        }

        for (rx, emitted) in [(&mut rx1, 50u32), (&mut rx2, 106u32)] {
            let events = collect(rx);
            let done = events.iter().find_map(|e| match e {
                EngineEvent::Done {
                    reason,
                    completion_tokens,
                    ..
                } => Some((*reason, *completion_tokens)),
                _ => None,
            });
            assert_eq!(
                done,
                Some((FinishReason::Eos, 2)),
                "eos_token_ids {EOS_IDS_A_31B_CHECKPOINT_CARRIES:?}: token {emitted} is a member, \
                 so the sequence must stop on it after 2 tokens; a first-id-only stop runs to \
                 max_new_tokens instead. events={events:?}"
            );
        }
    }
}
