#![allow(dead_code)]

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ChunkPlan {
    pub fresh_cache: bool,
    pub new_start: usize,
    pub emit: bool,
}

pub fn plan_prefill_chunk(
    chunk_start: usize,
    total_tokens: usize,
    cache_len: Option<usize>,
    is_final: bool,
) -> anyhow::Result<ChunkPlan> {
    anyhow::ensure!(
        chunk_start < total_tokens,
        "prefill chunk has no new tokens: start {chunk_start} >= total {total_tokens}"
    );
    if chunk_start == 0 {
        return Ok(ChunkPlan {
            fresh_cache: true,
            new_start: 0,
            emit: is_final,
        });
    }
    match cache_len {
        Some(len) if len == chunk_start => Ok(ChunkPlan {
            fresh_cache: false,
            new_start: chunk_start,
            emit: is_final,
        }),
        Some(len) => anyhow::bail!(
            "prefill chunk desync: cache holds {len} tokens but chunk starts at {chunk_start}"
        ),
        None => anyhow::bail!("prefill chunk starting at {chunk_start} has no prior cache state"),
    }
}

pub fn sampling_params_for(cfg: &nv_engine::SamplingConfig) -> nv_layers::sampler::SamplingParams {
    nv_layers::sampler::SamplingParams {
        temperature: cfg.temperature.max(0.0),
        top_k: cfg.top_k.map(|k| k as usize),
        top_p: cfg.top_p,
        min_p: cfg.min_p,
        presence_penalty: cfg.presence_penalty.unwrap_or(0.0),
        frequency_penalty: cfg.frequency_penalty.unwrap_or(0.0),
        repetition_penalty: cfg.repetition_penalty.unwrap_or(1.0),
    }
}

#[cfg(feature = "cuda")]
mod gemma4 {
    use std::collections::HashMap;
    use std::sync::Arc;

    use anyhow::{anyhow, Result};
    use candle_core::{Device, IndexOp, Tensor};
    use nv_engine::{BatchStepper, PrefillInput, SeqInput, StepResult};
    use nv_models::gemma4::{Gemma4, Gemma4KvCacheFp8};
    use rand_core::{Rng, SeedableRng};
    use rand_pcg::Pcg64;

    struct SeqState {
        cache: Gemma4KvCacheFp8,
        params: nv_layers::sampler::SamplingParams,
        rng: Pcg64,
        counts: HashMap<u32, u32>,

        prompt_tokens: Vec<u32>,
    }

    impl SeqState {
        fn sample(&mut self, logits: &[f32]) -> u32 {
            use nv_layers::sampler;
            if self.params.is_greedy() && !self.params.has_penalties() {
                return sampler::argmax(logits);
            }
            let mut lg = logits.to_vec();
            if self.params.has_penalties() {
                let seen: Vec<(u32, u32)> = self.counts.iter().map(|(&t, &c)| (t, c)).collect();
                sampler::apply_penalties_with_prompt(
                    &mut lg,
                    &seen,
                    &self.prompt_tokens,
                    &self.params,
                );
            }
            let raw = self.rng.next_u64() >> 11;
            let u = ((raw as f64) / ((1u64 << 53) as f64)) as f32;
            let tok = sampler::sample_token(&lg, &self.params, u);
            *self.counts.entry(tok).or_insert(0) += 1;
            tok
        }
    }

    pub struct Gemma4BatchStepper {
        model: Arc<Gemma4>,
        device: Device,
        kv_max_seq_len: usize,
        sampling: HashMap<u64, nv_engine::SamplingConfig>,
        state: HashMap<u64, SeqState>,
    }

    impl Gemma4BatchStepper {
        pub fn new(model: Arc<Gemma4>, device: Device, kv_max_seq_len: usize) -> Self {
            Self {
                model,
                device,
                kv_max_seq_len,
                sampling: HashMap::new(),
                state: HashMap::new(),
            }
        }

        fn last_row(logits: &Tensor) -> Result<Vec<f32>> {
            let dims = logits.dims();
            anyhow::ensure!(
                dims.len() == 3 && dims[0] == 1,
                "expected logits [1, seq, vocab], got {dims:?}"
            );
            let last = logits.i((0usize, dims[1] - 1, ..))?;
            Ok(last.to_dtype(candle_core::DType::F32)?.to_vec1::<f32>()?)
        }
    }

    impl BatchStepper for Gemma4BatchStepper {
        fn on_admit(&mut self, seq_id: u64, sampling: &nv_engine::SamplingConfig) {
            self.sampling.insert(seq_id, sampling.clone());
        }

        fn prefill(&mut self, items: &[PrefillInput]) -> Result<Vec<StepResult>> {
            let mut out = Vec::with_capacity(items.len());
            for it in items {
                let cfg = self.sampling.get(&it.seq_id).cloned().unwrap_or_default();
                let params = super::sampling_params_for(&cfg);
                let seed = cfg.seed.unwrap_or(0);
                let mut cache = self
                    .model
                    .new_kv_cache_fp8(self.kv_max_seq_len)
                    .map_err(|e| anyhow!("alloc fp8 kv cache seq {}: {e}", it.seq_id))?;

                let n = it.tokens.len();
                let tokens = Tensor::from_vec(it.tokens.clone(), (1usize, n), &self.device)?;
                let positions: Vec<i32> = (0..n as i32).collect();
                let positions_t = Tensor::from_vec(positions, n, &self.device)?;
                let logits = self
                    .model
                    .forward_with_cache(&tokens, &positions_t, &mut cache)
                    .map_err(|e| anyhow!("prefill seq {}: {e}", it.seq_id))?;
                let row = Self::last_row(&logits)?;

                let prompt_tokens = if (params.repetition_penalty - 1.0).abs() > f32::EPSILON {
                    let mut ids = it.tokens.clone();
                    ids.sort_unstable();
                    ids.dedup();
                    ids
                } else {
                    Vec::new()
                };
                let mut st = SeqState {
                    cache,
                    params,
                    rng: Pcg64::seed_from_u64(seed),
                    counts: HashMap::new(),
                    prompt_tokens,
                };
                let token = st.sample(&row);
                self.state.insert(it.seq_id, st);
                out.push(StepResult {
                    seq_id: it.seq_id,
                    token,
                });
            }
            Ok(out)
        }

        fn decode(&mut self, items: &[SeqInput]) -> Result<Vec<StepResult>> {
            let mut out = Vec::with_capacity(items.len());
            for it in items {
                let st = self
                    .state
                    .get_mut(&it.seq_id)
                    .ok_or_else(|| anyhow!("decode: no state for seq {}", it.seq_id))?;
                let tokens = Tensor::from_vec(vec![it.token], (1usize, 1usize), &self.device)?;
                let positions_t = Tensor::from_vec(vec![it.position as i32], 1usize, &self.device)?;
                let logits = self
                    .model
                    .forward_with_cache(&tokens, &positions_t, &mut st.cache)
                    .map_err(|e| anyhow!("decode seq {}: {e}", it.seq_id))?;
                let row = Self::last_row(&logits)?;
                let token = st.sample(&row);
                out.push(StepResult {
                    seq_id: it.seq_id,
                    token,
                });
            }
            Ok(out)
        }

        fn release(&mut self, seq_id: u64) {
            self.state.remove(&seq_id);
            self.sampling.remove(&seq_id);
        }
    }
}

#[cfg(feature = "cuda")]
mod paged {
    use std::collections::HashMap;
    use std::sync::{Arc, Mutex};

    use anyhow::{anyhow, bail, Result};
    use candle_core::{Device, IndexOp, Tensor};
    use nv_engine::{BatchStepper, PrefillChunk, PrefillInput, SeqInput, StepResult};
    use nv_models::gemma4::Gemma4;
    use nv_models::paged_fp8::{PagedGemma4Cache, PagedKvFp8Pool, PagedPoolConfig};
    use rand_core::{Rng, SeedableRng};
    use rand_pcg::Pcg64;

    struct SeqState {
        cache: PagedGemma4Cache,
        params: nv_layers::sampler::SamplingParams,
        rng: Pcg64,
        counts: HashMap<u32, u32>,

        prompt_tokens: Vec<u32>,
    }

    impl SeqState {
        fn sample(&mut self, logits: &[f32]) -> u32 {
            use nv_layers::sampler;
            if self.params.is_greedy() && !self.params.has_penalties() {
                return sampler::argmax(logits);
            }
            let mut lg = logits.to_vec();
            if self.params.has_penalties() {
                let seen: Vec<(u32, u32)> = self.counts.iter().map(|(&t, &c)| (t, c)).collect();
                sampler::apply_penalties_with_prompt(
                    &mut lg,
                    &seen,
                    &self.prompt_tokens,
                    &self.params,
                );
            }
            let raw = self.rng.next_u64() >> 11;
            let u = ((raw as f64) / ((1u64 << 53) as f64)) as f32;
            let tok = sampler::sample_token(&lg, &self.params, u);
            *self.counts.entry(tok).or_insert(0) += 1;
            tok
        }
    }

    pub struct Gemma4PagedBatchStepper {
        model: Arc<Gemma4>,
        device: Device,
        pool: Arc<Mutex<PagedKvFp8Pool>>,
        sampling: HashMap<u64, nv_engine::SamplingConfig>,
        state: HashMap<u64, SeqState>,
    }

    impl Gemma4PagedBatchStepper {
        pub fn new(
            model: Arc<Gemma4>,
            device: Device,
            num_blocks: usize,
            block_size: usize,
            lanes: usize,
        ) -> Result<Self> {
            let cfg = if lanes > 0 && nv_models::gemma4::kv_ring_enabled() {
                PagedPoolConfig::from_gemma4_hybrid(model.config(), num_blocks, block_size, lanes)
            } else {
                PagedPoolConfig::from_gemma4(model.config(), num_blocks, block_size)
            };

            let plan = nv_models::paged_fp8::DeriveVPlan::from_model(&model, &cfg)?;
            let pool = PagedKvFp8Pool::new_derive_v(cfg, &device, &plan)?;
            Ok(Self {
                model,
                device,
                pool: Arc::new(Mutex::new(pool)),
                sampling: HashMap::new(),
                state: HashMap::new(),
            })
        }

        pub fn pool(&self) -> Arc<Mutex<PagedKvFp8Pool>> {
            self.pool.clone()
        }

        pub fn cache_len(&self, seq_id: u64) -> Option<usize> {
            self.state.get(&seq_id).map(|st| st.cache.current_len())
        }

        pub fn sample_row(&mut self, seq_id: u64, row: &[f32]) -> Result<u32> {
            let st = self
                .state
                .get_mut(&seq_id)
                .ok_or_else(|| anyhow!("sample_row: no state for seq {seq_id}"))?;
            Ok(st.sample(row))
        }

        pub fn note_graph_decoded(&mut self, seq_id: u64) -> Result<()> {
            use nv_models::gemma4::Gemma4Cache;
            let st = self
                .state
                .get_mut(&seq_id)
                .ok_or_else(|| anyhow!("note_graph_decoded: no state for seq {seq_id}"))?;
            st.cache.advance(1);
            Ok(())
        }

        fn last_row(logits: &Tensor) -> Result<Vec<f32>> {
            let dims = logits.dims();
            anyhow::ensure!(
                dims.len() == 3 && dims[0] == 1,
                "expected logits [1, seq, vocab], got {dims:?}"
            );
            let last = logits.i((0usize, dims[1] - 1, ..))?;
            Ok(last.to_dtype(candle_core::DType::F32)?.to_vec1::<f32>()?)
        }

        fn row(logits: &Tensor, i: usize) -> Result<Vec<f32>> {
            let dims = logits.dims();
            anyhow::ensure!(dims.len() == 2, "expected logits [B, vocab], got {dims:?}");
            let r = logits.i((i, ..))?;
            Ok(r.to_dtype(candle_core::DType::F32)?.to_vec1::<f32>()?)
        }
    }

    impl BatchStepper for Gemma4PagedBatchStepper {
        fn on_admit(&mut self, seq_id: u64, sampling: &nv_engine::SamplingConfig) {
            self.sampling.insert(seq_id, sampling.clone());
        }

        fn prefill(&mut self, items: &[PrefillInput]) -> Result<Vec<StepResult>> {
            let mut out = Vec::with_capacity(items.len());
            for it in items {
                let cfg = self.sampling.get(&it.seq_id).cloned().unwrap_or_default();
                let params = super::sampling_params_for(&cfg);
                let seed = cfg.seed.unwrap_or(0);

                let mut cache = PagedGemma4Cache::new(self.pool.clone(), &self.device)?;
                cache.set_block_table(&it.block_table)?;

                let n = it.tokens.len();
                let chunk = nv_models::gemma4::VERIFY_PREFILL_CHUNK;
                let mut logits_opt = None;
                let mut off = 0usize;
                while off < n {
                    let c = (n - off).min(chunk);
                    let tokens = Tensor::from_vec(
                        it.tokens[off..off + c].to_vec(),
                        (1usize, c),
                        &self.device,
                    )?;
                    let positions: Vec<i32> = (off as i32..(off + c) as i32).collect();
                    let positions_t = Tensor::from_vec(positions, c, &self.device)?;
                    logits_opt = Some(
                        self.model
                            .forward_with_cache(&tokens, &positions_t, &mut cache)
                            .map_err(|e| anyhow!("paged prefill seq {}: {e}", it.seq_id))?,
                    );
                    off += c;
                }
                let logits = logits_opt
                    .ok_or_else(|| anyhow!("paged prefill seq {}: empty prompt", it.seq_id))?;
                let row = Self::last_row(&logits)?;

                let prompt_tokens = if (params.repetition_penalty - 1.0).abs() > f32::EPSILON {
                    let mut ids = it.tokens.clone();
                    ids.sort_unstable();
                    ids.dedup();
                    ids
                } else {
                    Vec::new()
                };
                let mut st = SeqState {
                    cache,
                    params,
                    rng: Pcg64::seed_from_u64(seed),
                    counts: HashMap::new(),
                    prompt_tokens,
                };
                let token = st.sample(&row);
                self.state.insert(it.seq_id, st);
                out.push(StepResult {
                    seq_id: it.seq_id,
                    token,
                });
            }
            Ok(out)
        }

        fn decode(&mut self, items: &[SeqInput]) -> Result<Vec<StepResult>> {
            if items.is_empty() {
                return Ok(Vec::new());
            }

            let mut tokens: Vec<u32> = Vec::with_capacity(items.len());
            let mut positions: Vec<usize> = Vec::with_capacity(items.len());
            for it in items {
                let st = self
                    .state
                    .get_mut(&it.seq_id)
                    .ok_or_else(|| anyhow!("paged decode: no state for seq {}", it.seq_id))?;
                st.cache.set_block_table(&it.block_table)?;
                tokens.push(it.token);
                positions.push(it.position);
            }

            let seq_ids: Vec<u64> = items.iter().map(|it| it.seq_id).collect();
            let mut caches: Vec<&mut PagedGemma4Cache> = Vec::with_capacity(items.len());
            {
                let mut map = std::mem::take(&mut self.state);
                let mut taken: Vec<(u64, SeqState)> = Vec::with_capacity(items.len());
                for sid in &seq_ids {
                    match map.remove(sid) {
                        Some(st) => taken.push((*sid, st)),
                        None => {
                            for (rid, rst) in taken {
                                map.insert(rid, rst);
                            }
                            self.state = map;
                            bail!("paged decode: missing state {sid}");
                        }
                    }
                }
                let logits = {
                    for (_, st) in taken.iter_mut() {
                        caches.push(&mut st.cache);
                    }
                    self.model
                        .forward_decode_batched(&tokens, &positions, &mut caches)
                };
                drop(caches);
                let logits = match logits {
                    Ok(l) => l,
                    Err(e) => {
                        for (sid, st) in taken {
                            map.insert(sid, st);
                        }
                        self.state = map;
                        bail!("paged batched decode: {e}");
                    }
                };
                let mut out = Vec::with_capacity(items.len());
                for (i, (sid, mut st)) in taken.into_iter().enumerate() {
                    let row = Self::row(&logits, i)?;
                    let token = st.sample(&row);
                    map.insert(sid, st);
                    out.push(StepResult { seq_id: sid, token });
                }
                self.state = map;
                Ok(out)
            }
        }

        fn step_mixed(
            &mut self,
            chunks: &[PrefillChunk],
            decodes: &[SeqInput],
        ) -> Result<Vec<StepResult>> {
            let mut out = Vec::with_capacity(chunks.len() + decodes.len());
            for c in chunks {
                let plan = super::plan_prefill_chunk(
                    c.chunk_start,
                    c.tokens.len(),
                    self.state.get(&c.seq_id).map(|st| st.cache.current_len()),
                    c.is_final,
                )
                .map_err(|e| anyhow!("paged step_mixed seq {}: {e}", c.seq_id))?;

                if plan.fresh_cache {
                    let cfg = self.sampling.get(&c.seq_id).cloned().unwrap_or_default();
                    let params = super::sampling_params_for(&cfg);
                    let seed = cfg.seed.unwrap_or(0);
                    let cache = PagedGemma4Cache::new(self.pool.clone(), &self.device)?;
                    self.state.insert(
                        c.seq_id,
                        SeqState {
                            cache,
                            params,
                            rng: Pcg64::seed_from_u64(seed),
                            counts: HashMap::new(),
                            prompt_tokens: Vec::new(),
                        },
                    );
                }
                let st = self
                    .state
                    .get_mut(&c.seq_id)
                    .ok_or_else(|| anyhow!("paged step_mixed: no state for seq {}", c.seq_id))?;
                st.cache.set_block_table(&c.block_table)?;

                let n = c.tokens.len();
                let sub = nv_models::gemma4::VERIFY_PREFILL_CHUNK;
                let mut logits_opt = None;
                let mut off = plan.new_start;
                while off < n {
                    let m = (n - off).min(sub);
                    let tokens = Tensor::from_vec(
                        c.tokens[off..off + m].to_vec(),
                        (1usize, m),
                        &self.device,
                    )?;
                    let positions: Vec<i32> = (off as i32..(off + m) as i32).collect();
                    let positions_t = Tensor::from_vec(positions, m, &self.device)?;
                    logits_opt = Some(
                        self.model
                            .forward_with_cache(&tokens, &positions_t, &mut st.cache)
                            .map_err(|e| {
                                anyhow!("paged chunk prefill seq {} at {off}: {e}", c.seq_id)
                            })?,
                    );
                    off += m;
                }
                if !plan.emit {
                    continue;
                }
                let logits = logits_opt
                    .ok_or_else(|| anyhow!("paged chunk prefill seq {}: empty chunk", c.seq_id))?;
                let row = Self::last_row(&logits)?;
                if (st.params.repetition_penalty - 1.0).abs() > f32::EPSILON {
                    let mut ids = c.tokens.clone();
                    ids.sort_unstable();
                    ids.dedup();
                    st.prompt_tokens = ids;
                }
                let token = st.sample(&row);
                out.push(StepResult {
                    seq_id: c.seq_id,
                    token,
                });
            }
            out.extend(self.decode(decodes)?);
            Ok(out)
        }

        fn release(&mut self, seq_id: u64) {
            self.state.remove(&seq_id);
            self.sampling.remove(&seq_id);
        }
    }
}

#[cfg(feature = "cuda")]
pub use gemma4::Gemma4BatchStepper;

#[cfg(feature = "cuda")]
pub use paged::Gemma4PagedBatchStepper;

#[cfg(test)]
mod tests {
    use super::{plan_prefill_chunk, sampling_params_for, ChunkPlan};

    #[test]
    fn sampling_params_carry_every_config_field_and_clamp_the_absent_ones() {
        let cfg = nv_engine::SamplingConfig {
            temperature: -0.5,
            top_p: Some(0.9),
            top_k: Some(40),
            min_p: Some(0.05),
            seed: Some(7),
            presence_penalty: None,
            frequency_penalty: Some(0.25),
            repetition_penalty: None,
        };
        let p = sampling_params_for(&cfg);
        assert_eq!(p.temperature, 0.0);
        assert_eq!(p.top_k, Some(40usize));
        assert_eq!(p.top_p, Some(0.9));
        assert_eq!(p.min_p, Some(0.05));
        assert_eq!(p.presence_penalty, 0.0);
        assert_eq!(p.frequency_penalty, 0.25);
        assert_eq!(p.repetition_penalty, 1.0);
    }

    #[test]
    fn first_chunk_gets_a_fresh_cache_even_over_stale_state() {
        assert_eq!(
            plan_prefill_chunk(0, 4, None, false).unwrap(),
            ChunkPlan {
                fresh_cache: true,
                new_start: 0,
                emit: false
            }
        );
        assert_eq!(
            plan_prefill_chunk(0, 4, Some(7), true).unwrap(),
            ChunkPlan {
                fresh_cache: true,
                new_start: 0,
                emit: true
            }
        );
    }

    #[test]
    fn continuation_chunk_forwards_only_the_new_tokens() {
        assert_eq!(
            plan_prefill_chunk(4, 8, Some(4), false).unwrap(),
            ChunkPlan {
                fresh_cache: false,
                new_start: 4,
                emit: false
            }
        );
        assert_eq!(
            plan_prefill_chunk(8, 10, Some(8), true).unwrap(),
            ChunkPlan {
                fresh_cache: false,
                new_start: 8,
                emit: true
            }
        );
    }

    #[test]
    fn desync_missing_state_and_empty_chunks_are_errors() {
        assert!(plan_prefill_chunk(4, 8, Some(3), false)
            .unwrap_err()
            .to_string()
            .contains("desync"));
        assert!(plan_prefill_chunk(4, 8, None, false)
            .unwrap_err()
            .to_string()
            .contains("no prior cache state"));
        assert!(plan_prefill_chunk(8, 8, Some(8), true)
            .unwrap_err()
            .to_string()
            .contains("no new tokens"));
    }

    #[cfg(feature = "cuda")]
    mod chunked_scheduler {
        use super::super::plan_prefill_chunk;
        use anyhow::Result;
        use nv_engine::{
            BatchEngine, BatchEvent, BatchStepper, GenRequest, PrefillChunk, PrefillInput,
            SamplingConfig, SchedulerConfig, SeqInput, StepResult,
        };
        use std::collections::HashMap;
        use std::sync::{Arc, Mutex};
        use tokio::sync::mpsc;

        #[derive(Clone, Copy, Debug, PartialEq, Eq)]
        struct ChunkEvent {
            seq_id: u64,
            forwarded: usize,
            emit: bool,
        }

        struct PlanCheckedStepper {
            next: u32,
            cache_len: HashMap<u64, usize>,
            events: Arc<Mutex<Vec<ChunkEvent>>>,
        }

        impl BatchStepper for PlanCheckedStepper {
            fn prefill(&mut self, _items: &[PrefillInput]) -> Result<Vec<StepResult>> {
                anyhow::bail!("chunk-native stepper must not receive prefill()");
            }

            fn decode(&mut self, items: &[SeqInput]) -> Result<Vec<StepResult>> {
                let mut out = Vec::new();
                for it in items {
                    let len = self
                        .cache_len
                        .get_mut(&it.seq_id)
                        .ok_or_else(|| anyhow::anyhow!("decode without cache for {}", it.seq_id))?;
                    anyhow::ensure!(
                        *len == it.position,
                        "decode position {} != cache len {len} for seq {}",
                        it.position,
                        it.seq_id
                    );
                    *len += 1;
                    let token = self.next;
                    self.next += 1;
                    out.push(StepResult {
                        seq_id: it.seq_id,
                        token,
                    });
                }
                Ok(out)
            }

            fn step_mixed(
                &mut self,
                chunks: &[PrefillChunk],
                decodes: &[SeqInput],
            ) -> Result<Vec<StepResult>> {
                let mut out = Vec::new();
                for c in chunks {
                    let plan = plan_prefill_chunk(
                        c.chunk_start,
                        c.tokens.len(),
                        self.cache_len.get(&c.seq_id).copied(),
                        c.is_final,
                    )?;
                    if plan.fresh_cache {
                        self.cache_len.insert(c.seq_id, 0);
                    }
                    let forwarded = c.tokens.len() - plan.new_start;
                    self.events.lock().unwrap().push(ChunkEvent {
                        seq_id: c.seq_id,
                        forwarded,
                        emit: plan.emit,
                    });
                    *self.cache_len.get_mut(&c.seq_id).unwrap() = c.tokens.len();
                    if plan.emit {
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

            fn release(&mut self, seq_id: u64) {
                self.cache_len.remove(&seq_id);
            }
        }

        #[test]
        fn scheduler_chunks_drive_the_plan_within_budget_and_without_desync() {
            const BUDGET: usize = 4;
            let events: Arc<Mutex<Vec<ChunkEvent>>> = Arc::new(Mutex::new(Vec::new()));
            let mut engine = BatchEngine::new(
                SchedulerConfig {
                    max_batch_size: 8,
                    max_batched_tokens: BUDGET,
                    block_size: 4,
                    num_blocks: 256,
                },
                PlanCheckedStepper {
                    next: 1000,
                    cache_len: HashMap::new(),
                    events: events.clone(),
                },
            );

            let (tx1, mut rx1) = mpsc::channel(64);
            let short_id = engine.admit(GenRequest {
                prompt_tokens: vec![1, 2, 3],
                max_new_tokens: 6,
                eos_token_ids: vec![],
                sampling: SamplingConfig::default(),
                reply: tx1,
            });
            engine.step().unwrap();

            let (tx2, mut rx2) = mpsc::channel(64);
            let long_prompt: Vec<u32> = (10..20).collect();
            let long_id = engine.admit(GenRequest {
                prompt_tokens: long_prompt.clone(),
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

            let collect = |rx: &mut mpsc::Receiver<BatchEvent>| {
                let mut out = Vec::new();
                while let Ok(ev) = rx.try_recv() {
                    out.push(ev);
                }
                out
            };
            let e1 = collect(&mut rx1);
            let e2 = collect(&mut rx2);
            for (name, evs) in [("short", &e1), ("long", &e2)] {
                assert!(
                    !evs.iter().any(|e| matches!(e, BatchEvent::Error { .. })),
                    "{name} must not error: {evs:?}"
                );
            }
            let tokens = |evs: &[BatchEvent]| {
                evs.iter()
                    .filter(|e| matches!(e, BatchEvent::Token { .. }))
                    .count()
            };
            assert_eq!(tokens(&e1), 6);
            assert_eq!(tokens(&e2), 2);

            let evs = events.lock().unwrap();
            for ev in evs.iter() {
                assert!(
                    ev.forwarded > 0 && ev.forwarded <= BUDGET,
                    "every chunk forwards within the token budget: {ev:?}"
                );
            }
            let per_seq = |sid: u64| -> (usize, usize) {
                evs.iter()
                    .filter(|e| e.seq_id == sid)
                    .fold((0, 0), |(fwd, emits), e| {
                        (fwd + e.forwarded, emits + usize::from(e.emit))
                    })
            };
            assert_eq!(per_seq(short_id), (3, 1));
            assert_eq!(per_seq(long_id), (long_prompt.len(), 1));
        }
    }
}
