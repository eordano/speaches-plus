#[cfg(feature = "cuda")]
use super::*;
pub(crate) fn graph_decode_eligible(
    items: &[nv_engine::SeqInput],
    accepts: impl FnOnce(usize, usize) -> bool,
    mut cache_len: impl FnMut(u64) -> Option<usize>,
) -> bool {
    let max_total = items.iter().map(|it| it.position + 1).max().unwrap_or(0);
    if items.is_empty() || !accepts(items.len(), max_total) {
        return false;
    }
    items
        .iter()
        .all(|it| cache_len(it.seq_id) == Some(it.position))
}

#[cfg(feature = "cuda")]
pub(crate) struct GraphedBatchStepper {
    pub(crate) inner: crate::oapi::batch_chat::Gemma4PagedBatchStepper,
    pub(crate) family: nv_models::gemma4_batch_graph::Gemma4BatchGraphFamily,
}

#[cfg(feature = "cuda")]
impl GraphedBatchStepper {
    pub(crate) fn graph_decode(
        &mut self,
        items: &[nv_engine::SeqInput],
    ) -> anyhow::Result<Option<Vec<nv_engine::StepResult>>> {
        use nv_models::gemma4_batch_graph::SlotUpdate;
        let family = &self.family;
        let inner = &self.inner;
        if !graph_decode_eligible(
            items,
            |batch, max_total| family.accepts(batch, max_total),
            |seq_id| inner.cache_len(seq_id),
        ) {
            return Ok(None);
        }
        let updates: Vec<SlotUpdate> = items
            .iter()
            .map(|it| SlotUpdate {
                token: it.token,
                pos: it.position as i32,
                n_total: it.position as i32 + 1,
                block_table: it.block_table.clone(),
                lora_slot: -1,
            })
            .collect();
        let rows = self.family.step(&updates)?;
        let mut out = Vec::with_capacity(items.len());
        for (it, row) in items.iter().zip(rows.iter()) {
            let token = self.inner.sample_row(it.seq_id, row)?;
            self.inner.note_graph_decoded(it.seq_id)?;
            out.push(nv_engine::StepResult {
                seq_id: it.seq_id,
                token,
            });
        }
        Ok(Some(out))
    }
}

#[cfg(feature = "cuda")]
impl nv_engine::BatchStepper for GraphedBatchStepper {
    fn on_admit(&mut self, seq_id: u64, sampling: &nv_engine::SamplingConfig) {
        self.inner.on_admit(seq_id, sampling)
    }

    fn prefill(
        &mut self,
        items: &[nv_engine::PrefillInput],
    ) -> anyhow::Result<Vec<nv_engine::StepResult>> {
        self.inner.prefill(items)
    }

    fn decode(
        &mut self,
        items: &[nv_engine::SeqInput],
    ) -> anyhow::Result<Vec<nv_engine::StepResult>> {
        match self.graph_decode(items)? {
            Some(out) => Ok(out),
            None => self.inner.decode(items),
        }
    }

    fn step_mixed(
        &mut self,
        chunks: &[nv_engine::PrefillChunk],
        decodes: &[nv_engine::SeqInput],
    ) -> anyhow::Result<Vec<nv_engine::StepResult>> {
        if nv_models::gemma4_batch_graph::is_uniform_decode(chunks.len(), decodes.len()) {
            return self.decode(decodes);
        }
        self.inner.step_mixed(chunks, decodes)
    }

    fn release(&mut self, seq_id: u64) {
        self.inner.release(seq_id)
    }
}

#[cfg(feature = "cuda")]
pub(crate) fn build_gemma4_batch_engine(
    model: Arc<nv_models::gemma4::Gemma4>,
    device: candle_core::Device,
    kv_max_seq_len: usize,
) -> anyhow::Result<nv_engine::BatchEngineHandle> {
    use crate::oapi::batch_chat::Gemma4PagedBatchStepper;

    let max_batch_size = std::env::var("NV_BATCH_MAX_SEQS")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(8)
        .max(1);
    let block_size = 16usize;
    let blocks_per_seq = kv_max_seq_len.div_ceil(block_size);
    let want_blocks = (max_batch_size * blocks_per_seq).max(block_size);

    let hybrid = nv_models::gemma4::kv_ring_enabled();
    let num_blocks = if let Some(n) = std::env::var("NV_BATCH_KV_BLOCKS")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
    {
        n.max(block_size)
    } else {
        let (per_block, const_bytes) = if hybrid {
            let g1 = nv_models::paged_fp8::PagedPoolConfig::from_gemma4_hybrid(
                model.config(),
                1,
                block_size,
                max_batch_size,
            );
            let g2 = nv_models::paged_fp8::PagedPoolConfig::from_gemma4_hybrid(
                model.config(),
                2,
                block_size,
                max_batch_size,
            );
            let marginal = g2.pool_bytes().saturating_sub(g1.pool_bytes()).max(1);
            let constant = g1.pool_bytes().saturating_sub(marginal);
            (marginal, constant)
        } else {
            let geom =
                nv_models::paged_fp8::PagedPoolConfig::from_gemma4(model.config(), 1, block_size);
            (geom.bytes_per_block().max(1), 0usize)
        };

        let frac: f64 = std::env::var("NV_BATCH_KV_FRACTION")
            .ok()
            .and_then(|s| s.parse::<f64>().ok())
            .filter(|f| *f > 0.0 && *f < 1.0)
            .unwrap_or(0.25);
        let free_bytes = match &device {
            candle_core::Device::Cuda(d) => {
                let _ = nv_layers::cuda_stream::current_stream(d);
                nv_layers::cudarc::driver::result::mem_get_info()
                    .ok()
                    .map(|(free, _total)| free)
            }
            _ => None,
        };
        match free_bytes {
            Some(free) => {
                let budget = ((free as f64 * frac) as usize).saturating_sub(const_bytes);
                let affordable = (budget / per_block).max(block_size);
                let chosen = affordable.min(want_blocks);
                tracing::info!(
                    free_gb = free as f64 / 1e9,
                    hybrid,
                    per_block_kb = per_block as f64 / 1e3,
                    const_mb = const_bytes as f64 / 1e6,
                    want_blocks,
                    affordable,
                    chosen,
                    "NV_BATCH_ENGINE: sizing paged KV pool from free VRAM"
                );
                chosen
            }
            None => {
                tracing::warn!(
                    "NV_BATCH_ENGINE: mem_get_info unavailable; capping KV pool at 2048 blocks"
                );
                want_blocks.min(2048)
            }
        }
    };

    let cfg = nv_engine::SchedulerConfig {
        max_batch_size,
        max_batched_tokens: kv_max_seq_len.max(4096),
        block_size,
        num_blocks,
    };

    let graph_on = std::env::var("NV_BATCH_GRAPH").is_ok_and(|v| v != "0");
    let plan = nv_models::gemma4_batch_graph::BucketPlan::from_env();
    let graph_usable = graph_on && !hybrid;
    if graph_on && hybrid {
        tracing::warn!(
            "NV_BATCH_GRAPH=1 ignored: hybrid KV-ring lanes are active (NV_KV_RING); \
             batched decode stays eager"
        );
    }
    let scratch_blocks = if graph_usable { plan.max_bucket() } else { 0 };
    let pool_blocks = num_blocks + scratch_blocks;
    let max_ctx = std::env::var("NV_BATCH_GRAPH_MAX_CTX")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(kv_max_seq_len);

    let lanes = if hybrid { max_batch_size } else { 0 };
    let build = move || -> Box<dyn nv_engine::BatchStepper> {
        let inner = Gemma4PagedBatchStepper::new(
            model.clone(),
            device.clone(),
            pool_blocks,
            block_size,
            lanes,
        )
        .expect("construct Gemma4PagedBatchStepper");
        if !graph_usable {
            return Box::new(inner);
        }
        match nv_models::gemma4_batch_graph::Gemma4BatchGraphFamily::new(
            model,
            inner.pool(),
            &device,
            plan,
            num_blocks as u32,
            max_ctx,
        ) {
            Ok(family) => {
                tracing::info!(
                    sizes = ?family.plan().sizes(),
                    max_ctx = family.max_ctx(),
                    "NV_BATCH_GRAPH=1: CUDA-graph family active for batched Gemma4 decode"
                );
                Box::new(GraphedBatchStepper { inner, family })
            }
            Err(e) => {
                tracing::warn!(error = %e, "NV_BATCH_GRAPH=1: graph family init failed; batched decode stays eager");
                Box::new(inner)
            }
        }
    };
    Ok(nv_engine::BatchEngineHandle::spawn(
        cfg,
        std::time::Duration::from_millis(1),
        build,
    ))
}
