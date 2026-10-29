#[cfg(feature = "cuda")]
use super::*;
#[cfg(any(feature = "cuda", test))]
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
    pool: crate::oapi::vram_plan::Gemma4BatchPool,
) -> anyhow::Result<nv_engine::BatchEngineHandle> {
    use crate::oapi::batch_chat::Gemma4PagedBatchStepper;

    let block_size = crate::oapi::vram_plan::GEMMA4_BATCH_BLOCK_SIZE_IS_THE_SCHEDULER_BLOCK;
    let num_blocks = pool.num_blocks;
    let cfg = nv_engine::SchedulerConfig {
        max_batch_size: pool.max_batch_size,
        max_batched_tokens: kv_max_seq_len.max(4096),
        block_size,
        num_blocks,
    };
    tracing::info!(
        num_blocks,
        pool_blocks = pool.pool_blocks,
        lanes = pool.lanes,
        pool_mib = pool.bytes >> 20,
        "NV_BATCH_ENGINE: paged KV pool sized by the vram plan"
    );
    if !pool.graph && nv_config::env_arm_opt_in_set_to_anything_but_0_reread_every_call("NV_BATCH_GRAPH") {
        tracing::warn!(
            "NV_BATCH_GRAPH=1 ignored: hybrid KV-ring lanes are active (NV_KV_RING); \
             batched decode stays eager"
        );
    }
    let plan = nv_models::gemma4_batch_graph::BucketPlan::from_env();
    let max_ctx = std::env::var("NV_BATCH_GRAPH_MAX_CTX")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(kv_max_seq_len);

    let build = move || -> Box<dyn nv_engine::BatchStepper> {
        let inner = Gemma4PagedBatchStepper::new(
            model.clone(),
            device.clone(),
            pool.pool_blocks,
            block_size,
            pool.lanes,
        )
        .expect("construct Gemma4PagedBatchStepper");
        if !pool.graph {
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
