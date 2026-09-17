use anyhow::{Context, Result};
use candle_core::{DType, Device, Storage, Tensor};
use cudarc::driver::{CudaSlice, CudaStream, DevicePtr};
use nv_kernels::graph::CudaGraphRunner;
use std::sync::Arc;

use crate::laguna::{Laguna, LagunaFfn};
use crate::laguna_fp8::LagunaKvCacheFp8;

pub struct LagunaBatchStepGraph {
    runner: CudaGraphRunner,
    verify: Option<BatchVerifyBuffers>,
    caches: [LagunaKvCacheFp8; 2],
    model: Arc<Laguna>,
    stream: Arc<CudaStream>,
    tokens: Tensor,
    positions: Tensor,
    host_tokens: Box<[u32; 2]>,
    host_positions: Box<[i32; 2]>,
    logits: CudaSlice<f32>,
    replays: [u64; 4],
    captured_m2_fused_layers: usize,
    healthy: bool,
}

impl LagunaBatchStepGraph {
    pub fn new(model: Arc<Laguna>, capacity: usize) -> Result<Self> {
        let dev = match model.device() {
            Device::Cuda(d) => d.clone(),
            _ => anyhow::bail!("prepared B2 graphs require CUDA"),
        };
        anyhow::ensure!(
            !nv_layers::cuda_stream::is_overridden(),
            "prepared B2 owner requires the model CUDA stream"
        );
        anyhow::ensure!(
            !model.host_moe()
                && nv_layers::moe_grouped::route_fuse_enabled()
                && nv_layers::moe_grouped::chain_slim_launches_enabled(),
            "prepared B2 graphs require device routed fused MoE chain"
        );
        for key in [
            "NV_LAGUNA_ROUTE_AB_PROBE",
            "NV_MOE_DECODE_PROF",
            "NV_MOE_MT_STALL",
        ] {
            anyhow::ensure!(
                std::env::var_os(key).is_none(),
                "prepared B2 graphs refuse capture-unsafe diagnostic {key}"
            );
        }
        let mut packed = 0;
        for layer in model.layers() {
            if let LagunaFfn::Moe(moe) = &layer.ffn {
                anyhow::ensure!(model.grouped_weights(moe)?.is_some(),
                    "prepared B2 graphs require resident NVFP4 grouped weights for every routed layer");
                packed += 1;
            }
        }
        anyhow::ensure!(
            packed > 0,
            "prepared B2 graph target has no packed routed layers"
        );
        let make = || -> Result<LagunaKvCacheFp8> {
            let mut cache = LagunaKvCacheFp8::new_fixed_one_with_prefill_chunk(
                model.config(),
                capacity,
                model.device(),
                model.prefill_chunk_tokens(),
            )?;
            cache.enable_fixed_one_batch_attention(model.config())?;
            Ok(cache)
        };
        let caches = [make()?, make()?];
        let stream = dev.cuda_stream();
        anyhow::ensure!(
            !stream.cu_stream().is_null(),
            "prepared B2 graphs capture on the model stream, so the model must live on Device::new_cuda_with_stream: the legacy null stream cannot begin capture and a forked capture would not include the body's candle-core ops"
        );
        let logits = stream.alloc_zeros::<f32>(2 * model.config().vocab_size)?;
        let tokens = Tensor::zeros(2, DType::U32, model.device())?;
        let positions = Tensor::zeros(2, DType::I32, model.device())?;
        eprintln!("[laguna_b2_graph] created physical_lanes=2 max_gpu_batch=2 full_attention=two_independent_m1_xqa sliding_attention=dynamic_gqa kv=fixed-one-e4m3 packed_routed_layers={packed} max_seq={capacity}");
        Ok(Self {
            runner: CudaGraphRunner::new(stream.clone()),
            verify: None,
            caches,
            model,
            stream,
            tokens,
            positions,
            host_tokens: Box::new([0; 2]),
            host_positions: Box::new([0; 2]),
            logits,
            replays: [0; 4],
            captured_m2_fused_layers: 0,
            healthy: true,
        })
    }

    pub(crate) fn cache_mut(&mut self, lane: usize) -> Result<&mut LagunaKvCacheFp8> {
        anyhow::ensure!(
            self.healthy,
            "engine busy: prepared B2 graph state is retired; try later."
        );
        self.caches
            .get_mut(lane)
            .context("invalid physical B2 lane")
    }

    pub fn current_lengths(&self) -> [usize; 2] {
        [self.caches[0].current_len(), self.caches[1].current_len()]
    }

    pub fn captured_masks(&self) -> [bool; 3] {
        [1, 2, 3].map(|m| self.runner.has_cached_token(m))
    }

    pub fn replay_counts(&self) -> [u64; 3] {
        [self.replays[1], self.replays[2], self.replays[3]]
    }

    pub fn report(&self, label: &str) {
        eprintln!("[laguna_b2_graph] {label} execution=shared_projection_m2 physical_masks=01,10,11 captured={:?} graph_nodes={} successful_replays={:?} lengths={:?} direct_fp8_prefill_calls={:?} captured_m2_fused_layers={} attn_fp8_resident={} attn_shape_policy={} lm_head_fp8_resident={} lm_head_i8_resident={}",
            self.captured_masks(), self.runner.cached_node_count(), self.replay_counts(),
            self.current_lengths(), self.caches.each_ref().map(|c| c.direct_fp8_prefill_enqueued_calls()),
            self.captured_m2_fused_layers, self.model.attn_fp8_active(), self.model.attn_w8_shape_policy(),
            self.model.lm_head_fp8_active(), self.model.lm_head_i8_active());
    }

    pub fn step(
        &mut self,
        active: &[usize],
        tokens: &[u32],
        positions: &[usize],
    ) -> Result<Vec<f32>> {
        self.run(active, tokens, positions, true)
    }

    fn run(
        &mut self,
        active: &[usize],
        tokens: &[u32],
        positions: &[usize],
        captured: bool,
    ) -> Result<Vec<f32>> {
        anyhow::ensure!(
            self.healthy,
            "engine busy: prepared B2 graph state is retired; try later."
        );
        let mask = match active {
            [0] => 1,
            [1] => 2,
            [0, 1] => 3,
            _ => anyhow::bail!("prepared B2 graph requires ordered physical masks 01, 10 or 11"),
        };
        anyhow::ensure!(
            tokens.len() == active.len() && positions.len() == active.len(),
            "prepared B2 input lengths differ"
        );
        for (row, &lane) in active.iter().enumerate() {
            self.caches[lane].require_initialized_batch_prefix()?;
            anyhow::ensure!(
                (tokens[row] as usize) < self.model.config().vocab_size
                    && positions[row] == self.caches[lane].current_len()
                    && positions[row] < self.caches[lane].max_seq_len(),
                "prepared B2 lane {lane} invalid token, position or capacity"
            );
        }
        let result = self.run_validated(active, tokens, positions, mask, captured);
        if result.is_err() {
            self.healthy = false;
        }
        result.context("engine busy: prepared B2 decode failed; worker must retire; try later.")
    }

    fn run_validated(
        &mut self,
        active: &[usize],
        tokens: &[u32],
        positions: &[usize],
        mask: u64,
        captured: bool,
    ) -> Result<Vec<f32>> {
        let b = active.len();
        self.stream.context().bind_to_thread()?;
        let dev = match self.model.device() {
            Device::Cuda(d) => d,
            _ => unreachable!(),
        };
        anyhow::ensure!(
            nv_layers::cuda_stream::current_stream(dev).cu_stream() == self.stream.cu_stream(),
            "prepared B2 owner cannot move between CUDA streams"
        );
        self.model.apply_attn_w8_for(b);
        for (row, &lane) in active.iter().enumerate() {
            self.caches[lane].prepare_for_decode_dev(positions[row], positions[row] + 1)?;
            self.host_tokens[row] = tokens[row];
            self.host_positions[row] = i32::try_from(positions[row])?;
        }
        let tokens_t = self.tokens.narrow(0, 0, b)?;
        let positions_t = self.positions.narrow(0, 0, b)?;
        copy_host_into_tensor(&self.stream, &tokens_t, &self.host_tokens[..b])?;
        copy_host_into_tensor(&self.stream, &positions_t, &self.host_positions[..b])?;
        let dev = match self.model.device() {
            Device::Cuda(d) => d,
            _ => unreachable!(),
        };
        crate::graph_engine::disable_event_tracking_no_earlier_than_the_first_capture(dev);
        let _cuda_graph_htod_cache_guard = dev.enable_cuda_graph_htod_cache();
        let replay = self.runner.has_cached_token(mask);
        let Self {
            runner,
            caches,
            model,
            stream,
            logits,
            captured_m2_fused_layers,
            ..
        } = self;
        let mut refs: Vec<&mut LagunaKvCacheFp8> = caches
            .iter_mut()
            .enumerate()
            .filter(|(lane, _)| active.contains(lane))
            .map(|(_, c)| c)
            .collect();
        if captured && !replay {
            prepared_body(model, &tokens_t, &positions_t, &mut refs, logits, stream)?;
            stream.synchronize()?;
        }
        if captured && !replay && b == 2 {
            model.take_batch_moe_fused_dispatches();
        }
        if captured {
            runner.run_on(mask, Some(stream), |s| {
                nv_layers::cuda_stream::with_stream(s.clone(), || {
                    prepared_body(model, &tokens_t, &positions_t, &mut refs, logits, s)
                })
            })?;
        } else {
            prepared_body(model, &tokens_t, &positions_t, &mut refs, logits, stream)?;
        }
        stream.synchronize()?;
        if captured && !replay && b == 2 {
            let count = model.take_batch_moe_fused_dispatches();
            let expected = model
                .layers()
                .iter()
                .filter(|l| matches!(l.ffn, LagunaFfn::Moe(_)))
                .count();
            anyhow::ensure!(
                count == expected && count > 0,
                "M2 graph capture did not execute all fused routed layers: {count}/{expected}"
            );
            *captured_m2_fused_layers = count;
        }
        for cache in &refs {
            cache.batch_attention_status_after_sync()?;
        }
        #[allow(deprecated)]
        let host = stream.memcpy_dtov(&logits.slice(..b * model.config().vocab_size))?;
        for cache in &mut refs {
            cache.note_graph_write();
            cache.advance(1);
        }
        if captured && replay {
            self.replays[mask as usize] += 1;
        }
        Ok(host)
    }

    pub(crate) fn warm_all_masks(&mut self) -> Result<()> {
        for active in [&[0usize][..], &[1usize][..], &[0usize, 1usize][..]] {
            for _ in 0..2 {
                let positions: Vec<usize> = active
                    .iter()
                    .map(|&i| self.caches[i].current_len())
                    .collect();
                let output = self.step(active, &vec![100; active.len()], &positions)?;
                for row in output.chunks_exact(self.model.config().vocab_size) {
                    crate::laguna_dflash::checked_target_argmax_row(row)
                        .context("invalid B2 warmup logits")?;
                }
            }
        }
        anyhow::ensure!(
            self.captured_masks() == [true; 3] && self.replay_counts().iter().all(|&n| n > 0),
            "B2 warmup must capture and replay every physical mask"
        );
        self.report("ready");
        for cache in &mut self.caches {
            cache.reset();
        }
        Ok(())
    }
}

struct BatchVerifyBuffers {
    runner: CudaGraphRunner,
    model_stream: Arc<CudaStream>,
    tokens: Tensor,
    positions: Tensor,
    host_tokens: Box<[u32; 6]>,
    host_positions: Box<[i32; 6]>,
    logits: Tensor,
    aux: Vec<Tensor>,
    aux_layers: Vec<usize>,
    replays: [u64; 8],
}

impl LagunaBatchStepGraph {
    pub(crate) fn new_spec(
        model: Arc<Laguna>,
        capacity: usize,
        aux_layers: &[usize],
    ) -> Result<Self> {
        anyhow::ensure!(
            model.dtype() == DType::BF16
                && aux_layers.len() == 5
                && aux_layers.iter().all(|&i| i < model.layers().len())
                && aux_layers
                    .iter()
                    .enumerate()
                    .all(|(i, x)| !aux_layers[..i].contains(x)),
            "m2-spec requires five distinct BF16 target taps"
        );
        let Device::Cuda(dev) = model.device() else {
            anyhow::bail!("m2-spec requires CUDA")
        };
        anyhow::ensure!(!dev.cuda_stream().cu_stream().is_null(),
            "m2-spec Candle body requires Device::new_cuda_with_stream; a TLS fork cannot capture Candle operations");
        crate::gemma4_batch_graph::graph_teardown::disable_event_tracking_before_capture(
            dev.cuda_stream().context(),
        );
        let mut owner = Self::new(model, capacity)?;
        for cache in &mut owner.caches {
            cache.enable_fixed_one_batch_verify_attention()?;
        }
        let model = &owner.model;
        6usize
            .checked_mul(model.config().vocab_size)
            .context("B2 verify logits size overflow")?;
        6usize
            .checked_mul(model.config().hidden_size)
            .context("B2 verify auxiliary size overflow")?;
        let model_stream = owner.stream.clone();
        nv_quant::nvfp4::ensure_workspace_for_stream(&model_stream)?;
        owner.verify = Some(BatchVerifyBuffers {
            runner: CudaGraphRunner::new(model_stream.clone()),
            model_stream,
            tokens: Tensor::zeros(6, DType::U32, model.device())?,
            positions: Tensor::zeros(6, DType::I32, model.device())?,
            host_tokens: Box::new([0; 6]),
            host_positions: Box::new([0; 6]),
            logits: Tensor::zeros(
                (1, 6, model.config().vocab_size),
                DType::F32,
                model.device(),
            )?,
            aux: (0..aux_layers.len())
                .map(|_| {
                    Tensor::zeros(
                        (1, 6, model.config().hidden_size),
                        DType::BF16,
                        model.device(),
                    )
                })
                .collect::<candle_core::Result<Vec<_>>>()?,
            aux_layers: aux_layers.to_vec(),
            replays: [0; 8],
        });
        owner.stream.synchronize()?;
        Ok(owner)
    }

    pub(crate) fn verify(
        &mut self,
        active: &[usize],
        tokens: &[u32],
        positions: &[usize],
        rows: usize,
    ) -> Result<(Tensor, Vec<Tensor>)> {
        self.verify_run(active, tokens, positions, rows, true)
    }

    fn verify_run(
        &mut self,
        active: &[usize],
        tokens: &[u32],
        positions: &[usize],
        rows: usize,
        captured: bool,
    ) -> Result<(Tensor, Vec<Tensor>)> {
        anyhow::ensure!(
            self.healthy,
            "engine busy: B2 spec state is retired; try later."
        );
        let mask = match active {
            [0] => 1usize,
            [1] => 2,
            [0, 1] => 3,
            _ => anyhow::bail!("B2 spec requires physical masks 01, 10 or 11"),
        };
        let m = active
            .len()
            .checked_mul(rows)
            .context("B2 verify row overflow")?;
        anyhow::ensure!(
            matches!(rows, 1 | 3) && tokens.len() == m && positions.len() == active.len(),
            "B2 spec input lengths or block size differ"
        );
        for (slot, &lane) in active.iter().enumerate() {
            let cache = &self.caches[lane];
            cache.require_initialized_batch_prefix()?;
            anyhow::ensure!(
                positions[slot] == cache.current_len()
                    && positions[slot]
                        .checked_add(rows)
                        .is_some_and(|end| end <= cache.max_seq_len())
                    && tokens[slot * rows..(slot + 1) * rows]
                        .iter()
                        .all(|&t| (t as usize) < self.model.config().vocab_size),
                "B2 spec lane {lane} invalid token, position or capacity"
            );
        }
        self.stream.synchronize()?;
        anyhow::ensure!(
            !nv_layers::cuda_stream::is_overridden(),
            "m2-spec requires the model stream without a TLS override"
        );
        let result = self.verify_validated(active, tokens, positions, rows, mask, captured);
        if result.is_err() {
            self.healthy = false;
        }
        result.context("engine busy: B2 spec verification failed; worker must retire; try later.")
    }

    fn verify_validated(
        &mut self,
        active: &[usize],
        tokens: &[u32],
        positions: &[usize],
        rows: usize,
        mask: usize,
        captured: bool,
    ) -> Result<(Tensor, Vec<Tensor>)> {
        let m = tokens.len();
        let key = mask + if rows == 3 { 4 } else { 0 };
        let graph_key = 0x2000 + key as u64;
        let stream = self
            .verify
            .as_ref()
            .context("B2 spec buffers are absent")?
            .model_stream
            .clone();
        stream.context().bind_to_thread()?;
        let Device::Cuda(dev) = self.model.device() else {
            anyhow::bail!("B2 spec requires CUDA")
        };
        anyhow::ensure!(
            dev.cuda_stream().cu_stream() == stream.cu_stream()
                && nv_layers::cuda_stream::current_stream(dev).cu_stream() == stream.cu_stream(),
            "B2 spec cannot move between CUDA streams"
        );
        self.model.apply_attn_w8_for(m);
        let buffers = self.verify.as_mut().context("B2 spec buffers are absent")?;
        for (slot, &lane) in active.iter().enumerate() {
            let end = positions[slot]
                .checked_add(rows)
                .context("B2 verify end overflow")?;
            self.caches[lane].prepare_for_decode_dev(positions[slot], end)?;
            for row in 0..rows {
                let i = slot * rows + row;
                buffers.host_tokens[i] = tokens[i];
                buffers.host_positions[i] = i32::try_from(
                    positions[slot]
                        .checked_add(row)
                        .context("B2 position overflow")?,
                )?;
            }
        }
        let tokens_t = buffers.tokens.narrow(0, 0, m)?;
        let positions_t = buffers.positions.narrow(0, 0, m)?;
        copy_host_into_tensor(&stream, &tokens_t, &buffers.host_tokens[..m])?;
        copy_host_into_tensor(&stream, &positions_t, &buffers.host_positions[..m])?;
        crate::graph_engine::disable_event_tracking_no_earlier_than_the_first_capture(dev);
        let _cuda_graph_htod_cache_guard = dev.enable_cuda_graph_htod_cache();
        let replay = self
            .verify
            .as_ref()
            .unwrap()
            .runner
            .has_cached_token(graph_key);
        let Self {
            caches,
            model,
            verify,
            ..
        } = self;
        let buffers = verify.as_mut().context("B2 spec buffers disappeared")?;
        let mut refs: Vec<&mut LagunaKvCacheFp8> = caches
            .iter_mut()
            .enumerate()
            .filter(|(lane, _)| active.contains(lane))
            .map(|(_, c)| c)
            .collect();
        if captured && !replay {
            prepared_verify_body(
                model,
                &tokens_t,
                &positions_t,
                &mut refs,
                rows,
                buffers,
                &stream,
            )?;
            stream.synchronize()?;
        }
        if captured {
            let BatchVerifyBuffers {
                runner,
                tokens: _,
                positions: _,
                host_tokens: _,
                host_positions: _,
                logits,
                aux,
                aux_layers,
                replays: _,
                model_stream: _,
            } = buffers;
            runner.run_on(graph_key, Some(&stream), |s| {
                prepared_verify_outputs(
                    model,
                    &tokens_t,
                    &positions_t,
                    &mut refs,
                    rows,
                    aux_layers,
                    logits,
                    aux,
                    s,
                )
            })?;
        } else {
            prepared_verify_body(
                model,
                &tokens_t,
                &positions_t,
                &mut refs,
                rows,
                buffers,
                &stream,
            )?;
        }
        stream.synchronize()?;
        for cache in &refs {
            cache.batch_verify_attention_status_after_sync(rows)?;
        }
        let logits = buffers.logits.narrow(1, 0, m)?;
        let aux = buffers
            .aux
            .iter()
            .map(|a| a.narrow(1, 0, m))
            .collect::<candle_core::Result<Vec<_>>>()?;
        for cache in &mut refs {
            cache.note_graph_write();
            cache.advance(rows);
        }
        if captured && replay {
            buffers.replays[key] += 1;
        }
        if captured && !replay {
            eprintln!("[laguna_b2_spec_graph] captured physical_mask={mask:02b} rows_per_lane={rows} target_rows={m} attention=per-lane-xqa-and-sliding-window head_tier=spec grouped_moe_rows={m} aux_taps={} graph_nodes={}", buffers.aux_layers.len(), buffers.runner.cached_node_count());
        }
        Ok((logits, aux))
    }

    pub(crate) fn report_spec(&self, label: &str) {
        if let Some(v) = &self.verify {
            eprintln!("[laguna_b2_spec_graph] {label} physical_lanes=2 max_target_rows=6 proposer_batch=1 captured={:?} successful_replays={:?} lengths={:?} direct_fp8_prefill_calls={:?}",
                [1u64,2,3,5,6,7].map(|k| v.runner.has_cached_token(0x2000+k)),
                [1usize,2,3,5,6,7].map(|k| v.replays[k]), self.current_lengths(),
                self.caches.each_ref().map(|c| c.direct_fp8_prefill_enqueued_calls()));
        }
    }

    #[cfg(test)]
    pub(crate) fn spec_graph_observation(&self) -> ([bool; 6], [u64; 6]) {
        let v = self
            .verify
            .as_ref()
            .expect("spec graph observation requires verification buffers");
        (
            [1u64, 2, 3, 5, 6, 7].map(|k| v.runner.has_cached_token(0x2000 + k)),
            [1usize, 2, 3, 5, 6, 7].map(|k| v.replays[k]),
        )
    }
}

fn copy_tensor_into<T: cudarc::driver::DeviceRepr + candle_core::cuda_backend::CudaDType>(
    src: &Tensor,
    dst: &Tensor,
    s: &Arc<CudaStream>,
) -> Result<()> {
    anyhow::ensure!(
        src.dtype() == dst.dtype()
            && src.is_contiguous()
            && dst.is_contiguous()
            && src.elem_count() <= dst.elem_count(),
        "prepared output buffer shape mismatch"
    );
    let (ss, sl) = src.storage_and_layout();
    let (ds, dl) = dst.storage_and_layout();
    let (Storage::Cuda(sc), Storage::Cuda(dc)) = (&*ss, &*ds) else {
        anyhow::bail!("prepared outputs require CUDA");
    };
    let sv = sc
        .as_cuda_slice::<T>()?
        .slice(sl.start_offset()..sl.start_offset() + src.elem_count());
    let dv = dc
        .as_cuda_slice::<T>()?
        .slice(dl.start_offset()..dl.start_offset() + src.elem_count());
    let (sp, _sg) = sv.device_ptr(s);
    let (dp, _dg) = dv.device_ptr(s);
    let bytes = src
        .elem_count()
        .checked_mul(std::mem::size_of::<T>())
        .context("prepared output byte overflow")?;
    unsafe {
        cudarc::driver::result::memcpy_dtod_async(dp, sp, bytes, s.cu_stream())?;
    }
    Ok(())
}

fn prepared_verify_body(
    model: &Laguna,
    tokens: &Tensor,
    positions: &Tensor,
    caches: &mut [&mut LagunaKvCacheFp8],
    rows: usize,
    out: &BatchVerifyBuffers,
    stream: &Arc<CudaStream>,
) -> Result<()> {
    prepared_verify_outputs(
        model,
        tokens,
        positions,
        caches,
        rows,
        &out.aux_layers,
        &out.logits,
        &out.aux,
        stream,
    )
}

fn prepared_verify_outputs(
    model: &Laguna,
    tokens: &Tensor,
    positions: &Tensor,
    caches: &mut [&mut LagunaKvCacheFp8],
    rows: usize,
    aux_layers: &[usize],
    output: &Tensor,
    outputs_aux: &[Tensor],
    stream: &Arc<CudaStream>,
) -> Result<()> {
    let (logits, aux) =
        model.forward_verify_batched_prepared(tokens, positions, caches, rows, aux_layers)?;
    copy_tensor_into::<f32>(&logits, output, stream)?;
    anyhow::ensure!(
        aux.len() == outputs_aux.len(),
        "B2 auxiliary output count changed"
    );
    for (source, dest) in aux.iter().zip(outputs_aux) {
        copy_tensor_into::<half::bf16>(source, dest, stream)?;
    }
    Ok(())
}

impl Drop for BatchVerifyBuffers {
    fn drop(&mut self) {
        let teardown = crate::gemma4_batch_graph::graph_teardown::GraphTeardown::for_a_stream_this_engine_did_not_fork(&self.model_stream);
        let runner = &mut self.runner;
        teardown.run_owned_streams_only(|| runner.invalidate());
    }
}

fn copy_host_into_tensor<T: cudarc::driver::DeviceRepr + candle_core::cuda_backend::CudaDType>(
    s: &Arc<CudaStream>,
    t: &Tensor,
    values: &[T],
) -> Result<()> {
    let (storage, layout) = t.storage_and_layout();
    let cuda = match &*storage {
        Storage::Cuda(v) => v,
        _ => anyhow::bail!("prepared input is not CUDA"),
    };
    let slice = cuda.as_cuda_slice::<T>()?;
    let (ptr, _guard) = slice.device_ptr(s);
    let ptr = ptr + (layout.start_offset() * std::mem::size_of::<T>()) as u64;
    unsafe {
        cudarc::driver::result::memcpy_htod_async(ptr, values, s.cu_stream())?;
    }
    Ok(())
}

fn prepared_body(
    model: &Laguna,
    tokens: &Tensor,
    positions: &Tensor,
    caches: &mut [&mut LagunaKvCacheFp8],
    out: &mut CudaSlice<f32>,
    stream: &Arc<CudaStream>,
) -> Result<()> {
    let logits = model.forward_decode_batched_prepared(tokens, positions, caches)?;
    let (storage, layout) = logits.storage_and_layout();
    let cuda = match &*storage {
        Storage::Cuda(v) => v,
        _ => anyhow::bail!("B2 logits are not CUDA"),
    };
    let view = cuda
        .as_cuda_slice::<f32>()?
        .slice(layout.start_offset()..layout.start_offset() + logits.elem_count());
    let mut dest = out.slice_mut(..logits.elem_count());
    stream.memcpy_dtod(&view, &mut dest)?;
    Ok(())
}

impl Drop for LagunaBatchStepGraph {
    fn drop(&mut self) {
        if self.verify.is_some() {
            self.report_spec("worker-exit");
        }
        let _ = self.stream.context().bind_to_thread();
        crate::gemma4_batch_graph::graph_teardown::GraphTeardown::for_a_stream_this_engine_did_not_fork(
            &self.stream,
        )
        .run_owned_streams_only(|| self.runner.invalidate());
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gemma4::{Gemma4Cache, LayerType};
    use crate::laguna::LagunaConfig;
    use nv_weights::{QuantizationConfig, WeightLoader};

    const TARGET_REVISION: &str = "d32afde8b09af1539b49ff96ff5551c674485f8e";
    const TARGET_CONFIG_SHA256: &str =
        "8bba8588c6197aa04be89a405a0af640e1914c6323934f7e0860f7cedc528f4e";
    const CAPACITY: usize = 262144;
    const MASKS: [&[usize]; 7] = [&[0, 1], &[0], &[1], &[0, 1], &[1], &[0], &[0, 1]];

    fn prefill_one(g: &mut LagunaBatchStepGraph, lane: usize, ids: &[u32]) -> u32 {
        let model = g.model.clone();
        let cache = g.cache_mut(lane).unwrap();
        cache.reset();
        let mut anchor = None;
        let mut offset = 0;
        while offset < ids.len() {
            let m = model.next_prefill_chunk_len(ids.len() - offset);
            let t =
                Tensor::from_vec(ids[offset..offset + m].to_vec(), (1, m), model.device()).unwrap();
            let p = Tensor::from_vec(
                (offset..offset + m).map(|p| p as i32).collect::<Vec<_>>(),
                m,
                model.device(),
            )
            .unwrap();
            let (logits, _) = model
                .prefill_with_cache_aux_scoped(&t, &p, cache, &[], false, offset + m == ids.len())
                .unwrap();
            offset += m;
            if offset == ids.len() {
                let logits = logits.expect("final prefill logits");
                let row = logits
                    .narrow(1, m - 1, 1)
                    .unwrap()
                    .flatten_all()
                    .unwrap()
                    .to_vec1::<f32>()
                    .unwrap();
                anchor = crate::laguna_dflash::checked_target_argmax_row(&row);
            }
        }
        assert_eq!(cache.current_len(), ids.len());
        assert!(
            cache.direct_fp8_prefill_enqueued_calls() > 0,
            "direct FP8 prefill did not execute"
        );
        anchor.expect("finite anchor")
    }

    fn prefill(g: &mut LagunaBatchStepGraph, prompts: &[Vec<u32>; 2]) -> [u32; 2] {
        std::array::from_fn(|lane| prefill_one(g, lane, &prompts[lane]))
    }

    fn cache_bytes(cache: &mut LagunaKvCacheFp8, layers: &[usize]) -> Vec<u16> {
        let mut bytes = Vec::new();
        for &layer in layers {
            let (k, v) = cache.view(layer, cache.current_len()).unwrap();
            for tensor in [k, v] {
                let row = tensor
                    .flatten_all()
                    .unwrap()
                    .to_vec1::<half::bf16>()
                    .unwrap();
                bytes.extend(row.into_iter().map(half::bf16::to_bits));
            }
        }
        bytes
    }

    struct Observation {
        rows: Vec<Vec<u32>>,
        state: [Vec<u16>; 2],
        lengths: [usize; 2],
    }

    fn sequence(
        g: &mut LagunaBatchStepGraph,
        mut tokens: [u32; 2],
        replacement: &[u32],
        captured: bool,
    ) -> Observation {
        let probe_layers = [
            g.model
                .config()
                .layer_types
                .iter()
                .position(|k| matches!(k, LayerType::FullAttention))
                .unwrap(),
            g.model
                .config()
                .layer_types
                .iter()
                .position(|k| matches!(k, LayerType::SlidingAttention))
                .unwrap(),
        ];
        let mut rows = Vec::new();
        let mut fed: [Vec<u32>; 2] = std::array::from_fn(|_| Vec::new());
        for (round, active) in MASKS.into_iter().enumerate() {
            if round == 3 {
                let other_len = g.caches[1].current_len();
                let other_bytes = cache_bytes(&mut g.caches[1], &probe_layers);
                tokens[0] = prefill_one(g, 0, replacement);
                assert_eq!(g.caches[1].current_len(), other_len);
                assert_eq!(
                    cache_bytes(&mut g.caches[1], &probe_layers),
                    other_bytes,
                    "replacing one physical lane changed the retained active lane"
                );
            }
            let before = g.current_lengths();
            let inactive = (active.len() == 1).then(|| 1 - active[0]);
            let inactive_bytes = inactive.map(|i| cache_bytes(&mut g.caches[i], &probe_layers));
            for &lane in active {
                fed[lane].push(tokens[lane]);
            }
            let t: Vec<u32> = active.iter().map(|&i| tokens[i]).collect();
            let p: Vec<usize> = active.iter().map(|&i| before[i]).collect();
            let logits = g.run(active, &t, &p, captured).unwrap();
            assert_eq!(logits.len(), active.len() * g.model.config().vocab_size);
            assert!(
                logits.iter().all(|v| v.is_finite()),
                "nonfinite full target logits"
            );
            for (row, &lane) in logits.chunks_exact(g.model.config().vocab_size).zip(active) {
                crate::laguna_dflash::checked_target_argmax_row(row).unwrap();
                tokens[lane] = 1000 + (round as u32) * 37 + (lane as u32) * 101;
                assert!((tokens[lane] as usize) < g.model.config().vocab_size);
            }
            rows.push(logits.into_iter().map(f32::to_bits).collect());
            let after = g.current_lengths();
            for lane in 0..2 {
                assert_eq!(
                    after[lane],
                    before[lane] + usize::from(active.contains(&lane))
                );
            }
            if let Some(i) = inactive {
                assert_eq!(
                    cache_bytes(&mut g.caches[i], &probe_layers),
                    inactive_bytes.unwrap(),
                    "inactive physical cache changed"
                );
            }
        }
        for values in &fed {
            assert!(
                values
                    .iter()
                    .copied()
                    .collect::<std::collections::HashSet<_>>()
                    .len()
                    > 2,
                "persistent token-input replay test must feed changing token IDs"
            );
        }
        eprintln!("[laguna_b2_graph_test] captured={captured} fed_ids={fed:?}");
        let layers: Vec<usize> = (0..g.model.config().num_hidden_layers).collect();
        Observation {
            rows,
            lengths: g.current_lengths(),
            state: [
                cache_bytes(&mut g.caches[0], &layers),
                cache_bytes(&mut g.caches[1], &layers),
            ],
        }
    }

    fn file_sha(path: &std::path::Path) -> String {
        let output = std::process::Command::new("sha256sum")
            .arg(path)
            .output()
            .expect("sha256sum required");
        assert!(
            output.status.success(),
            "sha256sum failed for {}",
            path.display()
        );
        let text = String::from_utf8(output.stdout).unwrap();
        let sha = text.split_whitespace().next().unwrap();
        assert_eq!(sha.len(), 64);
        sha.to_owned()
    }

    #[test]
    #[ignore]
    fn packed_m3_m6_moe_fusion_reports_real_inputs_and_preserves_owned_outputs() {
        let (model, tokenizer) = load_target();
        crate::laguna::moe_fusion_validation::run(&model, &tokenizer).unwrap();
    }

    fn load_target() -> (Arc<Laguna>, tokenizers::Tokenizer) {
        let dir = std::path::PathBuf::from(
            std::env::var("NV_LAGUNA_TEST_MODEL_DIR")
                .expect("explicit pinned model directory required"),
        );
        let resolved = dir.canonicalize().expect("model directory exists");
        let receipt_path = std::env::var("NV_LAGUNA_TEST_MODEL_MANIFEST")
            .expect("pinned target content manifest required");
        let receipt: serde_json::Value =
            serde_json::from_slice(&std::fs::read(&receipt_path).unwrap()).unwrap();
        assert_eq!(receipt["revision"].as_str(), Some(TARGET_REVISION));
        assert_eq!(
            receipt["config_sha256"].as_str(),
            Some(TARGET_CONFIG_SHA256)
        );
        assert_eq!(file_sha(&dir.join("config.json")), TARGET_CONFIG_SHA256);
        let expected = receipt["weights"]
            .as_object()
            .expect("weights must map safetensors filenames to SHA256");
        assert!(!expected.is_empty());
        let actual_names: std::collections::BTreeSet<String> = std::fs::read_dir(&dir)
            .unwrap()
            .map(|e| e.unwrap().file_name().into_string().unwrap())
            .filter(|s| s.ends_with(".safetensors"))
            .collect();
        assert_eq!(
            actual_names,
            expected.keys().cloned().collect(),
            "manifest must cover every loaded safetensors file"
        );
        for (name, expected_sha) in expected {
            assert_eq!(
                std::path::Path::new(name).file_name().unwrap().to_str(),
                Some(name.as_str())
            );
            assert_eq!(
                file_sha(&dir.join(name)),
                expected_sha.as_str().unwrap(),
                "target shard content mismatch: {name}"
            );
        }
        eprintln!("[laguna_b2_graph_test] target={} revision={TARGET_REVISION} content_manifest={} manifest_sha256={}",
            resolved.display(), receipt_path, file_sha(std::path::Path::new(&receipt_path)));
        let raw = std::fs::read_to_string(dir.join("config.json")).unwrap();
        let config = LagunaConfig::from_hf_json_str(&raw).unwrap();
        let quant = QuantizationConfig::from_hf_json_str(&raw).unwrap();
        assert_eq!(quant.scheme, nv_weights::QuantScheme::Nvfp4);
        let device = Device::new_cuda_with_stream(0)
            .expect("real CUDA model capture stream for m2 and m2-spec");
        if let Device::Cuda(dev) = &device {
            crate::gemma4_batch_graph::graph_teardown::disable_event_tracking_before_capture(
                dev.cuda_stream().context(),
            );
        }
        let weights = WeightLoader::open_dir(&dir, &device).unwrap();
        let model =
            Arc::new(Laguna::from_loader_quantized(config, &weights, &quant, &device).unwrap());
        drop(weights);
        assert_eq!(
            model.prefill_chunk_tokens(),
            256,
            "fixed chunk is part of this graph-state gate"
        );
        let tokenizer = tokenizers::Tokenizer::from_file(dir.join("tokenizer.json")).unwrap();
        (model, tokenizer)
    }

    #[test]
    #[ignore]
    fn fixed_one_b2_prepared_graph_matches_same_body_and_reuses_all_physical_masks() {
        assert_eq!(
            std::env::var("NV_LAGUNA_B2_GRAPH_TEST").as_deref(),
            Ok("1"),
            "explicit real CUDA test opt-in required"
        );
        assert_eq!(std::env::var("NV_LAGUNA_BATCH_GRAPH").as_deref(), Ok("m2"));
        crate::laguna_serve::validate_m2_graph_selection().unwrap();
        let (model, tokenizer) = load_target();
        let seed = [
            "A copper sphere rolls across the wooden table. ",
            "Explain why a careful cache preserves each request's history. ",
        ];
        let scenarios: Vec<[Vec<u32>; 2]> = [[767, 511], [1535, 1023]]
            .into_iter()
            .map(|depths| {
                std::array::from_fn(|lane| {
                    let text = seed[lane].repeat(depths[lane]);
                    let mut ids = tokenizer.encode(text, true).unwrap().get_ids().to_vec();
                    assert!(ids.len() >= depths[lane]);
                    ids.truncate(depths[lane]);
                    ids
                })
            })
            .collect();
        let device = model.device().clone();
        let mut g = LagunaBatchStepGraph::new(model, CAPACITY).unwrap();
        let mut expected = Vec::new();
        for prompts in &scenarios {
            let anchors = prefill(&mut g, prompts);
            expected.push(sequence(&mut g, anchors, &prompts[1][..509], false));
        }
        assert_eq!(g.captured_masks(), [false; 3]);
        let mut first_graph_output: Option<Vec<u32>> = None;
        for (index, prompts) in scenarios.iter().enumerate() {
            let anchors = prefill(&mut g, prompts);
            let observed = sequence(&mut g, anchors, &prompts[1][..509], true);
            assert_eq!(
                observed.rows, expected[index].rows,
                "same-body captured full logits differ, scenario {index}"
            );
            assert_eq!(
                observed.state, expected[index].state,
                "all full/sliding initialized cache values differ, scenario {index}"
            );
            assert_eq!(observed.lengths, expected[index].lengths);
            assert_eq!(g.captured_masks(), [true; 3]);
            if let Some(first) = &first_graph_output {
                assert_ne!(
                    &observed.rows[0], first,
                    "changed-prefix exercise is vacuous"
                );
            }
            first_graph_output = Some(observed.rows[0].clone());
        }
        assert!(g.replay_counts().iter().all(|&n| n >= 3));
        g.report("same-body-gate-passed");
        drop(g);
        device.synchronize().unwrap();
    }
    #[derive(Debug, PartialEq)]
    struct SpecObservation {
        logits: Vec<Vec<u32>>,
        auxiliary: Vec<Vec<u16>>,
        draft_logits: Vec<Vec<u32>>,
        draft_cache: [Vec<u16>; 2],
        acceptance: Vec<Vec<usize>>,
        cache: [Vec<u16>; 2],
        lengths: [usize; 2],
    }

    fn spec_prefill(
        g: &mut LagunaBatchStepGraph,
        draft: &crate::laguna_dflash::LagunaDflash,
        ctx: &mut crate::laguna_dflash::DflashCtxCache,
        lane: usize,
        ids: &[u32],
    ) -> u32 {
        g.model.device().synchronize().unwrap();
        let model = g.model.clone();
        let taps = g.verify.as_ref().unwrap().aux_layers.clone();
        let cache = g.cache_mut(lane).unwrap();
        cache.reset();
        ctx.reset();
        let mut offset = 0;
        let mut anchor = None;
        while offset < ids.len() {
            let m = model.next_prefill_chunk_len(ids.len() - offset);
            let positions: Vec<i32> = (offset..offset + m)
                .map(|p| i32::try_from(p).unwrap())
                .collect();
            let tokens =
                Tensor::from_vec(ids[offset..offset + m].to_vec(), (1, m), model.device()).unwrap();
            let pos = Tensor::from_vec(positions.clone(), m, model.device()).unwrap();
            let (logits, aux) = model
                .prefill_with_cache_aux_scoped(
                    &tokens,
                    &pos,
                    cache,
                    &taps,
                    true,
                    offset + m == ids.len(),
                )
                .unwrap();
            let combined = draft.combine_aux(&aux).unwrap();
            draft.append_context(ctx, &combined, &positions).unwrap();
            offset += m;
            if let Some(logits) = logits {
                let last = logits
                    .narrow(1, logits.dims()[1] - 1, 1)
                    .unwrap()
                    .flatten_all()
                    .unwrap()
                    .to_vec1::<f32>()
                    .unwrap();
                anchor = crate::laguna_dflash::checked_target_argmax_row(&last);
            }
        }
        assert_eq!(ctx.len(), ids.len());
        assert_eq!(cache.current_len(), ids.len());
        anchor.unwrap()
    }

    fn spec_sequence(
        g: &mut LagunaBatchStepGraph,
        draft: &crate::laguna_dflash::LagunaDflash,
        ctx: &mut [crate::laguna_dflash::DflashCtxCache; 2],
        proposer: &mut [crate::laguna_dflash::DflashGraphProposer; 2],
        mut anchor: [u32; 2],
        replacement: &[u32],
        captured: bool,
    ) -> SpecObservation {
        use crate::laguna_dflash::{accept_block_on_device_with_status, checked_target_argmax_row};
        let mut observation = SpecObservation {
            logits: Vec::new(),
            auxiliary: Vec::new(),
            draft_logits: Vec::new(),
            acceptance: Vec::new(),
            cache: [Vec::new(), Vec::new()],
            draft_cache: [Vec::new(), Vec::new()],
            lengths: [0; 2],
        };
        let mut commits = captured.then(|| {
            std::array::from_fn::<_, 2, _>(|lane| {
                crate::laguna_dflash::context_commit::DflashCtxCommitGraph::new(draft, &ctx[lane])
                    .unwrap()
            })
        });
        let mut slots: [std::collections::HashMap<usize, crate::laguna_dflash::AcceptSlots>; 2] =
            std::array::from_fn(|_| std::collections::HashMap::new());
        let all_layers: Vec<usize> = (0..g.model.layers().len()).collect();
        let pointers = g.caches.each_ref().map(|c| c.ring_meta_ptr());
        let mut fed: [Vec<u32>; 2] = std::array::from_fn(|_| Vec::new());
        let masks: [&[usize]; 8] = [&[0, 1], &[0], &[1], &[0, 1], &[1], &[0], &[1], &[0, 1]];
        for (round, active) in masks.into_iter().enumerate() {
            if round == 3 {
                let before = cache_bytes(&mut g.caches[1], &all_layers);
                let before_ctx = ctx[1].len();
                anchor[0] = spec_prefill(g, draft, &mut ctx[0], 0, replacement);
                assert_eq!(
                    cache_bytes(&mut g.caches[1], &all_layers),
                    before,
                    "canceled/reused lane changed other cache"
                );
                assert_eq!(ctx[1].len(), before_ctx);
                assert_eq!(g.caches.each_ref().map(|c| c.ring_meta_ptr()), pointers);
            }
            let rows = if round >= 5 { 1 } else { 3 };
            let positions: Vec<usize> = active.iter().map(|&i| g.current_lengths()[i]).collect();
            let inactive = (active.len() == 1).then(|| 1 - active[0]);
            let inactive_cache = inactive.map(|i| cache_bytes(&mut g.caches[i], &all_layers));
            let before_lengths = g.current_lengths();
            let mut blocks = Vec::new();
            for &lane in active {
                assert_eq!(ctx[lane].len(), g.current_lengths()[lane]);
                let ids = proposer[lane]
                    .propose(
                        draft,
                        &ctx[lane],
                        anchor[lane],
                        ctx[lane].len(),
                        g.model.as_ref(),
                        g.model.lm_head(),
                    )
                    .unwrap();
                assert_eq!(ids.len(), 2);
                let values = proposer[lane].recorded_logits_host().unwrap();
                assert!(values.iter().all(|v| v.is_finite()));
                observation
                    .draft_logits
                    .push(values.into_iter().map(f32::to_bits).collect());
                fed[lane].push(anchor[lane]);
                blocks.push(anchor[lane]);
                if rows == 3 {
                    blocks.extend_from_slice(&ids);
                }
            }
            let desired: Vec<usize> = active
                .iter()
                .map(|&lane| if rows == 3 { (round + lane) % 3 } else { 0 })
                .collect();
            if rows == 3 {
                for next in 0..2 {
                    let (logits, _) = g
                        .verify_run(active, &blocks, &positions, rows, false)
                        .unwrap();
                    let host = logits.flatten_all().unwrap().to_vec1::<f32>().unwrap();
                    for (slot, &lane) in active.iter().enumerate() {
                        let start = (slot * rows + next) * g.model.config().vocab_size;
                        let chosen = checked_target_argmax_row(
                            &host[start..start + g.model.config().vocab_size],
                        )
                        .unwrap();
                        blocks[slot * rows + next + 1] = if next < desired[slot] {
                            chosen
                        } else {
                            (chosen + 1) % g.model.config().vocab_size as u32
                        };
                        g.cache_mut(lane).unwrap().rollback(rows).unwrap();
                    }
                }
            }
            let (logits, aux) = g
                .verify_run(active, &blocks, &positions, rows, captured)
                .unwrap();
            let values = logits.flatten_all().unwrap().to_vec1::<f32>().unwrap();
            assert!(values.iter().all(|v| v.is_finite()));
            observation
                .logits
                .push(values.into_iter().map(f32::to_bits).collect());
            let mut aux_bits = Vec::new();
            for value in &aux {
                let values = value
                    .flatten_all()
                    .unwrap()
                    .to_vec1::<half::bf16>()
                    .unwrap();
                assert!(values.iter().all(|v| v.is_finite()));
                aux_bits.extend(values.into_iter().map(half::bf16::to_bits));
            }
            observation.auxiliary.push(aux_bits);
            let mut accepted = Vec::new();
            for (slot, &lane) in active.iter().enumerate() {
                let lane_logits = logits.narrow(1, slot * rows, rows).unwrap();
                let answer = if rows == 3 {
                    accept_block_on_device_with_status(
                        &g.model,
                        &mut slots[lane],
                        &lane_logits,
                        &blocks[slot * rows + 1..(slot + 1) * rows],
                        None,
                        None,
                    )
                    .unwrap()
                } else {
                    (
                        0,
                        vec![checked_target_argmax_row(
                            &lane_logits.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
                        )
                        .unwrap()],
                    )
                };
                assert_eq!(
                    answer.0, desired[slot],
                    "constructed causal acceptance case differs"
                );
                assert_eq!(answer.1.len(), answer.0 + 1);
                accepted.push(answer);
            }
            for (slot, &lane) in active.iter().enumerate() {
                let consumed = accepted[slot].0 + 1;
                g.cache_mut(lane)
                    .unwrap()
                    .rollback(rows - consumed)
                    .unwrap();
                let kept: Vec<Tensor> = aux
                    .iter()
                    .map(|a| a.narrow(1, slot * rows, consumed).unwrap())
                    .collect();
                let positions: Vec<i32> = (positions[slot]..positions[slot] + consumed)
                    .map(|i| i as i32)
                    .collect();
                if let Some(commits) = commits.as_mut() {
                    commits[lane]
                        .enqueue(draft, &ctx[lane], &kept, &positions)
                        .unwrap();
                    assert_eq!(
                        ctx[lane].len(),
                        before_lengths[lane],
                        "queued commit advanced the draft frontier"
                    );
                } else {
                    let combined = draft.combine_aux(&kept).unwrap();
                    draft
                        .append_context(&mut ctx[lane], &combined, &positions)
                        .unwrap();
                }
                assert_eq!(g.current_lengths()[lane], before_lengths[lane] + consumed);
                anchor[lane] = 1000 + round as u32 * 37 + lane as u32 * 101;
            }
            g.model.device().synchronize().unwrap();
            for &lane in active {
                if let Some(commits) = commits.as_mut() {
                    commits[lane].finish_after_sync(&mut ctx[lane]).unwrap();
                }
                assert_eq!(ctx[lane].len(), g.current_lengths()[lane]);
            }
            observation
                .acceptance
                .push(accepted.iter().map(|a| a.0).collect());
            if let Some(lane) = inactive {
                assert_eq!(g.current_lengths()[lane], before_lengths[lane]);
                assert_eq!(
                    cache_bytes(&mut g.caches[lane], &all_layers),
                    inactive_cache.unwrap()
                );
                assert_eq!(ctx[lane].len(), before_lengths[lane]);
            }
        }
        for lane in 0..2 {
            proposer[lane]
                .propose(
                    draft,
                    &ctx[lane],
                    anchor[lane],
                    ctx[lane].len(),
                    g.model.as_ref(),
                    g.model.lm_head(),
                )
                .unwrap();
            observation.draft_logits.push(
                proposer[lane]
                    .recorded_logits_host()
                    .unwrap()
                    .into_iter()
                    .map(f32::to_bits)
                    .collect(),
            );
            observation.cache[lane] = cache_bytes(&mut g.caches[lane], &all_layers);
            observation.draft_cache[lane] = ctx[lane].initialized_ring_rows_for_test().unwrap();
            assert!(
                fed[lane]
                    .iter()
                    .collect::<std::collections::HashSet<_>>()
                    .len()
                    > 2
            );
        }
        observation.lengths = g.current_lengths();
        eprintln!("[laguna_b2_spec_gate] captured={captured} fed_ids={fed:?} acceptance={:?} lengths={:?}",observation.acceptance, observation.lengths);
        observation
    }

    #[test]
    #[ignore = "real packed Laguna target/draft context commit comparison; NV_LAGUNA_M6_CONTEXT_TEST=1"]
    fn m6_context_commit_graph_preserves_real_auxiliary_window_and_proposer() {
        assert_eq!(
            std::env::var("NV_LAGUNA_M6_CONTEXT_TEST").as_deref(),
            Ok("1"),
            "explicit real GPU gate opt-in required"
        );
        assert_eq!(std::env::var("NV_LAGUNA_TEST").as_deref(), Ok("1"));
        assert_eq!(
            std::env::var("NV_LAGUNA_WEIGHT_LINEARS_FP8").as_deref(),
            Ok("1")
        );
        crate::laguna_serve::validate_m2_spec_selection().unwrap();
        let dir = std::path::PathBuf::from(
            std::env::var("NV_LAGUNA_DFLASH_DIR").expect("pinned target-trained draft required"),
        );
        assert_eq!(
            file_sha(&dir.join("model.safetensors")),
            "67beb7f0dcb8a121ef912782396335cb5103e39f30328decdc83981509d4db84"
        );
        assert_eq!(
            file_sha(&dir.join("config.json")),
            "22bdc5fd9fd41dc6945523fd504f1b70809af0f5ab26c09716e33616033ecf9a"
        );
        let (target, tokenizer) = load_target();
        let draft = crate::laguna_serve::load_dflash_draft(&dir, target.device()).unwrap();
        let packed = draft.quantization_summary();
        assert_eq!(
            (
                packed.packed_linears,
                packed.excluded_linears,
                packed.bf16_fallback_linears
            ),
            (36, 5, 0)
        );
        assert_eq!(packed.weight_format, "nvfp4-derived");
        assert_eq!(
            (packed.fp8_linears, packed.fp8_norms, packed.excluded_norms),
            (5, 27, 0)
        );
        crate::laguna_dflash::context_commit::real_weight_context_commit_gate(
            &target, &draft, &tokenizer,
        )
        .unwrap();
    }

    #[test]
    #[ignore]
    fn fixed_one_b2_k2_spec_graph_matches_same_body_acceptance_aux_and_lane_reuse() {
        assert_eq!(
            std::env::var("NV_LAGUNA_B2_SPEC_TEST").as_deref(),
            Ok("1"),
            "explicit real CUDA opt-in required"
        );
        assert_eq!(std::env::var("NV_LAGUNA_TEST").as_deref(), Ok("1"));
        assert_eq!(
            std::env::var("NV_LAGUNA_BATCH_GRAPH").as_deref(),
            Ok("m2-spec")
        );
        crate::laguna_serve::validate_m2_spec_selection().unwrap();
        let draft_dir = std::path::PathBuf::from(
            std::env::var("NV_LAGUNA_DFLASH_DIR").expect("explicit target-trained draft required"),
        );
        assert_eq!(
            file_sha(&draft_dir.join("model.safetensors")),
            "67beb7f0dcb8a121ef912782396335cb5103e39f30328decdc83981509d4db84"
        );
        assert_eq!(
            file_sha(&draft_dir.join("config.json")),
            "22bdc5fd9fd41dc6945523fd504f1b70809af0f5ab26c09716e33616033ecf9a"
        );
        eprintln!("[laguna_b2_spec_gate] draft_source={} revision=6c0564233472e9572ec3d9ecc6025a1dc77799e9 source_precision=bf16 effective_large_weights=nvfp4-derived",draft_dir.canonicalize().unwrap().display());
        let (model, tokenizer) = load_target();
        let draft = crate::laguna_serve::load_dflash_draft(&draft_dir, model.device()).unwrap();
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
        let taps = crate::laguna_dflash::resolve_tap_layers(
            draft.config(),
            crate::laguna_dflash::tap_list_mode(),
        );
        let mut graph = LagunaBatchStepGraph::new_spec(model, CAPACITY, &taps).unwrap();
        let mut ctx = std::array::from_fn(|_| draft.new_ctx_cache());
        assert!(ctx.iter().all(|c| c.has_ring()));
        let mut proposers = std::array::from_fn(|_| {
            crate::laguna_dflash::DflashGraphProposer::new_recording_logits(&draft, 2).unwrap()
        });
        let seeds = [
            "A copper sphere rolls across the wooden table. ",
            "Explain why each cache retains its own history. ",
        ];
        for depths in [[767usize, 511usize], [1535, 1023]] {
            let prompts: [Vec<u32>; 2] = std::array::from_fn(|lane| {
                let mut ids = tokenizer
                    .encode(seeds[lane].repeat(depths[lane]), true)
                    .unwrap()
                    .get_ids()
                    .to_vec();
                assert!(ids.len() >= depths[lane]);
                ids.truncate(depths[lane]);
                ids
            });
            let mut anchors = [0; 2];
            for lane in 0..2 {
                anchors[lane] =
                    spec_prefill(&mut graph, &draft, &mut ctx[lane], lane, &prompts[lane]);
            }
            let expected = spec_sequence(
                &mut graph,
                &draft,
                &mut ctx,
                &mut proposers,
                anchors,
                &prompts[1][..509],
                false,
            );
            for lane in 0..2 {
                anchors[lane] =
                    spec_prefill(&mut graph, &draft, &mut ctx[lane], lane, &prompts[lane]);
            }
            let actual = spec_sequence(
                &mut graph,
                &draft,
                &mut ctx,
                &mut proposers,
                anchors,
                &prompts[1][..509],
                true,
            );
            assert_eq!(actual,expected,"same-body spec capture changed logits, aux, acceptance, draft trajectory or initialized target KV");
        }
        let before = graph.current_lengths();
        assert!(graph.verify(&[0], &[100], &[CAPACITY], 1).is_err());
        assert_eq!(
            before,
            graph.current_lengths(),
            "invalid capacity admission changed state"
        );
        let buffers = graph.verify.as_ref().unwrap();
        for key in [1usize, 2, 3, 5, 6, 7] {
            assert!(buffers.runner.has_cached_token(0x2000 + key as u64));
        }
        assert!(buffers.replays.iter().sum::<u64>() > 0);
        graph.report_spec("same-body-spec-gate-passed");
        eprintln!("LAGUNA-B2-SPEC-GATE real_cuda=true physical_lanes=2 capacity_per_lane=262144 k=2 target_rows=1,2,3,6 masks=01,10,11 packed_draft=36/5/0 same_body_logits=bitwise aux=bitwise draft_graph_logits=bitwise initialized_target_kv=bitwise forced_acceptance=0,1,2 lane_reuse=true actual_near_cap_prefill=false http_qualification=false");
        let target = graph.model.clone();
        drop(proposers);
        let borrowed_key = graph.stream.cu_stream() as usize;
        assert_eq!(
            graph.verify.as_ref().unwrap().model_stream.cu_stream() as usize,
            borrowed_key
        );
        let legacy_key = graph.stream.context().default_stream().cu_stream() as usize;
        let borrowed_epoch = nv_quant::stream_epoch(borrowed_key);
        let legacy_epoch = nv_quant::stream_epoch(legacy_key);
        drop(graph);
        assert_eq!(
            nv_quant::stream_epoch(borrowed_key),
            borrowed_epoch,
            "M6 teardown released borrowed model-stream resources"
        );
        assert_eq!(
            nv_quant::stream_epoch(legacy_key),
            legacy_epoch,
            "M6 teardown released unrelated legacy-stream resources"
        );
        drop(ctx);
        let mut worker =
            crate::laguna_serve::BatchM1State::new_spec(&target, draft, 2, CAPACITY).unwrap();
        let events: Arc<std::sync::Mutex<Vec<(usize, Vec<u32>, bool)>>> =
            Arc::new(std::sync::Mutex::new(Vec::new()));
        let jobs = (0..2)
            .map(|lane| {
                let events = events.clone();
                crate::laguna_serve::SpecServeJob {
                    prompt_ids: (100..164).collect(),
                    prompt_text: String::new(),
                    max_new: 8,
                    eos_ids: Vec::new(),
                    emit: Box::new(move |event| {
                        match event {
                            crate::laguna_serve::SpecServeEvent::Tokens(tokens)
                                if !tokens.is_empty() =>
                            {
                                events.lock().unwrap().push((lane, tokens, false))
                            }
                            crate::laguna_serve::SpecServeEvent::Done => {
                                events.lock().unwrap().push((lane, Vec::new(), true))
                            }
                            crate::laguna_serve::SpecServeEvent::Error(error) => {
                                panic!("B2 packed serving gate failed: {error}")
                            }
                            _ => {}
                        }
                        true
                    }),
                }
            })
            .collect();
        worker.serve_batch(jobs).unwrap();
        let events = events.lock().unwrap();
        for lane in 0..2 {
            assert_eq!(
                events
                    .iter()
                    .filter(|(i, _, _)| *i == lane)
                    .map(|(_, t, _)| t.len())
                    .sum::<usize>(),
                8
            );
            assert_eq!(
                events
                    .iter()
                    .filter(|(i, _, done)| *i == lane && *done)
                    .count(),
                1
            );
        }
        drop(events);
        drop(worker);
    }
}
