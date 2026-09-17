use super::*;

struct PendingCommit {
    start: usize,
    stored: usize,
    end: usize,
    next_stored: usize,
    rows: usize,
    route: usize,
}

pub(crate) struct DflashCtxCommitGraph {
    runner: CudaGraphRunner,
    stream: Arc<CudaStream>,
    ring: DflashCtxRing,
    auxiliary: Vec<Tensor>,
    positions: Tensor,
    views: Vec<(Vec<Tensor>, Tensor)>,
    staging_ranges: Vec<(u64, u64)>,
    host_positions: Box<[i32; 3]>,
    mode: NormMode,
    theta: u32,
    hidden: usize,
    pending: Option<PendingCommit>,
    healthy: bool,
    completed: [[u64; 3]; 3],
}

impl DflashCtxCommitGraph {
    pub(crate) fn retained_bytes(hidden: usize) -> Result<u64> {
        (hidden as u64)
            .checked_mul(5 * 3 * 2)
            .and_then(|n| n.checked_add(3 * 4))
            .context("M6 context commit staging size overflow")
    }

    pub(crate) fn new(draft: &LagunaDflash, cache: &DflashCtxCache) -> Result<Self> {
        let Device::Cuda(dev) = &draft.device else {
            anyhow::bail!("M6 context commit requires CUDA")
        };
        let stream = dev.cuda_stream();
        anyhow::ensure!(!stream.cu_stream().is_null()
            && !nv_layers::cuda_stream::is_overridden() && !stream.context().is_event_tracking(),
            "M6 context commit requires the explicit model stream with capture event tracking disabled");
        let ring = cache
            .ring
            .as_ref()
            .context("M6 context commit requires a resident draft ring")?;
        anyhow::ensure!(
            draft.dtype == DType::BF16
                && draft.config.num_aux() == 5
                && draft.layers.len() == 5
                && ring.window >= 3
                && ring.k.len() == draft.layers.len()
                && ring.v.len() == ring.k.len()
                && ring.row_elems == draft.config.num_key_value_heads * draft.config.head_dim,
            "M6 context commit draft geometry differs"
        );
        nv_quant::nvfp4::ensure_workspace_for_stream(&stream)?;
        let auxiliary = (0..5)
            .map(|_| Tensor::zeros((1, 3, draft.config.hidden_size), DType::BF16, &draft.device))
            .collect::<candle_core::Result<Vec<_>>>()?;
        let positions = Tensor::zeros(3, DType::I32, &draft.device)?;
        let views = (1..=3)
            .map(|rows| {
                Ok((
                    auxiliary
                        .iter()
                        .map(|a| a.narrow(1, 0, rows))
                        .collect::<candle_core::Result<Vec<_>>>()?,
                    positions.narrow(0, 0, rows)?,
                ))
            })
            .collect::<candle_core::Result<Vec<_>>>()?;
        let staging_ranges = auxiliary
            .iter()
            .map(|tensor| {
                let start = pointer_bf16(tensor, &stream)?;
                let bytes = u64::try_from(tensor.elem_count())?
                    .checked_mul(2)
                    .context("M6 staging bytes overflow")?;
                Ok((
                    start,
                    start
                        .checked_add(bytes)
                        .context("M6 staging pointer overflow")?,
                ))
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            runner: CudaGraphRunner::new(stream.clone()),
            stream,
            ring: DflashCtxRing {
                k: ring.k.clone(),
                v: ring.v.clone(),
                scratch: ring.scratch.clone(),
                stored: ring.window,
                cap: ring.cap,
                window: ring.window,
                row_elems: ring.row_elems,
            },
            auxiliary,
            positions,
            views,
            staging_ranges,
            host_positions: Box::new([0; 3]),
            mode: norm_mode(),
            theta: draft.active_rope_theta().to_bits(),
            hidden: draft.config.hidden_size,
            pending: None,
            healthy: true,
            completed: [[0; 3]; 3],
        })
    }

    pub(crate) fn report(&self, label: &str) {
        eprintln!("[laguna_m6_context_commit] {label} rows=1,2,3 eager={:?} captured={:?} successful_replays={:?} pending={} healthy={}",
            self.completed[0], self.completed[1], self.completed[2], self.pending.is_some(), self.healthy);
    }

    pub(crate) fn require_idle(&self) -> Result<()> {
        anyhow::ensure!(
            self.healthy && self.pending.is_none(),
            "engine busy: M6 context commit is retired or has unfinished GPU work; try later."
        );
        Ok(())
    }

    fn check_ring(&self, cache: &DflashCtxCache) -> Result<()> {
        let ring = cache.ring.as_ref().context("M6 context ring disappeared")?;
        anyhow::ensure!(
            ring.cap == self.ring.cap
                && ring.window == self.ring.window
                && ring.row_elems == self.ring.row_elems
                && ring.k.len() == self.ring.k.len()
                && ring.v.len() == self.ring.v.len(),
            "M6 context ring geometry changed after binding"
        );
        for (actual, bound) in ring
            .k
            .iter()
            .chain(&ring.v)
            .chain(std::iter::once(&ring.scratch))
            .zip(
                self.ring
                    .k
                    .iter()
                    .chain(&self.ring.v)
                    .chain(std::iter::once(&self.ring.scratch)),
            )
        {
            anyhow::ensure!(
                actual.id() == bound.id(),
                "M6 context commit cannot change physical ring pointers after capture"
            );
        }
        Ok(())
    }

    pub(crate) fn enqueue(
        &mut self,
        draft: &LagunaDflash,
        cache: &DflashCtxCache,
        auxiliary: &[Tensor],
        positions: &[i32],
    ) -> Result<()> {
        self.require_idle()?;
        self.check_ring(cache)?;
        let rows = positions.len();
        anyhow::ensure!(
            (1..=3).contains(&rows) && auxiliary.len() == 5,
            "M6 context commit requires consumed lengths 1, 2 or 3 and five auxiliary taps"
        );
        let Device::Cuda(dev) = &draft.device else {
            anyhow::bail!("M6 context commit requires CUDA")
        };
        anyhow::ensure!(
            !nv_layers::cuda_stream::is_overridden()
                && dev.cuda_stream().cu_stream() == self.stream.cu_stream()
                && draft.active_rope_theta().to_bits() == self.theta
                && matches!(norm_mode(), NormMode::Checkpoint)
                    == matches!(self.mode, NormMode::Checkpoint),
            "M6 context commit stream, RoPE theta or normalization mode changed after binding"
        );
        draft.check_rope_span(cache.len, rows)?;
        anyhow::ensure!(
            positions
                .iter()
                .enumerate()
                .all(|(i, &p)| p >= 0 && p as usize == cache.len + i),
            "M6 context commit positions must extend the current draft frontier"
        );
        for source in auxiliary {
            anyhow::ensure!(
                source.dtype() == DType::BF16
                    && source.dims() == [1, rows, self.hidden]
                    && source.is_contiguous()
                    && source.device().same_device(&draft.device),
                "M6 context commit auxiliary input shape, dtype or device differs"
            );
            let src = pointer_bf16(source, &self.stream)?;
            let bytes = u64::try_from(source.elem_count())?
                .checked_mul(2)
                .context("M6 auxiliary bytes overflow")?;
            let end = src
                .checked_add(bytes)
                .context("M6 auxiliary pointer overflow")?;
            anyhow::ensure!(
                self.staging_ranges
                    .iter()
                    .all(|&(dst, dst_end)| end <= dst || dst_end <= src),
                "M6 context commit input must not alias any retained staging buffer"
            );
        }
        let ring = cache.ring.as_ref().unwrap();
        anyhow::ensure!(
            ring.stored <= ring.window && ring.stored <= cache.len,
            "M6 context commit requires the existing bounded window policy"
        );
        let plan = plan_ctx_append(ring.stored, ring.window, ring.window, rows);
        let end = cache
            .len
            .checked_add(rows)
            .context("M6 draft frontier overflow")?;
        let result = self.enqueue_validated(draft, ring, auxiliary, positions, plan);
        match result {
            Ok(route) => {
                self.pending = Some(PendingCommit {
                    start: cache.len,
                    stored: ring.stored,
                    end,
                    next_stored: plan.keep + rows - plan.input_skip,
                    rows,
                    route,
                });
                Ok(())
            }
            Err(error) => {
                self.healthy = false;
                Err(error).context(
                    "engine busy: M6 context commit failed; worker must retire; try later.",
                )
            }
        }
    }

    fn enqueue_validated(
        &mut self,
        draft: &LagunaDflash,
        ring: &DflashCtxRing,
        auxiliary: &[Tensor],
        positions: &[i32],
        plan: CtxAppendPlan,
    ) -> Result<usize> {
        let rows = positions.len();
        self.host_positions[..rows].copy_from_slice(positions);
        for (source, destination) in auxiliary.iter().zip(&self.auxiliary) {
            copy_rows_bf16(&self.stream, source, 0, destination, 0, rows, self.hidden)?;
        }
        upload_positions(&self.stream, &self.positions, &self.host_positions[..rows])?;
        let (staged, pos) = &self.views[rows - 1];
        if ring.stored != ring.window {
            commit_body(draft, ring, staged, pos, plan, rows, self.mode, true)?;
            return Ok(0);
        }
        let Device::Cuda(dev) = &draft.device else {
            unreachable!()
        };
        let _htod_guard = dev.enable_cuda_graph_htod_cache();
        let key = rows as u64;
        let replay = self.runner.has_cached_token(key);
        if !replay {
            commit_body(draft, &self.ring, staged, pos, plan, rows, self.mode, false)?;
            self.stream.synchronize()?;
        }
        let Self {
            runner,
            stream,
            ring,
            mode,
            ..
        } = self;
        runner.run_on(key, Some(stream), |s| {
            anyhow::ensure!(
                s.cu_stream() == stream.cu_stream(),
                "M6 context graph cannot capture on another stream"
            );
            commit_body(draft, ring, staged, pos, plan, rows, *mode, true)
        })?;
        Ok(if replay { 2 } else { 1 })
    }

    pub(crate) fn finish_after_sync(&mut self, cache: &mut DflashCtxCache) -> Result<()> {
        anyhow::ensure!(
            self.healthy,
            "M6 context commit cannot finish a retired operation"
        );
        self.check_ring(cache)?;
        let pending = self
            .pending
            .as_ref()
            .context("M6 context commit has no queued operation")?;
        let ring = cache.ring.as_mut().unwrap();
        anyhow::ensure!(
            cache.len == pending.start && ring.stored == pending.stored,
            "M6 context commit frontier changed before completion"
        );
        cache.len = pending.end;
        ring.stored = pending.next_stored;
        self.completed[pending.route][pending.rows - 1] += 1;
        self.pending = None;
        Ok(())
    }
}

fn commit_body(
    draft: &LagunaDflash,
    ring: &DflashCtxRing,
    auxiliary: &[Tensor],
    positions: &Tensor,
    plan: CtxAppendPlan,
    rows: usize,
    mode: NormMode,
    write: bool,
) -> Result<()> {
    let combined = draft.combine_aux_mode(auxiliary, mode)?;
    for (li, layer) in draft.layers.iter().enumerate() {
        let (k, v) = draft.context_layer_kv(layer, &combined, positions, rows, mode)?;
        if write {
            draft.ring_append_layer(ring, li, &k, &v, plan, rows)?;
        }
    }
    Ok(())
}

fn pointer_bf16(tensor: &Tensor, stream: &Arc<CudaStream>) -> Result<u64> {
    anyhow::ensure!(
        tensor.dtype() == DType::BF16 && tensor.is_contiguous(),
        "M6 ring/staging tensor must be contiguous BF16"
    );
    let (storage, layout) = tensor.storage_and_layout();
    let candle_core::Storage::Cuda(storage) = &*storage else {
        anyhow::bail!("M6 ring/staging tensor must be CUDA")
    };
    let (ptr, _guard) = storage.as_cuda_slice::<bf16>()?.device_ptr(stream);
    ptr.checked_add(
        u64::try_from(layout.start_offset())?
            .checked_mul(2)
            .context("M6 pointer offset overflow")?,
    )
    .context("M6 pointer overflow")
}

fn upload_positions(stream: &Arc<CudaStream>, tensor: &Tensor, values: &[i32]) -> Result<()> {
    let (storage, layout) = tensor.storage_and_layout();
    let candle_core::Storage::Cuda(storage) = &*storage else {
        anyhow::bail!("M6 positions must be CUDA")
    };
    let (ptr, _guard) = storage.as_cuda_slice::<i32>()?.device_ptr(stream);
    unsafe {
        cudarc::driver::result::memcpy_htod_async(
            ptr + (layout.start_offset() * 4) as u64,
            values,
            stream.cu_stream(),
        )?;
    }
    Ok(())
}

impl Drop for DflashCtxCommitGraph {
    fn drop(&mut self) {
        self.report("worker-exit");
        let teardown = crate::gemma4_batch_graph::graph_teardown::GraphTeardown::for_a_stream_this_engine_did_not_fork(&self.stream);
        let runner = &mut self.runner;
        teardown.run_owned_streams_only(|| runner.invalidate());
    }
}

#[cfg(test)]
fn visible_ring_bits(cache: &DflashCtxCache) -> Result<(usize, usize, Vec<u16>)> {
    let ring = cache
        .ring
        .as_ref()
        .context("real context gate needs a CUDA ring")?;
    let mut values = Vec::new();
    for tensor in ring.k.iter().chain(&ring.v) {
        let host = tensor
            .narrow(1, 0, ring.stored)?
            .flatten_all()?
            .to_vec1::<bf16>()?;
        anyhow::ensure!(
            host.iter().all(|v| v.is_finite()),
            "real context gate found nonfinite visible KV"
        );
        values.extend(host.into_iter().map(bf16::to_bits));
    }
    Ok((cache.len, ring.stored, values))
}

#[cfg(test)]
pub(crate) fn real_weight_context_commit_gate(
    target: &Arc<Laguna>,
    draft: &LagunaDflash,
    tokenizer: &tokenizers::Tokenizer,
) -> Result<()> {
    anyhow::ensure!(
        draft.rope_max_seq_len() == 262144 && draft.config.hidden_size == 2048,
        "context gate requires the pinned Laguna XS geometry and 262144 rotary capacity"
    );
    let device = target.device();
    let taps = resolve_tap_layers(draft.config(), tap_list_mode());
    let seeds = [
        "A copper sphere rolls across the wooden table. ",
        "Each independent cache retains a different conversation. ",
    ];
    let mut sources: Vec<Vec<Tensor>> = Vec::new();
    let mut anchors = [0u32; 2];
    for lane in 0..2 {
        let mut ids = tokenizer
            .encode(seeds[lane].repeat(160), true)
            .map_err(|error| anyhow::anyhow!("tokenization: {error}"))?
            .get_ids()
            .to_vec();
        anyhow::ensure!(ids.len() >= 640, "context gate prompt unexpectedly short");
        ids.truncate(640);
        let mut cache = target.new_kv_cache(1024)?;
        let mut pieces = Vec::new();
        let mut offset = 0;
        let mut last_logits = None;
        while offset < ids.len() {
            let rows = target.next_prefill_chunk_len(ids.len() - offset);
            let tokens = Tensor::from_vec(ids[offset..offset + rows].to_vec(), (1, rows), device)?;
            let host_pos = (offset..offset + rows)
                .map(i32::try_from)
                .collect::<std::result::Result<Vec<_>, _>>()?;
            let positions = Tensor::from_vec(host_pos, rows, device)?;
            let (logits, aux) = target.prefill_with_cache_aux_scoped(
                &tokens,
                &positions,
                &mut cache,
                &taps,
                true,
                offset + rows == ids.len(),
            )?;
            anyhow::ensure!(
                aux.len() == 5
                    && aux
                        .iter()
                        .all(|a| a.dtype() == DType::BF16 && a.dims() == [1, rows, 2048]),
                "real target auxiliary chunk geometry differs"
            );
            pieces.push(aux);
            last_logits = logits;
            offset += rows;
        }
        let aux = (0..5)
            .map(|tap| {
                let chunks = pieces.iter().map(|chunk| &chunk[tap]).collect::<Vec<_>>();
                Tensor::cat(&chunks, 1)
            })
            .collect::<candle_core::Result<Vec<_>>>()?;
        device.synchronize()?;
        let logits = last_logits.context("real auxiliary prefill must produce final logits")?;
        anchors[lane] = checked_target_argmax_row(
            &logits
                .narrow(1, logits.dims()[1] - 1, 1)?
                .flatten_all()?
                .to_vec1::<f32>()?,
        )
        .context("real auxiliary prefill produced invalid target logits")?;
        anyhow::ensure!(
            aux.len() == 5
                && aux
                    .iter()
                    .all(|a| a.dtype() == DType::BF16 && a.dims() == [1, 640, 2048]),
            "real target auxiliary tap geometry differs"
        );
        sources.push(aux);
    }
    let mut reference = [draft.new_ctx_cache(), draft.new_ctx_cache()];
    let mut captured = [draft.new_ctx_cache(), draft.new_ctx_cache()];
    let mut commits = [
        DflashCtxCommitGraph::new(draft, &captured[0])?,
        DflashCtxCommitGraph::new(draft, &captured[1])?,
    ];
    let mut reference_proposer = [
        DflashGraphProposer::new_recording_logits(draft, 2)?,
        DflashGraphProposer::new_recording_logits(draft, 2)?,
    ];
    let mut captured_proposer = [
        DflashGraphProposer::new_recording_logits(draft, 2)?,
        DflashGraphProposer::new_recording_logits(draft, 2)?,
    ];
    let mut checked = 0usize;
    let scenarios = [(0usize, 509usize), (0, 512), (167488, 512), (261626, 512)];
    for (scenario, (base, seed_rows)) in scenarios.into_iter().enumerate() {
        for lane in 0..2 {
            commits[lane].require_idle()?;
            device.synchronize()?;
            captured[lane].reset();
            reference[lane].reset();
            captured[lane].len = base;
            reference[lane].len = base;
            let aux = sources[lane]
                .iter()
                .map(|a| a.narrow(1, lane * 3, seed_rows))
                .collect::<candle_core::Result<Vec<_>>>()?;
            let combined = draft.combine_aux(&aux)?;
            let positions = (base..base + seed_rows)
                .map(i32::try_from)
                .collect::<std::result::Result<Vec<_>, _>>()?;
            draft.append_context(&mut captured[lane], &combined, &positions)?;
            draft.append_context(&mut reference[lane], &combined, &positions)?;
        }
        device.synchronize()?;
        for lane in 0..2 {
            assert_eq!(
                visible_ring_bits(&captured[lane])?,
                visible_ring_bits(&reference[lane])?
            );
        }
        let rounds = if scenario == 3 { 3 } else { 12 };
        for round in 0..rounds {
            let active: &[usize] = if scenario == 3 || round >= 3 {
                &[0, 1]
            } else if round % 2 == 0 {
                &[0]
            } else {
                &[1]
            };
            let inactive = if active.len() == 1 {
                Some(1 - active[0])
            } else {
                None
            };
            let inactive_before = inactive
                .map(|lane| visible_ring_bits(&captured[lane]))
                .transpose()?;
            let before = captured.each_ref().map(|cache| cache.len);
            for &lane in active {
                let rows = if scenario == 3 {
                    if lane == 0 {
                        round + 1
                    } else {
                        3 - round
                    }
                } else {
                    (round + lane) % 3 + 1
                };
                let offset = 520 + (round * 7 + lane * 13) % 100;
                let aux = sources[lane]
                    .iter()
                    .map(|a| a.narrow(1, offset, rows))
                    .collect::<candle_core::Result<Vec<_>>>()?;
                let positions = (before[lane]..before[lane] + rows)
                    .map(i32::try_from)
                    .collect::<std::result::Result<Vec<_>, _>>()?;
                let combined = draft.combine_aux(&aux)?;
                draft.append_context(&mut reference[lane], &combined, &positions)?;
                commits[lane].enqueue(draft, &captured[lane], &aux, &positions)?;
                assert_eq!(
                    captured[lane].len, before[lane],
                    "queued GPU work advanced the host frontier early"
                );
                assert!(
                    commits[lane].require_idle().is_err(),
                    "pending commit must block reset/staging reuse"
                );
            }
            device.synchronize()?;
            for &lane in active {
                commits[lane].finish_after_sync(&mut captured[lane])?;
                assert_eq!(visible_ring_bits(&captured[lane])?, visible_ring_bits(&reference[lane])?,
                    "real eager/captured context differs in scenario {scenario} round {round} lane {lane}");
                checked += 1;
            }
            if let Some(lane) = inactive {
                assert_eq!(
                    visible_ring_bits(&captured[lane])?,
                    inactive_before.unwrap(),
                    "active commit mutated the inactive lane"
                );
            }
        }
        for lane in 0..2 {
            if scenario == 3 {
                assert_eq!(
                    captured[lane].len, 262144,
                    "near-end fixture must reach the exact rotary boundary"
                );
                let before = visible_ring_bits(&captured[lane])?;
                let aux = sources[lane]
                    .iter()
                    .map(|a| a.narrow(1, 600, 1))
                    .collect::<candle_core::Result<Vec<_>>>()?;
                assert!(
                    commits[lane]
                        .enqueue(draft, &captured[lane], &aux, &[262144])
                        .is_err(),
                    "overflow must fail before mutating state"
                );
                commits[lane].require_idle()?;
                assert_eq!(
                    visible_ring_bits(&captured[lane])?,
                    before,
                    "overflow mutated the initialized ring"
                );
            } else {
                let ref_ids = reference_proposer[lane].propose(
                    draft,
                    &reference[lane],
                    anchors[lane],
                    reference[lane].len,
                    target.as_ref(),
                    target.lm_head(),
                )?;
                let ids = captured_proposer[lane].propose(
                    draft,
                    &captured[lane],
                    anchors[lane],
                    captured[lane].len,
                    target.as_ref(),
                    target.lm_head(),
                )?;
                let ref_logits = reference_proposer[lane].recorded_logits_host()?;
                let logits = captured_proposer[lane].recorded_logits_host()?;
                assert!(ref_logits
                    .iter()
                    .chain(&logits)
                    .all(|value| value.is_finite()));
                assert_eq!(
                    logits.into_iter().map(f32::to_bits).collect::<Vec<_>>(),
                    ref_logits.into_iter().map(f32::to_bits).collect::<Vec<_>>(),
                    "changed context altered same-body proposer logits"
                );
                assert_eq!(ids, ref_ids);
            }
        }
        eprintln!("[laguna_m6_context_gate] scenario={scenario} base={base} seed_rows={seed_rows} positions={:?} checked_commits={checked} near_end_is_position_window_fixture=true actual_full_depth_target_prefill=false", captured.each_ref().map(|cache| cache.len));
    }
    for lane in 0..2 {
        assert!(
            commits[lane].completed[0].iter().sum::<u64>() > 0,
            "ring-not-full eager path was not exercised"
        );
        assert_eq!(
            commits[lane].completed[1], [1; 3],
            "all consumed lengths must capture exactly once and survive reset"
        );
        assert!(
            commits[lane].completed[2].iter().all(|&count| count >= 3),
            "every captured length needs repeated nonvacuous replay"
        );
        commits[lane].require_idle()?;
        captured[lane].reset();
        let seed = sources[lane]
            .iter()
            .map(|a| a.narrow(1, 0, 256))
            .collect::<candle_core::Result<Vec<_>>>()?;
        let combined = draft.combine_aux(&seed)?;
        draft.append_context(
            &mut captured[lane],
            &combined,
            &(0..256i32).collect::<Vec<_>>(),
        )?;
        device.synchronize()?;
        let before = visible_ring_bits(&captured[lane])?;
        let source = (0..5)
            .map(|i| commits[lane].auxiliary[(i + 1) % 5].narrow(1, 0, 1))
            .collect::<candle_core::Result<Vec<_>>>()?;
        assert!(
            commits[lane]
                .enqueue(draft, &captured[lane], &source, &[256])
                .is_err(),
            "cross-tap staging alias must be rejected"
        );
        assert_eq!(visible_ring_bits(&captured[lane])?, before);
    }
    let wrong_lane_aux = sources[1]
        .iter()
        .map(|a| a.narrow(1, 600, 1))
        .collect::<candle_core::Result<Vec<_>>>()?;
    let before_wrong_lane = visible_ring_bits(&captured[1])?;
    let wrong_lane = commits[0]
        .enqueue(draft, &captured[1], &wrong_lane_aux, &[256])
        .unwrap_err();
    assert!(
        format!("{wrong_lane:#}").contains("physical ring pointers"),
        "wrong-lane rejection must identify graph binding"
    );
    assert_eq!(visible_ring_bits(&captured[1])?, before_wrong_lane);
    let Device::Cuda(dev) = device else {
        unreachable!()
    };
    let shared = dev.cuda_stream().cu_stream() as usize;
    let legacy = dev.cuda_stream().context().default_stream().cu_stream() as usize;
    let epoch = nv_quant::stream_epoch(shared);
    let legacy_epoch = nv_quant::stream_epoch(legacy);
    drop(commits);
    assert_eq!(
        nv_quant::stream_epoch(shared),
        epoch,
        "commit teardown released borrowed model-stream workspaces"
    );
    assert_eq!(
        nv_quant::stream_epoch(legacy),
        legacy_epoch,
        "commit teardown released unrelated legacy workspaces"
    );
    for lane in 0..2 {
        captured_proposer[lane].propose(
            draft,
            &captured[lane],
            anchors[lane],
            captured[lane].len,
            target.as_ref(),
            target.lm_head(),
        )?;
        anyhow::ensure!(
            captured_proposer[lane]
                .recorded_logits_host()?
                .iter()
                .all(|x| x.is_finite()),
            "surviving proposer failed after borrowed commit-graph teardown"
        );
    }
    device.synchronize()?;
    eprintln!("LAGUNA-M6-CONTEXT-COMMIT-GATE real_cuda=true packed_draft=36/5/0 source=real_target_auxiliary eager_reference=existing_append_context visible_kv=bitwise next_proposer_logits=bitwise physical_lanes=2 consumed=1,2,3 ring_not_full=true ring_full=true resets=true rotary_end=262144 overflow_rejected=true alias_rejected=true host_frontier_deferred=true borrowed_stream_preserved=true checked_commits={checked} actual_full_depth_target_prefill=false http_qualification=false");
    Ok(())
}
