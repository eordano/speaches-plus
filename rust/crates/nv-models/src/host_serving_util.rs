use anyhow::Result;

#[cfg(feature = "cuda")]
use std::sync::Arc;

#[cfg(feature = "cuda")]
use cudarc::driver::{CudaContext, CudaEvent, CudaSlice, CudaStream};

#[cfg(feature = "cuda")]
pub struct CachedPinnedHostStagingViaCuMemHostAllocFlags0BecauseCudarcAllocPinnedIsWriteCombinedAndHostReadsOfWcAreUncachedAndPageableDtohReallocatesEveryStep<
    T: Copy,
> {
    ptr: *mut T,
    len: usize,
}

#[cfg(feature = "cuda")]
impl<T: Copy>
    CachedPinnedHostStagingViaCuMemHostAllocFlags0BecauseCudarcAllocPinnedIsWriteCombinedAndHostReadsOfWcAreUncachedAndPageableDtohReallocatesEveryStep<T>
{
    pub const SEND_SYNC_BECAUSE_THE_PINNED_ALLOCATION_IS_UNIQUELY_OWNED_BY_THIS_VALUE_AND_FREED_ONLY_IN_DROP_SO_ALIASING_FOLLOWS_ORDINARY_BORROW_RULES:
        () = ();

    pub fn alloc_cu_mem_host_alloc_flags0_cached_page_locked(
        ctx: &Arc<CudaContext>,
        len: usize,
    ) -> Result<Self> {
        ctx.bind_to_thread()
            .map_err(|e| anyhow::anyhow!("bind ctx for cached-pinned host staging: {e:?}"))?;
        let ptr = unsafe {
            cudarc::driver::result::malloc_host(len * std::mem::size_of::<T>(), 0)
        }
        .map_err(|e| anyhow::anyhow!("cuMemHostAlloc cached-pinned host staging: {e:?}"))?
            as *mut T;
        Ok(Self { ptr, len })
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub fn as_slice(&self) -> &[T] {
        unsafe { std::slice::from_raw_parts(self.ptr, self.len) }
    }

    pub fn as_mut_slice(&mut self) -> &mut [T] {
        unsafe { std::slice::from_raw_parts_mut(self.ptr, self.len) }
    }
}

#[cfg(feature = "cuda")]
unsafe impl<T: Copy> Send
    for CachedPinnedHostStagingViaCuMemHostAllocFlags0BecauseCudarcAllocPinnedIsWriteCombinedAndHostReadsOfWcAreUncachedAndPageableDtohReallocatesEveryStep<T>
{
}

#[cfg(feature = "cuda")]
unsafe impl<T: Copy> Sync
    for CachedPinnedHostStagingViaCuMemHostAllocFlags0BecauseCudarcAllocPinnedIsWriteCombinedAndHostReadsOfWcAreUncachedAndPageableDtohReallocatesEveryStep<T>
{
}

#[cfg(feature = "cuda")]
impl<T: Copy> Drop
    for CachedPinnedHostStagingViaCuMemHostAllocFlags0BecauseCudarcAllocPinnedIsWriteCombinedAndHostReadsOfWcAreUncachedAndPageableDtohReallocatesEveryStep<T>
{
    fn drop(&mut self) {
        let _ = unsafe { cudarc::driver::result::free_host(self.ptr as _) };
    }
}

#[cfg(feature = "cuda")]
pub struct DeviceArgmaxTokenExit {
    logits_f32: CudaSlice<f32>,
    part_val: CudaSlice<f32>,
    part_idx: CudaSlice<i32>,
    token_out: CudaSlice<u32>,
    rows: usize,
    vocab: usize,
}

#[cfg(feature = "cuda")]
impl DeviceArgmaxTokenExit {
    pub const CASTS_BF16_TO_F32_THEN_ARGMAX_F32_ROWS_THE_PATH_EVERY_PUBLISHED_IDENTITY_NUMBER_USED_ARGMAX_BF16_ALSO_MATCHES_ARGMAX_CHECKED_SKIPPING_NONFINITE_AND_RETURNING_0_ON_ALL_NAN:
        () = ();

    pub fn alloc_scratch_and_sync(
        ctx: &Arc<CudaContext>,
        stream: &Arc<CudaStream>,
        vocab: usize,
    ) -> Result<Self> {
        Self::alloc_scratch_and_sync_rows(ctx, stream, 1, vocab)
    }

    pub fn alloc_scratch_and_sync_rows(
        ctx: &Arc<CudaContext>,
        stream: &Arc<CudaStream>,
        rows: usize,
        vocab: usize,
    ) -> Result<Self> {
        anyhow::ensure!(rows >= 1, "argmax exit needs at least one row");
        ctx.bind_to_thread()
            .map_err(|e| anyhow::anyhow!("bind ctx for argmax exit: {e:?}"))?;
        let parts = nv_kernels::cuda::argmax_parts();
        let logits_f32 = stream
            .alloc_zeros::<f32>(rows * vocab)
            .map_err(|e| anyhow::anyhow!("alloc argmax exit logits_f32: {e:?}"))?;
        let part_val = stream
            .alloc_zeros::<f32>(rows * parts)
            .map_err(|e| anyhow::anyhow!("alloc argmax exit part_val: {e:?}"))?;
        let part_idx = stream
            .alloc_zeros::<i32>(rows * parts)
            .map_err(|e| anyhow::anyhow!("alloc argmax exit part_idx: {e:?}"))?;
        let token_out = stream
            .alloc_zeros::<u32>(rows)
            .map_err(|e| anyhow::anyhow!("alloc argmax exit token_out: {e:?}"))?;
        stream
            .synchronize()
            .map_err(|e| anyhow::anyhow!("sync after argmax exit alloc: {e:?}"))?;
        Ok(Self {
            logits_f32,
            part_val,
            part_idx,
            token_out,
            rows,
            vocab,
        })
    }

    pub fn vocab(&self) -> usize {
        self.vocab
    }

    pub fn rows(&self) -> usize {
        self.rows
    }

    pub fn next_token_argmax_on_device(
        &mut self,
        stream: &Arc<CudaStream>,
        logits_bf16: &CudaSlice<half::bf16>,
    ) -> Result<u32> {
        let toks = self.next_tokens_argmax_rows_on_device(stream, logits_bf16, 1)?;
        Ok(toks[0])
    }

    pub fn next_tokens_argmax_rows_on_device(
        &mut self,
        stream: &Arc<CudaStream>,
        logits_bf16: &CudaSlice<half::bf16>,
        rows_active: usize,
    ) -> Result<Vec<u32>> {
        let () = Self::CASTS_BF16_TO_F32_THEN_ARGMAX_F32_ROWS_THE_PATH_EVERY_PUBLISHED_IDENTITY_NUMBER_USED_ARGMAX_BF16_ALSO_MATCHES_ARGMAX_CHECKED_SKIPPING_NONFINITE_AND_RETURNING_0_ON_ALL_NAN;
        anyhow::ensure!(
            rows_active >= 1 && rows_active <= self.rows,
            "argmax exit rows_active {rows_active} out of 1..={}",
            self.rows
        );
        let n = rows_active * self.vocab;
        anyhow::ensure!(
            logits_bf16.len() >= n,
            "argmax exit logits len {} < rows_active {rows_active} x vocab {}",
            logits_bf16.len(),
            self.vocab
        );
        {
            use cudarc::driver::{DevicePtr, DevicePtrMut};
            let cu = stream.cu_stream() as *mut std::ffi::c_void;
            let (lp, _gl) = logits_bf16.device_ptr(stream);
            let (fp, _gf) = self.logits_f32.device_ptr_mut(stream);
            let rc = unsafe {
                nv_kernels::cuda::cast_bf16_f32(cu, lp as *const u16, fp as *mut f32, n as i32)
            };
            anyhow::ensure!(rc == 0, "serving argmax exit cast_bf16_f32 rc={rc}");
            let (vp, _gv) = self.part_val.device_ptr_mut(stream);
            let (ip, _gi) = self.part_idx.device_ptr_mut(stream);
            let (op, _go) = self.token_out.device_ptr_mut(stream);
            let rc = unsafe {
                nv_kernels::cuda::argmax_f32_rows(
                    cu,
                    fp as *const f32,
                    rows_active as i32,
                    self.vocab as i32,
                    vp as *mut f32,
                    ip as *mut i32,
                    op as *mut u32,
                )
            };
            anyhow::ensure!(rc == 0, "serving argmax exit argmax_f32_rows rc={rc}");
        }
        let host: Vec<u32> = stream
            .clone_dtoh(&self.token_out)
            .map_err(|e| anyhow::anyhow!("dtoh serving argmax token: {e:?}"))?;
        Ok(host[..rows_active].to_vec())
    }
}

pub const PHASE_LADDER_DUMP_EVERY_32_REPLAYED_STEPS_MATCHES_THE_OUTPUT_THE_SOL_LANES_PARSE: u64 =
    32;

pub struct PhaseLadderReplayAccumulatorHostCore {
    tag: &'static str,
    dump_every_replayed_steps: u64,
    acc_ms: std::collections::BTreeMap<&'static str, f64>,
    replayed_steps: u64,
}

impl PhaseLadderReplayAccumulatorHostCore {
    pub fn new(tag: &'static str, dump_every_replayed_steps: u64) -> Self {
        Self {
            tag,
            dump_every_replayed_steps,
            acc_ms: std::collections::BTreeMap::new(),
            replayed_steps: 0,
        }
    }

    pub fn accumulate_replayed_step_returning_dump_due(
        &mut self,
        interval_end_labels_and_deltas_ms: &[(&'static str, f64)],
    ) -> bool {
        for (label, ms) in interval_end_labels_and_deltas_ms {
            *self.acc_ms.entry(label).or_default() += ms;
        }
        self.replayed_steps += 1;
        self.dump_every_replayed_steps > 0
            && self.replayed_steps % self.dump_every_replayed_steps == 0
    }

    pub fn render_sol_lane_lines(&self, n_events: usize) -> Vec<String> {
        let n = self.replayed_steps as f64;
        let total: f64 = self.acc_ms.values().sum();
        let mut rows: Vec<(&'static str, f64)> =
            self.acc_ms.iter().map(|(k, ms)| (*k, ms / n)).collect();
        rows.sort_by(|a, b| b.1.total_cmp(&a.1));
        let mut lines = Vec::with_capacity(rows.len() + 1);
        lines.push(format!(
            "[{}] replayed_steps={} event_sum={:.3} ms/step ({} events)",
            self.tag,
            self.replayed_steps,
            total / n,
            n_events
        ));
        for (k, ms) in rows {
            lines.push(format!("[{}]   {k:<14} {ms:9.4} ms/step", self.tag));
        }
        lines
    }
}

#[cfg(feature = "cuda")]
pub struct InGraphPhaseEventLadder {
    events: Vec<CudaEvent>,
    labels: Vec<&'static str>,
    cursor: usize,
    core: PhaseLadderReplayAccumulatorHostCore,
}

#[cfg(feature = "cuda")]
impl InGraphPhaseEventLadder {
    pub const MARKS_INSIDE_THE_CAPTURED_BODY_SO_EVENT_RECORDS_BECOME_GRAPH_NODES_AND_ELAPSED_IS_READ_AFTER_REPLAY_PLUS_SYNC:
        () = ();

    pub fn new(tag: &'static str) -> Self {
        Self {
            events: Vec::new(),
            labels: Vec::new(),
            cursor: 0,
            core: PhaseLadderReplayAccumulatorHostCore::new(
                tag,
                PHASE_LADDER_DUMP_EVERY_32_REPLAYED_STEPS_MATCHES_THE_OUTPUT_THE_SOL_LANES_PARSE,
            ),
        }
    }

    pub fn begin_step_rewinds_cursor_over_the_captured_events(&mut self) {
        self.cursor = 0;
    }

    pub fn mark(&mut self, stream: &Arc<CudaStream>, label: &'static str) -> Result<()> {
        if self.cursor == self.events.len() {
            let ev = stream
                .context()
                .new_event(Some(cudarc::driver::sys::CUevent_flags::CU_EVENT_DEFAULT))
                .map_err(|e| anyhow::anyhow!("phase event alloc {label}: {e:?}"))?;
            self.events.push(ev);
            self.labels.push(label);
        }
        self.events[self.cursor]
            .record(stream)
            .map_err(|e| anyhow::anyhow!("phase event record {label}: {e:?}"))?;
        self.cursor += 1;
        Ok(())
    }

    pub fn accumulate_replayed_step_and_dump_when_due(&mut self) {
        if self.events.len() < 2 {
            return;
        }
        let mut deltas: Vec<(&'static str, f64)> = Vec::with_capacity(self.events.len() - 1);
        for i in 1..self.events.len() {
            if let Ok(ms) = self.events[i - 1].elapsed_ms(&self.events[i]) {
                deltas.push((self.labels[i], ms as f64));
            }
        }
        if self.core.accumulate_replayed_step_returning_dump_due(&deltas) {
            for line in self.core.render_sol_lane_lines(self.events.len()) {
                eprintln!("{line}");
            }
        }
    }
}

pub use nv_config::env_arm::{
    env_arm_dependency_assert_enabled_arm_requires_prerequisite,
    env_arm_opt_in_eq_some_1_reread_every_call,
    env_arm_opt_in_set_to_anything_but_0_reread_every_call,
    env_arm_opt_out_default_on_unless_eq_some_0_reread_every_call,
    env_arm_present_at_all_reread_every_call,
};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn phase_ladder_core_renders_the_exact_g4_sol_lane_format() {
        let mut core = PhaseLadderReplayAccumulatorHostCore::new("g4-phase", 2);
        assert!(!core
            .accumulate_replayed_step_returning_dump_due(&[("attn", 1.0), ("moe", 3.0)]));
        assert!(core.accumulate_replayed_step_returning_dump_due(&[("attn", 1.0), ("moe", 3.0)]));
        let lines = core.render_sol_lane_lines(3);
        assert_eq!(
            lines[0],
            "[g4-phase] replayed_steps=2 event_sum=4.000 ms/step (3 events)"
        );
        assert_eq!(lines[1], "[g4-phase]   moe               3.0000 ms/step");
        assert_eq!(lines[2], "[g4-phase]   attn              1.0000 ms/step");
    }

    #[test]
    fn phase_ladder_core_sorts_rows_descending_by_per_step_ms() {
        let mut core = PhaseLadderReplayAccumulatorHostCore::new("t", 1);
        assert!(core.accumulate_replayed_step_returning_dump_due(&[
            ("small", 0.5),
            ("big", 9.0),
            ("mid", 2.0)
        ]));
        let lines = core.render_sol_lane_lines(4);
        assert!(lines[1].contains("big"));
        assert!(lines[2].contains("mid"));
        assert!(lines[3].contains("small"));
    }
}
