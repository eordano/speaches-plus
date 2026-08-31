#![cfg_attr(not(feature = "cuda"), allow(dead_code))]

use super::*;

pub(crate) const Q38_SCHED_POLICY_FORMATION_WINDOW_THEN_SOLO_XOR_GROUP: &str =
    "after the first pending request the dispatcher drains the queue up to the lane count; a \
     request still alone after NV_Q38_BATCH_WINDOW_MS (default 30) runs SOLO on the existing \
     MTP/eager path, two or more run as a lane group with MTP bypassed for the whole group; the \
     window is a deadline, never a sleep: a companion already queued or arriving mid-window \
     forms the group the moment it lands, because the first lane prefill outlasts the window \
     anyway";

pub(crate) const Q38_SCHED_POLICY_LATE_JOINERS_WAIT_FOR_FULL_GROUP_DRAIN: &str =
    "a group admits joiners while its formation prefills run, capped at the lane count, and \
     closes at its first batch step; requests arriving after that stay queued until the whole \
     group drains, because a mid-decode join needs its own eager prefill and that invalidates \
     every captured bucket graph under the running group";

pub(crate) const GROUP_LANES_ARE_ALWAYS_PREFILLED_AT_FORMATION_SO_A_PAD_CLOBBERED_LANE_NEVER_REACHES_A_LIVE_REQUEST:
    () = ();

pub(crate) const GROUP_FORMATION_PREFILL_WALL_IS_THE_GDN_PREFILL_SCAN_NOT_THE_LANE_COUNT_SO_STREAM_CONCURRENCY_IS_THE_WRONG_LEVER:
    &str = "at c=8 formation costs one prefill_lane per lane and the wall attributes 85% (20 \
            token prompt) to 98% (320 token prompt) of each call to mixer_linear_attn, whose \
            hot inner loop is run_gdn_scan_candle_stateful -- a token-sequential chain of candle \
            broadcast_mul/sum ops over the [n_v, d_k, d_v] recurrent state. Those are candle ops \
            and candle 0.10 launches every op on CudaDevice::cuda_stream(), never on the \
            nv_layers::cuda_stream override, so forking the prefills onto lane streams moves no \
            work off the device stream and leaves the nv kernels that DO honour the override \
            racing the candle ops that do not. The lever that moves this wall is \
            NV_Q38_GDN_CHUNK_PREFILL=1, which replaces the token-sequential scan with the fused \
            chunk kernel: a 20-token prefill drops 134.3 ms -> 29.9 ms and a 320-token prefill \
            1631.3 ms -> 73.0 ms on unsloth/Qwen3.8-27B-NVFP4";

pub(crate) fn group_formation_prefill_scan_arm() -> &'static str {
    let _ = GROUP_FORMATION_PREFILL_WALL_IS_THE_GDN_PREFILL_SCAN_NOT_THE_LANE_COUNT_SO_STREAM_CONCURRENCY_IS_THE_WRONG_LEVER;
    if nv_layers::linear_attn::chunk_prefill_env_read_per_call_so_one_process_can_ab_both_scan_paths()
    {
        "gdn_chunk"
    } else {
        "gdn_candle_token_sequential"
    }
}

pub(crate) fn q38_batch_window_ms_env_nv_q38_batch_window_ms() -> u64 {
    std::env::var("NV_Q38_BATCH_WINDOW_MS")
        .ok()
        .and_then(|v| v.trim().parse().ok())
        .unwrap_or(30)
}

pub(crate) fn q36_batch_window_ms_env_nv_q36_batch_window_ms() -> u64 {
    std::env::var("NV_Q36_BATCH_WINDOW_MS")
        .ok()
        .and_then(|v| v.trim().parse().ok())
        .unwrap_or(30)
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum LaneSeat {
    Vacant,
    ActiveAwaitingPrefill,
    ActivePrefilled,
    VacantPadClobbered,
}

pub(crate) struct GroupRoster {
    seats: Vec<LaneSeat>,
}

impl GroupRoster {
    pub(crate) fn new(lanes: usize) -> Self {
        Self {
            seats: vec![LaneSeat::Vacant; lanes],
        }
    }

    pub(crate) fn seat(&self, lane: usize) -> LaneSeat {
        self.seats[lane]
    }

    pub(crate) fn join(&mut self) -> Option<usize> {
        let lane = self
            .seats
            .iter()
            .position(|s| matches!(s, LaneSeat::Vacant | LaneSeat::VacantPadClobbered))?;
        self.seats[lane] = LaneSeat::ActiveAwaitingPrefill;
        Some(lane)
    }

    pub(crate) fn mark_prefilled(&mut self, lane: usize) -> anyhow::Result<()> {
        anyhow::ensure!(
            self.seats[lane] == LaneSeat::ActiveAwaitingPrefill,
            "roster: mark_prefilled on lane {lane} in state {:?}; only a freshly joined lane \
             takes a prefill",
            self.seats[lane]
        );
        self.seats[lane] = LaneSeat::ActivePrefilled;
        Ok(())
    }

    pub(crate) fn leave(&mut self, lane: usize) {
        self.seats[lane] = LaneSeat::Vacant;
    }

    pub(crate) fn step_plan(
        &mut self,
        feed: &[Option<u32>],
    ) -> anyhow::Result<Option<Vec<Option<u32>>>> {
        anyhow::ensure!(
            feed.len() == self.seats.len(),
            "roster: feed len {} != lane count {}",
            feed.len(),
            self.seats.len()
        );
        for (lane, (seat, tok)) in self.seats.iter().zip(feed.iter()).enumerate() {
            match seat {
                LaneSeat::ActivePrefilled => anyhow::ensure!(
                    tok.is_some(),
                    "roster: active lane {lane} got no token in the step feed"
                ),
                LaneSeat::ActiveAwaitingPrefill => anyhow::bail!(
                    "roster: lane {lane} would decode without a prefill; a joined lane must be \
                     prefilled before its first batch step (pad-clobber invariant)"
                ),
                LaneSeat::Vacant | LaneSeat::VacantPadClobbered => anyhow::ensure!(
                    tok.is_none(),
                    "roster: vacant lane {lane} got a token in the step feed"
                ),
            }
        }
        let Some(last) = feed.iter().rposition(|t| t.is_some()) else {
            return Ok(None);
        };
        let need = last + 1;
        for seat in self.seats[..need].iter_mut() {
            if *seat == LaneSeat::Vacant {
                *seat = LaneSeat::VacantPadClobbered;
            }
        }
        Ok(Some(feed[..need].to_vec()))
    }
}

#[cfg(feature = "cuda")]
pub(crate) use gpu::*;

#[cfg(feature = "cuda")]
mod gpu {
    use super::*;
    use nv_models::qwen3_5_moe::qwen36_batch::Qwen36BatchLanes;
    use nv_models::qwen3_5_moe::qwen38_batch::form_pack_env_opt_in_nv_q38_form_pack_1_packed_formation_prefill_rides_the_m_row_gemm_routes_so_it_requires_nv_q38_batch_gemm_1 as form_pack_env;
    use nv_models::qwen3_5_moe::qwen38_batch::Qwen38BatchLanes;
    use std::time::Instant;

    pub(crate) fn step_overlap_env_opt_in_nv_q38_step_overlap_1_the_next_step_launches_before_detok_and_sse_so_host_consume_rides_the_gpu_step(
    ) -> bool {
        std::env::var("NV_Q38_STEP_OVERLAP").ok().as_deref() == Some("1")
    }

    pub(crate) const THE_EAGER_SOLO_BODY_STEPS_UNCAPTURED_AND_SAMPLES_BY_A_FULL_VOCAB_LOGITS_DTOH_WHILE_A_ONE_LANE_GROUP_REPLAYS_THE_BUCKET_1_GRAPH_AND_EXITS_ON_THE_DEVICE_ARGMAX:
        &str = "run_qwen38_dense_solo_body calls forward_with_cache_dispatched once per token \
                with no graph capture and then copies a whole vocab row to the host to sample it; \
                a group of one replays the captured bucket-1 graph and returns a single u32. \
                NV_Q38_SOLO_VIA_LANES=1 sends a lone request down the group route so the c=1 rung \
                measures the same engine every other rung measures. The arm is refused whenever an \
                MTP head is loaded, because a group bypasses MTP for its whole membership and a \
                silent spec bypass is a worse trade than an uncaptured step";

    pub(crate) fn solo_via_lanes_env_opt_in_nv_q38_solo_via_lanes_1_a_lone_request_runs_as_a_one_lane_group(
    ) -> bool {
        std::env::var("NV_Q38_SOLO_VIA_LANES").ok().as_deref() == Some("1")
    }

    pub(crate) const Q38_SCHED_PACKED_FORMATION_PREFILLS_EVERY_ADMITTED_LANE_IN_ONE_WIDE_CALL: &str =
        "with NV_Q38_FORM_PACK=1 (which requires NV_Q38_BATCH_GEMM=1) an all-greedy group defers \
         its per-lane prefills until admission closes and runs them as one packed forward whose \
         ffn, norms and head ride the packed rows while each mixer segment runs the solo prefill \
         route on its own lane cache; chat prefill is per-call weight-traffic bound (29.9 ms at \
         20 tokens, 73 ms at 320), so n serial prefills pay the weights n times and one wide \
         call pays them once. Late joiners arriving after the packed call still get the serial \
         per-lane prefill, and a group with any non-greedy or logprobs member falls back to the \
         serial formation unchanged";

    pub(crate) struct SchedJob {
        pub(crate) req: ChatGenerateRequest,
        pub(crate) tx: mpsc::Sender<ChatEvent>,
    }

    pub(crate) const SCHED_SENDS_TIME_OUT_AND_EVICT_BECAUSE_ONE_UNREAD_CLIENT_CHANNEL_PARKED_EVERY_LANE_ON_THE_SINGLE_THREAD_DISPATCHER_COST_THE_STALLED_CLIENT_LOSES_ITS_STREAM_AFTER_NV_SSE_SEND_TIMEOUT_MS_AND_0_RESTORES_THE_OLD_INFINITE_WAIT:
        () = ();

    fn sched_send_timeout() -> Option<std::time::Duration> {
        let () = SCHED_SENDS_TIME_OUT_AND_EVICT_BECAUSE_ONE_UNREAD_CLIENT_CHANNEL_PARKED_EVERY_LANE_ON_THE_SINGLE_THREAD_DISPATCHER_COST_THE_STALLED_CLIENT_LOSES_ITS_STREAM_AFTER_NV_SSE_SEND_TIMEOUT_MS_AND_0_RESTORES_THE_OLD_INFINITE_WAIT;
        sse_send_timeout_opt_for_the_batch_sched_zero_disables_unlike_the_solo_paths_1ms_clamp()
    }

    async fn sched_push(
        tx: &mpsc::Sender<ChatEvent>,
        ev: ChatEvent,
        timeout: Option<std::time::Duration>,
    ) -> SsePush {
        let out = push_event_async_opt(tx, ev, timeout).await;
        log_sse_abort(out, "qwen-batch-sched");
        out
    }

    pub(crate) fn group_device_argmax_default_on_kill_nv_q38_batch_device_argmax_0_byte_exact_per_the_widening_and_tie_break_constants_in_qwen38_batch(
    ) -> bool {
        std::env::var("NV_Q38_BATCH_DEVICE_ARGMAX").ok().as_deref() != Some("0")
    }

    pub(crate) fn q36_group_device_argmax_default_on_kill_nv_q36_batch_device_argmax_0_byte_exact_per_the_same_widening_and_tie_break_constants(
    ) -> bool {
        std::env::var("NV_Q36_BATCH_DEVICE_ARGMAX").ok().as_deref() != Some("0")
    }

    pub(crate) const LANE_PREFILL_MATCHES_EACH_FAMILYS_SOLO_SERVING_PREFILL_ROUTE_DENSE_HAS_NO_MOE_TO_ROUTE_AND_MOE_GOES_THROUGH_THE_INSTALLED_HOOK_LIKE_THE_EAGER_SOLO_BODY_SO_A_GROUP_LANE_AND_ITS_SOLO_REFERENCE_SHARE_PREFILL_BYTES:
        () = ();

    pub(crate) enum QwenLanePool {
        Dense38(Qwen38BatchLanes),
        Moe36(Qwen36BatchLanes),
    }

    impl QwenLanePool {
        pub(crate) fn lanes(&self) -> usize {
            match self {
                Self::Dense38(l) => l.lanes(),
                Self::Moe36(l) => l.lanes(),
            }
        }

        fn lane_pos(&self, lane: usize) -> usize {
            match self {
                Self::Dense38(l) => l.lane_pos(lane),
                Self::Moe36(l) => l.lane_pos(lane),
            }
        }

        fn lane_max_seq_len(&self) -> usize {
            match self {
                Self::Dense38(l) => l.lane_max_seq_len(),
                Self::Moe36(l) => l.lane_max_seq_len(),
            }
        }

        fn captures(&self) -> u64 {
            match self {
                Self::Dense38(l) => l.captures(),
                Self::Moe36(l) => l.captures(),
            }
        }

        fn replays(&self) -> u64 {
            match self {
                Self::Dense38(l) => l.replays(),
                Self::Moe36(l) => l.replays(),
            }
        }

        fn family_tag(&self) -> &'static str {
            match self {
                Self::Dense38(_) => "qwen3.8",
                Self::Moe36(_) => "qwen3.6",
            }
        }

        fn log_prefix(&self) -> &'static str {
            match self {
                Self::Dense38(_) => "[q38-batch-sched]",
                Self::Moe36(_) => "[q36-batch-sched]",
            }
        }

        fn attrib_prefix(&self) -> &'static str {
            match self {
                Self::Dense38(_) => "[q38-sched-attrib]",
                Self::Moe36(_) => "[q36-sched-attrib]",
            }
        }

        pub(crate) fn window_ms(&self) -> u64 {
            match self {
                Self::Dense38(_) => q38_batch_window_ms_env_nv_q38_batch_window_ms(),
                Self::Moe36(_) => q36_batch_window_ms_env_nv_q36_batch_window_ms(),
            }
        }

        fn device_argmax_default_on(&self) -> bool {
            match self {
                Self::Dense38(_) => {
                    group_device_argmax_default_on_kill_nv_q38_batch_device_argmax_0_byte_exact_per_the_widening_and_tie_break_constants_in_qwen38_batch()
                }
                Self::Moe36(_) => {
                    q36_group_device_argmax_default_on_kill_nv_q36_batch_device_argmax_0_byte_exact_per_the_same_widening_and_tie_break_constants()
                }
            }
        }

        fn quiesce_graphs_before_external_eager_work_because_solo_requests_share_the_capture_stream(
            &mut self,
        ) -> anyhow::Result<()> {
            match self {
                Self::Dense38(l) => l
                    .quiesce_graphs_before_external_eager_work_because_solo_requests_share_the_capture_stream(),
                Self::Moe36(l) => l
                    .quiesce_graphs_before_external_eager_work_because_solo_requests_share_the_capture_stream(),
            }
        }

        fn prefill_lane(&mut self, lane: usize, prompt: &[u32]) -> anyhow::Result<Vec<f32>> {
            let () = LANE_PREFILL_MATCHES_EACH_FAMILYS_SOLO_SERVING_PREFILL_ROUTE_DENSE_HAS_NO_MOE_TO_ROUTE_AND_MOE_GOES_THROUGH_THE_INSTALLED_HOOK_LIKE_THE_EAGER_SOLO_BODY_SO_A_GROUP_LANE_AND_ITS_SOLO_REFERENCE_SHARE_PREFILL_BYTES;
            match self {
                Self::Dense38(l) => l.prefill_lane(lane, prompt),
                Self::Moe36(l) => l
                    .prefill_lane_hooked_matching_the_eager_serving_prefill_route_so_a_group_lane_and_its_solo_reference_share_prefill_bytes(
                        lane, prompt,
                    ),
            }
        }

        fn prefill_lane_greedy_token_on_device(
            &mut self,
            lane: usize,
            prompt: &[u32],
        ) -> anyhow::Result<u32> {
            let () = LANE_PREFILL_MATCHES_EACH_FAMILYS_SOLO_SERVING_PREFILL_ROUTE_DENSE_HAS_NO_MOE_TO_ROUTE_AND_MOE_GOES_THROUGH_THE_INSTALLED_HOOK_LIKE_THE_EAGER_SOLO_BODY_SO_A_GROUP_LANE_AND_ITS_SOLO_REFERENCE_SHARE_PREFILL_BYTES;
            match self {
                Self::Dense38(l) => l
                    .prefill_lane_greedy_token_on_device_byte_exact_vs_host_sampling_by_the_widening_and_tie_break_constants(
                        lane, prompt,
                    ),
                Self::Moe36(l) => l
                    .prefill_lane_hooked_greedy_token_on_device_byte_exact_vs_host_sampling_by_the_widening_and_tie_break_constants(
                        lane, prompt,
                    ),
            }
        }

        fn step_batch(&mut self, plan: &[Option<u32>]) -> anyhow::Result<Vec<Option<Vec<f32>>>> {
            match self {
                Self::Dense38(l) => l.step_batch(plan),
                Self::Moe36(l) => l.step_batch(plan),
            }
        }

        fn step_batch_greedy_tokens_on_device(
            &mut self,
            plan: &[Option<u32>],
        ) -> anyhow::Result<Vec<Option<u32>>> {
            match self {
                Self::Dense38(l) => l
                    .step_batch_greedy_tokens_on_device_byte_exact_vs_host_sampling_by_the_widening_and_tie_break_constants(
                        plan,
                    ),
                Self::Moe36(l) => l
                    .step_batch_greedy_tokens_on_device_byte_exact_vs_host_sampling_by_the_widening_and_tie_break_constants(
                        plan,
                    ),
            }
        }

        fn prefill_lanes_packed_greedy_tokens_on_device(
            &mut self,
            joins: &[(usize, &[u32])],
        ) -> anyhow::Result<Vec<u32>> {
            match self {
                Self::Dense38(l) => l.prefill_lanes_packed_greedy_tokens_on_device(joins),
                Self::Moe36(_) => anyhow::bail!(
                    "packed formation prefill is a dense-lane arm; the MoE pool prefills per lane"
                ),
            }
        }

        fn step_batch_greedy_tokens_launch(&mut self, plan: &[Option<u32>]) -> anyhow::Result<()> {
            match self {
                Self::Dense38(l) => l
                    .step_batch_greedy_tokens_launch_without_final_sync_so_the_callers_detok_and_sse_of_the_prior_step_overlap_this_gpu_step(
                        plan,
                    ),
                Self::Moe36(_) => anyhow::bail!(
                    "step overlap is a dense-lane arm; the MoE pool steps synchronously"
                ),
            }
        }

        fn step_batch_greedy_tokens_collect(
            &mut self,
            plan: &[Option<u32>],
        ) -> anyhow::Result<Vec<Option<u32>>> {
            match self {
                Self::Dense38(l) => l.step_batch_greedy_tokens_collect_after_launch(plan),
                Self::Moe36(_) => anyhow::bail!(
                    "step overlap is a dense-lane arm; the MoE pool steps synchronously"
                ),
            }
        }
    }

    enum GroupStepOut {
        HostSampledRows(Vec<Option<Vec<f32>>>),
        DeviceArgmaxTokens(Vec<Option<u32>>),
    }

    pub(crate) struct QwenBatchScheduler {
        jobs: std::sync::Mutex<Option<tokio::sync::mpsc::UnboundedSender<SchedJob>>>,
        worker: std::sync::Mutex<Option<std::thread::JoinHandle<()>>>,
        lanes_n: usize,
    }

    impl QwenBatchScheduler {
        #[allow(clippy::too_many_arguments)]
        pub(crate) fn spawn(
            lanes: QwenLanePool,
            mtp: Option<QwenMtpHead>,
            tokenizer: Arc<tokenizers::Tokenizer>,
            device: candle_core::Device,
            eos_ids: Vec<u32>,
            kv_max_seq_len: usize,
            default_max_new: usize,
        ) -> anyhow::Result<Arc<Self>> {
            match (&lanes, &mtp) {
                (QwenLanePool::Dense38(_), Some(QwenMtpHead::Moe(_))) => anyhow::bail!(
                    "qwen batch scheduler: a dense lane pool was handed the MoE MTP head; the \
                     solo spec route would deref the wrong session family"
                ),
                (QwenLanePool::Moe36(_), Some(QwenMtpHead::Dense(_))) => anyhow::bail!(
                    "qwen batch scheduler: a MoE lane pool was handed the dense MTP head; the \
                     solo spec route would deref the wrong session family"
                ),
                _ => {}
            }
            let (jobs_tx, jobs_rx) = tokio::sync::mpsc::unbounded_channel();
            let lanes_n = lanes.lanes();
            let worker = std::thread::Builder::new()
                .name("qwen-batch-sched".into())
                .spawn(move || {
                    let rt = tokio::runtime::Builder::new_current_thread()
                        .enable_all()
                        .build()
                        .expect("qwen batch scheduler: current-thread runtime");
                    rt.block_on(dispatch_loop(
                        jobs_rx,
                        lanes,
                        mtp,
                        tokenizer,
                        device,
                        eos_ids,
                        kv_max_seq_len,
                        default_max_new,
                    ));
                })
                .map_err(|e| anyhow::anyhow!("spawn qwen batch scheduler thread: {e}"))?;
            Ok(Arc::new(Self {
                jobs: std::sync::Mutex::new(Some(jobs_tx)),
                worker: std::sync::Mutex::new(Some(worker)),
                lanes_n,
            }))
        }

        pub(crate) fn lanes(&self) -> usize {
            self.lanes_n
        }

        pub(crate) fn submit(
            &self,
            req: ChatGenerateRequest,
            tx: mpsc::Sender<ChatEvent>,
        ) -> anyhow::Result<()> {
            let guard = self.jobs.lock().unwrap_or_else(|e| e.into_inner());
            let sender = guard
                .as_ref()
                .ok_or_else(|| anyhow::anyhow!("qwen batch scheduler is shut down"))?;
            sender
                .send(SchedJob { req, tx })
                .map_err(|_| anyhow::anyhow!("qwen batch scheduler dispatcher exited"))
        }
    }

    impl Drop for QwenBatchScheduler {
        fn drop(&mut self) {
            self.jobs
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .take();
            if let Some(h) = self
                .worker
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .take()
            {
                let _ = h.join();
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    async fn dispatch_loop(
        mut rx: tokio::sync::mpsc::UnboundedReceiver<SchedJob>,
        mut lanes: QwenLanePool,
        mtp: Option<QwenMtpHead>,
        tokenizer: Arc<tokenizers::Tokenizer>,
        device: candle_core::Device,
        eos_ids: Vec<u32>,
        kv_max_seq_len: usize,
        default_max_new: usize,
    ) {
        eprintln!(
            "{} up: {} lanes, window {} ms, mtp_solo={}. Policy A: {}. Policy B: {}",
            lanes.log_prefix(),
            lanes.lanes(),
            lanes.window_ms(),
            mtp.is_some(),
            Q38_SCHED_POLICY_FORMATION_WINDOW_THEN_SOLO_XOR_GROUP,
            Q38_SCHED_POLICY_LATE_JOINERS_WAIT_FOR_FULL_GROUP_DRAIN,
        );
        while let Some(first) = rx.recv().await {
            let t_first = Instant::now();
            let window_ms = lanes.window_ms();
            let admission_deadline = t_first + std::time::Duration::from_millis(window_ms);
            let mut jobs = vec![first];
            while jobs.len() < lanes.lanes() {
                match rx.try_recv() {
                    Ok(j) => jobs.push(j),
                    Err(_) => break,
                }
            }
            if jobs.len() == 1 && window_ms > 0 {
                let the_window_is_a_deadline_not_a_sleep_so_a_companion_arriving_at_1ms_forms_the_group_at_1ms_not_at_window_end =
                    tokio::time::Instant::from_std(admission_deadline);
                while jobs.len() < lanes.lanes() {
                    match tokio::time::timeout_at(
                        the_window_is_a_deadline_not_a_sleep_so_a_companion_arriving_at_1ms_forms_the_group_at_1ms_not_at_window_end,
                        rx.recv(),
                    )
                    .await
                    {
                        Ok(Some(j)) => {
                            jobs.push(j);
                            while jobs.len() < lanes.lanes() {
                                match rx.try_recv() {
                                    Ok(j) => jobs.push(j),
                                    Err(_) => break,
                                }
                            }
                            if jobs.len() > 1 {
                                break;
                            }
                        }
                        Ok(None) | Err(_) => break,
                    }
                }
            }
            let a_lone_request_keeps_the_eager_solo_route_unless_the_lanes_arm_is_on_and_no_mtp_head_would_be_bypassed =
                mtp.is_some()
                    || !solo_via_lanes_env_opt_in_nv_q38_solo_via_lanes_1_a_lone_request_runs_as_a_one_lane_group();
            if jobs.len() == 1
                && a_lone_request_keeps_the_eager_solo_route_unless_the_lanes_arm_is_on_and_no_mtp_head_would_be_bypassed
            {
                let _ = THE_EAGER_SOLO_BODY_STEPS_UNCAPTURED_AND_SAMPLES_BY_A_FULL_VOCAB_LOGITS_DTOH_WHILE_A_ONE_LANE_GROUP_REPLAYS_THE_BUCKET_1_GRAPH_AND_EXITS_ON_THE_DEVICE_ARGMAX;
                let job = jobs.pop().unwrap();
                run_solo(
                    &mut lanes,
                    mtp.as_ref(),
                    &tokenizer,
                    &device,
                    &eos_ids,
                    kv_max_seq_len,
                    default_max_new,
                    job,
                )
                .await;
            } else {
                run_group(
                    &mut lanes,
                    &mut rx,
                    &tokenizer,
                    &eos_ids,
                    default_max_new,
                    t_first,
                    admission_deadline,
                    jobs,
                )
                .await;
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    async fn run_solo(
        lanes: &mut QwenLanePool,
        mtp: Option<&QwenMtpHead>,
        tokenizer: &Arc<tokenizers::Tokenizer>,
        device: &candle_core::Device,
        eos_ids: &[u32],
        kv_max_seq_len: usize,
        default_max_new: usize,
        job: SchedJob,
    ) {
        let SchedJob { req, tx } = job;
        let send_timeout = sched_send_timeout();
        if let Err(err) = lanes
            .quiesce_graphs_before_external_eager_work_because_solo_requests_share_the_capture_stream()
        {
            let _ = sched_push(&tx, ChatEvent::Error(format!("{err:#}")), send_timeout).await;
            return;
        }
        let res = match lanes {
            QwenLanePool::Dense38(l) => {
                let dense_mtp = match mtp {
                    Some(QwenMtpHead::Dense(m)) => Some(m.as_ref()),
                    _ => None,
                };
                run_qwen38_solo_via_scheduler(
                    l.model(),
                    dense_mtp,
                    tokenizer,
                    device,
                    req,
                    kv_max_seq_len,
                    default_max_new,
                    eos_ids,
                    &tx,
                )
                .await
            }
            QwenLanePool::Moe36(l) => {
                let moe_mtp = match mtp {
                    Some(QwenMtpHead::Moe(m)) => Some(m.as_ref()),
                    _ => None,
                };
                run_qwen36_solo_via_scheduler(
                    l.model(),
                    moe_mtp,
                    Some(l.moe_dispatch_ref()),
                    tokenizer,
                    device,
                    req,
                    kv_max_seq_len,
                    default_max_new,
                    eos_ids,
                    &tx,
                )
                .await
            }
        };
        if let Err(err) = res {
            let _ = sched_push(&tx, ChatEvent::Error(format!("{err:#}")), send_timeout).await;
        }
    }

    struct GroupMember {
        lane: usize,
        tx: mpsc::Sender<ChatEvent>,
        detok: IncrementalDetok,
        emitter: StreamEmitter,
        sampler: ChatSampler,
        last: SampleOutput,
        generated_ids: Vec<u32>,
        max_new: usize,
        completion_tokens: u32,
        finish_reason: String,
        logprobs: bool,
        alive: bool,
        abandoned: bool,
        emitted_tokens: usize,
        pending_finish: Option<&'static str>,
        stop_free_so_deferred_emission_cannot_miss_a_stop_hit: bool,
        sse_timeout_none_is_the_old_infinite_wait: Option<std::time::Duration>,
    }

    impl GroupMember {
        fn abandon(&mut self) {
            self.alive = false;
            self.abandoned = true;
        }

        async fn push_evicting_this_member_on_timeout_or_close(&mut self, ev: ChatEvent) -> bool {
            match push_event_async_opt(&self.tx, ev, self.sse_timeout_none_is_the_old_infinite_wait)
                .await
            {
                SsePush::Sent => true,
                out => {
                    log_sse_abort(out, "qwen-batch-sched");
                    self.abandon();
                    false
                }
            }
        }

        fn advance_on_device_argmax_token_deferring_detok_and_sse_to_emit_pending(
            &mut self,
            tok: u32,
            eos_ids: &[u32],
            lane_pos: usize,
            lane_max_seq_len: usize,
        ) {
            self.sampler.record_token(tok);
            self.last = SampleOutput {
                token: tok,
                logprob: None,
                top: Vec::new(),
                exhausted: false,
            };
            if eos_ids.contains(&tok) {
                self.pending_finish = Some("stop");
                self.alive = false;
                return;
            }
            self.generated_ids.push(tok);
            self.completion_tokens = self.generated_ids.len() as u32;
            if self.generated_ids.len() >= self.max_new || lane_pos >= lane_max_seq_len {
                self.pending_finish = Some("length");
                self.alive = false;
            }
        }

        async fn emit_pending_deltas_then_any_deferred_finish(
            &mut self,
            tokenizer: &tokenizers::Tokenizer,
        ) {
            if self.abandoned {
                return;
            }
            while self.emitted_tokens < self.generated_ids.len() {
                let tok = self.generated_ids[self.emitted_tokens];
                self.emitted_tokens += 1;
                let piece = match self.detok.push(tok) {
                    Ok(new_text) => {
                        let (piece, stop_hit) = self.emitter.step(new_text);
                        assert!(
                            !stop_hit,
                            "deferred emission saw a stop hit on a member the overlap gate \
                             admitted as stop-free"
                        );
                        piece
                    }
                    Err(e) => {
                        let msg = format!("{e:#}");
                        let _ = self
                            .push_evicting_this_member_on_timeout_or_close(ChatEvent::Error(msg))
                            .await;
                        self.abandon();
                        return;
                    }
                };
                if !piece.is_empty()
                    && !self
                        .push_evicting_this_member_on_timeout_or_close(ChatEvent::TextDelta(piece))
                        .await
                {
                    return;
                }
            }
            if let Some(reason) = self.pending_finish.take() {
                self.finish_and_send_done(tokenizer, reason).await;
            }
        }

        async fn finish_and_send_done(&mut self, tokenizer: &tokenizers::Tokenizer, reason: &str) {
            self.finish_reason = reason.to_string();
            self.alive = false;
            if self.abandoned {
                return;
            }
            if !self.generated_ids.is_empty() {
                if let Ok(full) = decode_keeping_wire(tokenizer, &self.generated_ids) {
                    let tail = self.emitter.finish(&full);
                    if !tail.is_empty()
                        && !self
                            .push_evicting_this_member_on_timeout_or_close(ChatEvent::TextDelta(
                                tail,
                            ))
                            .await
                    {
                        return;
                    }
                }
            }
            let _ = self
                .push_evicting_this_member_on_timeout_or_close(ChatEvent::Done {
                    finish_reason: self.finish_reason.clone(),
                    completion_tokens: self.completion_tokens,
                })
                .await;
        }

        async fn consume_sampled_token_and_stream_it_so_ttft_is_paid_at_the_lanes_own_prefill(
            &mut self,
            tokenizer: &tokenizers::Tokenizer,
            eos_ids: &[u32],
            lane_pos: usize,
            lane_max_seq_len: usize,
        ) {
            let tok = self.last.token;
            if eos_ids.contains(&tok) {
                self.finish_and_send_done(tokenizer, "stop").await;
                return;
            }
            self.generated_ids.push(tok);
            self.completion_tokens = self.generated_ids.len() as u32;
            let step_or_err: Result<(String, bool), String> = match self.detok.push(tok) {
                Ok(new_text) => Ok(self.emitter.step(new_text)),
                Err(e) => Err(format!("{e:#}")),
            };
            let (piece, stop_hit) = match step_or_err {
                Ok(s) => s,
                Err(msg) => {
                    let _ = self
                        .push_evicting_this_member_on_timeout_or_close(ChatEvent::Error(msg))
                        .await;
                    self.abandon();
                    return;
                }
            };
            if !piece.is_empty()
                && !self
                    .push_evicting_this_member_on_timeout_or_close(ChatEvent::TextDelta(piece))
                    .await
            {
                return;
            }
            if self.logprobs {
                let entry = build_logprob_entry(tokenizer, &self.last);
                if !self
                    .push_evicting_this_member_on_timeout_or_close(ChatEvent::Logprob(entry))
                    .await
                {
                    return;
                }
            }
            if stop_hit {
                self.finish_and_send_done(tokenizer, "stop").await;
                return;
            }
            if self.generated_ids.len() >= self.max_new || lane_pos >= lane_max_seq_len {
                self.finish_and_send_done(tokenizer, "length").await;
            }
        }
    }

    struct AdmittedLane {
        lane: usize,
        tx: mpsc::Sender<ChatEvent>,
        req: ChatGenerateRequest,
        prompt_ids: Vec<u32>,
        sampler: ChatSampler,
        max_new: usize,
    }

    async fn admit_job_roster_tokenize_started_window_sampler_without_prefilling(
        lanes: &mut QwenLanePool,
        tokenizer: &Arc<tokenizers::Tokenizer>,
        eos_ids: &[u32],
        default_max_new: usize,
        roster: &mut GroupRoster,
        job: SchedJob,
    ) -> Option<AdmittedLane> {
        let SchedJob { req, tx } = job;
        let send_timeout = sched_send_timeout();
        let Some(lane) = roster.join() else {
            let _ = sched_push(
                &tx,
                ChatEvent::Error(
                    "qwen batch scheduler formed a group larger than its lane pool".into(),
                ),
                send_timeout,
            )
            .await;
            return None;
        };
        let prompt_ids = match tokenizer.encode(req.prompt.as_str(), false) {
            Ok(e) => e.get_ids().to_vec(),
            Err(e) => {
                roster.leave(lane);
                let _ =
                    sched_push(&tx, ChatEvent::Error(format!("tokenize: {e}")), send_timeout).await;
                return None;
            }
        };
        if sched_push(
            &tx,
            ChatEvent::Started {
                prompt_tokens: prompt_ids.len() as u32,
            },
            send_timeout,
        )
        .await
            != SsePush::Sent
        {
            roster.leave(lane);
            return None;
        }
        if prompt_ids.is_empty() {
            roster.leave(lane);
            let _ = sched_push(
                &tx,
                ChatEvent::Done {
                    finish_reason: "length".into(),
                    completion_tokens: 0,
                },
                send_timeout,
            )
            .await;
            return None;
        }
        let requested = effective_max_new(req.max_new_tokens, default_max_new);
        let Some((_cache_len, max_new)) =
            kv_window(prompt_ids.len(), requested, lanes.lane_max_seq_len())
        else {
            roster.leave(lane);
            let _ = sched_push(
                &tx,
                ChatEvent::Error(format!(
                    "prompt of {} tokens does not fit the {}-token KV window",
                    prompt_ids.len(),
                    lanes.lane_max_seq_len()
                )),
                send_timeout,
            )
            .await;
            return None;
        };
        if max_new == 0 {
            roster.leave(lane);
            let _ = sched_push(
                &tx,
                ChatEvent::Done {
                    finish_reason: "length".into(),
                    completion_tokens: 0,
                },
                send_timeout,
            )
            .await;
            return None;
        }
        let mut sampler = match ChatSampler::for_request(&req, tokenizer, eos_ids, max_new) {
            Ok(s) => s,
            Err(e) => {
                roster.leave(lane);
                let _ =
                    sched_push(&tx, ChatEvent::Error(format!("{e:#}")), send_timeout).await;
                return None;
            }
        };
        sampler.seed_prompt(&prompt_ids);
        Some(AdmittedLane {
            lane,
            tx,
            req,
            prompt_ids,
            sampler,
            max_new,
        })
    }

    async fn attach_member_and_stream_the_first_token(
        lanes: &mut QwenLanePool,
        tokenizer: &Arc<tokenizers::Tokenizer>,
        eos_ids: &[u32],
        roster: &mut GroupRoster,
        members: &mut Vec<GroupMember>,
        adm: AdmittedLane,
        last: SampleOutput,
    ) {
        let AdmittedLane {
            lane,
            tx,
            req,
            prompt_ids: _,
            sampler,
            max_new,
        } = adm;
        let send_timeout = sched_send_timeout();
        if roster.mark_prefilled(lane).is_err() {
            roster.leave(lane);
            let _ = sched_push(
                &tx,
                ChatEvent::Error("qwen batch roster desync at prefill".into()),
                send_timeout,
            )
            .await;
            return;
        }
        let logprobs = req.logprobs;
        let mut m = GroupMember {
            lane,
            tx,
            detok: IncrementalDetok::new(tokenizer.clone()),
            emitter: StreamEmitter::new(&req.stop),
            sampler,
            last,
            generated_ids: Vec::with_capacity(max_new),
            max_new,
            completion_tokens: 0,
            finish_reason: "length".to_string(),
            logprobs,
            alive: true,
            abandoned: false,
            emitted_tokens: 0,
            pending_finish: None,
            stop_free_so_deferred_emission_cannot_miss_a_stop_hit: req.stop.is_empty(),
            sse_timeout_none_is_the_old_infinite_wait: send_timeout,
        };
        m.consume_sampled_token_and_stream_it_so_ttft_is_paid_at_the_lanes_own_prefill(
            tokenizer,
            eos_ids,
            lanes.lane_pos(lane),
            lanes.lane_max_seq_len(),
        )
        .await;
        if !m.alive {
            roster.leave(lane);
        }
        members.push(m);
    }

    #[allow(clippy::too_many_arguments)]
    async fn prefill_admitted_lane_and_stream_the_first_token(
        lanes: &mut QwenLanePool,
        tokenizer: &Arc<tokenizers::Tokenizer>,
        eos_ids: &[u32],
        roster: &mut GroupRoster,
        members: &mut Vec<GroupMember>,
        prefill_ms: &mut Vec<f64>,
        mut adm: AdmittedLane,
    ) {
        let lane = adm.lane;
        let send_timeout = sched_send_timeout();
        let prefill_by_device_argmax =
            adm.sampler.fast_greedy() && lanes.device_argmax_default_on();
        let t_prefill = Instant::now();
        let last = if prefill_by_device_argmax {
            match lanes.prefill_lane_greedy_token_on_device(lane, &adm.prompt_ids) {
                Ok(tok) => {
                    prefill_ms.push(t_prefill.elapsed().as_secs_f64() * 1e3);
                    adm.sampler.record_token(tok);
                    SampleOutput {
                        token: tok,
                        logprob: None,
                        top: Vec::new(),
                        exhausted: false,
                    }
                }
                Err(e) => {
                    roster.leave(lane);
                    let _ = sched_push(
                        &adm.tx,
                        ChatEvent::Error(format!("batch prefill: {e:#}")),
                        send_timeout,
                    )
                    .await;
                    return;
                }
            }
        } else {
            let row = match lanes.prefill_lane(lane, &adm.prompt_ids) {
                Ok(r) => r,
                Err(e) => {
                    roster.leave(lane);
                    let _ = sched_push(
                        &adm.tx,
                        ChatEvent::Error(format!("batch prefill: {e:#}")),
                        send_timeout,
                    )
                    .await;
                    return;
                }
            };
            prefill_ms.push(t_prefill.elapsed().as_secs_f64() * 1e3);
            let last = adm.sampler.sample(&row);
            if last.exhausted {
                roster.leave(lane);
                let _ = sched_push(
                    &adm.tx,
                    ChatEvent::Error(
                        "no legal token at prefill: the sampling mask left every candidate at -inf"
                            .into(),
                    ),
                    send_timeout,
                )
                .await;
                return;
            }
            last
        };
        attach_member_and_stream_the_first_token(
            lanes, tokenizer, eos_ids, roster, members, adm, last,
        )
        .await;
    }

    #[allow(clippy::too_many_arguments)]
    async fn join_prefill_and_stream_the_first_token(
        lanes: &mut QwenLanePool,
        tokenizer: &Arc<tokenizers::Tokenizer>,
        eos_ids: &[u32],
        default_max_new: usize,
        roster: &mut GroupRoster,
        members: &mut Vec<GroupMember>,
        prefill_ms: &mut Vec<f64>,
        job: SchedJob,
    ) {
        let Some(adm) = admit_job_roster_tokenize_started_window_sampler_without_prefilling(
            lanes,
            tokenizer,
            eos_ids,
            default_max_new,
            roster,
            job,
        )
        .await
        else {
            return;
        };
        prefill_admitted_lane_and_stream_the_first_token(
            lanes, tokenizer, eos_ids, roster, members, prefill_ms, adm,
        )
        .await;
    }

    #[allow(clippy::too_many_arguments)]
    async fn run_group(
        lanes: &mut QwenLanePool,
        rx: &mut tokio::sync::mpsc::UnboundedReceiver<SchedJob>,
        tokenizer: &Arc<tokenizers::Tokenizer>,
        eos_ids: &[u32],
        default_max_new: usize,
        t_first: Instant,
        admission_deadline: Instant,
        jobs: Vec<SchedJob>,
    ) {
        let () =
            GROUP_LANES_ARE_ALWAYS_PREFILLED_AT_FORMATION_SO_A_PAD_CLOBBERED_LANE_NEVER_REACHES_A_LIVE_REQUEST;
        let t_group = Instant::now();
        let mut pending: std::collections::VecDeque<SchedJob> = jobs.into();
        let mut roster = GroupRoster::new(lanes.lanes());
        let mut members: Vec<GroupMember> = Vec::with_capacity(lanes.lanes());
        let mut admitted = 0usize;
        let mut prefill_ms: Vec<f64> = Vec::with_capacity(lanes.lanes());
        let packed_formation_armed = matches!(lanes, QwenLanePool::Dense38(_))
            && form_pack_env()
            && lanes.device_argmax_default_on();
        let mut deferred: Vec<AdmittedLane> = Vec::new();
        let mut packed_used = false;
        while admitted < lanes.lanes() {
            if let Some(job) = pending.pop_front() {
                admitted += 1;
                if packed_formation_armed {
                    if let Some(adm) =
                        admit_job_roster_tokenize_started_window_sampler_without_prefilling(
                            lanes,
                            tokenizer,
                            eos_ids,
                            default_max_new,
                            &mut roster,
                            job,
                        )
                        .await
                    {
                        deferred.push(adm);
                    }
                } else {
                    join_prefill_and_stream_the_first_token(
                        lanes,
                        tokenizer,
                        eos_ids,
                        default_max_new,
                        &mut roster,
                        &mut members,
                        &mut prefill_ms,
                        job,
                    )
                    .await;
                }
                continue;
            }
            match rx.try_recv() {
                Ok(j) => pending.push_back(j),
                Err(_) => {
                    let now = Instant::now();
                    if now >= admission_deadline {
                        break;
                    }
                    match tokio::time::timeout_at(
                        tokio::time::Instant::from_std(admission_deadline),
                        rx.recv(),
                    )
                    .await
                    {
                        Ok(Some(j)) => pending.push_back(j),
                        Ok(None) | Err(_) => break,
                    }
                }
            }
        }
        if !deferred.is_empty() {
            let every_deferred_member_is_greedy_so_one_device_argmax_serves_the_packed_call =
                deferred.iter().all(|a| a.sampler.fast_greedy());
            if deferred.len() >= 2
                && every_deferred_member_is_greedy_so_one_device_argmax_serves_the_packed_call
            {
                let _ = Q38_SCHED_PACKED_FORMATION_PREFILLS_EVERY_ADMITTED_LANE_IN_ONE_WIDE_CALL;
                let joins: Vec<(usize, &[u32])> = deferred
                    .iter()
                    .map(|a| (a.lane, a.prompt_ids.as_slice()))
                    .collect();
                let t_prefill = Instant::now();
                match lanes.prefill_lanes_packed_greedy_tokens_on_device(&joins) {
                    Ok(toks) => {
                        prefill_ms.push(t_prefill.elapsed().as_secs_f64() * 1e3);
                        packed_used = true;
                        for (mut adm, tok) in deferred.drain(..).zip(toks) {
                            adm.sampler.record_token(tok);
                            let last = SampleOutput {
                                token: tok,
                                logprob: None,
                                top: Vec::new(),
                                exhausted: false,
                            };
                            attach_member_and_stream_the_first_token(
                                lanes,
                                tokenizer,
                                eos_ids,
                                &mut roster,
                                &mut members,
                                adm,
                                last,
                            )
                            .await;
                        }
                    }
                    Err(e) => {
                        let msg = format!("packed batch prefill: {e:#}");
                        let send_timeout = sched_send_timeout();
                        for adm in deferred.drain(..) {
                            roster.leave(adm.lane);
                            let _ =
                                sched_push(&adm.tx, ChatEvent::Error(msg.clone()), send_timeout)
                                    .await;
                        }
                    }
                }
            } else {
                for adm in deferred.drain(..) {
                    prefill_admitted_lane_and_stream_the_first_token(
                        lanes,
                        tokenizer,
                        eos_ids,
                        &mut roster,
                        &mut members,
                        &mut prefill_ms,
                        adm,
                    )
                    .await;
                }
            }
            while admitted < lanes.lanes() {
                match rx.try_recv() {
                    Ok(job) => {
                        admitted += 1;
                        join_prefill_and_stream_the_first_token(
                            lanes,
                            tokenizer,
                            eos_ids,
                            default_max_new,
                            &mut roster,
                            &mut members,
                            &mut prefill_ms,
                            job,
                        )
                        .await;
                    }
                    Err(_) => break,
                }
            }
        }
        let group_n = admitted;
        let form_ms = t_first.elapsed().as_secs_f64() * 1e3;
        tracing::info!(
            group = group_n,
            "{} batch group formed: MTP bypassed for the whole group (batch XOR spec)",
            lanes.family_tag()
        );

        let every_member_greedy_so_the_group_samples_by_device_argmax_and_streams_tokens_without_a_logits_dtoh =
            members.iter().all(|m| m.sampler.fast_greedy()) && lanes.device_argmax_default_on();
        let mut step_ms: Vec<f64> = Vec::new();
        let mut host_ms = 0f64;
        let step_overlap_engaged =
            step_overlap_env_opt_in_nv_q38_step_overlap_1_the_next_step_launches_before_detok_and_sse_so_host_consume_rides_the_gpu_step()
                && every_member_greedy_so_the_group_samples_by_device_argmax_and_streams_tokens_without_a_logits_dtoh
                && matches!(lanes, QwenLanePool::Dense38(_))
                && members.iter().all(|m| {
                    !m.logprobs && m.stop_free_so_deferred_emission_cannot_miss_a_stop_hit
                });
        if step_overlap_engaged {
            for m in members.iter_mut() {
                m.emitted_tokens = m.generated_ids.len();
            }
            'pgroup: loop {
                let mut feed: Vec<Option<u32>> = vec![None; lanes.lanes()];
                for m in members.iter().filter(|m| m.alive) {
                    feed[m.lane] = Some(m.last.token);
                }
                let plan = match roster.step_plan(&feed) {
                    Ok(Some(p)) => p,
                    Ok(None) => break 'pgroup,
                    Err(e) => {
                        for m in members.iter_mut().filter(|m| m.alive) {
                            let _ = m
                                .push_evicting_this_member_on_timeout_or_close(ChatEvent::Error(
                                    format!("{e:#}"),
                                ))
                                .await;
                            m.abandon();
                        }
                        break 'pgroup;
                    }
                };
                let t_step = Instant::now();
                if let Err(e) = lanes.step_batch_greedy_tokens_launch(&plan) {
                    for m in members.iter_mut().filter(|m| m.alive) {
                        let _ = m
                            .push_evicting_this_member_on_timeout_or_close(ChatEvent::Error(
                                format!("batch decode step: {e:#}"),
                            ))
                            .await;
                        m.abandon();
                    }
                    break 'pgroup;
                }
                let t_host = Instant::now();
                for m in members.iter_mut() {
                    let was_alive_so_an_eviction_during_emit_must_free_its_lane = m.alive;
                    m.emit_pending_deltas_then_any_deferred_finish(tokenizer).await;
                    if was_alive_so_an_eviction_during_emit_must_free_its_lane && !m.alive {
                        roster.leave(m.lane);
                    }
                }
                host_ms += t_host.elapsed().as_secs_f64() * 1e3;
                let toks = match lanes.step_batch_greedy_tokens_collect(&plan) {
                    Ok(t) => t,
                    Err(e) => {
                        for m in members.iter_mut().filter(|m| m.alive) {
                            let _ = m
                                .push_evicting_this_member_on_timeout_or_close(ChatEvent::Error(
                                    format!("batch decode step: {e:#}"),
                                ))
                                .await;
                            m.abandon();
                        }
                        break 'pgroup;
                    }
                };
                step_ms.push(t_step.elapsed().as_secs_f64() * 1e3);
                for m in members.iter_mut().filter(|m| m.alive) {
                    let Some(tok) = toks.get(m.lane).copied().flatten() else {
                        let _ = m
                            .push_evicting_this_member_on_timeout_or_close(ChatEvent::Error(
                                "batch step returned no token for an active lane".into(),
                            ))
                            .await;
                        m.abandon();
                        roster.leave(m.lane);
                        continue;
                    };
                    m.advance_on_device_argmax_token_deferring_detok_and_sse_to_emit_pending(
                        tok,
                        eos_ids,
                        lanes.lane_pos(m.lane),
                        lanes.lane_max_seq_len(),
                    );
                    if !m.alive {
                        roster.leave(m.lane);
                    }
                }
            }
            let t_host = Instant::now();
            for m in members.iter_mut() {
                m.emit_pending_deltas_then_any_deferred_finish(tokenizer).await;
            }
            host_ms += t_host.elapsed().as_secs_f64() * 1e3;
            emit_group_drain_attrib(
                lanes,
                &members,
                group_n,
                t_group,
                form_ms,
                &prefill_ms,
                &step_ms,
                host_ms,
                packed_used,
                true,
            );
            return;
        }
        'group: loop {
            let mut feed: Vec<Option<u32>> = vec![None; lanes.lanes()];
            for m in members.iter().filter(|m| m.alive) {
                feed[m.lane] = Some(m.last.token);
            }
            let plan = match roster.step_plan(&feed) {
                Ok(Some(p)) => p,
                Ok(None) => break 'group,
                Err(e) => {
                    for m in members.iter_mut().filter(|m| m.alive) {
                        let _ = m
                            .push_evicting_this_member_on_timeout_or_close(ChatEvent::Error(
                                format!("{e:#}"),
                            ))
                            .await;
                        m.abandon();
                    }
                    break 'group;
                }
            };
            let t_step = Instant::now();
            let stepped = if every_member_greedy_so_the_group_samples_by_device_argmax_and_streams_tokens_without_a_logits_dtoh
            {
                lanes
                    .step_batch_greedy_tokens_on_device(&plan)
                    .map(GroupStepOut::DeviceArgmaxTokens)
            } else {
                lanes.step_batch(&plan).map(GroupStepOut::HostSampledRows)
            };
            let stepped = match stepped {
                Ok(r) => r,
                Err(e) => {
                    for m in members.iter_mut().filter(|m| m.alive) {
                        let _ = m
                            .push_evicting_this_member_on_timeout_or_close(ChatEvent::Error(
                                format!("batch decode step: {e:#}"),
                            ))
                            .await;
                        m.abandon();
                    }
                    break 'group;
                }
            };
            step_ms.push(t_step.elapsed().as_secs_f64() * 1e3);
            let t_host = Instant::now();
            for m in members.iter_mut().filter(|m| m.alive) {
                m.last = match &stepped {
                    GroupStepOut::HostSampledRows(rows) => {
                        let Some(row) = rows.get(m.lane).and_then(|r| r.as_ref()) else {
                            let _ = m
                                .push_evicting_this_member_on_timeout_or_close(ChatEvent::Error(
                                    "batch step returned no row for an active lane".into(),
                                ))
                                .await;
                            m.abandon();
                            roster.leave(m.lane);
                            continue;
                        };
                        m.sampler.sample(row)
                    }
                    GroupStepOut::DeviceArgmaxTokens(toks) => {
                        let Some(tok) = toks.get(m.lane).copied().flatten() else {
                            let _ = m
                                .push_evicting_this_member_on_timeout_or_close(ChatEvent::Error(
                                    "batch step returned no token for an active lane".into(),
                                ))
                                .await;
                            m.abandon();
                            roster.leave(m.lane);
                            continue;
                        };
                        m.sampler.record_token(tok);
                        SampleOutput {
                            token: tok,
                            logprob: None,
                            top: Vec::new(),
                            exhausted: false,
                        }
                    }
                };
                if m.last.exhausted {
                    let msg = format!(
                        "no legal token at step {}: the sampling mask left every candidate at -inf",
                        m.generated_ids.len()
                    );
                    let _ = m
                        .push_evicting_this_member_on_timeout_or_close(ChatEvent::Error(msg))
                        .await;
                    m.abandon();
                    roster.leave(m.lane);
                    continue;
                }
                m.consume_sampled_token_and_stream_it_so_ttft_is_paid_at_the_lanes_own_prefill(
                    tokenizer,
                    eos_ids,
                    lanes.lane_pos(m.lane),
                    lanes.lane_max_seq_len(),
                )
                .await;
                if !m.alive {
                    roster.leave(m.lane);
                }
            }
            host_ms += t_host.elapsed().as_secs_f64() * 1e3;
        }

        emit_group_drain_attrib(
            lanes,
            &members,
            group_n,
            t_group,
            form_ms,
            &prefill_ms,
            &step_ms,
            host_ms,
            packed_used,
            false,
        );
    }

    #[allow(clippy::too_many_arguments)]
    fn emit_group_drain_attrib(
        lanes: &QwenLanePool,
        members: &[GroupMember],
        group_n: usize,
        t_group: Instant,
        form_ms: f64,
        prefill_ms: &[f64],
        step_ms: &[f64],
        host_ms: f64,
        packed_used: bool,
        step_overlap: bool,
    ) {
        let total_tokens: u32 = members.iter().map(|m| m.completion_tokens).sum();
        let wall = t_group.elapsed().as_secs_f64();
        tracing::info!(
            group = group_n,
            total_tokens,
            wall_s = format!("{wall:.2}").as_str(),
            agg_tok_s = format!("{:.1}", total_tokens as f64 / wall.max(1e-9)).as_str(),
            captures = lanes.captures(),
            replays = lanes.replays(),
            "{} batch group drained",
            lanes.family_tag()
        );
        eprintln!(
            "{} group={group_n} drained: total_tokens={total_tokens} wall={wall:.2}s \
             agg={:.1} tok/s captures={} replays={}",
            lanes.log_prefix(),
            total_tokens as f64 / wall.max(1e-9),
            lanes.captures(),
            lanes.replays()
        );
        let first_step_ms = step_ms.first().copied().unwrap_or(0.0);
        let p50_step_ms = {
            let mut s: Vec<f64> = step_ms.iter().skip(1).copied().collect();
            s.sort_by(|a, b| a.partial_cmp(b).unwrap());
            s.get(s.len() / 2).copied().unwrap_or(0.0)
        };
        eprintln!(
            "{} group={group_n} prefill_scan={} form={} step_overlap={} form_ms={form_ms:.0} \
             prefill_sum_ms={:.0} prefill_each_ms={:?} first_step_ms={first_step_ms:.0} \
             steps={} p50_step_ms={p50_step_ms:.1} steady_step_sum_ms={:.0} \
             host_sample_detok_sse_ms={host_ms:.0} wall_ms={:.0} tokens={total_tokens}",
            lanes.attrib_prefix(),
            group_formation_prefill_scan_arm(),
            if packed_used { "packed" } else { "serial" },
            step_overlap as u8,
            prefill_ms.iter().sum::<f64>(),
            prefill_ms.iter().map(|x| x.round()).collect::<Vec<f64>>(),
            step_ms.len(),
            step_ms.iter().skip(1).sum::<f64>(),
            wall * 1e3,
        );
    }

    #[cfg(test)]
    mod sched_sse_eviction {
        use super::*;
        use std::time::Duration;

        fn tiny_tokenizer() -> Arc<tokenizers::Tokenizer> {
            let model = tokenizers::models::wordlevel::WordLevel::builder()
                .vocab(
                    ["alpha", "beta", "<eos>"]
                        .iter()
                        .enumerate()
                        .map(|(i, s)| ((*s).to_string(), i as u32))
                        .collect(),
                )
                .build()
                .expect("word-level model");
            Arc::new(tokenizers::Tokenizer::new(model))
        }

        fn greedy_req() -> ChatGenerateRequest {
            ChatGenerateRequest {
                prompt: "alpha".into(),
                max_new_tokens: 4,
                stop: vec![],
                seed: Some(0),
                temperature: Some(0.0),
                top_p: None,
                top_k: None,
                min_p: None,
                presence_penalty: None,
                frequency_penalty: None,
                repetition_penalty: None,
                guided: None,
                guided_think_close: None,
                logit_bias: vec![],
                logprobs: false,
                top_logprobs: 0,
                kv_resume: None,
                kv_store: None,
                mm: None,
            }
        }

        const EOS: u32 = 2;

        fn member(
            lane: usize,
            tx: mpsc::Sender<ChatEvent>,
            tok: &Arc<tokenizers::Tokenizer>,
            timeout: Option<Duration>,
            max_new: usize,
        ) -> GroupMember {
            GroupMember {
                lane,
                tx,
                detok: IncrementalDetok::new(tok.clone()),
                emitter: StreamEmitter::new(&[]),
                sampler: ChatSampler::for_request(&greedy_req(), tok, &[EOS], max_new)
                    .expect("greedy sampler"),
                last: SampleOutput {
                    token: 0,
                    logprob: None,
                    top: Vec::new(),
                    exhausted: false,
                },
                generated_ids: Vec::with_capacity(max_new),
                max_new,
                completion_tokens: 0,
                finish_reason: "length".into(),
                logprobs: false,
                alive: true,
                abandoned: false,
                emitted_tokens: 0,
                pending_finish: None,
                stop_free_so_deferred_emission_cannot_miss_a_stop_hit: true,
                sse_timeout_none_is_the_old_infinite_wait: timeout,
            }
        }

        #[tokio::test]
        async fn a_non_reading_client_is_evicted_after_the_timeout_while_its_sibling_streams_to_done(
        ) {
            let tok = tiny_tokenizer();
            let max_new = 4usize;
            let timeout = Some(Duration::from_millis(80));
            let (htx, mut hrx) = mpsc::channel::<ChatEvent>(64);
            let (stx, _srx_held_open_so_a_full_channel_times_out_instead_of_closing) =
                mpsc::channel::<ChatEvent>(1);
            let mut roster = GroupRoster::new(2);
            let mut healthy = member(roster.join().unwrap(), htx, &tok, timeout, max_new);
            let mut stalled = member(roster.join().unwrap(), stx, &tok, timeout, max_new);
            roster.mark_prefilled(healthy.lane).unwrap();
            roster.mark_prefilled(stalled.lane).unwrap();
            let drain = tokio::spawn(async move {
                let mut events = Vec::new();
                while let Some(ev) = hrx.recv().await {
                    events.push(ev);
                }
                events
            });
            let t0 = std::time::Instant::now();
            for step in 0..max_new {
                for m in [&mut healthy, &mut stalled] {
                    if !m.alive {
                        continue;
                    }
                    m.last = SampleOutput {
                        token: (step % 2) as u32,
                        logprob: None,
                        top: Vec::new(),
                        exhausted: false,
                    };
                    m.consume_sampled_token_and_stream_it_so_ttft_is_paid_at_the_lanes_own_prefill(
                        &tok,
                        &[EOS],
                        step + 1,
                        1024,
                    )
                    .await;
                    if !m.alive {
                        roster.leave(m.lane);
                    }
                }
            }
            let elapsed = t0.elapsed();
            assert!(
                stalled.abandoned,
                "the stalled member must be evicted, not waited on forever"
            );
            assert!(
                !healthy.abandoned,
                "evicting the stalled member must not touch its sibling"
            );
            assert_eq!(healthy.finish_reason, "length");
            assert_eq!(healthy.completion_tokens, max_new as u32);
            assert!(
                elapsed >= Duration::from_millis(80),
                "an open-but-full channel must be a timeout, not a close: {elapsed:?}"
            );
            assert!(
                elapsed < Duration::from_secs(5),
                "eviction must be bounded by the send timeout: {elapsed:?}"
            );
            drop(healthy);
            let events = drain.await.unwrap();
            assert!(
                matches!(
                    events.last(),
                    Some(ChatEvent::Done { finish_reason, .. }) if finish_reason == "length"
                ),
                "the healthy stream must complete with Done; got {events:?}"
            );
            assert!(
                events
                    .iter()
                    .any(|e| matches!(e, ChatEvent::TextDelta(s) if !s.is_empty())),
                "the healthy stream must have carried deltas; got {events:?}"
            );
            assert!(
                roster.join().is_some(),
                "the evicted member's lane must be free for the next join"
            );
        }

        #[tokio::test]
        async fn deferred_emission_to_a_stalled_client_aborts_after_the_timeout() {
            let tok = tiny_tokenizer();
            let (stx, _srx_held_open) = mpsc::channel::<ChatEvent>(1);
            let mut m = member(0, stx, &tok, Some(Duration::from_millis(50)), 8);
            m.generated_ids = vec![0, 1, 0];
            let t0 = std::time::Instant::now();
            m.emit_pending_deltas_then_any_deferred_finish(&tok).await;
            assert!(m.abandoned);
            assert!(
                t0.elapsed() < Duration::from_secs(5),
                "deferred emission must not park: {:?}",
                t0.elapsed()
            );
        }

        #[tokio::test]
        async fn the_infinite_wait_escape_hatch_still_detects_a_closed_receiver() {
            let (tx, rx) = mpsc::channel::<ChatEvent>(1);
            drop(rx);
            assert_eq!(
                push_event_async_opt(&tx, ChatEvent::TextDelta("alpha".into()), None).await,
                SsePush::Closed
            );
        }

        #[test]
        fn zero_ms_maps_to_the_old_infinite_wait_and_positive_ms_maps_to_a_deadline() {
            assert_eq!(sse_timeout_opt_from_ms_zero_keeps_the_old_infinite_wait(0), None);
            assert_eq!(
                sse_timeout_opt_from_ms_zero_keeps_the_old_infinite_wait(10_000),
                Some(Duration::from_millis(10_000))
            );
        }
    }
}

#[cfg(test)]
mod roster_state_machine {
    use super::*;

    #[test]
    fn join_assigns_lowest_vacant_lane_and_caps_at_lane_count() {
        let mut r = GroupRoster::new(4);
        assert_eq!(r.join(), Some(0));
        assert_eq!(r.join(), Some(1));
        assert_eq!(r.join(), Some(2));
        assert_eq!(r.join(), Some(3));
        assert_eq!(r.join(), None, "a fifth join must wait for a free lane");
    }

    #[test]
    fn leave_frees_the_lane_for_the_next_join() {
        let mut r = GroupRoster::new(2);
        assert_eq!(r.join(), Some(0));
        assert_eq!(r.join(), Some(1));
        r.leave(0);
        assert_eq!(r.join(), Some(0));
    }

    #[test]
    fn step_plan_trims_to_last_active_and_pads_holes() {
        let mut r = GroupRoster::new(4);
        for _ in 0..4 {
            let lane = r.join().unwrap();
            r.mark_prefilled(lane).unwrap();
        }
        r.leave(1);
        let plan = r
            .step_plan(&[Some(10), None, Some(12), Some(13)])
            .unwrap()
            .unwrap();
        assert_eq!(plan, vec![Some(10), None, Some(12), Some(13)]);
        r.leave(3);
        let plan = r
            .step_plan(&[Some(10), None, Some(12), None])
            .unwrap()
            .unwrap();
        assert_eq!(
            plan,
            vec![Some(10), None, Some(12)],
            "trailing vacancy must shrink the plan so the bucket can narrow"
        );
        r.leave(0);
        r.leave(2);
        assert!(
            r.step_plan(&[None, None, None, None]).unwrap().is_none(),
            "an all-vacant roster has no step"
        );
    }

    #[test]
    fn a_padded_lane_is_clobbered_and_rejoin_requires_prefill_before_decode() {
        let mut r = GroupRoster::new(3);
        for _ in 0..3 {
            let lane = r.join().unwrap();
            r.mark_prefilled(lane).unwrap();
        }
        r.leave(0);
        let _ = r.step_plan(&[None, Some(21), Some(22)]).unwrap().unwrap();
        assert_eq!(
            r.seat(0),
            LaneSeat::VacantPadClobbered,
            "a vacancy covered by the step's bucket is clobbered by the pad row"
        );
        let lane = r.join().unwrap();
        assert_eq!(lane, 0);
        let err = r
            .step_plan(&[Some(30), Some(21), Some(22)])
            .expect_err("decoding a rejoined lane without a prefill must be refused");
        assert!(err.to_string().contains("pad-clobber"), "{err}");
        r.mark_prefilled(0).unwrap();
        let plan = r
            .step_plan(&[Some(30), Some(21), Some(22)])
            .unwrap()
            .unwrap();
        assert_eq!(plan.len(), 3);
    }

    #[test]
    fn step_feed_must_match_the_roster_exactly() {
        let mut r = GroupRoster::new(2);
        let lane = r.join().unwrap();
        r.mark_prefilled(lane).unwrap();
        assert!(r.step_plan(&[Some(1)]).is_err(), "short feed must be refused");
        assert!(
            r.step_plan(&[Some(1), Some(2)]).is_err(),
            "a token on a vacant lane must be refused"
        );
        assert!(
            r.step_plan(&[None, None]).is_err(),
            "an active lane with no token must be refused"
        );
    }

    #[test]
    fn mark_prefilled_is_only_legal_on_a_freshly_joined_lane() {
        let mut r = GroupRoster::new(2);
        assert!(r.mark_prefilled(0).is_err());
        let lane = r.join().unwrap();
        r.mark_prefilled(lane).unwrap();
        assert!(r.mark_prefilled(lane).is_err());
    }
}
