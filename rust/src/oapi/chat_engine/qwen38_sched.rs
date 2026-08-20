#![cfg_attr(not(feature = "cuda"), allow(dead_code))]

use super::*;

pub(crate) const Q38_SCHED_POLICY_FORMATION_WINDOW_THEN_SOLO_XOR_GROUP: &str =
    "after the first pending request the dispatcher drains the queue up to the lane count; a \
     request still alone after NV_Q38_BATCH_WINDOW_MS (default 30) runs SOLO on the existing \
     MTP/eager path, two or more run as a lane group with MTP bypassed for the whole group, and \
     a group with a companion already queued skips the window sleep because the first lane \
     prefill outlasts the window anyway";

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
    use nv_models::qwen3_5_moe::qwen38_batch::Qwen38BatchLanes;
    use std::time::Instant;

    pub(crate) struct SchedJob {
        pub(crate) req: ChatGenerateRequest,
        pub(crate) tx: mpsc::Sender<ChatEvent>,
    }

    pub(crate) struct Qwen38BatchScheduler {
        jobs: std::sync::Mutex<Option<tokio::sync::mpsc::UnboundedSender<SchedJob>>>,
        worker: std::sync::Mutex<Option<std::thread::JoinHandle<()>>>,
        lanes_n: usize,
    }

    impl Qwen38BatchScheduler {
        #[allow(clippy::too_many_arguments)]
        pub(crate) fn spawn(
            lanes: Qwen38BatchLanes,
            mtp: Option<Arc<nv_specdecode::qwen38_mtp::Qwen38DenseMtpHead>>,
            tokenizer: Arc<tokenizers::Tokenizer>,
            device: candle_core::Device,
            eos_ids: Vec<u32>,
            kv_max_seq_len: usize,
            default_max_new: usize,
        ) -> anyhow::Result<Arc<Self>> {
            let (jobs_tx, jobs_rx) = tokio::sync::mpsc::unbounded_channel();
            let lanes_n = lanes.lanes();
            let worker = std::thread::Builder::new()
                .name("q38-batch-sched".into())
                .spawn(move || {
                    let rt = tokio::runtime::Builder::new_current_thread()
                        .enable_all()
                        .build()
                        .expect("q38 batch scheduler: current-thread runtime");
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
                .map_err(|e| anyhow::anyhow!("spawn q38 batch scheduler thread: {e}"))?;
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
                .ok_or_else(|| anyhow::anyhow!("q38 batch scheduler is shut down"))?;
            sender
                .send(SchedJob { req, tx })
                .map_err(|_| anyhow::anyhow!("q38 batch scheduler dispatcher exited"))
        }
    }

    impl Drop for Qwen38BatchScheduler {
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
        mut lanes: Qwen38BatchLanes,
        mtp: Option<Arc<nv_specdecode::qwen38_mtp::Qwen38DenseMtpHead>>,
        tokenizer: Arc<tokenizers::Tokenizer>,
        device: candle_core::Device,
        eos_ids: Vec<u32>,
        kv_max_seq_len: usize,
        default_max_new: usize,
    ) {
        eprintln!(
            "[q38-batch-sched] up: {} lanes, window {} ms, mtp_solo={}. Policy A: {}. Policy B: {}",
            lanes.lanes(),
            q38_batch_window_ms_env_nv_q38_batch_window_ms(),
            mtp.is_some(),
            Q38_SCHED_POLICY_FORMATION_WINDOW_THEN_SOLO_XOR_GROUP,
            Q38_SCHED_POLICY_LATE_JOINERS_WAIT_FOR_FULL_GROUP_DRAIN,
        );
        while let Some(first) = rx.recv().await {
            let t_first = Instant::now();
            let window_ms = q38_batch_window_ms_env_nv_q38_batch_window_ms();
            let admission_deadline = t_first + std::time::Duration::from_millis(window_ms);
            let mut jobs = vec![first];
            while jobs.len() < lanes.lanes() {
                match rx.try_recv() {
                    Ok(j) => jobs.push(j),
                    Err(_) => break,
                }
            }
            if jobs.len() == 1 && window_ms > 0 {
                tokio::time::sleep_until(tokio::time::Instant::from_std(admission_deadline))
                    .await;
                while jobs.len() < lanes.lanes() {
                    match rx.try_recv() {
                        Ok(j) => jobs.push(j),
                        Err(_) => break,
                    }
                }
            }
            if jobs.len() == 1 {
                let job = jobs.pop().unwrap();
                run_solo(
                    &mut lanes,
                    mtp.as_deref(),
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
        lanes: &mut Qwen38BatchLanes,
        mtp: Option<&nv_specdecode::qwen38_mtp::Qwen38DenseMtpHead>,
        tokenizer: &Arc<tokenizers::Tokenizer>,
        device: &candle_core::Device,
        eos_ids: &[u32],
        kv_max_seq_len: usize,
        default_max_new: usize,
        job: SchedJob,
    ) {
        let SchedJob { req, tx } = job;
        if let Err(err) = lanes
            .quiesce_graphs_before_external_eager_work_because_solo_requests_share_the_capture_stream()
        {
            let _ = tx.send(ChatEvent::Error(format!("{err:#}"))).await;
            return;
        }
        let res = run_qwen38_solo_via_scheduler(
            lanes.model(),
            mtp,
            tokenizer,
            device,
            req,
            kv_max_seq_len,
            default_max_new,
            eos_ids,
            &tx,
        )
        .await;
        if let Err(err) = res {
            let _ = tx.send(ChatEvent::Error(format!("{err:#}"))).await;
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
    }

    impl GroupMember {
        fn abandon(&mut self) {
            self.alive = false;
            self.abandoned = true;
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
                    if !tail.is_empty() && self.tx.send(ChatEvent::TextDelta(tail)).await.is_err() {
                        self.abandoned = true;
                        return;
                    }
                }
            }
            let _ = self
                .tx
                .send(ChatEvent::Done {
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
            let (piece, stop_hit) = match self.detok.push(tok) {
                Ok(new_text) => self.emitter.step(new_text),
                Err(e) => {
                    let _ = self.tx.send(ChatEvent::Error(format!("{e:#}"))).await;
                    self.abandon();
                    return;
                }
            };
            if !piece.is_empty() && self.tx.send(ChatEvent::TextDelta(piece)).await.is_err() {
                self.abandon();
                return;
            }
            if self.logprobs
                && self
                    .tx
                    .send(ChatEvent::Logprob(build_logprob_entry(tokenizer, &self.last)))
                    .await
                    .is_err()
            {
                self.abandon();
                return;
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

    #[allow(clippy::too_many_arguments)]
    async fn join_prefill_and_stream_the_first_token(
        lanes: &mut Qwen38BatchLanes,
        tokenizer: &Arc<tokenizers::Tokenizer>,
        eos_ids: &[u32],
        default_max_new: usize,
        roster: &mut GroupRoster,
        members: &mut Vec<GroupMember>,
        prefill_ms: &mut Vec<f64>,
        job: SchedJob,
    ) {
        let SchedJob { req, tx } = job;
        let Some(lane) = roster.join() else {
            let _ = tx
                .send(ChatEvent::Error(
                    "q38 batch scheduler formed a group larger than its lane pool".into(),
                ))
                .await;
            return;
        };
        let prompt_ids = match tokenizer.encode(req.prompt.as_str(), false) {
            Ok(e) => e.get_ids().to_vec(),
            Err(e) => {
                roster.leave(lane);
                let _ = tx.send(ChatEvent::Error(format!("tokenize: {e}"))).await;
                return;
            }
        };
        if tx
            .send(ChatEvent::Started {
                prompt_tokens: prompt_ids.len() as u32,
            })
            .await
            .is_err()
        {
            roster.leave(lane);
            return;
        }
        if prompt_ids.is_empty() {
            roster.leave(lane);
            let _ = tx
                .send(ChatEvent::Done {
                    finish_reason: "length".into(),
                    completion_tokens: 0,
                })
                .await;
            return;
        }
        let requested = effective_max_new(req.max_new_tokens, default_max_new);
        let Some((_cache_len, max_new)) =
            kv_window(prompt_ids.len(), requested, lanes.lane_max_seq_len())
        else {
            roster.leave(lane);
            let _ = tx
                .send(ChatEvent::Error(format!(
                    "prompt of {} tokens does not fit the {}-token KV window",
                    prompt_ids.len(),
                    lanes.lane_max_seq_len()
                )))
                .await;
            return;
        };
        if max_new == 0 {
            roster.leave(lane);
            let _ = tx
                .send(ChatEvent::Done {
                    finish_reason: "length".into(),
                    completion_tokens: 0,
                })
                .await;
            return;
        }
        let mut sampler = match ChatSampler::for_request(&req, tokenizer, eos_ids, max_new) {
            Ok(s) => s,
            Err(e) => {
                roster.leave(lane);
                let _ = tx.send(ChatEvent::Error(format!("{e:#}"))).await;
                return;
            }
        };
        sampler.seed_prompt(&prompt_ids);
        let t_prefill = Instant::now();
        let row = match lanes.prefill_lane(lane, &prompt_ids) {
            Ok(r) => r,
            Err(e) => {
                roster.leave(lane);
                let _ = tx.send(ChatEvent::Error(format!("batch prefill: {e:#}"))).await;
                return;
            }
        };
        prefill_ms.push(t_prefill.elapsed().as_secs_f64() * 1e3);
        if roster.mark_prefilled(lane).is_err() {
            roster.leave(lane);
            let _ = tx
                .send(ChatEvent::Error("q38 batch roster desync at prefill".into()))
                .await;
            return;
        }
        let last = sampler.sample(&row);
        if last.exhausted {
            roster.leave(lane);
            let _ = tx
                .send(ChatEvent::Error(
                    "no legal token at prefill: the sampling mask left every candidate at -inf"
                        .into(),
                ))
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
    async fn run_group(
        lanes: &mut Qwen38BatchLanes,
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
        while admitted < lanes.lanes() {
            if let Some(job) = pending.pop_front() {
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
                continue;
            }
            match rx.try_recv() {
                Ok(j) => pending.push_back(j),
                Err(_) => {
                    let now = Instant::now();
                    if now >= admission_deadline {
                        break;
                    }
                    tokio::time::sleep_until(tokio::time::Instant::from_std(admission_deadline))
                        .await;
                    match rx.try_recv() {
                        Ok(j) => pending.push_back(j),
                        Err(_) => break,
                    }
                }
            }
        }
        let group_n = admitted;
        let form_ms = t_first.elapsed().as_secs_f64() * 1e3;
        tracing::info!(
            group = group_n,
            "qwen3.8 batch group formed: MTP bypassed for the whole group (batch XOR spec)"
        );

        let mut step_ms: Vec<f64> = Vec::new();
        let mut host_ms = 0f64;
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
                        let _ = m.tx.send(ChatEvent::Error(format!("{e:#}"))).await;
                        m.abandon();
                    }
                    break 'group;
                }
            };
            let t_step = Instant::now();
            let rows = match lanes.step_batch(&plan) {
                Ok(r) => r,
                Err(e) => {
                    for m in members.iter_mut().filter(|m| m.alive) {
                        let _ = m
                            .tx
                            .send(ChatEvent::Error(format!("batch decode step: {e:#}")))
                            .await;
                        m.abandon();
                    }
                    break 'group;
                }
            };
            step_ms.push(t_step.elapsed().as_secs_f64() * 1e3);
            let t_host = Instant::now();
            for m in members.iter_mut().filter(|m| m.alive) {
                let Some(row) = rows.get(m.lane).and_then(|r| r.as_ref()) else {
                    let _ = m
                        .tx
                        .send(ChatEvent::Error(
                            "batch step returned no row for an active lane".into(),
                        ))
                        .await;
                    m.abandon();
                    roster.leave(m.lane);
                    continue;
                };
                m.last = m.sampler.sample(row);
                if m.last.exhausted {
                    let _ = m
                        .tx
                        .send(ChatEvent::Error(format!(
                            "no legal token at step {}: the sampling mask left every candidate at -inf",
                            m.generated_ids.len()
                        )))
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

        let total_tokens: u32 = members.iter().map(|m| m.completion_tokens).sum();
        let wall = t_group.elapsed().as_secs_f64();
        tracing::info!(
            group = group_n,
            total_tokens,
            wall_s = format!("{wall:.2}").as_str(),
            agg_tok_s = format!("{:.1}", total_tokens as f64 / wall.max(1e-9)).as_str(),
            captures = lanes.captures(),
            replays = lanes.replays(),
            "qwen3.8 batch group drained"
        );
        eprintln!(
            "[q38-batch-sched] group={group_n} drained: total_tokens={total_tokens} wall={wall:.2}s \
             agg={:.1} tok/s captures={} replays={}",
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
            "[q38-sched-attrib] group={group_n} prefill_scan={} form_ms={form_ms:.0} \
             prefill_sum_ms={:.0} prefill_each_ms={:?} first_step_ms={first_step_ms:.0} \
             steps={} p50_step_ms={p50_step_ms:.1} steady_step_sum_ms={:.0} \
             host_sample_detok_sse_ms={host_ms:.0} wall_ms={:.0} tokens={total_tokens}",
            group_formation_prefill_scan_arm(),
            prefill_ms.iter().sum::<f64>(),
            prefill_ms.iter().map(|x| x.round()).collect::<Vec<f64>>(),
            step_ms.len(),
            step_ms.iter().skip(1).sum::<f64>(),
            wall * 1e3,
        );
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
