use super::*;
#[cfg(any(feature = "cuda", test))]
use anyhow::Context;

#[cfg(feature = "cuda")]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ModelFamily {
    Qwen3,
    Gemma4,
    Gemma4E4b,
    Gemma4Moe,
    Qwen3_5Moe,
    Laguna,
    Omni,
    GptOss,
    Qwen4Exp,
}

pub struct NvEngineChat {
    #[cfg(feature = "cuda")]
    pub(crate) inner: NvEngineInner,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum EngineBuildMode {
    Full,
    #[cfg(any(feature = "cuda", test))]
    AdapterServingLean,
}

#[cfg(feature = "cuda")]
pub(crate) enum LoadedModel {
    Qwen3(Arc<tokio::sync::Mutex<nv_models::qwen3::Qwen3>>),
    Gemma4(Arc<nv_models::gemma4::Gemma4>),
    Gemma4E4b(Arc<nv_models::gemma4_e4b::Gemma4E4b>),
    Gemma4Moe(Arc<nv_models::gemma4_moe::Gemma4Moe>),
    Qwen3_5Moe(QwenMoeShared),
    Laguna(LagunaShared),
    Omni(Arc<omni_loop::OmniShared>),
    GptOss(Arc<nv_models::gpt_oss_cuda::GptOssCuda>),
    Qwen4Exp(Arc<nv_models::qwen4_exp::Qwen4Exp>),
}

#[cfg(feature = "cuda")]
#[derive(Clone)]
pub(crate) enum QwenMoeShared {
    Eager(Arc<tokio::sync::Mutex<nv_models::qwen3_5_moe::Qwen3Moe>>),
    Graphed(Arc<tokio::sync::Mutex<nv_models::graph_engine::GraphedQwen3Moe>>),
    Batch(Arc<QwenBatchScheduler>),
}

#[cfg(feature = "cuda")]
#[derive(Clone)]
pub(crate) struct LagunaShared(
    Arc<nv_models::laguna::Laguna>,
    pub(crate) Arc<LagunaAdmission>,
);
#[cfg(feature = "cuda")]
unsafe impl Send for LagunaShared {}
#[cfg(feature = "cuda")]
unsafe impl Sync for LagunaShared {}
#[cfg(feature = "cuda")]
impl std::ops::Deref for LagunaShared {
    type Target = nv_models::laguna::Laguna;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

#[cfg(feature = "cuda")]
pub(crate) struct LagunaSpecServe {
    pub(crate) jobs: LagunaSpecWorker<nv_models::laguna_serve::SpecServeJob>,
    pub(crate) max_seq: usize,
    pub(crate) num_spec: usize,
    pub(crate) tail: LagunaWorkerTail,
    pub(crate) capacity: usize,
    pub(crate) queue: std::time::Duration,
}

#[cfg(feature = "cuda")]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum LagunaWorkerTail {
    Reserved,
    PreparedPlain,
    PreparedSpec,
}

#[cfg(feature = "cuda")]
impl LagunaWorkerTail {
    pub(crate) fn fits(
        self,
        prompt: usize,
        output: usize,
        num_spec: usize,
        capacity: usize,
    ) -> bool {
        if prompt == 0 || output == 0 {
            return false;
        }
        let required = match self {
            Self::Reserved => output.checked_add(num_spec).and_then(|n| n.checked_add(2)),
            Self::PreparedPlain => Some(output.max(2)),
            Self::PreparedSpec => Some(output),
        };
        required
            .and_then(|n| prompt.checked_add(n))
            .is_some_and(|end| end <= capacity)
    }
}

#[cfg(any(feature = "cuda", test))]
#[derive(Debug)]
pub(crate) struct LagunaSpecWorker<T> {
    jobs: Option<std::sync::mpsc::SyncSender<T>>,
    thread: Option<std::thread::JoinHandle<anyhow::Result<()>>>,
}

#[cfg(any(feature = "cuda", test))]
impl<T> LagunaSpecWorker<T> {
    pub(crate) fn try_send(&self, job: T) -> Result<(), std::sync::mpsc::TrySendError<T>> {
        match self.jobs.as_ref() {
            Some(jobs) => jobs.try_send(job),
            None => Err(std::sync::mpsc::TrySendError::Disconnected(job)),
        }
    }

    #[cfg(all(test, feature = "cuda"))]
    pub(crate) fn from_test_sender(jobs: std::sync::mpsc::SyncSender<T>) -> Self {
        Self {
            jobs: Some(jobs),
            thread: None,
        }
    }

    fn close_and_join(&mut self) -> anyhow::Result<()> {
        drop(self.jobs.take());
        let Some(thread) = self.thread.take() else {
            return Ok(());
        };
        if thread.thread().id() == std::thread::current().id() {
            tracing::warn!("Laguna spec worker owner dropped on its own thread; queue closed without self-join");
            return Ok(());
        }
        match thread.join() {
            Ok(result) => result.context("Laguna spec worker exited with an error"),
            Err(panic) => {
                let message = panic
                    .downcast_ref::<String>()
                    .map(String::as_str)
                    .or_else(|| panic.downcast_ref::<&str>().copied())
                    .unwrap_or("non-string panic");
                anyhow::bail!("Laguna spec worker panicked: {message}")
            }
        }
    }
}

#[cfg(any(feature = "cuda", test))]
impl<T> Drop for LagunaSpecWorker<T> {
    fn drop(&mut self) {
        if let Err(error) = self.close_and_join() {
            tracing::warn!(error = %format!("{error:#}"), "Laguna spec worker shutdown failed");
        }
    }
}

#[cfg(any(feature = "cuda", test))]
fn validate_laguna_packed_draft_arm(
    required: bool,
    mode: EngineBuildMode,
    draft_enabled: bool,
) -> anyhow::Result<()> {
    if required {
        anyhow::ensure!(
            mode != EngineBuildMode::AdapterServingLean && draft_enabled,
            "NV_LAGUNA_DFLASH_NVFP4=1 requires the full Laguna spec engine and draft; it conflicts with adapter-serving lean mode or NV_LAGUNA_SERVE_DRAFT=0"
        );
    }
    Ok(())
}

#[cfg(any(feature = "cuda", test))]
fn required_laguna_draft_dir(raw: Option<std::ffi::OsString>) -> anyhow::Result<PathBuf> {
    let path = raw.map(PathBuf::from).context(
        "NV_LAGUNA_DFLASH_NVFP4=1 requires NV_LAGUNA_DFLASH_DIR to identify the draft source",
    )?;
    anyhow::ensure!(
        path.is_dir(),
        "NV_LAGUNA_DFLASH_NVFP4=1 draft source is not a directory: {}",
        path.display()
    );
    Ok(path)
}

#[cfg(test)]
fn start_laguna_spec_worker<T: Send + 'static>(
    capacity: usize,
    required: bool,
    worker: impl FnOnce(std::sync::mpsc::Receiver<T>, std::sync::mpsc::Sender<bool>) -> anyhow::Result<()>
        + Send
        + 'static,
) -> anyhow::Result<Option<(LagunaSpecWorker<T>, bool)>> {
    start_laguna_spec_worker_with_graph_requirement(capacity, required, false, false, worker)
}

#[cfg(any(feature = "cuda", test))]
fn start_laguna_spec_worker_with_graph_requirement<T: Send + 'static>(
    capacity: usize,
    required: bool,
    graph_required: bool,
    handoff_required: bool,
    worker: impl FnOnce(std::sync::mpsc::Receiver<T>, std::sync::mpsc::Sender<bool>) -> anyhow::Result<()>
        + Send
        + 'static,
) -> anyhow::Result<Option<(LagunaSpecWorker<T>, bool)>> {
    let start = || -> anyhow::Result<_> {
        let (jobs, requests) = std::sync::mpsc::sync_channel(capacity);
        let (ready_tx, ready_rx) = std::sync::mpsc::channel();
        let thread = std::thread::Builder::new()
            .name("laguna-spec-serve".into())
            .spawn(move || {
                let result = worker(requests, ready_tx);
                if let Err(e) = &result {
                    tracing::warn!(error = %format!("{e:#}"), "laguna spec serve loop exited");
                }
                result
            })
            .context("spawn Laguna spec worker")?;
        let mut owned = LagunaSpecWorker {
            jobs: Some(jobs),
            thread: Some(thread),
        };
        match ready_rx.recv() {
            Ok(has_draft) => {
                anyhow::ensure!(
                    !required || has_draft,
                    "Laguna spec worker became ready without the required packed DFlash draft"
                );
                Ok((owned, has_draft))
            }
            Err(_) => {
                owned
                    .close_and_join()
                    .context("initialize Laguna spec worker")?;
                anyhow::bail!("Laguna spec worker exited before readiness")
            }
        }
    };
    match start() {
        Ok(ready) => Ok(Some(ready)),
        Err(e) if required => Err(e).context(
            "NV_LAGUNA_DFLASH_NVFP4=1 requires packed DFlash serving; refusing plain or BF16 fallback",
        ),
        Err(e) if handoff_required => Err(e).context(
            "NV_LAGUNA_PREFILL_BF16_HANDOFF=1 requires the handoff worker; refusing per-request fallback",
        ),
        Err(e) if graph_required => Err(e).context(
            "NV_LAGUNA_BATCH_GRAPH=1, m2 or m2-spec requires graph lane serving; refusing per-request fallback",
        ),
        Err(e) => {
            tracing::warn!(error = %format!("{e:#}"), "laguna spec serving engine init failed; using the per-request path");
            Ok(None)
        }
    }
}

#[cfg(any(feature = "cuda", test))]
fn laguna_engine_spec_state<T>(ready: Option<(T, bool)>) -> (Option<T>, Option<&'static str>) {
    match ready {
        Some((worker, has_draft)) => (Some(worker), has_draft.then_some("on")),
        None => (None, None),
    }
}

#[cfg(feature = "cuda")]
pub(crate) fn build_laguna_spec_serve(
    model: &LagunaShared,
    kv_max_seq_len: usize,
    require_nvfp4: bool,
) -> anyhow::Result<Option<(Arc<LagunaSpecServe>, bool)>> {
    let graph_required = nv_models::laguna_serve::batch_graph_enabled_from_env()?;
    nv_models::laguna_serve::validate_m2_graph_selection()?;
    nv_models::laguna_serve::validate_m2_spec_selection()?;
    let spec_batch = nv_models::laguna_serve::batch_m2_spec_enabled_from_env();
    let tail = if spec_batch {
        LagunaWorkerTail::PreparedSpec
    } else if nv_models::laguna_serve::batch_m2_graph_enabled_from_env() {
        LagunaWorkerTail::PreparedPlain
    } else {
        LagunaWorkerTail::Reserved
    };
    if spec_batch {
        anyhow::ensure!(
            require_nvfp4 && laguna_serve_draft_enabled(),
            "m2-spec requires packed DFlash; refusing plain or BF16 fallback"
        );
    }
    if graph_required {
        anyhow::ensure!(spec_batch || (!laguna_serve_draft_enabled() && !require_nvfp4), "NV_LAGUNA_BATCH_GRAPH=1 or m2 requires NV_LAGUNA_SERVE_DRAFT=0 and packed DFlash disabled");
    }
    let draft_enabled = laguna_serve_draft_enabled();
    let num_spec = std::env::var("NV_DFLASH_K")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(4);
    let max_seq = spec_serve_max_seq(
        std::env::var("NV_LAGUNA_SPEC_MAX_SEQ")
            .ok()
            .and_then(|v| v.parse::<usize>().ok()),
        kv_max_seq_len,
    );
    let capacity = chat_permits(std::env::var("NV_CHAT_CONCURRENCY").ok().as_deref());
    let queue = std::time::Duration::from_millis(chat_queue_ms(
        std::env::var("NV_CHAT_QUEUE_MS").ok().as_deref(),
    ));
    let shared = model.clone();
    let draft_dir = if require_nvfp4 {
        anyhow::ensure!(
            draft_enabled,
            "NV_LAGUNA_DFLASH_NVFP4=1 requires NV_LAGUNA_SERVE_DRAFT=1"
        );
        Some(required_laguna_draft_dir(std::env::var_os(
            "NV_LAGUNA_DFLASH_DIR",
        ))?)
    } else {
        draft_enabled.then(laguna_dflash_dir).flatten()
    };
    if draft_enabled && draft_dir.is_none() {
        tracing::warn!("NV_LAGUNA_SERVE_DRAFT=1 but no dflash snapshot found; spec serving will run M=1 step graph only");
    }
    let handoff_required = nv_models::laguna_serve::bf16_prefill_handoff_opt_in();
    let ready = start_laguna_spec_worker_with_graph_requirement(
        capacity,
        require_nvfp4,
        graph_required,
        handoff_required,
        move |job_rx, ready_tx| {
            let target = shared.0.clone();
            let device = target.device().clone();
            let draft = match draft_dir {
                Some(dir) => match nv_models::laguna_serve::load_dflash_draft(&dir, &device) {
                    Ok(d) => {
                        if require_nvfp4 {
                            let summary = d.quantization_summary();
                            anyhow::ensure!(
                            summary.packed_linears > 0 && summary.bf16_fallback_linears == 0,
                            "required packed DFlash source {} loaded an incompatible weight format: {summary:?}",
                            dir.display()
                        );
                        }
                        Some(d)
                    }
                    Err(e) if require_nvfp4 => {
                        return Err(e).with_context(|| {
                            format!("load required packed DFlash draft from {}", dir.display())
                        });
                    }
                    Err(e) => {
                        tracing::warn!(error = %format!("{e:#}"), "laguna dflash draft load failed; spec serving will run M=1 step graph only");
                        None
                    }
                },
                None => None,
            };
            nv_models::laguna_serve::spec_serve_loop(
                target, draft, num_spec, max_seq, job_rx, ready_tx,
            )
            .context("initialize or run Laguna spec serving")
        },
    )?;
    match ready {
        Some((job_tx, has_draft)) => {
            tracing::info!(
                has_draft,
                num_spec,
                max_seq,
                "laguna spec serving engine ready"
            );
            Ok(Some((
                Arc::new(LagunaSpecServe {
                    jobs: job_tx,
                    max_seq,
                    num_spec,
                    tail,
                    capacity,
                    queue,
                }),
                has_draft,
            )))
        }
        None => Ok(None),
    }
}

#[cfg(feature = "cuda")]
pub(crate) fn qwen_graph_enabled() -> bool {
    matches!(
        std::env::var("NV_QWEN_GRAPH").ok().as_deref(),
        Some("1") | Some("on")
    )
}

#[cfg(feature = "cuda")]
pub(crate) const Q38_SERVE_GRAPHED_YIELDS_ONLY_TO_THE_BATCH_LANE_POOL_WHICH_OWNS_THE_MODEL: &str =
    "the graphed solo session serves 1.6-2.0x eager decode with faster TTFT at every measured \
     depth (gserve rows, perf/runs.jsonl); it serves every dense qwen3.8 boot unless \
     NV_Q38_BATCH=1 booted the lane pool, which owns the model";

#[cfg(feature = "cuda")]
pub(crate) fn qwen38_serve_graphed_enabled() -> bool {
    let _ = Q38_SERVE_GRAPHED_YIELDS_ONLY_TO_THE_BATCH_LANE_POOL_WHICH_OWNS_THE_MODEL;
    !nv_models::qwen3_5_moe::qwen38_batch::nv_q38_batch_env_opt_in_nv_q38_batch_1_the_serving_loop_routes_batch_xor_spec_per_request_group()
}

#[cfg(feature = "cuda")]
pub(crate) fn qwen38_serve_graphed_selected(family: ModelFamily, raw_cfg: &str) -> bool {
    matches!(family, ModelFamily::Qwen3_5Moe)
        && qwen3_5_config_declares_dense_ffn(raw_cfg)
        && qwen3_5_dense_cuda_serve_enabled()
        && qwen38_serve_graphed_enabled()
}

#[cfg(feature = "cuda")]
#[cfg(test)]
pub(crate) const A_CANDLE_GRAPH_BODY_NEEDS_THE_DEVICE_STREAM: &str =
    "Device::new_cuda_with_stream \
     is what a family whose CUDA-graph body is a candle-core forward would need, because candle \
     launches only on candle_core::CudaDevice::cuda_stream() and \
     nv_models::gemma4_batch_graph::capture_stream::CaptureStream therefore has to capture on that \
     very stream; on Device::new_cuda it is the legacy NULL stream, which cuStreamBeginCapture \
     refuses outright, so the engine forks and any candle op in the captured body escapes the \
     capture. Gemma4 qualifies through nv_models::gemma4_graph::GraphedGemma4Decoder, \
     GraphedGemma4Verify and nv_models::gemma4_batch_graph::Gemma4BatchGraphFamily -- they \
     capture Gemma4::forward_with_cache_into, forward_verify_dev and a candle layer loop. \
     Gemma4E4b also qualifies (gemma4_e4b.rs captures Gemma4E4b::forward_step_fast_body). Laguna \
     raw solo path does not: laguna_graph.rs does the candle handoff as a memcpy_dtod outside \
     capture. The opt-in m2 and m2-spec prepared Candle bodies require the non-null model stream; \
     this does not clear Laguna's other routes. Gemma4Moe and Qwen3 build no graph";

#[cfg(all(test, feature = "cuda"))]
pub(crate) const THE_GEMMA4_DEVICE_STREAM_FLIP_IS_BLOCKED_BY_A_MEASURED_SERVING_BREAK: &str =
    "Gemma4 is NOT flipped onto Device::new_cuda_with_stream, even though its graph bodies are \
     candle forwards, because the flip was built and measured on 2026-08-11 against \
     nvidia/Gemma-4-31B-IT-NVFP4 and breaks serving: with NV_NO_SPEC=1 every request fails at \
     PREFILL with 'the sampling mask left every candidate at -inf' (prefill is eager and never \
     touches a graph, so this is the device stream alone), and with the default spec path the \
     drafter chain dies with CUDA_ERROR_ILLEGAL_ADDRESS in \
     chain_draft_cached_shift_graphed (nv-specdecode eagle3_loader.rs forks its capture \
     unconditionally). tool_calling_e2e passes 2/2 on the legacy device and 0/2 on the stream \
     device. This is the same shape as task #63, where grouped MoE on a non-default stream \
     produces all-NaN logits, so the blocker is shared and is not gemma4's to fix. The capture \
     itself is fine on the device stream -- nv-models gemma4_capture_stream_policy decodes the \
     31B checkpoint through it with logits bit-equal to the eager path (3653 nodes vs the fork's \
     4133) -- so this refusal is about the eager work the flip drags along, not about capture";

#[cfg(feature = "cuda")]
pub(crate) fn graph_body_is_a_candle_forward(family: ModelFamily) -> bool {
    matches!(family, ModelFamily::Gemma4 | ModelFamily::Gemma4E4b)
}

#[cfg(feature = "cuda")]
pub(crate) fn device_stream_flip_is_cleared_for(family: ModelFamily) -> bool {
    match family {
        ModelFamily::Gemma4 | ModelFamily::Gemma4E4b => false,
        _ => false,
    }
}

#[cfg(feature = "cuda")]
fn laguna_spec_batch_needs_device_stream(family: ModelFamily, selector: &str) -> bool {
    matches!(family, ModelFamily::Laguna) && selector == "m2-spec"
}

#[cfg(feature = "cuda")]
pub(crate) fn capture_needs_the_device_stream(family: ModelFamily, raw_cfg: &str) -> bool {
    family == ModelFamily::Qwen4Exp
        || qwen_graph_decode_selected(family, raw_cfg)
        || qwen38_serve_graphed_selected(family, raw_cfg)
        || qwen38_batch_lanes_boot_selected(family, raw_cfg)
        || qwen36_batch_lanes_boot_selected(family, raw_cfg)
        || qwen36_mtp_captured_serving_selected(family, raw_cfg)
        || laguna_spec_batch_needs_device_stream(
            family,
            &std::env::var("NV_LAGUNA_BATCH_GRAPH").unwrap_or_default(),
        )
        || (graph_body_is_a_candle_forward(family) && device_stream_flip_is_cleared_for(family))
        || laguna_prepared_b2_graph_captures_the_candle_forward_on_the_model_stream(family)
}

#[cfg(feature = "cuda")]
pub(crate) fn laguna_prepared_b2_graph_captures_the_candle_forward_on_the_model_stream(
    family: ModelFamily,
) -> bool {
    matches!(family, ModelFamily::Laguna)
        && nv_models::laguna_serve::batch_m2_graph_enabled_from_env()
}

#[cfg(feature = "cuda")]
pub(crate) fn qwen38_batch_lanes_boot_selected(family: ModelFamily, raw_cfg: &str) -> bool {
    matches!(family, ModelFamily::Qwen3_5Moe)
        && qwen3_5_config_declares_dense_ffn(raw_cfg)
        && qwen3_5_dense_cuda_serve_enabled()
        && nv_models::qwen3_5_moe::qwen38_batch::nv_q38_batch_env_opt_in_nv_q38_batch_1_the_serving_loop_routes_batch_xor_spec_per_request_group()
}

#[cfg(feature = "cuda")]
pub(crate) fn qwen36_batch_lanes_boot_selected(family: ModelFamily, raw_cfg: &str) -> bool {
    matches!(family, ModelFamily::Qwen3_5Moe)
        && !qwen3_5_config_declares_dense_ffn(raw_cfg)
        && nv_models::qwen3_5_moe::qwen36_batch::nv_q36_batch_env_opt_in_nv_q36_batch_1_the_serving_loop_routes_batch_xor_spec_per_request_group()
}

#[cfg(feature = "cuda")]
pub(crate) const QWEN35_DENSE_CUDA_SERVE_ENV: &str = "NV_QWEN35_DENSE_CUDA_SERVE";

#[cfg(feature = "cuda")]
pub(crate) const QWEN35_DENSE_CUDA_OPT_IN_MATCHES_THE_WGPU_ORACLE_AND_STAYS_OPT_IN: &str = "qwen3.5-dense cuda serving is wired and opted into per boot: with NV_QWEN35_DENSE_CUDA_SERVE=1 detect_family stops refusing and try_load parses the checkpoint with nv_models::qwen3_5_moe::Qwen3_5DenseConfig (the same parser the wgpu decoder uses, top-level fields read out of text_config) and builds it through nv_models::qwen3_5_moe::Qwen3Moe::from_loader_dense_quantized, which gives every linear_attention entry of layer_types a LayerMixer::Linear over nv_layers::linear_attn::LinearAttention and every layer a LayerFfn::Dense over nv_layers::mlp::Mlp. Every trunk component this uses -- the output-gated attention, the partial rope, the GDN linear-attention mixer, the nvfp4 linears -- is byte-for-byte the one the qwen3.5-moe cuda path already serves, because Qwen3.5-9B and Qwen3.5-35B-A3B declare the same head_dim, attn_output_gate, partial_rotary_factor and linear_* block; the ONLY surface this path adds over the served MoE path is LayerFfn::Dense. That surface is now executed and compared against nv_models::qwen3_5_dense_wgpu, the real-weights-verified decoder for this family, by the suite covering the dense cuda serving arm (rust/tests/qwen35_dense_cuda_serving_ab.rs, NV_QWEN35_DENSE_CUDA_SERVE_TEST=1): on ig1/Qwen3.5-9B-NVFP4 snapshot 3b9e07b0 the two backends return the same 7 tokens byte for byte through the checkpoint's own chat template, cuda repeats itself across two requests and an engine reload, and a 512-token thinking trace stays on subject while separating from the oracle at character 225, which is NVFP4 accumulation order and not a decode defect. Both the unarmed trunk and the eleven-arm serving stack QWEN35_DENSE_9B_OPT_IN_ARMS_ITS_MEASURED_STACK describes reproduce that seven-token answer byte for byte, and the armed stack separates from the oracle LATER on the long trace (character 357 against 218) while reusing more of its content words (38 of 49 against 35 of 49), so arming is not a coarsening on this checkpoint. It stays opt-in because that is two prompts on one checkpoint rather than a parity measurement, and because flipping it would move backend_select's auto choice for this whole checkpoint class onto a backend that holds more VRAM for the same weights. Without the variable, detect_family keeps refusing with backend_select::QWEN35_DENSE_NO_CUDA and the default behavior is unchanged";

#[cfg(feature = "cuda")]
pub(crate) const GPTOSS_CUDA_OPT_IN_IS_A_DEQUANT_RUNG_AND_WGPU_STAYS_THE_GPT_OSS_DEFAULT: &str = "gpt-oss cuda serving is wired and opted into per boot: with NV_GPTOSS_CUDA_SERVE=1 detect_family answers ModelFamily::GptOss instead of refusing, and try_load parses the checkpoint with nv_models::gpt_oss::GptOssConfig -- the very parser nv_models::gpt_oss_wgpu uses, so the two backends cannot disagree about layer_types, sliding_window, swiglu_limit or the YaRN block -- and builds nv_models::gpt_oss_cuda::GptOssCuda. That decoder adds exactly two things cuda did not have. First, mxfp4: every expert tensor is dequantized to bf16 at load through nv_quant::mxfp4::Mxfp4Tensor::dequantize, which is the same host semantics the wgpu mxfp4 GEMV is already pinned against, and which is lossless into bf16 because an e2m1 value times an e8m0 block scale needs one mantissa bit. Second, attention sinks: nv_layers::attn::sdpa_with_sinks appends the learned per-head sink as one more score column and one all-zero value row, which is algebraically the fold gow_attn.wgsl and gow_prefill.wgsl perform as `m = max(red[0], sink); z = red[0] + exp(sink - m)`. candle_flash_attn takes no sink argument, so this decoder is eager by construction and the alternating sliding_window=128 layers are masked in the same builder rather than by flash_attn_windowed. It stays opt-in on cost, not correctness: dequant-to-bf16 puts openai/gpt-oss-20b at about 41.8 GB resident against about 13.7 GB for a native mxfp4 path, and eager scoring costs heads*rows*context f32 of transient scratch per prefill chunk that the wgpu decoder never materializes. Flipping it would move backend_select's auto choice for this checkpoint class onto the backend that holds three times the weights and cannot yet stream its softmax. Without the variable, detect_family keeps refusing with backend_select::GPTOSS_NO_CUDA_WITHOUT_THE_OPT_IN and the default behavior is unchanged";

#[cfg(feature = "cuda")]
pub(crate) fn gpt_oss_cuda_serve_enabled() -> bool {
    crate::oapi::backend_select::gpt_oss_cuda_serve_enabled()
}

#[cfg(feature = "cuda")]
#[cfg(feature = "cuda")]
pub(crate) fn qwen3_weight_qconfig_default_nvfp4(
    raw_cfg: &str,
) -> anyhow::Result<nv_weights::QuantizationConfig> {
    let declared = nv_weights::QuantizationConfig::from_hf_json_str(raw_cfg)?;
    if !matches!(declared.scheme, nv_weights::QuantScheme::None) {
        return Ok(declared);
    }
    Ok(nv_weights::QuantizationConfig {
        scheme: nv_weights::QuantScheme::Nvfp4,
        ignored_modules: Vec::new(),
        group_size: None,
    })
}

#[cfg(feature = "cuda")]
pub(crate) const GEMMA4_NVFP4_SERVES_THE_THREE_KEY_RECORD_ARM: &str = "gemma4 dense checkpoints whose config.json declares QuantScheme::Nvfp4 (ModelFamily::Gemma4 over nv_models::gemma4::Gemma4, e.g. nvidia/Gemma-4-31B-IT-NVFP4) serve with three keys armed, and try_load_inner arms them before the boot plan so oapi::vram_plan::plan_gemma4 charges the requanted attention pool rather than the bf16 one: NV_ATTN_PROJ_QUANT=nvfp4 requants q/k/v/o at load through Gemma4::from_loader_quantized (16.956 -> 4.769 GB read per token, 60 layers), NV_NVFP4_M1_GEMV=1 routes every short-row NVFP4 linear off the padded cutlass block-scaled tile onto nvfp4_gemv_bf16act, and NV_G4_LMHEAD_I8=1 makes the m=1 decode epilogue read the resident int8 lm_head copy the verify path already reads (2.819 -> 1.410 GB, no new weights). The three are ONE decision over GEMMA4_RECORD_ARM_KEYS_ARE_ARMED_TOGETHER_OR_NOT_AT_ALL, never a subset, because NV_ATTN_PROJ_QUANT=nvfp4 alone is a quality REJECT and only NV_NVFP4_M1_GEMV rescues it -- nv_models::gemma4::ATTN_PROJ_NVFP4_ALONE_IS_A_MEASURED_QUALITY_REJECT_RESCUED_ONLY_BY_NV_NVFP4_M1_GEMV carries that measurement, and try_load refuses to boot any process that reaches the pair without the rescue, escape or no escape. What the default buys, measured on the default per-request serving decode (GraphedGemma4Decoder over new_kv_cache_fp8_windowed, the shape run_sampling_gemma4 builds), interleaved same-session base/arm pairs on a quiet card: 27.876/27.985 -> 19.419/19.542 ms/tok at 256 and 29.472/29.638 -> 21.313/21.515 at 8192, with free VRAM at the paged ladder's own choice point going 59.6 -> 74.3 GB because the requantized projections replace the bf16 originals. What it costs is stated where it is measured, not hidden: this is a quantization-class arm, so it is read against the family's OWN measured null -- gemma4-31B's reassociation ladder spreads 0.41 ppl on the family's only valid eval (ppl_gemma4::gemma4_chat_wrapped_continuation_teacher_forced_ppl_via_serving_decode; #94 ruled raw-corpus gemma4 ppl invalid; the +0.0973 figure once cited here is a qwen3.8 wikitext ARM delta, not a floor, and binds no other family per 08.2) -- and it reads -0.018 ppl on the eager bf16-KV scoring path (16.308 -> 16.290) and +0.068 on the graphed fp8-KV serving shape (16.276 -> 16.344): both inside the 0.41 family null by ~6x, so the quality cost is contained-and-unresolved rather than held-or-better, with top-1 accuracy 0.6699 -> 0.6641 (3 tokens of 512, 0.28 sd) and the shuffled control intact on every row. One suite changes colour in the other direction: gemma4_verify_forward went 1-pass/2-fail under this arm and is 5-pass/0-fail since verify blocks started taking the same nvfp4_gemv_bf16act row by row. The one measured harm is speculative: a k=4 GraphedGemma4Verify round costs 21.57 -> 54.3 ms armed on the fixed route, so a deployment that sets NV_EAGLE3_DRAFT_DIR or NV_DFLASH_DRAFT_DIR (spec decode is itself opt-in -- load_eagle3_state_for_target returns None without the directory, so the default deployment never enters that path) pays it until gemm_nvfp4_w4a16_mk_dual is wired as the verify-block route. Published numbers measured on the pre-flip default are enumerated in /home/user/tmp/wf/servingfix/RESULTS.md by document and line; all of them are ladder rungs in nv-models tests, which do not call this file and therefore still measure what they always measured -- what changes is that they now describe the unarmed process rather than a live request stream";

#[cfg(feature = "cuda")]
pub(crate) const GEMMA4_RECORD_ARM_KEYS_ARE_ARMED_TOGETHER_OR_NOT_AT_ALL: [(&str, &str); 3] = [
    ("NV_ATTN_PROJ_QUANT", "nvfp4"),
    ("NV_NVFP4_M1_GEMV", "1"),
    ("NV_G4_LMHEAD_I8", "1"),
];

#[cfg(feature = "cuda")]
pub(crate) const QWEN35_DENSE_9B_OPT_IN_ARMS_ITS_MEASURED_STACK: &str = "the qwen3.5-dense cuda path stays opt-in per QWEN35_DENSE_CUDA_OPT_IN_MATCHES_THE_WGPU_ORACLE_AND_STAYS_OPT_IN -- this arms it, it does not promote it, and without NV_QWEN35_DENSE_CUDA_SERVE=1 detect_family still refuses with backend_select::QWEN35_DENSE_NO_CUDA and the checkpoint still goes to wgpu. What changes is that taking the opt-in no longer means decoding with none of the eleven arms this checkpoint's speed-of-light round measured: try_load_inner arms QWEN35_DENSE_9B_ARM_KEYS_ARE_ARMED_TOGETHER_OR_NOT_AT_ALL before Qwen3Moe::from_loader_dense_quantized, because NV_Q36_GDN_FP8 and NV_Q36_LM_HEAD_FP8 decide resident weight dtype at load. Unarmed the path decodes at 12.514 ms/tok at 256 and 79.629 at 8192; armed it decodes at 5.425 and 5.900, and the depth term is where the defect really was -- serial_smem serializes a block reduce per KV position, which is why 256->8k cost 6.4x unarmed and costs 1.09x armed. It costs no quality and no VRAM: on ppl_qwen35_dense::qwen35_dense_chat_wrapped_continuation_teacher_forced_ppl_via_serving_decode the stack scores 4.833 against a 4.862 zero-env basis (better on the headline, and the largest single-arm cost in it is NV_Q36_GDN_FP8 at +0.042 -- a cost that stays UNRESOLVED because qwen3.5-9B has no measured reassociation null, its 3-rung ladder spanning only 0.025; the +0.0973 figure once cited here is a qwen3.8 wikitext arm delta, not a floor), its eager and graphed 128-step greedy chains are token-identical, and peak VRAM through a live 256-depth run drops 15877 -> 9183 MiB because the fp8 copies replace their bf16 originals. The unarmed path now carries the splitk-flash graphed decode and the split GDN step by default, so the 12.514/79.629 unarmed rows predate those defaults. The stack is armed ONLY for the trunk it was measured on, QWEN35_DENSE_9B_TRUNK_THE_ARM_STACK_WAS_MEASURED_ON: qwen3.8-27B declares the same model_type qwen3_5 and lands in the same dense arm, but its published serving band belongs to a different twelve-arm stack with NV_NVFP4_M1_GEMV unset, so arming this one for it would flip a performance default nobody measured. Any other dense trunk keeps today's behavior and says so at load";

#[cfg(feature = "cuda")]
pub(crate) const QWEN35_DENSE_9B_ARM_KEYS_ARE_ARMED_TOGETHER_OR_NOT_AT_ALL: [(&str, &str); 8] = [
    ("NV_Q36_GDN_FP8", "1"),
    ("NV_Q36_LM_HEAD_FP8", "1"),
    ("NV_Q38_FP4_MLP", "1"),
    ("NV_Q38_MLP_W4A8", "1"),
    ("NV_Q38_W4A8_MK", "1"),
    ("NV_Q38_SILUQ_SPLIT", "1"),
    ("NV_Q38_NORM_QUANT_FOLD", "1"),
    ("NV_NVFP4_M1_GEMV", "1"),
];

#[cfg(feature = "cuda")]
pub(crate) const QWEN35_DENSE_9B_TRUNK_THE_ARM_STACK_WAS_MEASURED_ON: (usize, usize, usize) =
    (4096, 32, 12288);

#[cfg(feature = "cuda")]
pub(crate) const QWEN38_DENSE_27B_ARM_KEYS_ARE_ARMED_TOGETHER_OR_NOT_AT_ALL: [(&str, &str); 2] =
    [("NV_Q38_GDN_NVFP4", "1"), ("NV_NVFP4_M1_GEMV", "1")];

#[cfg(feature = "cuda")]
pub(crate) const QWEN38_DENSE_27B_TRUNK_THE_GDN_NVFP4_STACK_WAS_MEASURED_ON: (usize, usize, usize) =
    (5120, 64, 17408);

#[cfg(feature = "cuda")]
pub(crate) const QWEN38_DENSE_27B_NVFP4_GDN_ARMS_ITS_TWO_KEY_STACK: &str =
    "qwen3.8-27B (the qwen3_5 dense trunk 5120/64/17408, distinct from the Qwen3.5-9B \
     4096/32/12288 the eleven-arm stack was measured on) arms two keys together, armed by \
     try_load_inner before from_loader_dense_quantized: NV_Q38_GDN_NVFP4=1 double-quants the \
     fp8-checkpoint GDN qkvz/out projections to nvfp4 half-byte resident, and NV_NVFP4_M1_GEMV=1 \
     routes those short-row nvfp4 linears onto nvfp4_gemv_bf16act instead of the padded generic \
     nvfp4 GEMV. They are ONE decision: NV_Q38_GDN_NVFP4 alone is decode-NEUTRAL (the generic \
     nvfp4 GEMV is no faster than fp8 for the 10240x5120 GDN shapes), and only the M1_GEMV route \
     turns the half-byte read into speed. The instrument is the qwen3.8 wikitext ppl gate: \
     decode-stepped chat-wrapped 9.879 -> 9.818 and raw-corpus prefill 11.994 -> 11.964 -- both \
     BETTER, because the M1_GEMV bf16-activation precision gain outweighs the nvfp4 weight loss. \
     What it buys, graphed serving decode: 16.780 -> 15.584 ms/tok at 8k (+7.7%) and 23.229 -> \
     22.070 at 172k (+5.4%), closing about half the base-decode gap to llama.cpp-cuda. \
     The stack is armed ONLY when spec decode is NOT selected: NV_NVFP4_M1_GEMV \
     routes short-row nvfp4 linears onto nvfp4_gemv_bf16act, which is the WRONG route for an MTP \
     verify block, and arming it measurably slows the verify. That is the same harm \
     GEMMA4_NVFP4_SERVES_THE_THREE_KEY_RECORD_ARM records \
     for gemma4 (k=4 verify 21.57 -> 54.3 ms). So with NV_DRAFTER=mtp this checkpoint arms \
     nothing and keeps the fp8 GDN projections; without a drafter it takes the two-key stack and \
     its base-decode win. \
     Two things about that harm are now measured rather than assumed, and both say the gate \
     stays. First, batching the verify block into one nvfp4_gemv_bf16act_kernel<MROWS> launch \
     (ONE_MROWS_LAUNCH_READS_THE_WEIGHT_BLOCK_ONCE_FOR_EVERY_VERIFY_ROW_INSTEAD_OF_ONCE_PER_ROW) \
     shrinks it but does not remove it: verify 26.91 -> 35.42 ms/round at 256 armed, against the \
     23.91 -> 37.59 the pre-mrows row loop paid, and served decode still falls 276.9 -> 214.9 \
     tok/s at 256 and 258.6 -> 205.2 at 8192. The residual is not GDN but MLP: \
     nvfp4_m1_gemv_opted_in() is ONE global gate in nv_layers::linear::matmul_nvfp4_impl, so \
     arming it also drags the 56 nvfp4 MLP layers off the padded cutlass tile -- which is FLAT in \
     m -- onto a gemv that only wins at one row, and a 2.1-3.1x faster gemv still loses to the \
     tile at the m=8 block the suffix arm produces. Second, the obvious escape is a measured \
     QUALITY REJECT, not an oversight: NV_NVFP4_GEMV_ROWS=1 leaves the verify block on the tile \
     and reads FASTER than the unarmed basis on the serving comparator (289.8/292.1 at 256, \
     271.5/270.2 at 8192, +5.6% and +5.1%, verify 25.42 ms) with accept_rate an unchanged 1.000 \
     -- and that witness is vacuous, because the comparator's exact-token tiled prompt makes \
     suffix acceptance trivially perfect. The named three-class instrument rejects it: \
     qwen38_tctile_cert_teacher_forced_verify_walk_divergence_and_nll_mk_kvshare_tctile_matched_streams \
     fails at agentic nll delta +0.02661 against a 0.02248 band, self-scored NLL worsens on every \
     class (agentic 0.3041 -> 0.3453, tiled 0.2326 -> 0.2390, freeform 0.6334 -> 0.7210) and every \
     emitted_fnv moves, because m=1 decode on the gemv and m=8 verify on the tile is precisely what \
     A_VERIFY_BLOCK_MUST_TAKE_THE_SAME_KERNEL_DECODE_TAKES_OR_IT_CANNOT_REPRODUCE_DECODES_ROWS \
     forbids. A real fix therefore needs ONE kernel that is both tile-fast at m>1 and the kernel \
     m=1 decode takes -- gemm_nvfp4_w4a16_mk_dual wired as BOTH routes, never as the verify \
     route alone.";

#[cfg(feature = "cuda")]
pub(crate) fn qwen38_spec_decode_selected_so_the_m1_gemv_verify_harm_outweighs_the_base_decode_win(
) -> bool {
    nv_specdecode::qwen38_mtp::mtp_drafter_selected_from_env()
}

#[cfg(feature = "cuda")]
pub(crate) fn qwen35_dense_trunk_is_the_measured_27b(raw_cfg: &str) -> bool {
    let (hidden, layers, intermediate) = QWEN38_DENSE_27B_TRUNK_THE_GDN_NVFP4_STACK_WAS_MEASURED_ON;
    nv_models::qwen3_5_moe::Qwen3_5DenseConfig::from_hf_json_str(raw_cfg)
        .map(|c| {
            c.hidden_size == hidden
                && c.num_hidden_layers == layers
                && c.intermediate_size == intermediate
        })
        .unwrap_or(false)
}

#[cfg(feature = "cuda")]
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum ServingArm {
    ArmEveryKey,
    CheckpointOutsideTheMeasuredBasis,
}

#[cfg(feature = "cuda")]
pub(crate) fn serving_arm_from(checkpoint_is_the_measured_basis: bool) -> ServingArm {
    if checkpoint_is_the_measured_basis {
        ServingArm::ArmEveryKey
    } else {
        ServingArm::CheckpointOutsideTheMeasuredBasis
    }
}

#[cfg(feature = "cuda")]
pub(crate) fn arm_keys_the_operator_already_set_to_something_else<'a>(
    keys: &'a [(&'a str, &'a str)],
    read: impl Fn(&str) -> Option<String>,
) -> Vec<(&'a str, String, &'a str)> {
    keys.iter()
        .filter_map(|(k, want)| {
            read(k)
                .filter(|held| held != want)
                .map(|held| (*k, held, *want))
        })
        .collect()
}

#[cfg(feature = "cuda")]
pub(crate) fn keys_armed_by_this_one_decision<'a>(
    arm: ServingArm,
    keys: &'a [(&'a str, &'a str)],
) -> &'a [(&'a str, &'a str)] {
    match arm {
        ServingArm::ArmEveryKey => keys,
        ServingArm::CheckpointOutsideTheMeasuredBasis => &[],
    }
}

#[cfg(feature = "cuda")]
fn apply_serving_arm(
    label: &str,
    why: &str,
    keys: &[(&str, &str)],
    arm: ServingArm,
) -> anyhow::Result<()> {
    let armed = keys_armed_by_this_one_decision(arm, keys);
    if armed.is_empty() {
        eprintln!(
            "[{label}] serving arm OFF (this checkpoint is outside the trunk the stack was \
             measured on): none of {} is armed, so this process runs the pre-flip default, which \
             is the absence of those variables",
            keys.iter().map(|(k, _)| *k).collect::<Vec<_>>().join(" ")
        );
        return Ok(());
    }
    let conflicts =
        arm_keys_the_operator_already_set_to_something_else(armed, |k| std::env::var(k).ok());
    anyhow::ensure!(
        conflicts.is_empty(),
        "{}",
        conflicts
            .iter()
            .map(|(k, held, want)| format!(
                "{k}={held} is set in the environment but this checkpoint serves {k}={want} by \
                 default, and these keys are armed together or not at all because a subset is a \
                 configuration nothing measured"
            ))
            .collect::<Vec<_>>()
            .join("; ")
    );
    for (k, v) in armed {
        unsafe { std::env::set_var(k, v) };
    }
    eprintln!(
        "[{label}] serving arm ON: {}",
        armed
            .iter()
            .map(|(k, v)| format!("{k}={v}"))
            .collect::<Vec<_>>()
            .join(" ")
    );
    eprintln!("[{label}] {why}");
    Ok(())
}

#[cfg(feature = "cuda")]
pub(crate) fn gemma4_checkpoint_declares_nvfp4(raw_cfg: &str) -> bool {
    nv_weights::QuantizationConfig::from_hf_json_str(raw_cfg)
        .map(|q| matches!(q.scheme, nv_weights::QuantScheme::Nvfp4))
        .unwrap_or(false)
}

#[cfg(feature = "cuda")]
pub(crate) fn qwen35_dense_trunk_is_the_measured_9b(raw_cfg: &str) -> bool {
    let (hidden, layers, intermediate) = QWEN35_DENSE_9B_TRUNK_THE_ARM_STACK_WAS_MEASURED_ON;
    nv_models::qwen3_5_moe::Qwen3_5DenseConfig::from_hf_json_str(raw_cfg)
        .map(|c| {
            c.hidden_size == hidden
                && c.num_hidden_layers == layers
                && c.intermediate_size == intermediate
        })
        .unwrap_or(false)
}

#[cfg(feature = "cuda")]
pub(crate) fn gemma4_attn_nvfp4_without_the_m1_gemv_rescue(
    attn_proj_quant: Option<&str>,
    m1_gemv: Option<&str>,
) -> bool {
    nv_models::gemma4::attn_proj_env_selects_nvfp4_projections(attn_proj_quant)
        && m1_gemv != Some("1")
}

#[cfg(feature = "cuda")]
pub(crate) fn arm_the_serving_defaults_for_this_checkpoint(
    family: ModelFamily,
    qwen3_5_dense: bool,
    raw_cfg: &str,
) -> anyhow::Result<()> {
    match family {
        ModelFamily::Gemma4 => {
            apply_serving_arm(
                "gemma4",
                GEMMA4_NVFP4_SERVES_THE_THREE_KEY_RECORD_ARM,
                &GEMMA4_RECORD_ARM_KEYS_ARE_ARMED_TOGETHER_OR_NOT_AT_ALL,
                serving_arm_from(gemma4_checkpoint_declares_nvfp4(raw_cfg)),
            )?;
            anyhow::ensure!(
                !gemma4_attn_nvfp4_without_the_m1_gemv_rescue(
                    std::env::var("NV_ATTN_PROJ_QUANT").ok().as_deref(),
                    std::env::var("NV_NVFP4_M1_GEMV").ok().as_deref(),
                ),
                "{}",
                nv_models::gemma4::ATTN_PROJ_NVFP4_ALONE_IS_A_MEASURED_QUALITY_REJECT_RESCUED_ONLY_BY_NV_NVFP4_M1_GEMV
            );
            Ok(())
        }
        ModelFamily::Qwen3_5Moe
            if qwen3_5_dense && qwen35_dense_trunk_is_the_measured_27b(raw_cfg) =>
        {
            apply_serving_arm(
                "qwen3.8-27B",
                QWEN38_DENSE_27B_NVFP4_GDN_ARMS_ITS_TWO_KEY_STACK,
                &QWEN38_DENSE_27B_ARM_KEYS_ARE_ARMED_TOGETHER_OR_NOT_AT_ALL,
                serving_arm_from(
                    !qwen38_spec_decode_selected_so_the_m1_gemv_verify_harm_outweighs_the_base_decode_win(),
                ),
            )
        }
        ModelFamily::Qwen3_5Moe if qwen3_5_dense => apply_serving_arm(
            "qwen3.5-dense",
            QWEN35_DENSE_9B_OPT_IN_ARMS_ITS_MEASURED_STACK,
            &QWEN35_DENSE_9B_ARM_KEYS_ARE_ARMED_TOGETHER_OR_NOT_AT_ALL,
            serving_arm_from(qwen35_dense_trunk_is_the_measured_9b(raw_cfg)),
        ),
        _ => Ok(()),
    }
}

#[cfg(feature = "cuda")]
pub(crate) fn qwen3_5_dense_cuda_serve_enabled() -> bool {
    matches!(
        std::env::var(QWEN35_DENSE_CUDA_SERVE_ENV).ok().as_deref(),
        Some("1") | Some("on")
    )
}

#[cfg(feature = "cuda")]
pub(crate) fn qwen3_5_config_declares_dense_ffn(raw_cfg: &str) -> bool {
    let Ok(v) = serde_json::from_str::<serde_json::Value>(raw_cfg) else {
        return false;
    };
    let Some(text) = v.get("text_config") else {
        return false;
    };
    if nv_models::qwen3_5_moe::MOE_ONLY_KEYS
        .iter()
        .any(|k| text.get(k).is_some())
    {
        return false;
    }
    text.get("intermediate_size")
        .and_then(|x| x.as_u64())
        .is_some_and(|n| n > 0)
}

#[cfg(feature = "cuda")]
pub(crate) fn qwen_graph_decode_selected(family: ModelFamily, raw_cfg: &str) -> bool {
    matches!(family, ModelFamily::Qwen3_5Moe)
        && !qwen3_5_config_declares_dense_ffn(raw_cfg)
        && qwen_graph_enabled()
}

#[cfg(feature = "cuda")]
pub(crate) fn qwen36_mtp_captured_serving_selected(family: ModelFamily, raw_cfg: &str) -> bool {
    matches!(family, ModelFamily::Qwen3_5Moe)
        && !qwen3_5_config_declares_dense_ffn(raw_cfg)
        && nv_specdecode::qwen38_mtp::mtp_drafter_selected_from_env()
}

#[cfg(feature = "cuda")]
pub(crate) fn build_qwen_moe_dispatch(
    model: &nv_models::qwen3_5_moe::Qwen3Moe,
) -> Option<Arc<nv_models::qwen3_5_moe::GroupedMoeDispatch>> {
    let free_before = nv_layers::cudarc::driver::result::mem_get_info()
        .ok()
        .map(|(free, _total)| free);
    match nv_models::qwen3_5_moe::GroupedMoeDispatch::from_model(model) {
        Ok(d) => {
            let delta_mib = free_before
                .zip(
                    nv_layers::cudarc::driver::result::mem_get_info()
                        .ok()
                        .map(|(free, _total)| free),
                )
                .map(|(before, after)| before.saturating_sub(after) as f64 / (1024.0 * 1024.0));
            match delta_mib {
                Some(mib) if mib >= 1024.0 => eprintln!(
                    "[qwen3.6-moe] MoE dispatch: GROUPED -- WARNING: grouped expert copy holds {mib:.0} MiB extra VRAM on top of the host-path weights"
                ),
                Some(mib) => eprintln!(
                    "[qwen3.6-moe] MoE dispatch: GROUPED, grouped expert copy +{mib:.0} MiB VRAM"
                ),
                None => eprintln!(
                    "[qwen3.6-moe] MoE dispatch: GROUPED, VRAM delta unavailable"
                ),
            }
            Some(Arc::new(d))
        }
        Err(e) => {
            eprintln!(
                "[qwen3.6-moe] MoE dispatch: HOST -- grouped dispatch build FAILED, falling back: {e:#}"
            );
            None
        }
    }
}

#[cfg(feature = "cuda")]
pub(crate) struct Eagle3State {
    pub(crate) verify:
        Option<nv_models::gemma4_graph::GraphedGemma4Verify<Arc<nv_models::gemma4::Gemma4>>>,

    pub(crate) lease_out: bool,

    pub(crate) chain: Option<nv_specdecode::eagle3_loader::DraftChainGraph>,

    pub(crate) dflash_draft: Option<(
        nv_specdecode::dflash::DFlashContextKv,
        nv_specdecode::dflash::DFlashBlockGraph,
    )>,
}

#[cfg(feature = "cuda")]
pub(crate) struct Eagle3Shared {
    pub(crate) proposer:
        nv_specdecode::eagle3::Eagle3Proposer<nv_specdecode::eagle3_loader::LoadedEagle3Scorer>,
    pub(crate) aux_layers: Vec<usize>,

    pub(crate) pool: tokio::sync::Mutex<Eagle3State>,
}

#[cfg(feature = "cuda")]
pub(crate) struct DFlashShared {
    pub(crate) drafter: nv_specdecode::dflash::LoadedDFlashDrafter,
    pub(crate) aux_layers: Vec<usize>,

    pub(crate) pool: tokio::sync::Mutex<Eagle3State>,
}

#[cfg(feature = "cuda")]
#[derive(Clone)]
pub(crate) enum QwenMtpHead {
    Dense(Arc<nv_specdecode::qwen38_mtp::Qwen38DenseMtpHead>),
    Moe(Arc<nv_models::qwen3_5_mtp::MtpHead>),
}

#[cfg(feature = "cuda")]
pub(crate) struct NvEngineInner {
    pub(crate) model_id: String,
    pub(crate) family: ModelFamily,
    pub(crate) capture_shape: Option<(usize, usize)>,
    pub(crate) model: LoadedModel,
    pub(crate) tokenizer: Arc<tokenizers::Tokenizer>,
    pub(crate) device: candle_core::Device,
    pub(crate) kv_max_seq_len: usize,
    pub(crate) default_max_new_tokens: u32,
    pub(crate) eos_token_ids: Vec<u32>,
    pub(crate) bos_token_id: Option<u32>,
    pub(crate) eagle3: Option<Arc<Eagle3Shared>>,
    pub(crate) dflash: Option<Arc<DFlashShared>>,

    pub(crate) spec_status: Option<&'static str>,
    pub(crate) chat_template: Option<Arc<crate::oapi::chat_template::ChatTemplate>>,
    pub(crate) gemma4_engine: Option<Arc<nv_engine::BatchEngineHandle>>,
    pub(crate) laguna_spec: Option<Arc<LagunaSpecServe>>,
    pub(crate) qwen_moe_dispatch: Option<Arc<nv_models::qwen3_5_moe::GroupedMoeDispatch>>,
    pub(crate) qwen_mtp: Option<QwenMtpHead>,
    pub(crate) mm_towers: Option<Arc<crate::oapi::chat_multimodal::Gemma4MmTowers>>,
    pub(crate) vram_plan_record_released_on_engine_drop:
        Option<crate::oapi::vram_plan::RecordedBootPlan>,
}

#[cfg(feature = "cuda")]
pub(crate) fn refuse_gguf_checkpoint_dir(model_dir: &Path) -> anyhow::Result<()> {
    if model_dir.join("config.json").exists() {
        return Ok(());
    }
    let Some(gguf) = nv_weights::gguf::lone_gguf_file(model_dir) else {
        return Ok(());
    };
    anyhow::bail!(
        "{} is a GGUF checkpoint dir ({}): the CUDA chat engine cannot boot from a bare GGUF \
         dir -- it detects the family from config.json and reads weights through \
         nv_weights::WeightLoader::open_dir, which is safetensors-only, so a dir without \
         config.json never reaches nv_models::gemma4_moe::Gemma4Moe::from_gguf. The gemma4-MoE \
         serving family itself is wired on cuda (LoadedModel::Gemma4Moe + \
         run_sampling_gemma4_moe), and a gemma4-MoE safetensors snapshot dir can opt its \
         weight VALUES into the QAT Q4 GGUF via that from_gguf (NV_G4MOE_SERVE_QAT_Q4=1; \
         opt-in only, its top-1 accuracy is out of band). Serve this bare dir on the wgpu \
         chat engine (feature \"wgpu\"; NV_SERVE_BACKEND=wgpu) via Gemma4MoeWgpu::from_gguf, \
         or serve the safetensors snapshot dir.",
        model_dir.display(),
        gguf.display()
    );
}

#[cfg(not(feature = "cuda"))]
pub(crate) fn refuse_gguf_checkpoint_dir(_model_dir: &Path) -> anyhow::Result<()> {
    Ok(())
}

#[cfg(feature = "cuda")]
pub(crate) fn plan_family_for(
    family: ModelFamily,
    qwen3_5_dense: bool,
) -> crate::oapi::vram_plan::PlanFamily {
    use crate::oapi::vram_plan::PlanFamily;
    match family {
        ModelFamily::Gemma4 => PlanFamily::Gemma4,
        ModelFamily::Qwen3_5Moe if qwen3_5_dense => PlanFamily::Qwen35Dense,
        ModelFamily::Qwen3_5Moe => PlanFamily::Qwen35Moe {
            grouped_expert_copy: true,
        },
        ModelFamily::Qwen3 => PlanFamily::Unmodeled("qwen3"),
        ModelFamily::Gemma4E4b => PlanFamily::Unmodeled("gemma4-e4b"),
        ModelFamily::Gemma4Moe => PlanFamily::Unmodeled("gemma4-moe"),
        ModelFamily::Laguna => PlanFamily::Laguna,
        ModelFamily::Omni => PlanFamily::Unmodeled("omni"),
        ModelFamily::Qwen4Exp => PlanFamily::Qwen4Exp,
        ModelFamily::GptOss => PlanFamily::Unmodeled("gpt-oss"),
    }
}

#[cfg(feature = "cuda")]
pub(crate) fn load_mm_towers_or_serve_text_only(
    model_dir: &Path,
    device: &candle_core::Device,
    model_id: &str,
) -> Option<Arc<crate::oapi::chat_multimodal::Gemma4MmTowers>> {
    let towers_dir = std::env::var("NV_MM_TOWERS_DIR")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|_| model_dir.to_path_buf());
    match crate::oapi::chat_multimodal::Gemma4MmTowers::from_model_dir(&towers_dir, device) {
        Ok(t) if t.vision.is_none() && t.audio.is_none() => {
            tracing::info!(
                model_id = %model_id,
                "no vision_config or audio_config in this checkpoint; serving text-only and \
                 routing image/audio chat parts to the perception bridge"
            );
            None
        }
        Ok(t) => {
            tracing::info!(
                model_id = %model_id,
                vision = t.vision.is_some(),
                audio = t.audio.is_some(),
                "gemma4 mm towers loaded; image_url and input_audio chat parts served"
            );
            Some(Arc::new(t))
        }
        Err(e) => {
            tracing::warn!(
                model_id = %model_id,
                error = format!("{e:#}"),
                "gemma4 mm towers failed to load; serving text-only"
            );
            None
        }
    }
}

impl NvEngineChat {
    pub fn try_load(model_dir: &Path) -> anyhow::Result<Self> {
        Self::preflight_checkpoint_dir(model_dir)?;
        Self::try_load_inner(model_dir, EngineBuildMode::Full)
    }

    #[cfg(feature = "cuda")]
    pub(crate) fn try_load_for_adapter_serving(model_dir: &Path) -> anyhow::Result<Self> {
        Self::preflight_checkpoint_dir(model_dir)?;
        Self::try_load_inner(model_dir, EngineBuildMode::AdapterServingLean)
    }

    fn preflight_checkpoint_dir(model_dir: &Path) -> anyhow::Result<()> {
        refuse_gguf_checkpoint_dir(model_dir)?;
        let required = [
            ("config.json", model_dir.join("config.json")),
            ("tokenizer.json", model_dir.join("tokenizer.json")),
        ];
        for (name, p) in &required {
            if !p.is_file() {
                anyhow::bail!("missing required file {name} in {}", model_dir.display());
            }
        }
        if !has_safetensors(model_dir) {
            anyhow::bail!(
                "missing model.safetensors / model.safetensors.index.json in {}",
                model_dir.display()
            );
        }
        Ok(())
    }

    #[cfg(feature = "cuda")]
    pub(crate) fn try_load_inner(model_dir: &Path, mode: EngineBuildMode) -> anyhow::Result<Self> {
        use anyhow::Context;
        use candle_core::Device;

        let cfg_path = model_dir.join("config.json");
        refuse_gguf_checkpoint_dir(model_dir)?;
        let raw_cfg = std::fs::read_to_string(&cfg_path)
            .with_context(|| format!("read {}", cfg_path.display()))?;
        let family = detect_family(&raw_cfg)
            .with_context(|| format!("detect model family from {}", cfg_path.display()))?;
        const ADAPTER_SERVING_LEAN_FAMILIES_WHOSE_EAGER_DECODE_ROUTES_EVERY_LORA_TARGET_THROUGH_NV_LAYERS_LINEAR:
            [ModelFamily; 2] = [ModelFamily::Gemma4, ModelFamily::Laguna];
        if mode == EngineBuildMode::AdapterServingLean
            && !ADAPTER_SERVING_LEAN_FAMILIES_WHOSE_EAGER_DECODE_ROUTES_EVERY_LORA_TARGET_THROUGH_NV_LAYERS_LINEAR
                .contains(&family)
        {
            anyhow::bail!(
                "cuda lora serving attaches nv-layers Linear delta hooks, which only the \
                 gemma4 dense family (the has_lora seams in nv_models::gemma4) and the laguna \
                 family (eager q/k/v/o/g_proj and dense-mlp Linears, with the w8 gemv \
                 disabled at install) route through; {} loads as family {:?}, whose \
                 projections are not hookable Linears. Serve this base on the wgpu engine to \
                 use its lora path.",
                model_dir.display(),
                family
            );
        }

        let chat_template = crate::oapi::chat_template::ChatTemplate::load(model_dir);
        if chat_template.is_some() {
            tracing::info!(model_dir = %model_dir.display(), "loaded official chat template");
        } else {
            tracing::warn!(model_dir = %model_dir.display(), "no official chat template found; using built-in renderer");
        }

        let qwen3_5_dense = qwen3_5_config_declares_dense_ffn(&raw_cfg);
        arm_the_serving_defaults_for_this_checkpoint(family, qwen3_5_dense, &raw_cfg)?;
        let qwen_graph = qwen_graph_decode_selected(family, &raw_cfg);
        let qwen36_mtp_captured = qwen36_mtp_captured_serving_selected(family, &raw_cfg);
        let (boot_plan, recorded_boot_plan) = crate::oapi::vram_plan::plan_and_enforce_boot_shaped(
            plan_family_for(family, qwen3_5_dense),
            model_dir,
            &raw_cfg,
            match mode {
                EngineBuildMode::Full => crate::oapi::vram_plan::BootPlanShape::Full,
                EngineBuildMode::AdapterServingLean => {
                    crate::oapi::vram_plan::BootPlanShape::AdapterServingLeanKeepsTheI8DecodeHeadNeverBuildsDrafters
                }
            },
        )?;
        let device = if capture_needs_the_device_stream(family, &raw_cfg) {
            Device::new_cuda_with_stream(0).context("init CUDA device 0 (stream mode)")?
        } else {
            Device::new_cuda(0).context("init CUDA device 0")?
        };

        if let Device::Cuda(d) = &device {
            let ctx = d.cuda_stream().context().clone();
            if ctx.is_event_tracking() {
                unsafe { ctx.disable_event_tracking() };
            }
        }
        let weights = nv_weights::WeightLoader::open_dir(model_dir, &device)
            .with_context(|| format!("open weights in {}", model_dir.display()))?;

        let tokenizer_path = model_dir.join("tokenizer.json");
        let mut tokenizer = tokenizers::Tokenizer::from_file(&tokenizer_path)
            .map_err(|e| anyhow::anyhow!("load tokenizer: {e}"))?;

        nv_tokenizer::sanitize_for_serving(&mut tokenizer);
        let tokenizer = tokenizer;

        let model_id = crate::oapi::model_ids::model_id_for_dir(model_dir);

        let built = match family {
            ModelFamily::Qwen3 => {
                let config = nv_models::qwen3::Qwen3Config::from_hf_json_str(&raw_cfg)
                    .with_context(|| format!("parse {}", cfg_path.display()))?;
                let qconfig = qwen3_weight_qconfig_default_nvfp4(&raw_cfg)
                    .with_context(|| format!("parse quant {}", cfg_path.display()))?;
                let model = nv_models::qwen3::Qwen3::from_loader_quantized(
                    config.clone(),
                    &weights,
                    &qconfig,
                    &device,
                )
                .context("instantiate Qwen3 from weights")?;
                let eos_token_ids = vec![config.eos_token_id];
                let bos_token_id = bos_id_from_dir(model_dir);
                let kv_max_seq_len =
                    kv_max_seq_len_for_unplanned_family("qwen3", config.max_position_embeddings);
                Ok(Self {
                    inner: NvEngineInner {
                        model_id,
                        family,
                        capture_shape: None,
                        model: LoadedModel::Qwen3(Arc::new(tokio::sync::Mutex::new(model))),
                        tokenizer: Arc::new(tokenizer),
                        device,
                        kv_max_seq_len,
                        default_max_new_tokens: 512,
                        eos_token_ids,
                        bos_token_id,
                        eagle3: None,
                        dflash: None,
                        spec_status: None,
                        chat_template,
                        gemma4_engine: None,
                        laguna_spec: None,
                        qwen_moe_dispatch: None,
                        qwen_mtp: None,
                        mm_towers: None,
                        vram_plan_record_released_on_engine_drop: None,
                    },
                })
            }
            ModelFamily::Gemma4 => {
                let config = nv_models::gemma4::Gemma4Config::from_hf_json_str(&raw_cfg)
                    .with_context(|| format!("parse gemma4 {}", cfg_path.display()))?;
                let qconfig = nv_weights::QuantizationConfig::from_hf_json_str(&raw_cfg)
                    .with_context(|| format!("parse quant {}", cfg_path.display()))?;
                let model = nv_models::gemma4::Gemma4::from_loader_quantized(
                    config.clone(),
                    &weights,
                    &qconfig,
                    &device,
                )
                .context("instantiate Gemma4 from weights")?;
                let eos_token_ids = eos_ids_from_dir(model_dir)?;
                let bos_token_id = bos_id_from_dir(model_dir);

                let kv_max_seq_len = boot_plan.kv_max_seq_len;
                let no_spec = nv_no_spec(std::env::var("NV_NO_SPEC").ok().as_deref());
                let drafter_kind = nv_drafter_kind(std::env::var("NV_DRAFTER").ok().as_deref());
                let target_dims = gemma4_target_dims(&model_id, &config);
                let td = Some(&target_dims);
                let (eagle3, dflash) = if mode == EngineBuildMode::AdapterServingLean {
                    tracing::info!(
                        "adapter-serving lean build: drafters are never built because \
                         spec verify chains and drafter KV would charge the vram plan for \
                         arms the adapter engine strips"
                    );
                    (None, None)
                } else if no_spec {
                    tracing::info!("NV_NO_SPEC set: skipping drafter load");
                    (None, None)
                } else if drafter_kind == "dflash" {
                    (
                        None,
                        load_dflash_state_for_target(&device, Some(model.embed_weight()), td),
                    )
                } else if drafter_kind == "auto" || drafter_kind == "route" {
                    (
                        load_eagle3_state_for_target(&device, Some(model.embed_weight()), td),
                        load_dflash_state_for_target(&device, Some(model.embed_weight()), td),
                    )
                } else {
                    (
                        load_eagle3_state_for_target(&device, Some(model.embed_weight()), td),
                        None,
                    )
                };
                let spec_requested = spec_requested(
                    no_spec,
                    env_flag_enabled(std::env::var("NV_USE_EAGLE3").ok().as_deref())
                        || drafter_kind == "dflash"
                        || drafter_kind == "auto"
                        || drafter_kind == "route",
                    std::env::var_os("NV_EAGLE3_DRAFT_DIR").is_some()
                        || std::env::var_os("NV_DFLASH_DRAFT_DIR").is_some(),
                );
                let required = eagle3_required(std::env::var("NV_EAGLE3_REQUIRED").ok().as_deref());
                let dflash_required_flag =
                    dflash_required(std::env::var("NV_DFLASH_REQUIRED").ok().as_deref());
                match eagle3_gate(
                    mode != EngineBuildMode::AdapterServingLean
                        && dflash_spec_requested(no_spec, drafter_kind),
                    dflash_required_flag,
                    dflash.is_some(),
                ) {
                    Eagle3Gate::NotRequested | Eagle3Gate::Enabled => {}
                    Eagle3Gate::DegradedWarn => {
                        tracing::error!(
                            "DFlash spec decode was requested (NV_DRAFTER={drafter_kind}) \
                             but the DFlash drafter did not load (see the preceding warn \
                             for the cause); serving will NOT use DFlash. Set \
                             NV_DFLASH_REQUIRED=1 to make this fatal instead, or \
                             NV_NO_SPEC=1 to silence this."
                        );
                    }
                    Eagle3Gate::RequiredFail => {
                        anyhow::bail!(
                            "NV_DFLASH_REQUIRED=1: DFlash spec decode was requested \
                             (NV_DRAFTER={drafter_kind}) but the drafter did not load \
                             (unset/bad NV_DFLASH_DRAFT_DIR -- the DFlash snapshot is not \
                             in the model hub, so the default lookup finds nothing -- or a \
                             load failure such as OOM; see the preceding warn); refusing \
                             to start a silently degraded server"
                        );
                    }
                }
                let drafter_loaded = eagle3.is_some() || dflash.is_some();
                let spec_status = match eagle3_gate(
                    mode != EngineBuildMode::AdapterServingLean && spec_requested,
                    required,
                    drafter_loaded,
                ) {
                    Eagle3Gate::NotRequested => None,
                    Eagle3Gate::Enabled => Some("on"),
                    Eagle3Gate::DegradedWarn => {
                        tracing::error!(
                            "Spec decode was requested (NV_DRAFTER / NV_USE_EAGLE3 / \
                             NV_EAGLE3_DRAFT_DIR / NV_DFLASH_DRAFT_DIR) but no drafter \
                             loaded (see the preceding warn for the cause); serving will be \
                             NON-SPECULATIVE (degraded, ~half throughput). Set \
                             NV_EAGLE3_REQUIRED=1 or NV_DFLASH_REQUIRED=1 to make this \
                             fatal instead, or NV_NO_SPEC=1 to silence this."
                        );
                        Some("degraded")
                    }
                    Eagle3Gate::RequiredFail => {
                        anyhow::bail!(
                            "NV_EAGLE3_REQUIRED=1: Eagle3 spec decode was requested \
                             (NV_USE_EAGLE3 or NV_EAGLE3_DRAFT_DIR set) but the drafter \
                             did not load (unset/bad NV_EAGLE3_DRAFT_DIR or a \
                             load failure such as OOM -- see the preceding warn); \
                             refusing to start a silently degraded non-speculative server"
                        );
                    }
                };
                let eagle3_row_elems = eagle3
                    .as_ref()
                    .map(|sh| sh.proposer.scorer().config().kv_out_dim())
                    .unwrap_or(0);
                let dflash_row_elems = dflash
                    .as_ref()
                    .map(|sh| {
                        let c = sh.drafter.config();
                        c.num_hidden_layers * c.kv_out_dim()
                    })
                    .unwrap_or(0);
                let drafter_row_elems =
                    drafter_row_elems_charge(eagle3_row_elems, dflash_row_elems);
                let model = Arc::new(model);

                let gemma4_engine = if mode == EngineBuildMode::AdapterServingLean {
                    None
                } else if let Some(pool) = boot_plan.gemma4_batch_pool {
                    match build_gemma4_batch_engine(
                        model.clone(),
                        device.clone(),
                        kv_max_seq_len,
                        pool,
                    ) {
                        Ok(h) => {
                            tracing::info!(
                                "NV_BATCH_ENGINE=1: continuous-batching engine active for Gemma4"
                            );
                            Some(Arc::new(h))
                        }
                        Err(e) => {
                            tracing::warn!(error = %e, "failed to build Gemma4 batch engine; falling back to per-request path");
                            None
                        }
                    }
                } else {
                    None
                };

                let mm_towers = load_mm_towers_or_serve_text_only(model_dir, &device, &model_id);
                if mode != EngineBuildMode::AdapterServingLean {
                    let static_measured_after_the_batch_pool_and_the_mm_towers =
                        our_process_vram_bytes();
                    crate::oapi::admission::init_gemma4(
                        static_measured_after_the_batch_pool_and_the_mm_towers,
                        drafter_row_elems,
                    );
                }

                Ok(Self {
                    inner: NvEngineInner {
                        model_id,
                        family,
                        capture_shape: None,
                        model: LoadedModel::Gemma4(model),
                        tokenizer: Arc::new(tokenizer),
                        device,
                        kv_max_seq_len,
                        default_max_new_tokens: 512,
                        eos_token_ids,
                        bos_token_id,
                        eagle3,
                        dflash,
                        spec_status,
                        chat_template,
                        gemma4_engine,
                        laguna_spec: None,
                        qwen_moe_dispatch: None,
                        qwen_mtp: None,
                        mm_towers,
                        vram_plan_record_released_on_engine_drop: None,
                    },
                })
            }
            ModelFamily::Gemma4E4b => {
                if std::env::var("NV_E4B_CUDA_SERVE").as_deref() != Ok("1") {
                    anyhow::bail!(
                        "gemma-4 E4B cuda serving is opt-in: set NV_E4B_CUDA_SERVE=1 to serve this \
                         checkpoint on cuda (the backend selector demands the same value), or serve \
                         it on the wgpu backend"
                    );
                }
                let config = nv_models::gemma4::Gemma4Config::from_hf_json_str(&raw_cfg)
                    .with_context(|| format!("parse gemma4 e4b {}", cfg_path.display()))?;
                anyhow::ensure!(
                    config.has_per_layer_embeddings(),
                    "detect_family routed to Gemma4E4b but the config has no per-layer-embedding stack"
                );
                let model = nv_models::gemma4_e4b::Gemma4E4b::from_loader(
                    config.clone(),
                    &weights,
                    &device,
                )
                .context("instantiate Gemma4E4b from weights")?;
                let eos_token_ids = eos_ids_from_dir(model_dir)?;
                let bos_token_id = bos_id_from_dir(model_dir);
                let kv_max_seq_len = kv_max_seq_len_for_gemma4(config.max_position_embeddings);
                let mm_towers = load_mm_towers_or_serve_text_only(model_dir, &device, &model_id);

                Ok(Self {
                    inner: NvEngineInner {
                        model_id,
                        family,
                        capture_shape: None,
                        model: LoadedModel::Gemma4E4b(Arc::new(model)),
                        tokenizer: Arc::new(tokenizer),
                        device,
                        kv_max_seq_len,
                        default_max_new_tokens: 512,
                        eos_token_ids,
                        bos_token_id,
                        eagle3: None,
                        dflash: None,
                        spec_status: None,
                        chat_template,
                        gemma4_engine: None,
                        laguna_spec: None,
                        qwen_moe_dispatch: None,
                        qwen_mtp: None,
                        mm_towers,
                        vram_plan_record_released_on_engine_drop: None,
                    },
                })
            }
            ModelFamily::Gemma4Moe => {
                let config = nv_models::gemma4_moe::Gemma4MoeConfig::from_hf_json_str(&raw_cfg)
                    .with_context(|| format!("parse gemma4 moe {}", cfg_path.display()))?;
                let model = match nv_models::gemma4_gguf::opted_in_qat_q4_gguf_for_snapshot(&config)
                {
                    Some(gguf) => {
                        tracing::info!(
                            weights = %gguf.display(),
                            tokenizer_and_template = %model_dir.display(),
                            opt_in_env = nv_models::gemma4_gguf::G4MOE_SERVE_QAT_Q4_OPT_IN_ENV,
                            why_not_default = nv_models::gemma4_gguf::QAT_Q4_GGUF_STAYS_OPT_IN_TOP1_ACC_0_4531_VS_BF16_0_5254_SAME_SESSION,
                            "gemma4-moe serves the 4-bit QAT GGUF weight values because the \
                             opt-in env is set; this from_gguf path keeps bf16-stacked expert \
                             residency, because requantizing the Q4_K dequant to nvfp4 would \
                             stack a third quantization on the double one that already put \
                             this arm out of band"
                        );
                        nv_models::gemma4_moe::Gemma4Moe::from_gguf(
                            &gguf,
                            &device,
                            candle_core::DType::BF16,
                        )
                        .with_context(|| {
                            format!(
                                "instantiate Gemma4Moe from default qat gguf {}",
                                gguf.display()
                            )
                        })?
                    }
                    None => nv_models::gemma4_moe::Gemma4Moe::from_loader(
                        config.clone(),
                        &weights,
                        &device,
                    )
                    .context("instantiate Gemma4Moe from weights")?,
                };
                let eos_token_ids = eos_ids_from_dir(model_dir)?;
                let bos_token_id = bos_id_from_dir(model_dir);
                let kv_max_seq_len = kv_max_seq_len_for_unplanned_family(
                    "gemma4-moe",
                    config.base.max_position_embeddings,
                );
                let mm_towers = load_mm_towers_or_serve_text_only(model_dir, &device, &model_id);
                Ok(Self {
                    inner: NvEngineInner {
                        model_id,
                        family,
                        capture_shape: None,
                        model: LoadedModel::Gemma4Moe(Arc::new(model)),
                        tokenizer: Arc::new(tokenizer),
                        device,
                        kv_max_seq_len,
                        default_max_new_tokens: 512,
                        eos_token_ids,
                        bos_token_id,
                        eagle3: None,
                        dflash: None,
                        spec_status: None,
                        chat_template,
                        gemma4_engine: None,
                        laguna_spec: None,
                        qwen_moe_dispatch: None,
                        qwen_mtp: None,
                        mm_towers,
                        vram_plan_record_released_on_engine_drop: None,
                    },
                })
            }
            ModelFamily::Qwen3_5Moe if qwen3_5_dense => {
                let config = nv_models::qwen3_5_moe::Qwen3_5DenseConfig::from_hf_json_str(&raw_cfg)
                    .with_context(|| format!("parse qwen3.5 dense {}", cfg_path.display()))?;
                let qconfig = nv_weights::QuantizationConfig::from_hf_json_str(&raw_cfg)
                    .with_context(|| format!("parse quant {}", cfg_path.display()))?;
                let trunk = config.trunk();
                let n_linear = trunk
                    .layer_types
                    .iter()
                    .filter(|t| **t == nv_models::qwen3_5_moe::LayerType::LinearAttention)
                    .count();
                eprintln!(
                    "[qwen3.5-dense] {QWEN35_DENSE_CUDA_OPT_IN_MATCHES_THE_WGPU_ORACLE_AND_STAYS_OPT_IN}"
                );
                eprintln!(
                    "[qwen3.5-dense] trunk: {n_linear}/{} layers are nv_layers::linear_attn::LinearAttention, dense ffn intermediate_size={}",
                    trunk.num_hidden_layers, config.intermediate_size
                );
                let model = nv_models::qwen3_5_moe::Qwen3Moe::from_loader_dense_quantized(
                    config.clone(),
                    &weights,
                    &qconfig,
                    &device,
                )
                .context("instantiate Qwen3.5-dense from weights")?;
                anyhow::ensure!(
                    model.dense_intermediate() == Some(config.intermediate_size),
                    "from_loader_dense_quantized did not carry intermediate_size {} into the model \
                     (got {:?}), so the per-layer ffn is not the dense Mlp this arm loaded weights \
                     for and every layer would route through a MoeBlock with zero experts",
                    config.intermediate_size,
                    model.dense_intermediate()
                );
                let eos_token_ids = resolve_qwen3_5_moe_eos_ids(&tokenizer, model_dir, &trunk);
                let bos_token_id = bos_id_from_dir(model_dir);
                let kv_max_seq_len = boot_plan.kv_max_seq_len;
                let qwen_mtp = if nv_specdecode::qwen38_mtp::mtp_drafter_selected_from_env() {
                    let head = nv_specdecode::qwen38_mtp::Qwen38DenseMtpHead::from_checkpoint(
                        nv_specdecode::qwen38_mtp::mtp_draft_dir_override_from_env().as_deref(),
                        model_dir,
                        &model,
                        &device,
                    )
                    .context(
                        "NV_DRAFTER=mtp was requested, so a missing or misshapen MTP head is a \
                         boot failure rather than a silent fall-back to non-speculative decode",
                    )?;
                    let k = nv_specdecode::qwen38_mtp::mtp_chain_depth_from_env();
                    eprintln!(
                        "[qwen3.8-mtp] drafter: shipped MTP head loaded (k={k}, greedy requests \
                         self-speculate; sampled requests decode normally)"
                    );
                    Some(Arc::new(head))
                } else {
                    None
                };
                let tokenizer = Arc::new(tokenizer);
                let shared = if nv_models::qwen3_5_moe::qwen38_batch::nv_q38_batch_env_opt_in_nv_q38_batch_1_the_serving_loop_routes_batch_xor_spec_per_request_group() {
                    let plan = boot_plan
                        .batch_bucket_sizes
                        .clone()
                        .map(nv_models::gemma4_batch_graph::BucketPlan::new)
                        .unwrap_or_else(
                            nv_models::qwen3_5_moe::qwen38_batch::q38_batch_bucket_plan_env_nv_q38_batch_sizes,
                        );
                    let lane_kv = boot_plan.batch_lane_kv.unwrap_or(kv_max_seq_len);
                    let lanes = nv_models::qwen3_5_moe::qwen38_batch::Qwen38BatchLanes::new(
                        model,
                        &device,
                        lane_kv,
                        plan,
                    )
                    .context(
                        "NV_Q38_BATCH=1 was requested, so a lane pool that cannot build is a \
                         boot failure rather than a silent fall-back to solo serving",
                    )?;
                    eprintln!(
                        "[qwen3.8-batch] request-group scheduler on: {} lanes x {} kv slots \
                         ({} caps lane depth below the solo kv window {}), window {} ms",
                        lanes.lanes(),
                        lane_kv,
                        crate::oapi::vram_plan::Q38_BATCH_KV_ENV_CAPS_LANE_POOL_DEPTH_SO_PREPAID_BATCH_LANES_STAY_SHALLOW_WHILE_SOLO_REQUESTS_GO_DEEP,
                        kv_max_seq_len,
                        q38_batch_window_ms_env_nv_q38_batch_window_ms()
                    );
                    let sched = QwenBatchScheduler::spawn(
                        QwenLanePool::Dense38(lanes),
                        qwen_mtp.clone().map(QwenMtpHead::Dense),
                        tokenizer.clone(),
                        device.clone(),
                        eos_token_ids.clone(),
                        kv_max_seq_len,
                        256,
                    )?;
                    QwenMoeShared::Batch(sched)
                } else {
                    let mut graphed = nv_models::graph_engine::GraphedQwen3Moe::new(
                        model,
                        &device,
                        kv_max_seq_len,
                    )
                    .context(
                        "the dense qwen3.8 solo engine is the graphed one, so a graphed engine \
                         that cannot build is a boot failure rather than a silent fall-back",
                    )?;
                    graphed.install_grouped_moe().context(
                        "arm capture on the dense trunk (no dispatch is built; every ffn is a \
                         plain mlp)",
                    )?;
                    eprintln!(
                        "[qwen3.8] decode: GRAPHED (persistent GraphedQwen3Moe, kv cache bound \
                         once at boot to the {kv_max_seq_len}-token window; greedy MTP requests \
                         self-speculate through Qwen38MtpGraphedDecodeSession, whose rows-keyed \
                         verify graph is recaptured after each request's prefill)"
                    );
                    QwenMoeShared::Graphed(Arc::new(tokio::sync::Mutex::new(graphed)))
                };
                Ok(Self {
                    inner: NvEngineInner {
                        model_id,
                        family,
                        capture_shape: Some((config.num_hidden_layers, config.hidden_size)),
                        model: LoadedModel::Qwen3_5Moe(shared),
                        tokenizer,
                        device,
                        kv_max_seq_len,
                        default_max_new_tokens: 256,
                        eos_token_ids,
                        bos_token_id,
                        eagle3: None,
                        dflash: None,
                        spec_status: None,
                        chat_template,
                        gemma4_engine: None,
                        laguna_spec: None,
                        qwen_moe_dispatch: None,
                        qwen_mtp: qwen_mtp.map(QwenMtpHead::Dense),
                        mm_towers: None,
                        vram_plan_record_released_on_engine_drop: None,
                    },
                })
            }
            ModelFamily::Qwen3_5Moe => {
                let config = nv_models::qwen3_5_moe::Qwen3MoeConfig::from_hf_json_str(&raw_cfg)
                    .with_context(|| format!("parse qwen3_5_moe {}", cfg_path.display()))?;
                let qconfig = nv_weights::QuantizationConfig::from_hf_json_str(&raw_cfg)
                    .with_context(|| format!("parse quant {}", cfg_path.display()))?;
                let model = nv_models::qwen3_5_moe::Qwen3Moe::from_loader_quantized(
                    config.clone(),
                    &weights,
                    &qconfig,
                    &device,
                )
                .context("instantiate Qwen3.5-MoE from weights")?;
                let eos_token_ids = resolve_qwen3_5_moe_eos_ids(&tokenizer, model_dir, &config);
                let bos_token_id = bos_id_from_dir(model_dir);
                let kv_max_seq_len = boot_plan.kv_max_seq_len;
                let qwen_mtp = if nv_specdecode::qwen38_mtp::mtp_drafter_selected_from_env() {
                    let mtp_path = nv_specdecode::qwen38_mtp::mtp_draft_dir_override_from_env()
                        .unwrap_or_else(|| model_dir.to_path_buf())
                        .join(nv_specdecode::qwen38_mtp::MTP_WEIGHTS_FILE_NAME);
                    let head = nv_models::qwen3_5_mtp::MtpHead::from_safetensors(
                        &mtp_path, &model, &device,
                    )
                    .with_context(|| {
                        format!(
                            "NV_DRAFTER=mtp was requested, so a missing or misshapen MoE MTP \
                             head ({}) is a boot failure rather than a silent fall-back to \
                             non-speculative decode",
                            mtp_path.display()
                        )
                    })?;
                    let k = nv_specdecode::qwen38_mtp::mtp_chain_depth_from_env();
                    eprintln!(
                        "[qwen3.6-mtp] drafter: shipped MoE MTP head loaded (k={k}, greedy \
                         requests self-speculate through the CAPTURED verify-lane session; \
                         sampled requests decode normally)"
                    );
                    Some(QwenMtpHead::Moe(Arc::new(head)))
                } else {
                    None
                };
                let tokenizer = Arc::new(tokenizer);
                let qwen36_batch = qwen36_batch_lanes_boot_selected(family, &raw_cfg);
                let (shared, qwen_moe_dispatch) = if qwen36_batch {
                    anyhow::ensure!(
                        !qwen_graph,
                        "NV_Q36_BATCH=1 boots the lane-pool scheduler, which owns the model, but \
                         NV_QWEN_GRAPH=1 pins the graphed solo engine, which also owns it -- \
                         drop one of the two"
                    );
                    let plan = boot_plan
                        .batch_bucket_sizes
                        .clone()
                        .map(nv_models::gemma4_batch_graph::BucketPlan::new)
                        .unwrap_or_else(
                            nv_models::qwen3_5_moe::qwen36_batch::q36_batch_bucket_plan_env_nv_q36_batch_sizes,
                        );
                    let lane_kv = boot_plan.batch_lane_kv.unwrap_or(kv_max_seq_len);
                    let lanes = nv_models::qwen3_5_moe::qwen36_batch::Qwen36BatchLanes::new(
                        model, &device, lane_kv, plan,
                    )
                    .context(
                        "NV_Q36_BATCH=1 was requested, so a lane pool that cannot build is a \
                         boot failure rather than a silent fall-back to solo serving",
                    )?;
                    eprintln!(
                        "[qwen3.6-batch] request-group scheduler on: {} lanes x {} kv slots \
                         ({} caps lane depth below the solo kv window {}), window {} ms; solo \
                         greedy requests {} (batch XOR spec)",
                        lanes.lanes(),
                        lane_kv,
                        crate::oapi::vram_plan::Q36_BATCH_KV_ENV_CAPS_LANE_POOL_DEPTH_SO_PREPAID_BATCH_LANES_STAY_SHALLOW_WHILE_SOLO_REQUESTS_GO_DEEP,
                        kv_max_seq_len,
                        q36_batch_window_ms_env_nv_q36_batch_window_ms(),
                        if qwen_mtp.is_some() {
                            "self-speculate through the EAGER MoE MTP session off the lane pool's model (the captured verify lane needs the graphed engine this boot did not build)"
                        } else {
                            "decode through the eager solo body with the lane pool's installed MoE hook"
                        }
                    );
                    let sched = QwenBatchScheduler::spawn(
                        QwenLanePool::Moe36(lanes),
                        qwen_mtp.clone(),
                        tokenizer.clone(),
                        device.clone(),
                        eos_token_ids.clone(),
                        kv_max_seq_len,
                        256,
                    )?;
                    (QwenMoeShared::Batch(sched), None)
                } else if qwen_graph || qwen36_mtp_captured {
                    let mut graphed = nv_models::graph_engine::GraphedQwen3Moe::new(
                        model,
                        &device,
                        kv_max_seq_len,
                    )
                    .context("build GraphedQwen3Moe decode engine")?;
                    let free_before = nv_layers::cudarc::driver::result::mem_get_info()
                        .ok()
                        .map(|(free, _total)| free);
                    match graphed.install_grouped_moe() {
                        Ok(()) => {
                            let delta_mib = free_before
                                .zip(
                                    nv_layers::cudarc::driver::result::mem_get_info()
                                        .ok()
                                        .map(|(free, _total)| free),
                                )
                                .map(|(before, after)| {
                                    before.saturating_sub(after) as f64 / (1024.0 * 1024.0)
                                });
                            eprintln!(
                                "[qwen3.6-moe] decode: GRAPH (grouped device MoE + CUDA graph capture; {}), kv window {kv_max_seq_len}, grouped expert copy +{} MiB VRAM",
                                if qwen_graph {
                                    "NV_QWEN_GRAPH=0 for eager"
                                } else {
                                    "selected by NV_DRAFTER=mtp captured serving"
                                },
                                delta_mib.map(|m| format!("{m:.0}")).unwrap_or_else(|| "?".into())
                            );
                        }
                        Err(e) => {
                            eprintln!(
                                "[qwen3.6-moe] decode: GRAPH-DEGRADED -- grouped MoE install FAILED, decode will run uncaptured host-routed: {e:#}"
                            );
                        }
                    }
                    (
                        QwenMoeShared::Graphed(Arc::new(tokio::sync::Mutex::new(graphed))),
                        None,
                    )
                } else {
                    eprintln!(
                        "[qwen3.6-moe] decode: EAGER (NV_QWEN_GRAPH={:?}; graph decode needs it unset/on)",
                        std::env::var("NV_QWEN_GRAPH").ok()
                    );
                    let dispatch = build_qwen_moe_dispatch(&model);
                    (
                        QwenMoeShared::Eager(Arc::new(tokio::sync::Mutex::new(model))),
                        dispatch,
                    )
                };
                Ok(Self {
                    inner: NvEngineInner {
                        model_id,
                        family,
                        capture_shape: Some((config.num_hidden_layers, config.hidden_size)),
                        model: LoadedModel::Qwen3_5Moe(shared),
                        tokenizer,
                        device,
                        kv_max_seq_len,
                        default_max_new_tokens: 256,
                        eos_token_ids,
                        bos_token_id,
                        eagle3: None,
                        dflash: None,
                        spec_status: None,
                        chat_template,
                        gemma4_engine: None,
                        laguna_spec: None,
                        qwen_moe_dispatch,
                        qwen_mtp,
                        mm_towers: None,
                        vram_plan_record_released_on_engine_drop: None,
                    },
                })
            }
            ModelFamily::Laguna => {
                if nv_models::laguna_serve::bf16_prefill_handoff_opt_in() {
                    anyhow::ensure!(mode != EngineBuildMode::AdapterServingLean,
                        "NV_LAGUNA_PREFILL_BF16_HANDOFF=1 requires full Laguna serving");
                }
                if nv_models::laguna_serve::batch_graph_enabled_from_env()? {
                    anyhow::ensure!(mode != EngineBuildMode::AdapterServingLean, "NV_LAGUNA_BATCH_GRAPH=1, m2 or m2-spec requires the full Laguna serving engine; adapter-serving lean mode is incompatible");
                }
                let require_nvfp4 = nv_models::laguna_dflash::dflash_nvfp4_enabled()?;
                validate_laguna_packed_draft_arm(require_nvfp4, mode, laguna_serve_draft_enabled())?;
                let config = nv_models::laguna::LagunaConfig::from_hf_json_str(&raw_cfg)
                    .with_context(|| format!("parse laguna {}", cfg_path.display()))?;
                let qconfig = nv_weights::QuantizationConfig::from_hf_json_str(&raw_cfg)
                    .with_context(|| format!("parse quant {}", cfg_path.display()))?;
                let model = nv_models::laguna::Laguna::from_loader_quantized(
                    config.clone(),
                    &weights,
                    &qconfig,
                    &device,
                )
                .context("instantiate Laguna from weights")?;
                let eos_token_ids = if config.eos_token_id.is_empty() {
                    eos_ids_from_dir(model_dir)?
                } else {
                    config.eos_token_id.clone()
                };
                let bos_token_id = bos_id_from_dir(model_dir);
                let kv_max_seq_len =
                    kv_max_seq_len_for_unplanned_family("laguna", config.max_position_embeddings);
                let shared = LagunaShared(Arc::new(model), Arc::new(LagunaAdmission::from_env()));
                let (laguna_spec, spec_status) =
                    laguna_engine_spec_state(if mode == EngineBuildMode::AdapterServingLean {
                        tracing::info!(
                            "adapter-serving lean build: the laguna spec-serve engine is never \
                         built because its captured step-graph body (raw nv-* kernels plus \
                         candle proj fallbacks captured on a forked stream) never routes the \
                         nv-layers Linear hooks a resident adapter needs; adapter requests \
                         decode on the eager per-request path"
                        );
                        None
                    } else {
                        build_laguna_spec_serve(&shared, kv_max_seq_len, require_nvfp4)?
                    });
                Ok(Self {
                    inner: NvEngineInner {
                        model_id,
                        family,
                        capture_shape: None,
                        model: LoadedModel::Laguna(shared),
                        tokenizer: Arc::new(tokenizer),
                        device,
                        kv_max_seq_len,
                        default_max_new_tokens: 512,
                        eos_token_ids,
                        bos_token_id,
                        eagle3: None,
                        dflash: None,
                        spec_status,
                        chat_template,
                        gemma4_engine: None,
                        laguna_spec,
                        qwen_moe_dispatch: None,
                        qwen_mtp: None,
                        mm_towers: None,
                        vram_plan_record_released_on_engine_drop: None,
                    },
                })
            }
            ModelFamily::GptOss => {
                let config = nv_models::gpt_oss::GptOssConfig::from_hf_json_str(&raw_cfg)
                    .with_context(|| format!("parse gpt_oss {}", cfg_path.display()))?;
                eprintln!(
                    "[gpt-oss] {GPTOSS_CUDA_OPT_IN_IS_A_DEQUANT_RUNG_AND_WGPU_STAYS_THE_GPT_OSS_DEFAULT}"
                );
                let sliding = config
                    .layer_types
                    .iter()
                    .filter(|t| **t == nv_models::gpt_oss::GptOssLayerType::Sliding)
                    .count();
                eprintln!(
                    "[gpt-oss] decode: EAGER sink attention (no flash, no graph capture), {sliding}/{} layers slide at window {}, {} experts top-{}, mxfp4 dequantized to bf16 at load",
                    config.num_hidden_layers,
                    config.sliding_window,
                    config.num_local_experts,
                    config.num_experts_per_tok
                );
                let model = nv_models::gpt_oss_cuda::GptOssCuda::from_loader(
                    config.clone(),
                    &weights,
                    &device,
                )
                .context("instantiate GptOss from weights")?;
                let eos_token_ids = eos_ids_from_dir(model_dir)?;
                let bos_token_id = bos_id_from_dir(model_dir);
                let kv_max_seq_len =
                    kv_max_seq_len_for_unplanned_family("gpt-oss", config.max_position_embeddings);
                Ok(Self {
                    inner: NvEngineInner {
                        model_id,
                        family,
                        capture_shape: None,
                        model: LoadedModel::GptOss(Arc::new(model)),
                        tokenizer: Arc::new(tokenizer),
                        device,
                        kv_max_seq_len,
                        default_max_new_tokens: 512,
                        eos_token_ids,
                        bos_token_id,
                        eagle3: None,
                        dflash: None,
                        spec_status: None,
                        chat_template,
                        gemma4_engine: None,
                        laguna_spec: None,
                        qwen_moe_dispatch: None,
                        qwen_mtp: None,
                        mm_towers: None,
                        vram_plan_record_released_on_engine_drop: None,
                    },
                })
            }
            ModelFamily::Qwen4Exp => {
                let config = nv_models::qwen4_exp::Qwen4ExpConfig::from_hf_json_str(&raw_cfg)
                    .with_context(|| format!("parse qwen4_exp {}", cfg_path.display()))?;
                eprintln!(
                    "[qwen4_exp] {} layers ({} GDN + {} QSA), {} experts top-{}, hc_count {}, PLE at {:?}; dense attention up to {} tokens, sparse QSA beyond",
                    config.num_hidden_layers,
                    config.full_attention.iter().filter(|f| !**f).count(),
                    config.full_attention.iter().filter(|f| **f).count(),
                    config.num_experts,
                    config.num_experts_per_tok,
                    config.hc_count,
                    config.ple_layer_ids_zero_based,
                    config.dense_attention_context_limit()
                );
                let max_pos = config.max_position_embeddings;
                let model = nv_models::qwen4_exp::Qwen4Exp::from_loader(config, weights, &device)
                    .context("instantiate Qwen3.8-Flash-Next (qwen4_exp) from weights")?;
                model.set_weights_tag(nv_models::qwen4_exp::weights_tag_for_dir(model_dir));
                match nv_models::qwen4_exp::lora::expert::base_identity_for_dir(model_dir) {
                    Ok(id) => model.set_base_identity(id),
                    Err(e) => tracing::warn!(error = %format!("{e:#}"), "qwen4_exp base identity unreadable; adapters load without the trained-against check"),
                }
                model
                    .configure_session_residency(nv_models::qwen4_exp::SessionResidency::from_env())
                    .context("qwen4_exp long-session residency (NV_Q4EXP_SESSION_*, NV_KV_CACHE_DIR)")?;
                let eos_token_ids = eos_ids_from_dir(model_dir)?;
                let bos_token_id = bos_id_from_dir(model_dir);
                let kv_max_seq_len = kv_max_seq_len_for_unplanned_family("qwen4_exp", max_pos);
                let model = Arc::new(model);
                if let Err(e) = super::qwen4_exp_mm::register_vision(&model, model_dir, &raw_cfg, &tokenizer, &device) {
                    tracing::warn!(error = %format!("{e:#}"), "qwen4_exp native vision unavailable; image parts go through the OCR bridge");
                }
                Ok(Self {
                    inner: NvEngineInner {
                        model_id,
                        family,
                        capture_shape: None,
                        model: LoadedModel::Qwen4Exp(model),
                        tokenizer: Arc::new(tokenizer),
                        device,
                        kv_max_seq_len,
                        default_max_new_tokens: 512,
                        eos_token_ids,
                        bos_token_id,
                        eagle3: None,
                        dflash: None,
                        spec_status: None,
                        chat_template,
                        gemma4_engine: None,
                        laguna_spec: None,
                        qwen_moe_dispatch: None,
                        qwen_mtp: None,
                        mm_towers: None,
                        vram_plan_record_released_on_engine_drop: None,
                    },
                })
            }
            ModelFamily::Omni => {
                let shared = omni_loop::build_omni(model_dir, &device)
                    .context("instantiate Qwen3-Omni thinker/vision/audio from weights")?;
                let text_config = nv_omni::OmniThinkerConfig::from_hf_config_json(&cfg_path)
                    .with_context(|| format!("parse omni thinker config {}", cfg_path.display()))?;
                let eos_token_ids = eos_ids_from_dir(model_dir)?;
                let bos_token_id = bos_id_from_dir(model_dir);
                let kv_max_seq_len = kv_max_seq_len_for_unplanned_family(
                    "omni",
                    text_config.max_position_embeddings,
                );
                Ok(Self {
                    inner: NvEngineInner {
                        model_id,
                        family,
                        capture_shape: None,
                        model: LoadedModel::Omni(shared),
                        tokenizer: Arc::new(tokenizer),
                        device,
                        kv_max_seq_len,
                        default_max_new_tokens: 512,
                        eos_token_ids,
                        bos_token_id,
                        eagle3: None,
                        dflash: None,
                        spec_status: None,
                        chat_template,
                        gemma4_engine: None,
                        laguna_spec: None,
                        qwen_moe_dispatch: None,
                        qwen_mtp: None,
                        mm_towers: None,
                        vram_plan_record_released_on_engine_drop: None,
                    },
                })
            }
        };
        built.map(|mut engine| {
            engine.inner.vram_plan_record_released_on_engine_drop = Some(recorded_boot_plan);
            engine
        })
    }

    #[cfg(not(feature = "cuda"))]
    pub(crate) fn try_load_inner(
        _model_dir: &Path,
        _mode: EngineBuildMode,
    ) -> anyhow::Result<Self> {
        anyhow::bail!(
            "NvEngineChat::try_load requires the `cuda` cargo feature; \
             rebuild with --features cuda or unset NV_CHAT_MODEL_DIR"
        )
    }
}

#[cfg(any(feature = "cuda", test))]
pub(crate) fn spec_serve_max_seq(env: Option<usize>, kv_max_seq_len: usize) -> usize {
    match env {
        Some(n) if n > 0 => n.min(kv_max_seq_len),
        _ => kv_max_seq_len,
    }
}

#[cfg(any(feature = "cuda", test))]
pub(crate) const KV_DEFAULT_8192_TOKENS_FOR_FAMILIES_WITHOUT_A_VRAM_SOLVER_MODEL_SO_MULTI_ENGINE_BOOTS_DO_NOT_OOM_LIFT_VIA_THE_GLOBAL_OR_PER_FAMILY_KV_ENV: usize = 8192;

#[cfg(any(feature = "cuda", test))]
pub(crate) fn kv_max_seq_len_resolved(
    env_bound: Option<usize>,
    max_position_embeddings: usize,
) -> usize {
    match env_bound {
        Some(n) if n > 0 => n.min(max_position_embeddings.max(1)),
        _ => max_position_embeddings
            .min(KV_DEFAULT_8192_TOKENS_FOR_FAMILIES_WITHOUT_A_VRAM_SOLVER_MODEL_SO_MULTI_ENGINE_BOOTS_DO_NOT_OOM_LIFT_VIA_THE_GLOBAL_OR_PER_FAMILY_KV_ENV),
    }
}

#[cfg(feature = "cuda")]
pub(crate) fn kv_max_seq_len_for_unplanned_family(
    family: &'static str,
    max_position_embeddings: usize,
) -> usize {
    let per_env = crate::oapi::vram_plan::per_engine_kv_env_for_family(family);
    let bound = crate::oapi::vram_plan::kv_binding_from(
        per_env,
        per_env.and_then(|e| std::env::var(e).ok()).as_deref(),
        std::env::var(
            crate::oapi::vram_plan::KV_MAX_SEQ_LEN_GLOBAL_ENV_THE_DEFAULT_EVERY_ENGINE_INHERITS,
        )
        .ok()
        .as_deref(),
    );
    if let Some((n, env)) = bound {
        tracing::info!(family, kv_max_seq_len = n, env, "kv ceiling from env");
    }
    kv_max_seq_len_resolved(bound.map(|(n, _)| n), max_position_embeddings)
}

#[cfg(feature = "cuda")]
pub(crate) fn kv_max_seq_len_for_gemma4(max_position_embeddings: usize) -> usize {
    match std::env::var("NV_KV_MAX_SEQ_LEN")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
    {
        Some(n) if n > 0 => n.min(max_position_embeddings.max(1)),
        _ => max_position_embeddings.max(1),
    }
}

#[cfg(feature = "cuda")]
pub(crate) fn our_process_vram_bytes() -> Option<u64> {
    crate::oapi::vram_plan::process_vram_bytes_via_nvidia_smi()
}

#[cfg(feature = "cuda")]
pub(crate) fn drafter_kv_cap_env() -> Option<(usize, usize)> {
    let p = |name: &str| {
        std::env::var(name)
            .ok()
            .and_then(|v| v.trim().parse::<usize>().ok())
    };
    nv_specdecode::chain::drafter_kv_cap_from_env(
        p("NV_DRAFTER_KV_WINDOW"),
        p("NV_DRAFTER_KV_SINK"),
        nv_specdecode::eagle3_loader::DRAFTER_KV_CAP_DEFAULT_SINK,
    )
}

#[cfg(feature = "cuda")]
pub(crate) fn capped_drafter_kv_rows(kv_max: usize, cap: Option<(usize, usize)>) -> usize {
    nv_specdecode::chain::effective_drafter_kv_rows(
        kv_max,
        cap,
        nv_specdecode::eagle3_loader::DRAFTER_KV_CAP_SLACK,
    )
}

#[cfg(feature = "cuda")]
#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct DrafterKvCharge {
    pub(crate) eagle3_row_elems: usize,
    pub(crate) dflash_row_elems: usize,
    pub(crate) eagle3_cap: Option<(usize, usize)>,
}

#[cfg(feature = "cuda")]
impl DrafterKvCharge {
    pub(crate) fn from_env(eagle3_row_elems: usize, dflash_row_elems: usize) -> Self {
        Self {
            eagle3_row_elems,
            dflash_row_elems,
            eagle3_cap: drafter_kv_cap_env(),
        }
    }

    pub(crate) fn rows_elems(&self, kv_max: usize) -> (usize, usize) {
        let e3 = (
            capped_drafter_kv_rows(kv_max, self.eagle3_cap),
            self.eagle3_row_elems,
        );
        let df = (kv_max, self.dflash_row_elems);
        if e3.0 * e3.1 >= df.0 * df.1 {
            e3
        } else {
            df
        }
    }
}

#[cfg(feature = "cuda")]
pub(crate) fn detect_family(raw_cfg: &str) -> anyhow::Result<ModelFamily> {
    detect_family_with_dense_cuda_serve(raw_cfg, qwen3_5_dense_cuda_serve_enabled())
}

#[cfg(feature = "cuda")]
fn gpt_oss_family(gpt_oss_cuda_serve: bool) -> anyhow::Result<ModelFamily> {
    if gpt_oss_cuda_serve {
        Ok(ModelFamily::GptOss)
    } else {
        anyhow::bail!(
            "{}",
            crate::oapi::backend_select::GPTOSS_NO_CUDA_WITHOUT_THE_OPT_IN
        )
    }
}

#[cfg(feature = "cuda")]
fn qwen3_5_dense_family(dense_cuda_serve: bool) -> anyhow::Result<ModelFamily> {
    if dense_cuda_serve {
        Ok(ModelFamily::Qwen3_5Moe)
    } else {
        anyhow::bail!("{}", crate::oapi::backend_select::QWEN35_DENSE_NO_CUDA)
    }
}

#[cfg(feature = "cuda")]
pub(crate) fn detect_family_with_dense_cuda_serve(
    raw_cfg: &str,
    dense_cuda_serve: bool,
) -> anyhow::Result<ModelFamily> {
    detect_family_with_opt_ins(raw_cfg, dense_cuda_serve, gpt_oss_cuda_serve_enabled())
}

#[cfg(feature = "cuda")]
pub(crate) fn detect_family_with_opt_ins(
    raw_cfg: &str,
    dense_cuda_serve: bool,
    gpt_oss_cuda_serve: bool,
) -> anyhow::Result<ModelFamily> {
    let v: serde_json::Value =
        serde_json::from_str(raw_cfg).map_err(|e| anyhow::anyhow!("parse config.json: {e}"))?;

    let e4b = v
        .get("hidden_size_per_layer_input")
        .or_else(|| {
            v.get("text_config")
                .and_then(|t| t.get("hidden_size_per_layer_input"))
        })
        .and_then(|x| x.as_u64())
        .unwrap_or(0)
        > 0;

    let moe = v
        .get("enable_moe_block")
        .or_else(|| v.get("text_config").and_then(|t| t.get("enable_moe_block")))
        .and_then(|x| x.as_bool())
        .unwrap_or(false);
    let gemma4_family = |moe: bool, e4b: bool| {
        if moe {
            ModelFamily::Gemma4Moe
        } else if e4b {
            ModelFamily::Gemma4E4b
        } else {
            ModelFamily::Gemma4
        }
    };
    if let Some(arch_arr) = v.get("architectures").and_then(|x| x.as_array()) {
        for a in arch_arr {
            if let Some(s) = a.as_str() {
                let l = s.to_ascii_lowercase();
                if l.starts_with("qwen3omni") || l.starts_with("qwen3_omni") {
                    return Ok(ModelFamily::Omni);
                }
                if l.starts_with("qwen4exp") || l.starts_with("qwen4_exp") {
                    return Ok(ModelFamily::Qwen4Exp);
                }
                if l.starts_with("qwen3_5moe")
                    || l.starts_with("qwen3.5moe")
                    || l == "qwen3_5moeforcausallm"
                    || l == "qwen3_5moeforconditionalgeneration"
                {
                    return Ok(ModelFamily::Qwen3_5Moe);
                }
                if l.starts_with("gemma4") {
                    return Ok(gemma4_family(moe, e4b));
                }
                if l.starts_with("qwen3_5") || l.starts_with("qwen3.5") {
                    return qwen3_5_dense_family(dense_cuda_serve);
                }
                if l.starts_with("qwen3") {
                    return Ok(ModelFamily::Qwen3);
                }
                if l.starts_with("laguna") {
                    return Ok(ModelFamily::Laguna);
                }
                if l.starts_with("gptoss") || l.starts_with("gpt_oss") {
                    return gpt_oss_family(gpt_oss_cuda_serve);
                }
            }
        }
    }
    if let Some(mt) = v.get("model_type").and_then(|x| x.as_str()) {
        let l = mt.to_ascii_lowercase();
        if l.starts_with("qwen3omni") || l.starts_with("qwen3_omni") {
            return Ok(ModelFamily::Omni);
        }
        if l == "qwen3_5_moe" || l == "qwen3.5_moe" || l.starts_with("qwen3_5_moe") {
            return Ok(ModelFamily::Qwen3_5Moe);
        }
        if l.starts_with("gemma4") {
            return Ok(gemma4_family(moe, e4b));
        }
        if l.starts_with("qwen3_5") || l.starts_with("qwen3.5") {
            return qwen3_5_dense_family(dense_cuda_serve);
        }
        if l.starts_with("qwen3") {
            return Ok(ModelFamily::Qwen3);
        }
        if l.starts_with("laguna") {
            return Ok(ModelFamily::Laguna);
        }
        if l.starts_with("gpt_oss") || l.starts_with("gptoss") {
            return gpt_oss_family(gpt_oss_cuda_serve);
        }
    }
    anyhow::bail!(
        "could not detect model family (need architectures or model_type starting with gemma4/qwen3/qwen3_5_moe/laguna/gpt_oss)"
    )
}

#[cfg(feature = "cuda")]
pub(crate) fn resolve_qwen3_5_moe_eos_ids(
    tokenizer: &tokenizers::Tokenizer,
    model_dir: &Path,
    cfg: &nv_models::qwen3_5_moe::Qwen3MoeConfig,
) -> Vec<u32> {
    let mut ids: Vec<u32> = Vec::new();
    let mut push = |id: u32| {
        if !ids.contains(&id) {
            ids.push(id);
        }
    };

    let tok_cfg = model_dir.join("tokenizer_config.json");
    if let Ok(raw) = std::fs::read_to_string(&tok_cfg) {
        if let Ok(v) = serde_json::from_str::<serde_json::Value>(&raw) {
            if let Some(s) = v.get("eos_token").and_then(|x| x.as_str()) {
                if let Some(id) = tokenizer.token_to_id(s) {
                    push(id);
                }
            }
        }
    }
    if let Some(id) = tokenizer.token_to_id("<|im_end|>") {
        push(id);
    }
    if cfg.eos_token_id != 0 {
        push(cfg.eos_token_id);
    }
    if ids.is_empty() {
        ids.push(cfg.eos_token_id);
    }
    ids
}

#[cfg(feature = "cuda")]
pub(crate) fn gemma4_target_dims(
    model_id: &str,
    cfg: &nv_models::gemma4::Gemma4Config,
) -> TargetDims {
    TargetDims {
        model_id: model_id.to_string(),
        hidden_size: cfg.hidden_size,
        vocab_size: cfg.vocab_size,
        num_hidden_layers: cfg.num_hidden_layers,
    }
}

#[cfg(feature = "cuda")]
#[cfg(test)]
pub(crate) fn load_eagle3_state(
    device: &candle_core::Device,
    target_embed: Option<&candle_core::Tensor>,
) -> Option<Arc<Eagle3Shared>> {
    load_eagle3_state_for_target(device, target_embed, None)
}

#[cfg(feature = "cuda")]
pub(crate) fn load_eagle3_state_for_target(
    device: &candle_core::Device,
    target_embed: Option<&candle_core::Tensor>,
    target: Option<&TargetDims>,
) -> Option<Arc<Eagle3Shared>> {
    let dir_os = std::env::var_os("NV_EAGLE3_DRAFT_DIR")?;
    let dir = PathBuf::from(dir_os);
    if !dir.is_dir() {
        warn!(
            path = %dir.display(),
            "NV_EAGLE3_DRAFT_DIR set but path is not a directory; spec-decode disabled"
        );
        return None;
    }
    let mut scorer = match nv_specdecode::eagle3_loader::LoadedEagle3Scorer::try_load(&dir, device)
    {
        Ok(s) => s,
        Err(err) => {
            warn!(
                path = %dir.display(),
                error = %err,
                "failed to load Eagle3 scorer; spec-decode disabled"
            );
            return None;
        }
    };
    if let Some(embed) = target_embed {
        match scorer.share_embed_tokens_with_target(embed) {
            Ok(true) => tracing::info!(
                "eagle3 embed_tokens identical to target; sharing target embedding (frees ~2.8 GiB)"
            ),
            Ok(false) => tracing::info!(
                "eagle3 embed_tokens differ from target; keeping drafter's private copy"
            ),
            Err(err) => warn!(
                error = %err,
                "eagle3 embed_tokens equality check failed; keeping drafter's private copy"
            ),
        }
    }
    let cfg = scorer.config();
    if let Some(t) = target {
        let dims = DrafterDims {
            fc_in_dim: cfg.fc_in_dim(),
            target_vocab_size: cfg.target_vocab_size,
            aux_layer_ids: cfg.eagle_aux_hidden_state_layer_ids.clone(),
        };
        let problems = drafter_target_mismatch(t, &dims);
        if !problems.is_empty() {
            tracing::error!(
                "{}",
                drafter_mismatch_message("eagle3", &dir.display().to_string(), t, &problems)
            );
            return None;
        }
    }
    let aux_layers: Vec<usize> = cfg
        .eagle_aux_hidden_state_layer_ids
        .iter()
        .map(|&c| c.saturating_sub(1))
        .collect();
    let draft_vocab = cfg.draft_vocab_size;
    let max_depth = std::env::var("NV_EAGLE3_DEPTH")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(3);
    let branch_factor = std::env::var("NV_EAGLE3_BRANCH")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(2);
    let total_budget = std::env::var("NV_EAGLE3_BUDGET")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(7);
    let proposer = nv_specdecode::eagle3::Eagle3Proposer::new(
        scorer,
        nv_specdecode::eagle3::Eagle3Config {
            max_depth,
            branch_factor,
            total_budget,
            vocab_size: draft_vocab,
        },
    );
    tracing::info!(
        path = %dir.display(),
        draft_vocab,
        aux_layers = ?aux_layers,
        "Eagle3 draft scorer loaded; speculative decoding enabled for Gemma4"
    );
    Some(Arc::new(Eagle3Shared {
        proposer,
        aux_layers,
        pool: tokio::sync::Mutex::new(Eagle3State {
            verify: None,
            lease_out: false,
            chain: None,
            dflash_draft: None,
        }),
    }))
}

#[cfg(feature = "cuda")]
pub(crate) fn load_dflash_state_for_target(
    device: &candle_core::Device,
    target_embed: Option<&candle_core::Tensor>,
    target: Option<&TargetDims>,
) -> Option<Arc<DFlashShared>> {
    let Some(dir_os) = std::env::var_os("NV_DFLASH_DRAFT_DIR") else {
        warn!(
            "NV_DFLASH_DRAFT_DIR is unset and there is no default lookup for the DFlash \
             drafter (its snapshot is not part of the model hub); DFlash spec-decode disabled"
        );
        return None;
    };
    let dir = PathBuf::from(dir_os);
    if !dir.is_dir() {
        warn!(
            path = %dir.display(),
            "NV_DFLASH_DRAFT_DIR set but path is not a directory; spec-decode disabled"
        );
        return None;
    }
    let drafter = match nv_specdecode::dflash::LoadedDFlashDrafter::try_load_with_target_embed(
        &dir,
        device,
        target_embed,
    ) {
        Ok(d) => d,
        Err(err) => {
            warn!(
                path = %dir.display(),
                error = %err,
                "failed to load DFlash drafter; spec-decode disabled"
            );
            return None;
        }
    };

    if let Some(t) = target {
        let cfg = drafter.config();
        let dims = DrafterDims {
            fc_in_dim: cfg.fc_in_dim(),
            target_vocab_size: cfg.target_vocab_size,
            aux_layer_ids: cfg.aux_hidden_state_layer_ids.clone(),
        };
        let problems = drafter_target_mismatch(t, &dims);
        if !problems.is_empty() {
            tracing::error!(
                "{}",
                drafter_mismatch_message("dflash", &dir.display().to_string(), t, &problems)
            );
            return None;
        }
    }
    let aux_layers: Vec<usize> = drafter
        .config()
        .aux_hidden_state_layer_ids
        .iter()
        .map(|&c| c.saturating_sub(1))
        .collect();
    tracing::info!(
        path = %dir.display(),
        block_size = drafter.config().block_size,
        aux_layers = ?aux_layers,
        "DFlash drafter loaded; speculative decoding enabled for Gemma4"
    );
    Some(Arc::new(DFlashShared {
        drafter,
        aux_layers,
        pool: tokio::sync::Mutex::new(Eagle3State {
            verify: None,
            lease_out: false,
            chain: None,
            dflash_draft: None,
        }),
    }))
}

#[cfg(any(feature = "cuda", test))]
pub(crate) const SERVING_TOKEN_ID_FILES_IN_PRECEDENCE_ORDER: [&str; 2] =
    ["generation_config.json", "config.json"];

#[cfg(any(feature = "cuda", test))]
fn declaring_scopes(v: &serde_json::Value) -> impl Iterator<Item = &serde_json::Value> {
    [Some(v), v.get("text_config")].into_iter().flatten()
}

#[cfg(any(feature = "cuda", test))]
pub(crate) fn parse_bos_id(raw_cfg: &str) -> Option<u32> {
    let v: serde_json::Value = serde_json::from_str(raw_cfg).ok()?;
    for scope in declaring_scopes(&v) {
        if let Some(n) = scope.get("bos_token_id").and_then(|x| x.as_u64()) {
            return Some(n as u32);
        }
    }
    None
}

#[cfg(any(feature = "cuda", test))]
pub(crate) fn parse_eos_ids(raw_cfg: &str) -> Option<Vec<u32>> {
    let v: serde_json::Value = serde_json::from_str(raw_cfg).ok()?;
    let mut out: Vec<u32> = Vec::new();
    for scope in declaring_scopes(&v) {
        match scope.get("eos_token_id") {
            Some(serde_json::Value::Number(n)) => out.extend(n.as_u64().map(|x| x as u32)),
            Some(serde_json::Value::Array(a)) => {
                out.extend(a.iter().filter_map(|x| x.as_u64()).map(|x| x as u32))
            }
            _ => {}
        }
    }
    out.sort_unstable();
    out.dedup();
    (!out.is_empty()).then_some(out)
}

#[cfg(any(feature = "cuda", test))]
pub(crate) fn bos_id_from_dir(dir: &Path) -> Option<u32> {
    SERVING_TOKEN_ID_FILES_IN_PRECEDENCE_ORDER
        .iter()
        .find_map(|file| {
            std::fs::read_to_string(dir.join(file))
                .ok()
                .and_then(|raw| parse_bos_id(&raw))
        })
}

#[cfg(any(feature = "cuda", test))]
pub(crate) fn eos_ids_from_dir(dir: &Path) -> anyhow::Result<Vec<u32>> {
    for file in SERVING_TOKEN_ID_FILES_IN_PRECEDENCE_ORDER {
        if let Some(ids) = std::fs::read_to_string(dir.join(file))
            .ok()
            .and_then(|raw| parse_eos_ids(&raw))
        {
            return Ok(ids);
        }
    }
    anyhow::bail!(
        "no eos_token_id in {}/generation_config.json or config.json (nor in their text_config): \
         refusing to serve a model with no stop condition. The [1, 106] this engine used to \
         default to is gemma's, invented for every family, and it also disarmed \
         splice_bos_at_position_0_only: a fabricated EOS set never contains the checkpoint's \
         real BOS, so the position-0 rule answered `splice it` unconditionally",
        dir.display()
    )
}

pub(crate) fn has_safetensors(dir: &Path) -> bool {
    if dir.join("model.safetensors").is_file() {
        return true;
    }
    if dir.join("model.safetensors.index.json").is_file() {
        return true;
    }

    if let Ok(rd) = std::fs::read_dir(dir) {
        for entry in rd.flatten() {
            let p: PathBuf = entry.path();
            if p.extension().and_then(|s| s.to_str()) == Some("safetensors") {
                return true;
            }
        }
    }
    false
}

pub fn default_engine() -> Option<Arc<dyn ChatEngine>> {
    default_engine_from_env()
}

#[cfg(feature = "cuda")]
pub fn default_engine_from_env() -> Option<Arc<dyn ChatEngine>> {
    let dir = std::env::var_os("NV_CHAT_MODEL_DIR")?;
    let dir = PathBuf::from(dir);
    Some(load_engine_dir(&dir))
}

#[cfg(not(feature = "cuda"))]
pub fn default_engine_from_env() -> Option<Arc<dyn ChatEngine>> {
    if std::env::var_os("NV_CHAT_MODEL_DIR").is_some() {
        tracing::warn!(
            "NV_CHAT_MODEL_DIR set but this build lacks the `cuda` feature; nv-engine chat disabled"
        );
    }
    None
}

#[cfg(feature = "cuda")]
pub(crate) fn try_load_engine_dir(dir: &Path) -> anyhow::Result<Arc<dyn ChatEngine>> {
    let eng = NvEngineChat::try_load(dir)?;
    tracing::info!(model_dir = %dir.display(), model_id = %eng.model_id(), "NvEngineChat loaded");
    Ok(Arc::new(eng))
}

#[cfg(feature = "cuda")]
pub(crate) fn load_engine_dir(dir: &Path) -> Arc<dyn ChatEngine> {
    match try_load_engine_dir(dir) {
        Ok(eng) => eng,
        Err(err) => {
            panic!(
                "NvEngineChat::try_load failed for {}: {err:#}\n\
                 Refusing to start with a stub chat engine. \
                 Fix the model load error or unset NV_CHAT_MODEL_DIR(S) to disable chat.",
                dir.display()
            )
        }
    }
}

pub const ALLOW_UNKNOWN_MODEL_ENV: &str = "NV_CHAT_ALLOW_UNKNOWN_MODEL";

pub fn allow_unknown_model_from(raw: Option<&str>) -> bool {
    matches!(
        raw.unwrap_or_default().trim().to_ascii_lowercase().as_str(),
        "1" | "true" | "yes" | "on"
    )
}

pub fn allow_unknown_model() -> bool {
    allow_unknown_model_from(std::env::var(ALLOW_UNKNOWN_MODEL_ENV).ok().as_deref())
}

fn warn_unknown_model_fallback(requested: &str, served: &str) {
    static SEEN: std::sync::OnceLock<std::sync::Mutex<std::collections::HashSet<String>>> =
        std::sync::OnceLock::new();
    let first = SEEN
        .get_or_init(Default::default)
        .lock()
        .map(|mut s| s.insert(requested.to_string()))
        .unwrap_or(false);
    if !first {
        return;
    }
    tracing::warn!(
        requested = %requested,
        served = %served,
        "{ALLOW_UNKNOWN_MODEL_ENV} is set: serving an unknown model id with the only loaded \
         engine instead of returning 404 model_not_found. The response will echo `{served}`, so \
         a client typo is indistinguishable from a hit. Unset {ALLOW_UNKNOWN_MODEL_ENV} to get \
         OpenAI's behaviour."
    );
}

struct RegistryInner {
    engines: std::collections::HashMap<String, Arc<dyn ChatEngine>>,
    default_id: String,
    order: Vec<String>,
    the_unknown_model_fallback_reads_boot_engine_count_so_runtime_adapters_cannot_flip_it: usize,
}

#[derive(Clone)]
pub struct ChatRegistry {
    inner: Arc<std::sync::RwLock<RegistryInner>>,
}

#[derive(Debug, PartialEq, Eq)]
pub enum RemoveDenied {
    NotFound,
    DefaultEngine,
}

impl ChatRegistry {
    pub fn from_engines(engines: Vec<Arc<dyn ChatEngine>>) -> Option<Self> {
        if engines.is_empty() {
            return None;
        }
        let default_id = engines[0].model_id().to_string();
        let mut order = Vec::with_capacity(engines.len());
        let mut map = std::collections::HashMap::with_capacity(engines.len());
        for eng in engines {
            let id = eng.model_id().to_string();
            if map.insert(id.clone(), eng).is_none() {
                order.push(id);
            }
        }
        let boot_count = order.len();
        Some(Self {
            inner: Arc::new(std::sync::RwLock::new(RegistryInner {
                engines: map,
                default_id,
                order,
                the_unknown_model_fallback_reads_boot_engine_count_so_runtime_adapters_cannot_flip_it:
                    boot_count,
            })),
        })
    }

    pub fn single(engine: Arc<dyn ChatEngine>) -> Self {
        Self::from_engines(vec![engine]).expect("single engine is non-empty")
    }

    fn read(&self) -> std::sync::RwLockReadGuard<'_, RegistryInner> {
        self.inner
            .read()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }

    pub fn resolve(&self, model: Option<&str>) -> Option<Arc<dyn ChatEngine>> {
        self.resolve_with(model, allow_unknown_model())
    }

    pub fn resolve_with(
        &self,
        model: Option<&str>,
        allow_unknown: bool,
    ) -> Option<Arc<dyn ChatEngine>> {
        let inner = self.read();
        match model {
            None | Some("") => inner.engines.get(&inner.default_id).cloned(),
            Some(m) => match inner.engines.get(m) {
                Some(eng) => Some(eng.clone()),
                None => {
                    if let Some(canon) = crate::oapi::model_ids::canonical_model_id(m) {
                        if let Some(eng) = inner.engines.get(&canon) {
                            return Some(eng.clone());
                        }
                    }
                    if allow_unknown
                        && inner
                            .the_unknown_model_fallback_reads_boot_engine_count_so_runtime_adapters_cannot_flip_it
                            == 1
                        && inner.engines.contains_key(&inner.default_id)
                    {
                        warn_unknown_model_fallback(m, &inner.default_id);
                        return inner.engines.get(&inner.default_id).cloned();
                    }
                    None
                }
            },
        }
    }

    pub fn default_engine(&self) -> Arc<dyn ChatEngine> {
        let inner = self.read();
        inner
            .engines
            .get(&inner.default_id)
            .cloned()
            .expect("default engine present")
    }

    pub fn model_ids(&self) -> Vec<String> {
        self.read().order.clone()
    }

    pub fn contains(&self, model: &str) -> bool {
        self.read().engines.contains_key(model)
    }

    pub fn entries_in_order(&self) -> Vec<(String, Arc<dyn ChatEngine>)> {
        let inner = self.read();
        inner
            .order
            .iter()
            .filter_map(|id| inner.engines.get(id).map(|e| (id.clone(), e.clone())))
            .collect()
    }

    pub fn insert(&self, engine: Arc<dyn ChatEngine>) {
        let mut inner = self
            .inner
            .write()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let id = engine.model_id().to_string();
        if inner.engines.insert(id.clone(), engine).is_none() {
            inner.order.push(id);
        }
    }

    pub fn remove(&self, id: &str) -> Result<(), RemoveDenied> {
        let mut inner = self
            .inner
            .write()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if id == inner.default_id {
            return Err(RemoveDenied::DefaultEngine);
        }
        if inner.engines.remove(id).is_none() {
            return Err(RemoveDenied::NotFound);
        }
        inner.order.retain(|x| x != id);
        Ok(())
    }
}

static LIVE_REGISTRY: std::sync::OnceLock<ChatRegistry> = std::sync::OnceLock::new();

pub fn publish_live_registry(reg: &ChatRegistry) {
    let _ = LIVE_REGISTRY.set(reg.clone());
}

pub fn live_registry() -> Option<ChatRegistry> {
    LIVE_REGISTRY.get().cloned()
}

#[cfg(any(feature = "cuda", test))]
pub(crate) fn build_registry_strict<F>(dirs: &[PathBuf], mut load: F) -> Option<ChatRegistry>
where
    F: FnMut(&Path) -> anyhow::Result<Arc<dyn ChatEngine>>,
{
    let mut engines: Vec<Arc<dyn ChatEngine>> = Vec::with_capacity(dirs.len());
    for d in dirs {
        match load(d) {
            Ok(eng) => engines.push(eng),
            Err(err) => {
                panic!(
                    "chat model dir failed to load: {}: {err:#}\n\
                     Every NV_CHAT_MODEL_DIRS entry must load. A model that is \
                     configured but absent from /v1/models is a silent capability \
                     regression, so this is fatal rather than skipped. Fix the dir \
                     or remove it from NV_CHAT_MODEL_DIRS.",
                    d.display()
                );
            }
        }
    }
    if engines.is_empty() {
        panic!(
            "NV_CHAT_MODEL_DIRS resolved to no entries; refusing to start a chat-less \
             server when chat was explicitly requested. Unset NV_CHAT_MODEL_DIRS to \
             disable chat."
        );
    }
    ChatRegistry::from_engines(engines)
}

pub fn registry_from_env() -> Option<ChatRegistry> {
    #[cfg(all(not(feature = "cuda"), feature = "wgpu"))]
    {
        if std::env::var_os("NV_WGPU_CHAT_MODEL_DIRS").is_some()
            || std::env::var_os("NV_CHAT_MODEL_DIR").is_some()
        {
            match crate::oapi::chat_engine_wgpu::engine_from_env_dirs() {
                Ok(engines) => return ChatRegistry::from_engines(engines),
                Err(err) => {
                    tracing::warn!("wgpu chat engines failed to load: {err:#}");
                    return None;
                }
            }
        }
    }
    #[cfg(feature = "cuda")]
    if let Some(list) = std::env::var_os("NV_CHAT_MODEL_DIRS") {
        let list = list.to_string_lossy().into_owned();
        let dirs: Vec<PathBuf> = list
            .split([',', ':'])
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .map(PathBuf::from)
            .collect();
        if dirs.is_empty() {
            return None;
        }
        let mut bases: Vec<(String, PathBuf)> = Vec::with_capacity(dirs.len());
        let reg = build_registry_strict(&dirs, |d| {
            let eng = try_load_engine_dir(d)?;
            bases.push((eng.model_id().to_string(), d.to_path_buf()));
            Ok(eng)
        })?;
        let mut engines: Vec<Arc<dyn ChatEngine>> =
            reg.entries_in_order().into_iter().map(|(_, e)| e).collect();
        super::lora_cuda::attach_boot_adapters(&mut engines, &bases);
        return ChatRegistry::from_engines(engines);
    }
    #[cfg(not(feature = "cuda"))]
    if std::env::var_os("NV_CHAT_MODEL_DIRS").is_some() {
        tracing::warn!(
            "NV_CHAT_MODEL_DIRS set but this build lacks the `cuda` feature; nv-engine chat disabled"
        );
    }
    #[cfg(feature = "cuda")]
    if let Some(dir) = std::env::var_os("NV_CHAT_MODEL_DIR") {
        let dir = PathBuf::from(dir);
        let eng = load_engine_dir(&dir);
        let bases = vec![(eng.model_id().to_string(), dir)];
        let mut engines: Vec<Arc<dyn ChatEngine>> = vec![eng];
        super::lora_cuda::attach_boot_adapters(&mut engines, &bases);
        return ChatRegistry::from_engines(engines);
    }
    default_engine_from_env().map(ChatRegistry::single)
}

#[cfg(all(test, feature = "cuda"))]
mod gguf_seam_tests {
    use super::*;

    fn seam_dir(tag: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "cuda_gguf_seam_{}_{tag}_{:?}",
            std::process::id(),
            std::thread::current().id()
        ));
        std::fs::remove_dir_all(&dir).ok();
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn a_gguf_only_dir_is_refused_naming_the_gap_and_a_dir_with_config_json_is_not() {
        let dir = seam_dir("outer");
        std::fs::write(dir.join("model.gguf"), b"x").unwrap();
        let err = match NvEngineChat::try_load(&dir) {
            Ok(_) => panic!("gguf dir must be refused"),
            Err(e) => e,
        };
        let msg = format!("{err:#}");
        assert!(
            msg.contains("GGUF checkpoint dir") && msg.contains("wgpu"),
            "try_load must name the gap and the working backend, not a generic \
             missing-file error: {msg}"
        );
        assert!(
            !msg.contains("missing required file"),
            "the generic required-files bail must not shadow the GGUF refusal: {msg}"
        );
        assert!(
            !msg.contains("no Gemma4Moe variant"),
            "the refusal must not assert a variant this file declares: {msg}"
        );
        assert!(
            msg.contains("safetensors-only") && msg.contains("from_gguf"),
            "the refusal must name the real gap (safetensors-only loader): {msg}"
        );
        std::fs::remove_dir_all(&dir).ok();
        {
            let dir = seam_dir("cfg");
            std::fs::write(dir.join("model.gguf"), b"x").unwrap();
            std::fs::write(dir.join("config.json"), b"{}").unwrap();
            refuse_gguf_checkpoint_dir(&dir).expect("config.json present: not a GGUF-only dir");
            let err = NvEngineChat::try_load(&dir).err().expect("must still fail");
            let msg = format!("{err:#}");
            assert!(
                msg.contains("missing required file tokenizer.json"),
                "a dir with config.json must follow the normal required-files path: {msg}"
            );
            std::fs::remove_dir_all(&dir).ok();
        }
    }

}

#[cfg(test)]
mod laguna_spec_startup_tests {
    use super::*;
    use std::sync::mpsc;
    use std::time::Duration;

    #[test]
    fn packed_arm_rejects_disabled_routes_and_invalid_draft_directories() {
        for (mode, draft) in [
            (EngineBuildMode::AdapterServingLean, true),
            (EngineBuildMode::Full, false),
        ] {
            let err = validate_laguna_packed_draft_arm(true, mode, draft).unwrap_err();
            assert!(err.to_string().contains("NV_LAGUNA_DFLASH_NVFP4=1"));
            validate_laguna_packed_draft_arm(false, mode, draft).unwrap();
        }
        validate_laguna_packed_draft_arm(true, EngineBuildMode::Full, true).unwrap();
        {
            assert!(required_laguna_draft_dir(None)
                .unwrap_err()
                .to_string()
                .contains("NV_LAGUNA_DFLASH_DIR"));
            let file = std::env::current_exe().unwrap();
            let err = required_laguna_draft_dir(Some(file.clone().into_os_string())).unwrap_err();
            assert!(err.to_string().contains(&file.display().to_string()));
            let dir = file.parent().unwrap().to_path_buf();
            assert_eq!(
                required_laguna_draft_dir(Some(dir.clone().into_os_string())).unwrap(),
                dir
            );
        }
    }

    #[test]
    fn packed_startup_preserves_errors_and_rejects_panic_or_exit_without_readiness() {
        for cause in [
            "load draft: malformed packed scales",
            "warmup: CUDA_ERROR_OUT_OF_MEMORY",
        ] {
            let err =
                start_laguna_spec_worker::<()>(1, true, move |_, _ready| anyhow::bail!("{cause}"))
                    .unwrap_err();
            let message = format!("{err:#}");
            assert!(message.contains(cause), "lost startup cause: {message}");
            assert!(
                message.contains("NV_LAGUNA_DFLASH_NVFP4=1"),
                "lost required arm: {message}"
            );
            assert!(
                message.contains("refusing plain or BF16 fallback"),
                "lost refusal: {message}"
            );
        }
        {
            let panic = start_laguna_spec_worker::<()>(1, true, |_, _ready| {
                panic!("draft initialization panic")
            })
            .unwrap_err();
            assert!(format!("{panic:#}").contains("draft initialization panic"));
            let closed = start_laguna_spec_worker::<()>(1, true, |requests, ready| {
                drop(ready);
                assert!(requests.recv().is_err());
                Ok(())
            })
            .unwrap_err();
            assert!(format!("{closed:#}").contains("exited before readiness"));
        }
    }

    #[test]
    fn graph_lane_startup_preserves_failures_and_accepts_plain_readiness() {
        for cause in [
            "capture lane 1: CUDA_ERROR_OUT_OF_MEMORY",
            "invalid graph lane selector",
        ] {
            let err = start_laguna_spec_worker_with_graph_requirement::<()>(
                1,
                false,
                true,
                false,
                move |_, _ready| anyhow::bail!("{cause}"),
            )
            .unwrap_err();
            let message = format!("{err:#}");
            assert!(
                message.contains(cause) && message.contains("refusing per-request fallback"),
                "{message}"
            );
        }
        let (exited_tx, exited_rx) = mpsc::channel();
        let (jobs, has_draft) = start_laguna_spec_worker_with_graph_requirement::<()>(
            1,
            false,
            true,
            false,
            move |requests, ready| {
                ready.send(false)?;
                assert!(requests.recv().is_err());
                exited_tx.send(())?;
                Ok(())
            },
        )
        .unwrap()
        .expect("plain graph lanes must report worker readiness");
        assert!(!has_draft);
        drop(jobs);
        exited_rx
            .try_recv()
            .expect("owned graph lane worker was not joined before Drop returned");
    }

    #[test]
    fn default_startup_keeps_fallbacks_that_an_explicit_handoff_route_refuses() {
        for (has_draft, expected) in [(false, "off"), (true, "on")] {
            let ready = start_laguna_spec_worker::<u32>(1, false, move |requests, ready| {
                ready.send(has_draft)?;
                assert_eq!(requests.recv()?, 7);
                assert!(requests.recv().is_err());
                Ok(())
            })
            .unwrap();
            let (jobs, status) = laguna_engine_spec_state(ready);
            assert_eq!(crate::oapi::chat::spec_decode_header_value(status), expected,
                "the header reports the ready worker's loaded draft, not its name or a request's eventual route");
            let mut jobs = jobs.expect("plain and draft-enabled readiness both retain the worker");
            jobs.try_send(7).unwrap();
            jobs.close_and_join().unwrap();
        }
        let unavailable = start_laguna_spec_worker::<()>(1, false, |_, _ready| {
            anyhow::bail!("optional graph initialization failed")
        })
        .unwrap();
        let (jobs, status) = laguna_engine_spec_state(unavailable);
        assert!(jobs.is_none());
        assert_eq!(crate::oapi::chat::spec_decode_header_value(status), "off");
        {
            let err = start_laguna_spec_worker_with_graph_requirement::<()>(1, false, false, true, move |_, _ready| {
                anyhow::bail!("refusing the explicitly selected BF16 handoff route before allocating: source 10.18 GiB")
            }).unwrap_err();
            let message = format!("{err:#}");
            assert!(message.contains("NV_LAGUNA_PREFILL_BF16_HANDOFF=1 requires the handoff worker; refusing per-request fallback") && message.contains("source 10.18 GiB"), "{message}");
            let ready = start_laguna_spec_worker_with_graph_requirement::<()>(
                1,
                false,
                false,
                false,
                move |_, _ready| anyhow::bail!("same failure without the handoff route selected"),
            )
            .unwrap();
            assert!(
                ready.is_none(),
                "without an explicit route the init failure still demotes to the per-request path"
            );
        }
    }

    struct TeardownGate {
        entered: mpsc::Sender<()>,
        release: mpsc::Receiver<()>,
    }

    impl Drop for TeardownGate {
        fn drop(&mut self) {
            self.entered.send(()).unwrap();
            self.release.recv().unwrap();
        }
    }

    #[test]
    fn worker_drop_shutdown_and_startup_failures_join_teardown_without_self_joining() {
        let (entered, teardown_started) = mpsc::channel();
        let (release, released) = mpsc::channel();
        let (dropped, drop_finished) = mpsc::channel();
        let (worker, _) = start_laguna_spec_worker::<()>(1, true, move |requests, ready| {
            let _teardown = TeardownGate {
                entered,
                release: released,
            };
            ready.send(true)?;
            anyhow::ensure!(requests.recv().is_err(), "drop did not close the queue");
            Ok(())
        })
        .unwrap()
        .unwrap();
        let dropping = std::thread::spawn(move || {
            drop(worker);
            dropped.send(()).unwrap();
        });
        teardown_started
            .recv_timeout(Duration::from_secs(1))
            .unwrap();
        let premature = drop_finished.recv_timeout(Duration::from_millis(20));
        release.send(()).unwrap();
        dropping.join().unwrap();
        assert!(
            matches!(premature, Err(mpsc::RecvTimeoutError::Timeout)),
            "worker owner returned before worker teardown completed"
        );
        drop_finished.recv_timeout(Duration::from_secs(1)).unwrap();
        {
            for wrong_ready in [false, true] {
                let (entered, teardown_started) = mpsc::channel();
                let (release, released) = mpsc::channel();
                let (completed, completion) = mpsc::channel();
                let starting = std::thread::spawn(move || {
                    let result = start_laguna_spec_worker::<()>(1, true, move |requests, ready| {
                        let _teardown = TeardownGate {
                            entered,
                            release: released,
                        };
                        if wrong_ready {
                            ready.send(false)?;
                            anyhow::ensure!(
                                requests.recv().is_err(),
                                "refusal did not close the queue"
                            );
                            Ok(())
                        } else {
                            drop(ready);
                            anyhow::bail!("startup failure before teardown")
                        }
                    });
                    completed.send(result).unwrap();
                });
                teardown_started
                    .recv_timeout(Duration::from_secs(1))
                    .unwrap();
                let premature = completion.recv_timeout(Duration::from_millis(20));
                release.send(()).unwrap();
                starting.join().unwrap();
                assert!(
                    matches!(premature, Err(mpsc::RecvTimeoutError::Timeout)),
                    "startup returned while worker teardown remained active"
                );
                let error = completion
                    .recv_timeout(Duration::from_secs(1))
                    .unwrap()
                    .unwrap_err();
                let expected = if wrong_ready {
                    "without the required packed DFlash draft"
                } else {
                    "startup failure before teardown"
                };
                assert!(format!("{error:#}").contains(expected));
            }
        }
        {
            for panics in [false, true] {
                let (mut worker, _) =
                    start_laguna_spec_worker::<()>(1, true, move |requests, ready| {
                        ready.send(true)?;
                        anyhow::ensure!(requests.recv().is_err(), "shutdown did not close the queue");
                        if panics {
                            panic!("worker teardown panic");
                        }
                        anyhow::bail!("worker teardown error")
                    })
                    .unwrap()
                    .unwrap();
                let error = worker.close_and_join().unwrap_err();
                let expected = if panics {
                    "worker teardown panic"
                } else {
                    "worker teardown error"
                };
                assert!(format!("{error:#}").contains(expected));
                worker
                    .close_and_join()
                    .expect("shutdown must consume the handle even on error");
            }
        }
        {
            let (handle_tx, handle_rx) = mpsc::channel();
            let (finished_tx, finished_rx) = mpsc::channel();
            let thread = std::thread::spawn(move || -> anyhow::Result<()> {
                let own_handle = handle_rx.recv()?;
                let worker = LagunaSpecWorker::<()> {
                    jobs: None,
                    thread: Some(own_handle),
                };
                drop(worker);
                finished_tx.send(())?;
                Ok(())
            });
            handle_tx.send(thread).unwrap();
            finished_rx
                .recv_timeout(Duration::from_secs(1))
                .expect("worker attempted to join itself");
        }
    }

    #[test]
    fn packed_ready_worker_keeps_the_bounded_queue_until_jobs_are_dropped() {
        let (release_tx, release_rx) = mpsc::channel();
        let (exited_tx, exited_rx) = mpsc::channel();
        let (started_tx, started_rx) = mpsc::channel();
        let starter = std::thread::spawn(move || {
            started_tx
                .send(start_laguna_spec_worker::<u32>(
                    1,
                    true,
                    move |requests, ready| {
                        ready.send(true)?;
                        release_rx.recv()?;
                        anyhow::ensure!(requests.recv()? == 7, "worker received the wrong request");
                        anyhow::ensure!(requests.recv().is_err(), "worker queue should be closed");
                        exited_tx.send(())?;
                        Ok(())
                    },
                ))
                .unwrap();
        });
        let (jobs, has_draft) = started_rx
            .recv_timeout(Duration::from_secs(1))
            .expect("startup waited for a ready worker to exit")
            .unwrap()
            .unwrap();
        assert!(has_draft);
        jobs.try_send(7).unwrap();
        assert!(matches!(jobs.try_send(8), Err(mpsc::TrySendError::Full(8))));
        release_tx.send(()).unwrap();
        drop(jobs);
        exited_rx
            .recv_timeout(Duration::from_secs(1))
            .expect("worker did not exit after queue closure");
        starter.join().unwrap();
    }
}

#[cfg(test)]
mod spec_serve_max_seq_tests {
    use super::spec_serve_max_seq;

    #[test]
    fn a_spec_cap_below_kv_max_shrinks_only_the_spec_engine_window() {
        assert_eq!(spec_serve_max_seq(Some(131072), 262144), 131072);
        assert_eq!(spec_serve_max_seq(None, 262144), 262144);
        assert_eq!(spec_serve_max_seq(Some(0), 262144), 262144);
        assert_eq!(spec_serve_max_seq(Some(400000), 262144), 262144);
    }

}

#[cfg(test)]
mod registry_policy_tests {
    use super::*;
    use crate::oapi::chat_engine::EchoEngine;

    fn single_reg() -> ChatRegistry {
        ChatRegistry::single(Arc::new(EchoEngine::new("gemma-4-E4B-it", "x")))
    }

    #[test]
    fn unknown_model_is_served_by_the_only_engine_only_behind_the_escape_hatch() {
        let reg = single_reg();
        assert!(reg.resolve_with(Some("gemma-4-e4b"), false).is_none());
        assert!(reg.resolve_with(Some("nope"), false).is_none());
        assert_eq!(
            reg.resolve_with(Some("gemma-4-e4b"), true)
                .unwrap()
                .model_id(),
            "gemma-4-E4B-it"
        );
        for allow in [false, true] {
            for absent in [None, Some("")] {
                assert_eq!(
                    reg.resolve_with(absent, allow).unwrap().model_id(),
                    "gemma-4-E4B-it"
                );
            }
        }
        let two = ChatRegistry::from_engines(vec![
            Arc::new(EchoEngine::new("a", "x")) as Arc<dyn ChatEngine>,
            Arc::new(EchoEngine::new("b", "y")),
        ])
        .unwrap();
        assert!(two.resolve_with(Some("missing"), true).is_none());
    }

    #[test]
    fn aliases_resolve_good_dirs_serve_and_the_escape_hatch_env_parses() {
        let reg = ChatRegistry::single(Arc::new(EchoEngine::new(
            "google/gemma-4-E4B-it-qat-w4a16-ct",
            "x",
        )));
        let store = "/nix/store/0123456789abcdfghijklmnpqrsvwxyz-hf-model-google-\
                     gemma-4-E4B-it-qat-w4a16-ct-6cd26aaa2357fb2bad8c51699a7558a4d1a965bb";
        let base = store.rsplit('/').next().unwrap();
        for alias in [store, base] {
            assert_eq!(
                reg.resolve_with(Some(alias), false).unwrap().model_id(),
                "google/gemma-4-E4B-it-qat-w4a16-ct"
            );
        }
        assert!(reg.resolve_with(Some("google/other"), false).is_none());
        {
            let dirs: Vec<PathBuf> = ["/good/a", "/good/b"].iter().map(PathBuf::from).collect();
            let reg = build_registry_strict(&dirs, |d| echo(d.file_name().unwrap().to_str().unwrap()))
                .expect("all dirs loaded");
            assert_eq!(reg.model_ids(), &["a".to_string(), "b".to_string()]);
        }
        {
            assert!(!allow_unknown_model_from(None));
            assert!(!allow_unknown_model_from(Some("0")));
            assert!(!allow_unknown_model_from(Some("")));
            assert!(allow_unknown_model_from(Some("1")));
            assert!(allow_unknown_model_from(Some(" TRUE ")));
            assert!(allow_unknown_model_from(Some("on")));
        }
    }

    fn echo(id: &str) -> anyhow::Result<Arc<dyn ChatEngine>> {
        Ok(Arc::new(EchoEngine::new(id, "x")) as Arc<dyn ChatEngine>)
    }

    #[test]
    #[should_panic(expected = "chat model dir failed to load")]
    fn one_bad_dir_aborts_even_when_others_are_good() {
        let dirs: Vec<PathBuf> = ["/good/a", "/bad/b", "/good/c"]
            .iter()
            .map(PathBuf::from)
            .collect();
        let _ = build_registry_strict(&dirs, |d| {
            if d.starts_with("/bad") {
                anyhow::bail!("boom: {}", d.display())
            }
            echo(d.file_name().unwrap().to_str().unwrap())
        });
    }

    #[test]
    fn runtime_inserts_are_visible_through_every_clone_and_remove_refuses_the_default() {
        let reg = single_reg();
        let clone_taken_before_the_insert = reg.clone();
        reg.insert(Arc::new(EchoEngine::new("ft:e4b:custom:1234", "y")));
        assert_eq!(
            clone_taken_before_the_insert
                .resolve_with(Some("ft:e4b:custom:1234"), false)
                .unwrap()
                .model_id(),
            "ft:e4b:custom:1234"
        );
        assert_eq!(
            reg.model_ids(),
            &[
                "gemma-4-E4B-it".to_string(),
                "ft:e4b:custom:1234".to_string()
            ]
        );
        reg.insert(Arc::new(EchoEngine::new("ft:e4b:custom:1234", "b")));
        assert_eq!(
            reg.model_ids(),
            &[
                "gemma-4-E4B-it".to_string(),
                "ft:e4b:custom:1234".to_string()
            ],
            "an insert of an existing id replaces without duplicating the order row"
        );
        {
            let reg = single_reg();
            assert_eq!(
                reg.remove("gemma-4-E4B-it"),
                Err(RemoveDenied::DefaultEngine)
            );
            assert_eq!(reg.remove("never-loaded"), Err(RemoveDenied::NotFound));
            reg.insert(Arc::new(EchoEngine::new("ft:x", "a")));
            assert_eq!(reg.remove("ft:x"), Ok(()));
            assert!(reg.resolve_with(Some("ft:x"), false).is_none());
            assert_eq!(reg.model_ids(), &["gemma-4-E4B-it".to_string()]);
        }
    }

}

#[cfg(all(test, feature = "cuda"))]
mod kv_fit_tests {
    use super::*;

    fn cfg() -> nv_models::gemma4::Gemma4Config {
        nv_models::gemma4::Gemma4Config::from_hf_json_str(&cfg_json()).expect("flat config parses")
    }

    fn cfg_json() -> String {
        serde_json::json!({
            "tie_word_embeddings": true,
            "hidden_size": 256,
            "intermediate_size": 512,
            "num_hidden_layers": 4,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "num_global_key_value_heads": serde_json::Value::Null,
            "head_dim": 256,
            "global_head_dim": 512,
            "vocab_size": 1000,
            "max_position_embeddings": 4096,
            "rms_norm_eps": 1e-6,
            "sliding_window": 512,
            "final_logit_softcapping": serde_json::Value::Null,
            "num_kv_shared_layers": 4,
            "layer_types": ["sliding_attention","sliding_attention","sliding_attention","full_attention"],
            "attention_k_eq_v": false,
            "hidden_activation": "gelu_pytorch_tanh",
            "rope_parameters": {
                "full_attention": { "rope_theta": 1000000.0 },
                "sliding_attention": { "rope_theta": 10000.0 }
            }
        })
        .to_string()
    }

    #[test]
    fn nv_batch_engine_plans_the_paged_pool_the_batch_engine_allocates() {
        let dir = crate::oapi::vram_plan::synthetic_checkpoint_dir(
            "g4_batch_paged_pool",
            &[("model.language_model.layers.0.self_attn.q_proj.weight", 1 << 20)],
        );
        let block = crate::oapi::vram_plan::GEMMA4_BATCH_BLOCK_SIZE_IS_THE_SCHEDULER_BLOCK;
        let _the_plan_reads_the_ledger_as_prior_engines =
            crate::oapi::vram_plan::ledger_tests_run_one_at_a_time();
        for (engine, pinned_blocks) in [(None, None), (Some("1"), None), (Some("1"), Some("64"))] {
            let _env = crate::oapi::vram_plan::ScopedEnvRestoredOnDrop::with(&[
                ("NV_BATCH_ENGINE", engine),
                ("NV_BATCH_KV_BLOCKS", pinned_blocks),
                ("NV_BATCH_KV_FRACTION", None),
                ("NV_BATCH_MAX_SEQS", None),
                ("NV_BATCH_GRAPH", None),
                ("NV_NO_SPEC", Some("1")),
                ("NV_KV_MAX_SEQ_LEN", None),
                ("NV_KV_MAX_SEQ_LEN_GEMMA4", None),
                ("NV_VRAM_PLAN_KV_CONCURRENT", None),
            ]);
            let boot = crate::oapi::vram_plan::plan_gemma4_shaped(
                &dir,
                &cfg_json(),
                64 << 30,
                crate::oapi::vram_plan::BootPlanShape::Full,
            )
            .unwrap();
            let charged: Vec<u64> = boot
                .plan
                .items
                .iter()
                .filter(|i| i.label.starts_with("batch-paged-kv-pool"))
                .map(|i| i.bytes)
                .collect();
            let Some(pool) = boot.gemma4_batch_pool else {
                assert!(
                    engine.is_none() && charged.is_empty(),
                    "NV_BATCH_ENGINE={engine:?}: the engine builds a pool exactly when the plan \
                     hands it one\n{}",
                    boot.plan.render()
                );
                continue;
            };
            let the_config_gemma4_paged_batch_stepper_new_allocates =
                if pool.lanes > 0 && nv_models::gemma4::kv_ring_enabled() {
                    nv_models::paged_fp8::PagedPoolConfig::from_gemma4_hybrid(
                        &cfg(),
                        pool.pool_blocks,
                        block,
                        pool.lanes,
                    )
                } else {
                    nv_models::paged_fp8::PagedPoolConfig::from_gemma4(&cfg(), pool.pool_blocks, block)
                };
            assert_eq!(
                charged,
                vec![the_config_gemma4_paged_batch_stepper_new_allocates.pool_bytes() as u64],
                "NV_BATCH_ENGINE allocates a paged KV pool at boot, so the plan must charge the \
                 pool build_gemma4_batch_engine is handed\n{}",
                boot.plan.render()
            );
            assert_eq!(
                pool.num_blocks,
                pinned_blocks.map_or(8 * boot.kv_max_seq_len.div_ceil(block), |b| b.parse().unwrap()),
                "eight sequences at kv_max fit this budget's fraction, and NV_BATCH_KV_BLOCKS pins"
            );
            assert!(boot.plan.fits(), "{}", boot.plan.render());
        }
        std::fs::remove_dir_all(&dir).unwrap();
    }

    fn kv_gib(c: &nv_models::gemma4::Gemma4Config, kv_max: usize, drafter: usize) -> f64 {
        let rings = nv_models::gemma4::kv_ring_enabled();
        let b = nv_models::gemma4::kv_budget(c, kv_max, true, rings, drafter);
        (b.worst_total() + nv_models::gemma4::gqa512_verify_scratch_bytes(c)) as f64
            / (1u64 << 30) as f64
    }

    #[test]
    fn plan_solver_lands_on_the_largest_kv_that_fits_reports_no_fit_and_is_bypassed_by_an_explicit_env() {
        let c = cfg();
        let weights = 40.0f64;
        let budget = weights + kv_gib(&c, 65536, 0) + 0.001;
        let fits = |kv: usize| weights + kv_gib(&c, kv, 0) <= budget;
        let kv = crate::oapi::vram_plan::solve_max_kv_tokens_monotone(fits, 131072)
            .expect("65536 fits by construction so the solver must find at least that");
        assert!(fits(kv), "solver answered a kv that does not fit: {kv}");
        assert!(
            kv == 131072 || !fits(kv + 1024),
            "solver left budget on the table: {kv} fits and so does {}",
            kv + 1024
        );
        assert!(kv >= 65536);
        assert_eq!(kv % 1024, 0);
        {
            let c = cfg();
            let fits = |kv: usize| 100.0 + kv_gib(&c, kv, 0) <= 76.0;
            assert!(
                crate::oapi::vram_plan::solve_max_kv_tokens_monotone(fits, 131072).is_none(),
                "weights alone exceed the budget, so no kv_max fits -- that must not read as a fit"
            );
        }
        {
            let c = cfg();
            assert_eq!(c.max_position_embeddings, 4096);
            let requested = 999_999usize;
            assert_eq!(
                requested.min(c.max_position_embeddings.max(1)),
                4096,
                "an explicit NV_KV_MAX_SEQ_LEN is the operator's decision, only clamped to the \
                 checkpoint's max_position_embeddings, never fitted"
            );
        }
    }

    #[test]
    fn drafter_kv_charges_match_the_legacy_product_uncapped_and_fit_where_capped() {
        let c = cfg();
        let requested = 131072usize;
        let elems = 4096usize;
        let uncapped = DrafterKvCharge {
            eagle3_row_elems: elems,
            dflash_row_elems: 0,
            eagle3_cap: None,
        };
        let capped = DrafterKvCharge {
            eagle3_row_elems: elems,
            dflash_row_elems: 0,
            eagle3_cap: Some((16, 2048)),
        };
        let capped_rows = 16 + 2048 + nv_specdecode::eagle3_loader::DRAFTER_KV_CAP_SLACK;
        let (rows, got_elems) = capped.rows_elems(requested);
        assert_eq!((rows, got_elems), (capped_rows, elems));
        let capped_drafter_gib = (4 * capped_rows * elems) as f64 / (1u64 << 30) as f64;
        let weights = 40.0f64;
        let budget = weights + kv_gib(&c, requested, 0) + capped_drafter_gib + 0.01;
        let fits_with = |ch: DrafterKvCharge, kv: usize| {
            let (rows, row_elems) = ch.rows_elems(kv);
            let rings = nv_models::gemma4::kv_ring_enabled();
            let b = nv_models::gemma4::kv_budget_capped(&c, kv, true, rings, row_elems, rows);
            let total = (b.worst_total() + nv_models::gemma4::gqa512_verify_scratch_bytes(&c))
                as f64
                / (1u64 << 30) as f64;
            weights + total <= budget
        };
        assert!(
            fits_with(capped, requested),
            "with the drafter KV cap the accounting must admit the full context"
        );
        assert!(
            !fits_with(uncapped, requested),
            "the uncapped drafter charge (4*131072*4096 = 2 GiB) must not fit this budget"
        );
        {
            let ch = DrafterKvCharge {
                eagle3_row_elems: 128,
                dflash_row_elems: 4096,
                eagle3_cap: Some((16, 2048)),
            };
            let (rows, elems) = ch.rows_elems(32768);
            assert_eq!(
                (rows, elems),
                (32768, 4096),
                "dflash's DFlashContextKv grows uncapped to the context length, so its term \
                 must stay charged at kv_max rows even when the eagle3 drafter cap is set"
            );
        }
        {
            for (e3, df) in [(128usize, 0usize), (0, 4096), (128, 4096), (4096, 128)] {
                let ch = DrafterKvCharge {
                    eagle3_row_elems: e3,
                    dflash_row_elems: df,
                    eagle3_cap: None,
                };
                let (rows, elems) = ch.rows_elems(31337);
                assert_eq!(
                    rows * elems,
                    31337 * drafter_row_elems_charge(e3, df),
                    "with no cap the charge must be byte-identical to the legacy \
                     kv_max * max(eagle3, dflash) elems product"
                );
            }
        }
    }

}

#[cfg(all(test, feature = "cuda"))]
mod qwen3_5_dense_cuda_opt_in {
    use super::*;

    fn hybrid_layer_types() -> String {
        (0..32)
            .map(|i| {
                if (i + 1) % 4 == 0 {
                    "\"full_attention\""
                } else {
                    "\"linear_attention\""
                }
            })
            .collect::<Vec<_>>()
            .join(", ")
    }

    fn dense_config_json() -> String {
        format!(
            r#"{{
  "architectures": ["Qwen3_5ForConditionalGeneration"],
  "model_type": "qwen3_5",
  "tie_word_embeddings": false,
  "text_config": {{
    "model_type": "qwen3_5_text",
    "hidden_size": 4096,
    "num_hidden_layers": 32,
    "num_attention_heads": 16,
    "num_key_value_heads": 4,
    "head_dim": 256,
    "intermediate_size": 12288,
    "vocab_size": 248320,
    "max_position_embeddings": 262144,
    "rms_norm_eps": 1e-06,
    "attn_output_gate": true,
    "linear_num_key_heads": 16,
    "linear_num_value_heads": 32,
    "linear_key_head_dim": 128,
    "linear_value_head_dim": 128,
    "linear_conv_kernel_dim": 4,
    "eos_token_id": 248044,
    "rope_parameters": {{"rope_theta": 10000000.0, "partial_rotary_factor": 0.25}},
    "layer_types": [{}]
  }}
}}"#,
            hybrid_layer_types()
        )
    }

    fn moe_config_json() -> String {
        format!(
            r#"{{
  "architectures": ["Qwen3_5MoeForConditionalGeneration"],
  "model_type": "qwen3_5_moe",
  "tie_word_embeddings": false,
  "text_config": {{
    "model_type": "qwen3_5_moe_text",
    "hidden_size": 2048,
    "num_hidden_layers": 32,
    "num_attention_heads": 16,
    "num_key_value_heads": 2,
    "head_dim": 256,
    "intermediate_size": 6144,
    "moe_intermediate_size": 512,
    "shared_expert_intermediate_size": 512,
    "num_experts": 256,
    "num_experts_per_tok": 8,
    "vocab_size": 248320,
    "max_position_embeddings": 262144,
    "rms_norm_eps": 1e-06,
    "attn_output_gate": true,
    "linear_num_key_heads": 16,
    "linear_num_value_heads": 32,
    "linear_key_head_dim": 128,
    "linear_value_head_dim": 128,
    "linear_conv_kernel_dim": 4,
    "eos_token_id": 248044,
    "rope_parameters": {{"rope_theta": 10000000.0, "partial_rotary_factor": 0.25}},
    "layer_types": [{}]
  }}
}}"#,
            hybrid_layer_types()
        )
    }

    #[test]
    fn the_opt_in_routes_only_the_dense_checkpoint_and_the_default_refusal_names_wgpu() {
        for cfg in [
            dense_config_json(),
            dense_config_json().replace(
                "\"architectures\": [\"Qwen3_5ForConditionalGeneration\"],",
                "",
            ),
        ] {
            let err = detect_family_with_dense_cuda_serve(&cfg, false)
                .expect_err("qwen3.5-dense must still be refused by default");
            let msg = format!("{err:#}");
            assert_eq!(
                msg,
                crate::oapi::backend_select::QWEN35_DENSE_NO_CUDA,
                "the default refusal must stay backend_select::QWEN35_DENSE_NO_CUDA verbatim so \
                 the operator is still told where the checkpoint CAN be served"
            );
            assert!(msg.contains("wgpu"));
            assert_eq!(
                detect_family_with_dense_cuda_serve(&cfg, true).expect(
                    "with the opt-in, both the architectures entry and the bare model_type must \
                     reach the qwen3.5 trunk that nv_models::qwen3_5_moe::Qwen3Moe decodes"
                ),
                ModelFamily::Qwen3_5Moe
            );
        }
        let qwen3 = r#"{"architectures": ["Qwen3ForCausalLM"], "model_type": "qwen3"}"#;
        for gate in [false, true] {
            assert_eq!(
                detect_family_with_dense_cuda_serve(&moe_config_json(), gate).unwrap(),
                ModelFamily::Qwen3_5Moe
            );
            assert_eq!(
                detect_family_with_dense_cuda_serve(qwen3, gate).unwrap(),
                ModelFamily::Qwen3
            );
        }
    }

    #[test]
    fn the_dense_trunk_is_hybrid_expertless_never_graph_decoded_and_agreed_on_by_router_and_parsers() {
        let dense = dense_config_json();
        let moe = moe_config_json();
        assert!(qwen3_5_config_declares_dense_ffn(&dense));
        assert!(
            !qwen3_5_config_declares_dense_ffn(&moe),
            "a config declaring nv_models::qwen3_5_moe::MOE_ONLY_KEYS must never be routed to the \
             dense arm: it would load MoE weights as a dense Mlp"
        );
        nv_models::qwen3_5_moe::Qwen3_5DenseConfig::from_hf_json_str(&dense)
            .expect("the dense arm's parser must accept what the router calls dense");
        nv_models::qwen3_5_moe::Qwen3_5DenseConfig::from_hf_json_str(&moe)
            .expect_err("the dense parser must reject a MoE checkpoint");
        nv_models::qwen3_5_moe::Qwen3MoeConfig::from_hf_json_str(&moe)
            .expect("the MoE arm's parser must accept what the router calls MoE");
        nv_models::qwen3_5_moe::Qwen3MoeConfig::from_hf_json_str(&dense)
            .expect_err("the MoE parser must reject a dense checkpoint");
        {
            let cfg =
                nv_models::qwen3_5_moe::Qwen3_5DenseConfig::from_hf_json_str(&dense_config_json())
                    .expect("parse dense config");
            let trunk = cfg.trunk();
            assert_eq!(trunk.num_hidden_layers, 32);
            assert_eq!(trunk.num_experts, 0);
            let n_linear = trunk
                .layer_types
                .iter()
                .filter(|t| **t == nv_models::qwen3_5_moe::LayerType::LinearAttention)
                .count();
            assert_eq!(
                (n_linear, trunk.layer_types.len() - n_linear),
                (24, 8),
                "the dense build arm must construct 24 LayerMixer::Linear mixers over \
                 nv_layers::linear_attn::LinearAttention and 8 full-attention mixers"
            );
            trunk.moe_config().expect_err(
                "a zero-expert trunk must not be accepted as a MoeConfig: build_layer picks \
                 LayerFfn::Dense only when dense_intermediate is Some, and this is the guard that \
                 makes the other branch fail loudly instead of building empty experts",
            );
            assert_eq!(cfg.rotary_dim(), 64);
        }
        {
            assert!(
                !qwen_graph_decode_selected(ModelFamily::Qwen3_5Moe, &dense_config_json()),
                "GraphedQwen3Moe captures the grouped-MoE decode step and its selection also flips the \
                 device to Device::new_cuda_with_stream; neither applies to a checkpoint whose every \
                 ffn is a dense Mlp"
            );
        }
    }

    #[test]
    fn nv_drafter_mtp_plans_the_mtp_head_both_qwen35_arms_load() {
        let mtp_head = 3 << 20;
        let dir = crate::oapi::vram_plan::synthetic_checkpoint_dir(
            "q35_mtp_head",
            &[
                ("model.language_model.layers.0.mlp.experts.0.down_proj.weight", 7 << 20),
                ("mtp.fc.weight", mtp_head),
            ],
        );
        let _env = crate::oapi::vram_plan::ScopedEnvRestoredOnDrop::with(&[
            ("NV_DRAFTER", Some("mtp")),
            ("NV_MTP_DRAFT_DIR", None),
            ("NV_Q36_BATCH", None),
            ("NV_Q38_BATCH", None),
            ("NV_KV_MAX_SEQ_LEN", None),
            ("NV_KV_MAX_SEQ_LEN_Q38", None),
            ("NV_VRAM_PLAN_KV_CONCURRENT", None),
        ]);
        for (dense, raw) in [(false, moe_config_json()), (true, dense_config_json())] {
            let boot = crate::oapi::vram_plan::plan_qwen35(&dir, &raw, dense, false, 1 << 40).unwrap();
            let item = |prefix: &str| {
                boot.plan
                    .items
                    .iter()
                    .find(|i| i.label.starts_with(prefix))
                    .map(|i| (i.bytes, i.unmeasured))
            };
            assert_eq!(
                item("mtp-head-weights"),
                Some((mtp_head, false)),
                "dense={dense}: NV_DRAFTER=mtp loads the MTP head on both qwen3.5 arms and \
                 weights-resident excludes mtp.*, so only this item can charge it\n{}",
                boot.plan.render()
            );
            assert_eq!(
                item("mtp-draft-runtime").map(|(_, unmeasured)| unmeasured),
                Some(!dense),
                "dense={dense}: the MTP session's draft KV and verify graphs need a runtime item, \
                 measured on the q38 ledger and flagged unmeasured where only that stands in\n{}",
                boot.plan.render()
            );
        }
        std::fs::remove_dir_all(&dir).unwrap();
    }
}

#[cfg(all(test, feature = "cuda"))]
mod qwen3_8_routing_pins_mirror_release_keys_no_shipped_config_json_was_diffed {
    use super::*;

    fn three_linear_then_one_full(n: usize) -> String {
        (0..n)
            .map(|i| {
                if (i + 1) % 4 == 0 {
                    "\"full_attention\""
                } else {
                    "\"linear_attention\""
                }
            })
            .collect::<Vec<_>>()
            .join(", ")
    }

    pub(crate) fn qwen38_27b_config_json() -> String {
        format!(
            r#"{{
  "architectures": ["Qwen3_5ForConditionalGeneration"],
  "model_type": "qwen3_5",
  "tie_word_embeddings": false,
  "transformers_version": "4.57.3",
  "text_config": {{
    "model_type": "qwen3_5_text",
    "hidden_size": 5120,
    "num_hidden_layers": 64,
    "num_attention_heads": 20,
    "num_key_value_heads": 4,
    "head_dim": 256,
    "intermediate_size": 17408,
    "full_attention_interval": 4,
    "mamba_ssm_dtype": "float32",
    "output_gate_type": "swish",
    "vocab_size": 248320,
    "max_position_embeddings": 262144,
    "rms_norm_eps": 1e-06,
    "attn_output_gate": true,
    "linear_num_key_heads": 16,
    "linear_num_value_heads": 48,
    "linear_key_head_dim": 128,
    "linear_value_head_dim": 128,
    "linear_conv_kernel_dim": 4,
    "eos_token_id": 248044,
    "rope_parameters": {{"rope_theta": 10000000.0, "partial_rotary_factor": 0.25, "rope_type": "default"}},
    "layer_types": [{}]
  }},
  "vision_config": {{"hidden_size": 1152, "depth": 27, "patch_size": 16, "temporal_patch_size": 2}}
}}"#,
            three_linear_then_one_full(64)
        )
    }

    fn qwen38_flagship_config_json() -> String {
        format!(
            r#"{{
  "architectures": ["Qwen3_5MoeForCausalLM"],
  "model_type": "qwen3_5_moe_text",
  "attention_bias": false,
  "attn_output_gate": true,
  "output_gate_type": "swish",
  "mamba_ssm_dtype": "float32",
  "full_attention_interval": 4,
  "head_dim": 256,
  "hidden_size": 8192,
  "num_hidden_layers": 96,
  "num_attention_heads": 32,
  "num_key_value_heads": 4,
  "num_experts": 512,
  "num_experts_per_tok": 10,
  "moe_intermediate_size": 2048,
  "shared_expert_intermediate_size": 2048,
  "router_aux_loss_coef": 0.001,
  "mtp_num_hidden_layers": 1,
  "mtp_use_dedicated_embeddings": true,
  "linear_num_key_heads": 16,
  "linear_num_value_heads": 128,
  "linear_key_head_dim": 128,
  "linear_value_head_dim": 128,
  "linear_conv_kernel_dim": 4,
  "max_position_embeddings": 262144,
  "rms_norm_eps": 1e-06,
  "rope_parameters": {{"partial_rotary_factor": 0.25, "rope_theta": 10000000.0, "rope_type": "default"}},
  "tie_word_embeddings": false,
  "transformers_version": "4.57.3",
  "vocab_size": 248320,
  "layer_types": [{}]
}}"#,
            three_linear_then_one_full(96)
        )
    }

    #[test]
    fn the_27b_is_dense_behind_the_opt_in_and_the_flagship_routes_moe_either_way() {
        for cfg in [
            qwen38_27b_config_json(),
            qwen38_27b_config_json().replace(
                "\"architectures\": [\"Qwen3_5ForConditionalGeneration\"],",
                "",
            ),
        ] {
            let err = detect_family_with_dense_cuda_serve(&cfg, false)
                .expect_err("qwen3.8-27B must be refused by default exactly like qwen3.5-dense");
            assert_eq!(
                format!("{err:#}"),
                crate::oapi::backend_select::QWEN35_DENSE_NO_CUDA,
                "the 27B declares the same model_type qwen3_5 trunk as Qwen3.5-9B, so the \
                 refusal must stay backend_select::QWEN35_DENSE_NO_CUDA verbatim and keep \
                 naming the wgpu serving path"
            );
            assert_eq!(
                detect_family_with_dense_cuda_serve(&cfg, true).expect(
                    "NV_QWEN35_DENSE_CUDA_SERVE=1 must route qwen3.8-27B onto the same dense \
                     arm that serves Qwen3.5-9B"
                ),
                ModelFamily::Qwen3_5Moe
            );
        }
        let dense = qwen38_27b_config_json();
        let flagship = qwen38_flagship_config_json();
        assert!(
            qwen3_5_config_declares_dense_ffn(&dense),
            "the 27B nests intermediate_size and none of nv_models::qwen3_5_moe::MOE_ONLY_KEYS \
             under text_config, so the opt-in arm must build LayerFfn::Dense for it"
        );
        assert!(
            !qwen3_5_config_declares_dense_ffn(&flagship),
            "the flagship declares num_experts=512 at top level and has no text_config; the \
             dense predicate answering true would build 96 dense Mlps out of expert weights"
        );
        nv_models::qwen3_5_moe::Qwen3_5DenseConfig::from_hf_json_str(&dense)
            .expect("the dense arm's parser must accept the 27B shape the router calls dense");
        nv_models::qwen3_5_moe::Qwen3MoeConfig::from_hf_json_str(&dense)
            .expect_err("the MoE parser must keep rejecting the dense 27B");
        {
            let full = qwen38_flagship_config_json();
            let template_only = full.replace("\"architectures\": [\"Qwen3_5MoeForCausalLM\"],", "");
            assert!(template_only.contains("\"model_type\": \"qwen3_5_moe_text\""));
            for cfg in [full, template_only] {
                for dense_gate in [false, true] {
                    assert_eq!(
                        detect_family_with_dense_cuda_serve(&cfg, dense_gate).expect(
                            "a qwen3_5_moe_text checkpoint must never fall to the qwen3 catch-all: \
                             flat nv_models::qwen3::Qwen3Config has no linear-attention mixer and \
                             dies on the missing flat fields"
                        ),
                        ModelFamily::Qwen3_5Moe,
                        "both the Qwen3_5MoeForCausalLM architectures scan and the bare \
                         model_type qwen3_5_moe_text prefix must answer the MoE family, \
                         independent of the dense opt-in"
                    );
                }
            }
        }
    }

}

#[cfg(all(test, feature = "cuda"))]
mod a_serving_arm_is_one_decision_and_a_subset_of_its_keys_is_unreachable {
    use super::qwen3_8_routing_pins_mirror_release_keys_no_shipped_config_json_was_diffed::qwen38_27b_config_json;
    use super::*;
    use std::collections::HashMap;

    fn qwen35_9b_config_json() -> String {
        let layer_types = (0..32)
            .map(|i| {
                if (i + 1) % 4 == 0 {
                    "\"full_attention\""
                } else {
                    "\"linear_attention\""
                }
            })
            .collect::<Vec<_>>()
            .join(", ");
        format!(
            r#"{{
  "architectures": ["Qwen3_5ForConditionalGeneration"],
  "model_type": "qwen3_5",
  "tie_word_embeddings": false,
  "text_config": {{
    "model_type": "qwen3_5_text",
    "hidden_size": 4096,
    "num_hidden_layers": 32,
    "num_attention_heads": 16,
    "num_key_value_heads": 4,
    "head_dim": 256,
    "intermediate_size": 12288,
    "full_attention_interval": 4,
    "vocab_size": 248320,
    "max_position_embeddings": 262144,
    "rms_norm_eps": 1e-06,
    "attn_output_gate": true,
    "linear_num_key_heads": 16,
    "linear_num_value_heads": 32,
    "linear_key_head_dim": 128,
    "linear_value_head_dim": 128,
    "linear_conv_kernel_dim": 4,
    "eos_token_id": 248044,
    "rope_parameters": {{"rope_theta": 10000000.0, "partial_rotary_factor": 0.25, "rope_type": "default"}},
    "layer_types": [{layer_types}]
  }}
}}"#
        )
    }

    const GEMMA4_NVFP4_CFG: &str = r#"{"architectures": ["Gemma4ForConditionalGeneration"],
      "model_type": "gemma4",
      "quantization_config": {"format": "nvfp4-pack-quantized",
        "config_groups": {"group_0": {"targets": ["Linear"],
          "weights": {"num_bits": 4, "type": "float", "group_size": 16, "symmetric": true, "strategy": "tensor_group"}}}}}"#;

    const GEMMA4_BF16_CFG: &str =
        r#"{"architectures": ["Gemma4ForConditionalGeneration"], "model_type": "gemma4"}"#;

    #[test]
    fn arms_are_all_or_nothing_refuse_disagreeing_keys_and_the_gemv_rescue_is_mandatory() {
        for keys in [
            &GEMMA4_RECORD_ARM_KEYS_ARE_ARMED_TOGETHER_OR_NOT_AT_ALL[..],
            &QWEN35_DENSE_9B_ARM_KEYS_ARE_ARMED_TOGETHER_OR_NOT_AT_ALL[..],
        ] {
            for arm in [
                ServingArm::ArmEveryKey,
                ServingArm::CheckpointOutsideTheMeasuredBasis,
            ] {
                let armed = keys_armed_by_this_one_decision(arm, keys);
                assert!(
                    armed.is_empty() || armed.len() == keys.len(),
                    "{arm:?} armed {} of {} keys: a subset of a measured stack is a \
                     configuration nothing measured, and for gemma4 the specific subset \
                     NV_ATTN_PROJ_QUANT-without-NV_NVFP4_M1_GEMV is a quality REJECT",
                    armed.len(),
                    keys.len()
                );
            }
        }
        {
            let held: HashMap<&str, &str> =
                HashMap::from([("NV_ATTN_PROJ_QUANT", "nvfp4"), ("NV_NVFP4_M1_GEMV", "0")]);
            let conflicts = arm_keys_the_operator_already_set_to_something_else(
                &GEMMA4_RECORD_ARM_KEYS_ARE_ARMED_TOGETHER_OR_NOT_AT_ALL,
                |k| held.get(k).map(|v| v.to_string()),
            );
            assert_eq!(
                conflicts.iter().map(|(k, _, _)| *k).collect::<Vec<_>>(),
                vec!["NV_NVFP4_M1_GEMV"],
                "a key the operator pinned away from the measured value is a refusal, not a silent \
                 overwrite and not a partial arm"
            );
            assert!(arm_keys_the_operator_already_set_to_something_else(
                &GEMMA4_RECORD_ARM_KEYS_ARE_ARMED_TOGETHER_OR_NOT_AT_ALL,
                |_| None
            )
            .is_empty());
            assert!(arm_keys_the_operator_already_set_to_something_else(
                &GEMMA4_RECORD_ARM_KEYS_ARE_ARMED_TOGETHER_OR_NOT_AT_ALL,
                |k| GEMMA4_RECORD_ARM_KEYS_ARE_ARMED_TOGETHER_OR_NOT_AT_ALL
                    .iter()
                    .find(|(n, _)| *n == k)
                    .map(|(_, v)| v.to_string())
            )
            .is_empty());
        }
        {
            for raw in [
                "nvfp4", "1", "on", "true", "yes", "all", "fp4", "qkv+o", "qkv", "o",
            ] {
                assert!(
                    gemma4_attn_nvfp4_without_the_m1_gemv_rescue(Some(raw), None),
                    "NV_ATTN_PROJ_QUANT={raw} without NV_NVFP4_M1_GEMV=1 is the +0.748 ppl \
                     configuration and must never boot"
                );
                assert!(gemma4_attn_nvfp4_without_the_m1_gemv_rescue(
                    Some(raw),
                    Some("0")
                ));
                assert!(!gemma4_attn_nvfp4_without_the_m1_gemv_rescue(
                    Some(raw),
                    Some("1")
                ));
            }
            for raw in [None, Some("off"), Some("0"), Some("bf16")] {
                assert!(!gemma4_attn_nvfp4_without_the_m1_gemv_rescue(raw, None));
            }
            assert!(
                !gemma4_attn_nvfp4_without_the_m1_gemv_rescue(Some("fp8"), None),
                "the fp8 scheme is a different route the m=1 nvfp4 gemv does not serve, and it \
                 refuses at model load on sm_120 with its own message"
            );
        }
    }

    #[test]
    fn each_arm_reaches_only_the_checkpoints_and_trunks_it_was_measured_on() {
        assert_eq!(
            GEMMA4_RECORD_ARM_KEYS_ARE_ARMED_TOGETHER_OR_NOT_AT_ALL,
            [
                ("NV_ATTN_PROJ_QUANT", "nvfp4"),
                ("NV_NVFP4_M1_GEMV", "1"),
                ("NV_G4_LMHEAD_I8", "1"),
            ],
            "these three keys and these values are the arm the 27.876/29.472 -> 19.419/21.313 \
             pairs and the 16.290 / 16.344 ppl rows were measured on"
        );
        assert!(gemma4_checkpoint_declares_nvfp4(GEMMA4_NVFP4_CFG));
        assert!(!gemma4_checkpoint_declares_nvfp4(GEMMA4_BF16_CFG));
        assert_eq!(
            serving_arm_from(gemma4_checkpoint_declares_nvfp4(GEMMA4_NVFP4_CFG)),
            ServingArm::ArmEveryKey
        );
        assert_eq!(
            serving_arm_from(gemma4_checkpoint_declares_nvfp4(GEMMA4_BF16_CFG)),
            ServingArm::CheckpointOutsideTheMeasuredBasis,
            "attn_proj_quant_mode ignores NV_ATTN_PROJ_QUANT unless the checkpoint declares \
             QuantScheme::Nvfp4, so arming a bf16 gemma4 would move only the int8 lm_head, on a \
             checkpoint class nothing measured"
        );
        {
            assert!(
                qwen35_dense_trunk_is_the_measured_9b(&qwen35_9b_config_json()),
                "hidden 4096 / 32 layers / intermediate 12288 is the Qwen3.5-9B trunk the eleven \
                 arms were measured on"
            );
            assert!(
                !qwen35_dense_trunk_is_the_measured_9b(&qwen38_27b_config_json()),
                "qwen3.8-27B (5120/64/17408) is NOT the Qwen3.5-9B trunk, so the eleven-arm 9B stack \
                 stops here; the 27B lands in the same dense arm but takes its OWN measured stack via \
                 qwen35_dense_trunk_is_the_measured_27b"
            );
            assert!(
                qwen35_dense_trunk_is_the_measured_27b(&qwen38_27b_config_json()),
                "hidden 5120 / 64 layers / intermediate 17408 is the qwen3.8-27B trunk the \
                 GDN-nvfp4 + M1_GEMV two-key stack was measured on (ppl 9.879->9.818 decode, \
                 11.994->11.964 prefill; +7.7%/+5.4% graphed decode)"
            );
            assert!(
                !qwen35_dense_trunk_is_the_measured_27b(&qwen35_9b_config_json()),
                "the 9B trunk must not trip the 27B GDN-nvfp4 stack"
            );
            assert_eq!(
                serving_arm_from(qwen35_dense_trunk_is_the_measured_27b(&qwen38_27b_config_json())),
                ServingArm::ArmEveryKey,
                "the 27B trunk arms its two-key GDN-nvfp4 stack by default"
            );
        }
    }

    #[test]
    fn every_rationale_names_its_instrument_and_its_cost() {
        for (why, needles) in [
            (
                GEMMA4_NVFP4_SERVES_THE_THREE_KEY_RECORD_ARM,
                vec![
                    "NV_ATTN_PROJ_QUANT",
                    "NV_NVFP4_M1_GEMV",
                    "NV_G4_LMHEAD_I8",
                    "ppl_gemma4::gemma4_chat_wrapped_continuation_teacher_forced_ppl_via_serving_decode",
                    "0.41 family null",
                    "16.290",
                    "16.344",
                    "19.419",
                    "gemma4_verify_forward",
                ],
            ),
            (
                QWEN35_DENSE_9B_OPT_IN_ARMS_ITS_MEASURED_STACK,
                vec![
                    QWEN35_DENSE_CUDA_SERVE_ENV,
                    "ppl_qwen35_dense::qwen35_dense_chat_wrapped_continuation_teacher_forced_ppl_via_serving_decode",
                    "4.833",
                    "4.862",
                    "5.900",
                    "9183",
                    "qwen3.8-27B",
                ],
            ),
            (
                QWEN38_DENSE_27B_NVFP4_GDN_ARMS_ITS_TWO_KEY_STACK,
                vec![
                    "NV_Q38_GDN_NVFP4",
                    "NV_NVFP4_M1_GEMV",
                    "9.879 -> 9.818",
                    "11.994 -> 11.964",
                    "16.780 -> 15.584",
                    "nvfp4_gemv_bf16act",
                ],
            ),
        ] {
            for needle in needles {
                assert!(
                    why.contains(needle),
                    "the notice printed at load is the only durable record of what this default \
                     does and what it cost; it must name {needle}"
                );
            }
        }
    }
}

#[cfg(all(test, feature = "cuda"))]
mod capture_stream_device_policy {
    use super::*;

    const GEMMA4_CFG: &str = r#"{"architectures": ["Gemma4ForCausalLM"], "model_type": "gemma4"}"#;
    const QWEN3_CFG: &str = r#"{"architectures": ["Qwen3ForCausalLM"], "model_type": "qwen3"}"#;

    #[test]
    fn candle_graph_families_are_named_and_only_explicit_laguna_m2_spec_needs_its_stream() {
        for family in [ModelFamily::Gemma4, ModelFamily::Gemma4E4b] {
            assert!(
                graph_body_is_a_candle_forward(family),
                "{family:?} captures a candle forward on a forked stream. Recording that is the \
                 whole point: {A_CANDLE_GRAPH_BODY_NEEDS_THE_DEVICE_STREAM}"
            );
        }
        for family in [
            ModelFamily::Qwen3,
            ModelFamily::Gemma4Moe,
            ModelFamily::Qwen3_5Moe,
            ModelFamily::Laguna,
        ] {
            assert!(!graph_body_is_a_candle_forward(family));
        }
        for family in [
            ModelFamily::Qwen3,
            ModelFamily::Gemma4,
            ModelFamily::Gemma4E4b,
            ModelFamily::Gemma4Moe,
            ModelFamily::Qwen3_5Moe,
            ModelFamily::Laguna,
        ] {
            assert!(
                !device_stream_flip_is_cleared_for(family),
                "{family:?} was cleared for Device::new_cuda_with_stream. Clearing a family means \
                 claiming its EAGER work survives a non-default stream, not just its capture. \
                 {THE_GEMMA4_DEVICE_STREAM_FLIP_IS_BLOCKED_BY_A_MEASURED_SERVING_BREAK}"
            );
        }
        {
            for selector in ["", "0", "1", "m2", "m2-spec"] {
                assert_eq!(
                    laguna_spec_batch_needs_device_stream(ModelFamily::Laguna, selector),
                    selector == "m2-spec"
                );
                assert!(!laguna_spec_batch_needs_device_stream(
                    ModelFamily::Qwen3,
                    selector
                ));
                assert!(!laguna_spec_batch_needs_device_stream(
                    ModelFamily::Gemma4,
                    selector
                ));
            }
        }
    }

    #[test]
    fn the_device_selection_includes_only_the_named_captured_routes() {
        for (family, cfg) in [
            (ModelFamily::Qwen3, QWEN3_CFG),
            (ModelFamily::Gemma4, GEMMA4_CFG),
            (ModelFamily::Gemma4E4b, GEMMA4_CFG),
            (ModelFamily::Gemma4Moe, GEMMA4_CFG),
            (ModelFamily::Qwen3_5Moe, QWEN3_CFG),
            (ModelFamily::Laguna, QWEN3_CFG),
        ] {
            assert_eq!(
                capture_needs_the_device_stream(family, cfg),
                qwen_graph_decode_selected(family, cfg)
                    || qwen38_batch_lanes_boot_selected(family, cfg)
                    || qwen36_batch_lanes_boot_selected(family, cfg)
                    || qwen36_mtp_captured_serving_selected(family, cfg)
                    || laguna_spec_batch_needs_device_stream(family, &std::env::var("NV_LAGUNA_BATCH_GRAPH").unwrap_or_default())
                    || laguna_prepared_b2_graph_captures_the_candle_forward_on_the_model_stream(family),
                "{family:?}: try_load_inner must keep choosing Device::new_cuda_with_stream on \
                 exactly the qwen-graph condition plus the NV_Q38_BATCH dense lane opt-in plus \
                 the NV_Q36_BATCH moe lane opt-in plus the NV_DRAFTER=mtp captured-verify \
                 serving default plus explicit Laguna m2 or m2-spec. Task #63 records grouped MoE on a \
                 non-default stream producing all-NaN logits, and gemma4 prefill produced all \
                 -inf on the same axis; the permitted widenings are \
                 qwen38_batch_lanes_boot_selected, whose serving measurement is \
                 rust/tests/qwen38_batch_serving_e2e.rs (solo MTP and batch groups byte-match \
                 their flag-off references on the stream-mode device), \
                 qwen36_batch_lanes_boot_selected, whose serving measurement is \
                 rust/tests/qwen36_http_concurrency_ladder.rs (lane groups byte-match their \
                 flag-off eager references on the stream-mode device under the rowwise=moe pin \
                 plus twin basis), and \
                 qwen36_mtp_captured_serving_selected, whose serving measurement is \
                 rust/tests/qwen36_mtp_serving_e2e.rs (byte-identical greedy repeats through \
                 the captured verify lane on the stream-mode device)"
            );
        }
    }

    #[test]
    fn both_rationales_name_what_a_future_lane_has_to_re_measure() {
        for needle in [
            "Device::new_cuda_with_stream",
            "GraphedGemma4Decoder",
            "Gemma4BatchGraphFamily",
            "gemma4_e4b.rs",
            "laguna_graph.rs",
        ] {
            assert!(
                A_CANDLE_GRAPH_BODY_NEEDS_THE_DEVICE_STREAM.contains(needle),
                "the device-stream rationale must name {needle}"
            );
        }
        for needle in [
            "-inf",
            "CUDA_ERROR_ILLEGAL_ADDRESS",
            "tool_calling_e2e",
            "#63",
            "3653",
        ] {
            assert!(
                THE_GEMMA4_DEVICE_STREAM_FLIP_IS_BLOCKED_BY_A_MEASURED_SERVING_BREAK
                    .contains(needle),
                "the refusal must name {needle}: a refusal without its measurement is an opinion"
            );
        }
    }
}

#[cfg(test)]
mod bos_and_eos_resolution_tests {
    use super::*;

    fn checkpoint_dir(tag: &str, files: &[(&str, &str)]) -> PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "nv_bos_resolution_{}_{tag}_{:?}",
            std::process::id(),
            std::thread::current().id()
        ));
        std::fs::remove_dir_all(&dir).ok();
        std::fs::create_dir_all(&dir).unwrap();
        for (name, body) in files {
            std::fs::write(dir.join(name), body).unwrap();
        }
        dir
    }

    #[test]
    fn bos_and_eos_resolve_from_text_config_and_generation_config_or_refuse() {
        let dir = checkpoint_dir(
            "nested",
            &[(
                "config.json",
                r#"{"text_config":{"bos_token_id":2,"eos_token_id":[1,106]}}"#,
            )],
        );
        assert_eq!(
            bos_id_from_dir(&dir),
            Some(2),
            "multimodal checkpoints declare the text tower's ids under text_config; a top-level \
             only parser reads None there and the caller then invents one"
        );
        assert_eq!(eos_ids_from_dir(&dir).unwrap(), vec![1, 106]);
        std::fs::remove_dir_all(&dir).ok();
        {
            let dir = checkpoint_dir(
                "precedence",
                &[
                    ("config.json", r#"{"bos_token_id":2,"eos_token_id":1}"#),
                    (
                        "generation_config.json",
                        r#"{"bos_token_id":105,"eos_token_id":[1,106]}"#,
                    ),
                ],
            );
            assert_eq!(bos_id_from_dir(&dir), Some(105));
            assert_eq!(
                eos_ids_from_dir(&dir).unwrap(),
                vec![1, 106],
                "generation_config.json is where a checkpoint states its serving stops; config.json \
                 alone loses the second stop id"
            );
            std::fs::remove_dir_all(&dir).ok();
        }
        {
            let dir = checkpoint_dir("silent", &[("config.json", r#"{"model_type":"gemma4"}"#)]);
            assert_eq!(bos_id_from_dir(&dir), None);
            let err = eos_ids_from_dir(&dir).expect_err("must be refused");
            let msg = format!("{err:#}");
            assert!(
                msg.contains("no stop condition") && msg.contains("[1, 106]"),
                "the refusal must name what it refuses to invent: {msg}"
            );
            std::fs::remove_dir_all(&dir).ok();
        }
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn pairs_read_off_disk_make_the_position_0_rule_refuse_laguna_and_splice_gemma() {
        let dir = checkpoint_dir(
            "laguna",
            &[(
                "config.json",
                r#"{"model_type":"laguna","bos_token_id":2,"eos_token_id":[2,24]}"#,
            )],
        );
        let bos = bos_id_from_dir(&dir);
        let eos = eos_ids_from_dir(&dir).unwrap();
        assert_eq!((bos, eos.as_slice()), (Some(2), [2u32, 24].as_slice()));

        let mut ids = vec![9204u32, 610, 24];
        let before = ids.clone();
        let head = splice_bos_at_position_0_only(&mut ids, bos, &eos);
        assert_eq!(
            head,
            PromptHead::RefusedToSpliceAnIdTheCheckpointAlsoDeclaresEos,
            "with both sides resolved from the checkpoint the rule must be able to refuse; \
             against the fabricated pair it shipped with (bos 2 vs eos [1, 106]) it never could"
        );
        assert_eq!(ids, before);
        assert!(
            eos.contains(&2),
            "{BOS_IS_A_POSITION_0_ROLE_AND_THE_SAME_ID_LATER_IS_A_STOP}"
        );
        std::fs::remove_dir_all(&dir).ok();

        let dir = checkpoint_dir(
            "qwen",
            &[
                ("config.json", r#"{"model_type":"qwen3"}"#),
                (
                    "generation_config.json",
                    r#"{"bos_token_id":248044,"eos_token_id":[248044,248046]}"#,
                ),
            ],
        );
        let bos = bos_id_from_dir(&dir);
        let eos = eos_ids_from_dir(&dir).unwrap();
        let mut ids = vec![151644u32, 872, 198];
        let head = splice_bos_at_position_0_only(&mut ids, bos, &eos);
        assert_eq!(
            head,
            PromptHead::RefusedToSpliceAnIdTheCheckpointAlsoDeclaresEos
        );
        assert!(
            !eos.contains(&ids[0]),
            "{BOS_IS_A_POSITION_0_ROLE_AND_THE_SAME_ID_LATER_IS_A_STOP}"
        );
        std::fs::remove_dir_all(&dir).ok();
        {
            let dir = checkpoint_dir(
                "gemma",
                &[
                    ("config.json", r#"{"model_type":"gemma4"}"#),
                    (
                        "generation_config.json",
                        r#"{"bos_token_id":2,"eos_token_id":[1,106]}"#,
                    ),
                ],
            );
            let bos = bos_id_from_dir(&dir);
            let eos = eos_ids_from_dir(&dir).unwrap();
            let mut ids = vec![105u32, 2364, 107];
            let head = splice_bos_at_position_0_only(&mut ids, bos, &eos);
            assert_eq!(head, PromptHead::EnginePrependedBosAtPosition0);
            assert_eq!(ids, vec![2, 105, 2364, 107]);
            std::fs::remove_dir_all(&dir).ok();
        }
    }

}

#[cfg(test)]
mod kv_ceiling_for_unplanned_families_tests {
    use super::*;
    use crate::oapi::vram_plan::{kv_binding_from, per_engine_kv_env_for_family};

    #[test]
    fn the_kv_ceiling_defaults_to_8192_and_each_unplanned_family_lifts_through_its_own_env() {
        for mpe in [0usize, 1, 4096, 8192, 32768, 262144] {
            assert_eq!(
                kv_max_seq_len_resolved(None, mpe),
                mpe.min(8192),
                "mpe={mpe} with no env must resolve exactly as the old min(mpe, 8192)"
            );
        }
        assert_eq!(kv_max_seq_len_resolved(Some(196608), 262144), 196608);
        assert_eq!(
            kv_max_seq_len_resolved(Some(500_000), 262144),
            262144,
            "the env lift clamps to max_position_embeddings, the operator cannot exceed the model"
        );
        assert_eq!(kv_max_seq_len_resolved(Some(0), 262144), 8192);
        {
            for family in ["qwen3", "gemma4-moe", "laguna", "gpt-oss", "omni"] {
                let env = per_engine_kv_env_for_family(family)
                    .unwrap_or_else(|| panic!("{family} lost its per-engine kv env"));
                let bound = kv_binding_from(Some(env), Some("196608"), None);
                assert_eq!(bound, Some((196608, env)), "{family}");
                assert_eq!(
                    kv_max_seq_len_resolved(bound.map(|(n, _)| n), 262144),
                    196608,
                    "{family} must reach 196k via {env} while cobooted engines keep their default"
                );
                let global_only = kv_binding_from(Some(env), None, Some("32768"));
                assert_eq!(
                    kv_max_seq_len_resolved(global_only.map(|(n, _)| n), 262144),
                    32768,
                    "{family} still inherits the global NV_KV_MAX_SEQ_LEN when its own env is unset"
                );
            }
        }
    }

}
