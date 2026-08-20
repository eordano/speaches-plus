use super::*;

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
}

pub struct NvEngineChat {
    pub(crate) inner: NvEngineInner,
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
}

#[cfg(feature = "cuda")]
#[derive(Clone)]
pub(crate) enum QwenMoeShared {
    Eager(Arc<tokio::sync::Mutex<nv_models::qwen3_5_moe::Qwen3Moe>>),
    Graphed(Arc<tokio::sync::Mutex<nv_models::graph_engine::GraphedQwen3Moe>>),
    Batch(Arc<Qwen38BatchScheduler>),
}

#[cfg(feature = "cuda")]
#[derive(Clone)]
pub(crate) struct LagunaShared(Arc<nv_models::laguna::Laguna>);
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
    pub(crate) jobs:
        std::sync::Mutex<std::sync::mpsc::Sender<nv_models::laguna_serve::SpecServeJob>>,
    pub(crate) max_seq: usize,
    pub(crate) num_spec: usize,
}

#[cfg(feature = "cuda")]
pub(crate) fn build_laguna_spec_serve(
    model: &LagunaShared,
    kv_max_seq_len: usize,
) -> Option<Arc<LagunaSpecServe>> {
    if !laguna_serve_spec_enabled() {
        return None;
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
    let (job_tx, job_rx) = std::sync::mpsc::channel();
    let (ready_tx, ready_rx) = std::sync::mpsc::channel();
    let shared = model.clone();
    let draft_dir = draft_enabled.then(laguna_dflash_dir).flatten();
    if draft_enabled && draft_dir.is_none() {
        tracing::warn!("NV_LAGUNA_SERVE_DRAFT=1 but no dflash snapshot found; spec serving will run M=1 step graph only");
    }
    let spawned = std::thread::Builder::new()
        .name("laguna-spec-serve".into())
        .spawn(move || {
            let target = shared.0.clone();
            let device = target.device().clone();
            let draft = draft_dir.and_then(|dir| {
                match nv_models::laguna_serve::load_dflash_draft(&dir, &device) {
                    Ok(d) => Some(d),
                    Err(e) => {
                        tracing::warn!(error = %format!("{e:#}"), "laguna dflash draft load failed; spec serving will run M=1 step graph only");
                        None
                    }
                }
            });
            if let Err(e) = nv_models::laguna_serve::spec_serve_loop(
                target, draft, num_spec, max_seq, job_rx, ready_tx,
            ) {
                tracing::warn!(error = %format!("{e:#}"), "laguna spec serve loop exited");
            }
        });
    if spawned.is_err() {
        tracing::warn!("failed to spawn laguna spec serve thread");
        return None;
    }
    match ready_rx.recv() {
        Ok(has_draft) => {
            tracing::info!(
                has_draft,
                num_spec,
                max_seq,
                "laguna spec serving engine ready"
            );
            Some(Arc::new(LagunaSpecServe {
                jobs: std::sync::Mutex::new(job_tx),
                max_seq,
                num_spec,
            }))
        }
        Err(_) => {
            tracing::warn!("laguna spec serving engine init failed; using the per-request path");
            None
        }
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
pub(crate) fn qwen_moe_grouped_enabled() -> bool {
    !matches!(
        std::env::var("NV_QWEN_MOE_GROUPED").ok().as_deref(),
        Some("0") | Some("off")
    )
}

#[cfg(feature = "cuda")]
pub(crate) const A_CANDLE_GRAPH_BODY_NEEDS_THE_DEVICE_STREAM: &str = "Device::new_cuda_with_stream \
     is what a family whose CUDA-graph body is a candle-core forward would need, because candle \
     launches only on candle_core::CudaDevice::cuda_stream() and \
     nv_models::gemma4_batch_graph::capture_stream::CaptureStream therefore has to capture on that \
     very stream; on Device::new_cuda it is the legacy NULL stream, which cuStreamBeginCapture \
     refuses outright, so the engine forks and any candle op in the captured body escapes the \
     capture. Gemma4 qualifies through nv_models::gemma4_graph::GraphedGemma4Decoder, \
     GraphedGemma4Verify and nv_models::gemma4_batch_graph::Gemma4BatchGraphFamily -- they \
     capture Gemma4::forward_with_cache_into, forward_verify_dev and a candle layer loop. \
     Gemma4E4b also qualifies (gemma4_e4b.rs captures Gemma4E4b::forward_step_fast_body). Laguna \
     does not: laguna_graph.rs does the candle handoff as a memcpy_dtod outside the capture and \
     its captured body is raw, so forking is correct there. Gemma4Moe and Qwen3 build no graph";

#[cfg(feature = "cuda")]
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
pub(crate) fn capture_needs_the_device_stream(family: ModelFamily, raw_cfg: &str) -> bool {
    qwen_graph_decode_selected(family, raw_cfg)
        || qwen38_batch_lanes_boot_selected(family, raw_cfg)
        || (graph_body_is_a_candle_forward(family) && device_stream_flip_is_cleared_for(family))
}

#[cfg(feature = "cuda")]
pub(crate) fn qwen38_batch_lanes_boot_selected(family: ModelFamily, raw_cfg: &str) -> bool {
    matches!(family, ModelFamily::Qwen3_5Moe)
        && qwen3_5_config_declares_dense_ffn(raw_cfg)
        && qwen3_5_dense_cuda_serve_enabled()
        && nv_models::qwen3_5_moe::qwen38_batch::nv_q38_batch_env_opt_in_nv_q38_batch_1_the_serving_loop_routes_batch_xor_spec_per_request_group()
}

#[cfg(feature = "cuda")]
pub(crate) const QWEN35_DENSE_CUDA_SERVE_ENV: &str = "NV_QWEN35_DENSE_CUDA_SERVE";

#[cfg(feature = "cuda")]
pub(crate) const QWEN35_DENSE_CUDA_OPT_IN_MATCHES_THE_WGPU_ORACLE_AND_STAYS_OPT_IN: &str = "qwen3.5-dense cuda serving is wired and opted into per boot: with NV_QWEN35_DENSE_CUDA_SERVE=1 detect_family stops refusing and try_load parses the checkpoint with nv_models::qwen3_5_moe::Qwen3_5DenseConfig (the same parser the wgpu decoder uses, top-level fields read out of text_config) and builds it through nv_models::qwen3_5_moe::Qwen3Moe::from_loader_dense_quantized, which gives every linear_attention entry of layer_types a LayerMixer::Linear over nv_layers::linear_attn::LinearAttention and every layer a LayerFfn::Dense over nv_layers::mlp::Mlp. Every trunk component this uses -- the output-gated attention, the partial rope, the GDN linear-attention mixer, the nvfp4 linears -- is byte-for-byte the one the qwen3.5-moe cuda path already serves, because Qwen3.5-9B and Qwen3.5-35B-A3B declare the same head_dim, attn_output_gate, partial_rotary_factor and linear_* block; the ONLY surface this path adds over the served MoE path is LayerFfn::Dense. That surface is now executed and compared against nv_models::qwen3_5_dense_wgpu, the real-weights-verified decoder for this family, by the suite covering the dense cuda serving arm (rust/tests/qwen35_dense_cuda_serving_ab.rs, NV_QWEN35_DENSE_CUDA_SERVE_TEST=1): on ig1/Qwen3.5-9B-NVFP4 snapshot 3b9e07b0 the two backends return the same 7 tokens byte for byte through the checkpoint's own chat template, cuda repeats itself across two requests and an engine reload, and a 512-token thinking trace stays on subject while separating from the oracle at character 225, which is NVFP4 accumulation order and not a decode defect. It stays opt-in because that is two prompts on one checkpoint rather than a parity measurement, and because flipping it would move backend_select's auto choice for this whole checkpoint class onto a backend that holds more VRAM for the same weights. Without the variable, detect_family keeps refusing with backend_select::QWEN35_DENSE_NO_CUDA and the default behavior is unchanged";

#[cfg(feature = "cuda")]
pub(crate) const GPTOSS_CUDA_OPT_IN_IS_A_DEQUANT_RUNG_AND_WGPU_STAYS_THE_GPT_OSS_DEFAULT: &str = "gpt-oss cuda serving is wired and opted into per boot: with NV_GPTOSS_CUDA_SERVE=1 detect_family answers ModelFamily::GptOss instead of refusing, and try_load parses the checkpoint with nv_models::gpt_oss::GptOssConfig -- the very parser nv_models::gpt_oss_wgpu uses, so the two backends cannot disagree about layer_types, sliding_window, swiglu_limit or the YaRN block -- and builds nv_models::gpt_oss_cuda::GptOssCuda. That decoder adds exactly two things cuda did not have. First, mxfp4: every expert tensor is dequantized to bf16 at load through nv_quant::mxfp4::Mxfp4Tensor::dequantize, which is the same host semantics the wgpu mxfp4 GEMV is already pinned against, and which is lossless into bf16 because an e2m1 value times an e8m0 block scale needs one mantissa bit. Second, attention sinks: nv_layers::attn::sdpa_with_sinks appends the learned per-head sink as one more score column and one all-zero value row, which is algebraically the fold gow_attn.wgsl and gow_prefill.wgsl perform as `m = max(red[0], sink); z = red[0] + exp(sink - m)`. candle_flash_attn takes no sink argument, so this decoder is eager by construction and the alternating sliding_window=128 layers are masked in the same builder rather than by flash_attn_windowed. It stays opt-in on cost, not correctness: dequant-to-bf16 puts openai/gpt-oss-20b at about 41.8 GB resident against about 13.7 GB for a native mxfp4 path, and eager scoring costs heads*rows*context f32 of transient scratch per prefill chunk that the wgpu decoder never materializes. Flipping it would move backend_select's auto choice for this checkpoint class onto the backend that holds three times the weights and cannot yet stream its softmax. Without the variable, detect_family keeps refusing with backend_select::GPTOSS_NO_CUDA_WITHOUT_THE_OPT_IN and the default behavior is unchanged";

#[cfg(feature = "cuda")]
pub(crate) fn gpt_oss_cuda_serve_enabled() -> bool {
    crate::oapi::backend_select::gpt_oss_cuda_serve_enabled()
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
        && qwen_moe_grouped_enabled()
}

#[cfg(feature = "cuda")]
pub(crate) fn build_qwen_moe_dispatch(
    model: &nv_models::qwen3_5_moe::Qwen3Moe,
) -> Option<Arc<nv_models::qwen3_5_moe::GroupedMoeDispatch>> {
    if !qwen_moe_grouped_enabled() {
        eprintln!(
            "[qwen3.6-moe] MoE dispatch: HOST (NV_QWEN_MOE_GROUPED={:?} opted out of the grouped default)",
            std::env::var("NV_QWEN_MOE_GROUPED").ok()
        );
        return None;
    }
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
                    "[qwen3.6-moe] MoE dispatch: GROUPED (default; NV_QWEN_MOE_GROUPED=0 for host) -- WARNING: grouped expert copy holds {mib:.0} MiB extra VRAM on top of the host-path weights"
                ),
                Some(mib) => eprintln!(
                    "[qwen3.6-moe] MoE dispatch: GROUPED (default; NV_QWEN_MOE_GROUPED=0 for host), grouped expert copy +{mib:.0} MiB VRAM"
                ),
                None => eprintln!(
                    "[qwen3.6-moe] MoE dispatch: GROUPED (default; NV_QWEN_MOE_GROUPED=0 for host), VRAM delta unavailable"
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
pub(crate) struct NvEngineInner {
    pub(crate) model_id: String,
    pub(crate) family: ModelFamily,
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
    pub(crate) qwen_mtp: Option<Arc<nv_specdecode::qwen38_mtp::Qwen38DenseMtpHead>>,
    pub(crate) mm_towers: Option<Arc<crate::oapi::chat_multimodal::Gemma4MmTowers>>,
}

#[cfg(not(feature = "cuda"))]
pub(crate) struct NvEngineInner {
    pub(crate) _unused: (),
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
        "{} is a GGUF checkpoint dir ({}): the CUDA chat engine has no GGUF load path -- it \
         detects the family from config.json and reads weights through \
         nv_weights::WeightLoader::open_dir, which is safetensors-only, so it never reaches \
         nv_models::gemma4_moe::Gemma4Moe::from_gguf. The gemma4-MoE serving family itself is \
         wired on cuda (LoadedModel::Gemma4Moe + run_sampling_gemma4_moe); only the GGUF \
         checkpoint format is missing. Serve this dir on the wgpu chat engine (feature \
         \"wgpu\"; NV_SERVE_BACKEND=wgpu) via Gemma4MoeWgpu::from_gguf, or convert it to \
         safetensors.",
        model_dir.display(),
        gguf.display()
    );
}

#[cfg(not(feature = "cuda"))]
pub(crate) fn refuse_gguf_checkpoint_dir(_model_dir: &Path) -> anyhow::Result<()> {
    Ok(())
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
    pub fn last_routed_drafter_arm() -> Option<&'static str> {
        last_routed_drafter_arm_name()
    }

    pub fn try_load(model_dir: &Path) -> anyhow::Result<Self> {
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

        Self::try_load_inner(model_dir)
    }

    #[cfg(feature = "cuda")]
    pub(crate) fn try_load_inner(model_dir: &Path) -> anyhow::Result<Self> {
        use anyhow::Context;
        use candle_core::Device;

        let cfg_path = model_dir.join("config.json");
        refuse_gguf_checkpoint_dir(model_dir)?;
        let raw_cfg = std::fs::read_to_string(&cfg_path)
            .with_context(|| format!("read {}", cfg_path.display()))?;
        let family = detect_family(&raw_cfg)
            .with_context(|| format!("detect model family from {}", cfg_path.display()))?;

        let chat_template = crate::oapi::chat_template::ChatTemplate::load(model_dir);
        if chat_template.is_some() {
            tracing::info!(model_dir = %model_dir.display(), "loaded official chat template");
        } else {
            tracing::warn!(model_dir = %model_dir.display(), "no official chat template found; using built-in renderer");
        }

        let qwen3_5_dense = qwen3_5_config_declares_dense_ffn(&raw_cfg);
        let qwen_graph = qwen_graph_decode_selected(family, &raw_cfg);
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

        match family {
            ModelFamily::Qwen3 => {
                let config = nv_models::qwen3::Qwen3Config::from_hf_json_str(&raw_cfg)
                    .with_context(|| format!("parse {}", cfg_path.display()))?;
                let model = nv_models::qwen3::Qwen3::from_loader(config.clone(), &weights, &device)
                    .context("instantiate Qwen3 from weights")?;
                let eos_token_ids = vec![config.eos_token_id];
                let bos_token_id = bos_id_from_dir(model_dir);
                let kv_max_seq_len = kv_max_seq_len_for(config.max_position_embeddings);
                Ok(Self {
                    inner: NvEngineInner {
                        model_id,
                        family,
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

                let kv_max_seq_len = kv_max_seq_len_for_gemma4(config.max_position_embeddings);
                let no_spec = nv_no_spec(std::env::var("NV_NO_SPEC").ok().as_deref());
                let drafter_kind = nv_drafter_kind(std::env::var("NV_DRAFTER").ok().as_deref());
                let target_dims = gemma4_target_dims(&model_id, &config);
                let td = Some(&target_dims);
                let (eagle3, dflash) = if no_spec {
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
                    dflash_spec_requested(no_spec, drafter_kind),
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
                let spec_status = match eagle3_gate(spec_requested, required, drafter_loaded) {
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
                let drafter_charge = DrafterKvCharge::from_env(eagle3_row_elems, dflash_row_elems);
                let kv_max_seq_len = fit_gemma4_kv_max(&config, kv_max_seq_len, drafter_charge);
                let measured_static =
                    enforce_gemma4_vram_budget(&config, kv_max_seq_len, drafter_charge)?;
                let model = Arc::new(model);

                let gemma4_engine = if std::env::var("NV_BATCH_ENGINE").is_ok() {
                    match build_gemma4_batch_engine(model.clone(), device.clone(), kv_max_seq_len) {
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
                let admit_static = {
                    let post = gemma4_engine
                        .is_some()
                        .then(our_process_vram_bytes)
                        .flatten();
                    let gib = |b: Option<u64>| {
                        b.map(|v| format!("{:.2}", v as f64 / (1u64 << 30) as f64))
                            .unwrap_or_else(|| "unknown".into())
                    };
                    if post.is_some() {
                        tracing::info!(
                            static_pre_engine_gib = gib(measured_static),
                            static_post_engine_gib = gib(post),
                            post_engine_used = std::env::var("NV_ADMIT_POST_ENGINE").is_ok(),
                            "gemma4 admission static footprint: the batch engine's paged KV pool \
                             lands between these two measurements"
                        );
                    }
                    if std::env::var("NV_ADMIT_POST_ENGINE").as_deref() != Ok("0") {
                        post.or(measured_static)
                    } else {
                        measured_static
                    }
                };
                crate::oapi::admission::init_gemma4(admit_static, drafter_row_elems);

                Ok(Self {
                    inner: NvEngineInner {
                        model_id,
                        family,
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
                    },
                })
            }
            ModelFamily::Gemma4E4b => {
                if std::env::var("NV_E4B_CUDA_SERVE").as_deref() == Ok("0") {
                    anyhow::bail!(
                        "gemma-4 E4B cuda serving disabled by NV_E4B_CUDA_SERVE=0; unset it to \
                         serve this checkpoint on cuda, or serve it on the wgpu backend"
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
                    },
                })
            }
            ModelFamily::Gemma4Moe => {
                let config = nv_models::gemma4_moe::Gemma4MoeConfig::from_hf_json_str(&raw_cfg)
                    .with_context(|| format!("parse gemma4 moe {}", cfg_path.display()))?;
                let model = nv_models::gemma4_moe::Gemma4Moe::from_loader(
                    config.clone(),
                    &weights,
                    &device,
                )
                .context("instantiate Gemma4Moe from weights")?;
                let eos_token_ids = eos_ids_from_dir(model_dir)?;
                let bos_token_id = bos_id_from_dir(model_dir);
                let kv_max_seq_len = kv_max_seq_len_for(config.base.max_position_embeddings);
                let mm_towers = load_mm_towers_or_serve_text_only(model_dir, &device, &model_id);
                Ok(Self {
                    inner: NvEngineInner {
                        model_id,
                        family,
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
                    "[qwen3.5-dense] decode: EAGER (no graph capture, no grouped MoE), {n_linear}/{} layers are nv_layers::linear_attn::LinearAttention, dense ffn intermediate_size={}",
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
                let kv_max_seq_len = kv_max_seq_len_for(config.max_position_embeddings);
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
                    let plan =
                        nv_models::qwen3_5_moe::qwen38_batch::q38_batch_bucket_plan_env_nv_q38_batch_sizes();
                    let lanes = nv_models::qwen3_5_moe::qwen38_batch::Qwen38BatchLanes::new(
                        model,
                        &device,
                        kv_max_seq_len,
                        plan,
                    )
                    .context(
                        "NV_Q38_BATCH=1 was requested, so a lane pool that cannot build is a \
                         boot failure rather than a silent fall-back to solo serving",
                    )?;
                    eprintln!(
                        "[qwen3.8-batch] request-group scheduler on: {} lanes x {} kv slots, \
                         window {} ms",
                        lanes.lanes(),
                        kv_max_seq_len,
                        q38_batch_window_ms_env_nv_q38_batch_window_ms()
                    );
                    let sched = Qwen38BatchScheduler::spawn(
                        lanes,
                        qwen_mtp.clone(),
                        tokenizer.clone(),
                        device.clone(),
                        eos_token_ids.clone(),
                        kv_max_seq_len,
                        256,
                    )?;
                    QwenMoeShared::Batch(sched)
                } else {
                    QwenMoeShared::Eager(Arc::new(tokio::sync::Mutex::new(model)))
                };
                Ok(Self {
                    inner: NvEngineInner {
                        model_id,
                        family,
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
                        qwen_mtp,
                        mm_towers: None,
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
                let kv_max_seq_len = kv_max_seq_len_for(config.max_position_embeddings);
                let (shared, qwen_moe_dispatch) = if qwen_graph {
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
                                "[qwen3.6-moe] decode: GRAPH (grouped device MoE + CUDA graph capture; NV_QWEN_GRAPH=0 for eager), kv window {kv_max_seq_len}, grouped expert copy +{} MiB VRAM",
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
                        "[qwen3.6-moe] decode: EAGER (NV_QWEN_GRAPH={:?} NV_QWEN_MOE_GROUPED={:?}; graph decode needs both unset/on)",
                        std::env::var("NV_QWEN_GRAPH").ok(),
                        std::env::var("NV_QWEN_MOE_GROUPED").ok()
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
                        model: LoadedModel::Qwen3_5Moe(shared),
                        tokenizer: Arc::new(tokenizer),
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
                        qwen_mtp: None,
                        mm_towers: None,
                    },
                })
            }
            ModelFamily::Laguna => {
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
                let kv_max_seq_len = kv_max_seq_len_for(config.max_position_embeddings);
                let shared = LagunaShared(Arc::new(model));
                let laguna_spec = build_laguna_spec_serve(&shared, kv_max_seq_len);
                Ok(Self {
                    inner: NvEngineInner {
                        model_id,
                        family,
                        model: LoadedModel::Laguna(shared),
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
                        laguna_spec,
                        qwen_moe_dispatch: None,
                        qwen_mtp: None,
                        mm_towers: None,
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
                let kv_max_seq_len = kv_max_seq_len_for(config.max_position_embeddings);
                Ok(Self {
                    inner: NvEngineInner {
                        model_id,
                        family,
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
                let kv_max_seq_len = kv_max_seq_len_for(text_config.max_position_embeddings);
                Ok(Self {
                    inner: NvEngineInner {
                        model_id,
                        family,
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
                    },
                })
            }
        }
    }

    #[cfg(not(feature = "cuda"))]
    pub(crate) fn try_load_inner(_model_dir: &Path) -> anyhow::Result<Self> {
        anyhow::bail!(
            "NvEngineChat::try_load requires the `cuda` cargo feature; \
             rebuild with --features cuda or unset NV_CHAT_MODEL_DIR"
        )
    }
}

pub(crate) fn spec_serve_max_seq(env: Option<usize>, kv_max_seq_len: usize) -> usize {
    match env {
        Some(n) if n > 0 => n.min(kv_max_seq_len),
        _ => kv_max_seq_len,
    }
}

pub(crate) fn kv_max_seq_len_for(max_position_embeddings: usize) -> usize {
    let default = max_position_embeddings.min(8192);
    match std::env::var("NV_KV_MAX_SEQ_LEN")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
    {
        Some(n) if n > 0 => n.min(max_position_embeddings.max(1)),
        _ => default,
    }
}

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
    let out = std::process::Command::new("nvidia-smi")
        .args([
            "--query-compute-apps=pid,used_memory",
            "--format=csv,noheader,nounits",
        ])
        .output()
        .ok()?;
    let me = std::process::id();
    let text = String::from_utf8_lossy(&out.stdout).to_string();
    for line in text.lines() {
        let mut it = line.split(',').map(|f| f.trim());
        let (Some(pid), Some(mib)) = (it.next(), it.next()) else {
            continue;
        };
        if pid.parse::<u32>().ok() == Some(me) {
            return mib.parse::<u64>().ok().map(|m| m * 1024 * 1024);
        }
    }
    None
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
pub(crate) fn enforce_gemma4_vram_budget(
    config: &nv_models::gemma4::Gemma4Config,
    kv_max: usize,
    drafter_charge: DrafterKvCharge,
) -> anyhow::Result<Option<u64>> {
    let fp8 = nv_models::gemma4::verify_kv_use_fp8();
    let rings = nv_models::gemma4::kv_ring_enabled();
    let (drafter_rows, drafter_row_elems) = drafter_charge.rows_elems(kv_max);
    let b = nv_models::gemma4::kv_budget_capped(
        config,
        kv_max,
        fp8,
        rings,
        drafter_row_elems,
        drafter_rows,
    );
    let hd512_scratch = nv_models::gemma4::gqa512_verify_scratch_bytes(config);
    let gib = |x: usize| x as f64 / (1u64 << 30) as f64;
    let weights = our_process_vram_bytes();
    let budget_gib: f64 = crate::oapi::admission::default_budget_gib();
    let weights_gib = weights.map(|w| w as f64 / (1u64 << 30) as f64);
    tracing::info!(
        kv_max,
        verify_kv = if fp8 { "fp8" } else { "bf16" },
        rings,
        ring_slots = b.ring_slots,
        verify_full_gib = format!("{:.2}", gib(b.verify_full_bytes)),
        verify_sliding_gib = format!("{:.3}", gib(b.verify_sliding_bytes)),
        verify_scratch_gib = format!("{:.3}", gib(b.verify_scratch_bytes)),
        verify_hd512 = hd512_scratch > 0,
        verify_hd512_scratch_gib = format!("{:.3}", gib(hd512_scratch)),
        decode_full_gib = format!("{:.2}", gib(b.decode_full_bytes)),
        decode_sliding_gib = format!("{:.3}", gib(b.decode_sliding_bytes)),
        drafter_kv_gib = format!("{:.2}", gib(b.drafter_kv_bytes)),
        kv_worst_total_gib = format!("{:.2}", gib(b.worst_total())),
        weights_measured_gib = weights_gib
            .map(|w| format!("{w:.2}"))
            .unwrap_or_else(|| "unknown".into()),
        budget_gib,
        "gemma4 VRAM budget at kv_max"
    );
    if weights_gib.is_none() {
        tracing::warn!(
            kv_max,
            budget_gib,
            kv_worst_total_gib = format!("{:.2}", gib(b.worst_total() + hd512_scratch)),
            "gemma4 VRAM BUDGET GUARD IS INOPERATIVE: nvidia-smi did not report this process's \
             memory, so the weights term is unknown and the check below can only see the KV \
             term. It will not refuse a configuration that does not fit. Treat the \
             weights_measured_gib=unknown line above as the real signal."
        );
    }
    let total_gib = weights_gib.unwrap_or(0.0) + gib(b.worst_total() + hd512_scratch);
    if total_gib > budget_gib {
        anyhow::bail!(
            "gemma4 VRAM budget exceeded: weights {} GiB + worst-case KV {:.2} GiB at              kv_max={} > budget {budget_gib} GiB. Lower NV_KV_MAX_SEQ_LEN, keep              NV_KV_RING enabled, or raise NV_VRAM_BUDGET_GIB if the GPU allows.",
            weights_gib.map(|w| format!("{w:.2}")).unwrap_or_else(|| "?".into()),
            gib(b.worst_total() + hd512_scratch),
            kv_max,
        )
    }
    Ok(weights)
}

#[cfg(feature = "cuda")]
pub(crate) fn gemma4_kv_max_is_explicit() -> bool {
    std::env::var("NV_KV_MAX_SEQ_LEN")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .is_some_and(|n| n > 0)
}

#[cfg(feature = "cuda")]
fn gemma4_kv_max_fits(
    config: &nv_models::gemma4::Gemma4Config,
    kv_max: usize,
    drafter_charge: DrafterKvCharge,
    weights_gib: f64,
    budget_gib: f64,
) -> bool {
    let fp8 = nv_models::gemma4::verify_kv_use_fp8();
    let rings = nv_models::gemma4::kv_ring_enabled();
    let (drafter_rows, drafter_row_elems) = drafter_charge.rows_elems(kv_max);
    let b = nv_models::gemma4::kv_budget_capped(
        config,
        kv_max,
        fp8,
        rings,
        drafter_row_elems,
        drafter_rows,
    );
    let extra = b.worst_total() + nv_models::gemma4::gqa512_verify_scratch_bytes(config);
    weights_gib + (extra as f64 / (1u64 << 30) as f64) <= budget_gib
}

#[cfg(feature = "cuda")]
pub(crate) const KV_FIT_FLOOR: usize = 4096;

#[cfg(feature = "cuda")]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum KvFit {
    Explicit(usize),
    Fits(usize),
    Reduced { requested: usize, fitted: usize },
    ProbeFailed(usize),
    NoFit { requested: usize, floor: usize },
}

#[cfg(feature = "cuda")]
impl KvFit {
    pub(crate) fn value(self) -> usize {
        match self {
            KvFit::Explicit(n) | KvFit::Fits(n) | KvFit::ProbeFailed(n) => n,
            KvFit::Reduced { fitted, .. } => fitted,
            KvFit::NoFit { requested, .. } => requested,
        }
    }
}

#[cfg(feature = "cuda")]
pub(crate) fn decide_gemma4_kv_max(
    config: &nv_models::gemma4::Gemma4Config,
    requested: usize,
    drafter_charge: DrafterKvCharge,
    weights_gib: Option<f64>,
    budget_gib: f64,
    explicit: bool,
) -> KvFit {
    if explicit {
        return KvFit::Explicit(requested);
    }
    let Some(weights_gib) = weights_gib else {
        return KvFit::ProbeFailed(requested);
    };
    if gemma4_kv_max_fits(config, requested, drafter_charge, weights_gib, budget_gib) {
        return KvFit::Fits(requested);
    }
    let mut fitted = requested;
    while fitted > KV_FIT_FLOOR {
        fitted /= 2;
        if gemma4_kv_max_fits(config, fitted, drafter_charge, weights_gib, budget_gib) {
            return KvFit::Reduced { requested, fitted };
        }
    }
    KvFit::NoFit {
        requested,
        floor: KV_FIT_FLOOR,
    }
}

#[cfg(feature = "cuda")]
pub(crate) fn fit_gemma4_kv_max(
    config: &nv_models::gemma4::Gemma4Config,
    requested: usize,
    drafter_charge: DrafterKvCharge,
) -> usize {
    let budget_gib: f64 = crate::oapi::admission::default_budget_gib();
    let explicit = gemma4_kv_max_is_explicit();
    let weights_gib = if explicit {
        None
    } else {
        our_process_vram_bytes().map(|w| w as f64 / (1u64 << 30) as f64)
    };
    let decision = decide_gemma4_kv_max(
        config,
        requested,
        drafter_charge,
        weights_gib,
        budget_gib,
        explicit,
    );
    match decision {
        KvFit::Explicit(_) | KvFit::Fits(_) => {}
        KvFit::Reduced { requested, fitted } => tracing::warn!(
            requested,
            fitted,
            weights_gib = weights_gib.map(|w| format!("{w:.2}")).unwrap_or_default(),
            budget_gib,
            "gemma4 kv_max does not fit the VRAM budget at the checkpoint's full context; \
             auto-reduced. Set NV_KV_MAX_SEQ_LEN to choose explicitly, or raise \
             NV_VRAM_BUDGET_GIB if the GPU allows."
        ),
        KvFit::ProbeFailed(requested) => tracing::warn!(
            requested,
            budget_gib,
            "gemma4 kv_max AUTO-FIT IS DISABLED: could not read this process's VRAM usage from \
             nvidia-smi, so there is no weights figure to fit against and kv_max is being used \
             as requested. If the checkpoint does not actually fit, this boot will fail later \
             with an allocation error rather than being reduced here. Set NV_KV_MAX_SEQ_LEN \
             explicitly to make the choice yourself."
        ),
        KvFit::NoFit { requested, floor } => tracing::warn!(
            requested,
            floor,
            weights_gib = weights_gib.map(|w| format!("{w:.2}")).unwrap_or_default(),
            budget_gib,
            "gemma4 kv_max does not fit the VRAM budget even at the auto-fit floor; proceeding \
             at the requested value, which is expected to fail. Raise NV_VRAM_BUDGET_GIB, or set \
             NV_KV_MAX_SEQ_LEN below the floor to choose explicitly."
        ),
    }
    decision.value()
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
pub(crate) fn load_dflash_state(
    device: &candle_core::Device,
    target_embed: Option<&candle_core::Tensor>,
) -> Option<Arc<DFlashShared>> {
    load_dflash_state_for_target(device, target_embed, None)
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

pub(crate) const SERVING_TOKEN_ID_FILES_IN_PRECEDENCE_ORDER: [&str; 2] =
    ["generation_config.json", "config.json"];

fn declaring_scopes(v: &serde_json::Value) -> impl Iterator<Item = &serde_json::Value> {
    [Some(v), v.get("text_config")].into_iter().flatten()
}

pub(crate) fn parse_bos_id(raw_cfg: &str) -> Option<u32> {
    let v: serde_json::Value = serde_json::from_str(raw_cfg).ok()?;
    for scope in declaring_scopes(&v) {
        if let Some(n) = scope.get("bos_token_id").and_then(|x| x.as_u64()) {
            return Some(n as u32);
        }
    }
    None
}

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

pub(crate) fn bos_id_from_dir(dir: &Path) -> Option<u32> {
    SERVING_TOKEN_ID_FILES_IN_PRECEDENCE_ORDER
        .iter()
        .find_map(|file| {
            std::fs::read_to_string(dir.join(file))
                .ok()
                .and_then(|raw| parse_bos_id(&raw))
        })
}

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

    std::thread::spawn(|| {
        let _ = nv_grammar::JsonConstraint::from_regex(&nv_grammar::json_object_regex(3));
    });
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

#[derive(Clone)]
pub struct ChatRegistry {
    pub(crate) engines: Arc<std::collections::HashMap<String, Arc<dyn ChatEngine>>>,
    pub(crate) default_id: String,
    pub(crate) order: Arc<Vec<String>>,
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
        Some(Self {
            engines: Arc::new(map),
            default_id,
            order: Arc::new(order),
        })
    }

    pub fn single(engine: Arc<dyn ChatEngine>) -> Self {
        Self::from_engines(vec![engine]).expect("single engine is non-empty")
    }

    pub fn resolve(&self, model: Option<&str>) -> Option<Arc<dyn ChatEngine>> {
        self.resolve_with(model, allow_unknown_model())
    }

    pub fn resolve_with(
        &self,
        model: Option<&str>,
        allow_unknown: bool,
    ) -> Option<Arc<dyn ChatEngine>> {
        match model {
            None | Some("") => self.engines.get(&self.default_id).cloned(),
            Some(m) => match self.engines.get(m) {
                Some(eng) => Some(eng.clone()),
                None => {
                    if let Some(canon) = crate::oapi::model_ids::canonical_model_id(m) {
                        if let Some(eng) = self.engines.get(&canon) {
                            return Some(eng.clone());
                        }
                    }
                    if allow_unknown && self.order.len() == 1 {
                        warn_unknown_model_fallback(m, &self.default_id);
                        return self.engines.get(&self.default_id).cloned();
                    }
                    None
                }
            },
        }
    }

    pub fn default_engine(&self) -> Arc<dyn ChatEngine> {
        self.engines
            .get(&self.default_id)
            .cloned()
            .expect("default engine present")
    }

    pub fn model_ids(&self) -> &[String] {
        &self.order
    }

    pub fn contains(&self, model: &str) -> bool {
        self.engines.contains_key(model)
    }
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
            .split(|c| c == ',' || c == ':')
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .map(PathBuf::from)
            .collect();
        if dirs.is_empty() {
            return None;
        }
        return build_registry_strict(&dirs, try_load_engine_dir);
    }
    #[cfg(not(feature = "cuda"))]
    if std::env::var_os("NV_CHAT_MODEL_DIRS").is_some() {
        tracing::warn!(
            "NV_CHAT_MODEL_DIRS set but this build lacks the `cuda` feature; nv-engine chat disabled"
        );
    }
    default_engine_from_env().map(ChatRegistry::single)
}

#[cfg(all(test, feature = "cuda"))]
mod drafter_dim_gate_real_checkpoints {
    use super::*;

    fn required_dir(key: &str) -> PathBuf {
        let raw = std::env::var_os(key)
            .unwrap_or_else(|| panic!("{key} must point at a checkpoint directory for this test"));
        let p = PathBuf::from(raw);
        assert!(p.is_dir(), "{key}={} is not a directory", p.display());
        p
    }

    fn expectation() -> &'static str {
        match std::env::var("NV_DIMCHECK_EXPECT").ok().as_deref() {
            Some("ok") => "ok",
            Some("mismatch") => "mismatch",
            other => panic!(
                "set NV_DIMCHECK_EXPECT=ok|mismatch to say which verdict this checkpoint pair \
                 should produce (got {other:?})"
            ),
        }
    }

    #[test]
    #[ignore]
    fn eagle3_config_pair_from_env() {
        let expect = expectation();
        let model_dir = required_dir("NV_CHAT_MODEL_DIR");
        let draft_dir = required_dir("NV_EAGLE3_DRAFT_DIR");

        let raw = std::fs::read_to_string(model_dir.join("config.json")).expect("target config");
        let target_cfg =
            nv_models::gemma4::Gemma4Config::from_hf_json_str(&raw).expect("parse target config");
        let target = gemma4_target_dims(
            &crate::oapi::model_ids::model_id_for_dir(&model_dir),
            &target_cfg,
        );

        let draft_cfg = nv_specdecode::eagle3_loader::Eagle3SpeculatorConfig::from_hf_json_file(
            &draft_dir.join("config.json"),
        )
        .expect("parse eagle3 config");
        let drafter = DrafterDims {
            fc_in_dim: draft_cfg.fc_in_dim(),
            target_vocab_size: draft_cfg.target_vocab_size,
            aux_layer_ids: draft_cfg.eagle_aux_hidden_state_layer_ids.clone(),
        };

        let problems = drafter_target_mismatch(&target, &drafter);
        eprintln!(
            "target {} hidden={} vocab={} layers={}",
            target.model_id, target.hidden_size, target.vocab_size, target.num_hidden_layers
        );
        eprintln!(
            "drafter {} fc_in={} target_vocab={} aux={:?}",
            draft_dir.display(),
            drafter.fc_in_dim,
            drafter.target_vocab_size,
            drafter.aux_layer_ids
        );
        for p in &problems {
            eprintln!("problem: {p}");
        }
        match expect {
            "ok" => assert!(
                problems.is_empty(),
                "this pair must be accepted, got {problems:?}"
            ),
            _ => assert!(
                !problems.is_empty(),
                "this pair must be rejected but the validator found nothing"
            ),
        }
    }

    #[test]
    #[ignore]
    fn the_matching_pair_still_loads_with_spec_on() {
        assert_eq!(
            expectation(),
            "ok",
            "this test wants the shipped, matching pair"
        );
        let model_dir = required_dir("NV_CHAT_MODEL_DIR");
        let _draft_dir = required_dir("NV_EAGLE3_DRAFT_DIR");
        assert!(env_flag_enabled(
            std::env::var("NV_USE_EAGLE3").ok().as_deref()
        ));
        let eng = NvEngineChat::try_load(&model_dir).expect("engine must load");
        assert!(
            eng.inner.eagle3.is_some(),
            "the dim check must not reject a drafter that agrees with its target"
        );
        assert_eq!(eng.inner.spec_status, Some("on"));
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    #[ignore]
    async fn a_mismatched_eagle3_drafter_disables_spec_and_still_serves() {
        let _ = tracing_subscriber::fmt()
            .with_max_level(tracing::Level::INFO)
            .with_test_writer()
            .try_init();
        assert_eq!(
            expectation(),
            "mismatch",
            "this test wants a deliberately mismatched pair"
        );
        let model_dir = required_dir("NV_CHAT_MODEL_DIR");
        let _draft_dir = required_dir("NV_EAGLE3_DRAFT_DIR");
        assert!(
            env_flag_enabled(std::env::var("NV_USE_EAGLE3").ok().as_deref()),
            "set NV_USE_EAGLE3=1 so spec-decode is actually requested"
        );

        let eng = match NvEngineChat::try_load(&model_dir) {
            Ok(eng) => eng,
            Err(err) => panic!(
                "the TARGET at {} did not load, so this fixture never reached the drafter gate \
                 and says nothing about it: {err:#}. Pick a mismatched pair whose target loads \
                 on this backend. google/gemma-4-E4B-it no longer qualifies -- gemma4::Gemma4 \
                 refuses MatFormer checkpoints (#34) -- but google/gemma-4-26B-A4B-it against \
                 the gemma-4-31B-it eagle3 speculator does.",
                model_dir.display()
            ),
        };
        assert!(
            eng.inner.eagle3.is_none(),
            "a drafter whose dims disagree with the target must be refused at load"
        );
        assert!(eng.inner.dflash.is_none());
        assert_eq!(
            eng.inner.spec_status,
            Some("degraded"),
            "the engine must advertise the degradation rather than pretend spec-decode is on"
        );

        let messages = vec![crate::oapi::chat::ChatMessageIn {
            role: "user".into(),
            content: Some(crate::oapi::chat::MessageContent::Text(
                "Name one primary colour.".into(),
            )),
            tool_calls: None,
            tool_call_id: None,
            name: None,
        }];
        let prompt = eng.render_chat(&messages, &[], &crate::oapi::chat::ToolChoice::None);
        let req = ChatGenerateRequest {
            prompt,
            max_new_tokens: 8,
            stop: Vec::new(),
            seed: Some(1),
            temperature: Some(0.0),
            top_p: None,
            top_k: None,
            min_p: None,
            presence_penalty: None,
            frequency_penalty: None,
            repetition_penalty: None,
            guided: None,
            guided_think_close: None,
            logit_bias: Vec::new(),
            logprobs: false,
            top_logprobs: 0,
            kv_resume: None,
            kv_store: None,
            mm: None,
        };

        let (tx, mut rx) = mpsc::channel(64);
        eng.generate(req, tx).await.expect("generate must start");

        let mut text = String::new();
        let mut done = None;
        while let Some(ev) = rx.recv().await {
            match ev {
                ChatEvent::TextDelta(d) => text.push_str(&d),
                ChatEvent::Done { finish_reason, .. } => done = Some(finish_reason),
                ChatEvent::Error(e) => panic!("the request path still fails: {e}"),
                _ => {}
            }
        }
        let reason = done.expect("the stream must terminate with Done, not silence");
        eprintln!("finish_reason={reason} text={text:?}");
        assert!(
            !text.is_empty(),
            "a served request must produce at least one token"
        );
    }
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
    fn gguf_dir_is_refused_with_the_gap_named() {
        let dir = seam_dir("inner");
        std::fs::write(dir.join("model.gguf"), b"x").unwrap();

        let err = match NvEngineChat::try_load_inner(&dir) {
            Ok(_) => panic!("gguf dir must be refused"),
            Err(e) => e,
        };
        let msg = format!("{err:#}");
        assert!(
            msg.contains("GGUF checkpoint dir") && msg.contains("wgpu"),
            "refusal must name the gap and the working backend: {msg}"
        );
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn the_public_entry_point_refuses_a_gguf_dir_before_the_required_files_loop() {
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
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn the_refusal_names_the_format_gap_not_a_missing_variant() {
        let dir = seam_dir("text");
        std::fs::write(dir.join("model.gguf"), b"x").unwrap();
        let err = NvEngineChat::try_load(&dir).err().expect("must be refused");
        let msg = format!("{err:#}");
        assert!(
            !msg.contains("no Gemma4Moe variant"),
            "the refusal must not assert a variant this file declares: {msg}"
        );
        assert!(
            msg.contains("safetensors-only") && msg.contains("from_gguf"),
            "the refusal must name the real gap (safetensors-only loader): {msg}"
        );
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn a_dir_with_config_json_is_not_refused_as_gguf() {
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

#[cfg(test)]
mod spec_serve_max_seq_tests {
    use super::spec_serve_max_seq;

    #[test]
    fn a_spec_cap_below_kv_max_shrinks_only_the_spec_engine_window() {
        assert_eq!(spec_serve_max_seq(Some(131072), 262144), 131072);
    }

    #[test]
    fn unset_zero_or_oversized_caps_leave_the_full_kv_window() {
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
    fn unknown_model_is_not_served_by_the_only_engine() {
        let reg = single_reg();
        assert!(reg.resolve_with(Some("gemma-4-e4b"), false).is_none());
        assert!(reg.resolve_with(Some("nope"), false).is_none());
    }

    #[test]
    fn unknown_model_falls_back_only_behind_the_escape_hatch() {
        let reg = single_reg();
        assert_eq!(
            reg.resolve_with(Some("gemma-4-e4b"), true)
                .unwrap()
                .model_id(),
            "gemma-4-E4B-it"
        );
    }

    #[test]
    fn absent_and_empty_model_resolve_to_the_default_either_way() {
        let reg = single_reg();
        for allow in [false, true] {
            assert_eq!(
                reg.resolve_with(None, allow).unwrap().model_id(),
                "gemma-4-E4B-it"
            );
            assert_eq!(
                reg.resolve_with(Some(""), allow).unwrap().model_id(),
                "gemma-4-E4B-it"
            );
        }
    }

    #[test]
    fn the_escape_hatch_never_fires_with_more_than_one_engine() {
        let reg = ChatRegistry::from_engines(vec![
            Arc::new(EchoEngine::new("a", "x")) as Arc<dyn ChatEngine>,
            Arc::new(EchoEngine::new("b", "y")),
        ])
        .unwrap();
        assert!(reg.resolve_with(Some("missing"), true).is_none());
    }

    #[test]
    fn store_path_and_basename_aliases_resolve_to_the_pretty_id() {
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
    fn all_good_dirs_serve() {
        let dirs: Vec<PathBuf> = ["/good/a", "/good/b"].iter().map(PathBuf::from).collect();
        let reg = build_registry_strict(&dirs, |d| echo(d.file_name().unwrap().to_str().unwrap()))
            .expect("all dirs loaded");
        assert_eq!(reg.model_ids(), &["a".to_string(), "b".to_string()]);
    }

    #[test]
    #[should_panic(expected = "chat model dir failed to load")]
    fn all_dirs_bad_is_a_clear_named_fatal_not_a_silent_stub() {
        let dirs: Vec<PathBuf> = ["/bad/a", "/bad/b"].iter().map(PathBuf::from).collect();
        let _ = build_registry_strict(&dirs, |d| anyhow::bail!("boom: {}", d.display()));
    }

    #[test]
    fn escape_hatch_env_parsing() {
        assert!(!allow_unknown_model_from(None));
        assert!(!allow_unknown_model_from(Some("0")));
        assert!(!allow_unknown_model_from(Some("")));
        assert!(allow_unknown_model_from(Some("1")));
        assert!(allow_unknown_model_from(Some(" TRUE ")));
        assert!(allow_unknown_model_from(Some("on")));
    }
}

#[cfg(all(test, feature = "cuda"))]
mod kv_fit_tests {
    use super::*;

    fn cfg() -> nv_models::gemma4::Gemma4Config {
        let flat = serde_json::json!({
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
        .to_string();
        nv_models::gemma4::Gemma4Config::from_hf_json_str(&flat).expect("flat config parses")
    }

    fn kv_gib(c: &nv_models::gemma4::Gemma4Config, kv_max: usize, drafter: usize) -> f64 {
        let fp8 = nv_models::gemma4::verify_kv_use_fp8();
        let rings = nv_models::gemma4::kv_ring_enabled();
        let b = nv_models::gemma4::kv_budget(c, kv_max, fp8, rings, drafter);
        (b.worst_total() + nv_models::gemma4::gqa512_verify_scratch_bytes(c)) as f64
            / (1u64 << 30) as f64
    }

    #[test]
    fn probe_failure_is_reported_not_silently_treated_as_a_fit() {
        let c = cfg();
        let d = decide_gemma4_kv_max(&c, 131072, DrafterKvCharge::default(), None, 76.0, false);
        assert_eq!(
            d,
            KvFit::ProbeFailed(131072),
            "a failed VRAM probe must be its own outcome. It used to return the requested \
             value indistinguishably from a genuine fit, so the auto-fit silently became a \
             no-op and the boot failed later in allocation with no hint why."
        );
        assert_eq!(
            d.value(),
            131072,
            "the value is unchanged; only the reporting is"
        );
    }

    #[test]
    fn explicit_request_is_honoured_without_probing() {
        let c = cfg();
        assert_eq!(
            decide_gemma4_kv_max(&c, 8192, DrafterKvCharge::default(), None, 0.001, true),
            KvFit::Explicit(8192),
            "an explicit NV_KV_MAX_SEQ_LEN is the operator's decision and must not be fitted"
        );
    }

    #[test]
    fn generous_budget_fits_unchanged() {
        let c = cfg();
        let budget = 40.0 + kv_gib(&c, 131072, 0) + 1.0;
        assert_eq!(
            decide_gemma4_kv_max(
                &c,
                131072,
                DrafterKvCharge::default(),
                Some(40.0),
                budget,
                false
            ),
            KvFit::Fits(131072)
        );
    }

    #[test]
    fn budget_below_the_floor_reports_nofit_rather_than_a_fit() {
        let c = cfg();
        let d = decide_gemma4_kv_max(
            &c,
            131072,
            DrafterKvCharge::default(),
            Some(100.0),
            76.0,
            false,
        );
        assert_eq!(
            d,
            KvFit::NoFit {
                requested: 131072,
                floor: KV_FIT_FLOOR
            },
            "weights alone exceed the budget, so no kv_max fits -- that must not read as a fit"
        );
        assert_eq!(d.value(), 131072);
    }

    #[test]
    fn a_budget_between_two_levels_reduces_to_the_larger_one_that_fits() {
        let c = cfg();
        let (requested, weights) = (131072usize, 40.0f64);
        let mut oracle = requested;
        let budget = weights + kv_gib(&c, requested / 4, 0);
        while oracle > KV_FIT_FLOOR && weights + kv_gib(&c, oracle, 0) > budget {
            oracle /= 2;
        }
        if oracle == requested {
            eprintln!("SKIP: KV does not shrink with kv_max for this config; nothing to reduce");
            return;
        }
        assert_eq!(
            decide_gemma4_kv_max(
                &c,
                requested,
                DrafterKvCharge::default(),
                Some(weights),
                budget,
                false
            ),
            KvFit::Reduced {
                requested,
                fitted: oracle
            },
            "auto-fit must land on the largest halving that fits, not overshoot past it"
        );
    }

    #[test]
    fn uncapped_charge_matches_legacy_max_elems_times_kv_max() {
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

    #[test]
    fn capped_eagle3_charge_never_undercuts_the_uncapped_dflash_term() {
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

    #[test]
    fn capped_drafter_charge_fits_where_uncapped_reduces() {
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
        assert_eq!(
            decide_gemma4_kv_max(&c, requested, capped, Some(weights), budget, false),
            KvFit::Fits(requested),
            "with the drafter KV cap the accounting must admit the full context"
        );
        let d = decide_gemma4_kv_max(&c, requested, uncapped, Some(weights), budget, false);
        assert!(
            !matches!(d, KvFit::Fits(_)),
            "the uncapped drafter charge (4*131072*4096 = 2 GiB) must not fit this \
             budget, got {d:?}"
        );
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
    fn without_the_opt_in_the_refusal_is_unchanged_and_still_names_wgpu() {
        for cfg in [dense_config_json(), dense_config_json().replace(
            "\"architectures\": [\"Qwen3_5ForConditionalGeneration\"],",
            "",
        )] {
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
        }
    }

    #[test]
    fn the_opt_in_routes_the_dense_checkpoint_at_the_qwen3_5_trunk() {
        for cfg in [dense_config_json(), dense_config_json().replace(
            "\"architectures\": [\"Qwen3_5ForConditionalGeneration\"],",
            "",
        )] {
            assert_eq!(
                detect_family_with_dense_cuda_serve(&cfg, true).expect(
                    "with the opt-in, both the architectures entry and the bare model_type must \
                     reach the qwen3.5 trunk that nv_models::qwen3_5_moe::Qwen3Moe decodes"
                ),
                ModelFamily::Qwen3_5Moe
            );
        }
    }

    #[test]
    fn the_opt_in_does_not_move_a_moe_checkpoint_or_a_qwen3_checkpoint() {
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
    fn the_router_predicate_and_the_config_parsers_agree_on_which_is_dense() {
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
    }

    #[test]
    fn the_dense_trunk_is_hybrid_and_carries_no_experts() {
        let cfg = nv_models::qwen3_5_moe::Qwen3_5DenseConfig::from_hf_json_str(&dense_config_json())
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

    #[test]
    fn graph_decode_is_never_selected_for_a_dense_checkpoint() {
        assert!(
            !qwen_graph_decode_selected(ModelFamily::Qwen3_5Moe, &dense_config_json()),
            "GraphedQwen3Moe captures the grouped-MoE decode step and its selection also flips the \
             device to Device::new_cuda_with_stream; neither applies to a checkpoint whose every \
             ffn is a dense Mlp"
        );
    }

    #[test]
    fn the_opt_in_notice_names_the_variable_the_loader_the_oracle_and_the_run_that_compared_them() {
        for needle in [
            QWEN35_DENSE_CUDA_SERVE_ENV,
            "from_loader_dense_quantized",
            "nv_models::qwen3_5_dense_wgpu",
            "LayerFfn::Dense",
            "nv_layers::linear_attn::LinearAttention",
            "rust/tests/qwen35_dense_cuda_serving_ab.rs",
            "NV_QWEN35_DENSE_CUDA_SERVE_TEST=1",
            "ig1/Qwen3.5-9B-NVFP4",
        ] {
            assert!(
                QWEN35_DENSE_CUDA_OPT_IN_MATCHES_THE_WGPU_ORACLE_AND_STAYS_OPT_IN.contains(needle),
                "the opt-in notice printed at load must name {needle}: it is the only durable \
                 record of what this path does, what was compared against the wgpu oracle, and \
                 which run an operator can repeat"
            );
        }
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

    fn qwen38_27b_config_json() -> String {
        format!(
            r#"{{
  "architectures": ["Qwen3_5ForConditionalGeneration"],
  "model_type": "qwen3_5",
  "tie_word_embeddings": false,
  "transformers_version": "4.57.3",
  "text_config": {{
    "model_type": "qwen3_5_text",
    "hidden_size": 5120,
    "num_hidden_layers": 48,
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
    "linear_num_value_heads": 32,
    "linear_key_head_dim": 128,
    "linear_value_head_dim": 128,
    "linear_conv_kernel_dim": 4,
    "eos_token_id": 248044,
    "rope_parameters": {{"rope_theta": 10000000.0, "partial_rotary_factor": 0.25, "rope_type": "default"}},
    "layer_types": [{}]
  }},
  "vision_config": {{"hidden_size": 1152, "depth": 27, "patch_size": 16, "temporal_patch_size": 2}}
}}"#,
            three_linear_then_one_full(48)
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
    fn the_27b_shape_is_dense_refused_by_default_and_routed_to_the_dense_arm_by_the_opt_in() {
        for cfg in [qwen38_27b_config_json(), qwen38_27b_config_json().replace(
            "\"architectures\": [\"Qwen3_5ForConditionalGeneration\"],",
            "",
        )] {
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
    }

    #[test]
    fn the_flagship_shape_routes_qwen3_5_moe_from_architectures_and_from_model_type_alone() {
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

    #[test]
    fn the_27b_declares_dense_ffn_and_only_the_dense_parser_accepts_it() {
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
    }
}

#[cfg(all(test, feature = "cuda"))]
mod capture_stream_device_policy {
    use super::*;

    const GEMMA4_CFG: &str = r#"{"architectures": ["Gemma4ForCausalLM"], "model_type": "gemma4"}"#;
    const QWEN3_CFG: &str = r#"{"architectures": ["Qwen3ForCausalLM"], "model_type": "qwen3"}"#;

    #[test]
    fn the_families_whose_graph_body_is_candle_are_named_even_though_none_is_flipped() {
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
    }

    #[test]
    fn no_family_is_cleared_for_the_device_stream_flip_yet() {
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
    }

    #[test]
    fn the_device_selection_is_the_qwen_graph_selector_plus_only_the_q38_batch_opt_in() {
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
                    || qwen38_batch_lanes_boot_selected(family, cfg),
                "{family:?}: try_load_inner must keep choosing Device::new_cuda_with_stream on \
                 exactly the qwen-graph condition plus the NV_Q38_BATCH dense lane opt-in and \
                 nothing else. Task #63 records grouped MoE on a non-default stream producing \
                 all-NaN logits, and gemma4 prefill produced all -inf on the same axis; the one \
                 permitted widening is qwen38_batch_lanes_boot_selected, whose serving \
                 measurement is rust/tests/qwen38_batch_serving_e2e.rs (solo MTP and batch \
                 groups byte-match their flag-off references on the stream-mode device)"
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
    fn a_bos_declared_only_inside_text_config_is_found() {
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
    }

    #[test]
    fn generation_config_outranks_config_json_for_both_ids() {
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

    #[test]
    fn a_checkpoint_with_no_eos_anywhere_is_refused_rather_than_given_gemmas() {
        let dir = checkpoint_dir("silent", &[("config.json", r#"{"model_type":"gemma4"}"#)]);
        assert_eq!(bos_id_from_dir(&dir), None);
        let err = eos_ids_from_dir(&dir).err().expect("must be refused");
        let msg = format!("{err:#}");
        assert!(
            msg.contains("no stop condition") && msg.contains("[1, 106]"),
            "the refusal must name what it refuses to invent: {msg}"
        );
        std::fs::remove_dir_all(&dir).ok();
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn the_laguna_pair_read_off_disk_makes_the_position_0_rule_refuse() {
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
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn a_generation_config_bos_inside_its_own_eos_set_is_refused_too() {
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
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn a_gemma_pair_read_off_disk_still_splices() {
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
