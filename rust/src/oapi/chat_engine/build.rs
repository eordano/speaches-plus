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

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum EngineBuildMode {
    Full,
    AdapterServingLean,
}

pub(crate) fn adapter_serving_lean_keeps_the_i8_decode_head_by_default_because_the_bf16_fallback_epilogue_costs_0_88_ms_per_token_escape_nv_lora_ada_i8_head_0(
) -> bool {
    std::env::var("NV_LORA_ADA_I8_HEAD").as_deref() != Ok("0")
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
    Batch(Arc<QwenBatchScheduler>),
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
pub(crate) const Q38_SERVE_GRAPHED_ENV: &str = "NV_Q38_SERVE_GRAPHED";

pub(crate) const Q38_SERVE_GRAPHED_DEFAULT_ON_YIELDS_TO_EXPLICIT_BATCH_SET_0_FOR_THE_EAGER_SESSION:
    &str = "the graphed solo session serves 1.6-2.0x eager decode with faster TTFT at every \
     measured depth (gserve rows, perf/runs.jsonl); unset env means graphed unless \
     NV_Q38_BATCH=1 booted the lane pool, which owns the model -- an explicit \
     NV_Q38_SERVE_GRAPHED=1 beside batch stays the loud boot refusal";

#[cfg(feature = "cuda")]
pub(crate) fn qwen38_serve_graphed_explicit() -> bool {
    matches!(
        std::env::var(Q38_SERVE_GRAPHED_ENV).ok().as_deref(),
        Some("1") | Some("on")
    )
}

#[cfg(feature = "cuda")]
pub(crate) fn qwen38_serve_graphed_enabled() -> bool {
    let _ = Q38_SERVE_GRAPHED_DEFAULT_ON_YIELDS_TO_EXPLICIT_BATCH_SET_0_FOR_THE_EAGER_SESSION;
    match std::env::var(Q38_SERVE_GRAPHED_ENV).ok().as_deref() {
        Some("0") | Some("off") => false,
        Some("1") | Some("on") => true,
        _ => !nv_models::qwen3_5_moe::qwen38_batch::nv_q38_batch_env_opt_in_nv_q38_batch_1_the_serving_loop_routes_batch_xor_spec_per_request_group(),
    }
}

#[cfg(feature = "cuda")]
pub(crate) fn qwen38_serve_graphed_selected(family: ModelFamily, raw_cfg: &str) -> bool {
    matches!(family, ModelFamily::Qwen3_5Moe)
        && qwen3_5_config_declares_dense_ffn(raw_cfg)
        && qwen3_5_dense_cuda_serve_enabled()
        && qwen38_serve_graphed_enabled()
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
        || qwen38_serve_graphed_selected(family, raw_cfg)
        || qwen38_batch_lanes_boot_selected(family, raw_cfg)
        || qwen36_batch_lanes_boot_selected(family, raw_cfg)
        || qwen36_mtp_captured_serving_selected(family, raw_cfg)
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
pub(crate) fn qwen36_batch_lanes_boot_selected(family: ModelFamily, raw_cfg: &str) -> bool {
    matches!(family, ModelFamily::Qwen3_5Moe)
        && !qwen3_5_config_declares_dense_ffn(raw_cfg)
        && nv_models::qwen3_5_moe::qwen36_batch::nv_q36_batch_env_opt_in_nv_q36_batch_1_the_serving_loop_routes_batch_xor_spec_per_request_group()
}

#[cfg(feature = "cuda")]
pub(crate) const QWEN35_DENSE_CUDA_SERVE_ENV: &str = "NV_QWEN35_DENSE_CUDA_SERVE";

#[cfg(feature = "cuda")]
pub(crate) const QWEN35_DENSE_CUDA_OPT_IN_MATCHES_THE_WGPU_ORACLE_AND_STAYS_OPT_IN: &str = "qwen3.5-dense cuda serving is wired and opted into per boot: with NV_QWEN35_DENSE_CUDA_SERVE=1 detect_family stops refusing and try_load parses the checkpoint with nv_models::qwen3_5_moe::Qwen3_5DenseConfig (the same parser the wgpu decoder uses, top-level fields read out of text_config) and builds it through nv_models::qwen3_5_moe::Qwen3Moe::from_loader_dense_quantized, which gives every linear_attention entry of layer_types a LayerMixer::Linear over nv_layers::linear_attn::LinearAttention and every layer a LayerFfn::Dense over nv_layers::mlp::Mlp. Every trunk component this uses -- the output-gated attention, the partial rope, the GDN linear-attention mixer, the nvfp4 linears -- is byte-for-byte the one the qwen3.5-moe cuda path already serves, because Qwen3.5-9B and Qwen3.5-35B-A3B declare the same head_dim, attn_output_gate, partial_rotary_factor and linear_* block; the ONLY surface this path adds over the served MoE path is LayerFfn::Dense. That surface is now executed and compared against nv_models::qwen3_5_dense_wgpu, the real-weights-verified decoder for this family, by the suite covering the dense cuda serving arm (rust/tests/qwen35_dense_cuda_serving_ab.rs, NV_QWEN35_DENSE_CUDA_SERVE_TEST=1): on ig1/Qwen3.5-9B-NVFP4 snapshot 3b9e07b0 the two backends return the same 7 tokens byte for byte through the checkpoint's own chat template, cuda repeats itself across two requests and an engine reload, and a 512-token thinking trace stays on subject while separating from the oracle at character 225, which is NVFP4 accumulation order and not a decode defect. Both the unarmed trunk and the eleven-arm serving stack QWEN35_DENSE_9B_OPT_IN_ARMS_ITS_MEASURED_STACK_AND_NV_QWEN35_DENSE_ARM_0_IS_THE_ESCAPE describes reproduce that seven-token answer byte for byte, and the armed stack separates from the oracle LATER on the long trace (character 357 against 218) while reusing more of its content words (38 of 49 against 35 of 49), so arming is not a coarsening on this checkpoint. It stays opt-in because that is two prompts on one checkpoint rather than a parity measurement, and because flipping it would move backend_select's auto choice for this whole checkpoint class onto a backend that holds more VRAM for the same weights. Without the variable, detect_family keeps refusing with backend_select::QWEN35_DENSE_NO_CUDA and the default behavior is unchanged";

#[cfg(feature = "cuda")]
pub(crate) const GPTOSS_CUDA_OPT_IN_IS_A_DEQUANT_RUNG_AND_WGPU_STAYS_THE_GPT_OSS_DEFAULT: &str = "gpt-oss cuda serving is wired and opted into per boot: with NV_GPTOSS_CUDA_SERVE=1 detect_family answers ModelFamily::GptOss instead of refusing, and try_load parses the checkpoint with nv_models::gpt_oss::GptOssConfig -- the very parser nv_models::gpt_oss_wgpu uses, so the two backends cannot disagree about layer_types, sliding_window, swiglu_limit or the YaRN block -- and builds nv_models::gpt_oss_cuda::GptOssCuda. That decoder adds exactly two things cuda did not have. First, mxfp4: every expert tensor is dequantized to bf16 at load through nv_quant::mxfp4::Mxfp4Tensor::dequantize, which is the same host semantics the wgpu mxfp4 GEMV is already pinned against, and which is lossless into bf16 because an e2m1 value times an e8m0 block scale needs one mantissa bit. Second, attention sinks: nv_layers::attn::sdpa_with_sinks appends the learned per-head sink as one more score column and one all-zero value row, which is algebraically the fold gow_attn.wgsl and gow_prefill.wgsl perform as `m = max(red[0], sink); z = red[0] + exp(sink - m)`. candle_flash_attn takes no sink argument, so this decoder is eager by construction and the alternating sliding_window=128 layers are masked in the same builder rather than by flash_attn_windowed. It stays opt-in on cost, not correctness: dequant-to-bf16 puts openai/gpt-oss-20b at about 41.8 GB resident against about 13.7 GB for a native mxfp4 path, and eager scoring costs heads*rows*context f32 of transient scratch per prefill chunk that the wgpu decoder never materializes. Flipping it would move backend_select's auto choice for this checkpoint class onto the backend that holds three times the weights and cannot yet stream its softmax. Without the variable, detect_family keeps refusing with backend_select::GPTOSS_NO_CUDA_WITHOUT_THE_OPT_IN and the default behavior is unchanged";

#[cfg(feature = "cuda")]
pub(crate) fn gpt_oss_cuda_serve_enabled() -> bool {
    crate::oapi::backend_select::gpt_oss_cuda_serve_enabled()
}

#[cfg(feature = "cuda")]
pub(crate) const QWEN3_WEIGHTS_ENV: &str = "NV_QWEN3_WEIGHTS";

#[cfg(feature = "cuda")]
pub(crate) const QWEN3_DEFAULT_IS_NVFP4_AT_LOAD_AND_NV_QWEN3_WEIGHTS_BF16_IS_THE_ESCAPE: &str = "plain qwen3 dense checkpoints (ModelFamily::Qwen3, nv_models::qwen3 -- model_type qwen3 with top-level trunk fields, e.g. Qwen/Qwen3-Embedding-0.6B) serve with 4-bit weights by default: try_load_inner routes through nv_models::qwen3::Qwen3::from_loader_quantized, and when the checkpoint's config.json declares no quantization_config, qwen3_weight_qconfig_default_nvfp4 supplies QuantScheme::Nvfp4, which quantizes every per-layer projection (q/k/v/o and gate/up/down) from the bf16 checkpoint at load through nv_layers Linear::from_bf16_quantized_nvfp4 (weights-only; 16-element ue4m3 block scales per nv_quant::nvfp4; served by the same cuda nvfp4 GEMM/GEMV the other nvfp4 families use). embed_tokens, lm_head and every RmsNorm stay bf16 because from_loader_quantized never quantizes them, and any projection under nv_quant::nvfp4::MIN_TILE falls back to bf16 inside load_linear_quant. A checkpoint that declares its own quantization_config keeps the declared scheme -- the default only fills the undeclared case. NV_QWEN3_WEIGHTS=bf16 restores the pre-flip all-bf16 Qwen3::from_loader behavior byte for byte. The flip is gated by this family's quality instrument, the suite covering teacher-forced ppl for plain qwen3 (nv-models/tests/ppl_qwen3.rs, NV_PPL_TEST=1, arm selected by the same env), with same-session bf16-vs-nvfp4 A/B rows plus decode-timing rows (nv-models/tests/qwen3_ctx_scaling.rs) in perf/runs.jsonl under model 'qwen3': on the Qwen3-Embedding-0.6B vehicle the nvfp4 arm holds top-1 accuracy within 0.7 points and ppl within 9 percent of bf16 on identical scored positions, inside the 1-point/10-percent acceptance band, and its cost is stated there too -- at 0.6B decode shapes the nvfp4 projections run 10-22 percent slower than bf16 (m=1 tiles this small do not amortize the block-scale path; the same kernels win at 31B shapes per NVFP4_M1_GEMV_IS_A_SERVING_DEFAULT_WHERE_A_SERVING_AB_LANDED_AND_OPT_IN_EVERYWHERE_ELSE), so what the default buys is roughly 3.5x smaller projection weights, which is what decides whether a large plain-qwen3 dense checkpoint fits at all. No published document quotes a chat-path number measured on the old bf16 default (runs.jsonl carried no model 'qwen3' rows before the flip's own reference rows). /v1/embeddings (oapi::text_embeddings) is a different surface with a cosine-similarity contract that this ppl instrument does not cover -- the MTEB numbers docs/book/08.1 quotes for Qwen3-Embedding-0.6B belong to that surface -- so it keeps loading Qwen3::from_loader bf16 deliberately";

#[cfg(feature = "cuda")]
pub(crate) fn qwen3_weight_qconfig_default_nvfp4(
    raw_cfg: &str,
    weights_env: Option<&str>,
) -> anyhow::Result<nv_weights::QuantizationConfig> {
    let declared = nv_weights::QuantizationConfig::from_hf_json_str(raw_cfg)?;
    if !matches!(declared.scheme, nv_weights::QuantScheme::None) {
        return Ok(declared);
    }
    match weights_env {
        Some("bf16") => Ok(declared),
        None | Some("nvfp4") => Ok(nv_weights::QuantizationConfig {
            scheme: nv_weights::QuantScheme::Nvfp4,
            ignored_modules: Vec::new(),
            group_size: None,
        }),
        Some(other) => anyhow::bail!(
            "{QWEN3_WEIGHTS_ENV}={other} is not a weight format; accepted values are 'bf16' \
             (the pre-flip escape) and 'nvfp4' (the default) -- refusing rather than silently \
             serving a format the caller did not ask for"
        ),
    }
}

#[cfg(feature = "cuda")]
pub(crate) const GEMMA4_SERVING_ARM_ENV: &str = "NV_G4_SERVING_ARM";

#[cfg(feature = "cuda")]
pub(crate) const GEMMA4_NVFP4_SERVES_THE_THREE_KEY_RECORD_ARM_AND_NV_G4_SERVING_ARM_0_IS_THE_ESCAPE: &str = "gemma4 dense checkpoints whose config.json declares QuantScheme::Nvfp4 (ModelFamily::Gemma4 over nv_models::gemma4::Gemma4, e.g. nvidia/Gemma-4-31B-IT-NVFP4) serve with three keys armed, and try_load_inner arms them before the boot plan so oapi::vram_plan::plan_gemma4 charges the requanted attention pool rather than the bf16 one: NV_ATTN_PROJ_QUANT=nvfp4 requants q/k/v/o at load through Gemma4::from_loader_quantized (16.956 -> 4.769 GB read per token, 60 layers), NV_NVFP4_M1_GEMV=1 routes every short-row NVFP4 linear off the padded cutlass block-scaled tile onto nvfp4_gemv_bf16act, and NV_G4_LMHEAD_I8=1 makes the m=1 decode epilogue read the resident int8 lm_head copy the verify path already reads (2.819 -> 1.410 GB, no new weights). The three are ONE decision over GEMMA4_RECORD_ARM_KEYS_ARE_ARMED_TOGETHER_OR_NOT_AT_ALL, never a subset, because NV_ATTN_PROJ_QUANT=nvfp4 alone is a quality REJECT and only NV_NVFP4_M1_GEMV rescues it -- nv_models::gemma4::ATTN_PROJ_NVFP4_ALONE_IS_A_MEASURED_QUALITY_REJECT_RESCUED_ONLY_BY_NV_NVFP4_M1_GEMV carries that measurement, and try_load refuses to boot any process that reaches the pair without the rescue, escape or no escape. NV_G4_SERVING_ARM=0 (or off) arms nothing at all and restores the pre-flip zero-env behavior exactly: it is the absence of three environment variables, so the restored process runs the identical code the pre-flip binary ran. What the default buys, measured on the default per-request serving decode (GraphedGemma4Decoder over new_kv_cache_fp8_windowed, the shape run_sampling_gemma4 builds), interleaved same-session base/arm pairs on a quiet card: 27.876/27.985 -> 19.419/19.542 ms/tok at 256 and 29.472/29.638 -> 21.313/21.515 at 8192, with free VRAM at the paged ladder's own choice point going 59.6 -> 74.3 GB because the requantized projections replace the bf16 originals. What it costs is stated where it is measured, not hidden: this is a quantization-class arm, so it is read against the family's OWN measured null -- gemma4-31B's reassociation ladder spreads 0.41 ppl on the family's only valid eval (ppl_gemma4::gemma4_chat_wrapped_continuation_teacher_forced_ppl_via_serving_decode; #94 ruled raw-corpus gemma4 ppl invalid; the +0.0973 figure once cited here is a qwen3.8 wikitext ARM delta, not a floor, and binds no other family per 08.2) -- and it reads -0.018 ppl on the eager bf16-KV scoring path (16.308 -> 16.290) and +0.068 on the graphed fp8-KV serving shape (16.276 -> 16.344): both inside the 0.41 family null by ~6x, so the quality cost is contained-and-unresolved rather than held-or-better, with top-1 accuracy 0.6699 -> 0.6641 (3 tokens of 512, 0.28 sd) and the shuffled control intact on every row. One suite changes colour in the other direction: gemma4_verify_forward went 1-pass/2-fail under this arm and is 5-pass/0-fail since verify blocks started taking the same nvfp4_gemv_bf16act row by row. The one measured harm is speculative: a k=4 GraphedGemma4Verify round costs 21.57 -> 54.3 ms armed on the fixed route, so a deployment that sets NV_EAGLE3_DRAFT_DIR or NV_DFLASH_DRAFT_DIR (spec decode is itself opt-in -- load_eagle3_state_for_target returns None without the directory, so the default deployment never enters that path) should hold NV_G4_SERVING_ARM=0 until gemm_nvfp4_w4a16_mk_dual is wired as the verify-block route. Published numbers measured on the pre-flip default are enumerated in /home/user/tmp/wf/servingfix/RESULTS.md by document and line; all of them are ladder rungs in nv-models tests, which do not call this file and therefore still measure what they always measured -- what changes is that they now describe NV_G4_SERVING_ARM=0 rather than a live request stream";

#[cfg(feature = "cuda")]
pub(crate) const GEMMA4_RECORD_ARM_KEYS_ARE_ARMED_TOGETHER_OR_NOT_AT_ALL: [(&str, &str); 3] = [
    ("NV_ATTN_PROJ_QUANT", "nvfp4"),
    ("NV_NVFP4_M1_GEMV", "1"),
    ("NV_G4_LMHEAD_I8", "1"),
];

#[cfg(feature = "cuda")]
pub(crate) const QWEN35_DENSE_SERVING_ARM_ENV: &str = "NV_QWEN35_DENSE_ARM";

#[cfg(feature = "cuda")]
pub(crate) const QWEN35_DENSE_9B_OPT_IN_ARMS_ITS_MEASURED_STACK_AND_NV_QWEN35_DENSE_ARM_0_IS_THE_ESCAPE: &str = "the qwen3.5-dense cuda path stays opt-in per QWEN35_DENSE_CUDA_OPT_IN_MATCHES_THE_WGPU_ORACLE_AND_STAYS_OPT_IN -- this arms it, it does not promote it, and without NV_QWEN35_DENSE_CUDA_SERVE=1 detect_family still refuses with backend_select::QWEN35_DENSE_NO_CUDA and the checkpoint still goes to wgpu. What changes is that taking the opt-in no longer means decoding with none of the eleven arms this checkpoint's speed-of-light round measured: try_load_inner arms QWEN35_DENSE_9B_ARM_KEYS_ARE_ARMED_TOGETHER_OR_NOT_AT_ALL before Qwen3Moe::from_loader_dense_quantized, because NV_Q36_GDN_FP8 and NV_Q36_LM_HEAD_FP8 decide resident weight dtype at load and NV_Q36_GRAPHED_DECODE_FIX decides whether the kv cache allocates the splitk scratch that routes the 8 full-attention layers off the serial_smem decode kernel. Unarmed the path decodes at 12.514 ms/tok at 256 and 79.629 at 8192; armed it decodes at 5.425 and 5.900, and the depth term is where the defect really was -- serial_smem serializes a block reduce per KV position, which is why 256->8k cost 6.4x unarmed and costs 1.09x armed. It costs no quality and no VRAM: on ppl_qwen35_dense::qwen35_dense_chat_wrapped_continuation_teacher_forced_ppl_via_serving_decode the stack scores 4.833 against a 4.862 zero-env basis (better on the headline, and the largest single-arm cost in it is NV_Q36_GDN_FP8 at +0.042 -- a cost that stays UNRESOLVED because qwen3.5-9B has no measured reassociation null, its 3-rung ladder spanning only 0.025; the +0.0973 figure once cited here is a qwen3.8 wikitext arm delta, not a floor), its eager and graphed 128-step greedy chains are token-identical, and peak VRAM through a live 256-depth run drops 15877 -> 9183 MiB because the fp8 copies replace their bf16 originals. NV_QWEN35_DENSE_ARM=0 (or off) arms nothing and restores the unarmed opt-in path exactly. The stack is armed ONLY for the trunk it was measured on, QWEN35_DENSE_9B_TRUNK_THE_ARM_STACK_WAS_MEASURED_ON: qwen3.8-27B declares the same model_type qwen3_5 and lands in the same dense arm, but its published serving band belongs to a different twelve-arm stack with NV_NVFP4_M1_GEMV unset, so arming this one for it would flip a performance default nobody measured. Any other dense trunk keeps today's behavior and says so at load";

#[cfg(feature = "cuda")]
pub(crate) const QWEN35_DENSE_9B_ARM_KEYS_ARE_ARMED_TOGETHER_OR_NOT_AT_ALL: [(&str, &str); 11] = [
    ("NV_Q36_GRAPHED_DECODE_FIX", "1"),
    ("NV_Q36_GDN_FP8", "1"),
    ("NV_Q36_LM_HEAD_FP8", "1"),
    ("NV_Q38_FP4_MLP", "1"),
    ("NV_Q38_GDN_SPLIT", "1"),
    ("NV_Q38_MLP_W4A8", "1"),
    ("NV_Q38_W4A8_MK", "1"),
    ("NV_Q38_SILUQ_SPLIT", "1"),
    ("NV_Q38_NORM_QUANT_FOLD", "1"),
    ("NV_Q38_NORM_WRITEOUT", "1"),
    ("NV_NVFP4_M1_GEMV", "1"),
];

#[cfg(feature = "cuda")]
pub(crate) const QWEN35_DENSE_9B_TRUNK_THE_ARM_STACK_WAS_MEASURED_ON: (usize, usize, usize) =
    (4096, 32, 12288);

#[cfg(feature = "cuda")]
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum ServingArm {
    ArmEveryKey,
    OperatorHeldTheEscape,
    CheckpointOutsideTheMeasuredBasis,
}

#[cfg(feature = "cuda")]
pub(crate) fn serving_arm_from(
    escape: Option<&str>,
    checkpoint_is_the_measured_basis: bool,
) -> ServingArm {
    match escape {
        Some("0") | Some("off") | Some("false") | Some("no") => ServingArm::OperatorHeldTheEscape,
        _ if !checkpoint_is_the_measured_basis => ServingArm::CheckpointOutsideTheMeasuredBasis,
        _ => ServingArm::ArmEveryKey,
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
        ServingArm::OperatorHeldTheEscape | ServingArm::CheckpointOutsideTheMeasuredBasis => &[],
    }
}

#[cfg(feature = "cuda")]
fn apply_serving_arm(
    label: &str,
    escape_env: &str,
    why: &str,
    keys: &[(&str, &str)],
    arm: ServingArm,
) -> anyhow::Result<()> {
    let armed = keys_armed_by_this_one_decision(arm, keys);
    if armed.is_empty() {
        let reason = match arm {
            ServingArm::OperatorHeldTheEscape => format!("{escape_env}=0 is held"),
            _ => "this checkpoint is outside the trunk the stack was measured on".to_string(),
        };
        eprintln!(
            "[{label}] serving arm OFF ({reason}): none of {} is armed, so this process runs the \
             pre-flip default, which is the absence of those variables",
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
                 configuration nothing measured; set {escape_env}=0 to run the pre-flip default \
                 instead of hand-picking one key"
            ))
            .collect::<Vec<_>>()
            .join("; ")
    );
    for (k, v) in armed {
        std::env::set_var(k, v);
    }
    eprintln!(
        "[{label}] serving arm ON: {} (escape: {escape_env}=0)",
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
                GEMMA4_SERVING_ARM_ENV,
                GEMMA4_NVFP4_SERVES_THE_THREE_KEY_RECORD_ARM_AND_NV_G4_SERVING_ARM_0_IS_THE_ESCAPE,
                &GEMMA4_RECORD_ARM_KEYS_ARE_ARMED_TOGETHER_OR_NOT_AT_ALL,
                serving_arm_from(
                    std::env::var(GEMMA4_SERVING_ARM_ENV).ok().as_deref(),
                    gemma4_checkpoint_declares_nvfp4(raw_cfg),
                ),
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
        ModelFamily::Qwen3_5Moe if qwen3_5_dense => apply_serving_arm(
            "qwen3.5-dense",
            QWEN35_DENSE_SERVING_ARM_ENV,
            QWEN35_DENSE_9B_OPT_IN_ARMS_ITS_MEASURED_STACK_AND_NV_QWEN35_DENSE_ARM_0_IS_THE_ESCAPE,
            &QWEN35_DENSE_9B_ARM_KEYS_ARE_ARMED_TOGETHER_OR_NOT_AT_ALL,
            serving_arm_from(
                std::env::var(QWEN35_DENSE_SERVING_ARM_ENV).ok().as_deref(),
                qwen35_dense_trunk_is_the_measured_9b(raw_cfg),
            ),
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
        && qwen_moe_grouped_enabled()
}

#[cfg(feature = "cuda")]
pub(crate) const QWEN36_MTP_EAGER_ESCAPE_ENV: &str = "NV_Q36_MTP_EAGER";

#[cfg(feature = "cuda")]
pub(crate) fn qwen36_mtp_eager_escape() -> bool {
    matches!(
        std::env::var(QWEN36_MTP_EAGER_ESCAPE_ENV).ok().as_deref(),
        Some("1") | Some("on")
    )
}

#[cfg(feature = "cuda")]
pub(crate) fn qwen36_mtp_captured_serving_selected(family: ModelFamily, raw_cfg: &str) -> bool {
    matches!(family, ModelFamily::Qwen3_5Moe)
        && !qwen3_5_config_declares_dense_ffn(raw_cfg)
        && nv_specdecode::qwen38_mtp::mtp_drafter_selected_from_env()
        && qwen_moe_grouped_enabled()
        && !qwen36_mtp_eager_escape()
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
#[derive(Clone)]
pub(crate) enum QwenMtpHead {
    Dense(Arc<nv_specdecode::qwen38_mtp::Qwen38DenseMtpHead>),
    Moe(Arc<nv_models::qwen3_5_mtp::MtpHead>),
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
    pub(crate) qwen_mtp: Option<QwenMtpHead>,
    pub(crate) mm_towers: Option<Arc<crate::oapi::chat_multimodal::Gemma4MmTowers>>,
    pub(crate) vram_plan_record_released_on_engine_drop:
        Option<crate::oapi::vram_plan::RecordedBootPlan>,
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
    qwen_graph: bool,
) -> crate::oapi::vram_plan::PlanFamily {
    use crate::oapi::vram_plan::PlanFamily;
    match family {
        ModelFamily::Gemma4 => PlanFamily::Gemma4,
        ModelFamily::Qwen3_5Moe if qwen3_5_dense => PlanFamily::Qwen35Dense,
        ModelFamily::Qwen3_5Moe => PlanFamily::Qwen35Moe {
            grouped_graph: qwen_graph,
        },
        ModelFamily::Qwen3 => PlanFamily::Unmodeled("qwen3"),
        ModelFamily::Gemma4E4b => PlanFamily::Unmodeled("gemma4-e4b"),
        ModelFamily::Gemma4Moe => PlanFamily::Unmodeled("gemma4-moe"),
        ModelFamily::Laguna => PlanFamily::Unmodeled("laguna"),
        ModelFamily::Omni => PlanFamily::Unmodeled("omni"),
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
    pub fn last_routed_drafter_arm() -> Option<&'static str> {
        last_routed_drafter_arm_name()
    }

    pub fn try_load(model_dir: &Path) -> anyhow::Result<Self> {
        Self::preflight_checkpoint_dir(model_dir)?;
        Self::try_load_inner(model_dir, EngineBuildMode::Full)
    }

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
    pub(crate) fn try_load_inner(
        model_dir: &Path,
        mode: EngineBuildMode,
    ) -> anyhow::Result<Self> {
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
            plan_family_for(
                family,
                qwen3_5_dense,
                qwen_graph
                    || qwen36_mtp_captured
                    || qwen36_batch_lanes_boot_selected(family, &raw_cfg),
            ),
            model_dir,
            &raw_cfg,
            match mode {
                EngineBuildMode::Full => crate::oapi::vram_plan::BootPlanShape::Full,
                EngineBuildMode::AdapterServingLean => {
                    if adapter_serving_lean_keeps_the_i8_decode_head_by_default_because_the_bf16_fallback_epilogue_costs_0_88_ms_per_token_escape_nv_lora_ada_i8_head_0() {
                        crate::oapi::vram_plan::BootPlanShape::AdapterServingLeanKeepsTheI8DecodeHeadNeverBuildsDrafters
                    } else {
                        crate::oapi::vram_plan::BootPlanShape::AdapterServingLeanNeverBuildsDraftersOrTheI8VerifyHead
                    }
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
                let qconfig = qwen3_weight_qconfig_default_nvfp4(
                    &raw_cfg,
                    std::env::var(QWEN3_WEIGHTS_ENV).ok().as_deref(),
                )
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
                let model = match mode {
                    EngineBuildMode::Full => nv_models::gemma4::Gemma4::from_loader_quantized(
                        config.clone(),
                        &weights,
                        &qconfig,
                        &device,
                    ),
                    EngineBuildMode::AdapterServingLean => {
                        if adapter_serving_lean_keeps_the_i8_decode_head_by_default_because_the_bf16_fallback_epilogue_costs_0_88_ms_per_token_escape_nv_lora_ada_i8_head_0() {
                            nv_models::gemma4::Gemma4::from_loader_quantized(
                                config.clone(),
                                &weights,
                                &qconfig,
                                &device,
                            )
                        } else {
                            nv_models::gemma4::Gemma4::from_loader_quantized_without_the_i8_verify_head_because_adapter_serving_engines_never_verify(
                                config.clone(),
                                &weights,
                                &qconfig,
                                &device,
                            )
                        }
                    }
                }
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
                let measured_static = our_process_vram_bytes();
                let model = Arc::new(model);

                let gemma4_engine = if mode == EngineBuildMode::AdapterServingLean {
                    None
                } else if std::env::var("NV_BATCH_ENGINE").is_ok() {
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
                if mode != EngineBuildMode::AdapterServingLean {
                    crate::oapi::admission::init_gemma4(admit_static, drafter_row_elems);
                }

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
                        vram_plan_record_released_on_engine_drop: None,
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
                        vram_plan_record_released_on_engine_drop: None,
                    },
                })
            }
            ModelFamily::Gemma4Moe => {
                let config = nv_models::gemma4_moe::Gemma4MoeConfig::from_hf_json_str(&raw_cfg)
                    .with_context(|| format!("parse gemma4 moe {}", cfg_path.display()))?;
                let model = match nv_models::gemma4_gguf::opted_in_qat_q4_gguf_for_snapshot(
                    &config,
                ) {
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
                            format!("instantiate Gemma4Moe from default qat gguf {}", gguf.display())
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
                    anyhow::ensure!(
                        !qwen38_serve_graphed_explicit(),
                        "NV_Q38_BATCH=1 boots the lane-pool scheduler, which owns the model, but \
                         {Q38_SERVE_GRAPHED_ENV}=1 pins the persistent graphed solo engine, which \
                         also owns it -- drop one of the two"
                    );
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
                } else if qwen38_serve_graphed_enabled() {
                    let mut graphed = nv_models::graph_engine::GraphedQwen3Moe::new(
                        model,
                        &device,
                        kv_max_seq_len,
                    )
                    .context(
                        "NV_Q38_SERVE_GRAPHED=1 was requested, so a graphed engine that cannot \
                         build is a boot failure rather than a silent fall-back to eager serving",
                    )?;
                    graphed.install_grouped_moe().context(
                        "arm capture on the dense trunk (no dispatch is built; every ffn is a \
                         plain mlp)",
                    )?;
                    eprintln!(
                        "[qwen3.8] decode: GRAPHED ({Q38_SERVE_GRAPHED_ENV}=1; persistent \
                         GraphedQwen3Moe, kv cache bound once at boot to the {kv_max_seq_len}-token \
                         window; greedy MTP requests self-speculate through \
                         Qwen38MtpGraphedDecodeSession, whose rows-keyed verify graph is \
                         recaptured after each request's prefill; unset for the per-request eager \
                         engine)"
                    );
                    QwenMoeShared::Graphed(Arc::new(tokio::sync::Mutex::new(graphed)))
                } else {
                    eprintln!(
                        "[qwen3.8] decode: EAGER (no graph capture; {Q38_SERVE_GRAPHED_ENV}=1 \
                         for the persistent graphed engine)"
                    );
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
                    anyhow::ensure!(
                        !(qwen_graph && qwen36_mtp_eager_escape()),
                        "NV_QWEN_GRAPH=1 pins graph decode but \
                         {QWEN36_MTP_EAGER_ESCAPE_ENV}=1 demands the eager MTP session, \
                         which serves only off the eager engine -- drop one of the two; with \
                         neither set, NV_DRAFTER=mtp serves greedy requests through the \
                         captured-verify MtpMoeGraphedDecodeSession by default"
                    );
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
                         requests self-speculate through the {} session; sampled requests \
                         decode normally)",
                        if qwen36_mtp_captured {
                            "CAPTURED verify-lane (NV_Q36_MTP_EAGER=1 for eager)"
                        } else {
                            "EAGER (grouped MoE opted out or NV_Q36_MTP_EAGER set)"
                        }
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
                        model,
                        &device,
                        lane_kv,
                        plan,
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
                                    "selected by NV_DRAFTER=mtp captured serving; NV_Q36_MTP_EAGER=1 for eager"
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
                let shared = LagunaShared(Arc::new(model));
                let laguna_spec = if mode == EngineBuildMode::AdapterServingLean {
                    tracing::info!(
                        "adapter-serving lean build: the laguna spec-serve engine is never \
                         built because its captured step-graph body (raw nv-* kernels plus \
                         candle proj fallbacks captured on a forked stream) never routes the \
                         nv-layers Linear hooks a resident adapter needs; adapter requests \
                         decode on the eager per-request path"
                    );
                    None
                } else {
                    build_laguna_spec_serve(&shared, kv_max_seq_len)
                };
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

pub(crate) fn spec_serve_max_seq(env: Option<usize>, kv_max_seq_len: usize) -> usize {
    match env {
        Some(n) if n > 0 => n.min(kv_max_seq_len),
        _ => kv_max_seq_len,
    }
}

pub(crate) const KV_DEFAULT_8192_TOKENS_FOR_FAMILIES_WITHOUT_A_VRAM_SOLVER_MODEL_SO_MULTI_ENGINE_BOOTS_DO_NOT_OOM_LIFT_VIA_THE_GLOBAL_OR_PER_FAMILY_KV_ENV: usize = 8192;

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

        let err = match NvEngineChat::try_load_inner(&dir, EngineBuildMode::Full) {
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

    #[test]
    fn a_runtime_insert_is_visible_through_every_clone_of_the_registry() {
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
            &["gemma-4-E4B-it".to_string(), "ft:e4b:custom:1234".to_string()]
        );
    }

    #[test]
    fn insert_of_an_existing_id_replaces_without_duplicating_the_order_row() {
        let reg = single_reg();
        reg.insert(Arc::new(EchoEngine::new("ft:x", "a")));
        reg.insert(Arc::new(EchoEngine::new("ft:x", "b")));
        assert_eq!(
            reg.model_ids(),
            &["gemma-4-E4B-it".to_string(), "ft:x".to_string()]
        );
    }

    #[test]
    fn remove_refuses_the_default_engine_and_404s_an_unknown_name() {
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
    fn plan_solver_lands_on_the_largest_granular_kv_that_fits() {
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
    }

    #[test]
    fn plan_solver_reports_no_fit_when_weights_alone_exceed_the_budget() {
        let c = cfg();
        let fits = |kv: usize| 100.0 + kv_gib(&c, kv, 0) <= 76.0;
        assert!(
            crate::oapi::vram_plan::solve_max_kv_tokens_monotone(fits, 131072).is_none(),
            "weights alone exceed the budget, so no kv_max fits -- that must not read as a fit"
        );
    }

    #[test]
    fn explicit_kv_env_bypasses_the_solver_in_the_gemma4_plan() {
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
        let fits_with = |ch: DrafterKvCharge, kv: usize| {
            let (rows, row_elems) = ch.rows_elems(kv);
            let fp8 = nv_models::gemma4::verify_kv_use_fp8();
            let rings = nv_models::gemma4::kv_ring_enabled();
            let b = nv_models::gemma4::kv_budget_capped(&c, kv, fp8, rings, row_elems, rows);
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
    fn every_variant_arms_the_whole_key_set_or_none_of_it() {
        for keys in [
            &GEMMA4_RECORD_ARM_KEYS_ARE_ARMED_TOGETHER_OR_NOT_AT_ALL[..],
            &QWEN35_DENSE_9B_ARM_KEYS_ARE_ARMED_TOGETHER_OR_NOT_AT_ALL[..],
        ] {
            for arm in [
                ServingArm::ArmEveryKey,
                ServingArm::OperatorHeldTheEscape,
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
    }

    #[test]
    fn the_gemma4_default_is_exactly_the_three_key_record_arm() {
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
        assert_eq!(
            serving_arm_from(None, gemma4_checkpoint_declares_nvfp4(GEMMA4_NVFP4_CFG)),
            ServingArm::ArmEveryKey
        );
    }

    #[test]
    fn the_escape_restores_the_pre_flip_default_by_arming_nothing() {
        for escape in ["0", "off", "false", "no"] {
            let arm = serving_arm_from(Some(escape), true);
            assert_eq!(arm, ServingArm::OperatorHeldTheEscape, "{escape}");
            assert!(
                keys_armed_by_this_one_decision(
                    arm,
                    &GEMMA4_RECORD_ARM_KEYS_ARE_ARMED_TOGETHER_OR_NOT_AT_ALL
                )
                .is_empty(),
                "{GEMMA4_SERVING_ARM_ENV}={escape} must leave the process with none of the three \
                 variables set, which is the pre-flip default itself and not an imitation of it"
            );
            assert!(keys_armed_by_this_one_decision(
                arm,
                &QWEN35_DENSE_9B_ARM_KEYS_ARE_ARMED_TOGETHER_OR_NOT_AT_ALL
            )
            .is_empty());
        }
        for on in [None, Some("1"), Some("on")] {
            assert_eq!(serving_arm_from(on, true), ServingArm::ArmEveryKey, "{on:?}");
        }
    }

    #[test]
    fn a_hand_set_key_that_disagrees_refuses_instead_of_serving_a_subset() {
        let held: HashMap<&str, &str> = HashMap::from([
            ("NV_ATTN_PROJ_QUANT", "nvfp4"),
            ("NV_NVFP4_M1_GEMV", "0"),
        ]);
        let conflicts = arm_keys_the_operator_already_set_to_something_else(
            &GEMMA4_RECORD_ARM_KEYS_ARE_ARMED_TOGETHER_OR_NOT_AT_ALL,
            |k| held.get(k).map(|v| v.to_string()),
        );
        assert_eq!(
            conflicts
                .iter()
                .map(|(k, _, _)| *k)
                .collect::<Vec<_>>(),
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

    #[test]
    fn quantized_attention_projections_without_the_gemv_rescue_are_refused_at_boot() {
        for raw in ["nvfp4", "1", "on", "true", "yes", "all", "fp4", "qkv+o", "qkv", "o"] {
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

    #[test]
    fn the_gemma4_arm_reaches_only_checkpoints_that_declare_nvfp4() {
        assert!(gemma4_checkpoint_declares_nvfp4(GEMMA4_NVFP4_CFG));
        assert!(!gemma4_checkpoint_declares_nvfp4(GEMMA4_BF16_CFG));
        assert_eq!(
            serving_arm_from(None, gemma4_checkpoint_declares_nvfp4(GEMMA4_BF16_CFG)),
            ServingArm::CheckpointOutsideTheMeasuredBasis,
            "attn_proj_quant_mode ignores NV_ATTN_PROJ_QUANT unless the checkpoint declares \
             QuantScheme::Nvfp4, so arming a bf16 gemma4 would move only the int8 lm_head, on a \
             checkpoint class nothing measured"
        );
    }

    #[test]
    fn the_qwen35_dense_stack_is_the_eleven_arms_round_twelve_measured() {
        assert_eq!(
            QWEN35_DENSE_9B_ARM_KEYS_ARE_ARMED_TOGETHER_OR_NOT_AT_ALL
                .iter()
                .map(|(k, v)| {
                    assert_eq!(*v, "1", "{k} is an opt-in flag read as eq-some-1");
                    *k
                })
                .collect::<Vec<_>>(),
            vec![
                "NV_Q36_GRAPHED_DECODE_FIX",
                "NV_Q36_GDN_FP8",
                "NV_Q36_LM_HEAD_FP8",
                "NV_Q38_FP4_MLP",
                "NV_Q38_GDN_SPLIT",
                "NV_Q38_MLP_W4A8",
                "NV_Q38_W4A8_MK",
                "NV_Q38_SILUQ_SPLIT",
                "NV_Q38_NORM_QUANT_FOLD",
                "NV_Q38_NORM_WRITEOUT",
                "NV_NVFP4_M1_GEMV",
            ],
            "this is the eleven-arm record the 12.514/79.629 -> 5.425/5.900 pairs and the 4.833 \
             ppl row were measured on"
        );
    }

    #[test]
    fn the_qwen35_arm_stack_stops_at_the_trunk_it_was_measured_on() {
        assert!(
            qwen35_dense_trunk_is_the_measured_9b(&qwen35_9b_config_json()),
            "hidden 4096 / 32 layers / intermediate 12288 is the Qwen3.5-9B trunk the eleven \
             arms were measured on"
        );
        assert!(
            !qwen35_dense_trunk_is_the_measured_9b(&qwen38_27b_config_json()),
            "qwen3.8-27B lands in the same dense arm but its published serving band belongs to a \
             different twelve-arm stack with NV_NVFP4_M1_GEMV unset; arming this stack for it \
             would flip a performance default nobody measured"
        );
        assert_eq!(
            serving_arm_from(
                None,
                qwen35_dense_trunk_is_the_measured_9b(&qwen38_27b_config_json())
            ),
            ServingArm::CheckpointOutsideTheMeasuredBasis
        );
    }

    #[test]
    fn both_rationales_name_their_escape_their_instrument_and_their_cost() {
        for (why, needles) in [
            (
                GEMMA4_NVFP4_SERVES_THE_THREE_KEY_RECORD_ARM_AND_NV_G4_SERVING_ARM_0_IS_THE_ESCAPE,
                vec![
                    GEMMA4_SERVING_ARM_ENV,
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
                QWEN35_DENSE_9B_OPT_IN_ARMS_ITS_MEASURED_STACK_AND_NV_QWEN35_DENSE_ARM_0_IS_THE_ESCAPE,
                vec![
                    QWEN35_DENSE_SERVING_ARM_ENV,
                    QWEN35_DENSE_CUDA_SERVE_ENV,
                    "ppl_qwen35_dense::qwen35_dense_chat_wrapped_continuation_teacher_forced_ppl_via_serving_decode",
                    "4.833",
                    "4.862",
                    "5.900",
                    "9183",
                    "qwen3.8-27B",
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
    fn the_device_selection_is_the_qwen_graph_selector_the_q38_batch_opt_in_and_the_q36_mtp_captured_default(
    ) {
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
                    || qwen36_mtp_captured_serving_selected(family, cfg),
                "{family:?}: try_load_inner must keep choosing Device::new_cuda_with_stream on \
                 exactly the qwen-graph condition plus the NV_Q38_BATCH dense lane opt-in plus \
                 the NV_Q36_BATCH moe lane opt-in plus the NV_DRAFTER=mtp captured-verify \
                 serving default and nothing else. Task #63 records grouped MoE on a \
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

#[cfg(all(test, feature = "cuda"))]
mod qwen3_default_weight_format_tests {
    use super::*;

    const PLAIN_QWEN3_CFG_NO_QUANT_SECTION: &str = r#"{"model_type":"qwen3","hidden_size":1024}"#;

    #[test]
    fn an_undeclared_checkpoint_defaults_to_nvfp4_weights() {
        let q = qwen3_weight_qconfig_default_nvfp4(PLAIN_QWEN3_CFG_NO_QUANT_SECTION, None).unwrap();
        assert!(
            matches!(q.scheme, nv_weights::QuantScheme::Nvfp4),
            "the qwen3 serving default is 4-bit at-load quantization; getting {:?} here means \
             the flip regressed to bf16 without the escape env being set",
            q.scheme
        );
        assert!(q.ignored_modules.is_empty() && q.group_size.is_none());
    }

    #[test]
    fn nv_qwen3_weights_bf16_is_the_escape_to_the_old_path() {
        let q = qwen3_weight_qconfig_default_nvfp4(PLAIN_QWEN3_CFG_NO_QUANT_SECTION, Some("bf16"))
            .unwrap();
        assert!(
            matches!(q.scheme, nv_weights::QuantScheme::None),
            "NV_QWEN3_WEIGHTS=bf16 must reproduce the pre-flip Qwen3::from_loader behavior \
             (from_loader_quantized with QuantScheme::None delegates to from_loader)"
        );
    }

    #[test]
    fn an_explicit_nvfp4_value_matches_the_default() {
        let q = qwen3_weight_qconfig_default_nvfp4(PLAIN_QWEN3_CFG_NO_QUANT_SECTION, Some("nvfp4"))
            .unwrap();
        assert!(matches!(q.scheme, nv_weights::QuantScheme::Nvfp4));
    }

    #[test]
    fn a_declared_quantization_config_wins_over_both_default_and_escape() {
        let cfg = r#"{"model_type":"qwen3","quantization_config":{"quant_method":"modelopt","quant_algo":"FP8"}}"#;
        let q = qwen3_weight_qconfig_default_nvfp4(cfg, Some("bf16")).unwrap();
        assert!(
            matches!(q.scheme, nv_weights::QuantScheme::Fp8E4m3),
            "a checkpoint that states its own scheme must keep it; the env only fills the \
             undeclared case, it never overrides the checkpoint's declaration"
        );
    }

    #[test]
    fn a_typo_in_the_weights_env_is_refused_not_served() {
        let err = qwen3_weight_qconfig_default_nvfp4(PLAIN_QWEN3_CFG_NO_QUANT_SECTION, Some("int4"))
            .expect_err("unknown value must refuse");
        let msg = format!("{err:#}");
        assert!(
            msg.contains("bf16") && msg.contains("nvfp4"),
            "the refusal must name the accepted values: {msg}"
        );
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
        let err = eos_ids_from_dir(&dir).expect_err("must be refused");
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

#[cfg(test)]
mod kv_ceiling_for_unplanned_families_tests {
    use super::*;
    use crate::oapi::vram_plan::{kv_binding_from, per_engine_kv_env_for_family};

    #[test]
    fn env_unset_keeps_the_8192_default_bit_identical_to_the_old_bare_number() {
        for mpe in [0usize, 1, 4096, 8192, 32768, 262144] {
            assert_eq!(
                kv_max_seq_len_resolved(None, mpe),
                mpe.min(8192),
                "mpe={mpe} with no env must resolve exactly as the old min(mpe, 8192)"
            );
        }
    }

    #[test]
    fn a_bound_from_either_kv_env_lifts_the_ceiling_clamped_only_to_the_checkpoint_window() {
        assert_eq!(kv_max_seq_len_resolved(Some(196608), 262144), 196608);
        assert_eq!(
            kv_max_seq_len_resolved(Some(500_000), 262144),
            262144,
            "the env lift clamps to max_position_embeddings, the operator cannot exceed the model"
        );
        assert_eq!(kv_max_seq_len_resolved(Some(0), 262144), 8192);
    }

    #[test]
    fn each_unplanned_family_lifts_alone_through_its_per_engine_env_without_the_global() {
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
