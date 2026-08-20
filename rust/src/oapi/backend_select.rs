use std::fmt;

use axum::extract::State;
use axum::http::{header, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::routing::get;
use axum::Router;
use serde_json::{json, Value};

use nv_layers::backend::{missing_on, probe_cuda, probe_wgpu, BackendKind, KernelId};

use crate::defaults;
use crate::AppState;

pub const WGPU_DECODERS_COMPILED_IN: bool = cfg!(feature = "wgpu");

pub const WGPU_FEATURE_OFF_REASON: &str = "this speaches-plus binary was built without the `wgpu` feature, so the nv-models wgpu decoders are not compiled in; rebuild with --features wgpu";

pub const AUTO_POLICY: &str = "auto is cuda-first: it returns cuda whenever the cuda probe succeeds and the model has no cuda-specific gap, because on every model measured on this box the cuda decode path is faster than the wgpu one (docs/book/08.4-PERFORMANCE.md, docs/book/05.1-wgpu-status.md) and only cuda has graph capture, paged KV and speculative decoding. auto falls back to wgpu only when cuda reports an explicit reason it cannot serve, so a wgpu answer is never a silent performance downgrade; and cuda is never returned for a model cuda cannot serve. when neither backend can serve, both reasons are returned instead of a silent default. an explicit NV_SERVE_BACKEND=wgpu is honoured without ever falling back to cuda.";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WgpuEvidence {
    RealWeights {
        checkpoint: &'static str,
        test: &'static str,
    },
    ArchitectureFamily {
        verified_sibling: &'static str,
    },
}

impl WgpuEvidence {
    pub fn kind(self) -> &'static str {
        match self {
            Self::RealWeights { .. } => "real-weights-decode",
            Self::ArchitectureFamily { .. } => "architecture-family-only",
        }
    }

    pub fn detail(self) -> String {
        match self {
            Self::RealWeights { checkpoint, test } => {
                format!("decoded {checkpoint} on real weights; gated by nv-models test {test}")
            }
            Self::ArchitectureFamily { verified_sibling } => format!(
                "same architecture family as {verified_sibling}, which is verified on real weights; this exact checkpoint has never been decoded on wgpu"
            ),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct WgpuDecoder {
    pub module: &'static str,
    pub entry: &'static str,
    pub evidence: WgpuEvidence,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ServeBackend {
    Cuda,
    Wgpu,
    Auto,
}

impl ServeBackend {
    pub fn name(self) -> &'static str {
        match self {
            Self::Cuda => "cuda",
            Self::Wgpu => "wgpu",
            Self::Auto => "auto",
        }
    }

    pub fn parse(s: &str) -> Result<Self, BackendSelectError> {
        match s.trim().to_ascii_lowercase().as_str() {
            "cuda" => Ok(Self::Cuda),
            "wgpu" => Ok(Self::Wgpu),
            "auto" => Ok(Self::Auto),
            other => Err(BackendSelectError::InvalidSelection(format!(
                "{}={:?} is not one of cuda|wgpu|auto (default: {})",
                defaults::env::NV_SERVE_BACKEND,
                other,
                defaults::serve_backend::DEFAULT,
            ))),
        }
    }

    pub fn from_env() -> Result<Self, BackendSelectError> {
        match std::env::var(defaults::env::NV_SERVE_BACKEND) {
            Err(_) => Ok(Self::Cuda),
            Ok(v) if v.trim().is_empty() => Ok(Self::Cuda),
            Ok(v) => Self::parse(&v),
        }
    }
}

impl fmt::Display for ServeBackend {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ModelClass {
    Gemma4Dense,
    Gemma4E4b,
    Gemma4Moe,
    Qwen36Moe,
    Qwen35Moe,
    Qwen35Dense,
    Qwen38Dense,
    Qwen38Max,
    GptOss,
    Laguna,
    DiffusionGemma,
    Unknown,
}

impl ModelClass {
    pub fn classify(model_id: &str) -> Self {
        let id = model_id.to_ascii_lowercase();

        if id.contains("diffusiongemma") || id.contains("diffusion_gemma") {
            return Self::DiffusionGemma;
        }
        if id.contains("e4b") {
            Self::Gemma4E4b
        } else if id.contains("gemma") && id.contains("a4b") {
            Self::Gemma4Moe
        } else if id.contains("gemma") {
            Self::Gemma4Dense
        } else if id.contains("qwen3.6") || id.contains("qwen3-6") || id.contains("a3b") {
            Self::Qwen36Moe
        } else if id.contains("qwen3.5") || id.contains("qwen3-5") {
            if id.contains("moe") {
                Self::Qwen35Moe
            } else {
                Self::Qwen35Dense
            }
        } else if id.contains("qwen3.8") || id.contains("qwen3-8") {
            if id.contains("2.4t") || id.contains("a95b") {
                Self::Qwen38Max
            } else {
                Self::Qwen38Dense
            }
        } else if id.contains("gpt-oss") || id.contains("gpt_oss") {
            Self::GptOss
        } else if id.contains("laguna") {
            Self::Laguna
        } else {
            Self::Unknown
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            Self::Gemma4Dense => "gemma4-dense-nvfp4",
            Self::Gemma4E4b => "gemma4-e4b-w4a16",
            Self::Gemma4Moe => "gemma4-moe-nvfp4",
            Self::Qwen36Moe => "qwen3.6-moe-nvfp4",
            Self::Qwen35Moe => "qwen3.5-moe",
            Self::Qwen35Dense => "qwen3.5-dense-nvfp4",
            Self::Qwen38Dense => "qwen3.8-dense-nvfp4",
            Self::Qwen38Max => "qwen3.8-max-2.4t-a95b",
            Self::GptOss => "gpt-oss-mxfp4",
            Self::Laguna => "laguna-nvfp4",
            Self::DiffusionGemma => "diffusion-gemma-26b-a4b",
            Self::Unknown => "unknown",
        }
    }

    pub fn required_kernels(self) -> Vec<KernelId> {
        let mut ks = KernelId::DENSE_DECODE_PATH.to_vec();
        match self {
            Self::Gemma4E4b => ks.push(KernelId::MarlinGemmW4a16),
            Self::Gemma4Moe | Self::Qwen36Moe | Self::Qwen35Moe => ks.extend([
                KernelId::MoePermute,
                KernelId::MoeUnpermuteScatter,
                KernelId::MoeGroupedGemmNvfp4,
            ]),
            Self::Gemma4Dense
            | Self::Qwen35Dense
            | Self::Qwen38Dense
            | Self::Qwen38Max
            | Self::GptOss
            | Self::Laguna
            | Self::DiffusionGemma
            | Self::Unknown => {}
        }
        ks
    }

    pub fn wgpu_required_kernels(self) -> Vec<KernelId> {
        let mut ks = KernelId::DENSE_DECODE_PATH.to_vec();
        if matches!(self, Self::Gemma4Moe | Self::Qwen36Moe | Self::Qwen35Moe) {
            ks.extend([KernelId::MoePermute, KernelId::MoeUnpermuteScatter]);
        }
        ks
    }

    pub fn wgpu_decoder(self) -> Option<WgpuDecoder> {
        match self {
            Self::Gemma4Dense => Some(WgpuDecoder {
                module: "nv_models::gemma4_wgpu",
                entry: "Gemma4Wgpu::new",
                evidence: WgpuEvidence::RealWeights {
                    checkpoint: "nvidia/Gemma-4-31B-IT-NVFP4",
                    test: "real_gemma4_31b_wgpu_decode_ms_per_token (NV_GEMMA4_WGPU_TEST=1)",
                },
            }),
            Self::Gemma4E4b => Some(WgpuDecoder {
                module: "nv_models::gemma4_e4b_wgpu",
                entry: "Gemma4E4bWgpu::new",
                evidence: WgpuEvidence::RealWeights {
                    checkpoint: "google/gemma-4-E4B-it",
                    test: "real_e4b_checkpoint_wgpu_decode (NV_E4B_WGPU_TEST=1)",
                },
            }),
            Self::Qwen36Moe => Some(WgpuDecoder {
                module: "nv_models::qwen3_5_moe_wgpu",
                entry: "Qwen3MoeWgpu::new",
                evidence: WgpuEvidence::RealWeights {
                    checkpoint: "RedHatAI/Qwen3.6-35B-A3B-NVFP4",
                    test: "qwen36_wgpu_real_weights_decode (NV_QWEN36_WGPU_TEST=1)",
                },
            }),
            Self::Qwen35Moe => Some(WgpuDecoder {
                module: "nv_models::qwen3_5_moe_wgpu",
                entry: "Qwen3MoeWgpu::new",
                evidence: WgpuEvidence::ArchitectureFamily {
                    verified_sibling: "RedHatAI/Qwen3.6-35B-A3B-NVFP4 (model_type qwen3_5_moe)",
                },
            }),
            Self::Qwen35Dense => Some(WgpuDecoder {
                module: "nv_models::qwen3_5_dense_wgpu",
                entry: "Qwen3_5DenseWgpu::from_loader",
                evidence: WgpuEvidence::RealWeights {
                    checkpoint: "ig1/Qwen3.5-9B-NVFP4",
                    test: "qwen35_dense_wgpu_real_weights_decode (NV_QWEN35_DENSE_WGPU_TEST=1)",
                },
            }),
            Self::Qwen38Dense => Some(WgpuDecoder {
                module: "nv_models::qwen3_5_dense_wgpu",
                entry: "Qwen3_5DenseWgpu::from_loader",
                evidence: WgpuEvidence::ArchitectureFamily {
                    verified_sibling: "ig1/Qwen3.5-9B-NVFP4 (same model_type qwen3_5 trunk; Qwen3.8 is a scale-up loading with the qwen3_5 Transformers classes, so the dense-hybrid decoder is the designed slot until a qwen3.8 real-weights decode is recorded and this row is flipped to RealWeights)",
                },
            }),
            Self::GptOss => Some(WgpuDecoder {
                module: "nv_models::gpt_oss_wgpu",
                entry: "GptOssWgpu::from_loader",
                evidence: WgpuEvidence::RealWeights {
                    checkpoint: "openai/gpt-oss-20b",
                    test: "gptoss_wgpu_real_weights_decode (NV_GPTOSS_WGPU_TEST=1)",
                },
            }),
            Self::Laguna => Some(WgpuDecoder {
                module: "nv_models::laguna_wgpu",
                entry: "LagunaWgpu::from_loader",
                evidence: WgpuEvidence::RealWeights {
                    checkpoint: "poolside/Laguna-XS-2.1-NVFP4",
                    test: "real_weight_greedy_decode_is_coherent_and_reproducible (NV_LAGUNA_DIR=<snapshot>); served over HTTP; the decode rate row lives in perf/runs.jsonl",
                },
            }),
            Self::Gemma4Moe => Some(WgpuDecoder {
                module: "nv_models::gemma4_moe_wgpu",
                entry: "Gemma4MoeWgpu::from_loader",
                evidence: WgpuEvidence::ArchitectureFamily {
                    verified_sibling: "chat_engine_wgpu::classify_wgpu_model routes it as WgpuModelKind::Gemma4Moe and builds it at chat_engine_wgpu.rs:1541; chunked prefill landed for this decoder in 10db591d5. No real-weights wgpu decode is recorded for google/gemma-4-26B-A4B-it",
                },
            }),
            Self::Qwen38Max | Self::DiffusionGemma | Self::Unknown => None,
        }
    }

    pub fn wgpu_absent_note(self) -> Option<&'static str> {
        match self {
            Self::DiffusionGemma => Some(
                "DiffusionGemma is not autoregressive. Its text_config is field-for-field \
                 identical to google/gemma-4-26B-A4B-it (hidden 2816, 30 layers, 16/8 heads, \
                 head_dim 256, 128 experts top-8, moe_intermediate 704, vocab 262144, \
                 sliding_window 1024), so the transformer would load and run -- and would \
                 emit garbage, because the model was trained to denoise a masked 256-token \
                 canvas (config canvas_length: 256, architectures \
                 DiffusionGemmaForBlockDiffusion, model_type diffusion_gemma) rather than to \
                 predict the next token. Refusing here is what stops it being misrouted to \
                 the gemma4 MoE decoder on the strength of an identical config. Serving it \
                 needs a block-diffusion decode loop, not a new config type",
            ),

            Self::Qwen38Max => Some(QWEN38_MAX_EXCEEDS_ANY_SINGLE_CARD),
            Self::Gemma4Moe => None,
            _ => None,
        }
    }
}

pub const WGPU_DECODER_CLASSES: &[ModelClass] = &[
    ModelClass::Gemma4Dense,
    ModelClass::Gemma4E4b,
    ModelClass::Qwen36Moe,
    ModelClass::Qwen35Moe,
    ModelClass::Qwen35Dense,
    ModelClass::Qwen38Dense,
    ModelClass::Gemma4Moe,
    ModelClass::GptOss,
    ModelClass::Laguna,
];

fn wgpu_decoder_class_list() -> String {
    WGPU_DECODER_CLASSES
        .iter()
        .map(|c| c.label())
        .collect::<Vec<_>>()
        .join(", ")
}

pub const KNOWN_MODELS: &[(&str, ModelClass)] = &[
    ("nvidia/Gemma-4-31B-IT-NVFP4", ModelClass::Gemma4Dense),
    ("google/gemma-4-E4B-it", ModelClass::Gemma4E4b),
    ("google/gemma-4-26B-A4B-it", ModelClass::Gemma4Moe),
    ("RedHatAI/Qwen3.6-35B-A3B-NVFP4", ModelClass::Qwen36Moe),
    ("Qwen/Qwen3.5-MoE", ModelClass::Qwen35Moe),
    ("ig1/Qwen3.5-9B-NVFP4", ModelClass::Qwen35Dense),
    ("Qwen/Qwen3.8-27B", ModelClass::Qwen38Dense),
    ("Qwen/Qwen3.8-2.4T-A95B", ModelClass::Qwen38Max),
    ("openai/gpt-oss-20b", ModelClass::GptOss),
    ("poolside/Laguna-XS-2.1-NVFP4", ModelClass::Laguna),
    (
        "google/diffusiongemma-26B-A4B-it",
        ModelClass::DiffusionGemma,
    ),
    (
        "nvidia/diffusiongemma-26B-A4B-it-NVFP4",
        ModelClass::DiffusionGemma,
    ),
];

#[derive(Debug)]
pub enum BackendSelectError {
    InvalidSelection(String),
    Unavailable {
        backend: &'static str,
        reason: String,
    },
    ModelUnservable {
        backend: &'static str,
        model: String,
        reason: String,
    },
    NoBackend {
        model: String,
        reasons: Vec<String>,
    },
}

impl fmt::Display for BackendSelectError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidSelection(m) => write!(f, "backend selection: {m}"),
            Self::Unavailable { backend, reason } => {
                write!(f, "{backend} backend unavailable: {reason}")
            }
            Self::ModelUnservable {
                backend,
                model,
                reason,
            } => write!(
                f,
                "model {model:?} cannot be served on the {backend} backend: {reason}; no silent fallback is attempted"
            ),
            Self::NoBackend { model, reasons } => write!(
                f,
                "no backend can serve model {model:?}: {}",
                reasons.join("; ")
            ),
        }
    }
}

impl std::error::Error for BackendSelectError {}

impl BackendSelectError {
    pub fn status(&self) -> StatusCode {
        match self {
            Self::InvalidSelection(_) | Self::ModelUnservable { .. } => StatusCode::BAD_REQUEST,
            Self::Unavailable { .. } | Self::NoBackend { .. } => StatusCode::SERVICE_UNAVAILABLE,
        }
    }

    pub fn code(&self) -> &'static str {
        match self {
            Self::InvalidSelection(_) => "backend_selection_invalid",
            Self::Unavailable { .. } => "backend_unavailable",
            Self::ModelUnservable { .. } => "backend_unsupported_model",
            Self::NoBackend { .. } => "no_backend_for_model",
        }
    }

    pub fn into_response(self) -> Response {
        let kind = match self.status() {
            StatusCode::BAD_REQUEST => super::kind::INVALID_REQUEST,
            _ => super::kind::SERVICE_UNAVAIL,
        };
        super::openai_error(
            self.status(),
            self.to_string(),
            kind,
            Some("backend"),
            Some(self.code()),
        )
    }
}

pub const QWEN35_DENSE_NO_CUDA: &str = "qwen3.5-dense has no cuda serving path, and the failure cuda produces today accuses the wrong party. chat_engine::detect_family's starts_with(\"qwen3\") catch-all matches both the architectures entry Qwen3_5ForConditionalGeneration and the top-level model_type qwen3_5, so it answers ModelFamily::Qwen3 and the loader hands a qwen3.5 config to nv_models::qwen3::Qwen3Config, whose serde struct wants hidden_size, num_hidden_layers and the rest at the TOP level while qwen3.5 nests every one of them under text_config. Left unguarded, the load dies with \"deserialize qwen3 config: missing field `hidden_size`\", which reads as a malformed config.json. The config.json is not malformed: the same bytes parse cleanly into nv_models::qwen3_5_moe::Qwen3_5DenseConfig, which is what the wgpu decoder loads. Cuda has no loader for this family, and closing that gap needs two things rather than one -- a qwen3.5-dense arm in detect_family, and a cuda dense-hybrid decoder, because qwen3.rs has no linear-attention mixer for the linear_attention entries of layer_types (24 of 32 layers in Qwen/Qwen3.5-9B's config.json; LayerMixer exists only in nv_models::qwen3_5_moe). Serve this checkpoint on the wgpu backend (NV_SERVE_BACKEND=wgpu), where nv_models::qwen3_5_dense_wgpu is real-weights verified";

pub const QWEN38_DENSE_NO_CUDA: &str = "qwen3.8-dense (the 27B multimodal checkpoint) has no default cuda serving path, for exactly the qwen3.5-dense reasons: Qwen3.8 is a scale-up of the Qwen3.5 architecture that loads with the same qwen3_5 Transformers classes -- its config declares architectures Qwen3_5ForConditionalGeneration, model_type qwen3_5, and nests hidden_size and every other trunk field under text_config, so chat_engine::detect_family's starts_with(\"qwen3\") catch-all hands the config to nv_models::qwen3::Qwen3Config and the load dies with \"deserialize qwen3 config: missing field `hidden_size`\" even though the same bytes parse cleanly into nv_models::qwen3_5_moe::Qwen3_5DenseConfig; and qwen3.rs has no linear-attention mixer for the 3-of-4 linear_attention entries that full_attention_interval=4 puts in layer_types. Serve this checkpoint on the wgpu backend (NV_SERVE_BACKEND=wgpu), where nv_models::qwen3_5_dense_wgpu parses the same model_type qwen3_5 trunk and now loads the published unsloth/Qwen3.8-27B-NVFP4 mixed-precision weights: its DeltaNet linear_attn in_proj_qkv/in_proj_z/out_proj projections and lm_head ship as F8_E4M3, which load_bf16's load_fp8_e4m3_as_bf16 path dequantizes to bf16 at load (real-weights boot+decode verified). Cuda still has no path because qwen3.rs has no linear-attention mixer for the 3-of-4 linear_attention entries that full_attention_interval=4 puts in layer_types";

pub const QWEN38_MAX_EXCEEDS_ANY_SINGLE_CARD: &str = "the qwen3.8 flagship (Qwen/Qwen3.8-2.4T-A95B: 2.4T total parameters, 95B active, 512 experts) cannot fit one 96 GiB card at any quantization: 2.4T weights are ~1.3 TB at 4-bit before KV cache or activations, over an order of magnitude past the card. This is a capacity refusal, not a decoder gap -- the checkpoint declares the same qwen3_5_moe layout the served qwen3.5/3.6 MoE paths already decode. Serving it needs multi-GPU expert/tensor parallelism this server does not implement, on either backend";

pub const QWEN38_DENSE_CUDA_SERVE_ENV_IS_THE_QWEN35_ONE: &str = "NV_QWEN35_DENSE_CUDA_SERVE";

pub const GPTOSS_CUDA_SERVE_ENV: &str = "NV_GPTOSS_CUDA_SERVE";

pub fn gpt_oss_cuda_serve_enabled() -> bool {
    matches!(
        std::env::var(GPTOSS_CUDA_SERVE_ENV).ok().as_deref(),
        Some("1") | Some("on")
    )
}

pub const GPTOSS_NO_CUDA_WITHOUT_THE_OPT_IN: &str = "gpt-oss cuda serving is wired but gated off by default: with NV_GPTOSS_CUDA_SERVE=1 chat_engine::detect_family answers ModelFamily::GptOss and try_load builds nv_models::gpt_oss_cuda::GptOssCuda, which dequantizes every mxfp4 expert tensor to bf16 at load (nv_quant::mxfp4::Mxfp4Tensor::dequantize, the same host semantics the wgpu mxfp4 GEMV is pinned against) and runs attention through nv_layers::attn::sdpa_with_sinks, the eager path that folds gpt-oss's learned per-head sink logit into the softmax max and denominator by appending one sink score column and one all-zero value row. the cuda flash-attention kernel takes no sink argument, so this decoder cannot take the flash branch, and the alternating sliding_window=128 layers are masked in the same builder. Without the variable the load refuses here and nothing changes. It stays opt-in for two reasons that are both about cost rather than correctness: dequant-to-bf16 makes openai/gpt-oss-20b about 41.8 GB resident against about 13.7 GB for a native mxfp4 path, roughly 28 GB more on a 96 GiB card that must also hold KV; and eager scoring costs heads*rows*context f32 of transient scratch per prefill chunk where the wgpu decoder's gow_attn.wgsl and gow_prefill.wgsl fold the sink inside a streaming softmax. wgpu remains the default backend for this checkpoint class, and nv_models::gpt_oss_wgpu remains the only gpt-oss path with an mxfp4 GEMV. Serve this checkpoint on the wgpu backend, or set NV_GPTOSS_CUDA_SERVE=1";

pub fn cuda_model_unservable_reason(class: ModelClass) -> Option<String> {
    match class {

        ModelClass::Gemma4Moe => None,
        ModelClass::Gemma4E4b => {

            if std::env::var("NV_E4B_CUDA_SERVE").ok().as_deref() == Some("1") {
                None
            } else {
                Some(
                    "gemma-4 E4B cuda serving is wired but gated off by default: chat_engine \
                     routes the checkpoint to nv_models::gemma4_e4b::Gemma4E4b (per-layer \
                     embedding stack, w4a16 decode standalone-measured in perf/runs.jsonl) only when \
                     NV_E4B_CUDA_SERVE=1 is set; without it try_load refuses so the default \
                     behavior is unchanged. Serve E4B on the wgpu backend, or set \
                     NV_E4B_CUDA_SERVE=1"
                        .to_string(),
                )
            }
        }
        ModelClass::GptOss => {
            if gpt_oss_cuda_serve_enabled() {
                None
            } else {
                Some(GPTOSS_NO_CUDA_WITHOUT_THE_OPT_IN.to_string())
            }
        }
        ModelClass::Qwen35Dense => Some(QWEN35_DENSE_NO_CUDA.to_string()),
        ModelClass::Qwen38Dense => {
            if std::env::var(QWEN38_DENSE_CUDA_SERVE_ENV_IS_THE_QWEN35_ONE)
                .ok()
                .as_deref()
                == Some("1")
            {
                None
            } else {
                Some(QWEN38_DENSE_NO_CUDA.to_string())
            }
        }
        ModelClass::Qwen38Max => Some(QWEN38_MAX_EXCEEDS_ANY_SINGLE_CARD.to_string()),
        ModelClass::DiffusionGemma => Some(
            "DiffusionGemma cannot be served on cuda: it is not autoregressive (trained to denoise a masked token canvas, not to predict the next token) and chat_engine::detect_family has no arm for it, so the load fails at family detection. Serving it needs a block-diffusion decode loop on either backend"
                .to_string(),
        ),
        _ => None,
    }
}

pub const CUDA_FEATURE_OFF_REASON: &str = "server binary compiled without the cuda feature";

pub fn cuda_unservable_reason(class: ModelClass) -> Option<String> {
    if let Some(reason) = cuda_model_unservable_reason(class) {
        return Some(reason);
    }
    if !cfg!(feature = "cuda") {
        return Some(CUDA_FEATURE_OFF_REASON.to_string());
    }
    None
}

pub fn wgpu_missing_kernels(class: ModelClass) -> Vec<KernelId> {
    missing_on(BackendKind::Wgpu, &class.wgpu_required_kernels())
}

pub fn cuda_only_fast_paths(class: ModelClass) -> Vec<KernelId> {
    missing_on(BackendKind::Wgpu, &class.required_kernels())
}

pub fn wgpu_model_support(class: ModelClass) -> Result<WgpuDecoder, String> {
    if class == ModelClass::Unknown {
        return Err(
            "unrecognized model id: wgpu serving requires a known architecture with a native wgpu decoder in nv-models"
                .to_string(),
        );
    }
    let decoder = match class.wgpu_decoder() {
        Some(d) => d,
        None => {
            let mut msg = format!(
                "no native wgpu decoder exists for {}: nv-models ships wgpu decoders only for {}",
                class.label(),
                wgpu_decoder_class_list()
            );
            if let Some(note) = class.wgpu_absent_note() {
                msg.push_str("; ");
                msg.push_str(note);
            }
            return Err(msg);
        }
    };
    let missing = wgpu_missing_kernels(class);
    if !missing.is_empty() {
        let names: Vec<&str> = missing.iter().map(|k| k.name()).collect();
        return Err(format!(
            "{} has a wgpu decoder ({}) but these kernels it needs have no wgpu implementation: {}",
            class.label(),
            decoder.module,
            names.join(", ")
        ));
    }
    Ok(decoder)
}

pub fn wgpu_unservable_reason_for_build(
    class: ModelClass,
    decoders_compiled_in: bool,
) -> Option<String> {
    if !decoders_compiled_in {
        return Some(WGPU_FEATURE_OFF_REASON.to_string());
    }
    if let Err(reason) = wgpu_model_support(class) {
        return Some(reason);
    }
    None
}

pub fn wgpu_unservable_reason(class: ModelClass) -> Option<String> {
    wgpu_unservable_reason_for_build(class, WGPU_DECODERS_COMPILED_IN)
}

pub fn resolve_with_build(
    sel: ServeBackend,
    model_id: &str,
    probe_cuda_fn: &dyn Fn() -> Result<(), String>,
    probe_wgpu_fn: &dyn Fn() -> Result<(), String>,
    decoders_compiled_in: bool,
) -> Result<BackendKind, BackendSelectError> {
    let class = ModelClass::classify(model_id);
    match sel {
        ServeBackend::Cuda => {
            probe_cuda_fn().map_err(|reason| BackendSelectError::Unavailable {
                backend: "cuda",
                reason,
            })?;
            match cuda_model_unservable_reason(class) {
                Some(reason) => Err(BackendSelectError::ModelUnservable {
                    backend: "cuda",
                    model: model_id.to_string(),
                    reason,
                }),
                None => Ok(BackendKind::Cuda),
            }
        }
        ServeBackend::Wgpu => {
            probe_wgpu_fn().map_err(|reason| BackendSelectError::Unavailable {
                backend: "wgpu",
                reason,
            })?;
            match wgpu_unservable_reason_for_build(class, decoders_compiled_in) {
                Some(reason) => Err(BackendSelectError::ModelUnservable {
                    backend: "wgpu",
                    model: model_id.to_string(),
                    reason,
                }),
                None => Ok(BackendKind::Wgpu),
            }
        }
        ServeBackend::Auto => {
            let mut reasons = Vec::new();
            match probe_cuda_fn() {
                Ok(()) => match cuda_model_unservable_reason(class) {
                    None => return Ok(BackendKind::Cuda),
                    Some(reason) => reasons.push(format!("cuda: {reason}")),
                },
                Err(reason) => reasons.push(format!("cuda: {reason}")),
            }
            match resolve_with_build(
                ServeBackend::Wgpu,
                model_id,
                probe_cuda_fn,
                probe_wgpu_fn,
                decoders_compiled_in,
            ) {
                Ok(kind) => return Ok(kind),
                Err(e) => reasons.push(format!("wgpu: {e}")),
            }
            Err(BackendSelectError::NoBackend {
                model: model_id.to_string(),
                reasons,
            })
        }
    }
}

pub fn resolve_with(
    sel: ServeBackend,
    model_id: &str,
    probe_cuda_fn: &dyn Fn() -> Result<(), String>,
    probe_wgpu_fn: &dyn Fn() -> Result<(), String>,
) -> Result<BackendKind, BackendSelectError> {
    resolve_with_build(
        sel,
        model_id,
        probe_cuda_fn,
        probe_wgpu_fn,
        WGPU_DECODERS_COMPILED_IN,
    )
}

pub fn resolve(sel: ServeBackend, model_id: &str) -> Result<BackendKind, BackendSelectError> {
    resolve_with(sel, model_id, &probe_cuda, &probe_wgpu)
}

pub fn resolve_from_env(model_id: &str) -> Result<BackendKind, BackendSelectError> {
    resolve(ServeBackend::from_env()?, model_id)
}

fn decoder_entry(d: WgpuDecoder) -> Value {
    json!({
        "module": d.module,
        "entry": d.entry,
        "evidence": d.evidence.kind(),
        "evidence_detail": d.evidence.detail(),
    })
}

fn model_entry(class: ModelClass) -> Value {
    let cuda_reason = cuda_unservable_reason(class);
    let wgpu_reason = wgpu_unservable_reason(class);
    let wgpu_missing: Vec<&str> = wgpu_missing_kernels(class)
        .iter()
        .map(|k| k.name())
        .collect();
    let fast_paths: Vec<&str> = cuda_only_fast_paths(class)
        .iter()
        .map(|k| k.name())
        .collect();
    let decoder = match class.wgpu_decoder() {
        Some(d) => decoder_entry(d),
        None => Value::Null,
    };
    json!({
        "class": class.label(),
        "cuda": {
            "servable": cuda_reason.is_none(),
            "reason": cuda_reason,
        },
        "wgpu": {
            "servable": wgpu_reason.is_none(),
            "reason": wgpu_reason,
            "decoder": decoder,
            "missing_kernels": wgpu_missing,
            "cuda_only_fast_paths_replaced_by_wgsl": fast_paths,
        },
    })
}

pub fn backends_report() -> Value {
    let (requested, selection_error) = match ServeBackend::from_env() {
        Ok(sel) => (Value::String(sel.name().to_string()), Value::Null),
        Err(e) => (Value::Null, Value::String(e.to_string())),
    };
    let available: Vec<Value> = nv_layers::backend::availability()
        .into_iter()
        .map(|(kind, res)| {
            json!({
                "name": kind.name(),
                "available": res.is_ok(),
                "reason": res.err(),
            })
        })
        .collect();
    let mut models = serde_json::Map::new();
    for (id, class) in KNOWN_MODELS {
        models.insert((*id).to_string(), model_entry(*class));
    }
    json!({
        "selection_env": defaults::env::NV_SERVE_BACKEND,
        "default": defaults::serve_backend::DEFAULT,
        "requested": requested,
        "selection_error": selection_error,
        "available": available,
        "auto_policy": AUTO_POLICY,
        "wgpu_decoders_compiled_in": WGPU_DECODERS_COMPILED_IN,
        "models": Value::Object(models),
    })
}

pub fn augment_capabilities(mut base: Value) -> Value {
    if let Some(obj) = base.as_object_mut() {
        obj.insert("backends".to_string(), backends_report());
    }
    base
}

pub async fn realtime_capabilities_with_backends(State(state): State<AppState>) -> Response {
    let body = augment_capabilities(crate::realtime::capabilities_json_with_models(
        &state.models,
    ));
    (
        StatusCode::OK,
        [(header::CONTENT_TYPE, "application/json")],
        body.to_string(),
    )
        .into_response()
}

pub async fn handle_backends_report() -> Response {
    (
        StatusCode::OK,
        [(header::CONTENT_TYPE, "application/json")],
        backends_report().to_string(),
    )
        .into_response()
}

pub const BACKENDS_REPORT_ROUTE: &str = "/v1/backends";

pub const REALTIME_CAPABILITIES_WITH_BACKENDS_ROUTE: &str = "/v1/realtime/capabilities/backends";

pub fn backends_report_router<S>() -> Router<S>
where
    S: Clone + Send + Sync + 'static,
{
    Router::new().route(BACKENDS_REPORT_ROUTE, get(handle_backends_report))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn qwen35_dense_config_json() -> String {
        let layer_types: Vec<&str> = (0..32)
            .map(|i| {
                if (i + 1) % 4 == 0 {
                    "\"full_attention\""
                } else {
                    "\"linear_attention\""
                }
            })
            .collect();
        format!(
            r#"{{
  "architectures": ["Qwen3_5ForConditionalGeneration"],
  "model_type": "qwen3_5",
  "tie_word_embeddings": false,
  "image_token_id": 151655,
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
    "full_attention_interval": 4,
    "linear_num_key_heads": 16,
    "linear_num_value_heads": 32,
    "linear_key_head_dim": 128,
    "linear_value_head_dim": 128,
    "linear_conv_kernel_dim": 4,
    "eos_token_id": 151645,
    "rope_parameters": {{"rope_theta": 10000000.0, "partial_rotary_factor": 0.25}},
    "layer_types": [{}]
  }}
}}"#,
            layer_types.join(", ")
        )
    }

    fn qwen38_27b_config_json_pinned_from_release_facts_not_a_fetched_file() -> String {
        let layer_types: Vec<&str> = (0..64)
            .map(|i| {
                if (i + 1) % 4 == 0 {
                    "\"full_attention\""
                } else {
                    "\"linear_attention\""
                }
            })
            .collect();
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
    "num_attention_heads": 24,
    "num_key_value_heads": 4,
    "head_dim": 256,
    "intermediate_size": 17408,
    "vocab_size": 248320,
    "max_position_embeddings": 262144,
    "rms_norm_eps": 1e-06,
    "attn_output_gate": true,
    "output_gate_type": "swish",
    "mamba_ssm_dtype": "float32",
    "full_attention_interval": 4,
    "linear_num_key_heads": 16,
    "linear_num_value_heads": 32,
    "linear_key_head_dim": 128,
    "linear_value_head_dim": 128,
    "linear_conv_kernel_dim": 4,
    "eos_token_id": 248055,
    "rope_parameters": {{"rope_type": "default", "rope_theta": 10000000.0, "partial_rotary_factor": 0.25}},
    "layer_types": [{}]
  }},
  "vision_config": {{
    "hidden_size": 1152,
    "num_hidden_layers": 27,
    "patch_size": 16,
    "temporal_patch_size": 2
  }}
}}"#,
            layer_types.join(", ")
        )
    }

    #[test]
    fn qwen38_ids_classify_dense_and_flagship_ids_classify_max() {
        for id in [
            "Qwen/Qwen3.8-27B",
            "qwen/qwen3-8-27b",
            "models--Qwen--Qwen3.8-27B",
        ] {
            assert_eq!(ModelClass::classify(id), ModelClass::Qwen38Dense, "{id}");
        }
        for id in [
            "Qwen/Qwen3.8-2.4T-A95B",
            "qwen3-8-2.4t-a95b",
            "Qwen/Qwen3.8-A95B-Instruct",
        ] {
            assert_eq!(
                ModelClass::classify(id),
                ModelClass::Qwen38Max,
                "a95b must reach the qwen3.8 arm, not the a3b substring arm: {id}"
            );
        }
        assert_eq!(
            ModelClass::classify("RedHatAI/Qwen3.6-35B-A3B-NVFP4"),
            ModelClass::Qwen36Moe
        );
        assert_eq!(
            ModelClass::classify("ig1/Qwen3.5-9B-NVFP4"),
            ModelClass::Qwen35Dense
        );
    }

    #[test]
    fn the_qwen38_flagship_is_refused_on_both_backends_for_capacity() {
        let reason = cuda_model_unservable_reason(ModelClass::Qwen38Max)
            .expect("the 2.4T flagship must carry a capacity refusal");
        assert_eq!(reason, QWEN38_MAX_EXCEEDS_ANY_SINGLE_CARD);
        for needle in ["2.4T", "96 GiB", "~1.3 TB", "either backend"] {
            assert!(
                reason.contains(needle),
                "the capacity refusal must state {needle:?}: {reason}"
            );
        }
        assert_eq!(ModelClass::Qwen38Max.wgpu_decoder(), None);
        let wgpu_err = wgpu_model_support(ModelClass::Qwen38Max)
            .expect_err("no wgpu decoder row may exist for a checkpoint no single card can hold");
        assert!(wgpu_err.contains(QWEN38_MAX_EXCEEDS_ANY_SINGLE_CARD), "{wgpu_err}");
        let err = resolve_with_build(
            ServeBackend::Auto,
            "Qwen/Qwen3.8-2.4T-A95B",
            &|| Ok(()),
            &|| Ok(()),
            true,
        )
        .expect_err("auto must refuse the flagship on both backends, never pick one silently");
        assert!(matches!(err, BackendSelectError::NoBackend { .. }), "{err}");
        let msg = err.to_string();
        assert!(msg.contains("96 GiB"), "{msg}");
        assert_eq!(err.status(), StatusCode::SERVICE_UNAVAILABLE);
        let entry = model_entry(ModelClass::Qwen38Max);
        assert_eq!(entry["cuda"]["servable"], false);
        assert_eq!(entry["wgpu"]["servable"], false);
        assert_eq!(entry["wgpu"]["decoder"], Value::Null);
    }

    #[test]
    fn a_qwen38_dense_checkpoint_is_refused_on_cuda_and_pointed_at_wgpu_unless_opted_in() {
        if std::env::var(QWEN38_DENSE_CUDA_SERVE_ENV_IS_THE_QWEN35_ONE)
            .ok()
            .as_deref()
            == Some("1")
        {
            eprintln!(
                "[qwen38-dense] skip: NV_QWEN35_DENSE_CUDA_SERVE=1 is set, the default-refusal \
                 branch under test is opted out in this environment"
            );
            return;
        }
        let reason = cuda_model_unservable_reason(ModelClass::Qwen38Dense)
            .expect("qwen3.8-dense must carry the qwen3.5-shaped cuda refusal by default");
        assert_eq!(reason, QWEN38_DENSE_NO_CUDA);
        for needle in [
            "qwen3_5",
            "text_config",
            "missing field `hidden_size`",
            "nv_models::qwen3_5_dense_wgpu",
            "NV_SERVE_BACKEND=wgpu",
            "F8_E4M3",
        ] {
            assert!(
                reason.contains(needle),
                "the cuda refusal must state {needle:?}: {reason}"
            );
        }

        assert_eq!(
            ModelClass::Qwen38Dense.required_kernels(),
            ModelClass::Qwen35Dense.required_kernels()
        );
        assert_eq!(
            ModelClass::Qwen38Dense.wgpu_required_kernels(),
            ModelClass::Qwen35Dense.wgpu_required_kernels()
        );
        let d = wgpu_model_support(ModelClass::Qwen38Dense)
            .expect("the qwen3.5 dense wgpu decoder is the designed slot for qwen3.8-dense");
        assert_eq!(d.module, "nv_models::qwen3_5_dense_wgpu");
        assert_eq!(d.entry, "Qwen3_5DenseWgpu::from_loader");
        assert_eq!(d.evidence.kind(), "architecture-family-only");
        assert!(
            d.evidence.detail().contains("ig1/Qwen3.5-9B-NVFP4"),
            "the evidence row must name the real-weights-verified sibling: {}",
            d.evidence.detail()
        );

        let err = resolve_with_build(
            ServeBackend::Cuda,
            "Qwen/Qwen3.8-27B",
            &|| Ok(()),
            &|| panic!("wgpu must not be probed for an explicit cuda selection"),
            true,
        )
        .expect_err("an explicit cuda selection must refuse a qwen3.8-dense checkpoint");
        assert!(
            matches!(
                err,
                BackendSelectError::ModelUnservable {
                    backend: "cuda",
                    ..
                }
            ),
            "{err}"
        );
        assert_eq!(err.status(), StatusCode::BAD_REQUEST);
        let got = resolve_with_build(
            ServeBackend::Auto,
            "Qwen/Qwen3.8-27B",
            &|| Ok(()),
            &|| Ok(()),
            true,
        )
        .expect("auto must fall through to the wgpu decoder, not return cuda");
        assert_eq!(got, BackendKind::Wgpu);

        let entry = model_entry(ModelClass::Qwen38Dense);
        assert_eq!(entry["cuda"]["servable"], false);
        assert_eq!(entry["cuda"]["reason"], QWEN38_DENSE_NO_CUDA);
        assert_eq!(
            entry["wgpu"]["decoder"]["module"],
            "nv_models::qwen3_5_dense_wgpu"
        );
        assert_eq!(
            entry["wgpu"]["decoder"]["evidence"],
            "architecture-family-only"
        );
    }

    #[test]
    fn a_pinned_qwen38_27b_config_parses_for_the_wgpu_dense_decoder_architectural_verification_only(
    ) {
        let cfg = qwen38_27b_config_json_pinned_from_release_facts_not_a_fetched_file();
        let dense = nv_models::qwen3_5_moe::Qwen3_5DenseConfig::from_hf_json_str(&cfg).expect(
            "release facts say qwen3.8 adds no config keys over qwen3.5, so the qwen3.5 dense \
             parser must accept the pinned qwen3.8-27b shape",
        );
        assert_eq!(dense.num_hidden_layers, 64);
        assert_eq!(dense.head_dim, 256);
        assert_eq!(dense.vocab_size, 248320);
        assert!(dense.attn_output_gate);
        assert!((dense.rope_theta - 1.0e7).abs() < 1.0);
        assert!((dense.partial_rotary_factor - 0.25).abs() < 1e-6);
        assert_eq!(
            dense
                .layer_types
                .iter()
                .filter(|t| **t == nv_models::qwen3_5_moe::LayerType::FullAttention)
                .count(),
            16,
            "full_attention_interval=4 over 64 layers is 16 gated-attention layers"
        );
        let err = nv_models::qwen3::Qwen3Config::from_hf_json_str(&cfg).expect_err(
            "the cuda qwen3 parser must still choke on the text_config nesting; if it now \
             parses, re-derive QWEN38_DENSE_NO_CUDA against whatever the load does instead",
        );
        assert!(format!("{err:#}").contains("missing field"), "{err:#}");
    }

    #[test]
    fn a_qwen35_dense_checkpoint_is_refused_on_cuda_and_pointed_at_wgpu() {
        for id in [
            "ig1/Qwen3.5-9B-NVFP4",
            "Qwen/Qwen3.5-9B",
            "Qwen/Qwen3-5-9B",
            "models--Qwen--Qwen3.5-9B",
        ] {
            assert_eq!(ModelClass::classify(id), ModelClass::Qwen35Dense, "{id}");
        }

        let reason = cuda_model_unservable_reason(ModelClass::Qwen35Dense)
            .expect("qwen3.5-dense must carry a documented cuda refusal");
        assert_eq!(reason, QWEN35_DENSE_NO_CUDA);
        for needle in [
            "detect_family",
            "text_config",
            "missing field `hidden_size`",
            "is not malformed",
            "nv_models::qwen3_5_dense_wgpu",
            "NV_SERVE_BACKEND=wgpu",
        ] {
            assert!(
                reason.contains(needle),
                "the cuda refusal must state {needle:?}, else the operator is left with a \
                 checkpoint-blaming serde error: {reason}"
            );
        }

        let err = resolve_with_build(
            ServeBackend::Cuda,
            "ig1/Qwen3.5-9B-NVFP4",
            &|| Ok(()),
            &|| panic!("wgpu must not be probed for an explicit cuda selection"),
            true,
        )
        .expect_err("an explicit cuda selection must refuse a qwen3.5-dense checkpoint");
        assert!(
            matches!(
                err,
                BackendSelectError::ModelUnservable {
                    backend: "cuda",
                    ..
                }
            ),
            "{err}"
        );
        let msg = err.to_string();
        assert!(msg.contains(QWEN35_DENSE_NO_CUDA), "{msg}");
        assert!(msg.contains("ig1/Qwen3.5-9B-NVFP4"), "{msg}");
        assert_eq!(err.status(), StatusCode::BAD_REQUEST);
        assert_eq!(err.code(), "backend_unsupported_model");

        let got = resolve_with_build(
            ServeBackend::Auto,
            "ig1/Qwen3.5-9B-NVFP4",
            &|| Ok(()),
            &|| Ok(()),
            true,
        )
        .expect("auto must fall through to the wgpu decoder, not return cuda");
        assert_eq!(got, BackendKind::Wgpu);

        let entry = model_entry(ModelClass::Qwen35Dense);
        assert_eq!(entry["cuda"]["servable"], false);
        assert_eq!(entry["cuda"]["reason"], QWEN35_DENSE_NO_CUDA);
        assert_eq!(
            entry["wgpu"]["decoder"]["module"],
            "nv_models::qwen3_5_dense_wgpu"
        );
        assert_eq!(entry["wgpu"]["decoder"]["evidence"], "real-weights-decode");
    }

    #[test]
    fn a_models_own_refusal_outranks_the_builds() {
        let dense = cuda_unservable_reason(ModelClass::Qwen35Dense)
            .expect("qwen3.5-dense is unservable on cuda in every build");
        assert_eq!(
            dense, QWEN35_DENSE_NO_CUDA,
            "the models map documents CHECKPOINTS, so a checkpoint cuda cannot serve must say \
             why it cannot -- collapsing every entry to the build's own capability throws away \
             the per-model detail on exactly the builds that need it most, and whether this \
             binary has cuda is already reported by the top-level available field. \
             wgpu_unservable_reason_for_build already resolves it this way round"
        );
        let servable = cuda_unservable_reason(ModelClass::Gemma4Moe);
        if cfg!(feature = "cuda") {
            assert_eq!(servable, None, "gemma4-moe is servable on a cuda build");
        } else {
            assert_eq!(
                servable.as_deref(),
                Some(CUDA_FEATURE_OFF_REASON),
                "a model with no refusal of its own still falls back to the build's"
            );
        }
    }

    #[test]
    fn the_cuda_loader_rejects_the_very_config_the_wgpu_decoder_accepts() {
        let cfg = qwen35_dense_config_json();

        let err = nv_models::qwen3::Qwen3Config::from_hf_json_str(&cfg).expect_err(
            "nv_models::qwen3::Qwen3Config must still reject a qwen3.5 config; if it now parses, \
             re-derive QWEN35_DENSE_NO_CUDA against whatever the load does instead",
        );
        let rendered = format!("{err:#}");
        eprintln!("[qwen35-dense] cuda Qwen3Config error: {rendered}");
        let quoted = rendered
            .split("missing field ")
            .nth(1)
            .unwrap_or_else(|| panic!("expected a serde `missing field` error, got: {rendered}"))
            .split_whitespace()
            .next()
            .unwrap_or_default()
            .to_string();
        assert!(
            QWEN35_DENSE_NO_CUDA.contains(&format!("missing field {quoted}")),
            "the refusal quotes an error the loader no longer produces: loader says \
             {rendered:?}, refusal quotes something else"
        );

        let field = quoted.trim_matches('`');
        let v: serde_json::Value = serde_json::from_str(&cfg).unwrap();
        assert!(
            v.get(field).is_none(),
            "{field} is present at the top level, so the refusal's nesting story is wrong"
        );
        assert!(
            v["text_config"].get(field).is_some(),
            "{field} is not one of the fields qwen3.5 nests under text_config"
        );

        let dense = nv_models::qwen3_5_moe::Qwen3_5DenseConfig::from_hf_json_str(&cfg)
            .expect("the same bytes must parse for the wgpu dense decoder");
        assert_eq!(dense.num_hidden_layers, 32);
        assert_eq!(dense.layer_types.len(), 32);
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn no_cuda_family_can_load_a_qwen35_dense_config() {
        use crate::oapi::chat_engine::{detect_family_with_dense_cuda_serve, ModelFamily};
        let cfg = qwen35_dense_config_json();
        match detect_family_with_dense_cuda_serve(&cfg, false) {
            Err(e) => {
                let msg = format!("{e:#}");
                eprintln!("[qwen35-dense] detect_family refuses: {msg}");
                assert!(
                    msg.contains("wgpu"),
                    "detect_family now refuses qwen3.5-dense, which is the right answer, but the \
                     refusal must also tell the operator where the checkpoint CAN be served -- \
                     bail with backend_select::QWEN35_DENSE_NO_CUDA rather than a bare \
                     family-not-detected message: {msg}"
                );
            }
            Ok(ModelFamily::Qwen3) => {
                eprintln!("[qwen35-dense] detect_family -> Qwen3 (the misroute the refusal names)");
                nv_models::qwen3::Qwen3Config::from_hf_json_str(&cfg).expect_err(
                    "detect_family routes qwen3.5-dense to ModelFamily::Qwen3 AND the dense \
                     config now parses, which would mean cuda can serve it -- delete the \
                     Qwen35Dense arm of cuda_model_unservable_reason rather than relaxing this",
                );
            }
            Ok(other) => panic!(
                "detect_family now routes qwen3.5-dense to {other:?}. If cuda gained a \
                 dense-hybrid decoder, drop the Qwen35Dense arm of cuda_model_unservable_reason; \
                 if it did not, this is a new misroute. Do not weaken this assert."
            ),
        }
    }
}
