use std::path::Path;

use anyhow::{anyhow, Context, Result};
use ort::ep::ExecutionProviderDispatch;
use ort::session::{builder::GraphOptimizationLevel, Session};
use tracing::{info, warn};

pub const EP_CUDA: &str = "CUDAExecutionProvider";
pub const EP_ROCM: &str = "ROCmExecutionProvider";
pub const EP_MIGRAPHX: &str = "MIGraphXExecutionProvider";
pub const EP_COREML: &str = "CoreMLExecutionProvider";
pub const EP_DIRECTML: &str = "DmlExecutionProvider";
pub const EP_CPU: &str = "CPUExecutionProvider";

const ENV_EP: &str = "DIAR_EP";
const ENV_DEVICE_ID: &str = "DIAR_DEVICE_ID";
const ENV_GPU_MEM_LIMIT_MB: &str = "DIAR_GPU_MEM_LIMIT_MB";
const ENV_INTRA_THREADS: &str = "DIAR_INTRA_THREADS";
const ENV_CONV_SEARCH: &str = "DIAR_CUDA_CONV_SEARCH";
const ENV_PREFER_NHWC: &str = "DIAR_CUDA_PREFER_NHWC";
const ENV_ARENA_STRATEGY: &str = "DIAR_CUDA_ARENA_STRATEGY";

fn env_usize(key: &str) -> Option<usize> {
    std::env::var(key).ok().and_then(|s| s.trim().parse().ok())
}

fn env_i32(key: &str) -> Option<i32> {
    std::env::var(key).ok().and_then(|s| s.trim().parse().ok())
}

fn cuda_dispatch() -> ExecutionProviderDispatch {
    let mut cuda = ort::ep::CUDAExecutionProvider::default();
    if let Some(id) = env_i32(ENV_DEVICE_ID) {
        cuda = cuda.with_device_id(id);
    }
    if let Some(mb) = env_usize(ENV_GPU_MEM_LIMIT_MB) {
        cuda = cuda.with_memory_limit(mb * 1024 * 1024);
    }
    let search = match std::env::var(ENV_CONV_SEARCH)
        .unwrap_or_default()
        .trim()
        .to_ascii_lowercase()
        .as_str()
    {
        "exhaustive" => ort::ep::cuda::ConvAlgorithmSearch::Exhaustive,
        "default" => ort::ep::cuda::ConvAlgorithmSearch::Default,
        _ => ort::ep::cuda::ConvAlgorithmSearch::Heuristic,
    };
    if std::env::var(ENV_ARENA_STRATEGY)
        .unwrap_or_default()
        .trim()
        .eq_ignore_ascii_case("same_as_requested")
    {
        cuda = cuda.with_arena_extend_strategy(ort::ep::ArenaExtendStrategy::SameAsRequested);
    }
    let nhwc = std::env::var(ENV_PREFER_NHWC)
        .map(|v| matches!(v.trim(), "1" | "true" | "yes"))
        .unwrap_or(false);
    cuda.with_conv_algorithm_search(search)
        .with_prefer_nhwc(nhwc)
        .build()
}

fn candidates() -> Vec<(&'static str, ExecutionProviderDispatch)> {
    let requested = std::env::var(ENV_EP)
        .ok()
        .map(|s| s.trim().to_ascii_lowercase());

    let mut out: Vec<(&'static str, ExecutionProviderDispatch)> = Vec::new();
    let want = |name: &str| match requested.as_deref() {
        None | Some("") | Some("auto") => true,
        Some(r) => r == name,
    };

    if cfg!(any(target_os = "linux", target_os = "windows")) && want("cuda") {
        out.push((EP_CUDA, cuda_dispatch()));
    }
    if cfg!(target_os = "linux") && want("rocm") {
        out.push((EP_ROCM, ort::ep::ROCmExecutionProvider::default().build()));
    }
    if cfg!(target_os = "linux") && requested.as_deref() == Some("migraphx") {
        out.push((
            EP_MIGRAPHX,
            ort::ep::MIGraphXExecutionProvider::default().build(),
        ));
    }
    if cfg!(any(target_os = "macos", target_os = "ios")) && want("coreml") {
        out.push((
            EP_COREML,
            ort::ep::CoreMLExecutionProvider::default().build(),
        ));
    }
    if cfg!(target_os = "windows") && want("directml") {
        out.push((
            EP_DIRECTML,
            ort::ep::DirectMLExecutionProvider::default().build(),
        ));
    }
    out.push((EP_CPU, ort::ep::CPUExecutionProvider::default().build()));
    out
}

pub fn warmup_enabled() -> bool {
    !matches!(
        std::env::var("DIAR_WARMUP").unwrap_or_default().trim(),
        "0" | "false" | "no"
    )
}

pub fn is_gpu(provider: &str) -> bool {
    matches!(
        provider,
        EP_CUDA | EP_ROCM | EP_MIGRAPHX | EP_COREML | EP_DIRECTML
    )
}

pub fn intra_threads(provider: &str) -> usize {
    if let Some(n) = env_usize(ENV_INTRA_THREADS) {
        return n;
    }
    if is_gpu(provider) {
        return 1;
    }
    std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1)
        .clamp(1, 8)
}

pub struct Loaded {
    pub session: Session,
    pub provider: &'static str,
    pub intra_threads: usize,
}

pub fn load_session(model_path: &Path, label: &str) -> Result<Loaded> {
    let mut last_err: Option<anyhow::Error> = None;

    for (name, dispatch) in candidates() {
        let builder = match Session::builder() {
            Ok(b) => b,
            Err(e) => {
                last_err = Some(crate::vad::ort_err(e));
                continue;
            }
        };
        let builder = match builder.with_execution_providers([dispatch.error_on_failure()]) {
            Ok(b) => b,
            Err(e) => {
                warn!(model = label, ep = name, error = %e, "diarization: execution provider unavailable, trying next");
                last_err = Some(crate::vad::ort_err(e));
                continue;
            }
        };

        let threads = intra_threads(name);
        let built = builder
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .map_err(crate::vad::ort_err)
            .and_then(|b| b.with_intra_threads(threads).map_err(crate::vad::ort_err))
            .and_then(|mut b| b.commit_from_file(model_path).map_err(crate::vad::ort_err));

        match built {
            Ok(session) => {
                info!(
                    model = label,
                    ep = name,
                    intra_threads = threads,
                    path = %model_path.display(),
                    "diarization session ready"
                );
                return Ok(Loaded {
                    session,
                    provider: name,
                    intra_threads: threads,
                });
            }
            Err(e) => {
                warn!(model = label, ep = name, error = %e, "diarization: session commit failed, trying next execution provider");
                last_err = Some(e);
            }
        }
    }

    Err(last_err.unwrap_or_else(|| anyhow!("no execution provider could be registered")))
        .with_context(|| format!("load {} from {}", label, model_path.display()))
}
