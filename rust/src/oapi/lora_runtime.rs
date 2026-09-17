use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::Router;
use serde::Deserialize;

use super::chat::ChatAppState;
use super::{kind, openai_error};

fn parse_json_400_like_vllm<T: serde::de::DeserializeOwned>(
    body: &axum::body::Bytes,
) -> Result<T, Box<Response>> {
    serde_json::from_slice(body).map_err(|e| {
        Box::new(openai_error(
            StatusCode::BAD_REQUEST,
            format!("{e}"),
            kind::INVALID_REQUEST,
            None,
            Some("invalid_request_body"),
        ))
    })
}

pub const RUNTIME_LORA_ENV: &str = "NV_ALLOW_RUNTIME_LORA_UPDATING";

pub fn enabled() -> bool {
    match std::env::var(RUNTIME_LORA_ENV) {
        Ok(v) => {
            let v = v.trim().to_ascii_lowercase();
            !v.is_empty() && v != "0" && v != "false" && v != "off"
        }
        Err(_) => false,
    }
}

#[derive(Debug, Clone, Deserialize)]
pub struct LoadLoraAdapterRequest {
    pub lora_name: String,
    pub lora_path: String,
    #[serde(default)]
    pub load_inplace: bool,
}

#[derive(Debug, Clone, Deserialize)]
pub struct UnloadLoraAdapterRequest {
    pub lora_name: String,
}

pub fn load_success_body(name: &str) -> String {
    format!("Success: LoRA adapter '{name}' added successfully.")
}

pub fn unload_success_body(name: &str) -> String {
    format!("Success: LoRA adapter '{name}' removed successfully.")
}

static LOADED_AT_UNIX_SECS_RUNTIME_LOADS_ONLY_BOOT_ADAPTERS_LIST_NULL: std::sync::OnceLock<
    std::sync::Mutex<std::collections::HashMap<String, u64>>,
> = std::sync::OnceLock::new();

fn loaded_at_lock() -> &'static std::sync::Mutex<std::collections::HashMap<String, u64>> {
    LOADED_AT_UNIX_SECS_RUNTIME_LOADS_ONLY_BOOT_ADAPTERS_LIST_NULL.get_or_init(Default::default)
}

pub fn mark_loaded_now(id: &str) {
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    loaded_at_lock().lock().unwrap().insert(id.to_string(), now);
}

pub fn unmark_loaded(id: &str) {
    loaded_at_lock().lock().unwrap().remove(id);
}

pub fn loaded_at(id: &str) -> Option<u64> {
    loaded_at_lock().lock().unwrap().get(id).copied()
}

pub fn adapter_rows(registry: &super::chat_engine::ChatRegistry) -> Vec<serde_json::Value> {
    super::lora::all_entries()
        .iter()
        .map(|e| {
            let loaded = registry.contains(&e.id);
            serde_json::json!({
                "id": e.id,
                "object": "adapter",
                "base_model": e.base_model,
                "path": e.dir.display().to_string(),
                "rank": e.rank,
                "alpha": e.alpha,
                "scaling": e.scaling,
                "target_modules": e.targets,
                "status": if loaded { "loaded" } else { "available" },
                "loaded_at": loaded_at(&e.id),
            })
        })
        .collect()
}

async fn handle_list(axum::extract::State(state): axum::extract::State<ChatAppState>) -> Response {
    axum::Json(serde_json::json!({
        "object": "list",
        "data": adapter_rows(&state.registry),
    }))
    .into_response()
}

pub fn routes() -> Router<ChatAppState> {
    let list = Router::new().route("/v1/adapters", get(handle_list));
    if !enabled() {
        return list;
    }
    list.route("/v1/load_lora_adapter", post(handle_load))
        .route("/v1/unload_lora_adapter", post(handle_unload))
        .route("/v1/adapters/promote", post(handle_promote))
}

#[derive(serde::Deserialize)]
struct PromoteRequest {
    lora_name: String,
}

pub fn promote_to_boot_default(id: &str) -> anyhow::Result<std::path::PathBuf> {
    let entry = super::lora::registered()
        .get(id)
        .cloned()
        .or_else(|| super::lora::served().get(id).cloned())
        .ok_or_else(|| {
            anyhow::anyhow!("adapter {id:?} is neither registered nor in the served catalog")
        })?;
    let current_root = super::fine_tuning::jobs_root().join(
        super::lora::CURRENT_LINKS_DIR_HOLDS_ONE_SYMLINK_PER_MODEL_SUFFIX_SO_BOOT_LOADS_THE_ROLLING_ADAPTER_NOT_EVERY_PAST_JOB,
    );
    super::lora::roll_current_adapter_symlink(&current_root, &entry.id, &entry.dir)
}

async fn handle_promote(body: axum::body::Bytes) -> Response {
    let req: PromoteRequest = match parse_json_400_like_vllm(&body) {
        Ok(r) => r,
        Err(resp) => return *resp,
    };
    match promote_to_boot_default(&req.lora_name) {
        Ok(link) => axum::Json(serde_json::json!({
            "promoted": req.lora_name,
            "link": link.display().to_string(),
        }))
        .into_response(),
        Err(err) => (
            axum::http::StatusCode::BAD_REQUEST,
            axum::Json(serde_json::json!({
                "error": {
                    "message": format!("{err:#}"),
                    "type": "invalid_request_error",
                }
            })),
        )
            .into_response(),
    }
}

async fn handle_unload(
    axum::extract::State(state): axum::extract::State<ChatAppState>,
    body: axum::body::Bytes,
) -> Response {
    let req: UnloadLoraAdapterRequest = match parse_json_400_like_vllm(&body) {
        Ok(r) => r,
        Err(resp) => return *resp,
    };
    let known_adapter = super::lora::all_entries()
        .iter()
        .any(|e| e.id == req.lora_name);
    if !known_adapter && state.registry.contains(&req.lora_name) {
        return openai_error(
            StatusCode::BAD_REQUEST,
            format!(
                "The model `{}` is a base model, not a LoRA adapter; unload_lora_adapter only \
                 removes adapters.",
                req.lora_name
            ),
            kind::INVALID_REQUEST,
            Some("lora_name"),
            None,
        );
    }
    match state.registry.remove(&req.lora_name) {
        Ok(()) => {
            super::lora::deregister_adapter(&req.lora_name);
            unmark_loaded(&req.lora_name);
            #[cfg(feature = "wgpu")]
            super::chat_engine_wgpu::unrecord_wgpu_model_id(&req.lora_name);
            (StatusCode::OK, unload_success_body(&req.lora_name)).into_response()
        }
        Err(super::chat_engine::RemoveDenied::NotFound) => openai_error(
            StatusCode::NOT_FOUND,
            format!("The lora adapter `{}` has not been loaded.", req.lora_name),
            kind::NOT_FOUND,
            Some("lora_name"),
            None,
        ),
        Err(super::chat_engine::RemoveDenied::DefaultEngine) => openai_error(
            StatusCode::BAD_REQUEST,
            format!(
                "The model `{}` is the default engine and cannot be unloaded.",
                req.lora_name
            ),
            kind::INVALID_REQUEST,
            Some("lora_name"),
            None,
        ),
    }
}

#[cfg(feature = "wgpu")]
async fn handle_load(
    axum::extract::State(state): axum::extract::State<ChatAppState>,
    body: axum::body::Bytes,
) -> Response {
    let req: LoadLoraAdapterRequest = match parse_json_400_like_vllm(&body) {
        Ok(r) => r,
        Err(resp) => return *resp,
    };
    if let Err(err) = super::lora::validate_id(&req.lora_name) {
        return openai_error(
            StatusCode::BAD_REQUEST,
            format!("{err:#}"),
            kind::INVALID_REQUEST,
            Some("lora_name"),
            None,
        );
    }
    let shadows_a_base_engine = state.registry.contains(&req.lora_name)
        && !super::lora::all_entries()
            .iter()
            .any(|e| e.id == req.lora_name);
    if shadows_a_base_engine {
        return openai_error(
            StatusCode::BAD_REQUEST,
            format!(
                "The model `{}` is a loaded base model; an adapter may not shadow it under any \
                 flag, because every client naming the base would silently get adapter output.",
                req.lora_name
            ),
            kind::INVALID_REQUEST,
            Some("lora_name"),
            None,
        );
    }
    if state.registry.contains(&req.lora_name) && !req.load_inplace {
        return openai_error(
            StatusCode::BAD_REQUEST,
            format!(
                "The lora adapter `{}` is already loaded; pass load_inplace to replace it.",
                req.lora_name
            ),
            kind::INVALID_REQUEST,
            Some("lora_name"),
            None,
        );
    }
    let dir = std::path::PathBuf::from(&req.lora_path);
    let entry = match super::lora::probe_adapter(&dir, Some(&req.lora_name)) {
        Ok(entry) => entry,
        Err(err) => {
            return openai_error(
                StatusCode::BAD_REQUEST,
                format!("lora_path {:?}: {err:#}", req.lora_path),
                kind::INVALID_REQUEST,
                Some("lora_path"),
                None,
            );
        }
    };
    let bases = super::chat_engine_wgpu::wgpu_base_dirs();
    let base = match super::chat_engine_wgpu::select_base(entry.base_model.as_deref(), &bases) {
        Ok(b) => b.clone(),
        Err(reason) => {
            return openai_error(
                StatusCode::BAD_REQUEST,
                reason,
                kind::INVALID_REQUEST,
                Some("lora_path"),
                None,
            );
        }
    };
    let name = req.lora_name.clone();
    let adapter_dir = entry.dir.clone();
    let loaded = tokio::task::spawn_blocking(move || {
        super::chat_engine_wgpu::WgpuChatEngine::load_with_lora(
            &base.1,
            super::chat_engine_wgpu::WgpuChatEngine::default_max_seq_for(&base.1),
            Some(name),
            Some(&adapter_dir),
        )
    })
    .await;
    match loaded {
        Ok(Ok(eng)) => {
            state.registry.insert(std::sync::Arc::new(eng));
            super::lora::register_adapter(entry);
            mark_loaded_now(&req.lora_name);
            super::chat_engine_wgpu::record_wgpu_model_id(&req.lora_name);
            (StatusCode::OK, load_success_body(&req.lora_name)).into_response()
        }
        Ok(Err(err)) => openai_error(
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("loading adapter {:?}: {err:#}", req.lora_name),
            kind::SERVER,
            None,
            None,
        ),
        Err(join) => openai_error(
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("adapter load worker panicked: {join}"),
            kind::SERVER,
            None,
            None,
        ),
    }
}

#[cfg(all(feature = "cuda", not(feature = "wgpu")))]
async fn handle_load(
    axum::extract::State(state): axum::extract::State<ChatAppState>,
    body: axum::body::Bytes,
) -> Response {
    let req: LoadLoraAdapterRequest = match parse_json_400_like_vllm(&body) {
        Ok(r) => r,
        Err(resp) => return *resp,
    };
    if let Err(err) = super::lora::validate_id(&req.lora_name) {
        return openai_error(
            StatusCode::BAD_REQUEST,
            format!("{err:#}"),
            kind::INVALID_REQUEST,
            Some("lora_name"),
            None,
        );
    }
    let shadows_a_base_engine = state.registry.contains(&req.lora_name)
        && !super::lora::all_entries()
            .iter()
            .any(|e| e.id == req.lora_name);
    if shadows_a_base_engine {
        return openai_error(
            StatusCode::BAD_REQUEST,
            format!(
                "The model `{}` is a loaded base model; an adapter may not shadow it under any \
                 flag, because every client naming the base would silently get adapter output.",
                req.lora_name
            ),
            kind::INVALID_REQUEST,
            Some("lora_name"),
            None,
        );
    }
    if state.registry.contains(&req.lora_name) && !req.load_inplace {
        return openai_error(
            StatusCode::BAD_REQUEST,
            format!(
                "The lora adapter `{}` is already loaded; pass load_inplace to replace it.",
                req.lora_name
            ),
            kind::INVALID_REQUEST,
            Some("lora_name"),
            None,
        );
    }
    let dir = std::path::PathBuf::from(&req.lora_path);
    let entry = match super::lora::probe_adapter(&dir, Some(&req.lora_name)) {
        Ok(entry) => entry,
        Err(err) => {
            return openai_error(
                StatusCode::BAD_REQUEST,
                format!("lora_path {:?}: {err:#}", req.lora_path),
                kind::INVALID_REQUEST,
                Some("lora_path"),
                None,
            );
        }
    };
    let bases = super::chat_engine::cuda_base_dirs();
    let base = match super::chat_engine::select_cuda_base(entry.base_model.as_deref(), &bases) {
        Ok(b) => b.clone(),
        Err(reason) => {
            return openai_error(
                StatusCode::BAD_REQUEST,
                reason,
                kind::INVALID_REQUEST,
                Some("lora_path"),
                None,
            );
        }
    };
    let name = req.lora_name.clone();
    let adapter_dir = entry.dir.clone();
    let loaded = tokio::task::spawn_blocking(move || {
        super::chat_engine::CudaLoraAdapterEngine::try_load(&base.1, &name, &adapter_dir)
    })
    .await;
    match loaded {
        Ok(Ok(eng)) => {
            state.registry.insert(std::sync::Arc::new(eng));
            super::lora::register_adapter(entry);
            mark_loaded_now(&req.lora_name);
            (StatusCode::OK, load_success_body(&req.lora_name)).into_response()
        }
        Ok(Err(err)) => openai_error(
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("loading adapter {:?}: {err:#}", req.lora_name),
            kind::SERVER,
            None,
            None,
        ),
        Err(join) => openai_error(
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("adapter load worker panicked: {join}"),
            kind::SERVER,
            None,
            None,
        ),
    }
}

#[cfg(not(any(feature = "wgpu", feature = "cuda")))]
async fn handle_load(
    axum::extract::State(_state): axum::extract::State<ChatAppState>,
    body: axum::body::Bytes,
) -> Response {
    let req: LoadLoraAdapterRequest = match parse_json_400_like_vllm(&body) {
        Ok(r) => r,
        Err(resp) => return *resp,
    };
    openai_error(
        StatusCode::NOT_IMPLEMENTED,
        format!(
            "runtime LoRA loading serves through the wgpu or cuda chat engines and this build \
             has neither backend (adapter {:?} not loaded)",
            req.lora_name
        ),
        kind::SERVER,
        None,
        None,
    )
}

#[cfg(any(feature = "wgpu", feature = "cuda"))]
fn declared_base_matches(declared: Option<&str>, bases: &[(String, std::path::PathBuf)]) -> bool {
    let Some(want) = declared else {
        return false;
    };
    let leaf = want.rsplit('/').next().unwrap_or(want);
    bases
        .iter()
        .any(|(id, _)| id == want || id == leaf || id.rsplit('/').next() == Some(leaf))
}

#[cfg(feature = "wgpu")]
fn hot_load_wgpu(
    entry: &super::lora::AdapterEntry,
    registry: &super::chat_engine::ChatRegistry,
) -> anyhow::Result<String> {
    let bases = super::chat_engine_wgpu::wgpu_base_dirs();
    let base = super::chat_engine_wgpu::select_base(entry.base_model.as_deref(), &bases)
        .map_err(|reason| anyhow::anyhow!(reason))?
        .clone();
    let eng = super::chat_engine_wgpu::WgpuChatEngine::load_with_lora(
        &base.1,
        super::chat_engine_wgpu::WgpuChatEngine::default_max_seq_for(&base.1),
        Some(entry.id.clone()),
        Some(&entry.dir),
    )?;
    registry.insert(std::sync::Arc::new(eng));
    mark_loaded_now(&entry.id);
    super::chat_engine_wgpu::record_wgpu_model_id(&entry.id);
    Ok(base.0)
}

#[cfg(feature = "cuda")]
fn hot_load_cuda(
    entry: &super::lora::AdapterEntry,
    registry: &super::chat_engine::ChatRegistry,
    bases: &[(String, std::path::PathBuf)],
) -> anyhow::Result<String> {
    let base = super::chat_engine::select_cuda_base(entry.base_model.as_deref(), bases)
        .map_err(|reason| anyhow::anyhow!(reason))?
        .clone();
    let eng = super::chat_engine::CudaLoraAdapterEngine::try_load(&base.1, &entry.id, &entry.dir)?;
    registry.insert(std::sync::Arc::new(eng));
    mark_loaded_now(&entry.id);
    Ok(base.0)
}

#[cfg(any(feature = "wgpu", feature = "cuda"))]
pub fn hot_load(entry: &super::lora::AdapterEntry) -> anyhow::Result<String> {
    let registry = super::chat_engine::live_registry()
        .ok_or_else(|| anyhow::anyhow!("no live chat registry to insert the adapter into"))?;
    #[cfg(feature = "wgpu")]
    let wgpu_bases = super::chat_engine_wgpu::wgpu_base_dirs();
    #[cfg(feature = "cuda")]
    let cuda_bases = super::chat_engine::cuda_base_dirs();
    #[cfg(feature = "wgpu")]
    if declared_base_matches(entry.base_model.as_deref(), &wgpu_bases) {
        return hot_load_wgpu(entry, &registry);
    }
    #[cfg(feature = "cuda")]
    if declared_base_matches(entry.base_model.as_deref(), &cuda_bases) {
        return hot_load_cuda(entry, &registry, &cuda_bases);
    }
    #[cfg(feature = "wgpu")]
    if !wgpu_bases.is_empty() {
        return hot_load_wgpu(entry, &registry);
    }
    #[cfg(feature = "cuda")]
    if !cuda_bases.is_empty() {
        return hot_load_cuda(entry, &registry, &cuda_bases);
    }
    anyhow::bail!(
        "no loaded base chat engine on any backend serves adapter {:?} (declared base {:?})",
        entry.id,
        entry.base_model
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn success_bodies_reproduce_vllm_bare_strings_not_json() {
        assert_eq!(
            load_success_body("a1"),
            "Success: LoRA adapter 'a1' added successfully."
        );
        assert_eq!(
            unload_success_body("a1"),
            "Success: LoRA adapter 'a1' removed successfully."
        );
        assert!(serde_json::from_str::<serde_json::Value>(&load_success_body("a1")).is_err());
    }
}
