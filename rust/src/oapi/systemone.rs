use std::path::PathBuf;
use std::sync::Arc;

use axum::body::Bytes;
use axum::extract::State;
use axum::http::{header, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::Router;
use nv_laya::engine::pick_devices;
use nv_laya::json::Json;
use nv_laya::router::{normalise, Policy};
use nv_laya::service::{Failure, FineTuned, Service, Shape, UNPROCESSABLE};
use tracing::{info, warn};

use super::cold_tier::ColdSlot;
use super::{kind, openai_error, systemone_clm, systemone_kev};

pub const ENV_MODEL_DIR: &str = "NV_LAYA_MODEL_DIR";
pub const ENV_PRELOAD: &str = "NV_LAYA_PRELOAD";

const ALL_THREE_F16_CHECKPOINTS_PLUS_ACTIVATIONS: u64 = 4 * 1024 * 1024 * 1024;
const TOKENS_MERGED_INTO_ONE_FORWARD_PASS: usize = 16384;
const ONE_WORKER_BECAUSE_A_SECOND_REPLICA_ON_THE_SAME_GPU_BUYS_NOTHING: usize = 1;
const FIRST_CHECKPOINT_PROBED: &str = "laya/model.safetensors";

pub const FINE_TUNED_ID_LEAF_INSIDE_A_MERGED_CHECKPOINT_IS_HOW_A_RESTART_REDISCOVERS_IT: &str =
    "fine_tuned_model";

pub fn fine_tuned() -> FineTuned {
    static REGISTRY: std::sync::OnceLock<FineTuned> = std::sync::OnceLock::new();
    REGISTRY
        .get_or_init(|| {
            let mut found = std::collections::HashMap::new();
            for job in std::fs::read_dir(super::fine_tuning::jobs_root())
                .into_iter()
                .flatten()
                .flatten()
            {
                let merged = job
                    .path()
                    .join("adapter")
                    .join(nv_train::laya::MERGED_CHECKPOINT_LEAF_BESIDE_THE_ADAPTER);
                if let Ok(id) = std::fs::read_to_string(merged.join(
                    FINE_TUNED_ID_LEAF_INSIDE_A_MERGED_CHECKPOINT_IS_HOW_A_RESTART_REDISCOVERS_IT,
                )) {
                    found.insert(id.trim().to_string(), merged);
                }
            }
            Arc::new(std::sync::RwLock::new(found))
        })
        .clone()
}

pub fn register_fine_tuned(id: &str, merged: &std::path::Path) -> std::io::Result<()> {
    std::fs::write(
        merged.join(FINE_TUNED_ID_LEAF_INSIDE_A_MERGED_CHECKPOINT_IS_HOW_A_RESTART_REDISCOVERS_IT),
        id,
    )?;
    fine_tuned()
        .write()
        .unwrap()
        .insert(id.to_string(), merged.to_path_buf());
    Ok(())
}

pub fn model_dir() -> Option<PathBuf> {
    std::env::var_os(ENV_MODEL_DIR).map(PathBuf::from)
}

pub fn layout_present(dir: &std::path::Path) -> bool {
    dir.join(FIRST_CHECKPOINT_PROBED).is_file()
}

fn load_from_env() -> Option<Arc<Service>> {
    let dir = model_dir()?;
    let settings = match pick_devices(
        "auto",
        ONE_WORKER_BECAUSE_A_SECOND_REPLICA_ON_THE_SAME_GPU_BUYS_NOTHING,
    ) {
        Ok(settings) => settings,
        Err(err) => {
            warn!(error = %format!("{err:#}"), "systemone: no device for the Laya checkpoints");
            return None;
        }
    };
    let policy = Policy {
        default: "english",
        auto_task_detection: false,
    };
    let service = Service::start(
        dir,
        settings,
        policy,
        TOKENS_MERGED_INTO_ONE_FORWARD_PASS,
        fine_tuned(),
    );
    let preload = std::env::var(ENV_PRELOAD).unwrap_or_else(|_| "english".into());
    for name in preload.split(',').filter(|n| !n.trim().is_empty()) {
        let loaded = normalise(name)
            .map_err(|e| e.to_string())
            .and_then(|m| service.checkpoint(m).map_err(|e| e.message));
        if let Err(error) = loaded {
            warn!(checkpoint = name, error, "systemone: preload failed");
            return None;
        }
    }
    Some(service)
}

fn laya_from_env() -> Option<Slot> {
    let dir = model_dir()?;
    if !layout_present(&dir) {
        warn!(
            path = %dir.display(),
            "NV_LAYA_MODEL_DIR set but laya/model.safetensors is missing under it; Laya disabled"
        );
        return None;
    }
    info!(path = %dir.display(), "systemone: Laya checkpoints configured; they load on first use");
    Some(ColdSlot::demand(
        "systemone-laya",
        ALL_THREE_F16_CHECKPOINTS_PLUS_ACTIVATIONS,
        Arc::new(load_from_env),
    ))
}

pub fn configured() -> bool {
    model_dir().is_some_and(|dir| dir.is_dir())
        || !systemone_kev::configured_dirs().is_empty()
        || !systemone_clm::configured().is_empty()
}

pub fn router_from_env() -> Option<Router> {
    let backends = Backends {
        laya: laya_from_env(),
        kev: systemone_kev::from_env(),
        clm: systemone_clm::from_env(),
    };
    if backends.laya.is_none() && backends.kev.is_none() && backends.clm.is_none() {
        return None;
    }
    Some(
        Router::new()
            .route("/v1/systemone", post(jev_shaped))
            .route("/v1/systemone/detailed", post(detailed))
            .route("/v1/systemone/presets", get(presets))
            .with_state(Arc::new(backends)),
    )
}

type Slot = Arc<ColdSlot<Service>>;

pub struct Backends {
    laya: Option<Slot>,
    kev: Option<Arc<systemone_kev::Models>>,
    clm: Option<Arc<systemone_clm::Models>>,
}

fn json_response(status: StatusCode, body: &Json) -> Response {
    (
        status,
        [(header::CONTENT_TYPE, "application/json")],
        body.to_string(),
    )
        .into_response()
}

fn failure_response(failure: &Failure) -> Response {
    if failure.status == UNPROCESSABLE {
        openai_error(
            StatusCode::UNPROCESSABLE_ENTITY,
            failure.message.clone(),
            kind::INVALID_REQUEST,
            None,
            None,
        )
    } else {
        openai_error(
            StatusCode::SERVICE_UNAVAILABLE,
            failure.message.clone(),
            kind::SERVICE_UNAVAIL,
            None,
            None,
        )
    }
}

async fn answer(backends: Arc<Backends>, body: Bytes, shape: Shape) -> Response {
    let request: Json = match serde_json::from_slice(&body) {
        Ok(request) => request,
        Err(err) => {
            return openai_error(
                StatusCode::UNPROCESSABLE_ENTITY,
                format!("request body is not JSON: {err}"),
                kind::INVALID_REQUEST,
                None,
                None,
            );
        }
    };
    let model = request
        .get("model")
        .and_then(Json::as_str)
        .unwrap_or_default();
    if let Some(slot) = backends.kev.as_ref().and_then(|k| k.slot(model)) {
        return answer_kev(slot, request).await;
    }
    if let Some(slot) = backends.clm.as_ref().and_then(|c| c.slot(model)) {
        let Some(clm) = slot.get().await else {
            return failure_response(&Failure {
                status: nv_laya::service::UNAVAILABLE,
                message: "the CLM checkpoint could not be loaded".into(),
            });
        };
        return match tokio::task::spawn_blocking(move || clm.answer(&request)).await {
            Ok(Ok(body)) => json_response(StatusCode::OK, &body),
            Ok(Err(error)) => failure_response(&error),
            Err(error) => openai_error(
                StatusCode::INTERNAL_SERVER_ERROR,
                error.to_string(),
                kind::SERVER,
                None,
                None,
            ),
        };
    }
    let Some(slot) = backends.laya.clone() else {
        let known: Vec<&str> = backends
            .kev
            .iter()
            .flat_map(|k| k.names())
            .chain(backends.clm.iter().flat_map(|c| c.names()))
            .collect();
        return openai_error(
            StatusCode::NOT_FOUND,
            format!("model {model:?} is not served here; /v1/systemone serves {known:?}"),
            kind::NOT_FOUND,
            Some("model"),
            None,
        );
    };
    let Some(service) = slot.get().await else {
        return openai_error(
            StatusCode::SERVICE_UNAVAILABLE,
            "the Laya checkpoints could not be loaded",
            kind::SERVICE_UNAVAIL,
            None,
            None,
        );
    };
    let outcome =
        tokio::task::spawn_blocking(
            move || match request.get("requests").and_then(Json::as_arr) {
                Some(list) => {
                    let results = service
                        .predict_many(list, shape)
                        .into_iter()
                        .map(|r| r.unwrap_or_else(|e| e.json()))
                        .collect();
                    Ok(Json::obj([("results", Json::Arr(results))]))
                }
                None => service.predict(&request, shape),
            },
        )
        .await;
    match outcome {
        Ok(Ok(body)) => json_response(StatusCode::OK, &body),
        Ok(Err(failure)) => failure_response(&failure),
        Err(join) => openai_error(
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("systemone task failed: {join}"),
            kind::SERVER,
            None,
            None,
        ),
    }
}

async fn answer_kev(slot: Arc<ColdSlot<systemone_kev::Kev>>, request: Json) -> Response {
    let Some(kev) = slot.get().await else {
        return openai_error(
            StatusCode::SERVICE_UNAVAILABLE,
            "the Kev checkpoint could not be loaded",
            kind::SERVICE_UNAVAIL,
            None,
            None,
        );
    };
    match tokio::task::spawn_blocking(move || kev.answer(&request)).await {
        Ok(Ok(body)) => json_response(StatusCode::OK, &body),
        Ok(Err(failure)) => failure_response(&failure),
        Err(join) => openai_error(
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("systemone kev task failed: {join}"),
            kind::SERVER,
            None,
            None,
        ),
    }
}

async fn jev_shaped(State(backends): State<Arc<Backends>>, body: Bytes) -> Response {
    answer(backends, body, Shape::Jev).await
}

async fn detailed(State(backends): State<Arc<Backends>>, body: Bytes) -> Response {
    answer(backends, body, Shape::Native).await
}

async fn presets() -> Response {
    (
        StatusCode::OK,
        [(header::CONTENT_TYPE, "application/json")],
        nv_laya::service::PRESETS,
    )
        .into_response()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_route_is_absent_until_a_checkpoint_directory_is_configured_and_populated() {
        let _env = crate::oapi::lock_serializing_every_test_that_mutates_the_process_global_env();
        unsafe { std::env::remove_var(ENV_MODEL_DIR) };
        unsafe { std::env::remove_var(systemone_kev::ENV_MODEL_DIRS) };
        unsafe { std::env::remove_var(systemone_clm::ENV_MODEL_DIRS) };
        assert!(router_from_env().is_none());
        let empty =
            std::env::temp_dir().join(format!("speaches-plus-systemone-{}", std::process::id()));
        std::fs::create_dir_all(&empty).unwrap();
        unsafe { std::env::set_var(ENV_MODEL_DIR, &empty) };
        assert!(
            router_from_env().is_none(),
            "an empty directory must not enable /v1/systemone"
        );
        unsafe { std::env::remove_var(ENV_MODEL_DIR) };
        let _ = std::fs::remove_dir_all(&empty);
    }

    #[test]
    fn a_validation_failure_is_a_422_and_a_load_failure_is_a_503() {
        let invalid = failure_response(&Failure {
            status: UNPROCESSABLE,
            message: "bad question".into(),
        });
        assert_eq!(invalid.status(), StatusCode::UNPROCESSABLE_ENTITY);
        let unavailable = failure_response(&Failure {
            status: nv_laya::service::UNAVAILABLE,
            message: "no weights".into(),
        });
        assert_eq!(unavailable.status(), StatusCode::SERVICE_UNAVAILABLE);
    }
}
