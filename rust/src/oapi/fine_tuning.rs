use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::{Mutex, OnceLock};
use std::time::{SystemTime, UNIX_EPOCH};

use axum::extract::Path as AxumPath;
use axum::extract::Query as AxumQuery;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use serde::{Deserialize, Serialize};

use nv_models::train_runner::{self, TrainArgs};

use super::{kind, openai_error};

#[derive(Debug, Clone, Deserialize)]
#[cfg_attr(
    feature = "ts-bindings",
    derive(ts_rs::TS),
    ts(export, optional_fields = nullable)
)]
pub struct Hyperparameters {
    pub rank: Option<usize>,

    pub alpha: Option<f64>,

    pub steps: Option<usize>,
    #[serde(alias = "n_steps")]
    pub n_steps: Option<usize>,

    pub lr: Option<f64>,
    #[serde(alias = "learning_rate")]
    pub learning_rate: Option<f64>,

    pub target: Option<Vec<String>>,

    pub seed: Option<u64>,
}

#[derive(Debug, Clone, Deserialize)]
#[cfg_attr(
    feature = "ts-bindings",
    derive(ts_rs::TS),
    ts(export, optional_fields = nullable)
)]
pub struct CreateFineTuningJobRequest {
    pub model: String,

    pub training_file: Option<String>,

    pub training_data: Option<Vec<serde_json::Value>>,

    pub suffix: Option<String>,

    pub output_dir: Option<String>,
    #[serde(default)]
    pub hyperparameters: Option<Hyperparameters>,
}

#[derive(Debug, Clone, Serialize)]
#[cfg_attr(feature = "ts-bindings", derive(ts_rs::TS), ts(export))]
pub struct JobHyperparameters {
    pub rank: usize,
    pub alpha: f64,
    pub steps: usize,
    pub learning_rate: f64,
    pub target: Vec<String>,
    pub seed: u64,
}

#[derive(Debug, Clone, Serialize)]
#[cfg_attr(feature = "ts-bindings", derive(ts_rs::TS), ts(export))]
pub struct FineTuningJob {
    #[cfg_attr(feature = "ts-bindings", ts(type = "\"fine_tuning.job\""))]
    pub object: &'static str,
    pub id: String,
    pub created_at: u64,
    pub finished_at: Option<u64>,
    pub model: String,

    pub fine_tuned_model: Option<String>,

    pub status: String,
    pub training_file: Option<String>,
    pub hyperparameters: JobHyperparameters,

    pub result_files: Vec<String>,
    pub trained_tokens: Option<u64>,
    pub error: Option<serde_json::Value>,

    pub metrics: Option<serde_json::Value>,
}

fn jobs() -> &'static Mutex<HashMap<String, FineTuningJob>> {
    static JOBS: OnceLock<Mutex<HashMap<String, FineTuningJob>>> = OnceLock::new();
    JOBS.get_or_init(|| Mutex::new(HashMap::new()))
}

fn now_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

fn jobs_root() -> PathBuf {
    if let Ok(p) = std::env::var("NV_FINE_TUNING_DIR") {
        return PathBuf::from(p);
    }
    let base = std::env::var_os("XDG_CACHE_HOME")
        .map(PathBuf::from)
        .or_else(|| std::env::var_os("HOME").map(|h| PathBuf::from(h).join(".cache")))
        .unwrap_or_else(std::env::temp_dir);
    base.join("speaches-fine-tuning")
}

fn store(job: &FineTuningJob) {
    jobs().lock().unwrap().insert(job.id.clone(), job.clone());
}

pub fn get_job(id: &str) -> Option<FineTuningJob> {
    jobs().lock().unwrap().get(id).cloned()
}

pub fn list_jobs() -> Vec<FineTuningJob> {
    let mut v: Vec<FineTuningJob> = jobs().lock().unwrap().values().cloned().collect();
    v.sort_by_key(|j| std::cmp::Reverse(j.created_at));
    v
}

fn resolve_hp(hp: &Option<Hyperparameters>) -> JobHyperparameters {
    let hp = hp.clone().unwrap_or(Hyperparameters {
        rank: None,
        alpha: None,
        steps: None,
        n_steps: None,
        lr: None,
        learning_rate: None,
        target: None,
        seed: None,
    });
    let rank = hp.rank.unwrap_or(8).max(1);
    JobHyperparameters {
        rank,
        alpha: hp.alpha.unwrap_or(rank as f64),
        steps: hp.steps.or(hp.n_steps).unwrap_or(20),
        learning_rate: hp.lr.or(hp.learning_rate).unwrap_or(0.05),
        target: hp.target.unwrap_or_else(TrainArgs::default_targets),
        seed: hp.seed.unwrap_or(0),
    }
}

fn prepare_dataset(
    req: &CreateFineTuningJobRequest,
    job_dir: &std::path::Path,
) -> Result<PathBuf, (StatusCode, String)> {
    if let Some(inline) = &req.training_data {
        if inline.is_empty() {
            return Err((StatusCode::BAD_REQUEST, "training_data is empty".into()));
        }
        std::fs::create_dir_all(job_dir).map_err(|e| {
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("mkdir job dir: {e}"),
            )
        })?;
        let path = job_dir.join("training.jsonl");
        let mut body = String::new();
        for v in inline {
            body.push_str(&v.to_string());
            body.push('\n');
        }
        std::fs::write(&path, body).map_err(|e| {
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("write dataset: {e}"),
            )
        })?;
        return Ok(path);
    }
    if let Some(file) = &req.training_file {
        let path = PathBuf::from(file);
        if !path.is_file() {
            return Err((
                StatusCode::BAD_REQUEST,
                format!(
                    "training_file {:?} is not a readable path (this API takes a jsonl path, not \
                     an uploaded file id -- see the /v1/files seam in the module docs)",
                    file
                ),
            ));
        }
        return Ok(path);
    }
    Err((
        StatusCode::BAD_REQUEST,
        "provide either training_file (a jsonl path) or training_data (inline jsonl objects)"
            .into(),
    ))
}

fn model_leaf(model: &str) -> String {
    std::path::Path::new(model)
        .file_stem()
        .and_then(|s| s.to_str())
        .filter(|s| !s.is_empty())
        .unwrap_or("base")
        .to_string()
}

struct PreparedJob {
    id: String,
    args: TrainArgs,
    out_dir: PathBuf,
    ft_model: String,
}

fn prepare_job(
    req: &CreateFineTuningJobRequest,
) -> Result<(FineTuningJob, PreparedJob), (StatusCode, String)> {
    let base = PathBuf::from(&req.model);
    let base_ok = base.extension().and_then(|e| e.to_str()) == Some("gguf") && base.is_file();
    if !base_ok && !base.is_dir() {
        return Err((
            StatusCode::BAD_REQUEST,
            format!(
                "model {:?} must be a .gguf file or a directory with config.json + \
                 model.safetensors",
                req.model
            ),
        ));
    }

    let hp = resolve_hp(&req.hyperparameters);
    let id = format!("ftjob-{}", uuid::Uuid::new_v4().simple());
    let job_dir = jobs_root().join(&id);
    let data_path = prepare_dataset(req, &job_dir)?;

    let out_dir = req
        .output_dir
        .as_ref()
        .map(PathBuf::from)
        .unwrap_or_else(|| job_dir.join("adapter"));

    let suffix = req
        .suffix
        .clone()
        .filter(|s| !s.is_empty() && !s.chars().any(char::is_whitespace))
        .unwrap_or_else(|| "custom".into());
    let short = id
        .trim_start_matches("ftjob-")
        .chars()
        .take(8)
        .collect::<String>();
    let ft_model = format!("ft:{}:{}:{}", model_leaf(&req.model), suffix, short);

    let job = FineTuningJob {
        object: "fine_tuning.job",
        id: id.clone(),
        created_at: now_secs(),
        finished_at: None,
        model: req.model.clone(),
        fine_tuned_model: None,
        status: "queued".into(),
        training_file: req.training_file.clone(),
        hyperparameters: hp.clone(),
        result_files: Vec::new(),
        trained_tokens: None,
        error: None,
        metrics: None,
    };
    store(&job);

    let args = TrainArgs {
        base,
        data: data_path,
        out: out_dir.clone(),
        rank: hp.rank,
        alpha: hp.alpha,
        targets: hp.target.clone(),
        steps: hp.steps,
        lr: hp.learning_rate,
        seed: hp.seed,
    };

    Ok((
        job,
        PreparedJob {
            id,
            args,
            out_dir,
            ft_model,
        },
    ))
}

fn run_job(prepared: PreparedJob) {
    let PreparedJob {
        id,
        args,
        out_dir,
        ft_model,
    } = prepared;

    set_status(&id, "running");

    match train_runner::run(&args) {
        Ok(summary) => {
            let registered = match super::lora::probe_adapter(&out_dir, Some(&ft_model)) {
                Ok(entry) => {
                    super::lora::register_adapter(entry);
                    true
                }
                Err(err) => {
                    tracing::warn!(
                        adapter = %ft_model,
                        dir = %out_dir.display(),
                        error = %format!("{err:#}"),
                        "fine-tuned adapter trained but failed catalog registration"
                    );
                    false
                }
            };
            let (loss_first, loss_last) = (
                summary.losses.first().copied(),
                summary.losses.last().copied(),
            );
            update_job(&id, |job| {
                job.status = "succeeded".into();
                job.finished_at = Some(now_secs());
                job.fine_tuned_model = Some(ft_model.clone());
                job.result_files = vec![
                    out_dir.display().to_string(),
                    summary.adapter_path.display().to_string(),
                    summary.config_path.display().to_string(),
                ];
                job.metrics = Some(serde_json::json!({
                    "loss_first": loss_first,
                    "loss_last": loss_last,
                    "serving_equiv_maxabs": summary.serving_equiv_maxabs,
                    "trainable_vars": summary.trainable_vars,
                    "num_examples": summary.num_examples,
                    "base_dtype": summary.base_dtype,
                    "device": summary.device,
                    "deterministic": summary.deterministic,
                    "modules": summary.modules,
                    "registered": registered,
                }));
            });
        }
        Err(err) => {
            update_job(&id, |job| {
                job.status = "failed".into();
                job.finished_at = Some(now_secs());
                job.error = Some(serde_json::json!({
                    "code": "training_failed",
                    "message": format!("{err:#}"),
                }));
            });
        }
    }
}

fn set_status(id: &str, status: &str) {
    let mut guard = jobs().lock().unwrap();
    if let Some(job) = guard.get_mut(id) {
        job.status = status.into();
    }
}

fn update_job(id: &str, f: impl FnOnce(&mut FineTuningJob)) {
    let mut guard = jobs().lock().unwrap();
    if let Some(job) = guard.get_mut(id) {
        f(job);
    }
}

pub fn create_job_async(
    req: &CreateFineTuningJobRequest,
) -> Result<FineTuningJob, (StatusCode, String)> {
    let (job, prepared) = prepare_job(req)?;
    if let Err(e) = std::thread::Builder::new()
        .name(format!("ft-worker-{}", job.id))
        .spawn(move || run_job(prepared))
    {
        update_job(&job.id, |j| {
            j.status = "failed".into();
            j.finished_at = Some(now_secs());
            j.error = Some(serde_json::json!({
                "code": "worker_spawn_failed",
                "message": format!("failed to spawn fine-tuning worker: {e}"),
            }));
        });
        return Err((
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("failed to spawn fine-tuning worker: {e}"),
        ));
    }
    Ok(job)
}

pub fn create_job(req: &CreateFineTuningJobRequest) -> Result<FineTuningJob, (StatusCode, String)> {
    let (job, prepared) = prepare_job(req)?;
    let id = job.id.clone();
    run_job(prepared);
    Ok(get_job(&id).unwrap_or(job))
}

pub fn router() -> Router {
    Router::new()
        .route(
            "/v1/fine_tuning/jobs",
            post(create_handler).get(list_handler),
        )
        .route("/v1/fine_tuning/jobs/{id}", get(get_handler))
}

#[derive(Debug, Clone, Default, Deserialize)]
#[cfg_attr(
    feature = "ts-bindings",
    derive(ts_rs::TS),
    ts(export, rename = "FineTuningCreateQuery")
)]
pub struct CreateQuery {
    #[serde(default)]
    pub wait: bool,
}

pub async fn create_handler(
    AxumQuery(q): AxumQuery<CreateQuery>,
    Json(req): Json<CreateFineTuningJobRequest>,
) -> Response {
    if q.wait {
        return match tokio::task::spawn_blocking(move || create_job(&req)).await {
            Ok(Ok(job)) => (StatusCode::OK, Json(job)).into_response(),
            Ok(Err((status, msg))) => openai_error(
                status,
                msg,
                kind::INVALID_REQUEST,
                None,
                Some("invalid_request"),
            ),
            Err(join) => openai_error(
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("fine-tuning worker panicked: {join}"),
                kind::SERVER,
                None,
                None,
            ),
        };
    }
    match create_job_async(&req) {
        Ok(job) => (StatusCode::OK, Json(job)).into_response(),
        Err((status, msg)) if status == StatusCode::BAD_REQUEST => openai_error(
            status,
            msg,
            kind::INVALID_REQUEST,
            None,
            Some("invalid_request"),
        ),
        Err((status, msg)) => openai_error(status, msg, kind::SERVER, None, None),
    }
}

pub async fn get_handler(AxumPath(id): AxumPath<String>) -> Response {
    match get_job(&id) {
        Some(job) => (StatusCode::OK, Json(job)).into_response(),
        None => openai_error(
            StatusCode::NOT_FOUND,
            format!("no fine_tuning job {id:?}"),
            kind::NOT_FOUND,
            None,
            Some("job_not_found"),
        ),
    }
}

pub async fn list_handler() -> Response {
    let data = list_jobs();
    Json(serde_json::json!({ "object": "list", "data": data })).into_response()
}

#[cfg(feature = "ts-bindings")]
mod ts_wire {
    #![allow(dead_code)]
    use super::FineTuningJob;
    use ts_rs::TS;

    #[derive(TS)]
    #[ts(export)]
    struct ListFineTuningJobsResponse {
        #[ts(type = "\"list\"")]
        object: (),
        data: Vec<FineTuningJob>,
    }
}
