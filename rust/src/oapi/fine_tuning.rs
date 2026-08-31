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

use nv_models::laguna_train::{self, LagunaTrainArgs};
use nv_models::train_runner::{self, TrainArgs};
use nv_train::dots::{self, DotsTrainArgs};
use nv_train::qwen38::{self, Q38JsonlArgs};

use super::{kind, openai_error};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TrainerFamily {
    Qwen3_5DenseWindowed,
    GemmaTrainRunner,
    LagunaFullGraph,
    DotsQloraWindowed,
}

pub fn trainer_family_for_config(raw_cfg: &str) -> TrainerFamily {
    let Ok(v) = serde_json::from_str::<serde_json::Value>(raw_cfg) else {
        return TrainerFamily::GemmaTrainRunner;
    };
    let tags = super::model_family_tags::architectures_and_model_type_tags(&v);
    if super::model_family_tags::is_laguna(&tags) {
        return TrainerFamily::LagunaFullGraph;
    }
    if super::model_family_tags::is_dots(&tags) {
        return TrainerFamily::DotsQloraWindowed;
    }
    match super::model_family_tags::qwen3_5_tag(&tags) {
        super::model_family_tags::Qwen3_5Tag::Dense => TrainerFamily::Qwen3_5DenseWindowed,
        _ => TrainerFamily::GemmaTrainRunner,
    }
}

pub fn trainer_family_for_base(base: &std::path::Path) -> TrainerFamily {
    match std::fs::read_to_string(base.join("config.json")) {
        Ok(raw) => trainer_family_for_config(&raw),
        Err(_) => TrainerFamily::GemmaTrainRunner,
    }
}

pub const QWEN_GRAD_LAYERS_DEFAULT_IS_THE_CLI_WINDOW_BECAUSE_GDN_ACTIVATION_MEMORY_BOUNDS_THE_WINDOW_NOT_THE_WEIGHTS:
    usize = 4;

pub const DOTS_GRAD_LAYERS_DEFAULT_IS_8_BECAUSE_THE_QLORA_SMOKE_MEASURED_ITS_FOOTPRINT_THERE:
    usize = 8;

fn default_targets_for(family: TrainerFamily) -> Vec<String> {
    match family {
        TrainerFamily::Qwen3_5DenseWindowed => nv_train::qwen38::Q38TrainArgs::default_targets(),
        TrainerFamily::GemmaTrainRunner => TrainArgs::default_targets(),
        TrainerFamily::LagunaFullGraph => LagunaTrainArgs::default_targets(),
        TrainerFamily::DotsQloraWindowed => dots::dots_default_targets(),
    }
}

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

    pub grad_layers: Option<usize>,

    pub b_lr_ratio: Option<f64>,
}

#[derive(Debug, Clone, Deserialize)]
#[cfg_attr(
    feature = "ts-bindings",
    derive(ts_rs::TS),
    ts(export, optional_fields = nullable)
)]
pub struct SpeculatorSpec {
    pub kind: String,

    pub drafter_dir: Option<String>,

    pub rank: Option<usize>,

    pub lr_ratio: Option<f64>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SpeculatorKind {
    MtpSelfDraftingHead,
    Eagle3SeparateDrafter,
}

#[derive(Debug, Clone)]
pub struct ResolvedSpeculator {
    pub kind: SpeculatorKind,
    pub drafter_dir: Option<PathBuf>,
    pub rank: usize,
    pub lr_ratio: f64,
}

pub const SPECULATOR_RANK_DEFAULT_IS_32_FROM_THE_MEDUSA_2_CO_TRAINING_RECIPE: usize = 32;

pub const SPECULATOR_LR_RATIO_DEFAULT_IS_4_BECAUSE_MEDUSA_2_TRAINS_THE_SPECULATOR_HOTTER_THAN_THE_TARGET:
    f64 = 4.0;

pub const EAGLE3_DRAFTER_DIR_FALLBACK_IS_THE_SAME_ENV_VAR_SERVING_READS: &str =
    "NV_EAGLE3_DRAFT_DIR";

pub const SPECULATOR_SUBDIR_INSIDE_OUT_DIR_RIDES_THE_SAME_STAMP_SYMLINK_AND_HOT_LOAD_AS_THE_TARGET_ADAPTER:
    &str = "speculator";

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

    pub validation_file: Option<String>,

    pub validation_data: Option<Vec<serde_json::Value>>,

    pub suffix: Option<String>,

    pub output_dir: Option<String>,
    #[serde(default)]
    pub hyperparameters: Option<Hyperparameters>,
}

#[derive(Debug, Clone, Deserialize)]
#[cfg_attr(
    feature = "ts-bindings",
    derive(ts_rs::TS),
    ts(export, optional_fields = nullable)
)]
pub struct CreateFineTuningJobBody {
    #[serde(flatten)]
    pub job: CreateFineTuningJobRequest,

    pub speculator: Option<SpeculatorSpec>,
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
    #[serde(skip_serializing_if = "Option::is_none")]
    #[cfg_attr(feature = "ts-bindings", ts(optional))]
    pub grad_layers: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    #[cfg_attr(feature = "ts-bindings", ts(optional))]
    pub b_lr_ratio: Option<f64>,
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

    #[serde(skip_serializing_if = "std::ops::Not::not")]
    #[cfg_attr(feature = "ts-bindings", ts(optional = nullable))]
    pub speculator_trained: bool,
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

pub(crate) fn jobs_root() -> PathBuf {
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

fn resolve_hp(
    hp: &Option<Hyperparameters>,
    family: TrainerFamily,
) -> Result<JobHyperparameters, (StatusCode, String)> {
    let hp = hp.clone().unwrap_or(Hyperparameters {
        rank: None,
        alpha: None,
        steps: None,
        n_steps: None,
        lr: None,
        learning_rate: None,
        target: None,
        seed: None,
        grad_layers: None,
        b_lr_ratio: None,
    });
    let (grad_layers, b_lr_ratio) = match family {
        TrainerFamily::Qwen3_5DenseWindowed => (
            Some(
                hp.grad_layers
                    .unwrap_or(QWEN_GRAD_LAYERS_DEFAULT_IS_THE_CLI_WINDOW_BECAUSE_GDN_ACTIVATION_MEMORY_BOUNDS_THE_WINDOW_NOT_THE_WEIGHTS)
                    .max(1),
            ),
            Some(hp.b_lr_ratio.unwrap_or(
                nv_train::qwen38::B_LR_RATIO_DEFAULT_OF_ONE_IS_EXACTLY_THE_SINGLE_OPTIMIZER_PATH_AND_ABOVE_ONE_IS_LORA_PLUS,
            )),
        ),
        TrainerFamily::DotsQloraWindowed => (
            Some(
                hp.grad_layers
                    .unwrap_or(DOTS_GRAD_LAYERS_DEFAULT_IS_8_BECAUSE_THE_QLORA_SMOKE_MEASURED_ITS_FOOTPRINT_THERE)
                    .max(1),
            ),
            Some(hp.b_lr_ratio.unwrap_or(
                nv_train::qwen38::B_LR_RATIO_DEFAULT_OF_ONE_IS_EXACTLY_THE_SINGLE_OPTIMIZER_PATH_AND_ABOVE_ONE_IS_LORA_PLUS,
            )),
        ),
        TrainerFamily::GemmaTrainRunner | TrainerFamily::LagunaFullGraph => {
            if hp.grad_layers.is_some() || hp.b_lr_ratio.is_some() {
                let trainer = match family {
                    TrainerFamily::GemmaTrainRunner => "train_runner",
                    _ => "the laguna full-graph trainer",
                };
                return Err((
                    StatusCode::BAD_REQUEST,
                    format!(
                        "grad_layers and b_lr_ratio are knobs of the qwen3.5 windowed trainer; \
                         this base routes to {trainer}, which trains full depth with a single \
                         optimizer and would silently ignore them"
                    ),
                ));
            }
            (None, None)
        }
    };
    let dots = family == TrainerFamily::DotsQloraWindowed;
    let rank = hp.rank.unwrap_or(if dots { 16 } else { 8 }).max(1);
    if dots && rank % 2 != 0 {
        return Err((
            StatusCode::BAD_REQUEST,
            format!(
                "rank {rank} is odd; the dots graph serving arm walks lora factors in bf16 \
                 pairs and refuses odd ranks, so the job would train an adapter the live \
                 hook cannot load -- use an even rank"
            ),
        ));
    }
    Ok(JobHyperparameters {
        rank,
        alpha: hp
            .alpha
            .unwrap_or(if dots { rank as f64 * 2.0 } else { rank as f64 }),
        steps: hp.steps.or(hp.n_steps).unwrap_or(20),
        learning_rate: hp
            .lr
            .or(hp.learning_rate)
            .unwrap_or(if dots { 1e-4 } else { 0.05 }),
        target: hp.target.unwrap_or_else(|| default_targets_for(family)),
        seed: hp.seed.unwrap_or(0),
        grad_layers,
        b_lr_ratio,
    })
}

fn resolve_speculator(
    spec: &Option<SpeculatorSpec>,
    family: TrainerFamily,
) -> Result<Option<ResolvedSpeculator>, (StatusCode, String)> {
    let Some(spec) = spec else {
        return Ok(None);
    };
    let kind = match spec.kind.as_str() {
        "mtp" => SpeculatorKind::MtpSelfDraftingHead,
        "eagle3" => SpeculatorKind::Eagle3SeparateDrafter,
        other => {
            return Err((
                StatusCode::BAD_REQUEST,
                format!(
                    "speculator kind {other:?} is neither \"mtp\" (the qwen3.5 checkpoint's \
                     self-drafting head) nor \"eagle3\" (the separate drafter model for the \
                     gemma train_runner family)"
                ),
            ));
        }
    };
    match (kind, family) {
        (_, TrainerFamily::DotsQloraWindowed) => {
            return Err((
                StatusCode::BAD_REQUEST,
                format!(
                    "speculator kind {:?} is unsupported for the dots.ocr family: the dots \
                     text decoder ships no MTP head and no trainer tap feeds an eagle3 \
                     drafter from its trunk -- drop the speculator",
                    spec.kind
                ),
            ));
        }
        (_, TrainerFamily::LagunaFullGraph) => {
            return Err((
                StatusCode::BAD_REQUEST,
                format!(
                    "speculator kind {:?} is unsupported for the laguna family: laguna \
                     checkpoints ship no MTP head to self-draft and no trainer tap feeds an \
                     eagle3 drafter from the laguna trunk, so co-training would ship weights \
                     laguna cannot serve -- drop the speculator",
                    spec.kind
                ),
            ));
        }
        (SpeculatorKind::MtpSelfDraftingHead, TrainerFamily::GemmaTrainRunner) => {
            return Err((
                StatusCode::BAD_REQUEST,
                "speculator kind \"mtp\" is the qwen3.5 windowed trainer's self-drafting head; \
                 this base routes to train_runner, whose checkpoints carry no MTP head, so \
                 co-training it would train weights the model cannot serve -- request kind \
                 \"eagle3\" with a drafter_dir, or drop the speculator"
                    .into(),
            ));
        }
        (SpeculatorKind::Eagle3SeparateDrafter, TrainerFamily::Qwen3_5DenseWindowed) => {
            return Err((
                StatusCode::BAD_REQUEST,
                "speculator kind \"eagle3\" is the separate drafter model for the train_runner \
                 (gemma) family; this base routes to the qwen3.5 windowed trainer, whose \
                 speculator is the checkpoint's own MTP head -- request kind \"mtp\", or drop \
                 the speculator"
                    .into(),
            ));
        }
        _ => {}
    }
    let lr_ratio = spec
        .lr_ratio
        .unwrap_or(SPECULATOR_LR_RATIO_DEFAULT_IS_4_BECAUSE_MEDUSA_2_TRAINS_THE_SPECULATOR_HOTTER_THAN_THE_TARGET);
    if lr_ratio <= 0.0 || !lr_ratio.is_finite() {
        return Err((
            StatusCode::BAD_REQUEST,
            format!(
                "speculator lr_ratio {lr_ratio} must be a positive finite multiplier of the \
                 target learning rate"
            ),
        ));
    }
    let drafter_dir = match (&spec.drafter_dir, kind) {
        (Some(dir), SpeculatorKind::MtpSelfDraftingHead) => {
            return Err((
                StatusCode::BAD_REQUEST,
                format!(
                    "speculator kind \"mtp\" takes no drafter_dir: the self-drafting head loads \
                     from the base checkpoint's own model_mtp.safetensors, so {dir:?} would be \
                     silently ignored"
                ),
            ));
        }
        (Some(dir), SpeculatorKind::Eagle3SeparateDrafter) => {
            let p = PathBuf::from(dir);
            if !p.is_dir() {
                return Err((
                    StatusCode::BAD_REQUEST,
                    format!("speculator drafter_dir {dir:?} is not a directory"),
                ));
            }
            Some(p)
        }
        (None, SpeculatorKind::Eagle3SeparateDrafter) => {
            match std::env::var(EAGLE3_DRAFTER_DIR_FALLBACK_IS_THE_SAME_ENV_VAR_SERVING_READS)
                .ok()
                .map(PathBuf::from)
            {
                Some(p) if p.is_dir() => Some(p),
                Some(p) => {
                    return Err((
                        StatusCode::BAD_REQUEST,
                        format!(
                            "{} is set to {:?}, which is not a directory -- serving disables \
                             spec-decode on the same condition; point the env var at the drafter \
                             checkpoint directory or pass speculator.drafter_dir to override it",
                            EAGLE3_DRAFTER_DIR_FALLBACK_IS_THE_SAME_ENV_VAR_SERVING_READS, p
                        ),
                    ));
                }
                None => {
                    return Err((
                        StatusCode::BAD_REQUEST,
                        format!(
                            "speculator kind \"eagle3\" trains a separate drafter model and \
                             needs its weights: pass speculator.drafter_dir or set {} to the \
                             drafter checkpoint directory",
                            EAGLE3_DRAFTER_DIR_FALLBACK_IS_THE_SAME_ENV_VAR_SERVING_READS
                        ),
                    ));
                }
            }
        }
        (None, SpeculatorKind::MtpSelfDraftingHead) => None,
    };
    Ok(Some(ResolvedSpeculator {
        kind,
        drafter_dir,
        rank: spec
            .rank
            .unwrap_or(SPECULATOR_RANK_DEFAULT_IS_32_FROM_THE_MEDUSA_2_CO_TRAINING_RECIPE)
            .max(1),
        lr_ratio,
    }))
}

fn materialize_jsonl(
    inline: &Option<Vec<serde_json::Value>>,
    file: &Option<String>,
    field: &str,
    job_dir: &std::path::Path,
    leaf: &str,
) -> Result<Option<PathBuf>, (StatusCode, String)> {
    if let Some(inline) = inline {
        if inline.is_empty() {
            return Err((StatusCode::BAD_REQUEST, format!("{field} is empty")));
        }
        std::fs::create_dir_all(job_dir).map_err(|e| {
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("mkdir job dir: {e}"),
            )
        })?;
        let path = job_dir.join(leaf);
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
        return Ok(Some(path));
    }
    if let Some(file) = &file {
        let path = PathBuf::from(file);
        if !path.is_file() {
            return Err((
                StatusCode::BAD_REQUEST,
                format!(
                    "{field} {:?} is not a readable path (this API takes a jsonl path, not \
                     an uploaded file id -- see the /v1/files seam in the module docs)",
                    file
                ),
            ));
        }
        return Ok(Some(path));
    }
    Ok(None)
}

fn prepare_dataset(
    req: &CreateFineTuningJobRequest,
    job_dir: &std::path::Path,
) -> Result<PathBuf, (StatusCode, String)> {
    materialize_jsonl(
        &req.training_data,
        &req.training_file,
        "training_data",
        job_dir,
        "training.jsonl",
    )?
    .ok_or((
        StatusCode::BAD_REQUEST,
        "provide either training_file (a jsonl path) or training_data (inline jsonl objects)"
            .into(),
    ))
}

fn dots_dump_dir(
    inline: &Option<Vec<serde_json::Value>>,
    file: &Option<String>,
    field_inline: &str,
    field_file: &str,
) -> Result<Option<PathBuf>, (StatusCode, String)> {
    if inline.is_some() {
        return Err((
            StatusCode::BAD_REQUEST,
            format!(
                "{field_inline} cannot be inline for the dots.ocr family: its samples carry \
                 precomputed vision-tower rows, so {field_file} must point at an \
                 nvk-dots-vision-dump output directory"
            ),
        ));
    }
    let Some(file) = file else {
        return Ok(None);
    };
    let path = PathBuf::from(file);
    if !path.is_dir() || !path.join("index.jsonl").is_file() {
        return Err((
            StatusCode::BAD_REQUEST,
            format!(
                "{field_file} {file:?} must be an nvk-dots-vision-dump output directory \
                 (safetensors samples + index.jsonl) for the dots.ocr family"
            ),
        ));
    }
    Ok(Some(path))
}

fn model_leaf(model: &str) -> String {
    std::path::Path::new(model)
        .file_stem()
        .and_then(|s| s.to_str())
        .filter(|s| !s.is_empty())
        .unwrap_or("base")
        .to_string()
}

enum PreparedTrain {
    Runner(TrainArgs, Option<train_runner::SpeculatorSpec>),
    Qwen38(Q38JsonlArgs, Option<qwen38::Q38MtpCoTrainArgs>),
    Laguna(LagunaTrainArgs),
    Dots(DotsTrainArgs),
}

struct PreparedJob {
    id: String,
    train: PreparedTrain,
    out_dir: PathBuf,
    ft_model: String,
    steps: usize,
    lr: f64,
}

struct Trained {
    adapter_path: PathBuf,
    config_path: PathBuf,
    losses: Vec<f32>,
    all_grads_finite: bool,
    speculator_dir: Option<PathBuf>,
    metrics: serde_json::Map<String, serde_json::Value>,
}

fn trained_from_runner(
    s: train_runner::TrainSummary,
    sp: Option<train_runner::SpeculatorTrainSummary>,
) -> Trained {
    let mut metrics = match serde_json::json!({
        "trainer": "train_runner",
        "eval_loss_adapter": s.eval_loss_adapter,
        "eval_loss_base": s.eval_loss_base,
        "eval_ce_adapter": s.eval_ce_adapter,
        "eval_ce_base": s.eval_ce_base,
        "serving_equiv_maxabs": s.serving_equiv_maxabs,
        "trainable_vars": s.trainable_vars,
        "num_examples": s.num_examples,
        "num_eval_examples": s.num_eval_examples,
        "base_dtype": s.base_dtype,
        "resident_base": s.resident_base,
        "device": s.device,
        "deterministic": s.deterministic,
        "all_grads_finite": s.all_grads_finite,
        "modules": s.modules,
    }) {
        serde_json::Value::Object(m) => m,
        _ => unreachable!("json! object literal"),
    };
    let mut speculator_dir = None;
    let mut all_grads_finite = s.all_grads_finite;
    if let Some(sp) = sp {
        all_grads_finite &= sp.all_grads_finite;
        metrics.insert(
            "speculator".into(),
            serde_json::json!({
                "kind": sp.kind,
                "head_loss_first": sp.losses.first(),
                "head_loss_last": sp.losses.last(),
                "trainable_vars": sp.trainable_vars,
                "all_grads_finite": sp.all_grads_finite,
                "modules": sp.modules,
            }),
        );
        speculator_dir = sp.adapter_path.parent().map(PathBuf::from);
    }
    Trained {
        adapter_path: s.adapter_path,
        config_path: s.config_path,
        losses: s.losses,
        all_grads_finite,
        speculator_dir,
        metrics,
    }
}

fn trained_from_qwen38(
    s: qwen38::Q38JsonlSummary,
    co: Option<qwen38::Q38MtpCoTrainSummary>,
) -> Trained {
    let mut metrics = match serde_json::json!({
        "trainer": "qwen38_windowed",
        "eval_loss_adapter": s.eval_loss_adapter,
        "eval_loss_base": s.eval_loss_base,
        "eval_ce_adapter": s.eval_ce_adapter,
        "eval_ce_base": s.eval_ce_base,
        "trainable_vars": s.trainable_vars,
        "num_examples": s.num_examples,
        "num_eval_examples": s.num_eval_examples,
        "base_dtype": s.base_dtype,
        "resident_base": s.resident_base,
        "device": s.device,
        "window_start": s.window_start,
        "window_kinds": s.window_kinds,
        "all_grads_finite": s.all_grads_finite,
        "checkpointed": s.checkpointed,
        "modules": s.modules,
    }) {
        serde_json::Value::Object(m) => m,
        _ => unreachable!("json! object literal"),
    };
    let mut speculator_dir = None;
    let mut all_grads_finite = s.all_grads_finite;
    if let Some(co) = co {
        all_grads_finite &= co.all_head_grads_finite;
        metrics.insert(
            "speculator".into(),
            serde_json::json!({
                "kind": co.speculator_kind,
                "head_loss_first": co.head_losses.first(),
                "head_loss_last": co.head_losses.last(),
                "trainable_vars": co.head_trainable_vars,
                "all_grads_finite": co.all_head_grads_finite,
                "modules": co.head_modules,
            }),
        );
        speculator_dir = Some(co.speculator_dir);
    }
    Trained {
        adapter_path: s.adapter_path,
        config_path: s.config_path,
        losses: s.losses,
        all_grads_finite,
        speculator_dir,
        metrics,
    }
}

fn trained_from_dots(s: dots::DotsTrainSummary) -> Trained {
    let metrics = match serde_json::json!({
        "trainer": "dots_qlora_windowed",
        "eval_loss_adapter": s.eval_loss_adapter(),
        "eval_loss_base": s.eval_loss_base(),
        "eval_ce_adapter": s.eval_ce_adapter,
        "eval_ce_base": s.eval_ce_base,
        "trainable_vars": s.trainable_vars,
        "num_examples": s.num_examples,
        "num_eval_examples": s.num_eval_examples,
        "device": s.device,
        "window_start": s.window_start,
        "prefix_seconds": s.prefix_seconds,
        "window_build_seconds": s.window_build_seconds,
        "min_module_delta_amax": s.min_module_delta_amax,
        "max_module_delta_amax": s.max_module_delta_amax,
        "all_grads_finite": s.all_grads_finite,
        "modules": s.modules,
    }) {
        serde_json::Value::Object(m) => m,
        _ => unreachable!("json! object literal"),
    };
    Trained {
        adapter_path: s.adapter_dir.join(super::lora::ADAPTER_WEIGHTS_FILE),
        config_path: s.adapter_dir.join(super::lora::ADAPTER_CONFIG_FILE),
        losses: s.losses,
        all_grads_finite: s.all_grads_finite,
        speculator_dir: None,
        metrics,
    }
}

fn trained_from_laguna(s: laguna_train::LagunaTrainSummary) -> Trained {
    let metrics = match serde_json::json!({
        "trainer": "laguna_full_graph",
        "eval_loss_adapter": s.eval_loss_adapter,
        "eval_loss_base": s.eval_loss_base,
        "eval_ce_adapter": s.eval_ce_adapter,
        "eval_ce_base": s.eval_ce_base,
        "trainable_vars": s.trainable_vars,
        "num_examples": s.num_examples,
        "num_eval_examples": s.num_eval_examples,
        "base_dtype": s.base_dtype,
        "device": s.device,
        "config_layers": s.config_layers,
        "moe_layers": s.moe_layers,
        "dense_mlp_layers": s.dense_mlp_layers,
        "all_grads_finite": s.all_grads_finite,
        "modules": s.modules,
    }) {
        serde_json::Value::Object(m) => m,
        _ => unreachable!("json! object literal"),
    };
    Trained {
        adapter_path: s.adapter_path,
        config_path: s.config_path,
        losses: s.losses,
        all_grads_finite: s.all_grads_finite,
        speculator_dir: None,
        metrics,
    }
}

pub const A_TARGETED_MODULE_WHOSE_LORA_B_IS_STILL_EXACTLY_ZERO_AFTER_A_TRAINING_STEP_RECEIVED_NO_GRADIENT_AND_MERGES_AS_AN_EXACT_NO_OP: &str =
    "every LoRA B factor is zero-initialised and only a gradient moves it, so a targeted module \
     whose B is still exactly zero after a step with a non-zero learning rate received no entry \
     in the GradStore at all -- candle's AdamW::step skips such a Var in silence, the loss falls \
     the whole run, and the adapter merges into the base as an exact no-op for those modules. \
     The known cause is an attention forward that scores through the fused \
     candle_nn::ops::softmax_last_dim, which candle installs with apply_op1_no_bwd: the \
     probabilities are a graph leaf, so self_attn.q_proj and self_attn.k_proj -- both DEFAULT \
     targets of this endpoint -- are unreachable from the loss while v still trains through the \
     probs @ v matmul. Score through \
     nv_layers::attn::softmax_last_dim_that_keeps_q_and_k_on_the_graph instead, or drop the dead \
     modules from `target`. The adapter that was written is left on disk untouched for \
     inspection.";

pub const A_NON_FINITE_GRADIENT_TURNS_THE_LORA_FACTORS_NAN_AND_THE_ZERO_B_CHECK_CANNOT_CATCH_IT: &str =
    "once any accumulated gradient is non-finite, AdamW writes NaN into the LoRA A/B factors and \
     every later step keeps them NaN, so the saved adapter serves NaN logits; the untrained-module \
     gate cannot catch this because a NaN B factor has NaN amax, which is not exactly zero. The \
     trainer surfaces all_grads_finite for exactly this, and the job must fail instead of \
     registering or hot-loading the adapter. Lower the learning rate or the step count. The \
     adapter that was written is left on disk untouched for inspection.";

fn modules_whose_lora_b_is_still_exactly_zero(
    adapter: &std::path::Path,
) -> anyhow::Result<Vec<String>> {
    let weights = nv_weights::WeightLoader::open_file(adapter, &candle_core::Device::Cpu)?;
    let mut dead = Vec::new();
    for name in weights.names() {
        if !name.ends_with("lora_B.weight") {
            continue;
        }
        let amax = weights
            .get(&name, candle_core::DType::F32)?
            .abs()?
            .flatten_all()?
            .max(0)?
            .to_scalar::<f32>()?;
        if amax == 0.0 {
            dead.push(name);
        }
    }
    dead.sort();
    Ok(dead)
}

fn prepare_job(
    req: &CreateFineTuningJobRequest,
    speculator: &Option<SpeculatorSpec>,
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

    let family = if base_ok {
        TrainerFamily::GemmaTrainRunner
    } else {
        trainer_family_for_base(&base)
    };
    let hp = resolve_hp(&req.hyperparameters, family)?;
    let speculator = resolve_speculator(speculator, family)?;
    let id = format!("ftjob-{}", uuid::Uuid::new_v4().simple());
    let job_dir = jobs_root().join(&id);
    let (data_path, eval_path) = if family == TrainerFamily::DotsQloraWindowed {
        let data = dots_dump_dir(
            &req.training_data,
            &req.training_file,
            "training_data",
            "training_file",
        )?
        .ok_or((
            StatusCode::BAD_REQUEST,
            "the dots.ocr family takes training_file = an nvk-dots-vision-dump output \
             directory"
                .to_string(),
        ))?;
        let eval = dots_dump_dir(
            &req.validation_data,
            &req.validation_file,
            "validation_data",
            "validation_file",
        )?;
        (data, eval)
    } else {
        let data = prepare_dataset(req, &job_dir)?;
        let eval = materialize_jsonl(
            &req.validation_data,
            &req.validation_file,
            "validation_data",
            &job_dir,
            "validation.jsonl",
        )?;
        (data, eval)
    };

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
        speculator_trained: false,
    };
    store(&job);

    let train = match family {
        TrainerFamily::Qwen3_5DenseWindowed => PreparedTrain::Qwen38(
            Q38JsonlArgs {
                base,
                data: data_path,
                eval_data: eval_path,
                out: out_dir.clone(),
                grad_layers: hp
                    .grad_layers
                    .expect("resolve_hp fills grad_layers for the qwen family"),
                rank: hp.rank,
                alpha: hp.alpha,
                targets: hp.target.clone(),
                steps: hp.steps,
                lr: hp.learning_rate,
                b_lr_ratio: hp
                    .b_lr_ratio
                    .expect("resolve_hp fills b_lr_ratio for the qwen family"),
                seed: hp.seed,
            },
            speculator.as_ref().map(|sp| qwen38::Q38MtpCoTrainArgs {
                rank: sp.rank,
                alpha: sp.rank as f64,
                speculator_lr_ratio: sp.lr_ratio,
            }),
        ),
        TrainerFamily::DotsQloraWindowed => PreparedTrain::Dots(DotsTrainArgs {
            base,
            data: data_path,
            eval_data: eval_path,
            out: out_dir.clone(),
            grad_layers: hp
                .grad_layers
                .expect("resolve_hp fills grad_layers for the dots family"),
            rank: hp.rank,
            alpha: hp.alpha,
            lr: hp.learning_rate,
            b_lr_ratio: hp
                .b_lr_ratio
                .expect("resolve_hp fills b_lr_ratio for the dots family"),
            steps: hp.steps,
            targets: hp.target.clone(),
            seed: hp.seed,
            dtype: candle_core::DType::F32,
        }),
        TrainerFamily::LagunaFullGraph => PreparedTrain::Laguna(LagunaTrainArgs {
            base,
            data: data_path,
            eval_data: eval_path,
            out: out_dir.clone(),
            rank: hp.rank,
            alpha: hp.alpha,
            targets: hp.target.clone(),
            steps: hp.steps,
            lr: hp.learning_rate,
            seed: hp.seed,
        }),
        TrainerFamily::GemmaTrainRunner => PreparedTrain::Runner(
            TrainArgs {
                base,
                data: data_path,
                eval_data: eval_path,
                out: out_dir.clone(),
                rank: hp.rank,
                alpha: hp.alpha,
                targets: hp.target.clone(),
                steps: hp.steps,
                lr: hp.learning_rate,
                seed: hp.seed,
            },
            speculator.as_ref().map(|sp| train_runner::SpeculatorSpec {
                drafter_dir: sp.drafter_dir.clone().expect(
                    "resolve_speculator always fills drafter_dir for the eagle3 kind, and \
                     eagle3 is the only kind that reaches the runner family",
                ),
                rank: sp.rank,
                lr_ratio: sp.lr_ratio,
            }),
        ),
    };

    Ok((
        job,
        PreparedJob {
            id,
            train,
            out_dir,
            ft_model,
            steps: hp.steps,
            lr: hp.learning_rate,
        },
    ))
}

pub const A_SUMMARY_WITHOUT_A_SPECULATOR_DIR_AFTER_A_SPECULATOR_WAS_REQUESTED_IS_A_DROPPED_KNOB_NOT_A_SUCCESS: &str =
    "the request asked for speculator co-training but training returned no speculator artifact, \
     so the dispatch from PreparedTrain to the co-train trainer entry points dropped the knob; \
     succeeding here would ship a job record that silently ignored the speculator request -- the \
     exact silent-ignore this endpoint's family-aware 400s exist to prevent -- so the job fails \
     instead.";

fn a_speculator_was_requested(train: &PreparedTrain) -> bool {
    matches!(
        train,
        PreparedTrain::Runner(_, Some(_)) | PreparedTrain::Qwen38(_, Some(_))
    )
}

pub const TRAINING_STACK_BYTES_ARE_64_MIB_BECAUSE_CANDLE_DROPS_THE_OP_GRAPH_RECURSIVELY_AND_A_42_LAYER_BASE_OVERFLOWS_THE_2_MIB_DEFAULT:
    usize = 64 << 20;

fn run_training_on_a_wide_stack(train: PreparedTrain) -> anyhow::Result<Trained> {
    std::thread::Builder::new()
        .name("ft-train".into())
        .stack_size(
            TRAINING_STACK_BYTES_ARE_64_MIB_BECAUSE_CANDLE_DROPS_THE_OP_GRAPH_RECURSIVELY_AND_A_42_LAYER_BASE_OVERFLOWS_THE_2_MIB_DEFAULT,
        )
        .spawn(move || match &train {
            PreparedTrain::Runner(args, spec) => train_runner::run_with_speculator(args, spec.as_ref())
                .map(|(s, sp)| trained_from_runner(s, sp)),
            PreparedTrain::Qwen38(args, co) => train_runner::select_device().and_then(|device| {
                match co {
                    None => qwen38::run_jsonl(args, &device).map(|s| trained_from_qwen38(s, None)),
                    Some(co) => qwen38::run_jsonl_with_mtp_cotrain(args, co, &device)
                        .map(|(s, c)| trained_from_qwen38(s, Some(c))),
                }
            }),
            PreparedTrain::Laguna(args) => laguna_train::run(args).map(trained_from_laguna),
            PreparedTrain::Dots(args) => train_runner::select_device()
                .and_then(|device| dots::run(args, &device).map(trained_from_dots)),
        })
        .map_err(|e| anyhow::anyhow!("spawn training thread: {e}"))?
        .join()
        .map_err(|panic| {
            anyhow::anyhow!(
                "training thread panicked: {}",
                panic
                    .downcast_ref::<String>()
                    .map(String::as_str)
                    .or_else(|| panic.downcast_ref::<&str>().copied())
                    .unwrap_or("non-string panic payload")
            )
        })?
}

fn run_job(prepared: PreparedJob) {
    let PreparedJob {
        id,
        train,
        out_dir,
        ft_model,
        steps,
        lr,
    } = prepared;

    set_status(&id, "running");
    persist_job_record_into_out_dir_for_the_gate_script_and_post_restart_forensics(&out_dir, &id);

    let a_gradient_was_actually_asked_for = steps > 0 && lr != 0.0;
    let the_request_asked_for_a_speculator = a_speculator_was_requested(&train);

    let trained = run_training_on_a_wide_stack(train);

    let the_one_terminal_transition_no_outcome_can_reach_disk_without: Box<
        dyn FnOnce(&mut FineTuningJob) + '_,
    > = match trained {
        Ok(summary) => {
            if the_request_asked_for_a_speculator && summary.speculator_dir.is_none() {
                Box::new(move |job| {
                    job.status = "failed".into();
                    job.finished_at = Some(now_secs());
                    job.error = Some(serde_json::json!({
                        "code": "speculator_missing_from_training_summary",
                        "message": A_SUMMARY_WITHOUT_A_SPECULATOR_DIR_AFTER_A_SPECULATOR_WAS_REQUESTED_IS_A_DROPPED_KNOB_NOT_A_SUCCESS,
                        "adapter_path": summary.adapter_path.display().to_string(),
                    }));
                })
            } else if !summary.all_grads_finite {
                Box::new(move |job| {
                    job.status = "failed".into();
                    job.finished_at = Some(now_secs());
                    job.error = Some(serde_json::json!({
                        "code": "non_finite_gradient",
                        "message": format!(
                            "a gradient went non-finite during {} steps at lr {}. {}",
                            steps,
                            lr,
                            A_NON_FINITE_GRADIENT_TURNS_THE_LORA_FACTORS_NAN_AND_THE_ZERO_B_CHECK_CANNOT_CATCH_IT,
                        ),
                        "adapter_path": summary.adapter_path.display().to_string(),
                        "loss_first": summary.losses.first().copied(),
                        "loss_last": summary.losses.last().copied(),
                    }));
                })
            } else {
                let dead = if a_gradient_was_actually_asked_for {
                    let mut dead = modules_whose_lora_b_is_still_exactly_zero(&summary.adapter_path)
                        .unwrap_or_else(|err| {
                            tracing::warn!(
                                adapter = %summary.adapter_path.display(),
                                error = %format!("{err:#}"),
                                "could not read back the adapter to check for untrained modules"
                            );
                            Vec::new()
                        });
                    if let Some(spec_dir) = &summary.speculator_dir {
                        let spec_adapter = spec_dir.join("adapter_model.safetensors");
                        match modules_whose_lora_b_is_still_exactly_zero(&spec_adapter) {
                            Ok(more) => dead.extend(more.into_iter().map(|m| {
                                format!(
                                    "{}/{m}",
                                    SPECULATOR_SUBDIR_INSIDE_OUT_DIR_RIDES_THE_SAME_STAMP_SYMLINK_AND_HOT_LOAD_AS_THE_TARGET_ADAPTER
                                )
                            })),
                            Err(err) => tracing::warn!(
                                adapter = %spec_adapter.display(),
                                error = %format!("{err:#}"),
                                "could not read back the speculator adapter to check for \
                                 untrained modules"
                            ),
                        }
                    }
                    dead
                } else {
                    Vec::new()
                };
                if !dead.is_empty() {
                    Box::new(move |job| {
                        job.status = "failed".into();
                        job.finished_at = Some(now_secs());
                        job.error = Some(serde_json::json!({
                            "code": "modules_received_no_gradient",
                            "message": format!(
                                "{} of the requested LoRA modules are still exactly zero after {} \
                                 steps at lr {}: {}. {}",
                                dead.len(),
                                steps,
                                lr,
                                dead.join(", "),
                                A_TARGETED_MODULE_WHOSE_LORA_B_IS_STILL_EXACTLY_ZERO_AFTER_A_TRAINING_STEP_RECEIVED_NO_GRADIENT_AND_MERGES_AS_AN_EXACT_NO_OP,
                            ),
                            "untrained_modules": dead,
                            "adapter_path": summary.adapter_path.display().to_string(),
                            "loss_first": summary.losses.first().copied(),
                            "loss_last": summary.losses.last().copied(),
                        }));
                    })
                } else {
                    finish_success_registering_and_hot_loading(summary, &out_dir, ft_model)
                }
            }
        }
        Err(err) => Box::new(move |job| {
            job.status = "failed".into();
            job.finished_at = Some(now_secs());
            job.error = Some(serde_json::json!({
                "code": "training_failed",
                "message": format!("{err:#}"),
            }));
        }),
    };
    update_job_then_persist_beside_the_adapter(
        &id,
        &out_dir,
        the_one_terminal_transition_no_outcome_can_reach_disk_without,
    );
}

fn finish_success_registering_and_hot_loading(
    summary: Trained,
    out_dir: &std::path::Path,
    ft_model: String,
) -> Box<dyn FnOnce(&mut FineTuningJob)> {
    if let Err(err) = super::lora::stamp_adapter_id(out_dir, &ft_model) {
        tracing::warn!(
            adapter = %ft_model,
            dir = %out_dir.display(),
            error = %format!("{err:#}"),
            "could not stamp {} into adapter_config.json; a restart will rediscover \
             this adapter under its directory name instead of its fine_tuned_model id",
            super::lora::STAMPED_ID_KEY
        );
    }
    tracing::info!(
        adapter = %ft_model,
        "job success does not persist a boot default; the held-out gate owner \
         promotes via POST /v1/adapters/promote, which rolls the current symlink"
    );
    if summary
        .metrics
        .get("trainer")
        .and_then(|v| v.as_str())
        == Some("dots_qlora_windowed")
    {
        tracing::info!(
            adapter = %ft_model,
            "a dots adapter serves through the OCR pipeline, not the chat catalog: \
             point NV_DOTS_LORA_DIR / NV_DOTS_GRAPH_LORA_DIR at the adapter dir, or \
             POST /v1/ocr/load_lora_adapter with the runtime env enabled"
        );
        let out_dir_display = out_dir.display().to_string();
        return Box::new(move |job| {
            job.status = "succeeded".into();
            job.finished_at = Some(now_secs());
            job.fine_tuned_model = Some(ft_model);
            job.result_files = vec![
                out_dir_display,
                summary.adapter_path.display().to_string(),
                summary.config_path.display().to_string(),
            ];
            let mut metrics = summary.metrics;
            metrics.insert(
                "loss_first".into(),
                serde_json::json!(summary.losses.first().copied()),
            );
            metrics.insert(
                "loss_last".into(),
                serde_json::json!(summary.losses.last().copied()),
            );
            metrics.insert("registered".into(), serde_json::json!(false));
            metrics.insert("hot_loaded".into(), serde_json::json!(false));
            metrics.insert("serving_route".into(), serde_json::json!("ocr"));
            job.metrics = Some(serde_json::Value::Object(metrics));
        });
    }
    #[cfg_attr(not(any(feature = "wgpu", feature = "cuda")), allow(unused_mut))]
    let mut hot_loaded = false;
    let registered = match super::lora::probe_adapter(out_dir, Some(&ft_model)) {
        Ok(entry) => {
            #[cfg(any(feature = "wgpu", feature = "cuda"))]
            if super::lora_runtime::enabled() {
                match super::lora_runtime::hot_load(&entry) {
                    Ok(base_id) => {
                        hot_loaded = true;
                        tracing::info!(
                            adapter = %ft_model,
                            base = %base_id,
                            "fine-tuned adapter hot-loaded; it now serves without a restart"
                        );
                    }
                    Err(err) => tracing::warn!(
                        adapter = %ft_model,
                        error = %format!("{err:#}"),
                        "fine-tuned adapter registered but hot-load failed; it serves \
                         after the next restart via {}",
                        super::lora::ADAPTER_DIRS_ENV
                    ),
                }
            }
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
    let out_dir_display = out_dir.display().to_string();
    Box::new(move |job| {
        job.status = "succeeded".into();
        job.finished_at = Some(now_secs());
        job.fine_tuned_model = Some(ft_model);
        job.result_files = vec![
            out_dir_display,
            summary.adapter_path.display().to_string(),
            summary.config_path.display().to_string(),
        ];
        if let Some(dir) = &summary.speculator_dir {
            job.result_files.push(dir.display().to_string());
        }
        job.speculator_trained = summary.speculator_dir.is_some();
        let mut metrics = summary.metrics;
        metrics.insert(
            "loss_first".into(),
            serde_json::json!(summary.losses.first().copied()),
        );
        metrics.insert(
            "loss_last".into(),
            serde_json::json!(summary.losses.last().copied()),
        );
        metrics.insert("registered".into(), serde_json::json!(registered));
        metrics.insert("hot_loaded".into(), serde_json::json!(hot_loaded));
        job.metrics = Some(serde_json::Value::Object(metrics));
    })
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

pub const JOB_JSON_LEAF_BESIDE_THE_ADAPTER_IS_THE_RECORD_OFFLINE_CONSUMERS_READ_BECAUSE_THE_JOBS_MAP_IS_RAM_ONLY:
    &str = "job.json";

pub const JOB_JSON_TMP_LEAF_NEVER_SURVIVES_A_TRANSITION_BECAUSE_RENAME_INTO_PLACE_IS_WHAT_MAKES_A_MID_WRITE_KILL_LEAVE_THE_LAST_CONSISTENT_RECORD:
    &str = "job.json.tmp";

fn persist_job_record_into_out_dir_for_the_gate_script_and_post_restart_forensics(
    out_dir: &std::path::Path,
    id: &str,
) {
    let Some(job) = get_job(id) else {
        return;
    };
    let body = match serde_json::to_vec_pretty(&job) {
        Ok(body) => body,
        Err(err) => {
            tracing::warn!(
                job = id,
                error = %err,
                "job record did not serialize; the RAM map stays the only copy"
            );
            return;
        }
    };
    let tmp = out_dir
        .join(JOB_JSON_TMP_LEAF_NEVER_SURVIVES_A_TRANSITION_BECAUSE_RENAME_INTO_PLACE_IS_WHAT_MAKES_A_MID_WRITE_KILL_LEAVE_THE_LAST_CONSISTENT_RECORD);
    let path = out_dir.join(
        JOB_JSON_LEAF_BESIDE_THE_ADAPTER_IS_THE_RECORD_OFFLINE_CONSUMERS_READ_BECAUSE_THE_JOBS_MAP_IS_RAM_ONLY,
    );
    if let Err(err) = std::fs::create_dir_all(out_dir)
        .and_then(|_| std::fs::write(&tmp, &body))
        .and_then(|_| std::fs::rename(&tmp, &path))
    {
        tracing::warn!(
            job = id,
            path = %path.display(),
            error = %err,
            "job record did not persist; the RAM map stays the only copy"
        );
    }
}

fn update_job_then_persist_beside_the_adapter(
    id: &str,
    out_dir: &std::path::Path,
    f: impl FnOnce(&mut FineTuningJob),
) {
    update_job(id, f);
    persist_job_record_into_out_dir_for_the_gate_script_and_post_restart_forensics(out_dir, id);
}

pub fn create_job_async(
    req: &CreateFineTuningJobRequest,
) -> Result<FineTuningJob, (StatusCode, String)> {
    create_job_async_with_speculator(req, &None)
}

pub fn create_job_async_with_speculator(
    req: &CreateFineTuningJobRequest,
    speculator: &Option<SpeculatorSpec>,
) -> Result<FineTuningJob, (StatusCode, String)> {
    let (job, prepared) = prepare_job(req, speculator)?;
    let out_dir_kept_out_of_the_spawned_closure_so_a_spawn_failure_still_persists_its_record =
        prepared.out_dir.clone();
    if let Err(e) = std::thread::Builder::new()
        .name(format!("ft-worker-{}", job.id))
        .spawn(move || run_job(prepared))
    {
        update_job_then_persist_beside_the_adapter(
            &job.id,
            &out_dir_kept_out_of_the_spawned_closure_so_a_spawn_failure_still_persists_its_record,
            |j| {
                j.status = "failed".into();
                j.finished_at = Some(now_secs());
                j.error = Some(serde_json::json!({
                    "code": "worker_spawn_failed",
                    "message": format!("failed to spawn fine-tuning worker: {e}"),
                }));
            },
        );
        return Err((
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("failed to spawn fine-tuning worker: {e}"),
        ));
    }
    Ok(job)
}

pub fn create_job(req: &CreateFineTuningJobRequest) -> Result<FineTuningJob, (StatusCode, String)> {
    create_job_with_speculator(req, &None)
}

pub fn create_job_with_speculator(
    req: &CreateFineTuningJobRequest,
    speculator: &Option<SpeculatorSpec>,
) -> Result<FineTuningJob, (StatusCode, String)> {
    let (job, prepared) = prepare_job(req, speculator)?;
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
    Json(body): Json<CreateFineTuningJobBody>,
) -> Response {
    let CreateFineTuningJobBody {
        job: req,
        speculator,
    } = body;
    if q.wait {
        return match tokio::task::spawn_blocking(move || {
            create_job_with_speculator(&req, &speculator)
        })
        .await
        {
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
    match create_job_async_with_speculator(&req, &speculator) {
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

#[cfg(test)]
mod tests {
    use super::*;

    use crate::oapi::lock_serializing_every_test_that_mutates_the_process_global_env;

    struct VarRemovedOnDropSoAFailedAssertCannotLeakItIntoLaterTests(&'static str);

    impl Drop for VarRemovedOnDropSoAFailedAssertCannotLeakItIntoLaterTests {
        fn drop(&mut self) {
            std::env::remove_var(self.0);
        }
    }

    fn qwen38_dense_cfg() -> &'static str {
        r#"{"architectures":["Qwen3_5ForConditionalGeneration"],"model_type":"qwen3_5",
            "text_config":{"hidden_size":5120,"num_hidden_layers":64}}"#
    }

    #[test]
    fn a_qwen38_dense_config_routes_to_the_windowed_trainer_and_everything_else_to_train_runner() {
        assert_eq!(
            trainer_family_for_config(qwen38_dense_cfg()),
            TrainerFamily::Qwen3_5DenseWindowed
        );
        assert_eq!(
            trainer_family_for_config(r#"{"model_type":"qwen3_5"}"#),
            TrainerFamily::Qwen3_5DenseWindowed,
            "model_type alone must classify, mirroring chat_engine detect_family's second pass"
        );
        assert_eq!(
            trainer_family_for_config(&nv_train::qwen38_fixture::tiny_config_json()),
            TrainerFamily::Qwen3_5DenseWindowed,
            "the tiny training fixture must route to the trainer that can actually load it"
        );
        for other in [
            r#"{"architectures":["Qwen3_5MoeForCausalLM"],"model_type":"qwen3_5_moe_text"}"#,
            r#"{"architectures":["Gemma4ForConditionalGeneration"],"model_type":"gemma4"}"#,
            r#"{"model_type":"qwen3"}"#,
            r#"{"model_type":"gpt_oss"}"#,
            r#"{}"#,
            "not json at all",
        ] {
            assert_eq!(
                trainer_family_for_config(other),
                TrainerFamily::GemmaTrainRunner,
                "{other}"
            );
        }
    }

    #[test]
    fn a_dots_config_routes_to_the_qlora_trainer_with_its_own_defaults() {
        for cfg in [
            r#"{"model_type":"dots_ocr"}"#,
            r#"{"architectures":["DotsOCRForCausalLM"],"model_type":"dots_ocr"}"#,
        ] {
            assert_eq!(
                trainer_family_for_config(cfg),
                TrainerFamily::DotsQloraWindowed,
                "{cfg}"
            );
        }
        let hp = resolve_hp(&None, TrainerFamily::DotsQloraWindowed).unwrap();
        assert_eq!(hp.rank, 16);
        assert_eq!(hp.alpha, 32.0);
        assert_eq!(hp.learning_rate, 1e-4);
        assert_eq!(
            hp.grad_layers,
            Some(DOTS_GRAD_LAYERS_DEFAULT_IS_8_BECAUSE_THE_QLORA_SMOKE_MEASURED_ITS_FOOTPRINT_THERE)
        );
        assert_eq!(hp.b_lr_ratio, Some(1.0));
        assert_eq!(
            hp.target,
            nv_train::dots::dots_default_targets(),
            "the api defaults must be the trainer's own"
        );
    }

    #[test]
    fn family_detection_reads_the_model_dirs_config_json_and_a_missing_one_stays_on_train_runner()
    {
        let root = std::env::temp_dir().join(format!("ft-family-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        let qwen = root.join("qwen");
        std::fs::create_dir_all(&qwen).unwrap();
        std::fs::write(qwen.join("config.json"), qwen38_dense_cfg()).unwrap();
        assert_eq!(
            trainer_family_for_base(&qwen),
            TrainerFamily::Qwen3_5DenseWindowed
        );
        let bare = root.join("bare");
        std::fs::create_dir_all(&bare).unwrap();
        assert_eq!(trainer_family_for_base(&bare), TrainerFamily::GemmaTrainRunner);
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn prepare_job_dispatches_on_the_config_without_touching_any_weights() {
        let _env = lock_serializing_every_test_that_mutates_the_process_global_env();
        let root = std::env::temp_dir().join(format!("ft-dispatch-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        std::env::set_var("NV_FINE_TUNING_DIR", root.join("jobs"));
        let qwen = root.join("qwen-base");
        std::fs::create_dir_all(&qwen).unwrap();
        std::fs::write(qwen.join("config.json"), qwen38_dense_cfg()).unwrap();
        let req = CreateFineTuningJobRequest {
            model: qwen.display().to_string(),
            training_file: None,
            training_data: Some(vec![serde_json::json!({"text": "hello world"})]),
            validation_file: None,
            validation_data: None,
            suffix: None,
            output_dir: None,
            hyperparameters: None,
        };
        let (job, prepared) = prepare_job(&req, &None).unwrap();
        assert!(
            matches!(prepared.train, PreparedTrain::Qwen38(..)),
            "a qwen3_5-dense config.json must select the windowed qwen38 trainer"
        );
        assert_eq!(
            job.hyperparameters.target,
            default_targets_for(TrainerFamily::Qwen3_5DenseWindowed),
            "the stored job must show the servable default target set"
        );

        let gemma = root.join("gemma-base");
        std::fs::create_dir_all(&gemma).unwrap();
        std::fs::write(
            gemma.join("config.json"),
            r#"{"architectures":["Gemma4ForConditionalGeneration"],"model_type":"gemma4"}"#,
        )
        .unwrap();
        let req = CreateFineTuningJobRequest {
            model: gemma.display().to_string(),
            ..req
        };
        let (job, prepared) = prepare_job(&req, &None).unwrap();
        assert!(matches!(prepared.train, PreparedTrain::Runner(..)));
        assert_eq!(job.hyperparameters.target, TrainArgs::default_targets());
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn qwen_default_targets_are_exactly_the_q38_trainer_defaults_now_that_serving_covers_gdn() {
        let defaults = default_targets_for(TrainerFamily::Qwen3_5DenseWindowed);
        assert_eq!(
            defaults,
            qwen38::Q38TrainArgs::default_targets(),
            "qwen3_5_dense_wgpu serves every projection the qwen38 trainer can attach \
             (deltanet sites included), so the endpoint withholds nothing; a new gap between \
             trainer and serving must reintroduce a withheld set deliberately, not by drift"
        );
        let explicit = Hyperparameters {
            rank: None,
            alpha: None,
            steps: None,
            n_steps: None,
            lr: None,
            learning_rate: None,
            target: Some(vec!["in_proj_qkv".into()]),
            seed: None,
            grad_layers: None,
            b_lr_ratio: None,
        };
        assert_eq!(
            resolve_hp(&Some(explicit), TrainerFamily::Qwen3_5DenseWindowed)
                .unwrap()
                .target,
            vec!["in_proj_qkv".to_string()],
            "GDN targets stay reachable by explicit request"
        );
    }

    #[test]
    fn grad_layers_defaults_to_the_gdn_bounded_window_and_a_requested_depth_reaches_the_qwen_trainer(
    ) {
        assert_eq!(
            resolve_hp(&None, TrainerFamily::Qwen3_5DenseWindowed)
                .unwrap()
                .grad_layers,
            Some(QWEN_GRAD_LAYERS_DEFAULT_IS_THE_CLI_WINDOW_BECAUSE_GDN_ACTIVATION_MEMORY_BOUNDS_THE_WINDOW_NOT_THE_WEIGHTS)
        );
        let deep = Hyperparameters {
            rank: None,
            alpha: None,
            steps: None,
            n_steps: None,
            lr: None,
            learning_rate: None,
            target: None,
            seed: None,
            grad_layers: Some(9),
            b_lr_ratio: None,
        };
        assert_eq!(
            resolve_hp(&Some(deep.clone()), TrainerFamily::Qwen3_5DenseWindowed)
                .unwrap()
                .grad_layers,
            Some(9)
        );
        assert_eq!(
            resolve_hp(
                &Some(Hyperparameters {
                    grad_layers: Some(0),
                    ..deep.clone()
                }),
                TrainerFamily::Qwen3_5DenseWindowed
            )
            .unwrap()
            .grad_layers,
            Some(1),
            "a zero window trains nothing; run_jsonl clamps the top to num_hidden_layers"
        );

        let _env = lock_serializing_every_test_that_mutates_the_process_global_env();
        let root = std::env::temp_dir().join(format!("ft-gradlayers-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        std::env::set_var("NV_FINE_TUNING_DIR", root.join("jobs"));
        let qwen = root.join("qwen-base");
        std::fs::create_dir_all(&qwen).unwrap();
        std::fs::write(qwen.join("config.json"), qwen38_dense_cfg()).unwrap();
        let req = CreateFineTuningJobRequest {
            model: qwen.display().to_string(),
            training_file: None,
            training_data: Some(vec![serde_json::json!({"text": "hello world"})]),
            validation_file: None,
            validation_data: None,
            suffix: None,
            output_dir: None,
            hyperparameters: Some(deep),
        };
        let (job, prepared) = prepare_job(&req, &None).unwrap();
        assert_eq!(job.hyperparameters.grad_layers, Some(9));
        match prepared.train {
            PreparedTrain::Qwen38(args, _) => assert_eq!(
                args.grad_layers, 9,
                "the requested window depth must reach Q38JsonlArgs, not the fixed default"
            ),
            PreparedTrain::Runner(..) | PreparedTrain::Laguna(..) | PreparedTrain::Dots(..) => {
                panic!("a qwen base must route to the windowed trainer")
            }
        }
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn the_runner_family_rejects_the_qwen_only_knobs_instead_of_silently_ignoring_them() {
        let unset = resolve_hp(&None, TrainerFamily::GemmaTrainRunner).unwrap();
        assert_eq!(
            (unset.grad_layers, unset.b_lr_ratio),
            (None, None),
            "a runner job must not report qwen defaults it never applied"
        );
        let with_knobs = Hyperparameters {
            rank: None,
            alpha: None,
            steps: None,
            n_steps: None,
            lr: None,
            learning_rate: None,
            target: None,
            seed: None,
            grad_layers: Some(2),
            b_lr_ratio: Some(8.0),
        };
        let (status, msg) = resolve_hp(&Some(with_knobs.clone()), TrainerFamily::GemmaTrainRunner)
            .err()
            .expect("train_runner has no windowed depth or LoRA+ path, so accepting the knobs \
                     would train something other than what the job record claims");
        assert_eq!(status, StatusCode::BAD_REQUEST);
        assert!(msg.contains("grad_layers"), "{msg}");
        for one in [
            Hyperparameters {
                b_lr_ratio: None,
                ..with_knobs.clone()
            },
            Hyperparameters {
                grad_layers: None,
                ..with_knobs
            },
        ] {
            assert!(resolve_hp(&Some(one), TrainerFamily::GemmaTrainRunner).is_err());
        }

        let _env = lock_serializing_every_test_that_mutates_the_process_global_env();
        let root = std::env::temp_dir().join(format!("ft-runner-knobs-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        std::env::set_var("NV_FINE_TUNING_DIR", root.join("jobs"));
        let gemma = root.join("gemma-base");
        std::fs::create_dir_all(&gemma).unwrap();
        std::fs::write(
            gemma.join("config.json"),
            r#"{"architectures":["Gemma4ForConditionalGeneration"],"model_type":"gemma4"}"#,
        )
        .unwrap();
        let req = CreateFineTuningJobRequest {
            model: gemma.display().to_string(),
            training_file: None,
            training_data: Some(vec![serde_json::json!({"text": "hello world"})]),
            validation_file: None,
            validation_data: None,
            suffix: None,
            output_dir: None,
            hyperparameters: Some(Hyperparameters {
                rank: None,
                alpha: None,
                steps: None,
                n_steps: None,
                lr: None,
                learning_rate: None,
                target: None,
                seed: None,
                grad_layers: Some(2),
                b_lr_ratio: None,
            }),
        };
        let (status, _) = prepare_job(&req, &None).err().expect("prepare_job must refuse");
        assert_eq!(status, StatusCode::BAD_REQUEST);
        let req = CreateFineTuningJobRequest {
            hyperparameters: None,
            ..req
        };
        let (job, _) = prepare_job(&req, &None).unwrap();
        assert_eq!(job.hyperparameters.grad_layers, None);
        assert_eq!(job.hyperparameters.b_lr_ratio, None);
        assert!(
            !serde_json::to_string(&job.hyperparameters)
                .unwrap()
                .contains("grad_layers"),
            "the stored record omits knobs that never reached the trainer"
        );
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn a_qwen38_summary_reporting_a_non_finite_gradient_surfaces_the_flag_run_job_fails_on() {
        let summary = |finite: bool| qwen38::Q38JsonlSummary {
            losses: vec![4.9, f32::NAN],
            eval_loss_adapter: None,
            eval_loss_base: None,
            eval_ce_adapter: vec![],
            eval_ce_base: vec![],
            modules: vec![],
            trainable_vars: 0,
            num_examples: 1,
            num_eval_examples: 0,
            window_start: 0,
            window_kinds: vec![],
            base_dtype: "BF16".into(),
            resident_base: false,
            all_grads_finite: finite,
            checkpointed: false,
            device: "cuda".into(),
            adapter_path: PathBuf::from("adapter/adapter_model.safetensors"),
            config_path: PathBuf::from("adapter/adapter_config.json"),
        };
        let bad = trained_from_qwen38(summary(false), None);
        assert!(
            !bad.all_grads_finite,
            "run_job's non_finite_gradient gate reads this field; dropping it re-opens the \
             succeeded-with-NaN-adapter hole"
        );
        assert_eq!(bad.metrics["all_grads_finite"], serde_json::json!(false));
        let good = trained_from_qwen38(summary(true), None);
        assert!(good.all_grads_finite);
        assert_eq!(good.metrics["all_grads_finite"], serde_json::json!(true));
        assert!(good.speculator_dir.is_none());

        let co = |finite: bool| qwen38::Q38MtpCoTrainSummary {
            head_losses: vec![3.0, 2.5],
            head_modules: vec!["mtp.fc".into()],
            head_trainable_vars: 2,
            all_head_grads_finite: finite,
            speculator_kind: "qwen38_mtp_head".into(),
            speculator_dir: PathBuf::from("adapter/speculator"),
        };
        let joint = trained_from_qwen38(summary(true), Some(co(false)));
        assert!(
            !joint.all_grads_finite,
            "a NaN head gradient must fail the job exactly like a NaN target gradient, or the \
             shipped pair carries NaN speculator deltas past the gate"
        );
        assert_eq!(
            joint.metrics["speculator"]["all_grads_finite"],
            serde_json::json!(false)
        );
        let joint = trained_from_qwen38(summary(true), Some(co(true)));
        assert!(joint.all_grads_finite);
        assert_eq!(
            joint.speculator_dir,
            Some(PathBuf::from("adapter/speculator"))
        );
        assert_eq!(
            joint.metrics["speculator"]["kind"],
            serde_json::json!("qwen38_mtp_head")
        );
    }

    #[test]
    fn a_runner_summary_reporting_a_non_finite_gradient_surfaces_the_flag_run_job_fails_on() {
        let summary = |finite: bool| train_runner::TrainSummary {
            losses: vec![4.9, f32::NAN],
            eval_loss_adapter: None,
            eval_loss_base: None,
            eval_ce_adapter: vec![],
            eval_ce_base: vec![],
            serving_equiv_maxabs: 0.0,
            trainable_vars: 0,
            num_examples: 1,
            num_eval_examples: 0,
            base_dtype: "BF16".into(),
            resident_base: false,
            nvfp4_base: false,
            dense_base: true,
            layers_built: 1,
            config_layers: 1,
            deterministic: false,
            device: "cuda".into(),
            checkpointed: false,
            all_grads_finite: finite,
            lmhead_chunk: 0,
            modules: vec![],
            adapter_path: PathBuf::from("adapter/adapter_model.safetensors"),
            config_path: PathBuf::from("adapter/adapter_config.json"),
        };
        let bad = trained_from_runner(summary(false), None);
        assert!(
            !bad.all_grads_finite,
            "the gemma arm must fail the job on a NaN gradient exactly like the qwen arm; a \
             hardcoded pass here re-opens the NaN-adapter-hot-load hole on every non-qwen base"
        );
        assert_eq!(bad.metrics["all_grads_finite"], serde_json::json!(false));
        let good = trained_from_runner(summary(true), None);
        assert!(good.all_grads_finite);
        assert_eq!(good.metrics["all_grads_finite"], serde_json::json!(true));
        assert!(good.speculator_dir.is_none());

        let sp = |finite: bool| train_runner::SpeculatorTrainSummary {
            kind: "eagle3".into(),
            losses: vec![5.0, 4.0],
            trainable_vars: 8,
            all_grads_finite: finite,
            modules: vec!["fc".into()],
            adapter_path: PathBuf::from("adapter/speculator/adapter_model.safetensors"),
            config_path: PathBuf::from("adapter/speculator/adapter_config.json"),
        };
        let joint = trained_from_runner(summary(true), Some(sp(false)));
        assert!(
            !joint.all_grads_finite,
            "a NaN eagle3 gradient must fail the job like a NaN target gradient"
        );
        let joint = trained_from_runner(summary(true), Some(sp(true)));
        assert!(joint.all_grads_finite);
        assert_eq!(
            joint.speculator_dir,
            Some(PathBuf::from("adapter/speculator"))
        );
        assert_eq!(
            joint.metrics["speculator"]["kind"],
            serde_json::json!("eagle3")
        );
    }

    #[cfg(feature = "wgpu")]
    #[test]
    fn every_default_target_has_a_serving_lora_site_and_the_only_servable_leaves_outside_the_defaults_are_the_fused_ab_halves(
    ) {
        use nv_models::qwen3_5_dense_wgpu::Q3D_LORA_TARGETS;
        let defaults = default_targets_for(TrainerFamily::Qwen3_5DenseWindowed);
        for t in &defaults {
            assert!(
                Q3D_LORA_TARGETS.iter().any(|s| s.ends_with(t.as_str())),
                "default target {t} has no serving lora site in qwen3_5_dense_wgpu"
            );
        }
        let untrainable: Vec<&&str> = Q3D_LORA_TARGETS
            .iter()
            .filter(|s| !defaults.iter().any(|t| s.ends_with(t.as_str())))
            .collect();
        assert_eq!(
            untrainable,
            ["linear_attn.in_proj_a", "linear_attn.in_proj_b"]
                .iter()
                .collect::<Vec<_>>(),
            "serving covers the fused ab gemv halves but TrainLayer::projections has no attach \
             point for them, so they stay out of the endpoint defaults; any other gap between \
             the tables is a policy drift"
        );
    }

    fn spec(kind: &str) -> SpeculatorSpec {
        SpeculatorSpec {
            kind: kind.into(),
            drafter_dir: None,
            rank: None,
            lr_ratio: None,
        }
    }

    fn laguna_cfg() -> &'static str {
        r#"{"architectures":["LagunaForCausalLM"],"model_type":"laguna",
            "hidden_size":2048,"num_hidden_layers":40}"#
    }

    #[test]
    fn a_laguna_config_routes_to_the_full_graph_trainer_with_the_attention_target_set() {
        assert_eq!(
            trainer_family_for_config(laguna_cfg()),
            TrainerFamily::LagunaFullGraph
        );
        assert_eq!(
            trainer_family_for_config(r#"{"model_type":"laguna"}"#),
            TrainerFamily::LagunaFullGraph,
            "model_type alone must classify, mirroring backend_select's laguna id match"
        );
        assert_eq!(
            trainer_family_for_config(&nv_models::laguna_train::tiny_laguna_config_json()),
            TrainerFamily::LagunaFullGraph,
            "the tiny laguna fixture must route to the trainer that can actually load it"
        );
        assert_eq!(
            default_targets_for(TrainerFamily::LagunaFullGraph),
            nv_models::laguna_train::LagunaTrainArgs::default_targets(),
            "the endpoint default must be exactly the laguna trainer's attachable set"
        );
    }

    #[test]
    fn the_laguna_family_rejects_the_qwen_only_knobs_and_every_speculator_kind() {
        let unset = resolve_hp(&None, TrainerFamily::LagunaFullGraph).unwrap();
        assert_eq!((unset.grad_layers, unset.b_lr_ratio), (None, None));
        let with_knobs = Hyperparameters {
            rank: None,
            alpha: None,
            steps: None,
            n_steps: None,
            lr: None,
            learning_rate: None,
            target: None,
            seed: None,
            grad_layers: Some(2),
            b_lr_ratio: Some(8.0),
        };
        let (status, msg) = resolve_hp(&Some(with_knobs), TrainerFamily::LagunaFullGraph)
            .err()
            .expect("the laguna trainer has no window or LoRA+ path, so accepting the knobs \
                     would train something other than what the job record claims");
        assert_eq!(status, StatusCode::BAD_REQUEST);
        assert!(msg.contains("laguna full-graph"), "{msg}");

        for kind in ["mtp", "eagle3"] {
            let (status, msg) =
                resolve_speculator(&Some(spec(kind)), TrainerFamily::LagunaFullGraph)
                    .err()
                    .expect("laguna has no MTP head and no eagle3 tap path");
            assert_eq!(status, StatusCode::BAD_REQUEST);
            assert!(
                msg.contains(kind) && msg.contains("laguna"),
                "the refusal must name the requested kind and the family: {msg}"
            );
        }
        assert!(resolve_speculator(&None, TrainerFamily::LagunaFullGraph)
            .unwrap()
            .is_none());
    }

    #[test]
    fn a_laguna_base_prepares_the_laguna_trainer_and_a_fixture_job_ships_the_gated_artifact() {
        let _env = lock_serializing_every_test_that_mutates_the_process_global_env();
        let root = std::env::temp_dir().join(format!("ft-laguna-e2e-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        std::env::set_var("NV_FINE_TUNING_DIR", root.join("jobs"));
        std::env::remove_var("NV_TRAIN_DEVICE");
        let base = root.join("base");
        nv_models::laguna_train::write_tiny_laguna_checkpoint(&base);
        let req = CreateFineTuningJobRequest {
            model: base.display().to_string(),
            training_file: None,
            training_data: Some(vec![
                serde_json::json!({
                    "text": "The battle system alternates a command map with an action phase."
                }),
                serde_json::json!({
                    "text": "Critics praised the art direction and the score of the third entry."
                }),
            ]),
            validation_file: None,
            validation_data: Some(vec![
                serde_json::json!({
                    "text": "The remake keeps the overworld theme and rescores the finale."
                }),
                serde_json::json!({"text": "A hidden dungeon opens after the credits roll."}),
            ]),
            suffix: Some("laguna".into()),
            output_dir: None,
            hyperparameters: Some(Hyperparameters {
                rank: Some(4),
                alpha: Some(4.0),
                steps: Some(3),
                n_steps: None,
                lr: Some(5e-3),
                learning_rate: None,
                target: None,
                seed: Some(11),
                grad_layers: None,
                b_lr_ratio: None,
            }),
        };
        let (_, prepared) = prepare_job(&req, &None).unwrap();
        assert!(
            matches!(prepared.train, PreparedTrain::Laguna(_)),
            "a laguna config.json must select the laguna full-graph trainer"
        );
        assert!(!a_speculator_was_requested(&prepared.train));

        let job = create_job(&req).unwrap();
        assert_eq!(
            job.status, "succeeded",
            "the laguna fixture job must train end to end; job error: {:?}",
            job.error
        );
        assert!(!job.speculator_trained);
        let metrics = job.metrics.as_ref().expect("succeeded jobs carry metrics");
        assert_eq!(metrics["trainer"], serde_json::json!("laguna_full_graph"));
        assert_eq!(metrics["all_grads_finite"], serde_json::json!(true));
        for arm in ["eval_loss_adapter", "eval_loss_base"] {
            assert!(
                metrics[arm].as_f64().is_some_and(f64::is_finite),
                "the promotion gate compares {arm}; it was {:?}",
                metrics[arm]
            );
        }
        for arm in ["eval_ce_adapter", "eval_ce_base"] {
            assert_eq!(
                metrics[arm].as_array().map(Vec::len),
                Some(2),
                "{arm} must carry one paired CE per held-out row"
            );
        }
        assert_eq!(
            metrics["modules"].as_array().map(Vec::len),
            Some(4 * 5),
            "four layers times q,k,v,o,g"
        );
        let adapter = PathBuf::from(job.result_files[1].clone());
        assert!(adapter.is_file(), "{adapter:?}");
        assert_eq!(
            modules_whose_lora_b_is_still_exactly_zero(&adapter).unwrap(),
            Vec::<String>::new(),
            "every default laguna target must receive gradient through the full graph"
        );
        let record: serde_json::Value = serde_json::from_slice(
            &std::fs::read(jobs_root().join(&job.id).join("adapter").join(
                JOB_JSON_LEAF_BESIDE_THE_ADAPTER_IS_THE_RECORD_OFFLINE_CONSUMERS_READ_BECAUSE_THE_JOBS_MAP_IS_RAM_ONLY,
            ))
            .unwrap(),
        )
        .unwrap();
        let job_through_the_same_lossy_float_parse: serde_json::Value =
            serde_json::from_str(&serde_json::to_string(&job).unwrap()).unwrap();
        assert_eq!(
            record, job_through_the_same_lossy_float_parse,
            "the terminal on-disk record must be the exact job the API returns"
        );
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn a_speculator_kind_that_mismatches_the_trainer_family_gets_the_same_400_shape_as_the_qwen_only_knobs(
    ) {
        let (status, msg) = resolve_speculator(&Some(spec("mtp")), TrainerFamily::GemmaTrainRunner)
            .err()
            .expect("train_runner bases have no MTP head to co-train");
        assert_eq!(status, StatusCode::BAD_REQUEST);
        assert!(msg.contains("mtp") && msg.contains("train_runner"), "{msg}");
        let (status, msg) =
            resolve_speculator(&Some(spec("eagle3")), TrainerFamily::Qwen3_5DenseWindowed)
                .err()
                .expect("the qwen family self-drafts via its MTP head, not a separate drafter");
        assert_eq!(status, StatusCode::BAD_REQUEST);
        assert!(msg.contains("eagle3") && msg.contains("mtp"), "{msg}");
        let (status, msg) =
            resolve_speculator(&Some(spec("medusa")), TrainerFamily::Qwen3_5DenseWindowed)
                .err()
                .expect("unknown kinds must be refused, not silently dropped");
        assert_eq!(status, StatusCode::BAD_REQUEST);
        assert!(msg.contains("medusa"), "{msg}");
        for family in [
            TrainerFamily::Qwen3_5DenseWindowed,
            TrainerFamily::GemmaTrainRunner,
        ] {
            assert!(
                resolve_speculator(&None, family).unwrap().is_none(),
                "an absent speculator resolves to nothing for every family"
            );
        }
    }

    #[test]
    fn a_valid_speculator_spec_resolves_medusa_2_defaults_and_explicit_values_pass_through() {
        let resolved = resolve_speculator(&Some(spec("mtp")), TrainerFamily::Qwen3_5DenseWindowed)
            .unwrap()
            .expect("mtp on the qwen family is the valid pairing");
        assert_eq!(resolved.kind, SpeculatorKind::MtpSelfDraftingHead);
        assert_eq!(
            resolved.rank,
            SPECULATOR_RANK_DEFAULT_IS_32_FROM_THE_MEDUSA_2_CO_TRAINING_RECIPE
        );
        assert_eq!(
            resolved.lr_ratio,
            SPECULATOR_LR_RATIO_DEFAULT_IS_4_BECAUSE_MEDUSA_2_TRAINS_THE_SPECULATOR_HOTTER_THAN_THE_TARGET
        );
        assert!(resolved.drafter_dir.is_none());

        let root = std::env::temp_dir().join(format!("ft-spec-resolve-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        let drafter = root.join("drafter");
        std::fs::create_dir_all(&drafter).unwrap();
        let explicit = SpeculatorSpec {
            kind: "eagle3".into(),
            drafter_dir: Some(drafter.display().to_string()),
            rank: Some(8),
            lr_ratio: Some(2.0),
        };
        let resolved = resolve_speculator(&Some(explicit.clone()), TrainerFamily::GemmaTrainRunner)
            .unwrap()
            .unwrap();
        assert_eq!(
            (resolved.rank, resolved.lr_ratio, resolved.drafter_dir),
            (8, 2.0, Some(drafter.clone()))
        );

        let resolved = resolve_speculator(
            &Some(SpeculatorSpec {
                rank: Some(0),
                ..explicit.clone()
            }),
            TrainerFamily::GemmaTrainRunner,
        )
        .unwrap()
        .unwrap();
        assert_eq!(
            resolved.rank, 1,
            "a zero rank trains nothing; clamp like hp.rank"
        );

        let (status, msg) = resolve_speculator(
            &Some(SpeculatorSpec {
                lr_ratio: Some(0.0),
                ..explicit
            }),
            TrainerFamily::GemmaTrainRunner,
        )
        .err()
        .expect("a zero lr_ratio freezes the speculator while the record claims co-training");
        assert_eq!(status, StatusCode::BAD_REQUEST);
        assert!(msg.contains("lr_ratio"), "{msg}");

        let (status, msg) = resolve_speculator(
            &Some(SpeculatorSpec {
                drafter_dir: Some(drafter.display().to_string()),
                ..spec("mtp")
            }),
            TrainerFamily::Qwen3_5DenseWindowed,
        )
        .err()
        .expect("the mtp head loads from the base checkpoint, so a drafter_dir would be ignored");
        assert_eq!(status, StatusCode::BAD_REQUEST);
        assert!(msg.contains("model_mtp.safetensors"), "{msg}");
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn eagle3_needs_drafter_weights_from_the_request_or_the_serving_env_var() {
        let _env = lock_serializing_every_test_that_mutates_the_process_global_env();
        let _unset = VarRemovedOnDropSoAFailedAssertCannotLeakItIntoLaterTests(
            EAGLE3_DRAFTER_DIR_FALLBACK_IS_THE_SAME_ENV_VAR_SERVING_READS,
        );
        std::env::remove_var(EAGLE3_DRAFTER_DIR_FALLBACK_IS_THE_SAME_ENV_VAR_SERVING_READS);
        let (status, msg) = resolve_speculator(&Some(spec("eagle3")), TrainerFamily::GemmaTrainRunner)
            .err()
            .expect("no drafter_dir and no env var leaves eagle3 without weights to co-train");
        assert_eq!(status, StatusCode::BAD_REQUEST);
        assert!(
            msg.contains(EAGLE3_DRAFTER_DIR_FALLBACK_IS_THE_SAME_ENV_VAR_SERVING_READS),
            "{msg}"
        );

        let root = std::env::temp_dir().join(format!("ft-spec-eagle3-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        let drafter = root.join("drafter");
        std::fs::create_dir_all(&drafter).unwrap();

        let missing = root.join("missing");
        let (status, msg) = resolve_speculator(
            &Some(SpeculatorSpec {
                drafter_dir: Some(missing.display().to_string()),
                ..spec("eagle3")
            }),
            TrainerFamily::GemmaTrainRunner,
        )
        .err()
        .expect("a drafter_dir that is not a directory must 400, mirroring the base model check");
        assert_eq!(status, StatusCode::BAD_REQUEST);
        assert!(msg.contains("not a directory"), "{msg}");

        let resolved = resolve_speculator(
            &Some(SpeculatorSpec {
                drafter_dir: Some(drafter.display().to_string()),
                ..spec("eagle3")
            }),
            TrainerFamily::GemmaTrainRunner,
        )
        .unwrap()
        .unwrap();
        assert_eq!(resolved.kind, SpeculatorKind::Eagle3SeparateDrafter);
        assert_eq!(resolved.drafter_dir, Some(drafter.clone()));

        std::env::set_var(
            EAGLE3_DRAFTER_DIR_FALLBACK_IS_THE_SAME_ENV_VAR_SERVING_READS,
            &missing,
        );
        let (status, msg) = resolve_speculator(&Some(spec("eagle3")), TrainerFamily::GemmaTrainRunner)
            .err()
            .expect(
                "an env var the operator already set must not be silently discarded into the \
                 generic set-the-env-var 400",
            );
        assert_eq!(status, StatusCode::BAD_REQUEST);
        assert!(
            msg.contains(EAGLE3_DRAFTER_DIR_FALLBACK_IS_THE_SAME_ENV_VAR_SERVING_READS)
                && msg.contains("is set to")
                && msg.contains("not a directory"),
            "the refusal must say the set env var itself is the broken path, mirroring \
             chat_engine's spec-decode-disabled log for the same condition: {msg}"
        );

        std::env::set_var(
            EAGLE3_DRAFTER_DIR_FALLBACK_IS_THE_SAME_ENV_VAR_SERVING_READS,
            &drafter,
        );
        let resolved = resolve_speculator(&Some(spec("eagle3")), TrainerFamily::GemmaTrainRunner)
            .unwrap()
            .unwrap();
        assert_eq!(resolved.drafter_dir, Some(drafter));
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn a_valid_speculator_request_threads_co_train_args_into_the_prepared_trainer_call() {
        let _env = lock_serializing_every_test_that_mutates_the_process_global_env();
        let root = std::env::temp_dir().join(format!("ft-spec-thread-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        std::env::set_var("NV_FINE_TUNING_DIR", root.join("jobs"));
        let qwen = root.join("qwen-base");
        std::fs::create_dir_all(&qwen).unwrap();
        std::fs::write(qwen.join("config.json"), qwen38_dense_cfg()).unwrap();
        let req = CreateFineTuningJobRequest {
            model: qwen.display().to_string(),
            training_file: None,
            training_data: Some(vec![serde_json::json!({"text": "hello world"})]),
            validation_file: None,
            validation_data: None,
            suffix: None,
            output_dir: None,
            hyperparameters: None,
        };
        let (_, prepared) = prepare_job(&req, &Some(spec("mtp"))).unwrap();
        assert!(
            a_speculator_was_requested(&prepared.train),
            "run_job's dropped-knob guard keys off this predicate, so it must see the request"
        );
        match prepared.train {
            PreparedTrain::Qwen38(_, Some(co)) => {
                assert_eq!(
                    (co.rank, co.alpha, co.speculator_lr_ratio),
                    (
                        SPECULATOR_RANK_DEFAULT_IS_32_FROM_THE_MEDUSA_2_CO_TRAINING_RECIPE,
                        SPECULATOR_RANK_DEFAULT_IS_32_FROM_THE_MEDUSA_2_CO_TRAINING_RECIPE as f64,
                        SPECULATOR_LR_RATIO_DEFAULT_IS_4_BECAUSE_MEDUSA_2_TRAINS_THE_SPECULATOR_HOTTER_THAN_THE_TARGET
                    ),
                    "the endpoint defaults must reach run_jsonl_with_mtp_cotrain unchanged"
                );
            }
            _ => panic!("an mtp speculator on a qwen base must prepare the co-train variant"),
        }

        let gemma = root.join("gemma-base");
        std::fs::create_dir_all(&gemma).unwrap();
        std::fs::write(
            gemma.join("config.json"),
            r#"{"architectures":["Gemma4ForConditionalGeneration"],"model_type":"gemma4"}"#,
        )
        .unwrap();
        let drafter = root.join("drafter");
        std::fs::create_dir_all(&drafter).unwrap();
        let req = CreateFineTuningJobRequest {
            model: gemma.display().to_string(),
            ..req
        };
        let (_, prepared) = prepare_job(
            &req,
            &Some(SpeculatorSpec {
                drafter_dir: Some(drafter.display().to_string()),
                rank: Some(16),
                lr_ratio: Some(3.0),
                ..spec("eagle3")
            }),
        )
        .unwrap();
        assert!(
            a_speculator_was_requested(&prepared.train),
            "run_job's dropped-knob guard keys off this predicate, so it must see the request"
        );
        match prepared.train {
            PreparedTrain::Runner(_, Some(sp)) => {
                assert_eq!(
                    (sp.drafter_dir, sp.rank, sp.lr_ratio),
                    (drafter, 16, 3.0),
                    "explicit speculator knobs must reach run_with_speculator unchanged"
                );
            }
            _ => panic!("an eagle3 speculator on a gemma base must prepare the co-train variant"),
        }

        let (status, _) = prepare_job(&req, &Some(spec("mtp")))
            .err()
            .expect("family validation still rejects mtp on a gemma base");
        assert_eq!(status, StatusCode::BAD_REQUEST);
        assert_eq!(
            SPECULATOR_SUBDIR_INSIDE_OUT_DIR_RIDES_THE_SAME_STAMP_SYMLINK_AND_HOT_LOAD_AS_THE_TARGET_ADAPTER,
            nv_train::SPECULATOR_SUBDIR,
            "the untrained-module gate labels speculator modules with this prefix, so it must \
             name the same subdir the trainers' save path writes"
        );
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn an_absent_speculator_keeps_the_request_parse_and_the_job_record_byte_identical() {
        let _env = lock_serializing_every_test_that_mutates_the_process_global_env();
        let body: CreateFineTuningJobBody = serde_json::from_str(
            r#"{"model":"m","training_data":[{"text":"x"}],
                "speculator":{"kind":"mtp","rank":16,"lr_ratio":2.5}}"#,
        )
        .unwrap();
        assert_eq!(body.job.model, "m");
        let parsed = body.speculator.expect("speculator field parses");
        assert_eq!(
            (parsed.kind.as_str(), parsed.rank, parsed.lr_ratio),
            ("mtp", Some(16), Some(2.5))
        );
        let body: CreateFineTuningJobBody =
            serde_json::from_str(r#"{"model":"m","training_data":[{"text":"x"}]}"#).unwrap();
        assert!(body.speculator.is_none());
        assert_eq!(body.job.model, "m");

        let root = std::env::temp_dir().join(format!("ft-spec-absent-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        std::env::set_var("NV_FINE_TUNING_DIR", root.join("jobs"));
        let qwen = root.join("qwen-base");
        std::fs::create_dir_all(&qwen).unwrap();
        std::fs::write(qwen.join("config.json"), qwen38_dense_cfg()).unwrap();
        let req = CreateFineTuningJobRequest {
            model: qwen.display().to_string(),
            training_file: None,
            training_data: Some(vec![serde_json::json!({"text": "hello world"})]),
            validation_file: None,
            validation_data: None,
            suffix: None,
            output_dir: None,
            hyperparameters: None,
        };
        let (mut job, prepared) = prepare_job(&req, &None).unwrap();
        assert!(
            matches!(prepared.train, PreparedTrain::Qwen38(_, None)),
            "no speculator in the request means the plain run_jsonl path, byte-identical to today"
        );
        assert!(
            !a_speculator_was_requested(&prepared.train),
            "the dropped-knob guard must stay silent on the plain path or every job fails"
        );
        let json = serde_json::to_string(&job).unwrap();
        assert!(
            !json.contains("speculator"),
            "a job without a speculator request must serialize exactly as before: {json}"
        );
        job.speculator_trained = true;
        assert!(
            serde_json::to_string(&job)
                .unwrap()
                .contains("\"speculator_trained\":true"),
            "the flag appears once a co-trained speculator artifact ships"
        );
        let _ = std::fs::remove_dir_all(&root);
    }

    fn write_a_tiny_mtp_head_shaped_like_nv_trains_test_fixture(
        dir: &std::path::Path,
        cfg: &qwen38::Q38Config,
    ) {
        let h = cfg.hidden_size;
        let hd = cfg.head_dim;
        let q_rows = if cfg.attn_output_gate {
            cfg.num_attention_heads * hd * 2
        } else {
            cfg.num_attention_heads * hd
        };
        let kv_rows = cfg.num_key_value_heads * hd;
        let mut rng = nv_train::qwen38_fixture::FixtureLcg(0x717E_5EED);
        let mut t = HashMap::new();
        for name in qwen38::mtp_expected_tensor_names() {
            let shape: Vec<usize> = if name == "mtp.fc.weight" {
                vec![h, 2 * h]
            } else if name.ends_with("q_proj.weight") {
                vec![q_rows, h]
            } else if name.ends_with("k_proj.weight") || name.ends_with("v_proj.weight") {
                vec![kv_rows, h]
            } else if name.ends_with("o_proj.weight") {
                vec![h, cfg.num_attention_heads * hd]
            } else if name.ends_with("q_norm.weight") || name.ends_with("k_norm.weight") {
                vec![hd]
            } else if name.ends_with("gate_proj.weight") || name.ends_with("up_proj.weight") {
                vec![cfg.intermediate_size, h]
            } else if name.ends_with("down_proj.weight") {
                vec![h, cfg.intermediate_size]
            } else {
                vec![h]
            };
            t.insert(
                name,
                rng.tensor(&shape, 0.06)
                    .to_dtype(candle_core::DType::BF16)
                    .unwrap(),
            );
        }
        candle_core::safetensors::save(&t, dir.join(qwen38::MTP_WEIGHTS_FILE_NAME)).unwrap();
    }

    #[test]
    fn a_requested_mtp_speculator_reaches_the_cotrain_entry_point_and_the_job_ships_the_pair() {
        let _env = lock_serializing_every_test_that_mutates_the_process_global_env();
        let root = std::env::temp_dir().join(format!("ft-spec-e2e-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        std::env::set_var("NV_FINE_TUNING_DIR", root.join("jobs"));
        std::env::remove_var("NV_TRAIN_DEVICE");
        let base = root.join("base");
        let cfg = nv_train::qwen38_fixture::write_tiny_checkpoint(&base);
        write_a_tiny_mtp_head_shaped_like_nv_trains_test_fixture(&base, &cfg);
        let req = CreateFineTuningJobRequest {
            model: base.display().to_string(),
            training_file: None,
            training_data: Some(vec![
                serde_json::json!({
                    "text": "The battle system alternates a command map with an action phase."
                }),
                serde_json::json!({
                    "text": "Critics praised the art direction and the score of the third entry."
                }),
            ]),
            validation_file: None,
            validation_data: Some(vec![
                serde_json::json!({
                    "text": "The remake keeps the overworld theme and rescores the finale."
                }),
                serde_json::json!({
                    "text": "A hidden dungeon opens after the credits roll."
                }),
            ]),
            suffix: Some("cospec".into()),
            output_dir: None,
            hyperparameters: Some(Hyperparameters {
                rank: Some(4),
                alpha: Some(4.0),
                steps: Some(4),
                n_steps: None,
                lr: Some(5e-3),
                learning_rate: None,
                target: None,
                seed: Some(19),
                grad_layers: None,
                b_lr_ratio: None,
            }),
        };
        let job = create_job_with_speculator(
            &req,
            &Some(SpeculatorSpec {
                kind: "mtp".into(),
                drafter_dir: None,
                rank: Some(4),
                lr_ratio: None,
            }),
        )
        .unwrap();
        assert_eq!(
            job.status, "succeeded",
            "a dispatch that drops the co-train args now fails via the dropped-knob guard \
             instead of succeeding silently; job error: {:?}",
            job.error
        );
        assert!(
            job.speculator_trained,
            "the Some-arm of run_training_on_a_wide_stack must execute \
             run_jsonl_with_mtp_cotrain, not plain run_jsonl"
        );
        let spec_dir = PathBuf::from(
            job.result_files
                .last()
                .expect("a co-trained job appends the speculator dir to result_files"),
        );
        assert!(
            spec_dir.ends_with(
                SPECULATOR_SUBDIR_INSIDE_OUT_DIR_RIDES_THE_SAME_STAMP_SYMLINK_AND_HOT_LOAD_AS_THE_TARGET_ADAPTER
            ),
            "{spec_dir:?}"
        );
        assert!(
            spec_dir.join("adapter_config.json").is_file()
                && spec_dir.join("adapter_model.safetensors").is_file(),
            "the speculator subdir must carry a peft-style pair so the artifact hot-swaps \
             atomically with the target adapter"
        );
        let metrics = job.metrics.as_ref().expect("succeeded jobs carry metrics");
        assert_eq!(
            metrics["speculator"]["kind"],
            serde_json::json!(qwen38::QWEN38_MTP_SPECULATOR_KIND)
        );
        assert_eq!(
            metrics["speculator"]["all_grads_finite"],
            serde_json::json!(true)
        );
        let out_dir = jobs_root().join(&job.id).join("adapter");
        let record_path = out_dir.join(
            JOB_JSON_LEAF_BESIDE_THE_ADAPTER_IS_THE_RECORD_OFFLINE_CONSUMERS_READ_BECAUSE_THE_JOBS_MAP_IS_RAM_ONLY,
        );
        let disk: serde_json::Value =
            serde_json::from_slice(&std::fs::read(&record_path).unwrap()).unwrap();
        assert_eq!(
            disk,
            serde_json::to_value(&job).unwrap(),
            "the terminal on-disk record must be the exact job the API returns, so the gate \
             script and post-restart forensics read the same truth"
        );
        for arm in ["eval_loss_adapter", "eval_loss_base"] {
            assert!(
                disk["metrics"][arm]
                    .as_f64()
                    .is_some_and(f64::is_finite),
                "the terminal record must carry the held-out eval metrics the promotion gate \
                 compares; {arm} was {:?}",
                disk["metrics"][arm]
            );
        }
        for arm in ["eval_ce_adapter", "eval_ce_base"] {
            let rows = disk["metrics"][arm]
                .as_array()
                .unwrap_or_else(|| panic!("{arm} must be a per-example array in the record"));
            assert_eq!(
                rows.len(),
                disk["metrics"]["num_eval_examples"].as_u64().unwrap() as usize,
                "{arm} must carry one paired CE per held-out row for the offline verdict"
            );
            assert!(
                rows.iter().all(|v| v.as_f64().is_some_and(f64::is_finite)),
                "{arm} rows must be finite: {rows:?}"
            );
        }
        assert!(
            !out_dir
                .join(JOB_JSON_TMP_LEAF_NEVER_SURVIVES_A_TRANSITION_BECAUSE_RENAME_INTO_PLACE_IS_WHAT_MAKES_A_MID_WRITE_KILL_LEAVE_THE_LAST_CONSISTENT_RECORD)
                .exists(),
            "no transition may leave its tmp file behind"
        );
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn the_persist_helpers_replace_job_json_atomically_and_a_torn_tmp_from_a_mid_write_kill_is_never_promoted(
    ) {
        let root = std::env::temp_dir().join(format!("ft-persist-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        let out_dir = root.join("out");
        let job = FineTuningJob {
            object: "fine_tuning.job",
            id: "ftjob-persisttest".into(),
            created_at: 7,
            finished_at: None,
            model: "m".into(),
            fine_tuned_model: None,
            status: "queued".into(),
            training_file: None,
            hyperparameters: JobHyperparameters {
                rank: 4,
                alpha: 4.0,
                steps: 2,
                learning_rate: 0.05,
                target: vec!["q_proj".into()],
                seed: 0,
                grad_layers: None,
                b_lr_ratio: None,
            },
            result_files: Vec::new(),
            trained_tokens: None,
            error: None,
            metrics: None,
            speculator_trained: false,
        };
        store(&job);
        let record_path = out_dir.join(
            JOB_JSON_LEAF_BESIDE_THE_ADAPTER_IS_THE_RECORD_OFFLINE_CONSUMERS_READ_BECAUSE_THE_JOBS_MAP_IS_RAM_ONLY,
        );
        let tmp_path = out_dir.join(
            JOB_JSON_TMP_LEAF_NEVER_SURVIVES_A_TRANSITION_BECAUSE_RENAME_INTO_PLACE_IS_WHAT_MAKES_A_MID_WRITE_KILL_LEAVE_THE_LAST_CONSISTENT_RECORD,
        );

        set_status(&job.id, "running");
        persist_job_record_into_out_dir_for_the_gate_script_and_post_restart_forensics(
            &out_dir, &job.id,
        );
        let running: serde_json::Value =
            serde_json::from_slice(&std::fs::read(&record_path).unwrap()).unwrap();
        assert_eq!(running["status"], "running");
        assert!(
            !tmp_path.exists(),
            "rename must consume the tmp file, so a kill between transitions can only ever \
             expose the previous complete record, never a torn one"
        );

        std::fs::write(&tmp_path, b"a kill mid-write leaves a torn tmp behind").unwrap();
        update_job_then_persist_beside_the_adapter(&job.id, &out_dir, |j| {
            j.status = "succeeded".into();
            j.finished_at = Some(9);
            j.metrics = Some(serde_json::json!({
                "eval_loss_adapter": 1.25,
                "eval_loss_base": 1.5,
                "per_example_eval_ce_adapter": [1.0, 1.5],
            }));
        });
        let done: serde_json::Value =
            serde_json::from_slice(&std::fs::read(&record_path).unwrap()).unwrap();
        assert_eq!(done["status"], "succeeded");
        assert_eq!(done["metrics"]["eval_loss_adapter"], serde_json::json!(1.25));
        assert_eq!(
            done["metrics"]["per_example_eval_ce_adapter"],
            serde_json::json!([1.0, 1.5]),
            "per-example arrays in metrics must land on disk verbatim for the offline gate"
        );
        assert!(
            !tmp_path.exists(),
            "the torn tmp a simulated kill left behind must be overwritten and renamed away, \
             never promoted as-is"
        );
        assert_eq!(
            serde_json::to_value(get_job(&job.id).unwrap()).unwrap(),
            done,
            "the on-disk record and the jobs API must agree after every transition"
        );
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn run_job_lands_the_running_record_before_training_and_every_outcome_reaches_disk_through_the_one_terminal_transition(
    ) {
        let _env = lock_serializing_every_test_that_mutates_the_process_global_env();
        let root = std::env::temp_dir().join(format!("ft-runjob-wiring-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        std::env::set_var("NV_FINE_TUNING_DIR", root.join("jobs"));
        std::env::remove_var("NV_TRAIN_DEVICE");
        let base = root.join("base");
        std::fs::create_dir_all(&base).unwrap();
        std::fs::write(base.join("config.json"), qwen38_dense_cfg()).unwrap();
        let req = CreateFineTuningJobRequest {
            model: base.display().to_string(),
            training_file: None,
            training_data: Some(vec![serde_json::json!({"text": "hello world"})]),
            validation_file: None,
            validation_data: None,
            suffix: None,
            output_dir: None,
            hyperparameters: None,
        };
        let (job, prepared) = prepare_job(&req, &None).unwrap();
        let out_dir = prepared.out_dir.clone();
        let record_path = out_dir.join(
            JOB_JSON_LEAF_BESIDE_THE_ADAPTER_IS_THE_RECORD_OFFLINE_CONSUMERS_READ_BECAUSE_THE_JOBS_MAP_IS_RAM_ONLY,
        );
        let config_turned_fifo_so_the_trainer_blocks_while_the_running_record_is_inspected =
            base.join("config.json");
        std::fs::remove_file(
            &config_turned_fifo_so_the_trainer_blocks_while_the_running_record_is_inspected,
        )
        .unwrap();
        assert!(
            std::process::Command::new("mkfifo")
                .arg(&config_turned_fifo_so_the_trainer_blocks_while_the_running_record_is_inspected)
                .status()
                .expect("mkfifo runs")
                .success(),
            "without the fifo the trainer fails instantly and the pre-training running record \
             is unobservable"
        );
        let worker = std::thread::spawn(move || run_job(prepared));
        let mut waited_ms = 0u64;
        while !record_path.is_file() {
            assert!(
                waited_ms < 30_000,
                "run_job must persist the running record after set_status and before training; \
                 the trainer is still blocked on the config fifo, so only that persist call can \
                 produce {}",
                record_path.display()
            );
            std::thread::sleep(std::time::Duration::from_millis(25));
            waited_ms += 25;
        }
        let running: serde_json::Value =
            serde_json::from_slice(&std::fs::read(&record_path).unwrap()).unwrap();
        assert_eq!(
            running["status"],
            "running",
            "the record on disk while training is in flight must say so, or a crash mid-round \
             reads as a job that never started"
        );
        assert_eq!(running["id"], serde_json::json!(job.id));
        std::fs::write(
            &config_turned_fifo_so_the_trainer_blocks_while_the_running_record_is_inspected,
            b"not json, so the released trainer fails and run_job reaches its terminal transition",
        )
        .unwrap();
        worker.join().expect("run_job returns instead of panicking on a failed training run");
        let failed: serde_json::Value =
            serde_json::from_slice(&std::fs::read(&record_path).unwrap()).unwrap();
        assert_eq!(
            failed["status"],
            "failed",
            "a failed round's on-disk record is the post-restart-forensics case this file \
             exists for"
        );
        assert_eq!(failed["error"]["code"], serde_json::json!("training_failed"));
        assert!(failed["finished_at"].is_u64(), "{failed}");
        assert_eq!(
            serde_json::to_value(get_job(&job.id).unwrap()).unwrap(),
            failed,
            "the terminal on-disk record and the jobs API must agree"
        );
        assert!(
            !out_dir
                .join(JOB_JSON_TMP_LEAF_NEVER_SURVIVES_A_TRANSITION_BECAUSE_RENAME_INTO_PLACE_IS_WHAT_MAKES_A_MID_WRITE_KILL_LEAVE_THE_LAST_CONSISTENT_RECORD)
                .exists(),
            "no transition may leave its tmp file behind"
        );
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn a_job_success_writes_no_boot_default_and_promote_is_what_rolls_the_symlink() {
        let _env = lock_serializing_every_test_that_mutates_the_process_global_env();
        let root = std::env::temp_dir().join(format!("ft-promote-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        std::env::set_var("NV_FINE_TUNING_DIR", root.join("jobs"));
        let current = jobs_root().join(
            super::super::lora::CURRENT_LINKS_DIR_HOLDS_ONE_SYMLINK_PER_MODEL_SUFFIX_SO_BOOT_LOADS_THE_ROLLING_ADAPTER_NOT_EVERY_PAST_JOB,
        );
        assert!(
            !current.exists(),
            "a fresh jobs root must carry no boot defaults before any promote"
        );

        let adapter_dir = root.join("job1").join("adapter");
        std::fs::create_dir_all(&adapter_dir).unwrap();
        super::super::lora::register_adapter(super::super::lora::AdapterEntry {
            id: "ft:e4b:promotetest:aaaa1111".into(),
            dir: adapter_dir.clone(),
            base_model: None,
            rank: 4,
            alpha: 8.0,
            scaling: 2.0,
            targets: vec!["q_proj".into()],
        });
        let link = super::super::lora_runtime::promote_to_boot_default("ft:e4b:promotetest:aaaa1111")
            .expect("a registered adapter promotes");
        assert_eq!(
            std::fs::read_link(&link).unwrap(),
            std::path::absolute(&adapter_dir).unwrap(),
            "promote must point the boot default at the gated adapter dir"
        );
        std::fs::remove_dir_all(&root).unwrap();
    }

    #[test]
    fn promoting_an_unknown_adapter_refuses_naming_it() {
        let _env = lock_serializing_every_test_that_mutates_the_process_global_env();
        let err = super::super::lora_runtime::promote_to_boot_default("ft:no:such:adapter")
            .unwrap_err()
            .to_string();
        assert!(err.contains("ft:no:such:adapter"), "refusal must name the adapter: {err}");
    }
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
