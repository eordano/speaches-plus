use std::sync::Arc;

use axum::extract::{Multipart, Path, State};
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::Json;
use nv_tts::{
    VoiceKind, VoiceMeta, VoiceProfileStore, VoiceStoreHandle, ADAPTER_FILE, META_FILE, PACK_FILE,
};
use serde::Serialize;
use tracing::warn;

use super::voice_profiles::is_safe_name;
use super::{kind, openai_error};
use crate::voices::{stages, Endpoints, Workspace};

#[derive(Clone)]
pub struct VoicesAppState {
    pub store: Arc<VoiceStoreHandle>,

    pub kokoro: Option<crate::tts::KokoroHandle>,
    pub embedding: Option<Arc<crate::diarization::embedding::EmbeddingModel>>,

    pub workspace_root: std::path::PathBuf,
    pub self_base: String,
    pub http: reqwest::Client,
}

impl VoicesAppState {
    fn store(&self) -> Result<Arc<VoiceProfileStore>, Response> {
        self.store.get().map_err(|err| {
            warn!(error = %err, "voice store unavailable");
            openai_error(
                StatusCode::SERVICE_UNAVAILABLE,
                format!("{err:#}"),
                kind::SERVICE_UNAVAIL,
                None,
                Some("voice_store_unavailable"),
            )
        })
    }

    pub fn workspace(&self) -> Workspace {
        Workspace::new(self.workspace_root.clone())
    }

    pub fn endpoints(&self) -> Endpoints {
        let env = |k: &str| std::env::var(k).ok().filter(|v| !v.is_empty());
        Endpoints {
            stt: Some(
                env("NV_VOICES_STT_URL")
                    .unwrap_or(format!("{}/v1/audio/transcriptions", self.self_base)),
            ),
            diar: Some(
                env("NV_VOICES_DIAR_URL")
                    .unwrap_or(format!("{}/v1/audio/diarization", self.self_base)),
            ),
            chunk_minutes: env("NV_VOICES_CHUNK_MINUTES")
                .and_then(|v| v.parse().ok())
                .unwrap_or(25.0),
        }
    }
}

pub const SAMPLE_FILES: [&str; 4] = [
    "sample_0.wav",
    "sample_1.wav",
    "sample_2.wav",
    "sample_3.wav",
];

#[derive(Serialize)]
pub struct VoiceResponse {
    pub object: &'static str,
    pub name: String,
    pub kind: VoiceKind,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub speaker_id: Option<String>,
    pub embedding_dim: usize,
    pub files: Vec<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub producer: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub eval: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub created: Option<String>,
}

fn respond(store: &VoiceProfileStore, meta: &VoiceMeta) -> VoiceResponse {
    let mut files = Vec::new();
    for f in [META_FILE, PACK_FILE, ADAPTER_FILE]
        .iter()
        .chain(SAMPLE_FILES.iter())
    {
        if store.file_path(&meta.name, f).is_file() {
            files.push(f.to_string());
        }
    }
    VoiceResponse {
        object: "voice",
        name: meta.name.clone(),
        kind: meta.kind,
        speaker_id: meta.speaker_id.clone(),
        embedding_dim: meta.embedding.len(),
        files,
        producer: meta.producer.clone(),
        eval: meta.eval.clone(),
        created: meta.created.clone(),
    }
}

#[derive(Serialize)]
pub struct ListVoicesResponse {
    pub object: &'static str,
    pub data: Vec<VoiceResponse>,
}

pub async fn handle_list(State(state): State<VoicesAppState>) -> Response {
    let store = match state.store() {
        Ok(s) => s,
        Err(r) => return r,
    };
    match store.list_meta() {
        Ok(metas) => Json(ListVoicesResponse {
            object: "list",
            data: metas.iter().map(|m| respond(&store, m)).collect(),
        })
        .into_response(),
        Err(err) => {
            warn!(error = %err, "voices list failed");
            openai_error(
                StatusCode::INTERNAL_SERVER_ERROR,
                "voice store read failed",
                kind::SERVER,
                None,
                Some("store_read_error"),
            )
        }
    }
}

pub async fn handle_get(State(state): State<VoicesAppState>, Path(name): Path<String>) -> Response {
    if !is_safe_name(&name) {
        return bad_name(&name);
    }
    let store = match state.store() {
        Ok(s) => s,
        Err(r) => return r,
    };
    match store.get_meta(&name) {
        Ok(meta) => Json(respond(&store, &meta)).into_response(),
        Err(_) => openai_error(
            StatusCode::NOT_FOUND,
            format!("voice {name:?} not found"),
            kind::INVALID_REQUEST,
            Some("name"),
            Some("voice_not_found"),
        ),
    }
}

pub async fn handle_delete(
    State(state): State<VoicesAppState>,
    Path(name): Path<String>,
) -> Response {
    if !is_safe_name(&name) {
        return bad_name(&name);
    }
    let store = match state.store() {
        Ok(s) => s,
        Err(r) => return r,
    };
    if !store.exists(&name) {
        return openai_error(
            StatusCode::NOT_FOUND,
            format!("voice {name:?} not found"),
            kind::INVALID_REQUEST,
            Some("name"),
            Some("voice_not_found"),
        );
    }
    match store.delete(&name) {
        Ok(()) => Json(serde_json::json!({"object": "voice", "name": name, "deleted": true}))
            .into_response(),
        Err(err) => {
            warn!(error = %err, name = %name, "voice delete failed");
            openai_error(
                StatusCode::INTERNAL_SERVER_ERROR,
                "voice delete failed",
                kind::SERVER,
                None,
                Some("store_write_error"),
            )
        }
    }
}

pub async fn handle_file(
    State(state): State<VoicesAppState>,
    Path((name, file)): Path<(String, String)>,
) -> Response {
    if !is_safe_name(&name) {
        return bad_name(&name);
    }
    if ![META_FILE, PACK_FILE, ADAPTER_FILE].contains(&file.as_str())
        && !SAMPLE_FILES.contains(&file.as_str())
    {
        return openai_error(
            StatusCode::BAD_REQUEST,
            format!("unknown voice file {file:?}"),
            kind::INVALID_REQUEST,
            Some("file"),
            Some("invalid_voice_file"),
        );
    }
    let store = match state.store() {
        Ok(s) => s,
        Err(r) => return r,
    };
    match store.read_file(&name, &file) {
        Ok(bytes) => {
            let ct = if file.ends_with(".json") {
                "application/json"
            } else if file.ends_with(".wav") {
                "audio/wav"
            } else {
                "application/octet-stream"
            };
            ([(axum::http::header::CONTENT_TYPE, ct)], bytes).into_response()
        }
        Err(_) => openai_error(
            StatusCode::NOT_FOUND,
            format!("voice {name:?} has no {file}"),
            kind::INVALID_REQUEST,
            Some("file"),
            Some("voice_file_not_found"),
        ),
    }
}

pub async fn handle_import(
    State(state): State<VoicesAppState>,
    mut multipart: Multipart,
) -> Response {
    let mut meta: Option<VoiceMeta> = None;
    let mut files: Vec<(String, Vec<u8>)> = Vec::new();
    while let Ok(Some(field)) = multipart.next_field().await {
        let fname = field
            .file_name()
            .map(|s| s.to_string())
            .unwrap_or_else(|| field.name().unwrap_or("").to_string());
        let bytes = match field.bytes().await {
            Ok(b) => b.to_vec(),
            Err(err) => {
                return openai_error(
                    StatusCode::BAD_REQUEST,
                    format!("multipart read: {err}"),
                    kind::INVALID_REQUEST,
                    Some(fname.as_str()),
                    Some("multipart_read_error"),
                )
            }
        };
        match fname.as_str() {
            META_FILE | "meta" => match serde_json::from_slice::<VoiceMeta>(&bytes) {
                Ok(m) => meta = Some(m),
                Err(err) => {
                    return openai_error(
                        StatusCode::BAD_REQUEST,
                        format!("{META_FILE}: {err}"),
                        kind::INVALID_REQUEST,
                        Some(META_FILE),
                        Some("invalid_voice_meta"),
                    )
                }
            },
            PACK_FILE | ADAPTER_FILE => files.push((fname, bytes)),
            other => {
                return openai_error(
                    StatusCode::BAD_REQUEST,
                    format!("unexpected part {other:?}"),
                    kind::INVALID_REQUEST,
                    Some(other),
                    Some("invalid_voice_file"),
                )
            }
        }
    }
    let Some(mut meta) = meta else {
        return openai_error(
            StatusCode::BAD_REQUEST,
            format!("{META_FILE} part is required"),
            kind::INVALID_REQUEST,
            Some(META_FILE),
            Some("missing_voice_meta"),
        );
    };
    if !is_safe_name(&meta.name) {
        return bad_name(&meta.name);
    }
    if meta.kind.is_kokoro() && !files.iter().any(|(f, _)| f == PACK_FILE) {
        return openai_error(
            StatusCode::BAD_REQUEST,
            format!("{:?} voices need a {PACK_FILE} part", meta.kind),
            kind::INVALID_REQUEST,
            Some(PACK_FILE),
            Some("missing_voice_pack"),
        );
    }
    if meta.kind == VoiceKind::KokoroAdapter && !files.iter().any(|(f, _)| f == ADAPTER_FILE) {
        return openai_error(
            StatusCode::BAD_REQUEST,
            format!("kokoro-adapter voices need an {ADAPTER_FILE} part"),
            kind::INVALID_REQUEST,
            Some(ADAPTER_FILE),
            Some("missing_voice_adapter"),
        );
    }
    meta.schema_version = nv_tts::voice_profile::SCHEMA_VERSION;
    let store = match state.store() {
        Ok(s) => s,
        Err(r) => return r,
    };
    for (f, bytes) in &files {
        if let Err(err) = store.write_file(&meta.name, f, bytes) {
            warn!(error = %err, name = %meta.name, file = %f, "voice file write failed");
            return openai_error(
                StatusCode::INTERNAL_SERVER_ERROR,
                "voice store write failed",
                kind::SERVER,
                None,
                Some("store_write_error"),
            );
        }
    }
    if let Err(err) = store.put_meta(&meta) {
        warn!(error = %err, name = %meta.name, "voice meta write failed");
        return openai_error(
            StatusCode::INTERNAL_SERVER_ERROR,
            "voice store write failed",
            kind::SERVER,
            None,
            Some("store_write_error"),
        );
    }
    (StatusCode::CREATED, Json(respond(&store, &meta))).into_response()
}

fn bad_name(name: &str) -> Response {
    openai_error(
        StatusCode::BAD_REQUEST,
        format!("invalid voice name {name:?}"),
        kind::INVALID_REQUEST,
        Some("name"),
        Some("invalid_voice_name"),
    )
}

use std::collections::HashMap;
use std::sync::{LazyLock, Mutex};

use axum::response::sse::{Event, KeepAlive, Sse};
use futures_util::stream::{self, StreamExt};
use tokio::sync::broadcast;

use crate::tts::fit::{FitEvent, FitRequest, Fitter};

pub struct FitJob {
    pub id: String,
    pub kind: &'static str,
    pub status: Mutex<&'static str>,
    pub events: Mutex<Vec<serde_json::Value>>,
    pub tx: broadcast::Sender<serde_json::Value>,
    pub request: serde_json::Value,
}

static JOBS: LazyLock<Mutex<HashMap<String, Arc<FitJob>>>> =
    LazyLock::new(|| Mutex::new(HashMap::new()));

static FIT_GATE: Mutex<()> = Mutex::new(());

fn job_public(job: &FitJob) -> serde_json::Value {
    serde_json::json!({
        "id": job.id,
        "kind": job.kind,
        "status": *job.status.lock().unwrap_or_else(std::sync::PoisonError::into_inner),
        "request": job.request,
        "n_events": job.events.lock().unwrap_or_else(std::sync::PoisonError::into_inner).len(),
        "events": format!("/v1/voices/fit/{}/events", job.id),
    })
}

fn push(job: &FitJob, v: serde_json::Value) {
    job.events
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .push(v.clone());
    let _ = job.tx.send(v);
}

fn new_job(kind: &'static str, request: serde_json::Value) -> Arc<FitJob> {
    let id = format!(
        "{:x}",
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0)
            & 0xffff_ffff_ffff
    );
    let (tx, _rx) = broadcast::channel(256);
    let job = Arc::new(FitJob {
        id: id.clone(),
        kind,
        status: Mutex::new("queued"),
        events: Mutex::new(Vec::new()),
        tx,
        request,
    });
    JOBS.lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .insert(id, job.clone());
    job
}

fn set_status(job: &FitJob, st: &'static str) {
    *job.status
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner) = st;
}

async fn resolve_speaker(
    state: &VoicesAppState,
    req: &mut FitRequest,
    speaker: &str,
) -> anyhow::Result<()> {
    let ws = state.workspace();
    let m = ws.load()?;
    let reg = ws.load_speakers()?;
    let clips = stages::speaker_clips(&ws, &m, &reg, speaker, 8.0)?;
    if clips.is_empty() {
        anyhow::bail!(
            "speaker {speaker:?} has no clean clips of 8 s or more; run the pipeline first"
        );
    }
    req.refs = stages::clip_wavs(&ws, &m, &clips, 12).await?;
    if req.sentences.is_empty() {
        let texts: Vec<String> = clips.iter().map(|c| c.text.clone()).collect();
        req.sentences = stages::pick_sentences(&texts, 4);
    }
    req.speaker_id = reg.find(speaker).map(|s| s.id.clone());
    if let Some(n) = req.steps {
        if req.tier == "adapter" {
            req.epochs = n;
        } else {
            req.budget = n;
        }
    }
    Ok(())
}

pub async fn handle_fit_start(
    State(state): State<VoicesAppState>,
    axum::Json(mut req): axum::Json<FitRequest>,
) -> Response {
    if !is_safe_name(&req.name) {
        return bad_name(&req.name);
    }
    if let Some(speaker) = req.speaker.clone() {
        if let Err(err) = resolve_speaker(&state, &mut req, &speaker).await {
            return openai_error(
                StatusCode::BAD_REQUEST,
                format!("{err:#}"),
                kind::INVALID_REQUEST,
                Some("speaker"),
                Some("speaker_unfit"),
            );
        }
    }
    let (Some(kokoro), Some(embedding)) = (state.kokoro.clone(), state.embedding.clone()) else {
        return openai_error(
            StatusCode::SERVICE_UNAVAILABLE,
            "voice fitting needs the Kokoro model and the speaker embedding model loaded",
            kind::SERVER,
            None,
            Some("fit_models_not_loaded"),
        );
    };
    let store = match state.store() {
        Ok(s) => s,
        Err(r) => return r,
    };
    let job = new_job("fit", req_public(&req));
    let fitter = Fitter {
        kokoro,
        embedding,
        store,
        root: state.workspace_root.clone(),
    };
    let job2 = job.clone();
    crate::oapi::cold_tier::ensure_free_async(
        crate::oapi::cold_tier::fit_min_free_bytes(),
        "voice fit",
    )
    .await;
    tokio::task::spawn_blocking(move || {
        let _gate = FIT_GATE
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        *job2
            .status
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner) = "running";
        push(
            &job2,
            serde_json::json!({"stage": "queue", "message": "gpu acquired"}),
        );
        let res = fitter.run(&req, |ev: FitEvent| {
            if let Ok(v) = serde_json::to_value(&ev) {
                push(&job2, v);
            }
        });
        match res {
            Ok(meta) => {
                *job2
                    .status
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner) = "done";
                push(
                    &job2,
                    serde_json::json!({"stage": "done", "voice": meta.name, "eval": meta.eval}),
                );
            }
            Err(err) => {
                warn!(error = %err, job = %job2.id, "voice fit failed");
                *job2
                    .status
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner) = "failed";
                push(
                    &job2,
                    serde_json::json!({"stage": "error", "error": format!("{err:#}")}),
                );
            }
        }
    });
    (StatusCode::ACCEPTED, Json(job_public(&job))).into_response()
}

fn req_public(req: &FitRequest) -> serde_json::Value {
    serde_json::json!({
        "name": req.name, "speaker_id": req.speaker_id, "refs": req.refs.len(), "sentences": req.sentences.len(),
        "budget": req.budget, "patience": req.patience, "population": req.population, "tier": req.tier, "epochs": req.epochs,
    })
}

static RUN_GATE: LazyLock<tokio::sync::Mutex<()>> = LazyLock::new(|| tokio::sync::Mutex::new(()));

pub async fn handle_ws_run(State(state): State<VoicesAppState>) -> Response {
    let ep = state.endpoints();
    let job = new_job(
        "run",
        serde_json::json!({"stt_url": ep.stt, "diar_url": ep.diar, "chunk_minutes": ep.chunk_minutes}),
    );
    let job2 = job.clone();
    tokio::spawn(async move {
        let _gate = RUN_GATE.lock().await;
        set_status(&job2, "running");
        push(
            &job2,
            serde_json::json!({"stage": "queue", "message": "worker acquired"}),
        );
        let ws = state.workspace();
        let mut log = |m: String| push(&job2, serde_json::json!({"stage": "run", "message": m}));
        match stages::auto(&ws, &state.http, &ep, &mut log).await {
            Ok(v) => {
                set_status(&job2, "done");
                push(&job2, serde_json::json!({"stage": "done", "result": v}));
            }
            Err(err) => {
                warn!(error = %err, job = %job2.id, "voices pipeline run failed");
                set_status(&job2, "failed");
                push(
                    &job2,
                    serde_json::json!({"stage": "error", "error": format!("{err:#}")}),
                );
            }
        }
    });
    (StatusCode::ACCEPTED, Json(job_public(&job))).into_response()
}

fn ws_error(err: anyhow::Error, code: &'static str) -> Response {
    warn!(error = %err, "voices workspace request failed");
    openai_error(
        StatusCode::INTERNAL_SERVER_ERROR,
        format!("{err:#}"),
        kind::SERVER,
        None,
        Some(code),
    )
}

fn is_sha(s: &str) -> bool {
    s.len() == 64 && s.bytes().all(|b| b.is_ascii_hexdigit())
}

pub async fn handle_ws_status(State(state): State<VoicesAppState>) -> Response {
    let ws = state.workspace();
    match tokio::task::spawn_blocking(move || ws.load().and_then(|m| ws.status(&m))).await {
        Ok(Ok(v)) => Json(v).into_response(),
        Ok(Err(e)) => ws_error(e, "workspace_read_error"),
        Err(e) => ws_error(e.into(), "workspace_read_error"),
    }
}

pub async fn handle_ws_config(State(state): State<VoicesAppState>) -> Response {
    let ep = state.endpoints();
    Json(serde_json::json!({
        "workspace": state.workspace_root, "stt_url": ep.stt, "diar_url": ep.diar, "chunk_minutes": ep.chunk_minutes,
        "candle": state.kokoro.as_ref().map(|k| k.candle().is_some()).unwrap_or(false),
    }))
    .into_response()
}

fn json_file(path: std::path::PathBuf) -> Response {
    match std::fs::read(&path) {
        Ok(bytes) => (
            [(axum::http::header::CONTENT_TYPE, "application/json")],
            bytes,
        )
            .into_response(),
        Err(_) => openai_error(
            StatusCode::NOT_FOUND,
            format!("{} not found", path.display()),
            kind::INVALID_REQUEST,
            None,
            Some("not_found"),
        ),
    }
}

pub async fn handle_ws_manifest(State(state): State<VoicesAppState>) -> Response {
    json_file(state.workspace().manifest_path())
}

pub async fn handle_ws_speakers(State(state): State<VoicesAppState>) -> Response {
    let ws = state.workspace();
    match ws.load_speakers() {
        Ok(reg) => Json(reg).into_response(),
        Err(e) => ws_error(e, "workspace_read_error"),
    }
}

#[derive(serde::Deserialize)]
pub struct SpeakerName {
    pub name: String,
}

pub async fn handle_ws_name_speaker(
    State(state): State<VoicesAppState>,
    Path(id): Path<String>,
    Json(body): Json<SpeakerName>,
) -> Response {
    let ws = state.workspace();
    let result = tokio::task::spawn_blocking(move || ws.rename_speaker(&id, &body.name)).await;
    match result {
        Ok(Ok(Some(speaker))) => Json(speaker).into_response(),
        Ok(Ok(None)) => openai_error(
            StatusCode::NOT_FOUND,
            "speaker not found",
            kind::INVALID_REQUEST,
            Some("id"),
            Some("speaker_not_found"),
        ),
        Ok(Err(e)) => ws_error(e, "workspace_write_error"),
        Err(e) => ws_error(e.into(), "workspace_write_error"),
    }
}

pub async fn handle_ws_identify_audio(
    State(state): State<VoicesAppState>,
    mut multipart: Multipart,
) -> Response {
    let mut audio: Option<(Vec<u8>, Option<String>)> = None;
    while let Ok(Some(field)) = multipart.next_field().await {
        if matches!(field.name(), Some("file") | Some("audio")) {
            let ct = field.content_type().map(|s| s.to_string());
            if let Ok(bytes) = field.bytes().await {
                audio = Some((bytes.to_vec(), ct));
            }
        } else {
            let _ = field.bytes().await;
        }
    }
    let Some((bytes, ct)) = audio else {
        return super::fastapi_validation_error(vec![super::missing_field(&["body", "file"])]);
    };
    let Some(embedding) = state.embedding.clone() else {
        return openai_error(
            StatusCode::SERVICE_UNAVAILABLE,
            "speaker identification needs the WeSpeaker embedding model loaded",
            kind::SERVICE_UNAVAIL,
            None,
            Some("no_speaker_embedding_model"),
        );
    };
    let identifier = crate::realtime::speaker_id::SpeakerIdentifier::new(
        state.workspace_root.clone(),
        embedding,
    );
    let res = tokio::task::spawn_blocking(move || {
        let samples = crate::audio::decode_any_to_16k_mono(&bytes, ct.as_deref())?;
        let seconds = samples.len() as f64 / 16_000.0;
        let m = identifier.identify(&samples)?;
        anyhow::Ok((seconds, m))
    })
    .await;
    match res {
        Ok(Ok((seconds, speaker))) => Json(serde_json::json!({
            "seconds": seconds,
            "speaker": speaker,
        }))
        .into_response(),
        Ok(Err(e)) => ws_error(e, "identify_audio_failed"),
        Err(e) => ws_error(e.into(), "identify_audio_failed"),
    }
}

pub async fn handle_ws_identify(State(state): State<VoicesAppState>) -> Response {
    let ws = state.workspace();
    let res = tokio::task::spawn_blocking(move || {
        let m = ws.load()?;
        stages::identify(&ws, &m, 60.0, &mut |_| {})
    })
    .await;
    match res {
        Ok(Ok(reg)) => Json(serde_json::json!({
            "speakers": reg.speakers.len(),
            "hosts": reg.speakers.iter().filter(|s| s.role.as_deref() == Some("host")).count(),
        }))
        .into_response(),
        Ok(Err(e)) => ws_error(e, "identify_failed"),
        Err(e) => ws_error(e.into(), "identify_failed"),
    }
}

pub async fn handle_ws_artifact(
    State(state): State<VoicesAppState>,
    Path((sha, kind_)): Path<(String, String)>,
) -> Response {
    if !is_sha(&sha) || !crate::voices::workspace::KINDS.contains(&kind_.as_str()) {
        return openai_error(
            StatusCode::BAD_REQUEST,
            "bad artifact reference",
            kind::INVALID_REQUEST,
            None,
            Some("bad_artifact_ref"),
        );
    }
    json_file(state.workspace().artifact_path(&sha, &kind_))
}

pub async fn handle_ws_clip(
    State(state): State<VoicesAppState>,
    Path((sha, index)): Path<(String, u64)>,
) -> Response {
    if !is_sha(&sha) {
        return openai_error(
            StatusCode::BAD_REQUEST,
            "bad recording reference",
            kind::INVALID_REQUEST,
            None,
            Some("bad_recording_ref"),
        );
    }
    let ws = state.workspace();
    let res: anyhow::Result<Vec<u8>> = async {
        let m = ws.load()?;
        let r = m
            .rows
            .get(&sha)
            .ok_or_else(|| anyhow::anyhow!("recording not found"))?;
        let clips = ws.get_artifact(&sha, "clips")?;
        let c = clips["clips"]
            .as_array()
            .into_iter()
            .flatten()
            .find(|c| c["i"].as_u64() == Some(index))
            .ok_or_else(|| anyhow::anyhow!("clip not found"))?
            .clone();
        let td = stages::TempDir::new("clip")?;
        let out = td.0.join("clip.wav");
        let dur = c["dur"].as_f64().unwrap_or(0.0).min(30.0);
        stages::to_wav(
            &ws.recording_path(r),
            &out,
            c["start"].as_f64(),
            Some(dur),
            24_000,
        )
        .await?;
        Ok(std::fs::read(&out)?)
    }
    .await;
    match res {
        Ok(bytes) => ([(axum::http::header::CONTENT_TYPE, "audio/wav")], bytes).into_response(),
        Err(e) => openai_error(
            StatusCode::NOT_FOUND,
            format!("{e:#}"),
            kind::INVALID_REQUEST,
            None,
            Some("clip_unavailable"),
        ),
    }
}

pub async fn handle_page() -> Response {
    (
        [(axum::http::header::CONTENT_TYPE, "text/html; charset=utf-8")],
        include_str!("../../static/voice-lab.html"),
    )
        .into_response()
}

pub async fn handle_fit_list() -> Response {
    let jobs = JOBS
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let mut data: Vec<serde_json::Value> = jobs.values().map(|j| job_public(j)).collect();
    data.sort_by(|a, b| a["id"].as_str().cmp(&b["id"].as_str()));
    Json(serde_json::json!({"object": "list", "data": data})).into_response()
}

pub async fn handle_fit_get(Path(id): Path<String>) -> Response {
    let job = JOBS
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .get(&id)
        .cloned();
    match job {
        Some(j) => Json(job_public(&j)).into_response(),
        None => openai_error(
            StatusCode::NOT_FOUND,
            "fit job not found",
            kind::INVALID_REQUEST,
            Some("id"),
            Some("job_not_found"),
        ),
    }
}

pub async fn handle_fit_events(Path(id): Path<String>) -> Response {
    let job = JOBS
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .get(&id)
        .cloned();
    let Some(job) = job else {
        return openai_error(
            StatusCode::NOT_FOUND,
            "fit job not found",
            kind::INVALID_REQUEST,
            Some("id"),
            Some("job_not_found"),
        );
    };

    Sse::new(
        job_events(job)
            .map(|v| Ok::<Event, std::convert::Infallible>(Event::default().data(v.to_string()))),
    )
    .keep_alive(KeepAlive::new().interval(std::time::Duration::from_secs(10)))
    .into_response()
}

fn job_events(job: Arc<FitJob>) -> impl futures_util::Stream<Item = serde_json::Value> {
    let rx = job.tx.subscribe();
    stream::unfold(
        (job, 0usize, rx, false),
        |(job, mut cursor, mut rx, ended)| async move {
            if ended {
                return None;
            }
            loop {
                let event = job
                    .events
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner)
                    .get(cursor)
                    .cloned();
                if let Some(v) = event {
                    cursor += 1;
                    let terminal = matches!(v["stage"].as_str(), Some("done") | Some("error"));
                    return Some((v, (job, cursor, rx, terminal)));
                }
                match rx.recv().await {
                    Ok(_) | Err(broadcast::error::RecvError::Lagged(_)) => {}
                    Err(broadcast::error::RecvError::Closed) => return None,
                }
            }
        },
    )
}

#[cfg(test)]
mod event_tests {
    use super::*;

    #[tokio::test]
    async fn voices_events_replay_live_and_terminal_are_delivered_once() {
        let job = new_job("fit", serde_json::json!({}));
        push(&job, serde_json::json!({"stage": "start"}));
        let events = job_events(job.clone());
        futures_util::pin_mut!(events);
        assert_eq!(events.next().await.unwrap()["stage"], "start");
        for i in 0..300 {
            push(&job, serde_json::json!({"stage": "train", "step": i}));
        }
        push(&job, serde_json::json!({"stage": "done"}));
        for i in 0..300 {
            assert_eq!(events.next().await.unwrap()["step"], i);
        }
        assert_eq!(events.next().await.unwrap()["stage"], "done");
        assert!(events.next().await.is_none());
        let replay: Vec<_> = job_events(job).collect().await;
        assert_eq!(replay.len(), 302);
        assert_eq!(replay.last().unwrap()["stage"], "done");
    }
}
