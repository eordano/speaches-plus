use std::net::SocketAddr;
use std::sync::{Arc, OnceLock};

use anyhow::Context;
use axum::{
    extract::{DefaultBodyLimit, FromRequest, Multipart, Query, Request, State},
    http::{header, HeaderMap, HeaderValue, Method, StatusCode},
    middleware::{self, Next},
    response::{IntoResponse, Response},
    routing::{get, post},
    Router,
};
use clap::Parser;
use std::fmt::Write as _;
use std::sync::atomic::{AtomicI64, AtomicU64, Ordering::Relaxed};
use tracing::{info, warn};

use speaches_plus::audio::decode_any_to_16k_mono;
use speaches_plus::{
    diarization, inspect, models, oapi, otel, pii, realtime, AppState, RealtimeQuery,
};

#[derive(Debug, Parser)]
#[command(name = "speaches-plus", version)]
struct Args {
    #[arg(long, env = "UVICORN_HOST", default_value = "127.0.0.1")]
    host: String,

    #[arg(long, env = "UVICORN_PORT", default_value_t = 8000)]
    port: u16,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    use tracing_subscriber::layer::SubscriberExt;
    use tracing_subscriber::util::SubscriberInitExt;
    let env_filter = tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| {
        tracing_subscriber::EnvFilter::new(
            "info,speaches_plus=debug,ort=warn,ort::logging=warn,hyper=warn,h2=warn,reqwest=warn",
        )
    });
    let fmt_layer = tracing_subscriber::fmt::layer();
    let registry = tracing_subscriber::registry()
        .with(env_filter)
        .with(fmt_layer);
    let otel_layer = match otel::try_install_layer() {
        Ok(l) => l,
        Err(err) => {
            eprintln!("OTel exporter init failed: {err:#}; continuing without spans");
            None
        }
    };
    if let Some(layer) = otel_layer {
        registry.with(layer).init();
    } else {
        registry.init();
    }

    if std::env::var_os("ORT_DYLIB_PATH").is_none() {
        if let Ok(home) = std::env::var("HOME") {
            let candidate = format!("{home}/.nix-profile/lib/libonnxruntime.dylib");
            if std::path::Path::new(&candidate).exists() {
                std::env::set_var("ORT_DYLIB_PATH", &candidate);
            }
        }
    }
    match std::env::var_os("ORT_DYLIB_PATH") {
        Some(p) => {
            let path = std::path::PathBuf::from(&p);
            if !path.exists() {
                anyhow::bail!(
                    "ORT_DYLIB_PATH={} does not exist -- point it at a libonnxruntime dylib \
                     (launch from the speaches-plus dev shell, or unset it only if a system \
                     onnxruntime is on the loader path)",
                    path.display()
                );
            }
            info!(path = %path.display(), "ORT_DYLIB_PATH resolved");
        }
        None => {
            warn!(
                "ORT_DYLIB_PATH is not set and no dylib was found at ~/.nix-profile/lib/\
                 libonnxruntime.dylib; ort will dlopen libonnxruntime from system paths -- \
                 if model load stalls or aborts, set ORT_DYLIB_PATH (e.g. launch from the \
                 speaches-plus dev shell)"
            );
        }
    }

    let args = Args::parse();
    let addr: SocketAddr = format!("{}:{}", args.host, args.port)
        .parse()
        .context("parsing listen addr")?;

    audio_eou_boot_gate()?;

    let models = models::Models::get_or_init().context("load models")?;
    #[cfg(feature = "wgpu")]
    let chat_registry = oapi::chat_engine_wgpu::registry_from_env_with_wgpu();
    #[cfg(not(feature = "wgpu"))]
    let chat_registry = oapi::chat_engine::registry_from_env();
    let chat_engine = chat_registry.as_ref().map(|r| r.default_engine());
    if let Some(eng) = chat_engine.as_ref() {
        warm_chat_engine(eng.clone()).await;
    }
    let speaker_encoder = load_speaker_encoder_from_env();
    let tts_talker = oapi::audio_speech_nvtts::Qwen3TtsAudioSpeech::from_env()
        .map(|s| s as Arc<dyn oapi::audio_speech::AudioSpeech + Send + Sync>);
    let state = AppState {
        models: models.clone(),
        chat_engine: chat_engine.clone(),
        chat_registry: chat_registry.clone(),
        tts_talker,
        speaker_encoder: speaker_encoder.clone(),
    };

    let _ = CHAT_MODEL_IDS.set(
        chat_registry
            .as_ref()
            .map(|r| r.model_ids().to_vec())
            .unwrap_or_default(),
    );

    let chat_router = chat_registry.clone().map(|registry| {
        match process_wired_gib() {
            Some(gib) => info!(
                models = ?registry.model_ids(),
                wired_gib = format!("{gib:.1}"),
                "chat models loaded"
            ),
            None => info!(models = ?registry.model_ids(), "chat models loaded"),
        }
        let chat_state = oapi::chat::ChatAppState { registry };
        Router::new()
            .route(
                "/v1/chat/completions",
                post(oapi::chat::handle_chat_completions),
            )
            .route(
                "/v1/completions",
                post(oapi::completions::handle_completions),
            )
            .route("/v1/messages", post(oapi::messages::handle_messages))
            .route(
                "/v1/messages/count_tokens",
                post(oapi::messages::handle_count_tokens),
            )
            .route("/v1/responses", post(oapi::responses::handle_responses))
            .route(
                "/v1/responses/{id}",
                get(oapi::responses::handle_get_response)
                    .delete(oapi::responses::handle_delete_response),
            )
            .with_state(chat_state)
    });
    if chat_router.is_none() {
        info!("NV_CHAT_MODEL_DIR(S) not set -- /v1/chat/completions and /v1/completions disabled");
    }

    let audio_speech_router = Router::new()
        .route("/v1/audio/speech", post(oapi::audio_speech::handle))
        .with_state(state.clone());

    let voice_profile_root = voice_profile_root_dir();
    let voice_profiles_router = match nv_tts::VoiceProfileStore::open(&voice_profile_root) {
        Ok(store) => {
            let vp_state = oapi::voice_profiles::VoiceProfilesAppState::new(store)
                .with_encoder(speaker_encoder.clone());
            Some(
                Router::new()
                    .route(
                        "/v1/voice-profiles",
                        post(oapi::voice_profiles::handle_create)
                            .get(oapi::voice_profiles::handle_list),
                    )
                    .route(
                        "/v1/voice-profiles/{name}",
                        get(oapi::voice_profiles::handle_get)
                            .delete(oapi::voice_profiles::handle_delete),
                    )
                    .with_state(vp_state),
            )
        }
        Err(err) => {
            warn!(
                error = %err,
                path = %voice_profile_root.display(),
                "voice-profiles store open failed; endpoint disabled",
            );
            None
        }
    };

    let pii_router = load_pii_classifier_from_env();
    let ocr_router = oapi::ocr::router_from_env();
    let _ = oapi::text_embeddings::warm_embedder_at_boot();

    let _ = READINESS.set(build_readiness(
        &state,
        chat_router.is_some(),
        voice_profiles_router.is_some(),
        pii_router.is_some(),
    ));
    if api_key().is_none() && cors().origin == "*" {
        warn!(
            "SPEACHES_API_KEY is unset and SPEACHES_CORS_ORIGIN defaults to \"*\": every route \
             is unauthenticated and callable from any web origin. Set SPEACHES_API_KEY, or pin \
             SPEACHES_CORS_ORIGIN, or bind to 127.0.0.1 behind a reverse proxy."
        );
    }

    let app = Router::new()
        .route("/health", get(health))
        .route("/health/ready", get(readiness))
        .route("/version", get(version_handler))
        .route("/metrics", get(metrics_handler))
        .route("/health/sessions", get(sessions_health))
        .route("/v1/internal/chat-engines", get(chat_engines_handler))
        .route(
            oapi::backend_select::REALTIME_CAPABILITIES_WITH_BACKENDS_ROUTE,
            get(oapi::backend_select::realtime_capabilities_with_backends),
        )
        .merge(oapi::backend_select::backends_report_router())
        .route(
            "/v1/realtime",
            post(realtime_post).get(realtime::websocket::realtime_ws),
        )
        .route("/v1/realtime/capabilities", get(realtime_capabilities))
        .route("/v1/audio/transcriptions", post(transcriptions_post))
        .route(
            "/v1/audio/diarization",
            post(diarization::http::diarization_post),
        )
        .route("/v1/audio/embeddings", post(audio_embeddings_dispatch))
        .route("/v1/embeddings", post(text_embeddings_handler))
        .route("/v1/models", get(oapi::models_handler::handle_list_models))
        .route(
            "/v1/inspect/sessions",
            get(inspect::routes::inspect_sessions),
        )
        .route(
            "/v1/inspect/sessions/history",
            get(inspect::routes::inspect_history),
        )
        .route(
            "/v1/inspect/sessions/history/{sid}",
            get(inspect::routes::inspect_history_stream),
        )
        .route(
            "/v1/inspect/sessions/{sid}/audio",
            get(inspect::routes::inspect_audio),
        )
        .route(
            "/v1/inspect/{sid}/stream",
            get(inspect::routes::inspect_stream_ws),
        )
        .route("/v1/audio/translations", post(translations_post))
        .with_state(state)
        .merge(audio_speech_router);
    let app = if let Some(cr) = chat_router {
        app.merge(cr)
    } else {
        app
    };
    let app = if let Some(vpr) = voice_profiles_router {
        app.merge(vpr)
    } else {
        app
    };
    let app = if let Some(pr) = pii_router {
        app.merge(pr)
    } else {
        app
    };
    let app = app.merge(ocr_router);
    let app = app.merge(oapi::fine_tuning::router());
    let app = app
        .layer(DefaultBodyLimit::max(200 * 1024 * 1024))
        .layer(middleware::from_fn(auth_mw))
        .layer(middleware::from_fn(metrics_mw))
        .layer(middleware::from_fn(cors_mw));

    let listener = tokio::net::TcpListener::bind(addr)
        .await
        .with_context(|| format!("bind {addr}"))?;
    let _ = speaches_plus::oapi::SELF_ADDR.set(addr);
    info!(%addr, "listening");
    let serve_result = axum::serve(listener, app)
        .with_graceful_shutdown(shutdown_signal())
        .await
        .context("axum serve");
    otel::shutdown();
    serve_result?;
    Ok(())
}

async fn health() -> &'static str {
    "ok"
}

fn audio_eou_boot_gate() -> anyhow::Result<()> {
    use speaches_plus::defaults::env as env_names;
    use speaches_plus::eou::{
        audio_eou_gate, audio_eou_missing_message, audio_eou_required, audio_eou_wanted,
        AudioEouGate, EouConfig,
    };

    let cfg = EouConfig::from_env();
    let kind = cfg.kind;
    let wanted = audio_eou_wanted(kind);
    let path = speaches_plus::eou::audio::resolve_audio_eou_paths();
    let present = wanted
        && speaches_plus::eou::audio::try_load_from_env(
            cfg.audio_window_ms,
            cfg.audio_pad_alignment,
        )
        .is_some();
    let required = audio_eou_required(std::env::var(env_names::EOU_AUDIO_REQUIRED).ok().as_deref());
    match audio_eou_gate(wanted, present, required) {
        AudioEouGate::NotWanted => Ok(()),
        AudioEouGate::Present => {
            info!(
                eou_kind = kind.as_str(),
                path = %path.unwrap_or_default(),
                "audio end-of-utterance model configured"
            );
            Ok(())
        }
        AudioEouGate::DegradedWarn => {
            tracing::error!(
                eou_kind = kind.as_str(),
                "{}",
                audio_eou_missing_message(kind, path.as_deref(), true)
            );
            Ok(())
        }
        AudioEouGate::RequiredFail => anyhow::bail!(
            "{}=1: {}",
            env_names::EOU_AUDIO_REQUIRED,
            audio_eou_missing_message(kind, path.as_deref(), false)
        ),
    }
}

static CHAT_MODEL_IDS: OnceLock<Vec<String>> = OnceLock::new();

#[cfg(feature = "wgpu")]
fn wgpu_served_ids() -> Vec<String> {
    oapi::chat_engine_wgpu::registered_wgpu_model_ids()
}

#[cfg(not(feature = "wgpu"))]
fn wgpu_served_ids() -> Vec<String> {
    Vec::new()
}

async fn chat_engines_handler() -> Response {
    let ids = CHAT_MODEL_IDS.get().cloned().unwrap_or_default();
    let wgpu = wgpu_served_ids();
    let engines: Vec<serde_json::Value> = ids
        .iter()
        .map(|id| {
            serde_json::json!({
                "id": id,
                "backend": if wgpu.contains(id) { "wgpu" } else { "cuda" },
            })
        })
        .collect();
    let body = serde_json::json!({
        "chat_models": ids,
        "wgpu_engines": wgpu,
        "engines": engines,
        "wgpu_feature": cfg!(feature = "wgpu"),
        "cuda_feature": cfg!(feature = "cuda"),
    });
    (
        StatusCode::OK,
        [(header::CONTENT_TYPE, "application/json")],
        body.to_string(),
    )
        .into_response()
}

#[cfg(target_os = "macos")]
fn process_wired_gib() -> Option<f64> {
    let mut info = std::mem::MaybeUninit::<libc::rusage_info_v4>::uninit();
    let ret = unsafe {
        libc::proc_pid_rusage(
            std::process::id() as libc::c_int,
            libc::RUSAGE_INFO_V4,
            info.as_mut_ptr() as *mut libc::rusage_info_t,
        )
    };
    if ret != 0 {
        return None;
    }
    let info = unsafe { info.assume_init() };
    Some(info.ri_phys_footprint as f64 / (1u64 << 30) as f64)
}

#[cfg(all(unix, not(target_os = "macos")))]
fn process_wired_gib() -> Option<f64> {
    let mut ru = std::mem::MaybeUninit::<libc::rusage>::uninit();
    let ret = unsafe { libc::getrusage(libc::RUSAGE_SELF, ru.as_mut_ptr()) };
    if ret != 0 {
        return None;
    }
    let ru = unsafe { ru.assume_init() };
    Some(ru.ru_maxrss as f64 * 1024.0 / (1u64 << 30) as f64)
}

#[cfg(not(unix))]
fn process_wired_gib() -> Option<f64> {
    None
}

async fn version_handler() -> Response {
    let body = serde_json::json!({ "version": env!("CARGO_PKG_VERSION") });
    (
        StatusCode::OK,
        [(header::CONTENT_TYPE, "application/json")],
        body.to_string(),
    )
        .into_response()
}

const LAT_BUCKETS_MS: [u64; 11] = [5, 10, 25, 50, 100, 250, 500, 1000, 2500, 5000, u64::MAX];

struct Metrics {
    requests_total: AtomicU64,
    in_flight: AtomicI64,
    resp_2xx: AtomicU64,
    resp_4xx: AtomicU64,
    resp_5xx: AtomicU64,
    latency_le: [AtomicU64; 11],
    latency_sum_ms: AtomicU64,
}

static METRICS: Metrics = Metrics {
    requests_total: AtomicU64::new(0),
    in_flight: AtomicI64::new(0),
    resp_2xx: AtomicU64::new(0),
    resp_4xx: AtomicU64::new(0),
    resp_5xx: AtomicU64::new(0),
    latency_le: [const { AtomicU64::new(0) }; 11],
    latency_sum_ms: AtomicU64::new(0),
};

impl Metrics {
    fn observe(&self, status: StatusCode, ms: u64) {
        let c = status.as_u16();
        if (200..300).contains(&c) {
            self.resp_2xx.fetch_add(1, Relaxed);
        } else if (400..500).contains(&c) {
            self.resp_4xx.fetch_add(1, Relaxed);
        } else if c >= 500 {
            self.resp_5xx.fetch_add(1, Relaxed);
        }
        self.latency_sum_ms.fetch_add(ms, Relaxed);
        for (i, &b) in LAT_BUCKETS_MS.iter().enumerate() {
            if ms <= b {
                self.latency_le[i].fetch_add(1, Relaxed);
            }
        }
    }

    fn render(&self) -> String {
        let total = self.requests_total.load(Relaxed);
        let mut s = String::new();
        let _ = writeln!(s, "# TYPE speaches_requests_total counter");
        let _ = writeln!(s, "speaches_requests_total {total}");
        let _ = writeln!(s, "# TYPE speaches_requests_in_flight gauge");
        let _ = writeln!(
            s,
            "speaches_requests_in_flight {}",
            self.in_flight.load(Relaxed)
        );
        let _ = writeln!(s, "# TYPE speaches_responses_total counter");
        for (name, v) in [
            ("2xx", &self.resp_2xx),
            ("4xx", &self.resp_4xx),
            ("5xx", &self.resp_5xx),
        ] {
            let _ = writeln!(
                s,
                "speaches_responses_total{{class=\"{name}\"}} {}",
                v.load(Relaxed)
            );
        }
        let _ = writeln!(s, "# TYPE speaches_request_latency_ms histogram");
        let mut inf_count = 0u64;
        for (i, &b) in LAT_BUCKETS_MS.iter().enumerate() {
            let le = if b == u64::MAX {
                "+Inf".to_string()
            } else {
                b.to_string()
            };
            let v = self.latency_le[i].load(Relaxed);
            if b == u64::MAX {
                inf_count = v;
            }
            let _ = writeln!(s, "speaches_request_latency_ms_bucket{{le=\"{le}\"}} {v}");
        }
        let _ = writeln!(
            s,
            "speaches_request_latency_ms_sum {}",
            self.latency_sum_ms.load(Relaxed)
        );
        let _ = writeln!(s, "speaches_request_latency_ms_count {inf_count}");
        s
    }
}

struct InFlightGuard;

impl InFlightGuard {
    fn enter() -> Self {
        METRICS.in_flight.fetch_add(1, Relaxed);
        Self
    }
}

impl Drop for InFlightGuard {
    fn drop(&mut self) {
        METRICS.in_flight.fetch_sub(1, Relaxed);
    }
}

struct Subsystem {
    name: &'static str,
    configured: bool,
    live: bool,
    required: bool,
    detail: String,
}

static READINESS: OnceLock<Vec<Subsystem>> = OnceLock::new();

fn readiness_report(subs: &[Subsystem]) -> (StatusCode, serde_json::Value) {
    let down: Vec<&str> = subs
        .iter()
        .filter(|s| s.required && s.configured && !s.live)
        .map(|s| s.name)
        .collect();
    let entries: Vec<serde_json::Value> = subs
        .iter()
        .map(|s| {
            serde_json::json!({
                "name": s.name,
                "configured": s.configured,
                "live": s.live,
                "required": s.required,
                "detail": s.detail,
            })
        })
        .collect();
    let status = if down.is_empty() {
        StatusCode::OK
    } else {
        StatusCode::SERVICE_UNAVAILABLE
    };
    let body = serde_json::json!({
        "status": if down.is_empty() { "ready" } else { "not_ready" },
        "down": down,
        "subsystems": entries,
    });
    (status, body)
}

async fn readiness() -> Response {
    let Some(subs) = READINESS.get() else {
        let body = serde_json::json!({
            "status": "starting",
            "down": ["startup"],
            "subsystems": [],
        });
        return (
            StatusCode::SERVICE_UNAVAILABLE,
            [(header::CONTENT_TYPE, "application/json")],
            body.to_string(),
        )
            .into_response();
    };
    let (status, body) = readiness_report(subs);
    (
        status,
        [(header::CONTENT_TYPE, "application/json")],
        body.to_string(),
    )
        .into_response()
}

fn build_readiness(
    state: &AppState,
    chat_live: bool,
    voice_profiles_live: bool,
    pii_live: bool,
) -> Vec<Subsystem> {
    let model_dir = state.models.model_dir.clone();
    let chat_configured = std::env::var_os("NV_CHAT_MODEL_DIR").is_some()
        || std::env::var_os("NV_CHAT_MODEL_DIRS").is_some();
    let talker_configured = std::env::var_os(oapi::audio_speech_nvtts::ENV_TALKER_DIR).is_some();
    let pii_configured = std::env::var_os("REDACT_MODEL_DIR").is_some();
    let kokoro_configured = model_dir.join("kokoro-v1.0.onnx").exists();
    let diar_seg_configured = model_dir.join("diarizen-segmentation.onnx").exists();
    let diar_emb_configured = model_dir.join("wespeaker-resnet293-LM.onnx").exists();

    vec![
        Subsystem {
            name: "vad",
            configured: true,
            live: state.models.vad().is_ok(),
            required: true,
            detail: match state.models.vad() {
                Ok(_) => "silero_vad.onnx loaded".into(),
                Err(e) => format!("{e:#}"),
            },
        },
        Subsystem {
            name: "stt",
            configured: true,
            live: state.models.whisper_opt().is_some(),
            required: true,
            detail: match state.models.whisper() {
                Ok(w) => format!("whisper backend: {}", w.model_id()),
                Err(e) => format!("{e:#}"),
            },
        },
        Subsystem {
            name: "chat",
            configured: chat_configured,
            live: chat_live,
            required: true,
            detail: "NV_CHAT_MODEL_DIR(S)".into(),
        },
        Subsystem {
            name: "tts_talker",
            configured: talker_configured,
            live: state.tts_talker.is_some(),
            required: true,
            detail: format!(
                "{} -> /v1/audio/speech",
                oapi::audio_speech_nvtts::ENV_TALKER_DIR
            ),
        },
        Subsystem {
            name: "tts_kokoro",
            configured: kokoro_configured,
            live: state.models.kokoro.is_some(),
            required: true,
            detail: "kokoro-v1.0.onnx + voices.bin under the model dir".into(),
        },
        Subsystem {
            name: "speaker_encoder",
            configured: talker_configured,
            live: state.speaker_encoder.is_some(),
            required: false,
            detail: "ECAPA-TDNN from the TTS checkpoint; absent on CustomVoice/VoiceDesign".into(),
        },
        Subsystem {
            name: "voice_profiles",
            configured: true,
            live: voice_profiles_live,
            required: true,
            detail: "on-disk voice-profile store".into(),
        },
        Subsystem {
            name: "pii",
            configured: pii_configured,
            live: pii_live,
            required: true,
            detail: "REDACT_MODEL_DIR ([onnx/]model.onnx + tokenizer.json)".into(),
        },
        Subsystem {
            name: "diarization",
            configured: diar_seg_configured && diar_emb_configured,
            live: state.models.diar_segmentation.is_some() && state.models.diar_embedding.is_some(),
            required: true,
            detail: "diarizen-segmentation.onnx + wespeaker-resnet293-LM.onnx".into(),
        },
    ]
}

async fn metrics_handler() -> impl IntoResponse {
    (
        [(header::CONTENT_TYPE, "text/plain; version=0.0.4")],
        METRICS.render(),
    )
}

async fn metrics_mw(req: Request, next: Next) -> Response {
    METRICS.requests_total.fetch_add(1, Relaxed);
    let _in_flight = InFlightGuard::enter();
    let t = std::time::Instant::now();
    let resp = next.run(req).await;
    METRICS.observe(resp.status(), t.elapsed().as_millis() as u64);
    resp
}

static API_KEY: OnceLock<Option<String>> = OnceLock::new();

fn api_key() -> Option<&'static str> {
    API_KEY
        .get_or_init(|| {
            std::env::var("SPEACHES_API_KEY")
                .ok()
                .filter(|k| !k.is_empty())
        })
        .as_deref()
}

use oapi::constant_time_eq;

async fn auth_mw(req: Request, next: Next) -> Response {
    if let Some(key) = api_key() {
        let path = req.uri().path();
        let exempt = matches!(path, "/health" | "/health/ready" | "/metrics" | "/version");
        if !exempt {
            if !oapi::request_key_ok(req.headers(), key) {
                if path.starts_with("/v1/messages") {
                    return oapi::messages::anthropic_error(
                        StatusCode::UNAUTHORIZED,
                        oapi::messages::akind::AUTH,
                        "missing or invalid API key (send x-api-key or Authorization: Bearer)",
                    );
                }
                return (StatusCode::UNAUTHORIZED, "missing or invalid API key").into_response();
            }
        }
    }
    next.run(req).await
}

struct CorsConfig {
    origin: String,
    header: Option<HeaderValue>,
}

static CORS: OnceLock<CorsConfig> = OnceLock::new();

fn cors() -> &'static CorsConfig {
    CORS.get_or_init(|| {
        let origin = std::env::var("SPEACHES_CORS_ORIGIN").unwrap_or_else(|_| "*".to_string());
        let header = HeaderValue::from_str(&origin).ok();
        if header.is_none() {
            warn!(
                origin,
                "SPEACHES_CORS_ORIGIN is not a valid header value; header omitted"
            );
        }
        CorsConfig { origin, header }
    })
}

async fn cors_mw(req: Request, next: Next) -> Response {
    let preflight = req.method() == Method::OPTIONS;
    let mut resp = if preflight {
        StatusCode::NO_CONTENT.into_response()
    } else {
        next.run(req).await
    };
    let h = resp.headers_mut();
    if let Some(v) = cors().header.clone() {
        h.insert("access-control-allow-origin", v);
    }
    h.insert(
        "access-control-allow-methods",
        HeaderValue::from_static("GET,POST,DELETE,OPTIONS"),
    );
    h.insert(
        "access-control-allow-headers",
        HeaderValue::from_static("authorization,content-type"),
    );
    resp
}

async fn shutdown_signal() {
    let ctrl_c = async {
        let _ = tokio::signal::ctrl_c().await;
    };
    #[cfg(unix)]
    let term = async {
        if let Ok(mut s) = tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())
        {
            s.recv().await;
        }
    };
    #[cfg(not(unix))]
    let term = std::future::pending::<()>();
    tokio::select! { _ = ctrl_c => {}, _ = term => {} }
    info!("shutdown signal received; draining in-flight requests");
}

async fn warm_chat_engine(engine: Arc<dyn oapi::chat::ChatEngine>) {
    if std::env::var("NV_CHAT_WARMUP").ok().as_deref() == Some("0") {
        return;
    }
    let t0 = std::time::Instant::now();
    let req = oapi::chat::ChatGenerateRequest {
        prompt: "Hi".to_string(),
        max_new_tokens: 8,
        stop: Vec::new(),
        seed: Some(0),
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
    let (tx, mut rx) = tokio::sync::mpsc::channel(256);
    let drain = tokio::spawn(async move { while rx.recv().await.is_some() {} });
    let outcome = engine.generate(req, tx).await;
    let _ = drain.await;
    match outcome {
        Ok(()) => {
            let warmed_note = if cfg!(feature = "cuda") {
                "CUDA graphs captured at boot"
            } else {
                "wgpu pipelines warmed at boot"
            };
            info!(
                model = %engine.model_id(),
                warmup_s = t0.elapsed().as_secs_f64(),
                "chat engine warmed ({warmed_note})"
            )
        }
        Err(e) => warn!(
            model = %engine.model_id(),
            error = %e,
            "chat warmup failed (serving continues cold)"
        ),
    }
}

fn load_speaker_encoder_from_env() -> Option<Arc<nv_tts::SpeakerEncoder>> {
    let dir = std::env::var_os(oapi::audio_speech_nvtts::ENV_TALKER_DIR)?;
    let dir = std::path::PathBuf::from(dir);
    if !nv_tts::SpeakerEncoder::checkpoint_has_speaker_encoder(&dir) {
        info!(
            path = %dir.display(),
            "TTS checkpoint has no speaker_encoder tensors (CustomVoice/VoiceDesign); \
             voice-profile enrollment disabled -- use a Qwen3-TTS Base checkpoint to enroll"
        );
        return None;
    }
    match nv_tts::SpeakerEncoder::from_qwen3_checkpoint(&dir, &candle_core::Device::Cpu) {
        Ok(encoder) => {
            info!(
                path = %dir.display(),
                enc_dim = encoder.config().enc_dim,
                "speaker-encoder ready (ECAPA-TDNN from TTS checkpoint)"
            );
            Some(Arc::new(encoder))
        }
        Err(err) => {
            warn!(error = %err, path = %dir.display(), "speaker-encoder open failed");
            None
        }
    }
}

fn voice_profile_root_dir() -> std::path::PathBuf {
    oapi::voice_profiles::voice_profile_root()
}

fn load_pii_classifier_from_env() -> Option<Router> {
    let dir = std::env::var("REDACT_MODEL_DIR").ok()?;
    let path = std::path::PathBuf::from(&dir);
    if !pii::classifier::layout_present(&path) {
        warn!(
            path = %path.display(),
            "REDACT_MODEL_DIR set but no PII model layout found (probed model.onnx and onnx/model.onnx, \
             with tokenizer.json/config.json beside either); PII disabled"
        );
        return None;
    }
    match pii::classifier::PiiClassifier::load(&path) {
        Ok(classifier) => {
            let state = Arc::new(classifier);
            oapi::models_handler::note_pii_loaded(&path);
            info!(path = %path.display(), "PII classifier loaded");
            Some(
                Router::new()
                    .route("/v1/pii/classify", post(pii::handler::classify_post))
                    .route(
                        "/v1/pii/classify/batch",
                        post(pii::handler::classify_batch_post),
                    )
                    .route(
                        "/v1/pii/redact/analyze",
                        post(pii::image_handler::analyze_post),
                    )
                    .route(
                        "/v1/pii/redact/render",
                        post(pii::image_handler::render_post),
                    )
                    .with_state(state),
            )
        }
        Err(err) => {
            warn!(error = %err, path = %path.display(), "PII classifier load failed; endpoint disabled");
            None
        }
    }
}

async fn realtime_capabilities(State(state): State<AppState>) -> Response {
    let body = realtime::capabilities_json_with_models(&state.models);
    (
        StatusCode::OK,
        [(header::CONTENT_TYPE, "application/json")],
        body.to_string(),
    )
        .into_response()
}

async fn sessions_health() -> Response {
    let count = realtime::live_session_count();
    let ws_count = realtime::websocket::active_session_count();
    let body = serde_json::json!({"live_sessions": count, "ws_sessions": ws_count});
    (
        StatusCode::OK,
        [(header::CONTENT_TYPE, "application/json")],
        body.to_string(),
    )
        .into_response()
}

async fn realtime_post(
    Query(q): Query<RealtimeQuery>,
    headers: HeaderMap,
    body: String,
) -> Response {
    let intent = q.intent.as_deref().unwrap_or("transcription");
    let content_type = headers
        .get(header::CONTENT_TYPE)
        .and_then(|v| v.to_str().ok())
        .unwrap_or("");
    if !content_type.starts_with("application/sdp") {
        return oapi::openai_error(
            StatusCode::UNSUPPORTED_MEDIA_TYPE,
            "expected Content-Type: application/sdp",
            oapi::kind::INVALID_REQUEST,
            None,
            Some("unsupported_media_type"),
        );
    }
    info!(
        intent,
        model = q.model.as_deref().unwrap_or("?"),
        transcription_model = q.transcription_model.as_deref().unwrap_or("?"),
        offer_bytes = body.len(),
        "realtime POST received"
    );

    match realtime::handle_offer(&body, &q).await {
        Ok(answer_sdp) => (
            StatusCode::OK,
            [(header::CONTENT_TYPE, "application/sdp")],
            answer_sdp,
        )
            .into_response(),
        Err(err) => {
            tracing::error!(error = ?err, "realtime offer handling failed");
            if is_capacity_error(&err) {
                oapi::openai_error(
                    StatusCode::SERVICE_UNAVAILABLE,
                    realtime::CAPACITY_ERROR,
                    oapi::kind::SERVICE_UNAVAIL,
                    None,
                    Some("session_cap_exceeded"),
                )
            } else if is_client_sdp_error(&err) {
                oapi::openai_error(
                    StatusCode::BAD_REQUEST,
                    format!("{err:#}"),
                    oapi::kind::INVALID_REQUEST,
                    None,
                    Some("sdp_invalid"),
                )
            } else {
                oapi::openai_error(
                    StatusCode::INTERNAL_SERVER_ERROR,
                    "realtime negotiation failed; see server logs",
                    oapi::kind::SERVER,
                    None,
                    Some("negotiate_failed"),
                )
            }
        }
    }
}

fn is_capacity_error(err: &anyhow::Error) -> bool {
    format!("{err:#}").contains(realtime::CAPACITY_ERROR)
}

fn is_client_sdp_error(err: &anyhow::Error) -> bool {
    if let Some(w) = err.downcast_ref::<webrtc::Error>() {
        return matches!(
            w,
            webrtc::Error::Sdp(_)
                | webrtc::Error::ErrSessionDescriptionNoFingerprint
                | webrtc::Error::ErrSessionDescriptionInvalidFingerprint
                | webrtc::Error::ErrSessionDescriptionConflictingFingerprints
                | webrtc::Error::ErrSessionDescriptionMissingIceUfrag
                | webrtc::Error::ErrSessionDescriptionMissingIcePwd
                | webrtc::Error::ErrSessionDescriptionConflictingIceUfrag
                | webrtc::Error::ErrSessionDescriptionConflictingIcePwd
                | webrtc::Error::ErrPeerConnSDPTypeInvalidValue
        );
    }
    is_client_sdp_error_msg(&format!("{err:#}"))
}

fn is_client_sdp_error_msg(msg: &str) -> bool {
    const NEEDLES: &[&str] = &[
        "parse offer SDP",
        "set_remote_description",
        "syntax error",
        "no ice-ufrag",
        "no ice-pwd",
        "no fingerprint",
        "unable to start",
        "SdpInvalidSyntax",
        "SdpEmpty",
    ];
    NEEDLES.iter().any(|n| msg.contains(n))
}

async fn transcriptions_post(State(state): State<AppState>, multipart: Multipart) -> Response {
    do_stt_post(
        state,
        multipart,
        speaches_plus::stt::WhisperTask::Transcribe,
    )
    .await
}

async fn translations_post(State(state): State<AppState>, multipart: Multipart) -> Response {
    do_stt_post(state, multipart, speaches_plus::stt::WhisperTask::Translate).await
}

async fn audio_embeddings_dispatch(
    State(state): State<AppState>,
    headers: HeaderMap,
    request: axum::extract::Request,
) -> Response {
    let ct = headers
        .get(header::CONTENT_TYPE)
        .and_then(|v| v.to_str().ok())
        .unwrap_or("");
    if ct.starts_with("application/json") {
        let body = match axum::body::to_bytes(request.into_body(), 2 * 1024 * 1024).await {
            Ok(b) => b,
            Err(err) => {
                return oapi::openai_error(
                    StatusCode::BAD_REQUEST,
                    format!("body read: {err}"),
                    oapi::kind::INVALID_REQUEST,
                    Some("body"),
                    Some("body_read_error"),
                );
            }
        };
        return oapi::text_embeddings::text_embeddings_post(body).await;
    }
    let multipart = match Multipart::from_request(request, &state).await {
        Ok(m) => m,
        Err(err) => {
            return oapi::openai_error(
                StatusCode::BAD_REQUEST,
                format!("multipart: {err}"),
                oapi::kind::INVALID_REQUEST,
                Some("body"),
                Some("multipart_decode_error"),
            );
        }
    };
    diarization::embeddings_http::audio_embeddings_post(State(state), multipart).await
}

async fn text_embeddings_handler(request: axum::extract::Request) -> Response {
    let body = match axum::body::to_bytes(request.into_body(), 2 * 1024 * 1024).await {
        Ok(b) => b,
        Err(err) => {
            return oapi::openai_error(
                StatusCode::BAD_REQUEST,
                format!("body read: {err}"),
                oapi::kind::INVALID_REQUEST,
                Some("body"),
                Some("body_read_error"),
            );
        }
    };
    oapi::text_embeddings::text_embeddings_post(body).await
}

fn check_stt_params(
    language: Option<&str>,
    prompt: Option<&str>,
    temperature: Option<&str>,
) -> Result<(), (&'static str, String)> {
    if let Some(v) = language {
        let v = v.trim();
        if !v.is_empty() && !v.eq_ignore_ascii_case("auto") {
            return Err((
                "language",
                format!(
                    "language={v:?} is not supported by this build: the STT backend always \
                     auto-detects the spoken language. Omit the field or send \"auto\"."
                ),
            ));
        }
    }
    if let Some(v) = prompt {
        if !v.trim().is_empty() {
            return Err((
                "prompt",
                "prompt is not supported by this build: the STT backend takes no decoder prompt. \
                 Omit the field."
                    .to_string(),
            ));
        }
    }
    if let Some(v) = temperature {
        let v = v.trim();
        if !v.is_empty() {
            match v.parse::<f64>() {
                Ok(0.0) => {}
                Ok(t) => {
                    return Err((
                        "temperature",
                        format!(
                            "temperature={t} is not supported by this build: the STT backend \
                             decodes greedily. Omit the field or send 0."
                        ),
                    ))
                }
                Err(_) => {
                    return Err((
                        "temperature",
                        format!("temperature must be a number, got {v:?}"),
                    ))
                }
            }
        }
    }
    Ok(())
}

fn stt_failure_response(err: &anyhow::Error) -> Response {
    if speaches_plus::stt::is_translate_unsupported(err) {
        warn!(error = %err, "translation requested on a transcribe-only speech model");
        return oapi::openai_error(
            StatusCode::BAD_REQUEST,
            err.to_string(),
            oapi::kind::INVALID_REQUEST,
            Some("model"),
            Some("unsupported_parameter"),
        );
    }
    warn!(error = ?err, "whisper transcribe failed");
    oapi::openai_error(
        StatusCode::INTERNAL_SERVER_ERROR,
        "transcription failed; see server logs",
        oapi::kind::SERVER,
        None,
        Some("transcribe_failed"),
    )
}

async fn do_stt_post(
    state: AppState,
    mut multipart: Multipart,
    task: speaches_plus::stt::WhisperTask,
) -> Response {
    let mut audio_bytes: Option<Vec<u8>> = None;
    let mut audio_content_type: Option<String> = None;
    let mut response_format = "text".to_string();
    let mut language: Option<String> = None;
    let mut prompt: Option<String> = None;
    let mut temperature: Option<String> = None;
    while let Ok(Some(field)) = multipart.next_field().await {
        let name = field.name().unwrap_or("").to_string();
        match name.as_str() {
            "file" => {
                audio_content_type = field.content_type().map(|s| s.to_string());
                match field.bytes().await {
                    Ok(b) => audio_bytes = Some(b.to_vec()),
                    Err(err) => {
                        return oapi::openai_error(
                            StatusCode::BAD_REQUEST,
                            format!("file read: {err}"),
                            oapi::kind::INVALID_REQUEST,
                            Some("file"),
                            Some("multipart_read_error"),
                        );
                    }
                }
            }
            "response_format" => {
                if let Ok(v) = field.text().await {
                    response_format = v;
                }
            }
            "language" => language = field.text().await.ok(),
            "prompt" => prompt = field.text().await.ok(),
            "temperature" => temperature = field.text().await.ok(),
            _ => {
                let _ = field.bytes().await;
            }
        }
    }
    if let Err((param, message)) = check_stt_params(
        language.as_deref(),
        prompt.as_deref(),
        temperature.as_deref(),
    ) {
        return oapi::openai_error(
            StatusCode::BAD_REQUEST,
            message,
            oapi::kind::INVALID_REQUEST,
            Some(param),
            Some("unsupported_parameter"),
        );
    }
    let Some(bytes) = audio_bytes else {
        return oapi::fastapi_validation_error(vec![oapi::missing_field(&["body", "file"])]);
    };

    let samples = match decode_any_to_16k_mono(&bytes, audio_content_type.as_deref()) {
        Ok(s) => s,
        Err(err) => {
            warn!(error = %err, "audio decode failed");
            return oapi::openai_error(
                StatusCode::BAD_REQUEST,
                format!("audio decode: {err}"),
                oapi::kind::INVALID_REQUEST,
                Some("file"),
                Some("audio_decode_error"),
            );
        }
    };

    let whisper = match state.models.whisper() {
        Ok(w) => w.clone(),
        Err(e) => {
            return oapi::openai_error(
                StatusCode::SERVICE_UNAVAILABLE,
                format!("{e:#}"),
                oapi::kind::SERVICE_UNAVAIL,
                None,
                Some("stt_unavailable"),
            );
        }
    };
    let samples_for_diar = if response_format == "diarized_json" {
        Some(samples.clone())
    } else {
        None
    };
    let stt = match tokio::task::spawn_blocking(move || {
        whisper.transcribe_full_with_task(&samples, task)
    })
    .await
    {
        Ok(Ok(r)) => r,
        Ok(Err(err)) => return stt_failure_response(&err),
        Err(err) => {
            warn!(error = %err, "whisper transcribe task join failed");
            return oapi::openai_error(
                StatusCode::INTERNAL_SERVER_ERROR,
                "transcription task failed; see server logs",
                oapi::kind::SERVER,
                None,
                Some("join_error"),
            );
        }
    };

    match response_format.as_str() {
        "" | "text" => (
            StatusCode::OK,
            [(header::CONTENT_TYPE, "text/plain; charset=utf-8")],
            stt.text,
        )
            .into_response(),
        "json" => {
            let body = serde_json::json!({"text": stt.text});
            (
                StatusCode::OK,
                [(header::CONTENT_TYPE, "application/json")],
                body.to_string(),
            )
                .into_response()
        }
        "srt" => oapi::transcriptions::srt_response(&stt),
        "vtt" => oapi::transcriptions::vtt_response(&stt),
        "verbose_json" => oapi::transcriptions::verbose_json_response(&stt),
        "diarized_json" => {
            let diar_segments = match (
                state.models.diar_segmentation.clone(),
                state.models.diar_embedding.clone(),
                samples_for_diar,
            ) {
                (Some(seg), Some(emb), Some(samples)) => {
                    let mut diarizer = diarization::Diarizer::new(
                        seg,
                        emb,
                        diarization::DiarConfig::default(),
                    );
                    match tokio::task::spawn_blocking(move || {
                        diarizer.diarize_utterance(&samples, 0)
                    })
                    .await
                    {
                        Ok(Ok(segs)) => segs,
                        _ => Vec::new(),
                    }
                }
                _ => Vec::new(),
            };

            let segments_json = build_diarized_segments_json(&diar_segments, &stt.segments);

            let body = serde_json::json!({
                "text": stt.text,
                "avg_logprob": stt.avg_logprob,
                "no_speech_prob": stt.no_speech_prob,
                "segments": segments_json,
            });
            (
                StatusCode::OK,
                [(header::CONTENT_TYPE, "application/json")],
                body.to_string(),
            )
                .into_response()
        }
        other => oapi::openai_error(
            StatusCode::BAD_REQUEST,
            format!(
                "unsupported response_format: {other:?} (supported: text, json, verbose_json, srt, vtt, diarized_json)"
            ),
            oapi::kind::INVALID_REQUEST,
            Some("response_format"),
            Some("unsupported_value"),
        ),
    }
}

fn build_diarized_segments_json(
    diar: &[speaches_plus::diarization::DiarSegment],
    timed: &[speaches_plus::stt::TimedSegment],
) -> Vec<serde_json::Value> {
    if diar.is_empty() {
        return timed
            .iter()
            .enumerate()
            .map(|(i, t)| {
                serde_json::json!({
                    "type": "transcript.text.segment",
                    "id": format!("seg_{:03}", i + 1),
                    "speaker": serde_json::Value::Null,
                    "start": t.t_start_ms as f64 / 1000.0,
                    "end": t.t_end_ms as f64 / 1000.0,
                    "duration": (t.t_end_ms.saturating_sub(t.t_start_ms)) as f64 / 1000.0,
                    "text": t.text,
                    "avg_logprob": t.avg_logprob,
                    "no_speech_prob": t.no_speech_prob,
                    "confidence": serde_json::Value::Null,
                })
            })
            .collect();
    }

    let mut buckets: Vec<Vec<usize>> = vec![Vec::new(); diar.len()];
    for (wi, w) in timed.iter().enumerate() {
        let mid_ms = (w.t_start_ms as u64 + w.t_end_ms as u64) / 2;
        let assigned = diar
            .iter()
            .position(|d| ({ d.t_start_ms }..={ d.t_end_ms }).contains(&mid_ms))
            .unwrap_or_else(|| nearest_diar_idx(diar, mid_ms));
        buckets[assigned].push(wi);
    }

    diar.iter()
        .enumerate()
        .map(|(di, d)| {
            let assigned: Vec<&speaches_plus::stt::TimedSegment> =
                buckets[di].iter().map(|&i| &timed[i]).collect();
            let text = join_whisper_segment_text(&assigned);
            let (avg_lp, nsp) = aggregate_segment_stats(&assigned);
            serde_json::json!({
                "type": "transcript.text.segment",
                "id": format!("seg_{:03}", di + 1),
                "speaker": format!("SPEAKER_{:02}", d.speaker),
                "start": d.t_start_ms as f64 / 1000.0,
                "end": d.t_end_ms as f64 / 1000.0,
                "duration": (d.t_end_ms.saturating_sub(d.t_start_ms)) as f64 / 1000.0,
                "text": text,
                "avg_logprob": avg_lp,
                "no_speech_prob": nsp,
                "confidence": d.confidence,
            })
        })
        .collect()
}

fn nearest_diar_idx(diar: &[speaches_plus::diarization::DiarSegment], mid_ms: u64) -> usize {
    diar.iter()
        .enumerate()
        .min_by_key(|(_, d)| {
            let s = d.t_start_ms;
            let e = d.t_end_ms;
            if mid_ms < s {
                s - mid_ms
            } else {
                mid_ms.saturating_sub(e)
            }
        })
        .map(|(i, _)| i)
        .unwrap_or(0)
}

fn join_whisper_segment_text(segs: &[&speaches_plus::stt::TimedSegment]) -> String {
    let mut out = String::new();
    for s in segs {
        let t = s.text.trim();
        if t.is_empty() {
            continue;
        }
        if !out.is_empty() {
            out.push(' ');
        }
        out.push_str(t);
    }
    out
}

fn aggregate_segment_stats(
    segs: &[&speaches_plus::stt::TimedSegment],
) -> (Option<f32>, Option<f32>) {
    let mut lp_sum = 0.0_f64;
    let mut lp_w = 0.0_f64;
    let mut nsp_sum = 0.0_f64;
    let mut nsp_w = 0.0_f64;
    for s in segs {
        let dur = s.t_end_ms.saturating_sub(s.t_start_ms).max(1) as f64;
        if let Some(v) = s.avg_logprob {
            lp_sum += v as f64 * dur;
            lp_w += dur;
        }
        if let Some(v) = s.no_speech_prob {
            nsp_sum += v as f64 * dur;
            nsp_w += dur;
        }
    }
    let lp = if lp_w > 0.0 {
        Some((lp_sum / lp_w) as f32)
    } else {
        None
    };
    let nsp = if nsp_w > 0.0 {
        Some((nsp_sum / nsp_w) as f32)
    } else {
        None
    };
    (lp, nsp)
}

#[cfg(test)]
mod diarized_segments_tests {
    use super::*;
    use speaches_plus::diarization::DiarSegment;
    use speaches_plus::stt::TimedSegment;

    fn ts(start: u32, end: u32, text: &str) -> TimedSegment {
        TimedSegment {
            t_start_ms: start,
            t_end_ms: end,
            text: text.into(),
            ..Default::default()
        }
    }
    fn ts_stats(
        start: u32,
        end: u32,
        text: &str,
        lp: Option<f32>,
        nsp: Option<f32>,
    ) -> TimedSegment {
        TimedSegment {
            t_start_ms: start,
            t_end_ms: end,
            text: text.into(),
            avg_logprob: lp,
            no_speech_prob: nsp,
            ..Default::default()
        }
    }
    fn ds(speaker: u32, start: u64, end: u64) -> DiarSegment {
        DiarSegment {
            speaker,
            t_start_ms: start,
            t_end_ms: end,
            confidence: 0.9,
        }
    }

    #[test]
    fn build_empty_inputs_yield_empty_array() {
        let out = build_diarized_segments_json(&[], &[]);
        assert!(out.is_empty(), "{out:?}");
    }

    #[test]
    fn build_no_diar_fallback_emits_per_whisper_segment_with_null_speaker() {
        let timed = vec![ts(0, 1000, "hello"), ts(1000, 2000, "world")];
        let out = build_diarized_segments_json(&[], &timed);
        assert_eq!(out.len(), 2);
        assert_eq!(out[0]["type"], "transcript.text.segment");
        assert_eq!(out[0]["id"], "seg_001");
        assert!(out[0]["speaker"].is_null());
        assert_eq!(out[0]["text"], "hello");
        assert_eq!(out[1]["id"], "seg_002");
    }

    #[test]
    fn build_midpoint_inside_diar_segment_routes_to_that_bucket() {
        let diar = vec![ds(0, 0, 2000), ds(1, 3000, 5000)];
        let timed = vec![
            ts(0, 1000, "hi"),
            ts(1500, 2500, "there"),
            ts(3000, 4000, "world"),
        ];
        let out = build_diarized_segments_json(&diar, &timed);
        assert_eq!(out.len(), 2);
        assert_eq!(out[0]["speaker"], "SPEAKER_00");
        assert_eq!(out[0]["text"], "hi there");
        assert_eq!(out[1]["speaker"], "SPEAKER_01");
        assert_eq!(out[1]["text"], "world");
    }

    #[test]
    fn build_midpoint_outside_all_diar_routes_to_nearest() {
        let diar = vec![ds(0, 0, 2000), ds(1, 3000, 5000)];
        let timed = vec![ts(2200, 2400, "stray")];
        let out = build_diarized_segments_json(&diar, &timed);
        assert_eq!(out.len(), 2);
        assert_eq!(out[0]["text"], "stray");
        assert_eq!(out[1]["text"], "");
    }

    #[test]
    fn build_empty_whisper_with_nonempty_diar_emits_diar_rows_with_empty_text() {
        let diar = vec![ds(0, 0, 2000), ds(1, 2000, 4000)];
        let out = build_diarized_segments_json(&diar, &[]);
        assert_eq!(out.len(), 2);
        assert_eq!(out[0]["text"], "");
        assert_eq!(out[1]["text"], "");
        assert!(out[0]["avg_logprob"].is_null());
    }

    #[test]
    fn build_preserves_openai_required_fields() {
        let diar = vec![ds(2, 100, 1000)];
        let timed = vec![ts(200, 800, "x")];
        let out = build_diarized_segments_json(&diar, &timed);
        assert_eq!(out[0]["type"], "transcript.text.segment");
        assert_eq!(out[0]["id"], "seg_001");
        assert_eq!(out[0]["speaker"], "SPEAKER_02");

        assert_eq!(out[0]["start"], 0.1);
        assert_eq!(out[0]["end"], 1.0);
    }

    #[test]
    fn nearest_idx_empty_returns_zero() {
        assert_eq!(nearest_diar_idx(&[], 500), 0);
    }

    #[test]
    fn nearest_idx_before_all_returns_first() {
        let d = vec![ds(0, 1000, 2000), ds(1, 3000, 4000)];
        assert_eq!(nearest_diar_idx(&d, 100), 0);
    }

    #[test]
    fn nearest_idx_after_all_returns_last() {
        let d = vec![ds(0, 1000, 2000), ds(1, 3000, 4000)];
        assert_eq!(nearest_diar_idx(&d, 9_000), 1);
    }

    #[test]
    fn nearest_idx_inside_returns_containing() {
        let d = vec![ds(0, 1000, 2000), ds(1, 3000, 4000)];
        assert_eq!(nearest_diar_idx(&d, 3500), 1);
    }

    #[test]
    fn nearest_idx_tie_breaks_to_first_match() {
        let d = vec![ds(0, 1000, 2000), ds(1, 3000, 4000)];
        assert_eq!(nearest_diar_idx(&d, 2500), 0);
    }

    #[test]
    fn join_empty_slice_yields_empty() {
        let out = join_whisper_segment_text(&[]);
        assert_eq!(out, "");
    }

    #[test]
    fn join_skips_empty_and_whitespace_only() {
        let s1 = ts(0, 1, "");
        let s2 = ts(0, 1, "   ");
        let s3 = ts(0, 1, "hello");
        let s4 = ts(0, 1, "world");
        let segs: Vec<&TimedSegment> = vec![&s1, &s2, &s3, &s4];
        assert_eq!(join_whisper_segment_text(&segs), "hello world");
    }

    #[test]
    fn join_trims_each_segment_edges_but_keeps_internal() {
        let s1 = ts(0, 1, "  hi  ");
        let s2 = ts(0, 1, " there world ");
        let segs: Vec<&TimedSegment> = vec![&s1, &s2];
        assert_eq!(join_whisper_segment_text(&segs), "hi there world");
    }

    #[test]
    fn aggregate_empty_yields_none() {
        let (lp, nsp) = aggregate_segment_stats(&[]);
        assert!(lp.is_none() && nsp.is_none());
    }

    #[test]
    fn aggregate_all_none_yields_none() {
        let s1 = ts(0, 1000, "");
        let s2 = ts(1000, 2000, "");
        let segs: Vec<&TimedSegment> = vec![&s1, &s2];
        let (lp, nsp) = aggregate_segment_stats(&segs);
        assert!(lp.is_none() && nsp.is_none());
    }

    #[test]
    fn aggregate_mixed_none_averages_only_present_values() {
        let s1 = ts_stats(0, 1000, "", Some(-1.0), None);
        let s2 = ts_stats(1000, 2000, "", None, Some(0.5));
        let s3 = ts_stats(2000, 3000, "", Some(-0.5), Some(0.1));
        let segs: Vec<&TimedSegment> = vec![&s1, &s2, &s3];
        let (lp, nsp) = aggregate_segment_stats(&segs);

        assert!((lp.unwrap() - -0.75).abs() < 1e-5, "lp={:?}", lp);
        assert!((nsp.unwrap() - 0.3).abs() < 1e-5, "nsp={:?}", nsp);
    }

    #[test]
    fn aggregate_weights_by_duration_not_count() {
        let s1 = ts_stats(0, 100, "", Some(-1.0), None);
        let s2 = ts_stats(100, 1000, "", Some(-0.1), None);
        let segs: Vec<&TimedSegment> = vec![&s1, &s2];
        let (lp, _) = aggregate_segment_stats(&segs);
        assert!((lp.unwrap() - -0.19).abs() < 1e-5, "lp={:?}", lp);
    }

    #[test]
    fn aggregate_zero_duration_clamped_to_min_weight_1() {
        let s1 = ts_stats(500, 500, "", Some(-1.0), None);
        let s2 = ts_stats(500, 1500, "", Some(-0.1), None);
        let segs: Vec<&TimedSegment> = vec![&s1, &s2];
        let (lp, _) = aggregate_segment_stats(&segs);

        assert!(lp.unwrap() < -0.099 && lp.unwrap() > -0.102, "lp={:?}", lp);
    }
}

#[cfg(test)]
mod serving_tests {
    use super::*;

    #[test]
    fn in_flight_guard_decrements_on_drop() {
        let before = METRICS.in_flight.load(Relaxed);
        {
            let _g = InFlightGuard::enter();
            assert_eq!(METRICS.in_flight.load(Relaxed), before + 1);
        }
        assert_eq!(METRICS.in_flight.load(Relaxed), before);
    }

    #[test]
    fn histogram_count_matches_inf_bucket() {
        let m = Metrics {
            requests_total: AtomicU64::new(7),
            in_flight: AtomicI64::new(3),
            resp_2xx: AtomicU64::new(0),
            resp_4xx: AtomicU64::new(0),
            resp_5xx: AtomicU64::new(0),
            latency_le: [const { AtomicU64::new(0) }; 11],
            latency_sum_ms: AtomicU64::new(0),
        };
        m.observe(StatusCode::OK, 3);
        m.observe(StatusCode::OK, 9_000);
        let out = m.render();
        let inf = out
            .lines()
            .find(|l| l.contains("le=\"+Inf\""))
            .and_then(|l| l.rsplit(' ').next().map(|s| s.to_string()))
            .expect("+Inf bucket line");
        let count = out
            .lines()
            .find(|l| l.starts_with("speaches_request_latency_ms_count"))
            .and_then(|l| l.rsplit(' ').next().map(|s| s.to_string()))
            .expect("count line");
        assert_eq!(inf, "2");
        assert_eq!(count, inf, "{out}");
    }

    #[test]
    fn constant_time_eq_matches_semantics_of_plain_eq() {
        assert!(constant_time_eq(b"secret", b"secret"));
        assert!(!constant_time_eq(b"secrez", b"secret"));
        assert!(!constant_time_eq(b"secre", b"secret"));
        assert!(!constant_time_eq(b"secrett", b"secret"));
        assert!(!constant_time_eq(b"", b"secret"));
        assert!(constant_time_eq(b"", b""));
    }

    fn sub(name: &'static str, configured: bool, live: bool, required: bool) -> Subsystem {
        Subsystem {
            name,
            configured,
            live,
            required,
            detail: String::new(),
        }
    }

    #[test]
    fn readiness_ok_when_nothing_configured_is_down() {
        let subs = vec![
            sub("stt", true, true, true),
            sub("chat", false, false, true),
            sub("pii", false, false, true),
        ];
        let (status, body) = readiness_report(&subs);
        assert_eq!(status, StatusCode::OK);
        assert_eq!(body["status"], "ready");
        assert!(body["down"].as_array().unwrap().is_empty());
    }

    #[test]
    fn readiness_503_lists_every_configured_but_dead_subsystem() {
        let subs = vec![
            sub("stt", true, true, true),
            sub("chat", true, false, true),
            sub("pii", true, false, true),
        ];
        let (status, body) = readiness_report(&subs);
        assert_eq!(status, StatusCode::SERVICE_UNAVAILABLE);
        assert_eq!(body["status"], "not_ready");
        let down: Vec<&str> = body["down"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_str().unwrap())
            .collect();
        assert_eq!(down, vec!["chat", "pii"]);
        assert_eq!(body["subsystems"].as_array().unwrap().len(), 3);
    }

    #[test]
    fn readiness_ignores_optional_subsystems() {
        let subs = vec![sub("speaker_encoder", true, false, false)];
        let (status, _) = readiness_report(&subs);
        assert_eq!(status, StatusCode::OK);
    }

    #[test]
    fn stt_params_accept_absent_and_neutral_values() {
        assert!(check_stt_params(None, None, None).is_ok());
        assert!(check_stt_params(Some(""), Some(""), Some("")).is_ok());
        assert!(check_stt_params(Some("auto"), None, None).is_ok());
        assert!(check_stt_params(Some("AUTO"), None, None).is_ok());
        assert!(check_stt_params(None, None, Some("0")).is_ok());
        assert!(check_stt_params(None, None, Some("0.0")).is_ok());
    }

    #[test]
    fn stt_params_reject_unsupported_language() {
        let (param, msg) = check_stt_params(Some("de"), None, None).unwrap_err();
        assert_eq!(param, "language");
        assert!(msg.contains("auto"), "{msg}");
    }

    #[test]
    fn stt_params_reject_prompt_and_nonzero_temperature() {
        assert_eq!(
            check_stt_params(None, Some("hello"), None).unwrap_err().0,
            "prompt"
        );
        assert_eq!(
            check_stt_params(None, None, Some("0.8")).unwrap_err().0,
            "temperature"
        );
        assert_eq!(
            check_stt_params(None, None, Some("hot")).unwrap_err().0,
            "temperature"
        );
    }

    #[test]
    fn sdp_typed_error_is_classified_as_client_error() {
        let err = anyhow::Error::new(webrtc::Error::ErrSessionDescriptionMissingIceUfrag)
            .context("set_remote_description");
        assert!(is_client_sdp_error(&err));
    }

    #[test]
    fn non_sdp_webrtc_error_is_not_a_client_error() {
        let err = anyhow::Error::new(webrtc::Error::ErrCertificateExpired).context("build PC");
        assert!(!is_client_sdp_error(&err));
    }

    #[test]
    fn non_webrtc_error_falls_back_to_message_match() {
        let err = anyhow::anyhow!("something").context("parse offer SDP");
        assert!(is_client_sdp_error(&err));
        let err = anyhow::anyhow!("out of capacity");
        assert!(!is_client_sdp_error(&err));
    }

    #[test]
    fn transcribe_only_checkpoint_translate_is_a_400_not_a_500() {
        let err = anyhow::Error::new(speaches_plus::stt::TranslateUnsupported {
            model_id: "ggml-large-v3-turbo".into(),
        })
        .context("transcribe_full_with_task");
        let resp = stt_failure_response(&err);
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
    }

    #[test]
    fn other_transcribe_failures_stay_5xx() {
        let err = anyhow::anyhow!("decoder blew up");
        let resp = stt_failure_response(&err);
        assert_eq!(resp.status(), StatusCode::INTERNAL_SERVER_ERROR);
    }

    fn pii_fixture_dir() -> std::path::PathBuf {
        let p = std::env::temp_dir().join(format!(
            "speaches-plus-main-pii-{}-{}",
            std::process::id(),
            uuid::Uuid::new_v4().simple()
        ));
        std::fs::create_dir_all(&p).unwrap();
        p
    }

    #[test]
    fn pii_gate_accepts_hf_snapshot_root_with_onnx_subdir() {
        let root = pii_fixture_dir();
        std::fs::create_dir_all(root.join("onnx")).unwrap();
        for p in [
            root.join("onnx").join("model.onnx"),
            root.join("tokenizer.json"),
            root.join("config.json"),
        ] {
            std::fs::write(&p, b"{}").unwrap();
        }
        assert!(!root.join("model.onnx").exists());
        assert!(pii::classifier::layout_present(&root));
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn pii_gate_accepts_flat_layout_and_rejects_a_dir_without_a_model() {
        let flat = pii_fixture_dir();
        for p in [
            flat.join("model.onnx"),
            flat.join("tokenizer.json"),
            flat.join("config.json"),
        ] {
            std::fs::write(&p, b"{}").unwrap();
        }
        assert!(pii::classifier::layout_present(&flat));
        let _ = std::fs::remove_dir_all(&flat);

        let empty = pii_fixture_dir();
        assert!(!pii::classifier::layout_present(&empty));
        let _ = std::fs::remove_dir_all(&empty);
    }

    #[test]
    fn capacity_error_is_not_a_5xx_negotiation_failure() {
        let err = anyhow::anyhow!("{}", realtime::CAPACITY_ERROR);
        assert!(is_capacity_error(&err));
        assert!(!is_client_sdp_error(&err));
        let err = anyhow::anyhow!("build PC failed");
        assert!(!is_capacity_error(&err));
    }
}
