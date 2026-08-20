from __future__ import annotations

RFC_VERSION = "v3"

class session_defaults:
    MAX_DURATION_S = 1800
    MAX_DURATION_HARD_CAP_S = 3600

class turn_detection:
    THRESHOLD = 0.5
    PREFIX_PADDING_MS = 300
    SILENCE_DURATION_MS = 350
    BARGE_IN_DELAY_MS = 0
    CREATE_RESPONSE = True

    PREFIX_PADDING_MS_MAX = 1000
    SILENCE_DURATION_MS_MIN = 50
    SILENCE_DURATION_MS_MAX = 5000
    BARGE_IN_DELAY_MS_MAX = 1000

class eou_defaults:
    P_THRESHOLD = 0.5
    MIN_DELAY_MS = 500
    MAX_DELAY_MS = 3000
    SILENCE_HARD_CAP_MS = 5000
    INFERENCE_TIMEOUT_MS = 250
    CONTEXT_TURNS = 4
    MAX_CONTEXT_TOKENS = 128
    AUDIO_WINDOW_MS = 8000
    CURVE_K = 12.0
    FAILURE_P_DEFAULT = 1.0

    FUSION_WEIGHT_TEXT = 0.5
    FUSION_RULE = "gated"

    CURVE_K_MAX = 30.0
    SILENCE_HARD_CAP_MS_MAX = 60_000
    INFERENCE_TIMEOUT_MS_MAX = 10_000
    CONTEXT_TURNS_MAX = 64
    SESSION_MAX_DURATION_S_MAX = 86_400

    EAGER_P_THRESHOLD_DISABLED = 1.0
    EAGER_P_THRESHOLD = 0.5
    EAGER_MAX_INFLIGHT = 1
    EAGER_PERIODIC_ENABLED = False
    EAGER_INTERVAL_MS = 250
    PREDICTED_TOKEN_BUFFER_CAP = 256
    EOT_THRESHOLD = 0.7
    EAGER_EOT_THRESHOLD = 0.5
    EAGER_TRANSCRIPT_MISMATCH_RATIO = 0.5

    EAGERNESS_LOW = (0.7, 800, 3000)
    EAGERNESS_MEDIUM = (0.5, 500, 2500)
    EAGERNESS_HIGH = (0.4, 300, 1500)

class buffer_defaults:
    MIN_SPEECH_MS = 100
    MIN_SPEECH_FOR_RESPONSE_MS = 600
    SEALED_BUFFER_RETENTION_COUNT = 4
    PARTIAL_INTERVAL_MS = 500

    MIN_SPEECH_MS_MAX = 60_000
    MIN_SPEECH_FOR_RESPONSE_MS_MAX = 60_000
    SEALED_BUFFER_RETENTION_COUNT_MAX = 1024

class response_defaults:
    DRAIN_CAP_FLOOR_MS = 5_000
    DRAIN_CAP_CEILING_MS = 60_000

class wire_defaults:
    OUTBOUND_QUEUE_CAP_EVENTS = 256
    OUTBOUND_QUEUE_CAP_MS = 5_000
    DATA_CHANNEL_FRAGMENT_MAX = 900

class ws_defaults:
    MAX_MESSAGE_BYTES = 4 * 1024 * 1024
    OUTBOUND_QUEUE_CAP = 256
    IDLE_TIMEOUT_S = 60
    MAX_CONCURRENT_SESSIONS = 64
    PING_INTERVAL_S = 20

class audio_defaults:
    TTS_SAMPLE_RATE = 24_000
    OUT_SAMPLE_RATE = 48_000
    FRAME_MS = 20
    FRAME_SAMPLES = OUT_SAMPLE_RATE * FRAME_MS // 1000
    IN_CHUNK_SAMPLES = 480
    INPUT_HZ = 48_000.0
    OUTPUT_HZ = 16_000.0
    MAX_DECODE_FRAMES = 5_760
    OPUS_SAMPLE_RATE_HZ = 48_000

AUDIO_FORMAT_PCM16 = "pcm16"
AUDIO_FORMAT_PCM16_16K = "pcm16_16k"
AUDIO_FORMAT_G711_ULAW = "g711_ulaw"
AUDIO_FORMAT_G711_ALAW = "g711_alaw"
AUDIO_FORMAT_DEFAULT = AUDIO_FORMAT_PCM16
AUDIO_FORMAT_SUPPORTED = (
    AUDIO_FORMAT_PCM16,
    AUDIO_FORMAT_PCM16_16K,
    AUDIO_FORMAT_G711_ULAW,
    AUDIO_FORMAT_G711_ALAW,
)

REALTIME_SESSION_OBJECT = "realtime.session"
TURN_DETECTION_SERVER_VAD = "server_vad"
TURN_DETECTION_NONE = "none"

MODALITY_TEXT = "text"
MODALITY_AUDIO = "audio"

from .eou_integrated import IntegratedVerdictAction
from .events import (
    ClientEventType,
    ServerEventType,
    parse_client_event,
)
from .framing import frame_event, MAX_FRAGMENT_SIZE
from .sdp_filter import normalize_offer
from .session import Session, Intent, RealtimeQuery, TurnDetectionConfig, TurnDetectionKind
from .state import (
    ConversationItem,
    InvariantViolation,
    ItemContent,
    ItemRole,
    ItemStatus,
    OpenBuffer,
    PendingBargein,
    PredictedRunner,
    PredictedSharedState,
    RespPhase,
    SealedBuffer,
    SessionPhase,
    SessionState,
    TerminationReason,
    Topic,
    VadPhase,
    apply_truncate_to_conversation,
    check_invariants,
    check_state,
)
from .transport import EventSink, OutboundAudioSpec
from .wire import (
    ErrorPayload,
    OutboundEvent,
    ResponsePayload,
    ResponseStatus,
    ResponseStatusDetails,
    ResponseStatusReason,
    serialize_outbound_event,
)

def capabilities_json() -> dict:
    return capabilities_json_with_models(None)

def capabilities_json_with_models(models: Any = None) -> dict:
    import os as _os

    import env as _env
    from eou.types import EouKind as _EouKind

    diar_enabled = False
    diar_max_per_chunk = 0
    diar_max_per_frame = 0
    diar_emb_dim = 0
    diar_frame_hz = 0
    loaded_models: list[str] = []
    if models is not None:
        diar = getattr(models, "diarizer", None) or getattr(models, "diar", None)
        if diar is not None:
            seg = getattr(diar, "segmentation", None)
            emb = getattr(diar, "embedding", None)
            diar_enabled = seg is not None and emb is not None
            if seg is not None:
                diar_max_per_chunk = int(getattr(seg, "max_speakers_per_chunk", 0) or 0)
                diar_max_per_frame = int(getattr(seg, "max_speakers_per_frame", 0) or 0)
                diar_frame_hz = int(getattr(seg, "frame_rate_hz", 0) or 0)
            if emb is not None:
                diar_emb_dim = int(getattr(emb, "embedding_dim", 0) or 0)
        names = getattr(models, "model_ids", None) or getattr(models, "loaded_model_ids", None)
        if isinstance(names, (list, tuple)):
            loaded_models = [str(n) for n in names]

    spec_kinds = [k.as_str() for k in _EouKind.V3_SPEC]
    ext_kinds = [k.as_str() for k in _EouKind.EXTENSIONS]

    eou_kind_env = _os.environ.get(_env.EOU_KIND, "").strip().lower()
    eager_env = _env.read_str_or_none(_env.EOU_EAGERNESS)
    audio_model_env = _env.read_str_or_none(_env.EOU_AUDIO_MODEL_PATH)

    eager_enabled = bool(eager_env)
    integrated_enabled = (eou_kind_env == "") or ("integrated" in eou_kind_env)
    audio_enabled = bool(audio_model_env)

    return {
        "rfc_version": RFC_VERSION,
        "features": {
            "eou_kinds": spec_kinds,
            "fusion_rules": ["noisy_or", "max", "mean", "weighted"],
            "input_audio_formats": list(AUDIO_FORMAT_SUPPORTED),
            "output_audio_formats": list(AUDIO_FORMAT_SUPPORTED),
            "transports": ["webrtc", "websocket"],
            "audio_codecs": ["opus", "pcm16", "g711_ulaw", "g711_alaw"],
            "vad_types": ["server_vad", "semantic_vad"],
        },
        "extensions": {
            "eou_kinds": ext_kinds,
            "fusion_rules": ["gated"],
            "eager_eou": eager_enabled,
            "integrated_eou": integrated_enabled,
            "audio_eou": audio_enabled,
            "predicted_resp_phase": True,
            "models": loaded_models,
            "diarization": {
                "enabled": diar_enabled,
                "max_speakers_per_chunk": diar_max_per_chunk,
                "max_speakers_per_frame": diar_max_per_frame,
                "embedding_dim": diar_emb_dim,
                "frame_rate_hz": diar_frame_hz,
                "endpoints": {
                    "audio_diarization": "/v1/audio/diarization",
                    "audio_embeddings": "/v1/audio/embeddings",
                    "transcription_diarized_json": "/v1/audio/transcriptions?response_format=diarized_json",
                    "realtime_event": "conversation.item.diarization",
                },
            },
        },
    }

def live_session_count() -> int:
    from .transport import webrtc_session_count as _rtc_count
    from .websocket import active_session_count as _ws_count

    return int(_ws_count()) + int(_rtc_count())

def lookup_session_pub(sid: str):
    from .transport import _sessions as _rtc_sessions
    from .websocket import _ws_sessions

    s = _rtc_sessions.get(sid)
    if s is not None:
        return s
    return _ws_sessions.get(sid)

def lookup_session_relay(sid: str):
    s = lookup_session_pub(sid)
    if s is None:
        return None
    obs = getattr(s, "_observer", None)
    return getattr(obs, "relay", None) if obs is not None else None

def register_routes(app) -> None:
    from starlette.responses import JSONResponse

    from .websocket import realtime_ws_endpoint

    @app.get("/v1/realtime/capabilities")
    async def _capabilities():
        return JSONResponse(capabilities_json())

    app.add_websocket_route("/v1/realtime", realtime_ws_endpoint)

__all__ = [
    "AUDIO_FORMAT_DEFAULT",
    "AUDIO_FORMAT_G711_ALAW",
    "AUDIO_FORMAT_G711_ULAW",
    "AUDIO_FORMAT_PCM16",
    "AUDIO_FORMAT_PCM16_16K",
    "AUDIO_FORMAT_SUPPORTED",
    "ClientEventType",
    "ConversationItem",
    "ErrorPayload",
    "EventSink",
    "Intent",
    "IntegratedVerdictAction",
    "InvariantViolation",
    "ItemContent",
    "ItemRole",
    "ItemStatus",
    "MAX_FRAGMENT_SIZE",
    "MODALITY_AUDIO",
    "MODALITY_TEXT",
    "OpenBuffer",
    "OutboundAudioSpec",
    "OutboundEvent",
    "PendingBargein",
    "PredictedRunner",
    "PredictedSharedState",
    "REALTIME_SESSION_OBJECT",
    "RFC_VERSION",
    "RealtimeQuery",
    "RespPhase",
    "ResponsePayload",
    "ResponseStatus",
    "ResponseStatusDetails",
    "ResponseStatusReason",
    "SealedBuffer",
    "ServerEventType",
    "Session",
    "SessionPhase",
    "SessionState",
    "TURN_DETECTION_NONE",
    "TURN_DETECTION_SERVER_VAD",
    "TerminationReason",
    "Topic",
    "TurnDetectionConfig",
    "TurnDetectionKind",
    "VadPhase",
    "apply_truncate_to_conversation",
    "audio_defaults",
    "buffer_defaults",
    "capabilities_json",
    "capabilities_json_with_models",
    "live_session_count",
    "lookup_session_pub",
    "lookup_session_relay",
    "check_invariants",
    "check_state",
    "eou_defaults",
    "frame_event",
    "normalize_offer",
    "parse_client_event",
    "register_routes",
    "response_defaults",
    "serialize_outbound_event",
    "session_defaults",
    "turn_detection",
    "wire_defaults",
    "ws_defaults",
]
