from __future__ import annotations

import threading
from typing import Any

import env

TRACER_NAME = "speaches/realtime"
SERVICE_NAME_DEFAULT = "speaches-plus"

_lock = threading.Lock()
_provider: Any | None = None
_enabled: bool = False

def is_enabled() -> bool:
    return _enabled

def init() -> bool:
    global _provider, _enabled
    with _lock:
        if _provider is not None:
            return _enabled

        endpoint = env.read_str_or_none(env.OTEL_EXPORTER_OTLP_ENDPOINT)
        if not endpoint:
            _enabled = False
            return False

        protocol = env.read_str(
            env.OTEL_EXPORTER_OTLP_PROTOCOL, "http/protobuf"
        ).strip().lower() or "http/protobuf"
        service_name = env.read_str(
            env.OTEL_SERVICE_NAME, SERVICE_NAME_DEFAULT
        ) or SERVICE_NAME_DEFAULT

        try:
            from opentelemetry import trace as ot_trace
            from opentelemetry.sdk.resources import Resource
            from opentelemetry.sdk.trace import TracerProvider
            from opentelemetry.sdk.trace.export import BatchSpanProcessor

            if protocol == "grpc":
                from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
                    OTLPSpanExporter,
                )
                exporter = OTLPSpanExporter(endpoint=endpoint)
            else:
                from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
                    OTLPSpanExporter,
                )
                exporter = OTLPSpanExporter(endpoint=endpoint)

            resource = Resource.create({"service.name": service_name})
            provider = TracerProvider(resource=resource)
            provider.add_span_processor(BatchSpanProcessor(exporter))
            ot_trace.set_tracer_provider(provider)

            _provider = provider
            _enabled = True
            return True
        except ImportError:
            _enabled = False
            return False
        except Exception:
            _enabled = False
            return False

def shutdown() -> None:
    global _provider, _enabled
    with _lock:
        provider = _provider
        _provider = None
        _enabled = False
    if provider is None:
        return
    try:
        flush = getattr(provider, "force_flush", None)
        if callable(flush):
            flush()
        sd = getattr(provider, "shutdown", None)
        if callable(sd):
            sd()
    except Exception:
        pass

def current_span_id_hex() -> str | None:
    if not _enabled:
        return None
    try:
        from opentelemetry import trace as ot_trace

        span = ot_trace.get_current_span()
        ctx = span.get_span_context()
        if not ctx.is_valid:
            return None
        return f"{ctx.span_id:016x}"
    except ImportError:
        return None
    except Exception:
        return None
