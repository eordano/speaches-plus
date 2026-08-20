#![allow(dead_code)]

use std::sync::OnceLock;

use opentelemetry::trace::TraceContextExt;
use opentelemetry::KeyValue;
use opentelemetry_otlp::{Protocol, WithExportConfig};
use opentelemetry_sdk::trace::SdkTracerProvider;
use opentelemetry_sdk::Resource;
use tracing::Subscriber;
use tracing_opentelemetry::{OpenTelemetryLayer, OpenTelemetrySpanExt};
use tracing_subscriber::registry::LookupSpan;
use tracing_subscriber::Layer;

use super::defaults;

static PROVIDER: OnceLock<SdkTracerProvider> = OnceLock::new();
static OTEL_ENABLED: OnceLock<bool> = OnceLock::new();

pub fn is_enabled() -> bool {
    *OTEL_ENABLED.get().unwrap_or(&false)
}

pub fn current_span_id_hex() -> Option<String> {
    if !is_enabled() {
        return None;
    }
    let span = tracing::Span::current();
    let cx = span.context();
    let span_ref = cx.span();
    let sc = span_ref.span_context();
    if !sc.is_valid() {
        return None;
    }
    Some(format!("{:016x}", sc.span_id()))
}

pub fn try_install_layer<S>() -> anyhow::Result<Option<Box<dyn Layer<S> + Send + Sync + 'static>>>
where
    S: Subscriber + for<'a> LookupSpan<'a> + Send + Sync,
{
    let endpoint = match std::env::var(defaults::env::OTEL_EXPORTER_OTLP_ENDPOINT) {
        Ok(v) if !v.trim().is_empty() => v,
        _ => {
            let _ = OTEL_ENABLED.set(false);
            return Ok(None);
        }
    };

    let protocol = std::env::var(defaults::env::OTEL_EXPORTER_OTLP_PROTOCOL)
        .ok()
        .map(|s| s.trim().to_ascii_lowercase())
        .unwrap_or_else(|| "http/protobuf".into());

    let service_name = std::env::var(defaults::env::OTEL_SERVICE_NAME)
        .unwrap_or_else(|_| defaults::tracing::SERVICE_NAME_DEFAULT.into());

    let resource = Resource::builder()
        .with_attribute(KeyValue::new("service.name", service_name))
        .build();

    let exporter = match protocol.as_str() {
        "grpc" => opentelemetry_otlp::SpanExporter::builder()
            .with_tonic()
            .with_endpoint(endpoint)
            .with_protocol(Protocol::Grpc)
            .build()?,
        _ => opentelemetry_otlp::SpanExporter::builder()
            .with_http()
            .with_endpoint(endpoint)
            .with_protocol(Protocol::HttpBinary)
            .build()?,
    };

    let provider = SdkTracerProvider::builder()
        .with_batch_exporter(exporter)
        .with_resource(resource)
        .build();

    let tracer =
        opentelemetry::trace::TracerProvider::tracer(&provider, defaults::tracing::TRACER_NAME);

    opentelemetry::global::set_tracer_provider(provider.clone());
    let _ = PROVIDER.set(provider);
    let _ = OTEL_ENABLED.set(true);

    let layer = OpenTelemetryLayer::new(tracer);
    Ok(Some(Box::new(layer)))
}

pub fn shutdown() {
    if let Some(provider) = PROVIDER.get() {
        let _ = provider.force_flush();
        let _ = provider.shutdown();
    }
}
