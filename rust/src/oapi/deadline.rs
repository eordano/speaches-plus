use std::time::Duration;

use axum::http::HeaderMap;

pub const HEADER: &str = "x-request-timeout-ms";
pub const MAX_VAR: &str = "NV_MAX_REQUEST_TIMEOUT_MS";
pub const DEFAULT_MAX_MS: u64 = 120_000;
pub const FLOOR_MS: u64 = 50;

pub fn floor() -> Duration {
    Duration::from_millis(FLOOR_MS)
}

pub fn max_ms_from(raw: Option<&str>) -> u64 {
    raw.and_then(|v| v.trim().parse::<u64>().ok())
        .filter(|n| *n > 0)
        .unwrap_or(DEFAULT_MAX_MS)
        .max(FLOOR_MS)
}

pub fn max_ms() -> u64 {
    max_ms_from(std::env::var(MAX_VAR).ok().as_deref())
}

pub fn max() -> Duration {
    Duration::from_millis(max_ms())
}

pub fn parse_ms(raw: &str) -> Option<Duration> {
    let raw = raw.trim();
    if raw.is_empty() {
        return None;
    }
    raw.parse::<u64>().ok().map(Duration::from_millis)
}

pub fn from_headers(headers: &HeaderMap) -> Option<Duration> {
    headers
        .get(HEADER)
        .and_then(|v| v.to_str().ok())
        .and_then(parse_ms)
}

pub fn from_body_ms(ms: Option<u64>) -> Option<Duration> {
    ms.map(Duration::from_millis)
}

pub fn from_body_seconds(secs: Option<f64>) -> Option<Duration> {
    let secs = secs?;
    if !secs.is_finite() || secs < 0.0 {
        return None;
    }
    let ms = (secs * 1000.0).round().min(u64::MAX as f64) as u64;
    Some(Duration::from_millis(ms))
}

pub fn client_budget(body: Option<Duration>, headers: &HeaderMap) -> Option<Duration> {
    body.or_else(|| from_headers(headers))
}

pub fn resolve_with_max(
    client: Option<Duration>,
    server_default: Duration,
    max: Duration,
) -> Duration {
    let floor = floor();
    let max = max.max(floor);
    match client {
        Some(requested) => requested.clamp(floor, max),
        None => server_default,
    }
}

pub fn resolve(client: Option<Duration>, server_default: Duration) -> Duration {
    resolve_with_max(client, server_default, max())
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::http::HeaderValue;

    fn hdr(v: &str) -> HeaderMap {
        let mut h = HeaderMap::new();
        h.insert(HEADER, HeaderValue::from_str(v).unwrap());
        h
    }

    fn ms(n: u64) -> Duration {
        Duration::from_millis(n)
    }

    #[test]
    fn header_is_read_case_insensitively_in_milliseconds() {
        assert_eq!(from_headers(&hdr("1500")), Some(ms(1500)));
        assert_eq!(from_headers(&hdr("  1500  ")), Some(ms(1500)));

        let mut mixed = HeaderMap::new();
        mixed.insert(
            axum::http::HeaderName::from_static("x-request-timeout-ms"),
            HeaderValue::from_static("250"),
        );
        assert_eq!(from_headers(&mixed), Some(ms(250)));
    }

    #[test]
    fn garbage_and_absent_headers_fall_back_to_the_server_default() {
        for junk in [
            "",
            "  ",
            "soon",
            "-1",
            "1.5",
            "1500ms",
            "99999999999999999999",
        ] {
            assert_eq!(from_headers(&hdr(junk)), None, "junk header {junk:?}");
            assert_eq!(
                resolve_with_max(from_headers(&hdr(junk)), ms(3000), ms(120_000)),
                ms(3000),
                "junk header {junk:?} must fall back, not error"
            );
        }
        assert_eq!(from_headers(&HeaderMap::new()), None);
        assert_eq!(
            resolve_with_max(from_headers(&HeaderMap::new()), ms(3000), ms(120_000)),
            ms(3000)
        );
    }

    #[test]
    fn a_short_client_budget_is_honoured_below_the_server_default() {
        assert_eq!(
            resolve_with_max(Some(ms(120)), ms(3000), ms(120_000)),
            ms(120)
        );
    }

    #[test]
    fn a_long_client_budget_is_clamped_to_the_server_maximum() {
        assert_eq!(
            resolve_with_max(Some(ms(10_000_000)), ms(3000), ms(120_000)),
            ms(120_000)
        );
        assert_eq!(
            resolve_with_max(Some(Duration::from_secs(3600)), ms(3000), ms(120_000)),
            ms(120_000)
        );
    }

    #[test]
    fn a_zero_or_tiny_client_budget_is_lifted_to_the_floor() {
        assert_eq!(resolve_with_max(Some(ms(0)), ms(3000), ms(120_000)), ms(50));
        assert_eq!(resolve_with_max(Some(ms(7)), ms(3000), ms(120_000)), ms(50));
        assert_eq!(
            resolve_with_max(Some(Duration::ZERO), ms(3000), ms(120_000)),
            floor()
        );
    }

    #[test]
    fn the_server_default_passes_through_unclamped() {
        assert_eq!(resolve_with_max(None, ms(0), ms(120_000)), ms(0));
        assert_eq!(
            resolve_with_max(None, ms(300_000), ms(120_000)),
            ms(300_000)
        );
    }

    #[test]
    fn a_degenerate_max_cannot_drop_below_the_floor() {
        assert_eq!(resolve_with_max(Some(ms(5000)), ms(3000), ms(1)), floor());
        assert_eq!(
            resolve_with_max(Some(ms(5000)), ms(3000), Duration::ZERO),
            floor()
        );
    }

    #[test]
    fn max_ms_parses_the_env_var_and_rejects_garbage() {
        assert_eq!(max_ms_from(None), DEFAULT_MAX_MS);
        assert_eq!(max_ms_from(Some("")), DEFAULT_MAX_MS);
        assert_eq!(max_ms_from(Some("forever")), DEFAULT_MAX_MS);
        assert_eq!(max_ms_from(Some("-1")), DEFAULT_MAX_MS);
        assert_eq!(max_ms_from(Some("0")), DEFAULT_MAX_MS);
        assert_eq!(max_ms_from(Some("30000")), 30_000);
        assert_eq!(max_ms_from(Some(" 30000 ")), 30_000);
        assert_eq!(max_ms_from(Some("5")), FLOOR_MS);
        assert_eq!(DEFAULT_MAX_MS, 120_000);
        assert_eq!(FLOOR_MS, 50);
    }

    #[test]
    fn the_body_field_beats_the_header() {
        let headers = hdr("9000");
        assert_eq!(
            client_budget(from_body_ms(Some(250)), &headers),
            Some(ms(250))
        );
        assert_eq!(
            client_budget(from_body_seconds(Some(0.25)), &headers),
            Some(ms(250))
        );
        assert_eq!(client_budget(None, &headers), Some(ms(9000)));
        assert_eq!(client_budget(None, &HeaderMap::new()), None);
    }

    #[test]
    fn body_seconds_rejects_nonsense_without_erroring() {
        assert_eq!(from_body_seconds(None), None);
        assert_eq!(from_body_seconds(Some(-1.0)), None);
        assert_eq!(from_body_seconds(Some(f64::NAN)), None);
        assert_eq!(from_body_seconds(Some(f64::INFINITY)), None);
        assert_eq!(from_body_seconds(Some(1.5)), Some(ms(1500)));
        assert_eq!(from_body_seconds(Some(0.0)), Some(Duration::ZERO));
        assert_eq!(
            resolve_with_max(from_body_seconds(Some(0.0)), ms(3000), ms(120_000)),
            floor()
        );
    }

    #[test]
    fn max_defaults_when_the_env_var_is_unset() {
        if std::env::var(MAX_VAR).is_err() {
            assert_eq!(max_ms(), DEFAULT_MAX_MS);
        }
    }
}
