use std::time::Duration;

use axum::http::HeaderMap;

pub const HEADER: &str = "x-request-timeout-ms";
pub const MAX_VAR: &str = "NV_MAX_REQUEST_TIMEOUT_MS";
pub const DEFAULT_MAX_MS_600S_SO_A_196K_PREFILL_PLUS_FULL_DECODE_FITS_A_CALLER_BUDGET_WITHOUT_NV_MAX_REQUEST_TIMEOUT_MS: u64 = 600_000;
pub const DEFAULT_MAX_MS: u64 =
    DEFAULT_MAX_MS_600S_SO_A_196K_PREFILL_PLUS_FULL_DECODE_FITS_A_CALLER_BUDGET_WITHOUT_NV_MAX_REQUEST_TIMEOUT_MS;
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
    fn header_is_read_case_insensitively_and_garbage_falls_back_to_the_server_default() {
        assert_eq!(from_headers(&hdr("1500")), Some(ms(1500)));
        assert_eq!(from_headers(&hdr("  1500  ")), Some(ms(1500)));

        let mut mixed = HeaderMap::new();
        mixed.insert(
            axum::http::HeaderName::from_static("x-request-timeout-ms"),
            HeaderValue::from_static("250"),
        );
        assert_eq!(from_headers(&mixed), Some(ms(250)));
        {
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
    }

    #[test]
    fn resolve_with_max_honours_clamps_floors_and_passes_the_server_default_through() {
        assert_eq!(
            resolve_with_max(Some(ms(120)), ms(3000), ms(120_000)),
            ms(120)
        );
        {
            assert_eq!(
                resolve_with_max(Some(ms(10_000_000)), ms(3000), ms(120_000)),
                ms(120_000)
            );
            assert_eq!(
                resolve_with_max(Some(Duration::from_secs(3600)), ms(3000), ms(120_000)),
                ms(120_000)
            );
        }
        {
            assert_eq!(
                resolve_with_max(Some(ms(600_000)), ms(3000), ms(DEFAULT_MAX_MS)),
                ms(600_000)
            );
            assert_eq!(
                resolve_with_max(Some(ms(300_000)), ms(3000), ms(DEFAULT_MAX_MS)),
                ms(300_000),
                "a 5-minute x-request-timeout-ms must not be cut to 120 s and shed"
            );
        }
        {
            assert_eq!(resolve_with_max(Some(ms(0)), ms(3000), ms(120_000)), ms(50));
            assert_eq!(resolve_with_max(Some(ms(7)), ms(3000), ms(120_000)), ms(50));
            assert_eq!(
                resolve_with_max(Some(Duration::ZERO), ms(3000), ms(120_000)),
                floor()
            );
        }
        {
            assert_eq!(resolve_with_max(None, ms(0), ms(120_000)), ms(0));
            assert_eq!(
                resolve_with_max(None, ms(300_000), ms(120_000)),
                ms(300_000)
            );
        }
        {
            assert_eq!(resolve_with_max(Some(ms(5000)), ms(3000), ms(1)), floor());
            assert_eq!(
                resolve_with_max(Some(ms(5000)), ms(3000), Duration::ZERO),
                floor()
            );
        }
    }

    #[test]
    fn max_ms_parses_the_env_var_rejects_garbage_and_defaults_when_unset() {
        assert_eq!(max_ms_from(None), DEFAULT_MAX_MS);
        assert_eq!(max_ms_from(Some("")), DEFAULT_MAX_MS);
        assert_eq!(max_ms_from(Some("forever")), DEFAULT_MAX_MS);
        assert_eq!(max_ms_from(Some("-1")), DEFAULT_MAX_MS);
        assert_eq!(max_ms_from(Some("0")), DEFAULT_MAX_MS);
        assert_eq!(max_ms_from(Some("30000")), 30_000);
        assert_eq!(max_ms_from(Some(" 30000 ")), 30_000);
        assert_eq!(max_ms_from(Some("5")), FLOOR_MS);
        assert_eq!(
            DEFAULT_MAX_MS, 600_000,
            "the default ceiling must admit a caller budget covering a 196k prefill plus a full \
             decode; 120_000 shed those requests (limits census ARB-5)"
        );
        assert_eq!(FLOOR_MS, 50);
        {
            if std::env::var(MAX_VAR).is_err() {
                assert_eq!(max_ms(), DEFAULT_MAX_MS);
            }
        }
    }

    #[test]
    fn the_body_field_beats_the_header_and_body_seconds_reject_nonsense() {
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
        {
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
    }

}
