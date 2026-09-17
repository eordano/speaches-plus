use std::io::{BufRead, Write};
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::time::{SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};

pub const TRAJECTORY_DIR_ENV: &str = "NV_TRAJECTORY_DIR";
pub const LABEL_HEADER: &str = "x-trajectory-label";
pub const TAG_HEADER: &str = "x-trajectory-tag";

#[derive(Debug, Clone, Default)]
pub struct TrajMeta {
    pub label: Option<String>,
    pub tag: Option<String>,
}

impl TrajMeta {
    pub fn from_headers(headers: &axum::http::HeaderMap) -> Self {
        Self {
            label: header_value(headers, LABEL_HEADER),
            tag: header_value(headers, TAG_HEADER),
        }
    }
}

fn header_value(headers: &axum::http::HeaderMap, name: &str) -> Option<String> {
    headers
        .get(name)
        .and_then(|v| v.to_str().ok())
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(String::from)
}

pub fn daily_label_file_name(ts: u64) -> String {
    daily_file_name_so_tmpfiles_age_rules_can_prune_old_days(ts).replace("trajectories-", "labels-")
}

pub fn daily_file_name_so_tmpfiles_age_rules_can_prune_old_days(ts: u64) -> String {
    let days = (ts / 86_400) as i64;
    let z = days + 719_468;
    let era = if z >= 0 { z } else { z - 146_096 } / 146_097;
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = if m <= 2 { y + 1 } else { y };
    format!("trajectories-{y:04}{m:02}{d:02}.jsonl")
}

#[typeshare::typeshare]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrajectoryRecord {
    #[typeshare(serialized_as = "U53")]
    pub ts: u64,
    pub endpoint: String,
    pub id: String,
    pub model: String,
    pub prompt: String,
    pub completion: String,
    pub reasoning: Option<String>,
    pub finish_reason: String,
    pub stopped_by: Option<String>,
    pub multimodal: bool,
    pub label: Option<String>,
    pub tag: Option<String>,
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    #[typeshare(serialized_as = "U53")]
    pub seed: u64,
    pub temperature: Option<f32>,
}

impl TrajectoryRecord {
    pub fn now_secs() -> u64 {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0)
    }
}

pub fn label_from_headers(headers: &axum::http::HeaderMap) -> Option<String> {
    header_value(headers, LABEL_HEADER)
}

pub fn dir_from_env() -> Option<PathBuf> {
    std::env::var_os(TRAJECTORY_DIR_ENV).map(PathBuf::from)
}

pub fn enabled() -> bool {
    dir_from_env().is_some()
}

static APPEND_SERIALIZES_WRITERS_IN_THIS_PROCESS: Mutex<()> = Mutex::new(());

pub fn append_to(dir: &Path, record: &TrajectoryRecord) -> anyhow::Result<()> {
    let mut line = serde_json::to_string(record)?;
    line.push('\n');
    let _guard = APPEND_SERIALIZES_WRITERS_IN_THIS_PROCESS
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    std::fs::create_dir_all(dir)?;
    let mut f = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(
            dir.join(daily_file_name_so_tmpfiles_age_rules_can_prune_old_days(
                record.ts,
            )),
        )?;
    f.write_all(line.as_bytes())?;
    Ok(())
}

fn append_sync(dir: PathBuf, record: TrajectoryRecord) {
    if let Err(err) = append_to(&dir, &record) {
        tracing::warn!(
            dir = %dir.display(),
            error = %format!("{err:#}"),
            "trajectory record dropped; generation is never failed over logging"
        );
    }
}

pub fn append(record: TrajectoryRecord) {
    let Some(dir) = dir_from_env() else {
        return;
    };
    match tokio::runtime::Handle::try_current() {
        Ok(handle) => {
            handle.spawn_blocking(move || append_sync(dir, record));
        }
        Err(_) => append_sync(dir, record),
    }
}

#[typeshare::typeshare]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LabelRecord {
    #[typeshare(serialized_as = "U53")]
    pub ts: u64,
    pub id: Option<String>,
    pub tag: Option<String>,
    pub label: String,
}

pub fn append_label_to(dir: &Path, record: &LabelRecord) -> anyhow::Result<()> {
    let mut line = serde_json::to_string(record)?;
    line.push('\n');
    let _guard = APPEND_SERIALIZES_WRITERS_IN_THIS_PROCESS
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    std::fs::create_dir_all(dir)?;
    let mut f = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(dir.join(daily_label_file_name(record.ts)))?;
    f.write_all(line.as_bytes())?;
    Ok(())
}

pub const LABEL_MAX_CHARS: usize = 64;

pub fn validate_label_request(
    id: Option<&str>,
    tag: Option<&str>,
    label: &str,
) -> Result<(), String> {
    let id_ok = id.is_some_and(|s| !s.trim().is_empty());
    let tag_ok = tag.is_some_and(|s| !s.trim().is_empty());
    if !id_ok && !tag_ok {
        return Err("provide id (upstream response id) or tag (x-trajectory-tag value)".into());
    }
    let label = label.trim();
    if label.is_empty() {
        return Err("label is empty".into());
    }
    if label.len() > LABEL_MAX_CHARS {
        return Err(format!("label exceeds {LABEL_MAX_CHARS} chars"));
    }
    if label.chars().any(|c| c.is_control()) {
        return Err("label contains control characters".into());
    }
    Ok(())
}

pub fn label_target_exists_in_retained_trajectories(
    dir: &Path,
    id: Option<&str>,
    tag: Option<&str>,
) -> bool {
    let id = id.map(str::trim).filter(|s| !s.is_empty());
    let tag = tag.map(str::trim).filter(|s| !s.is_empty());
    if id.is_none() && tag.is_none() {
        return false;
    }
    let Ok(entries) = std::fs::read_dir(dir) else {
        return false;
    };
    for entry in entries.flatten() {
        let name = entry.file_name();
        let Some(name) = name.to_str() else { continue };
        if !(name.starts_with("trajectories-") && name.ends_with(".jsonl")) {
            continue;
        }
        let Ok(f) = std::fs::File::open(entry.path()) else {
            continue;
        };
        for line in std::io::BufReader::new(f).lines() {
            let Ok(line) = line else { break };
            let cheap_prefilter_hit =
                id.is_some_and(|v| line.contains(v)) || tag.is_some_and(|v| line.contains(v));
            if !cheap_prefilter_hit {
                continue;
            }
            let Ok(v) = serde_json::from_str::<serde_json::Value>(&line) else {
                continue;
            };
            let id_hit = id.is_some_and(|want| v.get("id").and_then(|x| x.as_str()) == Some(want));
            let tag_hit =
                tag.is_some_and(|want| v.get("tag").and_then(|x| x.as_str()) == Some(want));
            if id_hit || tag_hit {
                return true;
            }
        }
    }
    false
}

#[typeshare::typeshare]
#[derive(Debug, Clone, Deserialize)]
pub struct LabelHttpRequest {
    pub id: Option<String>,
    pub tag: Option<String>,
    pub label: String,
}

#[typeshare::typeshare]
#[derive(Debug, Clone, Serialize)]
pub struct LabelHttpResponse {
    pub id: Option<String>,
    pub tag: Option<String>,
    pub label: String,
    pub recorded: bool,
    pub matched: bool,
}

pub async fn handle_label(
    crate::oapi::chat::json_ext::OaiJson(req): crate::oapi::chat::json_ext::OaiJson<
        LabelHttpRequest,
    >,
) -> axum::response::Response {
    use axum::response::IntoResponse;
    let Some(dir) = dir_from_env() else {
        return crate::oapi::openai_error(
            axum::http::StatusCode::BAD_REQUEST,
            format!(
                "trajectory capture is disabled ({TRAJECTORY_DIR_ENV} unset), so there is \
                     nothing to label"
            ),
            crate::oapi::kind::INVALID_REQUEST,
            None,
            Some("trajectory_capture_disabled"),
        );
    };
    if let Err(msg) = validate_label_request(req.id.as_deref(), req.tag.as_deref(), &req.label) {
        return crate::oapi::openai_error(
            axum::http::StatusCode::BAD_REQUEST,
            msg,
            crate::oapi::kind::INVALID_REQUEST,
            Some("label"),
            None,
        );
    }
    let trimmed = |v: Option<String>| v.map(|s| s.trim().to_string()).filter(|s| !s.is_empty());
    let record = LabelRecord {
        ts: TrajectoryRecord::now_secs(),
        id: trimmed(req.id),
        tag: trimmed(req.tag),
        label: req.label.trim().to_string(),
    };
    let stored = tokio::task::spawn_blocking(move || {
        append_label_to(&dir, &record).map(|_| {
            let matched = label_target_exists_in_retained_trajectories(
                &dir,
                record.id.as_deref(),
                record.tag.as_deref(),
            );
            (record, matched)
        })
    })
    .await;
    match stored {
        Ok(Ok((record, matched))) => (
            axum::http::StatusCode::OK,
            axum::Json(LabelHttpResponse {
                id: record.id,
                tag: record.tag,
                label: record.label,
                recorded: true,
                matched,
            }),
        )
            .into_response(),
        Ok(Err(err)) => crate::oapi::openai_error(
            axum::http::StatusCode::INTERNAL_SERVER_ERROR,
            format!("label append failed: {err:#}"),
            crate::oapi::kind::SERVER,
            None,
            None,
        ),
        Err(join) => crate::oapi::openai_error(
            axum::http::StatusCode::INTERNAL_SERVER_ERROR,
            format!("label writer panicked: {join}"),
            crate::oapi::kind::SERVER,
            None,
            None,
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn record(completion: &str) -> TrajectoryRecord {
        TrajectoryRecord {
            ts: 1,
            endpoint: "chat".into(),
            id: "chatcmpl-x".into(),
            model: "m".into(),
            prompt: "<start_of_turn>user\nhi<end_of_turn>\n".into(),
            completion: completion.into(),
            reasoning: None,
            finish_reason: "stop".into(),
            stopped_by: None,
            multimodal: false,
            label: Some("good".into()),
            tag: Some("tag-1".into()),
            prompt_tokens: 3,
            completion_tokens: 2,
            seed: 7,
            temperature: Some(0.6),
        }
    }

    #[test]
    fn the_daily_file_name_is_correct_civil_date_math() {
        assert_eq!(
            daily_file_name_so_tmpfiles_age_rules_can_prune_old_days(0),
            "trajectories-19700101.jsonl"
        );
        assert_eq!(
            daily_file_name_so_tmpfiles_age_rules_can_prune_old_days(1_772_064_000),
            "trajectories-20260226.jsonl"
        );
    }

    #[test]
    fn appended_lines_parse_back_and_carry_prompt_completion_keys_train_runner_reads() {
        let dir = std::env::temp_dir().join(format!("traj-test-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        append_to(&dir, &record("a\nb")).unwrap();
        append_to(&dir, &record("second")).unwrap();
        let raw = std::fs::read_to_string(
            dir.join(daily_file_name_so_tmpfiles_age_rules_can_prune_old_days(1)),
        )
        .unwrap();
        let lines: Vec<String> = raw.lines().map(String::from).collect();
        assert_eq!(lines.len(), 2, "one record per line: {raw:?}");
        for line in &lines {
            let v: serde_json::Value = serde_json::from_str(line).unwrap();
            assert!(v.get("prompt").and_then(|p| p.as_str()).is_some(), "{line}");
            assert!(
                v.get("completion").and_then(|c| c.as_str()).is_some(),
                "{line}"
            );
            let parsed: TrajectoryRecord = serde_json::from_str(line).unwrap();
            assert_eq!(parsed.model, "m");
            assert_eq!(parsed.label.as_deref(), Some("good"));
        }
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn label_requests_reject_empty_oversized_and_control_input() {
        assert!(validate_label_request(Some("chatcmpl-x"), None, "good").is_ok());
        assert!(validate_label_request(None, Some("tag-1"), "good").is_ok());
        assert!(validate_label_request(None, None, "good").is_err());
        assert!(validate_label_request(Some(""), Some(" "), "good").is_err());
        assert!(validate_label_request(Some("chatcmpl-x"), None, "  ").is_err());
        assert!(validate_label_request(Some("chatcmpl-x"), None, &"g".repeat(65)).is_err());
        assert!(validate_label_request(Some("chatcmpl-x"), None, "bad\nrow").is_err());
    }

    #[test]
    fn label_records_land_in_a_daily_labels_file_and_parse_back() {
        let dir = std::env::temp_dir().join(format!("traj-label-test-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let rec = LabelRecord {
            ts: 1,
            id: Some("chatcmpl-x".into()),
            tag: Some("tag-1".into()),
            label: "good".into(),
        };
        append_label_to(&dir, &rec).unwrap();
        let raw = std::fs::read_to_string(dir.join(daily_label_file_name(1))).unwrap();
        let parsed: LabelRecord = serde_json::from_str(raw.trim()).unwrap();
        assert_eq!(parsed.id.as_deref(), Some("chatcmpl-x"));
        assert_eq!(parsed.tag.as_deref(), Some("tag-1"));
        assert_eq!(parsed.label, "good");
        assert_eq!(daily_label_file_name(1), "labels-19700101.jsonl");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn label_matched_scan_finds_only_real_tag_or_id_fields_in_retained_files() {
        let dir = std::env::temp_dir().join(format!("traj-match-test-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        assert!(
            !label_target_exists_in_retained_trajectories(&dir, None, Some("tag-1")),
            "a missing dir means no retained record can match"
        );
        let mut rec = record("hello");
        rec.prompt = "user mentioned tag-2 in conversation".into();
        append_to(&dir, &rec).unwrap();
        assert!(label_target_exists_in_retained_trajectories(
            &dir,
            None,
            Some("tag-1")
        ));
        assert!(label_target_exists_in_retained_trajectories(
            &dir,
            Some("chatcmpl-x"),
            None
        ));
        assert!(label_target_exists_in_retained_trajectories(
            &dir,
            Some("chatcmpl-x"),
            Some("tag-unknown")
        ));
        assert!(
            !label_target_exists_in_retained_trajectories(&dir, None, Some("tag-2")),
            "a tag appearing only inside prompt text is not a join"
        );
        assert!(!label_target_exists_in_retained_trajectories(
            &dir,
            Some("chatcmpl-nope"),
            Some("tag-nope")
        ));
        assert!(!label_target_exists_in_retained_trajectories(
            &dir, None, None
        ));
        assert!(!label_target_exists_in_retained_trajectories(
            &dir,
            Some("  "),
            Some("")
        ));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn label_response_keeps_the_pre_named_wire_shape_with_null_id_and_tag_keys() {
        let v = serde_json::to_value(LabelHttpResponse {
            id: None,
            tag: Some("tag-1".into()),
            label: "good".into(),
            recorded: true,
            matched: false,
        })
        .unwrap();
        assert!(v.get("id").is_some_and(|x| x.is_null()));
        assert_eq!(v["tag"].as_str(), Some("tag-1"));
        assert_eq!(v["label"].as_str(), Some("good"));
        assert_eq!(v["recorded"].as_bool(), Some(true));
        assert_eq!(v["matched"].as_bool(), Some(false));
    }

    #[test]
    fn label_header_is_trimmed_and_empty_means_none() {
        let mut h = axum::http::HeaderMap::new();
        assert_eq!(label_from_headers(&h), None);
        h.insert(LABEL_HEADER, " good ".parse().unwrap());
        assert_eq!(label_from_headers(&h).as_deref(), Some("good"));
        h.insert(LABEL_HEADER, "  ".parse().unwrap());
        assert_eq!(label_from_headers(&h), None);
    }
}
