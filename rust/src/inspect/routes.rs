#![allow(dead_code)]

use std::io::{BufRead, Read, Seek, SeekFrom};
use std::path::{Path as FsPath, PathBuf};
use std::time::SystemTime;

use axum::body::Body;
use axum::extract::{
    ws::{Message, WebSocket, WebSocketUpgrade},
    Path, Query,
};
use axum::http::{header, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::Json;
use futures_util::SinkExt;
use serde::Deserialize;
use tracing::warn;

use super::audio_store::{wav_header, Channel};
use super::registry;
use super::types::{SessionHistoryEntry, SessionMeta};
use super::{retention_bytes, retention_count, retention_days, session_dir};

const STREAM_CHUNK: usize = 64 * 1024;

fn sanitize_sid(sid: &str) -> Option<&str> {
    if sid.is_empty() || sid.len() > 64 {
        return None;
    }
    if sid
        .bytes()
        .all(|b| b.is_ascii_alphanumeric() || b == b'_' || b == b'-')
    {
        Some(sid)
    } else {
        None
    }
}

pub async fn inspect_sessions() -> Json<Vec<SessionMeta>> {
    Json(registry::list_meta())
}

pub async fn inspect_history() -> Json<Vec<SessionHistoryEntry>> {
    let Some(sd) = session_dir() else {
        return Json(Vec::new());
    };
    let out = tokio::task::spawn_blocking(move || scan_history(&sd))
        .await
        .unwrap_or_else(|err| {
            warn!(error = %err, "scan history join");
            Vec::new()
        });
    Json(out)
}

fn scan_history(sd: &FsPath) -> Vec<SessionHistoryEntry> {
    let mut out: Vec<SessionHistoryEntry> = Vec::new();
    if let Ok(rd) = std::fs::read_dir(sd) {
        for entry in rd.flatten() {
            let p = entry.path();
            if p.extension().and_then(|e| e.to_str()) != Some("ndjson") {
                continue;
            }
            let stem = match p.file_stem().and_then(|n| n.to_str()) {
                Some(n) => n.to_string(),
                None => continue,
            };
            let meta = match std::fs::metadata(&p) {
                Ok(m) if m.is_file() => m,
                _ => continue,
            };
            let mtime = meta
                .modified()
                .ok()
                .and_then(|t| t.duration_since(SystemTime::UNIX_EPOCH).ok())
                .map(|d| d.as_secs_f64())
                .unwrap_or(0.0);
            out.push(SessionHistoryEntry {
                id: stem,
                size_bytes: meta.len(),
                mtime,
            });
        }
    }
    out.sort_by(|a, b| {
        b.mtime
            .partial_cmp(&a.mtime)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    out
}

pub async fn inspect_history_stream(Path(sid): Path<String>) -> Response {
    let Some(sid) = sanitize_sid(&sid) else {
        return (StatusCode::NOT_FOUND, "session not found").into_response();
    };
    let Some(sd) = session_dir() else {
        return (StatusCode::NOT_FOUND, "session not found").into_response();
    };
    let path = sd.join(format!("{}.ndjson", sid));
    let opened = {
        let path = path.clone();
        tokio::task::spawn_blocking(move || std::fs::File::open(&path)).await
    };
    let file = match opened {
        Ok(Ok(f)) => f,
        Ok(Err(err)) if err.kind() == std::io::ErrorKind::NotFound => {
            return (StatusCode::NOT_FOUND, "session not found").into_response();
        }
        Ok(Err(err)) => {
            warn!(error = %err, path = %path.display(), "open history ndjson");
            return (StatusCode::INTERNAL_SERVER_ERROR, "read failed").into_response();
        }
        Err(err) => {
            warn!(error = %err, "open history ndjson join");
            return (StatusCode::INTERNAL_SERVER_ERROR, "read failed").into_response();
        }
    };
    Response::builder()
        .status(StatusCode::OK)
        .header(header::CONTENT_TYPE, "application/x-ndjson")
        .body(Body::from_stream(chunk_stream(file)))
        .unwrap_or_else(|_| StatusCode::INTERNAL_SERVER_ERROR.into_response())
}

fn chunk_stream(
    mut file: std::fs::File,
) -> tokio_stream::wrappers::ReceiverStream<std::io::Result<Vec<u8>>> {
    let (tx, rx) = tokio::sync::mpsc::channel::<std::io::Result<Vec<u8>>>(4);
    tokio::task::spawn_blocking(move || {
        let mut buf = vec![0u8; STREAM_CHUNK];
        loop {
            match file.read(&mut buf) {
                Ok(0) => break,
                Ok(n) => {
                    if tx.blocking_send(Ok(buf[..n].to_vec())).is_err() {
                        break;
                    }
                }
                Err(err) if err.kind() == std::io::ErrorKind::Interrupted => continue,
                Err(err) => {
                    let _ = tx.blocking_send(Err(err));
                    break;
                }
            }
        }
    });
    tokio_stream::wrappers::ReceiverStream::new(rx)
}

#[derive(Debug, Deserialize)]
pub struct AudioQuery {
    pub channel: String,
    #[serde(default)]
    pub from_ms: u64,
    #[serde(default)]
    pub to_ms: u64,
}

pub async fn inspect_audio(Path(sid): Path<String>, Query(q): Query<AudioQuery>) -> Response {
    let Some(sid) = sanitize_sid(&sid) else {
        return (StatusCode::NOT_FOUND, "no audio for session").into_response();
    };
    let channel = match Channel::parse(&q.channel) {
        Some(c) => c,
        None => return (StatusCode::BAD_REQUEST, "invalid channel").into_response(),
    };
    let sid = sid.to_string();
    let joined = tokio::task::spawn_blocking(move || {
        try_live_slice(&sid, channel, q.from_ms, q.to_ms)
            .or_else(|| try_disk_slice(&sid, channel, q.from_ms, q.to_ms))
    })
    .await;
    let pcm: Vec<u8> = match joined {
        Ok(Some(b)) => b,
        Ok(None) => return (StatusCode::NOT_FOUND, "no audio for session").into_response(),
        Err(err) => {
            warn!(error = %err, "audio slice join");
            return (StatusCode::INTERNAL_SERVER_ERROR, "read failed").into_response();
        }
    };
    let num_samples = (pcm.len() / 2) as u64;
    let mut body = wav_header(num_samples, channel.sample_rate());
    body.extend_from_slice(&pcm);
    Response::builder()
        .status(StatusCode::OK)
        .header(header::CONTENT_TYPE, "audio/wav")
        .body(Body::from(body))
        .unwrap_or_else(|_| StatusCode::INTERNAL_SERVER_ERROR.into_response())
}

fn try_live_slice(sid: &str, channel: Channel, from_ms: u64, to_ms: u64) -> Option<Vec<u8>> {
    let session = crate::realtime::lookup_session_pub(sid)?;
    Some(session.audio_store.slice(channel, from_ms, to_ms))
}

fn byte_range(from_ms: u64, to_ms: u64, sample_rate: u32) -> (u64, Option<u64>) {
    let sr = sample_rate as u64;
    let start = from_ms.saturating_mul(sr).saturating_mul(2) / 1000;
    let end = if to_ms == 0 {
        None
    } else {
        Some(to_ms.saturating_mul(sr).saturating_mul(2) / 1000)
    };
    (start, end)
}

fn read_range(file: &mut std::fs::File, start: u64, end: Option<u64>) -> std::io::Result<Vec<u8>> {
    file.seek(SeekFrom::Start(start))?;
    match end {
        None => {
            let mut buf = Vec::new();
            file.read_to_end(&mut buf)?;
            Ok(buf)
        }
        Some(end) => {
            let n = end.saturating_sub(start).min(u32::MAX as u64) as usize;
            let mut buf = vec![0u8; n];
            let mut filled = 0usize;
            while filled < n {
                match file.read(&mut buf[filled..]) {
                    Ok(0) => break,
                    Ok(read) => filled += read,
                    Err(err) if err.kind() == std::io::ErrorKind::Interrupted => continue,
                    Err(err) => return Err(err),
                }
            }
            buf.truncate(filled);
            Ok(buf)
        }
    }
}

fn try_disk_slice(sid: &str, channel: Channel, from_ms: u64, to_ms: u64) -> Option<Vec<u8>> {
    let sd = session_dir()?;
    let raw = sd.join(format!("{}.audio_{}.raw", sid, channel.as_str()));
    let mut fh = std::fs::File::open(&raw).ok()?;
    let sidecar = sd.join(format!("{}.audio.json", sid));
    let offset_ms = std::fs::read_to_string(&sidecar)
        .ok()
        .and_then(|s| serde_json::from_str::<serde_json::Value>(&s).ok())
        .and_then(|v| {
            v.get("tracks")
                .and_then(|t| t.get(channel.as_str()))
                .and_then(|t| t.get("offset_ms"))
                .and_then(|v| v.as_u64())
        })
        .unwrap_or(0);
    let adj_from = from_ms.saturating_sub(offset_ms);
    let adj_to = if to_ms > 0 {
        to_ms.saturating_sub(offset_ms)
    } else {
        0
    };
    let (start, end) = byte_range(adj_from, adj_to, channel.sample_rate());
    if end.is_some_and(|e| e <= start) {
        return Some(Vec::new());
    }
    match read_range(&mut fh, start, end) {
        Ok(pcm) => Some(pcm),
        Err(err) => {
            warn!(error = %err, path = %raw.display(), "range read audio track");
            Some(Vec::new())
        }
    }
}

pub async fn inspect_stream_ws(Path(sid): Path<String>, ws: WebSocketUpgrade) -> Response {
    if sanitize_sid(&sid).is_none() {
        return (StatusCode::NOT_FOUND, "session not found").into_response();
    }
    ws.on_upgrade(move |socket| inspect_stream_loop(socket, sid))
}

async fn inspect_stream_loop(mut socket: WebSocket, sid: String) {
    let relay = registry::get_relay(&sid);
    if relay.is_none() {
        replay_history_to_socket(&mut socket, &sid).await;
        let _ = socket.close().await;
        return;
    }
    let relay = relay.unwrap();
    let sub = relay.subscribe();
    for line in sub.snapshot {
        if !send_line(&mut socket, &line).await {
            return;
        }
    }
    let mut rx = sub.rx;
    loop {
        tokio::select! {
            line = rx.recv() => {
                match line {
                    Ok(line) => {
                        if !send_line(&mut socket, &line).await {
                            return;
                        }
                    }
                    Err(tokio::sync::broadcast::error::RecvError::Lagged(_)) => continue,
                    Err(tokio::sync::broadcast::error::RecvError::Closed) => {
                        let _ = socket.close().await;
                        return;
                    }
                }
            }
            msg = socket.recv() => {
                match msg {
                    Some(Ok(Message::Close(_))) | None | Some(Err(_)) => {
                        return;
                    }
                    Some(Ok(_)) => {}
                }
            }
        }
    }
}

async fn replay_history_to_socket(socket: &mut WebSocket, sid: &str) {
    let Some(sd) = session_dir() else {
        return;
    };
    let path: PathBuf = sd.join(format!("{}.ndjson", sid));
    let (tx, mut rx) = tokio::sync::mpsc::channel::<Vec<u8>>(64);
    tokio::task::spawn_blocking(move || {
        let Ok(file) = std::fs::File::open(&path) else {
            return;
        };
        let mut reader = std::io::BufReader::with_capacity(STREAM_CHUNK, file);
        let mut line: Vec<u8> = Vec::new();
        loop {
            line.clear();
            match reader.read_until(b'\n', &mut line) {
                Ok(0) | Err(_) => return,
                Ok(_) => {}
            }
            let trimmed = match line.last() {
                Some(b'\n') => &line[..line.len() - 1],
                _ => &line[..],
            };
            if trimmed.is_empty() {
                continue;
            }
            if tx.blocking_send(trimmed.to_vec()).is_err() {
                return;
            }
        }
    });
    while let Some(line) = rx.recv().await {
        if !send_line(socket, &line).await {
            return;
        }
    }
}

async fn send_line(socket: &mut WebSocket, line: &[u8]) -> bool {
    let trimmed = if line.last() == Some(&b'\n') {
        &line[..line.len() - 1]
    } else {
        line
    };
    socket
        .send(Message::Binary(trimmed.to_vec().into()))
        .await
        .is_ok()
}

pub fn _unused_force_link() {
    let _ = retention_count();
    let _ = retention_bytes();
    let _ = retention_days();
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    fn temp_file(bytes: &[u8]) -> (PathBuf, std::fs::File) {
        let p = std::env::temp_dir().join(format!(
            "speaches-plus-inspect-{}-{}.raw",
            std::process::id(),
            uuid::Uuid::new_v4().simple()
        ));
        let mut f = std::fs::File::create(&p).unwrap();
        f.write_all(bytes).unwrap();
        f.flush().unwrap();
        (p.clone(), std::fs::File::open(&p).unwrap())
    }

    #[test]
    fn sanitize_sid_rejects_traversal_and_overlong() {
        assert_eq!(sanitize_sid("sess_abc-1"), Some("sess_abc-1"));
        assert_eq!(sanitize_sid(""), None);
        assert_eq!(sanitize_sid("../etc/passwd"), None);
        assert_eq!(sanitize_sid(&"a".repeat(65)), None);
    }

    #[test]
    fn byte_range_matches_pcm16_math() {
        assert_eq!(byte_range(0, 0, 16_000), (0, None));
        assert_eq!(byte_range(100, 200, 16_000), (3200, Some(6400)));
        assert_eq!(byte_range(1000, 0, 24_000), (48_000, None));
        let (s, e) = byte_range(u64::MAX, u64::MAX, 24_000);
        assert!(s > 0 && e.is_some());
    }

    #[test]
    fn read_range_open_ended_reads_to_eof() {
        let data: Vec<u8> = (0..200u32).map(|i| i as u8).collect();
        let (path, mut fh) = temp_file(&data);
        let got = read_range(&mut fh, 50, None).unwrap();
        assert_eq!(got, data[50..]);
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn read_range_bounded_truncates_at_eof() {
        let data: Vec<u8> = (0..200u32).map(|i| i as u8).collect();
        let (path, mut fh) = temp_file(&data);
        let got = read_range(&mut fh, 100, Some(150)).unwrap();
        assert_eq!(got, data[100..150]);
        let past = read_range(&mut fh, 190, Some(400)).unwrap();
        assert_eq!(past, data[190..]);
        let beyond = read_range(&mut fh, 500, Some(600)).unwrap();
        assert!(beyond.is_empty());
        let _ = std::fs::remove_file(&path);
    }
}
