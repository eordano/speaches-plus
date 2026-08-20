#![allow(dead_code)]

use std::collections::BTreeMap;
use std::fs::{File, OpenOptions};
use std::io::Write;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Mutex;
use std::time::SystemTime;

use serde_json::Value;
use tokio::sync::broadcast;
use tracing::warn;

use super::constants::is_error_kind;
use super::types::{Corr, WireEvent};
use crate::defaults;
use crate::otel;

const FANOUT_CAP: usize = defaults::inspector::RELAY_CAP as usize;
const REPLAY_CAP: usize = (defaults::inspector::RELAY_CAP * 4) as usize;

pub struct InspectorRelay {
    pub session_id: String,
    pub session_dir: Option<PathBuf>,
    seq: AtomicU64,
    inner: Mutex<RelayInner>,
    tx: broadcast::Sender<Vec<u8>>,
}

struct RelayInner {
    replay_buffer: std::collections::VecDeque<Vec<u8>>,
    ndjson: Option<File>,
    turn_count: u64,
    last_event_ts: Option<f64>,
    dropped_count: u64,
    turn_id: Option<String>,
    item_id: Option<String>,
    response_id: Option<String>,
    phrase_id: Option<String>,
}

impl InspectorRelay {
    pub fn new(session_id: String, session_dir: Option<PathBuf>) -> Self {
        let ndjson = session_dir.as_ref().and_then(|dir| {
            let _ = std::fs::create_dir_all(dir);
            let ndjson_path = dir.join(format!("{}.ndjson", session_id));
            OpenOptions::new()
                .create(true)
                .append(true)
                .open(&ndjson_path)
                .map_err(|err| {
                    warn!(error = %err, path = %ndjson_path.display(), "open inspector ndjson failed");
                    err
                })
                .ok()
        });
        let (tx, _rx) = broadcast::channel::<Vec<u8>>(FANOUT_CAP);
        Self {
            session_id,
            session_dir,
            seq: AtomicU64::new(0),
            inner: Mutex::new(RelayInner {
                replay_buffer: std::collections::VecDeque::with_capacity(REPLAY_CAP),
                ndjson,
                turn_count: 0,
                last_event_ts: None,
                dropped_count: 0,
                turn_id: None,
                item_id: None,
                response_id: None,
                phrase_id: None,
            }),
            tx,
        }
    }

    pub fn next_seq(&self) -> u64 {
        self.seq.fetch_add(1, Ordering::Relaxed)
    }

    pub fn turn_count(&self) -> u64 {
        self.inner.lock().expect("relay poisoned").turn_count
    }

    pub fn last_event_ts(&self) -> Option<f64> {
        self.inner.lock().expect("relay poisoned").last_event_ts
    }

    pub fn dropped_count(&self) -> u64 {
        self.inner.lock().expect("relay poisoned").dropped_count
    }

    pub fn corr(&self) -> Corr {
        let g = self.inner.lock().expect("relay poisoned");
        Corr {
            turn_id: g.turn_id.clone(),
            item_id: g.item_id.clone(),
            response_id: g.response_id.clone(),
            phrase_id: g.phrase_id.clone(),
        }
    }

    pub fn set_turn_id(&self, v: Option<String>) {
        self.inner.lock().expect("relay poisoned").turn_id = v;
    }
    pub fn set_item_id(&self, v: Option<String>) {
        self.inner.lock().expect("relay poisoned").item_id = v;
    }
    pub fn set_response_id(&self, v: Option<String>) {
        self.inner.lock().expect("relay poisoned").response_id = v;
    }
    pub fn set_phrase_id(&self, v: Option<String>) {
        self.inner.lock().expect("relay poisoned").phrase_id = v;
    }

    pub fn subscriber_count(&self) -> usize {
        self.tx.receiver_count()
    }

    pub fn subscribe(&self) -> InspectorSubscription {
        let snapshot = {
            let g = self.inner.lock().expect("relay poisoned");
            g.replay_buffer.iter().cloned().collect::<Vec<_>>()
        };
        InspectorSubscription {
            snapshot,
            rx: self.tx.subscribe(),
        }
    }

    pub fn publish(
        &self,
        lane: &str,
        kind: &str,
        corr_override: Option<Corr>,
        payload: BTreeMap<String, Value>,
    ) {
        let now = SystemTime::now();
        let ts_wall = now
            .duration_since(SystemTime::UNIX_EPOCH)
            .map(|d| d.as_secs_f64())
            .unwrap_or(0.0);
        let ts_mono_ns = ts_wall_to_monotonic_ns();
        let seq = self.next_seq();
        let merged_corr = self.merge_corr(corr_override);

        let event = WireEvent {
            session_id: self.session_id.clone(),
            seq,
            ts_mono_ns,
            ts_wall,
            lane: lane.to_string(),
            kind: kind.to_string(),
            corr: merged_corr,
            span_id: otel::current_span_id_hex(),
            payload,
        };

        let mut line = match serde_json::to_vec(&event) {
            Ok(b) => b,
            Err(err) => {
                warn!(error = %err, "serialize inspector event");
                return;
            }
        };
        line.push(b'\n');

        let mirror = if lane != "error" && is_error_kind(kind) {
            Some(self.build_error_mirror(&event))
        } else {
            None
        };

        {
            let mut g = self.inner.lock().expect("relay poisoned");
            if lane == "turn" && kind == "turn_end" {
                g.turn_count += 1;
            }
            g.last_event_ts = Some(ts_wall);
            push_replay(&mut g.replay_buffer, line.clone());
            if let Some(fh) = g.ndjson.as_mut() {
                if let Err(err) = fh.write_all(&line) {
                    warn!(error = %err, "write inspector ndjson");
                }
            }
        }
        let _ = self.tx.send(line);

        if let Some(mirror_event) = mirror {
            let mirror_seq = self.next_seq();
            let mirror_event = WireEvent {
                seq: mirror_seq,
                ..mirror_event
            };
            if let Ok(mut mline) = serde_json::to_vec(&mirror_event) {
                mline.push(b'\n');
                let mut g = self.inner.lock().expect("relay poisoned");
                push_replay(&mut g.replay_buffer, mline.clone());
                if let Some(fh) = g.ndjson.as_mut() {
                    let _ = fh.write_all(&mline);
                }
                drop(g);
                let _ = self.tx.send(mline);
            }
        }
    }

    fn merge_corr(&self, override_corr: Option<Corr>) -> Corr {
        let base = self.corr();
        match override_corr {
            None => base,
            Some(o) => Corr {
                turn_id: o.turn_id.or(base.turn_id),
                item_id: o.item_id.or(base.item_id),
                response_id: o.response_id.or(base.response_id),
                phrase_id: o.phrase_id.or(base.phrase_id),
            },
        }
    }

    fn build_error_mirror(&self, origin: &WireEvent) -> WireEvent {
        let mut payload: BTreeMap<String, Value> = BTreeMap::new();
        payload.insert("lane".into(), Value::String(origin.lane.clone()));
        payload.insert("origin_seq".into(), Value::from(origin.seq));
        payload.insert("origin_kind".into(), Value::String(origin.kind.clone()));
        let error_text = origin
            .payload
            .get("error")
            .or_else(|| origin.payload.get("reason"))
            .map(|v| match v {
                Value::String(s) => s.clone(),
                other => other.to_string(),
            })
            .unwrap_or_else(|| origin.kind.clone());
        payload.insert("error".into(), Value::String(error_text));
        payload.insert("severity".into(), Value::String("error".into()));
        WireEvent {
            session_id: origin.session_id.clone(),
            seq: 0,
            ts_mono_ns: origin.ts_mono_ns,
            ts_wall: origin.ts_wall,
            lane: "error".into(),
            kind: "raised".into(),
            corr: origin.corr.clone(),
            span_id: origin.span_id.clone(),
            payload,
        }
    }

    pub fn close(&self) {
        let mut g = self.inner.lock().expect("relay poisoned");
        if let Some(fh) = g.ndjson.as_mut() {
            let _ = fh.flush();
        }
        g.ndjson = None;
    }
}

pub struct InspectorSubscription {
    pub snapshot: Vec<Vec<u8>>,
    pub rx: broadcast::Receiver<Vec<u8>>,
}

fn push_replay(buf: &mut std::collections::VecDeque<Vec<u8>>, line: Vec<u8>) {
    if buf.len() >= REPLAY_CAP {
        buf.pop_front();
    }
    buf.push_back(line);
}

fn ts_wall_to_monotonic_ns() -> u128 {
    use std::time::Instant;
    static START: std::sync::OnceLock<Instant> = std::sync::OnceLock::new();
    let start = *START.get_or_init(Instant::now);
    start.elapsed().as_nanos()
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn temp_dir() -> PathBuf {
        let p = std::env::temp_dir().join(format!(
            "speaches-plus-relay-{}-{}",
            std::process::id(),
            uuid::Uuid::new_v4().simple()
        ));
        std::fs::create_dir_all(&p).unwrap();
        p
    }

    #[test]
    fn publish_writes_ndjson_and_replays_to_subscriber() {
        let dir = temp_dir();
        let relay = InspectorRelay::new("sess_test".into(), Some(dir.clone()));
        let mut payload = BTreeMap::new();
        payload.insert("audio_start_ms".into(), json!(100));
        relay.publish("vad", "confirmed_start", None, payload);

        let path = dir.join("sess_test.ndjson");
        let body = std::fs::read_to_string(&path).unwrap();
        assert!(body.contains("\"lane\":\"vad\""));
        assert!(body.contains("\"kind\":\"confirmed_start\""));
        assert!(body.contains("\"audio_start_ms\":100"));

        let sub = relay.subscribe();
        assert_eq!(sub.snapshot.len(), 1);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn error_mirror_emits_when_kind_is_in_err_set() {
        let dir = temp_dir();
        let relay = InspectorRelay::new("sess_err".into(), Some(dir.clone()));
        let mut payload = BTreeMap::new();
        payload.insert("error".into(), json!("kaboom"));
        relay.publish("llm", "failed", None, payload);

        let path = dir.join("sess_err.ndjson");
        let body = std::fs::read_to_string(&path).unwrap();
        let lines: Vec<&str> = body.lines().collect();
        assert_eq!(lines.len(), 2, "expected origin + mirror");
        assert!(lines[0].contains("\"lane\":\"llm\""));
        assert!(lines[1].contains("\"lane\":\"error\""));
        assert!(lines[1].contains("\"origin_kind\":\"failed\""));
        assert!(lines[1].contains("\"kaboom\""));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn corr_merges_with_override() {
        let dir = temp_dir();
        let relay = InspectorRelay::new("sess_corr".into(), Some(dir.clone()));
        relay.set_turn_id(Some("t1".into()));
        relay.set_item_id(Some("i1".into()));
        let corr_override = Corr {
            item_id: Some("i_override".into()),
            ..Default::default()
        };
        relay.publish(
            "turn",
            "user_committed",
            Some(corr_override),
            BTreeMap::new(),
        );

        let path = dir.join("sess_corr.ndjson");
        let body = std::fs::read_to_string(&path).unwrap();
        assert!(body.contains("\"turn_id\":\"t1\""));
        assert!(body.contains("\"item_id\":\"i_override\""));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn turn_count_tracks_turn_end_kind() {
        let dir = temp_dir();
        let relay = InspectorRelay::new("sess_turn".into(), Some(dir.clone()));
        assert_eq!(relay.turn_count(), 0);
        relay.publish("turn", "turn_start", None, BTreeMap::new());
        assert_eq!(relay.turn_count(), 0);
        relay.publish("turn", "turn_end", None, BTreeMap::new());
        assert_eq!(relay.turn_count(), 1);
        relay.publish("turn", "turn_end", None, BTreeMap::new());
        assert_eq!(relay.turn_count(), 2);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn subscriber_gets_replay_snapshot_first() {
        let dir = temp_dir();
        let relay = InspectorRelay::new("sess_replay".into(), Some(dir.clone()));
        for i in 0..3 {
            let mut p = BTreeMap::new();
            p.insert("i".into(), json!(i));
            relay.publish("turn", "turn_start", None, p);
        }
        let sub = relay.subscribe();
        assert_eq!(sub.snapshot.len(), 3);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn replay_buffer_caps_at_replay_cap() {
        let dir = temp_dir();
        let relay = InspectorRelay::new("sess_cap".into(), Some(dir.clone()));
        for _ in 0..(REPLAY_CAP + 100) {
            relay.publish("turn", "turn_start", None, BTreeMap::new());
        }
        let sub = relay.subscribe();
        assert_eq!(sub.snapshot.len(), REPLAY_CAP);
        let _ = std::fs::remove_dir_all(&dir);
    }
}
