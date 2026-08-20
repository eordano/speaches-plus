#![allow(dead_code)]

use std::collections::HashMap;

use serde_json::Value;

const TS_FIELDS: &[&str] = &["ts_ms", "created_at", "audio_start_ms", "audio_end_ms"];
const FLOAT_FIELDS: &[&str] = &[
    "score",
    "eou.score",
    "eou.eager_score",
    "eou.threshold",
    "vad.probability",
];

const ID_PREFIXES: &[(&str, &str)] = &[
    ("sess_", "sess"),
    ("item_", "item"),
    ("resp_", "resp"),
    ("evt_", "evt"),
];

#[derive(Clone, Debug)]
pub struct CanonicalTrace {
    pub events: Vec<Value>,
}

impl CanonicalTrace {
    pub fn len(&self) -> usize {
        self.events.len()
    }

    pub fn is_empty(&self) -> bool {
        self.events.is_empty()
    }
}

pub fn canonicalize_trace(trace: &[Value]) -> CanonicalTrace {
    let mut id_map: HashMap<String, String> = HashMap::new();
    let mut counters: HashMap<&'static str, usize> = HashMap::new();

    let mut out = Vec::with_capacity(trace.len());
    for ev in trace.iter() {
        let mut cloned = ev.clone();
        canonicalize_node(&mut cloned, &mut id_map, &mut counters);
        out.push(cloned);
    }
    CanonicalTrace { events: out }
}

fn canonicalize_node(
    v: &mut Value,
    id_map: &mut HashMap<String, String>,
    counters: &mut HashMap<&'static str, usize>,
) {
    match v {
        Value::Object(map) => {
            let keys: Vec<String> = map.keys().cloned().collect();
            for k in keys {
                if let Some(child) = map.get_mut(&k) {
                    if TS_FIELDS.iter().any(|f| *f == k) {
                        if let Some(_n) = child.as_u64() {
                            *child = Value::from(0u64);
                        }
                        continue;
                    }
                    if FLOAT_FIELDS.iter().any(|f| *f == k) {
                        if let Some(f) = child.as_f64() {
                            let r = (f * 1000.0).round() / 1000.0;
                            *child = serde_json::Number::from_f64(r)
                                .map(Value::Number)
                                .unwrap_or(Value::Null);
                        }
                        continue;
                    }
                    if let Value::String(s) = child {
                        if let Some(canon) = canon_id(s, id_map, counters) {
                            *s = canon;
                        }
                        continue;
                    }
                    if k == "audio" || k == "data" {
                        if let Value::String(s) = child {
                            let n = s.len();
                            *child = Value::from(format!("<{n} bytes>"));
                            continue;
                        }
                    }
                    canonicalize_node(child, id_map, counters);
                }
            }
        }
        Value::Array(items) => {
            for it in items.iter_mut() {
                canonicalize_node(it, id_map, counters);
            }
        }
        Value::String(s) => {
            if let Some(canon) = canon_id(s, id_map, counters) {
                *s = canon;
            }
        }
        _ => {}
    }
}

fn canon_id(
    s: &str,
    id_map: &mut HashMap<String, String>,
    counters: &mut HashMap<&'static str, usize>,
) -> Option<String> {
    for (prefix, kind) in ID_PREFIXES {
        if s.starts_with(prefix) {
            if let Some(c) = id_map.get(s) {
                return Some(c.clone());
            }
            let n = counters.entry(kind).or_insert(0);
            *n += 1;
            let canon = format!("{kind}_{n}");
            id_map.insert(s.to_string(), canon.clone());
            return Some(canon);
        }
    }
    None
}

pub fn trace_diff(a: &CanonicalTrace, b: &CanonicalTrace) -> Option<usize> {
    let n = a.events.len().min(b.events.len());
    for i in 0..n {
        if a.events[i] != b.events[i] {
            return Some(i);
        }
    }
    if a.events.len() != b.events.len() {
        return Some(n);
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn canonicalize_replaces_random_ids_with_counters() {
        let trace = vec![
            json!({"type": "session.created", "session": {"id": "sess_abc"}}),
            json!({"type": "input_audio_buffer.speech_started", "item_id": "item_xy"}),
            json!({"type": "input_audio_buffer.speech_stopped", "item_id": "item_xy"}),
            json!({"type": "response.created", "response": {"id": "resp_q"}}),
        ];
        let c = canonicalize_trace(&trace);
        assert_eq!(c.events[0]["session"]["id"], "sess_1");
        assert_eq!(c.events[1]["item_id"], "item_1");
        assert_eq!(c.events[2]["item_id"], "item_1");
        assert_eq!(c.events[3]["response"]["id"], "resp_1");
    }

    #[test]
    fn canonicalize_strips_timestamps() {
        let trace = vec![json!({
            "type": "input_audio_buffer.speech_stopped",
            "audio_end_ms": 12345,
            "ts_ms": 99999
        })];
        let c = canonicalize_trace(&trace);
        assert_eq!(c.events[0]["audio_end_ms"], 0);
        assert_eq!(c.events[0]["ts_ms"], 0);
    }

    #[test]
    fn canonicalize_rounds_floats() {
        let trace = vec![json!({
            "type": "eou.scored",
            "score": 0.123456789_f64,
            "threshold": 0.5_f64
        })];
        let c = canonicalize_trace(&trace);
        let s = c.events[0]["score"].as_f64().unwrap();
        assert!((s - 0.123).abs() < 1e-6, "got {s}");
    }

    #[test]
    fn trace_diff_returns_first_diverging_index() {
        let a = canonicalize_trace(&[
            json!({"type": "session.created", "session": {"id": "sess_1"}}),
            json!({"type": "response.created", "response": {"id": "resp_1"}}),
            json!({"type": "response.done", "response": {"id": "resp_1", "status": "completed"}}),
        ]);
        let b = canonicalize_trace(&[
            json!({"type": "session.created", "session": {"id": "sess_z"}}),
            json!({"type": "response.created", "response": {"id": "resp_z"}}),
            json!({"type": "response.done", "response": {"id": "resp_z", "status": "cancelled"}}),
        ]);
        assert_eq!(trace_diff(&a, &b), Some(2));
    }

    #[test]
    fn trace_diff_returns_none_for_identical() {
        let a = canonicalize_trace(&[json!({"type": "x", "id": "evt_a"})]);
        let b = canonicalize_trace(&[json!({"type": "x", "id": "evt_b"})]);
        assert_eq!(trace_diff(&a, &b), None);
    }
}
