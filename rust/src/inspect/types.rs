#![allow(dead_code)]

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};
use serde_json::Value;

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct Corr {
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub turn_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub item_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub response_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub phrase_id: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WireEvent {
    pub session_id: String,
    pub seq: u64,
    pub ts_mono_ns: u128,
    pub ts_wall: f64,
    pub lane: String,
    pub kind: String,
    pub corr: Corr,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub span_id: Option<String>,
    pub payload: BTreeMap<String, Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionMeta {
    pub id: String,
    pub created_at: f64,
    pub model: String,
    pub state: String,
    pub turn_count: u64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub last_event_ts: Option<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionHistoryEntry {
    pub id: String,
    pub size_bytes: u64,
    pub mtime: f64,
}
