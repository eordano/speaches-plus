#![allow(dead_code)]

use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};
use std::time::SystemTime;

use super::relay::InspectorRelay;
use super::types::SessionMeta;

struct Entry {
    relay: Arc<InspectorRelay>,
    created_at: f64,
    model: String,
    state: Box<dyn Fn() -> String + Send + Sync>,
}

static REGISTRY: OnceLock<Mutex<HashMap<String, Entry>>> = OnceLock::new();

fn registry() -> &'static Mutex<HashMap<String, Entry>> {
    REGISTRY.get_or_init(|| Mutex::new(HashMap::new()))
}

pub fn register(
    session_id: &str,
    relay: Arc<InspectorRelay>,
    model: impl Into<String>,
    state_fn: impl Fn() -> String + Send + Sync + 'static,
) {
    let created_at = SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .map(|d| d.as_secs_f64())
        .unwrap_or(0.0);
    let entry = Entry {
        relay,
        created_at,
        model: model.into(),
        state: Box::new(state_fn),
    };
    registry()
        .lock()
        .expect("inspect registry poisoned")
        .insert(session_id.to_string(), entry);
}

pub fn unregister(session_id: &str) {
    let entry = registry()
        .lock()
        .expect("inspect registry poisoned")
        .remove(session_id);
    if let Some(e) = entry {
        e.relay.close();
    }
}

pub fn get_relay(session_id: &str) -> Option<Arc<InspectorRelay>> {
    let g = registry().lock().expect("inspect registry poisoned");
    g.get(session_id).map(|e| e.relay.clone())
}

pub fn list_meta() -> Vec<SessionMeta> {
    let g = registry().lock().expect("inspect registry poisoned");
    g.iter()
        .map(|(id, e)| SessionMeta {
            id: id.clone(),
            created_at: e.created_at,
            model: e.model.clone(),
            state: (e.state)(),
            turn_count: e.relay.turn_count(),
            last_event_ts: e.relay.last_event_ts(),
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn temp_dir() -> PathBuf {
        let p = std::env::temp_dir().join(format!(
            "speaches-plus-reg-{}-{}",
            std::process::id(),
            uuid::Uuid::new_v4().simple()
        ));
        std::fs::create_dir_all(&p).unwrap();
        p
    }

    #[test]
    fn register_and_lookup_round_trip() {
        let dir = temp_dir();
        let id = format!("sess_reg_{}", uuid::Uuid::new_v4().simple());
        let relay = Arc::new(InspectorRelay::new(id.clone(), Some(dir.clone())));
        register(&id, relay.clone(), "gpt-test", || "active".into());
        assert!(get_relay(&id).is_some());
        let metas = list_meta();
        assert!(metas.iter().any(|m| m.id == id && m.model == "gpt-test"));
        unregister(&id);
        assert!(get_relay(&id).is_none());
        let _ = std::fs::remove_dir_all(&dir);
    }
}
