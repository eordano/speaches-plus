#![allow(dead_code)]

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use uuid::Uuid;

use super::types::{EventId, ItemId, ResponseId, SessionId};

pub trait IdSource: Send + Sync {
    fn session(&self) -> SessionId;
    fn item(&self) -> ItemId;
    fn response(&self) -> ResponseId;
    fn event(&self) -> EventId;
}

pub struct RandomIdSource;

impl IdSource for RandomIdSource {
    fn session(&self) -> SessionId {
        SessionId::new(format!("sess_{}", Uuid::new_v4().simple()))
    }
    fn item(&self) -> ItemId {
        ItemId::new(format!("item_{}", Uuid::new_v4().simple()))
    }
    fn response(&self) -> ResponseId {
        ResponseId::new(format!("resp_{}", Uuid::new_v4().simple()))
    }
    fn event(&self) -> EventId {
        EventId::new(format!("evt_{}", Uuid::new_v4().simple()))
    }
}

#[derive(Default)]
pub struct CounterIdSource {
    session: AtomicU64,
    item: AtomicU64,
    response: AtomicU64,
    event: AtomicU64,
}

impl CounterIdSource {
    pub fn new() -> Self {
        Self::default()
    }
}

impl IdSource for CounterIdSource {
    fn session(&self) -> SessionId {
        SessionId::new(format!(
            "sess_{:024}",
            self.session.fetch_add(1, Ordering::Relaxed)
        ))
    }
    fn item(&self) -> ItemId {
        ItemId::new(format!(
            "item_{:024}",
            self.item.fetch_add(1, Ordering::Relaxed)
        ))
    }
    fn response(&self) -> ResponseId {
        ResponseId::new(format!(
            "resp_{:024}",
            self.response.fetch_add(1, Ordering::Relaxed)
        ))
    }
    fn event(&self) -> EventId {
        EventId::new(format!(
            "evt_{:024}",
            self.event.fetch_add(1, Ordering::Relaxed)
        ))
    }
}

pub fn default_source() -> Arc<dyn IdSource> {
    Arc::new(RandomIdSource)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn random_ids_are_unique_and_prefixed() {
        let src = RandomIdSource;
        let a = src.item();
        let b = src.item();
        assert!(a.as_str().starts_with("item_"));
        assert!(b.as_str().starts_with("item_"));
        assert_ne!(a.as_str(), b.as_str());
        assert!(src.session().as_str().starts_with("sess_"));
        assert!(src.response().as_str().starts_with("resp_"));
        assert!(src.event().as_str().starts_with("evt_"));
    }

    #[test]
    fn counter_ids_are_deterministic() {
        let src = CounterIdSource::new();
        assert_eq!(src.item().as_str(), format!("item_{:024}", 0));
        assert_eq!(src.item().as_str(), format!("item_{:024}", 1));
        assert_eq!(src.response().as_str(), format!("resp_{:024}", 0));
        assert_eq!(src.session().as_str(), format!("sess_{:024}", 0));
        assert_eq!(src.event().as_str(), format!("evt_{:024}", 0));
    }

    #[test]
    fn counters_are_per_kind_independent() {
        let src = CounterIdSource::new();
        for _ in 0..3 {
            src.item();
        }
        assert_eq!(src.response().as_str(), format!("resp_{:024}", 0));
    }
}
