#![allow(dead_code)]

pub mod audio_store;
pub mod constants;
pub mod registry;
pub mod relay;
pub mod retention;
pub mod routes;
pub mod types;

pub use audio_store::AudioStore;
pub use registry::{register, unregister};
pub use relay::InspectorRelay;
pub use retention::cleanup_on_startup;
pub use types::Corr;

use std::path::PathBuf;
use std::sync::OnceLock;

use super::defaults;

pub fn session_dir() -> Option<PathBuf> {
    static CACHED: OnceLock<Option<PathBuf>> = OnceLock::new();
    CACHED
        .get_or_init(|| {
            std::env::var(defaults::env::INSPECT_SESSION_DIR)
                .ok()
                .map(|raw| expand_home(&raw))
        })
        .clone()
}

pub fn retention_count() -> usize {
    std::env::var(defaults::env::INSPECT_RETENTION_COUNT)
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(defaults::inspect::RETENTION_COUNT)
}

pub fn retention_bytes() -> u64 {
    std::env::var(defaults::env::INSPECT_RETENTION_BYTES)
        .ok()
        .and_then(|s| s.parse::<u64>().ok())
        .unwrap_or(defaults::inspect::RETENTION_BYTES)
}

pub fn retention_days() -> u64 {
    std::env::var(defaults::env::INSPECT_RETENTION_DAYS)
        .ok()
        .and_then(|s| s.parse::<u64>().ok())
        .unwrap_or(defaults::inspect::RETENTION_DAYS)
}

pub fn run_startup_cleanup() {
    static DONE: OnceLock<()> = OnceLock::new();
    DONE.get_or_init(|| {
        if let Some(dir) = session_dir() {
            cleanup_on_startup(&dir, retention_count(), retention_bytes(), retention_days());
        }
    });
}

fn expand_home(input: &str) -> PathBuf {
    if let Some(rest) = input.strip_prefix("~/") {
        if let Some(home) = std::env::var_os("HOME") {
            return PathBuf::from(home).join(rest);
        }
    }
    if input == "~" {
        if let Some(home) = std::env::var_os("HOME") {
            return PathBuf::from(home);
        }
    }
    PathBuf::from(input)
}
