#![allow(dead_code)]

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::time::SystemTime;

use tracing::warn;

pub fn cleanup_on_startup(session_dir: &Path, max_count: usize, max_bytes: u64, max_days: u64) {
    if !session_dir.exists() {
        return;
    }
    let entries = match std::fs::read_dir(session_dir) {
        Ok(e) => e,
        Err(err) => {
            warn!(error = %err, path = %session_dir.display(), "read session dir");
            return;
        }
    };
    let mut sessions: HashMap<String, Vec<PathBuf>> = HashMap::new();
    for entry in entries.flatten() {
        let path = entry.path();
        if !path.is_file() {
            continue;
        }
        let ext = match path.extension().and_then(|e| e.to_str()) {
            Some(e) => e,
            None => continue,
        };
        if !matches!(ext, "ndjson" | "raw" | "json") {
            continue;
        }
        let stem = match path.file_name().and_then(|n| n.to_str()) {
            Some(n) => n.split('.').next().unwrap_or("").to_string(),
            None => continue,
        };
        if stem.is_empty() {
            continue;
        }
        sessions.entry(stem).or_default().push(path);
    }

    let now = SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    let max_age_s = if max_days > 0 {
        Some(max_days.saturating_mul(86_400))
    } else {
        None
    };

    if let Some(max_age) = max_age_s {
        let to_remove: Vec<String> = sessions
            .iter()
            .filter_map(|(sid, paths)| {
                let mt = paths.iter().map(|p| file_mtime(p)).max().unwrap_or(0);
                if now.saturating_sub(mt) > max_age {
                    Some(sid.clone())
                } else {
                    None
                }
            })
            .collect();
        for sid in to_remove {
            if let Some(paths) = sessions.remove(&sid) {
                for p in paths {
                    unlink(&p);
                }
            }
        }
    }

    if max_count > 0 && sessions.len() > max_count {
        let mut ordered: Vec<(String, Vec<PathBuf>)> = sessions.into_iter().collect();
        ordered.sort_by_key(|(_, paths)| {
            std::cmp::Reverse(paths.iter().map(|p| file_mtime(p)).max().unwrap_or(0))
        });
        let keep: Vec<(String, Vec<PathBuf>)> = ordered.drain(..max_count).collect();
        for (_sid, paths) in ordered {
            for p in paths {
                unlink(&p);
            }
        }
        sessions = keep.into_iter().collect();
    }

    if max_bytes > 0 {
        let mut ordered: Vec<(String, Vec<PathBuf>)> = sessions.into_iter().collect();
        ordered.sort_by_key(|(_, paths)| {
            std::cmp::Reverse(paths.iter().map(|p| file_mtime(p)).max().unwrap_or(0))
        });
        let mut running: u64 = 0;
        for (_sid, paths) in ordered {
            let size: u64 = paths.iter().map(|p| file_size(p)).sum();
            if running.saturating_add(size) > max_bytes {
                for p in paths {
                    unlink(&p);
                }
            } else {
                running = running.saturating_add(size);
            }
        }
    }
}

fn file_mtime(p: &Path) -> u64 {
    p.metadata()
        .ok()
        .and_then(|m| m.modified().ok())
        .and_then(|t| t.duration_since(SystemTime::UNIX_EPOCH).ok())
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

fn file_size(p: &Path) -> u64 {
    p.metadata().ok().map(|m| m.len()).unwrap_or(0)
}

fn unlink(p: &Path) {
    if let Err(err) = std::fs::remove_file(p) {
        warn!(error = %err, path = %p.display(), "delete inspector artifact");
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs::File;
    use std::io::Write;

    fn temp_dir() -> PathBuf {
        let p = std::env::temp_dir().join(format!(
            "speaches-plus-ret-{}-{}",
            std::process::id(),
            uuid::Uuid::new_v4().simple()
        ));
        std::fs::create_dir_all(&p).unwrap();
        p
    }

    fn touch(dir: &Path, name: &str, body: &[u8]) -> PathBuf {
        let p = dir.join(name);
        let mut f = File::create(&p).unwrap();
        f.write_all(body).unwrap();
        p
    }

    #[test]
    fn cleanup_respects_max_count() {
        let dir = temp_dir();
        for i in 0..5 {
            touch(&dir, &format!("sess_{}.ndjson", i), b"x");
        }
        cleanup_on_startup(&dir, 2, 0, 0);
        let remaining = std::fs::read_dir(&dir).unwrap().count();
        assert_eq!(remaining, 2);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn cleanup_skips_nonexistent_dir() {
        let p = std::env::temp_dir().join(format!(
            "speaches-plus-ret-missing-{}",
            uuid::Uuid::new_v4().simple()
        ));
        cleanup_on_startup(&p, 10, 0, 0);
    }

    #[test]
    fn cleanup_respects_max_bytes() {
        let dir = temp_dir();
        for i in 0..5 {
            touch(&dir, &format!("sess_{}.ndjson", i), &[b'x'; 100]);
        }
        cleanup_on_startup(&dir, 0, 250, 0);
        let total: u64 = std::fs::read_dir(&dir)
            .unwrap()
            .flatten()
            .map(|e| e.metadata().map(|m| m.len()).unwrap_or(0))
            .sum();
        assert!(total <= 250, "total {} exceeded 250", total);
        let _ = std::fs::remove_dir_all(&dir);
    }
}
