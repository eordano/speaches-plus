#![allow(dead_code)]

use std::path::{Path, PathBuf};

pub const ALLOW_SKIP: &str = "NV_MODELS_ALLOW_SKIP";

pub fn hub_roots() -> Vec<PathBuf> {
    let mut roots: Vec<PathBuf> = Vec::new();
    for key in ["NV_MODELS_TEST_HUB", "HF_HUB_CACHE"] {
        if let Ok(p) = std::env::var(key) {
            if !p.is_empty() {
                roots.push(PathBuf::from(p));
            }
        }
    }
    if let Ok(home) = std::env::var("HOME") {
        roots.push(PathBuf::from(home).join(".cache/huggingface/hub"));
    }
    roots.retain(|r| r.is_dir());
    roots.dedup();
    roots
}

fn repo_dirname(repo: &str) -> String {
    format!("models--{}", repo.replace('/', "--"))
}

fn snapshot_candidates(repo: &str) -> Vec<PathBuf> {
    let mut out = Vec::new();
    for root in hub_roots() {
        let repo_dir = root.join(repo_dirname(repo));
        let snaps = repo_dir.join("snapshots");
        let Ok(rd) = std::fs::read_dir(&snaps) else {
            continue;
        };
        let mut here: Vec<PathBuf> = rd
            .flatten()
            .map(|e| e.path())
            .filter(|p| p.is_dir())
            .collect();
        here.sort();
        if let Ok(sha) = std::fs::read_to_string(repo_dir.join("refs/main")) {
            let pinned = snaps.join(sha.trim());
            if pinned.is_dir() {
                here.retain(|p| p != &pinned);
                here.insert(0, pinned);
            }
        }
        out.extend(here);
    }
    out
}

pub fn has_safetensors(dir: &Path) -> bool {
    let Ok(rd) = std::fs::read_dir(dir) else {
        return false;
    };
    rd.flatten().any(|e| {
        e.path()
            .extension()
            .map(|x| x == "safetensors")
            .unwrap_or(false)
    })
}

pub fn snapshot_of(repo: &str, need: &[&str]) -> Option<PathBuf> {
    snapshot_candidates(repo).into_iter().find(|p| {
        need.iter().all(|f| {
            if *f == "*.safetensors" {
                has_safetensors(p)
            } else {
                p.join(f).exists()
            }
        })
    })
}

pub fn dir_from_env_or_hub(env_var: &str, repo: &str, need: &[&str]) -> Option<PathBuf> {
    if let Ok(v) = std::env::var(env_var) {
        if !v.is_empty() {
            let p = PathBuf::from(v);
            if p.is_dir() {
                return Some(p);
            }
            eprintln!("{env_var}={} is not a directory", p.display());
            return None;
        }
    }
    snapshot_of(repo, need)
}

pub fn precondition_absent(test: &str, what: &str, how: &str) {
    let msg = format!(
        "{test}: PRECONDITION NOT MET, THIS TEST EXECUTED NOTHING.\n  missing: {what}\n  obtain it: {how}"
    );
    if std::env::var(ALLOW_SKIP).as_deref() == Ok("1") {
        eprintln!(
            "\n################ SKIPPED, NOT PASSED ################\n{msg}\n\
             (downgraded from a failure because {ALLOW_SKIP}=1)\n\
             ####################################################\n"
        );
        return;
    }
    panic!("{msg}\n  set {ALLOW_SKIP}=1 to downgrade this failure to a printed skip");
}
