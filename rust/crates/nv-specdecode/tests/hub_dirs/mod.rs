#![allow(dead_code)]

use std::path::PathBuf;

pub fn hub_roots() -> Vec<PathBuf> {
    let mut roots: Vec<PathBuf> = Vec::new();
    for k in ["NV_HUB_CACHE", "HF_HUB_CACHE"] {
        if let Some(v) = std::env::var_os(k) {
            roots.push(PathBuf::from(v));
        }
    }
    if let Some(v) = std::env::var_os("HF_HOME") {
        roots.push(PathBuf::from(v).join("hub"));
    }
    if let Some(v) = std::env::var_os("HOME") {
        roots.push(PathBuf::from(v).join(".cache/huggingface/hub"));
    }
    roots.dedup();
    roots
}

pub fn snapshot(repo: &str, markers: &[&str]) -> Option<PathBuf> {
    let ok = |p: &PathBuf| markers.iter().all(|m| p.join(m).exists());
    for root in hub_roots() {
        let repo_dir = root.join(repo);
        let snaps = repo_dir.join("snapshots");
        if let Ok(sha) = std::fs::read_to_string(repo_dir.join("refs/main")) {
            let p = snaps.join(sha.trim());
            if ok(&p) {
                return Some(p);
            }
        }
        let mut cands: Vec<PathBuf> = std::fs::read_dir(&snaps)
            .ok()
            .into_iter()
            .flatten()
            .flatten()
            .map(|e| e.path())
            .filter(ok)
            .collect();
        cands.sort();
        if let Some(p) = cands.pop() {
            return Some(p);
        }
    }
    None
}

pub fn require_snapshot(
    test: &str,
    repo: &str,
    markers: &[&str],
    allow_var: &str,
) -> Option<PathBuf> {
    if let Some(p) = snapshot(repo, markers) {
        return Some(p);
    }
    if std::env::var(allow_var).as_deref() == Ok("1") {
        eprintln!(
            "SKIP ({allow_var}=1): {test}: no {repo} snapshot carrying {markers:?}. \
             This is a SKIP, not a pass -- nothing in this test was exercised."
        );
        return None;
    }
    let roots: Vec<String> = hub_roots()
        .iter()
        .map(|p| p.display().to_string())
        .collect();
    panic!(
        "{test}: no {repo} snapshot carrying {markers:?} under any of {roots:?}. \
         This gate refuses to report success without running. Fetch the checkpoint (to /tank, \
         never zroot) or set {allow_var}=1 to skip on purpose."
    );
}

pub fn require_env_flag(test: &str, var: &str, why: &str) {
    let set = matches!(std::env::var(var).as_deref(), Ok(v) if !v.is_empty() && v != "0");
    if !set {
        panic!(
            "{test}: {var} is not set. {why} Asking for this test by name (or with --ignored) is \
             an explicit request to run it, so returning early here would report a pass having \
             executed nothing. Set {var}=1 to run it."
        );
    }
}
