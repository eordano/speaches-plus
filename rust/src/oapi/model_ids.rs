use std::path::Path;

pub fn model_id_for_dir(dir: &Path) -> String {
    canonical_model_id_for_path(dir).unwrap_or_else(|| {
        dir.file_name()
            .and_then(|s| s.to_str())
            .unwrap_or("nv-wgpu-model")
            .to_string()
    })
}

pub fn canonical_model_id(raw: &str) -> Option<String> {
    canonical_model_id_for_path(Path::new(raw))
}

fn canonical_model_id_for_path(dir: &Path) -> Option<String> {
    let name = dir.file_name()?.to_str()?;
    if let Some(id) = nix_store_model_id(name) {
        return Some(id);
    }
    hub_snapshot_model_id(dir)
}

fn hub_snapshot_model_id(dir: &Path) -> Option<String> {
    let parent = dir.parent()?;
    if parent.file_name()?.to_str()? != "snapshots" {
        return None;
    }
    let repo = parent
        .parent()?
        .file_name()?
        .to_str()?
        .strip_prefix("models--")?;
    let mut parts = repo.splitn(2, "--");
    match (parts.next(), parts.next()) {
        (Some(org), Some(rest)) => Some(format!("{org}/{}", rest.replace("--", "/"))),
        _ => Some(repo.replace("--", "/")),
    }
}

fn nix_store_model_id(name: &str) -> Option<String> {
    let (hash, rest) = name.split_once('-')?;
    if hash.len() != 32
        || !hash
            .bytes()
            .all(|b| b.is_ascii_lowercase() || b.is_ascii_digit())
    {
        return None;
    }
    let mut spec = rest.strip_prefix("hf-model-")?;
    if let Some(i) = spec.rfind('-') {
        let rev = &spec[i + 1..];
        if rev.len() == 40 && rev.bytes().all(|b| b.is_ascii_hexdigit()) {
            spec = &spec[..i];
        }
    }
    let (org, model) = spec.split_once('-')?;
    if org.is_empty() || model.is_empty() {
        return None;
    }
    Some(format!("{org}/{model}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    const E4B_STORE: &str = "/nix/store/0123456789abcdfghijklmnpqrsvwxyz-hf-model-google-\
                             gemma-4-E4B-it-qat-w4a16-ct-6cd26aaa2357fb2bad8c51699a7558a4d1a965bb";

    #[test]
    fn nix_store_hf_model_path_decodes_to_org_slash_name() {
        assert_eq!(
            model_id_for_dir(Path::new(E4B_STORE)),
            "google/gemma-4-E4B-it-qat-w4a16-ct"
        );
    }

    #[test]
    fn nix_store_path_without_rev_suffix_keeps_full_name() {
        let p = "/nix/store/0123456789abcdfghijklmnpqrsvwxyz-hf-model-Qwen-Qwen3-Embedding-0.6B";
        assert_eq!(model_id_for_dir(Path::new(p)), "Qwen/Qwen3-Embedding-0.6B");
    }

    #[test]
    fn hub_snapshot_path_decodes_to_org_slash_name() {
        let p = "/x/hub/models--nvidia--Gemma-4-31B-IT-NVFP4/snapshots/e5ef03af";
        assert_eq!(
            model_id_for_dir(Path::new(p)),
            "nvidia/Gemma-4-31B-IT-NVFP4"
        );
    }

    #[test]
    fn plain_local_dir_keeps_its_basename() {
        assert_eq!(
            model_id_for_dir(Path::new("/models/my-local-gemma")),
            "my-local-gemma"
        );
    }

    #[test]
    fn canonical_model_id_accepts_full_path_and_bare_store_basename() {
        let want = Some("google/gemma-4-E4B-it-qat-w4a16-ct".to_string());
        assert_eq!(canonical_model_id(E4B_STORE), want);
        let base = Path::new(E4B_STORE).file_name().unwrap().to_str().unwrap();
        assert_eq!(canonical_model_id(base), want);
    }

    #[test]
    fn canonical_model_id_rejects_ordinary_ids() {
        assert_eq!(
            canonical_model_id("google/gemma-4-E4B-it-qat-w4a16-ct"),
            None
        );
        assert_eq!(canonical_model_id("gpt-oss-20b"), None);
        assert_eq!(canonical_model_id("my-local-gemma"), None);
        assert_eq!(canonical_model_id(""), None);
    }
}

pub fn chat_model_dirs_from_env() -> Vec<std::path::PathBuf> {
    let mut dirs = Vec::new();
    if let Some(d) = std::env::var_os("NV_CHAT_MODEL_DIR") {
        dirs.push(std::path::PathBuf::from(d));
    }
    for var in ["NV_CHAT_MODEL_DIRS", "NV_WGPU_CHAT_MODEL_DIRS"] {
        if let Some(list) = std::env::var_os(var) {
            let list = list.to_string_lossy().into_owned();
            dirs.extend(
                list.split([',', ':'])
                    .map(str::trim)
                    .filter(|s| !s.is_empty())
                    .map(std::path::PathBuf::from),
            );
        }
    }
    dirs
}
