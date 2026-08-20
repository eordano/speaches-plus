
use std::path::PathBuf;

fn hub_roots() -> Vec<PathBuf> {
    let mut out: Vec<PathBuf> = Vec::new();
    if let Ok(v) = std::env::var("HF_HUB_CACHE") {
        out.push(PathBuf::from(v));
    }
    if let Some(home) = std::env::var_os("HOME") {
        out.push(PathBuf::from(home).join(".cache/huggingface/hub"));
    }
    out.retain(|p| p.is_dir());
    out
}

fn cached_tokenizer(repo: &str) -> Option<PathBuf> {
    for root in hub_roots() {
        let Ok(rd) = std::fs::read_dir(root.join(repo).join("snapshots")) else {
            continue;
        };
        for entry in rd.flatten() {
            let p = entry.path().join("tokenizer.json");
            if p.is_file() {
                return Some(p);
            }
        }
    }
    None
}

fn all_cached_tokenizers() -> Vec<(String, PathBuf)> {
    let mut out: Vec<(String, PathBuf)> = Vec::new();
    for root in hub_roots() {
        let Ok(rd) = std::fs::read_dir(&root) else {
            continue;
        };
        for repo in rd.flatten() {
            let name = repo.file_name().to_string_lossy().to_string();
            if !name.starts_with("models--") {
                continue;
            }
            if out.iter().any(|(n, _)| *n == name) {
                continue;
            }
            if let Some(p) = cached_tokenizer(&name) {
                out.push((name, p));
            }
        }
    }
    out.sort();
    out
}

#[test]
fn serving_tokenizer_load_strips_shipped_truncation_for_all_cached_models() {
    let found = all_cached_tokenizers();
    let mut checked: Vec<String> = Vec::new();
    let mut shipped_truncation: Vec<String> = Vec::new();
    for (repo, path) in &found {
        if tokenizers::Tokenizer::from_file(path)
            .ok()
            .is_some_and(|t| t.get_truncation().is_some())
        {
            shipped_truncation.push(repo.clone());
        }
        let tok = nv_tokenizer::load_tokenizer(path)
            .unwrap_or_else(|e| panic!("{repo}: serving load of {} failed: {e:#}", path.display()));
        assert!(
            tok.get_truncation().is_none(),
            "{repo}: serving tokenizer load must strip built-in truncation"
        );
        assert!(
            tok.get_padding().is_none(),
            "{repo}: serving tokenizer load must strip built-in padding"
        );
        checked.push(repo.clone());
    }
    eprintln!(
        "swept {} hub root(s) {:?}\nchecked {} cached tokenizer.json: {checked:?}\nof those, {} \
         ship built-in truncation and are the ones this gate exists for: {shipped_truncation:?}",
        hub_roots().len(),
        hub_roots(),
        checked.len(),
        shipped_truncation.len()
    );
    assert!(
        !checked.is_empty(),
        "not one cached tokenizer.json was found under {:?}: this gate proved nothing about \
         truncation stripping. That is a SKIP, not a pass.",
        hub_roots()
    );
}

#[test]
fn a_shipped_truncation_cap_does_not_survive_the_serving_load() {
    const QWEN: &str = "models--RedHatAI--Qwen3.6-35B-A3B-NVFP4";
    let candidates: Vec<(String, PathBuf)> = match cached_tokenizer(QWEN) {
        Some(p) => vec![(QWEN.to_string(), p)],
        None => all_cached_tokenizers(),
    };
    let mut found = None;
    for (repo, path) in candidates {
        let raw = match tokenizers::Tokenizer::from_file(&path) {
            Ok(t) => t,
            Err(_) => continue,
        };
        if let Some(tr) = raw.get_truncation() {
            found = Some((repo, path, tr.max_length));
            break;
        }
    }
    let Some((repo, path, cap)) = found else {
        eprintln!(
            "SKIP: not one of the {} cached tokenizer.json under {:?} ships built-in truncation, \
             so the defect this gate reproduces is not present in this corpus. This is a SKIP, \
             not a pass: nothing about truncation stripping was demonstrated.",
            all_cached_tokenizers().len(),
            hub_roots()
        );
        return;
    };
    eprintln!("reproducing against {repo}: shipped truncation.max_length={cap}");
    let tok = nv_tokenizer::load_tokenizer(&path).expect("serving load");
    let text = "Summarize the following incident report in detail. ".repeat(cap / 4 + 1100);
    let n = tok
        .encode(text.as_str(), false)
        .expect("encode")
        .get_ids()
        .len();
    assert!(
        n > cap,
        "expected > {cap} tokens, got {n} (still truncating?)"
    );
}

#[test]
#[ignore = "post-freeze GPU verify round: long-prompt serve probe, procedure in the panic message"]
fn post_freeze_gate_qwen_long_prompt_serve_never_silently_truncates() {
    panic!(
        "POST-FREEZE VERIFY GATE (GPU, timed round -- do not run during the freeze): serve \
         RedHatAI/Qwen3.6-35B-A3B-NVFP4 and POST the measurement pack's 9451-id probe prompt as \
         chat text, NOT pre-tokenized ids (the harness id-feeding workaround must stay retired \
         for this probe) to /v1/chat/completions. Required outcome, one of: (1) prompt_tokens \
         reported by the engine is ~9451 (not 4096) and the completion answers the full prompt, \
         if 9451 fits the engine's real context (kv_max_seq_len / NV_WGPU_MAX_SEQ as \
         configured); or (2) a clear over-length error: cuda path 'prompt of N tokens does not \
         fit the C-token KV window', wgpu path 'does not fit the C-token wgpu KV window', batch \
         path 'exceeds the engine's KV capacity'. FAIL if the serve answers with prompt_tokens \
         == 4096: that is the silent truncation regressing"
    );
}
