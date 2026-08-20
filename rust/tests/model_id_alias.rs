use std::sync::Arc;

use speaches_plus::oapi::chat_engine::{ChatRegistry, EchoEngine};
use speaches_plus::oapi::model_ids::{canonical_model_id, model_id_for_dir};

const E4B_STORE: &str = "/nix/store/0123456789abcdfghijklmnpqrsvwxyz-hf-model-google-\
                         gemma-4-E4B-it-qat-w4a16-ct-6cd26aaa2357fb2bad8c51699a7558a4d1a965bb";

#[test]
fn nix_store_and_hub_paths_decode_to_pretty_ids() {
    assert_eq!(
        model_id_for_dir(std::path::Path::new(E4B_STORE)),
        "google/gemma-4-E4B-it-qat-w4a16-ct"
    );
    assert_eq!(
        model_id_for_dir(std::path::Path::new(
            "/x/hub/models--Qwen--Qwen3-Embedding-0.6B/snapshots/97b0c614"
        )),
        "Qwen/Qwen3-Embedding-0.6B"
    );
    assert_eq!(
        model_id_for_dir(std::path::Path::new("/models/my-local-gemma")),
        "my-local-gemma"
    );
}

#[test]
fn canonical_model_id_only_fires_for_store_and_hub_layouts() {
    let want = Some("google/gemma-4-E4B-it-qat-w4a16-ct".to_string());
    assert_eq!(canonical_model_id(E4B_STORE), want);
    let base = E4B_STORE.rsplit('/').next().unwrap();
    assert_eq!(canonical_model_id(base), want);
    assert_eq!(
        canonical_model_id("google/gemma-4-E4B-it-qat-w4a16-ct"),
        None
    );
    assert_eq!(canonical_model_id("gpt-oss-20b"), None);
    assert_eq!(canonical_model_id(""), None);
}

#[test]
fn registry_resolves_pretty_id_and_both_legacy_aliases() {
    let reg = ChatRegistry::single(Arc::new(EchoEngine::new(
        "google/gemma-4-E4B-it-qat-w4a16-ct",
        "x",
    )));
    let base = E4B_STORE.rsplit('/').next().unwrap();
    for id in ["google/gemma-4-E4B-it-qat-w4a16-ct", E4B_STORE, base] {
        let eng = reg.resolve_with(Some(id), false).unwrap_or_else(|| {
            panic!("id form not accepted: {id}");
        });
        assert_eq!(eng.model_id(), "google/gemma-4-E4B-it-qat-w4a16-ct");
    }
    assert!(reg.resolve_with(Some("nope/missing"), false).is_none());
    assert_eq!(
        reg.model_ids(),
        &["google/gemma-4-E4B-it-qat-w4a16-ct".to_string()]
    );
}
