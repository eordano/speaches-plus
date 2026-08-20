use nv_tts::{VoiceProfile, VoiceProfileStore};
use std::path::PathBuf;
use std::sync::OnceLock;

static TEST_ROOT_WRITE_ONCE: OnceLock<PathBuf> = OnceLock::new();

fn test_root() -> PathBuf {
    TEST_ROOT_WRITE_ONCE
        .get_or_init(|| {
            let p = std::env::temp_dir()
                .join(format!("nv_tts_voice_profile_tests_{}", std::process::id()));
            if p.exists() {
                std::fs::remove_dir_all(&p)
                    .expect("stale test root from a previous run of this pid must be removable");
            }
            std::fs::create_dir_all(&p).expect("test root must be creatable under temp_dir");
            p
        })
        .clone()
}

fn store_in(subdir: &str) -> (VoiceProfileStore, PathBuf) {
    let root = test_root().join(subdir);
    let store = VoiceProfileStore::open(&root).expect("open must create the root and succeed");
    (store, root)
}

fn profile(name: &str, embedding: Vec<f32>) -> VoiceProfile {
    VoiceProfile {
        schema_version: 1,
        name: name.to_string(),
        embedding,
        design_params: None,
    }
}

#[test]
fn open_creates_missing_nested_root_and_reopening_existing_root_is_idempotent() {
    let root = test_root().join("open_nested").join("a").join("b");
    assert!(!root.exists(), "precondition: nested root absent");
    let _s1 = VoiceProfileStore::open(&root).expect("open must create_dir_all the root");
    assert!(root.is_dir(), "open must have created the nested root dir");
    let _s2 = VoiceProfileStore::open(&root).expect("reopening an existing root must not error");
}

#[test]
fn path_for_is_root_join_name_dot_json() {
    let (store, root) = store_in("path_for");
    assert_eq!(store.path_for("alice"), root.join("alice.json"));
    assert_eq!(store.path_for("a.b"), root.join("a.b.json"));
}

#[test]
fn path_for_does_no_name_sanitization_dotdot_escapes_root_so_callers_must_gate_names_like_oapi_is_safe_name(
) {
    let (store, root) = store_in("path_for_traversal");
    let escaped = store.path_for("../evil");
    assert_eq!(
        escaped,
        root.join("../evil.json"),
        "store trusts names verbatim; rejecting '..' is the HTTP handler's job, and this pin \
         documents that moving validation into the store would be a behavior change"
    );
}

#[test]
fn put_get_roundtrip_preserves_schema_version_name_exact_f32_embedding_bits_and_design_params_variants(
) {
    let (store, _root) = store_in("roundtrip");
    let tricky_f32s = vec![
        0.15625_f32,
        -0.0_f32,
        f32::MIN_POSITIVE,
        1.0e-7_f32,
        3.402_823_5e38_f32,
        1.1754942e-38_f32,
        core::f32::consts::PI,
    ];
    let mut none_dp = profile("round_none", tricky_f32s.clone());
    none_dp.schema_version = 7;
    store.put(&none_dp).expect("put none_dp");
    let got = store.get("round_none").expect("get round_none");
    assert_eq!(got.schema_version, 7);
    assert_eq!(got.name, "round_none");
    assert_eq!(
        got.embedding.len(),
        tricky_f32s.len(),
        "embedding length must survive"
    );
    for (i, (a, b)) in tricky_f32s.iter().zip(got.embedding.iter()).enumerate() {
        assert_eq!(
            a.to_bits(),
            b.to_bits(),
            "finite f32 at index {i} must roundtrip bit-exactly through the JSON file ({a} vs {b})"
        );
    }
    assert!(
        got.design_params.is_none(),
        "None design_params must stay None"
    );

    let mut some_dp = profile("round_some", vec![1.0, 2.0]);
    some_dp.design_params = Some(serde_json::json!({"timbre": "warm", "age": 30}));
    store.put(&some_dp).expect("put some_dp");
    let got2 = store.get("round_some").expect("get round_some");
    assert_eq!(
        got2.design_params,
        Some(serde_json::json!({"timbre": "warm", "age": 30})),
        "Some design_params must roundtrip structurally"
    );
}

#[test]
fn put_with_same_name_overwrites_get_returns_latest_and_list_has_one_entry() {
    let (store, _root) = store_in("overwrite");
    store.put(&profile("v", vec![1.0])).expect("first put");
    store.put(&profile("v", vec![2.0, 3.0])).expect("second put");
    let got = store.get("v").expect("get after overwrite");
    assert_eq!(
        got.embedding,
        vec![2.0, 3.0],
        "second put must replace, not append"
    );
    assert_eq!(
        store.list().expect("list"),
        vec!["v".to_string()],
        "overwriting must not duplicate the listing"
    );
}

#[test]
fn get_missing_name_errors_which_profile_speaker_embed_relies_on_to_fall_through_to_builtin_voices()
{
    let (store, _root) = store_in("get_missing");
    assert!(
        store.get("nope").is_err(),
        "get of an unknown name must be Err, never a default profile"
    );
}

#[test]
fn delete_is_idempotent_ok_on_missing_but_after_deleting_real_profile_get_errors_and_list_excludes()
{
    let (store, _root) = store_in("delete");
    store
        .delete("never_existed")
        .expect("delete of a missing profile must be Ok, negative control vs get which errors");
    store.put(&profile("gone", vec![0.5])).expect("put gone");
    assert_eq!(store.list().expect("list before"), vec!["gone".to_string()]);
    store.delete("gone").expect("delete existing");
    assert!(
        store.get("gone").is_err(),
        "get after delete must be Err"
    );
    assert!(
        store.list().expect("list after").is_empty(),
        "list after delete must be empty"
    );
    store
        .delete("gone")
        .expect("second delete of the same name must still be Ok");
}

#[test]
fn list_returns_lexicographically_sorted_json_stems_c10_before_c2_and_ignores_non_json_files() {
    let (store, root) = store_in("list_sorted");
    for name in ["b", "a", "c10", "c2"] {
        store.put(&profile(name, vec![1.0])).expect("put");
    }
    std::fs::write(root.join("readme.txt"), b"not a profile").expect("write txt decoy");
    std::fs::write(root.join("backup.json.bak"), b"{}").expect("write bak decoy");
    std::fs::write(root.join("noext"), b"{}").expect("write extensionless decoy");
    assert_eq!(
        store.list().expect("list"),
        vec![
            "a".to_string(),
            "b".to_string(),
            "c10".to_string(),
            "c2".to_string()
        ],
        "byte-order sort (c10 < c2) and only .json stems; decoys must be invisible"
    );
}

#[test]
fn list_includes_corrupt_json_files_because_list_never_parses_matching_handle_list_which_skips_unreadable_entries(
) {
    let (store, root) = store_in("list_corrupt");
    std::fs::write(root.join("broken.json"), b"{ this is not json").expect("write corrupt");
    assert_eq!(
        store.list().expect("list must not parse and so must not fail"),
        vec!["broken".to_string()]
    );
    assert!(
        store.get("broken").is_err(),
        "get of the same corrupt entry must error, proving list/get diverge by design"
    );
}

#[test]
fn on_disk_format_is_a_json_object_with_exactly_the_four_known_field_names() {
    let (store, root) = store_in("disk_format");
    let mut p = profile("wire", vec![0.25]);
    p.design_params = Some(serde_json::json!({"k": 1}));
    store.put(&p).expect("put wire");
    let bytes = std::fs::read(root.join("wire.json")).expect("profile file must exist at path_for");
    let v: serde_json::Value = serde_json::from_slice(&bytes).expect("file must be valid JSON");
    let obj = v.as_object().expect("top level must be an object");
    let mut keys: Vec<&str> = obj.keys().map(|s| s.as_str()).collect();
    keys.sort_unstable();
    assert_eq!(
        keys,
        vec!["design_params", "embedding", "name", "schema_version"],
        "renaming or adding serialized fields changes the on-disk contract shared with any \
         external reader of the profile dir"
    );
}

#[test]
fn get_tolerates_missing_design_params_as_none_but_errors_on_missing_embedding() {
    let (store, root) = store_in("schema_strictness");
    std::fs::write(
        root.join("legacy.json"),
        br#"{"schema_version":1,"name":"legacy","embedding":[1.0]}"#,
    )
    .expect("write legacy profile without design_params");
    let got = store
        .get("legacy")
        .expect("Option field absent in file must parse, so old files keep loading");
    assert!(got.design_params.is_none());
    assert_eq!(got.embedding, vec![1.0]);

    std::fs::write(
        root.join("truncated.json"),
        br#"{"schema_version":1,"name":"truncated"}"#,
    )
    .expect("write profile missing required embedding");
    assert!(
        store.get("truncated").is_err(),
        "missing embedding must be a hard parse error, not an implicit empty vec"
    );
}

#[test]
fn put_of_nonfinite_embedding_silently_writes_json_null_making_the_profile_unreadable_a_hazard_for_encoder_nan_output(
) {
    let (store, _root) = store_in("nonfinite");
    store
        .put(&profile("nanny", vec![f32::NAN, 1.0]))
        .expect("serde_json maps non-finite floats to null so put succeeds");
    assert!(
        store.get("nanny").is_err(),
        "reading back the null lands in f32 and must fail; if this ever becomes Ok the \
         serializer changed and enrollment should start rejecting NaN upstream"
    );
}
