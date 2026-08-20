use nv_layers::backend::{BackendKind, KernelId};
use speaches_plus::oapi::backend_select::{
    backends_report, cuda_model_unservable_reason, cuda_only_fast_paths, resolve, resolve_with,
    resolve_with_build, wgpu_missing_kernels, wgpu_model_support, wgpu_unservable_reason_for_build,
    BackendSelectError, ModelClass, ServeBackend, WgpuEvidence, AUTO_POLICY,
    WGPU_DECODERS_COMPILED_IN, WGPU_DECODER_CLASSES, WGPU_FEATURE_OFF_REASON,
};

const CACHED_MODELS: &[(&str, ModelClass, bool)] = &[
    ("nvidia/Gemma-4-31B-IT-NVFP4", ModelClass::Gemma4Dense, true),
    ("google/gemma-4-E4B-it", ModelClass::Gemma4E4b, true),
    (
        "RedHatAI/Qwen3.6-35B-A3B-NVFP4",
        ModelClass::Qwen36Moe,
        true,
    ),
    ("Qwen/Qwen3.5-MoE", ModelClass::Qwen35Moe, true),
    ("google/gemma-4-26B-A4B-it", ModelClass::Gemma4Moe, true),
    ("openai/gpt-oss-20b", ModelClass::GptOss, true),
    ("poolside/Laguna-XS-2.1-NVFP4", ModelClass::Laguna, true),
    ("ig1/Qwen3.5-9B-NVFP4", ModelClass::Qwen35Dense, true),
    (
        "google/diffusiongemma-26B-A4B-it",
        ModelClass::DiffusionGemma,
        false,
    ),
    ("mistralai/Mistral-7B-Instruct", ModelClass::Unknown, false),
];

fn probe_ok() -> Result<(), String> {
    Ok(())
}

fn probe_fail() -> Result<(), String> {
    Err("no adapter".to_string())
}

fn probe_must_not_run() -> Result<(), String> {
    panic!("this probe must not be consulted for this selection");
}

#[test]
fn the_candle_tensor_bridge_reason_is_gone_from_every_gate_string() {
    for (id, class, _) in CACHED_MODELS {
        for compiled in [false, true] {
            let reason = wgpu_unservable_reason_for_build(*class, compiled).unwrap_or_default();
            let lower = reason.to_ascii_lowercase();
            assert!(
                !lower.contains("candle"),
                "{id}: gate still blames candle: {reason}"
            );
            assert!(
                !lower.contains("residency bridge"),
                "{id}: gate still blames the residency bridge: {reason}"
            );
        }
    }
    let report = backends_report().to_string().to_ascii_lowercase();
    assert!(
        !report.contains("candle"),
        "backends_report still says candle"
    );
}

#[test]
fn the_three_verified_decoders_are_registered_with_their_nv_models_modules() {
    let dense = wgpu_model_support(ModelClass::Gemma4Dense).expect("gemma4 dense");
    assert_eq!(dense.module, "nv_models::gemma4_wgpu");
    assert_eq!(dense.entry, "Gemma4Wgpu::new");

    let e4b = wgpu_model_support(ModelClass::Gemma4E4b).expect("gemma4 e4b");
    assert_eq!(e4b.module, "nv_models::gemma4_e4b_wgpu");
    assert_eq!(e4b.entry, "Gemma4E4bWgpu::new");

    let qwen = wgpu_model_support(ModelClass::Qwen36Moe).expect("qwen3.6 moe");
    assert_eq!(qwen.module, "nv_models::qwen3_5_moe_wgpu");
    assert_eq!(qwen.entry, "Qwen3MoeWgpu::new");

    for class in [
        ModelClass::Gemma4Dense,
        ModelClass::Gemma4E4b,
        ModelClass::Qwen36Moe,
    ] {
        let d = wgpu_model_support(class).unwrap();
        assert!(
            matches!(d.evidence, WgpuEvidence::RealWeights { .. }),
            "{:?} must carry real-weights evidence, got {:?}",
            class,
            d.evidence
        );
        assert_eq!(d.evidence.kind(), "real-weights-decode");
        assert!(d.evidence.detail().contains("NV_"), "{:?}", d.evidence);
    }

    let q35 = wgpu_model_support(ModelClass::Qwen35Moe).unwrap();
    assert_eq!(q35.module, "nv_models::qwen3_5_moe_wgpu");
    assert!(
        matches!(q35.evidence, WgpuEvidence::ArchitectureFamily { .. }),
        "qwen3.5 must not claim real-weights evidence: {:?}",
        q35.evidence
    );
}

#[test]
fn classes_with_a_wgpu_decoder_have_no_missing_wgpu_kernels() {
    for class in WGPU_DECODER_CLASSES {
        let missing = wgpu_missing_kernels(*class);
        assert!(
            missing.is_empty(),
            "{:?} still reports missing wgpu kernels: {:?}",
            class,
            missing.iter().map(|k| k.name()).collect::<Vec<_>>()
        );
        assert!(wgpu_model_support(*class).is_ok(), "{:?}", class);
    }
}

#[test]
fn cuda_only_fast_paths_are_reported_but_never_block_the_wgpu_decoders() {
    let e4b = cuda_only_fast_paths(ModelClass::Gemma4E4b);
    assert!(
        e4b.contains(&KernelId::MarlinGemmW4a16),
        "e4b should still report marlin as a cuda-only fast path: {e4b:?}"
    );
    let moe = cuda_only_fast_paths(ModelClass::Qwen36Moe);
    assert!(
        moe.is_empty(),
        "nv-layers moe_wgpu dispatches a grouped nvfp4 expert gemm held bit-exact against the \
         CUDA grouped path by wgpu_correct_moe_cuda_shape_sweep, so no moe kernel is cuda-only: \
         {moe:?}"
    );

    assert!(wgpu_missing_kernels(ModelClass::Gemma4E4b).is_empty());
    assert!(wgpu_missing_kernels(ModelClass::Qwen36Moe).is_empty());
    assert!(cuda_only_fast_paths(ModelClass::Gemma4Dense).is_empty());

    let reason = wgpu_unservable_reason_for_build(ModelClass::Gemma4E4b, true);
    assert_eq!(reason, None, "marlin must not block the e4b wgpu decoder");
    let reason = wgpu_unservable_reason_for_build(ModelClass::Qwen36Moe, true);
    assert_eq!(
        reason, None,
        "the cutlass grouped gemm must not block the qwen wgpu decoder"
    );
}

#[test]
fn a_model_with_no_wgpu_module_is_refused_with_a_useful_message() {
    let err = wgpu_model_support(ModelClass::DiffusionGemma).unwrap_err();
    assert!(err.contains("no native wgpu decoder"), "{err}");
    assert!(err.contains("diffusion-gemma-26b-a4b"), "{err}");
    assert!(err.contains("not autoregressive"), "{err}");
    assert!(err.contains("block-diffusion"), "{err}");
    assert!(err.contains("gemma4-dense-nvfp4"), "{err}");

    let err = wgpu_model_support(ModelClass::Unknown).unwrap_err();
    assert!(err.contains("unrecognized model id"), "{err}");
    assert!(err.contains("nv-models"), "{err}");
}

#[test]
fn the_feature_gate_refuses_every_model_when_the_decoders_are_not_compiled_in() {
    for (id, class, has_decoder) in CACHED_MODELS {
        let reason = wgpu_unservable_reason_for_build(*class, false)
            .unwrap_or_else(|| panic!("{id}: must be refused when wgpu is not compiled in"));
        if *has_decoder {
            assert_eq!(reason, WGPU_FEATURE_OFF_REASON, "{id}");
            assert!(reason.contains("--features wgpu"), "{id}: {reason}");
        } else {
            assert!(
                !reason.contains("--features wgpu"),
                "{id}: a permanent model gap must win over the rebuild-me message: {reason}"
            );
        }
    }
}

#[test]
fn wgpu_routing_is_pinned_for_every_cached_model_in_a_wgpu_build() {
    for (id, class, has_decoder) in CACHED_MODELS {
        assert_eq!(ModelClass::classify(id), *class, "{id}");
        let got = resolve_with_build(ServeBackend::Wgpu, id, &probe_must_not_run, &probe_ok, true);
        if *has_decoder {
            assert_eq!(got.unwrap(), BackendKind::Wgpu, "{id}");
        } else {
            let err = got.expect_err(&format!("{id} must be refused on wgpu"));
            assert!(
                matches!(err, BackendSelectError::ModelUnservable { .. }),
                "{id}: {err}"
            );
            let msg = err.to_string();
            assert!(msg.contains(id), "{id}: {msg}");
            assert!(msg.contains("no silent fallback"), "{id}: {msg}");
        }
    }
}

#[test]
fn wgpu_routing_refuses_every_model_when_the_decoders_are_not_compiled_in() {
    for (id, _, _) in CACHED_MODELS {
        let err = resolve_with_build(
            ServeBackend::Wgpu,
            id,
            &probe_must_not_run,
            &probe_ok,
            false,
        )
        .expect_err(&format!("{id} must be refused without the wgpu feature"));
        assert!(
            matches!(err, BackendSelectError::ModelUnservable { .. }),
            "{id}: {err}"
        );
    }
}

#[test]
fn an_explicit_wgpu_selection_never_falls_back_to_cuda() {
    for (id, _, _) in CACHED_MODELS {
        for compiled in [false, true] {
            match resolve_with_build(
                ServeBackend::Wgpu,
                id,
                &probe_must_not_run,
                &probe_ok,
                compiled,
            ) {
                Ok(kind) => assert_eq!(kind, BackendKind::Wgpu, "{id}"),
                Err(BackendSelectError::ModelUnservable { backend, .. }) => {
                    assert_eq!(backend, "wgpu", "{id}")
                }
                Err(other) => panic!("{id}: unexpected error kind: {other}"),
            }
        }
    }
    let err = resolve_with_build(
        ServeBackend::Wgpu,
        "nvidia/Gemma-4-31B-IT-NVFP4",
        &probe_must_not_run,
        &probe_fail,
        true,
    )
    .unwrap_err();
    assert!(
        matches!(err, BackendSelectError::Unavailable { .. }),
        "{err}"
    );
    assert!(err.to_string().contains("wgpu backend unavailable"));
}

#[test]
fn auto_is_cuda_first_and_never_downgrades_to_wgpu_while_cuda_works() {
    for (id, class, _) in CACHED_MODELS {
        if cuda_model_unservable_reason(*class).is_some() {
            continue;
        }
        let got = resolve_with_build(ServeBackend::Auto, id, &probe_ok, &probe_must_not_run, true)
            .unwrap_or_else(|e| panic!("{id}: auto failed with cuda available: {e}"));
        assert_eq!(got, BackendKind::Cuda, "{id}: auto downgraded to wgpu");
    }
}

#[test]
fn auto_reaches_wgpu_only_when_cuda_gives_an_explicit_reason() {
    for (id, _, has_decoder) in CACHED_MODELS {
        let got = resolve_with_build(ServeBackend::Auto, id, &probe_fail, &probe_ok, true);
        if *has_decoder {
            assert_eq!(got.unwrap(), BackendKind::Wgpu, "{id}");
        } else {
            let err = got.expect_err(&format!("{id} must have no backend"));
            assert!(
                matches!(err, BackendSelectError::NoBackend { .. }),
                "{id}: {err}"
            );
            let msg = err.to_string();
            assert!(msg.contains("cuda: no adapter"), "{id}: {msg}");
            assert!(msg.contains("wgpu:"), "{id}: {msg}");
        }
    }
}

#[test]
fn auto_never_returns_cuda_for_a_model_cuda_cannot_serve() {
    let reason =
        cuda_model_unservable_reason(ModelClass::GptOss).expect("gpt-oss has no cuda serving path");
    assert!(reason.contains("detect_family"), "{reason}");
    assert!(reason.contains("wgpu"), "{reason}");

    let err = resolve_with_build(
        ServeBackend::Auto,
        "google/diffusiongemma-26B-A4B-it",
        &probe_ok,
        &probe_ok,
        true,
    )
    .expect_err("diffusion-gemma is servable on neither backend");
    assert!(matches!(err, BackendSelectError::NoBackend { .. }), "{err}");
    let msg = err.to_string();
    assert!(msg.contains("cuda:"), "{msg}");
    assert!(msg.contains("no native wgpu decoder"), "{msg}");

    for (id, class, has_decoder) in CACHED_MODELS {
        match cuda_model_unservable_reason(*class) {
            Some(_) => {
                let err = resolve_with_build(
                    ServeBackend::Cuda,
                    id,
                    &probe_ok,
                    &probe_must_not_run,
                    true,
                )
                .expect_err(&format!("{id}: explicit cuda must refuse"));
                assert!(
                    matches!(err, BackendSelectError::ModelUnservable { .. }),
                    "{id}: {err}"
                );
                let got = resolve_with_build(ServeBackend::Auto, id, &probe_ok, &probe_ok, true);
                match got {
                    Ok(kind) => {
                        assert_eq!(
                            kind,
                            BackendKind::Wgpu,
                            "{id}: auto returned unservable cuda"
                        );
                        assert!(has_decoder, "{id}: wgpu result without a decoder");
                    }
                    Err(e) => {
                        assert!(
                            matches!(e, BackendSelectError::NoBackend { .. }),
                            "{id}: {e}"
                        );
                        assert!(!has_decoder, "{id}: NoBackend despite a wgpu decoder");
                    }
                }
            }
            None => {
                let got = resolve_with_build(
                    ServeBackend::Cuda,
                    id,
                    &probe_ok,
                    &probe_must_not_run,
                    true,
                )
                .unwrap_or_else(|e| panic!("{id}: cuda selection failed: {e}"));
                assert_eq!(got, BackendKind::Cuda, "{id}");
            }
        }
    }
}

#[test]
fn the_production_resolve_path_tracks_the_build_feature() {
    for (id, _, has_decoder) in CACHED_MODELS {
        let got = resolve_with(ServeBackend::Wgpu, id, &probe_must_not_run, &probe_ok);
        let servable = got.is_ok();
        eprintln!(
            "[gate] wgpu_decoders_compiled_in={WGPU_DECODERS_COMPILED_IN} {id} -> servable={servable}"
        );
        assert_eq!(
            servable,
            *has_decoder && WGPU_DECODERS_COMPILED_IN,
            "{id}: resolve_with disagrees with the build flag"
        );
        if servable {
            assert_eq!(got.unwrap(), BackendKind::Wgpu, "{id}");
        }
    }
}

#[test]
#[ignore = "opens a real wgpu adapter; set NV_WGPU_GATE_ADAPTER_TEST=1"]
fn a_real_adapter_resolves_wgpu_for_the_three_verified_models() {
    if std::env::var("NV_WGPU_GATE_ADAPTER_TEST").ok().as_deref() != Some("1") {
        panic!(
            "this test is #[ignore]d, so it was asked for BY NAME, but \
             NV_WGPU_GATE_ADAPTER_TEST=1 is not set, so it would have opened no adapter. \
             Returning here prints `1 passed` in 0.00s. This is a SKIP, not a pass."
        );
    }
    #[allow(clippy::assertions_on_constants)]
    {
        assert!(
            WGPU_DECODERS_COMPILED_IN,
            "rebuild with --features wgpu before running this"
        );
    }
    match nv_layers::backend::probe_wgpu() {
        Ok(()) => eprintln!("[gate] wgpu adapter probe: ok"),
        Err(e) => panic!(
            "no usable wgpu adapter: {e}\nthe default devshell ships ICD JSONs but no loader; see docs/book/05.1-wgpu-status.md: export LD_LIBRARY_PATH=<vulkan-loader>/lib:/run/opengl-driver/lib and VK_ICD_FILENAMES=/run/opengl-driver/share/vulkan/icd.d/nvidia_icd.json"
        ),
    }
    for id in [
        "nvidia/Gemma-4-31B-IT-NVFP4",
        "google/gemma-4-E4B-it",
        "RedHatAI/Qwen3.6-35B-A3B-NVFP4",
    ] {
        let got = resolve(ServeBackend::Wgpu, id)
            .unwrap_or_else(|e| panic!("{id}: real-probe wgpu resolve failed: {e}"));
        assert_eq!(got, BackendKind::Wgpu, "{id}");
        eprintln!("[gate] real probe: {id} -> {}", got.name());
    }
    let err = resolve(ServeBackend::Wgpu, "google/gemma-4-26B-A4B-it").unwrap_err();
    eprintln!("[gate] real probe: google/gemma-4-26B-A4B-it -> {err}");
    assert!(
        matches!(err, BackendSelectError::ModelUnservable { .. }),
        "{err}"
    );
}

#[test]
fn the_report_states_the_auto_policy_and_the_build_flag() {
    let report = backends_report();
    eprintln!(
        "[gate] report.wgpu_decoders_compiled_in = {}",
        report["wgpu_decoders_compiled_in"]
    );
    assert_eq!(report["auto_policy"], AUTO_POLICY);
    assert_eq!(
        report["wgpu_decoders_compiled_in"],
        serde_json::Value::Bool(WGPU_DECODERS_COMPILED_IN)
    );
    let policy = report["auto_policy"].as_str().unwrap();
    assert!(policy.contains("cuda-first"), "{policy}");
    assert!(policy.contains("never"), "{policy}");
}

#[test]
fn the_report_names_the_decoder_module_for_every_wgpu_servable_class() {
    let report = backends_report();
    let models = report["models"].as_object().unwrap();

    let dense = &models["nvidia/Gemma-4-31B-IT-NVFP4"]["wgpu"];
    assert_eq!(dense["decoder"]["module"], "nv_models::gemma4_wgpu");
    assert_eq!(dense["decoder"]["evidence"], "real-weights-decode");
    assert_eq!(dense["missing_kernels"].as_array().unwrap().len(), 0);

    let e4b = &models["google/gemma-4-E4B-it"]["wgpu"];
    assert_eq!(e4b["decoder"]["module"], "nv_models::gemma4_e4b_wgpu");
    let fast: Vec<&str> = e4b["cuda_only_fast_paths_replaced_by_wgsl"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_str().unwrap())
        .collect();
    assert!(fast.contains(&"marlin_gemm_w4a16"), "{fast:?}");
    assert_eq!(e4b["missing_kernels"].as_array().unwrap().len(), 0);

    let qwen = &models["RedHatAI/Qwen3.6-35B-A3B-NVFP4"]["wgpu"];
    assert_eq!(qwen["decoder"]["module"], "nv_models::qwen3_5_moe_wgpu");
    let fast: Vec<&str> = qwen["cuda_only_fast_paths_replaced_by_wgsl"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_str().unwrap())
        .collect();
    assert!(
        fast.is_empty(),
        "the qwen moe expert gemm is served by moe_grouped_gemm.wgsl through nv-layers moe_wgpu, \
         so nothing on this class is a cuda-only fast path: {fast:?}"
    );

    let moe = &models["google/gemma-4-26B-A4B-it"]["wgpu"];
    assert_eq!(
        moe["decoder"]["module"], "nv_models::gemma4_moe_wgpu",
        "the 26B-A4B wgpu decoder landed (10db591d5); this report row went stale once already"
    );
    assert_eq!(
        moe["servable"],
        serde_json::Value::Bool(WGPU_DECODERS_COMPILED_IN),
        "the 26B-A4B row must report servable={WGPU_DECODERS_COMPILED_IN} in this build"
    );
    if !WGPU_DECODERS_COMPILED_IN {
        assert_eq!(moe["reason"], WGPU_FEATURE_OFF_REASON);
    }
    assert_eq!(
        moe["decoder"]["evidence"], "architecture-family-only",
        "upgrade this assert to real-weights evidence only after an actual wgpu decode of this \
         exact checkpoint is recorded"
    );

    for (_, entry) in models {
        let w = &entry["wgpu"];
        let servable = w["servable"].as_bool().unwrap();
        assert_eq!(servable, w["reason"].is_null());
        if servable {
            assert!(!w["decoder"].is_null());
            assert_eq!(w["missing_kernels"].as_array().unwrap().len(), 0);
        }
    }
}

#[test]
fn the_report_wgpu_servability_tracks_the_build_feature() {
    let report = backends_report();
    let dense = &report["models"]["nvidia/Gemma-4-31B-IT-NVFP4"]["wgpu"];
    if WGPU_DECODERS_COMPILED_IN {
        assert_eq!(dense["servable"], true);
        assert!(dense["reason"].is_null());
    } else {
        assert_eq!(dense["servable"], false);
        assert_eq!(dense["reason"], WGPU_FEATURE_OFF_REASON);
    }
}
