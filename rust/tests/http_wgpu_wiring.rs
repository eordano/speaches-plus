const WHY_THE_CONST_IS_NOT_AN_ORACLE: &str =
    "WGPU_DECODERS_COMPILED_IN is defined as `cfg!(feature = \"wgpu\")`, so comparing it to \
     cfg!(feature = \"wgpu\") is true == true by construction. That assertion was the whole body \
     of this test and stayed green under a planted mutation that made WgpuRegistryPlan::decide \
     return Delegate unconditionally -- i.e. with the chat router structurally unable to reach a \
     wgpu engine at all. Reachability has two halves and both must be asserted: the selection \
     point must route a wgpu-servable id to the wgpu backend, and the registry plan must hand \
     that model directory to the wgpu loader instead of delegating it to cuda.";

#[test]
fn the_chat_router_can_reach_a_wgpu_engine_in_this_build() {
    use nv_layers::backend::BackendKind;
    use speaches_plus::oapi::backend_select::{
        resolve_with, BackendSelectError, ServeBackend, WGPU_DECODERS_COMPILED_IN,
        WGPU_FEATURE_OFF_REASON,
    };

    let cuda_probe = || -> Result<(), String> { panic!("the cuda probe must not run for an explicit wgpu selection") };
    let wgpu_probe = || -> Result<(), String> { Ok(()) };
    let id = "nvidia/Gemma-4-31B-IT-NVFP4";
    let selected = resolve_with(ServeBackend::Wgpu, id, &cuda_probe, &wgpu_probe);

    if cfg!(feature = "wgpu") {
        assert!(
            WGPU_DECODERS_COMPILED_IN,
            "this binary was compiled with --features wgpu but backend_select reports otherwise"
        );
        assert_eq!(
            selected.unwrap_or_else(|e| panic!("{id} must be reachable on wgpu here: {e}")),
            BackendKind::Wgpu,
            "{WHY_THE_CONST_IS_NOT_AN_ORACLE}"
        );
    } else {
        let err = selected.expect_err("without the wgpu feature no model is reachable on wgpu");
        assert!(
            matches!(err, BackendSelectError::ModelUnservable { .. }),
            "{err}"
        );
        assert!(
            err.to_string().contains(WGPU_FEATURE_OFF_REASON),
            "the refusal must name the missing feature, not something incidental: {err}"
        );
        eprintln!(
            "wgpu feature OFF: the HTTP binary cannot serve wgpu in this build, which is \
             correct and is what backend_select reports."
        );
    }
}

#[cfg(feature = "wgpu")]
mod wired {
    use std::path::PathBuf;
    use std::sync::Arc;

    use speaches_plus::oapi::chat::ChatEngine;
    use speaches_plus::oapi::chat_engine::{ChatRegistry, EchoEngine};
    use speaches_plus::oapi::chat_engine_wgpu::{
        alias_engine, registered_wgpu_model_ids, split_model_dirs, wgpu_id_after_collisions,
        WgpuRegistryPlan, CHAT_MODEL_DIRS_ENV, CHAT_MODEL_DIR_ENV, SERVE_BACKEND_ENV,
        WGPU_ALIAS_SUFFIX, WGPU_CHAT_MODEL_DIRS_ENV,
    };

    fn p(s: &str) -> PathBuf {
        PathBuf::from(s)
    }

    #[test]
    fn no_wgpu_env_delegates_to_the_existing_cuda_selection_point() {
        assert_eq!(
            WgpuRegistryPlan::decide(None, None, None, Some("/models/a")),
            WgpuRegistryPlan::Delegate
        );
        assert_eq!(
            WgpuRegistryPlan::decide(Some("cuda"), None, Some("/models/a,/models/b"), None),
            WgpuRegistryPlan::Delegate
        );
        assert_eq!(
            WgpuRegistryPlan::decide(Some("auto"), None, None, Some("/models/a")),
            WgpuRegistryPlan::Delegate
        );
    }

    #[test]
    fn wgpu_model_dirs_alone_extend_the_registry() {
        assert_eq!(
            WgpuRegistryPlan::decide(None, Some("/models/w"), None, Some("/models/c")),
            WgpuRegistryPlan::Extend(vec![p("/models/w")])
        );
    }

    #[test]
    fn explicit_wgpu_backend_replaces_so_the_cuda_loader_is_never_invoked() {
        assert_eq!(
            WgpuRegistryPlan::decide(Some("wgpu"), Some("/models/w"), None, Some("/models/c")),
            WgpuRegistryPlan::Replace(vec![p("/models/w")])
        );
        assert_eq!(
            WgpuRegistryPlan::decide(Some("wgpu"), None, None, Some("/models/c")),
            WgpuRegistryPlan::Replace(vec![p("/models/c")]),
            "NV_SERVE_BACKEND=wgpu with only NV_CHAT_MODEL_DIR set must route that dir to wgpu, \
             not load it on cuda"
        );
        assert_eq!(
            WgpuRegistryPlan::decide(Some(" WGPU "), None, Some("/models/a:/models/b"), None),
            WgpuRegistryPlan::Replace(vec![p("/models/a"), p("/models/b")])
        );
    }

    #[test]
    fn chat_model_dirs_wins_over_chat_model_dir_like_the_cuda_loader() {
        assert_eq!(
            WgpuRegistryPlan::decide(
                Some("wgpu"),
                None,
                Some("/models/list"),
                Some("/models/one")
            ),
            WgpuRegistryPlan::Replace(vec![p("/models/list")])
        );
    }

    #[test]
    fn a_garbage_backend_value_does_not_silently_select_wgpu() {
        assert_eq!(
            WgpuRegistryPlan::decide(Some("vulkan"), None, None, Some("/models/c")),
            WgpuRegistryPlan::Delegate
        );
        assert_eq!(
            WgpuRegistryPlan::decide(Some(""), None, None, Some("/models/c")),
            WgpuRegistryPlan::Delegate
        );
    }

    #[test]
    fn empty_and_whitespace_dir_lists_are_not_plans() {
        assert_eq!(
            WgpuRegistryPlan::decide(None, Some("  ,, : "), None, None),
            WgpuRegistryPlan::Delegate
        );
        assert!(split_model_dirs(" , : ").is_empty());
        assert_eq!(
            split_model_dirs("/a, /b:/c"),
            vec![p("/a"), p("/b"), p("/c")]
        );
    }

    #[test]
    fn a_wgpu_engine_never_silently_shadows_a_cuda_engine_of_the_same_id() {
        let taken = vec!["google/gemma-4-E4B-it".to_string()];
        assert_eq!(
            wgpu_id_after_collisions(&taken, "google/gemma-4-E4B-it"),
            format!("google/gemma-4-E4B-it{WGPU_ALIAS_SUFFIX}")
        );
        assert_eq!(
            wgpu_id_after_collisions(&taken, "nvidia/Gemma-4-31B-IT-NVFP4"),
            "nvidia/Gemma-4-31B-IT-NVFP4"
        );
    }

    #[test]
    fn an_aliased_engine_is_addressable_by_its_alias_and_delegates_everything() {
        let inner: Arc<dyn ChatEngine> = Arc::new(EchoEngine::new("m", "hello there"));
        let aliased = alias_engine(format!("m{WGPU_ALIAS_SUFFIX}"), inner.clone());
        assert_eq!(aliased.model_id(), "m#wgpu");
        assert_eq!(aliased.render_prompt(&[]), inner.render_prompt(&[]));

        let reg = ChatRegistry::from_engines(vec![inner, aliased]).expect("registry");
        assert_eq!(reg.model_ids(), ["m".to_string(), "m#wgpu".to_string()]);
        assert_eq!(
            reg.resolve_with(Some("m#wgpu"), false).unwrap().model_id(),
            "m#wgpu"
        );
        assert_eq!(
            reg.resolve_with(Some("m"), false).unwrap().model_id(),
            "m",
            "the cuda engine keeps its plain id; nothing is overwritten"
        );
    }

    const WHY_FROM_ENV_NEEDS_A_SET_ENVIRONMENT: &str =
        "Printing WgpuRegistryPlan::from_env() asserts nothing. Under an environment where all \
         four variables are unset -- which is every test run -- from_env() returns Delegate no \
         matter which variable names it reads, so a rename would go unnoticed. The only oracle \
         that bites is to set each documented name in turn and require from_env() to produce the \
         plan decide() produces for that value.";

    #[test]
    fn from_env_reads_the_documented_variable_names() {
        assert_eq!(WGPU_CHAT_MODEL_DIRS_ENV, "NV_WGPU_CHAT_MODEL_DIRS");
        assert_eq!(SERVE_BACKEND_ENV, "NV_SERVE_BACKEND");
        assert_eq!(CHAT_MODEL_DIRS_ENV, "NV_CHAT_MODEL_DIRS");
        assert_eq!(CHAT_MODEL_DIR_ENV, "NV_CHAT_MODEL_DIR");

        let names = [
            SERVE_BACKEND_ENV,
            WGPU_CHAT_MODEL_DIRS_ENV,
            CHAT_MODEL_DIRS_ENV,
            CHAT_MODEL_DIR_ENV,
        ];
        let saved: Vec<Option<String>> = names.iter().map(|n| std::env::var(n).ok()).collect();
        for n in names {
            std::env::remove_var(n);
        }

        let cases: [(Option<&str>, Option<&str>, Option<&str>, Option<&str>); 4] = [
            (None, Some("/models/w"), None, Some("/models/c")),
            (Some("wgpu"), None, None, Some("/models/c")),
            (Some("wgpu"), None, Some("/models/a:/models/b"), None),
            (None, None, None, Some("/models/c")),
        ];
        let mut outcomes = Vec::new();
        for (backend, wgpu_dirs, listed, single) in cases {
            for (n, v) in names.iter().zip([backend, wgpu_dirs, listed, single]) {
                match v {
                    Some(v) => std::env::set_var(n, v),
                    None => std::env::remove_var(n),
                }
            }
            let got = WgpuRegistryPlan::from_env();
            let want = WgpuRegistryPlan::decide(backend, wgpu_dirs, listed, single);
            assert_eq!(
                got, want,
                "from_env() disagrees with decide({backend:?}, {wgpu_dirs:?}, {listed:?}, \
                 {single:?}). {WHY_FROM_ENV_NEEDS_A_SET_ENVIRONMENT}"
            );
            outcomes.push(got);
        }
        for (n, v) in names.iter().zip(saved) {
            match v {
                Some(v) => std::env::set_var(n, v),
                None => std::env::remove_var(n),
            }
        }

        assert!(
            outcomes.iter().any(|p| matches!(p, WgpuRegistryPlan::Replace(_)))
                && outcomes.iter().any(|p| matches!(p, WgpuRegistryPlan::Extend(_)))
                && outcomes.iter().any(|p| matches!(p, WgpuRegistryPlan::Delegate)),
            "the cases must reach all three plans, or a from_env() stubbed to one of them still \
             passes: {outcomes:?}"
        );
        eprintln!(
            "registered wgpu model ids = {:?}",
            registered_wgpu_model_ids()
        );
    }
}
