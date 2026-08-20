#[cfg(not(feature = "wgpu"))]
#[test]
fn wgpu_spec_kind_routes_is_cfg_out_without_the_wgpu_feature() {
    eprintln!(
        "wgpu_spec_kind_routes compiled OUT (no `wgpu` feature). This is a SKIP, not a pass: a \
         cfg-out prints 0 passed AND 0 ignored. Re-run with \
         NVK_PKG=speaches-plus NVK_FEATURES=cuda,wgpu."
    );
}

#[cfg(feature = "wgpu")]
mod gated {
    use speaches_plus::oapi::chat::ChatGenerateRequest;
    use speaches_plus::oapi::chat_engine_wgpu::{
        batch, batch_admits, spec, spec_route_eligible, wgpu_spec_decode_status, WgpuModelKind,
    };

    static ENV_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

    const ALL_KINDS: [WgpuModelKind; 7] = [
        WgpuModelKind::Gemma4Dense,
        WgpuModelKind::Gemma4E4b,
        WgpuModelKind::Gemma4Moe,
        WgpuModelKind::Qwen3_5Moe,
        WgpuModelKind::Qwen3_5Dense,
        WgpuModelKind::GptOss,
        WgpuModelKind::Laguna,
    ];

    const KINDS_WITH_A_CHAIN_TARGET_ARM: [WgpuModelKind; 3] = [
        WgpuModelKind::Gemma4E4b,
        WgpuModelKind::Qwen3_5Dense,
        WgpuModelKind::GptOss,
    ];

    fn greedy_request() -> ChatGenerateRequest {
        ChatGenerateRequest {
            prompt: String::new(),
            max_new_tokens: 8,
            stop: Vec::new(),
            seed: Some(1),
            temperature: None,
            top_p: None,
            top_k: None,
            min_p: None,
            presence_penalty: None,
            frequency_penalty: None,
            repetition_penalty: None,
            guided: None,
            guided_think_close: None,
            logit_bias: Vec::new(),
            logprobs: false,
            top_logprobs: 0,
            kv_resume: None,
            kv_store: None,
            mm: None,
        }
    }

    #[test]
    fn without_the_opt_in_env_only_gemma4_e4b_reaches_the_chain_route() {
        let knobs = spec::SpecKnobs::parse(None, None, None);
        assert!(knobs.enabled, "NV_WGPU_SPEC defaults on");
        for kind in ALL_KINDS {
            let want = kind == WgpuModelKind::Gemma4E4b;
            assert_eq!(
                spec_route_eligible(kind, false, knobs),
                want,
                "{}: {}",
                kind.label(),
                spec::SPEC_KINDS_DEFAULT_IS_GEMMA4_E4B_ALONE_BECAUSE_A_VERIFY_FORWARD_IS_NOT_A_MEASURED_WIN
            );
            assert_eq!(
                wgpu_spec_decode_status(kind, knobs),
                want.then_some("on"),
                "{}: the spec-decode header must not claim a route admission does not make",
                kind.label()
            );
        }
    }

    #[test]
    fn naming_a_kind_in_the_env_admits_that_kind_and_leaves_the_rest_refused() {
        for (list, kind) in [
            ("qwen3.5-dense", WgpuModelKind::Qwen3_5Dense),
            ("qwen3.8", WgpuModelKind::Qwen3_5Dense),
            ("gpt-oss", WgpuModelKind::GptOss),
            ("  GPT-OSS  ", WgpuModelKind::GptOss),
        ] {
            let knobs = spec::SpecKnobs::parse_with_kinds(None, None, None, Some(list));
            assert!(
                spec_route_eligible(kind, false, knobs),
                "NV_WGPU_SPEC_KINDS={list} must admit {}",
                kind.label()
            );
            assert_eq!(wgpu_spec_decode_status(kind, knobs), Some("on"));
            assert!(
                spec_route_eligible(WgpuModelKind::Gemma4E4b, false, knobs),
                "an opt-in list must not evict the default kind"
            );
            for other in ALL_KINDS {
                if other == kind || other == WgpuModelKind::Gemma4E4b {
                    continue;
                }
                assert!(
                    !spec_route_eligible(other, false, knobs),
                    "NV_WGPU_SPEC_KINDS={list} must not admit {}",
                    other.label()
                );
            }
        }
    }

    #[test]
    fn two_kinds_at_once_are_admitted_and_unknown_slugs_admit_nothing() {
        let both = spec::SpecKnobs::parse_with_kinds(None, None, None, Some("qwen3.8, gpt-oss"));
        for kind in KINDS_WITH_A_CHAIN_TARGET_ARM {
            assert!(spec_route_eligible(kind, false, both), "{}", kind.label());
        }
        let junk =
            spec::SpecKnobs::parse_with_kinds(None, None, None, Some("qwen3.9,,gptoss,gemma4"));
        for kind in ALL_KINDS {
            assert_eq!(
                spec_route_eligible(kind, false, junk),
                kind == WgpuModelKind::Gemma4E4b,
                "an unrecognized slug must admit nothing: {}",
                kind.label()
            );
        }
    }

    #[test]
    fn kinds_without_a_chain_target_arm_have_no_slug_that_admits_them() {
        let list = "gemma4-dense,gemma4-moe,qwen3.5-moe,laguna,laguna-xs";
        let knobs = spec::SpecKnobs::parse_with_kinds(None, None, None, Some(list));
        for kind in ALL_KINDS {
            if KINDS_WITH_A_CHAIN_TARGET_ARM.contains(&kind) {
                continue;
            }
            assert!(
                kind.spec_chain_slug().is_empty(),
                "{}: {}",
                kind.label(),
                spec::SPEC_KINDS_ABSENT_FROM_THE_SLUG_TABLE_HAVE_NO_CHAIN_TARGET_ARM_ON_THIS_SEAM
            );
            assert!(
                !spec_route_eligible(kind, false, knobs),
                "{}: {}",
                kind.label(),
                spec::SPEC_KINDS_ABSENT_FROM_THE_SLUG_TABLE_HAVE_NO_CHAIN_TARGET_ARM_ON_THIS_SEAM
            );
        }
    }

    #[test]
    fn every_slug_in_the_table_maps_back_to_a_kind_with_a_chain_target_arm() {
        for slugs in spec::SPEC_CHAIN_ROUTE_SLUGS {
            for &slug in slugs {
                let knobs = spec::SpecKnobs::parse_with_kinds(None, None, None, Some(slug));
                let admitted: Vec<WgpuModelKind> = ALL_KINDS
                    .into_iter()
                    .filter(|k| spec_route_eligible(*k, false, knobs))
                    .collect();
                assert!(
                    admitted
                        .iter()
                        .all(|k| KINDS_WITH_A_CHAIN_TARGET_ARM.contains(k)),
                    "slug {slug} admitted {admitted:?} but only \
                     {KINDS_WITH_A_CHAIN_TARGET_ARM:?} have a ChainVerifyTarget arm"
                );
            }
        }
    }

    #[test]
    fn spec_off_and_host_logits_outrank_any_kind_list() {
        let off = spec::SpecKnobs::parse_with_kinds(Some("0"), None, None, Some("qwen3.8,gpt-oss"));
        for kind in ALL_KINDS {
            assert!(!spec_route_eligible(kind, false, off), "{}", kind.label());
            assert_eq!(wgpu_spec_decode_status(kind, off), None, "{}", kind.label());
        }
        let on = spec::SpecKnobs::parse_with_kinds(None, None, None, Some("qwen3.8,gpt-oss"));
        for kind in KINDS_WITH_A_CHAIN_TARGET_ARM {
            assert!(
                !spec_route_eligible(kind, true, on),
                "{}: a request that needs host logits has no argmax-only verify to accept against",
                kind.label()
            );
        }
    }

    #[test]
    fn an_admitted_kind_is_refused_by_batched_admission_as_a_multi_row_route() {
        let _lock = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let req = greedy_request();
        std::env::remove_var(spec::SPEC_KINDS_ENV);
        assert!(
            !matches!(
                batch_admits(WgpuModelKind::Qwen3_5Dense, &req),
                Err(batch::Refusal::MultiRowRoute)
            ),
            "without the opt-in the qwen3.5-dense chain route does not own the row axis"
        );
        std::env::set_var(spec::SPEC_KINDS_ENV, "qwen3.8,gpt-oss");
        for kind in [WgpuModelKind::Qwen3_5Dense, WgpuModelKind::GptOss] {
            assert!(
                matches!(
                    batch_admits(kind, &req),
                    Err(batch::Refusal::MultiRowRoute)
                ),
                "{}: an admitted chain route and the batched decode graph both own the row axis",
                kind.label()
            );
        }
        std::env::remove_var(spec::SPEC_KINDS_ENV);
        assert!(
            !spec_route_eligible(
                WgpuModelKind::GptOss,
                false,
                spec::SpecKnobs::from_env()
            ),
            "removing NV_WGPU_SPEC_KINDS must close the route again"
        );
    }

    #[test]
    fn chain_capacity_falls_back_to_one_row_without_a_multi_row_verify_or_without_room() {
        assert_eq!(spec::chain_capacity(0, 0, 0, 4096), 1);
        assert_eq!(spec::chain_capacity(1, 0, 16, 4096), 1);
        assert_eq!(spec::chain_capacity(16, 0, 16, 4096), 16);
        assert_eq!(spec::chain_capacity(16, 4080, 16, 4096), 16);
        assert_eq!(
            spec::chain_capacity(16, 4081, 16, 4096),
            1,
            "a verify forward writes a whole baked chunk of KV rows, so the tail of the window \
             must step one row at a time"
        );
        assert_eq!(
            spec::chain_capacity(8, 4090, 4, 4096),
            1,
            "the span the guard uses is the wider of the verify rows and the prefill chunk"
        );
    }
}
