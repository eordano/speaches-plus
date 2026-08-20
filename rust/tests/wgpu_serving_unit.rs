#[cfg(not(feature = "wgpu"))]
#[test]
fn wgpu_serving_unit_is_cfg_out_without_the_wgpu_feature() {
    eprintln!(
        "wgpu_serving_unit compiled OUT (no `wgpu` feature). This is a SKIP, not a pass: a \
         cfg-out prints 0 passed AND 0 ignored. Re-run with \
         NVK_PKG=speaches-plus NVK_FEATURES=wgpu."
    );
}

#[cfg(feature = "wgpu")]
mod gated {
    use std::path::PathBuf;

    use speaches_plus::oapi::chat::{ChatGenerateRequest, ChatMessageIn, MessageContent, Tool};
    use speaches_plus::oapi::chat_engine_wgpu::{
        classify_wgpu_model, eos_ids_from_dir, model_id_for_dir, render_official_with_tools, spec,
        template_supports_tools, wgpu_spec_decode_status, HostSampler, StopScanner, WgpuModelKind,
    };
    use speaches_plus::oapi::chat_template::ChatTemplate;

    const GEMMA4_31B_REPO: &str = "models--nvidia--Gemma-4-31B-IT-NVFP4";
    const E4B_REPO: &str = "models--google--gemma-4-E4B-it";
    const QWEN36_REPO: &str = "models--RedHatAI--Qwen3.6-35B-A3B-NVFP4";
    const GEMMA4_MOE_REPO: &str = "models--google--gemma-4-26B-A4B-it";

    fn hub_roots() -> Vec<PathBuf> {
        let mut out: Vec<PathBuf> = Vec::new();
        if let Ok(v) = std::env::var("HF_HUB_CACHE") {
            out.push(PathBuf::from(v));
        }
        if let Ok(home) = std::env::var("HOME") {
            out.push(PathBuf::from(home).join(".cache/huggingface/hub"));
        }
        out.retain(|p| p.is_dir());
        out
    }

    fn snapshots_of(repo: &str) -> Vec<PathBuf> {
        let mut out: Vec<PathBuf> = Vec::new();
        for root in hub_roots() {
            let Ok(rd) = std::fs::read_dir(root.join(repo).join("snapshots")) else {
                continue;
            };
            for e in rd.flatten() {
                let p = e.path();
                if p.join("config.json").exists() && !out.contains(&p) {
                    out.push(p);
                }
            }
        }
        out.sort();
        out
    }

    fn hub(repo: &str) -> Option<PathBuf> {
        snapshots_of(repo).into_iter().next()
    }

    fn user(text: &str) -> ChatMessageIn {
        ChatMessageIn {
            role: "user".into(),
            content: Some(MessageContent::Text(text.into())),
            tool_calls: None,
            tool_call_id: None,
            name: None,
        }
    }

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
    fn classify_keys_gemma4_on_per_layer_embeddings_only() {
        let Some(dir) = hub(GEMMA4_31B_REPO) else {
            panic!(
                "no {GEMMA4_31B_REPO} snapshot is cached in {:?}. Returning here printed \
                 `1 passed` in 0.00s having classified nothing. This is a SKIP, not a pass.",
                hub_roots()
            )
        };
        let raw = std::fs::read_to_string(dir.join("config.json")).unwrap();
        let mut v: serde_json::Value = serde_json::from_str(&raw).unwrap();
        assert_eq!(
            classify_wgpu_model(&v.to_string()).unwrap(),
            WgpuModelKind::Gemma4Dense
        );
        v["text_config"]["hidden_size_per_layer_input"] = serde_json::json!(256);
        assert_eq!(
            classify_wgpu_model(&v.to_string()).unwrap(),
            WgpuModelKind::Gemma4E4b
        );

        let qwen = r#"{"architectures":["Qwen3_5MoeForConditionalGeneration"],
            "model_type":"qwen3_5_moe"}"#;
        assert_eq!(
            classify_wgpu_model(qwen).unwrap(),
            WgpuModelKind::Qwen3_5Moe
        );
        let laguna = r#"{"architectures":["LagunaForCausalLM"],"model_type":"laguna"}"#;
        assert_eq!(classify_wgpu_model(laguna).unwrap(), WgpuModelKind::Laguna);
        assert!(classify_wgpu_model(r#"{"model_type":"llama"}"#).is_err());
    }

    #[test]
    fn every_cached_snapshot_wgpu_can_decode_classifies_to_its_decoder() {
        let cases = [
            (GEMMA4_31B_REPO, WgpuModelKind::Gemma4Dense),
            (E4B_REPO, WgpuModelKind::Gemma4E4b),
            (QWEN36_REPO, WgpuModelKind::Qwen3_5Moe),
            (GEMMA4_MOE_REPO, WgpuModelKind::Gemma4Moe),
        ];
        let all: Vec<&str> = cases.iter().map(|(r, _)| *r).collect();
        let mut seen = 0;
        let mut absent: Vec<&str> = Vec::new();
        for (repo, want) in cases {
            let snaps = snapshots_of(repo);
            if snaps.is_empty() {
                absent.push(repo);
                continue;
            }
            for dir in snaps {
                let raw = std::fs::read_to_string(dir.join("config.json")).unwrap();
                let got = classify_wgpu_model(&raw).unwrap();
                eprintln!("{} -> {}", dir.display(), got.label());
                assert_eq!(got, want, "{}", dir.display());
                seen += 1;
            }
        }
        eprintln!("classified {seen} cached snapshot(s); repos not cached here: {absent:?}");
        assert!(
            seen > 0,
            "none of {all:?} is cached in {:?}, so classification was not exercised at all. That \
             is a SKIP, not a pass.",
            hub_roots()
        );
    }

    #[test]
    fn model_id_is_recovered_from_a_hub_snapshot_path() {
        let p =
            std::path::Path::new("/x/hub/models--nvidia--Gemma-4-31B-IT-NVFP4/snapshots/e5ef03af");
        assert_eq!(model_id_for_dir(p), "nvidia/Gemma-4-31B-IT-NVFP4");
        assert_eq!(
            model_id_for_dir(std::path::Path::new("/models/my-local-gemma")),
            "my-local-gemma"
        );
    }

    #[test]
    fn model_id_is_recovered_from_a_nix_store_hf_model_path() {
        let p = std::path::Path::new(
            "/nix/store/0123456789abcdfghijklmnpqrsvwxyz-hf-model-google-gemma-4-E4B-it-qat-\
             w4a16-ct-6cd26aaa2357fb2bad8c51699a7558a4d1a965bb",
        );
        assert_eq!(model_id_for_dir(p), "google/gemma-4-E4B-it-qat-w4a16-ct");
    }

    #[test]
    fn gemma4_eos_set_includes_the_end_of_turn_token() {
        let Some(dir) = hub(GEMMA4_31B_REPO).or_else(|| hub(E4B_REPO)) else {
            panic!(
                "neither {GEMMA4_31B_REPO} nor {E4B_REPO} is cached in {:?}, so no eos set was \
                 read. This is a SKIP, not a pass.",
                hub_roots()
            )
        };
        let ids = eos_ids_from_dir(&dir).unwrap();
        eprintln!("eos ids for {}: {ids:?}", model_id_for_dir(&dir));
        for want in [1u32, 50, 106] {
            assert!(ids.contains(&want), "gemma4 eos set must contain {want}");
        }
    }

    #[test]
    fn the_shipped_gemma4_template_renders_turn_markup_not_chatml() {
        let Some(dir) = hub(GEMMA4_31B_REPO) else {
            panic!(
                "no {GEMMA4_31B_REPO} snapshot is cached in {:?}, so no template was rendered. \
                 This is a SKIP, not a pass.",
                hub_roots()
            )
        };
        let t = ChatTemplate::load_reason(&dir).unwrap();
        let rendered =
            render_official_with_tools(&t, &[user("Name three primary colors.")], &[]).unwrap();
        eprintln!("--- rendered prompt ---\n{rendered}\n--- end ---");
        assert!(rendered.contains("<|turn>user"), "{rendered}");
        assert!(rendered.trim_end().ends_with("<|turn>model") || rendered.contains("<|turn>model"));
        assert!(
            !rendered.contains("<|im_start|>"),
            "ChatML leaked into a Gemma-4 prompt: {rendered}"
        );
    }

    const SNAPSHOT_TEMPLATE_INPUTS: [&str; 4] = [
        "chat_template.jinja",
        "chat_template.json",
        "tokenizer_config.json",
        "generation_config.json",
    ];

    const SYNTHETIC_TEMPLATE_SKEW: &str = "SYNTHETIC_TEMPLATE_SKEW\n";

    const WHY_A_SYNTHETIC_SECOND_SNAPSHOT: &str =
        "One 31B snapshot is cached on most boxes, and with one snapshot the cross-snapshot loop \
         iterates zero times: load and render still bite, but the named property -- that plain \
         no-tool chat is snapshot-independent -- executes on nothing and reports green. Forcing \
         that red would fail a box for legitimately caching one revision, so instead the \
         comparator is exercised against a mirror of the cached snapshot built from the only \
         files ChatTemplate::load_reason reads. The twin proves the equality arm actually runs \
         across two distinct directories; the skewed mirror proves the same comparator FAILS on \
         a template that differs, so a green result over real snapshots is a comparison that \
         happened rather than a loop that was skipped. What this cannot do is stand in for a \
         second real revision: a byte-copy shares the template it is compared against, so \
         cross-revision drift is only ever proven on a box that caches two revisions, and the \
         run output says which of the two situations produced the pass.";

    fn mirror_template_inputs(src: &std::path::Path, name: &str, skew: bool) -> PathBuf {
        let dst = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join(name);
        let _ = std::fs::remove_dir_all(&dst);
        std::fs::create_dir_all(&dst).unwrap();
        let mut copied = 0;
        for f in SNAPSHOT_TEMPLATE_INPUTS {
            let from = src.join(f);
            if !from.is_file() {
                continue;
            }
            let mut body = std::fs::read_to_string(&from).unwrap();
            if skew && f == "chat_template.jinja" {
                body.insert_str(0, SYNTHETIC_TEMPLATE_SKEW);
            }
            std::fs::write(dst.join(f), body).unwrap();
            copied += 1;
        }
        assert!(
            copied > 0,
            "mirrored none of {SNAPSHOT_TEMPLATE_INPUTS:?} out of {}, so the synthetic snapshot \
             would prove nothing",
            src.display()
        );
        dst
    }

    fn render_plain(dir: &std::path::Path) -> String {
        let t = ChatTemplate::load_reason(dir).unwrap();
        render_official_with_tools(&t, &[user("hello")], &[]).unwrap()
    }

    fn first_divergence(renders: &[(PathBuf, String)]) -> Option<String> {
        let (first_dir, first) = renders.first()?;
        renders[1..]
            .iter()
            .find(|(_, r)| r != first)
            .map(|(dir, _)| {
                format!(
                    "plain no-tool chat must be snapshot-independent, but {} and {} differ",
                    first_dir.display(),
                    dir.display()
                )
            })
    }

    #[test]
    fn every_cached_gemma4_31b_snapshot_renders_plain_chat_identically() {
        let snaps = snapshots_of(GEMMA4_31B_REPO);
        assert!(
            !snaps.is_empty(),
            "no {GEMMA4_31B_REPO} snapshot is cached in {:?}; nothing was rendered. That is a \
             SKIP, not a pass.",
            hub_roots()
        );
        let real: Vec<(PathBuf, String)> = snaps
            .iter()
            .map(|d| (d.clone(), render_plain(d)))
            .collect();
        let real_pairs = real.len() - 1;
        if let Some(msg) = first_divergence(&real) {
            panic!("{msg}");
        }

        let base = real[0].0.clone();
        let twin = mirror_template_inputs(&base, "gemma4_31b_twin", false);
        let skewed = mirror_template_inputs(&base, "gemma4_31b_skewed", true);
        let twin_pair = vec![real[0].clone(), (twin.clone(), render_plain(&twin))];
        let skewed_pair = vec![real[0].clone(), (skewed.clone(), render_plain(&skewed))];

        assert!(
            first_divergence(&twin_pair).is_none(),
            "{WHY_A_SYNTHETIC_SECOND_SNAPSHOT}\n\na byte-identical mirror of {} rendered plain \
             chat differently from the snapshot it was copied from, so the render depends on \
             something outside {SNAPSHOT_TEMPLATE_INPUTS:?}",
            base.display()
        );
        assert!(
            first_divergence(&skewed_pair).is_some(),
            "{WHY_A_SYNTHETIC_SECOND_SNAPSHOT}\n\na mirror of {} whose chat_template.jinja was \
             prefixed with {SYNTHETIC_TEMPLATE_SKEW:?} still rendered identically, so this \
             comparison cannot detect a snapshot whose template diverged and the green result \
             over the real snapshots means nothing",
            base.display()
        );

        eprintln!(
            "31B plain-chat render: {} cached snapshot(s) loaded and rendered, {real_pairs} real \
             cross-snapshot pair(s) compared{}. Comparator proven live on a synthetic pair: the \
             byte-identical twin matched and the skewed mirror was caught.",
            real.len(),
            if real_pairs == 0 {
                " -- only one revision is cached here, so snapshot-independence across REAL \
                 revisions was NOT tested"
            } else {
                ""
            }
        );
    }

    #[test]
    fn tools_reach_the_official_template_instead_of_a_synthetic_system_message() {
        let Some(dir) = hub(GEMMA4_31B_REPO).or_else(|| hub(E4B_REPO)) else {
            panic!(
                "neither {GEMMA4_31B_REPO} nor {E4B_REPO} is cached in {:?}, so the tools path \
                 was never rendered. This is a SKIP, not a pass.",
                hub_roots()
            )
        };
        let t = ChatTemplate::load_reason(&dir).unwrap();
        let msgs = [user("what is the weather in Montevideo?")];
        assert!(
            template_supports_tools(&t, &msgs),
            "gemma4's shipped template does reference `tools`"
        );
        let tools: Vec<Tool> = serde_json::from_str(
            r#"[{"type":"function","function":{"name":"get_weather",
                 "description":"Look up the weather","parameters":{"type":"object",
                 "properties":{"city":{"type":"string"}}}}}]"#,
        )
        .unwrap();
        let with = render_official_with_tools(&t, &msgs, &tools).unwrap();
        let without = render_official_with_tools(&t, &msgs, &[]).unwrap();
        assert_ne!(with, without);
        assert!(with.contains("get_weather"), "{with}");
        eprintln!("--- tool prompt delta ---\n{with}\n--- end ---");
    }

    #[test]
    fn stop_scanner_holds_back_partial_stop_strings() {
        let mut s = StopScanner::new(&["<END>".to_string()]);
        assert_eq!(s.step("hello <E"), ("hello ".to_string(), false));
        assert_eq!(s.step("hello <END> tail"), (String::new(), true));
        assert!(s.stopped);
        assert_eq!(s.step("hello <END> more"), (String::new(), true));
    }

    #[test]
    fn stop_scanner_streams_plain_text_and_trims_partial_utf8() {
        let mut s = StopScanner::new(&[]);
        assert_eq!(s.step("abc"), ("abc".to_string(), false));
        assert_eq!(s.step("abcde"), ("de".to_string(), false));
        assert_eq!(s.step("abcde\u{FFFD}"), (String::new(), false));
        assert_eq!(s.step("abcdeé"), ("é".to_string(), false));
        assert_eq!(s.finish("abcdeé!"), "!".to_string());
    }

    #[test]
    fn greedy_requests_do_not_pay_for_a_logits_readback() {
        let base = greedy_request();
        assert!(!HostSampler::new(&base, 1).needs_logits());

        let mut hot = base.clone();
        hot.temperature = Some(0.7);
        assert!(HostSampler::new(&hot, 1).needs_logits());

        let mut topk = base.clone();
        topk.temperature = Some(1.0);
        topk.top_k = Some(20);
        assert!(HostSampler::new(&topk, 1).needs_logits());

        let mut biased = base.clone();
        biased.logit_bias = vec![(3, 1.0)];
        assert!(HostSampler::new(&biased, 1).needs_logits());

        let mut lp = base.clone();
        lp.logprobs = true;
        assert!(HostSampler::new(&lp, 1).needs_logits());

        let mut rep = base.clone();
        rep.repetition_penalty = Some(1.1);
        assert!(HostSampler::new(&rep, 1).needs_logits());
    }

    #[test]
    fn host_sampling_is_deterministic_under_a_fixed_seed_and_differs_across_seeds() {
        let mut req = greedy_request();
        req.temperature = Some(1.0);
        req.top_k = Some(4);
        let logits: Vec<f32> = (0..64).map(|i| ((i * 37) % 11) as f32 * 0.5).collect();
        let run = |seed: u64| -> Vec<u32> {
            let mut s = HostSampler::new(&req, seed);
            (0..16).map(|_| s.pick(&logits).unwrap().token).collect()
        };
        let a = run(7);
        assert_eq!(a, run(7));
        assert_ne!(a, run(9));
        eprintln!("seed 7 -> {a:?}");
    }

    #[test]
    fn greedy_pick_is_argmax() {
        let mut s = HostSampler::new(&greedy_request(), 3);
        let mut logits = vec![0.0f32; 32];
        logits[19] = 5.0;
        assert_eq!(s.pick(&logits).unwrap().token, 19);
    }

    #[test]
    fn logprobs_are_populated_when_requested() {
        let mut req = greedy_request();
        req.logprobs = true;
        req.top_logprobs = 3;
        let mut s = HostSampler::new(&req, 3);
        let mut logits = vec![0.0f32; 32];
        logits[7] = 4.0;
        let p = s.pick(&logits).unwrap();
        assert_eq!(p.token, 7);
        assert!(p.logprob.unwrap() < 0.0);
        assert_eq!(p.top.len(), 3);
        assert_eq!(p.top[0].0, 7);
    }

    #[test]
    fn spec_decode_status_is_on_for_e4b_with_spec_enabled() {
        let knobs = spec::SpecKnobs::parse(None, None, None);
        assert!(knobs.enabled);
        assert_eq!(
            wgpu_spec_decode_status(WgpuModelKind::Gemma4E4b, knobs),
            Some("on")
        );
        assert_eq!(
            speaches_plus::oapi::chat::spec_decode_header_value(wgpu_spec_decode_status(
                WgpuModelKind::Gemma4E4b,
                knobs
            )),
            "on"
        );
    }

    #[test]
    fn spec_decode_status_is_off_when_spec_is_disabled() {
        let knobs = spec::SpecKnobs::parse(Some("0"), None, None);
        assert!(!knobs.enabled);
        assert_eq!(
            wgpu_spec_decode_status(WgpuModelKind::Gemma4E4b, knobs),
            None
        );
        assert_eq!(
            speaches_plus::oapi::chat::spec_decode_header_value(wgpu_spec_decode_status(
                WgpuModelKind::Gemma4E4b,
                knobs
            )),
            "off"
        );
    }

    #[test]
    fn spec_decode_status_is_off_for_kinds_without_a_spec_loop() {
        let knobs = spec::SpecKnobs::parse(None, None, None);
        for kind in [
            WgpuModelKind::Gemma4Dense,
            WgpuModelKind::Gemma4Moe,
            WgpuModelKind::Qwen3_5Moe,
            WgpuModelKind::Qwen3_5Dense,
            WgpuModelKind::GptOss,
            WgpuModelKind::Laguna,
        ] {
            assert_eq!(wgpu_spec_decode_status(kind, knobs), None);
        }
    }
}
