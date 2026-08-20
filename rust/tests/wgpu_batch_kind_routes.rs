#[cfg(not(feature = "wgpu"))]
#[test]
fn wgpu_batch_kind_routes_is_cfg_out_without_the_wgpu_feature() {
    eprintln!(
        "wgpu_batch_kind_routes compiled OUT (no `wgpu` feature). This is a SKIP, not a pass: a \
         cfg-out prints 0 passed AND 0 ignored. Re-run with \
         NVK_PKG=speaches-plus NVK_FEATURES=cuda,wgpu."
    );
}

#[cfg(feature = "wgpu")]
mod gated {
    use speaches_plus::oapi::chat::ChatGenerateRequest;
    use speaches_plus::oapi::chat_engine_wgpu::{
        batch, batch_admits, batch_route_gap, batch_route_refusal, spec, spec_route_eligible,
        WgpuModelKind,
    };

    static ENV_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

    const KINDS_WITH_A_BATCHED_DECODE_GRAPH: [WgpuModelKind; 1] = [WgpuModelKind::Gemma4Dense];

    const KINDS_CARRYING_RECURRENT_STATE: [WgpuModelKind; 3] = [
        WgpuModelKind::Qwen3_5Dense,
        WgpuModelKind::Qwen3_5Moe,
        WgpuModelKind::Laguna,
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
    fn only_the_kinds_with_a_slotted_graph_have_no_batch_gap() {
        for kind in WgpuModelKind::ALL {
            let want_graph = KINDS_WITH_A_BATCHED_DECODE_GRAPH.contains(&kind);
            assert_eq!(
                batch_route_gap(kind).is_none(),
                want_graph,
                "{}: nv-models exports reset_slot/prefill_slot/select_slot/decode_step_batch for \
                 gemma4_wgpu alone, so every other kind must carry a gap the seam can name",
                kind.label()
            );
        }
    }

    #[test]
    fn every_gap_names_the_kind_and_the_missing_pieces() {
        for kind in WgpuModelKind::ALL {
            let text = batch_route_refusal(kind);
            assert!(
                text.contains(kind.label()),
                "a batch refusal that does not name its kind sends an operator to the wrong \
                 decoder: {text}"
            );
            let Some(gap) = batch_route_gap(kind) else {
                continue;
            };
            let missing = gap.missing();
            assert!(
                !missing.is_empty(),
                "{}: a gap with nothing missing is a refusal without a reason",
                kind.label()
            );
            for piece in &missing {
                assert!(
                    text.contains(piece),
                    "{}: the refusal drops one of its own missing pieces: {piece}",
                    kind.label()
                );
            }
            assert!(
                gap.slotted_kv && gap.m_row_decode_step,
                "{}: a kind without a slotted KV region also has no M-row decode step -- the \
                 M-row prefill and verify_chain graphs it does have pack rows of one sequence",
                kind.label()
            );
            assert_eq!(
                gap.per_slot_recurrent_state,
                KINDS_CARRYING_RECURRENT_STATE.contains(&kind),
                "{}: per-slot recurrent state is exactly the DeltaNet/GDN kinds, whose state \
                 buffers are singletons",
                kind.label()
            );
        }
    }

    #[test]
    fn the_two_gap_shapes_differ_only_by_the_recurrent_state_row() {
        let attn = batch::BatchGap::ATTENTION_KV_IN_ONE_REGION;
        let rec = batch::BatchGap::RECURRENT_STATE_IN_ONE_REGION;
        assert_eq!(attn.missing().len() + 1, rec.missing().len());
        assert!(
            rec.to_string().contains("DeltaNet/GDN"),
            "the recurrent gap must say which state it means: {rec}"
        );
        assert!(
            !attn.to_string().contains("DeltaNet/GDN"),
            "an attention-KV kind must not be refused for state it does not hold: {attn}"
        );
    }

    #[test]
    fn a_kind_without_a_batch_graph_is_refused_by_admission_under_that_name() {
        let _lock = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        std::env::remove_var(spec::SPEC_KINDS_ENV);
        let req = greedy_request();
        for kind in WgpuModelKind::ALL {
            let got = batch_admits(kind, &req);
            if KINDS_WITH_A_BATCHED_DECODE_GRAPH.contains(&kind) {
                assert_eq!(got, Ok(()), "{}", kind.label());
                continue;
            }
            let want = if spec_route_eligible(kind, false, spec::SpecKnobs::from_env()) {
                batch::Refusal::MultiRowRoute
            } else {
                batch::Refusal::KindHasNoBatchGraph
            };
            assert_eq!(
                got,
                Err(want),
                "{}: batched admission must state which of the two reasons closed the route",
                kind.label()
            );
        }
    }

    #[test]
    fn a_request_level_refusal_outranks_the_kind_gap() {
        let _lock = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        std::env::remove_var(spec::SPEC_KINDS_ENV);
        let mut sampled = greedy_request();
        sampled.temperature = Some(0.7);
        for kind in WgpuModelKind::ALL {
            assert_eq!(
                batch_admits(kind, &sampled),
                Err(batch::Refusal::NeedsHostLogits),
                "{}: a host-sampled request is refused for what the REQUEST asks for; reporting \
                 the kind gap instead would tell an operator to port a decoder when the fix is \
                 the sampler",
                kind.label()
            );
        }
    }

    #[test]
    fn batching_stays_off_until_the_knob_is_set_and_pays_only_at_the_break_even_width() {
        assert!(
            std::env::var(batch::BATCH_ENV).is_err(),
            "{} is set in this process, so the default-off claim cannot be checked here",
            batch::BATCH_ENV
        );
        assert!(!batch::BatchKnobs::from_env().enabled());
        let width = batch::batch_break_even(nv_models::gemma4_wgpu::MK_MAX)
            .expect("some width in 2..=MK_MAX must pay, or the batch route is unreachable");
        assert!(batch::batch_pays(width) && !batch::batch_pays(width - 1));
        assert!(
            (2..=nv_models::gemma4_wgpu::MK_MAX).contains(&width),
            "the break-even width {width} sits outside the slots the gemma4 dense graph builds"
        );
    }

    #[test]
    fn admission_never_offers_more_slots_than_the_graph_declares() {
        let knobs = batch::BatchKnobs {
            max_batch: 64,
            ..batch::BatchKnobs::default()
        };
        let admission = batch::Admission {
            budget: batch::gemma4_31b_budget(batch::KV_ELEM_BYTES),
            max_seq: 4096,
            knobs,
        };
        for capacity in [1usize, 2, nv_models::gemma4_wgpu::MK_MAX] {
            assert!(
                admission.slots(capacity, None) <= capacity,
                "admission offered more than the {capacity} slots the graph holds, and a batched \
                 step would then advance a sequence the graph cannot create"
            );
        }
        assert_eq!(
            admission.slots(1, None),
            1,
            "a kind whose BatchStepper declares one slot must never be handed a batch"
        );
    }
}
