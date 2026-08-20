#[cfg(not(feature = "wgpu"))]
#[test]
fn wgpu_guided_unit_is_cfg_out_without_the_wgpu_feature() {
    eprintln!(
        "wgpu_guided_unit compiled OUT (no `wgpu` feature). This is a SKIP, not a pass: a \
         cfg-out prints 0 passed AND 0 ignored. Re-run with --features wgpu."
    );
}

#[cfg(feature = "wgpu")]
mod gated {
    use std::sync::Arc;

    use serde_json::json;

    use nv_grammar::{GuidedDecoder, VocabBytes};
    use speaches_plus::oapi::chat::ChatGenerateRequest;
    use speaches_plus::oapi::chat_engine_wgpu::HostSampler;

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

    fn vocab(strs: &[&str], eos: &[u32]) -> Arc<VocabBytes> {
        Arc::new(VocabBytes::new(
            strs.iter().map(|s| s.as_bytes().to_vec()).collect(),
            eos,
        ))
    }

    fn boolean_sampler(req: &ChatGenerateRequest, seed: u64) -> HostSampler {
        let mut s = HostSampler::new(req, seed);
        let v = vocab(&["true", "false", "9", "{", ""], &[4]);
        s.set_guided(
            GuidedDecoder::from_schema(&json!({"type": "boolean"}), v).unwrap(),
            req.max_new_tokens,
        );
        s
    }

    #[test]
    fn guided_forces_the_logits_path_even_for_greedy() {
        let req = greedy_request();
        assert!(
            !HostSampler::new(&req, 1).needs_logits(),
            "plain greedy must keep the in-shader argmax fast path"
        );
        assert!(
            boolean_sampler(&req, 1).needs_logits(),
            "a grammar must force the host-logits path"
        );
    }

    #[test]
    fn guided_greedy_masks_the_argmax_and_then_forces_eos() {
        let req = greedy_request();
        let mut s = boolean_sampler(&req, 1);
        let logits = vec![0.0, 1.0, 5.0, 4.0, 3.0];
        assert_eq!(
            s.pick(&logits).unwrap().token,
            1,
            "raw argmax is the grammar-illegal token '9'; the mask must reroute to 'false'"
        );
        assert_eq!(
            s.pick(&logits).unwrap().token,
            4,
            "grammar complete: every content token is masked, only eos stays legal"
        );
    }

    #[test]
    fn guided_json_schema_drives_adversarial_logits_to_schema_valid_json() {
        let schema = json!({
            "type": "object",
            "properties": {"a": {"type": "integer"}},
            "required": ["a"]
        });
        let toks = ["x", "{", "\"a\"", ":", "12", "}", ""];
        let v = vocab(&toks, &[6]);
        let req = greedy_request();
        let mut s = HostSampler::new(&req, 1);
        s.set_guided(
            GuidedDecoder::from_schema(&schema, v).unwrap(),
            req.max_new_tokens,
        );
        let logits = vec![9.0, 0.5, 0.4, 0.3, 0.2, 0.25, 0.05];
        let mut out = String::new();
        let mut hit_eos = false;
        for _ in 0..32 {
            let t = s.pick(&logits).unwrap().token as usize;
            if t == 6 {
                hit_eos = true;
                break;
            }
            out.push_str(toks[t]);
        }
        assert!(hit_eos, "grammar never released eos: {out:?}");
        let parsed: serde_json::Value =
            serde_json::from_str(&out).unwrap_or_else(|e| panic!("invalid JSON {out:?}: {e}"));
        assert!(parsed["a"].is_number(), "schema violated: {out:?}");
    }

    #[test]
    fn guided_sampling_is_deterministic_under_a_fixed_seed() {
        let schema = json!({
            "type": "object",
            "properties": {"a": {"type": "integer"}},
            "required": ["a"]
        });
        let toks = ["x", "{", "\"a\"", ":", "1", "2", "}", ""];
        let mut req = greedy_request();
        req.temperature = Some(0.9);
        let logits = vec![3.0, 1.0, 1.0, 1.0, 1.0, 1.2, 1.1, 0.5];
        let run = |seed: u64| -> Vec<u32> {
            let v = vocab(&toks, &[7]);
            let mut s = HostSampler::new(&req, seed);
            s.set_guided(
                GuidedDecoder::from_schema(&schema, v).unwrap(),
                req.max_new_tokens,
            );
            let mut seq = Vec::new();
            for _ in 0..64 {
                let t = s.pick(&logits).unwrap().token;
                seq.push(t);
                if t == 7 {
                    break;
                }
            }
            seq
        };
        let a = run(11);
        assert_eq!(a, run(11), "same seed must replay the same guided sample");
        assert_eq!(*a.last().unwrap(), 7, "sampled run must terminate at eos");
        eprintln!("guided sampled seq (seed 11): {a:?}");
    }

    #[test]
    fn repetition_penalty_moves_the_greedy_pick_off_the_repeated_token() {
        let req = greedy_request();
        let logits = vec![1.0, 0.95, 0.0, 0.0];
        let mut plain = HostSampler::new(&req, 1);
        assert_eq!(plain.pick(&logits).unwrap().token, 0);
        assert_eq!(
            plain.pick(&logits).unwrap().token,
            0,
            "without penalties greedy repeats the argmax"
        );

        let mut penalized = greedy_request();
        penalized.repetition_penalty = Some(1.5);
        let mut pen = HostSampler::new(&penalized, 1);
        assert!(pen.needs_logits());
        assert_eq!(pen.pick(&logits).unwrap().token, 0);
        assert_eq!(
            pen.pick(&logits).unwrap().token,
            1,
            "repetition penalty must demote the repeated token below the runner-up"
        );
    }

    #[test]
    fn logit_bias_and_grammar_compose() {
        let req = {
            let mut r = greedy_request();
            r.logit_bias = vec![(2, 50.0)];
            r
        };
        let mut s = HostSampler::new(&req, 1);
        let v = vocab(&["true", "false", "9", "{", ""], &[4]);
        s.set_guided(
            GuidedDecoder::from_schema(&json!({"type": "boolean"}), v).unwrap(),
            req.max_new_tokens,
        );
        let logits = vec![0.0, 1.0, 5.0, 4.0, 3.0];
        assert_eq!(
            s.pick(&logits).unwrap().token,
            1,
            "a +50 bias on an illegal token must not beat the grammar mask"
        );
    }

    #[test]
    fn a_runaway_thought_is_closed_so_the_grammar_always_arms() {
        let req = greedy_request();
        let mut s = boolean_sampler(&req, 1);
        let close = 3u32;
        s.guided_mut().unwrap().set_defer_until_token(close);
        let logits = vec![0.0, 1.0, 5.0, 0.5, 3.0];
        let budget = req.max_new_tokens.div_ceil(2);
        for i in 0..budget {
            let t = s.pick(&logits).unwrap().token;
            assert_eq!(
                t, 2,
                "step {i}: while thinking, the grammar must not mask anything -- the raw argmax \
                 token '9' is schema-illegal and must still win"
            );
            assert!(s.guided().unwrap().deferred(), "step {i}: still thinking");
        }
        assert_eq!(
            s.pick(&logits).unwrap().token,
            close,
            "the thinking budget is spent, so the close marker is the only legal token: without \
             this the model thinks to max_new_tokens, the grammar never arms, and a caller who \
             asked for a schema silently receives prose"
        );
        assert!(
            !s.guided().unwrap().deferred(),
            "emitting the close marker arms the grammar"
        );
        assert_eq!(
            s.pick(&logits).unwrap().token,
            1,
            "grammar now live: argmax '9' is masked and 'false' wins"
        );
    }

    #[test]
    fn the_thinking_budget_is_measured_against_the_cap_the_engine_will_actually_serve() {
        let mut req = greedy_request();
        req.max_new_tokens = 4096;
        let served = 8usize;
        let mut s = HostSampler::new(&req, 1);
        let v = vocab(&["true", "false", "9", "{", ""], &[4]);
        s.set_guided(
            GuidedDecoder::from_schema(&json!({"type": "boolean"}), v).unwrap(),
            served,
        );
        s.guided_mut().unwrap().set_defer_until_token(3);
        let logits = vec![0.0, 1.0, 5.0, 0.5, 3.0];

        let mut closed = None;
        for step in 0..served {
            if s.pick(&logits).unwrap().token == 3 {
                closed = Some(step);
                break;
            }
        }
        assert_eq!(
            closed,
            Some(served.div_ceil(2)),
            "a prompt that fills the KV window leaves the engine far fewer tokens than the \
             caller requested; a budget taken from the request is never reached inside the \
             served cap, so the grammar never arms and the schema is silently dropped"
        );
    }

    #[test]
    fn a_runaway_thought_is_closed_even_when_the_close_marker_is_many_tokens() {
        let req = greedy_request();
        let mut s = HostSampler::new(&req, 1);
        let v = vocab(
            &[
                "true",
                "false",
                "9",
                "<|end|>",
                "<|start|>",
                "final",
                "<|message|>",
                "",
            ],
            &[7],
        );
        s.set_guided(
            GuidedDecoder::from_schema(&json!({"type": "boolean"}), v).unwrap(),
            req.max_new_tokens,
        );
        let close = [3u32, 4, 5, 6];
        s.guided_mut().unwrap().set_defer_until_sequence(&close);
        let logits = vec![0.0, 1.0, 5.0, 0.5, 0.4, 0.3, 0.2, 3.0];

        let budget = req.max_new_tokens.div_ceil(2);
        for i in 0..budget {
            assert_eq!(
                s.pick(&logits).unwrap().token,
                2,
                "step {i}: while thinking the grammar must not mask anything"
            );
            assert!(s.guided().unwrap().deferred(), "step {i}: still thinking");
        }

        for (i, want) in close.iter().enumerate() {
            assert_eq!(
                s.pick(&logits).unwrap().token,
                *want,
                "forced-close step {i}: the budget is spent, so the only legal token is the next \
                 one of the model's own close sequence"
            );
        }
        assert!(
            !s.guided().unwrap().deferred(),
            "the forced close must terminate: a close marker of N tokens takes exactly N steps, \
             after which the grammar is live and the schema still has the reserve to be written"
        );
        assert_eq!(
            s.pick(&logits).unwrap().token,
            1,
            "grammar now live: argmax '9' is masked and 'false' wins"
        );
    }
}
