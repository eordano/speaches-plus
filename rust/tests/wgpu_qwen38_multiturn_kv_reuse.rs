#[cfg(not(feature = "wgpu"))]
#[test]
fn wgpu_qwen38_multiturn_kv_reuse_is_cfg_out_without_the_wgpu_feature() {
    eprintln!(
        "wgpu_qwen38_multiturn_kv_reuse compiled OUT (no `wgpu` feature). This is a SKIP, not a \
         pass: re-run with --features wgpu and NV_QWEN38_KV_REUSE_TEST=1."
    );
}

#[cfg(feature = "wgpu")]
mod gated {
    use std::path::PathBuf;
    use std::sync::Arc;
    use std::time::Instant;

    use speaches_plus::oapi::chat::{
        ChatEngine, ChatEvent, ChatGenerateRequest, ChatMessageIn, MessageContent,
    };
    use speaches_plus::oapi::chat_engine_wgpu::{WgpuChatEngine, PREFIX_REUSE_ENV_DEFAULT_ON_SET_0_FOR_THE_COLD_PREFILL_EVERY_TURN_ARM};

    const GATE_ENV: &str = "NV_QWEN38_KV_REUSE_TEST";

    const REPO_DIR: &str = "models--unsloth--Qwen3.8-27B-NVFP4";

    const MAX_SEQ: usize = 8192;

    const TTFT_COLLAPSE_FACTOR_HALF_IS_FAR_ABOVE_THE_EXPECTED_DELTA_ONLY_PREFILL: f64 = 0.5;

    struct EnvGuard;

    impl Drop for EnvGuard {
        fn drop(&mut self) {
            std::env::remove_var(PREFIX_REUSE_ENV_DEFAULT_ON_SET_0_FOR_THE_COLD_PREFILL_EVERY_TURN_ARM);
        }
    }

    fn require_gate() {
        if std::env::var(GATE_ENV).ok().as_deref() != Some("1") {
            panic!(
                "this test is #[ignore]d, so it was asked for BY NAME, but {GATE_ENV}=1 is not \
                 set. Returning here prints `1 passed` in 0.00s having served nothing. This is a \
                 SKIP, not a pass."
            );
        }
    }

    fn model_dir() -> PathBuf {
        if let Ok(d) = std::env::var("NV_QWEN38_DIR") {
            let p = PathBuf::from(&d);
            assert!(
                p.join("config.json").exists() && p.join("tokenizer.json").exists(),
                "NV_QWEN38_DIR={d} lacks config.json/tokenizer.json"
            );
            return p;
        }
        let mut hubs: Vec<PathBuf> = Vec::new();
        if let Ok(h) = std::env::var("HF_HUB_CACHE") {
            hubs.push(PathBuf::from(h));
        }
        if let Ok(home) = std::env::var("HOME") {
            hubs.push(PathBuf::from(home).join(".cache/huggingface/hub"));
        }
        let mut cand: Vec<PathBuf> = hubs
            .iter()
            .filter_map(|hub| std::fs::read_dir(hub.join(REPO_DIR).join("snapshots")).ok())
            .flatten()
            .flatten()
            .map(|e| e.path())
            .filter(|p| {
                p.join("config.json").exists()
                    && p.join("tokenizer.json").exists()
                    && p.join("model.safetensors").exists()
            })
            .collect();
        cand.sort();
        cand.pop().unwrap_or_else(|| {
            panic!("no complete {REPO_DIR} snapshot under any of {hubs:?}; set NV_QWEN38_DIR")
        })
    }

    fn passage() -> String {
        let repeats: usize = std::env::var("NV_QWEN38_KV_REUSE_REPEATS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(60);
        let mut s = String::from(
            "Here is the reference document for this conversation; every follow-up question is \
             about it. ",
        );
        for i in 0..repeats {
            s.push_str(&format!(
                "Paragraph {i}: Polynesian wayfinders crossed thousands of miles of open ocean \
                 by reading star paths, swell patterns, and the flight of birds, long before the \
                 magnetic compass reached Europe; later the marine chronometer solved the \
                 longitude problem, and satellite positioning made a fix a matter of \
                 milliseconds. "
            ));
        }
        s
    }

    fn user(text: String) -> ChatMessageIn {
        ChatMessageIn {
            role: "user".into(),
            content: Some(MessageContent::Text(text)),
            tool_calls: None,
            tool_call_id: None,
            name: None,
        }
    }

    fn assistant(text: String) -> ChatMessageIn {
        ChatMessageIn {
            role: "assistant".into(),
            content: Some(MessageContent::Text(text)),
            tool_calls: None,
            tool_call_id: None,
            name: None,
        }
    }

    struct Turn {
        text: String,
        reasoning: String,
        prompt_tokens: u32,
        cached_tokens: Option<u32>,
        completion_tokens: u32,
        wall_s: f64,
    }

    fn serve(
        rt: &tokio::runtime::Runtime,
        engine: &Arc<WgpuChatEngine>,
        label: &str,
        messages: &[ChatMessageIn],
        max_new: usize,
    ) -> Turn {
        let prompt = engine.render_with_tools(messages, &[]);
        assert!(!prompt.is_empty(), "[{label}] official template rendered nothing");
        let req = ChatGenerateRequest {
            prompt,
            max_new_tokens: max_new,
            stop: Vec::new(),
            seed: Some(7),
            temperature: Some(0.0),
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
        };
        let t0 = Instant::now();
        let mut turn = Turn {
            text: String::new(),
            reasoning: String::new(),
            prompt_tokens: 0,
            cached_tokens: None,
            completion_tokens: 0,
            wall_s: 0.0,
        };
        rt.block_on(async {
            let (tx, mut rx) = tokio::sync::mpsc::channel(64);
            (engine.clone() as Arc<dyn ChatEngine>)
                .generate(req, tx)
                .await
                .unwrap_or_else(|e| panic!("[{label}] generate refused: {e:#}"));
            while let Some(ev) = rx.recv().await {
                match ev {
                    ChatEvent::Started { prompt_tokens } => turn.prompt_tokens = prompt_tokens,
                    ChatEvent::PromptCached { cached_tokens } => {
                        turn.cached_tokens = Some(cached_tokens)
                    }
                    ChatEvent::TextDelta(s) => turn.text.push_str(&s),
                    ChatEvent::ReasoningDelta(s) => turn.reasoning.push_str(&s),
                    ChatEvent::Done {
                        completion_tokens, ..
                    } => turn.completion_tokens = completion_tokens,
                    ChatEvent::Error(e) => panic!("[{label}] engine error: {e}"),
                    ChatEvent::StoppedBy { .. } | ChatEvent::Logprob(_) => {}
                }
            }
        });
        turn.wall_s = t0.elapsed().as_secs_f64();
        eprintln!(
            "[{label}] prompt_tokens={} cached={:?} completion={} wall={:.2}s",
            turn.prompt_tokens, turn.cached_tokens, turn.completion_tokens, turn.wall_s
        );
        turn
    }

    fn stream_bytes(t: &Turn) -> String {
        format!("<think>{}</think>{}", t.reasoning, t.text)
    }

    #[test]
    #[ignore = "boots the ~22.6 GB unsloth/Qwen3.8-27B-NVFP4 checkpoint on wgpu; set \
                NV_QWEN38_KV_REUSE_TEST=1 (NV_QWEN38_DIR, NV_QWEN38_KV_REUSE_REPEATS optional) -- \
                multi-turn chat where turn 2 rewinds to the turn-1 end-of-prefill DeltaNet \
                checkpoint instead of re-prefilling the shared prefix"]
    fn a_second_turn_reuses_the_first_turns_prefill_and_serves_the_cold_completion() {
        require_gate();
        let dir = model_dir();
        let _env = EnvGuard;
        std::env::set_var(PREFIX_REUSE_ENV_DEFAULT_ON_SET_0_FOR_THE_COLD_PREFILL_EVERY_TURN_ARM, "0");
        let _ = tracing_subscriber::fmt()
            .with_env_filter("speaches_plus=info")
            .with_writer(std::io::stderr)
            .try_init();

        let engine = Arc::new(
            WgpuChatEngine::load_with(&dir, MAX_SEQ, None)
                .unwrap_or_else(|e| panic!("wgpu engine did not load: {e:#}")),
        );
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap();

        let doc = passage();
        let q1 = format!("{doc}\n\nIn one sentence, what solved the longitude problem?");
        let turn1_msgs = vec![user(q1.clone())];

        std::env::set_var(PREFIX_REUSE_ENV_DEFAULT_ON_SET_0_FOR_THE_COLD_PREFILL_EVERY_TURN_ARM, "1");
        let turn1 = serve(&rt, &engine, "turn1-prime", &turn1_msgs, 96);
        let turn2_msgs = vec![
            user(q1.clone()),
            assistant(turn1.text.clone()),
            user("And in one sentence, what did the wayfinders read instead of a compass?".into()),
        ];

        std::env::set_var(PREFIX_REUSE_ENV_DEFAULT_ON_SET_0_FOR_THE_COLD_PREFILL_EVERY_TURN_ARM, "0");
        let cold_ttft = serve(&rt, &engine, "turn2-cold-ttft", &turn2_msgs, 1);
        assert_eq!(
            cold_ttft.cached_tokens, None,
            "with {PREFIX_REUSE_ENV_DEFAULT_ON_SET_0_FOR_THE_COLD_PREFILL_EVERY_TURN_ARM}=0 every request must re-prefill"
        );

        std::env::set_var(PREFIX_REUSE_ENV_DEFAULT_ON_SET_0_FOR_THE_COLD_PREFILL_EVERY_TURN_ARM, "1");
        let _prime = serve(&rt, &engine, "turn1-reprime", &turn1_msgs, 96);
        let reused_before = engine.prefix_tokens_reused();
        let warm_ttft = serve(&rt, &engine, "turn2-warm-ttft", &turn2_msgs, 1);
        let gained = engine.prefix_tokens_reused() - reused_before;

        assert!(
            gained > 0,
            "turn 2 re-prefilled its whole prompt: the turn-1 end-of-prefill checkpoint was \
             not reused (prefix_reused=0)"
        );
        assert_eq!(
            warm_ttft.cached_tokens,
            Some(gained as u32),
            "PromptCached must report the tokens actually served out of the resident state"
        );
        assert!(
            (gained as u32) < warm_ttft.prompt_tokens,
            "cached {gained} of a {}-token prompt: at least the divergent turn-2 suffix is \
             always recomputed",
            warm_ttft.prompt_tokens
        );
        assert!(
            warm_ttft.wall_s
                < cold_ttft.wall_s * TTFT_COLLAPSE_FACTOR_HALF_IS_FAR_ABOVE_THE_EXPECTED_DELTA_ONLY_PREFILL,
            "kv-reuse TTFT {:.2}s did not collapse vs cold {:.2}s",
            warm_ttft.wall_s,
            cold_ttft.wall_s
        );

        std::env::set_var(PREFIX_REUSE_ENV_DEFAULT_ON_SET_0_FOR_THE_COLD_PREFILL_EVERY_TURN_ARM, "0");
        let cold_full = serve(&rt, &engine, "turn2-cold-full", &turn2_msgs, 64);

        std::env::set_var(PREFIX_REUSE_ENV_DEFAULT_ON_SET_0_FOR_THE_COLD_PREFILL_EVERY_TURN_ARM, "1");
        let prime2 = serve(&rt, &engine, "turn1-reprime2", &turn1_msgs, 96);
        assert_eq!(
            stream_bytes(&prime2),
            format!("<think>{}</think>{}", turn1.reasoning, turn1.text),
            "this decoder is not reproducible request to request, so a byte-identity claim \
             about kv reuse could not mean anything"
        );
        let reused_before_full = engine.prefix_tokens_reused();
        let warm_full = serve(&rt, &engine, "turn2-warm-full", &turn2_msgs, 64);
        let gained_full = engine.prefix_tokens_reused() - reused_before_full;
        assert!(gained_full > 0, "the full warm turn re-prefilled everything");
        assert_eq!(
            stream_bytes(&warm_full),
            stream_bytes(&cold_full),
            "a turn served from the rewound DeltaNet checkpoint must be byte-identical to the \
             same turn served cold"
        );

        let label = dir
            .file_name()
            .map(|s| s.to_string_lossy().into_owned())
            .unwrap_or_else(|| "unknown".into());
        for (arm, t, cached) in [
            ("cold", &cold_ttft, 0usize),
            ("kv-reuse", &warm_ttft, gained),
        ] {
            println!(
                "QWEN38-WGPU-PREFILL checkpoint={label}-multiturn-{arm} backend=wgpu \
                 prompt_tokens={} prefill_s={:.3} tok_s={:.1} arm={arm} cached_tokens={cached} \
                 measured=request_wall_at_max_new_1 max_seq={MAX_SEQ}",
                t.prompt_tokens,
                t.wall_s,
                t.prompt_tokens as f64 / t.wall_s,
            );
        }
    }
}
