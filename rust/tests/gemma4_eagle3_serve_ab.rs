#[cfg(feature = "cuda")]
mod serving {
    use serde_json::json;
    use speaches_plus::oapi::chat::{ChatEngine, ChatEvent, ChatGenerateRequest};
    use speaches_plus::oapi::chat_engine::NvEngineChat;
    use std::path::PathBuf;
    use std::time::Instant;

    const WHY_THIS_TEST_EXISTS: &str = "gemma4_graphed_verify_captures_and_replays measures the \
         graphed VERIFIER replay alone over a hard-coded 4-token chain and divides by an ASSUMED \
         3.25 tok/round; it runs no drafter at all, so its ms-per-accepted-token is not a serving \
         number for any spec arm. This test drives the real serving engine (run_sampling_gemma4 -> \
         run_sampling_gemma4_spec) with the RedHatAI eagle3 speculator loaded, A/B against \
         NV_NO_SPEC=1 normal decode on the same engine, sustained 64-token greedy generations \
         through the snapshot's own chat template.";

    fn g4_snapshot_dir() -> PathBuf {
        PathBuf::from(std::env::var("NV_G4_SNAPSHOT").unwrap_or_else(|_| {
            format!(
                "{}/.cache/huggingface/hub/models--nvidia--Gemma-4-31B-IT-NVFP4/snapshots/e5ef03afa233c35cb000323ff098d4291e1dd07c",
                std::env::var("HOME").unwrap_or_default()
            )
        }))
    }

    fn eagle3_speculator_dir() -> PathBuf {
        PathBuf::from(std::env::var("NV_EAGLE3_DRAFT_DIR").unwrap_or_else(|_| {
            format!(
                "{}/.cache/huggingface/hub/models--RedHatAI--gemma-4-31B-it-speculator.eagle3/snapshots/28a1c8b4bb64dbaee883ba35341841138bdf1fe3",
                std::env::var("HOME").unwrap_or_default()
            )
        }))
    }

    fn greedy_req(prompt: &str, max_new: usize) -> ChatGenerateRequest {
        ChatGenerateRequest {
            prompt: prompt.to_string(),
            max_new_tokens: max_new,
            stop: Vec::new(),
            seed: Some(0),
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
        }
    }

    #[derive(Debug, Clone)]
    struct RunOut {
        text: String,
        finish: String,
        completion: u32,
        prompt_tokens: u32,
        elapsed_s: f64,
    }

    async fn run_capture(
        engine: &NvEngineChat,
        prompt: &str,
        max_new: usize,
    ) -> Result<RunOut, String> {
        let (tx, mut rx) = tokio::sync::mpsc::channel::<ChatEvent>(256);
        let t0 = Instant::now();
        engine
            .generate(greedy_req(prompt, max_new), tx)
            .await
            .map_err(|e| format!("{e:#}"))?;
        let mut text = String::new();
        let mut finish = String::new();
        let mut completion = 0u32;
        let mut prompt_tokens = 0u32;
        while let Some(ev) = rx.recv().await {
            match ev {
                ChatEvent::Started { prompt_tokens: p } => prompt_tokens = p,
                ChatEvent::PromptCached { .. }
                | ChatEvent::StoppedBy { .. }
                | ChatEvent::ReasoningDelta(_)
                | ChatEvent::Logprob(_) => {}
                ChatEvent::TextDelta(p) => text.push_str(&p),
                ChatEvent::Done {
                    finish_reason,
                    completion_tokens,
                } => {
                    finish = finish_reason;
                    completion = completion_tokens;
                    break;
                }
                ChatEvent::Error(e) => return Err(e),
            }
        }
        Ok(RunOut {
            text,
            finish,
            completion,
            prompt_tokens,
            elapsed_s: t0.elapsed().as_secs_f64(),
        })
    }

    async fn run_once(engine: &NvEngineChat, prompt: &str, max_new: usize) -> RunOut {
        match run_capture(engine, prompt, max_new).await {
            Ok(r) => r,
            Err(e) => panic!("engine error: {e}"),
        }
    }

    fn report(arm: &str, name: &str, run_idx: usize, r: &RunOut) {
        let per_tok = if r.completion > 0 {
            1000.0 * r.elapsed_s / r.completion as f64
        } else {
            f64::NAN
        };
        eprintln!(
            "[g4_eagle3_ab] {name} {arm} run{run_idx}: prompt_tokens={} {} tokens in {:.1} ms wall ({:.2} ms/token wall incl prefill, {:.1} tok/s) finish={}",
            r.prompt_tokens,
            r.completion,
            1000.0 * r.elapsed_s,
            per_tok,
            1000.0 / per_tok,
            r.finish
        );
        let preview: String = r.text.chars().take(120).collect();
        eprintln!("[g4_eagle3_ab] {name} {arm} run{run_idx} text: {preview:?}");
    }

    fn sustained_cases(engine: &NvEngineChat) -> Vec<(String, String, usize)> {
        let template = engine
            .official_template()
            .expect("the NVFP4 snapshot ships chat_template.jinja; refusing a hand-built wrapper");
        let bodies = [
            (
                "chat-history",
                "Summarize the history of computing, era by era, in detail.",
            ),
            (
                "chat-sky",
                "Explain step by step why the sky is blue and why sunsets are red.",
            ),
            (
                "code-fib",
                "Write a Python function computing the nth Fibonacci number with a docstring and example usage.",
            ),
        ];
        bodies
            .iter()
            .map(|(label, body)| {
                let msgs = json!([{"role": "user", "content": body}]);
                let rendered = template
                    .render(&msgs, None, true)
                    .expect("official template render");
                (label.to_string(), rendered, 64usize)
            })
            .collect()
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    #[ignore]
    async fn gemma4_eagle3_spec_vs_normal_sustained() {
        if std::env::var("NV_G4_EAGLE3_TEST").is_err() {
            panic!(
                "set NV_G4_EAGLE3_TEST=1 to run this GPU test (it must never silently skip); \
                 optionally NV_G4_SNAPSHOT / NV_EAGLE3_DRAFT_DIR. {WHY_THIS_TEST_EXISTS}"
            );
        }
        let g4 = g4_snapshot_dir();
        let spec = eagle3_speculator_dir();
        assert!(g4.join("config.json").is_file(), "missing {}", g4.display());
        assert!(
            spec.join("model.safetensors").is_file(),
            "missing eagle3 speculator at {}",
            spec.display()
        );
        std::env::set_var("NV_EAGLE3_DRAFT_DIR", &spec);
        std::env::set_var("NV_EAGLE3_REQUIRED", "1");
        std::env::set_var("NV_PROF_CHAT", "1");
        std::env::remove_var("NV_NO_SPEC");
        std::env::remove_var("NV_USE_EAGLE3");
        std::env::remove_var("NV_DRAFTER");

        let t0 = Instant::now();
        let engine = NvEngineChat::try_load(&g4).expect("load gemma4 serving engine");
        eprintln!(
            "[g4_eagle3_ab] engine loaded in {:.1}s from {} with drafter {}",
            t0.elapsed().as_secs_f64(),
            g4.display(),
            spec.display()
        );
        assert_eq!(
            engine.spec_decode_status(),
            Some("on"),
            "NV_EAGLE3_REQUIRED=1 was set, so a loaded engine must report spec on"
        );

        let cases = sustained_cases(&engine);
        assert!(!cases.is_empty());
        for (name, prompt, max_new) in cases {
            std::env::remove_var("NV_NO_SPEC");
            let warm = run_once(&engine, &prompt, max_new).await;
            report("eagle3-warmup", &name, 0, &warm);
            for i in 1..=2usize {
                let r = run_once(&engine, &prompt, max_new).await;
                report("eagle3", &name, i, &r);
                assert!(r.completion >= 32, "case {name}: not a sustained generation");
            }

            std::env::set_var("NV_NO_SPEC", "1");
            let warm = run_once(&engine, &prompt, max_new).await;
            report("normal-warmup", &name, 0, &warm);
            for i in 1..=2usize {
                let r = run_once(&engine, &prompt, max_new).await;
                report("normal", &name, i, &r);
                assert!(r.completion >= 32, "case {name}: not a sustained generation");
            }
        }
        std::env::remove_var("NV_NO_SPEC");
        eprintln!("[g4_eagle3_ab] done. {WHY_THIS_TEST_EXISTS}");
    }

    fn dflash_speculator_dir() -> PathBuf {
        PathBuf::from(std::env::var("NV_G4_DFLASH_DIR").unwrap_or_else(|_| {
            format!(
                "{}/.cache/huggingface/hub/models--RedHatAI--gemma-4-31B-it-speculator.dflash/snapshots/1bccc881bac5d6d2a6874317007a156fac5ca98b",
                std::env::var("HOME").unwrap_or_default()
            )
        }))
    }

    fn corpus_text() -> String {
        let path = std::env::var("NV_PPL_CORPUS")
            .unwrap_or_else(|_| "/tmp/nv-corpus/wiki.txt".to_string());
        std::fs::read_to_string(&path)
            .unwrap_or_else(|e| panic!("read corpus {path}: {e}"))
    }

    fn corpus_prompt_at_depth(
        engine: &NvEngineChat,
        tokenizer: &tokenizers::Tokenizer,
        corpus: &str,
        target_ids: usize,
    ) -> (String, usize) {
        let template = engine
            .official_template()
            .expect("the NVFP4 snapshot ships chat_template.jinja");
        let tail = "\n\nSummarize the passage above in detail.";
        let mut take_chars = target_ids.saturating_mul(38) / 10;
        let mut best: Option<(String, usize)> = None;
        for _ in 0..8 {
            let body: String = corpus.chars().take(take_chars).collect();
            let msgs = json!([{"role": "user", "content": format!("{body}{tail}")}]);
            let rendered = template
                .render(&msgs, None, true)
                .expect("official template render at depth");
            let n_ids = tokenizer
                .encode(rendered.as_str(), false)
                .expect("encode depth prompt")
                .get_ids()
                .len();
            best = Some((rendered, n_ids));
            let err = n_ids as f64 / target_ids as f64;
            if (0.98..=1.02).contains(&err) {
                break;
            }
            take_chars = ((take_chars as f64) / err).round() as usize;
            assert!(
                take_chars / 4 < corpus.chars().count(),
                "corpus too small for {target_ids}-token depth arm"
            );
        }
        let (rendered, n_ids) = best.expect("at least one render attempt");
        (rendered, n_ids)
    }

    async fn depth_arm(
        engine: &NvEngineChat,
        label: &str,
        arm: &str,
        prompt: &str,
        max_new: usize,
        no_spec: bool,
    ) {
        if no_spec {
            std::env::set_var("NV_NO_SPEC", "1");
        } else {
            std::env::remove_var("NV_NO_SPEC");
        }
        match run_capture(engine, prompt, max_new).await {
            Ok(warm) => report(&format!("{arm}-warmup"), label, 0, &warm),
            Err(e) => {
                eprintln!("[g4_eagle3_ab] {label} {arm} REFUSED: {e}");
                return;
            }
        }
        for i in 1..=2usize {
            match run_capture(engine, prompt, max_new).await {
                Ok(r) => {
                    report(arm, label, i, &r);
                    assert!(r.completion >= 32, "case {label} {arm}: not sustained");
                }
                Err(e) => {
                    eprintln!("[g4_eagle3_ab] {label} {arm} run{i} REFUSED: {e}");
                    return;
                }
            }
        }
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    #[ignore]
    async fn gemma4_spec_depth_ladder() {
        if std::env::var("NV_G4_EAGLE3_TEST").is_err() {
            panic!(
                "set NV_G4_EAGLE3_TEST=1 to run this GPU test (it must never silently skip). \
                 NV_G4_ARM=eagle3|dflash picks which drafter the engine loads (dflash cannot \
                 share an engine with eagle3 for a forced A/B: with both loaded the per-request \
                 arm is routed, EMA for eagle3/dflash kinds and prompt-length-thresholded for \
                 auto/route, not selected). NV_G4_DEPTHS is a comma list where 0 means \
                 the three short prompts and any other value is a wiki-corpus prompt truncated \
                 to that many tokens."
            );
        }
        let arm_kind = std::env::var("NV_G4_ARM").unwrap_or_else(|_| "eagle3".to_string());
        let depths: Vec<usize> = std::env::var("NV_G4_DEPTHS")
            .unwrap_or_else(|_| "0,8192,32768,65536".to_string())
            .split(',')
            .filter_map(|s| s.trim().parse::<usize>().ok())
            .collect();
        assert!(!depths.is_empty(), "NV_G4_DEPTHS parsed to nothing");

        let g4 = g4_snapshot_dir();
        std::env::set_var("NV_PROF_CHAT", "1");
        std::env::remove_var("NV_NO_SPEC");
        std::env::remove_var("NV_USE_EAGLE3");
        match arm_kind.as_str() {
            "dflash" => {
                let dir = dflash_speculator_dir();
                assert!(
                    dir.join("model.safetensors").is_file(),
                    "dflash speculator weights missing at {}",
                    dir.display()
                );
                std::env::set_var("NV_DRAFTER", "dflash");
                std::env::set_var("NV_DFLASH_DRAFT_DIR", &dir);
                std::env::set_var("NV_DFLASH_REQUIRED", "1");
                std::env::remove_var("NV_EAGLE3_REQUIRED");
            }
            "eagle3" => {
                let dir = eagle3_speculator_dir();
                std::env::set_var("NV_DRAFTER", "eagle3");
                std::env::set_var("NV_EAGLE3_DRAFT_DIR", &dir);
                std::env::set_var("NV_EAGLE3_REQUIRED", "1");
            }
            other => panic!("NV_G4_ARM must be eagle3 or dflash, got {other}"),
        }

        let t0 = Instant::now();
        let engine = NvEngineChat::try_load(&g4).expect("load gemma4 serving engine");
        eprintln!(
            "[g4_eagle3_ab] ladder arm={arm_kind} engine loaded in {:.1}s",
            t0.elapsed().as_secs_f64()
        );
        assert_eq!(
            engine.spec_decode_status(),
            Some("on"),
            "REQUIRED flag was set for the {arm_kind} drafter, so a loaded engine must be spec-on"
        );
        let tokenizer = tokenizers::Tokenizer::from_file(g4.join("tokenizer.json"))
            .expect("tokenizer.json in the serving snapshot");
        let corpus = corpus_text();

        for depth in depths {
            if depth == 0 {
                for (name, prompt, max_new) in sustained_cases(&engine) {
                    let label = format!("short/{name}");
                    depth_arm(&engine, &label, &arm_kind, &prompt, max_new, false).await;
                    if arm_kind == "eagle3" {
                        depth_arm(&engine, &label, "normal", &prompt, max_new, true).await;
                    }
                }
                continue;
            }
            let (prompt, n_ids) = corpus_prompt_at_depth(&engine, &tokenizer, &corpus, depth);
            let label = format!("wiki-{depth}");
            eprintln!("[g4_eagle3_ab] {label}: built corpus prompt of {n_ids} ids (target {depth})");
            depth_arm(&engine, &label, &arm_kind, &prompt, 64, false).await;
            if arm_kind == "eagle3" {
                depth_arm(&engine, &label, "normal", &prompt, 64, true).await;
            }
        }
        std::env::remove_var("NV_NO_SPEC");
        eprintln!("[g4_eagle3_ab] ladder arm={arm_kind} done");
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    #[ignore]
    async fn gemma4_auto_drafter_routes_dflash_short_and_eagle3_past_switch_tokens() {
        if std::env::var("NV_G4_EAGLE3_TEST").is_err() {
            panic!(
                "set NV_G4_EAGLE3_TEST=1 to run this GPU test (it must never silently skip). \
                 It loads BOTH drafters with NV_DRAFTER=auto and asserts the per-request routing \
                 decision: a short chat prompt must route dflash (dflash wins every measured arm \
                 through 8k) and a 32k corpus prompt must route eagle3 (dflash acceptance \
                 collapses to ~0.1 by 32k), with the default \
                 NV_DRAFTER_AUTO_SWITCH_TOKENS boundary at 16384."
            );
        }
        let g4 = g4_snapshot_dir();
        let eagle3_dir = eagle3_speculator_dir();
        let dflash_dir = dflash_speculator_dir();
        assert!(
            eagle3_dir.join("model.safetensors").is_file(),
            "eagle3 speculator weights missing at {}",
            eagle3_dir.display()
        );
        assert!(
            dflash_dir.join("model.safetensors").is_file(),
            "dflash speculator weights missing at {}",
            dflash_dir.display()
        );
        std::env::set_var("NV_PROF_CHAT", "1");
        std::env::set_var("NV_DRAFTER", "auto");
        std::env::set_var("NV_EAGLE3_DRAFT_DIR", &eagle3_dir);
        std::env::set_var("NV_EAGLE3_REQUIRED", "1");
        std::env::set_var("NV_DFLASH_DRAFT_DIR", &dflash_dir);
        std::env::set_var("NV_DFLASH_REQUIRED", "1");
        std::env::remove_var("NV_NO_SPEC");
        std::env::remove_var("NV_USE_EAGLE3");
        std::env::remove_var("NV_DRAFTER_AUTO_SWITCH_TOKENS");

        let t0 = Instant::now();
        let engine = NvEngineChat::try_load(&g4).expect("load gemma4 serving engine");
        eprintln!(
            "[g4_eagle3_ab] auto-routing engine loaded in {:.1}s with both drafters",
            t0.elapsed().as_secs_f64()
        );
        assert_eq!(
            engine.spec_decode_status(),
            Some("on"),
            "both REQUIRED flags were set, so a loaded engine must be spec-on"
        );

        let (name, short_prompt, _) = sustained_cases(&engine)
            .into_iter()
            .next()
            .expect("at least one short case");
        let short = run_once(&engine, &short_prompt, 32).await;
        report("auto-short", &name, 1, &short);
        assert!(
            (short.prompt_tokens as usize) < 16384,
            "short case must sit below the auto switch boundary, got {} prompt tokens",
            short.prompt_tokens
        );
        assert_eq!(
            NvEngineChat::last_routed_drafter_arm(),
            Some("dflash"),
            "NV_DRAFTER=auto must route a {}-token prompt (below \
             NV_DRAFTER_AUTO_SWITCH_TOKENS=16384) to dflash",
            short.prompt_tokens
        );

        let tokenizer = tokenizers::Tokenizer::from_file(g4.join("tokenizer.json"))
            .expect("tokenizer.json in the serving snapshot");
        let corpus = corpus_text();
        let (deep_prompt, n_ids) = corpus_prompt_at_depth(&engine, &tokenizer, &corpus, 32768);
        eprintln!("[g4_eagle3_ab] auto-deep: built corpus prompt of {n_ids} ids (target 32768)");
        assert!(
            n_ids >= 16384,
            "deep prompt must sit above the auto switch boundary, got {n_ids} ids"
        );
        let deep = run_once(&engine, &deep_prompt, 32).await;
        report("auto-deep", "wiki-32768", 1, &deep);
        assert_eq!(
            NvEngineChat::last_routed_drafter_arm(),
            Some("eagle3"),
            "NV_DRAFTER=auto must route a {}-token prompt (at or above \
             NV_DRAFTER_AUTO_SWITCH_TOKENS=16384) to eagle3",
            deep.prompt_tokens
        );
        eprintln!(
            "[g4_eagle3_ab] auto routing flipped dflash->eagle3 across the 16384-token boundary \
             (short={} deep={} prompt tokens)",
            short.prompt_tokens, deep.prompt_tokens
        );
    }
}
