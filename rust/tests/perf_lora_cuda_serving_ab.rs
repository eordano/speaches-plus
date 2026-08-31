#[cfg(feature = "cuda")]
mod perf_instrument_cuda_adapter_serving_vs_base {
    use std::path::{Path, PathBuf};
    use std::sync::Arc;
    use std::time::Instant;

    use candle_core::{Device, Tensor};
    use serde_json::json;
    use speaches_plus::oapi::chat::{ChatEngine, ChatEvent, ChatGenerateRequest};
    use speaches_plus::oapi::chat_engine::{CudaLoraAdapterEngine, NvEngineChat};
    use speaches_plus::oapi::lora::{ADAPTER_CONFIG_FILE, ADAPTER_WEIGHTS_FILE};
    use speaches_plus::oapi::vram_plan::recorded_planned_total_bytes;

    const PERF_GATE_ENV: &str = "NV_LORA_CUDA_PERF_TEST";

    fn base_dir() -> PathBuf {
        PathBuf::from(std::env::var("NV_CHAT_MODEL_DIR").unwrap_or_else(|_| {
            format!(
                "{}/.cache/huggingface/hub/models--nvidia--Gemma-4-31B-IT-NVFP4/snapshots/e5ef03afa233c35cb000323ff098d4291e1dd07c",
                std::env::var("HOME").unwrap_or_default()
            )
        }))
    }

    fn lcg_vals(n: usize, seed: u64, scale: f32) -> Vec<f32> {
        let mut s = seed | 1;
        (0..n)
            .map(|_| {
                s = s
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                (((s >> 11) as f64 / (1u64 << 53) as f64) as f32 - 0.5) * 2.0 * scale
            })
            .collect()
    }

    fn write_all_layer_mlp_rank8_adapter(
        dir: &Path,
        base_dir: &Path,
        seed: u64,
        b_scale: f32,
        layer_cap: usize,
    ) {
        std::fs::create_dir_all(dir).unwrap();
        let raw = std::fs::read_to_string(base_dir.join("config.json")).expect("base config");
        let cfg =
            nv_models::gemma4::Gemma4Config::from_hf_json_str(&raw).expect("gemma4 config parses");
        let (hidden, inter, layers) = (
            cfg.hidden_size,
            cfg.intermediate_size,
            cfg.num_hidden_layers.min(layer_cap),
        );
        let rank = 8usize;
        let peft_cfg = json!({
            "r": rank,
            "lora_alpha": (2 * rank) as f64,
            "target_modules": ["gate_proj", "up_proj", "down_proj"],
            "base_model_name_or_path": null,
        });
        std::fs::write(dir.join(ADAPTER_CONFIG_FILE), peft_cfg.to_string()).unwrap();
        let mut tensors = std::collections::HashMap::new();
        for li in 0..layers {
            for (module, k, out) in [
                ("gate_proj", hidden, inter),
                ("up_proj", hidden, inter),
                ("down_proj", inter, hidden),
            ] {
                let name = format!("model.language_model.layers.{li}.mlp.{module}");
                let s = seed.wrapping_add((li * 7 + module.len()) as u64);
                let a = Tensor::from_vec(
                    lcg_vals(rank * k, s.wrapping_add(1), 0.05),
                    (rank, k),
                    &Device::Cpu,
                )
                .unwrap();
                let b = Tensor::from_vec(
                    lcg_vals(out * rank, s.wrapping_add(2), b_scale),
                    (out, rank),
                    &Device::Cpu,
                )
                .unwrap();
                tensors.insert(format!("{name}.lora_A.weight"), a);
                tensors.insert(format!("{name}.lora_B.weight"), b);
            }
        }
        candle_core::safetensors::save(&tensors, dir.join(ADAPTER_WEIGHTS_FILE)).unwrap();
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

    struct RunOut {
        text: String,
        finish: String,
        completion: u32,
        prompt_tokens: u32,
        elapsed_s: f64,
        hook_m1_applies: u64,
        hook_m1_applies_inside_open_capture: u64,
        hook_fallback_applies: u64,
    }

    async fn run_once(eng: &Arc<dyn ChatEngine>, prompt: &str, max_new: usize) -> RunOut {
        let eng0 = nv_specdecode::lora_spec::serving_hook_engagement_counted_host_side_so_replays_of_a_captured_graph_leave_it_flat();
        let (tx, mut rx) = tokio::sync::mpsc::channel::<ChatEvent>(256);
        let t0 = Instant::now();
        eng.generate(greedy_req(prompt, max_new), tx)
            .await
            .expect("generate");
        let mut text = String::new();
        let mut finish = String::new();
        let mut completion = 0u32;
        let mut prompt_tokens = 0u32;
        let mut engine_error = None;
        while let Some(ev) = rx.recv().await {
            match ev {
                ChatEvent::Started { prompt_tokens: p } => prompt_tokens = p,
                ChatEvent::TextDelta(s) => text.push_str(&s),
                ChatEvent::Done {
                    finish_reason,
                    completion_tokens,
                } => {
                    finish = finish_reason;
                    completion = completion_tokens;
                }
                ChatEvent::Error(e) => engine_error = Some(e),
                _ => {}
            }
        }
        if let Some(e) = engine_error {
            panic!("engine emitted an error after the stream closed: {e}");
        }
        let eng1 = nv_specdecode::lora_spec::serving_hook_engagement_counted_host_side_so_replays_of_a_captured_graph_leave_it_flat();
        RunOut {
            text,
            finish,
            completion,
            prompt_tokens,
            elapsed_s: t0.elapsed().as_secs_f64(),
            hook_m1_applies: eng1.m1_kernel_applies - eng0.m1_kernel_applies,
            hook_m1_applies_inside_open_capture: eng1.m1_applies_inside_an_open_capture
                - eng0.m1_applies_inside_an_open_capture,
            hook_fallback_applies: eng1.fallback_candle_applies - eng0.fallback_candle_applies,
        }
    }

    async fn decode_ms_per_tok_two_point_so_the_shared_prefill_cancels(
        eng: &Arc<dyn ChatEngine>,
        label: &str,
        prompt: &str,
    ) -> (f64, RunOut) {
        let short = run_once(eng, prompt, 16).await;
        let long = run_once(eng, prompt, 272).await;
        eprintln!(
            "[lora_perf_ab] {label} short: prompt_tokens={} completion={} elapsed_s={:.3} finish={}",
            short.prompt_tokens, short.completion, short.elapsed_s, short.finish
        );
        eprintln!(
            "[lora_perf_ab] {label} long: prompt_tokens={} completion={} elapsed_s={:.3} finish={}",
            long.prompt_tokens, long.completion, long.elapsed_s, long.finish
        );
        eprintln!(
            "[lora_perf_ab] {label} ENGAGEMENT short: m1={} m1_in_capture={} fallback={} \
             long: m1={} m1_in_capture={} fallback={} m1_per_extra_decode_token={:.2}",
            short.hook_m1_applies,
            short.hook_m1_applies_inside_open_capture,
            short.hook_fallback_applies,
            long.hook_m1_applies,
            long.hook_m1_applies_inside_open_capture,
            long.hook_fallback_applies,
            (long.hook_m1_applies as f64 - short.hook_m1_applies as f64)
                / long.completion.saturating_sub(short.completion).max(1) as f64
        );
        if nv_models::decode_probe::enabled() {
            nv_models::decode_probe::dump(&format!("cumulative after {label}"));
        }
        assert!(
            long.completion >= 200,
            "{label}: generation stopped at {} tokens; a stop this early makes the two-point \
             decode rate meaningless -- pick a more sustained prompt tail",
            long.completion
        );
        let dn = long.completion.saturating_sub(short.completion).max(1);
        let ms = 1000.0 * (long.elapsed_s - short.elapsed_s) / dn as f64;
        (ms, long)
    }

    fn corpus_prompt_at_depth(
        eng: &Arc<dyn ChatEngine>,
        tokenizer: &tokenizers::Tokenizer,
        corpus: &str,
        target_ids: usize,
    ) -> (String, usize) {
        let template = eng
            .official_template()
            .expect("the NVFP4 snapshot ships chat_template.jinja");
        let tail = "\n\nContinue the passage above in the same style, adding as much detail as possible.";
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
        best.expect("at least one render attempt")
    }

    fn paired_stats(deltas: &[f64]) -> (f64, f64, i32) {
        let n = deltas.len() as f64;
        let mean = deltas.iter().sum::<f64>() / n;
        let var = deltas.iter().map(|d| (d - mean) * (d - mean)).sum::<f64>() / (n - 1.0);
        let t = if var > 0.0 {
            mean / (var / n).sqrt()
        } else {
            f64::INFINITY
        };
        let signs = deltas.iter().filter(|d| **d > 0.0).count() as i32;
        (mean, t, signs)
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    #[ignore = "perf instrument, not a correctness gate: boots the gemma4 dense cuda base \
                (NvEngineChat::try_load, full mode) AND a CudaLoraAdapterEngine carrying a \
                synthetic rank-8 all-layer mlp adapter, then measures interleaved base/adapter \
                greedy decode ms/token pairs at NV_LORA_PERF_DEPTHS (default 256,4096), \
                NV_LORA_PERF_PAIRS per depth (default 3), two-point 16-vs-272 so prefill \
                cancels; also reports boot seconds, vram-plan ledger bytes per engine, and \
                per-run ServingLoraHook engagement counter deltas (host-side applies stay flat \
                across replayed graph tokens, so m1_per_extra_decode_token ~0 proves the hook \
                launches are captured graph nodes and ~hooked-linears proves per-token eager \
                launches); the adapter engine retains the i8 decode head by default so both \
                arms read the same lm_head epilogue, NV_LORA_ADA_I8_HEAD=0 restores the \
                head-dropping lean build for contrast. Run: \
                NVK_LANE=perfa NVK_PKG=speaches-plus NV_LORA_CUDA_PERF_TEST=1 \
                rust/scripts/nvk.sh probe --test perf_lora_cuda_serving_ab -- --ignored \
                --test-threads=1 --nocapture"]
    async fn adapter_engine_decode_ms_per_token_vs_base_interleaved_pairs_at_depth() {
        if std::env::var(PERF_GATE_ENV).ok().as_deref() != Some("1") {
            panic!(
                "{PERF_GATE_ENV}=1 not set. Green here without a run would report a perf \
                 contrast that never happened; this is a SKIP surfaced as a failure by design."
            );
        }
        for v in [
            "NV_DRAFTER",
            "NV_EAGLE3_DRAFT_DIR",
            "NV_DFLASH_DRAFT_DIR",
            "NV_USE_EAGLE3",
            "NV_NO_SPEC",
        ] {
            std::env::remove_var(v);
        }
        if std::env::var_os("NV_KV_MAX_SEQ_LEN_GEMMA4").is_none() {
            std::env::set_var("NV_KV_MAX_SEQ_LEN_GEMMA4", "8192");
            eprintln!(
                "[lora_perf_ab] NV_KV_MAX_SEQ_LEN_GEMMA4=8192 pinned: an unpinned first \
                 engine solves its kv-reserve to fill the vram budget, leaving none for the \
                 adapter engine this contrast boots beside it"
            );
        }
        let depths: Vec<usize> = std::env::var("NV_LORA_PERF_DEPTHS")
            .unwrap_or_else(|_| "256,4096".to_string())
            .split(',')
            .filter_map(|s| s.trim().parse::<usize>().ok())
            .collect();
        let pairs: usize = std::env::var("NV_LORA_PERF_PAIRS")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(3);
        let b_scale: f32 = std::env::var("NV_LORA_PERF_B_SCALE")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(1e-4);
        let layer_cap: usize = std::env::var("NV_LORA_PERF_LAYERS")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(usize::MAX);
        assert!(!depths.is_empty() && pairs >= 3);

        let dir = base_dir();
        assert!(
            dir.join("config.json").is_file(),
            "missing gemma4 checkpoint at {}",
            dir.display()
        );

        let ledger0 = recorded_planned_total_bytes();
        let t0 = Instant::now();
        let base: Arc<dyn ChatEngine> =
            Arc::new(NvEngineChat::try_load(&dir).expect("base engine loads"));
        let base_boot_s = t0.elapsed().as_secs_f64();
        let ledger_base = recorded_planned_total_bytes();

        let adapter_dir = std::env::temp_dir().join(format!(
            "lora-perf-ab-{}-{:x}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos() as u64
        ));
        write_all_layer_mlp_rank8_adapter(&adapter_dir, &dir, 424242, b_scale, layer_cap);
        let t1 = Instant::now();
        let ada: Arc<dyn ChatEngine> = Arc::new(
            CudaLoraAdapterEngine::try_load(&dir, "lora-perf-ab", &adapter_dir)
                .expect("adapter engine loads"),
        );
        let ada_boot_s = t1.elapsed().as_secs_f64();
        let ledger_ada = recorded_planned_total_bytes();

        eprintln!(
            "[lora_perf_ab] boot base_s={base_boot_s:.1} adapter_s={ada_boot_s:.1} \
             vram_plan_base_bytes={} vram_plan_adapter_bytes={} b_scale={b_scale} \
             adapter_keeps_i8_head={}",
            ledger_base - ledger0,
            ledger_ada - ledger_base,
            std::env::var("NV_LORA_ADA_I8_HEAD").as_deref() != Ok("0")
        );
        assert_eq!(
            base.spec_decode_status(),
            ada.spec_decode_status(),
            "the two arms must decode on the same spec-decode setting or the contrast \
             measures the drafter, not the adapter"
        );

        let tokenizer = tokenizers::Tokenizer::from_file(dir.join("tokenizer.json"))
            .expect("tokenizer.json in the serving snapshot");
        let corpus_path = std::env::var("NV_PPL_CORPUS")
            .unwrap_or_else(|_| "/home/user/tmp/wf/quant/corpus/wiki.txt".to_string());
        let corpus = std::fs::read_to_string(&corpus_path)
            .unwrap_or_else(|e| panic!("read corpus {corpus_path}: {e}"));

        let (warm_prompt, warm_ids) = corpus_prompt_at_depth(&base, &tokenizer, &corpus, 256);
        let warm_base = run_once(&base, &warm_prompt, 16).await;
        let warm_ada = run_once(&ada, &warm_prompt, 16).await;
        eprintln!(
            "[lora_perf_ab] warm ids={warm_ids} base: completion={} elapsed_s={:.3}; adapter: \
             completion={} elapsed_s={:.3} adapter_warm_engagement m1={} m1_in_capture={} \
             fallback={}",
            warm_base.completion,
            warm_base.elapsed_s,
            warm_ada.completion,
            warm_ada.elapsed_s,
            warm_ada.hook_m1_applies,
            warm_ada.hook_m1_applies_inside_open_capture,
            warm_ada.hook_fallback_applies
        );

        let mut any_depth_diverged = false;
        let mut summary: Vec<String> = Vec::new();
        for depth in &depths {
            let (prompt, n_ids) = corpus_prompt_at_depth(&base, &tokenizer, &corpus, *depth);
            eprintln!("[lora_perf_ab] depth={depth} prompt_ids={n_ids}");
            let mut base_ms: Vec<f64> = Vec::new();
            let mut ada_ms: Vec<f64> = Vec::new();
            let mut texts: Vec<(String, String)> = Vec::new();
            for pair in 0..pairs {
                let (bms, blong) = decode_ms_per_tok_two_point_so_the_shared_prefill_cancels(
                    &base,
                    &format!("depth{depth}-pair{pair}-base"),
                    &prompt,
                )
                .await;
                let (ams, along) = decode_ms_per_tok_two_point_so_the_shared_prefill_cancels(
                    &ada,
                    &format!("depth{depth}-pair{pair}-adapter"),
                    &prompt,
                )
                .await;
                eprintln!(
                    "[lora_perf_ab] ROW depth={depth} prompt_ids={n_ids} pair={pair} \
                     base_decode_ms_per_tok={bms:.3} adapter_decode_ms_per_tok={ams:.3} \
                     delta_ms={:.3}",
                    ams - bms
                );
                base_ms.push(bms);
                ada_ms.push(ams);
                texts.push((blong.text, along.text));
            }
            let diverged = texts.iter().any(|(b, a)| b != a);
            any_depth_diverged |= diverged;
            let deltas: Vec<f64> = base_ms
                .iter()
                .zip(ada_ms.iter())
                .map(|(b, a)| a - b)
                .collect();
            let (mean_d, t, signs) = paired_stats(&deltas);
            let mean_base = base_ms.iter().sum::<f64>() / base_ms.len() as f64;
            let mean_ada = ada_ms.iter().sum::<f64>() / ada_ms.len() as f64;
            let line = format!(
                "depth={depth} prompt_ids={n_ids} pairs={pairs} base_mean_ms_per_tok={mean_base:.3} \
                 adapter_mean_ms_per_tok={mean_ada:.3} paired_delta_mean_ms={mean_d:.3} t={t:.2} \
                 adapter_slower_sign_count={signs}/{pairs} greedy_texts_diverged={diverged}"
            );
            eprintln!("[lora_perf_ab] SUMMARY {line}");
            summary.push(line);
        }
        eprintln!("[lora_perf_ab] FINAL");
        for line in &summary {
            eprintln!("[lora_perf_ab] FINAL {line}");
        }
        assert!(
            any_depth_diverged,
            "base and adapter greedy long-run texts were identical at every depth: the \
             adapter hooks show no engagement, so every timing row above is void; raise \
             NV_LORA_PERF_B_SCALE and rerun"
        );
        let _ = std::fs::remove_dir_all(&adapter_dir);
    }
}
