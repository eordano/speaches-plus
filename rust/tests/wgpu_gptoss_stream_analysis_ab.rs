#[cfg(not(feature = "wgpu"))]
#[test]
fn wgpu_gptoss_stream_analysis_ab_is_cfg_out_without_the_wgpu_feature() {
    eprintln!(
        "wgpu_gptoss_stream_analysis_ab compiled OUT (no `wgpu` feature). This is a SKIP, not a \
         pass. Re-run with --features wgpu and NV_GPTOSS_SERVE_TEST=1."
    );
}

#[cfg(feature = "wgpu")]
mod gated {
    use std::path::PathBuf;
    use std::sync::Arc;

    use axum::body::Body;
    use axum::http::{Request, StatusCode};
    use axum::routing::post;
    use axum::Router;
    use futures_util::StreamExt;
    use tower::ServiceExt;

    use speaches_plus::oapi::chat::{handle_chat_completions, ChatAppState, ChatEngine};
    use speaches_plus::oapi::chat_engine::ChatRegistry;
    use speaches_plus::oapi::chat_engine_wgpu::WgpuChatEngine;

    const STREAM_ANALYSIS_ENV: &str = "NV_GPTOSS_STREAM_ANALYSIS";

    const HARMONY_MARKUP: [&str; 7] = [
        "<|channel|>",
        "<|message|>",
        "<|start|>",
        "<|end|>",
        "<|return|>",
        "<|call|>",
        "<|constrain|>",
    ];

    fn enabled() -> bool {
        std::env::var("NV_GPTOSS_SERVE_TEST").ok().as_deref() == Some("1")
    }

    fn model_dir() -> Option<PathBuf> {
        if let Ok(d) = std::env::var("NV_GPTOSS_DIR") {
            let p = PathBuf::from(d);
            return p.join("config.json").exists().then_some(p);
        }
        let base = format!(
            "{}/.cache/huggingface/hub/models--openai--gpt-oss-20b/snapshots",
            std::env::var("HOME").unwrap_or_default()
        );
        std::fs::read_dir(base)
            .ok()?
            .filter_map(|e| e.ok())
            .map(|e| e.path())
            .find(|p| p.join("config.json").is_file() && p.join("tokenizer.json").is_file())
    }

    fn app(engine: Arc<dyn ChatEngine>) -> Router {
        Router::new()
            .route("/v1/chat/completions", post(handle_chat_completions))
            .with_state(ChatAppState {
                registry: ChatRegistry::single(engine),
            })
    }

    struct ArmOut {
        reasoning: String,
        content: String,
        reasoning_frames: usize,
        finish_reason: String,
        completion_tokens: u64,
        ttft_reasoning_s: f64,
        ttft_any_s: f64,
        wall_s: f64,
    }

    fn run_stream(engine: Arc<dyn ChatEngine>, label: &str, prompt: &str, max_tokens: u32) -> ArmOut {
        let router = app(engine.clone());
        let model_id = engine.model_id().to_string();
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap();
        rt.block_on(async move {
            let body = format!(
                r#"{{"model":"{model_id}","max_tokens":{max_tokens},"temperature":0,
                     "stream":true,"stream_options":{{"include_usage":true}},
                     "messages":[{{"role":"user","content":{}}}]}}"#,
                serde_json::to_string(prompt).unwrap()
            );
            let req = Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header("content-type", "application/json")
                .body(Body::from(body))
                .unwrap();
            let t0 = std::time::Instant::now();
            let resp = router.oneshot(req).await.unwrap();
            assert_eq!(
                resp.status(),
                StatusCode::OK,
                "[{label}] the streaming request was rejected before any token"
            );
            let mut data = resp.into_body().into_data_stream();
            let mut pending = String::new();
            let mut out = ArmOut {
                reasoning: String::new(),
                content: String::new(),
                reasoning_frames: 0,
                finish_reason: String::new(),
                completion_tokens: 0,
                ttft_reasoning_s: 0.0,
                ttft_any_s: 0.0,
                wall_s: 0.0,
            };
            let mut ttft_reasoning: Option<f64> = None;
            let mut ttft_any: Option<f64> = None;
            while let Some(chunk) = data.next().await {
                let chunk = chunk
                    .unwrap_or_else(|e| panic!("[{label}] sse transport broke mid-stream: {e}"));
                pending.push_str(&String::from_utf8_lossy(&chunk));
                while let Some(at) = pending.find("\n\n") {
                    let frame = pending[..at].to_string();
                    pending = pending[at + 2..].to_string();
                    let Some(payload) = frame.strip_prefix("data: ") else {
                        continue;
                    };
                    if payload.trim() == "[DONE]" {
                        continue;
                    }
                    let v: serde_json::Value = serde_json::from_str(payload)
                        .unwrap_or_else(|e| panic!("[{label}] bad sse frame {payload:?}: {e}"));
                    if let Some(msg) = v["error"]["message"].as_str() {
                        panic!("[{label}] engine error mid-stream: {msg}");
                    }
                    if let Some(u) = v.get("usage").filter(|u| !u.is_null()) {
                        out.completion_tokens = u["completion_tokens"].as_u64().unwrap_or(0);
                    }
                    let delta = &v["choices"][0]["delta"];
                    let c = delta["content"].as_str().unwrap_or_default();
                    let r = delta["reasoning_content"].as_str().unwrap_or_default();
                    if !r.is_empty() {
                        out.reasoning_frames += 1;
                        ttft_reasoning.get_or_insert_with(|| t0.elapsed().as_secs_f64());
                    }
                    if !(c.is_empty() && r.is_empty()) {
                        ttft_any.get_or_insert_with(|| t0.elapsed().as_secs_f64());
                    }
                    out.content.push_str(c);
                    out.reasoning.push_str(r);
                    if let Some(f) = v["choices"][0]["finish_reason"].as_str() {
                        out.finish_reason = f.to_string();
                    }
                }
            }
            out.wall_s = t0.elapsed().as_secs_f64();
            out.ttft_reasoning_s = ttft_reasoning.unwrap_or(out.wall_s);
            out.ttft_any_s = ttft_any.unwrap_or(out.wall_s);
            eprintln!(
                "[gptoss-stream-ab] [{label}] reasoning_bytes={} reasoning_frames={} \
                 content_bytes={} finish={:?} completion_tokens={} ttft_reasoning={:.0}ms \
                 ttft_any={:.0}ms wall={:.2}s",
                out.reasoning.len(),
                out.reasoning_frames,
                out.content.len(),
                out.finish_reason,
                out.completion_tokens,
                out.ttft_reasoning_s * 1000.0,
                out.ttft_any_s * 1000.0,
                out.wall_s
            );
            out
        })
    }

    #[test]
    #[ignore = "loads 13 GB of gpt-oss-20b weights; set NV_GPTOSS_SERVE_TEST=1"]
    fn streamed_analysis_reassembles_the_buffered_flush_and_moves_ttft_to_first_tokens() {
        if !enabled() {
            panic!(
                "this test is #[ignore]d, so it was asked for BY NAME, but \
                 NV_GPTOSS_SERVE_TEST=1 is not set. This is a SKIP, not a pass."
            );
        }
        let Some(dir) = model_dir() else {
            panic!(
                "openai/gpt-oss-20b is not cached (set NV_GPTOSS_DIR). This is a SKIP, not a pass."
            )
        };
        std::env::remove_var(STREAM_ANALYSIS_ENV);
        let engine = Arc::new(
            WgpuChatEngine::load_with(&dir, 2048, None).expect("load gpt-oss wgpu engine"),
        ) as Arc<dyn ChatEngine>;

        let _ = run_stream(engine.clone(), "warmup", "Say hi.", 8);

        let truncated_prompt = "Count to five, digits only.";
        let buffered = run_stream(engine.clone(), "buffered/truncated", truncated_prompt, 64);
        std::env::set_var(STREAM_ANALYSIS_ENV, "1");
        let streamed = run_stream(engine.clone(), "streamed/truncated", truncated_prompt, 64);
        std::env::remove_var(STREAM_ANALYSIS_ENV);

        assert_eq!(
            buffered.finish_reason, "length",
            "the premise is a completion truncated inside the analysis channel; \
             lower max_tokens instead of raising anything"
        );
        assert!(
            !buffered.reasoning.trim().is_empty(),
            "the buffered arm returned no reasoning, so there is nothing to compare"
        );
        assert_eq!(
            streamed.reasoning, buffered.reasoning,
            "concatenated reasoning bytes must be identical across arms"
        );
        assert_eq!(
            streamed.content, buffered.content,
            "content bytes must be identical across arms"
        );
        assert_eq!(streamed.finish_reason, buffered.finish_reason);
        assert_eq!(streamed.completion_tokens, buffered.completion_tokens);
        assert_eq!(
            buffered.reasoning_frames, 1,
            "the buffered arm flushes analysis as one terminal delta; if this grew, the \
             default arm changed and the opt-in premise is stale"
        );
        assert!(
            streamed.reasoning_frames >= 2,
            "the opt-in arm must stream analysis incrementally, got {} frame(s)",
            streamed.reasoning_frames
        );
        for m in HARMONY_MARKUP {
            assert!(
                !streamed.reasoning.contains(m),
                "harmony markup {m} leaked into streamed reasoning"
            );
        }
        assert!(
            streamed.ttft_reasoning_s < buffered.ttft_reasoning_s * 0.5,
            "streaming must move TTFT(first reasoning delta) from ~wall ({:.0}ms) to \
             ~first-tokens, got {:.0}ms",
            buffered.ttft_reasoning_s * 1000.0,
            streamed.ttft_reasoning_s * 1000.0
        );
        eprintln!(
            "[gptoss-stream-ab] ROW build_name=\"gptoss stream-analysis\" \
             ttft_reasoning_buffered_ms={:.0} ttft_reasoning_streamed_ms={:.0} \
             wall_buffered_s={:.2} wall_streamed_s={:.2} reasoning_bytes={} frames={}->{}",
            buffered.ttft_reasoning_s * 1000.0,
            streamed.ttft_reasoning_s * 1000.0,
            buffered.wall_s,
            streamed.wall_s,
            buffered.reasoning.len(),
            buffered.reasoning_frames,
            streamed.reasoning_frames
        );

        let answered_prompt = "What is 2+2? Answer with just the number.";
        std::env::remove_var(STREAM_ANALYSIS_ENV);
        let buffered_ans = run_stream(engine.clone(), "buffered/answered", answered_prompt, 512);
        std::env::set_var(STREAM_ANALYSIS_ENV, "1");
        let streamed_ans = run_stream(engine.clone(), "streamed/answered", answered_prompt, 512);
        std::env::remove_var(STREAM_ANALYSIS_ENV);

        assert_eq!(
            buffered_ans.finish_reason, "stop",
            "the premise is a completion that reached its final channel"
        );
        assert_eq!(
            streamed_ans.content, buffered_ans.content,
            "final-channel content bytes must be identical across arms"
        );
        assert!(
            !streamed_ans.content.trim().is_empty(),
            "an answered completion with empty content proves nothing"
        );
        assert_eq!(
            buffered_ans.reasoning, "",
            "the default arm drops analysis once a final answer exists; if this grew, the \
             default changed and the opt-in premise is stale"
        );
        assert!(
            !streamed_ans.reasoning.trim().is_empty(),
            "the opt-in arm surfaces the analysis the default drops"
        );
        for m in HARMONY_MARKUP {
            assert!(
                !streamed_ans.reasoning.contains(m),
                "harmony markup {m} leaked into streamed reasoning (answered case)"
            );
        }
    }
}
