#[cfg(not(feature = "cuda"))]
#[test]
fn qwen36_mtp_serving_e2e_is_cfg_out_without_cuda() {
    eprintln!("qwen36_mtp_serving_e2e compiled OUT (no `cuda` feature). SKIP, not a pass.");
}

#[cfg(feature = "cuda")]
mod gated {
    use std::path::PathBuf;
    use std::sync::{Arc, Mutex};

    use axum::body::{to_bytes, Body};
    use axum::http::{Request, StatusCode};
    use axum::routing::post;
    use axum::Router;
    use tower::ServiceExt;

    use speaches_plus::oapi::chat::{handle_chat_completions, ChatAppState, ChatEngine};
    use speaches_plus::oapi::chat_engine::{ChatRegistry, NvEngineChat};

    fn snapshot_dir_with_mtp_shard() -> Option<PathBuf> {
        if let Ok(d) = std::env::var("NV_QWEN36_DIR") {
            return Some(PathBuf::from(d));
        }
        let home = std::env::var_os("HOME")?;
        let snaps = PathBuf::from(home)
            .join(".cache/huggingface/hub")
            .join("models--RedHatAI--Qwen3.6-35B-A3B-NVFP4")
            .join("snapshots");
        let mut cand: Vec<PathBuf> = std::fs::read_dir(&snaps)
            .ok()?
            .flatten()
            .map(|e| e.path())
            .filter(|p| {
                p.join("config.json").exists() && p.join("model_mtp.safetensors").exists()
            })
            .collect();
        cand.sort();
        cand.pop()
    }

    #[derive(Clone, Default)]
    struct LogCapture(Arc<Mutex<Vec<u8>>>);

    impl std::io::Write for LogCapture {
        fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
            self.0.lock().unwrap().extend_from_slice(buf);
            Ok(buf.len())
        }
        fn flush(&mut self) -> std::io::Result<()> {
            Ok(())
        }
    }

    impl<'a> tracing_subscriber::fmt::MakeWriter<'a> for LogCapture {
        type Writer = LogCapture;
        fn make_writer(&'a self) -> Self::Writer {
            self.clone()
        }
    }

    async fn post_json(app: &Router, body: String) -> (StatusCode, String) {
        let req = Request::builder()
            .method("POST")
            .uri("/v1/chat/completions")
            .header("content-type", "application/json")
            .body(Body::from(body))
            .unwrap();
        let resp = app.clone().oneshot(req).await.unwrap();
        let status = resp.status();
        let bytes = to_bytes(resp.into_body(), 1 << 22).await.unwrap();
        (status, String::from_utf8_lossy(&bytes).into_owned())
    }

    fn content_of(resp: &str) -> String {
        let v: serde_json::Value = serde_json::from_str(resp).unwrap();
        let msg = &v["choices"][0]["message"];
        let mut content = msg["content"].as_str().unwrap_or_default().to_string();
        if content.is_empty() {
            content = msg["reasoning_content"]
                .as_str()
                .unwrap_or_default()
                .to_string();
        }
        content
    }

    #[test]
    #[ignore]
    fn nv_drafter_mtp_boots_loudly_or_serves_speculatively_and_greedy_chat_is_deterministic() {
        if std::env::var("NV_QWEN36_MTP_E2E").as_deref() != Ok("1") {
            panic!(
                "PRECONDITION NOT MET, THIS TEST EXECUTED NOTHING: set NV_QWEN36_MTP_E2E=1 \
                 (loads the 35B onto the GPU, twice)"
            );
        }
        let Some(dir) = snapshot_dir_with_mtp_shard() else {
            panic!(
                "no Qwen3.6-35B snapshot carrying model_mtp.safetensors under the HF hub \
                 cache; set NV_QWEN36_DIR"
            );
        };

        std::env::set_var("NV_DRAFTER", "mtp");

        let cap = LogCapture::default();
        tracing_subscriber::fmt()
            .with_max_level(tracing::Level::INFO)
            .with_ansi(false)
            .with_writer(cap.clone())
            .init();

        let t0 = std::time::Instant::now();
        std::env::set_var("NV_MTP_DRAFT_DIR", "/nonexistent-mtp-draft-dir");
        let err = NvEngineChat::try_load(&dir)
            .err()
            .expect("NV_DRAFTER=mtp with a missing MTP shard must be a LOUD boot failure");
        std::env::remove_var("NV_MTP_DRAFT_DIR");
        assert!(
            format!("{err:#}").contains("NV_DRAFTER=mtp"),
            "the boot error must name the drafter request so the operator knows what \
             refused: {err:#}"
        );
        assert_eq!(
            speaches_plus::oapi::vram_plan::recorded_planned_total_bytes(),
            0,
            "the failed boot's enforced plan must be unrecorded, or this same-process retry \
             would be refused for phantom prior-engine bytes"
        );
        eprintln!(
            "loud-failure phase (in-process, {:.1}s): boot refused naming NV_DRAFTER=mtp and \
             left the boot-plan ledger empty",
            t0.elapsed().as_secs_f64()
        );

        let t1 = std::time::Instant::now();
        let engine = Arc::new(
            NvEngineChat::try_load(&dir)
                .expect("NvEngineChat must load the 35B with its shipped MoE MTP drafter"),
        ) as Arc<dyn ChatEngine>;
        eprintln!("engine ready in {:.1}s", t1.elapsed().as_secs_f64());
        let model_id = engine.model_id().to_string();

        let app = Router::new()
            .route("/v1/chat/completions", post(handle_chat_completions))
            .with_state(ChatAppState {
                registry: ChatRegistry::single(engine),
            });

        let body = serde_json::json!({
            "model": model_id,
            "messages": [{"role": "user", "content": "The capital of France is"}],
            "temperature": 0.0,
            "max_tokens": 256,
        })
        .to_string();

        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap();
        let contents: Vec<String> = rt.block_on(async {
            let mut out = Vec::new();
            for run in 0..2 {
                let t = std::time::Instant::now();
                let (status, resp) = post_json(&app, body.clone()).await;
                assert_eq!(status, StatusCode::OK, "run {run}: {resp}");
                let content = content_of(&resp);
                eprintln!(
                    "run {run} ({:.2}s): {content:?}",
                    t.elapsed().as_secs_f64()
                );
                out.push(content);
            }
            out
        });

        for (i, c) in contents.iter().enumerate() {
            assert!(
                c.contains("Paris"),
                "run {i} does not contain Paris -- the speculative serving path is WRONG, \
                 not merely nondeterministic: {contents:?}"
            );
        }
        assert_eq!(
            contents[0], contents[1],
            "greedy speculative serving is NONDETERMINISTIC across identical requests"
        );

        let log = String::from_utf8_lossy(&cap.0.lock().unwrap()).into_owned();
        assert!(
            log.contains("qwen3.6 moe mtp self-speculative request done"),
            "the requests were served but the MoE MTP spec loop never ran -- the drafter \
             loaded and then serving routed around it; captured log:\n{log}"
        );
        for needle in ["accept_rate", "tokens_per_round"] {
            assert!(
                log.contains(needle),
                "spec loop completion log carries no {needle}; stats reporting broke"
            );
        }
        let eager_escape = std::env::var("NV_Q36_MTP_EAGER").as_deref() == Ok("1");
        let (want_arm, want_capture) = if eager_escape {
            ("arm=\"eager\"", "capture_active=false")
        } else {
            ("arm=\"captured-verify\"", "capture_active=true")
        };
        for needle in [want_arm, want_capture] {
            assert!(
                log.contains(needle),
                "NV_Q36_MTP_EAGER={eager_escape}: every spec-loop completion line must carry \
                 {needle} -- serving routed the wrong session for this boot arm; spec lines:\n{}",
                log.lines()
                    .filter(|l| l.contains("self-speculative request done"))
                    .collect::<Vec<_>>()
                    .join("\n")
            );
        }
        let stats_line = log
            .lines()
            .find(|l| l.contains("self-speculative request done"))
            .unwrap_or_default()
            .to_string();
        eprintln!("VERDICT: spec-on e2e green; first stats line: {stats_line}");
    }
}
