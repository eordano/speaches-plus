#![cfg(feature = "cuda")]

#[path = "common/qwen38_fixture.rs"]
#[allow(dead_code)]
mod qwen38_fixture;

use std::sync::Arc;

use axum::body::{to_bytes, Body};
use axum::http::{Request, StatusCode};
use axum::routing::post;
use axum::Router;
use futures_util::StreamExt;
use tower::ServiceExt;

use qwen38_fixture::qwen38_nvfp4_snapshot_dir_env_override_then_home_hub;
use speaches_plus::oapi::chat::{handle_chat_completions, ChatAppState, ChatEngine};
use speaches_plus::oapi::chat_engine::{ChatRegistry, NvEngineChat};

const SOAK_GATE_ENV_NV_VRAM_SOAK_1_BECAUSE_A_SILENT_SKIP_WOULD_READ_AS_A_LEAK_FREE_PASS: &str =
    "NV_VRAM_SOAK";

const SOLO_REQUESTS_20_HALF_THE_40_REQUEST_SOAK: usize = 20;

const WARMUP_REQUESTS_4_ABSORB_GRAPH_CAPTURE_AND_ALLOCATOR_STEADY_STATE: usize = 4;

const PROMPT_SENTENCE_REPEATS_VARIED_1_15_30_60_SPAN_SHORT_TO_KILO_TOKEN_PREFILLS: [usize; 4] =
    [1, 15, 30, 60];

const MAX_TOKENS_VARIED_16_32_48_64_SO_KV_AND_DECODE_LIFETIMES_DIFFER_PER_REQUEST: [u32; 4] =
    [16, 32, 48, 64];

const POST_WARMUP_FREE_TOLERANCE_MIB_512_A_LEAKED_PER_REQUEST_KV_CACHE_COMPOUNDS_PAST_THIS: f64 =
    512.0;

const STREAM_DROP_MAX_TOKENS_256_LONG_ENOUGH_THAT_THE_CLIENT_DROP_LANDS_MID_GENERATION: u32 = 256;

const STREAM_FRAMES_TO_READ_3_THEN_THE_CLIENT_VANISHES: usize = 3;

const STREAM_DROP_RECOVERY_DEADLINE_S_90_COVERS_A_RUN_TO_MAX_TOKENS_IF_CANCEL_IS_NOT_NOTICED: u64 =
    90;

const BATCH_SEQ_REQUESTS_4_THE_SOLO_PATH_INSIDE_THE_BATCH_BUILD: usize = 4;

const BATCH_CONCURRENT_GROUPS_8_OF_4_LANES_SO_A_PER_GROUP_SLOPE_CANNOT_HIDE_IN_THE_TOLERANCE:
    usize = 8;

const BATCH_WARMUP_GROUPS_2_GROUP1_BOOTS_THE_LANES_GROUP2_ENTERS_STEADY_STATE: usize = 2;

const SETTLE_POLL_INTERVAL_MS_500_AND_BAND_16_MIB_LET_STREAM_ORDERED_FREES_LAND: (u64, f64) =
    (500, 16.0);

const SETTLE_POLL_CAP_S_10_A_FREE_THAT_NEVER_SETTLES_IS_ITSELF_THE_FINDING: u64 = 10;

async fn settled_free_mib() -> f64 {
    let deadline = std::time::Instant::now()
        + std::time::Duration::from_secs(SETTLE_POLL_CAP_S_10_A_FREE_THAT_NEVER_SETTLES_IS_ITSELF_THE_FINDING);
    let (interval_ms, band) = SETTLE_POLL_INTERVAL_MS_500_AND_BAND_16_MIB_LET_STREAM_ORDERED_FREES_LAND;
    let mut prev = device_free_mib();
    loop {
        tokio::time::sleep(std::time::Duration::from_millis(interval_ms)).await;
        let cur = device_free_mib();
        if (cur - prev).abs() <= band || std::time::Instant::now() > deadline {
            return cur;
        }
        prev = cur;
    }
}

fn soak_gate_or_panic() {
    if std::env::var(SOAK_GATE_ENV_NV_VRAM_SOAK_1_BECAUSE_A_SILENT_SKIP_WOULD_READ_AS_A_LEAK_FREE_PASS)
        .as_deref()
        != Ok("1")
    {
        panic!(
            "set {SOAK_GATE_ENV_NV_VRAM_SOAK_1_BECAUSE_A_SILENT_SKIP_WOULD_READ_AS_A_LEAK_FREE_PASS}=1 \
             to run this soak; it must never silently skip"
        );
    }
}

fn device_free_mib() -> f64 {
    let (free, _total) =
        nv_layers::cudarc::driver::result::mem_get_info().expect("cuMemGetInfo");
    free as f64 / (1024.0 * 1024.0)
}

fn app(engine: Arc<dyn ChatEngine>) -> Router {
    Router::new()
        .route("/v1/chat/completions", post(handle_chat_completions))
        .with_state(ChatAppState {
            registry: ChatRegistry::single(engine),
        })
}

fn prompt_of_repeats(repeats: usize) -> String {
    let sentence = "The lighthouse keeper climbed the spiral staircase every evening at dusk, \
                    carrying a small brass lamp and a logbook whose pages had softened with salt \
                    air, recording the wind, the visibility, and the ships that passed the point. ";
    let mut text = String::with_capacity(sentence.len() * repeats + 64);
    for _ in 0..repeats {
        text.push_str(sentence);
    }
    text.push_str("Summarize the scene above in one short sentence.");
    text
}

fn chat_request(model_id: &str, prompt: &str, max_tokens: u32, stream: bool) -> Request<Body> {
    let body = format!(
        r#"{{"model":"{}","max_tokens":{},"temperature":0,"stream":{},
             "enable_thinking":false,
             "messages":[{{"role":"user","content":{}}}]}}"#,
        model_id,
        max_tokens,
        stream,
        serde_json::to_string(prompt).unwrap()
    );
    Request::builder()
        .method("POST")
        .uri("/v1/chat/completions")
        .header("content-type", "application/json")
        .body(Body::from(body))
        .unwrap()
}

async fn run_one_ok(router: Router, label: &str, model_id: &str, prompt: &str, max_tokens: u32) {
    let resp = router
        .oneshot(chat_request(model_id, prompt, max_tokens, false))
        .await
        .unwrap();
    let status = resp.status();
    let bytes = to_bytes(resp.into_body(), 1 << 22).await.unwrap();
    let text = String::from_utf8_lossy(&bytes).into_owned();
    assert_eq!(status, StatusCode::OK, "[{label}] {text}");
    let v: serde_json::Value = serde_json::from_str(&text).unwrap();
    let completion_tokens = v["usage"]["completion_tokens"].as_u64().unwrap_or(0);
    assert!(completion_tokens > 0, "[{label}] no tokens generated: {text}");
}

fn load_engine(dir: &std::path::Path, label: &str) -> Arc<dyn ChatEngine> {
    let t = std::time::Instant::now();
    let engine = NvEngineChat::try_load(dir)
        .unwrap_or_else(|e| panic!("[{label}] NvEngineChat::try_load: {e:#}"));
    eprintln!(
        "[serving-vram-soak] [{label}] engine ready in {:.1}s free_mib={:.1}",
        t.elapsed().as_secs_f64(),
        device_free_mib()
    );
    Arc::new(engine)
}

fn assert_free_within_tolerance(label: &str, baseline_mib: f64, free_mib: f64) {
    let held = baseline_mib - free_mib;
    assert!(
        held <= POST_WARMUP_FREE_TOLERANCE_MIB_512_A_LEAKED_PER_REQUEST_KV_CACHE_COMPOUNDS_PAST_THIS,
        "[{label}] free {free_mib:.1} MiB sits {held:.1} MiB below the post-warmup baseline \
         {baseline_mib:.1} MiB, above the {} MiB tolerance -- request-scoped device memory \
         (KV cache, lane scratch, drafter state) is outliving its request",
        POST_WARMUP_FREE_TOLERANCE_MIB_512_A_LEAKED_PER_REQUEST_KV_CACHE_COMPOUNDS_PAST_THIS
    );
}

fn run_solo_phase(engine: &Arc<dyn ChatEngine>) -> f64 {
    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    rt.block_on(async {
        let model_id = engine.model_id().to_string();
        let mut baseline: Option<f64> = None;
        for i in 1..=SOLO_REQUESTS_20_HALF_THE_40_REQUEST_SOAK {
            let repeats =
                PROMPT_SENTENCE_REPEATS_VARIED_1_15_30_60_SPAN_SHORT_TO_KILO_TOKEN_PREFILLS
                    [i % PROMPT_SENTENCE_REPEATS_VARIED_1_15_30_60_SPAN_SHORT_TO_KILO_TOKEN_PREFILLS.len()];
            let max_tokens = MAX_TOKENS_VARIED_16_32_48_64_SO_KV_AND_DECODE_LIFETIMES_DIFFER_PER_REQUEST
                [i % MAX_TOKENS_VARIED_16_32_48_64_SO_KV_AND_DECODE_LIFETIMES_DIFFER_PER_REQUEST.len()];
            let prompt = prompt_of_repeats(repeats);
            run_one_ok(
                app(engine.clone()),
                &format!("solo-{i}"),
                &model_id,
                &prompt,
                max_tokens,
            )
            .await;
            let free = device_free_mib();
            eprintln!(
                "[serving-vram-soak] solo req={i} repeats={repeats} max_tokens={max_tokens} \
                 free_mib={free:.1}"
            );
            if i == WARMUP_REQUESTS_4_ABSORB_GRAPH_CAPTURE_AND_ALLOCATOR_STEADY_STATE {
                baseline = Some(free);
            }
            if let Some(b) = baseline {
                if i > WARMUP_REQUESTS_4_ABSORB_GRAPH_CAPTURE_AND_ALLOCATOR_STEADY_STATE {
                    assert_free_within_tolerance(&format!("solo-{i}"), b, free);
                }
            }
        }
        baseline.expect("warmup ran")
    })
}

fn run_stream_drop_phase(engine: &Arc<dyn ChatEngine>, baseline_mib: f64) {
    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    rt.block_on(async {
        let model_id = engine.model_id().to_string();
        let prompt = prompt_of_repeats(15);
        let resp = app(engine.clone())
            .oneshot(chat_request(
                &model_id,
                &prompt,
                STREAM_DROP_MAX_TOKENS_256_LONG_ENOUGH_THAT_THE_CLIENT_DROP_LANDS_MID_GENERATION,
                true,
            ))
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK, "stream request rejected");
        let mut frames = 0usize;
        {
            let mut data = resp.into_body().into_data_stream();
            while let Some(chunk) = data.next().await {
                let chunk = chunk.expect("stream chunk");
                if !chunk.is_empty() {
                    frames += 1;
                }
                if frames >= STREAM_FRAMES_TO_READ_3_THEN_THE_CLIENT_VANISHES {
                    break;
                }
            }
        }
        assert!(
            frames >= 1,
            "the stream produced no data frames before the drop; the client-drop case never \
             started generating"
        );
        eprintln!(
            "[serving-vram-soak] stream-drop client vanished after {frames} frames \
             free_mib={:.1}",
            device_free_mib()
        );

        let deadline = std::time::Instant::now()
            + std::time::Duration::from_secs(
                STREAM_DROP_RECOVERY_DEADLINE_S_90_COVERS_A_RUN_TO_MAX_TOKENS_IF_CANCEL_IS_NOT_NOTICED,
            );
        let mut recovered_at: Option<f64> = None;
        loop {
            tokio::time::sleep(std::time::Duration::from_secs(2)).await;
            let free = device_free_mib();
            if baseline_mib - free
                <= POST_WARMUP_FREE_TOLERANCE_MIB_512_A_LEAKED_PER_REQUEST_KV_CACHE_COMPOUNDS_PAST_THIS
            {
                recovered_at = Some(free);
                break;
            }
            if std::time::Instant::now() > deadline {
                break;
            }
        }
        let free = device_free_mib();
        assert!(
            recovered_at.is_some(),
            "free memory ({free:.1} MiB) never returned to the post-warmup baseline \
             ({baseline_mib:.1} MiB) within {}s of the client dropping mid-stream; the \
             abandoned request's KV cache or lane is not being freed",
            STREAM_DROP_RECOVERY_DEADLINE_S_90_COVERS_A_RUN_TO_MAX_TOKENS_IF_CANCEL_IS_NOT_NOTICED
        );
        eprintln!(
            "[serving-vram-soak] stream-drop recovered free_mib={:.1} baseline_mib={baseline_mib:.1}",
            recovered_at.unwrap()
        );

        run_one_ok(
            app(engine.clone()),
            "post-stream-drop",
            &model_id,
            &prompt_of_repeats(1),
            16,
        )
        .await;
        eprintln!(
            "[serving-vram-soak] post-stream-drop request served free_mib={:.1}",
            device_free_mib()
        );
    });
}

fn run_batch_phase(engine: &Arc<dyn ChatEngine>) {
    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    rt.block_on(async {
        let model_id = engine.model_id().to_string();
        let mut baseline: Option<f64> = None;
        for i in 1..=BATCH_SEQ_REQUESTS_4_THE_SOLO_PATH_INSIDE_THE_BATCH_BUILD {
            let repeats =
                PROMPT_SENTENCE_REPEATS_VARIED_1_15_30_60_SPAN_SHORT_TO_KILO_TOKEN_PREFILLS
                    [i % PROMPT_SENTENCE_REPEATS_VARIED_1_15_30_60_SPAN_SHORT_TO_KILO_TOKEN_PREFILLS.len()];
            run_one_ok(
                app(engine.clone()),
                &format!("batch-seq-{i}"),
                &model_id,
                &prompt_of_repeats(repeats),
                32,
            )
            .await;
            let free = device_free_mib();
            eprintln!("[serving-vram-soak] batch-seq req={i} free_mib={free:.1}");
        }

        for g in 1..=BATCH_CONCURRENT_GROUPS_8_OF_4_LANES_SO_A_PER_GROUP_SLOPE_CANNOT_HIDE_IN_THE_TOLERANCE
        {
            let futs = (0..4usize).map(|j| {
                let repeats =
                    PROMPT_SENTENCE_REPEATS_VARIED_1_15_30_60_SPAN_SHORT_TO_KILO_TOKEN_PREFILLS
                        [(g + j) % PROMPT_SENTENCE_REPEATS_VARIED_1_15_30_60_SPAN_SHORT_TO_KILO_TOKEN_PREFILLS.len()];
                let max_tokens =
                    MAX_TOKENS_VARIED_16_32_48_64_SO_KV_AND_DECODE_LIFETIMES_DIFFER_PER_REQUEST
                        [j % MAX_TOKENS_VARIED_16_32_48_64_SO_KV_AND_DECODE_LIFETIMES_DIFFER_PER_REQUEST.len()];
                let router = app(engine.clone());
                let model_id = model_id.clone();
                let label = format!("batch-conc-g{g}-r{j}");
                async move {
                    let prompt = prompt_of_repeats(repeats);
                    run_one_ok(router, &label, &model_id, &prompt, max_tokens).await;
                }
            });
            futures_util::future::join_all(futs).await;
            let free = settled_free_mib().await;
            eprintln!("[serving-vram-soak] batch-conc group={g} settled_free_mib={free:.1}");
            if g == BATCH_WARMUP_GROUPS_2_GROUP1_BOOTS_THE_LANES_GROUP2_ENTERS_STEADY_STATE {
                baseline = Some(free);
            }
            if g > BATCH_WARMUP_GROUPS_2_GROUP1_BOOTS_THE_LANES_GROUP2_ENTERS_STEADY_STATE {
                assert_free_within_tolerance(
                    &format!(
                        "batch-conc-g{g} (baseline is the settled free after group 2: group 1 \
                         materializes the bounded 4-lane boot allocation and group 2 enters \
                         steady state; a real per-group leak still compounds past the \
                         tolerance across the remaining groups)"
                    ),
                    baseline.expect("group-2 baseline set"),
                    free,
                );
            }
        }
    });
}

#[test]
#[ignore = "loads the Qwen3.8-27B NVFP4 checkpoint twice (mtp solo, then NV_Q38_BATCH=1) into an \
            in-process axum router and serves 40+ requests of varied prompt length and \
            max_tokens, sampling cuMemGetInfo between requests; set NV_VRAM_SOAK=1 -- free \
            memory must stay within 512 MiB of the post-warmup baseline for every later \
            request, a client dropping a stream mid-generation must not strand its KV, and \
            4x4 concurrent batch groups must return to the batch baseline"]
fn serving_soak_40_requests_solo_then_batch_free_memory_flat_after_warmup() {
    soak_gate_or_panic();
    let _ = tracing_subscriber::fmt()
        .with_env_filter("speaches_plus=warn")
        .with_writer(std::io::stderr)
        .try_init();
    let dir = qwen38_nvfp4_snapshot_dir_env_override_then_home_hub();
    std::env::set_var("NV_QWEN35_DENSE_CUDA_SERVE", "1");
    std::env::set_var("NV_GDN_FUSED_DECODE", "1");
    std::env::set_var("NV_DRAFTER", "mtp");
    std::env::remove_var("NV_Q38_BATCH");

    let solo_engine = load_engine(&dir, "solo");
    let solo_baseline = run_solo_phase(&solo_engine);
    run_stream_drop_phase(&solo_engine, solo_baseline);
    drop(solo_engine);
    let free_after_solo_drop = device_free_mib();
    eprintln!(
        "[serving-vram-soak] solo engine dropped free_mib={free_after_solo_drop:.1}"
    );

    std::env::set_var("NV_Q38_BATCH", "1");
    let batch_engine = load_engine(&dir, "batch");
    run_batch_phase(&batch_engine);
    drop(batch_engine);
    std::env::remove_var("NV_Q38_BATCH");
    eprintln!(
        "[serving-vram-soak] VERDICT solo=20 stream_drop=1 batch=20 final_free_mib={:.1}",
        device_free_mib()
    );
}
