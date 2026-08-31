#![cfg(feature = "cuda")]

#[path = "common/qwen38_fixture.rs"]
#[allow(dead_code)]
mod qwen38_fixture;

use std::sync::Arc;
use std::time::Duration;

use qwen38_fixture::qwen38_nvfp4_snapshot_dir_env_override_then_home_hub;
use speaches_plus::oapi::chat::{ChatEngine, ChatEvent, ChatGenerateRequest};
use speaches_plus::oapi::chat_engine::NvEngineChat;
use tokio::sync::mpsc;

const REAL_WEIGHTS_TEST_ENV: &str = "NV_Q38_SOLO_STALL_TEST";

const SHORT_SEND_TIMEOUT_MS_SO_THE_SOLO_EVICTION_FIRES_QUICKLY: &str = "300";

const STALLED_CLIENT_CHANNEL_CAP_1_FILLS_ON_STARTED_SO_THE_FIRST_DELTA_BLOCKS: usize = 1;

const SOLO_PICKUP_BOUND_S_STARTED_ARRIVES_BEFORE_PREFILL: u64 = 120;

const QUEUED_JOB_WALL_BOUND_S_PRE_FIX_THE_PARKED_DISPATCHER_MADE_THIS_INFINITE: u64 = 180;

fn req(prompt: &str, max_new: usize) -> ChatGenerateRequest {
    ChatGenerateRequest {
        prompt: prompt.to_string(),
        max_new_tokens: max_new,
        stop: vec![],
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
        logit_bias: vec![],
        logprobs: false,
        top_logprobs: 0,
        kv_resume: None,
        kv_store: None,
        mm: None,
    }
}

async fn drain_to_done(mut rx: mpsc::Receiver<ChatEvent>, label: &str) -> (usize, String) {
    let mut deltas = 0usize;
    loop {
        match rx.recv().await {
            Some(ChatEvent::TextDelta(_)) => deltas += 1,
            Some(ChatEvent::Done { finish_reason, .. }) => return (deltas, finish_reason),
            Some(ChatEvent::Error(e)) => panic!("[{label}] engine error: {e}"),
            Some(_) => {}
            None => panic!("[{label}] channel closed before Done"),
        }
    }
}

#[test]
#[ignore = "loads the ~16 GB Qwen3.8-27B NVFP4 checkpoint once with NV_Q38_BATCH=1, lets a \
            lone request take the scheduler's run_solo route with a client that never reads, \
            and asserts a queued second request is still served after NV_SSE_SEND_TIMEOUT_MS \
            and the stalled solo stream is closed (the pre-fix bare send in the solo bodies \
            parked the single-thread dispatcher forever); set NV_Q38_SOLO_STALL_TEST=1"]
fn a_non_reading_solo_client_is_shed_at_the_timeout_and_the_queued_job_is_served() {
    if std::env::var(REAL_WEIGHTS_TEST_ENV).as_deref() != Ok("1") {
        panic!(
            "this test was asked for BY NAME but {REAL_WEIGHTS_TEST_ENV}=1 is not set. This is \
             a SKIP, not a pass."
        );
    }
    let dir = qwen38_nvfp4_snapshot_dir_env_override_then_home_hub();
    std::env::set_var("NV_QWEN35_DENSE_CUDA_SERVE", "1");
    std::env::set_var("NV_GDN_FUSED_DECODE", "1");
    std::env::set_var("NV_Q38_BATCH", "1");
    std::env::remove_var("NV_Q38_SOLO_VIA_LANES");
    std::env::remove_var("NV_DRAFTER");
    std::env::set_var(
        "NV_SSE_SEND_TIMEOUT_MS",
        SHORT_SEND_TIMEOUT_MS_SO_THE_SOLO_EVICTION_FIRES_QUICKLY,
    );
    let engine: Arc<dyn ChatEngine> =
        Arc::new(NvEngineChat::try_load(&dir).expect("NvEngineChat::try_load"));
    let rt = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(2)
        .enable_all()
        .build()
        .unwrap();
    rt.block_on(async {
        let (stx, mut srx_held_open_and_never_read) = mpsc::channel::<ChatEvent>(
            STALLED_CLIENT_CHANNEL_CAP_1_FILLS_ON_STARTED_SO_THE_FIRST_DELTA_BLOCKS,
        );
        let stx_probe_watches_the_channel_fill_to_prove_run_solo_owns_the_dispatcher = stx.clone();
        engine
            .generate(
                req(
                    "Write one paragraph of about 150 words about why the sky is blue.",
                    96,
                ),
                stx,
            )
            .await
            .expect("submit stalled solo");
        let t_pickup = std::time::Instant::now();
        while stx_probe_watches_the_channel_fill_to_prove_run_solo_owns_the_dispatcher.capacity()
            > 0
        {
            assert!(
                t_pickup.elapsed()
                    < Duration::from_secs(SOLO_PICKUP_BOUND_S_STARTED_ARRIVES_BEFORE_PREFILL),
                "the lone request never reached run_solo: no Started event buffered"
            );
            tokio::time::sleep(Duration::from_millis(50)).await;
        }
        eprintln!(
            "[q38-solo-stall-e2e] solo picked up (Started buffered) after {:.2}s; submitting \
             the queued job behind the non-reading client",
            t_pickup.elapsed().as_secs_f64()
        );
        let (htx, hrx) = mpsc::channel::<ChatEvent>(64);
        engine
            .generate(
                req(
                    "Write one paragraph of about 150 words about a lighthouse keeper.",
                    96,
                ),
                htx,
            )
            .await
            .expect("submit queued job");
        let t0 = std::time::Instant::now();
        let (deltas, finish) = tokio::time::timeout(
            Duration::from_secs(QUEUED_JOB_WALL_BOUND_S_PRE_FIX_THE_PARKED_DISPATCHER_MADE_THIS_INFINITE),
            drain_to_done(hrx, "queued"),
        )
        .await
        .expect(
            "queued job never served: the dispatcher is still parked inside the stalled solo \
             body's send",
        );
        eprintln!(
            "[q38-solo-stall-e2e] queued job done: deltas={deltas} finish={finish} wall={:.1}s",
            t0.elapsed().as_secs_f64()
        );
        assert!(deltas > 0, "queued job streamed no deltas");
        assert_eq!(finish, "length");
        drop(stx_probe_watches_the_channel_fill_to_prove_run_solo_owns_the_dispatcher);
        let mut stalled_events = 0usize;
        loop {
            match tokio::time::timeout(
                Duration::from_secs(30),
                srx_held_open_and_never_read.recv(),
            )
            .await
            .expect(
                "the stalled stream never closed: the solo body did not drop its sender on \
                 eviction",
            ) {
                Some(_) => stalled_events += 1,
                None => break,
            }
        }
        eprintln!(
            "[q38-solo-stall-e2e] stalled stream closed after {stalled_events} buffered event(s)"
        );
        assert!(
            stalled_events >= 1,
            "the stalled client never even got Started, so this run proved pickup by \
             accident, not eviction"
        );
    });
}
