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

const REAL_WEIGHTS_TEST_ENV: &str = "NV_Q38_SCHED_STALL_TEST";

const SHORT_SEND_TIMEOUT_MS_SO_THE_EVICTION_FIRES_MID_GROUP: &str = "300";

const STALLED_CLIENT_CHANNEL_CAP_1_FILLS_ON_THE_FIRST_DELTA: usize = 1;

const HEALTHY_WALL_BOUND_S_PRE_FIX_THE_PARKED_DISPATCHER_MADE_THIS_INFINITE: u64 = 180;

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
#[ignore = "loads the ~16 GB Qwen3.8-27B NVFP4 checkpoint once with NV_Q38_BATCH=1, forms a \
            two-member group where one client never reads, and asserts the reader completes, \
            the stalled member is evicted after NV_SSE_SEND_TIMEOUT_MS, and a third request \
            still gets served (the pre-fix bare send parked the whole scheduler); set \
            NV_Q38_SCHED_STALL_TEST=1"]
fn a_stalled_reader_is_evicted_while_its_group_sibling_and_the_next_request_are_served() {
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
    std::env::remove_var("NV_DRAFTER");
    std::env::set_var(
        "NV_SSE_SEND_TIMEOUT_MS",
        SHORT_SEND_TIMEOUT_MS_SO_THE_EVICTION_FIRES_MID_GROUP,
    );
    let engine: Arc<dyn ChatEngine> =
        Arc::new(NvEngineChat::try_load(&dir).expect("NvEngineChat::try_load"));
    let rt = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(2)
        .enable_all()
        .build()
        .unwrap();
    rt.block_on(async {
        let (htx, hrx) = mpsc::channel::<ChatEvent>(64);
        let (stx, srx_held_open_and_never_read) =
            mpsc::channel::<ChatEvent>(STALLED_CLIENT_CHANNEL_CAP_1_FILLS_ON_THE_FIRST_DELTA);
        engine
            .generate(
                req("Write one paragraph of about 150 words about a lighthouse keeper.", 96),
                htx,
            )
            .await
            .expect("submit healthy");
        engine
            .generate(
                req("Write one paragraph of about 150 words about why the sky is blue.", 96),
                stx,
            )
            .await
            .expect("submit stalled");
        let t0 = std::time::Instant::now();
        let (deltas, finish) = tokio::time::timeout(
            Duration::from_secs(HEALTHY_WALL_BOUND_S_PRE_FIX_THE_PARKED_DISPATCHER_MADE_THIS_INFINITE),
            drain_to_done(hrx, "healthy"),
        )
        .await
        .expect("healthy sibling stalled: the scheduler is parked behind the non-reading client");
        eprintln!(
            "[q38-sched-stall-e2e] healthy member done: deltas={deltas} finish={finish} \
             wall={:.1}s",
            t0.elapsed().as_secs_f64()
        );
        assert!(deltas > 0, "healthy member streamed no deltas");
        assert_eq!(finish, "length");
        let (tx3, rx3) = mpsc::channel::<ChatEvent>(64);
        engine
            .generate(req("Write one sentence about rivers.", 24), tx3)
            .await
            .expect("submit follow-up");
        let t1 = std::time::Instant::now();
        let (deltas3, finish3) = tokio::time::timeout(
            Duration::from_secs(HEALTHY_WALL_BOUND_S_PRE_FIX_THE_PARKED_DISPATCHER_MADE_THIS_INFINITE),
            drain_to_done(rx3, "follow-up"),
        )
        .await
        .expect("follow-up request never served: the scheduler did not survive the eviction");
        eprintln!(
            "[q38-sched-stall-e2e] follow-up done: deltas={deltas3} finish={finish3} \
             wall={:.1}s",
            t1.elapsed().as_secs_f64()
        );
        assert!(deltas3 > 0, "follow-up streamed no deltas");
        drop(srx_held_open_and_never_read);
    });
}
