
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::sync::Arc;

use tokio::sync::Mutex as AsyncMutex;
use tokio::sync::Notify;
use tokio::task::JoinHandle;

use super::state::{PredictedRunner, PredictedSharedState};
use crate::conversation::llm::{complete_stream_messages, ChatMessage, LlmConfig};
use crate::stt::WhisperHandle;
use crate::types::MonoF32At16k;

pub(super) fn spawn_predicted_stt(
    cancel: &super::cancel::SessionCancel,
    whisper: WhisperHandle,
    audio: MonoF32At16k,
    language: Option<String>,
) -> PredictedRunner {
    let shared = Arc::new(PredictedSharedState::new());
    let shared_for_task = shared.clone();
    let task: JoinHandle<()> = tokio::spawn(cancel.wrap_unit(async move {
        let samples = audio.into_vec();
        let opts = crate::stt::SttRequestOptions {
            language,
            initial_prompt: None,
        };
        let result = tokio::task::spawn_blocking(move || {
            whisper
                .transcribe_full_with_opts(&samples, crate::stt::WhisperTask::Transcribe, &opts)
                .map(|r| r.text)
        })
        .await;
        let stored = match result {
            Ok(Ok(t)) => Ok(t),
            Ok(Err(e)) => Err(e.to_string()),
            Err(e) => Err(format!("speculative STT join failed: {e}")),
        };
        {
            let mut guard = shared_for_task.user_transcript.lock().await;
            *guard = Some(stored);
        }
        shared_for_task.done.notify_waiters();
    }));
    PredictedRunner { task, shared }
}

pub async fn await_predicted_stt(runner: &PredictedRunner) -> Result<String, String> {
    await_predicted_stt_shared(&runner.shared).await
}

pub(super) async fn await_predicted_stt_shared(
    shared: &Arc<PredictedSharedState>,
) -> Result<String, String> {
    wait_until_ready(&shared.done, || async move {
        shared.user_transcript.lock().await.clone()
    })
    .await
}

async fn wait_until_ready<T, F>(wakeup: &Notify, mut probe: impl FnMut() -> F) -> T
where
    F: std::future::Future<Output = Option<T>>,
{
    loop {
        let woken = wakeup.notified();
        tokio::pin!(woken);
        woken.as_mut().enable();
        if let Some(ready) = probe().await {
            return ready;
        }
        woken.await;
    }
}

pub struct PredictedLlmShared {
    pub buffer: AsyncMutex<Vec<String>>,
    pub overflowed: AtomicBool,
    pub dropped_tokens: AtomicU32,
    pub chars_seen: AtomicU32,
    pub done: AtomicBool,
    pub cancelled: AtomicBool,
    pub finished: Notify,
    pub progress: Notify,
    pub prompt_user_text: AsyncMutex<Option<String>>,
}

impl PredictedLlmShared {
    pub fn new() -> Self {
        Self {
            buffer: AsyncMutex::new(Vec::new()),
            overflowed: AtomicBool::new(false),
            dropped_tokens: AtomicU32::new(0),
            chars_seen: AtomicU32::new(0),
            done: AtomicBool::new(false),
            cancelled: AtomicBool::new(false),
            finished: Notify::new(),
            progress: Notify::new(),
            prompt_user_text: AsyncMutex::new(None),
        }
    }
}

pub struct PredictedLlmRunner {
    pub task: JoinHandle<()>,
    pub shared: Arc<PredictedLlmShared>,
    pub cap: u32,
}

impl std::fmt::Debug for PredictedLlmRunner {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PredictedLlmRunner")
            .field("cap", &self.cap)
            .finish_non_exhaustive()
    }
}

impl PredictedLlmRunner {
    pub fn abort(&self) {
        self.shared.cancelled.store(true, Ordering::SeqCst);
        self.shared.finished.notify_waiters();
        self.shared.progress.notify_waiters();
        self.task.abort();
    }

    pub async fn snapshot(&self) -> Vec<String> {
        let g = self.shared.buffer.lock().await;
        g.clone()
    }

    pub async fn snapshot_text(&self) -> String {
        let g = self.shared.buffer.lock().await;
        g.concat()
    }

    pub fn dropped_count(&self) -> u32 {
        self.shared.dropped_tokens.load(Ordering::Relaxed)
    }

    pub fn overflowed(&self) -> bool {
        self.shared.overflowed.load(Ordering::Relaxed)
    }

    pub fn is_done(&self) -> bool {
        self.shared.done.load(Ordering::Relaxed)
    }

    pub fn is_cancelled(&self) -> bool {
        self.shared.cancelled.load(Ordering::Relaxed)
    }

    pub fn chars_seen(&self) -> u32 {
        self.shared.chars_seen.load(Ordering::Relaxed)
    }

    pub async fn prompt_user_text(&self) -> Option<String> {
        self.shared.prompt_user_text.lock().await.clone()
    }

    pub async fn wait_finished(&self) {
        wait_until_ready(&self.shared.finished, || async move {
            (self.is_done() || self.is_cancelled()).then_some(())
        })
        .await
    }
}

async fn buffer_llm_stream_into(
    cfg: LlmConfig,
    messages: Vec<ChatMessage>,
    trajectory_tag: Option<String>,
    shared: &Arc<PredictedLlmShared>,
    cap_usize: usize,
) {
    let mut rx = complete_stream_messages(&cfg, messages, trajectory_tag);
    loop {
        if shared.cancelled.load(Ordering::Relaxed) {
            break;
        }
        let item = match rx.recv().await {
            Some(v) => v,
            None => break,
        };
        match item {
            Ok(delta) => {
                if delta.is_empty() {
                    continue;
                }
                let n = delta.chars().count() as u32;
                shared.chars_seen.fetch_add(n, Ordering::Relaxed);
                let mut buf = shared.buffer.lock().await;
                buf.push(delta);
                if buf.len() > cap_usize {
                    let drop_n = buf.len() - cap_usize;
                    let _ = buf.drain(..drop_n).count();
                    shared
                        .dropped_tokens
                        .fetch_add(drop_n as u32, Ordering::Relaxed);
                    shared.overflowed.store(true, Ordering::Relaxed);
                }
                drop(buf);
                shared.progress.notify_waiters();
            }
            Err(_) => {
                break;
            }
        }
    }
}

fn mark_llm_shared_finished(shared: &Arc<PredictedLlmShared>) {
    shared.done.store(true, Ordering::SeqCst);
    shared.finished.notify_waiters();
    shared.progress.notify_waiters();
}

#[cfg(test)]
pub(super) fn spawn_predicted_llm(
    cancel: &super::cancel::SessionCancel,
    cfg: LlmConfig,
    messages: Vec<ChatMessage>,
    cap: u32,
    trajectory_tag: Option<String>,
) -> PredictedLlmRunner {
    let shared = Arc::new(PredictedLlmShared::new());
    let shared_for_task = shared.clone();
    let cap_usize = cap.max(1) as usize;
    let task: JoinHandle<()> = tokio::spawn(cancel.wrap_unit(async move {
        buffer_llm_stream_into(cfg, messages, trajectory_tag, &shared_for_task, cap_usize).await;
        mark_llm_shared_finished(&shared_for_task);
    }));
    PredictedLlmRunner { task, shared, cap }
}

pub(super) fn spawn_predicted_llm_after_stt(
    cancel: &super::cancel::SessionCancel,
    cfg: LlmConfig,
    base_messages: Vec<ChatMessage>,
    stt_shared: Arc<PredictedSharedState>,
    cap: u32,
    trajectory_tag: Option<String>,
) -> PredictedLlmRunner {
    let shared = Arc::new(PredictedLlmShared::new());
    let shared_for_task = shared.clone();
    let cap_usize = cap.max(1) as usize;
    let task: JoinHandle<()> = tokio::spawn(cancel.wrap_unit(async move {
        let partial = match await_predicted_stt_shared(&stt_shared).await {
            Ok(t) if !t.trim().is_empty() => t,
            _ => {
                mark_llm_shared_finished(&shared_for_task);
                return;
            }
        };
        if shared_for_task.cancelled.load(Ordering::Relaxed) {
            mark_llm_shared_finished(&shared_for_task);
            return;
        }
        {
            let mut guard = shared_for_task.prompt_user_text.lock().await;
            *guard = Some(partial.clone());
        }
        let mut messages = base_messages;
        messages.push(ChatMessage {
            role: "user".into(),
            content: partial,
        });
        buffer_llm_stream_into(cfg, messages, trajectory_tag, &shared_for_task, cap_usize).await;
        mark_llm_shared_finished(&shared_for_task);
    }));
    PredictedLlmRunner { task, shared, cap }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn shared_state_starts_empty() {
        let s = PredictedSharedState::new();
        let g = s.user_transcript.lock().await;
        assert!(g.is_none());
    }

    #[tokio::test]
    async fn await_returns_stored_result() {
        let s = Arc::new(PredictedSharedState::new());
        let s_for_writer = s.clone();
        tokio::spawn(async move {
            tokio::time::sleep(std::time::Duration::from_millis(20)).await;
            *s_for_writer.user_transcript.lock().await = Some(Ok("hello".into()));
            s_for_writer.done.notify_waiters();
        });
        let runner = PredictedRunner {
            task: tokio::spawn(async {}),
            shared: s,
        };
        let r = await_predicted_stt(&runner).await;
        assert_eq!(r.ok().as_deref(), Some("hello"));
    }

    #[tokio::test]
    async fn a_result_stored_and_notified_between_the_check_and_the_wait_is_not_lost() {
        let shared = PredictedSharedState::new();
        let checks = AtomicU32::new(0);
        let (shared, checks) = (&shared, &checks);
        let got = tokio::time::timeout(
            std::time::Duration::from_secs(2),
            wait_until_ready(&shared.done, || async move {
                let seen = shared.user_transcript.lock().await.clone();
                if checks.fetch_add(1, Ordering::SeqCst) == 0 {
                    *shared.user_transcript.lock().await = Some(Ok("hello".into()));
                    shared.done.notify_waiters();
                }
                seen
            }),
        )
        .await
        .expect("the wakeup sent between the empty check and the wait was lost");
        assert_eq!(got.ok().as_deref(), Some("hello"));
    }
}

#[cfg(test)]
mod llm_runner_tests {
    use super::*;
    use crate::realtime::cancel::SessionCancel;
    use axum::body::Body;
    use axum::http::header;
    use axum::response::Response;
    use axum::routing::post;
    use axum::Router;
    use std::sync::Arc as StdArc;
    use std::sync::Mutex as StdMutex;
    use tokio::net::TcpListener;

    enum Behavior {
        Tokens(Vec<String>),
        ManyTokens(usize),
    }

    async fn spawn_mock(behavior: Behavior) -> String {
        let behavior = StdArc::new(StdMutex::new(Some(behavior)));
        let app = Router::new().route(
            "/v1/chat/completions",
            post(move || {
                let behavior = behavior.clone();
                async move {
                    let b = behavior.lock().unwrap().take().expect("one-shot mock");
                    let mut sse = String::new();
                    let toks: Vec<String> = match b {
                        Behavior::Tokens(t) => t,
                        Behavior::ManyTokens(n) => (0..n).map(|i| format!("t{i} ")).collect(),
                    };
                    for tok in toks {
                        let chunk = serde_json::json!({
                            "choices": [{"delta": {"content": tok}}]
                        });
                        sse.push_str(&format!("data: {chunk}\n\n"));
                    }
                    sse.push_str("data: [DONE]\n\n");
                    Response::builder()
                        .status(200)
                        .header(header::CONTENT_TYPE, "text/event-stream")
                        .body(Body::from(sse))
                        .unwrap()
                }
            }),
        );
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move {
            axum::serve(listener, app).await.unwrap();
        });
        format!("http://{addr}/v1")
    }

    fn cfg(base: String) -> LlmConfig {
        LlmConfig {
            base_url: base,
            api_key: None,
            model: "test-model".into(),
        }
    }

    fn user_msg(content: &str) -> ChatMessage {
        ChatMessage {
            role: "user".into(),
            content: content.into(),
        }
    }

    #[tokio::test]
    async fn predicted_llm_buffers_tokens_until_done() {
        let base = spawn_mock(Behavior::Tokens(vec![
            "hello".into(),
            " ".into(),
            "world".into(),
        ]))
        .await;
        let runner = spawn_predicted_llm(
            &SessionCancel::new(),
            cfg(base),
            vec![user_msg("hi")],
            16,
            None,
        );
        runner.wait_finished().await;
        assert!(runner.is_done());
        assert!(!runner.overflowed());
        let text = runner.snapshot_text().await;
        assert_eq!(text, "hello world");
        runner.abort();
    }

    #[tokio::test]
    async fn predicted_llm_overflow_drops_oldest_and_sets_flag() {
        let base = spawn_mock(Behavior::ManyTokens(20)).await;
        let runner = spawn_predicted_llm(
            &SessionCancel::new(),
            cfg(base),
            vec![user_msg("hi")],
            4,
            None,
        );
        runner.wait_finished().await;
        assert!(runner.is_done());
        assert!(runner.overflowed());
        assert!(runner.dropped_count() >= 1);
        let buf = runner.snapshot().await;
        assert!(buf.len() <= 4);
        runner.abort();
    }

    #[tokio::test]
    async fn predicted_llm_abort_marks_cancelled_and_finishes_wait() {
        let base = spawn_mock(Behavior::ManyTokens(2)).await;
        let runner = spawn_predicted_llm(
            &SessionCancel::new(),
            cfg(base),
            vec![user_msg("hi")],
            16,
            None,
        );
        runner.abort();
        let waited = tokio::time::timeout(
            std::time::Duration::from_millis(500),
            runner.wait_finished(),
        )
        .await;
        assert!(waited.is_ok(), "wait_finished must return after abort");
        assert!(runner.is_cancelled());
    }

    #[tokio::test]
    async fn predicted_llm_chars_seen_tracks_unicode_count() {
        let base = spawn_mock(Behavior::Tokens(vec!["héllo".into(), "世界".into()])).await;
        let runner = spawn_predicted_llm(
            &SessionCancel::new(),
            cfg(base),
            vec![user_msg("hi")],
            16,
            None,
        );
        runner.wait_finished().await;
        assert_eq!(runner.chars_seen(), "héllo".chars().count() as u32 + 2);
        runner.abort();
    }

    async fn spawn_echo_last_user_mock() -> String {
        let app = Router::new().route(
            "/v1/chat/completions",
            post(|body: String| async move {
                let v: serde_json::Value = serde_json::from_str(&body).unwrap();
                let last_user = v["messages"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .rev()
                    .find(|m| m["role"] == "user")
                    .map(|m| m["content"].as_str().unwrap_or_default().to_string())
                    .unwrap_or_default();
                let chunk = serde_json::json!({
                    "choices": [{"delta": {"content": last_user}}]
                });
                let sse = format!("data: {chunk}\n\ndata: [DONE]\n\n");
                Response::builder()
                    .status(200)
                    .header(header::CONTENT_TYPE, "text/event-stream")
                    .body(Body::from(sse))
                    .unwrap()
            }),
        );
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move {
            axum::serve(listener, app).await.unwrap();
        });
        format!("http://{addr}/v1")
    }

    fn stt_shared_with(result: Result<String, String>) -> StdArc<PredictedSharedState> {
        let shared = StdArc::new(PredictedSharedState::new());
        {
            let mut guard = shared.user_transcript.try_lock().unwrap();
            *guard = Some(result);
        }
        shared.done.notify_waiters();
        shared
    }

    #[tokio::test]
    async fn eager_llm_prompt_is_the_speculative_stt_partial_transcript() {
        let base = spawn_echo_last_user_mock().await;
        let stt = stt_shared_with(Ok("what is the weather in lisbon".into()));
        let runner = spawn_predicted_llm_after_stt(
            &SessionCancel::new(),
            cfg(base),
            vec![ChatMessage {
                role: "system".into(),
                content: "be brief".into(),
            }],
            stt,
            16,
            None,
        );
        runner.wait_finished().await;
        assert_eq!(
            runner.prompt_user_text().await.as_deref(),
            Some("what is the weather in lisbon"),
            "the eager LLM prompt must be the real partial transcript, never a placeholder"
        );
        assert_eq!(
            runner.snapshot_text().await,
            "what is the weather in lisbon",
            "echo mock proves the LLM request carried the partial as its last user message"
        );
        runner.abort();
    }

    #[tokio::test]
    async fn eager_llm_is_skipped_when_speculative_stt_fails_or_is_empty() {
        for stt_result in [Err("stt exploded".to_string()), Ok("   ".to_string())] {
            let base = spawn_echo_last_user_mock().await;
            let stt = stt_shared_with(stt_result);
            let runner = spawn_predicted_llm_after_stt(
                &SessionCancel::new(),
                cfg(base),
                vec![user_msg("earlier turn")],
                stt,
                16,
                None,
            );
            runner.wait_finished().await;
            assert!(runner.is_done());
            assert_eq!(runner.prompt_user_text().await, None);
            assert_eq!(
                runner.snapshot_text().await,
                "",
                "no partial transcript means no eager LLM call and no trajectory record"
            );
            runner.abort();
        }
    }
}
