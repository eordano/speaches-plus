#![allow(dead_code)]

use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::sync::Arc;

use tokio::sync::Mutex as AsyncMutex;
use tokio::sync::Notify;
use tokio::task::JoinHandle;

use super::state::{PredictedRunner, PredictedSharedState};
use crate::conversation::llm::{complete_stream_messages, ChatMessage, LlmConfig};
use crate::stt::WhisperHandle;
use crate::types::MonoF32At16k;

pub struct PredictedTokenBuffer {
    cap: usize,
    inner: VecDeque<String>,
    dropped: u32,
}

impl PredictedTokenBuffer {
    pub fn new(cap: u32) -> Self {
        let cap = cap.max(1) as usize;
        Self {
            cap,
            inner: VecDeque::with_capacity(cap),
            dropped: 0,
        }
    }

    pub fn push(&mut self, token: String) -> bool {
        let overflowed = self.inner.len() == self.cap;
        if overflowed {
            self.inner.pop_front();
            self.dropped = self.dropped.saturating_add(1);
        }
        self.inner.push_back(token);
        overflowed
    }

    pub fn cap(&self) -> usize {
        self.cap
    }

    pub fn len(&self) -> usize {
        self.inner.len()
    }

    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    pub fn dropped_count(&self) -> u32 {
        self.dropped
    }

    pub fn drain(&mut self) -> Vec<String> {
        self.inner.drain(..).collect()
    }
}

pub(super) fn spawn_predicted_stt(
    cancel: &super::cancel::SessionCancel,
    whisper: WhisperHandle,
    audio: MonoF32At16k,
) -> PredictedRunner {
    let shared = Arc::new(PredictedSharedState::new());
    let shared_for_task = shared.clone();
    let task: JoinHandle<()> = tokio::spawn(cancel.wrap_unit(async move {
        let samples = audio.into_vec();
        let result = tokio::task::spawn_blocking(move || whisper.transcribe(&samples)).await;
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
    loop {
        {
            let guard = runner.shared.user_transcript.lock().await;
            if let Some(r) = guard.as_ref() {
                return r.clone();
            }
        }
        runner.shared.done.notified().await;
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

    pub async fn wait_finished(&self) {
        loop {
            if self.is_done() || self.is_cancelled() {
                return;
            }
            self.shared.finished.notified().await;
        }
    }
}

pub(super) fn spawn_predicted_llm(
    cancel: &super::cancel::SessionCancel,
    cfg: LlmConfig,
    messages: Vec<ChatMessage>,
    cap: u32,
) -> PredictedLlmRunner {
    let shared = Arc::new(PredictedLlmShared::new());
    let shared_for_task = shared.clone();
    let cap_usize = cap.max(1) as usize;
    let task: JoinHandle<()> = tokio::spawn(cancel.wrap_unit(async move {
        let mut rx = complete_stream_messages(&cfg, messages);
        loop {
            if shared_for_task.cancelled.load(Ordering::Relaxed) {
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
                    shared_for_task.chars_seen.fetch_add(n, Ordering::Relaxed);
                    let mut buf = shared_for_task.buffer.lock().await;
                    buf.push(delta);
                    if buf.len() > cap_usize {
                        let drop_n = buf.len() - cap_usize;
                        let _ = buf.drain(..drop_n).count();
                        shared_for_task
                            .dropped_tokens
                            .fetch_add(drop_n as u32, Ordering::Relaxed);
                        shared_for_task.overflowed.store(true, Ordering::Relaxed);
                    }
                    drop(buf);
                    shared_for_task.progress.notify_waiters();
                }
                Err(_) => {
                    break;
                }
            }
        }
        shared_for_task.done.store(true, Ordering::SeqCst);
        shared_for_task.finished.notify_waiters();
        shared_for_task.progress.notify_waiters();
    }));
    PredictedLlmRunner { task, shared, cap }
}

pub fn transcripts_materially_differ(predicted: &str, finalized: &str, ratio: f32) -> bool {
    let p = predicted.trim().to_lowercase();
    let f = finalized.trim().to_lowercase();
    if p.is_empty() || f.is_empty() {
        return p != f;
    }
    if p == f {
        return false;
    }
    let pset: std::collections::HashSet<char> = p.chars().filter(|c| !c.is_whitespace()).collect();
    let fset: std::collections::HashSet<char> = f.chars().filter(|c| !c.is_whitespace()).collect();
    let intersect = pset.intersection(&fset).count();
    let union = pset.union(&fset).count().max(1);
    let jaccard = intersect as f32 / union as f32;
    let threshold = (1.0 - ratio).clamp(0.0, 1.0);
    jaccard < threshold
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn predicted_token_buffer_drops_oldest_on_overflow() {
        let mut b = PredictedTokenBuffer::new(3);
        b.push("a".into());
        b.push("b".into());
        b.push("c".into());
        assert_eq!(b.dropped_count(), 0);
        b.push("d".into());
        b.push("e".into());
        assert_eq!(b.dropped_count(), 2);
        let out = b.drain();
        assert_eq!(out, vec!["c", "d", "e"]);
    }

    #[test]
    fn predicted_token_buffer_default_cap_at_least_one() {
        let mut b = PredictedTokenBuffer::new(0);
        b.push("only".into());
        assert_eq!(b.len(), 1);
        b.push("overflow".into());
        assert!(b.dropped_count() >= 1);
    }

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

    #[test]
    fn transcripts_match_when_equal() {
        assert!(!transcripts_materially_differ(
            "hello there",
            "hello there",
            0.5
        ));
    }

    #[test]
    fn transcripts_match_with_minor_punctuation() {
        assert!(!transcripts_materially_differ(
            "hello there",
            "hello there.",
            0.5
        ));
    }

    #[test]
    fn transcripts_diverge_when_completely_different() {
        assert!(transcripts_materially_differ(
            "tell me about cats",
            "what is the weather",
            0.5,
        ));
    }

    #[test]
    fn transcripts_one_empty_diverges() {
        assert!(transcripts_materially_differ("hello", "", 0.5));
        assert!(transcripts_materially_differ("", "hello", 0.5));
    }

    #[test]
    fn transcripts_both_empty_match() {
        assert!(!transcripts_materially_differ("", "", 0.5));
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
        let runner =
            spawn_predicted_llm(&SessionCancel::new(), cfg(base), vec![user_msg("hi")], 16);
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
        let runner = spawn_predicted_llm(&SessionCancel::new(), cfg(base), vec![user_msg("hi")], 4);
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
        let runner =
            spawn_predicted_llm(&SessionCancel::new(), cfg(base), vec![user_msg("hi")], 16);
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
        let runner =
            spawn_predicted_llm(&SessionCancel::new(), cfg(base), vec![user_msg("hi")], 16);
        runner.wait_finished().await;
        assert_eq!(runner.chars_seen(), "héllo".chars().count() as u32 + 2);
        runner.abort();
    }
}
