use anyhow::{anyhow, Context, Result};
use futures_util::StreamExt;
use serde::Deserialize;
use serde_json::json;
use tokio::sync::mpsc;
use tracing::debug;

#[derive(Clone, Debug)]
pub struct LlmConfig {
    pub base_url: String,
    pub api_key: Option<String>,
    pub model: String,
}

impl LlmConfig {
    pub fn from_env() -> Option<Self> {
        let base_url = std::env::var(crate::defaults::env::CHAT_COMPLETION_BASE_URL).ok()?;
        let api_key = std::env::var(crate::defaults::env::CHAT_COMPLETION_API_KEY).ok();
        let model = std::env::var(crate::defaults::env::DEFAULT_REALTIME_CONVERSATION_MODEL)
            .unwrap_or_else(|_| "default".to_string());
        Some(Self {
            base_url,
            api_key,
            model,
        })
    }
}

#[allow(dead_code)]
pub async fn complete(
    cfg: &LlmConfig,
    instructions: Option<&str>,
    user_text: &str,
) -> Result<String> {
    let mut rx = complete_stream(cfg, instructions, user_text);
    let mut text = String::new();
    let mut last_err: Option<anyhow::Error> = None;
    while let Some(item) = rx.recv().await {
        match item {
            Ok(delta) => text.push_str(&delta),
            Err(err) => last_err = Some(err),
        }
    }
    if let Some(err) = last_err {
        return Err(err);
    }
    if text.is_empty() {
        return Err(anyhow!("LLM stream ended with no content"));
    }
    Ok(text)
}

#[allow(dead_code)]
pub fn complete_stream(
    cfg: &LlmConfig,
    instructions: Option<&str>,
    user_text: &str,
) -> mpsc::Receiver<Result<String>> {
    let mut messages = Vec::new();
    if let Some(sys) = instructions.filter(|s| !s.is_empty()) {
        messages.push(ChatMessage {
            role: "system".into(),
            content: sys.to_string(),
        });
    }
    messages.push(ChatMessage {
        role: "user".into(),
        content: user_text.to_string(),
    });
    complete_stream_messages(cfg, messages, None)
}

#[derive(Clone, Debug)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
}

pub fn complete_stream_messages(
    cfg: &LlmConfig,
    messages: Vec<ChatMessage>,
    trajectory_tag: Option<String>,
) -> mpsc::Receiver<Result<String>> {
    let (tx, rx) = mpsc::channel(64);
    let cfg = cfg.clone();
    tokio::spawn(async move {
        if let Err(err) = stream_messages_into(cfg, messages, trajectory_tag, tx.clone()).await {
            let _ = tx.send(Err(err)).await;
        }
    });
    rx
}

async fn stream_messages_into(
    cfg: LlmConfig,
    messages: Vec<ChatMessage>,
    trajectory_tag: Option<String>,
    tx: mpsc::Sender<Result<String>>,
) -> Result<()> {
    let url = format!("{}/chat/completions", cfg.base_url.trim_end_matches('/'));
    let messages_json: Vec<serde_json::Value> = messages
        .iter()
        .map(|m| json!({"role": m.role, "content": m.content}))
        .collect();
    let body = json!({
        "model": cfg.model,
        "messages": messages_json,
        "stream": true,
    });
    let mut req = reqwest::Client::new()
        .post(&url)
        .json(&body)
        .header("Accept", "text/event-stream");
    if let Some(key) = &cfg.api_key {
        req = req.header("Authorization", format!("Bearer {key}"));
    }
    if let Some(tag) = trajectory_tag
        .as_deref()
        .map(str::trim)
        .filter(|t| !t.is_empty())
    {
        req = req.header(crate::oapi::trajectory::TAG_HEADER, tag);
    }
    let resp = req.send().await.context("LLM request send")?;
    if !resp.status().is_success() {
        let status = resp.status();
        let body = resp.text().await.unwrap_or_default();
        anyhow::bail!("LLM upstream {status}: {body}");
    }

    let mut stream = resp.bytes_stream();
    let mut sse_buf = String::new();
    let mut emitted_any = false;
    while let Some(chunk) = stream.next().await {
        let chunk = chunk.context("stream chunk")?;
        let s = std::str::from_utf8(&chunk).context("non-utf8 SSE")?;
        sse_buf.push_str(s);
        while let Some(end) = sse_buf.find("\n\n") {
            let event: String = sse_buf.drain(..end + 2).collect();
            for line in event.lines() {
                let Some(payload) = line.strip_prefix("data: ") else {
                    continue;
                };
                if payload == "[DONE]" {
                    if !emitted_any {
                        return Err(anyhow!("LLM stream ended with no content"));
                    }
                    return Ok(());
                }
                let parsed: SseChunk = match serde_json::from_str(payload) {
                    Ok(v) => v,
                    Err(err) => {
                        debug!(error = %err, line = %payload, "skip unparseable SSE chunk");
                        continue;
                    }
                };
                for choice in parsed.choices {
                    if let Some(content) = choice.delta.and_then(|d| d.content) {
                        emitted_any = true;
                        if tx.send(Ok(content)).await.is_err() {
                            return Ok(());
                        }
                    }
                    if let Some(content) = choice.message.and_then(|m| m.content) {
                        emitted_any = true;
                        if tx.send(Ok(content)).await.is_err() {
                            return Ok(());
                        }
                    }
                }
            }
        }
    }
    if !emitted_any {
        return Err(anyhow!("LLM stream ended with no content"));
    }
    Ok(())
}

pub struct SentenceChunker {
    buf: String,
}

impl Default for SentenceChunker {
    fn default() -> Self {
        Self::new()
    }
}

impl SentenceChunker {
    pub fn new() -> Self {
        Self { buf: String::new() }
    }

    pub fn feed(&mut self, delta: &str) -> Vec<String> {
        self.buf.push_str(delta);
        let mut out = Vec::new();
        loop {
            let cut = self.buf.find(['.', '!', '?', '\n']);
            let Some(idx) = cut else { break };
            let end = idx + self.buf[idx..].chars().next().unwrap().len_utf8();
            let sentence: String = self.buf[..end].trim().to_string();
            self.buf = self.buf[end..].to_string();
            if !sentence.is_empty() {
                out.push(sentence);
            }
        }
        out
    }

    pub fn flush(self) -> Option<String> {
        let trimmed = self.buf.trim();
        if trimmed.is_empty() {
            None
        } else {
            Some(trimmed.to_string())
        }
    }

    pub fn take_flush(&mut self) -> Option<String> {
        let trimmed = self.buf.trim().to_string();
        self.buf.clear();
        if trimmed.is_empty() {
            None
        } else {
            Some(trimmed)
        }
    }
}

#[derive(Deserialize)]
struct SseChunk {
    #[serde(default)]
    choices: Vec<SseChoice>,
}

#[derive(Deserialize)]
struct SseChoice {
    #[serde(default)]
    delta: Option<SseDelta>,
    #[serde(default)]
    message: Option<SseMessage>,
}

#[derive(Deserialize)]
struct SseDelta {
    #[serde(default)]
    content: Option<String>,
}

#[derive(Deserialize)]
struct SseMessage {
    #[serde(default)]
    content: Option<String>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::body::Body;
    use axum::http::{header, StatusCode};
    use axum::response::Response;
    use axum::routing::post;
    use axum::Router;
    use std::sync::Arc;
    use std::sync::Mutex as StdMutex;
    use std::time::Duration;
    use tokio::net::TcpListener;

    enum MockBehavior {
        StreamTokens(Vec<String>),
        Status(u16, String),
        EmptyStream,
    }

    async fn spawn_mock(behavior: MockBehavior) -> String {
        let behavior = Arc::new(StdMutex::new(Some(behavior)));
        let app = Router::new().route(
            "/v1/chat/completions",
            post(move || {
                let behavior = behavior.clone();
                async move {
                    let b = behavior.lock().unwrap().take().expect("one-shot mock");
                    match b {
                        MockBehavior::Status(code, body) => Response::builder()
                            .status(StatusCode::from_u16(code).unwrap())
                            .body(Body::from(body))
                            .unwrap(),
                        MockBehavior::EmptyStream => Response::builder()
                            .status(200)
                            .header(header::CONTENT_TYPE, "text/event-stream")
                            .body(Body::from("data: [DONE]\n\n"))
                            .unwrap(),
                        MockBehavior::StreamTokens(tokens) => {
                            let mut sse = String::new();
                            for tok in tokens {
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
                    }
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

    #[tokio::test]
    async fn happy_path_assembles_streamed_tokens() {
        let base = spawn_mock(MockBehavior::StreamTokens(vec![
            "hello".into(),
            " world".into(),
        ]))
        .await;
        let text = complete(&cfg(base), None, "ping").await.unwrap();
        assert_eq!(text, "hello world");
    }

    #[tokio::test]
    async fn upstream_500_surfaces_as_error() {
        let base = spawn_mock(MockBehavior::Status(500, "boom".into())).await;
        let err = complete(&cfg(base), None, "ping").await.unwrap_err();
        let s = format!("{err:#}");
        assert!(s.contains("500"), "expected 500 in {s:?}");
        assert!(s.contains("boom"), "expected upstream body in {s:?}");
    }

    #[tokio::test]
    async fn upstream_429_surfaces_as_error() {
        let base = spawn_mock(MockBehavior::Status(429, "slow down".into())).await;
        let err = complete(&cfg(base), None, "ping").await.unwrap_err();
        assert!(format!("{err:#}").contains("429"));
    }

    #[tokio::test]
    async fn empty_stream_returns_error_not_empty_string() {
        let base = spawn_mock(MockBehavior::EmptyStream).await;
        let err = complete(&cfg(base), None, "ping").await.unwrap_err();
        assert!(format!("{err:#}").to_lowercase().contains("no content"));
    }

    #[tokio::test]
    async fn no_upstream_at_all_returns_error_quickly() {
        let cfg = cfg("http://127.0.0.1:1/v1".into());
        let result =
            tokio::time::timeout(Duration::from_secs(5), complete(&cfg, None, "ping")).await;
        let result = result.expect("should not hang past 5s");
        assert!(result.is_err());
    }

    #[test]
    fn sentence_chunker_splits_on_terminators_and_newlines_and_flushes_the_rest() {
        let mut c = SentenceChunker::new();
        assert!(c.feed("Hello").is_empty());
        let out = c.feed(" world. How are you?");
        assert_eq!(out, vec!["Hello world.", "How are you?"]);
        assert_eq!(c.flush(), None);
        {
            let mut c = SentenceChunker::new();
            c.feed("incomplete");
            assert_eq!(c.flush(), Some("incomplete".to_string()));
        }
        {
            let mut c = SentenceChunker::new();
            let out = c.feed("line one\nline two");
            assert_eq!(out, vec!["line one"]);
            assert_eq!(c.flush(), Some("line two".to_string()));
        }
    }

    #[tokio::test]
    async fn trajectory_tag_rides_as_the_tag_header_and_none_sends_no_header() {
        let captured: Arc<StdMutex<Vec<Option<String>>>> = Arc::new(StdMutex::new(Vec::new()));
        let captured_for_route = captured.clone();
        let app = Router::new().route(
            "/v1/chat/completions",
            post(move |headers: axum::http::HeaderMap| {
                let captured = captured_for_route.clone();
                async move {
                    captured.lock().unwrap().push(
                        headers
                            .get(crate::oapi::trajectory::TAG_HEADER)
                            .and_then(|v| v.to_str().ok())
                            .map(String::from),
                    );
                    Response::builder()
                        .status(200)
                        .header(header::CONTENT_TYPE, "text/event-stream")
                        .body(Body::from(
                            "data: {\"choices\":[{\"delta\":{\"content\":\"ok\"}}]}\n\ndata: [DONE]\n\n",
                        ))
                        .unwrap()
                }
            }),
        );
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move {
            axum::serve(listener, app).await.unwrap();
        });
        let base = format!("http://{addr}/v1");
        let user = vec![ChatMessage {
            role: "user".into(),
            content: "hi".into(),
        }];
        let mut rx =
            complete_stream_messages(&cfg(base.clone()), user.clone(), Some("resp_tag_1".into()));
        while rx.recv().await.is_some() {}
        let mut rx = complete_stream_messages(&cfg(base), user, None);
        while rx.recv().await.is_some() {}
        let seen = captured.lock().unwrap().clone();
        assert_eq!(
            seen,
            vec![Some("resp_tag_1".to_string()), None],
            "the tagged call carries x-trajectory-tag and the untagged call omits it"
        );
    }

    #[tokio::test]
    async fn instructions_included_as_system_message() {
        use std::sync::Mutex as StdMutex;
        let captured: Arc<StdMutex<Option<serde_json::Value>>> = Arc::new(StdMutex::new(None));
        let captured_for_route = captured.clone();
        let app = Router::new().route(
            "/v1/chat/completions",
            post(move |body: axum::Json<serde_json::Value>| {
                let captured = captured_for_route.clone();
                async move {
                    *captured.lock().unwrap() = Some(body.0);
                    Response::builder()
                        .status(200)
                        .header(header::CONTENT_TYPE, "text/event-stream")
                        .body(Body::from(
                            "data: {\"choices\":[{\"delta\":{\"content\":\"ok\"}}]}\n\ndata: [DONE]\n\n",
                        ))
                        .unwrap()
                }
            }),
        );
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move {
            axum::serve(listener, app).await.unwrap();
        });
        let base = format!("http://{addr}/v1");
        let _ = complete(&cfg(base), Some("you are a pirate"), "hi")
            .await
            .unwrap();
        let body = captured.lock().unwrap().take().expect("body captured");
        let messages = body.get("messages").and_then(|m| m.as_array()).unwrap();
        assert_eq!(messages.len(), 2);
        assert_eq!(messages[0]["role"], "system");
        assert_eq!(messages[0]["content"], "you are a pirate");
        assert_eq!(messages[1]["role"], "user");
    }
}
