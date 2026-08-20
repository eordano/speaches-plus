# 020-chat-greedy-basic

Greedy (temperature=0) chat-completions roundtrip.

PRD pins: §11.4 (conformance corpus), §5.3 (/v1/chat/completions endpoint).

## What it asserts

- Route exists and returns 200 (already passes today via the EchoEngine
  fallback in `rust/src/oapi/chat.rs::handle_chat_completions`).
- Response shape matches OpenAI: `object="chat.completion"`, exactly one
  choice with `finish_reason in {stop,length}`, message has role=assistant
  and non-empty content, `usage` is present with non-negative integers.

## Placeholder vs real-model mode

- **Placeholder** (no model loaded, default in CI without weights):
  structural assertion only. The fixture is marked `skip_when_no_model: true`
  so the runner does not require the regex match.
- **Real model** (`NV_CHAT_QWEN3_MODEL_DIR` set, NvEngineChat wired):
  `expected_response.choices[0].message.content_regex` becomes the canonical
  assertion — the model must produce a string matching `(?i).*paris.*`.

## Spec pins

- §11.4 — fixture family 020-chat-completions-*
- `rust/src/oapi/chat.rs::ChatCompletionRequest` / `ChatCompletionResponse`
- `rust/src/oapi/chat_engine.rs::NvEngineChat` (real engine) and
  `EchoEngine` fallback (structural-only path).
