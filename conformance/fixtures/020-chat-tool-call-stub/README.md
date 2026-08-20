# 020-chat-tool-call-stub

Tool-call request with a `get_weather` spec.

**Tool calling IS implemented** (verified 2026-08-04). This README previously
said it was not, and that the `tools` field was "silently ignored by serde" —
both false, and the likely source of the same stale claim in `docs/book/01.4-STATUS.md`.
The real implementation: `rust/src/oapi/tool_parse.rs` parses `<tool_call>`
blocks into OpenAI `ToolCall`s on both the streaming-SSE and full-response
paths and sets `finish_reason="tool_calls"`; `tool_args_grammar()`
(`oapi/chat.rs:331`) derives a schema grammar to force valid JSON arguments on
a forced tool choice; and `rust/tests/tool_calling_e2e.rs` covers six
scenarios (auto choice, tool-result follow-up, no-tool-needed, forced choice,
streaming SSE, multi-model routing) with **no** `#[ignore]` and **no** cuda
gate, so they run in the plain CPU suite.

The fixture name still says "stub" for path stability; the behaviour it
describes is live. Rename is deferred to avoid breaking the runner's fixture
discovery.

## What it asserts

- Placeholder mode: the request body is accepted (no 400/422 on `tools` or
  `tool_choice`); response is a syntactically valid chat-completion stream.
- Once tool calling lands: the streamed delta sequence matches
  `expected_response.expected_delta_sequence` (a tool_call start, one or more
  argument deltas, a finish with `finish_reason="tool_calls"`, then `[DONE]`),
  and the assembled tool_call arguments validate against the embedded JSON
  schema and pin `city == "Tokyo"`.

## Why this fixture exists today

Originally to lock the shape of the tool-call surface **before** implementing
it. That has since happened, so the fixture now serves as the regression gate
it was designed to become. `skip_when_no_feature: "tool_calling"` below should
no longer skip on capability grounds — if it still does, the feature flag is
what is stale, not the implementation.

## Spec pins

- §11.4 — 020-chat-completions-* family
- Skip flags: `skip_when_no_model: true`, `skip_when_no_feature: "tool_calling"`.
