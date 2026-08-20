# 020-chat-sampled-temp07

Sampled chat-completions with seeded RNG. Verifies that
`(seed, temperature, top_p)` is plumbed end-to-end and that the sampler
is deterministic.

## What it asserts

- Structural: route returns 200, single choice, non-empty assistant content,
  finish_reason in {stop, length}.
- Determinism: two back-to-back identical requests must produce byte-identical
  assistant strings (proves seed reaches the sampler).
- Once the real model is wired: the golden `first_token_id` and `final_content`
  in `ref_outputs` become exact-match assertions.

## Placeholder mode

Real Qwen3 weights are not required for the determinism check — the EchoEngine
fallback in `rust/src/oapi/chat.rs` echoes the prompt deterministically and so
trivially satisfies the double-run assertion. The fixture is therefore useful
even before the real engine lands; `ref_outputs.first_token_id` is `null` and
filled in once `NV_CHAT_QWEN3_MODEL_DIR` boot brings up NvEngineChat.

## Spec pins

- §11.4 — 020-chat-completions-* family
- §10.3 — determinism requirements (seeded sampler reproducibility)
- `rust/src/oapi/chat.rs::ChatGenerateRequest::seed`
