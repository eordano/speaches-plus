# 020-chat-completions-* family

Conformance fixtures for `/v1/chat/completions`. Surfaces request-shape,
response-shape, streaming-shape, and seeded-sampler determinism contracts
for the chat endpoint regardless of which engine (EchoEngine fallback,
NvEngineChat with real Qwen3 weights, future grammar-constrained decoder)
is serving.

## Fixtures (4)

| ID                            | Coverage                                                                  |
| ----------------------------- | ------------------------------------------------------------------------- |
| `020-chat-greedy-basic`       | temp=0 single-turn; case-insensitive regex match on assistant content     |
| `020-chat-sampled-temp07`     | temp=0.7 + seed=1337; deterministic double-run + golden first_token_id    |
| `020-chat-structured-json`    | `response_format=json_object` + embedded JSON schema; field-pin assertion |
| `020-chat-tool-call-stub`     | `tools=[get_weather]` + streaming SSE; structural placeholder today       |

## "Passing" in placeholder mode

The chat endpoint is wired today (see `rust/src/oapi/chat.rs`) with an
EchoEngine fallback when no real model is loaded. Placeholder-mode passing
means:

- Route returns 200 (not 404/405/422) for each fixture's request body.
- Response deserializes as `ChatCompletionResponse` (`object="chat.completion"`,
  one choice, non-empty assistant content, finish_reason in {stop, length})
  or a valid streaming SSE sequence (final `data: [DONE]`).
- Each fixture is annotated `skip_when_no_model: true`, so the regex / JSON
  schema / tool_call delta assertions are skipped in CI runs that don't have
  weights loaded; they turn on automatically once the runner is told a model
  is available (env: `NV_CHAT_QWEN3_MODEL_DIR` set + non-empty).

## "Passing" once real models land

| Fixture                       | Canonical assertion                                                   |
| ----------------------------- | --------------------------------------------------------------------- |
| `020-chat-greedy-basic`       | Assistant content matches the case-insensitive `(?i).*paris.*` regex  |
| `020-chat-sampled-temp07`     | Byte-identical content across two seeded runs + exact `first_token_id`|
| `020-chat-structured-json`    | Content parses as JSON, validates schema, `city`/`country` pinned     |
| `020-chat-tool-call-stub`     | SSE delta sequence with tool_call start + args + tool_calls finish    |

## Discovery

Each fixture directory has a `fixture.json` (not `input.jsonl` / `expected.jsonl`),
which means the legacy Realtime conformance runners
(`rust/tests/conformance.rs::canonical_passes_every_conformance_fixture`,
`go/internal/realtime/conformance_test.go::TestConformanceCorpus`,
`conformance/runner/run_fixture.py::cmd_validate_all`) intentionally **skip**
these directories — they only act on dirs that contain `expected.jsonl`.

The 020/030/040 fixtures are discovered by the family-aware runner extension
`conformance/runner/run_endpoint_fixture.py` (added alongside this family),
which globs `fixture.json` files under `conformance/fixtures/`.
