# 020-chat-structured-json

`response_format=json_object` with an embedded JSON schema. Asserts that
the assistant's content parses as JSON and matches the schema, plus that
two known string fields take their expected values.

## What it asserts

- Structural: 200 OK, one choice, assistant content is a syntactically valid
  JSON object.
- Schema match: object validates against the JSON schema embedded in the
  request (`request_body.response_format.schema`).
- Field-value pinning: `city == "Paris"`, `country == "France"`.

## Placeholder mode

Today the Rust ChatCompletionRequest doesn't yet model `response_format`
explicitly (serde silently drops the unknown field). The placeholder
assertion is therefore "the server accepts the request without 400/422".
Once grammar-constrained decode lands, the schema-match and field-value
assertions become canonical.

## Spec pins

- §11.4 — 020-chat-completions-* family
- §5.3 — chat/completions endpoint
- TODO marker for grammar/structured-output support: see
  `rust/src/oapi/chat.rs::ChatCompletionRequest` (no response_format field
  yet)
