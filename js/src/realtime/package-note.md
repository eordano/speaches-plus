# realtime module — integration notes for the core client

No package.json is required to run this module's tests: Node >= 23.6 strips
types natively and all imports use explicit `.ts` extensions.

```
node --test js/test/realtime/*.test.ts        # 25 tests, zero deps
```

What the workspace package.json should provide when it lands:

- `"type": "module"` (imports are ESM with `.ts` extensions; compile with
  `allowImportingTsExtensions` + `rewriteRelativeImportExtensions` or emit
  via a bundler).
- devDependency `@types/node` — the runtime sources under `js/src/realtime/`
  typecheck with `--lib es2022,dom` alone (browser-safe, no Node types); the
  test files under `js/test/realtime/` need `@types/node` for `node:test`,
  `node:http`, `Buffer`.
- a `test:realtime` script running the command above.

Swap points once `js/src/generated/` (ts-rs bindings) is vendored:

- `events.ts` re-states the `RealtimeOutboundEvent` variants by hand with
  narrower payload types; the variant list is gated against wire.rs by
  `js/test/realtime/events.test.ts`. When re-exporting generated types,
  keep the `event_id?: string` stamp: `EventSeq::stamp` adds it after serde
  serialization, so the generated union does not carry it.

Known server-side constraint: `SPEACHES_API_KEY` auth is header-only
(`auth_mw` in rust/src/main.rs), and the browser `WebSocket` constructor
cannot set headers — `RealtimeClient` sends `Authorization: Bearer` only in
Node (undici options object) or when `options.webSocket` is a ctor that
accepts `{ headers }`. Browser + API key + `/v1/realtime` WS requires a
server-side change (query-param or subprotocol auth) to work.
