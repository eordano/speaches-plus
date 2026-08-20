import { test } from "node:test";
import assert from "node:assert/strict";
import { namedSseJson, openaiSseJson, sseFrames } from "../../src/sse.ts";
import type { SseFrame } from "../../src/sse.ts";
import type { AnthropicMessagesStreamEvent, ChatCompletionChunk } from "../../src/api-types.ts";
import {
  anthropicEvents,
  anthropicSseText,
  byteChunkedStream,
  chatChunks,
  chatSseText,
  streamOf,
} from "./fixtures.ts";

async function collect<T>(iter: AsyncIterable<T>): Promise<T[]> {
  const out: T[] = [];
  for await (const item of iter) out.push(item);
  return out;
}

test("sseFrames parses whole-frame chunks", async () => {
  const frames = await collect(sseFrames(streamOf(["data: a\n\n", "event: e\ndata: b\n\n"])));
  assert.deepEqual(frames, [
    { event: null, data: "a" },
    { event: "e", data: "b" },
  ]);
});

test("sseFrames survives frames split at every 3 bytes across multibyte chars", async () => {
  const parsed = await collect(openaiSseJson<ChatCompletionChunk>(byteChunkedStream(chatSseText, 3)));
  assert.deepEqual(parsed, chatChunks);
});

test("sseFrames handles CRLF line endings", async () => {
  const text = "event: e\r\ndata: {\"x\":1}\r\n\r\n";
  const frames = await collect(sseFrames(streamOf([text])));
  assert.deepEqual(frames, [{ event: "e", data: '{"x":1}' }]);
});

test("sseFrames joins multi-line data with newline and skips comment lines", async () => {
  const text = ": keepalive\ndata: line1\ndata: line2\n\n";
  const frames = await collect(sseFrames(streamOf([text])));
  assert.deepEqual(frames, [{ event: null, data: "line1\nline2" }]);
});

test("sseFrames drops an incomplete trailing frame at EOF", async () => {
  const frames = await collect(sseFrames(streamOf(["data: complete\n\ndata: incompl"])));
  assert.deepEqual(frames, [{ event: null, data: "complete" }]);
});

test("sseFrames yields event-only frames so named heartbeats are visible", async () => {
  const frames = await collect(sseFrames(streamOf(["event: ping\n\n"])));
  assert.deepEqual(frames, [{ event: "ping", data: "" }]);
});

test("openaiSseJson stops at [DONE] and surfaces reasoning_content deltas", async () => {
  const parsed = await collect(openaiSseJson<ChatCompletionChunk>(streamOf([chatSseText])));
  assert.deepEqual(parsed, chatChunks);
  const reasoning = parsed
    .map((chunk) => chunk.choices[0]?.delta.reasoning_content ?? "")
    .join("");
  assert.equal(reasoning, "pondering…");
  const text = parsed.map((chunk) => chunk.choices[0]?.delta.content ?? "").join("");
  assert.equal(text, "¡Hola mundo!");
  assert.equal(parsed[parsed.length - 1]?.choices[0]?.finish_reason, "stop");
});

test("openaiSseJson does not read past [DONE]", async () => {
  const text = chatSseText + "data: {malformed json that must never be parsed\n\n";
  const parsed = await collect(openaiSseJson<ChatCompletionChunk>(streamOf([text])));
  assert.deepEqual(parsed, chatChunks);
});

test("namedSseJson parses the anthropic named-event dialect", async () => {
  const parsed = await collect(
    namedSseJson<AnthropicMessagesStreamEvent>(byteChunkedStream(anthropicSseText, 7)),
  );
  assert.deepEqual(parsed, anthropicEvents);
  assert.equal(parsed[0]?.type, "message_start");
  assert.equal(parsed[parsed.length - 1]?.type, "message_stop");
});

test("early consumer break cancels the underlying reader", async () => {
  let cancelled = false;
  const body = new ReadableStream<Uint8Array>({
    pull(controller) {
      controller.enqueue(new TextEncoder().encode("data: a\n\ndata: b\n\n"));
    },
    cancel() {
      cancelled = true;
    },
  });
  const seen: SseFrame[] = [];
  for await (const frame of sseFrames(body)) {
    seen.push(frame);
    break;
  }
  assert.equal(seen.length, 1);
  assert.ok(cancelled, "breaking out of the loop must cancel the stream");
});
