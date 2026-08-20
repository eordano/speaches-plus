import { test } from "node:test";
import assert from "node:assert/strict";

import {
  LEGACY_SERVER_EVENT_TYPES,
  SERVER_EVENT_TYPES,
  V2_NOOP_EVENT_TYPES,
  isKnownServerEventType,
} from "../../src/realtime/events.ts";

// Captured verbatim from the serde rename tags of OutboundEvent in
// rust/src/realtime/wire.rs (the type_name_for_each_variant_unique test).
const WIRE_RS_OUTBOUND_EVENT_NAMES = [
  "session.created",
  "session.updated",
  "session.done",
  "input_audio_buffer.speech_started",
  "input_audio_buffer.speech_stopped",
  "input_audio_buffer.committed",
  "input_audio_buffer.cleared",
  "input_audio_buffer.partial_transcription",
  "conversation.item.added",
  "conversation.item.deleted",
  "conversation.item.truncated",
  "conversation.item.assistant_truncated",
  "conversation.item.input_audio_transcription.completed",
  "conversation.item.input_audio_transcription.delta",
  "conversation.item.input_audio_transcription.failed",
  "conversation.item.done",
  "conversation.item.retrieved",
  "response.created",
  "response.output_item.added",
  "response.output_item.done",
  "response.content_part.added",
  "response.content_part.done",
  "response.output_audio_transcript.delta",
  "response.output_audio_transcript.done",
  "response.output_audio.delta",
  "response.output_audio.done",
  "response.output_text.delta",
  "response.output_text.done",
  "response.function_call_arguments.delta",
  "response.function_call_arguments.done",
  "response.tool_progress",
  "response.cancelled",
  "response.done",
  "output_audio_buffer.cleared",
  "output_audio_buffer.started",
  "output_audio_buffer.stopped",
  "rate_limits.updated",
  "error",
  "conversation.item.diarization",
];

// The EXPECTED_SERVER_EVENTS table in client/e2e_browser_events.mjs -- which
// omits conversation.item.diarization (it only whitelists it in its
// unknown-name filter). The typed union follows wire.rs, not the e2e table.
const E2E_TABLE_EVENT_NAMES = WIRE_RS_OUTBOUND_EVENT_NAMES.filter(
  (n) => n !== "conversation.item.diarization",
);

test("SERVER_EVENT_TYPES matches the wire.rs OutboundEvent list exactly", () => {
  assert.deepEqual([...SERVER_EVENT_TYPES], WIRE_RS_OUTBOUND_EVENT_NAMES);
});

test("wire.rs has 39 outbound events; the e2e table has 38 (diarization drift)", () => {
  assert.equal(SERVER_EVENT_TYPES.length, 39);
  assert.equal(E2E_TABLE_EVENT_NAMES.length, 38);
  assert.ok(isKnownServerEventType("conversation.item.diarization"));
});

test("no duplicate server event names", () => {
  assert.equal(new Set(SERVER_EVENT_TYPES).size, SERVER_EVENT_TYPES.length);
});

test("legacy names are disjoint from canonical names and not marked known", () => {
  for (const legacy of LEGACY_SERVER_EVENT_TYPES) {
    assert.ok(!isKnownServerEventType(legacy), `legacy leaked: ${legacy}`);
  }
});

test("v2 noop client events match v2_compat.rs KNOWN_V2_NOOP_EVENTS", () => {
  assert.deepEqual(
    [...V2_NOOP_EVENT_TYPES],
    [
      "output_audio_buffer.clear",
      "output_audio_buffer.append",
      "input_audio_buffer.dtmf.received",
      "transcription_session.update",
      "response.cancel_audio",
    ],
  );
});

test("isKnownServerEventType rejects arbitrary strings", () => {
  assert.ok(!isKnownServerEventType("totally.bogus.event"));
  assert.ok(!isKnownServerEventType(""));
  assert.ok(isKnownServerEventType("response.done"));
});
