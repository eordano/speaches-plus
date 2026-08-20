import { test } from "node:test";
import assert from "node:assert/strict";

import {
  RealtimeClient,
  buildRealtimeUrl,
  pcm16ToBase64,
} from "../../src/realtime/client.ts";
import type { RealtimeServerEvent } from "../../src/realtime/events.ts";
import { isKnownServerEventType } from "../../src/realtime/events.ts";
import {
  startMockRealtimeServer,
  type MockRealtimeServer,
} from "./mock_ws_server.ts";
import {
  DIARIZATION_EVENT,
  ERROR_EVENT,
  RESPONSE_LIFECYCLE,
  SESSION_CREATED,
  sessionUpdated,
} from "./fixtures.ts";

async function withServer(
  fn: (server: MockRealtimeServer) => Promise<void>,
): Promise<void> {
  const server = await startMockRealtimeServer();
  try {
    await fn(server);
  } finally {
    await server.close();
  }
}

function eventReceived(
  client: RealtimeClient,
  type: RealtimeServerEvent["type"],
  timeoutMs = 2000,
): Promise<RealtimeServerEvent> {
  return new Promise((resolve, reject) => {
    const timer = setTimeout(
      () => reject(new Error(`timed out waiting for ${type}`)),
      timeoutMs,
    );
    const off = client.on(type, (ev) => {
      clearTimeout(timer);
      off();
      resolve(ev);
    });
  });
}

test("buildRealtimeUrl maps scheme and query params to RealtimeQuery fields", () => {
  const url = buildRealtimeUrl("http://gpu:8000/", {
    model: "test-e2e",
    intent: "conversation",
    transcriptionModel: "whisper-x",
    voice: "af_heart",
    speechModel: "tts-y",
  });
  const u = new URL(url);
  assert.equal(u.protocol, "ws:");
  assert.equal(u.pathname, "/v1/realtime");
  assert.equal(u.searchParams.get("model"), "test-e2e");
  assert.equal(u.searchParams.get("intent"), "conversation");
  assert.equal(u.searchParams.get("transcription_model"), "whisper-x");
  assert.equal(u.searchParams.get("voice"), "af_heart");
  assert.equal(u.searchParams.get("speech_model"), "tts-y");
  assert.equal(
    new URL(buildRealtimeUrl("https://gpu:8443")).protocol,
    "wss:",
  );
});

test("pcm16ToBase64 encodes little-endian samples like the e2e driver", () => {
  const samples = new Int16Array([0, 1, -1, 256]);
  const b64 = pcm16ToBase64(samples);
  assert.equal(
    b64,
    Buffer.from(new Uint8Array(samples.buffer)).toString("base64"),
  );
  assert.equal(pcm16ToBase64(new Int16Array(2400)).length, 6400);
});

test("connect delivers session.created as the first event, typed", async () => {
  await withServer(async (server) => {
    const connP = server.waitForConnection();
    const client = await RealtimeClient.connect({
      baseUrl: server.url,
      model: "test-e2e",
    });
    const conn = await connP;
    assert.equal(conn.url.searchParams.get("model"), "test-e2e");

    const first = eventReceived(client, "session.created");
    conn.send(SESSION_CREATED);
    const ev = await first;
    assert.equal(ev.type, "session.created");
    if (ev.type === "session.created") {
      assert.equal(ev.session.id, "sess_mock_001");
      assert.equal(ev.session.object, "realtime.session");
      assert.equal(ev.session.type, "realtime");
      assert.equal(typeof ev.session.audio?.input, "object");
    }
    assert.equal(ev.event_id, "evt_000000000000000000000000");
    client.close();
  });
});

test("send serializes typed client events; session.update round-trips", async () => {
  await withServer(async (server) => {
    const connP = server.waitForConnection();
    const client = await RealtimeClient.connect({ baseUrl: server.url });
    const conn = await connP;
    conn.send(SESSION_CREATED);

    client.updateSession({ voice: "af_heart", instructions: "Test instructions." });
    const raw = await conn.waitForMessage((t) => t.includes("session.update"));
    assert.deepEqual(JSON.parse(raw), {
      type: "session.update",
      session: { voice: "af_heart", instructions: "Test instructions." },
    });

    const updated = eventReceived(client, "session.updated");
    conn.send(sessionUpdated({ instructions: "Test instructions." }));
    const ev = await updated;
    if (ev.type === "session.updated") {
      assert.equal(ev.session.instructions, "Test instructions.");
    }
    client.close();
  });
});

test("item/response/audio helpers emit the exact inbound event shapes", async () => {
  await withServer(async (server) => {
    const connP = server.waitForConnection();
    const client = await RealtimeClient.connect({ baseUrl: server.url });
    const conn = await connP;

    client.createUserText("Hello world", "test_item_001");
    client.createResponse();
    client.appendAudio(new Int16Array(2400));
    client.commitAudio();
    client.clearAudio();
    client.deleteItem("test_item_001");
    client.truncateItem("test_item_001", 0, 500);
    client.cancelResponse();

    await conn.waitForMessage((t) => t.includes("response.cancel"));
    const sent = conn.received.map((t) => JSON.parse(t));
    assert.deepEqual(sent[0], {
      type: "conversation.item.create",
      item: {
        id: "test_item_001",
        type: "message",
        role: "user",
        content: [{ type: "input_text", text: "Hello world" }],
      },
    });
    assert.deepEqual(sent[1], { type: "response.create" });
    assert.equal(sent[2].type, "input_audio_buffer.append");
    assert.equal(typeof sent[2].audio, "string");
    assert.deepEqual(sent[3], { type: "input_audio_buffer.commit" });
    assert.deepEqual(sent[4], { type: "input_audio_buffer.clear" });
    assert.deepEqual(sent[5], {
      type: "conversation.item.delete",
      item_id: "test_item_001",
    });
    assert.deepEqual(sent[6], {
      type: "conversation.item.truncate",
      item_id: "test_item_001",
      content_index: 0,
      audio_end_ms: 500,
    });
    assert.deepEqual(sent[7], { type: "response.cancel" });
    client.close();
  });
});

test("async iterator replays the full response lifecycle in order", async () => {
  await withServer(async (server) => {
    const connP = server.waitForConnection();
    const client = await RealtimeClient.connect({ baseUrl: server.url });
    const conn = await connP;

    conn.send(SESSION_CREATED);
    for (const ev of RESPONSE_LIFECYCLE) conn.send(ev);

    const seen: RealtimeServerEvent[] = [];
    for await (const ev of client) {
      seen.push(ev);
      assert.ok(
        isKnownServerEventType(ev.type),
        `unknown event type from fixture: ${ev.type}`,
      );
      if (ev.type === "response.done") {
        assert.equal(ev.response.status, "completed");
        assert.equal(ev.response.audio_end_ms, 1234);
        break;
      }
    }
    assert.deepEqual(
      seen.map((e) => e.type),
      ["session.created", ...RESPONSE_LIFECYCLE.map((e) => e.type)],
    );
    const delta = seen.find(
      (e) => e.type === "response.output_audio_transcript.delta",
    );
    assert.ok(delta && "delta" in delta && delta.delta === "Hello there.");
    client.close();
  });
});

test("error payload arrives with type/code/message; diarization event is typed", async () => {
  await withServer(async (server) => {
    const connP = server.waitForConnection();
    const client = await RealtimeClient.connect({ baseUrl: server.url });
    const conn = await connP;

    const errP = eventReceived(client, "error");
    const diarP = eventReceived(client, "conversation.item.diarization");
    conn.send(ERROR_EVENT);
    conn.send(DIARIZATION_EVENT);

    const err = await errP;
    if (err.type === "error") {
      assert.equal(err.error.type, "invalid_request_error");
      assert.equal(err.error.code, "unknown_event_type");
      assert.match(err.error.message, /totally\.bogus\.event/);
    }
    const diar = await diarP;
    if (diar.type === "conversation.item.diarization") {
      assert.equal(diar.item_id, "item_user_001");
      assert.equal(diar.segments.length, 1);
    }
    client.close();
  });
});

test("malformed frames go to onMalformedFrame, never to event handlers", async () => {
  await withServer(async (server) => {
    const malformed: string[] = [];
    const connP = server.waitForConnection();
    const client = await RealtimeClient.connect({
      baseUrl: server.url,
      onMalformedFrame: (raw) => malformed.push(raw),
    });
    const conn = await connP;

    const all: RealtimeServerEvent[] = [];
    client.onAny((ev) => all.push(ev));
    conn.sendRaw("this is not json {{{");
    conn.sendRaw('{"no_type_field": true}');
    conn.send(SESSION_CREATED);
    await eventReceived(client, "session.created");

    assert.deepEqual(malformed, [
      "this is not json {{{",
      '{"no_type_field": true}',
    ]);
    assert.deepEqual(
      all.map((e) => e.type),
      ["session.created"],
    );
    client.close();
  });
});

test("close() ends iterators cleanly and send() after close throws", async () => {
  await withServer(async (server) => {
    const client = await RealtimeClient.connect({ baseUrl: server.url });
    const conn = await server.waitForConnection();
    conn.send(SESSION_CREATED);

    const iterated = (async () => {
      const types: string[] = [];
      for await (const ev of client) types.push(ev.type);
      return types;
    })();
    await eventReceived(client, "session.created");
    client.close();

    assert.deepEqual(await iterated, ["session.created"]);
    assert.equal(client.state, "closed");
    assert.throws(() => client.commitAudio(), /cannot send in state 'closed'/);
  });
});

test("AbortSignal tears the connection down and rejects pending iteration", async () => {
  await withServer(async (server) => {
    const ac = new AbortController();
    const client = await RealtimeClient.connect({
      baseUrl: server.url,
      signal: ac.signal,
    });
    const conn = await server.waitForConnection();
    const pending = client.events();
    conn.send(SESSION_CREATED);
    const drained = await pending.next();
    assert.equal((drained.value as RealtimeServerEvent).type, "session.created");
    const nextP = pending.next();
    ac.abort();

    await assert.rejects(nextP, (err: Error) => err.name === "AbortError");
    assert.equal(client.state, "closed");
  });
});

test("connect() rejects when nothing is listening", async () => {
  await assert.rejects(
    RealtimeClient.connect({
      baseUrl: "ws://127.0.0.1:9",
      reconnect: true,
    }),
    /websocket connection failed/,
  );
});
