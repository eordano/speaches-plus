import { after, before, test } from "node:test";
import assert from "node:assert/strict";
import { NurClient, NurError } from "../../src/index.ts";
import { fetchRealtimeCapabilities } from "../../src/realtime/index.ts";
import { bootServer, sineWavBlob, type E2eServer } from "./harness.ts";

// Model-free e2e: the real binary booted with empty model/profile dirs and no
// NV_CHAT_MODEL_DIR(S). Chat/STT/TTS/embeddings answer with typed refusals; the
// suite proves route wiring, envelope mapping and the model-free surfaces.
// Echo-engine chat streaming is NOT reachable from the shipped binary
// (EchoEngine is test-only in rust/); those flows stay covered by unit tests
// until a serve-echo entrypoint exists server-side.

let server: E2eServer;
let client: NurClient;

before(async () => {
  server = await bootServer();
  client = new NurClient({ baseURL: server.baseURL });
});

after(async () => {
  await server.stop();
});

test("health and version answer", async () => {
  assert.equal(await client.health(), "ok");
  const v = await client.version();
  assert.equal(typeof v.version, "string");
  assert.ok(v.version.length > 0);
});

test("models.list returns an empty typed list on a model-free boot", async () => {
  const models = await client.models.list();
  assert.equal(models.object, "list");
  assert.deepEqual(models.data, []);
});

test("chat completion (non-stream) refuses with 404 when chat is unconfigured", async () => {
  await assert.rejects(
    client.chat.completions.create({ model: "m", messages: [{ role: "user", content: "hi" }] }),
    (err: unknown) => {
      assert.ok(err instanceof NurError);
      assert.equal(err.status, 404);
      return true;
    },
  );
});

test("chat completion (stream) refuses with 404 before any SSE frame", async () => {
  await assert.rejects(
    client.chat.completions.create({ model: "m", messages: [{ role: "user", content: "hi" }], stream: true }),
    (err: unknown) => {
      assert.ok(err instanceof NurError);
      assert.equal(err.status, 404);
      return true;
    },
  );
});

test("messages (non-stream) refuses with 404 when chat is unconfigured", async () => {
  await assert.rejects(
    client.messages.create({ model: "m", max_tokens: 16, messages: [{ role: "user", content: "hi" }] }),
    (err: unknown) => {
      assert.ok(err instanceof NurError);
      assert.equal(err.status, 404);
      return true;
    },
  );
});

test("transcription maps the OpenAI error envelope onto NurError", async () => {
  await assert.rejects(
    client.audio.transcriptions.create({ file: sineWavBlob(), model: "whisper-1" }),
    (err: unknown) => {
      assert.ok(err instanceof NurError);
      assert.equal(err.status, 503);
      assert.equal(err.code, "stt_unavailable");
      assert.equal(err.errorType, "service_unavailable_error");
      return true;
    },
  );
});

test("speech maps tts_not_configured with its param", async () => {
  await assert.rejects(client.audio.speech.create({ input: "hi", response_format: "wav" }), (err: unknown) => {
    assert.ok(err instanceof NurError);
    assert.equal(err.status, 503);
    assert.equal(err.code, "tts_not_configured");
    assert.equal(err.param, "model");
    return true;
  });
});

test("embeddings map model_not_loaded", async () => {
  await assert.rejects(client.embeddings.create({ input: "hi" }), (err: unknown) => {
    assert.ok(err instanceof NurError);
    assert.equal(err.status, 503);
    assert.equal(err.code, "model_not_loaded");
    return true;
  });
});

test("voice-profiles: list is empty, create refuses without a speaker encoder, get/delete 404 typed", async () => {
  const listed = await client.voiceProfiles.list();
  assert.equal(listed.object, "list");
  assert.deepEqual(listed.data, []);

  await assert.rejects(client.voiceProfiles.create({ name: "e2e-probe", file: sineWavBlob() }), (err: unknown) => {
    assert.ok(err instanceof NurError);
    assert.equal(err.status, 503);
    assert.equal(err.code, "no_speaker_encoder");
    return true;
  });

  await assert.rejects(client.voiceProfiles.retrieve("e2e-probe"), (err: unknown) => {
    assert.ok(err instanceof NurError);
    assert.equal(err.status, 404);
    assert.equal(err.code, "voice_profile_not_found");
    assert.equal(err.param, "name");
    return true;
  });

  await assert.rejects(client.voiceProfiles.delete("e2e-probe"), (err: unknown) => {
    assert.ok(err instanceof NurError);
    assert.equal(err.status, 404);
    assert.equal(err.code, "voice_profile_not_found");
    return true;
  });
});

test("realtime capabilities answer with the feature document", async () => {
  const caps = await fetchRealtimeCapabilities({ baseUrl: server.baseURL });
  assert.equal(typeof caps.rfc_version, "string");
  assert.ok(caps.features.input_audio_formats.includes("pcm16"));
});

test("request timeout header and local signal cooperate on a live route", async () => {
  const models = await client.models.list(undefined, { timeoutMs: 5_000 });
  assert.equal(models.object, "list");
});
