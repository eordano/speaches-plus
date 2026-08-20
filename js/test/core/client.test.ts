import { test } from "node:test";
import assert from "node:assert/strict";
import { NurClient, NurError } from "../../src/index.ts";
import type { ChatNonStreamingRequest, MessagesStreamingRequest } from "../../src/index.ts";
import {
  anthropicErrorSseText,
  anthropicEvents,
  anthropicSseText,
  chatChunks,
  chatSseText,
  responsesEvents,
  responsesSseText,
  streamOf,
} from "./fixtures.ts";

interface RecordedCall {
  url: string;
  method: string;
  headers: Headers;
  body: BodyInit | null | undefined;
  signal: AbortSignal | null | undefined;
}

function stubClient(
  respond: (call: RecordedCall) => Response,
  clientOptions: ConstructorParameters<typeof NurClient>[0] = {},
): { client: NurClient; calls: RecordedCall[] } {
  const calls: RecordedCall[] = [];
  const client = new NurClient({
    baseURL: "http://server.test",
    ...clientOptions,
    fetch: async (url, init) => {
      const call: RecordedCall = {
        url: String(url),
        method: init?.method ?? "GET",
        headers: new Headers(init?.headers),
        body: init?.body,
        signal: init?.signal,
      };
      calls.push(call);
      return respond(call);
    },
  });
  return { client, calls };
}

const jsonResponse = (body: unknown, status = 200): Response =>
  new Response(JSON.stringify(body), { status, headers: { "content-type": "application/json" } });

const CHAT_GOLDEN: ChatNonStreamingRequest = {
  model: "llm-default",
  messages: [{ role: "user", content: "¿por qué el cielo es azul?" }],
  temperature: 0.7,
  top_k: 40,
  min_p: 0.05,
  enable_thinking: true,
  guided_choice: ["a", "b"],
  seed: 7,
};

test("chat.completions.create serializes the request body verbatim", async () => {
  const reply = {
    id: "c1",
    object: "chat.completion",
    created: 1,
    model: "llm-default",
    choices: [{ index: 0, message: { role: "assistant", content: "azul" }, finish_reason: "stop" }],
    usage: { prompt_tokens: 1, completion_tokens: 1, total_tokens: 2 },
  };
  const { client, calls } = stubClient(() => jsonResponse(reply));
  const out = await client.chat.completions.create(CHAT_GOLDEN);
  assert.deepEqual(out, reply);
  assert.equal(calls.length, 1);
  const call = calls[0] as RecordedCall;
  assert.equal(call.url, "http://server.test/v1/chat/completions");
  assert.equal(call.method, "POST");
  assert.equal(call.headers.get("content-type"), "application/json");
  assert.deepEqual(JSON.parse(String(call.body)), CHAT_GOLDEN);
});

test("apiKey defaults to Authorization: Bearer", async () => {
  const { client, calls } = stubClient(() => jsonResponse({ object: "list", data: [] }), {
    apiKey: "sk-test",
  });
  await client.models.list();
  assert.equal((calls[0] as RecordedCall).headers.get("authorization"), "Bearer sk-test");
  assert.equal((calls[0] as RecordedCall).headers.get("x-api-key"), null);
});

test("apiKeyHeader x-api-key sends the anthropic-style header instead", async () => {
  const { client, calls } = stubClient(() => jsonResponse({ object: "list", data: [] }), {
    apiKey: "sk-test",
    apiKeyHeader: "x-api-key",
  });
  await client.models.list();
  assert.equal((calls[0] as RecordedCall).headers.get("x-api-key"), "sk-test");
  assert.equal((calls[0] as RecordedCall).headers.get("authorization"), null);
});

test("timeoutMs sets the server-honored x-request-timeout-ms header", async () => {
  const { client, calls } = stubClient(() => jsonResponse({ object: "list", data: [] }));
  await client.models.list(undefined, { timeoutMs: 2500 });
  const call = calls[0] as RecordedCall;
  assert.equal(call.headers.get("x-request-timeout-ms"), "2500");
  assert.ok(call.signal, "a timeout must install an AbortSignal");
});

test("an aborted caller signal reaches fetch aborted", async () => {
  const { client, calls } = stubClient(() => jsonResponse({ object: "list", data: [] }));
  const controller = new AbortController();
  controller.abort();
  await client.models.list(undefined, { signal: controller.signal });
  assert.equal((calls[0] as RecordedCall).signal?.aborted, true);
});

test("models.list forwards the task filter as a query parameter", async () => {
  const { client, calls } = stubClient(() => jsonResponse({ object: "list", data: [] }));
  await client.models.list({ task: "chat" });
  assert.equal((calls[0] as RecordedCall).url, "http://server.test/v1/models?task=chat");
});

test("chat.completions.create stream:true yields typed chunks", async () => {
  const { client, calls } = stubClient(() => new Response(streamOf([chatSseText])));
  const stream = await client.chat.completions.create({ ...CHAT_GOLDEN, stream: true });
  const chunks = [];
  for await (const chunk of stream) chunks.push(chunk);
  assert.deepEqual(chunks, chatChunks);
  assert.deepEqual(JSON.parse(String((calls[0] as RecordedCall).body)), { ...CHAT_GOLDEN, stream: true });
});

test("onResponse exposes headers like x-spec-decode on streaming calls", async () => {
  const { client } = stubClient(
    () => new Response(streamOf([chatSseText]), { headers: { "x-spec-decode": "on" } }),
  );
  let spec: string | null = null;
  await client.chat.completions.create(
    { ...CHAT_GOLDEN, stream: true },
    { onResponse: (r) => (spec = r.headers.get("x-spec-decode")) },
  );
  assert.equal(spec, "on");
});

test("completions.create serializes prompt requests and streams chunks", async () => {
  const nonStream = {
    id: "cmpl-1",
    object: "text_completion",
    created: 1,
    model: "llm-default",
    choices: [{ text: " world", index: 0, finish_reason: "stop", logprobs: null }],
    usage: { prompt_tokens: 1, completion_tokens: 1, total_tokens: 2 },
  };
  const { client, calls } = stubClient(() => jsonResponse(nonStream));
  const body = { model: "llm-default", prompt: "hello", echo: true, stop_token_ids: [7] };
  const out = await client.completions.create(body);
  assert.deepEqual(out, nonStream);
  assert.equal((calls[0] as RecordedCall).url, "http://server.test/v1/completions");
  assert.deepEqual(JSON.parse(String((calls[0] as RecordedCall).body)), body);
  const chunk = { ...nonStream, choices: [{ text: "hi", index: 0, finish_reason: null, logprobs: null }] };
  const { client: streaming } = stubClient(
    () => new Response(streamOf([`data: ${JSON.stringify(chunk)}\n\ndata: [DONE]\n\n`])),
  );
  const chunks = [];
  for await (const c of await streaming.completions.create({ ...body, stream: true })) chunks.push(c);
  assert.deepEqual(chunks, [chunk]);
});

test("messages.create parses the anthropic named-event stream", async () => {
  const { client, calls } = stubClient(() => new Response(streamOf([anthropicSseText])));
  const body: MessagesStreamingRequest = {
    model: "llm-default",
    max_tokens: 64,
    messages: [{ role: "user", content: "hi" }],
    stream: true,
  };
  const events = [];
  for await (const event of await client.messages.create(body)) events.push(event);
  assert.deepEqual(events, anthropicEvents);
  assert.equal((calls[0] as RecordedCall).url, "http://server.test/v1/messages");
  assert.deepEqual(JSON.parse(String((calls[0] as RecordedCall).body)), body);
});

test("an anthropic error stream event throws a NurError mid-iteration", async () => {
  const { client } = stubClient(() => new Response(streamOf([anthropicErrorSseText])));
  const stream = await client.messages.create({
    model: "llm-default",
    max_tokens: 64,
    messages: [{ role: "user", content: "hi" }],
    stream: true,
  });
  let count = 0;
  await assert.rejects(async () => {
    for await (const event of stream) {
      void event;
      count++;
    }
  });
  assert.equal(count, 4);
});

test("messages.countTokens posts to the count_tokens route", async () => {
  const { client, calls } = stubClient(() => jsonResponse({ input_tokens: 5 }));
  const out = await client.messages.countTokens({
    model: "llm-default",
    messages: [{ role: "user", content: "hi" }],
  });
  assert.deepEqual(out, { input_tokens: 5 });
  assert.equal((calls[0] as RecordedCall).url, "http://server.test/v1/messages/count_tokens");
});

test("responses.create stream:true yields the named responses events", async () => {
  const { client } = stubClient(() => new Response(streamOf([responsesSseText])));
  const stream = await client.responses.create({ model: "llm-default", input: "hi", stream: true });
  const events = [];
  for await (const event of stream) events.push(event);
  assert.deepEqual(events, responsesEvents);
});

test("responses.retrieve and delete hit the id routes with encoding", async () => {
  const { client, calls } = stubClient(() =>
    jsonResponse({ id: "resp 1", object: "response", deleted: true }),
  );
  await client.responses.delete("resp 1");
  assert.equal((calls[0] as RecordedCall).url, "http://server.test/v1/responses/resp%201");
  assert.equal((calls[0] as RecordedCall).method, "DELETE");
});

test("embeddings.create posts the request body verbatim", async () => {
  const reply = {
    object: "list",
    data: [{ object: "embedding", index: 0, embedding: [0.1, 0.2] }],
    model: "embed-default",
    usage: { prompt_tokens: 2, total_tokens: 2 },
  };
  const { client, calls } = stubClient(() => jsonResponse(reply));
  const out = await client.embeddings.create({ model: "embed-default", input: ["a", "b"] });
  assert.deepEqual(out, reply);
  assert.equal((calls[0] as RecordedCall).url, "http://server.test/v1/embeddings");
  assert.deepEqual(JSON.parse(String((calls[0] as RecordedCall).body)), {
    model: "embed-default",
    input: ["a", "b"],
  });
});

test("audio.speech.create returns the raw binary Response", async () => {
  const wav = new Uint8Array([82, 73, 70, 70]);
  const { client, calls } = stubClient(() => new Response(wav, { headers: { "content-type": "audio/wav" } }));
  const body = { model: "tts-default", input: "hola", voice: "eze", response_format: "wav" };
  const response = await client.audio.speech.create(body);
  assert.deepEqual(new Uint8Array(await response.arrayBuffer()), wav);
  assert.deepEqual(JSON.parse(String((calls[0] as RecordedCall).body)), body);
});

test("audio.transcriptions.create sends multipart without unsupported STT params", async () => {
  const { client, calls } = stubClient(() => jsonResponse({ text: "hola" }));
  const out = await client.audio.transcriptions.create({
    file: new Blob([new Uint8Array([1, 2])], { type: "audio/wav" }),
    model: "stt-default",
    response_format: "json",
  });
  assert.deepEqual(out, { text: "hola" });
  const call = calls[0] as RecordedCall;
  assert.equal(call.url, "http://server.test/v1/audio/transcriptions");
  assert.ok(call.body instanceof FormData);
  const form = call.body as FormData;
  assert.equal(form.get("model"), "stt-default");
  assert.equal(form.get("response_format"), "json");
  assert.equal((form.get("file") as File).name, "audio.wav");
  assert.deepEqual([...form.keys()].sort(), ["file", "model", "response_format"]);
  assert.equal(call.headers.get("content-type"), null, "multipart boundary must come from fetch");
});

test("audio.translations.create with a text format resolves to a string", async () => {
  const { client, calls } = stubClient(() => new Response("hello", { status: 200 }));
  const out = await client.audio.translations.create({
    file: new Blob([new Uint8Array([1])]),
    response_format: "text",
  });
  assert.equal(out, "hello");
  assert.equal((calls[0] as RecordedCall).url, "http://server.test/v1/audio/translations");
});

test("voiceProfiles CRUD hits the profile routes with multipart create", async () => {
  const profile = { name: "eze", schema_version: 1, embedding_dim: 256, embedding_state: "encoded" };
  const { client, calls } = stubClient(() => jsonResponse(profile));
  const out = await client.voiceProfiles.create({
    name: "eze",
    file: new Blob([new Uint8Array([1])]),
    design_params: { pace: 1 },
  });
  assert.deepEqual(out, profile);
  const form = (calls[0] as RecordedCall).body as FormData;
  assert.equal(form.get("name"), "eze");
  assert.equal(form.get("design_params"), '{"pace":1}');
  assert.ok(form.get("file") instanceof File);
  await client.voiceProfiles.retrieve("a b");
  assert.equal((calls[1] as RecordedCall).url, "http://server.test/v1/voice-profiles/a%20b");
  await client.voiceProfiles.delete("eze");
  assert.equal((calls[2] as RecordedCall).method, "DELETE");
  assert.equal((calls[2] as RecordedCall).url, "http://server.test/v1/voice-profiles/eze");
});

test("openai error envelopes map onto NurError fields", async () => {
  const { client } = stubClient(() =>
    jsonResponse(
      {
        error: {
          message: "content part not supported",
          type: "invalid_request_error",
          param: "messages",
          code: "unsupported_content_part",
        },
      },
      400,
    ),
  );
  try {
    await client.chat.completions.create(CHAT_GOLDEN);
    assert.fail("must throw");
  } catch (error) {
    assert.ok(error instanceof NurError);
    assert.equal(error.status, 400);
    assert.equal(error.code, "unsupported_content_part");
    assert.equal(error.errorType, "invalid_request_error");
    assert.equal(error.param, "messages");
    assert.equal(error.message, "content part not supported");
  }
});

test("anthropic error envelopes map the 529 overload signal", async () => {
  const { client } = stubClient(() =>
    jsonResponse({ type: "error", error: { type: "overloaded_error", message: "busy" } }, 529),
  );
  try {
    await client.messages.create({ model: "m", max_tokens: 1, messages: [{ role: "user", content: "x" }] });
    assert.fail("must throw");
  } catch (error) {
    assert.ok(error instanceof NurError);
    assert.equal(error.status, 529);
    assert.equal(error.code, "overloaded_error");
    assert.equal(error.message, "busy");
  }
});

test("non-JSON error bodies still produce a NurError with the status", async () => {
  const { client } = stubClient(() => new Response("gateway exploded", { status: 502 }));
  try {
    await client.models.list();
    assert.fail("must throw");
  } catch (error) {
    assert.ok(error instanceof NurError);
    assert.equal(error.status, 502);
    assert.equal(error.message, "gateway exploded");
    assert.equal(error.code, null);
  }
});

test("health and version use the unprefixed ops routes", async () => {
  const { client, calls } = stubClient((call) =>
    call.url.endsWith("/health") ? new Response("ok") : jsonResponse({ version: "1.2.3" }),
  );
  assert.equal(await client.health(), "ok");
  assert.deepEqual(await client.version(), { version: "1.2.3" });
  assert.equal((calls[0] as RecordedCall).url, "http://server.test/health");
  assert.equal((calls[1] as RecordedCall).url, "http://server.test/version");
});

test("baseURL trailing slashes are normalized", async () => {
  const calls: RecordedCall[] = [];
  const client = new NurClient({
    baseURL: "http://server.test///",
    fetch: async (url, init) => {
      calls.push({
        url: String(url),
        method: init?.method ?? "GET",
        headers: new Headers(init?.headers),
        body: init?.body,
        signal: init?.signal,
      });
      return jsonResponse({ object: "list", data: [] });
    },
  });
  await client.models.list();
  assert.equal((calls[0] as RecordedCall).url, "http://server.test/v1/models");
});
