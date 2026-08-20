import { test } from "node:test";
import assert from "node:assert/strict";

import {
  fetchRealtimeCapabilities,
  postRealtimeSdpOffer,
} from "../../src/realtime/http.ts";

const CAPABILITIES_FIXTURE = {
  rfc_version: "v3",
  features: {
    eou_kinds: ["server_vad"],
    fusion_rules: ["noisy_or", "max", "mean", "weighted"],
    input_audio_formats: ["pcm16"],
    output_audio_formats: ["pcm16"],
  },
  extensions: {
    eou_kinds: ["eager"],
    fusion_rules: ["gated"],
    eager_eou: true,
    integrated_eou: true,
    predicted_resp_phase: true,
    diarization: {
      enabled: false,
      max_speakers_per_chunk: 0,
      max_speakers_per_frame: 0,
      embedding_dim: 0,
      frame_rate_hz: 0,
      endpoints: {
        audio_diarization: "/v1/audio/diarization",
        audio_embeddings: "/v1/audio/embeddings",
        transcription_diarized_json:
          "/v1/audio/transcriptions?response_format=diarized_json",
        realtime_event: "conversation.item.diarization",
      },
    },
  },
};

function stubFetch(handler: (url: string, init?: RequestInit) => Response) {
  const calls: Array<{ url: string; init?: RequestInit }> = [];
  const f = (async (input: any, init?: RequestInit) => {
    const url = String(input);
    calls.push({ url, init });
    return handler(url, init);
  }) as typeof fetch;
  return { f, calls };
}

test("fetchRealtimeCapabilities hits the capabilities route with auth", async () => {
  const { f, calls } = stubFetch(
    () => new Response(JSON.stringify(CAPABILITIES_FIXTURE), { status: 200 }),
  );
  const caps = await fetchRealtimeCapabilities({
    baseUrl: "ws://gpu:8000/",
    apiKey: "k",
    fetch: f,
  });
  assert.equal(calls[0].url, "http://gpu:8000/v1/realtime/capabilities");
  assert.equal(
    (calls[0].init?.headers as Record<string, string>).authorization,
    "Bearer k",
  );
  assert.equal(caps.rfc_version, "v3");
  assert.equal(
    caps.extensions.diarization.endpoints.realtime_event,
    "conversation.item.diarization",
  );
});

test("fetchRealtimeCapabilities surfaces HTTP failure", async () => {
  const { f } = stubFetch(() => new Response("nope", { status: 503 }));
  await assert.rejects(
    fetchRealtimeCapabilities({ baseUrl: "http://gpu:8000", fetch: f }),
    /HTTP 503/,
  );
});

test("postRealtimeSdpOffer posts application/sdp with query params", async () => {
  const { f, calls } = stubFetch(
    () =>
      new Response("v=0\r\nanswer", {
        status: 200,
        headers: { "content-type": "application/sdp" },
      }),
  );
  const answer = await postRealtimeSdpOffer("v=0\r\noffer", {
    baseUrl: "https://gpu:8443",
    model: "test-e2e",
    intent: "conversation",
    fetch: f,
  });
  const u = new URL(calls[0].url);
  assert.equal(u.pathname, "/v1/realtime");
  assert.equal(u.protocol, "https:");
  assert.equal(u.searchParams.get("model"), "test-e2e");
  assert.equal(u.searchParams.get("intent"), "conversation");
  assert.equal(calls[0].init?.method, "POST");
  assert.equal(
    (calls[0].init?.headers as Record<string, string>)["content-type"],
    "application/sdp",
  );
  assert.equal(calls[0].init?.body, "v=0\r\noffer");
  assert.equal(answer, "v=0\r\nanswer");
});

test("postRealtimeSdpOffer surfaces the server refusal body", async () => {
  const { f } = stubFetch(() => new Response("sdp_invalid", { status: 400 }));
  await assert.rejects(
    postRealtimeSdpOffer("bogus", { baseUrl: "http://gpu:8000", fetch: f }),
    /HTTP 400 sdp_invalid/,
  );
});
