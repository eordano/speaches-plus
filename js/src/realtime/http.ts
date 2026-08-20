import type { RealtimeCapabilities } from "./events.ts";
import type { RealtimeSessionParams } from "./client.ts";

export interface RealtimeHttpOptions {
  baseUrl: string;
  apiKey?: string;
  fetch?: typeof fetch;
  signal?: AbortSignal;
}

function httpBase(baseUrl: string): string {
  return baseUrl
    .replace(/^ws:/, "http:")
    .replace(/^wss:/, "https:")
    .replace(/\/+$/, "");
}

function authHeaders(apiKey: string | undefined): Record<string, string> {
  return apiKey !== undefined ? { authorization: `Bearer ${apiKey}` } : {};
}

export async function fetchRealtimeCapabilities(
  opts: RealtimeHttpOptions,
): Promise<RealtimeCapabilities> {
  const f = opts.fetch ?? fetch;
  const resp = await f(`${httpBase(opts.baseUrl)}/v1/realtime/capabilities`, {
    headers: authHeaders(opts.apiKey),
    signal: opts.signal,
  });
  if (!resp.ok) {
    throw new Error(`realtime capabilities failed: HTTP ${resp.status}`);
  }
  return (await resp.json()) as RealtimeCapabilities;
}

// Typed surface for POST /v1/realtime (main.rs realtime_post): submit a
// WebRTC SDP offer, receive the answer SDP. Peer-connection management stays
// with the caller; this is capability surface only.
export async function postRealtimeSdpOffer(
  offerSdp: string,
  opts: RealtimeHttpOptions & RealtimeSessionParams,
): Promise<string> {
  const f = opts.fetch ?? fetch;
  const url = new URL(`${httpBase(opts.baseUrl)}/v1/realtime`);
  if (opts.model !== undefined) url.searchParams.set("model", opts.model);
  if (opts.intent !== undefined) url.searchParams.set("intent", opts.intent);
  if (opts.transcriptionModel !== undefined) {
    url.searchParams.set("transcription_model", opts.transcriptionModel);
  }
  if (opts.voice !== undefined) url.searchParams.set("voice", opts.voice);
  if (opts.speechModel !== undefined) {
    url.searchParams.set("speech_model", opts.speechModel);
  }
  const resp = await f(url.toString(), {
    method: "POST",
    headers: { "content-type": "application/sdp", ...authHeaders(opts.apiKey) },
    body: offerSdp,
    signal: opts.signal,
  });
  if (!resp.ok) {
    const body = await resp.text().catch(() => "");
    throw new Error(`realtime SDP offer failed: HTTP ${resp.status} ${body}`.trim());
  }
  return await resp.text();
}
