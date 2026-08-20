import type {
  JsonValue,
  RealtimeClientEvent,
  RealtimeItem,
  RealtimeServerEvent,
  RealtimeServerEventOf,
  RealtimeServerEventType,
  RealtimeSessionView,
} from "./events.ts";

export type RealtimeIntent = "conversation" | "transcription";

export interface RealtimeSessionParams {
  model?: string;
  intent?: RealtimeIntent;
  transcriptionModel?: string;
  voice?: string;
  speechModel?: string;
}

export interface RealtimeReconnectOptions {
  maxAttempts?: number;
  initialDelayMs?: number;
  maxDelayMs?: number;
}

type WebSocketLike = {
  readyState: number;
  send(data: string): void;
  close(code?: number, reason?: string): void;
  addEventListener(type: string, listener: (ev: any) => void): void;
};

export type WebSocketCtor = new (url: string, options?: unknown) => WebSocketLike;

export interface RealtimeConnectOptions extends RealtimeSessionParams {
  baseUrl: string;
  url?: string;
  apiKey?: string;
  signal?: AbortSignal;
  webSocket?: WebSocketCtor;
  reconnect?: boolean | RealtimeReconnectOptions;
  onMalformedFrame?: (raw: string) => void;
}

export type RealtimeClientState =
  | "connecting"
  | "open"
  | "reconnecting"
  | "closed";

const WS_OPEN = 1;

export function buildRealtimeUrl(
  baseUrl: string,
  params: RealtimeSessionParams = {},
): string {
  const wsBase = baseUrl
    .replace(/^http:/, "ws:")
    .replace(/^https:/, "wss:")
    .replace(/\/+$/, "");
  const url = new URL(`${wsBase}/v1/realtime`);
  if (params.model !== undefined) url.searchParams.set("model", params.model);
  if (params.intent !== undefined) url.searchParams.set("intent", params.intent);
  if (params.transcriptionModel !== undefined) {
    url.searchParams.set("transcription_model", params.transcriptionModel);
  }
  if (params.voice !== undefined) url.searchParams.set("voice", params.voice);
  if (params.speechModel !== undefined) {
    url.searchParams.set("speech_model", params.speechModel);
  }
  return url.toString();
}

export function pcm16ToBase64(
  samples: Int16Array | ArrayBuffer | Uint8Array,
): string {
  let bytes: Uint8Array;
  if (samples instanceof Int16Array) {
    bytes = new Uint8Array(samples.buffer, samples.byteOffset, samples.byteLength);
  } else if (samples instanceof ArrayBuffer) {
    bytes = new Uint8Array(samples);
  } else {
    bytes = samples;
  }
  const B = (globalThis as Record<string, any>).Buffer;
  if (typeof B?.from === "function") {
    return B.from(bytes).toString("base64");
  }
  let bin = "";
  const CHUNK = 0x8000;
  for (let i = 0; i < bytes.length; i += CHUNK) {
    bin += String.fromCharCode(...bytes.subarray(i, i + CHUNK));
  }
  return btoa(bin);
}

interface Subscriber {
  buffer: RealtimeServerEvent[];
  notify: (() => void) | null;
}

function normalizeReconnect(
  opt: boolean | RealtimeReconnectOptions | undefined,
): Required<RealtimeReconnectOptions> | null {
  if (!opt) return null;
  const o = opt === true ? {} : opt;
  return {
    maxAttempts: o.maxAttempts ?? 5,
    initialDelayMs: o.initialDelayMs ?? 250,
    maxDelayMs: o.maxDelayMs ?? 4000,
  };
}

function abortError(reason?: unknown): Error {
  if (reason instanceof Error) return reason;
  return new DOMException("realtime connection aborted", "AbortError");
}

function isNodeRuntime(): boolean {
  const p = (globalThis as Record<string, any>).process;
  return typeof p?.versions?.node === "string";
}

export class RealtimeClient {
  readonly url: string;

  #opts: RealtimeConnectOptions;
  #ws: WebSocketLike | null = null;
  #state: RealtimeClientState = "connecting";
  #handlers = new Map<string, Set<(ev: RealtimeServerEvent) => void>>();
  #anyHandlers = new Set<(ev: RealtimeServerEvent) => void>();
  #stateHandlers = new Set<(state: RealtimeClientState) => void>();
  #subscribers = new Set<Subscriber>();
  #reconnect: Required<RealtimeReconnectOptions> | null;
  #reconnectAttempt = 0;
  #reconnectTimer: ReturnType<typeof setTimeout> | null = null;
  #closeReason: unknown = null;
  #onAbort: (() => void) | null = null;

  private constructor(opts: RealtimeConnectOptions) {
    this.#opts = opts;
    this.url = opts.url ?? buildRealtimeUrl(opts.baseUrl, opts);
    this.#reconnect = normalizeReconnect(opts.reconnect);
    if (opts.signal) {
      const signal = opts.signal;
      this.#onAbort = () => this.#shutdown(abortError(signal.reason));
      if (signal.aborted) {
        this.#onAbort();
      } else {
        signal.addEventListener("abort", this.#onAbort, { once: true });
      }
    }
  }

  static async connect(opts: RealtimeConnectOptions): Promise<RealtimeClient> {
    const client = new RealtimeClient(opts);
    await client.#open();
    return client;
  }

  get state(): RealtimeClientState {
    return this.#state;
  }

  #setState(state: RealtimeClientState): void {
    if (this.#state === state) return;
    this.#state = state;
    for (const h of this.#stateHandlers) h(state);
  }

  #newSocket(): WebSocketLike {
    const Ctor: WebSocketCtor =
      this.#opts.webSocket ?? ((globalThis as Record<string, any>).WebSocket as WebSocketCtor);
    if (!Ctor) {
      throw new Error(
        "no WebSocket constructor available: pass options.webSocket (Node < 22 has no global WebSocket)",
      );
    }
    if (this.#opts.apiKey !== undefined && (this.#opts.webSocket || isNodeRuntime())) {
      return new Ctor(this.url, {
        headers: { authorization: `Bearer ${this.#opts.apiKey}` },
      });
    }
    return new Ctor(this.url);
  }

  #open(isReconnectAttempt = false): Promise<void> {
    return new Promise((resolve, reject) => {
      if (this.#state === "closed") {
        reject(abortError(this.#closeReason ?? undefined));
        return;
      }
      let ws: WebSocketLike;
      try {
        ws = this.#newSocket();
      } catch (err) {
        reject(err);
        return;
      }
      this.#ws = ws;
      let settled = false;
      ws.addEventListener("open", () => {
        settled = true;
        this.#reconnectAttempt = 0;
        this.#setState("open");
        resolve();
      });
      ws.addEventListener("message", (ev: { data: unknown }) => {
        this.#onFrame(ev.data);
      });
      ws.addEventListener("close", () => {
        if (this.#ws !== ws) return;
        this.#ws = null;
        if (this.#state === "closed") return;
        if (!settled) {
          settled = true;
          reject(new Error(`websocket connection failed: ${this.url}`));
          if (isReconnectAttempt) {
            this.#maybeReconnect();
          } else {
            this.#shutdown(null);
          }
          return;
        }
        this.#maybeReconnect();
      });
      ws.addEventListener("error", () => {});
    });
  }

  #maybeReconnect(): void {
    const rc = this.#reconnect;
    if (!rc || this.#reconnectAttempt >= rc.maxAttempts) {
      this.#shutdown(null);
      return;
    }
    this.#reconnectAttempt += 1;
    this.#setState("reconnecting");
    const delay = Math.min(
      rc.initialDelayMs * 2 ** (this.#reconnectAttempt - 1),
      rc.maxDelayMs,
    );
    this.#reconnectTimer = setTimeout(() => {
      this.#reconnectTimer = null;
      this.#open(true).catch(() => {});
    }, delay);
  }

  #onFrame(data: unknown): void {
    const text =
      typeof data === "string"
        ? data
        : data instanceof ArrayBuffer
          ? new TextDecoder().decode(data)
          : null;
    if (text === null) return;
    let parsed: unknown;
    try {
      parsed = JSON.parse(text);
    } catch {
      this.#opts.onMalformedFrame?.(text);
      return;
    }
    const type = (parsed as { type?: unknown })?.type;
    if (typeof type !== "string") {
      this.#opts.onMalformedFrame?.(text);
      return;
    }
    this.#dispatch(parsed as RealtimeServerEvent);
  }

  #dispatch(event: RealtimeServerEvent): void {
    for (const h of this.#handlers.get(event.type) ?? []) h(event);
    for (const h of this.#anyHandlers) h(event);
    for (const sub of this.#subscribers) {
      sub.buffer.push(event);
      sub.notify?.();
    }
  }

  on<K extends RealtimeServerEventType>(
    type: K,
    handler: (event: RealtimeServerEventOf<K>) => void,
  ): () => void {
    let set = this.#handlers.get(type);
    if (!set) {
      set = new Set();
      this.#handlers.set(type, set);
    }
    const h = handler as (ev: RealtimeServerEvent) => void;
    set.add(h);
    return () => set.delete(h);
  }

  onAny(handler: (event: RealtimeServerEvent) => void): () => void {
    this.#anyHandlers.add(handler);
    return () => this.#anyHandlers.delete(handler);
  }

  onStateChange(handler: (state: RealtimeClientState) => void): () => void {
    this.#stateHandlers.add(handler);
    return () => this.#stateHandlers.delete(handler);
  }

  events(): AsyncGenerator<RealtimeServerEvent, void, void> {
    const sub: Subscriber = { buffer: [], notify: null };
    this.#subscribers.add(sub);
    const self = this;
    async function* subscribedBeforeFirstNext(): AsyncGenerator<
      RealtimeServerEvent,
      void,
      void
    > {
      try {
        for (;;) {
          while (sub.buffer.length > 0) {
            yield sub.buffer.shift() as RealtimeServerEvent;
          }
          if (self.#state === "closed") {
            if (self.#closeReason != null) throw self.#closeReason;
            return;
          }
          await new Promise<void>((resolve) => {
            sub.notify = resolve;
          });
          sub.notify = null;
        }
      } finally {
        self.#subscribers.delete(sub);
      }
    }
    return subscribedBeforeFirstNext();
  }

  [Symbol.asyncIterator](): AsyncGenerator<RealtimeServerEvent, void, void> {
    return this.events();
  }

  send(event: RealtimeClientEvent): void {
    if (!this.#ws || this.#ws.readyState !== WS_OPEN) {
      throw new Error(
        `cannot send in state '${this.#state}': the server keeps no offline buffer`,
      );
    }
    this.#ws.send(JSON.stringify(event));
  }

  updateSession(session: RealtimeSessionView): void {
    this.send({ type: "session.update", session });
  }

  appendAudio(audio: string | Int16Array | ArrayBuffer | Uint8Array): void {
    const b64 = typeof audio === "string" ? audio : pcm16ToBase64(audio);
    this.send({ type: "input_audio_buffer.append", audio: b64 });
  }

  commitAudio(): void {
    this.send({ type: "input_audio_buffer.commit" });
  }

  clearAudio(): void {
    this.send({ type: "input_audio_buffer.clear" });
  }

  createItem(item: RealtimeItem, previousItemId?: string): void {
    this.send({
      type: "conversation.item.create",
      item,
      ...(previousItemId !== undefined ? { previous_item_id: previousItemId } : {}),
    });
  }

  createUserText(text: string, id?: string): void {
    this.createItem({
      ...(id !== undefined ? { id } : {}),
      type: "message",
      role: "user",
      content: [{ type: "input_text", text }],
    });
  }

  deleteItem(itemId: string): void {
    this.send({ type: "conversation.item.delete", item_id: itemId });
  }

  truncateItem(itemId: string, contentIndex: number, audioEndMs: number): void {
    this.send({
      type: "conversation.item.truncate",
      item_id: itemId,
      content_index: contentIndex,
      audio_end_ms: audioEndMs,
    });
  }

  createResponse(response?: JsonValue): void {
    this.send({
      type: "response.create",
      ...(response !== undefined ? { response } : {}),
    });
  }

  cancelResponse(): void {
    this.send({ type: "response.cancel" });
  }

  close(): void {
    this.#shutdown(null);
  }

  #shutdown(reason: unknown): void {
    if (this.#state === "closed") return;
    this.#closeReason = reason;
    if (this.#reconnectTimer !== null) {
      clearTimeout(this.#reconnectTimer);
      this.#reconnectTimer = null;
    }
    if (this.#onAbort && this.#opts.signal) {
      this.#opts.signal.removeEventListener("abort", this.#onAbort);
      this.#onAbort = null;
    }
    const ws = this.#ws;
    this.#ws = null;
    if (ws && ws.readyState <= WS_OPEN) {
      try {
        ws.close(1000);
      } catch {}
    }
    this.#setState("closed");
    for (const sub of this.#subscribers) sub.notify?.();
  }
}
