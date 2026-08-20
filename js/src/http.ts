import { NurError } from "./errors.ts";

export type FetchLike = (input: string | URL, init?: RequestInit) => Promise<Response>;

export interface ClientOptions {
  baseURL?: string;
  apiKey?: string | null;
  apiKeyHeader?: "authorization" | "x-api-key";
  fetch?: FetchLike;
  defaultHeaders?: Record<string, string>;
  timeoutMs?: number;
}

export interface RequestOptions {
  signal?: AbortSignal;
  timeoutMs?: number;
  headers?: Record<string, string>;
  onResponse?: (response: Response) => void;
}

export const SPEC_DECODE_HEADER = "x-spec-decode";
export const REQUEST_TIMEOUT_HEADER = "x-request-timeout-ms";

export class Transport {
  readonly baseURL: string;
  private readonly options: ClientOptions;

  constructor(options: ClientOptions = {}) {
    this.options = options;
    this.baseURL = (options.baseURL ?? "").replace(/\/+$/, "");
  }

  private buildHeaders(jsonBody: boolean, options?: RequestOptions): Record<string, string> {
    const headers: Record<string, string> = {};
    if (jsonBody) headers["content-type"] = "application/json";
    const key = this.options.apiKey;
    if (key) {
      if ((this.options.apiKeyHeader ?? "authorization") === "x-api-key") headers["x-api-key"] = key;
      else headers["authorization"] = `Bearer ${key}`;
    }
    const timeoutMs = options?.timeoutMs ?? this.options.timeoutMs;
    if (timeoutMs != null) headers[REQUEST_TIMEOUT_HEADER] = String(Math.ceil(timeoutMs));
    Object.assign(headers, this.options.defaultHeaders, options?.headers);
    return headers;
  }

  private buildSignal(options?: RequestOptions): AbortSignal | null {
    const timeoutMs = options?.timeoutMs ?? this.options.timeoutMs;
    const parts: AbortSignal[] = [];
    if (options?.signal) parts.push(options.signal);
    if (timeoutMs != null) parts.push(AbortSignal.timeout(timeoutMs));
    if (parts.length === 0) return null;
    return parts.length === 1 ? (parts[0] as AbortSignal) : AbortSignal.any(parts);
  }

  async send(
    path: string,
    init: { method: string; body?: BodyInit; jsonBody?: boolean },
    options?: RequestOptions,
  ): Promise<Response> {
    const fetchImpl: FetchLike = this.options.fetch ?? globalThis.fetch.bind(globalThis);
    const response = await fetchImpl(this.baseURL + path, {
      method: init.method,
      body: init.body ?? null,
      headers: this.buildHeaders(init.jsonBody === true, options),
      signal: this.buildSignal(options),
    });
    options?.onResponse?.(response);
    if (!response.ok) throw await NurError.fromResponse(response);
    return response;
  }

  postJson(path: string, body: unknown, options?: RequestOptions): Promise<Response> {
    return this.send(path, { method: "POST", body: JSON.stringify(body), jsonBody: true }, options);
  }

  postForm(path: string, form: FormData, options?: RequestOptions): Promise<Response> {
    return this.send(path, { method: "POST", body: form }, options);
  }

  get(path: string, options?: RequestOptions): Promise<Response> {
    return this.send(path, { method: "GET" }, options);
  }

  delete(path: string, options?: RequestOptions): Promise<Response> {
    return this.send(path, { method: "DELETE" }, options);
  }
}

export function sseBodyOf(response: Response): ReadableStream<Uint8Array> {
  if (!response.body) {
    throw new NurError("streaming response has no body stream", { status: response.status });
  }
  return response.body;
}
