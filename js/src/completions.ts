import type { RequestOptions, Transport } from "./http.ts";
import { sseBodyOf } from "./http.ts";
import { openaiSseJson } from "./sse.ts";
import type { CompletionChunk, CompletionRequest, CompletionResponse } from "./api-types.ts";

export type CompletionStreamingRequest = CompletionRequest & { stream: true };
export type CompletionNonStreamingRequest = CompletionRequest & { stream?: false | null };

export class Completions {
  private readonly transport: Transport;

  constructor(transport: Transport) {
    this.transport = transport;
  }

  create(body: CompletionStreamingRequest, options?: RequestOptions): Promise<AsyncIterable<CompletionChunk>>;
  create(body: CompletionNonStreamingRequest, options?: RequestOptions): Promise<CompletionResponse>;
  async create(
    body: CompletionRequest,
    options?: RequestOptions,
  ): Promise<AsyncIterable<CompletionChunk> | CompletionResponse> {
    const response = await this.transport.postJson("/v1/completions", body, options);
    if (body.stream === true) return openaiSseJson<CompletionChunk>(sseBodyOf(response));
    return (await response.json()) as CompletionResponse;
  }
}
