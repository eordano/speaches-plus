import type { RequestOptions, Transport } from "./http.ts";
import { sseBodyOf } from "./http.ts";
import { openaiSseJson } from "./sse.ts";
import type {
  ChatCompletionChunk,
  ChatCompletionRequest,
  ChatCompletionResponse,
} from "./api-types.ts";

export type ChatStreamingRequest = ChatCompletionRequest & { stream: true };
export type ChatNonStreamingRequest = ChatCompletionRequest & { stream?: false | null };

export class ChatCompletions {
  private readonly transport: Transport;

  constructor(transport: Transport) {
    this.transport = transport;
  }

  create(body: ChatStreamingRequest, options?: RequestOptions): Promise<AsyncIterable<ChatCompletionChunk>>;
  create(body: ChatNonStreamingRequest, options?: RequestOptions): Promise<ChatCompletionResponse>;
  async create(
    body: ChatCompletionRequest,
    options?: RequestOptions,
  ): Promise<AsyncIterable<ChatCompletionChunk> | ChatCompletionResponse> {
    const response = await this.transport.postJson("/v1/chat/completions", body, options);
    if (body.stream === true) return openaiSseJson<ChatCompletionChunk>(sseBodyOf(response));
    return (await response.json()) as ChatCompletionResponse;
  }
}

export class Chat {
  readonly completions: ChatCompletions;

  constructor(transport: Transport) {
    this.completions = new ChatCompletions(transport);
  }
}
