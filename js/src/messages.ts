import { NurError } from "./errors.ts";
import type { RequestOptions, Transport } from "./http.ts";
import { sseBodyOf } from "./http.ts";
import { namedSseJson } from "./sse.ts";
import type {
  AnthropicCountTokensResponse,
  AnthropicMessagesResponse,
  AnthropicMessagesStreamEvent,
  MessagesRequest,
} from "./api-types.ts";

export type MessagesStreamingRequest = MessagesRequest & { stream: true };
export type MessagesNonStreamingRequest = MessagesRequest & { stream?: false | null };

async function* anthropicStreamEvents(
  body: ReadableStream<Uint8Array>,
): AsyncGenerator<AnthropicMessagesStreamEvent, void, undefined> {
  for await (const event of namedSseJson<AnthropicMessagesStreamEvent>(body)) {
    if (event.type === "error") throw NurError.fromStreamErrorEvent(event);
    yield event;
  }
}

export class Messages {
  private readonly transport: Transport;

  constructor(transport: Transport) {
    this.transport = transport;
  }

  create(body: MessagesStreamingRequest, options?: RequestOptions): Promise<AsyncIterable<AnthropicMessagesStreamEvent>>;
  create(body: MessagesNonStreamingRequest, options?: RequestOptions): Promise<AnthropicMessagesResponse>;
  async create(
    body: MessagesRequest,
    options?: RequestOptions,
  ): Promise<AsyncIterable<AnthropicMessagesStreamEvent> | AnthropicMessagesResponse> {
    const response = await this.transport.postJson("/v1/messages", body, options);
    if (body.stream === true) return anthropicStreamEvents(sseBodyOf(response));
    return (await response.json()) as AnthropicMessagesResponse;
  }

  async countTokens(
    body: Omit<MessagesRequest, "stream">,
    options?: RequestOptions,
  ): Promise<AnthropicCountTokensResponse> {
    const response = await this.transport.postJson("/v1/messages/count_tokens", body, options);
    return (await response.json()) as AnthropicCountTokensResponse;
  }
}
