import type {
  AnthropicMessagesStreamEvent,
  ChatCompletionChunk,
  ResponseObject,
  ResponsesStreamEvent,
} from "../../src/api-types.ts";

export function streamOf(chunks: string[]): ReadableStream<Uint8Array> {
  const encoder = new TextEncoder();
  let i = 0;
  return new ReadableStream({
    pull(controller) {
      if (i < chunks.length) {
        controller.enqueue(encoder.encode(chunks[i] as string));
        i++;
      } else {
        controller.close();
      }
    },
  });
}

export function byteChunkedStream(text: string, size: number): ReadableStream<Uint8Array> {
  const bytes = new TextEncoder().encode(text);
  let offset = 0;
  return new ReadableStream({
    pull(controller) {
      if (offset >= bytes.length) return controller.close();
      controller.enqueue(bytes.slice(offset, offset + size));
      offset += size;
    },
  });
}

const chatChunk = (
  delta: { role?: string; content?: string; reasoning_content?: string },
  finish: string | null,
): ChatCompletionChunk => ({
  id: "chatcmpl-1",
  object: "chat.completion.chunk",
  created: 1,
  model: "llm-default",
  choices: [{ index: 0, delta, finish_reason: finish }],
});

export const chatChunks: ChatCompletionChunk[] = [
  chatChunk({ role: "assistant" }, null),
  chatChunk({ reasoning_content: "pondering…" }, null),
  chatChunk({ content: "¡Hola" }, null),
  chatChunk({ content: " mundo!" }, "stop"),
];

export const chatSseText =
  chatChunks.map((chunk) => `data: ${JSON.stringify(chunk)}\n\n`).join("") + "data: [DONE]\n\n";

export const anthropicEvents: AnthropicMessagesStreamEvent[] = [
  {
    type: "message_start",
    message: {
      id: "msg_1",
      type: "message",
      role: "assistant",
      model: "llm-default",
      content: [],
      stop_reason: null,
      stop_sequence: null,
      usage: { input_tokens: 3, output_tokens: 0, cache_creation_input_tokens: 0, cache_read_input_tokens: 0 },
    },
  },
  { type: "ping" },
  { type: "content_block_start", index: 0, content_block: { type: "thinking", thinking: "", signature: "" } },
  { type: "content_block_delta", index: 0, delta: { type: "thinking_delta", thinking: "hmm" } },
  { type: "content_block_stop", index: 0 },
  { type: "content_block_start", index: 1, content_block: { type: "text", text: "" } },
  { type: "content_block_delta", index: 1, delta: { type: "text_delta", text: "Hello" } },
  { type: "content_block_stop", index: 1 },
  {
    type: "message_delta",
    delta: { stop_reason: "end_turn", stop_sequence: null },
    usage: { output_tokens: 5, cache_read_input_tokens: 0 },
  },
  { type: "message_stop" },
];

export const anthropicSseText = anthropicEvents
  .map((event) => `event: ${event.type}\ndata: ${JSON.stringify(event)}\n\n`)
  .join("");

export const anthropicErrorEvent: AnthropicMessagesStreamEvent = {
  type: "error",
  error: { type: "overloaded_error", message: "engine busy" },
};

export const anthropicErrorSseText =
  anthropicEvents
    .slice(0, 4)
    .map((event) => `event: ${event.type}\ndata: ${JSON.stringify(event)}\n\n`)
    .join("") + `event: error\ndata: ${JSON.stringify(anthropicErrorEvent)}\n\n`;

export const responseObject: ResponseObject = {
  id: "resp_1",
  object: "response",
  created_at: 1,
  status: "completed",
  error: null,
  incomplete_details: null,
  instructions: null,
  max_output_tokens: null,
  model: "llm-default",
  output: [
    {
      type: "message",
      id: "msg_1",
      status: "completed",
      role: "assistant",
      content: [{ type: "output_text", text: "Hello", annotations: [] }],
    },
  ],
  parallel_tool_calls: false,
  previous_response_id: null,
  store: false,
  temperature: null,
  tool_choice: "auto",
  tools: [],
  top_p: null,
  truncation: "disabled",
  usage: null,
  metadata: {},
};

export const responsesEvents: ResponsesStreamEvent[] = [
  { type: "response.created", response: { ...responseObject, status: "in_progress" }, sequence_number: 0 },
  {
    type: "response.output_text.delta",
    item_id: "msg_1",
    output_index: 0,
    content_index: 0,
    delta: "Hello",
    sequence_number: 1,
  },
  {
    type: "response.output_text.done",
    item_id: "msg_1",
    output_index: 0,
    content_index: 0,
    text: "Hello",
    sequence_number: 2,
  },
  { type: "response.completed", response: responseObject, sequence_number: 3 },
];

export const responsesSseText = responsesEvents
  .map((event) => `event: ${event.type}\ndata: ${JSON.stringify(event)}\n\n`)
  .join("");
