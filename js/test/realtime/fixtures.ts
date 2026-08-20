// Event sequences captured from the shapes client/e2e_browser_events.mjs
// asserts against the live server (session.created v2 shape, response
// lifecycle brackets from rust/src/realtime/events.rs, error payloads).

export const SESSION_CREATED = {
  type: "session.created",
  event_id: "evt_000000000000000000000000",
  session: {
    id: "sess_mock_001",
    object: "realtime.session",
    type: "realtime",
    voice: "af_heart",
    input_audio_format: "pcm16",
    output_audio_format: "pcm16",
    modalities: ["audio", "text"],
    output_modalities: ["audio", "text"],
    audio: {
      input: { format: "pcm16", turn_detection: { type: "server_vad" } },
      output: { format: "pcm16", voice: "af_heart" },
    },
  },
} as const;

export function sessionUpdated(session: Record<string, unknown>) {
  return {
    type: "session.updated",
    event_id: "evt_000000000000000000000001",
    session: { ...SESSION_CREATED.session, ...session },
  };
}

export const RESPONSE_LIFECYCLE = [
  {
    type: "response.created",
    response: { id: "resp_001", object: "realtime.response", status: "in_progress" },
  },
  {
    type: "response.output_item.added",
    response_id: "resp_001",
    output_index: 0,
    item: {
      id: "item_asst_001",
      object: "realtime.item",
      type: "message",
      role: "assistant",
      status: "in_progress",
      content: [],
    },
  },
  {
    type: "response.content_part.added",
    response_id: "resp_001",
    item_id: "item_asst_001",
    output_index: 0,
    content_index: 0,
    part: { type: "audio", transcript: "" },
  },
  {
    type: "response.output_audio_transcript.delta",
    response_id: "resp_001",
    item_id: "item_asst_001",
    output_index: 0,
    content_index: 0,
    delta: "Hello there.",
  },
  {
    type: "response.output_audio.delta",
    response_id: "resp_001",
    item_id: "item_asst_001",
    output_index: 0,
    content_index: 0,
    delta: "AAAA",
  },
  {
    type: "response.output_audio_transcript.done",
    response_id: "resp_001",
    item_id: "item_asst_001",
    output_index: 0,
    content_index: 0,
    transcript: "Hello there.",
  },
  {
    type: "response.output_audio.done",
    response_id: "resp_001",
    item_id: "item_asst_001",
    output_index: 0,
    content_index: 0,
  },
  {
    type: "response.content_part.done",
    response_id: "resp_001",
    item_id: "item_asst_001",
    output_index: 0,
    content_index: 0,
    part: { type: "audio", transcript: "Hello there." },
  },
  {
    type: "response.output_item.done",
    response_id: "resp_001",
    output_index: 0,
    item: {
      id: "item_asst_001",
      type: "message",
      role: "assistant",
      status: "completed",
      content: [{ type: "audio", transcript: "Hello there." }],
    },
  },
  {
    type: "response.done",
    response: {
      id: "resp_001",
      object: "realtime.response",
      status: "completed",
      audio_end_ms: 1234,
      output: [
        {
          id: "item_asst_001",
          type: "message",
          role: "assistant",
          content: [{ type: "audio", transcript: "Hello there." }],
        },
      ],
    },
  },
] as const;

export const ERROR_EVENT = {
  type: "error",
  event_id: "evt_000000000000000000000009",
  error: {
    type: "invalid_request_error",
    code: "unknown_event_type",
    message: "unknown event type: totally.bogus.event",
    param: "type",
  },
} as const;

export const DIARIZATION_EVENT = {
  type: "conversation.item.diarization",
  item_id: "item_user_001",
  audio_end_ms: 2000,
  elapsed_ms: 40,
  segments: [{ speaker: 0, start_ms: 0, end_ms: 1800 }],
} as const;
