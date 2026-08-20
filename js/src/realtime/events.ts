// Typed mirror of rust/src/realtime/wire.rs (OutboundEvent) and the inbound
// dispatch in rust/src/realtime/session.rs::handle_client_event. At
// integration time the per-variant payload types below should be swapped for
// re-exports of the ts-rs generated bindings (js/src/generated/), which are
// produced from the same serde structs; the event-name lists here are gated by
// js/test/realtime/events.test.ts against the wire.rs-derived list.

export type JsonValue =
  | string
  | number
  | boolean
  | null
  | JsonValue[]
  | { [key: string]: JsonValue };

export type ItemId = string;
export type ResponseId = string;
export type EventId = string;

export interface RealtimeErrorPayload {
  type: string;
  code: string;
  message: string;
  event_id?: string;
  param?: string;
}

export type RealtimeResponseStatus =
  | "completed"
  | "cancelled"
  | "incomplete"
  | "failed";

export type RealtimeResponseStatusReason =
  | "drain_cap"
  | "token_limit"
  | "llm_error"
  | "tts_error"
  | "client_too_slow"
  | "barge_in"
  | "client_cancelled";

export interface RealtimeResponseStatusDetails {
  reason: RealtimeResponseStatusReason;
  error?: RealtimeErrorPayload;
}

export interface RealtimeResponsePayload {
  id: ResponseId;
  object: "realtime.response";
  status: RealtimeResponseStatus;
  audio_end_ms: number;
  output: JsonValue[];
  status_details?: RealtimeResponseStatusDetails;
}

export interface RealtimeContentPart {
  type?: string;
  text?: string;
  transcript?: string;
  audio_ms?: number;
  audio_end_ms?: number;
  [key: string]: JsonValue | undefined;
}

export interface RealtimeItem {
  id?: string;
  object?: "realtime.item";
  type?: "message" | (string & {});
  role?: "user" | "assistant" | "system" | (string & {});
  status?: string;
  content?: RealtimeContentPart[];
  [key: string]: JsonValue | RealtimeContentPart[] | undefined;
}

export interface RealtimeSessionView {
  id?: string;
  object?: "realtime.session";
  type?: "realtime";
  model?: string;
  voice?: string;
  instructions?: string;
  input_audio_format?: string;
  output_audio_format?: string;
  input_audio_transcription?: JsonValue;
  turn_detection?: JsonValue;
  modalities?: string[];
  output_modalities?: string[];
  audio?: {
    input?: { format?: string; transcription?: JsonValue; turn_detection?: JsonValue };
    output?: { format?: string; voice?: string };
  };
  [key: string]: JsonValue | undefined;
}

// The server stamps a monotonic event_id onto every outbound event
// (wire.rs EventSeq::stamp, applied after serde serialization -- which is why
// the ts-rs generated RealtimeOutboundEvent does not carry it).
interface Stamped {
  event_id?: EventId;
}

export type RealtimeServerEvent =
  | (Stamped & { type: "session.created"; session: RealtimeSessionView })
  | (Stamped & { type: "session.updated"; session: RealtimeSessionView })
  | (Stamped & { type: "session.done"; reason: string })
  | (Stamped & { type: "input_audio_buffer.speech_started"; item_id: ItemId; audio_start_ms: number })
  | (Stamped & { type: "input_audio_buffer.speech_stopped"; item_id: ItemId; audio_end_ms: number })
  | (Stamped & { type: "input_audio_buffer.committed"; item_id: ItemId })
  | (Stamped & { type: "input_audio_buffer.cleared" })
  | (Stamped & { type: "input_audio_buffer.partial_transcription"; item_id: ItemId; transcript: string; audio_end_ms: number })
  | (Stamped & { type: "conversation.item.added"; item: RealtimeItem })
  | (Stamped & { type: "conversation.item.deleted"; item_id: ItemId })
  | (Stamped & { type: "conversation.item.truncated"; item_id: ItemId; content_index: number; audio_end_ms: number })
  | (Stamped & { type: "conversation.item.assistant_truncated"; item_id: ItemId; audio_end_ms: number; transcript: string })
  | (Stamped & { type: "conversation.item.input_audio_transcription.completed"; item_id: ItemId; content_index: number; transcript: string })
  | (Stamped & { type: "conversation.item.input_audio_transcription.delta"; item_id: ItemId; content_index: number; delta: string })
  | (Stamped & { type: "conversation.item.input_audio_transcription.failed"; item_id: ItemId; content_index: number; error: JsonValue })
  | (Stamped & { type: "conversation.item.done"; item: RealtimeItem })
  | (Stamped & { type: "conversation.item.retrieved"; item: RealtimeItem })
  | (Stamped & { type: "response.created"; response: { id: ResponseId; object?: "realtime.response"; status?: "in_progress" | (string & {}); [key: string]: JsonValue | undefined } })
  | (Stamped & { type: "response.output_item.added"; response_id: ResponseId; output_index: number; item: RealtimeItem })
  | (Stamped & { type: "response.output_item.done"; response_id: ResponseId; output_index: number; item: RealtimeItem })
  | (Stamped & { type: "response.content_part.added"; response_id: ResponseId; item_id: ItemId; output_index: number; content_index: number; part: RealtimeContentPart })
  | (Stamped & { type: "response.content_part.done"; response_id: ResponseId; item_id: ItemId; output_index: number; content_index: number; part: RealtimeContentPart })
  | (Stamped & { type: "response.output_audio_transcript.delta"; response_id: ResponseId; item_id: ItemId; output_index: number; content_index: number; delta: string })
  | (Stamped & { type: "response.output_audio_transcript.done"; response_id: ResponseId; item_id: ItemId; output_index: number; content_index: number; transcript: string })
  | (Stamped & { type: "response.output_audio.delta"; response_id: ResponseId; item_id: ItemId; output_index: number; content_index: number; delta: string })
  | (Stamped & { type: "response.output_audio.done"; response_id: ResponseId; item_id: ItemId; output_index: number; content_index: number })
  | (Stamped & { type: "response.output_text.delta"; response_id: ResponseId; item_id: ItemId; output_index: number; content_index: number; delta: string })
  | (Stamped & { type: "response.output_text.done"; response_id: ResponseId; item_id: ItemId; output_index: number; content_index: number; text: string })
  | (Stamped & { type: "response.function_call_arguments.delta"; response_id: ResponseId; item_id: ItemId; output_index: number; call_id: string; delta: string })
  | (Stamped & { type: "response.function_call_arguments.done"; response_id: ResponseId; item_id: ItemId; output_index: number; call_id: string; arguments: string })
  | (Stamped & { type: "response.tool_progress"; response_id: ResponseId; item_id: ItemId; output_index: number; progress: JsonValue })
  | (Stamped & { type: "response.cancelled"; response_id: ResponseId })
  | (Stamped & { type: "response.done"; response: RealtimeResponsePayload })
  | (Stamped & { type: "output_audio_buffer.cleared" })
  | (Stamped & { type: "output_audio_buffer.started"; response_id: ResponseId })
  | (Stamped & { type: "output_audio_buffer.stopped"; response_id: ResponseId })
  | (Stamped & { type: "rate_limits.updated"; rate_limits: JsonValue })
  | (Stamped & { type: "error"; error: RealtimeErrorPayload })
  | (Stamped & { type: "conversation.item.diarization"; item_id: ItemId; audio_end_ms: number; elapsed_ms?: number | null; segments: JsonValue[] });

export type RealtimeServerEventType = RealtimeServerEvent["type"];

export type RealtimeServerEventOf<K extends RealtimeServerEventType> = Extract<
  RealtimeServerEvent,
  { type: K }
>;

// All 39 outbound variants of wire.rs OutboundEvent. Note this is a superset
// of the 38-name table in client/e2e_browser_events.mjs, which omits
// conversation.item.diarization (it only tolerates it in its unknown-filter).
export const SERVER_EVENT_TYPES = [
  "session.created",
  "session.updated",
  "session.done",
  "input_audio_buffer.speech_started",
  "input_audio_buffer.speech_stopped",
  "input_audio_buffer.committed",
  "input_audio_buffer.cleared",
  "input_audio_buffer.partial_transcription",
  "conversation.item.added",
  "conversation.item.deleted",
  "conversation.item.truncated",
  "conversation.item.assistant_truncated",
  "conversation.item.input_audio_transcription.completed",
  "conversation.item.input_audio_transcription.delta",
  "conversation.item.input_audio_transcription.failed",
  "conversation.item.done",
  "conversation.item.retrieved",
  "response.created",
  "response.output_item.added",
  "response.output_item.done",
  "response.content_part.added",
  "response.content_part.done",
  "response.output_audio_transcript.delta",
  "response.output_audio_transcript.done",
  "response.output_audio.delta",
  "response.output_audio.done",
  "response.output_text.delta",
  "response.output_text.done",
  "response.function_call_arguments.delta",
  "response.function_call_arguments.done",
  "response.tool_progress",
  "response.cancelled",
  "response.done",
  "output_audio_buffer.cleared",
  "output_audio_buffer.started",
  "output_audio_buffer.stopped",
  "rate_limits.updated",
  "error",
  "conversation.item.diarization",
] as const satisfies readonly RealtimeServerEventType[];

// Never emitted by this server (asserted by the e2e canonical-names test);
// kept so consumers migrating from OpenAI's older dialect can detect them.
export const LEGACY_SERVER_EVENT_TYPES = [
  "conversation.item.created",
  "response.audio.delta",
  "response.audio.done",
  "response.audio_transcript.delta",
  "response.audio_transcript.done",
  "response.text.delta",
  "response.text.done",
] as const;

const SERVER_EVENT_TYPE_SET: ReadonlySet<string> = new Set(SERVER_EVENT_TYPES);

export function isKnownServerEventType(
  type: string,
): type is RealtimeServerEventType {
  return SERVER_EVENT_TYPE_SET.has(type);
}

export interface RealtimeClientItemCreate {
  event_id?: string;
  type: "conversation.item.create";
  previous_item_id?: string;
  item: RealtimeItem;
}

// conversation.item.retrieve is accepted by the server but always answered
// with an invalid_request_error ("not yet implemented");
// session.rs::handle_client_event is the authority.
export type RealtimeClientEvent =
  | { event_id?: string; type: "session.update"; session: RealtimeSessionView }
  | { event_id?: string; type: "input_audio_buffer.append"; audio: string }
  | { event_id?: string; type: "input_audio_buffer.commit" }
  | { event_id?: string; type: "input_audio_buffer.clear" }
  | RealtimeClientItemCreate
  | { event_id?: string; type: "conversation.item.delete"; item_id: ItemId }
  | { event_id?: string; type: "conversation.item.truncate"; item_id: ItemId; content_index: number; audio_end_ms: number }
  | { event_id?: string; type: "conversation.item.retrieve"; item_id: ItemId }
  | { event_id?: string; type: "response.create"; response?: JsonValue }
  | { event_id?: string; type: "response.cancel" }
  | { event_id?: string; type: RealtimeV2NoopEventType };

// Accepted silently, no reply (v2_compat.rs KNOWN_V2_NOOP_EVENTS).
export const V2_NOOP_EVENT_TYPES = [
  "output_audio_buffer.clear",
  "output_audio_buffer.append",
  "input_audio_buffer.dtmf.received",
  "transcription_session.update",
  "response.cancel_audio",
] as const;

export type RealtimeV2NoopEventType = (typeof V2_NOOP_EVENT_TYPES)[number];

export type RealtimeClientEventType = RealtimeClientEvent["type"];

export interface RealtimeCapabilities {
  rfc_version: string;
  features: {
    eou_kinds: string[];
    fusion_rules: string[];
    input_audio_formats: string[];
    output_audio_formats: string[];
  };
  extensions: {
    eou_kinds: string[];
    fusion_rules: string[];
    eager_eou: boolean;
    integrated_eou: boolean;
    predicted_resp_phase: boolean;
    diarization: {
      enabled: boolean;
      max_speakers_per_chunk: number;
      max_speakers_per_frame: number;
      embedding_dim: number;
      frame_rate_hz: number;
      endpoints: {
        audio_diarization: string;
        audio_embeddings: string;
        transcription_diarized_json: string;
        realtime_event: string;
      };
    };
  };
}
