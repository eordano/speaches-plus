import type { RequestOptions, Transport } from "./http.ts";
import type {
  AudioSpeechRequest,
  TranscriptionDiarizedJsonResponse,
  TranscriptionJsonResponse,
  TranscriptionVerboseJsonResponse,
} from "./api-types.ts";

export type TranscriptionResponseFormat =
  | "json"
  | "text"
  | "verbose_json"
  | "srt"
  | "vtt"
  | "diarized_json";

export interface SpeechToTextParams {
  file: Blob;
  model?: string;
  response_format?: TranscriptionResponseFormat;
  fileName?: string;
}

const TEXTUAL_FORMATS: ReadonlySet<string> = new Set(["text", "srt", "vtt"]);

export class SpeechToText {
  private readonly transport: Transport;
  private readonly path: string;

  constructor(transport: Transport, path: string) {
    this.transport = transport;
    this.path = path;
  }

  create(
    params: SpeechToTextParams & { response_format: "verbose_json" },
    options?: RequestOptions,
  ): Promise<TranscriptionVerboseJsonResponse>;
  create(
    params: SpeechToTextParams & { response_format: "diarized_json" },
    options?: RequestOptions,
  ): Promise<TranscriptionDiarizedJsonResponse>;
  create(
    params: SpeechToTextParams & { response_format: "text" | "srt" | "vtt" },
    options?: RequestOptions,
  ): Promise<string>;
  create(
    params: SpeechToTextParams & { response_format?: "json" },
    options?: RequestOptions,
  ): Promise<TranscriptionJsonResponse>;
  async create(
    params: SpeechToTextParams,
    options?: RequestOptions,
  ): Promise<
    TranscriptionJsonResponse | TranscriptionVerboseJsonResponse | TranscriptionDiarizedJsonResponse | string
  > {
    const form = new FormData();
    const fileName = params.fileName ?? (params.file instanceof File ? params.file.name : "audio.wav");
    form.append("file", params.file, fileName);
    if (params.model != null) form.append("model", params.model);
    if (params.response_format != null) form.append("response_format", params.response_format);
    const response = await this.transport.postForm(this.path, form, options);
    if (TEXTUAL_FORMATS.has(params.response_format ?? "json")) return response.text();
    return (await response.json()) as
      | TranscriptionJsonResponse
      | TranscriptionVerboseJsonResponse
      | TranscriptionDiarizedJsonResponse;
  }
}

export class Speech {
  private readonly transport: Transport;

  constructor(transport: Transport) {
    this.transport = transport;
  }

  create(body: AudioSpeechRequest, options?: RequestOptions): Promise<Response> {
    return this.transport.postJson("/v1/audio/speech", body, options);
  }
}

export class Audio {
  readonly speech: Speech;
  readonly transcriptions: SpeechToText;
  readonly translations: SpeechToText;

  constructor(transport: Transport) {
    this.speech = new Speech(transport);
    this.transcriptions = new SpeechToText(transport, "/v1/audio/transcriptions");
    this.translations = new SpeechToText(transport, "/v1/audio/translations");
  }
}
