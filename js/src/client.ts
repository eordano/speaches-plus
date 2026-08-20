import { Audio } from "./audio.ts";
import { Chat } from "./chat.ts";
import { Completions } from "./completions.ts";
import { Embeddings } from "./embeddings.ts";
import { Transport, type ClientOptions, type RequestOptions } from "./http.ts";
import { Messages } from "./messages.ts";
import { Models } from "./models.ts";
import { Responses } from "./responses.ts";
import { VoiceProfiles } from "./voice-profiles.ts";

export class NurClient {
  readonly transport: Transport;
  readonly chat: Chat;
  readonly completions: Completions;
  readonly messages: Messages;
  readonly responses: Responses;
  readonly models: Models;
  readonly embeddings: Embeddings;
  readonly audio: Audio;
  readonly voiceProfiles: VoiceProfiles;

  constructor(options: ClientOptions = {}) {
    this.transport = new Transport(options);
    this.chat = new Chat(this.transport);
    this.completions = new Completions(this.transport);
    this.messages = new Messages(this.transport);
    this.responses = new Responses(this.transport);
    this.models = new Models(this.transport);
    this.embeddings = new Embeddings(this.transport);
    this.audio = new Audio(this.transport);
    this.voiceProfiles = new VoiceProfiles(this.transport);
  }

  async health(options?: RequestOptions): Promise<string> {
    const response = await this.transport.get("/health", options);
    return response.text();
  }

  async version(options?: RequestOptions): Promise<{ version: string }> {
    const response = await this.transport.get("/version", options);
    return (await response.json()) as { version: string };
  }
}
