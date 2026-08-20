import type { RequestOptions, Transport } from "./http.ts";
import type {
  JsonValue,
  ListVoiceProfilesResponse,
  VoiceProfileDeleteAck,
  VoiceProfileResponse,
} from "./api-types.ts";

export interface VoiceProfileCreateParams {
  name: string;
  file: Blob;
  design_params?: JsonValue;
  fileName?: string;
}

export class VoiceProfiles {
  private readonly transport: Transport;

  constructor(transport: Transport) {
    this.transport = transport;
  }

  async create(params: VoiceProfileCreateParams, options?: RequestOptions): Promise<VoiceProfileResponse> {
    const form = new FormData();
    form.append("name", params.name);
    const fileName = params.fileName ?? (params.file instanceof File ? params.file.name : "reference.wav");
    form.append("file", params.file, fileName);
    if (params.design_params !== undefined) form.append("design_params", JSON.stringify(params.design_params));
    const response = await this.transport.postForm("/v1/voice-profiles", form, options);
    return (await response.json()) as VoiceProfileResponse;
  }

  async list(options?: RequestOptions): Promise<ListVoiceProfilesResponse> {
    const response = await this.transport.get("/v1/voice-profiles", options);
    return (await response.json()) as ListVoiceProfilesResponse;
  }

  async retrieve(name: string, options?: RequestOptions): Promise<VoiceProfileResponse> {
    const response = await this.transport.get(`/v1/voice-profiles/${encodeURIComponent(name)}`, options);
    return (await response.json()) as VoiceProfileResponse;
  }

  async delete(name: string, options?: RequestOptions): Promise<VoiceProfileDeleteAck> {
    const response = await this.transport.delete(`/v1/voice-profiles/${encodeURIComponent(name)}`, options);
    return (await response.json()) as VoiceProfileDeleteAck;
  }
}
