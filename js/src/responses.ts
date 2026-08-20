import type { RequestOptions, Transport } from "./http.ts";
import { sseBodyOf } from "./http.ts";
import { namedSseJson } from "./sse.ts";
import type {
  ResponseDeleteAck,
  ResponseObject,
  ResponsesRequest,
  ResponsesStreamEvent,
} from "./api-types.ts";

export type ResponsesCreateParams = Omit<ResponsesRequest, "tools"> & {
  tools?: ResponsesRequest["tools"];
};
export type ResponsesStreamingParams = ResponsesCreateParams & { stream: true };
export type ResponsesNonStreamingParams = ResponsesCreateParams & { stream?: false | null };

export class Responses {
  private readonly transport: Transport;

  constructor(transport: Transport) {
    this.transport = transport;
  }

  create(body: ResponsesStreamingParams, options?: RequestOptions): Promise<AsyncIterable<ResponsesStreamEvent>>;
  create(body: ResponsesNonStreamingParams, options?: RequestOptions): Promise<ResponseObject>;
  async create(
    body: ResponsesCreateParams,
    options?: RequestOptions,
  ): Promise<AsyncIterable<ResponsesStreamEvent> | ResponseObject> {
    const response = await this.transport.postJson("/v1/responses", body, options);
    if (body.stream === true) return namedSseJson<ResponsesStreamEvent>(sseBodyOf(response));
    return (await response.json()) as ResponseObject;
  }

  async retrieve(id: string, options?: RequestOptions): Promise<ResponseObject> {
    const response = await this.transport.get(`/v1/responses/${encodeURIComponent(id)}`, options);
    return (await response.json()) as ResponseObject;
  }

  async delete(id: string, options?: RequestOptions): Promise<ResponseDeleteAck> {
    const response = await this.transport.delete(`/v1/responses/${encodeURIComponent(id)}`, options);
    return (await response.json()) as ResponseDeleteAck;
  }
}
