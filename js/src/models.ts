import type { RequestOptions, Transport } from "./http.ts";
import type { ListModelsResponse } from "./api-types.ts";

export class Models {
  private readonly transport: Transport;

  constructor(transport: Transport) {
    this.transport = transport;
  }

  async list(params?: { task?: string }, options?: RequestOptions): Promise<ListModelsResponse> {
    const query = params?.task != null ? "?" + new URLSearchParams({ task: params.task }).toString() : "";
    const response = await this.transport.get("/v1/models" + query, options);
    return (await response.json()) as ListModelsResponse;
  }
}
