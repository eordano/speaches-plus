import type { RequestOptions, Transport } from "./http.ts";
import type { TextEmbeddingRequest, TextEmbeddingResponse } from "./api-types.ts";

export class Embeddings {
  private readonly transport: Transport;

  constructor(transport: Transport) {
    this.transport = transport;
  }

  async create(body: TextEmbeddingRequest, options?: RequestOptions): Promise<TextEmbeddingResponse> {
    const response = await this.transport.postJson("/v1/embeddings", body, options);
    return (await response.json()) as TextEmbeddingResponse;
  }
}
