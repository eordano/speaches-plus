export interface NurErrorDetails {
  status: number;
  code?: string | null;
  errorType?: string | null;
  param?: string | null;
  raw?: unknown;
}

export class NurError extends Error {
  readonly status: number;
  readonly code: string | null;
  readonly errorType: string | null;
  readonly param: string | null;
  readonly raw: unknown;

  constructor(message: string, details: NurErrorDetails) {
    super(message);
    this.name = "NurError";
    this.status = details.status;
    this.code = details.code ?? null;
    this.errorType = details.errorType ?? null;
    this.param = details.param ?? null;
    this.raw = details.raw ?? null;
  }

  static async fromResponse(response: Response): Promise<NurError> {
    let text = "";
    try {
      text = await response.text();
    } catch {
      text = "";
    }
    let raw: unknown = null;
    try {
      raw = JSON.parse(text);
    } catch {
      raw = null;
    }
    const details = envelopeDetails(raw);
    return new NurError(details?.message ?? (text || `HTTP ${response.status}`), {
      status: response.status,
      code: details?.code ?? null,
      errorType: details?.errorType ?? null,
      param: details?.param ?? null,
      raw: raw ?? (text || null),
    });
  }

  static fromStreamErrorEvent(payload: { error: { type: string; message: string } }): NurError {
    return new NurError(payload.error.message, {
      status: 200,
      code: payload.error.type,
      errorType: "error",
      raw: payload,
    });
  }
}

interface Envelope {
  message: string;
  code: string | null;
  errorType: string | null;
  param: string | null;
}

const asString = (value: unknown): string | null => (typeof value === "string" ? value : null);

function envelopeDetails(raw: unknown): Envelope | null {
  if (typeof raw !== "object" || raw === null) return null;
  const body = raw as Record<string, unknown>;
  const err = body["error"];
  if (typeof err === "object" && err !== null) {
    const e = err as Record<string, unknown>;
    const message = asString(e["message"]) ?? JSON.stringify(err);
    if (body["type"] === "error") {
      return { message, code: asString(e["type"]), errorType: "error", param: null };
    }
    return { message, code: asString(e["code"]), errorType: asString(e["type"]), param: asString(e["param"]) };
  }
  const detail = body["detail"];
  if (Array.isArray(detail)) {
    return {
      message: JSON.stringify(detail),
      code: "validation_error",
      errorType: "invalid_request_error",
      param: null,
    };
  }
  return null;
}
