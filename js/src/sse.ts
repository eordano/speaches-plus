export interface SseFrame {
  event: string | null;
  data: string;
}

export async function* sseFrames(
  body: ReadableStream<Uint8Array>,
): AsyncGenerator<SseFrame, void, undefined> {
  const reader = body.getReader();
  const decoder = new TextDecoder();
  let buffered = "";
  let dataLines: string[] = [];
  let eventName: string | null = null;
  let sawField = false;
  try {
    for (;;) {
      const { done, value } = await reader.read();
      if (done) return;
      buffered += decoder.decode(value, { stream: true });
      let newline: number;
      while ((newline = buffered.indexOf("\n")) >= 0) {
        let line = buffered.slice(0, newline);
        buffered = buffered.slice(newline + 1);
        if (line.endsWith("\r")) line = line.slice(0, -1);
        if (line === "") {
          if (sawField) yield { event: eventName, data: dataLines.join("\n") };
          dataLines = [];
          eventName = null;
          sawField = false;
          continue;
        }
        if (line.startsWith(":")) continue;
        const colon = line.indexOf(":");
        const field = colon < 0 ? line : line.slice(0, colon);
        let value2 = colon < 0 ? "" : line.slice(colon + 1);
        if (value2.startsWith(" ")) value2 = value2.slice(1);
        if (field === "data") {
          dataLines.push(value2);
          sawField = true;
        } else if (field === "event") {
          eventName = value2;
          sawField = true;
        }
      }
    }
  } finally {
    try {
      await reader.cancel();
    } catch {
      reader.releaseLock();
    }
  }
}

export async function* openaiSseJson<T>(
  body: ReadableStream<Uint8Array>,
): AsyncGenerator<T, void, undefined> {
  for await (const frame of sseFrames(body)) {
    if (frame.data === "[DONE]") return;
    if (frame.data !== "") yield JSON.parse(frame.data) as T;
  }
}

export async function* namedSseJson<T>(
  body: ReadableStream<Uint8Array>,
): AsyncGenerator<T, void, undefined> {
  for await (const frame of sseFrames(body)) {
    if (frame.data !== "") yield JSON.parse(frame.data) as T;
  }
}
