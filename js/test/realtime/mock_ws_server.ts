// Minimal RFC 6455 server-side WebSocket over node:http -- just enough for
// the realtime client tests (text frames, close, ping/pong), zero deps.
import { createServer, type IncomingMessage, type Server } from "node:http";
import { createHash } from "node:crypto";
import type { Socket } from "node:net";
import type { AddressInfo } from "node:net";

const WS_GUID = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11";

export interface MockConnection {
  request: IncomingMessage;
  url: URL;
  received: string[];
  send(obj: unknown): void;
  sendRaw(text: string): void;
  close(code?: number): void;
  destroy(): void;
  onMessage(cb: (text: string) => void): void;
  waitForMessage(pred: (text: string) => boolean, timeoutMs?: number): Promise<string>;
}

export interface MockRealtimeServer {
  port: number;
  url: string;
  connections: MockConnection[];
  waitForConnection(index?: number, timeoutMs?: number): Promise<MockConnection>;
  close(): Promise<void>;
}

function encodeTextFrame(text: string): Buffer {
  const payload = Buffer.from(text, "utf8");
  const len = payload.length;
  let header: Buffer;
  if (len < 126) {
    header = Buffer.from([0x81, len]);
  } else if (len < 65536) {
    header = Buffer.alloc(4);
    header[0] = 0x81;
    header[1] = 126;
    header.writeUInt16BE(len, 2);
  } else {
    header = Buffer.alloc(10);
    header[0] = 0x81;
    header[1] = 127;
    header.writeBigUInt64BE(BigInt(len), 2);
  }
  return Buffer.concat([header, payload]);
}

function encodeCloseFrame(code: number): Buffer {
  const payload = Buffer.alloc(2);
  payload.writeUInt16BE(code, 0);
  return Buffer.concat([Buffer.from([0x88, 2]), payload]);
}

function makeConnection(request: IncomingMessage, socket: Socket): MockConnection {
  const received: string[] = [];
  const messageCbs: Array<(text: string) => void> = [];
  let closed = false;
  let pending = Buffer.alloc(0);

  const deliver = (text: string) => {
    received.push(text);
    for (const cb of messageCbs) cb(text);
  };

  socket.on("data", (chunk: Buffer) => {
    pending = Buffer.concat([pending, chunk]);
    for (;;) {
      if (pending.length < 2) return;
      const opcode = pending[0] & 0x0f;
      const masked = (pending[1] & 0x80) !== 0;
      let len = pending[1] & 0x7f;
      let offset = 2;
      if (len === 126) {
        if (pending.length < 4) return;
        len = pending.readUInt16BE(2);
        offset = 4;
      } else if (len === 127) {
        if (pending.length < 10) return;
        len = Number(pending.readBigUInt64BE(2));
        offset = 10;
      }
      const maskLen = masked ? 4 : 0;
      if (pending.length < offset + maskLen + len) return;
      const mask = masked ? pending.subarray(offset, offset + 4) : null;
      const payload = Buffer.from(
        pending.subarray(offset + maskLen, offset + maskLen + len),
      );
      if (mask) {
        for (let i = 0; i < payload.length; i++) payload[i] ^= mask[i % 4];
      }
      pending = pending.subarray(offset + maskLen + len);
      if (opcode === 0x1) {
        deliver(payload.toString("utf8"));
      } else if (opcode === 0x8) {
        if (!closed) {
          closed = true;
          try {
            socket.write(encodeCloseFrame(1000));
          } catch {}
        }
        socket.end();
        return;
      } else if (opcode === 0x9) {
        const pong = Buffer.concat([
          Buffer.from([0x8a, payload.length]),
          payload,
        ]);
        try {
          socket.write(pong);
        } catch {}
      }
    }
  });
  socket.on("error", () => {});

  return {
    request,
    url: new URL(request.url ?? "/", "ws://localhost"),
    received,
    send(obj: unknown) {
      this.sendRaw(JSON.stringify(obj));
    },
    sendRaw(text: string) {
      if (closed) return;
      socket.write(encodeTextFrame(text));
    },
    close(code = 1000) {
      if (closed) return;
      closed = true;
      try {
        socket.write(encodeCloseFrame(code));
      } catch {}
      socket.end();
    },
    destroy() {
      closed = true;
      socket.destroy();
    },
    onMessage(cb) {
      messageCbs.push(cb);
    },
    waitForMessage(pred, timeoutMs = 2000) {
      const existing = received.find(pred);
      if (existing !== undefined) return Promise.resolve(existing);
      return new Promise((resolve, reject) => {
        const timer = setTimeout(
          () => reject(new Error("timed out waiting for client message")),
          timeoutMs,
        );
        messageCbs.push((text) => {
          if (pred(text)) {
            clearTimeout(timer);
            resolve(text);
          }
        });
      });
    },
  };
}

export async function startMockRealtimeServer(
  onConnection?: (conn: MockConnection) => void,
): Promise<MockRealtimeServer> {
  const server: Server = createServer((_req, res) => {
    res.writeHead(426).end();
  });
  const connections: MockConnection[] = [];
  const connectionWaiters: Array<() => void> = [];

  server.on("upgrade", (request, socket: Socket) => {
    const key = request.headers["sec-websocket-key"];
    if (typeof key !== "string") {
      socket.destroy();
      return;
    }
    const accept = createHash("sha1").update(key + WS_GUID).digest("base64");
    socket.write(
      "HTTP/1.1 101 Switching Protocols\r\n" +
        "Upgrade: websocket\r\n" +
        "Connection: Upgrade\r\n" +
        `Sec-WebSocket-Accept: ${accept}\r\n\r\n`,
    );
    const conn = makeConnection(request, socket);
    connections.push(conn);
    onConnection?.(conn);
    for (const w of connectionWaiters.splice(0)) w();
  });

  await new Promise<void>((resolve) => server.listen(0, "127.0.0.1", resolve));
  const port = (server.address() as AddressInfo).port;

  return {
    port,
    url: `ws://127.0.0.1:${port}`,
    connections,
    waitForConnection(index = 0, timeoutMs = 2000) {
      if (connections.length > index) {
        return Promise.resolve(connections[index]);
      }
      return new Promise((resolve, reject) => {
        const timer = setTimeout(
          () => reject(new Error(`timed out waiting for connection #${index}`)),
          timeoutMs,
        );
        const check = () => {
          if (connections.length > index) {
            clearTimeout(timer);
            resolve(connections[index]);
          } else {
            connectionWaiters.push(check);
          }
        };
        connectionWaiters.push(check);
      });
    },
    close() {
      for (const c of connections) c.destroy();
      return new Promise<void>((resolve) => server.close(() => resolve()));
    },
  };
}
