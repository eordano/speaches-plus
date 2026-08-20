import { test } from "node:test";
import assert from "node:assert/strict";

import { RealtimeClient } from "../../src/realtime/client.ts";
import type { RealtimeClientState } from "../../src/realtime/client.ts";
import { startMockRealtimeServer } from "./mock_ws_server.ts";
import { SESSION_CREATED } from "./fixtures.ts";

test("server drop triggers reconnect; the new session emits session.created again", async () => {
  const server = await startMockRealtimeServer((conn) => {
    conn.send({
      ...SESSION_CREATED,
      session: {
        ...SESSION_CREATED.session,
        id: `sess_mock_${server.connections.length}`,
      },
    });
  });
  try {
    const states: RealtimeClientState[] = [];
    const client = await RealtimeClient.connect({
      baseUrl: server.url,
      reconnect: { maxAttempts: 3, initialDelayMs: 20, maxDelayMs: 50 },
    });
    client.onStateChange((s) => states.push(s));

    const sessionIds: string[] = [];
    client.on("session.created", (ev) => {
      if (typeof ev.session.id === "string") sessionIds.push(ev.session.id);
    });

    const firstConn = await server.waitForConnection(0);
    const secondCreated = new Promise<void>((resolve) => {
      const off = client.on("session.created", () => {
        if (sessionIds.length >= 2) {
          off();
          resolve();
        }
      });
    });
    firstConn.close();
    await secondCreated;

    assert.deepEqual(sessionIds, ["sess_mock_1", "sess_mock_2"]);
    assert.ok(states.includes("reconnecting"), `states: ${states.join(",")}`);
    assert.equal(client.state, "open");
    client.close();
    assert.equal(client.state, "closed");
  } finally {
    await server.close();
  }
});

test("user close never reconnects", async () => {
  const server = await startMockRealtimeServer();
  try {
    const client = await RealtimeClient.connect({
      baseUrl: server.url,
      reconnect: { maxAttempts: 5, initialDelayMs: 10 },
    });
    await server.waitForConnection(0);
    client.close();
    await new Promise((r) => setTimeout(r, 100));
    assert.equal(server.connections.length, 1);
    assert.equal(client.state, "closed");
  } finally {
    await server.close();
  }
});

test("reconnect gives up after maxAttempts and closes", async () => {
  const server = await startMockRealtimeServer();
  const client = await RealtimeClient.connect({
    baseUrl: server.url,
    reconnect: { maxAttempts: 2, initialDelayMs: 10, maxDelayMs: 20 },
  });
  await server.waitForConnection(0);
  const closed = new Promise<void>((resolve) => {
    client.onStateChange((s) => {
      if (s === "closed") resolve();
    });
  });
  await server.close();
  await closed;
  assert.equal(client.state, "closed");
});

test("without reconnect, a server drop closes the client and ends iteration", async () => {
  const server = await startMockRealtimeServer();
  try {
    const client = await RealtimeClient.connect({ baseUrl: server.url });
    const conn = await server.waitForConnection(0);
    conn.send(SESSION_CREATED);

    const types: string[] = [];
    const iterated = (async () => {
      for await (const ev of client) types.push(ev.type);
    })();
    conn.close();
    await iterated;
    assert.deepEqual(types, ["session.created"]);
    assert.equal(client.state, "closed");
  } finally {
    await server.close();
  }
});
