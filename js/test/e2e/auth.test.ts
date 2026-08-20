import { after, before, test } from "node:test";
import assert from "node:assert/strict";
import { NurClient, NurError } from "../../src/index.ts";
import { bootServer, type E2eServer } from "./harness.ts";

const KEY = "e2e-secret-key";

let server: E2eServer;

before(async () => {
  server = await bootServer({ apiKey: KEY });
});

after(async () => {
  await server.stop();
});

test("no key: /v1 routes refuse with 401, /health stays exempt", async () => {
  const bare = new NurClient({ baseURL: server.baseURL });
  await assert.rejects(bare.models.list(), (err: unknown) => {
    assert.ok(err instanceof NurError);
    assert.equal(err.status, 401);
    return true;
  });
  assert.equal(await bare.health(), "ok");
});

test("bearer authorization header is accepted", async () => {
  const client = new NurClient({ baseURL: server.baseURL, apiKey: KEY });
  const models = await client.models.list();
  assert.equal(models.object, "list");
});

test("anthropic-style x-api-key header is accepted", async () => {
  const client = new NurClient({ baseURL: server.baseURL, apiKey: KEY, apiKeyHeader: "x-api-key" });
  const models = await client.models.list();
  assert.equal(models.object, "list");
});

test("wrong key refuses with 401", async () => {
  const client = new NurClient({ baseURL: server.baseURL, apiKey: "not-the-key" });
  await assert.rejects(client.models.list(), (err: unknown) => {
    assert.ok(err instanceof NurError);
    assert.equal(err.status, 401);
    return true;
  });
});
