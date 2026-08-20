import { spawn, type ChildProcess } from "node:child_process";
import { mkdtempSync, readdirSync, rmSync, statSync } from "node:fs";
import { homedir, tmpdir } from "node:os";
import { join } from "node:path";

const BUILD_HINT =
  "build it with: NVK_LANE=<lane> NVK_PKG=speaches-plus NVK_FEATURES= rust/scripts/nvk.sh build --bin speaches-plus";

export function findServerBinary(): string {
  const fromEnv = process.env["NUR_E2E_SERVER_BIN"];
  if (fromEnv) {
    statSync(fromEnv);
    return fromEnv;
  }
  const laneRoot = join(homedir(), ".cache", "cargo-tmp");
  let newest: { path: string; mtime: number } | null = null;
  let lanes: string[] = [];
  try {
    lanes = readdirSync(laneRoot).filter((d) => d.startsWith("tgt-"));
  } catch {
    lanes = [];
  }
  for (const lane of lanes) {
    const candidate = join(laneRoot, lane, "debug", "speaches-plus");
    try {
      const st = statSync(candidate);
      if (!newest || st.mtimeMs > newest.mtime) newest = { path: candidate, mtime: st.mtimeMs };
    } catch {
      continue;
    }
  }
  if (!newest) {
    throw new Error(
      `no speaches-plus server binary found under ${laneRoot}/tgt-*/debug and NUR_E2E_SERVER_BIN is unset; ${BUILD_HINT}`,
    );
  }
  return newest.path;
}

export interface E2eServer {
  baseURL: string;
  port: number;
  stop: () => Promise<void>;
}

async function waitForHealth(baseURL: string, child: ChildProcess, logs: string[]): Promise<void> {
  const deadline = Date.now() + 60_000;
  for (;;) {
    if (child.exitCode != null) {
      throw new Error(`server exited with code ${child.exitCode} before /health answered:\n${logs.join("")}`);
    }
    try {
      const r = await fetch(baseURL + "/health", { signal: AbortSignal.timeout(2_000) });
      if (r.ok) return;
    } catch {
      /* not up yet */
    }
    if (Date.now() > deadline) {
      throw new Error(`server did not answer /health within 60s:\n${logs.join("")}`);
    }
    await new Promise((r) => setTimeout(r, 250));
  }
}

export async function bootServer(options: { apiKey?: string } = {}): Promise<E2eServer> {
  const binary = findServerBinary();
  const scratch = mkdtempSync(join(tmpdir(), "nur-e2e-"));
  const modelsDir = join(scratch, "models");
  const profilesDir = join(scratch, "voice-profiles");
  const port = 20000 + Math.floor(Math.random() * 20000);

  const env: Record<string, string> = {};
  for (const [k, v] of Object.entries(process.env)) {
    if (v == null) continue;
    if (k === "NV_CHAT_MODEL_DIR" || k === "NV_CHAT_MODEL_DIRS" || k === "SPEACHES_API_KEY") continue;
    env[k] = v;
  }
  env["UVICORN_HOST"] = "127.0.0.1";
  env["UVICORN_PORT"] = String(port);
  env["SPEACHES_PLUS_MODELS"] = modelsDir;
  env["SPEACHES_PLUS_VOICE_PROFILES_DIR"] = profilesDir;
  if (options.apiKey != null) env["SPEACHES_API_KEY"] = options.apiKey;

  const child = spawn(binary, [], { env, stdio: ["ignore", "pipe", "pipe"] });
  const logs: string[] = [];
  child.stdout?.on("data", (b: Buffer) => logs.push(b.toString()));
  child.stderr?.on("data", (b: Buffer) => logs.push(b.toString()));

  const baseURL = `http://127.0.0.1:${port}`;
  try {
    await waitForHealth(baseURL, child, logs);
  } catch (err) {
    child.kill("SIGKILL");
    rmSync(scratch, { recursive: true, force: true });
    throw err;
  }

  const stop = async (): Promise<void> => {
    if (child.exitCode == null) {
      const exited = new Promise<void>((resolve) => child.once("exit", () => resolve()));
      child.kill("SIGTERM");
      const timer = setTimeout(() => child.kill("SIGKILL"), 5_000);
      await exited;
      clearTimeout(timer);
    }
    rmSync(scratch, { recursive: true, force: true });
  };
  return { baseURL, port, stop };
}

export function sineWavBlob(): Blob {
  const sampleRate = 16_000;
  const samples = sampleRate / 2;
  const pcm = new Uint8Array(44 + samples * 2);
  const view = new DataView(pcm.buffer);
  const ascii = (offset: number, s: string): void => {
    for (let i = 0; i < s.length; i++) pcm[offset + i] = s.charCodeAt(i);
  };
  ascii(0, "RIFF");
  view.setUint32(4, 36 + samples * 2, true);
  ascii(8, "WAVEfmt ");
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true);
  view.setUint16(22, 1, true);
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * 2, true);
  view.setUint16(32, 2, true);
  view.setUint16(34, 16, true);
  ascii(36, "data");
  view.setUint32(40, samples * 2, true);
  for (let i = 0; i < samples; i++) {
    view.setInt16(44 + i * 2, Math.round(0.3 * 32767 * Math.sin((2 * Math.PI * 440 * i) / sampleRate)), true);
  }
  return new Blob([pcm], { type: "audio/wav" });
}
