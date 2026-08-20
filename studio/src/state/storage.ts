import { uuid } from '../lib/util';
import type { PastRecord } from './store';

export type StorageKind = 'opfs' | 'local';

export interface StoreItem { t: number; intent: number | null; kind: string; text: string }

const PAST_KEY = 'nur.past.v2';
const STORE_KEY = 'nur.store.v1';
const PATCH_PREFIX = 'nur.patch.v1.';

const strip = (k: string, v: unknown): unknown =>
  (k === 'url' || k === 'blob' || k === 'peaks' || k === 'src' || k === '_decoding' || k === 'chunks') ? undefined : v;

const lsGet = <T>(key: string): T | null => {
  try {
    const raw = localStorage.getItem(key);
    return raw ? (JSON.parse(raw) as T) : null;
  } catch { return null; }
};
const lsPast = (): PastRecord[] => lsGet<PastRecord[]>(PAST_KEY) || [];

function detectKind(): StorageKind {
  try { if (localStorage.getItem('nur.storage.force') === 'local') return 'local'; } catch {  }
  try {
    if (typeof location !== 'undefined' && /[?&]storage=local(&|$)/.test(location.search)) return 'local';
    if (typeof isSecureContext !== 'undefined' && !isSecureContext) return 'local';
    if (typeof navigator === 'undefined' || !navigator.storage || typeof navigator.storage.getDirectory !== 'function') return 'local';
    if (typeof FileSystemFileHandle === 'undefined' || !('createWritable' in FileSystemFileHandle.prototype)) return 'local';
    return 'opfs';
  } catch { return 'local'; }
}

let root: FileSystemDirectoryHandle | null = null;

async function getDir(path: string): Promise<FileSystemDirectoryHandle | null> {
  if (!root) return null;
  let dir = root;
  try { for (const p of path.split('/')) dir = await dir.getDirectoryHandle(p); } catch { return null; }
  return dir;
}
async function dirOf(path: string, create: boolean): Promise<{ dir: FileSystemDirectoryHandle; name: string } | null> {
  if (!root) return null;
  const parts = path.split('/');
  let dir = root;
  try { for (const p of parts.slice(0, -1)) dir = await dir.getDirectoryHandle(p, { create }); } catch { return null; }
  const name = parts[parts.length - 1];
  return name ? { dir, name } : null;
}
async function writeRaw(path: string, data: string | Blob): Promise<void> {
  const loc = await dirOf(path, true);
  if (!loc) throw new Error('opfs unavailable: ' + path);
  const fh = await loc.dir.getFileHandle(loc.name, { create: true });
  const w = await fh.createWritable();
  await w.write(data);
  await w.close();
}
async function readRaw(path: string): Promise<File | null> {
  const loc = await dirOf(path, false);
  if (!loc) return null;
  try { return await (await loc.dir.getFileHandle(loc.name)).getFile(); } catch { return null; }
}
async function removeRaw(path: string, recursive = false): Promise<void> {
  const loc = await dirOf(path, false);
  if (!loc) return;
  try { await loc.dir.removeEntry(loc.name, { recursive }); } catch {  }
}
const existsRaw = async (path: string): Promise<boolean> => (await readRaw(path)) != null;
const readJson = async (path: string): Promise<unknown> => {
  const f = await readRaw(path);
  try { return f ? JSON.parse(await f.text()) : null; } catch { return null; }
};

const tails = new Map<string, Promise<void>>();
function enqueue(key: string, fn: () => Promise<void>): Promise<void> {
  const next = (tails.get(key) || Promise.resolve())
    .then(() => storage.ready)
    .then(fn)
    .catch(() => {  });
  tails.set(key, next);
  return next;
}
async function flush(): Promise<void> {
  await storage.ready;
  await Promise.all([...tails.values()]);
  await Promise.all([...tails.values()]);
}

function saveDoc(lsKey: string, path: string, body: string | null): void {
  void enqueue(path, async () => {
    if (storage.kind === 'local') {
      try {
        if (body == null) localStorage.removeItem(lsKey);
        else localStorage.setItem(lsKey, body);
      } catch {  }
    } else if (body == null) await removeRaw(path);
    else await writeRaw(path, body);
  });
}

const patchCache = new Map<string, unknown>();
let artifactCache: StoreItem[] = [];
const audioStored = new Map<string, Set<number>>();

function persistLocalPast(all: PastRecord[]): PastRecord[] {
  let list = all.length > 50 ? all.slice(0, 50) : all;
  try { localStorage.setItem(PAST_KEY, JSON.stringify(list, strip)); }
  catch {
    list = list.slice(0, Math.ceil(list.length / 2));
    try { localStorage.setItem(PAST_KEY, JSON.stringify(list, strip)); } catch {  }
  }
  return list;
}

async function migrateLegacy(): Promise<void> {
  try {
    const raw = localStorage.getItem(PAST_KEY);
    if (raw != null) {
      try {
        const list = (JSON.parse(raw) as PastRecord[] | null) || [];
        const now = Date.now();
        for (let i = 0; i < list.length; i++) {
          const rec = list[i];
          if (!rec) continue;
          if (!rec.id) rec.id = uuid();
          if (rec.savedAt == null) rec.savedAt = now - i;
          if (!(await existsRaw(`sessions/${rec.id}.json`)))
            await writeRaw(`sessions/${rec.id}.json`, JSON.stringify(rec, strip));
        }
      } catch {}
      localStorage.removeItem(PAST_KEY);
    }
    const patchKeys: string[] = [];
    for (let i = 0; i < localStorage.length; i++) {
      const k = localStorage.key(i);
      if (k && k.startsWith(PATCH_PREFIX)) patchKeys.push(k);
    }
    for (const k of patchKeys) {
      const v = localStorage.getItem(k);
      const intent = k.slice(PATCH_PREFIX.length);
      if (v != null && !(await existsRaw(`patches/${intent}.json`))) await writeRaw(`patches/${intent}.json`, v);
      localStorage.removeItem(k);
    }
    const vs = localStorage.getItem(STORE_KEY);
    if (vs != null) {
      if (!(await existsRaw('artifacts.json'))) await writeRaw('artifacts.json', vs);
      localStorage.removeItem(STORE_KEY);
    }
  } catch {  }
}

async function hydrateCaches(): Promise<void> {
  if (storage.kind === 'opfs') {
    const pd = await getDir('patches');
    if (pd) {
      for await (const [name, h] of pd.entries()) {
        if (h.kind !== 'file' || !name.endsWith('.json')) continue;
        try {
          const f = await (h as FileSystemFileHandle).getFile();
          patchCache.set(name.slice(0, -5), JSON.parse(await f.text()));
        } catch {  }
      }
    }
    const v = await readJson('artifacts.json');
    if (Array.isArray(v)) artifactCache = v as StoreItem[];
    return;
  }
  try {
    for (let i = 0; i < localStorage.length; i++) {
      const k = localStorage.key(i);
      if (!k || !k.startsWith(PATCH_PREFIX)) continue;
      const v = localStorage.getItem(k);
      if (v == null) continue;
      try { patchCache.set(k.slice(PATCH_PREFIX.length), JSON.parse(v)); } catch {  }
    }
    const v = localStorage.getItem(STORE_KEY);
    if (v != null) {
      const a = JSON.parse(v) as StoreItem[] | null;
      if (Array.isArray(a)) artifactCache = a;
    }
  } catch {  }
}

async function init(): Promise<void> {
  if (storage.kind === 'opfs') {
    try { root = await navigator.storage.getDirectory(); } catch { root = null; }
    if (!root) storage.kind = 'local';
  }
  if (storage.kind === 'opfs') await migrateLegacy();
  await hydrateCaches();
}

const sessions = {
  async load(): Promise<PastRecord[]> {
    await storage.ready;
    if (storage.kind === 'local') return lsPast();
    const sd = await getDir('sessions');
    if (!sd) return [];
    const out: PastRecord[] = [];
    for await (const [name, h] of sd.entries()) {
      if (h.kind !== 'file' || !name.endsWith('.json')) continue;
      try {
        const rec = JSON.parse(await (await (h as FileSystemFileHandle).getFile()).text()) as PastRecord | null;
        if (rec && rec.id) out.push(rec);
      } catch {  }
    }
    out.sort((a, b) => (b.savedAt || 0) - (a.savedAt || 0));
    return out;
  },
  put(rec: PastRecord, all: PastRecord[]): PastRecord[] {
    rec.savedAt = Date.now();
    if (storage.kind === 'local') return persistLocalPast(all);
    const id = rec.id;
    rec.msgs.forEach((m, i) => {
      if (m.kind !== 'audio' || !m.blob) return;
      m.blobType = m.blob.type || m.blobType;
      const set = audioStored.get(id) || new Set<number>();
      if (set.has(i)) return;
      set.add(i);
      audioStored.set(id, set);
      const blob = m.blob;
      void enqueue(`sessions/${id}`, async () => {
        try { await writeRaw(`sessions/${id}/audio-${i}.wav`, blob); }
        catch (e) { set.delete(i); throw e; }
      });
    });
    const body = JSON.stringify(rec, strip);
    void enqueue(`sessions/${id}`, () => writeRaw(`sessions/${id}.json`, body));
    return all;
  },
  remove(id: string, all: PastRecord[]): PastRecord[] {
    if (storage.kind === 'local') return persistLocalPast(all);
    audioStored.delete(id);
    void enqueue(`sessions/${id}`, async () => {
      await removeRaw(`sessions/${id}.json`);
      await removeRaw(`sessions/${id}`, true);
    });
    return all;
  },
  persistAll(all: PastRecord[]): PastRecord[] {
    if (storage.kind === 'local') return persistLocalPast(all);
    all.forEach(rec => sessions.put(rec, all));
    return all;
  },
  async audio(id: string, msgIndex: number): Promise<Blob | null> {
    await storage.ready;
    if (storage.kind === 'local') return null;
    return readRaw(`sessions/${id}/audio-${msgIndex}.wav`);
  },
  markAudioStored(id: string, msgIndex: number): void {
    const set = audioStored.get(id) || new Set<number>();
    set.add(msgIndex);
    audioStored.set(id, set);
  },
};

const patches = {
  get: (intent: number | string): unknown => patchCache.get(String(intent)) ?? null,
  set(intent: number | string, doc: unknown): void {
    const key = String(intent);
    patchCache.set(key, doc);
    saveDoc(PATCH_PREFIX + key, `patches/${key}.json`, JSON.stringify(doc));
  },
  remove(intent: number | string): void {
    const key = String(intent);
    patchCache.delete(key);
    saveDoc(PATCH_PREFIX + key, `patches/${key}.json`, null);
  },
};

const artifacts = {
  read: (): StoreItem[] => artifactCache,
  write(items: StoreItem[]): void {
    artifactCache = items;
    saveDoc(STORE_KEY, 'artifacts.json', JSON.stringify(items));
  },
};

const debug = {
  async readPatch(intent: number | string): Promise<unknown> {
    await flush();
    if (storage.kind === 'local') return lsGet(PATCH_PREFIX + String(intent));
    return readJson(`patches/${intent}.json`);
  },
  async readSession(id: string): Promise<PastRecord | null> {
    await flush();
    if (storage.kind === 'local') return lsPast().find(r => r && r.id === id) || null;
    return (await readJson(`sessions/${id}.json`)) as PastRecord | null;
  },
  async listSessionIds(): Promise<string[]> {
    await flush();
    if (storage.kind === 'local') return lsPast().map(r => r && r.id).filter((x): x is string => !!x);
    const sd = await getDir('sessions');
    if (!sd) return [];
    const ids: string[] = [];
    for await (const [name, h] of sd.entries())
      if (h.kind === 'file' && name.endsWith('.json')) ids.push(name.slice(0, -5));
    return ids;
  },
  async hasAudio(id: string, msgIndex: number): Promise<boolean> {
    await flush();
    if (storage.kind === 'local') return false;
    return existsRaw(`sessions/${id}/audio-${msgIndex}.wav`);
  },
  async hasSessionDir(id: string): Promise<boolean> {
    await flush();
    if (storage.kind === 'local') return false;
    return !!(await getDir(`sessions/${id}`));
  },
  async readArtifacts(): Promise<StoreItem[]> {
    await flush();
    if (storage.kind === 'local') return lsGet<StoreItem[]>(STORE_KEY) || [];
    return ((await readJson('artifacts.json')) as StoreItem[] | null) || [];
  },
};

export const storage = {
  kind: detectKind(),
  ready: Promise.resolve(),
  flush,
  sessions,
  patches,
  artifacts,
  debug,
};
storage.ready = init();
