import { bus } from '../state/bus';
import {
  S, setState, scheduleRender, notify, type Msg, type PastRecord, type Screen, type Turn, type View,
} from '../state/store';
import { INTENTS, progFor, slugOf, ttsKind, voicesForModel } from '../data';
import {
  clamp, errMsg, pad, stopTracks, tryPause, tryPlay, uuid, OFFLINE_MSG,
} from '../lib/util';
import { attachAudioMeta, grabFrame, mergeWavBlobs } from './media';
import { capChunk, normalizeForSpeech, sentences, stripMarkdownEmphasis } from '../lib/punkt';
import { nurClientFor, postOcr } from '../api/client';
import type {
  ChatCompletionChunk, ChatMessageIn, ListModelsResponse, Model,
  TranscriptionDiarizedJsonResponse, TranscriptionJsonResponse, TranscriptionVerboseJsonResponse,
} from 'nur-client';
import { navTo } from './nav';
import { openRealtime, realtimeEnabled, type RealtimeSession } from './realtime';
import { storage } from '../state/storage';
import type { StoreItem } from '../state/storage';
export type { StoreItem } from '../state/storage';

try { localStorage.removeItem('nur.past.v1'); } catch {  }
let lastSttMs: number | null = null;

export type Mode = 'offline' | 'live';
let mode: Mode = 'offline';
export const getMode = (): Mode => mode;

export interface ModelRegistry { all: string[]; chat: string[]; stt: string[]; tts: string[]; emb: string[] }
export const models: ModelRegistry = { all: [], chat: [], stt: [], tts: [], emb: [] };

function classify(ids: string[]): Pick<ModelRegistry, 'chat' | 'stt' | 'tts' | 'emb'> {
  const stt = ids.filter(id => /whisper|^asr$|^asr-|^stt$|^stt-/i.test(id));
  const tts = ids.filter(id => /^kokoro|^qwen3-tts|^tts$|^tts-/i.test(id));
  const emb = ids.filter(id => /embed/i.test(id));
  const chat = ids.filter(id => stt.indexOf(id) < 0 && tts.indexOf(id) < 0 &&
    emb.indexOf(id) < 0 && !/ocr|rerank|pii|diariz/i.test(id));
  return { chat, stt, tts, emb };
}

// A base may point at a foreign OpenAI-compatible server whose model rows lack
// the nur extras, so only `id` is trusted beyond the OpenAI core keys.
type ProxiedModel = Pick<Model, 'id' | 'object' | 'created' | 'owned_by'> &
  Partial<Omit<Model, 'id' | 'object' | 'created' | 'owned_by'>>;
type ProxiedListModelsResponse = { object: 'list'; data: Array<ProxiedModel> };

async function fetchModels(signal: AbortSignal, base?: string | null): Promise<ProxiedListModelsResponse | null> {
  try {
    const j: ListModelsResponse = await nurClientFor(base).models.list(undefined, { signal });
    return j as ProxiedListModelsResponse;
  } catch {
    return null;
  }
}

const BASE_ERR_MSG = 'fetch failed — endpoint unreachable or not CORS-open';
export interface BaseReg { ids: string[]; chat: string[]; stt: string[]; tts: string[]; error: string | null }
const baseCache: Record<string, BaseReg> = {};
const basePending: Record<string, Promise<BaseReg>> = {};
export const normBase = (s: string): string => s.trim().replace(/\/+$/, '');
export function ensureBaseModels(base: string): Promise<BaseReg> {
  const b = normBase(base);
  const cached = baseCache[b];
  if (cached && !cached.error) return Promise.resolve(cached);
  const pending = basePending[b];
  if (pending) return pending;
  const p = (async (): Promise<BaseReg> => {
    let reg: BaseReg;
    try {
      const j = await fetchModels(AbortSignal.timeout(8000), b);
      const ids = j ? (j.data || []).map(m => m.id).filter(Boolean) : [];
      if (!ids.length) throw new Error('no models');
      const c = classify(ids);
      reg = { ids, chat: c.chat, stt: c.stt, tts: c.tts, error: null };
    } catch {
      reg = { ids: [], chat: [], stt: [], tts: [], error: BASE_ERR_MSG };
    }
    baseCache[b] = reg;
    delete basePending[b];
    bus.emit('endpoint.models', { base: b, models: reg.ids.length, error: reg.error });
    scheduleRender();
    return reg;
  })();
  basePending[b] = p;
  return p;
}

export type SvcKind = 'chat' | 'stt' | 'tts';
const baseKeyOf: Record<SvcKind, 'baseChat' | 'baseStt' | 'baseTts'> =
  { chat: 'baseChat', stt: 'baseStt', tts: 'baseTts' };
export interface EffModels { list: string[]; base: string; error: string | null; pending: boolean }
export function effModels(kind: SvcKind): EffModels {
  const base = normBase(S[baseKeyOf[kind]] || '');
  if (!base) return { list: models[kind], base: '', error: null, pending: false };
  const c = baseCache[base];
  if (!c) return { list: [], base, error: null, pending: true };
  return { list: c[kind], base, error: c.error, pending: false };
}
export const modelRefusal = (kind: SvcKind): string | null => {
  const eff = effModels(kind);
  if (!eff.base) return null;
  return eff.error ? eff.error : eff.pending ? 'endpoint models not loaded yet — check the ' + kind + ' panel' : null;
};

export interface ModelRow { id: string; note: string | null }
const ROLE_ALIAS: Record<SvcKind, RegExp> = {
  stt: /^(stt|asr)(-default)?$|^whisper(-1)?$/i,
  tts: /^tts(-default)?$/i,
  chat: /^(llm|chat)(-default)?$/i,
};
const famKey = (id: string): string =>
  id.toLowerCase().replace(/^[^/]+\//, '').replace(/^faster-/, '').replace(/-ct2$/, '');
export function displayModels(kind: SvcKind, list: string[], selId: string | null): { rows: ModelRow[]; total: number } {
  const alias: string[] = [], rest: string[] = [];
  list.forEach(id => (ROLE_ALIAS[kind].test(id) ? alias : rest).push(id));
  const fams = new Map<string, string[]>();
  rest.forEach(id => {
    const k = famKey(id);
    fams.set(k, [...(fams.get(k) || []), id]);
  });
  const rows: ModelRow[] = [];
  const kindDefault: Record<SvcKind, string> = { stt: 'stt-default', tts: 'tts-default', chat: 'llm-default' };
  const a0 = (selId && alias.indexOf(selId) >= 0 ? selId : null)
    || alias.find(a => a.toLowerCase() === kindDefault[kind])
    || alias.find(a => /-default$/i.test(a))
    || alias[0];
  if (a0) rows.push({ id: a0, note: 'backend default' + (alias.length > 1 ? ' · ' + alias.length + ' ids' : '') });
  [...fams.entries()].sort((a, b) => (a[0] < b[0] ? -1 : 1)).forEach(([k, ids]) => {
    const rep = ids.find(id => id.toLowerCase() === k) || ids.slice().sort((x, y) => x.length - y.length)[0];
    if (rep) rows.push({ id: rep, note: ids.length > 1 ? ids.length + ' ids' : null });
  });
  if (selId && list.indexOf(selId) >= 0 && !rows.some(r => r.id === selId))
    rows.push({ id: selId, note: 'selected alias' });
  return { rows, total: list.length };
}

const pickModel = (list: string[], sel: string | null, prefs: string[]): string | null => {
  if (sel && list.indexOf(sel) >= 0) return sel;
  for (const p of prefs) if (list.indexOf(p) >= 0) return p;
  return list[0] || null;
};
export const chatModel = (): string | null => pickModel(effModels('chat').list, S.selChat, ['llm-default']);
export const sttModel = (): string | null => pickModel(effModels('stt').list, S.selStt, ['stt-default']);
export const ttsModel = (): string | null => pickModel(effModels('tts').list, S.selTts, ['kokoro', 'tts-default']);
export function ttsVoice(): string | null {
  const id = ttsModel(), kind = ttsKind(id);
  if (kind === 'design') return (S.vdText || '').trim() || null;
  const list = voicesForModel(id);
  if (list) return S.voice != null && list.indexOf(S.voice) >= 0 ? S.voice : (list[0] ?? null);
  return (S.voice || '').trim() || null;
}

export const perf: { tok: Record<string, { n: number; avg: number }>; stt: number | null } = { tok: {}, stt: null };
export function recordTokRate(id: string, msPerTok: number): void {
  const r = perf.tok[id] || (perf.tok[id] = { n: 0, avg: 0 });
  r.avg = (r.avg * r.n + msPerTok) / (r.n + 1);
  r.n++;
}
export const tokRateLabel = (id: string | null): string => {
  const r = id ? perf.tok[id] : undefined;
  return r ? r.avg.toFixed(1) + ' ms/tok' : '—';
};

export const nurStore = {
  read: (): StoreItem[] => storage.artifacts.read(),
  write(items: StoreItem[]): void { storage.artifacts.write(items.slice(-200)); },
  add(kind: string, text: string): number {
    const items = this.read().slice();
    items.push({ t: Date.now(), intent: S.intent, kind, text: String(text).slice(0, 4000) });
    this.write(items);
    return items.length;
  },
  clear(): void { this.write([]); },
};

export async function loadPast(): Promise<void> {
  let loaded: PastRecord[] = [];
  try { loaded = await storage.sessions.load(); } catch { loaded = []; }
  const pre = S.past;
  S.past = pre.length ? pre.concat(loaded.filter(r => !pre.some(p => p.id === r.id))) : loaded;
  let assigned = false;
  S.past.forEach(ps => { if (!ps.id) { ps.id = uuid(); assigned = true; } });
  if (assigned) persistPast();
  setState({ hydrated: true });
}
function trimPast(): void {
  if (S.past.length <= 50) return;
  const dropped = S.past.slice(50);
  S.past = S.past.slice(0, 50);
  if (storage.kind === 'opfs') dropped.forEach(p => { S.past = storage.sessions.remove(p.id, S.past); });
}
export function persistPast(): void {
  trimPast();
  S.past = storage.sessions.persistAll(S.past);
}

export function appendLog(line: string): void {
  S.log.push(line);
  bus.emit('log.line', { line });
}

export function setBusy(label: string | null): void {
  S.busy = label;
  if (label) bus.emit('busy', { label });
  scheduleRender();
}

let scrollFlag = false;
export function scrollChatSoon(): void { scrollFlag = true; scheduleRender(); }
export function consumeScrollFlag(): boolean { const f = scrollFlag; scrollFlag = false; return f; }

export function progHas(kind: string): boolean {
  const scan = (items: typeof S.prog): boolean =>
    items.some(b => b && (b.k === kind || (b.children ? scan(b.children) : false)));
  return scan(S.prog) || S.stacks.some(st => scan(st.items));
}

export function sessionUrl(id: string, view: View = 'chat'): string {
  return `/s/${id}/${view === 'insp' ? 'inspector' : view}`;
}

function activeRecord(): PastRecord | null {
  if (S.screen !== 'duet' || !S.msgs.length) return null;
  const firstUser = S.msgs.find(m => m.role === 'u') || S.msgs[0];
  return {
    id: S.sessionId || uuid(),
    intent: S.intent,
    title: S.intent != null ? (INTENTS[S.intent]?.name ?? 'Session') : 'Session',
    summary: ((firstUser && firstUser.text) || '').slice(0, 110),
    turns: S.turns, sel: S.sel,
    msgs: S.msgs, log: S.log, prog: S.prog, stacks: S.stacks,
  };
}

function upsertPast(rec: PastRecord): void {
  S.past = [rec].concat(S.past.filter(p => p.id !== rec.id));
  trimPast();
  S.past = storage.sessions.put(rec, S.past);
}

let saveQueued = false;
function autosaveSoon(): void {
  if (saveQueued) return;
  saveQueued = true;
  setTimeout(() => {
    saveQueued = false;
    const rec = activeRecord();
    if (rec) upsertPast(rec);
  }, 400);
}
bus.on('msg.user', autosaveSoon);
bus.on('agent.done', autosaveSoon);
bus.on('tts.done', autosaveSoon);
bus.on('run.done', autosaveSoon);
bus.on('turn.recorded', autosaveSoon);

export function leaveSession(): void {
  const rec = activeRecord();
  if (rec) {
    upsertPast(rec);
    bus.emit('session.saved', { title: rec.title, turns: rec.turns.length });
  }
  bus.emit('session.leave', {});
  abortChat();
  stopAudio();
  setBusy(null);
  setState({ screen: 'intent', view: 'chat', sessionId: null, selNode: null, selPath: null, msgs: [], log: [], turns: [], sel: 0, msgInfo: {}, viewer: null, txtModal: null });
}

export function goHome(): Promise<void> {
  bus.emit('nav.home', {});
  return navTo('/');
}

export function applyIntent(i: number, screen: Screen = 'patch'): void {
  const presets: Partial<typeof S> = {
    sttFormat: i === 3 ? 2 : 0,
    fps: i === 10 ? 2 : undefined,
  };
  bus.emit('session.leave', {});
  stopAudio();
  setState(Object.assign({
    intent: i, screen, notice: null, sessionId: null,
    msgs: [], log: [], turns: [], sel: 0, dslMode: false,
    selNode: null, selPath: null, prog: progFor(i), stacks: [], msgInfo: {},
    viewer: null, txtModal: null,
  }, presets));
  bus.emit('intent.selected', { intent: i, name: INTENTS[i]?.name ?? '' });
}

export function pickIntent(i: number): Promise<void> {
  const it = INTENTS[i];
  if (!it) return Promise.resolve();
  if (mode !== 'live') {
    applyIntent(i);
    return navTo('/patch/' + slugOf(it.name));
  }
  applyIntent(i, 'intent');
  return stepDuet();
}

function rehydrateAudio(id: string, msgs: Msg[]): void {
  msgs.forEach((m, i) => {
    if (m.kind !== 'audio' || m.blob || m.url) return;
    storage.sessions.audio(id, i).then(b => {
      if (!b) return;
      const blob = m.blobType && b.type !== m.blobType ? new Blob([b], { type: m.blobType }) : b;
      m.blob = blob;
      m.url = URL.createObjectURL(blob);
      storage.sessions.markAudioStored(id, i);
      attachAudioMeta(m);
      scheduleRender();
    }).catch(() => {  });
  });
}

export function restoreSessionById(id: string, view: View = 'chat'): boolean {
  const idx = S.past.findIndex(p => p.id === id);
  if (idx < 0) return false;
  const ps = S.past[idx];
  if (!ps) return false;
  S.past = S.past.filter((_, j) => j !== idx);
  if (storage.kind === 'local') persistPast();
  const turns = Array.isArray(ps.turns) ? ps.turns : [];
  setState({
    screen: 'duet', view, intent: ps.intent, sessionId: id, notice: null,
    msgs: ps.msgs || [], log: ps.log || [], prog: ps.prog || [], stacks: ps.stacks || [],
    turns, sel: clamp(ps.sel || 0, 0, Math.max(turns.length - 1, 0)),
    selNode: null, selPath: null, msgInfo: {}, viewer: null, txtModal: null,
  });
  bus.emit('session.resumed', { title: ps.title, turns: turns.length });
  if (storage.kind === 'opfs') rehydrateAudio(id, ps.msgs || []);
  scrollChatSoon();
  return true;
}

export function resumeSession(idx: number): Promise<void> {
  const ps = S.past[idx];
  if (!ps) return Promise.resolve();
  return navTo(sessionUrl(ps.id));
}

export function deletePastSession(id: string): void {
  S.past = S.past.filter(p => p.id !== id);
  S.past = storage.sessions.remove(id, S.past);
  bus.emit('session.deleted', { id });
  scheduleRender();
}

export function stepDuet(): Promise<void> {
  if (S.intent == null || mode !== 'live') return Promise.resolve();
  if (!S.sessionId) setState({ sessionId: uuid() }, { silent: true });
  if (!S.log.length) appendLog('session started · ' + (INTENTS[S.intent]?.name ?? ''));
  bus.emit('session.start', { intent: S.intent, name: INTENTS[S.intent]?.name ?? '', mode, id: S.sessionId });
  scrollChatSoon();
  return navTo(sessionUrl(S.sessionId as string));
}

let audioEl: HTMLAudioElement | null = null;
let audioMsg: number | null = null;
export function stopAudio(): void {
  tryPause(audioEl);
  audioEl = null;
  audioMsg = null;
  const pipe = livePipe;
  livePipe = null;
  if (pipe) pipe.stop();
  if (S.playId != null) bus.emit('audio.stop', { msg: S.playId });
  setState({ playId: null, playT: 0 });
}
export function togglePlay(i: number): void {
  if (S.playId === i) {
    tryPause(audioEl);
    bus.emit('audio.pause', { msg: i });
    setState({ playId: null });
    return;
  }
  const m = S.msgs[i];
  if (!m || !m.url) return;
  if (audioEl && audioMsg === i) {
    tryPlay(audioEl);
    bus.emit('audio.play', { msg: i, resume: true });
    setState({ playId: i });
    return;
  }
  stopAudio();
  try {
    audioEl = new Audio(m.url);
    audioMsg = i;
    tryPlay(audioEl);
  } catch { audioEl = null; audioMsg = null; return; }
  bus.emit('audio.play', { msg: i });
  setState({ playId: i, playT: 0 });
  audioEl.ontimeupdate = () => {
    S.playT = audioEl && audioEl.duration ? audioEl.currentTime / audioEl.duration : 0;
    updateProgress();
  };
  audioEl.onended = () => stopAudio();
}
export interface RunPlayback { el: HTMLAudioElement; done: Promise<string>; stop: () => void }
export function playForRun(url: string): RunPlayback {
  stopAudio();
  const el = new Audio(url);
  audioEl = el;
  const done = new Promise<string>(res => {
    el.onended = () => res('ended');
    el.onpause = () => res('stopped');
    el.onerror = () => res('error');
  });
  tryPlay(el);
  return { el, done, stop: () => tryPause(el) };
}
export function updateProgress(): void {
  document.querySelectorAll<HTMLElement>('[data-prog]').forEach(el => {
    el.style.width = (S.playId != null && Number(el.getAttribute('data-prog')) === S.playId ? S.playT * 100 : 0) + '%';
  });
}

export async function detectMode(): Promise<void> {
  for (let attempt = 0; attempt < 2 && mode !== 'live'; attempt++) try {
    const ctl = new AbortController();
    const to = setTimeout(() => ctl.abort(), 4000);
    const j = await fetchModels(ctl.signal);
    clearTimeout(to);
    if (j) {
      const ids = (j.data || []).map(m => m.id).filter(Boolean);
      if (ids.length) {
        mode = 'live';
        const c = classify(ids);
        models.all = ids;
        models.chat = c.chat; models.stt = c.stt; models.tts = c.tts; models.emb = c.emb;
      }
    }
  } catch {}
  bus.emit('mode.detected', { mode, models: models.all.length });
  if (mode !== 'live') setTimeout(() => { void detectMode(); }, 10000);
  scheduleRender();
}

function chatHistory(upto?: number | null): ChatMessageIn[] {
  return S.msgs.slice(0, upto == null ? S.msgs.length : upto)
    .filter(m => m.text && (m.role === 'u' || m.role === 'a') && m.kind !== 'visual' && m.text[0] !== '(')
    .map(m => ({ role: m.role === 'u' ? 'user' : 'assistant', content: m.text.replace(/^“|”$/g, '') }));
}

export interface StreamChatOpts {
  model: string;
  messages: ChatMessageIn[];
  onToken?: (text: string) => void;
  onReasoning?: (reasoning: string) => void;
}
export interface StreamResult {
  reasoning: string;
  chunks: ChatCompletionChunk[];
  finish?: string;
  reason1?: number;
  reasonChars?: number;
  text: string;
  tok1: number | null;
  wall: number;
  msPerTok?: number;
  error?: string;
}
let chatAbort: AbortController | null = null;
export const chatStreaming = (): boolean => chatAbort != null;
export function abortChat(): void {
  if (!chatAbort) return;
  bus.emit('agent.abort', {});
  chatAbort.abort();
}

export async function streamChat(opts: StreamChatOpts): Promise<StreamResult> {
  const t0 = performance.now();
  let text = '', tok1: number | null = null, nTok = 0, tLast: number | null = null;
  const out: StreamResult = { reasoning: '', chunks: [], text: '', tok1: null, wall: 0 };
  const ctl = new AbortController();
  chatAbort = ctl;
  try {
    const stream = await nurClientFor(S.baseChat).chat.completions.create({
      model: opts.model, stream: true, temperature: S.temp, top_p: S.topP,
      max_tokens: 2048, messages: opts.messages,
      ...(S.effort && S.effort !== 'default'
        ? { chat_template_kwargs: { reasoning_effort: S.effort } }
        : {}),
    }, { signal: ctl.signal });
    for await (const d of stream) {
      const ch = d.choices && d.choices[0];
      if (ch && ch.finish_reason) out.finish = ch.finish_reason;
      const delta = (ch && ch.delta) || {};
      if (out.chunks.length < 4000) out.chunks.push(d);
      if (delta.reasoning_content) {
        out.reasoning += delta.reasoning_content;
        out.reasonChars = out.reasoning.length;
        if (!out.reason1) {
          out.reason1 = Math.round(performance.now() - t0);
          bus.emit('agent.reasoning', { ms: out.reason1 });
          if (S.busy && !/ · thinking…$/.test(S.busy)) setBusy(S.busy + ' · thinking…');
        }
        if (opts.onReasoning) opts.onReasoning(out.reasoning);
      }
      if (delta.content) {
        const now = performance.now();
        if (tok1 == null) {
          tok1 = Math.round(now - t0);
          bus.emit('agent.first_token', { ms: tok1 });
          if (S.busy && / · thinking…$/.test(S.busy)) setBusy(S.busy.replace(/ · thinking…$/, ''));
        }
        nTok++; tLast = now;
        text += delta.content;
        bus.emit('agent.token', { token: delta.content });
        if (opts.onToken) opts.onToken(text);
      }
    }
  } catch (e) {
    if (ctl.signal.aborted) out.finish = 'aborted';
    else out.error = errMsg(e);
  } finally {
    if (chatAbort === ctl) chatAbort = null;
  }
  out.text = text;
  out.tok1 = tok1;
  out.wall = Math.round(performance.now() - t0);
  if (tok1 != null && tLast != null && nTok > 4 && tLast - (t0 + tok1) > 50) {
    out.msPerTok = (tLast - (t0 + tok1)) / (nTok - 1);
    recordTokRate(opts.model, out.msPerTok);
  }
  return out;
}

export type TranscriptionResponse =
  | TranscriptionJsonResponse
  | TranscriptionVerboseJsonResponse
  | TranscriptionDiarizedJsonResponse;
export interface TranscribeResult {
  text: string; ms: number; model: string; detail: string | null; raw: TranscriptionResponse;
}
export async function transcribe(blob: Blob, opts?: { format?: number; quiet?: boolean }): Promise<TranscribeResult> {
  const model = sttModel();
  if (!model) throw new Error(modelRefusal('stt') || 'no stt model available');
  const stt = nurClientFor(S.baseStt).audio.transcriptions;
  const fmt = opts && opts.format != null ? opts.format : S.sttFormat;
  const t0 = performance.now();
  let detail: string | null = null;
  let raw: TranscriptionResponse;
  if (fmt === 2) {
    const j = await stt.create({ file: blob, fileName: 'turn.webm', model, response_format: 'diarized_json' });
    raw = j;
    if (Array.isArray(j.segments)) {
      detail = j.segments.map((s: { speaker?: string | number | null; text?: string }) =>
        (s.speaker != null ? '[' + s.speaker + '] ' : '') + ((s.text || '').trim())).filter(Boolean).join('\n');
    }
  } else if (fmt === 1) {
    const j = await stt.create({ file: blob, fileName: 'turn.webm', model, response_format: 'verbose_json' });
    raw = j;
    if (Array.isArray(j.words)) detail = j.words.length + ' word timestamps';
  } else {
    raw = await stt.create({ file: blob, fileName: 'turn.webm', model, response_format: 'json' });
  }
  const ms = Math.round(performance.now() - t0);
  const text = (raw.text || '').trim();
  if (!(opts && opts.quiet)) perf.stt = ms;
  return { text, ms, model, detail, raw };
}

export interface SpeakResult { blob: Blob; url: string; ms: number; model: string; voice: string | null }

export interface SpeechPipeline {
  autoplay: boolean;
  push: (textSoFar: string) => void;
  end: (finalText: string) => void;
  first: () => Promise<number | null>;
  result: Promise<SpeakResult | null>;
  playbackDone: Promise<void>;
  stop: () => void;
  error: () => string | null;
}

let livePipe: SpeechPipeline | null = null;
export const speechActive = (): boolean => livePipe != null;

export function speechPipeline(opts: { autoplay: boolean }): SpeechPipeline {
  const model = ttsModel();
  if (!model) throw new Error(modelRefusal('tts') || 'no tts model available');
  const voice = ttsVoice();
  stopAudio();
  const t0 = performance.now();
  const chunks: string[] = [];
  const blobs: Blob[] = [];
  let committed = 0, lastSegLen = 0, synthCursor = 0, playCursor = 0;
  let ended = false, stopped = false, finished = false, synthRunning = false, playRunning = false;
  let firstMs: number | null = null, err: string | null = null;
  let curEl: HTMLAudioElement | null = null;
  let resolveFirst: (v: number | null) => void = () => {};
  const firstP = new Promise<number | null>(r => { resolveFirst = r; });
  let resolveResult: (v: SpeakResult | null) => void = () => {};
  const result = new Promise<SpeakResult | null>(r => { resolveResult = r; });
  let resolvePlayback: () => void = () => {};
  const playbackDone = new Promise<void>(r => { resolvePlayback = r; });

  const synthIdle = (): boolean => !synthRunning && synthCursor >= chunks.length;
  const unregister = (): void => {
    if (livePipe === pipe) { livePipe = null; scheduleRender(); }
  };
  const checkPlayback = (): void => {
    if (stopped || (!opts.autoplay && finished) || (ended && finished && !playRunning && playCursor >= blobs.length)) {
      resolvePlayback();
      unregister();
    }
  };
  const finish = async (): Promise<void> => {
    if (finished || (!stopped && !(ended && synthIdle()))) return;
    finished = true;
    if (firstMs == null) resolveFirst(null);
    if (!blobs.length) {
      resolveResult(null);
    } else {
      const merged = await mergeWavBlobs(blobs);
      const ms = Math.round(performance.now() - t0);
      bus.emit('tts.done', { ms, bytes: merged.size, chunks: blobs.length });
      resolveResult({ blob: merged, url: URL.createObjectURL(merged), ms, model, voice });
    }
    checkPlayback();
  };
  const playLoop = async (): Promise<void> => {
    if (playRunning || !opts.autoplay) return;
    playRunning = true;
    while (!stopped && playCursor < blobs.length) {
      const url = URL.createObjectURL(blobs[playCursor++] as Blob);
      const el = new Audio(url);
      curEl = el;
      await new Promise<void>(res => {
        el.onended = () => res();
        el.onpause = () => res();
        el.onerror = () => res();
        tryPlay(el);
      });
      URL.revokeObjectURL(url);
      curEl = null;
    }
    playRunning = false;
    checkPlayback();
  };
  const synthLoop = async (): Promise<void> => {
    if (synthRunning) return;
    synthRunning = true;
    while (!stopped && synthCursor < chunks.length) {
      const text = chunks[synthCursor++] as string;
      if (!blobs.length) bus.emit('tts.start', { model, voice: voice || '(default)', streamed: true });
      try {
        const r = await nurClientFor(S.baseTts).audio.speech.create({
          model, input: text.slice(0, 4000), response_format: 'wav', speed: S.speed,
          ...(voice ? { voice } : {}),
        });
        const blob = await r.blob();
        if (stopped) break;
        blobs.push(blob);
        if (firstMs == null) { firstMs = Math.round(performance.now() - t0); resolveFirst(firstMs); }
        void playLoop();
      } catch (e) {
        err = errMsg(e);
        stopped = true;
      }
    }
    synthRunning = false;
    void finish();
  };
  const enqueue = (sentence: string): void => {
    const prepped = normalizeForSpeech(stripMarkdownEmphasis(sentence));
    if (!/[\p{L}\p{N}]/u.test(prepped)) return;
    for (const c of capChunk(prepped)) chunks.push(c);
    void synthLoop();
  };
  const pipe: SpeechPipeline = {
    autoplay: opts.autoplay,
    push: (text) => {
      if (stopped || ended) return;
      if (!/[.!?…:\n]/.test(text.slice(lastSegLen)) && text.length - lastSegLen < 200) return;
      lastSegLen = text.length;
      const segs = sentences(text);
      for (; committed < segs.length - 1; committed++) enqueue(segs[committed] as string);
    },
    end: (finalText) => {
      if (stopped || ended) return;
      ended = true;
      const segs = sentences(finalText);
      for (; committed < segs.length; committed++) enqueue(segs[committed] as string);
      void finish();
    },
    first: () => firstP,
    result,
    playbackDone,
    stop: () => {
      if (stopped && finished) return;
      stopped = true;
      ended = true;
      tryPause(curEl);
      curEl = null;
      void finish();
    },
    error: () => err,
  };
  livePipe = pipe;
  return pipe;
}

export function applyChatResult(m: Msg, got: StreamResult, model: string): void {
  m.thinking = got.reasoning || null;
  m.raw = { model, finish: got.finish || null, tok1: got.tok1, wall: got.wall,
    msPerTok: got.msPerTok || null, reasoning: got.reasoning || null, text: got.text, chunks: got.chunks };
  m.meta = model + ' · tok₁ ' + (got.tok1 != null ? got.tok1 + ' ms' : '—') +
    (got.msPerTok ? ' · ' + got.msPerTok.toFixed(1) + ' ms/tok' : '') +
    (got.finish === 'aborted' ? ' · (stopped)' : '');
}
export function applySttMsg(um: Msg, got: TranscribeResult): void {
  um.text = got.text ? '“' + got.text + '”' : '(no speech recognized)';
  if (S.sttFormat === 2 && got.detail && got.text) um.text = got.detail;
  um.meta = 'mic · stt ' + got.ms + ' ms · ' + got.model;
}

let composerHook: ((text: string) => boolean) | null = null;
export const setComposerHook = (fn: ((text: string) => boolean) | null): void => { composerHook = fn; };

export function doSend(text: string): void {
  if (!text || mode !== 'live') return;
  if (composerHook && composerHook(text)) { S.input = ''; scheduleRender(); return; }
  if (S.busy) return;
  S.msgs.push({ role: 'u', kind: 'text', text, meta: 'keyboard' });
  S.input = '';
  bus.emit('msg.user', { kind: 'text', text });
  scrollChatSoon();
  void liveReply(text);
}

export async function liveReply(userText: string): Promise<void> {
  const model = chatModel();
  if (!model) {
    S.msgs.push({ role: 'a', kind: 'text', text: '(no chat model available on the backend)', meta: 'error' });
    scrollChatSoon();
    return;
  }
  setBusy('agent — ' + model);
  bus.emit('run.node.start', { node: 'agent' });
  bus.emit('agent.start', { model });
  const msgIdx = S.msgs.length;
  S.msgs.push({ role: 'a', kind: 'text', text: '', meta: model });
  scrollChatSoon();
  const msgs: ChatMessageIn[] = [];
  if (S.sysPrompt.trim()) msgs.push({ role: 'system', content: S.sysPrompt.trim() });
  const wantTts = progHas('tts') && !!ttsModel();
  let pipe: SpeechPipeline | null = null;
  const feed = (t: string): void => {
    if (!wantTts) return;
    if (!pipe) { try { pipe = speechPipeline({ autoplay: true }); } catch { return; } }
    pipe.push(t);
  };
  const got = await streamChat({
    model, messages: msgs.concat(chatHistory(msgIdx)),
    onToken: (t) => { const m = S.msgs[msgIdx]; if (m) m.text = t; feed(t); scrollChatSoon(); },
    onReasoning: (t) => { const m = S.msgs[msgIdx]; if (m) m.thinking = t; scrollChatSoon(); },
  });
  const m = S.msgs[msgIdx];
  if (!m) return;
  m.text = got.text || (got.error ? '(reply failed — ' + got.error + ')'
    : '(empty reply' + (got.finish ? ' — finish ' + got.finish : '') + (got.reasonChars ? ', reasoning exhausted the token budget' : '') + ')');
  applyChatResult(m, got, model);
  if (got.error) {
    bus.emit('agent.error', { message: got.error });
    appendLog(pad('agent.error') + got.error.slice(0, 60));
  }
  bus.emit('agent.done', { text: m.text });
  bus.emit('run.node.end', { node: 'agent', ms: got.wall });
  appendLog(pad('agent.done') + 'tok₁ ' + (got.tok1 != null ? got.tok1 : '—') + ' ms · ' + got.wall + ' ms · ' + model);
  const turn: Turn = { eou: null, tok1: got.tok1, audio1: null, stt: lastSttMs, cancel: null, text: userText };
  lastSttMs = null;
  const spoken = !got.error && got.finish !== 'aborted' && !!got.text;
  if (wantTts && spoken && !pipe) {
    try { pipe = speechPipeline({ autoplay: true }); }
    catch (e) { appendLog(pad('tts.error') + errMsg(e).slice(0, 60)); }
  }
  if (pipe) {
    const p = pipe as SpeechPipeline;
    if (!spoken) p.stop();
    else {
      setBusy('synthesizing — ' + (ttsModel() || ''));
      p.end(got.text);
      const fm = await p.first();
      if (fm != null) {
        turn.audio1 = fm;
        appendLog(pad('tts.first') + fm + ' ms to first audio');
        void p.result.then(res => {
          if (!res) return;
          m.kind = 'audio'; m.url = res.url; m.blob = res.blob; m.peaks = null;
          m.meta += ' · tts₁ ' + fm + ' ms · ' + res.model;
          attachAudioMeta(m);
          appendLog(pad('tts.done') + res.ms + ' ms · ' + res.model);
          scheduleRender();
        });
      } else if (p.error()) {
        appendLog(pad('tts.error') + (p.error() as string).slice(0, 60));
      }
    }
  }
  S.turns.push(turn);
  S.sel = S.turns.length - 1;
  bus.emit('turn.recorded', { turn: { ...turn }, index: S.turns.length - 1 });
  setBusy(null);
  scrollChatSoon();
}

export type MicUiState = 'idle' | 'connecting' | 'listening' | 'committing' | 'monitoring';
let micUi: MicUiState = 'idle';
export const micState = (): MicUiState => micUi;
export function setMicState(s: MicUiState): void {
  if (micUi === s) return;
  micUi = s;
  bus.emit('mic.state', { state: s });
  scheduleRender();
}

export interface ChunkedPartialOpts {
  blob: () => Blob | null;
  wsSeen: () => boolean;
  apply: (text: string) => void;
}
export function chunkedPartials(opts: ChunkedPartialOpts): { stop: () => void } {
  let seq = 0, busyTick = false, done = false;
  const tick = async (): Promise<void> => {
    if (done || busyTick || opts.wsSeen() || !sttModel()) return;
    const soFar = opts.blob();
    if (!soFar || !soFar.size) return;
    busyTick = true;
    const s = ++seq;
    try {
      const got = await transcribe(soFar, { format: 0, quiet: true });
      if (!done && s === seq && !opts.wsSeen() && got.text) {
        opts.apply(got.text);
        bus.emit('stt.partial', { text: got.text, bytes: soFar.size });
      }
    } catch {}
    busyTick = false;
  };
  const iv = setInterval(() => { void tick(); }, 1400);
  return { stop: () => { done = true; clearInterval(iv); } };
}

export interface OcrResult { text: string; ms: number; elements: number }
export async function ocrFrame(blob: Blob): Promise<OcrResult> {
  const fd = new FormData();
  fd.append('file', blob, 'frame.png');
  const t0 = performance.now();
  const j = await postOcr(fd);
  const els = j.elements || [];
  return {
    text: els.map(e => e && e.text).filter((t): t is string => !!t).join('\n'),
    ms: Math.round(performance.now() - t0),
    elements: els.length,
  };
}

export async function scanShot(): Promise<void> {
  if (S.busy || mode !== 'live') return;
  setBusy('camera — one shot');
  let stream: MediaStream | null = null;
  try {
    stream = await navigator.mediaDevices.getUserMedia({ video: true });
    const frame = await grabFrame(stream);
    stopTracks(stream);
    stream = null;
    S.msgs.push({
      role: 'u', kind: 'visual', src: frame.dataUrl,
      text: 'camera frame · ' + frame.w + '×' + frame.h,
      meta: 'camera · one shot', file: 'camera-frame.png',
    });
    bus.emit('frame.captured', { source: 'cam', w: frame.w, h: frame.h });
    bus.emit('msg.user', { kind: 'visual' });
    scrollChatSoon();
    setBusy('ocr — /v1/ocr');
    const got = await ocrFrame(frame.blob);
    bus.emit('ocr.done', { ms: got.ms, elements: got.elements, chars: got.text.length });
    appendLog(pad('ocr.done') + got.elements + ' elements · ' + got.text.length + ' chars · ' + got.ms + ' ms');
    if (got.text) setState({ input: (S.input.trim() ? S.input.replace(/\s+$/, '') + '\n' : '') + got.text });
    else appendLog(pad('scan.note') + 'no text found in the frame');
  } catch (e) {
    const msg = errMsg(e);
    bus.emit('scan.error', { message: msg });
    appendLog(pad('scan.error') + msg.slice(0, 60));
  } finally {
    if (stream) stopTracks(stream);
    setBusy(null);
    scrollChatSoon();
  }
}

interface RecState { mr: MediaRecorder; stream: MediaStream }
let recState: RecState | null = null;
export const isRecording = (): boolean => !!recState;
export async function liveMicToggle(): Promise<void> {
  if (recState) { try { recState.mr.stop(); } catch {  } return; }
  let stream: MediaStream | null = null;
  try {
    stopAudio();
    setMicState('connecting');
    stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    const mr = new MediaRecorder(stream);
    const chunks: Blob[] = [];
    recState = { mr, stream };
    bus.emit('mic.start', {});
    mr.ondataavailable = (e) => { if (e.data && e.data.size) chunks.push(e.data); };
    let partialDone = false, wsPartialSeen = false;
    let rt: RealtimeSession | null = null;
    let wsLive = false, wsDownLogged = false;
    const wsDown = (reason: string): void => {
      const attempted = wsLive || !!rt;
      const dead = rt;
      wsLive = false;
      rt = null;
      if (dead) dead.stop();
      wsPartialSeen = false;
      if (!partialDone && attempted && !wsDownLogged) {
        wsDownLogged = true;
        appendLog(pad('stt.rt') + 'realtime ws ' + reason + ' — falling back to chunked partials');
      }
    };
    if (realtimeEnabled() && sttModel()) {
      rt = openRealtime(sttModel(), {
        partial: (text) => {
          if (partialDone || !text) return;
          wsPartialSeen = true;
          setState({ input: text });
          bus.emit('stt.partial', { text, via: 'ws' });
        },
        speechStart: (atMs) => bus.emit('vad.speech', { atMs, via: 'server' }),
        speechStop: (atMs) => bus.emit('vad.silence', { atMs, via: 'server' }),
        error: () => wsDown('errored'),
        close: () => wsDown('closed'),
      });
      rt.start(stream).then(() => { if (rt) wsLive = true; }).catch(() => wsDown('unavailable'));
    }
    const ticker = chunkedPartials({
      blob: () => chunks.length ? new Blob(chunks, { type: mr.mimeType || 'audio/webm' }) : null,
      wsSeen: () => wsPartialSeen,
      apply: (text) => setState({ input: text }),
    });
    mr.onstop = async () => {
      partialDone = true;
      ticker.stop();
      const rtOpen = rt;
      rt = null;
      if (rtOpen) rtOpen.stop();
      setMicState('committing');
      setState({ input: '' }, { silent: true });
      recState = null;
      stopTracks(stream);
      scheduleRender();
      const blob = new Blob(chunks, { type: mr.mimeType || 'audio/webm' });
      bus.emit('mic.stop', { bytes: blob.size });
      const idx = S.msgs.length;
      S.msgs.push({ role: 'u', kind: 'audio', url: URL.createObjectURL(blob), blob, text: '…', meta: 'mic' });
      attachAudioMeta(S.msgs[idx]);
      bus.emit('msg.user', { kind: 'audio' });
      scrollChatSoon();
      setBusy('transcribing — ' + (sttModel() || ''));
      let got: TranscribeResult | null = null;
      try { got = await transcribe(blob); }
      catch (e) {
        const msg = errMsg(e);
        bus.emit('stt.error', { message: msg });
        appendLog(pad('stt.error') + msg.slice(0, 60));
        const um = S.msgs[idx];
        if (um) um.text = '(transcription failed — ' + msg + ')';
      }
      setBusy(null);
      if (got) {
        lastSttMs = got.ms;
        bus.emit('stt.final', { text: got.text, ms: got.ms });
        appendLog(pad('stt.final') + '"' + got.text.slice(0, 44) + '" · ' + got.ms + ' ms · ' + got.model);
        const um = S.msgs[idx];
        if (um) applySttMsg(um, got);
        scrollChatSoon();
        if (got.text) void liveReply(got.text);
      }
      setMicState('idle');
      scheduleRender();
    };
    mr.start(500);
    setMicState('listening');
    scheduleRender();
  } catch (e) {
    stopTracks(stream);
    recState = null;
    setMicState('idle');
    const msg = errMsg(e);
    bus.emit('mic.error', { message: msg });
    appendLog(pad('mic.error') + msg.slice(0, 60));
    scheduleRender();
  }
}

export { notify, OFFLINE_MSG };
