import { useSyncExternalStore } from 'react';
import { flushSync } from 'react-dom';
import type { ChatCompletionChunk } from 'nur-client';

export type BlockKind =
  | 'mic' | 'keys' | 'screen' | 'cam' | 'on'
  | 'eou' | 'stt' | 'translate' | 'tts' | 'agent' | 'ocr'
  | 'spk' | 'store' | 'cancel' | 'if' | 'for'
  | 'set' | 'append' | 'repeat' | 'wait' | 'http';

export type Val = number | string | boolean;
export interface Block { k: BlockKind; children?: Block[]; c?: Record<string, string | number> }
export interface Stack { x: number; y: number; items: Block[] }
export type Path = Array<number | 'S'>;

export interface Turn {
  eou: number | null;
  tok1: number | null;
  audio1: number | null;
  stt: number | null;
  cancel: number | null;
  text: string;
}

export interface RawResult {
  model: string;
  finish: string | null;
  tok1?: number | null;
  wall: number;
  msPerTok?: number | null;
  reasoning: string | null;
  text: string;
  chunks: ChatCompletionChunk[];
}

export interface Msg {
  role: 'u' | 'a';
  kind: 'text' | 'audio' | 'visual';
  text: string;
  meta?: string;
  url?: string;
  blob?: Blob;
  blobType?: string;
  peaks?: number[] | null;
  durSec?: number;
  durLabel?: string;
  _decoding?: boolean;
  src?: string;
  file?: string;
  thinking?: string | null;
  raw?: RawResult;
}

export interface PastRecord {
  id: string;
  intent: number | null;
  title: string;
  summary: string;
  turns: Turn[];
  sel: number;
  msgs: Msg[];
  log: string[];
  prog: Block[];
  stacks: Stack[];
  savedAt?: number;
}

export type Screen = 'intent' | 'patch' | 'duet';
export type View = 'chat' | 'insp' | 'patch';

export interface AppState {
  screen: Screen;
  view: View;
  intent: number | null;
  sessionId: string | null;
  hydrated: boolean;
  notice: string | null;
  selChat: string | null;
  selStt: string | null;
  selTts: string | null;
  voice: string | null;
  vdText: string;
  barge: boolean;
  selNode: BlockKind | null;
  selPath: Path | null;
  sttFormat: number;
  speed: number;
  eouMs: number;
  temp: number;
  topP: number;
  effort: string;
  sysPrompt: string;
  showThinking: boolean;
  msgs: Msg[];
  input: string;
  busy: string | null;
  log: string[];
  turns: Turn[];
  sel: number;
  playId: number | null;
  playT: number;
  viewer: number | null;
  txtModal: number | null;
  past: PastRecord[];
  prog: Block[];
  stacks: Stack[];
  msgInfo: Record<number, boolean>;
  runPath: string | null;
  runError: string | null;
  runErrorMsg: string | null;
  runNotice: string | null;
  lastVars: Record<string, Val> | null;
  dslMode: boolean;
  baseChat: string;
  baseStt: string;
  baseTts: string;
  fps?: number;
  frames?: number;
  ifCond?: number;
  vPrev?: number | null;
  inspFilter?: string;
  palOpen?: boolean;
}

export const S: AppState = {
  screen: 'intent', view: 'chat',
  intent: null,
  sessionId: null,
  hydrated: false,
  notice: null,
  selChat: null, selStt: null, selTts: null, voice: null, vdText: '',
  barge: true,
  selNode: null, selPath: null,
  sttFormat: 0, speed: 1, eouMs: 400, temp: 0.7, topP: 0.95, effort: 'default', sysPrompt: '', showThinking: false,
  msgs: [], input: '', busy: null, log: [], turns: [], sel: 0,
  playId: null, playT: 0, viewer: null, txtModal: null, past: [],
  prog: [], stacks: [], msgInfo: {},
  runPath: null, runError: null, runErrorMsg: null, runNotice: null,
  lastVars: null, dslMode: false, baseChat: '', baseStt: '', baseTts: '',
};

let version = 0;
const listeners = new Set<() => void>();

const subscribeStore = (fn: () => void): (() => void) => {
  listeners.add(fn);
  return () => { listeners.delete(fn); };
};
const getVersion = (): number => version;

const renderSubs: Array<(s: AppState) => void> = [];
export const subscribe = (fn: (s: AppState) => void): (() => void) => {
  renderSubs.push(fn);
  return () => { const i = renderSubs.indexOf(fn); if (i >= 0) renderSubs.splice(i, 1); };
};

export function notify(): void {
  version++;
  listeners.forEach(f => { try { f(); } catch {  } });
  renderSubs.forEach(f => { try { f(S); } catch {  } });
}

let renderQueued = false;
export function scheduleRender(): void {
  if (renderQueued) return;
  renderQueued = true;
  const flush = (): void => { renderQueued = false; notify(); };
  if (typeof document !== 'undefined' && document.visibilityState === 'hidden') setTimeout(flush, 0);
  else requestAnimationFrame(flush);
}
if (typeof document !== 'undefined')
  document.addEventListener('visibilitychange', () => {
    if (document.visibilityState === 'visible' && renderQueued) notify();
  });

export function renderNow(): void {
  try { flushSync(() => notify()); } catch { notify(); }
}

export type StatePatch = Partial<AppState> | ((s: AppState) => Partial<AppState>);
export function setState(patch: StatePatch, opts?: { silent?: boolean }): void {
  Object.assign(S, typeof patch === 'function' ? patch(S) : patch);
  if (!(opts && opts.silent)) notify();
}

export function useAppState(): AppState {
  useSyncExternalStore(subscribeStore, getVersion);
  return S;
}
