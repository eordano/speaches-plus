import { bus } from '../state/bus';
import {
  S, setState, scheduleRender, type Block, type BlockKind, type Path, type Stack,
} from '../state/store';
import { clamp, errMsg, pad } from '../lib/util';
import { dlText } from '../lib/util';
import {
  chatModel, sttModel, ttsModel, ttsVoice, appendLog,
} from './session';
import { progFor, shortId, ttsKind } from '../data';
import { storage } from '../state/storage';
import { BUILTIN_NAMES, FUNC_SIGS, IF_PRESETS, VAR_RE } from './expr';
import { SIG, TAP_RE, checkProgram, tapOf, type Flow, type FlowResult } from './ports';

export const COLORS: Record<BlockKind, string> = {
  mic: '#8B6B2A', keys: '#8B6B2A', screen: '#8B6B2A', cam: '#8B6B2A', on: '#8B6B2A',
  eou: '#5A6B7A', stt: '#5A6E5A', translate: '#5A6E5A',
  tts: '#5A6E5A', agent: '#6E5A3C', ocr: '#7A4848', spk: '#2B2B2B',
  store: '#2B2B2B', cancel: '#A8201A', if: '#5A6B7A', for: '#6E5A3C',
  set: '#5A6B7A', append: '#5A6B7A', repeat: '#5A6B7A', wait: '#5A6B7A', http: '#2B2B2B',
};
export const HATS: Partial<Record<BlockKind, 1>> = { mic: 1, keys: 1, screen: 1, cam: 1, on: 1 };
export const LABELS: Partial<Record<BlockKind, string>> = {
  mic: 'when mic hears speech', keys: 'when text is sent', screen: 'when a frame arrives',
  cam: 'when camera captures', on: 'when stream arrives', eou: 'wait for end of utterance', stt: 'transcribe audio',
  translate: 'translate to english', tts: 'speak reply',
  agent: 'ask agent', ocr: 'read text on image', spk: 'play on speaker',
  store: 'store output', cancel: 'cancel playback',
  set: 'set variable', append: 'append text', repeat: 'repeat', wait: 'wait', http: 'http request',
};
export const PALETTE: Array<[string, string, BlockKind[]]> = [
  ['EVENTS', '#8B6B2A', ['mic', 'keys', 'screen', 'cam', 'on']],
  ['TURN', '#5A6B7A', ['eou']],
  ['SPEECH', '#5A6E5A', ['stt', 'translate', 'tts']],
  ['LANGUAGE', '#6E5A3C', ['agent']],
  ['VISION', '#7A4848', ['ocr']],
  ['LOGIC', '#5A6B7A', ['set', 'append', 'if']],
  ['OUTPUTS', '#2B2B2B', ['spk', 'store', 'http']],
  ['CONTROL', '#A8201A', ['repeat', 'wait', 'for', 'cancel']],
];
export const TITLES: Record<BlockKind, string> = {
  mic: 'Microphone', keys: 'Keyboard', screen: 'Screen share', cam: 'Camera', on: 'Stream subscriber',
  eou: 'End of utterance', stt: 'Transcribe', translate: 'Translate (via chat model)',
  agent: 'Chat agent', ocr: 'OCR', tts: 'Speak',
  spk: 'Speaker', store: 'Store output', cancel: 'Cancel playback', if: 'If', for: 'For each frame',
  set: 'Set variable', append: 'Append to variable', repeat: 'Repeat', wait: 'Wait', http: 'HTTP request',
};

export type ConfigKey =
  | 'selChat' | 'selStt' | 'selTts' | 'voice' | 'vdText' | 'barge'
  | 'sttFormat' | 'speed' | 'eouMs' | 'temp' | 'topP' | 'sysPrompt' | 'showThinking'
  | 'fps' | 'frames' | 'ifCond' | 'baseChat' | 'baseStt' | 'baseTts';
export type NumericConfigKey = 'fps' | 'frames' | 'ifCond' | 'sttFormat' | 'speed' | 'eouMs' | 'temp' | 'topP';
const DEF: Record<string, number> = { fps: 1, frames: 3, ifCond: 0 };
export const gv = (k: NumericConfigKey): number => {
  const v = S[k];
  return v !== undefined ? v : (DEF[k] ?? 0);
};
function clearErrNotice(): void {
  if (S.runErrorMsg && S.runNotice === S.runErrorMsg) S.runNotice = null;
}
export function sv<K extends ConfigKey>(k: K, v: typeof S[K]): void {
  clearErrNotice();
  setState({ [k]: v });
  savePatch();
}
export const CONFIG_KEYS: ConfigKey[] = ['selChat', 'selStt', 'selTts', 'voice', 'vdText', 'barge',
  'sttFormat', 'speed', 'eouMs', 'temp', 'topP', 'sysPrompt', 'showThinking',
  'fps', 'frames', 'ifCond', 'baseChat', 'baseStt', 'baseTts'];

export const bcNum = (b: Block, key: string, def: number): number => {
  const v = b.c ? b.c[key] : undefined;
  return typeof v === 'number' && isFinite(v) ? v : def;
};
export const bcStr = (b: Block, key: string, def = ''): string => {
  const v = b.c ? b.c[key] : undefined;
  return typeof v === 'string' ? v : def;
};
export const ifExpr = (b: Block): string =>
  bcStr(b, 'expr') || (IF_PRESETS[clamp(gv('ifCond'), 0, IF_PRESETS.length - 1)] ?? IF_PRESETS[0]);
export function selBlock(): Block | null {
  const L = getList(S.selPath);
  return (L && L.arr[L.idx]) || null;
}
export function svb(key: string, v: string | number): void {
  const b = selBlock();
  if (!b) return;
  b.c = { ...(b.c || {}), [key]: v };
  clearErrNotice();
  savePatch();
  scheduleRender();
}

export const labelOf = (k: BlockKind): string =>
  k === 'if' ? 'if' :
  k === 'for' ? 'for each frame' : (LABELS[k] ?? k);
export const SUBS: Partial<Record<BlockKind, (b: Block) => string>> = {
  mic: () => 'getUserMedia · vad',
  on: (b) => (tapOf(b) || '(unnamed)') + ' · tap subscriber',
  keys: () => 'composer text',
  screen: (b) => bcNum(b, 'fps', gv('fps')) + ' fps · getDisplayMedia',
  cam: () => 'getUserMedia frame',
  eou: () => S.eouMs + ' ms silence',
  stt: () => shortId(sttModel()) + ' · ' + (['json', 'verbose', 'diarized'][S.sttFormat] ?? 'json'),
  translate: () => 'to english · ' + shortId(chatModel()),
  tts: () => {
    const id = ttsModel(), kind = ttsKind(id);
    const v = kind === 'design' ? 'designed voice' : (ttsVoice() || 'default voice');
    return shortId(id) + ' · ' + v;
  },
  agent: () => shortId(chatModel()),
  ocr: () => 'POST /v1/ocr',
  spk: () => 'audio element',
  store: () => storage.kind === 'opfs' ? 'opfs' : 'localStorage',
  cancel: () => 'stops audio',
  if: (b) => ifExpr(b),
  for: (b) => bcNum(b, 'frames', gv('frames')) + ' frames · ' + bcNum(b, 'fps', gv('fps')) + ' fps',
  set: (b) => bcStr(b, 'var', '?') + ' = ' + bcStr(b, 'expr'),
  append: (b) => bcStr(b, 'var', '?') + ' += ' + bcStr(b, 'expr'),
  repeat: (b) => bcNum(b, 'n', 3) + '×',
  wait: (b) => bcNum(b, 'ms', 1000) + ' ms',
  http: (b) => bcStr(b, 'method', 'GET') + ' ' + (bcStr(b, 'url').replace(/^https?:\/\//, '').slice(0, 40) || '(no url)'),
};

export const stmtShape = (W: number): string =>
  `M8,0 H20 c1,7 4,9 8,9 s7,-2 8,-9 H${W - 8} a8 8 0 0 1 8,8 V32 a8 8 0 0 1 -8,8 H36 c-1,7 -4,9 -8,9 s-7,-2 -8,-9 H8 a8 8 0 0 1 -8,-8 V8 a8 8 0 0 1 8,-8 Z`;
export const hatShape = (W: number): string =>
  `M0,16 C${Math.round(W * 0.3)},-7 ${Math.round(W * 0.7)},-7 ${W},16 V38 a8 8 0 0 1 -8,8 H36 c-1,7 -4,9 -8,9 s-7,-2 -8,-9 H8 a8 8 0 0 1 -8,-8 Z`;
export const cShape = (W: number, mH: number): string => {
  const yb = 40 + mH;
  return `M8,0 H20 c1,7 4,9 8,9 s7,-2 8,-9 H${W - 8} a8 8 0 0 1 8,8 V32 a8 8 0 0 1 -8,8 H52 c-1,7 -4,9 -8,9 s-7,-2 -8,-9 H22 a6 6 0 0 0 -6,6 V${yb - 6} a6 6 0 0 0 6,6 H${W - 8} a8 8 0 0 1 8,8 V${yb + 14} a8 8 0 0 1 -8,8 H36 c-1,7 -4,9 -8,9 s-7,-2 -8,-9 H8 a8 8 0 0 1 -8,-8 V8 a8 8 0 0 1 8,-8 Z`;
};
export function measure(b: Block, asHat?: boolean): number {
  if (HATS[b.k]) return asHat === false ? 42 : 46;
  if (b.children) {
    const sum = b.children.reduce((a, c) => a + measure(c, false), 0);
    return 40 + Math.max(sum + 12, 42) + 22;
  }
  return 42;
}
export const widthOf = (b: Block): number => (b.children ? 248 : 208);
export const shapeFor = (b: Block): string => HATS[b.k] ? hatShape(208)
  : b.children ? cShape(248, measure(b) - 62) : stmtShape(208);

export interface LayoutBlockBox { b: Block; path: Path; x: number; y: number; w: number; h: number; hat: boolean }
export interface Slot { arr: Block[]; index: number; x: number; y: number }
export interface Layout { blocks: LayoutBlockBox[]; slots: Slot[]; wsH: number }
export function layout(): Layout {
  const blocks: LayoutBlockBox[] = [], slots: Slot[] = [];
  let maxB = 0;
  const walk = (items: Block[], x: number, y: number, base: Path): void => {
    const rootList = !base.length || (base.length === 2 && base[0] === 'S');
    let cy = y;
    items.forEach((b, i) => {
      slots.push({ arr: items, index: i, x, y: cy });
      const hat = !!HATS[b.k] && rootList && i === 0;
      const hh = measure(b, hat);
      blocks.push({ b, path: base.concat(i), x, y: cy, w: widthOf(b), h: hh, hat });
      if (b.children) walk(b.children, x + 16, cy + 40, base.concat(i));
      cy += hh;
      if (cy > maxB) maxB = cy;
    });
    slots.push({ arr: items, index: items.length, x, y: cy });
    if (cy > maxB) maxB = cy;
  };
  walk(S.prog, 24, 16, []);
  S.stacks.forEach((st, si) => walk(st.items, st.x, st.y, ['S', si]));
  return { blocks, slots, wsH: Math.max(maxB + 110, 440) };
}

export const pk = (path: Path): string => JSON.stringify(path);
export const deep = <T>(o: T): T => JSON.parse(JSON.stringify(o)) as T;
export function newItem(k: BlockKind): Block {
  const b: Block = { k };
  if (k === 'if' || k === 'for' || k === 'repeat') b.children = [];
  if (k === 'on') b.c = { tap: '' };
  if (k === 'if') b.c = { expr: IF_PRESETS[0] };
  else if (k === 'set') b.c = { var: 'x', expr: '0' };
  else if (k === 'append') b.c = { var: 'notes', expr: 'transcript' };
  else if (k === 'repeat') b.c = { n: 3 };
  else if (k === 'wait') b.c = { ms: 1000 };
  else if (k === 'http') b.c = { url: '', method: 'GET', into: 'http' };
  return b;
}
export const isPrefix = (a: Path, b: Path): boolean => a.length <= b.length && a.every((v, i) => b[i] === v);
export interface ListRef { arr: Block[]; idx: number }
export function getList(path: Path | null): ListRef | null {
  if (!path || !path.length) return null;
  let arr: Block[], rest: Path;
  if (path[0] === 'S') {
    const st = S.stacks[path[1] as number];
    if (!st) return null;
    arr = st.items; rest = path.slice(2);
  } else { arr = S.prog; rest = path; }
  if (!rest.length) return null;
  for (let i = 0; i < rest.length - 1; i++) {
    const b = arr[rest[i] as number];
    if (!b || !b.children) return null;
    arr = b.children;
  }
  return { arr, idx: rest[rest.length - 1] as number };
}
export function pruneStacks(): void { S.stacks = S.stacks.filter(st => st.items.length); }
export function fixSelection(): void {
  if (!S.selPath) return;
  const L = getList(S.selPath);
  if (!L || !L.arr[L.idx] || L.arr[L.idx]?.k !== S.selNode) { S.selNode = null; S.selPath = null; }
}
export interface HatRoot { items: Block[]; base: Path }
export function hatRoot(): HatRoot | null {
  const first = S.prog[0];
  if (first && HATS[first.k] && first.k !== 'on') return { items: S.prog, base: [] };
  for (let i = 0; i < S.stacks.length; i++) {
    const st = S.stacks[i];
    const h0 = st && st.items[0];
    if (st && h0 && HATS[h0.k] && h0.k !== 'on') return { items: st.items, base: ['S', i] };
  }
  return null;
}
export function flow(): FlowResult {
  return checkProgram({ prog: S.prog, stacks: S.stacks }, LABELS as Record<string, string>);
}
export const flowOut = (path: Path): Flow | null => flow().out.get(pk(path)) || null;
export const flowErr = (path: Path): string | null => flow().errs.get(pk(path)) || null;
export function tapPublishers(): Array<{ path: Path; name: string; f: Flow }> {
  const out: Array<{ path: Path; name: string; f: Flow }> = [];
  const walkP = (items: Block[], base: Path): void => items.forEach((b, i) => {
    const name = tapOf(b);
    const f = flowOut(base.concat(i));
    if (name && f) out.push({ path: base.concat(i), name, f });
    if (b.children) walkP(b.children, base.concat(i));
  });
  walkP(S.prog, []);
  S.stacks.forEach((st, i) => walkP(st.items, ['S', i]));
  return out;
}
function freshTapName(): string {
  const used = new Set(tapPublishers().map(t => t.name));
  for (let i = 1; ; i++) if (!used.has('s' + i)) return 's' + i;
}

export interface Completion { name: string; kind: 'var' | 'builtin' | 'fn' | 'lit'; detail: string; insert?: string; caret?: number }
const BUILTIN_DETAIL: Record<string, string> = {
  transcript: 'text · builtin', reply: 'text · builtin', ocr: 'text · builtin',
  translated: 'text · builtin', barged: 'true/false · builtin', endpointed: 'true/false · builtin',
  sttMs: 'number · builtin', tok1Ms: 'number · builtin', iter: 'number · builtin',
};
export function identsInScope(selPath: Path | null): Completion[] {
  const vars = new Map<string, string>();
  const stopKey = selPath ? pk(selPath) : null;
  const collect = (items: Block[], base: Path, stopBefore: string | null, stopAfter: string | null): boolean => {
    for (let i = 0; i < items.length; i++) {
      const b = items[i];
      if (!b) continue;
      const key = pk(base.concat(i));
      if (stopBefore && key === stopBefore) return true;
      if (b.k === 'set' || b.k === 'append') {
        const n = bcStr(b, 'var').trim();
        if (VAR_RE.test(n)) vars.set(n, (b.k === 'set' ? 'set' : 'appended') + ' upstream');
      }
      if (b.k === 'http') {
        const n = bcStr(b, 'into', 'http').trim() || 'http';
        if (VAR_RE.test(n)) { vars.set(n, 'http response body'); vars.set(n + 'Status', 'http status code'); }
      }
      if (b.children && collect(b.children, base.concat(i), stopBefore, stopAfter)) return true;
      if (stopAfter && key === stopAfter) return true;
    }
    return false;
  };
  const listFor = (path: Path): { items: Block[]; base: Path } => {
    if (path[0] === 'S') {
      const st = S.stacks[path[1] as number];
      return { items: st ? st.items : [], base: path.slice(0, 2) };
    }
    return { items: S.prog, base: [] };
  };
  if (selPath) {
    const chain: Array<{ items: Block[]; base: Path; stopBefore: string | null; stopAfter: string | null }> = [];
    let cur = listFor(selPath);
    let stopBefore: string | null = stopKey, stopAfter: string | null = null;
    for (let depth = 0; depth < 4; depth++) {
      chain.unshift({ ...cur, stopBefore, stopAfter });
      const first = cur.items[0];
      if (!first || first.k !== 'on') break;
      const pub = tapPublishers().find(t => t.name === tapOf(first));
      if (!pub) break;
      cur = listFor(pub.path);
      stopBefore = null;
      stopAfter = pk(pub.path);
    }
    chain.forEach(c => collect(c.items, c.base, c.stopBefore, c.stopAfter));
  }
  const out: Completion[] = [];
  vars.forEach((detail, name) => out.push({ name, kind: 'var', detail }));
  out.sort((a, b) => (a.name < b.name ? -1 : 1));
  BUILTIN_NAMES.forEach(n => out.push({ name: n, kind: 'builtin', detail: BUILTIN_DETAIL[n] || 'builtin' }));
  FUNC_SIGS.forEach(([n, sig]) => out.push({ name: n, kind: 'fn', detail: sig, insert: n + '()', caret: -1 }));
  out.push({ name: 'true', kind: 'lit', detail: 'boolean' }, { name: 'false', kind: 'lit', detail: 'boolean' });
  return out;
}

export function countNodes(items: Block[]): number {
  return items.reduce((a, b) => a + 1 + (b.children ? countNodes(b.children) : 0), 0);
}

interface Snap { prog: Block[]; stacks: Stack[] }
const undoStack: Snap[] = [];
const snapState = (): Snap => deep({ prog: S.prog, stacks: S.stacks });
export function pushUndo(s?: Snap): void {
  undoStack.push(s || snapState());
  if (undoStack.length > 50) undoStack.shift();
}
export const canUndo = (): boolean => undoStack.length > 0;
export function clearUndo(): void { undoStack.length = 0; }
export function undoOnce(): void {
  const s = undoStack.pop();
  if (!s) return;
  setState({
    prog: s.prog, stacks: s.stacks, dslMode: false,
    selNode: null, selPath: null, runPath: null, runError: null, runErrorMsg: null, runNotice: null,
  });
  savePatch();
  bus.emit('patch.undo', { nodes: countNodes(s.prog) });
  appendLog(pad('patch.undo') + countNodes(s.prog) + ' blocks restored');
}

let noticeTimer: ReturnType<typeof setTimeout> | null = null;
export function flashNotice(msg: string): void {
  S.runNotice = msg;
  if (noticeTimer) clearTimeout(noticeTimer);
  noticeTimer = setTimeout(() => {
    noticeTimer = null;
    if (S.runNotice === msg) { S.runNotice = null; scheduleRender(); }
  }, 4000);
}

export function removeSel(via = 'panel'): void {
  const L = getList(S.selPath);
  const b = L && L.arr[L.idx];
  if (!L || !b) return;
  pushUndo();
  L.arr.splice(L.idx, 1);
  pruneStacks();
  bus.emit('patch.block.remove', { node: b.k, via });
  setState({ selNode: null, selPath: null, runPath: null, runError: null, runErrorMsg: null });
  savePatch();
}

export let dslValid = true;
export const setDslValid = (v: boolean): void => { dslValid = v; };
let dslInsert: ((line: string) => void) | null = null;
export const setDslInsert = (fn: ((line: string) => void) | null): void => { dslInsert = fn; };
export const insertDslLine = (line: string): void => { if (dslInsert) dslInsert(line); };
export function toggleDsl(): void {
  if (S.dslMode) {
    if (!dslValid) flashNotice('text had errors — showing last applied blocks');
    setState({ dslMode: false });
  } else {
    dslValid = true;
    setState({ dslMode: true, selNode: null, selPath: null });
  }
  bus.emit('patch.dsl.mode', { on: S.dslMode });
}

let touched = false;
const BLOCK_CONF: Partial<Record<BlockKind, Record<string, 'n' | 's'>>> = {
  on: { tap: 's' },
  screen: { fps: 'n' },
  for: { frames: 'n', fps: 'n' },
  if: { expr: 's' },
  set: { var: 's', expr: 's' },
  append: { var: 's', expr: 's' },
  repeat: { n: 'n' },
  wait: { ms: 'n' },
  http: { url: 's', method: 's', body: 's', headers: 's', into: 's' },
};
const CONF_LIMITS: Record<string, [number, number]> = { fps: [1, 5], frames: [1, 10], n: [1, 20], ms: [0, 30000] };
export function sanitize(items: unknown): Block[] | null {
  if (!Array.isArray(items)) return null;
  const out: Block[] = [];
  for (const b of items as Array<{ k?: unknown; children?: unknown; c?: unknown }>) {
    if (!b || typeof b.k !== 'string' || !(b.k in COLORS)) continue;
    const k = b.k as BlockKind;
    const nb: Block = { k };
    if (k === 'if' || k === 'for' || k === 'repeat') nb.children = sanitize(b.children) || [];
    const spec = BLOCK_CONF[k];
    if (spec && b.c && typeof b.c === 'object') {
      const c: Record<string, string | number> = {};
      for (const key of Object.keys(spec)) {
        const raw = (b.c as Record<string, unknown>)[key];
        if (raw === undefined) continue;
        if (spec[key] === 'n') {
          const n = Number(raw);
          if (isFinite(n)) {
            const lim = CONF_LIMITS[key];
            c[key] = lim ? clamp(Math.round(n), lim[0], lim[1]) : Math.round(n);
          }
        } else c[key] = String(raw).slice(0, 4000);
      }
      if (Object.keys(c).length) nb.c = c;
    }
    const rawTap = (b.c && typeof b.c === 'object') ? (b.c as Record<string, unknown>).tap : undefined;
    if (typeof rawTap === 'string' && (k === 'on' || (SIG[k].gives && TAP_RE.test(rawTap.trim()))))
      nb.c = { ...(nb.c || {}), tap: rawTap.trim().slice(0, 24) };
    out.push(nb);
  }
  return out;
}
function bakeLegacy(items: Block[], cfg: Record<string, unknown>): void {
  const num = (k: string): number | undefined => (typeof cfg[k] === 'number' ? cfg[k] as number : undefined);
  const preset = IF_PRESETS[clamp(num('ifCond') ?? 0, 0, IF_PRESETS.length - 1)] ?? IF_PRESETS[0];
  const walk = (list: Block[]): void => list.forEach(b => {
    if (b.k === 'if' && !bcStr(b, 'expr')) b.c = { ...(b.c || {}), expr: preset };
    if (b.k === 'screen' && bcNum(b, 'fps', 0) === 0 && num('fps') != null)
      b.c = { ...(b.c || {}), fps: clamp(Math.round(num('fps') as number), 1, 5) };
    if (b.k === 'for') {
      if (bcNum(b, 'frames', 0) === 0 && num('frames') != null)
        b.c = { ...(b.c || {}), frames: clamp(Math.round(num('frames') as number), 1, 10) };
      if (bcNum(b, 'fps', 0) === 0 && num('fps') != null)
        b.c = { ...(b.c || {}), fps: clamp(Math.round(num('fps') as number), 1, 5) };
    }
    if (b.children) walk(b.children);
  });
  walk(items);
}
export type ConfigSnapshot = Partial<Pick<typeof S, ConfigKey>>;
export function configSnapshot(): ConfigSnapshot {
  const cfg: Record<string, unknown> = {};
  CONFIG_KEYS.forEach(k => { if (S[k] !== undefined) cfg[k] = S[k]; });
  return cfg as ConfigSnapshot;
}
export function savePatch(): void {
  if (S.intent == null) return;
  touched = true;
  storage.patches.set(S.intent, { v: 1, intent: S.intent, prog: S.prog, stacks: S.stacks, config: configSnapshot() });
}
interface StoredProgram {
  v?: number; intent?: number; prog?: unknown; stacks?: unknown; config?: Record<string, unknown>;
}
export function applyProgram(d: StoredProgram): { prog: Block[]; stacks: Stack[] } {
  const prog = sanitize(d.prog) || [];
  const stacks: Stack[] = Array.isArray(d.stacks)
    ? (d.stacks as Array<{ x?: unknown; y?: unknown; items?: unknown }>)
        .map(st => ({ x: Math.max(0, Number(st.x) || 0), y: Math.max(0, Number(st.y) || 0), items: sanitize(st.items) || [] }))
        .filter(st => st.items.length)
    : [];
  const legacyCfg = (d.config && typeof d.config === 'object') ? d.config : {};
  bakeLegacy(prog, legacyCfg);
  stacks.forEach(st => bakeLegacy(st.items, legacyCfg));
  const patch: Record<string, unknown> = { prog, stacks, selNode: null, selPath: null, runPath: null, runError: null, runErrorMsg: null };
  if (d.config && typeof d.config === 'object')
    CONFIG_KEYS.forEach(k => { if (d.config && d.config[k] !== undefined) patch[k] = d.config[k]; });
  setState(patch as Partial<typeof S>);
  return { prog, stacks };
}
bus.on('intent.selected', (ev) => {
  clearUndo();
  try {
    const d = storage.patches.get(ev.data.intent as number);
    if (d) applyProgram(d as StoredProgram);
  } catch {  }
});
void storage.ready.then(() => {
  if (touched || S.intent == null || S.screen !== 'patch') return;
  const d = storage.patches.get(S.intent);
  if (d) applyProgram(d as StoredProgram);
});

export function exportProg(): void {
  const data = { v: 1, intent: S.intent, prog: S.prog, stacks: S.stacks, config: configSnapshot() };
  dlText(JSON.stringify(data, null, 2), 'nur-patch.json', 'application/json');
  bus.emit('patch.export', { nodes: countNodes(S.prog) });
}
export interface DslApplyResult { ok: boolean; line?: number; error?: string; blocks?: number; stacks?: number }
let dslImporter: ((text: string) => DslApplyResult) | null = null;
export const registerDslImporter = (fn: (text: string) => DslApplyResult): void => { dslImporter = fn; };
export function importFile(f: File | null | undefined): void {
  if (!f) return;
  const rd = new FileReader();
  rd.onload = () => {
    try {
      const txt = String(rd.result);
      if (/^\s*[[{]/.test(txt)) {
        const data = JSON.parse(txt) as StoredProgram;
        pushUndo();
        const got = applyProgram(data);
        savePatch();
        bus.emit('patch.import', { nodes: countNodes(got.prog), stacks: got.stacks.length });
        appendLog(pad('patch.import') + `${countNodes(got.prog)} blocks · ${got.stacks.length} stacks`);
      } else if (dslImporter) {
        const r = dslImporter(txt);
        if (!r.ok) throw new Error('line ' + (r.line ?? 0) + ': ' + (r.error ?? 'parse failed'));
        bus.emit('patch.import', { nodes: r.blocks ?? 0, stacks: r.stacks ?? 0, dsl: true });
        appendLog(pad('patch.import') + `${r.blocks ?? 0} blocks · ${r.stacks ?? 0} stacks · dsl`);
      } else throw new Error('not a patch file');
    } catch (e) {
      bus.emit('patch.import.error', { message: errMsg(e) });
      appendLog(pad('patch.error') + 'import failed · ' + errMsg(e).slice(0, 50));
      scheduleRender();
    }
  };
  rd.readAsText(f);
}

export function resetProg(): void {
  if (S.intent == null) return;
  pushUndo();
  storage.patches.remove(S.intent);
  const prog = progFor(S.intent);
  setState({ prog, stacks: [], dslMode: false, selNode: null, selPath: null, runPath: null, runError: null, runErrorMsg: null });
  bus.emit('patch.reset', { nodes: countNodes(prog) });
  appendLog(pad('patch.reset') + `${countNodes(prog)} blocks · default program`);
}

export interface DragState {
  item: Block; srcPath: Path | null; started: boolean;
  sx: number; sy: number; x: number; y: number; inside: boolean;
  pre?: Snap;
  refuse?: string | null;
}

function slotRefusal(sl: Slot, item: Block): string | null {
  const beforeMsgs = new Set(flow().errs.values());
  const beforeN = flow().errs.size;
  sl.arr.splice(Math.min(sl.index, sl.arr.length), 0, item);
  const after = flow().errs;
  sl.arr.splice(sl.arr.indexOf(item), 1);
  if (after.size <= beforeN) return null;
  for (const msg of after.values()) if (!beforeMsgs.has(msg)) return msg;
  return 'the shapes do not fit here';
}
export let drag: DragState | null = null;
export let snap: Slot | null = null;

export interface WireDrag {
  srcPath: Path; f: Flow; started: boolean;
  sx: number; sy: number; x: number; y: number;
}
export let wireDrag: WireDrag | null = null;

function blockAt(path: Path): Block | null {
  const L = getList(path);
  return (L && L.arr[L.idx]) || null;
}

function ensureTapName(path: Path): string | null {
  const b = blockAt(path);
  if (!b) return null;
  const cur = tapOf(b);
  if (cur) return cur;
  const name = freshTapName();
  b.c = { ...(b.c || {}), tap: name };
  return name;
}

export function startWire(e: { clientX: number; clientY: number }, srcPath: Path, f: Flow): void {
  const st: WireDrag = { srcPath: srcPath.slice(), f, started: false, sx: e.clientX, sy: e.clientY, x: 0, y: 0 };
  const onMove = (ev: PointerEvent): void => {
    if (!st.started) {
      if (Math.hypot(ev.clientX - st.sx, ev.clientY - st.sy) < 6) return;
      st.started = true;
      wireDrag = st;
    }
    const ws = document.querySelector('[data-ws]');
    if (ws) {
      const r = ws.getBoundingClientRect();
      st.x = ev.clientX - r.left; st.y = ev.clientY - r.top;
    }
    scheduleRender();
  };
  const onUp = (): void => {
    document.removeEventListener('pointermove', onMove);
    document.removeEventListener('pointerup', onUp);
    const was = wireDrag;
    wireDrag = null;
    if (!was || !was.started) { scheduleRender(); return; }
    pushUndo();
    const name = ensureTapName(was.srcPath);
    if (!name) { scheduleRender(); return; }
    let target: LayoutBlockBox | null = null;
    for (const bl of layout().blocks) {
      if (bl.b.k !== 'on') continue;
      if (was.x >= bl.x && was.x <= bl.x + bl.w && was.y >= bl.y && was.y <= bl.y + bl.h) { target = bl; break; }
    }
    if (target) {
      target.b.c = { ...(target.b.c || {}), tap: name };
      bus.emit('patch.tap.wire', { name, to: 'existing' });
    } else {
      S.stacks.push({
        x: Math.max(Math.round(was.x) - 40, 0), y: Math.max(Math.round(was.y) - 20, 0),
        items: [{ k: 'on', c: { tap: name } }],
      });
      bus.emit('patch.tap.wire', { name, to: 'new-stack' });
    }
    appendLog(pad('patch.tap') + "stream '" + name + "' wired");
    savePatch();
    scheduleRender();
  };
  document.addEventListener('pointermove', onMove);
  document.addEventListener('pointerup', onUp);
}

export function startDrag(e: { button?: number; clientX: number; clientY: number }, item: Block, srcPath: Path | null): void {
  if (e.button != null && e.button !== 0) return;
  const st: DragState = { item, srcPath: srcPath ? srcPath.slice() : null, started: false, sx: e.clientX, sy: e.clientY, x: 0, y: 0, inside: false };
  const onMove = (ev: PointerEvent): void => {
    if (!st.started) {
      if (Math.hypot(ev.clientX - st.sx, ev.clientY - st.sy) < 6) return;
      st.started = true;
      st.pre = snapState();
      if (st.srcPath) {
        const L = getList(st.srcPath);
        if (L && L.arr[L.idx]) L.arr.splice(L.idx, 1);
        if (S.selPath && isPrefix(st.srcPath, S.selPath)) { S.selNode = null; S.selPath = null; }
        S.runPath = null; S.runError = null;
      }
      drag = st;
    }
    const ws = document.querySelector('[data-ws]');
    if (ws) {
      const r = ws.getBoundingClientRect();
      st.x = ev.clientX - r.left; st.y = ev.clientY - r.top;
      st.inside = ev.clientX >= r.left && ev.clientX <= r.right && ev.clientY >= r.top && ev.clientY <= r.bottom;
      let best: Slot | null = null, bd = 1e9;
      if (st.inside) {
        for (const sl of layout().slots) {
          const dyy = Math.abs(sl.y - st.y), dxx = st.x - sl.x;
          if (dyy < 30 && dxx > -70 && dxx < 250 && dyy < bd) { bd = dyy; best = sl; }
        }
      }
      st.refuse = null;
      if (best) {
        const why = slotRefusal(best, st.item);
        if (why) { st.refuse = why; best = null; }
      }
      snap = best;
    }
    scheduleRender();
  };
  const onUp = (): void => {
    document.removeEventListener('pointermove', onMove);
    document.removeEventListener('pointerup', onUp);
    const sn = snap;
    drag = null; snap = null;
    if (st.started && S.palOpen) setState({ palOpen: false }, { silent: true });
    if (!st.started) {
      if (st.srcPath) {
        if (S.selPath && pk(S.selPath) === pk(st.srcPath) && S.selNode === st.item.k)
          setState({ selNode: null, selPath: null });
        else setState({ selNode: st.item.k, selPath: st.srcPath });
      } else setState({ selNode: st.item.k, selPath: null });
      return;
    }
    if (sn) {
      if (st.pre) pushUndo(st.pre);
      sn.arr.splice(Math.min(sn.index, sn.arr.length), 0, st.item);
      bus.emit(st.srcPath ? 'patch.block.move' : 'patch.block.add', { node: st.item.k });
    } else if (st.inside) {
      if (st.pre) pushUndo(st.pre);
      S.stacks.push({ x: Math.max(Math.round(st.x) - 50, 0), y: Math.max(Math.round(st.y) - 14, 0), items: [st.item] });
      bus.emit(st.srcPath ? 'patch.block.move' : 'patch.block.add', { node: st.item.k, stack: true });
    } else if (st.srcPath) {
      const L = getList(st.srcPath);
      if (L) L.arr.splice(Math.min(L.idx, L.arr.length), 0, st.item);
      else S.prog.push(st.item);
      bus.emit('patch.block.move', { node: st.item.k, restored: true });
    } else {
      if (st.pre) pushUndo(st.pre);
      S.prog.push(st.item);
      bus.emit('patch.block.add', { node: st.item.k, appended: true });
    }
    pruneStacks();
    fixSelection();
    savePatch();
    scheduleRender();
  };
  document.addEventListener('pointermove', onMove);
  document.addEventListener('pointerup', onUp);
}
