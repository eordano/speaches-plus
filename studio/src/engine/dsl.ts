import { S, setState, type AppState, type Block, type BlockKind, type Stack } from '../state/store';
import { bus } from '../state/bus';
import { clamp } from '../lib/util';
import {
  COLORS, bcNum, bcStr, ifExpr, countNodes, pushUndo, savePatch, registerDslImporter,
  type DslApplyResult,
} from './patch';
import { parseExpr, isBuiltin, isWritableBuiltin, VAR_RE } from './expr';
import { SIG, TAP_RE, tapOf } from './ports';

type DirKey = 'selChat' | 'selStt' | 'selTts' | 'voice' | 'vdText' | 'sysPrompt' | 'eouMs'
  | 'barge' | 'sttFormat' | 'speed' | 'temp' | 'topP' | 'effort' | 'showThinking' | 'fps' | 'frames'
  | 'baseChat' | 'baseStt' | 'baseTts';
export type DslConfig = Partial<Pick<AppState, DirKey>>;
export interface DslDoc { prog: Block[]; stacks: Stack[]; config: DslConfig }

const DIR_DEFAULTS: Pick<AppState, DirKey> = {
  selChat: null, selStt: null, selTts: null, voice: null, vdText: '', sysPrompt: '',
  eouMs: 400, barge: true, sttFormat: 0, speed: 1, temp: 0.7, topP: 0.95,
  effort: 'default', showThinking: false, fps: 1, frames: 3,
  baseChat: '', baseStt: '', baseTts: '',
};
const DIR_KEYS = Object.keys(DIR_DEFAULTS) as DirKey[];
const STT_FORMATS = ['json', 'verbose', 'diarized'];

const bareSafe = (s: string): boolean => /^[^\s'"#]+$/.test(s);
const q = (s: string): string =>
  "'" + s.replace(/\\/g, '\\\\').replace(/'/g, "\\'").replace(/\n/g, '\\n') + "'";
const qIf = (s: string): string => (s && bareSafe(s) ? s : q(s));

function directiveLines(cfg: DslConfig): string[] {
  const out: string[] = [];
  const has = (k: DirKey): boolean => cfg[k] !== undefined && cfg[k] !== DIR_DEFAULTS[k];
  if (has('selChat')) out.push('@chat ' + qIf(String(cfg.selChat)));
  if (has('selStt')) out.push('@stt ' + qIf(String(cfg.selStt)));
  if (has('selTts')) out.push('@tts ' + qIf(String(cfg.selTts)));
  if (has('voice') && cfg.voice) out.push('@voice ' + qIf(String(cfg.voice)));
  if (has('vdText')) out.push('@voice-design ' + q(String(cfg.vdText)));
  if (has('sysPrompt')) out.push('@sys ' + q(String(cfg.sysPrompt)));
  if (has('eouMs')) out.push('@eou ' + Number(cfg.eouMs));
  if (has('barge')) out.push('@barge off');
  if (has('sttFormat')) out.push('@stt-format ' + (STT_FORMATS[Number(cfg.sttFormat)] ?? 'json'));
  if (has('speed')) out.push('@speed ' + Number(cfg.speed));
  if (has('temp')) out.push('@temp ' + Number(cfg.temp));
  if (has('topP')) out.push('@top-p ' + Number(cfg.topP));
  if (has('effort')) out.push('@effort ' + String(cfg.effort));
  if (has('showThinking')) out.push('@thinking on');
  if (has('fps')) out.push('@fps ' + Number(cfg.fps));
  if (has('frames')) out.push('@frames ' + Number(cfg.frames));
  const bases: string[] = [];
  if (has('baseChat')) bases.push('chat=' + cfg.baseChat);
  if (has('baseStt')) bases.push('stt=' + cfg.baseStt);
  if (has('baseTts')) bases.push('tts=' + cfg.baseTts);
  if (bases.length) out.push('@base ' + bases.join(' '));
  return out;
}

function lineFor(b: Block): string {
  const tap = tapOf(b);
  const tapSuffix = tap ? ' tap=' + tap : '';
  switch (b.k) {
    case 'set': return 'set ' + bcStr(b, 'var', 'x') + ' = ' + (bcStr(b, 'expr') || "''");
    case 'append': return 'append ' + bcStr(b, 'var', 'x') + ' ' + (bcStr(b, 'expr') || "''");
    case 'if': return 'if ' + ifExpr(b);
    case 'on': return 'on' + (tap ? ' ' + tap : '');
    case 'repeat': return 'repeat ' + bcNum(b, 'n', 3);
    case 'wait': return 'wait ' + bcNum(b, 'ms', 1000);
    case 'screen': return 'screen' + (b.c && b.c.fps !== undefined ? ' fps=' + bcNum(b, 'fps', 1) : '') + tapSuffix;
    case 'for': {
      let s = 'for';
      if (b.c && b.c.frames !== undefined) s += ' frames=' + bcNum(b, 'frames', 3);
      if (b.c && b.c.fps !== undefined) s += ' fps=' + bcNum(b, 'fps', 1);
      return s + tapSuffix;
    }
    case 'http': {
      let s = 'http';
      if (b.c && b.c.url !== undefined) s += ' url=' + (bcStr(b, 'url') === '' ? "''" : qIf(bcStr(b, 'url')));
      if (b.c && b.c.method !== undefined) s += ' method=' + bcStr(b, 'method', 'GET');
      if (b.c && b.c.into !== undefined) s += ' into=' + bcStr(b, 'into', 'http');
      if (b.c && b.c.body !== undefined) s += ' body=' + q(bcStr(b, 'body'));
      if (b.c && b.c.headers !== undefined) s += ' headers=' + q(bcStr(b, 'headers'));
      return s;
    }
    default: return b.k + (SIG[b.k].gives ? tapSuffix : '');
  }
}

function emitBlocks(items: Block[], depth: number, out: string[]): void {
  for (const b of items) {
    out.push('  '.repeat(depth) + lineFor(b));
    if (b.children) emitBlocks(b.children, depth + 1, out);
  }
}

export function emitDoc(doc: DslDoc): string {
  const out: string[] = directiveLines(doc.config);
  if (out.length && doc.prog.length) out.push('');
  emitBlocks(doc.prog, 0, out);
  for (const st of doc.stacks) {
    out.push('');
    out.push('--- stack ' + Math.round(st.x) + ' ' + Math.round(st.y));
    emitBlocks(st.items, 0, out);
  }
  return out.join('\n') + '\n';
}

export function emitDsl(): string {
  const config: DslConfig = {};
  DIR_KEYS.forEach(k => { (config as Record<string, unknown>)[k] = S[k]; });
  return emitDoc({ prog: S.prog, stacks: S.stacks, config });
}

export const canonicalLine = (b: Block): string => lineFor(b);

interface LineErr { line: number; error: string }
const fail = (line: number, error: string): never => { throw { line, error } as LineErr; };

function scanValue(rest: string, line: number): { v: string; rest: string } {
  const c0 = rest[0];
  if (c0 === "'" || c0 === '"') {
    let out = '', i = 1;
    for (;;) {
      if (i >= rest.length) fail(line, 'unterminated string');
      const c = rest[i] as string;
      if (c === c0) { i++; break; }
      if (c === '\\') {
        const e = rest[i + 1];
        if (e === 'n') out += '\n';
        else if (e === '\\' || e === "'" || e === '"') out += e;
        else fail(line, "bad escape '\\" + (e ?? '') + "'");
        i += 2;
      } else { out += c; i++; }
    }
    return { v: out, rest: rest.slice(i).trimStart() };
  }
  const m = /^\S+/.exec(rest);
  if (!m) fail(line, 'missing value');
  return { v: (m as RegExpExecArray)[0], rest: rest.slice((m as RegExpExecArray)[0].length).trimStart() };
}

function parseKV(rest: string, line: number, allowed: string[]): Record<string, string> {
  const out: Record<string, string> = {};
  let s = rest.trim();
  while (s.length) {
    const m = /^([A-Za-z][A-Za-z0-9-]*)=/.exec(s);
    if (!m) fail(line, "expected key=value, got '" + s.slice(0, 20) + "'");
    const key = (m as RegExpExecArray)[1] as string;
    if (allowed.indexOf(key) < 0) fail(line, "unknown argument '" + key + "'");
    const got = scanValue(s.slice((m as RegExpExecArray)[0].length), line);
    out[key] = got.v;
    s = got.rest;
  }
  return out;
}

const intArg = (rest: string, line: number, what: string, lo: number, hi: number): number => {
  if (!/^\d+$/.test(rest.trim())) fail(line, what + ' needs a whole number');
  return clamp(parseInt(rest.trim(), 10), lo, hi);
};
const numArg = (rest: string, line: number, what: string, lo: number, hi: number): number => {
  const n = Number(rest.trim());
  if (!isFinite(n)) fail(line, what + ' needs a number');
  return clamp(n, lo, hi);
};
const httpsArg = (v: string, line: number): string => {
  if (!/^https:\/\//.test(v)) fail(line, 'endpoint must be https://');
  return v.replace(/\/+$/, '');
};

const tapArg = (v: string, line: number): string => {
  if (!TAP_RE.test(v)) fail(line, 'tap needs a stream name (letters, digits, - or _)');
  return v;
};

function checkExpr(src: string, line: number): string {
  const p = parseExpr(src);
  if ('error' in p) fail(line, 'expression — ' + p.error + ' (at ' + p.pos + ')');
  return src;
}

function parseDirective(text: string, line: number, config: DslConfig): void {
  const sp = text.indexOf(' ');
  const name = sp < 0 ? text : text.slice(0, sp);
  const rest = sp < 0 ? '' : text.slice(sp + 1).trim();
  const one = (): string => {
    if (!rest) fail(line, name + ' needs a value');
    const got = scanValue(rest, line);
    if (got.rest) fail(line, name + " — unexpected '" + got.rest.slice(0, 20) + "'");
    return got.v;
  };
  switch (name) {
    case '@chat': config.selChat = one(); break;
    case '@stt': config.selStt = one(); break;
    case '@tts': config.selTts = one(); break;
    case '@voice': config.voice = one(); break;
    case '@voice-design': config.vdText = one(); break;
    case '@sys': config.sysPrompt = one(); break;
    case '@eou': config.eouMs = intArg(rest, line, '@eou', 80, 1200); break;
    case '@barge': {
      const v = one();
      if (v !== 'on' && v !== 'off') fail(line, '@barge needs on or off');
      config.barge = v === 'on';
      break;
    }
    case '@stt-format': {
      const i = STT_FORMATS.indexOf(one());
      if (i < 0) fail(line, '@stt-format needs json, verbose or diarized');
      config.sttFormat = i;
      break;
    }
    case '@speed': config.speed = numArg(rest, line, '@speed', 0.5, 2); break;
    case '@temp': config.temp = numArg(rest, line, '@temp', 0, 1.5); break;
    case '@top-p': config.topP = numArg(rest, line, '@top-p', 0.5, 1); break;
    case '@effort': {
      const v = one();
      if (!['default', 'low', 'medium', 'high', 'xhigh'].includes(v)) {
        fail(line, '@effort needs default, low, medium, high or xhigh');
      }
      config.effort = v;
      break;
    }
    case '@thinking': {
      const v = one();
      if (v !== 'on' && v !== 'off') fail(line, '@thinking needs on or off');
      config.showThinking = v === 'on';
      break;
    }
    case '@fps': config.fps = intArg(rest, line, '@fps', 1, 5); break;
    case '@frames': config.frames = intArg(rest, line, '@frames', 1, 10); break;
    case '@base': {
      const kv = parseKV(rest, line, ['chat', 'stt', 'tts']);
      if (kv.chat !== undefined) config.baseChat = httpsArg(kv.chat, line);
      if (kv.stt !== undefined) config.baseStt = httpsArg(kv.stt, line);
      if (kv.tts !== undefined) config.baseTts = httpsArg(kv.tts, line);
      break;
    }
    default: fail(line, "unknown directive '" + name + "'");
  }
}

function parseBlockLine(text: string, line: number): Block {
  const sp = text.search(/\s/);
  const kind = sp < 0 ? text : text.slice(0, sp);
  const rest = sp < 0 ? '' : text.slice(sp + 1).trim();
  if (!(kind in COLORS)) fail(line, "unknown block '" + kind + "'");
  const k = kind as BlockKind;
  const b: Block = { k };
  if (k === 'if' || k === 'for' || k === 'repeat') b.children = [];
  switch (k) {
    case 'set': case 'append': {
      const m = (k === 'set' ? /^([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.+)$/ : /^([A-Za-z_][A-Za-z0-9_]*)\s+(.+)$/).exec(rest);
      if (!m) fail(line, k === 'set' ? 'set needs: set <var> = <expr>' : 'append needs: append <var> <expr>');
      const name = (m as RegExpExecArray)[1] as string;
      if (isBuiltin(name) && (k === 'append' || !isWritableBuiltin(name)))
        fail(line, "'" + name + "' is read-only");
      b.c = { var: name, expr: checkExpr(((m as RegExpExecArray)[2] as string).trim(), line) };
      break;
    }
    case 'if':
      if (!rest) fail(line, 'if needs an expression');
      b.c = { expr: checkExpr(rest, line) };
      break;
    case 'repeat': b.c = { n: intArg(rest, line, 'repeat', 1, 20) }; break;
    case 'wait': b.c = { ms: intArg(rest, line, 'wait', 0, 30000) }; break;
    case 'on': {
      if (rest && !TAP_RE.test(rest)) fail(line, 'on needs a stream name (letters, digits, - or _)');
      b.c = { tap: rest };
      break;
    }
    case 'screen': {
      const kv = parseKV(rest, line, ['fps', 'tap']);
      const c: Record<string, string | number> = {};
      if (kv.fps !== undefined) c.fps = intArg(kv.fps, line, 'fps', 1, 5);
      if (kv.tap !== undefined) c.tap = tapArg(kv.tap, line);
      if (Object.keys(c).length) b.c = c;
      break;
    }
    case 'for': {
      const kv = parseKV(rest, line, ['frames', 'fps', 'tap']);
      const c: Record<string, string | number> = {};
      if (kv.frames !== undefined) c.frames = intArg(kv.frames, line, 'frames', 1, 10);
      if (kv.fps !== undefined) c.fps = intArg(kv.fps, line, 'fps', 1, 5);
      if (kv.tap !== undefined) c.tap = tapArg(kv.tap, line);
      if (Object.keys(c).length) b.c = c;
      break;
    }
    case 'http': {
      const kv = parseKV(rest, line, ['url', 'method', 'into', 'body', 'headers']);
      const c: Record<string, string> = {};
      if (kv.url !== undefined) c.url = kv.url;
      if (kv.method !== undefined) {
        if (kv.method !== 'GET' && kv.method !== 'POST') fail(line, 'method needs GET or POST');
        c.method = kv.method;
      }
      if (kv.into !== undefined) {
        if (!VAR_RE.test(kv.into) || isBuiltin(kv.into)) fail(line, "'" + kv.into + "' is not a usable variable name");
        c.into = kv.into;
      }
      if (kv.body !== undefined) c.body = kv.body;
      if (kv.headers !== undefined) c.headers = kv.headers;
      b.c = c;
      break;
    }
    default: {
      if (!rest) break;
      if (!SIG[k].gives) fail(line, kind + ' takes no arguments');
      const kv = parseKV(rest, line, ['tap']);
      if (kv.tap !== undefined) b.c = { tap: tapArg(kv.tap, line) };
    }
  }
  return b;
}

export function parseDsl(text: string): { ok: true; doc: DslDoc } | { ok: false; line: number; error: string } {
  try {
    const config: DslConfig = {};
    const prog: Block[] = [];
    const stacks: Stack[] = [];
    let containers: Array<Block[]> = [prog];
    let sawBlock = false;
    const lines = text.split('\n');
    for (let i = 0; i < lines.length; i++) {
      const raw = (lines[i] as string).replace(/\r$/, '');
      const n = i + 1;
      if (!raw.trim() || raw.trim().startsWith('#')) continue;
      const stackM = /^---\s+stack\s+(-?\d+)\s+(-?\d+)\s*$/.exec(raw.trim());
      if (stackM) {
        const st: Stack = {
          x: Math.max(0, parseInt((stackM as RegExpExecArray)[1] as string, 10)),
          y: Math.max(0, parseInt((stackM as RegExpExecArray)[2] as string, 10)),
          items: [],
        };
        stacks.push(st);
        containers = [st.items];
        sawBlock = true;
        continue;
      }
      if (raw.startsWith('@')) {
        if (sawBlock) fail(n, 'directives go before the first block');
        parseDirective(raw.trim(), n, config);
        continue;
      }
      const im = /^ */.exec(raw) as RegExpExecArray;
      const indent = im[0].length;
      if (/^\t/.test(raw)) fail(n, 'indent with 2 spaces, not tabs');
      if (indent % 2) fail(n, 'indent must be a multiple of 2 spaces');
      const depth = indent / 2;
      if (depth >= containers.length) fail(n, 'unexpected indent — no open block above');
      containers = containers.slice(0, depth + 1);
      const b = parseBlockLine(raw.trim(), n);
      (containers[depth] as Block[]).push(b);
      if (b.children) containers.push(b.children);
      sawBlock = true;
    }
    const empty = stacks.findIndex(st => !st.items.length);
    if (empty >= 0) fail(lines.length, 'stack ' + (empty + 1) + ' has no blocks');
    return { ok: true, doc: { prog, stacks, config } };
  } catch (e) {
    const le = e as LineErr;
    return { ok: false, line: le.line || 0, error: le.error || 'parse failed' };
  }
}

let burstSnapped = false;
export const beginDslBurst = (): void => { burstSnapped = false; };

export function applyDsl(text: string, snap: 'burst' | 'always'): DslApplyResult {
  const p = parseDsl(text);
  if (!p.ok) return { ok: false, line: p.line, error: p.error };
  if (snap === 'always') pushUndo();
  else if (!burstSnapped) { pushUndo(); burstSnapped = true; }
  const patch: Record<string, unknown> = {
    prog: p.doc.prog, stacks: p.doc.stacks,
    selNode: null, selPath: null, runPath: null, runError: null, runErrorMsg: null,
  };
  DIR_KEYS.forEach(k => {
    patch[k] = p.doc.config[k] !== undefined ? p.doc.config[k] : DIR_DEFAULTS[k];
  });
  setState(patch as Partial<AppState>);
  savePatch();
  const blocks = countNodes(p.doc.prog);
  bus.emit('patch.dsl.apply', { blocks, stacks: p.doc.stacks.length });
  return { ok: true, blocks, stacks: p.doc.stacks.length };
}

registerDslImporter((txt) => applyDsl(txt, 'always'));
