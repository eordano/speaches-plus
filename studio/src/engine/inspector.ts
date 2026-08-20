import { bus, type BusEvent } from '../state/bus';
import { S, scheduleRender, type Turn } from '../state/store';

export interface RecEvent { dt: number; kind: string; data: Record<string, unknown> }
export interface TurnRec { t0: number; index: number | null; events: RecEvent[] }
export const REC: { recs: Record<number, TurnRec>; cur: TurnRec | null } = { recs: {}, cur: null };

const START: Record<string, 1> = { 'msg.user': 1, 'mic.start': 1, 'run.start': 1 };
const APP_LEVEL = /^(app\.|mode\.|nav\.|session\.|intent\.|patch\.|log\.|audio\.|busy$|tts\.preview$)/;
bus.on('*', (ev: BusEvent) => {
  const k = ev.kind;
  if (!APP_LEVEL.test(k)) {
    if (START[k] && (!REC.cur || REC.cur.index != null)) REC.cur = { t0: ev.t, index: null, events: [] };
    if (!REC.cur && k === 'turn.recorded') REC.cur = { t0: ev.t, index: null, events: [] };
    if (REC.cur) {
      REC.cur.events.push({ dt: Math.max(0, ev.t - REC.cur.t0), kind: k, data: ev.data || {} });
      if (k === 'turn.recorded' && ev.data && ev.data.index != null) {
        REC.cur.index = ev.data.index as number;
        REC.recs[ev.data.index as number] = REC.cur;
      }
    }
    if (k === 'run.done' && REC.cur && REC.cur.index == null) REC.cur = null;
  }
  if (S.screen === 'duet' && S.view === 'insp') scheduleRender();
});
bus.on('intent.selected', () => { REC.recs = {}; REC.cur = null; });
bus.on('session.resumed', () => { REC.recs = {}; REC.cur = null; });

const evColor = (k: string): string => {
  if (/error|cancel|abort/.test(k)) return '#A8201A';
  const m: Record<string, string> = {
    mic: '#8B6B2A', frame: '#8B6B2A', eou: '#5A6B7A', turn: '#5A6B7A', vad: '#5A6B7A',
    stt: '#5A6E5A', tts: '#5A6E5A', translate: '#5A6E5A',
    agent: '#6E5A3C', ocr: '#7A4848', spk: '#2B2B2B', store: '#2B2B2B',
    msg: '#2B2B2B', run: '#645C55',
  };
  return m[k.split('.')[0] ?? ''] || '#645C55';
};
const payOf = (data: Record<string, unknown>): string => {
  let pay = '';
  try { pay = JSON.stringify(data) || ''; } catch { pay = ''; }
  return pay === '{}' ? '' : pay;
};

export interface Span { label: string; a: number; b: number; c: string }
export interface Partial_ { x: number; label: string }
export interface Marker { x: number; c: string; label: string | null; tip: [string, string] }
export interface EventModel {
  dur: number; spans: Span[]; LANES: string[]; partials: Partial_[]; markers: Marker[];
  m: { eou: number | null; tok1: number | null; audio1: number | null; cancel: number | null };
  wall: number | null;
}
export function eventModel(t: Partial<Turn>, rec: TurnRec | null): EventModel {
  const evs = rec ? rec.events : [];
  const first = (k: string): RecEvent | null => { for (const e of evs) if (e.kind === k) return e; return null; };
  const dtOf = (k: string): number | null => { const e = first(k); return e ? e.dt : null; };
  const dm = (k: string, f: string): number | null => {
    const e = first(k);
    return e && e.data[f] != null ? (e.data[f] as number) : null;
  };
  const last = evs.length ? (evs[evs.length - 1]?.dt ?? 0) : 0;
  const dur = Math.max(1000, Math.ceil(Math.max(last, 800) / 500) * 500);
  const X = (ms: number): number => ms / dur * 100;

  const spans: Span[] = [];
  const addSpan = (label: string, a: number | null, b: number | null, c: string): void => {
    if (a == null) return;
    if (b == null || b <= a) b = Math.min(a + 120, dur);
    spans.push({ label, a: X(a), b: X(b), c });
  };
  addSpan('mic', dtOf('mic.start'), dtOf('mic.stop'), '#8B6B2A');
  addSpan('agent', dtOf('agent.start'), dtOf('agent.done'), '#6E5A3C');
  addSpan('tts', dtOf('tts.start'), dtOf('tts.done'), '#5A6E5A');
  addSpan('playback', dtOf('spk.play'),
    dtOf('spk.done') != null ? dtOf('spk.done') : dtOf('tts.cancel'), '#5A6B7A');
  const LANES = ['mic', 'agent', 'tts', 'playback'];

  const partials: Partial_[] = [];
  evs.forEach(e => {
    if (e.kind === 'stt.final') partials.push({
      x: X(e.dt),
      label: 'final: “' + String(e.data.text ?? '') + '”' + (e.data.ms != null ? ' · ' + String(e.data.ms) + ' ms' : ''),
    });
  });

  const SKIP: Record<string, 1> = { 'agent.token': 1, 'turn.recorded': 1, 'run.node.start': 1 };
  const markers: Marker[] = [];
  evs.forEach(e => {
    if (SKIP[e.kind]) return;
    let label: string | null = null;
    if (e.kind === 'eou.detected') label = 'eou ' + (e.data.silenceMs != null ? String(e.data.silenceMs) : '');
    else if (e.kind === 'agent.first_token') label = 'tok₁ ' + (e.data.ms != null ? String(e.data.ms) : '');
    else if (e.kind === 'tts.done') label = 'tts ' + (e.data.ms != null ? String(e.data.ms) : '');
    else if (e.kind === 'tts.cancel') label = 'barge';
    const pay = payOf(e.data);
    markers.push({ x: X(e.dt), c: evColor(e.kind), label, tip: [e.kind + ' · t+' + e.dt + ' ms', pay.slice(0, 140)] });
  });

  const nz = (a: number | null | undefined, b: number | null | undefined): number | null =>
    a != null ? a : (b != null ? b : null);
  const m = {
    eou: nz(dm('eou.detected', 'silenceMs'), t.eou),
    tok1: nz(dm('agent.first_token', 'ms'), t.tok1),
    audio1: nz(t.audio1, dm('tts.done', 'ms')),
    cancel: nz(dm('tts.cancel', 'ms'), t.cancel),
  };
  return { dur, spans, LANES, partials, markers, m, wall: evs.length ? last : null };
}
