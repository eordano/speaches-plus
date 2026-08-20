import type { Block, BlockKind, Path, Stack } from '../state/store';

export type BaseType = 'audio' | 'text' | 'image' | 'speech' | 'any';
export type Card = 'stream' | 'one';
export interface Flow { base: BaseType; card: Card }
export interface PortSig { needs: BaseType | null; gives: BaseType | null; card: Card }

export const PORT_COLORS: Record<BaseType, string> = {
  audio: '#C7902B', text: '#4E90E0', image: '#7A4848', speech: '#8B5CC7', any: '#868D84',
};

export const SIG: Record<BlockKind, PortSig> = {
  mic: { needs: null, gives: 'audio', card: 'stream' },
  keys: { needs: null, gives: 'text', card: 'stream' },
  screen: { needs: null, gives: 'image', card: 'stream' },
  cam: { needs: null, gives: 'image', card: 'stream' },
  on: { needs: null, gives: null, card: 'stream' },
  eou: { needs: 'audio', gives: null, card: 'one' },
  stt: { needs: 'audio', gives: 'text', card: 'stream' },
  translate: { needs: 'text', gives: 'text', card: 'one' },
  ocr: { needs: 'image', gives: 'text', card: 'one' },
  agent: { needs: null, gives: 'text', card: 'one' },
  tts: { needs: 'text', gives: 'speech', card: 'one' },
  spk: { needs: 'speech', gives: null, card: 'one' },
  store: { needs: null, gives: null, card: 'one' },
  cancel: { needs: null, gives: null, card: 'one' },
  if: { needs: null, gives: null, card: 'one' },
  for: { needs: 'image', gives: 'image', card: 'stream' },
  set: { needs: null, gives: null, card: 'one' },
  append: { needs: null, gives: null, card: 'one' },
  repeat: { needs: null, gives: null, card: 'one' },
  wait: { needs: null, gives: null, card: 'one' },
  http: { needs: null, gives: null, card: 'one' },
};

export const TAP_RE = /^[a-z][a-z0-9_-]{0,23}$/i;
export const tapOf = (b: Block): string => {
  const v = b.c ? b.c.tap : undefined;
  return typeof v === 'string' ? v.trim() : '';
};

export interface FlowResult {
  errs: Map<string, string>;
  out: Map<string, Flow>;
  tail: Flow | null;
}

interface Prog { prog: Block[]; stacks: Stack[] }

function walk(
  items: Block[], base: Path, avail: Set<BaseType>, cur: Flow | null,
  labels: Record<string, string>, r: FlowResult,
): Flow | null {
  items.forEach((b, i) => {
    const path = base.concat(i);
    const key = JSON.stringify(path);
    const sig = SIG[b.k];
    const rootList = base.length === 0 || (base.length === 2 && base[0] === 'S');
    if (b.k === 'on' && !(rootList && i === 0))
      r.errs.set(key, "'when stream' must start its own stack");
    if (sig.needs && !avail.has(sig.needs) && !avail.has('any')) {
      const label = labels[b.k] || b.k;
      r.errs.set(key, label + ' needs ' + sig.needs +
        (cur ? ' — upstream gives ' + cur.base : ' — nothing upstream provides it'));
    }
    if (b.children) walk(b.children, path, b.k === 'if' ? new Set(avail) : avail, cur, labels, r);
    if (sig.gives) {
      avail.add(sig.gives);
      cur = { base: sig.gives, card: sig.card };
    }
    if (cur) r.out.set(key, cur);
  });
  return cur;
}

const isSourceHat = (b: Block | undefined): boolean =>
  !!b && (b.k === 'mic' || b.k === 'keys' || b.k === 'screen' || b.k === 'cam');

export function checkProgram(p: Prog, labels: Record<string, string>): FlowResult {
  const r: FlowResult = { errs: new Map(), out: new Map(), tail: null };
  const taps = new Map<string, Flow>();
  const lists: Array<{ items: Block[]; base: Path }> = [{ items: p.prog, base: [] }];
  p.stacks.forEach((st, i) => lists.push({ items: st.items, base: ['S', i] }));

  const collectTaps = (items: Block[], avail: Set<BaseType>, cur: Flow | null): void => {
    for (const b of items) {
      const sig = SIG[b.k];
      if (b.children) collectTaps(b.children, new Set(avail), cur);
      if (sig.gives) { avail.add(sig.gives); cur = { base: sig.gives, card: sig.card }; }
      const name = tapOf(b);
      if (name && cur && !taps.has(name)) taps.set(name, { base: cur.base, card: 'stream' });
    }
  };
  for (let pass = 0; pass < 3; pass++) {
    for (const L of lists) {
      const first = L.items[0];
      if (!first) continue;
      const seed = first.k === 'on' ? taps.get(tapOf(first)) : null;
      const avail = new Set<BaseType>(seed ? [seed.base] : []);
      collectTaps(L.items, avail, seed ? { base: seed.base, card: 'stream' } : null);
    }
  }

  for (const L of lists) {
    const first = L.items[0];
    const live = isSourceHat(first) || (first && first.k === 'on');
    if (!live) continue;
    let seed: Flow | null = null;
    if (first && first.k === 'on') {
      const name = tapOf(first);
      const key = JSON.stringify(L.base.concat(0));
      if (!name) r.errs.set(key, "'when stream' needs a stream name — set it in the config panel");
      else {
        seed = taps.get(name) || null;
        if (!seed) r.errs.set(key, "no block publishes a stream named '" + name + "'");
      }
    }
    const avail = new Set<BaseType>(seed ? [seed.base] : []);
    r.tail = walk(L.items, L.base, avail, seed, labels, r);
  }
  return r;
}

export function flowFor(items: Block[], base: Path, seed: Flow | null, labels: Record<string, string>): FlowResult {
  const r: FlowResult = { errs: new Map(), out: new Map(), tail: null };
  const avail = new Set<BaseType>(seed ? [seed.base] : []);
  r.tail = walk(items, base, avail, seed, labels, r);
  return r;
}

export const glyphSvg = (f: Flow, cx: number, cy: number, r: number): string => {
  const c = PORT_COLORS[f.base];
  if (f.card === 'stream')
    return `<path d="M${cx},${cy - r - 1.5} L${cx + r + 1.5},${cy} L${cx},${cy + r + 1.5} L${cx - r - 1.5},${cy} Z" fill="${c}" stroke="#1A1A1A" stroke-width="1.1"/>`;
  return `<circle cx="${cx}" cy="${cy}" r="${r}" fill="${c}" stroke="#1A1A1A" stroke-width="1.1"/>`;
};
