const CURATED_ABBREVS = [
  'adm', 'al', 'apr', 'approx', 'aug', 'ave', 'b.a', 'blvd', 'brig', 'capt', 'cf', 'cmdr',
  'co', 'col', 'corp', 'cpl', 'dec', 'dept', 'dr', 'e.g', 'ed', 'est', 'et', 'etc', 'feb',
  'fig', 'figs', 'fri', 'ft', 'gen', 'gov', 'hon', 'hr', 'i.e', 'inc', 'jan', 'jr', 'jul',
  'jun', 'lt', 'ltd', 'm.d', 'maj', 'mar', 'messrs', 'mfg', 'mgr', 'mlle', 'mme', 'mon',
  'mr', 'mrs', 'ms', 'msgr', 'mt', 'nov', 'oct', 'p.m', 'ph.d', 'pp', 'prof', 'pvt', 'rep',
  'rev', 'rd', 'sat', 'sen', 'sep', 'sept', 'sgt', 'sr', 'st', 'sun', 'thu', 'thurs', 'tue',
  'tues', 'u.k', 'u.n', 'u.s', 'u.s.a', 'univ', 'v', 'vol', 'vols', 'vs', 'wed',
];

const ORTHO_BEG_UC = 1 << 1;
const ORTHO_MID_UC = 1 << 2;
const ORTHO_UNK_UC = 1 << 3;
const ORTHO_BEG_LC = 1 << 4;
const ORTHO_MID_LC = 1 << 5;
const ORTHO_UNK_LC = 1 << 6;
const ORTHO_UC = ORTHO_BEG_UC | ORTHO_MID_UC | ORTHO_UNK_UC;
const ORTHO_LC = ORTHO_BEG_LC | ORTHO_MID_LC | ORTHO_UNK_LC;

interface Params {
  abbrevTypes: Set<string>;
  collocations: Set<string>;
  sentStarters: Set<string>;
  orthoContext: Map<string, number>;
}

interface Token {
  text: string;
  start: number;
  end: number;
  typ: string;
  periodFinal: boolean;
  sentbreak: boolean;
  abbr: boolean;
  ellipsis: boolean;
}

const isUpper = (r: string): boolean => /\p{Lu}/u.test(r);
const isLower = (r: string): boolean => /\p{Ll}/u.test(r);
const isLetter = (r: string): boolean => /\p{L}/u.test(r);
const isSpace = (r: string): boolean => /\s/u.test(r);

const typeNoPeriod = (t: Token): string =>
  t.typ.length > 1 && t.typ.endsWith('.') ? t.typ.slice(0, -1) : t.typ;
const typeNoSentPeriod = (t: Token): string => (t.sentbreak ? typeNoPeriod(t) : t.typ);
const firstRune = (t: Token): string => [...t.text][0] || '';
const firstUpper = (t: Token): boolean => isUpper(firstRune(t));
const firstLower = (t: Token): boolean => isLower(firstRune(t));

function isEllipsisTok(t: Token): boolean {
  if (t.text === '…') return true;
  let dots = 0;
  for (const r of t.text) {
    if (r === '.') dots++;
    else if (r !== ' ') return false;
  }
  return dots >= 2;
}

function isInitialTok(t: Token): boolean {
  const runes = [...t.text];
  return runes.length === 2 && isLetter(runes[0] as string) && runes[1] === '.';
}

function isNumeric(s: string): boolean {
  const runes = [...s];
  let i = 0;
  if (runes[i] === '-') i++;
  if (runes[i] === '.' || runes[i] === ',') i++;
  const d = runes[i];
  if (d === undefined || d < '0' || d > '9') return false;
  i++;
  for (; i < runes.length; i++) {
    const r = runes[i] as string;
    if (!((r >= '0' && r <= '9') || r === ',' || r === '.' || r === '-')) return false;
  }
  return true;
}

function newToken(text: string, start: number, end: number): Token {
  const lower = text.toLowerCase();
  return {
    text, start, end,
    typ: isNumeric(lower) ? '##number##' : lower,
    periodFinal: text.endsWith('.'),
    sentbreak: false, abbr: false, ellipsis: false,
  };
}

const NON_WORD = '?!)";}]*:@\'({[';
const WORD_START_EXCLUDE = '("`{[:;&#*@)}]-,';

function tokenizeText(text: string): Token[] {
  const out: Token[] = [];
  const chars: Array<{ off: number; r: string }> = [];
  for (let off = 0; off < text.length; ) {
    const r = String.fromCodePoint(text.codePointAt(off) as number);
    chars.push({ off, r });
    off += r.length;
  }
  const n = chars.length;
  let i = 0;
  while (i < n) {
    const c = (chars[i] as { r: string }).r;
    if (isSpace(c)) { i++; continue; }
    const startI = i;
    if ((c === '-' || c === '.') && i + 1 < n && chars[i + 1]?.r === c) {
      while (i < n && chars[i]?.r === c) i++;
    } else if (c === '…') {
      i++;
    } else if (!WORD_START_EXCLUDE.includes(c)) {
      i++;
      while (i < n) {
        const d = (chars[i] as { r: string }).r;
        if (isSpace(d) || NON_WORD.includes(d) || d === '…') break;
        if ((d === '-' || d === '.') && i + 1 < n && chars[i + 1]?.r === d) break;
        if (d === ',') {
          if (i + 1 >= n) break;
          const x = (chars[i + 1] as { r: string }).r;
          if (isSpace(x) || NON_WORD.includes(x)) break;
        }
        i++;
      }
    } else {
      i++;
    }
    const sb = (chars[startI] as { off: number }).off;
    const eb = i < n ? (chars[i] as { off: number }).off : text.length;
    out.push(newToken(text.slice(sb, eb), sb, eb));
  }
  return out;
}

function firstPass(t: Token, p: Params): void {
  if (t.text === '.' || t.text === '!' || t.text === '?') { t.sentbreak = true; return; }
  if (isEllipsisTok(t)) { t.ellipsis = true; return; }
  if (t.periodFinal && !t.text.endsWith('..')) {
    const base = t.text.slice(0, -1).toLowerCase();
    const dash = base.lastIndexOf('-');
    const lastDash = dash >= 0 ? base.slice(dash + 1) : base;
    if (p.abbrevTypes.has(base) || p.abbrevTypes.has(lastDash)) t.abbr = true;
    else t.sentbreak = true;
  }
}

function orthoHeuristic(p: Params, t: Token): [boolean, boolean] {
  if ([';', ':', ',', '.', '!', '?'].includes(t.text)) return [false, true];
  const ortho = p.orthoContext.get(typeNoSentPeriod(t)) || 0;
  if (firstUpper(t) && (ortho & ORTHO_LC) !== 0 && (ortho & ORTHO_MID_UC) === 0) return [true, true];
  if (firstLower(t) && ((ortho & ORTHO_UC) !== 0 || (ortho & ORTHO_BEG_LC) === 0)) return [false, true];
  return [false, false];
}

function secondPass(t1: Token, t2: Token, p: Params): void {
  if (!t1.periodFinal) return;
  const typ = typeNoPeriod(t1);
  const nextTyp = typeNoSentPeriod(t2);
  const tokIsInitial = isInitialTok(t1);

  if (p.collocations.has(typ + '\t' + nextTyp)) {
    t1.sentbreak = false;
    t1.abbr = true;
    return;
  }

  if ((t1.abbr || t1.ellipsis) && !tokIsInitial) {
    const [starter, known] = orthoHeuristic(p, t2);
    if (known && starter) { t1.sentbreak = true; return; }
    if (firstUpper(t2) && p.sentStarters.has(nextTyp)) { t1.sentbreak = true; return; }
  }

  if (tokIsInitial || typ === '##number##') {
    const [starter, known] = orthoHeuristic(p, t2);
    if (known && !starter) {
      t1.sentbreak = false;
      t1.abbr = true;
    } else if (!known && tokIsInitial && firstUpper(t2) &&
      ((p.orthoContext.get(nextTyp) || 0) & ORTHO_LC) === 0) {
      t1.sentbreak = false;
      t1.abbr = true;
    }
  }
}

export interface SentRange { start: number; end: number }

const isCloser = (r: string): boolean => ['"', "'", ')', ']', '}', '”', '’'].includes(r);

function realign(text: string, ranges: SentRange[]): SentRange[] {
  let i = 0;
  while (i + 1 < ranges.length) {
    const next = ranges[i + 1] as SentRange;
    let p = next.start;
    while (p < next.end) {
      const r = String.fromCodePoint(text.codePointAt(p) as number);
      if (!isCloser(r)) break;
      p += r.length;
    }
    if (p > next.start) {
      let afterOK = p >= text.length || text.startsWith('--', p);
      if (!afterOK) afterOK = isSpace(String.fromCodePoint(text.codePointAt(p) as number));
      if (afterOK) {
        (ranges[i] as SentRange).end = p;
        let q = p;
        while (q < next.end) {
          const r = String.fromCodePoint(text.codePointAt(q) as number);
          if (!isSpace(r)) break;
          q += r.length;
        }
        if (q >= next.end) { ranges.splice(i + 1, 1); continue; }
        next.start = q;
      }
    }
    i++;
  }
  return ranges;
}

let englishParams: Params | null = null;
function english(): Params {
  if (!englishParams) {
    englishParams = {
      abbrevTypes: new Set(CURATED_ABBREVS),
      collocations: new Set(),
      sentStarters: new Set(),
      orthoContext: new Map(),
    };
  }
  return englishParams;
}

function sentenceRanges(text: string): SentRange[] {
  const p = english();
  const toks = tokenizeText(text);
  for (const t of toks) firstPass(t, p);
  for (let i = 0; i + 1 < toks.length; i++) secondPass(toks[i] as Token, toks[i + 1] as Token, p);
  const ranges: SentRange[] = [];
  let start = -1, lastEnd = 0;
  for (const t of toks) {
    if (start < 0) start = t.start;
    lastEnd = t.end;
    if (t.sentbreak) { ranges.push({ start, end: lastEnd }); start = -1; }
  }
  if (start >= 0) ranges.push({ start, end: lastEnd });
  return realign(text, ranges);
}

export const sentences = (text: string): string[] =>
  sentenceRanges(text).map(r => text.slice(r.start, r.end));

const EMOJI_RE = /[✂-➰]|[\u{1F300}-\u{1FAFF}]/gu;

export const stripMarkdownEmphasis = (s: string): string => s
  .replace(/`{3,}\w*/g, ' ')
  .replace(/`([^`]*)`/g, '$1')
  .replace(/\*\*(.*?)\*\*/g, '$1')
  .replace(/\*(.*?)\*/g, '$1')
  .replace(/__(.*?)__/g, '$1')
  .replace(/_(.*?)_/g, '$1')
  .replace(/^#{1,4}\s+/gm, '')
  .replace(/\[([^\]]+)\]\([^)]*\)/g, '$1');

export const normalizeForSpeech = (s: string): string => s
  .replace(EMOJI_RE, '')
  .replace(/[\r\n]+/g, ' ')
  .replace(/\s+/g, ' ')
  .trim();

const SPEECH_CHUNK_CHARS = 300;

export function capChunk(sentence: string, maxChars = SPEECH_CHUNK_CHARS): string[] {
  if (sentence.length <= maxChars) return [sentence];
  const out: string[] = [];
  let cur = '';
  for (const word of sentence.split(/\s+/)) {
    if (!cur) cur = word;
    else if (cur.length + word.length + 1 <= maxChars) cur += ' ' + word;
    else { out.push(cur); cur = word; }
  }
  if (cur) out.push(cur);
  return out;
}
