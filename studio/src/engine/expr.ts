import type { Val } from '../state/store';

export class ExprError extends Error {}

export type Ast =
  | { t: 'lit'; v: Val }
  | { t: 'var'; name: string }
  | { t: 'call'; name: string; args: Ast[] }
  | { t: 'un'; op: '-' | '!'; a: Ast }
  | { t: 'bin'; op: string; a: Ast; b: Ast };

export type Scope = (name: string) => Val | undefined;

export const BUILTIN_NAMES = [
  'transcript', 'reply', 'ocr', 'translated', 'barged', 'endpointed', 'sttMs', 'tok1Ms', 'iter',
] as const;
export const WRITABLE_BUILTINS = ['transcript', 'reply'] as const;
export const IF_PRESETS = ['barged', 'endpointed', 'len(ocr) > 0'] as const;
export const isBuiltin = (name: string): boolean => (BUILTIN_NAMES as readonly string[]).indexOf(name) >= 0;
export const isWritableBuiltin = (name: string): boolean => (WRITABLE_BUILTINS as readonly string[]).indexOf(name) >= 0;
export const VAR_RE = /^[A-Za-z_][A-Za-z0-9_]*$/;

export const typeName = (v: Val): string =>
  typeof v === 'number' ? 'number' : typeof v === 'string' ? 'string' : 'boolean';
export const str = (v: Val): string =>
  typeof v === 'boolean' ? (v ? 'true' : 'false') : String(v);

interface Tok { t: 'num' | 'str' | 'id' | 'op'; v: string; n: number; pos: number }
const OPS = ['||', '&&', '==', '!=', '<=', '>=', '<', '>', '+', '-', '*', '/', '%', '(', ')', ',', '!'];
const FUNCS: Record<string, number> = { len: 1, contains: 2, lower: 1, num: 1 };
export const FUNC_SIGS: ReadonlyArray<[string, string]> = [
  ['len', 'len(text) -> number'], ['contains', 'contains(text, part) -> true/false'],
  ['lower', 'lower(text) -> text'], ['num', 'num(text) -> number'],
];

interface PErr { message: string; pos: number }
const perr = (message: string, pos: number): never => { throw { message, pos } as PErr; };

function lex(src: string): Tok[] {
  const toks: Tok[] = [];
  let i = 0;
  while (i < src.length) {
    const ch = src[i] as string;
    if (ch === ' ' || ch === '\t') { i++; continue; }
    if (ch >= '0' && ch <= '9') {
      const m = /^[0-9]+(\.[0-9]+)?/.exec(src.slice(i)) as RegExpExecArray;
      toks.push({ t: 'num', v: m[0], n: parseFloat(m[0]), pos: i });
      i += m[0].length;
      continue;
    }
    if (ch === "'" || ch === '"') {
      const start = i;
      let out = '';
      i++;
      for (;;) {
        if (i >= src.length) perr('unterminated string', start);
        const c = src[i] as string;
        if (c === ch) { i++; break; }
        if (c === '\\') {
          const e = src[i + 1];
          if (e === 'n') out += '\n';
          else if (e === '\\' || e === "'" || e === '"') out += e;
          else perr("bad escape '\\" + (e ?? '') + "'", i);
          i += 2;
        } else { out += c; i++; }
      }
      toks.push({ t: 'str', v: out, n: 0, pos: start });
      continue;
    }
    const idm = /^[A-Za-z_][A-Za-z0-9_]*/.exec(src.slice(i));
    if (idm) {
      toks.push({ t: 'id', v: idm[0], n: 0, pos: i });
      i += idm[0].length;
      continue;
    }
    const op = OPS.find(o => src.startsWith(o, i));
    if (op) { toks.push({ t: 'op', v: op, n: 0, pos: i }); i += op.length; continue; }
    perr("unexpected character '" + ch + "'", i);
  }
  return toks;
}

export function parseExpr(src: string): { ast: Ast } | { error: string; pos: number } {
  try {
    if (!src.trim()) perr('empty expression', 0);
    const toks = lex(src);
    let p = 0;
    const peek = (): Tok | undefined => toks[p];
    const at = (v: string): boolean => { const t = toks[p]; return !!t && t.t === 'op' && t.v === v; };
    const eat = (v: string): void => {
      if (!at(v)) { const t = toks[p]; perr("expected '" + v + "'", t ? t.pos : src.length); }
      p++;
    };
    const prim = (): Ast => {
      const t = toks[p];
      if (!t) return perr('expression ended early', src.length);
      if (t.t === 'num') { p++; return { t: 'lit', v: t.n }; }
      if (t.t === 'str') { p++; return { t: 'lit', v: t.v }; }
      if (t.t === 'id') {
        p++;
        if (t.v === 'true') return { t: 'lit', v: true };
        if (t.v === 'false') return { t: 'lit', v: false };
        if (at('(')) {
          const arity = FUNCS[t.v];
          if (arity == null) perr("unknown function '" + t.v + "'", t.pos);
          eat('(');
          const args: Ast[] = [or()];
          while (at(',')) { p++; args.push(or()); }
          eat(')');
          if (args.length !== arity) perr(t.v + '() takes ' + arity + ' argument' + (arity === 1 ? '' : 's'), t.pos);
          return { t: 'call', name: t.v, args };
        }
        return { t: 'var', name: t.v };
      }
      if (at('(')) { p++; const e = or(); eat(')'); return e; }
      return perr("unexpected '" + t.v + "'", t.pos);
    };
    const unary = (): Ast => {
      if (at('-')) { p++; return { t: 'un', op: '-', a: unary() }; }
      if (at('!')) { p++; return { t: 'un', op: '!', a: unary() }; }
      return prim();
    };
    const mul = (): Ast => {
      let a = unary();
      while (at('*') || at('/') || at('%')) { const op = (peek() as Tok).v; p++; a = { t: 'bin', op, a, b: unary() }; }
      return a;
    };
    const add = (): Ast => {
      let a = mul();
      while (at('+') || at('-')) { const op = (peek() as Tok).v; p++; a = { t: 'bin', op, a, b: mul() }; }
      return a;
    };
    const cmp = (): Ast => {
      const a = add();
      for (const op of ['==', '!=', '<=', '>=', '<', '>']) {
        if (at(op)) { p++; return { t: 'bin', op, a, b: add() }; }
      }
      return a;
    };
    const and = (): Ast => {
      let a = cmp();
      while (at('&&')) { p++; a = { t: 'bin', op: '&&', a, b: cmp() }; }
      return a;
    };
    const or = (): Ast => {
      let a = and();
      while (at('||')) { p++; a = { t: 'bin', op: '||', a, b: and() }; }
      return a;
    };
    const ast = or();
    const rest = toks[p];
    if (rest) perr("unexpected '" + rest.v + "'", rest.pos);
    return { ast };
  } catch (e) {
    const pe = e as PErr;
    return { error: pe.message || 'parse error', pos: pe.pos ?? 0 };
  }
}

const ee = (msg: string): never => { throw new ExprError(msg); };
const needBool = (v: Val, what: string): boolean =>
  typeof v === 'boolean' ? v : ee(what + ' needs true/false — got ' + typeName(v));
const needNum = (v: Val, what: string): number =>
  typeof v === 'number' ? v : ee(what + ' needs a number — got ' + typeName(v));
const needStr = (v: Val, what: string): string =>
  typeof v === 'string' ? v : ee(what + ' needs a string — got ' + typeName(v));

export function evalExpr(ast: Ast, scope: Scope): Val {
  switch (ast.t) {
    case 'lit': return ast.v;
    case 'var': {
      const v = scope(ast.name);
      return v === undefined ? ee("unknown variable '" + ast.name + "'") : v;
    }
    case 'call': {
      const a = evalExpr(ast.args[0] as Ast, scope);
      if (ast.name === 'len') return needStr(a, 'len()').length;
      if (ast.name === 'lower') return needStr(a, 'lower()').toLowerCase();
      if (ast.name === 'num') {
        const s = needStr(a, 'num()').trim();
        if (!/^-?[0-9]+(\.[0-9]+)?$/.test(s)) ee("num('" + s.slice(0, 24) + "') — not numeric");
        return parseFloat(s);
      }
      const b = evalExpr(ast.args[1] as Ast, scope);
      return needStr(a, 'contains()').includes(needStr(b, 'contains()'));
    }
    case 'un': {
      const v = evalExpr(ast.a, scope);
      return ast.op === '-' ? -needNum(v, "unary '-'") : !needBool(v, "'!'");
    }
    case 'bin': {
      if (ast.op === '&&' || ast.op === '||') {
        const a = needBool(evalExpr(ast.a, scope), "'" + ast.op + "'");
        if (ast.op === '&&' && !a) return false;
        if (ast.op === '||' && a) return true;
        return needBool(evalExpr(ast.b, scope), "'" + ast.op + "'");
      }
      const a = evalExpr(ast.a, scope), b = evalExpr(ast.b, scope);
      switch (ast.op) {
        case '+':
          if (typeof a === 'number' && typeof b === 'number') return a + b;
          if (typeof a === 'string' || typeof b === 'string') return str(a) + str(b);
          return ee("cannot add " + typeName(a) + ' and ' + typeName(b));
        case '-': case '*': case '/': case '%': {
          const x = needNum(a, "'" + ast.op + "'"), y = needNum(b, "'" + ast.op + "'");
          if ((ast.op === '/' || ast.op === '%') && y === 0) ee('division by zero');
          return ast.op === '-' ? x - y : ast.op === '*' ? x * y : ast.op === '/' ? x / y : x % y;
        }
        case '==': case '!=': {
          if (typeName(a) !== typeName(b)) ee('cannot compare ' + typeName(a) + ' with ' + typeName(b));
          return ast.op === '==' ? a === b : a !== b;
        }
        default: {
          if (typeName(a) !== typeName(b) || typeof a === 'boolean')
            ee("'" + ast.op + "' cannot compare " + typeName(a) + ' with ' + typeName(b));
          const c = (a as number | string) < (b as number | string);
          const d = a === b;
          return ast.op === '<' ? c : ast.op === '<=' ? c || d : ast.op === '>' ? !c && !d : !c;
        }
      }
    }
  }
}

export const TEMPLATE_RE = /\{([A-Za-z_][A-Za-z0-9_]*)\}/g;
export function templateStr(src: string, scope: Scope): string {
  return src.replace(TEMPLATE_RE, (_, name: string) => {
    const v = scope(name);
    if (v === undefined) ee("unknown variable '" + name + "' in template");
    return str(v as Val);
  });
}
export function templateIdents(src: string): string[] {
  const out: string[] = [];
  for (const m of src.matchAll(TEMPLATE_RE)) { const n = m[1] as string; if (out.indexOf(n) < 0) out.push(n); }
  return out;
}
