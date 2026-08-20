import type { CSSProperties, ReactNode } from 'react';

const codeStyle: CSSProperties = {
  font: '400 12px var(--mono)', background: 'var(--well)', border: '1px solid var(--divider)',
  borderRadius: 3, padding: '0 4px',
};
const preStyle: CSSProperties = {
  font: '400 11px/1.55 var(--mono)', background: 'var(--bg)', border: '1px solid var(--border)',
  borderRadius: 6, padding: '8px 10px', margin: '6px 0', overflowX: 'auto', whiteSpace: 'pre',
};
const linkStyle: CSSProperties = { color: 'var(--acc-turn)', textDecoration: 'underline' };

const INLINE_RE =
  /(`[^`\n]+`)|(\*\*[^*\n]+\*\*)|(__[^_\n]+__)|(\*[^*\s][^*\n]*\*)|(_[^_\s][^_\n]*_)|(\[[^\]\n]+\]\(https?:\/\/[^\s)]+\))/;

function inline(text: string, keyBase: string): ReactNode[] {
  const out: ReactNode[] = [];
  let rest = text, n = 0;
  for (;;) {
    const m = INLINE_RE.exec(rest);
    if (!m) { if (rest) out.push(rest); return out; }
    if (m.index > 0) out.push(rest.slice(0, m.index));
    const tok = m[0];
    const key = keyBase + '.' + n++;
    if (m[1]) out.push(<code key={key} style={codeStyle}>{tok.slice(1, -1)}</code>);
    else if (m[2] || m[3]) out.push(<strong key={key}>{inline(tok.slice(2, -2), key)}</strong>);
    else if (m[4] || m[5]) out.push(<em key={key}>{inline(tok.slice(1, -1), key)}</em>);
    else {
      const lm = /^\[([^\]]+)\]\((https?:\/\/[^\s)]+)\)$/.exec(tok);
      if (lm) out.push(<a key={key} href={lm[2]} target="_blank" rel="noreferrer" style={linkStyle}>{lm[1]}</a>);
      else out.push(tok);
    }
    rest = rest.slice(m.index + tok.length);
  }
}

const H_SIZE: Record<number, number> = { 1: 16, 2: 15, 3: 14, 4: 13.5 };

export function renderMd(text: string): ReactNode[] {
  const out: ReactNode[] = [];
  const lines = text.split('\n');
  let i = 0, key = 0;
  const para: string[] = [];
  const flushPara = (): void => {
    if (!para.length) return;
    out.push(
      <div key={'p' + key++} style={{ whiteSpace: 'pre-wrap', margin: out.length ? '6px 0 0' : 0 }}>
        {inline(para.join('\n'), 'p' + key)}
      </div>,
    );
    para.length = 0;
  };
  while (i < lines.length) {
    const line = lines[i] as string;
    const fence = /^\s*```/.exec(line);
    if (fence) {
      flushPara();
      const body: string[] = [];
      i++;
      while (i < lines.length && !/^\s*```/.test(lines[i] as string)) { body.push(lines[i] as string); i++; }
      i++;
      out.push(<pre key={'c' + key++} style={preStyle}>{body.join('\n')}</pre>);
      continue;
    }
    const h = /^(#{1,4})\s+(.*)$/.exec(line);
    if (h) {
      flushPara();
      const lvl = (h[1] as string).length;
      out.push(
        <div key={'h' + key++} style={{ font: `700 ${H_SIZE[lvl]}px var(--serif)`, margin: out.length ? '8px 0 2px' : '0 0 2px' }}>
          {inline(h[2] as string, 'h' + key)}
        </div>,
      );
      i++;
      continue;
    }
    const isBullet = (s: string): boolean => /^\s*[-*+]\s+/.test(s);
    const isOrdered = (s: string): boolean => /^\s*\d+[.)]\s+/.test(s);
    if (isBullet(line) || isOrdered(line)) {
      flushPara();
      const ordered = isOrdered(line);
      const items: string[] = [];
      while (i < lines.length && (ordered ? isOrdered(lines[i] as string) : isBullet(lines[i] as string))) {
        items.push((lines[i] as string).replace(ordered ? /^\s*\d+[.)]\s+/ : /^\s*[-*+]\s+/, ''));
        i++;
      }
      const li = items.map((it, j) => <li key={j} style={{ margin: '2px 0' }}>{inline(it, 'l' + key + '.' + j)}</li>);
      out.push(ordered
        ? <ol key={'o' + key++} style={{ margin: '4px 0', paddingLeft: 22 }}>{li}</ol>
        : <ul key={'u' + key++} style={{ margin: '4px 0', paddingLeft: 22 }}>{li}</ul>);
      continue;
    }
    if (/^\s*([-*_])\s*\1\s*\1[\s\-*_]*$/.test(line)) {
      flushPara();
      out.push(<div key={'r' + key++} style={{ borderTop: '1px solid var(--border)', margin: '8px 0' }} />);
      i++;
      continue;
    }
    if (!line.trim()) { flushPara(); i++; continue; }
    para.push(line);
    i++;
  }
  flushPara();
  return out;
}
