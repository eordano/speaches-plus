import { useEffect, useRef, useState } from 'react';
import { S, setState, useAppState, type AppState } from '../state/store';
import { tint, dlText, OFFLINE_MSG } from '../lib/util';
import { getMode, micState } from '../engine/session';
import * as P from '../engine/patch';
import { SIG, glyphSvg, tapOf, PORT_COLORS, type FlowResult } from '../engine/ports';
import { emitDsl, applyDsl, canonicalLine, beginDslBurst } from '../engine/dsl';
import { running, runPatch, abortRun } from '../engine/runner';
import ConfigPanel from './ConfigPanel';
import { Composer, MessageList } from './ChatView';
import type { CSSProperties } from 'react';

function PaletteCol({ dsl }: { dsl: boolean }) {
  return (
    <div className="palette" style={{ width: 190, flex: 'none', overflowY: 'auto', borderRight: '1px solid var(--divider)', paddingRight: 14, display: 'flex', flexDirection: 'column', gap: 14 }}>
      {P.PALETTE.map(([name, chip, kinds]) => (
        <div key={name} style={{ display: 'flex', flexDirection: 'column', gap: 5 }}>
          <div className="cap1" style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
            <span style={{ width: 9, height: 9, borderRadius: 3, background: chip, flex: 'none' }} />
            {name}
          </div>
          {kinds.map(k => {
            const c = P.COLORS[k];
            const sig = SIG[k];
            const dot = (b: NonNullable<typeof sig.needs>, card: 'stream' | 'one', x: number): string =>
              card === 'stream'
                ? `<path d="M${x},1 L${x + 4},5 L${x},9 L${x - 4},5 Z" fill="${PORT_COLORS[b]}"/>`
                : `<circle cx="${x}" cy="5" r="3.4" fill="${PORT_COLORS[b]}"/>`;
            let sg = '';
            if (sig.needs) sg += dot(sig.needs, 'one', 5);
            if (sig.needs && sig.gives) sg += '<path d="M11,5 h6 m-2.5,-2.5 l2.5,2.5 l-2.5,2.5" style="stroke:var(--muted)" stroke-width="1" fill="none"/>';
            if (sig.gives) sg += dot(sig.gives, sig.card, sig.needs ? 23 : 5);
            return (
              <div
                key={k}
                className="hl"
                style={{ padding: '7px 10px', borderRadius: 6, border: `1px solid ${c}`, background: tint(c, 0.10), font: '400 11px var(--ui)', color: 'var(--ink-body)', cursor: dsl ? 'pointer' : 'grab', touchAction: 'none', userSelect: 'none', WebkitUserSelect: 'none', display: 'flex', alignItems: 'center', gap: 6 }}
                onPointerDown={(e) => {
                  e.preventDefault();
                  if (dsl) { P.insertDslLine(canonicalLine(P.newItem(k))); return; }
                  P.startDrag(e, P.newItem(k), null);
                }}
              >
                <span style={{ flex: 1 }}>{P.labelOf(k)}</span>
                {sg && <span style={{ flex: 'none', lineHeight: 0 }} dangerouslySetInnerHTML={{ __html: `<svg width="30" height="10" style="display:block">${sg}</svg>` }} />}
              </div>
            );
          })}
        </div>
      ))}
      <div style={{ font: '400 10px var(--ui)', color: 'var(--muted)', lineHeight: 1.5, paddingBottom: 4 }}>
        {dsl ? 'click a pill to insert its line at the cursor' : 'click a pill to preview · drag to place'}
      </div>
    </div>
  );
}

function CanvasBlock({ bl, S, fl }: { bl: P.LayoutBlockBox; S: AppState; fl: FlowResult }) {
  const b = bl.b, c = P.COLORS[b.k], key = P.pk(bl.path);
  const selected = !!(S.selPath && P.pk(S.selPath) === key);
  const runningHere = S.runPath === key;
  const errored = S.runError === key;
  const typeErr = fl.errs.get(key);
  const port = fl.out.get(key);
  const tap = tapOf(b);
  const shape = bl.hat ? P.hatShape(bl.w) : b.children ? P.cShape(bl.w, bl.h - 62) : P.stmtShape(bl.w);
  const stroke = errored || typeErr ? 'var(--acc-danger)' : c;
  const sw = (selected || runningHere || errored) ? 2.5 : typeErr ? 2 : 1.4;
  const fill = tint(c, (selected || runningHere) ? 0.28 : 0.12);
  let svg = `<svg width="${bl.w}" height="${bl.h + 16}" style="position:absolute;left:0;top:0;pointer-events:none;overflow:visible">`;
  if (runningHere) svg += `<path d="${shape}" fill="none" stroke="${tint(c, 0.35)}" stroke-width="6"/>`;
  svg += `<path d="${shape}" style="fill:${fill};stroke:${stroke}" stroke-width="${sw}"${typeErr ? ' stroke-dasharray="6 4"' : ''}/>`;
  if (port) svg += glyphSvg(port, 28, bl.h, 4.5);
  if (b.k === 'on' && port) svg += glyphSvg(port, 28, bl.hat ? 4 : 0, 4.5);
  svg += '</svg>';
  const canTap = !!SIG[b.k].gives && !!port;
  const sub = P.SUBS[b.k];
  return (
    <div
      title={typeErr || undefined}
      style={{ position: 'absolute', left: bl.x, top: bl.y, width: bl.w, height: bl.h, cursor: 'grab', touchAction: 'none', userSelect: 'none', WebkitUserSelect: 'none' }}
      onPointerDown={(e) => { e.stopPropagation(); e.preventDefault(); P.startDrag(e, P.deep(b), bl.path); }}
    >
      <span dangerouslySetInnerHTML={{ __html: svg }} />
      <span style={{ position: 'relative', display: 'flex', alignItems: 'baseline', gap: 8, padding: bl.hat ? '19px 14px 0 18px' : '12px 14px 0 16px' }}>
        <span style={{ font: '600 11.5px var(--ui)', color: 'var(--ink)', whiteSpace: 'nowrap' }}>{P.labelOf(b.k)}</span>
        {sub && (
          <span className="ell" style={{ font: '400 9px var(--mono)', color: 'var(--muted)', marginLeft: 'auto', maxWidth: '55%' }}>
            {sub(b)}
          </span>
        )}
      </span>
      {canTap && port && (
        <span
          data-tapport="1"
          title={tap ? "publishes stream '" + tap + "' — drag to wire another subscriber" : 'drag out to publish this ' + port.base + ' stream to another stack'}
          style={{ position: 'absolute', right: -7, top: (bl.hat ? 22 : 16) - 7, width: 14, height: 14, cursor: 'crosshair', touchAction: 'none', zIndex: 2 }}
          onPointerDown={(e) => { e.stopPropagation(); e.preventDefault(); P.startWire(e, bl.path, port); }}
        >
          <span dangerouslySetInnerHTML={{ __html: `<svg width="14" height="14" style="display:block"><path d="M7,1.5 L12.5,7 L7,12.5 L1.5,7 Z" style="fill:${tap ? PORT_COLORS[port.base] : 'var(--well)'};stroke:${tap ? 'var(--ink)' : PORT_COLORS[port.base]}" stroke-width="1.3"/></svg>` }} />
        </span>
      )}
    </div>
  );
}

function WireOverlay({ lay }: { lay: P.Layout }) {
  const byKey = new Map(lay.blocks.map(bl => [P.pk(bl.path), bl] as const));
  const pubs = P.tapPublishers();
  const paths: string[] = [];
  for (const bl of lay.blocks) {
    if (bl.b.k !== 'on') continue;
    const name = tapOf(bl.b);
    const pub = name ? pubs.find(t => t.name === name) : null;
    const pubBox = pub ? byKey.get(P.pk(pub.path)) : null;
    if (!pub || !pubBox) continue;
    const x1 = pubBox.x + pubBox.w + 5, y1 = pubBox.y + (pubBox.hat ? 22 : 16);
    const x2 = bl.x + 28, y2 = bl.y + 2;
    const col = PORT_COLORS[pub.f.base];
    paths.push(`<path d="M${x1},${y1} C${x1 + 60},${y1} ${x2 - 60},${y2 - 30} ${x2},${y2}" fill="none" stroke="${col}" stroke-width="2.4" stroke-dasharray="7 5" opacity=".85"/>` +
      `<path d="M${x2},${y2 - 5} L${x2 + 5},${y2 + 1} L${x2 - 5},${y2 + 1} Z" fill="${col}"/>`);
  }
  const wd = P.wireDrag;
  if (wd && wd.started) {
    const src = byKey.get(P.pk(wd.srcPath));
    if (src) {
      const x1 = src.x + src.w + 5, y1 = src.y + (src.hat ? 22 : 16);
      paths.push(`<path d="M${x1},${y1} C${x1 + 60},${y1} ${wd.x - 60},${wd.y} ${wd.x},${wd.y}" fill="none" stroke="${PORT_COLORS[wd.f.base]}" stroke-width="2.4" stroke-dasharray="4 4"/>`);
    }
  }
  if (!paths.length) return null;
  return (
    <span
      style={{ position: 'absolute', left: 0, top: 0, pointerEvents: 'none', zIndex: 3 }}
      dangerouslySetInnerHTML={{ __html: `<svg width="1" height="1" style="overflow:visible;display:block">${paths.join('')}</svg>` }}
    />
  );
}

function ChipsRow({ S }: { S: AppState }) {
  const fiRef = useRef<HTMLInputElement>(null);
  const live = getMode() === 'live';
  const runnable = live && !!P.hatRoot();
  const chip = (label: string, color: string, on: () => void, title: string) => (
    <span
      key={label}
      className="hl chip"
      title={title}
      style={{ color, border: `1px solid ${color}`, userSelect: 'none' } as CSSProperties}
      onClick={() => { if (S.runNotice) setState({ runNotice: null }); on(); }}
    >{label}</span>
  );
  const mic = micState();
  return (
    <div className="chips-row" style={{ position: 'absolute', top: 0, right: 0, display: 'flex', gap: 6, zIndex: 6, alignItems: 'center' }}>
      <span
        className="hl chip pal-toggle"
        title={S.palOpen ? 'hide the block palette' : 'show the block palette'}
        style={{ color: 'var(--acc-lang)', border: '1px solid var(--acc-lang)', userSelect: 'none' } as CSSProperties}
        onClick={() => setState({ palOpen: !S.palOpen })}
      >☰ blocks</span>
      {(S.busy || mic === 'monitoring') && (
        <span data-run-busy="1" style={{ font: '400 11px var(--ui)', color: 'var(--muted)', fontStyle: 'italic', marginRight: 2 }}>
          {S.busy || 'mic armed — speak to interrupt'}
        </span>
      )}
      {S.runNotice && (
        <span data-run-notice="1" style={{ font: '400 11px var(--ui)', color: 'var(--muted)', fontStyle: 'italic', marginRight: 2 }}>
          {S.runNotice}
        </span>
      )}
      {running
        ? (
          <span
            key="stop"
            className="hl chip mic-btn"
            data-state={mic}
            title="stop the live session"
            style={{ color: 'var(--acc-danger)', border: '1px solid var(--acc-danger)', userSelect: 'none' } as CSSProperties}
            onClick={() => { if (S.runNotice) setState({ runNotice: null }); abortRun(); }}
          >■ stop</span>
        )
        : chip('▶ run', runnable ? 'var(--acc-lang)' : 'var(--muted)', () => { void runPatch(); },
            !live ? OFFLINE_MSG : runnable ? 'start the live session — each element runs the stack' : 'no hat-rooted stack — add a trigger block')}
      {P.canUndo() && chip('↶ undo', 'var(--acc-lang)', P.undoOnce, 'undo the last program change')}
      {chip(S.dslMode ? '▦ blocks' : '{} dsl', 'var(--acc-lang)', P.toggleDsl,
        S.dslMode ? 'back to the block canvas' : 'edit the program as text')}
      {chip('↺ reset', 'var(--acc-lang)', P.resetProg, "restore this intent's default program")}
      {chip('⇩ export', 'var(--acc-lang)', P.exportProg, 'download program JSON')}
      {S.dslMode && chip('⇩ .nur', 'var(--acc-lang)', () => dlText(emitDsl(), 'nur-patch.nur'), 'download the program as .nur text')}
      {chip('⇧ import', 'var(--acc-lang)', () => fiRef.current?.click(), 'load a program — .json or .nur')}
      <input
        ref={fiRef}
        type="file"
        accept="application/json,.json,.nur,text/plain"
        style={{ display: 'none' }}
        onChange={(e) => { P.importFile(e.target.files && e.target.files[0]); e.target.value = ''; }}
      />
    </div>
  );
}

function DslPane() {
  const taRef = useRef<HTMLTextAreaElement>(null);
  const timer = useRef<ReturnType<typeof setTimeout> | null>(null);
  const [text, setText] = useState(() => emitDsl());
  const [stat, setStat] = useState(() => ({
    ok: true,
    msg: '✓ current program · ' + P.countNodes(S.prog) + ' blocks · ' + S.stacks.length + (S.stacks.length === 1 ? ' stack' : ' stacks'),
  }));
  const apply = (t: string): void => {
    const r = applyDsl(t, 'burst');
    if (r.ok) {
      P.setDslValid(true);
      setStat({ ok: true, msg: '✓ applied · ' + (r.blocks ?? 0) + ' blocks · ' + (r.stacks ?? 0) + ((r.stacks ?? 0) === 1 ? ' stack' : ' stacks') });
    } else {
      P.setDslValid(false);
      setStat({ ok: false, msg: '✗ line ' + (r.line ?? 0) + ': ' + (r.error ?? 'parse failed') + ' · unapplied' });
    }
  };
  const onChange = (t: string): void => {
    setText(t);
    if (timer.current) clearTimeout(timer.current);
    timer.current = setTimeout(() => apply(t), 300);
  };
  const onChangeRef = useRef(onChange);
  onChangeRef.current = onChange;
  useEffect(() => {
    beginDslBurst();
    P.setDslValid(true);
    P.setDslInsert((line) => {
      const ta = taRef.current;
      if (!ta) return;
      const pos = ta.selectionStart ?? ta.value.length;
      const before = ta.value.slice(0, pos), after = ta.value.slice(pos);
      const ins = (before && !before.endsWith('\n') ? '\n' : '') + line + '\n';
      onChangeRef.current(before + ins + after);
      requestAnimationFrame(() => {
        ta.focus();
        const p = pos + ins.length;
        ta.setSelectionRange(p, p);
      });
    });
    return () => {
      P.setDslInsert(null);
      if (timer.current) clearTimeout(timer.current);
    };
  }, []);
  return (
    <div className="dsl-pane" style={{ flex: 1, minHeight: 0, display: 'flex', flexDirection: 'column', gap: 6, paddingTop: 30 }}>
      <textarea
        ref={taRef}
        data-dsl="1"
        data-keepfocus="dsl"
        className="inp"
        spellCheck={false}
        value={text}
        onChange={(e) => onChange(e.target.value)}
        style={{ flex: 1, width: '100%', minHeight: 0, font: '400 10.5px/1.55 var(--mono)', padding: 10, resize: 'none' }}
      />
      <div data-dsl-status="1" style={{ font: '400 9.5px var(--mono)', color: stat.ok ? 'var(--acc-speech)' : 'var(--acc-danger)' }}>
        {stat.msg}
      </div>
    </div>
  );
}

function CanvasArea({ S }: { S: AppState }) {
  const lay = P.layout();
  const fl = P.flow();
  const drag = P.drag, snap = P.snap;
  const dragBox = drag && drag.started
    ? { w: P.widthOf(drag.item), h: P.measure(drag.item) + 16, shape: P.shapeFor(drag.item) }
    : null;
  return (
    <div style={{ flex: 1, minWidth: 0, minHeight: 0, position: 'relative', display: 'flex', flexDirection: 'column' }}>
      <ChipsRow S={S} />
      {S.dslMode ? <DslPane /> : (
        <div style={{ flex: 1, overflow: 'auto', minWidth: 0, minHeight: 0 }}>
          <div
            data-ws="1"
            style={{ position: 'relative', minWidth: 420, height: lay.wsH }}
            onPointerDown={(e) => { if (e.target === e.currentTarget) setState({ selNode: null, selPath: null }); }}
          >
            {lay.blocks.map(bl => <CanvasBlock key={P.pk(bl.path)} bl={bl} S={S} fl={fl} />)}
            <WireOverlay lay={lay} />
            {!S.prog.length && !S.stacks.length && !dragBox && (
              <div style={{ position: 'absolute', left: 24, top: 24, font: '400 11px var(--ui)', color: 'var(--muted)', lineHeight: 1.5, maxWidth: 300 }}>
                Drag blocks in from the palette. A stack rooted at a trigger hat (mic, keys, screen, cam) is the runnable program. Or press {'{}'} dsl to type the program as text.
              </div>
            )}
            {drag && dragBox && snap && (
              <span
                style={{ position: 'absolute', left: snap.x, top: snap.y - 4, pointerEvents: 'none', zIndex: 4 }}
                dangerouslySetInnerHTML={{ __html: `<svg width="${dragBox.w}" height="${dragBox.h}" style="overflow:visible;display:block"><path d="${dragBox.shape}" fill="none" stroke="var(--acc-lang)" stroke-width="2" stroke-dasharray="5 4"/></svg>` }}
              />
            )}
            {drag && dragBox && (
              <div style={{ position: 'absolute', left: drag.x - 50, top: drag.y - 14, width: dragBox.w, opacity: 0.85, pointerEvents: 'none', zIndex: 5 }}>
                <span dangerouslySetInnerHTML={{ __html: `<svg width="${dragBox.w}" height="${dragBox.h}" style="position:absolute;left:0;top:0;overflow:visible"><path d="${dragBox.shape}" style="fill:${tint(P.COLORS[drag.item.k], 0.20)};stroke:${drag.refuse ? 'var(--acc-danger)' : P.COLORS[drag.item.k]}" stroke-width="1.6"${drag.refuse ? ' stroke-dasharray="6 4"' : ''}/></svg>` }} />
                <span style={{ position: 'relative', display: 'block', padding: '14px 12px 0 16px', font: '600 12px var(--ui)', color: 'var(--ink)', whiteSpace: 'nowrap' }}>{P.labelOf(drag.item.k)}</span>
                {drag.refuse && (
                  <span data-refuse="1" style={{ position: 'absolute', left: 0, top: dragBox.h + 2, font: '400 10px var(--ui)', color: 'var(--acc-danger)', background: 'var(--panel)', border: '1px solid var(--acc-danger)', borderRadius: 4, padding: '3px 7px', whiteSpace: 'nowrap' }}>
                    {drag.refuse}
                  </span>
                )}
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
}

function ConvoRail({ S }: { S: AppState }) {
  return (
    <div className="convo-rail" style={{ width: 330, flex: 'none', display: 'flex', flexDirection: 'column', minHeight: 0 }}>
      <div className="card" style={{ padding: '14px 16px 0', flex: 1, minHeight: 0, display: 'flex', flexDirection: 'column', gap: 8 }}>
        <div className="cap1">CONVERSATION</div>
        {S.msgs.length
          ? <MessageList S={S} style={{ padding: '4px 0 10px', gap: 14 }} />
          : <div style={{ flex: 1, font: '400 11px var(--ui)', color: 'var(--muted)', fontStyle: 'italic', paddingTop: 4 }}>run the patch — replies land here</div>}
        <Composer S={S} />
      </div>
    </div>
  );
}

export default function PatchView() {
  const St = useAppState();
  useEffect(() => {
    const onKey = (e: KeyboardEvent): void => {
      const t = e.target as HTMLElement | null;
      const inField = !!t && (t.tagName === 'INPUT' || t.tagName === 'TEXTAREA');
      if ((e.metaKey || e.ctrlKey) && !e.shiftKey && (e.key === 'z' || e.key === 'Z')) {
        if (inField) return;
        if (P.canUndo()) { e.preventDefault(); P.undoOnce(); }
        return;
      }
      if (inField) return;
      if (e.key === 'Delete' || e.key === 'Backspace') {
        if (S.selPath) { e.preventDefault(); P.removeSel('key'); }
      } else if (e.key === 'Escape') {
        if (S.dslMode) P.toggleDsl();
        else if (S.selNode) setState({ selNode: null, selPath: null });
      }
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, []);
  return (
    <div style={{ flex: 1, display: 'flex', gap: 18, paddingTop: 24, minWidth: 0, minHeight: 0 }}>
      <div className={St.palOpen ? 'card patch-card pal-open' : 'card patch-card'} style={{ flex: 1, padding: 18, minWidth: 0, overflow: 'hidden', display: 'flex', gap: 16 }}>
        <PaletteCol dsl={St.dslMode} />
        <CanvasArea S={St} />
      </div>
      {St.selNode && P.TITLES[St.selNode] ? <ConfigPanel /> : <ConvoRail S={St} />}
    </div>
  );
}
