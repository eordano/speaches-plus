import { useEffect, useRef, useState } from 'react';
import type { CSSProperties, ReactNode } from 'react';
import { S, setState, scheduleRender, useAppState, type Block, type Val } from '../state/store';
import { bus } from '../state/bus';
import { D, ttsKind, voicesForModel } from '../data';
import { clamp, dlText, errMsg, icon, pad, playIcon, selTone, tryPause, tryPlay, OFFLINE_MSG } from '../lib/util';
import { nurClientFor } from '../api/client';
import {
  getMode, chatModel, sttModel, ttsModel, ttsVoice, tokRateLabel, perf, nurStore, appendLog,
  effModels, ensureBaseModels, normBase, displayModels, type ModelRow, type SvcKind,
} from '../engine/session';
import * as P from '../engine/patch';
import { gv, sv } from '../engine/patch';
import { lastRun } from '../engine/runner';
import { parseExpr, templateIdents, isBuiltin, isWritableBuiltin, VAR_RE, IF_PRESETS, str } from '../engine/expr';
import { storage } from '../state/storage';
import { SIG, TAP_RE, tapOf, PORT_COLORS } from '../engine/ports';

const capLabel = (t: string) => <div className="cap">{t}</div>;
const mutedLine = (t: string) => (
  <div style={{ font: '400 11px var(--ui)', color: 'var(--muted)', fontStyle: 'italic', padding: '2px 2px' }}>{t}</div>
);
const Group = ({ cap, children }: { cap: string; children: ReactNode }) => (
  <div className="grp">{capLabel(cap)}{children}</div>
);
const rowCls = (on: boolean): string => on ? 'orow' : 'orow hl';

interface FieldStatus { ok: boolean; msg: string }
const statusLine = (st: FieldStatus) => (
  <div data-field-status="1" style={{ font: '400 9.5px var(--mono)', lineHeight: 1.5, color: st.ok ? 'var(--acc-speech)' : 'var(--acc-danger)' }}>
    {(st.ok ? '✓ ' : '✗ ') + st.msg}
  </div>
);

function OptGroup({ label, rows, selIdx, pick }: {
  label: string; rows: Array<[string, string?, string?]>; selIdx: number; pick: (i: number) => void;
}) {
  return (
    <Group cap={label}>
      {rows.map((r, i) => {
        const on = i === selIdx;
        return (
          <div key={r[0]} className={rowCls(on)} style={{ padding: '9px 11px', ...selTone(on) }} onClick={() => pick(i)}>
            <div style={{ flex: 1, display: 'flex', flexDirection: 'column', gap: 2, minWidth: 0 }}>
              <span style={{ font: '600 12.5px var(--ui)' }}>{r[0]}</span>
              {r[2] != null && <span style={{ font: '400 9.5px var(--ui)', opacity: 0.75, lineHeight: 1.4 }}>{r[2]}</span>}
            </div>
            {r[1] != null && <span style={{ font: '400 10px var(--mono)', flex: 'none' }}>{r[1]}</span>}
          </div>
        );
      })}
    </Group>
  );
}

function ToggleRow({ label, on, set }: { label: string; on: boolean; set: () => void }) {
  return (
    <div style={{ display: 'flex', alignItems: 'center', gap: 10, cursor: 'pointer' }} onClick={set}>
      <span style={{ position: 'relative', width: 34, height: 20, borderRadius: 10, background: on ? 'var(--acc-speech)' : 'var(--border)', transition: 'background .11s', flex: 'none', display: 'inline-block' }}>
        <span style={{ position: 'absolute', top: 2, left: on ? 16 : 2, width: 16, height: 16, borderRadius: '50%', background: 'var(--panel)', transition: 'left .11s' }} />
      </span>
      <span style={{ flex: 1, font: '400 12.5px var(--ui)', color: 'var(--ink-body)' }}>{label}</span>
    </div>
  );
}

function StepperRow({ label, valText, down, up }: { label: string; valText: string; down: () => void; up: () => void }) {
  const btnStyle: CSSProperties = { width: 26, height: 26, border: '1px solid var(--border)', borderRadius: 4, background: 'var(--bg)', display: 'flex', alignItems: 'center', justifyContent: 'center', cursor: 'pointer', font: '400 14px var(--ui)', flex: 'none' };
  return (
    <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
      <span style={{ flex: 1, font: '400 12.5px var(--ui)' }}>{label}</span>
      <span className="hl" style={btnStyle} onClick={down}>−</span>
      <span style={{ font: '700 15px var(--mono)', minWidth: 22, textAlign: 'center' }}>{valText}</span>
      <span className="hl" style={btnStyle} onClick={up}>+</span>
    </div>
  );
}

const step = (k: P.NumericConfigKey, d: number, lo: number, hi: number, fx?: number) => () =>
  sv(k, Number(clamp(gv(k) + d, lo, hi).toFixed(fx || 0)));
const stepRow = (label: string, k: P.NumericConfigKey, d: number, lo: number, hi: number, fx?: number) => (
  <StepperRow
    key={label}
    label={label}
    valText={fx ? gv(k).toFixed(fx) : String(gv(k))}
    down={step(k, -d, lo, hi, fx)}
    up={step(k, d, lo, hi, fx)}
  />
);
const stepRowB = (label: string, b: Block, key: string, def: number, d: number, lo: number, hi: number) => (
  <StepperRow
    key={label}
    label={label}
    valText={String(P.bcNum(b, key, def))}
    down={() => P.svb(key, clamp(P.bcNum(b, key, def) - d, lo, hi))}
    up={() => P.svb(key, clamp(P.bcNum(b, key, def) + d, lo, hi))}
  />
);
const factRow = (k: string, v: string, tone?: string) => (
  <div key={k} style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'baseline', gap: 10, font: '400 11px var(--ui)' }}>
    <span style={{ color: 'var(--muted)', flex: 'none' }}>{k}</span>
    <span style={{ color: tone || 'var(--ink)', fontFamily: 'var(--mono)', fontSize: 10.5, textAlign: 'right' }}>{v}</span>
  </div>
);

function ModelGroup({ label, list, selId, pick, rightFn, emptyMsg, kind }: {
  label: string; list: string[]; selId: string | null; pick: (id: string) => void;
  rightFn?: (id: string) => string; emptyMsg?: string; kind?: SvcKind;
}) {
  const [showAll, setShowAll] = useState(false);
  const collapsed = kind && !showAll ? displayModels(kind, list, selId) : null;
  const rows: ModelRow[] = collapsed ? collapsed.rows : list.map(id => ({ id, note: null }));
  const nHidden = collapsed ? collapsed.total - collapsed.rows.length : 0;
  return (
    <Group cap={label}>
      {!rows.length ? mutedLine(emptyMsg || OFFLINE_MSG) : rows.map(({ id, note }) => {
        const on = id === selId;
        return (
          <div key={id} className={rowCls(on)} style={{ padding: '8px 11px', ...selTone(on) }} onClick={() => pick(id)}>
            <span className="ell" style={{ flex: 1, font: '600 11.5px var(--ui)', minWidth: 0 }} title={id}>{id}</span>
            {rightFn
              ? <span style={{ font: '400 10px var(--mono)', flex: 'none' }}>{rightFn(id)}</span>
              : note && <span style={{ font: '400 9.5px var(--mono)', color: on ? undefined : 'var(--muted)', flex: 'none' }}>{note}</span>}
          </div>
        );
      })}
      {kind && (nHidden > 0 || showAll) && (
        <div
          className="hl"
          data-modelfilter="1"
          style={{ font: '400 10px var(--mono)', color: 'var(--muted)', padding: '5px 11px 2px', cursor: 'pointer', userSelect: 'none' }}
          onClick={() => setShowAll(v => !v)}
        >
          {showAll ? '▴ collapse aliases' : '▾ ' + nHidden + ' alias id' + (nHidden === 1 ? '' : 's') + ' hidden — show all'}
        </div>
      )}
    </Group>
  );
}

const AC_KIND_COLOR: Record<P.Completion['kind'], string> = {
  var: 'var(--acc-lang)', builtin: 'var(--acc-turn)', fn: 'var(--acc-speech)', lit: 'var(--muted)',
};

function CodeField({ cap, value, set, placeholder, keepfocus, rows, status, complete }: {
  cap: string; value: string; set: (v: string) => void; placeholder: string;
  keepfocus: string; rows?: number; status?: FieldStatus | null;
  complete?: () => P.Completion[];
}) {
  const elRef = useRef<HTMLInputElement | HTMLTextAreaElement | null>(null);
  const [ac, setAc] = useState<{ items: P.Completion[]; idx: number; prefix: string } | null>(null);
  const prefixAt = (): string => {
    const el = elRef.current;
    const pos = el && el.selectionStart != null ? el.selectionStart : value.length;
    const m = /[A-Za-z_][A-Za-z0-9_]*$/.exec(value.slice(0, pos));
    return m ? m[0] : '';
  };
  const filtered = (items: P.Completion[], prefix: string): P.Completion[] =>
    items.filter(it => !prefix || it.name.toLowerCase().startsWith(prefix.toLowerCase()));
  const accept = (it: P.Completion): void => {
    const el = elRef.current;
    const pos = el && el.selectionStart != null ? el.selectionStart : value.length;
    const prefix = ac ? ac.prefix : '';
    const before = value.slice(0, pos - prefix.length);
    const ins = it.insert ?? it.name;
    const next = before + ins + value.slice(pos);
    const caret = before.length + ins.length + (it.caret ?? 0);
    setAc(null);
    set(next);
    requestAnimationFrame(() => {
      const e2 = elRef.current;
      if (e2) { e2.focus(); e2.setSelectionRange(caret, caret); }
    });
  };
  const visible = ac ? filtered(ac.items, ac.prefix) : [];
  const common = {
    'data-keepfocus': keepfocus,
    placeholder,
    className: 'inp',
    spellCheck: false,
    value,
    onChange: (e: { target: { value: string } }) => {
      set(e.target.value);
      if (ac) requestAnimationFrame(() => setAc(a => (a ? { ...a, prefix: prefixAt(), idx: 0 } : a)));
    },
    onKeyDown: (e: React.KeyboardEvent) => {
      if (complete && (e.ctrlKey || e.metaKey) && (e.key === ' ' || e.code === 'Space')) {
        e.preventDefault();
        setAc({ items: complete(), idx: 0, prefix: prefixAt() });
        return;
      }
      if (!ac) return;
      if (e.key === 'ArrowDown' || e.key === 'ArrowUp') {
        e.preventDefault();
        const n = visible.length;
        if (n) setAc({ ...ac, idx: (ac.idx + (e.key === 'ArrowDown' ? 1 : n - 1)) % n });
      } else if (e.key === 'Enter' || e.key === 'Tab') {
        const it = visible[Math.min(ac.idx, visible.length - 1)];
        if (it) { e.preventDefault(); accept(it); } else setAc(null);
      } else if (e.key === 'Escape') {
        e.preventDefault();
        e.stopPropagation();
        setAc(null);
      }
    },
    onBlur: () => setTimeout(() => setAc(null), 150),
    style: {
      width: '100%', font: '400 10.5px/1.55 var(--mono)', padding: rows ? 10 : '8px 10px',
      resize: rows ? 'vertical' : undefined,
    } as CSSProperties,
  };
  const refCb = (el: HTMLInputElement | HTMLTextAreaElement | null): void => { elRef.current = el; };
  return (
    <Group cap={cap}>
      <div style={{ position: 'relative' }}>
        {rows ? <textarea rows={rows} ref={refCb} {...common} /> : <input ref={refCb} {...common} />}
        {ac && (
          <div
            data-autocomplete="1"
            style={{ position: 'absolute', left: 0, right: 0, top: '100%', zIndex: 20, background: 'var(--panel)', border: '1px solid var(--border)', borderRadius: 6, marginTop: 2, maxHeight: 180, overflowY: 'auto', boxShadow: '0 4px 14px rgba(43,43,43,.12)' }}
          >
            {!visible.length
              ? <div style={{ font: '400 10px var(--mono)', color: 'var(--muted)', padding: '6px 9px' }}>nothing in scope matches '{ac.prefix}'</div>
              : visible.map((it, i) => (
                <div
                  key={it.kind + it.name}
                  style={{ display: 'flex', alignItems: 'baseline', gap: 8, padding: '5px 9px', cursor: 'pointer', background: i === ac.idx ? 'var(--hover)' : undefined }}
                  onPointerDown={(e) => { e.preventDefault(); accept(it); }}
                  onPointerEnter={() => setAc(a => (a ? { ...a, idx: i } : a))}
                >
                  <span style={{ font: '600 10.5px var(--mono)', color: 'var(--ink)' }}>{it.name}</span>
                  <span style={{ font: '400 9px var(--mono)', color: AC_KIND_COLOR[it.kind], marginLeft: 'auto' }}>{it.detail}</span>
                </div>
              ))}
          </div>
        )}
      </div>
      {status && statusLine(status)}
    </Group>
  );
}

function EndpointGroup({ k, kind }: { k: 'baseChat' | 'baseStt' | 'baseTts'; kind: SvcKind }) {
  const v = S[k] || '';
  const bad = !!v.trim() && !/^https:\/\//.test(v.trim());
  return (
    <Group cap={'ENDPOINT — ' + kind + ' requests'}>
      <input
        data-keepfocus={k}
        className="inp"
        spellCheck={false}
        placeholder="same origin — /v1"
        value={v}
        onChange={(e) => sv(k, e.target.value)}
        onBlur={() => { const b = normBase(S[k] || ''); if (b && /^https:\/\//.test(b)) void ensureBaseModels(b); }}
        onKeyDown={(e) => { if (e.key === 'Enter') (e.target as HTMLInputElement).blur(); }}
        style={{ width: '100%', font: '400 10.5px/1.55 var(--mono)', padding: '8px 10px' }}
      />
      {bad && statusLine({ ok: false, msg: 'must be https:// — the browser blocks http targets from an https page' })}
      {factRow('effective', normBase(v) || '/v1')}
      {mutedLine('endpoint must allow browser (CORS) requests · no auth is sent')}
    </Group>
  );
}

function dupSel(): void {
  const L = P.getList(S.selPath);
  const b = L && L.arr[L.idx];
  if (!L || !b) return;
  P.pushUndo();
  L.arr.splice(L.idx + 1, 0, P.deep(b));
  bus.emit('patch.block.duplicate', { node: S.selNode });
  P.savePatch();
  scheduleRender();
}
function forkSel(): void {
  const L = P.getList(S.selPath);
  if (!L || !L.arr[L.idx]) return;
  const tail = L.arr.slice(L.idx).map(x => P.deep(x));
  const first = tail[0];
  if (!first) return;
  P.pushUndo();
  const bl = P.layout().blocks.find(x => S.selPath && P.pk(x.path) === P.pk(S.selPath));
  S.stacks.push({ x: (bl ? bl.x : 24) + 248 + 44, y: bl ? bl.y : 16, items: tail });
  bus.emit('patch.block.fork', { node: first.k, count: tail.length });
  setState({ selNode: first.k, selPath: ['S', S.stacks.length - 1, 0] });
  P.savePatch();
}
function BlockActions() {
  const act = (label: string, d: string, color: string, on: () => void, title?: string) => (
    <span
      key={label}
      className="hl"
      title={title}
      style={{ flex: 1, display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 5, border: '1px solid var(--border)', borderRadius: 6, padding: '6px 0', background: 'var(--bg)', cursor: 'pointer', font: '400 11px var(--ui)', color }}
      onClick={on}
    >
      <span style={{ display: 'flex' }} dangerouslySetInnerHTML={{ __html: icon(d, 11, color, 1.25) }} />
      {label}
    </span>
  );
  return (
    <div style={{ display: 'flex', gap: 6 }}>
      {act('Duplicate', D.dup, 'var(--muted)', dupSel)}
      {act('Copy tail', D.fork, 'var(--muted)', forkSel,
        'copy the tail into a side stack — side stacks are authoring scratch and do not run')}
      {act('Remove', D.trash, 'var(--acc-danger)', () => P.removeSel())}
    </div>
  );
}

function TextField({ cap, k, keepfocus, rows, placeholder, style, value }: {
  cap: string; k: 'sysPrompt' | 'vdText' | 'voice'; keepfocus: string;
  rows?: number; placeholder: string; style: CSSProperties; value: string;
}) {
  const common = {
    'data-keepfocus': keepfocus,
    placeholder,
    className: 'inp',
    style,
    value,
    onChange: (e: { target: { value: string } }) => { setState({ [k]: e.target.value }); P.savePatch(); },
  };
  return (
    <Group cap={cap}>
      {rows ? <textarea rows={rows} {...common} /> : <input {...common} />}
    </Group>
  );
}

const sysPromptSection = () => (
  <TextField
    key="sysprompt"
    cap="SYSTEM PROMPT — sent with every agent request" k="sysPrompt" keepfocus="sysprompt"
    rows={3} placeholder="optional — leave empty for none"
    style={{ fontSize: 12, lineHeight: 1.5, padding: 10, resize: 'vertical', width: '100%' }}
    value={S.sysPrompt}
  />
);

const prevCache: Record<string, string> = {};
let prevEl: HTMLAudioElement | null = null;
function stopPreview(): void {
  tryPause(prevEl);
  prevEl = null;
  if (S.vPrev != null) setState({ vPrev: null });
}
async function playVoice(model: string, voice: string, i: number): Promise<void> {
  if (S.vPrev === i) { stopPreview(); return; }
  stopPreview();
  setState({ vPrev: i });
  const key = model + ':' + voice;
  try {
    if (!prevCache[key]) {
      const r = await nurClientFor(S.baseTts).audio.speech.create({
        model, voice, input: 'This is the ' + voice.replace(/_/g, ' ') + ' voice.', response_format: 'wav', speed: 1,
      });
      prevCache[key] = URL.createObjectURL(await r.blob());
    }
    if (S.vPrev !== i) return;
    const url = prevCache[key];
    if (!url) return;
    prevEl = new Audio(url);
    prevEl.onended = stopPreview;
    tryPlay(prevEl);
    bus.emit('tts.preview', { model, voice });
  } catch (e) {
    appendLog(pad('tts.error') + 'preview failed · ' + errMsg(e).slice(0, 50));
    stopPreview();
  }
}
function VoiceSection({ model, list }: { model: string; list: string[] }) {
  const cur = ttsVoice();
  return (
    <Group cap="VOICE">
      {list.map((v, i) => {
        const on = v === cur;
        const playing = S.vPrev === i;
        return (
          <div key={v} className={rowCls(on)} style={{ padding: '9px 11px', ...selTone(on) }} onClick={() => sv('voice', v)}>
            <span
              title="preview — synthesizes a short phrase"
              style={{ width: 24, height: 24, flex: 'none', borderRadius: '50%', display: 'flex', alignItems: 'center', justifyContent: 'center', cursor: 'pointer', border: `1px solid ${on ? 'var(--muted)' : 'var(--border)'}`, background: on ? 'var(--ink-body)' : 'var(--bg)', color: on ? 'var(--bg)' : 'var(--ink-body)' }}
              dangerouslySetInnerHTML={{ __html: playIcon(playing, 9, 'currentColor') }}
              onClick={(e) => { e.stopPropagation(); void playVoice(model, v, i); }}
            />
            <span className="ell" style={{ flex: 1, font: '600 12.5px var(--ui)', minWidth: 0 }}>{v}</span>
          </div>
        );
      })}
    </Group>
  );
}
const vdSection = () => (
  <TextField
    key="voicedesign"
    cap="VOICE INSTRUCTIONS — sent as the voice field" k="vdText" keepfocus="voicedesign"
    rows={3} placeholder="describe the voice, e.g. a low, unhurried voice with a slight rasp; reads numbers carefully"
    style={{ fontSize: 12.5, lineHeight: 1.5, padding: 10, resize: 'vertical', width: '100%' }}
    value={S.vdText}
  />
);
const voiceIdInput = () => (
  <TextField
    key="voicefree"
    cap="VOICE ID — sent as the voice field (optional)" k="voice" keepfocus="voicefree"
    placeholder="leave empty for the model default"
    style={{ fontSize: 12, padding: '8px 10px', width: '100%' }}
    value={S.voice || ''}
  />
);

function StoreSection() {
  const items = nurStore.read();
  return (
    <div className="grp">
      <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
        {capLabel('STORED ARTIFACTS · ' + items.length)}
        {items.length > 0 && (
          <span
            className="hl chip"
            style={{ color: 'var(--acc-danger)', border: '1px solid var(--acc-danger)', marginLeft: 'auto' }}
            onClick={() => { nurStore.clear(); scheduleRender(); }}
          >clear</span>
        )}
      </div>
      {!items.length ? mutedLine('nothing stored yet — run a program that reaches this block') :
        items.slice(-6).reverse().map(it => {
          const d = new Date(it.t);
          return (
            <div
              key={it.t + it.kind}
              className="hl"
              title="download this artifact as .txt"
              style={{ display: 'flex', flexDirection: 'column', gap: 2, border: '1px solid var(--border)', borderRadius: 4, background: 'var(--bg)', padding: '6px 9px', cursor: 'pointer' }}
              onClick={() => dlText(it.text || '',
                'nur-' + it.kind + '-' + d.toISOString().slice(11, 19).replace(/:/g, '') + '.txt')}
            >
              <span style={{ font: '600 10px var(--mono)', color: 'var(--muted)' }}>{it.kind + ' · ' + d.toLocaleTimeString()}</span>
              <span className="ell" style={{ font: '400 10.5px var(--ui)', color: 'var(--ink-body)' }}>{(it.text || '').slice(0, 80)}</span>
            </div>
          );
        })}
    </div>
  );
}

function knownVars(): string[] {
  const out: string[] = [];
  const walk = (items: Block[]): void => items.forEach(b => {
    if (b.k === 'set' || b.k === 'append') {
      const n = P.bcStr(b, 'var');
      if (n && out.indexOf(n) < 0) out.push(n);
    }
    if (b.k === 'http') {
      const n = P.bcStr(b, 'into', 'http');
      if (out.indexOf(n) < 0) out.push(n, n + 'Status');
    }
    if (b.children) walk(b.children);
  });
  walk(S.prog);
  S.stacks.forEach(st => walk(st.items));
  return out;
}
const showVal = (v: Val): string => {
  const s = str(v);
  return s.length > 44 ? s.slice(0, 41) + '…' : s;
};
const exprStatus = (src: string): FieldStatus => {
  const p = parseExpr(src);
  return 'error' in p ? { ok: false, msg: p.error + ' · at ' + p.pos } : { ok: true, msg: 'ok' };
};
const tmplStatus = (src: string): FieldStatus | null => {
  const kv = knownVars();
  const bad = templateIdents(src).find(n => !isBuiltin(n) && kv.indexOf(n) < 0);
  return bad ? { ok: false, msg: "unknown variable '{" + bad + "}' — no set/append/http defines it" } : null;
};
const dropHint = (key: string) => (
  <div key={key}>{mutedLine('drag the block into the canvas, then click it there to edit its config')}</div>
);

export default function ConfigPanel() {
  const St = useAppState();
  useEffect(() => {
    (['baseChat', 'baseStt', 'baseTts'] as const).forEach(bk => {
      const v = normBase(S[bk] || '');
      if (v && /^https:\/\//.test(v)) void ensureBaseModels(v);
    });
  }, [St.baseChat, St.baseStt, St.baseTts, St.selNode]);
  useEffect(() => stopPreview, [St.selNode, St.selPath]);
  const k = St.selNode;
  if (!k || !P.TITLES[k]) return null;
  const groups: ReactNode[] = [], toggles: ReactNode[] = [], steppers: ReactNode[] = [], facts: Array<[string, string, string?]> = [];
  const extras: ReactNode[] = [];
  const live = getMode() === 'live';
  const sel = St.selPath ? P.selBlock() : null;
  const modelList = (kind: SvcKind): { list: string[]; emptyMsg: string } => {
    const eff = effModels(kind);
    if (eff.base) return { list: eff.list, emptyMsg: eff.error || 'fetching model list from the endpoint…' };
    return { list: live ? eff.list : [], emptyMsg: OFFLINE_MSG };
  };

  if (k === 'mic') {
    facts.push(['capture', 'getUserMedia · MediaRecorder'], ['endpointing', 'energy VAD (eou silence window)']);
  } else if (k === 'keys') {
    facts.push(['source', 'conversation composer']);
  } else if (k === 'screen') {
    steppers.push(sel ? stepRowB('Frame rate fps', sel, 'fps', gv('fps'), 1, 1, 5) : stepRow('Frame rate fps', 'fps', 1, 1, 5));
    facts.push(['capture', 'getDisplayMedia frame'], ['fps applies to', 'for-each-frame loops']);
  } else if (k === 'cam') {
    facts.push(['capture', 'getUserMedia video frame']);
  } else if (k === 'on') {
    if (sel) {
      const name = tapOf(sel);
      const pubs = P.tapPublishers();
      const pub = pubs.find(t => t.name === name);
      const bad = !name ? 'name the stream this stack subscribes to'
        : !TAP_RE.test(name) ? 'letters, digits, - or _ only'
        : !pub ? "no block publishes '" + name + "' — drag from a block's side port, or set a tap name on one" : null;
      extras.push(
        <CodeField key="tap" cap="STREAM — subscribes this stack · ⌃space completes" value={name}
          set={(v) => P.svb('tap', v.trim())} keepfocus="btap" placeholder="s1"
          complete={() => P.tapPublishers().map(tp => ({ name: tp.name, kind: 'var' as const, detail: tp.f.base + ' stream' }))}
          status={bad ? { ok: false, msg: bad } : { ok: true, msg: '✓ fed by ' + (pub ? pub.f.base + ' stream' : '') }} />,
      );
      if (pubs.length) facts.push(['published streams', pubs.map(t => t.name + ' (' + t.f.base + ')').join(' · ')]);
    } else extras.push(dropHint('hint'));
    facts.push(['runs', 'once per element of the publishing stream'],
      ['placement', 'must start its own stack']);
  } else if (k === 'eou') {
    steppers.push(stepRow('Silence window ms', 'eouMs', 40, 80, 1200));
    toggles.push(<ToggleRow key="barge" label="Barge-in (mic cancels playback)" on={St.barge} set={() => sv('barge', !St.barge)} />);
    facts.push(['method', 'client-side energy VAD']);
  } else if (k === 'stt') {
    const m = modelList('stt');
    groups.push(<ModelGroup key="m.stt" label="MODEL" kind="stt" list={m.list} emptyMsg={m.emptyMsg} selId={sttModel()} pick={(id) => sv('selStt', id)} />);
    groups.push(<OptGroup key="o" label="OUTPUT" rows={[
      ['json', undefined, 'plain transcript text — fastest, right for most patches'],
      ['verbose_json', 'segments', 'adds timed segments and word timestamps'],
      ['diarized', 'speakers', 'labels who said what — for meeting-style transcripts'],
    ]} selIdx={St.sttFormat} pick={(i) => sv('sttFormat', i)} />);
    extras.push(<EndpointGroup key="ep" k="baseStt" kind="stt" />);
    facts.push(['route', St.sttFormat === 2 ? '/v1/audio/transcriptions · diarized_json' : '/v1/audio/transcriptions'],
      ['last stt (measured)', perf.stt != null ? perf.stt + ' ms' : '—']);
  } else if (k === 'translate') {
    facts.push(['route', 'chat completion · fixed prompt'],
      ['model', live ? (chatModel() || '—') : '—'],
      ['target', 'english']);
  } else if (k === 'agent') {
    const m = modelList('chat');
    groups.push(<ModelGroup key="m.chat" label="MODEL" kind="chat" list={m.list} emptyMsg={m.emptyMsg} selId={chatModel()}
      pick={(id) => sv('selChat', id)} rightFn={(id) => tokRateLabel(id)} />);
    toggles.push(<ToggleRow key="think" label="Show thinking in the conversation" on={St.showThinking} set={() => sv('showThinking', !St.showThinking)} />);
    steppers.push(stepRow('Temperature', 'temp', 0.1, 0, 1.5, 1));
    steppers.push(stepRow('top_p', 'topP', 0.05, 0.5, 1, 2));
    extras.push(sysPromptSection());
    extras.push(<EndpointGroup key="ep" k="baseChat" kind="chat" />);
    facts.push(['ms/tok (measured)', live ? tokRateLabel(chatModel()) : '—']);
  } else if (k === 'ocr') {
    facts.push(['route', 'POST /v1/ocr · multipart frame'], ['input', 'captured screen/camera frame']);
  } else if (k === 'tts') {
    const m = modelList('tts');
    groups.push(<ModelGroup key="m.tts" label="MODEL" kind="tts" list={m.list} emptyMsg={m.emptyMsg} selId={ttsModel()}
      pick={(id) => { stopPreview(); setState({ selTts: id, voice: null }); P.savePatch(); }} />);
    const id = ttsModel(), kind = ttsKind(id);
    if (id && m.list.length) {
      if (kind === 'design') extras.push(vdSection());
      else {
        const vl = voicesForModel(id);
        if (vl) extras.push(<VoiceSection key="voices" model={id} list={vl} />);
        else extras.push(voiceIdInput());
      }
    }
    steppers.push(stepRow('Speed', 'speed', 0.25, 0.5, 2, 2));
    extras.push(<EndpointGroup key="ep" k="baseTts" kind="tts" />);
    facts.push(['route', 'POST /v1/audio/speech'],
      ['synthesis', 'sentence by sentence — playback starts at the first one']);
  } else if (k === 'spk') {
    facts.push(['output', 'browser audio element'],
      ['barge-in', St.barge ? 'on — mic energy cancels playback' : 'off']);
  } else if (k === 'store') {
    extras.push(<StoreSection key="store" />);
    facts.push(['sink', storage.kind === 'opfs' ? 'opfs · artifacts.json' : 'localStorage · nur.store.v1'],
      ['run variables', 'written as one vars artifact when any exist']);
  } else if (k === 'cancel') {
    facts.push(['action', 'pauses the playing audio element']);
  } else if (k === 'if') {
    if (sel) {
      const expr = P.ifExpr(sel);
      extras.push(
        <CodeField key="if.expr" cap="CONDITION — must be true/false · ⌃space completes" value={expr}
          set={(v) => P.svb('expr', v)} keepfocus="bexpr"
          complete={() => P.identsInScope(St.selPath)}
          placeholder="e.g. barged or len(ocr) > 0" status={exprStatus(expr)} />,
        <div key="presets" style={{ display: 'flex', gap: 6, flexWrap: 'wrap' }}>
          {IF_PRESETS.map(p => (
            <span key={p} className="hl chip" title="fill the condition"
              style={{ color: 'var(--acc-lang)', border: '1px solid var(--acc-lang)' }}
              onClick={() => P.svb('expr', p)}>{p}</span>
          ))}
        </div>,
      );
      const lo = St.selPath ? lastRun.branch[P.pk(St.selPath)] : undefined;
      facts.push(['last outcome', lo === undefined ? '—' : String(lo)]);
    } else extras.push(dropHint('hint'));
    facts.push(['runs its body when', 'the condition is true']);
  } else if (k === 'set' || k === 'append') {
    if (sel) {
      const name = P.bcStr(sel, 'var');
      const nameBad = !VAR_RE.test(name) ? 'not a usable variable name'
        : (k === 'append' ? isBuiltin(name) : (isBuiltin(name) && !isWritableBuiltin(name)))
          ? "'" + name + "' is read-only" : null;
      extras.push(
        <CodeField key="var" cap="VARIABLE" value={name} set={(v) => P.svb('var', v)}
          keepfocus="bvar" placeholder="name"
          complete={() => P.identsInScope(St.selPath).filter(cc =>
            cc.kind === 'var' || (k === 'set' && isWritableBuiltin(cc.name)))}
          status={nameBad ? { ok: false, msg: nameBad } : null} />,
        <CodeField key={k + '.expr'} cap={(k === 'set' ? 'EXPRESSION' : 'EXPRESSION — appended as a new line') + ' · ⌃space completes'}
          value={P.bcStr(sel, 'expr')} set={(v) => P.svb('expr', v)} keepfocus="bexpr"
          complete={() => P.identsInScope(St.selPath)}
          placeholder="e.g. len(transcript) or count + 1" status={exprStatus(P.bcStr(sel, 'expr'))} />,
      );
      if (k === 'set' && isWritableBuiltin(name)) facts.push(['writes through to', 'the run pipeline (' + name + ')']);
      const lv = S.lastVars && name in S.lastVars ? showVal(S.lastVars[name] as Val) : '—';
      facts.push(['last value', lv]);
    } else extras.push(dropHint('hint'));
    facts.push(['scope', 'one run — cleared when the run ends']);
  } else if (k === 'repeat') {
    if (sel) {
      steppers.push(stepRowB('Times', sel, 'n', 3, 1, 1, 20));
      const li = St.selPath ? lastRun.iters[P.pk(St.selPath)] : undefined;
      facts.push(['last iterations', li === undefined ? '—' : String(li)]);
    } else extras.push(dropHint('hint'));
    facts.push(['iter builtin', '1-based inside the loop · 0 outside'], ['abort', '■ stop ends it between passes']);
  } else if (k === 'wait') {
    if (sel) steppers.push(stepRowB('Milliseconds', sel, 'ms', 1000, 250, 0, 30000));
    else extras.push(dropHint('hint'));
    facts.push(['abort', '■ stop checked every 100 ms']);
  } else if (k === 'http') {
    if (sel) {
      const url = P.bcStr(sel, 'url');
      const method = P.bcStr(sel, 'method', 'GET') === 'POST' ? 1 : 0;
      const into = P.bcStr(sel, 'into', 'http');
      const urlStatus: FieldStatus | null = !url.trim()
        ? { ok: false, msg: 'url is required' }
        : /^http:\/\//.test(url.trim())
          ? { ok: false, msg: 'must be https:// — the browser blocks http targets from an https page' }
          : tmplStatus(url);
      extras.push(
        <CodeField key="url" cap="URL — {var} substitutes run variables · ⌃space completes" value={url}
          set={(v) => P.svb('url', v)} keepfocus="burl"
          complete={() => P.identsInScope(St.selPath).filter(cc => cc.kind === 'var' || cc.kind === 'builtin')}
          placeholder="https://ntfy.sh/your-topic" status={urlStatus} />,
      );
      groups.push(<OptGroup key="m" label="METHOD" rows={[['GET'], ['POST', 'sends body']]}
        selIdx={method} pick={(i) => P.svb('method', i ? 'POST' : 'GET')} />);
      if (method === 1) {
        extras.push(
          <CodeField key="body" cap="BODY — {var} substitutes run variables · ⌃space completes" rows={3}
            value={P.bcStr(sel, 'body')} set={(v) => P.svb('body', v)} keepfocus="bbody"
            complete={() => P.identsInScope(St.selPath).filter(cc => cc.kind === 'var' || cc.kind === 'builtin')}
            placeholder="{transcript}" status={tmplStatus(P.bcStr(sel, 'body'))} />,
        );
      }
      extras.push(
        <CodeField key="hdr" cap="HEADERS — Name: value per line" rows={2}
          value={P.bcStr(sel, 'headers')} set={(v) => P.svb('headers', v)} keepfocus="bhdr"
          placeholder="Content-Type: application/json" status={null} />,
        <div key="hdrnote">{mutedLine('saved with the program · included in exports')}</div>,
        <CodeField key="into" cap="RESPONSE → VARIABLE" value={into}
          set={(v) => P.svb('into', v)} keepfocus="binto" placeholder="http"
          status={!VAR_RE.test(into) || isBuiltin(into) ? { ok: false, msg: 'not a usable variable name' } : null} />,
      );
      if (method === 1) facts.push(['content-type', 'text/plain unless set — stays a CORS simple request']);
      const ls = S.lastVars && (into + 'Status') in S.lastVars ? String(S.lastVars[into + 'Status']) : '—';
      facts.push(['last status', ls]);
      facts.push(['non-2xx', 'not an error — status lands in ' + into + 'Status']);
    } else extras.push(dropHint('hint'));
    facts.push(['timeout', '15 s · ■ stop aborts']);
  } else if (k === 'for') {
    if (sel) {
      steppers.push(stepRowB('Frames', sel, 'frames', gv('frames'), 1, 1, 10));
      steppers.push(stepRowB('Frame rate fps', sel, 'fps', gv('fps'), 1, 1, 5));
      facts.push(['frame interval', Math.round(1000 / clamp(P.bcNum(sel, 'fps', gv('fps')), 1, 5)) + ' ms']);
    } else {
      steppers.push(stepRow('Frames', 'frames', 1, 1, 10));
      steppers.push(stepRow('Frame rate fps', 'fps', 1, 1, 5));
      facts.push(['frame interval', Math.round(1000 / clamp(gv('fps'), 1, 5)) + ' ms']);
    }
  }

  if (sel && k !== 'on' && SIG[k].gives && St.selPath && P.flowOut(St.selPath)) {
    const name = tapOf(sel);
    extras.push(
      <CodeField key="pubtap" cap="PUBLISH AS STREAM — other stacks subscribe with 'when stream arrives'"
        value={name} set={(v) => P.svb('tap', v.trim())} keepfocus="bpubtap" placeholder="(not published)"
        status={name && !TAP_RE.test(name) ? { ok: false, msg: 'letters, digits, - or _ only' } : null} />,
    );
    const f = P.flowOut(St.selPath);
    if (f) facts.push(['port', f.card + ' of ' + f.base, PORT_COLORS[f.base]]);
  }

  if (St.selPath && P.flowErr(St.selPath))
    facts.push(['type error', String(P.flowErr(St.selPath)), 'var(--acc-danger)']);

  if (St.selPath && S.runError === P.pk(St.selPath) && S.runErrorMsg)
    facts.push(['last error', S.runErrorMsg.length > 60 ? S.runErrorMsg.slice(0, 57) + '…' : S.runErrorMsg, 'var(--acc-danger)']);

  return (
    <div className="config-panel" style={{ width: 330, flex: 'none', display: 'flex', flexDirection: 'column', gap: 16 }}>
      <div className="card" style={{ padding: 16, display: 'flex', flexDirection: 'column', gap: 12 }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
          <span style={{ width: 9, height: 9, borderRadius: '50%', background: P.COLORS[k], flex: 'none' }} />
          <span style={{ font: '600 14px var(--serif)', color: 'var(--ink)', flex: 1 }}>{P.TITLES[k]}</span>
          <span
            className="hm"
            style={{ font: '400 13px var(--ui)', color: 'var(--muted)', cursor: 'pointer', padding: '0 4px' }}
            onClick={() => setState({ selNode: null, selPath: null })}
          >×</span>
        </div>
        {St.selPath && <BlockActions />}
        {groups}
        {extras}
        {toggles}
        {steppers}
        {facts.map(f => factRow(f[0], f[1], f[2]))}
      </div>
    </div>
  );
}
