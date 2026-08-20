import { useRef, type ReactNode } from 'react';
import { setState, useAppState, type Turn } from '../state/store';
import { shortId } from '../data';
import { clamp, dlText, selTone, tint } from '../lib/util';
import { getMode, chatModel, sttModel, ttsModel, tokRateLabel } from '../engine/session';
import { REC, eventModel } from '../engine/inspector';

const laneTitle = (txt: string) => (
  <div style={{ font: '400 9.5px var(--ui)', color: 'var(--muted)', marginBottom: 4 }}>{txt}</div>
);

export default function InspectorView() {
  const S2 = useAppState();
  const tipRef = useRef<HTMLDivElement>(null);

  const hoverProps = (title: string, body: string) => ({
    onMouseEnter: (e: { currentTarget: Element }) => {
      const tip = tipRef.current;
      if (!tip) return;
      const t0 = tip.children[0] as HTMLElement | undefined;
      const t1 = tip.children[1] as HTMLElement | undefined;
      if (!t0 || !t1) return;
      t0.textContent = title;
      t1.textContent = body || '';
      t1.style.display = body ? 'block' : 'none';
      tip.style.display = 'block';
      tip.style.left = '0px'; tip.style.top = '0px';
      const r = e.currentTarget.getBoundingClientRect(), tr = tip.getBoundingClientRect();
      const x = Math.max(8, Math.min(r.left + r.width / 2 - tr.width / 2, window.innerWidth - tr.width - 8));
      let y = r.top - tr.height - 6;
      if (y < 4) y = r.bottom + 6;
      tip.style.left = x + 'px'; tip.style.top = y + 'px';
    },
    onMouseLeave: () => { const tip = tipRef.current; if (tip) tip.style.display = 'none'; },
  });

  if (!S2.turns.length) {
    return (
      <div style={{ flex: 1, minHeight: 420, display: 'flex', flexDirection: 'column', gap: 16, marginTop: 14, overflowY: 'auto', minWidth: 0 }}>
        <div className="card" style={{ padding: '70px 22px', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
          <span style={{ font: '400 14px var(--serif)', fontStyle: 'italic', color: 'var(--muted)' }}>run a session to populate the inspector</span>
        </div>
      </div>
    );
  }

  const selIdx = clamp(S2.sel, 0, S2.turns.length - 1);
  const t: Partial<Turn> = S2.turns[selIdx] || {};
  const rec = REC.recs[selIdx];
  let M;
  try { M = eventModel(t, rec && rec.events.length ? rec : null); }
  catch { M = eventModel(t, null); }

  const live = getMode() === 'live';
  const axisTicks: ReactNode[] = [];
  for (let i = 0; i <= 6; i++) {
    const rv = Math.round(M.dur / 6 * i / 100) / 10;
    axisTicks.push(<span key={i}>{(Number.isInteger(rv) ? rv : rv.toFixed(1)) + 's'}</span>);
  }

  const q = (S2.inspFilter || '').toLowerCase();
  const matched = S2.log.filter(line => !q || line.toLowerCase().indexOf(q) >= 0);

  return (
    <div style={{ flex: 1, minHeight: 420, display: 'flex', flexDirection: 'column', gap: 16, marginTop: 14, overflowY: 'auto', minWidth: 0 }}>
      <div ref={tipRef} style={{ position: 'fixed', display: 'none', zIndex: 30, pointerEvents: 'none', maxWidth: 280, background: 'var(--panel)', border: '1px solid var(--border)', borderRadius: 4, padding: '6px 8px' }}>
        <div style={{ font: '600 9.5px var(--mono)', color: 'var(--ink)', whiteSpace: 'nowrap' }}></div>
        <div style={{ font: '400 9.5px var(--mono)', color: 'var(--muted)', wordBreak: 'break-all', lineHeight: 1.5 }}></div>
      </div>

      <div className="insp-top" style={{ display: 'flex', gap: 16, alignItems: 'stretch', flex: 'none' }}>
        <div className="card" style={{ flex: 1, padding: '20px 22px', minWidth: 0 }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
            <div style={{ font: '700 18px var(--serif)', color: 'var(--ink)' }}>Inspector</div>
            <div style={{ marginLeft: 'auto', display: 'flex', gap: 6, font: '400 10.5px var(--ui)', flexWrap: 'wrap', justifyContent: 'flex-end' }}>
              {S2.turns.map((_x, i) => {
                const on = i === selIdx;
                return (
                  <span
                    key={i}
                    className={on ? undefined : 'hl'}
                    style={{ padding: '4px 10px', borderRadius: 8, cursor: 'pointer', ...selTone(on) }}
                    onClick={() => setState({ sel: i })}
                  >{'turn ' + (i + 1)}</span>
                );
              })}
            </div>
          </div>

          {!rec || !rec.events.length ? (
            <div style={{ font: '400 12px var(--ui)', color: 'var(--muted)', fontStyle: 'italic', padding: '26px 0' }}>
              no event timeline recorded for this turn (resumed session)
            </div>
          ) : (
            <div className="insp-lanes">
              <div className="insp-lanes-inner">
              <div style={{ display: 'flex', justifyContent: 'space-between', font: '400 9px var(--ui)', color: 'var(--muted)', marginTop: 14 }}>
                {axisTicks}
              </div>
              <div style={{ marginTop: 8, display: 'flex', flexDirection: 'column', gap: 10 }}>
                <div>
                  {laneTitle('activity (from event timestamps)')}
                  {M.LANES.map(name => {
                    const span = M.spans.find(s => s.label === name);
                    return (
                      <div key={name} style={{ display: 'flex', alignItems: 'center', gap: 8, marginTop: 3 }}>
                        <span style={{ width: 56, flex: 'none', font: '400 9px var(--ui)', color: 'var(--muted)', textAlign: 'right' }}>{name}</span>
                        <span style={{ flex: 1, position: 'relative', display: 'block' }}>
                          <div style={{ position: 'relative', height: 15, background: 'var(--bg)', border: '1px solid var(--divider)', borderRadius: 3 }}>
                            {span && (
                              <span
                                style={{ position: 'absolute', left: `${span.a}%`, width: `${Math.max(span.b - span.a, 0.6)}%`, top: 2, bottom: 2, borderRadius: 2, background: tint(span.c, 0.4), border: `1px solid ${span.c}` }}
                                {...hoverProps(name, Math.round((span.b - span.a) / 100 * M.dur) + ' ms')}
                              />
                            )}
                          </div>
                        </span>
                      </div>
                    );
                  })}
                </div>

                {M.partials.length > 0 && (
                  <div>
                    {laneTitle('stt result')}
                    <div style={{ position: 'relative', height: 22 }}>
                      {M.partials.map((p, i) => (
                        <span
                          key={i}
                          className="ell"
                          style={{ position: 'absolute', left: `${Math.min(p.x, 55)}%`, top: 0, padding: '2px 7px', border: '1px solid var(--acc-speech)', borderRadius: 4, background: 'var(--bg)', color: 'var(--acc-speech)', font: '400 9.5px var(--mono)', maxWidth: '44%' }}
                        >{p.label}</span>
                      ))}
                    </div>
                  </div>
                )}

                <div>
                  {laneTitle('events')}
                  <div style={{ position: 'relative', height: 30 }}>
                    {M.markers.slice(0, 48).map((mk, i) => (
                      <span
                        key={i}
                        style={{ position: 'absolute', left: `${Math.min(mk.x, 97)}%`, top: i % 2 ? 15 : 0, whiteSpace: 'nowrap', color: mk.c, font: '400 9px var(--ui)', cursor: 'default' }}
                        {...hoverProps(mk.tip[0], mk.tip[1])}
                      >{'▲' + (mk.label ? ' ' + mk.label : '')}</span>
                    ))}
                  </div>
                </div>
              </div>
              </div>
            </div>
          )}
        </div>

        <div className="insp-side" style={{ width: 264, flex: 'none', display: 'flex', flexDirection: 'column', gap: 12 }}>
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 10 }}>
            {([[M.m.eou, 'EoU silence', 'var(--acc-turn)'], [M.m.tok1, 'first token', 'var(--acc-lang)'],
               [M.m.audio1, 'tts synthesis', 'var(--acc-speech)'], [M.m.cancel, 'barge cancel', 'var(--acc-danger)']] as Array<[number | null, string, string]>).map(([v, label, c]) => (
              <div key={label} className="card" style={{ padding: 12 }}>
                <div style={{ font: '700 22px var(--serif)', color: c, whiteSpace: 'nowrap' }}>
                  {v == null ? '—' : String(v)}
                  {v != null && <span style={{ font: '400 11px var(--ui)', color: 'var(--muted)' }}> ms</span>}
                </div>
                <div style={{ font: '400 9.5px var(--ui)', color: 'var(--muted)' }}>{label}</div>
              </div>
            ))}
          </div>
          <div className="card" style={{ padding: 16, display: 'flex', flexDirection: 'column', gap: 8, flex: 1 }}>
            {([['chat model', live ? shortId(chatModel()) : '—'],
               ['ms/tok (measured)', live ? tokRateLabel(chatModel()) : '—'],
               ['sampler', 'T ' + S2.temp.toFixed(1) + ' · p ' + S2.topP.toFixed(2)],
               ['stt', (live ? shortId(sttModel()) : '—') + (t.stt ? ' · ' + t.stt + ' ms' : '')],
               ['tts', (live ? shortId(ttsModel()) : '—') + ' · ' + S2.speed.toFixed(2) + '×'],
               ['wall clock', M.wall != null ? M.wall + ' ms' : '—']] as Array<[string, string]>).map(([k, v]) => (
              <div key={k} style={{ display: 'flex', justifyContent: 'space-between', font: '400 11px var(--ui)' }}>
                <span style={{ color: 'var(--muted)' }}>{k}</span>
                <span style={{ color: 'var(--ink)', fontFamily: 'var(--mono)', fontSize: 10.5 }}>{v}</span>
              </div>
            ))}
          </div>
        </div>
      </div>

      <div className="well" style={{ padding: '16px 20px', flex: 1, display: 'flex', flexDirection: 'column', minHeight: 120 }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 8 }}>
          <div className="cap" style={{ flex: 1 }}>SESSION LOG</div>
          <input
            data-keepfocus="inspfilter"
            placeholder="filter…"
            value={S2.inspFilter || ''}
            style={{ width: 170, border: '1px solid var(--border)', borderRadius: 4, background: 'var(--bg)', color: 'var(--ink-body)', padding: '2px 7px', font: '400 9.5px var(--mono)', outlineColor: 'var(--acc-lang)' }}
            onChange={(e) => setState({ inspFilter: e.target.value })}
          />
          <span
            className="hl chip"
            title="download session log as .txt"
            style={{ color: 'var(--acc-lang)', border: '1px solid var(--acc-lang)', userSelect: 'none' }}
            onClick={() => dlText(S2.log.join('\n'), 'session-log.txt')}
          >⇩</span>
        </div>
        <div
          style={{ flex: 1, minHeight: 0, maxHeight: 320, overflowY: 'auto' }}
          ref={(el) => { if (el) el.scrollTop = el.scrollHeight; }}
        >
          {matched.slice(-14).map((line, i) => (
            <div key={i} style={{ font: '400 11px var(--mono)', color: 'var(--ink-body)', lineHeight: 1.8, whiteSpace: 'pre-wrap' }}>{line}</div>
          ))}
          {!matched.length && (
            <div style={{ font: '400 11px var(--ui)', color: 'var(--muted)', fontStyle: 'italic' }}>
              {q ? 'no log lines match' : 'no log lines yet'}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
