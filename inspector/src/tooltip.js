import { laneColor } from './lanes.js';
import { formatMs, escapeHTML } from './util.js';

const STAT_ROWS = [
  { key: 'no_speech_prob',    label: 'no_speech',    thrKey: 'no_speech_prob_threshold', altThrKey: 'threshold' },
  { key: 'logprob',           label: 'logprob',      thrKey: 'avg_logprob_threshold', effThrKey: 'effective_avg_logprob_threshold', durKey: 'audio_duration_ms' },
  { key: 'compression_ratio', label: 'compression' },
];

export function setupTooltip({ store }) {
  const tt = document.getElementById('tooltip');

  function show(e, mx, my) {
    const palette = store.state.tweaks.palette;
    const col = laneColor(palette, e.lane, e.kind);
    const p = e.payload || {};
    const rows = [];

    if (p.text)  rows.push(['text',  String(p.text)]);
    if (p.delta) rows.push(['delta', String(p.delta)]);

    // Cumulative LLM text up to this chunk/done/cancelled.
    if (e.lane === 'llm' && (e.kind === 'chunk' || e.kind === 'done' || e.kind === 'cancelled')) {
      const rid = e.corr?.response_id;
      let cumm = '';
      for (const x of store.state.events) {
        if (x.lane !== 'llm' || x.kind !== 'chunk') continue;
        if (rid && x.corr?.response_id !== rid) continue;
        if (x.seq > e.seq) break;
        if (x.payload?.delta) cumm += x.payload.delta;
      }
      if (cumm) rows.push([e.kind === 'chunk' ? 'cumm' : 'text', cumm]);
    }

    // All phrases for a completed/cancelled response.
    if (e.lane === 'response' && (e.kind === 'done' || e.kind === 'cancelled')) {
      const rid = e.corr?.response_id;
      const phrases = [];
      for (const x of store.state.events) {
        if (x.lane !== 'tts_req' || x.kind !== 'phrase_sent') continue;
        if (rid && x.corr?.response_id !== rid) continue;
        if (x.seq > e.seq) break;
        const t = x.payload?.text;
        if (t) phrases.push(t);
      }
      if (phrases.length) rows.push(['phrases', phrases.map((t, i) => `${i + 1}. ${t}`).join('\n')]);
    }

    if (e.lane === 'stt' && e.kind === 'backfill') {
      rows.push(['backfill', p.text || '']);
      rows.push(['item', p.item_id || '']);
    }
    if (e.lane === 'turn' && e.kind === 'bargein_context') {
      if (p.heard)   rows.push(['heard',   String(p.heard)]);
      if (p.unheard) rows.push(['unheard', String(p.unheard)]);
    }
    if (e.lane === 'bargein') {
      if (p.delay_ms != null) rows.push(['delay', p.delay_ms + 'ms']);
      if (p.reason)           rows.push(['reason', p.reason]);
    }
    if (p.model)            rows.push(['model', p.model]);
    if (p.bytes != null)    rows.push(['bytes', p.bytes]);
    if (p.event_type)       rows.push(['event', p.event_type]);
    if (p.elapsed_ms != null) rows.push(['elapsed', p.elapsed_ms + 'ms']);
    if (p.ttft_ms != null)  rows.push(['ttft', p.ttft_ms + 'ms']);
    if (p.tok_out != null)  rows.push(['tok_out', p.tok_out]);
    if (p.prob != null)     rows.push(['prob', p.prob]);
    if (p.rms != null)      rows.push(['rms', p.rms]);
    if (p.ms_audio != null) rows.push(['ms_audio', p.ms_audio]);
    if (p.reason)           rows.push(['reason', p.reason]);
    if (p.error)            rows.push(['error', String(p.error)]);

    const fmt = v => (typeof v === 'number' ? (Math.abs(v) < 1e-3 ? v.toExponential(2) : v.toFixed(3)) : v);
    for (const spec of STAT_ROWS) {
      const v = p['avg_' + spec.key];
      if (v == null) continue;
      const parts = [`avg ${fmt(v)}`];
      const mn = p['min_' + spec.key], mx = p['max_' + spec.key];
      if (mn != null) parts.push(`min ${fmt(mn)}`);
      if (mx != null) parts.push(`max ${fmt(mx)}`);
      const eff = spec.effThrKey ? p[spec.effThrKey] : null;
      if (eff != null) {
        const base = p[spec.thrKey];
        const dur  = spec.durKey ? p[spec.durKey] : null;
        const baseStr = (base != null && base !== eff) ? ` (base ${base}${dur != null ? ` @ ${dur}ms` : ''})` : '';
        parts.push(`thr ${fmt(eff)}${baseStr}`);
      } else {
        const thr = spec.thrKey ? p[spec.thrKey] : null;
        const altThr = spec.altThrKey ? p[spec.altThrKey] : null;
        if (thr != null) parts.push(`thr ${thr}`);
        else if (altThr != null) parts.push(`thr ${altThr}`);
      }
      rows.push([spec.label, parts.join(' · ')]);
    }

    const c = e.corr || {};
    if (c.phrase_id)        rows.push(['phrase',   store.aliasFor('phrase', c.phrase_id)]);
    else if (c.response_id) rows.push(['response', store.aliasFor('response', c.response_id)]);
    else if (c.item_id)     rows.push(['item',     store.aliasFor('item', c.item_id)]);
    else if (c.turn_id)     rows.push(['turn',     store.aliasFor('turn', c.turn_id)]);

    tt.innerHTML = `
      <div class="t-head">
        <span class="t-lane" style="background:${col}40;color:${col}">${e.lane}</span>
        <span>${e.kind}</span>
        <span class="t-dim" style="margin-left:auto">${formatMs(e.t)}</span>
      </div>
      ${rows.map(([k, v]) => `<div class="t-row"><span class="t-dim">${k}</span><span>${escapeHTML(v)}</span></div>`).join('')}
      <div class="t-row" style="margin-top:6px"><span class="t-dim">seq</span><span>${e.seq}</span></div>`;
    move(mx, my);
    tt.style.display = 'block';
  }

  function move(mx, my) {
    const r = tt.getBoundingClientRect();
    let x = mx + 14, y = my + 14;
    if (x + r.width  > window.innerWidth  - 8) x = mx - r.width  - 14;
    if (y + r.height > window.innerHeight - 8) y = my - r.height - 14;
    tt.style.left = x + 'px';
    tt.style.top  = y + 'px';
  }

  function hide() { tt.style.display = 'none'; }

  return { show, move, hide };
}
