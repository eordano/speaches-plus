import { lookupLane, laneColor, ERROR_KINDS, PALETTES } from './lanes.js';
import { escapeHTML, syntaxHighlight, formatMs } from './util.js';

const $ = (s) => document.querySelector(s);

export function setupDetail({ store, timeline }) {
  const eyebrow  = $('#iEyebrow');
  const title    = $('#iTitle');
  const subline  = $('#iSubline');
  const body     = $('#inspBody');
  const tabs     = document.querySelectorAll('.insp-tab');
  const btnCopy  = $('#btnCopy');
  const navBtns  = document.querySelectorAll('.insp-foot .nav button');

  function corrRefs(e) {
    const c = e.corr || {};
    const out = [];
    if (c.turn_id)     out.push(['turn',     store.aliasFor('turn', c.turn_id)]);
    if (c.item_id)     out.push(['item',     store.aliasFor('item', c.item_id)]);
    if (c.response_id) out.push(['response', store.aliasFor('response', c.response_id)]);
    if (c.phrase_id)   out.push(['phrase',   store.aliasFor('phrase', c.phrase_id)]);
    if (e.lane === 'stt' && c.item_id) {
      const vadStart = store.state.events.find(x => x.lane === 'vad' && x.kind === 'confirmed_start' && x.corr?.item_id === c.item_id);
      const vadStop  = store.state.events.find(x => x.lane === 'vad' && x.kind === 'stopped'         && x.corr?.item_id === c.item_id);
      if (vadStart) {
        const dur = vadStop ? Math.round(vadStop.t - vadStart.t) : null;
        out.push(['vad', dur != null ? `speech · ${dur}ms` : 'speech']);
      }
    }
    return out;
  }

  function related(e) {
    const c = e.corr || {};
    if (!c.turn_id && !c.item_id && !c.response_id && !c.phrase_id) return [];
    return store.state.events.filter(x => {
      if (x.seq === e.seq) return false;
      const xc = x.corr || {};
      return (c.phrase_id && xc.phrase_id === c.phrase_id)
          || (c.response_id && xc.response_id === c.response_id)
          || (c.item_id && xc.item_id === c.item_id)
          || (c.turn_id && xc.turn_id === c.turn_id && !c.response_id && !c.item_id);
    });
  }

  function render() {
    const e = store.state.selected;
    if (!e) {
      eyebrow.textContent = 'Selected event';
      title.textContent = 'No event selected';
      subline.innerHTML = '<span class="tag">pick an event on the timeline</span>';
      body.classList.remove('raw'); body.classList.add('pretty');
      body.innerHTML = `
        <div class="insp-empty">Click a block on the timeline to inspect it.</div>
        <div class="insp-hint">Hover any event for a quick preview. Click-drag across lanes to measure a time range. Shift-click two events to diff.</div>`;
      const relTab = document.querySelector('.insp-tab[data-tab="related"]');
      relTab.innerHTML = `Related <span class="fg-faint--spaced">0</span>`;
      return;
    }
    const laneMeta = lookupLane(e.lane);
    eyebrow.textContent = `Seq ${e.seq} · ${formatMs(e.t)} from session start`;
    title.textContent = `${laneMeta.name} · ${e.kind}`;

    const isErr = e.lane === 'error' || ERROR_KINDS.has(e.kind);
    const tagColor = isErr
      ? PALETTES[store.state.tweaks.palette].error
      : laneColor(store.state.tweaks.palette, e.lane, e.kind);
    const tags = [
      `<span class="tag lane" style="--tag-color:${tagColor}">${e.lane}</span>`,
      `<span class="tag lane" style="--tag-color:${tagColor}">${e.kind}</span>`,
    ];
    for (const [k, v] of corrRefs(e)) {
      tags.push(`<span class="tag"><span style="color:var(--fg-dim);margin-right:4px">${k}</span>${v}</span>`);
    }
    if (e.span_id) tags.push(`<span class="tag">span ${e.span_id}</span>`);
    subline.innerHTML = tags.join('');

    body.classList.toggle('pretty', store.state.activeTab === 'pretty');
    body.classList.toggle('raw',    store.state.activeTab === 'raw');
    if (store.state.activeTab === 'pretty')      body.innerHTML = prettyHTML(e);
    else if (store.state.activeTab === 'raw')    body.innerHTML = '<pre>' + syntaxHighlight(JSON.stringify(stripT(e), null, 2)) + '</pre>';
    else                                          body.innerHTML = relatedHTML(e, related(e));

    const relTab = document.querySelector('.insp-tab[data-tab="related"]');
    relTab.innerHTML = `Related <span class="fg-faint--spaced">${related(e).length}</span>`;
  }

  function prettyHTML(e) {
    const meta = [
      ['lane',     e.lane],
      ['kind',     e.kind],
      ['seq',      e.seq],
      ['t (mono)', formatMs(e.t)],
      ['wall',     e.ts_wall ? new Date(e.ts_wall * 1000).toISOString().replace('T', ' ').replace('Z', '') : '--'],
      ['span_id',  e.span_id],
    ];
    const refs = corrRefs(e);
    const payload = Object.entries(e.payload || {});
    return `
      <div class="sec">
        <h4>Event</h4>
        ${meta.map(([k, v]) => row(k, v, k === 't (mono)' ? 'num' : null)).join('')}
      </div>
      <div class="sec">
        <h4>Correlation</h4>
        ${refs.length
          ? refs.map(([k, v]) => corrRow(e, k, v)).join('')
          : `<div class="row"><span class="k">--</span><span class="v nul">no references</span></div>`}
      </div>
      <div class="sec">
        <h4>Payload</h4>
        ${payload.length
          ? payload.map(([k, v]) => k === 'messages' && Array.isArray(v) ? messagesRow(k, v) : row(k, v, typeofClass(v))).join('')
          : `<div class="row"><span class="k">--</span><span class="v nul">no payload</span></div>`}
      </div>
      <div class="sec">
        <h4>Cross-reference</h4>
        ${row('session.id', e.session_id)}
        ${row('OTEL', e.span_id ? 'open in Tempo ↗' : '--', 'str')}
      </div>`;
  }

  function corrRow(e, k, label) {
    const c = e.corr || {};
    const rawId = k === 'vad' ? c.item_id : c[k + '_id'];
    const target = rawId ? firstEventForCorr(k, rawId) : null;
    if (!target) return row(k, label, 'str');
    return `<div class="row" data-seq="${target.seq}" style="cursor:pointer" title="Jump to ${escapeHTML(String(label))}"><span class="k">${k}</span><span class="v str">${escapeHTML(String(label))}</span></div>`;
  }

  function firstEventForCorr(kind, rawId) {
    if (kind === 'vad') {
      return store.state.events.find(x => x.lane === 'vad' && x.kind === 'confirmed_start' && x.corr?.item_id === rawId);
    }
    const field = kind + '_id';
    return store.state.events.find(x => x.corr && x.corr[field] === rawId);
  }

  function messagesRow(k, msgs) {
    const lines = msgs.map(m => {
      const role = m.role || '?';
      const text = sanitizeContentForDisplay(m.content);
      return `<div><span class="v str">${escapeHTML(role)}</span>: ${escapeHTML(text)}</div>`;
    }).join('');
    return `<div class="row"><span class="k">${k}</span><span class="v" style="white-space:pre-wrap;word-break:break-word">${lines}</span></div>`;
  }
  function sanitizeContentForDisplay(c) {
    if (typeof c === 'string') return c;
    if (!Array.isArray(c)) return JSON.stringify(c);
    return JSON.stringify(c.map(part => {
      if (part && part.type === 'audio_url' && part.audio_url && typeof part.audio_url.url === 'string' && part.audio_url.url.startsWith('data:')) {
        const kb = Math.round(part.audio_url.url.length * 0.75 / 1024);
        return { type: 'audio_url', audio_url: { url: `[WAV ${kb} KB]` } };
      }
      return part;
    }), null, 2);
  }

  function relatedHTML(e, rel) {
    if (!rel.length) {
      return '<div style="color:var(--fg-dim);font-family:var(--font-serif);font-style:italic">No correlated events.</div>';
    }
    const refs = corrRefs(e);
    return `
      <div style="font-family:var(--font-sans);font-size:10px;letter-spacing:0.1em;text-transform:uppercase;color:var(--fg-dim);margin-bottom:10px">
        ${rel.length} events · ${refs.map(([k, v]) => `${k}=${v}`).join(' · ')}
      </div>
      ${rel.map(x => `
        <div class="row" style="cursor:pointer;padding:6px 0;border-bottom:1px dashed var(--hair)" data-seq="${x.seq}">
          <span class="k" style="font-variant-numeric:tabular-nums">${formatMs(x.t)}</span>
          <span class="v"><span class="tag lane" style="--tag-color:${laneColor(store.state.tweaks.palette, x.lane, x.kind)}">${x.lane}</span> ${x.kind}</span>
        </div>`).join('')}`;
  }

  function row(k, v, cls) {
    let val;
    if (v == null) val = '<span class="v nul">null</span>';
    else if (typeof v === 'number') val = `<span class="v num">${v}</span>`;
    else if (typeof v === 'string') val = `<span class="v ${cls || 'str'}">${escapeHTML(v)}</span>`;
    else if (typeof v === 'object') val = `<pre class="v obj" style="white-space:pre-wrap;margin:0;font:inherit">${syntaxHighlight(JSON.stringify(v, null, 2))}</pre>`;
    else val = `<span class="v">${escapeHTML(String(v))}</span>`;
    return `<div class="row"><span class="k">${k}</span>${val}</div>`;
  }
  function typeofClass(v) {
    if (typeof v === 'number') return 'num';
    if (typeof v === 'string') return 'str';
    if (v == null) return 'nul';
    return '';
  }
  function stripT(e) { const { t, ...rest } = e; return rest; }

  // ── click wiring ───────────────────────────────────────────────────────
  tabs.forEach(t => {
    t.addEventListener('click', () => {
      tabs.forEach(x => x.setAttribute('aria-selected', 'false'));
      t.setAttribute('aria-selected', 'true');
      store.setActiveTab(t.dataset.tab);
    });
  });

  body.addEventListener('click', (ev) => {
    const r = ev.target.closest('[data-seq]');
    if (!r) return;
    const seq = parseInt(r.dataset.seq, 10);
    const target = store.state.events.find(x => x.seq === seq);
    if (target) {
      store.setSelection(target);
      timeline.centerOn(target.t);
    }
  });

  btnCopy.addEventListener('click', () => {
    const e = store.state.selected;
    if (!e) return;
    navigator.clipboard.writeText(JSON.stringify(stripT(e), null, 2));
    const prev = btnCopy.textContent;
    btnCopy.textContent = 'copied ✓';
    setTimeout(() => btnCopy.textContent = prev, 900);
  });

  navBtns.forEach((b, i) => {
    b.addEventListener('click', () => {
      const e = store.state.selected;
      if (!e) return;
      const lane = store.state.events.filter(x => x.lane === e.lane);
      const idx = lane.findIndex(x => x.seq === e.seq);
      const next = i === 0 ? lane[Math.max(0, idx - 1)] : lane[Math.min(lane.length - 1, idx + 1)];
      if (next) store.setSelection(next);
    });
  });

  store.subscribe('selection', render);
  store.subscribe('events', render);
  store.subscribe('tweaks', render);
  render();
}
