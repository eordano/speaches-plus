function makeBand(lane, kind, t0, t1, { open, close = null, corr = {}, ongoing = false } = {}) {
  return { lane, kind, t0, t1, open, close, corr, ongoing };
}

export function rebuildBands(events) {
  const s = {
    bands: [],
    vadOpen: new Map(), vadDone: new Map(),
    llmOpen: new Map(),
    respOpen: new Map(),
    ttsOpen: new Map(),
    ttsPlaybackOrigin: new Map(),
    ttsPlaybackCursor: new Map(),
    ttsPhraseTexts: new Map(),
    bargeinOpen: new Map(),
    toolOpen: new Map(),
    toolBand: new Map(),
  };

  for (const e of events) ingestBandEvent(s, e);

  const lastT = events.length ? events[events.length - 1].t : 0;
  closeOngoingBands(s, lastT);

  return {
    bands: s.bands,
    ttsPhraseTexts: s.ttsPhraseTexts,
    ttsChunkRows: assignTtsChunkRows(s.bands),
  };
}

// Greedy interval packing: each tts_chunk goes on the lowest row whose
// previous band ended before its t0. Returns the row count.
function assignTtsChunkRows(bands) {
  const chunks = bands.filter(b => b.lane === 'tts_chunk');
  chunks.sort((a, b) => a.t0 - b.t0);
  const rowEnds = [];
  let maxRow = 0;
  for (const band of chunks) {
    let row = -1;
    for (let r = 0; r < rowEnds.length; r++) {
      if (rowEnds[r] <= band.t0) { row = r; break; }
    }
    if (row < 0) { row = rowEnds.length; rowEnds.push(band.t1); }
    else { rowEnds[row] = band.t1; }
    band.row = row;
    if (row > maxRow) maxRow = row;
  }
  return Math.max(1, maxRow + 1);
}

function ingestBandEvent(s, e) {
  switch (e.lane) {
    case 'vad':       return ingestVad(s, e);
    case 'stt':       return (e.kind === 'final' || e.kind === 'audio_direct') ? ingestSttFinal(s, e) : undefined;
    case 'llm':       return ingestLlm(s, e);
    case 'response':  return ingestResponse(s, e);
    case 'tts_req':   return ingestTtsReq(s, e);
    case 'tts_chunk': return (e.kind === 'chunk' || e.kind === 'first_chunk') ? ingestTtsChunk(s, e) : undefined;
    case 'bargein':   return ingestBargein(s, e);
    case 'tool':      return ingestTool(s, e);
  }
}

function ingestVad(s, e) {
  const c = e.corr || {};
  if (e.kind === 'confirmed_start') {
    s.vadOpen.set(c.item_id, e);
  } else if (e.kind === 'stopped') {
    const open = s.vadOpen.get(c.item_id);
    if (!open) return;
    s.bands.push(makeBand('vad', 'speech', open.t, e.t, { open, close: e, corr: c }));
    s.vadDone.set(c.item_id, { t0: open.t, t1: e.t });
    s.vadOpen.delete(c.item_id);
  } else if (e.kind === 'pending_start') {
    s.bands.push(makeBand('vad', 'pending', e.t, e.t + 200, { open: e, corr: c }));
  }
}

function ingestSttFinal(s, e) {
  const p = e.payload || {};
  if (p.audio_start_ms == null || p.audio_end_ms == null) return;
  const c = e.corr || {};
  const vr = s.vadDone.get(c.item_id);
  const t0 = vr ? vr.t0 : p.audio_start_ms;
  const t1 = vr ? vr.t1 : p.audio_end_ms;
  s.bands.push(makeBand('stt', 'utterance', t0, t1, { open: e, corr: c }));
}

function ingestLlm(s, e) {
  const c = e.corr || {};
  if (e.kind === 'request') {
    s.llmOpen.set(c.response_id, e);
  } else if (e.kind === 'done') {
    const open = s.llmOpen.get(c.response_id);
    if (!open) return;
    s.bands.push(makeBand('llm', 'response', open.t, e.t, { open, close: e, corr: c }));
    s.llmOpen.delete(c.response_id);
  }
}

function ingestResponse(s, e) {
  const c = e.corr || {};
  if (e.kind === 'plan_start') {
    s.respOpen.set(c.response_id, e);
  } else if (e.kind === 'done') {
    const open = s.respOpen.get(c.response_id);
    if (!open) return;
    s.bands.push(makeBand('response', 'assembly', open.t, e.t, { open, close: e, corr: c }));
    s.respOpen.delete(c.response_id);
  }
}

function ingestTtsReq(s, e) {
  const c = e.corr || {};
  if (e.kind === 'phrase_sent') {
    s.ttsOpen.set(c.phrase_id, e);
    s.ttsPhraseTexts.set(c.phrase_id, e.payload?.text || '');
  } else if (e.kind === 'phrase_rendered' || e.kind === 'phrase_done' || e.kind === 'error') {
    const open = s.ttsOpen.get(c.phrase_id);
    if (!open) return;
    const bandKind = e.kind === 'error' ? 'phrase_error' : 'phrase';
    s.bands.push(makeBand('tts_req', bandKind, open.t, e.t, { open, close: e, corr: c }));
    s.ttsOpen.delete(c.phrase_id);
  }
}

// TTS chunks abut in a per-response playback cursor (not wall-clock time)
// -- the audio plays back-to-back even if the chunks arrive irregularly.
function ingestTtsChunk(s, e) {
  const c = e.corr || {};
  const p = e.payload || {};
  const rid = c.response_id;
  const ms = p.ms_audio || 0;
  if (!s.ttsPlaybackOrigin.has(rid)) {
    s.ttsPlaybackOrigin.set(rid, e.t);
    s.ttsPlaybackCursor.set(rid, 0);
  }
  const cursor = s.ttsPlaybackCursor.get(rid);
  const origin = s.ttsPlaybackOrigin.get(rid);
  s.bands.push(makeBand('tts_chunk', 'chunk', origin + cursor, origin + cursor + ms, { open: e, corr: c }));
  s.ttsPlaybackCursor.set(rid, cursor + ms);
}

function ingestBargein(s, e) {
  const c = e.corr || {};
  const key = c.response_id || c.item_id || 'default';
  if (e.kind === 'bargein_pending') {
    s.bargeinOpen.set(key, e);
  } else if (e.kind === 'bargein_fired' || e.kind === 'bargein_cancelled') {
    const open = s.bargeinOpen.get(key);
    if (!open) return;
    const bandKind = e.kind === 'bargein_fired' ? 'fired' : 'cancelled';
    s.bands.push(makeBand('bargein', bandKind, open.t, e.t, { open, close: e, corr: c }));
    s.bargeinOpen.delete(key);
  }
}

// Tool lifecycle: use_token -> result -> start_summary -> summary.
// Open the band on use_token; extend through whichever closer arrives last
// so the band's right edge tracks the most informative moment.
function ingestTool(s, e) {
  const p = e.payload || {};
  const name = p.name;
  if (!name) return;
  if (e.kind === 'use_token') {
    s.toolOpen.set(name, e);
    return;
  }
  if (e.kind !== 'result' && e.kind !== 'start_summary' && e.kind !== 'summary') return;
  const open = s.toolOpen.get(name);
  if (!open) return;
  const existing = s.toolBand.get(name);
  if (existing) {
    existing.t1 = e.t;
    existing.close = e;
    if (e.kind === 'summary' || (e.kind === 'start_summary' && existing.kind === 'result')) {
      existing.kind = e.kind;
    }
    existing.ongoing = false;
  } else {
    const band = makeBand('tool', e.kind, open.t, e.t, { open, close: e, corr: e.corr || {} });
    s.toolBand.set(name, band);
    s.bands.push(band);
  }
}

function closeOngoingBands(s, lastT) {
  const push = (lane, kind, open) => s.bands.push(
    makeBand(lane, kind, open.t, lastT, { open, corr: open.corr, ongoing: true })
  );
  s.vadOpen.forEach(o => push('vad', 'speech', o));
  s.llmOpen.forEach(o => push('llm', 'response', o));
  s.respOpen.forEach(o => push('response', 'assembly', o));
  s.ttsOpen.forEach(o => push('tts_req', 'phrase', o));
  s.bargeinOpen.forEach(o => push('bargein', 'pending', o));
}

export function rebuildTurns(events) {
  const turns = [];
  let cur = null;
  for (const e of events) {
    if (e.lane !== 'turn') continue;
    if (e.kind === 'turn_start' && e.payload?.role === 'user') {
      cur = { turn_id: e.corr.turn_id, t0: e.t, t1: null };
      turns.push(cur);
    } else if (e.kind === 'turn_end' && cur && cur.turn_id === e.corr.turn_id) {
      cur.t1 = e.t;
      cur = null;
    }
  }
  if (cur) cur.t1 = events.length ? events[events.length - 1].t : 0;
  return turns;
}

// Band labels -- what text shows on each band. Pulled from the legacy
// timeline.bandLabel for parity. Returns '' when nothing useful to show.
export function bandLabel(band, ttsPhraseTexts) {
  const o = band.open;
  switch (band.lane + ':' + band.kind) {
    case 'vad:speech':       return `speech · ${Math.round(band.t1 - band.t0)}ms`;
    case 'vad:pending':      return 'pending';
    case 'response:assembly': {
      const p = band.close?.payload;
      if (p?.failed_phrases) return `assembly · ${p.completed_phrases}/${p.phrases} phrases · ${p.failed_phrases} failed`;
      if (p?.phrases) return `assembly · ${p.phrases} phrase${p.phrases > 1 ? 's' : ''}`;
      return 'response assembly';
    }
    case 'tts_req:phrase':       return o.payload?.text || '';
    case 'tts_req:phrase_error': return '✕ error · worker closed';
  }
  if (band.lane === 'stt' && band.kind === 'utterance') {
    if (o.kind === 'audio_direct') {
      const ms = o.payload?.duration_ms || Math.round(band.t1 - band.t0);
      return `[audio direct · ${ms}ms]`;
    }
    const p = o.payload || {};
    const t = p.text || '';
    const ns = p.avg_no_speech_prob;
    return ns != null ? `"${t}" · ns ${ns}` : `"${t}"`;
  }
  if (band.lane === 'llm' && band.kind === 'response') {
    const ttft = band.close?.payload?.elapsed_ms ?? (band.t1 - band.t0);
    const tokOut = band.close?.payload?.tok_out;
    return tokOut ? `llm ${tokOut} tok · ${Math.round(ttft)}ms` : `llm ${Math.round(ttft)}ms`;
  }
  if (band.lane === 'bargein') {
    const ms = Math.round(band.t1 - band.t0);
    if (band.kind === 'fired')     return `barge-in fired · ${ms}ms`;
    if (band.kind === 'cancelled') return `false start · ${ms}ms`;
    return `pending · ${ms}ms`;
  }
  if (band.lane === 'tts_chunk') {
    const p = o.payload || {};
    const text = ttsPhraseTexts.get(band.corr?.phrase_id) || '';
    const ms = p.ms_audio || 0;
    return text ? `${text} · ${ms}ms` : `chunk #${p.chunk_idx || 0} · ${ms}ms`;
  }
  if (band.lane === 'tool') {
    const name = o.payload?.name || 'tool';
    if (band.close?.kind === 'summary') {
      const summary = band.close.payload?.summary;
      if (summary) return `${name} · ${summary}`;
    }
    if (band.close?.kind === 'start_summary') return `${name} · narrating...`;
    if (band.close?.kind === 'result') {
      const result = band.close.payload?.result;
      if (result) return `${name} · ${result}`;
    }
    const args = o.payload?.args;
    if (args && typeof args === 'object') return `${name}(${JSON.stringify(args)})`;
    return name;
  }
  return '';
}
