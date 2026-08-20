import { LANES, ERROR_KINDS, IMPORTANT_KINDS, isBandEndpoint, laneColor } from './lanes.js';
import { bandLabel } from './bands.js';
import { formatMs, niceStep } from './util.js';

const TURN_FILL_RULER  = 'rgba(168,144,106,0.08)';
const TURN_STROKE_RULER = 'rgba(168,144,106,0.35)';
const TURN_FILL_LANES  = 'rgba(168,144,106,0.04)';
const TURN_STROKE_LANES = 'rgba(168,144,106,0.22)';
const PLAYBACK_RANGE_FILL    = 'rgba(200,176,142,0.09)';
const PLAYBACK_RANGE_STROKE  = 'rgba(200,176,142,0.55)';
const SELECTION_STROKE       = '#C8B08E';
const CURSOR_STROKE_RULER    = 'rgba(200,176,142,0.6)';
const CURSOR_STROKE_LANES    = 'rgba(200,176,142,0.35)';

export class Timeline {
  constructor({ rulerCanvas, tlCanvas, mmCanvas, tlWrap, gutter }, store) {
    this.rulerCanvas = rulerCanvas;
    this.tlCanvas    = tlCanvas;
    this.mmCanvas    = mmCanvas;
    this.tlWrap      = tlWrap;
    this.gutter      = gutter;
    this.store       = store;
    this._dpr        = Math.max(1, window.devicePixelRatio || 1);
    this._dirty      = false;

    this.bindResize();
    this.renderGutter();

    // Re-draw on every relevant store change. The rAF debounce keeps us
    // from rendering more than once per frame even if many events arrive.
    const dirty = () => this.requestDraw();
    store.subscribe('events', dirty);
    store.subscribe('view', dirty);
    store.subscribe('selection', dirty);
    store.subscribe('playback', dirty);
    store.subscribe('lanes', () => { this.refreshGutter(); dirty(); });
    store.subscribe('tweaks', () => { this.refreshGutterHeights(); dirty(); });
  }

  // ── scheduling ─────────────────────────────────────────────────────────
  requestDraw() {
    if (this._dirty) return;
    this._dirty = true;
    requestAnimationFrame(() => {
      this._dirty = false;
      this.followTailIfActive();
      this.draw();
    });
  }

  followTailIfActive() {
    const s = this.store.state;
    if (!s.followTail || s.paused || !s.events.length) return;
    const last = s.events[s.events.length - 1].t;
    const spanMs = this.tlPxWidth() / s.view.pxPerMs;
    const pad = 80 / s.view.pxPerMs;
    s.view.t0 = Math.max(0, last + pad - spanMs);
  }

  bindResize() {
    const resize = () => {
      [this.rulerCanvas, this.tlCanvas, this.mmCanvas].forEach(c => {
        const r = c.getBoundingClientRect();
        c.width  = Math.max(1, Math.floor(r.width  * this._dpr));
        c.height = Math.max(1, Math.floor(r.height * this._dpr));
      });
      this.requestDraw();
    };
    new ResizeObserver(resize).observe(this.tlWrap);
    new ResizeObserver(resize).observe(this.rulerCanvas);
    new ResizeObserver(resize).observe(this.mmCanvas);
    requestAnimationFrame(resize);
  }
  resize() { this.bindResize(); }   // legacy compat for code that calls tl.resize()

  // ── coordinate helpers ─────────────────────────────────────────────────
  tlPxWidth()  { return this.tlCanvas.getBoundingClientRect().width; }
  tlPxHeight() { return this.tlCanvas.getBoundingClientRect().height; }
  msToPx(ms)   { return (ms - this.store.state.view.t0) * this.store.state.view.pxPerMs; }
  pxToMs(px)   { return this.store.state.view.t0 + px / this.store.state.view.pxPerMs; }

  // ── lane geometry ──────────────────────────────────────────────────────
  laneHeight(laneId) {
    const base = this.store.state.tweaks.density === 'compact' ? 32 : 42;
    if (laneId === 'tts_chunk') {
      const rows = Math.max(1, this.store.state.ttsChunkRows || 1);
      return base + (rows - 1) * this.subRowHeight();
    }
    return base;
  }
  subRowHeight() { return this.store.state.tweaks.density === 'compact' ? 16 : 20; }
  laneYOffset(laneIdx) {
    let y = 0;
    for (let i = 0; i < laneIdx; i++) y += this.laneHeight(LANES[i].id);
    return y;
  }
  bandRowMetrics(band, laneY, laneH) {
    if (band.lane === 'tts_chunk') {
      const base = this.store.state.tweaks.density === 'compact' ? 32 : 42;
      const subH = this.subRowHeight();
      const row = band.row || 0;
      if (row === 0) return { top: laneY + 6, rowH: base - 12 };
      return { top: laneY + base + (row - 1) * subH + 2, rowH: subH - 4 };
    }
    return { top: laneY + 6, rowH: laneH - 12 };
  }
  isLaneHidden(laneId) { return this.store.state.hiddenLanes.has(laneId); }

  // ── gutter (lane labels) ───────────────────────────────────────────────
  renderGutter() {
    const palette = this.store.state.tweaks.palette;
    const pal = (palette && (palette in (window.PALETTES || {}))) ? window.PALETTES[palette] : null;  // shim guard
    this.gutter.innerHTML = '';
    LANES.forEach(lane => {
      const div = document.createElement('div');
      div.className = 'lane-label';
      div.dataset.lane = lane.id;
      div.style.setProperty('--lane-color', laneColor(this.store.state.tweaks.palette, lane.id));
      div.innerHTML = `
        <span class="swatch"></span>
        <div>
          <div class="name">${lane.name}</div>
          <div class="sub">${lane.hint}</div>
        </div>
        <span class="count" data-count="${lane.id}">0</span>
      `;
      this.gutter.appendChild(div);
    });
    this.refreshGutterHeights();
  }
  refreshGutterHeights() {
    this.gutter.querySelectorAll('.lane-label').forEach(el => {
      el.style.height = this.laneHeight(el.dataset.lane) + 'px';
    });
  }
  refreshGutter() {
    const counts = {};
    for (const e of this.store.state.events) counts[e.lane] = (counts[e.lane] || 0) + 1;
    this.gutter.querySelectorAll('[data-count]').forEach(el => {
      el.textContent = counts[el.dataset.count] || 0;
    });
    const hidden = this.store.state.hiddenLanes;
    this.gutter.querySelectorAll('.lane-label').forEach(el => {
      const id = el.dataset.lane;
      const isHidden = hidden.has(id);
      el.style.opacity = isHidden ? '0.35' : '';
      el.style.textDecoration = isHidden ? 'line-through' : '';
      el.style.setProperty('--lane-color', laneColor(this.store.state.tweaks.palette, id));
    });
  }

  // ── master draw ────────────────────────────────────────────────────────
  draw() {
    this.refreshGutter();
    this.drawRuler();
    this.drawLaneRows();
    this.drawMinimap();
  }

  // ── ruler ──────────────────────────────────────────────────────────────
  drawRuler() {
    const c = this.rulerCanvas, ctx = c.getContext('2d');
    ctx.setTransform(this._dpr, 0, 0, this._dpr, 0, 0);
    const w = c.width / this._dpr, h = c.height / this._dpr;
    ctx.clearRect(0, 0, w, h);
    ctx.fillStyle = '#151515';
    ctx.fillRect(0, 0, w, h);

    const s = this.store.state;
    const msPerTick = niceStep(120 / s.view.pxPerMs);
    const t0 = Math.floor(s.view.t0 / msPerTick) * msPerTick;
    const t1 = s.view.t0 + w / s.view.pxPerMs;

    // Turn shading.
    for (const turn of s.turns) {
      const x0 = this.msToPx(turn.t0), x1 = this.msToPx(turn.t1);
      if (x1 < 0 || x0 > w) continue;
      ctx.fillStyle = TURN_FILL_RULER;
      ctx.fillRect(x0, 0, Math.max(2, x1 - x0), h);
      ctx.strokeStyle = TURN_STROKE_RULER;
      ctx.beginPath(); ctx.moveTo(x0 + 0.5, 0); ctx.lineTo(x0 + 0.5, h); ctx.stroke();
    }

    // Tick marks + labels.
    ctx.font = '11px ui-monospace, "SF Mono", Consolas, monospace';
    ctx.textBaseline = 'middle';
    for (let t = t0; t <= t1 + msPerTick; t += msPerTick) {
      const x = Math.round(this.msToPx(t)) + 0.5;
      ctx.strokeStyle = '#2E2E2E';
      ctx.beginPath(); ctx.moveTo(x, h - 12); ctx.lineTo(x, h); ctx.stroke();
      ctx.fillStyle = '#9B9590';
      ctx.fillText(formatMs(t), x + 4, h - 6);
      const sub = msPerTick / 5;
      for (let k = 1; k < 5; k++) {
        const sx = Math.round(this.msToPx(t + sub * k)) + 0.5;
        ctx.strokeStyle = '#232323';
        ctx.beginPath(); ctx.moveTo(sx, h - 6); ctx.lineTo(sx, h); ctx.stroke();
      }
    }
    ctx.strokeStyle = '#2E2E2E';
    ctx.beginPath(); ctx.moveTo(0, h - 0.5); ctx.lineTo(w, h - 0.5); ctx.stroke();

    if (s.cursorMs != null) {
      const x = Math.round(this.msToPx(s.cursorMs)) + 0.5;
      ctx.strokeStyle = CURSOR_STROKE_RULER;
      ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, h); ctx.stroke();
    }
    if (s.selected) {
      const x = Math.round(this.msToPx(s.selected.t)) + 0.5;
      ctx.strokeStyle = SELECTION_STROKE;
      ctx.lineWidth = 1.5;
      ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, h); ctx.stroke();
      ctx.lineWidth = 1;
    }
  }

  // ── lane rows + bands + ticks + sparkline + playhead ───────────────────
  drawLaneRows() {
    const c = this.tlCanvas, ctx = c.getContext('2d');
    ctx.setTransform(this._dpr, 0, 0, this._dpr, 0, 0);
    const w = c.width / this._dpr, h = c.height / this._dpr;
    ctx.clearRect(0, 0, w, h);

    const s = this.store.state;
    const laneY = new Array(LANES.length);
    const laneH = new Array(LANES.length);
    {
      let y = 0;
      for (let i = 0; i < LANES.length; i++) {
        laneY[i] = y;
        laneH[i] = this.laneHeight(LANES[i].id);
        y += laneH[i];
      }
    }

    // Striped row backgrounds.
    for (let i = 0; i < LANES.length; i++) {
      ctx.fillStyle = i % 2 ? '#1A1A1A' : '#181818';
      ctx.fillRect(0, laneY[i], w, laneH[i]);
    }
    ctx.strokeStyle = '#232323';
    for (let i = 1; i <= LANES.length; i++) {
      const y = (i < LANES.length ? laneY[i] : laneY[i - 1] + laneH[i - 1]) - 0.5;
      ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(w, y); ctx.stroke();
    }

    // Turn shading across lanes.
    for (const turn of s.turns) {
      const x0 = this.msToPx(turn.t0), x1 = this.msToPx(turn.t1);
      if (x1 < 0 || x0 > w) continue;
      ctx.fillStyle = TURN_FILL_LANES;
      ctx.fillRect(x0, 0, Math.max(2, x1 - x0), h);
      ctx.strokeStyle = TURN_STROKE_LANES;
      ctx.setLineDash([4, 4]);
      ctx.beginPath(); ctx.moveTo(x0 + 0.5, 0); ctx.lineTo(x0 + 0.5, h); ctx.stroke();
      ctx.setLineDash([]);
    }

    // Playback range highlight.
    if (s.playback.range) {
      const rx0 = this.msToPx(s.playback.range.t0);
      const rx1 = this.msToPx(s.playback.range.t1);
      ctx.fillStyle = PLAYBACK_RANGE_FILL;
      ctx.fillRect(rx0, 0, Math.max(2, rx1 - rx0), h);
      ctx.strokeStyle = PLAYBACK_RANGE_STROKE;
      ctx.setLineDash([2, 3]);
      ctx.beginPath(); ctx.moveTo(rx0 + 0.5, 0); ctx.lineTo(rx0 + 0.5, h); ctx.stroke();
      ctx.beginPath(); ctx.moveTo(rx1 - 0.5, 0); ctx.lineTo(rx1 - 0.5, h); ctx.stroke();
      ctx.setLineDash([]);
    }

    // Vertical grid (matches ruler ticks).
    const msPerTick = niceStep(120 / s.view.pxPerMs);
    const gridT0 = Math.floor(s.view.t0 / msPerTick) * msPerTick;
    const gridT1 = s.view.t0 + w / s.view.pxPerMs;
    ctx.strokeStyle = '#202020';
    for (let t = gridT0; t <= gridT1 + msPerTick; t += msPerTick) {
      const x = Math.round(this.msToPx(t)) + 0.5;
      ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, h); ctx.stroke();
    }

    this.drawAudioSparkline(ctx, w, laneY, laneH);

    // Bands.
    for (const band of s.bands) {
      if (this.isLaneHidden(band.lane)) continue;
      const laneIdx = LANES.findIndex(l => l.id === band.lane);
      if (laneIdx < 0) continue;
      const x0 = this.msToPx(band.t0);
      const x1 = this.msToPx(band.t1);
      if (x1 < 0 || x0 > w) continue;
      const col = laneColor(s.tweaks.palette, band.lane, band.kind);
      const { top, rowH } = this.bandRowMetrics(band, laneY[laneIdx], laneH[laneIdx]);
      this.drawBand(ctx, band, x0, x1, top, rowH, col);
    }

    // Ticks for non-band events.
    const visT0 = s.view.t0;
    const visT1 = s.view.t0 + w / s.view.pxPerMs;
    for (const e of s.events) {
      if (e.lane === 'audio_level') continue;
      if (this.isLaneHidden(e.lane)) continue;
      if (e.t < visT0 - 10 || e.t > visT1 + 10) continue;
      if (isBandEndpoint(e)) continue;
      const laneIdx = LANES.findIndex(l => l.id === e.lane);
      if (laneIdx < 0) continue;
      const col = laneColor(s.tweaks.palette, e.lane, e.kind);
      const x = this.msToPx(e.t);
      this.drawTick(ctx, e, x, laneY[laneIdx], laneH[laneIdx], col);
    }

    this.drawLlmTokenLabels(ctx, w, visT0, visT1, laneY);

    // Cursor + selection + playhead overlays.
    if (s.cursorMs != null) {
      const x = Math.round(this.msToPx(s.cursorMs)) + 0.5;
      ctx.strokeStyle = CURSOR_STROKE_LANES;
      ctx.setLineDash([3, 3]);
      ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, h); ctx.stroke();
      ctx.setLineDash([]);
    }
    if (s.selected) {
      const x = Math.round(this.msToPx(s.selected.t)) + 0.5;
      ctx.strokeStyle = SELECTION_STROKE;
      ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, h); ctx.stroke();
    }
    if (s.playback.playheadMs != null) {
      const x = Math.round(this.msToPx(s.playback.playheadMs)) + 0.5;
      ctx.strokeStyle = '#F2EDE4';
      ctx.lineWidth = 1.5;
      ctx.shadowColor = 'rgba(242,237,228,0.6)';
      ctx.shadowBlur = 6;
      ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, h); ctx.stroke();
      ctx.shadowBlur = 0;
      ctx.lineWidth = 1;
      ctx.fillStyle = '#F2EDE4';
      ctx.beginPath();
      ctx.moveTo(x - 4, 0); ctx.lineTo(x + 4, 0); ctx.lineTo(x, 5);
      ctx.closePath(); ctx.fill();
    }
  }

  drawBand(ctx, band, x0, x1, top, rowH, col) {
    const w = Math.max(2, x1 - x0);
    if (band.kind === 'phrase_error') {
      const grad = ctx.createLinearGradient(0, top, 0, top + rowH);
      grad.addColorStop(0, col);
      grad.addColorStop(1, 'rgba(184,128,128,0.35)');
      ctx.fillStyle = grad; ctx.fillRect(x0, top, w, rowH);
      ctx.strokeStyle = col; ctx.setLineDash([3, 2]);
      ctx.strokeRect(x0 + 0.5, top + 0.5, w - 1, rowH - 1);
      ctx.setLineDash([]);
      this.drawBandLabel(ctx, band, x0, top, w, rowH);
      return;
    }
    ctx.fillStyle = col + '40';
    ctx.fillRect(x0, top, w, rowH);
    ctx.fillStyle = col;
    ctx.fillRect(x0, top, 2, rowH);
    if (!band.ongoing) ctx.fillRect(x1 - 2, top, 2, rowH);
    ctx.globalAlpha = 0.85;
    ctx.fillRect(x0, top, w, 1.5);
    ctx.globalAlpha = 1;
    this.drawBandLabel(ctx, band, x0, top, w, rowH);
  }

  drawBandLabel(ctx, band, x, y, w, h) {
    if (w < 48) return;
    const label = bandLabel(band, this.store.state.ttsPhraseTexts);
    if (!label) return;
    ctx.font = '11px ui-monospace, monospace';
    ctx.textBaseline = 'middle';
    ctx.fillStyle = '#F2EDE4';
    // Sticky labels (STT/LLM/TTS) track the left viewport edge.
    const sticky = (band.lane === 'stt' && band.kind === 'utterance')
                || (band.lane === 'llm' && band.kind === 'response')
                || (band.lane === 'tts_chunk');
    const tx = sticky ? Math.max(x + 6, 6) : (x + 6);
    ctx.save();
    ctx.beginPath(); ctx.rect(x, y, w, h); ctx.clip();
    ctx.fillText(label, tx, y + h / 2);
    ctx.restore();
  }

  drawTick(ctx, e, x, y, laneH, col) {
    const treatment = this.store.state.tweaks.treatment;
    const important = IMPORTANT_KINDS.has(e.kind);
    if (treatment === 'blocks') {
      const h = laneH - 14;
      const rw = e.kind === 'partial' || e.kind === 'chunk' ? 2 : (important ? 4 : 2);
      ctx.fillStyle = col;
      ctx.fillRect(x - rw / 2, y + 7, rw, h);
      if (ERROR_KINDS.has(e.kind)) {
        ctx.fillStyle = '#B88080';
        ctx.fillRect(x - 2, y + 3, 4, 4);
      }
    } else if (treatment === 'ticks') {
      ctx.strokeStyle = col;
      ctx.lineWidth = important ? 1.8 : 1;
      ctx.beginPath();
      ctx.moveTo(x + 0.5, y + 8);
      ctx.lineTo(x + 0.5, y + laneH - 6);
      ctx.stroke();
      ctx.lineWidth = 1;
    } else { // lollipop
      const cy = y + laneH / 2 + 4;
      ctx.strokeStyle = col + 'AA';
      ctx.beginPath();
      ctx.moveTo(x + 0.5, y + laneH - 5);
      ctx.lineTo(x + 0.5, cy);
      ctx.stroke();
      ctx.fillStyle = col;
      const r = important ? 3.2 : 2;
      ctx.beginPath(); ctx.arc(x, cy, r, 0, Math.PI * 2); ctx.fill();
    }

    const sel = this.store.state.selected;
    const hov = this.store.state.hover;
    if (sel && sel.seq === e.seq) {
      ctx.strokeStyle = SELECTION_STROKE;
      ctx.lineWidth = 1.5;
      ctx.strokeRect(x - 6, y + 3, 12, laneH - 6);
      ctx.lineWidth = 1;
    }
    if (hov && hov.seq === e.seq) {
      ctx.strokeStyle = 'rgba(242,237,228,0.45)';
      ctx.strokeRect(x - 5, y + 5, 10, laneH - 10);
    }
  }

  drawLlmTokenLabels(ctx, w, visT0, visT1, laneY) {
    if (this.isLaneHidden('llm')) return;
    const laneIdx = LANES.findIndex(l => l.id === 'llm');
    if (laneIdx < 0) return;
    const laneH = this.laneHeight('llm');
    const y = laneY[laneIdx];
    ctx.font = '11px ui-monospace, monospace';
    ctx.textBaseline = 'middle';
    ctx.fillStyle = '#F2EDE4';
    const top = y + 6;
    const rowH = laneH - 12;
    const labelY = top + rowH / 2;
    const pad = 3;
    let lastEndX = -Infinity;
    for (const e of this.store.state.events) {
      if (e.lane !== 'llm' || e.kind !== 'chunk') continue;
      if (e.t < visT0 - 10 || e.t > visT1 + 10) continue;
      const delta = e.payload && e.payload.delta;
      if (!delta) continue;
      const text = String(delta).replace(/\s+/g, ' ');
      if (!text) continue;
      const x = this.msToPx(e.t);
      if (x < lastEndX + pad) continue;
      ctx.fillText(text, x + 6, labelY);
      lastEndX = x + 6 + ctx.measureText(text).width;
    }
  }

  drawAudioSparkline(ctx, w, laneY, laneH) {
    if (this.isLaneHidden('audio_level')) return;
    const laneIdx = LANES.findIndex(l => l.id === 'audio_level');
    if (laneIdx < 0) return;
    const y = laneY[laneIdx];
    const lh = laneH[laneIdx];
    const samples = this.store.state.events.filter(e => e.lane === 'audio_level');
    if (!samples.length) return;
    const mic = [], tts = [];
    for (const sample of samples) {
      const channel = sample.payload && sample.payload.channel;
      (channel === 'tts_out' ? tts : mic).push(sample);
    }
    const padY = 3;
    const half = (lh - padY * 2) / 2;
    const scale = (rms) => Math.min(1, rms / 0.18);
    const midY = y + padY + half;
    const drawSide = (arr, baselineY, dir, color) => {
      if (!arr.length) return;
      ctx.beginPath();
      let started = false;
      for (const sample of arr) {
        const x = this.msToPx(sample.t);
        if (x < -2 || x > w + 2) continue;
        const rms = (sample.payload && sample.payload.rms) || 0;
        const yy = baselineY + dir * half * scale(rms);
        if (!started) { ctx.moveTo(x, baselineY); ctx.lineTo(x, yy); started = true; }
        else ctx.lineTo(x, yy);
      }
      if (!started) return;
      const lastX = this.msToPx(arr[arr.length - 1].t);
      ctx.lineTo(lastX, baselineY);
      ctx.closePath();
      ctx.fillStyle = color + '40';
      ctx.fill();
      ctx.strokeStyle = color;
      ctx.lineWidth = 1;
      ctx.stroke();
    };
    drawSide(mic, midY, -1, laneColor(this.store.state.tweaks.palette, 'vad'));
    drawSide(tts, midY, +1, laneColor(this.store.state.tweaks.palette, 'tts_chunk'));
    ctx.strokeStyle = '#232323';
    ctx.beginPath();
    ctx.moveTo(0, Math.round(midY) + 0.5);
    ctx.lineTo(w, Math.round(midY) + 0.5);
    ctx.stroke();
  }

  // ── minimap ────────────────────────────────────────────────────────────
  drawMinimap() {
    const c = this.mmCanvas, ctx = c.getContext('2d');
    ctx.setTransform(this._dpr, 0, 0, this._dpr, 0, 0);
    const w = c.width / this._dpr, h = c.height / this._dpr;
    ctx.clearRect(0, 0, w, h);
    const s = this.store.state;
    if (!s.events.length) return;
    const t0 = 0;
    const t1 = Math.max(500, s.events[s.events.length - 1].t + 200);
    const laneH = h / LANES.length;
    for (let i = 0; i < LANES.length; i++) {
      ctx.fillStyle = i % 2 ? '#181818' : '#1A1A1A';
      ctx.fillRect(0, i * laneH, w, laneH);
    }
    for (const turn of s.turns) {
      const x0 = (turn.t0 / (t1 - t0)) * w;
      const x1 = (turn.t1 / (t1 - t0)) * w;
      ctx.fillStyle = 'rgba(168,144,106,0.10)';
      ctx.fillRect(x0, 0, Math.max(1, x1 - x0), h);
    }
    for (const e of s.events) {
      const laneIdx = LANES.findIndex(l => l.id === e.lane);
      if (laneIdx < 0) continue;
      const x = (e.t - t0) / (t1 - t0) * w;
      ctx.fillStyle = laneColor(s.tweaks.palette, e.lane, e.kind);
      ctx.fillRect(x, laneIdx * laneH + 1, 1, laneH - 2);
    }
    for (const band of s.bands) {
      const laneIdx = LANES.findIndex(l => l.id === band.lane);
      if (laneIdx < 0) continue;
      const x0 = (band.t0 - t0) / (t1 - t0) * w;
      const x1 = (band.t1 - t0) / (t1 - t0) * w;
      ctx.fillStyle = laneColor(s.tweaks.palette, band.lane, band.kind) + '60';
      ctx.fillRect(x0, laneIdx * laneH + 1, Math.max(1, x1 - x0), laneH - 2);
    }
    const vx0 = (s.view.t0 - t0) / (t1 - t0) * w;
    const vw  = (this.tlPxWidth() / s.view.pxPerMs) / (t1 - t0) * w;
    ctx.strokeStyle = SELECTION_STROKE;
    ctx.lineWidth = 1.5;
    ctx.strokeRect(vx0 + 0.5, 0.5, Math.max(8, vw) - 1, h - 1);
    ctx.fillStyle = 'rgba(200,176,142,0.08)';
    ctx.fillRect(vx0, 0, Math.max(8, vw), h);
    ctx.lineWidth = 1;
  }

  // ── hit testing ────────────────────────────────────────────────────────
  hitTest(px, py) {
    const s = this.store.state;
    let laneIdx = -1, laneY = 0, laneH = 0;
    for (let i = 0, y = 0; i < LANES.length; i++) {
      const hh = this.laneHeight(LANES[i].id);
      if (py >= y && py < y + hh) { laneIdx = i; laneY = y; laneH = hh; break; }
      y += hh;
    }
    if (laneIdx < 0) return null;
    const laneId = LANES[laneIdx].id;
    if (this.isLaneHidden(laneId)) return null;
    const tMs = this.pxToMs(px);
    const tolMs = 6 / s.view.pxPerMs;

    let chunkRow = null;
    if (laneId === 'tts_chunk') {
      const localY = py - laneY;
      const base = s.tweaks.density === 'compact' ? 32 : 42;
      if (localY < base) chunkRow = 0;
      else chunkRow = 1 + Math.floor((localY - base) / this.subRowHeight());
    }
    const band = findBandAt(s.bands, laneId, tMs, tolMs, chunkRow);

    if (band && isProgressiveBand(band)) {
      const candidates = [];
      const progressive = progressiveHit(s.events, band, tMs);
      if (progressive) candidates.push(progressive);
      if (band.open)   candidates.push(band.open);
      if (band.close)  candidates.push(band.close);
      if (!candidates.length) return null;
      let best = candidates[0], bestDist = Math.abs(best.t - tMs);
      for (let i = 1; i < candidates.length; i++) {
        const d = Math.abs(candidates[i].t - tMs);
        if (d < bestDist) { best = candidates[i]; bestDist = d; }
      }
      return best;
    }

    let best = null, bestDist = Infinity;
    for (const e of s.events) {
      if (e.lane !== laneId) continue;
      if (e.lane === 'audio_level') continue;
      const d = Math.abs(e.t - tMs);
      if (d < bestDist && d <= tolMs) { best = e; bestDist = d; }
    }
    if (best) return best;
    return band ? pickBandEndpoint(band, tMs) : null;
  }

  // ── view actions ───────────────────────────────────────────────────────
  zoomAtPx(px, factor) {
    const s = this.store.state;
    const t = this.pxToMs(px);
    const next = Math.max(0.01, Math.min(6, s.view.pxPerMs * factor));
    s.view.pxPerMs = next;
    s.view.t0 = Math.max(0, t - px / next);
    s.followTail = false;
    this.store.emit('view');
  }
  panBy(dx) {
    const s = this.store.state;
    s.view.t0 = Math.max(0, s.view.t0 - dx / s.view.pxPerMs);
    s.followTail = false;
    this.store.emit('view');
  }
  fit() {
    const s = this.store.state;
    if (!s.events.length) return;
    const last = s.events[s.events.length - 1].t;
    const w = this.tlPxWidth();
    s.view.pxPerMs = (w - 40) / Math.max(500, last);
    s.view.t0 = 0;
    this.store.emit('view');
  }
  toEnd() {
    const s = this.store.state;
    s.followTail = true;
    if (s.events.length) {
      const last = s.events[s.events.length - 1].t;
      const spanMs = this.tlPxWidth() / s.view.pxPerMs;
      const pad = 80 / s.view.pxPerMs;
      s.view.t0 = Math.max(0, last + pad - spanMs);
    }
    this.store.emit('view');
  }
  centerOn(tMs) {
    const s = this.store.state;
    const span = this.tlPxWidth() / s.view.pxPerMs;
    s.view.t0 = Math.max(0, tMs - span / 2);
    s.followTail = false;
    this.store.emit('view');
  }
}

function findBandAt(bands, laneId, tMs, tolMs, row = null) {
  let containing = null;
  let nearest = null, nearestDist = Infinity;
  for (const band of bands) {
    if (band.lane !== laneId) continue;
    if (row != null && (band.row || 0) !== row) continue;
    if (tMs >= band.t0 && tMs <= band.t1) { containing = band; break; }
    if (tMs >= band.t0 - tolMs && tMs <= band.t1 + tolMs) {
      const centerDist = Math.abs((band.t0 + band.t1) / 2 - tMs);
      if (centerDist < nearestDist) { nearest = band; nearestDist = centerDist; }
    }
  }
  return containing || nearest;
}

function pickBandEndpoint(band, tMs) {
  const oD = Math.abs(band.open.t - tMs);
  const cD = band.close ? Math.abs(band.close.t - tMs) : Infinity;
  return cD < oD ? band.close : band.open;
}

function isProgressiveBand(band) {
  return (band.lane === 'llm' && band.kind === 'response')
      || (band.lane === 'stt' && band.kind === 'utterance')
      || (band.lane === 'response' && band.kind === 'assembly')
      || (band.lane === 'tts_chunk');
}

function progressiveHit(events, band, tMs) {
  if (band.lane === 'tts_chunk') return band.open;
  let kinds, scopeField;
  if (band.lane === 'llm')           { kinds = new Set(['chunk']);            scopeField = 'response_id'; }
  else if (band.lane === 'stt')      { kinds = new Set(['partial', 'final']); scopeField = 'item_id'; }
  else if (band.lane === 'response') { kinds = new Set(['phrase_boundary']);  scopeField = 'response_id'; }
  else return null;
  const scope = band.corr && band.corr[scopeField];
  let latest = null;
  for (const e of events) {
    if (e.lane !== band.lane) continue;
    if (!kinds.has(e.kind)) continue;
    if (scope && e.corr && e.corr[scopeField] !== scope) continue;
    if (e.t > tMs) break;
    latest = e;
  }
  return latest;
}
