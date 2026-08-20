export class AudioPlayer {
  constructor(store, timeline, statusEl) {
    this.store = store;
    this.timeline = timeline;
    this.statusEl = statusEl;
    this.actx = null;
    this.sources = [];
    this.gain = { mic: null, tts: null };
    this.abort = null;
    this.rafId = null;
  }

  ctx() {
    if (!this.actx) {
      try { this.actx = new (window.AudioContext || window.webkitAudioContext)(); }
      catch (err) {
        console.error('[audio] AudioContext creation failed (replay disabled):', err);
        this.actx = null;
      }
    }
    return this.actx;
  }

  // Lane -> which channel(s) we want during replay.
  preferredChannels(anchor) {
    const prefer = { mic: true, tts: true };
    if (anchor && anchor.lane) {
      const l = anchor.lane;
      if (['vad','stt','turn','bargein'].includes(l)) prefer.tts = false;
      else if (['tts_req','tts_chunk'].includes(l)) prefer.mic = false;
      else if (l === 'audio_level' && anchor.payload?.channel === 'tts_out') prefer.mic = false;
      else if (l === 'audio_level' && anchor.payload?.channel === 'mic_in')  prefer.tts = false;
      else if (l === 'response' && anchor.kind === 'phrase_boundary')        prefer.mic = false;
    }
    const ch = this.store.state.playback.channels;
    return { mic: ch.mic && prefer.mic, tts: ch.tts && prefer.tts };
  }

  async startWindow({ t0, t1, anchor, mode = 'window' }) {
    if (!this.store.state.events.length) return;
    const last = this.store.state.events[this.store.state.events.length - 1].t;
    t0 = Math.max(0, t0);
    t1 = Math.min(last, t1);
    if (t1 <= t0) return;

    const speed = parseFloat(this.store.state.tweaks.replaySpeed || '1') || 1;
    this.store.setPlayback({
      active: true, mode,
      startWall: performance.now(),
      startMs: t0, endMs: t1,
      speed,
      playheadMs: t0,
      range: { t0, t1 },
      scrubbing: false,
    });

    if (mode === 'window') {
      const spanMs = this.timeline.tlPxWidth() / this.store.state.view.pxPerMs;
      const windowMs = t1 - t0;
      if (windowMs > spanMs * 0.9) {
        this.store.state.view.pxPerMs = (this.timeline.tlPxWidth() - 80) / windowMs;
      }
      this.store.state.view.t0 = Math.max(0, t0 - 60 / this.store.state.view.pxPerMs);
      this.store.setFollowTail(false);
    }

    this.scheduleAudio(t0 | 0, mode === 'live' ? 0 : Math.max(t0 + 100, t1) | 0, anchor);
    this.startRaf();
  }

  startLive(anchor = null) {
    const events = this.store.state.events;
    if (!events.length) return;
    const lastT = events[events.length - 1].t;
    const sel = this.store.state.selected;
    const cur = this.store.state.cursorMs;
    const t0 = Math.max(0, sel ? sel.t : (cur ?? 0));
    this.startWindow({ t0, t1: lastT, anchor: anchor || sel, mode: 'live' });
  }

  stop() {
    if (this.rafId) cancelAnimationFrame(this.rafId);
    this.rafId = null;
    this.stopAudio();
    this.store.setPlayback({
      active: false, mode: null,
      playheadMs: null, range: null, scrubbing: false,
    });
    this.updateIndicator(null);
  }

  // Restart audio sources at a new playhead (called when scrubbing settles).
  reseekAudio() {
    const ph = this.store.state.playback.playheadMs;
    if (ph == null) return;
    const mode = this.store.state.playback.mode;
    const from = ph | 0;
    const to = mode === 'window' ? (this.store.state.playback.endMs | 0) : 0;
    this.scheduleAudio(from, to, this.store.state.selected);
  }

  applyChannelGain(side, on) {
    const g = this.gain[side];
    if (g) g.gain.value = on ? 1 : 0;
  }

  // ── internals ──────────────────────────────────────────────────────────
  async scheduleAudio(fromMs, toMs, anchor) {
    this.stopAudio();
    const sid = this.store.state.sessionId;
    if (!sid) return;
    const actx = this.ctx();
    if (!actx) return;
    if (actx.state === 'suspended') {
      try { await actx.resume(); } catch (err) {
        console.error('[audio] resume failed:', err);
      }
    }
    const ctl = new AbortController();
    this.abort = ctl;
    const eff = this.preferredChannels(anchor);
    const [micBuf, ttsBuf] = await Promise.all([
      eff.mic ? this.fetchBuffer('mic_in',  fromMs, toMs, ctl.signal) : null,
      eff.tts ? this.fetchBuffer('tts_out', fromMs, toMs, ctl.signal) : null,
    ]);
    if (ctl.signal.aborted) return;
    const speed = parseFloat(this.store.state.tweaks.replaySpeed || '1') || 1;
    const t0 = actx.currentTime + 0.02;
    const schedule = (side, buf) => {
      if (!buf) return;
      const src = actx.createBufferSource();
      src.buffer = buf;
      src.playbackRate.value = speed;
      const g = actx.createGain();
      g.gain.value = 1;
      src.connect(g).connect(actx.destination);
      try { src.start(t0); } catch (err) {
        console.error('[audio] BufferSource.start failed:', side, t0, err);
      }
      this.sources.push(src);
      this.gain[side] = g;
    };
    schedule('mic', micBuf);
    schedule('tts', ttsBuf);
  }

  async fetchBuffer(channel, fromMs, toMs, signal) {
    const sid = this.store.state.sessionId;
    const url = `/v1/inspect/sessions/${encodeURIComponent(sid)}/audio?channel=${channel}&from_ms=${fromMs|0}&to_ms=${toMs|0}`;
    try {
      const r = await fetch(url, { signal });
      if (!r.ok) return null;
      const buf = await r.arrayBuffer();
      if (buf.byteLength <= 44) return null;
      return await this.ctx().decodeAudioData(buf);
    } catch (err) {
      if (err && err.name === 'AbortError') return null;
      console.error('[audio] fetch failed:', channel, err);
      return null;
    }
  }

  stopAudio() {
    if (this.abort) { this.abort.abort(); this.abort = null; }
    for (const src of this.sources) { try { src.stop(); } catch {} }
    this.sources = [];
    this.gain.mic = null;
    this.gain.tts = null;
  }

  startRaf() {
    const tick = () => {
      const pb = this.store.state.playback;
      if (!pb.active) return;
      if (!pb.scrubbing) {
        const elapsed = (performance.now() - pb.startWall) * pb.speed;
        pb.playheadMs = pb.startMs + elapsed;
      }
      const ph = pb.playheadMs;
      if (pb.mode === 'live' && !pb.scrubbing) {
        const spanMs = this.timeline.tlPxWidth() / this.store.state.view.pxPerMs;
        this.store.state.view.t0 = Math.max(0, ph - spanMs * 0.4);
      }
      this.updateIndicator(ph);
      this.store.emit('playback');
      this.store.emit('view');
      if (!pb.scrubbing && ph >= pb.endMs) { this.stop(); return; }
      this.rafId = requestAnimationFrame(tick);
    };
    this.rafId = requestAnimationFrame(tick);
  }

  updateIndicator(ph) {
    if (!this.statusEl) return;
    if (ph != null && this.store.state.playback.active) {
      this.statusEl.textContent = `audio ● playing @ ${formatMsLocal(ph)}`;
      this.statusEl.style.color = '';
    } else {
      this.statusEl.textContent = 'audio ○ idle';
      this.statusEl.style.color = 'var(--fg-faint)';
    }
  }
}

function formatMsLocal(ms) {
  if (ms == null) return '--';
  if (ms >= 1000) return (ms / 1000).toFixed(3) + 's';
  if (ms >= 1)    return ms.toFixed(1) + 'ms';
  return (ms * 1000).toFixed(1) + 'us';
}
