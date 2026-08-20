import { formatMs } from './util.js';

export function setupInteractions({ store, timeline, audio, tooltip }) {
  const wrap   = document.getElementById('tlWrap');
  const gutter = document.getElementById('laneGutter');
  const mm     = document.getElementById('mmCanvas');
  const stCursor = document.getElementById('stCursor');
  const stTips   = document.getElementById('stTips');
  const stEvents = document.getElementById('stEvents');
  const stSeq    = document.getElementById('stSeq');
  const stDropped= document.getElementById('stDropped');
  const tlCanvas = document.getElementById('tlCanvas');

  // ── timeline mouse: pan, click-to-select, hover, scrub ─────────────────
  let dragging = false;
  let dragLast = 0;
  let selDrag  = null;       // {startMs} for shift-drag range select
  let playheadDrag = false;

  wrap.addEventListener('wheel', (e) => {
    e.preventDefault();
    const r = wrap.getBoundingClientRect();
    const px = e.clientX - r.left;
    if (e.ctrlKey || e.metaKey || Math.abs(e.deltaY) > Math.abs(e.deltaX)) {
      const factor = Math.pow(1.0015, -e.deltaY);
      timeline.zoomAtPx(px, factor);
    } else {
      timeline.panBy(-e.deltaX);
    }
  }, { passive: false });

  wrap.addEventListener('mousedown', (e) => {
    if (e.button !== 0) return;
    const r = wrap.getBoundingClientRect();
    const px = e.clientX - r.left;
    const py = e.clientY - r.top;
    const tMs = timeline.pxToMs(px);

    // Scrub the playhead during active playback.
    const pb = store.state.playback;
    if (pb.active && pb.playheadMs != null) {
      const phPx = timeline.msToPx(pb.playheadMs);
      if (Math.abs(phPx - px) < 8) {
        playheadDrag = true;
        store.setPlayback({ scrubbing: true });
        e.stopPropagation();
        return;
      }
    }

    if (e.shiftKey) {
      selDrag = { startMs: tMs };
      e.stopPropagation();
      return;
    }

    const hit = timeline.hitTest(px, py);
    if (hit) {
      if (store.state.playback.range && !store.state.playback.active) {
        store.setPlayback({ range: null });
      }
      store.setSelection(hit);
      return;
    }
    if (store.state.playback.range && !store.state.playback.active) {
      store.setPlayback({ range: null });
    }
    dragging = true;
    dragLast = e.clientX;
  });

  window.addEventListener('mousemove', (e) => {
    const r = wrap.getBoundingClientRect();
    const px = e.clientX - r.left;
    const py = e.clientY - r.top;

    // Playhead scrubbing.
    if (playheadDrag) {
      const ph = Math.max(0, timeline.pxToMs(px));
      store.setPlayback({
        playheadMs: ph,
        startWall: performance.now(),
        startMs: ph,
      });
      return;
    }

    // Range selection (shift-drag).
    if (selDrag) {
      const tMs = timeline.pxToMs(px);
      store.setPlayback({
        range: { t0: Math.min(selDrag.startMs, tMs), t1: Math.max(selDrag.startMs, tMs) },
      });
      return;
    }

    // Plain drag pans the view.
    if (dragging) {
      timeline.panBy(e.clientX - dragLast);
      dragLast = e.clientX;
      return;
    }

    // Hover -> tooltip + cursor.
    if (px < 0 || py < 0 || px > r.width || py > r.height) {
      if (store.state.hover || store.state.cursorMs != null) {
        store.setHover(null);
        store.setCursor(null);
        tooltip.hide();
        if (stCursor) stCursor.textContent = 'cursor --';
      }
      return;
    }
    store.setCursor(timeline.pxToMs(px));
    if (stCursor) stCursor.textContent = 'cursor ' + formatMs(store.state.cursorMs);
    const hit = timeline.hitTest(px, py);
    if (hit !== store.state.hover) {
      store.setHover(hit);
      if (hit) tooltip.show(hit, e.clientX, e.clientY);
      else tooltip.hide();
    } else if (hit) {
      tooltip.move(e.clientX, e.clientY);
    }
  });

  window.addEventListener('mouseup', () => {
    dragging = false;
    if (playheadDrag) {
      playheadDrag = false;
      store.setPlayback({ scrubbing: false });
      audio.reseekAudio();
    }
    if (selDrag) {
      const r = store.state.playback.range;
      if (r && r.t1 - r.t0 > 40) {
        audio.startWindow({ t0: r.t0, t1: r.t1, mode: 'window', anchor: store.state.selected });
      } else {
        store.setPlayback({ range: null });
      }
      selDrag = null;
    }
  });

  // ── lane gutter clicks: toggle visibility / dblclick-jump ──────────────
  gutter.addEventListener('click', (e) => {
    const lbl = e.target.closest('.lane-label');
    if (!lbl) return;
    store.toggleLaneHidden(lbl.dataset.lane);
  });
  gutter.addEventListener('dblclick', (e) => {
    const lbl = e.target.closest('.lane-label');
    if (!lbl) return;
    if (store.state.hiddenLanes.has(lbl.dataset.lane)) store.toggleLaneHidden(lbl.dataset.lane);
    const center = store.state.view.t0 + (timeline.tlPxWidth() / store.state.view.pxPerMs) / 2;
    const next = store.state.events.find(x => x.lane === lbl.dataset.lane && x.t > center)
              || store.state.events.find(x => x.lane === lbl.dataset.lane);
    if (next) store.setSelection(next);
  });

  // ── minimap: click + drag jumps the viewport ───────────────────────────
  let mmDragging = false;
  function mmJump(px) {
    const events = store.state.events;
    if (!events.length) return;
    const r = mm.getBoundingClientRect();
    const t1 = Math.max(500, events[events.length - 1].t + 200);
    const tClick = (px / r.width) * t1;
    const spanMs = timeline.tlPxWidth() / store.state.view.pxPerMs;
    store.state.view.t0 = Math.max(0, tClick - spanMs / 2);
    store.setFollowTail(false);
  }
  mm.addEventListener('mousedown', (e) => {
    mmDragging = true;
    const r = mm.getBoundingClientRect();
    mmJump(e.clientX - r.left);
  });
  window.addEventListener('mousemove', (e) => {
    if (!mmDragging) return;
    const r = mm.getBoundingClientRect();
    mmJump(Math.max(0, Math.min(r.width, e.clientX - r.left)));
  });
  window.addEventListener('mouseup', () => { mmDragging = false; });

  // ── keyboard ───────────────────────────────────────────────────────────
  function turnEnds() {
    return store.state.events
      .filter(e => e.lane === 'turn' && e.kind === 'turn_end')
      .map(e => e.t).sort((a, b) => a - b);
  }
  function jumpTurn(dir) {
    const ends = turnEnds();
    if (!ends.length) return;
    const center = store.state.view.t0 + (timeline.tlPxWidth() / store.state.view.pxPerMs) / 2;
    let target = null;
    if (dir > 0) target = ends.find(t => t > center + 10);
    else { for (const t of ends) { if (t < center - 10) target = t; else break; } }
    if (target == null) target = dir > 0 ? ends[ends.length - 1] : ends[0];
    timeline.centerOn(target);
  }

  // Space conductor (matches the topbar Play button).
  function handleSpace() {
    const s = store.state;
    if (s.followTail) {
      store.setFollowTail(false);
      if (s.playback.active) audio.stop();
      return;
    }
    if (s.playback.active) {
      audio.stop();
      return;
    }
    if (!s.events.length) return;
    audio.startLive();
  }

  window.addEventListener('keydown', (e) => {
    const tag = e.target?.tagName;
    if (tag === 'INPUT' || tag === 'TEXTAREA' || tag === 'SELECT') {
      if (e.key === 'Escape') e.target.blur();
      return;
    }
    if (e.key === 'Escape' && store.state.playback.active) { audio.stop(); e.preventDefault(); return; }
    if (e.ctrlKey && (e.key === ',' || e.key === '<')) { jumpTurn(-1); e.preventDefault(); return; }
    if (e.ctrlKey && (e.key === '.' || e.key === '>')) { jumpTurn(1);  e.preventDefault(); return; }
    if (e.key === ' ') { handleSpace(); e.preventDefault(); return; }
    if (e.key === 'f') { document.getElementById('btnFollow').click(); return; }
    if (e.key === ',') { timeline.panBy(60); return; }
    if (e.key === '.') { timeline.panBy(-60); return; }
  });

  // ── status bar updates ─────────────────────────────────────────────────
  store.subscribe('events', () => {
    if (stEvents)  stEvents.textContent  = store.state.events.length;
    if (stSeq)     stSeq.textContent     = store.state.events.length ? store.state.events[store.state.events.length - 1].seq : '--';
    if (stDropped) stDropped.textContent = store.errorCount();
    refreshErrorBadge();
    refreshTips();
  });
  store.subscribe('view', refreshTips);

  function refreshTips() {
    const ends = turnEnds();
    if (!ends.length) return;
    const center = store.state.view.t0 + (timeline.tlPxWidth() / store.state.view.pxPerMs) / 2;
    let idx = ends.findIndex(t => t > center);
    if (idx < 0) idx = ends.length;
    if (stTips) {
      stTips.textContent = `turn ${Math.max(1, idx)}/${ends.length}  ·  wheel zoom · drag pan · [ ] zoom · / find · space pause · r replay · ⌃, ⌃. turns`;
    }
  }

  function refreshErrorBadge() {
    const errs = store.state.events.filter(e => e.lane === 'error' || isErrKind(e.kind));
    let badge = document.getElementById('errorBadge');
    if (errs.length) {
      if (!badge) {
        badge = document.createElement('button');
        badge.id = 'errorBadge';
        badge.className = 'btn';
        badge.style.cssText = 'color:#B88080;background:rgba(184,128,128,0.12);border:1px solid rgba(184,128,128,0.3);border-radius:999px;padding:4px 12px;cursor:pointer;font-weight:600';
        badge.addEventListener('click', () => {
          const cur = store.state.selected;
          const list = errs;
          const curIdx = cur ? list.findIndex(e => e.seq === cur.seq) : -1;
          const next = list[(curIdx + 1) % list.length];
          if (next) {
            store.setSelection(next);
            timeline.centerOn(next.t);
          }
        });
        const topbar = document.querySelector('.topbar');
        if (topbar) topbar.appendChild(badge);
      }
      badge.innerHTML = `● ${errs.length} ${errs.length === 1 ? 'error' : 'errors'}`;
      badge.style.display = '';
    } else if (badge) {
      badge.style.display = 'none';
    }
  }
}

const HARD_ERR_KINDS = new Set(['error', 'raised', 'dropped', 'failed', 'phrase_error', 'bargein_missed']);
function isErrKind(k) { return HARD_ERR_KINDS.has(k); }
