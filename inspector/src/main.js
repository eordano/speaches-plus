import { createStore } from './store.js';
import { Timeline } from './timeline.js';
import { Transport, startLivePoll } from './transport.js';
import { AudioPlayer } from './audio.js';
import { setupTopbar } from './topbar.js';
import { setupPicker, flashPill } from './picker.js';
import { setupDetail } from './detail.js';
import { setupTooltip } from './tooltip.js';
import { setupInteractions } from './interactions.js';
import { setupSettings } from './settings.js';
import { setupExporters } from './exporters.js';
import { setupNewSession } from './new-session.js';

const store = createStore();
if (new URLSearchParams(location.search).has('debug')) window.__store = store;

const timeline = new Timeline({
  rulerCanvas: document.getElementById('rulerCanvas'),
  tlCanvas:    document.getElementById('tlCanvas'),
  mmCanvas:    document.getElementById('mmCanvas'),
  tlWrap:      document.getElementById('tlWrap'),
  gutter:      document.getElementById('laneGutter'),
}, store);

const transport = new Transport(store);
store.transport = transport;        // topbar wants the live alive-check

const audio = new AudioPlayer(store, timeline, document.getElementById('stAudio'));

const tooltip = setupTooltip({ store });
setupInteractions({ store, timeline, audio, tooltip });
setupTopbar({ store, audio, timeline });
setupPicker({ store });
setupDetail({ store, timeline });
setupSettings();
setupExporters({ store });

// Theme attribute is now applied once at boot, since the Tweaks UI is gone.
document.documentElement.setAttribute('data-theme', store.state.tweaks.theme || 'dark');
setupNewSession();

// ── boot ─────────────────────────────────────────────────────────────────
async function boot() {
  const params = new URLSearchParams(location.search);
  const sid = params.get('sid');
  if (sid) {
    transport.connect(sid);
  } else {
    // No URL session -> pick the first live or historical session.
    const r = await fetch('/v1/inspect/sessions/history').then(r => r.ok ? r.json() : []).catch(() => []);
    const live = await fetch('/v1/inspect/sessions').then(r => r.ok ? r.json() : []).catch(() => []);
    const pick = (live && live[0]) || (r && r[0]);
    if (pick) {
      history.replaceState(null, '', `?sid=${encodeURIComponent(pick.id)}`);
      transport.connect(pick.id);
    } else {
      // Empty state.
      const ids = document.getElementById('sessionIdText');
      if (ids) ids.textContent = 'click to pick a session';
    }
  }
  startLivePoll(transport,
    (sid) => {
      history.replaceState(null, '', `?sid=${encodeURIComponent(sid)}`);
      transport.connect(sid);
      flashPill();
    },
    () => flashPill()
  );
}

window.addEventListener('inspector:openSession', (e) => {
  const id = e.detail?.sid;
  if (!id) return;
  history.replaceState(null, '', `?sid=${encodeURIComponent(id)}`);
  transport.connect(id);
});

// Once events stop arriving for a moment, fit the whole session into the
// viewport so the user sees t=0 -> end. Historical sessions: the WS floods
// then closes, so the debounce settles ~quickly. Live sessions: it settles
// during pauses between events.
let fitScheduled = false;
let fitTimer = null;
store.subscribe('session', () => {
  fitScheduled = false;
  if (fitTimer) { clearTimeout(fitTimer); fitTimer = null; }
});
store.subscribe('events', () => {
  if (fitScheduled || !store.state.events.length) return;
  if (fitTimer) clearTimeout(fitTimer);
  fitTimer = setTimeout(() => {
    if (fitScheduled) return;
    timeline.fit();
    fitScheduled = true;
    fitTimer = null;
  }, 400);
});

// Auto-select an interesting event ~600ms after we see events arrive.
let autoSelected = false;
store.subscribe('events', () => {
  if (autoSelected) return;
  if (!store.state.events.length) return;
  setTimeout(() => {
    if (autoSelected || store.state.selected) return;
    const interesting = store.state.events.find(x => x.kind === 'first_token')
                     || store.state.events.find(x => x.kind === 'final')
                     || store.state.events.find(x => x.lane !== 'audio_level');
    if (interesting) { store.setSelection(interesting); autoSelected = true; }
  }, 600);
});
window.addEventListener('inspector:openSession', () => { autoSelected = false; });

boot();
