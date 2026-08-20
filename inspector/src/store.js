import { rebuildBands, rebuildTurns } from './bands.js';
import { ERROR_KINDS } from './lanes.js';

const TWEAKS_KEY    = 'inspect.tweaks';
const HIDDEN_KEY    = 'inspect.hiddenLanes';
const TWEAK_FALLBACK = {
  treatment:    'blocks',
  density:      'comfortable',
  palette:      'semantic',
  theme:        'dark',
  replayPre:    '500',
  replayPost:   '1000',
  replaySpeed:  '1',
};

function loadTweaks() {
  let stored = {};
  try { stored = JSON.parse(localStorage.getItem(TWEAKS_KEY) || '{}') || {}; } catch {}
  return Object.assign({}, TWEAK_FALLBACK, stored);
}
function loadHidden() {
  try { return new Set(JSON.parse(localStorage.getItem(HIDDEN_KEY) || '[]')); }
  catch { return new Set(); }
}

export function createStore() {
  const state = {
    // Session identity + connection state.
    sessionId: null,
    sessionStartMs: 0,           // monotonic anchor, set from first event's ts_mono_ns
    mode: 'replay',              // 'live' | 'replay' | 'paused'
    wsAlive: false,

    // Event stream and derived data.
    events: [],
    bands: [],
    turns: [],
    ttsPhraseTexts: new Map(),
    ttsChunkRows: 1,
    aliases: { turn: new Map(), item: new Map(), response: new Map(), phrase: new Map() },

    // View state (timeline viewport).
    view: { t0: 0, pxPerMs: 0.25 },
    followTail: true,
    paused: false,

    // Selection / hover.
    selected: null,
    hover: null,
    cursorMs: null,

    // Playback (replay scheduler).
    playback: {
      active: false,
      mode: null,                // 'window' | 'live'
      startWall: 0,
      startMs: 0,
      endMs: 0,
      speed: 1,
      scrubbing: false,
      channels: { mic: true, tts: true },
      playheadMs: null,
      range: null,               // {t0, t1} highlighted range while playing
    },

    // Persistent UI prefs.
    tweaks: loadTweaks(),
    hiddenLanes: loadHidden(),

    // Detail panel.
    activeTab: 'pretty',         // 'pretty' | 'raw' | 'related'
  };

  // Subscribers receive a topic string so they can ignore changes they
  // don't care about. Topics:
  //   'session'   -- session id changed / events reset
  //   'events'    -- events appended or replaced
  //   'view'      -- viewport / followTail changed
  //   'selection' -- selected/hover/cursor changed
  //   'playback'  -- playback active/playhead/channels changed
  //   'tweaks'    -- tweak settings changed
  //   'lanes'     -- hiddenLanes changed
  //   'mode'      -- mode pill / wsAlive changed
  const subs = new Map();
  function subscribe(topic, fn) {
    if (!subs.has(topic)) subs.set(topic, new Set());
    subs.get(topic).add(fn);
    return () => subs.get(topic).delete(fn);
  }
  function emit(topic) {
    const set = subs.get(topic);
    if (!set) return;
    for (const fn of set) fn(state);
  }

  function registerAliases(ev) {
    const c = ev.corr || {};
    const m = { turn: c.turn_id, item: c.item_id, response: c.response_id, phrase: c.phrase_id };
    for (const k of Object.keys(m)) {
      const id = m[k];
      if (!id) continue;
      const reg = state.aliases[k];
      if (!reg.has(id)) reg.set(id, reg.size + 1);
    }
  }
  function aliasFor(kind, id) {
    if (!id) return id;
    const reg = state.aliases[kind];
    if (!reg) return id;
    const n = reg.get(id);
    return n == null ? id : `${kind} ${n}`;
  }

  // Time anchor: prefer ts_mono_ns (monotonic, no NTP jumps); fall back to
  // ts_wall when the server hasn't published mono yet.
  function eventTimeMs(raw) {
    if (raw.ts_mono_ns != null) {
      const t = raw.ts_mono_ns / 1e6;
      if (state.sessionStartMs === 0) state.sessionStartMs = t;
      return t - state.sessionStartMs;
    }
    if (raw.ts_wall != null) {
      const w = raw.ts_wall * 1000;
      if (state.sessionStartMs === 0) state.sessionStartMs = w;
      return w - state.sessionStartMs;
    }
    return 0;
  }

  function normalizeEvent(raw) {
    return Object.assign({}, raw, { t: eventTimeMs(raw) });
  }

  function loadSession(sessionId) {
    state.sessionId = sessionId;
    state.sessionStartMs = 0;
    state.events = [];
    state.bands = [];
    state.turns = [];
    state.ttsPhraseTexts = new Map();
    state.ttsChunkRows = 1;
    state.aliases = { turn: new Map(), item: new Map(), response: new Map(), phrase: new Map() };
    state.selected = null;
    state.hover = null;
    state.cursorMs = null;
    state.view = { t0: 0, pxPerMs: 0.25 };
    // Always start at t0=0 so the user sees the session from the beginning.
    // Live sessions can opt into autoscroll via the ↓ Tail button.
    state.followTail = false;
    state.paused = false;
    state.playback.active = false;
    state.playback.playheadMs = null;
    state.playback.range = null;
    emit('session');
    emit('events');
    emit('selection');
    emit('view');
    emit('playback');
  }

  function appendEvent(raw) {
    const ev = normalizeEvent(raw);
    state.events.push(ev);
    registerAliases(ev);
    rebuildDerived();
    emit('events');
  }

  function setEvents(rawList) {
    state.events = [];
    state.aliases = { turn: new Map(), item: new Map(), response: new Map(), phrase: new Map() };
    state.sessionStartMs = 0;
    for (const raw of rawList) {
      const ev = normalizeEvent(raw);
      state.events.push(ev);
      registerAliases(ev);
    }
    rebuildDerived();
    emit('events');
  }

  function rebuildDerived() {
    const built = rebuildBands(state.events);
    state.bands = built.bands;
    state.ttsPhraseTexts = built.ttsPhraseTexts;
    state.ttsChunkRows = built.ttsChunkRows;
    state.turns = rebuildTurns(state.events);
  }

  function setMode(mode) {
    if (state.mode === mode) return;
    state.mode = mode;
    emit('mode');
  }
  function setWsAlive(alive) {
    if (state.wsAlive === alive) return;
    state.wsAlive = alive;
    emit('mode');
  }

  function setSelection(ev) {
    state.selected = ev;
    emit('selection');
  }
  function setHover(ev) {
    state.hover = ev;
    emit('selection');
  }
  function setCursor(ms) {
    state.cursorMs = ms;
    emit('selection');
  }

  function setView(view) {
    Object.assign(state.view, view);
    emit('view');
  }
  function setFollowTail(on) {
    state.followTail = !!on;
    emit('view');
  }
  function setPaused(p) {
    state.paused = !!p;
    emit('view');
  }

  function setPlayback(partial) {
    Object.assign(state.playback, partial);
    emit('playback');
  }

  function setTweak(key, value) {
    state.tweaks = Object.assign({}, state.tweaks, { [key]: value });
    try { localStorage.setItem(TWEAKS_KEY, JSON.stringify(state.tweaks)); } catch {}
    emit('tweaks');
  }

  function toggleLaneHidden(laneId) {
    const next = new Set(state.hiddenLanes);
    if (next.has(laneId)) next.delete(laneId); else next.add(laneId);
    state.hiddenLanes = next;
    try { localStorage.setItem(HIDDEN_KEY, JSON.stringify([...next])); } catch {}
    emit('lanes');
  }

  function setActiveTab(tab) {
    state.activeTab = tab;
    emit('selection');
  }

  function errorCount() {
    let n = 0;
    for (const e of state.events) {
      if (e.lane === 'error' || ERROR_KINDS.has(e.kind)) n++;
    }
    return n;
  }

  return {
    state,
    subscribe, emit,
    aliasFor,
    loadSession, appendEvent, setEvents,
    setMode, setWsAlive,
    setSelection, setHover, setCursor,
    setView, setFollowTail, setPaused,
    setPlayback,
    setTweak,
    toggleLaneHidden,
    setActiveTab,
    errorCount,
  };
}
