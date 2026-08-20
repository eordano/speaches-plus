export class Transport {
  constructor(store) {
    this.store = store;
    this.ws = null;
  }

  connect(sessionId) {
    this.disconnect();
    this.store.loadSession(sessionId);

    const scheme = location.protocol === 'https:' ? 'wss' : 'ws';
    const host = location.host;
    const url = `${scheme}://${host}/v1/inspect/${encodeURIComponent(sessionId)}/stream`;
    const ws = new WebSocket(url);
    ws.binaryType = 'arraybuffer';
    this.ws = ws;

    this.store.setMode('live');
    this.store.setWsAlive(false);

    ws.onopen = () => {
      this.store.setWsAlive(true);
    };

    ws.onmessage = (ev) => {
      const text =
        typeof ev.data === 'string' ? ev.data :
        ev.data instanceof ArrayBuffer ? new TextDecoder().decode(ev.data) :
        null;
      if (!text) return;
      for (const line of text.split('\n')) {
        if (!line.trim()) continue;
        try {
          const raw = JSON.parse(line);
          this.store.appendEvent(raw);
        } catch (err) {
          console.error('[transport] failed to parse event line:', err, line);
        }
      }
    };

    ws.onclose = () => {
      this.store.setWsAlive(false);
      this.store.setMode('replay');
      if (this.ws === ws) this.ws = null;
    };

    ws.onerror = (ev) => {
      console.error('[transport] WS error:', ev);
      this.store.setWsAlive(false);
      this.store.setMode('replay');
    };
  }

  disconnect() {
    if (this.ws) {
      try { this.ws.close(); } catch {}
      this.ws = null;
    }
  }

  isAlive() {
    return !!(this.ws && this.ws.readyState === 1);
  }

  // Used by the New Session client to send session.update messages over a
  // separately-owned realtime WS. The inspector exposes the realtime WS
  // via a global handle to keep that contract simple.
  static realtimeWs() { return window.__inspectRealtimeWs || null; }
  static setRealtimeWs(ws) { window.__inspectRealtimeWs = ws; }

  static sendSessionUpdate(partial) {
    const ws = Transport.realtimeWs();
    if (!ws || ws.readyState !== 1) return false;
    ws.send(JSON.stringify({ type: 'session.update', session: partial }));
    return true;
  }
}

// Polls /v1/inspect/sessions every 2s to discover newly-started live sessions.
// Calls onAttach(sid) when one appears and the inspector is idle; calls
// onForeign(sid) when a different live session is in progress (so the UI
// can flash an indicator without yanking the user away from what they're
// looking at).
export function startLivePoll(transport, onAttach, onForeign) {
  let lastIds = new Set();
  setInterval(async () => {
    try {
      const r = await fetch('/v1/inspect/sessions', { cache: 'no-store' });
      if (!r.ok) return;
      const arr = await r.json();
      const ids = new Set(arr.map(s => s.id));
      const fresh = [...ids].filter(i => !lastIds.has(i));
      lastIds = ids;
      if (!fresh.length) return;
      const onDead = !transport.isAlive();
      if (onDead) {
        onAttach && onAttach(fresh[0]);
      } else if (!ids.has(transport.store.state.sessionId)) {
        onForeign && onForeign();
      }
    } catch (err) {
      console.error('[transport] live-poll failed:', err);
    }
  }, 2000);
}

// Read /v1/inspect/sessions and /sessions/history once.
export async function fetchSessions() {
  const [live, hist] = await Promise.all([
    fetch('/v1/inspect/sessions').then(r => r.ok ? r.json() : []).catch(() => []),
    fetch('/v1/inspect/sessions/history').then(r => r.ok ? r.json() : []).catch(() => []),
  ]);
  return { live, hist };
}
