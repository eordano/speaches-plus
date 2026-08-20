import { fetchSessions } from './transport.js';

export function setupPicker({ store }) {
  const pill = document.getElementById('sessionPill');
  const idText = document.getElementById('sessionIdText');

  store.subscribe('session', () => {
    idText.textContent = store.state.sessionId || '--';
  });
  // initial render in case session is already set
  if (store.state.sessionId) idText.textContent = store.state.sessionId;

  pill.addEventListener('click', async () => {
    const pop = document.createElement('div');
    pop.style.cssText = `
      position: absolute; z-index: 40;
      background: var(--bg-panel);
      border: 1px solid var(--hair-2);
      border-radius: 8px;
      padding: 6px;
      min-width: 320px;
      font-family: var(--font-mono);
      font-size: 12px;
      max-height: 60vh;
      overflow: auto;
      box-shadow: var(--shadow-raised);
    `;
    const r = pill.getBoundingClientRect();
    pop.style.left = r.left + 'px';
    pop.style.top  = (r.bottom + 4) + 'px';
    document.body.appendChild(pop);

    function close() {
      pop.remove();
      document.removeEventListener('mousedown', onDoc);
    }
    function onDoc(e) { if (!pop.contains(e.target)) close(); }
    setTimeout(() => document.addEventListener('mousedown', onDoc), 0);

    pop.innerHTML = '<div style="padding:8px;color:var(--fg-dim)">loading...</div>';
    const { live, hist } = await fetchSessions();
    const lines = [];
    lines.push(sectionHeader('Live'));
    if (!live.length) lines.push(emptyRow());
    for (const s of live) lines.push(liveRow(s));
    lines.push(sectionHeader('History'));
    if (!hist.length) lines.push(emptyRow());
    for (const s of hist) lines.push(historyRow(s));
    pop.innerHTML = lines.join('');

    pop.querySelectorAll('.__sess').forEach(el => {
      el.addEventListener('mouseover', () => el.style.background = 'rgba(255,255,255,0.04)');
      el.addEventListener('mouseout',  () => el.style.background = '');
      el.addEventListener('click', () => {
        const id = el.dataset.id;
        history.replaceState(null, '', `?sid=${encodeURIComponent(id)}`);
        window.dispatchEvent(new CustomEvent('inspector:openSession', { detail: { sid: id } }));
        close();
      });
    });
  });
}

function sectionHeader(label) {
  return `<div style="padding:4px 8px;color:var(--fg-faint);text-transform:uppercase;letter-spacing:0.1em;font-size:10px;margin-top:6px">${label}</div>`;
}
function emptyRow() {
  return '<div style="padding:6px 10px;color:var(--fg-dim)">--</div>';
}
function liveRow(s) {
  return `<div class="__sess" data-id="${s.id}" style="padding:6px 10px;cursor:pointer;border-radius:4px">${s.id} <span style="color:var(--fg-dim)">· ${s.model || ''}</span></div>`;
}
function historyRow(s) {
  const dt = new Date(s.mtime * 1000).toISOString().replace('T', ' ').replace(/\..*$/, '');
  const kb = (s.size_bytes / 1024).toFixed(0);
  return `<div class="__sess" data-id="${s.id}" style="padding:6px 10px;cursor:pointer;border-radius:4px">${s.id} <span style="color:var(--fg-dim)">· ${dt} · ${kb} KB</span></div>`;
}

export function flashPill() {
  const p = document.getElementById('sessionPill');
  if (!p) return;
  p.style.transition = 'box-shadow 0.4s';
  p.style.boxShadow = '0 0 0 2px var(--accent)';
  setTimeout(() => { p.style.boxShadow = ''; }, 900);
}
