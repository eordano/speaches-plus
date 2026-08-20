const $ = (s) => document.querySelector(s);

export function setupExporters({ store }) {
  $('#btnExport').addEventListener('click', async () => {
    const btn = $('#btnExport');
    const prev = btn.textContent;
    btn.textContent = '⤓ ...';
    try {
      const sid = store.state.sessionId;
      if (sid) {
        const res = await fetch(`/v1/inspect/sessions/history/${encodeURIComponent(sid)}`, { cache: 'no-store' });
        if (res.ok) {
          const blob = await res.blob();
          downloadBlob(URL.createObjectURL(blob), `${sid}.ndjson`);
          btn.textContent = '⤓ done';
          setTimeout(() => btn.textContent = prev, 900);
          return;
        }
      }
      // Fallback: serialize the in-memory event stream.
      const body = store.state.events.map(e => JSON.stringify(stripT(e))).join('\n') + '\n';
      const blob = new Blob([body], { type: 'application/x-ndjson' });
      const name = sid ? `${sid}.ndjson` : `inspector-${Date.now()}.ndjson`;
      downloadBlob(URL.createObjectURL(blob), name);
      btn.textContent = '⤓ done';
    } catch (err) {
      console.error('[export] ndjson failed:', err);
      btn.textContent = '⤓ fail';
    } finally {
      setTimeout(() => btn.textContent = prev, 900);
    }
  });

  $('#btnExportAudio').addEventListener('click', async () => {
    const sid = store.state.sessionId;
    if (!sid) return;
    const btn = $('#btnExportAudio');
    const prev = btn.textContent;
    btn.textContent = '⤓ ...';
    try {
      for (const ch of ['mic_in', 'tts_out']) {
        const res = await fetch(`/v1/inspect/sessions/${encodeURIComponent(sid)}/audio?channel=${ch}&from_ms=0&to_ms=0`, { cache: 'no-store' });
        if (!res.ok) continue;
        const blob = await res.blob();
        if (blob.size <= 44) continue;
        downloadBlob(URL.createObjectURL(blob), `${sid}.${ch}.wav`);
      }
      btn.textContent = '⤓ done';
    } catch (err) {
      console.error('[export] audio failed:', err);
      btn.textContent = '⤓ fail';
    } finally {
      setTimeout(() => btn.textContent = prev, 1200);
    }
  });
}

function stripT(e) { const { t, ...rest } = e; return rest; }
function downloadBlob(url, filename) {
  const a = document.createElement('a');
  a.href = url; a.download = filename;
  document.body.appendChild(a);
  a.click();
  setTimeout(() => {
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  }, 0);
}
