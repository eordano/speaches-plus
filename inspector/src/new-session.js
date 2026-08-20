import { Transport } from './transport.js';
import { resampleLinear, float32ToInt16, int16ToBase64, base64ToFloat32, escapeHTML } from './util.js';

const SAMPLE_RATE = 24000;
const BUFFER_SIZE = 4096;
const STATUS_MAP = {
  idle:       { label: 'New session',   cls: '',             dot: false },
  connecting: { label: 'connecting...', cls: 'st-active',    dot: true  },
  live:       { label: 'live',          cls: 'st-live',      dot: true  },
  speaking:   { label: 'listening...',  cls: 'st-speaking',  dot: true  },
  processing: { label: 'processing...', cls: 'st-active',    dot: true  },
  error:      { label: 'error',         cls: 'st-error',     dot: false },
};

export function setupNewSession() {
  const btn = document.getElementById('btnNewSession');
  if (!btn) return;

  const params = new URLSearchParams(location.search);
  btn.dataset.model = params.get('client_model') || 'llm-default';
  btn.dataset.voice = params.get('client_voice') || 'af_heart';
  btn.dataset.stt   = params.get('client_stt')   || 'deepdml/faster-whisper-large-v3-turbo-ct2';

  const state = {
    status: 'idle',
    ws: null,
    sessionId: null,
    micStream: null,
    micCtx: null,
    micProcessor: null,
    playCtx: null,
    playScheduledEnd: 0,
    playSources: [],
    lastTranscript: '',
    lastResponse: '',
    responseText: '',
  };

  function setStatus(s) { state.status = s; renderBadge(); }

  function renderBadge() {
    const info = STATUS_MAP[state.status] || STATUS_MAP.idle;
    btn.className = 'btn client-badge ' + info.cls;
    let html = '';
    if (info.dot) html += '<span class="client-dot"></span>';
    if (state.status === 'idle') {
      html += 'New session';
    } else {
      html += info.label;
      if (state.lastTranscript && (state.status === 'processing' || state.status === 'live')) {
        html += ' <span class="client-transcript">' + escapeHTML(state.lastTranscript) + '</span>';
      }
    }
    btn.innerHTML = html;
  }

  async function connect() {
    if (state.ws) return;
    setStatus('connecting');
    try {
      const micId = window.__inspectSelectedMic || localStorage.getItem('inspect.deviceMic') || undefined;
      const audioConstraints = { echoCancellation: true, noiseSuppression: true, autoGainControl: true };
      if (micId) audioConstraints.deviceId = { exact: micId };
      state.micStream = await navigator.mediaDevices.getUserMedia({ audio: audioConstraints });
      if (typeof window.__inspectEnumerateDevices === 'function') window.__inspectEnumerateDevices();
    } catch (err) {
      console.error('[new-session] mic permission denied:', err);
      setStatus('error');
      return;
    }
    try {
      state.micCtx = new AudioContext();
      if (state.micCtx.state === 'suspended') await state.micCtx.resume();
      state.playCtx = new AudioContext({ sampleRate: SAMPLE_RATE });
      if (state.playCtx.state === 'suspended') await state.playCtx.resume();
      state.playScheduledEnd = 0;
    } catch (err) {
      console.error('[new-session] AudioContext creation failed:', err);
      setStatus('error');
      return;
    }
    const source = state.micCtx.createMediaStreamSource(state.micStream);
    const processor = state.micCtx.createScriptProcessor(BUFFER_SIZE, 1, 1);
    const nativeRate = state.micCtx.sampleRate;
    processor.onaudioprocess = (e) => {
      if (!state.ws || state.ws.readyState !== 1) return;
      const raw = e.inputBuffer.getChannelData(0);
      const resampled = resampleLinear(raw, nativeRate, SAMPLE_RATE);
      const b64 = int16ToBase64(float32ToInt16(resampled));
      state.ws.send(JSON.stringify({ type: 'input_audio_buffer.append', audio: b64 }));
    };
    source.connect(processor);
    const muter = state.micCtx.createGain();
    muter.gain.value = 0;
    processor.connect(muter);
    muter.connect(state.micCtx.destination);
    state.micProcessor = processor;

    const model = btn.dataset.model;
    const sttModel = btn.dataset.stt;
    const scheme = location.protocol === 'https:' ? 'wss' : 'ws';
    const url = `${scheme}://${location.host}/v1/realtime?model=${encodeURIComponent(model)}&transcription_model=${encodeURIComponent(sttModel)}`;
    const ws = new WebSocket(url);
    state.ws = ws;
    ws.onopen   = () => setStatus('live');
    ws.onmessage = onMessage;
    ws.onclose  = () => disconnect();
    ws.onerror  = (err) => { console.error('[new-session] WS error:', err); setStatus('error'); disconnect(); };
  }

  function disconnect() {
    stopMic();
    teardownPlayback();
    if (state.ws) { try { state.ws.close(); } catch {} state.ws = null; }
    Transport.setRealtimeWs(null);
    state.sessionId = null;
    state.responseText = '';
    setStatus('idle');
  }

  function onMessage(ev) {
    let msg; try { msg = JSON.parse(ev.data); } catch { return; }
    switch (msg.type) {
      case 'session.created':
        state.sessionId = msg.session.id;
        Transport.setRealtimeWs(state.ws);
        if (typeof window.__inspectPopulateSettings === 'function') {
          window.__inspectPopulateSettings(msg.session);
        }
        state.ws.send(JSON.stringify({
          type: 'session.update',
          session: {
            voice: btn.dataset.voice || 'af_heart',
            instructions: 'Provide very short, concise answers. Keep responses to one or two sentences whenever possible.',
          },
        }));
        window.dispatchEvent(new CustomEvent('inspector:openSession', { detail: { sid: msg.session.id } }));
        break;
      case 'input_audio_buffer.speech_started':
        setStatus('speaking');
        silencePlayback();
        state.responseText = '';
        break;
      case 'input_audio_buffer.speech_stopped':
        setStatus('processing');
        break;
      case 'conversation.item.input_audio_transcription.completed':
        if (msg.transcript && msg.transcript.trim()) {
          state.lastTranscript = msg.transcript.trim();
          renderBadge();
        }
        break;
      case 'response.output_text.delta':
        state.responseText += (msg.delta || '');
        break;
      case 'response.output_audio.delta':
        if (msg.delta) playAudioChunk(msg.delta);
        break;
      case 'response.done':
        if (state.responseText.trim()) {
          state.lastResponse = state.responseText.trim();
          state.responseText = '';
        }
        setStatus('live');
        break;
      case 'error':
        console.error('[new-session] server error:', msg.error);
        setStatus('error');
        break;
    }
  }

  function stopMic() {
    if (state.micProcessor) { state.micProcessor.disconnect(); state.micProcessor = null; }
    if (state.micCtx) { try { state.micCtx.close(); } catch {} state.micCtx = null; }
    if (state.micStream) { state.micStream.getTracks().forEach(t => t.stop()); state.micStream = null; }
  }

  function playAudioChunk(b64) {
    if (!state.playCtx) return;
    const ctx = state.playCtx;
    if (ctx.state === 'suspended') {
      ctx.resume().catch(err => console.error('[new-session] resume failed:', err));
    }
    const samples = base64ToFloat32(b64);
    const buf = ctx.createBuffer(1, samples.length, SAMPLE_RATE);
    buf.getChannelData(0).set(samples);
    const src = ctx.createBufferSource();
    src.buffer = buf;
    src.connect(ctx.destination);
    const now = ctx.currentTime;
    const at = Math.max(now + 0.005, state.playScheduledEnd);
    src.start(at);
    state.playScheduledEnd = at + buf.duration;
    state.playSources.push(src);
    src.onended = () => {
      const i = state.playSources.indexOf(src);
      if (i >= 0) state.playSources.splice(i, 1);
    };
  }

  function silencePlayback() {
    for (const src of state.playSources) { try { src.stop(); } catch {} }
    state.playSources = [];
    state.playScheduledEnd = 0;
  }

  function teardownPlayback() {
    silencePlayback();
    if (state.playCtx) { try { state.playCtx.close(); } catch {} state.playCtx = null; }
  }

  btn.addEventListener('click', () => {
    if (state.status === 'idle' || state.status === 'error') connect();
    else disconnect();
  });

  renderBadge();
}
