import { Transport } from './transport.js';

const $ = (s) => document.querySelector(s);

export function setupSettings() {
  const btn      = $('#btnSettings');
  const panel    = $('#settingsPanel');
  const closeBtn = $('#settingsClose');
  const footer   = $('#ssetFooter');
  const footerText = $('#ssetFooterText');
  const audioMode = $('#ssetAudio');

  const catalog = { stt: [], tts: [], voices: {} };

  function flashFooter(msg) {
    footer.classList.add('sent');
    footerText.textContent = msg || 'session.update sent';
    clearTimeout(footer._timer);
    footer._timer = setTimeout(() => footer.classList.remove('sent'), 1200);
  }

  function send(partial) {
    if (Transport.sendSessionUpdate(partial)) flashFooter();
  }

  function toggle() {
    const open = document.body.classList.toggle('settings-open');
    btn.setAttribute('aria-pressed', open ? 'true' : 'false');
    window.dispatchEvent(new Event('inspector:layoutChanged'));
  }
  btn.addEventListener('click', toggle);
  closeBtn.addEventListener('click', () => {
    document.body.classList.remove('settings-open');
    btn.setAttribute('aria-pressed', 'false');
    window.dispatchEvent(new Event('inspector:layoutChanged'));
  });

  // Audio mode (STT vs Direct WAV).
  document.querySelectorAll('input[name="audio_mode"]').forEach(r => {
    r.addEventListener('change', () => {
      audioMode.dataset.mode = r.value;
      send({ audio_direct_to_llm: r.value === 'audio_direct' });
    });
  });
  let _adModelTimer = null;
  $('#audioDirectModel').addEventListener('input', (e) => {
    clearTimeout(_adModelTimer);
    _adModelTimer = setTimeout(() => send({ audio_direct_model: e.target.value }), 600);
  });
  let _promptTimer = null;
  $('#audioDirectPrompt').addEventListener('input', (e) => {
    clearTimeout(_promptTimer);
    _promptTimer = setTimeout(() => send({ audio_direct_prompt: e.target.value }), 600);
  });

  // VAD knobs.
  $('#vadThreshold').addEventListener('input', (e) => {
    const v = parseFloat(e.target.value);
    $('#vadThresholdVal').textContent = v.toFixed(2);
    send({ turn_detection: { threshold: v } });
  });
  numField('#vadMinSpeech', (n) => send({ turn_detection: { min_speech_duration_ms: n } }));
  numField('#vadSilence',   (n) => send({ turn_detection: { silence_duration_ms: n } }));
  numField('#vadBargeIn',   (n) => send({ turn_detection: { barge_in_delay_ms: n } }));

  // Models.
  $('#modelStt').addEventListener('change', (e) => send({ input_audio_transcription: { model: e.target.value } }));
  $('#modelTts').addEventListener('change', (e) => {
    send({ speech_model: e.target.value });
    updateVoiceOptions();
  });
  $('#modelVoice').addEventListener('change', (e) => send({ voice: e.target.value }));
  let _llmTimer = null;
  $('#modelLlm').addEventListener('input', (e) => {
    clearTimeout(_llmTimer);
    _llmTimer = setTimeout(() => send({ model: e.target.value }), 600);
  });
  let _instrTimer = null;
  $('#instructions').addEventListener('input', (e) => {
    clearTimeout(_instrTimer);
    _instrTimer = setTimeout(() => send({ instructions: e.target.value }), 600);
  });

  function numField(sel, onChange) {
    let t = null;
    $(sel).addEventListener('input', (e) => {
      clearTimeout(t);
      t = setTimeout(() => onChange(parseInt(e.target.value, 10) || 0), 400);
    });
  }

  // Pre-populate from session snapshot (called by new-session.js when the
  // server emits session.created).
  function populateSettings(session) {
    if (!session) return;
    const td = session.turn_detection || {};
    const iat = session.input_audio_transcription || {};
    $('#vadThreshold').value = td.threshold ?? 0.8;
    $('#vadThresholdVal').textContent = (td.threshold ?? 0.8).toFixed(2);
    $('#vadMinSpeech').value = td.min_speech_duration_ms ?? 120;
    $('#vadSilence').value   = td.silence_duration_ms ?? 350;
    $('#vadBargeIn').value   = td.barge_in_delay_ms ?? 400;
    $('#modelLlm').value     = session.model || '';
    setSelectValue($('#modelStt'), iat.model);
    setSelectValue($('#modelTts'), session.speech_model);
    updateVoiceOptions(session.voice);
    setSelectValue($('#modelVoice'), session.voice);
    $('#instructions').value = session.instructions || '';
    const mode = session.audio_direct_to_llm ? 'audio_direct' : 'stt';
    document.querySelector(`input[name="audio_mode"][value="${mode}"]`).checked = true;
    audioMode.dataset.mode = mode;
    if (session.audio_direct_model)  $('#audioDirectModel').value  = session.audio_direct_model;
    if (session.audio_direct_prompt) $('#audioDirectPrompt').value = session.audio_direct_prompt;
  }
  window.__inspectPopulateSettings = populateSettings;

  // Catalog.
  fetchModelCatalog();
  enumerateDevices();
  ensureDevicePermission();
  navigator.mediaDevices.addEventListener('devicechange', enumerateDevices);
  window.__inspectEnumerateDevices = enumerateDevices;

  async function fetchModelCatalog() {
    try {
      const r = await fetch('/v1/models');
      if (!r.ok) return;
      const data = await r.json();
      for (const m of data.data || []) {
        if (m.task === 'automatic-speech-recognition') catalog.stt.push(m.id);
        if (m.task === 'text-to-speech') {
          catalog.tts.push(m.id);
          const voices = (m.voices || []).map(v => typeof v === 'object' ? v.name : String(v));
          if (voices.length) catalog.voices[m.id] = voices;
        }
      }
      populateSelect('#modelStt', catalog.stt);
      populateSelect('#modelTts', catalog.tts);
      $('#modelTts').addEventListener('change', () => updateVoiceOptions());
      updateVoiceOptions();
    } catch {}
  }
  function populateSelect(sel, items, current) {
    const el = typeof sel === 'string' ? $(sel) : sel;
    el.innerHTML = '';
    for (const id of items) {
      const opt = document.createElement('option');
      opt.value = id; opt.textContent = id;
      if (id === current) opt.selected = true;
      el.appendChild(opt);
    }
  }
  function updateVoiceOptions(currentVoice) {
    const ttsModel = $('#modelTts').value;
    const voices = catalog.voices[ttsModel] || [];
    const all = voices.length ? voices : ['alloy', 'echo', 'fable', 'onyx', 'nova', 'shimmer'];
    populateSelect('#modelVoice', all, currentVoice);
  }
  function setSelectValue(el, val) {
    if (!val) return;
    if (![...el.options].some(o => o.value === val)) {
      const opt = document.createElement('option');
      opt.value = val; opt.textContent = val;
      el.insertBefore(opt, el.firstChild);
    }
    el.value = val;
  }

  async function enumerateDevices() {
    try {
      const devices = await navigator.mediaDevices.enumerateDevices();
      const micSel = $('#deviceMic');
      const spkSel = $('#deviceSpk');
      micSel.innerHTML = ''; spkSel.innerHTML = '';
      for (const d of devices) {
        const opt = document.createElement('option');
        opt.value = d.deviceId;
        if (d.kind === 'audioinput') {
          opt.textContent = d.label || `Microphone ${micSel.options.length + 1}`;
          micSel.appendChild(opt);
        } else if (d.kind === 'audiooutput') {
          opt.textContent = d.label || `Speaker ${spkSel.options.length + 1}`;
          spkSel.appendChild(opt);
        }
      }
      const savedMic = localStorage.getItem('inspect.deviceMic');
      const savedSpk = localStorage.getItem('inspect.deviceSpk');
      if (savedMic && [...micSel.options].some(o => o.value === savedMic)) micSel.value = savedMic;
      if (savedSpk && [...spkSel.options].some(o => o.value === savedSpk)) spkSel.value = savedSpk;
    } catch (err) {
      console.warn('[settings] enumerate failed:', err);
    }
  }

  $('#deviceMic').addEventListener('change', (e) => {
    localStorage.setItem('inspect.deviceMic', e.target.value);
    window.__inspectSelectedMic = e.target.value;
  });
  $('#deviceSpk').addEventListener('change', (e) => {
    localStorage.setItem('inspect.deviceSpk', e.target.value);
    window.__inspectSelectedSpk = e.target.value;
  });

  async function ensureDevicePermission() {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      stream.getTracks().forEach(t => t.stop());
    } catch (err) {
      console.warn('[settings] mic permission denied:', err);
    }
    enumerateDevices();
  }
}
