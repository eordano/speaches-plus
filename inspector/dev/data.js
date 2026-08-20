(function () {
  const MODEL  = 'gpt-4o-realtime-preview-2024-10-01';
  const VOICE  = 'alloy';
  const TTS_MODEL = 'speaches.kokoro';

  let _seq = 0;

  function mk(lane, kind, t_ms, payload = {}, corr = null) {
    return {
      session_id: 'sess_7f3a2b9c',
      seq: _seq++,
      ts_mono_ns: Math.round(t_ms * 1e6),
      ts_wall: 1713720600 + t_ms / 1000,
      lane,
      kind,
      corr: corr || {},
      span_id: Math.random().toString(16).slice(2, 18),
      payload,
      t: t_ms,
    };
  }

  function audioWindow(t0_ms, t1_ms, baseRms, peak, corr) {
    const out = [];
    for (let t = t0_ms; t < t1_ms; t += 40) {
      const phase = (t - t0_ms) / (t1_ms - t0_ms);
      const env = phase < 0.15 ? phase / 0.15
                : phase > 0.85 ? (1 - phase) / 0.15
                : 1;
      const rms = baseRms + (peak - baseRms) * env * (0.7 + Math.random() * 0.3);
      out.push(mk('audio_level', 'sample', t,
        { rms: +rms.toFixed(4), window_ms: 40 }, corr));
    }
    return out;
  }
  function audioSilence(t0_ms, t1_ms) {
    const out = [];
    for (let t = t0_ms; t < t1_ms; t += 40) {
      const rms = 0.002 + Math.random() * 0.006;
      out.push(mk('audio_level', 'sample', t, { rms: +rms.toFixed(4), window_ms: 40 }));
    }
    return out;
  }

  function generateClean() {
    _seq = 0;
    const E = [];
    const turn1 = { turn_id: 'turn_01' };
    const turn2 = { turn_id: 'turn_02' };
    const item1 = { ...turn1, item_id: 'item_01HN3Z9Q4K' };
    const item1a = { ...turn1, item_id: 'item_01HN3ZA2RT' };
    const resp1 = { ...turn1, response_id: 'resp_7a3f2e1c' };
    const item2 = { ...turn2, item_id: 'item_01HN3ZC1XY' };
    const item2a = { ...turn2, item_id: 'item_01HN3ZD7PQ' };
    const resp2 = { ...turn2, response_id: 'resp_8b4f3e2d' };

    E.push(...audioSilence(0, 200));
    E.push(mk('turn', 'turn_start', 0, { turn_id: 'turn_01', role: 'user' }, turn1));
    E.push(mk('wire', 'in', 40, { event_type: 'session.update', bytes: 412 }));
    E.push(mk('wire', 'out', 52, { event_type: 'session.created', bytes: 288 }));

    E.push(...audioSilence(200, 360));
    E.push(mk('vad', 'pending_start', 180, { prob: 0.62, rms: 0.074 }, item1));
    E.push(mk('vad', 'confirmed_start', 380, { prob: 0.91, rms: 0.12 }, item1));
    E.push(...audioWindow(360, 2080, 0.03, 0.14, item1));

    E.push(mk('stt', 'partial', 720, { text: 'hey' }, item1));
    E.push(mk('stt', 'partial', 1040, { text: 'hey so' }, item1));
    E.push(mk('stt', 'partial', 1360, { text: 'hey so what' }, item1));
    E.push(mk('stt', 'partial', 1620, { text: 'hey so what is' }, item1));
    E.push(mk('stt', 'partial', 1880, { text: 'hey so what is the weather' }, item1));

    E.push(mk('vad', 'stopped', 2080, { rms: 0.009, speech_ms: 1700 }, item1));
    E.push(mk('turn', 'user_committed', 2100, { item_id: item1.item_id }, item1));
    E.push(mk('stt', 'final', 2160, {
      text: "hey so what's the weather looking like in san francisco tomorrow",
      audio_start_ms: 380, audio_end_ms: 2080,
      words: 11, avg_no_speech_prob: 0.08,
    }, item1));
    E.push(...audioSilence(2080, 2400));
    E.push(mk('wire', 'out', 2200, { event_type: 'conversation.item.added', bytes: 612 }));

    E.push(mk('response', 'plan_start', 2220, { trigger: 'vad_commit' }, resp1));
    E.push(mk('turn', 'turn_start', 2220, { turn_id: 'turn_01', role: 'assistant' }, { ...turn1, response_id: resp1.response_id }));

    E.push(mk('llm', 'request', 2240, { model: MODEL, tok_in: 312 }, resp1));
    E.push(mk('wire', 'in', 2244, { event_type: 'response.create', bytes: 96 }));
    E.push(mk('llm', 'first_token', 2680, { elapsed_ms: 440, ttft_ms: 440 }, resp1));

    const reply1 = "Tomorrow in San Francisco, expect partly cloudy skies. Highs near 64 degrees, with light winds from the west.";
    let cursor = 0;
    let tok = 0;
    let tChunk = 2700;
    while (cursor < reply1.length) {
      const n = 1 + Math.floor(Math.random() * 4);
      const slice = reply1.slice(cursor, cursor + n);
      cursor += n;
      tok += 1;
      tChunk += 12 + Math.random() * 14;
      E.push(mk('llm', 'chunk', tChunk, {
        delta: slice, text_so_far_len: cursor, tok_out: tok,
      }, resp1));
    }

    const p1 = { ...resp1, phrase_id: `${resp1.response_id}:0` };
    const p2 = { ...resp1, phrase_id: `${resp1.response_id}:1` };
    E.push(mk('response', 'phrase_boundary', 2920, {
      phrase_id: p1.phrase_id, text: "Tomorrow in San Francisco, expect partly cloudy skies.", reason: 'sentence_end',
    }, p1));
    E.push(mk('response', 'phrase_boundary', 3260, {
      phrase_id: p2.phrase_id, text: "Highs near 64 degrees, with light winds from the west.", reason: 'sentence_end',
    }, p2));

    E.push(mk('tts_req', 'phrase_sent', 2940, {
      text: "Tomorrow in San Francisco, expect partly cloudy skies.", voice: VOICE, model: TTS_MODEL,
    }, p1));
    E.push(mk('tts_chunk', 'first_chunk', 3112, {
      chunk_idx: 0, pcm_samples: 3840, ms_audio: 160, cumulative_ms: 160,
    }, p1));
    for (let i = 1; i < 18; i++) {
      const t = 3112 + i * 62;
      E.push(mk('tts_chunk', 'chunk', t, {
        chunk_idx: i, pcm_samples: 3840, ms_audio: 160, cumulative_ms: (i + 1) * 160,
      }, p1));
    }
    E.push(mk('tts_req', 'phrase_done', 4240, { ms_audio: 2880 }, p1));

    E.push(mk('tts_req', 'phrase_sent', 3320, {
      text: "Highs near 64 degrees, with light winds from the west.", voice: VOICE, model: TTS_MODEL,
    }, p2));
    E.push(mk('tts_chunk', 'first_chunk', 3504, {
      chunk_idx: 0, pcm_samples: 3840, ms_audio: 160, cumulative_ms: 160,
    }, p2));
    for (let i = 1; i < 16; i++) {
      const t = 3504 + i * 62;
      E.push(mk('tts_chunk', 'chunk', t, {
        chunk_idx: i, pcm_samples: 3840, ms_audio: 160, cumulative_ms: (i + 1) * 160,
      }, p2));
    }
    E.push(mk('tts_req', 'phrase_done', 4510, { ms_audio: 2560 }, p2));

    E.push(mk('llm', 'done', 3420, { tok_in: 312, tok_out: tok, elapsed_ms: 1180 }, resp1));
    E.push(mk('response', 'done', 4540, { phrases: 2, total_audio_ms: 5440 }, resp1));
    E.push(mk('wire', 'out', 4620, { event_type: 'response.done', bytes: 540 }));
    E.push(mk('turn', 'turn_end', 4660, { turn_id: 'turn_01' }, { ...turn1, response_id: resp1.response_id }));

    E.push(...audioSilence(4660, 6200));

    E.push(mk('turn', 'turn_start', 6200, { turn_id: 'turn_02', role: 'user' }, turn2));
    E.push(mk('vad', 'pending_start', 6240, { prob: 0.66, rms: 0.08 }, item2));
    E.push(mk('vad', 'confirmed_start', 6420, { prob: 0.92, rms: 0.13 }, item2));
    E.push(...audioWindow(6220, 7700, 0.035, 0.15, item2));
    E.push(mk('stt', 'partial', 6780, { text: 'and' }, item2));
    E.push(mk('stt', 'partial', 7120, { text: 'and what about' }, item2));
    E.push(mk('stt', 'partial', 7480, { text: 'and what about oakland' }, item2));
    E.push(mk('vad', 'stopped', 7700, { rms: 0.01, speech_ms: 1280 }, item2));
    E.push(mk('turn', 'user_committed', 7720, { item_id: item2.item_id }, item2));
    E.push(mk('stt', 'final', 7780, {
      text: "and what about oakland",
      audio_start_ms: 6420, audio_end_ms: 7700, words: 4, avg_no_speech_prob: 0.05,
    }, item2));
    E.push(...audioSilence(7700, 8100));

    E.push(mk('response', 'plan_start', 7820, { trigger: 'vad_commit' }, resp2));
    E.push(mk('turn', 'turn_start', 7820, { turn_id: 'turn_02', role: 'assistant' }, { ...turn2, response_id: resp2.response_id }));
    E.push(mk('llm', 'request', 7840, { model: MODEL, tok_in: 356 }, resp2));
    E.push(mk('llm', 'first_token', 8180, { elapsed_ms: 340, ttft_ms: 340 }, resp2));

    const reply2 = "Oakland looks similar, maybe a touch warmer near the bay.";
    cursor = 0; tok = 0; tChunk = 8200;
    while (cursor < reply2.length) {
      const n = 1 + Math.floor(Math.random() * 4);
      const slice = reply2.slice(cursor, cursor + n);
      cursor += n; tok += 1; tChunk += 12 + Math.random() * 12;
      E.push(mk('llm', 'chunk', tChunk, { delta: slice, text_so_far_len: cursor, tok_out: tok }, resp2));
    }
    const p3 = { ...resp2, phrase_id: `${resp2.response_id}:0` };
    E.push(mk('response', 'phrase_boundary', 8400, {
      phrase_id: p3.phrase_id, text: reply2, reason: 'sentence_end',
    }, p3));
    E.push(mk('tts_req', 'phrase_sent', 8420, { text: reply2, voice: VOICE, model: TTS_MODEL }, p3));
    E.push(mk('tts_chunk', 'first_chunk', 8588, { chunk_idx: 0, pcm_samples: 3840, ms_audio: 160 }, p3));
    for (let i = 1; i < 14; i++) {
      E.push(mk('tts_chunk', 'chunk', 8588 + i * 62, {
        chunk_idx: i, pcm_samples: 3840, ms_audio: 160, cumulative_ms: (i + 1) * 160,
      }, p3));
    }
    E.push(mk('tts_req', 'phrase_done', 9480, { ms_audio: 2240 }, p3));
    E.push(mk('llm', 'done', 8600, { tok_in: 356, tok_out: tok, elapsed_ms: 760 }, resp2));
    E.push(mk('response', 'done', 9500, { phrases: 1, total_audio_ms: 2240 }, resp2));
    E.push(mk('turn', 'turn_end', 9540, { turn_id: 'turn_02' }, { ...turn2, response_id: resp2.response_id }));

    E.sort((a, b) => a.t - b.t);
    E.forEach((e, i) => (e.seq = i));
    return E;
  }

  function generateProblem() {
    _seq = 0;
    const E = [];
    const turn1 = { turn_id: 'turn_01' };
    const item1 = { ...turn1, item_id: 'item_01HN4X2A9K' };
    const resp1 = { ...turn1, response_id: 'resp_fa12c9e3' };
    const item1b = { ...turn1, item_id: 'item_01HN4XB7RR' }; // barge-in attempt

    E.push(...audioSilence(0, 140));
    E.push(mk('turn', 'turn_start', 0, { turn_id: 'turn_01', role: 'user' }, turn1));
    E.push(mk('vad', 'pending_start', 140, { prob: 0.58, rms: 0.071 }, item1));
    E.push(mk('vad', 'confirmed_start', 340, { prob: 0.89, rms: 0.11 }, item1));
    E.push(...audioWindow(140, 1840, 0.03, 0.13, item1));
    E.push(mk('stt', 'partial', 720, { text: 'can you' }, item1));
    E.push(mk('stt', 'partial', 1100, { text: 'can you tell me' }, item1));
    E.push(mk('stt', 'partial', 1480, { text: 'can you tell me a long story' }, item1));
    E.push(mk('vad', 'stopped', 1840, { rms: 0.008, speech_ms: 1500 }, item1));
    E.push(mk('turn', 'user_committed', 1850, { item_id: item1.item_id }, item1));
    E.push(mk('stt', 'final', 1930, {
      text: "can you tell me a long story about a lighthouse",
      audio_start_ms: 340, audio_end_ms: 1840, words: 9, avg_no_speech_prob: 0.06,
    }, item1));
    E.push(...audioSilence(1840, 2200));

    E.push(mk('response', 'plan_start', 1970, { trigger: 'vad_commit' }, resp1));
    E.push(mk('turn', 'turn_start', 1980, { turn_id: 'turn_01', role: 'assistant' }, { ...turn1, response_id: resp1.response_id }));
    E.push(mk('llm', 'request', 2000, { model: MODEL, tok_in: 286 }, resp1));
    E.push(mk('llm', 'first_token', 2980, { elapsed_ms: 980, ttft_ms: 980 }, resp1));

    const reply1 = "Once, on a rocky headland beyond the fog, there stood a lighthouse.";
    let cursor = 0, tok = 0, tChunk = 3000;
    while (cursor < reply1.length) {
      const n = 1 + Math.floor(Math.random() * 4);
      E.push(mk('llm', 'chunk', tChunk, {
        delta: reply1.slice(cursor, cursor + n), text_so_far_len: cursor + n, tok_out: ++tok,
      }, resp1));
      cursor += n;
      tChunk += 10 + Math.random() * 10;
    }
    const p1 = { ...resp1, phrase_id: `${resp1.response_id}:0` };
    E.push(mk('response', 'phrase_boundary', 3060, {
      phrase_id: p1.phrase_id, text: reply1, reason: 'sentence_end',
    }, p1));
    E.push(mk('tts_req', 'phrase_sent', 3080, { text: reply1, voice: VOICE, model: TTS_MODEL }, p1));
    E.push(mk('tts_chunk', 'first_chunk', 3240, { chunk_idx: 0, pcm_samples: 3840, ms_audio: 160 }, p1));
    for (let i = 1; i < 20; i++) {
      E.push(mk('tts_chunk', 'chunk', 3240 + i * 62, {
        chunk_idx: i, pcm_samples: 3840, ms_audio: 160, cumulative_ms: (i + 1) * 160,
      }, p1));
    }
    E.push(mk('tts_req', 'phrase_done', 4480, { ms_audio: 3200 }, p1));

    const p2 = { ...resp1, phrase_id: `${resp1.response_id}:1` };
    const reply2 = "Its keeper, a woman named Ines, kept the lamp burning through every gale.";
    cursor = 0; tChunk = 4400;
    while (cursor < reply2.length) {
      const n = 1 + Math.floor(Math.random() * 4);
      E.push(mk('llm', 'chunk', tChunk, {
        delta: reply2.slice(cursor, cursor + n), text_so_far_len: 67 + cursor + n, tok_out: ++tok,
      }, resp1));
      cursor += n;
      tChunk += 12 + Math.random() * 10;
    }
    E.push(mk('response', 'phrase_boundary', 4520, {
      phrase_id: p2.phrase_id, text: reply2, reason: 'sentence_end',
    }, p2));
    E.push(mk('tts_req', 'phrase_sent', 4540, { text: reply2, voice: VOICE, model: TTS_MODEL }, p2));

    E.push(...audioWindow(5100, 5460, 0.02, 0.09, item1b));
    E.push(mk('vad', 'pending_start', 5220, {
      prob: 0.68, rms: 0.082,
      note: 'min_speech_duration_ms=220 not met -- Silero reset before confirm',
    }, item1b));
    E.push(mk('turn', 'bargein_missed', 5460, {
      reason: 'below_min_speech', duration_ms: 240,
    }, item1b));
    E.push(...audioSilence(5460, 5940));

    E.push(mk('error', 'raised', 5940, {
      lane: 'tts_req',
      error: 'UpstreamClosedError: tts worker closed connection mid-phrase',
      severity: 'error',
    }, p2));
    E.push(mk('tts_req', 'error', 5940, {
      error: 'UpstreamClosedError: tts worker closed connection mid-phrase',
      voice: VOICE, model: TTS_MODEL,
    }, p2));
    E.push(mk('wire', 'dropped', 5944, {
      event_type: 'response.audio.delta', dropped_count: 2, reason: 'queue_full',
    }));

    for (let i = 0; i < 10; i++) {
      E.push(mk('llm', 'chunk', 5200 + i * 42, {
        delta: '.', text_so_far_len: 140 + i, tok_out: tok + i,
      }, resp1));
    }
    E.push(mk('llm', 'done', 6080, { tok_in: 286, tok_out: tok + 10, elapsed_ms: 4080 }, resp1));
    E.push(mk('response', 'done', 6100, {
      phrases: 2, completed_phrases: 1, failed_phrases: 1, total_audio_ms: 3200,
    }, resp1));

    const item2 = { turn_id: 'turn_02', item_id: 'item_01HN4XD1XY' };
    const turn2 = { turn_id: 'turn_02' };
    E.push(mk('turn', 'turn_start', 6400, { turn_id: 'turn_02', role: 'user' }, turn2));
    E.push(mk('vad', 'pending_start', 6420, { prob: 0.74, rms: 0.094 }, item2));
    E.push(mk('vad', 'confirmed_start', 6600, { prob: 0.93, rms: 0.13 }, item2));
    E.push(...audioWindow(6420, 7120, 0.04, 0.16, item2));
    E.push(mk('stt', 'partial', 6920, { text: 'stop' }, item2));
    E.push(mk('vad', 'stopped', 7120, { rms: 0.009, speech_ms: 520 }, item2));
    E.push(mk('turn', 'user_committed', 7140, { item_id: item2.item_id }, item2));
    E.push(mk('stt', 'final', 7200, {
      text: 'stop', audio_start_ms: 6600, audio_end_ms: 7120,
      words: 1, avg_no_speech_prob: 0.12,
    }, item2));
    E.push(mk('wire', 'out', 7240, { event_type: 'response.cancelled', bytes: 188 }));

    E.sort((a, b) => a.t - b.t);
    E.forEach((e, i) => (e.seq = i));
    return E;
  }

  function liveTick(currentEvents, sessionStartWall) {
    const last = currentEvents[currentEvents.length - 1];
    const lastT = last ? last.t : 0;
    const lane = pick(['audio_level', 'wire', 'llm', 'audio_level', 'tts_chunk', 'audio_level']);
    let ev;
    if (lane === 'audio_level') {
      ev = mk('audio_level', 'sample', lastT + 40,
        { rms: +(0.005 + Math.random() * 0.04).toFixed(4), window_ms: 40 });
    } else if (lane === 'wire') {
      ev = mk('wire', 'out', lastT + 80 + Math.random() * 80,
        { event_type: pick(['session.update', 'rate_limits.updated', 'conversation.item.updated']),
          bytes: 120 + Math.floor(Math.random() * 700) });
    } else if (lane === 'llm') {
      ev = mk('llm', 'chunk', lastT + 30 + Math.random() * 30,
        { delta: pick(['the ', 'a ', 'is ', '.']), text_so_far_len: Math.floor(Math.random() * 300) });
    } else {
      ev = mk('tts_chunk', 'chunk', lastT + 60,
        { chunk_idx: Math.floor(Math.random() * 40), pcm_samples: 3840, ms_audio: 160 });
    }
    ev.seq = currentEvents.length;
    return ev;
  }
  function pick(a) { return a[Math.floor(Math.random() * a.length)]; }

  const LANES = [
    { id: 'audio_level', name: 'Audio',        hint: 'PCM RMS · 40ms/window', cssVar: '--lane-audio' },
    { id: 'vad',         name: 'VAD',          hint: 'Silero',                cssVar: '--lane-vad' },
    { id: 'stt',         name: 'STT',          hint: 'whisper-large-v3',      cssVar: '--lane-stt' },
    { id: 'turn',        name: 'Turn',         hint: 'turn-control · barge-in', cssVar: '--lane-turn' },
    { id: 'llm',         name: 'LLM',          hint: 'gpt-4o-realtime',       cssVar: '--lane-llm' },
    { id: 'response',    name: 'Response',     hint: 'plan · phrase split',   cssVar: '--lane-response' },
    { id: 'tts_req',     name: 'TTS phrases',  hint: `${VOICE} · ${TTS_MODEL.replace('speaches.', '')}`, cssVar: '--lane-tts-req' },
    { id: 'tts_chunk',   name: 'TTS chunks',   hint: '24 kHz PCM · 160 ms',   cssVar: '--lane-tts-chunk' },
    { id: 'wire',        name: 'Wire',         hint: 'protocol only',         cssVar: '--lane-wire' },
  ];

  const PALETTES = {
    warm: {
      audio_level: '#6E7C7F', vad: '#7A92A8', stt: '#7F9B7F',
      turn: '#9F7E9B', llm: '#A8906A',
      response: '#C89B6A', tts_req: '#C4A45A', tts_chunk: '#B88B5A',
      wire: '#9B9590', error: '#B88080',
    },
    semantic: {
      audio_level: '#6BBED3', vad: '#6FA8DC', stt: '#6BBE7F',
      turn: '#C77BBA', llm: '#C8A2E8',
      response: '#E8A96B', tts_req: '#E8A96B', tts_chunk: '#E8C76B',
      wire: '#9B9590', error: '#E87878',
    },
    mono: {
      audio_level: '#5A564F', vad: '#8D7A5A', stt: '#A8906A',
      turn: '#8E7958', llm: '#C8B08E',
      response: '#DEC49B', tts_req: '#DEC49B', tts_chunk: '#BC9C6E',
      wire: '#726B62', error: '#B88080',
    },
  };

  window.INSPECTOR_DATA = {
    LANES, PALETTES, generateClean, generateProblem, liveTick,
    MODEL, VOICE,
  };
})();
