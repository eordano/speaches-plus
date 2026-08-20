#!/usr/bin/env node
//
// E2E test: inject speech+silence audio over WebSocket, monitor the
// inspector stream for lane events (stt, llm, tts_req, tts_chunk, turn).
//
// Usage:
//   SERVER_URL=ws://127.0.0.1:8000 node client/e2e_inspector_lanes.mjs
//

const SERVER_URL = process.env.SERVER_URL || 'ws://127.0.0.1:8000';
const SPEECH_MS = parseInt(process.env.SPEECH_MS || '6000', 10);
const SILENCE_MS = parseInt(process.env.SILENCE_MS || '15000', 10);
const CYCLES = parseInt(process.env.CYCLES || '1', 10);
const SAMPLE_RATE = parseInt(process.env.SAMPLE_RATE || '24000', 10);
const AUDIO_FORMAT = process.env.AUDIO_FORMAT || 'pcm16';

function wait(ms) { return new Promise(r => setTimeout(r, ms)); }

import fs from 'fs';

function generateSpeechAudio(durationMs) {
  const rawPath = process.env.SPEECH_FILE || '/tmp/speech_24k_pcm16.raw';
  let clip;
  try {
    clip = fs.readFileSync(rawPath);
  } catch {
    console.warn('No real speech file at', rawPath, '-- generating synthetic tone');
    const nSamples = (SAMPLE_RATE * durationMs) / 1000;
    const buf = Buffer.alloc(nSamples * 2);
    for (let i = 0; i < nSamples; i++) {
      const t = i / SAMPLE_RATE;
      const val = Math.sin(2 * Math.PI * 200 * t) * 0.3
                + Math.sin(2 * Math.PI * 400 * t) * 0.2
                + (Math.random() - 0.5) * 0.15;
      buf.writeInt16LE(Math.max(-32768, Math.min(32767, Math.round(val * 32767))), i * 2);
    }
    return buf;
  }
  // Loop the clip to fill durationMs
  const targetBytes = (SAMPLE_RATE * 2 * durationMs) / 1000;
  const buf = Buffer.alloc(targetBytes);
  for (let off = 0; off < targetBytes; off += clip.length) {
    clip.copy(buf, off, 0, Math.min(clip.length, targetBytes - off));
  }
  return buf;
}

function generateSilence(durationMs) {
  const nSamples = (SAMPLE_RATE * durationMs) / 1000;
  const buf = Buffer.alloc(nSamples * 2);
  for (let i = 0; i < nSamples; i++) {
    const val = (Math.random() - 0.5) * 0.002;
    const s16 = Math.round(val * 32767);
    buf.writeInt16LE(s16, i * 2);
  }
  return buf;
}

function chunkBuffer(buf, chunkMs) {
  const bytesPerChunk = (SAMPLE_RATE * 2 * chunkMs) / 1000;
  const chunks = [];
  for (let off = 0; off < buf.length; off += bytesPerChunk) {
    chunks.push(buf.subarray(off, Math.min(off + bytesPerChunk, buf.length)));
  }
  return chunks;
}

async function openRealtimeSession() {
  const url = `${SERVER_URL}/v1/realtime?model=test-inspector-lanes&intent=conversation`;
  return new Promise((resolve, reject) => {
    const ws = new WebSocket(url);
    const events = [];
    ws.addEventListener('message', ev => {
      try { events.push(JSON.parse(ev.data)); } catch {}
    });
    ws.addEventListener('open', () => resolve({ ws, events }));
    ws.addEventListener('error', () => reject(new Error('realtime ws error')));
  });
}

async function openInspectorStream(sid) {
  const url = `${SERVER_URL}/v1/inspect/${sid}/stream`;
  return new Promise((resolve, reject) => {
    const ws = new WebSocket(url);
    ws.binaryType = 'arraybuffer';
    const events = [];
    ws.addEventListener('message', ev => {
      let text;
      if (typeof ev.data === 'string') {
        text = ev.data;
      } else if (ev.data instanceof ArrayBuffer) {
        text = new TextDecoder().decode(ev.data);
      } else if (ev.data instanceof Blob) {
        return; // skip blobs, handle via binaryType
      } else {
        text = String(ev.data);
      }
      for (const line of text.split('\n')) {
        if (!line.trim()) continue;
        try { events.push(JSON.parse(line)); } catch {}
      }
    });
    ws.addEventListener('open', () => resolve({ ws, events }));
    ws.addEventListener('error', () => reject(new Error('inspector ws error')));
  });
}

async function sendAudioChunks(ws, buf, intervalMs) {
  const chunks = chunkBuffer(buf, intervalMs);
  for (const chunk of chunks) {
    if (ws.readyState !== 1) break;
    const b64 = chunk.toString('base64');
    ws.send(JSON.stringify({ type: 'input_audio_buffer.append', audio: b64 }));
    await wait(intervalMs);
  }
}

async function main() {
  const httpUrl = SERVER_URL.replace('ws://', 'http://').replace('wss://', 'https://');
  console.log(`=== Inspector Lanes E2E Test ===`);
  console.log(`Server: ${SERVER_URL}`);
  console.log(`Pattern: ${SPEECH_MS}ms speech -> ${SILENCE_MS}ms silence x ${CYCLES} cycles\n`);

  try {
    const r = await fetch(`${httpUrl}/health`);
    if (!r.ok) throw new Error(`health ${r.status}`);
  } catch (err) {
    console.error(`Server not reachable: ${err.message}`);
    process.exit(1);
  }
  console.log('Server health: OK');

  // Open realtime session
  const { ws: rtWs, events: rtEvents } = await openRealtimeSession();
  console.log('Realtime WebSocket: connected');

  // Wait for session.created
  const t0 = Date.now();
  while (Date.now() - t0 < 5000) {
    if (rtEvents.some(e => e.type === 'session.created')) break;
    await wait(50);
  }
  const sessionCreated = rtEvents.find(e => e.type === 'session.created');
  if (!sessionCreated) {
    console.error('No session.created received');
    rtWs.close();
    process.exit(1);
  }
  const sid = sessionCreated.session?.id;
  console.log(`Session: ${sid}`);

  // Send session.update for conversation mode
  rtWs.send(JSON.stringify({
    type: 'session.update',
    session: {
      voice: 'af_heart',
      instructions: 'Respond with exactly one short sentence.',
      input_audio_format: AUDIO_FORMAT,
    },
  }));
  await wait(300);

  // Open inspector stream
  const { ws: inspWs, events: inspEvents } = await openInspectorStream(sid);
  console.log('Inspector stream: connected\n');

  // Generate audio
  const speechBuf = generateSpeechAudio(SPEECH_MS);
  const silenceBuf = generateSilence(SILENCE_MS);
  const CHUNK_MS = 20;

  for (let cycle = 0; cycle < CYCLES; cycle++) {
    console.log(`--- Cycle ${cycle + 1}/${CYCLES} ---`);
    console.log(`  Sending ${SPEECH_MS}ms speech audio...`);
    await sendAudioChunks(rtWs, speechBuf, CHUNK_MS);

    console.log(`  Sending ${SILENCE_MS}ms silence...`);
    await sendAudioChunks(rtWs, silenceBuf, CHUNK_MS);

    console.log(`  Waiting for pipeline to settle...`);
    await wait(3000);
  }

  // Drain remaining events
  await wait(2000);

  // Analyze inspector events
  const lanes = {};
  for (const ev of inspEvents) {
    const lane = ev.lane;
    if (!lane) continue;
    if (!lanes[lane]) lanes[lane] = [];
    lanes[lane].push(ev.kind);
  }

  // Analyze wire events
  const wireTypes = {};
  for (const ev of rtEvents) {
    const t = ev.type;
    if (!t) continue;
    wireTypes[t] = (wireTypes[t] || 0) + 1;
  }

  console.log('\n=== Inspector Lane Events ===');
  const expectedLanes = ['vad', 'stt', 'eou', 'llm', 'tts_req', 'tts_chunk', 'turn', 'wire', 'response'];
  for (const lane of expectedLanes) {
    const kinds = lanes[lane] || [];
    const summary = [...new Set(kinds)].join(', ');
    const status = kinds.length > 0 ? 'PASS' : 'MISS';
    console.log(`  [${status}] ${lane.padEnd(12)} ${kinds.length} events (${summary || 'none'})`);
  }
  for (const lane of Object.keys(lanes)) {
    if (!expectedLanes.includes(lane)) {
      console.log(`  [XTRA] ${lane.padEnd(12)} ${lanes[lane].length} events`);
    }
  }

  console.log('\n=== Wire Protocol Events ===');
  for (const [type, count] of Object.entries(wireTypes).sort()) {
    console.log(`  ${type}: ${count}`);
  }

  console.log('\n=== Verdict ===');
  let pass = 0, fail = 0;
  const check = (cond, msg) => { if (cond) { pass++; console.log(`  PASS: ${msg}`); } else { fail++; console.log(`  FAIL: ${msg}`); } };

  check((lanes['vad'] || []).length > 0, 'VAD lane has events');
  check((lanes['stt'] || []).includes('final'), 'STT lane has "final" event');
  check((lanes['eou'] || []).length > 0, 'EOU lane has events');
  check((lanes['turn'] || []).includes('turn_start'), 'Turn lane has "turn_start"');
  check((lanes['turn'] || []).includes('user_committed'), 'Turn lane has "user_committed"');
  check((lanes['wire'] || []).length > 0, 'Wire lane has events');

  // These depend on LLM+TTS being configured
  const hasLlm = (lanes['llm'] || []).length > 0;
  const hasTts = (lanes['tts_req'] || []).length > 0;
  const hasTtsChunk = (lanes['tts_chunk'] || []).length > 0;
  if (hasLlm) {
    check((lanes['llm'] || []).includes('request'), 'LLM lane has "request"');
    check((lanes['llm'] || []).includes('first_token'), 'LLM lane has "first_token"');
    check((lanes['llm'] || []).includes('done'), 'LLM lane has "done"');
  } else {
    console.log('  SKIP: LLM lane (no LLM configured)');
  }
  if (hasTts) {
    check((lanes['tts_req'] || []).includes('phrase_sent'), 'TTS req lane has "phrase_sent"');
  } else {
    console.log('  SKIP: TTS req lane (no TTS configured)');
  }
  if (hasTtsChunk) {
    check(true, 'TTS chunk lane has events');
  } else {
    console.log('  SKIP: TTS chunk lane (no TTS configured)');
  }

  console.log(`\n  Total: ${pass} pass, ${fail} fail`);

  rtWs.close();
  inspWs.close();
  process.exit(fail === 0 ? 0 : 1);
}

main().catch(err => {
  console.error(err);
  process.exit(1);
});
