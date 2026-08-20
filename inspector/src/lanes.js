export const LANES = [
  { id: 'error',       name: 'Error',       hint: 'mirrored from any lane' },
  { id: 'audio_level', name: 'Audio',       hint: 'PCM RMS' },
  { id: 'vad',         name: 'VAD',         hint: 'Silero' },
  { id: 'stt',         name: 'STT',         hint: 'whisper' },
  { id: 'turn',        name: 'Turn',        hint: 'boundaries' },
  { id: 'bargein',     name: 'Barge-in',    hint: 'user interrupts' },
  { id: 'eou',         name: 'EOU',         hint: 'end-of-utterance classifier' },
  { id: 'diarization', name: 'Diarize',     hint: 'speaker ID' },
  { id: 'llm',         name: 'LLM',         hint: 'model' },
  { id: 'response',    name: 'Response',    hint: 'plan / phrase' },
  { id: 'tool',        name: 'Tool',        hint: 'use / result / summary' },
  { id: 'tts_req',     name: 'TTS phrases', hint: 'tts executor' },
  { id: 'tts_chunk',   name: 'TTS chunks',  hint: '24 kHz PCM' },
  { id: 'tts_pacer',   name: 'Pacer',       hint: 'played_ms cursor' },
  { id: 'wire',        name: 'Wire',        hint: 'protocol' },
  { id: 'state',       name: 'State',       hint: 'phase transitions' },
];

export const PALETTES = {
  warm: {
    audio_level: '#6E7C7F', vad: '#7A92A8', stt: '#7F9B7F',
    turn: '#9F7E9B', bargein: '#C06B8A', eou: '#8E7CB0',
    diarization: '#8A7CA0',
    llm: '#A8906A', response: '#C89B6A', tool: '#9CA88A',
    tts_req: '#C4A45A', tts_chunk: '#B88B5A', tts_pacer: '#A38A6F',
    wire: '#9B9590', state: '#7B7770', error: '#B88080',
  },
  semantic: {
    audio_level: '#6BBED3', vad: '#6FA8DC', stt: '#6BBE7F',
    turn: '#C77BBA', bargein: '#F07B90', eou: '#A88BD8',
    diarization: '#9B7BD8',
    llm: '#C8A2E8', response: '#E8A96B', tool: '#88C9A1',
    tts_req: '#E8A96B', tts_chunk: '#E8C76B', tts_pacer: '#D8B582',
    wire: '#9B9590', state: '#A0A0A0', error: '#E87878',
  },
  mono: {
    audio_level: '#5A564F', vad: '#8D7A5A', stt: '#A8906A',
    turn: '#8E7958', bargein: '#A67A66', eou: '#9A8675',
    diarization: '#8A7A68',
    llm: '#C8B08E', response: '#DEC49B', tool: '#A8B091',
    tts_req: '#DEC49B', tts_chunk: '#BC9C6E', tts_pacer: '#A89478',
    wire: '#726B62', state: '#888076', error: '#B88080',
  },
};

// Kinds that signal a hard problem worth surfacing in the error lane and
// the top-bar error badge. `cancelled`/`rejected_*` are warnings, not errors.
export const ERROR_KINDS = new Set([
  'error', 'phrase_error', 'dropped', 'raised', 'bargein_missed',
]);

// Kinds that get bigger ticks on the timeline (the "important" treatment).
export const IMPORTANT_KINDS = new Set([
  'first_token', 'pending_start',
  'error', 'dropped', 'raised', 'bargein_missed',
  'partial', 'final',
  'user_committed', 'turn_start', 'turn_end',
  'phrase_boundary',
  'use_token', 'result', 'start_summary', 'summary',
]);

// Events that pair into a band (start + end). The renderer skips drawing
// a tick for endpoint events because the band itself shows them.
const BAND_ENDPOINTS = {
  vad:       new Set(['confirmed_start', 'stopped']),
  llm:       new Set(['request', 'done']),
  tts_req:   new Set(['phrase_sent', 'phrase_rendered', 'phrase_done']),
  response:  new Set(['plan_start', 'done']),
  tts_chunk: new Set(['chunk', 'first_chunk']),
  stt:       new Set(['audio_direct']),
  bargein:   new Set(['bargein_pending', 'bargein_fired', 'bargein_cancelled']),
};

export function isBandEndpoint(ev) {
  const kinds = BAND_ENDPOINTS[ev.lane];
  return !!(kinds && kinds.has(ev.kind));
}

export function lookupLane(id) {
  return LANES.find(l => l.id === id) || { id, name: id, hint: '' };
}

export function laneColor(palette, laneId, kind) {
  const p = PALETTES[palette] || PALETTES.warm;
  if (laneId === 'error' || ERROR_KINDS.has(kind)) return p.error;
  return p[laneId] || '#999';
}
