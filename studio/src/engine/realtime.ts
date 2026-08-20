import type { RealtimeClientEvent, RealtimeServerEvent } from 'nur-client/realtime';

const RT_TARGET_SR = 24000;
const FLUSH_SAMPLES = 2400;
const BP_THRESHOLD = 64 * 1024;
const TURN_DETECTION = { threshold: 0.6, silence_duration_ms: 350, prefix_padding_ms: 300, create_response: false };

const WORKLET_SRC = `
class NurRtProcessor extends AudioWorkletProcessor {
  constructor(options) {
    super();
    const o = (options && options.processorOptions) || {};
    this._flush = o.flushSamples || ${FLUSH_SAMPLES};
    this._stride = sampleRate / (o.targetSR || ${RT_TARGET_SR});
    this._phase = 0;
    this._out = new Int16Array(this._flush);
    this._n = 0;
  }
  process(inputs) {
    const ch = inputs[0] && inputs[0][0];
    if (!ch) return true;
    let n = this._n, phase = this._phase;
    const out = this._out, flush = this._flush, stride = this._stride;
    for (let i = 0; i < ch.length; i++, phase++) {
      if (phase >= stride) {
        phase -= stride;
        let s = ch[i];
        if (s > 1) s = 1; else if (s < -1) s = -1;
        out[n++] = s < 0 ? (s * 0x8000) | 0 : (s * 0x7fff) | 0;
        if (n === flush) {
          this.port.postMessage(out.buffer, [out.buffer]);
          this._out = new Int16Array(flush);
          n = 0;
        }
      }
    }
    this._phase = phase;
    this._n = n;
    return true;
  }
}
registerProcessor('nur-rt', NurRtProcessor);
`;

let workletUrl: string | null = null;
const workletModuleUrl = (): string => {
  if (!workletUrl) workletUrl = URL.createObjectURL(new Blob([WORKLET_SRC], { type: 'application/javascript' }));
  return workletUrl;
};

const B64_SLICE = 0x8000;
function bufToBase64(buf: ArrayBuffer): string {
  const bytes = new Uint8Array(buf);
  let s = '';
  for (let i = 0; i < bytes.length; i += B64_SLICE) {
    s += String.fromCharCode.apply(null, bytes.subarray(i, i + B64_SLICE) as unknown as number[]);
  }
  return btoa(s);
}

export interface RtStats {
  attempts: number;
  opens: number;
  frames: number;
  partials: number;
  finals: number;
  appends: number;
  drops: number;
  lastEvent: string | null;
  lastError: string | null;
}
export const rtStats: RtStats = {
  attempts: 0, opens: 0, frames: 0, partials: 0, finals: 0, appends: 0, drops: 0,
  lastEvent: null, lastError: null,
};

const RT_OFF = /[?&]rt=off(?:&|$)/.test(location.search);
export const realtimeEnabled = (): boolean => !RT_OFF && typeof WebSocket === 'function';

export const realtimeUrl = (model: string | null): string =>
  (location.protocol === 'https:' ? 'wss://' : 'ws://') + location.host +
  '/v1/realtime?intent=transcription' + (model ? '&model=' + encodeURIComponent(model) : '');

export interface RealtimeHandlers {
  partial?: (text: string) => void;
  final?: (text: string) => void;
  speechStart?: (atMs: number) => void;
  speechStop?: (atMs: number) => void;
  error?: (message: string) => void;
  close?: () => void;
}

export interface RealtimeSession {
  start: (stream: MediaStream) => Promise<void>;
  commit: () => void;
  stop: () => void;
  live: () => boolean;
}

export function openRealtime(model: string | null, on: RealtimeHandlers): RealtimeSession {
  let ws: WebSocket | null = null;
  let ac: AudioContext | null = null;
  let node: AudioWorkletNode | null = null;
  let src: MediaStreamAudioSourceNode | null = null;
  let open = false;
  let stopped = false;

  const handle = (ev: RealtimeServerEvent): void => {
    rtStats.lastEvent = ev.type;
    switch (ev.type) {
      case 'input_audio_buffer.speech_started':
        if (on.speechStart) on.speechStart(ev.audio_start_ms);
        break;
      case 'input_audio_buffer.speech_stopped':
        if (on.speechStop) on.speechStop(ev.audio_end_ms);
        break;
      case 'input_audio_buffer.partial_transcription':
        rtStats.partials++;
        if (on.partial) on.partial(ev.transcript);
        break;
      case 'conversation.item.input_audio_transcription.completed':
        rtStats.finals++;
        if (on.final) on.final(ev.transcript);
        break;
      case 'error':
        rtStats.lastError = ev.error.message;
        if (on.error) on.error(ev.error.message);
        break;
      default:
        break;
    }
  };

  const openSocket = (): Promise<WebSocket> => new Promise((res, rej) => {
    rtStats.attempts++;
    let s: WebSocket;
    try { s = new WebSocket(realtimeUrl(model)); }
    catch (e) { rej(e instanceof Error ? e : new Error(String(e))); return; }
    s.binaryType = 'arraybuffer';
    s.addEventListener('open', () => {
      rtStats.opens++;
      open = true;
      const update: RealtimeClientEvent = {
        type: 'session.update',
        session: {
          input_audio_format: 'pcm16',
          turn_detection: { ...TURN_DETECTION },
          ...(model ? { input_audio_transcription: { model } } : {}),
        },
      };
      s.send(JSON.stringify(update));
      res(s);
    });
    s.addEventListener('error', () => {
      rtStats.lastError = 'realtime ws error';
      if (open) { if (on.error) on.error('realtime ws error'); }
      else rej(new Error('realtime ws error'));
    });
    s.addEventListener('close', () => {
      const was = open;
      open = false;
      if (was && !stopped && on.close) on.close();
    });
    s.addEventListener('message', (m: MessageEvent<unknown>) => {
      rtStats.frames++;
      const raw = m.data instanceof ArrayBuffer ? new TextDecoder().decode(m.data) : m.data;
      if (typeof raw !== 'string') return;
      let ev: RealtimeServerEvent;
      try { ev = JSON.parse(raw) as RealtimeServerEvent; } catch { return; }
      if (ev && typeof ev.type === 'string') handle(ev);
    });
  });

  const start = async (stream: MediaStream): Promise<void> => {
    ws = await openSocket();
    if (stopped) { try { ws.close(); } catch {  } return; }
    const AC: typeof AudioContext | undefined =
      window.AudioContext || (window as { webkitAudioContext?: typeof AudioContext }).webkitAudioContext;
    if (!AC) throw new Error('AudioContext unavailable');
    try { ac = new AC({ sampleRate: RT_TARGET_SR }); } catch { ac = new AC(); }
    if (ac.state === 'suspended') { try { await ac.resume(); } catch {  } }
    await ac.audioWorklet.addModule(workletModuleUrl());
    if (stopped || !ac) return;
    src = ac.createMediaStreamSource(stream);
    node = new AudioWorkletNode(ac, 'nur-rt', {
      numberOfInputs: 1, numberOfOutputs: 0, channelCount: 1,
      processorOptions: { flushSamples: FLUSH_SAMPLES, targetSR: RT_TARGET_SR },
    });
    node.port.onmessage = (ev: MessageEvent<unknown>) => {
      if (!ws || ws.readyState !== WebSocket.OPEN || !(ev.data instanceof ArrayBuffer)) return;
      if (ws.bufferedAmount > BP_THRESHOLD) { rtStats.drops++; return; }
      const append: RealtimeClientEvent = { type: 'input_audio_buffer.append', audio: bufToBase64(ev.data) };
      rtStats.appends++;
      ws.send(JSON.stringify(append));
    };
    src.connect(node);
  };

  const commit = (): void => {
    if (!ws || ws.readyState !== WebSocket.OPEN) return;
    const ev: RealtimeClientEvent = { type: 'input_audio_buffer.commit' };
    ws.send(JSON.stringify(ev));
  };

  const stop = (): void => {
    stopped = true;
    open = false;
    try { if (node) node.disconnect(); } catch {  }
    try { if (src) src.disconnect(); } catch {  }
    if (ac && ac.state !== 'closed') { try { void ac.close(); } catch {  } }
    ac = null; node = null; src = null;
    if (ws && ws.readyState <= WebSocket.OPEN) { try { ws.close(); } catch {  } }
    ws = null;
  };

  return { start, commit, stop, live: () => open };
}
