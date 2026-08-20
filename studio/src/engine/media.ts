import { fmtDur, stopTracks, tryPlay } from '../lib/util';
import { scheduleRender, type Msg } from '../state/store';

let actx: AudioContext | null = null;
export function audioCtx(): AudioContext | null {
  const AC: typeof AudioContext | undefined =
    window.AudioContext || (window as { webkitAudioContext?: typeof AudioContext }).webkitAudioContext;
  if (!actx && AC) { try { actx = new AC(); } catch { actx = null; } }
  if (actx && actx.state === 'suspended') { try { void actx.resume(); } catch {  } }
  return actx;
}
if (typeof document !== 'undefined') {
  document.addEventListener('pointerdown', () => { audioCtx(); }, { once: true, capture: true });
}

export interface Peaks { dur: number; peaks: number[] }

export async function decodePeaks(blob: Blob | null, buckets?: number): Promise<Peaks | null> {
  const ctx = audioCtx();
  if (!ctx || !blob) return null;
  const ab = await blob.arrayBuffer();
  const buf = await new Promise<AudioBuffer>((res, rej) => {
    try {
      const p = ctx.decodeAudioData(ab, res, rej);
      if (p && typeof p.then === 'function') p.then(res, rej);
    } catch (e) { rej(e); }
  });
  const ch = buf.getChannelData(0);
  const n = buckets || 44, per = Math.max(1, Math.floor(ch.length / n)), peaks: number[] = [];
  for (let i = 0; i < n; i++) {
    let m = 0;
    const off = i * per;
    for (let j = 0; j < per; j += 16) { const v = Math.abs(ch[off + j] || 0); if (v > m) m = v; }
    peaks.push(m);
  }
  const mx = Math.max(...peaks) || 1;
  return { dur: buf.duration, peaks: peaks.map(p => p / mx) };
}

export function attachAudioMeta(m: Msg | undefined): void {
  if (!m || !m.blob || m._decoding || m.peaks) return;
  m._decoding = true;
  decodePeaks(m.blob, 44).then(got => {
    if (got) { m.durSec = got.dur; m.durLabel = fmtDur(got.dur); m.peaks = got.peaks; }
    m._decoding = false;
    scheduleRender();
  }).catch(() => { m._decoding = false; scheduleRender(); });
}

export function peaksPoly(peaks: number[], w: number, hgt: number): string {
  const mid = hgt / 2, n = peaks.length, pts: string[] = [];
  for (let i = 0; i < n; i++) {
    const x = Math.round(i / (n - 1) * w);
    const a = Math.max(0.06, peaks[i] ?? 0) * (hgt / 2 - 1);
    pts.push(x + ',' + (mid - a).toFixed(1), x + ',' + (mid + a).toFixed(1));
  }
  return pts.join(' ');
}

export const SPEECH_RMS = 0.015;

const rmsOf = (an: AnalyserNode, buf: Float32Array<ArrayBuffer>): number => {
  an.getFloatTimeDomainData(buf);
  let s = 0;
  for (let i = 0; i < buf.length; i++) s += (buf[i] ?? 0) * (buf[i] ?? 0);
  return Math.sqrt(s / buf.length);
};

export interface Utterance { blob: Blob; ms: number; silence: number | null }

export async function recordUtterance(
  opts?: { silenceMs?: number; maxMs?: number; onStream?: (stream: MediaStream) => void; onChunk?: (soFar: Blob) => void; stopWhen?: () => boolean },
): Promise<Utterance> {
  const silenceMs = opts?.silenceMs || 500, maxMs = opts?.maxMs || 20000, THRESH = SPEECH_RMS;
  const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
  if (opts?.onStream) { try { opts.onStream(stream); } catch {  } }
  let mr: MediaRecorder;
  try { mr = new MediaRecorder(stream); }
  catch (e) { stopTracks(stream); throw e; }
  const chunks: Blob[] = [];
  mr.ondataavailable = (e) => {
    if (!e.data || !e.data.size) return;
    chunks.push(e.data);
    if (opts?.onChunk) { try { opts.onChunk(new Blob(chunks, { type: mr.mimeType || 'audio/webm' })); } catch {  } }
  };
  const stopped = new Promise<unknown>(res => { mr.onstop = res; });
  const ctx = audioCtx();
  let an: AnalyserNode | null = null, src: MediaStreamAudioSourceNode | null = null;
  let buf: Float32Array<ArrayBuffer> | null = null;
  if (ctx) {
    try {
      src = ctx.createMediaStreamSource(stream);
      an = ctx.createAnalyser(); an.fftSize = 2048;
      src.connect(an);
      buf = new Float32Array(an.fftSize);
    } catch { an = null; }
  }
  const t0 = performance.now();
  try { mr.start(500); }
  catch (e) {
    try { if (src) src.disconnect(); } catch {  }
    stopTracks(stream);
    throw e;
  }
  let speechAt: number | null = null, silentSince: number | null = null, silence: number | null = null;
  await new Promise<void>(res => {
    const iv = setInterval(() => {
      if (opts?.stopWhen && opts.stopWhen()) { clearInterval(iv); res(); return; }
      const now = performance.now();
      const level = an && buf ? rmsOf(an, buf) : 0;
      if (an) {
        if (level > THRESH) { if (speechAt == null) speechAt = now; silentSince = null; }
        else if (speechAt != null) {
          if (silentSince == null) silentSince = now;
          else if (now - silentSince >= silenceMs) {
            silence = Math.round(now - silentSince);
            clearInterval(iv); res(); return;
          }
        }
      }
      if (now - t0 > maxMs) { clearInterval(iv); res(); }
    }, 50);
  });
  try { mr.stop(); } catch {  }
  await stopped;
  stopTracks(stream);
  try { if (src) src.disconnect(); } catch {  }
  return {
    blob: new Blob(chunks, { type: mr.mimeType || 'audio/webm' }),
    ms: Math.round(performance.now() - t0),
    silence,
  };
}

interface WavParts { fmt: Uint8Array; data: Uint8Array }

function wavParts(buf: ArrayBuffer): WavParts | null {
  const u8 = new Uint8Array(buf);
  const dv = new DataView(buf);
  const tag = (off: number): string => String.fromCharCode(u8[off] ?? 0, u8[off + 1] ?? 0, u8[off + 2] ?? 0, u8[off + 3] ?? 0);
  if (buf.byteLength < 44 || tag(0) !== 'RIFF' || tag(8) !== 'WAVE') return null;
  let off = 12, fmt: Uint8Array | null = null, data: Uint8Array | null = null;
  while (off + 8 <= buf.byteLength) {
    const id = tag(off);
    const size = dv.getUint32(off + 4, true);
    const body = u8.subarray(off + 8, Math.min(off + 8 + size, buf.byteLength));
    if (id === 'fmt ') fmt = body;
    else if (id === 'data') data = body;
    off += 8 + size + (size % 2);
  }
  return fmt && data ? { fmt, data } : null;
}

export async function mergeWavBlobs(blobs: Blob[]): Promise<Blob> {
  if (blobs.length === 1) return blobs[0] as Blob;
  const parts: WavParts[] = [];
  for (const b of blobs) {
    const p = wavParts(await b.arrayBuffer());
    if (!p) return new Blob(blobs, { type: 'audio/wav' });
    parts.push(p);
  }
  const first = parts[0] as WavParts;
  const dataLen = parts.reduce((a, p) => a + p.data.length, 0);
  const total = 12 + 8 + first.fmt.length + (first.fmt.length % 2) + 8 + dataLen;
  const out = new Uint8Array(total);
  const dv = new DataView(out.buffer);
  const put = (off: number, s: string): void => { for (let i = 0; i < 4; i++) out[off + i] = s.charCodeAt(i); };
  put(0, 'RIFF');
  dv.setUint32(4, total - 8, true);
  put(8, 'WAVE');
  put(12, 'fmt ');
  dv.setUint32(16, first.fmt.length, true);
  out.set(first.fmt, 20);
  let off = 20 + first.fmt.length + (first.fmt.length % 2);
  put(off, 'data');
  dv.setUint32(off + 4, dataLen, true);
  off += 8;
  for (const p of parts) { out.set(p.data, off); off += p.data.length; }
  return new Blob([out], { type: 'audio/wav' });
}

export interface MicMonitor { level: () => number; stop: () => void }

export async function micMonitor(): Promise<MicMonitor | null> {
  const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
  const ctx = audioCtx();
  if (!ctx) { stopTracks(stream); return null; }
  const src = ctx.createMediaStreamSource(stream);
  const an = ctx.createAnalyser(); an.fftSize = 2048;
  src.connect(an);
  const buf: Float32Array<ArrayBuffer> = new Float32Array(an.fftSize);
  return {
    level: () => rmsOf(an, buf),
    stop: () => {
      try { src.disconnect(); } catch {  }
      stopTracks(stream);
    },
  };
}

export interface Frame { blob: Blob; dataUrl: string; w: number; h: number }

export function grabFrame(stream: MediaStream): Promise<Frame> {
  return new Promise<Frame>((res, rej) => {
    const v = document.createElement('video');
    v.muted = true; v.playsInline = true; v.srcObject = stream;
    const fail = setTimeout(() => rej(new Error('frame capture timed out')), 6000);
    const snap = () => {
      setTimeout(() => {
        try {
          const c = document.createElement('canvas');
          c.width = v.videoWidth || 640; c.height = v.videoHeight || 360;
          const g = c.getContext('2d');
          if (!g) throw new Error('no 2d context');
          g.drawImage(v, 0, 0);
          clearTimeout(fail);
          const dataUrl = c.toDataURL('image/png');
          c.toBlob(b => b ? res({ blob: b, dataUrl, w: c.width, h: c.height })
            : rej(new Error('frame encode failed')), 'image/png');
        } catch (e) { clearTimeout(fail); rej(e instanceof Error ? e : new Error(String(e))); }
      }, 150);
    };
    v.onloadeddata = () => { tryPlay(v); snap(); };
    v.onerror = () => { clearTimeout(fail); rej(new Error('video element error')); };
  });
}
