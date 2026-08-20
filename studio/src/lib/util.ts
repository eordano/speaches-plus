import type { CSSProperties } from 'react';

export const OFFLINE_MSG = 'backend unreachable — /v1 did not answer';

export const tint = (hex: string, a: number): string => {
  const v = parseInt(hex.slice(1), 16);
  return `rgba(${v >> 16},${(v >> 8) & 255},${v & 255},${a})`;
};

export const icon = (d: string, size: number, color: string, sw: number): string =>
  `<svg width="${size}" height="${size}" viewBox="0 0 14 14"><path d="${d}" fill="none" style="stroke:${color}" stroke-width="${sw}"/></svg>`;

export const playIcon = (playing: boolean, size: number, color: string): string =>
  `<svg width="${size}" height="${size}" viewBox="0 0 10 10"><path d="${playing ? 'M2 1.5h2.2v7H2zM5.8 1.5H8v7H5.8z' : 'M2.5 1.5 8.5 5 2.5 8.5z'}" style="fill:${color}"/></svg>`;

export const fmtDur = (sec: number): string => {
  sec = Math.max(0, Math.round(sec));
  const m = Math.floor(sec / 60);
  const s = sec % 60;
  return m + ':' + String(s).padStart(2, '0');
};

export const pad = (key: string): string =>
  key.length >= 18 ? key + ' ' : (key + '                  ').slice(0, 18);

export const clamp = (v: number, lo: number, hi: number): number => Math.min(hi, Math.max(lo, v));

export const errMsg = (e: unknown): string => {
  if (e && typeof e === 'object' && 'message' in e) return String((e as { message: unknown }).message);
  return String(e);
};

export const dataUri = (text: string, mime?: string): string =>
  'data:' + (mime || 'text/plain') + ';charset=utf-8,' + encodeURIComponent(text);

export type DlFn = (uri: string, name: string) => void;
let dlImpl: DlFn = (uri, name) => {
  const a = document.createElement('a');
  a.href = uri;
  a.download = name;
  document.body.append(a);
  a.click();
  a.remove();
};
export const dl: DlFn = (uri, name) => dlImpl(uri, name);
export const getDl = (): DlFn => dlImpl;
export const setDl = (fn: DlFn): void => { dlImpl = fn; };

export const dlText = (text: string, name: string, mime?: string): void => dl(dataUri(text, mime), name);

export const stopTracks = (stream: MediaStream | null | undefined): void => {
  if (stream) stream.getTracks().forEach(t => { try { t.stop(); } catch {  } });
};

export const tryPlay = (el: HTMLMediaElement): void => {
  const p = el.play();
  if (p && typeof p.catch === 'function') p.catch(() => {});
};

export const tryPause = (el: HTMLMediaElement | null): void => {
  if (el) { try { el.pause(); } catch {  } }
};

export const postJSON = (path: string, body: unknown, signal?: AbortSignal): Promise<Response> =>
  fetch(path, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body), ...(signal ? { signal } : {}) });

export const relTime = (t: number, now = Date.now()): string => {
  const m = Math.floor(Math.max(0, now - t) / 60000);
  if (m < 1) return 'just now';
  if (m < 60) return m + ' min ago';
  const h = Math.floor(m / 60);
  if (h < 24) return h + ' h ago';
  return Math.floor(h / 24) + ' d ago';
};

export const selTone = (on: boolean): CSSProperties => ({
  border: `1px solid ${on ? '#1A1A1A' : '#D4CFC9'}`,
  background: on ? '#1A1A1A' : '#F2EDE4',
  color: on ? '#F2EDE4' : '#2B2B2B',
});

export function uuid(): string {
  if (typeof crypto !== 'undefined' && typeof crypto.randomUUID === 'function') return crypto.randomUUID();
  const b = crypto.getRandomValues(new Uint8Array(16));
  b[6] = ((b[6] as number) & 0x0f) | 0x40;
  b[8] = ((b[8] as number) & 0x3f) | 0x80;
  const h = [...b].map(x => x.toString(16).padStart(2, '0')).join('');
  return h.slice(0, 8) + '-' + h.slice(8, 12) + '-' + h.slice(12, 16) + '-' + h.slice(16, 20) + '-' + h.slice(20);
}
