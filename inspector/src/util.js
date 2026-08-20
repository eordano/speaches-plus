export function formatMs(ms) {
  if (ms == null) return '--';
  if (ms >= 1000) return (ms / 1000).toFixed(3) + 's';
  if (ms >= 1)    return ms.toFixed(1) + 'ms';
  return (ms * 1000).toFixed(1) + 'us';
}

export function escapeHTML(s) {
  return String(s).replace(/[&<>]/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;' }[c]));
}

export function syntaxHighlight(json) {
  return json
    .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
    .replace(/("[^"]+")(\s*:)/g, '<span class="tok-k">$1</span>$2')
    .replace(/:\s*("[^"]*")/g, ': <span class="tok-s">$1</span>')
    .replace(/:\s*(-?\d+\.?\d*)/g, ': <span class="tok-n">$1</span>')
    .replace(/:\s*(true|false)/g, ': <span class="tok-b">$1</span>')
    .replace(/:\s*(null)/g, ': <span class="tok-p">$1</span>');
}

// Round `ms` up to a "nice" tick interval (1, 2, 5 x 10^k).
export function niceStep(ms) {
  const safe = Math.max(0.001, ms);
  const exp = Math.pow(10, Math.floor(Math.log10(safe)));
  const n = safe / exp;
  if (n < 1.5) return exp;
  if (n < 3)   return 2 * exp;
  if (n < 7)   return 5 * exp;
  return 10 * exp;
}

export function debounce(fn, ms) {
  let t = null;
  return (...args) => {
    clearTimeout(t);
    t = setTimeout(() => fn(...args), ms);
  };
}

export function loadJSON(key, fallback) {
  try {
    const raw = localStorage.getItem(key);
    return raw ? JSON.parse(raw) : fallback;
  } catch {
    return fallback;
  }
}

export function saveJSON(key, value) {
  try { localStorage.setItem(key, JSON.stringify(value)); } catch {}
}

export function resampleLinear(samples, fromRate, toRate) {
  if (fromRate === toRate) return samples;
  const ratio = fromRate / toRate;
  const len = Math.floor(samples.length / ratio);
  const out = new Float32Array(len);
  for (let i = 0; i < len; i++) {
    const idx = i * ratio;
    const lo = Math.floor(idx);
    const hi = Math.min(lo + 1, samples.length - 1);
    const f = idx - lo;
    out[i] = samples[lo] * (1 - f) + samples[hi] * f;
  }
  return out;
}

export function float32ToInt16(f32) {
  const i16 = new Int16Array(f32.length);
  for (let i = 0; i < f32.length; i++) {
    const s = Math.max(-1, Math.min(1, f32[i]));
    i16[i] = s < 0 ? s * 0x8000 : s * 0x7FFF;
  }
  return i16;
}

export function int16ToBase64(i16) {
  const bytes = new Uint8Array(i16.buffer);
  let bin = '';
  for (let i = 0; i < bytes.length; i++) bin += String.fromCharCode(bytes[i]);
  return btoa(bin);
}

export function base64ToFloat32(b64) {
  const bin = atob(b64);
  const bytes = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
  const i16 = new Int16Array(bytes.buffer);
  const f32 = new Float32Array(i16.length);
  for (let i = 0; i < i16.length; i++) f32[i] = i16[i] / 0x8000;
  return f32;
}
