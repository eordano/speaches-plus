'use strict';
import init, * as mod from './web/nv_ocr_web.js';

const params = new URLSearchParams(self.name);
const J = { started: new Date().toISOString(), ua: navigator.userAgent, tiers: (params.get('tiers') || 'codec,heads').split(',') };
const CASES = (params.get('cases') || 'en,es').split(',');
const VOICE = 'neutral_female';
const CKPT = './tts-model/consolidated.safetensors';
const log = s => postMessage({ log: s });
const finish = tag => postMessage({ finish: tag, J });
const now = () => performance.now();

function parseSafetensors(buf) {
  const dv = new DataView(buf);
  const n = Number(dv.getBigUint64(0, true));
  const header = JSON.parse(new TextDecoder().decode(new Uint8Array(buf, 8, n)));
  const base = 8 + n;
  return (name, ctor) => {
    const t = header[name];
    if (!t) throw new Error('safetensors lacks ' + name);
    const [a, b] = t.data_offsets;
    return { shape: t.shape, data: new ctor(buf.slice(base + a, base + b)) };
  };
}

async function fixture(c) {
  const t = parseSafetensors(await (await fetch('./tts-fixtures/voxtral_tts_' + c + '.safetensors')).arrayBuffer());
  const meta = await (await fetch('./tts-fixtures/voxtral_tts_' + c + '.json')).json();
  return { meta, t };
}

const SHARDS = [
  ['codec', k => k.startsWith('audio_tokenizer.')],
  ['heads', k => k.startsWith('acoustic_transformer.') || k === 'mm_audio_embeddings.audio_codebook_embeddings.embeddings.weight' || k === 'norm.weight'],
  ['embed', k => k === 'mm_audio_embeddings.tok_embeddings.weight'],
  ['layers-a', k => /^layers\.(\d+)\./.test(k) && Number(k.split('.')[1]) < 13],
  ['layers-b', k => /^layers\.(\d+)\./.test(k) && Number(k.split('.')[1]) >= 13],
];

async function fetchRange(url, a, b) {
  const r = await fetch(url, { headers: { Range: 'bytes=' + a + '-' + b } });
  if (r.status !== 206) throw new Error('range fetch ' + a + '-' + b + ': HTTP ' + r.status);
  return r.arrayBuffer();
}

async function virtualShards(url) {
  const hlen = Number(new DataView(await fetchRange(url, 0, 7)).getBigUint64(0, true));
  const header = JSON.parse(new TextDecoder().decode(await fetchRange(url, 8, 7 + hlen)));
  const base = 8 + hlen;
  const raw = rangeReader(url);
  const shards = [];
  for (const [name, keep] of SHARDS) {
    const entries = Object.entries(header).filter(([k]) => k !== '__metadata__' && keep(k)).sort((a, b) => a[1].data_offsets[0] - b[1].data_offsets[0]);
    const rebuilt = {};
    const spans = [];
    let cur = 0;
    for (const [k, v] of entries) {
      const [a, b] = v.data_offsets;
      rebuilt[k] = { dtype: v.dtype, shape: v.shape, data_offsets: [cur, cur + (b - a)] };
      spans.push({ v: cur, f: base + a, n: b - a });
      cur += b - a;
    }
    let hb = new TextEncoder().encode(JSON.stringify(rebuilt));
    const pad = (8 - hb.length % 8) % 8;
    if (pad) { const p = new Uint8Array(hb.length + pad); p.fill(0x20); p.set(hb); hb = p; }
    const prefix = new Uint8Array(8 + hb.length);
    new DataView(prefix.buffer).setBigUint64(0, BigInt(hb.length), true);
    prefix.set(hb, 8);
    const vbase = prefix.length;
    const read = (off, n) => {
      const out = new Uint8Array(n);
      let done = 0;
      while (done < n) {
        const o = off + done;
        if (o < vbase) {
          const take = Math.min(n - done, vbase - o);
          out.set(prefix.subarray(o, o + take), done);
          done += take;
          continue;
        }
        const vo = o - vbase;
        let lo = 0, hi = spans.length - 1;
        while (lo < hi) { const mid = (lo + hi + 1) >> 1; if (spans[mid].v <= vo) lo = mid; else hi = mid - 1; }
        const sp = spans[lo];
        const within = vo - sp.v;
        if (!sp || within >= sp.n) throw new Error(name + ': virtual read past the shard at ' + o);
        const take = Math.min(n - done, sp.n - within);
        out.set(raw(sp.f + within, take), done);
        done += take;
      }
      return out;
    };
    const bytes = vbase + cur;
    shards.push({ name: name + '.safetensors', len: bytes, read, tensors: entries.length });
    log('virtual shard ' + name + ': ' + entries.length + ' tensors, ' + bytes + ' bytes');
  }
  return shards;
}

function rangeReader(url) {
  return (off, n) => {
    const x = new XMLHttpRequest();
    x.open('GET', url, false);
    x.responseType = 'arraybuffer';
    x.setRequestHeader('Range', 'bytes=' + off + '-' + (off + n - 1));
    x.send();
    if (x.status !== 206) throw new Error('range ' + off + '+' + n + ': HTTP ' + x.status);
    return new Uint8Array(x.response);
  };
}

function maxAbs(a, b) {
  if (a.length !== b.length) throw new Error('length ' + a.length + ' vs ' + b.length);
  let m = 0;
  for (let i = 0; i < a.length; i++) m = Math.max(m, Math.abs(a[i] - b[i]));
  return m;
}
function maxRel(a, b) {
  let scale = 1e-6;
  for (const v of b) scale = Math.max(scale, Math.abs(v));
  return maxAbs(a, b) / scale;
}
function within(a, b, band) {
  let n = 0;
  for (let i = 0; i < a.length; i++) if (Math.abs(a[i] - b[i]) <= band) n++;
  return n / a.length;
}
function clampRef(w) { return w.map(s => Math.min(1, Math.max(-1, s))); }
function rows(t, i) { const w = t.shape[1]; return t.data.slice(i * w, (i + 1) * w); }
function codeDiffs(got, want) {
  const d = { semantic: 0, exact: 0, off_by_one: 0, worse: 0 };
  for (let f = 0; f < Math.min(got.length, want.length); f++) {
    if (got[f][0] !== want[f][0]) d.semantic++;
    for (let k = 1; k < 37; k++) {
      const x = Math.abs(got[f][k] - want[f][k]);
      if (x === 0) d.exact++; else if (x === 1) d.off_by_one++; else d.worse++;
    }
  }
  return d;
}
function argmaxMasked(logits, rows) {
  let best = -Infinity, idx = 1;
  for (let i = 1; i < rows; i++) if (logits[i] > best) { best = logits[i]; idx = i; }
  return idx;
}

async function adapterInfo() {
  const a = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
  if (!a) throw new Error('requestAdapter returned null');
  const info = a.info || {};
  const L = a.limits;
  return { vendor: info.vendor, architecture: info.architecture, device: info.device, description: info.description,
    features: [...a.features], limits: { maxBufferSize: L.maxBufferSize, maxStorageBufferBindingSize: L.maxStorageBufferBindingSize,
    maxStorageBuffersPerShaderStage: L.maxStorageBuffersPerShaderStage, maxComputeWorkgroupStorageSize: L.maxComputeWorkgroupStorageSize,
    maxComputeInvocationsPerWorkgroup: L.maxComputeInvocationsPerWorkgroup, maxComputeWorkgroupsPerDimension: L.maxComputeWorkgroupsPerDimension } };
}

async function codecTier(fx) {
  const R = {};
  let t = now();
  mod.tts_codec_boot();
  R.boot_ms = now() - t;
  log('codec booted in ' + Math.round(R.boot_ms) + ' ms');
  R.cases = {};
  let ok = true;
  for (const c of CASES) {
    const { t: fix } = fx[c];
    const frames = fix('frames', Int32Array);
    const codecCodes = fix('codec_codes', Int32Array);
    const ref = fix('waveform', Float32Array).data;
    let wav, best = Infinity;
    for (let i = 0; i < 3; i++) {
      t = now();
      wav = await mod.tts_codec_decode(frames.data);
      best = Math.min(best, now() - t);
    }
    const nFrames = codecCodes.shape[0];
    const diff = maxAbs(wav, ref);
    const diffClamped = maxAbs(wav, clampRef(ref));
    mod.tts_codec_stream_reset();
    const streamed = new Float32Array(wav.length);
    let at = 0;
    t = now();
    for (let f = 0; f < nFrames; f++) {
      const frame = codecCodes.data.slice(f * 37, (f + 1) * 37).map(x => x + 2);
      const pcm = await mod.tts_codec_decode_frames(frame);
      streamed.set(pcm, at);
      at += pcm.length;
    }
    const perFrame = (now() - t) / nFrames;
    const r = { frames: nFrames, samples: wav.length, max_abs: diff, max_abs_clamped: diffClamped, within_1e2: within(wav, ref, 1e-2),
      full_best_ms: best, per_frame_stateful_ms: perFrame, stream_vs_full_max_abs: maxAbs(streamed, wav) };
    R.cases[c] = r;
    log('codec ' + c + ': ' + JSON.stringify(r));
    ok = ok && diffClamped < 0.02 && r.stream_vs_full_max_abs < 1e-2;
  }
  R.verdict = ok ? 'CODEC-DECODES' : 'CODEC-OFF-BAND';
  return R;
}

async function stageVoice() {
  const pt = new Uint8Array(await (await fetch('./tts-model/voice_embedding/' + VOICE + '.pt')).arrayBuffer());
  mod.tts_stage_voice_torch(VOICE, pt);
}

async function headsTier(fx) {
  const R = {};
  let t = now();
  R.boot = JSON.parse(mod.tts_boot(false));
  R.boot_ms = now() - t;
  log('heads booted in ' + Math.round(R.boot_ms) + ' ms: ' + JSON.stringify(R.boot));
  R.cases = {};
  let ok = true;
  for (const c of CASES) {
    const { meta, t: fix } = fx[c];
    const noise = fix('noise', Float32Array);
    const frames = fix('frames', Int32Array);
    const last = frames.shape[0] - 1;
    const r = { frames: {} };
    for (const [k, name] of [[0, 'lm_hidden_frame_0'], [1, 'lm_hidden_frame_1'], [2, 'lm_hidden_frame_2'], [last, 'lm_hidden_frame_last']]) {
      const out = JSON.parse(await mod.tts_frame_from_hidden(fix(name, Float32Array).data, rows(noise, k), k === 0));
      const want = Array.from(rows(frames, k));
      const d = codeDiffs([out.frame], [want]);
      r.frames[k] = d;
      if (k === 0) {
        const refLogits = fix('semantic_logits_frame0', Float32Array).data;
        r.semantic_logits_max_abs = maxAbs(out.semantic_logits, refLogits);
        r.semantic_logits_rel = maxRel(out.semantic_logits, refLogits);
        r.semantic_argmax = out.frame[0];
        r.oracle_semantic = want[0];
        r.oracle_argmax_over_our_logits = argmaxMasked(out.semantic_logits, 8194);
        const vc = fix('flow_v_cond_frame0', Float32Array), vu = fix('flow_v_uncond_frame0', Float32Array), x = fix('flow_x_frame0', Float32Array);
        r.flow_rel = { v_cond: 0, v_uncond: 0, x: 0 };
        for (let i = 0; i < meta.nfe; i++) {
          r.flow_rel.v_cond = Math.max(r.flow_rel.v_cond, maxRel(out.v_cond[i], rows(vc, i)));
          r.flow_rel.v_uncond = Math.max(r.flow_rel.v_uncond, maxRel(out.v_uncond[i], rows(vu, i)));
          r.flow_rel.x = Math.max(r.flow_rel.x, maxRel(out.x[i + 1], rows(x, i + 1)));
        }
        ok = ok && r.flow_rel.v_cond < 0.05 && r.flow_rel.v_uncond < 0.05 && r.flow_rel.x < 0.05 && r.semantic_logits_rel < 0.05;
      }
      ok = ok && d.worse === 0;
    }
    let best = Infinity;
    const h0 = fix('lm_hidden_frame_0', Float32Array).data, n0 = rows(noise, 0);
    for (let i = 0; i < 5; i++) {
      t = now();
      await mod.tts_frame_from_hidden(h0, n0, false);
      best = Math.min(best, now() - t);
    }
    r.heads_plus_flow_best_ms = best;
    R.cases[c] = r;
    log('heads ' + c + ': ' + JSON.stringify(r));
  }
  R.verdict = ok ? 'HEADS-MATCH' : 'HEADS-OFF-BAND';
  return R;
}

async function fullTier(fx) {
  const R = {};
  let t = now();
  R.boot = JSON.parse(mod.tts_boot(true));
  R.boot_ms = now() - t;
  log('full model booted in ' + Math.round(R.boot_ms) + ' ms');
  if (!J.tiers.includes('codec')) mod.tts_codec_boot();
  await stageVoice();
  R.cases = {};
  let ok = true;
  for (const c of CASES) {
    const { meta, t: fix } = fx[c];
    const noise = fix('noise', Float32Array);
    const frames = fix('frames', Int32Array);
    const want = [];
    for (let f = 0; f < frames.shape[0]; f++) want.push(Array.from(rows(frames, f)));
    t = now();
    const out = JSON.parse(await mod.tts_synthesize(meta.text, VOICE, noise.data, BigInt(0), 4 * noise.shape[0], new Uint32Array(0)));
    const ms = now() - t;
    const r = { frames: out.frames.length, oracle_frames: want.length, ended: out.ended, prompt_ids_match: JSON.stringify(out.prompt_ids) === JSON.stringify(meta.prompt_token_ids),
      free_run_diffs: codeDiffs(out.frames, want), total_ms: ms, ms_per_frame: ms / Math.max(1, out.frames.length) };
    log('full ' + c + ' synthesized: ' + JSON.stringify(r));
    t = now();
    const forced = JSON.parse(await mod.tts_synthesize(meta.text, VOICE, noise.data, BigInt(0), 0, Uint32Array.from(frames.data)));
    r.teacher_forced_ms = now() - t;
    r.teacher_forced_diffs = codeDiffs(forced.frames, want);
    r.teacher_forced_in_band = r.teacher_forced_diffs.worse === 0 && r.teacher_forced_diffs.off_by_one < 0.05 * 36 * want.length;
    t = now();
    const pcm = await mod.tts_codec_decode(Int32Array.from(out.frames.flat()));
    r.codec_ms = now() - t;
    r.pcm_samples = pcm.length;
    R.cases[c] = r;
    log('full ' + c + ': ' + JSON.stringify(r));
    ok = ok && out.ended && r.prompt_ids_match;
  }
  const band = Object.values(R.cases).every(r => r.teacher_forced_in_band);
  R.verdict = ok ? (band ? 'FULL-RUNS-TEACHER-FORCED-IN-BAND' : 'FULL-RUNS') : 'FULL-INCOMPLETE';
  return R;
}

async function main() {
  if (!navigator.gpu) { J.error = 'navigator.gpu missing in the worker'; return finish('tts-no-webgpu'); }
  J.adapter = await adapterInfo();
  log('adapter: ' + JSON.stringify(J.adapter));
  let t = now();
  await init();
  J.wasm_init_ms = now() - t;
  mod.q38_install_panic_hook();
  t = now();
  const shards = await virtualShards(CKPT);
  await mod.tts_open_checkpoint(shards);
  J.open_ms = now() - t;
  J.shards = shards.map(s => ({ name: s.name, len: s.len, tensors: s.tensors }));
  log('checkpoint opened as ' + shards.length + ' virtual shards in ' + Math.round(J.open_ms) + ' ms');
  mod.tts_stage_text('params.json', await (await fetch('./tts-model/params.json')).text());
  mod.tts_stage_text('tekken.json', await (await fetch('./tts-model/tekken.json')).text());
  const fx = {};
  for (const c of CASES) fx[c] = await fixture(c);
  J.tts = {};
  if (J.tiers.includes('codec')) J.tts.codec = await codecTier(fx);
  if (J.tiers.includes('heads')) J.tts.heads = await headsTier(fx);
  if (J.tiers.includes('full')) J.tts.full = await fullTier(fx);
  const verdicts = Object.values(J.tts).map(x => x.verdict);
  J.verdict = verdicts.join('+');
  finish(verdicts.every(v => /DECODES|MATCH|RUNS/.test(v)) ? 'tts-ok' : 'tts-bad');
}

main().catch(e => { J.error = String(e && e.stack || e); log('ERROR ' + J.error); finish('tts-error'); });
