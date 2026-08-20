import { bus } from '../state/bus';
import { S, scheduleRender, type Block, type BlockKind, type Msg, type Path, type Turn, type Val } from '../state/store';
import { clamp, errMsg, pad, stopTracks, OFFLINE_MSG } from '../lib/util';
import { attachAudioMeta, grabFrame, micMonitor, recordUtterance, SPEECH_RMS } from './media';
import type { ChatMessageIn } from '../api/generated/ChatMessageIn';
import {
  getMode, chatModel, sttModel, ttsModel, appendLog, setBusy, scrollChatSoon,
  streamChat, transcribe, speechPipeline, applyChatResult, applySttMsg, setMicState,
  playForRun, stopAudio, nurStore, modelRefusal, ensureBaseModels,
  abortChat, chunkedPartials, setComposerHook, ocrFrame,
  type SpeakResult, type SpeechPipeline,
} from './session';
import { openRealtime, realtimeEnabled, type RealtimeSession } from './realtime';
import type { Utterance } from './media';
import { HATS, gv, hatRoot, countNodes, getList, pk, bcNum, bcStr, ifExpr, flashNotice, flow } from './patch';
import { tapOf } from './ports';
import {
  parseExpr, evalExpr, templateStr, str, typeName, isBuiltin, isWritableBuiltin, VAR_RE,
  type Scope,
} from './expr';
import { storage } from '../state/storage';

const sleep = (ms: number): Promise<void> => new Promise(r => setTimeout(r, ms));

export let running = false;
let runAbort = false;
export function abortRun(): void {
  if (running) { runAbort = true; abortChat(); stopAudio(); bus.emit('run.abort', {}); }
}

interface RunCtx {
  turn: Turn;
  vars: Record<string, Val>;
  iter: number;
  tapDepth?: number;
  ran?: boolean;
  audioBlob?: Blob;
  vadSilence?: number | null;
  userIdx?: number;
  raw?: string;
  transcript?: string;
  sttDetail?: string | null;
  translated?: string;
  frameBlob?: Blob;
  stream?: MediaStream | null;
  ocrText?: string;
  reply?: string;
  replyIdx?: number;
  audioUrl?: string;
  barged?: boolean;
  speech?: SpeechPipeline | null;
}

function downstreamHas(path: Path, kind: BlockKind): boolean {
  const L = getList(path);
  if (!L) return false;
  const scan = (items: Block[]): boolean =>
    items.some(x => x.k === kind || (x.children ? scan(x.children) : false));
  return scan(L.arr.slice(L.idx + 1));
}

export const lastRun: { branch: Record<string, boolean>; iters: Record<string, number> } =
  { branch: {}, iters: {} };

function scopeFor(ctx: RunCtx): Scope {
  return (name) => {
    if (name in ctx.vars) return ctx.vars[name];
    switch (name) {
      case 'transcript': return ctx.transcript ?? ctx.raw ?? '';
      case 'reply': return ctx.reply ?? '';
      case 'ocr': return ctx.ocrText ?? '';
      case 'translated': return ctx.translated ?? '';
      case 'barged': return !!ctx.barged;
      case 'endpointed': return ctx.vadSilence != null;
      case 'sttMs': return ctx.turn.stt ?? -1;
      case 'tok1Ms': return ctx.turn.tok1 ?? -1;
      case 'iter': return ctx.iter;
      default: return undefined;
    }
  };
}

const fmtVal = (v: Val): string =>
  typeof v === 'string' ? "'" + (v.length > 60 ? v.slice(0, 57) + '…' : v) + "'" : str(v);

function evalConfigExpr(b: Block, what: string): { src: string; get: (ctx: RunCtx) => Val } {
  const src = bcStr(b, 'expr');
  const p = parseExpr(src);
  if ('error' in p) throw new Error(what + ' expression — ' + p.error + ' (at ' + p.pos + ')');
  return { src, get: (ctx) => evalExpr(p.ast, scopeFor(ctx)) };
}

function pushMsg(msg: Msg): number {
  S.msgs.push(msg);
  bus.emit(msg.role === 'u' ? 'msg.user' : 'msg.assistant', { kind: msg.kind, text: msg.text });
  scrollChatSoon();
  return S.msgs.length - 1;
}

let liveMedia: MediaStream | null = null;
let pendingSends: string[] = [];

const mediaEnded = (): boolean =>
  !!liveMedia && liveMedia.getVideoTracks().every(tr => tr.readyState === 'ended');

function stopLiveMedia(): void {
  if (liveMedia) { stopTracks(liveMedia); liveMedia = null; }
}

async function waitComposer(): Promise<string | null> {
  S.runNotice = 'live session — type in the composer and send';
  scheduleRender();
  try {
    for (;;) {
      const queued = pendingSends.shift();
      if (queued !== undefined) return queued;
      if (runAbort) return null;
      await sleep(80);
    }
  } finally {
    if (S.runNotice && S.runNotice.startsWith('live session')) S.runNotice = null;
  }
}

interface SubStack { items: Block[]; base: Path }
function subscriberStacks(name: string): SubStack[] {
  const out: SubStack[] = [];
  const p0 = S.prog[0];
  if (p0 && p0.k === 'on' && tapOf(p0) === name) out.push({ items: S.prog, base: [] });
  S.stacks.forEach((st, i) => {
    const h = st.items[0];
    if (h && h.k === 'on' && tapOf(h) === name) out.push({ items: st.items, base: ['S', i] });
  });
  return out;
}

async function execTrigger(b: Block, ctx: RunCtx): Promise<void> {
  if (b.k === 'on') return;
  if (b.k === 'mic') {
    bus.emit('mic.start', {});
    setBusy('listening — speak, then pause ' + S.eouMs + ' ms to finish');
    setMicState('connecting');
    let rt: RealtimeSession | null = null;
    let liveIdx: number | null = null;
    let sawSpeech = false, sawSilence = false, wsPartialSeen = false;
    const model = sttModel();
    if (model) liveIdx = pushMsg({ role: 'u', kind: 'audio', text: '…', meta: 'mic' });
    const applyPartial = (text: string): void => {
      const um = liveIdx != null ? S.msgs[liveIdx] : undefined;
      if (um) { um.text = text; scrollChatSoon(); }
    };
    const rtDown = (): void => {
      if (!rt) return;
      const dead = rt;
      rt = null;
      dead.stop();
      wsPartialSeen = false;
      appendLog(pad('stt.rt') + 'realtime ws unavailable — falling back to chunked partials');
    };
    if (realtimeEnabled() && model) {
      rt = openRealtime(model, {
        partial: (text) => {
          if (!text) return;
          wsPartialSeen = true;
          applyPartial(text);
          bus.emit('stt.partial', { text, via: 'ws' });
        },
        speechStart: (atMs) => { sawSpeech = true; bus.emit('vad.speech', { atMs, via: 'server' }); },
        speechStop: (atMs) => { sawSilence = true; bus.emit('vad.silence', { atMs, via: 'server' }); },
        error: () => rtDown(),
        close: () => rtDown(),
      });
    }
    let soFar: Blob | null = null;
    const ticker = chunkedPartials({
      blob: () => soFar,
      wsSeen: () => wsPartialSeen,
      apply: applyPartial,
    });
    let got: Utterance;
    try {
      got = await recordUtterance({
        silenceMs: S.eouMs, maxMs: 20000,
        stopWhen: () => runAbort,
        onStream: (stream) => {
          setMicState('listening');
          if (rt) rt.start(stream).catch(rtDown);
        },
        onChunk: (blob) => { soFar = blob; },
      });
    } catch (e) {
      ticker.stop();
      if (rt) rt.stop();
      if (liveIdx != null) S.msgs.splice(liveIdx, 1);
      setMicState('idle');
      throw e;
    }
    ticker.stop();
    if (runAbort) {
      if (rt) rt.stop();
      if (liveIdx != null) S.msgs.splice(liveIdx, 1);
      setMicState('idle');
      setBusy(null);
      return;
    }
    if (rt && rt.live() && sawSpeech && !sawSilence) {
      rt.commit();
      const tGrace = performance.now();
      while (!sawSilence && !runAbort && performance.now() - tGrace < 1500) await sleep(50);
    }
    if (rt) rt.stop();
    setMicState('committing');
    bus.emit('mic.stop', { bytes: got.blob.size, ms: got.ms });
    if (!got.blob.size) {
      if (liveIdx != null) S.msgs.splice(liveIdx, 1);
      setMicState('idle');
      throw new Error('mic produced no audio');
    }
    ctx.audioBlob = got.blob;
    ctx.vadSilence = got.silence;
    const meta = 'mic · ' + (got.ms / 1000).toFixed(1) + ' s recorded';
    if (liveIdx != null) {
      ctx.userIdx = liveIdx;
      const um = S.msgs[liveIdx];
      if (um) { um.url = URL.createObjectURL(got.blob); um.blob = got.blob; um.meta = meta; }
    } else {
      ctx.userIdx = pushMsg({
        role: 'u', kind: 'audio', url: URL.createObjectURL(got.blob), blob: got.blob,
        text: '…', meta,
      });
    }
    attachAudioMeta(S.msgs[ctx.userIdx]);
    appendLog(pad('mic.stop') + (got.ms / 1000).toFixed(1) + ' s · ' + got.blob.size + ' bytes' +
      (got.silence != null ? ' · vad endpoint' : ' · max length'));
    setBusy(null);
  } else if (b.k === 'keys') {
    let text = (S.input || '').trim();
    if (!text) {
      const got = await waitComposer();
      if (got == null) throw new Error('session stopped while waiting for a composer send');
      text = got.trim();
    }
    if (!text) throw new Error('composer is empty — type the message in the conversation composer first');
    ctx.raw = text; ctx.transcript = text; S.input = '';
    pushMsg({ role: 'u', kind: 'text', text, meta: 'keys' });
  } else {
    if (liveMedia && mediaEnded()) stopLiveMedia();
    const media = liveMedia
      || (b.k === 'screen'
        ? await navigator.mediaDevices.getDisplayMedia({ video: true })
        : await navigator.mediaDevices.getUserMedia({ video: true }));
    liveMedia = media;
    ctx.stream = media;
    const frame = await grabFrame(media);
    ctx.frameBlob = frame.blob;
    pushMsg({
      role: 'u', kind: 'visual', src: frame.dataUrl,
      text: (b.k === 'screen' ? 'screen frame' : 'camera frame') + ' · ' + frame.w + '×' + frame.h,
      meta: b.k, file: b.k + '-frame.png',
    });
    bus.emit('frame.captured', { source: b.k, w: frame.w, h: frame.h });
    appendLog(pad('frame.captured') + b.k + ' · ' + frame.w + '×' + frame.h);
  }
}

async function exec(b: Block, path: Path, ctx: RunCtx): Promise<boolean> {
  const k = b.k;
  if (HATS[k]) { await execTrigger(b, ctx); return true; }
  ctx.ran = true;
  switch (k) {
    case 'eou': {
      if (ctx.vadSilence != null) {
        ctx.turn.eou = ctx.vadSilence;
        bus.emit('eou.detected', { silenceMs: ctx.vadSilence });
        appendLog(pad('eou.detected') + 'silence ' + ctx.vadSilence + ' ms (threshold ' + S.eouMs + ')');
      } else {
        appendLog(pad('eou.skip') + 'no vad endpoint this turn (no mic capture or max length hit)');
      }
      return true;
    }
    case 'stt': {
      if (!ctx.audioBlob) throw new Error('no audio captured upstream — needs a mic block');
      setBusy('transcribing — ' + (sttModel() || ''));
      const got = await transcribe(ctx.audioBlob);
      ctx.transcript = got.text;
      ctx.sttDetail = got.detail;
      ctx.turn.stt = got.ms;
      bus.emit('stt.final', { text: got.text, ms: got.ms });
      appendLog(pad('stt.final') + '"' + got.text.slice(0, 44) + '" · ' + got.ms + ' ms · ' + got.model);
      const um = ctx.userIdx != null ? S.msgs[ctx.userIdx] : undefined;
      if (um) {
        applySttMsg(um, got);
        if (S.sttFormat === 1 && got.detail) um.meta += ' · ' + got.detail;
      }
      setMicState('idle');
      setBusy(null);
      return true;
    }
    case 'translate': {
      const src = ctx.transcript || ctx.raw;
      if (!src) throw new Error('nothing to translate — needs text upstream');
      const model = chatModel();
      if (!model) throw new Error('no chat model available');
      setBusy('translating — ' + model);
      const got = await streamChat({
        model, messages: [
          { role: 'system', content: "Translate the user's message to English. Output only the translation, nothing else." },
          { role: 'user', content: src },
        ],
      });
      if (got.error) throw new Error(got.error);
      ctx.translated = got.text.trim();
      ctx.transcript = ctx.translated;
      bus.emit('translate.done', { ms: got.wall });
      appendLog(pad('translate.done') + '"' + ctx.translated.slice(0, 44) + '" · ' + got.wall + ' ms · ' + model);
      pushMsg({ role: 'a', kind: 'text', text: 'translation: ' + ctx.translated,
        meta: 'translate · ' + model.replace(/^.*\//, '') + ' · ' + got.wall + ' ms',
        thinking: got.reasoning || null,
        raw: { model, finish: got.finish || null, wall: got.wall, reasoning: got.reasoning || null, text: got.text, chunks: got.chunks } });
      setBusy(null);
      return true;
    }
    case 'ocr': {
      if (!ctx.frameBlob) throw new Error('no frame captured upstream — needs a screen or camera block');
      setBusy('ocr — /v1/ocr');
      const got = await ocrFrame(ctx.frameBlob);
      ctx.ocrText = got.text;
      bus.emit('ocr.done', { ms: got.ms, elements: got.elements, chars: got.text.length });
      appendLog(pad('ocr.done') + got.elements + ' elements · ' + got.text.length + ' chars · ' + got.ms + ' ms');
      pushMsg({
        role: 'a', kind: 'text',
        text: got.text ? 'ocr: ' + got.text.slice(0, 400) + (got.text.length > 400 ? '…' : '')
          : 'ocr: no text found (' + got.elements + ' layout elements)',
        meta: '/v1/ocr · ' + got.ms + ' ms',
      });
      setBusy(null);
      return true;
    }
    case 'agent': {
      const model = chatModel();
      if (!model) throw new Error(modelRefusal('chat') || 'no chat model available');
      const parts: string[] = [];
      if (ctx.ocrText) parts.push('Text read from the shared frame:\n' + ctx.ocrText.slice(0, 6000));
      parts.push(ctx.transcript || ctx.raw ||
        (ctx.ocrText ? 'Describe what the frame contains, based on the text above.' : 'Say hello in one short sentence.'));
      const msgs: ChatMessageIn[] = [];
      if (S.sysPrompt.trim()) msgs.push({ role: 'system', content: S.sysPrompt.trim() });
      msgs.push({ role: 'user', content: parts.join('\n\n') });
      setBusy('agent — ' + model);
      bus.emit('agent.start', { model });
      const idx = pushMsg({ role: 'a', kind: 'text', text: '', meta: model });
      if (downstreamHas(path, 'tts') && ttsModel()) {
        try { ctx.speech = speechPipeline({ autoplay: downstreamHas(path, 'spk') }); }
        catch { ctx.speech = null; }
      }
      const got = await streamChat({
        model, messages: msgs,
        onToken: (t) => {
          const m = S.msgs[idx];
          if (m) m.text = t;
          if (ctx.speech) ctx.speech.push(t);
          scrollChatSoon();
        },
        onReasoning: (t) => { const m = S.msgs[idx]; if (m) m.thinking = t; scrollChatSoon(); },
      });
      if (got.error) {
        if (ctx.speech) ctx.speech.stop();
        throw new Error(got.error);
      }
      if (ctx.speech) ctx.speech.end(got.text);
      ctx.reply = got.text;
      ctx.replyIdx = idx;
      ctx.turn.tok1 = got.tok1;
      const m = S.msgs[idx];
      if (m) {
        m.text = got.text || '(empty reply)';
        applyChatResult(m, got, model);
      }
      bus.emit('agent.done', { text: got.text });
      appendLog(pad('agent.done') + 'tok₁ ' + (got.tok1 != null ? got.tok1 : '—') + ' ms · ' + got.wall + ' ms · ' + model);
      setBusy(null);
      return true;
    }
    case 'tts': {
      const text = ctx.reply || ctx.translated || ctx.transcript || ctx.ocrText;
      if (!text) throw new Error('nothing to speak — needs text upstream');
      setBusy('synthesizing — ' + (ttsModel() || ''));
      let pipe = ctx.speech || null;
      if (!pipe) {
        pipe = speechPipeline({ autoplay: downstreamHas(path, 'spk') });
        ctx.speech = pipe;
        pipe.end(text);
      }
      const fm = await pipe.first();
      if (fm == null) {
        setBusy(null);
        if (runAbort) return false;
        throw new Error(pipe.error() || 'speech synthesis produced no audio');
      }
      ctx.turn.audio1 = fm;
      const attach = (res: SpeakResult | null): void => {
        if (!res) return;
        ctx.audioUrl = res.url;
        const rm = ctx.replyIdx != null ? S.msgs[ctx.replyIdx] : undefined;
        if (rm) {
          rm.kind = 'audio'; rm.url = res.url; rm.blob = res.blob; rm.peaks = null;
          rm.meta += ' · tts₁ ' + fm + ' ms · ' + res.model;
          attachAudioMeta(rm);
        } else {
          const idx = pushMsg({ role: 'a', kind: 'audio', url: res.url, blob: res.blob, text, meta: res.model + ' · tts₁ ' + fm + ' ms' });
          ctx.replyIdx = idx;
          attachAudioMeta(S.msgs[idx]);
        }
        scrollChatSoon();
      };
      if (pipe.autoplay) {
        void pipe.result.then(res => { attach(res); scheduleRender(); });
      } else {
        attach(await pipe.result);
      }
      appendLog(pad('tts.first') + fm + ' ms to first audio · ' + (ttsModel() || ''));
      setBusy(null);
      scrollChatSoon();
      return true;
    }
    case 'spk': {
      const pipe = ctx.speech && ctx.speech.autoplay ? ctx.speech : null;
      if (!pipe && ctx.speech) await ctx.speech.result;
      if (!pipe && !ctx.audioUrl) throw new Error('no synthesized audio — needs a speak block');
      const play = pipe
        ? { done: pipe.playbackDone, stop: () => pipe.stop() }
        : playForRun(ctx.audioUrl as string);
      bus.emit('spk.play', { streamed: !!pipe });
      appendLog(pad('spk.play') + 'playing' + (S.barge ? ' · barge-in armed' : ''));
      let mon = null;
      if (S.barge) {
        try { mon = await micMonitor(); }
        catch { mon = null; appendLog(pad('spk.note') + 'barge-in unavailable — mic permission denied'); }
      }
      if (mon) {
        setMicState('monitoring');
        bus.emit('mic.monitor', { on: true });
        const tPlay = performance.now();
        let over = 0, tOver = 0;
        const iv = setInterval(() => {
          if (performance.now() - tPlay < 400) return;
          if (mon.level() > SPEECH_RMS) { if (!over) tOver = performance.now(); over++; } else over = 0;
          if (over >= 3) {
            clearInterval(iv);
            ctx.barged = true;
            play.stop();
            ctx.turn.cancel = Math.round(performance.now() - tOver);
            bus.emit('tts.cancel', { reason: 'barge-in', ms: ctx.turn.cancel });
            appendLog(pad('tts.cancel') + 'barge-in — mic energy over threshold');
          }
        }, 60);
        await play.done;
        clearInterval(iv);
        mon.stop();
        setMicState('idle');
        bus.emit('mic.monitor', { on: false });
      } else {
        await play.done;
      }
      bus.emit('spk.done', { barged: !!ctx.barged });
      return true;
    }
    case 'store': {
      let wrote = 0;
      if (ctx.sttDetail && S.sttFormat === 2) { nurStore.add('transcript', ctx.sttDetail); wrote++; }
      else if (ctx.transcript && ctx.transcript !== ctx.reply) { nurStore.add('transcript', ctx.transcript); wrote++; }
      if (ctx.reply) { nurStore.add('reply', ctx.reply); wrote++; }
      if (ctx.ocrText) { nurStore.add('ocr', ctx.ocrText); wrote++; }
      const names = Object.keys(ctx.vars);
      if (names.length) {
        nurStore.add('vars', names.map(n => n + ' = ' + str(ctx.vars[n] as Val)).join('\n'));
        wrote++;
      }
      if (!wrote) { appendLog(pad('store.skip') + 'no artifacts to store this turn'); return true; }
      bus.emit('store.write', { count: wrote, total: nurStore.read().length });
      appendLog(pad('store.write') + wrote + ' artifact' + (wrote === 1 ? '' : 's') + ' · ' +
        (storage.kind === 'opfs' ? 'opfs' : 'localStorage') + ' (' + nurStore.read().length + ' total)');
      return true;
    }
    case 'cancel': {
      stopAudio();
      bus.emit('tts.cancel', { reason: 'cancel block' });
      appendLog(pad('cancel') + 'stopped playback (if any)');
      return true;
    }
    case 'if': {
      const src = ifExpr(b);
      const p = parseExpr(src);
      if ('error' in p) throw new Error('if expression — ' + p.error + ' (at ' + p.pos + ')');
      const v = evalExpr(p.ast, scopeFor(ctx));
      if (typeof v !== 'boolean')
        throw new Error('condition is ' + typeName(v) + ', not true/false — write a comparison like ' + src + ' > 0');
      lastRun.branch[pk(path)] = v;
      bus.emit('run.branch', { cond: src, taken: v });
      appendLog(pad('run.branch') + 'if ' + src + ' → ' + v);
      if (v) return runList(b.children || [], path, ctx);
      return true;
    }
    case 'set': case 'append': {
      const name = bcStr(b, 'var').trim();
      if (!VAR_RE.test(name)) throw new Error("'" + name + "' is not a usable variable name");
      const e = evalConfigExpr(b, k);
      if (k === 'append') {
        if (isBuiltin(name)) throw new Error("'" + name + "' is read-only — append needs your own variable");
        const prior = typeof ctx.vars[name] === 'string' ? ctx.vars[name] as string : '';
        const next = prior + (prior ? '\n' : '') + str(e.get(ctx));
        ctx.vars[name] = next;
        appendLog(pad('append') + name + ' = ' + fmtVal(next));
        bus.emit('run.var', { name, op: 'append' });
        return true;
      }
      const v = e.get(ctx);
      if (isWritableBuiltin(name)) {
        if (name === 'transcript') ctx.transcript = str(v);
        else ctx.reply = str(v);
      } else if (isBuiltin(name)) {
        throw new Error("'" + name + "' is read-only");
      } else ctx.vars[name] = v;
      appendLog(pad('set') + name + ' = ' + fmtVal(v));
      bus.emit('run.var', { name, op: 'set' });
      return true;
    }
    case 'repeat': {
      const n = clamp(bcNum(b, 'n', 3), 1, 20);
      const prev = ctx.iter;
      for (let i = 1; i <= n; i++) {
        if (runAbort) { ctx.iter = prev; return false; }
        ctx.iter = i;
        bus.emit('run.iter', { i, of: n });
        appendLog(pad('run.iter') + 'pass ' + i + ' of ' + n);
        if (!await runList(b.children || [], path, ctx)) { ctx.iter = prev; return false; }
      }
      ctx.iter = prev;
      lastRun.iters[pk(path)] = n;
      return true;
    }
    case 'wait': {
      const ms = clamp(bcNum(b, 'ms', 1000), 0, 30000);
      bus.emit('run.wait', { ms });
      appendLog(pad('run.wait') + ms + ' ms');
      const t0 = performance.now();
      while (performance.now() - t0 < ms) {
        if (runAbort) return false;
        await sleep(Math.min(100, ms - (performance.now() - t0)));
      }
      return true;
    }
    case 'http': {
      const scope = scopeFor(ctx);
      const urlT = bcStr(b, 'url').trim();
      if (!urlT) throw new Error('http needs a url — set it in the config panel');
      const url = templateStr(urlT, scope);
      const method = bcStr(b, 'method', 'GET') === 'POST' ? 'POST' : 'GET';
      const into = bcStr(b, 'into', 'http').trim() || 'http';
      if (!VAR_RE.test(into) || isBuiltin(into)) throw new Error("'" + into + "' is not a usable variable name");
      const headers: Record<string, string> = {};
      bcStr(b, 'headers').split('\n').forEach(ln => {
        const i = ln.indexOf(':');
        if (i > 0) headers[ln.slice(0, i).trim()] = ln.slice(i + 1).trim();
      });
      const init: RequestInit = { method, headers };
      if (method === 'POST') {
        if (!Object.keys(headers).some(h => h.toLowerCase() === 'content-type'))
          headers['Content-Type'] = 'text/plain';
        init.body = templateStr(bcStr(b, 'body'), scope);
      }
      setBusy('http — ' + method + ' ' + url.slice(0, 60));
      const ctl = new AbortController();
      init.signal = ctl.signal;
      const t0 = performance.now();
      const iv = setInterval(() => { if (runAbort || performance.now() - t0 > 15000) ctl.abort(); }, 100);
      let r: Response;
      try { r = await fetch(url, init); }
      catch {
        clearInterval(iv);
        setBusy(null);
        throw new Error('request failed — network, or the target does not allow browser (CORS) requests from this origin');
      }
      clearInterval(iv);
      const text = (await r.text()).slice(0, 65536);
      ctx.vars[into] = text;
      ctx.vars[into + 'Status'] = r.status;
      const ms = Math.round(performance.now() - t0);
      bus.emit('http.done', { status: r.status, ms, bytes: text.length });
      appendLog(pad('http.done') + method + ' ' + url.replace(/^https?:\/\//, '').slice(0, 40) +
        ' · ' + r.status + ' · ' + ms + ' ms · ' + text.length + ' bytes');
      setBusy(null);
      return true;
    }
    case 'for': {
      if (!ctx.stream) throw new Error('for-each-frame needs a screen or camera block (stream is closed)');
      const n = clamp(bcNum(b, 'frames', gv('frames')), 1, 10);
      const interval = 1000 / clamp(bcNum(b, 'fps', gv('fps')), 1, 5);
      const prev = ctx.iter;
      for (let i = 0; i < n; i++) {
        if (runAbort) { ctx.iter = prev; return false; }
        if (i > 0) {
          await sleep(interval);
          const frame = await grabFrame(ctx.stream);
          ctx.frameBlob = frame.blob;
          bus.emit('frame.captured', { i: i + 1, of: n });
        }
        ctx.iter = i + 1;
        bus.emit('run.iter', { i: i + 1, of: n });
        appendLog(pad('run.iter') + 'frame ' + (i + 1) + ' of ' + n);
        if (!await runList(b.children || [], path, ctx)) { ctx.iter = prev; return false; }
      }
      ctx.iter = prev;
      lastRun.iters[pk(path)] = n;
      return true;
    }
    default: return true;
  }
}

async function runNode(b: Block, path: Path, ctx: RunCtx): Promise<boolean> {
  if (runAbort) return false;
  const key = pk(path);
  S.runPath = key;
  scheduleRender();
  bus.emit('run.node.start', { node: b.k, path: key });
  const t0 = performance.now();
  let ok: boolean;
  try { ok = await exec(b, path, ctx); }
  catch (e) {
    const msg = errMsg(e);
    S.runError = key;
    S.runErrorMsg = msg;
    S.runNotice = msg;
    runAbort = true;
    bus.emit('run.node.error', { node: b.k, path: key, message: msg });
    appendLog(pad('run.error') + b.k + ' · ' + msg.slice(0, 70));
    setBusy(null);
    scheduleRender();
    return false;
  }
  bus.emit('run.node.end', { node: b.k, path: key, ms: Math.round(performance.now() - t0) });
  return ok !== false;
}

async function runList(items: Block[], base: Path, ctx: RunCtx): Promise<boolean> {
  for (let i = 0; i < items.length; i++) {
    if (runAbort) return false;
    const b = items[i];
    if (!b) continue;
    if (!await runNode(b, base.concat(i), ctx)) return false;
    const name = b.k === 'on' ? '' : tapOf(b);
    if (name && (ctx.tapDepth || 0) < 4) {
      for (const sub of subscriberStacks(name)) {
        if (runAbort) return false;
        const clone: RunCtx = { ...ctx, vars: { ...ctx.vars }, tapDepth: (ctx.tapDepth || 0) + 1 };
        bus.emit('run.tap', { name, blocks: countNodes(sub.items) });
        if (!await runList(sub.items, sub.base, clone)) return false;
      }
    }
  }
  return true;
}

export async function runPatch(): Promise<void> {
  if (running) { abortRun(); return; }
  if (getMode() !== 'live') {
    bus.emit('run.error', { message: 'backend unreachable' });
    appendLog(pad('run.error') + OFFLINE_MSG);
    flashNotice(OFFLINE_MSG);
    scheduleRender();
    return;
  }
  const root = hatRoot();
  if (!root) {
    bus.emit('run.error', { message: 'no hat-rooted stack' });
    appendLog(pad('run.error') + 'no hat-rooted stack — add a trigger block');
    flashNotice('no hat-rooted stack — add a trigger block');
    scheduleRender();
    return;
  }
  const ferr = flow().errs.values().next();
  if (!ferr.done) {
    bus.emit('run.error', { message: ferr.value });
    appendLog(pad('run.error') + 'ill-typed program — ' + ferr.value.slice(0, 70));
    flashNotice(ferr.value);
    scheduleRender();
    return;
  }
  running = true; runAbort = false;
  S.runError = null; S.runErrorMsg = null; S.runNotice = null; S.runPath = null;
  lastRun.branch = {}; lastRun.iters = {};
  await Promise.all([S.baseChat, S.baseStt, S.baseTts]
    .map(b => (b || '').trim()).filter(Boolean)
    .map(b => ensureBaseModels(b)));
  const t0 = performance.now();
  const hat = root.items[0] as Block;
  pendingSends = [];
  if (hat.k === 'keys')
    setComposerHook((text) => { pendingSends.push(text); return true; });
  bus.emit('run.start', {
    intent: S.intent, mode: getMode(), nodes: countNodes(root.items), live: true, source: hat.k,
  });
  appendLog(pad('run.start') + (root.base.length ? 'stack' : 'program') + ' · ' +
    countNodes(root.items) + ' nodes · live ' + hat.k + ' session');
  let elements = 0;
  try {
    for (;;) {
      if (runAbort) break;
      if (elements > 0 && liveMedia && mediaEnded()) {
        appendLog(pad('run.session') + 'capture ended — session complete');
        break;
      }
      if (elements > 0 && (hat.k === 'screen' || hat.k === 'cam')) {
        const interval = 1000 / clamp(bcNum(hat, 'fps', gv('fps')), 1, 5);
        const tw = performance.now();
        while (performance.now() - tw < interval) {
          if (runAbort) break;
          await sleep(Math.min(100, interval));
        }
        if (runAbort) break;
      }
      const ctx: RunCtx = {
        turn: { eou: null, tok1: null, audio1: null, stt: null, cancel: null, text: '' },
        vars: {}, iter: 0,
      };
      const ok = await runList(root.items, root.base, ctx);
      if (ctx.ran || ctx.raw || ctx.audioBlob) {
        ctx.turn.text = (ctx.transcript || ctx.raw || '').replace(/^“|”$/g, '');
        S.turns.push(ctx.turn);
        S.sel = S.turns.length - 1;
        bus.emit('turn.recorded', { turn: { ...ctx.turn }, index: S.turns.length - 1 });
        elements++;
        bus.emit('run.element', { n: elements, source: hat.k });
        appendLog(pad('run.element') + 'element ' + elements + ' done');
      }
      S.lastVars = { ...ctx.vars };
      if (!ok) break;
    }
  } catch {  }
  setComposerHook(null);
  pendingSends = [];
  stopLiveMedia();
  const ms = Math.round(performance.now() - t0);
  bus.emit('run.done', { ms, aborted: runAbort, error: S.runError, elements });
  appendLog(pad('run.done') + elements + ' element' + (elements === 1 ? '' : 's') + ' · ' + ms + ' ms' +
    (S.runError ? ' · with error' : runAbort ? ' · stopped' : ''));
  running = false; runAbort = false;
  S.runPath = null;
  setMicState('idle');
  setBusy(null);
  scheduleRender();
}

bus.on('session.start', () => {
  if (!running && hatRoot()) void runPatch();
});
bus.on('session.leave', () => abortRun());
