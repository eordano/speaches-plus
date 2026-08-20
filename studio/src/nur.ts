import {
  S, setState, subscribe, renderNow, scheduleRender, notify,
  type AppState, type View,
} from './state/store';
import { bus } from './state/bus';
import { D, INTENTS, progFor, KOKORO_VOICES, CUSTOMVOICE_SPEAKERS, ttsKind, voicesForModel, shortId, slugOf, intentBySlug } from './data';
import {
  clamp, dataUri, dlText, errMsg, fmtDur, getDl, icon, pad, playIcon, setDl,
  tint, OFFLINE_MSG, type DlFn,
} from './lib/util';
import {
  getMode, models, chatModel, sttModel, ttsModel, ttsVoice, perf, nurStore,
  loadPast, persistPast, appendLog, setBusy, scrollChatSoon, progHas,
  pickIntent, resumeSession, restoreSessionById, deletePastSession, stepDuet, goHome, leaveSession, sessionUrl,
  stopAudio, togglePlay, detectMode, streamChat, transcribe, speechPipeline, speechActive,
  doSend, liveReply, liveMicToggle, isRecording, micState, applyIntent,
  abortChat, chatStreaming,
} from './engine/session';
import { openRealtime, realtimeEnabled, realtimeUrl, rtStats } from './engine/realtime';
import {
  decodePeaks, recordUtterance, micMonitor, grabFrame, attachAudioMeta, peaksPoly,
} from './engine/media';
import * as patchEngine from './engine/patch';
import { parseExpr, evalExpr, templateStr, templateIdents, BUILTIN_NAMES, IF_PRESETS } from './engine/expr';
import { emitDsl, emitDoc, parseDsl, applyDsl, canonicalLine, beginDslBurst } from './engine/dsl';
import { runPatch, abortRun, lastRun } from './engine/runner';
import { ensureBaseModels, effModels, normBase } from './engine/session';
import { navTo } from './engine/nav';
import { storage as storageLayer } from './state/storage';

const storageSurface = {
  get kind() { return storageLayer.kind; },
  ready: storageLayer.ready,
  flush: storageLayer.flush,
  readPatch: storageLayer.debug.readPatch,
  readSession: storageLayer.debug.readSession,
  listSessionIds: storageLayer.debug.listSessionIds,
  hasAudio: storageLayer.debug.hasAudio,
  hasSessionDir: storageLayer.debug.hasSessionDir,
  readArtifacts: storageLayer.debug.readArtifacts,
};

const patch = {
  ...patchEngine,
  get drag() { return patchEngine.drag; },
  get snap() { return patchEngine.snap; },
  get dslValid() { return patchEngine.dslValid; },
  abortRun,
  parseExpr, evalExpr, templateStr, templateIdents, BUILTIN_NAMES, IF_PRESETS,
  emitDsl, emitDoc, parseDsl, applyDsl, canonicalLine, beginDslBurst,
  lastRun,
};

const nur = {
  booted: false,
  get mode() { return getMode(); },
  get state(): AppState { return S; },
  setState,
  subscribe,
  render: renderNow,
  scheduleRender,
  notify,
  bus,
  D, INTENTS, progFor, KOKORO_VOICES, CUSTOMVOICE_SPEAKERS,
  ttsKind, voicesForModel, shortId, slugOf, intentBySlug,
  icon, playIcon, tint, fmtDur, pad, clamp, errMsg, dataUri, dlText, OFFLINE_MSG,
  get dl(): DlFn { return getDl(); },
  set dl(fn: DlFn) { setDl(fn); },
  models, chatModel, sttModel, ttsModel, ttsVoice, perf,
  store: nurStore,
  storage: storageSurface,
  loadPast, persistPast, appendLog, setBusy, scrollChatSoon, progHas,
  pickIntent, resumeSession, restoreSessionById, deletePastSession, stepDuet, goHome, leaveSession,
  sessionUrl, applyIntent,
  navView: (v: View): Promise<void> => (S.sessionId ? navTo(sessionUrl(S.sessionId, v)) : Promise.resolve()),
  navTo,
  stopAudio, togglePlay, detectMode,
  ensureBaseModels, effModels, normBase,
  streamChat, transcribe, speechPipeline, speechActive, doSend, liveReply, liveMicToggle, isRecording, micState,
  abortChat, chatStreaming,
  realtime: { open: openRealtime, enabled: realtimeEnabled, url: realtimeUrl, stats: rtStats },
  decodePeaks, recordUtterance, micMonitor, grabFrame, attachAudioMeta, peaksPoly,
  patch,
  runPatch,
};

export type NurSurface = typeof nur;
declare global {
  interface Window { nur: NurSurface }
}
window.nur = nur;

export default nur;
