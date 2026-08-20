import type { Block, BlockKind } from './state/store';

export const D = {
  dl:     'M7 1.5v7M4 6l3 3 3-3M2.5 11.5h9',
  doc:    'M3 1.5h5l3 3v8H3zM8 1.5v3h3M5 7.5h4M5 9.5h4',
  expand: 'M2.5 2.5h9v9h-9zM5.5 8.5 2.5 11.5M8.5 5.5l3-3M8.5 5.5H11M8.5 5.5V3M5.5 8.5H3M5.5 8.5V11',
  info:   'M7 1.5a5.5 5.5 0 1 0 0 11a5.5 5.5 0 1 0 0-11M7 6.5V10M7 3.9v1',
  dup:    'M4.5 4.5h8v8h-8zM2.5 9.5h-1v-8h8v1',
  fork:   'M1.7 3a1.8 1.8 0 1 0 3.6 0a1.8 1.8 0 1 0-3.6 0M8.7 3a1.8 1.8 0 1 0 3.6 0a1.8 1.8 0 1 0-3.6 0M5.2 11a1.8 1.8 0 1 0 3.6 0a1.8 1.8 0 1 0-3.6 0M3.5 4.8v1.4a2 2 0 0 0 2 2h3a2 2 0 0 0 2-2V4.8M7 8.2v1',
  trash:  'M2 3.5h10M5.5 3.5V2h3v1.5M3.5 3.5 4 12.5h6l.5-9M5.8 5.5v5M8.2 5.5v5',
  mic:    'M7 1a2 2 0 0 0-2 2v3.5a2 2 0 0 0 4 0V3a2 2 0 0 0-2-2M2.5 7a4.5 4.5 0 0 0 9 0M7 11.5V13',
  cam:    'M1.5 4.5h2.6l1.2-1.7h3.4l1.2 1.7h2.6v7h-11zM7 5.9a2.3 2.3 0 1 0 0 4.6a2.3 2.3 0 1 0 0-4.6',
} as const;

export interface Intent { name: string; sub: string; chain: string }

export const INTENTS: Intent[] = [
  { name: 'Voice assistant loop',        sub: 'Speak; get a spoken reply. Speaking over playback cancels it.', chain: 'mic → eou → stt → agent → tts → speaker' },
  { name: 'Screenshare copilot',         sub: 'Share a screen frame; OCR it; the agent answers about it aloud.', chain: 'screen → ocr → agent → tts → speaker' },
  { name: 'Camera document capture',     sub: 'Camera frame → OCR → the agent summarizes; result is stored.', chain: 'camera → ocr → agent → store' },
  { name: 'Meeting notes',               sub: 'Record; transcribe with speaker diarization; store the transcript.', chain: 'mic → stt (diarized) → store' },
  { name: 'Type-to-speech bench',        sub: 'Type text; hear it spoken in the selected model and voice.', chain: 'keys → tts → speaker' },
  { name: 'Translation notes',           sub: 'Speak any language; the English translation lands in the store.', chain: 'mic → stt → translate → store' },
  { name: 'Screen reader',               sub: 'OCR what is on screen and read it out loud.', chain: 'screen → ocr → tts → speaker' },
  { name: 'Narrated frame',              sub: 'Capture a screen frame plus narration; store the transcript.', chain: 'screen + mic → stt → store' },
  { name: 'Re-voice bench',              sub: 'Record speech; translate to English; replay in a preset voice.', chain: 'mic → stt → translate → tts → speaker' },
  { name: 'Voice memo to agent',         sub: 'Speak a request; the streamed reply is stored with the transcript.', chain: 'mic → eou → stt → agent → store' },
  { name: 'Frame-loop scribe',           sub: 'Several camera frames OCR’d and summarized in a loop; stored.', chain: 'camera → for each frame → ocr → agent → store' },
  { name: 'Text chat, spoken replies',   sub: 'Type in the composer; replies stream in and are spoken aloud.', chain: 'keys → agent → tts → speaker' },
];

const B = (k: BlockKind): Block => ({ k });
const Bc = (k: BlockKind, c: Record<string, string | number>): Block => ({ k, c });
const Cc = (k: BlockKind, c: Record<string, string | number>, ch: Block[]): Block => ({ k, c, children: ch });
const PROGS: Array<() => Block[]> = [
  () => [B('mic'), B('eou'), B('stt'), B('agent'), B('tts'), B('spk')],
  () => [B('screen'), B('ocr'), B('agent'), B('tts'), B('spk')],
  () => [B('cam'), B('ocr'), B('agent'), B('store')],
  () => [B('mic'), B('stt'), B('store')],
  () => [B('keys'), B('tts'), B('spk')],
  () => [B('mic'), B('stt'), B('translate'), B('store')],
  () => [B('screen'), B('ocr'), B('tts'), B('spk')],
  () => [B('screen'), B('mic'), B('stt'), B('store')],
  () => [B('mic'), B('stt'), B('translate'), B('tts'), B('spk')],
  () => [B('mic'), B('eou'), B('stt'), B('agent'), B('store')],
  () => [
    B('cam'),
    Cc('for', { frames: 4, fps: 2 }, [B('ocr'), Bc('append', { var: 'notes', expr: 'ocr' })]),
    Bc('set', { var: 'transcript', expr: "'Summarize what these frames showed:\\n' + notes" }),
    B('agent'), B('store'),
  ],
  () => [B('keys'), B('agent'), B('tts'), B('spk')],
];
export const progFor = (i: number): Block[] => {
  const f = PROGS[i];
  return f ? f() : [];
};

export const slugOf = (name: string): string =>
  name.toLowerCase().replace(/[^a-z0-9]+/g, '-').replace(/^-+|-+$/g, '');
export const intentBySlug = (slug: string): number =>
  INTENTS.findIndex(it => slugOf(it.name) === slug);

export const KOKORO_VOICES = ['af_heart', 'af_bella', 'af_sky', 'af_nova', 'af_nicole', 'af_sarah'];
export const CUSTOMVOICE_SPEAKERS = ['Ryan', 'Aiden', 'Vivian', 'Serena', 'Uncle_Fu', 'Dylan', 'Eric', 'Ono_Anna'];
export type TtsKind = 'none' | 'design' | 'preset' | 'kokoro' | 'other';
export const ttsKind = (id: string | null): TtsKind =>
  !id ? 'none' :
  /voicedesign/i.test(id) ? 'design' :
  /customvoice/i.test(id) ? 'preset' :
  /kokoro/i.test(id) ? 'kokoro' : 'other';
export const voicesForModel = (id: string | null): string[] | null => {
  const k = ttsKind(id);
  return k === 'kokoro' ? KOKORO_VOICES : k === 'preset' ? CUSTOMVOICE_SPEAKERS : null;
};
export const shortId = (id: string | null): string => (id ? String(id).replace(/^.*\//, '') : 'default');
