import { useEffect, useRef } from 'react';
import type { CSSProperties } from 'react';
import { setState, useAppState, type AppState, type Msg } from '../state/store';
import { D } from '../data';
import { dl, dlText, icon, playIcon, OFFLINE_MSG } from '../lib/util';
import { renderMd } from '../lib/md';
import { peaksPoly } from '../engine/media';
import {
  getMode, isRecording, doSend, liveMicToggle, micState, togglePlay, consumeScrollFlag,
  abortChat, chatStreaming, speechActive, stopAudio, scanShot,
} from '../engine/session';
import { running, abortRun } from '../engine/runner';

function AudioCard({ m, i, S }: { m: Msg; i: number; S: AppState }) {
  const playing = S.playId === i;
  const wave = `<svg width="170" height="18" viewBox="0 0 170 18">${m.peaks
    ? `<polyline points="${peaksPoly(m.peaks, 170, 18)}" fill="none" style="stroke:var(--acc-events)" stroke-width="1.1"/>`
    : '<line x1="0" y1="9" x2="170" y2="9" style="stroke:var(--border)" stroke-width="1"/>'}</svg>`;
  return (
    <div className="card" style={{ display: 'flex', alignItems: 'center', gap: 10, padding: '8px 10px', position: 'relative', overflow: 'hidden' }}>
      <div
        data-prog={i}
        style={{ position: 'absolute', left: 0, top: 0, bottom: 0, background: 'var(--hover)', transition: 'width .2s linear', width: `${playing ? S.playT * 100 : 0}%` }}
      />
      <span
        className="hl"
        style={{ width: 28, height: 28, flex: 'none', border: '1px solid var(--border)', borderRadius: '50%', background: 'var(--bg)', display: 'flex', alignItems: 'center', justifyContent: 'center', cursor: 'pointer', position: 'relative' }}
        dangerouslySetInnerHTML={{ __html: playIcon(playing, 10, 'var(--ink-body)') }}
        onClick={() => togglePlay(i)}
      />
      <span
        style={{ position: 'relative', display: 'flex' }}
        title={m.peaks ? 'waveform (decoded)' : 'decoding audio…'}
        dangerouslySetInnerHTML={{ __html: wave }}
      />
      <span style={{ font: '400 9.5px var(--mono)', color: 'var(--muted)', position: 'relative' }}>{m.durLabel || '…'}</span>
      <span
        className="ho"
        title="download audio"
        style={{ display: 'flex', alignItems: 'center', position: 'relative', padding: 3, cursor: 'pointer' }}
        dangerouslySetInnerHTML={{ __html: icon(D.dl, 13, 'var(--muted)', 1.3) }}
        onClick={() => { if (m.url) dl(m.url, m.file || (m.role === 'a' ? 'assistant-turn.wav' : 'user-turn.webm')); }}
      />
    </div>
  );
}

function VisualCard({ m, i }: { m: Msg; i: number }) {
  return (
    <div
      style={{ width: 240, height: 140, border: '1px solid var(--border)', borderRadius: 4, background: 'var(--well)', cursor: 'pointer', position: 'relative', overflow: 'hidden' }}
      onClick={() => setState({ viewer: i })}
    >
      <div style={{ position: 'absolute', inset: 0, backgroundImage: `url("${m.src ?? ''}")`, backgroundSize: 'cover', backgroundPosition: 'center' }} />
    </div>
  );
}

function MsgDetails({ m }: { m: Msg }) {
  const chips: Array<[string, string, string, string | undefined]> = [];
  if (m.text) chips.push(['⇩ reply .txt', 'reply.txt', m.text, undefined]);
  if (m.thinking) chips.push(['⇩ thinking .txt', 'thinking.txt', m.thinking, undefined]);
  if (m.text && m.thinking) chips.push(['⇩ both .txt', 'turn.txt', '## thinking\n\n' + m.thinking + '\n\n## reply\n\n' + m.text, undefined]);
  if (m.raw) chips.push(['⇩ raw .json', 'response.json', JSON.stringify(m.raw, null, 2), 'application/json']);
  return (
    <div className="well" style={{ padding: 12, marginTop: 6, maxWidth: 540, display: 'flex', flexDirection: 'column', gap: 8 }}>
      {m.thinking && (
        <>
          <div className="cap1">THINKING</div>
          <div style={{ font: 'italic 400 12.5px var(--serif)', lineHeight: 1.55, color: 'var(--ink-body)', whiteSpace: 'pre-wrap', maxHeight: 220, overflowY: 'auto' }}>{m.thinking}</div>
        </>
      )}
      {chips.length > 0 && (
        <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6 }}>
          {chips.map(([label, name, text, mime]) => (
            <span
              key={name}
              className="hl tchip"
              title={'download ' + name}
              style={{ font: '400 9.5px var(--mono)', color: 'var(--muted)' }}
              onClick={() => dlText(text, name, mime)}
            >{label}</span>
          ))}
        </div>
      )}
    </div>
  );
}

function Message({ m, i, S, pending }: { m: Msg; i: number; S: AppState; pending: boolean }) {
  const isUser = m.role === 'u';
  const kind = (m.kind === 'audio' && m.url) ? 'audio' : (m.kind === 'visual' && m.src) ? 'visual' : 'text';
  const pendingEmpty = pending && kind === 'text' && !m.text && !m.thinking;
  const bubbleStyle = isUser
    ? { background: 'var(--well)', border: '1px solid var(--border)', borderRadius: '8px 8px 4px 8px', padding: '10px 13px', maxWidth: 460 }
    : { maxWidth: 540 };
  const chipStyle = { display: 'flex', alignItems: 'center', flex: 'none' } as const;
  return (
    <div style={{ display: 'flex', flexDirection: 'column', alignItems: isUser ? 'flex-end' : 'flex-start' }}>
      <div className={isUser ? 'bub-u' : 'bub-a'} style={bubbleStyle}>
        {kind === 'audio' ? (
          <>
            <AudioCard m={m} i={i} S={S} />
            {isUser
              ? <div style={{ font: '400 13.5px var(--serif)', lineHeight: 1.5, marginTop: 8, whiteSpace: 'pre-wrap' }}>{m.text || ''}</div>
              : <div style={{ font: '400 13.5px var(--serif)', lineHeight: 1.5, marginTop: 8 }}>{renderMd(m.text || '')}</div>}
          </>
        ) : kind === 'visual' ? (
          <>
            <VisualCard m={m} i={i} />
            <div style={{ font: '400 13.5px var(--serif)', lineHeight: 1.5, marginTop: 8 }}>{m.text || ''}</div>
          </>
        ) : (
          <>
            {m.thinking && (!m.text || S.showThinking) && (
              <>
                <div className="cap1" style={{ marginBottom: 4 }}>THINKING</div>
                <div style={{ font: 'italic 400 12.5px var(--serif)', lineHeight: 1.55, color: 'var(--muted)', whiteSpace: 'pre-wrap', maxHeight: 180, overflowY: 'auto', marginBottom: m.text ? 8 : 0 }}>{m.thinking}</div>
              </>
            )}
            {pendingEmpty ? (
              <div data-pending-busy="1" style={{ font: '400 11px var(--ui)', color: 'var(--muted)', fontStyle: 'italic' }}>{S.busy}</div>
            ) : isUser ? (
              <div style={{ font: '400 14px var(--serif)', lineHeight: 1.5, whiteSpace: 'pre-wrap' }}>{m.text || ''}</div>
            ) : (
              <div style={{ font: '400 14px var(--serif)', lineHeight: 1.5 }}>{renderMd(m.text || '')}</div>
            )}
          </>
        )}
      </div>

      {!pendingEmpty && <div style={{ display: 'flex', gap: 6, marginTop: 5, font: '400 9.5px var(--ui)', alignItems: 'center' }}>
        {kind === 'audio' && m.text && (
          <span className="hl tchip" title="transcript" style={chipStyle}
            dangerouslySetInnerHTML={{ __html: icon(D.doc, 12, 'var(--muted)', 1.3) }}
            onClick={() => setState({ txtModal: i })} />
        )}
        {kind === 'visual' && (
          <>
            <span className="hl tchip" title="full view" style={chipStyle}
              dangerouslySetInnerHTML={{ __html: icon(D.expand, 12, 'var(--muted)', 1.3) }}
              onClick={() => setState({ viewer: i })} />
            <span className="hl tchip" title="download frame" style={chipStyle}
              dangerouslySetInnerHTML={{ __html: icon(D.dl, 12, 'var(--muted)', 1.3) }}
              onClick={() => { if (m.src) dl(m.src, m.file || 'frame.png'); }} />
          </>
        )}
        <span className="hl tchip" title="turn details" style={chipStyle}
          dangerouslySetInnerHTML={{ __html: icon(D.info, 12, 'var(--muted)', 1.3) }}
          onClick={() => setState(st => ({ msgInfo: { ...st.msgInfo, [i]: !st.msgInfo[i] } }))} />
        {S.msgInfo[i] && <span style={{ color: 'var(--muted)' }}>{m.meta || ''}</span>}
      </div>}
      {S.msgInfo[i] && (m.thinking || m.raw) && <MsgDetails m={m} />}
    </div>
  );
}

export function Composer({ S }: { S: AppState }) {
  if (getMode() !== 'live') {
    return (
      <div className="composer" style={{ padding: '12px 0 16px', display: 'flex', alignItems: 'center', gap: 8, flex: 'none' }}>
        <div className="well" style={{ flex: 1, height: 36, display: 'flex', alignItems: 'center', padding: '0 12px', font: '400 12px var(--ui)', color: 'var(--muted)', fontStyle: 'italic' }}>
          {OFFLINE_MSG}
        </div>
      </div>
    );
  }
  const recording = isRecording();
  const busy = S.busy;
  const mic = micState();
  const streaming = chatStreaming();
  const stoppable = running || speechActive();
  return (
    <div className="composer" style={{ position: 'relative', padding: '12px 0 16px', display: 'flex', alignItems: 'center', gap: 8, flex: 'none' }}>
      {mic !== 'idle' ? (
        <span className="cap mic-cap">
          {mic === 'committing' ? 'transcribing…' : mic === 'monitoring' ? 'mic armed — speak to interrupt' : 'listening…'}
        </span>
      ) : S.runNotice ? (
        <span className="cap mic-cap">{S.runNotice}</span>
      ) : null}
      {stoppable && (
        <span
          className="hl"
          data-stop-session="1"
          title={running ? 'stop the live session' : 'stop the spoken reply'}
          style={{ width: 36, height: 36, border: '1px solid var(--acc-danger)', borderRadius: 8, background: 'var(--bg)', display: 'flex', alignItems: 'center', justifyContent: 'center', cursor: 'pointer', font: '600 13px var(--ui)', color: 'var(--acc-danger)', flex: 'none' }}
          onClick={() => { if (running) abortRun(); else stopAudio(); }}
        >■</span>
      )}
      <span
        className={busy ? 'mic-btn' : 'hl mic-btn'}
        data-state={mic}
        title={busy ? busy : recording ? 'stop recording' : 'record a voice turn'}
        style={{ width: 36, height: 36, border: '1px solid var(--border)', borderRadius: 8, background: 'var(--bg)', display: 'flex', alignItems: 'center', justifyContent: 'center', cursor: busy ? 'default' : 'pointer', opacity: busy ? 0.45 : 1, flex: 'none' }}
        dangerouslySetInnerHTML={{ __html: icon(D.mic, 14, recording ? 'var(--acc-danger)' : 'var(--acc-events)', 1.35) }}
        onClick={() => { if (!S.busy) void liveMicToggle(); }}
      />
      <span
        className={busy ? 'cam-btn' : 'hl cam-btn'}
        title={busy ? busy : 'scan — one camera shot; the text lands in the composer'}
        style={{ width: 36, height: 36, border: '1px solid var(--border)', borderRadius: 8, background: 'var(--bg)', display: 'flex', alignItems: 'center', justifyContent: 'center', cursor: busy ? 'default' : 'pointer', opacity: busy ? 0.45 : 1, flex: 'none' }}
        dangerouslySetInnerHTML={{ __html: icon(D.cam, 14, 'var(--acc-events)', 1.35) }}
        onClick={() => { if (!S.busy) void scanShot(); }}
      />
      <input
        data-keepfocus="composer"
        placeholder="Type a message, or record with the mic"
        className="inp"
        style={{ flex: 1, height: 36, padding: '0 12px', font: '400 13px var(--serif)' }}
        value={S.input}
        onChange={(e) => setState({ input: e.target.value })}
        onKeyDown={(e) => { if (e.key === 'Enter' && S.input.trim() && !S.busy) doSend(S.input.trim()); }}
      />
      <span
        className={busy && !streaming ? 'send-btn' : 'hd send-btn'}
        title={streaming ? 'stop this reply — keeps the text streamed so far' : busy || undefined}
        style={{ height: 36, padding: '0 16px', borderRadius: 8, background: 'var(--ink)', color: 'var(--bg)', display: 'flex', alignItems: 'center', font: '600 12px var(--ui)', cursor: busy && !streaming ? 'default' : 'pointer', opacity: busy && !streaming ? 0.45 : 1, flex: 'none' }}
        onClick={() => { if (streaming) abortChat(); else if (S.input.trim() && !S.busy) doSend(S.input.trim()); }}
      >{streaming ? 'Stop' : 'Send'}</span>
    </div>
  );
}

export function MessageList({ S, style }: { S: AppState; style?: CSSProperties }) {
  const scrollRef = useRef<HTMLDivElement>(null);
  const firstScroll = useRef(true);
  const pinned = useRef(true);
  useEffect(() => {
    if (!consumeScrollFlag()) return;
    if (firstScroll.current) { pinned.current = true; firstScroll.current = false; }
    if (!pinned.current) return;
    const el = scrollRef.current;
    if (el) el.scrollTop = el.scrollHeight;
    setTimeout(() => {
      const el2 = scrollRef.current;
      if (el2 && pinned.current) el2.scrollTop = el2.scrollHeight;
    }, 90);
  });
  const last = S.msgs[S.msgs.length - 1];
  const pendingIdx = S.busy && last && last.role === 'a' && last.kind === 'text' && !last.text && !last.thinking
    ? S.msgs.length - 1 : null;
  return (
    <div
      ref={scrollRef}
      data-chat-scroll="1"
      onScroll={(e) => {
        const el = e.currentTarget;
        pinned.current = el.scrollHeight - el.scrollTop - el.clientHeight < 48;
      }}
      style={{ flex: 1, minHeight: 0, padding: '24px 0', display: 'flex', flexDirection: 'column', gap: 18, overflowY: 'auto', ...style }}
    >
      {S.msgs.map((m, i) => <Message key={i} m={m} i={i} S={S} pending={i === pendingIdx} />)}
      {S.busy && pendingIdx == null && <div style={{ font: '400 11px var(--ui)', color: 'var(--muted)', fontStyle: 'italic' }}>{S.busy}</div>}
    </div>
  );
}

export default function ChatView() {
  const S = useAppState();
  return (
    <div style={{ flex: 1, minHeight: 280, display: 'flex', flexDirection: 'column', minWidth: 0 }}>
      <div style={{ flex: 1, display: 'flex', flexDirection: 'column', width: 680, maxWidth: '100%', margin: '0 auto', minHeight: 0 }}>
        <MessageList S={S} />
        <Composer S={S} />
      </div>
    </div>
  );
}
