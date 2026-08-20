import { Link } from 'react-router';
import { useAppState, type View } from '../state/store';
import { bus } from '../state/bus';
import { INTENTS, shortId } from '../data';
import { OFFLINE_MSG } from '../lib/util';
import { getMode, goHome, stepDuet, chatModel, ttsModel, sessionUrl } from '../engine/session';

const TABS: Array<[string, View]> = [['Conversation', 'chat'], ['Inspector', 'insp'], ['Patch', 'patch']];

export default function Header() {
  const S = useAppState();
  const liveMode = getMode() === 'live';

  return (
    <div className="hdr" style={{ display: 'flex', alignItems: 'center', gap: 18, flex: 'none' }}>
      <div
        style={{ font: '700 19px var(--serif)', color: 'var(--ink)', cursor: 'pointer', flex: 'none' }}
        onClick={() => { void goHome(); }}
      >
        nur <span style={{ fontWeight: 400 }}><i>studio</i></span>
      </div>

      <span
        title={liveMode ? '/v1 endpoints detected' : OFFLINE_MSG}
        style={{
          font: '400 9.5px var(--mono)',
          border: `1px solid ${liveMode ? 'var(--acc-speech)' : 'var(--border)'}`,
          color: liveMode ? 'var(--acc-speech)' : 'var(--muted)',
          borderRadius: 4, background: 'var(--bg)', padding: '2px 7px', flex: 'none',
        }}
      >{liveMode ? 'live' : 'offline'}</span>

      {S.screen === 'patch' ? (
        <>
          <div className="ell" style={{ font: '400 16px var(--ui)', color: 'var(--muted)', flex: 1, minWidth: 0 }}>
            {S.intent != null ? INTENTS[S.intent]?.name ?? '' : ''}
          </div>
          <Link
            className="hm"
            to="/"
            style={{ font: '400 12px var(--ui)', color: 'var(--muted)', cursor: 'pointer', flex: 'none', textDecoration: 'none' }}
          >← change intent</Link>
          {!liveMode && <span style={{ font: '400 11px var(--ui)', color: 'var(--muted)', fontStyle: 'italic', flex: 'none' }}>{OFFLINE_MSG}</span>}
          <span
            className={liveMode ? 'hd hdr-start' : 'hdr-start'}
            title={liveMode ? 'go live — starts running the patch' : OFFLINE_MSG}
            style={{ background: liveMode ? 'var(--ink)' : 'var(--well)', color: liveMode ? 'var(--bg)' : 'var(--muted)', borderRadius: 8, padding: '8px 18px', font: '600 12.5px var(--ui)', cursor: liveMode ? 'pointer' : 'default', flex: 'none' }}
            onClick={liveMode ? () => { void stepDuet(); } : undefined}
          >Start session</span>
        </>
      ) : S.screen === 'duet' ? (
        <>
          <div className="hdr-tabs" style={{ flex: 1, display: 'flex', justifyContent: 'center', alignItems: 'center', gap: 22, minWidth: 0 }}>
            {TABS.map(([label, v]) => {
              const active = S.view === v;
              return (
                <Link
                  key={v}
                  className="hdr-tab"
                  to={S.sessionId ? sessionUrl(S.sessionId, v) : '/'}
                  onClick={() => bus.emit('nav.view', { view: v })}
                  style={{
                    font: `${active ? '600' : '400'} 13px var(--ui)`,
                    color: active ? 'var(--ink)' : 'var(--muted)',
                    borderBottom: `2px solid ${active ? 'var(--ink)' : 'transparent'}`,
                    padding: '6px 0 4px', cursor: 'pointer', textDecoration: 'none',
                  }}
                >{label}</Link>
              );
            })}
          </div>
          <div className="ell" style={{ font: '400 11px var(--ui)', color: 'var(--muted)', minWidth: 0, flex: '0 1 auto', textAlign: 'right' }}>
            {(S.intent != null ? INTENTS[S.intent]?.name ?? '' : '') +
              (liveMode
                ? [chatModel(), ttsModel()].filter(Boolean).map(id => ' · ' + shortId(id)).join('')
                : ' · offline')}
          </div>
        </>
      ) : (
        <div style={{ flex: 1 }} />
      )}
    </div>
  );
}
