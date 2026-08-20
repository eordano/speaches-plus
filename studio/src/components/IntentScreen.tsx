import { Link } from 'react-router';
import { useAppState } from '../state/store';
import { INTENTS } from '../data';
import { relTime } from '../lib/util';
import { deletePastSession, pickIntent, sessionUrl } from '../engine/session';

export default function IntentScreen() {
  const S = useAppState();
  return (
    <div className="intent-wrap" style={{ padding: '40px 0 60px' }}>
      <div className="intent-title" style={{ font: '700 32px var(--serif)', color: '#1A1A1A' }}>How can I act today?</div>
      {S.notice && (
        <div style={{ font: '400 11.5px var(--ui)', color: '#645C55', fontStyle: 'italic', marginTop: 10 }}>
          {S.notice}
        </div>
      )}

      <div className="intent-grid" style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: 14, marginTop: 30 }}>
        {INTENTS.map((it, i) => (
          <div
            key={it.name}
            className="hl card"
            style={{ padding: 18, cursor: 'pointer', boxShadow: '0 1px 2px rgba(20,16,12,.04)' }}
            onClick={() => { void pickIntent(i); }}
          >
            <div style={{ font: '600 17px var(--serif)', color: '#1A1A1A' }}>{it.name}</div>
            <div style={{ font: '400 11px var(--ui)', color: '#645C55', marginTop: 5, lineHeight: 1.5 }}>{it.sub}</div>
            <div style={{ font: '400 10px var(--ui)', color: '#2B2B2B', marginTop: 12 }}>{it.chain}</div>
          </div>
        ))}
      </div>

      {S.past.length > 0 && (
        <div style={{ marginTop: 44 }}>
          <div style={{ font: '400 10.5px var(--ui)', letterSpacing: '.1em', color: '#645C55' }}>PREVIOUS SESSIONS</div>
          <div style={{ display: 'flex', flexDirection: 'column', marginTop: 10 }}>
            {S.past.map((ps) => {
              const tn = Array.isArray(ps.turns) ? ps.turns.length : 0;
              return (
                <Link
                  key={ps.id}
                  className="hl sess-row"
                  to={sessionUrl(ps.id)}
                  style={{ display: 'flex', alignItems: 'baseline', gap: 14, padding: '11px 10px', cursor: 'pointer', borderRadius: 6, textDecoration: 'none', color: 'inherit' }}
                >
                  <span style={{ font: '600 14px var(--serif)', color: '#1A1A1A', flex: 'none' }}>{ps.title}</span>
                  <span className="ell sess-sum" style={{ font: '400 11px var(--ui)', color: '#645C55', flex: 1, minWidth: 0 }}>{ps.summary || ''}</span>
                  {ps.savedAt != null && (
                    <span data-saved-at="1" style={{ font: '400 10px var(--mono)', color: '#645C55', flex: 'none', minWidth: 64, textAlign: 'right' }}>{relTime(ps.savedAt)}</span>
                  )}
                  <span style={{ font: '400 10px var(--mono)', color: '#645C55', flex: 'none' }}>{tn} turn{tn === 1 ? '' : 's'}</span>
                  <span
                    className="hl sess-x"
                    title="remove this session"
                    style={{ font: '400 13px var(--ui)', color: '#645C55', flex: 'none', padding: '0 6px', borderRadius: 4, alignSelf: 'center' }}
                    onClick={(e) => { e.preventDefault(); e.stopPropagation(); deletePastSession(ps.id); }}
                  >×</span>
                </Link>
              );
            })}
          </div>
        </div>
      )}
    </div>
  );
}
