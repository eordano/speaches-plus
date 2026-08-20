import { useEffect, useRef, type CSSProperties, type ReactNode } from 'react';
import { setState, useAppState } from '../state/store';
import { dataUri } from '../lib/util';

function Scrim({ onClose, children, width, pad, gap }: {
  onClose: () => void; children: ReactNode; width: number; pad: number; gap: number;
}) {
  const cardRef = useRef<HTMLDivElement>(null);
  const closeRef = useRef(onClose);
  closeRef.current = onClose;
  useEffect(() => {
    const prev = document.activeElement instanceof HTMLElement ? document.activeElement : null;
    cardRef.current?.focus();
    const onKey = (e: KeyboardEvent): void => {
      if (e.key === 'Escape') { e.stopPropagation(); closeRef.current(); }
    };
    window.addEventListener('keydown', onKey, true);
    return () => {
      window.removeEventListener('keydown', onKey, true);
      if (prev && document.contains(prev)) prev.focus();
    };
  }, []);
  const cardStyle: CSSProperties = {
    width, maxWidth: '92vw', background: 'var(--panel)', border: '1px solid var(--border)',
    borderRadius: 16, boxShadow: 'var(--shadow-modal)', padding: pad, cursor: 'default',
    display: 'flex', flexDirection: 'column', gap, outline: 'none',
  };
  return (
    <div
      style={{ position: 'fixed', inset: 0, background: 'rgba(20,16,12,.40)', display: 'flex', alignItems: 'center', justifyContent: 'center', zIndex: 20 }}
      onClick={onClose}
    >
      <div ref={cardRef} tabIndex={-1} className="modal-card" style={cardStyle} onClick={(e) => e.stopPropagation()}>
        {children}
      </div>
    </div>
  );
}

export function TranscriptModal() {
  const S = useAppState();
  const m = S.txtModal != null ? S.msgs[S.txtModal] : undefined;
  if (!m) return null;
  const close = () => setState({ txtModal: null });
  return (
    <Scrim onClose={close} width={520} pad={22} gap={14}>
      <div style={{ font: '600 16px var(--serif)', color: 'var(--ink)' }}>Transcript</div>
      <div className="well" style={{ padding: 12, maxHeight: 300, overflowY: 'auto', font: '400 13px var(--serif)', lineHeight: 1.55, whiteSpace: 'pre-wrap' }}>{m.text || ''}</div>
      <div style={{ display: 'flex', justifyContent: 'flex-end', gap: 8 }}>
        <span
          className="hl"
          style={{ color: 'var(--ink-body)', border: '1px solid var(--border)', borderRadius: 8, padding: '7px 14px', background: 'var(--bg)', font: '400 12px var(--ui)', cursor: 'pointer' }}
          onClick={close}
        >Cancel</span>
        <a
          className="hd"
          download="transcript.txt"
          href={dataUri(m.text || '')}
          style={{ background: 'var(--ink)', color: 'var(--bg)', borderRadius: 8, padding: '7px 16px', font: '600 12px var(--ui)', cursor: 'pointer', textDecoration: 'none' }}
          onClick={close}
        >Save</a>
      </div>
    </Scrim>
  );
}

export function ViewerModal() {
  const S = useAppState();
  const m = S.viewer != null ? S.msgs[S.viewer] : undefined;
  if (!m || !m.src) return null;
  const close = () => setState({ viewer: null });
  return (
    <Scrim onClose={close} width={640} pad={20} gap={12}>
      <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
        <div className="ell" style={{ font: '600 15px var(--serif)', color: 'var(--ink)', flex: 1, minWidth: 0 }}>{m.file || 'captured frame'}</div>
        <span
          className="hl tchip"
          style={{ font: '400 11px var(--ui)', color: 'var(--muted)' }}
          onClick={close}
        >close</span>
      </div>
      <div className="well" style={{ height: 350, backgroundImage: `url("${m.src}")`, backgroundSize: 'contain', backgroundPosition: 'center', backgroundRepeat: 'no-repeat' }} />
      <div style={{ display: 'flex', justifyContent: 'flex-end' }}>
        <a
          className="hl tchip"
          download={m.file || 'frame.png'}
          href={m.src}
          style={{ font: '400 11px var(--ui)', color: 'var(--muted)', textDecoration: 'none' }}
        >download</a>
      </div>
    </Scrim>
  );
}
