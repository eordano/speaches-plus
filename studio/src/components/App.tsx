import { useEffect } from 'react';
import { Outlet, useParams } from 'react-router';
import { setState, useAppState, type View } from '../state/store';
import { intentBySlug } from '../data';
import { navTo } from '../engine/nav';
import { applyIntent, leaveSession, restoreSessionById } from '../engine/session';
import Header from './Header';
import ChatView from './ChatView';
import PatchView from './PatchView';
import InspectorView from './InspectorView';
import { TranscriptModal, ViewerModal } from './Modals';

export default function App() {
  const S = useAppState();
  return (
    <div id="app">
      <Header />
      <Outlet />
      {S.txtModal != null && <TranscriptModal />}
      {S.viewer != null && <ViewerModal />}
    </div>
  );
}

export function useHomeSync(): void {
  const S = useAppState();
  useEffect(() => {
    if (S.screen === 'duet' && S.msgs.length) leaveSession();
    else if (S.screen !== 'intent') setState({ screen: 'intent', view: 'chat', sessionId: null });
  }, [S.screen, S.msgs.length]);
  useEffect(() => () => {
    if (S.notice) setState({ notice: null }, { silent: true });
  }, []);
}

export function PatchAuthoringRoute() {
  const { intent: slug } = useParams();
  const S = useAppState();
  const idx = slug ? intentBySlug(slug) : -1;
  useEffect(() => {
    if (idx < 0) {
      setState({ notice: 'that intent does not exist — pick one below' });
      void navTo('/', { replace: true });
      return;
    }
    if (S.screen === 'duet' && S.msgs.length) leaveSession();
    if (S.intent !== idx || S.screen !== 'patch') applyIntent(idx);
  }, [idx, S.screen, S.intent, S.msgs.length]);
  if (idx < 0 || S.intent !== idx || S.screen !== 'patch') return null;
  return <PatchView />;
}

export function SessionRoute({ view }: { view: View }) {
  const { sessionId } = useParams();
  const S = useAppState();
  useEffect(() => {
    if (!sessionId) return;
    if (S.sessionId === sessionId) {
      if (S.screen !== 'duet' || S.view !== view) setState({ screen: 'duet', view });
      return;
    }
    if (!S.hydrated) return;
    if (!restoreSessionById(sessionId, view)) {
      setState({ notice: 'that session is not stored in this browser — it may have expired' });
      void navTo('/', { replace: true });
    }
  }, [sessionId, view, S.sessionId, S.screen, S.view, S.hydrated]);
  if (S.sessionId !== sessionId || S.screen !== 'duet') return null;
  return view === 'chat' ? <ChatView /> : view === 'insp' ? <InspectorView /> : <PatchView />;
}
