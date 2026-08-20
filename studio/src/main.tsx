import { createRoot } from 'react-dom/client';
import { createBrowserRouter, RouterProvider, Navigate } from 'react-router';
import './styles.css';

if (localStorage.getItem('nur-theme') === 'paper') document.documentElement.dataset.theme = 'paper';
import nur from './nur';
import { bus } from './state/bus';
import { setNavigator } from './engine/nav';
import { loadPast, detectMode } from './engine/session';
import App, { PatchAuthoringRoute, SessionRoute } from './components/App';
import HomeRoute from './components/HomeRoute';

const router = createBrowserRouter([
  {
    path: '/',
    element: <App />,
    children: [
      { index: true, element: <HomeRoute /> },
      { path: 'patch/:intent', element: <PatchAuthoringRoute /> },
      { path: 's/:sessionId/chat', element: <SessionRoute view="chat" /> },
      { path: 's/:sessionId/inspector', element: <SessionRoute view="insp" /> },
      { path: 's/:sessionId/patch', element: <SessionRoute view="patch" /> },
      { path: '*', element: <Navigate to="/" replace /> },
    ],
  },
]);

setNavigator(async (to, opts) => { await router.navigate(to, opts); });

const vv = window.visualViewport;
if (vv) {
  const applyVvh = (): void => {
    document.documentElement.style.setProperty('--vvh', vv.height + 'px');
  };
  vv.addEventListener('resize', applyVvh);
  applyVvh();
}

void loadPast();
bus.emit('app.boot', { at: Date.now(), href: location.href });

const rootEl = document.getElementById('root');
if (rootEl) {
  createRoot(rootEl).render(<RouterProvider router={router} />);
  nur.booted = true;
  void detectMode();
} else {
  document.body.textContent = 'boot error: #root missing';
}
