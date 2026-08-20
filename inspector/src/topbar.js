const $ = (s) => document.querySelector(s);

export function setupTopbar({ store, audio, timeline }) {
  const statePill   = $('#statePill');
  const stateText   = $('#stateText');
  const btnPlay     = $('#btnPlay');
  const btnStop     = $('#btnReplayStop');
  const speedSelect = $('#speedSelect');
  const channelsWrap = $('#audioChannelsWrap');
  const liveWrap    = $('#liveFollowWrap');
  const btnChMic    = $('#btnChMic');
  const btnChTts    = $('#btnChTts');
  const btnFollow   = $('#btnFollow');

  // ── mode pill (live / replay / paused) ─────────────────────────────────
  function refreshModePill() {
    const m = store.state.mode;
    statePill.classList.remove('state-live', 'state-paused', 'state-replay');
    if (m === 'live')   { statePill.classList.add('state-live');   stateText.textContent = 'Live'; }
    if (m === 'paused') { statePill.classList.add('state-paused'); stateText.textContent = 'Paused'; }
    if (m === 'replay') { statePill.classList.add('state-replay'); stateText.textContent = 'Replay'; }
  }

  // Live tail group is only meaningful while a live WS is open.
  function refreshLiveGroup() {
    const alive = store.state.wsAlive;
    liveWrap.classList.toggle('tb-hide', !alive);
    btnFollow.setAttribute('aria-pressed', store.state.followTail ? 'true' : 'false');
    btnFollow.disabled = !alive && !store.state.followTail;
    btnFollow.style.opacity = btnFollow.disabled ? '0.5' : '';
    btnFollow.style.cursor  = btnFollow.disabled ? 'not-allowed' : '';
  }

  // Play button flips Play<->Stop based on playback.active. There used to be
  // a separate #btnReplayStop next to it but the dual-Stop was confusing,
  // so we keep the element in the DOM (other code still references it) and
  // hide it permanently -- this single button is the affordance.
  function refreshPlayback() {
    const playing = !!store.state.playback.active;
    btnPlay.innerHTML = playing
      ? '■ Stop <span class="glyph">esc</span>'
      : '▶ Play <span class="glyph">space</span>';
    btnPlay.classList.toggle('btn-stop', playing);
    btnPlay.classList.toggle('btn-primary', !playing);
    btnStop.style.display = 'none';
    channelsWrap.style.display = playing ? '' : 'none';
    btnChMic.setAttribute('aria-pressed', store.state.playback.channels.mic ? 'true' : 'false');
    btnChTts.setAttribute('aria-pressed', store.state.playback.channels.tts ? 'true' : 'false');
  }

  // ── click wiring ───────────────────────────────────────────────────────
  // Play conductor:
  //   playing               -> stop
  //   live + tail-following -> peel tail off (so user can pick a moment),
  //                           but only when actually attached to a live WS.
  //                           Otherwise (historical session) start playing
  //                           immediately -- the first click should always
  //                           do something useful.
  btnPlay.addEventListener('click', () => {
    const s = store.state;
    if (s.playback.active) { audio.stop(); return; }
    if (s.followTail && store.transport.isAlive()) {
      store.setFollowTail(false);
      return;
    }
    // For historical replay, drop tail before starting.
    if (s.followTail) store.setFollowTail(false);
    audio.startLive();
  });

  btnStop.addEventListener('click', () => audio.stop());

  speedSelect.value = String(store.state.tweaks.replaySpeed || '1');
  speedSelect.addEventListener('change', () => {
    store.setTweak('replaySpeed', speedSelect.value);
  });

  btnChMic.addEventListener('click', () => {
    const next = !store.state.playback.channels.mic;
    store.setPlayback({ channels: Object.assign({}, store.state.playback.channels, { mic: next }) });
    audio.applyChannelGain('mic', next);
  });
  btnChTts.addEventListener('click', () => {
    const next = !store.state.playback.channels.tts;
    store.setPlayback({ channels: Object.assign({}, store.state.playback.channels, { tts: next }) });
    audio.applyChannelGain('tts', next);
  });

  btnFollow.addEventListener('click', () => {
    if (!store.state.followTail && !store.transport.isAlive()) return;
    const next = !store.state.followTail;
    store.setFollowTail(next);
    if (next) {
      if (store.state.playback.active) audio.stop();
      timeline.toEnd();
    }
  });

  // ── subscriptions ──────────────────────────────────────────────────────
  store.subscribe('mode', () => { refreshModePill(); refreshLiveGroup(); });
  store.subscribe('view', refreshLiveGroup);
  store.subscribe('playback', refreshPlayback);
  store.subscribe('tweaks', () => { speedSelect.value = String(store.state.tweaks.replaySpeed || '1'); });

  refreshModePill();
  refreshLiveGroup();
  refreshPlayback();
}
