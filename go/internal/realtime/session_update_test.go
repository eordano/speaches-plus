package realtime

import (
	"encoding/json"
	"io"
	"log/slog"
	"testing"
	"time"

	"github.com/eordano/speaches-plus-go/internal/audio"
	"github.com/eordano/speaches-plus-go/internal/tts"
)

func newSessionUpdateTestPipeline(t *testing.T) *sessionPipeline {
	t.Helper()
	cfg := sessionConfig{}
	(&cfg).fillDefaults()
	p := &sessionPipeline{
		session: cfg,
		logger:  slog.New(slog.NewTextHandler(io.Discard, nil)),
		closed:  make(chan struct{}),
	}
	p.phase.startSession()
	return p
}

func ptrFloat32(v float32) *float32 { return &v }
func ptrInt(v int) *int             { return &v }
func ptrString(v string) *string    { return &v }

func TestApplySessionUpdate_ThresholdStored(t *testing.T) {
	p := newSessionUpdateTestPipeline(t)
	body := &sessionUpdateBody{
		TurnDetection: &turnDetectionBody{
			Threshold: ptrFloat32(0.7),
		},
	}
	if err := p.applySessionUpdate(body); err != nil {
		t.Fatalf("applySessionUpdate: %v", err)
	}
	if got := p.session.VADThreshold; got != 0.7 {
		t.Fatalf("VADThreshold not updated: want 0.7, got %v", got)
	}
}

func TestApplySessionUpdate_PrefixPaddingStored(t *testing.T) {
	p := newSessionUpdateTestPipeline(t)
	body := &sessionUpdateBody{
		TurnDetection: &turnDetectionBody{
			PrefixPaddingMs: ptrInt(150),
		},
	}
	if err := p.applySessionUpdate(body); err != nil {
		t.Fatalf("applySessionUpdate: %v", err)
	}
	if got := p.session.VADPrefixPaddingMs; got != 150 {
		t.Fatalf("VADPrefixPaddingMs not updated: want 150, got %d", got)
	}
}

func TestApplySessionUpdate_SilenceDurationReschedulesTimer(t *testing.T) {
	p := newSessionUpdateTestPipeline(t)
	cancelCh := make(chan struct{})
	p.timerMu.Lock()
	p.commitCancel = cancelCh
	p.commitItemID = "item_test"
	p.commitSamples = make(audio.MonoF32, 1)
	p.commitPartial = ""
	p.commitArmedAt = time.Now()
	p.commitDeadline = p.commitArmedAt.Add(1 * time.Second)
	p.commitHardCap = false
	p.commitTimer = time.AfterFunc(1*time.Hour, func() {})
	oldDeadline := p.commitDeadline
	p.timerMu.Unlock()

	body := &sessionUpdateBody{
		TurnDetection: &turnDetectionBody{
			SilenceDurationMs: ptrInt(200),
		},
	}
	if err := p.applySessionUpdate(body); err != nil {
		t.Fatalf("applySessionUpdate: %v", err)
	}
	if got := p.session.VADSilenceDurationMs; got != 200 {
		t.Fatalf("VADSilenceDurationMs not updated: want 200, got %d", got)
	}
	p.timerMu.Lock()
	newDeadline := p.commitDeadline
	timer := p.commitTimer
	p.timerMu.Unlock()
	if !newDeadline.Before(oldDeadline) {
		t.Fatalf("commitDeadline should have moved earlier: old=%v new=%v",
			oldDeadline, newDeadline)
	}
	if timer != nil {
		timer.Stop()
	}
}

func TestApplySessionUpdate_TypeNoneSuppressesCommitTimer(t *testing.T) {
	p := newSessionUpdateTestPipeline(t)
	cancelCh := make(chan struct{})
	p.timerMu.Lock()
	p.commitCancel = cancelCh
	p.commitItemID = "item_test"
	p.commitArmedAt = time.Now()
	p.commitDeadline = p.commitArmedAt.Add(1 * time.Second)
	p.commitTimer = time.AfterFunc(1*time.Hour, func() {})
	p.timerMu.Unlock()

	body := &sessionUpdateBody{
		TurnDetection: &turnDetectionBody{
			Type: ptrString("none"),
		},
	}
	if err := p.applySessionUpdate(body); err != nil {
		t.Fatalf("applySessionUpdate: %v", err)
	}
	if p.session.TurnDetectionType != "none" {
		t.Fatalf("TurnDetectionType not updated: got %q", p.session.TurnDetectionType)
	}
	p.timerMu.Lock()
	if p.commitTimer != nil {
		t.Fatalf("commit timer should have been cancelled; still set")
	}
	if p.commitCancel != nil {
		t.Fatalf("commit cancel chan should be nil; still set")
	}
	p.timerMu.Unlock()
}

func TestApplySessionUpdate_TypeNoneNoNewTimerArmed(t *testing.T) {
	p := newSessionUpdateTestPipeline(t)
	body := &sessionUpdateBody{
		TurnDetection: &turnDetectionBody{
			Type: ptrString("none"),
		},
	}
	if err := p.applySessionUpdate(body); err != nil {
		t.Fatalf("applySessionUpdate: %v", err)
	}
	cancelCh := make(chan struct{})
	p.timerMu.Lock()
	p.commitCancel = cancelCh
	p.timerMu.Unlock()
	p.eou = p.stubEOU
	p.runPartialAndScheduleCommit("item_test", make(audio.MonoF32, 1), cancelCh, time.Time{})
	p.timerMu.Lock()
	timerSet := p.commitTimer != nil
	p.timerMu.Unlock()
	if timerSet {
		t.Fatalf("commit timer should NOT be armed when turn_detection.type=none")
	}
}

func TestSessionConfig_EagerFieldDefaults(t *testing.T) {
	cfg := sessionConfig{}
	(&cfg).fillDefaults()
	if cfg.EagerMaxInflight != 1 {
		t.Fatalf("default EagerMaxInflight=1 want; got %d", cfg.EagerMaxInflight)
	}
	if cfg.EagerIntervalMs != 250 {
		t.Fatalf("default EagerIntervalMs=250 want; got %d", cfg.EagerIntervalMs)
	}
	if cfg.EagerPeriodicEnabled {
		t.Fatalf("default EagerPeriodicEnabled=false want; got %v", cfg.EagerPeriodicEnabled)
	}
	if cfg.TurnDetectionType != "server_vad" {
		t.Fatalf("default TurnDetectionType=server_vad want; got %q", cfg.TurnDetectionType)
	}
}

func TestApplySessionUpdate_AtomicValidationFailureLeavesStateUnchanged(t *testing.T) {
	p := newSessionUpdateTestPipeline(t)
	p.session.VADThreshold = 0.7
	body := &sessionUpdateBody{
		TurnDetection: &turnDetectionBody{
			Threshold:         ptrFloat32(0.5),
			SilenceDurationMs: ptrInt(999999),
		},
	}
	if err := p.applySessionUpdate(body); err == nil {
		t.Fatalf("expected validation failure")
	}
	if got := p.session.VADThreshold; got != 0.7 {
		t.Fatalf("VADThreshold must NOT have been applied: want 0.7, got %v", got)
	}
}

func TestApplySessionUpdate_EOUKindAccepted(t *testing.T) {
	p := newSessionUpdateTestPipeline(t)
	body := &sessionUpdateBody{
		TurnDetection: &turnDetectionBody{
			EOU: &eouBody{Kind: ptrString("heuristic")},
		},
	}
	if err := p.applySessionUpdate(body); err != nil {
		t.Fatalf("applySessionUpdate: %v", err)
	}
	if string(p.eouCfg.Kind) != "heuristic" {
		t.Fatalf("eou.kind not applied: got %q", p.eouCfg.Kind)
	}
}

func TestApplySessionUpdate_EOUKindUnknownRejected(t *testing.T) {
	p := newSessionUpdateTestPipeline(t)
	body := &sessionUpdateBody{
		TurnDetection: &turnDetectionBody{
			EOU: &eouBody{Kind: ptrString("magic")},
		},
	}
	if err := p.applySessionUpdate(body); err == nil {
		t.Fatalf("expected unknown kind to be rejected")
	}
}

func TestApplySessionUpdate_CurveKAcceptedAndApplied(t *testing.T) {
	p := newSessionUpdateTestPipeline(t)
	body := &sessionUpdateBody{
		TurnDetection: &turnDetectionBody{
			EOU: &eouBody{CurveK: ptrFloat32(8)},
		},
	}
	if err := p.applySessionUpdate(body); err != nil {
		t.Fatalf("applySessionUpdate: %v", err)
	}
	if p.session.CurveK != 8 {
		t.Fatalf("CurveK not applied: got %v", p.session.CurveK)
	}
}

func TestApplySessionUpdate_CurveKOutOfRangeRejected(t *testing.T) {
	p := newSessionUpdateTestPipeline(t)
	for _, v := range []float32{0, -1, 100} {
		body := &sessionUpdateBody{
			TurnDetection: &turnDetectionBody{
				EOU: &eouBody{CurveK: ptrFloat32(v)},
			},
		}
		if err := p.applySessionUpdate(body); err == nil {
			t.Fatalf("expected curve_k=%v to be rejected", v)
		}
	}
}

func TestApplySessionUpdate_ProcessScopedRejected(t *testing.T) {
	p := newSessionUpdateTestPipeline(t)
	for _, name := range []string{
		"vad_model",
		"chat_completion_base_url",
		"default_realtime_model",
	} {
		body := &sessionUpdateBody{ProcessScoped: []string{name}}
		if err := p.applySessionUpdate(body); err == nil {
			t.Fatalf("expected process-scoped %q to be rejected", name)
		}
	}
}

func TestApplySessionUpdate_ContextTurnsApplied(t *testing.T) {
	p := newSessionUpdateTestPipeline(t)
	body := &sessionUpdateBody{
		TurnDetection: &turnDetectionBody{
			EOU: &eouBody{ContextTurns: ptrInt(2)},
		},
	}
	if err := p.applySessionUpdate(body); err != nil {
		t.Fatalf("applySessionUpdate: %v", err)
	}
	if p.session.EOUContextTurns != 2 {
		t.Fatalf("EOUContextTurns not applied: got %d", p.session.EOUContextTurns)
	}
	if p.eouCfg.ContextTurns != 2 {
		t.Fatalf("eouCfg.ContextTurns not applied: got %d", p.eouCfg.ContextTurns)
	}
}

func TestApplySessionUpdate_PThresholdApplied(t *testing.T) {
	p := newSessionUpdateTestPipeline(t)
	body := &sessionUpdateBody{
		TurnDetection: &turnDetectionBody{
			EOU: &eouBody{PThreshold: ptrFloat32(0.6)},
		},
	}
	if err := p.applySessionUpdate(body); err != nil {
		t.Fatalf("applySessionUpdate: %v", err)
	}
	if p.eouCfg.PThreshold != 0.6 {
		t.Fatalf("PThreshold not applied: got %v", p.eouCfg.PThreshold)
	}
}

func TestApplySessionUpdate_GlobalsApplied(t *testing.T) {
	p := newSessionUpdateTestPipeline(t)
	body := &sessionUpdateBody{
		SessionMaxDurationS:        ptrInt(900),
		MinSpeechMs:                ptrInt(50),
		MinSpeechForResponseMs:     ptrInt(400),
		SealedBufferRetentionCount: ptrInt(8),
	}
	if err := p.applySessionUpdate(body); err != nil {
		t.Fatalf("applySessionUpdate: %v", err)
	}
	if p.session.SessionMaxDurSec != 900 {
		t.Fatalf("SessionMaxDurSec not applied: got %d", p.session.SessionMaxDurSec)
	}
	if p.session.MinSpeechMs != 50 {
		t.Fatalf("MinSpeechMs not applied: got %d", p.session.MinSpeechMs)
	}
	if p.session.MinSpeechForResponseMs != 400 {
		t.Fatalf("MinSpeechForResponseMs not applied: got %d", p.session.MinSpeechForResponseMs)
	}
	if p.session.SealedBufferRetentionCount != 8 {
		t.Fatalf("SealedBufferRetentionCount not applied: got %d", p.session.SealedBufferRetentionCount)
	}
}

func TestApplySessionUpdate_GlobalsValidated(t *testing.T) {
	for _, tc := range []struct {
		name string
		body *sessionUpdateBody
	}{
		{"session_max_duration_s=0", &sessionUpdateBody{SessionMaxDurationS: ptrInt(0)}},
		{"min_speech_ms<0", &sessionUpdateBody{MinSpeechMs: ptrInt(-1)}},
		{"min_speech_for_response_ms<0", &sessionUpdateBody{MinSpeechForResponseMs: ptrInt(-1)}},
		{"sealed_buffer_retention_count<0", &sessionUpdateBody{SealedBufferRetentionCount: ptrInt(-1)}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			p := newSessionUpdateTestPipeline(t)
			if err := p.applySessionUpdate(tc.body); err == nil {
				t.Fatalf("expected rejection")
			}
		})
	}
}

func TestSessionConfigDefault_EOUContextTurns(t *testing.T) {
	cfg := sessionConfig{}
	cfg.fillDefaults()
	if cfg.EOUContextTurns != 4 {
		t.Fatalf("RFC v3 §6.4.2 default: want 4, got %d", cfg.EOUContextTurns)
	}
}

func TestErrorTypeFor_SttFailedIsServerError(t *testing.T) {
	if got := errorTypeFor("stt_failed"); got != "server_error" {
		t.Fatalf("RFC v3 §10.5: stt_failed should be server_error, got %q", got)
	}
}

func TestPhase_EagerMaxInflightCapsAtOne(t *testing.T) {
	var s phaseState
	s.startSession()
	s.setEagerMaxInflight(1)
	if _, ok := s.onPredictedDispatch("p1", "i1", 0.8, &eagerRunner{}); !ok {
		t.Fatalf("first dispatch should succeed")
	}
	if _, ok := s.onPredictedDispatch("p2", "i2", 0.9, &eagerRunner{}); ok {
		t.Fatalf("second concurrent dispatch must be rejected")
	}
	if _, _, _, ok := s.onPredictedRollback(); !ok {
		t.Fatalf("rollback should succeed")
	}
	if _, ok := s.onPredictedDispatch("p3", "i3", 0.85, &eagerRunner{}); !ok {
		t.Fatalf("post-rollback dispatch should succeed; cap freed")
	}
}

type speedRecordingTTS struct {
	voices []string
	speeds []float32
}

func (s *speedRecordingTTS) Synthesize(_, voice, _ string, speed float32) (tts.Audio, error) {
	s.voices = append(s.voices, voice)
	s.speeds = append(s.speeds, speed)
	return tts.Audio{}, nil
}

func (s *speedRecordingTTS) Close() error { return nil }

type discardOutbound struct{}

func (discardOutbound) WriteAudio(_ audio.MonoF32, _ int) error { return nil }
func (discardOutbound) PlayedMs() int64                         { return 0 }
func (discardOutbound) ResetPlayedMs()                          {}

func synthesizeOnce(t *testing.T, p *sessionPipeline) *speedRecordingTTS {
	t.Helper()
	rec := &speedRecordingTTS{}
	p.server.TTS = rec
	voice := p.session.Voice
	if voice == "" {
		voice = defaultVoice
	}
	speed := p.session.Speed
	if speed == 0 {
		speed = defaultTTSSpeed
	}
	var firstAudio time.Time
	var ttsCount int
	flush := p.makeFlushChunk("resp_t", "item_t", 1, discardOutbound{}, voice, speed,
		&firstAudio, &ttsCount, time.Now(), time.Now())
	flush("hello there")
	if len(rec.speeds) != 1 {
		t.Fatalf("Synthesize calls: want 1, got %d", len(rec.speeds))
	}
	return rec
}

func TestApplySessionUpdate_SpeedDefault(t *testing.T) {
	p := newSessionUpdateTestPipeline(t)
	if got := p.session.Speed; got != 1.0 {
		t.Fatalf("default Speed: want 1.0, got %v", got)
	}
	body := &sessionUpdateBody{Voice: ptrString("af_bella")}
	if err := p.applySessionUpdate(body); err != nil {
		t.Fatalf("applySessionUpdate: %v", err)
	}
	if got := p.session.Speed; got != 1.0 {
		t.Fatalf("Speed after speed-less update: want 1.0, got %v", got)
	}
	rec := synthesizeOnce(t, p)
	if rec.speeds[0] != 1.0 {
		t.Fatalf("Synthesize speed: want 1.0, got %v", rec.speeds[0])
	}
}

func TestApplySessionUpdate_SpeedStoredAndReachesSynthesize(t *testing.T) {
	p := newSessionUpdateTestPipeline(t)
	body := &sessionUpdateBody{Speed: ptrFloat32(1.7)}
	if err := p.applySessionUpdate(body); err != nil {
		t.Fatalf("applySessionUpdate: %v", err)
	}
	if got := p.session.Speed; got != 1.7 {
		t.Fatalf("Speed not updated: want 1.7, got %v", got)
	}
	rec := synthesizeOnce(t, p)
	if rec.speeds[0] != 1.7 {
		t.Fatalf("Synthesize speed: want 1.7, got %v", rec.speeds[0])
	}
}

func TestApplySessionUpdate_SpeedClampedLow(t *testing.T) {
	p := newSessionUpdateTestPipeline(t)
	body := &sessionUpdateBody{Speed: ptrFloat32(0.1)}
	if err := p.applySessionUpdate(body); err != nil {
		t.Fatalf("applySessionUpdate: %v", err)
	}
	if got := p.session.Speed; got != 0.5 {
		t.Fatalf("Speed not clamped: want 0.5, got %v", got)
	}
}

func TestApplySessionUpdate_SpeedClampedHigh(t *testing.T) {
	p := newSessionUpdateTestPipeline(t)
	body := &sessionUpdateBody{Speed: ptrFloat32(3.0)}
	if err := p.applySessionUpdate(body); err != nil {
		t.Fatalf("applySessionUpdate: %v", err)
	}
	if got := p.session.Speed; got != 2.0 {
		t.Fatalf("Speed not clamped: want 2.0, got %v", got)
	}
}

func TestApplySessionUpdate_SpeedOnlyLeavesVoiceUnchanged(t *testing.T) {
	p := newSessionUpdateTestPipeline(t)
	if err := p.applySessionUpdate(&sessionUpdateBody{Voice: ptrString("af_bella")}); err != nil {
		t.Fatalf("applySessionUpdate: %v", err)
	}
	if err := p.applySessionUpdate(&sessionUpdateBody{Speed: ptrFloat32(1.7)}); err != nil {
		t.Fatalf("applySessionUpdate: %v", err)
	}
	if got := p.session.Voice; got != "af_bella" {
		t.Fatalf("Voice changed by speed-only update: want af_bella, got %q", got)
	}
	if got := p.session.Speed; got != 1.7 {
		t.Fatalf("Speed not updated: want 1.7, got %v", got)
	}
	rec := synthesizeOnce(t, p)
	if rec.voices[0] != "af_bella" || rec.speeds[0] != 1.7 {
		t.Fatalf("Synthesize args: want (af_bella, 1.7), got (%q, %v)", rec.voices[0], rec.speeds[0])
	}
}

func TestSessionUpdateBody_SpeedDecoded(t *testing.T) {
	var b sessionUpdateBody
	if err := json.Unmarshal([]byte(`{"voice":"af_bella","speed":1.7}`), &b); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	if b.Speed == nil || *b.Speed != 1.7 {
		t.Fatalf("speed not decoded: got %v", b.Speed)
	}
	if b.Voice == nil || *b.Voice != "af_bella" {
		t.Fatalf("voice not decoded: got %v", b.Voice)
	}
}
