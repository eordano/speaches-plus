package realtime

import (
	"context"
	"io"
	"log/slog"
	"sync"
	"testing"
	"time"

	"github.com/eordano/speaches-plus-go/internal/audio"
	"github.com/eordano/speaches-plus-go/internal/eou"
)

func newPipelineWithEOU(t *testing.T, model eou.Model) *sessionPipeline {
	t.Helper()
	cfg := sessionConfig{
		EOUMinDelayMs:   500,
		EOUMaxDelayMs:   3000,
		Language:        "en",
		EOUFailureP:     1.0,
		EOUFailureDelay: "min",
	}
	p := &sessionPipeline{session: cfg, logger: slog.New(slog.NewTextHandler(io.Discard, nil))}
	p.eouModel = model
	p.eouCfg = eou.Config{
		Kind:               eou.KindHeuristic,
		MinDelayMs:         cfg.EOUMinDelayMs,
		MaxDelayMs:         cfg.EOUMaxDelayMs,
		InferenceTimeoutMs: 100,
		Languages:          eou.DefaultLanguages(),
	}
	p.eou = p.runEOU
	return p
}

func TestRunEOU_HeuristicDrivesDelayDownwardOnTerminator(t *testing.T) {
	p := newPipelineWithEOU(t, eou.NewHeuristic())
	v := p.callEOU(context.Background(), "I'm done.", nil, time.Time{})
	if v.delayMs > 1000 {
		t.Fatalf("strong terminator must yield short delay; got %d ms", v.delayMs)
	}
	if v.score < 0.9 {
		t.Fatalf("strong terminator must yield score>=0.9; got %f", v.score)
	}
}

func TestRunEOU_HeuristicDrivesDelayUpwardOnHesitation(t *testing.T) {
	p := newPipelineWithEOU(t, eou.NewHeuristic())
	v := p.callEOU(context.Background(), "Wait, um", nil, time.Time{})
	if v.delayMs < 2000 {
		t.Fatalf("hesitation must yield long delay; got %d ms", v.delayMs)
	}
	if v.score > 0.2 {
		t.Fatalf("hesitation must yield low score; got %f", v.score)
	}
}

func TestRunEOU_HandlesNilModel(t *testing.T) {
	cfg := sessionConfig{EOUMinDelayMs: 500, EOUMaxDelayMs: 3000}
	p := &sessionPipeline{session: cfg, logger: slog.New(slog.NewTextHandler(io.Discard, nil))}
	p.eou = p.runEOU
	v := p.callEOU(context.Background(), "irrelevant", nil, time.Time{})
	if v.delayMs != 500 {
		t.Fatalf("nil model must use min_delay; got %d", v.delayMs)
	}
}

type errorModel struct{}

func (errorModel) Predict(ctx context.Context, req eou.Request) (eou.Verdict, error) {
	return eou.Verdict{}, io.ErrUnexpectedEOF
}
func (errorModel) Close() error { return nil }

func TestRunEOU_FallsBackOnError(t *testing.T) {
	p := newPipelineWithEOU(t, errorModel{})
	v := p.callEOU(context.Background(), "doesn't matter", nil, time.Time{})
	if v.delayMs != 500 {
		t.Fatalf("inference error must fall back to min_delay (§6.10); got %d", v.delayMs)
	}
}

type slowModel struct{}

func (slowModel) Predict(ctx context.Context, req eou.Request) (eou.Verdict, error) {
	<-ctx.Done()
	return eou.Verdict{}, ctx.Err()
}
func (slowModel) Close() error { return nil }

func TestRunEOU_TimeoutGivesMinDelay(t *testing.T) {
	p := newPipelineWithEOU(t, slowModel{})
	v := p.callEOU(context.Background(), "won't return in time", nil, time.Time{})
	if v.delayMs != 500 {
		t.Fatalf("§6.5: timeout must yield min_delay_ms=500; got %d", v.delayMs)
	}
	if v.score != 1.0 {
		t.Fatalf("§6.5: timeout must yield score=1.0 (fast-commit default); got %v", v.score)
	}
	if v.hardCapFired {
		t.Fatalf("EOU timeout (no hard-cap deadline) must not flag hardCapFired; got phase=%q", v.phase)
	}
}

func TestRunEOU_TimeoutWithSlowCommitOptIn(t *testing.T) {
	p := newPipelineWithEOU(t, slowModel{})
	p.session.EOUFailureP = 0.0
	p.session.EOUFailureDelay = "max"
	v := p.callEOU(context.Background(), "won't return in time", nil, time.Time{})
	if v.delayMs != 3000 {
		t.Fatalf("§6.5 slow-commit opt-in: timeout must yield max_delay_ms=3000; got %d", v.delayMs)
	}
	if v.score != 0.0 {
		t.Fatalf("§6.5 slow-commit opt-in: timeout must yield score=0.0; got %v", v.score)
	}
}

type recordingInspector struct {
	mu     sync.Mutex
	events []recordedEvent
}

type recordedEvent struct {
	name  string
	attrs []any
}

func (r *recordingInspector) Emit(event string, attrs ...any) {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.events = append(r.events, recordedEvent{name: event, attrs: append([]any(nil), attrs...)})
}

func (r *recordingInspector) byName(name string) []recordedEvent {
	r.mu.Lock()
	defer r.mu.Unlock()
	out := make([]recordedEvent, 0, len(r.events))
	for _, e := range r.events {
		if e.name == name {
			out = append(out, e)
		}
	}
	return out
}

func attrValue(e recordedEvent, key string) (any, bool) {
	for i := 0; i+1 < len(e.attrs); i += 2 {
		if k, ok := e.attrs[i].(string); ok && k == key {
			return e.attrs[i+1], true
		}
	}
	return nil, false
}

type blockingModel struct {
	block time.Duration
}

func (b blockingModel) Predict(ctx context.Context, req eou.Request) (eou.Verdict, error) {
	select {
	case <-time.After(b.block):
		return eou.Verdict{Score: 0.5}, nil
	case <-ctx.Done():
		return eou.Verdict{}, ctx.Err()
	}
}
func (blockingModel) Close() error { return nil }

func TestRunEOU_HardCapFiresDuringEOU(t *testing.T) {
	cfg := sessionConfig{
		EOUMinDelayMs: 500,
		EOUMaxDelayMs: 3000,
		Language:      "en",
		HardCapMs:     80,
	}
	p := &sessionPipeline{session: cfg, logger: slog.New(slog.NewTextHandler(io.Discard, nil))}
	p.eouModel = blockingModel{block: 5 * time.Second}
	p.eouCfg = eou.Config{
		Kind:               eou.KindHeuristic,
		MinDelayMs:         cfg.EOUMinDelayMs,
		MaxDelayMs:         cfg.EOUMaxDelayMs,
		InferenceTimeoutMs: 5000,
		HardCapMs:          80,
		Languages:          eou.DefaultLanguages(),
	}
	insp := &recordingInspector{}
	p.inspector = insp
	p.eou = p.runEOU

	deadline := time.Now().Add(80 * time.Millisecond)
	t0 := time.Now()
	v := p.callEOU(context.Background(), "wedged", nil, deadline)
	elapsed := time.Since(t0)

	if !v.hardCapFired {
		t.Fatalf("expected hardCapFired=true on stalled EOU; got verdict=%+v", v)
	}
	if v.phase != "during_eou" {
		t.Fatalf("expected phase=during_eou; got %q", v.phase)
	}
	if elapsed > 200*time.Millisecond {
		t.Fatalf("hard cap should fire near deadline (~80ms); elapsed=%v", elapsed)
	}
	if elapsed < 50*time.Millisecond {
		t.Fatalf("hard cap fired too early; elapsed=%v", elapsed)
	}
	hcEvents := insp.byName("eou.hard_cap_fired")
	if len(hcEvents) != 1 {
		t.Fatalf("expected 1 eou.hard_cap_fired event; got %d (%+v)", len(hcEvents), hcEvents)
	}
	phase, ok := attrValue(hcEvents[0], "phase")
	if !ok || phase != "during_eou" {
		t.Fatalf("expected phase=during_eou attr; got %v ok=%v", phase, ok)
	}
	if _, ok := attrValue(hcEvents[0], "score"); ok {
		t.Fatalf("RFC v3 §6.7 hard_cap_fired during_eou: score MUST be absent (no verdict yet)")
	}
}

type quickModel struct {
	score float32
}

func (q quickModel) Predict(_ context.Context, _ eou.Request) (eou.Verdict, error) {
	return eou.Verdict{Score: q.score}, nil
}
func (quickModel) Close() error { return nil }

func TestRunPartialAndScheduleCommit_HardCapFiresDuringWait(t *testing.T) {
	cfg := sessionConfig{
		EOUMinDelayMs:          500,
		EOUMaxDelayMs:          3000,
		Language:               "en",
		HardCapMs:              60,
		MinSpeechMs:            100,
		MinSpeechForResponseMs: 600,
		TurnDetectionType:      "server_vad",
	}
	p := &sessionPipeline{session: cfg, logger: slog.New(slog.NewTextHandler(io.Discard, nil))}
	p.eouModel = quickModel{score: 0.0}
	p.eouCfg = eou.Config{
		Kind:               eou.KindHeuristic,
		MinDelayMs:         cfg.EOUMinDelayMs,
		MaxDelayMs:         cfg.EOUMaxDelayMs,
		InferenceTimeoutMs: 100,
		HardCapMs:          cfg.HardCapMs,
		Languages:          eou.DefaultLanguages(),
	}
	insp := &recordingInspector{}
	p.inspector = insp
	p.eou = p.runEOU
	p.closed = make(chan struct{})
	p.turnDone = make(chan struct{})

	samples := make(audio.MonoF32, 1600)
	cancelCh := make(chan struct{})
	p.timerMu.Lock()
	p.commitCancel = cancelCh
	p.timerMu.Unlock()

	deadline := time.Now().Add(60 * time.Millisecond)
	done := make(chan struct{})
	go func() {
		defer close(done)
		p.runPartialAndScheduleCommit("item_test", samples, cancelCh, deadline)
	}()

	select {
	case <-done:
	case <-time.After(2 * time.Second):
		t.Fatalf("runPartialAndScheduleCommit did not return in time")
	}

	deadlineWait := time.Now().Add(500 * time.Millisecond)
	for time.Now().Before(deadlineWait) {
		hc := insp.byName("eou.hard_cap_fired")
		if len(hc) >= 1 {
			break
		}
		time.Sleep(5 * time.Millisecond)
	}
	hc := insp.byName("eou.hard_cap_fired")
	if len(hc) == 0 {
		t.Fatalf("expected eou.hard_cap_fired emit; got none (events=%v)", insp.events)
	}
	phase, _ := attrValue(hc[0], "phase")
	if phase != "during_wait" {
		t.Fatalf("expected phase=during_wait; got %v", phase)
	}
}

func TestRunEOU_RegressV2RustNoHardCap(t *testing.T) {
	cfg := sessionConfig{
		EOUMinDelayMs: 500,
		EOUMaxDelayMs: 3000,
		Language:      "en",
		HardCapMs:     80,
	}
	p := &sessionPipeline{session: cfg, logger: slog.New(slog.NewTextHandler(io.Discard, nil))}
	p.eouModel = blockingModel{block: 5 * time.Second}
	p.eouCfg = eou.Config{
		Kind:               eou.KindHeuristic,
		MinDelayMs:         cfg.EOUMinDelayMs,
		MaxDelayMs:         cfg.EOUMaxDelayMs,
		InferenceTimeoutMs: 5000,
		HardCapMs:          80,
		Languages:          eou.DefaultLanguages(),
	}
	p.inspector = &recordingInspector{}
	p.eou = p.runEOU
	deadline := time.Now().Add(80 * time.Millisecond)
	v := p.callEOU(context.Background(), "x", nil, deadline)
	if !v.hardCapFired || v.phase != "during_eou" {
		t.Fatalf("regress-v2-rust-no-hard-cap: hard cap MUST fire when EOU stalls past silence_hard_cap_ms; got %+v", v)
	}
}

func TestRunEOU_RegressV2GoClampNotTimer(t *testing.T) {
	cfg := sessionConfig{
		EOUMinDelayMs: 500,
		EOUMaxDelayMs: 3000,
		Language:      "en",
		HardCapMs:     50,
		MinSpeechMs:   1,
	}
	p := &sessionPipeline{session: cfg, logger: slog.New(slog.NewTextHandler(io.Discard, nil))}
	p.eouModel = blockingModel{block: 5 * time.Second}
	p.eouCfg = eou.Config{
		Kind:               eou.KindHeuristic,
		MinDelayMs:         cfg.EOUMinDelayMs,
		MaxDelayMs:         cfg.EOUMaxDelayMs,
		InferenceTimeoutMs: 5000,
		HardCapMs:          50,
		Languages:          eou.DefaultLanguages(),
	}
	p.inspector = &recordingInspector{}
	p.eou = p.runEOU
	deadline := time.Now().Add(50 * time.Millisecond)
	t0 := time.Now()
	v := p.callEOU(context.Background(), "x", nil, deadline)
	elapsed := time.Since(t0)
	if !v.hardCapFired {
		t.Fatalf("regress-v2-go-clamp-not-timer: clamp form would never fire; cap MUST fire as parallel timer (got %+v)", v)
	}
	if elapsed > 500*time.Millisecond {
		t.Fatalf("regress-v2-go-clamp-not-timer: cap should fire near deadline; elapsed=%v", elapsed)
	}
}

type fusionStubModel struct {
	textScore  float32
	audioScore float32
	textErr    error
	audioErr   error
}

func (f fusionStubModel) Predict(_ context.Context, req eou.Request) (eou.Verdict, error) {
	if req.Kind == eou.KindAudio {
		return eou.Verdict{Score: f.audioScore}, f.audioErr
	}
	return eou.Verdict{Score: f.textScore}, f.textErr
}
func (fusionStubModel) Close() error { return nil }

func TestRunEOUFusion_NoisyOrCombines(t *testing.T) {
	cfg := sessionConfig{
		EOUMinDelayMs: 500,
		EOUMaxDelayMs: 3000,
		CurveK:        12,
		Language:      "en",
		HardCapMs:     5000,
	}
	p := &sessionPipeline{session: cfg, logger: slog.New(slog.NewTextHandler(io.Discard, nil))}
	stub := fusionStubModel{textScore: 0.6, audioScore: 0.6}
	p.eouCfg = eou.Config{
		Kind:               eou.KindFusion,
		FusionRule:         eou.FusionNoisyOr,
		FusionWeightText:   0.5,
		MinDelayMs:         cfg.EOUMinDelayMs,
		MaxDelayMs:         cfg.EOUMaxDelayMs,
		InferenceTimeoutMs: 1000,
		HardCapMs:          5000,
		AudioWindowMs:      8000,
		Languages:          eou.DefaultLanguages(),
		TextModel:          stub,
		AudioModel:         stub,
	}
	p.eouModel = stub
	p.inspector = &recordingInspector{}
	p.eou = p.runEOU

	v := p.callEOU(context.Background(), "hello", make(audio.MonoF32, 16000), time.Time{})
	want := 1 - (1-0.6)*(1-0.6)
	if abs32(v.score-float32(want)) > 0.001 {
		t.Fatalf("noisy_or fusion: want score~%.3f, got %.3f", want, v.score)
	}
}

func TestRunEOUFusion_DegradesWhenAudioHeadFails(t *testing.T) {
	cfg := sessionConfig{
		EOUMinDelayMs: 500,
		EOUMaxDelayMs: 3000,
		CurveK:        12,
		Language:      "en",
		HardCapMs:     5000,
	}
	p := &sessionPipeline{session: cfg, logger: slog.New(slog.NewTextHandler(io.Discard, nil))}
	stub := fusionStubModel{textScore: 0.7, audioErr: context.DeadlineExceeded}
	p.eouCfg = eou.Config{
		Kind:               eou.KindFusion,
		FusionRule:         eou.FusionNoisyOr,
		FusionWeightText:   0.5,
		MinDelayMs:         cfg.EOUMinDelayMs,
		MaxDelayMs:         cfg.EOUMaxDelayMs,
		InferenceTimeoutMs: 1000,
		HardCapMs:          5000,
		AudioWindowMs:      8000,
		Languages:          eou.DefaultLanguages(),
		TextModel:          stub,
		AudioModel:         stub,
	}
	p.eouModel = stub
	p.eou = p.runEOU
	v := p.callEOU(context.Background(), "hello", make(audio.MonoF32, 16000), time.Time{})
	if v.score < 0.99 {
		t.Fatalf("when audio head fails, fusion (noisy_or) MUST treat audio as p=1; got score=%.3f", v.score)
	}
}

var _ = sync.Mutex{}

func abs32(x float32) float32 {
	if x < 0 {
		return -x
	}
	return x
}
