package realtime

import (
	"context"
	"errors"
	"fmt"
	"log/slog"
	"math"
	"sync"
	"sync/atomic"
	"time"

	"github.com/pion/opus"
	"github.com/pion/webrtc/v4"
	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/trace"

	"github.com/eordano/speaches-plus-go/internal/audio"
	"github.com/eordano/speaches-plus-go/internal/diarization"
	"github.com/eordano/speaches-plus-go/internal/eou"
	"github.com/eordano/speaches-plus-go/internal/inspect"
)

var silenceTimeout = time.Duration(defaultVADLessSilenceMs) * time.Millisecond

type vadDecision int

type sessionConfig struct {
	Model              string
	Intent             string
	TranscriptionModel string
	Voice              string
	Speed              float32
	SpeechModel        string
	Language           string
	Conversation       bool

	EOUMinDelayMs int
	EOUMaxDelayMs int
	HardCapMs     int

	EOUFailureP            float32
	EOUFailureDelay        string
	MinSpeechMs            int
	MinSpeechForResponseMs int
	BargeInDelayMs         int
	SessionMaxDurSec       int

	CurveK float32

	OutboundQueueCap       int
	OutboundBufferLimit    uint64
	DataChannelFragmentMax int

	DrainCapFloorMs   int
	DrainCapCeilingMs int

	PartialTickMs int

	LLMTimeoutSec int

	VADThreshold           float32
	VADNegThreshold        float32
	VADMinSpeechDurationMs int
	VADSilenceDurationMs   int
	VADPrefixPaddingMs     int

	NoSpeechProbThreshold *float32
	AvgLogprobThreshold   *float32

	StartSpeechSamples  int
	VADLessSilenceMs    int
	NonSilenceThreshold float32

	InspectorTransitions bool
	InspectorSampleRate  float32

	EOUContextTurns int

	SealedBufferRetentionCount int
	PredictedTokenBufferCap    int
	EOUAudioWindowMs           int
	VADModel                   string

	TurnDetectionType string

	EagerMaxInflight     int
	EagerPeriodicEnabled bool
	EagerIntervalMs      int

	InputAudioFormat  string
	OutputAudioFormat string
}

func zeroFill[T comparable](dst *T, def T) {
	var zero T
	if *dst == zero {
		*dst = def
	}
}

func (c *sessionConfig) fillDefaults() {
	if c.EOUMinDelayMs == 0 && c.EOUMaxDelayMs == 0 {
		c.EOUMinDelayMs = defaultEOUMinDelayMs
		c.EOUMaxDelayMs = defaultEOUMaxDelayMs
	}
	zeroFill(&c.HardCapMs, defaultHardCapMs)
	zeroFill(&c.EOUFailureP, defaultEOUFailureP)
	zeroFill(&c.EOUFailureDelay, defaultEOUFailureDelay)
	zeroFill(&c.MinSpeechMs, defaultMinSpeechMs)
	zeroFill(&c.MinSpeechForResponseMs, defaultMinSpeechForResponseMs)
	zeroFill(&c.CurveK, defaultEOUCurveK)
	zeroFill(&c.SessionMaxDurSec, defaultSessionMaxDurSec)
	zeroFill(&c.OutboundQueueCap, defaultOutboundQueueCap)
	zeroFill(&c.OutboundBufferLimit, defaultOutboundBufferLimit)
	zeroFill(&c.DataChannelFragmentMax, defaultDataChannelFragmentMax)
	zeroFill(&c.DrainCapFloorMs, defaultDrainCapFloorMs)
	zeroFill(&c.DrainCapCeilingMs, defaultDrainCapCeilingMs)
	zeroFill(&c.PartialTickMs, defaultPartialTickMs)
	zeroFill(&c.LLMTimeoutSec, defaultLLMTimeoutSec)
	zeroFill(&c.VADThreshold, defaultVADThreshold)
	zeroFill(&c.VADSilenceDurationMs, defaultVADSilenceDurationMs)
	zeroFill(&c.VADPrefixPaddingMs, defaultVADPrefixPaddingMs)
	zeroFill(&c.VADMinSpeechDurationMs, defaultVADMinSpeechDurationMs)

	zeroFill(&c.StartSpeechSamples, defaultStartSpeechSamples)
	zeroFill(&c.VADLessSilenceMs, defaultVADLessSilenceMs)
	zeroFill(&c.NonSilenceThreshold, defaultNonSilenceThreshold)
	zeroFill(&c.InspectorSampleRate, defaultInspectorSampleRate)
	zeroFill(&c.EOUContextTurns, defaultEOUContextTurns)
	zeroFill(&c.SealedBufferRetentionCount, defaultSealedBufferRetention)
	zeroFill(&c.PredictedTokenBufferCap, defaultPredictedTokenBufferCap)
	zeroFill(&c.EOUAudioWindowMs, defaultEOUAudioWindowMs)
	zeroFill(&c.VADModel, defaultVADModel)
	zeroFill(&c.TurnDetectionType, defaultTurnDetectionType)
	zeroFill(&c.EagerMaxInflight, defaultEagerMaxInflight)
	zeroFill(&c.EagerIntervalMs, defaultEagerIntervalMs)
	zeroFill(&c.InputAudioFormat, defaultInputAudioFormat)
	zeroFill(&c.OutputAudioFormat, defaultOutputAudioFormat)
	zeroFill(&c.Speed, defaultTTSSpeed)
}

type outboundWriter interface {
	WriteAudio(samples audio.MonoF32, sampleRate int) error
	PlayedMs() int64
	ResetPlayedMs()
}

type eouVerdict struct {
	score        float32
	delayMs      int
	hardCapFired bool

	phase string
}

type eouFn func(ctx context.Context, partialTranscript string, samples audio.MonoF32, hardCapDeadline time.Time) eouVerdict

type sessionPipeline struct {
	server    Config
	session   sessionConfig
	sessionID string
	logger    *slog.Logger

	traceCtx    context.Context
	sessionSpan trace.Span

	decoder   opus.Decoder
	closed    chan struct{}
	closeOnce sync.Once
	wg        sync.WaitGroup

	bufMu     sync.Mutex
	buf16k    audio.MonoF32
	vadCursor int
	lastAudio time.Time
	flushed   bool
	turnDone  chan struct{}

	audioCursor atomic.Int64

	chMu      sync.Mutex
	channel   *webrtc.DataChannel
	wsConn    wsTransport
	wsWriteMu sync.Mutex
	chReady   chan struct{}

	outMu       sync.Mutex
	outboundTTS outboundWriter

	vad         *vadAdapter
	vadFailures atomic.Uint32

	phase phaseState

	timerMu            sync.Mutex
	commitTimer        *time.Timer
	commitHardCapTimer *time.Timer
	commitFire         func(hardCapFired bool, phase string, score float32)
	commitCancel       chan struct{}
	commitItemID       string
	commitSamples      audio.MonoF32
	commitPartial      string
	commitArmedAt      time.Time
	commitDeadline     time.Time
	commitHardCap      bool

	partialMu      sync.Mutex
	partialCancel  context.CancelFunc
	partialPending string
	partialReady   bool

	eouMu    sync.Mutex
	eou      eouFn
	eouModel eou.Model
	eouCfg   eou.Config

	predictedMu     sync.Mutex
	predictedCancel context.CancelFunc

	bargeInMu     sync.Mutex
	bargeInCancel chan struct{}

	startedAt time.Time

	inspector  Inspector
	relay      *inspect.Relay
	audioStore *inspect.AudioStore

	outboundCh   chan outboundSend
	outboundOnce sync.Once

	diarMu   sync.Mutex
	diarizer *diarization.Diarizer
}

func (p *sessionPipeline) State() string { return "active" }
func (p *sessionPipeline) Model() string { return p.session.Model }

type outboundSend struct {
	event   any
	eventID string
}

func newSessionPipeline(server Config, sess sessionConfig, logger *slog.Logger) *sessionPipeline {
	return newSessionPipelineWithID(server, sess, logger, newSessID())
}

func newSessionPipelineWithID(server Config, sess sessionConfig, logger *slog.Logger, sessionID string) *sessionPipeline {
	p := &sessionPipeline{
		server:    server,
		session:   sess,
		sessionID: sessionID,
		logger:    logger,
		decoder:   opus.NewDecoder(),
		buf16k:    make(audio.MonoF32, 0, 30*whisperSampleRate),
		closed:    make(chan struct{}),
		chReady:   make(chan struct{}),
		turnDone:  make(chan struct{}),
	}
	p.session.fillDefaults()
	p.phase.setSealedRetention(p.session.SealedBufferRetentionCount)
	p.phase.setEagerMaxInflight(p.session.EagerMaxInflight)
	if server.EOUModel != nil {
		p.eouModel = server.EOUModel
		p.eouCfg = server.EOUConfig
	} else {
		base := server.EOUConfig
		if base.MinDelayMs == 0 {
			base.MinDelayMs = p.session.EOUMinDelayMs
		}
		if base.MaxDelayMs == 0 {
			base.MaxDelayMs = p.session.EOUMaxDelayMs
		}
		m, cfg, _ := eou.Load(base)
		p.eouModel = m
		p.eouCfg = cfg
	}
	p.eou = p.runEOU
	p.phase.startSession()
	p.startedAt = time.Now()
	dir := server.InspectSessionDir
	if dir == "" {
		dir = inspect.DefaultSessionDir()
	}
	relayCap := server.InspectorRelayCap
	if relayCap <= 0 {
		relayCap = defaultInspectorRelayCap
	}
	p.relay = inspect.NewRelayWithCap(p.sessionID, dir, relayCap)
	p.audioStore = inspect.NewAudioStore(p.sessionID, dir)
	p.inspector = NewInspector(p.relay)
	registerSessionAudioStore(p.sessionID, p.audioStore)
	inspect.Register(p.sessionID, p, p.relay)
	if want := server.EOUConfig.Kind; want != "" && want != eou.KindVad && want != eou.KindHeuristic {
		if got := p.eouCfg.Kind; got == eou.KindHeuristic || got == eou.KindVad {
			p.emitEOU("fallback", inspect.EOUFields{
				EouKind: string(got),
				Extra: map[string]any{
					"from_kind": string(want),
					"to_kind":   string(got),
					"reason":    "model_load_failed_or_missing",
				},
			})
		}
	}
	p.phase.transitionHook = func(phase, from, to string) {
		if p.inspector != nil {
			p.inspector.Emit("state.transition",
				"phase", phase, "from", from, "to", to)
		}
	}
	p.phase.violationHook = func(err error) {
		p.logger.Error("internal_state_error", "err", err)
		if p.inspector != nil {
			p.inspector.Emit("error.invariant_violation", "err", err.Error())
		}
		p.emitErrorCode("internal_state_error", err.Error())
		go func() {
			p.emitSessionDone("internal_state_error")
			p.phase.terminateSession()
			p.close()
		}()
	}
	p.vadFailures.Store(0)
	p.traceCtx, p.sessionSpan = startSpan(context.Background(), "realtime.session",
		attribute.String("model", p.session.Model),
		attribute.String("intent", p.session.Intent),
	)
	p.startSessionTimeout()
	p.startIntegratedConsumer()
	if server.DiarSegmentation != nil && server.DiarEmbedding != nil {
		cfg := server.DiarConfig
		if cfg.MaxSpeakers == 0 {
			cfg = diarization.DefaultConfig()
		}
		p.diarizer = diarization.NewDiarizer(server.DiarSegmentation, server.DiarEmbedding, cfg)
	}
	return p
}

func (p *sessionPipeline) onVADFailure(err error) bool {
	if p.vadFailures.Add(1) < vadFailureThreshold {
		return false
	}
	p.logger.Error("vad_failed", "err", err, "consecutive_failures", vadFailureThreshold)
	if p.inspector != nil {
		p.inspector.Emit("vad.failed", "err", err.Error())
	}
	p.emitErrorCode("vad_failed", err.Error())
	go func() {
		p.emitSessionDone("vad_failed")
		p.phase.terminateSessionWithReason(TermVadFailed)
		p.close()
	}()
	return true
}

func (p *sessionPipeline) startSessionTimeout() {
	if p.session.SessionMaxDurSec <= 0 {
		return
	}
	dur := time.Duration(p.session.SessionMaxDurSec) * time.Second
	p.wg.Add(1)
	go func() {
		defer p.wg.Done()
		select {
		case <-p.closed:
			return
		case <-time.After(dur):
			p.logger.Info("session.done: hard timeout",
				"after_s", p.session.SessionMaxDurSec)
			p.emitSessionDone("max_duration")
			p.phase.terminateSession()
			p.close()
		}
	}()
}

func (p *sessionPipeline) lang() string {
	if p.session.Language != "" {
		return p.session.Language
	}
	return "en"
}

func (p *sessionPipeline) stubEOU(_ context.Context, _ string, _ audio.MonoF32, _ time.Time) eouVerdict {
	return eouVerdict{score: 1.0, delayMs: p.session.EOUMinDelayMs}
}

func (p *sessionPipeline) runEOU(ctx context.Context, partial string, samples audio.MonoF32, hardCapDeadline time.Time) eouVerdict {
	p.eouMu.Lock()
	model := p.eouModel
	cfg := p.eouCfg
	p.eouMu.Unlock()
	kind := cfg.Kind
	if kind == "" {

		kind = eou.KindVad
	}

	if kind == eou.KindVad || model == nil {
		return eouVerdict{score: 1.0, delayMs: p.session.EOUMinDelayMs}
	}
	if kind == eou.KindFusion {
		return p.runEOUFusion(ctx, partial, samples, hardCapDeadline)
	}
	lang := p.lang()
	threshold := cfg.Languages.Threshold(lang)
	turns := p.priorTurns()
	timeoutMs := cfg.InferenceTimeoutMs
	if timeoutMs <= 0 {
		timeoutMs = 100
	}
	hardCapMs := cfg.HardCapMs
	if p.session.HardCapMs > 0 {
		hardCapMs = p.session.HardCapMs
	}
	if hardCapMs <= 0 {
		hardCapMs = 5000
	}

	_, eouSpan := startSpan(p.traceContext(), "eou.infer",
		attribute.String("eou.kind", string(kind)),
		attribute.Int("eou.partial_chars", len(partial)),
	)
	var (
		spanScore       float32
		spanDelayMs     int
		spanCancelledBy = "none"
	)
	defer func() {
		eouSpan.SetAttributes(
			attribute.Float64("eou.score", float64(spanScore)),
			attribute.Int("eou.delay_ms", spanDelayMs),
			attribute.String("eou.cancelled_by", spanCancelledBy),
		)
		eouSpan.End()
	}()

	infCtx, cancel := context.WithTimeout(ctx, time.Duration(timeoutMs)*time.Millisecond)
	defer cancel()

	type predictResult struct {
		v   eou.Verdict
		err error
		ms  int64
	}

	var audioWindow audio.MonoF32
	if kind == eou.KindAudio {
		audioWindow = tailWindow(samples, audioWindowSamples(cfg.AudioWindowMs))
	}

	predictCh := make(chan predictResult, 1)
	t0 := time.Now()
	go func() {
		v, err := model.Predict(infCtx, eou.Request{
			Kind:     kind,
			Turns:    turns,
			Partial:  partial,
			Language: lang,
			Audio:    audioWindow,
		})
		predictCh <- predictResult{v: v, err: err, ms: time.Since(t0).Milliseconds()}
	}()

	var hardCapCh <-chan time.Time
	if !hardCapDeadline.IsZero() {
		hardCapTimer := time.NewTimer(time.Until(hardCapDeadline))
		defer hardCapTimer.Stop()
		hardCapCh = hardCapTimer.C
	}

	var (
		v       eou.Verdict
		err     error
		elapsed int64
	)
	select {
	case res := <-predictCh:
		v, err, elapsed = res.v, res.err, res.ms
	case <-hardCapCh:

		cancel()
		thr := threshold
		p.emitEOU("hard_cap_fired", inspect.EOUFields{
			EouKind:      string(kind),
			Threshold:    &thr,
			Language:     lang,
			HardCapPhase: "during_eou",
			Extra:        map[string]any{"hard_cap_ms": hardCapMs},
		})
		p.logger.Warn("eou.hard_cap_fired during EOU compute",
			"phase", "during_eou",
			"hard_cap_ms", hardCapMs,
		)
		spanScore = 0
		spanDelayMs = 0
		spanCancelledBy = "hard_cap"
		return eouVerdict{score: 0, delayMs: 0, hardCapFired: true, phase: "during_eou"}
	case <-ctx.Done():

		spanCancelledBy = "context"
		return eouVerdict{score: 1.0, delayMs: p.session.EOUMinDelayMs}
	}

	uncertain := func(reason, kindStr string) eouVerdict {
		fp, fd := p.session.EOUFailureP, p.session.EOUFailureDelay
		score := float32(1.0)
		delay := p.session.EOUMinDelayMs
		if fp == 0 {
			score = 0
		}
		if fd == "max" {
			delay = p.session.EOUMaxDelayMs
		}
		thr := threshold
		sc := score
		dl := delay
		p.emitEOU("scored", inspect.EOUFields{
			EouKind:       kindStr,
			Score:         &sc,
			Threshold:     &thr,
			Language:      lang,
			DelayMs:       &dl,
			FailureReason: reason,
		})
		p.logger.Warn("eou.failure",
			"reason", reason,
			"eou.kind", kindStr,
			"eou.language", lang,
			"eou.threshold", threshold,
			"eou.delay_ms", delay,
			"eou.score", score,
		)
		spanScore = score
		spanDelayMs = delay
		spanCancelledBy = reason
		return eouVerdict{score: score, delayMs: delay}
	}

	cancelledBy := "none"
	if err != nil {
		if errors.Is(err, context.DeadlineExceeded) || errors.Is(err, context.Canceled) {

			return uncertain("timeout", string(kind))
		}
		return uncertain("error", string(kind))
	}

	switch {
	case math.IsNaN(float64(v.Score)) || math.IsInf(float64(v.Score), 0):
		return uncertain("garbage_prob", string(kind))
	case v.Score < 0 || v.Score > 1:
		return uncertain("garbage_prob", string(kind))
	}

	_ = cancelledBy

	delay := eou.SigmoidLerpK(v.Score, threshold, p.session.EOUMinDelayMs, p.session.EOUMaxDelayMs, p.session.CurveK)
	score := v.Score
	thr := threshold
	curveK := p.session.CurveK
	dlMs := delay
	elMs := int(elapsed)
	p.emitEOU("scored", inspect.EOUFields{
		EouKind:   string(kind),
		Score:     &score,
		Threshold: &thr,
		Language:  lang,
		CurveK:    &curveK,
		DelayMs:   &dlMs,
		ElapsedMs: &elMs,
		Extra: map[string]any{
			"input_chars": len(partial),
			"prior_turns": len(turns),
		},
	})
	p.logger.Info("eou.verdict",
		"eou.kind", string(kind),
		"eou.score", v.Score,
		"eou.threshold", threshold,
		"eou.language", lang,
		"eou.delay_ms", delay,
		"eou.elapsed_ms", elapsed,
		"eou.input_chars", len(partial),
		"prior_turns", len(turns),
		"partial", partial,
	)
	spanScore = v.Score
	spanDelayMs = delay
	return eouVerdict{score: v.Score, delayMs: delay}
}

func (p *sessionPipeline) runEOUFusion(ctx context.Context, partial string, samples audio.MonoF32, hardCapDeadline time.Time) eouVerdict {
	p.eouMu.Lock()
	cfg := p.eouCfg
	p.eouMu.Unlock()

	lang := p.lang()
	threshold := cfg.Languages.Threshold(lang)
	turns := p.priorTurns()
	timeoutMs := cfg.InferenceTimeoutMs
	if timeoutMs <= 0 {
		timeoutMs = defaultInferenceTimeoutMs
	}
	hardCapMs := cfg.HardCapMs
	if p.session.HardCapMs > 0 {
		hardCapMs = p.session.HardCapMs
	}
	if hardCapMs <= 0 {
		hardCapMs = defaultHardCapMs
	}

	_, eouSpan := startSpan(p.traceContext(), "eou.infer.fusion",
		attribute.String("eou.kind", string(eou.KindFusion)),
		attribute.String("eou.fusion_rule", string(cfg.FusionRule)),
		attribute.Int("eou.partial_chars", len(partial)),
		attribute.Int("eou.audio_samples", len(samples)),
	)
	var (
		spanScore       float32
		spanDelayMs     int
		spanCancelledBy = "none"
		spanScoreText   float32
		spanScoreAudio  float32
	)
	defer func() {
		eouSpan.SetAttributes(
			attribute.Float64("eou.score", float64(spanScore)),
			attribute.Float64("eou.score_text", float64(spanScoreText)),
			attribute.Float64("eou.score_audio", float64(spanScoreAudio)),
			attribute.Int("eou.delay_ms", spanDelayMs),
			attribute.String("eou.cancelled_by", spanCancelledBy),
		)
		eouSpan.End()
	}()

	infCtx, cancel := context.WithTimeout(ctx, time.Duration(timeoutMs)*time.Millisecond)
	defer cancel()

	textCh := make(chan headResult, 1)
	audioCh := make(chan headResult, 1)

	go func() {
		if cfg.TextModel == nil {
			textCh <- headResult{score: 1.0, err: errNoFusionHead}
			return
		}
		v, err := cfg.TextModel.Predict(infCtx, eou.Request{
			Kind:     eou.KindText,
			Turns:    turns,
			Partial:  partial,
			Language: lang,
		})
		textCh <- headResult{score: v.Score, err: err}
	}()

	go func() {
		if cfg.AudioModel == nil {
			audioCh <- headResult{score: 1.0, err: errNoFusionHead}
			return
		}
		audioWindow := tailWindow(samples, audioWindowSamples(cfg.AudioWindowMs))
		v, err := cfg.AudioModel.Predict(infCtx, eou.Request{
			Kind:     eou.KindAudio,
			Audio:    audioWindow,
			Language: lang,
		})
		audioCh <- headResult{score: v.Score, err: err}
	}()

	var hardCapCh <-chan time.Time
	if !hardCapDeadline.IsZero() {
		hardCapTimer := time.NewTimer(time.Until(hardCapDeadline))
		defer hardCapTimer.Stop()
		hardCapCh = hardCapTimer.C
	}

	var (
		textRes  *headResult
		audioRes *headResult
	)
	for textRes == nil || audioRes == nil {
		select {
		case r := <-textCh:
			textRes = &r
		case r := <-audioCh:
			audioRes = &r
		case <-hardCapCh:
			cancel()
			thr := threshold
			p.emitEOU("hard_cap_fired", inspect.EOUFields{
				EouKind:      string(eou.KindFusion),
				FusionRule:   string(cfg.FusionRule),
				Threshold:    &thr,
				Language:     lang,
				HardCapPhase: "during_eou",
				Extra:        map[string]any{"hard_cap_ms": hardCapMs},
			})
			p.logger.Warn("eou.hard_cap_fired during fusion compute",
				"phase", "during_eou",
				"hard_cap_ms", hardCapMs,
			)
			spanScore = 0
			spanDelayMs = 0
			spanCancelledBy = "hard_cap"
			return eouVerdict{score: 0, delayMs: 0, hardCapFired: true, phase: "during_eou"}
		case <-ctx.Done():
			spanCancelledBy = "context"
			return eouVerdict{score: 1.0, delayMs: p.session.EOUMinDelayMs}
		}
	}

	pText := normalizeFusionScore(textRes)
	pAudio := normalizeFusionScore(audioRes)
	spanScoreText = pText
	spanScoreAudio = pAudio

	gatedFeat := eou.ExtractGatedFusionFeatures(partial,
		len(samples)*1000/whisperSampleRate)
	score := eou.FuseScoresWithFeatures(cfg.FusionRule,
		pText, pAudio, cfg.FusionWeightText,
		gatedFeat, eou.DefaultGatedFusionWeights)
	if score < 0 {
		score = 0
	} else if score > 1 {
		score = 1
	}
	delay := eou.SigmoidLerpK(score, threshold, p.session.EOUMinDelayMs, p.session.EOUMaxDelayMs, p.session.CurveK)

	scoreOut := score
	thr := threshold
	st := pText
	sa := pAudio
	dlMs := delay
	curveK := p.session.CurveK
	p.emitEOU("scored", inspect.EOUFields{
		EouKind:    string(eou.KindFusion),
		FusionRule: string(cfg.FusionRule),
		Score:      &scoreOut,
		ScoreText:  &st,
		ScoreAudio: &sa,
		Threshold:  &thr,
		Language:   lang,
		CurveK:     &curveK,
		DelayMs:    &dlMs,
	})

	spanScore = score
	spanDelayMs = delay
	return eouVerdict{score: score, delayMs: delay}
}

func normalizeFusionScore(r *headResult) float32 {
	if r.err != nil {
		return 1.0
	}
	s := r.score
	if math.IsNaN(float64(s)) || math.IsInf(float64(s), 0) {
		return 1.0
	}
	if s < 0 {
		return 0
	}
	if s > 1 {
		return 1
	}
	return s
}

type headResult struct {
	score float32
	err   error
}

var errNoFusionHead = errors.New("fusion head not configured")

func tailWindow(samples audio.MonoF32, n int) audio.MonoF32 {
	if n <= 0 || len(samples) <= n {
		return samples
	}
	return samples[len(samples)-n:]
}

func audioWindowSamples(ms int) int {
	if ms <= 0 {
		ms = defaultEOUAudioWindowMs
	}
	return ms * whisperSampleRate / 1000
}

func (p *sessionPipeline) priorTurns() []eou.Turn {
	conv := p.phase.conversationSnapshot()
	out := make([]eou.Turn, 0, len(conv))
	for _, it := range conv {
		if it.Status == itemInProgress {
			continue
		}
		if it.Transcript == "" {
			continue
		}
		role := it.Role
		if role == "" {
			role = "user"
		}
		out = append(out, eou.Turn{Role: role, Content: it.Transcript})
	}
	window := p.session.EOUContextTurns
	if window <= 0 {
		window = eouHistoryFallbackTurns
	}
	return eou.RollingHistory(out, window)
}

func (p *sessionPipeline) setEOU(fn eouFn) {
	p.eouMu.Lock()
	p.eou = fn
	p.eouMu.Unlock()
}

func (p *sessionPipeline) callEOU(ctx context.Context, partial string, samples audio.MonoF32, hardCapDeadline time.Time) eouVerdict {
	p.eouMu.Lock()
	fn := p.eou
	p.eouMu.Unlock()
	if fn == nil {
		return eouVerdict{score: 1.0, delayMs: p.session.EOUMinDelayMs}
	}
	return fn(ctx, partial, samples, hardCapDeadline)
}

func (p *sessionPipeline) attachChannel(ch *webrtc.DataChannel) {
	p.chMu.Lock()
	defer p.chMu.Unlock()
	if p.channel != nil {
		return
	}
	p.channel = ch
	close(p.chReady)
}

func (p *sessionPipeline) getChannel() *webrtc.DataChannel {
	p.chMu.Lock()
	defer p.chMu.Unlock()
	return p.channel
}

func (p *sessionPipeline) close() {
	p.closeOnce.Do(func() {
		close(p.closed)
		p.cancelCommitTimer()
		p.wg.Wait()
		if p.sessionSpan != nil {
			p.sessionSpan.End()
			p.sessionSpan = nil
		}
		if p.audioStore != nil {
			p.audioStore.Close()
		}
		if p.sessionID != "" {
			inspect.Unregister(p.sessionID)
			unregisterSessionAudioStore(p.sessionID)
		}
	})
}

func (p *sessionPipeline) traceContext() context.Context {
	if p.traceCtx == nil {
		return context.Background()
	}
	return p.traceCtx
}

func (p *sessionPipeline) markTurnDone() {
	p.bufMu.Lock()
	defer p.bufMu.Unlock()
	if p.turnDone != nil {
		select {
		case <-p.turnDone:
		default:
			close(p.turnDone)
		}
	}
}

func (p *sessionPipeline) waitForTurnDone() {
	p.bufMu.Lock()
	ch := p.turnDone
	p.bufMu.Unlock()
	if ch == nil {
		return
	}
	select {
	case <-ch:
	case <-p.closed:
	}
}

func (p *sessionPipeline) resetTurn() {
	p.bufMu.Lock()
	p.flushed = false
	p.lastAudio = time.Time{}
	p.turnDone = make(chan struct{})
	p.bufMu.Unlock()
	if p.vad != nil {
		p.vad.Reset()
	}
}

var processScopedSessionUpdateFields = []string{

	"vad_model",
	"session_max_duration_hard_cap_s",

	"chat_completion_base_url",
	"chat_completion_api_key",
	"default_realtime_model",
	"default_realtime_stt_model",
	"default_realtime_partial_stt_model",
	"default_speech_model",
	"default_voice",
	"gpu_mem_limit_bytes",
}

func (p *sessionPipeline) applySessionUpdate(body *sessionUpdateBody) error {
	if err := validateSessionUpdate(body); err != nil {
		return err
	}

	if body.Instructions != nil {
		p.setInstructions(*body.Instructions)
	}
	if body.Voice != nil {
		p.session.Voice = *body.Voice
	}
	if body.Speed != nil {
		s := *body.Speed
		if !isFiniteFloat32(s) {
			s = defaultTTSSpeed
		}
		if s < minTTSSpeed {
			s = minTTSSpeed
		}
		if s > maxTTSSpeed {
			s = maxTTSSpeed
		}
		p.session.Speed = s
	}
	if body.TurnDetection != nil {
		td := body.TurnDetection
		if td.Threshold != nil {
			p.session.VADThreshold = *td.Threshold
			if p.vad != nil {
				p.vad.SetThreshold(*td.Threshold)
			}
		}
		if td.NegThreshold != nil {
			p.session.VADNegThreshold = *td.NegThreshold
			if p.vad != nil {
				p.vad.SetNegThreshold(*td.NegThreshold)
			}
		}
		if td.MinSpeechDurationMs != nil {
			p.session.VADMinSpeechDurationMs = *td.MinSpeechDurationMs
			if p.vad != nil {
				p.vad.SetMinSpeechMs(*td.MinSpeechDurationMs)
			}
		}
		if td.PrefixPaddingMs != nil {
			p.session.VADPrefixPaddingMs = *td.PrefixPaddingMs
			if p.vad != nil {
				p.vad.SetPrefixPaddingMs(*td.PrefixPaddingMs)
			}
		}
		if td.SilenceDurationMs != nil {
			p.session.VADSilenceDurationMs = *td.SilenceDurationMs
			if p.vad != nil {
				p.vad.SetSilenceMs(*td.SilenceDurationMs)
			}
			p.rescheduleCommitTimerForSilence(*td.SilenceDurationMs)
		}
		if td.BargeInDelayMs != nil {
			p.session.BargeInDelayMs = *td.BargeInDelayMs
		}
		if td.EOU != nil {
			p.applyEOUUpdate(td.EOU)
		}
		if td.Type != nil {
			oldType := p.session.TurnDetectionType
			p.session.TurnDetectionType = *td.Type
			if *td.Type == "none" && oldType != "none" {
				p.cancelCommitTimer()
				p.cancelBargeInTask()
			}
		}
	}
	if body.SessionMaxDurationS != nil {
		p.session.SessionMaxDurSec = *body.SessionMaxDurationS
	}
	if body.MinSpeechMs != nil {
		p.session.MinSpeechMs = *body.MinSpeechMs
	}
	if body.MinSpeechForResponseMs != nil {
		p.session.MinSpeechForResponseMs = *body.MinSpeechForResponseMs
	}
	if body.NoSpeechProbThresholdNull {
		p.session.NoSpeechProbThreshold = nil
	} else if body.NoSpeechProbThreshold != nil {
		v := *body.NoSpeechProbThreshold
		p.session.NoSpeechProbThreshold = &v
	}
	if body.AvgLogprobThresholdNull {
		p.session.AvgLogprobThreshold = nil
	} else if body.AvgLogprobThreshold != nil {
		v := *body.AvgLogprobThreshold
		p.session.AvgLogprobThreshold = &v
	}
	if body.SealedBufferRetentionCount != nil {
		p.session.SealedBufferRetentionCount = *body.SealedBufferRetentionCount
		p.phase.setSealedRetention(*body.SealedBufferRetentionCount)
	}
	if body.InputAudioFormat != nil {
		p.session.InputAudioFormat = *body.InputAudioFormat
	}
	if body.OutputAudioFormat != nil {
		p.session.OutputAudioFormat = *body.OutputAudioFormat
	}
	p.logger.Info("session.update applied",
		"instructions_changed", body.Instructions != nil,
		"voice_changed", body.Voice != nil,
		"speed_changed", body.Speed != nil,
		"turn_detection_changed", body.TurnDetection != nil,
	)
	return nil
}

func (p *sessionPipeline) applyEOUUpdate(e *eouBody) {
	if e.MinDelayMs != nil {
		p.session.EOUMinDelayMs = *e.MinDelayMs
	}
	if e.MaxDelayMs != nil {
		p.session.EOUMaxDelayMs = *e.MaxDelayMs
	}
	if e.CurveK != nil {
		p.session.CurveK = *e.CurveK
	}
	if e.FailurePDefault != nil {
		p.session.EOUFailureP = *e.FailurePDefault
	}
	if e.FailureDelay != nil {
		p.session.EOUFailureDelay = *e.FailureDelay
	}
	if e.HardCapMs != nil {
		p.session.HardCapMs = *e.HardCapMs
	}
	if e.ContextTurns != nil {
		p.session.EOUContextTurns = *e.ContextTurns
	}
	p.eouMu.Lock()
	defer p.eouMu.Unlock()
	if e.Kind != nil {
		p.eouCfg.Kind = eou.Kind(*e.Kind)
	}
	if e.InferenceTimeoutMs != nil {
		p.eouCfg.InferenceTimeoutMs = *e.InferenceTimeoutMs
	}
	if e.ContextTurns != nil {
		p.eouCfg.ContextTurns = *e.ContextTurns
	}
	if e.PThreshold != nil {
		p.eouCfg.PThreshold = *e.PThreshold
	}
	if e.FusionRule != nil {
		p.eouCfg.FusionRule = eou.FusionRule(*e.FusionRule)
	}
	if e.FusionWeightText != nil {
		p.eouCfg.FusionWeightText = *e.FusionWeightText
	}
}

func validateSessionUpdate(body *sessionUpdateBody) error {
	if body.Instructions != nil && *body.Instructions == "" {
		return fmt.Errorf("instructions: must be a non-empty string")
	}
	for _, name := range body.ProcessScoped {
		for _, ps := range processScopedSessionUpdateFields {
			if name == ps {
				return fmt.Errorf("%s: process-scoped (server-startup only); cannot be changed via session.update", ps)
			}
		}
	}
	if body.TurnDetection != nil {
		if err := validateTurnDetection(body.TurnDetection); err != nil {
			return err
		}
	}
	if body.SessionMaxDurationS != nil &&
		(*body.SessionMaxDurationS < 1 || *body.SessionMaxDurationS > maxSessionMaxDurationS) {
		return fmt.Errorf("session_max_duration_s: must be in [1,%d]", maxSessionMaxDurationS)
	}
	if body.MinSpeechMs != nil &&
		(*body.MinSpeechMs < 0 || *body.MinSpeechMs > maxMinSpeechMs) {
		return fmt.Errorf("min_speech_ms: must be in [0,%d]", maxMinSpeechMs)
	}
	if body.MinSpeechForResponseMs != nil &&
		(*body.MinSpeechForResponseMs < 0 || *body.MinSpeechForResponseMs > maxMinSpeechForResponseMs) {
		return fmt.Errorf("min_speech_for_response_ms: must be in [0,%d]", maxMinSpeechForResponseMs)
	}
	if body.SealedBufferRetentionCount != nil &&
		(*body.SealedBufferRetentionCount < 0 || *body.SealedBufferRetentionCount > maxSealedBufferRetentionCount) {
		return fmt.Errorf("sealed_buffer_retention_count: must be in [0,%d]", maxSealedBufferRetentionCount)
	}
	if body.InputAudioFormat != nil && !validAudioFormat(*body.InputAudioFormat, supportedInputAudioFormats()) {
		return fmt.Errorf("input_audio_format: unsupported value %q (supported: %v)", *body.InputAudioFormat, supportedInputAudioFormats())
	}
	if body.OutputAudioFormat != nil && !validAudioFormat(*body.OutputAudioFormat, supportedOutputAudioFormats()) {
		return fmt.Errorf("output_audio_format: unsupported value %q (supported: %v)", *body.OutputAudioFormat, supportedOutputAudioFormats())
	}
	if body.NoSpeechProbThreshold != nil {
		v := *body.NoSpeechProbThreshold
		if !isFiniteFloat32(v) || v < 0 || v > 1 {
			return fmt.Errorf("no_speech_prob_threshold: must be in [0,1]")
		}
	}
	if body.AvgLogprobThreshold != nil {
		v := *body.AvgLogprobThreshold
		if !isFiniteFloat32(v) {
			return fmt.Errorf("avg_logprob_threshold: must be finite")
		}
	}
	return nil
}

func isFiniteFloat32(v float32) bool {
	return v == v && v != float32(math.Inf(1)) && v != float32(math.Inf(-1))
}

func validateTurnDetection(td *turnDetectionBody) error {
	if td.Type != nil && *td.Type != TurnDetectionTypeServerVad && *td.Type != TurnDetectionTypeNone {
		return fmt.Errorf("turn_detection.type: must be server_vad or none")
	}
	if td.Threshold != nil && (!isFiniteFloat32(*td.Threshold) || *td.Threshold < 0 || *td.Threshold > 1) {
		return fmt.Errorf("turn_detection.threshold: must be in [0,1]")
	}
	if td.NegThreshold != nil && (!isFiniteFloat32(*td.NegThreshold) || *td.NegThreshold < 0 || *td.NegThreshold > 1) {
		return fmt.Errorf("turn_detection.neg_threshold: must be in [0,1]")
	}
	if td.MinSpeechDurationMs != nil &&
		(*td.MinSpeechDurationMs < 0 || *td.MinSpeechDurationMs > maxMinSpeechMs) {
		return fmt.Errorf("turn_detection.min_speech_duration_ms: must be in [0,%d]", maxMinSpeechMs)
	}
	if td.PrefixPaddingMs != nil && (*td.PrefixPaddingMs < 0 || *td.PrefixPaddingMs > maxPrefixPaddingMs) {
		return fmt.Errorf("turn_detection.prefix_padding_ms: must be in [0,%d]", maxPrefixPaddingMs)
	}
	if td.SilenceDurationMs != nil &&
		(*td.SilenceDurationMs < minSilenceDurationMs || *td.SilenceDurationMs > maxSilenceDurationMs) {
		return fmt.Errorf("turn_detection.silence_duration_ms: must be in [%d,%d]", minSilenceDurationMs, maxSilenceDurationMs)
	}
	if td.BargeInDelayMs != nil && (*td.BargeInDelayMs < 0 || *td.BargeInDelayMs > maxBargeInDelayMs) {
		return fmt.Errorf("turn_detection.barge_in_delay_ms: must be in [0,%d]", maxBargeInDelayMs)
	}
	if td.EOU != nil {
		return validateEOUUpdate(td.EOU)
	}
	return nil
}

func validateEOUUpdate(e *eouBody) error {
	if e.MinDelayMs != nil && *e.MinDelayMs < 0 {
		return fmt.Errorf("eou.min_delay_ms: must be >= 0")
	}
	if e.MaxDelayMs != nil && *e.MaxDelayMs < 0 {
		return fmt.Errorf("eou.max_delay_ms: must be >= 0")
	}
	if e.CurveK != nil && (*e.CurveK <= 0 || *e.CurveK > maxEOUCurveK) {
		return fmt.Errorf("eou.curve_k: must be in (0,%d]", maxEOUCurveK)
	}
	if e.Kind != nil {
		switch *e.Kind {
		case "vad", "heuristic", "text", "audio", "fusion", "integrated":
		default:
			return fmt.Errorf("eou.kind: must be one of vad|heuristic|text|audio|fusion|integrated")
		}
	}
	if e.FusionRule != nil && !eou.ValidFusionRule(eou.FusionRule(*e.FusionRule)) {
		return fmt.Errorf("eou.fusion_rule: must be one of noisy_or|max|mean|weighted|gated")
	}
	if e.FusionWeightText != nil && (*e.FusionWeightText < 0 || *e.FusionWeightText > 1) {
		return fmt.Errorf("eou.fusion_weight_text: must be in [0,1]")
	}
	if e.FailurePDefault != nil && *e.FailurePDefault != 0.0 && *e.FailurePDefault != 1.0 {
		return fmt.Errorf("eou.failure_p_default: must be 0.0 or 1.0")
	}
	if e.FailureDelay != nil && *e.FailureDelay != FailureDelayMin && *e.FailureDelay != FailureDelayMax {
		return fmt.Errorf("eou.failure_delay: must be \"min\" or \"max\"")
	}
	if e.HardCapMs != nil && (*e.HardCapMs < 0 || *e.HardCapMs > maxEOUHardCapMs) {
		return fmt.Errorf("eou.silence_hard_cap_ms: must be in [0,%d]", maxEOUHardCapMs)
	}
	if e.InferenceTimeoutMs != nil && (*e.InferenceTimeoutMs < 0 || *e.InferenceTimeoutMs > maxEOUInferenceTimeoutMs) {
		return fmt.Errorf("eou.inference_timeout_ms: must be in [0,%d]", maxEOUInferenceTimeoutMs)
	}
	if e.ContextTurns != nil && (*e.ContextTurns < 0 || *e.ContextTurns > maxEOUContextTurns) {
		return fmt.Errorf("eou.context_turns: must be in [0,%d]", maxEOUContextTurns)
	}
	if e.PThreshold != nil && (*e.PThreshold < 0 || *e.PThreshold > 1) {
		return fmt.Errorf("eou.p_threshold: must be in [0,1]")
	}
	return nil
}

func (p *sessionPipeline) setInstructions(s string) {
	p.phase.setInstructions(s)
}

func (p *sessionPipeline) getInstructions() string {
	return p.phase.getInstructions()
}

func (p *sessionPipeline) attachOutbound(o outboundWriter) {
	p.outMu.Lock()
	p.outboundTTS = o
	p.outMu.Unlock()
}

func (p *sessionPipeline) getOutboundTTS() outboundWriter {
	p.outMu.Lock()
	defer p.outMu.Unlock()
	return p.outboundTTS
}

func (p *sessionPipeline) startCommitTimer(itemID string, samples audio.MonoF32) {
	p.cancelCommitTimer()

	hardCapMs := p.session.HardCapMs
	if hardCapMs <= 0 {
		hardCapMs = defaultHardCapMs
	}
	hardCapDeadline := time.Now().Add(time.Duration(hardCapMs) * time.Millisecond)

	cancelCh := make(chan struct{})
	p.timerMu.Lock()
	p.commitCancel = cancelCh
	p.timerMu.Unlock()

	p.wg.Add(1)
	go func() {
		defer p.wg.Done()
		p.runPartialAndScheduleCommit(itemID, samples, cancelCh, hardCapDeadline)
	}()
}

func (p *sessionPipeline) runPartialAndScheduleCommit(itemID string, samples audio.MonoF32, cancelCh chan struct{}, hardCapDeadline time.Time) {
	partial := ""
	if p.server.STT != nil {
		t0 := time.Now()
		text, err := p.server.STT.Transcribe(samples, whisperSampleRate)
		if err != nil {
			p.logger.Warn("partial STT failed; continuing with empty transcript", "err", err)
		} else {
			partial = text
			p.logger.Info("partial STT done",
				"elapsed_ms", time.Since(t0).Milliseconds(),
				"text", text,
			)
		}
	}

	select {
	case <-cancelCh:
		p.logger.Debug("partial STT result discarded; commit_timer cancelled")
		return
	case <-p.closed:
		return
	default:
	}

	if partial != "" {
		p.emitInputBufferPartialTranscription(itemID, partial)
	}

	p.partialMu.Lock()
	p.partialPending = partial
	p.partialReady = true
	p.partialMu.Unlock()

	verdict := p.callEOU(context.Background(), partial, samples, hardCapDeadline)
	delay := time.Duration(verdict.delayMs) * time.Millisecond

	p.maybeDispatchEager(itemID, partial, verdict.score, samples)
	p.logger.Info("commit_timer scheduled",
		"item", itemID,
		"eou_score", verdict.score,
		"delay_ms", verdict.delayMs,
		"phase", verdict.phase,
	)

	if verdict.phase == "during_eou" {
		p.timerMu.Lock()
		if p.commitCancel != cancelCh {
			p.timerMu.Unlock()
			return
		}
		if p.session.TurnDetectionType == "none" {
			p.timerMu.Unlock()
			return
		}
		now := time.Now()
		p.commitItemID = itemID
		p.commitSamples = samples
		p.commitPartial = partial
		p.commitArmedAt = now
		p.commitDeadline = now
		p.commitHardCap = true
		p.timerMu.Unlock()
		select {
		case <-cancelCh:
			return
		case <-p.closed:
			return
		default:
		}
		p.fireCommitTimer(itemID, samples, partial, true)
		return
	}

	p.timerMu.Lock()
	if p.commitCancel != cancelCh {
		p.timerMu.Unlock()
		return
	}
	if p.commitTimer != nil {
		p.commitTimer.Stop()
	}

	if p.session.TurnDetectionType == "none" {
		p.timerMu.Unlock()
		p.logger.Debug("commit_timer skipped: turn_detection.type=none")
		return
	}
	now := time.Now()
	p.commitItemID = itemID
	p.commitSamples = samples
	p.commitPartial = partial
	p.commitArmedAt = now
	p.commitDeadline = now.Add(delay)

	var fireOnce sync.Once
	fire := func(hardCapFired bool, phase string, score float32) {
		fireOnce.Do(func() {
			select {
			case <-cancelCh:
				return
			default:
			}
			if hardCapFired && phase == "during_wait" {
				sc := score
				p.emitEOU("hard_cap_fired", inspect.EOUFields{
					Score:        &sc,
					HardCapPhase: "during_wait",
				})
				p.logger.Warn("eou.hard_cap_fired during post-verdict sleep",
					"phase", "during_wait",
					"score", score,
				)
			}
			p.timerMu.Lock()
			if hardCapFired {
				p.commitHardCap = true
			}
			p.timerMu.Unlock()
			p.fireCommitTimer(itemID, samples, partial, hardCapFired)
		})
	}
	p.commitHardCap = false
	p.commitFire = fire

	p.commitTimer = time.AfterFunc(delay, func() {
		fire(false, "", verdict.score)
	})

	timeUntilCap := time.Until(hardCapDeadline)
	if timeUntilCap < 0 {
		timeUntilCap = 0
	}
	p.commitHardCapTimer = time.AfterFunc(timeUntilCap, func() {
		fire(true, "during_wait", verdict.score)
	})
	p.timerMu.Unlock()
}

func (p *sessionPipeline) startPartialLoop(itemID string) {
	p.partialMu.Lock()
	if p.partialCancel != nil {
		p.partialCancel()
		p.partialCancel = nil
	}
	ctx, cancel := context.WithCancel(context.Background())
	p.partialCancel = cancel
	p.partialMu.Unlock()

	if p.session.EagerPeriodicEnabled {
		p.startEagerPeriodicTimer(ctx, itemID)
	}

	p.wg.Add(1)
	go func() {
		defer p.wg.Done()
		tickMs := p.session.PartialTickMs
		if tickMs <= 0 {
			tickMs = defaultPartialTickMs
		}
		tick := time.NewTicker(time.Duration(tickMs) * time.Millisecond)
		defer tick.Stop()
		for {
			select {
			case <-ctx.Done():
				return
			case <-p.closed:
				return
			case <-tick.C:
				p.bufMu.Lock()
				snapshot := append(audio.MonoF32(nil), p.buf16k...)
				p.bufMu.Unlock()
				if len(snapshot) < p.session.StartSpeechSamples {
					continue
				}
				if p.server.STT == nil {
					return
				}
				text, err := p.server.STT.Transcribe(snapshot, whisperSampleRate)
				if err != nil {
					continue
				}
				if text == "" {
					continue
				}
				p.emitInputBufferPartialTranscription(itemID, text)
			}
		}
	}()
}

func (p *sessionPipeline) startEagerPeriodicTimer(ctx context.Context, itemID string) {
	p.eouMu.Lock()
	kind := p.eouCfg.Kind
	p.eouMu.Unlock()
	if kind != eou.KindText && kind != eou.KindAudio {
		return
	}
	intervalMs := p.session.EagerIntervalMs
	if intervalMs <= 0 {
		intervalMs = defaultEagerIntervalMs
	}
	p.wg.Add(1)
	go func() {
		defer p.wg.Done()
		tick := time.NewTicker(time.Duration(intervalMs) * time.Millisecond)
		defer tick.Stop()
		for {
			select {
			case <-ctx.Done():
				return
			case <-p.closed:
				return
			case <-tick.C:

				_, vad, resp := p.phase.snapshot()
				if _, speaking := vad.(VadSpeaking); !speaking {
					return
				}
				if resp.Kind() != respKindNone {

					continue
				}
				p.partialMu.Lock()
				partial := p.partialPending
				p.partialMu.Unlock()
				if partial == "" {
					continue
				}

				p.bufMu.Lock()
				snapAudio := append(audio.MonoF32(nil), p.buf16k...)
				p.bufMu.Unlock()
				verdict := p.callEOU(ctx, partial, snapAudio, time.Time{})
				sc := verdict.score
				p.emitEOU("periodic_reeval", inspect.EOUFields{
					Score: &sc,
					Extra: map[string]any{
						"item_id":     itemID,
						"interval_ms": intervalMs,
					},
				})
				p.maybeDispatchEager(itemID, partial, verdict.score, snapAudio)
			}
		}
	}()
}

func (p *sessionPipeline) stopPartialLoop() {
	p.partialMu.Lock()
	defer p.partialMu.Unlock()
	if p.partialCancel != nil {
		p.partialCancel()
		p.partialCancel = nil
	}
}

func (p *sessionPipeline) startOutboundQueue() {
	p.outboundOnce.Do(func() {
		cap := p.session.OutboundQueueCap
		if cap <= 0 {
			cap = defaultOutboundQueueCap
		}
		p.outboundCh = make(chan outboundSend, cap)
		p.wg.Add(1)
		go func() {
			defer p.wg.Done()
			for {
				select {
				case <-p.closed:
					return
				case s, ok := <-p.outboundCh:
					if !ok {
						return
					}
					ch := p.getChannel()
					if ch == nil {
						continue
					}
					if err := sendFragmented(ch, s.event, s.eventID); err != nil {
						p.logger.Warn("outbound send failed", "err", err, "id", s.eventID)
					}
				}
			}
		}()
	})
}

func (p *sessionPipeline) enqueueOutbound(event any, eventID string) bool {
	p.startOutboundQueue()
	select {
	case p.outboundCh <- outboundSend{event: event, eventID: eventID}:
		return true
	default:
		return false
	}
}

func (p *sessionPipeline) maybeDispatchEager(itemID, partial string, score float32, samples []float32) {
	_, dispSpan := startSpan(p.traceContext(), "eou.predicted_dispatch",
		attribute.Float64("eou.score", float64(score)),
		attribute.Bool("eou.eager", true),
	)
	var (
		spanRespID        string
		spanRunnerStarted bool
	)
	defer func() {
		dispSpan.SetAttributes(
			attribute.String("eou.response_id", spanRespID),
			attribute.Bool("eou.runner_started", spanRunnerStarted),
		)
		dispSpan.End()
	}()
	p.eouMu.Lock()
	cfg := p.eouCfg
	p.eouMu.Unlock()
	if cfg.EagerPThreshold <= 0 || score < cfg.EagerPThreshold {
		return
	}
	if !p.session.Conversation {
		return
	}
	predID := newRespID()
	spanRespID = predID
	respItemID := newItemID()

	r := p.startEagerRunner(predID, respItemID, 0, partial, samples)
	if r == nil {
		p.logger.Debug("eou.eager_dispatch_skipped: llm unavailable",
			"item_id", respItemID, "score", score)
		return
	}
	epoch, ok := p.phase.onPredictedDispatch(ResponseID(predID), ItemID(respItemID), score, r)
	if !ok {

		r.abort()

		sc := score
		p.emitEOU("eager_already_active", inspect.EOUFields{
			Score: &sc,
			Extra: map[string]any{"item_id": respItemID},
		})
		p.logger.Debug("eou.eager_already_active",
			"item_id", respItemID, "score", score)
		return
	}
	spanRunnerStarted = true

	p.logger.Info("eou.eager_dispatch",
		"id", predID,
		"item_id", respItemID,
		"eou.score", score,
		"eou.eager_p_threshold", cfg.EagerPThreshold,
	)
	sc := score
	p.emitEOU("eager_dispatch", inspect.EOUFields{
		Score: &sc,
		Extra: map[string]any{"id": predID, "epoch": epoch},
	})
}

func (p *sessionPipeline) armBargeInTask(itemID string, startMs int64) {
	p.cancelBargeInTask()
	cancel := make(chan struct{})
	p.bargeInMu.Lock()
	p.bargeInCancel = cancel
	p.bargeInMu.Unlock()

	delay := time.Duration(p.session.BargeInDelayMs) * time.Millisecond
	if p.inspector != nil {
		p.inspector.Emit("bargein.pending",
			"item_id", itemID, "delay_ms", p.session.BargeInDelayMs)
	}
	p.wg.Add(1)
	go func() {
		defer p.wg.Done()
		select {
		case <-cancel:
			if p.inspector != nil {
				p.inspector.Emit("bargein.suppressed", "item_id", itemID)
			}
			p.logger.Info("barge_in suppressed: speech ended during delay",
				"item_id", itemID, "delay_ms", p.session.BargeInDelayMs)
			return
		case <-p.closed:
			return
		case <-time.After(delay):
		}

		var snap func() int64
		if out := p.getOutboundTTS(); out != nil {
			snap = out.PlayedMs
		}
		eff := p.phase.onVadSpeechStart(itemID, startMs, snap)
		if eff.cancel.cancelled {
			if p.inspector != nil {
				p.inspector.Emit("bargein.fired",
					"id", eff.cancel.id,
					"played_ms", eff.cancel.playedMs)
			}
			p.handleBargeIn(eff.cancel)
		}
		p.emitInputBufferSpeechStarted(itemID, startMs)
		p.startPartialLoop(itemID)
	}()
}

func (p *sessionPipeline) cancelBargeInTask() {
	p.bargeInMu.Lock()
	defer p.bargeInMu.Unlock()
	if p.bargeInCancel != nil {
		select {
		case <-p.bargeInCancel:
		default:
			close(p.bargeInCancel)
		}
		p.bargeInCancel = nil
	}
}

func (p *sessionPipeline) rollbackPredictedIfAny(reason string) bool {
	_, rbSpan := startSpan(p.traceContext(), "eou.predicted_rollback",
		attribute.String("eou.reason", reason),
	)
	var spanRunnerAborted bool
	defer func() {
		rbSpan.SetAttributes(attribute.Bool("eou.runner_aborted", spanRunnerAborted))
		rbSpan.End()
	}()

	id, _, r, ok := p.phase.onPredictedRollback()
	if !ok {
		return false
	}
	if r != nil {
		r.abort()
		spanRunnerAborted = true
	}
	p.predictedMu.Lock()
	cancel := p.predictedCancel
	p.predictedCancel = nil
	p.predictedMu.Unlock()
	if cancel != nil {
		cancel()
	}
	p.logger.Info("eou.predicted_rollback",
		"id", id,
		"reason", reason,
	)
	p.eouMu.Lock()
	eouKind := string(p.eouCfg.Kind)
	p.eouMu.Unlock()
	cb := reason
	if reason == "speech_resumed" {
		cb = "speech_started"
	}
	p.emitEOU("cancelled", inspect.EOUFields{
		EouKind:     eouKind,
		CancelledBy: cb,
		Extra:       map[string]any{"id": id, "rollback_reason": reason},
	})
	return true
}

func (p *sessionPipeline) cancelCommitTimer() {
	p.timerMu.Lock()
	defer p.timerMu.Unlock()
	if p.commitTimer != nil {
		p.commitTimer.Stop()
		p.commitTimer = nil
	}

	if p.commitHardCapTimer != nil {
		p.commitHardCapTimer.Stop()
		p.commitHardCapTimer = nil
	}
	if p.commitCancel != nil {
		select {
		case <-p.commitCancel:
		default:
			close(p.commitCancel)
		}
		p.commitCancel = nil
	}
	p.commitFire = nil
	p.commitItemID = ""
	p.commitSamples = nil
	p.commitPartial = ""
	p.commitArmedAt = time.Time{}
	p.commitDeadline = time.Time{}
	p.commitHardCap = false
	p.partialMu.Lock()
	p.partialPending = ""
	p.partialReady = false
	p.partialMu.Unlock()
}

func (p *sessionPipeline) rescheduleCommitTimerForSilence(newSilenceMs int) bool {
	p.timerMu.Lock()
	defer p.timerMu.Unlock()
	if p.commitTimer == nil || p.commitCancel == nil {
		return false
	}
	armedAt := p.commitArmedAt
	if armedAt.IsZero() {
		return false
	}

	newDeadline := armedAt.Add(time.Duration(newSilenceMs) * time.Millisecond)
	now := time.Now()
	delay := newDeadline.Sub(now)
	if delay < 0 {
		delay = 0
	}
	p.commitTimer.Stop()
	p.commitDeadline = now.Add(delay)

	fire := p.commitFire
	if fire == nil {

		itemID := p.commitItemID
		samples := p.commitSamples
		partial := p.commitPartial
		hardCap := p.commitHardCap
		cancelCh := p.commitCancel
		p.commitTimer = time.AfterFunc(delay, func() {
			select {
			case <-cancelCh:
				return
			default:
			}
			p.fireCommitTimer(itemID, samples, partial, hardCap)
		})
		return true
	}
	p.commitTimer = time.AfterFunc(delay, func() {
		fire(false, "", 0)
	})
	return true
}

func (p *sessionPipeline) fireCommitTimer(itemID string, samples audio.MonoF32, partial string, hardCapFired bool) {
	audioMs := int64(len(samples)) * 1000 / int64(whisperSampleRate)
	_, fireSpan := startSpan(p.traceContext(), "commit.fire",
		attribute.String("commit.item_id", itemID),
		attribute.Int64("commit.audio_ms", audioMs),
	)
	var (
		spanSuppressResponse bool
		spanWasEagerPromote  bool
	)
	defer func() {
		fireSpan.SetAttributes(
			attribute.Bool("commit.suppress_response", spanSuppressResponse),
			attribute.Bool("commit.was_eager_promote", spanWasEagerPromote),
		)
		fireSpan.End()
	}()
	speechMs := audioMs
	if speechMs < int64(p.session.MinSpeechMs) {
		p.logger.Info("commit suppressed: below min_speech_ms",
			"speech_ms", speechMs, "min_speech_ms", p.session.MinSpeechMs)
		_ = p.phase.clearInputBuffer()
		p.emitErrorCode("input_audio_buffer_commit_empty",
			fmt.Sprintf("buffer too short: %d ms < %d ms", speechMs, p.session.MinSpeechMs))
		spanSuppressResponse = true
		return
	}
	if hardCapFired {
		p.rollbackPredictedIfAny("hard_cap")
	}
	eff := p.phase.onCommitTimerFire()
	if !eff.committed {
		p.logger.Debug("commit_timer fired but buffer not Stopped; skipping")
		return
	}

	suppressResponse := speechMs < int64(p.session.MinSpeechForResponseMs)
	spanSuppressResponse = suppressResponse
	if suppressResponse {
		p.logger.Info("backchannel: commit but suppressing create_response",
			"speech_ms", speechMs, "threshold_ms", p.session.MinSpeechForResponseMs)
		p.rollbackPredictedIfAny("backchannel_suppressed")
	}

	if !suppressResponse && p.session.Conversation && p.phase.currentEagerRunner() != nil {
		spanWasEagerPromote = true
	}
	p.emitInputBufferCommitted(string(eff.itemID))
	p.emitConversationItemCreated(conversationItemDetail{
		ID:     string(eff.itemID),
		Object: "realtime.item",
		Type:   "message",
		Status: "in_progress",
		Role:   "user",
		Content: []responseContentPart{{
			Type: "input_audio",
		}},
	})
	if partial != "" {
		p.runFromPartial(string(eff.itemID), partial, !suppressResponse)
	} else {
		p.runTranscription(samples, string(eff.itemID), !suppressResponse)
	}
	p.wg.Add(1)
	go func() {
		defer p.wg.Done()
		p.waitForTurnDone()
		p.resetTurn()
	}()
}

func (p *sessionPipeline) runFromPartial(itemID, transcript string, allowResponse bool) {
	p.wg.Add(1)
	go func() {
		defer p.wg.Done()
		t0 := time.Now()
		autoResp := p.session.Conversation && allowResponse
		p.phase.onTranscriptionComplete(ItemID(itemID), transcript, autoResp)
		p.emitTranscription(itemID, transcript)
		if autoResp {
			if p.promotePredicted(transcript, t0) {
				p.markTurnDone()
				return
			}
			p.startResponse(newItemID(), transcript, t0.UnixMilli(), nil, nil)
		}
		p.markTurnDone()
	}()
}

func (p *sessionPipeline) handleBargeIn(eff cancelEffect) {
	_, biSpan := startSpan(p.traceContext(), "bargein.cancel",
		attribute.String("bargein.response_id", eff.id),
		attribute.Int64("bargein.played_ms", eff.playedMs),
		attribute.Bool("bargein.was_drain", eff.wasDrain),
	)
	defer biSpan.End()
	p.emitResponseTerminal(eff.id, eff.itemID, eff.playedMs, "cancelled", nil)
	if eff.itemID != "" && eff.playedMs > 0 {
		_ = p.phase.truncateItem(ItemID(eff.itemID), Millis(eff.playedMs), eff.transcript)
		p.emitConversationItemAssistantTruncated(eff.itemID, eff.playedMs, eff.transcript)
	}
	p.emitInputBufferSpeechStarted(eff.itemID, 0)
}

func (s *Server) negotiate(ctx context.Context, offerSDP string, cfg sessionConfig) (string, error) {
	pc, err := webrtc.NewPeerConnection(webrtc.Configuration{
		ICEServers: []webrtc.ICEServer{},
	})
	if err != nil {
		return "", err
	}

	sessionID := newSessID()
	logger := slog.With("session", sessionID)
	logger.Info("new realtime session", "model", cfg.Model, "intent", cfg.Intent,
		"stt_model", cfg.TranscriptionModel)

	vadAdp, err := newVADAdapter(s.cfg.SileroVADPath)
	if err != nil {
		logger.Warn("silero unavailable, falling back to silence-timeout",
			"err", err, "path", s.cfg.SileroVADPath)
		vadAdp = nil
	}
	if vadAdp != nil {
		logger.Info("silero VAD active", "path", s.cfg.SileroVADPath)
	}

	cfg.Conversation = (cfg.Intent == "conversation")

	pipeline := newSessionPipelineWithID(s.cfg, cfg, logger, sessionID)
	pipeline.vad = vadAdp

	pc.OnDataChannel(func(dc *webrtc.DataChannel) {
		logger.Info("data channel arrived", "label", dc.Label(), "id", dc.ID())
		pipeline.attachChannel(dc)
		dc.OnMessage(func(msg webrtc.DataChannelMessage) {
			pipeline.handleClientEvent(msg.Data)
		})
		dc.OnOpen(func() {
			logger.Info("data channel open", "label", dc.Label())
			ev := sessionCreatedEvent{
				EventID: newEventID(),
				Type:    SETSessionCreated,
				Session: session{
					ID:                sessionID,
					Object:            SessionObjectRealtimeSession,
					Model:             cfg.Model,
					Modalities:        modalitiesForIntent(cfg.Conversation),
					InputAudioFormat:  cfg.InputAudioFormat,
					OutputAudioFormat: cfg.OutputAudioFormat,
				},
			}
			if cfg.TranscriptionModel != "" {
				ev.Session.InputAudioTranscription = &audioTranscriptionConfig{
					Model: cfg.TranscriptionModel,
				}
			}
			if err := sendFragmented(dc, ev, ev.EventID); err != nil {
				logger.Error("send session.created failed", "err", err)
			}
		})
	})

	pc.OnConnectionStateChange(func(state webrtc.PeerConnectionState) {
		logger.Info("pc state", "state", state.String())
		if state == webrtc.PeerConnectionStateFailed ||
			state == webrtc.PeerConnectionStateClosed ||
			state == webrtc.PeerConnectionStateDisconnected {
			pipeline.close()
			_ = pc.Close()
		}
	})

	pc.OnTrack(func(remote *webrtc.TrackRemote, _ *webrtc.RTPReceiver) {
		if remote.Kind() != webrtc.RTPCodecTypeAudio {
			return
		}
		logger.Info("inbound audio track",
			"codec", remote.Codec().MimeType,
			"clock", remote.Codec().ClockRate,
			"channels", remote.Codec().Channels,
		)
		go pipeline.runAudioLoop(remote)
	})

	if cfg.Conversation {

		out, err := newOutboundAudio(logger, extractOpusChannels(offerSDP))
		if err != nil {
			_ = pc.Close()
			return "", fmt.Errorf("outbound audio init: %w", err)
		}
		if _, err := pc.AddTrack(out.Track()); err != nil {
			out.Close()
			_ = pc.Close()
			return "", fmt.Errorf("AddTrack: %w", err)
		}
		pipeline.attachOutbound(out)
		logger.Info("outbound TTS track attached")
	}

	if err := pc.SetRemoteDescription(webrtc.SessionDescription{
		Type: webrtc.SDPTypeOffer,
		SDP:  normalizeOffer(offerSDP),
	}); err != nil {
		_ = pc.Close()
		return "", err
	}

	answer, err := pc.CreateAnswer(nil)
	if err != nil {
		_ = pc.Close()
		return "", err
	}

	gatherComplete := webrtc.GatheringCompletePromise(pc)
	if err := pc.SetLocalDescription(answer); err != nil {
		_ = pc.Close()
		return "", err
	}

	select {
	case <-gatherComplete:
	case <-time.After(2 * time.Second):
		logger.Warn("ICE gather timed out; returning current SDP")
	case <-ctx.Done():
		_ = pc.Close()
		return "", ctx.Err()
	}

	return filterOpusOnly(pc.LocalDescription().SDP), nil
}
