package realtime

import (
	"fmt"
	"log/slog"
	"net/http"
	"net/http/httptest"
	"sync"
	"testing"
	"time"

	"github.com/eordano/speaches-plus-go/internal/conversation"
	"github.com/eordano/speaches-plus-go/internal/eou"
)

func newFakeLLMServer(t *testing.T) *httptest.Server {
	t.Helper()
	return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		flusher, _ := w.(http.Flusher)
		fmt.Fprintf(w, "data: {\"choices\":[{\"delta\":{\"content\":%q}}]}\n\n", "ok")
		flusher.Flush()
		fmt.Fprintf(w, "data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"stop\"}]}\n\n")
		fmt.Fprintf(w, "data: [DONE]\n\n")
		flusher.Flush()
	}))
}

type transitionRecorder struct {
	mu          sync.Mutex
	transitions []string
}

func (r *transitionRecorder) hook(phase, from, to string) {
	r.mu.Lock()
	r.transitions = append(r.transitions, phase+":"+from+"->"+to)
	r.mu.Unlock()
}

func (r *transitionRecorder) saw(s string) bool {
	r.mu.Lock()
	defer r.mu.Unlock()
	for _, t := range r.transitions {
		if t == s {
			return true
		}
	}
	return false
}

func newPipelineForIntegratedTest(t *testing.T, signals []eou.IntegratedSignal, conv bool, minSpeechMs int) (*sessionPipeline, *transitionRecorder) {
	t.Helper()
	src := eou.NewFakeIntegrated(eou.FakeIntegratedScript{Signals: signals})
	cfg := Config{
		IntegratedSource: src,
		EOUConfig:        eou.Config{Kind: eou.KindIntegrated},
	}
	if conv {
		srv := newFakeLLMServer(t)
		t.Cleanup(srv.Close)
		cfg.LLM = conversation.NewLLM(srv.URL+"/v1", "")
	}
	sess := sessionConfig{Model: "x", Conversation: conv, MinSpeechMs: minSpeechMs}
	rec := &transitionRecorder{}
	p := &sessionPipeline{
		server:   cfg,
		session:  sess,
		logger:   slog.Default(),
		closed:   make(chan struct{}),
		chReady:  make(chan struct{}),
		turnDone: make(chan struct{}),
		buf16k:   make([]float32, 0, 4096),
	}
	p.session.fillDefaults()
	p.session.MinSpeechMs = minSpeechMs
	p.phase.setSealedRetention(p.session.SealedBufferRetentionCount)
	p.eouModel, p.eouCfg, _ = eou.Load(eou.Config{Kind: eou.KindIntegrated})
	p.eou = p.runEOU
	p.phase.startSession()
	p.inspector = noopInspector{}
	p.phase.transitionHook = rec.hook
	return p, rec
}

func TestIntegrated_EotPredicted_Commit(t *testing.T) {
	p, rec := newPipelineForIntegratedTest(t, []eou.IntegratedSignal{
		{Type: "stt.eot_predicted", PEot: 0.95, TranscriptSoFar: "hello there"},
	}, false, 0)
	defer p.close()
	p.startIntegratedConsumer()

	deadline := time.Now().Add(2 * time.Second)
	for time.Now().Before(deadline) {
		if rec.saw("top:listen->process") || rec.saw("top:idle->process") {
			return
		}
		sess, vad, buf, resp := p.phase.snapshotFull()
		if buf.Kind() == bufKindCommitted ||
			derivedTopName(sess.Kind(), vad.Kind(), resp.Kind(), buf.Kind()) == "process" {
			return
		}
		time.Sleep(20 * time.Millisecond)
	}
	t.Fatalf("commit/process transition never observed; transitions=%v", rec.transitions)
}

func TestIntegrated_EagerEot_DispatchesPredicted(t *testing.T) {
	p, rec := newPipelineForIntegratedTest(t, []eou.IntegratedSignal{
		{Type: "stt.eot_predicted", PEot: 0.1, PEagerEot: 0.9, TranscriptSoFar: "hi"},
	}, true, 0)
	defer p.close()
	p.startIntegratedConsumer()

	deadline := time.Now().Add(2 * time.Second)
	for time.Now().Before(deadline) {
		if rec.saw("resp:none->predicted") {
			return
		}
		time.Sleep(20 * time.Millisecond)
	}
	t.Fatalf("Predicted transition never observed; transitions=%v", rec.transitions)
}

func TestIntegrated_TurnResumed_RollsBack(t *testing.T) {
	p, rec := newPipelineForIntegratedTest(t, []eou.IntegratedSignal{
		{Type: "stt.eot_predicted", PEot: 0.1, PEagerEot: 0.9, TranscriptSoFar: "hi"},
		{Type: "stt.turn_resumed", Reason: "speech_resumed"},
	}, true, 0)
	defer p.close()
	p.startIntegratedConsumer()

	deadline := time.Now().Add(2 * time.Second)
	for time.Now().Before(deadline) {
		if rec.saw("resp:none->predicted") && rec.saw("resp:predicted->none") {
			return
		}
		time.Sleep(20 * time.Millisecond)
	}
	t.Fatalf("Predicted dispatch+rollback not both observed; transitions=%v", rec.transitions)
}
