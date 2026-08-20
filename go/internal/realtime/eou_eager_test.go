package realtime

import (
	"log/slog"
	"testing"
	"time"

	"github.com/eordano/speaches-plus-go/internal/conversation"
	"github.com/eordano/speaches-plus-go/internal/eou"
)

func TestPromotePredicted_NoPredictedActive(t *testing.T) {
	p := &sessionPipeline{
		logger:   slog.Default(),
		closed:   make(chan struct{}),
		chReady:  make(chan struct{}),
		turnDone: make(chan struct{}),
	}
	p.session.fillDefaults()
	p.phase.setSealedRetention(p.session.SealedBufferRetentionCount)
	p.phase.startSession()

	if ok := p.promotePredicted("hello", time.Now()); ok {
		t.Fatalf("promotePredicted should return false when no Predicted runner is active")
	}
}

func TestPromotePredicted_WithActivePredicted(t *testing.T) {
	srv := newFakeLLMServer(t)
	t.Cleanup(srv.Close)

	rec := &transitionRecorder{}
	p := &sessionPipeline{
		server: Config{
			LLM:       conversation.NewLLM(srv.URL+"/v1", ""),
			EOUConfig: eou.Config{Kind: eou.KindText, EagerPThreshold: 0.5},
		},
		session: sessionConfig{
			Model:        "x",
			Conversation: true,
		},
		logger:   slog.Default(),
		closed:   make(chan struct{}),
		chReady:  make(chan struct{}),
		turnDone: make(chan struct{}),
	}
	p.session.fillDefaults()
	p.phase.setSealedRetention(p.session.SealedBufferRetentionCount)
	p.phase.setEagerMaxInflight(1)
	p.eouCfg = eou.Config{Kind: eou.KindText, EagerPThreshold: 0.5}
	p.phase.startSession()
	p.inspector = noopInspector{}
	p.phase.transitionHook = rec.hook

	p.maybeDispatchEager("itm", "hello there", 0.9, nil)

	deadline := time.Now().Add(time.Second)
	for time.Now().Before(deadline) {
		if rec.saw("resp:none->predicted") {
			break
		}
		time.Sleep(10 * time.Millisecond)
	}
	if !rec.saw("resp:none->predicted") {
		t.Fatalf("Predicted dispatch never observed; transitions=%v", rec.transitions)
	}

	close(p.chReady)

	if ok := p.promotePredicted("hello there", time.Now()); !ok {
		t.Fatalf("promotePredicted with active Predicted should return true")
	}

	if !rec.saw("resp:predicted->created") {
		t.Fatalf("Predicted->Created transition never observed; transitions=%v", rec.transitions)
	}
}
