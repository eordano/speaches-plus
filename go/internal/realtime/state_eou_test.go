package realtime

import (
	"context"
	"testing"
	"time"

	"github.com/eordano/speaches-plus-go/internal/audio"
)

func topOf(s *phaseState) string {
	sess, vad, buf, resp := s.snapshotFull()
	return derivedTopName(sess.Kind(), vad.Kind(), resp.Kind(), buf.Kind())
}

func TestEOU_BargeInDuringCommitTimer(t *testing.T) {
	var s phaseState
	s.startSession()

	s.onVadSpeechStart("item_a", 0, nil)
	if _, _, buf, _ := s.snapshotFull(); topOf(&s) != "listen" || buf.Kind() != bufKindVoiced {
		t.Fatalf("after speech_start: top=%s buf=%+v", topOf(&s), buf)
	}

	itemID, _, ok := s.onVadSpeechEnd(1500)
	if !ok || itemID != "item_a" {
		t.Fatalf("speech_end: ok=%v item=%q", ok, itemID)
	}
	if _, _, buf, _ := s.snapshotFull(); topOf(&s) != "listen" || buf.Kind() != bufKindStopped {
		t.Fatalf("after speech_stop, expected Listen+Stopped (commit_timer running): top=%s buf=%+v", topOf(&s), buf)
	}

	eff := s.onVadSpeechStart("item_a", 0, nil)
	if !eff.cancelTimer {
		t.Fatalf("re-fire of speech_start during commit_timer must report cancelTimer=true; got %+v", eff)
	}
	if eff.cancel.cancelled {
		t.Fatalf("no response was active; cancel must NOT report cancelled response: %+v", eff.cancel)
	}

	if _, _, buf, _ := s.snapshotFull(); topOf(&s) != "listen" || buf.Kind() != bufKindVoiced {
		t.Fatalf("after re-fire during timer, buffer must revert to Voiced (Listen): top=%s buf=%+v", topOf(&s), buf)
	}
	if id := bufItemIDOf(s.buf); id != "item_a" {
		t.Fatalf("itemID must persist across timer cancel; got %q", id)
	}
}

func TestEOU_TimerFiresWithoutInterrupt(t *testing.T) {
	var s phaseState
	s.startSession()
	s.onVadSpeechStart("item_a", 0, nil)
	s.onVadSpeechEnd(2000)
	eff := s.onCommitTimerFire()
	if !eff.committed {
		t.Fatal("commit_timer fire must produce commitEffect.committed=true")
	}
	if eff.itemID != "item_a" || eff.endMs != 2000 {
		t.Fatalf("commit eff: %+v", eff)
	}
	if _, _, buf, _ := s.snapshotFull(); topOf(&s) != "process" || buf.Kind() != bufKindCommitted {
		t.Fatalf("after commit_timer fire: top=%s buf=%+v", topOf(&s), buf)
	}
	conv := s.conversationSnapshot()
	if len(conv) != 1 || conv[0].ID != "item_a" || conv[0].Role != "user" || conv[0].Status != itemInProgress {
		t.Fatalf("conversation after commit: %+v", conv)
	}
}

func TestEOU_CommitFireOnNonStoppedBufferIsNoOp(t *testing.T) {
	var s phaseState
	s.startSession()

	if eff := s.onCommitTimerFire(); eff.committed {
		t.Fatalf("commit_timer fire on Empty must be no-op")
	}

	s.onVadSpeechStart("item_a", 0, nil)
	if eff := s.onCommitTimerFire(); eff.committed {
		t.Fatalf("commit_timer fire on Voiced must be no-op")
	}
}

func TestEOU_StubVerdict_PEqualsOneGivesMinDelay(t *testing.T) {
	cfg := sessionConfig{EOUMinDelayMs: 500, EOUMaxDelayMs: 3000}
	p := &sessionPipeline{session: cfg}
	p.eou = p.stubEOU
	v := p.callEOU(context.Background(), "", nil, time.Time{})
	if v.score != 1.0 {
		t.Fatalf("stub EOU must report p=1.0, got %f", v.score)
	}
	if v.delayMs != 500 {
		t.Fatalf("stub EOU at p=1.0 must yield min_delay_ms (500), got %d", v.delayMs)
	}
}

type contractEOU struct {
	score float32
	minMs int
	maxMs int
}

func (c contractEOU) verdict() eouVerdict {
	delay := int(float32(c.minMs) + (float32(c.maxMs)-float32(c.minMs))*(1.0-c.score))
	return eouVerdict{score: c.score, delayMs: delay}
}

func TestEOU_ContractFixtures_PR2Must_LerpInverseToScore(t *testing.T) {
	cases := []struct {
		name    string
		eou     contractEOU
		wantMin int
		wantMax int
	}{
		{"score=1.0 -> min", contractEOU{score: 1.0, minMs: 500, maxMs: 3000}, 500, 500},
		{"score=0.0 -> max", contractEOU{score: 0.0, minMs: 500, maxMs: 3000}, 3000, 3000},
		{"score=0.5 -> midpoint", contractEOU{score: 0.5, minMs: 500, maxMs: 3000}, 1750, 1750},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			v := c.eou.verdict()
			if v.delayMs < c.wantMin || v.delayMs > c.wantMax {
				t.Fatalf("EOU contract: score=%f want delay∈[%d,%d] got %d",
					c.eou.score, c.wantMin, c.wantMax, v.delayMs)
			}
		})
	}
}

func TestEOU_ContractFixtures_PR2Must_BargeInWithRealDelay(t *testing.T) {
	var s phaseState
	s.startSession()

	s.onVadSpeechStart("item_a", 0, nil)
	s.onVadSpeechEnd(1500)

	if _, _, buf, _ := s.snapshotFull(); buf.Kind() != bufKindStopped {
		t.Fatalf("PR2 contract: after speech_stop buffer MUST be Stopped (regardless of EOU delay), got %+v", buf)
	}

	eff := s.onVadSpeechStart("item_a", 0, nil)
	if !eff.cancelTimer {
		t.Fatalf("PR2 contract: speech_start during ANY non-zero commit delay MUST cancel the timer; got %+v", eff)
	}
	if _, _, buf, _ := s.snapshotFull(); buf.Kind() != bufKindVoiced {
		t.Fatalf("PR2 contract: after timer cancel, buffer MUST revert to Voiced; got %+v", buf)
	}
}

func TestEOU_ContractFixtures_PR2Must_NoCommittedEventWhenCancelled(t *testing.T) {
	var s phaseState
	s.startSession()

	s.onVadSpeechStart("item_a", 0, nil)
	s.onVadSpeechEnd(1000)

	s.onVadSpeechStart("item_a", 0, nil)

	if eff := s.onCommitTimerFire(); eff.committed {
		t.Fatalf("PR2 contract: after timer cancellation, commit_timer fire MUST NOT commit (got %+v)", eff)
	}
}

func TestEOU_ContractFixtures_PR2Must_TopLevelStaysListen(t *testing.T) {
	var s phaseState
	s.startSession()

	s.onVadSpeechStart("item_a", 0, nil)
	s.onVadSpeechEnd(1500)

	if topOf(&s) != "listen" {
		t.Fatalf("PR2 contract: top-level MUST stay Listen during commit_timer; got %s", topOf(&s))
	}

	s.onVadSpeechStart("item_a", 0, nil)
	if topOf(&s) != "listen" {
		t.Fatalf("PR2 contract: top-level stays Listen after timer cancel; got %s", topOf(&s))
	}
}

func TestEOU_ContractFixtures_PR2Must_CallEOUOnce_PerStop(t *testing.T) {
	calls := 0
	cfg := sessionConfig{EOUMinDelayMs: 0, EOUMaxDelayMs: 3000}
	p := &sessionPipeline{session: cfg}
	p.eou = func(ctx context.Context, partial string, _ audio.MonoF32, _ time.Time) eouVerdict {
		calls++
		return eouVerdict{score: 1.0, delayMs: 0}
	}
	_ = p.callEOU(context.Background(), "", nil, time.Time{})
	_ = p.callEOU(context.Background(), "", nil, time.Time{})
	if calls != 2 {
		t.Fatalf("PR2 contract: each callEOU MUST invoke the model once (got %d calls)", calls)
	}
}
