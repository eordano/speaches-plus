package realtime

import (
	"io"
	"log/slog"
	"testing"
	"time"
)

func TestArmBargeInTask_SimultaneousReplacesSlot(t *testing.T) {
	cfg := sessionConfig{
		BargeInDelayMs: 1000,
	}
	p := &sessionPipeline{
		session: cfg,
		logger:  slog.New(slog.NewTextHandler(io.Discard, nil)),
	}
	insp := &recordingInspector{}
	p.inspector = insp
	p.closed = make(chan struct{})

	p.armBargeInTask("item_a", 100)
	p.bargeInMu.Lock()
	firstCancel := p.bargeInCancel
	p.bargeInMu.Unlock()
	if firstCancel == nil {
		t.Fatalf("first arm: bargeInCancel must be set")
	}
	time.Sleep(20 * time.Millisecond)

	p.armBargeInTask("item_b", 200)

	deadline := time.Now().Add(500 * time.Millisecond)
	var supp []recordedEvent
	for time.Now().Before(deadline) {
		supp = insp.byName("bargein.suppressed")
		if len(supp) == 1 {
			break
		}
		time.Sleep(5 * time.Millisecond)
	}
	if len(supp) != 1 {
		t.Fatalf("§9.5: first goroutine MUST emit bargein.suppressed; got %d events (%v)", len(supp), insp.events)
	}
	suppressedItemID, ok := attrValue(supp[0], "item_id")
	if !ok || suppressedItemID != "item_a" {
		t.Fatalf("§9.5: suppressed event MUST carry item_id of FIRST arm; got %v", suppressedItemID)
	}

	select {
	case <-firstCancel:
	default:
		t.Fatalf("§9.5: first arm's cancel chan MUST be closed by the second arm")
	}

	p.bargeInMu.Lock()
	secondCancel := p.bargeInCancel
	p.bargeInMu.Unlock()
	if secondCancel == nil {
		t.Fatalf("§9.5: bargeInCancel slot MUST be re-armed for the second event")
	}
	if secondCancel == firstCancel {
		t.Fatalf("§9.5: the slot's cancel chan MUST be a fresh chan, not the first arm's")
	}

	pending := insp.byName("bargein.pending")
	if len(pending) != 2 {
		t.Fatalf("§9.5: bargein.pending MUST emit on every arm; got %d", len(pending))
	}

	close(p.closed)
	p.wg.Wait()
}
