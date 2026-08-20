package realtime

import (
	"fmt"
	"math/rand"
	"strings"
	"testing"
)

func TestPhase_BargeInCancelsActiveResponse(t *testing.T) {
	var s phaseState
	s.startSession()
	if _, err := s.onResponseCreate("resp_a", "item_a"); err != nil {
		t.Fatalf("create: %v", err)
	}
	eff := s.onVadSpeechStart("item_x", 0, nil)
	if !eff.cancel.cancelled || eff.cancel.id != "resp_a" {
		t.Fatalf("barge-in must cancel resp_a, got %+v", eff)
	}
	_, vad, resp := s.snapshot()
	if vad.Kind() != vadKindSpeaking || resp.Kind() != respKindNone {
		t.Fatalf("post barge-in: vad=%+v resp=%+v", vad, resp)
	}
}

func TestPhase_DoubleResponseCreateRejected(t *testing.T) {
	var s phaseState
	s.startSession()
	if _, err := s.onResponseCreate("r1", "i1"); err != nil {
		t.Fatalf("first create: %v", err)
	}
	if _, err := s.onResponseCreate("r2", "i2"); err == nil {
		t.Fatalf("second create must error")
	}
}

func TestPhase_StaleDeltaAfterCancelDropped(t *testing.T) {
	var s phaseState
	s.startSession()
	epoch, err := s.onResponseCreate("r1", "i1")
	if err != nil {
		t.Fatalf("create: %v", err)
	}
	eff := s.onResponseCancel()
	if !eff.cancelled || eff.epoch != uint64(epoch) {
		t.Fatalf("cancel: %+v", eff)
	}
	if alive := s.onUpstreamDelta(epoch, "tail", 100); alive {
		t.Fatalf("stale delta must be rejected (epoch %d after cancel)", epoch)
	}
}

func TestPhase_SessionUpdateBeforeActiveErrors(t *testing.T) {
	var s phaseState
	if err := s.updateSession(); err == nil {
		t.Fatalf("update before active must error")
	}
	s.startSession()
	if err := s.updateSession(); err != nil {
		t.Fatalf("update after active failed: %v", err)
	}
}

func TestPhase_VadEndClearsSpeaking(t *testing.T) {
	var s phaseState
	s.startSession()
	s.onVadSpeechStart("item_q", 0, nil)
	s.onVadSpeechEnd(1000)
	_, vad, _ := s.snapshot()
	if vad.Kind() == vadKindSpeaking {
		t.Fatalf("vad still speaking after end: %+v", vad)
	}
}

func TestPhase_UpstreamDeltaAdvancesToStreaming(t *testing.T) {
	var s phaseState
	s.startSession()
	epoch, _ := s.onResponseCreate("r1", "i1")
	if !s.onUpstreamDelta(epoch, "hello", 320) {
		t.Fatalf("first delta should be alive")
	}
	_, _, resp := s.snapshot()
	rs, ok := resp.(RespStreaming)
	if !ok || rs.Transcript != "hello" || rs.PlannedMs != 320 {
		t.Fatalf("post delta: %+v", resp)
	}
}

func TestPhase_UpstreamCompleteClearsResponse(t *testing.T) {
	var s phaseState
	s.startSession()
	epoch, _ := s.onResponseCreate("r1", "i1")
	s.onUpstreamDelta(epoch, "hi there", 200)
	s.updatePlayedMs(epoch, 200)
	transcript, audioMs, ok := s.onUpstreamComplete(epoch)
	if !ok || transcript != "hi there" || audioMs != 200 {
		t.Fatalf("complete: ok=%v transcript=%q audio=%d", ok, transcript, audioMs)
	}
	_, _, resp := s.snapshot()
	if resp.Kind() != respKindNone {
		t.Fatalf("response not cleared after complete: %+v", resp)
	}
}

func TestPhase_TopLevelTransitions_CleanTurn(t *testing.T) {
	var s phaseState
	s.startSession()
	topNow := func() string {
		sess, vad, buf, resp := s.snapshotFull()
		return derivedTopName(sess.Kind(), vad.Kind(), resp.Kind(), buf.Kind())
	}
	if topNow() != "idle" {
		t.Fatalf("expected idle after start, got %s", topNow())
	}
	s.onVadSpeechStart("item_1", 0, nil)
	if _, _, buf, _ := s.snapshotFull(); topNow() != "listen" || buf.Kind() != bufKindVoiced {
		t.Fatalf("after speech_start: top=%s buf=%+v", topNow(), buf)
	}
	s.onVadSpeechEnd(1500)
	if _, _, buf, _ := s.snapshotFull(); topNow() != "listen" || buf.Kind() != bufKindStopped {
		t.Fatalf("after speech_stop: top=%s buf=%+v", topNow(), buf)
	}
	eff := s.onCommitTimerFire()
	if !eff.committed {
		t.Fatalf("commit_timer fire: not committed")
	}
	if _, _, buf, _ := s.snapshotFull(); topNow() != "process" || buf.Kind() != bufKindCommitted {
		t.Fatalf("after commit_timer: top=%s buf=%+v", topNow(), buf)
	}
	s.onTranscriptionComplete("item_1", "hello", true)
	if _, _, buf, _ := s.snapshotFull(); topNow() != "idle" || buf.Kind() != bufKindEmpty {
		t.Fatalf("after transcription: top=%s buf=%+v", topNow(), buf)
	}
	epoch, err := s.onResponseCreate("r1", "i1")
	if err != nil {
		t.Fatalf("create: %v", err)
	}
	if topNow() != "generate" {
		t.Fatalf("after create: top=%s", topNow())
	}
	s.onUpstreamDelta(epoch, "hi", 100)
	if !s.onLLMComplete(epoch) {
		t.Fatalf("onLLMComplete should succeed")
	}
	if _, _, _, resp := s.snapshotFull(); topNow() != "drain" || resp.Kind() != respKindDrain {
		t.Fatalf("after llm complete: top=%s resp=%+v", topNow(), resp)
	}
	if !s.onAudioDrained(epoch) {
		t.Fatalf("onAudioDrained should succeed")
	}
	if _, _, _, resp := s.snapshotFull(); resp.Kind() != respKindFinalized {
		t.Fatalf("after drain: top=%s resp=%+v", topNow(), resp)
	}
}

func TestPhase_GenerateBufferStaysEmpty(t *testing.T) {
	var s phaseState
	s.startSession()
	if _, err := s.onResponseCreate("r1", "i1"); err != nil {
		t.Fatalf("create: %v", err)
	}
	sess, vad, buf, resp := s.snapshotFull()
	top := derivedTopName(sess.Kind(), vad.Kind(), resp.Kind(), buf.Kind())
	if top != "generate" || buf.Kind() != bufKindEmpty {
		t.Fatalf("Generate must have Empty buffer: top=%s buf=%+v", top, buf)
	}
}

func TestPhase_ClearInputBufferReturnsToIdle(t *testing.T) {
	var s phaseState
	s.startSession()
	s.onVadSpeechStart("item_1", 0, nil)
	if !s.clearInputBuffer() {
		t.Fatalf("clearInputBuffer should report change")
	}
	sess, vad, buf, resp := s.snapshotFull()
	top := derivedTopName(sess.Kind(), vad.Kind(), resp.Kind(), buf.Kind())
	if top != "idle" || buf.Kind() != bufKindEmpty {
		t.Fatalf("after clear: top=%s buf=%+v", top, buf)
	}
}

func TestPhase_DeleteItem(t *testing.T) {
	var s phaseState
	s.startSession()
	s.onVadSpeechStart("item_1", 0, nil)
	s.onVadSpeechEnd(1000)
	eff := s.onCommitTimerFire()
	if !eff.committed {
		t.Fatal("expected commit")
	}
	if !s.deleteItem("item_1") {
		t.Fatalf("deleteItem must succeed")
	}
	conv := s.conversationSnapshot()
	for _, it := range conv {
		if it.ID == "item_1" {
			t.Fatalf("item_1 still present after delete")
		}
	}
}

func TestPhase_FuzzInvariantsHold_MultiSeed(t *testing.T) {
	if testing.Short() {
		t.Skip("multi-seed fuzz: skipping in -short")
	}
	for _, seed := range []int64{2, 3, 7, 13, 42, 101} {
		seed := seed
		t.Run(fmt.Sprintf("seed=%d", seed), func(t *testing.T) {
			runPhaseFuzz(t, seed, 1000)
		})
	}
}

func TestPhase_FuzzInvariantsHold(t *testing.T) {
	runPhaseFuzz(t, 1, 5000)
}

type fuzzEmitter struct {
	trace   []TraceEvent
	turnOff bool
}

func (e *fuzzEmitter) emit(ev TraceEvent) {
	e.trace = append(e.trace, ev)
}

func (e *fuzzEmitter) recordRespCreate(after RespPhase) {
	rc, ok := after.(RespCreated)
	if !ok {
		return
	}
	e.emit(TraceEvent{
		"type":     string(SETResponseCreated),
		"response": map[string]any{"id": string(rc.ID)},
	})
}

func (e *fuzzEmitter) recordRespCancel(eff cancelEffect, before RespPhase) {
	if !eff.cancelled || eff.id == "" {
		return
	}
	if before != nil {
		switch before.Kind() {
		case respKindNone, respKindPredicted, respKindFinalized:
			return
		}
	}
	e.emit(TraceEvent{
		"type": string(SETResponseDone),
		"response": map[string]any{
			"id":           eff.id,
			"status":       "cancelled",
			"audio_end_ms": eff.playedMs,
		},
	})
}

func (e *fuzzEmitter) recordRespCancelFromBargeIn(eff bargeInEffect, before RespPhase) {
	if eff.cancel.cancelled {
		e.recordRespCancel(eff.cancel, before)
	}
}

func (e *fuzzEmitter) recordRespDoneFromFinalized(before, after RespPhase) {
	rf, ok := after.(RespFinalized)
	if !ok {
		return
	}
	if before != nil && before.Kind() == respKindFinalized {
		return
	}
	e.emit(TraceEvent{
		"type": string(SETResponseDone),
		"response": map[string]any{
			"id":           string(rf.ID),
			"status":       rf.Status.String(),
			"audio_end_ms": int64(rf.PlayedMs),
		},
	})
}

func (e *fuzzEmitter) recordVadStart(after VadPhase) {
	vs, ok := after.(VadSpeaking)
	if !ok {
		return
	}
	e.emit(TraceEvent{
		"type":    string(SETInputBufferSpeechStarted),
		"item_id": string(vs.ItemID),
	})
}

func (e *fuzzEmitter) recordVadEnd(itemID ItemID, ok bool) {
	if !ok || itemID == "" {
		return
	}
	e.emit(TraceEvent{
		"type":    string(SETInputBufferSpeechStopped),
		"item_id": string(itemID),
	})
}

func (e *fuzzEmitter) recordCommit(eff commitEffect) {
	if !eff.committed || eff.itemID == "" {
		return
	}
	e.emit(TraceEvent{
		"type":    string(SETInputBufferCommitted),
		"item_id": string(eff.itemID),
	})
	e.emit(TraceEvent{
		"type": string(SETConversationItemAdded),
		"item": map[string]any{"id": string(eff.itemID)},
	})
}

func (e *fuzzEmitter) recordTranscriptionComplete(itemID ItemID) {
	if itemID == "" {
		return
	}
	e.emit(TraceEvent{
		"type":    string(SETInputAudioTranscriptionCompleted),
		"item_id": string(itemID),
	})
}

func runPhaseFuzz(t *testing.T, seed int64, steps int) {
	t.Helper()
	rng := rand.New(rand.NewSource(seed))
	var s phaseState
	emitter := &fuzzEmitter{}

	type op int
	const (
		opStart op = iota
		opUpdate
		opVadStart
		opVadEnd
		opCommitFire
		opTransComplete
		opRespCreate
		opRespCancel
		opUpDelta
		opLLMDone
		opAudioDrained
		opPredictDispatch
		opPredictRollback
		opPredictPromote
		opTruncate
		opDelete
		opClearBuf
		opStartResponseCreate
		opSessionUpdateTurnDetectionNone
		opSessionUpdateRevert
		opSimultaneousSpeechStarts
		opEouHardCap
		opCount
	)

	uniq := func(prefix string) string {
		return fmt.Sprintf("%s_%d_%d_%d", prefix, seed, rng.Intn(1<<20), rng.Intn(1<<20))
	}

	for i := 0; i < steps; i++ {
		_, _, beforeResp := s.snapshot()
		respEpoch := respEpochOf(beforeResp)
		switch op(rng.Intn(int(opCount))) {
		case opStart:
			s.startSession()
		case opUpdate:
			_ = s.updateSession()
		case opVadStart:
			eff := s.onVadSpeechStart(uniq("i"), int64(i), nil)
			emitter.recordRespCancelFromBargeIn(eff, beforeResp)
			_, vad, _ := s.snapshot()
			emitter.recordVadStart(vad)
		case opVadEnd:
			itemID, _, ok := s.onVadSpeechEnd(Millis(i + 100))
			emitter.recordVadEnd(itemID, ok)
		case opCommitFire:
			eff := s.onCommitTimerFire()
			emitter.recordCommit(eff)
		case opTransComplete:
			id := ItemID(uniq("i"))
			s.onTranscriptionComplete(id, "x", true)
			emitter.recordTranscriptionComplete(id)
		case opRespCreate:
			_, err := s.onResponseCreate(ResponseID(uniq("r")), ItemID(uniq("i")))
			if err == nil {
				_, _, after := s.snapshot()
				emitter.recordRespCreate(after)
			}
		case opRespCancel:
			eff := s.onResponseCancel()
			emitter.recordRespCancel(eff, beforeResp)
		case opUpDelta:
			epoch := respEpoch
			if rng.Intn(4) == 0 {
				epoch += Epoch(rng.Intn(3))
			}
			s.onUpstreamDelta(epoch, "x", 10)
		case opLLMDone:
			s.onLLMComplete(respEpoch)
		case opAudioDrained:
			if s.onAudioDrained(respEpoch) {
				_, _, after := s.snapshot()
				emitter.recordRespDoneFromFinalized(beforeResp, after)
				_, _, _, _ = s.onResponseDoneEmitted(respEpoch)
			}
		case opPredictDispatch:
			_, _ = s.onPredictedDispatch(ResponseID(uniq("r")), ItemID(uniq("i")), float32(rng.Intn(100))/100, &eagerRunner{})
		case opPredictRollback:
			_, _, _, _ = s.onPredictedRollback()
		case opPredictPromote:
			id, _, _, ok := s.onPredictedPromote(respEpoch)
			if ok && id != "" {
				_, _, after := s.snapshot()
				emitter.recordRespCreate(after)
			}
		case opTruncate:
			s.truncateItem(ItemID(uniq("i")), Millis(rng.Intn(2000)), "x")
		case opDelete:
			s.deleteItem(ItemID(uniq("i")))
		case opClearBuf:
			s.clearInputBuffer()
		case opStartResponseCreate:
			_, err := s.onResponseCreate(ResponseID(uniq("r")), ItemID(uniq("i")))
			if err == nil {
				_, _, after := s.snapshot()
				emitter.recordRespCreate(after)
			}
		case opSessionUpdateTurnDetectionNone:
			_ = s.updateSession()
			emitter.turnOff = true
		case opSessionUpdateRevert:
			_ = s.updateSession()
			emitter.turnOff = false
		case opSimultaneousSpeechStarts:
			eff1 := s.onVadSpeechStart(uniq("i"), int64(i), nil)
			emitter.recordRespCancelFromBargeIn(eff1, beforeResp)
			_, vad1, _ := s.snapshot()
			emitter.recordVadStart(vad1)
			_, _, midResp := s.snapshot()
			eff2 := s.onVadSpeechStart(uniq("i"), int64(i)+1, nil)
			emitter.recordRespCancelFromBargeIn(eff2, midResp)
		case opEouHardCap:
			eff := s.onCommitTimerFire()
			emitter.recordCommit(eff)
		}
		if err := func() error {
			s.mu.Lock()
			defer s.mu.Unlock()
			return checkInvariants(&s)
		}(); err != nil {
			t.Fatalf("step %d: %v", i, err)
		}
	}

	_, _, finalResp := s.snapshot()
	switch r := finalResp.(type) {
	case RespCreated:
		emitter.emit(TraceEvent{
			"type": string(SETResponseDone),
			"response": map[string]any{
				"id":           string(r.ID),
				"status":       "cancelled",
				"audio_end_ms": int64(0),
			},
		})
	case RespStreaming:
		emitter.emit(TraceEvent{
			"type": string(SETResponseDone),
			"response": map[string]any{
				"id":           string(r.ID),
				"status":       "cancelled",
				"audio_end_ms": int64(r.PlayedMs),
			},
		})
	case RespDrain:
		emitter.emit(TraceEvent{
			"type": string(SETResponseDone),
			"response": map[string]any{
				"id":           string(r.ID),
				"status":       "completed",
				"audio_end_ms": int64(r.PlayedMs),
			},
		})
	case RespFinalized:
		emitter.emit(TraceEvent{
			"type": string(SETResponseDone),
			"response": map[string]any{
				"id":           string(r.ID),
				"status":       r.Status.String(),
				"audio_end_ms": int64(r.PlayedMs),
			},
		})
	}

	if vios := AssertTraceInvariants(emitter.trace); len(vios) != 0 {
		t.Fatalf("trace invariants violated (seed=%d, %d events): %s",
			seed, len(emitter.trace), strings.Join(vios, "; "))
	}
}

func TestPhase_FuzzBargeInAtomicity(t *testing.T) {
	rng := rand.New(rand.NewSource(7))
	const (
		startCreated   = 0
		startStreaming = 1
		startPredicted = 2
		startCount     = 3
	)
	for trial := 0; trial < 300; trial++ {
		var s phaseState
		s.startSession()
		startKind := rng.Intn(startCount)
		var expectCancelEffect bool
		switch startKind {
		case startCreated:
			if _, err := s.onResponseCreate(
				ResponseID(fmt.Sprintf("r%d", trial)),
				ItemID(fmt.Sprintf("i%d", trial)),
			); err != nil {
				t.Fatalf("trial %d: create: %v", trial, err)
			}
			expectCancelEffect = true
		case startStreaming:
			if _, err := s.onResponseCreate(
				ResponseID(fmt.Sprintf("r%d", trial)),
				ItemID(fmt.Sprintf("i%d", trial)),
			); err != nil {
				t.Fatalf("trial %d: create: %v", trial, err)
			}
			s.onUpstreamDelta(respEpochOf(s.resp), "partial", 100)
			expectCancelEffect = true
		case startPredicted:
			if _, ok := s.onPredictedDispatch(
				ResponseID(fmt.Sprintf("p%d", trial)),
				ItemID(fmt.Sprintf("pi%d", trial)),
				0.7,
				&eagerRunner{},
			); !ok {
				t.Fatalf("trial %d: predicted dispatch failed", trial)
			}
			expectCancelEffect = false
		}

		eff := s.onVadSpeechStart("i", int64(trial), nil)
		if expectCancelEffect && !eff.cancel.cancelled {
			t.Fatalf("trial %d (kind=%d): barge-in must cancel", trial, startKind)
		}
		if !expectCancelEffect && eff.cancel.cancelled {
			t.Fatalf("trial %d (kind=%d): predicted barge-in must NOT report cancel (I7)",
				trial, startKind)
		}
		if startKind == startPredicted {
			if !eff.predictedRolled {
				t.Fatalf("trial %d: predicted barge-in must set predictedRolled", trial)
			}
			if eff.runnerToAbort == nil {
				t.Fatalf("trial %d: predicted barge-in must hand back runner", trial)
			}
		}
		if startKind != startPredicted && eff.predictedRolled {
			t.Fatalf("trial %d (kind=%d): predictedRolled set on non-predicted barge-in",
				trial, startKind)
		}
		s.mu.Lock()
		if s.vad.Kind() != vadKindSpeaking {
			s.mu.Unlock()
			t.Fatalf("trial %d (kind=%d): vad not speaking after start", trial, startKind)
		}
		switch startKind {
		case startCreated, startStreaming:
			if s.resp.Kind() != respKindNone {
				s.mu.Unlock()
				t.Fatalf("trial %d (kind=%d): resp not cleared (got %s)",
					trial, startKind, s.resp.Kind())
			}
		case startPredicted:
			if s.resp.Kind() != respKindNone {
				s.mu.Unlock()
				t.Fatalf("trial %d (kind=%d): predicted resp not cleared (got %s)",
					trial, startKind, s.resp.Kind())
			}
			if s.inflightPredicted != 0 {
				s.mu.Unlock()
				t.Fatalf("trial %d (kind=%d): inflightPredicted=%d after barge-in (want 0)",
					trial, startKind, s.inflightPredicted)
			}
		}
		if err := checkInvariants(&s); err != nil {
			s.mu.Unlock()
			t.Fatalf("trial %d (kind=%d): invariant violated: %v",
				trial, startKind, err)
		}
		s.mu.Unlock()
	}
}

func TestPhase_SoakSealedBufRetention(t *testing.T) {
	const (
		retention = 4
		cycles    = 1000
	)
	var s phaseState
	s.setSealedRetention(retention)
	s.startSession()

	maxObserved := 0
	for i := 0; i < cycles; i++ {
		itemID := ItemID(fmt.Sprintf("soak_item_%d", i))
		s.onVadSpeechStart(string(itemID), int64(i)*100, nil)
		s.onVadSpeechEnd(Millis(int64(i)*100 + 50))
		eff := s.onCommitTimerFire()
		if !eff.committed {
			t.Fatalf("iter %d: commit_timer must fire", i)
		}
		s.mu.Lock()
		mid := len(s.sealedBufs)
		s.mu.Unlock()
		if mid > maxObserved {
			maxObserved = mid
		}
		if mid > retention {
			t.Fatalf("iter %d: sealedBufs exceeded retention pre-transcription: got %d cap %d",
				i, mid, retention)
		}
		s.onTranscriptionComplete(itemID, "soak", true)
		s.mu.Lock()
		post := len(s.sealedBufs)
		s.mu.Unlock()
		if post > retention {
			t.Fatalf("iter %d: sealedBufs exceeded retention post-transcription: got %d cap %d",
				i, post, retention)
		}
		if err := func() error {
			s.mu.Lock()
			defer s.mu.Unlock()
			return checkInvariants(&s)
		}(); err != nil {
			t.Fatalf("iter %d: invariant: %v", i, err)
		}
	}
	s.mu.Lock()
	final := len(s.sealedBufs)
	s.mu.Unlock()
	if final > retention {
		t.Fatalf("final sealedBufs len=%d exceeds retention=%d", final, retention)
	}
	t.Logf("soak: %d cycles, retention=%d, max sealedBufs observed=%d, final=%d",
		cycles, retention, maxObserved, final)
}
