package realtime

import (
	"fmt"
	"math/rand"
	"sync"
	"sync/atomic"
	"testing"
)

func runPhaseFuzzConcurrent(t *testing.T, seed int64, totalOps, workers int) {
	t.Helper()
	var s phaseState

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
		opSimultaneousSpeechStarts
		opEouHardCap
		opCount
	)

	var stepCounter atomic.Int64
	var wg sync.WaitGroup
	for w := 0; w < workers; w++ {
		w := w
		wg.Add(1)
		go func() {
			defer wg.Done()
			rng := rand.New(rand.NewSource(seed ^ (int64(w+1) * 0x9e3779b9)))
			uniq := func(prefix string) string {
				return fmt.Sprintf("%s_%d_%d_%d", prefix, w, rng.Intn(1<<20), rng.Intn(1<<20))
			}
			for {
				step := stepCounter.Add(1)
				if step > int64(totalOps) {
					return
				}
				_, _, beforeResp := s.snapshot()
				respEpoch := respEpochOf(beforeResp)
				switch op(rng.Intn(int(opCount))) {
				case opStart:
					s.startSession()
				case opUpdate:
					_ = s.updateSession()
				case opVadStart:
					_ = s.onVadSpeechStart(uniq("i"), int64(step), nil)
				case opVadEnd:
					_, _, _ = s.onVadSpeechEnd(Millis(step + 100))
				case opCommitFire:
					_ = s.onCommitTimerFire()
				case opTransComplete:
					s.onTranscriptionComplete(ItemID(uniq("i")), "x", true)
				case opRespCreate:
					_, _ = s.onResponseCreate(ResponseID(uniq("r")), ItemID(uniq("i")))
				case opRespCancel:
					_ = s.onResponseCancel()
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
						_, _, _, _ = s.onResponseDoneEmitted(respEpoch)
					}
				case opPredictDispatch:
					_, _ = s.onPredictedDispatch(ResponseID(uniq("r")), ItemID(uniq("i")), float32(rng.Intn(100))/100, &eagerRunner{})
				case opPredictRollback:
					_, _, _, _ = s.onPredictedRollback()
				case opPredictPromote:
					_, _, _, _ = s.onPredictedPromote(respEpoch)
				case opTruncate:
					s.truncateItem(ItemID(uniq("i")), Millis(rng.Intn(2000)), "x")
				case opDelete:
					s.deleteItem(ItemID(uniq("i")))
				case opClearBuf:
					s.clearInputBuffer()
				case opStartResponseCreate:
					_, _ = s.onResponseCreate(ResponseID(uniq("r")), ItemID(uniq("i")))
				case opSimultaneousSpeechStarts:
					_ = s.onVadSpeechStart(uniq("i"), int64(step), nil)
					_ = s.onVadSpeechStart(uniq("i"), int64(step)+1, nil)
				case opEouHardCap:
					_ = s.onCommitTimerFire()
				}
			}
		}()
	}
	wg.Wait()

	s.mu.Lock()
	defer s.mu.Unlock()
	if err := checkInvariants(&s); err != nil {
		t.Fatalf("post-run invariant violation: %v", err)
	}
}

func TestPhase_FuzzInvariantsHold_Concurrent(t *testing.T) {
	if testing.Short() {
		t.Skip("concurrent fuzz: skipping in -short")
	}
	runPhaseFuzzConcurrent(t, 1, 5000, 8)
}

func TestPhase_FuzzInvariantsHold_Concurrent_MultiSeed(t *testing.T) {
	if testing.Short() {
		t.Skip("concurrent fuzz multi-seed: skipping in -short")
	}
	for _, seed := range []int64{2, 3, 7, 13, 42, 101} {
		seed := seed
		t.Run(fmt.Sprintf("seed=%d", seed), func(t *testing.T) {
			t.Parallel()
			runPhaseFuzzConcurrent(t, seed, 1500, 6)
		})
	}
}

func TestPhase_FuzzInvariantsHold_MultiSession(t *testing.T) {
	if testing.Short() {
		t.Skip("multi-session fuzz: skipping in -short")
	}
	const sessions = 16
	const stepsPerSession = 1500

	var wg sync.WaitGroup
	errs := make(chan error, sessions)
	for sess := 0; sess < sessions; sess++ {
		sess := sess
		wg.Add(1)
		go func() {
			defer wg.Done()
			defer func() {
				if r := recover(); r != nil {
					errs <- fmt.Errorf("session %d panic: %v", sess, r)
				}
			}()
			runPhaseFuzzSequentialForSession(t, int64(sess)^0xdeadbeef, stepsPerSession)
		}()
	}
	wg.Wait()
	close(errs)
	for err := range errs {
		if err != nil {
			t.Fatalf("multi-session: %v", err)
		}
	}
}

func runPhaseFuzzSequentialForSession(t *testing.T, seed int64, steps int) {
	rng := rand.New(rand.NewSource(seed))
	var s phaseState

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
			_ = s.onVadSpeechStart(uniq("i"), int64(i), nil)
		case opVadEnd:
			_, _, _ = s.onVadSpeechEnd(Millis(i + 100))
		case opCommitFire:
			_ = s.onCommitTimerFire()
		case opTransComplete:
			s.onTranscriptionComplete(ItemID(uniq("i")), "x", true)
		case opRespCreate:
			_, _ = s.onResponseCreate(ResponseID(uniq("r")), ItemID(uniq("i")))
		case opRespCancel:
			_ = s.onResponseCancel()
		case opUpDelta:
			s.onUpstreamDelta(respEpoch, "x", 10)
		case opLLMDone:
			s.onLLMComplete(respEpoch)
		case opAudioDrained:
			if s.onAudioDrained(respEpoch) {
				_, _, _, _ = s.onResponseDoneEmitted(respEpoch)
			}
		case opPredictDispatch:
			_, _ = s.onPredictedDispatch(ResponseID(uniq("r")), ItemID(uniq("i")), float32(rng.Intn(100))/100, &eagerRunner{})
		case opPredictRollback:
			_, _, _, _ = s.onPredictedRollback()
		case opPredictPromote:
			_, _, _, _ = s.onPredictedPromote(respEpoch)
		case opTruncate:
			s.truncateItem(ItemID(uniq("i")), Millis(rng.Intn(2000)), "x")
		case opDelete:
			s.deleteItem(ItemID(uniq("i")))
		case opClearBuf:
			s.clearInputBuffer()
		}
	}
	_ = t
}
