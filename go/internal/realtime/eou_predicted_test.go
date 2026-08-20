package realtime

import (
	"errors"
	"reflect"
	"sync"
	"testing"
	"time"

	"github.com/eordano/speaches-plus-go/internal/audio"
)

func TestPredictedTokenBufferDropsOldest(t *testing.T) {
	b := NewPredictedTokenBuffer(3)
	b.Push("a")
	b.Push("b")
	b.Push("c")
	if b.DroppedCount() != 0 {
		t.Fatalf("dropped=%d want 0", b.DroppedCount())
	}
	b.Push("d")
	b.Push("e")
	if b.DroppedCount() != 2 {
		t.Fatalf("dropped=%d want 2", b.DroppedCount())
	}
	out := b.Drain()
	want := []string{"c", "d", "e"}
	if !reflect.DeepEqual(out, want) {
		t.Fatalf("drain=%v want %v", out, want)
	}
}

func TestPredictedTokenBufferDefaultCapAtLeastOne(t *testing.T) {
	b := NewPredictedTokenBuffer(0)
	b.Push("only")
	if b.Len() != 1 {
		t.Fatalf("len=%d want 1", b.Len())
	}
	b.Push("overflow")
	if b.DroppedCount() < 1 {
		t.Fatalf("dropped=%d want>=1", b.DroppedCount())
	}
}

func TestPredictedTokenBufferDrainResets(t *testing.T) {
	b := NewPredictedTokenBuffer(4)
	b.Push("x")
	b.Push("y")
	if got := b.Drain(); !reflect.DeepEqual(got, []string{"x", "y"}) {
		t.Fatalf("drain=%v want [x y]", got)
	}
	if b.Len() != 0 || !b.IsEmpty() {
		t.Fatalf("expected empty after drain")
	}
	b.Push("z")
	if b.Len() != 1 {
		t.Fatalf("expected 1 after refill, got %d", b.Len())
	}
}

func TestTranscriptsMatchWhenEqual(t *testing.T) {
	if transcriptsMateriallyDiffer("hello there", "hello there", 0.5) {
		t.Fatal("equal transcripts must match")
	}
}

func TestTranscriptsMatchWithMinorPunctuation(t *testing.T) {
	if transcriptsMateriallyDiffer("hello there", "hello there.", 0.5) {
		t.Fatal("minor punctuation must not trip mismatch")
	}
}

func TestTranscriptsDivergeWhenCompletelyDifferent(t *testing.T) {
	if !transcriptsMateriallyDiffer("tell me about cats", "what is the weather", 0.5) {
		t.Fatal("totally different transcripts must mismatch")
	}
}

func TestTranscriptsOneEmptyDiverges(t *testing.T) {
	if !transcriptsMateriallyDiffer("hello", "", 0.5) {
		t.Fatal("hello vs empty: expected mismatch")
	}
	if !transcriptsMateriallyDiffer("", "hello", 0.5) {
		t.Fatal("empty vs hello: expected mismatch")
	}
}

func TestTranscriptsBothEmptyMatch(t *testing.T) {
	if transcriptsMateriallyDiffer("", "", 0.5) {
		t.Fatal("empty vs empty: expected match")
	}
}

func TestTranscriptsCaseInsensitive(t *testing.T) {

	if transcriptsMateriallyDiffer("HELLO", "hello", 0.5) {
		t.Fatal("case-insensitive match expected")
	}
	if transcriptsMateriallyDiffer("  hello  ", "hello", 0.5) {
		t.Fatal("trim-then-compare match expected")
	}
}

type stubTranscriber struct {
	release chan struct{}
	text    string
	err     error
	gotLen  int
	calls   int
	mu      sync.Mutex
}

func (s *stubTranscriber) Transcribe(samples []float32, sampleRate int) (string, error) {
	if s.release != nil {
		<-s.release
	}
	s.mu.Lock()
	s.gotLen = len(samples)
	s.calls++
	s.mu.Unlock()
	return s.text, s.err
}

func (s *stubTranscriber) Close() error { return nil }

func TestSpawnPredictedSTTRunsInBackground(t *testing.T) {
	rel := make(chan struct{})
	tr := &stubTranscriber{release: rel, text: "speculative result"}
	samples := audio.MonoF32{1, 2, 3, 4, 5}
	r := SpawnPredictedSTT(tr, samples, 16000)

	if r.IsDone() {
		t.Fatal("runner should not be done while transcriber blocked")
	}
	close(rel)

	got := r.AwaitResult()
	if got == nil {
		t.Fatal("nil result")
	}
	if got.Text != "speculative result" {
		t.Fatalf("text=%q", got.Text)
	}
	if got.Err != "" {
		t.Fatalf("err=%q", got.Err)
	}
	if !r.IsDone() {
		t.Fatal("expected done after AwaitResult")
	}
	if tr.gotLen != len(samples) {
		t.Fatalf("transcriber got len %d want %d", tr.gotLen, len(samples))
	}
}

func TestSpawnPredictedSTTSurfacesError(t *testing.T) {
	rel := make(chan struct{})
	close(rel)
	tr := &stubTranscriber{release: rel, err: errors.New("whisper exploded")}
	r := SpawnPredictedSTT(tr, audio.MonoF32{0}, 16000)
	got := r.AwaitResult()
	if got.Err == "" {
		t.Fatal("expected error to be surfaced")
	}
	if got.Text != "" {
		t.Fatalf("text=%q want empty", got.Text)
	}
}

func TestSpawnPredictedSTTNilTranscriberCompletesImmediately(t *testing.T) {
	r := SpawnPredictedSTT(nil, audio.MonoF32{0}, 16000)

	done := make(chan *PredictedSTTResult, 1)
	go func() { done <- r.AwaitResult() }()
	select {
	case got := <-done:
		if got.Err == "" {
			t.Fatal("nil transcriber should yield an error result")
		}
	case <-time.After(time.Second):
		t.Fatal("AwaitResult deadlocked on nil transcriber")
	}
}

func TestSpawnPredictedSTTDefensiveCopy(t *testing.T) {

	rel := make(chan struct{})
	tr := &stubTranscriber{release: rel, text: "ok"}
	samples := audio.MonoF32{0.1, 0.2, 0.3, 0.4}
	r := SpawnPredictedSTT(tr, samples, 16000)

	for i := range samples {
		samples[i] = 99
	}
	close(rel)
	r.AwaitResult()
	if tr.gotLen != 4 {
		t.Fatalf("transcriber saw len %d, want 4 (defensive copy missed)", tr.gotLen)
	}
}

func TestPredictedSTTHandleAbortIsCancelled(t *testing.T) {
	h := &predictedSTTHandle{snapshotLen: 100}
	if h.isCancelled() {
		t.Fatal("fresh handle reports cancelled")
	}
	h.abort()
	if !h.isCancelled() {
		t.Fatal("abort should mark cancelled")
	}

	var nilH *predictedSTTHandle
	nilH.abort()
	if nilH.isCancelled() {
		t.Fatal("nil handle should not be cancelled")
	}
}
