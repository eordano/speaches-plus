package realtime

import (
	"strings"
	"sync"
	"sync/atomic"
	"time"
	"unicode"

	"github.com/eordano/speaches-plus-go/internal/audio"
	"github.com/eordano/speaches-plus-go/internal/stt"
)

type PredictedTokenBuffer struct {
	mu      sync.Mutex
	cap     int
	inner   []string
	dropped uint32
}

func NewPredictedTokenBuffer(cap int) *PredictedTokenBuffer {
	if cap < 1 {
		cap = 1
	}
	return &PredictedTokenBuffer{
		cap:   cap,
		inner: make([]string, 0, cap),
	}
}

func (b *PredictedTokenBuffer) Push(tok string) {
	b.mu.Lock()
	defer b.mu.Unlock()
	if len(b.inner) == b.cap {
		b.dropped++
		copy(b.inner, b.inner[1:])
		b.inner[len(b.inner)-1] = tok
		return
	}
	b.inner = append(b.inner, tok)
}

func (b *PredictedTokenBuffer) Cap() int {
	b.mu.Lock()
	defer b.mu.Unlock()
	return b.cap
}

func (b *PredictedTokenBuffer) Len() int {
	b.mu.Lock()
	defer b.mu.Unlock()
	return len(b.inner)
}

func (b *PredictedTokenBuffer) IsEmpty() bool {
	return b.Len() == 0
}

func (b *PredictedTokenBuffer) DroppedCount() uint32 {
	b.mu.Lock()
	defer b.mu.Unlock()
	return b.dropped
}

func (b *PredictedTokenBuffer) Drain() []string {
	b.mu.Lock()
	defer b.mu.Unlock()
	out := make([]string, len(b.inner))
	copy(out, b.inner)
	b.inner = b.inner[:0]
	return out
}

func transcriptsMateriallyDiffer(predicted, finalized string, ratio float32) bool {
	p := strings.ToLower(strings.TrimSpace(predicted))
	f := strings.ToLower(strings.TrimSpace(finalized))
	if p == "" || f == "" {
		return p != f
	}
	if p == f {
		return false
	}
	pset := charSet(p)
	fset := charSet(f)
	intersect := 0
	for c := range pset {
		if _, ok := fset[c]; ok {
			intersect++
		}
	}
	union := len(pset) + len(fset) - intersect
	if union < 1 {
		union = 1
	}
	jaccard := float32(intersect) / float32(union)
	threshold := 1.0 - ratio
	if threshold < 0 {
		threshold = 0
	} else if threshold > 1 {
		threshold = 1
	}
	return jaccard < threshold
}

func charSet(s string) map[rune]struct{} {
	m := make(map[rune]struct{}, len(s))
	for _, r := range s {
		if unicode.IsSpace(r) {
			continue
		}
		m[r] = struct{}{}
	}
	return m
}

type PredictedSTTResult struct {
	Text string
	Err  string
}

type PredictedSTTRunner struct {
	mu     sync.Mutex
	cond   *sync.Cond
	result *PredictedSTTResult
	done   bool

	startedAt time.Time
}

func SpawnPredictedSTT(transcriber stt.Transcriber, samples audio.MonoF32, sampleRate int) *PredictedSTTRunner {
	r := &PredictedSTTRunner{startedAt: time.Now()}
	r.cond = sync.NewCond(&r.mu)
	if transcriber == nil {
		r.store(&PredictedSTTResult{Err: "predicted stt: nil transcriber"})
		return r
	}

	snap := append(audio.MonoF32(nil), samples...)
	go func() {
		text, err := transcriber.Transcribe(snap, sampleRate)
		out := &PredictedSTTResult{Text: text}
		if err != nil {
			out.Err = err.Error()
		}
		r.store(out)
	}()
	return r
}

func (r *PredictedSTTRunner) store(res *PredictedSTTResult) {
	r.mu.Lock()
	r.result = res
	r.done = true
	r.cond.Broadcast()
	r.mu.Unlock()
}

func (r *PredictedSTTRunner) AwaitResult() *PredictedSTTResult {
	r.mu.Lock()
	defer r.mu.Unlock()
	for !r.done {
		r.cond.Wait()
	}
	return r.result
}

func (r *PredictedSTTRunner) IsDone() bool {
	r.mu.Lock()
	defer r.mu.Unlock()
	return r.done
}

func (r *PredictedSTTRunner) Elapsed() time.Duration {
	return time.Since(r.startedAt)
}

type predictedSTTHandle struct {
	runner      *PredictedSTTRunner
	snapshotLen int
	cancelled   atomic.Bool
}

func (h *predictedSTTHandle) abort() {
	if h == nil {
		return
	}
	h.cancelled.Store(true)
}

func (h *predictedSTTHandle) isCancelled() bool {
	if h == nil {
		return false
	}
	return h.cancelled.Load()
}

func (p *sessionPipeline) consumeSpeculativeSTT(currentLen int) *PredictedSTTRunner {
	r := p.phase.currentEagerRunner()
	if r == nil || r.stt == nil || r.stt.runner == nil {
		return nil
	}
	if r.stt.isCancelled() {
		return nil
	}
	if r.stt.snapshotLen != currentLen {
		return nil
	}
	runner := r.stt.runner
	r.stt.abort()
	return runner
}
