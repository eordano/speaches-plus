package realtime

import (
	"context"
	"sync"
	"time"

	"github.com/eordano/speaches-plus-go/internal/conversation"
	"github.com/eordano/speaches-plus-go/internal/inspect"
)

type eagerRunner struct {
	mu sync.Mutex

	respID string
	itemID string
	epoch  uint64

	ctx    context.Context
	cancel context.CancelFunc
	deltas <-chan conversation.Delta

	buffered []string
	bufCap   int
	overflow bool

	partial string

	stt *predictedSTTHandle

	promoted   chan struct{}
	cancelled  chan struct{}
	finished   chan struct{}
	llmDone    bool
	transcript string
}

func (p *sessionPipeline) startEagerRunner(respID, itemID string, epoch uint64, partial string, samples []float32) *eagerRunner {
	llm := p.server.LLM
	if llm == nil || !llm.Configured() {
		return nil
	}
	llmTimeout := time.Duration(p.session.LLMTimeoutSec) * time.Second
	if llmTimeout <= 0 {
		llmTimeout = time.Duration(defaultLLMTimeoutSec) * time.Second
	}
	ctx, cancel := context.WithTimeout(context.Background(), llmTimeout)
	deltas, err := llm.StreamWithInstructions(ctx, p.session.Model, p.getInstructions(), partial)
	if err != nil {
		cancel()
		p.logger.Warn("eager LLM stream failed to open", "err", err)
		return nil
	}
	bufCap := p.session.PredictedTokenBufferCap
	if bufCap <= 0 {
		bufCap = defaultPredictedTokenBufferCap
	}
	r := &eagerRunner{
		respID:    respID,
		itemID:    itemID,
		epoch:     epoch,
		ctx:       ctx,
		cancel:    cancel,
		deltas:    deltas,
		bufCap:    bufCap,
		partial:   partial,
		promoted:  make(chan struct{}),
		cancelled: make(chan struct{}),
		finished:  make(chan struct{}),
	}

	if p.server.STT != nil && len(samples) > 0 {
		runner := SpawnPredictedSTT(p.server.STT, append([]float32(nil), samples...), whisperSampleRate)
		r.stt = &predictedSTTHandle{runner: runner, snapshotLen: len(samples)}
	}
	p.wg.Add(1)
	go func() {
		defer p.wg.Done()
		r.bufferLoop(p)
	}()
	return r
}

func (r *eagerRunner) bufferLoop(p *sessionPipeline) {
	defer close(r.finished)
	for {
		select {
		case <-r.cancelled:
			return
		case d, ok := <-r.deltas:
			if !ok {
				r.mu.Lock()
				r.llmDone = true
				r.mu.Unlock()
				return
			}
			if d.Err != nil {
				p.logger.Warn("eager llm stream error", "err", d.Err)
				return
			}
			if d.Content != "" {
				r.mu.Lock()
				r.buffered = append(r.buffered, d.Content)
				r.transcript += d.Content
				if r.bufCap > 0 && len(r.buffered) > r.bufCap {
					drop := len(r.buffered) - r.bufCap
					r.buffered = r.buffered[drop:]
					if !r.overflow {
						r.overflow = true
						r.mu.Unlock()
						p.logger.Warn("eou.predicted_overflow",
							"id", r.respID, "cap", r.bufCap)
						p.emitEOU("predicted_overflow", inspect.EOUFields{
							Extra: map[string]any{"id": r.respID, "cap": r.bufCap},
						})
						r.mu.Lock()
					}
				}
				r.mu.Unlock()
			}
			if d.Done {
				r.mu.Lock()
				r.llmDone = true
				r.mu.Unlock()
				return
			}

			select {
			case <-r.cancelled:
				return
			case <-r.promoted:
				return
			default:
			}
		case <-r.promoted:
			return
		}
	}
}

func (r *eagerRunner) abort() {
	if r == nil {
		return
	}
	select {
	case <-r.cancelled:
	default:
		close(r.cancelled)
	}
	r.cancel()
}

func (p *sessionPipeline) promotePredicted(text string, transcribeStart time.Time) bool {

	r := p.phase.currentEagerRunner()
	if r == nil {
		return false
	}
	respID, itemID := r.respID, r.itemID

	if transcriptsMateriallyDiffer(r.partial, text, defaultEagerTranscriptMismatchRatio) {
		r.mu.Lock()
		llmChars := 0
		for _, s := range r.buffered {
			llmChars += len(s)
		}
		r.mu.Unlock()
		p.logger.Warn("eager.predicted_rollback",
			"reason", "transcript_mismatch",
			"resp_id", respID,
			"partial_chars", len(r.partial),
			"final_chars", len(text),
			"llm_chars_thrown", llmChars,
		)
		p.emitEOU("predicted_rollback", inspect.EOUFields{
			Extra: map[string]any{
				"reason":           "transcript_mismatch",
				"id":               respID,
				"partial_chars":    len(r.partial),
				"final_chars":      len(text),
				"llm_chars_thrown": llmChars,
			},
		})
		r.abort()
		return false
	}

	id, _, _, ok := p.phase.onPredictedPromote(Epoch(r.epoch))
	if !ok || id == "" {
		r.abort()
		return false
	}
	if out := p.getOutboundTTS(); out != nil {
		out.ResetPlayedMs()
	}
	p.emitResponseCreated(respID)
	p.emitResponseOutputItemAdded(respID, itemID, 0)
	p.emitResponseContentPartAdded(respID, itemID, 0, 0, "audio")

	close(r.promoted)
	p.runEagerStreaming(r, transcribeStart, text)
	return true
}

func (p *sessionPipeline) runEagerStreaming(r *eagerRunner, transcribeStart time.Time, _ string) {
	respID := r.respID
	itemID := r.itemID
	epoch := r.epoch

	out := p.getOutboundTTS()
	voice := p.session.Voice
	if voice == "" {
		voice = defaultVoice
	}
	speed := p.session.Speed
	if speed == 0 {
		speed = defaultTTSSpeed
	}
	chunks := newSentenceChunker(sentenceChunkerMinChars)
	var firstAudio time.Time
	llmStart := time.Now()
	var ttsCount int

	flushChunk := p.makeFlushChunk(respID, itemID, epoch, out, voice, speed, &firstAudio, &ttsCount, transcribeStart, llmStart)

	r.mu.Lock()
	buffered := append([]string(nil), r.buffered...)
	llmDone := r.llmDone
	r.mu.Unlock()
	for _, c := range buffered {
		for _, chunk := range chunks.feed(c) {
			if !flushChunk(chunk) {
				return
			}
		}
	}

	if !llmDone {
		<-r.finished
		r.mu.Lock()
		llmDone = r.llmDone

		for _, c := range r.buffered[len(buffered):] {
			r.mu.Unlock()
			for _, chunk := range chunks.feed(c) {
				if !flushChunk(chunk) {
					return
				}
			}
			r.mu.Lock()
		}
		r.mu.Unlock()
	}
	if tail := chunks.flush(); tail != "" {
		if !flushChunk(tail) {
			return
		}
	}

	if !p.phase.onLLMComplete(Epoch(epoch)) {
		return
	}
	drainStatus := p.drainResponse(epoch, out)
	transcript, audioMs, ok := p.phase.onUpstreamComplete(Epoch(epoch))
	if !ok {
		return
	}
	p.logger.Info("eager llm done",
		"elapsed_ms", time.Since(llmStart).Milliseconds(),
		"reply_chars", len(transcript),
		"tts_chunks", ttsCount,
		"audio_played_ms", audioMs,
		"drain_status", drainStatus,
	)
	p.emitAudioTranscriptDone(respID, itemID, transcript)
	p.emitAudioDone(respID, itemID)
	p.emitResponseContentPartDone(respID, itemID, 0, 0, "audio", transcript)
	p.emitResponseOutputItemDone(respID, itemID, 0, transcript)
	itemStatus := "completed"
	if drainStatus == "incomplete" {
		itemStatus = "incomplete"
	}
	p.emitConversationItemCreated(conversationItemDetail{
		ID:     itemID,
		Object: "realtime.item",
		Type:   "message",
		Status: itemStatus,
		Role:   "assistant",
		Content: []responseContentPart{{
			Type:       "audio",
			Transcript: transcript,
		}},
	})
	if drainStatus == "incomplete" {
		p.emitResponseTerminal(respID, itemID, int64(audioMs), "incomplete", &statusDetails{Reason: "drain_cap"})
		return
	}
	p.emitResponseCompleted(respID, itemID, transcript, int64(audioMs))
}
