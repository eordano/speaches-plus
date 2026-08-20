package realtime

import (
	"fmt"
	"time"

	"github.com/pion/webrtc/v4"

	"github.com/eordano/speaches-plus-go/internal/audio"
	"github.com/eordano/speaches-plus-go/internal/diarization"
	"github.com/eordano/speaches-plus-go/internal/inspect"
)

const (
	diarSpeakerLabelFmt = "SPEAKER_%02d"
	diarKindFailed      = "failed"
	diarKindEmpty       = "empty"
	diarKindEmitted     = "emitted"
	diarReasonNoChannel = "no_data_channel"
	msPerSecondF        = 1000.0
)

func (p *sessionPipeline) runDiarization(itemID string, samples audio.MonoF32, audioEndMs int64) {
	if p.diarizer == nil || len(samples) == 0 {
		return
	}

	uttStartMs := utteranceStartMs(samples, audioEndMs)

	t0 := time.Now()
	p.diarMu.Lock()
	segs, err := p.diarizer.DiarizeUtterance([]float32(samples), uttStartMs)
	p.diarMu.Unlock()
	elapsedMs := int(time.Since(t0).Milliseconds())

	if err != nil {
		p.logger.Warn("diarization failed", "err", err, "item_id", itemID)
		p.emitDiarization(diarKindFailed, inspect.DiarizationFields{
			ItemID:     itemID,
			AudioEndMs: audioEndMs,
			ElapsedMs:  elapsedMs,
			Failed:     true,
			Reason:     err.Error(),
		})
		return
	}
	if len(segs) == 0 {
		p.emitDiarization(diarKindEmpty, inspect.DiarizationFields{
			ItemID:     itemID,
			AudioEndMs: audioEndMs,
			ElapsedMs:  elapsedMs,
		})
		return
	}

	speakers := uniqueSpeakers(segs)

	ch := p.openDataChannel()
	if ch == nil {
		p.emitDiarization(diarKindEmitted, diarFields(itemID, audioEndMs, len(segs), speakers, elapsedMs, diarReasonNoChannel))
		return
	}

	ev := conversationItemDiarizationEvent{
		EventID:    newEventID(),
		Type:       SETConversationItemDiarization,
		ItemID:     itemID,
		AudioEndMs: audioEndMs,
		Segments:   buildSegmentEvents(segs),
	}
	p.safeSend(ch, ev, ev.EventID)

	p.emitDiarization(diarKindEmitted, diarFields(itemID, audioEndMs, len(segs), speakers, elapsedMs, ""))
}

func (p *sessionPipeline) openDataChannel() *webrtc.DataChannel {
	if !p.waitChannel() {
		return nil
	}
	return p.getChannel()
}

func utteranceStartMs(samples audio.MonoF32, audioEndMs int64) uint64 {
	dur := uint64(len(samples)) * 1000 / uint64(whisperSampleRate)
	if int64(dur) > audioEndMs {
		return 0
	}
	return uint64(audioEndMs) - dur
}

func uniqueSpeakers(segs []diarization.Segment) int {
	seen := make(map[diarization.ClusterID]struct{}, len(segs))
	for _, s := range segs {
		seen[s.Speaker] = struct{}{}
	}
	return len(seen)
}

func buildSegmentEvents(segs []diarization.Segment) []diarizationSegment {
	out := make([]diarizationSegment, 0, len(segs))
	for _, s := range segs {
		out = append(out, diarizationSegment{
			Speaker:    fmt.Sprintf(diarSpeakerLabelFmt, s.Speaker),
			Start:      float64(s.TStartMs) / msPerSecondF,
			End:        float64(s.TEndMs) / msPerSecondF,
			Confidence: s.Confidence,
		})
	}
	return out
}

func diarFields(itemID string, audioEndMs int64, segs, speakers, elapsedMs int, reason string) inspect.DiarizationFields {
	return inspect.DiarizationFields{
		ItemID:      itemID,
		AudioEndMs:  audioEndMs,
		NumSegments: segs,
		NumSpeakers: speakers,
		ElapsedMs:   elapsedMs,
		Reason:      reason,
	}
}
