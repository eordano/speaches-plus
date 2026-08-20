package realtime

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"errors"
	"time"

	"github.com/pion/webrtc/v4"
	"go.opentelemetry.io/otel/attribute"

	"github.com/eordano/speaches-plus-go/internal/audio"
	"github.com/eordano/speaches-plus-go/internal/stt"
)

func base64Encode(b []byte) string { return base64.StdEncoding.EncodeToString(b) }

func transcribeWithStats(t stt.Transcriber, samples audio.MonoF32, sampleRate int) (stt.Result, error) {
	if full, ok := t.(stt.FullTranscriber); ok {
		return full.TranscribeFull(samples, sampleRate)
	}
	text, err := t.Transcribe(samples, sampleRate)
	return stt.Result{Text: text}, err
}

func (p *sessionPipeline) safeSend(ch *webrtc.DataChannel, event any, id string) {
	err := sendFragmentedWith(ch, event, id,
		p.session.DataChannelFragmentMax, p.session.OutboundBufferLimit)
	if err != nil {
		if errors.Is(err, errClientTooSlow) {
			p.logger.Warn("client_too_slow: outbound buffer exceeded; terminating session")
			p.emitErrorCode("client_too_slow", "outbound buffer exceeded")
			p.phase.terminateSession()
			go p.close()
			return
		}
		p.logger.Error("send failed", "err", err, "id", id)
	}
}

func (p *sessionPipeline) dispatch(event any, eventID string) bool {
	if !p.waitChannel() {
		return false
	}
	p.chMu.Lock()
	ws := p.wsConn
	ch := p.channel
	p.chMu.Unlock()
	if ws != nil {
		p.sendWS(event, eventID)
		return true
	}
	if ch != nil {
		p.safeSend(ch, event, eventID)
		return true
	}
	return false
}

func (p *sessionPipeline) runTranscription(samples audio.MonoF32, itemID string, allowResponse bool) {
	p.wg.Add(1)
	go func() {
		defer p.wg.Done()
		t0 := time.Now()
		sttCtx, sttSpan := startSpan(p.traceContext(), "stt.transcribe",
			attribute.Int("samples", len(samples)),
			attribute.String("item_id", itemID),
		)

		var (
			text string
			err  error
		)
		var stats stt.Result
		if speculative := p.consumeSpeculativeSTT(len(samples)); speculative != nil {
			res := speculative.AwaitResult()
			if res.Err == "" {
				text = res.Text
				p.logger.Debug("stt.predicted_hit",
					"elapsed_ms", time.Since(t0).Milliseconds(),
					"speculative_elapsed_ms", speculative.Elapsed().Milliseconds(),
				)
			} else {
				p.logger.Warn("stt.predicted_failed; falling back to fresh transcribe",
					"err", res.Err)
				stats, err = transcribeWithStats(p.server.STT, samples, whisperSampleRate)
				text = stats.Text
			}
		} else {
			stats, err = transcribeWithStats(p.server.STT, samples, whisperSampleRate)
			text = stats.Text
		}
		sttSpan.End()
		_ = sttCtx
		if err != nil {
			p.logger.Error("transcription failed", "err", err)
			if itemID != "" {
				p.phase.onTranscriptionFailed(ItemID(itemID))
				p.emitTranscriptionFailed(itemID, err.Error())
			}
			return
		}

		if rejection := stt.EvaluateNoiseGate(
			stats.NoSpeechProb,
			stats.AvgLogprob,
			len(samples)*1000/whisperSampleRate,
			stt.GateThresholds{
				NoSpeechProb: p.session.NoSpeechProbThreshold,
				AvgLogprob:   p.session.AvgLogprobThreshold,
			},
		); rejection != stt.NoiseAccept {
			p.logger.Info("noise gate rejected transcript",
				"reason", rejection.String(),
				"item", itemID,
				"audio_ms", len(samples)*1000/whisperSampleRate,
			)
			if itemID != "" {
				p.phase.onTranscriptionComplete(ItemID(itemID), "", false)
			}
			return
		}
		p.logger.Info("whisper done",
			"elapsed_ms", time.Since(t0).Milliseconds(),
			"audio_ms", len(samples)*1000/whisperSampleRate,
		)
		autoResp := p.session.Conversation && allowResponse
		if itemID != "" {
			p.phase.onTranscriptionComplete(ItemID(itemID), text, autoResp)
		}
		p.emitTranscription(itemID, text)

		if autoResp {
			if p.promotePredicted(text, t0) {
				p.markTurnDone()
				return
			}
			p.startResponse(newItemID(), text, t0.UnixMilli(), nil, nil)
		}
		p.markTurnDone()
	}()
}

func (p *sessionPipeline) startResponse(itemID string, transcript string, t0Ms int64, instr *string, modalities []string) {
	llm := p.server.LLM
	if llm == nil || !llm.Configured() {

		p.logger.Warn("LLM not configured; emitting response.done(failed)")
		respID := newRespID()
		p.emitResponseCreated(respID)
		p.emitResponseFailed(respID, itemID, 0, "llm_error", "model_load_failed", "LLM not configured")
		return
	}

	respID := newRespID()
	epoch, err := p.phase.onResponseCreate(ResponseID(respID), ItemID(itemID))
	if err != nil {
		p.logger.Warn("response.create rejected", "err", err)
		return
	}
	transcribeStart := time.UnixMilli(t0Ms)
	_ = modalities
	instructions := p.getInstructions()
	if instr != nil {
		instructions = *instr
	}
	turnCtx, turnSpan := startSpan(p.traceContext(), "realtime.turn",
		attribute.String("response_id", respID),
		attribute.String("item_id", itemID),
		attribute.Int64("epoch", int64(epoch)),
	)
	defer turnSpan.End()
	_ = turnCtx
	if out := p.getOutboundTTS(); out != nil {
		out.ResetPlayedMs()
	}
	p.emitResponseCreated(respID)
	p.emitResponseOutputItemAdded(respID, itemID, 0)
	p.emitResponseContentPartAdded(respID, itemID, 0, 0, "audio")

	llmTimeout := time.Duration(p.session.LLMTimeoutSec) * time.Second
	if llmTimeout <= 0 {
		llmTimeout = time.Duration(defaultLLMTimeoutSec) * time.Second
	}
	ctx, cancel := context.WithTimeout(turnCtx, llmTimeout)
	defer cancel()
	llmCtx, llmSpan := startSpan(ctx, "llm.stream",
		attribute.String("model", p.session.Model),
		attribute.Int("user_text_chars", len(transcript)),
	)
	defer llmSpan.End()
	_ = llmCtx

	deltas, err := llm.StreamWithInstructions(ctx, p.session.Model, instructions, transcript)
	if err != nil {
		p.logger.Error("llm stream open failed", "err", err)

		var audioMs Millis
		if _, am, ok := p.phase.onUpstreamComplete(epoch); ok {
			audioMs = am
			p.emitResponseFailed(respID, itemID, int64(audioMs), "llm_error", "internal_state_error", err.Error())
		}
		return
	}

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

	flushChunk := p.makeFlushChunk(respID, itemID, uint64(epoch), out, voice, speed,
		&firstAudio, &ttsCount, transcribeStart, llmStart)

	for d := range deltas {
		if d.Err != nil {
			p.logger.Error("llm stream failed", "err", d.Err)

			if _, am, ok := p.phase.onUpstreamComplete(epoch); ok {
				p.emitResponseFailed(respID, itemID, int64(am), "llm_error", "internal_state_error", d.Err.Error())
			}
			return
		}
		if d.Content != "" {

			p.emitAudioTranscriptDelta(respID, itemID, d.Content)
			if !p.phase.onUpstreamDelta(Epoch(epoch), d.Content, 0) {
				return
			}
			alive := true
			for _, chunk := range chunks.feed(d.Content) {
				if !flushChunk(chunk) {
					alive = false
					break
				}
			}
			if !alive {
				return
			}
		}
		if d.Done {
			break
		}
	}
	if tail := chunks.flush(); tail != "" {
		if !flushChunk(tail) {
			return
		}
	}

	if !p.phase.onLLMComplete(epoch) {
		return
	}

	drainStatus := p.drainResponse(uint64(epoch), out)

	transcript, audioMs, ok := p.phase.onUpstreamComplete(epoch)
	if !ok {
		return
	}
	p.logger.Info("llm done",
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
		details := &statusDetails{Reason: "drain_cap"}
		p.emitResponseTerminal(respID, itemID, int64(audioMs), "incomplete", details)
		return
	}
	p.emitResponseCompleted(respID, itemID, transcript, int64(audioMs))
}

func (p *sessionPipeline) drainResponse(epoch uint64, out outboundWriter) string {
	if out == nil {
		return "completed"
	}
	plannedMs := p.plannedMsForEpoch(epoch)
	if plannedMs <= 0 {
		return "completed"
	}
	played := out.PlayedMs()
	if played >= plannedMs {
		p.phase.updatePlayedMs(Epoch(epoch), Millis(played))
		return "completed"
	}
	minCap := int64(p.session.DrainCapFloorMs)
	if minCap <= 0 {
		minCap = defaultDrainCapFloorMs
	}
	maxCap := int64(p.session.DrainCapCeilingMs)
	if maxCap <= 0 {
		maxCap = defaultDrainCapCeilingMs
	}
	cap := 2 * plannedMs
	if cap < minCap {
		cap = minCap
	}
	if cap > maxCap {
		cap = maxCap
	}
	deadline := time.Now().Add(time.Duration(cap) * time.Millisecond)
	tick := time.NewTicker(drainPollIntervalMs * time.Millisecond)
	defer tick.Stop()
	for {
		select {
		case <-p.closed:
			p.phase.updatePlayedMs(Epoch(epoch), Millis(out.PlayedMs()))
			return "incomplete"
		case <-tick.C:
			played = out.PlayedMs()
			if played >= plannedMs {
				p.phase.updatePlayedMs(Epoch(epoch), Millis(played))
				return "completed"
			}
			if k, e, _ := p.phase.responseEpoch(); k == respKindNone || k == respKindFinalized || uint64(e) != epoch {
				return "completed"
			}
			if time.Now().After(deadline) {
				p.logger.Warn("drain_cap expired",
					"played_ms", played, "planned_ms", plannedMs, "cap_ms", cap)
				p.phase.updatePlayedMs(Epoch(epoch), Millis(played))
				return "incomplete"
			}
		}
	}
}

func (p *sessionPipeline) plannedMsForEpoch(epoch uint64) int64 {
	_, _, _, resp := p.phase.snapshotFull()
	if uint64(respEpochOf(resp)) != epoch {
		return 0
	}
	switch r := resp.(type) {
	case RespStreaming:
		return int64(r.PlannedMs)
	case RespDrain:
		return int64(r.PlannedMs)
	}
	return 0
}

func (p *sessionPipeline) waitChannel() bool {
	select {
	case <-p.chReady:
		return true
	case <-time.After(2 * time.Second):
		return false
	}
}

func (p *sessionPipeline) emitResponseCreated(id string) {
	ev := responseCreatedEvent{
		EventID: newEventID(),
		Type:    SETResponseCreated,
		Response: responseCreatedResponse{
			ID:     id,
			Object: "realtime.response",
			Status: "in_progress",
		},
	}
	p.dispatch(ev, ev.EventID)
}

func (p *sessionPipeline) emitResponseOutputItemAdded(respID, itemID string, idx int) {
	ev := responseOutputItemAddedEvent{
		EventID:     newEventID(),
		Type:        SETResponseOutputItemAdded,
		ResponseID:  respID,
		OutputIndex: idx,
		Item: responseOutputItem{
			ID:     itemID,
			Object: "realtime.item",
			Type:   "message",
			Role:   "assistant",
			Status: "in_progress",
			Content: []responseContentPart{{
				Type: "audio",
			}},
		},
	}
	p.dispatch(ev, ev.EventID)
}

func (p *sessionPipeline) emitResponseOutputItemDone(respID, itemID string, idx int, transcript string) {
	ev := responseOutputItemDoneEvent{
		EventID:     newEventID(),
		Type:        SETResponseOutputItemDone,
		ResponseID:  respID,
		OutputIndex: idx,
		Item: responseOutputItem{
			ID:     itemID,
			Object: "realtime.item",
			Type:   "message",
			Role:   "assistant",
			Status: "completed",
			Content: []responseContentPart{{
				Type:       "audio",
				Transcript: transcript,
			}},
		},
	}
	p.dispatch(ev, ev.EventID)
}

func (p *sessionPipeline) emitResponseContentPartAdded(respID, itemID string, outIdx, contIdx int, partType string) {
	ev := responseContentPartAddedEvent{
		EventID:      newEventID(),
		Type:         SETResponseContentPartAdded,
		ResponseID:   respID,
		ItemID:       itemID,
		OutputIndex:  outIdx,
		ContentIndex: contIdx,
		Part:         responseContentPart{Type: partType},
	}
	p.dispatch(ev, ev.EventID)
}

func (p *sessionPipeline) emitResponseContentPartDone(respID, itemID string, outIdx, contIdx int, partType, transcript string) {
	ev := responseContentPartDoneEvent{
		EventID:      newEventID(),
		Type:         SETResponseContentPartDone,
		ResponseID:   respID,
		ItemID:       itemID,
		OutputIndex:  outIdx,
		ContentIndex: contIdx,
		Part:         responseContentPart{Type: partType, Transcript: transcript},
	}
	p.dispatch(ev, ev.EventID)
}

func (p *sessionPipeline) emitAudioTranscriptDone(respID, itemID, transcript string) {
	ev := responseAudioTranscriptDoneEvent{
		EventID:    newEventID(),
		Type:       SETResponseOutputAudioTranscriptDone,
		ResponseID: respID,
		ItemID:     itemID,
		Transcript: transcript,
	}
	p.dispatch(ev, ev.EventID)
}

func (p *sessionPipeline) emitAudioDone(respID, itemID string) {
	ev := responseAudioDoneEvent{
		EventID:    newEventID(),
		Type:       SETResponseOutputAudioDone,
		ResponseID: respID,
		ItemID:     itemID,
	}
	p.dispatch(ev, ev.EventID)
}

func (p *sessionPipeline) emitInputBufferSpeechStarted(itemID string, startMs int64) {
	ev := inputAudioBufferSpeechStartedEvent{
		EventID:      newEventID(),
		Type:         SETInputBufferSpeechStarted,
		ItemID:       itemID,
		AudioStartMs: startMs,
	}
	p.dispatch(ev, ev.EventID)
}

func (p *sessionPipeline) emitInputBufferSpeechStopped(itemID string, endMs int64) {
	ev := inputAudioBufferSpeechStoppedEvent{
		EventID:    newEventID(),
		Type:       SETInputBufferSpeechStopped,
		ItemID:     itemID,
		AudioEndMs: endMs,
	}
	p.dispatch(ev, ev.EventID)
}

func (p *sessionPipeline) emitInputBufferCommitted(itemID string) {
	ev := inputAudioBufferCommittedEvent{
		EventID: newEventID(),
		Type:    SETInputBufferCommitted,
		ItemID:  itemID,
	}
	p.dispatch(ev, ev.EventID)
}

func (p *sessionPipeline) emitInputBufferPartialTranscription(itemID, transcript string) {
	ev := inputAudioBufferPartialTranscriptionEvent{
		EventID:    newEventID(),
		Type:       SETInputBufferPartialTranscription,
		ItemID:     itemID,
		Transcript: transcript,
	}
	p.dispatch(ev, ev.EventID)
}

func (p *sessionPipeline) emitInputBufferCleared() {
	ev := inputAudioBufferClearedEvent{
		EventID: newEventID(),
		Type:    SETInputBufferCleared,
	}
	p.dispatch(ev, ev.EventID)
}

func (p *sessionPipeline) emitConversationItemCreated(item conversationItemDetail) {
	ev := conversationItemCreatedEvent{
		EventID: newEventID(),
		Type:    SETConversationItemAdded,
		Item:    item,
	}
	p.dispatch(ev, ev.EventID)
}

func (p *sessionPipeline) emitConversationItemTruncated(itemID string, audioEndMs int64) {
	ev := conversationItemTruncatedEvent{
		EventID:    newEventID(),
		Type:       SETConversationItemTruncated,
		ItemID:     itemID,
		AudioEndMs: audioEndMs,
	}
	p.dispatch(ev, ev.EventID)
}

func (p *sessionPipeline) emitConversationItemAssistantTruncated(itemID string, audioEndMs int64, transcript string) {
	ev := conversationItemAssistantTruncatedEvent{
		EventID:    newEventID(),
		Type:       SETConversationItemAssistantTruncated,
		ItemID:     itemID,
		AudioEndMs: audioEndMs,
		Transcript: transcript,
	}
	p.dispatch(ev, ev.EventID)
}

func (p *sessionPipeline) emitConversationItemDeleted(itemID string) {
	ev := conversationItemDeletedEvent{
		EventID: newEventID(),
		Type:    SETConversationItemDeleted,
		ItemID:  itemID,
	}
	p.dispatch(ev, ev.EventID)
}

func (p *sessionPipeline) emitResponseAudioDelta(respID, itemID string, samples audio.MonoF32, sampleRate int) {
	if len(samples) == 0 {
		return
	}
	pcm := make([]byte, len(samples)*2)
	for i, s := range samples {
		v := int32(s * 32767)
		if v > 32767 {
			v = 32767
		} else if v < -32768 {
			v = -32768
		}
		pcm[2*i] = byte(v)
		pcm[2*i+1] = byte(v >> 8)
	}
	encoded := base64Encode(pcm)
	ev := responseAudioDeltaEvent{
		EventID:    newEventID(),
		Type:       SETResponseOutputAudioDelta,
		ResponseID: respID,
		ItemID:     itemID,
		Delta:      encoded,
	}
	p.dispatch(ev, ev.EventID)
}

func (p *sessionPipeline) emitAudioTranscriptDelta(respID, itemID, delta string) {
	ev := responseAudioTranscriptDeltaEvent{
		EventID:    newEventID(),
		Type:       SETResponseOutputAudioTranscriptDelta,
		ResponseID: respID,
		ItemID:     itemID,
		Delta:      delta,
	}
	p.dispatch(ev, ev.EventID)
}

func (p *sessionPipeline) emitSessionDone(reason string) {
	ev := sessionDoneEvent{
		EventID: newEventID(),
		Type:    SETSessionDone,
		Reason:  reason,
	}
	p.dispatch(ev, ev.EventID)
}

func (p *sessionPipeline) emitSessionUpdated(sess session) {
	ev := sessionUpdatedEvent{
		EventID: newEventID(),
		Type:    SETSessionUpdated,
		Session: sess,
	}
	p.dispatch(ev, ev.EventID)
}

func (p *sessionPipeline) emitError(code, message string) {
	p.emitErrorTyped("invalid_request_error", code, message, "")
}

func (p *sessionPipeline) emitErrorCode(code, message string) {
	p.emitErrorTyped(errorTypeFor(code), code, message, "")
}

func (p *sessionPipeline) emitErrorTyped(typ, code, message, param string) {
	ev := errorEvent{
		EventID: newEventID(),
		Type:    SETError,
		Error: errorPayload{
			Type:    typ,
			Code:    code,
			Message: message,
			Param:   param,
		},
	}
	p.dispatch(ev, ev.EventID)
}

func errorTypeFor(code string) string {
	switch code {
	case "invalid_request_error",
		"unknown_event_type",
		"session_not_active",
		"session_update_invalid",
		"response_already_active",
		"response_cancel_not_active",
		"input_audio_buffer_commit_empty",
		"client_too_slow":
		return "invalid_request_error"
	case "internal_state_error",
		"vad_failed",
		"stt_failed",
		"model_load_failed":
		return "server_error"
	}
	return "invalid_request_error"
}

func (p *sessionPipeline) emitResponseCancelled(id, itemID string, audioEndMs int64) {
	p.emitResponseTerminal(id, itemID, audioEndMs, "cancelled", nil)
}

func (p *sessionPipeline) emitResponseFailed(respID, itemID string, audioEndMs int64, reason, code, message string) {
	details := &statusDetails{Reason: reason}
	if code != "" || message != "" {
		details.Error = &errorPayload{Type: errorTypeFor(code), Code: code, Message: message}
	}
	p.emitResponseTerminal(respID, itemID, audioEndMs, "failed", details)
}

func (p *sessionPipeline) emitResponseCompleted(respID, itemID, transcript string, audioEndMs int64) {
	resp := responseDoneResponse{
		ID:         respID,
		Object:     "realtime.response",
		Status:     "completed",
		AudioEndMs: audioEndMs,
	}
	if transcript != "" {
		resp.Output = []responseOutputItem{{
			ID:     itemID,
			Type:   "message",
			Role:   "assistant",
			Status: "completed",
			Content: []responseContentPart{{
				Type:       "audio",
				Transcript: transcript,
			}},
		}}
	}
	ev := responseDoneEvent{
		EventID:  newEventID(),
		Type:     SETResponseDone,
		Response: resp,
	}
	p.dispatch(ev, ev.EventID)
}

func (p *sessionPipeline) emitResponseTerminal(id, itemID string, audioEndMs int64, status string, details *statusDetails) {
	resp := responseDoneResponse{
		ID:            id,
		Object:        "realtime.response",
		Status:        status,
		StatusDetails: details,
		AudioEndMs:    audioEndMs,
	}
	if itemID != "" {
		itemStatus := "incomplete"
		if status == "completed" {
			itemStatus = "completed"
		}
		resp.Output = []responseOutputItem{{
			ID:      itemID,
			Type:    "message",
			Role:    "assistant",
			Status:  itemStatus,
			Content: []responseContentPart{{Type: "audio"}},
		}}
	}
	ev := responseDoneEvent{
		EventID:  newEventID(),
		Type:     SETResponseDone,
		Response: resp,
	}
	p.dispatch(ev, ev.EventID)
}

func (p *sessionPipeline) emitTranscription(itemID, text string) {
	if itemID == "" {
		itemID = newItemID()
	}
	ev := transcriptionCompletedEvent{
		EventID:    newEventID(),
		Type:       SETInputAudioTranscriptionCompleted,
		ItemID:     itemID,
		ContentIdx: 0,
		Transcript: text,
	}
	if !p.dispatch(ev, ev.EventID) {
		p.logger.Warn("transport not ready; dropping transcription", "text", text)
		return
	}
	p.logger.Info("transcription emitted", "transcript", text)
}

func (p *sessionPipeline) emitTranscriptionFailed(itemID, message string) {
	ev := transcriptionFailedEvent{
		EventID:    newEventID(),
		Type:       SETInputAudioTranscriptionFailed,
		ItemID:     itemID,
		ContentIdx: 0,
		Error: errorPayload{
			Type:    "transcription_error",
			Code:    "stt_failed",
			Message: message,
		},
	}
	p.dispatch(ev, ev.EventID)
}

func (p *sessionPipeline) emitInputAudioTranscriptionDelta(itemID, delta string) {
	ev := transcriptionDeltaEvent{
		EventID:    newEventID(),
		Type:       SETInputAudioTranscriptionDelta,
		ItemID:     itemID,
		ContentIdx: 0,
		Delta:      delta,
	}
	p.dispatch(ev, ev.EventID)
}

func (p *sessionPipeline) emitConversationItemDone(item conversationItemDetail) {
	ev := conversationItemDoneEvent{
		EventID: newEventID(),
		Type:    SETConversationItemDone,
		Item:    item,
	}
	p.dispatch(ev, ev.EventID)
}

func (p *sessionPipeline) emitConversationItemRetrieved(item conversationItemDetail) {
	ev := conversationItemRetrievedEvent{
		EventID: newEventID(),
		Type:    SETConversationItemRetrieved,
		Item:    item,
	}
	p.dispatch(ev, ev.EventID)
}

func (p *sessionPipeline) emitResponseOutputTextDelta(respID, itemID string, outIdx, contIdx int, delta string) {
	ev := responseOutputTextDeltaEvent{
		EventID:      newEventID(),
		Type:         SETResponseOutputTextDelta,
		ResponseID:   respID,
		ItemID:       itemID,
		OutputIndex:  outIdx,
		ContentIndex: contIdx,
		Delta:        delta,
	}
	p.dispatch(ev, ev.EventID)
}

func (p *sessionPipeline) emitResponseOutputTextDone(respID, itemID string, outIdx, contIdx int, text string) {
	ev := responseOutputTextDoneEvent{
		EventID:      newEventID(),
		Type:         SETResponseOutputTextDone,
		ResponseID:   respID,
		ItemID:       itemID,
		OutputIndex:  outIdx,
		ContentIndex: contIdx,
		Text:         text,
	}
	p.dispatch(ev, ev.EventID)
}

func (p *sessionPipeline) emitResponseFunctionCallArgumentsDelta(respID, itemID string, outIdx int, callID, delta string) {
	ev := responseFunctionCallArgumentsDeltaEvent{
		EventID:     newEventID(),
		Type:        SETResponseFunctionCallArgumentsDelta,
		ResponseID:  respID,
		ItemID:      itemID,
		OutputIndex: outIdx,
		CallID:      callID,
		Delta:       delta,
	}
	p.dispatch(ev, ev.EventID)
}

func (p *sessionPipeline) emitResponseFunctionCallArgumentsDone(respID, itemID string, outIdx int, callID, arguments string) {
	ev := responseFunctionCallArgumentsDoneEvent{
		EventID:     newEventID(),
		Type:        SETResponseFunctionCallArgumentsDone,
		ResponseID:  respID,
		ItemID:      itemID,
		OutputIndex: outIdx,
		CallID:      callID,
		Arguments:   arguments,
	}
	p.dispatch(ev, ev.EventID)
}

func (p *sessionPipeline) emitResponseToolProgress(respID, itemID string, outIdx int, progress json.RawMessage) {
	ev := responseToolProgressEvent{
		EventID:     newEventID(),
		Type:        SETResponseToolProgress,
		ResponseID:  respID,
		ItemID:      itemID,
		OutputIndex: outIdx,
		Progress:    progress,
	}
	p.dispatch(ev, ev.EventID)
}

func (p *sessionPipeline) emitResponseCancelledStandalone(respID string) {
	ev := responseCancelledEvent{
		EventID:    newEventID(),
		Type:       SETResponseCancelled,
		ResponseID: respID,
	}
	p.dispatch(ev, ev.EventID)
}

func (p *sessionPipeline) emitOutputAudioBufferCleared() {
	ev := outputAudioBufferClearedEvent{
		EventID: newEventID(),
		Type:    SETOutputAudioBufferCleared,
	}
	p.dispatch(ev, ev.EventID)
}

func (p *sessionPipeline) emitOutputAudioBufferStarted(respID string) {
	ev := outputAudioBufferStartedEvent{
		EventID:    newEventID(),
		Type:       SETOutputAudioBufferStarted,
		ResponseID: respID,
	}
	p.dispatch(ev, ev.EventID)
}

func (p *sessionPipeline) emitOutputAudioBufferStopped(respID string) {
	ev := outputAudioBufferStoppedEvent{
		EventID:    newEventID(),
		Type:       SETOutputAudioBufferStopped,
		ResponseID: respID,
	}
	p.dispatch(ev, ev.EventID)
}

func (p *sessionPipeline) emitRateLimitsUpdated(limits []rateLimitInfo) {
	ev := rateLimitsUpdatedEvent{
		EventID:    newEventID(),
		Type:       SETRateLimitsUpdated,
		RateLimits: limits,
	}
	p.dispatch(ev, ev.EventID)
}

func (p *sessionPipeline) makeFlushChunk(
	respID, itemID string,
	epoch uint64,
	out outboundWriter,
	voice string,
	speed float32,
	firstAudio *time.Time,
	ttsCount *int,
	transcribeStart, llmStart time.Time,
) func(string) bool {
	return func(chunk string) bool {
		if chunk == "" {
			return true
		}

		if p.server.TTS == nil || out == nil {
			return true
		}
		t1 := time.Now()
		_, ttsSpan := startSpan(p.traceContext(), "tts.synthesize",
			attribute.Int("chunk_chars", len(chunk)),
			attribute.String("voice", voice),
		)
		aud, terr := p.server.TTS.Synthesize(chunk, voice, "en-us", speed)
		ttsSpan.End()
		if terr != nil {
			p.logger.Error("tts failed", "err", terr, "chunk", chunk)
			return true
		}
		if k, e, _ := p.phase.responseEpoch(); k == respKindNone || k == respKindFinalized || uint64(e) != epoch {
			p.logger.Debug("response cancelled while TTS in flight; dropping audio", "chunk", chunk)
			return false
		}
		if aud.SampleRate > 0 {
			plannedMs := int64(len(aud.Samples)) * 1000 / int64(aud.SampleRate)
			p.phase.onUpstreamDelta(Epoch(epoch), "", DurationMs(plannedMs))
		}
		p.emitResponseAudioDelta(respID, itemID, aud.Samples, aud.SampleRate)
		(*ttsCount)++
		if firstAudio.IsZero() {
			*firstAudio = time.Now()
			p.logger.Info("first-audio-byte ready",
				"transcribe_to_first_audio_ms", time.Since(transcribeStart).Milliseconds(),
				"llm_to_first_audio_ms", time.Since(llmStart).Milliseconds(),
				"chunk", chunk,
			)
		}
		p.logger.Debug("tts chunk done",
			"elapsed_ms", time.Since(t1).Milliseconds(),
			"samples", len(aud.Samples),
			"chunk", chunk,
		)
		if werr := out.WriteAudio(aud.Samples, aud.SampleRate); werr != nil {
			p.logger.Error("write outbound audio", "err", werr)
		}
		if p.audioStore != nil && aud.SampleRate == defaultTTSStoreSampleRate {
			p.audioStore.AppendTTSOutFloat32(append(audio.MonoF32(nil), aud.Samples...))
		}
		if p.inspector != nil {
			p.inspector.Emit("pacer.played_ms", "played_ms", out.PlayedMs())
		}
		return true
	}
}
