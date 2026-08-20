package realtime

import (
	"encoding/base64"
	"encoding/json"
	"time"

	"github.com/eordano/speaches-plus-go/internal/audio"
	"github.com/eordano/speaches-plus-go/internal/inspect"
)

var validClientEvents = map[string]struct{}{
	"session.update":             {},
	"input_audio_buffer.append":  {},
	"input_audio_buffer.commit":  {},
	"input_audio_buffer.clear":   {},
	"conversation.item.create":   {},
	"conversation.item.delete":   {},
	"conversation.item.truncate": {},
	"conversation.item.retrieve": {},
	"response.create":            {},
	"response.cancel":            {},
}

func (p *sessionPipeline) handleClientEvent(payload []byte) {
	var env clientEventEnvelope
	if err := json.Unmarshal(payload, &env); err != nil {
		p.logger.Debug("client event parse failed", "err", err)
		p.emitErrorCode("invalid_request_error", "could not parse event JSON")
		return
	}

	if sess, _, _ := p.phase.snapshot(); sess.Kind() != sessKindActive {
		p.emitErrorCode("session_not_active",
			"inbound event before session.created: "+env.Type)
		return
	}
	if _, ok := validClientEvents[env.Type]; !ok {

		if isKnownV2NoopEvent(env.Type) {
			return
		}
		p.emitErrorCode("unknown_event_type", "unknown event type: "+env.Type)
		return
	}

	switch env.Type {
	case "session.update":
		echo := func() {
			p.emitSessionUpdated(session{
				ID:                p.sessionID,
				Object:            SessionObjectRealtimeSession,
				Model:             p.session.Model,
				Modalities:        modalitiesForIntent(p.session.Conversation),
				InputAudioFormat:  p.session.InputAudioFormat,
				OutputAudioFormat: p.session.OutputAudioFormat,
				Instructions:      p.getInstructions(),
				Voice:             p.session.Voice,
			})
		}
		if env.Session == nil {
			p.emitErrorTyped("invalid_request_error", "session_update_invalid",
				"missing session field", "session")
			echo()
			return
		}
		if err := p.applySessionUpdate(env.Session); err != nil {
			p.emitErrorTyped("invalid_request_error", "session_update_invalid",
				err.Error(), "")
			echo()
			return
		}
		echo()

	case "response.cancel":

		if p.rollbackPredictedIfAny("cancel_event") {
			return
		}
		eff := p.phase.onResponseCancel()
		if !eff.cancelled {
			p.emitErrorCode("response_cancel_not_active", "no active response to cancel")
			return
		}
		if out := p.getOutboundTTS(); out != nil {
			eff.playedMs = out.PlayedMs()
		}
		p.logger.Info("response.cancel from client", "id", eff.id, "played_ms", eff.playedMs)
		p.cancelCommitTimer()
		p.emitResponseCancelled(eff.id, eff.itemID, eff.playedMs)

	case "response.create":

		_, _, _, resp := p.phase.snapshotFull()
		if k := resp.Kind(); k == respKindPredicted || k == respKindCreated || k == respKindStreaming {
			p.emitErrorTyped("invalid_request_error", "response_already_active",
				"a response is already in progress", "")
			return
		}
		var instr *string
		var modalities []string
		if env.Response != nil {
			if env.Response.Instructions != "" {
				s := env.Response.Instructions
				instr = &s
			}
			modalities = env.Response.Modalities
		}
		itemID := newItemID()
		p.logger.Info("response.create from client (manual trigger)",
			"item_id", itemID,
			"has_instr_override", instr != nil,
			"modalities_override", modalities,
		)

		p.startResponse(itemID, "", time.Now().UnixMilli(), instr, modalities)

	case "input_audio_buffer.append":
		if env.Audio == "" {
			return
		}
		raw, err := base64.StdEncoding.DecodeString(env.Audio)
		if err != nil {
			p.emitErrorCode("invalid_request_error", "audio payload not valid base64")
			return
		}

		samples := decodeAppendAudio(raw, p.session.InputAudioFormat)
		p.appendAudio(samples)

	case "input_audio_buffer.commit":

		p.bufMu.Lock()
		samples := append(audio.MonoF32(nil), p.buf16k...)
		p.bufMu.Unlock()
		if int64(Samples(len(samples)).ToMillis(SR16k)) < int64(p.session.MinSpeechMs) {
			p.emitErrorCode("input_audio_buffer_commit_empty",
				"buffer is below min_speech_ms")
			return
		}

		p.timerMu.Lock()
		hadCommitTimer := p.commitTimer != nil
		p.timerMu.Unlock()
		if hadCommitTimer {
			p.eouMu.Lock()
			eouKind := string(p.eouCfg.Kind)
			p.eouMu.Unlock()
			p.emitEOU("cancelled", inspect.EOUFields{
				EouKind:     eouKind,
				CancelledBy: "commit_event",
			})
		}

		itemID := p.bufItemID()

		endMs := Samples(p.audioCursor.Load()).ToMillis(SR16k)
		p.phase.onVadSpeechEnd(endMs)
		p.fireCommitTimer(itemID, samples, "", false)

	case "input_audio_buffer.clear":
		if p.phase.clearInputBuffer() {
			p.cancelCommitTimer()
			p.bufMu.Lock()
			p.buf16k = p.buf16k[:0]
			p.flushed = false
			p.bufMu.Unlock()
			p.emitInputBufferCleared()
		}

	case "conversation.item.create":
		if env.Item == nil || env.Item.ID == "" {
			p.emitErrorTyped("invalid_request_error", "invalid_request_error",
				"missing item.id", "item.id")
			return
		}
		role := env.Item.Role
		if role == "" {
			role = "user"
		}

		var transcript string
		for _, c := range env.Item.Content {
			if c.Transcript != "" {
				transcript = c.Transcript
				break
			}
			if c.Text != "" {
				transcript = c.Text
				break
			}
		}
		if !p.phase.insertItem(conversationItem{
			ID:         ItemID(env.Item.ID),
			Role:       role,
			Status:     itemCompleted,
			Transcript: transcript,
		}) {
			p.emitErrorTyped("invalid_request_error", "item_already_exists",
				"item with this id already exists: "+env.Item.ID, "item.id")
			return
		}
		p.emitConversationItemCreated(conversationItemDetail{
			ID:      env.Item.ID,
			Object:  "realtime.item",
			Type:    "message",
			Status:  "completed",
			Role:    role,
			Content: env.Item.Content,
		})

	case "conversation.item.truncate":
		if env.ItemID == "" {
			p.emitErrorTyped("invalid_request_error", "invalid_request_error",
				"missing item_id", "item_id")
			return
		}
		if p.phase.truncateItem(ItemID(env.ItemID), Millis(env.AudioEnd), "") {
			p.emitConversationItemTruncated(env.ItemID, env.AudioEnd)
		} else {
			p.emitErrorCode("invalid_request_error", "item not found: "+env.ItemID)
		}

	case "conversation.item.delete":
		if env.ItemID == "" {
			p.emitErrorTyped("invalid_request_error", "invalid_request_error",
				"missing item_id", "item_id")
			return
		}
		if p.phase.deleteItem(ItemID(env.ItemID)) {
			p.emitConversationItemDeleted(env.ItemID)
		} else {
			p.emitErrorCode("invalid_request_error", "item not found: "+env.ItemID)
		}

	case "conversation.item.retrieve":
		p.emitErrorCode("invalid_request_error",
			"conversation.item.retrieve is not yet implemented")
	}
}

func pcm16BytesToFloat32(b audio.PCM16Bytes) audio.MonoF32 {
	n := len(b) / 2
	if n == 0 {
		return nil
	}
	int16s := make(audio.MonoS16, n)
	for i := 0; i < n; i++ {
		int16s[i] = int16(b[2*i]) | int16(b[2*i+1])<<8
	}
	return audio.MonoS16ToF32(int16s)
}

func decodeAppendAudio(raw []byte, format string) audio.MonoF32 {
	switch format {
	case "pcm16":
		f32 := pcm16BytesToFloat32(raw)
		return audio.LinearResampleF32(f32, 24000, whisperSampleRate)
	case "g711_ulaw":
		f32 := audio.ULawBytesToF32(raw)
		return audio.LinearResampleF32(f32, 8000, whisperSampleRate)
	case "g711_alaw":
		f32 := audio.ALawBytesToF32(raw)
		return audio.LinearResampleF32(f32, 8000, whisperSampleRate)
	case "pcm16_16k":
		fallthrough
	default:
		return pcm16BytesToFloat32(raw)
	}
}

func (p *sessionPipeline) appendAudio(samples audio.MonoF32) {
	if len(samples) == 0 {
		return
	}

	p.processIncomingAudio(samples)
}

func (p *sessionPipeline) bufItemID() string {
	_, _, buf, _ := p.phase.snapshotFull()
	if id := bufItemIDOf(buf); id != "" {
		return string(id)
	}
	return newItemID()
}
