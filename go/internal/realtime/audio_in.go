package realtime

import (
	"errors"
	"io"
	"time"

	"github.com/pion/webrtc/v4"
	"go.opentelemetry.io/otel/attribute"

	"github.com/eordano/speaches-plus-go/internal/audio"
	"github.com/eordano/speaches-plus-go/internal/inspect"
)

func (p *sessionPipeline) runAudioLoop(track *webrtc.TrackRemote) {
	p.wg.Add(1)
	defer p.wg.Done()

	pcmInt16 := make([]int16, maxOpusFrameInt16)
	clockRate := int(track.Codec().ClockRate)
	if clockRate == 0 {
		clockRate = opusSampleRate
	}

	if p.vad == nil {
		p.wg.Add(1)
		go func() {
			defer p.wg.Done()
			p.silenceWatchdog()
		}()
	} else {
		defer p.vad.Close()
	}

	defer func() {
		p.bufMu.Lock()
		hasAudio := len(p.buf16k) >= p.session.StartSpeechSamples && !p.flushed
		samples := append(audio.MonoF32(nil), p.buf16k...)
		p.buf16k = p.buf16k[:0]
		p.flushed = hasAudio || p.flushed
		p.bufMu.Unlock()
		if hasAudio {
			p.runTranscription(samples, "", true)
		}
	}()

	for {
		select {
		case <-p.closed:
			return
		default:
		}

		pkt, _, err := track.ReadRTP()
		if err != nil {
			if errors.Is(err, io.EOF) {
				return
			}
			p.logger.Debug("ReadRTP ended", "err", err)
			return
		}
		if len(pkt.Payload) == 0 {
			continue
		}

		n, decErr := p.decoder.DecodeToInt16(pkt.Payload, pcmInt16)
		if decErr != nil {
			p.logger.Debug("opus decode failed", "err", decErr)
			continue
		}
		monoF32 := audio.MonoS16ToF32(pcmInt16[:n])
		resampled := audio.LinearResampleF32(monoF32, clockRate, whisperSampleRate)

		if quit := p.processIncomingAudio(resampled); quit {
			return
		}
	}
}

func (p *sessionPipeline) processIncomingAudio(resampled audio.MonoF32) (quit bool) {
	if len(resampled) == 0 {
		return false
	}

	var window audio.MonoF32
	var firedSpeechStop bool
	var stoppedItemID string
	var stoppedEndMs int64
	var stoppedSamples audio.MonoF32

	p.bufMu.Lock()
	p.buf16k = append(p.buf16k, resampled...)

	p.audioCursor.Add(int64(len(resampled)))
	if p.audioStore != nil {
		p.audioStore.AppendMicIn(append(audio.MonoF32(nil), resampled...))
	}
	if hasNonSilenceWith(resampled, p.session.NonSilenceThreshold) {
		p.lastAudio = time.Now()
	}

	if p.vad != nil && !p.flushed {
		win := p.vad.WindowSamples()
		for p.vadCursor+win <= len(p.buf16k) {
			window = append(window[:0], p.buf16k[p.vadCursor:p.vadCursor+win]...)
			p.vadCursor += win
			p.bufMu.Unlock()
			dec, vadErr := p.vad.Process(window)
			p.bufMu.Lock()
			if vadErr != nil {
				p.logger.Warn("vad process error", "err", vadErr)
				if p.onVADFailure(vadErr) {
					p.bufMu.Unlock()
					return true
				}
				continue
			}
			p.vadFailures.Store(0)
			if dec == vadSpeechStart {
				p.logger.Debug("vad speech_start", "cursor", p.vadCursor)
				if p.inspector != nil {
					p.inspector.Emit("vad.confirmed_start", "cursor", p.vadCursor)
				}
				itemID := newItemID()

				startMs := int64(Samples(p.vadCursor).ToMillis(SR16k))
				if pad := DurationMs(p.session.VADPrefixPaddingMs); pad > 0 {
					startMs -= int64(pad)
					if startMs < 0 {
						startMs = 0
					}
				}

				if p.session.BargeInDelayMs > 0 {
					_, _, _, resp := p.phase.snapshotFull()
					if k := resp.Kind(); k == respKindCreated || k == respKindStreaming {
						p.armBargeInTask(itemID, startMs)
						continue
					}
				}

				var snap func() int64
				if out := p.getOutboundTTS(); out != nil {
					snap = out.PlayedMs
				}
				eff := p.phase.onVadSpeechStart(itemID, startMs, snap)
				if eff.predictedRolled {

					_, rbSpan := startSpan(p.traceContext(), "eou.predicted_rollback",
						attribute.String("eou.reason", "speech_resumed"),
					)
					runnerAborted := false
					if eff.runnerToAbort != nil {
						eff.runnerToAbort.abort()
						runnerAborted = true
					}
					p.predictedMu.Lock()
					cancel := p.predictedCancel
					p.predictedCancel = nil
					p.predictedMu.Unlock()
					if cancel != nil {
						cancel()
					}
					rbSpan.SetAttributes(attribute.Bool("eou.runner_aborted", runnerAborted))
					rbSpan.End()
					p.logger.Info("eou.predicted_rollback",
						"reason", "speech_resumed")
					p.eouMu.Lock()
					eouKind := string(p.eouCfg.Kind)
					p.eouMu.Unlock()
					p.emitEOU("cancelled", inspect.EOUFields{
						EouKind:     eouKind,
						CancelledBy: "speech_started",
					})
				}
				if eff.cancelTimer {
					p.logger.Info("commit_timer cancelled by new speech")
					p.cancelCommitTimer()

					_, _, buf, _ := p.phase.snapshotFull()
					p.startPartialLoop(string(bufItemIDOf(buf)))
					continue
				}
				if eff.cancel.cancelled {
					p.logger.Info("barge-in: cancelling response",
						"id", eff.cancel.id,
						"drain", eff.cancel.wasDrain,
						"played_ms", eff.cancel.playedMs,
					)
					if p.inspector != nil {
						p.inspector.Emit("bargein.fired",
							"id", eff.cancel.id,
							"played_ms", eff.cancel.playedMs)
					}
					p.handleBargeIn(eff.cancel)
				}
				p.emitInputBufferSpeechStarted(itemID, startMs)

				p.startPartialLoop(itemID)
			}
			if dec == vadSpeechEnd {

				p.cancelBargeInTask()
			}
			if dec == vadSpeechEnd && len(p.buf16k) >= p.session.StartSpeechSamples {
				stoppedSamples = append(stoppedSamples[:0], p.buf16k...)
				stoppedEndMs = int64(p.vadCursor) * 1000 / int64(whisperSampleRate)
				p.buf16k = p.buf16k[:0]
				p.vadCursor = 0

				if p.vad != nil {
					p.vad.Reset()
				}
				firedSpeechStop = true
				var ok bool
				var stoppedItemIDTyped ItemID
				stoppedItemIDTyped, _, ok = p.phase.onVadSpeechEnd(Millis(stoppedEndMs))
				stoppedItemID = string(stoppedItemIDTyped)
				if !ok {
					firedSpeechStop = false
				}
				break
			}
		}
	}
	p.bufMu.Unlock()
	if firedSpeechStop && stoppedItemID != "" {
		p.logger.Info("vad speech_stop -> starting commit_timer", "samples", len(stoppedSamples))
		if p.inspector != nil {
			p.inspector.Emit("vad.confirmed_stop",
				"item_id", stoppedItemID,
				"end_ms", stoppedEndMs)
		}

		p.stopPartialLoop()
		p.emitInputBufferSpeechStopped(stoppedItemID, stoppedEndMs)
		p.startCommitTimer(stoppedItemID, stoppedSamples)

		diarItemID := stoppedItemID
		diarEndMs := stoppedEndMs
		diarSamples := append(audio.MonoF32(nil), stoppedSamples...)
		p.wg.Add(1)
		go func() {
			defer p.wg.Done()
			p.runDiarization(diarItemID, diarSamples, diarEndMs)
		}()
	}
	return false
}

func (p *sessionPipeline) silenceWatchdog() {
	ticker := time.NewTicker(silenceWatchdogTickMs * time.Millisecond)
	defer ticker.Stop()
	for {
		select {
		case <-p.closed:
			return
		case <-ticker.C:
			p.bufMu.Lock()
			vadlessSilence := time.Duration(p.session.VADLessSilenceMs) * time.Millisecond
			if p.flushed || p.lastAudio.IsZero() ||
				time.Since(p.lastAudio) < vadlessSilence ||
				len(p.buf16k) < p.session.StartSpeechSamples {
				p.bufMu.Unlock()
				continue
			}
			samples := append(audio.MonoF32(nil), p.buf16k...)
			p.buf16k = p.buf16k[:0]
			p.flushed = true
			p.bufMu.Unlock()
			itemID := newItemID()
			audioMs := int64(len(samples)) * 1000 / int64(whisperSampleRate)
			p.emitInputBufferSpeechStarted(itemID, 0)
			p.emitInputBufferSpeechStopped(itemID, audioMs)
			p.emitInputBufferCommitted(itemID)
			p.emitConversationItemCreated(conversationItemDetail{
				ID:     itemID,
				Object: "realtime.item",
				Type:   "message",
				Status: "in_progress",
				Role:   "user",
				Content: []responseContentPart{{
					Type: "input_audio",
				}},
			})
			p.runTranscription(samples, itemID, true)
			p.wg.Add(1)
			go func() {
				defer p.wg.Done()
				p.waitForTurnDone()
				p.resetTurn()
			}()
		}
	}
}

func hasNonSilence(samples audio.MonoF32) bool {
	return hasNonSilenceWith(samples, defaultNonSilenceThreshold)
}

func hasNonSilenceWith(samples audio.MonoF32, threshold float32) bool {
	if len(samples) == 0 {
		return false
	}
	if threshold <= 0 {
		threshold = defaultNonSilenceThreshold
	}
	var sum float32
	for _, s := range samples {
		if s < 0 {
			s = -s
		}
		sum += s
	}
	return (sum / float32(len(samples))) > threshold
}
