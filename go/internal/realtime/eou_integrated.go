package realtime

import (
	"go.opentelemetry.io/otel/attribute"

	"github.com/eordano/speaches-plus-go/internal/audio"
	"github.com/eordano/speaches-plus-go/internal/eou"
	"github.com/eordano/speaches-plus-go/internal/inspect"
)

func (p *sessionPipeline) startIntegratedConsumer() {
	src := p.server.IntegratedSource
	if src == nil {
		return
	}
	if p.eouCfg.Kind != eou.KindIntegrated {
		return
	}
	signals := src.Signals()
	if signals == nil {
		return
	}
	p.wg.Add(1)
	go func() {
		defer p.wg.Done()
		for {
			select {
			case <-p.closed:
				return
			case sig, ok := <-signals:
				if !ok {
					return
				}
				p.dispatchIntegratedSignal(sig)
			}
		}
	}()
}

func (p *sessionPipeline) dispatchIntegratedSignal(sig eou.IntegratedSignal) {
	cfg := p.eouCfg
	switch sig.Type {
	case "stt.eot_predicted":
		if sig.PEot >= cfg.EotThreshold {
			pEot := sig.PEot
			thr := cfg.EotThreshold
			p.emitEOU("integrated_commit", inspect.EOUFields{
				EouKind:   string(eou.KindIntegrated),
				Score:     &pEot,
				Threshold: &thr,
			})
			p.logger.Info("integrated EOU: commit",
				"p_eot", sig.PEot, "threshold", cfg.EotThreshold)
			p.bufMu.Lock()
			samples := append(audio.MonoF32(nil), p.buf16k...)
			p.buf16k = p.buf16k[:0]
			p.bufMu.Unlock()
			itemID := p.bufItemID()
			if itemID == "" {
				itemID = newItemID()
			}
			endMs := int64(len(samples)) * 1000 / int64(whisperSampleRate)
			eff := p.phase.forceCommitForIntegrated(ItemID(itemID), Millis(endMs))
			if !eff.committed {
				return
			}
			p.cancelCommitTimer()
			p.emitInputBufferCommitted(string(eff.itemID))
			p.emitConversationItemCreated(conversationItemDetail{
				ID:      string(eff.itemID),
				Object:  "realtime.item",
				Type:    "message",
				Status:  "in_progress",
				Role:    "user",
				Content: []responseContentPart{{Type: "input_audio"}},
			})
			if sig.TranscriptSoFar != "" {
				p.runFromPartial(string(eff.itemID), sig.TranscriptSoFar, true)
			} else if p.server.STT != nil {
				p.runTranscription(samples, string(eff.itemID), true)
			}
			p.wg.Add(1)
			go func() {
				defer p.wg.Done()
				p.waitForTurnDone()
				p.resetTurn()
			}()
			return
		}
		if sig.PEagerEot >= cfg.EagerEotThreshold {
			_, dispSpan := startSpan(p.traceContext(), "eou.predicted_dispatch",
				attribute.Float64("eou.score", float64(sig.PEagerEot)),
				attribute.Bool("eou.eager", true),
			)
			var (
				spanRespID        string
				spanRunnerStarted bool
			)
			func() {
				defer func() {
					dispSpan.SetAttributes(
						attribute.String("eou.response_id", spanRespID),
						attribute.Bool("eou.runner_started", spanRunnerStarted),
					)
					dispSpan.End()
				}()
				_, _, _, resp := p.phase.snapshotFull()
				if resp.Kind() != respKindNone {
					return
				}
				predID := newRespID()
				spanRespID = predID
				respItemID := newItemID()

				r := p.startEagerRunner(predID, respItemID, 0, sig.TranscriptSoFar, nil)
				if r == nil {
					return
				}
				_, ok := p.phase.onPredictedDispatch(ResponseID(predID), ItemID(respItemID), sig.PEagerEot, r)
				if !ok {
					r.abort()
					return
				}
				spanRunnerStarted = true

				pEager := sig.PEagerEot
				p.emitEOU("integrated_eager_dispatch", inspect.EOUFields{
					EouKind: string(eou.KindIntegrated),
					Score:   &pEager,
					Extra:   map[string]any{"id": predID},
				})
			}()
		}
	case "stt.turn_resumed":
		p.rollbackPredictedIfAny("turn_resumed:" + sig.Reason)
	}
}
