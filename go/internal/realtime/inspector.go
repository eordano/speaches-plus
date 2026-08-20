package realtime

import (
	"context"

	"github.com/eordano/speaches-plus-go/internal/inspect"
)

type Inspector interface {
	Emit(event string, attrs ...any)
}

type relayInspector struct {
	relay *inspect.Relay
	ctx   context.Context
}

func NewInspector(relay *inspect.Relay) Inspector {
	if relay == nil {
		return noopInspector{}
	}
	return &relayInspector{relay: relay, ctx: context.Background()}
}

func (i *relayInspector) Emit(event string, attrs ...any) {
	if i == nil || i.relay == nil {
		return
	}
	lane, kind := splitInspectorEventName(event)
	payload := attrsToMap(attrs)
	inspect.Emit(i.relay, i.ctx, lane, kind, nil, payload)
}

func attrsToMap(attrs []any) map[string]any {
	if len(attrs) == 0 {
		return nil
	}
	m := make(map[string]any, len(attrs)/2)
	for i := 0; i+1 < len(attrs); i += 2 {
		k, ok := attrs[i].(string)
		if !ok {
			continue
		}
		m[k] = attrs[i+1]
	}
	return m
}

func splitInspectorEventName(event string) (inspect.LaneID, string) {
	for i := 0; i < len(event); i++ {
		if event[i] == '.' {
			lane := inspect.LaneID(event[:i])
			if !inspect.ValidLane(lane) {
				lane = inspect.LaneWire
			}
			return lane, event[i+1:]
		}
	}
	return inspect.LaneWire, event
}

type noopInspector struct{}

func (noopInspector) Emit(string, ...any) {}

func (p *sessionPipeline) emitEOU(kind string, fields inspect.EOUFields) {
	inspect.EmitEOU(p.relay, p.traceContext(), kind, fields)
	if p.inspector != nil {
		p.inspector.Emit("eou."+kind, eouFieldsToAttrs(fields)...)
	}
}

func (p *sessionPipeline) emitDiarization(kind string, fields inspect.DiarizationFields) {
	inspect.EmitDiarization(p.relay, p.traceContext(), kind, fields)
	if p.inspector != nil {
		attrs := []any{
			"item_id", fields.ItemID,
			"audio_end_ms", fields.AudioEndMs,
			"num_segments", fields.NumSegments,
			"num_speakers", fields.NumSpeakers,
			"overlap_count", fields.OverlapCount,
			"elapsed_ms", fields.ElapsedMs,
		}
		if fields.Failed {
			attrs = append(attrs, "failed", true)
		}
		if fields.Reason != "" {
			attrs = append(attrs, "reason", fields.Reason)
		}
		p.inspector.Emit("diarization."+kind, attrs...)
	}
}

func eouFieldsToAttrs(f inspect.EOUFields) []any {
	out := make([]any, 0, 28)
	if f.EouKind != "" {
		out = append(out, "eou_kind", f.EouKind)
	}
	if f.Score != nil {
		out = append(out, "score", *f.Score)
	}
	if f.ScoreText != nil {
		out = append(out, "score_text", *f.ScoreText)
	}
	if f.ScoreAudio != nil {
		out = append(out, "score_audio", *f.ScoreAudio)
	}
	if f.FusionRule != "" {
		out = append(out, "fusion_rule", f.FusionRule)
	}
	if f.Threshold != nil {
		out = append(out, "threshold", *f.Threshold)
	}
	if f.Language != "" {
		out = append(out, "language", f.Language)
	}
	if f.CurveK != nil {
		out = append(out, "curve_k", *f.CurveK)
	}
	if f.DelayMs != nil {
		out = append(out, "delay_ms", *f.DelayMs)
	}
	if f.ElapsedMs != nil {
		out = append(out, "elapsed_ms", *f.ElapsedMs)
	}
	if f.CancelledBy != "" {
		out = append(out, "cancelled_by", f.CancelledBy)
	}
	if f.HardCapPhase != "" {
		out = append(out, "phase", f.HardCapPhase)
	}
	if f.FailureReason != "" {
		out = append(out, "failure_reason", f.FailureReason)
	}
	for k, v := range f.Extra {
		out = append(out, k, v)
	}
	return out
}
