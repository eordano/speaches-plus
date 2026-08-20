package inspect

import (
	"context"
	"time"

	"go.opentelemetry.io/otel/trace"
)

func Emit(r *Relay, ctx context.Context, lane LaneID, kind string, corr *Corr, payload map[string]any) {
	if r == nil {
		return
	}
	c := r.Corr()
	if corr != nil {
		if corr.TurnID != "" {
			c.TurnID = corr.TurnID
		}
		if corr.ItemID != "" {
			c.ItemID = corr.ItemID
		}
		if corr.ResponseID != "" {
			c.ResponseID = corr.ResponseID
		}
		if corr.PhraseID != "" {
			c.PhraseID = corr.PhraseID
		}
	}
	span := ""
	if ctx != nil {
		s := trace.SpanFromContext(ctx)
		if sc := s.SpanContext(); sc.IsValid() {
			span = sc.SpanID().String()
		}
	}
	now := time.Now()
	ev := Event{
		SessionID: r.SessionID(),
		Seq:       r.NextSeq(),
		TSMonoNS:  now.UnixNano(),
		TSWall:    float64(now.UnixNano()) / 1e9,
		Lane:      lane,
		Kind:      kind,
		Corr:      c,
		SpanID:    span,
		Payload:   payload,
	}
	r.Publish(ev)
}

func SetTurnID(r *Relay, v string)     { r.SetCorr(&v, nil, nil, nil) }
func SetItemID(r *Relay, v string)     { r.SetCorr(nil, &v, nil, nil) }
func SetResponseID(r *Relay, v string) { r.SetCorr(nil, nil, &v, nil) }
func SetPhraseID(r *Relay, v string)   { r.SetCorr(nil, nil, nil, &v) }

type EOUFields struct {
	EouKind       string
	Score         *float32
	ScoreText     *float32
	ScoreAudio    *float32
	FusionRule    string
	Threshold     *float32
	Language      string
	CurveK        *float32
	DelayMs       *int
	ElapsedMs     *int
	CancelledBy   string
	HardCapPhase  string
	FailureReason string
	Extra         map[string]any
}

type DiarizationFields struct {
	ItemID       string
	AudioEndMs   int64
	NumSegments  int
	NumSpeakers  int
	OverlapCount int
	ElapsedMs    int
	Failed       bool
	Reason       string
	Extra        map[string]any
}

func EmitDiarization(r *Relay, ctx context.Context, kind string, fields DiarizationFields) {
	if r == nil {
		return
	}
	payload := map[string]any{
		"audio_end_ms":  fields.AudioEndMs,
		"num_segments":  fields.NumSegments,
		"num_speakers":  fields.NumSpeakers,
		"overlap_count": fields.OverlapCount,
		"elapsed_ms":    fields.ElapsedMs,
	}
	if fields.Failed {
		payload["failed"] = true
	}
	if fields.Reason != "" {
		payload["reason"] = fields.Reason
	}
	for k, v := range fields.Extra {
		payload[k] = v
	}
	corr := &Corr{}
	if fields.ItemID != "" {
		corr.ItemID = fields.ItemID
	}
	Emit(r, ctx, LaneDiarization, kind, corr, payload)
}

func EmitEOU(r *Relay, ctx context.Context, kind string, fields EOUFields) {
	if r == nil {
		return
	}
	payload := map[string]any{}
	if fields.EouKind != "" {
		payload["eou_kind"] = fields.EouKind
	}
	if fields.Score != nil {
		payload["score"] = *fields.Score
	}
	if fields.ScoreText != nil {
		payload["score_text"] = *fields.ScoreText
	}
	if fields.ScoreAudio != nil {
		payload["score_audio"] = *fields.ScoreAudio
	}
	if fields.FusionRule != "" {
		payload["fusion_rule"] = fields.FusionRule
	}
	if fields.Threshold != nil {
		payload["threshold"] = *fields.Threshold
	}
	if fields.Language != "" {
		payload["language"] = fields.Language
	}
	if fields.CurveK != nil {
		payload["curve_k"] = *fields.CurveK
	}
	if fields.DelayMs != nil {
		payload["delay_ms"] = *fields.DelayMs
	}
	if fields.ElapsedMs != nil {
		payload["elapsed_ms"] = *fields.ElapsedMs
	}
	if fields.CancelledBy != "" {
		payload["cancelled_by"] = fields.CancelledBy
	}
	if fields.HardCapPhase != "" {
		payload["hard_cap_phase"] = fields.HardCapPhase
	}
	if fields.FailureReason != "" {
		payload["failure_reason"] = fields.FailureReason
	}
	for k, v := range fields.Extra {
		payload[k] = v
	}
	Emit(r, ctx, LaneEOU, kind, nil, payload)
}
