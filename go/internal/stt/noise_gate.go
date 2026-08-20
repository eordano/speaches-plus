package stt

const (
	gateFullMs = 1500

	gateOffMs = 5000

	gateLooseFloor = -3.0
)

func EffectiveAvgLogprobThreshold(base *float32, durationMs int) *float32 {
	if base == nil {
		return nil
	}
	if durationMs <= gateFullMs {
		v := *base
		return &v
	}
	if durationMs >= gateOffMs {
		return nil
	}
	frac := float32(durationMs-gateFullMs) / float32(gateOffMs-gateFullMs)
	v := *base + frac*(gateLooseFloor-*base)
	return &v
}

type NoiseRejection int

const (
	NoiseAccept NoiseRejection = iota
	NoiseRejectNoSpeechProb
	NoiseRejectAvgLogprob
)

func (r NoiseRejection) String() string {
	switch r {
	case NoiseRejectNoSpeechProb:
		return "no_speech_prob"
	case NoiseRejectAvgLogprob:
		return "avg_logprob"
	default:
		return "accept"
	}
}

type GateThresholds struct {
	NoSpeechProb *float32
	AvgLogprob   *float32
}

func EvaluateNoiseGate(
	avgNoSpeechProb *float32,
	avgLogprob *float32,
	durationMs int,
	thr GateThresholds,
) NoiseRejection {
	if avgNoSpeechProb != nil && thr.NoSpeechProb != nil {
		if *avgNoSpeechProb > *thr.NoSpeechProb {
			return NoiseRejectNoSpeechProb
		}
	}
	if avgLogprob != nil {
		eff := EffectiveAvgLogprobThreshold(thr.AvgLogprob, durationMs)
		if eff != nil && *avgLogprob < *eff {
			return NoiseRejectAvgLogprob
		}
	}
	return NoiseAccept
}
