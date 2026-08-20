package eou

import (
	"context"
	"math"
	"time"
)

type Kind string

type FusionRule string

func ValidFusionRule(r FusionRule) bool {
	switch r {
	case FusionNoisyOr, FusionMax, FusionMean, FusionWeighted, FusionGated:
		return true
	}
	return false
}

func FuseScores(rule FusionRule, pText, pAudio, weightText float32) float32 {

	tFail := isGarbageProb(pText)
	aFail := isGarbageProb(pAudio)
	if tFail && aFail {
		return 1
	}
	if tFail {
		return clamp01f(pAudio)
	}
	if aFail {
		return clamp01f(pText)
	}
	pt := clamp01f(pText)
	pa := clamp01f(pAudio)
	switch rule {
	case FusionMax:
		if pt > pa {
			return pt
		}
		return pa
	case FusionMean:
		return (pt + pa) / 2
	case FusionWeighted:
		w := weightText

		if math.IsNaN(float64(w)) {
			w = 0
		} else if w < 0 {
			w = 0
		} else if w > 1 {
			w = 1
		}
		return w*pt + (1-w)*pa
	case FusionGated:

		return (pt + pa) / 2
	case FusionNoisyOr:
		fallthrough
	default:
		return 1 - (1-pt)*(1-pa)
	}
}

func clamp01f(v float32) float32 {
	if math.IsNaN(float64(v)) {
		return 0
	}
	if v < 0 {
		return 0
	}
	if v > 1 {
		return 1
	}
	return v
}

func FuseScoresWithFeatures(rule FusionRule, pText, pAudio, weightText float32,
	feat GatedFusionFeatures, weights GatedFusionWeights) float32 {
	if rule == FusionGated {
		return FuseScoresGated(pText, pAudio, feat, weights)
	}
	return FuseScores(rule, pText, pAudio, weightText)
}

type Verdict struct {
	Score      Score
	EagerScore Score
	Latency    time.Duration
}

type Score = float32

type Request struct {
	Kind     Kind
	Turns    []Turn
	Partial  string
	Language string
	Audio    []float32
}

type Model interface {
	Predict(ctx context.Context, req Request) (Verdict, error)
	Close() error
}

type IntegratedSignal struct {
	Type            string
	PEot            float32
	PEagerEot       float32
	TranscriptSoFar string
	Reason          string
}

type IntegratedSource interface {
	Signals() <-chan IntegratedSignal
}

func SigmoidLerpK(score, threshold float32, minMs, maxMs int, k float32) int {
	if maxMs < minMs {
		maxMs = minMs
	}
	if score < threshold {
		return maxMs
	}
	span := 1.0 - float64(threshold)
	if span <= 0 {
		return minMs
	}
	x := (float64(score) - float64(threshold)) / span
	if x < 0 {
		x = 0
	} else if x > 1 {
		x = 1
	}
	if k <= 0 {
		k = DefaultCurveK
	}
	kf := float64(k)
	logistic := 1.0 / (1.0 + math.Exp(-kf*(x-0.5)))

	lo := 1.0 / (1.0 + math.Exp(-kf*(0-0.5)))
	hi := 1.0 / (1.0 + math.Exp(-kf*(1-0.5)))
	norm := (logistic - lo) / (hi - lo)
	d := float64(maxMs) - (float64(maxMs)-float64(minMs))*norm
	return int(d + 0.5)
}

func SigmoidLerp(score, threshold float32, minMs, maxMs int) int {
	return SigmoidLerpK(score, threshold, minMs, maxMs, DefaultCurveK)
}
