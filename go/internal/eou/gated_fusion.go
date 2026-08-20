package eou

import (
	"math"
	"strings"
	"unicode"
)

type GatedFusionFeatures struct {
	AudioMs int

	PartialChars int

	PartialEndsWithStrongTerminator bool

	PartialEndsWithSoftTerminator bool

	PartialLastWordIsContinuation bool
}

func ExtractGatedFusionFeatures(partial string, audioMs int) GatedFusionFeatures {
	trimmed := strings.TrimSpace(partial)
	last := lastRuneOf(trimmed)

	feat := GatedFusionFeatures{
		AudioMs:      audioMs,
		PartialChars: len(trimmed),
	}
	switch last {
	case '.', '!', '?':
		feat.PartialEndsWithStrongTerminator = true
	case ',', ';', ':', '-':
		feat.PartialEndsWithSoftTerminator = true
	}
	feat.PartialLastWordIsContinuation = isContinuationLastWord(trimmed)
	return feat
}

func lastRuneOf(s string) rune {
	r := rune(0)
	for _, c := range s {
		r = c
	}
	return r
}

func isContinuationLastWord(s string) bool {
	if s == "" {
		return false
	}

	end := len(s)
	for end > 0 {
		r, size := lastRuneSize(s[:end])
		if unicode.IsLetter(r) || unicode.IsDigit(r) || r == '\'' || r == '-' {
			break
		}
		end -= size
	}
	start := end
	for start > 0 {
		r, size := lastRuneSize(s[:start])
		if !(unicode.IsLetter(r) || unicode.IsDigit(r) || r == '\'' || r == '-') {
			break
		}
		start -= size
	}
	if start >= end {
		return false
	}
	word := strings.ToLower(s[start:end])
	for _, c := range continuationWords {
		if word == c {
			return true
		}
	}
	return false
}

func lastRuneSize(s string) (rune, int) {
	if s == "" {
		return 0, 0
	}
	for i := len(s) - 1; i >= 0; i-- {
		if (s[i] & 0xC0) != 0x80 {
			r := []rune(s[i:])
			if len(r) == 0 {
				return 0, 0
			}
			return r[0], len(s) - i
		}
	}
	return 0, 0
}

var continuationWords = []string{
	"and", "or", "but", "with", "the", "a", "an", "to", "of", "for",
	"is", "was", "are", "were", "because", "since", "if", "when",
	"while", "as", "than", "that", "which", "who", "whom", "whose",
}

type GatedFusionWeights struct {
	Bias                  float32
	WPText                float32
	WPAudio               float32
	WAudioLogSec          float32
	WPartialLogChars      float32
	WStrongTerminator     float32
	WSoftTerminator       float32
	WContinuationLastWord float32

	TrainedSamples int
	TrainedAcc     float32
}

var DefaultGatedFusionWeights = GatedFusionWeights{
	Bias:                  0.866202,
	WPText:                0.283641,
	WPAudio:               0.018662,
	WAudioLogSec:          0.560501,
	WPartialLogChars:      1.195453,
	WStrongTerminator:     0.258435,
	WSoftTerminator:       0.003248,
	WContinuationLastWord: 0.081883,
	TrainedSamples:        350,
	TrainedAcc:            0.9314,
}

func (f GatedFusionFeatures) FeatureVector(pText, pAudio float32) [8]float32 {
	logSec := float32(math.Log1p(float64(f.AudioMs) / 1000.0))
	logChars := float32(math.Log1p(float64(f.PartialChars)))
	bool01 := func(b bool) float32 {
		if b {
			return 1
		}
		return 0
	}
	return [8]float32{
		1.0,
		clamp01(pText),
		clamp01(pAudio),
		logSec,
		logChars,
		bool01(f.PartialEndsWithStrongTerminator),
		bool01(f.PartialEndsWithSoftTerminator),
		bool01(f.PartialLastWordIsContinuation),
	}
}

func (w GatedFusionWeights) Gate(pText, pAudio float32, feat GatedFusionFeatures) float32 {
	x := feat.FeatureVector(pText, pAudio)
	z := w.Bias*x[0] +
		w.WPText*x[1] +
		w.WPAudio*x[2] +
		w.WAudioLogSec*x[3] +
		w.WPartialLogChars*x[4] +
		w.WStrongTerminator*x[5] +
		w.WSoftTerminator*x[6] +
		w.WContinuationLastWord*x[7]
	return float32(1.0 / (1.0 + math.Exp(-float64(z))))
}

func FuseScoresGated(pText, pAudio float32, feat GatedFusionFeatures, w GatedFusionWeights) float32 {
	if isGarbageProb(pText) && isGarbageProb(pAudio) {
		return 1
	}
	if isGarbageProb(pText) {
		return clamp01(pAudio)
	}
	if isGarbageProb(pAudio) {
		return clamp01(pText)
	}
	pt := clamp01(pText)
	pa := clamp01(pAudio)
	g := w.Gate(pt, pa, feat)
	return clamp01(g*pa + (1-g)*pt)
}

func isGarbageProb(p float32) bool {
	f := float64(p)
	if math.IsNaN(f) || math.IsInf(f, 0) {
		return true
	}
	return p < 0 || p > 1
}

func clamp01(p float32) float32 {
	if math.IsNaN(float64(p)) || math.IsInf(float64(p), 0) {
		return 0
	}
	if p < 0 {
		return 0
	}
	if p > 1 {
		return 1
	}
	return p
}
