package stt

import "testing"

func ptrF32(v float32) *float32 { return &v }

func TestEffective_BelowFullMsReturnsBase(t *testing.T) {
	got := EffectiveAvgLogprobThreshold(ptrF32(-1.0), 1500)
	if got == nil || *got != -1.0 {
		t.Fatalf("got %v want -1.0", got)
	}
}

func TestEffective_AboveOffMsDisabled(t *testing.T) {
	if EffectiveAvgLogprobThreshold(ptrF32(-1.0), 5000) != nil {
		t.Fatalf("expected nil at exactly OFF_MS")
	}
	if EffectiveAvgLogprobThreshold(ptrF32(-1.0), 60_000) != nil {
		t.Fatalf("expected nil at high duration")
	}
}

func TestEffective_LinearMidwayLerp(t *testing.T) {

	got := EffectiveAvgLogprobThreshold(ptrF32(-1.0), 3250)
	if got == nil {
		t.Fatalf("got nil")
	}
	if abs := *got + 2.0; abs > 1e-3 || abs < -1e-3 {
		t.Fatalf("got %v want ~-2.0", *got)
	}
}

func TestEffective_NilBaseDisabled(t *testing.T) {
	if EffectiveAvgLogprobThreshold(nil, 1000) != nil {
		t.Fatalf("nil base must yield nil")
	}
}

func TestEvaluate_PassesWhenDisabled(t *testing.T) {
	got := EvaluateNoiseGate(ptrF32(0.99), ptrF32(-10.0), 1000, GateThresholds{})
	if got != NoiseAccept {
		t.Fatalf("got %v want accept", got)
	}
}

func TestEvaluate_RejectsNspFirst(t *testing.T) {
	thr := GateThresholds{NoSpeechProb: ptrF32(0.6), AvgLogprob: ptrF32(-1.0)}
	got := EvaluateNoiseGate(ptrF32(0.9), ptrF32(-5.0), 500, thr)
	if got != NoiseRejectNoSpeechProb {
		t.Fatalf("got %v want NSP", got)
	}
}

func TestEvaluate_RejectsLogprob(t *testing.T) {
	thr := GateThresholds{NoSpeechProb: ptrF32(0.6), AvgLogprob: ptrF32(-1.0)}
	got := EvaluateNoiseGate(ptrF32(0.1), ptrF32(-2.0), 500, thr)
	if got != NoiseRejectAvgLogprob {
		t.Fatalf("got %v want logprob", got)
	}
}

func TestEvaluate_LongAudioBypassesLogprobGate(t *testing.T) {
	thr := GateThresholds{NoSpeechProb: ptrF32(0.6), AvgLogprob: ptrF32(-0.5)}
	got := EvaluateNoiseGate(ptrF32(0.1), ptrF32(-10.0), 6000, thr)
	if got != NoiseAccept {
		t.Fatalf("got %v want accept (logprob gate disabled at long durations)", got)
	}
}

func TestEvaluate_SkipsWhenStatsMissing(t *testing.T) {
	thr := GateThresholds{NoSpeechProb: ptrF32(0.0), AvgLogprob: ptrF32(0.0)}
	got := EvaluateNoiseGate(nil, nil, 1000, thr)
	if got != NoiseAccept {
		t.Fatalf("got %v want accept (no stats -> no rejection)", got)
	}
}
