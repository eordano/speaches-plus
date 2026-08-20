package diarization

import (
	"math"
	"testing"
)

func TestFbankSilenceGivesFloor(t *testing.T) {
	fb := NewFBank(80, 400, 160)
	audio := make([]float32, 16000)
	feats, err := fb.Compute(audio)
	if err != nil {
		t.Fatalf("compute: %v", err)
	}
	frames := len(feats) / 80
	if frames < 90 {
		t.Fatalf("want >=90 frames for 1s @ 10ms hop, got %d", frames)
	}
	for _, v := range feats {
		if math.Abs(float64(v)) >= 1e-3 {
			t.Fatalf("post-CMN silence should be near zero, got %f", v)
		}
	}
}

func TestFbankFrameCountMatchesKaldiFormula(t *testing.T) {
	fb := NewFBank(80, 400, 160)
	audio := make([]float32, 16000)
	for i := range audio {
		audio[i] = 0.1
	}
	feats, err := fb.Compute(audio)
	if err != nil {
		t.Fatalf("compute: %v", err)
	}
	expected := 1 + (16000-400)/160
	if got := len(feats) / 80; got != expected {
		t.Fatalf("want %d frames, got %d", expected, got)
	}
}

func TestMelFiltersCoverBand(t *testing.T) {
	filters := buildMelFilters(80, 512, 16000.0, 20.0, 7600.0)
	if len(filters) != 80 {
		t.Fatalf("want 80 filters, got %d", len(filters))
	}
	for m, f := range filters {
		if len(f) == 0 {
			t.Fatalf("mel %d has no taps", m)
		}
	}
}

func TestHzMelRoundTrip(t *testing.T) {
	for _, hz := range []float64{20, 1000, 4000, 7600} {
		back := melToHz(hzToMel(hz))
		if math.Abs(back-hz) >= 0.5 {
			t.Fatalf("hz %f -> %f (drift > 0.5)", hz, back)
		}
	}
}
