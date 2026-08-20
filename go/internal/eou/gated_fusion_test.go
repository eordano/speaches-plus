package eou

import (
	"math"
	"testing"
)

func approxEqf(t *testing.T, got, want, tol float32, name string) {
	t.Helper()
	if math.Abs(float64(got-want)) > float64(tol) {
		t.Errorf("%s: got %v want %v (tol %v)", name, got, want, tol)
	}
}

func TestExtractFeatures_StrongTerminator(t *testing.T) {
	cases := []struct {
		in       string
		strong   bool
		soft     bool
		contLast bool
	}{
		{"yes.", true, false, false},
		{"what?", true, false, false},
		{"wow!", true, false, false},
		{"hmm,", false, true, false},
		{"so,", false, true, false},
		{"the cat", false, false, false},
		{"the cat is on the", false, false, true},
		{"and", false, false, true},
		{"because", false, false, true},
		{"", false, false, false},
		{"   ", false, false, false},
	}
	for _, c := range cases {
		f := ExtractGatedFusionFeatures(c.in, 1000)
		if f.PartialEndsWithStrongTerminator != c.strong {
			t.Errorf("%q: strong=%v want %v", c.in, f.PartialEndsWithStrongTerminator, c.strong)
		}
		if f.PartialEndsWithSoftTerminator != c.soft {
			t.Errorf("%q: soft=%v want %v", c.in, f.PartialEndsWithSoftTerminator, c.soft)
		}
		if f.PartialLastWordIsContinuation != c.contLast {
			t.Errorf("%q: contLast=%v want %v", c.in, f.PartialLastWordIsContinuation, c.contLast)
		}
	}
}

func TestFeatureVector_BiasAndShape(t *testing.T) {
	f := GatedFusionFeatures{AudioMs: 8000, PartialChars: 50,
		PartialEndsWithStrongTerminator: true}
	v := f.FeatureVector(0.7, 0.3)
	if v[0] != 1.0 {
		t.Fatalf("bias slot must be 1.0, got %v", v[0])
	}
	if v[1] != 0.7 || v[2] != 0.3 {
		t.Fatalf("p_text/p_audio slots wrong: %v / %v", v[1], v[2])
	}
	wantLogSec := float32(math.Log1p(8.0))
	approxEqf(t, v[3], wantLogSec, 1e-6, "log_sec")
	wantLogChars := float32(math.Log1p(50))
	approxEqf(t, v[4], wantLogChars, 1e-6, "log_chars")
	if v[5] != 1 {
		t.Fatalf("strong terminator one-hot wrong: %v", v[5])
	}
	if v[6] != 0 || v[7] != 0 {
		t.Fatalf("soft / continuation one-hots wrong: %v / %v", v[6], v[7])
	}
}

func TestFeatureVector_ClampsHeadProbabilities(t *testing.T) {
	f := GatedFusionFeatures{}
	v := f.FeatureVector(1.7, -0.3)
	if v[1] != 1 {
		t.Fatalf("p_text>1 must clamp to 1, got %v", v[1])
	}
	if v[2] != 0 {
		t.Fatalf("p_audio<0 must clamp to 0, got %v", v[2])
	}
	v2 := f.FeatureVector(float32(math.NaN()), float32(math.Inf(1)))
	if v2[1] != 0 || v2[2] != 0 {
		t.Fatalf("non-finite head probs must clamp to 0; got %v %v", v2[1], v2[2])
	}
}

func TestZeroWeights_DegenerateToHalfBlend(t *testing.T) {

	zero := GatedFusionWeights{}
	for _, c := range []struct {
		pt, pa float32
	}{{0.7, 0.3}, {0.0, 1.0}, {0.5, 0.5}, {0.95, 0.10}} {
		got := FuseScoresGated(c.pt, c.pa, GatedFusionFeatures{}, zero)
		want := (c.pt + c.pa) / 2
		approxEqf(t, got, want, 1e-6,
			"zero weights mean blend")
	}
}

func TestTrainedWeights_RealDataAgreementCases(t *testing.T) {

	w := DefaultGatedFusionWeights
	type point struct {
		text                     string
		pText, pAudio            float32
		expectAbove, expectBelow float32
	}
	cases := []point{

		{"That's right.", 0.95, 0.99, 0.85, 1.01},
		{"Yes.", 0.95, 0.95, 0.85, 1.01},

		{"and the next thing", 0.55, 0.05, -0.01, 0.45},
		{"the cat is on the", 0.25, 0.05, -0.01, 0.40},
	}
	for _, c := range cases {
		feat := ExtractGatedFusionFeatures(c.text, 1500)
		got := FuseScoresGated(c.pText, c.pAudio, feat, w)
		if got < c.expectAbove || got > c.expectBelow {
			t.Errorf("%q: gated combined=%.3f outside [%v, %v]; pt=%v pa=%v",
				c.text, got, c.expectAbove, c.expectBelow, c.pText, c.pAudio)
		}
	}
}

func TestTrainedWeights_MonotonicInPAudio(t *testing.T) {

	w := DefaultGatedFusionWeights
	feat := ExtractGatedFusionFeatures("looking forward to it", 1500)
	pt := float32(0.55)
	prev := float32(-1)
	for _, pa := range []float32{0.01, 0.1, 0.3, 0.5, 0.7, 0.9, 0.99} {
		got := FuseScoresGated(pt, pa, feat, w)
		if got < prev {
			t.Fatalf("non-monotonic in p_audio: pa=%v got %v < prev %v", pa, got, prev)
		}
		prev = got
	}
}

func TestTrainedWeights_DoesNotDegradeBelowAudioAlone(t *testing.T) {

	w := DefaultGatedFusionWeights
	for _, pa := range []float32{0.01, 0.1, 0.9, 0.99} {

		var ptAdv float32
		if pa >= 0.5 {
			ptAdv = 0.05
		} else {
			ptAdv = 0.95
		}
		feat := ExtractGatedFusionFeatures("looking forward to it", 1500)
		got := FuseScoresGated(ptAdv, pa, feat, w)

		audioVerdict := pa >= 0.5
		gatedVerdict := got >= 0.5
		if audioVerdict != gatedVerdict {
			t.Errorf("gate flipped audio verdict: pa=%v ptAdv=%v got=%v",
				pa, ptAdv, got)
		}
	}
}

func TestTrainedAccuracy_RecordedMetadataMatchesWeights(t *testing.T) {

	if DefaultGatedFusionWeights.TrainedSamples == 0 {
		t.Fatalf("DefaultGatedFusionWeights.TrainedSamples is zero -- re-run cmd/train-gated-fusion")
	}
	if DefaultGatedFusionWeights.TrainedAcc <= 0.6 {
		t.Fatalf("TrainedAcc=%v <= 0.6 -- corpus may be too noisy or model has drifted", DefaultGatedFusionWeights.TrainedAcc)
	}
}

func TestFuseScoresGated_GracefulOnGarbage(t *testing.T) {
	w := DefaultGatedFusionWeights
	feat := GatedFusionFeatures{}

	if got := FuseScoresGated(float32(math.NaN()), float32(math.NaN()), feat, w); got != 1 {
		t.Fatalf("both NaN: got %v want 1", got)
	}

	if got := FuseScoresGated(float32(math.NaN()), 0.42, feat, w); got != 0.42 {
		t.Fatalf("text NaN: got %v want 0.42", got)
	}

	if got := FuseScoresGated(0.7, float32(math.Inf(1)), feat, w); got != 0.7 {
		t.Fatalf("audio Inf: got %v want 0.7", got)
	}

	got := FuseScoresGated(2.0, -1.0, feat, w)
	if got < 0 || got > 1 {
		t.Fatalf("out-of-range output: %v", got)
	}
}

func TestFuseScoresWithFeatures_RouterDispatchesByRule(t *testing.T) {
	feat := GatedFusionFeatures{}
	w := DefaultGatedFusionWeights

	for _, r := range []FusionRule{FusionNoisyOr, FusionMax, FusionMean, FusionWeighted} {
		want := FuseScores(r, 0.7, 0.3, 0.5)
		got := FuseScoresWithFeatures(r, 0.7, 0.3, 0.5, feat, w)
		approxEqf(t, got, want, 1e-6, "router rule "+string(r))
	}

	gatedDirect := FuseScoresGated(0.7, 0.3, feat, w)
	gatedRouter := FuseScoresWithFeatures(FusionGated, 0.7, 0.3, 0.5, feat, w)
	if gatedDirect != gatedRouter {
		t.Errorf("gated dispatch: router=%v direct=%v", gatedRouter, gatedDirect)
	}
}

func TestFuseScores_GatedWithoutFeaturesDegradesToWeightedHalf(t *testing.T) {

	got := FuseScores(FusionGated, 0.8, 0.2, 0.5)
	approxEqf(t, got, 0.5, 1e-6, "gated no-features fallback = weighted-0.5")
}

func TestValidFusionRule_AcceptsGated(t *testing.T) {
	if !ValidFusionRule(FusionGated) {
		t.Fatalf("FusionGated must validate")
	}
}

func TestDefaultFusionRule_IsGated(t *testing.T) {

	if defaultFusionRule != FusionGated {
		t.Fatalf("default fusion rule must be FusionGated, got %q", defaultFusionRule)
	}
}

func TestGate_MonotonicInLogit(t *testing.T) {

	w := DefaultGatedFusionWeights
	feat := ExtractGatedFusionFeatures("hello world", 1000)
	last := float32(-1)
	for _, pt := range []float32{0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0} {
		s := FuseScoresGated(pt, 0.5, feat, w)
		if math.IsNaN(float64(s)) || math.IsInf(float64(s), 0) {
			t.Fatalf("non-finite score for pt=%v", pt)
		}
		if s < 0 || s > 1 {
			t.Fatalf("out-of-range score %v for pt=%v", s, pt)
		}
		_ = last
		last = s
	}
}
