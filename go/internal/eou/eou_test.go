package eou

import (
	"context"
	"testing"
)

func TestSigmoidLerp_Boundaries(t *testing.T) {
	if d := SigmoidLerp(1.0, 0.5, 500, 3000); d != 500 {
		t.Fatalf("score=1 -> min: got %d", d)
	}
	if d := SigmoidLerp(0.0, 0.5, 500, 3000); d != 3000 {
		t.Fatalf("score below threshold -> max: got %d", d)
	}
	d := SigmoidLerp(0.75, 0.5, 500, 3000)
	if d < 1500 || d > 2000 {
		t.Fatalf("score=0.75 (mid of [t,1]) should map near 1750: got %d", d)
	}
}

func TestSigmoidLerp_BelowThreshold(t *testing.T) {
	if d := SigmoidLerp(0.4, 0.5, 500, 3000); d != 3000 {
		t.Fatalf("score<threshold must give maxMs: got %d", d)
	}
	if d := SigmoidLerp(0.5, 0.5, 500, 3000); d != 3000 {
		t.Fatalf("score=threshold rounds to maxMs (verdict barely 'done'): got %d", d)
	}
}

func TestSigmoidLerp_Clamps(t *testing.T) {
	if d := SigmoidLerp(2.0, 0.5, 500, 3000); d != 500 {
		t.Fatalf("score>1 must clamp to min: got %d", d)
	}
	if d := SigmoidLerp(-0.5, 0.5, 500, 3000); d != 3000 {
		t.Fatalf("score<0 must give max: got %d", d)
	}
	if d := SigmoidLerp(1.0, 0.5, 1000, 500); d == 0 {
		t.Fatalf("inverted bounds must not give 0")
	}
}

func TestSigmoidLerpK_DefaultMatchesSigmoidLerp(t *testing.T) {
	a := SigmoidLerpK(0.8, 0.5, 500, 3000, DefaultCurveK)
	b := SigmoidLerp(0.8, 0.5, 500, 3000)
	if a != b {
		t.Fatalf("SigmoidLerpK(k=12) must equal SigmoidLerp: %d vs %d", a, b)
	}
}

func TestSigmoidLerpK_HigherKIsSharper(t *testing.T) {
	flat := SigmoidLerpK(0.95, 0.5, 500, 3000, 4)
	sharp := SigmoidLerpK(0.95, 0.5, 500, 3000, 24)
	if sharp >= flat {
		t.Fatalf("higher k must drive a high-confidence delay closer to min: flat=%d sharp=%d", flat, sharp)
	}
}

func TestHeuristic_StrongTerminator(t *testing.T) {
	h := NewHeuristic()
	cases := []string{
		"That's all I needed.",
		"Are you there?",
		"Wow!",
	}
	for _, c := range cases {
		v, err := h.Predict(context.Background(), Request{Partial: c, Language: "en"})
		if err != nil {
			t.Fatalf("predict %q: %v", c, err)
		}
		if v.Score < 0.9 {
			t.Fatalf("strong terminator %q: score=%f (want >=0.9)", c, v.Score)
		}
	}
}

func TestHeuristic_Hesitation(t *testing.T) {
	h := NewHeuristic()
	cases := []string{
		"I think I'd like, um",
		"Wait, uh",
		"Well",
	}
	for _, c := range cases {
		v, err := h.Predict(context.Background(), Request{Partial: c, Language: "en"})
		if err != nil {
			t.Fatalf("predict %q: %v", c, err)
		}
		if v.Score > 0.3 {
			t.Fatalf("hesitation %q: score=%f (want <=0.3)", c, v.Score)
		}
	}
}

func TestHeuristic_Continuation(t *testing.T) {
	h := NewHeuristic()
	cases := []string{
		"I want to talk about the",
		"Maybe we could go and",
		"This is a problem because",
	}
	for _, c := range cases {
		v, err := h.Predict(context.Background(), Request{Partial: c, Language: "en"})
		if err != nil {
			t.Fatalf("predict %q: %v", c, err)
		}
		if v.Score > 0.4 {
			t.Fatalf("continuation %q: score=%f (want <=0.4)", c, v.Score)
		}
	}
}

func TestHeuristic_SoftTerminator(t *testing.T) {
	h := NewHeuristic()
	v, err := h.Predict(context.Background(), Request{Partial: "First, this is important,", Language: "en"})
	if err != nil {
		t.Fatal(err)
	}
	if v.Score != heuristicScoreSoftTerminator {
		t.Fatalf("soft terminator: score=%f (want %f)", v.Score, heuristicScoreSoftTerminator)
	}
}

func TestHeuristic_LangFallback(t *testing.T) {
	h := NewHeuristic()
	v, _ := h.Predict(context.Background(), Request{Partial: "I would like to and", Language: "ja"})
	if v.Score > 0.4 {
		t.Fatalf("unknown lang must fall back to en table; score=%f", v.Score)
	}
}

func TestHeuristic_Empty(t *testing.T) {
	h := NewHeuristic()
	v, err := h.Predict(context.Background(), Request{Partial: "", Language: "en"})
	if err != nil {
		t.Fatal(err)
	}
	if v.Score != heuristicScoreEmpty {
		t.Fatalf("empty transcript: score=%f (want %f)", v.Score, heuristicScoreEmpty)
	}
}

func TestHeuristic_SpanishHesitation(t *testing.T) {
	h := NewHeuristic()
	v, _ := h.Predict(context.Background(), Request{Partial: "Quería decir, eh", Language: "es"})
	if v.Score > 0.3 {
		t.Fatalf("spanish hesitation: score=%f", v.Score)
	}
}

func TestLanguages_Default(t *testing.T) {
	tbl := DefaultLanguages()
	if tbl.Threshold("en") != 0.55 {
		t.Fatalf("en default: %f", tbl.Threshold("en"))
	}
	if tbl.Threshold("xx") != 0.55 {
		t.Fatalf("unknown lang must fall back to en threshold")
	}
	if tbl.Threshold("ja") != 0.45 {
		t.Fatalf("ja threshold: %f", tbl.Threshold("ja"))
	}
}

func TestLoad_HeuristicWhenNoModelPath(t *testing.T) {
	m, cfg, err := Load(Config{})
	if err != nil {
		t.Fatal(err)
	}
	if m == nil {
		t.Fatal("Load must return a non-nil model")
	}
	if cfg.MinDelayMs != 500 || cfg.MaxDelayMs != 3000 {
		t.Fatalf("defaults: min=%d max=%d", cfg.MinDelayMs, cfg.MaxDelayMs)
	}
	if cfg.Languages == nil {
		t.Fatalf("languages must be populated")
	}
	if _, ok := m.(*Heuristic); !ok {
		t.Fatalf("Load with empty path must return Heuristic, got %T", m)
	}
}

func TestLoad_FallsBackWhenModelMissing(t *testing.T) {
	m, _, err := Load(Config{ModelPath: "/nonexistent/path/model.onnx"})
	if err != nil {
		t.Fatal(err)
	}
	if _, ok := m.(*Heuristic); !ok {
		t.Fatalf("Load with missing path must fall back to Heuristic, got %T", m)
	}
}
