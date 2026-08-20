package pii

import (
	"math"
	"testing"
)

func TestViterbiDecode_Empty(t *testing.T) {
	result := ViterbiDecode(nil, []string{"O", "B-x", "E-x"})
	if len(result) != 0 {
		t.Fatalf("expected empty, got %v", result)
	}
}

func TestViterbiDecode_SingleO(t *testing.T) {
	labels := []string{"O", "B-name", "I-name", "E-name", "S-name"}
	logits := [][]float32{{10.0, -5.0, -5.0, -5.0, -5.0}}
	result := ViterbiDecode(logits, labels)
	if len(result) != 1 {
		t.Fatalf("expected 1 result, got %d", len(result))
	}
	if result[0] != 0 {
		t.Fatalf("expected O (idx 0), got %d", result[0])
	}
}

func TestViterbiDecode_ForcesValidTransitions(t *testing.T) {
	labels := []string{"O", "B-name", "I-name", "E-name", "S-name"}
	logits := [][]float32{
		{-5.0, 10.0, -5.0, -5.0, -5.0},
		{-5.0, -5.0, -5.0, 10.0, -5.0},
		{10.0, -5.0, -5.0, -5.0, -5.0},
	}
	result := ViterbiDecode(logits, labels)
	if len(result) != 3 {
		t.Fatalf("expected 3 results, got %d", len(result))
	}
	if labels[result[0]] != "B-name" {
		t.Errorf("step 0: expected B-name, got %s", labels[result[0]])
	}
	if labels[result[1]] != "E-name" {
		t.Errorf("step 1: expected E-name, got %s", labels[result[1]])
	}
	if labels[result[2]] != "O" {
		t.Errorf("step 2: expected O, got %s", labels[result[2]])
	}
}

func TestViterbiDecode_CannotStartWithIorE(t *testing.T) {
	labels := []string{"O", "B-x", "I-x", "E-x", "S-x"}
	logits := [][]float32{
		{-5.0, -5.0, 100.0, -5.0, -5.0},
	}
	result := ViterbiDecode(logits, labels)
	p := labels[result[0]]
	tg := splitLabel(p)
	if tg.prefix == "I" || tg.prefix == "E" {
		t.Errorf("should not start with I or E, got %s", p)
	}
}

func TestViterbiDecode_BIESequence(t *testing.T) {
	labels := []string{"O", "B-email", "I-email", "E-email", "S-email"}
	logits := [][]float32{
		{-10, 10, -10, -10, -10},
		{-10, -10, 10, -10, -10},
		{-10, -10, 10, -10, -10},
		{-10, -10, -10, 10, -10},
		{10, -10, -10, -10, -10},
	}
	result := ViterbiDecode(logits, labels)
	expected := []string{"B-email", "I-email", "I-email", "E-email", "O"}
	for i, idx := range result {
		if labels[idx] != expected[i] {
			t.Errorf("step %d: expected %s, got %s", i, expected[i], labels[idx])
		}
	}
}

func TestLogSoftmax(t *testing.T) {
	row := []float32{1.0, 2.0, 3.0}
	result := logSoftmax(row)
	var sum float64
	for _, v := range result {
		sum += math.Exp(v)
	}
	if math.Abs(sum-1.0) > 1e-6 {
		t.Errorf("softmax should sum to 1, got %f", sum)
	}
	if result[2] <= result[1] || result[1] <= result[0] {
		t.Error("softmax should preserve ordering")
	}
}
