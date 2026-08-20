package eou

import (
	"math"
	"os"
	"testing"
)

func TestExtractImEndProb_Basic(t *testing.T) {
	logits := []float32{1.0, 2.0, 3.0, 0.5}
	shape := []int64{1, 1, 4}
	p, err := extractImEndProb(logits, shape, 2)
	if err != nil {
		t.Fatal(err)
	}
	if p < 0.6 || p > 0.7 {
		t.Fatalf("highest logit (idx=2) must dominate softmax; got %f", p)
	}
}

func TestExtractImEndProb_LowestLogit(t *testing.T) {
	logits := []float32{5.0, 5.0, -5.0}
	shape := []int64{1, 1, 3}
	p, err := extractImEndProb(logits, shape, 2)
	if err != nil {
		t.Fatal(err)
	}
	if p > 0.001 {
		t.Fatalf("very negative logit must give near-zero p; got %f", p)
	}
}

func TestExtractImEndProb_LastTokenRow(t *testing.T) {
	vocab := 4
	seq := 3
	logits := make([]float32, seq*vocab)
	logits[seq*vocab-vocab+1] = 10.0
	shape := []int64{1, int64(seq), int64(vocab)}
	p, err := extractImEndProb(logits, shape, 1)
	if err != nil {
		t.Fatal(err)
	}
	if p < 0.99 {
		t.Fatalf("last-row logit dominates; got %f", p)
	}
}

func TestExtractImEndProb_OutOfVocab(t *testing.T) {
	logits := []float32{1.0, 2.0, 3.0}
	shape := []int64{1, 1, 3}
	if _, err := extractImEndProb(logits, shape, 99); err == nil {
		t.Fatalf("imEndID out of vocab must error")
	}
}

func TestExtractImEndProb_EmptyShape(t *testing.T) {
	if _, err := extractImEndProb(nil, nil, 0); err == nil {
		t.Fatalf("empty must error")
	}
}

func TestExtractImEndProb_SoftmaxSumsToOne(t *testing.T) {
	logits := []float32{1.0, 2.0, 3.0, 4.0}
	shape := []int64{1, 1, 4}
	var total float64
	for i := 0; i < 4; i++ {
		p, _ := extractImEndProb(logits, shape, i)
		total += float64(p)
	}
	if math.Abs(total-1.0) > 1e-5 {
		t.Fatalf("softmax must sum to 1; got %f", total)
	}
}

func TestNewONNXModel_RejectsMissingModel(t *testing.T) {
	if _, err := NewONNXModel("/nonexistent.onnx", ONNXOptions{}); err == nil {
		t.Fatalf("missing model must error")
	}
}

func TestNewONNXModel_RejectsMissingTokenizer(t *testing.T) {
	dir := t.TempDir()
	dummyModel := dir + "/m.onnx"
	if err := writeFile(dummyModel, []byte("not an onnx file")); err != nil {
		t.Fatal(err)
	}
	if _, err := NewONNXModel(dummyModel, ONNXOptions{}); err == nil {
		t.Fatalf("missing tokenizer must error")
	}
}

func writeFile(path string, content []byte) error {
	return os.WriteFile(path, content, 0644)
}
