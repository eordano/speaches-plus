package diarization

import (
	"math"
	"reflect"
	"sort"
	"testing"
)

func TestDiariZenV2Topology(t *testing.T) {
	dec := NewPowersetDecoder(4, 2)
	if dec.NumClasses() != 11 {
		t.Fatalf("want 11 classes, got %d", dec.NumClasses())
	}
	if !reflect.DeepEqual(dec.mapping[0], []int{}) {
		t.Fatalf("class 0 should be silence (empty); got %v", dec.mapping[0])
	}
	if !reflect.DeepEqual(dec.mapping[1], []int{0}) {
		t.Fatalf("class 1 should be {0}; got %v", dec.mapping[1])
	}
	if !reflect.DeepEqual(dec.mapping[4], []int{3}) {
		t.Fatalf("class 4 should be {3}; got %v", dec.mapping[4])
	}
	if !reflect.DeepEqual(dec.mapping[5], []int{0, 1}) {
		t.Fatalf("class 5 should be {0,1}; got %v", dec.mapping[5])
	}
	if !reflect.DeepEqual(dec.mapping[10], []int{2, 3}) {
		t.Fatalf("class 10 should be {2,3}; got %v", dec.mapping[10])
	}
}

func TestPyannote3SpkTopology(t *testing.T) {
	dec := NewPowersetDecoder(3, 2)
	if dec.NumClasses() != 7 {
		t.Fatalf("want 7 classes for K=3 M=2, got %d", dec.NumClasses())
	}
}

func TestClassIndicesAreUniqueAndSorted(t *testing.T) {
	dec := NewPowersetDecoder(4, 2)
	seen := make(map[string]struct{})
	for _, combo := range dec.mapping {

		copyc := make([]int, len(combo))
		copy(copyc, combo)
		sort.Ints(copyc)
		if !reflect.DeepEqual(combo, copyc) {
			t.Fatalf("class indices must be sorted: %v", combo)
		}
		key := ""
		for _, v := range combo {
			key += string(rune(v + '0'))
		}
		if _, dup := seen[key]; dup {
			t.Fatalf("duplicate class %v", combo)
		}
		seen[key] = struct{}{}
	}
}

func TestArgmaxPicksSilence(t *testing.T) {
	dec := NewPowersetDecoder(4, 2)
	row := make([]float32, 11)
	for i := range row {
		row[i] = -10
	}
	row[0] = 0
	logits := &SegmentationLogits{Frames: 1, Classes: 11, Data: row}
	ml := dec.ToMultilabelHard(logits)
	if !reflect.DeepEqual(ml.Row(0), []uint8{0, 0, 0, 0}) {
		t.Fatalf("silence row mismatch: %v", ml.Row(0))
	}
}

func TestArgmaxPicksOverlap(t *testing.T) {
	dec := NewPowersetDecoder(4, 2)
	row := make([]float32, 11)
	for i := range row {
		row[i] = -10
	}
	row[5] = 0
	logits := &SegmentationLogits{Frames: 1, Classes: 11, Data: row}
	ml := dec.ToMultilabelHard(logits)
	if !reflect.DeepEqual(ml.Row(0), []uint8{1, 1, 0, 0}) {
		t.Fatalf("overlap row mismatch: %v", ml.Row(0))
	}
}

func TestCosineSimL2Identity(t *testing.T) {
	a := unit([]float32{1, 2, 3})
	if math.Abs(float64(CosineSim(a, a)-1.0)) > 1e-5 {
		t.Fatalf("self cosine should be ~1, got %f", CosineSim(a, a))
	}
}
