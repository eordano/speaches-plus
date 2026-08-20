package pii

import (
	"reflect"
	"testing"
)

func TestAssembleSpans_Empty(t *testing.T) {
	result := AssembleSpans(nil, nil, nil)
	if len(result) != 0 {
		t.Fatalf("expected empty, got %v", result)
	}
}

func TestAssembleSpans_AllO(t *testing.T) {
	labels := []string{"O", "O", "O"}
	offsets := [][2]int{{0, 1}, {1, 2}, {2, 3}}
	mask := []int{1, 1, 1}
	result := AssembleSpans(labels, offsets, mask)
	if len(result) != 0 {
		t.Fatalf("expected empty spans for all-O, got %v", result)
	}
}

func TestAssembleSpans_SingleS(t *testing.T) {
	labels := []string{"O", "S-email", "O"}
	offsets := [][2]int{{0, 2}, {2, 10}, {10, 12}}
	mask := []int{1, 1, 1}
	result := AssembleSpans(labels, offsets, mask)
	expected := []PiiSpan{{Start: 2, EndExclusive: 10, Label: "email"}}
	if !reflect.DeepEqual(result, expected) {
		t.Fatalf("expected %v, got %v", expected, result)
	}
}

func TestAssembleSpans_BIE(t *testing.T) {
	labels := []string{"O", "B-phone", "I-phone", "E-phone", "O"}
	offsets := [][2]int{{0, 1}, {2, 5}, {5, 8}, {8, 12}, {12, 13}}
	mask := []int{1, 1, 1, 1, 1}
	result := AssembleSpans(labels, offsets, mask)
	expected := []PiiSpan{{Start: 2, EndExclusive: 12, Label: "phone"}}
	if !reflect.DeepEqual(result, expected) {
		t.Fatalf("expected %v, got %v", expected, result)
	}
}

func TestAssembleSpans_BE(t *testing.T) {
	labels := []string{"B-name", "E-name"}
	offsets := [][2]int{{0, 3}, {4, 8}}
	mask := []int{1, 1}
	result := AssembleSpans(labels, offsets, mask)
	expected := []PiiSpan{{Start: 0, EndExclusive: 8, Label: "name"}}
	if !reflect.DeepEqual(result, expected) {
		t.Fatalf("expected %v, got %v", expected, result)
	}
}

func TestAssembleSpans_SkipsZeroMask(t *testing.T) {
	labels := []string{"O", "S-email", "O"}
	offsets := [][2]int{{0, 0}, {2, 10}, {10, 12}}
	mask := []int{1, 0, 1}
	result := AssembleSpans(labels, offsets, mask)
	if len(result) != 0 {
		t.Fatalf("expected empty (masked token), got %v", result)
	}
}

func TestAssembleSpans_SkipsZeroLenOffset(t *testing.T) {
	labels := []string{"S-email", "S-name"}
	offsets := [][2]int{{0, 0}, {2, 5}}
	mask := []int{1, 1}
	result := AssembleSpans(labels, offsets, mask)
	expected := []PiiSpan{{Start: 2, EndExclusive: 5, Label: "name"}}
	if !reflect.DeepEqual(result, expected) {
		t.Fatalf("expected %v, got %v", expected, result)
	}
}

func TestAssembleSpans_MultipleBIESpans(t *testing.T) {
	labels := []string{"B-email", "I-email", "E-email", "O", "S-phone"}
	offsets := [][2]int{{0, 4}, {4, 8}, {8, 12}, {12, 13}, {14, 20}}
	mask := []int{1, 1, 1, 1, 1}
	result := AssembleSpans(labels, offsets, mask)
	expected := []PiiSpan{
		{Start: 0, EndExclusive: 12, Label: "email"},
		{Start: 14, EndExclusive: 20, Label: "phone"},
	}
	if !reflect.DeepEqual(result, expected) {
		t.Fatalf("expected %v, got %v", expected, result)
	}
}

func TestAssembleSpans_MismatchedEClosesAndEmits(t *testing.T) {
	labels := []string{"B-email", "E-phone"}
	offsets := [][2]int{{0, 5}, {5, 10}}
	mask := []int{1, 1}
	result := AssembleSpans(labels, offsets, mask)
	expected := []PiiSpan{
		{Start: 0, EndExclusive: 5, Label: "email"},
		{Start: 5, EndExclusive: 10, Label: "phone"},
	}
	if !reflect.DeepEqual(result, expected) {
		t.Fatalf("expected %v, got %v", expected, result)
	}
}
