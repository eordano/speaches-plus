package diarization

import (
	"reflect"
	"testing"
)

func TestSlideChunksPadsShortUtterance(t *testing.T) {
	audio := make([]float32, 8000)
	for i := range audio {
		audio[i] = 1.0
	}
	chunks := SlideChunks(audio, 16000, 5.0, 0.1)
	if len(chunks) != 1 {
		t.Fatalf("want 1 chunk, got %d", len(chunks))
	}
	if len(chunks[0].Samples) != 80000 {
		t.Fatalf("want 80000 samples, got %d", len(chunks[0].Samples))
	}
	if chunks[0].Samples[10000] != 0.0 {
		t.Fatalf("expected zero padding past audio length, got %f", chunks[0].Samples[10000])
	}
}

func TestSlideChunksOverlappingLongUtterance(t *testing.T) {
	audio := make([]float32, 16000*11)
	chunks := SlideChunks(audio, 16000, 5.0, 0.1)
	if len(chunks) < 12 {
		t.Fatalf("want >=12 chunks for 11s @ 0.5s hop, got %d", len(chunks))
	}
	if chunks[1].TOffsetMs != 500 {
		t.Fatalf("second chunk t_offset should be 500ms, got %d", chunks[1].TOffsetMs)
	}
}

func TestMedianFilterSmoothsSingletonBlip(t *testing.T) {
	ml := &Multilabel{
		Frames:   7,
		Speakers: 1,
		Data:     []uint8{1, 1, 1, 0, 1, 1, 1},
	}
	smoothed := MedianFilterMultihot(ml, 3)
	if !reflect.DeepEqual(smoothed.Row(3), []uint8{1}) {
		t.Fatalf("singleton blip should be filtered: %v", smoothed.Row(3))
	}
}

func TestCoalesceMergesAdjacentSameSpeaker(t *testing.T) {
	segs := []Segment{
		{Speaker: 0, TStartMs: 0, TEndMs: 500, Confidence: 0.9},
		{Speaker: 0, TStartMs: 600, TEndMs: 1000, Confidence: 0.85},
		{Speaker: 1, TStartMs: 1100, TEndMs: 1500, Confidence: 0.8},
	}
	merged := CoalesceSegments(segs)
	if len(merged) != 2 {
		t.Fatalf("want 2 merged segments, got %d", len(merged))
	}
	if merged[0].TEndMs != 1000 {
		t.Fatalf("first merged segment should end at 1000ms, got %d", merged[0].TEndMs)
	}
}
