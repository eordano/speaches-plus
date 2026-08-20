package stt

import (
	"strings"
	"testing"
)

func TestChunkAudio_Short(t *testing.T) {
	samples := make([]float32, 16000*10)
	chunks := chunkAudio(samples, 16000)
	if len(chunks) != 1 {
		t.Fatalf("want 1 chunk for 10s, got %d", len(chunks))
	}
	if chunks[0].offsetMs != 0 {
		t.Errorf("offset = %d, want 0", chunks[0].offsetMs)
	}
}

func TestChunkAudio_Exact30(t *testing.T) {
	samples := make([]float32, 16000*30)
	chunks := chunkAudio(samples, 16000)
	if len(chunks) != 1 {
		t.Fatalf("want 1 chunk for exactly 30s, got %d", len(chunks))
	}
}

func TestChunkAudio_Long(t *testing.T) {
	samples := make([]float32, 16000*75)
	chunks := chunkAudio(samples, 16000)
	if len(chunks) != 3 {
		t.Fatalf("want 3 chunks for 75s, got %d", len(chunks))
	}
	if chunks[0].offsetMs != 0 {
		t.Errorf("chunk0 offset = %d, want 0", chunks[0].offsetMs)
	}
	if chunks[1].offsetMs != 30000 {
		t.Errorf("chunk1 offset = %d, want 30000", chunks[1].offsetMs)
	}
	if chunks[2].offsetMs != 60000 {
		t.Errorf("chunk2 offset = %d, want 60000", chunks[2].offsetMs)
	}
	if len(chunks[2].data) != 16000*15 {
		t.Errorf("chunk2 len = %d, want %d", len(chunks[2].data), 16000*15)
	}
}

func TestShiftSegments(t *testing.T) {
	lp := float32(-0.5)
	segs := []Segment{
		{TStartMs: 100, TEndMs: 500, Text: "hello", AvgLogprob: &lp},
		{TStartMs: 500, TEndMs: 1000, Text: "world"},
	}
	shifted := shiftSegments(segs, 30000)
	if shifted[0].TStartMs != 30100 || shifted[0].TEndMs != 30500 {
		t.Errorf("seg0 = %d-%d, want 30100-30500", shifted[0].TStartMs, shifted[0].TEndMs)
	}
	if shifted[1].TStartMs != 30500 || shifted[1].TEndMs != 31000 {
		t.Errorf("seg1 = %d-%d, want 30500-31000", shifted[1].TStartMs, shifted[1].TEndMs)
	}
	if shifted[0].AvgLogprob == nil || *shifted[0].AvgLogprob != -0.5 {
		t.Errorf("seg0 logprob lost")
	}
}

type fakeTranscriber struct {
	calls int
}

func (f *fakeTranscriber) Transcribe(samples []float32, sampleRate int) (string, error) {
	f.calls++
	return "chunk", nil
}

func (f *fakeTranscriber) Close() error { return nil }

func TestTranscribeLong_ChunksAndJoins(t *testing.T) {
	ft := &fakeTranscriber{}
	samples := make([]float32, 16000*75)
	for i := range samples {
		samples[i] = 0.1
	}
	text, err := TranscribeLong(ft, samples, 16000)
	if err != nil {
		t.Fatal(err)
	}
	if ft.calls != 3 {
		t.Errorf("want 3 calls, got %d", ft.calls)
	}
	if text != "chunk chunk chunk" {
		t.Errorf("text = %q", text)
	}
}

func TestTranscribeLong_ShortPassthrough(t *testing.T) {
	ft := &fakeTranscriber{}
	samples := make([]float32, 16000*5)
	for i := range samples {
		samples[i] = 0.1
	}
	text, err := TranscribeLong(ft, samples, 16000)
	if err != nil {
		t.Fatal(err)
	}
	if ft.calls != 1 {
		t.Errorf("want 1 call for short audio, got %d", ft.calls)
	}
	if text != "chunk" {
		t.Errorf("text = %q", text)
	}
}

func TestTranscribeLong_Silence(t *testing.T) {
	ft := &fakeTranscriber{}
	samples := make([]float32, 16000*60)
	text, err := TranscribeLong(ft, samples, 16000)
	if err != nil {
		t.Fatal(err)
	}
	if ft.calls != 0 {
		t.Errorf("want 0 calls for silence, got %d", ft.calls)
	}
	if text != "" {
		t.Errorf("text = %q, want empty", text)
	}
}

func TestTranscribeLong_SkipsSilentChunks(t *testing.T) {
	ft := &fakeTranscriber{}
	samples := make([]float32, 16000*60)
	for i := 0; i < 16000*30; i++ {
		samples[i] = 0.1
	}
	text, err := TranscribeLong(ft, samples, 16000)
	if err != nil {
		t.Fatal(err)
	}
	if ft.calls != 1 {
		t.Errorf("want 1 call (second chunk silent), got %d", ft.calls)
	}
	if !strings.Contains(text, "chunk") {
		t.Errorf("text = %q", text)
	}
}

func TestPeakAmplitude(t *testing.T) {
	if got := peakAmplitude(nil); got != 0 {
		t.Errorf("nil = %f", got)
	}
	samples := []float32{-0.5, 0.3, -0.1}
	if got := peakAmplitude(samples); got != 0.5 {
		t.Errorf("got %f, want 0.5", got)
	}
}
