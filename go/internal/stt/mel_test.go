package stt

import (
	"encoding/binary"
	"math"
	"os"
	"testing"
)

func TestMelSpectrogram_MatchesHF(t *testing.T) {
	audio, err := loadFloat32Bin("/tmp/audio16k_ref.bin")
	if err != nil {
		t.Skipf("missing reference audio /tmp/audio16k_ref.bin (%v)", err)
	}
	refMel, err := loadF32Npy("/tmp/mel_features_ref.npy")
	if err != nil {
		t.Skipf("missing reference mel /tmp/mel_features_ref.npy (%v)", err)
	}

	fb := NewMelFilterbank(128, whisperNFFT, whisperSamplingHz)
	got := LogMelSpectrogram(audio, fb)

	if len(got) != len(refMel) {
		t.Fatalf("size mismatch: got %d want %d (= 128*3000)", len(got), len(refMel))
	}

	var maxAbs float64
	var sumAbs float64
	for i := range got {
		d := math.Abs(float64(got[i] - refMel[i]))
		if d > maxAbs {
			maxAbs = d
		}
		sumAbs += d
	}
	meanAbs := sumAbs / float64(len(got))
	t.Logf("max |Δ|=%.5g  mean |Δ|=%.5g  (n=%d)", maxAbs, meanAbs, len(got))

	for _, idx := range []struct{ bin, frame int }{{0, 0}, {0, 1}, {0, 2}, {64, 100}, {127, 2999}} {
		i := idx.bin*whisperNbFrames + idx.frame
		t.Logf("  bin=%d frame=%-4d  got=%+.4f  ref=%+.4f  Δ=%+.4f",
			idx.bin, idx.frame, got[i], refMel[i], got[i]-refMel[i])
	}

	if maxAbs > 0.05 {
		t.Errorf("max |Δ|=%g exceeds tolerance 0.05", maxAbs)
	}
	if meanAbs > 0.005 {
		t.Errorf("mean |Δ|=%g exceeds tolerance 0.005", meanAbs)
	}
}

func loadFloat32Bin(path string) ([]float32, error) {
	b, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	out := make([]float32, len(b)/4)
	for i := range out {
		out[i] = math.Float32frombits(binary.LittleEndian.Uint32(b[i*4 : i*4+4]))
	}
	return out, nil
}

func loadF32Npy(path string) ([]float32, error) {
	b, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	if len(b) < 10 || string(b[:6]) != "\x93NUMPY" {
		return nil, os.ErrInvalid
	}
	var headerLen, dataOff int
	switch b[6] {
	case 1:
		headerLen = int(binary.LittleEndian.Uint16(b[8:10]))
		dataOff = 10 + headerLen
	default:
		headerLen = int(binary.LittleEndian.Uint32(b[8:12]))
		dataOff = 12 + headerLen
	}
	out := make([]float32, (len(b)-dataOff)/4)
	for i := range out {
		out[i] = math.Float32frombits(binary.LittleEndian.Uint32(b[dataOff+i*4 : dataOff+i*4+4]))
	}
	return out, nil
}
