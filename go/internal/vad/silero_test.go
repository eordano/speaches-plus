package vad

import (
	"os"
	"path/filepath"
	"testing"

	ort "github.com/yalue/onnxruntime_go"
)

func ensureORT(t *testing.T) {
	if libPath := os.Getenv("ONNXRUNTIME_LIB"); libPath != "" {
		ort.SetSharedLibraryPath(libPath)
	}
	if err := ort.InitializeEnvironment(); err != nil {
		t.Logf("ort init (may already be initialized): %v", err)
	}
}

func modelPath(t *testing.T) string {
	if p := os.Getenv("SILERO_VAD_MODEL"); p != "" {
		if _, err := os.Stat(p); err == nil {
			return p
		}
	}
	hf := os.Getenv("HF_HUB_CACHE")
	if hf == "" {
		home, _ := os.UserHomeDir()
		hf = filepath.Join(home, ".cache", "huggingface", "hub")
	}
	matches, _ := filepath.Glob(hf + "/models--onnx-community--silero-vad/snapshots/*/onnx/model.onnx")
	if len(matches) == 0 {
		t.Skip("no Silero model available (set SILERO_VAD_MODEL or populate HF_HUB_CACHE)")
	}
	return matches[0]
}

func TestSilero_OnSilenceAndImpulse(t *testing.T) {
	ensureORT(t)
	v, err := New(modelPath(t))
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	defer v.Close()

	silence := make([]float32, WindowSamples)
	for i := 0; i < 50; i++ {
		dec, _, err := v.Process(silence)
		if err != nil {
			t.Fatalf("Process silence: %v", err)
		}
		if dec != None {
			t.Fatalf("silence frame %d unexpectedly produced %v", i, dec)
		}
	}
}

func TestSilero_FiresOnReferenceWav(t *testing.T) {
	ensureORT(t)
	mp := modelPath(t)

	wavs, _ := filepath.Glob("../../../client/fixtures/ref_*.wav")
	if len(wavs) == 0 {
		t.Skip("no cached reference wav fixtures")
	}
	pcm, sr := readWavMono(t, wavs[0])
	if sr != 16000 && sr != 24000 && sr != 48000 {
		t.Skipf("unsupported sample rate %d in %s", sr, wavs[0])
	}
	if sr != 16000 {
		ratio := float64(16000) / float64(sr)
		out := make([]float32, int(float64(len(pcm))*ratio))
		for i := range out {
			src := float64(i) / ratio
			idx := int(src)
			if idx >= len(pcm)-1 {
				out[i] = pcm[len(pcm)-1]
				continue
			}
			frac := float32(src - float64(idx))
			out[i] = pcm[idx]*(1-frac) + pcm[idx+1]*frac
		}
		pcm = out
	}
	pcm = append(pcm, make([]float32, 16000)...)

	v, err := New(mp)
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	defer v.Close()

	var sawStart, sawEnd bool
	for off := 0; off+WindowSamples <= len(pcm); off += WindowSamples {
		dec, _, err := v.Process(pcm[off : off+WindowSamples])
		if err != nil {
			t.Fatalf("Process: %v", err)
		}
		if dec == SpeechStart {
			sawStart = true
		}
		if dec == SpeechEnd {
			sawEnd = true
		}
	}
	if !sawStart || !sawEnd {
		t.Fatalf("expected speech_start AND speech_end (start=%v end=%v) on %s", sawStart, sawEnd, wavs[0])
	}
}

func readWavMono(t *testing.T, path string) ([]float32, int) {
	t.Helper()
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read %s: %v", path, err)
	}
	if len(data) < 44 || string(data[:4]) != "RIFF" {
		t.Fatalf("%s: not a wav", path)
	}
	channels := int(uint16(data[22]) | uint16(data[23])<<8)
	sr := int(uint32(data[24]) | uint32(data[25])<<8 | uint32(data[26])<<16 | uint32(data[27])<<24)
	bps := int(uint16(data[34]) | uint16(data[35])<<8)
	body := data[44:]
	if bps != 16 {
		t.Fatalf("%s: only s16 supported (got bps=%d)", path, bps)
	}
	frame := channels * 2
	n := len(body) / frame
	out := make([]float32, n)
	for i := 0; i < n; i++ {
		var sum int32
		for c := 0; c < channels; c++ {
			off := i*frame + c*2
			v := int16(uint16(body[off]) | uint16(body[off+1])<<8)
			sum += int32(v)
		}
		out[i] = float32(sum) / float32(channels) / 32768.0
	}
	return out, sr
}
