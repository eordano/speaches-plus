package tts

import (
	"os"
	"path/filepath"
	"testing"
)

func TestLoadVoicesNPZ_Kokoro(t *testing.T) {
	hf := os.Getenv("HF_HUB_CACHE")
	if hf == "" {
		home, _ := os.UserHomeDir()
		hf = filepath.Join(home, ".cache", "huggingface", "hub")
	}
	matches, _ := filepath.Glob(hf + "/models--speaches-ai--Kokoro-82M-v1.0-ONNX/snapshots/*/voices.bin")
	if len(matches) == 0 {
		t.Skip("Kokoro voices.bin not found in HF cache; skipping")
	}
	voices, err := LoadVoicesNPZ(matches[0])
	if err != nil {
		t.Fatalf("LoadVoicesNPZ: %v", err)
	}
	v, ok := voices["af_heart"]
	if !ok {
		t.Fatalf("af_heart missing; have %d voices", len(voices))
	}
	if len(v.Shape) != 3 || v.Shape[0] != 510 || v.Shape[1] != 1 || v.Shape[2] != 256 {
		t.Fatalf("unexpected shape %v (want [510 1 256])", v.Shape)
	}
	if len(v.Data) != 510*1*256 {
		t.Fatalf("unexpected data len %d", len(v.Data))
	}
	var nonzero int
	for _, x := range v.Data {
		if x != 0 {
			nonzero++
			if nonzero > 100 {
				break
			}
		}
	}
	if nonzero == 0 {
		t.Fatalf("af_heart data is all zero")
	}
	t.Logf("loaded %d voices; af_heart shape=%v data[:5]=%v", len(voices), v.Shape, v.Data[:5])
}
