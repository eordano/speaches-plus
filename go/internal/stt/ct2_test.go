package stt

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestCT2_Transcribe_TurboFromCache(t *testing.T) {
	hf := os.Getenv("HF_HUB_CACHE")
	if hf == "" {
		home, _ := os.UserHomeDir()
		hf = filepath.Join(home, ".cache", "huggingface", "hub")
	}
	matches, _ := filepath.Glob(hf + "/models--deepdml--faster-whisper-large-v3-turbo-ct2/snapshots/*/model.bin")
	if len(matches) == 0 {
		t.Skip("Distil-Whisper-Large-v3-Turbo CT2 model not in HF cache")
	}
	modelDir := filepath.Dir(matches[0])

	audio, err := loadFloat32Bin("/tmp/audio16k_ref.bin")
	if err != nil {
		t.Skipf("no reference audio: %v", err)
	}

	tr, err := NewCT2(CT2Config{ModelDir: modelDir, Device: "cpu", Language: "en"})
	if err != nil {
		t.Fatalf("NewCT2: %v", err)
	}
	defer tr.Close()

	text, err := tr.Transcribe(audio, 16000)
	if err != nil {
		t.Fatalf("Transcribe: %v", err)
	}
	t.Logf("CT2 transcript: %q", text)
	low := strings.ToLower(text)
	if !strings.Contains(low, "quick brown fox") && !strings.Contains(low, "lazy dog") {
		t.Fatalf("transcript doesn't contain expected substring: %q", text)
	}
}
