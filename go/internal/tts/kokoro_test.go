package tts

import (
	"os"
	"path/filepath"
	"testing"
)

func TestKokoro_Synthesize_Acknowledged(t *testing.T) {
	hf := os.Getenv("HF_HUB_CACHE")
	if hf == "" {
		home, _ := os.UserHomeDir()
		hf = filepath.Join(home, ".cache", "huggingface", "hub")
	}
	mPath, _ := filepath.Glob(hf + "/models--speaches-ai--Kokoro-82M-v1.0-ONNX/snapshots/*/model.onnx")
	vPath, _ := filepath.Glob(hf + "/models--speaches-ai--Kokoro-82M-v1.0-ONNX/snapshots/*/voices.bin")
	if len(mPath) == 0 || len(vPath) == 0 {
		t.Skip("Kokoro model files not present in HF cache")
	}
	espeak := os.Getenv("ESPEAK_DATA_PATH")

	syn, err := NewKokoro(KokoroConfig{
		ModelPath:  mPath[0],
		VoicesPath: vPath[0],
		EspeakData: espeak,
	})
	if err != nil {
		t.Fatalf("NewKokoro: %v", err)
	}
	defer syn.Close()

	audio, err := syn.Synthesize("acknowledged", "af_heart", "en-us", 1.0)
	if err != nil {
		t.Fatalf("Synthesize: %v", err)
	}
	if audio.SampleRate != 24000 {
		t.Fatalf("sr: got %d want 24000", audio.SampleRate)
	}
	if len(audio.Samples) < 8000 {
		t.Fatalf("audio too short: %d samples (~%dms)",
			len(audio.Samples), len(audio.Samples)*1000/24000)
	}
	t.Logf("kokoro produced %d samples (%.2fs at 24kHz)",
		len(audio.Samples), float64(len(audio.Samples))/24000.0)
}
