package tts

import (
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/eordano/speaches-plus-go/internal/stt"
)

func TestKokoro_RoundTripWhisper(t *testing.T) {
	hf := os.Getenv("HF_HUB_CACHE")
	if hf == "" {
		home, _ := os.UserHomeDir()
		hf = filepath.Join(home, ".cache", "huggingface", "hub")
	}
	mPath, _ := filepath.Glob(hf + "/models--speaches-ai--Kokoro-82M-v1.0-ONNX/snapshots/*/model.onnx")
	vPath, _ := filepath.Glob(hf + "/models--speaches-ai--Kokoro-82M-v1.0-ONNX/snapshots/*/voices.bin")
	if len(mPath) == 0 || len(vPath) == 0 {
		t.Skip("Kokoro model files missing")
	}
	ct2Match, _ := filepath.Glob(hf + "/models--deepdml--faster-whisper-large-v3-turbo-ct2/snapshots/*/model.bin")
	if len(ct2Match) == 0 {
		t.Skip("CT2 turbo model missing")
	}
	ct2Dir := filepath.Dir(ct2Match[0])

	syn, err := NewKokoro(KokoroConfig{
		ModelPath:  mPath[0],
		VoicesPath: vPath[0],
		EspeakData: os.Getenv("ESPEAK_DATA_PATH"),
	})
	if err != nil {
		t.Fatalf("NewKokoro: %v", err)
	}
	defer syn.Close()

	whisp, err := stt.NewCT2(stt.CT2Config{ModelDir: ct2Dir, Device: "cpu", Language: "en"})
	if err != nil {
		t.Fatalf("NewCT2: %v", err)
	}
	defer whisp.Close()

	for _, phrase := range []string{"acknowledged", "the quick brown fox"} {
		audio, err := syn.Synthesize(phrase, "af_heart", "en-us", 1.0)
		if err != nil {
			t.Fatalf("Synthesize(%q): %v", phrase, err)
		}
		mono16k := linearResample(audio.Samples, audio.SampleRate, 16000)
		text, err := whisp.Transcribe(mono16k, 16000)
		if err != nil {
			t.Fatalf("Transcribe(%q): %v", phrase, err)
		}
		t.Logf("%-30s -> %q", phrase, text)
		if !strings.Contains(strings.ToLower(text), strings.Split(phrase, " ")[0]) {
			t.Errorf("transcription %q does not contain head of %q", text, phrase)
		}
	}
}

func linearResample(in []float32, srIn, srOut int) []float32 {
	if srIn == srOut || len(in) == 0 {
		out := make([]float32, len(in))
		copy(out, in)
		return out
	}
	nIn := len(in)
	nOut := nIn * srOut / srIn
	if nOut <= 1 {
		return in
	}
	out := make([]float32, nOut)
	step := float64(nIn-1) / float64(nOut-1)
	for i := 0; i < nOut; i++ {
		x := float64(i) * step
		idx := int(x)
		if idx >= nIn-1 {
			out[i] = in[nIn-1]
			continue
		}
		frac := float32(x - float64(idx))
		out[i] = in[idx]*(1-frac) + in[idx+1]*frac
	}
	return out
}
