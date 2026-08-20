package tts

import (
	"fmt"
	"os"
	"sync"

	ort "github.com/yalue/onnxruntime_go"
)

type kokoro struct {
	mu      sync.Mutex
	session *ort.DynamicAdvancedSession
	voices  map[string]Voice
	dataDir string
}

func NewKokoro(cfg KokoroConfig) (Synthesizer, error) {
	if cfg.ModelPath == "" || cfg.VoicesPath == "" {
		return nil, fmt.Errorf("kokoro: model and voices paths required")
	}

	if err := initOnnxRuntime(); err != nil {
		return nil, err
	}

	if err := initPhonemizer(cfg.EspeakData); err != nil {
		return nil, fmt.Errorf("phonemizer init: %w", err)
	}

	voices, err := LoadVoicesNPZ(cfg.VoicesPath)
	if err != nil {
		return nil, fmt.Errorf("load voices: %w", err)
	}

	session, err := ort.NewDynamicAdvancedSession(
		cfg.ModelPath,
		[]string{"tokens", "style", "speed"},
		[]string{"audio"},
		nil,
	)
	if err != nil {
		return nil, fmt.Errorf("ort session: %w", err)
	}

	return &kokoro{
		session: session,
		voices:  voices,
		dataDir: cfg.EspeakData,
	}, nil
}

var (
	ortInitOnce sync.Once
	ortInitErr  error
)

func initOnnxRuntime() error {
	ortInitOnce.Do(func() {
		if libPath := os.Getenv("ONNXRUNTIME_LIB"); libPath != "" {
			ort.SetSharedLibraryPath(libPath)
		}
		ortInitErr = ort.InitializeEnvironment()
	})
	return ortInitErr
}

func (k *kokoro) Voices() []string {
	out := make([]string, 0, len(k.voices))
	for name := range k.voices {
		out = append(out, name)
	}
	return out
}

func (k *kokoro) Close() error {
	k.mu.Lock()
	defer k.mu.Unlock()
	if k.session != nil {
		k.session.Destroy()
		k.session = nil
	}
	return nil
}

func (k *kokoro) Synthesize(text, voice, lang string, speed float32) (Audio, error) {
	if k.session == nil {
		return Audio{}, fmt.Errorf("kokoro: session closed")
	}
	if voice == "" {
		voice = "af_heart"
	}
	if lang == "" {
		lang = "en-us"
	}
	if speed < 0.5 || speed > 2.0 {
		return Audio{}, fmt.Errorf("kokoro: speed out of range: %v", speed)
	}

	style, ok := k.voices[voice]
	if !ok {
		return Audio{}, fmt.Errorf("kokoro: voice %q not found (have %d)", voice, len(k.voices))
	}

	rawIPA, err := globalPhon.Phonemize(text, lang)
	if err != nil {
		return Audio{}, fmt.Errorf("phonemize: %w", err)
	}
	cleaned := CleanPhonemes(rawIPA)
	if cleaned == "" {
		return Audio{}, fmt.Errorf("phonemize produced empty output for %q", text)
	}

	tokens := Tokenize(cleaned)
	if len(tokens) == 0 {
		return Audio{}, fmt.Errorf("tokenize empty for cleaned phonemes %q", cleaned)
	}
	if len(tokens) > MaxPhonemeLength {
		tokens = tokens[:MaxPhonemeLength]
	}
	n := len(tokens)

	padded := make([]int64, n+2)
	copy(padded[1:], tokens)

	if n >= style.Shape[0] {
		return Audio{}, fmt.Errorf("kokoro: token count %d >= style rows %d", n, style.Shape[0])
	}
	rowSize := style.Shape[1] * style.Shape[2]
	off := n * rowSize
	styleVec := make([]float32, rowSize)
	copy(styleVec, style.Data[off:off+rowSize])

	tokenTensor, err := ort.NewTensor(ort.NewShape(1, int64(n+2)), padded)
	if err != nil {
		return Audio{}, fmt.Errorf("token tensor: %w", err)
	}
	defer tokenTensor.Destroy()

	styleTensor, err := ort.NewTensor(ort.NewShape(1, 256), styleVec)
	if err != nil {
		return Audio{}, fmt.Errorf("style tensor: %w", err)
	}
	defer styleTensor.Destroy()

	speedTensor, err := ort.NewTensor(ort.NewShape(1), []float32{speed})
	if err != nil {
		return Audio{}, fmt.Errorf("speed tensor: %w", err)
	}
	defer speedTensor.Destroy()

	k.mu.Lock()
	defer k.mu.Unlock()
	outputs := []ort.Value{nil}
	inputs := []ort.Value{tokenTensor, styleTensor, speedTensor}
	if err := k.session.Run(inputs, outputs); err != nil {
		return Audio{}, fmt.Errorf("ort run: %w", err)
	}
	defer outputs[0].Destroy()

	audioTensor, ok := outputs[0].(*ort.Tensor[float32])
	if !ok {
		return Audio{}, fmt.Errorf("unexpected output type %T", outputs[0])
	}
	src := audioTensor.GetData()
	out := make([]float32, len(src))
	copy(out, src)
	return Audio{Samples: out, SampleRate: kokoroSampleRate}, nil
}
