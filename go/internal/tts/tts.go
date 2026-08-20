package tts

type Audio struct {
	Samples    []float32
	SampleRate int
}

type Synthesizer interface {
	Synthesize(text, voice, lang string, speed float32) (Audio, error)
	Close() error
}

type KokoroConfig struct {
	ModelPath  string
	VoicesPath string
	EspeakData string
}
