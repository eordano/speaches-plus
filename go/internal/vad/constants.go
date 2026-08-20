package vad

const (
	WindowSamples  = 512
	ContextSamples = 64
	SampleRate     = 16000

	defaultThreshold       = 0.5
	defaultPrefixPaddingMs = 300
	defaultSilenceMs       = 500
	defaultMinSpeechMs     = 90

	MaxVadWindowMs      = 3000
	MaxVadWindowSamples = MaxVadWindowMs * SampleRate / 1000

	MinSilenceAtMaxSpeechMs = 98

	NegThresholdDelta = 0.15
	NegThresholdFloor = 0.01
)

const (
	None Decision = iota
	SpeechStart
	SpeechEnd
)
