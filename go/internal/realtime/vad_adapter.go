package realtime

import (
	"github.com/eordano/speaches-plus-go/internal/vad"
)

type vadAdapter struct {
	v *vad.Silero
}

func newVADAdapter(modelPath string) (*vadAdapter, error) {
	if modelPath == "" {
		return nil, nil
	}
	v, err := vad.New(modelPath)
	if err != nil {
		return nil, err
	}
	return &vadAdapter{v: v}, nil
}

func (a *vadAdapter) WindowSamples() int { return vad.WindowSamples }

func (a *vadAdapter) Process(window []float32) (vadDecision, error) {
	dec, _, err := a.v.Process(window)
	if err != nil {
		return vadNone, err
	}
	switch dec {
	case vad.SpeechStart:
		return vadSpeechStart, nil
	case vad.SpeechEnd:
		return vadSpeechEnd, nil
	default:
		return vadNone, nil
	}
}

func (a *vadAdapter) PrefixPaddingFrames() int { return vad.PrefixPaddingFrames() }

func (a *vadAdapter) SetThreshold(t float32) {
	if a == nil || a.v == nil {
		return
	}
	a.v.SetThreshold(t)
}

func (a *vadAdapter) SetSilenceMs(ms int) {
	if a == nil || a.v == nil {
		return
	}
	a.v.SetSilenceMs(ms)
}

func (a *vadAdapter) SetNegThreshold(t float32) {
	if a == nil || a.v == nil {
		return
	}
	a.v.SetNegThreshold(t)
}

func (a *vadAdapter) SetMinSpeechMs(ms int) {
	if a == nil || a.v == nil {
		return
	}
	a.v.SetMinSpeechMs(ms)
}

func (a *vadAdapter) SetPrefixPaddingMs(ms int) {
	if a == nil || a.v == nil {
		return
	}
	a.v.SetPrefixPaddingMs(ms)
}

func (a *vadAdapter) Reset() {
	if a != nil && a.v != nil {
		a.v.Reset()
	}
}

func (a *vadAdapter) Close() {
	if a != nil && a.v != nil {
		_ = a.v.Close()
	}
}
