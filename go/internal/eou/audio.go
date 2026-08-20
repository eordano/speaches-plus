package eou

import (
	"context"
	"fmt"
	"log/slog"
	"math"
	"os"
	"strings"
	"sync"
	"time"

	ort "github.com/yalue/onnxruntime_go"
	"gonum.org/v1/gonum/dsp/fourier"
)

const (
	audioSampleRate    = 16000
	audioNMels         = 80
	audioNFFT          = 400
	audioHopLength     = 160
	audioChunkSeconds  = 8
	audioTargetSamples = audioChunkSeconds * audioSampleRate
	audioNFrames       = audioTargetSamples / audioHopLength
)

const (
	AudioPadLeading  = "leading"
	AudioPadTrailing = "trailing"
)

type AudioONNXModel struct {
	mu sync.Mutex

	session *ort.DynamicAdvancedSession

	audioWindowMs int
	padAlignment  string

	melFilters []float32
	hann       []float32
	hann64     []float64

	fft    *fourier.FFT
	padded []float32
	frame  []float64
	spec   []complex128
	power  []float32
	mel    []float32
}

type AudioONNXOptions struct {
	AudioWindowMs int
	PadAlignment  string
	InputName     string
	OutputName    string
}

func NewAudioONNXModel(modelPath string, opts AudioONNXOptions) (*AudioONNXModel, error) {
	inputName := opts.InputName
	if inputName == "" {
		inputName = "input_features"
	}
	outputName := opts.OutputName
	if outputName == "" {
		outputName = "logits"
	}
	sess, err := ort.NewDynamicAdvancedSession(
		modelPath,
		[]string{inputName},
		[]string{outputName},
		nil,
	)
	if err != nil {
		return nil, fmt.Errorf("eou audio: load %q: %w", modelPath, err)
	}
	pad := normalizeAudioPadAlignment(opts.PadAlignment)
	window := opts.AudioWindowMs
	if window <= 0 {
		window = defaultAudioWindowMs
	}
	hann := buildAudioHannWindow()
	hann64 := make([]float64, audioNFFT)
	for i, v := range hann {
		hann64[i] = float64(v)
	}
	nBins := audioNFFT/2 + 1
	return &AudioONNXModel{
		session:       sess,
		audioWindowMs: window,
		padAlignment:  pad,
		melFilters:    buildAudioMelFilters(),
		hann:          hann,
		hann64:        hann64,
		fft:           fourier.NewFFT(audioNFFT),
		padded:        make([]float32, audioTargetSamples+audioNFFT),
		frame:         make([]float64, audioNFFT),
		spec:          make([]complex128, nBins),
		power:         make([]float32, nBins*audioNFrames),
		mel:           make([]float32, audioNMels*audioNFrames),
	}, nil
}

func (m *AudioONNXModel) Close() error {
	m.mu.Lock()
	defer m.mu.Unlock()
	if m.session != nil {
		m.session.Destroy()
		m.session = nil
	}
	return nil
}

func (m *AudioONNXModel) Predict(ctx context.Context, req Request) (Verdict, error) {
	t0 := time.Now()
	if req.Kind != "" && req.Kind != KindAudio {
		return Verdict{Latency: time.Since(t0)}, fmt.Errorf("eou audio: got kind=%q, expected %q", req.Kind, KindAudio)
	}
	select {
	case <-ctx.Done():
		return Verdict{Latency: time.Since(t0)}, ctx.Err()
	default:
	}
	audioCopy := append([]float32(nil), req.Audio...)
	type result struct {
		score float32
		err   error
	}
	ch := make(chan result, 1)
	go func() {
		score, err := m.run(audioCopy)
		ch <- result{score: score, err: err}
	}()
	select {
	case r := <-ch:
		if r.err != nil {
			return Verdict{Latency: time.Since(t0)}, r.err
		}
		score := r.score
		if math.IsNaN(float64(score)) || math.IsInf(float64(score), 0) {
			slog.Warn("eou audio: model returned non-finite score", "value", score)
			score = float32(math.NaN())
		}
		return Verdict{Score: score, Latency: time.Since(t0)}, nil
	case <-ctx.Done():
		return Verdict{Latency: time.Since(t0)}, ctx.Err()
	}
}

func (m *AudioONNXModel) run(audio []float32) (float32, error) {
	m.mu.Lock()
	defer m.mu.Unlock()
	if m.session == nil {
		return 0, fmt.Errorf("eou audio: session destroyed")
	}
	m.prepareInto(audio)
	m.computeLogMel()

	in, err := ort.NewTensor(ort.NewShape(1, audioNMels, audioNFrames), m.mel)
	if err != nil {
		return 0, err
	}
	defer in.Destroy()

	outputs := []ort.Value{nil}
	if err := m.session.Run([]ort.Value{in}, outputs); err != nil {
		return 0, fmt.Errorf("eou audio: run: %w", err)
	}
	defer func() {
		if outputs[0] != nil {
			outputs[0].Destroy()
		}
	}()

	tensor, ok := outputs[0].(*ort.Tensor[float32])
	if !ok {
		return 0, fmt.Errorf("eou audio: unexpected output type %T", outputs[0])
	}
	data := tensor.GetData()
	if len(data) == 0 {
		return 0, fmt.Errorf("eou audio: empty output")
	}
	return normalizeAudioOutput(data[0]), nil
}

func (m *AudioONNXModel) prepareInto(audio []float32) {
	const target = audioTargetSamples
	pad := audioNFFT / 2

	maxWindow := m.audioWindowMs * audioSampleRate / 1000
	if maxWindow > target {
		maxWindow = target
	}
	src := audio
	if len(src) > maxWindow {
		src = src[len(src)-maxWindow:]
	}
	mid := m.padded[pad : pad+target]
	for i := range mid {
		mid[i] = 0
	}
	copyAlignedAudio(mid, src, target, m.padAlignment)
	clampAudioSamples(mid)

	for i := 0; i < pad; i++ {
		m.padded[i] = mid[pad-i]
	}
	for i := 0; i < pad; i++ {
		srcIdx := target - 2 - i
		if srcIdx < 0 {
			srcIdx = 0
		}
		m.padded[pad+target+i] = mid[srcIdx]
	}
}

func (m *AudioONNXModel) computeLogMel() {
	nBins := audioNFFT/2 + 1
	frame := m.frame
	spec := m.spec
	power := m.power
	mel := m.mel
	hann64 := m.hann64
	melFilters := m.melFilters

	for f := 0; f < audioNFrames; f++ {
		start := f * audioHopLength
		for i := 0; i < audioNFFT; i++ {
			frame[i] = float64(m.padded[start+i]) * hann64[i]
		}
		m.fft.Coefficients(spec, frame)
		powRow := power[f*nBins : (f+1)*nBins]
		for k := 0; k < nBins; k++ {
			re := real(spec[k])
			im := imag(spec[k])
			powRow[k] = float32(re*re + im*im)
		}
	}

	for i := range mel {
		mel[i] = 0
	}
	for f := 0; f < audioNFrames; f++ {
		powRow := power[f*nBins : (f+1)*nBins]
		for mb := 0; mb < audioNMels; mb++ {
			filt := melFilters[mb*nBins : (mb+1)*nBins]
			var sum float32
			for k := 0; k < nBins; k++ {
				sum += filt[k] * powRow[k]
			}
			mel[mb*audioNFrames+f] = sum
		}
	}

	const eps float32 = 1e-10
	maxVal := float32(math.Inf(-1))
	for i, v := range mel {
		if v < eps {
			v = eps
		}
		lv := float32(math.Log10(float64(v)))
		mel[i] = lv
		if lv > maxVal {
			maxVal = lv
		}
	}
	floor := maxVal - 8.0
	for i, v := range mel {
		if v < floor {
			v = floor
		}
		mel[i] = (v + 4.0) / 4.0
	}
}

func normalizeAudioOutput(raw float32) float32 {
	f := float64(raw)
	if math.IsNaN(f) || math.IsInf(f, 0) {
		return raw
	}
	if raw >= 0.0 && raw <= 1.0 {
		return raw
	}
	return float32(1.0 / (1.0 + math.Exp(-f)))
}

func PrepareAudio(audio []float32, audioWindowMs int, padAlignment string) []float32 {
	if audioWindowMs <= 0 {
		audioWindowMs = defaultAudioWindowMs
	}
	target := audioTargetSamples
	maxWindow := audioWindowMs * audioSampleRate / 1000
	if maxWindow > target {
		maxWindow = target
	}
	src := audio
	if len(src) > maxWindow {
		src = src[len(src)-maxWindow:]
	}

	out := make([]float32, target)
	copyAlignedAudio(out, src, target, normalizeAudioPadAlignment(padAlignment))
	clampAudioSamples(out)
	return out
}

func copyAlignedAudio(dst, src []float32, target int, pad string) {
	switch {
	case len(src) >= target:
		copy(dst, src[len(src)-target:])
	case pad == AudioPadTrailing:
		copy(dst, src)
	default:
		copy(dst[target-len(src):], src)
	}
}

func clampAudioSamples(s []float32) {
	for i, v := range s {
		f := float64(v)
		if math.IsNaN(f) || math.IsInf(f, 0) {
			s[i] = 0
		} else if v > 1 {
			s[i] = 1
		} else if v < -1 {
			s[i] = -1
		}
	}
}

func normalizeAudioPadAlignment(s string) string {
	switch strings.ToLower(strings.TrimSpace(s)) {
	case AudioPadTrailing:
		return AudioPadTrailing
	default:
		return AudioPadLeading
	}
}

func buildAudioHannWindow() []float32 {
	w := make([]float32, audioNFFT)
	for i := range w {
		phase := 2.0 * math.Pi * float64(i) / float64(audioNFFT)
		w[i] = float32(0.5 - 0.5*math.Cos(phase))
	}
	return w
}

func buildAudioMelFilters() []float32 {
	nBins := audioNFFT/2 + 1
	const fMin = 0.0
	fMax := float64(audioSampleRate) / 2.0
	mMin := audioHzToMel(fMin)
	mMax := audioHzToMel(fMax)

	melPoints := make([]float64, audioNMels+2)
	for i := range melPoints {
		frac := float64(i) / float64(audioNMels+1)
		melPoints[i] = mMin + (mMax-mMin)*frac
	}
	hzPoints := make([]float64, audioNMels+2)
	for i, m := range melPoints {
		hzPoints[i] = audioMelToHz(m)
	}

	fftFreqs := make([]float64, nBins)
	for i := range fftFreqs {
		fftFreqs[i] = float64(i) * float64(audioSampleRate) / float64(audioNFFT)
	}

	filters := make([]float32, audioNMels*nBins)
	const eps = 1e-30
	for m := 0; m < audioNMels; m++ {
		lower := hzPoints[m]
		center := hzPoints[m+1]
		upper := hzPoints[m+2]
		lowerSlope := center - lower
		if lowerSlope < eps {
			lowerSlope = eps
		}
		upperSlope := upper - center
		if upperSlope < eps {
			upperSlope = eps
		}
		span := upper - lower
		if span < eps {
			span = eps
		}
		enorm := 2.0 / span
		for k := 0; k < nBins; k++ {
			freq := fftFreqs[k]
			var weight float64
			switch {
			case freq >= lower && freq <= center:
				weight = (freq - lower) / lowerSlope
			case freq > center && freq <= upper:
				weight = (upper - freq) / upperSlope
			}
			filters[m*nBins+k] = float32(weight * enorm)
		}
	}
	return filters
}

const (
	melFSp      = 200.0 / 3.0
	melMinLogHz = 1000.0
)

var (
	melMinLogMel = melMinLogHz / melFSp
	melLogStep   = math.Log(6.4) / 27.0
)

func audioHzToMel(f float64) float64 {
	if f >= melMinLogHz {
		return melMinLogMel + math.Log(f/melMinLogHz)/melLogStep
	}
	return f / melFSp
}

func audioMelToHz(m float64) float64 {
	if m >= melMinLogMel {
		return melMinLogHz * math.Exp((m-melMinLogMel)*melLogStep)
	}
	return melFSp * m
}

func logMelSpectrogramAudioEou(audio []float32, hann []float32, melFilters []float32, fft *fourier.FFT) ([]float32, error) {
	if len(hann) != audioNFFT {
		return nil, fmt.Errorf("eou audio: hann length %d != %d", len(hann), audioNFFT)
	}
	nBins := audioNFFT/2 + 1
	if len(melFilters) != audioNMels*nBins {
		return nil, fmt.Errorf("eou audio: mel filter size %d != %d", len(melFilters), audioNMels*nBins)
	}
	if len(audio) < audioNFFT {
		return nil, fmt.Errorf("eou audio: audio length %d < N_FFT (%d)", len(audio), audioNFFT)
	}

	pad := audioNFFT / 2
	padded := make([]float32, len(audio)+audioNFFT)
	for i := 0; i < pad; i++ {
		padded[i] = audio[pad-i]
	}
	copy(padded[pad:pad+len(audio)], audio)
	for i := 0; i < pad; i++ {
		src := len(audio) - 2 - i
		if src < 0 {
			src = 0
		}
		padded[pad+len(audio)+i] = audio[src]
	}

	power := make([]float32, nBins*audioNFrames)
	frame := make([]float64, audioNFFT)
	spec := make([]complex128, nBins)
	for f := 0; f < audioNFrames; f++ {
		start := f * audioHopLength
		for i := 0; i < audioNFFT; i++ {
			frame[i] = float64(padded[start+i]) * float64(hann[i])
		}
		fft.Coefficients(spec, frame)
		powRow := power[f*nBins : (f+1)*nBins]
		for k := 0; k < nBins; k++ {
			re := real(spec[k])
			im := imag(spec[k])
			powRow[k] = float32(re*re + im*im)
		}
	}

	mel := make([]float32, audioNMels*audioNFrames)
	for f := 0; f < audioNFrames; f++ {
		powRow := power[f*nBins : (f+1)*nBins]
		for mb := 0; mb < audioNMels; mb++ {
			filt := melFilters[mb*nBins : (mb+1)*nBins]
			var sum float32
			for k := 0; k < nBins; k++ {
				sum += filt[k] * powRow[k]
			}
			mel[mb*audioNFrames+f] = sum
		}
	}

	const eps float32 = 1e-10
	maxVal := float32(math.Inf(-1))
	for i, v := range mel {
		if v < eps {
			v = eps
		}
		lv := float32(math.Log10(float64(v)))
		mel[i] = lv
		if lv > maxVal {
			maxVal = lv
		}
	}
	floor := maxVal - 8.0
	for i, v := range mel {
		if v < floor {
			v = floor
		}
		mel[i] = (v + 4.0) / 4.0
	}
	return mel, nil
}

func LoadAudioFromEnv(audioWindowMs int, padAlignment string) Model {
	return loadAudioFromPath(strings.TrimSpace(os.Getenv("EOU_AUDIO_MODEL_PATH")),
		audioWindowMs, padAlignment)
}

func loadAudioFromPath(path string, audioWindowMs int, padAlignment string) Model {
	path = strings.TrimSpace(path)
	if path == "" {
		return nil
	}
	if _, err := os.Stat(path); err != nil {
		slog.Warn("eou: smart-turn model path set but file not found; falling back",
			"path", path, "err", err)
		return nil
	}
	m, err := NewAudioONNXModel(path, AudioONNXOptions{
		AudioWindowMs: audioWindowMs,
		PadAlignment:  padAlignment,
	})
	if err != nil {
		slog.Warn("eou: smart-turn audio model load failed; falling back",
			"path", path, "err", err)
		return nil
	}
	slog.Info("eou: smart-turn audio model loaded",
		"path", path, "window_ms", audioWindowMs, "pad_alignment", normalizeAudioPadAlignment(padAlignment))
	return m
}
