package eou

import (
	"context"
	"math"
	"os"
	"path/filepath"
	"testing"

	ort "github.com/yalue/onnxruntime_go"
	"gonum.org/v1/gonum/dsp/fourier"
)

func ensureORTAudio(t *testing.T) {
	if libPath := os.Getenv("ONNXRUNTIME_LIB"); libPath != "" {
		ort.SetSharedLibraryPath(libPath)
	}
	if err := ort.InitializeEnvironment(); err != nil {
		t.Logf("ort init (may already be initialized): %v", err)
	}
}

func TestPrepareAudioPadsLeadingWhenShort(t *testing.T) {
	audio := make([]float32, 1600)
	for i := range audio {
		audio[i] = (float32(i) + 1.0) / 2000.0
	}
	prepared := PrepareAudio(audio, 8000, AudioPadLeading)
	if len(prepared) != audioTargetSamples {
		t.Fatalf("len=%d want %d", len(prepared), audioTargetSamples)
	}
	padLen := audioTargetSamples - 1600
	for i := 0; i < padLen; i++ {
		if prepared[i] != 0 {
			t.Fatalf("expected zero pad at %d, got %v", i, prepared[i])
		}
	}
	for i := 0; i < 1600; i++ {
		if math.Abs(float64(prepared[padLen+i]-audio[i])) > 1e-6 {
			t.Fatalf("mismatch at %d got %v want %v", i, prepared[padLen+i], audio[i])
		}
	}
}

func TestPrepareAudioPadsTrailingWhenConfigured(t *testing.T) {
	audio := make([]float32, 1600)
	for i := range audio {
		audio[i] = (float32(i) + 1.0) / 2000.0
	}
	prepared := PrepareAudio(audio, 8000, AudioPadTrailing)
	if len(prepared) != audioTargetSamples {
		t.Fatalf("len=%d want %d", len(prepared), audioTargetSamples)
	}
	for i := 0; i < 1600; i++ {
		if math.Abs(float64(prepared[i]-audio[i])) > 1e-6 {
			t.Fatalf("mismatch at %d got %v want %v", i, prepared[i], audio[i])
		}
	}
	for i := 1600; i < audioTargetSamples; i++ {
		if prepared[i] != 0 {
			t.Fatalf("expected zero at %d, got %v", i, prepared[i])
		}
	}
}

func TestPrepareAudioTruncatesToLastWindow(t *testing.T) {
	audio := make([]float32, audioTargetSamples+10000)
	for i := range audio {
		audio[i] = ((float32(i) - float32((i/1000)*1000)) - 500.0) / 1000.0
	}
	prepared := PrepareAudio(audio, 8000, AudioPadLeading)
	if len(prepared) != audioTargetSamples {
		t.Fatalf("len=%d want %d", len(prepared), audioTargetSamples)
	}
	wantFirst := audio[len(audio)-audioTargetSamples]
	if math.Abs(float64(prepared[0]-wantFirst)) > 1e-9 {
		t.Fatalf("first: got %v want %v", prepared[0], wantFirst)
	}
	wantLast := audio[len(audio)-1]
	if math.Abs(float64(prepared[audioTargetSamples-1]-wantLast)) > 1e-9 {
		t.Fatalf("last: got %v want %v", prepared[audioTargetSamples-1], wantLast)
	}
}

func TestPrepareAudioClampsOutOfRangeSamples(t *testing.T) {
	audio := []float32{
		5.0, -3.0, float32(math.NaN()), 0.5, float32(math.Inf(1)),
	}
	prepared := PrepareAudio(audio, 8000, AudioPadLeading)
	if len(prepared) != audioTargetSamples {
		t.Fatalf("len=%d want %d", len(prepared), audioTargetSamples)
	}
	tail := prepared[audioTargetSamples-len(audio):]
	if tail[0] != 1.0 {
		t.Fatalf("tail[0]=%v want 1.0", tail[0])
	}
	if tail[1] != -1.0 {
		t.Fatalf("tail[1]=%v want -1.0", tail[1])
	}
	if tail[2] != 0.0 {
		t.Fatalf("tail[2]=%v want 0.0 (NaN clamp)", tail[2])
	}
	if math.Abs(float64(tail[3]-0.5)) > 1e-6 {
		t.Fatalf("tail[3]=%v want 0.5", tail[3])
	}
	if tail[4] != 0.0 {
		t.Fatalf("tail[4]=%v want 0.0 (Inf clamp)", tail[4])
	}
}

func TestMelFiltersAreFiniteAndNonNegative(t *testing.T) {
	filters := buildAudioMelFilters()
	for i, v := range filters {
		if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
			t.Fatalf("filter[%d] not finite: %v", i, v)
		}
		if v < 0 {
			t.Fatalf("filter[%d] negative: %v", i, v)
		}
	}
}

func TestLogMelShapeIsCorrect(t *testing.T) {
	hann := buildAudioHannWindow()
	filters := buildAudioMelFilters()
	fft := fourier.NewFFT(audioNFFT)
	audio := make([]float32, audioTargetSamples)
	for i := range audio {
		audio[i] = 0.1
	}
	mel, err := logMelSpectrogramAudioEou(audio, hann, filters, fft)
	if err != nil {
		t.Fatalf("logMel: %v", err)
	}
	if len(mel) != audioNMels*audioNFrames {
		t.Fatalf("len=%d want %d", len(mel), audioNMels*audioNFrames)
	}
	for i, v := range mel {
		if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
			t.Fatalf("mel[%d] not finite: %v", i, v)
		}
	}
}

func TestLogMelRejectsShortAudio(t *testing.T) {
	hann := buildAudioHannWindow()
	filters := buildAudioMelFilters()
	fft := fourier.NewFFT(audioNFFT)
	if _, err := logMelSpectrogramAudioEou([]float32{}, hann, filters, fft); err == nil {
		t.Fatalf("expected error on empty audio")
	}
	if _, err := logMelSpectrogramAudioEou(make([]float32, audioNFFT-1), hann, filters, fft); err == nil {
		t.Fatalf("expected error on N_FFT-1 audio")
	}
}

func TestPrepareAudioEmptyAndShort(t *testing.T) {
	for _, n := range []int{0, 1, 2, audioNFFT, audioTargetSamples - 1, audioTargetSamples} {
		got := PrepareAudio(make([]float32, n), 8000, AudioPadLeading)
		if len(got) != audioTargetSamples {
			t.Fatalf("len=%d for n=%d want %d", len(got), n, audioTargetSamples)
		}
	}
}

func TestPrepareAudioTrimsThenPads(t *testing.T) {

	const windowMs = 4000
	maxWindow := windowMs * audioSampleRate / 1000
	in := make([]float32, audioTargetSamples)
	for i := range in {
		in[i] = float32(i) / float32(len(in))
	}
	out := PrepareAudio(in, windowMs, AudioPadLeading)
	if len(out) != audioTargetSamples {
		t.Fatalf("len=%d want %d", len(out), audioTargetSamples)
	}
	padLen := audioTargetSamples - maxWindow
	for i := 0; i < padLen; i++ {
		if out[i] != 0 {
			t.Fatalf("expected zero pad at %d, got %v", i, out[i])
		}
	}
	wantFirstPayload := in[len(in)-maxWindow]
	if math.Abs(float64(out[padLen]-wantFirstPayload)) > 1e-6 {
		t.Fatalf("payload[0]=%v want %v", out[padLen], wantFirstPayload)
	}
}

func TestLoadAudioFromEnvSkipsWhenUnset(t *testing.T) {
	t.Setenv("EOU_AUDIO_MODEL_PATH", "")
	if m := LoadAudioFromEnv(8000, AudioPadLeading); m != nil {
		t.Fatalf("expected nil model, got %T", m)
	}
}

func TestLoadAudioFromEnvSkipsWhenAbsent(t *testing.T) {
	t.Setenv("EOU_AUDIO_MODEL_PATH", "/this/does/not/exist/smart-turn.onnx")
	if m := LoadAudioFromEnv(8000, AudioPadLeading); m != nil {
		t.Fatalf("expected nil model, got %T", m)
	}
}

func TestNormalizeAudioOutputProbAndLogit(t *testing.T) {
	if got := normalizeAudioOutput(0.42); got != 0.42 {
		t.Fatalf("prob passthrough: got %v want 0.42", got)
	}
	if got := normalizeAudioOutput(0); got != 0 {
		t.Fatalf("zero passthrough: got %v want 0", got)
	}
	if got := normalizeAudioOutput(1); got != 1 {
		t.Fatalf("one passthrough: got %v want 1", got)
	}

	got := float64(normalizeAudioOutput(2.0))
	want := 1.0 / (1.0 + math.Exp(-2.0))
	if math.Abs(got-want) > 1e-6 {
		t.Fatalf("logit: got %v want %v", got, want)
	}
	got = float64(normalizeAudioOutput(-2.0))
	want = 1.0 / (1.0 + math.Exp(2.0))
	if math.Abs(got-want) > 1e-6 {
		t.Fatalf("neg logit: got %v want %v", got, want)
	}
}

func TestLogMelGoldenVectorMatchesRust(t *testing.T) {
	audio := make([]float32, audioTargetSamples)
	for i := range audio {
		tt := float32(i) / float32(audioSampleRate)
		phase := float32(2.0*math.Pi*440.0) * tt
		audio[i] = 0.5 * float32(math.Sin(float64(phase)))
	}
	prepared := PrepareAudio(audio, 8000, AudioPadLeading)
	hann := buildAudioHannWindow()
	filters := buildAudioMelFilters()
	fft := fourier.NewFFT(audioNFFT)
	mel, err := logMelSpectrogramAudioEou(prepared, hann, filters, fft)
	if err != nil {
		t.Fatalf("logMel: %v", err)
	}
	if len(mel) != audioNMels*audioNFrames {
		t.Fatalf("mel len=%d want %d", len(mel), audioNMels*audioNFrames)
	}

	stride := audioNFrames
	want := []struct {
		m, f int
		v    float32
	}{
		{0, 0, 0.983278811},
		{0, 1, 0.471271873},
		{20, 0, 0.817899704},
		{40, 0, 0.418184519},
		{60, 0, 0.096875548},
		{40, 400, -0.548648715},
		{40, 600, -0.510974765},
		{0, 2, -0.561791658},
		{0, 19, -0.561791658},
	}
	const tol = 5e-4
	for _, w := range want {
		got := mel[w.m*stride+w.f]
		if math.Abs(float64(got-w.v)) > tol {
			t.Errorf("mel[m=%d,f=%d]=%v want %v (tol %v)", w.m, w.f, got, w.v, tol)
		}
	}
}

func TestSmartTurnLoadsWhenModelPresentAndRuns(t *testing.T) {
	candidates := []string{
		os.Getenv("SMART_TURN_V3_PATH"),
		filepath.Join("..", "..", "..", "rust", "models", "smart-turn-v3.onnx"),
	}
	var path string
	for _, c := range candidates {
		if c == "" {
			continue
		}
		if _, err := os.Stat(c); err == nil {
			path = c
			break
		}
	}
	if path == "" {
		t.Skip("smart-turn-v3.onnx not found; skipping (set SMART_TURN_V3_PATH)")
	}
	ensureORTAudio(t)
	m, err := NewAudioONNXModel(path, AudioONNXOptions{
		AudioWindowMs: 8000,
		PadAlignment:  AudioPadLeading,
	})
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	defer m.Close()
	silence := make([]float32, audioTargetSamples)
	v, err := m.Predict(context.Background(), Request{
		Kind:  KindAudio,
		Audio: silence,
	})
	if err != nil {
		t.Fatalf("predict: %v", err)
	}
	if math.IsNaN(float64(v.Score)) || math.IsInf(float64(v.Score), 0) {
		t.Fatalf("score not finite: %v", v.Score)
	}
	if v.Score < 0 || v.Score > 1 {
		t.Fatalf("score %v out of [0,1]", v.Score)
	}
}

func TestPredictRejectsMismatchedKind(t *testing.T) {
	m := &AudioONNXModel{audioWindowMs: 8000, padAlignment: AudioPadLeading}
	_, err := m.Predict(context.Background(), Request{Kind: KindText})
	if err == nil {
		t.Fatalf("expected error for kind=text")
	}
}

func TestLoadAudioFromPathRejectsGarbageFile(t *testing.T) {
	dir := t.TempDir()
	bogus := filepath.Join(dir, "not-an-onnx.bin")
	if err := os.WriteFile(bogus, []byte("definitely not an onnx model"), 0o644); err != nil {
		t.Fatalf("write: %v", err)
	}
	ensureORTAudio(t)
	if got := loadAudioFromPath(bogus, 8000, AudioPadLeading); got != nil {
		t.Fatalf("expected nil for garbage file, got %T", got)
	}
}

func TestLoadFallsBackOnAudioPathMissing(t *testing.T) {
	t.Setenv("EOU_AUDIO_MODEL_PATH", "")
	cfg := Config{Kind: KindAudio, AudioModelPath: "/nope/missing.onnx"}
	m, got, err := Load(cfg)
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	if m == nil {
		t.Fatalf("expected fallback model, got nil")
	}
	if got.AudioModel != nil {
		t.Fatalf("expected AudioModel=nil on missing path, got %T", got.AudioModel)
	}
}

func TestFusionLoadPopulatesTextModel(t *testing.T) {
	t.Setenv("EOU_AUDIO_MODEL_PATH", "")
	cfg := Config{Kind: KindFusion}
	_, got, err := Load(cfg)
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	if got.TextModel == nil {
		t.Fatalf("expected TextModel populated for KindFusion")
	}
}
