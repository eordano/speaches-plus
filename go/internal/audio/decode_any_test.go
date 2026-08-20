package audio_test

import (
	"bytes"
	"encoding/binary"
	"math"
	"os"
	"os/exec"
	"path/filepath"
	"testing"

	"github.com/eordano/speaches-plus-go/internal/audio"
)

func sine(sampleRate int, freqHz float64) audio.MonoF32 {
	n := sampleRate
	out := make(audio.MonoF32, n)
	for i := 0; i < n; i++ {
		out[i] = float32(0.5 * math.Sin(2*math.Pi*freqHz*float64(i)/float64(sampleRate)))
	}
	return out
}

func encodeRawS16LE(samples audio.MonoF32) []byte {
	buf := bytes.NewBuffer(make([]byte, 0, len(samples)*2))
	for _, s := range samples {
		v := int32(s * 32767.0)
		if v > 32767 {
			v = 32767
		} else if v < -32768 {
			v = -32768
		}
		_ = binary.Write(buf, binary.LittleEndian, int16(v))
	}
	return buf.Bytes()
}

func rmsf32(s audio.MonoF32) float64 {
	if len(s) == 0 {
		return 0
	}
	var ss float64
	for _, v := range s {
		ss += float64(v) * float64(v)
	}
	return math.Sqrt(ss / float64(len(s)))
}

func alignAndDiff(a, b audio.MonoF32, skip int) (maxAbs float32, relRMS float64) {
	n := len(a)
	if len(b) < n {
		n = len(b)
	}
	if skip > n/4 {
		skip = n / 4
	}
	a = a[skip : n-skip]
	b = b[skip : n-skip]
	if len(a) == 0 {
		return 0, 0
	}
	maxAbs = 0
	var sumDiffSq, sumA float64
	for i := range a {
		d := a[i] - b[i]
		if d < 0 {
			d = -d
		}
		if d > maxAbs {
			maxAbs = d
		}
		sumDiffSq += float64(d) * float64(d)
		sumA += float64(a[i]) * float64(a[i])
	}
	if sumA == 0 {
		return maxAbs, 0
	}
	return maxAbs, math.Sqrt(sumDiffSq / sumA)
}

func TestDecodeUploadedAudio_RawPCM(t *testing.T) {
	src := sine(16000, 440.0)
	bytes := encodeRawS16LE(src)
	got, err := audio.DecodeUploadedAudio(bytes, "audio/pcm")
	if err != nil {
		t.Fatalf("DecodeUploadedAudio audio/pcm: %v", err)
	}
	if len(got) != len(src) {
		t.Fatalf("sample count: got %d want %d", len(got), len(src))
	}
	maxAbs, relRMS := alignAndDiff(src, got, 0)

	if maxAbs > 1e-3 || relRMS > 1e-3 {
		t.Fatalf("raw PCM roundtrip drift too high: maxAbs=%g relRMS=%g", maxAbs, relRMS)
	}
}

func TestDecodeUploadedAudio_WAV(t *testing.T) {
	src := sine(16000, 440.0)
	wavBytes := audio.EncodeWAVMono16(src, 16000)
	got, err := audio.DecodeUploadedAudio(wavBytes, "audio/wav")
	if err != nil {
		t.Fatalf("DecodeUploadedAudio audio/wav: %v", err)
	}
	if len(got) != len(src) {
		t.Fatalf("sample count: got %d want %d", len(got), len(src))
	}
	maxAbs, relRMS := alignAndDiff(src, got, 0)
	if maxAbs > 1e-3 || relRMS > 1e-3 {
		t.Fatalf("WAV roundtrip drift too high: maxAbs=%g relRMS=%g", maxAbs, relRMS)
	}
}

func TestDecodeUploadedAudio_WAV_Resample(t *testing.T) {
	src := sine(48000, 440.0)
	wavBytes := audio.EncodeWAVMono16(src, 48000)
	got, err := audio.DecodeUploadedAudio(wavBytes, "audio/wav")
	if err != nil {
		t.Fatalf("DecodeUploadedAudio audio/wav 48k: %v", err)
	}

	if got := len(got); got < 16000-2 || got > 16000+2 {
		t.Fatalf("resampled length: got %d want ~16000", got)
	}
	if rmsf32(got) < 0.1 {
		t.Fatalf("resampled rms suspiciously low: %g", rmsf32(got))
	}
}

func TestDecodeUploadedAudio_Empty(t *testing.T) {
	if _, err := audio.DecodeUploadedAudio(nil, ""); err == nil {
		t.Fatalf("expected error on empty input, got nil")
	}
}

func TestDecodeUploadedAudio_LibAV_MP3(t *testing.T) {
	requireFfmpeg(t)
	src := sine(16000, 440.0)
	mp3 := encodeViaFfmpeg(t, src, 16000, "mp3")

	got, err := audio.DecodeUploadedAudio(mp3, "")
	if err != nil {
		t.Fatalf("DecodeUploadedAudio mp3: %v", err)
	}

	if len(got) < 12000 {
		t.Fatalf("decoded mp3 too short: %d samples", len(got))
	}
	maxAbs, relRMS := alignAndDiff(src, got, 1500)

	if relRMS > 0.3 {
		t.Fatalf("mp3 roundtrip relRMS too high: %g (maxAbs=%g)", relRMS, maxAbs)
	}
	if rmsf32(got) < 0.1 {
		t.Fatalf("decoded mp3 rms suspiciously low: %g", rmsf32(got))
	}
}

func TestDecodeUploadedAudio_LibAV_FLAC(t *testing.T) {
	requireFfmpeg(t)
	src := sine(16000, 440.0)
	flac := encodeViaFfmpeg(t, src, 16000, "flac")

	got, err := audio.DecodeUploadedAudio(flac, "audio/flac")
	if err != nil {
		t.Fatalf("DecodeUploadedAudio flac: %v", err)
	}
	if len(got) != len(src) {
		t.Fatalf("flac sample count: got %d want %d", len(got), len(src))
	}

	maxAbs, relRMS := alignAndDiff(src, got, 0)
	if maxAbs > 1e-3 || relRMS > 1e-3 {
		t.Fatalf("flac roundtrip drift too high: maxAbs=%g relRMS=%g", maxAbs, relRMS)
	}
}

func TestDecodeUploadedAudio_LibAV_OGGResample(t *testing.T) {
	requireFfmpeg(t)
	src := sine(44100, 440.0)
	ogg := encodeViaFfmpeg(t, src, 44100, "ogg")

	got, err := audio.DecodeUploadedAudio(ogg, "audio/ogg")
	if err != nil {
		t.Fatalf("DecodeUploadedAudio ogg: %v", err)
	}

	if got := len(got); got < 16000-200 || got > 16000+200 {
		t.Fatalf("ogg resampled length: got %d want ~16000", got)
	}
	if rmsf32(got) < 0.1 {
		t.Fatalf("ogg decoded rms suspiciously low: %g", rmsf32(got))
	}
}

func requireFfmpeg(t *testing.T) {
	t.Helper()
	if _, err := exec.LookPath("ffmpeg"); err != nil {
		t.Skipf("ffmpeg not on PATH; skipping libav fallback test")
	}
}

func encodeViaFfmpeg(t *testing.T, samples audio.MonoF32, sampleRate int, ext string) []byte {
	t.Helper()
	tmpDir := t.TempDir()
	srcPath := filepath.Join(tmpDir, "src.wav")
	dstPath := filepath.Join(tmpDir, "out."+ext)
	if err := os.WriteFile(srcPath, audio.EncodeWAVMono16(samples, sampleRate), 0o644); err != nil {
		t.Fatalf("write src wav: %v", err)
	}
	cmd := exec.Command("ffmpeg", "-y", "-loglevel", "error", "-i", srcPath, dstPath)
	if out, err := cmd.CombinedOutput(); err != nil {
		t.Fatalf("ffmpeg encode %s: %v\n%s", ext, err, out)
	}
	bytes, err := os.ReadFile(dstPath)
	if err != nil {
		t.Fatalf("read encoded %s: %v", ext, err)
	}
	return bytes
}
