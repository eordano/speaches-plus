package stt

import (
	"fmt"
	"os"
	"path/filepath"
	"regexp"
	"strings"
	"testing"
)

func TestCT2_TranscribeSegments_MatchesLegacy(t *testing.T) {
	audioPath := os.Getenv("CT2_DIFF_AUDIO")
	if audioPath == "" {
		audioPath = "/tmp/audio16k_ref.bin"
	}
	if _, err := os.Stat(audioPath); err != nil {
		t.Skipf("no reference audio at %s: %v", audioPath, err)
	}

	modelDir := os.Getenv("CT2_DIFF_MODEL_DIR")
	if modelDir == "" {
		modelDir = findCT2Model(t)
		if modelDir == "" {
			t.Skip("no ct2 whisper model found (set CT2_DIFF_MODEL_DIR)")
		}
	}

	audio, err := loadFloat32Bin(audioPath)
	if err != nil {
		t.Fatalf("loadFloat32Bin: %v", err)
	}

	tr, err := NewCT2(CT2Config{ModelDir: modelDir, Device: "cpu", Language: "en"})
	if err != nil {
		t.Fatalf("NewCT2: %v", err)
	}
	defer tr.Close()

	legacy, err := tr.Transcribe(audio, 16000)
	if err != nil {
		t.Fatalf("Transcribe (legacy): %v", err)
	}

	segT, ok := tr.(SegmentTranscriber)
	if !ok {
		t.Fatal("CT2 doesn't implement SegmentTranscriber")
	}
	res, err := segT.TranscribeSegments(audio, 16000)
	if err != nil {
		t.Fatalf("TranscribeSegments: %v", err)
	}

	t.Logf("legacy : %q", legacy)
	t.Logf("new    : %q", res.Text)
	t.Logf("segments (%d):", len(res.Segments))
	for i, s := range res.Segments {
		t.Logf("  [%d] %d-%d ms  %q", i, s.TStartMs, s.TEndMs, s.Text)
	}

	lw := words(legacy)
	nw := words(res.Text)
	if len(lw) == 0 {
		t.Fatal("legacy decode produced no words; sample may be too quiet")
	}
	if !equalSlices(lw, nw) {
		t.Fatalf("text drifted between legacy and new path:\nlegacy: %v\n   new: %v", lw, nw)
	}

	var segWords []string
	for _, s := range res.Segments {
		segWords = append(segWords, words(s.Text)...)
	}
	if !equalSlices(segWords, nw) {
		t.Fatalf("joined text != concat(segment.Text):\nsegs:    %v\njoined:  %v", segWords, nw)
	}

	if len(res.Segments) < 2 {
		t.Errorf("expected multi-sentence sample to produce >=2 segments, got %d", len(res.Segments))
	}
}

var wordRE = regexp.MustCompile(`[A-Za-z0-9]+`)

func words(s string) []string {
	out := wordRE.FindAllString(strings.ToLower(s), -1)
	if out == nil {
		return nil
	}
	return out
}

func equalSlices(a, b []string) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

func findCT2Model(t *testing.T) string {
	t.Helper()
	caches, _ := filepath.Glob("/nix/store/*-hf-hub-cache")
	for _, c := range caches {
		matches, _ := filepath.Glob(fmt.Sprintf("%s/models--*faster-whisper*/snapshots/*", c))
		for _, m := range matches {

			real, err := filepath.EvalSymlinks(m)
			if err != nil {
				continue
			}
			if _, err := os.Stat(filepath.Join(real, "model.bin")); err != nil {
				continue
			}
			if _, err := os.Stat(filepath.Join(real, "tokenizer.json")); err != nil {
				continue
			}
			return real
		}
	}

	cwd, _ := os.Getwd()
	for dir := cwd; dir != "/" && dir != ""; dir = filepath.Dir(dir) {
		candidate := filepath.Join(dir, "rust", "models", "whisper-ct2")
		if _, err := os.Stat(filepath.Join(candidate, "model.bin")); err == nil {
			return candidate
		}
	}
	return ""
}
