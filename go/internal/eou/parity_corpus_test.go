package eou

import (
	"encoding/json"
	"os"
	"path/filepath"
	"testing"
)

type parityCase struct {
	Text   string  `json:"text"`
	Score  float32 `json:"score"`
	Branch string  `json:"branch"`
}

type parityFixture struct {
	Cases []parityCase `json:"cases"`
}

func TestHeuristicMatchesSharedParityCorpus(t *testing.T) {
	var path string
	for _, c := range []string{
		"../../../conformance/fixtures/060-eou-heuristic-parity/fixture.json",
		"../../conformance/fixtures/060-eou-heuristic-parity/fixture.json",
	} {
		if _, err := os.Stat(c); err == nil {
			path = c
			break
		}
	}
	if path == "" {
		t.Fatal("060-eou-heuristic-parity fixture not found; the shared corpus is required, not optional")
	}
	raw, err := os.ReadFile(filepath.Clean(path))
	if err != nil {
		t.Fatalf("read fixture: %v", err)
	}
	var fx parityFixture
	if err := json.Unmarshal(raw, &fx); err != nil {
		t.Fatalf("parse fixture: %v", err)
	}
	if len(fx.Cases) == 0 {
		t.Fatal("fixture has no cases")
	}
	h := NewHeuristic()
	for _, c := range fx.Cases {
		got := h.score(c.Text, "en")
		if got != c.Score {
			t.Errorf("%q (%s): go=%v want=%v", c.Text, c.Branch, got, c.Score)
		}
	}
	t.Logf("%d cases matched the canonical Rust scores", len(fx.Cases))
}
