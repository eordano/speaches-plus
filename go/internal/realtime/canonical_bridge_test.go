package realtime

import (
	"bytes"
	"encoding/json"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
)

func canonicalLibPath() string {
	candidates := []string{
		"../../../conformance/lib/trace_invariants.py",
		"../../conformance/lib/trace_invariants.py",
	}
	for _, c := range candidates {
		if st, err := os.Stat(c); err == nil && !st.IsDir() {
			abs, _ := filepath.Abs(c)
			return abs
		}
	}
	return ""
}

func runCanonicalAgainst(t *testing.T, lib string, events []TraceEvent) (string, error) {
	t.Helper()
	dir := t.TempDir()
	path := filepath.Join(dir, "trace.jsonl")
	f, err := os.Create(path)
	if err != nil {
		t.Fatalf("create temp trace: %v", err)
	}
	enc := json.NewEncoder(f)
	if err := enc.Encode(map[string]any{"kind": "config", "source": "go-conformance-bridge"}); err != nil {
		f.Close()
		t.Fatalf("encode config: %v", err)
	}
	for _, ev := range events {
		if err := enc.Encode(map[string]any{"kind": "event", "event": ev}); err != nil {
			f.Close()
			t.Fatalf("encode trace event: %v", err)
		}
	}
	f.Close()

	only := strings.Join([]string{
		"W4_response_done_carries_audio_end_ms",
		"W6_no_response_events_after_done",
		"W7_assistant_truncated_paired_with_cancelled_done",
		"W8_client_create_paired_with_server_created",
		"W1_response_done_per_created",
		"W2_delta_only_between_created_and_done",
		"W3_committed_after_stopped_before_created",
	}, ",")
	var stdout, stderr bytes.Buffer
	cmd := exec.Command("python3", lib, "--only", only, path)
	cmd.Stdout = &stdout
	cmd.Stderr = &stderr
	err = cmd.Run()
	out := stdout.String() + stderr.String()
	return out, err
}

func TestCanonicalLib_AgreesWithGoOnConformanceCorpus(t *testing.T) {
	lib := canonicalLibPath()
	if lib == "" {
		t.Skip("RFC §15.2 canonical lib not found; skipping cross-language bridge")
	}
	if _, err := exec.LookPath("python3"); err != nil {
		t.Skip("python3 not on PATH; skipping cross-language bridge")
	}

	root := corpusRoot(t)
	entries, err := os.ReadDir(root)
	if err != nil {
		t.Fatalf("read corpus dir %s: %v", root, err)
	}
	any := 0
	for _, e := range entries {
		if !e.IsDir() || strings.HasPrefix(e.Name(), ".") {
			continue
		}
		dir := filepath.Join(root, e.Name())
		inputPath := filepath.Join(dir, "input.jsonl")
		if _, err := os.Stat(inputPath); err != nil {
			continue
		}
		any++
		t.Run(e.Name(), func(t *testing.T) {
			input := loadJSONL(t, inputPath)
			actual := replay(t, input)
			out, runErr := runCanonicalAgainst(t, lib, actual)
			if runErr != nil {
				t.Fatalf("canonical lib reported failures for %s\n%s", e.Name(), out)
			}
			if !strings.Contains(out, "[PASS]") {
				t.Fatalf("canonical lib produced no [PASS] lines for %s\n%s", e.Name(), out)
			}
		})
	}
	if any == 0 {
		t.Fatalf("no conformance scenarios found under %s", root)
	}
}

func TestCanonicalLib_FlagsW4OnAllStatuses(t *testing.T) {
	lib := canonicalLibPath()
	if lib == "" {
		t.Skip("canonical lib not found")
	}
	if _, err := exec.LookPath("python3"); err != nil {
		t.Skip("python3 not on PATH")
	}
	for _, status := range []string{"completed", "cancelled", "incomplete", "failed"} {
		t.Run(status, func(t *testing.T) {
			trace := []TraceEvent{
				{"type": "session.created", "session": map[string]any{"id": "sess_1"}},
				{"type": "response.created", "response": map[string]any{"id": "resp_1"}},
				{"type": "response.done", "response": map[string]any{
					"id":     "resp_1",
					"status": status,
				}},
			}
			out, runErr := runCanonicalAgainst(t, lib, trace)
			if runErr == nil {
				t.Fatalf("canonical must flag W4 for status=%s; clean exit. output=\n%s", status, out)
			}
			if !strings.Contains(out, "audio_end_ms") {
				t.Fatalf("canonical output for status=%s lacked audio_end_ms diagnostic:\n%s", status, out)
			}
		})
	}
}
