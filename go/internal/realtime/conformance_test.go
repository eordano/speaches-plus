package realtime

import (
	"bufio"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"testing"
)

func loadJSONL(t *testing.T, path string) []TraceEvent {
	t.Helper()
	f, err := os.Open(path)
	if err != nil {
		t.Fatalf("open %s: %v", path, err)
	}
	defer f.Close()
	var events []TraceEvent
	sc := bufio.NewScanner(f)
	sc.Buffer(make([]byte, 64*1024), 1<<20)
	for ln := 1; sc.Scan(); ln++ {
		line := strings.TrimSpace(sc.Text())
		if line == "" || strings.HasPrefix(line, "//") || strings.HasPrefix(line, "#") {
			continue
		}
		var ev TraceEvent
		if err := json.Unmarshal([]byte(line), &ev); err != nil {
			t.Fatalf("%s:%d: invalid JSON: %v\nline=%s", path, ln, err, line)
		}
		events = append(events, ev)
	}
	if err := sc.Err(); err != nil {
		t.Fatalf("scan %s: %v", path, err)
	}
	return events
}

type confEmitter struct {
	trace []TraceEvent
}

func (e *confEmitter) emit(ev TraceEvent) { e.trace = append(e.trace, ev) }

func replay(t *testing.T, ops []TraceEvent) []TraceEvent {
	t.Helper()
	var s phaseState
	em := &confEmitter{}

	sessionStarted := false

	maybeStartSession := func() {
		if sessionStarted {
			return
		}
		s.startSession()
		em.emit(TraceEvent{
			"type":    string(SETSessionCreated),
			"session": map[string]any{"id": "sess_1"},
		})
		sessionStarted = true
	}

	for i, op := range ops {
		opName, _ := op["op"].(string)
		switch opName {
		case "markActive":
			maybeStartSession()

		case "session_update":
			maybeStartSession()
			if err := s.updateSession(); err != nil {
				em.emit(TraceEvent{
					"type":    string(SETError),
					"code":    "session_update_invalid",
					"message": err.Error(),
				})
			}
			em.emit(TraceEvent{
				"type":    string(SETSessionUpdated),
				"session": map[string]any{"id": "sess_1"},
			})

		case "session_update_invalid":
			maybeStartSession()
			code, _ := op["code"].(string)
			if code == "" {
				code = "session_update_invalid"
			}
			msg, _ := op["message"].(string)
			em.emit(TraceEvent{
				"type":    string(SETError),
				"code":    code,
				"message": msg,
			})
			em.emit(TraceEvent{
				"type":    string(SETSessionUpdated),
				"session": map[string]any{"id": "sess_1"},
			})

		case "vad_speech_start":
			maybeStartSession()
			itemID, _ := op["item_id"].(string)
			startMs := numFromAny(op["start_ms"])
			_, beforeVad, beforeResp := s.snapshot()
			eff := s.onVadSpeechStart(itemID, startMs, nil)
			if eff.cancel.cancelled {
				switch beforeResp.Kind() {
				case respKindCreated, respKindStreaming, respKindDrain:
					em.emit(TraceEvent{
						"type": string(SETResponseDone),
						"response": map[string]any{
							"id":           eff.cancel.id,
							"status":       "cancelled",
							"audio_end_ms": eff.cancel.playedMs,
							"output": []any{
								map[string]any{"id": eff.cancel.itemID},
							},
						},
					})
					em.emit(TraceEvent{
						"type":         string(SETConversationItemAssistantTruncated),
						"item_id":      eff.cancel.itemID,
						"audio_end_ms": eff.cancel.playedMs,
					})
				}
			}
			if eff.cancelTimer {
				continue
			}
			if _, wasSpeaking := beforeVad.(VadSpeaking); wasSpeaking {
				continue
			}
			_, vad, _ := s.snapshot()
			if vs, ok := vad.(VadSpeaking); ok {
				em.emit(TraceEvent{
					"type":           string(SETInputBufferSpeechStarted),
					"item_id":        string(vs.ItemID),
					"audio_start_ms": int64(vs.AudioStartMs),
				})
			}

		case "vad_speech_end":
			maybeStartSession()
			endMs := numFromAny(op["end_ms"])
			itemID, _, ok := s.onVadSpeechEnd(Millis(endMs))
			if ok {
				em.emit(TraceEvent{
					"type":         string(SETInputBufferSpeechStopped),
					"item_id":      string(itemID),
					"audio_end_ms": endMs,
				})
			}

		case "commit_fire":
			maybeStartSession()
			eff := s.onCommitTimerFire()
			if !eff.committed {
				continue
			}
			em.emit(TraceEvent{
				"type":    string(SETInputBufferCommitted),
				"item_id": string(eff.itemID),
			})
			em.emit(TraceEvent{
				"type": string(SETConversationItemAdded),
				"item": map[string]any{"id": string(eff.itemID), "role": "user"},
			})

		case "transcription_complete":
			maybeStartSession()
			itemID, _ := op["item_id"].(string)
			transcript, _ := op["transcript"].(string)
			autoResp := true
			if v, ok := op["auto_response"].(bool); ok {
				autoResp = v
			}
			s.onTranscriptionComplete(ItemID(itemID), transcript, autoResp)
			em.emit(TraceEvent{
				"type":       string(SETInputAudioTranscriptionCompleted),
				"item_id":    itemID,
				"transcript": transcript,
			})

		case "response_create":
			maybeStartSession()
			respID, _ := op["resp_id"].(string)
			itemID, _ := op["item_id"].(string)
			instructions, hasInstr := op["instructions"].(string)
			if _, err := s.onResponseCreate(ResponseID(respID), ItemID(itemID)); err != nil {
				t.Fatalf("op %d response_create: %v", i, err)
			}
			ev := TraceEvent{
				"type":     string(SETResponseCreated),
				"response": map[string]any{"id": respID},
			}
			if hasInstr {
				ev["response"].(map[string]any)["instructions"] = instructions
			}
			em.emit(ev)

		case "audio_delta":
			respID, _ := op["resp_id"].(string)
			audioBytes := int(numFromAny(op["audio_bytes"]))
			if audioBytes <= 0 {
				audioBytes = 1024
			}
			em.emit(TraceEvent{
				"type":        string(SETResponseOutputAudioDelta),
				"response_id": respID,
				"audio":       map[string]any{"audio_bytes": audioBytes},
			})

		case "llm_complete":
			epoch := Epoch(numFromAny(op["epoch"]))
			transcript, _ := op["transcript"].(string)
			plannedMs := DurationMs(numFromAny(op["planned_ms"]))
			s.onUpstreamDelta(epoch, transcript, plannedMs)
			s.onLLMComplete(epoch)

		case "audio_drained":
			epoch := Epoch(numFromAny(op["epoch"]))
			playedMs := Millis(numFromAny(op["played_ms"]))
			s.updatePlayedMs(epoch, playedMs)
			if !s.onAudioDrained(epoch) {
				continue
			}
			_, _, after := s.snapshot()
			if rf, ok := after.(RespFinalized); ok {
				em.emit(TraceEvent{
					"type": string(SETResponseOutputAudioDone),
					"response": map[string]any{
						"id":           string(rf.ID),
						"audio_end_ms": int64(rf.PlayedMs),
					},
				})
				em.emit(TraceEvent{
					"type": string(SETResponseDone),
					"response": map[string]any{
						"id":           string(rf.ID),
						"status":       rf.Status.String(),
						"audio_end_ms": int64(rf.PlayedMs),
					},
				})
				_, _, _, _ = s.onResponseDoneEmitted(rf.Epoch)
			}

		case "response_failed":
			epoch := Epoch(numFromAny(op["epoch"]))
			playedMs := numFromAny(op["played_ms"])
			reason, _ := op["reason"].(string)
			if reason == "" {
				reason = "llm_error"
			}
			_, _, beforeResp := s.snapshot()
			var respID, itemID string
			switch r := beforeResp.(type) {
			case RespCreated:
				respID, itemID = string(r.ID), string(r.ItemID)
			case RespStreaming:
				respID, itemID = string(r.ID), string(r.ItemID)
			case RespDrain:
				respID, itemID = string(r.ID), string(r.ItemID)
			}
			if respID == "" {
				t.Fatalf("op %d response_failed: no in-flight response", i)
			}
			em.emit(TraceEvent{
				"type": string(SETResponseDone),
				"response": map[string]any{
					"id":             respID,
					"status":         "failed",
					"audio_end_ms":   playedMs,
					"status_details": map[string]any{"reason": reason},
					"output":         []any{map[string]any{"id": itemID}},
				},
			})
			_, _, _ = s.onUpstreamComplete(epoch)

		case "response_drain_cap_expired":
			epoch := Epoch(numFromAny(op["epoch"]))
			transcript, _ := op["transcript"].(string)
			plannedMs := DurationMs(numFromAny(op["planned_ms"]))
			playedMs := Millis(numFromAny(op["played_ms"]))
			s.onUpstreamDelta(epoch, transcript, plannedMs)
			s.onLLMComplete(epoch)
			s.updatePlayedMs(epoch, playedMs)
			_, _, after := s.snapshot()
			rd, ok := after.(RespDrain)
			if !ok {
				continue
			}
			em.emit(TraceEvent{
				"type": string(SETResponseDone),
				"response": map[string]any{
					"id":             string(rd.ID),
					"status":         "incomplete",
					"audio_end_ms":   int64(playedMs),
					"status_details": map[string]any{"reason": "drain_cap"},
					"output":         []any{map[string]any{"id": string(rd.ItemID)}},
				},
			})
			_, _, _ = s.onUpstreamComplete(epoch)

		default:
			t.Fatalf("op %d: unknown op %q", i, opName)
		}
	}
	return em.trace
}

func numFromAny(v any) int64 {
	switch x := v.(type) {
	case float64:
		return int64(x)
	case int64:
		return x
	case int:
		return int64(x)
	}
	return 0
}

func assertTracesEqual(t *testing.T, expected, actual CanonicalTrace) {
	t.Helper()
	idx := TraceDiff(expected, actual)
	if idx < 0 {
		return
	}
	var b strings.Builder
	fmt.Fprintf(&b, "trace diverges at index %d (expected=%d events, actual=%d events)\n",
		idx, len(expected), len(actual))
	dump := func(label string, tr CanonicalTrace) {
		fmt.Fprintf(&b, "%s:\n", label)
		for i, ev := range tr {
			marker := "  "
			if i == idx {
				marker = "> "
			}
			fmt.Fprintf(&b, "%s%2d %s\n", marker, i, jsonOneLine(ev))
		}
	}
	dump("expected", expected)
	dump("actual  ", actual)
	t.Fatalf("%s", b.String())
}

func jsonOneLine(ev TraceEvent) string {
	keys := make([]string, 0, len(ev))
	for k := range ev {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	out := make(map[string]any, len(ev))
	for _, k := range keys {
		out[k] = ev[k]
	}
	b, _ := json.Marshal(out)
	return string(b)
}

func corpusRoot(t *testing.T) string {
	t.Helper()
	candidates := []string{
		"../../../conformance/fixtures",
		"../../conformance/fixtures",
		"../../conformance",
		"../../../conformance",
	}
	for _, c := range candidates {
		if st, err := os.Stat(c); err == nil && st.IsDir() {
			abs, _ := filepath.Abs(c)
			return abs
		}
	}
	t.Fatalf("conformance corpus directory not found (looked in %v)", candidates)
	return ""
}

func TestConformanceCorpus(t *testing.T) {
	root := corpusRoot(t)
	entries, err := os.ReadDir(root)
	if err != nil {
		t.Fatalf("read corpus dir %s: %v", root, err)
	}
	scenarios := 0
	for _, e := range entries {
		if !e.IsDir() {
			continue
		}
		name := e.Name()
		if strings.HasPrefix(name, ".") {
			continue
		}
		dir := filepath.Join(root, name)
		inputPath := filepath.Join(dir, "input.jsonl")
		expectedPath := filepath.Join(dir, "expected.jsonl")
		if _, err := os.Stat(inputPath); err != nil {
			continue
		}
		if _, err := os.Stat(expectedPath); err != nil {
			continue
		}
		scenarios++
		t.Run(name, func(t *testing.T) {
			input := loadJSONL(t, inputPath)
			expected := loadJSONL(t, expectedPath)
			actual := replay(t, input)
			canonActual := CanonicalizeTrace(actual)
			canonExpected := CanonicalizeTrace(expected)
			assertTracesEqual(t, canonExpected, canonActual)
			if vios := AssertTraceInvariants(actual); len(vios) != 0 {
				t.Fatalf("wire invariants failed for scenario %s: %s",
					name, strings.Join(vios, "; "))
			}
		})
	}
	if scenarios == 0 {
		t.Fatalf("no conformance scenarios found under %s", root)
	}
	t.Logf("conformance: %d scenarios passed", scenarios)
}
