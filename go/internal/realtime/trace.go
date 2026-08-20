package realtime

import (
	"encoding/json"
	"fmt"
	"math"
	"regexp"
	"sort"
	"strings"
)

type TraceEvent map[string]any

type CanonicalTrace []TraceEvent

func CanonicalizeTrace(events []TraceEvent) CanonicalTrace {
	out := make(CanonicalTrace, len(events))
	idMap := map[string]string{}
	idCounters := map[string]int{}

	rewriteID := func(s string) string {
		if s == "" {
			return s
		}
		if v, ok := idMap[s]; ok {
			return v
		}
		typ := idType(s)
		idCounters[typ]++
		canonical := fmt.Sprintf("%s_%d", typ, idCounters[typ])
		idMap[s] = canonical
		return canonical
	}

	for i, ev := range events {
		canon := make(TraceEvent, len(ev))
		for k, v := range ev {
			canon[k] = v
		}

		for _, key := range []string{"ts_ms", "created_at"} {
			if _, ok := canon[key]; ok {
				canon[key] = i
			}
		}

		for _, key := range []string{"event_id", "id", "item_id", "response_id", "previous_item_id", "session_id"} {
			if v, ok := canon[key].(string); ok {
				canon[key] = rewriteID(v)
			}
		}

		for _, key := range []string{"eou.score", "vad.probability", "p", "score"} {
			if v, ok := canon[key].(float64); ok {
				canon[key] = roundTo(v, 3)
			}
			if v, ok := canon[key].(float32); ok {
				canon[key] = roundTo(float64(v), 3)
			}
		}

		for _, key := range []string{"audio", "delta"} {
			if v, ok := canon[key].(string); ok && looksLikeBase64Audio(key, v) {
				canon[key] = map[string]any{"audio_bytes": (len(v) * 3) / 4}
			}
		}
		out[i] = canon
	}
	return out
}

func AssertTraceInvariants(trace []TraceEvent) []string {
	var violations []string

	createdAt := map[string]int{}
	doneAt := map[string]int{}
	for i, ev := range trace {
		if ev["type"] == "response.created" {
			id, _ := getNested(ev, "response", "id")
			if id != "" {
				if _, dup := createdAt[id]; dup {
					violations = append(violations, fmt.Sprintf("W1: duplicate response.created id=%s at %d", id, i))
				}
				createdAt[id] = i
			}
		}
		if ev["type"] == "response.done" {
			id, _ := getNested(ev, "response", "id")
			if id != "" {
				if _, dup := doneAt[id]; dup {
					violations = append(violations, fmt.Sprintf("W1: duplicate response.done id=%s at %d", id, i))
				}
				doneAt[id] = i
			}
		}
	}
	for id := range createdAt {
		if _, ok := doneAt[id]; !ok {
			violations = append(violations, fmt.Sprintf("W1: response.created id=%s without matching response.done", id))
		}
	}
	for id := range doneAt {
		if _, ok := createdAt[id]; !ok {
			violations = append(violations, fmt.Sprintf("W1: response.done id=%s without matching response.created", id))
		}
	}

	for i, ev := range trace {
		t, _ := ev["type"].(string)
		if !strings.HasPrefix(t, "response.") {
			continue
		}
		if t == "response.created" || t == "response.done" {
			continue
		}
		respID, _ := ev["response_id"].(string)
		if respID == "" {
			continue
		}
		c, hasC := createdAt[respID]
		d, hasD := doneAt[respID]
		if !hasC || i <= c {
			violations = append(violations, fmt.Sprintf("W2: %s for resp=%s before response.created at index %d", t, respID, i))
		}
		if hasD && i > d {
			violations = append(violations, fmt.Sprintf("W6: %s for resp=%s after response.done at index %d", t, respID, i))
		}
	}

	startedAt := map[string]int{}
	stoppedAt := map[string]int{}
	committedAt := map[string]int{}
	itemCreatedAt := map[string]int{}
	for i, ev := range trace {
		t, _ := ev["type"].(string)
		switch t {
		case "input_audio_buffer.speech_started":
			id, _ := ev["item_id"].(string)
			if id != "" {
				startedAt[id] = i
			}
		case "input_audio_buffer.speech_stopped":
			id, _ := ev["item_id"].(string)
			if id != "" {
				stoppedAt[id] = i
			}
		case "input_audio_buffer.committed":
			id, _ := ev["item_id"].(string)
			if id != "" {
				committedAt[id] = i
			}
		case "conversation.item.added":
			id, _ := getNested(ev, "item", "id")
			if id != "" {
				itemCreatedAt[id] = i
			}
		}
	}
	for id, ci := range committedAt {
		if si, ok := stoppedAt[id]; ok && si > ci {
			violations = append(violations, fmt.Sprintf("W3: committed(item=%s) before speech_stopped", id))
		}
		if ii, ok := itemCreatedAt[id]; ok && ii < ci {
			violations = append(violations, fmt.Sprintf("W3: conversation.item.added(item=%s) before committed", id))
		}
	}

	for _, ev := range trace {
		if ev["type"] != "response.done" {
			continue
		}
		status, _ := getNested(ev, "response", "status")
		switch status {
		case "completed", "cancelled", "incomplete", "failed":
		default:
			id, _ := getNested(ev, "response", "id")
			violations = append(violations, fmt.Sprintf("W4: response.done id=%s has unknown status=%q", id, status))
			continue
		}
		if _, ok := getNestedAny(ev, "response", "audio_end_ms"); !ok {
			id, _ := getNested(ev, "response", "id")
			violations = append(violations, fmt.Sprintf("W4: response.done(status=%s id=%s) missing audio_end_ms", status, id))
		}
	}

	cancelledItems := map[string]bool{}
	for _, ev := range trace {
		if ev["type"] != "response.done" {
			continue
		}
		status, _ := getNested(ev, "response", "status")
		if status != "cancelled" && status != "incomplete" {
			continue
		}
		out, ok := getNestedAny(ev, "response", "output")
		if !ok {
			continue
		}
		arr, ok := out.([]any)
		if !ok {
			continue
		}
		for _, it := range arr {
			m, ok := it.(map[string]any)
			if !ok {
				continue
			}
			id, _ := m["id"].(string)
			if id != "" {
				cancelledItems[id] = true
			}
		}
	}
	for i, ev := range trace {
		if ev["type"] != "conversation.item.assistant_truncated" {
			continue
		}
		id, _ := ev["item_id"].(string)
		if id == "" {
			continue
		}
		if !cancelledItems[id] {
			violations = append(violations,
				fmt.Sprintf("W7: conversation.item.assistant_truncated(item=%s) at index %d without prior response.done(cancelled|incomplete) referencing it", id, i))
		}
	}

	clientCreated := map[string]int{}
	serverCreated := map[string]int{}
	for i, ev := range trace {
		t, _ := ev["type"].(string)
		switch t {
		case "conversation.item.create":
			id, _ := getNested(ev, "item", "id")
			if id != "" {
				clientCreated[id] = i
			}
		case "conversation.item.added":
			id, _ := getNested(ev, "item", "id")
			if id != "" {
				serverCreated[id] = i
			}
		}
	}
	for id, ci := range clientCreated {
		si, ok := serverCreated[id]
		if !ok {
			violations = append(violations,
				fmt.Sprintf("W8: conversation.item.create(id=%s) at index %d not followed by conversation.item.added", id, ci))
		} else if si < ci {
			violations = append(violations,
				fmt.Sprintf("W8: conversation.item.added(id=%s) at index %d precedes its conversation.item.create at index %d", id, si, ci))
		}
	}

	return violations
}

func TraceDiff(a, b CanonicalTrace) int {
	n := len(a)
	if len(b) < n {
		n = len(b)
	}
	for i := 0; i < n; i++ {
		if !mapsEqual(a[i], b[i]) {
			return i
		}
	}
	if len(a) != len(b) {
		return n
	}
	return -1
}

func mapsEqual(a, b TraceEvent) bool {
	if len(a) != len(b) {
		return false
	}
	keys := make([]string, 0, len(a))
	for k := range a {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	for _, k := range keys {
		bv, ok := b[k]
		if !ok {
			return false
		}
		am, _ := json.Marshal(a[k])
		bm, _ := json.Marshal(bv)
		if string(am) != string(bm) {
			return false
		}
	}
	return true
}

func roundTo(v float64, decimals int) float64 {
	mult := math.Pow(10, float64(decimals))
	return math.Round(v*mult) / mult
}

var idTypeRe = regexp.MustCompile(`^(evt|item|resp|sess)_`)

func idType(s string) string {
	m := idTypeRe.FindStringSubmatch(s)
	if len(m) == 2 {
		return m[1]
	}
	return "id"
}

func looksLikeBase64Audio(key, val string) bool {
	if key == "audio" {
		return len(val) > 0
	}

	if len(val) < 64 || strings.ContainsAny(val, " ,.!?") {
		return false
	}
	for _, r := range val {
		if !((r >= 'A' && r <= 'Z') || (r >= 'a' && r <= 'z') || (r >= '0' && r <= '9') || r == '+' || r == '/' || r == '=') {
			return false
		}
	}
	return true
}

func getNested(ev TraceEvent, path ...string) (string, bool) {
	v, ok := getNestedAny(ev, path...)
	if !ok {
		return "", false
	}
	s, ok := v.(string)
	return s, ok
}

func getNestedAny(ev TraceEvent, path ...string) (any, bool) {
	var cur any = map[string]any(ev)
	for _, key := range path {
		m, ok := cur.(map[string]any)
		if !ok {
			return nil, false
		}
		cur, ok = m[key]
		if !ok {
			return nil, false
		}
	}
	return cur, true
}
