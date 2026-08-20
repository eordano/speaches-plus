package realtime

import (
	"encoding/json"
	"testing"
)

func ev(t *testing.T, src string) TraceEvent {
	t.Helper()
	var m TraceEvent
	if err := json.Unmarshal([]byte(src), &m); err != nil {
		t.Fatalf("bad event literal: %v\n%s", err, src)
	}
	return m
}

func TestCanonicalize_RewritesIdsAndTimestamps(t *testing.T) {
	in := []TraceEvent{
		ev(t, `{"type":"session.created","ts_ms":1700000000,"event_id":"evt_AAA","session_id":"sess_X"}`),
		ev(t, `{"type":"input_audio_buffer.speech_started","ts_ms":1700000050,"event_id":"evt_BBB","item_id":"item_Y"}`),
		ev(t, `{"type":"input_audio_buffer.speech_stopped","ts_ms":1700000150,"event_id":"evt_CCC","item_id":"item_Y"}`),
	}
	canon := CanonicalizeTrace(in)
	if len(canon) != 3 {
		t.Fatalf("len: %d", len(canon))
	}
	if canon[0]["ts_ms"].(int) != 0 {
		t.Fatalf("ts_ms must be ordinal index: %v", canon[0]["ts_ms"])
	}
	if canon[2]["ts_ms"].(int) != 2 {
		t.Fatalf("ts_ms[2]: %v", canon[2]["ts_ms"])
	}
	if canon[1]["item_id"] != canon[2]["item_id"] {
		t.Fatalf("cross-event identity not preserved: %v vs %v", canon[1]["item_id"], canon[2]["item_id"])
	}
	if canon[0]["session_id"] != "sess_1" {
		t.Fatalf("session_id rewrite: %v", canon[0]["session_id"])
	}
}

func TestAssertTraceInvariants_W1Violation(t *testing.T) {
	trace := []TraceEvent{
		ev(t, `{"type":"response.created","response":{"id":"resp_1"}}`),
	}
	v := AssertTraceInvariants(trace)
	if len(v) == 0 {
		t.Fatalf("expected W1 violation; got none")
	}
}

func TestAssertTraceInvariants_W2Violation(t *testing.T) {
	trace := []TraceEvent{
		ev(t, `{"type":"response.output_audio.delta","response_id":"resp_1","delta":"x"}`),
		ev(t, `{"type":"response.created","response":{"id":"resp_1"}}`),
		ev(t, `{"type":"response.done","response":{"id":"resp_1","status":"completed"}}`),
	}
	v := AssertTraceInvariants(trace)
	if len(v) == 0 {
		t.Fatalf("expected W2 violation; got none")
	}
}

func TestAssertTraceInvariants_W4Violation(t *testing.T) {
	trace := []TraceEvent{
		ev(t, `{"type":"response.created","response":{"id":"resp_1"}}`),
		ev(t, `{"type":"response.done","response":{"id":"resp_1","status":"cancelled"}}`),
	}
	v := AssertTraceInvariants(trace)
	if len(v) == 0 {
		t.Fatalf("expected W4 violation (no audio_end_ms on cancelled); got none")
	}
}

func TestAssertTraceInvariants_W4PassWithAudioEndMs(t *testing.T) {
	trace := []TraceEvent{
		ev(t, `{"type":"response.created","response":{"id":"resp_1"}}`),
		ev(t, `{"type":"response.done","response":{"id":"resp_1","status":"cancelled","audio_end_ms":420}}`),
	}
	v := AssertTraceInvariants(trace)
	for _, vio := range v {
		if vio == "" {
			continue
		}
		if len(v) > 0 && vio[:2] == "W4" {
			t.Fatalf("unexpected W4 violation: %s", vio)
		}
	}
}

func TestAssertTraceInvariants_W4PerStatus_v2D4Regression(t *testing.T) {
	for _, status := range []string{"completed", "cancelled", "incomplete", "failed"} {
		t.Run(status, func(t *testing.T) {
			trace := []TraceEvent{
				ev(t, `{"type":"response.created","response":{"id":"resp_x"}}`),
				ev(t, `{"type":"response.done","response":{"id":"resp_x","status":"`+status+`"}}`),
			}
			v := AssertTraceInvariants(trace)
			found := false
			for _, vio := range v {
				if len(vio) >= 2 && vio[:2] == "W4" {
					found = true
				}
			}
			if !found {
				t.Fatalf("§8.5: W4 must flag missing audio_end_ms for status=%s; got %v", status, v)
			}
		})
	}
}

func TestAssertTraceInvariants_W4PerStatus_PassesWithAudioEndMs(t *testing.T) {
	for _, status := range []string{"completed", "cancelled", "incomplete", "failed"} {
		t.Run(status, func(t *testing.T) {
			trace := []TraceEvent{
				ev(t, `{"type":"response.created","response":{"id":"resp_x"}}`),
				ev(t, `{"type":"response.done","response":{"id":"resp_x","status":"`+status+`","audio_end_ms":150}}`),
			}
			v := AssertTraceInvariants(trace)
			for _, vio := range v {
				if len(vio) >= 2 && vio[:2] == "W4" {
					t.Fatalf("unexpected W4 violation for status=%s: %s", status, vio)
				}
			}
		})
	}
}

func TestAssertTraceInvariants_W3Order(t *testing.T) {
	good := []TraceEvent{
		ev(t, `{"type":"input_audio_buffer.speech_stopped","item_id":"item_a"}`),
		ev(t, `{"type":"input_audio_buffer.committed","item_id":"item_a"}`),
		ev(t, `{"type":"conversation.item.added","item":{"id":"item_a"}}`),
	}
	v := AssertTraceInvariants(good)
	for _, vio := range v {
		if len(vio) >= 2 && vio[:2] == "W3" {
			t.Fatalf("good order flagged W3: %s", vio)
		}
	}

	bad := []TraceEvent{
		ev(t, `{"type":"input_audio_buffer.committed","item_id":"item_a"}`),
		ev(t, `{"type":"conversation.item.added","item":{"id":"item_a"}}`),
		ev(t, `{"type":"input_audio_buffer.speech_stopped","item_id":"item_a"}`),
	}
	v = AssertTraceInvariants(bad)
	if len(v) == 0 {
		t.Fatalf("bad order should flag W3")
	}
}

func TestTraceDiff_IdenticalTraces(t *testing.T) {
	a := []TraceEvent{
		ev(t, `{"type":"x","event_id":"evt_1"}`),
		ev(t, `{"type":"y","event_id":"evt_2"}`),
	}
	b := []TraceEvent{
		ev(t, `{"type":"x","event_id":"evt_AAA"}`),
		ev(t, `{"type":"y","event_id":"evt_BBB"}`),
	}
	if d := TraceDiff(CanonicalizeTrace(a), CanonicalizeTrace(b)); d != -1 {
		t.Fatalf("traces should match after canonicalization; diff at %d", d)
	}
}

func TestCanonicalize_RoundsScore(t *testing.T) {
	in := []TraceEvent{
		ev(t, `{"type":"eou.verdict","eou.score":0.7395412381}`),
	}
	canon := CanonicalizeTrace(in)
	got, ok := canon[0]["eou.score"].(float64)
	if !ok {
		t.Fatalf("eou.score type: %T", canon[0]["eou.score"])
	}
	if got != 0.74 {
		t.Fatalf("score should round to 3 decimals; got %v", got)
	}
}
