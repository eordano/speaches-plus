package realtime

import (
	"context"
	"encoding/json"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/coder/websocket"

	"github.com/eordano/speaches-plus-go/internal/inspect"
)

func TestInspectEOU_LaneAndKindOnSpec(t *testing.T) {
	_, r, dir := newInspectTestServer(t)
	ts := httptest.NewServer(r)
	defer ts.Close()

	sid := "sess_eou_lane"
	relay := inspect.NewRelay(sid, dir)
	inspect.Register(sid, fakeSessionState{}, relay)
	defer inspect.Unregister(sid)

	score := float32(0.82)
	thr := float32(0.5)
	dly := 700
	curve := float32(12.0)
	inspect.EmitEOU(relay, context.Background(), "scored", inspect.EOUFields{
		EouKind:   "text",
		Score:     &score,
		Threshold: &thr,
		Language:  "en",
		CurveK:    &curve,
		DelayMs:   &dly,
	})

	wsURL := "ws" + strings.TrimPrefix(ts.URL, "http") + "/v1/inspect/" + sid
	conn, _, err := websocket.Dial(context.Background(), wsURL, &websocket.DialOptions{Subprotocols: []string{"inspect"}})
	if err != nil {
		t.Fatalf("dial: %v", err)
	}
	defer conn.CloseNow()

	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()
	_, data, err := conn.Read(ctx)
	if err != nil {
		t.Fatalf("read: %v", err)
	}
	var ev inspect.Event
	if err := json.Unmarshal(data, &ev); err != nil {
		t.Fatalf("decode: %v", err)
	}
	if ev.Lane != inspect.LaneEOU {
		t.Fatalf("expected lane=eou, got %q", ev.Lane)
	}
	if ev.Kind != "scored" {
		t.Fatalf("expected kind=scored, got %q", ev.Kind)
	}
	if ev.Payload["eou_kind"] != "text" {
		t.Fatalf("expected payload.eou_kind=text, got %v", ev.Payload["eou_kind"])
	}
	for _, k := range []string{"score", "threshold", "language", "curve_k", "delay_ms"} {
		if _, ok := ev.Payload[k]; !ok {
			t.Fatalf("expected payload.%s present, got %+v", k, ev.Payload)
		}
	}
}

func TestInspectEOU_HardCapAndCancelledShape(t *testing.T) {
	dir := t.TempDir()
	sid := "sess_eou_kinds"
	relay := inspect.NewRelay(sid, dir)
	inspect.Register(sid, fakeSessionState{}, relay)
	defer inspect.Unregister(sid)

	sub := relay.Subscribe()
	defer sub.Close()

	inspect.EmitEOU(relay, context.Background(), "hard_cap_fired", inspect.EOUFields{
		EouKind:      "text",
		HardCapPhase: "during_eou",
	})
	inspect.EmitEOU(relay, context.Background(), "cancelled", inspect.EOUFields{
		EouKind:     "fusion",
		CancelledBy: "speech_started",
	})

	read := func() inspect.Event {
		select {
		case line := <-sub.Channel():
			var ev inspect.Event
			if err := json.Unmarshal(line, &ev); err != nil {
				t.Fatalf("decode: %v", err)
			}
			return ev
		case <-time.After(2 * time.Second):
			t.Fatalf("read timeout")
			return inspect.Event{}
		}
	}

	hc := read()
	if hc.Lane != inspect.LaneEOU || hc.Kind != "hard_cap_fired" {
		t.Fatalf("expected eou/hard_cap_fired, got %+v", hc)
	}
	if hc.Payload["hard_cap_phase"] != "during_eou" || hc.Payload["eou_kind"] != "text" {
		t.Fatalf("hard_cap payload missing required fields: %+v", hc.Payload)
	}

	cnc := read()
	if cnc.Lane != inspect.LaneEOU || cnc.Kind != "cancelled" {
		t.Fatalf("expected eou/cancelled, got %+v", cnc)
	}
	if cnc.Payload["cancelled_by"] != "speech_started" || cnc.Payload["eou_kind"] != "fusion" {
		t.Fatalf("cancelled payload missing required fields: %+v", cnc.Payload)
	}
}

func TestInspectRelay_DroppedEventInjectedOnOverflow(t *testing.T) {
	dir := t.TempDir()
	sid := "sess_drop"
	relay := inspect.NewRelayWithCap(sid, dir, 4)
	inspect.Register(sid, fakeSessionState{}, relay)
	defer inspect.Unregister(sid)

	sub := relay.Subscribe()
	defer sub.Close()

	for i := 0; i < 12; i++ {
		inspect.Emit(relay, context.Background(), inspect.LaneTurn, "noise",
			nil, map[string]any{"i": i})
	}

	deadline := time.Now().Add(2 * time.Second)
	sawDropped := false
	for time.Now().Before(deadline) && !sawDropped {
		select {
		case line := <-sub.Channel():
			var ev inspect.Event
			if json.Unmarshal(line, &ev) != nil {
				continue
			}
			if ev.Lane == inspect.LaneError && ev.Kind == "dropped" {
				sawDropped = true
			}
		case <-time.After(100 * time.Millisecond):
		}
	}
	if !sawDropped {
		t.Fatalf("expected error.dropped event after overflow; relay.dropped=%d", relay.DroppedCount())
	}
}

func TestInspectStream_CanonicalRouteMatchesAlias(t *testing.T) {
	_, r, dir := newInspectTestServer(t)
	ts := httptest.NewServer(r)
	defer ts.Close()

	sid := "sess_alias"
	relay := inspect.NewRelay(sid, dir)
	inspect.Register(sid, fakeSessionState{}, relay)
	defer inspect.Unregister(sid)

	for _, suffix := range []string{"/v1/inspect/" + sid, "/v1/inspect/" + sid + "/stream"} {
		wsURL := "ws" + strings.TrimPrefix(ts.URL, "http") + suffix
		conn, _, err := websocket.Dial(context.Background(), wsURL, &websocket.DialOptions{Subprotocols: []string{"inspect"}})
		if err != nil {
			t.Fatalf("dial %s: %v", suffix, err)
		}
		conn.CloseNow()
	}
}
