package realtime

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"strings"
	"testing"
	"time"

	"github.com/coder/websocket"
	"github.com/go-chi/chi/v5"

	"github.com/eordano/speaches-plus-go/internal/audio"
	"github.com/eordano/speaches-plus-go/internal/inspect"
)

func newInspectTestServer(t *testing.T) (*Server, *chi.Mux, string) {
	t.Helper()
	dir := t.TempDir()
	srv := &Server{cfg: Config{InspectSessionDir: dir}}
	r := chi.NewRouter()
	r.Get("/v1/inspect/sessions", srv.HandleInspectListSessions)
	r.Get("/v1/inspect/sessions/history", srv.HandleInspectListHistory)
	r.Get("/v1/inspect/sessions/history/{sid}", srv.HandleInspectGetHistory)
	r.Get("/v1/inspect/sessions/{sid}/audio", srv.HandleInspectGetAudio)
	r.Get("/v1/inspect/{sid}/stream", srv.HandleInspectStream)
	r.Get("/v1/inspect/{sid}", srv.HandleInspectStream)
	return srv, r, dir
}

type fakeSessionState struct {
	state string
	model string
}

func (f fakeSessionState) State() string { return f.state }
func (f fakeSessionState) Model() string { return f.model }

func TestInspectStream_DeliversLaneKindEvent(t *testing.T) {
	_, r, dir := newInspectTestServer(t)
	ts := httptest.NewServer(r)
	defer ts.Close()

	sid := "sess_test_1"
	relay := inspect.NewRelay(sid, dir)
	inspect.Register(sid, fakeSessionState{state: "active", model: "gpt-test"}, relay)
	defer inspect.Unregister(sid)

	wsURL := "ws" + strings.TrimPrefix(ts.URL, "http") + "/v1/inspect/" + sid + "/stream"
	dialCtx, dialCancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer dialCancel()
	conn, _, err := websocket.Dial(dialCtx, wsURL, &websocket.DialOptions{Subprotocols: []string{"inspect"}})
	if err != nil {
		t.Fatalf("dial stream: %v", err)
	}
	defer conn.CloseNow()

	deadline := time.Now().Add(2 * time.Second)
	for time.Now().Before(deadline) {
		if relay.HasSubscribers() {
			break
		}
		time.Sleep(5 * time.Millisecond)
	}
	if !relay.HasSubscribers() {
		t.Fatalf("subscriber never registered with relay")
	}

	inspect.SetTurnID(relay, "turn_1")
	inspect.SetItemID(relay, "item_1")
	inspect.Emit(relay, context.Background(), inspect.LaneVAD, "confirmed_start",
		nil, map[string]any{"cursor": 7000})

	readCtx, readCancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer readCancel()
	_, data, err := conn.Read(readCtx)
	if err != nil {
		t.Fatalf("read: %v", err)
	}
	var ev inspect.Event
	if err := json.Unmarshal(data, &ev); err != nil {
		t.Fatalf("decode %q: %v", string(data), err)
	}
	if ev.Lane != inspect.LaneVAD || ev.Kind != "confirmed_start" {
		t.Fatalf("unexpected lane/kind: %+v", ev)
	}
	if ev.Corr.TurnID != "turn_1" || ev.Corr.ItemID != "item_1" {
		t.Fatalf("expected corr turn=turn_1 item=item_1, got %+v", ev.Corr)
	}
	if ev.SessionID != sid || ev.TSWall <= 0 {
		t.Fatalf("missing session_id/ts: %+v", ev)
	}
	if v, ok := ev.Payload["cursor"]; !ok || v != float64(7000) {
		t.Fatalf("expected payload.cursor=7000, got %v", v)
	}
}

func TestInspectStream_ErrorMirror(t *testing.T) {
	_, r, dir := newInspectTestServer(t)
	ts := httptest.NewServer(r)
	defer ts.Close()

	sid := "sess_test_err"
	relay := inspect.NewRelay(sid, dir)
	inspect.Register(sid, fakeSessionState{state: "active"}, relay)
	defer inspect.Unregister(sid)

	wsURL := "ws" + strings.TrimPrefix(ts.URL, "http") + "/v1/inspect/" + sid + "/stream"
	conn, _, err := websocket.Dial(context.Background(), wsURL, &websocket.DialOptions{Subprotocols: []string{"inspect"}})
	if err != nil {
		t.Fatalf("dial: %v", err)
	}
	defer conn.CloseNow()
	for !relay.HasSubscribers() {
		time.Sleep(5 * time.Millisecond)
	}

	inspect.Emit(relay, context.Background(), inspect.LaneSTT, "failed",
		nil, map[string]any{"error": "transcription deadline"})

	var origin, mirror inspect.Event
	for i := 0; i < 2; i++ {
		ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
		_, data, err := conn.Read(ctx)
		cancel()
		if err != nil {
			t.Fatalf("read[%d]: %v", i, err)
		}
		var ev inspect.Event
		if err := json.Unmarshal(data, &ev); err != nil {
			t.Fatalf("decode: %v", err)
		}
		if i == 0 {
			origin = ev
		} else {
			mirror = ev
		}
	}
	if origin.Lane != inspect.LaneSTT || origin.Kind != "failed" {
		t.Fatalf("origin event wrong: %+v", origin)
	}
	if mirror.Lane != inspect.LaneError || mirror.Kind != "raised" {
		t.Fatalf("mirror event wrong: %+v", mirror)
	}
	if mirror.Payload["origin_seq"] != float64(origin.Seq) {
		t.Fatalf("mirror.origin_seq=%v, want %d", mirror.Payload["origin_seq"], origin.Seq)
	}
	if mirror.Payload["error"] != "transcription deadline" {
		t.Fatalf("mirror.error=%v", mirror.Payload["error"])
	}
}

func TestInspectStream_ReplayBufferOnLateSubscribe(t *testing.T) {
	_, r, dir := newInspectTestServer(t)
	ts := httptest.NewServer(r)
	defer ts.Close()

	sid := "sess_test_replay"
	relay := inspect.NewRelay(sid, dir)
	inspect.Register(sid, fakeSessionState{}, relay)
	defer inspect.Unregister(sid)

	for i := 0; i < 3; i++ {
		inspect.Emit(relay, context.Background(), inspect.LaneTurn, "turn_start", nil, map[string]any{"i": i})
	}

	wsURL := "ws" + strings.TrimPrefix(ts.URL, "http") + "/v1/inspect/" + sid + "/stream"
	conn, _, err := websocket.Dial(context.Background(), wsURL, &websocket.DialOptions{Subprotocols: []string{"inspect"}})
	if err != nil {
		t.Fatalf("dial: %v", err)
	}
	defer conn.CloseNow()

	for i := 0; i < 3; i++ {
		ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
		_, data, err := conn.Read(ctx)
		cancel()
		if err != nil {
			t.Fatalf("replay read[%d]: %v", i, err)
		}
		var ev inspect.Event
		_ = json.Unmarshal(data, &ev)
		if ev.Lane != inspect.LaneTurn || ev.Payload["i"] != float64(i) {
			t.Fatalf("expected replay event i=%d, got %+v", i, ev)
		}
	}
}

func TestInspectListSessions_ReturnsLiveMeta(t *testing.T) {
	_, r, dir := newInspectTestServer(t)
	ts := httptest.NewServer(r)
	defer ts.Close()

	sid := "sess_test_meta"
	relay := inspect.NewRelay(sid, dir)
	inspect.Register(sid, fakeSessionState{state: "active", model: "m1"}, relay)
	defer inspect.Unregister(sid)
	inspect.Emit(relay, context.Background(), inspect.LaneTurn, "turn_end", nil, nil)

	resp, err := http.Get(ts.URL + "/v1/inspect/sessions")
	if err != nil {
		t.Fatalf("get sessions: %v", err)
	}
	body, _ := io.ReadAll(resp.Body)
	var meta []inspect.SessionMeta
	if err := json.Unmarshal(body, &meta); err != nil {
		t.Fatalf("decode: %v body=%s", err, string(body))
	}
	var found *inspect.SessionMeta
	for i := range meta {
		if meta[i].ID == sid {
			found = &meta[i]
			break
		}
	}
	if found == nil {
		t.Fatalf("session %s not in list: %s", sid, string(body))
	}
	if found.Model != "m1" || found.State != "active" {
		t.Fatalf("meta wrong: %+v", *found)
	}
	if found.TurnCount != 1 {
		t.Fatalf("expected turn_count=1 (turn_end emitted), got %d", found.TurnCount)
	}
}

func TestInspectHistory_ListsAndStreamsNDJSON(t *testing.T) {
	_, r, dir := newInspectTestServer(t)
	ts := httptest.NewServer(r)
	defer ts.Close()

	sid := "sess_test_hist"
	relay := inspect.NewRelay(sid, dir)
	inspect.Register(sid, fakeSessionState{}, relay)
	inspect.Emit(relay, context.Background(), inspect.LaneVAD, "confirmed_start", nil, nil)
	inspect.Emit(relay, context.Background(), inspect.LaneVAD, "confirmed_stop", nil, nil)
	inspect.Unregister(sid)

	resp, err := http.Get(ts.URL + "/v1/inspect/sessions/history")
	if err != nil {
		t.Fatalf("history list: %v", err)
	}
	body, _ := io.ReadAll(resp.Body)
	var entries []inspect.SessionHistoryEntry
	if err := json.Unmarshal(body, &entries); err != nil {
		t.Fatalf("decode hist: %v", err)
	}
	var hit *inspect.SessionHistoryEntry
	for i := range entries {
		if entries[i].ID == sid {
			hit = &entries[i]
			break
		}
	}
	if hit == nil {
		t.Fatalf("ndjson not in history list: %s", string(body))
	}
	if hit.SizeBytes <= 0 {
		t.Fatalf("expected positive ndjson size, got %d", hit.SizeBytes)
	}

	resp2, err := http.Get(ts.URL + "/v1/inspect/sessions/history/" + sid)
	if err != nil {
		t.Fatalf("history stream: %v", err)
	}
	if resp2.Header.Get("Content-Type") != "application/x-ndjson" {
		t.Fatalf("bad ct: %s", resp2.Header.Get("Content-Type"))
	}
	streamBody, _ := io.ReadAll(resp2.Body)
	if !strings.Contains(string(streamBody), "confirmed_stop") {
		t.Fatalf("ndjson missing event: %s", string(streamBody))
	}
}

func TestInspectStream_AllLanesDelivered(t *testing.T) {
	_, r, dir := newInspectTestServer(t)
	ts := httptest.NewServer(r)
	defer ts.Close()

	sid := "sess_test_all_lanes"
	relay := inspect.NewRelay(sid, dir)
	inspect.Register(sid, fakeSessionState{state: "active"}, relay)
	defer inspect.Unregister(sid)

	wsURL := "ws" + strings.TrimPrefix(ts.URL, "http") + "/v1/inspect/" + sid + "/stream"
	conn, _, err := websocket.Dial(context.Background(), wsURL, &websocket.DialOptions{Subprotocols: []string{"inspect"}})
	if err != nil {
		t.Fatalf("dial: %v", err)
	}
	defer conn.CloseNow()
	for !relay.HasSubscribers() {
		time.Sleep(5 * time.Millisecond)
	}

	allLanes := []inspect.LaneID{
		inspect.LaneAudioLevel,
		inspect.LaneVAD,
		inspect.LaneSTT,
		inspect.LaneTurn,
		inspect.LaneBargein,
		inspect.LaneEOU,
		inspect.LaneDiarization,
		inspect.LaneLLM,
		inspect.LaneResponse,
		inspect.LaneTool,
		inspect.LaneTTSReq,
		inspect.LaneTTSChunk,
		inspect.LaneTTSPacer,
		inspect.LaneWire,
		inspect.LaneState,
	}

	for _, lane := range allLanes {
		if !inspect.ValidLane(lane) {
			t.Fatalf("ValidLane(%q) returned false -- constant defined but not in ValidLane switch", lane)
		}
	}

	for _, lane := range allLanes {
		inspect.Emit(relay, context.Background(), lane, "test_ping",
			nil, map[string]any{"lane_check": string(lane)})
	}

	seen := make(map[inspect.LaneID]bool)
	deadline := time.Now().Add(3 * time.Second)
	for len(seen) < len(allLanes) && time.Now().Before(deadline) {
		ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
		_, data, err := conn.Read(ctx)
		cancel()
		if err != nil {
			t.Fatalf("read after %d lanes: %v", len(seen), err)
		}
		var ev inspect.Event
		if err := json.Unmarshal(data, &ev); err != nil {
			t.Fatalf("decode: %v", err)
		}
		if ev.Kind == "test_ping" {
			seen[ev.Lane] = true
		}
	}

	for _, lane := range allLanes {
		if !seen[lane] {
			t.Errorf("lane %q event emitted but never received by inspector stream", lane)
		}
	}
}

func TestInspectStream_ErrorMirroredFromEveryLane(t *testing.T) {
	_, r, dir := newInspectTestServer(t)
	ts := httptest.NewServer(r)
	defer ts.Close()

	sid := "sess_test_mirror_all"
	relay := inspect.NewRelay(sid, dir)
	inspect.Register(sid, fakeSessionState{state: "active"}, relay)
	defer inspect.Unregister(sid)

	wsURL := "ws" + strings.TrimPrefix(ts.URL, "http") + "/v1/inspect/" + sid + "/stream"
	conn, _, err := websocket.Dial(context.Background(), wsURL, &websocket.DialOptions{Subprotocols: []string{"inspect"}})
	if err != nil {
		t.Fatalf("dial: %v", err)
	}
	defer conn.CloseNow()
	for !relay.HasSubscribers() {
		time.Sleep(5 * time.Millisecond)
	}

	errorLanes := []inspect.LaneID{
		inspect.LaneVAD,
		inspect.LaneSTT,
		inspect.LaneLLM,
		inspect.LaneTTSReq,
		inspect.LaneDiarization,
	}

	for _, lane := range errorLanes {
		inspect.Emit(relay, context.Background(), lane, "failed",
			nil, map[string]any{"error": "test_" + string(lane)})
	}

	originCount := 0
	mirrorCount := 0
	deadline := time.Now().Add(3 * time.Second)
	expect := len(errorLanes) * 2
	for (originCount+mirrorCount) < expect && time.Now().Before(deadline) {
		ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
		_, data, err := conn.Read(ctx)
		cancel()
		if err != nil {
			t.Fatalf("read: %v (origin=%d mirror=%d)", err, originCount, mirrorCount)
		}
		var ev inspect.Event
		_ = json.Unmarshal(data, &ev)
		if ev.Lane == inspect.LaneError && ev.Kind == "raised" {
			mirrorCount++
		} else if ev.Kind == "failed" {
			originCount++
		}
	}

	if originCount != len(errorLanes) {
		t.Errorf("expected %d origin events, got %d", len(errorLanes), originCount)
	}
	if mirrorCount != len(errorLanes) {
		t.Errorf("expected %d error mirror events, got %d", len(errorLanes), mirrorCount)
	}
}

func TestInspectLanes_JSMatchesGo(t *testing.T) {
	lanesJS, err := os.ReadFile("../../../inspector/src/lanes.js")
	if err != nil {
		t.Skipf("cannot read lanes.js: %v", err)
	}
	content := string(lanesJS)

	goLanes := []inspect.LaneID{
		inspect.LaneAudioLevel,
		inspect.LaneVAD,
		inspect.LaneSTT,
		inspect.LaneTurn,
		inspect.LaneBargein,
		inspect.LaneEOU,
		inspect.LaneDiarization,
		inspect.LaneLLM,
		inspect.LaneResponse,
		inspect.LaneTool,
		inspect.LaneTTSReq,
		inspect.LaneTTSChunk,
		inspect.LaneTTSPacer,
		inspect.LaneWire,
		inspect.LaneState,
		inspect.LaneError,
	}

	for _, lane := range goLanes {
		needle := "id: '" + string(lane) + "'"
		if !strings.Contains(content, needle) {
			t.Errorf("Go lane %q not found in inspector/src/lanes.js (expected %q)", lane, needle)
		}
	}
}

func TestInspectAudioStore_SliceReturnsWAV(t *testing.T) {
	_, r, dir := newInspectTestServer(t)
	ts := httptest.NewServer(r)
	defer ts.Close()

	sid := "sess_test_audio"
	as := inspect.NewAudioStore(sid, dir)
	registerSessionAudioStore(sid, as)
	relay := inspect.NewRelay(sid, dir)
	inspect.Register(sid, fakeSessionState{}, relay)
	defer func() {
		inspect.Unregister(sid)
		unregisterSessionAudioStore(sid)
		as.Close()
	}()

	samples := make(audio.MonoF32, 16000)
	for i := range samples {
		samples[i] = 0.5
	}
	as.AppendMicIn(samples)

	resp, err := http.Get(ts.URL + "/v1/inspect/sessions/" + sid + "/audio?channel=mic_in&from_ms=0&to_ms=500")
	if err != nil {
		t.Fatalf("audio: %v", err)
	}
	body, _ := io.ReadAll(resp.Body)
	if resp.StatusCode != 200 {
		t.Fatalf("status=%d body=%s", resp.StatusCode, string(body))
	}
	if string(body[:4]) != "RIFF" || string(body[8:12]) != "WAVE" {
		t.Fatalf("not a WAV: prefix=%q", string(body[:12]))
	}
	minBytes := 44 + 400*16000*2/1000
	if len(body) < minBytes {
		t.Fatalf("WAV too small: got %d, want >= %d (allows up to ~100ms track-start offset)", len(body), minBytes)
	}
}
