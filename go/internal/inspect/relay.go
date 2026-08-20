package inspect

import (
	"bufio"
	"encoding/json"
	"fmt"
	"log/slog"
	"os"
	"path/filepath"
	"sync"
	"sync/atomic"
)

type Relay struct {
	sessionID  string
	sessionDir string
	relayCap   int

	mu        sync.Mutex
	buffer    [][]byte
	subs      map[uint64]chan []byte
	nextSubID uint64

	seq         atomic.Uint64
	turnCount   atomic.Uint64
	lastEventTS atomic.Uint64
	dropped     atomic.Uint64

	corrMu     sync.RWMutex
	turnID     string
	itemID     string
	responseID string
	phraseID   string

	fileMu      sync.Mutex
	ndjsonPath  string
	ndjsonFile  *os.File
	ndjsonBufWr *bufio.Writer
}

func NewRelay(sessionID, sessionDir string) *Relay {
	return NewRelayWithCap(sessionID, sessionDir, defaultRelayCap)
}

func NewRelayWithCap(sessionID, sessionDir string, relayCap int) *Relay {
	if relayCap <= 0 {
		relayCap = defaultRelayCap
	}
	r := &Relay{
		sessionID:  sessionID,
		sessionDir: sessionDir,
		relayCap:   relayCap,
		subs:       map[uint64]chan []byte{},
	}
	if sessionDir != "" {
		if err := os.MkdirAll(sessionDir, 0o755); err != nil {
			slog.Warn("inspect: mkdir session dir", "err", err, "path", sessionDir)
		} else {
			r.ndjsonPath = filepath.Join(sessionDir, sessionID+".ndjson")
			f, err := os.OpenFile(r.ndjsonPath, os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0o644)
			if err != nil {
				slog.Warn("inspect: open ndjson", "err", err, "path", r.ndjsonPath)
			} else {
				r.ndjsonFile = f
				r.ndjsonBufWr = bufio.NewWriter(f)
			}
		}
	}
	return r
}

func (r *Relay) SessionID() string    { return r.sessionID }
func (r *Relay) NextSeq() uint64      { return r.seq.Add(1) - 1 }
func (r *Relay) TurnCount() uint64    { return r.turnCount.Load() }
func (r *Relay) LastEventTS() float64 { return float64FromBits(r.lastEventTS.Load()) }
func (r *Relay) DroppedCount() uint64 { return r.dropped.Load() }

func (r *Relay) HasSubscribers() bool {
	r.mu.Lock()
	defer r.mu.Unlock()
	return len(r.subs) > 0
}

func (r *Relay) SetCorr(turnID, itemID, responseID, phraseID *string) {
	r.corrMu.Lock()
	defer r.corrMu.Unlock()
	if turnID != nil {
		r.turnID = *turnID
	}
	if itemID != nil {
		r.itemID = *itemID
	}
	if responseID != nil {
		r.responseID = *responseID
	}
	if phraseID != nil {
		r.phraseID = *phraseID
	}
}

func (r *Relay) Corr() Corr {
	r.corrMu.RLock()
	defer r.corrMu.RUnlock()
	return Corr{
		TurnID:     r.turnID,
		ItemID:     r.itemID,
		ResponseID: r.responseID,
		PhraseID:   r.phraseID,
	}
}

func (r *Relay) Publish(ev Event) {
	if ev.Lane == LaneTurn && ev.Kind == "turn_end" {
		r.turnCount.Add(1)
	}
	r.lastEventTS.Store(float64Bits(ev.TSWall))

	line, err := json.Marshal(ev)
	if err != nil {
		slog.Warn("inspect: marshal event", "err", err)
		return
	}
	line = append(line, '\n')

	r.mu.Lock()
	r.buffer = append(r.buffer, line)
	subs := make([]chan []byte, 0, len(r.subs))
	for _, ch := range r.subs {
		subs = append(subs, ch)
	}
	r.mu.Unlock()

	r.writeNDJSON(line)
	for _, ch := range subs {
		r.enqueue(ch, line)
	}

	if ev.Lane != LaneError && IsErrorKind(ev.Kind) {
		mirror := Event{
			SessionID: ev.SessionID,
			Seq:       r.NextSeq(),
			TSMonoNS:  ev.TSMonoNS,
			TSWall:    ev.TSWall,
			Lane:      LaneError,
			Kind:      "raised",
			Corr:      ev.Corr,
			SpanID:    ev.SpanID,
			Payload: map[string]any{
				"lane":        string(ev.Lane),
				"origin_seq":  ev.Seq,
				"origin_kind": ev.Kind,
				"error":       errMessage(ev),
				"severity":    "error",
			},
		}
		mline, err := json.Marshal(mirror)
		if err == nil {
			mline = append(mline, '\n')
			r.mu.Lock()
			r.buffer = append(r.buffer, mline)
			subs := make([]chan []byte, 0, len(r.subs))
			for _, ch := range r.subs {
				subs = append(subs, ch)
			}
			r.mu.Unlock()
			r.writeNDJSON(mline)
			for _, ch := range subs {
				r.enqueue(ch, mline)
			}
		}
	}
}

func errMessage(ev Event) string {
	if v, ok := ev.Payload["error"]; ok {
		return fmt.Sprint(v)
	}
	if v, ok := ev.Payload["reason"]; ok {
		return fmt.Sprint(v)
	}
	return ev.Kind
}

func (r *Relay) writeNDJSON(line []byte) {
	r.fileMu.Lock()
	defer r.fileMu.Unlock()
	if r.ndjsonBufWr == nil {
		return
	}
	if _, err := r.ndjsonBufWr.Write(line); err != nil {
		slog.Warn("inspect: ndjson write", "err", err, "path", r.ndjsonPath)
	}
	if err := r.ndjsonBufWr.Flush(); err != nil {
		slog.Warn("inspect: ndjson flush", "err", err)
	}
}

func (r *Relay) enqueue(ch chan []byte, line []byte) {
	if tryEnqueue(ch, line) {
		return
	}
	dropOldest(ch)
	if !tryEnqueue(ch, line) {
		r.dropped.Add(1)
		return
	}
	r.dropped.Add(1)
	r.injectDroppedEvent(ch)
}

func tryEnqueue(ch chan []byte, line []byte) bool {
	select {
	case ch <- line:
		return true
	default:
		return false
	}
}

func dropOldest(ch chan []byte) bool {
	select {
	case <-ch:
		return true
	default:
		return false
	}
}

func (r *Relay) injectDroppedEvent(ch chan []byte) {
	ev := Event{
		SessionID: r.sessionID,
		Seq:       r.NextSeq(),
		TSMonoNS:  nowMonoNS(),
		TSWall:    nowWall(),
		Lane:      LaneError,
		Kind:      "dropped",
		Corr:      r.Corr(),
		Payload: map[string]any{
			"dropped_total": r.dropped.Load(),
			"reason":        "subscriber_queue_full",
		},
	}
	line, err := jsonMarshalLine(ev)
	if err != nil {
		return
	}
	r.writeNDJSON(line)
	if tryEnqueue(ch, line) {
		return
	}
	dropOldest(ch)
	_ = tryEnqueue(ch, line)
}

type Subscription struct {
	id    uint64
	ch    chan []byte
	relay *Relay
}

func (s *Subscription) Channel() <-chan []byte { return s.ch }

func (s *Subscription) Close() {
	if s == nil || s.relay == nil {
		return
	}
	s.relay.mu.Lock()
	if ch, ok := s.relay.subs[s.id]; ok {
		delete(s.relay.subs, s.id)
		close(ch)
	}
	s.relay.mu.Unlock()
	s.relay = nil
}

func (r *Relay) Subscribe() *Subscription {
	ch := make(chan []byte, r.relayCap)
	r.mu.Lock()
	snapshot := make([][]byte, len(r.buffer))
	copy(snapshot, r.buffer)
	id := r.nextSubID
	r.nextSubID++
	r.subs[id] = ch
	r.mu.Unlock()
	for _, line := range snapshot {
		select {
		case ch <- line:
		default:
		}
	}
	return &Subscription{id: id, ch: ch, relay: r}
}

func (r *Relay) Close() {
	r.fileMu.Lock()
	if r.ndjsonBufWr != nil {
		_ = r.ndjsonBufWr.Flush()
	}
	if r.ndjsonFile != nil {
		_ = r.ndjsonFile.Close()
		r.ndjsonFile = nil
		r.ndjsonBufWr = nil
	}
	r.fileMu.Unlock()
}
