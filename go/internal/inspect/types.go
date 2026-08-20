package inspect

type LaneID string

func ValidLane(l LaneID) bool {
	switch l {
	case LaneAudioLevel, LaneVAD, LaneSTT, LaneTurn, LaneBargein,
		LaneEOU, LaneDiarization, LaneLLM, LaneResponse, LaneTool, LaneTTSReq,
		LaneTTSChunk, LaneTTSPacer, LaneWire, LaneState, LaneError:
		return true
	}
	return false
}

var errKinds = map[string]struct{}{
	"error":          {},
	"raised":         {},
	"dropped":        {},
	"failed":         {},
	"phrase_error":   {},
	"bargein_missed": {},
}

func IsErrorKind(k string) bool {
	_, ok := errKinds[k]
	return ok
}

type Corr struct {
	TurnID     string `json:"turn_id,omitempty"`
	ItemID     string `json:"item_id,omitempty"`
	ResponseID string `json:"response_id,omitempty"`
	PhraseID   string `json:"phrase_id,omitempty"`
}

type Event struct {
	SessionID string         `json:"session_id"`
	Seq       uint64         `json:"seq"`
	TSMonoNS  int64          `json:"ts_mono_ns"`
	TSWall    float64        `json:"ts_wall"`
	Lane      LaneID         `json:"lane"`
	Kind      string         `json:"kind"`
	Corr      Corr           `json:"corr"`
	SpanID    string         `json:"span_id,omitempty"`
	Payload   map[string]any `json:"payload,omitempty"`
}

type SessionMeta struct {
	ID          string   `json:"id"`
	CreatedAt   float64  `json:"created_at"`
	Model       string   `json:"model"`
	State       string   `json:"state"`
	TurnCount   uint64   `json:"turn_count"`
	LastEventTS *float64 `json:"last_event_ts,omitempty"`
}

type SessionHistoryEntry struct {
	ID        string  `json:"id"`
	SizeBytes int64   `json:"size_bytes"`
	MTime     float64 `json:"mtime"`
}
