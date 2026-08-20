package inspect

import (
	"sort"
	"sync"
	"time"
)

type sessionState interface {
	State() string
	Model() string
}

type sessionEntry struct {
	state     sessionState
	relay     *Relay
	createdAt float64
}

var (
	regMu   sync.RWMutex
	entries = map[string]sessionEntry{}
)

func Register(sessionID string, state sessionState, relay *Relay) {
	regMu.Lock()
	defer regMu.Unlock()
	entries[sessionID] = sessionEntry{
		state:     state,
		relay:     relay,
		createdAt: float64(time.Now().UnixNano()) / 1e9,
	}
}

func Unregister(sessionID string) {
	regMu.Lock()
	if e, ok := entries[sessionID]; ok {
		if e.relay != nil {
			e.relay.Close()
		}
		delete(entries, sessionID)
	}
	regMu.Unlock()
}

func GetRelay(sessionID string) *Relay {
	regMu.RLock()
	defer regMu.RUnlock()
	if e, ok := entries[sessionID]; ok {
		return e.relay
	}
	return nil
}

func GetState(sessionID string) sessionState {
	regMu.RLock()
	defer regMu.RUnlock()
	if e, ok := entries[sessionID]; ok {
		return e.state
	}
	return nil
}

func ListMeta() []SessionMeta {
	regMu.RLock()
	out := make([]SessionMeta, 0, len(entries))
	for sid, e := range entries {
		var lastTSPtr *float64
		if e.relay != nil {
			ts := e.relay.LastEventTS()
			if ts > 0 {
				lastTSPtr = &ts
			}
		}
		var (
			turn  uint64
			model string
			state string
		)
		if e.relay != nil {
			turn = e.relay.TurnCount()
		}
		if e.state != nil {
			model = e.state.Model()
			state = e.state.State()
		}
		out = append(out, SessionMeta{
			ID:          sid,
			CreatedAt:   e.createdAt,
			Model:       model,
			State:       state,
			TurnCount:   turn,
			LastEventTS: lastTSPtr,
		})
	}
	regMu.RUnlock()
	sort.Slice(out, func(i, j int) bool { return out[i].CreatedAt > out[j].CreatedAt })
	return out
}
