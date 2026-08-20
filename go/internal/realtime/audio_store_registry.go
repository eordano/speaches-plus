package realtime

import (
	"sync"

	"github.com/eordano/speaches-plus-go/internal/inspect"
)

var (
	audioStoreMu      sync.RWMutex
	audioStoresBySess = map[string]*inspect.AudioStore{}
)

func registerSessionAudioStore(sessionID string, as *inspect.AudioStore) {
	if sessionID == "" || as == nil {
		return
	}
	audioStoreMu.Lock()
	audioStoresBySess[sessionID] = as
	audioStoreMu.Unlock()
}

func unregisterSessionAudioStore(sessionID string) {
	audioStoreMu.Lock()
	delete(audioStoresBySess, sessionID)
	audioStoreMu.Unlock()
}

func getSessionAudioStore(sessionID string) *inspect.AudioStore {
	audioStoreMu.RLock()
	defer audioStoreMu.RUnlock()
	return audioStoresBySess[sessionID]
}
