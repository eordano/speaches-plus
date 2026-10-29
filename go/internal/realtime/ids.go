package realtime

import (
	"crypto/rand"
	"encoding/hex"
	"time"
)

func newEventID() string { return "event_" + randomHex(12) }
func newItemID() string  { return "item_" + randomHex(12) }
func newSessID() string  { return "sess_" + randomHex(12) }
func newRespID() string  { return "resp_" + randomHex(12) }

func randomHex(n int) string {
	b := make([]byte, n)
	if _, err := rand.Read(b); err != nil {
		return time.Now().Format("20060102150405.000000")
	}
	return hex.EncodeToString(b)
}
