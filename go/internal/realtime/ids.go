package realtime

import (
	"crypto/rand"
	"encoding/hex"
	"fmt"
	"sync"
	"sync/atomic"
	"time"
)

type IdSource interface {
	NewSession() string
	NewItem() string
	NewResponse() string
	NewEvent() string
}

type randomIDs struct{}

func (randomIDs) NewSession() string  { return "sess_" + randomHex(12) }
func (randomIDs) NewItem() string     { return "item_" + randomHex(12) }
func (randomIDs) NewResponse() string { return "resp_" + randomHex(12) }
func (randomIDs) NewEvent() string    { return "event_" + randomHex(12) }

type DeterministicIDs struct {
	mu        sync.Mutex
	prefixes  map[string]uint64
	separator string
}

func NewDeterministicIDs() *DeterministicIDs {
	return &DeterministicIDs{prefixes: map[string]uint64{}, separator: "_"}
}

func (d *DeterministicIDs) next(prefix string) string {
	d.mu.Lock()
	defer d.mu.Unlock()
	d.prefixes[prefix]++
	return fmt.Sprintf("%s%s%d", prefix, d.separator, d.prefixes[prefix])
}

func (d *DeterministicIDs) NewSession() string  { return d.next("sess") }
func (d *DeterministicIDs) NewItem() string     { return d.next("item") }
func (d *DeterministicIDs) NewResponse() string { return d.next("resp") }
func (d *DeterministicIDs) NewEvent() string    { return d.next("event") }

var defaultIDs atomic.Pointer[IdSource]

func init() {
	var src IdSource = randomIDs{}
	defaultIDs.Store(&src)
}

func setDefaultIDs(src IdSource) {
	defaultIDs.Store(&src)
}

func currentIDs() IdSource {
	if p := defaultIDs.Load(); p != nil {
		return *p
	}
	return randomIDs{}
}

func newEventID() string { return currentIDs().NewEvent() }
func newItemID() string  { return currentIDs().NewItem() }
func newSessID() string  { return currentIDs().NewSession() }
func newRespID() string  { return currentIDs().NewResponse() }

func randomHex(n int) string {
	b := make([]byte, n)
	if _, err := rand.Read(b); err != nil {
		return time.Now().Format("20060102150405.000000")
	}
	return hex.EncodeToString(b)
}
