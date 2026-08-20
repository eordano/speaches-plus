package realtime

import (
	"fmt"
	"sync"
)

type sessionKind int

func (k sessionKind) String() string {
	switch k {
	case sessKindPending:
		return "pending"
	case sessKindActive:
		return "active"
	case sessKindTerminated:
		return "terminated"
	}
	return "?"
}

type TerminationReason int

func (r TerminationReason) String() string {
	switch r {
	case TermClientClosed:
		return "client_closed"
	case TermMaxDuration:
		return "max_duration"
	case TermInternalStateError:
		return "internal_state_error"
	case TermVadFailed:
		return "vad_failed"
	case TermSttFailed:
		return "stt_failed"
	case TermModelLoadFailed:
		return "model_load_failed"
	case TermClientTooSlow:
		return "client_too_slow"
	}
	return "unknown"
}

type SessionPhase interface {
	isSession()
	Kind() sessionKind
}

type SessionPending struct{}

func (SessionPending) isSession()        {}
func (SessionPending) Kind() sessionKind { return sessKindPending }

type SessionActive struct {
	CreatedAtMs Millis

	Instructions string
}

func (SessionActive) isSession()        {}
func (SessionActive) Kind() sessionKind { return sessKindActive }

type SessionTerminated struct {
	Reason TerminationReason
}

func (SessionTerminated) isSession()        {}
func (SessionTerminated) Kind() sessionKind { return sessKindTerminated }

type vadKind int

func (k vadKind) String() string {
	switch k {
	case vadKindSilent:
		return "silent"
	case vadKindSpeaking:
		return "speaking"
	case vadKindStopped:
		return "stopped"
	}
	return "?"
}

type partialLoop struct{}

type commitTimer struct{}

type VadPhase interface {
	isVad()
	Kind() vadKind
}

type VadSilent struct{}

func (VadSilent) isVad()        {}
func (VadSilent) Kind() vadKind { return vadKindSilent }

type VadSpeaking struct {
	ItemID       ItemID
	AudioStartMs Millis
	PartialTask  *partialLoop
}

func (VadSpeaking) isVad()        {}
func (VadSpeaking) Kind() vadKind { return vadKindSpeaking }

type VadStopped struct {
	ItemID       ItemID
	AudioStartMs Millis
	AudioEndMs   Millis
	CommitTimer  *commitTimer
	HardCapAt    Millis
}

func (VadStopped) isVad()        {}
func (VadStopped) Kind() vadKind { return vadKindStopped }

type bufKind int

func (k bufKind) String() string {
	switch k {
	case bufKindEmpty:
		return "empty"
	case bufKindVoiced:
		return "voiced"
	case bufKindStopped:
		return "stopped"
	case bufKindCommitted:
		return "committed"
	}
	return "?"
}

type InputBuffer interface {
	isBuf()
	Kind() bufKind
}

type BufEmpty struct{}

func (BufEmpty) isBuf()        {}
func (BufEmpty) Kind() bufKind { return bufKindEmpty }

type BufVoiced struct {
	ItemID  ItemID
	StartMs Millis
}

func (BufVoiced) isBuf()        {}
func (BufVoiced) Kind() bufKind { return bufKindVoiced }

type BufStopped struct {
	ItemID  ItemID
	StartMs Millis
	EndMs   Millis
}

func (BufStopped) isBuf()        {}
func (BufStopped) Kind() bufKind { return bufKindStopped }

type BufCommitted struct {
	ItemID  ItemID
	StartMs Millis
	EndMs   Millis
}

func (BufCommitted) isBuf()        {}
func (BufCommitted) Kind() bufKind { return bufKindCommitted }

func bufItemIDOf(b InputBuffer) ItemID {
	switch v := b.(type) {
	case BufVoiced:
		return v.ItemID
	case BufStopped:
		return v.ItemID
	case BufCommitted:
		return v.ItemID
	}
	return ""
}

type respPhaseKind int

func (k respPhaseKind) String() string {
	switch k {
	case respKindNone:
		return "none"
	case respKindPredicted:
		return "predicted"
	case respKindCreated:
		return "created"
	case respKindStreaming:
		return "streaming"
	case respKindDrain:
		return "drain"
	case respKindFinalized:
		return "finalized"
	}
	return "?"
}

type responseStatus int

func (s responseStatus) String() string {
	switch s {
	case respStatusCompleted:
		return "completed"
	case respStatusCancelled:
		return "cancelled"
	case respStatusIncomplete:
		return "incomplete"
	case respStatusFailed:
		return "failed"
	}
	return "unknown"
}

type RespPhase interface {
	isRespPhase()
	Kind() respPhaseKind
}

type RespNone struct {
	Epoch Epoch
}

func (RespNone) isRespPhase()        {}
func (RespNone) Kind() respPhaseKind { return respKindNone }

type RespPredicted struct {
	ID       ResponseID
	ItemID   ItemID
	Epoch    Epoch
	EouScore float32

	Runner *eagerRunner
}

func (RespPredicted) isRespPhase()        {}
func (RespPredicted) Kind() respPhaseKind { return respKindPredicted }

type RespCreated struct {
	ID     ResponseID
	ItemID ItemID
	Epoch  Epoch
}

func (RespCreated) isRespPhase()        {}
func (RespCreated) Kind() respPhaseKind { return respKindCreated }

type RespStreaming struct {
	ID         ResponseID
	ItemID     ItemID
	Epoch      Epoch
	Transcript string
	PlannedMs  DurationMs
	PlayedMs   Millis
}

func (RespStreaming) isRespPhase()        {}
func (RespStreaming) Kind() respPhaseKind { return respKindStreaming }

type RespDrain struct {
	ID         ResponseID
	ItemID     ItemID
	Epoch      Epoch
	Transcript string
	PlannedMs  DurationMs
	PlayedMs   Millis
}

func (RespDrain) isRespPhase()        {}
func (RespDrain) Kind() respPhaseKind { return respKindDrain }

type RespFinalized struct {
	ID         ResponseID
	ItemID     ItemID
	Epoch      Epoch
	Status     responseStatus
	Transcript string
	PlayedMs   Millis
}

func (RespFinalized) isRespPhase()        {}
func (RespFinalized) Kind() respPhaseKind { return respKindFinalized }

func respEpochOf(r RespPhase) Epoch {
	switch v := r.(type) {
	case RespNone:
		return v.Epoch
	case RespPredicted:
		return v.Epoch
	case RespCreated:
		return v.Epoch
	case RespStreaming:
		return v.Epoch
	case RespDrain:
		return v.Epoch
	case RespFinalized:
		return v.Epoch
	}
	return 0
}

func respIDOf(r RespPhase) ResponseID {
	switch v := r.(type) {
	case RespPredicted:
		return v.ID
	case RespCreated:
		return v.ID
	case RespStreaming:
		return v.ID
	case RespDrain:
		return v.ID
	case RespFinalized:
		return v.ID
	}
	return ""
}

type sealedBuffer struct {
	itemID   ItemID
	startMs  Millis
	endMs    Millis
	sealedAt int64
}

type itemStatus int

func (s itemStatus) String() string {
	switch s {
	case itemInProgress:
		return "in_progress"
	case itemCompleted:
		return "completed"
	case itemIncomplete:
		return "incomplete"
	}
	return "?"
}

type conversationItem struct {
	ID         ItemID
	Role       string
	Status     itemStatus
	Transcript string
	AudioEndMs Millis
}

type phaseState struct {
	mu sync.Mutex

	session SessionPhase
	buf     InputBuffer
	vad     VadPhase
	resp    RespPhase

	conv            map[ItemID]*conversationItem
	convOrd         []ItemID
	sealedBufs      map[ItemID]sealedBuffer
	sealedOrd       []ItemID
	sealedRetention int

	inflightPredicted int
	eagerMaxInflight  int

	transitionHook func(phase, from, to string)
	violationHook  func(err error)
}

func (s *phaseState) initialize() {
	if s.session == nil {
		s.session = SessionPending{}
	}
	if s.buf == nil {
		s.buf = BufEmpty{}
	}
	if s.vad == nil {
		s.vad = VadSilent{}
	}
	if s.resp == nil {
		s.resp = RespNone{}
	}
}

type violation int

func (v violation) String() string {
	switch v {
	case violationI1SpeakingWithActiveResponse:
		return "I1: speaking with active response"
	case violationEmptyResponseID:
		return "response phase has empty id"
	case violationSessionUpdateBeforeActive:
		return "session.update before session active"
	case violationCommittedBufNoItem:
		return "Committed buffer without item_id"
	case violationConvHasVoiced:
		return "conversation contains a Voiced user item"
	case violationI7RotationBeforeCommit:
		return "I7: Stopped buffer has sealed entry (rotation before commit)"
	case violationI9PredictedNoRunner:
		return "I9: Predicted with nil runner reference"
	}
	return "?"
}

func checkInvariants(s *phaseState) error {
	s.initialize()

	if _, speaking := s.vad.(VadSpeaking); speaking {
		switch s.resp.(type) {
		case RespCreated, RespStreaming, RespDrain:
			return fmt.Errorf("invariant: %s (vad=%T resp=%T)",
				violationI1SpeakingWithActiveResponse, s.vad, s.resp)
		}
	}

	switch r := s.resp.(type) {
	case RespCreated:
		if r.ID == "" {
			return fmt.Errorf("invariant: %s", violationEmptyResponseID)
		}
	case RespStreaming:
		if r.ID == "" {
			return fmt.Errorf("invariant: %s", violationEmptyResponseID)
		}
	case RespDrain:
		if r.ID == "" {
			return fmt.Errorf("invariant: %s", violationEmptyResponseID)
		}
	}

	switch s.resp.(type) {
	case RespCreated, RespStreaming, RespDrain:
		if _, empty := s.buf.(BufEmpty); !empty {
			return fmt.Errorf("invariant: active response with non-empty buffer (buf=%T)", s.buf)
		}
	}

	if bc, ok := s.buf.(BufCommitted); ok && bc.ItemID == "" {
		return fmt.Errorf("invariant: %s", violationCommittedBufNoItem)
	}

	if bs, ok := s.buf.(BufStopped); ok {
		if _, dup := s.sealedBufs[bs.ItemID]; dup {
			return fmt.Errorf("invariant: %s (item=%s)",
				violationI7RotationBeforeCommit, bs.ItemID)
		}
	}

	if rp, ok := s.resp.(RespPredicted); ok {
		if rp.Runner == nil {
			return fmt.Errorf("invariant: %s (id=%s)",
				violationI9PredictedNoRunner, rp.ID)
		}
	}

	if bv, ok := s.buf.(BufVoiced); ok {
		for id, it := range s.conv {
			if it == nil {
				continue
			}
			if it.Role == "user" && it.Status == itemInProgress && bv.ItemID == id {
				return fmt.Errorf("invariant: %s (id=%s)",
					violationConvHasVoiced, id)
			}
		}
	}

	return nil
}

func (s *phaseState) withLock(fn func()) {
	s.mu.Lock()
	s.initialize()
	prevSession := s.session.Kind()
	prevResp := s.resp.Kind()
	prevVad := s.vad.Kind()
	prevBuf := s.buf.Kind()
	fn()
	s.initialize()
	violationErr := checkInvariants(s)

	type transition struct{ phase, from, to string }
	var transitions []transition
	if prevSession != s.session.Kind() {
		transitions = append(transitions, transition{"session", prevSession.String(), s.session.Kind().String()})
	}
	if prevResp != s.resp.Kind() {
		transitions = append(transitions, transition{"resp", prevResp.String(), s.resp.Kind().String()})
	}
	if prevVad != s.vad.Kind() {
		transitions = append(transitions, transition{"vad", prevVad.String(), s.vad.Kind().String()})
	}
	if prevBuf != s.buf.Kind() {
		transitions = append(transitions, transition{"buf", prevBuf.String(), s.buf.Kind().String()})
	}
	prevTop := derivedTopName(prevSession, prevVad, prevResp, prevBuf)
	newTop := derivedTopName(s.session.Kind(), s.vad.Kind(), s.resp.Kind(), s.buf.Kind())
	if prevTop != newTop {
		transitions = append(transitions, transition{"top", prevTop, newTop})
	}
	transitionHook := s.transitionHook
	violationHook := s.violationHook
	s.mu.Unlock()

	if transitionHook != nil {
		for _, t := range transitions {
			transitionHook(t.phase, t.from, t.to)
		}
	}
	if violationErr != nil {
		if violationHook != nil {
			violationHook(violationErr)
		} else if debugInvariants {
			panic(violationErr)
		}
	}
}

func derivedTopName(sk sessionKind, vk vadKind, rk respPhaseKind, bk bufKind) string {
	switch rk {
	case respKindDrain:
		return "drain"
	case respKindCreated, respKindStreaming, respKindPredicted:
		return "generate"
	}
	switch bk {
	case bufKindCommitted:
		return "process"
	case bufKindVoiced, bufKindStopped:
		return "listen"
	}
	if vk == vadKindSpeaking {
		return "listen"
	}
	return "idle"
}

type cancelEffect struct {
	cancelled  bool
	id         string
	itemID     string
	epoch      uint64
	playedMs   int64
	transcript string
	wasDrain   bool
}

type bargeInEffect struct {
	cancel cancelEffect

	cancelTimer bool

	predictedRolled bool

	runnerToAbort *eagerRunner
}

func boolStr(b bool) string {
	if b {
		return "speaking"
	}
	return "silent"
}

func (s *phaseState) setSealedRetention(n int) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.sealedRetention = n
}

func (s *phaseState) setEagerMaxInflight(n int) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if n <= 0 {
		n = 1
	}
	s.eagerMaxInflight = n
}

func (s *phaseState) startSession() bool {
	var ok bool
	s.withLock(func() {
		if _, pending := s.session.(SessionPending); pending {
			s.session = SessionActive{}
			if s.conv == nil {
				s.conv = make(map[ItemID]*conversationItem)
			}
			ok = true
		}
	})
	return ok
}

func (s *phaseState) terminateSession() {
	s.withLock(func() {

		if _, term := s.session.(SessionTerminated); term {
			return
		}
		s.session = SessionTerminated{Reason: TermClientClosed}
	})
}

func (s *phaseState) terminateSessionWithReason(r TerminationReason) {
	s.withLock(func() {
		if _, term := s.session.(SessionTerminated); term {
			return
		}
		s.session = SessionTerminated{Reason: r}
	})
}

func (s *phaseState) setInstructions(instr string) bool {
	var ok bool
	s.withLock(func() {
		sa, active := s.session.(SessionActive)
		if !active {
			return
		}
		sa.Instructions = instr
		s.session = sa
		ok = true
	})
	return ok
}

func (s *phaseState) getInstructions() string {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.initialize()
	if sa, active := s.session.(SessionActive); active {
		return sa.Instructions
	}
	return ""
}

func (s *phaseState) updateSession() error {
	var err error
	s.withLock(func() {
		if _, active := s.session.(SessionActive); !active {
			err = fmt.Errorf("%s", violationSessionUpdateBeforeActive)
		}
	})
	return err
}

func (s *phaseState) onVadSpeechStart(itemID string, startMs int64, playedMsSnap func() int64) bargeInEffect {
	var eff bargeInEffect
	s.withLock(func() {
		if _, active := s.session.(SessionActive); !active {
			return
		}

		if bs, ok := s.buf.(BufStopped); ok {
			eff.cancelTimer = true
			s.buf = BufVoiced{ItemID: bs.ItemID, StartMs: bs.StartMs}
			s.vad = VadSpeaking{ItemID: bs.ItemID, AudioStartMs: bs.StartMs}
			return
		}

		snap := func() int64 {
			if playedMsSnap == nil {
				return 0
			}
			return playedMsSnap()
		}

		switch r := s.resp.(type) {
		case RespCreated:
			eff.cancel.cancelled = true
			eff.cancel.id = string(r.ID)
			eff.cancel.itemID = string(r.ItemID)
			eff.cancel.epoch = uint64(r.Epoch)
			s.resp = RespNone{Epoch: r.Epoch + 1}
		case RespStreaming:
			eff.cancel.cancelled = true
			eff.cancel.id = string(r.ID)
			eff.cancel.itemID = string(r.ItemID)
			eff.cancel.epoch = uint64(r.Epoch)
			eff.cancel.playedMs = snap()
			eff.cancel.transcript = r.Transcript
			s.resp = RespNone{Epoch: r.Epoch + 1}
		case RespDrain:
			eff.cancel.cancelled = true
			eff.cancel.id = string(r.ID)
			eff.cancel.itemID = string(r.ItemID)
			eff.cancel.epoch = uint64(r.Epoch)
			eff.cancel.playedMs = snap()
			eff.cancel.transcript = r.Transcript
			eff.cancel.wasDrain = true
			s.resp = RespNone{Epoch: r.Epoch + 1}
		case RespPredicted:

			eff.predictedRolled = true
			eff.runnerToAbort = r.Runner
			s.resp = RespNone{Epoch: r.Epoch + 1}
			if s.inflightPredicted > 0 {
				s.inflightPredicted--
			}
		}
		s.buf = BufVoiced{ItemID: ItemID(itemID), StartMs: Millis(startMs)}
		s.vad = VadSpeaking{ItemID: ItemID(itemID), AudioStartMs: Millis(startMs)}
	})
	return eff
}

func (s *phaseState) onVadSpeechEnd(endMs Millis) (ItemID, Millis, bool) {
	var itemID ItemID
	var startMs Millis
	var ok bool
	s.withLock(func() {
		if _, active := s.session.(SessionActive); !active {
			return
		}
		bv, isVoiced := s.buf.(BufVoiced)
		if !isVoiced {
			return
		}
		itemID = bv.ItemID
		startMs = bv.StartMs
		s.buf = BufStopped{ItemID: itemID, StartMs: startMs, EndMs: endMs}
		s.vad = VadStopped{ItemID: itemID, AudioStartMs: startMs, AudioEndMs: endMs}
		ok = true
	})
	return itemID, startMs, ok
}

type commitEffect struct {
	committed bool
	itemID    ItemID
	startMs   Millis
	endMs     Millis
}

func (s *phaseState) forceCommitForIntegrated(itemID ItemID, endMs Millis) commitEffect {
	var eff commitEffect
	s.withLock(func() {
		if _, active := s.session.(SessionActive); !active {
			return
		}

		if itemID == "" {
			itemID = bufItemIDOf(s.buf)
		}
		if itemID == "" {
			return
		}
		var startMs Millis
		switch b := s.buf.(type) {
		case BufStopped:
			startMs = b.StartMs
			endMs = b.EndMs
		case BufVoiced:
			startMs = b.StartMs
		case BufCommitted:
			startMs = b.StartMs

		}
		eff.committed = true
		eff.itemID = itemID
		eff.startMs = startMs
		eff.endMs = endMs

		s.rotateSealed(itemID, startMs, endMs)
		s.buf = BufCommitted{ItemID: itemID, StartMs: startMs, EndMs: endMs}
		s.vad = VadSilent{}
		if s.conv == nil {
			s.conv = make(map[ItemID]*conversationItem)
		}
		s.conv[itemID] = &conversationItem{
			ID:         itemID,
			Role:       "user",
			Status:     itemInProgress,
			AudioEndMs: endMs,
		}
		s.convOrd = append(s.convOrd, itemID)
	})
	return eff
}

func (s *phaseState) onCommitTimerFire() commitEffect {
	var eff commitEffect
	s.withLock(func() {
		bs, ok := s.buf.(BufStopped)
		if !ok {
			return
		}
		eff.committed = true
		eff.itemID = bs.ItemID
		eff.startMs = bs.StartMs
		eff.endMs = bs.EndMs

		s.rotateSealed(bs.ItemID, bs.StartMs, bs.EndMs)
		s.buf = BufCommitted{ItemID: bs.ItemID, StartMs: bs.StartMs, EndMs: bs.EndMs}
		s.vad = VadSilent{}
		if s.conv == nil {
			s.conv = make(map[ItemID]*conversationItem)
		}
		s.conv[bs.ItemID] = &conversationItem{
			ID:         bs.ItemID,
			Role:       "user",
			Status:     itemInProgress,
			AudioEndMs: bs.EndMs,
		}
		s.convOrd = append(s.convOrd, bs.ItemID)
	})
	return eff
}

func (s *phaseState) rotateSealed(itemID ItemID, startMs, endMs Millis) {
	if s.sealedBufs == nil {
		s.sealedBufs = make(map[ItemID]sealedBuffer)
	}
	if _, dup := s.sealedBufs[itemID]; !dup {
		s.sealedOrd = append(s.sealedOrd, itemID)
	}
	s.sealedBufs[itemID] = sealedBuffer{
		itemID:  itemID,
		startMs: startMs,
		endMs:   endMs,
	}
	retention := s.sealedRetention
	if retention <= 0 {
		retention = 4
	}
	for len(s.sealedOrd) > retention {
		oldest := s.sealedOrd[0]
		s.sealedOrd = s.sealedOrd[1:]
		delete(s.sealedBufs, oldest)
	}
}

func (s *phaseState) onTranscriptionComplete(itemID ItemID, transcript string, autoResponse bool) {
	s.withLock(func() {
		if it, ok := s.conv[itemID]; ok && it != nil {
			it.Transcript = transcript
			it.Status = itemCompleted
		}
		if _, found := s.sealedBufs[itemID]; found {
			delete(s.sealedBufs, itemID)
			for i, id := range s.sealedOrd {
				if id == itemID {
					s.sealedOrd = append(s.sealedOrd[:i], s.sealedOrd[i+1:]...)
					break
				}
			}
		}
		s.buf = BufEmpty{}
	})
}

func (s *phaseState) onTranscriptionFailed(itemID ItemID) {
	s.withLock(func() {
		if it, ok := s.conv[itemID]; ok && it != nil {
			it.Status = itemIncomplete
		}
		s.buf = BufEmpty{}
	})
}

func (s *phaseState) onPredictedDispatch(id ResponseID, itemID ItemID, score float32, runner *eagerRunner) (Epoch, bool) {
	var ok bool
	var epoch Epoch
	s.withLock(func() {
		if runner == nil {
			return
		}
		if _, active := s.session.(SessionActive); !active {
			return
		}
		if _, none := s.resp.(RespNone); !none {
			return
		}
		cap := s.eagerMaxInflight
		if cap <= 0 {
			cap = 1
		}
		if s.inflightPredicted >= cap {
			return
		}
		prevEpoch := respEpochOf(s.resp)
		newEpoch := prevEpoch + 1

		runner.epoch = uint64(newEpoch)
		s.resp = RespPredicted{
			ID:       id,
			ItemID:   itemID,
			Epoch:    newEpoch,
			EouScore: score,
			Runner:   runner,
		}
		s.inflightPredicted++
		ok = true
		epoch = newEpoch
	})
	return epoch, ok
}

func (s *phaseState) currentEagerRunner() *eagerRunner {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.initialize()
	if rp, ok := s.resp.(RespPredicted); ok {
		return rp.Runner
	}
	return nil
}

func (s *phaseState) onPredictedRollback() (ResponseID, Epoch, *eagerRunner, bool) {
	var id ResponseID
	var epoch Epoch
	var runner *eagerRunner
	var ok bool
	s.withLock(func() {
		rp, isPred := s.resp.(RespPredicted)
		if !isPred {
			return
		}
		id = rp.ID
		epoch = rp.Epoch
		runner = rp.Runner
		s.resp = RespNone{Epoch: rp.Epoch + 1}
		if s.inflightPredicted > 0 {
			s.inflightPredicted--
		}
		ok = true
	})
	return id, epoch, runner, ok
}

func (s *phaseState) onPredictedPromote(epoch Epoch) (ResponseID, ItemID, *eagerRunner, bool) {
	var id ResponseID
	var itemID ItemID
	var runner *eagerRunner
	var ok bool
	s.withLock(func() {
		rp, isPred := s.resp.(RespPredicted)
		if !isPred || rp.Epoch != epoch {
			return
		}
		if _, speaking := s.vad.(VadSpeaking); speaking {
			return
		}
		if _, empty := s.buf.(BufEmpty); !empty {
			return
		}
		id = rp.ID
		itemID = rp.ItemID
		runner = rp.Runner
		s.resp = RespCreated{ID: rp.ID, ItemID: rp.ItemID, Epoch: rp.Epoch}

		if s.inflightPredicted > 0 {
			s.inflightPredicted--
		}
		ok = true
	})
	return id, itemID, runner, ok
}

func (s *phaseState) onResponseCreate(id ResponseID, itemID ItemID) (Epoch, error) {
	if id == "" {
		return 0, fmt.Errorf("%s", violationEmptyResponseID)
	}
	var epoch Epoch
	var err error
	s.withLock(func() {
		if _, active := s.session.(SessionActive); !active {
			err = fmt.Errorf("session not active")
			return
		}
		if _, speaking := s.vad.(VadSpeaking); speaking {
			err = fmt.Errorf("response.create while VAD speaking: item=%s", bufItemIDOf(s.buf))
			return
		}
		switch s.resp.(type) {
		case RespNone, RespFinalized:

		default:
			err = fmt.Errorf("response already active: id=%s", respIDOf(s.resp))
			return
		}
		if _, empty := s.buf.(BufEmpty); !empty {
			err = fmt.Errorf("response.create while input buffer is %s", s.buf.Kind())
			return
		}
		prevEpoch := respEpochOf(s.resp)
		newEpoch := prevEpoch + 1
		s.resp = RespCreated{ID: id, ItemID: itemID, Epoch: newEpoch}
		epoch = newEpoch
	})
	return epoch, err
}

func (s *phaseState) onResponseCancel() cancelEffect {
	var eff cancelEffect
	s.withLock(func() {
		var (
			id         ResponseID
			itemID     ItemID
			epoch      Epoch
			playedMs   Millis
			transcript string
			wasDrain   bool
			wasPredict bool
		)
		switch r := s.resp.(type) {
		case RespNone, RespFinalized:
			return
		case RespPredicted:
			id, itemID, epoch = r.ID, r.ItemID, r.Epoch
			wasPredict = true
		case RespCreated:
			id, itemID, epoch = r.ID, r.ItemID, r.Epoch
		case RespStreaming:
			id, itemID, epoch = r.ID, r.ItemID, r.Epoch
			playedMs = r.PlayedMs
			transcript = r.Transcript
		case RespDrain:
			id, itemID, epoch = r.ID, r.ItemID, r.Epoch
			playedMs = r.PlayedMs
			transcript = r.Transcript
			wasDrain = true
		}
		eff.cancelled = true
		eff.id = string(id)
		eff.itemID = string(itemID)
		eff.epoch = uint64(epoch)
		eff.playedMs = int64(playedMs)
		eff.transcript = transcript
		eff.wasDrain = wasDrain

		if it, ok := s.conv[itemID]; ok && it != nil {
			it.Status = itemIncomplete
			it.Transcript = transcript
			it.AudioEndMs = playedMs
		}
		s.resp = RespNone{Epoch: epoch + 1}
		if wasPredict && s.inflightPredicted > 0 {
			s.inflightPredicted--
		}
	})
	return eff
}

func (s *phaseState) onUpstreamDelta(epoch Epoch, textDelta string, plannedDeltaMs DurationMs) bool {
	var alive bool
	s.withLock(func() {
		switch r := s.resp.(type) {
		case RespCreated:
			if r.Epoch != epoch {
				return
			}
			alive = true
			s.resp = RespStreaming{
				ID:         r.ID,
				ItemID:     r.ItemID,
				Epoch:      r.Epoch,
				Transcript: textDelta,
				PlannedMs:  plannedDeltaMs,
			}
		case RespStreaming:
			if r.Epoch != epoch {
				return
			}
			alive = true
			r.Transcript += textDelta
			r.PlannedMs += plannedDeltaMs
			s.resp = r
		}
	})
	return alive
}

func (s *phaseState) updatePlayedMs(epoch Epoch, playedMs Millis) {
	s.withLock(func() {
		switch r := s.resp.(type) {
		case RespStreaming:
			if r.Epoch == epoch {
				r.PlayedMs = playedMs
				s.resp = r
			}
		case RespDrain:
			if r.Epoch == epoch {
				r.PlayedMs = playedMs
				s.resp = r
			}
		}
	})
}

func (s *phaseState) onLLMComplete(epoch Epoch) bool {
	var ok bool
	s.withLock(func() {
		switch r := s.resp.(type) {
		case RespCreated:
			if r.Epoch != epoch {
				return
			}
			s.resp = RespDrain{
				ID: r.ID, ItemID: r.ItemID, Epoch: r.Epoch,
			}
			ok = true
		case RespStreaming:
			if r.Epoch != epoch {
				return
			}
			s.resp = RespDrain{
				ID:         r.ID,
				ItemID:     r.ItemID,
				Epoch:      r.Epoch,
				Transcript: r.Transcript,
				PlannedMs:  r.PlannedMs,
				PlayedMs:   r.PlayedMs,
			}
			ok = true
		}
	})
	return ok
}

func (s *phaseState) onAudioDrained(epoch Epoch) bool {
	var ok bool
	s.withLock(func() {
		r, isDrain := s.resp.(RespDrain)
		if !isDrain || r.Epoch != epoch {
			return
		}
		s.resp = RespFinalized{
			ID:         r.ID,
			ItemID:     r.ItemID,
			Epoch:      r.Epoch,
			Status:     respStatusCompleted,
			Transcript: r.Transcript,
			PlayedMs:   r.PlayedMs,
		}
		if it, found := s.conv[r.ItemID]; found && it != nil {
			it.Status = itemCompleted
			it.Transcript = r.Transcript
		}
		ok = true
	})
	return ok
}

func (s *phaseState) onResponseDoneEmitted(epoch Epoch) (string, Millis, string, bool) {
	var transcript, status string
	var audioMs Millis
	var ok bool
	s.withLock(func() {
		r, isFin := s.resp.(RespFinalized)
		if !isFin || r.Epoch != epoch {
			return
		}
		transcript = r.Transcript
		audioMs = r.PlayedMs
		status = r.Status.String()
		s.resp = RespNone{Epoch: r.Epoch + 1}
		ok = true
	})
	return transcript, audioMs, status, ok
}

func (s *phaseState) onUpstreamComplete(epoch Epoch) (string, Millis, bool) {
	var transcript string
	var audioMs Millis
	var ok bool
	s.withLock(func() {
		switch r := s.resp.(type) {
		case RespCreated:
			if r.Epoch != epoch {
				return
			}
			s.resp = RespNone{Epoch: r.Epoch + 1}
			ok = true
		case RespStreaming:
			if r.Epoch != epoch {
				return
			}
			transcript = r.Transcript
			audioMs = r.PlayedMs
			s.resp = RespNone{Epoch: r.Epoch + 1}
			ok = true
		case RespDrain:
			if r.Epoch != epoch {
				return
			}
			transcript = r.Transcript
			audioMs = r.PlayedMs
			s.resp = RespNone{Epoch: r.Epoch + 1}
			ok = true
		}
	})
	return transcript, audioMs, ok
}

func (s *phaseState) responseEpoch() (respPhaseKind, Epoch, ResponseID) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.initialize()
	return s.resp.Kind(), respEpochOf(s.resp), respIDOf(s.resp)
}

func (s *phaseState) snapshot() (SessionPhase, VadPhase, RespPhase) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.initialize()
	return s.session, s.vad, s.resp
}

func (s *phaseState) snapshotFull() (SessionPhase, VadPhase, InputBuffer, RespPhase) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.initialize()
	return s.session, s.vad, s.buf, s.resp
}

func (s *phaseState) conversationSnapshot() []conversationItem {
	s.mu.Lock()
	defer s.mu.Unlock()
	out := make([]conversationItem, 0, len(s.convOrd))
	for _, id := range s.convOrd {
		if it, ok := s.conv[id]; ok && it != nil {
			out = append(out, *it)
		}
	}
	return out
}

func (s *phaseState) truncateItem(itemID ItemID, audioEndMs Millis, transcript string) bool {
	var ok bool
	s.withLock(func() {
		it, found := s.conv[itemID]
		if !found || it == nil {
			return
		}
		it.Status = itemIncomplete
		it.AudioEndMs = audioEndMs
		it.Transcript = transcript
		ok = true
	})
	return ok
}

func (s *phaseState) deleteItem(itemID ItemID) bool {
	var ok bool
	s.withLock(func() {
		if _, found := s.conv[itemID]; !found {
			return
		}
		delete(s.conv, itemID)
		for i, id := range s.convOrd {
			if id == itemID {
				s.convOrd = append(s.convOrd[:i], s.convOrd[i+1:]...)
				break
			}
		}
		ok = true
	})
	return ok
}

func (s *phaseState) sealedBufferFor(itemID ItemID) (sealedBuffer, bool) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.sealedBufs == nil {
		return sealedBuffer{}, false
	}
	b, ok := s.sealedBufs[itemID]
	return b, ok
}

func (s *phaseState) insertItem(item conversationItem) bool {
	var ok bool
	s.withLock(func() {
		if _, active := s.session.(SessionActive); !active {
			return
		}
		if s.conv == nil {
			s.conv = make(map[ItemID]*conversationItem)
		}
		if _, dup := s.conv[item.ID]; dup {
			return
		}
		copy := item
		s.conv[item.ID] = &copy
		s.convOrd = append(s.convOrd, item.ID)
		ok = true
	})
	return ok
}

func (s *phaseState) clearInputBuffer() bool {
	var ok bool
	s.withLock(func() {
		if _, empty := s.buf.(BufEmpty); empty {
			return
		}
		s.buf = BufEmpty{}
		s.vad = VadSilent{}
		ok = true
	})
	return ok
}
