package vad

import (
	"fmt"
	"sync"

	ort "github.com/yalue/onnxruntime_go"
)

type Silero struct {
	mu         sync.Mutex
	session    *ort.DynamicAdvancedSession
	stateShape ort.Shape

	state   []float32
	context [ContextSamples]float32

	probs       []float32
	frameOffset int

	speaking   bool
	startFrame int

	threshold    float32
	negThreshold float32
	silenceMs    int
	minSpeechMs  int
	prefixPadMs  int
}

const MaxProbRing = (MaxVadWindowSamples + WindowSamples - 1) / WindowSamples

func New(modelPath string) (*Silero, error) {
	sess, err := ort.NewDynamicAdvancedSession(
		modelPath,
		[]string{"input", "state", "sr"},
		[]string{"output", "stateN"},
		nil,
	)
	if err != nil {
		return nil, fmt.Errorf("vad: load %q: %w", modelPath, err)
	}
	return &Silero{
		session:      sess,
		state:        make([]float32, 2*1*128),
		stateShape:   ort.NewShape(2, 1, 128),
		probs:        make([]float32, 0, MaxProbRing),
		threshold:    defaultThreshold,
		negThreshold: 0,
		silenceMs:    defaultSilenceMs,
		minSpeechMs:  defaultMinSpeechMs,
		prefixPadMs:  defaultPrefixPaddingMs,
	}, nil
}

func (s *Silero) SetThreshold(t float32) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if t <= 0 {
		t = defaultThreshold
	}
	s.threshold = t
}

func (s *Silero) SetNegThreshold(t float32) {
	s.mu.Lock()
	defer s.mu.Unlock()

	if t < 0 {
		t = 0
	}
	s.negThreshold = t
}

func (s *Silero) SetSilenceMs(ms int) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if ms <= 0 {
		ms = defaultSilenceMs
	}
	s.silenceMs = ms
}

func (s *Silero) SetMinSpeechMs(ms int) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if ms < 0 {
		ms = 0
	}
	s.minSpeechMs = ms
}

func (s *Silero) SetPrefixPaddingMs(ms int) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if ms < 0 {
		ms = 0
	}
	s.prefixPadMs = ms
}

func (s *Silero) Reset() {
	s.mu.Lock()
	defer s.mu.Unlock()
	for i := range s.state {
		s.state[i] = 0
	}
	for i := range s.context {
		s.context[i] = 0
	}
	s.probs = s.probs[:0]
	s.frameOffset = 0
	s.speaking = false
	s.startFrame = 0
}

type Decision int

func (s *Silero) Process(window []float32) (Decision, int, error) {
	if len(window) != WindowSamples {
		return None, 0, fmt.Errorf("vad: window must be %d samples (got %d)", WindowSamples, len(window))
	}

	s.mu.Lock()
	defer s.mu.Unlock()

	prob, err := s.runLocked(window)
	if err != nil {
		return None, 0, err
	}
	if len(s.probs) == MaxProbRing {

		copy(s.probs, s.probs[1:])
		s.probs = s.probs[:len(s.probs)-1]
	}
	s.probs = append(s.probs, prob)

	frame := s.frameOffset
	s.frameOffset++

	threshold := s.threshold
	if threshold <= 0 {
		threshold = defaultThreshold
	}
	negThreshold := s.negThreshold
	if negThreshold <= 0 {
		negThreshold = threshold - NegThresholdDelta
		if negThreshold < NegThresholdFloor {
			negThreshold = NegThresholdFloor
		}
	}
	silenceMs := s.silenceMs
	if silenceMs <= 0 {
		silenceMs = defaultSilenceMs
	}
	minSpeechMs := s.minSpeechMs
	if minSpeechMs < 0 {
		minSpeechMs = 0
	}
	prefixPadMs := s.prefixPadMs
	if prefixPadMs < 0 {
		prefixPadMs = 0
	}

	ringSamples := len(s.probs) * WindowSamples
	timestamps := speechTimestampsFromProbs(s.probs, ringSamples,
		threshold, negThreshold, silenceMs, minSpeechMs, prefixPadMs)

	ringMs := ringSamples * 1000 / SampleRate
	var lastEnd int
	haveLast := false
	if n := len(timestamps); n > 0 {
		lastEnd = timestamps[n-1].End * 1000 / SampleRate
		haveLast = true
	}

	if !s.speaking {
		if !haveLast {
			return None, 0, nil
		}
		s.speaking = true
		s.startFrame = frame
		return SpeechStart, s.startFrame, nil
	}

	if !haveLast {
		s.speaking = false
		return SpeechEnd, s.startFrame, nil
	}
	trailing := ringMs - lastEnd
	if trailing >= silenceMs {
		s.speaking = false
		return SpeechEnd, s.startFrame, nil
	}
	return None, 0, nil
}

func PrefixPaddingFrames() int {
	return framesForMs(defaultPrefixPaddingMs)
}

func framesForMs(ms int) int {
	return (ms * SampleRate) / (1000 * WindowSamples)
}

func (s *Silero) Close() error {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.session != nil {
		s.session.Destroy()
		s.session = nil
	}
	return nil
}

type SpeechTimestamp struct {
	Start int
	End   int
}

func speechTimestampsFromProbs(
	probs []float32,
	audioLength int,
	threshold float32,
	negThreshold float32,
	minSilenceMs int,
	minSpeechMs int,
	speechPadMs int,
) []SpeechTimestamp {
	speechPadSamples := speechPadMs * SampleRate / 1000
	minSpeechSamples := minSpeechMs * SampleRate / 1000
	minSilenceSamples := minSilenceMs * SampleRate / 1000
	minSilenceSamplesAtMaxSpeech := MinSilenceAtMaxSpeechMs * SampleRate / 1000
	maxSpeechSamples := 30*SampleRate - WindowSamples - 2*speechPadSamples

	speeches := make([]SpeechTimestamp, 0, 4)
	triggered := false
	currentStart := 0
	haveCurrent := false
	tempEnd := 0
	prevEnd := 0
	nextStart := 0

	for i, prob := range probs {
		pos := WindowSamples * i

		if prob >= threshold && tempEnd != 0 {
			tempEnd = 0
			if nextStart < prevEnd {
				nextStart = pos
			}
		}

		if prob >= threshold && !triggered {
			triggered = true
			currentStart = pos
			haveCurrent = true
			continue
		}

		if triggered && pos-currentStart > maxSpeechSamples {
			if prevEnd != 0 {
				speeches = append(speeches, SpeechTimestamp{Start: currentStart, End: prevEnd})
				haveCurrent = false
				if nextStart < prevEnd {
					triggered = false
				} else {
					currentStart = nextStart
					haveCurrent = true
				}
				prevEnd = 0
				nextStart = 0
				tempEnd = 0
			} else {
				speeches = append(speeches, SpeechTimestamp{Start: currentStart, End: pos})
				haveCurrent = false
				prevEnd = 0
				nextStart = 0
				tempEnd = 0
				triggered = false
				continue
			}
		}

		if prob < negThreshold && triggered {
			if tempEnd == 0 {
				tempEnd = pos
			}
			if pos-tempEnd > minSilenceSamplesAtMaxSpeech {
				prevEnd = tempEnd
			}
			if pos-tempEnd < minSilenceSamples {
				continue
			}
			segEnd := tempEnd
			if haveCurrent && segEnd > currentStart && segEnd-currentStart > minSpeechSamples {
				speeches = append(speeches, SpeechTimestamp{Start: currentStart, End: segEnd})
			}
			haveCurrent = false
			prevEnd = 0
			nextStart = 0
			tempEnd = 0
			triggered = false
		}
	}

	if haveCurrent && audioLength > currentStart && audioLength-currentStart > minSpeechSamples {
		speeches = append(speeches, SpeechTimestamp{Start: currentStart, End: audioLength})
	}

	n := len(speeches)
	for i := 0; i < n; i++ {
		if i == 0 {
			if speeches[i].Start < speechPadSamples {
				speeches[i].Start = 0
			} else {
				speeches[i].Start -= speechPadSamples
			}
		}
		if i != n-1 {
			silence := speeches[i+1].Start - speeches[i].End
			if silence < 2*speechPadSamples {
				half := silence / 2
				speeches[i].End += half
				if speeches[i+1].Start < half {
					speeches[i+1].Start = 0
				} else {
					speeches[i+1].Start -= half
				}
			} else {
				if speeches[i].End+speechPadSamples > audioLength {
					speeches[i].End = audioLength
				} else {
					speeches[i].End += speechPadSamples
				}
				if speeches[i+1].Start < speechPadSamples {
					speeches[i+1].Start = 0
				} else {
					speeches[i+1].Start -= speechPadSamples
				}
			}
		} else {
			if speeches[i].End+speechPadSamples > audioLength {
				speeches[i].End = audioLength
			} else {
				speeches[i].End += speechPadSamples
			}
		}
	}

	return speeches
}

func (s *Silero) runLocked(window []float32) (float32, error) {
	var inBuf [ContextSamples + WindowSamples]float32
	copy(inBuf[:ContextSamples], s.context[:])
	copy(inBuf[ContextSamples:], window)
	copy(s.context[:], window[len(window)-ContextSamples:])

	in, err := ort.NewTensor(ort.NewShape(1, int64(ContextSamples+len(window))), inBuf[:])
	if err != nil {
		return 0, err
	}
	defer in.Destroy()

	st, err := ort.NewTensor(s.stateShape, s.state)
	if err != nil {
		return 0, err
	}
	defer st.Destroy()

	sr, err := ort.NewTensor(ort.NewShape(1), []int64{int64(SampleRate)})
	if err != nil {
		return 0, err
	}
	defer sr.Destroy()

	outputs := []ort.Value{nil, nil}
	if err := s.session.Run([]ort.Value{in, st, sr}, outputs); err != nil {
		return 0, err
	}
	defer outputs[0].Destroy()
	defer outputs[1].Destroy()

	prob := outputs[0].(*ort.Tensor[float32]).GetData()
	stateN := outputs[1].(*ort.Tensor[float32]).GetData()
	if len(stateN) == len(s.state) {
		copy(s.state, stateN)
	}
	if len(prob) == 0 {
		return 0, fmt.Errorf("vad: empty output")
	}
	return prob[0], nil
}
