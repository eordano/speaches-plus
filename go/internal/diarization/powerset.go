package diarization

import "math"

type SegmentationLogits struct {
	Frames  int
	Classes int
	Data    []float32
}

func (s *SegmentationLogits) Row(frame int) []float32 {
	start := frame * s.Classes
	return s.Data[start : start+s.Classes]
}

type Multilabel struct {
	Frames   int
	Speakers int
	Data     []uint8
}

func (m *Multilabel) Row(frame int) []uint8 {
	start := frame * m.Speakers
	return m.Data[start : start+m.Speakers]
}

type PowersetDecoder struct {
	MaxSpeakersPerChunk int
	MaxSpeakersPerFrame int
	mapping             [][]int
}

func NewPowersetDecoder(maxSpeakersPerChunk, maxSpeakersPerFrame int) *PowersetDecoder {
	return &PowersetDecoder{
		MaxSpeakersPerChunk: maxSpeakersPerChunk,
		MaxSpeakersPerFrame: maxSpeakersPerFrame,
		mapping:             buildMapping(maxSpeakersPerChunk, maxSpeakersPerFrame),
	}
}

func (d *PowersetDecoder) NumClasses() int { return len(d.mapping) }

func (d *PowersetDecoder) ToMultilabelHard(logits *SegmentationLogits) *Multilabel {
	speakers := d.MaxSpeakersPerChunk
	out := &Multilabel{
		Frames:   logits.Frames,
		Speakers: speakers,
		Data:     make([]uint8, logits.Frames*speakers),
	}
	for f := 0; f < logits.Frames; f++ {
		cls := argmax(logits.Row(f))
		for _, spk := range d.mapping[cls] {
			out.Data[f*speakers+spk] = 1
		}
	}
	return out
}

func argmax(row []float32) int {
	best := 0
	bestV := float32(math.Inf(-1))
	for i, v := range row {
		if v > bestV {
			bestV = v
			best = i
		}
	}
	return best
}

func buildMapping(numClasses, maxSetSize int) [][]int {
	var out [][]int
	for size := 0; size <= maxSetSize; size++ {
		for _, combo := range combinations(numClasses, size) {
			out = append(out, combo)
		}
	}
	return out
}

func combinations(n, k int) [][]int {
	var result [][]int
	buf := make([]int, 0, k)
	var pick func(start int)
	pick = func(start int) {
		if len(buf) == k {
			cp := make([]int, k)
			copy(cp, buf)
			result = append(result, cp)
			return
		}
		for i := start; i < n; i++ {
			buf = append(buf, i)
			pick(i + 1)
			buf = buf[:len(buf)-1]
		}
	}
	pick(0)
	return result
}
