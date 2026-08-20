package diarization

import "sort"

const (
	frameSampleRate    = 16000
	msPerSecond        = 1000
	overlapMinSpeakers = 2
	coalesceGapMs      = 250
)

type Chunk struct {
	Samples   []float32
	TOffsetMs uint64
}

type Span struct {
	SampleStart  int
	SampleEnd    int
	TStartMs     uint64
	TEndMs       uint64
	LocalSpeaker int
	Overlap      bool
}

type ChunkSpans struct {
	ChunkIndex int
	Spans      []Span
}

func SlideChunks(audio []float32, sampleRate uint32, chunkSeconds, hopRatio float32) []Chunk {
	chunkSamples := int(chunkSeconds * float32(sampleRate))
	hopSamples := int(chunkSeconds * hopRatio * float32(sampleRate))
	if hopSamples < 1 {
		hopSamples = 1
	}

	if len(audio) < chunkSamples {
		padded := make([]float32, chunkSamples)
		n := len(audio)
		if n > chunkSamples {
			n = chunkSamples
		}
		copy(padded, audio[:n])
		return []Chunk{{Samples: padded, TOffsetMs: 0}}
	}

	var out []Chunk
	for start := 0; start+chunkSamples <= len(audio); start += hopSamples {
		samples := make([]float32, chunkSamples)
		copy(samples, audio[start:start+chunkSamples])
		tOffsetMs := uint64(start) * msPerSecond / uint64(sampleRate)
		out = append(out, Chunk{Samples: samples, TOffsetMs: tOffsetMs})
	}
	return out
}

func MedianFilterMultihot(input *Multilabel, window int) *Multilabel {
	if window <= 1 {
		clone := &Multilabel{
			Frames:   input.Frames,
			Speakers: input.Speakers,
			Data:     make([]uint8, len(input.Data)),
		}
		copy(clone.Data, input.Data)
		return clone
	}
	half := window / 2
	out := &Multilabel{
		Frames:   input.Frames,
		Speakers: input.Speakers,
		Data:     make([]uint8, input.Frames*input.Speakers),
	}
	for f := 0; f < input.Frames; f++ {
		for s := 0; s < input.Speakers; s++ {
			lo := f - half
			if lo < 0 {
				lo = 0
			}
			hi := f + half + 1
			if hi > input.Frames {
				hi = input.Frames
			}
			ones, total := 0, hi-lo
			for ff := lo; ff < hi; ff++ {
				if input.Data[ff*input.Speakers+s] != 0 {
					ones++
				}
			}
			if ones*2 > total {
				out.Data[f*input.Speakers+s] = 1
			}
		}
	}
	return out
}

type spanCtx struct {
	overlap         []bool
	frameMs         float32
	tOffsetMs       uint64
	samplesPerFrame int
	minFrames       int
}

func ExtractSpans(multihot *Multilabel, frameRateHz uint32, tOffsetMs uint64, minFrames int) []Span {
	overlap := make([]bool, multihot.Frames)
	for f := 0; f < multihot.Frames; f++ {
		active := 0
		for _, v := range multihot.Row(f) {
			if v != 0 {
				active++
			}
		}
		overlap[f] = active >= overlapMinSpeakers
	}

	ctx := spanCtx{
		overlap:         overlap,
		frameMs:         float32(msPerSecond) / float32(frameRateHz),
		tOffsetMs:       tOffsetMs,
		samplesPerFrame: frameSampleRate / int(frameRateHz),
		minFrames:       minFrames,
	}

	var out []Span
	for s := 0; s < multihot.Speakers; s++ {
		runStart := -1
		for f := 0; f < multihot.Frames; f++ {
			active := multihot.Data[f*multihot.Speakers+s] != 0
			switch {
			case runStart < 0 && active:
				runStart = f
			case runStart >= 0 && !active:
				out = ctx.appendSpan(out, runStart, f, s)
				runStart = -1
			}
		}
		if runStart >= 0 {
			out = ctx.appendSpan(out, runStart, multihot.Frames, s)
		}
	}
	return out
}

func (c spanCtx) appendSpan(out []Span, start, end, speaker int) []Span {
	length := end - start
	if length < c.minFrames {
		return out
	}
	overlapFrames := 0
	for i := start; i < end; i++ {
		if c.overlap[i] {
			overlapFrames++
		}
	}
	tStartMs := c.tOffsetMs + uint64(float32(start)*c.frameMs)
	tEndMs := c.tOffsetMs + uint64(float32(end)*c.frameMs)
	baseSamples := int(c.tOffsetMs) * frameSampleRate / msPerSecond
	return append(out, Span{
		SampleStart:  baseSamples + start*c.samplesPerFrame,
		SampleEnd:    baseSamples + end*c.samplesPerFrame,
		TStartMs:     tStartMs,
		TEndMs:       tEndMs,
		LocalSpeaker: speaker,
		Overlap:      overlapFrames*2 > length,
	})
}

func CoalesceSegments(segments []Segment) []Segment {
	if len(segments) == 0 {
		return segments
	}
	sort.SliceStable(segments, func(i, j int) bool {
		return segments[i].TStartMs < segments[j].TStartMs
	})

	out := make([]Segment, 0, len(segments))
	for _, s := range segments {
		if n := len(out); n > 0 {
			last := &out[n-1]
			if last.Speaker == s.Speaker && s.TStartMs <= last.TEndMs+coalesceGapMs {
				if s.TEndMs > last.TEndMs {
					last.TEndMs = s.TEndMs
				}
				if s.Confidence > last.Confidence {
					last.Confidence = s.Confidence
				}
				continue
			}
		}
		out = append(out, s)
	}
	return out
}
