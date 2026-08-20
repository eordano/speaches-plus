package stt

import (
	"strings"
)

const silencePeakThreshold = 0.005

func nan32(f float32) bool { return f != f }

func peakAmplitude(samples []float32) float32 {
	var peak float32
	for _, s := range samples {
		if s < 0 {
			s = -s
		}
		if s > peak {
			peak = s
		}
	}
	return peak
}

func chunkAudio(samples []float32, sampleRate int) []struct {
	offsetMs int
	data     []float32
} {
	chunkSamples := whisperChunkSecs * sampleRate
	if len(samples) <= chunkSamples {
		return []struct {
			offsetMs int
			data     []float32
		}{{0, samples}}
	}
	var out []struct {
		offsetMs int
		data     []float32
	}
	pos := 0
	for pos < len(samples) {
		end := pos + chunkSamples
		if end > len(samples) {
			end = len(samples)
		}
		offsetMs := (pos * 1000) / sampleRate
		out = append(out, struct {
			offsetMs int
			data     []float32
		}{offsetMs, samples[pos:end]})
		pos = end
	}
	return out
}

func shiftSegments(segs []Segment, offsetMs int) []Segment {
	out := make([]Segment, len(segs))
	for i, s := range segs {
		out[i] = Segment{
			TStartMs:     s.TStartMs + uint32(offsetMs),
			TEndMs:       s.TEndMs + uint32(offsetMs),
			Text:         s.Text,
			AvgLogprob:   s.AvgLogprob,
			NoSpeechProb: s.NoSpeechProb,
		}
	}
	return out
}

func TranscribeLong(t Transcriber, samples []float32, sampleRate int) (string, error) {
	if peakAmplitude(samples) < silencePeakThreshold {
		return "", nil
	}
	chunks := chunkAudio(samples, sampleRate)
	if len(chunks) == 1 {
		return t.Transcribe(samples, sampleRate)
	}
	var texts []string
	for _, c := range chunks {
		if peakAmplitude(c.data) < silencePeakThreshold {
			continue
		}
		text, err := t.Transcribe(c.data, sampleRate)
		if err != nil {
			return "", err
		}
		text = strings.TrimSpace(text)
		if text != "" {
			texts = append(texts, text)
		}
	}
	return strings.Join(texts, " "), nil
}

func TranscribeFullLong(t FullTranscriber, samples []float32, sampleRate int) (Result, error) {
	if peakAmplitude(samples) < silencePeakThreshold {
		return Result{}, nil
	}
	chunkSamples := whisperChunkSecs * sampleRate
	if len(samples) <= chunkSamples {
		return t.TranscribeFull(samples, sampleRate)
	}
	chunks := chunkAudio(samples, sampleRate)
	var allSegments []Segment
	var texts []string
	var lpSum, lpWeight, nspSum, nspWeight float64
	for _, c := range chunks {
		if peakAmplitude(c.data) < silencePeakThreshold {
			continue
		}
		res, err := t.TranscribeFull(c.data, sampleRate)
		if err != nil {
			return Result{}, err
		}
		text := strings.TrimSpace(res.Text)
		if text == "" && len(res.Segments) == 0 {
			continue
		}
		if text != "" {
			texts = append(texts, text)
		}
		durMs := float64(len(c.data)*1000) / float64(sampleRate)
		if res.AvgLogprob != nil && !nan32(*res.AvgLogprob) {
			lpSum += float64(*res.AvgLogprob) * durMs
			lpWeight += durMs
		}
		if res.NoSpeechProb != nil && !nan32(*res.NoSpeechProb) {
			nspSum += float64(*res.NoSpeechProb) * durMs
			nspWeight += durMs
		}
		shifted := shiftSegments(res.Segments, c.offsetMs)
		allSegments = append(allSegments, shifted...)
	}
	result := Result{
		Text:     strings.Join(texts, " "),
		Segments: allSegments,
	}
	if lpWeight > 0 {
		v := float32(lpSum / lpWeight)
		result.AvgLogprob = &v
	}
	if nspWeight > 0 {
		v := float32(nspSum / nspWeight)
		result.NoSpeechProb = &v
	}
	return result, nil
}

func TranscribeSegmentsLong(t SegmentTranscriber, samples []float32, sampleRate int) (Result, error) {
	if peakAmplitude(samples) < silencePeakThreshold {
		return Result{}, nil
	}
	chunkSamples := whisperChunkSecs * sampleRate
	if len(samples) <= chunkSamples {
		return t.TranscribeSegments(samples, sampleRate)
	}
	chunks := chunkAudio(samples, sampleRate)
	var allSegments []Segment
	var texts []string
	var lpSum, lpWeight, nspSum, nspWeight float64
	for _, c := range chunks {
		if peakAmplitude(c.data) < silencePeakThreshold {
			continue
		}
		res, err := t.TranscribeSegments(c.data, sampleRate)
		if err != nil {
			return Result{}, err
		}
		text := strings.TrimSpace(res.Text)
		if text == "" && len(res.Segments) == 0 {
			continue
		}
		if text != "" {
			texts = append(texts, text)
		}
		durMs := float64(len(c.data)*1000) / float64(sampleRate)
		if res.AvgLogprob != nil && !nan32(*res.AvgLogprob) {
			lpSum += float64(*res.AvgLogprob) * durMs
			lpWeight += durMs
		}
		if res.NoSpeechProb != nil && !nan32(*res.NoSpeechProb) {
			nspSum += float64(*res.NoSpeechProb) * durMs
			nspWeight += durMs
		}
		shifted := shiftSegments(res.Segments, c.offsetMs)
		allSegments = append(allSegments, shifted...)
	}
	result := Result{
		Text:     strings.Join(texts, " "),
		Segments: allSegments,
	}
	if lpWeight > 0 {
		v := float32(lpSum / lpWeight)
		result.AvgLogprob = &v
	}
	if nspWeight > 0 {
		v := float32(nspSum / nspWeight)
		result.NoSpeechProb = &v
	}
	return result, nil
}
