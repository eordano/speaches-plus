package diarization

import (
	"os"
	"strconv"
)

const (

	defaultChunkSeconds        float32 = 16.0
	defaultHopRatio            float32 = 0.1
	defaultMedianFilterWindow          = 11
	defaultMinSpanFrames               = 8
	defaultClusteringThreshold float32 = 0.55
	defaultMaxSpeakers                 = 16

	envThreshold          = "DIAR_THRESHOLD"
	envMaxSpeakers        = "DIAR_MAX_SPEAKERS"
	envMinSpanFrames      = "DIAR_MIN_SPAN_FRAMES"
	envMedianFilterFrames = "DIAR_MEDIAN_FILTER_FRAMES"
)

type Segment struct {
	Speaker    ClusterID
	TStartMs   uint64
	TEndMs     uint64
	Confidence float32
}

type Config struct {
	ChunkSeconds        float32
	HopRatio            float32
	MedianFilterWindow  int
	MinSpanFrames       int
	ClusteringThreshold float32
	MaxSpeakers         int
}

func DefaultConfig() Config {
	cfg := Config{
		ChunkSeconds:        defaultChunkSeconds,
		HopRatio:            defaultHopRatio,
		MedianFilterWindow:  defaultMedianFilterWindow,
		MinSpanFrames:       defaultMinSpanFrames,
		ClusteringThreshold: defaultClusteringThreshold,
		MaxSpeakers:         defaultMaxSpeakers,
	}
	if v, ok := envFloat(envThreshold); ok {
		cfg.ClusteringThreshold = clamp01(v)
	}
	if n, ok := envInt(envMaxSpeakers); ok && n >= 1 {
		cfg.MaxSpeakers = n
	}
	if n, ok := envInt(envMinSpanFrames); ok && n > 0 {
		cfg.MinSpanFrames = n
	}
	if n, ok := envInt(envMedianFilterFrames); ok && n > 0 {
		cfg.MedianFilterWindow = n
	}
	return cfg
}

func clamp01(v float32) float32 {
	switch {
	case v < 0:
		return 0
	case v > 1:
		return 1
	default:
		return v
	}
}

func envFloat(key string) (float32, bool) {
	if s := os.Getenv(key); s != "" {
		if v, err := strconv.ParseFloat(s, 32); err == nil {
			return float32(v), true
		}
	}
	return 0, false
}

func envInt(key string) (int, bool) {
	if s := os.Getenv(key); s != "" {
		if n, err := strconv.Atoi(s); err == nil {
			return n, true
		}
	}
	return 0, false
}

type Diarizer struct {
	cfg       Config
	seg       *SegmentationModel
	emb       *EmbeddingModel
	decoder   *PowersetDecoder
	clusterer *OnlineClusterer
}

func NewDiarizer(seg *SegmentationModel, emb *EmbeddingModel, cfg Config) *Diarizer {
	return &Diarizer{
		cfg:       cfg,
		seg:       seg,
		emb:       emb,
		decoder:   NewPowersetDecoder(seg.MaxSpeakersPerChunk(), seg.MaxSpeakersPerFrame()),
		clusterer: NewOnlineClusterer(cfg.ClusteringThreshold, cfg.MaxSpeakers),
	}
}

func (d *Diarizer) DiarizeUtterance(audio []float32, tStartMs uint64) ([]Segment, error) {
	chunks := SlideChunks(audio, d.seg.SampleRate(), d.cfg.ChunkSeconds, d.cfg.HopRatio)

	var emitted []Segment
	for _, chunk := range chunks {
		logits, err := d.seg.Run(chunk.Samples)
		if err != nil {
			return nil, err
		}
		multihot := d.decoder.ToMultilabelHard(logits)
		smoothed := MedianFilterMultihot(multihot, d.cfg.MedianFilterWindow)
		spans := ExtractSpans(smoothed, d.seg.FrameRateHz(), chunk.TOffsetMs, d.cfg.MinSpanFrames)

		for _, span := range spans {
			end := span.SampleEnd
			if end > len(audio) {
				end = len(audio)
			}
			if end <= span.SampleStart {
				continue
			}
			spanAudio := audio[span.SampleStart:end]
			if len(spanAudio) < d.emb.MinInputSamples() {
				continue
			}
			emb, err := d.emb.Embed(spanAudio)
			if err != nil {
				return nil, err
			}
			clusterID, score := d.clusterer.Assign(emb)
			emitted = append(emitted, Segment{
				Speaker:    clusterID,
				TStartMs:   tStartMs + span.TStartMs,
				TEndMs:     tStartMs + span.TEndMs,
				Confidence: score,
			})
		}
	}

	return CoalesceSegments(emitted), nil
}

func (d *Diarizer) Reset() { d.clusterer.Reset() }
