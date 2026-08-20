package stt

type Result struct {
	Text             string
	AvgLogprob       *float32
	NoSpeechProb     *float32
	CompressionRatio *float32

	Segments []Segment
}

type Segment struct {
	TStartMs     uint32
	TEndMs       uint32
	Text         string
	AvgLogprob   *float32
	NoSpeechProb *float32
}

type Transcriber interface {
	Transcribe(samples []float32, sampleRate int) (string, error)
	Close() error
}

type FullTranscriber interface {
	TranscribeFull(samples []float32, sampleRate int) (Result, error)
}

type SegmentTranscriber interface {
	TranscribeSegments(samples []float32, sampleRate int) (Result, error)
}
