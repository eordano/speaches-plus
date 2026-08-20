package diarization

import (
	"fmt"
	"os"
	"sync"

	ort "github.com/yalue/onnxruntime_go"
)

const (
	SegmentationSampleRate      uint32 = 16000
	SegmentationFrameRateHz     uint32 = 50
	SegmentationSamplesPerFrame        = int(SegmentationSampleRate / SegmentationFrameRateHz)

	DefaultMaxSpeakersPerChunk = 4
	DefaultMaxSpeakersPerFrame = 2

	segmentationInputName  = "waveform"
	segmentationOutputName = "scores"
	envOnnxRuntimeLib      = "ONNXRUNTIME_LIB"
)

var (
	ortInitOnce sync.Once
	ortInitErr  error
)

func initOnnxRuntime() error {
	ortInitOnce.Do(func() {
		if libPath := os.Getenv(envOnnxRuntimeLib); libPath != "" {
			ort.SetSharedLibraryPath(libPath)
		}
		ortInitErr = ort.InitializeEnvironment()
	})
	return ortInitErr
}

type SegmentationModel struct {
	mu                  sync.Mutex
	session             *ort.DynamicAdvancedSession
	maxSpeakersPerChunk int
	maxSpeakersPerFrame int
}

func LoadSegmentation(modelPath string) (*SegmentationModel, error) {
	if modelPath == "" {
		return nil, fmt.Errorf("segmentation: empty model path")
	}
	if err := initOnnxRuntime(); err != nil {
		return nil, fmt.Errorf("segmentation: init onnxruntime: %w", err)
	}
	sess, err := ort.NewDynamicAdvancedSession(
		modelPath,
		[]string{segmentationInputName},
		[]string{segmentationOutputName},
		nil,
	)
	if err != nil {
		return nil, fmt.Errorf("segmentation: load %q: %w", modelPath, err)
	}
	return &SegmentationModel{
		session:             sess,
		maxSpeakersPerChunk: DefaultMaxSpeakersPerChunk,
		maxSpeakersPerFrame: DefaultMaxSpeakersPerFrame,
	}, nil
}

func (m *SegmentationModel) SampleRate() uint32       { return SegmentationSampleRate }
func (m *SegmentationModel) FrameRateHz() uint32      { return SegmentationFrameRateHz }
func (m *SegmentationModel) MaxSpeakersPerChunk() int { return m.maxSpeakersPerChunk }
func (m *SegmentationModel) MaxSpeakersPerFrame() int { return m.maxSpeakersPerFrame }

func (m *SegmentationModel) Close() error {
	m.mu.Lock()
	defer m.mu.Unlock()
	if m.session != nil {
		m.session.Destroy()
		m.session = nil
	}
	return nil
}

func (m *SegmentationModel) Run(samples []float32) (*SegmentationLogits, error) {
	if len(samples) == 0 {
		return nil, fmt.Errorf("segmentation: empty input")
	}

	in, err := ort.NewTensor(ort.NewShape(1, 1, int64(len(samples))), samples)
	if err != nil {
		return nil, fmt.Errorf("segmentation: input tensor: %w", err)
	}
	defer in.Destroy()

	m.mu.Lock()
	defer m.mu.Unlock()
	if m.session == nil {
		return nil, fmt.Errorf("segmentation: session closed")
	}

	outputs := []ort.Value{nil}
	if err := m.session.Run([]ort.Value{in}, outputs); err != nil {
		return nil, fmt.Errorf("segmentation: run: %w", err)
	}
	defer outputs[0].Destroy()

	tensor, ok := outputs[0].(*ort.Tensor[float32])
	if !ok {
		return nil, fmt.Errorf("segmentation: unexpected output type %T", outputs[0])
	}
	shape := tensor.GetShape()
	if len(shape) != 3 {
		return nil, fmt.Errorf("segmentation: expected 3D output, got shape %v", shape)
	}
	frames, classes := int(shape[1]), int(shape[2])
	src := tensor.GetData()
	if frames*classes != len(src) {
		return nil, fmt.Errorf("segmentation: shape %v disagrees with %d elements", shape, len(src))
	}
	data := make([]float32, len(src))
	copy(data, src)
	return &SegmentationLogits{Frames: frames, Classes: classes, Data: data}, nil
}
