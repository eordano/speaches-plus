package diarization

import (
	"fmt"
	"math"
	"sync"

	ort "github.com/yalue/onnxruntime_go"
)

const (
	EmbeddingSampleRate         uint32 = 16000
	EmbeddingDim                       = 256
	EmbeddingFrameLengthSamples        = 400
	EmbeddingFrameShiftSamples         = 160
	EmbeddingNumMelBins                = 80
	EmbeddingMinInputSamples           = 16000

	embeddingInputName  = "feats"
	embeddingOutputName = "embs"
)

type EmbeddingModel struct {
	mu      sync.Mutex
	session *ort.DynamicAdvancedSession
	fbank   *FBank
}

func LoadEmbedding(modelPath string) (*EmbeddingModel, error) {
	return LoadEmbeddingWithIONames(modelPath, embeddingInputName, embeddingOutputName)
}

func LoadEmbeddingWithIONames(modelPath, inputName, outputName string) (*EmbeddingModel, error) {
	if modelPath == "" {
		return nil, fmt.Errorf("embedding: empty model path")
	}
	if err := initOnnxRuntime(); err != nil {
		return nil, fmt.Errorf("embedding: init onnxruntime: %w", err)
	}
	sess, err := ort.NewDynamicAdvancedSession(
		modelPath,
		[]string{inputName},
		[]string{outputName},
		nil,
	)
	if err != nil {
		return nil, fmt.Errorf("embedding: load %q: %w", modelPath, err)
	}
	return &EmbeddingModel{
		session: sess,
		fbank:   NewFBank(EmbeddingNumMelBins, EmbeddingFrameLengthSamples, EmbeddingFrameShiftSamples),
	}, nil
}

func (m *EmbeddingModel) SampleRate() uint32   { return EmbeddingSampleRate }
func (m *EmbeddingModel) Dim() int             { return EmbeddingDim }
func (m *EmbeddingModel) MinInputSamples() int { return EmbeddingMinInputSamples }

func (m *EmbeddingModel) Close() error {
	m.mu.Lock()
	defer m.mu.Unlock()
	if m.session != nil {
		m.session.Destroy()
		m.session = nil
	}
	return nil
}

func (m *EmbeddingModel) Embed(samples []float32) ([]float32, error) {
	if len(samples) < EmbeddingFrameLengthSamples {
		return nil, fmt.Errorf("embedding: input %d samples shorter than frame length %d",
			len(samples), EmbeddingFrameLengthSamples)
	}

	feats, err := m.fbank.Compute(samples)
	if err != nil {
		return nil, err
	}
	frames := len(feats) / EmbeddingNumMelBins

	in, err := ort.NewTensor(ort.NewShape(1, int64(frames), int64(EmbeddingNumMelBins)), feats)
	if err != nil {
		return nil, fmt.Errorf("embedding: input tensor: %w", err)
	}
	defer in.Destroy()

	m.mu.Lock()
	defer m.mu.Unlock()
	if m.session == nil {
		return nil, fmt.Errorf("embedding: session closed")
	}

	outputs := []ort.Value{nil}
	if err := m.session.Run([]ort.Value{in}, outputs); err != nil {
		return nil, fmt.Errorf("embedding: run: %w", err)
	}
	defer outputs[0].Destroy()

	tensor, ok := outputs[0].(*ort.Tensor[float32])
	if !ok {
		return nil, fmt.Errorf("embedding: unexpected output type %T", outputs[0])
	}
	shape := tensor.GetShape()
	if len(shape) == 0 || int(shape[len(shape)-1]) != EmbeddingDim {
		return nil, fmt.Errorf("embedding: expected last dim %d, got shape %v", EmbeddingDim, shape)
	}
	src := tensor.GetData()
	out := make([]float32, len(src))
	copy(out, src)
	l2Normalize(out)
	return out, nil
}

func l2Normalize(v []float32) {
	var sum float32
	for _, x := range v {
		sum += x * x
	}
	norm := float32(math.Sqrt(float64(sum)))
	if norm < normFloor {
		norm = normFloor
	}
	inv := 1 / norm
	for i := range v {
		v[i] *= inv
	}
}
