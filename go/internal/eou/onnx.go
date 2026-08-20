package eou

import (
	"context"
	"fmt"
	"math"
	"path/filepath"
	"sync"
	"time"

	ort "github.com/yalue/onnxruntime_go"
)

type ONNXModel struct {
	mu sync.Mutex

	session   *ort.DynamicAdvancedSession
	tokenizer *Tokenizer

	heuristic *Heuristic

	maxCtx   int
	imEndID  int
	inNames  []string
	outNames []string
}

type ONNXOptions struct {
	MaxContextTokens int
	TokenizerPath    string
	InputNames       []string
	OutputNames      []string
}

func NewONNXModel(modelPath string, opts ONNXOptions) (*ONNXModel, error) {
	if opts.MaxContextTokens <= 0 {
		opts.MaxContextTokens = defaultMaxContextTokens
	}

	tokPath := opts.TokenizerPath
	if tokPath == "" {
		tokPath = filepath.Join(filepath.Dir(modelPath), "tokenizer.json")
	}
	tok, err := LoadTokenizer(tokPath)
	if err != nil {
		return nil, fmt.Errorf("eou onnx: tokenizer: %w", err)
	}
	if tok.ImEndID() < 0 {
		return nil, fmt.Errorf("eou onnx: tokenizer has no <|im_end|> token")
	}

	inNames := opts.InputNames
	if len(inNames) == 0 {
		inNames = []string{onnxInputIDs, onnxAttentionMask}
	}
	outNames := opts.OutputNames
	if len(outNames) == 0 {
		outNames = []string{onnxOutputLogits}
	}
	sess, err := ort.NewDynamicAdvancedSession(modelPath, inNames, outNames, nil)
	if err != nil {
		return nil, fmt.Errorf("eou onnx: load %q: %w", modelPath, err)
	}

	return &ONNXModel{
		session:   sess,
		tokenizer: tok,
		heuristic: NewHeuristic(),
		maxCtx:    opts.MaxContextTokens,
		imEndID:   tok.ImEndID(),
		inNames:   inNames,
		outNames:  outNames,
	}, nil
}

func (m *ONNXModel) Close() error {
	m.mu.Lock()
	defer m.mu.Unlock()
	if m.session != nil {
		m.session.Destroy()
		m.session = nil
	}
	return nil
}

func (m *ONNXModel) Predict(ctx context.Context, req Request) (Verdict, error) {
	t0 := time.Now()

	prompt := FormatQwenChat(req.Turns, req.Partial)
	ids := m.tokenizer.Encode(prompt)
	if len(ids) == 0 {
		v, _ := m.heuristic.Predict(ctx, req)
		v.Latency = time.Since(t0)
		return v, nil
	}
	if len(ids) > m.maxCtx {
		ids = ids[len(ids)-m.maxCtx:]
	}

	select {
	case <-ctx.Done():
		return Verdict{Latency: time.Since(t0)}, ctx.Err()
	default:
	}

	score, err := m.runInference(ids)
	if err != nil {
		return Verdict{Latency: time.Since(t0)}, err
	}
	return Verdict{Score: score, Latency: time.Since(t0)}, nil
}

func (m *ONNXModel) runInference(ids []int) (float32, error) {
	m.mu.Lock()
	defer m.mu.Unlock()
	if m.session == nil {
		return 0, fmt.Errorf("eou onnx: session destroyed")
	}

	n := int64(len(ids))
	inputIDs := make([]int64, len(ids))
	mask := make([]int64, len(ids))
	for i, id := range ids {
		inputIDs[i] = int64(id)
		mask[i] = 1
	}

	inTensor, err := ort.NewTensor(ort.NewShape(1, n), inputIDs)
	if err != nil {
		return 0, err
	}
	defer inTensor.Destroy()

	inputs := []ort.Value{inTensor}
	if len(m.inNames) >= 2 {
		maskTensor, err := ort.NewTensor(ort.NewShape(1, n), mask)
		if err != nil {
			return 0, err
		}
		defer maskTensor.Destroy()
		inputs = append(inputs, maskTensor)
	}

	outputs := make([]ort.Value, len(m.outNames))
	if err := m.session.Run(inputs, outputs); err != nil {
		return 0, fmt.Errorf("eou onnx: run: %w", err)
	}
	for _, o := range outputs {
		if o != nil {
			defer o.Destroy()
		}
	}

	tensor, ok := outputs[0].(*ort.Tensor[float32])
	if !ok {
		return 0, fmt.Errorf("eou onnx: unexpected output type %T", outputs[0])
	}
	logits := tensor.GetData()
	shape := tensor.GetShape()
	return extractImEndProb(logits, shape, m.imEndID)
}

func extractImEndProb(logits []float32, shape []int64, imEndID int) (float32, error) {
	if len(shape) < 2 || len(logits) == 0 {
		return 0, fmt.Errorf("eou onnx: empty logits (shape=%v)", shape)
	}
	vocab := int(shape[len(shape)-1])
	if vocab <= 0 || imEndID >= vocab {
		return 0, fmt.Errorf("eou onnx: imEndID %d out of vocab %d", imEndID, vocab)
	}
	lastStart := len(logits) - vocab
	if lastStart < 0 {
		return 0, fmt.Errorf("eou onnx: logits length %d < vocab %d", len(logits), vocab)
	}
	row := logits[lastStart : lastStart+vocab]
	maxLogit := row[0]
	for _, v := range row[1:] {
		if v > maxLogit {
			maxLogit = v
		}
	}
	var sum float64
	for _, v := range row {
		sum += math.Exp(float64(v - maxLogit))
	}
	if sum <= 0 {
		return 0, fmt.Errorf("eou onnx: degenerate logits (sum<=0)")
	}
	p := math.Exp(float64(row[imEndID]-maxLogit)) / sum
	return float32(p), nil
}
