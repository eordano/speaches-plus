package pii

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"sync"

	ort "github.com/yalue/onnxruntime_go"
)

type Classifier struct {
	session   *ort.DynamicAdvancedSession
	tokenizer *piiTokenizer
	labels    []string
	mu        sync.Mutex
}

type classifierConfig struct {
	ID2Label map[string]string `json:"id2label"`
}

func NewClassifier(modelDir string, device string) (*Classifier, error) {
	modelPath := filepath.Join(modelDir, "model.onnx")
	if _, err := os.Stat(modelPath); err != nil {
		return nil, fmt.Errorf("pii: model not found at %s: %w", modelPath, err)
	}

	tokPath := filepath.Join(modelDir, "tokenizer.json")
	tok, err := loadPiiTokenizer(tokPath)
	if err != nil {
		return nil, fmt.Errorf("pii: tokenizer: %w", err)
	}

	configPath := filepath.Join(modelDir, "config.json")
	labels, err := loadLabels(configPath)
	if err != nil {
		return nil, fmt.Errorf("pii: config: %w", err)
	}

	inNames := []string{"input_ids", "attention_mask"}
	outNames := []string{"logits"}

	sess, err := ort.NewDynamicAdvancedSession(modelPath, inNames, outNames, nil)
	if err != nil {
		return nil, fmt.Errorf("pii: onnx session: %w", err)
	}

	return &Classifier{
		session:   sess,
		tokenizer: tok,
		labels:    labels,
	}, nil
}

func (c *Classifier) Close() error {
	c.mu.Lock()
	defer c.mu.Unlock()
	if c.session != nil {
		c.session.Destroy()
		c.session = nil
	}
	return nil
}

func (c *Classifier) ClassifyOne(text string) ([]PiiSpan, error) {
	if len(text) == 0 {
		return []PiiSpan{}, nil
	}
	enc := c.tokenizer.Encode(text)
	if len(enc.IDs) == 0 {
		return []PiiSpan{}, nil
	}

	logits, err := c.runInference(enc.IDs, enc.AttentionMask)
	if err != nil {
		return nil, err
	}

	path := ViterbiDecode(logits, c.labels)
	labelNames := make([]string, len(path))
	for i, idx := range path {
		labelNames[i] = c.labels[idx]
	}

	return AssembleSpans(labelNames, enc.Offsets, enc.AttentionMask), nil
}

func (c *Classifier) ClassifyBatch(texts []string) ([][]PiiSpan, error) {
	if len(texts) == 0 {
		return nil, nil
	}

	results := make([][]PiiSpan, len(texts))
	for i, text := range texts {
		spans, err := c.ClassifyOne(text)
		if err != nil {
			return nil, fmt.Errorf("pii: batch item %d: %w", i, err)
		}
		results[i] = spans
	}
	return results, nil
}

func (c *Classifier) runInference(ids []int, mask []int) ([][]float32, error) {
	c.mu.Lock()
	defer c.mu.Unlock()
	if c.session == nil {
		return nil, fmt.Errorf("pii: session destroyed")
	}

	n := int64(len(ids))
	inputIDs := make([]int64, len(ids))
	attMask := make([]int64, len(ids))
	for i := range ids {
		inputIDs[i] = int64(ids[i])
		attMask[i] = int64(mask[i])
	}

	inTensor, err := ort.NewTensor(ort.NewShape(1, n), inputIDs)
	if err != nil {
		return nil, err
	}
	defer inTensor.Destroy()

	maskTensor, err := ort.NewTensor(ort.NewShape(1, n), attMask)
	if err != nil {
		return nil, err
	}
	defer maskTensor.Destroy()

	outputs := make([]ort.Value, 1)
	if err := c.session.Run([]ort.Value{inTensor, maskTensor}, outputs); err != nil {
		return nil, fmt.Errorf("pii: onnx run: %w", err)
	}
	if outputs[0] != nil {
		defer outputs[0].Destroy()
	}

	tensor, ok := outputs[0].(*ort.Tensor[float32])
	if !ok {
		return nil, fmt.Errorf("pii: unexpected output type %T", outputs[0])
	}
	data := tensor.GetData()
	shape := tensor.GetShape()

	T := int(shape[1])
	L := int(shape[2])
	logits := make([][]float32, T)
	for t := 0; t < T; t++ {
		row := make([]float32, L)
		copy(row, data[t*L:(t+1)*L])
		logits[t] = row
	}
	return logits, nil
}

func loadLabels(configPath string) ([]string, error) {
	raw, err := os.ReadFile(configPath)
	if err != nil {
		return nil, err
	}
	var cfg classifierConfig
	if err := json.Unmarshal(raw, &cfg); err != nil {
		return nil, err
	}
	if len(cfg.ID2Label) == 0 {
		return nil, fmt.Errorf("config.json has no id2label")
	}
	labels := make([]string, len(cfg.ID2Label))
	for i := 0; i < len(labels); i++ {
		key := fmt.Sprintf("%d", i)
		l, ok := cfg.ID2Label[key]
		if !ok {
			return nil, fmt.Errorf("id2label missing key %q", key)
		}
		labels[i] = l
	}
	return labels, nil
}
