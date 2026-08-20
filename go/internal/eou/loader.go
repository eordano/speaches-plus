package eou

import (
	"log/slog"
	"os"
	"path/filepath"
	"strings"
)

type Config struct {
	Kind                Kind
	ModelPath           string
	TokenizerPath       string
	LanguagesPath       string
	AudioModelPath      string
	MinDelayMs          int
	MaxDelayMs          int
	HardCapMs           int
	InferenceTimeoutMs  int
	ContextTurns        int
	MaxContextTokens    int
	PThreshold          float32
	EagerPThreshold     float32
	EotThreshold        float32
	EagerEotThreshold   float32
	EagerMaxInflight    int
	EagerPeriodicEnable bool
	EagerIntervalMs     int
	MinSpeechForCommit  int
	Eagerness           Eagerness
	Languages           LanguageTable
	AudioPadAlignment   string
	AudioWindowMs       int
	FusionRule          FusionRule
	FusionWeightText    float32
	TextModel           Model
	AudioModel          Model
}

type Eagerness string

func (e Eagerness) Apply(c *Config) {
	switch e {
	case EagernessLow:
		c.PThreshold, c.MinDelayMs, c.MaxDelayMs = eagernessLowPThreshold, eagernessLowMinDelayMs, eagernessLowMaxDelayMs
	case EagernessMedium:
		c.PThreshold, c.MinDelayMs, c.MaxDelayMs = eagernessMediumPThreshold, eagernessMediumMinDelayMs, eagernessMediumMaxDelayMs
	case EagernessHigh:
		c.PThreshold, c.MinDelayMs, c.MaxDelayMs = eagernessHighPThreshold, eagernessHighMinDelayMs, eagernessHighMaxDelayMs
	}
}

func (c Config) WithDefaults() Config {
	if c.Kind == "" {
		c.Kind = KindVad
	}
	if c.MinDelayMs == 0 {
		c.MinDelayMs = defaultMinDelayMs
	}
	if c.MaxDelayMs == 0 {
		c.MaxDelayMs = defaultMaxDelayMs
	}
	if c.HardCapMs == 0 {
		c.HardCapMs = defaultHardCapMs
	}
	if c.InferenceTimeoutMs == 0 {
		c.InferenceTimeoutMs = defaultInferenceTimeoutMs
	}
	if c.ContextTurns == 0 {
		c.ContextTurns = defaultContextTurns
	}
	if c.EagerMaxInflight == 0 {
		c.EagerMaxInflight = defaultEagerMaxInflight
	}
	if c.EagerIntervalMs == 0 {
		c.EagerIntervalMs = defaultEagerIntervalMs
	}
	if c.MinSpeechForCommit == 0 {
		c.MinSpeechForCommit = defaultMinSpeechForCommit
	}
	if c.MaxContextTokens == 0 {
		c.MaxContextTokens = defaultMaxContextTokens
	}
	if c.PThreshold == 0 {
		c.PThreshold = defaultPThreshold
	}
	if c.EotThreshold == 0 {
		c.EotThreshold = defaultEotThreshold
	}
	if c.EagerEotThreshold == 0 {
		c.EagerEotThreshold = defaultEagerEotThreshold
	}
	if c.AudioWindowMs == 0 {
		c.AudioWindowMs = defaultAudioWindowMs
	}
	if c.FusionRule == "" {
		c.FusionRule = defaultFusionRule
	}

	if v := strings.TrimSpace(os.Getenv("EOU_FUSION_RULE")); v != "" {
		switch FusionRule(strings.ToLower(v)) {
		case FusionNoisyOr, FusionMax, FusionMean, FusionWeighted, FusionGated:
			c.FusionRule = FusionRule(strings.ToLower(v))
		default:
			slog.Warn("eou: unknown EOU_FUSION_RULE; using default",
				"requested", v, "default", c.FusionRule)
		}
	}
	if c.FusionWeightText == 0 {
		c.FusionWeightText = defaultFusionWeightText
	}
	if c.Eagerness != EagernessUnset {
		c.Eagerness.Apply(&c)
	}
	if c.LanguagesPath == "" && c.ModelPath != "" {
		guess := filepath.Join(filepath.Dir(c.ModelPath), "languages.json")
		if _, err := os.Stat(guess); err == nil {
			c.LanguagesPath = guess
		}
	}
	if c.Languages == nil {
		tbl, err := LoadLanguages(c.LanguagesPath)
		if err != nil && c.LanguagesPath != "" {
			slog.Warn("eou: languages.json load failed; using defaults",
				"path", c.LanguagesPath, "err", err)
		}
		c.Languages = tbl
	}
	return c
}

func Load(cfg Config) (Model, Config, error) {
	cfg = cfg.WithDefaults()

	if cfg.Kind == KindAudio || cfg.Kind == KindFusion {
		if cfg.AudioModel == nil {
			path := strings.TrimSpace(cfg.AudioModelPath)
			if path == "" {
				path = strings.TrimSpace(os.Getenv("EOU_AUDIO_MODEL_PATH"))
			}
			if m := loadAudioFromPath(path, cfg.AudioWindowMs, cfg.AudioPadAlignment); m != nil {
				cfg.AudioModel = m
			}
		}
	}

	if cfg.Kind == KindVad {
		slog.Info("eou: kind=vad (no classifier -- silence-only)")
		return NewHeuristic(), cfg, nil
	}

	if cfg.Kind == KindHeuristic {
		slog.Info("eou: kind=heuristic (rule-based partial-transcript classifier)")
		return NewHeuristic(), cfg, nil
	}

	if cfg.Kind == KindAudio {
		if cfg.AudioModel != nil {
			slog.Info("eou: kind=audio (smart-turn classifier active)")
			return cfg.AudioModel, cfg, nil
		}
		slog.Warn("eou: kind=audio but EOU_AUDIO_MODEL_PATH not set or load failed; falling back to heuristic")
		return NewHeuristic(), cfg, nil
	}
	var textModel Model
	switch {
	case cfg.ModelPath == "":
		slog.Info("eou: using heuristic model (no SPEACHES_EOU_MODEL set)")
		textModel = NewHeuristic()
	default:
		if _, err := os.Stat(cfg.ModelPath); err != nil {
			slog.Warn("eou: model file missing; falling back to heuristic",
				"path", cfg.ModelPath, "err", err)
			textModel = NewHeuristic()
			break
		}
		m, err := NewONNXModel(cfg.ModelPath, ONNXOptions{
			MaxContextTokens: cfg.MaxContextTokens,
			TokenizerPath:    cfg.TokenizerPath,
		})
		if err != nil {
			slog.Warn("eou: ONNX model load failed; falling back to heuristic",
				"path", cfg.ModelPath, "err", err)
			textModel = NewHeuristic()
			break
		}
		slog.Info("eou: ONNX model loaded",
			"path", cfg.ModelPath,
			"tokenizer", cfg.TokenizerPath,
			"max_ctx", cfg.MaxContextTokens,
		)
		textModel = m
	}

	if cfg.Kind == KindFusion && cfg.TextModel == nil {
		cfg.TextModel = textModel
	}
	return textModel, cfg, nil
}
