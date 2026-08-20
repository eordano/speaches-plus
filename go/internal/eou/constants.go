package eou

const (
	DefaultCurveK float32 = 12.0

	defaultMinDelayMs         = 500
	defaultMaxDelayMs         = 3000
	defaultHardCapMs          = 5000
	defaultInferenceTimeoutMs = 250
	defaultContextTurns       = 4
	defaultEagerMaxInflight   = 1
	defaultEagerIntervalMs    = 250
	defaultMinSpeechForCommit = 600
	defaultMaxContextTokens   = 128
	defaultPThreshold         = 0.5
	defaultEotThreshold       = 0.7
	defaultEagerEotThreshold  = 0.5
	defaultAudioWindowMs      = 8000

	defaultFusionRule       = FusionGated
	defaultFusionWeightText = 0.5

	heuristicScoreEmpty            float32 = 0.1
	heuristicScoreStrongTerminator float32 = 0.95
	heuristicScoreSoftTerminator   float32 = 0.25
	heuristicScoreEmptyLastWord    float32 = 0.3
	heuristicScoreHesitation       float32 = 0.15
	heuristicScoreContinuation     float32 = 0.2
	heuristicScoreDefault          float32 = 0.6

	eagernessLowPThreshold    float32 = 0.7
	eagernessLowMinDelayMs            = 800
	eagernessLowMaxDelayMs            = 3000
	eagernessMediumPThreshold float32 = 0.5
	eagernessMediumMinDelayMs         = 500
	eagernessMediumMaxDelayMs         = 2500
	eagernessHighPThreshold   float32 = 0.4
	eagernessHighMinDelayMs           = 300
	eagernessHighMaxDelayMs           = 1500

	onnxInputIDs      = "input_ids"
	onnxAttentionMask = "attention_mask"
	onnxOutputLogits  = "logits"

	ImStart = "<|im_start|>"
	ImEnd   = "<|im_end|>"
)

const (
	KindVad        Kind = "vad"
	KindHeuristic  Kind = "heuristic"
	KindText       Kind = "text"
	KindAudio      Kind = "audio"
	KindFusion     Kind = "fusion"
	KindIntegrated Kind = "integrated"
)

const (
	FusionNoisyOr  FusionRule = "noisy_or"
	FusionMax      FusionRule = "max"
	FusionMean     FusionRule = "mean"
	FusionWeighted FusionRule = "weighted"

	FusionGated FusionRule = "gated"
)

const (
	EagernessUnset  Eagerness = ""
	EagernessLow    Eagerness = "low"
	EagernessMedium Eagerness = "medium"
	EagernessHigh   Eagerness = "high"
	EagernessAuto   Eagerness = "auto"
)
