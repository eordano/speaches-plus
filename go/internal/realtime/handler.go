package realtime

import (
	"io"
	"log/slog"
	"net/http"
	"strings"

	"github.com/eordano/speaches-plus-go/internal/conversation"
	"github.com/eordano/speaches-plus-go/internal/diarization"
	"github.com/eordano/speaches-plus-go/internal/eou"
	"github.com/eordano/speaches-plus-go/internal/oapi"
	"github.com/eordano/speaches-plus-go/internal/stt"
	"github.com/eordano/speaches-plus-go/internal/tts"
)

type Config struct {
	STT              stt.Transcriber
	TTS              tts.Synthesizer
	LLM              *conversation.LLM
	SileroVADPath    string
	EOUModel         eou.Model
	EOUConfig        eou.Config
	IntegratedSource eou.IntegratedSource

	DiarSegmentation *diarization.SegmentationModel
	DiarEmbedding    *diarization.EmbeddingModel
	DiarConfig       diarization.Config

	SessionMaxDurSec       int
	HardCapMs              int
	MinSpeechMs            int
	MinSpeechForResponseMs int
	BargeInDelayMs         int
	OutboundQueueCap       int
	OutboundBufferLimit    uint64
	DataChannelFragmentMax int
	DrainCapFloorMs        int
	DrainCapCeilingMs      int
	PartialTickMs          int
	LLMTimeoutSec          int
	VADThreshold           float32
	VADSilenceDurationMs   int
	VADPrefixPaddingMs     int
	StartSpeechSamples     int
	VADLessSilenceMs       int
	NonSilenceThreshold    float32
	InspectorTransitions   bool
	InspectorSampleRate    float32
	EOUContextTurns        int
	EOUMinDelayMs          int
	EOUMaxDelayMs          int

	SealedBufferRetentionCount int
	PredictedTokenBufferCap    int
	EOUAudioWindowMs           int
	VADModel                   string

	TurnDetectionType string

	EagerMaxInflight     int
	EagerPeriodicEnabled bool
	EagerIntervalMs      int

	InspectSessionDir string
	InspectorRelayCap int
}

type Server struct {
	cfg Config
}

func NewServer(cfg Config) *Server {
	return &Server{cfg: cfg}
}

func (s *Server) makeSessionConfig(model, intent, transcriptionModel, voice, speechModel, language string) sessionConfig {
	return sessionConfig{
		Model:                  model,
		Intent:                 intent,
		TranscriptionModel:     transcriptionModel,
		Voice:                  voice,
		SpeechModel:            speechModel,
		Language:               language,
		SessionMaxDurSec:       s.cfg.SessionMaxDurSec,
		HardCapMs:              s.cfg.HardCapMs,
		MinSpeechMs:            s.cfg.MinSpeechMs,
		MinSpeechForResponseMs: s.cfg.MinSpeechForResponseMs,
		BargeInDelayMs:         s.cfg.BargeInDelayMs,
		OutboundQueueCap:       s.cfg.OutboundQueueCap,
		OutboundBufferLimit:    s.cfg.OutboundBufferLimit,
		DataChannelFragmentMax: s.cfg.DataChannelFragmentMax,
		DrainCapFloorMs:        s.cfg.DrainCapFloorMs,
		DrainCapCeilingMs:      s.cfg.DrainCapCeilingMs,
		PartialTickMs:          s.cfg.PartialTickMs,
		LLMTimeoutSec:          s.cfg.LLMTimeoutSec,
		VADThreshold:           s.cfg.VADThreshold,
		VADSilenceDurationMs:   s.cfg.VADSilenceDurationMs,
		VADPrefixPaddingMs:     s.cfg.VADPrefixPaddingMs,
		StartSpeechSamples:     s.cfg.StartSpeechSamples,
		VADLessSilenceMs:       s.cfg.VADLessSilenceMs,
		NonSilenceThreshold:    s.cfg.NonSilenceThreshold,
		InspectorTransitions:   s.cfg.InspectorTransitions,
		InspectorSampleRate:    s.cfg.InspectorSampleRate,
		EOUContextTurns:        s.cfg.EOUContextTurns,
		EOUMinDelayMs:          s.cfg.EOUMinDelayMs,
		EOUMaxDelayMs:          s.cfg.EOUMaxDelayMs,

		SealedBufferRetentionCount: s.cfg.SealedBufferRetentionCount,
		PredictedTokenBufferCap:    s.cfg.PredictedTokenBufferCap,
		EOUAudioWindowMs:           s.cfg.EOUAudioWindowMs,
		VADModel:                   s.cfg.VADModel,

		TurnDetectionType:    s.cfg.TurnDetectionType,
		EagerMaxInflight:     s.cfg.EagerMaxInflight,
		EagerPeriodicEnabled: s.cfg.EagerPeriodicEnabled,
		EagerIntervalMs:      s.cfg.EagerIntervalMs,
	}
}

func (s *Server) HandleRealtime(w http.ResponseWriter, r *http.Request) {
	model := r.URL.Query().Get("model")
	if model == "" {
		oapi.WriteError(w, http.StatusBadRequest,
			"missing ?model", oapi.TypeInvalidRequest, "model", "missing")
		return
	}
	intent := r.URL.Query().Get("intent")
	if intent == "" {
		intent = "conversation"
	}
	transcriptionModel := r.URL.Query().Get("transcription_model")
	voice := r.URL.Query().Get("voice")
	speechModel := r.URL.Query().Get("speech_model")
	language := r.URL.Query().Get("language")

	body, err := io.ReadAll(r.Body)
	if err != nil {
		oapi.WriteError(w, http.StatusBadRequest,
			"read body: "+err.Error(), oapi.TypeInvalidRequest, "", "body_read_error")
		return
	}

	answerSDP, err := s.negotiate(r.Context(), string(body), s.makeSessionConfig(model, intent, transcriptionModel, voice, speechModel, language))
	if err != nil {
		slog.Error("negotiate failed", "err", err)
		status := http.StatusInternalServerError
		kind := oapi.TypeServerError
		code := "negotiate_failed"
		msg := err.Error()
		for _, needle := range []string{
			"failed to unmarshal SDP",
			"syntax error",
			"no ice-ufrag",
			"no ice-pwd",
			"no fingerprint",
			"unable to start media",
		} {
			if strings.Contains(msg, needle) {
				status = http.StatusBadRequest
				kind = oapi.TypeInvalidRequest
				code = "sdp_invalid"
				break
			}
		}
		oapi.WriteError(w, status, msg, kind, "", code)
		return
	}

	w.Header().Set("Content-Type", "text/plain; charset=utf-8")
	_, _ = w.Write([]byte(answerSDP))
}
