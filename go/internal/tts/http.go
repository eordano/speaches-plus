package tts

import (
	"encoding/base64"
	"encoding/json"
	"fmt"
	"log/slog"
	"net/http"
	"strings"

	"github.com/eordano/speaches-plus-go/internal/oapi"
)

const (
	DefaultVoice          = "af_heart"
	DefaultLanguage       = "en-us"
	DefaultResponseFormat = FormatMP3
	KokoroSampleRate      = 24000
	MinSampleRate         = 8000
	MaxSampleRate         = 48000
	SpeedMin              = 0.5
	SpeedMax              = 2.0
)

type SpeechHandler struct {
	syn  Synthesizer
	lang string
}

func NewSpeechHandler(s Synthesizer) *SpeechHandler {
	return &SpeechHandler{syn: s, lang: DefaultLanguage}
}

func (h *SpeechHandler) WithLanguage(lang string) *SpeechHandler {
	h.lang = lang
	return h
}

type speechRequest struct {
	Model          string   `json:"model"`
	Input          *string  `json:"input"`
	Voice          *string  `json:"voice"`
	ResponseFormat *string  `json:"response_format"`
	Speed          *float64 `json:"speed"`
	StreamFormat   *string  `json:"stream_format"`
	SampleRate     *int     `json:"sample_rate"`
}

func (h *SpeechHandler) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	if h.syn == nil {
		oapi.WriteError(w, http.StatusServiceUnavailable,
			"TTS not configured", oapi.TypeServiceUnavail, "", "")
		return
	}

	var req speechRequest
	dec := json.NewDecoder(r.Body)
	dec.DisallowUnknownFields()
	if err := dec.Decode(&req); err != nil {
		oapi.WriteValidationError(w, oapi.FastAPIErrorEntry{
			Type: "json_invalid",
			Loc:  []string{"body"},
			Msg:  "Invalid JSON: " + err.Error(),
		})
		return
	}

	var entries []oapi.FastAPIErrorEntry
	if req.Model == "" {
		entries = append(entries, oapi.FastAPIErrorEntry{
			Type: "missing", Loc: []string{"body", "model"}, Msg: "Field required",
		})
	}
	if req.Input == nil {
		entries = append(entries, oapi.FastAPIErrorEntry{
			Type: "missing", Loc: []string{"body", "input"}, Msg: "Field required",
		})
	}
	if req.Voice == nil {
		entries = append(entries, oapi.FastAPIErrorEntry{
			Type: "missing", Loc: []string{"body", "voice"}, Msg: "Field required",
		})
	}

	responseFormat := DefaultResponseFormat
	if req.ResponseFormat != nil {
		if !ValidResponseFormat(*req.ResponseFormat) {
			entries = append(entries, oapi.FastAPIErrorEntry{
				Type: "enum",
				Loc:  []string{"body", "response_format"},
				Msg:  "Input should be 'pcm', 'mp3', 'wav', 'flac', 'opus' or 'aac'",
			})
		} else {
			responseFormat = ResponseFormat(*req.ResponseFormat)
		}
	}

	streamFormat := "audio"
	if req.StreamFormat != nil {
		switch *req.StreamFormat {
		case "audio", "sse":
			streamFormat = *req.StreamFormat
		default:
			entries = append(entries, oapi.FastAPIErrorEntry{
				Type: "enum",
				Loc:  []string{"body", "stream_format"},
				Msg:  "Input should be 'audio' or 'sse'",
			})
		}
	}

	if req.SampleRate != nil {
		if *req.SampleRate < MinSampleRate || *req.SampleRate > MaxSampleRate {
			entries = append(entries, oapi.FastAPIErrorEntry{
				Type:  "less_than_equal",
				Loc:   []string{"body", "sample_rate"},
				Msg:   fmt.Sprintf("Input should be between %d and %d", MinSampleRate, MaxSampleRate),
				Input: *req.SampleRate,
			})
		}
	}

	speed := 1.0
	if req.Speed != nil {
		speed = *req.Speed
	}

	if len(entries) > 0 {
		oapi.WriteValidationError(w, entries...)
		return
	}

	if speed < SpeedMin || speed > SpeedMax {
		oapi.WriteError(w,
			http.StatusBadRequest,
			fmt.Sprintf("speed must be between %.1f and %.1f, got %v", SpeedMin, SpeedMax, speed),
			oapi.TypeInvalidRequest,
			"speed",
			"out_of_range",
		)
		return
	}

	voice := *req.Voice
	if !h.voiceSupported(voice) && isOpenAIVoiceAlias(voice) {
		slog.Warn("openai voice alias falling back to default", "voice", voice, "fallback", DefaultVoice)
		voice = DefaultVoice
	}

	cleaned := StripMarkdownEmphasis(StripEmojis(*req.Input))
	cleaned = NormalizeForTTS(cleaned)
	chunks := SplitIntoChunks(cleaned, MaxChunkChars)
	if len(chunks) == 0 {
		w.Header().Set("Content-Type", string(MimeTypeForFormat(responseFormat)))
		w.WriteHeader(http.StatusOK)
		return
	}

	if streamFormat == "sse" {
		h.streamSSE(w, r, chunks, voice, float32(speed))
		return
	}
	h.streamAudio(w, r, chunks, voice, float32(speed), responseFormat, req.SampleRate)
}

func (h *SpeechHandler) voiceSupported(voice string) bool {
	if voice == "" {
		return false
	}
	type voiceLister interface {
		Voices() []string
	}
	if vl, ok := h.syn.(voiceLister); ok {
		for _, v := range vl.Voices() {
			if v == voice {
				return true
			}
		}
		return false
	}
	return true
}

func isOpenAIVoiceAlias(v string) bool {
	switch strings.ToLower(v) {
	case "alloy", "ash", "ballad", "coral", "echo", "sage", "shimmer", "verse":
		return true
	default:
		return false
	}
}

func (h *SpeechHandler) streamAudio(
	w http.ResponseWriter,
	r *http.Request,
	chunks []string,
	voice string,
	speed float32,
	format ResponseFormat,
	sampleRate *int,
) {
	w.Header().Set("Content-Type", MimeTypeForFormat(format))
	w.WriteHeader(http.StatusOK)

	flusher, _ := w.(http.Flusher)
	target := KokoroSampleRate
	if sampleRate != nil && *sampleRate > 0 {
		target = *sampleRate
	}

	pipe := make(chan []float32, 4)
	errCh := make(chan error, 1)
	go func() {
		defer close(pipe)
		for _, chunk := range chunks {
			if r.Context().Err() != nil {
				return
			}
			a, err := h.syn.Synthesize(chunk, voice, h.lang, speed)
			if err != nil {
				slog.Warn("synthesize chunk failed", "err", err, "chunk_len", len(chunk))
				continue
			}
			pipe <- a.Samples
			if flusher != nil {
				flusher.Flush()
			}
		}
		errCh <- nil
	}()

	if err := EncodeAudio(r.Context(), w, pipe, KokoroSampleRate, target, format); err != nil {
		slog.Error("encode audio failed", "err", err)
	}
	<-errCh
}

func (h *SpeechHandler) streamSSE(
	w http.ResponseWriter,
	r *http.Request,
	chunks []string,
	voice string,
	speed float32,
) {
	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("Connection", "keep-alive")
	w.WriteHeader(http.StatusOK)

	flusher, _ := w.(http.Flusher)

	for _, chunk := range chunks {
		if r.Context().Err() != nil {
			return
		}
		a, err := h.syn.Synthesize(chunk, voice, h.lang, speed)
		if err != nil {
			slog.Warn("synthesize chunk failed", "err", err, "chunk_len", len(chunk))
			continue
		}
		writeSSEDelta(w, a.Samples)
		if flusher != nil {
			flusher.Flush()
		}
	}
	writeSSEDone(w)
	if flusher != nil {
		flusher.Flush()
	}
}

func writeSSEDelta(w http.ResponseWriter, samples []float32) {
	body := map[string]string{
		"type":  "speech.audio.delta",
		"audio": base64.StdEncoding.EncodeToString(F32ToS16LE(samples)),
	}
	encoded, _ := json.Marshal(body)
	_, _ = fmt.Fprintf(w, "data: %s\n\n", encoded)
}

func writeSSEDone(w http.ResponseWriter) {
	body := map[string]any{
		"type": "speech.audio.done",
		"token_usage": map[string]int{
			"input_tokens":  0,
			"output_tokens": 0,
			"total_tokens":  0,
		},
	}
	encoded, _ := json.Marshal(body)
	_, _ = fmt.Fprintf(w, "data: %s\n\n", encoded)
}
