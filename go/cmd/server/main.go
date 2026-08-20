package main

import (
	"context"
	"errors"
	"flag"
	"log/slog"
	"net/http"
	"os"
	"os/signal"
	"path/filepath"
	"strings"
	"syscall"
	"time"

	"github.com/go-chi/chi/v5"

	"github.com/eordano/speaches-plus-go/internal/conversation"
	"github.com/eordano/speaches-plus-go/internal/diarization"
	"github.com/eordano/speaches-plus-go/internal/eou"
	"github.com/eordano/speaches-plus-go/internal/inspect"
	"github.com/eordano/speaches-plus-go/internal/oapi"
	"github.com/eordano/speaches-plus-go/internal/pii"
	"github.com/eordano/speaches-plus-go/internal/realtime"
	"github.com/eordano/speaches-plus-go/internal/stt"
	"github.com/eordano/speaches-plus-go/internal/tts"
)

func firstNonEmpty(vs ...string) string {
	for _, v := range vs {
		if v != "" {
			return v
		}
	}
	return ""
}

func atoiDefault(v string, def int) int {
	if v == "" {
		return def
	}
	n := 0
	for _, c := range v {
		if c < '0' || c > '9' {
			return def
		}
		n = n*10 + int(c-'0')
	}
	return n
}

func main() {
	addr := flag.String("addr", defaultAddr(), "listen address (env: UVICORN_HOST/UVICORN_PORT)")
	sttBackend := flag.String("stt-backend", os.Getenv("STT_BACKEND"),
		"STT backend: ct2 (default) | whisper_cpp (env: STT_BACKEND)")
	ct2ModelDir := flag.String("ct2-model", os.Getenv("CT2_MODEL"),
		"path to a CTranslate2 Whisper model dir (env: CT2_MODEL). "+
			"Auto-discovers deepdml/faster-whisper-large-v3-turbo-ct2 from $HF_HUB_CACHE if unset.")
	ct2Device := flag.String("ct2-device", os.Getenv("CT2_DEVICE"),
		"CT2 device: cpu|cuda (env: CT2_DEVICE)")
	ct2Compute := flag.String("ct2-compute", os.Getenv("CT2_COMPUTE_TYPE"),
		"CT2 compute type: default|float32|float16|int8|int8_float16 (env: CT2_COMPUTE_TYPE)")
	whisperModel := flag.String("whisper-model", os.Getenv("WHISPER_MODEL"),
		"path to a ggml whisper model file (env: WHISPER_MODEL). Used when stt-backend=whisper_cpp.")
	kokoroModel := flag.String("kokoro-model", os.Getenv("KOKORO_MODEL"),
		"path to Kokoro ONNX model (env: KOKORO_MODEL)")
	kokoroVoices := flag.String("kokoro-voices", os.Getenv("KOKORO_VOICES"),
		"path to Kokoro voices.bin (env: KOKORO_VOICES)")
	espeakData := flag.String("espeak-data", os.Getenv("ESPEAK_DATA_PATH"),
		"path to espeak-ng-data dir (env: ESPEAK_DATA_PATH)")
	siloreVadModel := flag.String("silero-vad", os.Getenv("SILERO_VAD_MODEL"),
		"path to Silero VAD onnx (env: SILERO_VAD_MODEL)")
	chatBase := flag.String("chat-base", os.Getenv("CHAT_COMPLETION_BASE_URL"),
		"upstream chat completions base URL (env: CHAT_COMPLETION_BASE_URL)")
	chatKey := flag.String("chat-key", os.Getenv("CHAT_COMPLETION_API_KEY"),
		"upstream chat completions API key (env: CHAT_COMPLETION_API_KEY)")
	eouModel := flag.String("eou-model", os.Getenv("SPEACHES_EOU_MODEL"),
		"path to EOU ONNX model (env: SPEACHES_EOU_MODEL); empty -> heuristic")
	eouTokenizer := flag.String("eou-tokenizer", os.Getenv("SPEACHES_EOU_TOKENIZER"),
		"path to tokenizer.json (env: SPEACHES_EOU_TOKENIZER); defaults to model_dir/tokenizer.json")
	eouLanguages := flag.String("eou-languages", os.Getenv("SPEACHES_EOU_LANGUAGES"),
		"path to languages.json with per-lang thresholds (env: SPEACHES_EOU_LANGUAGES)")
	eouMin := flag.Int("eou-min-delay-ms", atoiDefault(os.Getenv("SPEACHES_EOU_MIN_DELAY_MS"), 500),
		"min commit delay (ms) at score~1 (env: SPEACHES_EOU_MIN_DELAY_MS, default 500 per RFC §6.12)")
	eouMax := flag.Int("eou-max-delay-ms", atoiDefault(os.Getenv("SPEACHES_EOU_MAX_DELAY_MS"), 3000),
		"max commit delay (ms) at score<threshold (env: SPEACHES_EOU_MAX_DELAY_MS, default 3000 per RFC §6.12)")
	eouHardCap := flag.Int("eou-hard-cap-ms", atoiDefault(os.Getenv("SPEACHES_EOU_HARD_CAP_MS"), 5000),
		"absolute cap on commit delay regardless of EOU verdict (RFC §6.3, default 5000)")
	eouTimeout := flag.Int("eou-timeout-ms", atoiDefault(os.Getenv("SPEACHES_EOU_TIMEOUT_MS"), 100),
		"per-evaluation EOU inference timeout (RFC §6.4.1, default 100)")
	eouKind := flag.String("eou-kind", os.Getenv("SPEACHES_EOU_KIND"),
		"EOU model kind: vad|heuristic|text|audio|integrated (v3 §6.2, default vad)")
	sessionMaxDur := flag.Int("session-max-duration-s", atoiDefault(os.Getenv("SPEACHES_SESSION_MAX_DURATION_S"), 1800),
		"hard session timeout in seconds (RFC §11.3, default 1800)")

	minSpeechMs := flag.Int("min-speech-ms", atoiDefault(os.Getenv("SPEACHES_MIN_SPEECH_MS"), 100),
		"minimum buffered speech (ms) for commit acceptance (RFC §7.1, default 100)")
	minSpeechResponseMs := flag.Int("min-speech-for-response-ms", atoiDefault(os.Getenv("SPEACHES_MIN_SPEECH_FOR_RESPONSE_MS"), 600),
		"below this duration the buffer commits but no auto-response fires (v3 §6.5/§17.4, default 600)")
	bargeInDelayMs := flag.Int("barge-in-delay-ms", atoiDefault(os.Getenv("SPEACHES_BARGE_IN_DELAY_MS"), 0),
		"deferred barge-in commitment (RFC §9.2, default 0)")
	outboundQueueCap := flag.Int("outbound-queue-cap", atoiDefault(os.Getenv("SPEACHES_OUTBOUND_QUEUE_CAP"), 256),
		"non-audio outbound event queue cap (RFC §7.4, default 256)")
	dataChannelFragmentMax := flag.Int("data-channel-fragment-max", atoiDefault(os.Getenv("SPEACHES_DATA_CHANNEL_FRAGMENT_MAX"), 900),
		"data-channel fragment payload size in bytes (RFC §10.4, default 900)")
	drainCapFloorMs := flag.Int("drain-cap-floor-ms", atoiDefault(os.Getenv("SPEACHES_DRAIN_CAP_FLOOR_MS"), 5000),
		"floor on drain_cap clamp (RFC §8.3, default 5000)")
	drainCapCeilingMs := flag.Int("drain-cap-ceiling-ms", atoiDefault(os.Getenv("SPEACHES_DRAIN_CAP_CEILING_MS"), 60000),
		"ceiling on drain_cap clamp (RFC §8.3, default 60000)")
	partialTickMs := flag.Int("partial-tick-ms", atoiDefault(os.Getenv("SPEACHES_PARTIAL_TICK_MS"), 500),
		"partial transcription cadence (RFC §5, default 500)")
	llmTimeoutSec := flag.Int("llm-timeout-s", atoiDefault(os.Getenv("SPEACHES_LLM_TIMEOUT_S"), 60),
		"LLM stream context timeout in seconds (operational, default 60)")
	vadSilenceMs := flag.Int("vad-silence-duration-ms", atoiDefault(os.Getenv("SPEACHES_VAD_SILENCE_DURATION_MS"), 350),
		"trailing silence required to declare end-of-speech (RFC §4.2, default 350)")
	vadPrefixMs := flag.Int("vad-prefix-padding-ms", atoiDefault(os.Getenv("SPEACHES_VAD_PREFIX_PADDING_MS"), 300),
		"audio captured before speech_start is confirmed (RFC §4.2, default 300)")
	vadlessSilenceMs := flag.Int("vad-less-silence-ms", atoiDefault(os.Getenv("SPEACHES_VAD_LESS_SILENCE_MS"), 1500),
		"silence threshold for the no-Silero fallback path (default 1500)")
	startSpeechSamples := flag.Int("start-speech-samples", atoiDefault(os.Getenv("SPEACHES_START_SPEECH_SAMPLES"), 800),
		"minimum buffered samples before commit is allowed (50 ms @ 16 kHz, default 800)")
	eouContextTurns := flag.Int("eou-context-turns", atoiDefault(os.Getenv("SPEACHES_EOU_CONTEXT_TURNS"), 6),
		"prior turns fed to text-EOU (RFC §6.4, default 6)")
	sealedBufferRetention := flag.Int("sealed-buffer-retention-count", atoiDefault(os.Getenv("SPEACHES_SEALED_BUFFER_RETENTION_COUNT"), 4),
		"FIFO cap on sealed-buffer map (RFC §3.4, default 4)")
	predictedTokenBufCap := flag.Int("predicted-token-buffer-cap", atoiDefault(os.Getenv("SPEACHES_PREDICTED_TOKEN_BUFFER_CAP"), 256),
		"max LLM tokens buffered while Predicted (RFC §6.5.3, default 256)")
	eouAudioWindowMs := flag.Int("eou-audio-window-ms", atoiDefault(os.Getenv("SPEACHES_EOU_AUDIO_WINDOW_MS"), 8000),
		"audio window fed to kind=audio EOU model (RFC §6.4, default 8000)")
	eouAudioModel := flag.String("eou-audio-model", firstNonEmpty(os.Getenv("SPEACHES_EOU_AUDIO_MODEL"), os.Getenv("EOU_AUDIO_MODEL_PATH")),
		"path to smart-turn-v3 ONNX model for kind=audio/fusion EOU (RFC §6.2.2)")
	vadModel := flag.String("vad-model", os.Getenv("SPEACHES_VAD_MODEL"),
		"VAD model id: silero_v5 (default) | silero_v6 (RFC §4.1)")
	diarSegModel := flag.String("diar-segmentation", os.Getenv("DIAR_SEGMENTATION_MODEL"),
		"path to DiariZen segmentation ONNX (env: DIAR_SEGMENTATION_MODEL); empty -> diarization disabled")
	diarEmbModel := flag.String("diar-embedding", os.Getenv("DIAR_EMBEDDING_MODEL"),
		"path to WeSpeaker embedding ONNX (env: DIAR_EMBEDDING_MODEL); empty -> diarization disabled")
	inspectSessionDir := flag.String("inspect-session-dir", os.Getenv("SPEACHES_INSPECT_SESSION_DIR"),
		"directory for per-session inspector ndjson + audio (env: SPEACHES_INSPECT_SESSION_DIR; default ~/.cache/speaches/sessions)")
	inspectMaxSessions := flag.Int("inspect-max-sessions", atoiDefault(os.Getenv("SPEACHES_INSPECT_MAX_SESSIONS"), 200),
		"keep at most N most-recent inspector sessions on disk")
	inspectMaxBytes := flag.Int("inspect-max-bytes", atoiDefault(os.Getenv("SPEACHES_INSPECT_MAX_BYTES"), 2*1024*1024*1024),
		"cap total bytes of inspector artifacts on disk")
	inspectMaxAgeDays := flag.Int("inspect-max-age-days", atoiDefault(os.Getenv("SPEACHES_INSPECT_MAX_AGE_DAYS"), 14),
		"delete inspector artifacts older than N days")

	piiModel := flag.String("pii-model", os.Getenv("REDACT_MODEL_ID"),
		"path to PII classifier model dir (env: REDACT_MODEL_ID); empty -> PII endpoints disabled")
	piiDevice := flag.String("pii-device", os.Getenv("REDACT_DEVICE"),
		"PII model device: cpu|cuda (env: REDACT_DEVICE)")

	logLevel := flag.String("log-level", logLevelDefault(), "debug|info|warn|error (env: LOG_LEVEL)")
	flag.Parse()

	logger := newLogger(*logLevel)
	slog.SetDefault(logger)

	resolveSileroDefault(siloreVadModel)

	backend := strings.ToLower(*sttBackend)
	if backend == "" {
		backend = "ct2"
	}

	var (
		transcriber stt.Transcriber
		err         error
	)
	switch backend {
	case "ct2":
		resolveCT2Default(ct2ModelDir)
		transcriber, err = stt.NewCT2(stt.CT2Config{
			ModelDir:    *ct2ModelDir,
			Device:      *ct2Device,
			ComputeType: *ct2Compute,
			Language:    "en",
		})
		if err != nil {
			slog.Error("ct2 init failed; cannot start", "err", err, "model_dir", *ct2ModelDir)
			os.Exit(1)
		}
	case "whisper_cpp", "whisper.cpp", "whispercpp":
		resolveWhisperDefault(whisperModel)
		transcriber, err = stt.NewWhisper(*whisperModel)
		if err != nil {
			slog.Error("whisper.cpp init failed; cannot start", "err", err, "model", *whisperModel)
			os.Exit(1)
		}
	default:
		slog.Error("unknown STT backend", "value", backend, "allowed", "ct2 | whisper_cpp")
		os.Exit(1)
	}
	slog.Info("STT backend selected", "backend", backend)
	defer transcriber.Close()

	var synth tts.Synthesizer
	if *kokoroModel != "" || *kokoroVoices != "" {
		if *kokoroModel == "" || *kokoroVoices == "" {
			slog.Error("kokoro: --kokoro-model and --kokoro-voices must both be set or both empty")
			os.Exit(1)
		}
		synth, err = tts.NewKokoro(tts.KokoroConfig{
			ModelPath:  *kokoroModel,
			VoicesPath: *kokoroVoices,
			EspeakData: *espeakData,
		})
		if err != nil {
			slog.Error("kokoro init failed; cannot start", "err", err, "model", *kokoroModel)
			os.Exit(1)
		}
		defer synth.Close()
	}

	llmClient := conversation.NewLLM(*chatBase, *chatKey)

	eouM, eouCfg, eouErr := eou.Load(eou.Config{
		Kind:               eou.Kind(strings.ToLower(*eouKind)),
		ModelPath:          *eouModel,
		TokenizerPath:      *eouTokenizer,
		LanguagesPath:      *eouLanguages,
		AudioModelPath:     *eouAudioModel,
		MinDelayMs:         *eouMin,
		MaxDelayMs:         *eouMax,
		HardCapMs:          *eouHardCap,
		InferenceTimeoutMs: *eouTimeout,
		AudioWindowMs:      *eouAudioWindowMs,
	})
	if eouErr != nil {
		slog.Warn("eou: load reported error; continuing with fallback", "err", eouErr)
	}
	defer eouM.Close()
	if eouCfg.AudioModel != nil && eouCfg.AudioModel != eouM {
		defer eouCfg.AudioModel.Close()
	}

	var integratedSource eou.IntegratedSource
	if eouCfg.Kind == eou.KindIntegrated {
		integratedSource = eou.NewFakeIntegrated(eou.FakeIntegratedScript{})
		slog.Info("integrated EOU mode: stub source wired (no real integrated STT yet)")
	}

	var diarSeg *diarization.SegmentationModel
	var diarEmb *diarization.EmbeddingModel
	if *diarEmbModel != "" {
		emb, err := diarization.LoadEmbedding(*diarEmbModel)
		if err != nil {
			slog.Warn("diarization: embedding load failed; disabled", "err", err, "path", *diarEmbModel)
		} else {
			diarEmb = emb
			defer diarEmb.Close()
		}
	}
	switch {
	case *diarSegModel == "":
	case diarEmb == nil:
		slog.Warn("diarization: --diar-segmentation set but embedding model unavailable; segmentation disabled")
	default:
		seg, err := diarization.LoadSegmentation(*diarSegModel)
		if err != nil {
			slog.Warn("diarization: segmentation load failed; disabled", "err", err, "path", *diarSegModel)
		} else {
			diarSeg = seg
			defer diarSeg.Close()
		}
	}
	switch {
	case diarSeg != nil:
		slog.Info("diarization enabled", "segmentation", *diarSegModel, "embedding", *diarEmbModel)
	case diarEmb != nil:
		slog.Info("audio embeddings enabled", "embedding", *diarEmbModel)
	}

	rt := realtime.NewServer(realtime.Config{
		STT:                    transcriber,
		TTS:                    synth,
		LLM:                    llmClient,
		SileroVADPath:          *siloreVadModel,
		EOUModel:               eouM,
		EOUConfig:              eouCfg,
		IntegratedSource:       integratedSource,
		DiarSegmentation:       diarSeg,
		DiarEmbedding:          diarEmb,
		DiarConfig:             diarization.DefaultConfig(),
		SessionMaxDurSec:       *sessionMaxDur,
		HardCapMs:              *eouHardCap,
		MinSpeechMs:            *minSpeechMs,
		MinSpeechForResponseMs: *minSpeechResponseMs,
		BargeInDelayMs:         *bargeInDelayMs,
		OutboundQueueCap:       *outboundQueueCap,
		DataChannelFragmentMax: *dataChannelFragmentMax,
		DrainCapFloorMs:        *drainCapFloorMs,
		DrainCapCeilingMs:      *drainCapCeilingMs,
		PartialTickMs:          *partialTickMs,
		LLMTimeoutSec:          *llmTimeoutSec,
		VADSilenceDurationMs:   *vadSilenceMs,
		VADPrefixPaddingMs:     *vadPrefixMs,
		VADLessSilenceMs:       *vadlessSilenceMs,
		StartSpeechSamples:     *startSpeechSamples,
		EOUContextTurns:        *eouContextTurns,
		EOUMinDelayMs:          *eouMin,
		EOUMaxDelayMs:          *eouMax,

		SealedBufferRetentionCount: *sealedBufferRetention,
		PredictedTokenBufferCap:    *predictedTokenBufCap,
		EOUAudioWindowMs:           *eouAudioWindowMs,
		VADModel:                   *vadModel,
		InspectSessionDir:          inspectSessionDirResolved(*inspectSessionDir),
	})

	inspect.CleanupOnStartup(
		inspectSessionDirResolved(*inspectSessionDir),
		*inspectMaxSessions,
		int64(*inspectMaxBytes),
		*inspectMaxAgeDays,
	)
	transcriptionsHandler := stt.NewTranscriptionsHandler(transcriber)
	if diarSeg != nil && diarEmb != nil {
		transcriptionsHandler = transcriptionsHandler.WithDiarization(diarSeg, diarEmb, diarization.DefaultConfig())
	}
	speechHandler := tts.NewSpeechHandler(synth)

	var piiClassifier *pii.Classifier
	if *piiModel != "" {
		pc, err := pii.NewClassifier(*piiModel, *piiDevice)
		if err != nil {
			slog.Warn("pii: classifier load failed; endpoints disabled", "err", err, "model", *piiModel)
		} else {
			piiClassifier = pc
			defer piiClassifier.Close()
			slog.Info("pii: classifier loaded", "model", *piiModel, "device", *piiDevice)
		}
	}

	listing := oapi.ModelsListing{
		SileroLoaded: *siloreVadModel != "",
	}
	if *whisperModel != "" {
		listing.WhisperModelID = *whisperModel
	}
	if synth != nil {
		if vl, ok := synth.(interface{ Voices() []string }); ok {
			listing.KokoroVoices = vl.Voices()
		}
	}
	if *piiModel != "" {
		listing.PiiModelID = *piiModel
	}
	modelsHandler := oapi.NewModelsHandler(listing)

	r := chi.NewRouter()
	r.Get("/health", func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusOK)
	})
	r.Get("/v1/models", modelsHandler)
	r.Post("/v1/realtime", rt.HandleRealtime)
	r.Get("/v1/realtime", rt.HandleRealtimeWS)
	r.Get("/v1/realtime/capabilities", rt.HandleCapabilities)
	r.Get("/v1/inspect/sessions", rt.HandleInspectListSessions)
	r.Get("/v1/inspect/sessions/history", rt.HandleInspectListHistory)
	r.Get("/v1/inspect/sessions/history/{sid}", rt.HandleInspectGetHistory)
	r.Get("/v1/inspect/sessions/{sid}/audio", rt.HandleInspectGetAudio)
	r.Get("/v1/inspect/{sid}/stream", rt.HandleInspectStream)
	r.Get("/v1/inspect/{sid}", rt.HandleInspectStream)
	r.Post("/v1/audio/transcriptions", transcriptionsHandler.ServeHTTP)
	r.Post("/v1/audio/speech", speechHandler.ServeHTTP)
	if diarSeg != nil && diarEmb != nil {
		diarHandler := diarization.NewHandler(diarSeg, diarEmb, diarization.DefaultConfig())
		r.Post("/v1/audio/diarization", diarHandler.ServeHTTP)
	} else {
		r.Post("/v1/audio/diarization", func(w http.ResponseWriter, _ *http.Request) {
			oapi.WriteError(w, http.StatusServiceUnavailable,
				"diarization model not loaded; set --diar-segmentation and --diar-embedding",
				oapi.TypeServiceUnavail, "", "model_not_loaded")
		})
	}
	if diarEmb != nil {
		embHandler := diarization.NewEmbeddingsHandler(diarEmb)
		r.Post("/v1/audio/embeddings", embHandler.ServeHTTP)
	} else {
		r.Post("/v1/audio/embeddings", func(w http.ResponseWriter, _ *http.Request) {
			oapi.WriteError(w, http.StatusServiceUnavailable,
				"embedding model not loaded; set --diar-embedding",
				oapi.TypeServiceUnavail, "", "model_not_loaded")
		})
	}

	if piiClassifier != nil {
		r.Post("/v1/pii/classify", pii.ClassifyHandler(piiClassifier))
		r.Post("/v1/pii/classify/batch", pii.ClassifyBatchHandler(piiClassifier))
		r.Post("/v1/pii/redact/analyze", pii.AnalyzeHandler(piiClassifier))
		r.Post("/v1/pii/redact/render", pii.RenderHandler())
	} else {
		r.Post("/v1/pii/classify", func(w http.ResponseWriter, _ *http.Request) {
			oapi.WriteError(w, http.StatusServiceUnavailable,
				"PII model not loaded; set --pii-model",
				oapi.TypeServiceUnavail, "", "model_not_loaded")
		})
		r.Post("/v1/pii/classify/batch", func(w http.ResponseWriter, _ *http.Request) {
			oapi.WriteError(w, http.StatusServiceUnavailable,
				"PII model not loaded; set --pii-model",
				oapi.TypeServiceUnavail, "", "model_not_loaded")
		})
		r.Post("/v1/pii/redact/analyze", pii.UnavailableHandler("PII model not loaded; set --pii-model"))
		r.Post("/v1/pii/redact/render", pii.RenderHandler())
	}

	httpServer := &http.Server{
		Addr:              *addr,
		Handler:           r,
		ReadHeaderTimeout: 10 * time.Second,
	}

	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()

	go func() {
		slog.Info("listening", "addr", *addr)
		if err := httpServer.ListenAndServe(); err != nil && !errors.Is(err, http.ErrServerClosed) {
			slog.Error("http listener died", "err", err)
			stop()
		}
	}()

	<-ctx.Done()
	slog.Info("shutting down")

	shutdownCtx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	_ = httpServer.Shutdown(shutdownCtx)
}

func inspectSessionDirResolved(v string) string {
	if v != "" {
		return os.ExpandEnv(v)
	}
	return inspect.DefaultSessionDir()
}

func defaultAddr() string {
	host := os.Getenv("UVICORN_HOST")
	port := os.Getenv("UVICORN_PORT")
	if port == "" {
		return ":8000"
	}
	if host == "" {
		host = "127.0.0.1"
	}
	return host + ":" + port
}

func resolveCT2Default(modelDir *string) {
	if *modelDir != "" {
		return
	}
	hf := os.Getenv("HF_HUB_CACHE")
	if hf == "" {
		hf = os.ExpandEnv("$HOME/.cache/huggingface/hub")
	}
	for _, repo := range []string{
		"models--deepdml--faster-whisper-large-v3-turbo-ct2",
	} {
		matches, _ := filepath.Glob(filepath.Join(hf, repo, "snapshots/*/model.bin"))
		if len(matches) > 0 {
			*modelDir = filepath.Dir(matches[0])
			slog.Info("auto-discovered CT2 model", "path", *modelDir)
			return
		}
	}
}

func resolveSileroDefault(silero *string) {
	if *silero != "" {
		return
	}
	exe, _ := os.Executable()
	exeDir := filepath.Dir(exe)
	for _, p := range []string{
		"./models/silero_vad.onnx",
		"./models/silero-vad-v6/silero_vad.onnx",
		filepath.Join(exeDir, "models", "silero_vad.onnx"),
		filepath.Join(exeDir, "..", "models", "silero_vad.onnx"),
		filepath.Join(exeDir, "..", "..", "models", "silero_vad.onnx"),
	} {
		abs, err := filepath.Abs(p)
		if err != nil {
			continue
		}
		if _, err := os.Stat(abs); err == nil {
			*silero = abs
			slog.Info("auto-discovered Silero VAD model", "path", abs)
			return
		}
	}
}

func resolveWhisperDefault(whisper *string) {
	if *whisper != "" {
		return
	}
	exe, _ := os.Executable()
	exeDir := filepath.Dir(exe)
	for _, p := range []string{
		"./models/ggml-tiny.en.bin",
		"./models/ggml-small.en.bin",
		"./models/ggml-base.en.bin",
		filepath.Join(exeDir, "..", "models", "ggml-tiny.en.bin"),
		filepath.Join(exeDir, "..", "..", "models", "ggml-tiny.en.bin"),
	} {
		abs, err := filepath.Abs(p)
		if err != nil {
			continue
		}
		if _, err := os.Stat(abs); err == nil {
			*whisper = abs
			slog.Info("auto-discovered whisper.cpp model", "path", abs)
			return
		}
	}
}

func logLevelDefault() string {
	if v := os.Getenv("LOG_LEVEL"); v != "" {
		return strings.ToLower(v)
	}
	return "info"
}

func newLogger(level string) *slog.Logger {
	var l slog.Level
	switch level {
	case "debug":
		l = slog.LevelDebug
	case "warn":
		l = slog.LevelWarn
	case "error":
		l = slog.LevelError
	default:
		l = slog.LevelInfo
	}
	return slog.New(slog.NewTextHandler(os.Stderr, &slog.HandlerOptions{Level: l}))
}
