package realtime

const (
	opusSampleRate    = 48000
	whisperSampleRate = 16000
	maxOpusFrameInt16 = 5760

	rtpOutSampleRate     = 48000
	opusFrameMs          = 20
	opusEncodeScratchCap = 1500

	defaultEOUMinDelayMs      = 500
	defaultEOUMaxDelayMs      = 3000
	defaultHardCapMs          = 5000
	defaultEOUCurveK          = 12.0
	defaultEOUContextTurns    = 4
	defaultEOUFailureP        = 1.0
	defaultEOUFailureDelay    = "min"
	defaultInferenceTimeoutMs = 250
	eouHistoryFallbackTurns   = 6
	defaultEOUAudioWindowMs   = 8000

	defaultMinSpeechMs            = 100
	defaultMinSpeechForResponseMs = 600
	defaultPartialTickMs          = 500
	defaultStartSpeechSamples     = 800
	defaultSealedBufferRetention  = 4

	defaultSessionMaxDurSec  = 1800
	defaultLLMTimeoutSec     = 60
	defaultDrainCapFloorMs   = 5000
	defaultDrainCapCeilingMs = 60000
	drainPollIntervalMs      = 20

	defaultOutboundQueueCap       = 256
	defaultDataChannelFragmentMax = 900
	defaultOutboundBufferLimit    = uint64(1) << 20
	envelopeBudget                = 100
	wsWriteTimeoutSec             = 5

	defaultVADThreshold           = 0.5
	defaultVADSilenceDurationMs   = 350
	defaultVADPrefixPaddingMs     = 300
	defaultVADMinSpeechDurationMs = 100
	defaultVADLessSilenceMs       = 1500
	defaultNonSilenceThreshold    = 0.005
	silenceWatchdogTickMs         = 200
	defaultVADModel               = "silero_v5"
	defaultTurnDetectionType      = TurnDetectionTypeServerVad

	defaultPredictedTokenBufferCap = 256
	defaultEagerMaxInflight        = 1
	defaultEagerIntervalMs         = 250

	defaultEagerTranscriptMismatchRatio float32 = 0.5

	defaultInspectorTransitions = false
	defaultInspectorSampleRate  = 1.0

	vadFailureThreshold = 3

	sentenceChunkerMinChars = 120
	defaultVoice            = "af_heart"

	defaultTTSSpeed float32 = 1.0
	minTTSSpeed     float32 = 0.5
	maxTTSSpeed     float32 = 2.0

	tracerName = "speaches/realtime"

	debugInvariants = true

	capabilityRFCVersion = "v3"

	AudioFormatPCM16     = "pcm16"
	AudioFormatPCM16_16K = "pcm16_16k"
	AudioFormatG711Ulaw  = "g711_ulaw"
	AudioFormatG711Alaw  = "g711_alaw"

	SessionObjectRealtimeSession = "realtime.session"

	ModalityText  = "text"
	ModalityAudio = "audio"

	TurnDetectionTypeServerVad = "server_vad"
	TurnDetectionTypeNone      = "none"

	FailureDelayMin = "min"
	FailureDelayMax = "max"

	defaultInputAudioFormat  = AudioFormatPCM16
	defaultOutputAudioFormat = AudioFormatPCM16

	defaultFusionRule       = "gated"
	defaultFusionWeightText = 0.5

	inspectorBusBufferPerSub     = 256
	inspectorWSWriteTimeout      = 5
	inspectStreamWriteTimeoutSec = 5

	defaultTTSStoreSampleRate = 24000

	defaultInspectorRelayCap = 1024

	maxPrefixPaddingMs            = 1000
	minSilenceDurationMs          = 50
	maxSilenceDurationMs          = 5000
	maxBargeInDelayMs             = 1000
	maxEOUCurveK                  = 30
	maxEOUHardCapMs               = 60000
	maxEOUInferenceTimeoutMs      = 10000
	maxEOUContextTurns            = 64
	maxSessionMaxDurationS        = 86400
	maxMinSpeechMs                = 60000
	maxMinSpeechForResponseMs     = 60000
	maxSealedBufferRetentionCount = 1024
)

const (
	SR16k SampleRate = 16000
	SR24k SampleRate = 24000
	SR48k SampleRate = 48000
)

const (
	CETSessionUpdate            ClientEventType = "session.update"
	CETInputBufferAppend        ClientEventType = "input_audio_buffer.append"
	CETInputBufferCommit        ClientEventType = "input_audio_buffer.commit"
	CETInputBufferClear         ClientEventType = "input_audio_buffer.clear"
	CETConversationItemCreate   ClientEventType = "conversation.item.create"
	CETConversationItemTruncate ClientEventType = "conversation.item.truncate"
	CETConversationItemDelete   ClientEventType = "conversation.item.delete"
	CETResponseCreate           ClientEventType = "response.create"
	CETResponseCancel           ClientEventType = "response.cancel"
	CETConversationItemRetrieve ClientEventType = "conversation.item.retrieve"
)

const (
	SETError                              ServerEventType = "error"
	SETSessionCreated                     ServerEventType = "session.created"
	SETSessionUpdated                     ServerEventType = "session.updated"
	SETSessionDone                        ServerEventType = "session.done"
	SETInputBufferSpeechStarted           ServerEventType = "input_audio_buffer.speech_started"
	SETInputBufferSpeechStopped           ServerEventType = "input_audio_buffer.speech_stopped"
	SETInputBufferCommitted               ServerEventType = "input_audio_buffer.committed"
	SETInputBufferCleared                 ServerEventType = "input_audio_buffer.cleared"
	SETInputBufferPartialTranscription    ServerEventType = "input_audio_buffer.partial_transcription"
	SETInputAudioTranscriptionDelta       ServerEventType = "conversation.item.input_audio_transcription.delta"
	SETInputAudioTranscriptionCompleted   ServerEventType = "conversation.item.input_audio_transcription.completed"
	SETInputAudioTranscriptionFailed      ServerEventType = "conversation.item.input_audio_transcription.failed"
	SETConversationItemAdded              ServerEventType = "conversation.item.added"
	SETConversationItemTruncated          ServerEventType = "conversation.item.truncated"
	SETConversationItemAssistantTruncated ServerEventType = "conversation.item.assistant_truncated"
	SETConversationItemDeleted            ServerEventType = "conversation.item.deleted"
	SETResponseCreated                    ServerEventType = "response.created"
	SETResponseDone                       ServerEventType = "response.done"
	SETResponseOutputItemAdded            ServerEventType = "response.output_item.added"
	SETResponseOutputItemDone             ServerEventType = "response.output_item.done"
	SETResponseContentPartAdded           ServerEventType = "response.content_part.added"
	SETResponseContentPartDone            ServerEventType = "response.content_part.done"
	SETResponseOutputAudioTranscriptDelta ServerEventType = "response.output_audio_transcript.delta"
	SETResponseOutputAudioTranscriptDone  ServerEventType = "response.output_audio_transcript.done"
	SETResponseOutputAudioDelta           ServerEventType = "response.output_audio.delta"
	SETResponseOutputAudioDone            ServerEventType = "response.output_audio.done"
	SETConversationItemDiarization        ServerEventType = "conversation.item.diarization"
	SETConversationItemDone               ServerEventType = "conversation.item.done"
	SETConversationItemRetrieved          ServerEventType = "conversation.item.retrieved"
	SETResponseOutputTextDelta            ServerEventType = "response.output_text.delta"
	SETResponseOutputTextDone             ServerEventType = "response.output_text.done"
	SETResponseFunctionCallArgumentsDelta ServerEventType = "response.function_call_arguments.delta"
	SETResponseFunctionCallArgumentsDone  ServerEventType = "response.function_call_arguments.done"
	SETResponseToolProgress               ServerEventType = "response.tool_progress"
	SETResponseCancelled                  ServerEventType = "response.cancelled"
	SETOutputAudioBufferCleared           ServerEventType = "output_audio_buffer.cleared"
	SETOutputAudioBufferStarted           ServerEventType = "output_audio_buffer.started"
	SETOutputAudioBufferStopped           ServerEventType = "output_audio_buffer.stopped"
	SETRateLimitsUpdated                  ServerEventType = "rate_limits.updated"
)

const (
	vadNone vadDecision = iota
	vadSpeechStart
	vadSpeechEnd
)

const (
	sessKindPending sessionKind = iota
	sessKindActive
	sessKindTerminated
)

const (
	TermClientClosed TerminationReason = iota
	TermMaxDuration
	TermInternalStateError
	TermVadFailed
	TermSttFailed
	TermModelLoadFailed
	TermClientTooSlow
)

const (
	vadKindSilent vadKind = iota
	vadKindSpeaking
	vadKindStopped
)

const (
	bufKindEmpty bufKind = iota
	bufKindVoiced
	bufKindStopped
	bufKindCommitted
)

const (
	respKindNone respPhaseKind = iota
	respKindPredicted
	respKindCreated
	respKindStreaming
	respKindDrain
	respKindFinalized
)

const (
	respStatusCompleted responseStatus = iota
	respStatusCancelled
	respStatusIncomplete
	respStatusFailed
)

const (
	itemInProgress itemStatus = iota
	itemCompleted
	itemIncomplete
)

const (
	violationI1SpeakingWithActiveResponse violation = iota
	violationEmptyResponseID
	violationSessionUpdateBeforeActive
	violationCommittedBufNoItem
	violationConvHasVoiced
	violationI7RotationBeforeCommit
	violationI9PredictedNoRunner
)
