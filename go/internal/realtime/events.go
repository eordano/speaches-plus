package realtime

import (
	"encoding/json"
)

type ClientEventType string

func (t ClientEventType) String() string { return string(t) }

type ServerEventType string

func (t ServerEventType) String() string { return string(t) }

type sessionCreatedEvent struct {
	EventID string          `json:"event_id"`
	Type    ServerEventType `json:"type"`
	Session session         `json:"session"`
}

type sessionUpdatedEvent struct {
	EventID string          `json:"event_id"`
	Type    ServerEventType `json:"type"`
	Session session         `json:"session"`
}

type sessionDoneEvent struct {
	EventID string          `json:"event_id"`
	Type    ServerEventType `json:"type"`
	Reason  string          `json:"reason"`
}

type session struct {
	ID                      string                    `json:"id"`
	Object                  string                    `json:"object"`
	Model                   string                    `json:"model"`
	Modalities              []string                  `json:"modalities"`
	InputAudioFormat        string                    `json:"input_audio_format"`
	OutputAudioFormat       string                    `json:"output_audio_format"`
	Instructions            string                    `json:"instructions,omitempty"`
	Voice                   string                    `json:"voice,omitempty"`
	InputAudioTranscription *audioTranscriptionConfig `json:"input_audio_transcription,omitempty"`
}

func (s session) MarshalJSON() ([]byte, error) {
	type alias session
	flat, err := json.Marshal(alias(s))
	if err != nil {
		return nil, err
	}
	var m map[string]any
	if err := json.Unmarshal(flat, &m); err != nil {
		return nil, err
	}
	return json.Marshal(enrichSessionPayload(m))
}

type audioTranscriptionConfig struct {
	Model string `json:"model"`
}

type inputAudioBufferSpeechStartedEvent struct {
	EventID      string          `json:"event_id"`
	Type         ServerEventType `json:"type"`
	ItemID       string          `json:"item_id"`
	AudioStartMs int64           `json:"audio_start_ms"`
}

type inputAudioBufferSpeechStoppedEvent struct {
	EventID    string          `json:"event_id"`
	Type       ServerEventType `json:"type"`
	ItemID     string          `json:"item_id"`
	AudioEndMs int64           `json:"audio_end_ms"`
}

type inputAudioBufferCommittedEvent struct {
	EventID        string          `json:"event_id"`
	Type           ServerEventType `json:"type"`
	PreviousItemID string          `json:"previous_item_id,omitempty"`
	ItemID         string          `json:"item_id"`
}

type inputAudioBufferClearedEvent struct {
	EventID string          `json:"event_id"`
	Type    ServerEventType `json:"type"`
}

type inputAudioBufferPartialTranscriptionEvent struct {
	EventID    string          `json:"event_id"`
	Type       ServerEventType `json:"type"`
	ItemID     string          `json:"item_id"`
	Transcript string          `json:"transcript"`
}

type diarizationSegment struct {
	Speaker    string  `json:"speaker"`
	Start      float64 `json:"start"`
	End        float64 `json:"end"`
	Confidence float32 `json:"confidence"`
}

type conversationItemDiarizationEvent struct {
	EventID    string               `json:"event_id"`
	Type       ServerEventType      `json:"type"`
	ItemID     string               `json:"item_id"`
	AudioEndMs int64                `json:"audio_end_ms"`
	Segments   []diarizationSegment `json:"segments"`
}

type conversationItemCreatedEvent struct {
	EventID        string                 `json:"event_id"`
	Type           ServerEventType        `json:"type"`
	PreviousItemID string                 `json:"previous_item_id,omitempty"`
	Item           conversationItemDetail `json:"item"`
}

type conversationItemTruncatedEvent struct {
	EventID      string          `json:"event_id"`
	Type         ServerEventType `json:"type"`
	ItemID       string          `json:"item_id"`
	ContentIndex int             `json:"content_index"`
	AudioEndMs   int64           `json:"audio_end_ms"`
}

type conversationItemAssistantTruncatedEvent struct {
	EventID    string          `json:"event_id"`
	Type       ServerEventType `json:"type"`
	ItemID     string          `json:"item_id"`
	AudioEndMs int64           `json:"audio_end_ms"`
	Transcript string          `json:"transcript,omitempty"`
}

type conversationItemDeletedEvent struct {
	EventID string          `json:"event_id"`
	Type    ServerEventType `json:"type"`
	ItemID  string          `json:"item_id"`
}

type conversationItemDetail struct {
	ID      string                `json:"id"`
	Object  string                `json:"object"`
	Type    string                `json:"type"`
	Status  string                `json:"status"`
	Role    string                `json:"role"`
	Content []responseContentPart `json:"content"`
}

type transcriptionCompletedEvent struct {
	EventID    string          `json:"event_id"`
	Type       ServerEventType `json:"type"`
	ItemID     string          `json:"item_id"`
	ContentIdx int             `json:"content_index"`
	Transcript string          `json:"transcript"`
}

type transcriptionFailedEvent struct {
	EventID    string          `json:"event_id"`
	Type       ServerEventType `json:"type"`
	ItemID     string          `json:"item_id"`
	ContentIdx int             `json:"content_index"`
	Error      errorPayload    `json:"error"`
}

type responseCreatedEvent struct {
	EventID  string                  `json:"event_id"`
	Type     ServerEventType         `json:"type"`
	Response responseCreatedResponse `json:"response"`
}

type responseCreatedResponse struct {
	ID     string `json:"id"`
	Object string `json:"object"`
	Status string `json:"status"`
}

type responseOutputItemAddedEvent struct {
	EventID     string             `json:"event_id"`
	Type        ServerEventType    `json:"type"`
	ResponseID  string             `json:"response_id"`
	OutputIndex int                `json:"output_index"`
	Item        responseOutputItem `json:"item"`
}

type responseOutputItemDoneEvent struct {
	EventID     string             `json:"event_id"`
	Type        ServerEventType    `json:"type"`
	ResponseID  string             `json:"response_id"`
	OutputIndex int                `json:"output_index"`
	Item        responseOutputItem `json:"item"`
}

type responseContentPartAddedEvent struct {
	EventID      string              `json:"event_id"`
	Type         ServerEventType     `json:"type"`
	ResponseID   string              `json:"response_id"`
	ItemID       string              `json:"item_id"`
	OutputIndex  int                 `json:"output_index"`
	ContentIndex int                 `json:"content_index"`
	Part         responseContentPart `json:"part"`
}

type responseContentPartDoneEvent struct {
	EventID      string              `json:"event_id"`
	Type         ServerEventType     `json:"type"`
	ResponseID   string              `json:"response_id"`
	ItemID       string              `json:"item_id"`
	OutputIndex  int                 `json:"output_index"`
	ContentIndex int                 `json:"content_index"`
	Part         responseContentPart `json:"part"`
}

type responseAudioTranscriptDeltaEvent struct {
	EventID      string          `json:"event_id"`
	Type         ServerEventType `json:"type"`
	ResponseID   string          `json:"response_id"`
	ItemID       string          `json:"item_id"`
	OutputIndex  int             `json:"output_index"`
	ContentIndex int             `json:"content_index"`
	Delta        string          `json:"delta"`
}

type responseAudioTranscriptDoneEvent struct {
	EventID      string          `json:"event_id"`
	Type         ServerEventType `json:"type"`
	ResponseID   string          `json:"response_id"`
	ItemID       string          `json:"item_id"`
	OutputIndex  int             `json:"output_index"`
	ContentIndex int             `json:"content_index"`
	Transcript   string          `json:"transcript"`
}

type responseAudioDeltaEvent struct {
	EventID      string          `json:"event_id"`
	Type         ServerEventType `json:"type"`
	ResponseID   string          `json:"response_id"`
	ItemID       string          `json:"item_id"`
	OutputIndex  int             `json:"output_index"`
	ContentIndex int             `json:"content_index"`
	Delta        string          `json:"delta"`
}

type responseAudioDoneEvent struct {
	EventID      string          `json:"event_id"`
	Type         ServerEventType `json:"type"`
	ResponseID   string          `json:"response_id"`
	ItemID       string          `json:"item_id"`
	OutputIndex  int             `json:"output_index"`
	ContentIndex int             `json:"content_index"`
}

type responseTextDeltaEvent struct {
	EventID      string          `json:"event_id"`
	Type         ServerEventType `json:"type"`
	ResponseID   string          `json:"response_id"`
	ItemID       string          `json:"item_id"`
	OutputIndex  int             `json:"output_index"`
	ContentIndex int             `json:"content_index"`
	Delta        string          `json:"delta"`
}

type responseTextDoneEvent struct {
	EventID      string          `json:"event_id"`
	Type         ServerEventType `json:"type"`
	ResponseID   string          `json:"response_id"`
	ItemID       string          `json:"item_id"`
	OutputIndex  int             `json:"output_index"`
	ContentIndex int             `json:"content_index"`
	Text         string          `json:"text"`
}

type responseDoneEvent struct {
	EventID  string               `json:"event_id"`
	Type     ServerEventType      `json:"type"`
	Response responseDoneResponse `json:"response"`
}

type responseDoneResponse struct {
	ID            string               `json:"id"`
	Object        string               `json:"object"`
	Status        string               `json:"status"`
	StatusDetails *statusDetails       `json:"status_details,omitempty"`
	AudioEndMs    int64                `json:"audio_end_ms"`
	Output        []responseOutputItem `json:"output,omitempty"`
}

type statusDetails struct {
	Reason string        `json:"reason"`
	Error  *errorPayload `json:"error,omitempty"`
}

type responseOutputItem struct {
	ID      string                `json:"id"`
	Object  string                `json:"object,omitempty"`
	Type    string                `json:"type"`
	Role    string                `json:"role"`
	Status  string                `json:"status"`
	Content []responseContentPart `json:"content"`
}

type responseContentPart struct {
	Type       string `json:"type"`
	Transcript string `json:"transcript,omitempty"`
	Text       string `json:"text,omitempty"`
}

type errorEvent struct {
	EventID string          `json:"event_id"`
	Type    ServerEventType `json:"type"`
	Error   errorPayload    `json:"error"`
}

type errorPayload struct {
	Type    string `json:"type"`
	Code    string `json:"code"`
	Message string `json:"message"`
	Param   string `json:"param,omitempty"`
	EventID string `json:"event_id,omitempty"`
}

type clientEventEnvelope struct {
	Type     string              `json:"type"`
	EventID  string              `json:"event_id,omitempty"`
	Session  *sessionUpdateBody  `json:"session,omitempty"`
	Item     *clientItemBody     `json:"item,omitempty"`
	Audio    string              `json:"audio,omitempty"`
	ItemID   string              `json:"item_id,omitempty"`
	ContentI int                 `json:"content_index,omitempty"`
	AudioEnd int64               `json:"audio_end_ms,omitempty"`
	Response *clientResponseBody `json:"response,omitempty"`
}

type sessionUpdateBody struct {
	Instructions  *string            `json:"instructions,omitempty"`
	Voice         *string            `json:"voice,omitempty"`
	Speed         *float32           `json:"speed,omitempty"`
	TurnDetection *turnDetectionBody `json:"turn_detection,omitempty"`

	SessionMaxDurationS        *int    `json:"session_max_duration_s,omitempty"`
	MinSpeechMs                *int    `json:"min_speech_ms,omitempty"`
	MinSpeechForResponseMs     *int    `json:"min_speech_for_response_ms,omitempty"`
	SealedBufferRetentionCount *int    `json:"sealed_buffer_retention_count,omitempty"`
	InputAudioFormat           *string `json:"input_audio_format,omitempty"`
	OutputAudioFormat          *string `json:"output_audio_format,omitempty"`

	NoSpeechProbThreshold     *float32 `json:"no_speech_prob_threshold,omitempty"`
	NoSpeechProbThresholdNull bool     `json:"-"`
	AvgLogprobThreshold       *float32 `json:"avg_logprob_threshold,omitempty"`
	AvgLogprobThresholdNull   bool     `json:"-"`

	ProcessScoped []string `json:"-"`
}

func (b *sessionUpdateBody) UnmarshalJSON(data []byte) error {

	data = normalizeSessionRaw(data)
	type alias sessionUpdateBody
	var a alias
	if err := json.Unmarshal(data, &a); err != nil {
		return err
	}
	*b = sessionUpdateBody(a)
	var raw map[string]json.RawMessage
	if err := json.Unmarshal(data, &raw); err != nil {
		return err
	}
	for _, name := range []string{
		"vad_model",
		"session_max_duration_hard_cap_s",
		"chat_completion_base_url",
		"chat_completion_api_key",
		"default_realtime_model",
		"default_realtime_stt_model",
		"default_realtime_partial_stt_model",
		"default_speech_model",
		"default_voice",
		"gpu_mem_limit_bytes",
	} {
		if _, present := raw[name]; present {
			b.ProcessScoped = append(b.ProcessScoped, name)
		}
	}

	if msg, present := raw["no_speech_prob_threshold"]; present && string(msg) == "null" {
		b.NoSpeechProbThresholdNull = true
	}
	if msg, present := raw["avg_logprob_threshold"]; present && string(msg) == "null" {
		b.AvgLogprobThresholdNull = true
	}
	return nil
}

type turnDetectionBody struct {
	Type                *string  `json:"type,omitempty"`
	Threshold           *float32 `json:"threshold,omitempty"`
	NegThreshold        *float32 `json:"neg_threshold,omitempty"`
	MinSpeechDurationMs *int     `json:"min_speech_duration_ms,omitempty"`
	PrefixPaddingMs     *int     `json:"prefix_padding_ms,omitempty"`
	SilenceDurationMs   *int     `json:"silence_duration_ms,omitempty"`
	BargeInDelayMs      *int     `json:"barge_in_delay_ms,omitempty"`
	CreateResponse      *bool    `json:"create_response,omitempty"`
	EOU                 *eouBody `json:"eou,omitempty"`
}

type eouBody struct {
	Enabled            *bool    `json:"enabled,omitempty"`
	Kind               *string  `json:"kind,omitempty"`
	PThreshold         *float32 `json:"p_threshold,omitempty"`
	CurveK             *float32 `json:"curve_k,omitempty"`
	MinDelayMs         *int     `json:"min_delay_ms,omitempty"`
	MaxDelayMs         *int     `json:"max_delay_ms,omitempty"`
	ContextTurns       *int     `json:"context_turns,omitempty"`
	FailurePDefault    *float32 `json:"failure_p_default,omitempty"`
	FailureDelay       *string  `json:"failure_delay,omitempty"`
	HardCapMs          *int     `json:"silence_hard_cap_ms,omitempty"`
	InferenceTimeoutMs *int     `json:"inference_timeout_ms,omitempty"`
	FusionRule         *string  `json:"fusion_rule,omitempty"`
	FusionWeightText   *float32 `json:"fusion_weight_text,omitempty"`
}

type clientItemBody struct {
	ID      string                `json:"id,omitempty"`
	Type    string                `json:"type,omitempty"`
	Role    string                `json:"role,omitempty"`
	Content []responseContentPart `json:"content,omitempty"`
}

type clientResponseBody struct {
	Modalities   []string `json:"modalities,omitempty"`
	Instructions string   `json:"instructions,omitempty"`
}

type transcriptionDeltaEvent struct {
	EventID    string          `json:"event_id"`
	Type       ServerEventType `json:"type"`
	ItemID     string          `json:"item_id"`
	ContentIdx int             `json:"content_index"`
	Delta      string          `json:"delta"`
}

type conversationItemDoneEvent struct {
	EventID string                 `json:"event_id"`
	Type    ServerEventType        `json:"type"`
	Item    conversationItemDetail `json:"item"`
}

type conversationItemRetrievedEvent struct {
	EventID string                 `json:"event_id"`
	Type    ServerEventType        `json:"type"`
	Item    conversationItemDetail `json:"item"`
}

type responseOutputTextDeltaEvent struct {
	EventID      string          `json:"event_id"`
	Type         ServerEventType `json:"type"`
	ResponseID   string          `json:"response_id"`
	ItemID       string          `json:"item_id"`
	OutputIndex  int             `json:"output_index"`
	ContentIndex int             `json:"content_index"`
	Delta        string          `json:"delta"`
}

type responseOutputTextDoneEvent struct {
	EventID      string          `json:"event_id"`
	Type         ServerEventType `json:"type"`
	ResponseID   string          `json:"response_id"`
	ItemID       string          `json:"item_id"`
	OutputIndex  int             `json:"output_index"`
	ContentIndex int             `json:"content_index"`
	Text         string          `json:"text"`
}

type responseFunctionCallArgumentsDeltaEvent struct {
	EventID     string          `json:"event_id"`
	Type        ServerEventType `json:"type"`
	ResponseID  string          `json:"response_id"`
	ItemID      string          `json:"item_id"`
	OutputIndex int             `json:"output_index"`
	CallID      string          `json:"call_id"`
	Delta       string          `json:"delta"`
}

type responseFunctionCallArgumentsDoneEvent struct {
	EventID     string          `json:"event_id"`
	Type        ServerEventType `json:"type"`
	ResponseID  string          `json:"response_id"`
	ItemID      string          `json:"item_id"`
	OutputIndex int             `json:"output_index"`
	CallID      string          `json:"call_id"`
	Arguments   string          `json:"arguments"`
}

type responseToolProgressEvent struct {
	EventID     string          `json:"event_id"`
	Type        ServerEventType `json:"type"`
	ResponseID  string          `json:"response_id"`
	ItemID      string          `json:"item_id"`
	OutputIndex int             `json:"output_index"`
	Progress    json.RawMessage `json:"progress"`
}

type responseCancelledEvent struct {
	EventID    string          `json:"event_id"`
	Type       ServerEventType `json:"type"`
	ResponseID string          `json:"response_id"`
}

type outputAudioBufferClearedEvent struct {
	EventID string          `json:"event_id"`
	Type    ServerEventType `json:"type"`
}

type outputAudioBufferStartedEvent struct {
	EventID    string          `json:"event_id"`
	Type       ServerEventType `json:"type"`
	ResponseID string          `json:"response_id"`
}

type outputAudioBufferStoppedEvent struct {
	EventID    string          `json:"event_id"`
	Type       ServerEventType `json:"type"`
	ResponseID string          `json:"response_id"`
}

type rateLimitInfo struct {
	Name         string  `json:"name"`
	Limit        int     `json:"limit"`
	Remaining    int     `json:"remaining"`
	ResetSeconds float64 `json:"reset_seconds"`
}

type rateLimitsUpdatedEvent struct {
	EventID    string          `json:"event_id"`
	Type       ServerEventType `json:"type"`
	RateLimits []rateLimitInfo `json:"rate_limits"`
}
