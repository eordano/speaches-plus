package oapi

import "encoding/json"

type Model struct {
	ID        string
	Object    string
	Created   int64
	OwnedBy   string
	Languages []string
	Task      string
	Extras    map[string]any
}

func (m Model) MarshalJSON() ([]byte, error) {
	out := map[string]any{
		"id":       m.ID,
		"object":   "model",
		"created":  m.Created,
		"owned_by": m.OwnedBy,
		"task":     m.Task,
		"language": m.Languages,
	}
	for k, v := range m.Extras {
		out[k] = v
	}
	return json.Marshal(out)
}

type ListModelsResponse struct {
	Object string  `json:"object"`
	Data   []Model `json:"data"`
}

const (
	TaskASR                 = "automatic-speech-recognition"
	TaskTTS                 = "text-to-speech"
	TaskVAD                 = "voice-activity-detection"
	TaskTokenClassification = "token-classification"
	TaskEmbedding           = "embedding"
)

var WhisperLanguages = []string{
	"en", "zh", "de", "es", "ru", "ko", "fr", "ja", "pt", "tr", "pl", "ca",
	"nl", "ar", "sv", "it", "id", "hi", "fi", "vi", "he", "uk", "el", "ms",
	"cs", "ro", "da", "hu", "ta", "no", "th", "ur", "hr", "bg", "lt", "la",
	"mi", "ml", "cy", "sk", "te", "fa", "lv", "bn", "sr", "az", "sl", "kn",
	"et", "mk", "br", "eu", "is", "hy", "ne", "mn", "bs", "kk", "sq", "sw",
	"gl", "mr", "pa", "si", "km", "sn", "yo", "so", "af", "oc", "ka", "be",
	"tg", "sd", "gu", "am", "yi", "lo", "uz", "fo", "ht", "ps", "tk", "nn",
	"mt", "sa", "lb", "my", "bo", "tl", "mg", "as", "tt", "haw", "ln", "ha",
	"ba", "jw", "su", "yue",
}

var KokoroLanguages = []string{"en", "es", "fr", "hi", "it", "ja", "pt", "zh"}
