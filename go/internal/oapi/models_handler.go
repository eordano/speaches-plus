package oapi

import (
	"encoding/json"
	"net/http"
	"path/filepath"
	"strings"
)

type ModelsListing struct {
	WhisperModelID   string
	KokoroVoices     []string
	SileroLoaded     bool
	PiiModelID       string
	EmbeddingModelID string
}

func NewModelsHandler(listing ModelsListing) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		filter := r.URL.Query().Get("task")
		out := ListModelsResponse{Object: "list", Data: []Model{}}
		for _, m := range buildModels(listing) {
			if filter != "" && m.Task != filter {
				continue
			}
			out.Data = append(out.Data, m)
		}
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(out)
	}
}

func buildModels(listing ModelsListing) []Model {
	models := make([]Model, 0, 3)
	if listing.WhisperModelID != "" {
		id, owner := whisperIDAndOwner(listing.WhisperModelID)
		models = append(models, Model{
			ID:        id,
			OwnedBy:   owner,
			Created:   1,
			Languages: WhisperLanguages,
			Task:      TaskASR,
		})
	}
	if len(listing.KokoroVoices) > 0 {
		models = append(models, Model{
			ID:        "speaches-ai/Kokoro-82M-v1.0-ONNX",
			OwnedBy:   "speaches-ai",
			Created:   1,
			Languages: KokoroLanguages,
			Task:      TaskTTS,
			Extras: map[string]any{
				"sample_rate": 24000,
				"voices":      listing.KokoroVoices,
			},
		})
	}
	if listing.SileroLoaded {
		models = append(models, Model{
			ID:      "silero_vad_v6",
			OwnedBy: "snakers4",
			Created: 1,
			Task:    TaskVAD,
		})
	}
	if listing.EmbeddingModelID != "" {
		parts := strings.SplitN(listing.EmbeddingModelID, "/", 2)
		owner := "unknown"
		if len(parts) == 2 {
			owner = parts[0]
		}
		models = append(models, Model{
			ID:      listing.EmbeddingModelID,
			OwnedBy: owner,
			Created: 1,
			Task:    TaskEmbedding,
		})
	}
	if listing.PiiModelID != "" {
		parts := strings.SplitN(listing.PiiModelID, "/", 2)
		owner := "openai"
		if len(parts) == 2 {
			owner = parts[0]
		}
		models = append(models, Model{
			ID:      listing.PiiModelID,
			OwnedBy: owner,
			Created: 1,
			Task:    TaskTokenClassification,
		})
	}
	return models
}

func whisperIDAndOwner(input string) (id, owner string) {
	hasExt := filepath.Ext(input) != ""
	looksLikeHF := strings.Contains(input, "/") &&
		!strings.HasPrefix(input, "/") &&
		!strings.HasPrefix(input, ".") &&
		!hasExt
	if looksLikeHF {
		parts := strings.SplitN(input, "/", 2)
		return input, parts[0]
	}
	base := strings.TrimSuffix(filepath.Base(input), filepath.Ext(input))
	if base == "" {
		base = input
	}
	return base, "speaches-plus"
}
