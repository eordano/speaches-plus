package oapi

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"
)

func TestModelsHandlerEmpty(t *testing.T) {
	rr := httptest.NewRecorder()
	NewModelsHandler(ModelsListing{})(rr, httptest.NewRequest(http.MethodGet, "/v1/models", nil))
	if rr.Code != 200 {
		t.Fatalf("status %d", rr.Code)
	}
	var out map[string]any
	if err := json.Unmarshal(rr.Body.Bytes(), &out); err != nil {
		t.Fatalf("unmarshal: %v\nbody=%s", err, rr.Body.String())
	}
	if out["object"] != "list" {
		t.Errorf("object = %v, want list", out["object"])
	}
	if data, ok := out["data"].([]any); !ok || len(data) != 0 {
		t.Errorf("data = %v, want empty list", out["data"])
	}
}

func TestModelsHandlerFullListing(t *testing.T) {
	rr := httptest.NewRecorder()
	NewModelsHandler(ModelsListing{
		WhisperModelID: "deepdml/faster-whisper-large-v3-turbo-ct2",
		KokoroVoices:   []string{"af_heart", "am_michael"},
		SileroLoaded:   true,
	})(rr, httptest.NewRequest(http.MethodGet, "/v1/models", nil))
	if rr.Code != 200 {
		t.Fatalf("status %d", rr.Code)
	}
	var out struct {
		Object string                   `json:"object"`
		Data   []map[string]interface{} `json:"data"`
	}
	if err := json.Unmarshal(rr.Body.Bytes(), &out); err != nil {
		t.Fatalf("unmarshal: %v\nbody=%s", err, rr.Body.String())
	}
	if out.Object != "list" {
		t.Errorf("object = %q", out.Object)
	}
	if len(out.Data) != 3 {
		t.Fatalf("expected 3 models, got %d", len(out.Data))
	}
	byID := map[string]map[string]interface{}{}
	for _, m := range out.Data {
		byID[m["id"].(string)] = m
	}
	if w, ok := byID["deepdml/faster-whisper-large-v3-turbo-ct2"]; !ok {
		t.Fatalf("missing whisper entry: %v", byID)
	} else if w["task"] != TaskASR || w["owned_by"] != "deepdml" {
		t.Errorf("whisper bad: %v", w)
	}
	if k, ok := byID["speaches-ai/Kokoro-82M-v1.0-ONNX"]; !ok {
		t.Fatalf("missing kokoro entry")
	} else {
		if k["task"] != TaskTTS {
			t.Errorf("kokoro task=%v", k["task"])
		}
		if voices, ok := k["voices"].([]interface{}); !ok || len(voices) != 2 {
			t.Errorf("kokoro voices missing/wrong: %v", k["voices"])
		}
		if k["sample_rate"].(float64) != 24000 {
			t.Errorf("kokoro sample_rate=%v", k["sample_rate"])
		}
	}
	if v, ok := byID["silero_vad_v6"]; !ok {
		t.Fatalf("missing silero entry")
	} else if v["task"] != TaskVAD || v["owned_by"] != "snakers4" {
		t.Errorf("silero bad: %v", v)
	}
}

func TestModelsHandlerTaskFilter(t *testing.T) {
	rr := httptest.NewRecorder()
	NewModelsHandler(ModelsListing{
		WhisperModelID: "deepdml/faster-whisper-large-v3-turbo-ct2",
		KokoroVoices:   []string{"af_heart"},
		SileroLoaded:   true,
	})(rr, httptest.NewRequest(http.MethodGet, "/v1/models?task=text-to-speech", nil))
	if rr.Code != 200 {
		t.Fatalf("status %d", rr.Code)
	}
	var out struct {
		Data []map[string]interface{} `json:"data"`
	}
	_ = json.Unmarshal(rr.Body.Bytes(), &out)
	if len(out.Data) != 1 {
		t.Fatalf("expected 1 model after filter, got %d", len(out.Data))
	}
	if out.Data[0]["task"] != TaskTTS {
		t.Errorf("filter returned non-TTS: %v", out.Data[0])
	}
}

func TestWhisperIDInferOwnerFromPath(t *testing.T) {
	id, owner := whisperIDAndOwner("models/ggml-tiny.en.bin")
	if id != "ggml-tiny.en" {
		t.Errorf("id=%q", id)
	}
	if owner != "speaches-plus" {
		t.Errorf("owner=%q", owner)
	}
}
