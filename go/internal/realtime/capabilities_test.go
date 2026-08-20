package realtime

import (
	"encoding/json"
	"net/http/httptest"
	"testing"
)

func TestCapabilities_ShapeMatchesRust(t *testing.T) {
	srv := &Server{}
	rec := httptest.NewRecorder()
	srv.HandleCapabilities(rec, httptest.NewRequest("GET", "/v1/realtime/capabilities", nil))

	if got := rec.Code; got != 200 {
		t.Fatalf("status: want 200, got %d", got)
	}

	var body map[string]any
	if err := json.Unmarshal(rec.Body.Bytes(), &body); err != nil {
		t.Fatalf("decode: %v", err)
	}

	if body["rfc_version"] != "v3" {
		t.Fatalf("rfc_version: want v3, got %v", body["rfc_version"])
	}

	feat, ok := body["features"].(map[string]any)
	if !ok {
		t.Fatalf("features: missing or wrong type")
	}
	for _, key := range []string{"eou_kinds", "fusion_rules", "input_audio_formats", "output_audio_formats"} {
		if _, ok := feat[key]; !ok {
			t.Errorf("features.%s: missing", key)
		}
	}

	ext, ok := body["extensions"].(map[string]any)
	if !ok {
		t.Fatalf("extensions: missing or wrong type")
	}

	for _, key := range []string{"eager_eou", "integrated_eou", "predicted_resp_phase"} {
		v, ok := ext[key]
		if !ok {
			t.Errorf("extensions.%s: missing", key)
			continue
		}
		if v != true {
			t.Errorf("extensions.%s: expected true, got %v", key, v)
		}
	}

	rules, ok := ext["fusion_rules"].([]any)
	if !ok {
		t.Fatalf("extensions.fusion_rules: missing or wrong type")
	}
	hasGated := false
	for _, r := range rules {
		if r == "gated" {
			hasGated = true
			break
		}
	}
	if !hasGated {
		t.Errorf("extensions.fusion_rules: must include 'gated', got %v", rules)
	}

	diar, ok := ext["diarization"].(map[string]any)
	if !ok {
		t.Fatalf("extensions.diarization: missing or wrong type")
	}
	for _, key := range []string{
		"enabled", "max_speakers_per_chunk", "max_speakers_per_frame",
		"embedding_dim", "frame_rate_hz", "endpoints",
	} {
		if _, ok := diar[key]; !ok {
			t.Errorf("extensions.diarization.%s: missing", key)
		}
	}
	endpoints, ok := diar["endpoints"].(map[string]any)
	if !ok {
		t.Fatalf("extensions.diarization.endpoints: missing or wrong type")
	}
	for _, key := range []string{
		"audio_diarization", "audio_embeddings",
		"transcription_diarized_json", "realtime_event",
	} {
		if _, ok := endpoints[key]; !ok {
			t.Errorf("extensions.diarization.endpoints.%s: missing", key)
		}
	}

	if diar["enabled"] != false {
		t.Errorf("extensions.diarization.enabled: expected false (no models configured), got %v", diar["enabled"])
	}
}

func TestCapabilities_FeaturesHasOnlyV3MandatoryRules(t *testing.T) {
	srv := &Server{}
	rec := httptest.NewRecorder()
	srv.HandleCapabilities(rec, httptest.NewRequest("GET", "/v1/realtime/capabilities", nil))

	var body map[string]any
	_ = json.Unmarshal(rec.Body.Bytes(), &body)
	feat := body["features"].(map[string]any)
	rules := feat["fusion_rules"].([]any)
	for _, r := range rules {
		if r == "gated" {
			t.Errorf("features.fusion_rules must not include extension 'gated'; got %v", rules)
		}
	}
	want := map[string]bool{"noisy_or": true, "max": true, "mean": true, "weighted": true}
	for _, r := range rules {
		if _, ok := want[r.(string)]; !ok {
			t.Errorf("unexpected v3-mandatory rule %q", r)
		}
	}
}
