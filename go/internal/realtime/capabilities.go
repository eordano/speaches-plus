package realtime

import (
	"encoding/json"
	"net/http"

	"github.com/eordano/speaches-plus-go/internal/diarization"
)

type capabilitiesResponse struct {
	RFCVersion string                `json:"rfc_version"`
	Features   capabilitiesFeature   `json:"features"`
	Extensions capabilitiesExtension `json:"extensions"`
}

type capabilitiesFeature struct {
	EouKinds           []string `json:"eou_kinds"`
	FusionRules        []string `json:"fusion_rules"`
	InputAudioFormats  []string `json:"input_audio_formats"`
	OutputAudioFormats []string `json:"output_audio_formats"`
}

type capabilitiesExtension struct {
	EouKinds    []string `json:"eou_kinds"`
	FusionRules []string `json:"fusion_rules"`

	EagerEou bool `json:"eager_eou"`

	IntegratedEou bool `json:"integrated_eou"`

	PredictedRespPhase bool `json:"predicted_resp_phase"`

	Diarization capabilitiesDiarization `json:"diarization"`
}

type capabilitiesDiarization struct {
	Enabled             bool                       `json:"enabled"`
	MaxSpeakersPerChunk uint32                     `json:"max_speakers_per_chunk"`
	MaxSpeakersPerFrame uint32                     `json:"max_speakers_per_frame"`
	EmbeddingDim        uint32                     `json:"embedding_dim"`
	FrameRateHz         uint32                     `json:"frame_rate_hz"`
	Endpoints           capabilitiesDiarizationEPs `json:"endpoints"`
}

type capabilitiesDiarizationEPs struct {
	AudioDiarization          string `json:"audio_diarization"`
	AudioEmbeddings           string `json:"audio_embeddings"`
	TranscriptionDiarizedJSON string `json:"transcription_diarized_json"`
	RealtimeEvent             string `json:"realtime_event"`
}

func (s *Server) HandleCapabilities(w http.ResponseWriter, _ *http.Request) {
	var seg *diarization.SegmentationModel
	var emb *diarization.EmbeddingModel
	if s != nil {
		seg = s.cfg.DiarSegmentation
		emb = s.cfg.DiarEmbedding
	}
	diar := capabilitiesDiarization{
		Enabled: seg != nil && emb != nil,
		Endpoints: capabilitiesDiarizationEPs{
			AudioDiarization:          "/v1/audio/diarization",
			AudioEmbeddings:           "/v1/audio/embeddings",
			TranscriptionDiarizedJSON: "/v1/audio/transcriptions?response_format=diarized_json",
			RealtimeEvent:             "conversation.item.diarization",
		},
	}
	if seg != nil {
		diar.MaxSpeakersPerChunk = uint32(seg.MaxSpeakersPerChunk())
		diar.MaxSpeakersPerFrame = uint32(seg.MaxSpeakersPerFrame())
		diar.FrameRateHz = uint32(seg.FrameRateHz())
	}
	if emb != nil {
		diar.EmbeddingDim = uint32(emb.Dim())
	}

	resp := capabilitiesResponse{
		RFCVersion: capabilityRFCVersion,
		Features: capabilitiesFeature{
			EouKinds:           supportedEouKindsV3(),
			FusionRules:        supportedFusionRulesV3(),
			InputAudioFormats:  supportedInputAudioFormats(),
			OutputAudioFormats: supportedOutputAudioFormats(),
		},
		Extensions: capabilitiesExtension{
			EouKinds:           supportedEouKindsExtensions(),
			FusionRules:        supportedFusionRulesExtensions(),
			EagerEou:           true,
			IntegratedEou:      true,
			PredictedRespPhase: true,
			Diarization:        diar,
		},
	}
	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(resp)
}

func supportedEouKindsV3() []string {
	return []string{"vad", "text", "audio", "fusion"}
}

func supportedEouKindsExtensions() []string {
	return []string{"heuristic", "integrated"}
}

var SupportedAudioFormats = []string{
	AudioFormatPCM16,
	AudioFormatPCM16_16K,
	AudioFormatG711Ulaw,
	AudioFormatG711Alaw,
}

func supportedInputAudioFormats() []string  { return SupportedAudioFormats }
func supportedOutputAudioFormats() []string { return SupportedAudioFormats }

func supportedFusionRulesV3() []string {
	return []string{"noisy_or", "max", "mean", "weighted"}
}

func supportedFusionRulesExtensions() []string {
	return []string{"gated"}
}

func supportedEouKinds() []string {
	return []string{"vad", "heuristic", "text", "audio", "fusion", "integrated"}
}

func supportedFusionRules() []string {
	return []string{"noisy_or", "max", "mean", "weighted", "gated"}
}

func validAudioFormat(v string, supported []string) bool {
	for _, s := range supported {
		if s == v {
			return true
		}
	}
	return false
}
