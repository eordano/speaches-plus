package realtime

import (
	"encoding/json"
)

var knownV2NoopEvents = map[string]struct{}{
	"output_audio_buffer.clear":         {},
	"output_audio_buffer.append":        {},
	"input_audio_buffer.dtmf.received":  {},
	"transcription_session.update":      {},
	"response.cancel_audio":             {},
}

func isKnownV2NoopEvent(t string) bool {
	_, ok := knownV2NoopEvents[t]
	return ok
}

func modalitiesForIntent(conversation bool) []string {
	if conversation {
		return []string{ModalityText, ModalityAudio}
	}
	return []string{ModalityText}
}

func normalizeSessionRaw(data []byte) []byte {
	var m map[string]json.RawMessage
	if err := json.Unmarshal(data, &m); err != nil {
		return data
	}

	if audioRaw, ok := m["audio"]; ok {
		var audio map[string]json.RawMessage
		if json.Unmarshal(audioRaw, &audio) == nil {
			if inputRaw, ok := audio["input"]; ok {
				var input map[string]json.RawMessage
				if json.Unmarshal(inputRaw, &input) == nil {
					if v, ok := input["format"]; ok {
						if _, present := m["input_audio_format"]; !present {
							m["input_audio_format"] = v
						}
					}
					if v, ok := input["transcription"]; ok {
						if _, present := m["input_audio_transcription"]; !present {
							m["input_audio_transcription"] = v
						}
					}
					if v, ok := input["turn_detection"]; ok {
						if _, present := m["turn_detection"]; !present {
							m["turn_detection"] = v
						}
					}
				}
			}
			if outputRaw, ok := audio["output"]; ok {
				var output map[string]json.RawMessage
				if json.Unmarshal(outputRaw, &output) == nil {
					if v, ok := output["format"]; ok {
						if _, present := m["output_audio_format"]; !present {
							m["output_audio_format"] = v
						}
					}
					if v, ok := output["voice"]; ok {
						if _, present := m["voice"]; !present {
							m["voice"] = v
						}
					}
					if v, ok := output["speed"]; ok {
						if _, present := m["speed"]; !present {
							m["speed"] = v
						}
					}
				}
			}
		}
	}

	if outMods, ok := m["output_modalities"]; ok {
		if _, present := m["modalities"]; !present {
			m["modalities"] = outMods
		}
	}

	out, err := json.Marshal(m)
	if err != nil {
		return data
	}
	return out
}

func enrichSessionPayload(m map[string]any) map[string]any {
	if _, present := m["type"]; !present {
		m["type"] = "realtime"
	}

	audio, _ := m["audio"].(map[string]any)
	if audio == nil {
		audio = map[string]any{}
		m["audio"] = audio
	}
	input, _ := audio["input"].(map[string]any)
	if input == nil {
		input = map[string]any{}
		audio["input"] = input
	}
	if v, ok := m["input_audio_format"]; ok {
		if _, present := input["format"]; !present {
			input["format"] = v
		}
	}
	if v, ok := m["input_audio_transcription"]; ok {
		if _, present := input["transcription"]; !present {
			input["transcription"] = v
		}
	}
	if v, ok := m["turn_detection"]; ok {
		if _, present := input["turn_detection"]; !present {
			input["turn_detection"] = v
		}
	}
	output, _ := audio["output"].(map[string]any)
	if output == nil {
		output = map[string]any{}
		audio["output"] = output
	}
	if v, ok := m["output_audio_format"]; ok {
		if _, present := output["format"]; !present {
			output["format"] = v
		}
	}
	if v, ok := m["voice"]; ok {
		if _, present := output["voice"]; !present {
			output["voice"] = v
		}
	}

	if v, ok := m["modalities"]; ok {
		if _, present := m["output_modalities"]; !present {
			m["output_modalities"] = v
		}
	}

	return m
}
