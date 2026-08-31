package realtime

import (
	"reflect"
	"sort"
	"strings"
	"testing"

	"github.com/eordano/speaches-plus-go/internal/wire"
)

func jsonKeySet(t reflect.Type) (required, optional map[string]bool) {
	required = map[string]bool{}
	optional = map[string]bool{}
	for i := 0; i < t.NumField(); i++ {
		tag := t.Field(i).Tag.Get("json")
		if tag == "" || tag == "-" {
			continue
		}
		parts := strings.Split(tag, ",")
		name := parts[0]
		opt := false
		for _, p := range parts[1:] {
			if p == "omitempty" {
				opt = true
			}
		}
		if opt {
			optional[name] = true
		} else {
			required[name] = true
		}
	}
	return
}

func sorted(m map[string]bool) []string {
	out := make([]string, 0, len(m))
	for k := range m {
		out = append(out, k)
	}
	sort.Strings(out)
	return out
}

func TestServerEventStructsMatchGeneratedWire(t *testing.T) {
	type divergence struct {
		missing []string
		extra   []string
	}
	knownDivergences := map[string]divergence{
		"input_audio_buffer.committed":             {extra: []string{"previous_item_id"}},
		"input_audio_buffer.partial_transcription": {missing: []string{"audio_end_ms"}},
		"conversation.item.added":                  {extra: []string{"previous_item_id"}},
	}

	pairs := []struct {
		name string
		hand any
		gen  any
	}{
		{"session.created", sessionCreatedEvent{}, wire.SessionCreatedEvent{}},
		{"session.updated", sessionUpdatedEvent{}, wire.SessionUpdatedEvent{}},
		{"session.done", sessionDoneEvent{}, wire.SessionDoneEvent{}},
		{"input_audio_buffer.speech_started", inputAudioBufferSpeechStartedEvent{}, wire.SpeechStartedEvent{}},
		{"input_audio_buffer.speech_stopped", inputAudioBufferSpeechStoppedEvent{}, wire.SpeechStoppedEvent{}},
		{"input_audio_buffer.committed", inputAudioBufferCommittedEvent{}, wire.BufferCommittedEvent{}},
		{"input_audio_buffer.partial_transcription", inputAudioBufferPartialTranscriptionEvent{}, wire.PartialTranscriptionEvent{}},
		{"conversation.item.diarization", conversationItemDiarizationEvent{}, wire.DiarizationEvent{}},
		{"conversation.item.added", conversationItemCreatedEvent{}, wire.ItemAddedEvent{}},
		{"conversation.item.truncated", conversationItemTruncatedEvent{}, wire.ItemTruncatedClientAckEvent{}},
		{"conversation.item.assistant_truncated", conversationItemAssistantTruncatedEvent{}, wire.AssistantTruncatedEvent{}},
		{"conversation.item.deleted", conversationItemDeletedEvent{}, wire.ItemDeletedEvent{}},
		{"conversation.item.input_audio_transcription.completed", transcriptionCompletedEvent{}, wire.TranscriptionCompletedEvent{}},
		{"conversation.item.input_audio_transcription.delta", transcriptionDeltaEvent{}, wire.TranscriptionDeltaEvent{}},
		{"conversation.item.input_audio_transcription.failed", transcriptionFailedEvent{}, wire.TranscriptionFailedEvent{}},
		{"conversation.item.done", conversationItemDoneEvent{}, wire.ItemDoneEvent{}},
		{"conversation.item.retrieved", conversationItemRetrievedEvent{}, wire.ItemRetrievedEvent{}},
		{"response.created", responseCreatedEvent{}, wire.ResponseCreatedEvent{}},
		{"response.output_item.added", responseOutputItemAddedEvent{}, wire.ResponseOutputItemAddedEvent{}},
		{"response.output_item.done", responseOutputItemDoneEvent{}, wire.ResponseOutputItemDoneEvent{}},
		{"response.content_part.added", responseContentPartAddedEvent{}, wire.ResponseContentPartAddedEvent{}},
		{"response.content_part.done", responseContentPartDoneEvent{}, wire.ResponseContentPartDoneEvent{}},
		{"response.output_audio_transcript.delta", responseAudioTranscriptDeltaEvent{}, wire.ResponseOutputAudioTranscriptDeltaEvent{}},
		{"response.output_audio_transcript.done", responseAudioTranscriptDoneEvent{}, wire.ResponseOutputAudioTranscriptDoneEvent{}},
		{"response.output_audio.delta", responseAudioDeltaEvent{}, wire.ResponseOutputAudioDeltaEvent{}},
		{"response.output_audio.done", responseAudioDoneEvent{}, wire.ResponseOutputAudioDoneEvent{}},
		{"response.output_text.delta", responseOutputTextDeltaEvent{}, wire.ResponseOutputTextDeltaEvent{}},
		{"response.output_text.done", responseOutputTextDoneEvent{}, wire.ResponseOutputTextDoneEvent{}},
		{"response.function_call_arguments.delta", responseFunctionCallArgumentsDeltaEvent{}, wire.ResponseFunctionCallArgumentsDeltaEvent{}},
		{"response.function_call_arguments.done", responseFunctionCallArgumentsDoneEvent{}, wire.ResponseFunctionCallArgumentsDoneEvent{}},
		{"response.tool_progress", responseToolProgressEvent{}, wire.ResponseToolProgressEvent{}},
		{"response.cancelled", responseCancelledEvent{}, wire.ResponseCancelledEvent{}},
		{"response.done", responseDoneEvent{}, wire.ResponseDoneEvent{}},
		{"output_audio_buffer.started", outputAudioBufferStartedEvent{}, wire.OutputAudioBufferStartedEvent{}},
		{"output_audio_buffer.stopped", outputAudioBufferStoppedEvent{}, wire.OutputAudioBufferStoppedEvent{}},
		{"rate_limits.updated", rateLimitsUpdatedEvent{}, wire.RateLimitsUpdatedEvent{}},
		{"error", errorEvent{}, wire.ErrorEvent{}},
	}

	for _, p := range pairs {
		handReq, handOpt := jsonKeySet(reflect.TypeOf(p.hand))
		genReq, genOpt := jsonKeySet(reflect.TypeOf(p.gen))
		handAll := map[string]bool{}
		for k := range handReq {
			handAll[k] = true
		}
		for k := range handOpt {
			handAll[k] = true
		}
		genAll := map[string]bool{"event_id": true, "type": true}
		for k := range genReq {
			genAll[k] = true
		}
		for k := range genOpt {
			genAll[k] = true
		}

		missing := map[string]bool{}
		for k := range genReq {
			if !handAll[k] {
				missing[k] = true
			}
		}
		for _, envelope := range []string{"event_id", "type"} {
			if !handAll[envelope] {
				missing[envelope] = true
			}
		}
		extra := map[string]bool{}
		for k := range handAll {
			if !genAll[k] {
				extra[k] = true
			}
		}

		known := knownDivergences[p.name]
		wantMissing := map[string]bool{}
		for _, k := range known.missing {
			wantMissing[k] = true
		}
		wantExtra := map[string]bool{}
		for _, k := range known.extra {
			wantExtra[k] = true
		}
		if !reflect.DeepEqual(missing, wantMissing) && (len(missing) > 0 || len(wantMissing) > 0) {
			t.Errorf("%s: keys missing vs canonical wire = %v, known divergence table says %v",
				p.name, sorted(missing), known.missing)
		}
		if !reflect.DeepEqual(extra, wantExtra) && (len(extra) > 0 || len(wantExtra) > 0) {
			t.Errorf("%s: keys beyond canonical wire = %v, known divergence table says %v",
				p.name, sorted(extra), known.extra)
		}
	}
}

func TestToolProgressToolsFieldIsTheGeneratedEntryType(t *testing.T) {
	f, ok := reflect.TypeOf(responseToolProgressEvent{}).FieldByName("Tools")
	if !ok {
		t.Fatal("responseToolProgressEvent lost its Tools field")
	}
	want := reflect.TypeOf([]wire.ToolProgressEntry{})
	if f.Type != want {
		t.Fatalf("Tools field type = %v, want %v (the generated wire entry)", f.Type, want)
	}
}
