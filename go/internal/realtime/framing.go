package realtime

import (
	"encoding/base64"
	"encoding/json"
	"fmt"
	"log/slog"

	"github.com/pion/webrtc/v4"
)

var errClientTooSlow = fmt.Errorf("client_too_slow")

type fullMessageEnvelope struct {
	Type string `json:"type"`
	ID   string `json:"id"`
	Data string `json:"data"`
}

type partialMessageEnvelope struct {
	Type           string `json:"type"`
	ID             string `json:"id"`
	Data           string `json:"data"`
	FragmentIndex  int    `json:"fragment_index"`
	TotalFragments int    `json:"total_fragments"`
}

func sendFragmented(ch *webrtc.DataChannel, event any, eventID string) error {
	return sendFragmentedWith(ch, event, eventID,
		defaultDataChannelFragmentMax, defaultOutboundBufferLimit)
}

func sendFragmentedWith(ch *webrtc.DataChannel, event any, eventID string, fragMax int, bufLimit uint64) error {
	if fragMax <= 0 {
		fragMax = defaultDataChannelFragmentMax
	}
	if bufLimit == 0 {
		bufLimit = defaultOutboundBufferLimit
	}
	if ch.BufferedAmount() > bufLimit {
		return errClientTooSlow
	}
	body, err := json.Marshal(event)
	if err != nil {
		return err
	}
	if len(body) <= fragMax {
		envelope := fullMessageEnvelope{
			Type: "full_message",
			ID:   eventID,
			Data: base64.StdEncoding.EncodeToString(body),
		}
		raw, err := json.Marshal(envelope)
		if err != nil {
			return err
		}
		return ch.SendText(string(raw))
	}
	fragData := fragMax - envelopeBudget
	if fragData < 1 {
		fragData = 1
	}
	encoded := base64.StdEncoding.EncodeToString(body)
	total := (len(encoded) + fragData - 1) / fragData
	for i := 0; i < total; i++ {
		start := i * fragData
		end := start + fragData
		if end > len(encoded) {
			end = len(encoded)
		}
		envelope := partialMessageEnvelope{
			Type:           "partial_message",
			ID:             eventID,
			Data:           encoded[start:end],
			FragmentIndex:  i,
			TotalFragments: total,
		}
		raw, err := json.Marshal(envelope)
		if err != nil {
			return err
		}
		if err := ch.SendText(string(raw)); err != nil {
			return err
		}
	}
	slog.Debug("sent fragmented message", "id", eventID, "fragments", total, "bytes", len(encoded))
	return nil
}
