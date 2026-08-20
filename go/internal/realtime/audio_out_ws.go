package realtime

import (
	"context"
	"encoding/base64"
	"errors"
	"fmt"
	"log/slog"
	"sync"
	"sync/atomic"
	"time"

	"github.com/coder/websocket"

	"github.com/eordano/speaches-plus-go/internal/audio"
)

type wsOutbound struct {
	conn       wsTransport
	logger     *slog.Logger
	format     string
	sampleRate int
	frameMs    int
	frameSamps int

	upsamplerInRate int
	upsampler       *audio.PolyphaseUpsampler

	writeMu  *sync.Mutex
	playedMs atomic.Int64

	eventID func() string
}

func newWSOutbound(conn wsTransport, logger *slog.Logger, format string, eventID func() string, writeMu *sync.Mutex) *wsOutbound {
	sr := wireSampleRateFor(format)
	return &wsOutbound{
		conn:       conn,
		logger:     logger,
		format:     format,
		sampleRate: sr,
		frameMs:    opusFrameMs,
		frameSamps: sr * opusFrameMs / 1000,
		writeMu:    writeMu,
		eventID:    eventID,
	}
}

func wireSampleRateFor(format string) int {
	switch format {
	case "pcm16_16k":
		return 16000
	case "g711_ulaw", "g711_alaw":
		return 8000
	case "pcm16":
		fallthrough
	default:
		return 24000
	}
}

func (o *wsOutbound) PlayedMs() int64 {
	if o == nil {
		return 0
	}
	return o.playedMs.Load()
}

func (o *wsOutbound) ResetPlayedMs() {
	if o == nil {
		return
	}
	o.playedMs.Store(0)
}

func (o *wsOutbound) WriteAudio(samples audio.MonoF32, sampleRate int) error {
	if o == nil || o.conn == nil {
		return errors.New("ws outbound not initialized")
	}

	var resampled audio.MonoF32
	switch {
	case sampleRate == o.sampleRate:
		resampled = samples
	case o.sampleRate%sampleRate == 0:
		if o.upsampler == nil || o.upsamplerInRate != sampleRate {
			o.upsampler = audio.NewPolyphaseUpsampler(sampleRate, o.sampleRate, 24)
			o.upsamplerInRate = sampleRate
		}
		resampled = o.upsampler.Process(samples)
	default:
		resampled = audio.LinearResampleF32(samples, sampleRate, o.sampleRate)
	}

	if pad := len(resampled) % o.frameSamps; pad != 0 {
		resampled = append(resampled, make(audio.MonoF32, o.frameSamps-pad)...)
	}

	frameDuration := time.Duration(o.frameMs) * time.Millisecond
	deadline := time.Now()
	pcmBuf := make([]byte, o.frameSamps*2)

	for i := 0; i+o.frameSamps <= len(resampled); i += o.frameSamps {
		frame := resampled[i : i+o.frameSamps]
		var payload []byte
		switch o.format {
		case "g711_ulaw":
			payload = audio.F32ToULawBytes(frame)
		case "g711_alaw":
			payload = audio.F32ToALawBytes(frame)
		default:
			for j, s := range frame {
				v := int16(clampF32(s, -1.0, 1.0) * 32767.0)
				pcmBuf[j*2] = byte(v)
				pcmBuf[j*2+1] = byte(v >> 8)
			}
			payload = pcmBuf
		}
		b64 := base64.StdEncoding.EncodeToString(payload)
		ev := map[string]string{
			"event_id": o.eventID(),
			"type":     string(SETResponseOutputAudioDelta),
			"delta":    b64,
		}
		body, err := jsonMarshalEvent(ev)
		if err != nil {
			return fmt.Errorf("audio.delta marshal: %w", err)
		}
		ctx, cancel := context.WithTimeout(context.Background(), wsWriteTimeoutSec*time.Second)
		if o.writeMu != nil {
			o.writeMu.Lock()
		}
		err = o.conn.Write(ctx, websocket.MessageText, body)
		if o.writeMu != nil {
			o.writeMu.Unlock()
		}
		cancel()
		if err != nil {
			return fmt.Errorf("ws write: %w", err)
		}

		o.playedMs.Add(int64(o.frameMs))

		deadline = deadline.Add(frameDuration)
		if d := time.Until(deadline); d > 0 {
			time.Sleep(d)
		}
	}
	return nil
}

func clampF32(v, lo, hi float32) float32 {
	if v < lo {
		return lo
	}
	if v > hi {
		return hi
	}
	return v
}
