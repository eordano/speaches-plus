package realtime

import (
	"errors"
	"fmt"
	"log/slog"
	"sync"
	"sync/atomic"
	"time"

	"github.com/pion/webrtc/v4"
	"github.com/pion/webrtc/v4/pkg/media"

	"github.com/eordano/speaches-plus-go/internal/audio"
)

type outboundAudio struct {
	track   *webrtc.TrackLocalStaticSample
	encoder *audio.OpusEncoder
	logger  *slog.Logger

	writeMu  sync.Mutex
	bytePool sync.Pool

	upsamplerInRate int
	upsampler       *audio.PolyphaseUpsampler

	playedMs atomic.Int64
}

func newOutboundAudio(logger *slog.Logger, channels uint16) (*outboundAudio, error) {
	if channels == 0 {
		channels = 1
	}
	if channels > 2 {
		channels = 2
	}
	enc, err := audio.NewOpusEncoder(rtpOutSampleRate)
	if err != nil {
		return nil, fmt.Errorf("opus encoder: %w", err)
	}

	track, err := webrtc.NewTrackLocalStaticSample(
		webrtc.RTPCodecCapability{
			MimeType:  webrtc.MimeTypeOpus,
			ClockRate: rtpOutSampleRate,
			Channels:  channels,
		},
		"audio",
		"speaches-plus",
	)
	if err != nil {
		enc.Close()
		return nil, fmt.Errorf("new track: %w", err)
	}

	o := &outboundAudio{track: track, encoder: enc, logger: logger}
	o.bytePool.New = func() any {
		b := make([]byte, 0, opusEncodeScratchCap)
		return &b
	}
	return o, nil
}

func (o *outboundAudio) Track() webrtc.TrackLocal { return o.track }

func (o *outboundAudio) PlayedMs() int64 {
	if o == nil {
		return 0
	}
	return o.playedMs.Load()
}

func (o *outboundAudio) ResetPlayedMs() {
	if o == nil {
		return
	}
	o.playedMs.Store(0)
}

func (o *outboundAudio) Close() {
	if o == nil {
		return
	}
	if o.encoder != nil {
		o.encoder.Close()
	}
}

func (o *outboundAudio) WriteAudio(samples audio.MonoF32, sampleRate int) error {
	if o == nil || o.encoder == nil || o.track == nil {
		return errors.New("outbound audio not initialized")
	}
	o.writeMu.Lock()
	defer o.writeMu.Unlock()

	frameSamples := rtpOutSampleRate * opusFrameMs / 1000
	frameDuration := time.Duration(opusFrameMs) * time.Millisecond

	var mono48 audio.MonoF32
	switch {
	case sampleRate == rtpOutSampleRate:
		mono48 = samples
	case rtpOutSampleRate%sampleRate == 0:
		if o.upsampler == nil || o.upsamplerInRate != sampleRate {
			o.upsampler = audio.NewPolyphaseUpsampler(sampleRate, rtpOutSampleRate, 24)
			o.upsamplerInRate = sampleRate
		}
		mono48 = o.upsampler.Process(samples)
	default:
		mono48 = audio.LinearResampleF32(samples, sampleRate, rtpOutSampleRate)
	}

	if pad := len(mono48) % frameSamples; pad != 0 {
		mono48 = append(mono48, make(audio.MonoF32, frameSamples-pad)...)
	}

	scratch := make([]byte, opusEncodeScratchCap)
	deadline := time.Now()
	for i := 0; i+frameSamples <= len(mono48); i += frameSamples {
		frame := mono48[i : i+frameSamples]
		encoded, err := o.encoder.EncodeFrame(frame, scratch)
		if err != nil {
			return fmt.Errorf("opus encode: %w", err)
		}
		bufp := o.bytePool.Get().(*[]byte)
		buf := append((*bufp)[:0], encoded...)
		err = o.track.WriteSample(media.Sample{
			Data:     buf,
			Duration: frameDuration,
		})
		*bufp = buf
		o.bytePool.Put(bufp)
		if err != nil {
			return fmt.Errorf("write sample: %w", err)
		}

		o.playedMs.Add(int64(opusFrameMs))

		deadline = deadline.Add(frameDuration)
		if d := time.Until(deadline); d > 0 {
			time.Sleep(d)
		}
	}
	return nil
}
