package audio

import (
	"bytes"
	"encoding/binary"
	"fmt"
	"strings"
)

const (
	SampleRate16k     = 16000
	mimeRawPCM        = "audio/pcm"
	mimeRawPCMAlias   = "audio/raw"
	bytesPerS16Sample = 2
	s16Scale          = 32768.0
)

func DecodeUploadedAudio(data []byte, mime string) (MonoF32, error) {
	if len(data) == 0 {
		return nil, fmt.Errorf("audio: empty input")
	}
	mime = strings.ToLower(strings.TrimSpace(mime))
	if mime == mimeRawPCM || mime == mimeRawPCMAlias {
		return decodePCMS16LE16k(data), nil
	}
	if wav, err := DecodeWAV(bytes.NewReader(data)); err == nil {
		samples := wav.Samples
		if wav.SampleRate != SampleRate16k {
			samples = LinearResampleF32(samples, wav.SampleRate, SampleRate16k)
		}
		return samples, nil
	}
	return DecodeAnyToMono16k(data)
}

func decodePCMS16LE16k(data []byte) MonoF32 {
	n := len(data) / bytesPerS16Sample
	out := make(MonoF32, n)
	for i := 0; i < n; i++ {
		s := int16(binary.LittleEndian.Uint16(data[i*bytesPerS16Sample : (i+1)*bytesPerS16Sample]))
		out[i] = float32(s) / s16Scale
	}
	return out
}
