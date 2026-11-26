package audio

import "math"

type (
	MonoF32    []float32
	MonoS16    []int16
	StereoF32  []float32
	StereoS16  []int16
	PCM16Bytes []byte
)

const (
	MinDecodeSampleRate = 1000
	MaxDecodeSampleRate = 384000
)

func F32ToI16(s float32) int16 {
	if math.IsNaN(float64(s)) {
		return 0
	}
	if s > 1.0 {
		s = 1.0
	} else if s < -1.0 {
		s = -1.0
	}
	return int16(math.Round(float64(s * 32767.0)))
}
