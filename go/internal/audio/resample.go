package audio

import "math"

func MonoS16ToF32(samples MonoS16) MonoF32 {
	out := make(MonoF32, len(samples))
	for i, s := range samples {
		out[i] = float32(s) / 32768.0
	}
	return out
}

func LinearResampleF32(in MonoF32, srIn, srOut int) MonoF32 {
	if srIn < MinDecodeSampleRate || srIn > MaxDecodeSampleRate || srOut < MinDecodeSampleRate || srOut > MaxDecodeSampleRate {
		return MonoF32{}
	}
	if srIn == srOut {
		return append(MonoF32{}, in...)
	}
	nIn := len(in)
	nOut := int(uint64(nIn) * uint64(srOut) / uint64(srIn))
	out := make(MonoF32, nOut)
	for i := range out {
		pos := float64(i) * float64(srIn) / float64(srOut)
		lo := min(int(math.Floor(pos)), nIn-1)
		hi := min(lo+1, nIn-1)
		t := pos - float64(lo)
		out[i] = float32(float64(float64(in[lo])*(1.0-t)) + float64(float64(in[hi])*t))
	}
	return out
}
