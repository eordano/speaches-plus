package audio

func MonoS16ToF32(samples MonoS16) MonoF32 {
	out := make(MonoF32, len(samples))
	for i, s := range samples {
		out[i] = float32(s) / 32768.0
	}
	return out
}

func LinearResampleF32(in MonoF32, srIn, srOut int) MonoF32 {
	if srIn == srOut || len(in) == 0 {
		out := make(MonoF32, len(in))
		copy(out, in)
		return out
	}
	nIn := len(in)
	nOut := nIn * srOut / srIn
	if nOut <= 0 {
		return nil
	}
	out := make(MonoF32, nOut)
	if nOut == 1 {
		out[0] = in[0]
		return out
	}
	step := float64(nIn-1) / float64(nOut-1)
	for i := 0; i < nOut; i++ {
		x := float64(i) * step
		idx := int(x)
		if idx >= nIn-1 {
			out[i] = in[nIn-1]
			continue
		}
		frac := float32(x - float64(idx))
		out[i] = in[idx]*(1-frac) + in[idx+1]*frac
	}
	return out
}
