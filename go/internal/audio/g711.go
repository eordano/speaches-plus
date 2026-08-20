package audio

const g711SignBit = 0x80

var ulawDecodeTable [256]int16
var alawDecodeTable [256]int16

func init() {
	for i := 0; i < 256; i++ {
		ulawDecodeTable[i] = ulawToLinearSlow(byte(i))
		alawDecodeTable[i] = alawToLinearSlow(byte(i))
	}
}

func ulawToLinearSlow(u byte) int16 {
	u = ^u
	sign := u & g711SignBit
	exponent := (u >> 4) & 0x07
	mantissa := u & 0x0F
	magnitude := (int32(mantissa) << 3) | 0x84
	magnitude <<= exponent
	magnitude -= 0x84
	if sign != 0 {
		return int16(-magnitude)
	}
	return int16(magnitude)
}

func alawToLinearSlow(a byte) int16 {
	a ^= 0x55
	sign := a & g711SignBit
	exponent := (a >> 4) & 0x07
	mantissa := int32(a & 0x0F)
	var magnitude int32
	if exponent == 0 {
		magnitude = (mantissa << 4) | 0x08
	} else {
		magnitude = ((mantissa << 4) | 0x108) << (exponent - 1)
	}
	if sign == 0 {
		return int16(-magnitude)
	}
	return int16(magnitude)
}

func linearToUlaw(s int16) byte {
	const bias = 0x84
	const clip = 32635
	sample := int32(s)
	var sign byte
	if sample < 0 {
		sample = -sample
		sign = g711SignBit
	}
	if sample > clip {
		sample = clip
	}
	sample += bias
	exponent := byte(7)
	for mask := int32(0x4000); sample&mask == 0 && exponent > 0; mask >>= 1 {
		exponent--
	}
	mantissa := byte((sample >> (exponent + 3)) & 0x0F)
	return ^(sign | (exponent << 4) | mantissa)
}

func linearToAlaw(s int16) byte {
	sample := int32(s)
	var sign byte = 0x80
	if sample < 0 {
		sample = -sample - 1
		sign = 0x00
	}
	if sample > 32767 {
		sample = 32767
	}
	var compressed byte
	if sample < 256 {
		compressed = byte(sample >> 4)
	} else {
		exponent := byte(7)
		for mask := int32(0x4000); sample&mask == 0 && exponent > 1; mask >>= 1 {
			exponent--
		}
		mantissa := byte((sample >> (exponent + 3)) & 0x0F)
		compressed = (exponent << 4) | mantissa
	}
	return (compressed | sign) ^ 0x55
}

func ULawBytesToF32(b []byte) MonoF32 {
	out := make(MonoF32, len(b))
	for i, c := range b {
		out[i] = float32(ulawDecodeTable[c]) / 32768.0
	}
	return out
}

func ALawBytesToF32(b []byte) MonoF32 {
	out := make(MonoF32, len(b))
	for i, c := range b {
		out[i] = float32(alawDecodeTable[c]) / 32768.0
	}
	return out
}

func F32ToULawBytes(samples MonoF32) []byte {
	out := make([]byte, len(samples))
	for i, s := range samples {
		v := int32(clampF32_(s, -1.0, 1.0) * 32767.0)
		if v > 32767 {
			v = 32767
		} else if v < -32768 {
			v = -32768
		}
		out[i] = linearToUlaw(int16(v))
	}
	return out
}

func F32ToALawBytes(samples MonoF32) []byte {
	out := make([]byte, len(samples))
	for i, s := range samples {
		v := int32(clampF32_(s, -1.0, 1.0) * 32767.0)
		if v > 32767 {
			v = 32767
		} else if v < -32768 {
			v = -32768
		}
		out[i] = linearToAlaw(int16(v))
	}
	return out
}

func clampF32_(v, lo, hi float32) float32 {
	if v < lo {
		return lo
	}
	if v > hi {
		return hi
	}
	return v
}
