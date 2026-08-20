package audio

import "math"

type PolyphaseUpsampler struct {
	in, out int
	taps    []float32
	half    int
	state   []float32
}

func NewPolyphaseUpsampler(srIn, srOut, halfTaps int) *PolyphaseUpsampler {
	if srOut%srIn != 0 {
		panic("polyphase: srOut must be an integer multiple of srIn")
	}
	if halfTaps <= 0 {
		halfTaps = 16
	}
	L := srOut / srIn
	n := 2*halfTaps*L + 1
	taps := make([]float32, n)
	cutoff := 0.5 / float64(L)
	for i := 0; i < n; i++ {
		x := float64(i-(n-1)/2) / float64(L)
		var s float64
		if x == 0 {
			s = 1.0
		} else {
			s = math.Sin(2*math.Pi*cutoff*x*float64(L)) / (math.Pi * x * float64(L))
		}
		w := 0.42 - 0.5*math.Cos(2*math.Pi*float64(i)/float64(n-1)) + 0.08*math.Cos(4*math.Pi*float64(i)/float64(n-1))
		taps[i] = float32(s * w)
	}
	return &PolyphaseUpsampler{
		in:    srIn,
		out:   srOut,
		taps:  taps,
		half:  halfTaps,
		state: make([]float32, halfTaps*2),
	}
}

func (p *PolyphaseUpsampler) Process(in MonoF32) MonoF32 {
	L := p.out / p.in
	outLen := len(in) * L
	out := make(MonoF32, outLen)
	hist := append(p.state, in...)
	for j := 0; j < outLen; j++ {
		srcIdx := j / L
		phase := j % L
		center := srcIdx + p.half
		var acc float32
		for k := -p.half; k <= p.half; k++ {
			tapIdx := (k+p.half)*L + phase
			if tapIdx < 0 || tapIdx >= len(p.taps) {
				continue
			}
			h := center + k
			if h < 0 || h >= len(hist) {
				continue
			}
			acc += p.taps[tapIdx] * hist[h]
		}
		out[j] = acc
	}
	if len(in) >= 2*p.half {
		copy(p.state, in[len(in)-2*p.half:])
	} else {
		newState := make([]float32, 2*p.half)
		old := append([]float32(nil), p.state...)
		combined := append(old, in...)
		if len(combined) >= 2*p.half {
			copy(newState, combined[len(combined)-2*p.half:])
		} else {
			copy(newState, combined)
		}
		p.state = newState
	}
	return out
}

func (p *PolyphaseUpsampler) Reset() {
	for i := range p.state {
		p.state[i] = 0
	}
}
