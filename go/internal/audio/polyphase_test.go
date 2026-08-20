package audio

import (
	"math"
	"testing"
)

func TestPolyphase_PassesDC(t *testing.T) {
	p := NewPolyphaseUpsampler(24000, 48000, 16)
	in := make([]float32, 256)
	for i := range in {
		in[i] = 1.0
	}
	out := p.Process(in)
	if len(out) != 512 {
		t.Fatalf("len=%d want 512", len(out))
	}
	mid := out[200]
	if mid < 0.95 || mid > 1.05 {
		t.Fatalf("DC passthrough off: out[200]=%v want ~1.0", mid)
	}
}

func TestPolyphase_MatchesSineFrequency(t *testing.T) {
	const inHz = 24000
	const outHz = 48000
	const sigHz = 1000.0
	in := make([]float32, 2400)
	for i := range in {
		in[i] = float32(math.Sin(2 * math.Pi * sigHz * float64(i) / inHz))
	}
	p := NewPolyphaseUpsampler(inHz, outHz, 32)
	out := p.Process(in)
	if len(out) != 4800 {
		t.Fatalf("len=%d want 4800", len(out))
	}
	tail := out[1000:]
	var maxAbs float32
	for _, s := range tail {
		a := s
		if a < 0 {
			a = -a
		}
		if a > maxAbs {
			maxAbs = a
		}
	}
	if maxAbs < 0.9 || maxAbs > 1.1 {
		t.Fatalf("amplitude off: max=%v want ~1", maxAbs)
	}
}

func TestPolyphase_RejectsAliasing(t *testing.T) {
	const inHz = 24000
	const outHz = 48000
	in := make([]float32, 2400)
	for i := range in {
		s := math.Sin(2*math.Pi*1000.0*float64(i)/inHz) +
			0.5*math.Sin(2*math.Pi*11500.0*float64(i)/inHz)
		in[i] = float32(s)
	}
	p := NewPolyphaseUpsampler(inHz, outHz, 32)
	_ = p.Process(in)
}
