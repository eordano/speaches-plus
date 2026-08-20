package diarization

import (
	"fmt"
	"math"
	"math/cmplx"
)

const (
	preEmphasis     = 0.97
	fbankSampleRate = 16000.0
	fbankLowFreqHz  = 20.0
	fbankHighFreqHz = 7600.0
	fbankLogFloor   = 1e-10
	poveyExponent   = 0.85
	melHzScale      = 700.0
	melLogScale     = 1127.0
)

type FBank struct {
	numMels     int
	frameLength int
	frameShift  int
	nFFT        int
	window      []float32
	melFilters  [][]melTap
}

type melTap struct {
	bin    int
	weight float32
}

func NewFBank(numMels, frameLength, frameShift int) *FBank {
	nFFT := nextPowerOfTwo(frameLength)
	return &FBank{
		numMels:     numMels,
		frameLength: frameLength,
		frameShift:  frameShift,
		nFFT:        nFFT,
		window:      poveyWindow(frameLength),
		melFilters:  buildMelFilters(numMels, nFFT, fbankSampleRate, fbankLowFreqHz, fbankHighFreqHz),
	}
}

func (f *FBank) NumMels() int { return f.numMels }

func (f *FBank) Compute(audio []float32) ([]float32, error) {
	if len(audio) < f.frameLength {
		return nil, fmt.Errorf("fbank: audio too short (%d < %d)", len(audio), f.frameLength)
	}

	numFrames := 1 + (len(audio)-f.frameLength)/f.frameShift
	out := make([]float32, numFrames*f.numMels)
	frameBuf := make([]float64, f.nFFT)
	power := make([]float32, f.nFFT/2+1)

	for fi := 0; fi < numFrames; fi++ {
		start := fi * f.frameShift

		var prev0 float32
		if start == 0 {
			prev0 = audio[0]
		} else {
			prev0 = audio[start-1]
		}
		for i := range frameBuf {
			frameBuf[i] = 0
		}
		frameBuf[0] = float64(audio[start] - preEmphasis*prev0)
		for i := 1; i < f.frameLength; i++ {
			frameBuf[i] = float64(audio[start+i] - preEmphasis*audio[start+i-1])
		}
		for i := 0; i < f.frameLength; i++ {
			frameBuf[i] *= float64(f.window[i])
		}
		spectrum := realFFT(frameBuf)

		for i, c := range spectrum {
			re, im := real(c), imag(c)
			power[i] = float32(re*re + im*im)
		}

		row := out[fi*f.numMels : (fi+1)*f.numMels]
		for m, taps := range f.melFilters {
			var acc float32
			for _, t := range taps {
				acc += power[t.bin] * t.weight
			}
			if acc < float32(fbankLogFloor) {
				acc = float32(fbankLogFloor)
			}
			row[m] = float32(math.Log(float64(acc)))
		}
	}

	cmnInPlace(out, f.numMels)
	return out, nil
}

func nextPowerOfTwo(n int) int {
	p := 1
	for p < n {
		p <<= 1
	}
	return p
}

func poveyWindow(n int) []float32 {
	denom := float64(n - 1)
	if denom < 1 {
		denom = 1
	}
	out := make([]float32, n)
	for i := 0; i < n; i++ {
		raised := 0.5 - 0.5*math.Cos(2*math.Pi*float64(i)/denom)
		if raised < 0 {
			raised = 0
		}
		out[i] = float32(math.Pow(raised, poveyExponent))
	}
	return out
}

func hzToMel(hz float64) float64 { return melLogScale * math.Log(1.0+hz/melHzScale) }
func melToHz(m float64) float64  { return melHzScale * (math.Exp(m/melLogScale) - 1.0) }

func buildMelFilters(numMels, nFFT int, sampleRate, lowHz, highHz float64) [][]melTap {
	numBins := nFFT/2 + 1
	lowMel, highMel := hzToMel(lowHz), hzToMel(highHz)
	melPoints := make([]float64, numMels+2)
	for i := 0; i < numMels+2; i++ {
		melPoints[i] = melToHz(lowMel + (highMel-lowMel)*float64(i)/float64(numMels+1))
	}
	bins := make([]float64, numMels+2)
	for i, hz := range melPoints {
		bins[i] = hz * float64(nFFT) / sampleRate
	}
	filters := make([][]melTap, numMels)
	for m := 0; m < numMels; m++ {
		left, center, right := bins[m], bins[m+1], bins[m+2]
		lo, hi := int(math.Floor(left)), int(math.Ceil(right))
		var taps []melTap
		for k := lo; k <= hi; k++ {
			if k < 0 || k >= numBins {
				continue
			}
			kf := float64(k)
			var w float64
			switch {
			case kf < center:
				if center > left {
					w = (kf - left) / (center - left)
				}
			case kf <= right:
				if right > center {
					w = (right - kf) / (right - center)
				}
			}
			if w > 0 {
				taps = append(taps, melTap{bin: k, weight: float32(w)})
			}
		}
		filters[m] = taps
	}
	return filters
}

func cmnInPlace(feats []float32, numMels int) {
	if len(feats) == 0 || numMels == 0 {
		return
	}
	numFrames := len(feats) / numMels
	mean := make([]float32, numMels)
	for f := 0; f < numFrames; f++ {
		for m := 0; m < numMels; m++ {
			mean[m] += feats[f*numMels+m]
		}
	}
	inv := 1.0 / float32(numFrames)
	for m := range mean {
		mean[m] *= inv
	}
	for f := 0; f < numFrames; f++ {
		for m := 0; m < numMels; m++ {
			feats[f*numMels+m] -= mean[m]
		}
	}
}

func realFFT(input []float64) []complex128 {
	n := len(input)
	cBuf := make([]complex128, n)
	for i := 0; i < n; i++ {
		cBuf[i] = complex(input[i], 0)
	}
	fft(cBuf)
	half := n/2 + 1
	out := make([]complex128, half)
	copy(out, cBuf[:half])
	return out
}

func fft(a []complex128) {
	n := len(a)
	if n <= 1 {
		return
	}
	j := 0
	for i := 1; i < n; i++ {
		bit := n >> 1
		for ; j&bit != 0; bit >>= 1 {
			j ^= bit
		}
		j ^= bit
		if i < j {
			a[i], a[j] = a[j], a[i]
		}
	}
	for length := 2; length <= n; length <<= 1 {
		half := length >> 1
		angle := -2 * math.Pi / float64(length)
		wn := cmplx.Exp(complex(0, angle))
		for i := 0; i < n; i += length {
			w := complex(1, 0)
			for k := 0; k < half; k++ {
				t := w * a[i+k+half]
				u := a[i+k]
				a[i+k] = u + t
				a[i+k+half] = u - t
				w *= wn
			}
		}
	}
}
