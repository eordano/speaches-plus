package stt

import (
	"math"
	"sync"

	"gonum.org/v1/gonum/dsp/fourier"
)

type MelFilterbank struct {
	NMels   int
	NFFT    int
	Filters [][]float32
}

type fftResources struct {
	fft    *fourier.FFT
	window []float64
}

var (
	fftCacheMu sync.Mutex
	fftCache   = map[int]*fftResources{}
)

func getFFTResources(nFFT int) *fftResources {
	fftCacheMu.Lock()
	defer fftCacheMu.Unlock()
	if r, ok := fftCache[nFFT]; ok {
		return r
	}
	window := make([]float64, nFFT)
	for i := range window {
		window[i] = 0.5 - 0.5*math.Cos(2*math.Pi*float64(i)/float64(nFFT))
	}
	r := &fftResources{
		fft:    fourier.NewFFT(nFFT),
		window: window,
	}
	fftCache[nFFT] = r
	return r
}

func NewMelFilterbank(nMels, nFFT, srHz int) *MelFilterbank {
	const fMin = 0.0
	fMax := float64(srHz) / 2.0

	melMin := hzToMelSlaney(fMin)
	melMax := hzToMelSlaney(fMax)
	melPts := make([]float64, nMels+2)
	for i := range melPts {
		t := float64(i) / float64(nMels+1)
		melPts[i] = melMin + t*(melMax-melMin)
	}
	hzPts := make([]float64, nMels+2)
	for i, m := range melPts {
		hzPts[i] = melToHzSlaney(m)
	}

	nBins := nFFT/2 + 1
	binHz := make([]float64, nBins)
	for i := range binHz {
		binHz[i] = float64(i) * float64(srHz) / float64(nFFT)
	}

	filters := make([][]float32, nMels)
	for m := 0; m < nMels; m++ {
		row := make([]float32, nBins)
		left, center, right := hzPts[m], hzPts[m+1], hzPts[m+2]
		enorm := 2.0 / (right - left)
		for i, hz := range binHz {
			var v float64
			if hz >= left && hz <= center {
				v = (hz - left) / (center - left)
			} else if hz > center && hz <= right {
				v = (right - hz) / (right - center)
			}
			if v > 0 {
				row[i] = float32(v * enorm)
			}
		}
		filters[m] = row
	}
	return &MelFilterbank{NMels: nMels, NFFT: nFFT, Filters: filters}
}

func hzToMelSlaney(f float64) float64 {
	const fSp = 200.0 / 3.0
	const minLogHz = 1000.0
	minLogMel := minLogHz / fSp
	logStep := math.Log(6.4) / 27.0
	if f < minLogHz {
		return f / fSp
	}
	return minLogMel + math.Log(f/minLogHz)/logStep
}

func melToHzSlaney(m float64) float64 {
	const fSp = 200.0 / 3.0
	const minLogHz = 1000.0
	minLogMel := minLogHz / fSp
	logStep := math.Log(6.4) / 27.0
	if m < minLogMel {
		return m * fSp
	}
	return minLogHz * math.Exp(logStep*(m-minLogMel))
}

func LogMelSpectrogram(audio []float32, fb *MelFilterbank) []float32 {
	padded := make([]float32, whisperPadSamples)
	copy(padded, audio)

	const halfFFT = whisperNFFT / 2
	reflected := make([]float32, halfFFT+len(padded)+halfFFT)
	for i := 0; i < halfFFT; i++ {
		reflected[i] = padded[halfFFT-i]
	}
	copy(reflected[halfFFT:halfFFT+len(padded)], padded)
	for i := 0; i < halfFFT; i++ {
		reflected[halfFFT+len(padded)+i] = padded[len(padded)-2-i]
	}

	res := getFFTResources(whisperNFFT)
	window := res.window
	fft := res.fft

	nBins := whisperNFFT/2 + 1
	nFrames := whisperNbFrames

	frame := make([]float64, whisperNFFT)
	power := make([]float32, nBins)

	mel := make([]float32, fb.NMels*nFrames)

	for fIdx := 0; fIdx < nFrames; fIdx++ {
		off := fIdx * whisperHopLength
		for i := 0; i < whisperNFFT; i++ {
			frame[i] = float64(reflected[off+i]) * window[i]
		}
		spec := fft.Coefficients(nil, frame)
		for i := 0; i < nBins; i++ {
			re := real(spec[i])
			im := imag(spec[i])
			power[i] = float32(re*re + im*im)
		}
		for m := 0; m < fb.NMels; m++ {
			row := fb.Filters[m]
			var sum float32
			for i := 0; i < nBins; i++ {
				sum += row[i] * power[i]
			}
			mel[m*nFrames+fIdx] = sum
		}
	}

	const minPower = 1e-10
	for i, v := range mel {
		if v < minPower {
			v = minPower
		}
		mel[i] = float32(math.Log10(float64(v)))
	}

	var maxLog float32 = -math.MaxFloat32
	for _, v := range mel {
		if v > maxLog {
			maxLog = v
		}
	}
	floor := maxLog - 8.0
	for i, v := range mel {
		if v < floor {
			v = floor
		}
		mel[i] = (v + 4.0) / 4.0
	}

	return mel
}
