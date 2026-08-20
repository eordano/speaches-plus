package stt

/*
#cgo CXXFLAGS: -O2 -std=c++17
#cgo darwin LDFLAGS: -lctranslate2 -lc++
#cgo linux  LDFLAGS: -lctranslate2 -lstdc++

#include <stdlib.h>

typedef struct sp_ct2_whisper sp_ct2_whisper;

extern sp_ct2_whisper* sp_ct2_open(const char* model_path, const char* device, const char* compute_type);
extern void              sp_ct2_close(sp_ct2_whisper* h);
extern int               sp_ct2_n_mels(sp_ct2_whisper* h);
extern int               sp_ct2_generate(sp_ct2_whisper* h,
                                         const float* mel,
                                         int n_mels,
                                         int n_frames,
                                         const char* language_token,
                                         int beam_size,
                                         char* out,
                                         int* out_size,
                                         float* no_speech_prob_out,
                                         float* avg_logprob_out);
extern int               sp_ct2_generate_segmented(sp_ct2_whisper* h,
                                                   const float* mel,
                                                   int n_mels,
                                                   int n_frames,
                                                   const char* language_token,
                                                   int beam_size,
                                                   char* out,
                                                   int* out_size,
                                                   char* tokens_out,
                                                   int* tokens_out_size,
                                                   float* no_speech_prob_out,
                                                   float* avg_logprob_out);
*/
import "C"

import (
	"errors"
	"fmt"
	"log/slog"
	"os"
	"runtime"
	"strconv"
	"strings"
	"sync"
	"unsafe"
)

type CT2 struct {
	mu                    sync.Mutex
	handle                *C.sp_ct2_whisper
	filterbank            *MelFilterbank
	beamSize              int
	languageTok           string
	noSpeechProbThreshold float32
}

type CT2Config struct {
	ModelDir              string
	Device                string
	ComputeType           string
	Language              string
	BeamSize              int
	NoSpeechProbThreshold float32
}

func NewCT2(cfg CT2Config) (Transcriber, error) {
	if cfg.ModelDir == "" {
		return nil, errors.New("ct2: ModelDir is empty")
	}
	if _, err := os.Stat(cfg.ModelDir); err != nil {
		return nil, fmt.Errorf("ct2: model dir not accessible: %w", err)
	}
	device := cfg.Device
	if device == "" {
		device = os.Getenv("CT2_DEVICE")
	}
	if device == "" {
		device = "cpu"
	}
	cType := cfg.ComputeType
	if cType == "" {
		cType = os.Getenv("CT2_COMPUTE_TYPE")
	}
	if cType == "" {
		cType = "default"
	}
	beam := cfg.BeamSize
	if beam <= 0 {
		beam = 5
	}
	lang := cfg.Language
	if lang == "" {
		lang = "en"
	}

	cPath := C.CString(cfg.ModelDir)
	defer C.free(unsafe.Pointer(cPath))
	cDevice := C.CString(device)
	defer C.free(unsafe.Pointer(cDevice))
	cCType := C.CString(cType)
	defer C.free(unsafe.Pointer(cCType))

	h := C.sp_ct2_open(cPath, cDevice, cCType)
	if h == nil {
		return nil, fmt.Errorf("ct2: open(%q, %q, %q) failed; check stderr for cause",
			cfg.ModelDir, device, cType)
	}
	nMels := int(C.sp_ct2_n_mels(h))
	if nMels <= 0 {
		C.sp_ct2_close(h)
		return nil, fmt.Errorf("ct2: model reports n_mels=%d", nMels)
	}
	thresh := cfg.NoSpeechProbThreshold
	if thresh == 0 {
		thresh = 0.6
	}
	return &CT2{
		handle:                h,
		filterbank:            NewMelFilterbank(nMels, whisperNFFT, whisperSamplingHz),
		beamSize:              beam,
		languageTok:           "<|" + lang + "|>",
		noSpeechProbThreshold: thresh,
	}, nil
}

func (c *CT2) Transcribe(samples []float32, sampleRate int) (string, error) {
	if sampleRate != 16000 {
		return "", fmt.Errorf("ct2: expected 16 kHz, got %d", sampleRate)
	}
	if c.handle == nil {
		return "", errors.New("ct2: handle closed")
	}
	if len(samples) == 0 {
		return "", nil
	}

	const silencePeak = 0.005
	var peak float32
	for _, s := range samples {
		if s < 0 {
			s = -s
		}
		if s > peak {
			peak = s
		}
	}
	if peak < silencePeak {
		slog.Debug("ct2 silence pre-gate fired", "peak", peak)
		return "", nil
	}

	mel := LogMelSpectrogram(samples, c.filterbank)
	nMels := c.filterbank.NMels
	nFrames := len(mel) / nMels

	c.mu.Lock()
	defer c.mu.Unlock()

	cLang := C.CString(c.languageTok)
	defer C.free(unsafe.Pointer(cLang))

	var pinner runtime.Pinner
	pinner.Pin(&mel[0])
	defer pinner.Unpin()

	var noSpeechProb C.float
	bufSize := 4096
	for {
		out := make([]byte, bufSize)
		size := C.int(bufSize)
		pinner.Pin(&out[0])
		rc := C.sp_ct2_generate(
			c.handle,
			(*C.float)(unsafe.Pointer(&mel[0])),
			C.int(nMels),
			C.int(nFrames),
			cLang,
			C.int(c.beamSize),
			(*C.char)(unsafe.Pointer(&out[0])),
			&size,
			&noSpeechProb,
			nil,
		)
		switch rc {
		case 0:
			text := DecodeBPE(string(out[:int(size)]))
			if float32(noSpeechProb) >= c.noSpeechProbThreshold {
				slog.Debug("ct2 no_speech gate fired",
					"prob", float32(noSpeechProb),
					"threshold", c.noSpeechProbThreshold,
					"discarded_text", text,
				)
				return "", nil
			}
			slog.Debug("ct2 transcribe ok",
				"prob", float32(noSpeechProb),
				"text", text,
			)
			return text, nil
		case -2:
			bufSize = int(size) + 1
			continue
		default:
			return "", fmt.Errorf("ct2: generate failed (rc=%d)", int(rc))
		}
	}
}

func (c *CT2) Close() error {
	c.mu.Lock()
	defer c.mu.Unlock()
	if c.handle != nil {
		C.sp_ct2_close(c.handle)
		c.handle = nil
	}
	return nil
}

func (c *CT2) TranscribeFull(samples []float32, sampleRate int) (Result, error) {
	if sampleRate != 16000 {
		return Result{}, fmt.Errorf("ct2: expected 16 kHz, got %d", sampleRate)
	}
	if c.handle == nil {
		return Result{}, errors.New("ct2: handle closed")
	}
	if len(samples) == 0 {
		return Result{}, nil
	}

	const silencePeak = 0.005
	var peak float32
	for _, s := range samples {
		if s < 0 {
			s = -s
		}
		if s > peak {
			peak = s
		}
	}
	if peak < silencePeak {
		slog.Debug("ct2 silence pre-gate fired", "peak", peak)
		return Result{}, nil
	}

	mel := LogMelSpectrogram(samples, c.filterbank)
	nMels := c.filterbank.NMels
	nFrames := len(mel) / nMels

	c.mu.Lock()
	defer c.mu.Unlock()

	cLang := C.CString(c.languageTok)
	defer C.free(unsafe.Pointer(cLang))

	var pinner runtime.Pinner
	pinner.Pin(&mel[0])
	defer pinner.Unpin()

	var noSpeechProb C.float
	var avgLogprob C.float
	bufSize := 4096
	for {
		out := make([]byte, bufSize)
		size := C.int(bufSize)
		pinner.Pin(&out[0])
		rc := C.sp_ct2_generate(
			c.handle,
			(*C.float)(unsafe.Pointer(&mel[0])),
			C.int(nMels),
			C.int(nFrames),
			cLang,
			C.int(c.beamSize),
			(*C.char)(unsafe.Pointer(&out[0])),
			&size,
			&noSpeechProb,
			&avgLogprob,
		)
		switch rc {
		case 0:
			text := DecodeBPE(string(out[:int(size)]))
			nsp := float32(noSpeechProb)
			res := Result{Text: text, NoSpeechProb: &nsp}
			if lp := float32(avgLogprob); !isNaN32(lp) {
				res.AvgLogprob = &lp
			}
			return res, nil
		case -2:
			bufSize = int(size) + 1
			continue
		default:
			return Result{}, fmt.Errorf("ct2: generate failed (rc=%d)", int(rc))
		}
	}
}

func isNaN32(f float32) bool { return f != f }

func (c *CT2) TranscribeSegments(samples []float32, sampleRate int) (Result, error) {
	if sampleRate != 16000 {
		return Result{}, fmt.Errorf("ct2: expected 16 kHz, got %d", sampleRate)
	}
	if c.handle == nil {
		return Result{}, errors.New("ct2: handle closed")
	}
	if len(samples) == 0 {
		return Result{}, nil
	}

	const silencePeak = 0.005
	var peak float32
	for _, s := range samples {
		if s < 0 {
			s = -s
		}
		if s > peak {
			peak = s
		}
	}
	if peak < silencePeak {
		slog.Debug("ct2 silence pre-gate fired", "peak", peak)
		return Result{}, nil
	}

	mel := LogMelSpectrogram(samples, c.filterbank)
	nMels := c.filterbank.NMels
	nFrames := len(mel) / nMels

	c.mu.Lock()
	defer c.mu.Unlock()

	cLang := C.CString(c.languageTok)
	defer C.free(unsafe.Pointer(cLang))

	var pinner runtime.Pinner
	pinner.Pin(&mel[0])
	defer pinner.Unpin()

	var noSpeechProb C.float
	var avgLogprob C.float
	textCap := 4096
	tokCap := 16384
	for {
		out := make([]byte, textCap)
		toks := make([]byte, tokCap)
		outSize := C.int(textCap)
		toksSize := C.int(tokCap)
		pinner.Pin(&out[0])
		pinner.Pin(&toks[0])
		rc := C.sp_ct2_generate_segmented(
			c.handle,
			(*C.float)(unsafe.Pointer(&mel[0])),
			C.int(nMels),
			C.int(nFrames),
			cLang,
			C.int(c.beamSize),
			(*C.char)(unsafe.Pointer(&out[0])),
			&outSize,
			(*C.char)(unsafe.Pointer(&toks[0])),
			&toksSize,
			&noSpeechProb,
			&avgLogprob,
		)
		switch rc {
		case 0:
			text := DecodeBPE(string(out[:int(outSize)]))
			segs := parseCT2SegmentsFromTokens(toks[:int(toksSize)])
			res := Result{Text: text, Segments: segs}
			nsp := float32(noSpeechProb)
			res.NoSpeechProb = &nsp
			if lp := float32(avgLogprob); !isNaN32(lp) {
				res.AvgLogprob = &lp
			}
			return res, nil
		case -2:
			if int(outSize) > textCap {
				textCap = int(outSize) + 1
			}
			if int(toksSize) > tokCap {
				tokCap = int(toksSize) + 1
			}
			continue
		default:
			return Result{}, fmt.Errorf("ct2: generate_segmented failed (rc=%d)", int(rc))
		}
	}
}

func parseCT2SegmentsFromTokens(blob []byte) []Segment {
	if len(blob) == 0 {
		return nil
	}
	tokens := strings.Split(strings.TrimRight(string(blob), "\n"), "\n")
	var segs []Segment
	var curStart int32 = -1
	var curTokens []string
	flush := func(endMs uint32) {
		if curStart < 0 {
			return
		}
		text := DecodeBPE(strings.Join(curTokens, ""))
		text = strings.TrimSpace(text)
		if text != "" {
			segs = append(segs, Segment{
				TStartMs: uint32(curStart),
				TEndMs:   endMs,
				Text:     text,
			})
		}
		curTokens = curTokens[:0]
	}
	for _, tok := range tokens {
		if ms, ok := parseTimestampToken(tok); ok {
			if curStart < 0 {
				curStart = int32(ms)
			} else {
				flush(ms)
				curStart = int32(ms)
			}
			continue
		}
		if curStart < 0 {
			continue
		}
		curTokens = append(curTokens, tok)
	}
	return segs
}

func parseTimestampToken(tok string) (uint32, bool) {
	if len(tok) < 5 || tok[0] != '<' || tok[1] != '|' ||
		tok[len(tok)-2] != '|' || tok[len(tok)-1] != '>' {
		return 0, false
	}
	inner := tok[2 : len(tok)-2]
	dot := strings.IndexByte(inner, '.')
	if dot < 1 || dot >= len(inner)-1 {
		return 0, false
	}
	whole := inner[:dot]
	frac := inner[dot+1:]
	for _, c := range whole {
		if c < '0' || c > '9' {
			return 0, false
		}
	}
	for _, c := range frac {
		if c < '0' || c > '9' {
			return 0, false
		}
	}
	secs, err := strconv.ParseUint(whole, 10, 32)
	if err != nil {
		return 0, false
	}
	fracVal, err := strconv.ParseUint(frac, 10, 32)
	if err != nil {
		return 0, false
	}
	var fracMs uint64
	switch len(frac) {
	case 1:
		fracMs = fracVal * 100
	case 2:
		fracMs = fracVal * 10
	case 3:
		fracMs = fracVal
	default:

		div := uint64(1)
		for i := 0; i < len(frac)-3; i++ {
			div *= 10
		}
		fracMs = fracVal / div
	}
	return uint32(secs*1000 + fracMs), true
}
