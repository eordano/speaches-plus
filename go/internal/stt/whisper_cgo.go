package stt

/*
#cgo CFLAGS: -O2
#cgo LDFLAGS: -lwhisper

#include <stdlib.h>
#include "whisper.h"

extern struct whisper_context* sp_whisper_init(const char* path);
extern void sp_whisper_free(struct whisper_context* ctx);
extern int sp_whisper_transcribe(
    struct whisper_context* ctx,
    const float* samples,
    int n_samples,
    char* out,
    int* out_size
);
extern int sp_whisper_transcribe_full(
    struct whisper_context* ctx,
    const float* samples,
    int n_samples,
    char* out,
    int* out_size,
    float* avg_logprob_out,
    float* no_speech_prob_out
);
extern int sp_whisper_transcribe_segmented(
    struct whisper_context* ctx,
    const float* samples,
    int n_samples,
    char* out,
    int* out_size,
    char* segments_out,
    int* segments_out_size,
    float* avg_logprob_out,
    float* no_speech_prob_out
);
*/
import "C"

import (
	"errors"
	"fmt"
	"runtime"
	"strconv"
	"strings"
	"sync"
	"unsafe"
)

type Whisper struct {
	ctx *C.struct_whisper_context
	mu  sync.Mutex
}

func NewWhisper(modelPath string) (Transcriber, error) {
	if modelPath == "" {
		return nil, errors.New("whisper: model path is empty")
	}
	cPath := C.CString(modelPath)
	defer C.free(unsafe.Pointer(cPath))
	ctx := C.sp_whisper_init(cPath)
	if ctx == nil {
		return nil, fmt.Errorf("whisper: load failed for %q", modelPath)
	}
	return &Whisper{ctx: ctx}, nil
}

func (w *Whisper) Transcribe(samples []float32, sampleRate int) (string, error) {
	if sampleRate != 16000 {
		return "", fmt.Errorf("whisper: expected 16 kHz, got %d", sampleRate)
	}
	if len(samples) == 0 {
		return "", nil
	}

	w.mu.Lock()
	defer w.mu.Unlock()

	var pinner runtime.Pinner
	pinner.Pin(&samples[0])
	defer pinner.Unpin()

	bufSize := 65536
	for {
		out := make([]byte, bufSize)
		size := C.int(bufSize)
		pinner.Pin(&out[0])
		rc := C.sp_whisper_transcribe(
			w.ctx,
			(*C.float)(unsafe.Pointer(&samples[0])),
			C.int(len(samples)),
			(*C.char)(unsafe.Pointer(&out[0])),
			&size,
		)
		switch rc {
		case 0:
			return string(out[:int(size)]), nil
		case -2:
			bufSize = int(size) + 1
			continue
		default:
			return "", fmt.Errorf("whisper_full failed: %d", int(rc))
		}
	}
}

func (w *Whisper) Close() error {
	if w.ctx != nil {
		C.sp_whisper_free(w.ctx)
		w.ctx = nil
	}
	return nil
}

func (w *Whisper) TranscribeFull(samples []float32, sampleRate int) (Result, error) {
	if sampleRate != 16000 {
		return Result{}, fmt.Errorf("whisper: expected 16 kHz, got %d", sampleRate)
	}
	if len(samples) == 0 {
		return Result{}, nil
	}

	w.mu.Lock()
	defer w.mu.Unlock()

	var pinner runtime.Pinner
	pinner.Pin(&samples[0])
	defer pinner.Unpin()

	var avgLogprob C.float
	var noSpeechProb C.float
	bufSize := 65536
	for {
		out := make([]byte, bufSize)
		size := C.int(bufSize)
		pinner.Pin(&out[0])
		rc := C.sp_whisper_transcribe_full(
			w.ctx,
			(*C.float)(unsafe.Pointer(&samples[0])),
			C.int(len(samples)),
			(*C.char)(unsafe.Pointer(&out[0])),
			&size,
			&avgLogprob,
			&noSpeechProb,
		)
		switch rc {
		case 0:
			text := string(out[:int(size)])
			res := Result{Text: text}
			if lp := float32(avgLogprob); !isNaN32(lp) {
				res.AvgLogprob = &lp
			}
			if nsp := float32(noSpeechProb); !isNaN32(nsp) {
				res.NoSpeechProb = &nsp
			}
			return res, nil
		case -2:
			bufSize = int(size) + 1
			continue
		default:
			return Result{}, fmt.Errorf("whisper_full failed: %d", int(rc))
		}
	}
}

func (w *Whisper) TranscribeSegments(samples []float32, sampleRate int) (Result, error) {
	if sampleRate != 16000 {
		return Result{}, fmt.Errorf("whisper: expected 16 kHz, got %d", sampleRate)
	}
	if len(samples) == 0 {
		return Result{}, nil
	}

	w.mu.Lock()
	defer w.mu.Unlock()

	var pinner runtime.Pinner
	pinner.Pin(&samples[0])
	defer pinner.Unpin()

	var avgLogprob C.float
	var noSpeechProb C.float
	textCap := 65536
	segCap := 65536
	for {
		out := make([]byte, textCap)
		segOut := make([]byte, segCap)
		outSize := C.int(textCap)
		segSize := C.int(segCap)
		pinner.Pin(&out[0])
		pinner.Pin(&segOut[0])
		rc := C.sp_whisper_transcribe_segmented(
			w.ctx,
			(*C.float)(unsafe.Pointer(&samples[0])),
			C.int(len(samples)),
			(*C.char)(unsafe.Pointer(&out[0])),
			&outSize,
			(*C.char)(unsafe.Pointer(&segOut[0])),
			&segSize,
			&avgLogprob,
			&noSpeechProb,
		)
		switch rc {
		case 0:
			text := string(out[:int(outSize)])
			segs, err := parseWhisperSegmentBlob(segOut[:int(segSize)])
			if err != nil {
				return Result{}, fmt.Errorf("whisper_segmented: parse: %w", err)
			}
			res := Result{Text: text, Segments: segs}
			if lp := float32(avgLogprob); !isNaN32(lp) {
				res.AvgLogprob = &lp
			}
			if nsp := float32(noSpeechProb); !isNaN32(nsp) {
				res.NoSpeechProb = &nsp
			}
			return res, nil
		case -2:

			if int(outSize) > textCap {
				textCap = int(outSize) + 1
			}
			if int(segSize) > segCap {
				segCap = int(segSize) + 1
			}
			continue
		default:
			return Result{}, fmt.Errorf("whisper_segmented failed: %d", int(rc))
		}
	}
}

func parseWhisperSegmentBlob(blob []byte) ([]Segment, error) {
	if len(blob) == 0 {
		return nil, nil
	}
	lines := strings.Split(strings.TrimRight(string(blob), "\n"), "\n")
	segs := make([]Segment, 0, len(lines))
	for i, line := range lines {
		if line == "" {
			continue
		}
		parts := strings.SplitN(line, "\t", 5)
		if len(parts) != 5 {
			return nil, fmt.Errorf("segment %d: expected 5 tab-separated fields, got %d", i, len(parts))
		}
		t0, err := strconv.ParseUint(parts[0], 10, 32)
		if err != nil {
			return nil, fmt.Errorf("segment %d: t0: %w", i, err)
		}
		t1, err := strconv.ParseUint(parts[1], 10, 32)
		if err != nil {
			return nil, fmt.Errorf("segment %d: t1: %w", i, err)
		}
		lp64, err := strconv.ParseFloat(parts[2], 32)
		if err != nil {
			return nil, fmt.Errorf("segment %d: avg_lp: %w", i, err)
		}
		nsp64, err := strconv.ParseFloat(parts[3], 32)
		if err != nil {
			return nil, fmt.Errorf("segment %d: nsp: %w", i, err)
		}
		lp := float32(lp64)
		nsp := float32(nsp64)
		seg := Segment{
			TStartMs:     uint32(t0),
			TEndMs:       uint32(t1),
			Text:         strings.TrimSpace(parts[4]),
			AvgLogprob:   &lp,
			NoSpeechProb: &nsp,
		}
		segs = append(segs, seg)
	}
	return segs, nil
}
