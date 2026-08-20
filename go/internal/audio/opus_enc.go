package audio

/*
#cgo CFLAGS: -O2
#cgo LDFLAGS: -lopus

#include <stdlib.h>
#include <opus/opus.h>

extern OpusEncoder* opus_encoder_create(opus_int32 Fs, int channels, int application, int *error);
extern opus_int32   opus_encode_float(OpusEncoder *st, const float *pcm, int frame_size,
                                      unsigned char *data, opus_int32 max_data_bytes);
extern void         opus_encoder_destroy(OpusEncoder *st);
*/
import "C"

import (
	"errors"
	"fmt"
	"unsafe"
)

type OpusEncoder struct {
	enc        *C.OpusEncoder
	sampleRate int
}

func NewOpusEncoder(sampleRate int) (*OpusEncoder, error) {
	var errc C.int
	enc := C.opus_encoder_create(C.opus_int32(sampleRate), 1, C.OPUS_APPLICATION_VOIP, &errc)
	if enc == nil || errc != 0 {
		return nil, fmt.Errorf("opus_encoder_create failed: errc=%d", int(errc))
	}
	return &OpusEncoder{enc: enc, sampleRate: sampleRate}, nil
}

func (e *OpusEncoder) EncodeFrame(pcm MonoF32, out []byte) ([]byte, error) {
	if e == nil || e.enc == nil {
		return nil, errors.New("opus encoder not initialized")
	}
	if len(pcm) == 0 || len(out) == 0 {
		return nil, errors.New("opus: empty pcm or out buffer")
	}
	n := C.opus_encode_float(
		e.enc,
		(*C.float)(unsafe.Pointer(&pcm[0])),
		C.int(len(pcm)),
		(*C.uchar)(unsafe.Pointer(&out[0])),
		C.opus_int32(len(out)),
	)
	if n < 0 {
		return nil, fmt.Errorf("opus_encode_float failed: %d", int(n))
	}
	return out[:int(n)], nil
}

func (e *OpusEncoder) Close() {
	if e != nil && e.enc != nil {
		C.opus_encoder_destroy(e.enc)
		e.enc = nil
	}
}
