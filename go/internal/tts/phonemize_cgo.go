package tts

/*
#cgo CFLAGS: -O2
#cgo LDFLAGS: -lespeak-ng

#include <stdlib.h>
#include "espeak-ng/speak_lib.h"

extern int  sp_espeak_init(const char* data_path);
extern int  sp_espeak_set_voice(const char* name);
extern int  sp_espeak_text_to_ipa(const char* text, char* out, int* out_size);
extern void sp_espeak_terminate(void);
*/
import "C"

import (
	"errors"
	"fmt"
	"sync"
	"unsafe"
)

type Phonemizer struct {
	mu     sync.Mutex
	voice  string
	inited bool
}

var globalPhon = &Phonemizer{}

func initPhonemizer(dataPath string) error {
	globalPhon.mu.Lock()
	defer globalPhon.mu.Unlock()
	if globalPhon.inited {
		return nil
	}
	cPath := (*C.char)(nil)
	if dataPath != "" {
		cPath = C.CString(dataPath)
		defer C.free(unsafe.Pointer(cPath))
	}
	rc := C.sp_espeak_init(cPath)
	if rc < 0 {
		return fmt.Errorf("espeak_Initialize failed: %d", int(rc))
	}
	globalPhon.inited = true
	return nil
}

func (p *Phonemizer) Phonemize(text, lang string) (string, error) {
	p.mu.Lock()
	defer p.mu.Unlock()
	if !p.inited {
		return "", errors.New("phonemizer not initialised")
	}
	if lang != p.voice {
		cName := C.CString(lang)
		rc := C.sp_espeak_set_voice(cName)
		C.free(unsafe.Pointer(cName))
		if rc != 0 {
			return "", fmt.Errorf("espeak_SetVoiceByName(%q) failed: %d", lang, int(rc))
		}
		p.voice = lang
	}

	cText := C.CString(text)
	defer C.free(unsafe.Pointer(cText))

	bufSize := 4096
	for {
		out := make([]byte, bufSize)
		size := C.int(bufSize)
		rc := C.sp_espeak_text_to_ipa(cText,
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
			return "", fmt.Errorf("text_to_ipa failed: %d", int(rc))
		}
	}
}

func closePhonemizer() {
	globalPhon.mu.Lock()
	defer globalPhon.mu.Unlock()
	if globalPhon.inited {
		C.sp_espeak_terminate()
		globalPhon.inited = false
	}
}
