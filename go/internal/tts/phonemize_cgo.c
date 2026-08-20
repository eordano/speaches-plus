#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include "espeak-ng/speak_lib.h"

int sp_espeak_init(const char* data_path) {
    int rc = espeak_Initialize(AUDIO_OUTPUT_RETRIEVAL, 0, data_path, 0);
    if (rc < 0) return rc;
    return rc;
}

int sp_espeak_set_voice(const char* name) {
    return (int) espeak_SetVoiceByName(name);
}

int sp_espeak_text_to_ipa(const char* text, char* out, int* out_size) {
    if (!text || !out || !out_size) return -1;
    int cap = *out_size;
    if (cap < 1) return -1;
    out[0] = '\0';

    const void* ptr = (const void*) text;
    int written = 0;
    int phoneme_mode = 0x02;
    int textmode = 1;

    while (ptr != NULL) {
        const char* phon = espeak_TextToPhonemes(&ptr, textmode, phoneme_mode);
        if (phon == NULL) break;
        int plen = (int) strlen(phon);
        if (plen == 0) continue;
        if (written > 0) {
            if (written + 1 + 1 >= cap) {
                *out_size = written + plen + 2;
                return -2;
            }
            out[written++] = ' ';
            out[written] = '\0';
        }
        if (written + plen + 1 >= cap) {
            *out_size = written + plen + 1;
            return -2;
        }
        memcpy(out + written, phon, plen);
        written += plen;
        out[written] = '\0';
    }
    *out_size = written;
    return 0;
}

void sp_espeak_terminate(void) {
    espeak_Terminate();
}
