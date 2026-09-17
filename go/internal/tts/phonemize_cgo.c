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

static const char* CLAUSE_PUNCTUATION[] = { ".", ",", ";", ":", "!", "?", "\xE2\x80\xA6", NULL };

static int clause_skippable(char c) {
    return c == ' ' || c == '\n' || c == '\t' || c == '\r' || c == '"' || c == '\'' || c == ')' || c == ']';
}

static const char* terminator_ending_at(const char* start, const char* p) {
    for (int i = 0; CLAUSE_PUNCTUATION[i]; i++) {
        int n = (int) strlen(CLAUSE_PUNCTUATION[i]);
        if (p - start >= n && memcmp(p - n, CLAUSE_PUNCTUATION[i], n) == 0) return CLAUSE_PUNCTUATION[i];
    }
    return NULL;
}

static const char* clause_terminator(const char* start, const char* end) {
    const char* p = end;
    for (int pass = 0; pass < 2; pass++) {
        while (p > start && clause_skippable(p[-1])) p--;
        if (p == start) return NULL;
        const char* t = terminator_ending_at(start, p);
        if (t) return t;
        p--;
        while (p > start && (*p & 0xC0) == 0x80) p--;
    }
    return NULL;
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
        const char* clause_start = (const char*) ptr;
        const char* phon = espeak_TextToPhonemes(&ptr, textmode, phoneme_mode);
        if (phon == NULL) break;
        const char* clause_end = ptr ? (const char*) ptr : clause_start + strlen(clause_start);
        const char* punct = clause_terminator(clause_start, clause_end);
        int plen = (int) strlen(phon);
        int qlen = punct ? (int) strlen(punct) : 0;
        if (plen == 0 && qlen == 0) continue;
        if (written > 0) {
            if (written + 2 >= cap) { *out_size = written + plen + qlen + 2; return -2; }
            out[written++] = ' ';
            out[written] = '\0';
        }
        if (written + plen + qlen + 1 >= cap) {
            *out_size = written + plen + qlen + 1;
            return -2;
        }
        memcpy(out + written, phon, plen);
        written += plen;
        if (qlen) { memcpy(out + written, punct, qlen); written += qlen; }
        out[written] = '\0';
    }
    *out_size = written;
    return 0;
}

void sp_espeak_terminate(void) {
    espeak_Terminate();
}
