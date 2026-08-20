#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <stdbool.h>
#include <math.h>
#include "whisper.h"

struct whisper_context* sp_whisper_init(const char* path) {
    struct whisper_context_params cparams = whisper_context_default_params();
    return whisper_init_from_file_with_params(path, cparams);
}

void sp_whisper_free(struct whisper_context* ctx) {
    if (ctx != NULL) {
        whisper_free(ctx);
    }
}

// Internal helper used by both sp_whisper_transcribe (text-only) and
// sp_whisper_transcribe_full (text + per-utterance stats). avg_logprob_out /
// no_speech_prob_out may be NULL -- when set, both are filled with NaN if the
// stat could not be computed (e.g. zero non-special tokens, zero segments).
static int sp_whisper_run(
    struct whisper_context* ctx,
    const float* samples,
    int n_samples,
    char* out,
    int* out_size,
    float* avg_logprob_out,
    float* no_speech_prob_out
) {
    if (ctx == NULL || samples == NULL || out == NULL || out_size == NULL) {
        return -1;
    }

    struct whisper_full_params wparams = whisper_full_default_params(WHISPER_SAMPLING_GREEDY);
    wparams.print_realtime    = false;
    wparams.print_progress    = false;
    wparams.print_timestamps  = false;
    wparams.print_special     = false;
    wparams.translate         = false;
    wparams.language          = "en";
    wparams.n_threads         = 4;
    wparams.suppress_blank    = true;
    wparams.no_context        = true;
    wparams.single_segment    = false;

    int rc = whisper_full(ctx, wparams, samples, n_samples);
    if (rc != 0) {
        return rc;
    }

    int n_segments = whisper_full_n_segments(ctx);
    int written = 0;
    int cap = *out_size;
    if (cap < 1) {
        return -1;
    }
    out[0] = '\0';

    // Stats accumulators -- only used if the caller asked for them.
    double log_sum = 0.0;
    long tok_count = 0;
    double nsp_sum = 0.0;
    long nsp_count = 0;

    for (int i = 0; i < n_segments; i++) {
        const char* text = whisper_full_get_segment_text(ctx, i);
        if (text != NULL) {
            int tlen = (int) strlen(text);
            if (written + tlen + 1 >= cap) {
                *out_size = written + tlen + 1;
                return -2;
            }
            memcpy(out + written, text, tlen);
            written += tlen;
            out[written] = '\0';
        }

        if (avg_logprob_out != NULL || no_speech_prob_out != NULL) {
            nsp_sum += whisper_full_get_segment_no_speech_prob(ctx, i);
            nsp_count++;

            int n_tokens = whisper_full_n_tokens(ctx, i);
            for (int t = 0; t < n_tokens; t++) {
                const char* tok_text = whisper_full_get_token_text(ctx, i, t);
                // Skip whisper specials (`<|sot|>`, `<|en|>`, `<|notimestamps|>`,
                // timestamps `<|N.NN|>`, `<|eot|>`, ...) so the metric matches
                // faster-whisper's segment.avg_logprob, which is computed over
                // text tokens only.
                if (tok_text != NULL) {
                    int tlen = (int) strlen(tok_text);
                    if (tlen >= 2 && tok_text[0] == '<' && tok_text[tlen - 1] == '>') {
                        continue;
                    }
                }
                float p = whisper_full_get_token_p(ctx, i, t);
                if (!(p > 0.0f)) {
                    p = 1.175494e-38f; // ~f32::MIN_POSITIVE -- keeps log finite.
                }
                if (p > 1.0f) p = 1.0f;
                log_sum += log((double) p);
                tok_count++;
            }
        }
    }
    *out_size = written;

    if (avg_logprob_out != NULL) {
        *avg_logprob_out = (tok_count > 0)
            ? (float)(log_sum / (double) tok_count)
            : NAN;
    }
    if (no_speech_prob_out != NULL) {
        *no_speech_prob_out = (nsp_count > 0)
            ? (float)(nsp_sum / (double) nsp_count)
            : NAN;
    }
    return 0;
}

int sp_whisper_transcribe(
    struct whisper_context* ctx,
    const float* samples,
    int n_samples,
    char* out,
    int* out_size
) {
    return sp_whisper_run(ctx, samples, n_samples, out, out_size, NULL, NULL);
}

int sp_whisper_transcribe_full(
    struct whisper_context* ctx,
    const float* samples,
    int n_samples,
    char* out,
    int* out_size,
    float* avg_logprob_out,
    float* no_speech_prob_out
) {
    return sp_whisper_run(ctx, samples, n_samples, out, out_size,
                          avg_logprob_out, no_speech_prob_out);
}

// Per-segment emission, used by the diarized-transcription endpoint to get
// Whisper's own segment timings + text in a single decode pass.
//
// `segments_out` is filled with one record per segment:
//   "<t0_ms>\t<t1_ms>\t<avg_logprob>\t<no_speech_prob>\t<text>\n"
//
// Tabs/newlines in segment text would break the framing, so we replace them
// with spaces before writing. Whisper outputs natural language so this is
// effectively a no-op in practice.
//
// Returns 0 on success, -2 if either output buffer needs to grow (caller
// retries with the requested size in `*out_size` / `*segments_out_size`),
// other negative values on error.
int sp_whisper_transcribe_segmented(
    struct whisper_context* ctx,
    const float* samples,
    int n_samples,
    char* out, int* out_size,
    char* segments_out, int* segments_out_size,
    float* avg_logprob_out,
    float* no_speech_prob_out
) {
    if (ctx == NULL || samples == NULL || out == NULL || out_size == NULL ||
        segments_out == NULL || segments_out_size == NULL) {
        return -1;
    }

    struct whisper_full_params wparams = whisper_full_default_params(WHISPER_SAMPLING_GREEDY);
    wparams.print_realtime    = false;
    wparams.print_progress    = false;
    wparams.print_timestamps  = false;
    wparams.print_special     = false;
    wparams.translate         = false;
    wparams.language          = "en";
    wparams.n_threads         = 4;
    wparams.suppress_blank    = true;
    wparams.no_context        = true;
    wparams.single_segment    = false;

    int rc = whisper_full(ctx, wparams, samples, n_samples);
    if (rc != 0) {
        return rc;
    }

    int n_segments = whisper_full_n_segments(ctx);
    int text_cap = *out_size;
    int seg_cap = *segments_out_size;
    if (text_cap < 1 || seg_cap < 1) {
        return -1;
    }
    int text_written = 0;
    int seg_written = 0;
    out[0] = '\0';
    segments_out[0] = '\0';

    double total_log_sum = 0.0;
    long total_tok_count = 0;
    double total_nsp_sum = 0.0;
    long total_nsp_count = 0;

    int needed_text = 0;
    int needed_seg = 0;

    for (int i = 0; i < n_segments; i++) {
        const char* text = whisper_full_get_segment_text(ctx, i);
        int tlen = (text != NULL) ? (int) strlen(text) : 0;

        // Append to the joined text buffer.
        if (text != NULL) {
            if (text_written + tlen + 1 >= text_cap) {
                needed_text = text_written + tlen + 1;
            } else if (needed_text == 0) {
                memcpy(out + text_written, text, tlen);
                text_written += tlen;
                out[text_written] = '\0';
            }
        }

        // Per-segment stats.
        float seg_nsp = whisper_full_get_segment_no_speech_prob(ctx, i);
        total_nsp_sum += seg_nsp;
        total_nsp_count++;

        double seg_log_sum = 0.0;
        long seg_tok_count = 0;
        int n_tokens = whisper_full_n_tokens(ctx, i);
        for (int t = 0; t < n_tokens; t++) {
            const char* tok_text = whisper_full_get_token_text(ctx, i, t);
            if (tok_text != NULL) {
                int tk_len = (int) strlen(tok_text);
                if (tk_len >= 2 && tok_text[0] == '<' && tok_text[tk_len - 1] == '>') {
                    continue;
                }
            }
            float p = whisper_full_get_token_p(ctx, i, t);
            if (!(p > 0.0f)) p = 1.175494e-38f;
            if (p > 1.0f) p = 1.0f;
            double lp = log((double) p);
            total_log_sum += lp;
            total_tok_count++;
            seg_log_sum += lp;
            seg_tok_count++;
        }

        // Whisper segment timestamps are in 10 ms ticks.
        long t0_ms = whisper_full_get_segment_t0(ctx, i) * 10;
        long t1_ms = whisper_full_get_segment_t1(ctx, i) * 10;
        if (t0_ms < 0) t0_ms = 0;
        if (t1_ms < 0) t1_ms = 0;
        float seg_avg_lp = (seg_tok_count > 0)
            ? (float)(seg_log_sum / (double) seg_tok_count)
            : NAN;

        // Format: t0\tt1\tavg_lp\tnsp\ttext\n
        // Write to a stack buffer first so we can sanitize text without
        // touching the output until we know it fits.
        char header[128];
        int header_len = snprintf(header, sizeof(header), "%ld\t%ld\t%g\t%g\t",
                                  t0_ms, t1_ms,
                                  isnan(seg_avg_lp) ? 0.0 : (double) seg_avg_lp,
                                  (double) seg_nsp);
        if (header_len < 0 || header_len >= (int) sizeof(header)) {
            return -1;
        }
        int sanitized_text_len = tlen;
        // +1 for trailing newline, +1 for terminator.
        int needed = seg_written + header_len + sanitized_text_len + 2;
        if (needed > seg_cap) {
            needed_seg = needed;
        } else if (needed_seg == 0) {
            memcpy(segments_out + seg_written, header, header_len);
            seg_written += header_len;
            for (int k = 0; k < tlen; k++) {
                char c = text[k];
                if (c == '\t' || c == '\n' || c == '\r') c = ' ';
                segments_out[seg_written++] = c;
            }
            segments_out[seg_written++] = '\n';
            segments_out[seg_written] = '\0';
        }
    }

    if (needed_text > 0 || needed_seg > 0) {
        if (needed_text > 0) *out_size = needed_text;
        if (needed_seg > 0) *segments_out_size = needed_seg;
        return -2;
    }

    *out_size = text_written;
    *segments_out_size = seg_written;
    if (avg_logprob_out != NULL) {
        *avg_logprob_out = (total_tok_count > 0)
            ? (float)(total_log_sum / (double) total_tok_count)
            : NAN;
    }
    if (no_speech_prob_out != NULL) {
        *no_speech_prob_out = (total_nsp_count > 0)
            ? (float)(total_nsp_sum / (double) total_nsp_count)
            : NAN;
    }
    return 0;
}
