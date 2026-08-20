#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <whisper.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace py = pybind11;

namespace {

extern "C" {

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

        if (text != NULL) {
            if (text_written + tlen + 1 >= text_cap) {
                needed_text = text_written + tlen + 1;
            } else if (needed_text == 0) {
                memcpy(out + text_written, text, tlen);
                text_written += tlen;
                out[text_written] = '\0';
            }
        }

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

        long t0_ms = whisper_full_get_segment_t0(ctx, i) * 10;
        long t1_ms = whisper_full_get_segment_t1(ctx, i) * 10;
        if (t0_ms < 0) t0_ms = 0;
        if (t1_ms < 0) t1_ms = 0;
        float seg_avg_lp = (seg_tok_count > 0)
            ? (float)(seg_log_sum / (double) seg_tok_count)
            : NAN;

        char header[128];
        int header_len = snprintf(header, sizeof(header), "%ld\t%ld\t%g\t%g\t",
                                  t0_ms, t1_ms,
                                  isnan(seg_avg_lp) ? 0.0 : (double) seg_avg_lp,
                                  (double) seg_nsp);
        if (header_len < 0 || header_len >= (int) sizeof(header)) {
            return -1;
        }
        int sanitized_text_len = tlen;
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

}

class WhisperContext {
public:
    explicit WhisperContext(const std::string& model_path) {
        if (model_path.empty()) {
            throw std::invalid_argument(
                "WhisperContext: model_path must be non-empty");
        }
        ctx_ = sp_whisper_init(model_path.c_str());
        if (!ctx_) {
            throw std::runtime_error(
                "WhisperContext: failed to load model at '" + model_path +
                "'; check stderr for the underlying whisper.cpp error.");
        }
    }

    ~WhisperContext() {
        if (ctx_) {
            sp_whisper_free(ctx_);
            ctx_ = nullptr;
        }
    }

    WhisperContext(const WhisperContext&) = delete;
    WhisperContext& operator=(const WhisperContext&) = delete;

    void close() {
        if (ctx_) {
            sp_whisper_free(ctx_);
            ctx_ = nullptr;
        }
    }

    bool is_open() const { return ctx_ != nullptr; }

    py::dict transcribe(
        py::array_t<float, py::array::c_style | py::array::forcecast> samples,
        py::object language
    ) {
        if (!ctx_) throw std::runtime_error("WhisperContext: handle is closed");
        if (samples.ndim() != 1) {
            throw std::invalid_argument(
                "WhisperContext.transcribe: samples must be 1-D float32");
        }
        const int n_samples = static_cast<int>(samples.shape(0));
        const float* samples_ptr = samples.data();

        std::string lang_storage;
        const char* lang_cstr = nullptr;
        if (!language.is_none()) {
            lang_storage = language.cast<std::string>();
            lang_cstr = lang_storage.c_str();
        }

        int buf_size = 65536;
        std::vector<char> buf;
        for (;;) {
            buf.assign(buf_size, '\0');
            int size = buf_size;
            int rc;
            {
                py::gil_scoped_release release;
                if (lang_cstr) {
                    rc = sp_whisper_transcribe_with_language(
                        ctx_, samples_ptr, n_samples,
                        buf.data(), &size, lang_cstr);
                } else {
                    rc = sp_whisper_transcribe(
                        ctx_, samples_ptr, n_samples,
                        buf.data(), &size);
                }
            }
            if (rc == 0) {
                py::dict out;
                out["text"] = std::string(buf.data(), static_cast<size_t>(size));
                return out;
            }
            if (rc == -2) {
                buf_size = size + 1;
                continue;
            }
            throw std::runtime_error(
                "WhisperContext.transcribe: whisper_full failed (rc=" +
                std::to_string(rc) + ")");
        }
    }

    py::dict transcribe_full(
        py::array_t<float, py::array::c_style | py::array::forcecast> samples
    ) {
        if (!ctx_) throw std::runtime_error("WhisperContext: handle is closed");
        if (samples.ndim() != 1) {
            throw std::invalid_argument(
                "WhisperContext.transcribe_full: samples must be 1-D float32");
        }
        const int n_samples = static_cast<int>(samples.shape(0));
        const float* samples_ptr = samples.data();

        float avg_logprob = std::numeric_limits<float>::quiet_NaN();
        float no_speech_prob = std::numeric_limits<float>::quiet_NaN();

        int buf_size = 65536;
        std::vector<char> buf;
        for (;;) {
            buf.assign(buf_size, '\0');
            int size = buf_size;
            int rc;
            {
                py::gil_scoped_release release;
                rc = sp_whisper_transcribe_full(
                    ctx_, samples_ptr, n_samples,
                    buf.data(), &size,
                    &avg_logprob, &no_speech_prob);
            }
            if (rc == 0) {
                py::dict out;
                out["text"] = std::string(buf.data(), static_cast<size_t>(size));
                if (std::isnan(avg_logprob)) {
                    out["avg_logprob"] = py::none();
                } else {
                    out["avg_logprob"] = avg_logprob;
                }
                if (std::isnan(no_speech_prob)) {
                    out["no_speech_prob"] = py::none();
                } else {
                    out["no_speech_prob"] = no_speech_prob;
                }
                return out;
            }
            if (rc == -2) {
                buf_size = size + 1;
                continue;
            }
            throw std::runtime_error(
                "WhisperContext.transcribe_full: whisper_full failed (rc=" +
                std::to_string(rc) + ")");
        }
    }

    py::dict transcribe_segmented(
        py::array_t<float, py::array::c_style | py::array::forcecast> samples,
        py::object language
    ) {
        if (!ctx_) throw std::runtime_error("WhisperContext: handle is closed");
        if (samples.ndim() != 1) {
            throw std::invalid_argument(
                "WhisperContext.transcribe_segmented: samples must be 1-D float32");
        }
        const int n_samples = static_cast<int>(samples.shape(0));
        const float* samples_ptr = samples.data();

        std::string lang_storage;
        const char* lang_cstr = nullptr;
        if (!language.is_none()) {
            lang_storage = language.cast<std::string>();
            lang_cstr = lang_storage.c_str();
        }

        float avg_logprob = std::numeric_limits<float>::quiet_NaN();
        float no_speech_prob = std::numeric_limits<float>::quiet_NaN();

        int text_cap = 4096;
        int seg_cap = 8192;
        std::vector<char> text_buf;
        std::vector<char> seg_buf;
        for (;;) {
            text_buf.assign(text_cap, '\0');
            seg_buf.assign(seg_cap, '\0');
            int text_size = text_cap;
            int seg_size = seg_cap;
            int rc;
            {
                py::gil_scoped_release release;
                if (lang_cstr) {
                    rc = sp_whisper_transcribe_segmented_with_language(
                        ctx_, samples_ptr, n_samples,
                        text_buf.data(), &text_size,
                        seg_buf.data(), &seg_size,
                        &avg_logprob, &no_speech_prob,
                        lang_cstr);
                } else {
                    rc = sp_whisper_transcribe_segmented(
                        ctx_, samples_ptr, n_samples,
                        text_buf.data(), &text_size,
                        seg_buf.data(), &seg_size,
                        &avg_logprob, &no_speech_prob);
                }
            }
            if (rc == 0) {
                py::dict out;
                out["text"] = std::string(text_buf.data(), static_cast<size_t>(text_size));
                py::list segments;
                const char* p = seg_buf.data();
                const char* end = p + seg_size;
                while (p < end) {
                    const char* line_end = static_cast<const char*>(memchr(p, '\n', end - p));
                    if (line_end == nullptr) {
                        line_end = end;
                    }
                    if (line_end > p) {
                        const char* fields[5] = {p, nullptr, nullptr, nullptr, nullptr};
                        int field_idx = 1;
                        for (const char* q = p; q < line_end && field_idx < 5; ++q) {
                            if (*q == '\t') {
                                fields[field_idx++] = q + 1;
                            }
                        }
                        if (field_idx == 5) {
                            const char* f0_end = static_cast<const char*>(memchr(fields[0], '\t', line_end - fields[0]));
                            const char* f1_end = static_cast<const char*>(memchr(fields[1], '\t', line_end - fields[1]));
                            const char* f2_end = static_cast<const char*>(memchr(fields[2], '\t', line_end - fields[2]));
                            const char* f3_end = static_cast<const char*>(memchr(fields[3], '\t', line_end - fields[3]));
                            if (f0_end && f1_end && f2_end && f3_end) {
                                std::string s_t0(fields[0], f0_end);
                                std::string s_t1(fields[1], f1_end);
                                std::string s_lp(fields[2], f2_end);
                                std::string s_nsp(fields[3], f3_end);
                                std::string s_txt(fields[4], line_end);
                                py::dict seg;
                                seg["t_start_ms"] = static_cast<long>(std::strtol(s_t0.c_str(), nullptr, 10));
                                seg["t_end_ms"] = static_cast<long>(std::strtol(s_t1.c_str(), nullptr, 10));
                                seg["avg_logprob"] = static_cast<double>(std::strtod(s_lp.c_str(), nullptr));
                                seg["no_speech_prob"] = static_cast<double>(std::strtod(s_nsp.c_str(), nullptr));
                                seg["text"] = s_txt;
                                segments.append(seg);
                            }
                        }
                    }
                    p = (line_end < end) ? line_end + 1 : end;
                }
                out["segments"] = segments;
                if (std::isnan(avg_logprob)) {
                    out["avg_logprob"] = py::none();
                } else {
                    out["avg_logprob"] = avg_logprob;
                }
                if (std::isnan(no_speech_prob)) {
                    out["no_speech_prob"] = py::none();
                } else {
                    out["no_speech_prob"] = no_speech_prob;
                }
                return out;
            }
            if (rc == -2) {
                if (text_size > text_cap) text_cap = text_size + 1;
                if (seg_size > seg_cap) seg_cap = seg_size + 1;
                continue;
            }
            throw std::runtime_error(
                "WhisperContext.transcribe_segmented: whisper_full failed (rc=" +
                std::to_string(rc) + ")");
        }
    }

private:
    struct whisper_context* ctx_ = nullptr;

    // Convenience: when a caller passes language=..., we want to override the
    // hardcoded "en" in sp_whisper_run without forking the whole helper. We
    // re-implement the minimal text-only path here, matching the upstream's
    // wparams shape but with caller-supplied language.
    static int sp_whisper_transcribe_with_language(
        struct whisper_context* ctx,
        const float* samples,
        int n_samples,
        char* out,
        int* out_size,
        const char* language
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
        wparams.language          = language;
        wparams.n_threads         = 4;
        wparams.suppress_blank    = true;
        wparams.no_context        = true;
        wparams.single_segment    = false;

        int rc = whisper_full(ctx, wparams, samples, n_samples);
        if (rc != 0) return rc;

        int n_segments = whisper_full_n_segments(ctx);
        int written = 0;
        int cap = *out_size;
        if (cap < 1) return -1;
        out[0] = '\0';

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
        }
        *out_size = written;
        return 0;
    }

    static int sp_whisper_transcribe_segmented_with_language(
        struct whisper_context* ctx,
        const float* samples,
        int n_samples,
        char* out, int* out_size,
        char* segments_out, int* segments_out_size,
        float* avg_logprob_out,
        float* no_speech_prob_out,
        const char* language
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
        wparams.language          = language;
        wparams.n_threads         = 4;
        wparams.suppress_blank    = true;
        wparams.no_context        = true;
        wparams.single_segment    = false;

        int rc = whisper_full(ctx, wparams, samples, n_samples);
        if (rc != 0) return rc;

        int n_segments = whisper_full_n_segments(ctx);
        int text_cap = *out_size;
        int seg_cap = *segments_out_size;
        if (text_cap < 1 || seg_cap < 1) return -1;
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

            if (text != NULL) {
                if (text_written + tlen + 1 >= text_cap) {
                    needed_text = text_written + tlen + 1;
                } else if (needed_text == 0) {
                    memcpy(out + text_written, text, tlen);
                    text_written += tlen;
                    out[text_written] = '\0';
                }
            }

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

            long t0_ms = whisper_full_get_segment_t0(ctx, i) * 10;
            long t1_ms = whisper_full_get_segment_t1(ctx, i) * 10;
            if (t0_ms < 0) t0_ms = 0;
            if (t1_ms < 0) t1_ms = 0;
            float seg_avg_lp = (seg_tok_count > 0)
                ? (float)(seg_log_sum / (double) seg_tok_count)
                : NAN;

            char header[128];
            int header_len = snprintf(header, sizeof(header), "%ld\t%ld\t%g\t%g\t",
                                      t0_ms, t1_ms,
                                      isnan(seg_avg_lp) ? 0.0 : (double) seg_avg_lp,
                                      (double) seg_nsp);
            if (header_len < 0 || header_len >= (int) sizeof(header)) return -1;
            int needed = seg_written + header_len + tlen + 2;
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
};

}

PYBIND11_MODULE(_whisper, m) {
    m.doc() = "whisper.cpp Python bindings (port of speaches-plus/go/internal/stt/whisper_cgo.c)";

    py::class_<WhisperContext>(m, "WhisperContext")
        .def(py::init<const std::string&>(),
             py::arg("model_path"))
        .def_static("open",
                    [](const std::string& model_path) {
                        return std::unique_ptr<WhisperContext>(
                            new WhisperContext(model_path));
                    },
                    py::arg("model_path"))
        .def_property_readonly("is_open", &WhisperContext::is_open)
        .def("close", &WhisperContext::close)
        .def("transcribe", &WhisperContext::transcribe,
             py::arg("samples"),
             py::arg("language") = py::none())
        .def("transcribe_full", &WhisperContext::transcribe_full,
             py::arg("samples"))
        .def("transcribe_segmented", &WhisperContext::transcribe_segmented,
             py::arg("samples"),
             py::arg("language") = py::none())
        .def("__enter__", [](WhisperContext& self) -> WhisperContext& { return self; })
        .def("__exit__",
             [](WhisperContext& self, py::object, py::object, py::object) {
                 self.close();
             });
}
