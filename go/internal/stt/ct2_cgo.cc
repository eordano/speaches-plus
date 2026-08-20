#include <ctranslate2/models/whisper.h>
#include <ctranslate2/storage_view.h>
#include <ctranslate2/devices.h>
#include <ctranslate2/types.h>

#include <memory>
#include <string>
#include <vector>
#include <cstring>
#include <cstdio>
#include <limits>

extern "C" {

struct sp_ct2_whisper {
    std::unique_ptr<ctranslate2::models::Whisper> model;
};

sp_ct2_whisper* sp_ct2_open(const char* model_path,
                            const char* device,
                            const char* compute_type) {
    try {
        ctranslate2::Device dev = ctranslate2::str_to_device(device ? device : "cpu");
        ctranslate2::ComputeType ct = ctranslate2::str_to_compute_type(
            compute_type && *compute_type ? compute_type : "default");
        auto* h = new sp_ct2_whisper();
        h->model = std::make_unique<ctranslate2::models::Whisper>(
            std::string(model_path), dev, ct, std::vector<int>{0}, false);
        return h;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "[ct2] sp_ct2_open: %s\n", e.what());
        return nullptr;
    }
}

void sp_ct2_close(sp_ct2_whisper* h) {
    if (h) {
        delete h;
    }
}

int sp_ct2_n_mels(sp_ct2_whisper* h) {
    if (!h || !h->model) return -1;
    return static_cast<int>(h->model->n_mels());
}

// Internal: single decode pass against the Whisper model. `with_timestamps`
// controls whether `<|notimestamps|>` is included in the prompt; with it
// off, ct2 emits timestamp tokens (`<|0.00|>`, ...) in the result sequence
// which the per-segment caller parses to derive segment boundaries.
//
// `tokens_out` (if non-null + `tokens_out_size` non-null) is filled with the
// raw token strings (BPE pieces and special tokens), one per line, separated
// by `\n`. Tokens themselves don't contain newlines (BPE space markers use
// `Ġ`, not space; specials are `<|...|>`), so the framing is unambiguous.
static int sp_ct2_generate_inner(sp_ct2_whisper* h,
                                 const float* mel,
                                 int n_mels,
                                 int n_frames,
                                 const char* language_token,
                                 int beam_size,
                                 bool with_timestamps,
                                 char* out, int* out_size,
                                 char* tokens_out, int* tokens_out_size,
                                 float* no_speech_prob_out,
                                 float* avg_logprob_out) {
    if (!h || !h->model || !mel) return -1;
    try {
        std::vector<float> mel_data(mel, mel + (n_mels * n_frames));
        ctranslate2::StorageView features(
            ctranslate2::Shape{1, static_cast<ctranslate2::dim_t>(n_mels),
                                  static_cast<ctranslate2::dim_t>(n_frames)},
            mel_data);

        std::vector<std::string> prompt_vec = {
            "<|startoftranscript|>",
            language_token && *language_token ? language_token : "<|en|>",
            "<|transcribe|>",
        };
        if (!with_timestamps) {
            prompt_vec.push_back("<|notimestamps|>");
        }
        std::vector<std::vector<std::string>> prompts = { prompt_vec };

        ctranslate2::models::WhisperOptions options;
        options.beam_size = beam_size > 0 ? beam_size : 1;
        options.return_no_speech_prob = (no_speech_prob_out != nullptr);
        options.return_scores = (avg_logprob_out != nullptr);

        auto futures = h->model->generate(features, prompts, options);
        if (futures.empty()) {
            std::fprintf(stderr, "[ct2] generate returned no futures\n");
            return -1;
        }
        auto result = futures[0].get();
        if (no_speech_prob_out) {
            *no_speech_prob_out = result.no_speech_prob;
        }
        if (avg_logprob_out) {
            // result.scores is one entry per returned beam; with beam_size=1
            // (or any greedy/beam config) the top hypothesis is index 0 and
            // its score is the sequence-averaged log-probability -- the same
            // signal faster-whisper exposes as `segment.avg_logprob` for a
            // single-segment chunk.
            *avg_logprob_out = result.has_scores() ? result.scores[0]
                                                   : std::numeric_limits<float>::quiet_NaN();
        }
        if (result.sequences.empty() || result.sequences[0].empty()) {
            if (out && out_size) {
                if (*out_size < 1) return -1;
                out[0] = '\0';
                *out_size = 0;
            }
            if (tokens_out && tokens_out_size) {
                if (*tokens_out_size < 1) return -1;
                tokens_out[0] = '\0';
                *tokens_out_size = 0;
            }
            return 0;
        }

        const auto& tokens = result.sequences[0];

        // Joined text: skip timestamp tokens since they're not real text.
        // (When `with_timestamps` is false they're absent anyway, so this
        // is a no-op on the legacy path.)
        std::string text;
        for (const auto& tok : tokens) {
            if (tok.size() >= 4 && tok[0] == '<' && tok[1] == '|' &&
                tok[tok.size() - 2] == '|' && tok[tok.size() - 1] == '>') {
                // Either a special (<|sot|>, <|en|>, ...) or a timestamp
                // (<|0.00|>) -- neither belongs in the joined text. Note
                // we accept the joined text on the legacy path may include
                // surrounding specials when `with_timestamps` is false, but
                // those were stripped on the Go side via DecodeBPE-then-trim
                // already.
                continue;
            }
            text += tok;
        }

        if (out && out_size) {
            int needed = static_cast<int>(text.size()) + 1;
            if (needed > *out_size) {
                *out_size = needed;
                return -2;
            }
        }

        // Tokens blob: tokens separated by '\n', for the segmented path.
        std::string tokens_blob;
        if (tokens_out && tokens_out_size) {
            size_t total = 0;
            for (const auto& tok : tokens) total += tok.size() + 1;
            tokens_blob.reserve(total);
            for (const auto& tok : tokens) {
                tokens_blob += tok;
                tokens_blob.push_back('\n');
            }
            int needed = static_cast<int>(tokens_blob.size()) + 1;
            if (needed > *tokens_out_size) {
                *tokens_out_size = needed;
                return -2;
            }
        }

        if (out && out_size) {
            std::memcpy(out, text.data(), text.size());
            out[text.size()] = '\0';
            *out_size = static_cast<int>(text.size());
        }
        if (tokens_out && tokens_out_size) {
            std::memcpy(tokens_out, tokens_blob.data(), tokens_blob.size());
            tokens_out[tokens_blob.size()] = '\0';
            *tokens_out_size = static_cast<int>(tokens_blob.size());
        }
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "[ct2] sp_ct2_generate: %s\n", e.what());
        return -1;
    }
}

int sp_ct2_generate(sp_ct2_whisper* h,
                    const float* mel,
                    int n_mels,
                    int n_frames,
                    const char* language_token,
                    int beam_size,
                    char* out,
                    int* out_size,
                    float* no_speech_prob_out,
                    float* avg_logprob_out) {
    if (!out || !out_size) return -1;
    return sp_ct2_generate_inner(h, mel, n_mels, n_frames, language_token,
                                 beam_size, /*with_timestamps=*/false,
                                 out, out_size, nullptr, nullptr,
                                 no_speech_prob_out, avg_logprob_out);
}

int sp_ct2_generate_segmented(sp_ct2_whisper* h,
                              const float* mel,
                              int n_mels,
                              int n_frames,
                              const char* language_token,
                              int beam_size,
                              char* out, int* out_size,
                              char* tokens_out, int* tokens_out_size,
                              float* no_speech_prob_out,
                              float* avg_logprob_out) {
    if (!out || !out_size || !tokens_out || !tokens_out_size) return -1;
    return sp_ct2_generate_inner(h, mel, n_mels, n_frames, language_token,
                                 beam_size, /*with_timestamps=*/true,
                                 out, out_size, tokens_out, tokens_out_size,
                                 no_speech_prob_out, avg_logprob_out);
}

}
