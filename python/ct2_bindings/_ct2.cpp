#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <ctranslate2/models/whisper.h>
#include <ctranslate2/storage_view.h>
#include <ctranslate2/devices.h>
#include <ctranslate2/types.h>

#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace py = pybind11;

namespace {

struct sp_ct2_whisper {
    std::unique_ptr<ctranslate2::models::Whisper> model;
};

extern "C" {

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

        std::string text;
        for (const auto& tok : tokens) {
            if (tok.size() >= 4 && tok[0] == '<' && tok[1] == '|' &&
                tok[tok.size() - 2] == '|' && tok[tok.size() - 1] == '>') {
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

class Ct2Whisper {
public:
    Ct2Whisper(const std::string& model_path,
               const std::string& device,
               const std::string& compute_type) {
        handle_ = sp_ct2_open(model_path.c_str(),
                              device.c_str(),
                              compute_type.c_str());
        if (!handle_) {
            throw std::runtime_error(
                "Ct2Whisper: failed to open model at '" + model_path +
                "' (device='" + device + "', compute_type='" + compute_type +
                "'); check stderr for the underlying CTranslate2 error.");
        }
    }

    ~Ct2Whisper() {
        if (handle_) {
            sp_ct2_close(handle_);
            handle_ = nullptr;
        }
    }

    Ct2Whisper(const Ct2Whisper&) = delete;
    Ct2Whisper& operator=(const Ct2Whisper&) = delete;

    int n_mels() const {
        if (!handle_) throw std::runtime_error("Ct2Whisper: handle is closed");
        return sp_ct2_n_mels(handle_);
    }

    void close() {
        if (handle_) {
            sp_ct2_close(handle_);
            handle_ = nullptr;
        }
    }

    bool is_open() const { return handle_ != nullptr; }

    py::dict generate(py::array_t<float, py::array::c_style | py::array::forcecast> mel,
                      const std::string& language_token,
                      int beam_size,
                      bool return_no_speech_prob,
                      bool return_avg_logprob) {
        if (!handle_) throw std::runtime_error("Ct2Whisper: handle is closed");
        if (mel.ndim() != 2) {
            throw std::invalid_argument(
                "Ct2Whisper.generate: mel must be 2-D (n_mels, n_frames)");
        }
        const int n_mels = static_cast<int>(mel.shape(0));
        const int n_frames = static_cast<int>(mel.shape(1));
        const float* mel_ptr = mel.data();

        float no_speech_prob = 0.0f;
        float avg_logprob = std::numeric_limits<float>::quiet_NaN();
        float* nsp_out = return_no_speech_prob ? &no_speech_prob : nullptr;
        float* alp_out = return_avg_logprob ? &avg_logprob : nullptr;

        int buf_size = 4096;
        std::vector<char> buf;
        for (;;) {
            buf.assign(buf_size, '\0');
            int size = buf_size;
            int rc;
            {
                py::gil_scoped_release release;
                rc = sp_ct2_generate(handle_,
                                     mel_ptr,
                                     n_mels,
                                     n_frames,
                                     language_token.c_str(),
                                     beam_size,
                                     buf.data(),
                                     &size,
                                     nsp_out,
                                     alp_out);
            }
            if (rc == 0) {
                py::dict out;
                out["text"] = std::string(buf.data(), static_cast<size_t>(size));
                if (return_no_speech_prob) {
                    out["no_speech_prob"] = no_speech_prob;
                } else {
                    out["no_speech_prob"] = py::none();
                }
                if (return_avg_logprob) {
                    if (std::isnan(avg_logprob)) {
                        out["avg_logprob"] = py::none();
                    } else {
                        out["avg_logprob"] = avg_logprob;
                    }
                } else {
                    out["avg_logprob"] = py::none();
                }
                return out;
            }
            if (rc == -2) {
                buf_size = size + 1;
                continue;
            }
            throw std::runtime_error(
                "Ct2Whisper.generate: failed (rc=" + std::to_string(rc) + ")");
        }
    }

    py::dict generate_segmented(py::array_t<float, py::array::c_style | py::array::forcecast> mel,
                                const std::string& language_token,
                                int beam_size,
                                bool return_no_speech_prob,
                                bool return_avg_logprob) {
        if (!handle_) throw std::runtime_error("Ct2Whisper: handle is closed");
        if (mel.ndim() != 2) {
            throw std::invalid_argument(
                "Ct2Whisper.generate_segmented: mel must be 2-D (n_mels, n_frames)");
        }
        const int n_mels = static_cast<int>(mel.shape(0));
        const int n_frames = static_cast<int>(mel.shape(1));
        const float* mel_ptr = mel.data();

        float no_speech_prob = 0.0f;
        float avg_logprob = std::numeric_limits<float>::quiet_NaN();
        float* nsp_out = return_no_speech_prob ? &no_speech_prob : nullptr;
        float* alp_out = return_avg_logprob ? &avg_logprob : nullptr;

        int text_cap = 4096;
        int tok_cap = 16384;
        std::vector<char> text_buf;
        std::vector<char> tok_buf;
        for (;;) {
            text_buf.assign(text_cap, '\0');
            tok_buf.assign(tok_cap, '\0');
            int out_size = text_cap;
            int toks_size = tok_cap;
            int rc;
            {
                py::gil_scoped_release release;
                rc = sp_ct2_generate_segmented(handle_,
                                               mel_ptr,
                                               n_mels,
                                               n_frames,
                                               language_token.c_str(),
                                               beam_size,
                                               text_buf.data(),
                                               &out_size,
                                               tok_buf.data(),
                                               &toks_size,
                                               nsp_out,
                                               alp_out);
            }
            if (rc == 0) {
                py::dict out;
                out["text"] = std::string(text_buf.data(), static_cast<size_t>(out_size));
                out["tokens_blob"] = std::string(tok_buf.data(), static_cast<size_t>(toks_size));
                if (return_no_speech_prob) {
                    out["no_speech_prob"] = no_speech_prob;
                } else {
                    out["no_speech_prob"] = py::none();
                }
                if (return_avg_logprob) {
                    if (std::isnan(avg_logprob)) {
                        out["avg_logprob"] = py::none();
                    } else {
                        out["avg_logprob"] = avg_logprob;
                    }
                } else {
                    out["avg_logprob"] = py::none();
                }
                return out;
            }
            if (rc == -2) {
                if (out_size > text_cap) text_cap = out_size + 1;
                if (toks_size > tok_cap) tok_cap = toks_size + 1;
                continue;
            }
            throw std::runtime_error(
                "Ct2Whisper.generate_segmented: failed (rc=" + std::to_string(rc) + ")");
        }
    }

private:
    sp_ct2_whisper* handle_ = nullptr;
};

}

PYBIND11_MODULE(_ct2, m) {
    m.doc() = "CTranslate2 Whisper bindings (port of speaches-plus/go/internal/stt/ct2_cgo.cc)";

    py::class_<Ct2Whisper>(m, "Ct2Whisper")
        .def(py::init<const std::string&, const std::string&, const std::string&>(),
             py::arg("model_path"),
             py::arg("device") = "cpu",
             py::arg("compute_type") = "default")
        .def_static("open",
                    [](const std::string& model_path,
                       const std::string& device,
                       const std::string& compute_type) {
                        return std::unique_ptr<Ct2Whisper>(
                            new Ct2Whisper(model_path, device, compute_type));
                    },
                    py::arg("model_path"),
                    py::arg("device") = "cpu",
                    py::arg("compute_type") = "default")
        .def_property_readonly("n_mels", &Ct2Whisper::n_mels)
        .def_property_readonly("is_open", &Ct2Whisper::is_open)
        .def("close", &Ct2Whisper::close)
        .def("generate", &Ct2Whisper::generate,
             py::arg("mel"),
             py::arg("language_token") = "<|en|>",
             py::arg("beam_size") = 5,
             py::arg("return_no_speech_prob") = true,
             py::arg("return_avg_logprob") = true)
        .def("generate_segmented", &Ct2Whisper::generate_segmented,
             py::arg("mel"),
             py::arg("language_token") = "<|en|>",
             py::arg("beam_size") = 5,
             py::arg("return_no_speech_prob") = true,
             py::arg("return_avg_logprob") = true)
        .def("__enter__", [](Ct2Whisper& self) -> Ct2Whisper& { return self; })
        .def("__exit__",
             [](Ct2Whisper& self, py::object, py::object, py::object) {
                 self.close();
             });
}
