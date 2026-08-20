package diarization

import (
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"mime/multipart"
	"net/http"

	"github.com/eordano/speaches-plus-go/internal/audio"
	"github.com/eordano/speaches-plus-go/internal/oapi"
)

const (
	defaultEmbeddingModelName = "wespeaker-resnet293-LM"
	maxUploadBytes            = 200 << 20
)

const (
	fieldFile  = "file"
	fieldAudio = "audio"
	fieldModel = "model"
	locBody    = "body"
)

const (
	objectList      = "list"
	objectEmbedding = "embedding"
)

const (
	headerContentType = "Content-Type"
	contentTypeJSON   = "application/json"
)

const (
	codeModelNotLoaded    = "model_not_loaded"
	codeMultipartParse    = "multipart_parse_error"
	codeMultipartRead     = "multipart_read_error"
	codeFileTooLarge      = "file_too_large"
	codeDataURLDecode     = "data_url_decode_error"
	codeDataURLTooLarge   = "data_url_too_large"
	codeAudioDecode       = "audio_decode_error"
	codeAudioTooShort     = "audio_too_short"
	codeEmbedFailed       = "embed_failed"
	validationTypeMissing = "missing"
	validationMsgRequired = "Field required"
	logEmbedFailed        = "embed failed"
	logKeyErr             = "err"
	logKeyIndex           = "index"
)

type EmbeddingsHandler struct {
	Emb *EmbeddingModel
}

func NewEmbeddingsHandler(emb *EmbeddingModel) *EmbeddingsHandler {
	return &EmbeddingsHandler{Emb: emb}
}

type embeddingItem struct {
	Object    string    `json:"object"`
	Index     int       `json:"index"`
	Embedding []float32 `json:"embedding"`
}

type embeddingsResponse struct {
	Object string          `json:"object"`
	Data   []embeddingItem `json:"data"`
	Model  string          `json:"model"`
	Usage  embeddingsUsage `json:"usage"`
}

type embeddingsUsage struct {
	AudioSeconds float64 `json:"audio_seconds"`
}

type audioInput struct {
	bytes []byte
	mime  string
}

func (h *EmbeddingsHandler) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	if err := h.handle(w, r); err != nil {
		writeAPIError(w, err)
	}
}

func (h *EmbeddingsHandler) handle(w http.ResponseWriter, r *http.Request) error {
	if h.Emb == nil {
		return &apiErr{
			Status: http.StatusServiceUnavailable,
			Msg:    "embedding model not loaded; set --diar-embedding",
			Kind:   oapi.TypeServiceUnavail,
			Code:   codeModelNotLoaded,
		}
	}

	if err := r.ParseMultipartForm(maxUploadBytes); err != nil {
		return &apiErr{
			Status: http.StatusBadRequest,
			Msg:    "parse multipart: " + err.Error(),
			Kind:   oapi.TypeInvalidRequest,
			Code:   codeMultipartParse,
		}
	}

	inputs, err := collectEmbeddingInputs(r.MultipartForm)
	if err != nil {
		return err
	}
	if len(inputs) == 0 {
		return &apiErr{
			Validation: &oapi.FastAPIErrorEntry{
				Type: validationTypeMissing,
				Loc:  []string{locBody, fieldFile},
				Msg:  validationMsgRequired,
			},
		}
	}

	data := make([]embeddingItem, 0, len(inputs))
	totalSeconds := 0.0
	for idx, in := range inputs {
		samples, err := audio.DecodeUploadedAudio(in.bytes, in.mime)
		if err != nil {
			return &apiErr{
				Status: http.StatusBadRequest,
				Msg:    fmt.Sprintf("audio decode (file index %d): %s", idx, err.Error()),
				Kind:   oapi.TypeInvalidRequest,
				Param:  fieldFile,
				Code:   codeAudioDecode,
			}
		}
		if len(samples) < h.Emb.MinInputSamples() {
			return &apiErr{
				Status: http.StatusBadRequest,
				Msg: fmt.Sprintf("input audio too short (file index %d, %d samples; need >=%d)",
					idx, len(samples), h.Emb.MinInputSamples()),
				Kind:  oapi.TypeInvalidRequest,
				Param: fieldFile,
				Code:  codeAudioTooShort,
			}
		}
		totalSeconds += float64(len(samples)) / float64(EmbeddingSampleRate)

		vec, err := h.Emb.Embed([]float32(samples))
		if err != nil {
			slog.Error(logEmbedFailed, logKeyErr, err, logKeyIndex, idx)
			return &apiErr{
				Status: http.StatusInternalServerError,
				Msg:    fmt.Sprintf("embed (file index %d): %s", idx, err.Error()),
				Kind:   oapi.TypeServerError,
				Code:   codeEmbedFailed,
			}
		}
		data = append(data, embeddingItem{Object: objectEmbedding, Index: idx, Embedding: vec})
	}

	model := r.FormValue(fieldModel)
	if model == "" {
		model = defaultEmbeddingModelName
	}

	w.Header().Set(headerContentType, contentTypeJSON)
	w.WriteHeader(http.StatusOK)
	_ = json.NewEncoder(w).Encode(embeddingsResponse{
		Object: objectList,
		Data:   data,
		Model:  model,
		Usage:  embeddingsUsage{AudioSeconds: totalSeconds},
	})
	return nil
}

func collectEmbeddingInputs(form *multipart.Form) ([]audioInput, error) {
	if form == nil {
		return nil, nil
	}
	inputs := make([]audioInput, 0, len(form.File[fieldFile])+len(form.Value[fieldAudio]))
	for _, fh := range form.File[fieldFile] {
		data, err := readMultipartFile(fh)
		if errors.Is(err, errFileTooLarge) {
			return nil, &apiErr{
				Status: http.StatusRequestEntityTooLarge,
				Msg:    fmt.Sprintf("file exceeds maximum size of %d bytes", maxUploadBytes),
				Kind:   oapi.TypeInvalidRequest,
				Param:  fieldFile,
				Code:   codeFileTooLarge,
			}
		}
		if err != nil {
			return nil, &apiErr{
				Status: http.StatusBadRequest,
				Msg:    "file read: " + err.Error(),
				Kind:   oapi.TypeInvalidRequest,
				Param:  fieldFile,
				Code:   codeMultipartRead,
			}
		}
		inputs = append(inputs, audioInput{bytes: data, mime: fh.Header.Get(headerContentType)})
	}
	for _, dataURL := range form.Value[fieldAudio] {
		if len(dataURL) > maxUploadBytes {
			return nil, &apiErr{
				Status: http.StatusRequestEntityTooLarge,
				Msg:    fmt.Sprintf("audio data URL exceeds maximum size of %d bytes", maxUploadBytes),
				Kind:   oapi.TypeInvalidRequest,
				Param:  fieldAudio,
				Code:   codeDataURLTooLarge,
			}
		}
		b, mime, err := decodeDataURL(dataURL)
		if err != nil {
			return nil, &apiErr{
				Status: http.StatusBadRequest,
				Msg:    "audio data URL: " + err.Error(),
				Kind:   oapi.TypeInvalidRequest,
				Param:  fieldAudio,
				Code:   codeDataURLDecode,
			}
		}
		inputs = append(inputs, audioInput{bytes: b, mime: mime})
	}
	return inputs, nil
}

var errFileTooLarge = errors.New("file exceeds maximum upload size")

func readMultipartFile(fh *multipart.FileHeader) ([]byte, error) {
	f, err := fh.Open()
	if err != nil {
		return nil, err
	}
	defer f.Close()
	buf, err := io.ReadAll(io.LimitReader(f, maxUploadBytes+1))
	if err != nil {
		return nil, err
	}
	if int64(len(buf)) > maxUploadBytes {
		return nil, errFileTooLarge
	}
	return buf, nil
}
