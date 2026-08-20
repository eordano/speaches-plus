package diarization_test

import (
	"bytes"
	"encoding/json"
	"io"
	"mime/multipart"
	"net/http"
	"net/http/httptest"
	"net/textproto"
	"strings"
	"testing"

	"github.com/eordano/speaches-plus-go/internal/audio"
	"github.com/eordano/speaches-plus-go/internal/diarization"
)

func multipartWithWavFile(t *testing.T, samples audio.MonoF32) (io.Reader, string) {
	t.Helper()
	wav := audio.EncodeWAVMono16(samples, 16000)
	body := &bytes.Buffer{}
	mw := multipart.NewWriter(body)
	hdr := textproto.MIMEHeader{}
	hdr.Set("Content-Disposition", `form-data; name="file"; filename="in.wav"`)
	hdr.Set("Content-Type", "audio/wav")
	w, _ := mw.CreatePart(hdr)
	_, _ = w.Write(wav)
	_ = mw.Close()
	return body, mw.FormDataContentType()
}

func postEmbeddings(h *diarization.EmbeddingsHandler, body io.Reader, contentType string) *httptest.ResponseRecorder {
	req := httptest.NewRequest(http.MethodPost, "/v1/audio/embeddings", body)
	req.Header.Set("Content-Type", contentType)
	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, req)
	return rec
}

func TestEmbeddingsHandler_NoModel(t *testing.T) {
	h := diarization.NewEmbeddingsHandler(nil)

	body := &bytes.Buffer{}
	mw := multipart.NewWriter(body)
	_ = mw.Close()
	rec := postEmbeddings(h, body, mw.FormDataContentType())

	if rec.Code != http.StatusServiceUnavailable {
		t.Fatalf("status: got %d want 503; body=%s", rec.Code, rec.Body.String())
	}
	if !strings.Contains(rec.Body.String(), "model_not_loaded") {
		t.Fatalf("expected error code 'model_not_loaded' in body, got: %s", rec.Body.String())
	}
}

func TestEmbeddingsHandler_MissingFile(t *testing.T) {
	_, emb := loadModelsOrSkip(t)
	defer emb.Close()
	h := diarization.NewEmbeddingsHandler(emb)

	body := &bytes.Buffer{}
	mw := multipart.NewWriter(body)
	_ = mw.WriteField("model", "wespeaker-resnet293-LM")
	_ = mw.Close()
	rec := postEmbeddings(h, body, mw.FormDataContentType())

	if rec.Code != http.StatusUnprocessableEntity {
		t.Fatalf("status: got %d want 422; body=%s", rec.Code, rec.Body.String())
	}
	if !strings.Contains(rec.Body.String(), `"loc":["body","file"]`) {
		t.Fatalf("expected FastAPI-shape validation error mentioning 'file', got: %s",
			rec.Body.String())
	}
}

func TestEmbeddingsHandler_AudioTooShort(t *testing.T) {
	_, emb := loadModelsOrSkip(t)
	defer emb.Close()
	h := diarization.NewEmbeddingsHandler(emb)

	body, ct := multipartWithWavFile(t, sine16kDiar(440)[:8000])
	rec := postEmbeddings(h, body, ct)

	if rec.Code != http.StatusBadRequest {
		t.Fatalf("status: got %d want 400; body=%s", rec.Code, rec.Body.String())
	}
	if !strings.Contains(rec.Body.String(), "audio_too_short") {
		t.Fatalf("expected audio_too_short in body, got: %s", rec.Body.String())
	}
}

func TestEmbeddingsHandler_OK(t *testing.T) {
	_, emb := loadModelsOrSkip(t)
	defer emb.Close()
	h := diarization.NewEmbeddingsHandler(emb)

	body, ct := multipartWithWavFile(t, sine16kDiar(440))
	rec := postEmbeddings(h, body, ct)

	if rec.Code != http.StatusOK {
		t.Fatalf("status: got %d want 200; body=%s", rec.Code, rec.Body.String())
	}
	var resp struct {
		Object string `json:"object"`
		Data   []struct {
			Object    string    `json:"object"`
			Index     int       `json:"index"`
			Embedding []float32 `json:"embedding"`
		} `json:"data"`
		Model string `json:"model"`
		Usage struct {
			AudioSeconds float64 `json:"audio_seconds"`
		} `json:"usage"`
	}
	if err := json.Unmarshal(rec.Body.Bytes(), &resp); err != nil {
		t.Fatalf("json unmarshal: %v; body=%s", err, rec.Body.String())
	}
	if resp.Object != "list" {
		t.Fatalf("object: got %q want list", resp.Object)
	}
	if len(resp.Data) != 1 {
		t.Fatalf("data len: got %d want 1", len(resp.Data))
	}
	if resp.Data[0].Object != "embedding" || resp.Data[0].Index != 0 {
		t.Fatalf("data[0]: got %+v want object=embedding index=0", resp.Data[0])
	}
	if got := len(resp.Data[0].Embedding); got != diarization.EmbeddingDim {
		t.Fatalf("embedding dim: got %d want %d", got, diarization.EmbeddingDim)
	}
	if resp.Model == "" {
		t.Fatalf("model field empty")
	}
	if resp.Usage.AudioSeconds <= 0 {
		t.Fatalf("usage.audio_seconds: got %v want >0", resp.Usage.AudioSeconds)
	}
}
