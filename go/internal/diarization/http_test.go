package diarization_test

import (
	"bytes"
	"math"
	"mime/multipart"
	"net/http"
	"net/http/httptest"
	"net/textproto"
	"os"
	"strings"
	"testing"

	"github.com/eordano/speaches-plus-go/internal/audio"
	"github.com/eordano/speaches-plus-go/internal/diarization"
)

func sine16kDiar(freq float64) audio.MonoF32 {
	out := make(audio.MonoF32, 16000)
	for i := 0; i < 16000; i++ {
		out[i] = float32(0.5 * math.Sin(2*math.Pi*freq*float64(i)/16000.0))
	}
	return out
}

func TestDiarizationHandler_NoModels(t *testing.T) {
	h := diarization.NewHandler(nil, nil, diarization.DefaultConfig())

	body := &bytes.Buffer{}
	mw := multipart.NewWriter(body)
	_ = mw.WriteField("response_format", "json")
	_ = mw.Close()

	req := httptest.NewRequest(http.MethodPost, "/v1/audio/diarization", body)
	req.Header.Set("Content-Type", mw.FormDataContentType())
	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, req)

	if rec.Code != http.StatusServiceUnavailable {
		t.Fatalf("status: got %d want 503; body=%s", rec.Code, rec.Body.String())
	}
	if !strings.Contains(rec.Body.String(), "model_not_loaded") {
		t.Fatalf("expected error code 'model_not_loaded' in body, got: %s", rec.Body.String())
	}
}

func TestDiarizationHandler_MissingFile_RealModels(t *testing.T) {
	seg, emb := loadModelsOrSkip(t)
	defer seg.Close()
	defer emb.Close()
	h := diarization.NewHandler(seg, emb, diarization.DefaultConfig())

	body := &bytes.Buffer{}
	mw := multipart.NewWriter(body)
	_ = mw.WriteField("response_format", "json")
	_ = mw.Close()

	req := httptest.NewRequest(http.MethodPost, "/v1/audio/diarization", body)
	req.Header.Set("Content-Type", mw.FormDataContentType())
	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, req)

	if rec.Code != http.StatusUnprocessableEntity {
		t.Fatalf("status: got %d want 422; body=%s", rec.Code, rec.Body.String())
	}
	if !strings.Contains(rec.Body.String(), `"loc":["body","file"]`) {
		t.Fatalf("expected FastAPI-shape validation error mentioning 'file', got: %s",
			rec.Body.String())
	}
}

func TestDiarizationHandler_BadDataURL(t *testing.T) {
	seg, emb := loadModelsOrSkip(t)
	defer seg.Close()
	defer emb.Close()
	h := diarization.NewHandler(seg, emb, diarization.DefaultConfig())

	wavBytes := audio.EncodeWAVMono16(sine16kDiar(440), 16000)
	body := &bytes.Buffer{}
	mw := multipart.NewWriter(body)
	hdr := textproto.MIMEHeader{}
	hdr.Set("Content-Disposition", `form-data; name="file"; filename="in.wav"`)
	hdr.Set("Content-Type", "audio/wav")
	w, _ := mw.CreatePart(hdr)
	_, _ = w.Write(wavBytes)
	_ = mw.WriteField("known_speaker_names[]", "alice")
	_ = mw.WriteField("known_speaker_references[]", "this is not a data URL")
	_ = mw.Close()

	req := httptest.NewRequest(http.MethodPost, "/v1/audio/diarization", body)
	req.Header.Set("Content-Type", mw.FormDataContentType())
	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, req)

	if rec.Code != http.StatusBadRequest {
		t.Fatalf("status: got %d want 400; body=%s", rec.Code, rec.Body.String())
	}
	if !strings.Contains(rec.Body.String(), "data_url_decode_error") {
		t.Fatalf("expected data_url_decode_error in body, got: %s", rec.Body.String())
	}
}

func loadModelsOrSkip(t *testing.T) (*diarization.SegmentationModel, *diarization.EmbeddingModel) {
	t.Helper()
	segPath, embPath := diarModelPathsFromEnv()
	if segPath == "" || embPath == "" {
		t.Skip("DIAR_SEGMENTATION_MODEL / DIAR_EMBEDDING_MODEL not set; skipping real-model test")
	}
	seg, err := diarization.LoadSegmentation(segPath)
	if err != nil {
		t.Skipf("LoadSegmentation(%q): %v; skipping", segPath, err)
	}
	emb, err := diarization.LoadEmbedding(embPath)
	if err != nil {
		_ = seg.Close()
		t.Skipf("LoadEmbedding(%q): %v; skipping", embPath, err)
	}
	return seg, emb
}

func diarModelPathsFromEnv() (string, string) {
	return os.Getenv("DIAR_SEGMENTATION_MODEL"), os.Getenv("DIAR_EMBEDDING_MODEL")
}
