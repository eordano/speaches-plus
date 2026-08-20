package stt_test

import (
	"bytes"
	"encoding/binary"
	"encoding/json"
	"math"
	"mime/multipart"
	"net/http"
	"net/http/httptest"
	"net/textproto"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"sync"
	"testing"

	"github.com/eordano/speaches-plus-go/internal/audio"
	"github.com/eordano/speaches-plus-go/internal/stt"
)

type fakeTranscriber struct {
	mu          sync.Mutex
	gotSamples  []float32
	gotSR       int
	returnText  string
	returnError error
}

func (f *fakeTranscriber) Transcribe(samples []float32, sampleRate int) (string, error) {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.gotSamples = append([]float32(nil), samples...)
	f.gotSR = sampleRate
	return f.returnText, f.returnError
}

func (f *fakeTranscriber) Close() error { return nil }

func sine16k(freq float64) audio.MonoF32 {
	out := make(audio.MonoF32, 16000)
	for i := 0; i < 16000; i++ {
		out[i] = float32(0.5 * math.Sin(2*math.Pi*freq*float64(i)/16000.0))
	}
	return out
}

func encodeRawS16LE(samples audio.MonoF32) []byte {
	buf := bytes.NewBuffer(make([]byte, 0, len(samples)*2))
	for _, s := range samples {
		v := int32(s * 32767.0)
		if v > 32767 {
			v = 32767
		} else if v < -32768 {
			v = -32768
		}
		_ = binary.Write(buf, binary.LittleEndian, int16(v))
	}
	return buf.Bytes()
}

func buildMultipart(t *testing.T, fileName, fileMime string, fileBody []byte, fields map[string]string) (*bytes.Buffer, string) {
	t.Helper()
	body := &bytes.Buffer{}
	mw := multipart.NewWriter(body)

	hdr := textproto.MIMEHeader{}
	hdr.Set("Content-Disposition", `form-data; name="file"; filename="`+fileName+`"`)
	if fileMime != "" {
		hdr.Set("Content-Type", fileMime)
	}
	w, err := mw.CreatePart(hdr)
	if err != nil {
		t.Fatalf("create part: %v", err)
	}
	if _, err := w.Write(fileBody); err != nil {
		t.Fatalf("write part: %v", err)
	}

	for k, v := range fields {
		if err := mw.WriteField(k, v); err != nil {
			t.Fatalf("write field %s: %v", k, err)
		}
	}
	if err := mw.Close(); err != nil {
		t.Fatalf("close mw: %v", err)
	}
	return body, mw.FormDataContentType()
}

func TestTranscriptionsHandler_WAV(t *testing.T) {
	src := sine16k(440)
	wavBytes := audio.EncodeWAVMono16(src, 16000)
	fake := &fakeTranscriber{returnText: "hello"}
	h := stt.NewTranscriptionsHandler(fake)

	body, ct := buildMultipart(t, "in.wav", "audio/wav", wavBytes, map[string]string{
		"response_format": "json",
	})
	req := httptest.NewRequest(http.MethodPost, "/v1/audio/transcriptions", body)
	req.Header.Set("Content-Type", ct)
	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("status: got %d want 200; body=%s", rec.Code, rec.Body.String())
	}
	var got struct {
		Text string `json:"text"`
	}
	if err := json.Unmarshal(rec.Body.Bytes(), &got); err != nil {
		t.Fatalf("parse json: %v body=%q", err, rec.Body.String())
	}
	if got.Text != "hello" {
		t.Fatalf("text: got %q want %q", got.Text, "hello")
	}
	if fake.gotSR != 16000 {
		t.Fatalf("sample rate passed to transcriber: got %d want 16000", fake.gotSR)
	}
	if len(fake.gotSamples) != 16000 {
		t.Fatalf("samples passed to transcriber: got %d want 16000", len(fake.gotSamples))
	}
}

func TestTranscriptionsHandler_RawPCM(t *testing.T) {
	src := sine16k(440)
	pcm := encodeRawS16LE(src)
	fake := &fakeTranscriber{returnText: "raw ok"}
	h := stt.NewTranscriptionsHandler(fake)

	body, ct := buildMultipart(t, "in.raw", "audio/pcm", pcm, nil)
	req := httptest.NewRequest(http.MethodPost, "/v1/audio/transcriptions", body)
	req.Header.Set("Content-Type", ct)
	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("status: got %d want 200; body=%s", rec.Code, rec.Body.String())
	}
	if fake.gotSR != 16000 {
		t.Fatalf("sample rate: got %d want 16000", fake.gotSR)
	}
	if len(fake.gotSamples) != 16000 {
		t.Fatalf("samples: got %d want 16000", len(fake.gotSamples))
	}
}

func TestTranscriptionsHandler_LibAV_MP3(t *testing.T) {
	if _, err := exec.LookPath("ffmpeg"); err != nil {
		t.Skip("ffmpeg not on PATH; skipping libav transcription test")
	}
	src := sine16k(440)
	mp3 := encodeViaFfmpeg(t, src, 16000, "mp3")

	fake := &fakeTranscriber{returnText: "mp3 ok"}
	h := stt.NewTranscriptionsHandler(fake)

	body, ct := buildMultipart(t, "in.mp3", "audio/mpeg", mp3, map[string]string{
		"response_format": "json",
	})
	req := httptest.NewRequest(http.MethodPost, "/v1/audio/transcriptions", body)
	req.Header.Set("Content-Type", ct)
	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("status: got %d want 200; body=%s", rec.Code, rec.Body.String())
	}
	if fake.gotSR != 16000 {
		t.Fatalf("sample rate: got %d want 16000", fake.gotSR)
	}
	if len(fake.gotSamples) < 12000 {
		t.Fatalf("decoded mp3 too short: %d samples", len(fake.gotSamples))
	}
}

func TestTranscriptionsHandler_MissingFile(t *testing.T) {
	fake := &fakeTranscriber{}
	h := stt.NewTranscriptionsHandler(fake)

	body := &bytes.Buffer{}
	mw := multipart.NewWriter(body)
	_ = mw.WriteField("response_format", "json")
	_ = mw.Close()

	req := httptest.NewRequest(http.MethodPost, "/v1/audio/transcriptions", body)
	req.Header.Set("Content-Type", mw.FormDataContentType())
	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, req)

	if rec.Code != http.StatusUnprocessableEntity {
		t.Fatalf("status: got %d want 422; body=%s", rec.Code, rec.Body.String())
	}
	if !strings.Contains(rec.Body.String(), "file") {
		t.Fatalf("expected error to mention 'file', got: %s", rec.Body.String())
	}
}

func TestTranscriptionsHandler_NilTranscriber(t *testing.T) {
	h := stt.NewTranscriptionsHandler(nil)
	body, ct := buildMultipart(t, "in.wav", "audio/wav",
		audio.EncodeWAVMono16(sine16k(440), 16000), nil)
	req := httptest.NewRequest(http.MethodPost, "/v1/audio/transcriptions", body)
	req.Header.Set("Content-Type", ct)
	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, req)
	if rec.Code != http.StatusServiceUnavailable {
		t.Fatalf("status: got %d want 503", rec.Code)
	}
}

func TestTranscriptionsHandler_GarbageBody(t *testing.T) {
	fake := &fakeTranscriber{}
	h := stt.NewTranscriptionsHandler(fake)
	body, ct := buildMultipart(t, "in.wav", "audio/wav",
		[]byte("not a wav, not an mp3, not anything"), nil)
	req := httptest.NewRequest(http.MethodPost, "/v1/audio/transcriptions", body)
	req.Header.Set("Content-Type", ct)
	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, req)
	if rec.Code != http.StatusBadRequest {
		t.Fatalf("status: got %d want 400; body=%s", rec.Code, rec.Body.String())
	}
}

func encodeViaFfmpeg(t *testing.T, samples audio.MonoF32, sampleRate int, ext string) []byte {
	t.Helper()
	tmpDir := t.TempDir()
	srcPath := filepath.Join(tmpDir, "src.wav")
	dstPath := filepath.Join(tmpDir, "out."+ext)
	if err := os.WriteFile(srcPath, audio.EncodeWAVMono16(samples, sampleRate), 0o644); err != nil {
		t.Fatalf("write src wav: %v", err)
	}
	cmd := exec.Command("ffmpeg", "-y", "-loglevel", "error", "-i", srcPath, dstPath)
	if out, err := cmd.CombinedOutput(); err != nil {
		t.Fatalf("ffmpeg encode %s: %v\n%s", ext, err, out)
	}
	b, err := os.ReadFile(dstPath)
	if err != nil {
		t.Fatalf("read encoded %s: %v", ext, err)
	}
	return b
}
