package diarization

import (
	"encoding/base64"
	"encoding/json"
	"fmt"
	"io"
	"log/slog"
	"mime/multipart"
	"net/http"
	"path/filepath"
	"strings"

	"github.com/eordano/speaches-plus-go/internal/audio"
	"github.com/eordano/speaches-plus-go/internal/oapi"
)

const (
	defaultFileID = "audio"
	formatJSON    = "json"
	formatRTTM    = "rttm"
	dataURLPrefix = "data:"
	base64Marker  = "base64"
)

type Handler struct {
	Seg *SegmentationModel
	Emb *EmbeddingModel
	Cfg Config
}

func NewHandler(seg *SegmentationModel, emb *EmbeddingModel, cfg Config) *Handler {
	return &Handler{Seg: seg, Emb: emb, Cfg: cfg}
}

func (h *Handler) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	if err := h.handle(w, r); err != nil {
		writeAPIError(w, err)
	}
}

func (h *Handler) handle(w http.ResponseWriter, r *http.Request) error {
	if h.Seg == nil || h.Emb == nil {
		return &apiErr{
			Status: http.StatusServiceUnavailable,
			Msg:    "diarization model not loaded; run scripts/fetch-models.sh and scripts/export-diarizen-onnx.py",
			Kind:   oapi.TypeServiceUnavail,
			Code:   "model_not_loaded",
		}
	}
	if err := r.ParseMultipartForm(maxUploadBytes); err != nil {
		return &apiErr{
			Status: http.StatusBadRequest,
			Msg:    "parse multipart: " + err.Error(),
			Kind:   oapi.TypeInvalidRequest,
			Code:   "multipart_parse_error",
		}
	}

	samples, fileHdr, err := h.readAudioFile(r)
	if err != nil {
		return err
	}

	known, err := h.embedKnownSpeakers(r)
	if err != nil {
		return err
	}

	d := NewDiarizer(h.Seg, h.Emb, h.Cfg)
	segments, err := d.DiarizeUtterance(samples, 0)
	if err != nil {
		slog.Error("diarize failed", "err", err)
		return &apiErr{
			Status: http.StatusInternalServerError,
			Msg:    "diarize: " + err.Error(),
			Kind:   oapi.TypeServerError,
			Code:   "diarize_failed",
		}
	}

	labelFor := h.buildSpeakerLabelMap(segments, samples, known)

	switch r.FormValue("response_format") {
	case formatRTTM:
		writeRTTM(w, segments, labelFor, fileIDFromHeader(fileHdr))
	default:
		writeJSON(w, segments, labelFor, float64(len(samples))/float64(frameSampleRate))
	}
	return nil
}

func (h *Handler) readAudioFile(r *http.Request) ([]float32, *multipart.FileHeader, error) {
	file, fileHdr, err := r.FormFile("file")
	if err != nil {
		return nil, nil, &apiErr{
			Validation: &oapi.FastAPIErrorEntry{
				Type: "missing", Loc: []string{"body", "file"}, Msg: "Field required",
			},
		}
	}
	defer file.Close()

	rawBytes, err := io.ReadAll(io.LimitReader(file, maxUploadBytes+1))
	if err != nil {
		return nil, nil, &apiErr{
			Status: http.StatusBadRequest,
			Msg:    "read file: " + err.Error(),
			Kind:   oapi.TypeInvalidRequest,
			Param:  "file",
			Code:   "file_read_error",
		}
	}
	if int64(len(rawBytes)) > maxUploadBytes {
		return nil, nil, &apiErr{
			Status: http.StatusRequestEntityTooLarge,
			Msg:    fmt.Sprintf("file exceeds maximum size of %d bytes", maxUploadBytes),
			Kind:   oapi.TypeInvalidRequest,
			Param:  "file",
			Code:   "file_too_large",
		}
	}

	contentType := ""
	if fileHdr != nil {
		contentType = fileHdr.Header.Get("Content-Type")
	}
	samples, err := audio.DecodeUploadedAudio(rawBytes, contentType)
	if err != nil {
		return nil, nil, &apiErr{
			Status: http.StatusBadRequest,
			Msg:    "audio decode: " + err.Error(),
			Kind:   oapi.TypeInvalidRequest,
			Param:  "file",
			Code:   "audio_decode_error",
		}
	}
	return []float32(samples), fileHdr, nil
}

func (h *Handler) embedKnownSpeakers(r *http.Request) ([]namedEmbedding, error) {
	names := multiFormValue(r, "known_speaker_names[]", "known_speaker_names")
	refs := multiFormValue(r, "known_speaker_references[]", "known_speaker_references")
	if len(names) == 0 || len(names) != len(refs) {
		return nil, nil
	}
	out := make([]namedEmbedding, 0, len(names))
	for i, name := range names {
		refBytes, refMime, err := decodeDataURL(refs[i])
		if err != nil {
			return nil, &apiErr{
				Status: http.StatusBadRequest,
				Msg:    fmt.Sprintf("known_speaker_references[%s]: %s", name, err.Error()),
				Kind:   oapi.TypeInvalidRequest,
				Param:  "known_speaker_references",
				Code:   "data_url_decode_error",
			}
		}
		refSamples, err := audio.DecodeUploadedAudio(refBytes, refMime)
		if err != nil {
			return nil, &apiErr{
				Status: http.StatusBadRequest,
				Msg:    fmt.Sprintf("known_speaker_references[%s] decode: %s", name, err.Error()),
				Kind:   oapi.TypeInvalidRequest,
				Param:  "known_speaker_references",
				Code:   "audio_decode_error",
			}
		}
		vec, err := h.Emb.Embed([]float32(refSamples))
		if err != nil {
			return nil, &apiErr{
				Status: http.StatusInternalServerError,
				Msg:    fmt.Sprintf("embed reference %s: %s", name, err.Error()),
				Kind:   oapi.TypeServerError,
				Code:   "embed_failed",
			}
		}
		out = append(out, namedEmbedding{Name: name, Vec: vec})
	}
	return out, nil
}

func (h *Handler) buildSpeakerLabelMap(segments []Segment, audioBuf []float32, known []namedEmbedding) func(ClusterID) string {
	clusterToName := make(map[ClusterID]string)
	if len(known) > 0 {
		for cid, pooled := range poolPerCluster(segments, audioBuf) {
			if len(pooled) < EmbeddingMinInputSamples {
				continue
			}
			vec, err := h.Emb.Embed(pooled)
			if err != nil {
				slog.Warn("diarize: per-cluster embed failed", "cluster", cid, "err", err)
				continue
			}
			if name, ok := closestKnown(vec, known); ok {
				clusterToName[cid] = name
			}
		}
	}
	return func(cid ClusterID) string {
		if name, ok := clusterToName[cid]; ok {
			return name
		}
		return fmt.Sprintf("SPEAKER_%02d", cid)
	}
}

func poolPerCluster(segments []Segment, audioBuf []float32) map[ClusterID][]float32 {
	out := make(map[ClusterID][]float32)
	for _, s := range segments {
		startIdx := int(s.TStartMs) * frameSampleRate / msPerSecond
		endIdx := int(s.TEndMs) * frameSampleRate / msPerSecond
		if endIdx > len(audioBuf) {
			endIdx = len(audioBuf)
		}
		if endIdx <= startIdx {
			continue
		}
		out[s.Speaker] = append(out[s.Speaker], audioBuf[startIdx:endIdx]...)
	}
	return out
}

func closestKnown(vec []float32, known []namedEmbedding) (string, bool) {
	bestName := ""
	bestSim := float32(-2)
	for _, k := range known {
		if s := CosineSim(vec, k.Vec); s > bestSim {
			bestSim = s
			bestName = k.Name
		}
	}
	return bestName, bestName != ""
}

func writeJSON(w http.ResponseWriter, segments []Segment, labelFor func(ClusterID) string, durationSec float64) {
	out := struct {
		Duration float64       `json:"duration"`
		Segments []segmentJSON `json:"segments"`
	}{
		Duration: durationSec,
		Segments: make([]segmentJSON, 0, len(segments)),
	}
	for _, s := range segments {
		out.Segments = append(out.Segments, segmentJSON{
			Start:   float64(s.TStartMs) / float64(msPerSecond),
			End:     float64(s.TEndMs) / float64(msPerSecond),
			Speaker: labelFor(s.Speaker),
		})
	}
	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(out)
}

func writeRTTM(w http.ResponseWriter, segments []Segment, labelFor func(ClusterID) string, fileID string) {
	w.Header().Set("Content-Type", "text/plain; charset=utf-8")
	w.WriteHeader(http.StatusOK)
	for _, s := range segments {
		start := float64(s.TStartMs) / float64(msPerSecond)
		dur := 0.0
		if s.TEndMs > s.TStartMs {
			dur = float64(s.TEndMs-s.TStartMs) / float64(msPerSecond)
		}
		fmt.Fprintf(w, "SPEAKER %s 1 %.3f %.3f <NA> <NA> %s <NA> <NA>\n",
			fileID, start, dur, labelFor(s.Speaker))
	}
}

func fileIDFromHeader(fileHdr *multipart.FileHeader) string {
	if fileHdr == nil || fileHdr.Filename == "" {
		return defaultFileID
	}
	base := filepath.Base(fileHdr.Filename)
	if ext := filepath.Ext(base); ext != "" {
		base = strings.TrimSuffix(base, ext)
	}
	if base == "" {
		return defaultFileID
	}
	return base
}

type segmentJSON struct {
	Start   float64 `json:"start"`
	End     float64 `json:"end"`
	Speaker string  `json:"speaker"`
}

type namedEmbedding struct {
	Name string
	Vec  []float32
}

func multiFormValue(r *http.Request, keys ...string) []string {
	if r.MultipartForm == nil {
		return nil
	}
	var out []string
	for _, k := range keys {
		if vs, ok := r.MultipartForm.Value[k]; ok {
			out = append(out, vs...)
		}
	}
	return out
}

func decodeDataURL(s string) ([]byte, string, error) {
	s = strings.TrimSpace(s)
	rest, ok := strings.CutPrefix(s, dataURLPrefix)
	if !ok {
		return nil, "", fmt.Errorf("not a data URL")
	}
	comma := strings.Index(rest, ",")
	if comma < 0 {
		return nil, "", fmt.Errorf("missing comma")
	}
	header, body := rest[:comma], rest[comma+1:]
	mime, isBase64 := parseDataURLHeader(header)
	if !isBase64 {
		return nil, mime, fmt.Errorf("only base64 data URLs are supported")
	}
	for _, enc := range []*base64.Encoding{
		base64.StdEncoding,
		base64.RawStdEncoding,
		base64.URLEncoding,
		base64.RawURLEncoding,
	} {
		if v, err := enc.DecodeString(body); err == nil {
			return v, mime, nil
		}
	}
	return nil, mime, fmt.Errorf("base64 decode failed")
}

func parseDataURLHeader(header string) (mime string, isBase64 bool) {
	for j, p := range strings.Split(header, ";") {
		if j == 0 && p != "" {
			mime = p
			continue
		}
		if strings.EqualFold(p, base64Marker) {
			isBase64 = true
		}
	}
	return mime, isBase64
}
