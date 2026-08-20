package stt

import (
	"encoding/json"
	"fmt"
	"io"
	"log/slog"
	"net/http"

	"github.com/eordano/speaches-plus-go/internal/audio"
	"github.com/eordano/speaches-plus-go/internal/diarization"
	"github.com/eordano/speaches-plus-go/internal/oapi"
)

const (
	multipartParseLimit = 100 << 20
	fileReadLimit       = 200 << 20
	formatText          = "text"
	formatJSON          = "json"
	formatDiarizedJSON  = "diarized_json"
	contentTypeText     = "text/plain; charset=utf-8"
	contentTypeJSON     = "application/json"
)

type TranscriptionsHandler struct {
	t       Transcriber
	diarSeg *diarization.SegmentationModel
	diarEmb *diarization.EmbeddingModel
	diarCfg diarization.Config
}

func NewTranscriptionsHandler(t Transcriber) *TranscriptionsHandler {
	return &TranscriptionsHandler{t: t}
}

func (h *TranscriptionsHandler) WithDiarization(seg *diarization.SegmentationModel, emb *diarization.EmbeddingModel, cfg diarization.Config) *TranscriptionsHandler {
	h.diarSeg = seg
	h.diarEmb = emb
	h.diarCfg = cfg
	return h
}

func (h *TranscriptionsHandler) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	if h.t == nil {
		oapi.WriteError(w, http.StatusServiceUnavailable,
			"transcriber not configured", oapi.TypeServiceUnavail, "", "")
		return
	}
	if err := r.ParseMultipartForm(multipartParseLimit); err != nil {
		oapi.WriteError(w, http.StatusBadRequest,
			"parse multipart: "+err.Error(), oapi.TypeInvalidRequest, "", "multipart_parse_error")
		return
	}
	file, fileHdr, err := r.FormFile("file")
	if err != nil {
		oapi.WriteValidationError(w, oapi.FastAPIErrorEntry{
			Type: "missing", Loc: []string{"body", "file"}, Msg: "Field required",
		})
		return
	}
	defer file.Close()

	contentType := ""
	if fileHdr != nil {
		contentType = fileHdr.Header.Get("Content-Type")
	}
	rawBytes, err := io.ReadAll(io.LimitReader(file, fileReadLimit+1))
	if err != nil {
		oapi.WriteError(w, http.StatusBadRequest,
			"read file: "+err.Error(), oapi.TypeInvalidRequest, "file", "file_read_error")
		return
	}
	if int64(len(rawBytes)) > fileReadLimit {
		oapi.WriteError(w, http.StatusRequestEntityTooLarge,
			fmt.Sprintf("file exceeds maximum size of %d bytes", fileReadLimit),
			oapi.TypeInvalidRequest, "file", "file_too_large")
		return
	}
	samples, err := audio.DecodeUploadedAudio(rawBytes, contentType)
	if err != nil {
		oapi.WriteError(w, http.StatusBadRequest,
			"audio decode: "+err.Error(), oapi.TypeInvalidRequest, "file", "audio_decode_error")
		return
	}

	format := r.FormValue("response_format")
	if format == formatDiarizedJSON {
		h.serveDiarized(w, samples)
		return
	}

	text, err := TranscribeLong(h.t, samples, audio.SampleRate16k)
	if err != nil {
		slog.Error("transcription failed", "err", err)
		oapi.WriteError(w, http.StatusInternalServerError,
			"transcribe: "+err.Error(), oapi.TypeServerError, "", "transcribe_failed")
		return
	}

	switch format {
	case "", formatText:
		w.Header().Set("Content-Type", contentTypeText)
		_, _ = fmt.Fprint(w, text)
	case formatJSON:
		w.Header().Set("Content-Type", contentTypeJSON)
		_, _ = fmt.Fprintf(w, `{"text":%q}`, text)
	default:
		oapi.WriteError(w, http.StatusBadRequest,
			fmt.Sprintf("unsupported response_format: %q (supported: text, json, diarized_json)", format),
			oapi.TypeInvalidRequest, "response_format", "unsupported_value")
	}
}

func (h *TranscriptionsHandler) serveDiarized(w http.ResponseWriter, samples []float32) {
	segT, ok := h.t.(SegmentTranscriber)
	if !ok {
		oapi.WriteError(w, http.StatusServiceUnavailable,
			"diarized_json requires a transcriber that exposes Whisper segments",
			oapi.TypeServiceUnavail, "response_format", "unsupported_backend")
		return
	}
	res, err := TranscribeSegmentsLong(segT, samples, audio.SampleRate16k)
	if err != nil {
		slog.Error("diarized transcription failed", "err", err)
		oapi.WriteError(w, http.StatusInternalServerError,
			"transcribe: "+err.Error(), oapi.TypeServerError, "", "transcribe_failed")
		return
	}

	var diarSegs []diarization.Segment
	if h.diarSeg != nil && h.diarEmb != nil {
		d := diarization.NewDiarizer(h.diarSeg, h.diarEmb, h.diarCfg)
		ds, err := d.DiarizeUtterance(samples, 0)
		if err != nil {
			slog.Warn("diarize failed; returning whisper segments only", "err", err)
		} else {
			diarSegs = ds
		}
	}

	body := buildDiarizedResponse(res, diarSegs)
	w.Header().Set("Content-Type", contentTypeJSON)
	_ = json.NewEncoder(w).Encode(body)
}

type diarizedSegment struct {
	Type         string   `json:"type"`
	ID           string   `json:"id"`
	Speaker      *string  `json:"speaker"`
	Start        float64  `json:"start"`
	End          float64  `json:"end"`
	Duration     float64  `json:"duration"`
	Text         string   `json:"text"`
	AvgLogprob   *float32 `json:"avg_logprob"`
	NoSpeechProb *float32 `json:"no_speech_prob"`
	Confidence   *float32 `json:"confidence"`
}

type diarizedResponse struct {
	Text         string            `json:"text"`
	AvgLogprob   *float32          `json:"avg_logprob"`
	NoSpeechProb *float32          `json:"no_speech_prob"`
	Segments     []diarizedSegment `json:"segments"`
}

func buildDiarizedResponse(stt Result, diar []diarization.Segment) diarizedResponse {
	resp := diarizedResponse{
		Text:         stt.Text,
		AvgLogprob:   stt.AvgLogprob,
		NoSpeechProb: stt.NoSpeechProb,
	}
	if len(diar) == 0 {

		resp.Segments = make([]diarizedSegment, 0, len(stt.Segments))
		for i, s := range stt.Segments {
			resp.Segments = append(resp.Segments, diarizedSegment{
				Type:         "transcript.text.segment",
				ID:           fmt.Sprintf("seg_%03d", i+1),
				Start:        float64(s.TStartMs) / 1000.0,
				End:          float64(s.TEndMs) / 1000.0,
				Duration:     float64(s.TEndMs-s.TStartMs) / 1000.0,
				Text:         s.Text,
				AvgLogprob:   s.AvgLogprob,
				NoSpeechProb: s.NoSpeechProb,
			})
		}
		return resp
	}

	buckets := make([][]int, len(diar))
	for wi, ws := range stt.Segments {
		mid := (uint64(ws.TStartMs) + uint64(ws.TEndMs)) / 2
		idx := -1
		for di, d := range diar {
			if mid >= d.TStartMs && mid <= d.TEndMs {
				idx = di
				break
			}
		}
		if idx < 0 {
			idx = nearestDiarIdx(diar, mid)
		}
		buckets[idx] = append(buckets[idx], wi)
	}

	resp.Segments = make([]diarizedSegment, 0, len(diar))
	for di, d := range diar {
		assigned := make([]Segment, 0, len(buckets[di]))
		for _, wi := range buckets[di] {
			assigned = append(assigned, stt.Segments[wi])
		}
		text := joinSegmentText(assigned)
		lp, nsp := aggregateSegmentStats(assigned)
		speaker := fmt.Sprintf("SPEAKER_%02d", d.Speaker)
		conf := d.Confidence
		resp.Segments = append(resp.Segments, diarizedSegment{
			Type:         "transcript.text.segment",
			ID:           fmt.Sprintf("seg_%03d", di+1),
			Speaker:      &speaker,
			Start:        float64(d.TStartMs) / 1000.0,
			End:          float64(d.TEndMs) / 1000.0,
			Duration:     float64(d.TEndMs-d.TStartMs) / 1000.0,
			Text:         text,
			AvgLogprob:   lp,
			NoSpeechProb: nsp,
			Confidence:   &conf,
		})
	}
	return resp
}

func nearestDiarIdx(diar []diarization.Segment, mid uint64) int {
	best := 0
	var bestDist uint64 = ^uint64(0)
	for i, d := range diar {
		var dist uint64
		switch {
		case mid < d.TStartMs:
			dist = d.TStartMs - mid
		case mid > d.TEndMs:
			dist = mid - d.TEndMs
		default:
			dist = 0
		}
		if dist < bestDist {
			best = i
			bestDist = dist
		}
	}
	return best
}

func joinSegmentText(segs []Segment) string {
	var out string
	for _, s := range segs {
		t := s.Text
		if t == "" {
			continue
		}
		if out != "" {
			out += " "
		}
		out += t
	}
	return out
}

func aggregateSegmentStats(segs []Segment) (*float32, *float32) {
	var lpSum, lpW, nspSum, nspW float64
	for _, s := range segs {
		dur := float64(s.TEndMs - s.TStartMs)
		if dur < 1 {
			dur = 1
		}
		if s.AvgLogprob != nil {
			lpSum += float64(*s.AvgLogprob) * dur
			lpW += dur
		}
		if s.NoSpeechProb != nil {
			nspSum += float64(*s.NoSpeechProb) * dur
			nspW += dur
		}
	}
	var lp, nsp *float32
	if lpW > 0 {
		v := float32(lpSum / lpW)
		lp = &v
	}
	if nspW > 0 {
		v := float32(nspSum / nspW)
		nsp = &v
	}
	return lp, nsp
}
