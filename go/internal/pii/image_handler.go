package pii

import (
	"encoding/json"
	"image/color"
	"io"
	"net/http"
)

const imageUploadLimit = 50 << 20

type analyzeTokenJSON struct {
	Start        int  `json:"start"`
	EndExclusive int  `json:"endExclusive"`
	Rect         Rect `json:"rect"`
}

type analyzeResponse struct {
	Text   string             `json:"text"`
	Tokens []analyzeTokenJSON `json:"tokens"`
	Spans  []PiiSpan          `json:"spans"`
	Rects  []LabeledRect      `json:"rects"`
}

func AnalyzeHandler(c *Classifier) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if err := r.ParseMultipartForm(imageUploadLimit); err != nil {
			writeJSON(w, http.StatusBadRequest, map[string]string{"error": "parse multipart: " + err.Error()})
			return
		}
		file, _, err := r.FormFile("file")
		if err != nil {
			writeJSON(w, http.StatusBadRequest, map[string]string{"error": "missing file field"})
			return
		}
		defer file.Close()

		imgBytes, err := io.ReadAll(io.LimitReader(file, imageUploadLimit))
		if err != nil {
			writeJSON(w, http.StatusBadRequest, map[string]string{"error": "read file: " + err.Error()})
			return
		}

		ocrResult, err := RunOCR(imgBytes)
		if err != nil {
			writeJSON(w, http.StatusInternalServerError, map[string]string{"error": err.Error()})
			return
		}

		spans, err := c.ClassifyOne(ocrResult.Text)
		if err != nil {
			writeJSON(w, http.StatusInternalServerError, map[string]string{"error": "classify: " + err.Error()})
			return
		}

		img, err := DecodeImage(imgBytes)
		if err != nil {
			writeJSON(w, http.StatusBadRequest, map[string]string{"error": "decode image: " + err.Error()})
			return
		}
		bounds := img.Bounds()
		rects := MapSpansToRects(ocrResult.Tokens, spans, bounds.Dx(), bounds.Dy())

		tokens := make([]analyzeTokenJSON, len(ocrResult.Tokens))
		for i, t := range ocrResult.Tokens {
			tokens[i] = analyzeTokenJSON{
				Start:        t.Start,
				EndExclusive: t.EndExclusive,
				Rect:         t.Rect,
			}
		}

		writeJSON(w, http.StatusOK, analyzeResponse{
			Text:   ocrResult.Text,
			Tokens: tokens,
			Spans:  spans,
			Rects:  rects,
		})
	}
}

func RenderHandler() http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if err := r.ParseMultipartForm(imageUploadLimit); err != nil {
			writeJSON(w, http.StatusBadRequest, map[string]string{"error": "parse multipart: " + err.Error()})
			return
		}
		file, _, err := r.FormFile("file")
		if err != nil {
			writeJSON(w, http.StatusBadRequest, map[string]string{"error": "missing file field"})
			return
		}
		defer file.Close()

		imgBytes, err := io.ReadAll(io.LimitReader(file, imageUploadLimit))
		if err != nil {
			writeJSON(w, http.StatusBadRequest, map[string]string{"error": "read file: " + err.Error()})
			return
		}

		rectsJSON := r.FormValue("rects")
		if rectsJSON == "" {
			writeJSON(w, http.StatusBadRequest, map[string]string{"error": "missing rects field"})
			return
		}

		var rects []RenderRect
		if err := json.Unmarshal([]byte(rectsJSON), &rects); err != nil {
			writeJSON(w, http.StatusBadRequest, map[string]string{"error": "invalid rects JSON: " + err.Error()})
			return
		}

		fillMode := r.FormValue("fill_mode")
		if fillMode == "" {
			fillMode = "solid"
		}

		fillColorStr := r.FormValue("fill_color")
		if fillColorStr == "" {
			fillColorStr = "#000000"
		}

		var fillColor color.Color
		fillColor, err = ParseHexColor(fillColorStr)
		if err != nil {
			writeJSON(w, http.StatusBadRequest, map[string]string{"error": err.Error()})
			return
		}

		img, err := DecodeImage(imgBytes)
		if err != nil {
			writeJSON(w, http.StatusBadRequest, map[string]string{"error": "decode image: " + err.Error()})
			return
		}

		pngBytes, err := RenderRedactions(img, rects, fillMode, fillColor)
		if err != nil {
			writeJSON(w, http.StatusInternalServerError, map[string]string{"error": err.Error()})
			return
		}

		w.Header().Set("Content-Type", "image/png")
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write(pngBytes)
	}
}

func UnavailableHandler(msg string) http.HandlerFunc {
	return func(w http.ResponseWriter, _ *http.Request) {
		writeJSON(w, http.StatusServiceUnavailable, map[string]string{"error": msg})
	}
}
