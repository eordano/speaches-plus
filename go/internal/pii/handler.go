package pii

import (
	"encoding/json"
	"net/http"
)

const maxBatchSize = 32

type classifyRequest struct {
	Text string `json:"text"`
}

type classifyResponse struct {
	Spans []PiiSpan `json:"spans"`
}

type batchRequest struct {
	Texts []string `json:"texts"`
}

type batchResponse struct {
	Results []classifyResponse `json:"results"`
}

func ClassifyHandler(c *Classifier) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		var req classifyRequest
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			writeJSON(w, http.StatusBadRequest, map[string]string{"error": "invalid JSON"})
			return
		}
		spans, err := c.ClassifyOne(req.Text)
		if err != nil {
			writeJSON(w, http.StatusInternalServerError, map[string]string{"error": err.Error()})
			return
		}
		writeJSON(w, http.StatusOK, classifyResponse{Spans: spans})
	}
}

func ClassifyBatchHandler(c *Classifier) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		var req batchRequest
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			writeJSON(w, http.StatusBadRequest, map[string]string{"error": "invalid JSON"})
			return
		}
		if len(req.Texts) > maxBatchSize {
			writeJSON(w, http.StatusRequestEntityTooLarge,
				map[string]string{"error": "batch size exceeds maximum of 32"})
			return
		}
		results, err := c.ClassifyBatch(req.Texts)
		if err != nil {
			writeJSON(w, http.StatusInternalServerError, map[string]string{"error": err.Error()})
			return
		}
		resp := batchResponse{Results: make([]classifyResponse, len(results))}
		for i, spans := range results {
			resp.Results[i] = classifyResponse{Spans: spans}
		}
		writeJSON(w, http.StatusOK, resp)
	}
}

func writeJSON(w http.ResponseWriter, status int, v interface{}) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(v)
}
