package oapi

import (
	"encoding/json"
	"net/http"
)

type ErrorEnvelope struct {
	Error ErrorBody `json:"error"`
}

type ErrorBody struct {
	Message string `json:"message"`
	Type    string `json:"type"`
	Param   string `json:"param,omitempty"`
	Code    string `json:"code,omitempty"`
}

func WriteError(w http.ResponseWriter, status int, msg, kind, param, code string) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(ErrorEnvelope{Error: ErrorBody{
		Message: msg, Type: kind, Param: param, Code: code,
	}})
}

type FastAPIValidationError struct {
	Detail []FastAPIErrorEntry `json:"detail"`
}

type FastAPIErrorEntry struct {
	Type  string   `json:"type"`
	Loc   []string `json:"loc"`
	Msg   string   `json:"msg"`
	Input any      `json:"input,omitempty"`
}

func WriteValidationError(w http.ResponseWriter, entries ...FastAPIErrorEntry) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(http.StatusUnprocessableEntity)
	_ = json.NewEncoder(w).Encode(FastAPIValidationError{Detail: entries})
}

const (
	TypeInvalidRequest = "invalid_request_error"
	TypeAuthError      = "authentication_error"
	TypeNotFound       = "not_found_error"
	TypeServerError    = "internal_server_error"
	TypeServiceUnavail = "service_unavailable_error"
)
