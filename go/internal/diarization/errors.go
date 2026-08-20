package diarization

import (
	"errors"
	"net/http"

	"github.com/eordano/speaches-plus-go/internal/oapi"
)

type apiErr struct {
	Status     int
	Msg        string
	Kind       string
	Param      string
	Code       string
	Validation *oapi.FastAPIErrorEntry
}

func (e *apiErr) Error() string { return e.Msg }

func writeAPIError(w http.ResponseWriter, err error) {
	var e *apiErr
	if !errors.As(err, &e) {
		oapi.WriteError(w, http.StatusInternalServerError, err.Error(), oapi.TypeServerError, "", "")
		return
	}
	if e.Validation != nil {
		oapi.WriteValidationError(w, *e.Validation)
		return
	}
	oapi.WriteError(w, e.Status, e.Msg, e.Kind, e.Param, e.Code)
}
