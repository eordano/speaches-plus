//go:build !tesseract

package pii

import "errors"

func RunOCR(_ []byte) (*OCRResult, error) {
	return nil, errors.New("OCR not available: build with -tags tesseract")
}
