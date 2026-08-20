//go:build tesseract

package pii

import (
	"fmt"
	"strings"

	"github.com/otiai10/gosseract/v2"
)

func RunOCR(imgBytes []byte) (*OCRResult, error) {
	client := gosseract.NewClient()
	defer client.Close()

	if err := client.SetImageFromBytes(imgBytes); err != nil {
		return nil, fmt.Errorf("ocr: set image: %w", err)
	}

	boxes, err := client.GetBoundingBoxesVerbose()
	if err != nil {
		return nil, fmt.Errorf("ocr: get boxes: %w", err)
	}

	var textBuilder strings.Builder
	tokens := make([]OCRToken, 0, len(boxes))

	for i, box := range boxes {
		if i > 0 {
			textBuilder.WriteByte(' ')
		}
		start := textBuilder.Len()
		textBuilder.WriteString(box.Word)
		end := textBuilder.Len()

		tokens = append(tokens, OCRToken{
			Start:        start,
			EndExclusive: end,
			Rect: Rect{
				Left:   box.Box.Min.X,
				Top:    box.Box.Min.Y,
				Right:  box.Box.Max.X,
				Bottom: box.Box.Max.Y,
			},
		})
	}

	return &OCRResult{
		Text:   textBuilder.String(),
		Tokens: tokens,
	}, nil
}
