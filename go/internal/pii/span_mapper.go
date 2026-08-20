package pii

import "sort"

type LabeledRect struct {
	Left   int    `json:"left"`
	Top    int    `json:"top"`
	Right  int    `json:"right"`
	Bottom int    `json:"bottom"`
	Label  string `json:"label"`
}

func MapSpansToRects(tokens []OCRToken, spans []PiiSpan, imgWidth, imgHeight int) []LabeledRect {
	if len(spans) == 0 || len(tokens) == 0 {
		return []LabeledRect{}
	}

	var out []LabeledRect
	for _, span := range spans {
		var rects []Rect
		for _, tok := range tokens {
			if tok.EndExclusive <= span.Start || tok.Start >= span.EndExclusive {
				continue
			}
			rects = append(rects, tok.Rect)
		}
		if len(rects) == 0 {
			continue
		}

		sort.Slice(rects, func(i, j int) bool {
			if rects[i].Top != rects[j].Top {
				return rects[i].Top < rects[j].Top
			}
			return rects[i].Left < rects[j].Left
		})

		merged := mergeRects(rects)

		for _, r := range merged {
			r = padRect(r, 4)
			r = clampRect(r, imgWidth, imgHeight)
			out = append(out, LabeledRect{
				Left:   r.Left,
				Top:    r.Top,
				Right:  r.Right,
				Bottom: r.Bottom,
				Label:  span.Label,
			})
		}
	}

	if out == nil {
		return []LabeledRect{}
	}
	return out
}

func mergeRects(rects []Rect) []Rect {
	if len(rects) == 0 {
		return nil
	}
	merged := []Rect{rects[0]}
	for i := 1; i < len(rects); i++ {
		last := &merged[len(merged)-1]
		cur := rects[i]
		if shouldMerge(*last, cur) {
			last.Left = minInt(last.Left, cur.Left)
			last.Top = minInt(last.Top, cur.Top)
			last.Right = maxInt(last.Right, cur.Right)
			last.Bottom = maxInt(last.Bottom, cur.Bottom)
		} else {
			merged = append(merged, cur)
		}
	}
	return merged
}

func shouldMerge(a, b Rect) bool {
	hGap := b.Left - a.Right
	if hGap >= 8 {
		return false
	}

	overlapTop := maxInt(a.Top, b.Top)
	overlapBot := minInt(a.Bottom, b.Bottom)
	overlap := overlapBot - overlapTop
	if overlap <= 0 {
		return false
	}

	hA := a.Bottom - a.Top
	hB := b.Bottom - b.Top
	minH := minInt(hA, hB)
	if minH <= 0 {
		return false
	}

	return overlap*2 > minH
}

func padRect(r Rect, pad int) Rect {
	return Rect{
		Left:   r.Left - pad,
		Top:    r.Top - pad,
		Right:  r.Right + pad,
		Bottom: r.Bottom + pad,
	}
}

func clampRect(r Rect, w, h int) Rect {
	if r.Left < 0 {
		r.Left = 0
	}
	if r.Top < 0 {
		r.Top = 0
	}
	if r.Right > w {
		r.Right = w
	}
	if r.Bottom > h {
		r.Bottom = h
	}
	return r
}

func minInt(a, b int) int {
	if a < b {
		return a
	}
	return b
}

func maxInt(a, b int) int {
	if a > b {
		return a
	}
	return b
}
