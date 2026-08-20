package pii

type PiiSpan struct {
	Start        int    `json:"start"`
	EndExclusive int    `json:"endExclusive"`
	Label        string `json:"label"`
}

func AssembleSpans(labels []string, offsets [][2]int, attentionMask []int) []PiiSpan {
	var out []PiiSpan
	var openLabel string
	openStart := -1
	openEnd := -1
	hasOpen := false

	closeSpan := func() {
		if hasOpen && openStart >= 0 && openEnd > openStart {
			out = append(out, PiiSpan{Start: openStart, EndExclusive: openEnd, Label: openLabel})
		}
		hasOpen = false
		openLabel = ""
		openStart = -1
		openEnd = -1
	}

	for i, tg := range labels {
		if i >= len(attentionMask) || attentionMask[i] == 0 {
			continue
		}
		s := offsets[i][0]
		e := offsets[i][1]
		if e <= s {
			continue
		}
		if tg == "O" {
			closeSpan()
			continue
		}
		dash := -1
		for k := 0; k < len(tg); k++ {
			if tg[k] == '-' {
				dash = k
				break
			}
		}
		if dash < 0 {
			closeSpan()
			continue
		}
		prefix := tg[:dash]
		cls := tg[dash+1:]

		switch prefix {
		case "B":
			closeSpan()
			openLabel = cls
			openStart = s
			openEnd = e
			hasOpen = true
		case "I":
			if hasOpen && openLabel == cls {
				openEnd = e
			} else {
				closeSpan()
				openLabel = cls
				openStart = s
				openEnd = e
				hasOpen = true
			}
		case "E":
			if hasOpen && openLabel == cls {
				openEnd = e
				closeSpan()
			} else {
				closeSpan()
				out = append(out, PiiSpan{Start: s, EndExclusive: e, Label: cls})
			}
		case "S":
			closeSpan()
			out = append(out, PiiSpan{Start: s, EndExclusive: e, Label: cls})
		default:
			closeSpan()
		}
	}

	closeSpan()
	if out == nil {
		return []PiiSpan{}
	}
	return out
}
