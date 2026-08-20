package punkt

import (
	"strings"
	"unicode"
)

const (
	orthoBegUC uint8 = 1 << 1
	orthoMidUC uint8 = 1 << 2
	orthoUnkUC uint8 = 1 << 3
	orthoBegLC uint8 = 1 << 4
	orthoMidLC uint8 = 1 << 5
	orthoUnkLC uint8 = 1 << 6
	orthoUC    uint8 = orthoBegUC | orthoMidUC | orthoUnkUC
	orthoLC    uint8 = orthoBegLC | orthoMidLC | orthoUnkLC
)

type Params struct {
	AbbrevTypes  map[string]bool
	Collocations map[[2]string]bool
	SentStarters map[string]bool
	OrthoContext map[string]uint8
}

type token struct {
	text        string
	start       int
	end         int
	typ         string
	periodFinal bool
	sentbreak   bool
	abbr        bool
	ellipsis    bool
}

func (t *token) typeNoPeriod() string {
	if len(t.typ) > 1 && strings.HasSuffix(t.typ, ".") {
		return t.typ[:len(t.typ)-1]
	}
	return t.typ
}

func (t *token) typeNoSentPeriod() string {
	if t.sentbreak {
		return t.typeNoPeriod()
	}
	return t.typ
}

func (t *token) firstUpper() bool {
	for _, r := range t.text {
		return unicode.IsUpper(r)
	}
	return false
}

func (t *token) firstLower() bool {
	for _, r := range t.text {
		return unicode.IsLower(r)
	}
	return false
}

func (t *token) isEllipsis() bool {
	if t.text == "…" {
		return true
	}
	dots := 0
	for _, r := range t.text {
		if r == '.' {
			dots++
		} else if r != ' ' {
			return false
		}
	}
	return dots >= 2
}

func (t *token) isInitial() bool {
	runes := []rune(t.text)
	return len(runes) == 2 && unicode.IsLetter(runes[0]) && runes[1] == '.'
}

func (t *token) isNumber() bool {
	return strings.HasPrefix(t.typ, "##number##")
}

func isNumeric(s string) bool {
	runes := []rune(s)
	i := 0
	if i < len(runes) && runes[i] == '-' {
		i++
	}
	if i < len(runes) && (runes[i] == '.' || runes[i] == ',') {
		i++
	}
	if i >= len(runes) || runes[i] < '0' || runes[i] > '9' {
		return false
	}
	i++
	for i < len(runes) {
		r := runes[i]
		if (r >= '0' && r <= '9') || r == ',' || r == '.' || r == '-' {
			i++
		} else {
			return false
		}
	}
	return true
}

func newToken(text string, start, end int) token {
	lower := strings.ToLower(text)
	typ := lower
	if isNumeric(lower) {
		typ = "##number##"
	}
	return token{
		text:        text,
		start:       start,
		end:         end,
		typ:         typ,
		periodFinal: strings.HasSuffix(text, "."),
	}
}

const nonWordChars = "?!)\";}]*:@'({["
const wordStartExclude = "(\"`{[:;&#*@)}]-,"

func isNonWord(r rune) bool  { return strings.ContainsRune(nonWordChars, r) }
func isExcluded(r rune) bool { return strings.ContainsRune(wordStartExclude, r) }

type posRune struct {
	off int
	r   rune
}

func tokenizeText(text string) []token {
	var out []token
	chars := make([]posRune, 0, len(text))
	for off, r := range text {
		chars = append(chars, posRune{off, r})
	}
	n := len(chars)
	i := 0
	for i < n {
		c := chars[i].r
		if unicode.IsSpace(c) {
			i++
			continue
		}
		startI := i
		switch {
		case (c == '-' || c == '.') && i+1 < n && chars[i+1].r == c:
			for i < n && chars[i].r == c {
				i++
			}
		case c == '…':
			i++
		case !isExcluded(c):
			i++
			for i < n {
				d := chars[i].r
				if unicode.IsSpace(d) || isNonWord(d) || d == '…' {
					break
				}
				if (d == '-' || d == '.') && i+1 < n && chars[i+1].r == d {
					break
				}
				if d == ',' {
					if i+1 >= n {
						break
					}
					x := chars[i+1].r
					if unicode.IsSpace(x) || isNonWord(x) {
						break
					}
				}
				i++
			}
		default:
			i++
		}
		sb := chars[startI].off
		eb := len(text)
		if i < n {
			eb = chars[i].off
		}
		out = append(out, newToken(text[sb:eb], sb, eb))
	}
	return out
}

func firstPass(t *token, p *Params) {
	switch t.text {
	case ".", "!", "?":
		t.sentbreak = true
		return
	}
	if t.isEllipsis() {
		t.ellipsis = true
		return
	}
	if t.periodFinal && !strings.HasSuffix(t.text, "..") {
		base := strings.ToLower(t.text[:len(t.text)-1])
		lastDash := base
		if idx := strings.LastIndex(base, "-"); idx >= 0 {
			lastDash = base[idx+1:]
		}
		if p.AbbrevTypes[base] || p.AbbrevTypes[lastDash] {
			t.abbr = true
		} else {
			t.sentbreak = true
		}
	}
}

func orthoHeuristic(p *Params, t *token) (bool, bool) {
	switch t.text {
	case ";", ":", ",", ".", "!", "?":
		return false, true
	}
	ortho := p.OrthoContext[t.typeNoSentPeriod()]
	if t.firstUpper() && ortho&orthoLC != 0 && ortho&orthoMidUC == 0 {
		return true, true
	}
	if t.firstLower() && (ortho&orthoUC != 0 || ortho&orthoBegLC == 0) {
		return false, true
	}
	return false, false
}

func secondPass(t1 *token, t2 *token, p *Params) {
	if !t1.periodFinal {
		return
	}
	typ := t1.typeNoPeriod()
	nextTyp := t2.typeNoSentPeriod()
	tokIsInitial := t1.isInitial()

	if p.Collocations[[2]string{typ, nextTyp}] {
		t1.sentbreak = false
		t1.abbr = true
		return
	}

	if (t1.abbr || t1.ellipsis) && !tokIsInitial {
		starter, known := orthoHeuristic(p, t2)
		if known && starter {
			t1.sentbreak = true
			return
		}
		if t2.firstUpper() && p.SentStarters[nextTyp] {
			t1.sentbreak = true
			return
		}
	}

	if tokIsInitial || typ == "##number##" {
		starter, known := orthoHeuristic(p, t2)
		if known && !starter {
			t1.sentbreak = false
			t1.abbr = true
		} else if !known && tokIsInitial && t2.firstUpper() &&
			p.OrthoContext[nextTyp]&orthoLC == 0 {
			t1.sentbreak = false
			t1.abbr = true
		}
	}
}

type Range struct {
	Start int
	End   int
}

type Segmenter struct {
	params *Params
}

func NewSegmenter(p *Params) *Segmenter {
	return &Segmenter{params: p}
}

func (s *Segmenter) SentenceRanges(text string) []Range {
	toks := tokenizeText(text)
	for i := range toks {
		firstPass(&toks[i], s.params)
	}
	for i := 0; i+1 < len(toks); i++ {
		secondPass(&toks[i], &toks[i+1], s.params)
	}
	var ranges []Range
	start := -1
	lastEnd := 0
	for i := range toks {
		if start < 0 {
			start = toks[i].start
		}
		lastEnd = toks[i].end
		if toks[i].sentbreak {
			ranges = append(ranges, Range{start, lastEnd})
			start = -1
		}
	}
	if start >= 0 {
		ranges = append(ranges, Range{start, lastEnd})
	}
	return realign(text, ranges)
}

func (s *Segmenter) Sentences(text string) []string {
	ranges := s.SentenceRanges(text)
	out := make([]string, 0, len(ranges))
	for _, r := range ranges {
		out = append(out, text[r.Start:r.End])
	}
	return out
}

func isCloser(r rune) bool {
	switch r {
	case '"', '\'', ')', ']', '}', '”', '’':
		return true
	}
	return false
}

func realign(text string, ranges []Range) []Range {
	i := 0
	for i+1 < len(ranges) {
		next := ranges[i+1]
		p := next.Start
		for p < next.End {
			r, size := decodeRune(text[p:])
			if !isCloser(r) {
				break
			}
			p += size
		}
		if p > next.Start {
			afterOK := p >= len(text) || strings.HasPrefix(text[p:], "--")
			if !afterOK {
				r, _ := decodeRune(text[p:])
				afterOK = unicode.IsSpace(r)
			}
			if afterOK {
				ranges[i].End = p
				q := p
				for q < next.End {
					r, size := decodeRune(text[q:])
					if !unicode.IsSpace(r) {
						break
					}
					q += size
				}
				if q >= next.End {
					ranges = append(ranges[:i+1], ranges[i+2:]...)
					continue
				}
				ranges[i+1].Start = q
			}
		}
		i++
	}
	return ranges
}

func decodeRune(s string) (rune, int) {
	for _, r := range s {
		return r, len(string(r))
	}
	return ' ', 1
}
