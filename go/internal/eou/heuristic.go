package eou

import (
	"context"
	"strings"
	"time"
	"unicode"
)

type Heuristic struct{}

func NewHeuristic() *Heuristic { return &Heuristic{} }

func (h *Heuristic) Close() error { return nil }

func (h *Heuristic) Predict(ctx context.Context, req Request) (Verdict, error) {
	t0 := time.Now()
	score := h.score(req.Partial, req.Language)
	return Verdict{Score: score, Latency: time.Since(t0)}, nil
}

func (h *Heuristic) score(text, lang string) float32 {
	trimmed := strings.TrimSpace(text)
	if trimmed == "" {
		return heuristicScoreEmpty
	}
	if endsWithStrongTerminator(trimmed) {
		return heuristicScoreStrongTerminator
	}
	if endsWithSoftTerminator(trimmed) {
		return heuristicScoreSoftTerminator
	}
	last := lastWord(trimmed)
	if last == "" {
		return heuristicScoreEmptyLastWord
	}
	if isHesitation(last, lang) {
		return heuristicScoreHesitation
	}
	if isContinuation(last, lang) {
		return heuristicScoreContinuation
	}
	return heuristicScoreDefault
}

func endsWithStrongTerminator(s string) bool {
	if len(s) == 0 {
		return false
	}
	r := lastRune(s)
	switch r {
	case '.', '!', '?', '。', '！', '？', '…':
		return true
	}
	return false
}

func endsWithSoftTerminator(s string) bool {
	if len(s) == 0 {
		return false
	}
	r := lastRune(s)
	switch r {
	case ',', ';', ':', '-', '、', '，':
		return true
	}
	return false
}

func lastRune(s string) rune {
	var last rune
	for _, r := range s {
		last = r
	}
	return last
}

func lastWord(s string) string {
	rs := []rune(s)
	start := len(rs)
	for start > 0 {
		r := rs[start-1]
		if !unicode.IsLetter(r) && !unicode.IsDigit(r) && r != '\'' && r != '-' {
			break
		}
		start--
	}
	return strings.ToLower(string(rs[start:]))
}

var hesitationsByLang = map[string]map[string]struct{}{
	"en": setOf("um", "uh", "uhm", "er", "ah", "hmm", "like", "well"),
	"es": setOf("eh", "este", "pues", "bueno", "o", "sea"),
	"fr": setOf("euh", "hum", "ben", "alors", "donc"),
	"de": setOf("äh", "ähm", "also", "halt"),
	"it": setOf("ehm", "uhm", "cioè", "tipo"),
	"pt": setOf("é", "tipo", "tipo", "então"),
}

var continuationsByLang = map[string]map[string]struct{}{
	"en": setOf("and", "but", "or", "the", "a", "an", "to", "of", "in", "for", "with", "that", "which", "if", "because", "so"),
	"es": setOf("y", "pero", "o", "el", "la", "los", "las", "un", "una", "de", "en", "para", "con", "que", "si", "porque"),
	"fr": setOf("et", "mais", "ou", "le", "la", "les", "un", "une", "de", "en", "pour", "avec", "que", "si", "parce"),
	"de": setOf("und", "aber", "oder", "der", "die", "das", "ein", "eine", "von", "mit", "für", "dass", "wenn", "weil"),
	"it": setOf("e", "ma", "o", "il", "la", "lo", "un", "una", "di", "in", "per", "con", "che", "se", "perché"),
	"pt": setOf("e", "mas", "ou", "o", "a", "os", "as", "um", "uma", "de", "em", "para", "com", "que", "se", "porque"),
}

func setOf(words ...string) map[string]struct{} {
	m := make(map[string]struct{}, len(words))
	for _, w := range words {
		m[w] = struct{}{}
	}
	return m
}

func isHesitation(word, lang string) bool {
	if word == "" {
		return false
	}
	tbl, ok := hesitationsByLang[lang]
	if !ok {
		tbl = hesitationsByLang["en"]
	}
	_, found := tbl[word]
	return found
}

func isContinuation(word, lang string) bool {
	if word == "" {
		return false
	}
	tbl, ok := continuationsByLang[lang]
	if !ok {
		tbl = continuationsByLang["en"]
	}
	_, found := tbl[word]
	return found
}
