package tts

var kokoroVocab map[rune]int

func init() {
	kokoroVocab = make(map[rune]int, kokoroVocabInitCap)
	idx := 0
	add := func(s string) {
		for _, r := range s {
			kokoroVocab[r] = idx
			idx++
		}
	}
	add(kokoroPad)
	add(kokoroPunctuation)
	add(kokoroLetters)
	add(kokoroLettersIPA)
}

func Tokenize(phonemes string) []int64 {
	tokens := make([]int64, 0, len(phonemes))
	for _, r := range phonemes {
		if id, ok := kokoroVocab[r]; ok {
			tokens = append(tokens, int64(id))
		}
	}
	return tokens
}

func CleanPhonemes(phonemes string) string {
	rep := []struct{ from, to string }{
		{"kəkˈoːɹoʊ", "kˈoʊkəɹoʊ"},
		{"kəkˈɔːɹəʊ", "kˈəʊkəɹəʊ"},
	}
	for _, r := range rep {
		phonemes = stringReplaceAll(phonemes, r.from, r.to)
	}
	mapped := make([]rune, 0, len(phonemes))
	for _, r := range phonemes {
		switch r {
		case 'ʲ':
			mapped = append(mapped, 'j')
		case 'r':
			mapped = append(mapped, 'ɹ')
		case 'x':
			mapped = append(mapped, 'k')
		case 'ɬ':
			mapped = append(mapped, 'l')
		default:
			mapped = append(mapped, r)
		}
	}
	out := string(mapped)
	clean := make([]rune, 0, len(out))
	for _, r := range out {
		if _, ok := kokoroVocab[r]; ok {
			clean = append(clean, r)
		}
	}
	return stringTrimSpace(string(clean))
}

func stringReplaceAll(s, old, new string) string {
	if old == "" {
		return s
	}
	out := ""
	for {
		i := indexOf(s, old)
		if i < 0 {
			return out + s
		}
		out += s[:i] + new
		s = s[i+len(old):]
	}
}

func indexOf(s, sub string) int {
	for i := 0; i+len(sub) <= len(s); i++ {
		if s[i:i+len(sub)] == sub {
			return i
		}
	}
	return -1
}

func stringTrimSpace(s string) string {
	start, end := 0, len(s)
	for start < end && (s[start] == ' ' || s[start] == '\t' || s[start] == '\n') {
		start++
	}
	for end > start && (s[end-1] == ' ' || s[end-1] == '\t' || s[end-1] == '\n') {
		end--
	}
	return s[start:end]
}
