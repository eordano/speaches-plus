package eou

import (
	"encoding/json"
	"fmt"
	"math"
	"os"
	"sort"
	"strings"
	"unicode"
)

type Tokenizer struct {
	vocab       map[string]int
	idToToken   map[int]string
	merges      map[bigram]int
	addedTokens map[string]int
	specialTrie *specialNode
	imStartID   int
	imEndID     int
	hasImTokens bool
}

type bigram struct {
	left  string
	right string
}

type tokenizerJSON struct {
	AddedTokens []struct {
		ID      int    `json:"id"`
		Content string `json:"content"`
		Special bool   `json:"special"`
	} `json:"added_tokens"`
	Model struct {
		Type   string         `json:"type"`
		Vocab  map[string]int `json:"vocab"`
		Merges []interface{}  `json:"merges"`
	} `json:"model"`
}

func LoadTokenizer(path string) (*Tokenizer, error) {
	raw, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("tokenizer: read %q: %w", path, err)
	}
	var tj tokenizerJSON
	if err := json.Unmarshal(raw, &tj); err != nil {
		return nil, fmt.Errorf("tokenizer: parse %q: %w", path, err)
	}
	if tj.Model.Type != "BPE" && tj.Model.Type != "" {
		return nil, fmt.Errorf("tokenizer: unsupported model type %q (only BPE is supported)", tj.Model.Type)
	}

	t := &Tokenizer{
		vocab:       make(map[string]int, len(tj.Model.Vocab)),
		idToToken:   make(map[int]string, len(tj.Model.Vocab)),
		merges:      make(map[bigram]int, len(tj.Model.Merges)),
		addedTokens: make(map[string]int),
		imStartID:   -1,
		imEndID:     -1,
	}
	for tok, id := range tj.Model.Vocab {
		t.vocab[tok] = id
		t.idToToken[id] = tok
	}
	for rank, m := range tj.Model.Merges {
		l, r, ok := parseMerge(m)
		if !ok {
			continue
		}
		t.merges[bigram{l, r}] = rank
	}
	t.specialTrie = newSpecialTrie()
	for _, at := range tj.AddedTokens {
		t.addedTokens[at.Content] = at.ID
		t.idToToken[at.ID] = at.Content
		if _, exists := t.vocab[at.Content]; !exists {
			t.vocab[at.Content] = at.ID
		}
		t.specialTrie.insert(at.Content, at.ID)
		switch at.Content {
		case ImStart:
			t.imStartID = at.ID
		case ImEnd:
			t.imEndID = at.ID
		}
	}
	if t.imEndID < 0 {
		if id, ok := t.vocab[ImEnd]; ok {
			t.imEndID = id
		}
	}
	if t.imStartID < 0 {
		if id, ok := t.vocab[ImStart]; ok {
			t.imStartID = id
		}
	}
	t.hasImTokens = t.imEndID >= 0
	return t, nil
}

func parseMerge(raw interface{}) (string, string, bool) {
	switch v := raw.(type) {
	case string:
		i := strings.IndexByte(v, ' ')
		if i <= 0 || i >= len(v)-1 {
			return "", "", false
		}
		return v[:i], v[i+1:], true
	case []interface{}:
		if len(v) != 2 {
			return "", "", false
		}
		l, ok1 := v[0].(string)
		r, ok2 := v[1].(string)
		return l, r, ok1 && ok2
	}
	return "", "", false
}

func (t *Tokenizer) ImEndID() int      { return t.imEndID }
func (t *Tokenizer) ImStartID() int    { return t.imStartID }
func (t *Tokenizer) VocabSize() int    { return len(t.idToToken) }
func (t *Tokenizer) HasImTokens() bool { return t.hasImTokens }

func (t *Tokenizer) Encode(text string) []int {
	if text == "" {
		return nil
	}
	pieces := t.specialTrie.split(text)
	var out []int
	for _, p := range pieces {
		if p.special {
			out = append(out, p.id)
			continue
		}
		out = append(out, t.encodePlain(p.text)...)
	}
	return out
}

func (t *Tokenizer) encodePlain(text string) []int {
	if text == "" {
		return nil
	}
	splits := gpt2PreSplitChars(text)
	var out []int
	for _, s := range splits {
		bytes := bytesToBPETokens(s)
		merged := t.bpeMerges(bytes)
		for _, tok := range merged {
			if id, ok := t.vocab[tok]; ok {
				out = append(out, id)
			} else {
				for _, r := range tok {
					id, ok := t.vocab[string(r)]
					if !ok {
						continue
					}
					out = append(out, id)
				}
			}
		}
	}
	return out
}

func gpt2PreSplitChars(text string) []string {
	chars := []rune(text)
	n := len(chars)
	if n == 0 {
		return nil
	}
	out := make([]string, 0, n)
	i := 0
	for i < n {
		if chars[i] == '\'' && i+1 < n {
			next := chars[i+1]
			if next == 's' || next == 'd' || next == 'm' || next == 't' {
				out = append(out, string(chars[i:i+2]))
				i += 2
				continue
			}
			if i+2 < n {
				two := string(chars[i+1 : i+3])
				if two == "ll" || two == "ve" || two == "re" {
					out = append(out, string(chars[i:i+3]))
					i += 3
					continue
				}
			}
		}
		start := i
		leadingSpace := chars[i] == ' '
		probe := i
		if leadingSpace {
			probe = i + 1
		}
		if probe < n {
			c := chars[probe]
			if unicode.IsLetter(c) {
				j := probe
				for j < n && unicode.IsLetter(chars[j]) {
					j++
				}
				if j > probe {
					out = append(out, string(chars[start:j]))
					i = j
					continue
				}
			}
			if unicode.IsNumber(c) {
				j := probe
				for j < n && unicode.IsNumber(chars[j]) {
					j++
				}
				if j > probe {
					out = append(out, string(chars[start:j]))
					i = j
					continue
				}
			}
			if !unicode.IsSpace(c) && !unicode.IsLetter(c) && !unicode.IsNumber(c) {
				j := probe
				for j < n && !unicode.IsSpace(chars[j]) && !unicode.IsLetter(chars[j]) && !unicode.IsNumber(chars[j]) {
					j++
				}
				if j > probe {
					out = append(out, string(chars[start:j]))
					i = j
					continue
				}
			}
		}
		if unicode.IsSpace(chars[i]) {
			j := i
			for j < n && unicode.IsSpace(chars[j]) {
				j++
			}
			if j < n {
				takeTo := j
				if takeTo-1 > i {
					out = append(out, string(chars[i:takeTo-1]))
				}
				out = append(out, string(chars[takeTo-1:j+1]))
				i = j + 1
				continue
			}
			out = append(out, string(chars[i:j]))
			i = j
			continue
		}
		out = append(out, string(chars[i:i+1]))
		i++
	}
	return out
}

func (t *Tokenizer) bpeMerges(s string) []string {
	if s == "" {
		return nil
	}
	tokens := make([]string, 0, len(s))
	for _, r := range s {
		tokens = append(tokens, string(r))
	}
	for {
		bestRank := math.MaxInt
		bestIdx := -1
		for i := 0; i < len(tokens)-1; i++ {
			rank, ok := t.merges[bigram{tokens[i], tokens[i+1]}]
			if !ok {
				continue
			}
			if rank < bestRank {
				bestRank = rank
				bestIdx = i
			}
		}
		if bestIdx < 0 {
			break
		}
		merged := tokens[bestIdx] + tokens[bestIdx+1]
		tokens = append(tokens[:bestIdx], append([]string{merged}, tokens[bestIdx+2:]...)...)
	}
	return tokens
}

func (t *Tokenizer) Decode(ids []int) string {
	var sb strings.Builder
	for _, id := range ids {
		tok, ok := t.idToToken[id]
		if !ok {
			continue
		}
		sb.WriteString(tok)
	}
	return bpeTokensToString(sb.String())
}

func (t *Tokenizer) IDsByRank() []string {
	ids := make([]int, 0, len(t.idToToken))
	for id := range t.idToToken {
		ids = append(ids, id)
	}
	sort.Ints(ids)
	out := make([]string, 0, len(ids))
	for _, id := range ids {
		out = append(out, t.idToToken[id])
	}
	return out
}
