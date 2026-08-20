package pii

import (
	"encoding/json"
	"fmt"
	"math"
	"os"
	"strings"
	"unicode"
)

type piiTokenizer struct {
	vocab       map[string]int
	merges      map[bigram]int
	addedTokens map[string]int
	specialTrie *specialNode
	clsID       int
	sepID       int
	hasCLS      bool
	hasSEP      bool
}

type bigram struct {
	left  string
	right string
}

type encodingResult struct {
	IDs           []int
	Offsets       [][2]int
	AttentionMask []int
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
	PostProcessor *struct {
		Type          string `json:"type"`
		SingleSeq     []any  `json:"single"`
		PairSeq       []any  `json:"pair"`
		SpecialTokens []struct {
			ID      string `json:"id"`
			IDs     []int  `json:"ids"`
			Tokens  []string `json:"tokens"`
		} `json:"special_tokens"`
	} `json:"post_processor"`
}

func loadPiiTokenizer(path string) (*piiTokenizer, error) {
	raw, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("tokenizer: read %q: %w", path, err)
	}
	var tj tokenizerJSON
	if err := json.Unmarshal(raw, &tj); err != nil {
		return nil, fmt.Errorf("tokenizer: parse %q: %w", path, err)
	}

	t := &piiTokenizer{
		vocab:       make(map[string]int, len(tj.Model.Vocab)),
		merges:      make(map[bigram]int, len(tj.Model.Merges)),
		addedTokens: make(map[string]int),
		specialTrie: newSpecialTrie(),
		clsID:       -1,
		sepID:       -1,
	}

	for tok, id := range tj.Model.Vocab {
		t.vocab[tok] = id
	}
	for rank, m := range tj.Model.Merges {
		l, r, ok := parseMerge(m)
		if !ok {
			continue
		}
		t.merges[bigram{l, r}] = rank
	}
	for _, at := range tj.AddedTokens {
		t.addedTokens[at.Content] = at.ID
		if _, exists := t.vocab[at.Content]; !exists {
			t.vocab[at.Content] = at.ID
		}
		if at.Special {
			t.specialTrie.insert(at.Content, at.ID)
		}
		switch at.Content {
		case "[CLS]":
			t.clsID = at.ID
			t.hasCLS = true
		case "[SEP]":
			t.sepID = at.ID
			t.hasSEP = true
		}
	}
	if !t.hasCLS {
		if id, ok := t.vocab["[CLS]"]; ok {
			t.clsID = id
			t.hasCLS = true
		}
	}
	if !t.hasSEP {
		if id, ok := t.vocab["[SEP]"]; ok {
			t.sepID = id
			t.hasSEP = true
		}
	}
	return t, nil
}

func (t *piiTokenizer) Encode(text string) encodingResult {
	if text == "" {
		return encodingResult{}
	}

	var ids []int
	var offsets [][2]int
	var mask []int

	if t.hasCLS {
		ids = append(ids, t.clsID)
		offsets = append(offsets, [2]int{0, 0})
		mask = append(mask, 1)
	}

	pieces := t.specialTrie.split(text)
	cursor := 0
	for _, p := range pieces {
		if p.special {
			ids = append(ids, p.id)
			offsets = append(offsets, [2]int{0, 0})
			mask = append(mask, 1)
			continue
		}

		start := strings.Index(text[cursor:], p.text)
		if start >= 0 {
			cursor += start
		}

		splits := preSplitWords(p.text)
		wordCursor := 0
		for _, word := range splits {
			wordStart := strings.Index(p.text[wordCursor:], word)
			if wordStart >= 0 {
				wordCursor += wordStart
			}
			charStart := cursor + wordCursor

			bpeTokens := t.bpeEncode(word)
			charPos := charStart
			for _, tok := range bpeTokens {
				tokLen := tokenCharLen(tok)
				ids = append(ids, t.lookupID(tok))
				offsets = append(offsets, [2]int{charPos, charPos + tokLen})
				mask = append(mask, 1)
				charPos += tokLen
			}
			wordCursor += len(word)
		}
		cursor += len(p.text)
	}

	if t.hasSEP {
		ids = append(ids, t.sepID)
		offsets = append(offsets, [2]int{0, 0})
		mask = append(mask, 1)
	}

	return encodingResult{
		IDs:           ids,
		Offsets:       offsets,
		AttentionMask: mask,
	}
}

func (t *piiTokenizer) bpeEncode(word string) []string {
	bpeStr := bytesToBPETokens(word)
	runes := []rune(bpeStr)
	if len(runes) == 0 {
		return nil
	}
	tokens := make([]string, len(runes))
	for i, r := range runes {
		tokens[i] = string(r)
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
		newTokens := make([]string, 0, len(tokens)-1)
		newTokens = append(newTokens, tokens[:bestIdx]...)
		newTokens = append(newTokens, merged)
		newTokens = append(newTokens, tokens[bestIdx+2:]...)
		tokens = newTokens
	}
	return tokens
}

func (t *piiTokenizer) lookupID(tok string) int {
	if id, ok := t.vocab[tok]; ok {
		return id
	}
	return 0
}

func tokenCharLen(tok string) int {
	n := 0
	for range tok {
		n++
	}
	return n
}

func preSplitWords(text string) []string {
	runes := []rune(text)
	n := len(runes)
	if n == 0 {
		return nil
	}
	var out []string
	i := 0
	for i < n {
		if runes[i] == '\'' && i+1 < n {
			next := runes[i+1]
			if next == 's' || next == 'd' || next == 'm' || next == 't' {
				out = append(out, string(runes[i:i+2]))
				i += 2
				continue
			}
			if i+2 < n {
				two := string(runes[i+1 : i+3])
				if two == "ll" || two == "ve" || two == "re" {
					out = append(out, string(runes[i:i+3]))
					i += 3
					continue
				}
			}
		}
		start := i
		leadingSpace := runes[i] == ' '
		probe := i
		if leadingSpace {
			probe = i + 1
		}
		if probe < n {
			c := runes[probe]
			if unicode.IsLetter(c) {
				j := probe
				for j < n && unicode.IsLetter(runes[j]) {
					j++
				}
				if j > probe {
					out = append(out, string(runes[start:j]))
					i = j
					continue
				}
			}
			if unicode.IsNumber(c) {
				j := probe
				for j < n && unicode.IsNumber(runes[j]) {
					j++
				}
				if j > probe {
					out = append(out, string(runes[start:j]))
					i = j
					continue
				}
			}
			if !unicode.IsSpace(c) && !unicode.IsLetter(c) && !unicode.IsNumber(c) {
				j := probe
				for j < n && !unicode.IsSpace(runes[j]) && !unicode.IsLetter(runes[j]) && !unicode.IsNumber(runes[j]) {
					j++
				}
				if j > probe {
					out = append(out, string(runes[start:j]))
					i = j
					continue
				}
			}
		}
		if unicode.IsSpace(runes[i]) {
			j := i
			for j < n && unicode.IsSpace(runes[j]) {
				j++
			}
			if j < n {
				takeTo := j
				if takeTo-1 > i {
					out = append(out, string(runes[i:takeTo-1]))
				}
				out = append(out, string(runes[takeTo-1:j+1]))
				i = j + 1
				continue
			}
			out = append(out, string(runes[i:j]))
			i = j
			continue
		}
		out = append(out, string(runes[i:i+1]))
		i++
	}
	return out
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

var byteToRune = func() [256]rune {
	var m [256]rune
	keep := func(b int) bool {
		return (b >= '!' && b <= '~') || (b >= 0xA1 && b <= 0xAC) || (b >= 0xAE && b <= 0xFF)
	}
	bs := []int{}
	for b := 0; b < 256; b++ {
		if keep(b) {
			bs = append(bs, b)
		}
	}
	cs := make([]int, len(bs))
	copy(cs, bs)
	n := 0
	for b := 0; b < 256; b++ {
		if !keep(b) {
			bs = append(bs, b)
			cs = append(cs, 256+n)
			n++
		}
	}
	for i, b := range bs {
		m[b] = rune(cs[i])
	}
	return m
}()

var runeToByte = func() map[rune]byte {
	m := make(map[rune]byte, 256)
	for b, r := range byteToRune {
		m[r] = byte(b)
	}
	return m
}()

func bytesToBPETokens(s string) string {
	out := make([]rune, 0, len(s))
	for i := 0; i < len(s); i++ {
		out = append(out, byteToRune[s[i]])
	}
	return string(out)
}

type specialNode struct {
	children map[byte]*specialNode
	terminal bool
	id       int
	content  string
}

type specialPiece struct {
	text    string
	id      int
	special bool
}

func newSpecialTrie() *specialNode {
	return &specialNode{children: map[byte]*specialNode{}}
}

func (n *specialNode) insert(s string, id int) {
	cur := n
	for i := 0; i < len(s); i++ {
		c := s[i]
		next, ok := cur.children[c]
		if !ok {
			next = newSpecialTrie()
			cur.children[c] = next
		}
		cur = next
	}
	cur.terminal = true
	cur.id = id
	cur.content = s
}

func (n *specialNode) split(text string) []specialPiece {
	var out []specialPiece
	if text == "" {
		return nil
	}
	plain := strings.Builder{}
	flush := func() {
		if plain.Len() > 0 {
			out = append(out, specialPiece{text: plain.String()})
			plain.Reset()
		}
	}
	i := 0
	for i < len(text) {
		_, matchLen, matchText := n.matchAt(text, i)
		if matchLen > 0 {
			flush()
			id := 0
			if at, ok := n.lookupID(matchText); ok {
				id = at
			}
			out = append(out, specialPiece{text: matchText, id: id, special: true})
			i += matchLen
			continue
		}
		plain.WriteByte(text[i])
		i++
	}
	flush()
	return out
}

func (n *specialNode) lookupID(content string) (int, bool) {
	cur := n
	for i := 0; i < len(content); i++ {
		next, ok := cur.children[content[i]]
		if !ok {
			return 0, false
		}
		cur = next
	}
	if cur.terminal {
		return cur.id, true
	}
	return 0, false
}

func (n *specialNode) matchAt(text string, start int) (int, int, string) {
	cur := n
	matchID := -1
	matchEnd := 0
	matchText := ""
	for i := start; i < len(text); i++ {
		next, ok := cur.children[text[i]]
		if !ok {
			break
		}
		cur = next
		if cur.terminal {
			matchID = cur.id
			matchEnd = i + 1
			matchText = cur.content
		}
	}
	if matchEnd == 0 {
		return -1, 0, ""
	}
	return matchID, matchEnd - start, matchText
}
