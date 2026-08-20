package eou

import "strings"

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
		matchID, matchLen, matchText := n.matchAt(text, i)
		if matchLen > 0 {
			flush()
			out = append(out, specialPiece{text: matchText, id: matchID, special: true})
			i += matchLen
			continue
		}
		plain.WriteByte(text[i])
		i++
	}
	flush()
	return out
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
