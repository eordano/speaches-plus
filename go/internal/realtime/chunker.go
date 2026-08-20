package realtime

import "strings"

type sentenceChunker struct {
	buf    strings.Builder
	minLen int
}

func newSentenceChunker(minLen int) *sentenceChunker {
	if minLen <= 0 {
		minLen = 1
	}
	return &sentenceChunker{minLen: minLen}
}

func (c *sentenceChunker) feed(s string) []string {
	c.buf.WriteString(s)
	cur := c.buf.String()
	var out []string
	start := 0
	for i := 0; i < len(cur); i++ {
		switch cur[i] {
		case '.', '!', '?', '\n':
			end := i + 1
			next := byte(' ')
			if end < len(cur) {
				next = cur[end]
			}
			if end == len(cur) || next == ' ' || next == '\t' || next == '\n' {
				if end-start >= c.minLen {
					out = append(out, strings.TrimSpace(cur[start:end]))
					start = end
				}
			}
		}
	}
	if start > 0 {
		c.buf.Reset()
		c.buf.WriteString(cur[start:])
	}
	return out
}

func (c *sentenceChunker) flush() string {
	out := strings.TrimSpace(c.buf.String())
	c.buf.Reset()
	return out
}
