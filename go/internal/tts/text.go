package tts

import (
	"regexp"
	"strings"

	"github.com/eordano/speaches-plus-go/internal/punkt"
)

const MaxChunkChars = 400

var (
	emojiRe = regexp.MustCompile(
		"[\U0001F600-\U0001F64F" +
			"\U0001F300-\U0001F5FF" +
			"\U0001F680-\U0001F6FF" +
			"\U0001F700-\U0001F77F" +
			"\U0001F780-\U0001F7FF" +
			"\U0001F800-\U0001F8FF" +
			"\U0001F900-\U0001F9FF" +
			"\U0001FA00-\U0001FA6F" +
			"\U0001FA70-\U0001FAFF" +
			"✂-➰]+",
	)
	mdBoldRe        = regexp.MustCompile(`\*\*(.*?)\*\*`)
	mdItalicStarRe  = regexp.MustCompile(`\*(.*?)\*`)
	mdUnderRe       = regexp.MustCompile(`__(.*?)__`)
	mdItalicUnderRe = regexp.MustCompile(`_(.*?)_`)
	whitespaceRe    = regexp.MustCompile(`\s+`)
	newlineRe       = regexp.MustCompile(`[\r\n]+`)
	sentenceSplitRe = regexp.MustCompile(`(?:[.!?])\s+`)
)

func StripEmojis(s string) string {
	return emojiRe.ReplaceAllString(s, "")
}

func StripMarkdownEmphasis(s string) string {
	s = mdBoldRe.ReplaceAllString(s, "$1")
	s = mdItalicStarRe.ReplaceAllString(s, "$1")
	s = mdUnderRe.ReplaceAllString(s, "$1")
	s = mdItalicUnderRe.ReplaceAllString(s, "$1")
	return s
}

func NormalizeForTTS(s string) string {
	s = newlineRe.ReplaceAllString(s, " ")
	s = whitespaceRe.ReplaceAllString(s, " ")
	return strings.TrimSpace(s)
}

func SplitIntoChunks(text string, maxChars int) []string {
	if maxChars <= 0 {
		maxChars = MaxChunkChars
	}
	if len(text) <= maxChars {
		if text == "" {
			return nil
		}
		return []string{text}
	}

	sentences := splitSentences(text)
	chunks := []string{}
	current := ""
	flush := func() {
		if current != "" {
			chunks = append(chunks, strings.TrimSpace(current))
			current = ""
		}
	}

	for _, sentence := range sentences {
		if len(sentence) > maxChars {
			flush()
			for _, word := range strings.Fields(sentence) {
				if len(current)+len(word)+1 <= maxChars {
					if current == "" {
						current = word
					} else {
						current = current + " " + word
					}
				} else {
					flush()
					current = word
				}
			}
		} else if len(current)+len(sentence)+1 <= maxChars {
			if current == "" {
				current = sentence
			} else {
				current = current + " " + sentence
			}
		} else {
			flush()
			current = sentence
		}
	}
	flush()
	return chunks
}

func splitSentences(text string) []string {
	return punkt.English().Sentences(text)
}
