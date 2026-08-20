package tts

import (
	"reflect"
	"strings"
	"testing"
)

func TestStripEmojis(t *testing.T) {
	cases := []struct {
		in, want string
	}{
		{"hello 🌍 world", "hello  world"},
		{"😀😃😄 plain", " plain"},
		{"no emojis here", "no emojis here"},
		{"✂ scissors ✂", " scissors "},
	}
	for _, c := range cases {
		if got := StripEmojis(c.in); got != c.want {
			t.Errorf("StripEmojis(%q) = %q, want %q", c.in, got, c.want)
		}
	}
}

func TestStripMarkdownEmphasis(t *testing.T) {
	cases := []struct {
		in, want string
	}{
		{"**bold**", "bold"},
		{"*italic*", "italic"},
		{"__under__", "under"},
		{"_under_", "under"},
		{"plain text", "plain text"},
		{"a **bold** and *italic* mix", "a bold and italic mix"},
	}
	for _, c := range cases {
		if got := StripMarkdownEmphasis(c.in); got != c.want {
			t.Errorf("StripMarkdownEmphasis(%q) = %q, want %q", c.in, got, c.want)
		}
	}
}

func TestNormalizeForTTS(t *testing.T) {
	in := "  hello\n\nworld\t\ttest\r\n"
	want := "hello world test"
	if got := NormalizeForTTS(in); got != want {
		t.Errorf("NormalizeForTTS(%q) = %q, want %q", in, got, want)
	}
}

func TestSplitIntoChunks(t *testing.T) {
	t.Run("short text returns single chunk", func(t *testing.T) {
		got := SplitIntoChunks("short text", 100)
		want := []string{"short text"}
		if !reflect.DeepEqual(got, want) {
			t.Errorf("got %v, want %v", got, want)
		}
	})
	t.Run("empty returns nil", func(t *testing.T) {
		got := SplitIntoChunks("", 100)
		if got != nil {
			t.Errorf("got %v, want nil", got)
		}
	})
	t.Run("splits on sentence boundary", func(t *testing.T) {
		text := "First sentence. Second sentence. Third sentence."
		got := SplitIntoChunks(text, 25)
		for _, c := range got {
			if len(c) > 25 {
				t.Errorf("chunk too long: %q (len=%d)", c, len(c))
			}
		}
		joined := strings.Join(got, " ")
		if !strings.Contains(joined, "First sentence.") || !strings.Contains(joined, "Third sentence.") {
			t.Errorf("missing content in %v", got)
		}
	})
	t.Run("breaks single very-long sentence on words", func(t *testing.T) {
		long := strings.Repeat("word ", 200)
		got := SplitIntoChunks(long, 50)
		for _, c := range got {
			if len(c) > 50 {
				t.Errorf("chunk too long: %q (len=%d)", c, len(c))
			}
		}
	})
}
