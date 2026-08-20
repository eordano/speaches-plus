package punkt

import (
	"reflect"
	"strings"
	"testing"
)

func naiveSplit(text string) []string {
	var out []string
	var cur strings.Builder
	runes := []rune(text)
	for i := 0; i < len(runes); i++ {
		cur.WriteRune(runes[i])
		if runes[i] == '.' || runes[i] == '!' || runes[i] == '?' {
			if i+1 >= len(runes) || runes[i+1] == ' ' {
				out = append(out, strings.TrimSpace(cur.String()))
				cur.Reset()
			}
		}
	}
	if strings.TrimSpace(cur.String()) != "" {
		out = append(out, strings.TrimSpace(cur.String()))
	}
	return out
}

func TestAbbreviationTrap(t *testing.T) {
	text := "Dr. Smith went to Washington. He arrived."
	got := English().Sentences(text)
	want := []string{"Dr. Smith went to Washington.", "He arrived."}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("got %q want %q", got, want)
	}
	if len(naiveSplit(text)) != 3 {
		t.Fatalf("naive should split after Dr.")
	}
}

func TestDecimalNumbers(t *testing.T) {
	text := "It costs 3.50 today. Tomorrow it will cost more."
	got := English().Sentences(text)
	want := []string{"It costs 3.50 today.", "Tomorrow it will cost more."}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("got %q want %q", got, want)
	}
}

func TestInitials(t *testing.T) {
	text := "J. R. R. Tolkien wrote it. Many people read it."
	got := English().Sentences(text)
	want := []string{"J. R. R. Tolkien wrote it.", "Many people read it."}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("got %q want %q", got, want)
	}
	if len(naiveSplit(text)) != 5 {
		t.Fatalf("naive should split after each initial, got %q", naiveSplit(text))
	}
}

func TestEllipsis(t *testing.T) {
	text := "I waited for a long time... They came at last."
	got := English().Sentences(text)
	want := []string{"I waited for a long time...", "They came at last."}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("got %q want %q", got, want)
	}
	text2 := "He paused... and went on speaking."
	got2 := English().Sentences(text2)
	if len(got2) != 1 {
		t.Fatalf("mid-sentence ellipsis should not break: %q", got2)
	}
}

func TestQuotesAfterPeriod(t *testing.T) {
	text := `He said "Stop." Then he left the room.`
	got := English().Sentences(text)
	want := []string{`He said "Stop."`, "Then he left the room."}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("got %q want %q", got, want)
	}
	if len(naiveSplit(text)) != 1 {
		t.Fatalf("naive should miss the quote-final break")
	}
}

func TestRangesMonotonic(t *testing.T) {
	text := "Dr. Smith met Mr. Jones at 4.30 p.m. on Jan. 5. They talked... Then everyone went home! Was it late? Yes."
	ranges := English().SentenceRanges(text)
	if len(ranges) == 0 {
		t.Fatal("no ranges")
	}
	prev := 0
	for _, r := range ranges {
		if r.Start < prev || r.End <= r.Start || r.End > len(text) {
			t.Fatalf("bad range %+v", r)
		}
		prev = r.End
	}
}
