package tts

import "testing"

func TestTokenize_KnownPhrase(t *testing.T) {
	got := Tokenize("ðˈʌ kwˈɪk bɹˈaʊn")
	want := []int64{81, 156, 138, 16, 53, 65, 156, 102, 53, 16, 44, 123, 156, 43, 135, 56}
	if len(got) != len(want) {
		t.Fatalf("len mismatch: got %d want %d (got=%v)", len(got), len(want), got)
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("token[%d]=%d want %d (got=%v want=%v)", i, got[i], want[i], got, want)
		}
	}
	if id, ok := kokoroVocab['$']; !ok || id != 0 {
		t.Fatalf("pad: want 0, got %d (ok=%v)", id, ok)
	}
	if id, ok := kokoroVocab[' ']; !ok || id != 16 {
		t.Fatalf("space: want 16, got %d (ok=%v)", id, ok)
	}
}
