package eou

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func writeMockTokenizer(t *testing.T) string {
	t.Helper()
	vocab := map[string]int{}
	id := 0
	add := func(tok string) {
		if _, ok := vocab[tok]; ok {
			return
		}
		vocab[tok] = id
		id++
	}
	for r := rune(0); r < 0x100; r++ {
		add(string(byteToRune[r]))
	}
	add("Ġh")
	add("Ġhe")
	add("Ġhel")
	add("Ġhell")
	add("Ġhello")
	add("Ġworld")
	add("Ġfoo")
	add("Ġbar")

	merges := []string{
		"Ġ h",
		"Ġh e",
		"Ġhe l",
		"Ġhel l",
		"Ġhell o",
		"Ġ w",
		"Ġw o",
		"Ġwo r",
		"Ġwor l",
		"Ġworl d",
	}

	imStartID := id
	add(ImStart)
	imEndID := id
	add(ImEnd)

	addedTokens := []map[string]interface{}{
		{"id": imStartID, "content": ImStart, "special": true},
		{"id": imEndID, "content": ImEnd, "special": true},
	}

	doc := map[string]interface{}{
		"added_tokens": addedTokens,
		"model": map[string]interface{}{
			"type":   "BPE",
			"vocab":  vocab,
			"merges": merges,
		},
	}
	raw, err := json.Marshal(doc)
	if err != nil {
		t.Fatal(err)
	}
	dir := t.TempDir()
	path := filepath.Join(dir, "tokenizer.json")
	if err := os.WriteFile(path, raw, 0644); err != nil {
		t.Fatal(err)
	}
	return path
}

func TestTokenizer_LoadAndEncode_Basic(t *testing.T) {
	path := writeMockTokenizer(t)
	tok, err := LoadTokenizer(path)
	if err != nil {
		t.Fatal(err)
	}
	if tok.ImEndID() < 0 {
		t.Fatalf("ImEndID not detected")
	}
	if !tok.HasImTokens() {
		t.Fatalf("HasImTokens should be true")
	}

	ids := tok.Encode(" hello")
	if len(ids) == 0 {
		t.Fatalf("encode returned no ids")
	}
	helloID := tok.Encode(" hello world")
	if len(helloID) < 2 {
		t.Fatalf("encode multi-word: %v", helloID)
	}
}

func TestTokenizer_SplitsSpecialTokens(t *testing.T) {
	path := writeMockTokenizer(t)
	tok, err := LoadTokenizer(path)
	if err != nil {
		t.Fatal(err)
	}
	ids := tok.Encode(ImStart + "user\nhello" + ImEnd)
	foundImStart := false
	foundImEnd := false
	for _, id := range ids {
		if id == tok.ImStartID() {
			foundImStart = true
		}
		if id == tok.ImEndID() {
			foundImEnd = true
		}
	}
	if !foundImStart {
		t.Fatalf("ImStart not in encoded ids: %v", ids)
	}
	if !foundImEnd {
		t.Fatalf("ImEnd not in encoded ids: %v", ids)
	}
}

func TestTokenizer_DecodeRoundTrip(t *testing.T) {
	path := writeMockTokenizer(t)
	tok, err := LoadTokenizer(path)
	if err != nil {
		t.Fatal(err)
	}
	text := " hello world"
	ids := tok.Encode(text)
	got := tok.Decode(ids)
	if got != text {
		t.Fatalf("roundtrip: got %q want %q", got, text)
	}
}

func TestTokenizer_DecodeIncludesSpecialTokens(t *testing.T) {
	path := writeMockTokenizer(t)
	tok, err := LoadTokenizer(path)
	if err != nil {
		t.Fatal(err)
	}
	text := ImStart + "user\nhi" + ImEnd
	ids := tok.Encode(text)
	got := tok.Decode(ids)
	if !strings.Contains(got, ImStart) || !strings.Contains(got, ImEnd) {
		t.Fatalf("decode lost special tokens: %q", got)
	}
}

func TestTokenizer_Empty(t *testing.T) {
	path := writeMockTokenizer(t)
	tok, err := LoadTokenizer(path)
	if err != nil {
		t.Fatal(err)
	}
	if ids := tok.Encode(""); len(ids) != 0 {
		t.Fatalf("empty must return no ids; got %v", ids)
	}
}

func TestTokenizer_MissingFile(t *testing.T) {
	if _, err := LoadTokenizer("/nonexistent/tokenizer.json"); err == nil {
		t.Fatalf("missing file must error")
	}
}

func TestTokenizer_UnsupportedModel(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "tokenizer.json")
	doc := map[string]interface{}{
		"model": map[string]interface{}{"type": "WordPiece"},
	}
	raw, _ := json.Marshal(doc)
	_ = os.WriteFile(path, raw, 0644)
	if _, err := LoadTokenizer(path); err == nil {
		t.Fatalf("unsupported model type must error")
	}
}

func TestByteMap_RoundTrip(t *testing.T) {
	for b := 0; b < 256; b++ {
		r := byteToRune[b]
		if back, ok := runeToByte[r]; !ok || back != byte(b) {
			t.Fatalf("byte %d: r=%U ok=%v back=%d", b, r, ok, back)
		}
	}
}

func TestByteMap_BPETokensToString_RoundTrip(t *testing.T) {
	for _, s := range []string{
		"hello world",
		"line\nbreak",
		"\t\rweird",
		"\x00\x01\x02",
		"unicode: αβγ",
	} {
		got := bpeTokensToString(bytesToBPETokens(s))
		if got != s {
			t.Fatalf("roundtrip %q -> %q", s, got)
		}
	}
}
