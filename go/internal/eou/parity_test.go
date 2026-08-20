package eou

import (
	"encoding/json"
	"os"
	"path/filepath"
	"reflect"
	"testing"
)

func rustParityTokenizer(t *testing.T) *Tokenizer {
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
	for _, e := range []string{"Ġh", "Ġhe", "Ġhel", "Ġhell", "Ġhello", "Ġworld"} {
		add(e)
	}
	imStartID := id
	add(ImStart)
	imEndID := id
	add(ImEnd)

	merges := []string{
		"Ġ h", "Ġh e", "Ġhe l", "Ġhel l", "Ġhell o",
		"Ġ w", "Ġw o", "Ġwo r", "Ġwor l", "Ġworl d",
	}
	added := []map[string]interface{}{
		{"id": imStartID, "content": ImStart, "special": true},
		{"id": imEndID, "content": ImEnd, "special": true},
	}
	doc := map[string]interface{}{
		"added_tokens": added,
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
	if err := os.WriteFile(path, raw, 0o644); err != nil {
		t.Fatal(err)
	}
	tok, err := LoadTokenizer(path)
	if err != nil {
		t.Fatal(err)
	}
	return tok
}

func TestEncodeMatchesRustGoldenIDs(t *testing.T) {
	tok := rustParityTokenizer(t)
	cases := []struct {
		input string
		want  []int
	}{
		{"", nil},
		{" hello", []int{260}},
		{" hello world", []int{260, 261}},
		{"  hello", []int{32, 256, 101, 108, 108, 111}},
		{"a   b", []int{97, 32, 32, 32, 98}},
		{"a b c", []int{97, 32, 98, 32, 99}},
		{
			"I'm don't can't won't they're we've I'll",
			[]int{73, 39, 109, 32, 100, 111, 110, 39, 116, 32, 99, 97, 110, 39, 116, 32, 119, 111, 110, 39, 116, 32, 116, 104, 101, 121, 39, 114, 101, 32, 119, 101, 39, 118, 101, 32, 73, 39, 108, 108},
		},
		{
			"<|im_start|>user\nhello<|im_end|>",
			[]int{262, 117, 115, 101, 114, 10, 104, 101, 108, 108, 111, 263},
		},
		{"abc123def 456", []int{97, 98, 99, 49, 50, 51, 100, 101, 102, 32, 52, 53, 54}},
	}
	for _, c := range cases {
		got := tok.Encode(c.input)
		if !reflect.DeepEqual(got, c.want) {
			t.Errorf("Encode(%q) = %v\n  want %v", c.input, got, c.want)
		}
	}
}

func TestPreSplitMatchesRustGolden(t *testing.T) {
	cases := []struct {
		input string
		want  []string
	}{
		{"", nil},
		{" hello", []string{" hello"}},
		{" hello world", []string{" hello", " world"}},
		{"  hello", []string{" ", " h", "ello"}},
		{"a   b", []string{"a", "  ", " b"}},
		{"a b c", []string{"a", " b", " c"}},
		{
			"I'm don't can't won't they're we've I'll",
			[]string{"I", "'m", " don", "'t", " can", "'t", " won", "'t", " they", "'re", " we", "'ve", " I", "'ll"},
		},
		{
			"<|im_start|>user\nhello<|im_end|>",
			[]string{"<|", "im", "_", "start", "|>", "user", "\nh", "ello", "<|", "im", "_", "end", "|>"},
		},
		{"abc123def 456", []string{"abc", "123", "def", " 456"}},
	}
	for _, c := range cases {
		got := gpt2PreSplitChars(c.input)
		if !reflect.DeepEqual(got, c.want) {
			t.Errorf("gpt2PreSplitChars(%q) = %#v\n  want %#v", c.input, got, c.want)
		}
	}
}

func TestRollingHistoryMatchesRust(t *testing.T) {
	turns := make([]Turn, 0, 7)
	for i := 1; i <= 7; i++ {
		turns = append(turns, Turn{Role: "user", Content: string(rune('0' + i))})
	}
	got := RollingHistory(turns, 4)
	if len(got) != 4 {
		t.Fatalf("len=%d want 4", len(got))
	}
	if got[0].Content != "4" || got[3].Content != "7" {
		t.Fatalf("bounds wrong: got[0]=%q got[3]=%q", got[0].Content, got[3].Content)
	}
}

func TestFormatQwenChatGolden(t *testing.T) {
	if got := FormatQwenChat(nil, "hello world"); got != "<|im_start|>user\nhello world" {
		t.Fatalf("single-partial: %q", got)
	}
	turns := []Turn{
		{Role: "user", Content: "what's the weather"},
		{Role: "assistant", Content: "it's sunny"},
	}
	want := "<|im_start|>user\nwhat's the weather<|im_end|>\n" +
		"<|im_start|>assistant\nit's sunny<|im_end|>\n" +
		"<|im_start|>user\nand humid"
	if got := FormatQwenChat(turns, "and humid"); got != want {
		t.Fatalf("multi-turn:\n got %q\nwant %q", got, want)
	}
	got := FormatQwenChat([]Turn{{Role: "", Content: "no role"}}, "")
	wantRolePrefix := "<|im_start|>user\nno role"
	if !contains(got, wantRolePrefix) {
		t.Fatalf("default-role: got %q want contains %q", got, wantRolePrefix)
	}
}

func contains(haystack, needle string) bool {
	return len(haystack) >= len(needle) && (haystack == needle || indexString(haystack, needle) >= 0)
}

func indexString(s, sub string) int {
	if len(sub) == 0 {
		return 0
	}
	if len(sub) > len(s) {
		return -1
	}
	for i := 0; i+len(sub) <= len(s); i++ {
		if s[i:i+len(sub)] == sub {
			return i
		}
	}
	return -1
}
