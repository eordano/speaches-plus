package eou

import (
	"strings"
	"testing"
)

func TestFormatQwenChat_SingleUserPartial(t *testing.T) {
	got := FormatQwenChat(nil, "hello world")
	want := "<|im_start|>user\nhello world"
	if got != want {
		t.Fatalf("got %q want %q", got, want)
	}
}

func TestFormatQwenChat_PriorTurnsThenPartial(t *testing.T) {
	turns := []Turn{
		{Role: "user", Content: "what's the weather"},
		{Role: "assistant", Content: "it's sunny"},
	}
	got := FormatQwenChat(turns, "and humid")
	expected := "<|im_start|>user\nwhat's the weather<|im_end|>\n" +
		"<|im_start|>assistant\nit's sunny<|im_end|>\n" +
		"<|im_start|>user\nand humid"
	if got != expected {
		t.Fatalf("got:\n%s\nwant:\n%s", got, expected)
	}
}

func TestFormatQwenChat_EmptyPartial(t *testing.T) {
	turns := []Turn{{Role: "user", Content: "hi"}}
	got := FormatQwenChat(turns, "")
	if !strings.HasSuffix(got, "<|im_end|>\n") {
		t.Fatalf("must end with <|im_end|>\\n; got %q", got)
	}
}

func TestFormatQwenChat_DefaultsRoleToUser(t *testing.T) {
	got := FormatQwenChat([]Turn{{Content: "no role"}}, "")
	if !strings.Contains(got, "<|im_start|>user\nno role") {
		t.Fatalf("missing default role; got %q", got)
	}
}

func TestRollingHistory_Truncates(t *testing.T) {
	turns := []Turn{
		{Role: "user", Content: "1"},
		{Role: "assistant", Content: "2"},
		{Role: "user", Content: "3"},
		{Role: "assistant", Content: "4"},
		{Role: "user", Content: "5"},
		{Role: "assistant", Content: "6"},
		{Role: "user", Content: "7"},
	}
	got := RollingHistory(turns, 4)
	if len(got) != 4 {
		t.Fatalf("len: %d", len(got))
	}
	if got[0].Content != "4" || got[3].Content != "7" {
		t.Fatalf("contents: %+v", got)
	}
}

func TestRollingHistory_Passthrough(t *testing.T) {
	turns := []Turn{{Content: "a"}, {Content: "b"}}
	got := RollingHistory(turns, 5)
	if len(got) != 2 {
		t.Fatalf("len: %d", len(got))
	}
}
