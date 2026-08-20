package eou

import (
	"context"
	"testing"
)

func TestPredict_PriorTurnsArePassedToModel(t *testing.T) {
	captured := captureModel{}
	turns := []Turn{
		{Role: "user", Content: "what's your name"},
		{Role: "assistant", Content: "I'm Claude"},
	}
	_, err := captured.Predict(context.Background(), Request{
		Turns:    turns,
		Partial:  "and what's mine",
		Language: "en",
	})
	if err != nil {
		t.Fatal(err)
	}
	if len(captured.lastReq.Turns) != 2 {
		t.Fatalf("turns not propagated; got %d", len(captured.lastReq.Turns))
	}
	if captured.lastReq.Partial != "and what's mine" {
		t.Fatalf("partial not propagated: %q", captured.lastReq.Partial)
	}
	if captured.lastReq.Language != "en" {
		t.Fatalf("lang not propagated: %q", captured.lastReq.Language)
	}
}

func TestLoad_HeuristicReceivesCompleteRequest(t *testing.T) {
	m, _, err := Load(Config{})
	if err != nil {
		t.Fatal(err)
	}
	v, err := m.Predict(context.Background(), Request{
		Turns:    []Turn{{Role: "user", Content: "hi"}},
		Partial:  "yes please.",
		Language: "en",
	})
	if err != nil {
		t.Fatal(err)
	}
	if v.Score < 0.9 {
		t.Fatalf("strong terminator should yield high score; got %f", v.Score)
	}
}

func TestQwenChatPrompt_LooksRight(t *testing.T) {
	turns := []Turn{
		{Role: "user", Content: "hi"},
		{Role: "assistant", Content: "hello"},
	}
	got := FormatQwenChat(turns, "yes")
	want := "<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\nhello<|im_end|>\n<|im_start|>user\nyes"
	if got != want {
		t.Fatalf("got:\n%q\n\nwant:\n%q", got, want)
	}
}

type captureModel struct {
	lastReq Request
}

func (c *captureModel) Predict(ctx context.Context, req Request) (Verdict, error) {
	c.lastReq = req
	return Verdict{Score: 0.5}, nil
}
func (c *captureModel) Close() error { return nil }
