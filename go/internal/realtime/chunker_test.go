package realtime

import (
	"strings"
	"testing"
)

func TestSentenceChunker_StreamingDeltas(t *testing.T) {
	c := newSentenceChunker(2)
	var got []string
	for _, d := range []string{"Hello", " world", ". How", " are", " you?", " Fine", "."} {
		got = append(got, c.feed(d)...)
	}
	if tail := c.flush(); tail != "" {
		got = append(got, tail)
	}
	want := []string{"Hello world.", "How are you?", "Fine."}
	if strings.Join(got, "|") != strings.Join(want, "|") {
		t.Fatalf("got %v want %v", got, want)
	}
}

func TestSentenceChunker_NoBoundary(t *testing.T) {
	c := newSentenceChunker(2)
	var got []string
	for _, d := range []string{"hello ", "world without ", "punctuation"} {
		got = append(got, c.feed(d)...)
	}
	if tail := c.flush(); tail != "" {
		got = append(got, tail)
	}
	want := []string{"hello world without punctuation"}
	if strings.Join(got, "|") != strings.Join(want, "|") {
		t.Fatalf("got %v want %v", got, want)
	}
}

func TestSentenceChunker_DotInsideNumber(t *testing.T) {
	c := newSentenceChunker(2)
	var got []string
	for _, d := range []string{"price ", "is 3.14", " dollars."} {
		got = append(got, c.feed(d)...)
	}
	if tail := c.flush(); tail != "" {
		got = append(got, tail)
	}
	want := []string{"price is 3.14 dollars."}
	if strings.Join(got, "|") != strings.Join(want, "|") {
		t.Fatalf("got %v want %v", got, want)
	}
}
