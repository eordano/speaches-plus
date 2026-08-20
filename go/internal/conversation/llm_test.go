package conversation

import (
	"context"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"
)

func TestLLM_StreamSSE_Basic(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		flusher, _ := w.(http.Flusher)
		for _, c := range []string{"hello", " ", "world"} {
			fmt.Fprintf(w, "data: {\"choices\":[{\"delta\":{\"content\":%q}}]}\n\n", c)
			flusher.Flush()
		}
		fmt.Fprintf(w, "data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"stop\"}]}\n\n")
		fmt.Fprintf(w, "data: [DONE]\n\n")
		flusher.Flush()
	}))
	defer srv.Close()

	c := NewLLM(srv.URL+"/v1", "")
	out, err := c.Complete(context.Background(), "any", "hi")
	if err != nil {
		t.Fatalf("Complete: %v", err)
	}
	if out != "hello world" {
		t.Fatalf("got %q want %q", out, "hello world")
	}
}

func TestLLM_StreamSSE_DeltasObservable(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		flusher, _ := w.(http.Flusher)
		for _, c := range []string{"a", "bc", "def"} {
			fmt.Fprintf(w, "data: {\"choices\":[{\"delta\":{\"content\":%q}}]}\n\n", c)
			flusher.Flush()
		}
		fmt.Fprintf(w, "data: [DONE]\n\n")
		flusher.Flush()
	}))
	defer srv.Close()

	c := NewLLM(srv.URL+"/v1", "")
	stream, err := c.Stream(context.Background(), "any", "x")
	if err != nil {
		t.Fatalf("Stream: %v", err)
	}
	var got []string
	for d := range stream {
		if d.Err != nil {
			t.Fatalf("delta err: %v", d.Err)
		}
		if d.Done {
			break
		}
		got = append(got, d.Content)
	}
	want := []string{"a", "bc", "def"}
	if strings.Join(got, "|") != strings.Join(want, "|") {
		t.Fatalf("deltas: got %v want %v", got, want)
	}
}

func TestLLM_Upstream5xx(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusInternalServerError)
		_, _ = w.Write([]byte("simulated upstream failure"))
	}))
	defer srv.Close()

	c := NewLLM(srv.URL+"/v1", "fake-key")
	_, err := c.Complete(context.Background(), "any", "hello")
	if err == nil {
		t.Fatal("expected error on 5xx, got nil")
	}
	if !strings.Contains(err.Error(), "HTTP 500") {
		t.Fatalf("error doesn't mention status: %v", err)
	}
}

func TestLLM_RateLimited429(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusTooManyRequests)
		_, _ = w.Write([]byte("rate limited"))
	}))
	defer srv.Close()

	c := NewLLM(srv.URL+"/v1", "")
	_, err := c.Complete(context.Background(), "any", "hi")
	if err == nil || !strings.Contains(err.Error(), "HTTP 429") {
		t.Fatalf("expected HTTP 429 error, got %v", err)
	}
}

func TestLLM_StreamMidwayError(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		flusher, _ := w.(http.Flusher)
		fmt.Fprintf(w, "data: {\"choices\":[{\"delta\":{\"content\":\"start\"}}]}\n\n")
		flusher.Flush()
		fmt.Fprintf(w, "data: not-valid-json\n\n")
		flusher.Flush()
	}))
	defer srv.Close()

	c := NewLLM(srv.URL+"/v1", "")
	_, err := c.Complete(context.Background(), "any", "hi")
	if err == nil {
		t.Fatal("expected parse error mid-stream, got nil")
	}
	if !strings.Contains(err.Error(), "parse delta") {
		t.Fatalf("error doesn't mention parse: %v", err)
	}
}

func TestLLM_StreamEndedWithoutDone(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		flusher, _ := w.(http.Flusher)
		fmt.Fprintf(w, "data: {\"choices\":[{\"delta\":{\"content\":\"hi\"}}]}\n\n")
		flusher.Flush()
	}))
	defer srv.Close()

	c := NewLLM(srv.URL+"/v1", "")
	_, err := c.Complete(context.Background(), "any", "x")
	if err == nil {
		t.Fatal("expected error, got nil")
	}
	if !strings.Contains(err.Error(), "without [DONE]") && !strings.Contains(err.Error(), "stream") {
		t.Fatalf("error doesn't mention abrupt end: %v", err)
	}
}

func TestLLM_EmptyStream(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		flusher, _ := w.(http.Flusher)
		fmt.Fprintf(w, "data: [DONE]\n\n")
		flusher.Flush()
	}))
	defer srv.Close()

	c := NewLLM(srv.URL+"/v1", "")
	_, err := c.Complete(context.Background(), "any", "x")
	if err == nil {
		t.Fatal("expected empty-stream error, got nil")
	}
	if !strings.Contains(err.Error(), "empty stream") {
		t.Fatalf("error doesn't mention empty stream: %v", err)
	}
}

func TestLLM_UpstreamTimeout(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		time.Sleep(5 * time.Second)
	}))
	defer srv.Close()

	c := NewLLM(srv.URL+"/v1", "")
	ctx, cancel := context.WithTimeout(context.Background(), 100*time.Millisecond)
	defer cancel()
	_, err := c.Complete(ctx, "any", "hello")
	if err == nil {
		t.Fatal("expected timeout error, got nil")
	}
	if ctx.Err() == nil {
		t.Fatalf("ctx not cancelled: %v", err)
	}
}
