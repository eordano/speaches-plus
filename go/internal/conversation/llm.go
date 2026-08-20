package conversation

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
)

type LLM struct {
	baseURL string
	apiKey  string
	http    *http.Client
}

func NewLLM(baseURL, apiKey string) *LLM {
	return &LLM{
		baseURL: strings.TrimRight(baseURL, "/"),
		apiKey:  apiKey,
		http:    &http.Client{},
	}
}

func (l *LLM) Configured() bool { return l.baseURL != "" }

type Delta struct {
	Content string
	Err     error
	Done    bool
}

func (l *LLM) Stream(ctx context.Context, model, userText string) (<-chan Delta, error) {
	return l.StreamWithInstructions(ctx, model, "", userText)
}

func (l *LLM) StreamWithInstructions(ctx context.Context, model, instructions, userText string) (<-chan Delta, error) {
	if !l.Configured() {
		return nil, fmt.Errorf("llm: CHAT_COMPLETION_BASE_URL not set")
	}
	messages := []map[string]string{}
	if instructions != "" {
		messages = append(messages, map[string]string{"role": "system", "content": instructions})
	}
	messages = append(messages, map[string]string{"role": "user", "content": userText})
	body := map[string]any{
		"model":    model,
		"messages": messages,
		"stream":   true,
	}
	raw, err := json.Marshal(body)
	if err != nil {
		return nil, err
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodPost,
		l.baseURL+"/chat/completions", bytes.NewReader(raw))
	if err != nil {
		return nil, err
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Accept", "text/event-stream")
	if l.apiKey != "" {
		req.Header.Set("Authorization", "Bearer "+l.apiKey)
	}

	resp, err := l.http.Do(req)
	if err != nil {
		return nil, err
	}
	if resp.StatusCode >= 400 {
		body, _ := io.ReadAll(resp.Body)
		resp.Body.Close()
		return nil, fmt.Errorf("llm: HTTP %d: %s", resp.StatusCode, body)
	}

	out := make(chan Delta, 16)
	go func() {
		defer resp.Body.Close()
		defer close(out)

		scanner := bufio.NewScanner(resp.Body)
		scanner.Buffer(make([]byte, 0, 16*1024), 1<<20)
		for scanner.Scan() {
			line := strings.TrimSpace(scanner.Text())
			if line == "" || !strings.HasPrefix(line, "data:") {
				continue
			}
			payload := strings.TrimSpace(line[len("data:"):])
			if payload == "[DONE]" {
				out <- Delta{Done: true}
				return
			}
			var chunk struct {
				Choices []struct {
					Delta struct {
						Content string `json:"content"`
					} `json:"delta"`
					FinishReason *string `json:"finish_reason"`
				} `json:"choices"`
			}
			if err := json.Unmarshal([]byte(payload), &chunk); err != nil {
				out <- Delta{Err: fmt.Errorf("llm: parse delta: %w (body=%s)", err, payload)}
				return
			}
			if len(chunk.Choices) == 0 {
				continue
			}
			if c := chunk.Choices[0].Delta.Content; c != "" {
				select {
				case out <- Delta{Content: c}:
				case <-ctx.Done():
					out <- Delta{Err: ctx.Err()}
					return
				}
			}
			if chunk.Choices[0].FinishReason != nil {
				out <- Delta{Done: true}
				return
			}
		}
		if err := scanner.Err(); err != nil {
			out <- Delta{Err: fmt.Errorf("llm: stream read: %w", err)}
			return
		}
		out <- Delta{Err: fmt.Errorf("llm: stream ended without [DONE] or finish_reason")}
	}()
	return out, nil
}

func (l *LLM) Complete(ctx context.Context, model, userText string) (string, error) {
	deltas, err := l.Stream(ctx, model, userText)
	if err != nil {
		return "", err
	}
	var sb strings.Builder
	gotAny := false
	for d := range deltas {
		if d.Err != nil {
			return "", d.Err
		}
		if d.Done {
			if !gotAny {
				return "", fmt.Errorf("llm: empty stream (no deltas)")
			}
			return sb.String(), nil
		}
		if d.Content != "" {
			sb.WriteString(d.Content)
			gotAny = true
		}
	}
	if !gotAny {
		return "", fmt.Errorf("llm: empty choices")
	}
	return sb.String(), nil
}
