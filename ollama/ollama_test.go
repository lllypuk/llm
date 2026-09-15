package ollama_test

import (
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/lllypuk/llm"
	"github.com/lllypuk/llm/ollama"
)

func server(t *testing.T, h http.HandlerFunc) *ollama.Provider {
	t.Helper()

	srv := httptest.NewServer(h)
	t.Cleanup(srv.Close)

	return ollama.New(srv.URL + "/")
}

func reply(w http.ResponseWriter, body map[string]any) {
	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(body)
}

// TestCompleteReadsEnvelope — текст, модель, причина, токены и серверная длительность из ответа.
func TestCompleteReadsEnvelope(t *testing.T) {
	t.Parallel()

	p := server(t, func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/api/chat" || r.Method != http.MethodPost {
			t.Errorf("%s %s", r.Method, r.URL.Path)
		}

		reply(w, map[string]any{
			"model": "m:latest", "message": map[string]any{"role": "assistant", "content": `{"a":1}`},
			"done_reason": "stop", "total_duration": int64(1_500_000_000), "prompt_eval_count": 12, "eval_count": 7,
		})
	})

	res, err := p.Complete(context.Background(), llm.Request{Model: "m"})
	if err != nil {
		t.Fatal(err)
	}

	want := llm.Result{
		Model: "m:latest", Text: `{"a":1}`, FinishReason: "stop",
		Usage: llm.Usage{InputTokens: 12, OutputTokens: 7, Known: true}, ServerLatency: 1500 * time.Millisecond,
	}
	if res != want {
		t.Errorf("результат %+v, ожидался %+v", res, want)
	}
}

// TestCompleteEncodesRequest — роли, кадры base64 при данных, format по режиму, think выключен,
// температура в options.
func TestCompleteEncodesRequest(t *testing.T) {
	t.Parallel()

	var got map[string]any

	p := server(t, func(w http.ResponseWriter, r *http.Request) {
		_ = json.NewDecoder(r.Body).Decode(&got)
		reply(w, map[string]any{"message": map[string]any{"content": "ok"}})
	})

	req := llm.Request{
		Model: "m",
		Messages: []llm.Message{
			{Role: llm.RoleSystem, Text: "rules"},
			{Role: llm.RoleUser, Text: "data", Images: []llm.Image{{MIME: "image/png", Data: []byte{1, 2, 3}}}},
		},
		Output:      llm.Output{Mode: llm.ModeSchema, Schema: json.RawMessage(`{"type":"object"}`)},
		Temperature: llm.Ptr(0.1),
	}

	if _, err := p.Complete(context.Background(), req); err != nil {
		t.Fatal(err)
	}

	msgs, _ := got["messages"].([]any)
	if len(msgs) != 2 {
		t.Fatalf("сообщений %d", len(msgs))
	}

	sys, _ := msgs[0].(map[string]any)
	user, _ := msgs[1].(map[string]any)

	if sys["role"] != "system" || sys["images"] != nil {
		t.Errorf("инструкция %v", sys)
	}

	if imgs, _ := user["images"].([]any); user["role"] != "user" || len(imgs) != 1 || imgs[0] != "AQID" {
		t.Errorf("данные %v", user)
	}

	if got["think"] != false || got["stream"] != false {
		t.Errorf("think %v, stream %v", got["think"], got["stream"])
	}

	if format, _ := got["format"].(map[string]any); format["type"] != "object" {
		t.Errorf("format %v", got["format"])
	}

	if opts, _ := got["options"].(map[string]any); opts["temperature"] != 0.1 {
		t.Errorf("options %v", got["options"])
	}
}

// TestCompleteFormatByMode — json подсказкой строкой, text без format вовсе.
func TestCompleteFormatByMode(t *testing.T) {
	t.Parallel()

	for _, tc := range []struct {
		mode llm.Mode
		want any
	}{{llm.ModeJSON, "json"}, {llm.ModeText, nil}} {
		var got map[string]any

		p := server(t, func(w http.ResponseWriter, r *http.Request) {
			_ = json.NewDecoder(r.Body).Decode(&got)
			reply(w, map[string]any{"message": map[string]any{"content": "ok"}})
		})

		req := llm.Request{Model: "m", Output: llm.Output{Mode: tc.mode}}
		if _, err := p.Complete(context.Background(), req); err != nil {
			t.Fatal(err)
		}

		if got["format"] != tc.want {
			t.Errorf("%s: format %v, ожидалось %v", tc.mode, got["format"], tc.want)
		}
	}
}

// TestCompleteRejectsRouteFailures — отказ под кодом 200, пустое содержимое и битый конверт —
// ResponseError, а не ответ модели.
func TestCompleteRejectsRouteFailures(t *testing.T) {
	t.Parallel()

	bodies := map[string]string{
		"error":  `{"error":"model not loaded"}`,
		"empty":  `{"message":{"content":"   "}}`,
		"broken": `{"message":`,
		"huge":   `{"message":{"content":"` + strings.Repeat("x", 600<<10) + `"}}`,
	}

	for name, body := range bodies {
		p := server(t, func(w http.ResponseWriter, _ *http.Request) {
			w.Header().Set("Content-Type", "application/json")
			_, _ = w.Write([]byte(body))
		})

		_, err := p.Complete(context.Background(), llm.Request{Model: "m"})

		var response *llm.ResponseError
		if !errors.As(err, &response) {
			t.Errorf("%s: ожидался ResponseError, получено %v", name, err)
		}
	}
}

// TestCompleteReportsStatus — не-200 едет StatusError с текстом конверта и Retry-After.
func TestCompleteReportsStatus(t *testing.T) {
	t.Parallel()

	p := server(t, func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Retry-After", "3")
		w.WriteHeader(http.StatusTooManyRequests)
		_, _ = w.Write([]byte(`{"error":"quota"}`))
	})

	_, err := p.Complete(context.Background(), llm.Request{Model: "m"})

	var status *llm.StatusError
	if !errors.As(err, &status) || status.Status != 429 || status.Message != "quota" ||
		status.RetryAfter != 3*time.Second {
		t.Errorf("ошибка %v", err)
	}
}

// TestClientOverOllama — связка: клиент повторяет 503 демона и отдаёт ответ.
func TestClientOverOllama(t *testing.T) {
	t.Parallel()

	calls := 0

	p := server(t, func(w http.ResponseWriter, _ *http.Request) {
		calls++
		if calls == 1 {
			w.WriteHeader(http.StatusServiceUnavailable)

			return
		}

		reply(w, map[string]any{"message": map[string]any{"content": "ok"}})
	})

	c := llm.New(p, time.Second)
	c.Pause = time.Millisecond

	res, err := c.Chat(context.Background(), llm.Request{Model: "m"})
	if err != nil || res.Attempts != 2 || res.Model != "m" {
		t.Errorf("результат %+v, %v", res, err)
	}
}
