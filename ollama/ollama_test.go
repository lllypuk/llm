package ollama_test

import (
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"reflect"
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
			"model":             "m:latest",
			"message":           map[string]any{"role": "assistant", "content": `{"a":1}`},
			"done":              true,
			"done_reason":       "stop",
			"total_duration":    int64(1_500_000_000),
			"prompt_eval_count": 12,
			"eval_count":        7,
		})
	})

	res, err := p.Complete(context.Background(), llm.Request{Model: "m"})
	if err != nil {
		t.Fatal(err)
	}

	usage := llm.Usage{
		Raw:           map[string]int{"prompt_eval_count": 12, "eval_count": 7},
		BillableInput: 12,
		Output:        7,
		Known:         true,
	}
	stop := llm.Finish{Raw: "stop", Kind: llm.FinishStop}
	if res.Model != "m:latest" || res.Text != `{"a":1}` || res.Finish != stop || res.Cleanup != nil ||
		!reflect.DeepEqual(res.Usage, usage) || res.ServerLatency != 1500*time.Millisecond {
		t.Errorf("результат %+v", res)
	}
}

// TestCompleteEncodesRequest — роли, кадры base64 при данных, format по режиму, think выключен,
// температура в options.
func TestCompleteEncodesRequest(t *testing.T) {
	t.Parallel()

	var got map[string]any

	p := server(t, func(w http.ResponseWriter, r *http.Request) {
		_ = json.NewDecoder(r.Body).Decode(&got)
		reply(w, map[string]any{"done": true, "message": map[string]any{"content": "ok"}})
	})

	req := llm.Request{
		Model: "m",
		Messages: []llm.Message{
			{Role: llm.RoleSystem, Text: "rules"},
			{Role: llm.RoleUser, Text: "data", Images: []llm.Image{{MIME: "image/png", Data: []byte{1, 2, 3}}}},
		},
		Output:  llm.Output{Mode: llm.ModeSchema, Schema: json.RawMessage(`{"type":"object"}`)},
		Options: llm.Options{Temperature: llm.Ptr(0.1), MaxOutputTokens: 256},
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

	if opts, _ := got["options"].(map[string]any); opts["temperature"] != 0.1 || opts["num_predict"] != 256.0 {
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
			reply(w, map[string]any{"done": true, "message": map[string]any{"content": "ok"}})
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
		"error":   `{"error":"model not loaded"}`,
		"empty":   `{"done":true,"message":{"content":"   "}}`,
		"broken":  `{"message":`,
		"huge":    `{"done":true,"message":{"content":"` + strings.Repeat("x", 600<<10) + `"}}`,
		"notdone": `{"done":false,"message":{"content":"part"}}`,
		"bracket": `{"done":true,"message":{"content":"a"}}]`,
		"brace":   `{"done":true,"message":{"content":"a"}}}`,
		"stream":  `{"done":true,"message":{"content":"a"}}` + "\n" + `{"done":true,"message":{"content":"b"}}`,
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

// TestCompleteAllowsTrailingWhitespace — перевод строки после конверта — не хвост.
func TestCompleteAllowsTrailingWhitespace(t *testing.T) {
	t.Parallel()

	p := server(t, func(w http.ResponseWriter, _ *http.Request) {
		_, _ = w.Write([]byte(`{"done":true,"message":{"content":"ok"}}` + "\n\n"))
	})

	if _, err := p.Complete(context.Background(), llm.Request{Model: "m"}); err != nil {
		t.Error(err)
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

		reply(w, map[string]any{"done": true, "message": map[string]any{"content": "ok"}})
	})

	c := llm.New(p, time.Second)
	c.Pause = time.Millisecond

	res, err := c.Chat(context.Background(), llm.Request{Model: "m"})
	if err != nil || res.Report.Attempts != 2 || res.Model != "m" {
		t.Errorf("результат %+v, %v", res, err)
	}
}

// TestCompleteUsageKnownOnlyWithCounters — без счётчиков расход неизвестен, с ними — известен
// и при негодном содержимом уезжает в ResponseError.
func TestCompleteUsageKnownOnlyWithCounters(t *testing.T) {
	t.Parallel()

	p := server(t, func(w http.ResponseWriter, _ *http.Request) {
		reply(w, map[string]any{"done": true, "message": map[string]any{"content": "ok"}})
	})

	res, err := p.Complete(context.Background(), llm.Request{Model: "m"})
	if err != nil || res.Usage.Known {
		t.Errorf("без счётчиков: %+v, %v", res.Usage, err)
	}

	p = server(t, func(w http.ResponseWriter, _ *http.Request) {
		reply(w, map[string]any{
			"done": true, "model": "m:latest", "total_duration": int64(2_000_000_000),
			"message": map[string]any{"content": ""}, "prompt_eval_count": 12, "eval_count": 7,
		})
	})

	_, err = p.Complete(context.Background(), llm.Request{Model: "m"})

	var response *llm.ResponseError
	if !errors.As(err, &response) || response.Usage.BillableInput != 12 || !response.Usage.Known {
		t.Errorf("расход при пустом содержимом потерян: %v", err)
	}

	if response.Model != "m:latest" || response.ServerLatency != 2*time.Second {
		t.Errorf("метаданные конверта потеряны: %+v", response)
	}

	p = server(t, func(w http.ResponseWriter, _ *http.Request) {
		reply(w, map[string]any{
			"done": true, "message": map[string]any{"content": "ok"}, "prompt_eval_count": -1, "eval_count": 7,
		})
	})

	res, err = p.Complete(context.Background(), llm.Request{Model: "m"})
	if err != nil || res.Usage.Known || res.Usage.BillableInput != 0 || res.Usage.Output != 7 ||
		res.Usage.Raw["prompt_eval_count"] != -1 {
		t.Errorf("отрицательный счётчик: %+v, %v", res.Usage, err)
	}
}

// TestCompleteKeepsDeadlineWhileReadingBody — просрочка при чтении тела остаётся просрочкой.
func TestCompleteKeepsDeadlineWhileReadingBody(t *testing.T) {
	t.Parallel()

	p := server(t, func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		w.(http.Flusher).Flush()
		<-r.Context().Done()
	})

	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Millisecond)
	defer cancel()

	_, err := p.Complete(ctx, llm.Request{Model: "m"})
	if !errors.Is(err, context.DeadlineExceeded) {
		t.Errorf("причина потеряна: %v", err)
	}
}

// TestClientStopsOnLength — done_reason length оплачен: клиент не повторяет, расход в отчёте,
// даже когда обрезанный ответ пуст.
func TestClientStopsOnLength(t *testing.T) {
	t.Parallel()

	for _, content := range []string{`{"a":`, ""} {
		calls := 0

		p := server(t, func(w http.ResponseWriter, _ *http.Request) {
			calls++

			reply(w, map[string]any{
				"done": true, "done_reason": "length", "message": map[string]any{"content": content},
				"prompt_eval_count": 5, "eval_count": 64,
			})
		})

		c := llm.New(p, time.Second)
		c.Pause = time.Millisecond

		_, err := c.Chat(context.Background(), llm.Request{Model: "m", Output: llm.Output{Mode: llm.ModeJSON}})

		var call *llm.CallError
		if !errors.As(err, &call) || !errors.Is(err, llm.ErrTruncated) || calls != 1 ||
			call.Report.Outcome != llm.OutcomeTruncated || call.Report.Usage.Output != 64 {
			t.Errorf("%q: ошибка %v, запросов %d", content, err, calls)
		}
	}
}

// TestFinishNormalized — load и unload демона не причина конца генерации.
func TestFinishNormalized(t *testing.T) {
	t.Parallel()

	for raw, want := range map[string]llm.FinishKind{"stop": llm.FinishStop, "length": llm.FinishLength, "unload": ""} {
		p := server(t, func(w http.ResponseWriter, _ *http.Request) {
			reply(w, map[string]any{"done": true, "done_reason": raw, "message": map[string]any{"content": "ok"}})
		})

		res, err := p.Complete(context.Background(), llm.Request{Model: "m"})
		if err != nil || res.Finish != (llm.Finish{Raw: raw, Kind: want}) {
			t.Errorf("%s: %+v, %v", raw, res.Finish, err)
		}
	}
}

// TestCapabilities — профиль протокола: кадры, json и схема есть, рассуждений нет,
// и клиент отбивает их до демона.
func TestCapabilities(t *testing.T) {
	t.Parallel()

	calls := 0

	p := server(t, func(w http.ResponseWriter, _ *http.Request) {
		calls++

		reply(w, map[string]any{"done": true, "message": map[string]any{"content": "ok"}})
	})

	caps, known := p.Capabilities("any")
	if !known || !caps.Vision || !caps.Schema || caps.Reasoning {
		t.Errorf("профиль %+v, известен %v", caps, known)
	}

	req := llm.Request{Model: "m", Options: llm.Options{Reasoning: llm.EffortHigh}}
	if _, err := llm.New(p, time.Second).
		Chat(context.Background(), req); err == nil || llm.Recoverable(err) ||
		calls != 0 {
		t.Errorf("рассуждения ушли демону: %v, запросов %d", err, calls)
	}
}
