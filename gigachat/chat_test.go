package gigachat_test

import (
	"context"
	"errors"
	"io"
	"net/http"
	"reflect"
	"strconv"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/lllypuk/llm"
)

// chatServer — `/chat/completions` по сценарию поверх `/files`: n-й запрос генерации получает reply(n).
type chatServer struct {
	files *files
	reply func(n int) (status int, body string)

	mu     sync.Mutex
	bodies []string
	tokens []string
}

func (s *chatServer) serve(t *testing.T) func(w http.ResponseWriter, r *http.Request) {
	t.Helper()

	serveFiles := s.files.serve(t)

	return func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/api/chat/completions" {
			serveFiles(w, r)

			return
		}

		raw, _ := io.ReadAll(r.Body)

		s.mu.Lock()
		s.bodies = append(s.bodies, string(raw))
		s.tokens = append(s.tokens, bearer(r))
		n := len(s.bodies)
		s.mu.Unlock()

		status, body := s.reply(n)
		w.Header().Set("X-Request-Id", "req-"+strconv.Itoa(n))
		w.WriteHeader(status)
		_, _ = io.WriteString(w, body)
	}
}

func (s *chatServer) requests() []string {
	s.mu.Lock()
	defer s.mu.Unlock()

	return append([]string(nil), s.bodies...)
}

func always(status int, body string) func(int) (int, string) {
	return func(int) (int, string) { return status, body }
}

func chatClient(t *testing.T, s *chatServer) (*llm.Client, *backend) {
	t.Helper()

	if s.files == nil {
		s.files = &files{}
	}

	b := &backend{api: s.serve(t)}
	c := llm.New(provider(t, b), 5*time.Second)
	c.Pause = time.Millisecond

	return c, b
}

const okReply = `{"choices":[{"message":{"role":"assistant","content":"{\"brand\":\"Bosch\"}"},"index":0,` +
	`"finish_reason":"stop"}],"created":1,"model":"GigaChat-2-Max:2.0.28.2","object":"chat.completion",` +
	`"usage":{"prompt_tokens":120,"completion_tokens":9,"total_tokens":150,"precached_prompt_tokens":21}}`

// TestChatExactRequestAndResult — тело запроса байт в байт: attachments по сообщениям, схема со strict,
// температура и предел ответа; модель, request id, finish и расход — из ответа, кеш отдельной частью.
func TestChatExactRequestAndResult(t *testing.T) {
	t.Parallel()

	s := &chatServer{reply: always(http.StatusOK, okReply)}
	c, _ := chatClient(t, s)

	res, err := c.Chat(context.Background(), llm.Request{
		CallID: "call-1",
		Task:   "nameplate",
		Model:  "GigaChat-2-Max",
		Messages: []llm.Message{
			{Role: llm.RoleSystem, Text: "правила"},
			{Role: llm.RoleUser, Text: "шильдик", Images: []llm.Image{jpeg("a")}},
			{Role: llm.RoleUser, Text: "ещё", Images: []llm.Image{jpeg("b")}},
		},
		Output:  llm.Output{Mode: llm.ModeSchema, Schema: []byte(`{"type":"object"}`), Strict: true},
		Options: llm.Options{Temperature: llm.Ptr(0.1), MaxOutputTokens: 512},
	})
	if err != nil {
		t.Fatal(err)
	}

	want := `{"model":"GigaChat-2-Max","messages":[{"role":"system","content":"правила"},` +
		`{"role":"user","content":"шильдик","attachments":["f1"]},{"role":"user","content":"ещё","attachments":["f2"]}],` +
		`"stream":false,"temperature":0.1,"max_tokens":512,` +
		`"response_format":{"type":"json_schema","schema":{"type":"object"},"strict":true}}`
	if got := s.requests(); len(got) != 1 || got[0] != want {
		t.Errorf("запросы %q", got)
	}

	if s.tokens[0] != "token-1" {
		t.Errorf("токен %q", s.tokens[0])
	}

	usage := llm.Usage{
		Raw: map[string]int{
			"prompt_tokens":           120,
			"completion_tokens":       9,
			"precached_prompt_tokens": 21,
			"total_tokens":            150,
		},
		BillableInput: 120,
		CachedInput:   21,
		Output:        9,
		Known:         true,
	}

	if res.Text != `{"brand":"Bosch"}` || res.Model != "GigaChat-2-Max:2.0.28.2" || res.RequestID != "req-1" ||
		res.Finish != (llm.Finish{Raw: "stop", Kind: llm.FinishStop}) || !reflect.DeepEqual(res.Usage, usage) {
		t.Errorf("результат %+v", res)
	}

	if res.Usage.InputTokens() != 141 || len(res.Attempts) != 1 || res.Attempts[0].RequestID != "req-1" {
		t.Errorf("вход %d, попытки %+v", res.Usage.InputTokens(), res.Attempts)
	}

	if _, deleted := s.files.state(); !reflect.DeepEqual(deleted, []string{"f1", "f2"}) {
		t.Errorf("удалены %v", deleted)
	}
}

// TestChatTextWithoutOptions — текстовый запрос без кадров и опций: ни attachments, ни response_format.
func TestChatTextWithoutOptions(t *testing.T) {
	t.Parallel()

	s := &chatServer{reply: always(http.StatusOK, okReply)}
	c, _ := chatClient(t, s)

	_, err := c.Chat(context.Background(), llm.Request{
		Model:    "GigaChat-2",
		Messages: []llm.Message{{Role: llm.RoleUser, Text: "вопрос"}},
		Output:   llm.Output{Mode: llm.ModeText},
	})
	if err != nil {
		t.Fatal(err)
	}

	want := `{"model":"GigaChat-2","messages":[{"role":"user","content":"вопрос"}],"stream":false}`
	if got := s.requests(); len(got) != 1 || got[0] != want {
		t.Errorf("запросы %q", got)
	}
}

// TestChatProfileRejects — ModeJSON без схемы, кадры у Lite и неизвестная модель отбиваются до сети,
// и клиентом, и самим плечом.
func TestChatProfileRejects(t *testing.T) {
	t.Parallel()

	text := []llm.Message{{Role: llm.RoleUser, Text: "вопрос"}}
	image := []llm.Message{{Role: llm.RoleUser, Text: "кадр", Images: []llm.Image{jpeg("a")}}}

	for _, tc := range []struct {
		name string
		req  llm.Request
	}{
		{name: "json без схемы", req: llm.Request{Model: "GigaChat-2-Max", Messages: text, Output: llm.Output{Mode: llm.ModeJSON}}},
		{name: "кадр у lite", req: llm.Request{Model: "GigaChat-2", Messages: image}},
		{name: "неизвестная модель со схемой", req: llm.Request{
			Model: "GigaChat-3", Messages: text, Output: llm.Output{Mode: llm.ModeSchema, Schema: []byte(`{}`)},
		}},
		{name: "рассуждения", req: llm.Request{
			Model: "GigaChat-2-Pro", Messages: text, Options: llm.Options{Reasoning: llm.EffortHigh},
		}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()

			s := &chatServer{reply: always(http.StatusOK, okReply)}
			c, b := chatClient(t, s)

			_, err := c.Chat(context.Background(), tc.req)

			var call *llm.CallError
			if !errors.As(err, &call) || call.Class != llm.RetryNever || call.Report.Outcome != llm.OutcomeBadRequest {
				t.Errorf("клиент: %v", err)
			}

			var request *llm.RequestError
			if _, err = c.Provider.Complete(context.Background(), tc.req); !errors.As(err, &request) {
				t.Errorf("плечо: %v", err)
			}

			if uploads, _ := s.files.state(); len(s.requests()) != 0 || uploads != 0 || b.oauthCalls.Load() != 0 {
				t.Errorf("до сети дошло: генераций %d, загрузок %d, OAuth %d",
					len(s.requests()), uploads, b.oauthCalls.Load())
			}
		})
	}
}

// TestChatTerminalFinishNotRetried — фильтр и предел длины оплачены: одна генерация на вызов, расход,
// модель и request id в отчёте, файлы убраны; пустое содержимое при length — тот же исход.
func TestChatTerminalFinishNotRetried(t *testing.T) {
	t.Parallel()

	for _, tc := range []struct {
		name    string
		reply   string
		sentry  error
		outcome string
	}{
		{
			name: "blacklist",
			reply: `{"choices":[{"message":{"content":"Не люблю менять тему разговора"},"finish_reason":"blacklist"}],` +
				`"model":"GigaChat-2-Max:2.0","usage":{"prompt_tokens":10,"completion_tokens":7,"precached_prompt_tokens":0}}`,
			sentry:  llm.ErrFiltered,
			outcome: llm.OutcomeFiltered,
		},
		{
			name: "length",
			reply: `{"choices":[{"message":{"content":"{\"brand\":"},"finish_reason":"length"}],` +
				`"model":"GigaChat-2-Max:2.0","usage":{"prompt_tokens":10,"completion_tokens":7,"precached_prompt_tokens":0}}`,
			sentry:  llm.ErrTruncated,
			outcome: llm.OutcomeTruncated,
		},
		{
			name: "length без содержимого",
			reply: `{"choices":[{"message":{"content":""},"finish_reason":"length"}],` +
				`"model":"GigaChat-2-Max:2.0","usage":{"prompt_tokens":10,"completion_tokens":7,"precached_prompt_tokens":0}}`,
			sentry:  llm.ErrTruncated,
			outcome: llm.OutcomeTruncated,
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()

			s := &chatServer{reply: always(http.StatusOK, tc.reply)}
			c, _ := chatClient(t, s)

			_, err := c.Chat(context.Background(), llm.Request{
				Model:    "GigaChat-2-Max",
				Messages: []llm.Message{{Role: llm.RoleUser, Text: "кадр", Images: []llm.Image{jpeg("a")}}},
			})

			var call *llm.CallError
			if !errors.As(err, &call) || !errors.Is(err, tc.sentry) || call.Class != llm.RetryNever {
				t.Fatalf("отказ %v", err)
			}

			spent := llm.Usage{BillableInput: 10, Output: 7, Known: true}
			if len(s.requests()) != 1 || len(call.Attempts) != 1 || call.Report.Outcome != tc.outcome {
				t.Errorf("генераций %d, отчёт %+v", len(s.requests()), call.Report)
			}

			a := call.Attempts[0]
			if a.Model != "GigaChat-2-Max:2.0" || a.RequestID != "req-1" || a.Outcome != tc.outcome ||
				!sameParts(a.Usage, spent) || !sameParts(call.Report.Usage, spent) {
				t.Errorf("попытка %+v, расход вызова %+v", a, call.Report.Usage)
			}

			if uploads, deleted := s.files.state(); uploads != 1 || !reflect.DeepEqual(deleted, []string{"f1"}) {
				t.Errorf("загрузок %d, удалены %v", uploads, deleted)
			}
		})
	}
}

// TestChatFailuresCountInference — повторяемые отказы: каждая попытка — одна генерация со своей загрузкой
// и уборкой; расход негодных ответов и request id отказов по статусу остаются в отчётах.
func TestChatFailuresCountInference(t *testing.T) {
	t.Parallel()

	empty := `{"choices":[{"message":{"content":" "},"finish_reason":"stop"}],"model":"GigaChat-2-Max:2.0",` +
		`"usage":{"prompt_tokens":10,"completion_tokens":1,"precached_prompt_tokens":4}}`

	for _, tc := range []struct {
		name        string
		reply       func(int) (int, string)
		inference   int
		ok          bool
		spent       llm.Usage
		failRequest string
	}{
		{
			name:      "пустое содержимое трижды",
			reply:     always(http.StatusOK, empty),
			inference: 3,
			spent:     llm.Usage{BillableInput: 30, CachedInput: 12, Output: 3, Known: true},
		},
		{
			name: "500, затем ответ",
			reply: func(n int) (int, string) {
				if n == 1 {
					return http.StatusInternalServerError, `{"status":500,"message":"Internal Server Error"}`
				}

				return http.StatusOK, okReply
			},
			inference:   2,
			ok:          true,
			failRequest: "req-1",
		},
		{
			name:      "finish_reason error",
			reply:     always(http.StatusOK, `{"choices":[{"message":{"content":"x"},"finish_reason":"error"}]}`),
			inference: 3,
		},
		{
			name:      "без choices",
			reply:     always(http.StatusOK, `{"model":"GigaChat-2-Max:2.0","choices":[]}`),
			inference: 3,
		},
		{
			name:      "400 не повторяется",
			reply:     always(http.StatusBadRequest, `{"status":400,"message":"Invalid params: schema"}`),
			inference: 1,
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()

			s := &chatServer{reply: tc.reply}
			c, _ := chatClient(t, s)

			res, err := c.Chat(context.Background(), llm.Request{
				Model:    "GigaChat-2-Pro",
				Messages: []llm.Message{{Role: llm.RoleUser, Text: "кадр", Images: []llm.Image{jpeg("a")}}},
			})

			attempts, report := res.Attempts, res.Report

			var call *llm.CallError

			switch {
			case tc.ok && err != nil, !tc.ok && !errors.As(err, &call):
				t.Fatalf("отказ %v", err)
			case !tc.ok:
				attempts, report = call.Attempts, call.Report
			}

			uploads, deleted := s.files.state()
			if len(s.requests()) != tc.inference || len(attempts) != tc.inference || uploads != tc.inference ||
				len(deleted) != tc.inference {
				t.Errorf("генераций %d, попыток %d, загрузок %d, удалено %d",
					len(s.requests()), len(attempts), uploads, len(deleted))
			}

			if tc.spent.Known && !sameParts(report.Usage, tc.spent) {
				t.Errorf("расход вызова %+v", report.Usage)
			}

			if tc.failRequest != "" && attempts[0].RequestID != tc.failRequest {
				t.Errorf("попытка %+v", attempts[0])
			}
		})
	}
}

// TestChatUsagePartial — не названный кеш или отсутствующий usage делают расход неполным, названное остаётся.
func TestChatUsagePartial(t *testing.T) {
	t.Parallel()

	for _, tc := range []struct {
		name  string
		usage string
		want  llm.Usage
	}{
		{
			name:  "без кеша",
			usage: `,"usage":{"prompt_tokens":5,"completion_tokens":2}`,
			want:  llm.Usage{Raw: map[string]int{"prompt_tokens": 5, "completion_tokens": 2}, BillableInput: 5, Output: 2},
		},
		{name: "без usage"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()

			reply := `{"choices":[{"message":{"content":"ok"},"finish_reason":"stop"}]` + tc.usage + `}`
			s := &chatServer{reply: always(http.StatusOK, reply)}
			c, _ := chatClient(t, s)

			res, err := c.Chat(context.Background(), llm.Request{
				Model: "GigaChat-2", Messages: []llm.Message{{Role: llm.RoleUser, Text: "вопрос"}},
			})
			if err != nil {
				t.Fatal(err)
			}

			if !reflect.DeepEqual(res.Usage, tc.want) || res.Report.Usage.Known {
				t.Errorf("расход %+v, вызова %+v", res.Usage, res.Report.Usage)
			}
		})
	}
}

// TestChat401Refreshes — 401 генерации гасит токен: одна повторная генерация с новым, в том же вызове.
func TestChat401Refreshes(t *testing.T) {
	t.Parallel()

	s := &chatServer{reply: func(n int) (int, string) {
		if n == 1 {
			return http.StatusUnauthorized, `{"status":401,"message":"Unauthorized"}`
		}

		return http.StatusOK, okReply
	}}
	c, b := chatClient(t, s)

	res, err := c.Chat(context.Background(), llm.Request{
		Model: "GigaChat-2", Messages: []llm.Message{{Role: llm.RoleUser, Text: "вопрос"}},
	})
	if err != nil {
		t.Fatal(err)
	}

	if len(res.Attempts) != 1 || b.oauthCalls.Load() != 2 || strings.Join(s.tokens, ",") != "token-1,token-2" {
		t.Errorf("попыток %d, OAuth %d, токены %v", len(res.Attempts), b.oauthCalls.Load(), s.tokens)
	}
}

// sameParts — части расхода без Raw: сумма попыток Raw складывает, а сверяются нормализованные.
func sameParts(got, want llm.Usage) bool {
	got.Raw, want.Raw = nil, nil

	return reflect.DeepEqual(got, want)
}
