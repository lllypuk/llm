package yandex_test

import (
	"context"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"reflect"
	"strconv"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/lllypuk/llm"
	"github.com/lllypuk/llm/yandex"
)

// server — `/v1/chat/completions` по сценарию: n-й запрос получает reply(n).
type server struct {
	reply func(n int) (status int, body string)

	mu      sync.Mutex
	bodies  []string
	auth    []string
	project []string
}

func (s *server) start(t *testing.T) string {
	t.Helper()

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v1/chat/completions" {
			http.NotFound(w, r)

			return
		}

		raw, _ := io.ReadAll(r.Body)

		s.mu.Lock()
		s.bodies = append(s.bodies, string(raw))
		s.auth = append(s.auth, r.Header.Get("Authorization"))
		s.project = append(s.project, r.Header.Get("Openai-Project"))
		n := len(s.bodies)
		s.mu.Unlock()

		status, body := s.reply(n)
		w.Header().Set("X-Request-Id", "req-"+strconv.Itoa(n))
		w.WriteHeader(status)
		_, _ = io.WriteString(w, body)
	}))
	t.Cleanup(srv.Close)

	return srv.URL + "/v1/"
}

func (s *server) requests() []string {
	s.mu.Lock()
	defer s.mu.Unlock()

	return append([]string(nil), s.bodies...)
}

func always(status int, body string) func(int) (int, string) {
	return func(int) (int, string) { return status, body }
}

func client(t *testing.T, s *server, creds yandex.CredentialSource) *llm.Client {
	t.Helper()

	p, err := yandex.New(yandex.Config{Endpoint: s.start(t), Folder: "b1g", Credentials: creds})
	if err != nil {
		t.Fatal(err)
	}

	c := llm.New(p, 5*time.Second)
	c.Pause = time.Millisecond

	return c
}

func reply(finish, content string) string {
	return `{"id":"x","object":"chat.completion","model":"gpt://b1g/qwen3.6-35b-a3b/latest",` +
		`"choices":[{"index":0,"message":{"role":"assistant","content":` + strconv.Quote(content) + `},` +
		`"finish_reason":"` + finish + `"}],"usage":{"prompt_tokens":120,"completion_tokens":40,"total_tokens":160,` +
		`"prompt_tokens_details":{"cached_tokens":20},"completion_tokens_details":{"reasoning_tokens":30}}}`
}

// fullUsage — расход reply: кеш и рассуждения вычтены из общих, а не прибавлены к ним.
func fullUsage() llm.Usage {
	return llm.Usage{
		Raw: map[string]int{
			"prompt_tokens":                              120,
			"completion_tokens":                          40,
			"total_tokens":                               160,
			"prompt_tokens_details.cached_tokens":        20,
			"completion_tokens_details.reasoning_tokens": 30,
		},
		BillableInput: 100,
		CachedInput:   20,
		Reasoning:     30,
		Output:        10,
		Known:         true,
	}
}

// TestChatExactRequestAndResult — тело байт в байт: модель из каталога, кадры data-URI частями, схема с именем
// и strict; подпись ключом API и каталог заголовком; модель, request id, finish и расход — из ответа.
func TestChatExactRequestAndResult(t *testing.T) {
	t.Parallel()

	s := &server{reply: always(http.StatusOK, reply("stop", `{"brand":"Bosch"}`))}
	c := client(t, s, yandex.APIKey("key-1"))

	res, err := c.Chat(context.Background(), llm.Request{
		CallID: "call-1",
		Model:  "qwen3.6-35b-a3b/latest",
		Messages: []llm.Message{
			{Role: llm.RoleSystem, Text: "правила"},
			{Role: llm.RoleUser, Text: "шильдик", Images: []llm.Image{{MIME: "image/jpeg", Data: []byte("ab")}}},
			{Role: llm.RoleUser, Images: []llm.Image{{MIME: "image/png", Data: []byte("c")}}},
		},
		Output: llm.Output{
			Mode:   llm.ModeSchema,
			Schema: []byte(`{"type":"object"}`),
			Name:   "nameplate",
			Strict: true,
		},
		Options: llm.Options{Temperature: llm.Ptr(0.0), MaxOutputTokens: 512},
	})
	if err != nil {
		t.Fatal(err)
	}

	want := `{"model":"gpt://b1g/qwen3.6-35b-a3b/latest","messages":[{"role":"system","content":"правила"},` +
		`{"role":"user","content":[{"type":"text","text":"шильдик"},` +
		`{"type":"image_url","image_url":{"url":"data:image/jpeg;base64,YWI="}}]},` +
		`{"role":"user","content":[{"type":"image_url","image_url":{"url":"data:image/png;base64,Yw=="}}]}],` +
		`"stream":false,"temperature":0,"max_tokens":512,"response_format":{"type":"json_schema",` +
		`"json_schema":{"name":"nameplate","schema":{"type":"object"},"strict":true}}}`
	if got := s.requests(); len(got) != 1 || got[0] != want {
		t.Errorf("запросы %q", got)
	}

	if s.auth[0] != "Api-Key key-1" || s.project[0] != "b1g" {
		t.Errorf("подпись %q, каталог %q", s.auth[0], s.project[0])
	}

	if res.Text != `{"brand":"Bosch"}` || res.Model != "gpt://b1g/qwen3.6-35b-a3b/latest" || res.RequestID != "req-1" ||
		res.Finish != (llm.Finish{Raw: "stop", Kind: llm.FinishStop}) || !reflect.DeepEqual(res.Usage, fullUsage()) {
		t.Errorf("результат %+v", res)
	}

	if res.Usage.InputTokens() != 120 || res.Usage.OutputTokens() != 40 {
		t.Errorf("вход %d, выход %d", res.Usage.InputTokens(), res.Usage.OutputTokens())
	}
}

// TestIAMRefreshedBetweenAttempts — подпись спрашивается каждой попыткой: повтор после 5xx идёт с токеном,
// обновлённым снаружи; json_object и reasoning_effort у reasoning-модели.
func TestIAMRefreshedBetweenAttempts(t *testing.T) {
	t.Parallel()

	s := &server{reply: func(n int) (int, string) {
		if n == 1 {
			return http.StatusServiceUnavailable, `{"error":{"message":"перегружен"}}`
		}

		return http.StatusOK, reply("stop", `{}`)
	}}

	var issued atomic.Int32

	iam := yandex.IAMToken(func(context.Context) (string, error) {
		return "iam-" + strconv.Itoa(int(issued.Add(1))), nil
	})

	res, err := client(t, s, iam).Chat(context.Background(), llm.Request{
		Model:    "gpt-oss-120b",
		Messages: []llm.Message{{Role: llm.RoleUser, Text: "вопрос"}},
		Output:   llm.Output{Mode: llm.ModeJSON},
		Options:  llm.Options{Reasoning: llm.EffortLow},
	})
	if err != nil {
		t.Fatal(err)
	}

	want := `{"model":"gpt://b1g/gpt-oss-120b","messages":[{"role":"user","content":"вопрос"}],"stream":false,` +
		`"reasoning_effort":"low","response_format":{"type":"json_object"}}`
	if got := s.requests(); len(got) != 2 || got[1] != want {
		t.Errorf("запросы %q", got)
	}

	if !reflect.DeepEqual(s.auth, []string{"Bearer iam-1", "Bearer iam-2"}) || len(res.Attempts) != 2 {
		t.Errorf("подписи %q, попытки %d", s.auth, len(res.Attempts))
	}

	if res.Attempts[0].Outcome != llm.OutcomeHTTP5xx || res.Attempts[0].RequestID != "req-1" {
		t.Errorf("первая попытка %+v", res.Attempts[0])
	}
}

// TestTerminalFinishNotRetried — обрезанный и отфильтрованный ответ оплачен: один запрос, класс never,
// расход и finish в отчёте — и с содержимым, и без него.
func TestTerminalFinishNotRetried(t *testing.T) {
	t.Parallel()

	cases := map[string]struct {
		body   string
		sentry error
		kind   llm.FinishKind
	}{
		"length":               {reply("length", `{"brand":`), llm.ErrTruncated, llm.FinishLength},
		"content_filter":       {reply("content_filter", `отказ`), llm.ErrFiltered, llm.FinishContentFilter},
		"content_filter пусто": {reply("content_filter", ""), llm.ErrFiltered, llm.FinishContentFilter},
	}

	for name, tc := range cases {
		t.Run(name, func(t *testing.T) {
			t.Parallel()

			s := &server{reply: always(http.StatusOK, tc.body)}

			_, err := client(t, s, yandex.APIKey("k")).Chat(context.Background(), llm.Request{
				Model:    "yandexgpt/latest",
				Messages: []llm.Message{{Role: llm.RoleUser, Text: "x"}},
			})

			var call *llm.CallError
			if !errors.As(err, &call) || !errors.Is(err, tc.sentry) || call.Class != llm.RetryNever {
				t.Fatalf("отказ %v", err)
			}

			if n := len(s.requests()); n != 1 || len(call.Attempts) != 1 {
				t.Fatalf("запросов %d, попыток %d", n, len(call.Attempts))
			}

			a := call.Attempts[0]
			if a.Finish.Kind != tc.kind || !reflect.DeepEqual(a.Usage, fullUsage()) || a.RequestID != "req-1" ||
				!reflect.DeepEqual(call.Report.Usage, fullUsage()) {
				t.Errorf("попытка %+v, отчёт %+v", a, call.Report)
			}
		})
	}
}

// TestUsageOnBadResponse — пустой ответ без причины повторяется, и расход каждой оплаченной попытки в отчёте.
func TestUsageOnBadResponse(t *testing.T) {
	t.Parallel()

	s := &server{reply: always(http.StatusOK, reply("stop", " "))}
	c := client(t, s, yandex.APIKey("k"))
	c.Attempts = 2

	_, err := c.Chat(context.Background(), llm.Request{
		Model:    "yandexgpt-lite",
		Messages: []llm.Message{{Role: llm.RoleUser, Text: "x"}},
	})

	var call *llm.CallError
	if !errors.As(err, &call) || len(call.Attempts) != 2 || call.Attempts[1].Outcome != llm.OutcomeBadResponse {
		t.Fatalf("отказ %v", err)
	}

	if call.Report.Usage.BillableInput != 200 || call.Report.Usage.Reasoning != 60 || !call.Report.Usage.Known {
		t.Errorf("расход %+v", call.Report.Usage)
	}
}

// TestUsageParts — не названная деталь — ноль; часть больше общего и не названный общий делают расход неполным.
func TestUsageParts(t *testing.T) {
	t.Parallel()

	cases := map[string]struct {
		usage string
		want  llm.Usage
	}{
		"без деталей": {
			`{"prompt_tokens":7,"completion_tokens":3}`,
			llm.Usage{
				Raw:           map[string]int{"prompt_tokens": 7, "completion_tokens": 3},
				BillableInput: 7,
				Output:        3,
				Known:         true,
			},
		},
		"кеш больше входа": {
			`{"prompt_tokens":7,"completion_tokens":3,"prompt_tokens_details":{"cached_tokens":9}}`,
			llm.Usage{
				Raw: map[string]int{
					"prompt_tokens":                       7,
					"completion_tokens":                   3,
					"prompt_tokens_details.cached_tokens": 9,
				},
				CachedInput: 9,
				Output:      3,
			},
		},
		"без выхода": {
			`{"prompt_tokens":7}`,
			llm.Usage{Raw: map[string]int{"prompt_tokens": 7}, BillableInput: 7},
		},
	}

	for name, tc := range cases {
		t.Run(name, func(t *testing.T) {
			t.Parallel()

			body := `{"model":"m","choices":[{"message":{"content":"ok"},"finish_reason":"stop"}],"usage":` +
				tc.usage + `}`
			s := &server{reply: always(http.StatusOK, body)}

			res, err := client(t, s, yandex.APIKey("k")).Chat(context.Background(), llm.Request{
				Model:    "yandexgpt",
				Messages: []llm.Message{{Role: llm.RoleUser, Text: "x"}},
			})
			if err != nil {
				t.Fatal(err)
			}

			if !reflect.DeepEqual(res.Usage, tc.want) {
				t.Errorf("расход %+v", res.Usage)
			}
		})
	}
}

// TestRejectedBeforeNetwork — неподтверждённое профилем, схема без имени, адрес модели с каталогом,
// не-изображение и отказ подписи до сервера не доходят.
func TestRejectedBeforeNetwork(t *testing.T) {
	t.Parallel()

	text := []llm.Message{{Role: llm.RoleUser, Text: "x"}}

	cases := map[string]struct {
		req   llm.Request
		creds yandex.CredentialSource
		class llm.RetryClass
		phase llm.Phase
	}{
		"рассуждения у yandexgpt": {
			req: llm.Request{Model: "yandexgpt/rc", Messages: text, Options: llm.Options{Reasoning: llm.EffortHigh}},
		},
		"кадры у yandexgpt": {
			req: llm.Request{Model: "yandexgpt", Messages: []llm.Message{
				{Role: llm.RoleUser, Images: []llm.Image{{MIME: "image/jpeg", Data: []byte("a")}}},
			}},
		},
		"предел у неизвестной модели": {
			req: llm.Request{Model: "chatgpt", Messages: text, Options: llm.Options{MaxOutputTokens: 5}},
		},
		"схема без имени": {
			req: llm.Request{Model: "yandexgpt", Messages: text, Output: llm.Output{
				Mode:   llm.ModeSchema,
				Schema: []byte(`{}`),
			}},
		},
		"модель с каталогом": {
			req: llm.Request{Model: "gpt://b1g/yandexgpt/latest", Messages: text},
		},
		"не изображение": {
			req: llm.Request{Model: "qwen3.6-35b-a3b", Messages: []llm.Message{
				{Role: llm.RoleUser, Images: []llm.Image{{MIME: "application/pdf", Data: []byte("a")}}},
			}},
		},
		"пустой ключ": {
			req:   llm.Request{Model: "yandexgpt", Messages: text},
			creds: yandex.APIKey(""),
			class: llm.RetryImmediate,
			phase: llm.PhaseAuth,
		},
	}

	for name, tc := range cases {
		t.Run(name, func(t *testing.T) {
			t.Parallel()

			if tc.creds == nil {
				tc.creds, tc.class = yandex.APIKey("k"), llm.RetryNever
			}

			s := &server{reply: always(http.StatusOK, reply("stop", "ok"))}
			c := client(t, s, tc.creds)
			c.Attempts = 1

			_, err := c.Chat(context.Background(), tc.req)

			var call *llm.CallError
			if !errors.As(err, &call) || call.Class != tc.class ||
				(tc.phase != "" && call.Phase != tc.phase) {
				t.Fatalf("отказ %v (%+v)", err, call)
			}

			if n := len(s.requests()); n != 0 {
				t.Errorf("запросов %d", n)
			}
		})
	}
}

// TestProviderChecksProfile — плечо сверяет профиль и без клиента: reasoning_effort не уходит модели, которая его не знает.
func TestProviderChecksProfile(t *testing.T) {
	t.Parallel()

	s := &server{reply: always(http.StatusOK, reply("stop", "ok"))}

	p, err := yandex.New(yandex.Config{Endpoint: s.start(t), Folder: "b1g", Credentials: yandex.APIKey("k")})
	if err != nil {
		t.Fatal(err)
	}

	_, err = p.Complete(context.Background(), llm.Request{
		Model:    "yandexgpt",
		Messages: []llm.Message{{Role: llm.RoleUser, Text: "x"}},
		Options:  llm.Options{Reasoning: llm.EffortLow},
	})

	var request *llm.RequestError
	if !errors.As(err, &request) || len(s.requests()) != 0 {
		t.Fatalf("отказ %v", err)
	}
}

// TestStatusEnvelope — текст отказа из конверта ошибки; 401 — к оператору без повтора, 429 — с паузой.
func TestStatusEnvelope(t *testing.T) {
	t.Parallel()

	cases := map[string]struct {
		status int
		body   string
		class  llm.RetryClass
		msg    string
	}{
		"401 openai": {http.StatusUnauthorized, `{"error":{"message":"bad key","type":"auth"}}`,
			llm.RetryNeedsConfiguration, "bad key"},
		"429 плоский": {http.StatusTooManyRequests, `{"message":"quota"}`, llm.RetryAfterDelay, "quota"},
	}

	for name, tc := range cases {
		t.Run(name, func(t *testing.T) {
			t.Parallel()

			s := &server{reply: always(tc.status, tc.body)}
			c := client(t, s, yandex.APIKey("k"))
			c.Attempts = 1

			_, err := c.Chat(context.Background(), llm.Request{
				Model:    "yandexgpt",
				Messages: []llm.Message{{Role: llm.RoleUser, Text: "x"}},
			})

			var call *llm.CallError
			if !errors.As(err, &call) || call.Class != tc.class || call.Message != tc.msg || call.Status != tc.status {
				t.Fatalf("отказ %v (%+v)", err, call)
			}

			if call.Attempts[0].RequestID != "req-1" {
				t.Errorf("попытка %+v", call.Attempts[0])
			}
		})
	}
}

// TestIAMErrorIsAuthPhase — отказ источника IAM помечен фазой auth и не доходит до сервера.
func TestIAMErrorIsAuthPhase(t *testing.T) {
	t.Parallel()

	boom := errors.New("метаданные недоступны")
	s := &server{reply: always(http.StatusOK, reply("stop", "ok"))}
	c := client(t, s, yandex.IAMToken(func(context.Context) (string, error) { return "", boom }))
	c.Attempts = 1

	_, err := c.Chat(context.Background(), llm.Request{
		Model:    "yandexgpt",
		Messages: []llm.Message{{Role: llm.RoleUser, Text: "x"}},
	})

	var call *llm.CallError
	if !errors.As(err, &call) || call.Phase != llm.PhaseAuth || !errors.Is(err, boom) || len(s.requests()) != 0 {
		t.Fatalf("отказ %v", err)
	}
}

// TestNewRejectsIncompleteConfig — без адреса, каталога и подписи плечо не собирается.
func TestNewRejectsIncompleteConfig(t *testing.T) {
	t.Parallel()

	full := yandex.Config{Endpoint: "http://x", Folder: "b1g", Credentials: yandex.APIKey("k")}

	for name, cfg := range map[string]yandex.Config{
		"адрес":   {Folder: full.Folder, Credentials: full.Credentials},
		"каталог": {Endpoint: full.Endpoint, Credentials: full.Credentials},
		"подпись": {Endpoint: full.Endpoint, Folder: full.Folder},
	} {
		if _, err := yandex.New(cfg); err == nil {
			t.Errorf("%s: собрано", name)
		}
	}

	if _, err := yandex.New(full); err != nil {
		t.Error(err)
	}
}
