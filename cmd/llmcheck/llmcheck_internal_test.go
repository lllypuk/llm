//go:build live

package main

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"flag"
	"io"
	"maps"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"
	"time"

	"github.com/lllypuk/llm"
)

func TestParseFlags(t *testing.T) {
	t.Parallel()

	o, err := parseFlags([]string{"-config", "llm.json", "-tasks", "ask, nameplate,,ask"}, io.Discard)
	if err != nil {
		t.Fatalf("parseFlags: %v", err)
	}

	if o.config != "llm.json" || !slices.Equal(o.tasks, []string{"ask", "nameplate"}) {
		t.Fatalf("config %q, tasks %v", o.config, o.tasks)
	}

	if !slices.Equal(o.checks, defaultChecks()) || slices.Contains(o.checks, checkLimits) {
		t.Fatalf("проверки по умолчанию %v", o.checks)
	}

	if o.maxRequests != defaultMaxRequests || o.maxCost != defaultMaxCost || o.timeout != defaultTimeout {
		t.Fatalf("потолки по умолчанию %+v", o)
	}
}

func TestParseFlagsRejects(t *testing.T) {
	t.Parallel()

	cases := map[string]struct {
		args []string
		want string
	}{
		"без конфига":          {nil, "-config"},
		"неизвестная проверка": {[]string{"-config", "c", "-checks", "model,bogus"}, `"bogus"`},
		"пустые проверки":      {[]string{"-config", "c", "-checks", " , "}, "не выбраны"},
		"нулевой потолок":      {[]string{"-config", "c", "-max-requests", "0"}, "-max-requests"},
		"отрицательный расход": {[]string{"-config", "c", "-max-cost", "-1"}, "-max-cost"},
		"нулевой срок":         {[]string{"-config", "c", "-timeout", "0s"}, "-timeout"},
		"лишний аргумент":      {[]string{"-config", "c", "extra"}, "лишние аргументы"},
	}

	for name, tc := range cases {
		t.Run(name, func(t *testing.T) {
			t.Parallel()

			_, err := parseFlags(tc.args, io.Discard)
			if err == nil || !strings.Contains(err.Error(), tc.want) {
				t.Fatalf("ошибка %v, ждали %q", err, tc.want)
			}
		})
	}

	if _, err := parseFlags([]string{"-h"}, io.Discard); !errors.Is(err, flag.ErrHelp) {
		t.Fatalf("-h: %v", err)
	}
}

// fakeOllama отвечает всем полям проверок сразу, маршруту со схемой — только полем схемы, текстовому —
// словом проверки связи; на предел в один токен — обрезанным ответом.
func fakeOllama(t *testing.T, calls *int) *httptest.Server {
	t.Helper()

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		*calls++

		var req struct {
			Format  json.RawMessage `json:"format"`
			Options struct {
				NumPredict int `json:"num_predict"`
			} `json:"options"`
		}

		_ = json.NewDecoder(r.Body).Decode(&req)

		full := map[string]any{fieldOK: true, fieldColor: "красный", fieldAnswer: 391}

		var schema struct {
			Properties map[string]any `json:"properties"`
		}

		if json.Unmarshal(req.Format, &schema) == nil && len(schema.Properties) > 0 {
			maps.DeleteFunc(full, func(k string, _ any) bool { _, ok := schema.Properties[k]; return !ok })
		}

		answer, _ := json.Marshal(full)
		content, reason := string(answer), "stop"

		if len(req.Format) == 0 {
			content = wantTrue
		}

		if req.Options.NumPredict == 1 {
			content, reason = "Пон", "length"
		}

		_ = json.NewEncoder(w).Encode(map[string]any{
			"model":             "gemma",
			"message":           map[string]string{"content": content},
			"done":              true,
			"done_reason":       reason,
			"prompt_eval_count": 12,
			"eval_count":        3,
		})
	}))
	t.Cleanup(srv.Close)

	return srv
}

func writeConfig(t *testing.T, endpoint string) string {
	t.Helper()

	return writeFile(t, `{
  "providers": {
    "local": {"kind": "ollama", "endpoint": "`+endpoint+`"},
    "giga": {"kind": "gigachat", "endpoint": "https://giga.invalid", "oauth_endpoint": "https://oauth.invalid",
             "scope": "PERS", "auth": {"authorization_key": "${GIGACHAT_KEY_UNSET}"}}
  },
  "tasks": {
    "ask": {"provider": "local", "model": "gemma", "output": {"mode": "json"}, "price_plan": "free"},
    "nameplate": {"provider": "local", "model": "gemma", "output": {"mode": "schema", "schema_name": "n"}},
    "twin": {"provider": "local", "model": "gemma", "output": {"mode": "json"}},
    "paid": {"provider": "local", "model": "gemma", "price_plan": "rub"},
    "cloud": {"provider": "giga", "model": "GigaChat-2-Max"}
  },
  "prices": {
    "free": [{"revision": "free-1", "valid_from": "2026-01-01T00:00:00Z", "free": true}],
    "rub": [{"revision": "rub-1", "currency": "RUB", "valid_from": "2026-01-01T00:00:00Z",
             "rates": {"billable_input": 1000000000, "output": 1000000000}}]
  }
}`)
}

func writeFile(t *testing.T, cfg string) string {
	t.Helper()

	path := filepath.Join(t.TempDir(), "llm.json")
	if err := os.WriteFile(path, []byte(cfg), 0o600); err != nil {
		t.Fatal(err)
	}

	return path
}

func noEnv(string) (string, bool) { return "", false }

func TestRunAgainstFakeOllama(t *testing.T) {
	t.Parallel()

	var calls int

	srv := fakeOllama(t, &calls)
	o := options{
		config:      writeConfig(t, srv.URL),
		tasks:       []string{"ask", "nameplate", "twin"},
		checks:      defaultChecks(),
		maxRequests: defaultMaxRequests,
		maxCost:     defaultMaxCost,
		timeout:     time.Minute,
	}

	var stdout, stderr bytes.Buffer

	if code := run(context.Background(), o, &stdout, &stderr, noEnv); code != exitOK {
		t.Fatalf("код %d, stderr %s\n%s", code, stderr.String(), stdout.String())
	}

	out := stdout.String()
	for _, want := range []string{
		"[pass] model ask", "[pass] vision nameplate", "[skip] reasoning ask", "[pass] finish ask",
		"[skip] model twin", "маршрут тот же, что у задачи ask", "[skip] finish nameplate",
		"пройдено 5, нарушений 0",
	} {
		if !strings.Contains(out, want) {
			t.Errorf("в протоколе нет %q:\n%s", want, out)
		}
	}

	if strings.Contains(out, "giga") {
		t.Errorf("неактивное плечо попало в протокол:\n%s", out)
	}

	if calls != 5 {
		t.Errorf("обращений к плечу %d, ждали 5", calls)
	}
}

func TestRunStopsAtRequestCap(t *testing.T) {
	t.Parallel()

	var calls int

	srv := fakeOllama(t, &calls)
	o := options{
		config:      writeConfig(t, srv.URL),
		tasks:       []string{"ask"},
		checks:      defaultChecks(),
		maxRequests: 1,
		maxCost:     defaultMaxCost,
		timeout:     time.Minute,
	}

	var stdout bytes.Buffer

	if code := run(context.Background(), o, &stdout, io.Discard, noEnv); code != exitIncomplete {
		t.Fatalf("код %d, ждали %d:\n%s", code, exitIncomplete, stdout.String())
	}

	if calls != 1 || !strings.Contains(stdout.String(), "прогон неполный") {
		t.Fatalf("обращений %d:\n%s", calls, stdout.String())
	}
}

func TestRunRejectsUnknownTask(t *testing.T) {
	t.Parallel()

	o := options{config: writeConfig(t, "http://127.0.0.1:1"), tasks: []string{"nope"}, checks: defaultChecks(),
		maxRequests: 1, maxCost: 1, timeout: time.Minute}

	var stderr bytes.Buffer

	if code := run(context.Background(), o, io.Discard, &stderr, noEnv); code != exitUsage {
		t.Fatalf("код %d", code)
	}

	if !strings.Contains(stderr.String(), `"nope"`) {
		t.Fatalf("stderr %s", stderr.String())
	}
}

// TestRunNeedsOnlySelectedSecrets — ключ плеча невыбранной задачи не нужен, а без -tasks его отсутствие
// — отказ конфига.
func TestRunNeedsOnlySelectedSecrets(t *testing.T) {
	t.Parallel()

	var calls int

	srv := fakeOllama(t, &calls)
	o := options{config: writeConfig(t, srv.URL), checks: []string{checkModel},
		maxRequests: defaultMaxRequests, maxCost: defaultMaxCost, timeout: time.Minute}

	var stderr bytes.Buffer

	if code := run(context.Background(), o, io.Discard, &stderr, noEnv); code != exitUsage ||
		!strings.Contains(stderr.String(), "GIGACHAT_KEY_UNSET") {
		t.Fatalf("все задачи: код %d, stderr %s", code, stderr.String())
	}

	o.tasks = []string{"ask"}

	if code := run(context.Background(), o, io.Discard, &stderr, noEnv); code != exitOK {
		t.Fatalf("задача ask: код %d, stderr %s", code, stderr.String())
	}
}

// TestRunFailsWhenLastCallOverspends — вызов, перелетевший потолок расхода, не оставляет выход нулевым.
func TestRunFailsWhenLastCallOverspends(t *testing.T) {
	t.Parallel()

	var calls int

	srv := fakeOllama(t, &calls)
	o := options{config: writeConfig(t, srv.URL), tasks: []string{"paid"}, checks: []string{checkModel},
		maxRequests: defaultMaxRequests, maxCost: 1, timeout: time.Minute}

	var stdout bytes.Buffer

	if code := run(context.Background(), o, &stdout, io.Discard, noEnv); code != exitIncomplete ||
		!strings.Contains(stdout.String(), "потолок расхода 1 мк. превышен") {
		t.Fatalf("код %d:\n%s", code, stdout.String())
	}
}

// TestProtocolHidesProviderText — ни текст отказа поставщика, ни адрес с учётными данными в протокол
// не попадают.
func TestProtocolHidesProviderText(t *testing.T) {
	t.Parallel()

	const leak = "текст-ответа-модели"

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusBadRequest)
		_ = json.NewEncoder(w).Encode(map[string]string{"error": leak})
	}))
	t.Cleanup(srv.Close)

	o := options{config: writeConfig(t, srv.URL), tasks: []string{"ask"}, checks: []string{checkModel},
		maxRequests: defaultMaxRequests, maxCost: defaultMaxCost, timeout: time.Minute}

	var stdout bytes.Buffer

	if code := run(context.Background(), o, &stdout, io.Discard, noEnv); code != exitViolation {
		t.Fatalf("код %d:\n%s", code, stdout.String())
	}

	if strings.Contains(stdout.String(), leak) || !strings.Contains(stdout.String(), "HTTP 400") {
		t.Fatalf("протокол:\n%s", stdout.String())
	}

	o.config = writeFile(t, `{"providers": {"local": {"kind": "ollama", "endpoint": "http://u:hunter2@o"}},
  "tasks": {"ask": {"provider": "local", "model": "gemma"}}}`)

	var stderr bytes.Buffer

	if code := run(context.Background(), o, &stdout, &stderr, noEnv); code != exitUsage ||
		strings.Contains(stderr.String(), "hunter2") {
		t.Fatalf("адрес с паролем: код %d, stderr %s", code, stderr.String())
	}
}

// TestAnswerMatchesChecksTypes — поле ответа сверяется типом и значением, у схемы лишних полей нет.
func TestAnswerMatchesChecksTypes(t *testing.T) {
	t.Parallel()

	isTrue := func(v any) bool { return v == true }

	cases := []struct {
		mode llm.Mode
		text string
		want bool
	}{
		{llm.ModeSchema, `{"ok":true}`, true},
		{llm.ModeSchema, `{"ok":"not true","extra":1}`, false},
		{llm.ModeSchema, `{"ok":true,"extra":1}`, false},
		{llm.ModeJSON, `{"ok":true,"extra":1}`, true},
		{llm.ModeJSON, `{"ok":"true"}`, false},
		{llm.ModeJSON, `{"ok":true} {"ok":false}`, false},
		{llm.ModeText, "true", true},
		{llm.ModeText, "нет", false},
		{llm.ModeText, "True.", true},
		{llm.ModeText, "not true", false},
	}

	for _, tc := range cases {
		if got := answerMatches(tc.mode, tc.text, fieldOK, isTrue, wantTrue); got != tc.want {
			t.Errorf("%s %s: %t", tc.mode, tc.text, got)
		}
	}
}

// TestAnswerMatchesRejectsNegatedColor — отрицание цвета не проходит ни полем, ни текстом.
func TestAnswerMatchesRejectsNegatedColor(t *testing.T) {
	t.Parallel()

	isRed := func(v any) bool {
		s, ok := v.(string)

		return ok && isWord(s, colorWords()...)
	}

	cases := []struct {
		mode llm.Mode
		text string
		want bool
	}{
		{llm.ModeSchema, `{"color":"Красный"}`, true},
		{llm.ModeSchema, `{"color":"not red"}`, false},
		{llm.ModeJSON, `{"color":"некрасный"}`, false},
		{llm.ModeText, "Красный.", true},
		{llm.ModeText, "не красный", false},
	}

	for _, tc := range cases {
		if got := answerMatches(tc.mode, tc.text, fieldColor, isRed, colorWords()...); got != tc.want {
			t.Errorf("%s %s: %t", tc.mode, tc.text, got)
		}
	}
}
