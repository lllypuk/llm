//go:build live

package main

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"flag"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"
	"time"
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

// fakeOllama отвечает всем полям проверок сразу, а на предел в один токен — обрезанным ответом.
func fakeOllama(t *testing.T, calls *int) *httptest.Server {
	t.Helper()

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		*calls++

		var req struct {
			Options struct {
				NumPredict int `json:"num_predict"`
			} `json:"options"`
		}

		_ = json.NewDecoder(r.Body).Decode(&req)

		content, reason := `{"ok":true,"color":"красный","answer":391}`, "stop"
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

	cfg := `{
  "providers": {
    "local": {"kind": "ollama", "endpoint": "` + endpoint + `"},
    "giga": {"kind": "gigachat", "endpoint": "https://giga.invalid", "oauth_endpoint": "https://oauth.invalid",
             "scope": "PERS", "auth": {"authorization_key": "${GIGACHAT_KEY_UNSET}"}}
  },
  "tasks": {
    "ask": {"provider": "local", "model": "gemma", "output": {"mode": "json"}, "price_plan": "free"},
    "nameplate": {"provider": "local", "model": "gemma", "output": {"mode": "schema", "schema_name": "n"}},
    "twin": {"provider": "local", "model": "gemma", "output": {"mode": "json"}}
  },
  "prices": {"free": [{"revision": "free-1", "valid_from": "2026-01-01T00:00:00Z", "free": true}]}
}`

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
