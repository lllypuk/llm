package llmbuild_test

import (
	"maps"
	"slices"
	"strings"
	"testing"

	"github.com/lllypuk/llm/llmbuild"
	"github.com/lllypuk/llm/llmconfig"
)

func TestRouterMaps(t *testing.T) {
	r, err := llmbuild.Router(*load(t), llmbuild.Options{})
	if err != nil {
		t.Fatal(err)
	}

	for _, c := range []struct{ got, want []string }{
		{slices.Sorted(maps.Keys(r.Providers)), []string{"giga", "local", "yc"}},
		{slices.Sorted(maps.Keys(r.Speech)), []string{"salute", "stt"}},
		{slices.Sorted(maps.Keys(r.OCR)), []string{"vision"}},
		{slices.Sorted(maps.Keys(r.Tasks)), []string{"ask", "chat", "draft"}},
		{slices.Sorted(maps.Keys(r.SpeechTasks)), []string{"call", "dictation"}},
		{slices.Sorted(maps.Keys(r.OCRTasks)), []string{"scan"}},
	} {
		if !slices.Equal(c.got, c.want) {
			t.Errorf("ключи %v, want %v", c.got, c.want)
		}
	}

	draft, call, scan := r.Tasks["draft"], r.SpeechTasks["call"], r.OCRTasks["scan"]
	if draft.Provider != "yc" || call.Model != "general" || scan.Model != "page" {
		t.Errorf("задачи не из конфига: %+v %+v %+v", draft, call, scan)
	}

	if err = r.Validate(nil); err != nil {
		t.Fatal(err)
	}
}

func TestRouterSkipsInactive(t *testing.T) {
	const data = `{
  "providers": {
    "local": {"kind": "ollama", "endpoint": "http://ollama:11434"},
    "spare": {"kind": "gigachat", "endpoint": "https://giga.invalid", "oauth_endpoint": "https://oauth.invalid",
              "scope": "GIGACHAT_API_PERS", "ca_file": "/nonexistent/ca.pem", "auth": {"authorization_key": "${UNSET}"}}
  },
  "tasks": {"ask": {"provider": "local", "model": "gemma"}}
}`

	cfg, err := llmconfig.Load([]byte(data), func(string) (string, bool) { return "", false })
	if err != nil {
		t.Fatal(err)
	}

	r, err := llmbuild.Router(*cfg, llmbuild.Options{})
	if err != nil {
		t.Fatal(err)
	}

	if _, ok := r.Providers["spare"]; ok || r.Providers["local"] == nil {
		t.Fatalf("плечи %v", slices.Sorted(maps.Keys(r.Providers)))
	}
}

func TestRouterJoinsArmErrors(t *testing.T) {
	junk := []byte("не PEM")

	r, err := llmbuild.Router(*load(t), llmbuild.Options{PEM: map[string][]byte{"yc": junk, "local": junk}})
	if err == nil {
		t.Fatalf("ошибки нет, роутер %v", r)
	}

	msg := err.Error()
	local, yc := strings.Index(msg, "providers.local:"), strings.Index(msg, "providers.yc:")

	if local < 0 || yc < 0 || local > yc {
		t.Fatalf("ошибка %q: нужны обе, local раньше yc", msg)
	}
}

func TestRouterRefusesUnexpanded(t *testing.T) {
	cfg, err := llmconfig.Parse([]byte(allKinds))
	if err != nil {
		t.Fatal(err)
	}

	r, err := llmbuild.Router(*cfg, llmbuild.Options{})
	if err == nil {
		t.Fatalf("ошибки нет, роутер %v", r)
	}

	for _, arm := range []string{"giga", "salute", "yc", "stt", "vision"} {
		if !strings.Contains(err.Error(), "providers."+arm+": ") {
			t.Errorf("нет плеча %s в %q", arm, err)
		}
	}

	if strings.Contains(err.Error(), "providers.local") {
		t.Errorf("ollama в ошибке: %q", err)
	}
}
