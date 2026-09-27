//go:build live

package main

import (
	"bytes"
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"sync/atomic"
	"testing"
	"time"
)

const (
	pageText  = "Счёт № 42"
	ocrAPIKey = "ocr-key"
	ocrTimer  = time.Minute
)

// Кадры — сигнатура формата и нули: http.DetectContentType узнаёт их по первым байтам.
const (
	pngFrame  = "\x89PNG\r\n\x1a\n\x00\x00\x00\x0dIHDR"
	jpegFrame = "\xff\xd8\xff\xe0\x00\x10JFIF\x00"
)

// fakeVision — recognizeText: на PNG отвечает pageText, на JPEG — пустым текстом.
func fakeVision(t *testing.T, calls *atomic.Int32) *httptest.Server {
	t.Helper()

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		calls.Add(1)

		var req struct {
			MIMEType string `json:"mimeType"`
		}

		if r.Header.Get("Authorization") != "Api-Key "+ocrAPIKey ||
			r.Header.Get("X-Data-Logging-Enabled") != "false" ||
			json.NewDecoder(r.Body).Decode(&req) != nil {
			w.WriteHeader(http.StatusBadRequest)

			return
		}

		text := ""
		if req.MIMEType == "PNG" {
			text = pageText
		}

		_, _ = io.WriteString(w, `{"result": {"textAnnotation": {"fullText": "`+text+`"}}}`)
	}))
	t.Cleanup(srv.Close)

	return srv
}

func writeOCRConfig(t *testing.T, endpoint string) string {
	t.Helper()

	return writeFile(t, `{
  "providers": {
    "local": {"kind": "ollama", "endpoint": "http://ollama.invalid"},
    "stt": {"kind": "speechkit", "endpoint": "https://stt.invalid", "folder": "b1g", "auth": {"api_key": "${STT_KEY_UNSET}"}},
    "vision": {"kind": "visionocr", "endpoint": "`+endpoint+`", "folder": "b1g", "auth": {"api_key": "${OCR_KEY}"}},
    "spare": {"kind": "visionocr", "endpoint": "https://ocr.invalid", "folder": "b1g", "auth": {"api_key": "${SPARE_UNSET}"}}
  },
  "tasks": {"ask": {"provider": "local", "model": "gemma"}},
  "speech": {"dictation": {"provider": "stt", "model": "general"}},
  "ocr": {
    "scan": {"provider": "vision", "model": "page", "languages": ["ru", "en"], "attempts": 1, "price_plan": "ocr"},
    "backup": {"provider": "spare", "model": "page", "languages": ["ru"]}
  },
  "prices": {
    "ocr": [{"revision": "ocr-1", "currency": "RUB", "valid_from": "2026-01-01T00:00:00Z", "rates": {"page": 132100}}]
  }
}`)
}

func ocrEnv(name string) (string, bool) {
	if name == "OCR_KEY" {
		return ocrAPIKey, true
	}

	return "", false
}

func ocrOpts(config, dir string) dirOptions {
	return dirOptions{
		config:      config,
		dir:         dir,
		task:        "scan",
		maxFiles:    defaultMaxFiles,
		maxRequests: defaultMaxRequests,
		maxCost:     defaultMaxCost,
		timeout:     ocrTimer,
	}
}

func TestOCRRun(t *testing.T) {
	t.Parallel()

	var calls atomic.Int32

	srv := fakeVision(t, &calls)
	dir := writeClips(t, map[string][]byte{
		"a.png":     []byte(pngFrame),
		"b.JPG":     []byte(jpegFrame),
		"notes.txt": []byte("не кадр"),
	})

	var stdout, stderr bytes.Buffer

	code := dispatch(context.Background(),
		[]string{cmdOCR, "-config", writeOCRConfig(t, srv.URL), "-dir", dir, "-task", "scan"}, &stdout, &stderr, ocrEnv)
	if code != exitOK {
		t.Fatalf("код %d, stderr %s\n%s", code, stderr.String(), stdout.String())
	}

	out := stdout.String()
	for _, want := range []string{
		"# llmcheck ocr", "- задача: scan: vision/page, языки ru,en", "тариф ocr",
		"- [pass] ocr a.png: задержка", "страниц 1, попыток 1, стоимость 0.132100 RUB по ocr-1 (estimated)",
		"- [pass] ocr b.JPG", "пустой текст",
		"пройдено 2, нарушений 0", "обращений 2 из 20",
	} {
		if !strings.Contains(out, want) {
			t.Errorf("в протоколе нет %q:\n%s", want, out)
		}
	}

	for _, leak := range []string{pageText, ocrAPIKey, dir, "stt", "gemma", "spare"} {
		if strings.Contains(out, leak) {
			t.Errorf("в протокол попало %q:\n%s", leak, out)
		}
	}

	for name, want := range map[string]string{"a.txt": pageText, "b.txt": ""} {
		got, err := os.ReadFile(filepath.Join(dir, name))
		if err != nil || string(got) != want {
			t.Errorf("%s: %q, %v; ждали %q", name, got, err, want)
		}
	}
}

// TestOCRRefusesBeforeCalls — потолок кадров и чужое содержимое отбивают прогон до первого обращения.
func TestOCRRefusesBeforeCalls(t *testing.T) {
	t.Parallel()

	var calls atomic.Int32

	srv := fakeVision(t, &calls)
	config := writeOCRConfig(t, srv.URL)

	cases := map[string]struct {
		frames   map[string][]byte
		maxFiles int
		task     string
		want     string
	}{
		"потолок кадров": {
			map[string][]byte{"a.png": []byte(pngFrame), "b.png": []byte(pngFrame), "c.jpg": []byte(jpegFrame)},
			2,
			"scan",
			"кадров 3, потолок 2",
		},
		"не картинка": {map[string][]byte{"a.png": []byte(pngFrame), "bad.png": []byte("GIF89a......")}, 5, "scan",
			"bad.png: содержимое image/gif"},
		"нет кадров": {map[string][]byte{"a.txt": []byte(pngFrame)}, 5, "scan", "нет кадров"},
		"jpg и png": {map[string][]byte{"scan.jpg": []byte(jpegFrame), "scan.png": []byte(pngFrame)}, 5, "scan",
			"scan.jpg и scan.png пишут текст в один scan.txt"},
		"jpg и jpeg": {map[string][]byte{"page.jpg": []byte(jpegFrame), "page.jpeg": []byte(jpegFrame)}, 5, "scan",
			"пишут текст в один page.txt"},
		"регистр": {map[string][]byte{"a.PNG": []byte(pngFrame), "a.png": []byte(pngFrame)}, 5, "scan",
			"a.PNG и a.png пишут текст в один a.txt"},
		"регистр основы": {map[string][]byte{"Scan.png": []byte(pngFrame), "scan.jpg": []byte(jpegFrame)}, 5, "scan",
			"пишут текст в один"},
		"задачи не выбрать": {map[string][]byte{"a.png": []byte(pngFrame)}, 5, "", "в разделе ocr задач 2, нужна одна"},
		"задача речи":       {map[string][]byte{"a.png": []byte(pngFrame)}, 5, "dictation", `"dictation" не объявлена`},
	}

	for name, tc := range cases {
		t.Run(name, func(t *testing.T) {
			t.Parallel()

			o := ocrOpts(config, writeClips(t, tc.frames))
			o.maxFiles, o.task = tc.maxFiles, tc.task

			var stdout, stderr bytes.Buffer

			code := runOCR(context.Background(), o, &stdout, &stderr, ocrEnv)
			if code != exitUsage || !strings.Contains(stderr.String(), tc.want) {
				t.Fatalf("код %d, stderr %q, ждали %q", code, stderr.String(), tc.want)
			}
		})
	}

	if n := calls.Load(); n != 0 {
		t.Fatalf("обращений к плечу %d, ждали ноль", n)
	}
}

func TestOCRRequestCap(t *testing.T) {
	t.Parallel()

	var calls atomic.Int32

	srv := fakeVision(t, &calls)
	dir := writeClips(t, map[string][]byte{"a.png": []byte(pngFrame), "b.png": []byte(pngFrame)})
	o := ocrOpts(writeOCRConfig(t, srv.URL), dir)
	o.maxRequests = 1

	var stdout, stderr bytes.Buffer

	if code := runOCR(context.Background(), o, &stdout, &stderr, ocrEnv); code != exitIncomplete {
		t.Fatalf("код %d, ждали %d:\n%s", code, exitIncomplete, stdout.String())
	}

	if n := calls.Load(); n != 1 {
		t.Fatalf("обращений %d, ждали одно", n)
	}

	if out := stdout.String(); !strings.Contains(out, "- [skip] ocr b.png") ||
		!strings.Contains(out, "страниц 0, попыток 1, стоимость 0.000000 RUB") {
		t.Fatalf("второй кадр не пропущен или оплачен:\n%s", out)
	}
}

func TestOCRProviderFailure(t *testing.T) {
	t.Parallel()

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusUnauthorized)
	}))
	t.Cleanup(srv.Close)

	dir := writeClips(t, map[string][]byte{"a.png": []byte(pngFrame)})

	var stdout, stderr bytes.Buffer

	code := runOCR(context.Background(), ocrOpts(writeOCRConfig(t, srv.URL), dir), &stdout, &stderr, ocrEnv)
	if out := stdout.String(); code != exitViolation || !strings.Contains(out, "HTTP 401") ||
		!strings.Contains(out, "страниц 0") {
		t.Fatalf("код %d:\n%s", code, out)
	}

	if _, err := os.Stat(filepath.Join(dir, "a.txt")); !os.IsNotExist(err) {
		t.Fatalf("текст отказа записан: %v", err)
	}
}
