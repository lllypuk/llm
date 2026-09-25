package yandex_test

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/lllypuk/llm"
	"github.com/lllypuk/llm/yandex"
)

// ocrServer — `/ocr/v1/recognizeText` с одним ответом; запоминает запросы.
type ocrServer struct {
	status int
	body   string
	header http.Header

	mu     sync.Mutex
	reqs   []*http.Request
	bodies [][]byte
}

func (s *ocrServer) start(t *testing.T) string {
	t.Helper()

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		raw, _ := io.ReadAll(r.Body)

		s.mu.Lock()
		s.reqs = append(s.reqs, r)
		s.bodies = append(s.bodies, raw)
		s.mu.Unlock()

		for k, v := range s.header {
			w.Header()[k] = v
		}

		w.Header().Set("X-Request-Id", "ocr-1")
		w.WriteHeader(s.status)
		_, _ = io.WriteString(w, s.body)
	}))
	t.Cleanup(srv.Close)

	return srv.URL + "/ocr/v1/"
}

func (s *ocrServer) count() int {
	s.mu.Lock()
	defer s.mu.Unlock()

	return len(s.reqs)
}

func ocr(t *testing.T, endpoint string, creds yandex.CredentialSource) *yandex.OCR {
	t.Helper()

	p, err := yandex.NewOCR(yandex.OCRConfig{Endpoint: endpoint, Folder: "b1g", Credentials: creds})
	if err != nil {
		t.Fatal(err)
	}

	return p
}

func ocrRequest(image []byte) llm.OCRRequest {
	return llm.OCRRequest{Model: "page", Image: image, MIME: llm.MIMEPNG, Languages: []string{"ru", "en"}}
}

// ocrAnswer — ответ `recognizeText` в документированной обёртке `result`.
func ocrAnswer(text string) string {
	raw, _ := json.Marshal(map[string]any{
		"result": map[string]any{
			"textAnnotation": map[string]any{"width": "100", "height": "100", "blocks": []any{}, "fullText": text},
			"page":           "0",
		},
	})

	return string(raw)
}

// TestOCRExactRequestAndResult — адрес, заголовки и тело; ключ только в заголовке; ответ в OCRText.
func TestOCRExactRequestAndResult(t *testing.T) {
	t.Parallel()

	s := &ocrServer{status: http.StatusOK, body: ocrAnswer(" Счёт №12\nИтого 500 ")}
	p := ocr(t, s.start(t), yandex.APIKey("secret-key"))

	image := []byte{0x89, 'P', 'N', 'G', 0, 1, 2}

	got, err := p.Recognize(context.Background(), ocrRequest(image))
	if err != nil {
		t.Fatal(err)
	}

	want := llm.OCRText{Text: "Счёт №12\nИтого 500", Model: "page", RequestID: "ocr-1"}
	if got != want {
		t.Errorf("результат %+v, ждали %+v", got, want)
	}

	r := s.reqs[0]
	if r.Method != http.MethodPost || r.URL.Path != "/ocr/v1/recognizeText" || r.URL.RawQuery != "" {
		t.Errorf("запрос %s %s?%s", r.Method, r.URL.Path, r.URL.RawQuery)
	}

	for k, v := range map[string]string{
		"Authorization":          "Api-Key secret-key",
		"X-Folder-Id":            "b1g",
		"X-Data-Logging-Enabled": "false",
		"Content-Type":           "application/json",
	} {
		if got := r.Header.Get(k); got != v {
			t.Errorf("%s %q, ждали %q", k, got, v)
		}
	}

	if strings.Contains(string(s.bodies[0]), "secret-key") {
		t.Error("ключ в теле")
	}

	var body map[string]any
	if err = json.Unmarshal(s.bodies[0], &body); err != nil {
		t.Fatal(err)
	}

	wantBody := map[string]any{
		"mimeType":      "PNG",
		"languageCodes": []any{"ru", "en"},
		"model":         "page",
		"content":       base64.StdEncoding.EncodeToString(image),
	}
	if b, w := mustJSON(t, body), mustJSON(t, wantBody); b != w {
		t.Errorf("тело %s, ждали %s", b, w)
	}
}

func mustJSON(t *testing.T, v any) string {
	t.Helper()

	raw, err := json.Marshal(v)
	if err != nil {
		t.Fatal(err)
	}

	return string(raw)
}

// TestOCRJPEGMimeType — image/jpeg уходит как JPEG.
func TestOCRJPEGMimeType(t *testing.T) {
	t.Parallel()

	s := &ocrServer{status: http.StatusOK, body: ocrAnswer("x")}

	req := ocrRequest([]byte{0xff, 0xd8})
	req.MIME = llm.MIMEJPEG

	if _, err := ocr(t, s.start(t), yandex.APIKey("k")).Recognize(context.Background(), req); err != nil {
		t.Fatal(err)
	}

	if !strings.Contains(string(s.bodies[0]), `"mimeType":"JPEG"`) {
		t.Errorf("тело %s", s.bodies[0])
	}
}

// TestOCRLargeResponse — плотная страница в несколько мегабайт координат разбирается.
func TestOCRLargeResponse(t *testing.T) {
	t.Parallel()

	words := make([]any, 0, 40000)
	for range 40000 {
		words = append(words, map[string]any{
			"boundingBox": map[string]any{"vertices": []any{
				map[string]string{"x": "100", "y": "200"}, map[string]string{"x": "150", "y": "200"},
				map[string]string{"x": "150", "y": "220"}, map[string]string{"x": "100", "y": "220"},
			}},
			"text": "слово", "entityIndex": "-1",
		})
	}

	raw, _ := json.Marshal(map[string]any{
		"result": map[string]any{"textAnnotation": map[string]any{
			"blocks":   []any{map[string]any{"lines": []any{map[string]any{"words": words}}}},
			"fullText": "плотная страница",
		}},
	})
	if len(raw) < 4<<20 {
		t.Fatalf("ответ %d байт, ждали несколько МБ", len(raw))
	}

	s := &ocrServer{status: http.StatusOK, body: string(raw)}

	got, err := ocr(t, s.start(t), yandex.APIKey("k")).Recognize(context.Background(), ocrRequest([]byte{1}))
	if err != nil || got.Text != "плотная страница" {
		t.Fatalf("результат %+v, %v", got, err)
	}
}

// TestOCREmptyTextIsSuccess — текста на кадре нет: успех с пустым текстом, а не отказ.
func TestOCREmptyTextIsSuccess(t *testing.T) {
	t.Parallel()

	for name, body := range map[string]string{"пустой": ocrAnswer(""), "без поля": `{"result":{}}`} {
		t.Run(name, func(t *testing.T) {
			t.Parallel()

			s := &ocrServer{status: http.StatusOK, body: body}

			got, err := ocr(t, s.start(t), yandex.APIKey("k")).Recognize(context.Background(), ocrRequest([]byte{1}))
			if err != nil || got.Text != "" {
				t.Fatalf("результат %+v, %v", got, err)
			}
		})
	}
}

// TestOCRStatus — не-2xx становится StatusError с текстом конверта, Retry-After и идентификатором.
func TestOCRStatus(t *testing.T) {
	t.Parallel()

	cases := map[string]struct {
		status int
		body   string
		header http.Header
		want   llm.StatusError
	}{
		"400": {
			http.StatusBadRequest, `{"code":3,"message":"Image is too large","details":[]}`, nil,
			llm.StatusError{Status: http.StatusBadRequest, Message: "Image is too large", RequestID: "ocr-1"},
		},
		"429": {
			http.StatusTooManyRequests, `{"code":8}`, http.Header{"Retry-After": {"1"}},
			llm.StatusError{
				Status: http.StatusTooManyRequests, Message: "8",
				RetryAfter: time.Second, RequestID: "ocr-1",
			},
		},
		"503": {
			http.StatusServiceUnavailable, "upstream down", nil,
			llm.StatusError{Status: http.StatusServiceUnavailable, Message: "upstream down", RequestID: "ocr-1"},
		},
	}

	for name, tc := range cases {
		t.Run(name, func(t *testing.T) {
			t.Parallel()

			s := &ocrServer{status: tc.status, body: tc.body, header: tc.header}

			_, err := ocr(t, s.start(t), yandex.APIKey("k")).Recognize(context.Background(), ocrRequest([]byte{1}))

			var st *llm.StatusError
			if !errors.As(err, &st) || *st != tc.want {
				t.Fatalf("отказ %v (%+v), ждали %+v", err, st, tc.want)
			}
		})
	}
}

// TestOCRRejectedBeforeNetwork — крупный и негодный кадр не доходят до сервера, как и отказ IAM.
func TestOCRRejectedBeforeNetwork(t *testing.T) {
	t.Parallel()

	boom := errors.New("метаданные недоступны")

	cases := map[string]struct {
		creds yandex.CredentialSource
		req   llm.OCRRequest
	}{
		"больше 10 МБ": {yandex.APIKey("k"), ocrRequest(make([]byte, 10<<20+1))},
		"без модели":   {yandex.APIKey("k"), llm.OCRRequest{Image: []byte{1}, MIME: llm.MIMEPNG}},
		"PDF":          {yandex.APIKey("k"), llm.OCRRequest{Model: "page", Image: []byte{1}, MIME: "application/pdf"}},
		"отказ IAM": {
			yandex.IAMToken(func(context.Context) (string, error) { return "", boom }),
			ocrRequest([]byte{1}),
		},
	}

	for name, tc := range cases {
		t.Run(name, func(t *testing.T) {
			t.Parallel()

			s := &ocrServer{status: http.StatusOK, body: ocrAnswer("x")}

			_, err := ocr(t, s.start(t), tc.creds).Recognize(context.Background(), tc.req)

			var reqErr *llm.RequestError

			var phase *llm.PhaseError
			if !errors.As(err, &reqErr) && (!errors.As(err, &phase) || phase.Phase != llm.PhaseAuth) {
				t.Errorf("отказ %v", err)
			}

			if s.count() != 0 {
				t.Error("запрос ушёл в сеть")
			}
		})
	}
}

// TestOCRCapabilities — `page` с пределом 10 МБ, неизвестная не подтверждается.
func TestOCRCapabilities(t *testing.T) {
	t.Parallel()

	p := ocr(t, "http://x", yandex.APIKey("k"))

	if caps, ok := p.OCRCapabilities("page"); !ok || caps.MaxBytes != 10<<20 {
		t.Errorf("page %+v %v", caps, ok)
	}

	if _, ok := p.OCRCapabilities("handwritten"); ok {
		t.Error("неизвестная модель подтверждена")
	}

	if p.Name() != yandex.OCRName {
		t.Error(p.Name())
	}
}

// TestNewOCRRejectsIncompleteConfig — без адреса, каталога и подписи плечо не собирается.
func TestNewOCRRejectsIncompleteConfig(t *testing.T) {
	t.Parallel()

	full := yandex.OCRConfig{Endpoint: "http://x", Folder: "b1g", Credentials: yandex.APIKey("k")}

	for name, cfg := range map[string]yandex.OCRConfig{
		"адрес":   {Folder: full.Folder, Credentials: full.Credentials},
		"каталог": {Endpoint: full.Endpoint, Credentials: full.Credentials},
		"подпись": {Endpoint: full.Endpoint, Folder: full.Folder},
	} {
		if _, err := yandex.NewOCR(cfg); err == nil {
			t.Errorf("%s: собрано", name)
		}
	}
}
