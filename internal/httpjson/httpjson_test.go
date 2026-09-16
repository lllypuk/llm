package httpjson_test

import (
	"context"
	"errors"
	"io"
	"net/http"
	"strings"
	"testing"
	"testing/iotest"
	"time"

	"github.com/lllypuk/llm/internal/httpjson"
)

// TestReadAllLimit — ровно предел читается, байт сверх — TooLargeError, а не обрезанное тело.
func TestReadAllLimit(t *testing.T) {
	t.Parallel()

	raw, err := httpjson.ReadAll(strings.NewReader("12345"), 5)
	if err != nil || string(raw) != "12345" {
		t.Errorf("ровно предел: %q, %v", raw, err)
	}

	_, err = httpjson.ReadAll(strings.NewReader("123456"), 5)

	var tooLarge *httpjson.TooLargeError
	if !errors.As(err, &tooLarge) || tooLarge.Limit != 5 {
		t.Errorf("сверх предела: %v", err)
	}
}

// TestReadAllKeepsCause — ошибка чтения не подменяется: просрочка остаётся просрочкой.
func TestReadAllKeepsCause(t *testing.T) {
	t.Parallel()

	_, err := httpjson.ReadAll(iotest.ErrReader(context.DeadlineExceeded), 5)
	if !errors.Is(err, context.DeadlineExceeded) {
		t.Errorf("причина потеряна: %v", err)
	}
}

// TestDecode — один документ с пробелами после; хвост, второй документ и не-JSON — отказ.
func TestDecode(t *testing.T) {
	t.Parallel()

	var v struct {
		A int `json:"a"`
	}

	if err := httpjson.Decode(strings.NewReader("{\"a\":1}\n\n"), 64, &v); err != nil || v.A != 1 {
		t.Errorf("документ: %+v, %v", v, err)
	}

	for _, body := range []string{`{"a":1}}`, `{"a":1}` + "\n" + `{"a":2}`, `<html>502</html>`, ``} {
		if err := httpjson.Decode(strings.NewReader(body), 64, &v); err == nil {
			t.Errorf("%q принят", body)
		}
	}

	var tooLarge *httpjson.TooLargeError
	if err := httpjson.Decode(strings.NewReader(`{"a":123456}`), 8, &v); !errors.As(err, &tooLarge) {
		t.Errorf("предел: %v", err)
	}
}

func errorEnvelope(raw []byte) string {
	if strings.HasPrefix(string(raw), `{"error":"`) {
		return strings.TrimSuffix(strings.TrimPrefix(string(raw), `{"error":"`), `"}`)
	}

	return ""
}

func response(code int, body string, header http.Header) *http.Response {
	return &http.Response{StatusCode: code, Header: header, Body: io.NopCloser(strings.NewReader(body))}
}

// TestReadStatus — текст конверта; не-JSON — тело обрезанным пределом без битой руны; пустое — пусто.
func TestReadStatus(t *testing.T) {
	t.Parallel()

	h := http.Header{}
	h.Set("Retry-After", "3")

	st := httpjson.ReadStatus(response(http.StatusTooManyRequests, `{"error":"quota"}`, h), 64, errorEnvelope)
	if st.Code != http.StatusTooManyRequests || st.Message != "quota" || st.RetryAfter != 3*time.Second {
		t.Errorf("конверт: %+v", st)
	}

	st = httpjson.ReadStatus(response(http.StatusBadGateway, " <html>шлюз</html>", http.Header{}), 14, errorEnvelope)
	if st.Message != "<html>шлю" {
		t.Errorf("не-JSON: %q", st.Message)
	}

	st = httpjson.ReadStatus(response(http.StatusBadGateway, "", http.Header{}), 64, errorEnvelope)
	if st.Message != "" || st.Code != http.StatusBadGateway {
		t.Errorf("пустое тело: %+v", st)
	}
}

// TestNewRequest — POST, JSON-заголовки и дописанные поверх; несериализуемое тело — отказ.
func TestNewRequest(t *testing.T) {
	t.Parallel()

	req, err := httpjson.NewRequest(context.Background(), "http://x/api", map[string]int{"a": 1},
		http.Header{"x-request-id": {"r1"}})
	if err != nil {
		t.Fatal(err)
	}

	body, _ := io.ReadAll(req.Body)
	if req.Method != http.MethodPost || string(body) != `{"a":1}` ||
		req.Header.Get("Content-Type") != "application/json" || req.Header.Get("X-Request-Id") != "r1" {
		t.Errorf("запрос %s %q %v", req.Method, body, req.Header)
	}

	if _, err = httpjson.NewRequest(context.Background(), "http://x", func() {}, nil); err == nil {
		t.Error("функция в теле принята")
	}
}

// TestRetryAfter — секунды, HTTP-дата, мусор и отрицательное.
func TestRetryAfter(t *testing.T) {
	t.Parallel()

	h := http.Header{}
	h.Set("Retry-After", "7")

	if got := httpjson.RetryAfter(h); got != 7*time.Second {
		t.Errorf("секунды: %s", got)
	}

	h.Set("Retry-After", time.Now().Add(time.Minute).UTC().Format(http.TimeFormat))
	if got := httpjson.RetryAfter(h); got < 50*time.Second || got > time.Minute {
		t.Errorf("дата: %s", got)
	}

	for _, raw := range []string{"", "мусор", "-5", "0", time.Now().Add(-time.Minute).UTC().Format(http.TimeFormat)} {
		h.Set("Retry-After", raw)
		if got := httpjson.RetryAfter(h); got != 0 {
			t.Errorf("%q: %s", raw, got)
		}
	}
}

// TestRetryAfterSaturates — чрезмерное значение насыщается, а не переполняется в минус.
func TestRetryAfterSaturates(t *testing.T) {
	t.Parallel()

	h := http.Header{}

	for _, raw := range []string{"9223372037", "9223372036854775808", "99999999999999999999999"} {
		h.Set("Retry-After", raw)

		if got := httpjson.RetryAfter(h); got < time.Hour {
			t.Errorf("Retry-After %s переполнился: %s", raw, got)
		}
	}

	h.Set("Retry-After", "-9223372036854775808")
	if got := httpjson.RetryAfter(h); got != 0 {
		t.Errorf("отрицательное переполнение: %s", got)
	}
}
