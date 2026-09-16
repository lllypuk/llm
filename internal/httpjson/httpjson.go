// Package httpjson — общее у плеч поверх JSON по HTTP: запрос, ограниченное тело, отказ по статусу.
// Wire-структуры остаются в адаптерах: конверты поставщиков не совпадают.
package httpjson

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math"
	"net/http"
	"strconv"
	"strings"
	"time"
)

// TooLargeError — тело длиннее предела: отказ, а не молча обрезанный ответ.
type TooLargeError struct {
	Limit int64
}

func (e *TooLargeError) Error() string {
	return fmt.Sprintf("тело длиннее %d байт", e.Limit)
}

// Status — не-2xx поставщика: код, текст конверта и просьба подождать.
type Status struct {
	Code       int
	Message    string
	RetryAfter time.Duration
}

// NewRequest — POST с JSON-телом; header дописывается поверх Content-Type и Accept.
func NewRequest(ctx context.Context, url string, body any, header http.Header) (*http.Request, error) {
	raw, err := json.Marshal(body)
	if err != nil {
		return nil, fmt.Errorf("сборка тела: %w", err)
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewReader(raw))
	if err != nil {
		return nil, err
	}

	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Accept", "application/json")

	for name, values := range header {
		req.Header[http.CanonicalHeaderKey(name)] = values
	}

	return req, nil
}

// ReadAll читает не больше limit байт; тело длиннее — [TooLargeError].
// Ошибка чтения возвращается как есть: просрочка обязана остаться просрочкой.
func ReadAll(r io.Reader, limit int64) ([]byte, error) {
	raw, err := io.ReadAll(io.LimitReader(r, limit+1))
	if err != nil {
		return nil, err
	}

	if int64(len(raw)) > limit {
		return nil, &TooLargeError{Limit: limit}
	}

	return raw, nil
}

// Decode разбирает ровно один JSON-документ: хвост после объекта — отказ (Unmarshal, не Decoder),
// иначе первый фрагмент потока прошёл бы за ответ.
func Decode(r io.Reader, limit int64, v any) error {
	raw, err := ReadAll(r, limit)
	if err != nil {
		return err
	}

	if err = json.Unmarshal(raw, v); err != nil {
		return fmt.Errorf("конверт: %w", err)
	}

	return nil
}

// ReadStatus собирает отказ по не-2xx. envelope достаёт текст из конверта поставщика;
// пустой результат — тело обрезанным текстом как есть.
func ReadStatus(resp *http.Response, limit int64, envelope func(raw []byte) string) Status {
	st := Status{Code: resp.StatusCode, RetryAfter: RetryAfter(resp.Header)}

	raw, err := io.ReadAll(io.LimitReader(resp.Body, limit))
	if err != nil || len(raw) == 0 {
		return st
	}

	if msg := strings.TrimSpace(envelope(raw)); msg != "" {
		st.Message = msg
	} else {
		st.Message = strings.TrimSpace(strings.ToValidUTF8(string(raw), ""))
	}

	return st
}

// RetryAfter разбирает заголовок в обеих формах: секунды числом и HTTP-дата.
// Отсутствующий, непонятный и отрицательный — ноль; чрезмерный насыщается.
func RetryAfter(h http.Header) time.Duration {
	raw := strings.TrimSpace(h.Get("Retry-After"))
	if raw == "" {
		return 0
	}

	secs, err := strconv.ParseInt(raw, 10, 64)

	switch {
	case err == nil && secs <= 0:
		return 0
	case err == nil && secs > int64(math.MaxInt64/time.Second):
		return math.MaxInt64
	case err == nil:
		return time.Duration(secs) * time.Second
	case errors.Is(err, strconv.ErrRange) && !strings.HasPrefix(raw, "-"):
		return math.MaxInt64
	}

	at, dateErr := http.ParseTime(raw)
	if dateErr != nil {
		return 0
	}

	return max(time.Until(at), 0)
}
