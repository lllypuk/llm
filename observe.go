package llm

import (
	"context"
	"errors"
	"net/http"
	"time"
)

// Исходы попытки — про транспорт и годность конверта, а не про контракт модели.
const (
	OutcomeOK          = "ok"
	OutcomeCancelled   = "cancelled"
	OutcomeTimeout     = "timeout"
	OutcomeHTTP4xx     = "http_4xx"
	OutcomeHTTP5xx     = "http_5xx"
	OutcomeBadResponse = "bad_response"
	OutcomeNetwork     = "network"
	OutcomeError       = "error"
)

// AttemptReport — одна попытка: исход, фаза, длительность и расход, если он известен.
type AttemptReport struct {
	Provider string
	Model    string
	Task     string
	Outcome  string
	Phase    Phase
	Duration time.Duration
	Usage    Usage
}

// CallReport — вызов целиком: исход, число попыток, суммарный расход попыток.
type CallReport struct {
	Provider string
	Model    string
	Task     string
	Outcome  string
	Class    RetryClass
	Attempts int
	Duration time.Duration
	Usage    Usage
}

// Observer — приёмник отчётов; реализует его потребитель (Prometheus, журнал).
type Observer interface {
	Attempt(AttemptReport)
	Call(CallReport)
}

type noopObserver struct{}

func (noopObserver) Attempt(AttemptReport) {}

func (noopObserver) Call(CallReport) {}

// attemptOutcome называет исход попытки: срок и отмена снаружи — cancelled, свой срок — timeout.
func attemptOutcome(ctx context.Context, err error) string {
	var status *StatusError

	var response *ResponseError

	switch {
	case ctx.Err() != nil:
		return OutcomeCancelled
	case errors.Is(err, context.DeadlineExceeded):
		return OutcomeTimeout
	case errors.As(err, &status):
		return statusOutcome(status.Status)
	case errors.As(err, &response):
		return OutcomeBadResponse
	default:
		return OutcomeNetwork
	}
}

// statusOutcome — исход по коду; не-2xx ниже 400 — bad_response, а не отказ нашей стороны.
func statusOutcome(status int) string {
	switch {
	case status >= http.StatusInternalServerError:
		return OutcomeHTTP5xx
	case status >= http.StatusBadRequest:
		return OutcomeHTTP4xx
	default:
		return OutcomeBadResponse
	}
}
