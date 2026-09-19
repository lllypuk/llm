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
	OutcomeTruncated   = "truncated"
	OutcomeFiltered    = "filtered"
	OutcomeRefused     = "refused"
	OutcomeCancelled   = "cancelled"
	OutcomeTimeout     = "timeout"
	OutcomeHTTP4xx     = "http_4xx"
	OutcomeHTTP5xx     = "http_5xx"
	OutcomeBadResponse = "bad_response"
	OutcomeBadRequest  = "bad_request"
	OutcomeNetwork     = "network"
	OutcomeConfig      = "config"
	OutcomeError       = "error"
)

// AttemptReport — одна попытка. Model и RequestID — из ответа, когда поставщик их назвал.
type AttemptReport struct {
	CallID         string
	Attempt        int
	Provider       string
	RequestedModel string
	Model          string
	Task           string
	Outcome        string
	Phase          Phase
	RequestID      string
	Finish         Finish
	StartedAt      time.Time
	Duration       time.Duration
	ServerLatency  time.Duration
	Usage          Usage
	// AudioMillis — запись, отправленная плечу речи; у чата ноль.
	AudioMillis int64
	Cleanup     *CleanupWarning
}

// CallReport — вызов целиком: исход, класс отказа, число попыток, расход всех попыток,
// у речи — и запись всех попыток.
type CallReport struct {
	CallID         string
	Provider       string
	RequestedModel string
	Model          string
	Task           string
	Outcome        string
	Class          RetryClass
	Attempts       int
	Duration       time.Duration
	Usage          Usage
	AudioMillis    int64
}

// Observer — приёмник отчётов: быстрый и потокобезопасный, зовётся синхронно
// внутри вызова без контекста и без права на ошибку — счётчики, не журнал.
// Запись в базу делает обёртка потребителя по [Result.Report] и [CallError.Report].
type Observer interface {
	Attempt(AttemptReport)
	Call(CallReport)
}

type noopObserver struct{}

func (noopObserver) Attempt(AttemptReport) {}

func (noopObserver) Call(CallReport) {}

// attemptOutcome называет исход попытки: срок и отмена снаружи — cancelled, свой срок — timeout;
// обрезанный, отфильтрованный и отклонённый ответ — свои исходы, даже если содержимое негодно.
func attemptOutcome(ctx context.Context, err error, finish Finish) string {
	var status *StatusError

	var response *ResponseError

	var request *RequestError

	switch {
	case finish.Kind == FinishLength:
		return OutcomeTruncated
	case finish.Kind == FinishContentFilter:
		return OutcomeFiltered
	case finish.Kind == FinishRefusal:
		return OutcomeRefused
	case err == nil:
		return OutcomeOK
	case ctx.Err() != nil:
		return OutcomeCancelled
	case errors.Is(err, context.DeadlineExceeded):
		return OutcomeTimeout
	case misconfigured(err):
		return OutcomeConfig
	case errors.As(err, &request):
		return OutcomeBadRequest
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
