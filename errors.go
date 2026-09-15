package llm

import (
	"errors"
	"fmt"
	"net/http"
	"time"
)

// RetryClass — что делать с отказом. Немедленный повтор и повтор после паузы делает
// клиент; NeedsConfiguration и Never клиент не повторяет, но первый остаётся
// восстановимым — его чинит оператор, а не запрос.
type RetryClass string

// Классы отказа.
const (
	RetryImmediate          RetryClass = "immediate"
	RetryAfterDelay         RetryClass = "after_delay"
	RetryNeedsConfiguration RetryClass = "needs_configuration"
	RetryNever              RetryClass = "never"
)

// Phase — где отказал вызов: у облачных плеч перед генерацией стоят вход и загрузка кадров.
type Phase string

// Фазы вызова.
const (
	PhaseAuth      Phase = "auth"
	PhaseUpload    Phase = "upload"
	PhaseInference Phase = "inference"
)

// StatusError — не-2xx от поставщика вместе с просьбой подождать.
type StatusError struct {
	Status     int
	Message    string
	RetryAfter time.Duration
	Phase      Phase
}

func (e *StatusError) Error() string {
	if e.Message == "" {
		return fmt.Sprintf("HTTP %d", e.Status)
	}

	return fmt.Sprintf("HTTP %d: %s", e.Status, e.Message)
}

// ResponseError — поставщик ответил, но ответа модели в теле нет: отказ маршрута
// под кодом 200, битый конверт, пустое содержимое.
type ResponseError struct {
	Message string
}

func (e *ResponseError) Error() string { return e.Message }

// CallError — отказ вызова целиком, с решением о повторе и потраченным временем.
type CallError struct {
	Provider   string
	Model      string
	Phase      Phase
	Status     int // код HTTP; 0 — ответа не было (сеть, просрочка, отмена)
	Message    string
	Attempts   int
	RetryAfter time.Duration
	Latency    time.Duration
	Class      RetryClass
	Err        error
}

func (e *CallError) Error() string {
	head := fmt.Sprintf("%s/%s", e.Provider, e.Model)

	switch {
	case e.Status != 0 && e.Message != "":
		head += fmt.Sprintf(": HTTP %d: %s", e.Status, e.Message)
	case e.Status != 0:
		head += fmt.Sprintf(": HTTP %d", e.Status)
	case e.Err != nil:
		head += ": " + e.Err.Error()
	}

	if e.Attempts > 1 {
		head += fmt.Sprintf(" (попыток %d)", e.Attempts)
	}

	return head
}

func (e *CallError) Unwrap() error { return e.Err }

// Recoverable отвечает, имеет ли смысл повторить вызов позже: всё, кроме [RetryNever].
func Recoverable(err error) bool {
	var call *CallError

	return errors.As(err, &call) && call.Class != RetryNever
}

// Latency — сколько длился отказавший вызов; не наш отказ даёт ноль.
func Latency(err error) time.Duration {
	var call *CallError
	if !errors.As(err, &call) {
		return 0
	}

	return call.Latency
}

// classify переводит отказ попытки в класс. 429 — после паузы; ключ, баланс и
// отсутствующая модель — к оператору; прочие 4xx повторятся тем же ответом.
func classify(provider, model string, attempt int, err error) *CallError {
	fail := &CallError{
		Provider: provider,
		Model:    model,
		Phase:    PhaseInference,
		Attempts: attempt,
		Class:    RetryImmediate,
		Err:      err,
	}

	var status *StatusError
	if !errors.As(err, &status) {
		return fail
	}

	fail.Status = status.Status
	fail.Message = status.Message
	fail.RetryAfter = status.RetryAfter

	if status.Phase != "" {
		fail.Phase = status.Phase
	}

	switch {
	case status.Status == http.StatusTooManyRequests:
		fail.Class = RetryAfterDelay
	case status.Status == http.StatusUnauthorized,
		status.Status == http.StatusPaymentRequired,
		status.Status == http.StatusForbidden,
		status.Status == http.StatusNotFound:
		fail.Class = RetryNeedsConfiguration
	case status.Status == http.StatusRequestTimeout:
		fail.Class = RetryImmediate
	case status.Status >= http.StatusBadRequest && status.Status < http.StatusInternalServerError:
		fail.Class = RetryNever
	}

	return fail
}

// errProviderMissing — клиент собран без плеча.
var errProviderMissing = errors.New("плечо не задано")

func errorsAs(err error, target any) bool { return errors.As(err, target) }
