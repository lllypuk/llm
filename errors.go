package llm

import (
	"crypto/tls"
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
	RequestID  string
}

func (e *StatusError) Error() string {
	if e.Message == "" {
		return fmt.Sprintf("HTTP %d", e.Status)
	}

	return fmt.Sprintf("HTTP %d: %s", e.Status, e.Message)
}

// ResponseError — поставщик ответил, но ответа модели в теле нет: отказ маршрута
// под кодом 200, битый или незаконченный конверт, пустое содержимое. Причина
// сохраняется: просрочка при чтении тела обязана остаться просрочкой.
type ResponseError struct {
	Message       string
	Usage         Usage
	Model         string
	Finish        Finish
	RequestID     string
	ServerLatency time.Duration
	Err           error
}

func (e *ResponseError) Error() string {
	if e.Err != nil {
		return e.Message + ": " + e.Err.Error()
	}

	return e.Message
}

func (e *ResponseError) Unwrap() error { return e.Err }

// RequestError — запрос не собрать: поставщика не звали, повтор бессмыслен.
type RequestError struct {
	Message string
	Err     error
}

func (e *RequestError) Error() string {
	if e.Err != nil {
		return e.Message + ": " + e.Err.Error()
	}

	return e.Message
}

func (e *RequestError) Unwrap() error { return e.Err }

// ConfigError — отказ до отправки запроса, который чинит оператор, а не повтор: пустой ключ,
// отозванная подпись. Временный сбой источника подписи (сеть до IAM) ConfigError не является.
type ConfigError struct {
	Message string
	Err     error
}

func (e *ConfigError) Error() string {
	if e.Err != nil {
		return e.Message + ": " + e.Err.Error()
	}

	return e.Message
}

func (e *ConfigError) Unwrap() error { return e.Err }

// misconfigured — отказ конфигурации: [ConfigError] или непроверенный сертификат поставщика (чужой CA).
func misconfigured(err error) bool {
	var config *ConfigError

	var verify *tls.CertificateVerificationError

	return errors.As(err, &config) || errors.As(err, &verify)
}

// PhaseError помечает фазой любую причину — сеть при OAuth, отказ загрузки кадра.
type PhaseError struct {
	Phase Phase
	Err   error
}

func (e *PhaseError) Error() string { return string(e.Phase) + ": " + e.Err.Error() }

func (e *PhaseError) Unwrap() error { return e.Err }

// CleanupWarning — созданное попыткой у поставщика могло остаться неубранным: генерацию это не
// отменяет, и повтор ради уборки не оплачивается. Files — что осталось, Err — почему; Uncertain —
// загрузки без ответа: файл мог создаться, а id его неизвестен.
type CleanupWarning struct {
	Files     []string
	Uncertain int
	Err       error
}

// WarnedError — отказ попытки вместе с предупреждением уборки; цепочка ведёт к отказу, а не к уборке.
type WarnedError struct {
	Err     error
	Cleanup *CleanupWarning
}

func (e *WarnedError) Error() string { return e.Err.Error() }

func (e *WarnedError) Unwrap() error { return e.Err }

// ErrTruncated, ErrFiltered и ErrRefused — генерация кончилась пределом длины, фильтром или
// отказом модели: ответ оплачен, и повтор оплатил бы тот же исход ещё раз.
var (
	ErrTruncated = errors.New("ответ обрезан пределом длины")
	ErrFiltered  = errors.New("ответ остановлен фильтром")
	ErrRefused   = errors.New("модель отказалась отвечать")
)

// CallError — отказ вызова целиком, с решением о повторе, потраченным временем
// и тем же отчётом, что получает наблюдатель.
type CallError struct {
	Provider   string
	Model      string
	Phase      Phase
	Status     int // код HTTP; 0 — ответа не было (сеть, просрочка, отмена)
	Message    string
	RetryAfter time.Duration
	Finish     Finish
	StartedAt  time.Time
	Latency    time.Duration
	Class      RetryClass
	Report     CallReport
	Attempts   []AttemptReport
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

	if len(e.Attempts) > 1 {
		head += fmt.Sprintf(" (попыток %d)", len(e.Attempts))
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
// отсутствующая модель, [ConfigError] и чужой CA — к оператору; прочие 4xx и негодный запрос повторятся тем же.
func classify(provider, model string, err error) *CallError {
	fail := &CallError{
		Provider: provider,
		Model:    model,
		Phase:    phaseOf(err),
		Class:    RetryImmediate,
		Err:      err,
	}

	if misconfigured(err) {
		fail.Class = RetryNeedsConfiguration

		return fail
	}

	var request *RequestError
	if errors.As(err, &request) {
		fail.Class = RetryNever
		fail.Message = request.Message

		return fail
	}

	var status *StatusError
	if !errors.As(err, &status) {
		return fail
	}

	fail.Status = status.Status
	fail.Message = status.Message
	fail.RetryAfter = status.RetryAfter

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

// phaseOf — фаза отказа: явная пометка [PhaseError] или [StatusError.Phase], иначе генерация.
func phaseOf(err error) Phase {
	var phased *PhaseError
	if errors.As(err, &phased) && phased.Phase != "" {
		return phased.Phase
	}

	var status *StatusError
	if errors.As(err, &status) && status.Phase != "" {
		return status.Phase
	}

	return PhaseInference
}

// terminalFinish — отказ по причине конца генерации: обрезанный, отфильтрованный и отклонённый
// моделью ответ не повторяется. Причина из конверта попытки дописывается к сообщению.
func terminalFinish(provider, model string, finish Finish, attemptErr error) *CallError {
	var err error

	switch finish.Kind {
	case FinishLength:
		err = ErrTruncated
	case FinishContentFilter:
		err = ErrFiltered
	case FinishRefusal:
		err = ErrRefused
	case "", FinishStop, FinishToolCall:
		return nil
	default:
		return nil
	}

	message := err.Error()

	var response *ResponseError
	if errors.As(attemptErr, &response) && response.Message != "" {
		message += ": " + response.Message
	}

	return &CallError{
		Provider: provider,
		Model:    model,
		Phase:    PhaseInference,
		Message:  message,
		Finish:   finish,
		Class:    RetryNever,
		Err:      err,
	}
}

// errProviderMissing — клиент собран без плеча.
var errProviderMissing = errors.New("плечо не задано")
