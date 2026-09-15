package llm

import (
	"context"
	"fmt"
	"net/http"
	"strconv"
	"strings"
	"time"
)

// durationBits — разрядность time.Duration: сдвиг на неё и больше даёт ноль.
const durationBits = 63

// Умолчания клиента. Attempts × Timeout + паузы — то, что вызывающий обязан себе
// отвести ([Client.Budget]), иначе очередь оборвёт последнюю попытку.
const (
	DefaultTimeout       = 90 * time.Second
	DefaultAttempts      = 3
	DefaultPause         = 2 * time.Second
	DefaultMaxPause      = time.Minute
	DefaultMaxRetryAfter = 30 * time.Second
)

// Client — вызов с повторами поверх одного плеча. Поля открыты: бюджет сверяется
// с таймаутом джобы снаружи, а не прячется в константах.
type Client struct {
	Provider Provider
	// Timeout — на одну попытку, не на весь вызов: повтор после просрочки получает полный срок.
	Timeout  time.Duration
	Attempts int
	// Pause — базовая пауза между попытками, дальше удваивается до MaxPause.
	Pause    time.Duration
	MaxPause time.Duration
	// MaxRetryAfter — потолок просьбы подождать: дольше — отказ [RetryAfterDelay]
	// без ожидания, срок в нём исходный; повторяет его очередь потребителя.
	MaxRetryAfter time.Duration
	// Observe — приёмник отчётов; пустое поле не считает ничего.
	Observe Observer
}

// New собирает клиента с умолчаниями бюджета.
func New(p Provider, timeout time.Duration) *Client {
	if timeout <= 0 {
		timeout = DefaultTimeout
	}

	return &Client{
		Provider:      p,
		Timeout:       timeout,
		Attempts:      DefaultAttempts,
		Pause:         DefaultPause,
		MaxPause:      DefaultMaxPause,
		MaxRetryAfter: DefaultMaxRetryAfter,
	}
}

// Budget — верхняя оценка времени вызова: все попытки со своими таймаутами плюс худшие паузы.
func (c *Client) Budget() time.Duration {
	attempts := c.attempts()

	budget := time.Duration(attempts) * c.timeout()
	for attempt := 1; attempt < attempts; attempt++ {
		budget += c.retryPause(attempt, c.maxRetryAfter())
	}

	return budget
}

// Chat вызывает модель, повторяя отказы классов [RetryImmediate] и [RetryAfterDelay]
// в пределах числа попыток и потолка ожидания.
func (c *Client) Chat(ctx context.Context, req Request) (Result, error) {
	if c.Provider == nil {
		return Result{}, &CallError{Model: req.Model, Class: RetryNever, Err: errProviderMissing}
	}

	provider := c.Provider.Name()
	started := time.Now()
	attempts := c.attempts()
	obs := c.observer()
	spent := Usage{Known: true}

	failed := func(fail *CallError) (Result, error) {
		fail.Latency = time.Since(started)
		obs.Call(CallReport{
			Provider: provider, Model: req.Model, Task: req.Task,
			Outcome: OutcomeError, Class: fail.Class, Attempts: fail.Attempts,
			Duration: fail.Latency, Usage: spent,
		})

		return Result{}, fail
	}

	for attempt := 1; ; attempt++ {
		res, err := c.once(ctx, req)
		spent = add(spent, res.Usage)
		obs.Attempt(AttemptReport{
			Provider: provider, Model: req.Model, Task: req.Task,
			Outcome: outcomeOf(ctx, err), Phase: phaseOf(err), Duration: res.Latency, Usage: res.Usage,
		})

		if err == nil {
			res.Latency = time.Since(started)
			res.Attempts = attempt

			if res.Model == "" {
				res.Model = req.Model
			}

			obs.Call(CallReport{
				Provider: provider, Model: req.Model, Task: req.Task,
				Outcome: OutcomeOK, Attempts: attempt, Duration: res.Latency, Usage: spent,
			})

			return res, nil
		}

		fail := classify(provider, req.Model, attempt, err)

		switch {
		case fail.Class == RetryNever, fail.Class == RetryNeedsConfiguration, attempt == attempts:
			return failed(fail)
		case fail.RetryAfter > c.maxRetryAfter():
			// Просьба дольше потолка — «сегодня не обслужим»: слот не держим, срок отдаём наверх.
			return failed(fail)
		}

		if waitErr := sleep(ctx, c.retryPause(attempt, fail.RetryAfter)); waitErr != nil {
			fail.Err = fmt.Errorf("ожидание повтора: %w", waitErr)

			return failed(fail)
		}
	}
}

// once — одна попытка под своим сроком; Latency результата — длительность попытки.
func (c *Client) once(ctx context.Context, req Request) (Result, error) {
	attemptCtx, cancel := context.WithTimeout(ctx, c.timeout())
	defer cancel()

	started := time.Now()
	res, err := c.Provider.Complete(attemptCtx, req)
	res.Latency = time.Since(started)

	return res, err
}

func (c *Client) observer() Observer {
	if c.Observe != nil {
		return c.Observe
	}

	return noopObserver{}
}

func (c *Client) timeout() time.Duration {
	if c.Timeout > 0 {
		return c.Timeout
	}

	return DefaultTimeout
}

func (c *Client) attempts() int {
	if c.Attempts > 0 {
		return c.Attempts
	}

	return DefaultAttempts
}

func (c *Client) maxPause() time.Duration {
	if c.MaxPause > 0 {
		return c.MaxPause
	}

	return DefaultMaxPause
}

func (c *Client) maxRetryAfter() time.Duration {
	if c.MaxRetryAfter > 0 {
		return c.MaxRetryAfter
	}

	return DefaultMaxRetryAfter
}

// pause — выдержка перед попыткой, следующей за attempt-й: удвоение с потолком.
// Сдвиг на разрядность и больше даёт ноль, переполнение — отрицательное число:
// обе величины меньше базы и обе означают «дошли до потолка».
func (c *Client) pause(attempt int) time.Duration {
	base := c.Pause
	if base <= 0 {
		return 0
	}

	if attempt-1 >= durationBits {
		return c.maxPause()
	}

	if paused := base << (attempt - 1); paused >= base && paused <= c.maxPause() {
		return paused
	}

	return c.maxPause()
}

// retryPause — худшая из своей выдержки и просьбы плеча, не дольше потолка.
func (c *Client) retryPause(attempt int, retryAfter time.Duration) time.Duration {
	pause := c.pause(attempt)
	if wait := min(retryAfter, c.maxRetryAfter()); wait > pause {
		pause = wait
	}

	return pause
}

func outcomeOf(ctx context.Context, err error) string {
	if err == nil {
		return OutcomeOK
	}

	return attemptOutcome(ctx, err)
}

func phaseOf(err error) Phase {
	var status *StatusError
	if errorsAs(err, &status) && status.Phase != "" {
		return status.Phase
	}

	return PhaseInference
}

func add(total, part Usage) Usage {
	if !part.Known {
		return Usage{InputTokens: total.InputTokens, OutputTokens: total.OutputTokens, Known: false}
	}

	total.InputTokens += part.InputTokens
	total.OutputTokens += part.OutputTokens

	return total
}

// RetryAfter разбирает заголовок в обеих формах: секунды числом и HTTP-дата.
// Отсутствующий и непонятный — ноль, пауза остаётся за клиентом.
func RetryAfter(h http.Header) time.Duration {
	raw := strings.TrimSpace(h.Get("Retry-After"))
	if raw == "" {
		return 0
	}

	if secs, err := strconv.Atoi(raw); err == nil {
		return max(time.Duration(secs)*time.Second, 0)
	}

	at, err := http.ParseTime(raw)
	if err != nil {
		return 0
	}

	return max(time.Until(at), 0)
}

// sleep ждёт паузу, но не дольше контекста.
func sleep(ctx context.Context, d time.Duration) error {
	if err := ctx.Err(); err != nil {
		return err
	}

	if d <= 0 {
		return nil
	}

	timer := time.NewTimer(d)
	defer timer.Stop()

	select {
	case <-ctx.Done():
		return ctx.Err()
	case <-timer.C:
		return nil
	}
}
