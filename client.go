package llm

import (
	"context"
	"fmt"
	"math"
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

// Budget — верхняя оценка времени вызова: все попытки со своими таймаутами плюс
// худшие паузы; переполнение насыщается до максимальной длительности.
func (c *Client) Budget() time.Duration {
	attempts := c.attempts()
	budget := mulDuration(c.timeout(), attempts)

	for attempt := 1; attempt < attempts; attempt++ {
		budget = addDuration(budget, c.retryPause(attempt, c.maxRetryAfter()))
	}

	return budget
}

// Chat вызывает модель, повторяя отказы классов [RetryImmediate] и [RetryAfterDelay]
// в пределах числа попыток и потолка ожидания. Негодный запрос отбивается до плеча.
func (c *Client) Chat(ctx context.Context, req Request) (Result, error) {
	if c.Provider == nil {
		return Result{}, &CallError{Model: req.Model, Class: RetryNever, Err: errProviderMissing}
	}

	if err := req.Validate(); err != nil {
		return Result{}, &CallError{
			Provider: c.Provider.Name(), Model: req.Model, Class: RetryNever, Message: err.Error(), Err: err,
		}
	}

	run := &call{client: c, req: req, started: time.Now(), spent: Usage{Known: true}}

	return run.do(ctx)
}

// call — состояние одного вызова: расход по попыткам и отчёт.
type call struct {
	client  *Client
	req     Request
	started time.Time
	spent   Usage
}

func (r *call) do(ctx context.Context) (Result, error) {
	c := r.client
	attempts := c.attempts()

	for attempt := 1; ; attempt++ {
		res, err := c.once(ctx, r.req)
		r.spent = r.spent.Add(usageOfAttempt(res, err))
		r.observeAttempt(ctx, attempt, res, err)

		if err == nil {
			return r.succeed(attempt, res), nil
		}

		fail := classify(c.Provider.Name(), r.req.Model, attempt, err)

		switch {
		case fail.Class == RetryNever, fail.Class == RetryNeedsConfiguration, attempt == attempts:
			return Result{}, r.fail(fail)
		case fail.RetryAfter > c.maxRetryAfter():
			// Просьба дольше потолка — «сегодня не обслужим»: слот не держим, срок и класс отдаём наверх.
			fail.Class = RetryAfterDelay

			return Result{}, r.fail(fail)
		}

		if waitErr := sleep(ctx, c.retryPause(attempt, fail.RetryAfter)); waitErr != nil {
			fail.Err = fmt.Errorf("ожидание повтора: %w", waitErr)

			return Result{}, r.fail(fail)
		}
	}
}

func (r *call) report(outcome string, class RetryClass, attempts int, model string) CallReport {
	return CallReport{
		CallID:         r.req.CallID,
		Provider:       r.client.Provider.Name(),
		RequestedModel: r.req.Model,
		Model:          model,
		Task:           r.req.Task,
		Outcome:        outcome,
		Class:          class,
		Attempts:       attempts,
		Duration:       time.Since(r.started),
		Usage:          r.spent,
	}
}

func (r *call) observeAttempt(ctx context.Context, attempt int, res Result, err error) {
	r.client.observer().Attempt(AttemptReport{
		CallID:         r.req.CallID,
		Attempt:        attempt,
		Provider:       r.client.Provider.Name(),
		RequestedModel: r.req.Model,
		Model:          res.Model,
		Task:           r.req.Task,
		Outcome:        attemptOutcome(ctx, err),
		Phase:          phaseOf(err),
		Duration:       res.Latency,
		ServerLatency:  res.ServerLatency,
		Usage:          usageOfAttempt(res, err),
	})
}

func (r *call) succeed(attempt int, res Result) Result {
	if res.Model == "" {
		res.Model = r.req.Model
	}

	res.Report = r.report(OutcomeOK, "", attempt, res.Model)
	res.Latency = res.Report.Duration
	r.client.observer().Call(res.Report)

	return res
}

func (r *call) fail(fail *CallError) *CallError {
	fail.Report = r.report(OutcomeError, fail.Class, fail.Attempts, "")
	fail.Latency = fail.Report.Duration
	r.client.observer().Call(fail.Report)

	return fail
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

func usageOfAttempt(res Result, err error) Usage {
	if err != nil {
		return usageOf(err)
	}

	return res.Usage
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
// Потолок сверяется до сдвига: сдвинутое значение переполняется и в плюс тоже.
func (c *Client) pause(attempt int) time.Duration {
	base := c.Pause
	if base <= 0 {
		return 0
	}

	shift := attempt - 1
	if shift >= durationBits || base > c.maxPause()>>shift {
		return c.maxPause()
	}

	return base << shift
}

// retryPause — худшая из своей выдержки и просьбы плеча, не дольше потолка.
func (c *Client) retryPause(attempt int, retryAfter time.Duration) time.Duration {
	pause := c.pause(attempt)
	if wait := min(retryAfter, c.maxRetryAfter()); wait > pause {
		pause = wait
	}

	return pause
}

// RetryAfter разбирает заголовок в обеих формах: секунды числом и HTTP-дата.
// Отсутствующий, непонятный и отрицательный — ноль; чрезмерный насыщается.
func RetryAfter(h http.Header) time.Duration {
	raw := strings.TrimSpace(h.Get("Retry-After"))
	if raw == "" {
		return 0
	}

	if secs, err := strconv.ParseInt(raw, 10, 64); err == nil {
		return mulDuration(time.Second, clampInt(secs))
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

// mulDuration — d × n с насыщением; отрицательные множители дают ноль.
func mulDuration(d time.Duration, n int) time.Duration {
	if d <= 0 || n <= 0 {
		return 0
	}

	if d > math.MaxInt64/time.Duration(n) {
		return math.MaxInt64
	}

	return d * time.Duration(n)
}

// addDuration — a + b с насыщением у неотрицательных слагаемых.
func addDuration(a, b time.Duration) time.Duration {
	if b > math.MaxInt64-a {
		return math.MaxInt64
	}

	return a + b
}

// clampInt — int64 в int без переполнения; отрицательное — ноль.
func clampInt(v int64) int {
	switch {
	case v <= 0:
		return 0
	case v > math.MaxInt:
		return math.MaxInt
	default:
		return int(v)
	}
}

// saturate — сумма счётчиков без переполнения; отрицательные считаются нулём.
func saturate(a, b int) int {
	a, b = max(a, 0), max(b, 0)
	if b > math.MaxInt-a {
		return math.MaxInt
	}

	return a + b
}
