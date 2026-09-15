package llm

import (
	"context"
	"errors"
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
	run := &call{client: c, req: req, started: time.Now(), spent: Usage{Known: true}}

	if c.Provider == nil {
		return Result{}, run.fail(&CallError{Model: req.Model, Class: RetryNever, Err: errProviderMissing})
	}

	run.provider = c.Provider.Name()

	if err := req.Validate(); err != nil {
		run.outcome = OutcomeBadRequest

		return Result{}, run.fail(&CallError{
			Provider: run.provider, Model: req.Model, Class: RetryNever, Message: err.Error(), Err: err,
		})
	}

	return run.do(ctx)
}

// call — состояние одного вызова: расход по попыткам, модель из последнего конверта, отчёт.
type call struct {
	client   *Client
	provider string
	req      Request
	started  time.Time
	spent    Usage
	model    string
	outcome  string
}

func (r *call) do(ctx context.Context) (Result, error) {
	c := r.client
	attempts := c.attempts()

	for attempt := 1; ; attempt++ {
		res, err := c.once(ctx, r.req)
		usage, model, server := attemptMeta(res, err)
		r.spent = r.spent.Add(usage)

		if model != "" {
			r.model = model
		}

		r.observeAttempt(ctx, attempt, res, err, usage, model, server)

		if err == nil {
			return r.succeed(attempt, res), nil
		}

		fail := classify(r.provider, r.req.Model, attempt, err)

		// Просьба дольше потолка — «сегодня не обслужим»: класс меняется до проверки числа попыток,
		// иначе на последней очередь получила бы immediate и повторила раньше разрешённого.
		tooLong := fail.RetryAfter > c.maxRetryAfter() && fail.Class != RetryNever &&
			fail.Class != RetryNeedsConfiguration
		if tooLong {
			fail.Class = RetryAfterDelay
		}

		if fail.Class == RetryNever || fail.Class == RetryNeedsConfiguration || tooLong || attempt == attempts {
			return Result{}, r.fail(fail)
		}

		if waitErr := sleep(ctx, c.retryPause(attempt, fail.RetryAfter)); waitErr != nil {
			fail.Err = fmt.Errorf("ожидание повтора: %w", waitErr)

			return Result{}, r.fail(fail)
		}
	}
}

// report — отчёт вызова; model — из удавшегося ответа как есть, у отказа — из последнего конверта.
func (r *call) report(outcome string, class RetryClass, attempts int, model string) CallReport {
	return CallReport{
		CallID:         r.req.CallID,
		Provider:       r.provider,
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

func (r *call) observeAttempt(
	ctx context.Context, attempt int, res Result, err error, usage Usage, model string, server time.Duration,
) {
	r.client.observer().Attempt(AttemptReport{
		CallID:         r.req.CallID,
		Attempt:        attempt,
		Provider:       r.provider,
		RequestedModel: r.req.Model,
		Model:          model,
		Task:           r.req.Task,
		Outcome:        attemptOutcome(ctx, err),
		Phase:          phaseOf(err),
		Duration:       res.Latency,
		ServerLatency:  server,
		Usage:          usage,
	})
}

// succeed — отчёт несёт модель из ответа как есть; подстановка запрошенной — только в Result.
func (r *call) succeed(attempt int, res Result) Result {
	res.Report = r.report(OutcomeOK, "", attempt, res.Model)
	res.Latency = res.Report.Duration
	r.client.observer().Call(res.Report)

	if res.Model == "" {
		res.Model = r.req.Model
	}

	return res
}

func (r *call) fail(fail *CallError) *CallError {
	outcome := OutcomeError
	if r.outcome != "" {
		outcome = r.outcome
	}

	fail.Report = r.report(outcome, fail.Class, fail.Attempts, r.model)
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

// attemptMeta — расход, модель и серверное время попытки: у отказа — из конверта, если он был.
func attemptMeta(res Result, err error) (Usage, string, time.Duration) {
	if err != nil {
		return metaOf(err)
	}

	return res.Usage, res.Model, res.ServerLatency
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

	secs, err := strconv.ParseInt(raw, 10, 64)

	switch {
	case err == nil:
		return mulDuration(time.Second, clampInt(secs))
	case errors.Is(err, strconv.ErrRange) && !strings.HasPrefix(raw, "-"):
		return math.MaxInt64
	}

	at, dateErr := http.ParseTime(raw)
	if dateErr != nil {
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

// saturate — сумма счётчиков; переполнение и отрицательные слагаемые делают её неточной.
func saturate(a, b int) (int, bool) {
	if a < 0 || b < 0 {
		return max(a, 0) + max(b, 0), false
	}

	if b > math.MaxInt-a {
		return math.MaxInt, false
	}

	return a + b, true
}
