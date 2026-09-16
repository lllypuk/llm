package llm

import (
	"context"
	"errors"
	"fmt"
	"math"
	"net/http"
	"time"

	"github.com/lllypuk/llm/internal/httpjson"
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

// Budget — все попытки со своими таймаутами плюс худшие паузы. Уборку плеча после срока
// попытки (gigachat.CleanupBudget) не включает; переполнение насыщается.
func (c *Client) Budget() time.Duration {
	attempts := c.attempts()
	budget := mulDuration(c.timeout(), attempts)

	for attempt := 1; attempt < attempts; attempt++ {
		budget = addDuration(budget, c.retryPause(attempt, c.maxRetryAfter()))
	}

	return budget
}

// Chat вызывает модель, повторяя отказы классов [RetryImmediate] и [RetryAfterDelay]
// в пределах числа попыток и потолка ожидания. Негодный запрос и запрос сверх
// профиля плеча отбиваются до вызова; обрезанный и отфильтрованный ответ — [RetryNever].
func (c *Client) Chat(ctx context.Context, req Request) (Result, error) {
	run := &call{client: c, req: req, started: time.Now(), spent: Usage{Known: true}}

	if c.Provider == nil {
		return Result{}, run.fail(&CallError{Model: req.Model, Class: RetryNever, Err: errProviderMissing})
	}

	run.provider = c.Provider.Name()

	if err := c.admit(req); err != nil {
		run.outcome = OutcomeBadRequest

		return Result{}, run.fail(&CallError{
			Provider: run.provider, Model: req.Model, Class: RetryNever, Message: err.Error(), Err: err,
		})
	}

	return run.do(ctx)
}

// admit — запрос собираем и профиль плеча его подтверждает; неизвестный профиль не подтверждает ничего.
func (c *Client) admit(req Request) error {
	if err := req.Validate(); err != nil {
		return err
	}

	caps, known := c.Provider.Capabilities(req.Model)
	if !known {
		caps = Capabilities{}
	}

	return caps.Check(req)
}

// call — состояние одного вызова: расход и отчёты попыток, модель из последнего конверта.
// Отчёты живут в вызове, а не в клиенте: конкурентные вызовы одного клиента их не смешивают.
type call struct {
	client   *Client
	provider string
	req      Request
	started  time.Time
	spent    Usage
	model    string
	outcome  string
	attempts []AttemptReport
}

func (r *call) do(ctx context.Context) (Result, error) {
	c := r.client
	attempts := c.attempts()

	for attempt := 1; ; attempt++ {
		started := time.Now()
		res, err := c.once(ctx, r.req)
		meta := attemptMetaOf(res, err)
		r.spent = r.spent.Add(meta.usage)

		if meta.model != "" {
			r.model = meta.model
		}

		outcome := attemptOutcome(ctx, err, meta.finish)
		r.observeAttempt(attempt, started, res.Latency, outcome, err, meta)

		if fail := terminalFinish(r.provider, r.req.Model, meta.finish, err); fail != nil {
			r.outcome = outcome

			return Result{}, r.fail(fail)
		}

		if err == nil {
			return r.succeed(res), nil
		}

		fail := classify(r.provider, r.req.Model, err)

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
func (r *call) report(outcome string, class RetryClass, model string) CallReport {
	return CallReport{
		CallID:         r.req.CallID,
		Provider:       r.provider,
		RequestedModel: r.req.Model,
		Model:          model,
		Task:           r.req.Task,
		Outcome:        outcome,
		Class:          class,
		Attempts:       len(r.attempts),
		Duration:       time.Since(r.started),
		Usage:          r.spent,
	}
}

func (r *call) observeAttempt(
	attempt int, started time.Time, duration time.Duration, outcome string, err error, meta attemptMeta,
) {
	report := AttemptReport{
		CallID:         r.req.CallID,
		Attempt:        attempt,
		Provider:       r.provider,
		RequestedModel: r.req.Model,
		Model:          meta.model,
		Task:           r.req.Task,
		Outcome:        outcome,
		Phase:          phaseOf(err),
		RequestID:      meta.requestID,
		Finish:         meta.finish,
		StartedAt:      started,
		Duration:       duration,
		ServerLatency:  meta.server,
		Usage:          meta.usage,
		Cleanup:        meta.cleanup,
	}

	r.attempts = append(r.attempts, report)
	r.client.observer().Attempt(report)
}

// succeed — отчёт несёт модель из ответа как есть; подстановка запрошенной — только в Result.
func (r *call) succeed(res Result) Result {
	res.Report = r.report(OutcomeOK, "", res.Model)
	res.StartedAt = r.started
	res.Latency = res.Report.Duration
	res.Attempts = r.attempts
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

	fail.Report = r.report(outcome, fail.Class, r.model)
	fail.StartedAt = r.started
	fail.Latency = fail.Report.Duration
	fail.Attempts = r.attempts
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

// attemptMeta — что известно о попытке из конверта поставщика.
type attemptMeta struct {
	usage     Usage
	model     string
	requestID string
	finish    Finish
	server    time.Duration
	cleanup   *CleanupWarning
}

// attemptMetaOf — у отказа метаданные из конверта, если он был: негодное содержимое или не-2xx.
func attemptMetaOf(res Result, err error) attemptMeta {
	if err == nil {
		return attemptMeta{
			usage:     res.Usage,
			model:     res.Model,
			requestID: res.RequestID,
			finish:    res.Finish,
			server:    res.ServerLatency,
			cleanup:   res.Cleanup,
		}
	}

	var meta attemptMeta

	var warned *WarnedError
	if errors.As(err, &warned) {
		meta.cleanup = warned.Cleanup
	}

	var response *ResponseError

	var status *StatusError

	switch {
	case errors.As(err, &response):
		meta.usage = response.Usage
		meta.model = response.Model
		meta.requestID = response.RequestID
		meta.finish = response.Finish
		meta.server = response.ServerLatency
	case errors.As(err, &status):
		meta.requestID = status.RequestID
	}

	// Известный ноль — только доказуемый отказ до генерации: вход, загрузка кадров, конфигурация и 4xx, кроме 408.
	// 5xx и 408 могли прийти после генерации, их расход неизвестен.
	rejected := status != nil && status.Status >= http.StatusBadRequest &&
		status.Status < http.StatusInternalServerError && status.Status != http.StatusRequestTimeout
	if rejected || misconfigured(err) || phaseOf(err) != PhaseInference {
		meta.usage = Usage{Known: true}
	}

	return meta
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
func RetryAfter(h http.Header) time.Duration { return httpjson.RetryAfter(h) }

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
