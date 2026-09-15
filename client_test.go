package llm_test

import (
	"context"
	"errors"
	"net/http"
	"sync"
	"testing"
	"time"

	"github.com/lllypuk/llm"
)

// step — исход одной попытки фейкового плеча: результат либо ошибка.
type step struct {
	res llm.Result
	err error
}

// fake — плечо по сценарию: попытки отдают шаги по порядку, лишние — последний шаг.
type fake struct {
	mu    sync.Mutex
	steps []step
	calls int
	block bool // блокировать до отмены контекста
}

func (f *fake) Name() string { return "fake" }

func (f *fake) Complete(ctx context.Context, _ llm.Request) (llm.Result, error) {
	f.mu.Lock()
	f.calls++
	i := min(f.calls-1, len(f.steps)-1)
	s := f.steps[i]
	block := f.block
	f.mu.Unlock()

	if block {
		<-ctx.Done()

		return llm.Result{}, ctx.Err()
	}

	return s.res, s.err
}

func (f *fake) count() int {
	f.mu.Lock()
	defer f.mu.Unlock()

	return f.calls
}

// recorder — наблюдатель, копящий отчёты.
type recorder struct {
	mu       sync.Mutex
	attempts []llm.AttemptReport
	calls    []llm.CallReport
}

func (r *recorder) Attempt(a llm.AttemptReport) {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.attempts = append(r.attempts, a)
}

func (r *recorder) Call(c llm.CallReport) {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.calls = append(r.calls, c)
}

func ok(text string, in, out int) step {
	return step{res: llm.Result{Text: text, Usage: llm.Usage{InputTokens: in, OutputTokens: out, Known: true}}}
}

func status(code int, retryAfter time.Duration) step {
	return step{err: &llm.StatusError{Status: code, Message: http.StatusText(code), RetryAfter: retryAfter}}
}

func client(f *fake, obs llm.Observer) *llm.Client {
	c := llm.New(f, time.Second)
	c.Pause = time.Millisecond
	c.Observe = obs

	return c
}

func req() llm.Request {
	return llm.Request{
		CallID:   "c1",
		Task:     "test",
		Model:    "m",
		Messages: []llm.Message{{Role: llm.RoleUser, Text: "hi"}},
	}
}

func callError(t *testing.T, err error) *llm.CallError {
	t.Helper()

	var call *llm.CallError
	if !errors.As(err, &call) {
		t.Fatalf("ожидался CallError, получено %v", err)
	}

	return call
}

// TestChatRetriesImmediateAndSucceeds — два 5xx и удача: три попытки, один вызов,
// расход суммируется по попыткам, модель из ответа пустая — подставляется запрошенная.
func TestChatRetriesImmediateAndSucceeds(t *testing.T) {
	t.Parallel()

	f := &fake{steps: []step{status(502, 0), status(503, 0), ok("done", 10, 5)}}
	obs := &recorder{}

	res, err := client(f, obs).Chat(context.Background(), req())
	if err != nil {
		t.Fatal(err)
	}

	if res.Text != "done" || res.Report.Attempts != 3 || res.Model != "m" || res.Latency <= 0 {
		t.Errorf("результат %+v", res)
	}

	if len(obs.attempts) != 3 || len(obs.calls) != 1 {
		t.Fatalf("попыток %d, вызовов %d", len(obs.attempts), len(obs.calls))
	}

	if obs.attempts[1].Attempt != 2 || obs.attempts[1].CallID != "c1" || res.Report.CallID != "c1" {
		t.Errorf("корреляция: попытка %+v, отчёт %+v", obs.attempts[1], res.Report)
	}

	if obs.attempts[0].Outcome != llm.OutcomeHTTP5xx || obs.attempts[2].Outcome != llm.OutcomeOK {
		t.Errorf("исходы попыток %v", obs.attempts)
	}

	if c := obs.calls[0]; c.Outcome != llm.OutcomeOK || c.Attempts != 3 || c.Usage.InputTokens != 10 {
		t.Errorf("отчёт вызова %+v", c)
	}
}

// TestChatDoesNotRetryNeedsConfiguration — ключ, баланс и модель чинит оператор:
// попытка одна, но отказ восстановим.
func TestChatDoesNotRetryNeedsConfiguration(t *testing.T) {
	t.Parallel()

	for _, code := range []int{401, 402, 403, 404} {
		f := &fake{steps: []step{status(code, 0)}}

		_, err := client(f, nil).Chat(context.Background(), req())

		call := callError(t, err)
		if f.count() != 1 || call.Class != llm.RetryNeedsConfiguration || !llm.Recoverable(err) {
			t.Errorf("%d: попыток %d, класс %s, восстановим %v", code, f.count(), call.Class, llm.Recoverable(err))
		}
	}
}

// TestChatStopsOnTerminalStatus — прочие 4xx повторятся тем же ответом: одна попытка, невосстановим.
func TestChatStopsOnTerminalStatus(t *testing.T) {
	t.Parallel()

	f := &fake{steps: []step{status(400, 0)}}
	obs := &recorder{}

	_, err := client(f, obs).Chat(context.Background(), req())

	call := callError(t, err)
	if f.count() != 1 || call.Class != llm.RetryNever || llm.Recoverable(err) || call.Status != 400 {
		t.Errorf("попыток %d, ошибка %+v", f.count(), call)
	}

	if obs.calls[0].Outcome != llm.OutcomeError || obs.calls[0].Class != llm.RetryNever {
		t.Errorf("отчёт вызова %+v", obs.calls[0])
	}
}

// TestChatWaitsRetryAfter — 429 с короткой просьбой: ждём её и повторяем.
func TestChatWaitsRetryAfter(t *testing.T) {
	t.Parallel()

	f := &fake{steps: []step{status(429, 40*time.Millisecond), ok("late", 0, 0)}}

	started := time.Now()

	res, err := client(f, nil).Chat(context.Background(), req())
	if err != nil {
		t.Fatal(err)
	}

	if res.Report.Attempts != 2 || time.Since(started) < 40*time.Millisecond {
		t.Errorf("попыток %d за %s", res.Report.Attempts, time.Since(started))
	}
}

// TestChatReturnsLongRetryAfterWithoutWaiting — просьба дольше потолка не ждётся:
// отказ после первой попытки, класс after_delay, срок исходный.
func TestChatReturnsLongRetryAfterWithoutWaiting(t *testing.T) {
	t.Parallel()

	f := &fake{steps: []step{status(429, time.Hour), ok("never", 0, 0)}}
	c := client(f, nil)
	c.MaxRetryAfter = 10 * time.Millisecond

	started := time.Now()
	_, err := c.Chat(context.Background(), req())

	call := callError(t, err)
	if f.count() != 1 || call.Class != llm.RetryAfterDelay || call.RetryAfter != time.Hour {
		t.Errorf("попыток %d, ошибка %+v", f.count(), call)
	}

	if time.Since(started) > 5*time.Millisecond*10 || !llm.Recoverable(err) {
		t.Errorf("ждали %s, восстановим %v", time.Since(started), llm.Recoverable(err))
	}
}

// TestChatTimesOutAndRetries — просрочка попытки восстановима, срок стоит на попытке.
func TestChatTimesOutAndRetries(t *testing.T) {
	t.Parallel()

	f := &fake{steps: []step{ok("", 0, 0)}, block: true}
	obs := &recorder{}
	c := client(f, obs)
	c.Timeout = 10 * time.Millisecond
	c.Attempts = 2

	_, err := c.Chat(context.Background(), req())

	call := callError(t, err)
	if f.count() != 2 || !errors.Is(err, context.DeadlineExceeded) || call.Class != llm.RetryImmediate {
		t.Errorf("попыток %d, ошибка %+v", f.count(), call)
	}

	if obs.attempts[0].Outcome != llm.OutcomeTimeout {
		t.Errorf("исход попытки %s", obs.attempts[0].Outcome)
	}
}

// TestChatStopsOnCanceledContext — срок вызывающего кончился: исход cancelled, паузу не досиживаем,
// а отказ восстановим — он про срок джобы, а не про причину, и повторяет его очередь.
func TestChatStopsOnCanceledContext(t *testing.T) {
	t.Parallel()

	f := &fake{steps: []step{ok("", 0, 0)}, block: true}
	obs := &recorder{}
	c := client(f, obs)

	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Millisecond)
	defer cancel()

	_, err := c.Chat(ctx, req())

	call := callError(t, err)
	if f.count() != 1 || !llm.Recoverable(err) || obs.attempts[0].Outcome != llm.OutcomeCancelled {
		t.Errorf("попыток %d, ошибка %+v, исход %s", f.count(), call, obs.attempts[0].Outcome)
	}
}

// TestChatFailureCarriesLatency — отказ несёт потраченное время сам.
func TestChatFailureCarriesLatency(t *testing.T) {
	t.Parallel()

	f := &fake{steps: []step{status(500, 0)}}
	c := client(f, nil)
	c.Attempts = 2
	c.Pause = 20 * time.Millisecond

	_, err := c.Chat(context.Background(), req())
	if llm.Latency(err) < 20*time.Millisecond {
		t.Errorf("латентность отказа %s", llm.Latency(err))
	}

	if llm.Latency(errors.New("чужая")) != 0 {
		t.Error("чужая ошибка несёт время")
	}
}

// TestChatWithoutProvider — клиент без плеча отказывает сразу и невосстановимо.
func TestChatWithoutProvider(t *testing.T) {
	t.Parallel()

	_, err := (&llm.Client{}).Chat(context.Background(), req())
	if callError(t, err).Class != llm.RetryNever {
		t.Error("ожидался невосстановимый отказ")
	}
}

// TestBudgetCoversEveryAttempt — бюджет: попытки по таймауту плюс худшие паузы с потолками.
func TestBudgetCoversEveryAttempt(t *testing.T) {
	t.Parallel()

	c := llm.New(&fake{}, 10*time.Second)
	c.Attempts = 3
	c.Pause = 2 * time.Second
	c.MaxRetryAfter = 30 * time.Second

	if got, want := c.Budget(), 3*10*time.Second+2*30*time.Second; got != want {
		t.Errorf("бюджет %s, ожидался %s", got, want)
	}

	c.Attempts = 100
	if got := c.Budget(); got <= 0 || got > 100*10*time.Second+99*time.Minute {
		t.Errorf("бюджет на ста попытках %s", got)
	}
}

// TestRetryAfter — секунды, HTTP-дата, мусор и отрицательное.
func TestRetryAfter(t *testing.T) {
	t.Parallel()

	h := http.Header{}
	h.Set("Retry-After", "7")

	if got := llm.RetryAfter(h); got != 7*time.Second {
		t.Errorf("секунды: %s", got)
	}

	h.Set("Retry-After", time.Now().Add(time.Minute).UTC().Format(http.TimeFormat))
	if got := llm.RetryAfter(h); got < 50*time.Second || got > time.Minute {
		t.Errorf("дата: %s", got)
	}

	for _, raw := range []string{"", "мусор", "-5", time.Now().Add(-time.Minute).UTC().Format(http.TimeFormat)} {
		h.Set("Retry-After", raw)
		if got := llm.RetryAfter(h); got != 0 {
			t.Errorf("%q: %s", raw, got)
		}
	}
}

// TestFingerprint — детерминирован, порядок частей значим.
func TestFingerprint(t *testing.T) {
	t.Parallel()

	a, b := llm.Fingerprint("ollama", "m", "p1"), llm.Fingerprint("ollama", "m", "p1")
	if a != b || len(a) != 32 {
		t.Errorf("отпечатки %q и %q", a, b)
	}

	if llm.Fingerprint("a", "b") == llm.Fingerprint("b", "a") || llm.Fingerprint("ab") == llm.Fingerprint("a", "b") {
		t.Error("отпечаток не различает порядок и границы частей")
	}

	if llm.Fingerprint("a\x00b", "c") == llm.Fingerprint("a", "b\x00c") {
		t.Error("разделитель внутри части смешивает наборы")
	}
}

// TestChatFailureCarriesReport — окончательный отказ несёт отчёт с расходом всех попыток,
// а удача — расход попыток в Report и расход удавшейся в Usage.
func TestChatFailureCarriesReport(t *testing.T) {
	t.Parallel()

	spent := step{err: &llm.ResponseError{Message: "пусто", Usage: llm.Usage{InputTokens: 7, Known: true}}}
	f := &fake{steps: []step{spent, status(500, 0)}}
	c := client(f, nil)
	c.Attempts = 2

	_, err := c.Chat(context.Background(), req())

	call := callError(t, err)
	if call.Report.Usage.InputTokens != 7 || call.Report.Attempts != 2 || call.Report.CallID != "c1" {
		t.Errorf("отчёт отказа %+v", call.Report)
	}

	f = &fake{steps: []step{spent, ok("late", 11, 1)}}

	res, err := client(f, nil).Chat(context.Background(), req())
	if err != nil {
		t.Fatal(err)
	}

	if res.Usage.InputTokens != 11 || res.Report.Usage.InputTokens != 18 || !res.Report.Usage.Known {
		t.Errorf("расход %+v, отчёт %+v", res.Usage, res.Report.Usage)
	}
}

// TestChatRejectsBadRequestBeforeProvider — схема без схемы и пустая модель: плечо не зовётся,
// класс never, исход вызова error.
func TestChatRejectsBadRequestBeforeProvider(t *testing.T) {
	t.Parallel()

	f := &fake{steps: []step{ok("never", 0, 0)}}

	for _, r := range []llm.Request{
		{Model: "m", Output: llm.Output{Mode: llm.ModeSchema}},
		{Model: ""},
		{Model: "m", Output: llm.Output{Mode: "xml"}},
	} {
		obs := &recorder{}

		_, err := client(f, obs).Chat(context.Background(), r)

		call := callError(t, err)
		if call.Class != llm.RetryNever || f.count() != 0 {
			t.Errorf("запрос %+v: %v, вызовов плеча %d", r, err, f.count())
		}

		if rep := call.Report; rep.Outcome != llm.OutcomeBadRequest || rep.Attempts != 0 || !rep.Usage.Known ||
			rep.RequestedModel != r.Model || len(obs.calls) != 1 || len(obs.attempts) != 0 {
			t.Errorf("отчёт раннего отказа %+v, вызовов %d, попыток %d", rep, len(obs.calls), len(obs.attempts))
		}
	}
}

// TestChatLongRetryAfterOnLastAttempt — с единственной попыткой 503 и часовая просьба:
// after_delay, а не immediate; терминальный 4xx с тем же заголовком остаётся never.
func TestChatLongRetryAfterOnLastAttempt(t *testing.T) {
	t.Parallel()

	for _, tc := range []struct {
		code int
		want llm.RetryClass
	}{{503, llm.RetryAfterDelay}, {400, llm.RetryNever}, {404, llm.RetryNeedsConfiguration}} {
		f := &fake{steps: []step{status(tc.code, time.Hour)}}
		c := client(f, nil)
		c.Attempts = 1
		c.MaxRetryAfter = time.Millisecond

		_, err := c.Chat(context.Background(), req())
		if got := callError(t, err).Class; got != tc.want {
			t.Errorf("%d: класс %s, ожидался %s", tc.code, got, tc.want)
		}
	}
}

// TestReportKeepsReportedModel — в отчёте модель из ответа как есть, подстановка только в Result;
// у отказа — модель последнего конверта.
func TestReportKeepsReportedModel(t *testing.T) {
	t.Parallel()

	f := &fake{steps: []step{ok("x", 1, 1)}}

	res, err := client(f, nil).Chat(context.Background(), req())
	if err != nil || res.Model != "m" || res.Report.Model != "" {
		t.Errorf("результат %+v", res)
	}

	rejected := step{err: &llm.ResponseError{Message: "пусто", Model: "m:latest", ServerLatency: time.Second}}
	f = &fake{steps: []step{rejected, status(500, 0)}}
	obs := &recorder{}
	c := client(f, obs)
	c.Attempts = 2

	_, err = c.Chat(context.Background(), req())
	if callError(t, err).Report.Model != "m:latest" || obs.attempts[0].ServerLatency != time.Second {
		t.Errorf("метаданные отказа потеряны: %v, %+v", err, obs.attempts[0])
	}

	f = &fake{steps: []step{rejected, ok("x", 1, 1)}}

	res, err = client(f, nil).Chat(context.Background(), req())
	if err != nil || res.Report.Model != "" || res.Model != "m" {
		t.Errorf("успех без модели унаследовал имя отказа: %+v, %v", res.Report, err)
	}
}

// TestChatLongRetryAfterOnAnyStatus — 503 с часовой просьбой: класс after_delay, а не immediate.
func TestChatLongRetryAfterOnAnyStatus(t *testing.T) {
	t.Parallel()

	f := &fake{steps: []step{status(503, time.Hour)}}
	c := client(f, nil)
	c.MaxRetryAfter = time.Millisecond

	_, err := c.Chat(context.Background(), req())
	if call := callError(t, err); call.Class != llm.RetryAfterDelay || f.count() != 1 {
		t.Errorf("ошибка %+v, попыток %d", call, f.count())
	}
}

// TestUsageAdd — неполная часть делает сумму неполной, но счётчики не теряются.
func TestUsageAdd(t *testing.T) {
	t.Parallel()

	sum := (llm.Usage{Known: true}).Add(llm.Usage{InputTokens: 3, Known: true}).Add(llm.Usage{InputTokens: 4})
	if sum.InputTokens != 7 || sum.Known {
		t.Errorf("сумма %+v", sum)
	}

	big := llm.Usage{InputTokens: 1<<62 + 1<<61, Known: true}
	if over := big.Add(big); over.Known || over.InputTokens <= 0 {
		t.Errorf("переполнение %+v", over)
	}

	neg := (llm.Usage{Known: true}).Add(llm.Usage{InputTokens: -1, Known: true})
	if neg.Known || neg.InputTokens != 0 {
		t.Errorf("отрицательное %+v", neg)
	}
}

// TestArithmeticSaturates — чрезмерный Retry-After, бюджет и сдвиг паузы не переполняются.
func TestArithmeticSaturates(t *testing.T) {
	t.Parallel()

	h := http.Header{}

	for _, raw := range []string{"9223372037", "9223372036854775808", "99999999999999999999999"} {
		h.Set("Retry-After", raw)

		if got := llm.RetryAfter(h); got < time.Hour {
			t.Errorf("Retry-After %s переполнился: %s", raw, got)
		}
	}

	h.Set("Retry-After", "-9223372036854775808")
	if got := llm.RetryAfter(h); got != 0 {
		t.Errorf("отрицательное переполнение: %s", got)
	}

	c := llm.New(&fake{}, 1<<62)
	c.Attempts = 3

	if got := c.Budget(); got <= 0 {
		t.Errorf("бюджет переполнился: %s", got)
	}

	c = llm.New(&fake{}, time.Second)
	c.Attempts = 40
	c.Pause = (1 << 32) + 1
	c.MaxPause = time.Minute
	c.MaxRetryAfter = time.Millisecond

	if got := c.Budget(); got > 40*time.Second+39*time.Minute || got < 40*time.Second+30*time.Minute {
		t.Errorf("бюджет с большой паузой %s", got)
	}
}
