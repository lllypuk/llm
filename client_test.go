package llm_test

import (
	"context"
	"encoding/json"
	"errors"
	"math"
	"net/http"
	"strconv"
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
	caps  *llm.Capabilities
}

func (f *fake) Name() string { return "fake" }

// Capabilities — заданный профиль; без него подтверждено всё, кроме пределов кадров.
func (f *fake) Capabilities(string) (llm.Capabilities, bool) {
	if f.caps != nil {
		return *f.caps, true
	}

	return llm.Capabilities{
		Vision: true, JSON: true, Schema: true, Strict: true, Temperature: true, Reasoning: true, MaxOutputTokens: true,
	}, true
}

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
	return step{res: llm.Result{Text: text, Usage: llm.Usage{BillableInput: in, Output: out, Known: true}}}
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

	if c := obs.calls[0]; c.Outcome != llm.OutcomeOK || c.Attempts != 3 || c.Usage.BillableInput != 10 {
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

// TestRetryAfter — экспортированный разбор для внешних адаптеров; формы проверяет httpjson.
func TestRetryAfter(t *testing.T) {
	t.Parallel()

	h := http.Header{}
	h.Set("Retry-After", "7")

	if got := llm.RetryAfter(h); got != 7*time.Second {
		t.Errorf("секунды: %s", got)
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

	spent := step{err: &llm.ResponseError{Message: "пусто", Usage: llm.Usage{BillableInput: 7, Known: true}}}
	f := &fake{steps: []step{spent, status(500, 0)}}
	c := client(f, nil)
	c.Attempts = 2

	_, err := c.Chat(context.Background(), req())

	call := callError(t, err)
	if call.Report.Usage.BillableInput != 7 || call.Report.Attempts != 2 || len(call.Attempts) != 2 ||
		call.Report.CallID != "c1" {
		t.Errorf("отчёт отказа %+v", call.Report)
	}

	f = &fake{steps: []step{spent, ok("late", 11, 1)}}

	res, err := client(f, nil).Chat(context.Background(), req())
	if err != nil {
		t.Fatal(err)
	}

	if res.Usage.BillableInput != 11 || res.Report.Usage.BillableInput != 18 || !res.Report.Usage.Known {
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
		{Model: "m", Output: llm.Output{Mode: llm.ModeJSON, Strict: true}},
		{Model: "m", Output: llm.Output{Mode: llm.ModeText, Name: "answer"}},
		{Model: "m", Options: llm.Options{Temperature: llm.Ptr(-0.1)}},
		{Model: "m", Options: llm.Options{MaxOutputTokens: -1}},
		{Model: "m", Options: llm.Options{Reasoning: "max"}},
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

	sum := (llm.Usage{Known: true}).Add(llm.Usage{BillableInput: 3, Known: true}).Add(llm.Usage{BillableInput: 4})
	if sum.BillableInput != 7 || sum.Known {
		t.Errorf("сумма %+v", sum)
	}

	big := llm.Usage{Output: 1<<62 + 1<<61, Known: true}
	if over := big.Add(big); over.Known || over.Output <= 0 {
		t.Errorf("переполнение %+v", over)
	}

	neg := (llm.Usage{Known: true}).Add(llm.Usage{CachedInput: -1, Known: true})
	if neg.Known || neg.CachedInput != 0 {
		t.Errorf("отрицательное %+v", neg)
	}

	raw := llm.Usage{Raw: map[string]int{"x": math.MaxInt}, Known: true}
	if over := raw.Add(raw); over.Known || over.Raw["x"] != math.MaxInt {
		t.Errorf("переполнение сырого счётчика %+v", over)
	}
}

// TestUsageAddKeepsPartsDisjoint — кеш GigaChat сверх оплачиваемого входа и рассуждения сверх
// выхода: сумма двух попыток складывает части, ничего не вычитая, и сырые счётчики по ключам.
func TestUsageAddKeepsPartsDisjoint(t *testing.T) {
	t.Parallel()

	attempt := llm.Usage{
		Raw:           map[string]int{"prompt_tokens": 100, "precached_prompt_tokens": 40, "completion_tokens": 9},
		BillableInput: 100,
		CachedInput:   40,
		Reasoning:     5,
		Output:        4,
		Known:         true,
	}

	sum := attempt.Add(attempt)
	if sum.BillableInput != 200 || sum.CachedInput != 80 || sum.InputTokens() != 280 ||
		sum.OutputTokens() != 18 || !sum.Known {
		t.Errorf("сумма %+v", sum)
	}

	if sum.Raw["prompt_tokens"] != 200 || sum.Raw["precached_prompt_tokens"] != 80 ||
		attempt.Raw["prompt_tokens"] != 100 {
		t.Errorf("сырые счётчики %v, слагаемое %v", sum.Raw, attempt.Raw)
	}

	if empty := (llm.Usage{Known: true}).Add(llm.Usage{Known: true}); empty.Raw != nil {
		t.Errorf("пустая сумма завела сырые счётчики %v", empty.Raw)
	}
}

// TestChatStopsOnTerminalFinish — предел длины и фильтр оплачены: попытка одна, класс never,
// исход свой, расход и причина в отчёте — и у удавшегося конверта, и у негодного содержимого.
func TestChatStopsOnTerminalFinish(t *testing.T) {
	t.Parallel()

	usage := llm.Usage{BillableInput: 12, Output: 30, Known: true}

	for _, tc := range []struct {
		step    step
		outcome string
		want    error
	}{
		{
			step:    step{res: llm.Result{Text: "{", Usage: usage, Finish: llm.Finish{Raw: "length", Kind: llm.FinishLength}}},
			outcome: llm.OutcomeTruncated,
			want:    llm.ErrTruncated,
		},
		{
			step: step{err: &llm.ResponseError{
				Message: "пусто", Usage: usage, Finish: llm.Finish{Raw: "blacklist", Kind: llm.FinishContentFilter},
			}},
			outcome: llm.OutcomeFiltered,
			want:    llm.ErrFiltered,
		},
	} {
		f := &fake{steps: []step{tc.step, ok("never", 0, 0)}}
		obs := &recorder{}

		_, err := client(f, obs).Chat(context.Background(), req())

		call := callError(t, err)
		if f.count() != 1 || call.Class != llm.RetryNever || llm.Recoverable(err) || !errors.Is(err, tc.want) {
			t.Errorf("%s: попыток %d, ошибка %+v", tc.outcome, f.count(), call)
		}

		if call.Report.Outcome != tc.outcome || call.Report.Usage.Output != 30 || call.Finish.Kind == "" ||
			obs.attempts[0].Outcome != tc.outcome || obs.attempts[0].Finish != call.Finish {
			t.Errorf("%s: отчёт %+v, попытка %+v", tc.outcome, call.Report, obs.attempts[0])
		}
	}
}

// TestChatRejectsUnconfirmedCapabilities — сверх профиля плеча запрос не уходит: класс never,
// плечо не зовётся; неизвестный профиль пропускает только текст без опций.
func TestChatRejectsUnconfirmedCapabilities(t *testing.T) {
	t.Parallel()

	img := llm.Image{MIME: "image/png", Data: []byte{1}}
	one := llm.Message{Role: llm.RoleUser, Images: []llm.Image{img}}
	two := []llm.Message{{Role: llm.RoleUser, Images: []llm.Image{img, img}}}
	profile := llm.Capabilities{Vision: true, Schema: true, MaxImagesPerMessage: 1, MaxImagesPerRequest: 2}

	for name, tc := range map[string]struct {
		caps llm.Capabilities
		req  llm.Request
	}{
		"vision":      {llm.Capabilities{}, llm.Request{Messages: []llm.Message{{Images: []llm.Image{img}}}}},
		"per message": {profile, llm.Request{Messages: two}},
		"per request": {profile, llm.Request{Messages: []llm.Message{one, one, one}}},
		"json":        {profile, llm.Request{Output: llm.Output{Mode: llm.ModeJSON}}},
		"strict": {profile, llm.Request{
			Output: llm.Output{Mode: llm.ModeSchema, Schema: json.RawMessage(`{}`), Strict: true},
		}},
		"temperature": {profile, llm.Request{Options: llm.Options{Temperature: llm.Ptr(0.2)}}},
		"reasoning":   {profile, llm.Request{Options: llm.Options{Reasoning: llm.EffortLow}}},
		"max output":  {profile, llm.Request{Options: llm.Options{MaxOutputTokens: 100}}},
	} {
		f := &fake{steps: []step{ok("never", 0, 0)}, caps: &tc.caps}
		tc.req.Model = "m"

		_, err := client(f, nil).Chat(context.Background(), tc.req)

		var request *llm.RequestError
		if call := callError(t, err); call.Class != llm.RetryNever || f.count() != 0 || !errors.As(err, &request) ||
			call.Report.Outcome != llm.OutcomeBadRequest {
			t.Errorf("%s: %v, вызовов плеча %d", name, err, f.count())
		}
	}

	f := &fake{steps: []step{ok("fits", 0, 0)}, caps: &profile}
	fits := llm.Request{Model: "m", Messages: []llm.Message{one, one}}

	if _, err := client(f, nil).Chat(context.Background(), fits); err != nil {
		t.Errorf("запрос в пределах профиля: %v", err)
	}

	unknown := &unprofiled{fake{steps: []step{ok("text", 0, 0)}}}
	if _, err := llm.New(unknown, time.Second).Chat(context.Background(), req()); err != nil {
		t.Errorf("текст при неизвестном профиле: %v", err)
	}

	jsonReq := req()
	jsonReq.Output.Mode = llm.ModeJSON

	if _, err := llm.New(unknown, time.Second).Chat(context.Background(), jsonReq); err == nil || unknown.count() != 1 {
		t.Errorf("json при неизвестном профиле: %v, вызовов %d", err, unknown.count())
	}
}

// unprofiled — плечо, не знающее профиля модели.
type unprofiled struct{ fake }

func (*unprofiled) Capabilities(string) (llm.Capabilities, bool) {
	return llm.Capabilities{Vision: true, JSON: true}, false
}

// TestChatCarriesAttemptReports — результат и отказ несут начало вызова и отчёты попыток
// с идентификатором запроса поставщика — и из ответа, и из не-2xx.
func TestChatCarriesAttemptReports(t *testing.T) {
	t.Parallel()

	failed := step{err: &llm.StatusError{Status: 502, RequestID: "r1"}}
	done := ok("done", 1, 1)
	done.res.RequestID = "r2"
	f := &fake{steps: []step{failed, done}}
	before := time.Now()

	res, err := client(f, nil).Chat(context.Background(), req())
	if err != nil {
		t.Fatal(err)
	}

	if len(res.Attempts) != 2 || res.Attempts[0].RequestID != "r1" || res.Attempts[1].RequestID != "r2" ||
		res.RequestID != "r2" || res.StartedAt.Before(before) || res.Attempts[1].StartedAt.Before(res.StartedAt) ||
		res.Attempts[1].Attempt != 2 {
		t.Errorf("отчёты попыток %+v, начало %s", res.Attempts, res.StartedAt)
	}

	f = &fake{steps: []step{failed}}
	c := client(f, nil)
	c.Attempts = 1

	_, err = c.Chat(context.Background(), req())
	if call := callError(t, err); len(call.Attempts) != 1 || call.Attempts[0].RequestID != "r1" ||
		call.StartedAt.IsZero() {
		t.Errorf("отказ %+v", call)
	}
}

// TestChatUsageKnownWithoutGeneration — отказ статусом и фазы до генерации расхода не несут:
// повтор после них не делает расход вызова неизвестным, а сетевой отказ генерации делает.
func TestChatUsageKnownWithoutGeneration(t *testing.T) {
	t.Parallel()

	upload := step{err: &llm.PhaseError{Phase: llm.PhaseUpload, Err: errors.New("сеть")}}
	f := &fake{steps: []step{status(429, 0), upload, ok("done", 2, 3)}}

	res, err := client(f, nil).Chat(context.Background(), req())
	if err != nil {
		t.Fatal(err)
	}

	if !res.Report.Usage.Known || !res.Attempts[0].Usage.Known || !res.Attempts[1].Usage.Known {
		t.Errorf("расход %+v, попытки %+v", res.Report.Usage, res.Attempts)
	}

	f = &fake{steps: []step{{err: errors.New("сеть")}, ok("done", 2, 3)}}

	res, err = client(f, nil).Chat(context.Background(), req())
	if err != nil {
		t.Fatal(err)
	}

	if res.Report.Usage.Known {
		t.Errorf("сетевой отказ генерации: расход %+v", res.Report.Usage)
	}
}

// TestChatConcurrentReportsStayApart — конкурентные вызовы одного клиента не смешивают отчёты попыток.
func TestChatConcurrentReportsStayApart(t *testing.T) {
	t.Parallel()

	c := client(&fake{steps: []step{status(500, 0), status(500, 0), ok("done", 1, 1)}}, nil)

	var wg sync.WaitGroup

	for i := range 20 {
		wg.Go(func() {
			r := req()
			r.CallID = strconv.Itoa(i)

			res, err := c.Chat(context.Background(), r)
			if err != nil {
				return
			}

			for _, a := range res.Attempts {
				if a.CallID != r.CallID {
					t.Errorf("вызов %s получил попытку %s", r.CallID, a.CallID)
				}
			}

			if len(res.Attempts) != res.Report.Attempts {
				t.Errorf("вызов %s: попыток %d, в отчёте %d", r.CallID, len(res.Attempts), res.Report.Attempts)
			}
		})
	}

	wg.Wait()
}

// TestArithmeticSaturates — бюджет и сдвиг паузы не переполняются.
func TestArithmeticSaturates(t *testing.T) {
	t.Parallel()

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

// TestCleanupWarningKeepsClassAndReport — предупреждение уборки не меняет класс отказа и едет в отчёт своей попытки.
func TestCleanupWarningKeepsClassAndReport(t *testing.T) {
	t.Parallel()

	warning := &llm.CleanupWarning{Files: []string{"f1"}, Err: errors.New("delete")}
	failed := step{err: &llm.WarnedError{
		Err:     &llm.StatusError{Status: http.StatusBadGateway, RequestID: "r1"},
		Cleanup: warning,
	}}
	f := &fake{steps: []step{failed, ok("done", 1, 1)}}

	res, err := client(f, nil).Chat(context.Background(), req())
	if err != nil {
		t.Fatal(err)
	}

	if f.count() != 2 || res.Attempts[0].Cleanup != warning || res.Attempts[0].RequestID != "r1" ||
		res.Attempts[0].Outcome != llm.OutcomeHTTP5xx || res.Attempts[1].Cleanup != nil || res.Cleanup != nil {
		t.Errorf("попыток %d, отчёты %+v", f.count(), res.Attempts)
	}
}
