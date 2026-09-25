package llm

import (
	"context"
	"errors"
	"fmt"
	"slices"
	"strconv"
	"time"
)

// ocrRouteVersion — версия набора частей OCRDescriptor.Fingerprint.
const ocrRouteVersion = "ocr/1"

// OCRTaskConfig — привязка задачи OCR к плечу распознавания; нулевые срок и число попыток — умолчания клиента.
type OCRTaskConfig struct {
	Provider       string
	Model          string
	Languages      []string
	AttemptTimeout time.Duration
	Attempts       int
	PricePlan      string
}

// OCRDescriptor — что маршрут OCR фиксирует о вызове. Provider — имя плеча в конфиге, Kind — адаптер.
type OCRDescriptor struct {
	Task      string
	Provider  string
	Kind      string
	Model     string
	Languages []string
	PricePlan string
}

// Fingerprint — отпечаток маршрута OCR с частями потребителя; имя плеча, тариф и бюджет не входят.
// Число языков входит, чтобы граница языков и частей потребителя была однозначной.
func (d OCRDescriptor) Fingerprint(parts ...string) string {
	head := append([]string{ocrRouteVersion, d.Kind, d.Model, strconv.Itoa(len(d.Languages))}, d.Languages...)

	return Fingerprint(append(head, parts...)...)
}

// ResolveOCR разрешает задачу OCR в неизменяемый маршрут; false без ошибки — задача не объявлена.
func (r *Router) ResolveOCR(task string) (OCRRoute, bool, error) {
	cfg, ok := r.OCRTasks[task]
	if !ok {
		return OCRRoute{}, false, nil
	}

	route, err := r.resolveOCR(task, cfg)
	if err != nil {
		return OCRRoute{}, false, err
	}

	return route, true, nil
}

func (r *Router) resolveOCR(task string, cfg OCRTaskConfig) (OCRRoute, error) {
	provider := r.OCR[cfg.Provider]

	var msg string

	switch {
	case cfg.Provider == "":
		msg = msgNoProvider
	case provider == nil:
		msg = fmt.Sprintf("плечо %q не собрано", cfg.Provider)
	case cfg.Model == "":
		msg = msgNoModel
	case len(cfg.Languages) == 0:
		msg = "языки не заданы"
	case slices.Contains(cfg.Languages, ""):
		msg = "пустой язык"
	case cfg.AttemptTimeout < 0:
		msg = msgNegativeTimeout
	case cfg.Attempts < 0:
		msg = msgNegativeAttempts
	}

	if msg != "" {
		return OCRRoute{}, fmt.Errorf("ocr.%s: %s", task, msg)
	}

	caps, known := provider.OCRCapabilities(cfg.Model)
	if !known {
		return OCRRoute{}, fmt.Errorf(
			"ocr.%s: %s/%s: профиль модели не подтверждён",
			task,
			provider.Name(),
			cfg.Model,
		)
	}

	client := New(nil, cfg.AttemptTimeout)
	if cfg.Attempts > 0 {
		client.Attempts = cfg.Attempts
	}

	client.Observe = r.Observe

	return OCRRoute{
		desc: OCRDescriptor{
			Task:      task,
			Provider:  cfg.Provider,
			Kind:      provider.Name(),
			Model:     cfg.Model,
			Languages: slices.Clone(cfg.Languages),
			PricePlan: cfg.PricePlan,
		},
		caps:     caps,
		provider: provider,
		client:   client,
	}, nil
}

// OCRInput — кадр от вызывающего: JPEG или PNG целиком. Модель и языки задаёт маршрут.
type OCRInput struct {
	CallID string
	Image  []byte
	MIME   string
}

// OCRResult — распознанный текст кадра с отчётом вызова; Model — из ответа, иначе запрошенная.
type OCRResult struct {
	OCRText

	StartedAt time.Time
	Latency   time.Duration
	Attempts  []AttemptReport
	Report    CallReport
}

// OCRRoute — задача OCR, разрешённая в плечо; нулевое значение не вызывает ничего.
// Клиент в нём — только бюджет, паузы и наблюдатель: чатового плеча у него нет.
type OCRRoute struct {
	desc     OCRDescriptor
	caps     OCRCapabilities
	provider Recognizer
	client   *Client
}

// Descriptor — копия дескриптора.
func (r OCRRoute) Descriptor() OCRDescriptor {
	desc := r.desc
	desc.Languages = slices.Clone(r.desc.Languages)

	return desc
}

// Budget — [Client.Budget] маршрута: все попытки со сроками и худшими паузами.
func (r OCRRoute) Budget() time.Duration {
	if r.client == nil {
		return 0
	}

	return r.client.Budget()
}

// Recognize распознаёт кадр маршрутом с повторами тех же классов, что у [Client.Chat].
// Кадр, который профиль не примет, отбивается до плеча отказом [RetryNever].
func (r OCRRoute) Recognize(ctx context.Context, in OCRInput) (OCRResult, error) {
	run := &call{
		client:   r.client,
		provider: r.desc.Kind,
		req:      Request{CallID: in.CallID, Task: r.desc.Task, Model: r.desc.Model},
		started:  time.Now(),
		spent:    Usage{Known: true},
	}

	if r.client == nil {
		run.client = &Client{}

		return OCRResult{}, run.fail(&CallError{Class: RetryNever, Err: errRouteMissing})
	}

	req := OCRRequest{Model: r.desc.Model, Image: in.Image, MIME: in.MIME, Languages: slices.Clone(r.desc.Languages)}
	if err := req.Validate(r.caps.MaxBytes); err != nil {
		run.outcome = OutcomeBadRequest

		return OCRResult{}, run.fail(&CallError{
			Provider: run.provider, Model: req.Model, Class: RetryNever, Message: err.Error(), Err: err,
		})
	}

	return r.do(ctx, run, req)
}

// do — цикл попыток OCR. Client.do не обобщается намеренно: общие у них пауза, срок и классификация.
func (r OCRRoute) do(ctx context.Context, run *call, req OCRRequest) (OCRResult, error) {
	c := r.client
	attempts := c.attempts()

	for attempt := 1; ; attempt++ {
		started := time.Now()
		text, latency, err := r.once(ctx, req)
		meta := ocrMetaOf(text, err)
		run.pages += meta.pages

		if meta.model != "" {
			run.model = meta.model
		}

		run.observeAttempt(attempt, started, latency, attemptOutcome(ctx, err, Finish{}), err, meta)

		if err == nil {
			return run.succeedOCR(text), nil
		}

		fail := classify(run.provider, req.Model, err)

		tooLong := fail.RetryAfter > c.maxRetryAfter() && fail.Class != RetryNever &&
			fail.Class != RetryNeedsConfiguration
		if tooLong {
			fail.Class = RetryAfterDelay
		}

		if fail.Class == RetryNever || fail.Class == RetryNeedsConfiguration || tooLong || attempt == attempts {
			return OCRResult{}, run.fail(fail)
		}

		if waitErr := sleep(ctx, c.retryPause(attempt, fail.RetryAfter)); waitErr != nil {
			fail.Err = fmt.Errorf("ожидание повтора: %w", waitErr)

			return OCRResult{}, run.fail(fail)
		}
	}
}

func (r OCRRoute) once(ctx context.Context, req OCRRequest) (OCRText, time.Duration, error) {
	attemptCtx, cancel := context.WithTimeout(ctx, r.client.timeout())
	defer cancel()

	started := time.Now()
	text, err := r.provider.Recognize(attemptCtx, req)

	return text, time.Since(started), err
}

// ocrMetaOf — токенов у OCR нет, их ноль известен; кадр — страница в расходе,
// кроме доказуемого отказа до работы плеча ([refusedBeforeWork]).
func ocrMetaOf(text OCRText, err error) attemptMeta {
	meta := attemptMeta{usage: Usage{Known: true}}

	if err == nil {
		meta.model = text.Model
		meta.requestID = text.RequestID
		meta.pages = 1

		return meta
	}

	var status *StatusError
	if errors.As(err, &status) {
		meta.requestID = status.RequestID
	}

	if !refusedBeforeWork(err) {
		meta.pages = 1
	}

	return meta
}

func (r *call) succeedOCR(text OCRText) OCRResult {
	res := OCRResult{OCRText: text}
	res.Report = r.report(OutcomeOK, "", text.Model)
	res.StartedAt = r.started
	res.Latency = res.Report.Duration
	res.Attempts = r.attempts
	r.client.observer().Call(res.Report)

	if res.Model == "" {
		res.Model = r.req.Model
	}

	return res
}
