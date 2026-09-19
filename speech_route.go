package llm

import (
	"context"
	"errors"
	"fmt"
	"net/http"
	"time"
)

// speechRouteVersion — версия набора частей SpeechDescriptor.Fingerprint.
const speechRouteVersion = "speech/1"

// SpeechTaskConfig — привязка задачи речи к плечу распознавания; нулевые срок и число попыток — умолчания клиента.
type SpeechTaskConfig struct {
	Provider       string
	Model          string
	Language       string
	AttemptTimeout time.Duration
	Attempts       int
	PricePlan      string
}

// SpeechDescriptor — что маршрут речи фиксирует о вызове. Provider — имя плеча в конфиге, Kind — адаптер.
type SpeechDescriptor struct {
	Task      string
	Provider  string
	Kind      string
	Model     string
	Language  string
	PricePlan string
}

// Fingerprint — отпечаток маршрута речи с частями потребителя; имя плеча, тариф и бюджет не входят.
func (d SpeechDescriptor) Fingerprint(parts ...string) string {
	return Fingerprint(append([]string{speechRouteVersion, d.Kind, d.Model, d.Language}, parts...)...)
}

// ResolveSpeech разрешает задачу речи в неизменяемый маршрут; false без ошибки — задача не объявлена.
func (r *Router) ResolveSpeech(task string) (SpeechRoute, bool, error) {
	cfg, ok := r.SpeechTasks[task]
	if !ok {
		return SpeechRoute{}, false, nil
	}

	route, err := r.resolveSpeech(task, cfg)
	if err != nil {
		return SpeechRoute{}, false, err
	}

	return route, true, nil
}

func (r *Router) resolveSpeech(task string, cfg SpeechTaskConfig) (SpeechRoute, error) {
	provider := r.Speech[cfg.Provider]

	var msg string

	switch {
	case cfg.Provider == "":
		msg = "плечо не задано"
	case provider == nil:
		msg = fmt.Sprintf("плечо %q не собрано", cfg.Provider)
	case cfg.Model == "":
		msg = msgNoModel
	case cfg.AttemptTimeout < 0:
		msg = "отрицательный срок попытки"
	case cfg.Attempts < 0:
		msg = "отрицательное число попыток"
	}

	if msg != "" {
		return SpeechRoute{}, fmt.Errorf("speech.%s: %s", task, msg)
	}

	caps, known := provider.SpeechCapabilities(cfg.Model)
	if !known {
		return SpeechRoute{}, fmt.Errorf(
			"speech.%s: %s/%s: профиль модели не подтверждён",
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

	return SpeechRoute{
		desc: SpeechDescriptor{
			Task:      task,
			Provider:  cfg.Provider,
			Kind:      provider.Name(),
			Model:     cfg.Model,
			Language:  cfg.Language,
			PricePlan: cfg.PricePlan,
		},
		caps:     caps,
		provider: provider,
		client:   client,
	}, nil
}

// SpeechInput — запись от вызывающего: сырые сэмплы LPCM, 16 бит, моно. Модель и язык задаёт маршрут.
type SpeechInput struct {
	CallID     string
	SampleRate int
	PCM        []byte
}

// SpeechResult — распознанный текст с отчётом вызова; Model — из ответа, иначе запрошенная.
type SpeechResult struct {
	Transcript

	StartedAt time.Time
	Latency   time.Duration
	Attempts  []AttemptReport
	Report    CallReport
}

// SpeechRoute — задача речи, разрешённая в плечо; нулевое значение не вызывает ничего.
// Клиент в нём — только бюджет, паузы и наблюдатель: чатового плеча у него нет.
type SpeechRoute struct {
	desc     SpeechDescriptor
	caps     SpeechCapabilities
	provider Transcriber
	client   *Client
}

// Descriptor — копия дескриптора.
func (r SpeechRoute) Descriptor() SpeechDescriptor { return r.desc }

// Budget — [Client.Budget] маршрута: все попытки со сроками и худшими паузами.
func (r SpeechRoute) Budget() time.Duration {
	if r.client == nil {
		return 0
	}

	return r.client.Budget()
}

// Transcribe распознаёт запись маршрутом с повторами тех же классов, что у [Client.Chat].
// Запись, которую профиль не примет, отбивается до плеча отказом [RetryNever].
func (r SpeechRoute) Transcribe(ctx context.Context, in SpeechInput) (SpeechResult, error) {
	run := &call{
		client:   r.client,
		provider: r.desc.Kind,
		req:      Request{CallID: in.CallID, Task: r.desc.Task, Model: r.desc.Model},
		started:  time.Now(),
		spent:    Usage{Known: true},
	}

	if r.client == nil {
		run.client = &Client{}

		return SpeechResult{}, run.fail(&CallError{Class: RetryNever, Err: errRouteMissing})
	}

	req := SpeechRequest{Model: r.desc.Model, Language: r.desc.Language, SampleRate: in.SampleRate, PCM: in.PCM}
	if err := req.Validate(r.caps.MaxAudio); err != nil {
		run.outcome = OutcomeBadRequest

		return SpeechResult{}, run.fail(&CallError{
			Provider: run.provider, Model: req.Model, Class: RetryNever, Message: err.Error(), Err: err,
		})
	}

	return r.do(ctx, run, req)
}

// do — цикл попыток речи. Client.do не обобщается намеренно: общие у них пауза, срок и классификация.
func (r SpeechRoute) do(ctx context.Context, run *call, req SpeechRequest) (SpeechResult, error) {
	c := r.client
	attempts := c.attempts()

	for attempt := 1; ; attempt++ {
		started := time.Now()
		tr, latency, err := r.once(ctx, req)
		meta := speechMetaOf(tr, req, err)
		run.audio += meta.audio

		if meta.model != "" {
			run.model = meta.model
		}

		run.observeAttempt(attempt, started, latency, attemptOutcome(ctx, err, Finish{}), err, meta)

		if err == nil {
			return run.succeedSpeech(tr), nil
		}

		fail := classify(run.provider, req.Model, err)

		tooLong := fail.RetryAfter > c.maxRetryAfter() && fail.Class != RetryNever &&
			fail.Class != RetryNeedsConfiguration
		if tooLong {
			fail.Class = RetryAfterDelay
		}

		if fail.Class == RetryNever || fail.Class == RetryNeedsConfiguration || tooLong || attempt == attempts {
			return SpeechResult{}, run.fail(fail)
		}

		if waitErr := sleep(ctx, c.retryPause(attempt, fail.RetryAfter)); waitErr != nil {
			fail.Err = fmt.Errorf("ожидание повтора: %w", waitErr)

			return SpeechResult{}, run.fail(fail)
		}
	}
}

func (r SpeechRoute) once(ctx context.Context, req SpeechRequest) (Transcript, time.Duration, error) {
	attemptCtx, cancel := context.WithTimeout(ctx, r.client.timeout())
	defer cancel()

	started := time.Now()
	tr, err := r.provider.Transcribe(attemptCtx, req)

	return tr, time.Since(started), err
}

// speechMetaOf — токенов у речи нет, их ноль известен; запись считается отправленной,
// кроме доказуемого отказа до работы плеча: конфигурация, не-генерация и 4xx, кроме 408.
func speechMetaOf(tr Transcript, req SpeechRequest, err error) attemptMeta {
	meta := attemptMeta{usage: Usage{Known: true}}

	if err == nil {
		meta.model = tr.Model
		meta.requestID = tr.RequestID
		meta.audio = tr.AudioMillis

		if meta.audio == 0 {
			meta.audio = req.Duration().Milliseconds()
		}

		return meta
	}

	var status *StatusError
	if errors.As(err, &status) {
		meta.requestID = status.RequestID
	}

	rejected := status != nil && status.Status >= http.StatusBadRequest &&
		status.Status < http.StatusInternalServerError && status.Status != http.StatusRequestTimeout
	if !rejected && !misconfigured(err) && phaseOf(err) == PhaseInference {
		meta.audio = req.Duration().Milliseconds()
	}

	return meta
}

func (r *call) succeedSpeech(tr Transcript) SpeechResult {
	res := SpeechResult{Transcript: tr}
	res.Report = r.report(OutcomeOK, "", tr.Model)
	res.StartedAt = r.started
	res.Latency = res.Report.Duration
	res.Attempts = r.attempts
	r.client.observer().Call(res.Report)

	if res.Model == "" {
		res.Model = r.req.Model
	}

	return res
}
