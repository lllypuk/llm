package llm

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"maps"
	"slices"
	"strconv"
	"time"
)

// routeVersion — версия набора частей Descriptor.Fingerprint: смена меняет все отпечатки маршрутов.
const routeVersion = "route/1"

// TaskConfig — привязка задачи к плечу: модель, ревизия, форма ответа без схемы, опции и бюджет.
// Схема ответа — артефакт потребителя и едет входом; нулевые срок и число попыток — умолчания клиента.
type TaskConfig struct {
	Provider       string
	Model          string
	Revision       string
	Output         Output
	Options        Options
	AttemptTimeout time.Duration
	Attempts       int
	PricePlan      string
}

// Needs — сколько кадров задача шлёт на сообщение и на запрос; нулевое — задача текстовая.
type Needs struct {
	ImagesPerMessage int
	ImagesPerRequest int
}

// Router — задачи потребителя и собранные плечи; речь — свои плечи и задачи, чатовых не касаются. Собирается один раз на процесс: запасного
// плеча и перечитывания конфига нет намеренно — они делают неоднозначными расход, версию
// и причину отказа. Смена плеча — правка конфига и рестарт.
type Router struct {
	Providers   map[string]Provider
	Tasks       map[string]TaskConfig
	Speech      map[string]Transcriber
	SpeechTasks map[string]SpeechTaskConfig
	Observe     Observer
}

// Validate проверяет задачи потребителя до первого вызова: каждая из needs объявлена, лишних
// нет, плечо собрано, профиль модели подтверждён и принимает конфиг задачи вместе с её кадрами.
// nil — требований нет, задачи проверяются как текстовые; пустая карта — задачи не нужны вовсе.
// Задачи речи проверяются все: needs их не касается.
func (r *Router) Validate(needs map[string]Needs) error {
	var errs []error

	for _, name := range slices.Sorted(maps.Keys(needs)) {
		if _, ok := r.Tasks[name]; !ok {
			errs = append(errs, fmt.Errorf("tasks.%s: задача не объявлена", name))
		}

		if n := needs[name]; n.ImagesPerMessage < 0 || n.ImagesPerRequest < 0 {
			errs = append(errs, fmt.Errorf("tasks.%s: отрицательное число кадров", name))
		}
	}

	for _, name := range slices.Sorted(maps.Keys(r.Tasks)) {
		need, wanted := needs[name]
		if needs != nil && !wanted {
			errs = append(errs, fmt.Errorf("tasks.%s: задача не нужна потребителю", name))

			continue
		}

		if _, err := r.resolve(name, need); err != nil {
			errs = append(errs, err)
		}
	}

	for _, name := range slices.Sorted(maps.Keys(r.SpeechTasks)) {
		if _, err := r.resolveSpeech(name, r.SpeechTasks[name]); err != nil {
			errs = append(errs, err)
		}
	}

	return errors.Join(errs...)
}

// Resolve разрешает задачу в неизменяемый маршрут; правка Router после него маршрут не меняет.
// Кадры задачи Resolve не знает — их сверяет Validate и каждый вызов.
func (r *Router) Resolve(task string) (Route, error) {
	return r.resolve(task, Needs{})
}

func (r *Router) resolve(task string, need Needs) (Route, error) {
	cfg, ok := r.Tasks[task]
	if !ok {
		return Route{}, fmt.Errorf("tasks.%s: задача не объявлена", task)
	}

	provider := r.Providers[cfg.Provider]

	var msg string

	switch {
	case cfg.Provider == "":
		msg = "плечо не задано"
	case provider == nil:
		msg = fmt.Sprintf("плечо %q не собрано", cfg.Provider)
	case len(cfg.Output.Schema) > 0:
		msg = "схема в конфиге: она едет входом"
	case cfg.AttemptTimeout < 0:
		msg = "отрицательный срок попытки"
	case cfg.Attempts < 0:
		msg = "отрицательное число попыток"
	}

	if msg != "" {
		return Route{}, fmt.Errorf("tasks.%s: %s", task, msg)
	}

	desc := Descriptor{
		Task:      task,
		Provider:  cfg.Provider,
		Kind:      provider.Name(),
		Model:     cfg.Model,
		Revision:  cfg.Revision,
		Output:    Output{Mode: cfg.Output.Mode, Name: cfg.Output.Name, Strict: cfg.Output.Strict},
		Options:   cfg.Options.clone(),
		PricePlan: cfg.PricePlan,
	}

	if desc.Output.Mode == "" {
		desc.Output.Mode = ModeText
	}

	if err := admitProfile(provider, desc.probe(need)); err != nil {
		return Route{}, fmt.Errorf("tasks.%s: %s/%s: %w", task, desc.Kind, desc.Model, err)
	}

	client := New(provider, cfg.AttemptTimeout)
	if cfg.Attempts > 0 {
		client.Attempts = cfg.Attempts
	}

	client.Observe = r.Observe

	return Route{desc: desc, client: client}, nil
}

// admitProfile — запрос собираем и подтверждённый профиль его принимает; неизвестный профиль — отказ.
func admitProfile(p Provider, req Request) error {
	if err := req.Validate(); err != nil {
		return err
	}

	caps, known := p.Capabilities(req.Model)
	if !known {
		return errors.New("профиль модели не подтверждён")
	}

	return caps.Check(req)
}

// Descriptor — всё, что маршрут фиксирует о вызове: источник версии вызова и полей журнала.
// Provider — имя плеча в конфиге, Kind — адаптер.
type Descriptor struct {
	Task      string
	Provider  string
	Kind      string
	Model     string
	Revision  string
	Output    Output
	Options   Options
	PricePlan string
}

// Fingerprint — отпечаток маршрута вместе с частями потребителя (хеш промпта, схемы, подготовка
// входа). Имя плеча в конфиге, тариф и бюджет не входят: переименование и смена цены вызов не меняют.
func (d Descriptor) Fingerprint(parts ...string) string {
	temperature := ""
	if d.Options.Temperature != nil {
		temperature = strconv.FormatFloat(*d.Options.Temperature, 'g', -1, 64)
	}

	route := []string{
		routeVersion,
		d.Kind,
		d.Model,
		d.Revision,
		string(d.Output.Mode),
		d.Output.Name,
		strconv.FormatBool(d.Output.Strict),
		temperature,
		string(d.Options.Reasoning),
		strconv.Itoa(d.Options.MaxOutputTokens),
	}

	return Fingerprint(append(route, parts...)...)
}

// probe — запрос с формой и опциями маршрута и кадрами задачи, для сверки профиля без вызова.
func (d Descriptor) probe(need Needs) Request {
	req := Request{Model: d.Model, Output: d.Output, Options: d.Options}
	if d.Output.Mode == ModeSchema {
		req.Output.Schema = json.RawMessage(`{}`)
	}

	for left := max(need.ImagesPerRequest, need.ImagesPerMessage); left > 0; {
		n := left
		if need.ImagesPerMessage > 0 {
			n = min(left, need.ImagesPerMessage)
		}

		req.Messages = append(req.Messages, Message{Role: RoleUser, Images: make([]Image, n)})
		left -= n
	}

	return req
}

// Input — то, что задаёт вызывающий. Модель и опции задаёт маршрут; Mode — форма, которую ждёт
// разбор потребителя: пустое принимает маршрут, иное обязано с ним совпасть.
type Input struct {
	CallID   string
	Messages []Message
	Mode     Mode
	Schema   json.RawMessage
}

// Route — задача, разрешённая в плечо; нулевое значение не вызывает ничего.
type Route struct {
	desc   Descriptor
	client *Client
}

// Descriptor — копия дескриптора: правка её маршрут не меняет.
func (r Route) Descriptor() Descriptor {
	d := r.desc
	d.Options = d.Options.clone()

	return d
}

// Budget — [Client.Budget] маршрута плюс [Provider.AttemptOverhead] на каждую попытку.
func (r Route) Budget() time.Duration {
	if r.client == nil {
		return 0
	}

	return addDuration(r.client.Budget(), mulDuration(r.client.Provider.AttemptOverhead(), r.client.attempts()))
}

// Chat вызывает модель маршрутом. Форма или схема входа, расходящаяся с маршрутом, — отказ
// never до плеча: подмена формы конфигом иначе дошла бы до разбора потребителя молча.
func (r Route) Chat(ctx context.Context, in Input) (Result, error) {
	if r.client == nil {
		return Result{}, &CallError{Class: RetryNever, Err: errRouteMissing}
	}

	req := Request{
		CallID:   in.CallID,
		Task:     r.desc.Task,
		Model:    r.desc.Model,
		Messages: in.Messages,
		Output:   r.desc.Output,
		Options:  r.desc.Options.clone(),
	}
	req.Output.Schema = in.Schema

	var msg string

	switch {
	case in.Mode != "" && in.Mode != r.desc.Output.Mode:
		msg = fmt.Sprintf("вход ждёт форму %s, маршрут задачи %s — %s", in.Mode, r.desc.Task, r.desc.Output.Mode)
	case len(in.Schema) > 0 && r.desc.Output.Mode != ModeSchema:
		msg = fmt.Sprintf("схема во входе, маршрут задачи %s — %s", r.desc.Task, r.desc.Output.Mode)
	default:
		return r.client.Chat(ctx, req)
	}

	err := &RequestError{Message: msg}
	run := &call{
		client:   r.client,
		provider: r.desc.Kind,
		req:      req,
		started:  time.Now(),
		spent:    Usage{Known: true},
		outcome:  OutcomeBadRequest,
	}

	return Result{}, run.fail(&CallError{
		Provider: run.provider, Model: req.Model, Class: RetryNever, Message: msg, Err: err,
	})
}

// clone — опции без общего указателя температуры.
func (o Options) clone() Options {
	if o.Temperature != nil {
		o.Temperature = Ptr(*o.Temperature)
	}

	return o
}

// errRouteMissing — маршрут не разрешён: нулевой Route.
var errRouteMissing = errors.New("маршрут не разрешён")
