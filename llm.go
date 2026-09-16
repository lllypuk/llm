// Package llm — вызов языковой модели одним контрактом поверх нескольких плеч:
// повторы, бюджет времени, классы отказа и наблюдатель — здесь, протокол
// поставщика — в подпакете (ollama, gigachat, yandex).
package llm

import (
	"context"
	"encoding/json"
	"fmt"
	"maps"
	"time"
)

// Role — роль сообщения. Инструкция и данные едут разными ролями: приехавшее
// с данными «забудь инструкции» иначе получило бы ту же роль, что правила.
type Role string

// Роли сообщений.
const (
	RoleSystem    Role = "system"
	RoleUser      Role = "user"
	RoleAssistant Role = "assistant"
)

// Mode — форма ответа: свободный текст, любой JSON или JSON по схеме.
type Mode string

// Формы ответа. Схему соблюдают не все модели — контракт полей живёт и в промпте.
const (
	ModeText   Mode = "text"
	ModeJSON   Mode = "json"
	ModeSchema Mode = "schema"
)

// Image — кадр при сообщении: байты и MIME, как их видит поставщик.
type Image struct {
	MIME string
	Data []byte
}

// Message — одно сообщение разговора; кадры едут при данных, а не при инструкции.
type Message struct {
	Role   Role
	Text   string
	Images []Image
}

// Output — требуемая форма ответа; Schema обязательна при ModeSchema. Name и Strict —
// только при схеме: имя схемы требует OpenAI-совместимый протокол, Strict — точное соблюдение.
type Output struct {
	Mode   Mode
	Schema json.RawMessage
	Name   string
	Strict bool
}

// Effort — глубина рассуждений модели; пустое — умолчание модели.
type Effort string

// Глубина рассуждений.
const (
	EffortLow    Effort = "low"
	EffortMedium Effort = "medium"
	EffortHigh   Effort = "high"
)

// Options — параметры генерации; нулевое значение поля — умолчание модели, не ноль.
type Options struct {
	Temperature     *float64
	Reasoning       Effort
	MaxOutputTokens int
}

// Request — один вызов модели. CallID связывает отчёты попыток и вызова между
// собой и с журналом потребителя; Task — имя задачи для метрик, набор значений
// ограничен конфигом потребителя.
type Request struct {
	CallID   string
	Task     string
	Model    string
	Messages []Message
	Output   Output
	Options  Options
}

// Validate отбивает запрос, который поставщик не сможет собрать: без модели,
// с неизвестным режимом, со схемой без схемы. Ошибка — [*RequestError].
func (r Request) Validate() error {
	var msg string

	switch {
	case r.Model == "":
		msg = "модель не задана"
	case r.Output.Mode == ModeSchema && len(r.Output.Schema) == 0:
		msg = "режим schema без схемы"
	case r.Output.Mode != "" && r.Output.Mode != ModeText && r.Output.Mode != ModeJSON && r.Output.Mode != ModeSchema:
		msg = "неизвестный режим ответа " + string(r.Output.Mode)
	case r.Output.Mode != ModeSchema && (r.Output.Name != "" || r.Output.Strict):
		msg = "имя схемы и strict без режима schema"
	case r.Options.Temperature != nil && *r.Options.Temperature < 0:
		msg = "отрицательная температура"
	case r.Options.MaxOutputTokens < 0:
		msg = "отрицательный предел ответа"
	case r.Options.Reasoning != "" && r.Options.Reasoning != EffortLow && r.Options.Reasoning != EffortMedium &&
		r.Options.Reasoning != EffortHigh:
		msg = "неизвестная глубина рассуждений " + string(r.Options.Reasoning)
	default:
		return nil
	}

	return &RequestError{Message: msg}
}

// Capabilities — что подтверждено у сочетания плеча, модели и режима API. Предел
// кадров ноль — предела нет; незаявленная возможность отбивается до вызова.
type Capabilities struct {
	Vision              bool
	JSON                bool
	Schema              bool
	Strict              bool
	SchemaName          bool // схема без имени отбивается
	MaxImagesPerMessage int
	MaxImagesPerRequest int
	Temperature         bool
	Reasoning           bool
	MaxOutputTokens     bool
}

// Check отбивает запрос, требующий неподтверждённого профилем; ошибка — [*RequestError].
func (c Capabilities) Check(r Request) error {
	total := 0
	perMessage := 0

	for _, m := range r.Messages {
		total += len(m.Images)
		perMessage = max(perMessage, len(m.Images))
	}

	var msg string

	switch {
	case total > 0 && !c.Vision:
		msg = "кадры не поддержаны"
	case c.MaxImagesPerMessage > 0 && perMessage > c.MaxImagesPerMessage:
		msg = fmt.Sprintf("кадров в сообщении больше %d", c.MaxImagesPerMessage)
	case c.MaxImagesPerRequest > 0 && total > c.MaxImagesPerRequest:
		msg = fmt.Sprintf("кадров в запросе больше %d", c.MaxImagesPerRequest)
	case r.Output.Mode == ModeJSON && !c.JSON:
		msg = "режим json не поддержан"
	case r.Output.Mode == ModeSchema && !c.Schema:
		msg = "режим schema не поддержан"
	case r.Output.Mode == ModeSchema && c.SchemaName && r.Output.Name == "":
		msg = "схема без имени"
	case r.Output.Strict && !c.Strict:
		msg = "strict не поддержан"
	case r.Options.Temperature != nil && !c.Temperature:
		msg = "температура не поддержана"
	case r.Options.Reasoning != "" && !c.Reasoning:
		msg = "глубина рассуждений не поддержана"
	case r.Options.MaxOutputTokens > 0 && !c.MaxOutputTokens:
		msg = "предел ответа не поддержан"
	default:
		return nil
	}

	return &RequestError{Message: msg}
}

// Usage — расход токенов. Части не пересекаются: вход — BillableInput плюс
// CachedInput, выход — Output плюс Reasoning; Raw — счётчики поставщика как есть.
// Known ложно, когда хотя бы часть расхода не сообщена: счёт тогда — нижняя граница.
type Usage struct {
	Raw           map[string]int
	BillableInput int
	CachedInput   int
	Reasoning     int
	Output        int
	Known         bool
}

// Add складывает расход по частям; неполнота любой части и переполнение делают сумму неполной.
func (u Usage) Add(part Usage) Usage {
	billable, okBillable := saturate(u.BillableInput, part.BillableInput)
	cached, okCached := saturate(u.CachedInput, part.CachedInput)
	reasoning, okReasoning := saturate(u.Reasoning, part.Reasoning)
	out, okOut := saturate(u.Output, part.Output)

	sum := Usage{
		BillableInput: billable,
		CachedInput:   cached,
		Reasoning:     reasoning,
		Output:        out,
		Known:         u.Known && part.Known && okBillable && okCached && okReasoning && okOut,
	}

	if len(u.Raw)+len(part.Raw) == 0 {
		return sum
	}

	sum.Raw = make(map[string]int, len(u.Raw)+len(part.Raw))
	maps.Copy(sum.Raw, u.Raw)

	for key, v := range part.Raw {
		var fits bool

		sum.Raw[key], fits = saturate(sum.Raw[key], v)
		sum.Known = sum.Known && fits
	}

	return sum
}

// InputTokens — весь вход, с кешем.
func (u Usage) InputTokens() int {
	n, _ := saturate(u.BillableInput, u.CachedInput)

	return n
}

// OutputTokens — весь выход, с рассуждениями.
func (u Usage) OutputTokens() int {
	n, _ := saturate(u.Output, u.Reasoning)

	return n
}

// FinishKind — нормализованная причина конца генерации; пустая — поставщик не сообщил
// или сообщил неизвестную.
type FinishKind string

// Причины конца генерации.
const (
	FinishStop          FinishKind = "stop"
	FinishLength        FinishKind = "length"
	FinishContentFilter FinishKind = "content_filter"
	FinishRefusal       FinishKind = "refusal"
	FinishToolCall      FinishKind = "tool_call"
)

// Finish — причина конца генерации: как её назвал поставщик и нормализованная.
type Finish struct {
	Raw  string
	Kind FinishKind
}

// Result — ответ модели. Usage, Finish и ServerLatency — удавшейся попытки; расход
// всех попыток — в Report, тот же отчёт получает наблюдатель.
type Result struct {
	// Model — имя из ответа поставщика; пустое у ответа заменяется запрошенным.
	Model  string
	Text   string
	Finish Finish
	Usage  Usage
	// RequestID — идентификатор запроса у поставщика, если он его вернул.
	RequestID string
	// ServerLatency — время на стороне поставщика, если он его сообщает; ноль — не сообщил.
	ServerLatency time.Duration
	// Cleanup — уборка удавшейся попытки не удалась; nil — убирать было нечего или убрано.
	Cleanup *CleanupWarning
	// StartedAt — начало вызова; Latency — сколько ждал вызывающий: все попытки и паузы.
	StartedAt time.Time
	Latency   time.Duration
	Report    CallReport
	Attempts  []AttemptReport
}

// Provider — одна попытка вызова у конкретного плеча. Отказ HTTP возвращается
// [*StatusError], негодный ответ — [*ResponseError] (с расходом, если он был в
// конверте), негодный запрос — [*RequestError], сеть и контекст — как есть;
// неудавшаяся уборка при отказе — [*WarnedError] поверх любого из них.
// Capabilities без сети; ложное ok — профиль модели неизвестен, не подтверждено ничего.
type Provider interface {
	Name() string
	Capabilities(model string) (Capabilities, bool)
	Complete(ctx context.Context, req Request) (Result, error)
	// AttemptOverhead — время плеча после срока попытки (уборка файлов); Route.Budget прибавляет его
	// на попытку. Обязательный метод, а не опция: обёртка потребителя иначе прятала бы его молча.
	AttemptOverhead() time.Duration
}

// Chat — узкий интерфейс потребителя: то, что делает [Client].
type Chat interface {
	Chat(ctx context.Context, req Request) (Result, error)
}

// Ptr — указатель на значение для необязательных полей запроса.
func Ptr[T any](v T) *T { return &v }
