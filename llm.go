// Package llm — вызов языковой модели одним контрактом поверх нескольких плеч:
// повторы, бюджет времени, классы отказа и наблюдатель — здесь, протокол
// поставщика — в подпакете (ollama; gigachat и yandex следом).
package llm

import (
	"context"
	"encoding/json"
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

// Output — требуемая форма ответа; Schema обязательна при ModeSchema.
type Output struct {
	Mode   Mode
	Schema json.RawMessage
}

// Request — один вызов модели. CallID связывает отчёты попыток и вызова между
// собой и с журналом потребителя; Task — имя задачи для метрик, набор значений
// ограничен конфигом потребителя.
type Request struct {
	CallID      string
	Task        string
	Model       string
	Messages    []Message
	Output      Output
	Temperature *float64
}

// Validate отбивает запрос, который поставщик не сможет собрать: без модели,
// с неизвестным режимом, со схемой без схемы. Ошибка — [*RequestError].
func (r Request) Validate() error {
	switch {
	case r.Model == "":
		return &RequestError{Message: "модель не задана"}
	case r.Output.Mode == ModeSchema && len(r.Output.Schema) == 0:
		return &RequestError{Message: "режим schema без схемы"}
	case r.Output.Mode != "" && r.Output.Mode != ModeText && r.Output.Mode != ModeJSON && r.Output.Mode != ModeSchema:
		return &RequestError{Message: "неизвестный режим ответа " + string(r.Output.Mode)}
	}

	return nil
}

// Usage — расход токенов. Known ложно, когда хотя бы часть расхода поставщик не
// сообщил: счётчики тогда — известная нижняя граница, а не полный счёт.
type Usage struct {
	InputTokens  int
	OutputTokens int
	Known        bool
}

// Add складывает расход; неполнота любой части делает сумму неполной.
func (u Usage) Add(part Usage) Usage {
	return Usage{
		InputTokens:  saturate(u.InputTokens, part.InputTokens),
		OutputTokens: saturate(u.OutputTokens, part.OutputTokens),
		Known:        u.Known && part.Known,
	}
}

// Result — ответ модели. Usage и ServerLatency — удавшейся попытки; расход всех
// попыток и число попыток — в Report, тот же отчёт получает наблюдатель.
type Result struct {
	// Model — имя из ответа поставщика; пустое у ответа заменяется запрошенным.
	Model        string
	Text         string
	FinishReason string
	Usage        Usage
	// ServerLatency — время на стороне поставщика, если он его сообщает; ноль — не сообщил.
	ServerLatency time.Duration
	// Latency — сколько ждал вызывающий: все попытки и паузы между ними.
	Latency time.Duration
	Report  CallReport
}

// Provider — одна попытка вызова у конкретного плеча. Отказ HTTP возвращается
// [*StatusError], негодный ответ — [*ResponseError] (с расходом, если он был в
// конверте), негодный запрос — [*RequestError], сеть и контекст — как есть.
type Provider interface {
	Name() string
	Complete(ctx context.Context, req Request) (Result, error)
}

// Chat — узкий интерфейс потребителя: то, что делает [Client].
type Chat interface {
	Chat(ctx context.Context, req Request) (Result, error)
}

// Ptr — указатель на значение для необязательных полей запроса.
func Ptr[T any](v T) *T { return &v }
