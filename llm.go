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

// Output — требуемая форма ответа; Schema читается только при ModeSchema.
type Output struct {
	Mode   Mode
	Schema json.RawMessage
}

// Request — один вызов модели. Task — имя задачи для метрик и журнала,
// не текст: набор значений ограничен конфигом потребителя.
type Request struct {
	Task        string
	Model       string
	Messages    []Message
	Output      Output
	Temperature *float64
}

// Usage — расход токенов. Known ложно, когда поставщик расход не сообщил:
// неизвестный ноль в счёте неотличим от бесплатного вызова.
type Usage struct {
	InputTokens  int
	OutputTokens int
	Known        bool
}

// Result — ответ модели с тем, что ложится в журнал вызова.
type Result struct {
	// Model — имя из ответа поставщика; пустое у ответа заменяется запрошенным.
	Model        string
	Text         string
	FinishReason string
	Usage        Usage
	// ServerLatency — время на стороне поставщика, если он его сообщает; ноль — не сообщил.
	ServerLatency time.Duration
	// Latency — сколько ждал вызывающий: все попытки и паузы между ними.
	Latency  time.Duration
	Attempts int
}

// Provider — одна попытка вызова у конкретного плеча. Отказ HTTP возвращается
// [*StatusError], негодный ответ — [*ResponseError], сеть и контекст — как есть.
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
