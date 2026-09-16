package gigachat

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"strings"

	"github.com/lllypuk/llm"
	"github.com/lllypuk/llm/internal/httpjson"
)

// maxChatBody — предел ответа `/chat/completions`; длиннее — отказ попытки, а не обрезанный ответ.
const maxChatBody = 1 << 20

// Capabilities — профиль моделей второго поколения. Аналога `json_object` у GigaChat нет: ModeJSON
// не подтверждён, только схема. Кадры — у Pro и Max; неизвестная модель не подтверждает ничего.
func (p *Provider) Capabilities(model string) (llm.Capabilities, bool) {
	caps := llm.Capabilities{
		JSON:            false,
		Schema:          true,
		Strict:          true,
		Temperature:     true,
		MaxOutputTokens: true,
	}

	switch model {
	case "GigaChat-2":
	case "GigaChat-2-Pro", "GigaChat-2-Max":
		caps.Vision = true
		caps.MaxImagesPerMessage = MaxImagesPerMessage
		caps.MaxImagesPerRequest = MaxImagesPerRequest
	default:
		return llm.Capabilities{}, false
	}

	return caps, true
}

type chatMessage struct {
	Role        string   `json:"role"`
	Content     string   `json:"content"`
	Attachments []string `json:"attachments,omitempty"`
}

type responseFormat struct {
	Type   string          `json:"type"`
	Schema json.RawMessage `json:"schema"`
	Strict bool            `json:"strict"`
}

type chatRequest struct {
	Model          string          `json:"model"`
	Messages       []chatMessage   `json:"messages"`
	Stream         bool            `json:"stream"`
	Temperature    *float64        `json:"temperature,omitempty"`
	MaxTokens      int             `json:"max_tokens,omitempty"`
	ResponseFormat *responseFormat `json:"response_format,omitempty"`
}

// chatResponse — конверт ответа. Счётчики указателями: отсутствующий отличается от нуля.
type chatResponse struct {
	Model   string `json:"model"`
	Choices []struct {
		Message struct {
			Content string `json:"content"`
		} `json:"message"`
		FinishReason string `json:"finish_reason"`
	} `json:"choices"`
	Usage *struct {
		PromptTokens          *int `json:"prompt_tokens"`
		CompletionTokens      *int `json:"completion_tokens"`
		PrecachedPromptTokens *int `json:"precached_prompt_tokens"`
		TotalTokens           *int `json:"total_tokens"`
	} `json:"usage"`
}

// usage — расход из конверта. prompt_tokens у GigaChat уже без кеша: кеш не вычитается, а
// ложится отдельной частью; не названный кеш делает расход неполным.
func (r chatResponse) usage() llm.Usage {
	if r.Usage == nil {
		return llm.Usage{}
	}

	u := llm.Usage{Raw: map[string]int{}, Known: true}
	u.BillableInput, u.Known = counter(u.Raw, "prompt_tokens", r.Usage.PromptTokens, u.Known)
	u.CachedInput, u.Known = counter(u.Raw, "precached_prompt_tokens", r.Usage.PrecachedPromptTokens, u.Known)
	u.Output, u.Known = counter(u.Raw, "completion_tokens", r.Usage.CompletionTokens, u.Known)

	if r.Usage.TotalTokens != nil {
		u.Raw["total_tokens"] = *r.Usage.TotalTokens
	}

	return u
}

// counter — счётчик конверта; присланный кладётся в raw как есть, даже отрицательный.
func counter(raw map[string]int, key string, v *int, known bool) (int, bool) {
	if v == nil {
		return 0, false
	}

	raw[key] = *v
	if *v < 0 {
		return 0, false
	}

	return *v, known
}

// finish — finish_reason первого варианта; blacklist — фильтр GigaChat.
func (r chatResponse) finish() llm.Finish {
	if len(r.Choices) == 0 {
		return llm.Finish{}
	}

	f := llm.Finish{Raw: r.Choices[0].FinishReason}

	switch f.Raw {
	case "stop":
		f.Kind = llm.FinishStop
	case "length":
		f.Kind = llm.FinishLength
	case "blacklist":
		f.Kind = llm.FinishContentFilter
	case "function_call":
		f.Kind = llm.FinishToolCall
	}

	return f
}

// reject — отказ по негодному содержимому с метаданными конверта: расход и модель уже были.
func (r chatResponse) reject(msg, requestID string) error {
	return &llm.ResponseError{
		Message:   msg,
		Usage:     r.usage(),
		Model:     r.Model,
		Finish:    r.finish(),
		RequestID: requestID,
	}
}

// Complete — одна попытка: загрузка кадров, генерация, уборка файлов. Профиль сверяется и здесь:
// ModeJSON протоколом не выражается и без проверки ушёл бы свободным текстом.
func (p *Provider) Complete(ctx context.Context, req llm.Request) (llm.Result, error) {
	if err := req.Validate(); err != nil {
		return llm.Result{}, err
	}

	caps, _ := p.Capabilities(req.Model)
	if err := caps.Check(req); err != nil {
		return llm.Result{}, err
	}

	return p.withFiles(ctx, req.Messages, func(ctx context.Context, attachments [][]string) (llm.Result, error) {
		return p.chat(ctx, encode(req, attachments))
	})
}

func (p *Provider) chat(ctx context.Context, body chatRequest) (llm.Result, error) {
	resp, err := p.do(ctx, func(ctx context.Context) (*http.Request, error) {
		return httpjson.NewRequest(ctx, p.api+"/chat/completions", body, nil)
	})
	if err != nil {
		return llm.Result{}, fmt.Errorf("запрос /chat/completions: %w", err)
	}

	defer func() { _ = resp.Body.Close() }()

	requestID := resp.Header.Get("X-Request-Id")

	if resp.StatusCode != http.StatusOK {
		st := httpjson.ReadStatus(resp, maxErrorBody, errorMessage)

		return llm.Result{}, &llm.StatusError{
			Status:     st.Code,
			Message:    st.Message,
			RetryAfter: st.RetryAfter,
			RequestID:  requestID,
		}
	}

	var env chatResponse
	if err = httpjson.Decode(resp.Body, maxChatBody, &env); err != nil {
		return llm.Result{}, &llm.ResponseError{Message: "ответ /chat/completions", RequestID: requestID, Err: err}
	}

	switch {
	case len(env.Choices) == 0:
		return llm.Result{}, env.reject("ответ без choices", requestID)
	case env.Choices[0].FinishReason == "error":
		return llm.Result{}, env.reject("генерация завершилась ошибкой (finish_reason=error)", requestID)
	case strings.TrimSpace(env.Choices[0].Message.Content) == "":
		return llm.Result{}, env.reject("пустой ответ (HTTP 200 без содержимого)", requestID)
	}

	return llm.Result{
		Model:     env.Model,
		Text:      env.Choices[0].Message.Content,
		Finish:    env.finish(),
		Usage:     env.usage(),
		RequestID: requestID,
	}, nil
}

// encode — тело запроса; attachments — id файлов по сообщениям в том же порядке.
func encode(req llm.Request, attachments [][]string) chatRequest {
	out := chatRequest{
		Model:       req.Model,
		Messages:    make([]chatMessage, 0, len(req.Messages)),
		Temperature: req.Options.Temperature,
		MaxTokens:   req.Options.MaxOutputTokens,
	}

	for i, m := range req.Messages {
		msg := chatMessage{Role: string(m.Role), Content: m.Text}
		if i < len(attachments) {
			msg.Attachments = attachments[i]
		}

		out.Messages = append(out.Messages, msg)
	}

	if req.Output.Mode == llm.ModeSchema {
		out.ResponseFormat = &responseFormat{Type: "json_schema", Schema: req.Output.Schema, Strict: req.Output.Strict}
	}

	return out
}

var _ llm.Provider = (*Provider)(nil)
