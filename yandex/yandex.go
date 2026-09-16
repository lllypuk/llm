// Package yandex — плечо Yandex AI Studio через OpenAI-совместимый `chat/completions`: ключ API
// или IAM-токен, каталог заголовком `OpenAI-Project`, кадры data-URI в частях сообщения.
package yandex

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"strings"

	"github.com/lllypuk/llm"
	"github.com/lllypuk/llm/internal/httpjson"
)

// Name — имя плеча в отчётах и журнале.
const Name = "yandex"

const (
	// maxChatBody — предел ответа; длиннее — отказ попытки, а не обрезанный ответ.
	maxChatBody = 1 << 20
	// maxErrorBody — сколько байт тела читаем у не-2xx.
	maxErrorBody = 8 << 10
)

// Config — адрес, каталог и подпись плеча.
type Config struct {
	// Endpoint — корень OpenAI-совместимого API до `/chat/completions`.
	Endpoint string
	// Folder — каталог: едет в `OpenAI-Project` и в адрес модели `gpt://folder/model`.
	Folder      string
	Credentials CredentialSource
	HTTP        *http.Client
}

// Provider — плечо Yandex AI Studio; состояния нет, токен спрашивается у Credentials каждой попыткой.
type Provider struct {
	endpoint string
	folder   string
	creds    CredentialSource
	http     *http.Client
}

// New собирает плечо без сети.
func New(cfg Config) (*Provider, error) {
	switch {
	case cfg.Endpoint == "":
		return nil, errors.New("yandex: адрес API не задан")
	case cfg.Folder == "":
		return nil, errors.New("yandex: каталог не задан")
	case cfg.Credentials == nil:
		return nil, errors.New("yandex: подпись не задана")
	}

	client := cfg.HTTP
	if client == nil {
		client = http.DefaultClient
	}

	return &Provider{
		endpoint: strings.TrimRight(cfg.Endpoint, "/"),
		folder:   cfg.Folder,
		creds:    cfg.Credentials,
		http:     client,
	}, nil
}

// Name — [Name].
func (p *Provider) Name() string { return Name }

// Capabilities — профиль по имени модели без версии (`yandexgpt/rc` — `yandexgpt`). Рассуждения — только
// у открытых reasoning-моделей, кадры — только у gemma; неизвестная модель не подтверждает ничего.
func (p *Provider) Capabilities(model string) (llm.Capabilities, bool) {
	name, _, _ := strings.Cut(model, "/")

	caps := llm.Capabilities{
		JSON:            true,
		Schema:          true,
		Strict:          true,
		Temperature:     true,
		MaxOutputTokens: true,
	}

	switch name {
	case "yandexgpt", "yandexgpt-lite":
	case "gpt-oss-120b", "gpt-oss-20b", "qwen3-235b-a22b-fp8":
		caps.Reasoning = true
	case "gemma-3-27b-it":
		caps.Vision = true
	default:
		return llm.Capabilities{}, false
	}

	return caps, true
}

type imageURL struct {
	URL string `json:"url"`
}

type contentPart struct {
	Type     string    `json:"type"`
	Text     string    `json:"text,omitempty"`
	ImageURL *imageURL `json:"image_url,omitempty"`
}

// chatMessage — content строкой без кадров и списком частей с кадрами.
type chatMessage struct {
	Role    string `json:"role"`
	Content any    `json:"content"`
}

type jsonSchema struct {
	Name   string          `json:"name"`
	Schema json.RawMessage `json:"schema"`
	Strict bool            `json:"strict"`
}

type responseFormat struct {
	Type       string      `json:"type"`
	JSONSchema *jsonSchema `json:"json_schema,omitempty"`
}

type chatRequest struct {
	Model           string          `json:"model"`
	Messages        []chatMessage   `json:"messages"`
	Stream          bool            `json:"stream"`
	Temperature     *float64        `json:"temperature,omitempty"`
	MaxTokens       int             `json:"max_tokens,omitempty"`
	ReasoningEffort string          `json:"reasoning_effort,omitempty"`
	ResponseFormat  *responseFormat `json:"response_format,omitempty"`
}

type tokenDetails struct {
	CachedTokens    *int `json:"cached_tokens"`
	ReasoningTokens *int `json:"reasoning_tokens"`
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
		PromptTokens            *int          `json:"prompt_tokens"`
		CompletionTokens        *int          `json:"completion_tokens"`
		TotalTokens             *int          `json:"total_tokens"`
		PromptTokensDetails     *tokenDetails `json:"prompt_tokens_details"`
		CompletionTokensDetails *tokenDetails `json:"completion_tokens_details"`
	} `json:"usage"`
}

// usage — расход из конверта. Кеш и рассуждения уже входят в prompt_tokens и completion_tokens: они
// вычитаются из общих, а не прибавляются; не названная деталь — ноль, часть больше общего — расход неполный.
func (r chatResponse) usage() llm.Usage {
	if r.Usage == nil {
		return llm.Usage{}
	}

	u := llm.Usage{Raw: map[string]int{}, Known: true}

	var prompt, completion, cached, reasoning int

	prompt, u.Known = counter(u.Raw, "prompt_tokens", r.Usage.PromptTokens, u.Known)
	completion, u.Known = counter(u.Raw, "completion_tokens", r.Usage.CompletionTokens, u.Known)

	if d := r.Usage.PromptTokensDetails; d != nil && d.CachedTokens != nil {
		cached, u.Known = counter(u.Raw, "prompt_tokens_details.cached_tokens", d.CachedTokens, u.Known)
	}

	if d := r.Usage.CompletionTokensDetails; d != nil && d.ReasoningTokens != nil {
		reasoning, u.Known = counter(u.Raw, "completion_tokens_details.reasoning_tokens", d.ReasoningTokens, u.Known)
	}

	if r.Usage.TotalTokens != nil {
		u.Raw["total_tokens"] = *r.Usage.TotalTokens
	}

	u.BillableInput, u.CachedInput, u.Known = split(prompt, cached, u.Known)
	u.Output, u.Reasoning, u.Known = split(completion, reasoning, u.Known)

	return u
}

// split делит общий счётчик на остаток и часть; часть больше общего остаётся частью, а расход — неполным.
func split(total, part int, known bool) (int, int, bool) {
	if part > total {
		return 0, part, false
	}

	return total - part, part, known
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

// finish — finish_reason первого варианта.
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
	case "content_filter":
		f.Kind = llm.FinishContentFilter
	case "tool_calls", "function_call":
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

// Complete — одна попытка. Профиль сверяется и здесь: мимо клиента неподтверждённые
// reasoning_effort и max_tokens ушли бы в модель, которая их молча игнорирует.
func (p *Provider) Complete(ctx context.Context, req llm.Request) (llm.Result, error) {
	if err := req.Validate(); err != nil {
		return llm.Result{}, err
	}

	caps, _ := p.Capabilities(req.Model)
	if err := caps.Check(req); err != nil {
		return llm.Result{}, err
	}

	body, err := p.encode(req)
	if err != nil {
		return llm.Result{}, err
	}

	scheme, value, err := p.creds.Token(ctx)
	if err != nil {
		return llm.Result{}, &llm.PhaseError{Phase: llm.PhaseAuth, Err: err}
	}

	header := http.Header{
		"Authorization":  {scheme + " " + value},
		"OpenAI-Project": {p.folder},
	}

	httpReq, err := httpjson.NewRequest(ctx, p.endpoint+"/chat/completions", body, header)
	if err != nil {
		return llm.Result{}, &llm.RequestError{Message: "запрос yandex", Err: err}
	}

	resp, err := p.http.Do(httpReq)
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

// encode — тело запроса. Имя схемы протокол требует, а умолчание спрятало бы его отсутствие в конфиге.
func (p *Provider) encode(req llm.Request) (chatRequest, error) {
	if strings.Contains(req.Model, "://") {
		return chatRequest{}, &llm.RequestError{Message: "модель задаётся без gpt:// и каталога: " + req.Model}
	}

	out := chatRequest{
		Model:           "gpt://" + p.folder + "/" + req.Model,
		Messages:        make([]chatMessage, 0, len(req.Messages)),
		Temperature:     req.Options.Temperature,
		MaxTokens:       req.Options.MaxOutputTokens,
		ReasoningEffort: string(req.Options.Reasoning),
	}

	for _, m := range req.Messages {
		msg, err := message(m)
		if err != nil {
			return chatRequest{}, err
		}

		out.Messages = append(out.Messages, msg)
	}

	switch req.Output.Mode {
	case llm.ModeJSON:
		out.ResponseFormat = &responseFormat{Type: "json_object"}
	case llm.ModeSchema:
		if req.Output.Name == "" {
			return chatRequest{}, &llm.RequestError{Message: "режим schema без имени схемы"}
		}

		out.ResponseFormat = &responseFormat{
			Type:       "json_schema",
			JSONSchema: &jsonSchema{Name: req.Output.Name, Schema: req.Output.Schema, Strict: req.Output.Strict},
		}
	case "", llm.ModeText:
	}

	return out, nil
}

// message — сообщение без кадров строкой, с кадрами — текст и data-URI частями.
func message(m llm.Message) (chatMessage, error) {
	if len(m.Images) == 0 {
		return chatMessage{Role: string(m.Role), Content: m.Text}, nil
	}

	parts := make([]contentPart, 0, len(m.Images)+1)
	if m.Text != "" {
		parts = append(parts, contentPart{Type: "text", Text: m.Text})
	}

	for _, img := range m.Images {
		if !strings.HasPrefix(img.MIME, "image/") || len(img.Data) == 0 {
			return chatMessage{}, &llm.RequestError{Message: "кадр не изображение или пуст: " + img.MIME}
		}

		uri := "data:" + img.MIME + ";base64," + base64.StdEncoding.EncodeToString(img.Data)
		parts = append(parts, contentPart{Type: "image_url", ImageURL: &imageURL{URL: uri}})
	}

	return chatMessage{Role: string(m.Role), Content: parts}, nil
}

// errorMessage — текст конверта: OpenAI-совместимый `{"error": {"message"}}` или плоский `{"message"}`.
func errorMessage(raw []byte) string {
	var body struct {
		Message string `json:"message"`
		Error   struct {
			Message string `json:"message"`
		} `json:"error"`
	}

	_ = json.Unmarshal(raw, &body)

	if body.Error.Message != "" {
		return body.Error.Message
	}

	return body.Message
}

var _ llm.Provider = (*Provider)(nil)
