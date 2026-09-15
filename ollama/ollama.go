// Package ollama — плечо Ollama поверх нативного `/api/chat`: кадры base64 в
// сообщении, `format` подсказкой или схемой, серверные длительности в ответе.
package ollama

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"

	"github.com/lllypuk/llm"
)

// Name — имя плеча в отчётах и журнале.
const Name = "ollama"

const (
	// maxErrorBody — сколько байт тела читаем у не-200: сообщение укладывается в строку.
	maxErrorBody = 8 << 10
	// maxBody — предел тела под кодом 200: демон или прокси произвольные, а текст уезжает дальше
	// без границы. Тело длиннее — отказ попытки, а не молча обрезанный ответ.
	maxBody = 512 << 10
)

// Provider — клиент демона. Поля открыты: хост из конфигурации, транспорт подменяем.
type Provider struct {
	Host string
	HTTP *http.Client
	// Think — оставить цепочку рассуждений; по умолчанию выключена: только цена и латентность.
	Think bool
}

// New собирает плечо по адресу демона.
func New(host string) *Provider {
	return &Provider{Host: strings.TrimRight(host, "/")}
}

// Name — [Name].
func (p *Provider) Name() string { return Name }

type message struct {
	Role    string   `json:"role"`
	Content string   `json:"content"`
	Images  [][]byte `json:"images,omitempty"` // encoding/json кладёт []byte как base64 — формат Ollama
}

type options struct {
	Temperature *float64 `json:"temperature,omitempty"`
}

type request struct {
	Model    string          `json:"model"`
	Messages []message       `json:"messages"`
	Stream   bool            `json:"stream"`
	Think    bool            `json:"think"`
	Format   json.RawMessage `json:"format,omitempty"`
	Options  *options        `json:"options,omitempty"`
}

// response — конверт `/api/chat`. Счётчики указателями: отсутствующий отличается от нуля.
type response struct {
	Model   string `json:"model"`
	Message struct {
		Content string `json:"content"`
	} `json:"message"`
	// Error — `{"error": "..."}` под кодом 200: так отвечают и демон, и прокси перед ним.
	Error           string `json:"error"`
	Done            bool   `json:"done"`
	DoneReason      string `json:"done_reason"`
	TotalDuration   int64  `json:"total_duration"`
	PromptEvalCount *int   `json:"prompt_eval_count"`
	EvalCount       *int   `json:"eval_count"`
}

func (r response) usage() llm.Usage {
	u := llm.Usage{Known: r.PromptEvalCount != nil && r.EvalCount != nil}
	if r.PromptEvalCount != nil {
		u.InputTokens = *r.PromptEvalCount
	}

	if r.EvalCount != nil {
		u.OutputTokens = *r.EvalCount
	}

	return u
}

// Complete — одна попытка; срок ставит вызывающий контекстом.
func (p *Provider) Complete(ctx context.Context, req llm.Request) (llm.Result, error) {
	if err := req.Validate(); err != nil {
		return llm.Result{}, err
	}

	body, err := json.Marshal(p.encode(req))
	if err != nil {
		return llm.Result{}, &llm.RequestError{Message: "сборка запроса", Err: err}
	}

	url := p.Host + "/api/chat"

	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewReader(body))
	if err != nil {
		return llm.Result{}, &llm.RequestError{Message: "запрос " + url, Err: err}
	}

	httpReq.Header.Set("Content-Type", "application/json")

	resp, err := p.client().Do(httpReq)
	if err != nil {
		return llm.Result{}, fmt.Errorf("запрос %s: %w", url, err)
	}

	defer func() { _ = resp.Body.Close() }()

	if resp.StatusCode != http.StatusOK {
		return llm.Result{}, &llm.StatusError{
			Status:     resp.StatusCode,
			Message:    errorMessage(resp.Body),
			RetryAfter: llm.RetryAfter(resp.Header),
		}
	}

	env, err := decode(resp.Body)
	if err != nil {
		return llm.Result{}, &llm.ResponseError{Message: "ответ " + url, Err: err}
	}

	if msg := strings.TrimSpace(env.Error); msg != "" {
		return llm.Result{}, &llm.ResponseError{Message: "ответ маршрута: " + msg, Usage: env.usage()}
	}

	switch {
	case !env.Done:
		return llm.Result{}, &llm.ResponseError{Message: "генерация не завершена (done=false)", Usage: env.usage()}
	case strings.TrimSpace(env.Message.Content) == "":
		return llm.Result{}, &llm.ResponseError{
			Message: "пустой ответ маршрута (HTTP 200 без содержимого)",
			Usage:   env.usage(),
		}
	}

	return llm.Result{
		Model:         env.Model,
		Text:          env.Message.Content,
		FinishReason:  env.DoneReason,
		Usage:         env.usage(),
		ServerLatency: time.Duration(env.TotalDuration),
	}, nil
}

// decode читает ровно один конверт: тело длиннее предела и хвост после первого
// объекта — отказ, иначе первый фрагмент потока прошёл бы за весь ответ.
func decode(r io.Reader) (response, error) {
	raw, err := io.ReadAll(io.LimitReader(r, maxBody+1))
	if err != nil {
		return response{}, err
	}

	if len(raw) > maxBody {
		return response{}, fmt.Errorf("тело длиннее %d байт", maxBody)
	}

	dec := json.NewDecoder(bytes.NewReader(raw))

	var env response
	if err = dec.Decode(&env); err != nil {
		return response{}, err
	}

	if dec.More() {
		return response{}, errors.New("после конверта есть ещё данные — похоже на поток")
	}

	return env, nil
}

func (p *Provider) encode(req llm.Request) request {
	out := request{Model: req.Model, Think: p.Think, Messages: make([]message, 0, len(req.Messages))}

	for _, m := range req.Messages {
		enc := message{Role: string(m.Role), Content: m.Text}
		for _, img := range m.Images {
			enc.Images = append(enc.Images, img.Data)
		}

		out.Messages = append(out.Messages, enc)
	}

	switch req.Output.Mode {
	case llm.ModeJSON:
		out.Format = json.RawMessage(`"json"`)
	case llm.ModeSchema:
		out.Format = req.Output.Schema
	case llm.ModeText:
	}

	if req.Temperature != nil {
		out.Options = &options{Temperature: req.Temperature}
	}

	return out
}

func (p *Provider) client() *http.Client {
	if p.HTTP != nil {
		return p.HTTP
	}

	return http.DefaultClient
}

// errorMessage достаёт текст из конверта `{"error": "..."}`; не-JSON отдаётся обрезанным как есть.
func errorMessage(r io.Reader) string {
	raw, err := io.ReadAll(io.LimitReader(r, maxErrorBody))
	if err != nil || len(raw) == 0 {
		return ""
	}

	var body struct {
		Error string `json:"error"`
	}

	if unmarshalErr := json.Unmarshal(raw, &body); unmarshalErr == nil && body.Error != "" {
		return body.Error
	}

	return strings.TrimSpace(string(raw))
}

var _ llm.Provider = (*Provider)(nil)
