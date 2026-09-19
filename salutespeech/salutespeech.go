// Package salutespeech — синхронное распознавание SaluteSpeech: OAuth Сбера со своим ключом и scope,
// корни Минцифры — как у GigaChat.
package salutespeech

import (
	"bytes"
	"context"
	"crypto/x509"
	"errors"
	"fmt"
	"net/http"
	"net/url"
	"strconv"
	"strings"
	"time"

	"github.com/lllypuk/llm"
	"github.com/lllypuk/llm/internal/httpjson"
	"github.com/lllypuk/llm/internal/sberauth"
)

// Name — имя плеча в отчётах и журнале.
const Name = "salutespeech"

const (
	// maxAudio — предел синхронного `speech:recognize` у любой модели; 2 МБ при 16 кГц он не превышает.
	maxAudio      = time.Minute
	maxResultBody = 256 << 10
)

// Config — адреса, ключ авторизации SaluteSpeech и доверенные корни.
type Config struct {
	OAuthEndpoint string
	// APIEndpoint — корень REST v1 до `/speech:recognize`.
	APIEndpoint string
	// AuthorizationKey — ключ проекта SaluteSpeech, не GigaChat; в ошибки не попадает.
	AuthorizationKey string
	Scope            string
	CA               *x509.CertPool
	HTTP             *http.Client
}

// Provider — плечо SaluteSpeech; один экземпляр на ключ, иначе у каждого свой токен.
type Provider struct {
	api  string
	auth *sberauth.Source
}

// New собирает плечо без сети.
func New(cfg Config) (*Provider, error) {
	switch {
	case cfg.OAuthEndpoint == "":
		return nil, errors.New("salutespeech: адрес OAuth не задан")
	case cfg.APIEndpoint == "":
		return nil, errors.New("salutespeech: адрес API не задан")
	case cfg.AuthorizationKey == "":
		return nil, errors.New("salutespeech: ключ авторизации не задан")
	case cfg.Scope == "":
		return nil, errors.New("salutespeech: scope не задан")
	}

	auth, err := sberauth.New(sberauth.Config{
		Name:     Name,
		Endpoint: cfg.OAuthEndpoint,
		Key:      cfg.AuthorizationKey,
		Scope:    cfg.Scope,
		CA:       cfg.CA,
		HTTP:     cfg.HTTP,
	})
	if err != nil {
		return nil, err
	}

	return &Provider{api: strings.TrimRight(cfg.APIEndpoint, "/"), auth: auth}, nil
}

// Name — [Name].
func (p *Provider) Name() string { return Name }

// SpeechCapabilities — неизвестная модель не подтверждается.
func (p *Provider) SpeechCapabilities(model string) (llm.SpeechCapabilities, bool) {
	switch model {
	case "general", "callcenter", "media", "ivr":
		return llm.SpeechCapabilities{MaxAudio: maxAudio}, true
	}

	return llm.SpeechCapabilities{}, false
}

// Transcribe — одна попытка; гипотезы по фразам склеиваются пробелом.
func (p *Provider) Transcribe(ctx context.Context, req llm.SpeechRequest) (llm.Transcript, error) {
	if err := req.Validate(maxAudio); err != nil {
		return llm.Transcript{}, err
	}

	query := url.Values{"model": {req.Model}}
	if req.Language != "" {
		query.Set("language", req.Language)
	}

	endpoint := p.api + "/speech:recognize?" + query.Encode()
	contentType := "audio/x-pcm;bit=16;rate=" + strconv.Itoa(req.SampleRate)

	resp, err := p.auth.Do(ctx, func(ctx context.Context) (*http.Request, error) {
		r, reqErr := http.NewRequestWithContext(ctx, http.MethodPost, endpoint, bytes.NewReader(req.PCM))
		if reqErr != nil {
			return nil, reqErr
		}

		r.Header.Set("Content-Type", contentType)
		r.Header.Set("Accept", "application/json")

		return r, nil
	})
	if err != nil {
		return llm.Transcript{}, fmt.Errorf("запрос /speech:recognize: %w", err)
	}

	defer func() { _ = resp.Body.Close() }()

	requestID := resp.Header.Get("X-Request-Id")

	if resp.StatusCode != http.StatusOK {
		st := httpjson.ReadStatus(resp, sberauth.MaxErrorBody, sberauth.ErrorMessage)

		return llm.Transcript{}, &llm.StatusError{
			Status:     st.Code,
			Message:    st.Message,
			RetryAfter: st.RetryAfter,
			RequestID:  requestID,
		}
	}

	var env struct {
		Result []string `json:"result"`
	}
	if err = httpjson.Decode(resp.Body, maxResultBody, &env); err != nil {
		return llm.Transcript{}, &llm.ResponseError{Message: "ответ /speech:recognize", RequestID: requestID, Err: err}
	}

	parts := make([]string, 0, len(env.Result))

	for _, r := range env.Result {
		if r = strings.TrimSpace(r); r != "" {
			parts = append(parts, r)
		}
	}

	return llm.Transcript{
		Text:        strings.Join(parts, " "),
		Model:       req.Model,
		RequestID:   requestID,
		AudioMillis: req.Duration().Milliseconds(),
	}, nil
}

var _ llm.Transcriber = (*Provider)(nil)
