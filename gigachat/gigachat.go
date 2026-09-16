// Package gigachat — плечо GigaChat: OAuth-токен с кешем на клиенте, корневой сертификат
// Минцифры в пуле транспорта, кадры через `/files` и `attachments`.
package gigachat

import (
	"context"
	"crypto/tls"
	"crypto/x509"
	"errors"
	"io"
	"net/http"
	"strings"

	"github.com/lllypuk/llm"
	"github.com/lllypuk/llm/internal/httpjson"
)

// Name — имя плеча в отчётах и журнале.
const Name = "gigachat"

// maxErrorBody — сколько байт тела читаем у не-2xx: сообщение укладывается в строку.
const maxErrorBody = 8 << 10

// Config — адреса, ключ авторизации и доверенные корни плеча.
type Config struct {
	OAuthEndpoint string
	// APIEndpoint — корень API до `/chat/completions` и `/files`.
	APIEndpoint string
	// AuthorizationKey — ключ из личного кабинета как есть, уже base64; в ошибки не попадает.
	AuthorizationKey string
	Scope            string
	// CA — корни целиком, системные не добавляются; пустой — корни транспорта.
	CA *x509.CertPool
	// HTTP — основа клиента; при CA транспорт обязан быть *http.Transport, он клонируется.
	HTTP *http.Client
}

// Provider — плечо GigaChat; один экземпляр на ключ, иначе у каждого свой токен.
type Provider struct {
	api    string
	http   *http.Client
	tokens *tokenCache
}

// New собирает плечо без сети: негодная конфигурация — отказ здесь, а не на первом вызове.
func New(cfg Config) (*Provider, error) {
	switch {
	case cfg.OAuthEndpoint == "":
		return nil, errors.New("gigachat: адрес OAuth не задан")
	case cfg.APIEndpoint == "":
		return nil, errors.New("gigachat: адрес API не задан")
	case cfg.AuthorizationKey == "":
		return nil, errors.New("gigachat: ключ авторизации не задан")
	case cfg.Scope == "":
		return nil, errors.New("gigachat: scope не задан")
	}

	client, err := trusting(cfg.HTTP, cfg.CA)
	if err != nil {
		return nil, err
	}

	o := &oauth{endpoint: cfg.OAuthEndpoint, key: cfg.AuthorizationKey, scope: cfg.Scope, http: client}

	return &Provider{
		api:    strings.TrimRight(cfg.APIEndpoint, "/"),
		http:   client,
		tokens: newTokenCache(o.fetch),
	}, nil
}

// Name — [Name].
func (p *Provider) Name() string { return Name }

// do шлёт запрос с токеном. 401 гасит использованный токен и повторяется один раз с новым:
// второй 401 — [*llm.StatusError] фазы auth, ключ чинит оператор.
func (p *Provider) do(
	ctx context.Context,
	build func(ctx context.Context) (*http.Request, error),
) (*http.Response, error) {
	for forced := false; ; forced = true {
		tok, err := p.tokens.get(ctx)
		if err != nil {
			return nil, err
		}

		req, err := build(ctx)
		if err != nil {
			return nil, &llm.RequestError{Message: "запрос gigachat", Err: err}
		}

		req.Header.Set("Authorization", "Bearer "+tok.value)

		resp, err := p.http.Do(req)
		if err != nil {
			return nil, err
		}

		if resp.StatusCode != http.StatusUnauthorized {
			return resp, nil
		}

		if forced {
			st := httpjson.ReadStatus(resp, maxErrorBody, errorMessage)
			_ = resp.Body.Close()

			return nil, &llm.StatusError{Status: st.Code, Message: st.Message, Phase: llm.PhaseAuth}
		}

		_, _ = io.Copy(io.Discard, io.LimitReader(resp.Body, maxErrorBody))
		_ = resp.Body.Close()

		p.tokens.invalidate(tok)
	}
}

// trusting — клиент с корнями CA на клоне транспорта; проверка TLS включается, даже если основа её выключила.
func trusting(base *http.Client, ca *x509.CertPool) (*http.Client, error) {
	if base == nil {
		base = http.DefaultClient
	}

	if ca == nil {
		return base, nil
	}

	rt := base.Transport
	if rt == nil {
		rt = http.DefaultTransport
	}

	tr, ok := rt.(*http.Transport)
	if !ok {
		return nil, errors.New("gigachat: CA задан, а транспорт не *http.Transport")
	}

	tr = tr.Clone()
	if tr.TLSClientConfig == nil {
		tr.TLSClientConfig = &tls.Config{MinVersion: tls.VersionTLS12}
	}

	tr.TLSClientConfig.RootCAs = ca
	tr.TLSClientConfig.InsecureSkipVerify = false

	client := *base
	client.Transport = tr

	return &client, nil
}
