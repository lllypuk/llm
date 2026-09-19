// Package gigachat — плечо GigaChat: OAuth-токен с кешем на клиенте, корневой сертификат
// Минцифры в пуле транспорта, кадры через `/files` и `attachments`.
package gigachat

import (
	"context"
	"crypto/x509"
	"errors"
	"net/http"
	"strings"

	"github.com/lllypuk/llm/internal/sberauth"
)

// Name — имя плеча в отчётах и журнале.
const Name = "gigachat"

// maxErrorBody — сколько байт тела читаем у не-2xx.
const maxErrorBody = sberauth.MaxErrorBody

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
	api  string
	http *http.Client
	auth *sberauth.Source
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

	return &Provider{api: strings.TrimRight(cfg.APIEndpoint, "/"), http: auth.HTTP(), auth: auth}, nil
}

// Name — [Name].
func (p *Provider) Name() string { return Name }

// do — запрос с токеном и одним повтором на 401 ([sberauth.Source.Do]).
func (p *Provider) do(
	ctx context.Context,
	build func(ctx context.Context) (*http.Request, error),
) (*http.Response, error) {
	return p.auth.Do(ctx, build)
}
