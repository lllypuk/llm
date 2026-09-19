// Package sberauth — OAuth Сбера, общий у GigaChat и SaluteSpeech: токен на экземпляр плеча
// и его пару (ключ, scope), корни Минцифры в транспорте, один повтор на 401.
package sberauth

import (
	"context"
	"crypto/tls"
	"crypto/x509"
	"errors"
	"io"
	"net/http"

	"github.com/lllypuk/llm"
	"github.com/lllypuk/llm/internal/httpjson"
)

// Config — адрес OAuth, ключ, scope и доверенные корни одного плеча.
type Config struct {
	// Name — префикс ошибок и сообщений: имя плеча.
	Name     string
	Endpoint string
	// Key — ключ авторизации как есть, уже base64; в ошибки не попадает.
	Key   string
	Scope string
	// CA — корни целиком, системные не добавляются; пустой — корни транспорта.
	CA *x509.CertPool
	// HTTP — основа клиента; при CA транспорт обязан быть *http.Transport, он клонируется.
	HTTP *http.Client
}

// Source — клиент с корнями и кеш токена одного плеча; делить его между плечами нельзя.
type Source struct {
	name   string
	http   *http.Client
	tokens *Cache
}

// New собирает источник без сети.
func New(cfg Config) (*Source, error) {
	client, err := trusting(cfg.Name, cfg.HTTP, cfg.CA)
	if err != nil {
		return nil, err
	}

	o := &oauth{endpoint: cfg.Endpoint, key: cfg.Key, scope: cfg.Scope, http: client}

	return &Source{name: cfg.Name, http: client, tokens: NewCache(o.fetch)}, nil
}

// HTTP — клиент с корнями CA.
func (s *Source) HTTP() *http.Client { return s.http }

// Tokens — кеш токена; наружу — только ради подмены часов в тестах.
func (s *Source) Tokens() *Cache { return s.tokens }

// Do шлёт запрос с токеном. 401 гасит использованный токен и повторяется один раз с новым:
// второй 401 — [*llm.StatusError] фазы auth, ключ чинит оператор.
func (s *Source) Do(
	ctx context.Context,
	build func(ctx context.Context) (*http.Request, error),
) (*http.Response, error) {
	for forced := false; ; forced = true {
		tok, err := s.tokens.Get(ctx)
		if err != nil {
			return nil, err
		}

		req, err := build(ctx)
		if err != nil {
			return nil, &llm.RequestError{Message: "запрос " + s.name, Err: err}
		}

		req.Header.Set("Authorization", "Bearer "+tok.Value)

		resp, err := s.http.Do(req)
		if err != nil {
			return nil, err
		}

		if resp.StatusCode != http.StatusUnauthorized {
			return resp, nil
		}

		if forced {
			st := httpjson.ReadStatus(resp, MaxErrorBody, ErrorMessage)
			_ = resp.Body.Close()

			return nil, &llm.StatusError{
				Status:     st.Code,
				Message:    st.Message,
				RetryAfter: st.RetryAfter,
				Phase:      llm.PhaseAuth,
				RequestID:  resp.Header.Get("X-Request-Id"),
			}
		}

		_, _ = io.Copy(io.Discard, io.LimitReader(resp.Body, MaxErrorBody))
		_ = resp.Body.Close()

		s.tokens.Invalidate(tok)
	}
}

// trusting — клиент с корнями CA на клоне транспорта; проверка TLS включается, даже если основа её выключила.
func trusting(name string, base *http.Client, ca *x509.CertPool) (*http.Client, error) {
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
		return nil, errors.New(name + ": CA задан, а транспорт не *http.Transport")
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
