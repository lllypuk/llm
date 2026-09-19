package gigachat

import (
	"context"
	"net/http"
	"time"

	"github.com/lllypuk/llm"
	"github.com/lllypuk/llm/internal/sberauth"
)

// Send — GET по пути API через do: вход и 401 без протокола chat.
func (p *Provider) Send(ctx context.Context, path string) (*http.Response, error) {
	return p.do(ctx, func(ctx context.Context) (*http.Request, error) {
		return http.NewRequestWithContext(ctx, http.MethodGet, p.api+path, nil)
	})
}

// SetClock подменяет часы кеша токенов; до первого вызова.
func (p *Provider) SetClock(now func() time.Time) { p.auth.Tokens().SetClock(now) }

// Waiters — сколько вызовов ждут текущего обновления токена.
func (p *Provider) Waiters() int {
	n, _ := p.auth.Tokens().Flight()

	return n
}

// WithFiles — загрузка кадров, send и уборка без протокола chat.
func (p *Provider) WithFiles(
	ctx context.Context,
	msgs []llm.Message,
	send func(ctx context.Context, attachments [][]string) (llm.Result, error),
) (llm.Result, error) {
	return p.withFiles(ctx, msgs, send)
}

// TokenCache — кеш токенов с подменным запросом OAuth.
type TokenCache struct{ c *sberauth.Cache }

// NewTokenCache — кеш поверх fetch, отдающего значение токена и срок.
func NewTokenCache(fetch func(ctx context.Context) (string, time.Time, error)) TokenCache {
	return TokenCache{c: sberauth.NewCache(func(ctx context.Context) (*sberauth.Token, error) {
		value, expires, err := fetch(ctx)
		if err != nil {
			return nil, err
		}

		return &sberauth.Token{Value: value, Expires: expires}, nil
	})}
}

// Get — значение токена.
func (t TokenCache) Get(ctx context.Context) (string, error) {
	tok, err := t.c.Get(ctx)
	if err != nil {
		return "", err
	}

	return tok.Value, nil
}

// Flight — ждущие текущего обновления и канал его завершения; nil — обновления нет.
func (t TokenCache) Flight() (int, <-chan struct{}) { return t.c.Flight() }
