package gigachat

import (
	"context"
	"net/http"
	"time"

	"github.com/lllypuk/llm"
)

// Send — GET по пути API через do: вход и 401 без протокола chat.
func (p *Provider) Send(ctx context.Context, path string) (*http.Response, error) {
	return p.do(ctx, func(ctx context.Context) (*http.Request, error) {
		return http.NewRequestWithContext(ctx, http.MethodGet, p.api+path, nil)
	})
}

// SetClock подменяет часы кеша токенов; до первого вызова.
func (p *Provider) SetClock(now func() time.Time) { p.tokens.now = now }

// Waiters — сколько вызовов ждут текущего обновления токена.
func (p *Provider) Waiters() int {
	p.tokens.mu.Lock()
	defer p.tokens.mu.Unlock()

	if p.tokens.flight == nil {
		return 0
	}

	return p.tokens.flight.waiters
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
type TokenCache struct{ c *tokenCache }

// NewTokenCache — кеш поверх fetch, отдающего значение токена и срок.
func NewTokenCache(fetch func(ctx context.Context) (string, time.Time, error)) TokenCache {
	return TokenCache{c: newTokenCache(func(ctx context.Context) (*token, error) {
		value, expires, err := fetch(ctx)
		if err != nil {
			return nil, err
		}

		return &token{value: value, expires: expires}, nil
	})}
}

// Get — значение токена.
func (t TokenCache) Get(ctx context.Context) (string, error) {
	tok, err := t.c.get(ctx)
	if err != nil {
		return "", err
	}

	return tok.value, nil
}

// Flight — ждущие текущего обновления и канал его завершения; nil — обновления нет.
func (t TokenCache) Flight() (int, <-chan struct{}) {
	t.c.mu.Lock()
	defer t.c.mu.Unlock()

	if t.c.flight == nil {
		return 0, nil
	}

	return t.c.flight.waiters, t.c.flight.done
}
