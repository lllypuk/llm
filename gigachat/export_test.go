package gigachat

import (
	"context"
	"net/http"
	"time"
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
