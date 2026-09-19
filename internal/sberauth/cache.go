package sberauth

import (
	"context"
	"sync"
	"time"

	"github.com/lllypuk/llm"
)

// flight — одно обновление на всех ждущих; ушли все ждущие — запрос отменяется.
type flight struct {
	done    chan struct{}
	cancel  context.CancelFunc
	waiters int
	tok     *Token
	err     error
}

// Cache — токен до срока с запасом и слияние конкурентных обновлений.
type Cache struct {
	fetch func(ctx context.Context) (*Token, error)
	now   func() time.Time

	mu     sync.Mutex
	cur    *Token
	flight *flight
}

// NewCache — кеш поверх одного запроса токена.
func NewCache(fetch func(ctx context.Context) (*Token, error)) *Cache {
	return &Cache{fetch: fetch, now: time.Now}
}

// Get — живой токен или результат общего обновления. Отмена ждущего не отменяет обновление,
// пока его ждёт кто-то ещё; токен из обновления отдаётся и с истекающим сроком.
func (c *Cache) Get(ctx context.Context) (*Token, error) {
	c.mu.Lock()

	if c.cur != nil && c.now().Add(refreshMargin).Before(c.cur.Expires) {
		tok := c.cur
		c.mu.Unlock()

		return tok, nil
	}

	f := c.flight
	if f == nil {
		// Обновление живёт дольше вызвавшего: его срок — сроки ждущих, отмену делает leave.
		fctx, cancel := context.WithCancel(context.WithoutCancel(ctx))
		f = &flight{done: make(chan struct{}), cancel: cancel}
		c.flight = f

		go c.refresh(fctx, f)
	}

	f.waiters++
	c.mu.Unlock()

	select {
	case <-f.done:
		return f.tok, f.err
	case <-ctx.Done():
		c.leave(f)

		return nil, &llm.PhaseError{Phase: llm.PhaseAuth, Err: ctx.Err()}
	}
}

// Invalidate гасит токен, получивший 401, если он всё ещё текущий: поздний 401 старым токеном
// не трогает уже обновлённый.
func (c *Cache) Invalidate(tok *Token) {
	c.mu.Lock()
	defer c.mu.Unlock()

	if c.cur == tok {
		c.cur = nil
	}
}

func (c *Cache) refresh(ctx context.Context, f *flight) {
	tok, err := c.fetch(ctx)
	f.cancel()

	c.mu.Lock()

	// Брошенное обновление кеш не трогает: его токен мог прийти позже токена следующего.
	current := c.flight == f

	switch {
	case !current:
	case err == nil:
		c.cur = tok
	case c.cur != nil && c.now().Before(c.cur.Expires):
		// Отказ раннего обновления не гасит ещё живой токен: он доживает свой срок.
		tok, err = c.cur, nil
	}

	if current {
		c.flight = nil
	}

	f.tok, f.err = tok, err
	c.mu.Unlock()

	close(f.done)
}

func (c *Cache) leave(f *flight) {
	c.mu.Lock()
	defer c.mu.Unlock()

	f.waiters--
	if f.waiters == 0 && c.flight == f {
		c.flight = nil
		f.cancel()
	}
}

// SetClock подменяет часы; до первого вызова.
func (c *Cache) SetClock(now func() time.Time) { c.now = now }

// Flight — ждущие текущего обновления и канал его завершения; nil — обновления нет.
func (c *Cache) Flight() (int, <-chan struct{}) {
	c.mu.Lock()
	defer c.mu.Unlock()

	if c.flight == nil {
		return 0, nil
	}

	return c.flight.waiters, c.flight.done
}
