package gigachat

import (
	"context"
	"crypto/rand"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"net/http"
	"net/url"
	"strings"
	"sync"
	"time"

	"github.com/lllypuk/llm"
	"github.com/lllypuk/llm/internal/httpjson"
)

const (
	// refreshMargin — токен обновляется заранее: попытка, взявшая его перед концом срока, иначе получит 401.
	refreshMargin = 5 * time.Minute
	// maxOAuthBody — предел ответа OAuth: токен и срок.
	maxOAuthBody = 64 << 10
	// millisSince — expires_at больше этого — миллисекунды, как в документации; меньше — секунды.
	millisSince = 100_000_000_000
)

// Биты версии и варианта UUID (RFC 9562).
const (
	lowNibble      = 0x0f
	uuidVersion4   = 0x40
	variantMask    = 0x3f
	variantRFC4122 = 0x80
)

// token — выданный токен; сравнивается указателем: гасится только тот экземпляр, что получил 401.
type token struct {
	value   string
	expires time.Time
}

// oauth — обмен ключа авторизации на токен.
type oauth struct {
	endpoint string
	key      string
	scope    string
	http     *http.Client
}

// fetch — один запрос токена; отказы помечены фазой auth.
func (o *oauth) fetch(ctx context.Context) (*token, error) {
	body := strings.NewReader(url.Values{"scope": {o.scope}}.Encode())

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, o.endpoint, body)
	if err != nil {
		return nil, &llm.PhaseError{Phase: llm.PhaseAuth, Err: &llm.RequestError{Message: "запрос OAuth", Err: err}}
	}

	req.Header.Set("Content-Type", "application/x-www-form-urlencoded")
	req.Header.Set("Accept", "application/json")
	req.Header.Set("Rquid", uuid4())
	req.Header.Set("Authorization", "Basic "+o.key)

	resp, err := o.http.Do(req)
	if err != nil {
		return nil, &llm.PhaseError{Phase: llm.PhaseAuth, Err: fmt.Errorf("запрос OAuth: %w", err)}
	}

	defer func() { _ = resp.Body.Close() }()

	requestID := resp.Header.Get("X-Request-Id")

	if resp.StatusCode != http.StatusOK {
		st := httpjson.ReadStatus(resp, maxErrorBody, errorMessage)

		return nil, &llm.StatusError{
			Status:     st.Code,
			Message:    st.Message,
			RetryAfter: st.RetryAfter,
			Phase:      llm.PhaseAuth,
			RequestID:  requestID,
		}
	}

	var env struct {
		AccessToken string `json:"access_token"`
		ExpiresAt   int64  `json:"expires_at"`
	}

	if err = httpjson.Decode(resp.Body, maxOAuthBody, &env); err != nil {
		return nil, &llm.PhaseError{
			Phase: llm.PhaseAuth,
			Err:   &llm.ResponseError{Message: "ответ OAuth", RequestID: requestID, Err: err},
		}
	}

	if env.AccessToken == "" || env.ExpiresAt <= 0 {
		return nil, &llm.PhaseError{
			Phase: llm.PhaseAuth,
			Err:   &llm.ResponseError{Message: "ответ OAuth без токена или срока", RequestID: requestID},
		}
	}

	expires := time.Unix(env.ExpiresAt, 0)
	if env.ExpiresAt > millisSince {
		expires = time.UnixMilli(env.ExpiresAt)
	}

	return &token{value: env.AccessToken, expires: expires}, nil
}

// flight — одно обновление на всех ждущих; ушли все ждущие — запрос отменяется.
type flight struct {
	done    chan struct{}
	cancel  context.CancelFunc
	waiters int
	tok     *token
	err     error
}

// tokenCache — токен до срока с запасом и слияние конкурентных обновлений.
type tokenCache struct {
	fetch func(ctx context.Context) (*token, error)
	now   func() time.Time

	mu     sync.Mutex
	cur    *token
	flight *flight
}

func newTokenCache(fetch func(ctx context.Context) (*token, error)) *tokenCache {
	return &tokenCache{fetch: fetch, now: time.Now}
}

// get — живой токен или результат общего обновления. Отмена ждущего не отменяет обновление,
// пока его ждёт кто-то ещё; токен из обновления отдаётся и с истекающим сроком.
func (c *tokenCache) get(ctx context.Context) (*token, error) {
	c.mu.Lock()

	if c.cur != nil && c.now().Add(refreshMargin).Before(c.cur.expires) {
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

// invalidate гасит токен, получивший 401, если он всё ещё текущий: поздний 401 старым токеном
// не трогает уже обновлённый.
func (c *tokenCache) invalidate(tok *token) {
	c.mu.Lock()
	defer c.mu.Unlock()

	if c.cur == tok {
		c.cur = nil
	}
}

func (c *tokenCache) refresh(ctx context.Context, f *flight) {
	tok, err := c.fetch(ctx)
	f.cancel()

	c.mu.Lock()

	// Брошенное обновление кеш не трогает: его токен мог прийти позже токена следующего.
	current := c.flight == f

	switch {
	case !current:
	case err == nil:
		c.cur = tok
	case c.cur != nil && c.now().Before(c.cur.expires):
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

func (c *tokenCache) leave(f *flight) {
	c.mu.Lock()
	defer c.mu.Unlock()

	f.waiters--
	if f.waiters == 0 && c.flight == f {
		c.flight = nil
		f.cancel()
	}
}

// uuid4 — RqUID: случайный UUID версии 4.
func uuid4() string {
	var b [16]byte
	_, _ = rand.Read(b[:])

	b[6] = b[6]&lowNibble | uuidVersion4
	b[8] = b[8]&variantMask | variantRFC4122

	h := hex.EncodeToString(b[:])

	return h[:8] + "-" + h[8:12] + "-" + h[12:16] + "-" + h[16:20] + "-" + h[20:]
}

// errorMessage — текст конверта `{"message": "..."}`.
func errorMessage(raw []byte) string {
	var body struct {
		Message string `json:"message"`
	}

	_ = json.Unmarshal(raw, &body)

	return body.Message
}
