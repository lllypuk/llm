package sberauth

import (
	"context"
	"crypto/rand"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"net/http"
	"net/url"
	"strings"
	"time"

	"github.com/lllypuk/llm"
	"github.com/lllypuk/llm/internal/httpjson"
)

const (
	// MaxErrorBody — сколько байт тела читаем у не-2xx: сообщение укладывается в строку.
	MaxErrorBody = 8 << 10
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

// Token — выданный токен; сравнивается указателем: гасится только тот экземпляр, что получил 401.
type Token struct {
	Value   string
	Expires time.Time
}

// oauth — обмен ключа авторизации на токен одного scope.
type oauth struct {
	endpoint string
	key      string
	scope    string
	http     *http.Client
}

// fetch — один запрос токена; отказы помечены фазой auth.
func (o *oauth) fetch(ctx context.Context) (*Token, error) {
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
		st := httpjson.ReadStatus(resp, MaxErrorBody, ErrorMessage)

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

	return &Token{Value: env.AccessToken, Expires: expires}, nil
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

// ErrorMessage — текст конверта Сбера `{"message": "..."}`.
func ErrorMessage(raw []byte) string {
	var body struct {
		Message string `json:"message"`
	}

	_ = json.Unmarshal(raw, &body)

	return body.Message
}
