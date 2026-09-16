package yandex

import (
	"context"
	"errors"

	"github.com/lllypuk/llm"
)

// Схемы заголовка Authorization.
const (
	SchemeAPIKey = "Api-Key"
	SchemeBearer = "Bearer"
)

// CredentialSource — чем подписать попытку; спрашивается перед каждой, поэтому обновлённый снаружи
// IAM-токен подхватывается следующей попыткой, а не следующим процессом.
type CredentialSource interface {
	Token(ctx context.Context) (scheme, value string, err error)
}

// APIKey — статичный ключ сервисного аккаунта.
func APIKey(key string) CredentialSource { return apiKey(key) }

type apiKey string

func (k apiKey) Token(context.Context) (string, string, error) {
	if k == "" {
		return "", "", &llm.ConfigError{Message: "ключ API пуст"}
	}

	return SchemeAPIKey, string(k), nil
}

// IAMToken — IAM-токен, который выпускает и обновляет вызывающий: адаптер его не кеширует.
// Постоянный отказ (отозванный аккаунт) источник возвращает [llm.ConfigError], иначе его повторяют.
type IAMToken func(ctx context.Context) (string, error)

// Token — [SchemeBearer] с текущим токеном; пустой — отказ.
func (f IAMToken) Token(ctx context.Context) (string, string, error) {
	tok, err := f(ctx)
	if err != nil {
		return "", "", err
	}

	if tok == "" {
		return "", "", errors.New("IAM-токен пуст")
	}

	return SchemeBearer, tok, nil
}
