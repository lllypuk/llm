// Package llmbuild собирает плечи и [llm.Router] из [llmconfig.Config]: вид плеча — конструктор
// адаптера. Транспорт, обёртки и наблюдатель остаются у потребителя.
package llmbuild

import (
	"crypto/x509"
	"errors"
	"fmt"
	"net/http"
	"os"

	"github.com/lllypuk/llm"
	"github.com/lllypuk/llm/gigachat"
	"github.com/lllypuk/llm/internal/trust"
	"github.com/lllypuk/llm/llmconfig"
	"github.com/lllypuk/llm/ollama"
	"github.com/lllypuk/llm/salutespeech"
	"github.com/lllypuk/llm/yandex"
)

// Options — корни и основа транспорта плеч.
type Options struct {
	// PEM — корни по имени плеча целиком, без системных; пустые байты — корни системы.
	// Нет ключа — читается Provider.CAFile.
	PEM map[string][]byte
	// HTTP — основа клиента плеча по имени и виду; nil — &http.Client{}. При корнях её транспорт
	// обязан быть *http.Transport: он клонируется.
	HTTP func(name, kind string) *http.Client
}

// Built — собранное плечо: ровно одно поле не nil, по виду.
type Built struct {
	Provider llm.Provider
	Speech   llm.Transcriber
	OCR      llm.Recognizer
}

// Arm собирает плечо name без сети; неразвёрнутый секрет p — отказ.
func Arm(name string, p llmconfig.Provider, o Options) (Built, error) {
	if err := expanded(p); err != nil {
		return Built{}, err
	}

	pool, err := roots(name, p, o)
	if err != nil {
		return Built{}, err
	}

	base := &http.Client{}
	if o.HTTP != nil {
		base = o.HTTP(name, p.Kind)
	}

	if p.Kind == llmconfig.KindGigaChat || p.Kind == llmconfig.KindSaluteSpeech {
		return sber(p, pool, base)
	}

	client, err := trust.Client(base, pool)
	if err != nil {
		return Built{}, err
	}

	switch p.Kind {
	case llmconfig.KindOllama:
		prov := ollama.New(p.Endpoint)
		prov.HTTP = client

		return Built{Provider: prov}, nil
	case llmconfig.KindYandex:
		prov, buildErr := yandex.New(yandex.Config{
			Endpoint:    p.Endpoint,
			Folder:      p.Folder,
			Credentials: yandex.APIKey(p.Auth.APIKey.Value()),
			HTTP:        client,
		})
		if buildErr != nil {
			return Built{}, buildErr
		}

		return Built{Provider: prov}, nil
	case llmconfig.KindSpeechKit:
		s, buildErr := yandex.NewSpeech(yandex.SpeechConfig{
			Endpoint:    p.Endpoint,
			Folder:      p.Folder,
			Credentials: yandex.APIKey(p.Auth.APIKey.Value()),
			HTTP:        client,
		})
		if buildErr != nil {
			return Built{}, buildErr
		}

		return Built{Speech: s}, nil
	case llmconfig.KindVisionOCR:
		rec, buildErr := yandex.NewOCR(yandex.OCRConfig{
			Endpoint:    p.Endpoint,
			Folder:      p.Folder,
			Credentials: yandex.APIKey(p.Auth.APIKey.Value()),
			HTTP:        client,
		})
		if buildErr != nil {
			return Built{}, buildErr
		}

		return Built{OCR: rec}, nil
	default:
		return Built{}, fmt.Errorf("неизвестный вид плеча %q", p.Kind)
	}
}

// expanded — ключ плеча развёрнут: пустой ключ Яндекса адаптер принимает и отказывает лишь на вызове.
func expanded(p llmconfig.Provider) error {
	var (
		field string
		key   llmconfig.Secret
	)

	switch p.Kind {
	case llmconfig.KindGigaChat, llmconfig.KindSaluteSpeech:
		field, key = "auth.authorization_key", p.Auth.AuthorizationKey
	case llmconfig.KindYandex, llmconfig.KindSpeechKit, llmconfig.KindVisionOCR:
		field, key = "auth.api_key", p.Auth.APIKey
	default:
		return nil
	}

	switch {
	case key.Value() != "":
		return nil
	case key.IsZero():
		return fmt.Errorf("%s: ключ не задан у плеча %s", field, p.Kind)
	default:
		return fmt.Errorf("%s: %s не развёрнут у плеча %s — конфиг без Expand", field, key, p.Kind)
	}
}

// sber — плечи Сбера: корни в транспорт кладёт их OAuth, основа уходит как есть.
func sber(p llmconfig.Provider, pool *x509.CertPool, base *http.Client) (Built, error) {
	if p.Kind == llmconfig.KindSaluteSpeech {
		s, err := salutespeech.New(salutespeech.Config{
			OAuthEndpoint:    p.OAuthEndpoint,
			APIEndpoint:      p.Endpoint,
			AuthorizationKey: p.Auth.AuthorizationKey.Value(),
			Scope:            p.Scope,
			CA:               pool,
			HTTP:             base,
		})
		if err != nil {
			return Built{}, err
		}

		return Built{Speech: s}, nil
	}

	prov, err := gigachat.New(gigachat.Config{
		OAuthEndpoint:    p.OAuthEndpoint,
		APIEndpoint:      p.Endpoint,
		AuthorizationKey: p.Auth.AuthorizationKey.Value(),
		Scope:            p.Scope,
		CA:               pool,
		HTTP:             base,
	})
	if err != nil {
		return Built{}, err
	}

	return Built{Provider: prov}, nil
}

// roots — корни плеча; nil — корни системы.
func roots(name string, p llmconfig.Provider, o Options) (*x509.CertPool, error) {
	pem, ok := o.PEM[name]
	if !ok {
		if p.CAFile == "" {
			return nil, nil //nolint:nilnil // пустой пул — корни системы, это не отказ
		}

		var err error
		if pem, err = os.ReadFile(p.CAFile); err != nil {
			return nil, fmt.Errorf("ca_file: %w", err)
		}

		if len(pem) == 0 {
			return nil, errors.New("ca_file: сертификатов PEM нет")
		}
	}

	if len(pem) == 0 {
		return nil, nil //nolint:nilnil // пустой пул — корни системы, это не отказ
	}

	pool := x509.NewCertPool()
	if !pool.AppendCertsFromPEM(pem) {
		return nil, errors.New("ca_file: сертификатов PEM нет")
	}

	return pool, nil
}
