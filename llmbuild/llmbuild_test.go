package llmbuild_test

import (
	"crypto/ecdsa"
	"crypto/elliptic"
	"crypto/rand"
	"crypto/tls"
	"crypto/x509"
	"crypto/x509/pkix"
	"encoding/pem"
	"math/big"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/lllypuk/llm/gigachat"
	"github.com/lllypuk/llm/llmbuild"
	"github.com/lllypuk/llm/llmconfig"
	"github.com/lllypuk/llm/ollama"
	"github.com/lllypuk/llm/salutespeech"
	"github.com/lllypuk/llm/yandex"
)

const allKinds = `{
  "providers": {
    "local": {"kind": "ollama", "endpoint": "http://ollama:11434"},
    "giga": {"kind": "gigachat", "endpoint": "https://giga.invalid", "oauth_endpoint": "https://oauth.invalid",
             "scope": "GIGACHAT_API_PERS", "auth": {"authorization_key": "${KEY}"}},
    "yc": {"kind": "yandex", "endpoint": "https://yc.invalid", "folder": "b1g", "auth": {"api_key": "${KEY}"}},
    "stt": {"kind": "speechkit", "endpoint": "https://stt.invalid", "folder": "b1g", "auth": {"api_key": "${KEY}"}},
    "salute": {"kind": "salutespeech", "endpoint": "https://salute.invalid", "oauth_endpoint": "https://oauth.invalid",
               "scope": "SALUTE_SPEECH_PERS", "auth": {"authorization_key": "${KEY}"}},
    "vision": {"kind": "visionocr", "endpoint": "https://ocr.invalid", "folder": "b1g", "auth": {"api_key": "${KEY}"}}
  },
  "tasks": {
    "ask": {"provider": "local", "model": "gemma"},
    "chat": {"provider": "giga", "model": "GigaChat-2"},
    "draft": {"provider": "yc", "model": "yandexgpt"}
  },
  "speech": {
    "dictation": {"provider": "stt", "model": "general"},
    "call": {"provider": "salute", "model": "general"}
  },
  "ocr": {"scan": {"provider": "vision", "model": "page", "languages": ["ru"]}}
}`

func load(t *testing.T) *llmconfig.Config {
	t.Helper()

	cfg, err := llmconfig.Load([]byte(allKinds), func(string) (string, bool) { return "secret", true })
	if err != nil {
		t.Fatal(err)
	}

	return cfg
}

func TestArmKinds(t *testing.T) {
	cfg := load(t)

	cases := map[string]struct{ field, name string }{
		"local":  {"Provider", ollama.Name},
		"giga":   {"Provider", gigachat.Name},
		"yc":     {"Provider", yandex.Name},
		"stt":    {"Speech", yandex.SpeechName},
		"salute": {"Speech", salutespeech.Name},
		"vision": {"OCR", yandex.OCRName},
	}

	for arm, want := range cases {
		b, err := llmbuild.Arm(arm, cfg.Providers[arm], llmbuild.Options{})
		if err != nil {
			t.Fatalf("%s: %v", arm, err)
		}

		var got []string

		if b.Provider != nil {
			got = append(got, "Provider", b.Provider.Name())
		}

		if b.Speech != nil {
			got = append(got, "Speech", b.Speech.Name())
		}

		if b.OCR != nil {
			got = append(got, "OCR", b.OCR.Name())
		}

		if strings.Join(got, " ") != want.field+" "+want.name {
			t.Errorf("%s: собрано %v, ждали %s %s", arm, got, want.field, want.name)
		}
	}
}

func TestArmUnknownKind(t *testing.T) {
	_, err := llmbuild.Arm("x", llmconfig.Provider{Kind: "openai"}, llmbuild.Options{})
	if err == nil || !strings.Contains(err.Error(), `"openai"`) {
		t.Fatalf("ошибка %v, ждали неизвестный вид", err)
	}
}

func TestArmBadRoots(t *testing.T) {
	cfg := load(t)
	empty := filepath.Join(t.TempDir(), "empty.pem")

	if err := os.WriteFile(empty, nil, 0o600); err != nil {
		t.Fatal(err)
	}

	fromFile := cfg.Providers["yc"]
	fromFile.CAFile = empty

	junk := func(arm string) llmbuild.Options {
		return llmbuild.Options{PEM: map[string][]byte{arm: []byte("junk")}}
	}

	cases := map[string]struct {
		arm string
		p   llmconfig.Provider
		o   llmbuild.Options
	}{
		"битый PEM":         {"yc", cfg.Providers["yc"], junk("yc")},
		"битый PEM у Сбера": {"giga", cfg.Providers["giga"], junk("giga")},
		"пустой файл":       {"yc", fromFile, llmbuild.Options{}},
	}

	for name, c := range cases {
		if _, err := llmbuild.Arm(c.arm, c.p, c.o); err == nil || !strings.Contains(err.Error(), "ca_file") {
			t.Errorf("%s: ошибка %v, ждали ca_file", name, err)
		}
	}
}

func TestArmRootsKeepBaseTLS(t *testing.T) {
	cfg := load(t)
	certPEM := selfSigned(t)

	want := x509.NewCertPool()
	want.AppendCertsFromPEM(certPEM)

	file := filepath.Join(t.TempDir(), "ca.pem")
	if err := os.WriteFile(file, certPEM, 0o600); err != nil {
		t.Fatal(err)
	}

	fromFile := cfg.Providers["local"]
	fromFile.CAFile = file

	for name, o := range map[string]llmbuild.Options{
		"PEM":     {PEM: map[string][]byte{"local": certPEM}},
		"ca_file": {},
	} {
		baseTLS := &tls.Config{
			ServerName:         "ollama.lan",
			MinVersion:         tls.VersionTLS13,
			InsecureSkipVerify: true,
		}
		base := &http.Client{Transport: &http.Transport{TLSClientConfig: baseTLS}, Timeout: time.Minute}

		var calls []string

		o.HTTP = func(name, kind string) *http.Client {
			calls = append(calls, name+"/"+kind)

			return base
		}

		p := fromFile
		if o.PEM != nil {
			p = cfg.Providers["local"]
		}

		b, err := llmbuild.Arm("local", p, o)
		if err != nil {
			t.Fatalf("%s: %v", name, err)
		}

		client := b.Provider.(*ollama.Provider).HTTP
		got := client.Transport.(*http.Transport).TLSClientConfig

		switch {
		case strings.Join(calls, ",") != "local/ollama":
			t.Errorf("%s: основа запрошена как %v", name, calls)
		case !got.RootCAs.Equal(want):
			t.Errorf("%s: корни не из PEM", name)
		case got.InsecureSkipVerify:
			t.Errorf("%s: проверка TLS осталась выключенной", name)
		case got.ServerName != "ollama.lan" || got.MinVersion != tls.VersionTLS13:
			t.Errorf("%s: TLS основы затёрт: %+v", name, got)
		case client.Timeout != time.Minute:
			t.Errorf("%s: срок клиента основы потерян", name)
		case baseTLS.RootCAs != nil || !baseTLS.InsecureSkipVerify:
			t.Errorf("%s: основа изменена на месте", name)
		}
	}
}

func TestArmWithoutRootsKeepsBase(t *testing.T) {
	cfg := load(t)
	base := &http.Client{Transport: roundTripper{}}

	b, err := llmbuild.Arm("local", cfg.Providers["local"], llmbuild.Options{
		PEM:  map[string][]byte{"local": nil},
		HTTP: func(string, string) *http.Client { return base },
	})
	if err != nil {
		t.Fatal(err)
	}

	if b.Provider.(*ollama.Provider).HTTP != base {
		t.Error("без корней основа подменена")
	}

	_, err = llmbuild.Arm("local", cfg.Providers["local"], llmbuild.Options{
		PEM:  map[string][]byte{"local": selfSigned(t)},
		HTTP: func(string, string) *http.Client { return base },
	})
	if err == nil || !strings.Contains(err.Error(), "*http.Transport") {
		t.Errorf("ошибка %v, ждали отказ от чужого транспорта при корнях", err)
	}
}

type roundTripper struct{}

func (roundTripper) RoundTrip(*http.Request) (*http.Response, error) {
	return nil, http.ErrNotSupported
}

func selfSigned(t *testing.T) []byte {
	t.Helper()

	key, err := ecdsa.GenerateKey(elliptic.P256(), rand.Reader)
	if err != nil {
		t.Fatal(err)
	}

	tmpl := &x509.Certificate{
		SerialNumber:          big.NewInt(1),
		Subject:               pkix.Name{CommonName: "llmbuild test"},
		NotBefore:             time.Now().Add(-time.Hour),
		NotAfter:              time.Now().Add(time.Hour),
		IsCA:                  true,
		BasicConstraintsValid: true,
	}

	der, err := x509.CreateCertificate(rand.Reader, tmpl, tmpl, &key.PublicKey, key)
	if err != nil {
		t.Fatal(err)
	}

	return pem.EncodeToMemory(&pem.Block{Type: "CERTIFICATE", Bytes: der})
}
