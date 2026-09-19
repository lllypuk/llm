package salutespeech_test

import (
	"context"
	"crypto/ecdsa"
	"crypto/elliptic"
	"crypto/rand"
	"crypto/tls"
	"crypto/x509"
	"crypto/x509/pkix"
	"encoding/json"
	"errors"
	"io"
	"math/big"
	"net/http"
	"net/http/httptest"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/lllypuk/llm"
	"github.com/lllypuk/llm/gigachat"
	"github.com/lllypuk/llm/salutespeech"
)

const (
	speechKey   = "c3BlZWNoLWtleQ=="
	speechScope = "SALUTE_SPEECH_PERS"
	chatKey     = "Y2hhdC1rZXk="
	chatScope   = "GIGACHAT_API_PERS"
)

// sber — OAuth и оба API на одном сервере: токен выдаётся по scope и помнит ключ, которым получен.
type sber struct {
	oauthCalls atomic.Int32
	oauth      func(w http.ResponseWriter, r *http.Request)
	recognize  func(w http.ResponseWriter, r *http.Request)

	mu     sync.Mutex
	issued map[string]string
	reqs   []*http.Request
	bodies [][]byte
}

func (s *sber) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	switch r.URL.Path {
	case "/oauth":
		n := s.oauthCalls.Add(1)
		if s.oauth != nil {
			s.oauth(w, r)

			return
		}

		_ = r.ParseForm()
		tok := r.PostForm.Get("scope") + "-" + strconv.Itoa(int(n))

		s.mu.Lock()
		if s.issued == nil {
			s.issued = map[string]string{}
		}
		s.issued[tok] = r.Header.Get("Authorization")
		s.mu.Unlock()

		issue(w, tok, time.Now().Add(30*time.Minute))
	case "/rest/v1/speech:recognize":
		raw, _ := io.ReadAll(r.Body)

		s.mu.Lock()
		s.reqs = append(s.reqs, r)
		s.bodies = append(s.bodies, raw)
		s.mu.Unlock()

		if s.recognize != nil {
			s.recognize(w, r)

			return
		}

		w.Header().Set("X-Request-Id", "rq-1")
		_, _ = io.WriteString(w, `{"result":["включи котёл"," на двадцать ",""],"emotions":[],"status":200}`)
	case "/api/chat/completions":
		s.mu.Lock()
		s.reqs = append(s.reqs, r)
		s.mu.Unlock()

		_, _ = io.WriteString(w, `{"choices":[{"message":{"content":"x"},"finish_reason":"stop"}],"model":"GigaChat"}`)
	}
}

func (s *sber) requests() []*http.Request {
	s.mu.Lock()
	defer s.mu.Unlock()

	return append([]*http.Request(nil), s.reqs...)
}

func (s *sber) keyOf(tok string) string {
	s.mu.Lock()
	defer s.mu.Unlock()

	return s.issued[tok]
}

func issue(w http.ResponseWriter, tok string, expires time.Time) {
	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(map[string]any{"access_token": tok, "expires_at": expires.UnixMilli()})
}

func config(url string) salutespeech.Config {
	return salutespeech.Config{
		OAuthEndpoint:    url + "/oauth",
		APIEndpoint:      url + "/rest/v1/",
		AuthorizationKey: speechKey,
		Scope:            speechScope,
	}
}

func provider(t *testing.T, s *sber) (*salutespeech.Provider, string) {
	t.Helper()

	srv := httptest.NewServer(s)
	t.Cleanup(srv.Close)

	p, err := salutespeech.New(config(srv.URL))
	if err != nil {
		t.Fatal(err)
	}

	return p, srv.URL
}

// second — секунда тишины при 16 кГц.
func second() []byte { return make([]byte, 32000) }

func request() llm.SpeechRequest {
	return llm.SpeechRequest{Model: "general", Language: "ru-RU", SampleRate: 16000, PCM: second()}
}

func bearer(r *http.Request) string {
	return strings.TrimPrefix(r.Header.Get("Authorization"), "Bearer ")
}

// TestExactRequestAndResult — адрес, query, Content-Type LPCM, тело байт в байт; фразы склеены, пустые выброшены.
func TestExactRequestAndResult(t *testing.T) {
	t.Parallel()

	s := &sber{}
	p, _ := provider(t, s)

	pcm := second()
	pcm[0], pcm[31999] = 7, 9

	req := request()
	req.PCM = pcm

	res, err := p.Transcribe(context.Background(), req)
	if err != nil {
		t.Fatal(err)
	}

	want := llm.Transcript{Text: "включи котёл на двадцать", Model: "general", RequestID: "rq-1", AudioMillis: 1000}
	if res != want {
		t.Errorf("результат %+v", res)
	}

	reqs := s.requests()
	if len(reqs) != 1 {
		t.Fatalf("запросов %d", len(reqs))
	}

	r := reqs[0]
	if r.Method != http.MethodPost || r.URL.RawQuery != "language=ru-RU&model=general" {
		t.Errorf("запрос %s %s", r.Method, r.URL)
	}

	if ct := r.Header.Get("Content-Type"); ct != "audio/x-pcm;bit=16;rate=16000" {
		t.Errorf("Content-Type %q", ct)
	}

	if bearer(r) != speechScope+"-1" {
		t.Errorf("токен %q", bearer(r))
	}

	if string(s.bodies[0]) != string(pcm) {
		t.Error("тело не совпало с PCM")
	}
}

// TestEmptyResult — речи не расслышали: успех с пустым текстом.
func TestEmptyResult(t *testing.T) {
	t.Parallel()

	for _, body := range []string{`{"result":[]}`, `{"result":[" "]}`, `{}`} {
		s := &sber{recognize: func(w http.ResponseWriter, _ *http.Request) { _, _ = io.WriteString(w, body) }}
		p, _ := provider(t, s)

		res, err := p.Transcribe(context.Background(), request())
		if err != nil || res.Text != "" || res.AudioMillis != 1000 {
			t.Errorf("%s: %+v, %v", body, res, err)
		}
	}
}

// TestTokensDoNotCross — GigaChat и SaluteSpeech на общем механизме и общем OAuth получают каждый свой токен
// своим ключом и не отдают его друг другу.
func TestTokensDoNotCross(t *testing.T) {
	t.Parallel()

	s := &sber{}
	p, url := provider(t, s)

	chat, err := gigachat.New(gigachat.Config{
		OAuthEndpoint:    url + "/oauth",
		APIEndpoint:      url + "/api/",
		AuthorizationKey: chatKey,
		Scope:            chatScope,
	})
	if err != nil {
		t.Fatal(err)
	}

	for range 2 {
		if _, err = p.Transcribe(context.Background(), request()); err != nil {
			t.Fatal(err)
		}

		if _, err = chat.Complete(context.Background(), llm.Request{
			Model:    "GigaChat",
			Messages: []llm.Message{{Role: llm.RoleUser, Text: "привет"}},
		}); err != nil {
			t.Fatal(err)
		}
	}

	if n := s.oauthCalls.Load(); n != 2 {
		t.Errorf("запросов OAuth %d, ожидалось по одному на плечо", n)
	}

	for _, r := range s.requests() {
		tok := bearer(r)

		wantScope, wantKey := speechScope, speechKey
		if strings.HasPrefix(r.URL.Path, "/api/") {
			wantScope, wantKey = chatScope, chatKey
		}

		if !strings.HasPrefix(tok, wantScope+"-") || s.keyOf(tok) != "Basic "+wantKey {
			t.Errorf("%s: токен %q выдан ключом %q", r.URL.Path, tok, s.keyOf(tok))
		}
	}
}

// TestConcurrentRefreshMerged — одновременные распознавания на пустом кеше дают один запрос OAuth.
func TestConcurrentRefreshMerged(t *testing.T) {
	t.Parallel()

	release := make(chan struct{})

	s := &sber{}
	s.oauth = func(w http.ResponseWriter, _ *http.Request) {
		<-release
		issue(w, "token-1", time.Now().Add(30*time.Minute))
	}

	p, _ := provider(t, s)

	const callers = 8

	var wg sync.WaitGroup
	for range callers {
		wg.Go(func() {
			if _, err := p.Transcribe(context.Background(), request()); err != nil {
				t.Error(err)
			}
		})
	}

	time.Sleep(50 * time.Millisecond)
	close(release)
	wg.Wait()

	if n := s.oauthCalls.Load(); n != 1 {
		t.Errorf("запросов OAuth %d, ожидался один", n)
	}
}

// TestUnauthorizedRetriedOnce — 401 гасит токен и повторяется с новым; второй 401 — отказ фазы auth без цикла.
func TestUnauthorizedRetriedOnce(t *testing.T) {
	t.Parallel()

	var calls atomic.Int32

	s := &sber{}
	s.recognize = func(w http.ResponseWriter, _ *http.Request) {
		if calls.Add(1) == 1 {
			w.WriteHeader(http.StatusUnauthorized)

			return
		}

		_, _ = io.WriteString(w, `{"result":["да"]}`)
	}

	p, _ := provider(t, s)

	res, err := p.Transcribe(context.Background(), request())
	if err != nil || res.Text != "да" {
		t.Fatalf("после одного 401: %+v, %v", res, err)
	}

	if n := s.oauthCalls.Load(); n != 2 {
		t.Errorf("запросов OAuth %d", n)
	}

	denied := &sber{recognize: func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusUnauthorized)
		_, _ = io.WriteString(w, `{"status":401,"message":"bad credentials"}`)
	}}
	p, _ = provider(t, denied)

	_, err = p.Transcribe(context.Background(), request())

	var status *llm.StatusError
	if !errors.As(err, &status) || status.Status != http.StatusUnauthorized || status.Phase != llm.PhaseAuth ||
		status.Message != "bad credentials" {
		t.Errorf("второй 401: %v", err)
	}

	if n := len(denied.requests()); n != 2 {
		t.Errorf("распознаваний %d, ожидалось два", n)
	}

	if strings.Contains(err.Error(), speechKey) {
		t.Error("ключ в ошибке")
	}
}

// TestStatusErrors — 4xx и 5xx распознавания уходят [*llm.StatusError] без фазы auth, с request id и Retry-After.
func TestStatusErrors(t *testing.T) {
	t.Parallel()

	for _, code := range []int{http.StatusBadRequest, http.StatusTooManyRequests, http.StatusServiceUnavailable} {
		s := &sber{recognize: func(w http.ResponseWriter, _ *http.Request) {
			w.Header().Set("X-Request-Id", "rq-err")
			w.Header().Set("Retry-After", "3")
			w.WriteHeader(code)
			_, _ = io.WriteString(w, `{"status":`+strconv.Itoa(code)+`,"message":"отказ"}`)
		}}
		p, _ := provider(t, s)

		_, err := p.Transcribe(context.Background(), request())

		var status *llm.StatusError
		if !errors.As(err, &status) || status.Status != code || status.Phase != "" || status.Message != "отказ" ||
			status.RequestID != "rq-err" || status.RetryAfter != 3*time.Second {
			t.Errorf("%d: %v", code, err)
		}
	}
}

// TestOAuthFailure — отказ OAuth — фаза auth, распознавание не зовётся.
func TestOAuthFailure(t *testing.T) {
	t.Parallel()

	s := &sber{oauth: func(w http.ResponseWriter, _ *http.Request) { w.WriteHeader(http.StatusBadRequest) }}
	p, _ := provider(t, s)

	_, err := p.Transcribe(context.Background(), request())

	var status *llm.StatusError
	if !errors.As(err, &status) || status.Phase != llm.PhaseAuth || len(s.requests()) != 0 {
		t.Errorf("отказ OAuth: %v", err)
	}
}

// TestRejectedBeforeNetwork — запись длиннее минуты и пустая — отказ до OAuth и сети.
func TestRejectedBeforeNetwork(t *testing.T) {
	t.Parallel()

	s := &sber{}
	p, _ := provider(t, s)

	long := request()
	long.PCM = make([]byte, 32000*61)

	empty := request()
	empty.PCM = nil

	for name, req := range map[string]llm.SpeechRequest{"длинная": long, "пустая": empty} {
		_, err := p.Transcribe(context.Background(), req)

		var reqErr *llm.RequestError
		if !errors.As(err, &reqErr) {
			t.Errorf("%s: %v", name, err)
		}
	}

	if s.oauthCalls.Load() != 0 || len(s.requests()) != 0 {
		t.Error("запрос ушёл в сеть")
	}
}

// TestCancelled — отмена контекста доходит до вызывающего.
func TestCancelled(t *testing.T) {
	t.Parallel()

	s := &sber{recognize: func(_ http.ResponseWriter, r *http.Request) { <-r.Context().Done() }}
	p, _ := provider(t, s)

	ctx, cancel := context.WithTimeout(context.Background(), 50*time.Millisecond)
	defer cancel()

	if _, err := p.Transcribe(ctx, request()); !errors.Is(err, context.DeadlineExceeded) {
		t.Errorf("отмена: %v", err)
	}
}

// TestCapabilities — известные модели с пределом в минуту, неизвестная не подтверждается.
func TestCapabilities(t *testing.T) {
	t.Parallel()

	p, _ := provider(t, &sber{})

	if c, ok := p.SpeechCapabilities("general"); !ok || c.MaxAudio != time.Minute {
		t.Errorf("general: %+v %v", c, ok)
	}

	if _, ok := p.SpeechCapabilities("GigaChat"); ok {
		t.Error("чужая модель подтверждена")
	}
}

// TestConfig — пустое обязательное поле и CA при чужом транспорте — отказ сборки без ключа в тексте.
func TestConfig(t *testing.T) {
	t.Parallel()

	for name, mutate := range map[string]func(*salutespeech.Config){
		"oauth": func(c *salutespeech.Config) { c.OAuthEndpoint = "" },
		"api":   func(c *salutespeech.Config) { c.APIEndpoint = "" },
		"key":   func(c *salutespeech.Config) { c.AuthorizationKey = "" },
		"scope": func(c *salutespeech.Config) { c.Scope = "" },
		"transport": func(c *salutespeech.Config) {
			c.CA = x509.NewCertPool()
			c.HTTP = &http.Client{Transport: http.NewFileTransport(http.Dir("."))}
		},
	} {
		cfg := config("http://x")
		mutate(&cfg)

		if _, err := salutespeech.New(cfg); err == nil || strings.Contains(err.Error(), speechKey) {
			t.Errorf("%s: %v", name, err)
		}
	}
}

// TestCA — свой корень пропускает, чужой отбивает на OAuth и тогда, когда основа выключила проверку.
func TestCA(t *testing.T) {
	t.Parallel()

	srv := httptest.NewTLSServer(&sber{})
	t.Cleanup(srv.Close)

	own := x509.NewCertPool()
	own.AddCert(srv.Certificate())

	cfg := config(srv.URL)
	cfg.CA = own

	p, err := salutespeech.New(cfg)
	if err != nil {
		t.Fatal(err)
	}

	if _, err = p.Transcribe(context.Background(), request()); err != nil {
		t.Fatalf("свой корень: %v", err)
	}

	cfg.CA = foreignPool(t)
	cfg.HTTP = &http.Client{Transport: &http.Transport{TLSClientConfig: &tls.Config{InsecureSkipVerify: true}}}

	if p, err = salutespeech.New(cfg); err != nil {
		t.Fatal(err)
	}

	_, err = p.Transcribe(context.Background(), request())

	var (
		unknown x509.UnknownAuthorityError
		phased  *llm.PhaseError
	)
	if !errors.As(err, &unknown) || !errors.As(err, &phased) || phased.Phase != llm.PhaseAuth {
		t.Errorf("чужой корень: %v", err)
	}
}

// foreignPool — пул с самоподписанным корнем, которым сервер не подписан.
func foreignPool(t *testing.T) *x509.CertPool {
	t.Helper()

	priv, err := ecdsa.GenerateKey(elliptic.P256(), rand.Reader)
	if err != nil {
		t.Fatal(err)
	}

	tmpl := &x509.Certificate{
		SerialNumber:          big.NewInt(1),
		Subject:               pkix.Name{CommonName: "foreign root"},
		NotBefore:             time.Now().Add(-time.Hour),
		NotAfter:              time.Now().Add(time.Hour),
		IsCA:                  true,
		BasicConstraintsValid: true,
		KeyUsage:              x509.KeyUsageCertSign,
	}

	der, err := x509.CreateCertificate(rand.Reader, tmpl, tmpl, &priv.PublicKey, priv)
	if err != nil {
		t.Fatal(err)
	}

	cert, err := x509.ParseCertificate(der)
	if err != nil {
		t.Fatal(err)
	}

	pool := x509.NewCertPool()
	pool.AddCert(cert)

	return pool
}
